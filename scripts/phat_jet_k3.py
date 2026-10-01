"""
PHAT-JeT ported to Keras 3 (backend-agnostic), in a hardware-synthesizable form.

WHY THIS FILE EXISTS (for humans and other agents)
--------------------------------------------------
The public PHAT-JeT repo (github.com/aaronw5/PHAT-JeT) is TF/Keras-2 with dynamic
shapes, tf.scatter_nd / tf.gather_nd and per-jet dynamic grid extents. None of that
can go through the HGQ2 -> da4ml FPGA flow, which requires:
  * static shapes everywhere,
  * only ops in the da4ml trace registry (Einsum, Conv2D, Dense, Softmax, LUTs,
    GetItem, Sum, Repeat, Reshape, Concatenate, scalar arithmetic, ...),
  * no dynamic indexing (scatter/gather must be re-expressed),
  * ops applied FUNCTIONALLY on Keras tensors (custom Layer subclasses are
    invisible to the tracer).

Deliberate, documented deviations from the paper's float model (all shared
between the float reference and the QAT model so weights transfer 1:1):

1. GMP grid FIXED to [-0.4, 0.4]^2, 8x8 bins (delta=0.1 == paper's delta=0.1).
   Valid because inputs are jet-relative (etarel, phirel): 99.9% of constituents
   fall within +/-0.4 (measured on train). Edge bins clamp outliers.
   scatter_nd/gather_nd -> per-bin indicator LUTs, outer product, einsum
   scatter-add, depthwise conv, einsum gather-back. Mathematically identical.
2. Depthwise kernel 3x3 on the 8x8 grid (paper: 8x8 kernel on delta=0.05 grid;
   comparable physical receptive field ~0.3 vs ~0.4).
3. Patch size P=8 -> 16 patches of 128 particles, no padding (paper: P=10 on 150).
   Paper's sweep shows low sensitivity to P in this range.
4. No LayerNorm in the hardware variant (use_ln=False): runtime division/sqrt is
   hostile to a fully unrolled II=1 design; JEDI-Linear's FPGA models likewise
   have no runtime normalization. use_ln=True restores the paper's float model.
5. Attention scale 1/sqrt(d_head) is reparameterized into W_q (a Dense
   reparameterization, not an approximation). Model outputs LOGITS; train with
   from_logits=True. Argmax on-chip needs no softmax.
"""

import keras
from keras import layers, ops


def _dense(units, quantized, activation=None, name=None):
    if quantized:
        from hgq.layers import QDense
        return QDense(units, activation=activation, name=name)
    return layers.Dense(units, activation=activation, name=name)


def _einsum(eq, xs, quantized, name=None):
    if quantized:
        from hgq.layers import QEinsum
        return QEinsum(eq, name=name)(xs)
    return ops.einsum(eq, *xs)


def _softmax(x, quantized, name=None, exp_i_max=5):
    """Softmax with a BOUNDED exp-table input range.

    Why exp_i_max: QSoftmax's exp table is addressed by the quantized value of
    (max - score). HGQ sizes that quantizer's integer bits from the OBSERVED
    MAXIMUM, and attention scores have long tails: on local attention the p99.9
    of (max - score) is 14 but the max is 189.5, so HGQ allocated i=9 -- a
    1024-entry table -- to represent outliers in the 0.1% tail.

    That is pure waste, because the table's own OUTPUT is only 3 bits: stable
    softmax computes exp(-d) for d >= 0, and exp(-d) < 2^-3 for every d > 2.1,
    so all those extra entries encode values that quantize to the same near-zero
    output. Measured on the epoch-141 checkpoint: lookup ops were 29% of total
    logic (389k of 1.33M LUT) at ~192 LUT/op, with 123 of 181 tables >= 1024
    entries.

    Capping i at 5 (range +/-32, still >2x the p99.9) cuts lookup cost 389k ->
    94k, a 4.1x reduction, for -24.6% TOTAL LUT at +0.05% accuracy -- the
    clipped values are all deep in the exp tail where the output is zero anyway.
    The cliff is just below: i<=4 costs -3.1% accuracy, i<=3 costs -16.2%, so 5
    is the last safe setting rather than an arbitrary one. Raise it only with a
    measurement; do not lower it.

    IMPORTANT -- the MinMax(ic=...) constraint set here does NOT work by itself
    on hgq 0.1.9. QSoftmax's exp table uses FixedPointQuantizerKIF, and in that
    class's WRAP training branch the constraint is applied to the wrong variable
    (fixed_point_quantizer.py, FixedPointQuantizerKIF.call):

        new_i = stop_gradient(maximum(self._i - decay, _new_i))  # from UNconstrained
        if self._i.constraint is not None:
            _new_i = self._i.constraint(_new_i)   # constrained value -> _new_i
        self._i.assign(new_i)                     # ...but new_i is assigned

    The constrained result lands in `_new_i`, which is then discarded. The sister
    class FixedPointQuantizerKBI has the same block in the correct order, and the
    KIF *tracing* branch is also correct -- only the KIF training path is wrong,
    which is why this looks fine on inspection. Verified empirically: with
    MinMax(-16, 5) attached, i still grew 8.0 -> 10.99 over 8 gradient steps.

    Because `i` is non-trainable under WRAP (i_trainable = overflow_mode !=
    'WRAP'), Keras's own constraint machinery never runs either. So the cap is
    enforced externally by ClampSoftmaxExpBits in train_qat.py, which reasserts
    it after each epoch. Keep BOTH: the config is correct-by-intent and will
    start working if upstream fixes the ordering; the callback is what actually
    binds today.
    """
    if quantized:
        from hgq.config import QuantizerConfig
        from hgq.constraints import MinMax
        from hgq.layers import QSoftmax
        exp_iq = QuantizerConfig('default', 'datalane')
        exp_iq.config['ic'] = MinMax(-16, exp_i_max)
        return QSoftmax(axis=-1, name=name, exp_iq_conf=exp_iq)(x)
    return layers.Softmax(axis=-1, name=name)(x)


def _bin_indicator(lo, hi, first, last):
    if first:
        return lambda x: ops.cast(x < hi, x.dtype)
    if last:
        return lambda x: ops.cast(x >= lo, x.dtype)
    return lambda x: ops.cast((x >= lo) & (x < hi), x.dtype)


def onehot_bins(coord, edges, quantized, tag):
    """[B,N] -> [B,N,len(edges)-1] one-hot bin indicators with edge clamping.
    `edges`: ascending bin-edge sequence in the units of `coord` (non-uniform OK;
    hardware cost is unchanged — each bin is one threshold LUT either way)."""
    edges = list(edges)
    n_bins = len(edges) - 1
    n = coord.shape[1]
    cols = []
    for g in range(n_bins):
        f = _bin_indicator(edges[g], edges[g + 1], g == 0, g == n_bins - 1)
        if quantized:
            from hgq.layers import QUnaryFunctionLUT
            c = QUnaryFunctionLUT(f, name=f"oh_{tag}_{g}")(coord)
        else:
            c = layers.Lambda(f, name=f"oh_{tag}_{g}")(coord)
        cols.append(ops.reshape(c, (-1, n, 1)))
    return layers.Concatenate(axis=-1, name=f"oh_{tag}")(cols)


def gmp_block(x, eta, phi, channels, quantized, eta_edges, phi_edges, name="gmp"):
    """Geometric Message Passing (static form): scatter-add onto (eta,phi) grid,
    3x3 depthwise conv, gather back. Returns message [B,N,C] (added residually).
    `eta_edges`/`phi_edges` are ascending bin-edge lists in the units of the
    network INPUT coordinates: if inputs are robust-scaled (x-med)/iqr, pass
    raw-unit edges divided by that feature's iqr. Non-uniform edges supported
    (physics-informed fine-core grids)."""
    eta_oh = onehot_bins(eta, eta_edges, quantized, f"{name}_eta")
    phi_oh = onehot_bins(phi, phi_edges, quantized, f"{name}_phi")
    cell = _einsum("bne,bnp->bnep", [eta_oh, phi_oh], quantized, name=f"{name}_cell")
    grid = _einsum("bnep,bnc->bepc", [cell, x], quantized, name=f"{name}_scatter")
    if quantized:
        from hgq.layers import QConv2D
        conv = QConv2D(channels, 3, padding="same", groups=channels, name=f"{name}_dwconv")(grid)
    else:
        conv = layers.Conv2D(channels, 3, padding="same", groups=channels, name=f"{name}_dwconv")(grid)
    msg = _einsum("bnep,bepc->bnc", [cell, conv], quantized, name=f"{name}_gather")
    msg = _dense(channels, quantized, name=f"{name}_pointwise")(msg)
    return msg


def patch_mha(x4, d_model, num_heads, quantized, name):
    """Exact MHA within each patch. x4: [B,NP,P,D] -> [B,NP,P,D].
    No 1/sqrt(dk) op: absorbed in W_q (see module docstring, deviation 5)."""
    NP, P = x4.shape[1], x4.shape[2]
    dh = d_model // num_heads
    q = _dense(d_model, quantized, name=f"{name}_wq")(x4)
    k = _dense(d_model, quantized, name=f"{name}_wk")(x4)
    v = _dense(d_model, quantized, name=f"{name}_wv")(x4)
    q = ops.reshape(q, (-1, NP, P, num_heads, dh))
    k = ops.reshape(k, (-1, NP, P, num_heads, dh))
    v = ops.reshape(v, (-1, NP, P, num_heads, dh))
    scores = _einsum("bnphd,bnqhd->bnhpq", [q, k], quantized, name=f"{name}_qk")
    w = _softmax(scores, quantized, name=f"{name}_softmax")
    o = _einsum("bnhpq,bnqhd->bnphd", [w, v], quantized, name=f"{name}_av")
    o = ops.reshape(o, (-1, NP, P, d_model))
    return _dense(d_model, quantized, name=f"{name}_wo")(o)


def token_mha(x, d_model, num_heads, quantized, name):
    """Exact MHA over patch tokens. x: [B,NP,D] -> [B,NP,D]."""
    T = x.shape[1]
    dh = d_model // num_heads
    q = _dense(d_model, quantized, name=f"{name}_wq")(x)
    k = _dense(d_model, quantized, name=f"{name}_wk")(x)
    v = _dense(d_model, quantized, name=f"{name}_wv")(x)
    q = ops.reshape(q, (-1, T, num_heads, dh))
    k = ops.reshape(k, (-1, T, num_heads, dh))
    v = ops.reshape(v, (-1, T, num_heads, dh))
    scores = _einsum("bthd,bshd->bhts", [q, k], quantized, name=f"{name}_qk")
    w = _softmax(scores, quantized, name=f"{name}_softmax")
    o = _einsum("bhts,bshd->bthd", [w, v], quantized, name=f"{name}_av")
    o = ops.reshape(o, (-1, T, d_model))
    return _dense(d_model, quantized, name=f"{name}_wo")(o)


def build_phat_jet_k3(
    num_particles=128,
    num_feats=3,
    d_model=16,
    num_heads=4,
    patch_size=8,
    n_classes=5,
    use_gmp=True,
    use_ln=False,
    gmp_bins=8,
    quantized=False,
    aggregation="mean",
    gmp_bounds=3.7,  # ±0.4 raw / iqr 0.108 (robust-scaled coords); use 0.4 for raw
    gmp_edges=None,  # explicit ascending edge list (input units); overrides
                     # gmp_bins/gmp_bounds. Same list used for eta and phi.
    # ---- depth ablation flags (all default True = unchanged behaviour) ----
    # JEDI-linear traces to 10-12 pipeline stages because it is purely linear.
    # Our serial chain embed->gmp->local_attn->patch_attn->ffn->head is 40-63.
    # These let us price each block in stages and search the shallow band.
    use_local_attn=True,
    use_patch_attn=True,
    use_ffn=True,
    # Run local_attn and patch_attn as PARALLEL branches off the same x instead
    # of chaining them. Both are residual (x = x + f(x)), so
    #     x + local(x) + patch(x + local(x))   [serial]
    # becomes
    #     x + local(x) + patch(x)              [parallel]
    # which is a different function but the same parameter count and receptive
    # field; critically the two branches then pipeline CONCURRENTLY, so depth
    # is max(local, patch) rather than local + patch.
    parallel_attn=False,
):
    """Trigger-scale PHAT-JeT (1 PHAT block). Outputs LOGITS."""
    NP = num_particles // patch_size
    assert NP * patch_size == num_particles

    feats = keras.Input((num_particles, num_feats), name="features")
    eta = feats[..., 1]   # GetItem — traceable
    phi = feats[..., 2]

    x = _dense(d_model, quantized, activation="relu", name="embed")(feats)

    if use_gmp:
        if gmp_edges is None:
            import numpy as _np
            gmp_edges = _np.linspace(-gmp_bounds, gmp_bounds, gmp_bins + 1).tolist()
        msg = gmp_block(x, eta, phi, d_model, quantized,
                        eta_edges=gmp_edges, phi_edges=gmp_edges)
        x = x + msg

    def maybe_ln(t, name):
        return layers.LayerNormalization(epsilon=1e-6, name=name)(t) if use_ln else t

    # local patched attention (residual)
    x_pre_attn = x          # branch point for parallel_attn
    if use_local_attn:
        h = maybe_ln(x, "ln1")
        h4 = ops.reshape(h, (-1, NP, patch_size, d_model))
        a = patch_mha(h4, d_model, num_heads, quantized, name="local_attn")
        a = ops.reshape(a, (-1, num_particles, d_model))
        x = x + a

    # patch-message broadcast (residual): mean tokenizer -> token MHA -> proj -> repeat
    if use_patch_attn:
      # parallel_attn: feed this branch the PRE-local-attention tensor so the
      # two attention blocks trace as concurrent pipelines.
      h = maybe_ln(x_pre_attn if parallel_attn else x, "ln_msg")
      h4 = ops.reshape(h, (-1, NP, patch_size, d_model))
      # exact mean tokenizer; 1/patch_size is a power-of-two shift in fixed point.
      if quantized:
        from hgq.layers import QSum
        ptok = QSum(axes=2, scale=1.0 / patch_size, name="patch_tokens")(h4)
      else:
        ptok = ops.sum(h4, axis=2) * (1.0 / patch_size)
      ptok = token_mha(ptok, d_model, num_heads, quantized, name="patch_attn")
      ptok = _dense(d_model, quantized, name="patch_msg_proj")(ptok)
      msg = ops.repeat(ptok, patch_size, axis=1)
      x = x + msg

    # FFN (residual)
    if use_ffn:
        h = maybe_ln(x, "ln2")
        h = _dense(4 * d_model, quantized, activation="relu", name="ffn1")(h)
        h = _dense(d_model, quantized, name="ffn2")(h)
        x = x + h

    # aggregate + head. Default MEAN (matches Aaron's run config `aggregation=mean`;
    # 1/128 is a power-of-two shift -> free in fixed point). "max" kept for
    # loading run-1 checkpoints trained before the switch.
    if aggregation == "mean":
        if quantized:
            from hgq.layers import QSum
            pooled = QSum(axes=1, scale=1.0 / num_particles, name="agg_mean")(x)
        else:
            pooled = layers.GlobalAveragePooling1D(name="agg_mean")(x)
    else:
        if quantized:
            from hgq.layers import QGlobalMaxPooling1D
            pooled = QGlobalMaxPooling1D(name="agg_max")(x)
        else:
            pooled = layers.GlobalMaxPooling1D(name="agg_max")(x)
    hd = _dense(d_model, quantized, activation="relu", name="head1")(pooled)
    logits = _dense(n_classes, quantized, name="head_out")(hd)

    return keras.Model(feats, logits, name="phat_jet_k3" + ("_q" if quantized else ""))
