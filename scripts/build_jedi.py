"""Rebuild JEDI-linear's gnn model with the LOCALLY installed hgq, then load
their official weights by position.

Their published .keras checkpoints were saved with an older HGQ2 whose
_analyze_einsum_string returned a 3-tuple; hgq 0.1.9 returns 4. Downgrading
hgq would break our own pipeline, so instead we reconstruct the architecture
verbatim from refs/JEDI-linear-master/src/model.py (get_gnn) under the same
QuantizerConfigScope stack from get_model(), and transfer weights.

Config values come from configs/sweep-n128-f3.yaml:
    n_constituents=128, pt_eta_phi=true, model_class=gnn
    init_bw_a=7, init_bw_k=7, k_bw_l1_reg=1e-8, a_bw_l1_reg=1e-8
"""
from math import log2
import keras

# ---- compat shim: hgq 0.1.9 vs keras 3.15 ----
# hgq's QEinsumDenseBatchnorm._compute_fused_einsum_specs unpacks
# keras' _analyze_einsum_string into 3 values; keras >=3.13 returns 5
# (kernel_shape, bias_shape, output_shape, +2 axis lists). The first 3
# are unchanged, so we wrap it to return exactly those 3 when called
# from hgq. Verified: keras 3.15 returns [3,64],[64],[None,128,64],[0],[1].
import hgq.layers.einsum_dense_batchnorm as _edbn
if not getattr(_edbn, "_ANALYZE_PATCHED", False):
    _orig_analyze = _edbn._analyze_einsum_string
    def _analyze3(*a, **k):
        r = _orig_analyze(*a, **k)
        return r[:3] if len(r) > 3 else r
    _edbn._analyze_einsum_string = _analyze3
    _edbn._ANALYZE_PATCHED = True
from hgq.config import LayerConfigScope, QuantizerConfig, QuantizerConfigScope
from hgq.constraints import MinMax
from hgq.layers import QAdd, QEinsumDense, QEinsumDenseBatchnorm, QSum
from hgq.regularizers import MonoL1


def get_gnn(N=128, n=3, uq1=False):
    """Verbatim from JEDI-linear src/model.py::get_gnn."""
    heterogeneous_axis = None if not uq1 else (-1,)
    with (
        QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis),
    ):
        inp = keras.layers.Input((N, n))
        pool_scale = 2.0 ** -round(log2(N))
        x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, 64), bias_axes='C', activation='relu')(inp)
        s = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, 64), bias_axes='C', activation='relu')(x)
        d = QEinsumDenseBatchnorm('bnc,cC->bnC', (1, 64), bias_axes='C', activation='relu')(
            QSum(axes=1, scale=pool_scale, keepdims=True)(x))
        x = QAdd()([s, d])
        x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, 64), bias_axes='C', activation='relu')(x)
        x = QSum(axes=1, scale=1 / 16, keepdims=False)(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 64, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 32, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C')(x)
    return keras.Model(inputs=inp, outputs=out)


def build_jedi(N=128, n=3, uq1=False, init_bw_a=7, init_bw_k=7,
               k_bw_l1_reg=1e-8, a_bw_l1_reg=1e-8):
    """Verbatim from JEDI-linear src/model.py::get_model scope stack."""
    scope0 = QuantizerConfigScope(default_q_type='kbi', b0=init_bw_k,
                                  overflow_mode='wrap', i0=0,
                                  fr=MonoL1(k_bw_l1_reg), ir=MonoL1(k_bw_l1_reg))
    scope1 = QuantizerConfigScope(default_q_type='kif', place='datalane',
                                  overflow_mode='wrap', f0=init_bw_a,
                                  fr=MonoL1(a_bw_l1_reg), ic=MinMax(0, 12))
    scope2 = LayerConfigScope(beta0=0)
    with scope0, scope1, scope2:
        return get_gnn(N=N, n=n, uq1=uq1)


def load_official_weights(model, ckpt_path):
    """Transfer weights from their .keras (a zip containing model.weights.h5)."""
    import zipfile, tempfile, os
    with zipfile.ZipFile(ckpt_path) as z:
        names = [x for x in z.namelist() if x.endswith('.h5')]
        assert names, f'no .h5 inside {ckpt_path}: {z.namelist()}'
        with tempfile.TemporaryDirectory() as td:
            z.extract(names[0], td)
            model.load_weights(os.path.join(td, names[0]))
    return model
