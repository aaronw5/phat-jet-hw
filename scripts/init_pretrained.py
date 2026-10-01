"""
Initialize the HGQ2 hardware model from Aaron's pretrained float checkpoint
(model.weights.h5, run config: hls4ml dataset, pt-sorted, 150p x 3f,
enc patch_size=150 i.e. full attention, d_model=16, 4 heads, ffn 64,
aggregation=mean, patch_tokenizer=mean, cpe_k=8, grid_size=0.2, GELU FFN).

WHAT TRANSFERS (shape-matched, 1:1):
  embed(3x16), local MHA q/k/v (16x4x4 -> 16x16) + out (4x4x16 -> 16x16),
  patch-token attn wq/wk/wv/wo (16x16), patch_msg proj (16x16),
  ffn (16x64, 64x16), gmp pointwise (16x16), head1 (16x16), head_out (16x5).
  Attention scale: our port folds 1/sqrt(d_head)=0.5 into wq -> wq scaled by 0.5.

WHAT DOESN'T (documented gaps, QAT must adapt):
  - LayerNorms (hardware model has none; their statistics are lost)
  - cpe conv2d 8x8 kernel vs our 3x3 depthwise -> center-crop 3x3 slice as init
  - checkpoint trained @150 particles w/ full attention; we run 128p, 16 patches of 8
    (projection weights are patch-size agnostic, so they transfer cleanly)
  - input scaling: theirs robust-scaled, ours pT/64 -> embed sees shifted distribution
  - GELU vs ReLU in FFN
"""

import os

import h5py
import numpy as np

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.weights.h5")
P = "_layer_checkpoint_dependencies/"


def _get(f, path):
    return np.array(f[P + path])


def load_pretrained_into(model, ckpt_path=CKPT, verbose=True):
    f = h5py.File(ckpt_path, "r")
    b = "p_tv3_block/"
    scale = 0.5  # 1/sqrt(d_head=4), folded into wq in our port

    def mha_qkv(name):  # (16,4,4)->(16,16), bias (4,4)->(16,)
        k = _get(f, f"{b}attn/mha/_{name}_dense/vars/0").reshape(16, 16)
        bi = _get(f, f"{b}attn/mha/_{name}_dense/vars/1").reshape(16)
        return k, bi

    wq, bq = mha_qkv("query")
    wk, bk = mha_qkv("key")
    wv, bv = mha_qkv("value")
    wo = _get(f, f"{b}attn/mha/_output_dense/vars/0").reshape(16, 16)
    bo = _get(f, f"{b}attn/mha/_output_dense/vars/1")

    conv8 = _get(f, f"{b}cpe/conv2d/vars/0")            # (8,8,1,16)
    conv3 = conv8[3:6, 3:6]                             # center 3x3 crop

    mapping = {
        "embed": (_get(f, "dense/vars/0"), _get(f, "dense/vars/1")),
        "local_attn_wq": (wq * scale, bq * scale),
        "local_attn_wk": (wk, bk),
        "local_attn_wv": (wv, bv),
        "local_attn_wo": (wo, bo),
        "patch_attn_wq": (_get(f, f"{b}patch_msg/patch_attn/wq/vars/0") * scale,
                          _get(f, f"{b}patch_msg/patch_attn/wq/vars/1") * scale),
        "patch_attn_wk": (_get(f, f"{b}patch_msg/patch_attn/wk/vars/0"),
                          _get(f, f"{b}patch_msg/patch_attn/wk/vars/1")),
        "patch_attn_wv": (_get(f, f"{b}patch_msg/patch_attn/wv/vars/0"),
                          _get(f, f"{b}patch_msg/patch_attn/wv/vars/1")),
        "patch_attn_wo": (_get(f, f"{b}patch_msg/patch_attn/wo/vars/0"),
                          _get(f, f"{b}patch_msg/patch_attn/wo/vars/1")),
        "patch_msg_proj": (_get(f, f"{b}patch_msg/proj/vars/0"),
                           _get(f, f"{b}patch_msg/proj/vars/1")),
        "ffn1": (_get(f, f"{b}ffn/{P}dense/vars/0"), _get(f, f"{b}ffn/{P}dense/vars/1")),
        "ffn2": (_get(f, f"{b}ffn/{P}dense_2/vars/0"), _get(f, f"{b}ffn/{P}dense_2/vars/1")),
        "gmp_dwconv": (conv3, _get(f, f"{b}cpe/conv2d/vars/1")),
        "gmp_pointwise": (_get(f, f"{b}cpe/pointwise/vars/0"), _get(f, f"{b}cpe/pointwise/vars/1")),
        "head1": (_get(f, "dense_2/vars/0"), _get(f, "dense_2/vars/1")),
        "head_out": (_get(f, "dense_4/vars/0"), _get(f, "dense_4/vars/1")),
    }

    n_ok = 0
    for lname, (kern, bias) in mapping.items():
        layer = model.get_layer(lname)
        ws = layer.get_weights()
        # HGQ2 layers: ws[0]=kernel, ws[1]=bias, rest are quantizer params — keep those.
        if ws[0].shape != kern.shape or ws[1].shape != bias.shape:
            if verbose:
                print(f"SKIP {lname}: ckpt {kern.shape}/{bias.shape} vs model {ws[0].shape}/{ws[1].shape}")
            continue
        layer.set_weights([kern.astype(ws[0].dtype), bias.astype(ws[1].dtype)] + ws[2:])
        n_ok += 1
    f.close()
    if verbose:
        print(f"transferred {n_ok}/{len(mapping)} layers from {ckpt_path}")
    return n_ok
