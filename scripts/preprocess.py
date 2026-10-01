"""Single source of truth for input preprocessing.

Previously this ~6-line block was copy-pasted into train_qat.py,
float_pretrain.py, grid_shootout.py and extract_hw.py. They happened to agree
bit-exactly, but an ad-hoc reimplementation during debugging got the mask order
wrong and silently produced 24.5% accuracy (~chance) instead of 78.3% -- the
model still ran, so nothing flagged it. Import from here instead of retyping.

The mask MUST be computed BEFORE scaling. pT median is 6.6, so after subtracting
it every constituent with raw pT < 6.6 goes negative; a `> 0` test applied after
scaling therefore misidentifies ~50% of REAL constituents as zero padding.
`assert_scaled` catches exactly that.
"""

import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAD_FRACTION_EXPECTED = 0.617  # measured on x_val: 61.7% of the 128 slots are padding


def load_stats(root=None):
    root = root or _ROOT
    return json.load(open(os.path.join(root, "data/robust_stats.json")))


def robust_scale(x, stats=None, copy=True):
    """(x - median) / iqr per feature, with zero-padded constituents re-zeroed.

    Padding is identified from RAW pT > 0, before any shifting.
    """
    stats = stats if stats is not None else load_stats()
    x = x.copy() if copy else x
    mask_real = x[:, :, 0] > 0            # BEFORE scaling -- see module docstring
    for i, s in enumerate(stats):
        x[:, :, i] = (x[:, :, i] - s["median"]) / s["iqr"]
    x[~mask_real] = 0.0
    return x


def load_jets(splits=("train", "val"), n=None, check=None, stats=None, root=None):
    """Load data/jets_128x3.npz, robust-scaled. Returns (x, y) per split, flattened.

    n: keep only the first n constituents (highest pT). check: run assert_scaled
    on the full 128-slot array (defaults to on when n is None -- the expected
    padding fraction only holds for all 128 slots).

        x_train, y_train, x_val, y_val = load_jets()
        x_val, y_val = load_jets("val")
    """
    root = root or _ROOT
    splits = (splits,) if isinstance(splits, str) else splits
    stats = stats if stats is not None else load_stats(root)
    check = (n is None) if check is None else check
    d = np.load(os.path.join(root, "data/jets_128x3.npz"))
    out = []
    for s in splits:
        x = d[f"x_{s}"]
        if not check and n is not None:
            x = x[:, :n]                  # slice first: cheaper, elementwise-identical
        x = robust_scale(x, stats)
        if check:
            assert_scaled(x, f"x_{s}")
        out += [x if n is None else x[:, :n], d[f"y_{s}"]]
    return tuple(out)


def gmp_edges_scaled(grid_name, stats=None, root=None):
    """GMP bin edges in raw eta/phi units -> robust-scaled units."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from grid_shootout import GRIDS

    stats = stats if stats is not None else load_stats(root)
    iqr = 0.5 * (stats[1]["iqr"] + stats[2]["iqr"])
    return (GRIDS[grid_name] / iqr).tolist()


def assert_scaled(x, tag=""):
    """Fail loudly on the known silent-corruption modes."""
    pad = float((np.abs(x).sum(axis=-1) == 0).mean())
    assert abs(pad - PAD_FRACTION_EXPECTED) < 0.05, (
        f"{tag}: padding fraction {pad:.3f} != expected ~{PAD_FRACTION_EXPECTED}. "
        "Most likely the real/pad mask was computed AFTER scaling, which zeroes "
        "~50% of real constituents (see preprocess.py docstring)."
    )
    return x
