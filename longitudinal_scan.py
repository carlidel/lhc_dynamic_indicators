import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
from tqdm import tqdm

from dynamic_indicators import H5pyWriter

context = xo.ContextCupy()
MASK_PATH = "masks/"

selected_masks = [
    "mask_b1_without_bb_33.json",
    "mask_b1_without_bb_51.json",
    "mask_b1_without_bb_21.json",
]

samples = 51
n_turns = 2500

zeta_min = 0.0
zeta_max = 0.5
zeta_list = np.linspace(zeta_min, zeta_max, samples)

for m in tqdm(selected_masks):
    with open(f"masks/{m}", "r") as fid:
        loaded_dct = json.load(fid)

    particles = xp.Particles(
        _context=context,
        p0c=7000e9,
        zeta=zeta_list,
    )

    line = xt.Line.from_dict(loaded_dct)
    tracker = xt.Tracker(_context=context, line=line)
    tracker.track(particles, num_turns=n_turns, turn_by_turn_monitor=True)

    with h5py.File(f"out/longitudinal_scan/{m[:-5]}_stability.h5", "w") as fid:
        fid.create_dataset(
            "zeta",
            data=tracker.record_last_track.zeta,
        )
        fid.create_dataset(
            "delta",
            data=tracker.record_last_track.delta,
        )
