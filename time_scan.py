import datetime
import json
import os
import pickle
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
from tqdm import tqdm

from dynamic_indicators import H5py_writer

context = xo.ContextCupy()
MASK_PATH = "masks/"

selected_mask = "mask_b1_without_bb_21.json"

sample_list = [10, 100, 1000, 10000]
n_turns = [10, 100, 1000, 10000]

ss, nn = np.meshgrid(sample_list, n_turns)
ss = ss.flatten()
nn = nn.flatten()

with open(f"masks/{selected_mask}", "r") as fid:
    loaded_dct = json.load(fid)


line = xt.Line.from_dict(loaded_dct)
tracker = xt.Tracker(_context=context, line=line)

times = []

for s, n in tqdm(zip(ss, nn)):
    print(f"Tracking with {s} samples and {n} turns")

    particles = xp.Particles(
        _context=context,
        p0c=7000e9,
        x=np.ones(s) * 0.0001,
        y=np.ones(s) * 0.0001,
    )

    start = time.time()
    print(particles.x)
    tracker.track(particles, num_turns=n)
    print(particles.x)
    end = time.time()
    print(f"Time: {end - start}")

    times.append((s, n, end - start))

with open("out/time_scan.pickle", "wb") as fid:
    pickle.dump(times, fid)
