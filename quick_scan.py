import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
from tqdm import tqdm

from dynamic_indicators import H5py_writer, track_stability

TURNS = 1000
samples = 100
bounds = 0.002 # [m]

masks = os.listdir("masks/")

x = np.linspace(0.0, bounds, samples)
xx, yy = np.meshgrid(x, x)
xx = xx.flatten()
yy = yy.flatten()

context = xo.ContextCupy()

for m in tqdm(masks):
    with open(f'masks/{m}', 'r') as fid:
        loaded_dct = json.load(fid)
    
    p = xp.Particles(
        _context=context,
        p0c=7000e9,
        x=xx,
        y=yy,
    )
    
    line = xt.Line.from_dict(loaded_dct)
    tracker = xt.Tracker(_context=context, line=line)
    outfile = H5py_writer(f"out/quick_scan/{m}_stability.h5")
    track_stability(tracker, p, TURNS, context, outfile)
