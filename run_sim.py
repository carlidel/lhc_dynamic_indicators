import argparse
import json
import os
from dataclasses import dataclass, field
from random import shuffle
from typing import Literal

import h5py
import henon_map_cpp as hm
import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
from numba import njit
from tqdm import tqdm

from dynamic_indicators_script import (
    H5py_writer,
    add_normed_displacement_to_particles,
    track_log_displacement,
    track_reverse_error_method,
    track_stability,
    track_tune_birkhoff,
)

# create parser
parser = argparse.ArgumentParser(description="Run simulation")
parser.add_argument("--mask", type=str, help="Path to mask json config")
parser.add_argument("--tracking", type=str, help="Path to tracking json config")
parser.add_argument("--kind", type=str, help="Kind of simulation to run")
parser.add_argument("--output", type=str, help="Path to output file")
parser.add_argument(
    "--zeta",
    type=str,
    help="Zeta value for henon map",
    default="min",
    choices=["min", "max", "avg"],
)

context = xo.ContextCupy()

# parse arguments
args = parser.parse_args()

# load mask config
with open(args.mask, "r") as f:
    mask_config = json.load(f)

# load tracking config
with open(args.tracking, "r") as f:
    tracking_config = json.load(f)

# create h5py writer
h5py_writer = H5py_writer(
    os.path.join(
        tracking_config["output"]["path"],
        args.output + "_" + mask_config["name"] + ".h5",
    )
)


# create tracker from mask file
with open(os.path.join(mask_config["maskpath"], mask_config["filename"]), "r") as f:
    mask = json.load(f)

line = xt.Line.from_dict(mask)
tracker = xt.Tracker(_context=context, line=line)


# create particles
x_linspace = np.linspace(
    mask_config["x_min"],
    mask_config["x_max"],
    tracking_config["sampling"]["samples_per_side"],
)
y_linspace = np.linspace(
    mask_config["y_min"],
    mask_config["y_max"],
    tracking_config["sampling"]["samples_per_side"],
)
xx, yy = np.meshgrid(x_linspace, y_linspace)
xx = xx.flatten()
yy = yy.flatten()

if args.zeta == "min":
    zeta = mask_config["zeta_min"]
elif args.zeta == "max":
    zeta = mask_config["zeta_max"]
elif args.zeta == "avg":
    zeta = mask_config["zeta_avg"]

part = xp.Particles(
    _context=context,
    p0c=7000e9,
    x=xx,
    y=yy,
    zeta=zeta,
)

# prepare samples for when necessary
if tracking_config["tracking"]["sampling_method"] == "log":
    t_samples = np.logspace(
        1,
        np.log10(tracking_config["tracking"]["max_iterations"]),
        tracking_config["tracking"]["n_samples"],
    )
    t_samples = np.unique(np.round(t_samples).astype(int))
elif tracking_config["tracking"]["sampling_method"] == "linear":
    t_samples = np.linspace(
        1,
        tracking_config["tracking"]["max_iterations"],
        tracking_config["tracking"]["n_samples"],
    )
    t_samples = np.unique(np.round(t_samples).astype(int))
elif tracking_config["tracking"]["sampling_method"] == "all":
    t_samples = np.arange(1, tracking_config["tracking"]["max_iterations"] + 1)
elif tracking_config["tracking"]["sampling_method"] == "custom":
    t_samples = tracking_config["tracking"]["custom_samples"]
else:
    raise ValueError(
        "Sampling method not recognized. Choose from 'log', 'linear', 'all', or 'custom'"
    )

# run simulation
if args.kind == "stability":
    track_stability(
        tracker,
        part,
        tracking_config["tracking"]["max_iterations"],
        context,
        h5py_writer,
    )
elif args.kind == "rem":
    track_reverse_error_method(
        tracker,
        part,
        t_samples,
        context,
        h5py_writer,
    )
elif args.kind == "log_displacement":
    part_x = add_normed_displacement_to_particles(part, 1.0, "x")
    part_y = add_normed_displacement_to_particles(part, 1.0, "y")
    part_px = add_normed_displacement_to_particles(part, 1.0, "px")
    part_py = add_normed_displacement_to_particles(part, 1.0, "py")
    part_zeta = add_normed_displacement_to_particles(part, 1.0, "zeta")
    part_delta = add_normed_displacement_to_particles(part, 1.0, "delta")

    track_log_displacement(
        tracker,
        part,
        [part_x, part_y, part_px, part_py, part_zeta, part_delta],
        ["disp/x", "disp/y", "disp/px", "disp/py", "disp/zeta", "disp/delta"],
        1.0,
        t_samples,
        10,
        context,
        h5py_writer,
    )
elif args.kind == "tune_birkhoff":
    track_tune_birkhoff(
        tracker,
        part,
        t_samples,
        context,
        h5py_writer,
    )
else:
    raise NotImplementedError("Kind of simulation not recognized.")
