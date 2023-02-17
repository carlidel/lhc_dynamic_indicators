import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

import normed_dynamic_indicators as ndi

# create parser
parser = argparse.ArgumentParser(description="Run simulation")
parser.add_argument("--mask", type=str, help="Path to mask json config")
parser.add_argument("--tracking", type=str, help="Path to tracking json config")
parser.add_argument("--kind", type=str, help="Kind of simulation to run")
parser.add_argument("--output", type=str, help="Output file name")
parser.add_argument(
    "--zeta",
    type=str,
    help="Zeta value for tracking",
    default="min",
    choices=["min", "max", "avg"],
)
parser.add_argument(
    "--displacement",
    type=float,
    help="Displacement value for tracking",
    default=1e-6,
)

# parse arguments
args = parser.parse_args()
context = xo.ContextCupy()


@dataclass
class MaskConfig:
    maskpath: str
    filename: str
    name: str
    x_norm_min: float
    x_norm_max: float
    y_norm_min: float
    y_norm_max: float
    zeta_norm_min: float
    zeta_norm_max: float
    zeta_norm_avg: float
    p0c: float
    nemitt_x: float
    nemitt_y: float


@dataclass
class TrackingConfig:
    samples_per_side: int
    max_iterations: int
    time_samples: int
    sampling_method: Literal["log", "linear", "all", "custom"]
    outpath: str
    custom_samples: list = field(default_factory=list)


# load mask config
with open(args.mask, "r") as f:
    mask_config = json.load(f)
    mask_config = MaskConfig(**mask_config)

# load tracking config
with open(args.tracking, "r") as f:
    tracking_config = json.load(f)
    tracking_config = TrackingConfig(**tracking_config)

# create h5py writer
h5py_writer = ndi.H5pyWriter(
    os.path.join(
        tracking_config.outpath,
        args.output + "_" + mask_config.name + "_zeta_" + args.zeta + ".h5",
    )
)

# create tracker from mask file
with open(os.path.join(mask_config.maskpath, mask_config.filename), "r") as f:
    mask = json.load(f)

line = xt.Line.from_dict(mask)

# little improvement for performance!
line.merge_consecutive_drifts(inplace=True)

tracker = xt.Tracker(_context=context, line=line)
tracker.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=mask_config.p0c)
twiss = tracker.twiss()

# create particles
x_linspace = np.linspace(
    mask_config.x_norm_min,
    mask_config.x_norm_max,
    tracking_config.samples_per_side,
)
y_linspace = np.linspace(
    mask_config.y_norm_min,
    mask_config.y_norm_max,
    tracking_config.samples_per_side,
)
xx, yy = np.meshgrid(x_linspace, y_linspace)
xx = xx.flatten()
yy = yy.flatten()

if args.zeta == "min":
    zeta = mask_config.zeta_norm_min
elif args.zeta == "max":
    zeta = mask_config.zeta_norm_max
elif args.zeta == "avg":
    zeta = mask_config.zeta_norm_avg

part = tracker.build_particles(
    _context=context,
    zeta_norm=zeta,
    x_norm=xx,
    y_norm=yy,
    nemitt_x=mask_config.nemitt_x,
    nemitt_y=mask_config.nemitt_y,
)

# prepare samples for when necessary
if tracking_config.sampling_method == "log":
    t_samples = np.logspace(
        1,
        np.log10(tracking_config.max_iterations),
        tracking_config.time_samples,
    )
    t_samples = np.unique(np.round(t_samples).astype(int))
elif tracking_config.sampling_method == "linear":
    t_samples = np.linspace(
        1,
        tracking_config.max_iterations,
        tracking_config.time_samples,
    )
    t_samples = np.unique(np.round(t_samples).astype(int))
elif tracking_config.sampling_method == "all":
    t_samples = np.arange(1, tracking_config.max_iterations + 1)
elif tracking_config.sampling_method == "custom":
    t_samples = tracking_config.custom_samples
else:
    raise ValueError(
        "Sampling method not recognized. Choose from 'log', 'linear', 'all', or 'custom'"
    )

# run simulation
if args.kind == "stability":
    ndi.track_stability(
        tracker,
        part,
        tracking_config.max_iterations,
        context,
        h5py_writer,
    )
elif args.kind == "rem":
    ndi.track_reverse_error_method(
        tracker,
        twiss,
        (mask_config.nemitt_x, mask_config.nemitt_y),
        part,
        t_samples,
        context,
        h5py_writer,
    )
elif args.kind == "log_displacement":
    part_disp_list = []
    part_name_list = []

    part_x_norm = tracker.build_particles(
        _context=context,
        zeta_norm=zeta,
        x_norm=xx + args.displacement,
        y_norm=yy,
        nemitt_x=mask_config.nemitt_x,
        nemitt_y=mask_config.nemitt_y,
    )
    part_disp_list.append(part_x_norm)
    part_name_list.append("disp/x_norm")

    part_px_norm = tracker.build_particles(
        _context=context,
        zeta_norm=zeta,
        x_norm=xx,
        px_norm=args.displacement,
        y_norm=yy,
        nemitt_x=mask_config.nemitt_x,
        nemitt_y=mask_config.nemitt_y,
    )
    part_disp_list.append(part_px_norm)
    part_name_list.append("disp/px_norm")

    part_y_norm = tracker.build_particles(
        _context=context,
        zeta_norm=zeta,
        x_norm=xx,
        y_norm=yy + args.displacement,
        nemitt_x=mask_config.nemitt_x,
        nemitt_y=mask_config.nemitt_y,
    )
    part_disp_list.append(part_y_norm)
    part_name_list.append("disp/y_norm")

    part_py_norm = tracker.build_particles(
        _context=context,
        zeta_norm=zeta,
        x_norm=xx,
        y_norm=yy,
        py_norm=args.displacement,
        nemitt_x=mask_config.nemitt_x,
        nemitt_y=mask_config.nemitt_y,
    )
    part_disp_list.append(part_py_norm)
    part_name_list.append("disp/py_norm")

    part_zeta_norm = tracker.build_particles(
        _context=context,
        zeta_norm=zeta + args.displacement,
        x_norm=xx,
        y_norm=yy,
        nemitt_x=mask_config.nemitt_x,
        nemitt_y=mask_config.nemitt_y,
    )
    part_disp_list.append(part_zeta_norm)
    part_name_list.append("disp/zeta_norm")

    part_pzeta_norm = tracker.build_particles(
        _context=context,
        zeta_norm=zeta,
        x_norm=xx,
        y_norm=yy,
        pzeta_norm=args.displacement,
        nemitt_x=mask_config.nemitt_x,
        nemitt_y=mask_config.nemitt_y,
    )
    part_disp_list.append(part_pzeta_norm)
    part_name_list.append("disp/pzeta_norm")

    ndi.track_log_displacement_singles_birkhoff(
        tracker,
        twiss,
        (mask_config.nemitt_x, mask_config.nemitt_y),
        part,
        part_disp_list,
        part_name_list,
        args.displacement,
        t_samples,
        context,
        h5py_writer,
    )
elif args.kind == "tune_birkhoff":
    t_samples = t_samples[t_samples > 10]
    samples_half = t_samples // 2
    samples_full = samples_half * 2
    samples_from = np.concatenate((np.zeros(samples_half.size), samples_half))
    samples_to = np.concatenate((samples_half, samples_full))

    samples_from = np.asarray(samples_from, dtype=int)
    samples_to = np.asarray(samples_to, dtype=int)

    ndi.track_tune_birkhoff(
        tracker,
        twiss,
        (mask_config.nemitt_x, mask_config.nemitt_y),
        part,
        samples_from,
        samples_to,
        context,
        h5py_writer,
    )
else:
    raise NotImplementedError("Kind of simulation not recognized.")
