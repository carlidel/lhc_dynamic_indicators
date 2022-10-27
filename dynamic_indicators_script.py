import datetime
import logging
from dataclasses import dataclass
from typing import List

import cupy as cp
import h5py
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt  # To avoid circular imports
from numba import njit
from scipy.constants import c as clight
from scipy.optimize import fsolve

from . import linear_normal_form as lnf

## FROM GIANNI'S CODE ############################

DEFAULT_STEPS_R_MATRIX = {
    "dx": 1e-7,
    "dpx": 1e-10,
    "dy": 1e-7,
    "dpy": 1e-10,
    "dzeta": 1e-6,
    "ddelta": 1e-7,
}

#################################################

UNITARY_STEPS_R_MATRIX = {
    "dx": 1.0,
    "dpx": 1.0,
    "dy": 1.0,
    "dpy": 1.0,
    "dzeta": 1.0,
    "ddelta": 1.0,
}

#################################################


def normed_distance(
    particles_a: xp.Particles,
    particles_b: xp.Particles,
    kind: str = "4d",
    metric: dict = DEFAULT_STEPS_R_MATRIX,
):
    if kind == "4d":
        return (
            ((particles_a.x - particles_b.x) / metric["dx"]) ** 2
            + ((particles_a.px - particles_b.px) / metric["dpx"]) ** 2
            + ((particles_a.y - particles_b.y) / metric["dy"]) ** 2
            + ((particles_a.py - particles_b.py) / metric["dpy"]) ** 2
        ) ** 0.5
    elif kind == "6d":
        return (
            ((particles_a.x - particles_b.x) / metric["dx"]) ** 2
            + ((particles_a.px - particles_b.px) / metric["dpx"]) ** 2
            + ((particles_a.y - particles_b.y) / metric["dy"]) ** 2
            + ((particles_a.py - particles_b.py) / metric["dpy"]) ** 2
            + ((particles_a.zeta - particles_b.zeta) / metric["dzeta"]) ** 2
            + ((particles_a.delta - particles_b.delta) / metric["ddelta"]) ** 2
        ) ** 0.5
    elif kind == "x-px":
        return (
            ((particles_a.x - particles_b.x) / metric["dx"]) ** 2
            + ((particles_a.px - particles_b.px) / metric["dpx"]) ** 2
        ) ** 0.5
    elif kind == "y-py":
        return (
            ((particles_a.y - particles_b.y) / metric["dy"]) ** 2
            + ((particles_a.py - particles_b.py) / metric["dpy"]) ** 2
        ) ** 0.5
    elif kind == "zeta-delta":
        return (
            ((particles_a.zeta - particles_b.zeta) / metric["dzeta"]) ** 2
            + ((particles_a.delta - particles_b.delta) / metric["ddelta"]) ** 2
        ) ** 0.5
    else:
        raise ValueError("Unknown kind of distance")


def normed_direction(
    particles_a: xp.Particles,
    particles_b: xp.Particles,
    kind: str = "4d",
    metric: dict = DEFAULT_STEPS_R_MATRIX,
):
    if kind == "4d":
        distance = normed_distance(particles_a, particles_b, kind="4d", metric=metric)
        return (
            ((particles_b.x - particles_a.x) / metric["dx"]) / distance,
            ((particles_b.px - particles_a.px) / metric["dpx"]) / distance,
            ((particles_b.y - particles_a.y) / metric["dy"]) / distance,
            ((particles_b.py - particles_a.py) / metric["dpy"]) / distance,
        ), distance
    elif kind == "6d":
        distance = normed_distance(particles_a, particles_b, kind="6d", metric=metric)
        return (
            ((particles_b.x - particles_a.x) / metric["dx"]) / distance,
            ((particles_b.px - particles_a.px) / metric["dpx"]) / distance,
            ((particles_b.y - particles_a.y) / metric["dy"]) / distance,
            ((particles_b.py - particles_a.py) / metric["dpy"]) / distance,
            ((particles_b.zeta - particles_a.zeta) / metric["dzeta"]) / distance,
            ((particles_b.delta - particles_a.delta) / metric["ddelta"]) / distance,
        ), distance
    elif kind == "x-px":
        distance = normed_distance(particles_a, particles_b, kind="x-px", metric=metric)
        return (
            ((particles_b.x - particles_a.x) / metric["dx"]) / distance,
            ((particles_b.px - particles_a.px) / metric["dpx"]) / distance,
        ), distance
    elif kind == "y-py":
        distance = normed_distance(particles_a, particles_b, kind="y-py", metric=metric)
        return (
            ((particles_b.y - particles_a.y) / metric["dy"]) / distance,
            ((particles_b.py - particles_a.py) / metric["dpy"]) / distance,
        ), distance
    elif kind == "zeta-delta":
        distance = normed_distance(
            particles_a, particles_b, kind="zeta-delta", metric=metric
        )
        return (
            ((particles_b.zeta - particles_a.zeta) / metric["dzeta"]) / distance,
            ((particles_b.delta - particles_a.delta) / metric["ddelta"]) / distance,
        ), distance
    else:
        raise ValueError("Unknown kind of distance")


def add_normed_displacement_to_particles(
    part: xp.Particles,
    displacement_module: float,
    displacement_kind: str,
    metric: dict = DEFAULT_STEPS_R_MATRIX,
):
    part_d = part.copy()
    if displacement_kind == "x":
        part_d.x += displacement_module * metric["dx"]
    elif displacement_kind == "px":
        part_d.px += displacement_module * metric["dpx"]
    elif displacement_kind == "y":
        part_d.y += displacement_module * metric["dy"]
    elif displacement_kind == "py":
        part_d.py += displacement_module * metric["dpy"]
    elif displacement_kind == "zeta":
        part_d.zeta += displacement_module * metric["dzeta"]
    elif displacement_kind == "delta":
        part_d.delta += displacement_module * metric["ddelta"]
    elif displacement_kind == "random_4d":
        n_part = len(part_d.x)
        v = np.random.uniform(size=n_part * 4).reshape((4, n_part))
        v /= np.linalg.norm(v, axis=0)
        part_d.x += displacement_module * v[0] * metric["dx"]
        part_d.px += displacement_module * v[1] * metric["dpx"]
        part_d.y += displacement_module * v[2] * metric["dy"]
        part_d.py += displacement_module * v[3] * metric["dpy"]
    elif displacement_kind == "random_6d":
        n_part = len(part_d.x)
        v = np.random.uniform(size=n_part * 6).reshape((6, n_part))
        v /= np.linalg.norm(v, axis=0)
        part_d.x += displacement_module * v[0] * metric["dx"]
        part_d.px += displacement_module * v[1] * metric["dpx"]
        part_d.y += displacement_module * v[2] * metric["dy"]
        part_d.py += displacement_module * v[3] * metric["dpy"]
        part_d.zeta += displacement_module * v[4] * metric["dzeta"]
        part_d.delta += displacement_module * v[5] * metric["ddelta"]

    return part_d


def realign_normed_particles(
    part_ref: xp.Particles,
    part_target: xp.Particles,
    module: float,
    kind: str = "4d",
    metric: dict = DEFAULT_STEPS_R_MATRIX,
):
    norm_d, distance = normed_direction(part_ref, part_target, kind=kind, metric=metric)
    if kind == "4d":
        part_target.x = (module / distance) * norm_d[0] * metric["dx"] + part_ref.x
        part_target.px = (module / distance) * norm_d[1] * metric["dpx"] + part_ref.px
        part_target.y = (module / distance) * norm_d[2] * metric["dy"] + part_ref.y
        part_target.py = (module / distance) * norm_d[3] * metric["dpy"] + part_ref.py
    elif kind == "6d":
        part_target.x = (module / distance) * norm_d[0] * metric["dx"] + part_ref.x
        part_target.px = (module / distance) * norm_d[1] * metric["dpx"] + part_ref.px
        part_target.y = (module / distance) * norm_d[2] * metric["dy"] + part_ref.y
        part_target.py = (module / distance) * norm_d[3] * metric["dpy"] + part_ref.py
        part_target.zeta = (module / distance) * norm_d[4] * metric[
            "dzeta"
        ] + part_ref.zeta
        part_target.delta = (module / distance) * norm_d[5] * metric[
            "ddelta"
        ] + part_ref.delta
    elif kind == "x-px":
        part_target.x = (module / distance) * norm_d[0] * metric["dx"] + part_ref.x
        part_target.px = (module / distance) * norm_d[1] * metric["dpx"] + part_ref.px
    elif kind == "y-py":
        part_target.y = (module / distance) * norm_d[0] * metric["dy"] + part_ref.y
        part_target.py = (module / distance) * norm_d[1] * metric["dpy"] + part_ref.py
    elif kind == "zeta-delta":
        part_target.zeta = (module / distance) * norm_d[0] * metric[
            "dzeta"
        ] + part_ref.zeta
        part_target.delta = (module / distance) * norm_d[1] * metric[
            "ddelta"
        ] + part_ref.delta
    else:
        raise ValueError("Unknown kind of distance")


class H5py_writer:
    def __init__(self, filename, compression=None):
        self.filename = filename
        self.compression = compression

    def write_data(self, data, dataset_name, overwrite=False):
        with h5py.File(self.filename, mode="a") as f:
            # check if dataset already exists
            if dataset_name in f:
                if overwrite:
                    del f[dataset_name]
                else:
                    raise ValueError(
                        f"Dataset {dataset_name} already exists in file {self.filename}"
                    )
            if self.compression is None:
                f.create_dataset(dataset_name, data=data)
            else:
                f.create_dataset(dataset_name, data=data, compression=self.compression)


@dataclass
class ParticlesData:
    x: np.ndarray
    px: np.ndarray
    y: np.ndarray
    py: np.ndarray
    zeta: np.ndarray
    delta: np.ndarray
    steps: np.ndarray

    def create_particles(self, particle_ref: xp.Particles, _context):
        p = xp.build_particles(
            x=self.x,
            px=self.px,
            y=self.y,
            py=self.py,
            zeta=self.zeta,
            delta=self.delta,
            particle_ref=particle_ref,
            _context=_context,
        )
        p.at_turn = _context.nparray_to_context_array(self.steps)
        return p


def get_particle_data(particles: xp.Particles, _context, retidx=False):
    x = _context.nparray_from_context_array(particles.x)
    px = _context.nparray_from_context_array(particles.px)
    y = _context.nparray_from_context_array(particles.y)
    py = _context.nparray_from_context_array(particles.py)
    zeta = _context.nparray_from_context_array(particles.zeta)
    delta = _context.nparray_from_context_array(particles.delta)
    at_turn = _context.nparray_from_context_array(particles.at_turn)
    particle_id = _context.nparray_from_context_array(particles.particle_id)

    argsort = particle_id.argsort()
    n_turns = int(np.max(at_turn))

    x = x[argsort]
    px = px[argsort]
    y = y[argsort]
    py = py[argsort]
    zeta = zeta[argsort]
    delta = delta[argsort]
    at_turn = at_turn[argsort]

    x[at_turn < n_turns] = np.nan
    px[at_turn < n_turns] = np.nan
    y[at_turn < n_turns] = np.nan
    py[at_turn < n_turns] = np.nan
    zeta[at_turn < n_turns] = np.nan
    delta[at_turn < n_turns] = np.nan

    if not retidx:
        return ParticlesData(
            x=x, px=px, y=y, py=py, zeta=zeta, delta=delta, steps=at_turn
        )
    else:
        return (
            ParticlesData(
                x=x, px=px, y=y, py=py, zeta=zeta, delta=delta, steps=at_turn
            ),
            argsort,
        )


def compute_t_events(samples: List[int], turns_per_normalization: int):
    event_list = []
    # sort samples
    samples = sorted(samples)
    max_turns = np.max(samples)
    event_list += [["sample", t] for t in samples]

    event_list += [
        ["normalize", t]
        for t in np.arange(turns_per_normalization, max_turns, turns_per_normalization)
    ]
    event_list = list(
        sorted(
            event_list,
            key=lambda x: x[1]
            + (0.0 if x[0] == "sample" else 0.1 if x[0] == "normalize" else 0.2),
        )
    )
    return event_list


def track_stability(
    tracker: xt.Tracker,
    part: xp.Particles,
    n_turns: int,
    _context,
    outfile: H5py_writer,
):
    start = datetime.datetime.now()
    print(f"Starting at: {start}")

    tracker.track(part, n_turns=n_turns)

    end = datetime.datetime.now()
    print(f"Finished at: {end}")
    delta = (end - start).total_seconds()
    print(f"Hours elapsed: {delta / (60*60)}")

    turns = get_particle_data(part, _context, retidx=False).steps
    outfile.write_data("stability", turns)


def track_log_displacement(
    tracker: xt.Tracker,
    part: xp.Particles,
    d_part_list: List[xp.Particles],
    d_part_names: List[str],
    initial_displacement: float,
    samples: List[int],
    turns_per_normalization: int,
    _context,
    outfile: H5py_writer,
    kind: str = "4d",
    metric: dict = DEFAULT_STEPS_R_MATRIX,
):
    n_particles = len(part.x)
    event_list = compute_t_events(samples, turns_per_normalization)
    current_t = 0
    log_displacement = [np.zeros(n_particles) for i in range(len(d_part_list))]

    part_data = get_particle_data(part, _context, retidx=False)
    outfile.write_data(f"reference/initial/x", part_data.x)
    outfile.write_data(f"reference/initial/px", part_data.px)
    outfile.write_data(f"reference/initial/y", part_data.y)
    outfile.write_data(f"reference/initial/py", part_data.py)
    outfile.write_data(f"reference/initial/zeta", part_data.zeta)
    outfile.write_data(f"reference/initial/delta", part_data.delta)

    for i, d_part in enumerate(d_part_list):
        part_data = get_particle_data(d_part, _context, retidx=False)
        outfile.write_data(f"{d_part_names[i]}/initial/x", part_data.x)
        outfile.write_data(f"{d_part_names[i]}/initial/px", part_data.px)
        outfile.write_data(f"{d_part_names[i]}/initial/y", part_data.y)
        outfile.write_data(f"{d_part_names[i]}/initial/py", part_data.py)
        outfile.write_data(f"{d_part_names[i]}/initial/zeta", part_data.zeta)
        outfile.write_data(f"{d_part_names[i]}/initial/delta", part_data.delta)

    for kind, time in event_list:
        print(f"Event {kind}, at time {time}. Current time {current_t}.")
        delta_t = time - current_t
        if delta_t != 0:
            tracker.track(part, num_turns=delta_t)
            for d_part in d_part_list:
                tracker.track(d_part, num_turns=delta_t)
            current_t = time

        if kind == "normalize":
            for i, d_part in enumerate(d_part_list):
                displacement = _context.nparray_from_context_array(
                    normed_distance(part, d_part, kind=kind, metric=metric)
                )

                log_displacement[i] += np.log(displacement / initial_displacement)
                realign_normed_particles(
                    part, d_part, initial_displacement, kind=kind, metric=metric
                )

        elif kind == "sample":
            for i, d_part in enumerate(d_part_list):
                displacement = _context.nparray_from_context_array(
                    normed_distance(part, d_part, kind=kind, metric=metric)
                )
                disp_to_save = log_displacement[i] + np.log(
                    displacement / initial_displacement
                )

                part_data = get_particle_data(part, _context, retidx=False)
                outfile.write_data(f"{d_part_names[i]}/log_disp/{time}", disp_to_save)
                outfile.write_data(f"{d_part_names[i]}/x/{time}", part_data.x)
                outfile.write_data(f"{d_part_names[i]}/px/{time}", part_data.px)
                outfile.write_data(f"{d_part_names[i]}/y/{time}", part_data.y)
                outfile.write_data(f"{d_part_names[i]}/py/{time}", part_data.py)
                outfile.write_data(f"{d_part_names[i]}/zeta/{time}", part_data.zeta)
                outfile.write_data(f"{d_part_names[i]}/delta/{time}", part_data.delta)


def track_reverse_error_method(
    tracker: xt.Tracker,
    part: xp.Particles,
    samples: List[int],
    _context,
    outfile: H5py_writer,
):
    backtracker = tracker.get_backtracker(_context=_context)

    part_data = get_particle_data(part, _context, retidx=False)
    outfile.write_data(f"reference/initial/x", part_data.x)
    outfile.write_data(f"reference/initial/px", part_data.px)
    outfile.write_data(f"reference/initial/y", part_data.y)
    outfile.write_data(f"reference/initial/py", part_data.py)
    outfile.write_data(f"reference/initial/zeta", part_data.zeta)
    outfile.write_data(f"reference/initial/delta", part_data.delta)

    f_part = part.copy()
    current_t = 0
    for i, t in enumerate(samples):
        print(f"Tracking {t} turns... ({i+1}/{len(samples)})")
        delta_t = t - current_t
        tracker.track(f_part, n_turns=delta_t)
        current_t = t
        r_part = f_part.copy()
        backtracker.track(r_part, n_turns=t)

        part_data = get_particle_data(r_part, _context=_context)

        outfile.write_data(f"reverse/x/{t}", part_data.x)
        outfile.write_data(f"reverse/px/{t}", part_data.px)
        outfile.write_data(f"reverse/y/{t}", part_data.y)
        outfile.write_data(f"reverse/py/{t}", part_data.py)
        outfile.write_data(f"reverse/zeta/{t}", part_data.zeta)
        outfile.write_data(f"reverse/delta/{t}", part_data.delta)


def birkhoff_weights(n):
    weights = np.arange(n, dtype=np.float64)
    weights /= n
    weights = np.exp(-1 / (weights * (1 - weights)))
    return weights / np.sum(weights)


def birkhoff_weights_cupy(n):
    weights = cp.arange(n, dtype=np.float64)
    weights /= n
    weights = cp.exp(-1 / (weights * (1 - weights)))
    return weights / cp.sum(weights)


def track_tune_birkhoff(
    tracker: xt.Tracker,
    part: xp.Particles,
    samples: List[int],
    _context,
    outfile: H5py_writer,
):
    part_data = get_particle_data(part, _context, retidx=False)
    outfile.write_data(f"reference/initial/x", part_data.x)
    outfile.write_data(f"reference/initial/px", part_data.px)
    outfile.write_data(f"reference/initial/y", part_data.y)
    outfile.write_data(f"reference/initial/py", part_data.py)
    outfile.write_data(f"reference/initial/zeta", part_data.zeta)
    outfile.write_data(f"reference/initial/delta", part_data.delta)

    max_turns = np.max(samples)

    tracker.track(part, n_turns=max_turns, turn_by_turn_monitor=True)

    # combine x and px in a complex number
    z_x = tracker.record_last_track.x + 1j * tracker.record_last_track.px
    z_y = tracker.record_last_track.y + 1j * tracker.record_last_track.py

    z_x = np.angle(z_x)
    z_y = np.angle(z_y)

    z_x = z_x[:, 1:] - z_x[:, :-1]
    z_y = z_y[:, 1:] - z_y[:, :-1]

    z_x[z_x < 0] += 2 * np.pi
    z_y[z_y < 0] += 2 * np.pi

    for i, s in enumerate(samples):
        s_half = s // 2
        s = s_half * 2

        if type(_context) == xo.ContextCpu:
            weights = birkhoff_weights(s_half)
        elif type(_context) == xo.ContextCupy:
            weights = birkhoff_weights_cupy(s_half)
        else:
            raise NotImplementedError

        tune_x_1 = np.sum(z_x[:, :s_half] * weights, axis=1)
        tune_x_2 = np.sum(z_x[:, s_half:s] * weights, axis=1)

        tune_y_1 = np.sum(z_y[:, :s_half] * weights, axis=1)
        tune_y_2 = np.sum(z_y[:, s_half:s] * weights, axis=1)

        outfile.write_data(f"tune/x/0/{s_half}", tune_x_1)
        outfile.write_data(f"tune/x/{s_half}/{s}", tune_x_2)
        outfile.write_data(f"tune/y/0/{s_half}", tune_y_1)
        outfile.write_data(f"tune/y/{s_half}/{s}", tune_y_2)
