import datetime
import logging
from dataclasses import dataclass

import h5py
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt  # To avoid circular imports
from scipy.constants import c as clight
from scipy.optimize import fsolve


class H5py_writer:
    def __init__(self, filename, compression=None):
        self.filename = filename
        self.compression = compression

    def write_data(self, dataset_name:str, data:np.ndarray):
        with h5py.File(self.filename, mode="w") as f:
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


def add_displacement_to_particles(
    part: xp.Particles, displacement_module: float, displacement_kind: str
):
    part_d = part.copy()
    if displacement_kind == "x":
        part_d.x += displacement_module
    elif displacement_kind == "px":
        part_d.px += displacement_module
    elif displacement_kind == "y":
        part_d.y += displacement_module
    elif displacement_kind == "py":
        part_d.py += displacement_module
    elif displacement_kind == "zeta":
        part_d.zeta += displacement_module
    elif displacement_kind == "delta":
        part_d.delta += displacement_module
    elif displacement_kind == "random_4d":
        n_part = len(part_d.x)
        v = np.random.uniform(size=n_part * 4).reshape((4, n_part))
        v /= np.linalg.norm(v, axis=0)
        part_d.x += displacement_module * v[0]
        part_d.px += displacement_module * v[1]
        part_d.y += displacement_module * v[2]
        part_d.py += displacement_module * v[3]
    elif displacement_kind == "random_6d":
        n_part = len(part_d.x)
        v = np.random.uniform(size=n_part * 6).reshape((6, n_part))
        v /= np.linalg.norm(v, axis=0)
        part_d.x += displacement_module * v[0]
        part_d.px += displacement_module * v[1]
        part_d.y += displacement_module * v[2]
        part_d.py += displacement_module * v[3]
        part_d.zeta += displacement_module * v[4]
        part_d.delta += displacement_module * v[5]

    return part_d


def realign_particles(
    particles_ref: xp.Particles,
    particles_target: xp.Particles,
    module: float,
    realign_4d_only: bool,
    _context,
):
    p_ref = get_particle_data(particles_ref, _context=_context, retidx=True)
    p_target, argsort = get_particle_data(
        particles_target, _context=_context, retidx=True
    )
    idxs = argsort.argsort()

    if realign_4d_only:
        distance = np.sqrt(
            (p_ref.x - p_target.x) ** 2
            + (p_ref.px - p_target.px) ** 2
            + (p_ref.y - p_target.y) ** 2
            + (p_ref.py - p_target.py) ** 2
        )
    else:
        distance = np.sqrt(
            (p_ref.x - p_target.x) ** 2
            + (p_ref.px - p_target.px) ** 2
            + (p_ref.y - p_target.y) ** 2
            + (p_ref.py - p_target.py) ** 2
            + (p_ref.zeta - p_target.zeta) ** 2
            + (p_ref.delta - p_target.delta) ** 2
        )

    ratio = module / distance

    particles_target.x = _context.nparray_to_context_array(
        (p_ref.x + (p_target.x - p_ref.x) * ratio)[idxs]
    )
    particles_target.px = _context.nparray_to_context_array(
        (p_ref.px + (p_target.px - p_ref.px) * ratio)[idxs]
    )
    particles_target.y = _context.nparray_to_context_array(
        (p_ref.y + (p_target.y - p_ref.y) * ratio)[idxs]
    )
    particles_target.py = _context.nparray_to_context_array(
        (p_ref.py + (p_target.py - p_ref.py) * ratio)[idxs]
    )
    if not realign_4d_only:
        particles_target.zeta = _context.nparray_to_context_array(
            (p_ref.zeta + (p_target.zeta - p_ref.zeta) * ratio)[idxs]
        )
        particles_target.delta = _context.nparray_to_context_array(
            (p_ref.delta + (p_target.delta - p_ref.delta) * ratio)[idxs]
        )
    return particles_target


def get_displacement_module(
    particles_1: xp.Particles, particles_2: xp.Particles, _context
):
    p_1 = get_particle_data(particles_1, _context=_context)
    p_2 = get_particle_data(particles_2, _context=_context)

    distance = np.sqrt(
        (p_1.x - p_2.x) ** 2
        + (p_1.px - p_2.px) ** 2
        + (p_1.y - p_2.y) ** 2
        + (p_1.py - p_2.py) ** 2
        + (p_1.zeta - p_2.zeta) ** 2
        + (p_1.delta - p_2.delta) ** 2
    )

    return distance


def get_displacement_direction(
    particles_1: xp.Particles, particles_2: xp.Particles, _context
):
    p_1 = get_particle_data(particles_1, _context=_context)
    p_2 = get_particle_data(particles_2, _context=_context)

    direction = np.array(
        [
            p_1.x - p_2.x,
            p_1.px - p_2.px,
            p_1.y - p_2.y,
            p_1.py - p_2.py,
            p_1.zeta - p_2.zeta,
            p_1.delta - p_2.delta,
        ]
    )
    direction /= np.linalg.norm(direction, axis=0)

    return direction


def compute_t_events(samples, turns_per_sample, turns_per_normalization):
    event_list = []
    max_turns = turns_per_sample * samples + 1
    event_list += [
        ["sample", t] for t in np.arange(turns_per_sample, max_turns, turns_per_sample)
    ]

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


def track_stability(tracker, part, n_turns, _context, outfile: H5py_writer):
    start = datetime.datetime.now()
    print(f"Starting at: {start}")

    tracker.track(part, num_turns=n_turns)
    turns = get_particle_data(part, _context, retidx=False).steps
    
    end = datetime.datetime.now()
    print(f"Finished at: {end}")
    delta = (end - start).total_seconds()
    print(f"Hours elapsed: {delta / (60*60)}")
    
    outfile.write_data("stability", turns)


def track_log_displacement(
    tracker,
    part,
    d_part,
    initial_displacement,
    samples,
    turns_per_sample,
    turns_per_normalization,
    _context,
    outfile: H5py_writer,
):
    n_particles = len(part.x)
    event_list = compute_t_events(samples, turns_per_sample, turns_per_normalization)
    current_t = 0
    log_displacement = np.zeros(n_particles)

    for kind, time in event_list:
        print(f"Event {kind}, at time {time}. Current time {current_t}.")
        delta_t = time - current_t
        if delta_t != 0:
            tracker.track(part, num_turns=delta_t)
            tracker.track(d_part, num_turns=delta_t)
            current_t = time

        if kind == "normalize":
            log_displacement += np.log(
                get_displacement_module(part, d_part, _context=_context)
                / initial_displacement
            )
            realign_particles(
                part,
                d_part,
                initial_displacement,
                realign_4d_only=False,
                _context=_context,
            )

        elif kind == "sample":
            disp_to_save = log_displacement + np.log(
                get_displacement_module(part, d_part, _context=_context)
                / initial_displacement
            )
            outfile.write_data(f"lyapunov/{time}", disp_to_save)


def track_6d_displacements(
    tracker,
    part,
    displacement_module,
    samples,
    turns_per_sample,
    turns_per_normalization,
    _context,
    outfile: H5py_writer,
):

    part_x = add_displacement_to_particles(part, displacement_module, "x")
    part_px = add_displacement_to_particles(part, displacement_module, "px")
    part_y = add_displacement_to_particles(part, displacement_module, "y")
    part_py = add_displacement_to_particles(part, displacement_module, "py")
    part_zeta = add_displacement_to_particles(part, displacement_module, "zeta")
    part_delta = add_displacement_to_particles(part, displacement_module, "delta")

    n_particles = len(part.x)
    event_list = compute_t_events(samples, turns_per_sample, turns_per_normalization)
    current_t = 0

    log_displacement_x = np.zeros(n_particles)
    log_displacement_px = np.zeros(n_particles)
    log_displacement_y = np.zeros(n_particles)
    log_displacement_py = np.zeros(n_particles)
    log_displacement_zeta = np.zeros(n_particles)
    log_displacement_delta = np.zeros(n_particles)

    for kind, time in event_list:
        print(f"Event {kind}, at time {time}. Current time {current_t}.")
        delta_t = time - current_t
        if delta_t != 0:
            tracker.track(part, num_turns=delta_t)
            tracker.track(part_x, num_turns=delta_t)
            tracker.track(part_px, num_turns=delta_t)
            tracker.track(part_y, num_turns=delta_t)
            tracker.track(part_py, num_turns=delta_t)
            tracker.track(part_zeta, num_turns=delta_t)
            tracker.track(part_delta, num_turns=delta_t)
            current_t = time

        if kind == "normalize":
            log_displacement_x += np.log(
                get_displacement_module(part, part_x, _context=_context)
                / displacement_module
            )
            realign_particles(
                part,
                part_x,
                displacement_module,
                realign_4d_only=False,
                _context=_context,
            )
            log_displacement_px += np.log(
                get_displacement_module(part, part_px, _context=_context)
                / displacement_module
            )
            realign_particles(
                part,
                part_px,
                displacement_module,
                realign_4d_only=False,
                _context=_context,
            )
            log_displacement_y += np.log(
                get_displacement_module(part, part_y, _context=_context)
                / displacement_module
            )
            realign_particles(
                part,
                part_y,
                displacement_module,
                realign_4d_only=False,
                _context=_context,
            )
            log_displacement_py += np.log(
                get_displacement_module(part, part_py, _context=_context)
                / displacement_module
            )
            realign_particles(
                part,
                part_py,
                displacement_module,
                realign_4d_only=False,
                _context=_context,
            )
            log_displacement_zeta += np.log(
                get_displacement_module(part, part_zeta, _context=_context)
                / displacement_module
            )
            realign_particles(
                part,
                part_zeta,
                displacement_module,
                realign_4d_only=False,
                _context=_context,
            )
            log_displacement_delta += np.log(
                get_displacement_module(part, part_delta, _context=_context)
                / displacement_module
            )
            realign_particles(
                part,
                part_delta,
                displacement_module,
                realign_4d_only=False,
                _context=_context,
            )

        elif kind == "sample":
            module_x = log_displacement_x + np.log(
                get_displacement_module(part, part_x, _context=_context)
                / displacement_module
            )
            module_px = log_displacement_px + np.log(
                get_displacement_module(part, part_px, _context=_context)
                / displacement_module
            )
            module_y = log_displacement_y + np.log(
                get_displacement_module(part, part_y, _context=_context)
                / displacement_module
            )
            module_py = log_displacement_py + np.log(
                get_displacement_module(part, part_py, _context=_context)
                / displacement_module
            )
            module_zeta = log_displacement_zeta + np.log(
                get_displacement_module(part, part_zeta, _context=_context)
                / displacement_module
            )
            module_delta = log_displacement_delta + np.log(
                get_displacement_module(part, part_delta, _context=_context)
                / displacement_module
            )
            outfile.write_data(f"lyapunov/x/{time}", module_x)
            outfile.write_data(f"lyapunov/px/{time}", module_px)
            outfile.write_data(f"lyapunov/y/{time}", module_y)
            outfile.write_data(f"lyapunov/py/{time}", module_py)
            outfile.write_data(f"lyapunov/zeta/{time}", module_zeta)
            outfile.write_data(f"lyapunov/delta/{time}", module_delta)

            direction_x = get_displacement_direction(part, part_x)
            direction_px = get_displacement_direction(part, part_px)
            direction_y = get_displacement_direction(part, part_y)
            direction_py = get_displacement_direction(part, part_py)
            direction_zeta = get_displacement_direction(part, part_zeta)
            direction_delta = get_displacement_direction(part, part_delta)

            outfile.write_data(f"direction/x/{time}", direction_x)
            outfile.write_data(f"direction/px/{time}", direction_px)
            outfile.write_data(f"direction/y/{time}", direction_y)
            outfile.write_data(f"direction/py/{time}", direction_py)
            outfile.write_data(f"direction/zeta/{time}", direction_zeta)
            outfile.write_data(f"direction/delta/{time}", direction_delta)


def track_reverse_error_method(
    tracker,
    part,
    samples,
    turns_per_sample,
    _context,
    outfile: H5py_writer,
    only_4d=False,
):
    backtracker = tracker.get_backtracker(_context=_context)

    f_part = part.copy()
    for i in range(samples):
        print(f"Tracking {i * turns_per_sample} turns... ({i+1}/{samples})")

        tracker.track(f_part, num_turns=turns_per_sample)
        r_part = f_part.copy()
        backtracker.track(r_part, num_turns=turns_per_sample * (i + 1))

        data_0 = get_particle_data(f_part, _context=_context)
        data_1 = get_particle_data(r_part, _context=_context)

        if only_4d:
            distance = np.sqrt(
                (data_0.x - data_1.x) ** 2
                + (data_0.px - data_1.px) ** 2
                + (data_0.y - data_1.y) ** 2
                + (data_0.py - data_1.py) ** 2
            )
        else:
            distance = np.sqrt(
                (data_0.x - data_1.x) ** 2
                + (data_0.px - data_1.px) ** 2
                + (data_0.y - data_1.y) ** 2
                + (data_0.py - data_1.py) ** 2
                + (data_0.zeta - data_1.zeta) ** 2
                + (data_0.delta - data_1.delta) ** 2
            )
        outfile.write_data(f"rem_distance/{(i+1)*turns_per_sample}", distance)
