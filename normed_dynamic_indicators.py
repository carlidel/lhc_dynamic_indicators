import datetime
import warnings
from typing import List, Tuple

import cupy as cp
import h5py
import numpy as np
import xpart as xp
import xtrack as xt
import xtrack.twiss as xtw
from tqdm import tqdm


def get_twiss_data(
    twiss: xtw.TwissTable, nemitt_x: float, nemitt_y: float, _context, idx_pos=0
):
    """Get the twiss data for the given twiss object and the given normalized emittance values.

    Parameters
    ----------
    twiss : xtrack.Twiss
        Twiss object
    nemitt_x : float
        Normalized emittance in x
    nemitt_y : float
        Normalized emittance in y
    _context : xobjects.Context
        Context to use
    idx_pos : int, optional
        Index of the position to use, by default 0

    Returns
    -------
    twiss_data : xobjects.Array
        Twiss data with the following structure:
        [nemitt_x, nemitt_y, twiss.x[idx_pos], twiss.px[idx_pos], twiss.y[idx_pos], twiss.py[idx_pos], twiss.zeta[idx_pos], twiss.ptau[idx_pos]]
    w : xobjects.Array
        Twiss W matrix
    w_inv : xobjects.Array
        Twiss W inverse matrix
    """
    twiss_data = _context.nplike_array_type(8)

    twiss_data[0] = nemitt_x
    twiss_data[1] = nemitt_y

    twiss_data[2] = twiss.x[idx_pos]
    twiss_data[3] = twiss.px[idx_pos]
    twiss_data[4] = twiss.y[idx_pos]
    twiss_data[5] = twiss.py[idx_pos]
    twiss_data[6] = twiss.zeta[idx_pos]
    twiss_data[7] = twiss.ptau[idx_pos]

    w = _context.nparray_to_context_array(twiss.W_matrix[idx_pos])
    w_inv = _context.nparray_to_context_array(np.linalg.inv(twiss.W_matrix[idx_pos]))

    return twiss_data, w, w_inv


def phys_to_norm(part: xp.Particles, normed_part, twiss_data, w_inv):
    """Transform the physical coordinates to normalized coordinates.

    Parameters
    ----------
    part : xp.Particles
        Particles object
    normed_part : xo.ContextArray
        Normalized particles object
    twiss_data : xo.ContextArray
        Twiss data
    w_inv : xo.ContextArray
        Twiss W inverse matrix

    Returns
    -------
    normed_part : xo.ContextArray
        Normalized particles object
    """
    mask = part.state <= 0
    gemitt_x = twiss_data[0] / part._xobject.beta0[0] / part._xobject.gamma0[0]
    gemitt_y = twiss_data[1] / part._xobject.beta0[0] / part._xobject.gamma0[0]

    normed_part[0] = part.x - twiss_data[2]
    normed_part[1] = part.px - twiss_data[3]
    normed_part[2] = part.y - twiss_data[4]
    normed_part[3] = part.py - twiss_data[5]
    normed_part[4] = part.zeta - twiss_data[6]
    normed_part[5] = (part.ptau - twiss_data[7]) / part._xobject.beta0[0]

    normed_part = np.dot(w_inv, normed_part)

    normed_part[0] /= np.sqrt(gemitt_x)
    normed_part[1] /= np.sqrt(gemitt_x)
    normed_part[2] /= np.sqrt(gemitt_y)
    normed_part[3] /= np.sqrt(gemitt_y)

    normed_part[:, mask] = np.nan

    return normed_part


def norm_to_phys(normed_part, part: xp.Particles, twiss_data, w):
    """Transform the normalized coordinates to physical coordinates.

    Parameters
    ----------
    normed_part : xo.ContextArray
        Normalized particles object
    part : xp.Particles
        Particles object
    twiss_data : xo.ContextArray
        Twiss data
    w : xo.ContextArray
        Twiss W matrix

    Returns
    -------
    part : xp.Particles
        Particles object
    """
    mask = part.state <= 0
    gemitt_x = twiss_data[0] / part._xobject.beta0[0] / part._xobject.gamma0[0]
    gemitt_y = twiss_data[1] / part._xobject.beta0[0] / part._xobject.gamma0[0]

    normed = normed_part.copy()
    normed[0] *= np.sqrt(gemitt_x)
    normed[1] *= np.sqrt(gemitt_x)
    normed[2] *= np.sqrt(gemitt_y)
    normed[3] *= np.sqrt(gemitt_y)

    normed = np.dot(w, normed)

    part.zeta = normed[4] + twiss_data[6]
    part.ptau = normed[5] * part._xobject.beta0[0] + twiss_data[7]

    part.x = normed[0] + twiss_data[2]
    part.px = normed[1] + twiss_data[3]
    part.y = normed[2] + twiss_data[4]
    part.py = normed[3] + twiss_data[5]

    return part


def create_normed_placeholder(particles, twiss_data, w_inv, _context):
    """Create a placeholder for the normalized particles.

    Parameters
    ----------
    particles : xp.Particles
        Particles object
    twiss_data : xo.ContextArray
        Twiss data
    w_inv : xo.ContextArray
        Twiss W inverse matrix
    _context : xobjects.Context
        Context

    Returns
    -------
    normed_particles : xo.ContextArray
        Normalized particles object
    """

    normed_particles = _context.nplike_array_type([6, len(particles.x)])
    normed_particles = phys_to_norm(particles, normed_particles, twiss_data, w_inv)
    return normed_particles


def normalized_distance(norm_repart, norm_part, ref_argsort, part_argsort):
    """Get the distance between the normalized particles.

    Parameters
    ----------
    norm_repart : xo.ContextArray
        Normalized reference particles
    norm_part : xo.ContextArray
        Normalized particles
    ref_argsort : np.ndarray
        Reference argsort
    part_argsort : np.ndarray
        Particles argsort

    Returns
    -------
    distance : xo.ContextArray
        Distance between the normalized particles
    """
    distance = (
        np.sum((norm_repart[:, ref_argsort] - norm_part[:, part_argsort]) ** 2, axis=0)
        ** 0.5
    )
    return distance


def normalized_direction(norm_repart, norm_part, ref_argsort, part_argsort):
    """Get the direction between the normalized particles.

    Parameters
    ----------
    norm_repart : xo.ContextArray
        Normalized reference particles
    norm_part : xo.ContextArray
        Normalized particles
    ref_argsort : np.ndarray
        Reference argsort
    part_argsort : np.ndarray
        Particles argsort

    Returns
    -------
    direction : xo.ContextArray
        Direction between the normalized particles
    distance : xo.ContextArray
        Distance between the normalized particles
    """
    distance = normalized_distance(norm_repart, norm_part, ref_argsort, part_argsort)

    direction = (norm_part[:, part_argsort] - norm_repart[:, ref_argsort]) / distance

    return direction, distance


def renormalize_distance(
    repart,
    part,
    norm_repart,
    norm_part,
    twiss_data,
    w,
    w_inv,
    _context,
    target_distance,
):
    """Renormalize the distance between the particles. Also returns the direction and distance between the particles before the renormalization.

    Parameters
    ----------
    repart : xp.Particles
        Reference particles
    part : xp.Particles
        Particles
    norm_repart : xo.ContextArray
        Normalized reference particles
    norm_part : xo.ContextArray
        Normalized particles
    twiss_data : xo.ContextArray
        Twiss data
    w : xo.ContextArray
        Twiss W matrix
    w_inv : xo.ContextArray
        Twiss W inverse matrix
    _context : xo.Context
        Context
    target_distance : float
        Target distance

    Returns
    -------
    part : xp.Particles
        Particles
    direction : xo.ContextArray
        Direction between the normalized particles
    distance : xo.ContextArray
        Distance between the normalized particles
    """
    ref_argsort = np.argsort(repart.particle_id)
    part_argsort = np.argsort(part.particle_id)

    norm_repart = phys_to_norm(repart, norm_repart, twiss_data, w_inv)
    norm_part = phys_to_norm(part, norm_part, twiss_data, w_inv)

    direction, distance = normalized_direction(
        norm_repart, norm_part, ref_argsort, part_argsort
    )

    meta_argsort = np.argsort(part_argsort)

    norm_part = norm_repart[:, ref_argsort] + direction * target_distance

    part = norm_to_phys(norm_part[:, meta_argsort], part, twiss_data, w)

    return part, direction, distance


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


def birkhoff_weights_context(n, _context):
    weights = _context.nplike_array_type([n])
    if type(weights) == np.ndarray:
        return birkhoff_weights(n)
    elif type(weights) == cp.ndarray:
        return birkhoff_weights_cupy(n)
    else:
        raise ValueError("Unknown type of weights")


class H5pyWriter:
    """Class to write data to an HDF5 file."""

    def __init__(self, filename, compression=None):
        self.filename = filename
        self.compression = compression

    def write_data(self, dataset_name: str, data: np.ndarray, overwrite=False):
        """Write data to an HDF5 file.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        data : np.ndarray
            Data to write
        overwrite : bool, optional
            If True, overwrite the dataset if it already exists, by default False, if set to "raise" raise an error if the dataset already exists
        """
        with h5py.File(self.filename, mode="a") as f:
            # check if dataset already exists
            if dataset_name in f:
                if overwrite:
                    del f[dataset_name]
                elif overwrite == "raise":
                    raise ValueError(
                        f"Dataset {dataset_name} already exists in file {self.filename}"
                    )
                else:
                    # just raise a warning and continue
                    warnings.warn(
                        f"Dataset {dataset_name} already exists in file {self.filename}"
                    )
                    return
            if self.compression is None:
                f.create_dataset(dataset_name, data=data)
            else:
                f.create_dataset(dataset_name, data=data, compression=self.compression)


def get_particle_phys_data(particles: xp.Particles, _context, retidx=False):
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
        return dict(x=x, px=px, y=y, py=py, zeta=zeta, delta=delta, at_turn=at_turn)
    else:
        return (
            dict(x=x, px=px, y=y, py=py, zeta=zeta, delta=delta, at_turn=at_turn),
            argsort,
        )


def get_particle_norm_data(
    particles: xp.Particles,
    twiss: xtw.TwissTable,
    nemitt: Tuple[float, float],
    _context,
    retidx=False,
):
    part_norm = twiss.get_normalized_coordinates(
        particles, nemitt_x=nemitt[0], nemitt_y=nemitt[1]
    )

    argsort = np.argsort(part_norm.particle_id)

    x_norm = part_norm.x_norm[argsort]
    px_norm = part_norm.px_norm[argsort]
    y_norm = part_norm.y_norm[argsort]
    py_norm = part_norm.py_norm[argsort]
    zeta_norm = part_norm.zeta_norm[argsort]
    pzeta_norm = part_norm.pzeta_norm[argsort]

    at_turn = _context.nparray_from_context_array(particles.at_turn)[argsort]
    n_turns = int(np.max(at_turn))

    mask = at_turn < n_turns
    x_norm[mask] = np.nan
    px_norm[mask] = np.nan
    y_norm[mask] = np.nan
    py_norm[mask] = np.nan
    zeta_norm[mask] = np.nan
    pzeta_norm[mask] = np.nan

    if not retidx:
        return dict(
            x_norm=x_norm,
            px_norm=px_norm,
            y_norm=y_norm,
            py_norm=py_norm,
            zeta_norm=zeta_norm,
            pzeta_norm=pzeta_norm,
            at_turn=at_turn,
        )
    else:
        return (
            dict(
                x_norm=x_norm,
                px_norm=px_norm,
                y_norm=y_norm,
                py_norm=py_norm,
                zeta_norm=zeta_norm,
                pzeta_norm=pzeta_norm,
                at_turn=at_turn,
            ),
            argsort,
        )


def track_stability(
    tracker: xt.Tracker,
    part: xp.Particles,
    n_turns: int,
    _context,
    outfile: H5pyWriter,
):
    """Track a set of particles for a given number of turns and write the stability times to an HDF5 file.

    Parameters
    ----------
    tracker : xt.Tracker
        Tracker object
    part : xp.Particles
        Particles to track
    n_turns : int
        Number of turns to track
    _context : xo.Context
        Context object
    outfile : H5pyWriter
        HDF5 file to write to
    """
    start = datetime.datetime.now()
    print(f"Starting at: {start}")

    tracker.track(part, num_turns=n_turns)
    turns = get_particle_phys_data(part, _context, retidx=False)["at_turn"]

    end = datetime.datetime.now()
    print(f"Finished at: {end}")
    delta = (end - start).total_seconds()
    print(f"Hours elapsed: {delta / (60*60)}")

    outfile.write_data("stability", turns)


def track_reverse_error_method(
    tracker: xt.Tracker,
    twiss: xtw.TwissTable,
    nemitt: Tuple[float, float],
    part: xp.Particles,
    samples: List[int],
    _context,
    outfile: H5pyWriter,
):
    """Track and backtrack a set of particles for a given number of turns and write the positions in normalized coordinates to an HDF5 file.

    Parameters
    ----------
    tracker : xt.Tracker
        Tracker object
    twiss : xtw.TwissTable
        Twiss table
    nemitt : Tuple[float, float]
        Normalized emittance
    part : xp.Particles
        Particles to track
    samples : List[int]
        List of sample numbers to track
    _context : xo.Context
        Context object
    outfile : H5pyWriter
        HDF5 file to write to
    """
    backtracker = tracker.get_backtracker()

    part_data = get_particle_norm_data(part, twiss, nemitt, _context, retidx=False)
    outfile.write_data("reference/initial/x_norm", part_data["x_norm"])
    outfile.write_data("reference/initial/px_norm", part_data["px_norm"])
    outfile.write_data("reference/initial/y_norm", part_data["y_norm"])
    outfile.write_data("reference/initial/py_norm", part_data["py_norm"])
    outfile.write_data("reference/initial/zeta_norm", part_data["zeta_norm"])
    outfile.write_data("reference/initial/pzeta_norm", part_data["pzeta_norm"])

    current_t = 0

    samples = sorted(samples)
    for i, t in enumerate(samples):
        print(f"Tracking {t} turns... ({i+1}/{len(samples)})")
        delta_t = t - current_t
        tracker.track(part, num_turns=delta_t)
        current_t = t
        r_part = part.copy()
        backtracker.track(r_part, num_turns=t)

        part_data = get_particle_norm_data(
            r_part, twiss, nemitt, _context, retidx=False
        )

        outfile.write_data(f"reverse/x_norm/{t}", part_data["x_norm"])
        outfile.write_data(f"reverse/px_norm/{t}", part_data["px_norm"])
        outfile.write_data(f"reverse/y_norm/{t}", part_data["y_norm"])
        outfile.write_data(f"reverse/py_norm/{t}", part_data["py_norm"])
        outfile.write_data(f"reverse/zeta_norm/{t}", part_data["zeta_norm"])
        outfile.write_data(f"reverse/pzeta_norm/{t}", part_data["pzeta_norm"])


def track_log_displacement_singles_birkhoff(
    tracker: xt.Tracker,
    twiss: xtw.TwissTable,
    nemitt: Tuple[float, float],
    part: xp.Particles,
    d_part_list: List[xp.Particles],
    d_part_names: List[str],
    initial_displacement: float,
    samples: List[int],
    _context,
    outfile: H5pyWriter,
):
    """Track a set of particles for a given number of turns and write the log displacement to an HDF5 file.

    Parameters
    ----------
    tracker : xt.Tracker
        Tracker object
    twiss : xtw.TwissTable
        Twiss table
    nemitt : Tuple[float, float]
        Normalized emittance, (x, y)
    part : xp.Particles
        Particles to track
    d_part_list : List[xp.Particles]
        List of displaced particles
    d_part_names : List[str]
        List of displaced particle names
    initial_displacement : float
        Initial displacement
    samples : List[int]
        List of sample numbers to track
    _context : xo.Context
        Context object
    outfile : H5pyWriter
        HDF5 file to write to
    """
    n_particles = len(part.x)
    # sort the samples
    samples = sorted(samples)
    birkhoff_list = [birkhoff_weights_context(s, _context) for s in samples]

    log_displacement = [
        [cp.zeros(n_particles) for i in range(len(d_part_list))]
        for j in range(len(samples))
    ]

    pure_log_displacement = [
        [cp.zeros(n_particles) for i in range(len(d_part_list))]
        for j in range(len(samples))
    ]

    part_data = get_particle_norm_data(part, twiss, nemitt, _context, retidx=False)
    outfile.write_data("reference/initial/x_norm", part_data["x_norm"])
    outfile.write_data("reference/initial/px_norm", part_data["px_norm"])
    outfile.write_data("reference/initial/y_norm", part_data["y_norm"])
    outfile.write_data("reference/initial/py_norm", part_data["py_norm"])
    outfile.write_data("reference/initial/zeta_norm", part_data["zeta_norm"])
    outfile.write_data("reference/initial/pzeta_norm", part_data["pzeta_norm"])

    twiss_data, w, w_inv = get_twiss_data(
        twiss, nemitt_x=nemitt[0], nemitt_y=nemitt[1], _context=_context
    )
    norm_part = create_normed_placeholder(part, twiss_data, w_inv, _context)
    norm_d_part_list = [
        create_normed_placeholder(d_part, twiss_data, w_inv, _context)
        for d_part in d_part_list
    ]

    direction_list = [[] for i in range(len(d_part_list))]

    for i, d_part in enumerate(d_part_list):
        part_data = get_particle_norm_data(
            d_part, twiss, nemitt, _context, retidx=False
        )
        outfile.write_data(f"{d_part_names[i]}/initial/x_norm", part_data["x_norm"])
        outfile.write_data(f"{d_part_names[i]}/initial/px_norm", part_data["px_norm"])
        outfile.write_data(f"{d_part_names[i]}/initial/y_norm", part_data["y_norm"])
        outfile.write_data(f"{d_part_names[i]}/initial/py_norm", part_data["py_norm"])
        outfile.write_data(
            f"{d_part_names[i]}/initial/zeta_norm", part_data["zeta_norm"]
        )
        outfile.write_data(
            f"{d_part_names[i]}/initial/pzeta_norm", part_data["pzeta_norm"]
        )

    for time in tqdm(range(1, np.max(samples) + 1)):
        tracker.track(part, num_turns=1)
        for i, d_part in enumerate(d_part_list):
            tracker.track(d_part, num_turns=1)

            d_part, direction_list[i], distance = renormalize_distance(
                part,
                d_part,
                norm_part,
                norm_d_part_list[i],
                twiss_data,
                w,
                w_inv,
                _context,
                initial_displacement,
            )

            for j, sample in enumerate(samples):
                if time <= sample:
                    log_displacement[j][i] += (
                        cp.log10(distance / initial_displacement)
                        * birkhoff_list[j][time - 1]
                    )

                    pure_log_displacement[j][i] += (
                        cp.log10(distance / initial_displacement) / sample
                    )

        if time in samples:
            j = samples.index(time)
            part_data = get_particle_norm_data(
                part, twiss, nemitt, _context, retidx=False
            )
            outfile.write_data(f"reference/x_norm/{time}", part_data["x_norm"])
            outfile.write_data(f"reference/px_norm/{time}", part_data["px_norm"])
            outfile.write_data(f"reference/y_norm/{time}", part_data["y_norm"])
            outfile.write_data(f"reference/py_norm/{time}", part_data["py_norm"])
            outfile.write_data(f"reference/zeta_norm/{time}", part_data["zeta_norm"])
            outfile.write_data(f"reference/pzeta_norm/{time}", part_data["pzeta_norm"])

            for i, (d_part, n_dist) in enumerate(zip(d_part_list, direction_list)):
                part_data = get_particle_norm_data(
                    d_part, twiss, nemitt, _context, retidx=False
                )
                outfile.write_data(
                    f"{d_part_names[i]}/log_disp/{time}",
                    _context.nparray_from_context_array(log_displacement[j][i]),
                )
                outfile.write_data(
                    f"{d_part_names[i]}/log_disp_nobirk/{time}",
                    _context.nparray_from_context_array(pure_log_displacement[j][i]),
                )
                outfile.write_data(
                    f"{d_part_names[i]}/x_norm/{time}", part_data["x_norm"]
                )
                outfile.write_data(
                    f"{d_part_names[i]}/px_norm/{time}", part_data["px_norm"]
                )
                outfile.write_data(
                    f"{d_part_names[i]}/y_norm/{time}", part_data["y_norm"]
                )
                outfile.write_data(
                    f"{d_part_names[i]}/py_norm/{time}", part_data["py_norm"]
                )
                outfile.write_data(
                    f"{d_part_names[i]}/zeta_norm/{time}", part_data["zeta_norm"]
                )
                outfile.write_data(
                    f"{d_part_names[i]}/pzeta_norm/{time}", part_data["pzeta_norm"]
                )

                outfile.write_data(
                    f"{d_part_names[i]}/normed_direction/x_norm/{time}",
                    _context.nparray_from_context_array(n_dist[0]),
                )
                outfile.write_data(
                    f"{d_part_names[i]}/normed_direction/px_norm/{time}",
                    _context.nparray_from_context_array(n_dist[1]),
                )
                outfile.write_data(
                    f"{d_part_names[i]}/normed_direction/y_norm/{time}",
                    _context.nparray_from_context_array(n_dist[2]),
                )
                outfile.write_data(
                    f"{d_part_names[i]}/normed_direction/py_norm/{time}",
                    _context.nparray_from_context_array(n_dist[3]),
                )
                outfile.write_data(
                    f"{d_part_names[i]}/normed_direction/zeta_norm/{time}",
                    _context.nparray_from_context_array(n_dist[4]),
                )
                outfile.write_data(
                    f"{d_part_names[i]}/normed_direction/pzeta_norm/{time}",
                    _context.nparray_from_context_array(n_dist[5]),
                )


def track_tune_birkhoff(
    tracker: xt.Tracker,
    twiss: xtw.TwissTable,
    nemitt: Tuple[float, float],
    part: xp.Particles,
    samples_from: List[int],
    samples_to: List[int],
    _context,
    outfile: H5pyWriter,
):
    """Track particles and save tune evaluated with birkhoff weights.

    Parameters
    ----------
    tracker : xt.Tracker
        Tracker object.
    twiss : xtw.TwissTable
        Twiss table.
    nemitt : Tuple[float, float]
        Normalized emittance.
    part : xp.Particles
        Particles.
    samples_from : List[int]
        List of sample times from which to start evaluating the tune.
    samples_to : List[int]
        List of sample times to which to stop evaluating the tune.
    _context : xo.Context
        Context.
    outfile : H5pyWriter
        Output file.
    """
    assert len(samples_from) == len(samples_to)

    samples_length = [s_to - s_from for s_from, s_to in zip(samples_from, samples_to)]

    n_particles = len(part.x)
    birkhoff_list = [birkhoff_weights_context(s, _context) for s in samples_length]

    part_data = get_particle_norm_data(part, twiss, nemitt, _context, retidx=False)
    outfile.write_data("reference/initial/x_norm", part_data["x_norm"])
    outfile.write_data("reference/initial/px_norm", part_data["px_norm"])
    outfile.write_data("reference/initial/y_norm", part_data["y_norm"])
    outfile.write_data("reference/initial/py_norm", part_data["py_norm"])
    outfile.write_data("reference/initial/zeta_norm", part_data["zeta_norm"])
    outfile.write_data("reference/initial/pzeta_norm", part_data["pzeta_norm"])

    twiss_data, w, w_inv = get_twiss_data(
        twiss, nemitt_x=nemitt[0], nemitt_y=nemitt[1], _context=_context
    )
    norm_part = create_normed_placeholder(part, twiss_data, w_inv, _context)

    angle_1_x = cp.zeros(n_particles)
    angle_1_y = cp.zeros(n_particles)
    angle_2_x = cp.zeros(n_particles)
    angle_2_y = cp.zeros(n_particles)

    sum_birkhoff_x = [cp.zeros(n_particles) for j in range(len(samples_length))]
    sum_birkhoff_y = [cp.zeros(n_particles) for j in range(len(samples_length))]

    norm_part = phys_to_norm(part, norm_part, twiss_data, w_inv)
    angle_1_x = cp.angle(norm_part[0] + 1j * norm_part[1])
    angle_1_y = cp.angle(norm_part[2] + 1j * norm_part[3])
    angle_1_x[angle_1_x < 0] += 2 * cp.pi
    angle_1_y[angle_1_y < 0] += 2 * cp.pi

    for time in tqdm(range(1, np.max(samples_to) + 1)):
        tracker.track(part, num_turns=1)
        norm_part = phys_to_norm(part, norm_part, twiss_data, w_inv)
        angle_2_x = cp.angle(norm_part[0] + 1j * norm_part[1])
        angle_2_y = cp.angle(norm_part[2] + 1j * norm_part[3])
        angle_2_x[angle_2_x < 0] += 2 * cp.pi
        angle_2_y[angle_2_y < 0] += 2 * cp.pi

        delta_angle_x = angle_2_x - angle_1_x
        delta_angle_y = angle_2_y - angle_1_y

        delta_angle_x[delta_angle_x < 0] += 2 * cp.pi
        delta_angle_y[delta_angle_y < 0] += 2 * cp.pi

        for j, (t_from, t_to) in enumerate(zip(samples_from, samples_to)):
            if time > t_from and time <= t_to:
                sum_birkhoff_x[j] += birkhoff_list[j][time - 1 - t_from] * delta_angle_x
                sum_birkhoff_y[j] += birkhoff_list[j][time - 1 - t_from] * delta_angle_y

        angle_1_x = angle_2_x
        angle_1_y = angle_2_y

        if time in samples_to:
            part_data = get_particle_norm_data(
                part, twiss, nemitt, _context, retidx=False
            )
            outfile.write_data(f"reference/x_norm/{time}", part_data["x_norm"])
            outfile.write_data(f"reference/px_norm/{time}", part_data["px_norm"])
            outfile.write_data(f"reference/y_norm/{time}", part_data["y_norm"])
            outfile.write_data(f"reference/py_norm/{time}", part_data["py_norm"])
            outfile.write_data(f"reference/zeta_norm/{time}", part_data["zeta_norm"])
            outfile.write_data(f"reference/pzeta_norm/{time}", part_data["pzeta_norm"])

    for j, (t_from, t_to) in enumerate(zip(samples_from, samples_to)):
        outfile.write_data(
            f"tune_birkhoff_x/{t_from}/{t_to}",
            1 - sum_birkhoff_x[j].get() / (2 * np.pi),
        )
        outfile.write_data(
            f"tune_birkhoff_y/{t_from}/{t_to}",
            1 - sum_birkhoff_y[j].get() / (2 * np.pi),
        )
