import itertools
import os
import pickle
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from clustering_scripts import *
from tqdm import tqdm


def simple_gali(gali_matrix):
    if np.any(np.isnan(gali_matrix)):
        return np.nan
    else:
        _, s, _ = np.linalg.svd(gali_matrix)
        return np.prod(s)


def gali(gali_matrix):
    gali_matrix = np.transpose(gali_matrix, (2, 0, 1))
    gali = []
    for m in gali_matrix:
        gali.append(simple_gali(m))
    gali = np.asarray(gali)
    return gali


OUTDIR = "/home/HPC/camontan/lhc_dynamic_indicators/out/htcondor_out"
TUNEDIR = "/home/HPC/camontan/lhc_dynamic_indicators/out/merged_tune"
IMGDIR = "/home/HPC/camontan/lhc_dynamic_indicators/img/paper_tier/"

lattice_list = ["b1_worst", "b1_avg", "b1_best", "b2_worst", "b2_avg", "b2_best"]
lattice_list = ["b1_worst", "b1_avg", "b1_best"]
extent_list = [
    np.array([0.0, 0.00125, 0.0, 0.00125]),
    np.array([0.0, 0.00130, 0.0, 0.00130]),
    np.array([0.0, 0.00140, 0.0, 0.00140]),
]
lattice_name_list = ["Worst", "Median", "Best"]
zeta_list = ["zeta_min", "zeta_avg", "zeta_max"]
zeta_name_list = ["0.0", "0.15", "0.3"]


l_list = []
l_name_list = []
e_list = []
z_list = []
z_name_list = []
stability_list = []
mask_list = []

gt_raw_data_list = []
threshold_list = []
gt_list = []
gt_plot_list = []

for (lattice, extent, l_name), (zeta, z_name) in itertools.product(
    zip(lattice_list, extent_list, lattice_name_list), zip(zeta_list, zeta_name_list)
):
    l_list.append(lattice)
    l_name_list.append(l_name)
    e_list.append(extent)
    z_list.append(zeta)
    z_name_list.append(z_name)
    stability_file = h5py.File(
        os.path.join(OUTDIR, f"stability_{lattice}_{zeta}.h5"), "r"
    )
    log_disp_file = h5py.File(
        os.path.join(OUTDIR, f"log_displacement_singles_{lattice}_{zeta}.h5"), "r"
    )
    t = 100000
    stability = stability_file["stability"][:]
    stability_list.append(stability)
    mask = np.log10(stability) == 5
    mask_list.append(mask)

    times = np.array(sorted(int(j) for j in log_disp_file[f"disp/x/log_disp"].keys()))

    disp_x = log_disp_file[f"disp/x/log_disp/{t}"][:]
    threshold = find_threshold_density(np.log10(disp_x[mask]), where_chaos="higher")
    threshold_list.append(threshold)

    gt = np.log10(disp_x) < threshold
    gt_list.append(gt)
    gt = np.array(gt, dtype=float)
    gt[~mask] = np.nan
    gt_plot_list.append(gt)
    gt_raw_data_list.append(np.log10(disp_x))


gali_x_px_data_list = []
gali_y_py_data_list = []
gali_zeta_delta_data_list = []
gali_x_y_data_list = []
gali_px_py_data_list = []
gali_6_data_list = []

gali_x_px_post_data_list = []
gali_y_py_post_data_list = []
gali_zeta_delta_post_data_list = []
gali_x_y_post_data_list = []
gali_px_py_post_data_list = []
gali_6_post_data_list = []

gali_x_px_thresh_list = []
gali_y_py_thresh_list = []
gali_zeta_delta_thresh_list = []
gali_x_y_thresh_list = []
gali_px_py_thresh_list = []
gali_6_thresh_list = []

gali_x_px_guesses_list = []
gali_y_py_guesses_list = []
gali_zeta_delta_guesses_list = []
gali_x_y_guesses_list = []
gali_px_py_guesses_list = []
gali_6_guesses_list = []

gali_x_px_scores_list = []
gali_y_py_scores_list = []
gali_zeta_delta_scores_list = []
gali_x_y_scores_list = []
gali_px_py_scores_list = []
gali_6_scores_list = []

for i, ((lattice, extent), zeta) in tqdm(
    enumerate(itertools.product(zip(lattice_list, extent_list), zeta_list))
):
    log_disp_file = h5py.File(
        os.path.join(OUTDIR, f"log_displacement_singles_{lattice}_{zeta}.h5"), "r"
    )
    for q in tqdm(range(6)):

        data_list = []
        post_data_list = []
        thresh_list = []
        guesses_list = []
        scores_list = []
        for t in tqdm(times):
            disp_x_x = log_disp_file[f"disp/x/normed_distance/x/{t}"][:]
            disp_x_px = log_disp_file[f"disp/x/normed_distance/px/{t}"][:]
            disp_x_y = log_disp_file[f"disp/x/normed_distance/y/{t}"][:]
            disp_x_py = log_disp_file[f"disp/x/normed_distance/py/{t}"][:]
            disp_x_zeta = log_disp_file[f"disp/x/normed_distance/zeta/{t}"][:]
            disp_x_delta = log_disp_file[f"disp/x/normed_distance/delta/{t}"][:]

            disp_px_x = log_disp_file[f"disp/px/normed_distance/x/{t}"][:]
            disp_px_px = log_disp_file[f"disp/px/normed_distance/px/{t}"][:]
            disp_px_y = log_disp_file[f"disp/px/normed_distance/y/{t}"][:]
            disp_px_py = log_disp_file[f"disp/px/normed_distance/py/{t}"][:]
            disp_px_zeta = log_disp_file[f"disp/px/normed_distance/zeta/{t}"][:]
            disp_px_delta = log_disp_file[f"disp/px/normed_distance/delta/{t}"][:]

            disp_y_x = log_disp_file[f"disp/y/normed_distance/x/{t}"][:]
            disp_y_px = log_disp_file[f"disp/y/normed_distance/px/{t}"][:]
            disp_y_y = log_disp_file[f"disp/y/normed_distance/y/{t}"][:]
            disp_y_py = log_disp_file[f"disp/y/normed_distance/py/{t}"][:]
            disp_y_zeta = log_disp_file[f"disp/y/normed_distance/zeta/{t}"][:]
            disp_y_delta = log_disp_file[f"disp/y/normed_distance/delta/{t}"][:]

            disp_py_x = log_disp_file[f"disp/py/normed_distance/x/{t}"][:]
            disp_py_px = log_disp_file[f"disp/py/normed_distance/px/{t}"][:]
            disp_py_y = log_disp_file[f"disp/py/normed_distance/y/{t}"][:]
            disp_py_py = log_disp_file[f"disp/py/normed_distance/py/{t}"][:]
            disp_py_zeta = log_disp_file[f"disp/py/normed_distance/zeta/{t}"][:]
            disp_py_delta = log_disp_file[f"disp/py/normed_distance/delta/{t}"][:]

            disp_zeta_x = log_disp_file[f"disp/zeta/normed_distance/x/{t}"][:]
            disp_zeta_px = log_disp_file[f"disp/zeta/normed_distance/px/{t}"][:]
            disp_zeta_y = log_disp_file[f"disp/zeta/normed_distance/y/{t}"][:]
            disp_zeta_py = log_disp_file[f"disp/zeta/normed_distance/py/{t}"][:]
            disp_zeta_zeta = log_disp_file[f"disp/zeta/normed_distance/zeta/{t}"][:]
            disp_zeta_delta = log_disp_file[f"disp/zeta/normed_distance/delta/{t}"][:]

            disp_delta_x = log_disp_file[f"disp/delta/normed_distance/x/{t}"][:]
            disp_delta_px = log_disp_file[f"disp/delta/normed_distance/px/{t}"][:]
            disp_delta_y = log_disp_file[f"disp/delta/normed_distance/y/{t}"][:]
            disp_delta_py = log_disp_file[f"disp/delta/normed_distance/py/{t}"][:]
            disp_delta_zeta = log_disp_file[f"disp/delta/normed_distance/zeta/{t}"][:]
            disp_delta_delta = log_disp_file[f"disp/delta/normed_distance/delta/{t}"][:]
            if q == 0:
                gali_matrix = np.asarray(
                    [
                        [disp_x_x, disp_x_px],
                        [disp_px_x, disp_px_px],
                    ]
                )
            elif q == 1:
                gali_matrix = np.asarray(
                    [
                        [disp_y_y, disp_y_py],
                        [disp_py_y, disp_py_py],
                    ]
                )
            elif q == 2:
                gali_matrix = np.asarray(
                    [
                        [disp_zeta_zeta, disp_zeta_delta],
                        [disp_delta_zeta, disp_delta_delta],
                    ]
                )
            elif q == 3:
                gali_matrix = np.asarray(
                    [
                        [disp_x_x, disp_x_y],
                        [disp_y_x, disp_y_y],
                    ]
                )
            elif q == 4:
                gali_matrix = np.asarray(
                    [
                        [disp_px_px, disp_px_py],
                        [disp_py_px, disp_py_py],
                    ]
                )
            elif q == 5:
                gali_matrix = np.asarray(
                    [
                        [
                            disp_x_x,
                            disp_x_px,
                            disp_x_y,
                            disp_x_py,
                            disp_x_zeta,
                            disp_x_delta,
                        ],
                        [
                            disp_px_x,
                            disp_px_px,
                            disp_px_y,
                            disp_px_py,
                            disp_px_zeta,
                            disp_px_delta,
                        ],
                        [
                            disp_y_x,
                            disp_y_px,
                            disp_y_y,
                            disp_y_py,
                            disp_y_zeta,
                            disp_y_delta,
                        ],
                        [
                            disp_py_x,
                            disp_py_px,
                            disp_py_y,
                            disp_py_py,
                            disp_py_zeta,
                            disp_py_delta,
                        ],
                        [
                            disp_zeta_x,
                            disp_zeta_px,
                            disp_zeta_y,
                            disp_zeta_py,
                            disp_zeta_zeta,
                            disp_zeta_delta,
                        ],
                        [
                            disp_delta_x,
                            disp_delta_px,
                            disp_delta_y,
                            disp_delta_py,
                            disp_delta_zeta,
                            disp_delta_delta,
                        ],
                    ]
                )

            disp = gali(gali_matrix)
            data_list.append(disp)

            disp[~mask_list[i]] = np.nan
            if np.count_nonzero(disp[mask_list[i]] > 0) == 0:
                disp[mask_list[i]] = 1e-14

            disp = np.log10(disp / t)
            disp[np.isinf(disp)] = np.nan
            disp[np.logical_and(mask_list[i], np.isnan(disp))] = np.nanmin(disp)

            post_data_list.append(disp)

            try:
                threshold = find_threshold_smart(disp[mask_list[i]], where_chaos="gali")
            except:
                threshold = 1e-13
            thresh_list.append(threshold)

            guess = disp < threshold
            score = classify_data(gt_list[i][mask_list[i]], guess[mask_list[i]])
            guess = np.asarray(guess, dtype=float)
            guess[~mask_list[i]] = np.nan
            guesses_list.append(guess)
            scores_list.append(score)

        if q == 0:
            gali_x_px_data_list.append(data_list)
            gali_x_px_post_data_list.append(post_data_list)
            gali_x_px_thresh_list.append(thresh_list)
            gali_x_px_guesses_list.append(guesses_list)
            gali_x_px_scores_list.append(scores_list)
        elif q == 1:
            gali_y_py_data_list.append(data_list)
            gali_y_py_post_data_list.append(post_data_list)
            gali_y_py_thresh_list.append(thresh_list)
            gali_y_py_guesses_list.append(guesses_list)
            gali_y_py_scores_list.append(scores_list)
        elif q == 2:
            gali_zeta_delta_data_list.append(data_list)
            gali_zeta_delta_post_data_list.append(post_data_list)
            gali_zeta_delta_thresh_list.append(thresh_list)
            gali_zeta_delta_guesses_list.append(guesses_list)
            gali_zeta_delta_scores_list.append(scores_list)
        elif q == 3:
            gali_x_y_data_list.append(data_list)
            gali_x_y_post_data_list.append(post_data_list)
            gali_x_y_thresh_list.append(thresh_list)
            gali_x_y_guesses_list.append(guesses_list)
            gali_x_y_scores_list.append(scores_list)
        elif q == 4:
            gali_px_py_data_list.append(data_list)
            gali_px_py_post_data_list.append(post_data_list)
            gali_px_py_thresh_list.append(thresh_list)
            gali_px_py_guesses_list.append(guesses_list)
            gali_px_py_scores_list.append(scores_list)
        elif q == 5:
            gali_6_data_list.append(data_list)
            gali_6_post_data_list.append(post_data_list)
            gali_6_thresh_list.append(thresh_list)
            gali_6_guesses_list.append(guesses_list)
            gali_6_scores_list.append(scores_list)


with open("gali.pkl", "wb") as f:
    pickle.dump(
        {
            "gali_x_px_data_list": gali_x_px_data_list,
            "gali_x_px_post_data_list": gali_x_px_post_data_list,
            "gali_x_px_thresh_list": gali_x_px_thresh_list,
            "gali_x_px_guesses_list": gali_x_px_guesses_list,
            "gali_x_px_scores_list": gali_x_px_scores_list,
            "gali_y_py_data_list": gali_y_py_data_list,
            "gali_y_py_post_data_list": gali_y_py_post_data_list,
            "gali_y_py_thresh_list": gali_y_py_thresh_list,
            "gali_y_py_guesses_list": gali_y_py_guesses_list,
            "gali_y_py_scores_list": gali_y_py_scores_list,
            "gali_zeta_delta_data_list": gali_zeta_delta_data_list,
            "gali_zeta_delta_post_data_list": gali_zeta_delta_post_data_list,
            "gali_zeta_delta_thresh_list": gali_zeta_delta_thresh_list,
            "gali_zeta_delta_guesses_list": gali_zeta_delta_guesses_list,
            "gali_zeta_delta_scores_list": gali_zeta_delta_scores_list,
            "gali_x_y_data_list": gali_x_y_data_list,
            "gali_x_y_post_data_list": gali_x_y_post_data_list,
            "gali_x_y_thresh_list": gali_x_y_thresh_list,
            "gali_x_y_guesses_list": gali_x_y_guesses_list,
            "gali_x_y_scores_list": gali_x_y_scores_list,
            "gali_px_py_data_list": gali_px_py_data_list,
            "gali_px_py_post_data_list": gali_px_py_post_data_list,
            "gali_px_py_thresh_list": gali_px_py_thresh_list,
            "gali_px_py_guesses_list": gali_px_py_guesses_list,
            "gali_px_py_scores_list": gali_px_py_scores_list,
            "gali_6_data_list": gali_6_data_list,
            "gali_6_post_data_list": gali_6_post_data_list,
            "gali_6_thresh_list": gali_6_thresh_list,
            "gali_6_guesses_list": gali_6_guesses_list,
            "gali_6_scores_list": gali_6_scores_list,
        },
        f,
    )
