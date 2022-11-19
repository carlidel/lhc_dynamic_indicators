import argparse
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


# create parser
parser = argparse.ArgumentParser(description="Create Gali file")
# accept arbitrary number of string arguments
parser.add_argument("-l", "--list", nargs="+", help="variables to use", required=True)

# parse the arguments from standard input
args = parser.parse_args()

# get the list of variables
variables = args.list
print("Variables: ", variables)
if "six" in variables:
    variables = ["x", "px", "y", "py", "zeta", "delta"]
    is_six = True
else:
    is_six = False

if is_six:
    filename = "gali_six.pkl"
else:
    filename = "gali_"
    for s in variables:
        filename += s
        filename += "_"
    filename = filename[:-1] + ".pkl"

print("Creating file: " + filename)

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
    disp_x_log = np.log10(disp_x)
    disp_x_log[~mask] = np.nan
    disp_x_log[np.logical_and(mask, np.isnan(disp_x_log))] = np.nanmax(disp_x_log)

    threshold = find_threshold_density(disp_x_log, where_chaos="higher")
    threshold_list.append(threshold)

    gt = disp_x_log < threshold
    gt_list.append(gt)
    gt = np.array(gt, dtype=float)
    gt[~mask] = np.nan
    gt_plot_list.append(gt)
    gt_raw_data_list.append(disp_x_log)


gali_data_list = []
gali_post_data_list = []
gali_thresh_list = []
gali_guesses_list = []
gali_scores_list = []

for i, ((lattice, extent), zeta) in tqdm(
    enumerate(itertools.product(zip(lattice_list, extent_list), zeta_list))
):
    log_disp_file = h5py.File(
        os.path.join(OUTDIR, f"log_displacement_singles_{lattice}_{zeta}.h5"), "r"
    )

    data_list = []
    post_data_list = []
    thresh_list = []
    guesses_list = []
    scores_list = []

    for t in tqdm(times):
        disp_dict = {}

        disp_dict["x_x"] = log_disp_file[f"disp/x/normed_distance/x/{t}"][:]
        disp_dict["x_px"] = log_disp_file[f"disp/x/normed_distance/px/{t}"][:]
        disp_dict["x_y"] = log_disp_file[f"disp/x/normed_distance/y/{t}"][:]
        disp_dict["x_py"] = log_disp_file[f"disp/x/normed_distance/py/{t}"][:]
        disp_dict["x_zeta"] = log_disp_file[f"disp/x/normed_distance/zeta/{t}"][:]
        disp_dict["x_delta"] = log_disp_file[f"disp/x/normed_distance/delta/{t}"][:]

        disp_dict["px_x"] = log_disp_file[f"disp/px/normed_distance/x/{t}"][:]
        disp_dict["px_px"] = log_disp_file[f"disp/px/normed_distance/px/{t}"][:]
        disp_dict["px_y"] = log_disp_file[f"disp/px/normed_distance/y/{t}"][:]
        disp_dict["px_py"] = log_disp_file[f"disp/px/normed_distance/py/{t}"][:]
        disp_dict["px_zeta"] = log_disp_file[f"disp/px/normed_distance/zeta/{t}"][:]
        disp_dict["px_delta"] = log_disp_file[f"disp/px/normed_distance/delta/{t}"][:]

        disp_dict["y_x"] = log_disp_file[f"disp/y/normed_distance/x/{t}"][:]
        disp_dict["y_px"] = log_disp_file[f"disp/y/normed_distance/px/{t}"][:]
        disp_dict["y_y"] = log_disp_file[f"disp/y/normed_distance/y/{t}"][:]
        disp_dict["y_py"] = log_disp_file[f"disp/y/normed_distance/py/{t}"][:]
        disp_dict["y_zeta"] = log_disp_file[f"disp/y/normed_distance/zeta/{t}"][:]
        disp_dict["y_delta"] = log_disp_file[f"disp/y/normed_distance/delta/{t}"][:]

        disp_dict["py_x"] = log_disp_file[f"disp/py/normed_distance/x/{t}"][:]
        disp_dict["py_px"] = log_disp_file[f"disp/py/normed_distance/px/{t}"][:]
        disp_dict["py_y"] = log_disp_file[f"disp/py/normed_distance/y/{t}"][:]
        disp_dict["py_py"] = log_disp_file[f"disp/py/normed_distance/py/{t}"][:]
        disp_dict["py_zeta"] = log_disp_file[f"disp/py/normed_distance/zeta/{t}"][:]
        disp_dict["py_delta"] = log_disp_file[f"disp/py/normed_distance/delta/{t}"][:]

        disp_dict["zeta_x"] = log_disp_file[f"disp/zeta/normed_distance/x/{t}"][:]
        disp_dict["zeta_px"] = log_disp_file[f"disp/zeta/normed_distance/px/{t}"][:]
        disp_dict["zeta_y"] = log_disp_file[f"disp/zeta/normed_distance/y/{t}"][:]
        disp_dict["zeta_py"] = log_disp_file[f"disp/zeta/normed_distance/py/{t}"][:]
        disp_dict["zeta_zeta"] = log_disp_file[f"disp/zeta/normed_distance/zeta/{t}"][:]
        disp_dict["zeta_delta"] = log_disp_file[f"disp/zeta/normed_distance/delta/{t}"][
            :
        ]

        disp_dict["delta_x"] = log_disp_file[f"disp/delta/normed_distance/x/{t}"][:]
        disp_dict["delta_px"] = log_disp_file[f"disp/delta/normed_distance/px/{t}"][:]
        disp_dict["delta_y"] = log_disp_file[f"disp/delta/normed_distance/y/{t}"][:]
        disp_dict["delta_py"] = log_disp_file[f"disp/delta/normed_distance/py/{t}"][:]
        disp_dict["delta_zeta"] = log_disp_file[f"disp/delta/normed_distance/zeta/{t}"][
            :
        ]
        disp_dict["delta_delta"] = log_disp_file[
            f"disp/delta/normed_distance/delta/{t}"
        ][:]

        gali_matrix = np.asarray(
            [[disp_dict[f"{var1}_{var2}"] for var2 in variables] for var1 in variables]
        )

        disp = gali(gali_matrix)
        data_list.append(disp)

        disp[~mask_list[i]] = np.nan
        if np.count_nonzero(disp[mask_list[i]] > 0) == 0:
            disp[mask_list[i]] = 1e-14

        disp = np.log10(disp)
        disp[np.logical_and(mask_list[i], np.isnan(disp))] = np.nanmin(disp)
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

    gali_data_list.append(data_list)
    gali_post_data_list.append(post_data_list)
    gali_thresh_list.append(thresh_list)
    gali_guesses_list.append(guesses_list)
    gali_scores_list.append(scores_list)

with open(os.path.join("gali_files/", filename), "wb") as f:
    pickle.dump(
        [
            gali_data_list,
            gali_post_data_list,
            gali_thresh_list,
            gali_guesses_list,
            gali_scores_list,
        ],
        f,
    )
