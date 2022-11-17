import matplotlib.patches as patches
import numpy as np
import scipy as sp
from KDEpy import FFTKDE
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity


def filter_data(mask, data, data_fix=True):
    if data_fix:
        data[np.isinf(data)] = np.nanmax(data)
        data[np.isnan(data)] = np.nanmax(data)
    data = data[mask]
    return data


def cover_extreme_outliers(data, m=10):
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    data_min = data_mean - m * data_std
    data_max = data_mean + m * data_std
    data_mask = np.logical_and(data >= data_min, data <= data_max)
    data[~data_mask] = np.nan
    return data


def find_threshold(data):
    data = data[~np.isnan(data) & ~np.isinf(data)]
    labels = KMeans(n_clusters=2, random_state=42).fit_predict(data.reshape(-1, 1))
    # labels = DBSCAN(eps=0.1).fit_predict(data.reshape(-1, 1))
    # print(np.unique(labels))
    max_1 = np.max(data[labels == 0])
    max_2 = np.max(data[labels == 1])
    min_1 = np.min(data[labels == 0])
    min_2 = np.min(data[labels == 1])
    if max_1 > max_2:
        thresh = (max_2 + min_1) / 2
    else:
        thresh = (max_1 + min_2) / 2

    return thresh


def find_threshold_density(
    data, bandwidth_divider=50.0, sampling=1000, where_chaos="higher"
):
    data = data[~np.isnan(data) & ~np.isinf(data)]
    max_val = np.nanmax(data)
    min_val = np.nanmin(data)
    bandwidth = (max_val - min_val) / bandwidth_divider
    x_grid, density = FFTKDE(kernel="gaussian", bw=bandwidth).fit(data).evaluate()
    # kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(data.reshape(-1, 1))
    # x_grid = np.linspace(min_val, max_val, sampling)
    # log_dens = kde.score_samples(x_grid.reshape(-1, 1))
    # density = np.exp(log_dens)
    minima, maxima = (
        argrelextrema(density, np.less)[0],
        argrelextrema(density, np.greater)[0],
    )

    # sort togheter minima and maxima
    values = [("min", x_grid[m], density[m]) for m in minima] + [
        ("max", x_grid[m], density[m]) for m in maxima
    ]
    values = sorted(values, key=lambda x: x[1])

    # find the maximum x[2] in values
    max_val_idx = np.argmax([x[2] for x in values])

    # apply strategy
    if where_chaos == "higher":
        if max_val_idx + 2 < len(values):
            thresh = values[max_val_idx + 1][1]
        else:
            thresh = max_val
    elif where_chaos == "lower":
        if max_val_idx - 2 >= 0:
            thresh = values[max_val_idx - 1][1]
        else:
            thresh = min_val
    else:
        raise ValueError("where_chaos not recognized")

    return thresh


def find_threshold_smart(data, haircut=0.002, starting_value=10, where_chaos="higher"):
    data = data[~np.isnan(data)]
    data = data[~np.isinf(data)]

    ### data haircutting
    data = np.sort(data)
    data = data[int(haircut * len(data)) : -int(haircut * len(data))]

    bandwidth_n_list = np.logspace(np.log10(starting_value), 3, 100)

    max_val = np.nanmax(data[~np.isnan(data) & ~np.isinf(data)])
    min_val = np.nanmin(data[~np.isnan(data) & ~np.isinf(data)])

    if max_val == min_val:
        return max_val

    threshold_evolution = []
    # lines_list = []

    for bandwidth_n in bandwidth_n_list:
        x_grid, density = (
            FFTKDE(kernel="gaussian", bw=(max_val - min_val) / bandwidth_n)
            .fit(data)
            .evaluate()
        )
        # lines_list.append([x_grid, density])
        minima, maxima = (
            argrelextrema(density, np.less)[0],
            argrelextrema(density, np.greater)[0],
        )
        if len(maxima) <= 1:
            threshold_evolution.append(np.nan)
            continue

        idx_maxima = [(x_grid[i], density[i], n) for n, i in enumerate(maxima)]
        idx_maxima = sorted(idx_maxima, key=lambda x: x[1], reverse=True)

        maxima_1 = idx_maxima[0]
        maxima_2 = idx_maxima[1]
        high_x = np.max([maxima_1[0], maxima_2[0]])
        low_x = np.min([maxima_1[0], maxima_2[0]])

        filtered_minima = [
            (x_grid[i], density[i], n)
            for n, i in enumerate(minima)
            if x_grid[i] > low_x and x_grid[i] < high_x
        ]

        if len(filtered_minima) == 0:
            print("What the actual fuck?")
            threshold_evolution.append((high_x + low_x) / 2)
            continue

        min_minima = sorted(filtered_minima, key=lambda x: x[1])[0]
        threshold_evolution.append(min_minima[0])

    threshold_diff = np.abs(np.diff(threshold_evolution))
    max_diff = np.nanmax(threshold_diff)
    # find the first index where the difference is larger than 20% of the maximum difference
    idx = np.nanargmax(threshold_diff > 0.2 * max_diff)
    the_threshold = threshold_evolution[idx]
    if where_chaos == "gali":
        the_threshold = np.max([the_threshold, -30])
    return the_threshold


def find_threshold_smart_v2(
    data, haircut=0.002, starting_value=10, where_chaos="higher"
):
    data = data[~np.isnan(data)]
    data = data[~np.isinf(data)]

    ### data haircutting
    data = np.sort(data)
    data = data[int(haircut * len(data)) : -int(haircut * len(data))]

    bandwidth_n_list = np.logspace(np.log10(starting_value), 3, 100)

    max_val = np.nanmax(data[~np.isnan(data) & ~np.isinf(data)])
    min_val = np.nanmin(data[~np.isnan(data) & ~np.isinf(data)])

    if max_val == min_val:
        return max_val

    threshold_evolution = []
    # lines_list = []

    for bandwidth_n in bandwidth_n_list:
        x_grid, density = (
            FFTKDE(kernel="gaussian", bw=(max_val - min_val) / bandwidth_n)
            .fit(data)
            .evaluate()
        )
        # lines_list.append([x_grid, density])
        minima, maxima = (
            argrelextrema(density, np.less)[0],
            argrelextrema(density, np.greater)[0],
        )
        if len(maxima) <= 2:
            threshold_evolution.append(np.nan)
            continue

        idx_maxima = [(x_grid[i], density[i], n) for n, i in enumerate(maxima)]
        idx_maxima = sorted(idx_maxima, key=lambda x: x[1], reverse=True)

        maxima_1 = idx_maxima[0]
        maxima_2 = idx_maxima[1]
        maxima_3 = idx_maxima[2]
        sorted_x = sorted([maxima_1[0], maxima_2[0], maxima_3[0]])
        high_x = sorted_x[2]
        low_x = sorted_x[0]
        mid_x = sorted_x[1]

        filtered_minima = [
            (x_grid[i], density[i], n)
            for n, i in enumerate(minima)
            if x_grid[i] > mid_x and x_grid[i] < high_x
        ]

        if len(filtered_minima) == 0:
            print("What the actual fuck?")
            threshold_evolution.append((high_x + low_x) / 2)
            continue

        min_minima = sorted(filtered_minima, key=lambda x: x[1])[0]
        threshold_evolution.append(min_minima[0])

    threshold_diff = np.abs(np.diff(threshold_evolution))
    max_diff = np.nanmax(threshold_diff)
    # find the first index where the difference is larger than 20% of the maximum difference
    idx = np.nanargmax(threshold_diff > 0.1 * max_diff)
    the_threshold = threshold_evolution[idx]
    return the_threshold


def classify_data(ground_truth: np.ndarray, guess: np.ndarray):
    total = ground_truth.size
    true_positive = np.sum(ground_truth & guess)
    true_negative = np.sum(~ground_truth & ~guess)
    false_positive = np.sum(~ground_truth & guess)
    false_negative = np.sum(ground_truth & ~guess)

    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    return dict(
        total=total,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def find_best_threshold(
    ground_truth: np.ndarray, data: np.ndarray, n_samples=100, operator="greater"
):
    max_val = np.nanmax(data[~np.isnan(data) & ~np.isinf(data)])
    min_val = np.nanmin(data[~np.isnan(data) & ~np.isinf(data)])
    thresh_list = np.linspace(min_val, max_val, n_samples)
    accuracy_list = []
    for thresh in thresh_list:
        if operator == "greater":
            guess = data > thresh
        elif operator == "less":
            guess = data < thresh
        else:
            raise ValueError("Operator not recognized")
        accuracy_list.append(classify_data(ground_truth, guess)["accuracy"])
    best_thresh = thresh_list[np.argmax(accuracy_list)]
    return best_thresh


def get_extents(data_list, cover_m=10, clean_data=True):
    val_min = np.nan
    val_max = np.nan
    for d in data_list:
        if clean_data:
            d = d[~np.isnan(d) & ~np.isinf(d)]
        if cover_m is not None:
            d = cover_extreme_outliers(d, m=cover_m)
        val_min = np.nanmin([val_min, np.nanmin(d)])
        val_max = np.nanmax([val_max, np.nanmax(d)])
    return val_min, val_max


def compose_count_map(data_list, val_min, val_max, nbins=50, density=True):
    count_map = np.zeros((len(data_list), nbins))
    for i, d in enumerate(data_list):
        count, bins = np.histogram(
            d, bins=nbins, density=density, range=(val_min, val_max)
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        count_map[i, :] = count
    return count_map, bin_centers


def make_colormap(
    ax_colormap, count_map, data_list, time_list, val_min, val_max, patches_list=None
):
    mappable = ax_colormap.imshow(
        np.log10(count_map),
        aspect="auto",
        origin="lower",
        extent=(val_min, val_max, 0, len(data_list)),
    )
    ax_colormap.set_yticks(np.arange(len(time_list)) + 0.5)
    ax_colormap.set_yticklabels([f"$10^{int(np.log10(t))}$" for t in time_list])

    if patches_list is not None:
        d_rect_x = (val_max - val_min) * 0.005
        d_rect_y = 0.01
        for p in patches_list:
            rect = patches.Rectangle(
                (val_min + d_rect_x, p + d_rect_y),
                val_max - val_min - d_rect_x * 2,
                1 - d_rect_y * 2,
                linewidth=3,
                edgecolor="r",
                facecolor="none",
            )
            ax_colormap.add_patch(rect)

    return ax_colormap, mappable
