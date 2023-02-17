import itertools

mask_file_list = [
    "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/23_02_15_job_files/configs/mask_worst.json",
    "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/23_02_15_job_files/configs/mask_best.json",
]

tracking_config_path_1e5 = "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/23_02_15_job_files/configs/tracking_1e5.json"
tracking_config_path_1e6 = "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/23_02_15_job_files/configs/tracking_1e6.json"

zeta_list = ["min", "max", "avg"]

tracking_options_list = [
    ("stability", tracking_config_path_1e6),
    ("log_displacement", tracking_config_path_1e5),
    ("rem", tracking_config_path_1e5),
    ("tune_birkhoff", tracking_config_path_1e5),
]


for i, (mask, zeta, (track_option, track_path)) in enumerate(
    itertools.product(mask_file_list, zeta_list, tracking_options_list)
):
    with open("configs/all_jobs_no_tune.txt", "a" if i > 0 else "w") as f:
        f.write(f"{mask}, {track_path}, {track_option}, {zeta}\n")
