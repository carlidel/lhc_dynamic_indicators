import itertools

mask_file_list = [
    "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/configs/mask1.json",
    "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/configs/mask2.json",
    "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/configs/mask3.json",
]

tracking_config_path = "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/configs/tracking_htcondor.json"

zeta_list = ["min", "max", "avg"]

tracking_options_list = [
    "stability",
    "log_displacement_singles",
    "log_displacement_singles_birkhoff",
    "megno_displacement",
    "rem",
    "tune_birkhoff",
]


for i, (mask, zeta, track) in enumerate(
    itertools.product(mask_file_list, zeta_list, tracking_options_list)
):
    with open("configs/all_jobs_no_tune.txt", "a" if i > 0 else "w") as f:
        f.write(f"{mask}, {tracking_config_path}, {track}, {zeta}\n")
