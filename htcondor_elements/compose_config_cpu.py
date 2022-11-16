import itertools

mask_file_list = [
    "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/configs/mask1_afs.json",
    "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/configs/mask2_afs.json",
    "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/configs/mask3_afs.json",
]

tracking_config_path = "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/configs/tracking_htcondor.json"

zeta_list = ["min", "max", "avg"]

tracking_options_list = [
    "tune_birkhoff",
]

samples = 300 * 300
batch_size = 100
sample_num = samples // batch_size

for i, (mask, zeta, track, s_num) in enumerate(
    itertools.product(
        mask_file_list, zeta_list, tracking_options_list, range(sample_num)
    )
):
    with open("configs_cpu/all_jobs_no_tune.txt", "a" if i > 0 else "w") as f:
        f.write(
            f"{mask}, {tracking_config_path}, {track}, {zeta}, {batch_size}, {s_num}, {track}_{s_num}\n"
        )
