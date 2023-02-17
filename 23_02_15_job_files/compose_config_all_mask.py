import itertools
import json
import os

mask_files = os.listdir("../masks")

base_mask_file = "./configs_all_mask/mask_base.json"

with open(base_mask_file, "r") as f:
    base_mask_config = json.load(f)

final_config_list = []

for i, mask_file in enumerate(mask_files):
    beam_name = mask_file.split("_")[1]
    seed_number = mask_file.split("_")[-1].split(".")[0]

    base_mask_config["name"] = f"{beam_name}_s_{seed_number}"
    base_mask_config["filename"] = mask_file

    with open(f"./configs_all_mask/mask_{i}.json", "w") as f:
        json.dump(base_mask_config, f, indent=4)

    final_config_list.append(f"/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/23_02_15_job_files/configs_all_mask/mask_{i}.json")

tracking_config_path_1e6 = "/afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/23_02_15_job_files/configs_all_mask/tracking_1e6.json"

zeta_list = ["min", "max", "avg"]

tracking_options_list = [
    ("stability", tracking_config_path_1e6)
]


for i, (mask, zeta, (track_option, track_path)) in enumerate(
    itertools.product(final_config_list, zeta_list, tracking_options_list)
):
    with open("configs_all_mask/all_jobs_no_tune.txt", "a" if i > 0 else "w") as f:
        f.write(f"{mask}, {track_path}, {track_option}, {zeta}\n")
