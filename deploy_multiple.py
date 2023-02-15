import argparse
import datetime
import multiprocessing
import os

from tqdm import tqdm


# launch a python process with multiprocess which also sets the correct gpu to use
def launch_process(cmd, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = multiprocessing.Process(target=os.system, args=(cmd,))
    process.start()
    return process


# create parser
parser = argparse.ArgumentParser(
    description="Launch multiple simulations in series on a specific GPU"
)
parser.add_argument("--gpu", type=int, help="GPU to use")
parser.add_argument("--command-list", type=str, help="List of commands to run")

# parse arguments
args = parser.parse_args()

# get gpu
gpu = args.gpu
# get command list
command_list = args.command_list
with open(command_list, "r") as f:
    commands = f.readlines()

global_time_start = datetime.datetime.now()

# launch processes in series
for cmd in tqdm(commands):
    cmd = cmd.strip()
    print(f"Launching command: {cmd}")
    time_start = datetime.datetime.now()
    process = launch_process(cmd, gpu)
    process.join()
    time_end = datetime.datetime.now()
    print(f"Time elapsed: {time_end - time_start}")

global_time_end = datetime.datetime.now()

# print time in hh:mm:ss
print(f"Time elapsed: {global_time_end - global_time_start}")
print("Command list:")
for cmd in commands:
    print(cmd)
