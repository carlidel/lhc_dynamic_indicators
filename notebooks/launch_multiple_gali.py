import itertools
import multiprocessing as mp
import os

variables = ["x", "px", "y", "py", "zeta", "delta"]
gali_2 = list(itertools.combinations(variables, 2))
gali_4 = list(itertools.combinations(variables, 4))

commands_to_run = ["python gali_creator.py -l six"]

for vars in gali_2:
    commands_to_run.append(f"python gali_creator.py -l {vars[0]} {vars[1]}")

for vars in gali_4:
    commands_to_run.append(
        f"python gali_creator.py -l {vars[0]} {vars[1]} {vars[2]} {vars[3]}"
    )

# run the commands in parallel with multiprocessing


def run_command(command):
    os.system(command)


pool = mp.Pool(processes=mp.cpu_count())
pool.map(run_command, commands_to_run)
