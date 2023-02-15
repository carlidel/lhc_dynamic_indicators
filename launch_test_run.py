import datetime
import multiprocessing
import os


# launch a python process with multiprocess which also sets the correct gpu to use
def launch_process(cmd, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = multiprocessing.Process(target=os.system, args=(cmd,))
    process.start()
    return process


start_time = datetime.datetime.now()

# gpu_0 = 0
# command_0 = "python run_sim.py --mask configs/mask1.json --tracking configs/quick_tracking.json --kind megno_displacement --output megno_displacement"
# process_0 = launch_process(command_0, gpu_0)

# gpu_1 = 1
# command_1 = "python run_sim.py --mask configs/mask1.json --tracking configs/quick_tracking.json --kind log_displacement_singles --output log_displacement_singles"
# process_1 = launch_process(command_1, gpu_1)

gpu_2 = 2
command_2 = "python run_sim.py --mask configs/mask1.json --tracking configs/quick_tracking.json --kind log_displacement_singles_birkhoff --output log_displacement_singles_birkhoff"
process_2 = launch_process(command_2, gpu_2)

# wait for all processes to finish
print("Waiting for processes to finish")
# process_0.join()
# print("Process 0 finished")
# process_1.join()
# print("Process 1 finished")
process_2.join()
print("Process 2 finished")

end_time = datetime.datetime.now()

# print time in hh:mm:ss
print("Time elapsed: " + str(end_time - start_time))
