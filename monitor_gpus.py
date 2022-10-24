# simple script
# runs nvidia-smi and if a GPU's memory usage goes below 10% for 5 minutes
# print a message to the console

import subprocess
import time

# how many seconds to wait between checks
wait_seconds = 1

# how much memory to wait for before printing a message
min_memory_MB = (1024*11) * 0.1

# call nvidia-smi and return the output
def get_nvidia_smi():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])

# get the memory usage for each GPU
def get_memory_usage():
    return [int(x) for x in get_nvidia_smi().decode("utf-8").strip().split("\n")]

print("Monitoring GPUs...")

# loop forever
while True:
    try:
        # get the memory usage for each GPU
        memory_usage = get_memory_usage() # returns list[int]
        
        # check if any GPU is below the minimum memory usage
        if any(x < min_memory_MB for x in memory_usage):
            # print a message
            print("GPU memory usage below 10% for GPUs {}".format([i for i, x in enumerate(memory_usage) if x < min_memory_MB]))
            # play BEL
            print("\a", end="")
        
        # wait a bit then print newline
        time.sleep(wait_seconds / 2)
        print("")
        # (this will make the output animate if GPU usage is low to catch your eye)
        
        # wait the remaining time
        time.sleep(wait_seconds/2)
    except KeyboardInterrupt:
        print("Pausing monitoring for 2 minutes...")
        try:
            time.sleep(120)
        except KeyboardInterrupt:
            print("Pausing monitoring for 10 minutes...")
            try:
                time.sleep(600)
            except KeyboardInterrupt:
                print("Exiting...")
                exit()
        print("Resuming monitoring...")
        
