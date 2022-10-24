import os
from tqdm import tqdm

checkpoints = [
    "logs/vae_run_03_day5/checkpoints/weights_2875.ckpt",
    "logs/vae_run_03/checkpoints/weights_2500.ckpt",
]
_ = [
    "logs/vae_run_03_day2_dis/checkpoints/weights_7290.ckpt",
    "logs/vae_run_03/checkpoints/weights_4000.ckpt",
    "pretrained/sd-v1-4.ckpt",
    "pretrained/kl-f8-anime.ckpt",
]
seed = 1  # default: 1, "-1" is random
output_directory = r"C:\Users\DefAcc\Downloads\IMAGESERVER\static\vae_03"

def maybe_strip_datetime(s):
    """If s is a string of the form "2022-09-24T11-39-47_run_a9_crop_off", return "run_a9_crop_off"."""
    # "2022-09-24T11-39-47_run_a9_crop_off"
    #  012345678901234567890123456789012345
    if s[:4].isdigit() and \
            s[5:7].isdigit() and \
            s[8:10].isdigit() and \
            s[10] == 'T' and \
            s[11:13].isdigit() and \
            s[13] == '-' and \
            s[14:16].isdigit() and \
            s[16] == '-' and \
            s[17:19].isdigit() and \
            s[19] == '_':
        return s[20:]
    return s

for checkpoint in tqdm(checkpoints, smoothing=0.0):
    for ema in [False, ]:
        tqdm.write(checkpoint)
        # if "weights_{int}" in checkpoint: extract number
        # else: number = 0
        cp_iter = 'NA'
        if "weights_" in checkpoint:
            last_ = checkpoint.split("_")[-1].split(".")[0]
            if last_.isdigit():
                cp_iter = last_
        
        # get the name of the run
        if 'run' in checkpoint:
            checkpoint_short = [c for c in checkpoint.split("/") if "run" in c][0]
            checkpoint_short = maybe_strip_datetime(checkpoint_short)
        else:
            checkpoint_short = checkpoint.split("/")[-1].split(".")[0]
        if ',' in checkpoint_short:
            checkpoint_short = checkpoint_short.split(',')[0]
        
        # remove text before _ and after .pt
        checkpoint_short += f"_{cp_iter}mb"
        if ema:
            checkpoint_short += "_ema"
        # vae_run_03_1000mb
        
        outdir = f"{output_directory}/{checkpoint_short}"
        tqdm.write(checkpoint_short)
        input_args = r'--from_dir "C:\Users\DefAcc\Downloads\image_dir"'
        cmd = f'python scripts/vae2img.py' \
              f' --seed "{seed}"' \
              f' --ckpt "{checkpoint}"' \
              f' --outdir "{outdir}"' \
              f' {input_args}' \
              f' --H {384}' \
              f' --W {384}'
        if ema:
            cmd += ' --ema'
    
        # subprocess.check_call(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.system(cmd)
        tqdm.write("")
