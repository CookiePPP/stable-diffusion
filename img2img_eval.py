import random
import time

from tqdm.contrib.itertools import product
import os
from tqdm import tqdm

checkpoints = [
    # a13
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a13_day2/checkpoints/weights_250.pt",
    
    # a12 var aspect ratio
    "logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a12_day6_var_aspect_ratio/checkpoints/weights_750.pt",
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a12_day5_var_aspect_ratio/checkpoints/weights_2250.pt",
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a12_day5_var_aspect_ratio/checkpoints/weights_1125.pt",
    
    # a12
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a12_day5_ema_unet_only/checkpoints/weights_125.pt",
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a12_day5_ema/checkpoints/weights_2125.pt",
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a12_day4_ema/checkpoints/weights_1312.pt",
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a12_day2_ema/checkpoints/weights_2062.pt",
    
    # a11
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a11_day7_ema/checkpoints/weights_687.pt",
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a11_day6_ema/checkpoints/weights_2124.pt",
    #"logs_vae/vae_run_03_day6/checkpoints/last.ckpt,logs/run_a11_day5_ema/checkpoints/weights_3687.pt",
    #"logs/run_a11_day4_ema/checkpoints/weights_2250.pt"
    #"logs/run_a11_day3/checkpoints/weights_1750.pt",
    #"logs/run_a11_day2/checkpoints/weights_3750.pt"
    
    # best a10 checkpoint
#   "logs/run_a10_day3/checkpoints/weights_2000.pt",
    
    # best 09 checkpoint
    #"logs/2022-09-16T12-35-47_run09_ds3_cont_dataloaderv3/checkpoints/weights.pt",
    
#   "logs/2022-09-22T12-06-42_run3_01_ds3_dataloaderv4_cond_frozen/checkpoints/weights_3500.pt",
#   "logs/2022-09-22T12-06-42_run3_01_ds3_dataloaderv4_cond_frozen/checkpoints/weights_7000.pt",
#   "logs/2022-09-22T12-06-42_run3_01_ds3_dataloaderv4_cond_frozen/checkpoints/weights_9000.pt",
#   "logs/2022-09-19T05-55-19_run2_01_ds3_dataloaderv4/checkpoints/weights_8000.pt",
#   "logs/2022-09-19T05-55-19_run2_01_ds3_dataloaderv4/checkpoints/weights_9000.pt",
#   "logs/2022-09-19T05-55-19_run2_01_ds3_dataloaderv4/checkpoints/weights_13000.pt",
#   "logs/2022-09-19T05-55-19_run2_01_ds3_dataloaderv4/checkpoints/weights_16000.pt",
#   "logs/2022-09-19T05-55-19_run2_01_ds3_dataloaderv4/checkpoints/weights_23000.pt",
#   "logs/2022-09-19T05-55-19_run2_01_ds3_dataloaderv4/checkpoints/weights_31000.pt",
#   "logs/2022-09-19T05-55-19_run2_01_ds3_dataloaderv4/checkpoints/weights_33000.pt",
#   "logs/2022-09-18T08-23-47_run10_ds3_cont_dataloaderv4/checkpoints/weights.pt",
#   "logs/2022-09-15T08-55-23_run08_ds3_cont_dataloaderv2/checkpoints/weights.pt",
#   "logs/2022-09-13T22-37-58_run07_ds3_with_cond_ds4_pretrained/checkpoints/weights.pt",
#   "logs/2022-09-12T18-46-32_run05_ds4_with_cond/checkpoints/weights.pt",
#   "logs/2022-09-12T08-07-46_run04_ds3_frozen_cond/checkpoints/weights.pt",
#   "logs/2022-09-11T07-48-19_run03_ds2_frozen_cond/checkpoints/weights.pt",
#   "logs/2022-09-09T03-51-52_run02_ds2_ordered_tags_warm1/checkpoints/weights_768.pt",
#   "logs/2022-09-09T03-51-52_run02_ds2_ordered_tags_warm1/checkpoints/weights_1280.pt",
#   "logs/2022-09-09T03-51-52_run02_ds2_ordered_tags_warm1/checkpoints/weights_2560.pt",
#   "logs/2022-09-09T03-51-52_run02_ds2_ordered_tags_warm1/checkpoints/weights_5120.pt",
#   "logs/2022-09-09T03-51-52_run02_ds2_ordered_tags_warm1/checkpoints/weights_10240.pt",
#   "logs/2022-09-09T03-51-52_run02_ds2_ordered_tags_warm1/checkpoints/weights_19712.pt",
#   "logs/2022-09-08T23-41-50_run01_ordered_tags_warm1/checkpoints/weights.pt",
#r"C:\Users\DefAcc\Downloads\sd_run_09_weights.pt",
]
#sampler = 'ddim'
#assert sampler in ['ddim', 'plms', 'klms'], 'sampler must be one of ddim, plms, klms'
#sampling_steps = 64 # default: 64
prompt_passes = 3 # how many additional times to repeat the prompt, default: 0
seed = 1121515872 # default: 1, "-1" is random seeds, "-2" is same random seed for all models
pretrained_path = r"C:\Users\DefAcc\Downloads\sd-v1-4.ckpt,"
output_directory = r"C:/Users/DefAcc/Downloads/sd_out_a11"

opt_sampling_steps = [64] # [8, 16, 32, 64, 128, 250]:
opt_scale = [8] # [7, 8, 9]:
opt_sampler = ['klms'] # ['klms', 'plms', 'ddim']:
opt_checkpoint = checkpoints
opt_ema = [True, ]
additional_args = '--export_model'
input_args = '--from_file "evaluation_prompts.txt"'

allow_ema = lambda file_name: 'ema' in file_name

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

opt_list = product(
    opt_sampling_steps, opt_scale, opt_sampler, opt_checkpoint, opt_ema,
    smoothing=0.0, position=2)

if seed == -2:
    seed = random.Random(time.time()).randint(0, 2**32 - 1)

for (sampling_steps, scale, sampler, checkpoint, ema) in opt_list:
    checkpoint_arg = checkpoint # clone the checkpoint path
    tqdm.write(checkpoint)
    # if "weights_{int}" in checkpoint: extract number
    # else: number = 0
    cp_iter = 'NA'
    if "weights_" in checkpoint:
        last_ = checkpoint.split("_")[-1].split(".")[0]
        if last_.isdigit():
            cp_iter = last_
    
    # get the name of the run
    if ',' in checkpoint:
        checkpoint = checkpoint.split(',')[-1] # get the last path if comma-separated
    if 'run' in checkpoint:
        checkpoint_short = [c for c in checkpoint.split("/") if "run" in c][0]
        checkpoint_short = maybe_strip_datetime(checkpoint_short)
    else:
        checkpoint_short = checkpoint.split("/")[-1].split(".")[0]
    
    # remove text before _ and after .pt
    checkpoint_short += f"_{cp_iter}mb"
    checkpoint_short += f"_{sampling_steps}ss"
    checkpoint_short += f"_{scale}sc"
    checkpoint_short += f"_{sampler.upper()}"
    if ema:
        checkpoint_short += "_ema"
    # run_a9_1000mb_8ss_7sc_KLMS
    
    outdir = f"{output_directory}/{checkpoint_short}"
    tqdm.write(checkpoint_short)
    
    cmd = f'python scripts/txt2img.py' \
          f' --seed "{seed}"' \
          f' --{sampler}' \
          f' --half' \
          f' --ckpt "{pretrained_path}{checkpoint_arg}"' \
          f' --sampling_steps {sampling_steps}' \
          f' --n_iter 1' \
          f' --n_rows 5' \
          f' --n_samples 2' \
          f' --scale {scale}' \
          f' --outdir "{outdir}"' \
          f' {input_args}' \
          f' --retry {prompt_passes}' \
          f' --H {512}' \
          f' --W {512}' \
          f' {additional_args}'
    if ema and allow_ema(checkpoint):
        cmd += ' --ema'
    
    #subprocess.check_call(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.system(cmd)
    tqdm.write("")
