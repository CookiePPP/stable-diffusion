import numpy as np

txt2img_payload = {
    "fn_index": 12,
    "data": [
        "rainbow dash, cute", # txt2img_prompt
        "", # txt2img_negative_prompt
        "None", # txt2img_prompt_style
        "None", # txt2img_prompt_style2
        16, # steps
        "DDIM", # sampler_index
        False, # restore_faces
        False, # tiling
        1, # batch_count
        1, # batch_size
        8, # cfg_scale
        -1, # seed
        -1, # subseed
        0, # subseed_strength
        0, # seed_resize_from_h
        0, # seed_resize_from_w
        False, # seed_checkbox
        512, # height
        512, # width
        False, # enable_hr
        False, # scale_latent
        0.7, # denoising_strength
        "None", # custom_inputs[0]
        False,  # custom_inputs[1]
        False,  # custom_inputs[2]
        None,   # custom_inputs[3]
        "",     # .... (continues till end of list)
        "Seed",
        "",
        "Nothing",
        "",
        True,
        False,
        None,
        "{\"prompt\": \"rainbow dash, cute\", \"all_prompts\": [\"rainbow dash, cute\"], \"negative_prompt\": \"\", \"seed\": 2329796450, \"all_seeds\": [2329796450], \"subseed\": 2176741151, \"all_subseeds\": [2176741151], \"subseed_strength\": 0, \"width\": 512, \"height\": 512, \"sampler_index\": 11, \"sampler\": \"DDIM\", \"cfg_scale\": 8, \"steps\": 16, \"batch_size\": 1, \"restore_faces\": false, \"face_restoration_model\": null, \"sd_model_hash\": \"0dfd2e7a\", \"seed_resize_from_w\": 0, \"seed_resize_from_h\": 0, \"denoising_strength\": null, \"extra_generation_params\": {}, \"index_of_first_image\": 0, \"infotexts\": [\"rainbow dash, cute\\nSteps: 16, Sampler: DDIM, CFG scale: 8, Seed: 2329796450, Size: 512x512, Model hash: 0dfd2e7a, Model: run_a12_day3_ema_1375_fp16\"], \"styles\": [\"None\", \"None\"], \"job_timestamp\": \"20221019150223\", \"clip_skip\": 1}","<p>rainbow dash, cute<br>\nSteps: 16, Sampler: DDIM, CFG scale: 8, Seed: 2329796450, Size: 512x512, Model hash: 0dfd2e7a, Model: run_a12_day3_ema_1375_fp16</p><div class='performance'><p class='time'>Time taken: <wbr>2.43s</p><p class='vram'>Torch active/reserved: 1784/2838 MiB, <wbr>Sys VRAM: 4532/12288 MiB (36.88%)</p></div>"
    ],
    "session_hash": "m66ie4jo1ph"
}
select_model_payload = {
    "fn_index": 142,
    "data": ["run_a12_day4_ema_1312_fp16.ckpt [a0ef266a]"],
    "session_hash": "m66ie4jo1ph"
}

def exp_linspace(start, end, steps):
    return np.exp(np.linspace(np.log(start), np.log(end), steps))

print([f'{round(x-10):04}+' for x in exp_linspace(10, 5010, 8)])

import torch
d = torch.load(r"X:\media\cookie\SN850\stable-diffusion\logs\run_a12_day5_ema_unet_only\checkpoints\weights_375.pt")
for k, v in d.items():
    if "cond_stage_model_ema" in k:
        del d[k]
torch.save(d, r"X:\media\cookie\SN850\stable-diffusion\logs\run_a12_day5_ema_unet_only\checkpoints\weights_375_.pt")

d = torch.load(r"X:\media\cookie\SN850\stable-diffusion\logs\run_a12_day5_ema_unet_only\checkpoints\weights_500.pt")
for k, v in d.items():
    if "cond_stage_model_ema" in k:
        del d[k]
torch.save(d, r"X:\media\cookie\SN850\stable-diffusion\logs\run_a12_day5_ema_unet_only\checkpoints\weights_500_.pt")