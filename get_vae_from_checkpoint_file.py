import torch
d_vae = torch.load("logs/vae_run_01/checkpoints/last.ckpt", map_location='cpu')
print(d_vae.keys())
print(d_vae['state_dict'].keys())

d_sd = torch.load("pretrained/sd-v1-4.ckpt", map_location='cpu')
print(d_sd.keys())
print(d_sd['state_dict'].keys())

def state_dict_sd_to_vae(sd_sd):
    sd_vae = {}
    for k, v in sd_sd.items():
        # remove "first_stage_model." from key
        if k.startswith("first_stage_model."):
            k = k[18:]
        sd_vae[k] = v
    return sd_vae

d_vae['state_dict'].update(state_dict_sd_to_vae(d_sd['state_dict']))

torch.save(d_vae, "pretrained/sd-v1-4_vae.ckpt")