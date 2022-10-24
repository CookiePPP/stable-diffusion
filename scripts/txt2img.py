import argparse, os
import random
import re
from typing import List

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from torch import autocast
from contextlib import nullcontext

from ldm.models.diffusion.ksampler import KSampler
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

torch.set_float32_matmul_precision("high") # enable TF32 for faster matmul
torch.backends.cuda.matmul.allow_tf32 = True

# load safety model
#from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
#from transformers import AutoFeatureExtractor
#safety_model_id = "CompVis/stable-diffusion-safety-checker"
#safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
#safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    
    return pil_images


def maybe_del(d, k):
    if k in d:
        del d[k]

def load_model_from_config(config, ckpt, device='cuda', verbose=True):
    # ckpt can be comma separated list of ckpts
    # merge them into one sd dict
    sd = {}
    for ckpt in ckpt.split(","):
        print(f"Loading '{ckpt}'")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if len(pl_sd) < 64:
            pl_sd = pl_sd["state_dict"]
        pl_sd = {k \
            .split('module.')[-1] \
            .replace("cond_stage_model.transformer.encoder.", "cond_stage_model.transformer.text_model.encoder.") \
            .replace("cond_stage_model.transformer.embeddings.", "cond_stage_model.transformer.text_model.embeddings.")\
            .replace("cond_stage_model.transformer.final_layer_norm.", "cond_stage_model.transformer.text_model.final_layer_norm.")\
            : v for k, v in pl_sd.items()}
        
        # if state_dict doesn't have "first_stage_model" or "cond_stage_model"
        # assume it is a VAE state_dict and load it into "first_stage_model"
        is_vae = not any("first_stage_model" in k or "cond_stage_model" in k for k in pl_sd.keys())
        if is_vae:
            pl_sd = {f'first_stage_model.{k}': v for k, v in pl_sd.items()}
        sd.update(pl_sd)
        del pl_sd
    
    maybe_del(sd, 'model_ema.decay'      ) # added for strict=True compatibility
    maybe_del(sd, 'model_ema.num_updates') # added for strict=True compatibility
    
    model = instantiate_from_config(config.model)
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    if len(missing_keys) > 0 and verbose:
        print("\nmissing keys:")
        print(missing_keys)
        
        # if cond_stage_model EMA weights are missing, try to load them from the main weights
        if any(k.startswith("cond_stage_model_ema.") for k in missing_keys) and hasattr(model, "cond_stage_model_ema"):
            sname = model.cond_stage_model_ema.sname_lookup
            cond_stage_model_sd = {}
            for k, v in model.cond_stage_model.state_dict().items():
                if k in sname:
                    cond_stage_model_sd[sname[k]] = v
            model.cond_stage_model_ema.load_state_dict(cond_stage_model_sd)
        
        # if model EMA weights are missing, try to load them from the main weights
        if any(k.startswith("model_ema.") for k in missing_keys) and hasattr(model, "model_ema"):
            sname = model.model_ema.sname_lookup
            model_sd = {}
            for k, v in model.model.state_dict().items():
                if k in sname:
                    model_sd[sname[k]] = v
            model.model_ema.load_state_dict(model_sd)
    
    if len(unexpected_keys) > 0 and verbose:
        print("\nunexpected keys:")
        print(unexpected_keys)
    
    model.to(device)
    model.eval()
    return model

def export_model(model, path):
    sd = {k: v for k, v in model.state_dict().items()}
    # move to cpu, detach and clone
    sd = {k: v.cpu().detach().data.clone() for k, v in sd.items()}
    # if float, cast to float16
    sd = {k: v.half() if v.dtype == torch.float else v for k, v in sd.items()}
    
    d = {
        'epoch': 0,
        'global_step': 0,
        'pytorch-lightning_version': pl.__version__,
        'callbacks': {},
        'lr_schedulers': [],
        'trainer': 'Cookie / CookiePPP / CookieGalaxy#8351 / cookietriplep@gmail.com',
        'state_dict': sd,
    }
    torch.save(d, path)


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def render_text(text: str, width: int, height: int, bottom_aligned: bool = False, font_size: int = 12):
    """
    Render text as (h w c) torch.FloatTensor using matplotlib

    Returns:
        torch.FloatTensor of shape (h, w, c) with values in [0., 1.]
    """
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)  # create a figure with the right size
    ax = fig.add_axes([0, 0, 1, 1])  # remove the white border
    ax.axis("off")  # remove the axes
    ax.text(0.0, 0.5 if not bottom_aligned else 0.05, text, ha="left", va="bottom",
            fontsize=font_size)  # write the text
    fig.canvas.draw()  # draw the figure to a buffer
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                         sep="")  # convert the figure to a numpy array with shape (h, w, c)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # reshape to (h, w, c)
    plt.close(fig)  # close the figure
    data = torch.from_numpy(data)  # convert to torch tensor
    return data / 255.0  # return the tensor of shape (h, w, c)


def main():
    opt = parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"
    
    if opt.seed == -1:
        opt.seed = random.randint(0, 2 ** 32 - 1)
    
    seed_everything(opt.seed)
    
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", device='cpu')
    
    if opt.ema:
        assert hasattr(model, "model_ema"), "Model does not have an EMA module"
        if hasattr(model, "model_ema"):
            kw = 'model.' + list(model.model_ema.sname_lookup.keys())[-1]
            mu = model.state_dict()[kw].mean().item()
            model.model_ema._swap_state_local(model.model)
            if hasattr(model, 'cond_stage_model_ema'):
                model.cond_stage_model_ema._swap_state_local(model.cond_stage_model)
            mu_ema = model.state_dict()[kw].mean().item()
            assert mu != mu_ema, "model and model_ema are identical"
        else:
            print("WARNING: Model does not have EMA weights")
            time.sleep(2)
    model.model_ema = None
    model.cond_stage_model_ema = None
    model.use_ema = False
    
    if opt.half:
        model = model.half()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    if opt.export_model:
        print("Exporting model...")
        export_model(model, f"{opt.outdir}/model_fp16.ckpt")
        print("Done!")
    
    assert opt.plms + opt.klms + opt.ddim == 1, "only one of plms, klms, ddim can be enabled"
    if opt.plms:
        sampler = PLMSSampler(model)
    elif opt.klms:
        sampler = KSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    if opt.use_watermark:
        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    else:
        wm_encoder = None
    
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    
    scale_sweep = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    steps_sweep = [2, 4, 8, 16, 32, 64, 128, 250]
    use_scale_sweep = -1 # -1 = disabled, 0+ = index
    use_steps_sweep = -1 # -1 = disabled, 0+ = index
    retry_prompt = opt.retry # 0 = disabled, 1+ = number of retries
    
    while True:
        print('')
        torch.manual_seed(opt.seed)
        prompt = None
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]
        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                # remove lines starting with #
                data = [line for line in data if not line.startswith("#")]
                data = list(chunk(data, batch_size))
        
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len([p for p in os.listdir(outpath)]) - 1
        
        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        if use_scale_sweep > -1:
            if use_scale_sweep >= len(scale_sweep):
                print("Scale sweep done.")
            opt.scale = scale_sweep[use_scale_sweep]
        if use_steps_sweep > -1:
            if use_steps_sweep >= len(steps_sweep):
                print("Steps sweep done.")
            opt.sampling_steps = steps_sweep[use_steps_sweep]
        
        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    # create tqdm object for opt.n_iter*data double for loops
                    pbar = tqdm(total=opt.n_iter * len(data), desc="Iters", position=1)

                    tic = time.time()
                    all_samples = list()
                    for n in range(opt.n_iter):
                        torch.manual_seed(opt.seed + 1024 * n)
                        for prompts in data:
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.sampling_steps,
                                                             conditioning=c,
                                                             batch_size=opt.n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=opt.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=opt.ddim_eta,
                                                             x_T=start_code)
                            
                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    
                            # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                            x_checked_image = x_samples_ddim
                            
                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                            
                            if not opt.skip_save:
                                for x_sample, prompt in zip(x_checked_image_torch, prompts):
                                    x_sample = add_captions(x_sample, prompt)
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    if wm_encoder is not None:
                                        img = put_watermark(img, wm_encoder)
                                    img_path = os.path.join(sample_path, f"{base_count:05}.png")
                                    while os.path.exists(img_path):
                                        base_count += 1
                                        img_path = os.path.join(sample_path, f"{base_count:05}.png")
                                    img.save(img_path)
                                    base_count += 1
                            
                            if not opt.skip_grid:
                                all_samples.append(x_checked_image_torch)
                            pbar.update(1)
                    pbar.close()
                    
                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)  # (n_iters, batch_size, C, H, W)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        
                        grid = add_captions(grid, data)
                        
                        padding = 2
                        grid = make_grid(grid, nrow=n_rows, padding=padding)  # (C, H*n_iters, W*n_rows)
                        
                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        
                        img_path = os.path.join(outpath, f'grid-{grid_count:04}.png')
                        while os.path.exists(img_path):
                            grid_count += 1
                            img_path = os.path.join(outpath, f'grid-{grid_count:04}.png')
                        img.save(img_path)
                        
                        # save a copy of the prompt and generated images for debugging
                        rand_int = f'{random.Random(time.time()).randint(0, 1000000):07}'
                        os.makedirs(os.path.join(outpath, 'prompts'), exist_ok=True)
                        with open(os.path.join(outpath, 'prompts', f'grid-{grid_count:04}-{rand_int}.txt'), 'w') as f:
                            f.write(f'prompt: {prompt}\n')
                            f.write(f'ckpt: {opt.ckpt}\n')
                            f.write(f'steps: {opt.sampling_steps}\n')
                            f.write(f'scale: {opt.scale}\n')
                            f.write(f'seed: {opt.seed}\n')
                            f.write(f'batch_size: {opt.n_samples}\n')
                        os.link(
                            os.path.join(outpath, f'grid-{grid_count:04}.png'),
                            os.path.join(outpath, 'prompts', f'grid-{grid_count:04}-{rand_int}.png')
                        )
                        grid_count += 1
                    toc = time.time()
        if use_steps_sweep > -1:
            use_steps_sweep += 1
            if use_steps_sweep >= len(steps_sweep):
                use_steps_sweep = -1
        if use_scale_sweep > -1:
            use_scale_sweep += 1
            if use_scale_sweep >= len(scale_sweep):
                use_scale_sweep = -1
        if retry_prompt > 0:
            opt.seed = opt.seed + 864635 # add a random number to the seed to avoid repeating the same prompt
            retry_prompt -= 1
            continue
        
        total_n_sampling_steps = opt.sampling_steps * opt.n_samples * len(data)
        sampling_time_elapsed = toc - tic
        sampling_time_per_step = total_n_sampling_steps / sampling_time_elapsed
        print(f"Your samples took {toc - tic:0.1f} seconds ({sampling_time_per_step:.1f}i/s) and waiting for you here: '{img_path}'")
        if use_scale_sweep > -1 or use_steps_sweep > -1:
            continue
        elif opt.persistent:
            prompt = input("Enter new prompt (or 'exit' to quit): \n> ")
            if prompt == "exit" or len(prompt.strip()) == 0:
                break
            
            # maybe extract "steps:{int}" or "steps:sweep" from prompt (e.g. "steps:1000", "steps:-1", "steps:sweep")
            steps = re.findall(r'steps:(-?\d+)', prompt) or re.findall(r'steps:sweep', prompt)
            if len(steps) > 0:
                if steps != ['steps:sweep']:
                    opt.sampling_steps = int(steps[0])
                else:
                    use_steps_sweep = 0
                prompt = prompt.replace(f"steps:{steps[0]}", "").replace("steps:sweep", "")
            
            # maybe extract "rows:{int}" from prompt
            rows = re.findall(r'rows:(-?\d+)', prompt)
            if len(rows) > 0:
                opt.n_iter = int(rows[0])
                prompt = prompt.replace(f"cols:{rows[0]}", "")
            
            # maybe extract "scale:{float}" or "scale:sweep" from prompt (e.g. "scale:0.5", "scale:1.0")
            scale = re.findall(r'scale:(-?\d+\.?\d*)', prompt) or re.findall(r'scale:sweep', prompt)
            if len(scale) > 0:
                if scale != ['scale:sweep']:
                    opt.scale = float(scale[0])
                else:
                    use_scale_sweep = 0
                prompt = prompt.replace(f"scale:{scale[0]}", "").replace("scale:sweep", "")
            
            # maybe extract "seed:{int}" from prompt
            seed = re.findall(r'seed:(-?\d+)', prompt)
            if len(seed) > 0:
                opt.seed = int(seed[0])
                prompt = prompt.replace(f"seed:{seed[0]}", "")
            if opt.seed < 1:
                opt.seed = time.time()
            
            # maybe extract "retry:{int}" from prompt
            retry = re.findall(r'retry:(-?\d+)', prompt)
            if len(retry) > 0:
                retry_prompt = int(retry[0])
                prompt = prompt.replace(f"retry:{retry[0]}", "")
            
            opt.prompt = prompt.strip()
        else:
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=50,
        help="number of sampling steps, higher = more detail",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--klms",
        action='store_true',
        help="use klms sampling",
    )
    parser.add_argument(
        "--ddim",
        action='store_true',
        help="use ddim sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--use_watermark",
        action='store_true',
        help="add a watermark to the images",
    )
    parser.add_argument(
        "--persistent",
        action='store_true',
        help="keep model loaded and reuse it for multiple prompts",
    )
    parser.add_argument(
        "--half",
        action='store_true',
        help="use half precision for weights",
    )
    parser.add_argument(
        "--export_model",
        action='store_true',
        help="save a copy of the model to the output dir",
    )
    parser.add_argument(
        "--ema",
        action='store_true',
        help="use EMA/SWA weights",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-finetune.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="number of passes to retry sampling",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()
    return opt


def add_captions(grid: torch.Tensor, data: List[List[str]], font_size = 10):
    if len(grid.shape) == 3:
        # grid.shape: (c, h, w)
        title_img = render_text(
            data, height=font_size*2, width=grid.shape[2],
            bottom_aligned=True, font_size=font_size).permute(2, 0, 1)
        return torch.cat([title_img, grid], dim=1)
    elif len(grid.shape) == 4:
        # grid.shape: (n, c, h, w)
        # render text of every prompt and add to top of each image
        total_prompts = [p for b in data for p in b]
        col_titles = [render_text(
            p, height=font_size*2, width=grid.shape[3],
            bottom_aligned=True, font_size=font_size).permute(2, 0, 1)
              for i, p in enumerate(total_prompts)
        ] # list[torch.FloatTensor(c, h, w)]
        
        # add column titles to images
        grid_with_titles = []
        for i, (title_img, img) in enumerate(zip(col_titles, grid)):
            grid_with_titles.append(torch.cat([title_img, img], dim=1))
        grid = torch.stack(grid_with_titles)
        return grid
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
