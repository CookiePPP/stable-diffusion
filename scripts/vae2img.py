import argparse, os
import glob
import random

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything

from ldm.data.personalized import load_image
from ldm.util import instantiate_from_config

torch.set_float32_matmul_precision("high")  # enable TF32 for faster matmul
torch.backends.cuda.matmul.allow_tf32 = True


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


def load_model_from_config(config, ckpt, device='cuda', verbose=True):
    # ckpt can be comma separated list of ckpts
    # merge them into one sd dict
    sd = {}
    for ckpt in ckpt.split(","):
        print(f"Loading '{ckpt}'")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if len(pl_sd) < 64:
            pl_sd = pl_sd["state_dict"]
        pl_sd = {k.split('module.')[-1].replace('first_stage_model.', ''): v for k, v in pl_sd.items()}
        sd.update(pl_sd)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("\nmissing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("\nunexpected keys:")
        print(u)
    
    model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/vae2img-samples"
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
        "--half",
        action='store_true',
        help="use half precision for weights",
    )
    parser.add_argument(
        "--ema",
        action='store_true',
        help="use EMA/SWA weights",
    )
    parser.add_argument(
        "--skip_vae",
        action='store_true',
        help="ignore VAE and output the input images",
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
        "--from_dir",
        type=str,
        help="if specified, load images from this directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/autoencoder/sd_1.4v_vae_finetune.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/vae_run_03/checkpoints/step=4000.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
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


@torch.no_grad()
def main():
    opt = parse_args()
    
    if opt.seed == -1:
        opt.seed = random.randint(0, 2 ** 32 - 1)
    
    seed_everything(opt.seed)
    
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", device='cpu')
    
    if opt.ema:
        assert hasattr(model, "model_ema"), "Model does not have EMA weights"
        kw = 'model.' + list(model.model_ema.sname_lookup.keys())[-1]
        ke = 'model_ema.' + kw[len('model.'):].replace('.', '')
        mu = model.state_dict()[kw].mean().item()
        model.model_ema._swap_state_local(model.model)
        mu_ema = model.state_dict()[kw].mean().item()
        assert mu != mu_ema, "model and model_ema are identical"
    model.model_ema = None
    model.use_ema = False
    
    if opt.half:
        model = model.half()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)
    
    #batch_size = opt.batch_size
    batch_size = 16
    
    if opt.from_dir:
        image_paths = glob.glob(os.path.join(opt.from_dir, "*.*"))
        formats = ['png', 'jpg', 'jpeg', 'bmp', 'webp']
        image_paths = [p for p in image_paths if p.split('.')[-1].lower() in formats]
        image_paths = sorted(image_paths)
    
    # create tqdm object for opt.n_iter*data double for loops
    pbar = tqdm(total=len(image_paths), desc="Iters", position=1)
    
    tic = time.time()
    torch.manual_seed(opt.seed)
    for image_path in image_paths:
        imagename = os.path.splitext(os.path.basename(image_path))[0]
        image = load_image_to_model_input(image_path, opt)
        
        if opt.skip_vae:
            rec_image = image.permute(0, 3, 1, 2)
        else:
            p = next(model.parameters())
            image = image.to(device=p.device, dtype=p.dtype)
            rec_image, _ = model(model.get_input({'image': image}))
        rec_image = rec_image.cpu().clamp(min=-1.0, max=1.0).add(1.0).div(2.0)
        
        if not opt.skip_save:
            for rec_image_i in rec_image:
                rec_image_i = 255. * rearrange(rec_image_i.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(rec_image_i.astype(np.uint8))
                img_path = os.path.join(outpath, f"{imagename}.png")
                img.save(img_path)
        pbar.update(1)
    pbar.close()
    toc = time.time()
    print(f"Took {toc - tic:.2f} seconds")

def load_image_to_model_input(image_path, opt):
    image = load_image(image_path)
    img = np.array(image).astype(np.uint8)
    crop = min(img.shape[0], img.shape[1])
    h, w, = img.shape[0], img.shape[1]
    img = img[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
    image = Image.fromarray(img)  # convert back to PIL image
    image = image.resize((opt.W, opt.H), Image.LANCZOS)
    image = np.array(image).astype(np.uint8)  # convert back to numpy array
    image = torch.from_numpy(image / 127.5 - 1.0).float()  # shape = (H, W, 3)
    image = image.unsqueeze(0)  # shape = (1, H, W, 3)
    return image


if __name__ == "__main__":
    main()
