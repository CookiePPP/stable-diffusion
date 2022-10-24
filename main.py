import argparse, os, sys, datetime, glob, importlib, csv
import math
import random
import re
from copy import deepcopy
from typing import Iterator, Optional, List, Dict

import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.strategies import DeepSpeedStrategy
from torchvision.transforms import Resize

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by sqrt(ngpu * batch_size * n_accumulate)",
    )
    parser.add_argument(
        "--use_ema_weights",
        action='store_true',
        help="replace model weights with EMAs",
    )
    
    # storetrue arg for --dated_logdir
    parser.add_argument(
        "--dated_logdir",
        action='store_true',
        help="add date to logdir",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

class AspectRatioBucketedCollateFunc:
    def __init__(self, img_size: int, rectangle_batches: bool, variable_image_size: bool):
        self.img_size = img_size
        self.rectangle_batches = rectangle_batches # use non-square images
        self.variable_image_size = variable_image_size # use variable image sizes (reduce batch size if necessary to fit in memory)
    
    def __call__(self, batch: List[Dict]):
        """
        Get a list where each element is a dict of form
        {'image': tensor, 'caption': str}
        if rectangle_batches:
            image tensor has shape (w, h, 3)
        else:
            image tensor has shape (256, 256, 3)
        """
        if self.rectangle_batches:
            # calculate average aspect ratio of images
            aspect_ratios = [d['image'].shape[1] / d['image'].shape[0] for d in batch]
            avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
            # limit avg aspect ratio to 4, 1/4
            avg_aspect_ratio = max(min(avg_aspect_ratio, 4), 1/4)
            
            target_pixels = self.img_size ** 2
            batch_size = len(batch)
            if self.variable_image_size:
                # maybe increase image resolution but decrease batch size
                # (keeping total number of pixels constant)
                
                # calc maximum resolution scale factor
                max_resolution_factor = min(d['image'].shape[0]*d['image'].shape[1]/target_pixels for d in batch)
                max_resolution_factor = min(max_resolution_factor, 2.0) # limit to 2x total pixels
                max_resolution_factor = min(max_resolution_factor, batch_size) # ensure at least 1 image per batch
                
                log_resolution_factor = random.uniform(math.log2(0.25), math.log2(max_resolution_factor))
                resolution_factor = 2 ** log_resolution_factor
                resolution_factor = max(resolution_factor, 1.0)
                
                batch_size = int(batch_size / resolution_factor)
                target_pixels = int(target_pixels * resolution_factor)
                
                batch = batch[:batch_size]
            
            multiple = 64 # must dims must be multiple of 64 for autoencder
            # calculate width and height
            w = round(np.sqrt(target_pixels * avg_aspect_ratio) / multiple) * multiple # calc ideal width
            h = int(target_pixels / w / multiple) * multiple # then get the largest height that fits
            w = max(w, multiple)
            h = max(h, multiple)
            # resize images
            for d in batch:
                # replace d['image'] with trimmed version (so all images in batch have exactly same size and aspect ratio)
                image = d['image'] # (H, W, 3)
                H, W, _ = image.shape
                if W/H > avg_aspect_ratio: # if image is wider than avg
                    # trim left and right
                    pad = round(W - H * avg_aspect_ratio) / 2
                    image = image[:, int(pad):-int(pad) or None] # (H, W, 3) -> (H, H * avg_aspect_ratio, 3)
                else: # if image is taller than avg
                    # trim top and bottom
                    pad = round(H - W / avg_aspect_ratio) // 2
                    image = image[int(pad):-int(pad) or None, :] # (H, W, 3) -> (W / avg_aspect_ratio, W, 3)
                # resize using PIL
                image = Image.fromarray(image.numpy()).resize((w, h), resample=Image.BICUBIC)
                # convert back to tensor and normalize to [-1, 1]
                image = np.array(image).astype(np.uint8)
                d['image'] = torch.from_numpy(image).div(127.5).sub(1) # (H, W, 3)
            
            #print("aspect_ratios:", [f'{ar:.02f}' for ar in aspect_ratios])
            print("resolutions:", [f'{d["image"].shape[1]}x{d["image"].shape[0]}' for d in batch])
            
            # collate into batch
            images = torch.stack([d['image'] for d in batch], dim=0) # (N, 3, H, W)
            captions = [d['caption'] for d in batch]
            return {'image': images, 'caption': captions}
        else:
            images = [b['image'] for b in batch]
            captions = [b['caption'] for b in batch]
            images = torch.stack(images, dim=0)
            return {'image': images, 'caption': captions}

class AspectRatioBucketedDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 aspect_ratios: List[float] = None, n_buckets: int = None, batch_size: Optional[int] = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.n_rank = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.n_rank != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.n_rank) / self.n_rank  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.n_rank)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.n_rank

        if batch_size is None:
            batch_size = getattr(dataset, 'batch_size', None)
        if aspect_ratios is None:
            aspect_ratios = getattr(dataset, 'aspect_ratios', None)
        if n_buckets is None:
            n_buckets = getattr(dataset, 'n_buckets', None)
        assert batch_size    is not None, 'batch_size must be specified'
        assert aspect_ratios is not None, 'aspect_ratios must be specified'
        assert n_buckets     is not None, 'n_buckets must be specified'
        assert batch_size > 0, 'batch_size must be greater than 0'
        self.batch_size    = batch_size
        self.aspect_ratios = aspect_ratios # dict of {path: aspect_ratio}
        self.n_buckets     = n_buckets
        self.buckets = self._get_buckets()
        
        self.do_shuffle = shuffle
        self.seed = seed

    def _get_buckets(self):
        """
        Returns a list of buckets, where each bucket is a list of indices of roughly the same aspect ratio.
        """
        # get aspect ratios
        aspect_ratios = self.aspect_ratios
        # match order to self.dataset.image_paths
        aspect_ratios = [aspect_ratios[path] for path in self.dataset.image_paths]
        # sort aspect ratios
        sorted_idx = np.argsort(aspect_ratios)
        # split sorted_idx into equal sized buckets
        buckets = np.array_split(sorted_idx, self.n_buckets)
        return buckets

    def __iter__(self) -> Iterator:
        if self.do_shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        
        # subsample
        indices = indices[self.rank:self.total_size:self.n_rank]
        assert len(indices) == self.num_samples
        
        indices_buckets = []
        for bucket in self.buckets:
            indices_buckets.append([i for i in indices if i in bucket])
        
        # [[0, 2, 4, 6], [1, 3, 5, 7]] with batch size 2
        # -> [[0, 2], [1, 3], [4, 6], [5, 7]]
        # -> [0, 2, 1, 3, 4, 6, 5, 7]
        indices = []
        random_obj = random.Random(self.seed + self.epoch)
        while any(len(bucket) > 0 for bucket in indices_buckets):
            # add batch_size elements from random non-empty bucket
            buckets = [b for b in indices_buckets if len(b) > 0]
            bucket_idx = random_obj.choice(range(len(buckets)))
            bucket_idx = (bucket_idx + self.rank) % len(buckets) # offset bucket by rank
            bucket = buckets[bucket_idx]
            indices.extend(bucket[:self.batch_size])
            del bucket[:self.batch_size]
        assert len(indices) == self.num_samples
        
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, double_dataloaders=False, bucket_sampler=False, rank=0, n_rank=0):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.double_dataloaders = double_dataloaders # use two dataloaders for evaluations
        self.bucket_sampler = bucket_sampler
        self.rank = rank
        self.n_rank = n_rank
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        self.collate_fns = {
            k: AspectRatioBucketedCollateFunc(
                self.datasets[k].size,
                self.datasets[k].allow_rectangle_batches,
                self.datasets[k].variable_image_size,
            ) for k in self.datasets}
    
    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        
        shuffle = False if is_iterable_dataset else True
        
        if self.bucket_sampler:
            sampler = AspectRatioBucketedDistributedSampler(
                self.datasets['train'],
                num_replicas=self.n_rank,
                rank=self.rank,
                seed=0,
                batch_size=self.batch_size,
                shuffle=shuffle,
            )
            shuffle = None
        
        return DataLoader(self.datasets["train"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle,
                          sampler=sampler,
                          worker_init_fn=init_fn,
                          collate_fn=self.collate_fns["train"])

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        
        if self.bucket_sampler:
            sampler = AspectRatioBucketedDistributedSampler(
                self.datasets['validation'],
                num_replicas=self.n_rank,
                rank=self.rank,
                seed=0,
                batch_size=self.batch_size,
                shuffle=shuffle,
            )
            shuffle = None
        
        dataloader = DataLoader(self.datasets["validation"],
                   batch_size=self.batch_size,
                   num_workers=self.num_workers,
                   shuffle=shuffle,
                   sampler=sampler,
                   worker_init_fn=init_fn,
                   collate_fn=self.collate_fns["validation"])
        if self.double_dataloaders:
            return [dataloader, dataloader]
        return dataloader
    
    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['test'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        
        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)
        
        if self.bucket_sampler:
            sampler = AspectRatioBucketedDistributedSampler(
                self.datasets['test'],
                num_replicas=self.n_rank,
                rank=self.rank,
                seed=0,
                batch_size=self.batch_size,
                shuffle=shuffle,
            )
            shuffle = None

        dataloader = DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle,
                          sampler=sampler, collate_fn=self.collate_fns["test"])
        if self.double_dataloaders:
            return [dataloader, dataloader]
        return dataloader

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn,
                          collate_fn=self.collate_fns["predict"])


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        raise KeyboardInterrupt
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        elif 0:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        min_batch_freq = 256
        self.log_steps = [2 ** n for n in range(int(np.log2(min_batch_freq)), int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"{global_step:06}it_{batch_idx:06}_{k}.png"
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    weight_path = None
    checkpoint_iteration_from_name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        logname = _tmp[-1]
    else:
        if opt.name:
            name = opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = cfg_name
        else:
            name = ""
        logname = name + opt.postfix
        if opt.dated_logdir:
            logname = f'{now}_{logname}'
        logdir = os.path.join(opt.logdir, logname)
        
        if opt.resume_from_checkpoint is not None:
            # using regex extract int from "/weights_{int}.pt"
            match = re.search(r'weights_(\d+).pt', opt.resume_from_checkpoint)
            if match:
                checkpoint_iteration_from_name = int(match.group(1))
            
            # load weight path manually instead of using trainer.load_checkpoint (hacky workaround)
            weight_path = opt.resume_from_checkpoint # load weights only
            opt.resume_from_checkpoint = None        # load weights only
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)
    
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    use_deepspeed = trainer_config.pop("use_deepspeed", False)
    # default to ddp
    #trainer_config["accelerator"] = "deepspeed_stage_2_offload"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    #if not "gpus" in trainer_config:
    #    del trainer_config["accelerator"]
    #    cpu = True
    #else:
    #    gpuinfo = trainer_config["gpus"]
    #    print(f"Running on GPUs {gpuinfo}")
    cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # init model
    model = instantiate_from_config(config.model)
    
    # load checkpoint(s) before Lightning/DeepSpeed init
    # since these checkpoints are loaded earlier, they don't have to follow the Deepspeed format.
    missing_keys = []
    if weight_path:
        # weight_path can be comma separated list of weight_paths
        # merge them into one sd dict
        sd = {}
        for w_path in weight_path.split(","):
            pl_sd = torch.load(w_path, map_location="cpu")
            if "global_step" in pl_sd:
                print(f"Global Step: {pl_sd['global_step']}")
            if len(pl_sd) < 64 and "state_dict" in pl_sd:
                pl_sd = pl_sd["state_dict"]
            if len(pl_sd) < 64 and "module" in pl_sd:
                module = pl_sd["module"]
                del pl_sd["module"]
                pl_sd = {**module, **pl_sd}
            
            pl_sd = {k.split('module.')[-1]: v for k, v in pl_sd.items()}
            
            # if state_dict doesn't have "first_stage_model" or "cond_stage_model"
            # assume it is a VAE state_dict and load it into "first_stage_model"
            is_vae = not any("first_stage_model" in k or "cond_stage_model" in k for k in pl_sd.keys())
            if is_vae:
                pl_sd = {f'first_stage_model.{k}': v for k, v in pl_sd.items()}
            sd.update(pl_sd)
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        del pl_sd, sd
        print('')
        print(f'Loaded checkpoint {weight_path}')
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print(f"Missing {len(missing_keys)} keys and {len(unexpected_keys)} unexpected keys")
        print('')
    
    print(sorted(set(".".join(k.split(".")[:2]) for k in model.state_dict().keys())))
    
    if opt.use_ema_weights:
        print('\n\nUsing EMA weights!')
        time.sleep(2)
        if not any(k.startswith('model_ema.') for k in missing_keys):
            # if model_ema is not missing, load it
            model.model_ema.copy_to(model.model)
        if not any(k.startswith('cond_stage_model_ema.') for k in missing_keys) and hasattr(model, 'cond_stage_model_ema'):
            # if cond_stage_model_ema is not missing, load it
            model.cond_stage_model_ema.copy_to(model.cond_stage_model)
    
    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    if 1:
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": logname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": logname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    
    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    if 1:
        default_modelckpt_cfg = {"target": "pytorch_lightning.callbacks.ModelCheckpoint", "params": {}}
        default_modelckpt_cfg["params"]['dirpath'  ] = ckptdir
        default_modelckpt_cfg["params"]['filename' ] = "{step}"
        default_modelckpt_cfg["params"]['verbose'  ] = True
        default_modelckpt_cfg["params"]['save_last'] = True
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 1
        
        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    if 1:
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 2**20, # 1_048_576
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        
        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint': {
                    'target': 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 500,
                        'save_weights_only': True,
                    }
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)
        
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg}) # add modelcheckpoint to end of callbacks
            if getattr(config.model.params, 'use_ema'):
                ema_modelckpt_cfg = deepcopy(modelckpt_cfg)
                ema_modelckpt_cfg["params"]["monitor"] = f'{model.monitor}_ema'
                ema_modelckpt_cfg["params"]['save_last'] = False
                default_callbacks_cfg.update({'ema_checkpoint_callback': ema_modelckpt_cfg}) # add ema modelcheckpoint to end of callbacks
        
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']
    
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    from pytorch_lightning.strategies import DeepSpeedStrategy
    trainer = Trainer.from_argparse_args(
        trainer_opt,
        **trainer_kwargs,
        strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer =True,
            offload_parameters=False, # needs stage=3 to work
            allgather_bucket_size=100_000_000,
            reduce_bucket_size   =100_000_000,
            logging_batch_size_per_gpu=config.data.params.batch_size,
        ) if use_deepspeed else DDPPlugin(find_unused_parameters=False),
    )
    trainer.logdir = logdir
    
    # data
    if hasattr(config.model.params, 'use_ema'):
        config.data.params.double_dataloaders = config.model.params.use_ema
    else:
        print('No use_ema in config.model.params. Assuming False.')
        config.data.params.double_dataloaders = False
    config.data.params.rank = trainer.global_rank
    config.data.params.n_rank = trainer.world_size
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    
    # get num_gpus
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    
    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = (accumulate_grad_batches * ngpu * bs)**0.5 * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
    
    # run
    if opt.train:
        try:
            trainer.fit(model, data)
            print("Training finished")
        except Exception:
            raise
    
    # perform final evaluation and save results
    trainer.validate(model, data)
    
    ckpt_path = os.path.join(ckptdir, "last.ckpt")
    trainer.save_checkpoint(ckpt_path)
    
    if trainer.global_rank == 0:
        os.system(f'cd {ckpt_path}; python zero_to_fp32.py . ./weights.pt')
        sd = torch.load(f'{ckpt_path}/final_weights.pt')
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
