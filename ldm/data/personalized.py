import glob
import json
import math
import multiprocessing
import os
import shutil
import time
import traceback
import urllib
from copy import deepcopy

import numpy as np
import PIL
import torch
import webp
import pickle as pkl
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as vF

import random

from torchvision import transforms
from tqdm import tqdm

class LowResolutionError(Exception):
    pass

class ExtremeAspectRatioError(Exception):
    pass

PIL.Image.MAX_IMAGE_PIXELS = 933120000

special_token_list = [
    "~> ", "~@ ", "~~ ", "°: ", "Â¡ ", "Â¢ ", "Â£ ", "Â¥ ", "Â© ", "Â« ", "Â® ",
    "Â¯ ", "Â° ", "Â² ", "Â´ ", "Â· ", "Âº ", "Â» ", "Â½ ", "Â¿ ", "Ã¡ ", "Ã¤ ",
    "Ã¨ ", "Ã© ", "Ã« ", "Ã³ ", "Ã¶ ", "Ã¸ ", "Ãº ", "Ã¼ ", "Ä± ", "Ð° ", "Ð¾ ",
    "Ø§ ", "Ø¨ ", "Ø© ", "Øª ", "Ø¯ ", "Ø± ",
    "Ã¡n ", "Ã£o ", "Ã§a ", "Ã¨s ", "Ã©e ", "Ã©s ", "Ã±a ", "Ã±o ", "Ã³n ",
    "à¤¤ ", "à¤¦ ", "à¤¨ ", "à¤ª ", "à¤¬ ", "à¤® ", "à¤¯ ", "à¤° ", "à¤² ",
    "à¤µ ", "à¤¶ ", "à¤¸ ", "à¤¹ ", "à¤¾ ", "à¤¿ ", "à¦¾ ", "à®¤ ", "à®© ",
    "à®ª ", "à®® ", "à®° ", "à®² ", "à®¾ ", "à®¿ ", "à²¿ ", "à¸¡ ", "à¸¢ ",
    "à¸£ ", "à¸¥ ", "à¸§ ", "à¸ª ", "à¸° ", "à¸± ", "à¸² ", "à¸´ ", "à¸µ ",
    "à¸· ", "à¸¸ ", "à¸¹ ", "ë¯¼ ", "ï¿½ "
]
# add new entries to the end of below lines
hflipped_token = special_token_list.pop(0)
rotated_token  = special_token_list.pop(0)
cropped_token  = special_token_list.pop(0)
vflipped_token = special_token_list.pop(0)
squashed_token = special_token_list.pop(0)
smallres_token = special_token_list.pop(0)



# taken from https://stackoverflow.com/q/9166400
def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background

def load_image(image_path):
    if image_path.endswith(".webp"):
        image = webp.load_image(image_path, mode='RGBA')
    else:
        image = Image.open(image_path).convert('RGBA')
    image = pure_pil_alpha_to_color_v2(image, (255, 255, 255))
    return image

def get_aspect_ratio(image_path):
    image = load_image(image_path)
    return [image_path, image.width / image.height]

def rand_join(s, t, sep=', '):
    if random.random() < 0.5:
        return f"{s}{sep}{t}"
    else:
        return f"{t}{sep}{s}"
    # This function should be renamed to "unbiased_join" or something like that

def range_with_exclusion(start, length, exclusion):
    """
    Get a list of ints [start, ..., start+length+len(exclusion)]
    excluding the values in `exclusion`.
    """
    cur_len = 0
    start = deepcopy(start)
    while cur_len < length:
        if start not in exclusion:
            yield start
            cur_len += 1
        start += 1

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=1,
                 interpolation="bicubic",
                 flip_p=0.5,
                 flip_text_p=None,
                 set="train",
                 center_crop=False,
                 allow_rectangle_batches=False,
                 n_buckets=None,
                 variable_image_size=False,
                 tag_dropout_prob=0.1,
                 tag_use_alias_prob=0.5,
                 caption_drop_prob=0.1,
                 human_caption_prob=0.5,
                 shuffle_tags_prob=0.0,
                 tagdata_path=None,
                 dump_dataset_sample_path=None,
                 scoreperc_path=None,
                 scoreperc_key='score',
                 max_zoom=1.0,
                 max_image_squash=0.05,
                 use_rotated_images=False,
                 use_vertical_flip=False,
                 aspect_ratios_path='aspect_ratios.json',
                 ):
        self.set = set
        self.data_root = data_root
        self.dump_dataset_sample_path = dump_dataset_sample_path
        
        self.image_paths = glob.glob(os.path.join(data_root, "**", "*.*"), recursive=True)
        # filter out non-image files
        exts = [".jpg", ".jpeg", ".png", ".webp"]
        self.image_paths = [file_path for file_path in self.image_paths if os.path.splitext(file_path)[1].lower() in exts]
        # deterministic shuffle
        random.Random(0).shuffle(self.image_paths)
        
        self.scoreperc_key = scoreperc_key
        if scoreperc_path is not None:
            # load {image_id: {
            #   'score': percentile,
            #   'wilson_score': percentile
            # }} dict
            if scoreperc_path.endswith('.pkl'):
                with open(scoreperc_path, 'rb') as f:
                    self.scoreperc = pkl.load(f)
            elif scoreperc_path.endswith('.json'):
                import json
                with open(scoreperc_path, 'r') as f:
                    self.scoreperc = json.load(f)
            else:
                raise ValueError('scoreperc_path must be either a .pkl or .json file')
        
        # derpibooru specific attributes
        self.human_caption_prob = human_caption_prob # if there is a human written caption, this is the chance it will be used instead of the auto-generated one
        self.possible_ratings = ["safe", "suggestive", "questionable", "explicit", "grotesque", "grimdark", "semi-grimdark"]
        assert tagdata_path is not None, 'tagdata_path must be specified for this dataset'
        self.tagdata = pkl.load(open(tagdata_path, 'rb'))
        is_character = lambda tag: tag['category'] == 'character' # character
        is_oc = lambda tag: tag['name'].startswith('oc:') # original character
        is_human = lambda tag: tag['name'] == 'human' # unnamed human character
        self.character_tags = [tag for tag in self.tagdata.values() if is_character(tag) or is_oc(tag) or is_human(tag)]
        self.character_names = {tag['name'] for tag in self.character_tags}
        
        # hardcoded list of tags that must never be hidden from the model. (so the model only produces images with these tags when explicitly told to)
        self.tag_keep_list = [
            #*self.possible_ratings,  # image ratings
            "text", "dialogue", "speech bubble", "signature", "patreon logo", "dirty talk", "sex noises", "watermark", "meme",  # images with text or icons overlays
            "internal cumshot", "egg cell", "internal", "an egg being attacked by sperm",  # Picture-in-Picture tags
            "white background", "simple background", "gradient background", "abstract background", "blue background", # lower quality backgrounds
            "monochrome", "sketch", "line art", "grayscale",  # WIP styles
            "futa", "teats", "crotchboobs", "huge crotchboobs", "impossibly large ponut", "huge butt", "spread anus",
            "foalcon", "younger",  # legally / morally questionable
            "running makeup", "piercing", "sad",  # sad/unhappy images
            "human", "hand", "disembodied hand",  # human
            "anthro", "equestria girls", "humanized", # species
            "self ponidox", "collage", "comic", "lewd emotions", "expressions", "picture in picture", "drawing meme", # images with more than 1 scene
            "double penetration", "group sex", "gangbang", "mmf threesome", "double blowjob", "multiple blowjob" # more than 2 characters in a scene
        ]
        # hardcoded list of tags that must never be hidden from the model. (so the model only produces images with these tags when explicitly told to)
        
        # tags which imply text focus
        self.text_tags = ["caption", "comic", "talking", "text", "dialogue", "speech bubble", "sex noises"]
        
        # tags that the model should infer for the user
        # (tags that are often forgotten)
        self.auto_add_tags = [
            "safe", "suggestive", "questionable", "explicit",
            "sitting",
            "bipedal",
            "grin",
            "crying",
            "lying down",
            "presenting",
            "spreading",
            "spread legs",
            "on side",
            "on back",
            "prone",
            "raised hoof",
            "underhoof",
            "frog (hoof)",
            "standing",
            "rearing",
            "raised tail",
            "belly",
            "flexible",
            "wingboner",
            "spread wings",
            "partially open wings",
            "upside down",
            "bent over",
            "face down ass up",
            "flying",
            "underhoof",
            "cute",
            "smiling",
            "blush",
            "blushing",
            "eyes closed",
            "lidded eyes",
            "bedroom eyes",
            "wide eyes",
            "wingding eyes",
            "crazy eyes",
            "open mouth",
            "tongue out",
            "floppy ears",
            "looking back",
            "looking at you",
            "female focus",
            "male focus",
            "solo focus",
            "close-up",
            "extreme close-up",
            "front view",
            "three quarter view",
            "side view",
            "rear view",
            "low angle",
            "submissive pov",
            "imminent facesitting",
            "pov",
            "first person view",
            "male pov",
            "female pov",
            "portrait",
            "tail",
            "plot",
            "butt",
            "looking back at you",
            "belly button",
            "dock",
            "taint",
            "medial ring",
            "nipples",
            "perineal raphe",
            "dark genitals",
            "vulva",
            "clitoris",
            "vagina",
            "anus",
            "anatomically correct",
            "ponut",
            "vulvar winking",
            "human vagina on pony",
            "penis",
            "balls",
            "horsecock",
            "glazed dick",
            "big penis",
            "medial ring",
            "erection",
            "flared",
        ]
        
        # artist to ID mapping so model has dedicated tokens for styles.
        # TODO
        
        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        
        self.center_crop = center_crop
        self.allow_rectangle_batches = allow_rectangle_batches
        self.variable_image_size = variable_image_size
        self.n_buckets = n_buckets
        if self.allow_rectangle_batches:
            assert self.n_buckets is not None
            
        self.shuffle_tags_prob = shuffle_tags_prob
        self.tag_dropout_prob = tag_dropout_prob
        self.tag_use_alias_prob = tag_use_alias_prob
        self.caption_drop_prob = caption_drop_prob
        
        self._length = int(self.num_images * repeats)

        self.size = size
        self.max_zoom = max_zoom
        self.max_image_squash = max_image_squash
        self.use_rotated_images = use_rotated_images
        self.use_vertical_flip = use_vertical_flip
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_text_p = flip_p if flip_text_p is None else flip_text_p
        self.flip_p = flip_p
        
        # calculate aspect ratios
        self.aspect_ratios_path = aspect_ratios_path
        self.aspect_ratios = self._get_aspect_ratios()
        
        # maybe dump a sample of the processed dataset for debugging
        if self.dump_dataset_sample_path is not None:
            self.dump_dataset_sample()
    
    def _get_aspect_ratios(self):
        # load aspect ratios from json if they exist
        aspect_ratios = {}
        if os.path.exists(self.aspect_ratios_path):
            with open(self.aspect_ratios_path, "r") as f:
                aspect_ratios = json.load(f)
        
        s_len = len(aspect_ratios)
        # use multiprocessing with tqdm to calculate aspect ratios for images that don't have them
        missing_paths = [path for path in self.image_paths if path not in aspect_ratios]
        if len(missing_paths):
            with multiprocessing.Pool() as p:
                for image_path, aspect_ratio in tqdm(p.imap_unordered(get_aspect_ratio, missing_paths), total=len(missing_paths)):
                    aspect_ratios[image_path] = aspect_ratio
        
        # save aspect ratios to json
        if len(aspect_ratios) != s_len:
            tmp_path = self.aspect_ratios_path + ".tmp" + str(random.randint(0, 1000000))
            with open(tmp_path, "w") as f:
                json.dump(aspect_ratios, f)
            os.rename(tmp_path, self.aspect_ratios_path)
        return aspect_ratios
        
    def dump_dataset_sample(self):
        # create folder for set
        set_folder = os.path.join(self.dump_dataset_sample_path, self.set)
        if False and os.path.exists(set_folder): # TODO: make this work without multi-process race condition
            # check folder only contains ".png" and ".txt" files
            for f in os.listdir(set_folder):
                if not f.endswith(".png") and not f.endswith(".txt"):
                    raise Exception(f"Dump folder \"{set_folder}\" already exists and contains non-dump files.")
            shutil.rmtree(set_folder)
        os.makedirs(set_folder, exist_ok=True)
        print('dumping processed dataset sample to', set_folder)
        for i in range(20):
            # get processed data
            d = self.__getitem__(i)
            image_basename = os.path.splitext(os.path.basename(d['image_path']))[0]
            
            # convert pt tensor to PIL image
            image = d['image'].numpy()
            if not self.allow_rectangle_batches:
                image = (image + 1) / 2 * 255 # [-1, 1] -> [0, 255]
            image = image.astype(np.uint8)
            image = PIL.Image.fromarray(image)
            
            # save image (write to tmp file first to avoid corrupting existing files)
            random_num = random.Random(time.time()).randint(0, 1000000)
            tmp_image_path = os.path.join(set_folder, image_basename + f'.tmp{random_num}.png')
            image.save(tmp_image_path)
            os.rename(tmp_image_path, os.path.join(set_folder, image_basename + '.png'))
            
            # save caption (write to tmp file first to avoid corrupting existing files)
            tmp_caption_path = os.path.join(set_folder, image_basename + f'.tmp{random_num}.txt')
            with open(tmp_caption_path, 'w') as f:
                f.write(', '.join(d['tags']) + '\n\n')
                f.write(d['caption'])
            os.rename(tmp_caption_path, os.path.join(set_folder, image_basename + '.txt'))
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        example = None
        while example is None:
            try:
                example = self.load_image_and_quote(i)
            except Exception as e:
                # print stack trace
                traceback.print_exc()
                #print(f"Error loading image {self.image_paths[i % self.num_images]}: {e}")
                random_obj = random if self.set == "train" else random.Random(i)
                i = random_obj.randint(0, len(self.image_paths) - 1)
        return example

    def load_image_and_quote(self, i):
        # init custom random object, because this class runs in a separate process
        # it's not deterministic for validation and test sets without this
        random_obj = random if self.set == "train" else random.Random(i)
        
        example = {}
        image_path = self.image_paths[i % self.num_images]
        example['image_path'] = image_path
        
        # load text
        if 1:
            # load comma-separated-tags from .txt file with same basename as image
            text_file_path = os.path.splitext(image_path)[0] + ".txt"
            with open(text_file_path, "r") as f:
                text = f.read()
            # unescape URL-encoded characters
            text = urllib.parse.unquote(text.replace('+', ' '))
            
            # get lines (line 0 is image tags, any other lines are human written captions)
            lines = text.splitlines()
            lines = [line for line in lines if line.strip() != ""]
            
            # get all tags
            tags = [tag.strip().lower() for tag in lines[0].split(",")]
            example['tags'] = tags
            
            # get image content rating
            content_rating = [tag for tag in tags if tag in self.possible_ratings][0]
            
            # calculate drop chance for each tag. This is independent of self.tag_dropout_prob
            # each tags drop chance is equal to 1 - 0.5**n where n is how many other tags in the image imply this tag
            # e.g: if the tags are 'twilight sparkle' and 'twilight sparkle (alicorn)'
            # then the drop chance for 'twilight sparkle' is 1 - 0.5**1 = 0.5
            # since twilight sparkle can be inferred from the other tag by the network
            # and the drop chance for 'twilight sparkle (alicorn)' is 1 - 0.5**0 = 0
            # since it is the only tag that implies this tag
            # TLDR: remove tags that can be guessed from other tags
            drop_chances = {}
            for tagname in tags:
                tagname = tagname.lower()
                if tagname not in self.tagdata:
                    if f'oc:{tagname}' in self.tagdata:
                        tagname = f'oc:{tagname}'
                    else:
                        drop_chances[tagname] = 0.0
                        continue
                tag = self.tagdata[tagname]
                implied_by = tag['implied_by_tags'] # list of tags that imply this tag exists
                n_implied_by_found = sum([1 for implied_tag in implied_by if implied_tag in tags])
                drop_chances[tagname] = 1 - 0.5**n_implied_by_found
            
            # auto_add_tags are tags that the model must be able to infer
            # so they must be dropped at least some % of the time
            for tagname in tags:
                if tagname in self.auto_add_tags:
                    drop_chances[tagname] = max(drop_chances[tagname], 0.5)
            
            # pick random line if self.set == 'train' else deterministic line
            use_human_caption = (self.human_caption_prob > random_obj.random()) and len(lines) > 1
            
            if use_human_caption:
                line = random_obj.choice(lines[1:]) # get random human caption
                # insert content rating into caption if missing
                if content_rating not in line:
                    line = f"{content_rating}, {line}"
                example["caption"] = line.strip()
            else:
                # useful funcs
                is_character = lambda tag: tag.lower() in self.character_names or tag.lower().startswith('oc:') or tag.lower() == 'human'
                is_important = lambda tag: tag.lower() in self.tag_keep_list
                
                # drop redundant tags
                is_lucky = lambda tagname: random_obj.random() > drop_chances.get(tagname.lower(), 0.0)
                tags = [tag for tag in tags if is_lucky(tag) or is_important(tag) or is_character(tag)]
                
                # remove "comic:*", "spoiler:*", "series:*", "art-pack:*", "art pack:*"
                def is_unrelated(tag):
                    tag = tag.lower()
                    prefixes = ["fanfic:", "commissioner:", "comic:", "spoiler:", "series:", "art-pack:", "art pack:"]
                    return any([tag.startswith(prefix) for prefix in prefixes])
                tags = [tag for tag in tags if not is_unrelated(tag)]
                
                # maybe drop tags
                if self.tag_dropout_prob > 0:
                    is_lucky = lambda: random_obj.random() > self.tag_dropout_prob
                    tags = [tag for i, tag in enumerate(tags) if i<4 or is_lucky() or is_important(tag) or is_character(tag)]
                
                # if there is no artist, use "artist:unknown"
                if not any([tag.lower().startswith("artist:") for tag in tags]):
                    tags.insert(1, "artist:unknown") # insert after content rating
                
                # maybe drop all artists (so prompts without artist still produce good and varied results)
                if random_obj.random() < 0.2:
                    tags = [tag for tag in tags if not tag.startswith("artist:")]
                
                # maybe replace "artist:" with any of ["drawn by ", "created by ", "made by ", "by "]
                # e.g: "artist:unknown" -> "drawn by unknown"
                if random_obj.random() < 0.5:
                    replace_with = random_obj.choice(["drawn by ", "created by ", "made by ", "by "])
                    tags = [tag.replace("artist:", replace_with) if tag.startswith("artist:") else tag for tag in tags]
                
                # maybe shuffle tags
                if random_obj.random() < self.shuffle_tags_prob:
                    random_obj.shuffle(tags)
                
                # capitalize first letter of each name (e.g: 'twilight sparkle' -> 'Twilight Sparkle')
                # (turns out this doesn't do anything for the model, but it's still nice to have)
                tags = [tag.title() if is_character(tag) else tag for tag in tags]
                
                # maybe replace tags with their alternative names
                if self.tag_use_alias_prob:
                    for i in range(len(tags)):
                        tag = tags[i]
                        if tag in self.tagdata and random_obj.random() < self.tag_use_alias_prob:
                            aliases = self.tagdata[tag]['aliases']
                            aliases = [alias for alias in aliases if len(alias) > 3] # ignore short aliases (normally acronyms)
                            tags[i] = random_obj.choice([*aliases, tag]).replace('+', ' ').replace('-colon-', ':')
                
                # rejoin tags into comma-separated string
                example["caption"] = ", ".join(tags)
            
            # add quality rating to caption if self.scoreperc is not None
            if self.scoreperc is not None:
                img_id = os.path.splitext(os.path.basename(image_path))[0]
                if img_id in self.scoreperc.keys():
                    score_percs = self.scoreperc[img_id]
                    # {'score': float, 'wilson_score': float, 'upvotes': float, 'downvotes': float}
                    img_score = score_percs[self.scoreperc_key]
                    # add "very low quality", "low quality", "average quality", "high quality", "very high quality"
                    # (where dropout is applied to high quality tags)
                    uncond_chance = [0.9, 0.6, 0.2, 0.0, 0.0] # delete entire caption chance
                    drop_chance   = [0.0, 0.0, 0.1, 0.3, 0.5] # don't add this tag chance
                    def update_caption(example, i, tag):
                        rand_num = random_obj.random()
                        if 0 < rand_num < uncond_chance[i]:
                            example["caption"] = ""
                        elif uncond_chance[i] < rand_num < uncond_chance[i] + drop_chance[i]:
                            pass
                        elif uncond_chance[i] + drop_chance[i] < rand_num:
                            example["caption"] = rand_join(example["caption"], tag)
                        else:
                            raise Exception(f"unreachable code: {rand_num}, {uncond_chance[i]}, {drop_chance[i]}")
                    
                    if   0/5 <= img_score <= 1/5:
                        update_caption(example, 0, "very low quality")
                    elif 1/5 <= img_score <= 2/5:
                        update_caption(example, 1, "low quality")
                    elif 2/5 <= img_score <= 3/5:
                        update_caption(example, 2, "average quality")
                    elif 3/5 <= img_score <= 4/5:
                        update_caption(example, 3, "high quality")
                    elif 4/5 <= img_score <= 5/5 and random_obj.random() > 0.5:
                        update_caption(example, 4, "very high quality")
                else:
                    example["caption"] = rand_join(example["caption"], "average quality")
            
            # add "jpg" to caption if the image is a jpeg
            # Should reduce the jpeg artifacts in generated images
            if example["caption"] != "" and image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
                example["caption"] = rand_join(example["caption"], "jpg")
            
            # (maybe) drop entire caption so model can operate unconditionally.
            if random_obj.random() < self.caption_drop_prob:
                example["caption"] = ""
        
        # load image
        if 1:
            image = load_image(image_path)
            
            # LowResolutionError
            # ExtremeAspectRatioError
            if (image.size[0] * image.size[1]) < (self.size * self.size):
                raise LowResolutionError(f"Image is too small: {image.size[0]}x{image.size[1]}\npath: \"{image_path}\"")
            
            if (image.size[0] / image.size[1]) > 3 or (image.size[1] / image.size[0]) > 3:
                raise ExtremeAspectRatioError(f"Image has extreme aspect ratio: {image.size[0]}x{image.size[1]}\npath: \"{image_path}\"")
            
            # if any tag in self.text_tags, use self.flip_text_p else self.flip_p
            flip_p = self.flip_text_p if any([tag in self.text_tags for tag in tags]) else self.flip_p
            if random_obj.random() < flip_p:
                # flip image
                image = vF.hflip(image)
                # add a "hflipped" tag to caption if image is flipped
                if example.get("caption", "") != "":
                    example["caption"] = hflipped_token + example["caption"]
            
            if self.use_vertical_flip and random_obj.random() < flip_p:
                image = vF.vflip(image)
                # add a "vflipped" tag to caption if image is flipped
                if example.get("caption", "") != "":
                    example["caption"] = vflipped_token + example["caption"]
            
            if self.use_rotated_images:
                # rotate images by any right angle
                rotate_angle = random_obj.choice([0, 90, 180, 270])
                if rotate_angle != 0:
                    image = image.rotate(rotate_angle, expand=True)
                    
                    # add a "rotated" tag to caption if image is rotated
                    if example.get("caption", "") != "":
                        example["caption"] = rotated_token + example["caption"]
            
            # randomly squash image
            # (while ensuring width and height are no less than self.size)
            if self.max_image_squash > 0:
                orig_n_pixels = image.size[0] * image.size[1]
                if random_obj.random() < 0.5:
                    min_width = max(min(self.size, image.width), round(image.width * (1 - self.max_image_squash)))
                    new_width = random_obj.randint(min_width, image.width)
                    image = image.resize((new_width, image.height), resample=self.interpolation)
                else:
                    min_height = max(min(self.size, image.height), round(image.height * (1 - self.max_image_squash)))
                    new_height = random_obj.randint(min_height, image.height)
                    image = image.resize((image.width, new_height), resample=self.interpolation)
                new_n_pixels = image.size[0] * image.size[1]
                if new_n_pixels < 0.9*orig_n_pixels:
                    # if image was squashed by more than 10%, add a "squashed" tag to caption
                    if example.get("caption", "") != "":
                        example["caption"] = squashed_token + example["caption"]
            
            if not self.allow_rectangle_batches:
                model_hw = (self.size, self.size)
                model_n_pixels = model_hw[0] * model_hw[1]
                
                if self.center_crop:
                    img = np.array(image).astype(np.uint8)
                    crop = min(img.shape[0], img.shape[1])
                    img_h, img_w = img.shape[0], img.shape[1]
                    img = img[(img_h - crop) // 2:(img_h + crop) // 2,
                          (img_w - crop) // 2:(img_w + crop) // 2]
                    n_pixels = img.shape[0] * img.shape[1]
                    image = Image.fromarray(img)  # convert back to PIL image
                    # if more than 20% of image is cropped, add a "cropped" tag to caption
                    if n_pixels < 0.8*(img_h*img_w) and example.get("caption", "") != "":
                        example["caption"] = cropped_token + example["caption"]
                
                if self.max_zoom is not None and self.max_zoom > 1:
                    # resize image to a random size between full resolution and full resolution / max_zoom
                    # calc scale_min that ensures image is not upscaled
                    scale_min = max(model_hw[0] / image.width, model_hw[1] / image.height)
                    # don't zoom in more than self.max_zoom
                    scale_min = max(scale_min, 1 / self.max_zoom)
                    # sample with log distribution
                    scale = 2**random_obj.uniform(math.log2(scale_min), 0)
                    new_w = round(image.width  * scale)
                    new_h = round(image.height * scale)
                    image = image.resize((new_w, new_h), resample=self.interpolation)
                    # use RandomCrop
                    image = transforms.RandomCrop(model_hw)(image)
                
                n_pixels = img.shape[0] * img.shape[1]
                if n_pixels < model_n_pixels and example.get("caption", "") != "":
                    example["caption"] = smallres_token + example["caption"]
                
                image = image.resize(model_hw, resample=self.interpolation)
                image = np.array(image).astype(np.uint8)  # convert back to numpy array
                example["image"] = torch.from_numpy(image / 127.5 - 1.0).float()  # shape = (H, W, 3)
            else:
                image = np.array(image).astype(np.uint8)
                example["image"] = torch.from_numpy(image)
        
        return example