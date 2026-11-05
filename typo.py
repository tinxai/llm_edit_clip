"""Typographic-attack data: write a WRONG class name on the (resized+center-cropped) 224px image."""
import os, random, glob, torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import harness as H, zs_meta
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
base = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])
norm = T.Compose([T.ToTensor(), T.Normalize(H.MEAN, H.STD)])
def attack(img224, text, seed):
    """img224: PIL 224x224. Draws `text` in a white box with black text at a seeded random position."""
    r = random.Random(seed); im = img224.copy(); d = ImageDraw.Draw(im)
    size = 26
    while True:
        f = ImageFont.truetype(FONT, size); x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
        if x1 - x0 <= 200 or size <= 12: break
        size -= 2
    w, h = x1 - x0, y1 - y0; px = r.randint(4, max(4, 224 - w - 10)); py = r.randint(4, max(4, 224 - h - 12))
    d.rectangle([px - 4, py - 4, px + w + 4, py + h + 6], fill=(255, 255, 255)); d.text((px - x0, py - y0), text, font=f, fill=(0, 0, 0))
    return im
def wrong_name(label, seed):
    r = random.Random(seed * 7919 + 1); j = r.randrange(999); j = j + 1 if j >= label else j
    return zs_meta.IMAGENET_CLASSNAMES[j]
def make(path, label, seed):
    im = base(Image.open(path).convert('RGB')); return norm(im), norm(attack(im, wrong_name(label, seed), seed))  # (clean, attacked)
def train_split(lo, hi): # train files [lo:hi) per class
    out = []
    for c, w in enumerate(H.wnids): out += [(f'{H.TRAIN}/{w}/{f}', c) for f in sorted(os.listdir(f'{H.TRAIN}/{w}'))[lo:hi]]
    return out
class DS(torch.utils.data.Dataset):
    def __init__(s, items, seed0): s.items, s.seed0 = items, seed0
    def __len__(s): return len(s.items)
    def __getitem__(s, i): p, y = s.items[i]; c, a = make(p, y, s.seed0 + i); return c, a, y
