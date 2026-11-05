"""Eval harness for editing CLIP ViT-B/32 weights. Run with CUDA_VISIBLE_DEVICES=0."""
import os, glob, torch, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPTokenizer
import torchvision.transforms as T
import zs_meta

VAL = os.environ.get('IMAGENET_VAL', '/path/to/ILSVRC2012_img_val/val')
TRAIN = os.environ.get('IMAGENET_TRAIN', '/path/to/ILSVRC2012_img_train')
HF = 'openai/clip-vit-base-patch32'
dev = 'cuda'
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                T.ToTensor(), T.Normalize(MEAN, STD)])
wnids = sorted(os.listdir(VAL))

class DS(Dataset):
    def __init__(s, items): s.items = items
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, y = s.items[i]
        return tf(Image.open(p).convert('RGB')), y

def val_items():
    return [(p, c) for c, w in enumerate(wnids) for p in sorted(glob.glob(f'{VAL}/{w}/*.JPEG'))]

def search_items(per_class=10):  # held-out-from-val search set drawn from TRAIN
    out = []
    for c, w in enumerate(wnids):
        fs = sorted(os.listdir(f'{TRAIN}/{w}'))[:per_class]
        out += [(f'{TRAIN}/{w}/{f}', c) for f in fs]
    return out

def load_model():
    m = CLIPModel.from_pretrained(HF).to(dev).eval()
    return m

@torch.no_grad()
def text_classifier(m, templates=None):
    tok = CLIPTokenizer.from_pretrained(HF)
    templates = templates or zs_meta.OPENAI_IMAGENET_TEMPLATES
    W = []
    for name in zs_meta.IMAGENET_CLASSNAMES:
        t = tok([tp(name) for tp in templates], padding='max_length', max_length=77, return_tensors='pt').to(dev)
        e = m.get_text_features(**t)
        e = e.pooler_output if hasattr(e, 'pooler_output') else e
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        W.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.stack(W)  # [1000, 512]

@torch.no_grad()
def encode_images(m, loader):
    F, Y = [], []
    for x, y in loader:
        f = m.get_image_features(pixel_values=x.to(dev))
        f = f.pooler_output if hasattr(f, 'pooler_output') else f
        F.append(torch.nn.functional.normalize(f.float(), dim=-1)); Y.append(y)
    return torch.cat(F), torch.cat(Y).to(dev)

def top1(feats, labels, W):
    return (feats @ W.T).argmax(1).eq(labels).float().mean().item() * 100

def loader(items, bs=500, nw=16):
    return DataLoader(DS(items), batch_size=bs, num_workers=nw, shuffle=False)
