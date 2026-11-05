import torch, harness as H, zs_meta
from transformers import CLIPTokenizer
m = H.load_model()
@torch.no_grad()
def img_pooled(items):
    out, ys = [], []
    for x, y in H.loader(items):
        o = m.vision_model(pixel_values=x.to(H.dev)).pooler_output  # post-LN CLS, 768-d, pre-projection
        out.append(o.float().cpu()); ys.append(y)
    return torch.cat(out), torch.cat(ys)
tok = CLIPTokenizer.from_pretrained(H.HF)
T = zs_meta.OPENAI_IMAGENET_TEMPLATES
with torch.no_grad():
    tp = []
    for n in zs_meta.IMAGENET_CLASSNAMES:
        t = tok([f(n) for f in T], padding='max_length', max_length=77, return_tensors='pt').to(H.dev)
        tp.append(m.text_model(**t).pooler_output.float().cpu())  # [80,512]
torch.save(torch.stack(tp), 'cache_text.pt')   # [1000,80,512]
for name, items in [('search', H.search_items()), ('val', H.val_items())]:
    f, y = img_pooled(items); torch.save((f, y), f'cache_{name}.pt'); print(name, f.shape, flush=True)
torch.save({'Wv': m.visual_projection.weight.detach().cpu(), 'Wt': m.text_projection.weight.detach().cpu()}, 'proj.pt')
