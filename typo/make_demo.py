import torch, torch.nn.functional as F, harness as H, typo, zs_meta
from PIL import Image, ImageDraw, ImageFont
from wsp import classifier, Wt0
m = H.load_model(); vm, proj = m.vision_model, m.visual_projection; Wc = classifier(Wt0)
P = {('vision_model.' + n): p for n, p in vm.named_parameters()}; P['visual_projection.weight'] = proj.weight
orig = {n: p.detach().clone() for n, p in P.items()}; sub = torch.load('typo/agents/heads/submission.pt')
def setw(edited):
    with torch.no_grad():
        for n, p in P.items(): p.copy_(orig[n])
        if edited:
            for n, t in sub.items(): P[n].copy_(t.to(p.device))
@torch.no_grad()
def pred(x):
    with torch.autocast('cuda', dtype=torch.float16): z = F.normalize(proj(vm(pixel_values=x.cuda()).pooler_output).float(), dim=-1)
    return (z @ Wc.T).argmax(1).cpu()
items = H.val_items(); N = zs_meta.IMAGENET_CLASSNAMES; idx = list(range(0, 50000, 167))[:300]
cl, at = zip(*[typo.make(items[i][0], items[i][1], 300000 + i) for i in idx]); cl, at = torch.stack(cl), torch.stack(at); y = torch.tensor([items[i][1] for i in idx])
setw(False); pc0, pa0 = pred(cl), pred(at); setw(True); pa1 = pred(at)
good = [k for k in range(len(idx)) if pc0[k] == y[k] and pa0[k] != y[k] and pa1[k] == y[k]]
print('clean-right & fooled & fixed:', len(good), 'of', len(idx))
seen, pick = set(), []
for k in good:
    if int(y[k]) not in seen: seen.add(int(y[k])); pick.append(k)
    if len(pick) == 2: break
f = ImageFont.truetype(typo.FONT, 15); out = []
for k in pick:
    i = idx[k]; p = items[i][0]; im = typo.base(Image.open(p).convert('RGB')); w = typo.wrong_name(items[i][1], 300000 + i)
    canvas = Image.new('RGB', (224 * 2 + 30, 224 + 90), 'white'); canvas.paste(im, (10, 10)); canvas.paste(typo.attack(im, w, 300000 + i), (244, 10)); d = ImageDraw.Draw(canvas)
    d.text((10, 238), f'clean   (true: {N[int(y[k])]})', fill='black', font=f)
    d.text((244, 238), f'attacked, text says "{w}"', fill='black', font=f)
    d.text((244, 258), f'original -> {N[int(pa0[k])]}', fill=(190, 0, 0), font=f)
    d.text((244, 278), f'edited   -> {N[int(pa1[k])]}', fill=(0, 130, 0), font=f)
    out.append(canvas); print(N[int(y[k])], '| written', w, '| original:', N[int(pa0[k])], '| edited:', N[int(pa1[k])])
W = Image.new('RGB', (out[0].width, sum(o.height for o in out)), 'white'); yy = 0
for o in out: W.paste(o, (0, yy)); yy += o.height
W.save('typo/demo/attack_demo.png')
