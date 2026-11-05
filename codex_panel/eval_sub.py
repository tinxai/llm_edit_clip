"""Grade panel submissions. usage: eval_sub.py name1 name2 ...  (names of dirs under panel/)"""
import sys, os, json, torch, torch.nn.functional as F, harness as H
from wsp import classifier, Wt0
m = H.load_model(); vm, proj = m.vision_model, m.visual_projection; Wc = classifier(Wt0)
P = {('vision_model.' + n): p for n, p in vm.named_parameters()}; P['visual_projection.weight'] = proj.weight
orig = {n: p.detach().clone() for n, p in P.items()}
X, Y = torch.load('search_imgs.pt'); ib = [i for i in range(10000) if i % 10 >= 5]; XB, YB = X[ib].cuda(), Y[ib].cuda()
@torch.no_grad()
def enc(x):
    with torch.autocast('cuda', dtype=torch.float16): return F.normalize(proj(vm(pixel_values=x).pooler_output).float(), dim=-1)
@torch.no_grad()
def accB(): return ((torch.cat([enc(XB[i:i+1000]) for i in range(0, 5000, 1000)]) @ Wc.T).argmax(1) == YB).float().mean().item()*100
@torch.no_grad()
def accVal():
    z, y = [], []
    for x, yy in H.loader(H.val_items()): z.append(enc(x.cuda())); y.append(yy)
    return ((torch.cat(z) @ Wc.T).argmax(1) == torch.cat(y).cuda()).float().mean().item()*100
names = [n for n in sys.argv[1:] if os.path.exists(f'panel/{n}/submission.pt')]
variants = {'original': {}}; variants.update({n: torch.load(f'panel/{n}/submission.pt') for n in names})
def setw(v):
    with torch.no_grad():
        for n, p in P.items(): p.copy_(orig[n])
        for n, t in variants[v].items(): P[n].copy_(t.to(P[n].device))
res = {}
for v in variants:
    setw(v); res[v] = {'B': accB(), 'tensors': list(variants[v]), 'entries_changed': sum(int((t.cpu() != orig[k].cpu()).sum()) for k, t in variants[v].items())}; print(v, 'B %.2f' % res[v]['B'], flush=True)
corr = {v: 0 for v in variants}; tot = 0
with torch.no_grad():
    for x, yy in H.loader(H.val_items()):
        x = x.cuda(); yy = yy.cuda(); tot += len(yy)
        for v in variants: setw(v); corr[v] += int(((enc(x) @ Wc.T).argmax(1) == yy).sum())
for v in variants: res[v]['val'] = corr[v] / tot * 100; print(v, 'VAL %.2f' % res[v]['val'], flush=True)
json.dump(res, open('panel/results.json', 'w'), indent=1)
