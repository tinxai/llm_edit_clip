"""Grade submissions: usage eval_typo.py name... (dirs under typo/agents/). Reports attacked+clean acc on held-out TB and ImageNet val."""
import sys, os, json, torch, torch.nn.functional as F, harness as H, typo
from wsp import classifier, Wt0
from torch.utils.data import DataLoader
m = H.load_model(); vm, proj = m.vision_model, m.visual_projection; Wc = classifier(Wt0)
P = {('vision_model.' + n): p for n, p in vm.named_parameters()}; P['visual_projection.weight'] = proj.weight
orig = {n: p.detach().clone() for n, p in P.items()}
AD = os.environ.get('AGENTS_DIR', 'typo/agents'); RES = os.environ.get('RESULTS', 'typo/results.json')
names = [n for n in sys.argv[1:] if os.path.exists(f'{AD}/{n}/submission.pt')]
variants = {'original': {}}; variants.update({n: torch.load(f'{AD}/{n}/submission.pt') for n in names})
def setw(v):
    with torch.no_grad():
        for n, p in P.items(): p.copy_(orig[n])
        for n, t in variants[v].items(): P[n].copy_(t.to(P[n].device))
@torch.no_grad()
def enc(x):
    with torch.autocast('cuda', dtype=torch.float16): return F.normalize(proj(vm(pixel_values=x).pooler_output).float(), dim=-1)
@torch.no_grad()
def hits(X, Y): return int(((torch.cat([enc(X[i:i+1000]) for i in range(0, len(X), 1000)]) @ Wc.T).argmax(1) == Y).sum())
C, A, Y = torch.load('typo/TB.pt'); C, A, Y = C.cuda(), A.cuda(), Y.cuda(); res = {}
for v in variants:
    setw(v); res[v] = {'TB_clean': hits(C, Y)/len(Y)*100, 'TB_attacked': hits(A, Y)/len(Y)*100, 'tensors': list(variants[v]), 'entries_changed': sum(int((t.cpu() != orig[k].cpu()).sum()) for k, t in variants[v].items()), 'entries_set_to_zero': sum(int(((t.cpu() != orig[k].cpu()) & (t.cpu() == 0)).sum()) for k, t in variants[v].items())}; print(v, res[v], flush=True)
# val: clean on all 50k; attacked on the first 10 val images of each class (10k)
items = H.val_items(); cnt = {}; sub = []
for i, (p, y) in enumerate(items): cnt[y] = cnt.get(y, 0) + 1; sub.append(cnt[y] <= 10)
class VD(torch.utils.data.Dataset):
    def __len__(s): return len(items)
    def __getitem__(s, i):
        p, y = items[i]; c, a = typo.make(p, y, 300000 + i); return c, a, y, sub[i]
vc = {v: 0 for v in variants}; va = {v: 0 for v in variants}; n_sub = 0
with torch.no_grad():
    for c, a, y, s in DataLoader(VD(), batch_size=500, num_workers=16):
        c, a, y, s = c.cuda(), a.cuda(), y.cuda(), s.cuda(); n_sub += int(s.sum())
        for v in variants:
            setw(v); vc[v] += int(((enc(c) @ Wc.T).argmax(1) == y).sum())
            if s.any(): va[v] += int(((enc(a[s]) @ Wc.T).argmax(1) == y[s]).sum())
for v in variants: res[v]['val_clean'] = vc[v]/len(items)*100; res[v]['val_attacked'] = va[v]/n_sub*100; print(v, 'VAL clean %.2f attacked %.2f' % (res[v]['val_clean'], res[v]['val_attacked']), flush=True)
json.dump(res, open(RES, 'w'), indent=1)
