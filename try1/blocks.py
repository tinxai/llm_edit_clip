"""Edits inside the vision transformer blocks. Search = 10k train imgs (halves A/B); val only for the final report."""
import os, json, time, torch, torch.nn.functional as F, harness as H
from wsp import classifier, Wt0
dev = 'cuda'
m = H.load_model(); vm = m.vision_model; Wc = classifier(Wt0)  # text side frozen (original)
if not os.path.exists('search_imgs.pt'):
    X, Y = [], []
    for x, y in H.loader(H.search_items()): X.append(x.half()); Y.append(y)
    torch.save((torch.cat(X), torch.cat(Y)), 'search_imgs.pt')
X, Y = torch.load('search_imgs.pt'); X, Y = X.to(dev), Y.to(dev)
ia = torch.tensor([i for i in range(10000) if i % 10 < 5], device=dev); ib = torch.tensor([i for i in range(10000) if i % 10 >= 5], device=dev)
@torch.no_grad()
def feats(X):
    out = []
    with torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(X), 1000):
            out.append(F.normalize(m.visual_projection(vm(pixel_values=X[i:i+1000]).pooler_output).float(), dim=-1))
    return torch.cat(out)
def evalAB():
    z = feats(X); c = (z @ Wc.T).argmax(1) == Y
    return c[ia].float().mean().item()*100, c[ib].float().mean().item()*100
L = vm.encoder.layers
def target(kind, l, d):  # returns (param, slicer)
    if kind == 'head': return L[l].self_attn.out_proj.weight, (slice(None), slice(d*64, (d+1)*64))
    if kind == 'attn': return L[l].self_attn.out_proj.weight, (slice(None), slice(None))
    if kind == 'mlp':  return L[l].mlp.fc2.weight, (slice(None), slice(None))
    if kind == 'neuron': return L[l].mlp.fc2.weight, (slice(None), d)
def apply(e, undo=False):
    kind, l, d, s = e; p, sl = target(kind, l, d)
    with torch.no_grad(): p[sl] *= (1/s if undo else s) if s != 0 else 1
def trial(e):
    kind, l, d, s = e; p, sl = target(kind, l, d)
    with torch.no_grad():
        orig = p[sl].clone(); p[sl] *= s; r = evalAB(); p[sl] = orig
    return r
t = time.time(); a0, b0 = evalAB(); print(f'baseline A {a0:.2f} B {b0:.2f}  ({time.time()-t:.1f}s/eval)', flush=True)
cands = [('head', l, d, s) for l in range(12) for d in range(12) for s in (0.0, 0.5, 1.5)]
cands += [(k, l, 0, s) for k in ('attn', 'mlp') for l in range(12) for s in (0.5, 0.8, 1.2)]
res = []
for i, e in enumerate(cands):
    a, b = trial(e); res.append((e, a - a0, b - b0))
    if i % 50 == 0: print(i, len(cands), e, f'dA {a-a0:+.2f} dB {b-b0:+.2f}', flush=True)
json.dump(res, open('blocks_screen.json', 'w'))
# greedy: apply screened edits in order of (dA+dB) if still improving A and not hurting B
cur_a, cur_b = a0, b0; kept = []
for e, da, db in sorted([r for r in res if r[1] > 0 and r[2] >= 0], key=lambda r: -(r[1]+r[2]))[:30]:
    kind, l, d, s = e; p, sl = target(kind, l, d)
    with torch.no_grad(): orig = p[sl].clone(); p[sl] *= s
    a, b = evalAB()
    if a > cur_a and b >= cur_b: cur_a, cur_b = a, b; kept.append(e); print('KEEP', e, f'A {a:.2f} B {b:.2f}', flush=True)
    else:
        with torch.no_grad(): p[sl] = orig
torch.save({'state': {k: v.cpu() for k, v in m.state_dict().items()}, 'kept': kept}, 'edited_blocks.pt')
# FINAL: full ImageNet val, edited vs original weights
@torch.no_grad()
def val_acc():
    z, y = [], []
    for x, yy in H.loader(H.val_items()):
        with torch.autocast('cuda', dtype=torch.float16): z.append(F.normalize(m.visual_projection(vm(pixel_values=x.to(dev)).pooler_output).float(), dim=-1))
        y.append(yy)
    z = torch.cat(z); y = torch.cat(y).to(dev); return ((z @ Wc.T).argmax(1) == y).float().mean().item()*100
print('kept edits:', kept, flush=True)
print('VAL edited %.2f' % val_acc(), flush=True)
