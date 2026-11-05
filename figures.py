"""Quantitative + qualitative figures for the blog. Run from the repo root (needs typo/TB.pt, proj.pt, cache_text.pt, ui/data.json).
Everything is computed on the held-out split TB (never used to pick the heads)."""
import os, sys, json, random, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, 'ui')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import harness as H, typo, zs_meta
from wsp import classifier, Wt0
from edits import apply_ops
OUT = os.environ.get('OUT', 'docs/img'); os.makedirs(OUT, exist_ok=True); NAMES = zs_meta.IMAGENET_CLASSNAMES
m = H.load_model(); vm, proj = m.vision_model, m.visual_projection; Wc = classifier(Wt0)
orig = {n: p.detach().clone() for n, p in vm.named_parameters()}
C, A, Y = torch.load('typo/TB.pt'); C, A, Y = C.cuda(), A.cuda(), Y.cuda(); N = len(Y)
typed = torch.tensor([NAMES.index(typo.wrong_name(int(Y[i]), 200000 + i)) for i in range(N)]).cuda()
D = json.load(open('ui/data.json')); EDIT = [{'type': 'head', 'layer': 11, 'head': 8, 'mode': 'zero'}, {'type': 'head', 'layer': 8, 'head': 4, 'mode': 'zero'}]
@torch.no_grad()
def logits(X):
    z = []
    with torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(X), 1000): z.append(F.normalize(proj(vm(pixel_values=X[i:i+1000]).pooler_output).float(), dim=-1))
    return 100 * torch.cat(z) @ Wc.T
def run(ops):
    apply_ops(vm, orig, ops); a, c = logits(A), logits(C); apply_ops(vm, orig, []); return a, c
def acc(l): return (l.argmax(1) == Y).float().mean().item() * 100
R = {}
la0, lc0 = run([]); la1, lc1 = run(EDIT)
R['base'] = [acc(la0), acc(lc0)]; R['edit'] = [acc(la1), acc(lc1)]
# ---- 1. outcome table: right/wrong before -> after
def trans(l0, l1):
    r0, r1 = l0.argmax(1) == Y, l1.argmax(1) == Y
    return {'fixed': int((~r0 & r1).sum()), 'broken': int((r0 & ~r1).sum()), 'both_right': int((r0 & r1).sum()), 'both_wrong': int((~r0 & ~r1).sum()),
            'wrong_to_written_word_before': int((l0.argmax(1) == typed).sum()), 'wrong_to_written_word_after': int((l1.argmax(1) == typed).sum())}
R['trans_attacked'] = trans(la0, la1); R['trans_clean'] = trans(lc0, lc1); R['n'] = N
# ---- 2. heads switched off, ranked by the single-head scan on the SEARCH split
b = D['base'][0]; rank = sorted([(D['scan'][f'H{l}.{h}'][0] - b, l, h) for l in range(12) for h in range(12)], reverse=True)
curve = []
for k in range(0, 13):
    a, c = run([{'type': 'head', 'layer': l, 'head': h, 'mode': 'zero'} for _, l, h in rank[:k]]); curve.append([k, acc(a), acc(c)])
R['curve_k'] = curve; R['rank'] = [[l, h] for _, l, h in rank[:12]]
sweep = []
for s in [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0]:
    a, c = run([{'type': 'head', 'layer': 11, 'head': 8, 'mode': 'scale', 'value': s}, {'type': 'head', 'layer': 8, 'head': 4, 'mode': 'scale', 'value': s}]); sweep.append([s, acc(a), acc(c)])
R['sweep'] = sweep
col = {'att': '#2a5bd7', 'cl': '#d9632b'}
def lineplot(xs, ya, yc, xl, path, rev=False):
    f, ax = plt.subplots(figsize=(6, 3.4)); ax.plot(xs, ya, '-o', color=col['att'], ms=4); ax.plot(xs, yc, '-o', color=col['cl'], ms=4)
    ax.text(xs[-1], ya[-1] + 1.5, 'attacked', color=col['att'], ha='right', va='bottom', fontsize=10); ax.text(xs[-1], yc[-1] - 1.5, 'clean', color=col['cl'], ha='right', va='top', fontsize=10)
    ax.set_xlabel(xl); ax.set_ylabel('top-1 accuracy (%)'); ax.set_ylim(15, 75); ax.grid(alpha=.25); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    if rev: ax.invert_xaxis()
    f.tight_layout(); f.savefig(path, dpi=160); plt.close(f)
lineplot([r[0] for r in curve], [r[1] for r in curve], [r[2] for r in curve], 'number of heads switched off (best single heads first)', f'{OUT}/fig_heads_off.png')
lineplot([r[0] for r in sweep], [r[1] for r in sweep], [r[2] for r in sweep], 'strength kept in L11 head 8 and L8 head 4 (1 = original, 0 = off)', f'{OUT}/fig_strength.png', rev=True)
# ---- 3. attention: how much of the head's CLS attention lands on the written-text box
import typo as T
def box(text, seed):
    r = random.Random(seed); d = ImageDraw.Draw(Image.new('RGB', (224, 224))); size = 26
    while True:
        f = ImageFont.truetype(T.FONT, size); x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
        if x1 - x0 <= 200 or size <= 12: break
        size -= 2
    w, h = x1 - x0, y1 - y0; px = r.randint(4, max(4, 224 - w - 10)); py = r.randint(4, max(4, 224 - h - 12)); return px - 4, py - 4, px + w + 4, py + h + 6
cap = {}
for l in (8, 11): vm.encoder.layers[l].layer_norm1.register_forward_hook(lambda mod, a, o, l=l: cap.__setitem__(l, o.detach()))
@torch.no_grad()
def attn(X, l):
    with torch.autocast('cuda', dtype=torch.float16): vm(pixel_values=X)
    x = cap[l].float(); at = vm.encoder.layers[l].self_attn; n = len(X)
    q = at.q_proj(x[:, :1]).view(n, 1, 12, 64).transpose(1, 2); k = at.k_proj(x).view(n, 50, 12, 64).transpose(1, 2)
    return ((q @ k.transpose(-1, -2)) * 64 ** -0.5).softmax(-1)[:, :, 0, 1:]  # [n,12,49]
idx = list(range(0, N, 8))[:500]; masks = []
for i in idx:
    x0, y0, x1, y1 = box(NAMES[int(typed[i])], 200000 + i); mk = torch.zeros(7, 7)
    for r_ in range(7):
        for c_ in range(7):
            if c_ * 32 < x1 and (c_ + 1) * 32 > x0 and r_ * 32 < y1 and (r_ + 1) * 32 > y0: mk[r_, c_] = 1
    masks.append(mk.flatten())
masks = torch.stack(masks).cuda(); share = {}
for l, h in [(11, 8), (8, 4)]:
    a = attn(A[idx], l); ac = attn(C[idx], l)
    share[f'L{l}H{h}'] = {'attacked_on_text_cells': (a[:, h] * masks).sum(1).mean().item(), 'clean_on_same_cells': (ac[:, h] * masks).sum(1).mean().item(), 'cells_share_of_image': masks.mean().item(),
                          'other_heads_same_layer_attacked': ((a * masks[:, None]).sum(2).sum(1) - (a[:, h] * masks).sum(1)).mean().item() / 11}
R['attention_share'] = share
sel = [3, 17, 29, 41]; sel = [idx[s] for s in sel]
def den(t): return (t.float().cpu() * torch.tensor(H.STD).view(3, 1, 1) + torch.tensor(H.MEAN).view(3, 1, 1)).clamp(0, 1).permute(1, 2, 0).numpy()
f, axs = plt.subplots(len(sel), 3, figsize=(7.5, 2.5 * len(sel)))
for r_, i in enumerate(sel):
    a8, a11 = attn(A[i:i+1], 8)[0, 4], attn(A[i:i+1], 11)[0, 8]
    axs[r_, 0].imshow(den(A[i])); axs[r_, 0].set_title(f'true: {NAMES[int(Y[i])]}\nwritten: {NAMES[int(typed[i])]}', fontsize=8)
    for c_, (aa, t) in enumerate([(a11, 'L11 head 8: where it looks'), (a8, 'L8 head 4: where it looks')], 1):
        axs[r_, c_].imshow(den(A[i])); axs[r_, c_].imshow(F.interpolate(aa.view(1, 1, 7, 7).float(), size=224, mode='bilinear')[0, 0].cpu(), alpha=.6, cmap='inferno'); axs[r_, c_].set_title(t, fontsize=8)
    for ax in axs[r_]: ax.axis('off')
f.tight_layout(); f.savefig(f'{OUT}/fig_attention.png', dpi=150); plt.close(f)
# ---- 4. random before/after examples (fixed seed, NOT chosen by outcome)
rnd = random.Random(0); ex = sorted(rnd.sample(range(N), 12)); f, axs = plt.subplots(2, 6, figsize=(16, 5.4))
for ax, i in zip(axs.flatten(), ex):
    p0, p1 = la0[i].softmax(0), la1[i].softmax(0); j0, j1 = int(p0.argmax()), int(p1.argmax()); ok = lambda j: '✓' if j == int(Y[i]) else '✗'
    ax.imshow(den(A[i])); ax.axis('off')
    ax.set_title(f'true: {NAMES[int(Y[i])]}\nwritten: {NAMES[int(typed[i])]}', fontsize=7.5)
    ax.text(0.5, -0.04, f'before: {NAMES[j0]} {p0[j0]*100:.0f}% {ok(j0)}\nafter: {NAMES[j1]} {p1[j1]*100:.0f}% {ok(j1)}', transform=ax.transAxes, ha='center', va='top', fontsize=6.8, color=('#1a7f4b' if (j1 == int(Y[i])) else '#c0392b'))
f.tight_layout(); f.subplots_adjust(hspace=.4, wspace=.25); f.savefig(f'{OUT}/fig_examples.png', dpi=150); plt.close(f)
# ---- 5. animation: one attacked image, both heads turned down from 1 to 0
fixed = [i for i in range(N) if int(la0[i].argmax()) != int(Y[i]) and int(la1[i].argmax()) == int(Y[i])]; gi = fixed[7]; frames = []
for s in [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0]:
    apply_ops(vm, orig, [{'type': 'head', 'layer': 11, 'head': 8, 'mode': 'scale', 'value': s}, {'type': 'head', 'layer': 8, 'head': 4, 'mode': 'scale', 'value': s}]); p = logits(A[gi:gi+1])[0].softmax(0); apply_ops(vm, orig, [])
    show = [int(Y[gi]), int(typed[gi])]; extra = [j for j in p.topk(4).indices.tolist() if j not in show][:2]; ids = show + extra
    f, (a0, a1) = plt.subplots(1, 2, figsize=(7.5, 2.8), gridspec_kw={'width_ratios': [1, 2.2]}); a0.imshow(den(A[gi])); a0.axis('off')
    a1.barh([NAMES[j] for j in ids][::-1], [p[j].item() * 100 for j in ids][::-1], color=['#999'] * len(extra) + ['#d9632b', '#2a5bd7']); a1.set_xlim(0, 100); a1.set_xlabel('probability (%)')
    a1.set_title(f'heads kept at {s:.1f}  (blue = true class, orange = written word)', fontsize=8.5); [a1.spines[q].set_visible(False) for q in ('top', 'right')]
    f.tight_layout(); f.canvas.draw(); frames.append(Image.fromarray(np.asarray(f.canvas.buffer_rgba())[..., :3])); plt.close(f)
frames[0].save(f'{OUT}/sweep.gif', save_all=True, append_images=frames[1:] + frames[-1:] * 4, duration=450, loop=0)
json.dump(R, open(os.path.join(OUT, '..', 'results.json'), 'w'), indent=1); print(json.dumps({k: R[k] for k in ['base', 'edit', 'trans_attacked', 'trans_clean', 'attention_share']}, indent=1)); print('curve', [(c[0], round(c[1], 1), round(c[2], 1)) for c in curve]); print('rank', R['rank'])
