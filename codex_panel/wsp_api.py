"""API given to the Codex editing agents. Vision tower of CLIP ViT-B/32; text classifier is frozen.
score() = zero-shot top-1 on 5,000 train images (split A), costs 1 forward-pass budget unit."""
import os, json, torch, torch.nn.functional as F
import harness as H
from wsp import classifier, Wt0
class Session:
    def __init__(self, agent_dir=None):
        self.dir = agent_dir or os.environ.get('AGENT_DIR', '.'); self.cap = int(os.environ.get('BUDGET', 40))
        self.m = H.load_model(); self.vm = self.m.vision_model; self.proj = self.m.visual_projection
        self._orig = {n: p.detach().clone() for n, p in self.params().items()}
        X, Y = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../search_imgs.pt')); ia = [i for i in range(10000) if i % 10 < 5]
        self.X, self.Y = X[ia].cuda(), Y[ia].cuda(); self.Wc = classifier(Wt0)
        self.bf = os.path.join(self.dir, 'budget.json')
        if not os.path.exists(self.bf): json.dump({'used': 0, 'log': []}, open(self.bf, 'w'))
    def params(self):
        d = {'vision_model.' + n: p for n, p in self.vm.named_parameters()}; d['visual_projection.weight'] = self.proj.weight; return d
    def remaining(self): return self.cap - json.load(open(self.bf))['used']
    @torch.no_grad()
    def score(self, note=''):
        b = json.load(open(self.bf))
        if b['used'] >= self.cap: raise RuntimeError(f'forward-pass budget of {self.cap} exhausted')
        z = []
        with torch.autocast('cuda', dtype=torch.float16):
            for i in range(0, len(self.X), 1000): z.append(F.normalize(self.proj(self.vm(pixel_values=self.X[i:i+1000]).pooler_output).float(), dim=-1))
        a = ((torch.cat(z) @ self.Wc.T).argmax(1) == self.Y).float().mean().item() * 100
        b['used'] += 1; b['log'].append([note, round(a, 3)]); json.dump(b, open(self.bf, 'w')); print(f'score A = {a:.2f}  (budget left {self.cap-b["used"]})'); return a
    @torch.no_grad()
    def reset(self):
        for n, p in self.params().items(): p.copy_(self._orig[n])
    @torch.no_grad()
    def save(self, path=None):
        diff = {n: p.detach().cpu().clone() for n, p in self.params().items() if not torch.equal(p, self._orig[n])}
        torch.save(diff, path or os.path.join(self.dir, 'submission.pt')); print('saved', len(diff), 'changed tensors:', list(diff))
