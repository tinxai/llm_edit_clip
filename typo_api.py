"""API for the Claude editing agents (typographic-attack task). CLIP ViT-B/32 vision tower; text classifier frozen."""
import os, json, torch, torch.nn.functional as F
import harness as H
from wsp import classifier, Wt0
class Session:
    def __init__(self, agent_dir=None):
        self.dir = agent_dir or os.environ.get('AGENT_DIR', '.'); self.cap = int(os.environ.get('BUDGET', 40))
        self.m = H.load_model(); self.vm = self.m.vision_model; self.proj = self.m.visual_projection
        self._orig = {n: p.detach().clone() for n, p in self.params().items()}
        C, A, Y = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'typo/TA.pt')); self.C, self.A, self.Y = C.cuda(), A.cuda(), Y.cuda()
        self.Wc = classifier(Wt0); self.bf = os.path.join(self.dir, 'budget.json')
        if not os.path.exists(self.bf): json.dump({'used': 0, 'log': []}, open(self.bf, 'w'))
    def params(self):
        d = {'vision_model.' + n: p for n, p in self.vm.named_parameters()}; d['visual_projection.weight'] = self.proj.weight; return d
    def remaining(self): return self.cap - json.load(open(self.bf))['used']
    @torch.no_grad()
    def encode(self, x):
        """FREE forward for SYNTHETIC images you build yourself (normalized tensors [N,3,224,224]); returns normalized embeddings.
        Never feed it self.C/self.A/self.Y - eval images only go through score()."""
        with torch.autocast('cuda', dtype=torch.float16):
            return F.normalize(self.proj(self.vm(pixel_values=x.cuda().half() if x.dtype != torch.float32 else x.cuda()).pooler_output).float(), dim=-1)
    @torch.no_grad()
    def _acc(self, X):
        z = []
        with torch.autocast('cuda', dtype=torch.float16):
            for i in range(0, len(X), 1000): z.append(F.normalize(self.proj(self.vm(pixel_values=X[i:i+1000]).pooler_output).float(), dim=-1))
        return ((torch.cat(z) @ self.Wc.T).argmax(1) == self.Y).float().mean().item() * 100
    def score(self, note=''):
        """returns (attacked_acc, clean_acc) on 4000 train images; costs 1 of the HARD budget."""
        b = json.load(open(self.bf))
        if b['used'] >= self.cap: raise RuntimeError(f'budget of {self.cap} exhausted')
        a, c = self._acc(self.A), self._acc(self.C); b['used'] += 1; b['log'].append([note, round(a, 2), round(c, 2)]); json.dump(b, open(self.bf, 'w'))
        print(f'attacked {a:.2f} | clean {c:.2f}   (budget left {self.cap-b["used"]})'); return a, c
    @torch.no_grad()
    def reset(self):
        for n, p in self.params().items(): p.copy_(self._orig[n])
    @torch.no_grad()
    def save(self, path=None):
        diff = {n: p.detach().cpu().clone() for n, p in self.params().items() if not torch.equal(p, self._orig[n])}
        torch.save(diff, path or os.path.join(self.dir, 'submission.pt')); print('saved', len(diff), 'changed tensors:', list(diff))
