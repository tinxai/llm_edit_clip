"""Local app: python ui/server.py  (GPU via CUDA_VISIBLE_DEVICES). Serves the page + live scoring of edits on held-out split TB."""
import sys, os, json, re, subprocess, threading, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'ui')
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import harness as H
from wsp import classifier, Wt0
from edits import apply_ops
m = H.load_model(); vm, proj = m.vision_model, m.visual_projection; Wc = classifier(Wt0)
orig = {n: p.detach().clone() for n, p in vm.named_parameters()}
C, A, Y = torch.load('typo/TB.pt'); C, A, Y = C.cuda(), A.cuda(), Y.cuda(); LOCK = threading.Lock()
DATA = json.load(open('ui/data.json'))
@torch.no_grad()
def acc(X):
    z = []
    with torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(X), 1000): z.append(F.normalize(proj(vm(pixel_values=X[i:i+1000]).pooler_output).float(), dim=-1))
    return ((torch.cat(z) @ Wc.T).argmax(1) == Y).float().mean().item() * 100
def score(ops):
    with LOCK:
        apply_ops(vm, orig, ops); r = {'attacked': acc(A), 'clean': acc(C)}; apply_ops(vm, orig, []); return r
BASE = score([]); print('baseline', BASE, flush=True)
def context():
    sc = DATA['scan']; b = DATA['base']; rows = []
    for k, (a, c) in sc.items(): rows.append((a - b[0], c - b[1], k))
    rows.sort(reverse=True); out = ['Switching a part OFF changes accuracy on the search split (attacked / clean), baseline attacked %.1f clean %.1f. Names: H<layer>.<head> = attention head, M<layer> = MLP block. Layers 0-11.' % tuple(b)]
    out += ['%s: attacked %+.1f, clean %+.1f' % (k, da, dc) for da, dc, k in rows[:12]]
    ro = DATA['readout']; flat = sorted(((ro[l][h], f'H{l}.{h}') for l in range(12) for h in range(12)), reverse=True)[:8]
    out.append('Heads that most push the model toward the word written on the image: ' + ', '.join(f'{k} ({v:.2f})' for v, k in flat))
    return '\n'.join(out)
CTX = context()
def chat(msg, cur):
    prompt = f"""You are the edit assistant in a UI for editing the weights of CLIP ViT-B/32 (vision tower) with no training. Task context: some ImageNet images have a WRONG class name written on them (typographic attack), which fools CLIP.
{CTX}
Current edits applied: {json.dumps(cur)}
User request: {msg}
Allowed edit ops (JSON): {{"type":"head","layer":0-11,"head":0-11,"mode":"zero"}} | {{"type":"head",...,"mode":"scale","value":0.0-2.0}} | {{"type":"mlp","layer":0-11,"mode":"zero"|"scale","value":..}}.
Use ONLY the evidence above; do not invent facts about parts not listed. Reply with ONLY a JSON object: {{"reply": "<2-3 plain sentences: what you propose and why, citing the numbers>", "ops": [<full list of edits to apply, including any current ones you keep>]}}"""
    r = subprocess.run(['claude', '-p', '--model', 'sonnet', '--tools', ''], input=prompt, capture_output=True, text=True, timeout=180, cwd='/tmp')
    t = r.stdout.strip(); mt = re.search(r'\{.*\}', t, re.S)
    if not mt: return {'reply': 'LLM gave no JSON: ' + (t or r.stderr)[:300], 'ops': cur}
    o = json.loads(mt.group(0)); ops = [x for x in o.get('ops', []) if x.get('type') in ('head', 'mlp')]
    return {'reply': o.get('reply', ''), 'ops': ops}
class Hd(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, body, ct='application/json', code=200):
        b = body if isinstance(body, bytes) else json.dumps(body).encode(); self.send_response(code); self.send_header('Content-Type', ct); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p = self.path.split('?')[0]; p = '/index.html' if p == '/' else p
        if p == '/base': return self.send(BASE)
        f = os.path.normpath('ui' + p)
        if not f.startswith('ui/') or not os.path.isfile(f) or f.endswith('.py'): return self.send({'error': 'nf'}, code=404)
        ct = {'html': 'text/html', 'json': 'application/json', 'png': 'image/png'}.get(f.rsplit('.', 1)[-1], 'application/octet-stream'); self.send(open(f, 'rb').read(), ct)
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        try:
            if self.path == '/score': self.send(score(d['ops']))
            elif self.path == '/chat': self.send(chat(d['message'], d['ops']))
        except Exception as e: self.send({'error': str(e)}, code=500)
ThreadingHTTPServer(('127.0.0.1', int(os.environ.get('PORT', 8765))), Hd).serve_forever()
