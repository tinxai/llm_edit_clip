# LLM-edited CLIP

Can an LLM edit CLIP's weights directly (no gradients) to fix a weakness? Experiments with CLIP ViT-B/32, Claude and Codex agents, and a small editor UI.
Inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). **Blog / write-up: `docs/index.html`** (enable GitHub Pages on `/docs` to serve it).

## Summary
- Generic ImageNet accuracy: no real gain (all edits within noise on the 50k val images).
- Typographic attacks (wrong class name written on the image): zeroing two attention heads (`edits/zero_two_heads.py`) raises attacked-image accuracy 26.9% -> 42.4% on val, at about -0.6 points clean.
- Agents forced to write new values (new neuron pathway, bias writes, ...) reached 40-41.6%, not better than zeroing.
- This is a known type of fix, see [Dyslexify](https://hufe.info/dyslexify/) (Hufe et al., ICLR 2026). We have not compared against their method.

## Layout

- `harness.py`, `zs_meta.py`, `wsp.py`, `cache_feats.py`: CLIP eval, class names and 80 templates, cached features
- `typo.py`, `build_typo.py`, `typo_api.py`, `eval_typo.py`: attack renderer, attacked train splits, the budgeted API the agents used, and the grader
- `typo/brief.md`, `typo/brief2.md`, `typo/results.json`, `typo/results2.json`: agent briefs (round 1, round 2 "write new values") and graded results
- `try1/`, `codex_panel/`: generic-accuracy experiments and the Codex panel
- `ui/`: editor app. `precompute.py` (per-head switch-off scan, readouts, attention), `server.py` (live scoring and Claude chat), `index.html`
- `docs/`: the blog page
- `edits/zero_two_heads.py`: the winning two-head edit

## Setup
Needs a GPU, `torch`, `torchvision`, `transformers`, `pillow`, and ImageNet on disk. Data and caches are not in the repo.
```
export IMAGENET_VAL=/path/to/ILSVRC2012_img_val/val IMAGENET_TRAIN=/path/to/ILSVRC2012_img_train
python cache_feats.py        # writes cache_*.pt, proj.pt
python build_typo.py         # writes typo/TA.pt, typo/TB.pt (about 2.3 GB each)
python ui/precompute.py && PORT=8791 python ui/server.py   # then open http://localhost:8791
```
Scripts expect to run from the repo root with `PYTHONPATH=.`. The UI's chat shells out to the `claude` CLI; the server listens on localhost only.
Some scripts under `try1/` and `codex_panel/` were run in an earlier layout and may need small path fixes.
