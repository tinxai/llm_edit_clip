# Task: fix CLIP ViT-B/32's typographic-attack weakness by editing WEIGHTS ("weight-space programming")
Failure mode: when a wrong class name is written on an ImageNet image (white box, black text, e.g. a tench photo with "bustard" on it), CLIP reads the text and misclassifies. Zero-shot top-1 on such images: **~28% attacked vs ~65% clean**. (Examples: <repo>/typo/sample0.png, sample1.png - you can view them.)
Goal: edit the vision tower's weights so attacked accuracy goes up while CLEAN accuracy stays ~unchanged (clean drop must stay <= ~0.5 pts). Text classifier is frozen.
You read and write weights directly: NO gradient descent, NO optimizers, NO training/fine-tuning/fitting weights to data. Allowed: inspect weights/activations, reason about the text-reading circuit, then make small targeted edits (zero/scale/subtract a direction/add a pathway to specific entries, rows, columns, heads, neurons, channels; projections like W <- (I-uu^T)W are fine).

## Tool (Python API)  -- run with:
cd <repo> && AGENT_DIR=<your dir> CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python <your_script.py>
```
from typo_api import Session
s = Session()          # CLIP B/32 (HF CLIPModel). s.vm = vision_model, s.proj = visual_projection. Text side frozen.
s.params()             # dict name -> nn.Parameter (vision tower + visual_projection.weight); edit IN PLACE under torch.no_grad()
s.score('note')        # -> (attacked_acc, clean_acc) on 4000 train images; costs 1 unit of a HARD budget of 40 calls total (persisted on disk)
s.remaining(); s.reset()   # budget left; restore original weights
s.encode(x)            # FREE forward for SYNTHETIC images YOU build (normalized [N,3,224,224]) -> normalized embeddings. For mechanism analysis:
                       #   e.g. render words on blank/random backgrounds, with/without text, and compare activations (use forward hooks on s.vm).
s.save()               # writes diff of changed tensors to $AGENT_DIR/submission.pt  <- YOUR FINAL SUBMISSION (only edits you want graded)
```
Helpers in typo.py: `typo.attack(pil224, text, seed)`, `typo.base` (resize+center-crop), `typo.norm` (to normalized tensor). Class names: zs_meta.IMAGENET_CLASSNAMES. Each python launch reloads the model (~15 s): do many edits per script and call score() sparingly. Weight reading / analysis is free.

## Rules
- Never feed s.C / s.A / s.Y (the eval images) through the model except via score(). Use only synthetic images of your own for analysis (you may use a few real photos from outside the project only if they're not ImageNet; synthetic is fine).
- Do NOT read or use: <imagenet val>, any ImageNet val data, typo/TB.pt, typo/results.json, other agents' directories, or cache_val*. Final grading is on held-out data you never see (a held-out train split + ImageNet val), so overfitting the 4000-image score with many tweaks will not transfer; prefer edits with a mechanistic reason.
- Write only inside your own AGENT_DIR. Use only GPU 5. Do not kill processes you didn't start. No network needed.
- Noise: differences under ~1 pt in attacked acc or ~0.3 in clean acc are noise-level on 4000 images.
- Finish: call s.save() with only the edits to be graded, write notes.md in your AGENT_DIR (hypothesis, evidence, edits, scores seen incl. clean), final reply 3-6 lines.
