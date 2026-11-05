# Task: edit CLIP ViT-B/32 vision-tower WEIGHTS to improve zero-shot ImageNet accuracy ("weight-space programming")
You read and write weights directly. NO gradient descent, NO optimizers, NO training/fine-tuning loops, no fitting weights to data.
Allowed: inspect weights/statistics with numpy/torch, reason about structure, then make small, targeted edits (zero/scale/subtract-a-direction/add-a-pathway to specific entries, rows, columns, heads, neurons, channels).

## Tool (Python API)
Run python as: cd <repo> && AGENT_DIR=$AGENT_DIR CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python your_script.py
(AGENT_DIR is given below; write your scripts inside your own agent dir, not in the project dir.)
```
from wsp_api import Session
s = Session()            # loads CLIP B/32 (HF CLIPModel); s.vm = vision_model, s.proj = visual_projection; text classifier is FROZEN
s.params()               # dict name -> nn.Parameter (vision tower + visual_projection.weight); edit IN PLACE under torch.no_grad()
s.score('note')          # zero-shot top-1 (%) on 5000 train images; costs 1 unit of a HARD budget of 40 forward passes total
s.remaining()            # budget left (persisted on disk across scripts); s.reset() restores original weights
s.save()                 # writes the diff of changed tensors to $AGENT_DIR/submission.pt  <- YOUR FINAL SUBMISSION
```
Each python launch reloads the model (~15s); do many edits per script, e.g. loop edits and call score() between them, since score() is the scarce resource.
Weight reading/analysis is free (no budget): use s.params() tensors, SVD, norms, etc. Do not run forward passes over images yourself outside score() - that would circumvent the budget. (You may run on synthetic/random inputs for mechanism analysis if you want.)

## Rules
- Do NOT read, list or use <imagenet val> or any ImageNet validation data, or any file in the project dir named cache_val* / val_*. The final grade is on held-out ImageNet val + a held-out split of train images you never see.
- Stay inside your AGENT_DIR for writes. Use only GPU index 5 (CUDA_VISIBLE_DEVICES=0). Do not kill any processes you did not start. No network use needed.
- The score you see (split A, 5000 images, ~64.9% baseline) is noisy: differences under ~0.3 pts are noise. Prefer edits with a mechanistic reason and a consistent effect; spending the whole budget picking the best of many random tweaks will overfit and score nothing on the held-out set.
- When done (or budget is exhausted), call s.save() with ONLY the edits you want graded (reset/redo if needed), then write notes.md in AGENT_DIR: what you found, what you edited and why, scores seen. Final reply: 3-5 lines.
