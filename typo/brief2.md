# Task: fix CLIP ViT-B/32's typographic-attack weakness by WRITING new weights ("weight-space programming", round 2)
Failure mode: when a wrong class name is written on an ImageNet image (white box, black text, e.g. a tench photo with "bustard" on it), CLIP reads the text and misclassifies. Zero-shot top-1 on such images: **~28% attacked vs ~65% clean**. (Examples: <repo>/typo/sample0.png, sample1.png - you can view them.)
Goal: edit the vision tower's weights so attacked accuracy goes up while CLEAN accuracy stays ~unchanged (clean drop must stay <= ~0.5 pts). Text classifier is frozen.
You read and write weights directly: NO gradient descent, NO optimizers, NO training/fine-tuning/fitting weights to data. Allowed: inspect weights/activations, reason about the text-reading circuit, then make small targeted edits (zero/scale/subtract a direction/add a pathway to specific entries, rows, columns, heads, neurons, channels; projections like W <- (I-uu^T)W are fine).

## Round 2 rule: WRITE new values, do not only switch things off
Round 1 agents fixed ~40% of the gap just by zeroing two attention heads (layer 11 head 8, layer 8 head 4; zeroing their out_proj columns gave ~42% attacked / ~-0.6 clean). That is a deletion. In this round your final submission must contain NEWLY WRITTEN values: numbers you computed and wrote that were not in the model before and are not just 0 or the old value times a constant. Examples of what counts (you may invent others):
- Replace a head's output by writing a constant into out_proj.bias (e.g. the mean contribution that head makes on normal photos) while zeroing its columns, so the residual stream keeps its usual offset. (HF CLIP linear layers have biases: q_proj/k_proj/v_proj/out_proj, fc1, fc2, layer norms; all are in s.params().)
- Write a small NEW pathway: repurpose a few barely-used MLP neurons in a late layer by writing their fc1 row + fc1 bias (a detector for "written-text evidence", e.g. aligned with what the text-reading head reads/writes) and their fc2 column (a correction that cancels the text signal only when the detector fires), as the original post did for a car network.
- Rewrite q/k/v (or q/k biases) of a head so its attention moves away from text regions, or rewrite its value path so it carries something else.
- Edit the final projection/LayerNorm by a computed, mechanistically motivated change.
Zeroing or uniform scaling of existing rows/columns is allowed as a diagnostic (and to find where the text signal lives; you can compare against the zero-the-head baseline with score()) but a submission made ONLY of zeroed/scaled entries will be disqualified. Say in notes.md exactly which values you wrote and why; the grader reports how many changed entries are zeros.
Reminder of the target: beat ~42% attacked while keeping clean accuracy within ~0.5 pts (the zero-the-head edit is the bar; matching it with less clean loss also counts).

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
