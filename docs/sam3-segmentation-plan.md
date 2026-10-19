# Improving segmentation with SAM 3

The pipeline uses SAM 3 only through its text prompt (`"bird feather"`, plus two darker-feather prompts on
retry) on the whole slide at SAM 3's native 1008 px. SAM 3 can do more than that. This page lists what to try,
how it is measured, and what was found.

## Baseline: the 100-slide run (`slice100-20261019`)

| | |
|---|---|
| Feathers | 339 |
| Feathers that failed a check | 57 |
| Feather count right (30 hand-checked slides) | 29 / 30 |
| Missed-feather flags | 3 slides |

The failures fall into a few kinds:
- **Merged:** one mask covers several feathers, e.g. `BB17692 1998-im0513` B holds three feathers.
- **Split:** one feather becomes two masks, e.g. `A3818 2002-im1708`, a feather partly under the board.
- **Outline cut:** the mask stops short of the fringe or the downy base, e.g. `A2328 2000` C at the bottom.
- **Small measured holes:** pockets between separated barbs. These look like false alarms of the hole
  measurement (threshold 0.2% of the feather), not segmentation errors.

## Ideas, in the order to test them

1. **Example-box prompts.** Besides text, SAM 3 takes boxes as examples: positive ("find feathers like
   this one") and negative ("not this"). Give it a well-segmented feather from the same slide as a positive
   example, and the label and date areas as negatives. Aimed at merged and split feathers.
2. **A second pass per feather.** Re-segment each feather on a crop around it, with SAM 3's interactive
   predictor: a box around the feather, a positive point on it and negative points on its neighbours. The crop
   is far smaller than the slide, so SAM 3 sees it at a much higher effective resolution. Aimed at cut fringe and
   downy bases, and at separating touching feathers.
3. **Part prompts.** Text prompts for parts ("downy feather base", "feather fringe") to add back where a
   feather's check reports a side cut.
4. **Repair only flagged feathers.** Re-prompt only feathers that failed a check, instead of re-segmenting
   whole slides on retry.

Also to tune separately: the measured-hole threshold (0.2% → about 1%, or only when the model also sees a hole).

## How it is measured

Each variant goes through the pipeline's own processing (mask clean-up, crops, per-feather checks with the
same prompt and model, missed-feather check) on the same slides: the problem slides above plus clean slides as
controls. Compared per variant: feathers that fail a check and which check, feather counts against the hand
counts, and the overlays side by side. A variant is worth keeping if it fixes failures without breaking the
clean slides.

Script: `scripts/sam3_experiments.py` (runs on one worker node, outside the job queue).

## Results: ideas 1 and 2 (2026-10-19)

17 slides: 12 problem slides from the 100-slide run, 2 with hole flags only, 3 clean. Each went through the
pipeline 4 times. The run ID was `test-sam3exp-10190455`, on one worker node.

| Variant | Feathers failing a check, holes aside | Hole flags | Feather counts right | Mean score | s / slide |
|---|---|---|---|---|---|
| Text prompt (now) | 20 | 15 | 8/9 | 8.31 | 90 |
| + example boxes (idea 1) | 16 | 15 | 9/9 | 8.51 | 91 |
| + second pass per feather (idea 2) | 21 | 23 | 9/9 | 7.91 | 95 |
| both | 19 | 27 | 9/9 | 7.62 | 96 |

**Example boxes: a small, real gain. Switched on.**
- It fixes a split feather: `A3818 2002-im1708` goes from 5 crops to 4, matching the hand count, and the
  fragment is gone.
- One mask improved (`BX40715 2005` A grew 6.5%, and its outline-cut flag cleared).
- Clean slides are unchanged: mask areas move by at most 1.6%.
- It costs about 1 s per slide.
- It does not fix merged feathers: the `BB17692 1998` slides are unchanged.

**Second pass per feather: worse. Off.** The refined masks have more holes (23 and 27 hole flags against
15), and the scores drop.

**Checks vary between runs.** Four of the five flag changes between the text prompt and the example boxes
happened on masks whose area changed by less than 1%. Three flags cleared and one appeared on a clean slide.
So the per-feather checks are not repeatable to within a few feathers. Compare variants on mask changes and
feather counts, not only on flag counts.

Next:
- Merged feathers still need a fix: part prompts (idea 3) or flagged-only repair (idea 4).
- The second pass could be retried with the first mask as a mask prompt instead of box + points.
