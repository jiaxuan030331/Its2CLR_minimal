# ItS2CLR Minimal Repro (A/B-256)

This repository provides a minimal, fast-to-run reproduction of the ItS2CLR idea with two pipelines:
- A (MIL-only baseline)
- B (Iterative fine-tuning with MIL + pseudo-label updates)

It focuses on a small demonstration dataset (≈256 patches per bag) to make the workflow transparent and quick to execute.

## What this shows
- End-to-end flow: feature extraction → pseudo-labeling → (optional) iterative MIL → visualization.
- Side-by-side A/B comparison (pseudo-label distributions and summary metrics).

## Quickstart
1) Prepare data (paths follow our demo):
- training: 
- testing: 
- labels: 

2) Run the original training in your environment (optional) to regenerate outputs:
- A-256 outputs at: 
- B-256 outputs at: 

3) Open the notebook  and run the cells.

## Results snapshot (what to expect)
- With tiny data and constant-init pseudo labels, class separation is weak. The notebook visualizes this clearly.

## Disclaimer & Limitations (DP)
This project was run out of personal interest to understand the workflow. Due to limited resources, the sample size is very small and results have little reference value. This repo is a minimal workflow reference, not a faithful reimplementation of the full paper setting. With small data (~256 per bag), same-WSI positive/negative bags, constant-init pseudo labels, and simplified training settings, instance-level separation may be weak and metrics unstable. Do not directly compare these numbers to the paper.

## Acknowledgement of Differences
- Features: online image features (ResNet18) vs. paper often using pre-extracted features.
- Data scale and WR: small demo vs. larger, realistic WR distributions in paper.
- Pseudo labels: constant-init here vs. better initial labels in paper setups.
- Training details: no apex/fp16; simplified scheduler/batch splits; Comet disabled.
- Preprocessing: no stain normalization; augmentations/cropping may differ from the paper.

## License
MIT
