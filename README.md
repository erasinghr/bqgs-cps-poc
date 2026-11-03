# BQGS — proof-of-concept and manuscript sources

Open-source code and build sources accompanying the paper:

> **Bio-Quantum Generative Shield (BQGS): A Systematic Review and Integrated
> Architecture for Securing Cyber-Physical Systems**
> Abhinav Singh and Vivek Prakash.
> *International Journal of Modern Education and Computer Science (IJMECS)* — under review.
> DOI: to be added on acceptance.

This repository contains (1) an open, reproducible **proof-of-concept**
simulation of the paper's three-layer anomaly-detection pipeline on a public
industrial-control dataset, and (2) the scripts that generate the paper's **figures**.

## What the proof-of-concept does

It implements the detection and fusion functions of the three BQGS layers and
evaluates them on **HAI 21.03** (a public hardware-in-the-loop ICS dataset,
CC BY-SA 4.0, <https://github.com/icsdataset/hai>):

| Layer | Module |
|---|---|
| L1 — edge | Ant-colony-selected sensor subset + linear one-step residual + pheromone-style smoothing |
| L2 — fog | Variational autoencoder anomaly score (and a USAD baseline) |
| L3 — cloud | 6-qubit ZZ-feature-map quantum-kernel one-class SVM (Qiskit), with a NISQ noise model and a classical RBF fallback |
| Fusion | Tail-surprisal calibration + PSO-tuned or equal weights |

It also includes baselines (USAD, Isolation Forest, PCA residual monitoring),
a synthetic-data (WGAN-GP) distributional-bias study, and per-layer latency
and memory measurements. Every experiment is repeated over 10 random seeds
with Wilcoxon signed-rank tests and Holm correction.

## Headline results (HAI 21.03 test split, mean ± SD over 10 seeds)

- Equal-weight fusion of the three layers: **F1 = 0.713 ± 0.018**, versus
  **0.692 ± 0.012** for the best single layer and **0.691 ± 0.003** for the
  strongest baseline (PCA residual monitoring); all 25 test attack events
  detected.
- Removing any one layer lowers F1 by 0.08–0.20 (significant).
- The simulated quantum kernel gives **no measurable advantage** over its
  classical fallback, and becomes unstable under NISQ noise — so in this work
  the quantum layer is reported as an **optional enhancement**, not a core
  component.
- Edge-layer decision cost ≈ 3.6 µs on a single CPU thread.

These numbers are honest and modest; they are reported as-is, including the
negative results. See the paper for full context and limitations.

## Reproduce

```bash
pip install -r code/requirements.txt
git clone https://github.com/icsdataset/hai
export HAI_DIR=$PWD/hai/hai-21.03
cd code/src
python data.py                         # preprocessing + cache
python run_all.py 0,1,2,3,4,5,6,7,8,9  # 10 seeds
python aggregate.py                    # summary + stats + figures 13-14
python gan_bias.py                     # synthetic-data bias study
python latency.py                      # per-layer latency and memory
```

Chronological split (no shuffling): train on `train1–3` (normal; last 20 % of
`train3` for score calibration), tune on `test1–2`, report on `test3–5`.
Runtime ≈ 5 min/seed on 2 CPU cores. Precomputed outputs are in
`code/results/` (`summary.json`, `summary.csv`, `latency.json`,
`gan_bias.json`, per-seed JSON).

## Repository layout

- `code/` — the proof-of-concept (`src/`, `requirements.txt`, `results/`, `figures/`). See `code/README.md`.
- `manuscript/` — figure-generation scripts and rendered figures (no manuscript text). See `manuscript/README.md`.

## Data and licence

The HAI dataset is **not** redistributed here; the scripts download it from
its official repository, which is released under CC BY-SA 4.0. If you use HAI,
cite Shin et al., *Two ICS Security Datasets and Anomaly Detection Contest on
the HIL-Based Augmented ICS Testbed*, CSET 2021.

The code in this repository is released under the MIT Licence (see `LICENSE`).

## Citation

See `CITATION.cff`. Please cite the paper above; the DOI will be added once the
paper is accepted.
EOF
