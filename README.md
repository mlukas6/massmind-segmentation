# MassMIND Segmentation

Semantic segmentation of long-wave infrared (LWIR) maritime imagery from the
[MassMIND dataset](https://github.com/uml-marine-robotics/MassMIND)
(Nirgudkar et al., 2023). Coursework for *Computer Vision — Assignment 2,
FEUP 2025/26* (group of three: Lars Husemann, Lucas Coelho, Lucas Aparicio).

## Deliverable

**[`notebooks/final_run.ipynb`](notebooks/final_run.ipynb)** — self-contained
Kaggle notebook. Downloads the dataset, trains all eight configurations end
to end, and renders the full report (methodology, results, discussion,
conclusion) inline.

Designed for a single Kaggle T4 session (~9 h wall-clock at 35 epochs).

## What we ran

Four architectures × two augmentation conditions = **8 training runs**, each
35 epochs at 384×480 resolution.

| Architecture | Encoder weights | Params | GFLOPs |
|---|---|---:|---:|
| U-Net with VGG-16 encoder | ImageNet | 23.75 M | 139.0 |
| U-Net with VGG-16 encoder | from scratch | 23.75 M | 139.0 |
| `CustomLWIRUNet` (depthwise-separable + transformer bottleneck) | from scratch | 4.69 M | 29.4 |
| U-Net with MobileNetV2 encoder | from scratch | 6.63 M | 19.1 |

Each architecture is trained with and without the geometric augmentation
schedule from the MassMIND paper (horizontal flip + small rotations).

**Loss:** Focal-Tversky combined (focal γ = 2 + Tversky α = 0.3, β = 0.7).
The β > α weighting penalises false negatives on rare classes more than false
positives — the training-side counterpart to the threshold-tuning approach
used in the original paper.

## Headline results (test mIoU)

| Architecture | no aug | aug | Δ |
|---|---:|---:|---:|
| **VGG-16 pretrained** | **0.830** | 0.817 | −1.30 pp |
| VGG-16 scratch | 0.789 | 0.780 | −0.94 pp |
| custom_lwir | 0.774 | 0.761 | −1.23 pp |
| MobileNetV2 | 0.737 | 0.713 | −2.44 pp |

Two non-obvious findings:

- **Geometric augmentation hurts every architecture on test mIoU.** The
  ±2°/±5°/±7° rotation schedule breaks the strong horizon prior of maritime
  LWIR imagery (sky-over-water-over-self), and the rare classes (bridge /
  obstacle / living_obstacle) lose the most. Intensity-domain augmentation,
  which preserves the horizon prior, is the obvious next experiment.
- **Focal-Tversky beats threshold-tuning on rare classes.** Even our
  from-scratch VGG-16 at lower resolution (384×480 vs the paper's native
  640×512) reaches F1 = 72.1 on living_obstacle at standard `argmax`
  decoding, vs F1 = 54.5 for the paper's UNet at their tuned τ = 0.3.

Full per-class breakdown, confusion matrices, and Pareto analysis are in the
deliverable notebook (§6.3–§6.8).

## Repo layout

```
.
├── README.md
├── requirements.txt
├── project_proposal_and_deliveries.md   ← assignment brief
├── notebooks/
│   └── final_run.ipynb                  ← THE DELIVERABLE
├── src/
│   ├── models/
│   │   ├── custom_lwir_unet.py          ← our custom architecture
│   │   ├── unet.py                      ← VGG-16 U-Net wrapper
│   │   ├── unet_vgg16_ext.py            ← architecture-probe variants
│   │   ├── _attention_gate.py
│   │   ├── _transformer_bottleneck.py
│   │   ├── _blocks.py                   ← DSConv, DoubleDSConv
│   │   └── _adapt.py                    ← 3→1 channel conv adapter
│   ├── dataset.py                       ← LWIR loader + on-the-fly split
│   ├── augmentations.py                 ← deterministic per-(image, epoch)
│   ├── losses.py                        ← Focal + Tversky + FocalTversky
│   ├── metrics.py                       ← streaming confusion matrix
│   ├── splits.py                        ← 70/20/10 session-stratified
│   ├── stats.py                         ← dataset mean/std
│   └── train.py                         ← reusable trainer entrypoint
├── scripts/
│   ├── download.py                      ← fetch MassMIND from upstream Drive
│   ├── export_runs_onnx.py              ← export trained checkpoints
│   ├── export_onnx.py                   ← export a single model
│   └── probe_architectures.py           ← legacy probe sweep driver
├── tests/                               ← pytest, src/ unit tests
├── archive/                             ← development history (see below)
├── data/                                ← gitignored; populated by download.py
└── runs/                                ← gitignored; per-run checkpoints + metrics
```

## Quickstart

The notebook is self-contained — clone the repo, open
`notebooks/final_run.ipynb` on Kaggle (T4 instance), and run all cells. No
data uploads needed; `gdown` fetches the dataset from the upstream MassMIND
Drive links.

For local development of the `src/` library:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/         # ~30s on CPU
```

To re-export trained checkpoints to ONNX:

```bash
python scripts/export_runs_onnx.py --runs vgg16_pretrained custom_lwir
```

## The `archive/` directory

Everything we built and discarded on the way to the final 8-run benchmark:

- `archive/README_dev.md` — the original 697-line development README
  documenting the architecture probe (`VGG16UNetExt`), the three modular
  experiments (attention gates / transformer bottleneck / aux heads), and
  the per-mod metrics that informed `CustomLWIRUNet`.
- `archive/notebooks/01_data_exploration.ipynb` … `08_final.ipynb` — the
  development notebooks in chronological order. `08_final.ipynb` is the
  immediate predecessor of `notebooks/final_run.ipynb`; the others trace the
  earlier iterations (initial Colab smoke-test, Kaggle adaptation, probe
  experiments, augmentation study).
- `archive/kaggle_*.ipynb` — historical Kaggle runs preserved for output
  provenance (e.g. the probe-experiment runs whose metrics are summarised in
  `archive/{att,att_transformer,base,trans,trans_aux}_metrics.csv`).
- `archive/12_Pytorch_SemanticSegmentation.ipynb`,
  `archive/mobilenet-cv (1).ipynb` — third-party / teammate drafts retained
  for attribution.

Nothing in `archive/` is required for the deliverable to run; it is kept for
audit and to document the development trajectory.

## Dataset

The MassMIND dataset (Nirgudkar et al., 2023) is **not redistributed** with
this repo for copyright reasons. The deliverable notebook downloads it on
first run from the upstream Google Drive links, into `data/massmind/`
(gitignored).

- 2,916 LWIR images at 640×512 native resolution.
- Seven classes: sky, water, bridge, obstacle, living_obstacle, background,
  self.
- 70 / 20 / 10 session-stratified split into train / val / test, computed
  deterministically by `src/splits.py`.

## References

- Nirgudkar, S., DeFilippo, M., Sacarny, M., Benjamin, M., Robinette, P.
  (2023). *MassMIND: Massachusetts Maritime INfrared Dataset.* International
  Journal of Robotics Research, 42(1–2), 21–32. doi:10.1177/02783649231153020
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional
  Networks for Biomedical Image Segmentation.* MICCAI.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017).
  *Focal Loss for Dense Object Detection.* ICCV.
- Salehi, S. S. M., Erdogmus, D., Gholipour, A. (2017). *Tversky loss
  function for image segmentation using 3D fully convolutional deep
  networks.* MLMI.
- Howard, A. et al. (2017). *MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications.* arXiv:1704.04861.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Wu, Y., He, K. (2018). *Group Normalization.* ECCV.
