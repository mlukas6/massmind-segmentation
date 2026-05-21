# MassMIND Segmentation


Semantic segmentation of long-wave infrared (LWIR) maritime imagery from the
[MassMIND dataset](https://github.com/uml-marine-robotics/MassMIND) (Nirgudkar
et al., 2023). Coursework for *Computer Vision — Assignment 2, FEUP 2025/26*.

The assignment asks for two things: **(1)** a custom segmentation architecture
proposed and defended by us, and **(2)** a comparison against at least one
existing model trained on the same data. We deliver:

- **Custom model — `CustomLWIRUNet`:** a from-scratch, deliberately
  lightweight U-Net (~4.7 M parameters, ~52 GFLOPs) purpose-built for
  single-channel LWIR. Depthwise-separable convolutions, GroupNorm, SiLU
  activations, a Transformer-encoder bottleneck, and deep-supervision
  auxiliary heads. Every block is built from `torch.nn` primitives — no
  pretrained backbone. The design incorporates the findings of an earlier
  architecture probe (`VGG16UNetExt`, see below).
- **Existing-model baselines:** a U-Net with a VGG-16 encoder (SMP, Nirgudkar
  et al.'s strongest CNN baseline) in two variants — **pretrained** on
  ImageNet (the accuracy ceiling) and **from scratch** (the fair-capacity
  comparison). Same data, same loss, same training scaffolding.

The headline benchmark is a **three-way comparison at 50 epochs**, run
end-to-end in the self-contained `notebooks/06_assignment_colab.ipynb`.

## Status

| Phase | State |
|---|---|
| Dataset download + 70/20/10 session-stratified split | ✅ |
| Pixel mean/std + per-class pixel counts | ✅ |
| U-Net + VGG-16 trainer (PyTorch; CUDA / MPS / CPU autodetect) | ✅ |
| Augmentation pipelines A / B / C | ✅ |
| Colab + Kaggle training notebooks | ✅ |
| `VGG16UNetExt` architecture probe (`base` / `att` / `trans` / `att_trans` / `trans_aux`) on T4 | ✅ — see "Architecture probe results" below |
| Hand-implemented `AttentionGate`, `TransformerBottleneck`, aux heads | ✅ |
| `CustomLWIRUNet` from-scratch model (`src/models/custom_lwir_unet.py`) | ✅ |
| Hand-implemented `DepthwiseSeparableConv` / `DoubleDSConv` blocks (`src/models/_blocks.py`) | ✅ |
| Focal loss (γ=2) + AdamW + cosine schedule + linear warmup | ✅ |
| AMP (mixed precision) opt-in flag for CUDA | ✅ |
| Self-contained assignment notebook (`notebooks/06_assignment_colab.ipynb`) | ✅ |
| Final 50-epoch three-way run (`vgg16_pretrained` / `vgg16_scratch` / `custom_lwir`) | ✅ — see "Final results" below |
| ONNX export of trained models (`scripts/export_runs_onnx.py`) | ✅ |
| Final writeup | ⏳ |

## Custom architecture (`CustomLWIRUNet`)

`CustomLWIRUNet` (`src/models/custom_lwir_unet.py`) is the headline
contribution: a from-scratch U-Net designed *for single-channel thermal
imagery specifically*, not a shrunk copy of VGG-16. Every choice the 2015
U-Net inherited from its era has been deliberately replaced. At **stem
width 48** (channel multipliers `(1, 2, 4, 8, 8)` → widths
`48, 96, 192, 384, 384`) it has **~4.7 M parameters** and **~52 GFLOPs** at
the native 512×640 LWIR resolution — about **5× smaller and ~4.7× cheaper**
than the VGG-16 baselines (~23.75 M params, ~247 GFLOPs).

```
[B, 1, H, W]
  │
  ▼  Stem (full res):      2× StandardConvBlock (1 → 48)            → skip0
  ▼  Encoder stage 1:      MaxPool/2 + DoubleDSConv (48  → 96)      → skip1 (stride 2)
  ▼  Encoder stage 2:      MaxPool/2 + DoubleDSConv (96  → 192)     → skip2 (stride 4)
  ▼  Encoder stage 3:      MaxPool/2 + DoubleDSConv (192 → 384)     → skip3 (stride 8)
  ▼  Encoder stage 4:      MaxPool/2 + DoubleDSConv (384 → 384)     → bottleneck in (stride 16)
  ▼  Bottleneck:           TransformerBottleneck (384ch, 2 layers, 8 heads)
  ▼  Decoder ×4:           ConvT/2 + concat skip + DoubleDSConv
  ▼  Heads:                main 1×1 (48 → 7)  +  2 aux heads (training only)
```

Seven concrete departures from a standard U-Net:

| Aspect | Standard U-Net | `CustomLWIRUNet` | Effect |
|---|---|---|---|
| Conv block | 3×3 Conv | `DepthwiseSeparableConv` (3×3 depthwise + 1×1 pointwise) | ~8× fewer params/FLOPs per block, same receptive field |
| Normalisation | BatchNorm | GroupNorm (8 groups) | Batch-size-independent; works at batch=1 for deployment |
| Activation | ReLU | SiLU (swish) | Smooth gradient, no dead-unit failure mode |
| Bottleneck | DoubleConv | `TransformerBottleneck` (2 layers, 8 heads) | Global receptive field at the deepest stage |
| Supervision | Single head | Main + 2 aux heads, weights 0.4 / 0.2 (training only) | Faster convergence, **zero** inference cost |
| Width schedule | 64-128-256-512-1024 | 48-96-192-384-384 | 3× narrower at the bottleneck |
| First layer | 3-ch RGB → adapted | Native 1-channel from epoch 0 | No pretrained-RGB feature baggage |

Building blocks (`DepthwiseSeparableConv`, `StandardConvBlock`, `DoubleDSConv`)
are hand-implemented in `src/models/_blocks.py`; the bottleneck reuses the
same hand-implemented `TransformerBottleneck` from the probe. The stem uses a
plain 3×3 conv because depthwise convolution is degenerate at 1 input
channel. Aux heads are active only in `model.train()` — in `eval()` the
forward returns a single tensor, so the deployed graph is exactly the
~4.7 M-param / ~52-GFLOP model.

**Why these choices**: depthwise-separable convs and a narrow width schedule
keep the parameter/FLOP budget tiny; the Transformer bottleneck (the probe's
strongest single contributor — see below) restores the global context a
small conv stack would lack; deep-supervision aux heads accelerate
convergence of a from-scratch model at no deployment cost. GroupNorm + SiLU
are the modern, batch-size-robust replacements for BatchNorm + ReLU.

> **Stem-width note:** an earlier `CustomLWIRUNet` used stem width 32
> (~2.1 M params) but under-fitted the rare classes. Widening to 48
> (~4.7 M) — every block scales quadratically with stem width — gave the
> capacity headroom that closed most of the gap to the baselines.

## Architecture probe (`VGG16UNetExt`)

Before building `CustomLWIRUNet` from scratch we ran an **architecture
probe**: which decoder-side modifications actually help on MassMIND? The
probe model `VGG16UNetExt` reuses the same encoder/decoder *pattern* as the
in-class demonstrator (cell 30 of `12_Pytorch_SemanticSegmentation.ipynb`:
pretrained VGG-16 sliced into encoder stages + hand-rolled `conv`/`up_conv`
decoder helpers + skip concatenation), with three independently-toggleable
modifications. Each is a flag on `build_unet_vgg16_ext()` so we can ablate
them cleanly. **The probe's findings directly determined `CustomLWIRUNet`'s
design** (transformer bottleneck: keep; attention gates: drop; aux heads:
keep).

| Flag | Component | Status |
|---|---|---|
| `use_attention_gates=True` | Oktay-style attention gates on each skip | hand-implemented in `src/models/_attention_gate.py` |
| `use_transformer_bottleneck=True` | Multi-head self-attention bottleneck body | hand-implemented in `src/models/_transformer_bottleneck.py` |
| `use_aux_heads=True` | Deep-supervision aux heads at decoder mids | hand-implemented in `src/models/unet_vgg16_ext.py` |

All four flag combinations (none / one / two / three on) plus the
`trans + aux` combination are wired into the probe script
(`scripts/probe_architectures.py`) and were measured on Kaggle T4.

```
Input (1×H×W, LWIR)
   │
   ▼  channel-mean-adapted first conv (3-ch ImageNet → 1-ch LWIR)
   │
   ▼  VGG-16 encoder (pretrained), 6 features at strides [1, 2, 4, 8, 16, 32]
Enc0  Enc1  Enc2  Enc3  Enc4  Enc5
                              │
                              ▼  [Mod A] AttentionGate on each skip (optional)
                              │  [Mod B] Transformer bottleneck (optional)
                         bottleneck
                              │
                              ▼  4× (ConvTranspose → concat skip → DoubleConv)
                         Dec1 → Dec2 → Dec3 → Dec4
                                 │      │      │
                                 ▼      ▼      ▼
                             [Mod C aux head]  [Mod C aux head]  Final upsample
                             (training only)   (training only)         │
                                                                       ▼
                                                                  1×1 Conv → 7 classes
```

### Mod A — Attention Gate skip refinement (`use_attention_gates=True`)

Hand-implemented `AttentionGate` (Oktay et al. 2018) in
`src/models/_attention_gate.py`. Replaces the `Up.skip_refine = nn.Identity()`
default on each decoder block with an additive attention module that takes
the encoder skip and the upsampled decoder gating signal, computes a spatial
attention map `α ∈ (0, 1)` via `(W_skip(skip) + W_gating(gating)) → ReLU →
1×1 conv → BN → sigmoid`, and returns `α · skip`. Channels collapsed to
`skip_channels // 2` internally, BatchNorm throughout.

*Why:* the vanilla U-Net's skip forwards encoder features unchanged — the
decoder gets no learned control over what it receives. The gate suppresses
clutter/background regions of the skip based on the decoder's coarse-grained
context. For LWIR this matters: the same thermal edge can be a boat hull
or a wave artefact, distinguishable only from broader scene structure.

Approx. parameter cost: ~610 K across all four levels (1×1 convs at
512/512/256/128 channels).

### Mod B — Transformer bottleneck (`use_transformer_bottleneck=True`)

Hand-implemented `TransformerBottleneck` in `src/models/_transformer_bottleneck.py`.
Replaces the default `DoubleConv` body of the bottleneck wrapper. The 512-channel
encoder Stage-5 feature map (8×8 at 256-px input, 16×20 at 640×512 native LWIR)
is flattened to tokens, summed with a learnable 2-D positional embedding
(bilinearly resized to runtime spatial), and passed through **2 stacked
`nn.TransformerEncoderLayer` blocks** (8 attention heads, embed dim 512,
FFN hidden 1024, GELU, pre-norm), then reshaped back to spatial form.

The composition is hand-rolled; we use `nn.TransformerEncoderLayer` as a
primitive for the math just as we use `nn.Conv2d` for convolutions.

*Why here:* attention is global by construction but quadratic in tokens; at
the bottleneck the spatial grid is tiny (64 tokens at 256 px), so it's
cheap. The MassMIND task has class-level scene structure — water spans the
whole frame, bridges are extended objects, rare classes need global context
to be distinguished from clutter — exactly the kind of dependency stacked
3×3 convs struggle to capture. Reference pattern: TransUNet (Chen et al.
2021); attention mechanism: Vaswani et al. 2017.

Approx. parameter cost: 4.7 M (replaces the 5.2 M `DoubleConv`-bottleneck,
so the *net* cost is slightly negative).

### Mod C — Deep supervision auxiliary heads (`use_aux_heads=True`)

Two extra 1×1-conv heads attached to the Up2 (deeper, weight 0.2) and Up3
(shallower, weight 0.4) decoder outputs in
`src/models/unet_vgg16_ext.py`, each bilinear-upsampled to the input
resolution. Total training loss:

```
L_total = L_main  +  0.4 · L_aux_shallow  +  0.2 · L_aux_deep
```

The model returns a tuple `(main, aux_shallow, aux_deep)` in `model.train()`
mode and just `main` (a single tensor) in `model.eval()` — so the heads
add **zero deployed parameters** and zero inference cost. Loss combination
lives in `src/train._compute_loss()`.

*Why:* gradient for the main output flows back through the entire decoder
before reaching the bottleneck; for `living_obs` at 0.05 % of pixels, that
gradient is dominated by majority-class signal at every intermediate layer.
Auxiliary heads inject direct full-class-distribution supervision into the
deeper decoder layers. Reference patterns: PSPNet (Zhao et al. 2017),
UNet++ (Zhou et al. 2018).

Approx. parameter cost: ~5 K (negligible; two 1×1 convs of 512×7 and 256×7).

### Demonstrator vs `VGG16UNetExt` — at a glance

| Component | Demonstrator (`12_Pytorch_SemanticSegmentation.ipynb`) | `VGG16UNetExt` (ours, `trans_aux`) |
|---|---|---|
| Encoder | `vgg16_bn(pretrained=True).features` (torchvision) | SMP `vgg16` (`pretrained="imagenet"`) + 3-ch → 1-ch channel-mean adapter |
| Bottleneck | `conv(512, 1024)` (single DoubleConv) | Hand-implemented `TransformerBottleneck` (2 layers, 8 heads) |
| Skip connections | Raw concatenation (`torch.cat`) | Hand-implemented `AttentionGate` (Oktay-style) — opt-in via flag |
| Output supervision | Single 1×1 conv head | Main + 2 aux heads with weighted loss (train-only) |
| Approx. parameter count | ~24 M | ~38 M |
| Approx. model code | ~80 lines | ~530 lines (model files only) |
| Pretrained weights | ImageNet (`pretrained=True` default) | ImageNet (channel-mean-adapted for 1-channel LWIR) |
| Loss | Cross-entropy | Focal loss (γ=2) |

## Training methodology

- **Loss** — `FocalLoss(gamma=2.0)` from Lin et al. (2017), implemented in
  `src/losses.py`. Down-weights easy-to-classify pixels (the majority sky /
  water) and focuses gradient on hard rare-class pixels. Dropped the planned
  Dice+CE because focal loss matched or beat it in our baseline probe at
  the same data scale, with a simpler single-term formulation. Both `--loss
  focal` (default) and `--loss ce` are supported in `src/train.py`.
- **Optimizer** — AdamW, `weight_decay = 1e-4`.
- **Schedule** — `CosineAnnealingLR` from `lr = 1e-4`, with an optional
  linear warmup (5 % of total steps) for the from-scratch configs
  (`vgg16_scratch`, `custom_lwir`) where the early gradients are noisier.
- **Epochs** — 10 for the `VGG16UNetExt` architecture probe; **50 for the
  final three-way run** on full data, matching the MassMIND paper.
- **Batch size** — 4 (the `06_assignment_colab.ipynb` default; drop to 2 on
  Kaggle T4 if the VGG-16 configs OOM — Kaggle holds slightly more baseline
  GPU memory than Colab).
- **Mixed precision** — opt-in via `--amp` CLI flag (default off in
  `src/train.py` to preserve baseline numerics; default *on* in the probe
  driver and the assignment notebook). Implementation: `torch.amp.autocast(fp16)`
  + `GradScaler`, gated on `device.type == "cuda"` so the same code path is a
  no-op on CPU/MPS. Delivered ~2.5× speedup on Kaggle T4.
- **Weight initialisation** — the two VGG-16 baselines build through SMP;
  `vgg16_pretrained` loads ImageNet weights and adapts the first conv from
  3-ch RGB to 1-ch LWIR via channel-mean initialisation (`src/models/_adapt.py`),
  `vgg16_scratch` is randomly initialised. `CustomLWIRUNet` is trained
  entirely from scratch (Kaiming init, no pretrained weights).
- **Augmentations** — three pipelines defined in `src/augmentations.py`; see
  table below.

## Final results

Three-way comparison, **50 epochs**, full data (2042 train / 583 val),
focal loss, AMP, Kaggle T4. Run end-to-end in
`notebooks/06_assignment_colab.ipynb`. Validation every 5 epochs to fit the
Kaggle session budget.

| Config | Params | GFLOPs | best mIoU | best ep | train (min) |
|---|---:|---:|---:|---:|---:|
| `vgg16_pretrained` | 23.75 M | 247.16 | **0.8558** | 50 | 129.4 |
| `vgg16_scratch` | 23.75 M | 247.16 | 0.8319 | 50 | 130.9 |
| **`custom_lwir`** | **4.69 M** | **52.31** | **0.8095** | 50 | 254.7 |

Per-class IoU at the best epoch:

| Class | Pixel-% | `vgg16_pretrained` | `vgg16_scratch` | `custom_lwir` |
|---|---:|---:|---:|---:|
| sky | 30.3 | 0.987 | 0.985 | 0.981 |
| water | 51.1 | 0.991 | 0.988 | 0.985 |
| bridge | 1.6 | 0.745 | 0.692 | 0.650 |
| obstacle | 0.9 | 0.730 | 0.675 | 0.637 |
| living_obs | 0.05 | 0.685 | 0.657 | 0.629 |
| background | 10.9 | 0.907 | 0.887 | 0.858 |
| self | 3.1 | 0.947 | 0.938 | 0.928 |

**Findings:**

1. **`custom_lwir` reaches 95 % of the pretrained ceiling at 20 % of the
   parameters and 21 % of the GFLOPs** — and 97 % of the from-scratch
   baseline's mIoU. The gap to `vgg16_scratch` (the fair, equal-pretraining
   comparison) is just **2.2 pp**.
2. **The residual gap is concentrated in the minority classes** (`bridge`
   −4.2 pp, `obstacle` −3.8 pp, `living_obs` −2.8 pp vs `vgg16_scratch`).
   Dominant classes (`sky`, `water`, `self`) are effectively tied — a 5×
   smaller model has 5× less capacity to spend on rare classes.
3. **`living_obs` no longer collapses to 0.** Focal loss + 50 epochs lifts
   it to 0.63–0.69 IoU across all three configs — the earlier 20-epoch
   cross-entropy baseline's `living_obs = 0.000` failure mode is resolved
   *without* a threshold sweep.
4. **All three configs were still improving at epoch 50** (`best_epoch = 50`
   everywhere) — none had plateaued. Extending to 80–100 epochs would lift
   all three further, most of all `custom_lwir` (GroupNorm + depthwise nets
   train slower per step).
5. **FLOPs ≠ wall-clock on GPU.** `custom_lwir` has 4.7× fewer GFLOPs but
   runs ~2× *slower* per epoch on the T4: depthwise convs bypass Tensor
   Cores and are memory-bandwidth-bound. The efficiency win is real for
   memory footprint and edge/CPU deployment, not for GPU latency.

## Architecture probe results

Five `VGG16UNetExt` configurations trained on Kaggle T4 with focal loss,
600 training images, 10 epochs, AMP on the AMP runs. Same seed, same
hyperparameters across all configs — the only thing that varies is the
architecture flag combination. Probe driver: `scripts/probe_architectures.py`.
These results determined `CustomLWIRUNet`'s design.

| Config | Att. gates | Transformer | Aux heads | Precision | best mIoU | min/cfg |
|---|:---:|:---:|:---:|---|---:|---:|
| `base` | ❌ | ❌ | ❌ | FP32 | 0.579 | 41 |
| `att` | ✅ | ❌ | ❌ | AMP | 0.620 | 16 |
| `trans` | ❌ | ✅ | ❌ | FP32 | 0.640 | 41 |
| `att_trans` | ✅ | ✅ | ❌ | AMP | 0.624 | 14 |
| **`trans_aux`** | ❌ | ✅ | ✅ | AMP | **0.645** | 15 |

Per-class IoU at the best epoch:

| Class | Pixel-% | `base` | `att` | `trans` | `att_trans` | `trans_aux` |
|---|---:|---:|---:|---:|---:|---:|
| sky | 30.3 | 0.983 | 0.985 | 0.984 | 0.984 | 0.985 |
| water | 51.1 | 0.982 | 0.983 | 0.983 | 0.981 | 0.983 |
| **bridge** | 1.6 | 0.330 | 0.410 | 0.424 | 0.386 | **0.450** |
| **obstacle** | 0.9 | 0.059 | 0.222 | 0.351 | 0.297 | 0.351 |
| **living_obs** | 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| background | 10.9 | 0.813 | 0.826 | 0.826 | 0.812 | 0.830 |
| self | 3.1 | 0.889 | 0.913 | 0.915 | 0.905 | 0.919 |

**Three findings drive the architecture choice:**

1. **The Transformer bottleneck is the dominant contributor:** +6.1 pp mIoU
   over `base`, with the gain concentrated on `obstacle` (+29 pp) and
   `bridge` (+9 pp) — the two mid-rarity classes whose disambiguation
   requires global scene context. Sky/water are already saturated for all
   configs. Matches the TransUNet hypothesis exactly.
2. **Attention Gates help individually but not additively with Transformer:**
   `att` alone is +4.0 pp, but `att_trans` (0.624) is *worse* than `trans`
   alone (0.640). The two mechanisms partially compete for the same
   feature-reweighting role; once the transformer provides strong global
   context, the gates suppress skip detail that the decoder needs back.
3. **Deep supervision yields a small but real gain at the same compute
   budget:** `trans_aux` 0.645 vs `trans` 0.640 (+0.5 pp). The mechanism is
   *convergence acceleration*, not a ceiling lift — at epoch 5 `trans_aux`
   already has `obstacle = 0.125` vs `trans = 0.023` (5× higher). Both
   plateau at similar levels by epoch 10. Bridge gains most (+2.6 pp). Zero
   inference-time cost.

**`living_obs` stays at 0.000 across all five probe configurations.** At
10 epochs with argmax decoding this 0.05 %-pixel class never wins an argmax.
This was *not* an architecture limitation: the final 50-epoch runs (see
"Final results" above) lift `living_obs` to 0.63–0.69 IoU on all three
configs — longer training plus focal loss resolves it without a
threshold-sweep decoding fix.

The probe's verdict — **transformer bottleneck helps most, attention gates
do not combine with it, aux heads accelerate convergence** — is what
`CustomLWIRUNet` was built on: it keeps the transformer bottleneck and the
aux heads, and omits the attention gates.

## Project layout

```
massmind_segmentation/
├── data/
│   ├── massmind/                          # raw LWIR images + masks (gitignored)
│   └── splits/
│       ├── split.json                     # 70/20/10 session-stratified
│       ├── stats.json                     # train-set pixel mean & std
│       └── class_pixel_counts.json        # global per-class pixel count
├── src/
│   ├── dataset.py                         # MassMINDDataset, bit-depth aware
│   ├── splits.py                          # session-stratified split builder
│   ├── stats.py                           # pixel mean/std + class counts
│   ├── augmentations.py                   # albumentations pipelines A / B / C
│   ├── metrics.py                         # ConfusionMatrixTracker → IoU, pixel acc
│   ├── losses.py                          # FocalLoss (Lin et al. 2017)
│   ├── train.py                           # single-file trainer; AMP, model+loss dispatch
│   └── models/
│       ├── __init__.py                    # builder exports
│       ├── _adapt.py                      # adapt first conv 3-ch → 1-ch via channel mean
│       ├── _blocks.py                     # DepthwiseSeparableConv / DoubleDSConv / StandardConvBlock
│       ├── _attention_gate.py             # AttentionGate (Oktay 2018), hand-implemented
│       ├── _transformer_bottleneck.py     # TransformerBottleneck, hand-implemented
│       ├── unet.py                        # build_unet_vgg16 (SMP) — existing-model baseline
│       ├── unet_vgg16_ext.py              # VGG16UNetExt — architecture-probe model
│       ├── custom_lwir_unet.py            # CustomLWIRUNet — the headline custom model
│       └── custom_unet.py                 # earlier from-scratch U-Net (kept for reference)
├── scripts/
│   ├── download.py                        # idempotent Google-Drive download via gdown
│   ├── probe_architectures.py             # runs the VGG16UNetExt architecture probe
│   ├── export_onnx.py                     # exports a VGG16UNetExt variant to ONNX (Netron)
│   └── export_runs_onnx.py                # exports trained run checkpoints to ONNX
├── notebooks/
│   ├── 01_data_exploration.ipynb          # class balance, image stats, sample renders
│   ├── 02_train_colab.ipynb               # Colab/Kaggle driver: live tqdm + plots
│   ├── 03_probe_kaggle.ipynb              # VGG16UNetExt probe notebook (T4 + AMP)
│   ├── 04_sequential_runs.ipynb           # sequential multi-config training driver
│   ├── 05_analysis.ipynb                  # cross-run comparison tables + plots
│   └── 06_assignment_colab.ipynb          # SELF-CONTAINED assignment notebook (all code inlined)
├── 12_Pytorch_SemanticSegmentation.ipynb  # the in-class demonstrator (reference)
├── exports/                               # ONNX exports for Netron (gitignored, generated)
├── runs/                                  # one subdir per training run: config.json, metrics.csv, checkpoints
│   └── analysis/                          # aggregated comparison tables + plots
├── tests/                                 # pytest suite
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_models.py                     # CustomUNet + VGG16 builder + adaptation
│   └── models/
│       ├── test_unet_vgg16_ext.py         # VGG16UNetExt + AttentionGate + TransformerBottleneck + aux
│       └── test_custom_lwir_unet.py       # CustomLWIRUNet + DSConv blocks + param-count regression
├── requirements.txt
└── .gitignore
```

## Quickstart

### Local laptop — smoke test

Validates the pipeline end-to-end on Mac (MPS) or CPU in ~1 minute.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download.py        # ~2 min, ~500 MB; idempotent
python -m src.splits              # writes data/splits/split.json
python -m src.stats               # writes data/splits/stats.json + class_pixel_counts.json

python -m src.train \
    --augmentation A --epochs 1 --subset 30 \
    --output-dir runs/smoke_test
```

`src.train` autodetects device: CUDA → MPS → CPU. `--subset N` caps the
training set to the first N images for fast iteration.

To smoke-test the models instead of the SMP baseline:

```bash
# CustomLWIRUNet — the headline custom model
python -m src.train --model custom_lwir --loss focal --epochs 1 --subset 30

# VGG16UNetExt probe model, plain
python -m src.train --model vgg16_ext --epochs 1 --subset 30

# VGG16UNetExt with the transformer bottleneck
python -m src.train --model vgg16_ext --transformer-bottleneck --epochs 1 --subset 30

# VGG16UNetExt full stack (aux heads), focal loss, AMP (CUDA-only)
python -m src.train --model vgg16_ext \
    --attention-gates --transformer-bottleneck --aux-heads \
    --loss focal --amp \
    --epochs 1 --subset 30
```

### Architecture probe (recommended for a clean comparison)

`scripts/probe_architectures.py` runs all six configs (`base`, `att`,
`trans`, `att_trans`, `trans_aux`, `att_trans_aux`) sequentially with
matched hyperparameters and writes a `summary.json` plus per-config
`metrics.csv` and checkpoints:

```bash
python scripts/probe_architectures.py                          # all four core configs
python scripts/probe_architectures.py --configs trans_aux      # just one
python scripts/probe_architectures.py --no-amp                 # FP32 forced
```

### ONNX export

Two scripts, two purposes (both need `pip install onnx onnxscript onnxruntime`,
not in `requirements.txt`):

**`scripts/export_runs_onnx.py`** — export *trained* run checkpoints to ONNX
for deployment (ONNX Runtime, TensorRT, mobile). Reads each
`runs/<name>/checkpoint_best.pt`, exports `runs/<name>/model.onnx` (opset 18,
dynamic batch + spatial axes, weights inlined into a single self-contained
file), and verifies the export numerically against PyTorch:

```bash
python scripts/export_runs_onnx.py                              # all three configs
python scripts/export_runs_onnx.py --runs custom_lwir           # just one
python scripts/export_runs_onnx.py --runs-dir runs --runs custom_kaggle scratch_kaggle
```

**`scripts/export_onnx.py`** — export an *untrained* `VGG16UNetExt` variant
purely for graph inspection in [netron.app](https://netron.app):

```bash
python scripts/export_onnx.py --transformer-bottleneck --output exports/trans.onnx
```

The `06_assignment_colab.ipynb` notebook also has a built-in ONNX export
cell (section 8) that runs after training and writes the files into each
run directory.

### Colab / Kaggle — full runs

- **`notebooks/06_assignment_colab.ipynb`** — **the assignment deliverable.**
  Fully self-contained: *only the dataset is downloaded*, every line of model,
  data, and training code is inlined as notebook cells (no `git clone`). Runs
  the complete three-way 50-epoch benchmark end-to-end, renders the full
  results section (comparison table, Pareto plots, per-class IoU, convergence
  curves, confusion matrices, qualitative predictions, FLOPs-vs-resolution),
  and exports ONNX. Open in Colab or Kaggle → GPU (T4) → Run all. ~2.5–3 h.
- **`notebooks/05_analysis.ipynb`** — cross-run comparison: reads `runs/*/`
  artefacts and builds the comparison table + Pareto / per-class / convergence
  plots. Runs on a laptop, no GPU.
- **`notebooks/02_train_colab.ipynb`** — original baseline driver. Streams
  `src.train` stdout into a tqdm bar + live loss/mIoU plot, optional Drive
  sync. One augmentation per cell.
- **`notebooks/03_probe_kaggle.ipynb`** / **`04_sequential_runs.ipynb`** —
  the `VGG16UNetExt` architecture-probe notebooks for Kaggle. Clone repo,
  install deps, download data, run `scripts/probe_architectures.py`, plot
  per-config training curves. Designed for headless "Save & Run All".

**Kaggle tips** (lessons learned the hard way):

- **Pick T4 ×2**, *not* P100. PyTorch ≥ 2.5 dropped sm_60 from the official
  CUDA binaries, and Kaggle's pre-installed PyTorch now refuses to run on
  P100 (`CUDA error: no kernel image is available for execution on the
  device`). The trainer's single-GPU code only uses one of the two T4s; the
  second slot is unused but harmless.
- **Enable AMP** for T4 — `scripts/probe_architectures.py` does this by
  default. Delivers ~2.5× speedup and ~50 % less activation memory, which
  is required to fit the attention-gate configs at batch=8 on T4's 16 GB.
- **Enable Internet** in the right-hand notebook settings (one-time phone
  verification on the Kaggle account).
- **Save your GitHub PAT as a Kaggle Secret** named `github_pat` (Add-ons →
  Secrets) so the notebook clones in headless "Save & Run All" mode. Public
  repo? Leave it empty.
- **`NUM_WORKERS = 4`** matches Kaggle's ~4 vCPUs.
- Save Version → Output tab → Download All for the run artefacts.

**Colab tips:**

- Free T4 has only 2 vCPUs, so `NUM_WORKERS = 2` is correct (default).
- Free GPU access is rate-limited; expect cooldowns after heavy use.
- Use `DRIVE_RUNS_DIR = '/content/drive/MyDrive/massmind_runs'` (not bare
  `/content/drive/...`, which is read-only).
- The keepalive cell helps with idle disconnects; pair it with a no-sleep
  laptop setting.

## How the pieces fit together

```
LWIR image (640×512, 8 or 16 bit)
        + 7-class mask                ──► augmentation (A | B | C)
                                          + normalise → tensor
                                                │
                                                ▼
                                ┌──────────────────────────────┐
                                │ CustomLWIRUNet (custom)  OR   │
                                │ U-Net + VGG-16 (baseline)     │
                                └──────────────────────────────┘
                                                │
                                                ▼ argmax
                                       per-pixel class ID ∈ [0..6]
                                                │
                                                ▼
                          ConfusionMatrixTracker → mIoU, Precision, Recall, F1, per-class IoU
```

### Class scheme

| ID | Class | Pixel share | Notes |
|----|-------|-------------|-------|
| 0 | sky | 30.3 % | usually top of frame |
| 1 | water | 51.1 % | dominant class |
| 2 | bridge | 1.6 % | static, urban |
| 3 | obstacle | 0.9 % | inanimate (buoys, boats, kayaks) |
| 4 | living_obs | **0.05 %** | animate (humans, birds) — extremely rare |
| 5 | background | 10.9 % | shoreline, trees, land |
| 6 | self | 3.1 % | the recording vessel itself |
| 255 | (ignore) | — | augmentation border sentinel; trainer uses `ignore_index` |

### Augmentation pipelines (`src/augmentations.py`)

| | Name | Contents |
|---|------|----------|
| A | "MassMIND-replicated" | Rotations ±2/±5/±7°, horizontal flip. Mirrors the paper's Sec. 5.1. |
| B | "Extended"           | A + random crop+resize, CLAHE, mild Gaussian noise. |
| C | "None"               | Normalisation + tensor conversion. Used as the no-aug baseline (Run 1) and as the val/test pipeline. |

Two things deliberately excluded from all three pipelines:

- **Vertical flip** — sky stays on top in maritime imagery; flipping is
  physically wrong.
- **Brightness / contrast jitter** — in LWIR the pixel intensity *is* the
  class signal. The paper explicitly reports this hurt their results, and our
  pipeline B's mild noise injection alone was enough to wipe out the
  `obstacle` class in the baseline runs (see Results).

## Early baseline runs — U-Net + VGG-16, 20 epochs, cross-entropy

These are the **earliest pipeline-validation runs** (cross-entropy loss,
20 epochs) — superseded by the "Final results" above, but kept because two
of their findings shaped later decisions. From `runs/kaggle_extract/summary.csv`:

| Aug | best ep | val mIoU | sky | water | bridge | obstacle | **living_obs** | background | self |
|-----|---------|----------|-------|-------|--------|----------|----------------|------------|------|
| A   | 18      | 0.739    | 0.986 | 0.990 | 0.718  | 0.641    | **0.000**      | 0.893      | 0.945 |
| B   | 20      | 0.643    | 0.985 | 0.979 | 0.719  | **0.000**| **0.000**      | 0.877      | 0.940 |
| C   | 20      | 0.744    | 0.986 | 0.990 | 0.723  | 0.666    | **0.000**      | 0.896      | 0.946 |

Two findings, both **consistent with the MassMIND paper**:

1. **`living_obs` IoU collapsed to exactly 0** with plain cross-entropy at
   20 epochs. The class is ~1/2000 as common as water; CE gradient is
   dominated by the easy majority and never pushes the rare logit high
   enough to win an `argmax`. **This drove the switch to focal loss** — and
   the final 50-epoch focal-loss runs lift `living_obs` to 0.63–0.69 IoU on
   all three configs (see "Final results"), so the planned τ-threshold
   decoding fix turned out to be unnecessary.
2. **Pipeline B destroys the `obstacle` class** (0.666 → 0.00008). Even
   modest photometric perturbations break LWIR — the MassMIND paper's
   Section 5 explicitly warns against brightness jitter. This is why the
   final runs use Pipeline A (geometric-only augmentation).

A and C land within 0.005 mIoU of each other for the baseline, suggesting the
geometric augmentations in A add little over plain normalisation when training
from a pretrained encoder.

## Evaluation metrics

Required by the assignment:

- **IoU** per class and macro-averaged (mIoU)
- **Precision** per class
- **Recall** per class
- **Total parameter count** and **trainable parameter count**

Additional, reported in `06_assignment_colab.ipynb`'s results section:

- **Pixel accuracy** and per-class IoU at the best epoch
- **Forward-pass GFLOPs** at the native 512×640 resolution, and a
  compute-vs-resolution sensitivity sweep
- **Per-class confusion matrices** (one per config)
- **Training time per epoch** (logged to each run's `metrics.csv`)

Metrics are computed via the `ConfusionMatrixTracker` in `src/metrics.py` —
a streaming confusion matrix accumulated over the full validation pass,
giving exact dataset-level IoU (`TP / (TP + FP + FN)`) per class plus a
macro aggregate.

## Tests

```bash
pytest tests/ -q
```

106 tests, ~10 s on a laptop. Covers:

- Dataset loading + augmentation invariants (`test_dataset.py`)
- `ConfusionMatrixTracker` correctness (`test_metrics.py`)
- SMP-VGG16 builder + channel-mean adaptation (`test_models.py`)
- Hand-rolled `CustomUNet` blocks + forward + backward (`test_models.py`)
- `VGG16UNetExt` full 2×2 ablation, seam wiring (Identity vs AttentionGate,
  DoubleConv vs TransformerBottleneck), pretrained channel-mean adaptation,
  deep-supervision tuple-vs-tensor output dispatch, determinism
  (`tests/models/test_unet_vgg16_ext.py`)
- `CustomLWIRUNet` + depthwise-separable blocks: forward/backward, train-vs-eval
  output dispatch (3-tuple vs single tensor), GroupNorm group-count safety,
  and a parameter-count regression guard (`tests/models/test_custom_lwir_unet.py`)

## References

- Nirgudkar, S., DeFilippo, M., Sacarny, M., Benjamin, M., Robinette, P.
  (2023). *MassMIND: Massachusetts Maritime INfrared Dataset.* International
  Journal of Robotics Research, 42(1–2), 21–32.
  DOI: [10.1177/02783649231153020](https://doi.org/10.1177/02783649231153020).
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional
  Networks for Biomedical Image Segmentation.* MICCAI.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017).
  *Focal Loss for Dense Object Detection.* ICCV.
- Chen, J. et al. (2021). *TransUNet: Transformers Make Strong Encoders for
  Medical Image Segmentation.* arXiv:2102.04306.
- Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the
  Pancreas.* MIDL.
- Zhao, H. et al. (2017). *Pyramid Scene Parsing Network (PSPNet).* CVPR.
- Zhou, Z. et al. (2018). *UNet++: A Nested U-Net Architecture for Medical
  Image Segmentation.* DLMIA.
- Howard, A. et al. (2017). *MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications.* arXiv:1704.04861. — depthwise-
  separable convolutions, the core of `CustomLWIRUNet`'s efficiency.
- Wu, Y., He, K. (2018). *Group Normalization.* ECCV. — the batch-size-
  independent normaliser used throughout `CustomLWIRUNet`.
- Elfwing, S., Uchibe, E., Doya, K. (2018). *Sigmoid-Weighted Linear Units
  (SiLU).* Neural Networks. — the activation used throughout `CustomLWIRUNet`.
- Upstream MassMIND repository: <https://github.com/uml-marine-robotics/MassMIND>
