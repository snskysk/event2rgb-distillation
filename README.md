# The Shape of Events: Edge-Based Inductive Biases via Cross-Domain Distillation

Official code release for our **NeurIPS 2026** paper.

Soshun Kihara¹, Shunsuke Yasuki², Masato Taki¹ ³ &nbsp;(*equal contribution*)
¹ Graduate School of Artificial Intelligence and Science, Rikkyo University &nbsp;·&nbsp; ² The University of Tokyo &nbsp;·&nbsp; ³ RIKEN

📄 Paper: arXiv link — *to appear* &nbsp;·&nbsp; 📦 Pretrained checkpoints & auxiliary data: Google Drive (links below)

The code is organised into two parts:

1. **`train_event_teacher/`** — Trains an event-based ImageNet-1K classifier. This model serves as the teacher in knowledge distillation.
2. **`resnet34_variant/`** — Trains student ResNet34 variants (distilled from the teacher, and several baselines for comparison). *See the corresponding section below.*

Shared utilities used by both parts live in `train_event_teacher/base/`.

---

## Pretrained Checkpoints

We release pretrained checkpoints for every model used in the paper so multi-day training runs can be skipped. Each link below points to the corresponding checkpoint on Google Drive.

### Teacher models (Part 1 outputs)

| Checkpoint | Backbone | Input modality | Primary use |
|---|---|---|---|
| [DiST_resnet34](https://drive.google.com/file/d/1DsAhYVew7MF5WdORH6LFnhTeXOkVcoQ2/view?usp=drive_link) | ResNet34 | Event (DiST, 2 ch) | Event teacher for `train_event_distilled.py --dist_weight` (main experiment and ED_* variants). |
| [Gray_resnet34](https://drive.google.com/file/d/1D2wwb6FQiCp5E8SxxX7ILnQY6G1CbiOw/view?usp=drive_link) | ResNet34 | 3-channel grayscale | Teacher for `train_gray_distilled.py --teacher_ckpt`. |
| [B&W_resnet34](https://drive.google.com/file/d/1Uwpg9rAlmNTZeTA-TLL_A6Zv5gzyhE59/view?usp=drive_link) | ResNet34 | 3-channel binarised | Teacher for `train_bw_distilled.py --teacher_ckpt`. |
| [DiST_resnet50](https://drive.google.com/file/d/1EEZiodwRBAVpsXo25F7E44zxDpzi_cyc/view?usp=drive_link) | ResNet50 | Event (DiST, 2 ch) | Architectural generality ablation (Section 6.2). |
| [DiST_vgg16_bn](https://drive.google.com/file/d/1lgSjTE4ToJB3YkpiF91XgD07Gr83FlVT/view?usp=drive_link) | VGG16-BN | Event (DiST, 2 ch) | Architectural generality ablation (Section 6.2). |
| [DiST_convnextB](https://drive.google.com/file/d/1KbpnhpOgl6priG6DIb1ILZYEEsdEOXZv/view?usp=drive_link) | ConvNeXt-Base | Event (DiST, 2 ch) | Architectural generality ablation (Section 6.2). |
| [DiST_replknet31B](https://drive.google.com/file/d/1Pxrdb_CQznqK5JCdh8XNN3HT9k-Njck1/view?usp=drive_link) | RepLKNet31B | Event (DiST, 2 ch) | Architectural generality ablation (Section 6.2). |

### Student models (Part 2 outputs)

All student checkpoints are ResNet34 trained on ImageNet-1k. The second column gives the minimal reproducing command; common flags (paths, optimiser, batch size, …) follow the Part 2 defaults.

| Checkpoint | Reproducing command |
|---|---|
| [resnet34_baseline](https://drive.google.com/file/d/1v77HSIhIGgyVW8IQQ4OX2I-29tGs7Rk0/view?usp=drive_link) | `train_event_distilled.py --alpha_ce 1.0 --alpha_kd 0.0 --dist_weight <DiST_resnet34>` |
| [resnet34_ED_Student](https://drive.google.com/file/d/1j0fpsp8pnYiKp3P-V6ZSznsaqhw7aQgv/view?usp=drive_link)  | `train_event_distilled.py --alpha_ce 0.8 --alpha_kd 0.2 --dist_weight <DiST_resnet34>` |
| [resnet34_ED_SoftLabelOnly](https://drive.google.com/file/d/1iFja5Yg4Vs7P_BFBUlQQTHYXtokKTYZM/view?usp=drive_link) | `train_event_distilled.py --alpha_ce 0.0 --alpha_kd 1.0 --dist_weight <DiST_resnet34>` |
| [resnet34_Self Distilled](https://drive.google.com/file/d/1g2HR9qoB2qeY8XJf2E7CDuzsyqVdcTUV/view?usp=drive_link) | `train_self_distilled.py --teacher_ckpt <RGB ResNet34 baseline>` |
| [resnet34_Gray Distilled](https://drive.google.com/file/d/1KriXe5dyVRYtfCal2mwnLKrIrFyI1N9B/view?usp=drive_link) | `train_gray_distilled.py --teacher_ckpt <Gray_resnet34>` |
| [resnet34_B&W Distilled](https://drive.google.com/file/d/1yPeZdoyeTJaEWehUySkkXu78CL8MKEk4/view?usp=drive_link) | `train_bw_distilled.py --teacher_ckpt <B&W_resnet34>` |
| [resnet34_Adversarial Training](https://drive.google.com/file/d/18x-YJTPZ6VmmMGKppZuwAJ-5Cd8ajteI/view?usp=drive_link) | `adversarial_training.py --pretrained_ckpt <resnet34_baseline>` |
| [resnet34_Noise Training](https://drive.google.com/file/d/164cVgXzdzVGo67NDwJ2o2dm8ukGWfKwm/view?usp=drive_link) | `noise_training.py --pretrained_ckpt <resnet34_baseline>` |
| [resnet34_SIN](https://drive.google.com/file/d/1jks7NcaNZw7gRJm34Jz5c651UsXNbZfn/view?usp=drive_link) | `train_resnet34_sin.py --sin_train_root <Stylized-ImageNet train>` (SIN+IN co-training; Geirhos et al. 2019) |

---

## Auxiliary Data Files (Hosted on Google Drive)

The two large index directories below are **not stored in this repository** (they exceed sensible repo size limits). Download their contents from the Drive folders linked in the table and place each file under the same relative path before running anything.

| Empty directory | Drive folder (download contents into the directory) | File(s) restored | Approx. size |
|---|---|---|---|
| `event-distill-neurips2026/lists/` | [lists/](https://drive.google.com/drive/folders/1h7uJ58eVKdKBAlSgZERnJbiJPEN3e9dy?usp=sharing) | `paired_train.tsv`, `paired_val.tsv`, `imagenet_train_ratio1.0_seed1.txt` | ~256 MB |
| `event-distill-neurips2026/Datasets/` | [Datasets/](https://drive.google.com/drive/folders/1VNT2bbAGzxuISsiOGPK0JKFJK8d_fRgG?usp=sharing) | `mapping.txt`, `N_Imagenet/train_list.txt`, `N_Imagenet/val_list.txt` | ~85 MB |

After download, the resulting layout must be:

```
event-distill-neurips2026/
├── lists/
│   ├── paired_train.tsv
│   ├── paired_val.tsv
│   └── imagenet_train_ratio1.0_seed1.txt
└── Datasets/
    ├── mapping.txt
    └── N_Imagenet/
        ├── train_list.txt
        └── val_list.txt
```

These files are large list / TSV indices (paths to N-ImageNet `.npz` events and ImageNet `.JPEG` images) consumed by the Part 1 `.ini` configs and by every Part 2 training script. Without them, training will fail at data-loading time. The actual image / event data themselves are *not* on Drive — those still need to be downloaded from the original ImageNet / N-ImageNet sources, as described in each Part's "Dataset Preparation" subsection.

---

## Part 1: Training the Event Teacher (`train_event_teacher/`)

### Overview

Events from the [N-ImageNet](https://github.com/82magnolia/n_imagenet) dataset are aggregated into the DiST representation (`reshape_then_acc_adj_sort`) and fed to a CNN classifier trained for 1000-way classification. **ResNet34 is the backbone used throughout the paper**; alternative backbones are provided for ablation.

### Directory Layout

```
train_event_teacher/
├── main.py                  # Entry point
├── configs/                 # Per-backbone config files (.ini)
├── data/
│   ├── data_container.py    # DataLoader wrapper
│   └── imagenet.py          # Dataset + event-aggregation functions
├── models/
│   ├── model_container.py   # Model construction / checkpoint loading
│   └── replknet.py          # RepLKNet31B (third-party, MIT)
├── train/
│   └── trainer.py           # Training / evaluation loop
└── base/                    # Shared abstract classes
                             # (DataContainer, ModelContainer, CommonTrainer,
                             #  metrics, config parser, …)
```

### Environment

- Python ≥ 3.8
- PyTorch with CUDA (any version compatible with a matching `torch-scatter` wheel)
- `torchvision`, `torch-scatter`, `timm`, `numpy`, `matplotlib`, `tensorboard`

We conducted our experiments with PyTorch `2.1.0` / CUDA `12.2`.

### Dataset Preparation

Part 1 only needs **N-ImageNet** — RGB ImageNet is not involved in the teacher training pipeline. The class-ID mapping (`Datasets/mapping.txt`) and the train / val list files (`Datasets/N_Imagenet/{train_list,val_list}.txt`) are restored from the Drive folder above.

Download N-ImageNet from its [official repository](https://github.com/82magnolia/n_imagenet).

**Zero-edit layout.** If you place N-ImageNet exactly as below, **no file needs to be edited** — the shipped `.ini` configs and the restored `.txt` lists already point here:

```
/datasets/N-ImageNet/
├── training/
│   ├── Part_1/<wnid>/*.npz
│   ├── Part_2/<wnid>/*.npz
│   └── ...
└── validation/extracted_val/<wnid>/*.npz
```

This is the directory layout produced by the authors' own extraction scripts.

**If your N-ImageNet lives elsewhere,** retarget the restored list files once. Each line in these files is an absolute path to a single `.npz` event file:

```bash
cd event-distill-neurips2026
sed -i 's|/datasets/N-ImageNet|/your/path/to/N-ImageNet|g' \
    Datasets/N_Imagenet/train_list.txt \
    Datasets/N_Imagenet/val_list.txt
```

The `.ini` configs reference these lists via `train_file=../Datasets/N_Imagenet/train_list.txt` / `val_file=../Datasets/N_Imagenet/val_list.txt` and need no further edits.

### Main Experiment — ResNet34

Run from the `train_event_teacher/` directory (config paths are resolved relative to this working directory):

```bash
cd train_event_teacher
python main.py --config configs/event_resnet34.ini --override mode=train
```

The shipped `event_resnet34.ini` has `mode=test` so that a trained checkpoint can be evaluated directly; use `--override mode=train` (or edit the config) to train from scratch.

To evaluate a trained checkpoint:

```bash
python main.py --config configs/event_resnet34.ini \
               --override load_model=./experiments/<run_name>/model_log/checkpoint_epoch_<E>_iter_<I>_val_<V>.tar
```

### Alternative Backbones

Invocation pattern is identical; only the config file changes.

| Backbone    | Config                              |
|-------------|-------------------------------------|
| ResNet34    | `configs/event_resnet34.ini`        |
| ResNet50    | `configs/event_resnet50.ini`        |
| VGG16-BN    | `configs/event_vgg16_bn.ini`        |
| ConvNeXt-B  | `configs/event_convnext_base.ini`   |
| RepLKNet31B | `configs/event_replknet31B.ini`     |

### Command-Line Flags

| Flag                   | Purpose                                                                                   |
|------------------------|-------------------------------------------------------------------------------------------|
| `--config PATH`        | Required. Path to the `.ini` config file.                                                 |
| `--override KEY=VALUE` | Override a single config value at run time (e.g. `--override learning_rate=1e-4`).        |
| `--cutmix`             | Enable CutMix augmentation.                                                               |
| `--mixup`              | Enable MixUp augmentation. Can be combined with `--cutmix` (one is chosen per batch).     |
| `--clean`              | Remove the experiment directory and exit.                                                 |

### Selected Config Fields

| Field | Meaning |
|-------|---------|
| `mode` | `train` or `test`. |
| `model` | Backbone name (see table above). |
| `channel_size` | Input channels fed to the backbone. The DiST representation uses `2`. |
| `loader_type` | Event-aggregation method. `reshape_then_acc_adj_sort` is DiST; other options are provided for ablation (see `LOADER_REGISTRY` in `data/imagenet.py`). |
| `slice_length`, `slice_method` | Number of events per sample and how to draw them. |
| `optimizer`, `learning_rate`, `weight_decay`, `epochs`, `batch_size` | Standard training hyper-parameters. |
| `save_root_dir`, `save_by`, `save_every` | Checkpoint destination and frequency. `save_by` is `epoch` or `iter`. |
| `parallel` | When `True`, wraps the model in `DataParallel` across all visible GPUs. |

### Outputs

A training run produces (under `save_root_dir / <cfg.name>/`):

- `config.ini` — the effective config after overrides.
- `model_log/checkpoint_epoch_{E}_iter_{I}_val_{V}.tar` — checkpoints saved every `save_every` units. Each payload is `{'epoch' or 'iter': counter, 'state_dict': state_dict}`.
- `{cfg.name}_{mode}_{timestamp}/` — TensorBoard event files.

A `mode=test` run instead writes `test_result.tar` containing the top-1 accuracy.

### Notes

- **Working directory matters.** Always invoke `python main.py` from `train_event_teacher/`; the paths inside the shipped configs (`./experiments/`, `../Datasets/…`) are resolved relative to cwd.
- **Experiment naming.** The `name` field in the shipped configs is a placeholder. Override it (e.g. `--override name=event_resnet34_seed1`) to keep runs distinguishable under `save_root_dir`.
- **Multi-GPU.** With `parallel=True`, `DataParallel` is applied automatically to all visible CUDA devices; adjust `CUDA_VISIBLE_DEVICES` to restrict.

---

## Part 2: Student Training and Robustness Baselines (`resnet34_variant/`)

### Overview

Part 2 trains the RGB ResNet34 students studied in the main paper plus every comparison variant used in the ablations and robustness sections. All scripts share the same TSV input layout and are driven by a common utility module (`variant_utils.py`), so adding a new variant mostly means writing a new LightningModule rather than duplicating data-loading code.

Paper ↔ script correspondence:

| Paper variant          | Script                                      | Recipe |
|------------------------|---------------------------------------------|--------|
| baseline               | `train_event_distilled.py`                  | `α_ce=1.0, α_kd=0.0` (pure CE) |
| **ED_Student** (main)  | `train_event_distilled.py`                  | `α_ce=0.8, α_kd=0.2` |
| ED_SoftLabelOnly       | `train_event_distilled.py`                  | `α_ce=0.0, α_kd=1.0` |
| Self Distilled         | `train_self_distilled.py`                   | RGB teacher → RGB student |
| Gray Distilled         | `train_gray_distilled.py`                   | 3-channel grayscale teacher → RGB student |
| B&W Distilled          | `train_bw_distilled.py`                     | Binarised (Otsu) teacher → RGB student |
| Adversarial Training   | `adversarial_training.py`                   | PGD fine-tuning |
| Noise Training         | `noise_training.py`                         | Gaussian noise augmentation |
| SIN                    | `train_resnet34_sin.py`                     | SIN+IN co-training (Geirhos et al. 2019) |

`teacher_train.py` is a helper that trains the Gray / B&W ResNet34 teachers consumed by the corresponding KD scripts (the RGB teacher used by `train_self_distilled.py` is just a vanilla CE checkpoint; any ImageNet-1k ResNet34 checkpoint works).

The **SIN** baseline trains a ResNet34 from scratch on `ConcatDataset(ImageNet-1k train, Stylized-ImageNet train)` with a single shared WNID-keyed label space, following [Geirhos et al. 2019](https://github.com/rgeirhos/texture-vs-shape). It uses the same optimiser / schedule / augmentation as `teacher_train.py` so that downstream evaluations are apples-to-apples with the rest of the Part 2 baselines. The pretrained checkpoint we evaluated is mirrored on Drive [here](https://drive.google.com/file/d/1jks7NcaNZw7gRJm34Jz5c651UsXNbZfn/view?usp=drive_link).

### Directory Layout

```
event-distill-neurips2026/
├── variant_utils.py       # Shared TSV loader, sampler, datasets, transforms,
│                          #   teacher-ckpt loader, KD loss, LR schedule, trainer helper
├── dist_utils.py          # Hooks into Part 1's DiST pipeline (event teacher)
├── kd_datamodules.py      # DataModule for Event → RGB distillation
├── kd_module.py           # LightningModule for Event → RGB distillation
├── lists/
│   ├── paired_train.tsv   # <rgb>\t<event>\t<wnid>\t<stem> training pairs
│   ├── paired_val.tsv     # same, validation split
│   └── imagenet_train_ratio1.0_seed1.txt  # Single-column RGB training list
└── resnet34_variant/
    ├── train_event_distilled.py
    ├── train_self_distilled.py
    ├── train_gray_distilled.py
    ├── train_bw_distilled.py
    ├── adversarial_training.py
    ├── noise_training.py
    ├── train_resnet34_sin.py
    └── teacher_train.py
```

### Environment

Additional to Part 1's requirements:

- `pytorch-lightning` (imported as `lightning`) ≥ 2.0
- `opencv-python` (only when the B&W script falls back to on-the-fly Otsu binarisation)

### Dataset Preparation

Part 2 scripts consume images in two ways:

- **Via absolute paths in `lists/paired_train.tsv`** (restored from Drive — see [Auxiliary Data Files](#auxiliary-data-files-hosted-on-google-drive) above) — each line is `<rgb_path>\t<event_path>\t<wnid>\t<stem>`. Columns 1 and 2 are the two image paths; columns 3 and 4 are bookkeeping (leave them as restored).
- **Via directory-root CLI arguments** — `--rgb_train_root`, `--rgb_val_root`, `--bw_train_root`, `--bw_val_root`. Their CLI defaults already match the zero-edit layout below.

Three datasets are involved:

1. **ImageNet-1k** — standard ILSVRC2012 train/val splits.
2. **N-ImageNet-1k** — downloaded from the [official repository](https://github.com/82magnolia/n_imagenet). Part 2 validates on the RGB split, so only the N-ImageNet *training* half is read (via TSV column 2).
3. **(Optional) Binarised ImageNet** — a parallel directory tree of Otsu-binarised images used by `train_bw_distilled.py` and `teacher_train.py --modality bw`.

#### Zero-edit layout

Place the data exactly as below and **no TSV edit or CLI override is required** — the restored `lists/paired_train.tsv` and all `--*_root` defaults already point here:

```
/datasets/imagenet/
├── train/<wnid>/*.JPEG
└── val/<wnid>/*.JPEG
/datasets/N-ImageNet/
└── training/Part_*/<wnid>/*.npz
/datasets/imagenet_bw/              # optional — B&W variants only
├── train/<wnid>/*.JPEG
└── val/<wnid>/*.JPEG
```

#### If your data lives elsewhere

Retarget the TSV once and pass the matching `--*_root` overrides at invocation:

```bash
cd event-distill-neurips2026
sed -i 's|/datasets/imagenet|/your/path/to/imagenet|g; \
        s|/datasets/N-ImageNet|/your/path/to/N-ImageNet|g' \
    lists/paired_train.tsv

# Then run any Part 2 script with the matching overrides, e.g.
python resnet34_variant/train_event_distilled.py \
    --rgb_train_root /your/path/to/imagenet/train \
    --rgb_val_root   /your/path/to/imagenet/val \
    --alpha_ce 0.8 --alpha_kd 0.2 \
    --dist_weight <path to DiST teacher from Part 1>
```

#### Which path argument does each script need?

| Script                                 | `--dist_weight`   | `--teacher_ckpt` | `--rgb_{train,val}_root` | `--bw_{train,val}_root` | `--sin_train_root` |
|----------------------------------------|:-----------------:|:----------------:|:------------------------:|:-----------------------:|:------------------:|
| `train_event_distilled.py`             | **required**      | —                | ✓                        | —                       | —                  |
| `train_self_distilled.py`              | —                 | **required**     | ✓                        | —                       | —                  |
| `train_gray_distilled.py`              | —                 | **required**     | ✓                        | —                       | —                  |
| `train_bw_distilled.py`                | —                 | **required**     | ✓                        | ✓ (`--bw_train_root`)   | —                  |
| `adversarial_training.py`              | —                 | optional `--pretrained_ckpt` | ✓              | —                       | —                  |
| `noise_training.py`                    | —                 | optional `--pretrained_ckpt` | ✓              | —                       | —                  |
| `train_resnet34_sin.py`                | —                 | —                | ✓                        | —                       | **required**       |
| `teacher_train.py --modality gray`     | —                 | —                | ✓                        | —                       | —                  |
| `teacher_train.py --modality bw`       | —                 | —                | —                        | ✓ (both)                | —                  |

A subtlety worth noting: for the TSV-driven scripts, `--rgb_train_root` is only used to compute the WNID → class-index map (a directory listing); the actual RGB training images are loaded from the absolute paths in `paired_train.tsv`. `--rgb_val_root`, in contrast, is handed to `datasets.ImageFolder` and is read image-by-image — so it must point at a real ImageNet validation tree. `train_resnet34_sin.py` is *not* TSV-driven: it walks `--rgb_train_root` and `--sin_train_root` directly to enumerate images, so both must point at real on-disk trees.

`lists/paired_val.tsv` and `lists/imagenet_train_ratio1.0_seed1.txt` are provided for external use but are **not consumed by any training script** in this release.

### Main Experiment: ED_Student (ResNet34)

This reproduces the paper's primary distillation result. Run from `event-distill-neurips2026/`:

```bash
cd event-distill-neurips2026
python resnet34_variant/train_event_distilled.py \
    --alpha_ce 0.8 \
    --alpha_kd 0.2 \
    --temperature 3.0 \
    --dist_weight /path/to/dist_resnet34_teacher.tar
```

The `--dist_weight` argument points at the DiST ResNet34 checkpoint produced by Part 1 (`train_event_teacher/experiments/<name>/model_log/checkpoint_*.tar`).

### Other Paper Variants

All commands assume cwd `event-distill-neurips2026/` and use ResNet34.

**baseline** (pure CE, no KD signal):

```bash
python resnet34_variant/train_event_distilled.py --alpha_ce 1.0 --alpha_kd 0.0 \
    --dist_weight /path/to/dist_resnet34_teacher.tar
```

**ED_SoftLabelOnly** (teacher signal only):

```bash
python resnet34_variant/train_event_distilled.py --alpha_ce 0.0 --alpha_kd 1.0 \
    --dist_weight /path/to/dist_resnet34_teacher.tar
```

**Self Distilled** (RGB teacher → RGB student, Consistent-Teaching style):

```bash
python resnet34_variant/train_self_distilled.py \
    --teacher_ckpt /path/to/rgb_resnet34_teacher.ckpt
```

**Gray Distilled** (3-channel grayscale teacher → RGB student):

```bash
# 1) Train the grayscale teacher from scratch (CE only)
python resnet34_variant/teacher_train.py --modality gray

# 2) Distill into the RGB student
python resnet34_variant/train_gray_distilled.py \
    --teacher_ckpt runs/resnet34_gray_teacher/version_0/last.ckpt
```

**B&W Distilled** (binarised teacher → RGB student):

```bash
# 1) Train the B&W teacher (requires a pre-binarised ImageNet tree)
python resnet34_variant/teacher_train.py --modality bw \
    --bw_train_root /datasets/imagenet_bw/train \
    --bw_val_root   /datasets/imagenet_bw/val

# 2) Distill into the RGB student
python resnet34_variant/train_bw_distilled.py \
    --teacher_ckpt runs/resnet34_bw_teacher/version_0/last.ckpt \
    --bw_train_root /datasets/imagenet_bw/train
```

**Adversarial Training** (PGD fine-tuning of a pretrained checkpoint):

```bash
python resnet34_variant/adversarial_training.py \
    --pretrained_ckpt /path/to/baseline_resnet34.ckpt \
    --adv_eps $(python -c "print(8/255)") \
    --adv_alpha $(python -c "print(2/255)") \
    --adv_iters 10
```

**Noise Training** (Gaussian-noise augmentation fine-tuning):

```bash
python resnet34_variant/noise_training.py \
    --pretrained_ckpt /path/to/baseline_resnet34.ckpt \
    --noise_std 0.1 \
    --noise_p 0.5
```

**SIN+IN co-training** (Geirhos et al. 2019; not a KD recipe — trains from scratch on `ConcatDataset(IN, SIN)`):

```bash
python resnet34_variant/train_resnet34_sin.py \
    --rgb_train_root  /datasets/imagenet/train \
    --rgb_val_root    /datasets/imagenet/val \
    --sin_train_root  /datasets/stylized_imagenet/train
```

Stylized-ImageNet must be generated separately following [rgeirhos/Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet); this repository does not bundle the stylised images.

### Common Command-Line Flags

Every training script in Part 2 accepts the arguments below. Defaults are chosen so that a standard run from `event-distill-neurips2026/` works with the restored `lists/` TSVs (see [Auxiliary Data Files](#auxiliary-data-files-hosted-on-google-drive)) after the absolute data paths inside them are updated.

| Flag                       | Purpose                                                                  |
|----------------------------|--------------------------------------------------------------------------|
| `--paired_tsv PATH`        | Path to the RGB ⟷ event TSV. Defaults to `./lists/paired_train.tsv`.      |
| `--rgb_train_root DIR`     | RGB ImageNet-1k training root (also used to derive the class label map). |
| `--rgb_val_root DIR`       | RGB ImageNet-1k validation root.                                         |
| `--train_ratio FLOAT`      | Per-class subsample ratio for the training split (`1.0` = full data).    |
| `--batch_size`, `--num_workers`, `--epochs`, `--lr`, `--weight_decay`, `--seed` | Standard hyper-parameters.           |
| `--outdir`, `--run_name`   | Where CSV logs and checkpoints are written (`{outdir}/{run_name}/`).     |
| `--devices ID [ID ...]`    | Which CUDA devices to use (space-separated list).                        |

KD scripts additionally expose:

| Flag                       | Purpose                                                                  |
|----------------------------|--------------------------------------------------------------------------|
| `--alpha_ce`, `--alpha_kd` | Weights of the CE and KL terms. Total loss is `α_ce·CE + α_kd·KL`.       |
| `--temperature`            | Softmax temperature `T` used in the KL term (default 3.0).               |
| `--teacher_ckpt PATH`      | Lightning checkpoint of the teacher (RGB / Gray / B&W variants).         |
| `--dist_weight PATH`       | *Event-distill only.* Checkpoint of the DiST event teacher (from Part 1).|

### Outputs

Each run creates `{outdir}/{run_name}/version_N/` containing:

- `metrics.csv` — Lightning's CSVLogger output (per-epoch train / val metrics).
- `hparams.yaml` — the effective hyper-parameters.
- `epoch={E}-val_acc1={V}.ckpt` + `last.ckpt` — Lightning checkpoints saved on the best and last validation epoch, respectively.

### Notes

- **Working directory.** The scripts add `event-distill-neurips2026/` to `sys.path` on import, so modules resolve regardless of where you launch from; however, path-typed CLI defaults (`lists/…`, `./runs`) are computed relative to the script location, so running from `event-distill-neurips2026/` is still the least-surprising option.
- **TSV pairing and DDP.** CE- and KD-branch DataLoaders share a `SyncedDistributedSampler` instance to keep their sample order identical. The scripts therefore pass `use_distributed_sampler=False` to Lightning's `Trainer`; do not change this unless you drop the paired-loader requirement.
- **Teacher checkpoints are user-provided.** The DiST event teacher (`--dist_weight`) comes from Part 1; the RGB / Gray / B&W teachers come from either a standard torchvision baseline or from `teacher_train.py`. None of these are bundled in this release.

---

## Citation

If you find this code or the released checkpoints useful, please cite:

```bibtex
@inproceedings{kihara2026shape,
  title     = {The Shape of Events: Edge-Based Inductive Biases via Cross-Domain Distillation},
  author    = {Kihara, Soshun and Yasuki, Shunsuke and Taki, Masato},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

---

## Licenses

This repository reuses third-party datasets, model checkpoints, and software libraries. We credit each one below together with its license / terms of use. Users of this code are responsible for complying with the upstream terms.

### Datasets

| Asset | Source | License / terms |
|---|---|---|
| ImageNet-1k (ILSVRC2012) | [image-net.org](https://image-net.org/download.php) | Custom, non-commercial research-use license. Users must register and accept the ImageNet terms of access before downloading. |
| N-ImageNet-1k | [82magnolia/n_imagenet](https://github.com/82magnolia/n_imagenet) | Released under the terms of the official repository. Built by recapturing ImageNet-1k images on a monitor with an event camera; downstream users inherit the ImageNet research-use terms. |

### Pretrained checkpoints

| Asset | Source | License / terms |
|---|---|---|
| DiST ResNet-34 event teacher | [82magnolia/n_imagenet](https://github.com/82magnolia/n_imagenet) | Public checkpoint released by the DiST authors; reused as-is and frozen as our main event teacher. We follow the upstream repository's license terms. |
| Other DiST teachers (ResNet-50, VGG16-BN, ConvNeXt-Base, RepLKNet31B) and all ResNet-34 students released alongside this release | This repository | Released by us under the **MIT License** for research and reproduction purposes. |

### Software dependencies

| Library | Version used | License |
|---|---|---|
| Python | 3.10 | PSF License |
| PyTorch | 2.1.0 (CUDA 12.2) | BSD-style |
| `torchvision` | matched to PyTorch 2.1 | BSD-style |
| `pytorch-lightning` (`lightning`) | $\geq$ 2.0 | Apache-2.0 |
| `torch-scatter` | wheel matching the PyTorch / CUDA version | MIT |
| `timm` | latest at the time of the experiments | Apache-2.0 |
| `numpy`, `matplotlib`, `tensorboard` | latest stable | BSD-style / BSD-style / Apache-2.0 |
| `opencv-python` | latest stable | Apache-2.0 |
| RepLKNet (`models/replknet.py`) | vendored from [DingXiaoH/RepLKNet-pytorch](https://github.com/DingXiaoH/RepLKNet-pytorch) | MIT |

### This code release

The code in this directory (excluding the vendored `replknet.py`, which retains its upstream MIT license) is released under the **MIT License** for research and reproduction purposes.

