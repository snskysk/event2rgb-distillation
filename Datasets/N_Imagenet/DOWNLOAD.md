# `Datasets/N_Imagenet/` — download the list files from Google Drive

The N-ImageNet train / val **list files** are too large for this repository and
are hosted on Google Drive. Download and place them **here** (keep the filenames):

- **Drive folder:** https://drive.google.com/drive/folders/1VNT2bbAGzxuISsiOGPK0JKFJK8d_fRgG?usp=sharing
- **Files restored:** `train_list.txt`, `val_list.txt`

Note: `Datasets/mapping.txt` (the class-ID map) is small and is already included
in this repository. The actual N-ImageNet event data (`.npz`) must be downloaded
from the [official N-ImageNet repository](https://github.com/82magnolia/n_imagenet).
See the top-level `README.md` (Part 1 "Dataset Preparation") for the expected
layout and how to retarget the paths.
