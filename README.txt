================================================================
 Quick Start
================================================================

Thank you for taking the time to review our paper. This
repository contains the training, evaluation, and analysis code
behind every experiment in the paper. We hope it makes the
results easy to inspect and reproduce.

For full reproduction instructions (paths, dataset prep, all
CLI flags, license notes, pretrained checkpoint links), please
see README.md in this directory.


----------------------------------------------------------------
 Top-level layout
----------------------------------------------------------------

train_event_teacher/
    Part 1. Trains the event-domain ImageNet-1k classifier
    (DiST representation, ResNet34 by default) that serves as
    the teacher for the cross-domain knowledge-distillation
    experiments.

resnet34_variant/
    Part 2. Trains every RGB ResNet34 student studied in the
    paper -- baseline, ED_Student (the main result), and the
    comparison variants used in the ablations and robustness
    sections. One script per recipe; see README.md for the
    paper-variant -> script mapping.

eval_adversarial_robustness/
    Standalone scripts for the FGSM / PGD adversarial-robustness
    evaluations reported in Sections 4.3 and 6.

variant_utils.py, dist_utils.py,
kd_datamodules.py, kd_module.py
    Shared helpers used by the Part 2 scripts: data loaders,
    transforms, the synced distributed sampler, KD loss,
    optimiser / schedule helpers, and the Event -> RGB
    distillation Lightning module.

lists/             [shipped EMPTY -- download from Google Drive]
Datasets/          [shipped EMPTY -- download from Google Drive]
    Two large index files (paired RGB/event TSVs and N-ImageNet
    .npz path lists) that drive the data loaders. They were
    too large to bundle directly; download links and the
    expected post-download layout are documented in README.md
    under "Auxiliary Data Files (Hosted on Google Drive)".

README.md
    Full reproduction guide -- pretrained checkpoint links,
    dataset preparation, command-line flags, license notes.

README.txt
    This file.
