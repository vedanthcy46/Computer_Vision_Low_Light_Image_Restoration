# Datasets (Problem Statement 1)

The datasets are **not** committed. The hackathon datasheet lists four sources:

| Dataset | Use here | Where |
|---|---|---|
| **LOL** (Wei et al., BMVC 2018) | paired fine-tuning / test (485 train + 15 test pairs) | https://daooshee.github.io/BMVC2018website/ |
| **LOL-v2** (Real_captured + Synthetic) | paired fine-tuning / unseen test | https://www.kaggle.com/datasets/tanhyml/lol-v2-dataset |
| **Zero-DCE** research implementation | model architecture / training recipe (re-implemented in `lowlight/dl/zero_dce.py`) | https://github.com/Li-Chongyi/Zero-DCE |
| **MIT-Adobe FiveK** | well-exposed images -> NIQE model fit, and synthetic-degradation training data | https://data.csail.mit.edu/graphics/fivek/ |

> Note: the datasheet links "Zero-DCE" to `cchen156/Zero-DCE`; the original Zero-DCE code is `Li-Chongyi/Zero-DCE`.

## Expected layout

```
data/
  LOL/
    our485/{low,high}/*.png        # train pairs
    eval15/{low,high}/*.png        # test pairs
  LOL-v2/
    Real_captured/{Train,Test}/{Low,Normal}/{low,normal}XXXXX.png
    Synthetic/{Train,Test}/{Low,Normal}/...
  unseen/                           # any extra low-light photos (no ground truth) - for the "unseen images" test
```

`lowlight.data.find_pairs` discovers `low/ + high/` (LOL) and `Low/ + Normal/` (LOL-v2) automatically and matches
files by number, so the exact sub-folder depth does not matter.

## Download helpers

LOL-v2 (needs a Kaggle token in `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d tanhyml/lol-v2-dataset -p data/LOL-v2 --unzip
```

LOL is distributed through Google Drive / the project page above - download the archive manually and unzip into `data/LOL/`.

## Train / test hygiene

* Fine-tune only on the **train** splits (`our485`, `Real_captured/Train`, `Synthetic/Train`).
* Report numbers on `eval15`, `Real_captured/Test`, `Synthetic/Test` and on the `unseen/` folder - never on images seen in training.
