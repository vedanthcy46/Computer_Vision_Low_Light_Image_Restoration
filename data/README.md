# Datasets (Problem Statement 1)

The datasets are **not** committed. The hackathon datasheet lists four sources (we used the first three plus the extra sets below; **MIT-Adobe FiveK was not used**):

| Dataset | Use here | Where |
|---|---|---|
| **LOL** (Wei et al., BMVC 2018) | paired fine-tuning / test (485 train + 15 test pairs) | https://daooshee.github.io/BMVC2018website/ |
| **LOL-v2** (Real_captured + Synthetic) | paired fine-tuning / unseen test | https://www.kaggle.com/datasets/tanhyml/lol-v2-dataset |
| **Zero-DCE** research implementation | model architecture / training recipe (re-implemented in `lowlight/dl/zero_dce.py`) | https://github.com/Li-Chongyi/Zero-DCE |
| **MIT-Adobe FiveK** | *not used* (optional: well-exposed images for NIQE fitting) | https://data.csail.mit.edu/graphics/fivek/ |
| **BrighteningTrain** (1,000 low/high pairs, provided with the hackathon data) | extra zero-reference training images; its `high/` images also help fit the NIQE model | provided locally (`BrighteningTrain.zip`) |
| **Unseen benchmark folders** `Test.zip` (DICM, LIME, MEF, NPE, VV, Fusion) | **unseen** test images, no ground truth, never used for training | provided locally (`Test.zip`) |

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
  BrighteningTrain/{low,high}/*.png # extra training images
  unseen/{DICM,LIME,MEF,NPE,VV,Fusion}/   # unseen low-light photos (no ground truth), from Test.zip
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

The archives we used were unpacked like this (paths relative to the repo root):

```bash
unzip LOLdataset.zip -d data/LOL            # -> data/LOL/{our485,eval15}
unzip archive.zip -d data                   # LOL-v2 -> data/LOL-v2/{Real_captured,Synthetic}
unzip BrighteningTrain.zip -d data          # -> data/BrighteningTrain/{low,high}
unzip Test.zip -d data/unseen_tmp && mkdir -p data/unseen && mv data/unseen_tmp/Test/{DICM,LIME,MEF,NPE,VV,Fusion} data/unseen/
```

## Train / test hygiene

* Fine-tune only on the **train** splits (`our485`, `Real_captured/Train`, `Synthetic/Train`; the zero-reference stage also uses `BrighteningTrain/low`).
* Report numbers on `eval15`, `Real_captured/Test`, `Synthetic/Test` and on the `unseen/` folder - never on images seen in training.
