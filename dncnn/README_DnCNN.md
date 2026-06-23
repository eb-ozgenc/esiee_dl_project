# DnCNN Reproduction — Image Denoising with Deep CNNs

A from-scratch reproduction of **DnCNN** (Zhang et al., 2017, *"Beyond a Gaussian
Denoiser: Residual Learning of Deep CNN for Image Denoising"*), covering both the
single-level (DnCNN-S) and blind (DnCNN-B) variants. Models are trained on Train400,
evaluated on Set12, and checked against the authors' pretrained weights.

All notebooks are written for Google Colab (GPU), with checkpointing to Google Drive
and resume-on-disconnect support.

## Results

| Model | Setting | Metric | This repro | Reference |
|-------|---------|--------|-----------|-----------|
| DnCNN-S | σ=25 | Set12 PSNR | **30.41 dB** | ~30.44 dB |
| DnCNN-S | σ=25 | Set12 SSIM | 0.862 | — |
| DnCNN-B | σ=15 | Set12 PSNR | 32.44 dB | 32.86 dB |
| DnCNN-B | σ=25 | Set12 PSNR | 30.18 dB | 30.44 dB |
| DnCNN-B | σ=50 | Set12 PSNR | 26.97 dB | 27.18 dB |

The single-level model lands within ~0.03 dB of the authors' pretrained reference.
The blind model sits ~0.2–0.4 dB below the matched single-level references across
the noise range — expected, since one blind network covers all levels rather than a
specialist per level, and these runs used a shortened epoch schedule (see Notes).

## Repository contents

**`Inference_baseline.ipynb`** — Sanity baseline using the authors' *pretrained*
DnCNN weights (σ=15/25/50). Downloads the official checkpoints and confirms the model
isolates the noise cleanly (reports RMSE between removed and true noise), establishing
a known-good reference before any training.

**`sigma25_model_train.ipynb`** — Trains **DnCNN-S** for a single noise level (σ=25).
Depth 17, 40×40 patches, residual learning, batch normalization, Adam. Built on the
authors' architecture and `sum_squared_error` loss imported directly from their code.

**`sigma25_model_evaluate.ipynb`** — Independent reload + evaluation of the
trained DnCNN-S. Rebuilds the BatchNorm architecture, loads the checkpoint in `eval()`
mode, and reports Set12 PSNR/SSIM. Confirms a clean reload (`missing 0 | unexpected 0`).

**`blind_model_train.ipynb`** — Trains **DnCNN-B** (blind). Depth 20, 50×50
patches, with a per-patch random noise level drawn from σ∈[0, 55] so one network
handles the whole range. Resume-aware, saves to Drive every epoch.

**`blind_model_evaluate.ipynb`** — Evaluates the blind model at σ=15/25/50 and
tabulates PSNR/SSIM against the matched single-level references with the gap per level.

## How to run

1. Open a notebook in Colab and set Runtime → GPU.
2. Run top to bottom. Training data (Train400) and test data (Set12) download
   automatically from the authors' repository.
3. For training notebooks, checkpoints save to a Google Drive folder (set `SAVE_DIR`).
   If Colab disconnects, just re-run the training cell — it resumes from the latest
   `model_NNN.pth` in that folder.
4. To evaluate, point the evaluate notebook's checkpoint path at your trained
   `model_NNN.pth` (or upload it).

Suggested order for a fresh read: `Inference_baseline` → `sigma25_model_train` →
`sigma25_model_evaluate` → `blind_model_train` → `blind_model_evaluate`.

## Method notes

- **Architecture and loss** are imported directly from the authors' `main_train.py`
  (the `DnCNN` class and `sum_squared_error`) rather than reimplemented, so the
  network matches the original exactly. The forward pass uses residual learning: it
  returns the input minus the learned residual, i.e. the predicted clean image.
- **Evaluation runs in `eval()` mode** so BatchNorm uses its learned running
  statistics, and rebuilds the BatchNorm architecture before loading weights.
- **Epoch schedule:** these runs use a shortened schedule (30 epochs)
  as a proof-of-training rather than the authors' full ~180-epoch run. Both models
  still reach close to the reference PSNR.
- **Metrics:** PSNR and SSIM on Set12 with `data_range=1.0`, fixed per-image noise
  seeds for reproducibility.

## Reference

K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, "Beyond a Gaussian Denoiser:
Residual Learning of Deep CNN for Image Denoising," *IEEE Transactions on Image
Processing*, 2017. Original code: https://github.com/cszn/DnCNN
