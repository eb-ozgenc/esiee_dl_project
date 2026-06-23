# SARCNN Reproduction: SAR Image Despeckling with Deep CNNs

A reproduction of **SAR-CNN** (Chierchia et al., 2017, *"SAR Image Despeckling
Through Convolutional Neural Networks"*), focused on the synthetic experiment
(Section 3.1). SAR-CNN adapts the DnCNN architecture to remove **multiplicative
speckle noise** from SAR imagery using a homomorphic (log-domain) approach.

This builds directly on a prior [DnCNN reproduction](#relationship-to-dncnn) — the
network is the same 17-layer residual CNN, with SAR-specific changes to the noise
model and loss. Notebooks run on Google Colab (GPU) with Drive checkpointing and
resume-on-disconnect.

## Result

| Setting | Metric | This repro | Paper (Tables 2–3) |
|---------|--------|-----------|--------------------|
| Synthetic, single-look | Set12 avg PSNR | **26.05 dB** | ~25.95 dB |
| Synthetic, single-look | Set12 avg SSIM | 0.750 | ~0.764 |

The independently reloaded model matches the paper's reported synthetic average and
exceeds the classical baselines it compares against (PPB ~23.4, NL-SAR ~24.0,
SAR-BM3D ~25.0 dB). Per-image PSNR ranges from 23.4 dB on the most textured image to
28.7 dB on the smoothest, with the same relative ordering as the paper.

## Repository contents

**`SARCNN_synthetic_train.ipynb`** — Trains SAR-CNN on Train400 with single-look
speckle. Depth 17, 40×40 patches, batch size 128, Adam. Two-stage schedule (30 epochs
@ 1e-3, then 20 @ 1e-4). Saves a checkpoint to Google Drive every epoch and resumes
from the latest one after a Colab disconnect; the full training history is bundled
into the checkpoint so the loss/PSNR curves survive a reconnect.

**`SARCNN_Verify.ipynb`** — Independent reload + evaluation. Rebuilds the architecture,
loads the trained checkpoint in `eval()` mode, injects single-look speckle on Set12,
and reports PSNR/SSIM plus clean/noisy/despeckled/residual visual panels. Confirms a
clean reload (`missing 0 | unexpected 0`).

## How it differs from DnCNN

SAR speckle is **multiplicative** (`noisy = clean × speckle`), not additive, so three
things change from a standard Gaussian denoiser:

1. **Noise model** — single-look speckle injected in amplitude format, instead of
   additive white Gaussian noise.
2. **Homomorphic loss** — training happens in the log domain, where
   `log(noisy) = log(clean) + log(speckle)` turns the multiplicative problem into an
   additive one. The loss is a **log-cosh** penalty (robust to speckle's heavy-tailed
   outliers) rather than MSE.
3. **Two-stage schedule** — 30 epochs at 1e-3 then 20 at 1e-4, per the paper's
   synthetic-data setup.

**A note on the loss vs. the paper's Eq. 1.** The authors' DnCNN forward pass returns
the input minus the learned residual, so the network output is already the predicted
*clean log-image*. Despeckling is therefore `exp(net(log y))` (no second subtraction),
and the loss compares `net(log y)` directly to `log(clean)`. This is the "clean-image"
form of the paper's "residual" form — the same log-cosh penalty in the log domain. The
paper's re-centering constant **c** (the mean of log-speckle, −γ/2 ≈ −0.289 for
single-look amplitude speckle) does not appear explicitly in this loss; with residual
learning the network learns the equivalent correction into its own parameters. The
value is kept in the code only to document the convention.

## How to run

1. Open a notebook in Colab, set Runtime → GPU.
2. Run top to bottom. Train400 and Set12 download automatically.
3. Training checkpoints save to a Drive folder (set `SAVE_DIR`). If Colab disconnects,
   re-run the training cell — it resumes from the latest checkpoint and applies the
   correct learning rate for whichever stage that epoch belongs to.
4. To evaluate, point `SARCNN_Verify.ipynb` at your trained checkpoint. Loading
   `last.pth` also restores the training curves; the per-epoch `model_NNN.pth` files
   are weights-only.

## Scope and deviations

- Only the **synthetic experiment (Section 3.1)** is reproduced. The paper's
  high-resolution experiment (Section 3.2) relies on a proprietary 25-component
  multitemporal COSMO-SkyMed acquisition — the clean reference is approximated by
  multilooking that temporal stack — which is not publicly available and has no
  substitute, so it is out of scope.
- Training uses the public **Train400** set (consistent with the paper's description
  of 400 training images) with an in-house single-look speckle simulator. Results
  therefore reproduce the paper's **magnitude and ranking**, not its exact decimals.
- PSNR/SSIM on speckled imagery depends on the speckle realization; per-image seeds
  are fixed so the reported numbers are internally reproducible.

## Relationship to DnCNN

SAR-CNN reuses the DnCNN architecture and training scaffold. The network, residual
learning, optimizer, and patch pipeline come from the DnCNN reproduction; only the
noise model, loss, and schedule are SAR-specific. See the DnCNN repository for the
baseline image-denoising work this is built on.

## Reference

G. Chierchia, D. Cozzolino, G. Poggi, and L. Verdoliva, "SAR Image Despeckling Through
Convolutional Neural Networks," *IEEE IGARSS*, 2017. arXiv:1704.00275.

Built on: K. Zhang et al., "Beyond a Gaussian Denoiser," *IEEE TIP*, 2017.
Original DnCNN code: https://github.com/cszn/DnCNN
