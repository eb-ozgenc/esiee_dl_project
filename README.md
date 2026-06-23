# CNN Image Restoration — DnCNN & SAR-CNN Reproductions

Reproductions of two related deep-learning restoration papers, built as a progression:
a Gaussian image denoiser (**DnCNN**) and its adaptation to multiplicative SAR speckle
(**SAR-CNN**). The second project reuses the first's architecture and training
scaffold, so together they show how one residual-CNN design transfers from additive to
multiplicative noise.

All notebooks run on Google Colab (GPU) with Google Drive checkpointing and
resume-on-disconnect support. Training data (Train400) and test data (Set12) download
automatically from the original authors' repositories.

## Headline results

| Project | Model / setting | Set12 PSNR | Reference |
|---------|-----------------|-----------|-----------|
| DnCNN | DnCNN-S, σ=25 | **30.41 dB** | ~30.44 dB |
| DnCNN | DnCNN-B (blind), σ=25 | 30.18 dB | 30.44 dB |
| SAR-CNN | Synthetic, single-look speckle | **26.05 dB** | ~25.95 dB |

Both reproductions land on or near their papers' reported numbers and beat the
classical baselines each paper compares against.

## Repository layout

```
.
├── dncnn/      Gaussian image denoising (Zhang et al., 2017)
└── sarcnn/     SAR despeckling (Chierchia et al., 2017)
```

Each folder has its own README with full details, per-image tables, and run
instructions. Summaries below.

---

## dncnn/ — Beyond a Gaussian Denoiser

Reproduces DnCNN for additive white Gaussian noise removal, covering both the
single-level (DnCNN-S) and blind (DnCNN-B) variants. The network is a 17–20 layer
residual CNN with batch normalization that predicts the noise and subtracts it.

**Notebooks**
- `Inference_baseline.ipynb` — sanity check using the authors' *pretrained* weights
  (σ=15/25/50), establishing a known-good reference before any training.
- `sigma25_model_train.ipynb` — trains DnCNN-S at σ=25 (depth 17, 40×40 patches).
- `sigma25_model_evaluate.ipynb` — independent reload + Set12 PSNR/SSIM (reaches
  30.41 dB, ~0.03 dB from the pretrained reference).
- `blind_model_train.ipynb` — trains DnCNN-B (depth 20, 50×50 patches) with a random
  per-patch noise level σ∈[0, 55] so one network covers the whole range.
- `blind_model_evaluate.ipynb` — evaluates the blind model at σ=15/25/50 against the
  matched single-level references.

**Result:** DnCNN-S matches the reference almost exactly; the blind model sits ~0.2–0.4
dB below the per-level specialists across the range, as expected for one network
covering all levels on a shortened schedule.

---

## sarcnn/ — SAR Image Despeckling

Adapts the DnCNN design to remove **multiplicative speckle** from SAR imagery using a
homomorphic (log-domain) approach. Reproduces the paper's synthetic experiment.

**Notebooks**
- `SARCNN_synthetic_train.ipynb` — trains SAR-CNN on Train400 with single-look speckle
  (depth 17, 40×40 patches, two-stage schedule: 30 epochs @ 1e-3 then 20 @ 1e-4).
- `SARCNN_Verify.ipynb` — independent reload, Set12 PSNR/SSIM, and
  clean/noisy/despeckled/residual visual panels.

**What changes from DnCNN.** Speckle is multiplicative (`noisy = clean × speckle`), so:
(1) single-look speckle is injected in amplitude format instead of additive Gaussian
noise; (2) training runs in the log domain, where the multiplicative noise becomes
additive, with a **log-cosh** loss (robust to speckle's heavy tails) replacing MSE;
(3) a two-stage learning-rate schedule per the paper.

Because the reused DnCNN forward pass returns the input minus the learned residual, the
network output is already the predicted clean log-image, so despeckling is
`exp(net(log y))` and the loss compares that directly to `log(clean)`. See the
`sarcnn/` README for the full explanation.

**Result:** 26.05 dB / 0.750 SSIM on Set12, matching the paper's ~25.95 / ~0.764 and
exceeding PPB, NL-SAR, and SAR-BM3D.

---

## Scope and deviations from originals

- **DnCNN** training uses a shortened epoch schedule (a proof-of-training, not the
  authors' full ~180 epochs), still reaches the reference PSNR.
- **SAR-CNN** reproduces only the synthetic experiment. The paper's high-resolution
  experiment needs a proprietary multitemporal COSMO-SkyMed acquisition (its clean
  reference is approximated by multilooking 25 temporal components), which is not
  public and has no substitute — so it is out of scope.
- Both projects train on the public **Train400** set and evaluate on **Set12** with
  fixed per-image noise/speckle seeds, so the numbers are internally reproducible.
  Results reproduce each paper's magnitude and ranking rather than exact decimals,
  since training data and noise realizations differ from the originals.

## References

- K. Zhang, W. Zuo, Y. Chen, D. Meng, L. Zhang, "Beyond a Gaussian Denoiser: Residual
  Learning of Deep CNN for Image Denoising," *IEEE TIP*, 2017.
- G. Chierchia, D. Cozzolino, G. Poggi, L. Verdoliva, "SAR Image Despeckling Through
  Convolutional Neural Networks," *IEEE IGARSS*, 2017. arXiv:1704.00275.
- Original DnCNN code: https://github.com/cszn/DnCNN
