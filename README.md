# SAM mechanistic interpretability on medical images

Answers to the three-question CMU task (`Test-VLM-mech-interp.pdf`):

1. What do SAM's intermediate reps look like when fed a chest X-ray? Where does "medical" signal appear at all?
2. Do the last four encoder layers separate X-rays from MRIs? Which layer separates them best?
3. If ablating head k in layer L drops MRI dice but not X-ray dice, what mechanism explains it and how would you test?

All three answers are backed by running code, not just prose.

## Repo layout

```
CMU_SAM/
├── requirements.txt
├── run_all.py                 # one-shot driver
├── src/
│   ├── config.py              # paths, seeds, SAM / MSD constants
│   ├── viz.py                 # NeurIPS-style matplotlib defaults + save_fig
│   ├── model.py               # SAM loader, hooked Attention.forward, encode(), ablation
│   ├── data.py                # chest X-ray (HF NIH) + prostate MRI (MSD Task05) loaders
│   └── metrics.py             # silhouette / Fisher / linear-probe / k-NN / linear-CKA
└── scripts/
    ├── download.py            # pulls data, saves Fig. 1
    ├── q1.py                  # layer-wise analysis — Figs. 2–3
    ├── q2.py                  # separation metrics + PCA/UMAP — Figs. 4–5
    ├── q3.py                  # full head-ablation sweep + H1/H2 tests — Figs. 6–7
    └── assemble.py            # composite teaser + LaTeX tables
```

## Run on Colab

1. **Turn on GPU**: `Runtime → Change runtime type → T4 GPU → Save`.
2. In a single cell:

   ```python
   !git clone https://github.com/<your-user>/CMU_SAM.git repo
   %cd repo
   !pip install -q -r requirements.txt
   !python run_all.py                      # full: ~75 min
   # or, faster:
   # !python run_all.py --sweep-blocks 2   # ~45 min
   ```

3. Download results:

   ```python
   !cd /content && zip -r sam_interp.zip sam_interp/
   from google.colab import files
   files.download("/content/sam_interp.zip")
   ```

Run steps individually if you prefer visible progress:

```python
!python -m scripts.download
!python -m scripts.q1
!python -m scripts.q2
!python -m scripts.q3 --sweep-blocks 2
!python -m scripts.assemble
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SAM_INTERP_ROOT=$PWD/out       # otherwise /content/sam_interp is used
python run_all.py
```

You need a CUDA GPU with ≥ 10 GB for the full sweep; CPU works but is slow.

## Runtime (free Colab T4)

| stage        | time       | notes                                       |
|--------------|------------|---------------------------------------------|
| `download`   | ~2 min     | 375 MB SAM + 230 MB MSD, cached thereafter  |
| `q1`         | ~3 min     | 40 forwards × 12 blocks, entropy on-the-fly |
| `q2`         | ~1 min     | CPU-bound sklearn                           |
| `q3` default | ~50 min    | 48 (block × head) ablations × 40 images     |
| `q3` `--sweep-blocks 2` | ~25 min | trailing 2 blocks only              |
| `assemble`   | <1 min     |                                             |

## Design notes

- **ViT-B, not ViT-H.** ViT-H at 1024² with per-block hooks OOMs a free T4. The interpretability conclusions carry over; architecture is identical except depth (12→32) and width (768→1280).
- **Prostate (Task05) MRI** instead of abdominal (CHAOS) MRI. CHAOS requires a signup that blocks one-shot Colab runs. Prostate is pelvic/lower-abdominal and the smallest MSD MRI task. If you have CHAOS locally, drop PNGs into `$SAM_INTERP_ROOT/data/mri/` before running — `q1` will use them.
- **Mean-pool over patches for Q2.** SAM has no CLS token; max-pool and mean-pool give qualitatively identical answers here.
- **Three metrics for Q2**, not one. Silhouette (geometric), Fisher (statistical), linear probe (discriminative) — layer-picking is robust only when all three agree.
- **Q3 goes beyond the prompt.** The task asks for hypotheses and test designs on a hypothetical ablation result. We actually perform the sweep (Fig. 6), then run both hypothesis tests (H1 via activation norms; H2 via CKA + pair-ablation) on the measured asymmetric head (Fig. 7). Evidence-backed answer.
- **Memory discipline.** Per-pass attention tensors (~800 MB per global block) are never accumulated across images. `src.model.set_store(...)` toggles capture only for the few cells that need full attention or per-head outputs.

## Expected outputs

Under `$SAM_INTERP_ROOT` (default `/content/sam_interp/`):

```
figures/
  fig1_dataset_examples.{pdf,png}
  fig2_layerwise_q1.{pdf,png}
  fig3_attention_maps_q1.{pdf,png}
  fig4_separation_metrics_q2.{pdf,png}
  fig5_pca_umap_q2.{pdf,png}
  fig6_head_importance_q3.{pdf,png}
  fig7_hypothesis_tests_q3.{pdf,png}
  fig_teaser_report.{pdf,png}
outputs/
  q1_features.npz
  q2_metrics.csv
  q3_ablation_sweep.csv
```

## References

1. Kirillov et al., *Segment Anything*, ICCV 2023 — architecture, SA-1B pretraining.
2. Ma et al., *Segment Anything in Medical Images (MedSAM)*, Nature Communications 2024.
3. Cheng et al., *SAM-Med2D*, 2023 — adapter-based adaptation; evidence the raw encoder lacks medical features.
4. Mazurowski et al., *Segment Anything Model for medical image analysis*, Medical Image Analysis 2023.
5. Raghu et al., *Do Vision Transformers See Like CNNs?*, NeurIPS 2021 — layer-wise ViT analysis; CKA methodology.
6. Kornblith et al., *Similarity of Neural Network Representations Revisited*, ICML 2019 — linear CKA (used in Q3 H2).
7. Wang et al., *Interpretability in the Wild (IOI)*, ICLR 2023 — head-level causal circuit analysis.
8. Meng et al., *Locating and Editing Factual Associations (ROME)*, NeurIPS 2022 — activation / path patching.
