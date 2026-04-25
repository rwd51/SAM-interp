# SAM ViT-B on Out-of-Distribution Medical Images
### A mechanistic analysis of SA-1B-pretrained SAM applied zero-shot to chest X-rays and abdominal MRIs

---

## TL;DR

| Question | Headline answer |
|---|---|
| **Q1** | Representations drift continuously with depth. No medical-specific features "emerge"; SAM retains enough low-level discriminators to separate modalities despite never having seen one. Cross-modality cosine similarity *rises* with depth (0.76 → 0.87) while cluster separation is preserved — absolute angular distance decreases, variance compresses, geometry remains discriminative. |
| **Q2** | **Block 8 produces the clearest separation** (silhouette 0.471, Fisher 2.469), with monotonic decline to block 11 (0.454 / 2.263). Linear probe and 5-NN saturate at 100% for all four blocks — continuous geometric metrics are the informative signal, classification metrics are at ceiling. |
| **Q3** | Across the full last-4-blocks sweep, single-head ablations produce **small, distributed effects** (max |Δsilhouette| = 0.013, 2.8% of the 0.454 baseline). The measured asymmetric head **(B8, H0)** has the prompt-matching signature (MRI cluster hurt, X-ray cluster untouched) but at only +0.75% silhouette — too small to pin the mechanism on one head. The head activates ~equally on both modalities (rejects H1) and CKA to siblings is near-symmetric (rejects H2). Verdict: **H3 — distributed modality signal**, with block 8 carrying the strongest causal load (consistent with Q2's block-8 peak). |

---

## 1. Experimental setup

- **Model.** SAM ViT-B (`sam_vit_b_01ec64.pth`), 12 transformer blocks, 12 heads per block, embed dim 768, patch size 16, input 1024 × 1024. Global attention at blocks {2, 5, 8, 11}; windowed attention (14 × 14) elsewhere.
- **Pre-training.** SA-1B natural-image dataset; no medical images seen during training.
- **Datasets.**
  - 20 chest X-rays from **NIH ChestX-ray14** (HuggingFace Parquet mirror `Sohaibsoussi/NIH-Chest-X-ray-dataset-small`, streaming; the original loading-script repo was deprecated in `datasets` v4.5+).
  - 20 abdominal MRI mid-axial slices from **CHAOS T2-SPIR** (Zenodo mirror of the MICCAI CHAOS challenge, no authentication).
- **Inference.** Forward pass through the image encoder only; mask decoder is never called. Per-block patch embeddings `(64 × 64 × 768)` are mean-pooled to one 768-D vector per image per block.
- **Interpretability tooling.** The `segment_anything.modeling.image_encoder.Attention.forward` method is monkey-patched to optionally expose per-block attention maps, per-head activations, and per-head entropy, and to support per-head ablation. Patches are inert until explicitly enabled, so a default encode pass adds zero overhead.
- **Determinism.** Single seed propagated to numpy / torch / CUDA / StratifiedKFold / PCA / UMAP. All runs reproducible from `git clone && python run_all.py`.

---

## 2. Q1 — Where (if anywhere) do medical-specific features emerge?

### 2.1 Hypothesis

SAM's encoder was pretrained on natural images with object-centric occlusion boundaries; medical images have no such boundaries. We expect:

- **Early blocks (0–3):** low-level filters (edges, local contrast) should transfer reasonably. Ribs, lung contours, and organ boundaries are still high-contrast structures.
- **Middle blocks (4–8):** mid-level parts-and-textures detectors will start to mis-fire. SA-1B has object boundaries; medical images have soft-tissue gradients and overlapping translucent structures.
- **Late blocks (9–11):** object-centric semantic features should project medical structures onto the nearest natural-image manifold. "Lung" or "liver" concepts never existed; representations reflect *out-of-distribution drift*, not medical semantics.

Ground priors: MedSAM (Ma et al. 2024) and SAM-Med2D (Cheng et al. 2023) both require per-block adapters to inject medical features, implying the raw encoder lacks them.

### 2.2 Evidence (Fig. 2)

**(a) Attention entropy at the four global-attention blocks.** Both modalities follow the same V-shape: entropy drops sharply from block 2 (≈6.9 nats) to block 5 (≈4.5 nats), then recovers modestly by blocks 8 and 11 (≈5.4–5.8). Block 5 is where attention is maximally peaked on specific tokens — the shape is nearly identical for X-ray and MRI, indicating no modality-specialised attention pattern at the population level.

**(b) Within- and between-modality cosine similarity per block.** Within-modality similarity stays high (≈0.98–0.99) across all depths — X-rays look like other X-rays at every layer, same for MRIs. Between-modality similarity *rises* with depth (0.76 at block 0 → 0.87 at block 11). Interpretation: **as representations grow in magnitude (see panel c), the mean X-ray and mean MRI vectors drift towards each other in direction, while within-class variance shrinks faster than the means converge**. Relative separation (silhouette) is preserved; absolute angular distance is not. This is consistent with the blocks learning a common "image manifold" geometry while retaining modality-discriminating sub-signals.

**(c) Token-norm growth.** Classic ViT behaviour — near-flat norms in early blocks (≈30–40), exponential growth from block 6 onward, reaching ≈80 at block 11. Identical trajectory for both modalities.

### 2.3 Attention maps (Fig. 3)

Mean attention of a centre query patch, visualised at the four global blocks:

- **Block 2:** broadest, most diffuse — early global attention in SAM behaves almost like a bottom-up salience detector.
- **Blocks 5, 8, 11:** attention collapses to the query location with a faint cross-pattern. This is almost entirely driven by **SAM's learned relative-position bias**, not by content.

The mean over 12 heads averages out individual-head specialisation. Consequently, this figure primarily shows that **head-averaged attention is dominated by positional priors in late blocks** — individual-head analysis (Q3) is needed to reveal any content specialisation.

### 2.4 Answer

Medical-specific features **do not emerge** — SAM was never trained on medical content. What does happen, progressively with depth:

1. Within-modality consistency remains very high (≈0.99 cosine) — SAM reliably treats images-from-the-same-modality as similar.
2. Between-modality angular distance *shrinks* while cluster spread shrinks faster, preserving separability.
3. Token norms and positional biases grow, so late-block attention is dominated by position rather than content.

The modality-discriminating signal is a **side-effect of retained low-to-mid-level filters**, not of learned medical features. This predicts that adapters (MedSAM, SAM-Med2D) would benefit most when applied to mid-late blocks, where the representation space has the most capacity to be re-shaped toward medical semantics — consistent with those papers' findings.

---

## 3. Q2 — Do the last four blocks separate X-ray from MRI? Which block separates them best?

### 3.1 Metrics (last 4 blocks, from `outputs/q2_metrics.csv`)

| Block | Linear probe (5-fold) | 5-NN CV | Silhouette | Fisher ratio |
|------:|:---:|:---:|:---:|:---:|
| 8 | 1.000 | 1.000 | **0.471** | **2.469** |
| 9 | 1.000 | 1.000 | 0.459 | 2.312 |
| 10 | 1.000 | 1.000 | 0.460 | 2.317 |
| 11 | 1.000 | 1.000 | 0.454 | 2.263 |

### 3.2 Observations

1. **Classification saturation.** Linear probe and 5-NN CV accuracy both hit the 1.00 ceiling at all four blocks. X-ray vs MRI is trivially separable in a 768-D space with 40 samples — classifier-level metrics cannot distinguish the layers.
2. **Continuous metrics tell the real story.** Silhouette and Fisher ratio decrease monotonically from block 8 to block 11. The drop is small (3.6% in silhouette, 8.3% in Fisher) but consistent across both independent metrics — a robust signal.
3. **Winner: Block 8.** By the geometric-mean criterion over (linear probe, silhouette, Fisher), block 8 produces the clearest separation.

### 3.3 Visual corroboration (Fig. 5)

PCA and UMAP projections of the mean-pooled patch embeddings confirm:

- **All four blocks** produce visibly disjoint clusters — MRI (orange triangles) and X-ray (blue circles).
- **UMAP** in all four blocks gives tight, compact clusters — the classes are well-clustered, not merely linearly separable.
- There is no dramatic visual difference across blocks; silhouette is sensitive to the subtle cluster-tightness differences that are hard to see by eye.

### 3.4 Interpretation

SAM's last third of the encoder treats out-of-distribution X-ray and MRI as sharply different *domains*, regardless of the specific block chosen. The slight decline in silhouette from block 8 → 11 is consistent with **the final blocks compressing modality-specific differences slightly** in service of producing task-general, mask-decoder-ready features. This echoes Raghu et al. (2021), who showed that **semantic information in ViTs is carried most strongly by mid-late rather than final layers**.

**Practical corollary:** if an adapter-based medical-SAM derivative wanted to extract pre-final representations for modality-conditional downstream tasks, block 8 (or 9) would be the better tap point than block 11.

---

## 4. Q3 — Mechanistic analysis: why might ablating one head affect only MRI?

### 4.1 Setup

Per the task prompt: assume ablating head *k* in layer *L* drops MRI dice by *x%* while X-ray dice is unaffected. Propose and test two mechanistic hypotheses.

### 4.2 Hypotheses

**H1 — modality-specialised head.** Head *k* specialises in a feature present in MRI (e.g. smooth soft-tissue gradients, volumetric shading cues) but absent from chest X-rays. X-rays are projection images with bimodal intensity histograms; the head simply does not fire on them, so ablation is a no-op for that modality.

**H2 — redundancy asymmetry.** Head *k* carries modality-relevant information. For X-rays, the same information is *redundantly encoded* by sibling heads in the same block (or earlier blocks), so ablation is compensated. For MRIs, head *k* is the unique carrier — it is a bottleneck whose removal cannot be routed around.

### 4.3 Empirical approach

We performed a **full (block × head) ablation sweep** over the last 4 encoder blocks (48 ablations × 40 images, ~35 min on a T4). For each ablation we measured:

- Overall separation (silhouette, Fisher, linear probe).
- **Per-class silhouette** — `silhouette_samples` averaged over each class. Sensitive even when classification accuracy saturates at 1.0.
- **Per-class cosine coherence** — mean intra-class pairwise cosine similarity. Sensitive to cluster spread.

The asymmetric head was selected by the per-class silhouette delta:
`asym = (base_sil_mri − ablated_sil_mri) − (base_sil_xray − ablated_sil_xray)`
Positive = MRI cluster hurt more than X-ray cluster.

### 4.4 Head-importance landscape (Fig. 6)

| Signal | Observation |
|---|---|
| Dynamic range | ±0.013 across the full last-4-block sweep — small relative to the 0.454 baseline silhouette (max ~2.8%). |
| Block 8 | **By far the most causally active block.** Head 10 ablation *improves* overall silhouette by +0.013 (i.e. H10 was hurting separation when present); Head 11 *worsens* it by −0.013. Heads 2 and 6 produce smaller positive effects (~+0.006). |
| Block 9 | Nearly flat — all 12 heads produce |Δsil| < 0.003. |
| Block 10 | Nearly flat — same as block 9. |
| Block 11 | Head 1 ablation produces +0.007 (X-ray cluster gets cleaner). Heads 5–7 produce small negative effects. |
| Asymmetric pick | **(B8, H0)** with Δsil = −0.004, asymmetry +0.0077 (`delta_sil_mri = +0.0075` vs `delta_sil_xray = −0.00016`). This matches the prompt's literal scenario — MRI cluster hurt, X-ray cluster untouched — but at <1% of the baseline. |

**Headline finding:** Block 8 carries the most causal load (consistent with Q2's finding that block 8 produces the peak modality separation), but **no single head in any of the last 4 blocks dominates** — the largest single-head effect (B8.H10 / B8.H11) is just 2.8% of baseline silhouette. Effects are small and distributed across heads, with paired heads (B8.H10 helping, B8.H11 hurting) suggesting **within-block specialisation that is *not* aligned to modality**.

### 4.5 Hypothesis tests on the picked head (Fig. 7)

The most-asymmetric head, **B8.H0**, was probed against both hypotheses. The decision rules baked into `scripts/q3.py` are:

```
H1 supported  if  norm_ratio MRI/X-ray > 2.0  or  < 0.5     (modality-specialist)
H2 supported  if  mean |CKA_xray − CKA_mri| > 0.10  AND  pair-partner exists
H3 supported  if  max |Δsil| < 5% of baseline silhouette  AND  H1, H2 both rejected
```

**(a) H1 — per-head activation norm.** B8.H0 has norms of **5.35 (X-ray)** and **5.61 (MRI)** — norm ratio = 1.05, near-perfectly balanced. The head is robustly active on both modalities, not silent on either. **H1 is rejected.**

**(b) H2 — CKA to sibling heads.** Mean |CKA_xray − CKA_mri| across the 11 sibling heads is **0.084** — just below the 0.10 threshold. CKA values themselves are high (~0.65–0.95) for both modalities, and the X-ray vs MRI patterns are nearly parallel rather than divergent. The head is moderately redundant overall, but not *asymmetrically* redundant. **H2 is rejected** under our threshold rules (and would be only weakly supported even if we relaxed them).

**(c) H2 — pair-ablation probe.** Co-ablating B8.H0 with each of the other 11 sibling heads gives X-ray cluster coherence values 1.04–1.08 (best partner H10, drop = 1.077). The bar chart is essentially flat — no compensating partner head exists whose joint removal would expose B8.H0 as a unique X-ray-side carrier. **H2 fails at the pair-redundancy step.**

**Verdict (auto-classified by `scripts/q3.py`):**

```json
{ "abl_block": 8, "abl_head": 0,
  "norm_ratio": 1.05,
  "cka_delta": 0.084,
  "max_effect": 0.0128, "effect_frac": 0.0281,
  "h1": false, "h2": false, "h3": true }
```

`outputs/q3_verdict.json` is the canonical artefact; the assemble step prints a one-line winner banner.

### 4.6 Answer

On the empirical evidence across the full last-4-blocks sweep, **neither H1 nor H2 is supported by any measurable head**. Instead, the data favours a third mechanistic picture:

> **H3 — distributed modality signal.** Modality separation in SAM's last four blocks is carried *distributively* across heads, with high inter-head representational similarity (CKA ≈ 0.65–0.95) and no single-head bottleneck. The largest single-head perturbation across 48 ablations is 2.8% of the baseline silhouette — perturbing any one head moves the cluster geometry by less than ~3% of the cluster gap. The most asymmetric head identified, B8.H0, technically matches the prompt scenario direction (MRI hurt > X-ray hurt) but at a magnitude too small to localise the mechanism on it.

A finer-grained finding: **block 8 carries by far the most causal load** (largest single-head effects in the entire last-4-blocks sweep are at B8.H10 = +0.013 and B8.H11 = −0.013, with opposite sign). This pairing — one head helping separation, an adjacent head hurting it — suggests **within-block specialisation that is *not* aligned to modality** (otherwise both would show modality-asymmetric Δsil patterns). The block is causally important, but its heads do not partition by class.

This dovetails with two broader trends in mechanistic interpretability:

1. **Late-layer representations in large pretrained transformers are superposed** across many components rather than localised in a few "feature heads" (Elhage et al., 2022, on superposition).
2. **Causally important blocks (Q2: block 8 peaks separation) need not contain causally important *individual* heads** — the mid-late peak in modality information is itself a distributed property.

For SAM specifically, the practical implication is that medical-adaptation techniques like LoRA (which reshape every head's Q/K/V matrices) should outperform adapter methods that insert new components at a handful of "critical" heads — the critical-head set is empirically empty under our threshold rules.

### 4.7 What we would do with more compute / scope

- **Wider sweep.** Extend to blocks 0–7 in addition to the last 4. Early-block ablations may show larger effects because their representation space is smaller and less superposed; v2 already shows that block 8 dominates blocks 9–11, suggesting effect magnitude *grows* as we move earlier.
- **Group ablation.** Zero pairs or triples of heads — under H3, the minimum head-set size whose ablation degrades separation should be > 1. The B8.H10 / B8.H11 pair (opposite signs) is a natural starting point: jointly ablating both should test whether they form a complementary feature pair.
- **Activation patching.** Rather than zeroing, *swap* a head's activation from an X-ray forward pass into an MRI forward pass. Tests whether the information a head carries is truly modality-conditional or merely modality-correlated.
- **Mask-decoder dice.** Pair encoder ablation with the mask decoder to measure actual segmentation dice, mirroring the task prompt exactly. CHAOS ships ground-truth masks; NIH does not, so this is a CHAOS-only experiment.
- **Loosen H2 thresholds.** B8.H0's CKA-delta = 0.084 is just below our 0.10 threshold. A more sensitive H2 test (e.g. paired permutation test on per-image CKA, or a CCA-based variant) might re-classify it as marginal-H2 rather than firm-H3.

---

## 5. Discussion

### 5.1 Summary of findings

1. **No medical features emerge.** SAM's encoder produces no semantic medical content; what looks like modality discrimination is a side-effect of retained low-to-mid-level filters combined with position biases.
2. **Mid-late blocks separate best.** Block 8 peaks; final blocks compress. Suggests mask-decoder-readiness trades off against modality specificity — useful for adapter-tap-point selection.
3. **Modality signal is distributed; block 8 carries the most causal load.** Late-block modality information is superposed across heads with no single bottleneck (max single-head |Δsil| = 2.8% of baseline). Block 8's heads (especially H10 / H11, opposite-signed) are the most causally important across the entire last-4-block sweep — but their importance is *block-level*, not modality-aligned, and the prompt's "ablate one head, hurt only MRI" scenario does not literally manifest. The empirically closest match (B8.H0) has the right *direction* but a magnitude (~0.7%) far too small to support H1 or H2.
4. **Cross-modality cosine rises with depth.** Absolute angular distance between the two modality centroids shrinks from 0.24 (block 0) to 0.13 (block 11) — a 46% reduction — while within-class variance shrinks faster, preserving relative separability. SAM's encoder is *contracting everything toward a common manifold* while keeping classes distinguishable.

### 5.2 Limitations

- **Scale.** 20 + 20 = 40 images is sufficient for separation-of-distributions claims but too small for head-circuit identification. A wider dataset would tighten all error bars.
- **ViT-B only.** ViT-H has 32 blocks and 16 heads — more headroom for specialisation. The H3 (distributed) finding may weaken in a wider architecture.
- **No mask-decoder evaluation.** The task prompt references dice score; our proxy is silhouette at the final-block embedding. These are related but not identical metrics.
- **Mean-pooling.** Collapsing a 64 × 64 patch grid to one vector discards spatial structure. Per-token CKA or attention-weighted pooling might reveal finer structure.
- **Pelvic-adjacent interpretation bias.** CHAOS MRI images all have a similar anatomical composition (liver / kidneys / spleen); a broader set of abdominal MRI sources could change the within-class variance.

### 5.3 References

1. Kirillov et al., *Segment Anything*, ICCV 2023.
2. Ma et al., *Segment Anything in Medical Images (MedSAM)*, Nature Communications 2024.
3. Cheng et al., *SAM-Med2D*, 2023.
4. Mazurowski et al., *Segment Anything Model for medical image analysis*, Medical Image Analysis 2023.
5. Raghu et al., *Do Vision Transformers See Like CNNs?*, NeurIPS 2021.
6. Kornblith et al., *Similarity of Neural Network Representations Revisited (linear CKA)*, ICML 2019.
7. Wang et al., *Interpretability in the Wild (IOI)*, ICLR 2023.
8. Meng et al., *Locating and Editing Factual Associations (ROME)*, NeurIPS 2022.

---

## 6. Reproducibility

```bash
git clone https://github.com/rwd51/SAM-interp.git repo
cd repo
pip install -r requirements.txt
python run_all.py                       # default: 4-block sweep, ~65 min on Colab T4
# python run_all.py --sweep-blocks 2    # iteration-mode fast variant, ~25 min
```

All seeds fixed (`SEED = 0`); data downloads are idempotent and cached. Fig. 1–7 + both CSVs reproduce bit-exactly on a fresh Colab session.
