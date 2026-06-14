// ============================================================
// Telecom Paris – Idemia Face Occlusion Data Challenge
// Technical Report — Join the Club, Group 17
// ============================================================
#set page(
  paper: "a4",
  margin: (top: 1.8cm, bottom: 1.8cm, left: 1.6cm, right: 1.6cm),
  columns: 2,
  footer: context [
    #align(center)[
      #text(9pt, fill: gray.darken(20%))[
        Page #counter(page).display() of #counter(page).final().first()
      ]
    ]
  ]
)
#set text(font: "New Computer Modern", size: 9.5pt, lang: "en")
#set par(justify: true, leading: 0.50em)

#show heading: it => [
  #v(0.35em)
  #block(text(font: "New Computer Modern", weight: "bold", it.body))
  #v(0.2em)
]
#set heading(numbering: "1.")

// ── Title block (full-width) ─────────────────────────────────
#place(
  top,
  scope: "parent",
  float: true,
)[
  #align(center)[
    #text(13pt, weight: "bold")[
      Face Occlusion Estimation via ConvNeXt Ensemble \
      with Classical CV Baseline
    ]
    #v(0.25em)
    #text(10pt)[
      *Group Name:* Join the Club · *Group Number:* 17 \
      *Repository:* #link("https://github.com/regkhalil/Data-Challenge")[github.com/regkhalil/Data-Challenge]
    ]
    #v(0.15em)
    #line(length: 100%, stroke: 0.5pt)
    #v(0.25em)
  ]
]

// ============================================================
= Introduction
// ============================================================

The Telecom Paris–Idemia Face Occlusion Challenge frames face occlusion
estimation as a regression task: given a 224 × 224 pixel aligned face
crop, predict the fraction of the face occluded, $hat(y) in [0, 1]$.
The official evaluation metric is a *gender-balanced, occlusion-weighted
mean squared error*:

$
  cal(L) = (E_F + E_M) / 2 + |E_F - E_M|
$

where for each gender group $g in \{F, M\}$:

$
  E_g = (sum_i w_i (hat(y)_i - y_i)^2) / (sum_i w_i), quad
  w_i = 1 / 30 + y_i
$

The weight $w_i$ makes high-occlusion samples up to *31× more influential*
than clear faces, heavily penalising false negatives on strongly occluded
faces. The term $|E_F - E_M|$ further requires equitable performance
across genders. The training set contains 100 000 images, the test set
29 980. The occlusion distribution is highly skewed (mean 0.083, 69% of
samples have GT < 0.1) and gender-imbalanced (32.4% female, 67.6% male).
Critically, *test gender labels are hidden* (replaced by `x`), making
per-gender post-hoc calibration impossible.

// ============================================================
= Classical CV Baseline
// ============================================================

Prior to deep learning, we built a *fully training-free classical CV
heuristic* to establish an analytical upper bound. The pipeline: (1)
CLAHE normalisation on the Y channel of YCrCb; (2) primary skin mask
from fixed chrominance ranges ($"Cr" in [133,173]$, $"Cb" in [77,127]$);
(3) adaptive HSV secondary mask; (4) Laplacian texture filter with
anatomy-aware exclusion zones (eyes, mouth); (5) occlusion fraction
estimated from the non-skin pixel ratio inside a central circular ROI.
This zero-training approach hit a hard ceiling of ~*0.023* on the
challenge metric. Compression artefacts at 224 × 224, variable lighting,
and the semantic richness of occlusion types (scarves, hands, hair) were
insurmountable for fixed-colour detectors. This validated the shift to
deep learning.

// ============================================================
= Deep Learning Solution
// ============================================================

== Architecture

Each model is a *transfer-learning regressor*: a pretrained backbone from
the `timm` library @rw2019timm with `num_classes=0`, followed by a
custom regression head:

$
  "Linear"(d_"feat" -> 512) -> "ReLU" -> "Dropout"(0.3) \
  -> "Linear"(512 -> 128) -> "ReLU" -> "Dropout"(0.3) \
  -> "Linear"(128 -> 1) -> "Sigmoid"
$

The terminal Sigmoid constrains predictions to $[0,1]$ without output
clipping. We explored EfficientNet-B4 @tan2019efficientnet,
ConvNeXt-Base/Large @liu2022convnext, ConvNeXtV2-Base @woo2023convnextv2,
ViT-Base, EfficientNetV2-M and Swin-Base.

== Loss Function

The training loss *replicates the evaluation metric exactly*:
$cal(L)_"train" = (E_F + E_M)/2 + |E_F - E_M|$, with per-sample weights
$w_i = 1/30 + y_i$ computed inside each forward pass. In the rare case of
a mono-gender mini-batch, the loss falls back to global weighted MSE. This
eliminates any optimisation gap that would arise from vanilla MSE or Huber.

== Training

*Data split:* stratified 80/20 on gender × occlusion quintile (seed 42).

*Weighted sampler:* draw probability proportional to
$s_i = (1/30 + y_i) times N/(2 N_g) times "male\_boost"$,
correcting both occlusion imbalance and gender skew. `male_boost = 2.5`
(moderate) reduces $|E_F - E_M|$ without overfitting.

*Optimiser:* AdamW @loshchilov2019decoupled, weight decay $10^{-4}$,
cosine annealing over 30 epochs. Learning rate: $10^{-4}$ for ConvNeXt,
$3 times 10^{-5}$ for ConvNeXtV2 (FCMAE backbone diverges at $10^{-4}$).

*Augmentation:* random horizontal flip, ColorJitter (0.3/0.3/0.2),
rotation ±20°, Gaussian blur ($p=0.4$). RandomErasing is excluded (it
would simulate occlusion and leak label information).

*Inference:* Test-Time Augmentation (TTA) with 15 passes (1 clean + 14
augmented), averaged.

// ============================================================
= Final Ensemble (Score 0.00112)
// ============================================================

After training 12+ models, the best leaderboard result combines
*seven ConvNeXt models* in two tiers:

- *ENS4* — four strong models: ConvNeXtV2-Base (seed 123),
  ConvNeXt-Large (seeds 123 and 999), ConvNeXt-Base (seed 456).
- *s6* — three diverse models: ConvNeXt-Base (seeds 42, 123),
  EfficientNet-B4 (seed 123).
- *Final:* $0.55 dot "ENS4" + 0.45 dot "s6"$.

#figure(
  table(
    columns: (2.8fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: (left, center, center, center),
    stroke: 0.5pt,
    [*Model*], [*Backbone*], [*Eval.*], [*Score*],
    [ConvNeXt-Large (seed 999)], [ConvNeXt-L], [val], [0.0011],
    [ConvNeXt-Large (seed 123)], [ConvNeXt-L], [val], [0.0011],
    [ConvNeXtV2-Base (seed 123)], [CnXV2-B], [val], [0.0014],
    [ConvNeXt-Base (seeds 42 & 123)], [ConvNeXt-B], [val], [0.0016],
    [ConvNeXt-Base (seed 456)], [ConvNeXt-B], [val], [0.0018],
    [EfficientNet-B4 (seed 123)], [EffNet-B4], [val], [0.0017],
    table.hline(),
    [Mean of 3 strong ConvNeXt models], [—], [leaderboard], [0.00115],
    [Mean of 2 ConvNeXt-Base + EfficientNet-B4], [—], [leaderboard], [0.00116],
    [Mean of 4 strong ConvNeXt models], [—], [leaderboard], [0.00115],
    table.hline(),
    [*Final blend: 0.55 × (4-model mean) + 0.45 × (3-model mean)*], [*—*], [*leaderboard*], [*0.00112*],
  ),
  caption: [Individual model validation scores and ensemble leaderboard scores.],
  placement: bottom,
  scope: "parent",
)

// ============================================================
= Key Findings
// ============================================================

*Validation scores can mislead.* A synthetic-degradation recipe
(artificial occlusions, aggressive tail weighting, GT-scaled male boost)
achieved the best val score (0.00129) but the *worst* test score (0.00218)
— a 70% regression versus the clean ensemble. Synthetic tricks overfit the
validation split. We learned to trust the leaderboard over local
validation.

*Architecture family matters more than weight tuning.* ViT-Base,
Swin-Base and EfficientNetV2-M were all clearly weaker (val > 0.0018);
adding ViT *degraded* the ensemble to 0.00121. ConvNeXt and ConvNeXtV2
form the dominant family. EfficientNet-B4 at low weight (~15% via s6)
contributes useful architectural decorrelation. Fine-tuning the 0.55/0.45
blend ratio produced no further gain — the plateau at 0.00112 can only be
broken by new, genuinely diverse models.

*Test gender is hidden.* Since `test_students.csv` contains no gender
column, per-gender post-hoc calibration is impossible. Global $(alpha,
beta)$ scaling on validation gave only 0.9% gain, confirming that the
clean models are already well-calibrated. Gender balancing is confined to
the training phase (`male_boost` + weighted sampler).

// ============================================================
= Limitations
// ============================================================

*Computational cost.* Seven deep networks, each trained 10–20 h on an
RTX 3090, plus TTA ×15 inference (~1–2 h per model). Not suited for
real-time deployment.

*Ensemble complexity.* ConvNeXt-Large has ~200 M parameters; the full
ensemble is heavy in memory and storage. Several hyperparameters require
careful tuning (`male_boost`, per-backbone learning rate, blend ratio).

*Validation reliability.* The interim leaderboard reflects a subset of
the test distribution; the final score on the held-out set may differ.

// ============================================================
= Conclusion
// ============================================================

We presented a two-stage solution: a training-free classical CV baseline
(ceiling ~0.023) that confirmed the necessity of deep learning, followed
by a seven-model ConvNeXt ensemble that achieves *0.00112* on the interim
leaderboard. The key design principles — loss identical to the metric,
occlusion- and gender-weighted sampling, TTA, and diverse multi-seed
multi-architecture ensembling — each directly target the structure of the
evaluation criterion. Critical lessons learned: validation overfitting is
severe with synthetic augmentation, backbone family dominates over weight
optimisation, and architectural diversity (not just seed diversity) is the
primary driver of ensemble gains.

#set bibliography(style: "ieee")
#bibliography("refs.bib")
