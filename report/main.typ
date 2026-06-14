// ============================================================
// Telecom Paris – Idemia Face Occlusion Data Challenge
// Technical Report
// ============================================================
#set page(
  paper: "a4",
  margin: (top: 1.8cm, bottom: 1.8cm, left: 1.6cm, right: 1.6cm),
  columns: 2,
)
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.55em)
#set heading(numbering: "1.")

// ── Title block (full-width) ─────────────────────────────────
#place(
  top,
  scope: "parent",
  float: true,
)[
  #align(center)[
    #text(13pt, weight: "bold")[
      Face Occlusion Estimation: A Deep Learning Approach \
      with Classical CV Baseline
    ]
    #v(0.3em)
    #text(10pt)[
      *Group Name:* Join the Club · *Group Number:* 17 \
      *Repository:* #link("https://github.com/regkhalil/Data-Challenge")[github.com/regkhalil/Data-Challenge]
    ]
    #v(0.2em)
    #line(length: 100%, stroke: 0.5pt)
    #v(0.3em)
  ]
]

// ============================================================
= Introduction
// ============================================================

The Telecom Paris–Idemia Face Occlusion Challenge frames face occlusion
estimation as a regression task: given a 224 × 224 pixel aligned face crop,
predict the fraction of the face that is occluded, $hat(y) in [0, 1]$.
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

The additive weight $w_i$ makes high-occlusion samples up to 30× more
influential than clear faces, explicitly penalising *false negatives* on
heavily occluded faces. The imbalance term $|E_F - E_M|$ further requires
that the model performs equitably across genders. A solution that excels
on one group at the expense of the other is harshly penalised.

// ============================================================
= Proposed Deep Learning Solution
// ============================================================

== Architecture

Our model is a lightweight *transfer-learning regressor* built on top of a
pre-trained *EfficientNet-B4* backbone @tan2019efficientnet extracted
via the `timm` library. EfficientNet-B4 was chosen for its favourable
accuracy-to-parameter ratio at 224 × 224 input resolution and the depth of
its compound-scaled feature hierarchy, which captures both low-level texture
(essential for detecting fabric/hair occlusions) and high-level semantic
structure (essential for partial face parsing).

The backbone is used as a fixed-geometry feature extractor (all layers
unfrozen for fine-tuning). Its output feature vector (1792-d) is passed
through a *custom regression head*:

$
  "Linear"(1792 -> 512) -> "ReLU" -> "Dropout"(0.3) \
  -> "Linear"(512 -> 128) -> "ReLU" -> "Dropout"(0.3) \
  -> "Linear"(128 -> 1) -> "Sigmoid"
$

The terminal Sigmoid constrains predictions to $[0, 1]$ and naturally
models the bounded regression target without additional output clipping.

// ── Horizontal Full-Page Figure ──────────────────────────────
#figure(
  image("architecture.png", width: 100%),
  caption: [Condensed architectural pipeline: EfficientNet-B4 feature extraction followed by a 3-layer regression head.],
  placement: bottom,
  scope: "parent",
)

== Data Augmentation

All training images are processed with ImageNet-standard normalisation
(mean $mu = (0.485, 0.456, 0.406)$, std $sigma = (0.229, 0.224, 0.225)$).
During training, the following stochastic augmentations are applied:

- *Random horizontal flip* (p = 0.5): face symmetry invariance.
- *Color jitter*: brightness ±0.2, contrast ±0.2, saturation ±0.1 —
  addresses variable lighting across identities and environments.
- *Random rotation* ±15°: robustness to slight head tilt.
- *Gaussian blur* (kernel 3 × 3, p = 0.3): simulates sensor blur and
  compression artefacts common in low-quality crops.

Validation inference uses the normalisation transform only (no augmentation).
Test-time augmentation (TTA) is supported at inference: $k$ passes with
training augmentations are averaged to reduce prediction variance.

== Training Methodology

Training uses ~80 000 labelled images from the 100K crop dataset, with a
stratified 80/20 train–validation split. Stratification keys are formed from
the cross-product of gender and occlusion quintile to guarantee that both
group distributions and target density are preserved in each split.

*Optimiser:* AdamW @loshchilov2019decoupled with learning rate
$eta = 10^(-4)$ and weight decay $lambda = 10^(-4)$.

*Scheduler:* Cosine annealing over 40 epochs with no warmup, decaying
to $eta_"min" = 0$ to allow fine-grained convergence near the end of training.

*Weighted sampler:* To counteract the heavy class imbalance toward
low-occlusion faces, a `WeightedRandomSampler` assigns each sample a draw
probability proportional to $1 / 30 + y_i$, mirroring the evaluation
metric. An additional *male boost* factor of 3.0 is applied to male samples
to force the model to reduce $|E_F - E_M|$. This yields a training
distribution that is simultaneously occlusion-heavy and gender-balanced,
directly targeting the structure of the loss.

== Custom Loss Function

The model is trained by directly *optimising the challenge metric* rather
than a surrogate. The training loss replicates $cal(L)$ exactly:

$
  cal(L)_"train" = (E_F + E_M) / 2 + |E_F - E_M|
$

with per-sample weights $w_i = 1 / 30 + y_i$ computed inside each
forward pass. In the rare case of a mono-gender mini-batch (mitigated by
the weighted sampler), the loss falls back to the global weighted MSE over
the batch. This end-to-end alignment between training objective and
evaluation criterion removes the optimisation gap that would otherwise arise
from a vanilla MSE or Huber loss.

== Ensemble Inference

Final predictions are produced by averaging the outputs of *three independent
EfficientNet-B4 checkpoints* (seeds 123, 42, 456) trained with identical
hyperparameters. Ensemble diversity is achieved through different random
initialisations and data shuffling. Each checkpoint optionally applies TTA
before the predictions are averaged, further reducing variance. The ensemble
prediction for sample $i$ is:

$
  hat(y)_i = 1 / K sum_(k=1)^K "TTA"(hat(y)_i^((k)))
$

// ============================================================
= Alternative Baseline Explored
// ============================================================

Prior to adopting the deep learning approach, we developed a *fully
training-free, deterministic classical CV heuristic* to establish a
performance ceiling for analytical methods. The pipeline operates as
follows: (1) CLAHE is applied to the Y channel of YCrCb to normalise
illumination; (2) a primary skin mask is generated from fixed YCrCb
chrominance ranges ($"Cr" in [133, 173]$, $"Cb" in [77, 127]$), which
are illumination-invariant across the human skin-tone gamut; (3) an
adaptive HSV secondary mask supplements the primary detector; (4) a
Laplacian texture filter with anatomy-aware exclusion masks for eyes and
mouth suppresses false edges from natural facial features; (5) the
occluded fraction is estimated from the non-skin pixel ratio inside a
central circular ROI. While this approach is *fully generalised* — requiring
no training data and operating on any face crop without prior calibration —
it encountered a hard performance ceiling (~0.023 on the challenge metric)
due to JPEG compression artefacts, low resolution, and the high semantic
complexity of occlusion types (scarves, hands, hair). This validated our
decision to pivot to deep learning, which can model the semantic richness
of the task at the cost of data dependency.

// ============================================================
= Limitations
// ============================================================

*Data dependency:* Unlike the classical baseline, our EfficientNet
regressor requires tens of thousands of labelled examples. Performance
degrades on out-of-distribution face types or occlusion styles not
represented in the training corpus.

*Computational cost:* EfficientNet-B4 inference requires a GPU for
practical throughput. The training pipeline (40 epochs × 80K images ×
3 seeds) requires several hours on a modern NVIDIA GPU, limiting the number
of hyperparameter configurations that can be evaluated.

*Hyperparameter sensitivity:* The gender balance and male boost
parameters of the weighted sampler interact non-trivially with the loss
imbalance term. The chosen value (3.0) was determined empirically and may
not generalise to data with different gender ratios.

*Ensemble correlation:* The three checkpoints share the same
backbone and architecture. Their predictions are correlated, reducing the
variance-reduction benefit compared to a diverse ensemble (e.g., mixing
EfficientNet-B4 with ConvNeXt or ViT architectures). We explored
ConvNeXt-Base and ViT-Base variants (configurations available in
`model-training/configs/`) but EfficientNet-B4 produced the best
single-model scores within our compute budget.

// ============================================================
= Conclusion
// ============================================================

We presented a deep learning solution based on fine-tuned EfficientNet-B4
that directly optimises the gender-balanced, occlusion-weighted evaluation
metric. Key design decisions — weighted sampling, a custom loss identical
to the metric, and multi-seed ensembling — each target a distinct aspect of
the challenge's evaluation structure. A deterministic classical CV baseline
was built and benchmarked first, validating the necessity of the learned
approach for handling the semantic complexity of face occlusion in
compressed, low-resolution crops.

// ============================================================
// References
// ============================================================
#bibliography("refs.bib")