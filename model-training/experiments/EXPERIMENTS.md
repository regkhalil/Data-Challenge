# Experiments Log

This folder documents every approach explored during the challenge. Only the code
in the parent `model-training/` reproduces the final submitted result; the configs,
scripts and tools here are kept for transparency and to support the report's
discussion of alternatives.

## Models trained

| Model | Backbone | Seed | Val score | Status | Used in final |
|-------|----------|------|-----------|--------|---------------|
| convnext_large_s999 | ConvNeXt-Large | 999 | **0.0011** | Best solo model | ✅ |
| convnext_large_v1 | ConvNeXt-Large | 123 | 0.0011 | Strong | ✅ |
| convnextv2_base_v1 | ConvNeXtV2-Base | 123 | 0.0014 | Strong | ✅ |
| convnext_base_v2 | ConvNeXt-Base | 123 | 0.0016 | Clean recipe | ✅ |
| convnext_base_v1 | ConvNeXt-Base | 42 | 0.0016 | Clean recipe | ✅ |
| convnext_base_s456 | ConvNeXt-Base | 456 | 0.0018 | Diversity (seed) | ✅ |
| b4_v2_s123 | EfficientNet-B4 | 123 | 0.0017 | Architectural diversity (low weight) | ✅ |
| cnx_synth / cnx_synth_resume | ConvNeXt-Base + synthetic degradation | 123 | 0.00129 | **Overfits val** (test 0.00218) | ❌ |
| convnext_base_s789 | ConvNeXt-Base | 789 | 0.0021 | High ErrM, weak | ❌ |
| vit_base_v1 | ViT-Base/16 | 123 | 0.00189 | Worse, hurts ensemble | ❌ |
| effnetv2_m_v1 | EfficientNetV2-M | 123 | 0.00188 | Worse | ❌ |
| swin_base_v1 | Swin-Base | 123 | 0.00242 | Worse | ❌ |

## Submitted ensembles (interim leaderboard)

| Submission | Composition | Score |
|------------|-------------|-------|
| **champions_s999** | 0.55·ENS4 + 0.45·s6 | **0.00112** (best) |
| champions_55_45 | 0.55·ENS3 + 0.45·s6 | 0.00113 |
| ensemble_3models (ENS3) | mean(convnextv2_base, large_v1, base_s456) | 0.00115 |
| s6 | mean(base_v1, base_v2, b4_v2) | 0.00116 |
| convnext_large_v1 (solo) | single model | 0.00118 |
| convnextv2_base_v1 (solo) | single model | 0.00124 |
| mega_ensemble (with ViT) | base_v1 + b4 + ViT | 0.00121 |
| cnx_synth_resume | synthetic-degradation recipe | 0.00218 |

- **ENS4** = mean(convnextv2_base, convnext_large_v1, convnext_base_s456, convnext_large_s999)
- **s6** = mean(convnext_base_v1, convnext_base_v2, b4_v2_s123)

## Key findings

1. **Validation score is unreliable for the synthetic-degradation recipe.** Adding
   `synth_degrad_p`, aggressive `tail_k` and gender-scaled boosting reached the best
   val score (0.00129) but the worst test score (0.00218): a clear case of val
   overfitting. The clean recipe (no synthetic tricks, moderate `male_boost=2.5`)
   generalises far better. Configs: `experiments/configs/config_cnx_synth*.yaml`,
   `config_convnext_tail_ft.yaml`.

2. **Test gender labels are hidden** (`test_students.csv` has no gender column;
   submissions use `gender=x`). Per-gender post-hoc calibration is therefore
   impossible. The only lever on the `|ErrF − ErrM|` term is `male_boost` during
   training. A global (single α, β) post-hoc scaling was tested
   (`experiments/tools/calibrate_global.py`) and gave a negligible 0.91 % gain,
   confirming the clean models are already well calibrated.

3. **Ensembling clean, diverse ConvNeXt models beats any single model.** Each model
   scores ~0.0011–0.0018 on val, but their average reaches 0.00112 on the leaderboard.

4. **Architecture matters for the ensemble.** ViT-Base, EfficientNetV2-M and
   Swin-Base were all clearly worse and adding ViT degraded the ensemble (0.00121).
   EfficientNet-B4 at a small weight, however, adds useful decorrelation and is part
   of the winning blend.

5. **Weighted blending of two ensembles helps slightly.** A 0.55/0.45 mix of the
   strong ConvNeXt ensemble (ENS3/ENS4) and the diverse s6 ensemble was consistently
   better than either alone. Fine-tuning weights further plateaued at 0.00112.

## Tools (`experiments/tools/`)

- `optimize_weights.py` — searches ensemble weights on a fixed validation split.
- `dump_val_preds.py` — dumps validation predictions for offline weight search.
- `calibrate_global.py` — global (α, β) post-hoc scaling fitted on validation.
- `calibrate_gender.py` — per-gender calibration (abandoned: test gender is hidden).
- `analyze_val_error.py` — per-sample validation error inspection.
- `ensemble_max.py` — max-based ensemble variant.
