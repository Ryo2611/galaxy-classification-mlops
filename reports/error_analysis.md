# Error Analysis

## Objective

Evaluate where the Galaxy Zoo morphology model fails, rather than relying only on average validation loss. This mirrors real data science work: identify failure modes, build hypotheses, and decide what to improve next.

## Analysis Scope

Week 2 error analysis covers:

- Class-wise RMSE
- High-error validation samples
- Systematic overprediction and underprediction
- Prediction distribution drift
- Qualitative image review
- Grad-CAM inspection for success and failure cases

## Expected Failure Modes

The model is expected to make larger errors for:

- Small or low-brightness galaxies
- Galaxies with ambiguous spiral structures
- Edge-on galaxies
- Images with nearby objects
- Noisy or uneven backgrounds
- Sparse Galaxy Zoo decision-tree labels

## How to Generate the Error Summary

```bash
python src/training/error_analysis.py \
  --labels data/external/training_solutions_rev1.csv \
  --predictions reports/resnet18_predictions.csv \
  --output reports/error_analysis_summary.json
```

The resulting JSON contains:

- Overall RMSE / MAE
- Top class-wise RMSE values
- Most overpredicted labels
- Most underpredicted labels
- Highest-error samples

## Explainability Link

Grad-CAM should be used on both successful and failed predictions:

- Correct examples: check whether the model focuses on galaxy center, spiral arms, or disk structure.
- Failure examples: check whether attention shifts to background noise, nearby objects, or image edges.

## Week 4 Operationalization

The error-analysis workflow is now represented as the `error_analysis` stage in `dvc.yaml`. Once `reports/resnet18_predictions.csv` exists, the summary can be regenerated with:

```bash
dvc repro error_analysis
```

This makes error analysis part of the same reproducible workflow as dataset statistics, baseline training, and evaluation.

## Discussion Template

After running the baseline, fill in:

| Question | Observation |
|---|---|
| Which labels have highest RMSE? | TBD |
| Are sparse labels harder? | TBD |
| What image patterns appear in high-error cases? | TBD |
| Does Grad-CAM focus on meaningful structures? | TBD |
| What is the next improvement? | TBD |

## Improvement Hypotheses

- Use class-wise loss weighting if sparse labels dominate error.
- Add brightness/background quality features to error analysis.
- Compare ResNet50 and EfficientNet-B0 to see whether capacity or architecture changes help.
- Add targeted augmentation for small, dim, or edge-on examples.
- Export a small gallery of high-error Grad-CAM examples for the final portfolio README.
