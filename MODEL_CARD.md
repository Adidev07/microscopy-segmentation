# Model / system card

## Intended use

Train a small neural pixel classifier on synthetic microscopy scenes and evaluate on separate images. Educational local experiment; no production deployment is implied.

## Data

Synthetic arrays, manually written documents or replayed JSON. No real user records. Seeds and sample inputs are recorded in app.py and sample_input.json.

## Method

Extract local 3x3 pixel windows and classify each center pixel using an eight-unit tanh hidden layer and logistic output. Backpropagate binary cross entropy through both layers.

## Evaluation protocol

Train and validate on entire independent images rather than randomly splitting neighboring pixels. Select the probability threshold on validation images and report test intersection over union against an intensity threshold baseline.

## Known limits

Synthetic circles stand in for cells. This predicts foreground pixels, not separate cell instances, and does not reproduce CellSAM or clinical pathology performance.

## Verification status

Local tests and default experiment are executed during packaging. Live model generation is not exercised in the build. A successful validation result is limited to the checks implemented in code.

## Further work

Use licensed microscopy masks, split by specimen, compare U-Net and pretrained models, and measure instance-level segmentation errors.
