# Diabetic Retinopathy Detection with EfficientNetB3

A deep-learning project for **binary diabetic retinopathy (DR)
classification** from retinal fundus images using transfer learning with
**EfficientNetB3** and the APTOS 2019 dataset setup.

The model predicts one of two classes:

-   `No DR` --- no diabetic retinopathy
-   `DR` --- diabetic retinopathy present

> **Important:** This is an experimental research project, not a
> clinically validated diagnostic tool. Predictions should not replace
> assessment by a qualified eye-care professional.

## Project Highlights

-   EfficientNetB3 pretrained on ImageNet
-   Two-stage transfer learning: train a classification head, then
    fine-tune the top backbone layers
-   Image resizing to `300 × 300`
-   Data augmentation: horizontal/vertical flips, rotation, zoom, and
    contrast adjustment
-   Class weighting to address class imbalance
-   Early stopping, learning-rate reduction, and best-checkpoint saving
    based on validation AUC
-   Evaluation using accuracy, precision, recall/sensitivity, F1-score,
    specificity, ROC-AUC, and a confusion matrix

## Dataset

The notebook expects images arranged in two class folders:

``` text
Diabetic Retinopathy/
├── No DR/
└── DR/
```

Reported class counts:

  Class         Number of images
  ----------- ------------------
  No DR                    1,805
  DR                         120
  **Total**            **1,925**

The code uses a stratified random split with `random_state=42`:

-   70% training
-   15% validation
-   15% testing

The reported test set contains **289 images**. Keep patient-level
separation in mind if patient identifiers are available; image-level
splitting alone may not prevent images from the same patient appearing
across splits.

## Model Architecture and Training

1.  **Backbone:** EfficientNetB3 pretrained on ImageNet, initially
    frozen.
2.  **Classification head:** Global average pooling → dropout (`0.4`) →
    one sigmoid output.
3.  **Stage 1 --- head training:** Adam optimizer with learning rate
    `1e-3`, binary cross-entropy.
4.  **Stage 2 --- fine-tuning:** Unfreeze the top 40 backbone layers
    while keeping batch-normalization layers frozen; Adam learning rate
    `1e-5`.
5.  **Training controls:** Early stopping based on validation AUC,
    ReduceLROnPlateau, and ModelCheckpoint.

Main settings in the notebook:

  Setting                 Value
  ----------------------- -----------
  Input size              300 × 300
  Batch size              16
  Head-training maximum   10 epochs
  Fine-tuning maximum     15 epochs
  Random seed             42
  Decision threshold      0.5

## Test Results

The uploaded results report the following performance on the 289-image
test set:

  Metric                              Result
  ---------------------------- -------------
  Accuracy                        **98.96%**
  ROC-AUC                         **0.9996**
  Sensitivity (DR recall)        **100.00%**
  Specificity (No DR recall)      **98.89%**
  DR precision                        85.71%
  DR F1-score                         92.31%
  Macro-average F1-score              95.88%
  Weighted-average F1-score           99.00%

### Confusion matrix

  Actual class     Predicted No DR   Predicted DR
  -------------- ----------------- --------------
  No DR                        268              3
  DR                             0             18

On this test split, the model identified all 18 DR cases, while 3 No DR
images were classified as DR. The DR class is under-represented in the
dataset, so the test support is small; the reported metrics should be
interpreted with that limitation in mind.

### Result plots

The notebook saves result figures in the `results/` directory:

``` text
results/
├── confusion_matrix.png
└── roc_curve.png
```

Training curves for loss, accuracy, and AUC are also plotted by the
notebook. Add the generated figures to the repository if you want them
displayed below.

```{=html}
<!-- Uncomment and update these paths after committing the images:
![Confusion Matrix](results/confusion_matrix.png)
![ROC Curve](results/roc_curve.png)
-->
```
## Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your actual repository URL and folder
name.

### 2. Install dependencies

Use Python 3.10 or 3.11 with a compatible TensorFlow version and
install:

``` bash
pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter
```

Exact package versions were not specified in the supplied notebook. For
reproducible experiments, record the versions used in your environment
in a `requirements.txt` file.

### 3. Prepare the dataset

Place the images in the two-folder structure described above. The
notebook currently uses a Google Drive path:

``` python
DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/dataset/Diabetic Retinopathy"
```

Update `DATA_DIR` to match your own dataset location. In Google Colab,
mount Google Drive before running the notebook.

### 4. Run the notebook

Open `Diabetic_retinopathy_Efficient_B3.ipynb` in Google Colab or
Jupyter Notebook and run the cells in order. A GPU runtime is
recommended to reduce training time.

The notebook saves model checkpoints such as `best_head.keras` and
`best_finetune.keras`, and creates evaluation plots in `results/`.

## Repository Structure

Adjust this example to match the files you actually commit:

``` text
.
├── README.md
├── Diabetic_retinopathy_Efficient_B3.ipynb
├── results/
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── requirements.txt
```

Do not commit private patient information or dataset images unless you
have permission and the dataset licence allows redistribution. If the
dataset is not included, explain how users can obtain it from the
original source.

## Limitations and Responsible Use

-   The test set contains only 18 DR-positive images, so sensitivity
    estimates have substantial uncertainty.
-   The class distribution is imbalanced (1,805 No DR vs. 120 DR
    images).
-   A high score on one split does not establish generalization to other
    hospitals, cameras, populations, or image-quality conditions.
-   Confirm there is no patient-level leakage and use an independent
    external test set before making clinical-performance claims.
-   This model is for educational and research purposes only and is not
    intended for diagnosis or treatment decisions.

## Future Improvements

-   Evaluate on an independent external dataset.
-   Use patient-level splitting when patient IDs are available.
-   Report confidence intervals and calibration in addition to point
    metrics.
-   Perform error analysis on false positives and clinically challenging
    cases.
-   Document preprocessing and software versions for full
    reproducibility.

## Citation

If you use this project, cite the repository and the original dataset
according to its official citation requirements. Add a publication
citation here if this work is published.

## Acknowledgements

-   APTOS 2019 Blindness Detection dataset setup
-   TensorFlow / Keras and the EfficientNetB3 ImageNet pretrained
    weights
-   Scikit-learn for data splitting and evaluation metrics
