# Breast Cancer Detection (Machine Learning)

Classifies breast masses as malignant or benign from 30 nucleus measurements
(Wisconsin Diagnostic Breast Cancer data, 569 samples).

**Educational project, not a medical device. Never use it for real diagnosis.**

## Approach
1. EDA: class balance (63% benign / 37% malignant), feature separation, multicollinearity
2. Stratified 80/20 split; scaling inside a Pipeline to avoid data leakage
3. Compared 6 models with 5-fold cross-validation, tuned the best with GridSearchCV
4. Final evaluation on a held-out test set; decision threshold chosen on training data only
5. Grouped permutation importance for interpretation
6. Gradio web app where you enter the 30 measurements and get a prediction

## Results (test set, 114 samples)
| Setting | Accuracy | Precision | Recall | Missed cancers | False alarms |
|---|---|---|---|---|---|
| SVM, threshold 0.50 | 98.2% | 100% | 95.2% | 2 | 0 |
| SVM, threshold 0.25 | 96.5% | 93.2% | 97.6% | 1 | 3 |

ROC-AUC 0.996. Logistic Regression performed essentially the same in cross-validation.

## Key findings
- Nucleus size (radius/perimeter/area) is the dominant signal; texture and concavity add smaller contributions.
- Simple linear/kernel models beat tree ensembles on this dataset.
- A lower threshold trades a few extra false alarms for fewer missed cancers.

## Limitations
Only 569 samples from one source, and 42 malignant cases in the test set, so each
error shifts recall by about 2.4 points. Needs external validation before any real use.

## Run the app
    pip install -r requirements.txt
    python app.py

Then open the local address it prints. The notebook `Breast_Cancer_Detection.ipynb`
contains the full analysis.
