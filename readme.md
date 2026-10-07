# Credit Card Fraud Detection

Detecting fraudulent credit card transactions in a highly imbalanced dataset (0.17% fraud), comparing logistic regression, Random Forest, and XGBoost, and testing three common imbalance-handling techniques.

**Author:** Santhiya V
**Tools:** Python, pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, matplotlib, seaborn

---

## Summary

- Fraud is extremely rare, so accuracy is misleading: a model that always predicts "legitimate" scores **99.83% accuracy** while catching zero fraud. This project uses **PR-AUC, precision, and recall** instead.
- A tuned **XGBoost** model reached a test **PR-AUC of 0.826** (baseline logistic regression: 0.695), catching **73 of 95 frauds** with only **2 false alarms**.
- SMOTE, under-sampling, and class weighting raised recall to about 87%, but precision dropped to about 5% (roughly 1,400 false alarms) with no PR-AUC gain.
- The model's decision threshold had little effect, because fraud scores are well separated from legitimate scores.

---

## Problem and business context

A fraud detection system must catch as much fraud as possible without blocking too many genuine customers. The two kinds of error have different costs:

- **Missed fraud (false negative):** direct financial loss.
- **False alarm (false positive):** a blocked legitimate payment, an annoyed customer, and manual review work.

Because fraud is so rare, the challenge is less about building a model and more about evaluating it honestly and understanding the precision/recall trade-off.

---

## Data

- **Source:** Kaggle, "Credit Card Fraud Detection" (ULB Machine Learning Group). European cardholder transactions over two days.
- **Size:** 284,807 transactions, 31 columns: `Time`, `Amount`, `V1`-`V28` (anonymized PCA components), and `Class` (1 = fraud).
- **Fraud rate:** 492 frauds (0.173%).
- **Quality:** no missing values; 1,081 exact duplicate rows (19 of them fraud).
- **After removing duplicates:** 283,726 rows, 473 frauds (0.167%).

> The dataset is not included in this repository. Download `creditcard.csv` from the Kaggle page and place it in `data/`.

**Citation:** please follow the citation requested on the Kaggle dataset page (verify the exact wording there). It lists: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi, *Calibration of Probability with Undersampling for Unbalanced Classification*, IEEE CIDM, 2015.

### Exploratory findings

![Class distribution](images/class_distribution.png)

- Amount is heavily skewed. For fraud, the median is **$9.25** but the mean is **$122.21**, so medians and percentiles describe it better than means.
- A quarter of frauds are $1 or less, but fraud also occurs at large amounts. Amount alone does not separate fraud from legitimate transactions.
- `Time` is seconds elapsed since the first transaction, not clock time, so no claims are made about time-of-day patterns.

---

## Approach

1. **Remove exact duplicates** to avoid the same row appearing in both train and test.
2. **Stratified 80/20 train/test split** (`random_state=42`), done *before* any scaling or resampling. Test set: 56,746 rows with 95 frauds.
3. **Scale `Amount` and `Time`** with `RobustScaler`, fitted on training data only (robust to Amount outliers). `V1`-`V28` are already PCA-scaled.
4. **Baseline:** logistic regression.
5. **Imbalance handling** (applied inside pipelines, so only the training data is resampled): class weights, random under-sampling, SMOTE.
6. **Tree ensembles:** Random Forest and XGBoost.
7. **Model comparison** with 5-fold stratified cross-validation on the training set, scored by PR-AUC.
8. **Tuning** XGBoost with `RandomizedSearchCV` (20 candidates, 5-fold stratified CV, PR-AUC).
9. **Threshold analysis** using out-of-fold predictions on the training set only.
10. **Single final evaluation** on the untouched test set, followed by SHAP and error analysis.

---

## Results

### Test-set comparison (threshold 0.5)

| Model | Precision | Recall | False alarms | Missed frauds | PR-AUC |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.862 | 0.590 | 9 | 39 | 0.695 |
| LR + class weights | 0.056 | 0.874 | 1,393 | 12 | 0.672 |
| LR + under-sampling | 0.051 | 0.874 | 1,552 | 12 | 0.600 |
| LR + SMOTE | 0.054 | 0.874 | 1,463 | 12 | 0.673 |
| Random Forest | 0.958 | 0.726 | 3 | 26 | 0.812 |
| XGBoost (default settings) | 0.914 | 0.779 | 7 | 21 | 0.828 |
| **XGBoost (tuned)** | **0.973** | **0.768** | **2** | **22** | **0.826** |

The test set contains only 95 frauds, so small differences between models (for example Random Forest vs. XGBoost) are within noise.

### Cross-validated PR-AUC (training set, 5-fold)

| Model | Mean | Std |
|---|---|---|
| Logistic regression | 0.755 | 0.030 |
| Random Forest | 0.835 | 0.033 |
| XGBoost | 0.847 | 0.024 |
| XGBoost (best tuned candidate) | 0.858 | n/a |

Tuned parameters: `n_estimators=500`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.7`, `scale_pos_weight=10`.

### Score distribution

![Model scores by true class](images/score_distribution.png)

Legitimate transactions score near 0 and most frauds score near 1. Almost no frauds fall between about 0.2 and 0.55.

---

## Key findings

1. **Accuracy is misleading.** The baseline scored 99.92% accuracy, only 0.08 percentage points above the "always legitimate" model, while missing 41% of frauds.
2. **Resampling shifted the trade-off but did not improve the model.** SMOTE, under-sampling, and class weights all raised recall from 59% to 87%, but at the cost of about 1,400 extra false alarms, and PR-AUC did not improve. They behave like a lower decision threshold, not a better ranking.
3. **Tree ensembles improved ranking quality.** PR-AUC rose from about 0.70 to about 0.83 on the test set, and the gap over logistic regression was several times larger than the cross-validation standard deviation.
4. **Random Forest and XGBoost are statistically comparable.** Their cross-validation difference (0.012) is smaller than the fold-to-fold standard deviation.
5. **Tuning added little.** Tuning raised CV PR-AUC by about 0.011, but test PR-AUC was essentially unchanged (0.828 vs. 0.826). Most of the gain came from choosing a tree ensemble.
6. **The threshold barely matters.** On out-of-fold predictions, F1 stays roughly flat (about 0.87) for thresholds from 0.2 to 0.7. On the test set, lowering the threshold from 0.5 to 0.3 caught no additional frauds and added 2 false alarms, so the default 0.5 was kept.

### Out-of-fold threshold table (training set)

| Threshold | Precision | Recall | False alarms | Missed |
|---|---|---|---|---|
| 0.1 | 0.866 | 0.841 | 49 | 60 |
| 0.3 | 0.913 | 0.828 | 30 | 65 |
| 0.5 | 0.925 | 0.820 | 25 | 68 |
| 0.7 | 0.944 | 0.807 | 18 | 73 |
| 0.9 | 0.954 | 0.767 | 14 | 88 |

---

## Interpretation

### Feature importance and SHAP

![XGBoost feature importance](images/feature_importance.png)
![SHAP summary plot](images/shap_summary.png)

- `V14` is the most influential feature in both views. Low `V14` values push predictions strongly toward fraud.
- Low `V12`, `V10`, `V17` and high `V4`, `V11` also push toward fraud.
- `Amount` plays a small role.
- The SHAP plot uses all 95 test frauds plus 1,000 sampled legitimate transactions, so it over-represents fraud compared with real data.
- `V1`-`V28` are anonymized, so no business meaning is assigned to them. SHAP describes what drives the model's predictions, not what causes fraud.

### The 22 missed frauds

- Missed frauds received very low scores (median probability 0.0003, maximum 0.195), while every caught fraud scored at least 0.555.
- This suggests the missed frauds look like ordinary transactions in the available features, which would explain why changing the threshold does not recover them. This is a hypothesis, not a proven cause.
- Their amounts span both very small and large values (median $2.01; 75th percentile $266.50), so amount alone does not explain why they were missed.

---

## Limitations

- **Small test set:** only 95 frauds, so recall (73/95) has an uncertainty of roughly 8 percentage points.
- **Anonymized features:** the V1-V28 components cannot be interpreted in business terms, and limited feature engineering is possible.
- **Random split:** a time-based split would better reflect real deployment, where models predict future transactions.
- **Single dataset:** results may not generalize to other card networks, regions, or time periods.
- **No cost model:** the threshold was not tied to explicit costs for missed fraud versus false alarms.
- **Duplicates removed:** this dropped 19 of 492 frauds (about 4%).

---

## How to run

```
credit-card-fraud-detection/
├── data/                 # place creditcard.csv here (not tracked)
├── images/               # charts used in this README
├── notebooks/
│   └── fraud_detection.ipynb
├── requirements.txt
└── README.md
```

```bash
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
jupyter
```

---

## Possible next steps

- Time-based train/test split.
- Cost-sensitive threshold selection using estimated costs of missed fraud vs. false alarms.
- Calibrating predicted probabilities.
- Testing LightGBM or CatBoost, and evaluating on additional fraud datasets.
