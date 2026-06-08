# ElderGuard Analytics – Activity Level Prediction Pipeline

## Group Information
| Role | Name |
|------|------|
| Group Member |  Azmi  |
| Group Member | Javier |
| Group Member | Yan De |

**GitHub Repository:** *https://github.com/drxs69/EGT309-Project*

---

## File Ownership

| File | Author |
|------|--------|
| `src/config.py` | Azmi |
| `src/data_ingestion.py` | Azmi |
| `src/preprocessing.py` | Javier |
| `src/model_training.py` | Javier |
| `src/model_evaluation.py` | Yan De |
| `src/pipeline.py` | Yan De |
| `eda.ipynb` | Azmi , Javier , Yan De |

---

## Project Structure

```
project/
├── data/
│   └── gas_monitoring.db          # SQLite sensor dataset
├── src/
│   ├── config.py                  # Central config (paths, hyperparams)
│   ├── data_ingestion.py          # SQLite loader
│   ├── preprocessing.py           # Cleaning, imputation, feature engineering, encoding
│   ├── model_training.py          # Model definitions, training, cross-validation
│   ├── model_evaluation.py        # Metrics, confusion matrices, feature importance plots
│   └── pipeline.py                # End-to-end orchestration entry point
├── saved_model/
│   ├── RandomForest.joblib
│   ├── GradientBoosting.joblib
│   ├── LogisticRegression.joblib
│   └── plots/                     # Confusion matrices and feature importance charts
├── eda.ipynb                      # Exploratory Data Analysis notebook
├── Dockerfile                     # Containerised runtime
├── requirements.txt
├── run.sh                         # Convenience runner script
└── README.md
```

---

## How to Run the Pipeline

### Option 1 – Direct Python (recommended for development)

```bash
# From project root
pip install -r requirements.txt
python src/pipeline.py
```

With CLI overrides:
```bash
python src/pipeline.py --rf_n_estimators 300 --gb_learning_rate 0.05 --test_size 0.25
```

Using the convenience script:
```bash
chmod +x run.sh
./run.sh
```

### Option 2 – Docker

```bash
# Build image
docker build -t elderguard-pipeline .

# Run pipeline (mount data directory)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/saved_model:/app/saved_model \
  elderguard-pipeline

# With CLI overrides
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/saved_model:/app/saved_model \
  elderguard-pipeline python src/pipeline.py --rf_n_estimators 300
```

### Docker Development Environment

```bash
# Start interactive development container
docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  elderguard-pipeline bash

# Inside container: run individual modules or the full pipeline
python src/pipeline.py
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/gas_monitoring.db` | Path to the SQLite database |
| `--test_size` | `0.2` | Fraction of data held out for testing |
| `--rf_n_estimators` | `200` | Number of trees in Random Forest |
| `--gb_n_estimators` | `200` | Number of boosting rounds |
| `--gb_learning_rate` | `0.1` | Gradient Boosting learning rate |
| `--lr_C` | `1.0` | Regularisation strength for Logistic Regression |
| `--skip_train` | `False` | If set, load saved models and skip training |

---

## Key EDA Findings

1. **Label normalisation required** – The `Activity Level` column contained 6 raw variants (e.g. `Low_Activity`, `LowActivity`) that all map to 3 canonical classes. HVAC Operation Mode similarly had mixed casing.

2. **Temperature outliers (8.7% of rows)** – Maximum recorded temperature was 307 °C, physically impossible indoors. These are synthetic contamination / sensor faults. Values are capped at the IQR upper fence (~29.5 °C) rather than dropped to preserve sample size.

3. **Missing data in 4 columns** – Humidity (~19%), MetalOxideSensor_Unit2 (~14%), Ambient Light Level (~10.5%), and CO_GasSensor (~8.3%) have missing values. Numeric columns are imputed with training-set medians; the categorical column uses the training-set mode. Imputation is fit on training data only to prevent leakage.

4. **Class imbalance** – Low Activity 57.7%, Moderate Activity 31.4%, High Activity 10.9%. Macro-averaged F1 is the primary evaluation metric because accuracy alone would be misleading, and missing a High Activity episode carries clinical risk.

5. **High sensor intercorrelation** – CO2 sensors (r ≈ 0.99) and Metal Oxide sensors are highly correlated. Sensor-fusion features (`CO2_mean`, `CO2_diff`, `MOS_mean`, `MOS_range`) were engineered to reduce redundancy and add discriminative signals.

6. **Ambient Light Level and Time of Day are strong discriminators** – Very dim/night conditions are strongly associated with Low Activity; bright conditions correlate with higher activity. CO2 levels increase progressively from Low → High Activity, consistent with respiration and movement.

---

## Engineered Features

| Feature | Formula | Justification |
|---|---|---|
| `CO2_mean` | mean(CO2_IR, CO2_EC) | Reduces noise from two highly correlated sensors |
| `CO2_diff` | |CO2_IR − CO2_EC| | Sensor disagreement signal; may reflect air stratification during movement |
| `MOS_mean` | mean(MOS units 1–4) | Overall VOC/gas load; rises with cooking and exertion |
| `MOS_range` | max(MOS) − min(MOS) | Spatial variance; increases when resident moves between rooms |
| `is_night` | 1 if Time of Day == 'night' | Encodes the strongest categorical signal compactly |
| `hvac_active` | 1 if HVAC not in {off, maintenance} | Active HVAC correlates with waking hours and resident behaviour |

---

## Model Choice and Justification

### Models Trained

| Model | Type | Justification |
|---|---|---|
| **Random Forest** | Ensemble (bagging) | Handles non-linear interactions and high-dimensionality well; robust to outliers and irrelevant features; provides native feature importances; no assumption about data distribution |
| **Gradient Boosting** | Ensemble (boosting) | Sequentially corrects residual errors; typically achieves best accuracy on structured tabular data; provides feature importances; handles class imbalance better than single trees |
| **Logistic Regression** | Linear (baseline) | Interpretable linear baseline; fast to train; probability estimates are well-calibrated; useful for benchmarking and explaining the contribution of each feature via coefficients |

All three are appropriate for multi-class classification without modification.

### Evaluation Metrics

**Primary: Macro-averaged F1** – chosen because the dataset is imbalanced (High Activity is only 11%) and all three classes carry clinical importance. Macro F1 weights each class equally, ensuring the model is not rewarded for ignoring the minority class.

**Secondary: Accuracy** – reported for comparison and interpretability, but understood to be inflated by the dominant Low Activity class.

**Per-class Recall for High Activity** – highlighted in the classification report because a false negative (missing a High Activity episode that could indicate distress or fall) is more costly than a false positive in an elder-care monitoring context.

### Results Summary

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 0.6865 | 0.5395 |
| Gradient Boosting | 0.6550 | 0.5103 |
| Logistic Regression | 0.6280 | 0.4095 |

Random Forest achieves the best macro F1 on this dataset. Gradient Boosting is competitive. Logistic Regression struggles with the High Activity minority class, confirming the dataset has non-linear structure that benefits from tree-based models.

### Tuning Notes

Hyperparameters are configurable via `src/config.py` or CLI arguments. The current values (200 estimators, learning rate 0.1, depth 5 for GBM) were selected as sensible defaults based on dataset size (10,000 rows, 27 features). Further tuning via `GridSearchCV` or `RandomizedSearchCV` can be enabled by modifying `model_training.py`.

---

## GenAI Usage Declaration

*Work in Progress!*
