# ElderGuard Analytics - Week 6 and Week 7 Project

## Group Information

Group Name: `<fill in group name>`

| Member | Admin No. | Contribution / File Owned |
|---|---|---|
| Member 1 | `<fill in>` | `src/data_loader.py` |
| Member 2 | `<fill in>` | `src/preprocessing.py` |
| Member 3 | `<fill in>` | `src/train.py`, `src/evaluate.py` |

> Each member should update this table based on the actual file they wrote or explained during presentation.

---

## Project Overview

ElderGuard Analytics aims to support elderly residents living independently by analysing smart home environmental sensor data. The final machine learning goal is to predict the resident's `Activity Level` using indoor air quality, environmental readings, HVAC operation, and ambient lighting information.

The dataset is stored in a SQLite database:

```text
data/gas_monitoring.db
```

The main target column is:

```text
Activity Level
```

---

## Folder Structure

```text
project_folder/
├── eda.ipynb
├── config.json
├── README.md
├── requirements.txt
├── run.sh
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   └── gas_monitoring.db
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
│
├── results/
│   ├── model_results.csv
│   ├── feature_importance.csv
│   └── EDA output csv files
│
├── visuals/
│   ├── EDA charts
│   └── confusion matrix charts
│
└── saved_model/
    └── best_model.pkl
```

---

## Week 6: EDA + Insights

The Week 6 EDA work is completed in:

```text
eda.ipynb
```

### EDA Steps Completed

1. Loaded the dataset from SQLite.
2. Checked available database tables.
3. Inspected dataset shape, columns, and data types.
4. Checked missing values.
5. Checked duplicate rows.
6. Cleaned inconsistent `Activity Level` labels for analysis.
7. Analysed target class distribution.
8. Generated numerical summaries.
9. Checked possible outliers using boxplots and IQR.
10. Analysed categorical feature distributions.
11. Generated a correlation matrix.
12. Compared sensor averages by activity level.
13. Saved EDA outputs into `results/` and `visuals/`.

### Key EDA Insights

- The dataset contains environmental sensor readings, categorical smart home information, and the target column `Activity Level`.
- The target labels required cleaning because some labels were inconsistent, such as labels without spacing.
- `Session ID` was treated as an identifier and should not be used as a predictive feature.
- Missing values exist in the dataset, so imputation is required before model training.
- Numerical columns contain possible outliers, which is important because the problem statement mentions possible synthetic or contaminated data.
- Activity class distribution should be considered during model evaluation. Macro F1 is useful because it gives equal importance to each class.

---

## Week 7: Machine Learning Pipeline

The Week 7 machine learning pipeline is implemented using `.py` files inside the `src/` folder.

The pipeline performs:

1. Data ingestion from SQLite.
2. Data cleaning.
3. Feature and target splitting.
4. Numerical and categorical preprocessing.
5. Model training.
6. Hyperparameter tuning using `GridSearchCV`.
7. Model evaluation.
8. Best model selection.
9. Feature importance analysis.
10. Saving model outputs.

---

## Models Trained

At least three models were trained and compared:

| Model | Reason for Use |
|---|---|
| Logistic Regression | Baseline model that is simple and easy to interpret. |
| Decision Tree | Easy to explain during code walkthrough and captures non-linear rules. |
| Random Forest | Ensemble model that handles non-linear patterns and usually performs better than a single tree. |

---

## Model Evaluation Metrics

The main metric used for model selection is:

```text
Macro F1 Score
```

### Why Macro F1?

Accuracy can be misleading if one activity class appears more often than the others. Macro F1 calculates the F1 score for each class and averages them equally, making it more suitable for multi-class activity prediction.

Other metrics saved:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Best cross-validation score

---

## Model Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Random Forest | 0.6124 | 0.5462 | 0.6406 |
| Logistic Regression | 0.5941 | 0.5145 | 0.6167 |
| Decision Tree | 0.6012 | 0.4979 | 0.6057 |

The best model based on Macro F1 is:

```text
Random Forest
```

The best model is saved at:

```text
saved_model/best_model.pkl
```

---

## Feature Importance

Permutation feature importance was used so that importance can be calculated in a consistent way for the selected model.

Top important features from the current run:

| Rank | Feature |
|---:|---|
| 1 | CO2_ElectroChemicalSensor |
| 2 | MetalOxideSensor_Unit4 |
| 3 | MetalOxideSensor_Unit2 |
| 4 | MetalOxideSensor_Unit1 |
| 5 | MetalOxideSensor_Unit3 |

These features suggest that air quality and gas-related readings are useful for predicting activity level.

---

## Feature Engineering and Preprocessing

### Label Cleaning

The target column was cleaned to standardise inconsistent labels such as:

```text
LowActivity -> Low Activity
ModerateActivity -> Moderate Activity
HighActivity -> High Activity
```

### Dropped Feature

```text
Session ID
```

Reason: It is an identifier and does not represent an environmental or behavioural reading.

### Missing Value Handling

- Numerical columns: median imputation
- Categorical columns: most frequent value imputation

### Categorical Encoding

Categorical features such as `Time of Day`, `HVAC Operation Mode`, and `Ambient Light Level` were encoded using One-Hot Encoding.

### Scaling

Numerical features were scaled using `StandardScaler` for models that are affected by feature scale.

---

## How to Run Locally

### 1. Install libraries

```bash
pip install -r requirements.txt
```

### 2. Run the ML pipeline

```bash
python src/main.py --config config.json
```

### 3. Or run using shell script

```bash
bash run.sh
```

---

## How to Run with Docker

### Build Docker image

```bash
docker build -t elderguard-pipeline .
```

### Run Docker container

```bash
docker run --rm elderguard-pipeline
```

### Or use Docker Compose

```bash
docker compose up --build
```

---

## Git and Version Control

Recommended commit sequence:

```bash
git add .
git commit -m "Add Week 6 EDA notebook and project structure"
git commit -m "Add configurable data loading and preprocessing pipeline"
git commit -m "Add model training, tuning and evaluation scripts"
git commit -m "Add Docker setup and update README"
```

Make the repository public and submit the GitHub link as required.

---

## Presentation Notes

Suggested split:

| Member | Presentation Part |
|---|---|
| Member 1 | Week 6 EDA, visualisations, missing values, outliers |
| Member 2 | Data loading, config file, preprocessing, feature engineering |
| Member 3 | Model choices, tuning, evaluation, results, feature importance |

Each member should be ready to explain their own `.py` file during the code walkthrough.
