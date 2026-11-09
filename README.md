# Agricultural Disease Presence Prediction

An experimental Streamlit application built around a classification model for an agricultural dataset.

## What the training notebook does

The notebook defines **Disease_Presence** as the target variable. It uses the remaining columns (excluding the raw Date column after extracting Day) as model inputs. The preprocessing pipeline imputes missing numeric and categorical values, scales numeric features, one-hot encodes categorical features, and applies SMOTE before Logistic Regression.

**Important distinction:** `Pest_Presence` is retained as an input feature in the notebook; `Disease_Presence` is the target. The Streamlit app now labels the prediction accordingly and avoids describing the target as pest presence.

## Files

- `train.ipynb` — model development and preprocessing workflow.
- `classification_data.csv` — dataset used by the notebook and app summary.
- `pest_model.joblib` — serialized fitted pipeline.
- `stream.py` — Streamlit interface.
- `requirements.txt` — Python dependencies.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run stream.py
```

Use package versions compatible with the environment that created `pest_model.joblib`. Scikit-learn/joblib serialized models are not guaranteed to be portable across arbitrary versions.

## Limitations and interpretation

- This is a prototype for learning and demonstration, not a substitute for field inspection or advice from agricultural extension professionals.
- The app displays the model's predicted class. The 0/1 mapping should be confirmed against the dataset documentation before interpreting it as disease absence or presence.
- The notebook's use of SMOTE and preprocessing should be checked for reproducibility, including evaluation on a held-out test set and class-wise metrics.
- The app's selected input fields must remain exactly aligned with the fitted pipeline's feature names and types.
- Dataset source, collection method, geographic coverage, crop coverage and limitations should be documented before drawing real-world conclusions.

## Suggested next steps

1. Add a reproducible evaluation report (precision, recall, F1, confusion matrix and class balance).
2. Add a test that compares app input columns with the fitted pipeline's expected feature schema.
3. Record the target label mapping and dataset provenance.
4. Pin a minimal set of tested dependencies rather than an environment-wide package freeze.
