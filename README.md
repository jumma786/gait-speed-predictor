# Gait Speed Predictor

Predicts walking speed (m/s) from ground reaction force data using machine learning.  
**Random Forest | MAE: 0.057 m/s | R²: 0.958 | 42 tests passing | FastAPI served**

---

## Author

**Jumma Mohammad Teli**  
Data Analyst | ML Engineer — Birmingham, UK  
[LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786) | jummamohammad477@gmail.com

---

## 🚀 Live Demo

| | |
|---|---|
| **API (Swagger UI)** | [https://jummamohammad477-gait-speed-predictor.hf.space/docs](https://jummamohammad477-gait-speed-predictor.hf.space/docs) |
| **Health Check** | [https://jummamohammad477-gait-speed-predictor.hf.space/health](https://jummamohammad477-gait-speed-predictor.hf.space/health) |
| **Predict Endpoint** | [https://jummamohammad477-gait-speed-predictor.hf.space/predict](https://jummamohammad477-gait-speed-predictor.hf.space/predict) |
| **Hosted on** | HuggingFace Spaces (Docker) |

---

## The Business Question

Can biomechanical foot pressure data alone predict how fast someone is walking — without cameras or wearables?

Rehabilitation clinics and sports science labs rely on expensive infrared motion capture systems for gait assessment. This project demonstrates that force plate data alone — measuring how the foot pushes against the ground — is sufficient to predict walking speed with high accuracy. This has implications for lower-cost clinical gait screening tools.

---

## Results

| Model | MAE (m/s) | RMSE (m/s) | R² |
|-------|-----------|------------|-----|
| Baseline (mean) | 0.300 | 0.345 | 0.000 |
| Linear Regression | 0.065 | 0.072 | 0.943 |
| Ridge Regression | 0.058 | 0.065 | 0.956 |
| XGBoost | 0.058 | 0.066 | 0.956 |
| **Random Forest** | **0.057** | **0.065** | **0.958** |

Validation: **Leave-One-Subject-Out cross-validation** (21 folds) — each fold trains on 20 subjects and tests on one unseen subject. This is the gold-standard approach for gait ML to ensure the model generalises to new individuals.

**Headline:** Random Forest predicts walking speed within ±0.057 m/s — an 81% improvement over baseline.

---

## Key Finding (SHAP)

The top predictive feature is `vgrf_to_ap_left` — the ratio of vertical to anterior-posterior (propulsive) force on the left foot.

> *Contrary to expectation, step frequency is not the primary driver. It is the balance between vertical loading and forward propulsion on the left foot that most strongly predicts walking speed.*

This makes biomechanical sense: as speed increases, more force is directed forward (propulsion) relative to vertical (body weight support), shifting the ratio.

**Top 5 SHAP features:**
1. `vgrf_to_ap_left` — vertical-to-propulsive force ratio, left foot
2. `vgrf_to_ap_right` — vertical-to-propulsive force ratio, right foot
3. `stance_duration_ms_left` — how long left foot is on the ground
4. `step_freq_left` — left foot step frequency
5. `step_freq_mean` — bilateral mean step frequency

---

## Dataset

**GaitPhase Database** — Hebenstreit et al. (2014), FAU Erlangen-Nürnberg  
- 21 healthy subjects (20 included; subject 4 excluded — incomplete protocol)
- Walking on split-belt treadmill at 12 speeds: 0.6–1.7 m/s
- Ground reaction forces sampled at 1000 Hz
- 3D marker positions sampled at 200 Hz
- ~756 CSV files (force, marker, oversteps per subject per speed)

> ⚠️ The dataset is not included in this repository. It can be requested from the original authors or accessed via Kaggle: [human-gait-phase-dataset](https://www.kaggle.com/datasets/dasmehdixtr/human-gait-phase-dataset)

---

## Project Structure

```
gait-speed-predictor/
│
├── data/
│   └── processed/
│       ├── features.csv            # Step-level features from preprocessor
│       ├── model_features.csv      # Engineered features for modelling
│       ├── model_results.csv       # LOSO CV results for all models
│       └── plots/                  # Evaluation and SHAP plots
│           ├── actual_vs_predicted.png
│           ├── residuals.png
│           ├── shap_summary.png
│           └── shap_dependence_step_freq_mean.png
│
├── src/
│   ├── data_loader.py              # Load and parse raw CSVs
│   ├── preprocessor.py             # Step segmentation, GRF feature extraction
│   ├── feature_engineering.py      # Derived biomechanical features
│   ├── train.py                    # Model training and LOSO evaluation
│   └── evaluate.py                 # Plots and SHAP explainability
│
├── api/
│   ├── main.py                     # FastAPI application
│   ├── schema.py                   # Pydantic request/response models
│   └── model/
│       └── xgb_speed_model.pkl     # Saved model (XGBoost pipeline)
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py                 # 17 API endpoint tests
│   ├── test_features.py            # 14 feature engineering tests
│   └── test_model.py               # 12 model tests
│
├── app.py                          # Streamlit demo app
├── Dockerfile                      # HuggingFace Spaces deployment
├── .github/workflows/ci.yml        # GitHub Actions CI
├── requirements.txt
└── README.md
```

---

## Feature Engineering

45 features derived from raw force signals:

| Group | Features | Count |
|-------|----------|-------|
| Bilateral force (left/right) | Peak VGRF, mean VGRF, impulse, loading rate, stance duration, AP/ML forces | 18 |
| Asymmetry indices | Left-right asymmetry % for key features | 5 |
| Bilateral ratios | Left/right dominance ratios | 7 |
| Step frequency | Steps per second proxies | 3 |
| Force balance | Vertical-to-horizontal ratios | 4 |
| Impulse efficiency | Impulse per millisecond of contact | 2 |
| Composite | Mean bilateral, combined asymmetry score | 6 |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model status, MAE, R² |
| POST | `/predict` | Predict walking speed from 45 features |
| GET | `/features` | Feature list and top SHAP features |

### Example request
```bash
curl -X POST https://jummamohammad477-gait-speed-predictor.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"step_freq_mean": 1.414, "vgrf_to_ap_left": 8.69, ...}'
```

### Example response
```json
{
  "predicted_speed_ms": 1.1907,
  "confidence_lower": 1.109,
  "confidence_upper": 1.273,
  "model": "Random Forest",
  "note": "Prediction based on force plate biomechanical features only."
}
```

---

## Reproducing the Results

### 1. Install dependencies
```bash
conda create -n gait python=3.11
conda activate gait
pip install -r requirements.txt
```

### 2. Download the dataset
Place raw CSVs in `data/raw/` following the naming convention:  
`GP{subject}_{speed}_force.csv`, `GP{subject}_{speed}_marker.csv`

### 3. Run the pipeline
```bash
python src/preprocessor.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
```

### 4. Run the API locally
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Run the Streamlit app locally
```bash
streamlit run app.py
```

### 6. Run tests
```bash
pytest tests/ -v
```
Expected: **42 passed**

---

## Deployment

| Service | Platform | URL |
|---------|----------|-----|
| FastAPI | HuggingFace Spaces (Docker) | [Live API](https://jummamohammad477-gait-speed-predictor.hf.space/docs) |
| Streamlit | Local / Streamlit Cloud | `streamlit run app.py` |

---

## Stack

Python · Scikit-learn · XGBoost · SHAP · FastAPI · Pydantic · Pandas · NumPy · Matplotlib · Plotly · Streamlit · Docker · Pytest · GitHub Actions · HuggingFace Spaces

---

## Citation

Hebenstreit, F., Leibold, A., Krinner, S., Welsch, G., Lochmann, M., & Eskofier, B. M. (2014).  
*Description of the GaitPhase Database.* FAU Erlangen-Nürnberg.

---

## Related Work

This project extends [GaitSync](https://github.com/jumma786/gaitsync-gait-analysis-ml) — my MSc dissertation at Queen Mary University of London (2023–2024) — which used the same sensor modality for gait phase classification and movement-abnormality detection.
