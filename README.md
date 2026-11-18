# Heart Disease Prediction System

Logistic Regression on the UCI Cleveland Heart Disease dataset, trained in Kaggle, served by FastAPI, with a plain HTML/CSS/JS frontend.

> **Medical disclaimer:** educational machine learning project. It is not a clinically validated diagnostic tool and must not be used for medical decisions. See a qualified doctor for any health concern.

## Architecture

```
Kaggle notebook --> heart_disease_model.pkl --> FastAPI (backend/main.py) <--HTTP/JSON-- frontend (HTML/JS)
                    (full sklearn Pipeline)      GET /  GET /health  POST /predict
```

The `.pkl` holds the whole fitted Pipeline (imputers, scaler, one-hot encoder, Logistic Regression). The API only builds a DataFrame and calls the pipeline.

## Folder structure

```
heart-disease-prediction/
├── kaggle_notebook/
│   ├── heart_disease_logistic_regression.ipynb   executed notebook (with outputs)
│   └── model_versions.txt                        library versions used to train
├── backend/
│   ├── main.py                                   FastAPI app (also serves the frontend at /app/)
│   ├── heart_disease_model.pkl                   trained pipeline
│   └── requirements.txt                          pinned runtime dependencies
├── frontend/   index.html, style.css, script.js
├── tests/test_api.py
├── requirements-dev.txt                          runtime + pytest + httpx
├── Dockerfile  .dockerignore  render.yaml  .gitignore
└── README.md
```

Changes from the suggested layout: `render.yaml` (one-click Render config), `requirements-dev.txt` (test tools kept out of production), and the backend also mounts `frontend/` at `/app/` so one deployment serves both. The frontend still talks to the backend only through HTTP.

## Dataset

- Source: https://archive.ics.uci.edu/dataset/45/heart+disease
- File used: `processed.cleveland.data` (303 rows, 13 features + `num`, missing values marked `?`)
- Missing: `ca` (4 rows), `thal` (2 rows). Duplicates: 0. No out-of-range category values.
- Target: `num` is 0-4. Converted to binary: 0 = no disease (164 rows), 1-4 = disease (139 rows).

| Feature | Meaning | Type |
|---|---|---|
| age | years | numeric |
| sex | 0 female, 1 male | categorical |
| cp | chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic | categorical |
| trestbps | resting blood pressure (mm Hg) | numeric |
| chol | cholesterol (mg/dl) | numeric |
| fbs | fasting blood sugar > 120: 0 no, 1 yes | categorical |
| restecg | 0 normal, 1 ST-T abnormality, 2 LV hypertrophy | categorical |
| thalach | max heart rate achieved | numeric |
| exang | exercise-induced angina: 0 no, 1 yes | categorical |
| oldpeak | ST depression from exercise | numeric |
| slope | 1 upsloping, 2 flat, 3 downsloping | categorical |
| ca | major vessels colored (0-3) | numeric (median-imputed) |
| thal | 3 normal, 6 fixed defect, 7 reversible defect | categorical |

## Model and real results

Pipeline: median imputer + StandardScaler for numeric columns; most-frequent imputer + OneHotEncoder for categorical columns; Logistic Regression (L2). Preprocessing lives inside the pipeline, so it is refit on training folds only (no leakage). `GridSearchCV` with stratified 5-fold CV tuned `C` and `class_weight` (scoring ROC-AUC) on the 80% training split. Best: `C=0.1`, `class_weight=None`.

| Evaluation | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 5-fold CV on training set (mean) | 0.851 | 0.878 | 0.792 | 0.830 | 0.911 |
| Held-out test set (61 patients) | 0.885 | 0.839 | 0.929 | 0.881 | 0.961 |

The test set is small, so these numbers can move a lot with a different split; the CV figures are the steadier estimate. The deployed model is **refit on all 303 rows** with the chosen hyperparameters, so it has no separate held-out score; the table above describes the 80% model.

If you retrain on Kaggle your numbers may differ slightly depending on library versions.

## Phase 1 - Kaggle

1. Go to kaggle.com, sign in, open **Datasets > New Dataset**, upload `heart_disease.zip` (Kaggle extracts it), name it e.g. `uci-heart-disease`, create it.
2. **Create > New Notebook**, then **Add Input** and add your dataset.
3. **File > Import Notebook** and upload `kaggle_notebook/heart_disease_logistic_regression.ipynb` (or paste the cells). The notebook finds `processed.cleveland.data` under `/kaggle/input` by itself. Internet is not needed.
4. **Run All**. Check the metrics printed in the notebook.
5. Open the **Output** section (right panel > Data > Output, or Save Version > Save & Run All, then the Output tab). Download `heart_disease_model.pkl` and `model_versions.txt`.

The file is written to `/kaggle/working/heart_disease_model.pkl`.

**Version match matters.** A pickle stores scikit-learn internals, so loading it with a different scikit-learn version can warn or fail. Open `model_versions.txt` from Kaggle and set the same `scikit-learn`, `numpy`, `pandas`, `joblib` versions in `backend/requirements.txt`. The `.pkl` included here was trained with scikit-learn 1.9.1 / Python 3.12 and matches the pinned requirements, so it works as-is.

## Phase 2 - Backend (local)

```bash
cd heart-disease-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp /path/to/downloaded/heart_disease_model.pkl backend/    # only if you retrained on Kaggle
cd backend
uvicorn main:app --reload
```

- API root: http://127.0.0.1:8000/
- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Phase 3 - Frontend

With the backend running, open **http://127.0.0.1:8000/app/**, keep the sample values (or change them), click **Predict**.

To host the frontend separately, set `ALLOWED_ORIGINS` on the backend to your frontend URL and open the page as `index.html?api=https://YOUR-BACKEND-URL`.

## API testing

Swagger UI: `/docs` > `POST /predict` > Try it out > Execute (an example body is prefilled).

curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
```

Response (actual output from the included model):

```json
{"prediction":1,"result":"Heart Disease Detected","probability":{"no_heart_disease":0.0175,"heart_disease":0.9825}}
```

Invalid input (e.g. `"age": 500` or `"cp": 9`) returns HTTP 422 with a list of which field is wrong. If the model is not loaded, `/health` and `/predict` return 503.

Automated tests (from the project root):

```bash
python -m pytest -q tests
```

## Phase 4 - Deployment on Render

Free-tier facts checked on 18 Nov 2026 (re-check on render.com/pricing before you rely on them):

- Free web services spin down after 15 minutes without traffic; the next request takes about a minute to wake.
- 750 free instance hours per workspace per month; free instances have 512 MB RAM and 0.1 CPU. This app is small (scikit-learn + pandas), but the first request after waking is slow.
- Removing the sleep means a paid instance (from $7/month).
- Prices and limits can change.

Steps:

1. Create a GitHub repo and push (the `.pkl` must be committed; it is not git-ignored):
   ```bash
   git init
   git add .
   git commit -m "Heart disease prediction app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USER/heart-disease-prediction.git
   git push -u origin main
   ```
2. On render.com: **New > Web Service**, connect the GitHub repo.
3. Settings (also pre-filled by `render.yaml` if you use **New > Blueprint**):
   - Runtime: Python 3
   - Build command: `pip install -r backend/requirements.txt`
   - Start command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Instance type: Free
   - Environment variable: `PYTHON_VERSION` = `3.12.3`
   - Health check path: `/health`
4. Deploy. Open `https://YOUR-SERVICE.onrender.com/health` and confirm `"model_loaded": true`.
5. Test `https://YOUR-SERVICE.onrender.com/docs` and the interface at `https://YOUR-SERVICE.onrender.com/app/`.

Docker alternative (any Docker host): `docker build -t heart-app .` then `docker run -p 8000:8000 heart-app`. On Render choose runtime **Docker**; the root `Dockerfile` is used and no commands are needed.

## Verification checklist

- [ ] Notebook runs top to bottom without errors
- [ ] `heart_disease_model.pkl` downloaded and placed in `backend/`
- [ ] `model_versions.txt` matches `backend/requirements.txt`
- [ ] `python -m pytest -q tests` passes
- [ ] `/health` returns `model_loaded: true`
- [ ] `/predict` works from Swagger and from the web page
- [ ] Invalid input shows an error message, not a crash
- [ ] Public URL works after deploy

## Troubleshooting

| Problem | Fix |
|---|---|
| `InconsistentVersionWarning` or unpickling error | Match scikit-learn/numpy/pandas versions to `model_versions.txt` |
| `/health` returns 503 | `heart_disease_model.pkl` missing from `backend/` or not committed to GitHub |
| 422 on `/predict` | A field is missing, out of range, or has an invalid category; read the `detail` list |
| Page says it cannot reach the backend | Server not running, wrong `?api=` URL, or the free Render service is still waking up |
| Notebook cannot find the data | Confirm the dataset is added as input and contains `processed.cleveland.data` |
| Render build fails on Python version | Set `PYTHON_VERSION` to a version supported by your pinned packages (3.12.3 here) |
| `pip` cannot find a pinned version | Use the versions listed in your own `model_versions.txt` |
