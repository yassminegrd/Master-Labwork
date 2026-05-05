# DysDetect — Dysgraphia Detection System

Analyses handwriting images to detect dysgraphia using classical ML + CNN deep learning.

## Run & Operate

### Python (Dysgraphia App)
- `cd dysgraphia-app && python app.py` — run Flask app (port 5000)
- `cd dysgraphia-app && python train_ml.py` — retrain SVM, RF, KNN with augmentation
- `cd dysgraphia-app && python train_cnn.py` — train CNN model (TensorFlow/Keras)
- Workflow "Start application" runs Flask automatically

### Node.js (monorepo)
- `pnpm run typecheck` — full typecheck
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema (dev only)

### Required Env Vars
- `SESSION_SECRET` — Flask session secret (optional, has default)
- `PORT` — overrides Flask port (default 5000)

## Stack

### Dysgraphia App (Python)
- **Framework**: Flask 3
- **Deep Learning**: TensorFlow 2.21 / Keras (CNN)
- **ML**: scikit-learn (SVM, RF, KNN)
- **Image Processing**: OpenCV (headless), NumPy
- **Charts**: Chart.js 4.4 (CDN)
- **Port**: 5000

### Node.js Monorepo
- **Runtime**: Node.js 24, pnpm workspaces
- **API**: Express 5, Drizzle ORM, PostgreSQL
- **Validation**: Zod v4, drizzle-zod
- **Build**: esbuild (CJS)

## Where things live

```
dysgraphia-app/
├── app.py              # Flask backend — routes: /, /predict, /api/metrics, /api/cnn, /api/dashboard
├── train_ml.py         # Classical ML pipeline: augmentation + 12 features + SVM/RF/KNN
├── train_cnn.py        # CNN pipeline: Keras ImageDataGenerator + 3-block CNN + early stopping
├── data/normal/        # Normal handwriting images
├── data/dysgraphia/    # Dysgraphic handwriting images
├── models/             # best_ml_model.pkl, svm.pkl, random_forest.pkl, knn.pkl, cnn_model.h5
├── reports/            # ml_metrics.json, cnn_metrics.json
├── uploads/            # Temp uploaded images
├── templates/index.html  # SPA: Predict / Dashboard / Evaluate / About
└── static/             # style.css (dark theme), script.js (Chart.js, prediction UI)
```

## Architecture Decisions

- **Augmentation x8**: rotation ±15°, zoom 10%, shift 5px, brightness ±30, noise, flip — applied at training time (streaming, no disk writes)
- **CNN input**: 64×64 grayscale, 3 Conv2D+MaxPool+Dropout blocks, Dense(256), sigmoid output
- **Model selector**: UI pill lets user choose CNN vs best-ML at prediction time; Flask handles both paths
- **Lazy training**: models auto-train on first startup if pkl/h5 files are missing
- **Chart.js dashboard**: bar chart compares all 4 models; line chart shows CNN training history per epoch

## Product

- Upload handwriting image → get CNN or ML prediction (Normal / Dysgraphia) with confidence % and probability bars
- Dashboard: side-by-side comparison of CNN, SVM, Random Forest, KNN with accuracy/F1/confusion matrices
- Evaluate tab: detailed per-model metrics, feature importance, dataset stats
- Retrain CNN button triggers background training via `/api/retrain-cnn` endpoint

## User Preferences

- Comments in French (student-friendly), code in English
- Dark academic theme with cyan/indigo palette
- Keep architecture simple but professional (university final-year project)
- Dataset: ~455 real images (270 dysgraphia, 185 normal) placed in data/ folders

## Gotchas

- CNN training takes 5–15 min; a "Entraîner CNN" button in the Dashboard triggers `/api/retrain-cnn`
- `cnn_model.h5` uses legacy Keras format — load with `tf.keras.models.load_model()`
- Do NOT run `pnpm dev` at workspace root — use workflows only
- Flask is NOT an artifact (no artifact.toml); it runs standalone on port 5000 via "Start application" workflow
