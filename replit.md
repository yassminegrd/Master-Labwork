# Workspace

## Overview

pnpm workspace monorepo using TypeScript, plus a standalone Python/Flask ML application for Dysgraphia Detection.

## Stack

### Node.js / TypeScript (monorepo)
- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

### Python Flask (Dysgraphia Detection)
- **Location**: `dysgraphia-app/`
- **Framework**: Flask 3
- **ML Models**: scikit-learn (SVM, Random Forest, KNN)
- **Image Processing**: OpenCV (headless)
- **Port**: 5000
- **Workflow**: "Start application"

## Key Commands

### Node.js
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)

### Python (Dysgraphia App)
- `cd dysgraphia-app && python train.py` — retrain all models
- `cd dysgraphia-app && python evaluate.py` — print evaluation report
- `cd dysgraphia-app && python app.py` — run Flask app (port 5000)

## Dysgraphia App Structure

```
dysgraphia-app/
├── app.py              # Flask backend (routes: /, /predict, /evaluate, /api/metrics)
├── train.py            # Training pipeline (SVM, RF, KNN) + synthetic data generator
├── evaluate.py         # Standalone evaluation / console report
├── model.pkl           # Saved best model (auto-generated)
├── metrics.json        # Evaluation metrics (auto-generated)
├── requirements.txt    # Python dependencies
├── data/
│   ├── normal/         # Normal handwriting images (synthetic or real)
│   └── dysgraphia/     # Dysgraphic handwriting images (synthetic or real)
├── uploads/            # Temporary uploaded images for prediction
├── templates/
│   └── index.html      # Single-page frontend (Predict + Evaluation + About)
└── static/
    ├── style.css       # Dark academic theme
    └── script.js       # Drag-drop upload, prediction, metrics rendering
```

## Features Extracted (12 total)
Mean Intensity, Std Intensity, Dark Pixel Ratio, Edge Density, Contour Count,
Mean Contour Area, Mean Contour Perimeter, Horizontal Variance, Vertical Variance,
Laplacian Variance, Pixel Skewness, Ink Irregularity

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
