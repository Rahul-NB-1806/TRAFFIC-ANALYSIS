# Smart Traffic Dashboard

Traffic data analysis and forecasting dashboard built with Dash, Flask, scikit-learn, and SQLAlchemy.

## Features

- **Upload CSV** — Upload traffic data files (columns: `date, time, location, two_wheeler, four_wheeler, heavy_vehicle, emergency_vehicle`)
- **Home** — Vehicle count time series and vehicle type distribution charts
- **Forecast** — Predict future traffic volume with Linear Regression or Random Forest, with train/test evaluation metrics
- **Analysis** — Filter by location/time range, view radar charts and semi-circle gauges

## Project Structure

```
├── dash_traffic_predictor.py    # Main Dash application
├── backend/
│   ├── config.py                # Pydantic-based settings (env configurable)
│   ├── logging_config.py        # Structured JSON logging
│   ├── database.py              # SQLAlchemy ORM (SQLite)
│   ├── data_pipeline.py         # CSV ingestion, validation (Pydantic), ETL
│   └── utils.py                 # Timestamp normalization, helpers
├── ml/
│   ├── features.py              # Time features, rolling windows, lag features
│   ├── models.py                # LinearRegression, RandomForest, evaluation
│   └── registry.py              # Model save/load/versioning
├── data/
│   ├── uploads/                 # Uploaded CSV files
│   └── archive/                 # Deduplicated backups
├── models/                      # Saved model pickles + metadata
├── data/traffic.db              # SQLite database (auto-created)
├── requirements.txt
└── .env                         # Environment variables (optional)
```

## Quick Start

```bash
pip install -r requirements.txt
python dash_traffic_predictor.py
```

Open http://127.0.0.1:8050

## Configuration

Set via environment variables with `TRAFFIC_` prefix or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `TRAFFIC_DATABASE_URL` | `sqlite:///data/traffic.db` | Database connection string |
| `TRAFFIC_UPLOAD_DIR` | `./data/uploads` | CSV upload directory |
| `TRAFFIC_DEFAULT_TZ` | `Asia/Kolkata` | Timezone for timestamps |
| `TRAFFIC_LOG_LEVEL` | `INFO` | Logging level |
| `TRAFFIC_DEBUG` | `false` | Enable debug mode |

## Data Format

CSV files must have these columns:
- `date` + `time` (two columns) **or** `timestamp` (single column)
- `location` — intersection/junction name
- `two_wheeler`, `four_wheeler`, `heavy_vehicle`, `emergency_vehicle` — vehicle counts
