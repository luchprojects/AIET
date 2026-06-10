# NASA training data

Place NASA Exoplanet Archive exports here (not committed by default due to size):

- `exoplanets.csv` — PS composite table
- `stellar_hosts.csv` — stellar parameters

Download from https://exoplanetarchive.ipac.caltech.edu/

Retrain the habitability model:

```bash
python ml_calibration/train_habitability.py
```

Outputs `../hab_xgb.json` and `../training_summary.json`.
