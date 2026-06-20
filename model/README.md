# Model folder

Generated automatically by running `train_model.py`:

- `churn_model.pkl` — trained Random Forest classifier
- `encoders.pkl` — fitted LabelEncoders for each categorical column
- `feature_names.pkl` — exact column order used at training time

These files are git-ignored (combined ~18MB). Run `python train_model.py`
once after adding `data/cleaned_churn_data.csv` to regenerate them.
