# WattWise — Household Energy Consumption Predictor

An end-to-end machine learning project that predicts household active power (`Energy_Consumption_kW`) from meter readings and time context. It includes a trained model and a four-page Streamlit website for the dashboard, data exploration, single and batch predictions, and model interpretation.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train and compare models:

   ```bash
   python train_model.py
   ```

   This creates `artifacts/energy_model.joblib` and `artifacts/metrics.json`. Training uses a chronological 80/20 split and expanding-window time-series cross-validation to choose among Ridge, Lasso, Decision Tree, and Random Forest.

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push this project folder, including the workbook and generated `artifacts/` files, to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and set the main file to `app.py`.
3. Deploy. The app reads the saved model and metrics, and includes `requirements.txt` for dependency installation.

If you change the dataset or training code, run `python train_model.py` again before deploying the updated model artifacts.

## Project workflow

- **Target:** household active power in kW.
- **Features:** reactive power, voltage, global intensity, three sub-meter readings, weekday, and calendar/time features derived from the timestamp.
- **Model choice:** select the candidate with the lowest average cross-validation RMSE on training-period folds.
- **Final evaluation:** MAE, RMSE, and R² on the newest 20% of observations, kept out of model selection.
- **Deployment:** four-page Streamlit site with an energy dashboard, filterable data explorer, single and CSV batch predictions, model comparison, holdout diagnostics, coefficient chart, and project methodology.

The measured electrical inputs are strongly related to active power, so the model is best understood as a meter-reading based estimator. It does not claim to forecast future consumption from weather, occupancy, or appliance schedules, which are not included in this dataset.

