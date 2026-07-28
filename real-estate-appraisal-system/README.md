# 🏠 Real Estate Appraisal System

An end-to-end machine learning project that predicts residential property prices from structural, location, and amenity features. Built as part of the **CodeTech IT Solutions Internship**.

The project covers the full ML lifecycle: data cleaning, exploratory data analysis, feature engineering, training and comparing multiple regression models, and deploying the best model behind an interactive Streamlit web app.

---

## Project Description

Estimating a fair market price for a home is a classic regression problem that depends on dozens of interacting factors — size, location, condition, amenities, and the local market. This project builds a reproducible pipeline that:

1. Cleans and validates a raw housing dataset
2. Explores relationships between features and price
3. Engineers additional predictive features
4. Trains and evaluates three regression models
5. Selects and saves the best-performing model
6. Serves live predictions through a simple web interface

---

## Features

- **Data Cleaning** — handles missing values, duplicate records, and price outliers (IQR method)
- **Exploratory Data Analysis (EDA)** — distribution plots, boxplots, and relationship charts
- **Correlation Matrix** — heatmap of numeric feature relationships with price
- **Feature Engineering** — derived features such as `rooms_total`, `lot_to_house_ratio`, `amenity_score`, and `is_new`
- **Train/Test Split** — 80/20 split with a fixed random seed for reproducibility
- **Multiple Models**
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Performance Comparison** using:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)
  - **R² Score**
- **Model Persistence** — best model (full preprocessing + model pipeline) saved with `pickle`/`joblib`
- **Streamlit Prediction Interface** — interactive UI to enter property details and get an instant price estimate

---

## Technologies Used

| Category            | Tools |
|---------------------|-------|
| Language             | Python 3.11 |
| Notebook             | Jupyter Notebook |
| Data manipulation    | pandas, numpy |
| Visualization        | matplotlib, seaborn |
| Machine Learning      | scikit-learn |
| Model serialization  | joblib (pickle-based) |
| Web app / deployment | Streamlit |

---

## Project Structure

```
real-estate-appraisal-system/
├── app.py                          # Streamlit prediction interface
├── generate_data.py                # Script used to generate the sample dataset
├── requirements.txt                # Python dependencies
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── housing_data.csv            # Sample dataset (1,500+ records)
├── notebooks/
│   └── Real_Estate_Appraisal.ipynb # Full walkthrough: EDA -> modeling -> evaluation
├── src/
│   ├── train.py                    # End-to-end training + evaluation pipeline
│   └── predict.py                  # CLI helper to test predictions from the saved model
├── models/
│   ├── best_model.pkl              # Saved best-performing model (pipeline)
│   ├── metadata.json               # Model results + feature schema used by app.py
│   └── model_comparison.csv        # MAE / RMSE / R2 for all trained models
└── screenshots/
    └── README.md                   # Placeholder / instructions for adding screenshots
```

---

## Installation

**Requirements:** Python 3.11

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/real-estate-appraisal-system.git
   cd real-estate-appraisal-system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Train and evaluate the models

Regenerates cleaned data, trains all three models, prints metrics, and saves the best model to `models/best_model.pkl`:

```bash
python src/train.py
```

### 2. Explore the full workflow in Jupyter

```bash
jupyter notebook notebooks/Real_Estate_Appraisal.ipynb
```

This notebook walks through data cleaning, EDA, the correlation matrix, feature engineering, model training, and evaluation with inline charts.

### 3. Launch the Streamlit prediction app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), fill in property details, and click **Estimate Price**.

### 4. Predict from the command line

```bash
python src/predict.py
```

Or import the helper in your own code:

```python
from src.predict import predict_price

price = predict_price({
    "square_footage": 2200, "bedrooms": 4, "bathrooms": 2.5,
    "lot_size": 9000, "house_age": 10, "distance_to_city_center": 6.5,
    "garage_spaces": 2, "has_pool": 0, "has_garden": 1,
    "school_rating": 8, "crime_rate": 2.5, "renovated": 1,
    "rooms_total": 6.5, "lot_to_house_ratio": 4.09, "amenity_score": 3,
    "is_new": 0, "neighborhood_quality": "High", "condition": "Good",
    "property_type": "Single Family",
})
print(price)
```

### (Optional) Regenerate the sample dataset

```bash
python generate_data.py
```

---

## Results

Trained and evaluated on an 80/20 train/test split of the sample dataset:

| Model               | MAE ($)   | RMSE ($)  | R² Score |
|---------------------|-----------|-----------|----------|
| Linear Regression   | 28,113    | 39,901    | 0.928    |
| Random Forest       | 32,204    | 42,329    | 0.919    |
| **Gradient Boosting** | **21,108**  | **28,331**  | **0.964**  |

**Best model: Gradient Boosting Regressor**, selected automatically by `src/train.py` based on the highest R² score on the held-out test set, and saved to `models/best_model.pkl`.

> Exact numbers will vary slightly depending on the dataset version and random seed used when you run the pipeline yourself.

---

## Screenshots

> Add screenshots after running the project locally — see `screenshots/README.md` for suggested filenames.

- **EDA — Price distribution:** `screenshots/eda_price_distribution.png`
- **Correlation matrix heatmap:** `screenshots/correlation_matrix.png`
- **Model comparison chart:** `screenshots/model_comparison.png`
- **Streamlit prediction interface:** `screenshots/streamlit_app.png`

---

## Future Improvements

- Add hyperparameter tuning (GridSearchCV / Optuna) for each model
- Incorporate real-world housing data (e.g. Zillow, Kaggle housing datasets) instead of the synthetic sample
- Add cross-validation instead of a single train/test split for more robust evaluation
- Support geospatial features (latitude/longitude, map-based input) in the Streamlit app
- Add model explainability (SHAP values) to show why a price was predicted
- Containerize the app with Docker and deploy to Streamlit Community Cloud / Render
- Add automated tests (pytest) for the data cleaning and feature engineering functions

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Built as part of the CodeTech IT Solutions Internship program.*
