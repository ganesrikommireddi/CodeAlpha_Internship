# 🎓 Student Performance Tracker

A complete, end-to-end machine learning project that predicts a student's final exam score from study habits, attendance, and lifestyle factors — built as **Project 1** for the **CodeTech IT Solutions Internship**.

The project covers the full data science lifecycle: data cleaning, exploratory data analysis, feature engineering, multi-model training and evaluation, and deployment through an interactive Streamlit prediction app.

---

## 📋 Project Description

Educational institutions increasingly want to identify students who may need academic support *before* their grades slip. This project builds a regression pipeline that takes in a student's study hours, attendance, previous grades, sleep habits, extracurricular load, and support environment, and predicts their **final exam score (0–100)**.

Three regression models are trained and compared, the best-performing model is serialized with `pickle`, and a Streamlit web app is provided so anyone can enter a student's details and get an instant prediction.

---

## ✨ Features

- **Data Cleaning** — missing value imputation, whitespace trimming, duplicate removal
- **Exploratory Data Analysis (EDA)** — distributions, scatter plots, group comparisons
- **Correlation Matrix** — heatmap of numeric feature relationships
- **Feature Engineering** — derived ratios, ordinal/one-hot encoding of categorical variables
- **Train/Test Split** — reproducible 80/20 split
- **Multiple ML Models**
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Performance Comparison** using:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
- **Best Model Persistence** — automatically saved via `pickle`
- **Streamlit Prediction Interface** — interactive web app for live predictions

---

## 🛠️ Technologies Used

| Category            | Tools / Libraries                                   |
|----------------------|------------------------------------------------------|
| Language             | Python 3.11                                           |
| Data Handling        | pandas, numpy                                        |
| Visualization        | matplotlib, seaborn                                  |
| Machine Learning     | scikit-learn (Linear Regression, Random Forest, Gradient Boosting) |
| Notebook Environment | Jupyter Notebook                                     |
| Model Persistence    | pickle                                               |
| Web App / Deployment | Streamlit                                            |

---

## 📁 Folder Structure

```
Student-Performance-Tracker/
├── data/
│   └── student_performance.csv        # Sample dataset (500 students)
├── notebooks/
│   └── Student_Performance_Analysis.ipynb   # Full EDA + modeling notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Cleaning & feature engineering functions
│   ├── train_models.py                # Trains, evaluates, and pickles the best model
│   └── generate_dataset.py            # Script used to generate the sample dataset
├── models/
│   ├── best_model.pkl                 # Serialized best-performing model
│   └── model_comparison.csv           # Saved metrics table
├── screenshots/
│   └── README.md                      # Placeholders for app/notebook screenshots
├── app.py                             # Streamlit prediction interface
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Student-Performance-Tracker.git
   cd Student-Performance-Tracker
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### 1. Explore the analysis in Jupyter Notebook
```bash
jupyter notebook notebooks/Student_Performance_Analysis.ipynb
```
Run all cells to reproduce the cleaning, EDA, correlation matrix, feature engineering, model training, and evaluation steps.

### 2. Retrain models from the command line
```bash
python src/train_models.py
```
This regenerates `models/best_model.pkl` and `models/model_comparison.csv`.

### 3. Launch the Streamlit prediction app
```bash
streamlit run app.py
```
Then open the local URL shown in your terminal (typically `http://localhost:8501`) and enter a student's details to get a predicted final score.

---

## 📊 Results

Models were trained on an 80/20 train/test split of the sample dataset (500 students) and evaluated on the held-out test set:

| Model                        | MAE   | RMSE  | R² Score |
|-------------------------------|-------|-------|----------|
| **Linear Regression** ⭐       | 4.50  | 6.11  | **0.377** |
| Random Forest Regressor        | 4.80  | 6.37  | 0.322    |
| Gradient Boosting Regressor    | 4.78  | 6.55  | 0.283    |

⭐ **Linear Regression** achieved the highest R² score on this dataset and was automatically selected and saved as `models/best_model.pkl`.

> **Note:** The included dataset is synthetically generated for demonstration purposes. Results will vary (and typically improve) when the pipeline is used with real, larger student performance datasets. The modular design of `src/data_preprocessing.py` and `src/train_models.py` makes it straightforward to swap in your own data.

---

## 🖼️ Screenshots

> Replace the placeholders below with actual screenshots after running the notebook and Streamlit app.

| Description                     | Screenshot                              |
|----------------------------------|------------------------------------------|
| Exploratory Data Analysis        | `screenshots/eda_placeholder.png`        |
| Correlation Matrix Heatmap       | `screenshots/correlation_placeholder.png`|
| Model Comparison Chart           | `screenshots/comparison_placeholder.png` |
| Streamlit Prediction Interface   | `screenshots/streamlit_placeholder.png`  |

---

## 🚀 Future Improvements

- Add hyperparameter tuning (GridSearchCV / Optuna) for each model
- Incorporate cross-validation instead of a single train/test split
- Add more advanced models (XGBoost, LightGBM, Neural Networks)
- Support uploading a custom CSV dataset directly in the Streamlit app
- Add SHAP-based model explainability to the prediction interface
- Deploy the Streamlit app to Streamlit Community Cloud / Docker
- Add automated unit tests (pytest) for the preprocessing pipeline
- Expand the dataset with real-world, anonymized student records

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Built as part of the CodeTech IT Solutions Internship Program.*
