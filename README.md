# End-to-End Machine Learning Prediction and Analysis System

This project is a modular and scalable machine learning application structured around four core pillars: Data Analysis, Model Training, FastAPI Backend, and Streamlit Frontend. It covers the entire lifecycle from raw data processing to delivering predictions to the end user.

---

## 🏗️ Project Architecture

The system is designed with a robust architecture consisting of four main components:
1. **Data Analysis:** Exploratory data analysis, detection of missing or inconsistent data, and statistical distribution reviews.
2. **Model Training:** Training of various classification algorithms, hyperparameter optimization, and performance evaluation.
3. **FastAPI Backend:** Serving the trained model and preprocessing pipeline with high performance via a RESTful API.
4. **Streamlit Frontend:** An interactive web interface where users can easily input parameters and receive predictions.

---

## 🔄 Data Preprocessing & Methodology

To maximize model performance and prevent data leakage, a **12-step validated preprocessing pipeline** has been integrated:
* **IQR Clipping:** Clipping steps applied to limit the negative impact of outliers on the model.
* **Yeo-Johnson Transformation:** Power transformation used to normalize data distributions.
* **ANOVA:** Statistical feature selection process.
* **SMOTE:** Synthetic minority oversampling technique for imbalanced datasets.
* **StandardScaler:** Standardizing features to have a mean of 0 and a variance of 1.
* *Categorical Encoding & Data Cleaning:* Consistent categorical transformation steps.

---

## 🤖 Models Used

The project evaluates and compares 6 different machine learning models:
* **Support Vector Machines (SVM)**
* **K-Nearest Neighbors (KNN)**
* **Decision Tree**
* **XGBoost**
* **Random Forest**
* **Logistic Regression**

---

## 🛠️ Technical Challenges & Solutions

* **NaN Input Error:** Missing data errors encountered during the inference phase were resolved by consistently applying the transformers from the training pipeline to the prediction inputs.
* **Endpoint Integration:** URL and endpoint mappings between FastAPI routers and the Streamlit interface were optimized to establish a stable communication infrastructure.

---

## 🚀 Getting Started

Follow the steps below to run the project locally:

### 1. Clone the Repository and Create a Virtual Environment
```bash
git clone <repository-url>
cd <project-folder>
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Start the FastAPI Backend Server
Navigate to the backend directory and start the server:
cd backend
uvicorn main:app --reload

### 4. Start the Streamlit Frontend
Open a new terminal and run:

streamlit run app.py
