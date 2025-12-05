# 🚗 In-Vehicle Coupon Recommendation System

## 📌 Project Overview
This project applies Machine Learning to predict whether a driver will accept a coupon recommended to them while driving. The solution includes a full data analysis pipeline, machine learning modeling, and an interactive deployment app.

**Key Results:**
- **Best Model:** Random Forest Classifier
- **Accuracy:** ~74% (Outperforming Logistic Regression and Decision Trees)
- **Key Drivers:** Coupon expiration time, Coffee House visit frequency, and Coupon type.

## 📂 Project Structure
- `ML_cw.ipynb`: The Jupyter Notebook containing Data Cleaning, EDA, Feature Engineering, and Model Training.
- `app.py`: The Streamlit application for live predictions (Deployment).
- `coupon_model.pkl`: The trained Random Forest model (saved via pickle).
- `requirements.txt`: List of Python libraries required to run the project.
- `in-vehicle-coupon-recommendation.csv`: The dataset (Source: UCI Machine Learning Repository).

Deployed on:
https://ml-cwcoupon-rjygpm3meorijvu2qhmxvp.streamlit.app/

Install dependencies:
pip install -r requirements.txt

Local running option:
Run the Streamlit App:
streamlit run app.py

📊 Dataset Origin

The dataset was sourced from the UCI Machine Learning Repository (In-Vehicle Coupon Recommendation). It contains driver scenarios including destination, current time, weather, and passenger attributes.
🛠️ Technologies Used

    Python: Core programming language.

    Pandas & NumPy: Data manipulation.

    Scikit-Learn: Machine Learning (Logistic Regression, Decision Tree, Random Forest).

    Streamlit: Web application framework for model deployment.

    Matplotlib & Seaborn: Data visualization.
