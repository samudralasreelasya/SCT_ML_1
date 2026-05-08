# SCT_ML_1 - House Price Prediction

## 📌 Task Objective
Build a Linear Regression model to predict house prices based on:
- Square footage (GrLivArea)
- Number of bedrooms (BedroomAbvGr)
- Number of bathrooms (FullBath)

## 📊 Dataset
Housing dataset containing multiple features.  
For this task, only selected features were used.

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn

## 🧠 Model Used
Linear Regression

## 🚀 How to Run
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python src/model.py

## 📈 Output
- Predicts house prices
- Displays Mean Squared Error (MSE)
- Shows feature coefficients

## 📂 Project Structure
SCT_ML_1/
 ┣ data/
 ┃ ┣ train.csv
 ┣ src/
 ┃ ┣ model.py
 ┣ README.md
 ┣ requirements.txt