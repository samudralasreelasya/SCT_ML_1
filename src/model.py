import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/train.csv")

# Select only required features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X = df[features]
y = df[target]

# Handle missing values
X = X.fillna(X.mean())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)
print("Coefficients:", model.coef_)
# bar plot for Neighborhood distribution
plt.figure(figsize=(12,5))

df['Neighborhood'].value_counts().plot(kind='bar')

plt.xlabel("Neighborhood")
plt.ylabel("Number of Houses")
plt.title("Houses in Each Neighborhood")
plt.show()
#scatter plot for GrLivArea vs SalePrice
plt.figure(figsize=(8,5))

plt.scatter(df['GrLivArea'], df['SalePrice'])

plt.xlabel("Ground Living Area")
plt.ylabel("Sale Price")
plt.title("Living Area vs Sale Price")

plt.show()