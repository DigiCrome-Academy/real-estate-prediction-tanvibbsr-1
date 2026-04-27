from src.ensemble import save_model
from sklearn.linear_model import LinearRegression

# Replace this with your real dataset later
X = [[1], [2], [3], [4]]
y = [100, 200, 300, 400]

model = LinearRegression()
model.fit(X, y)

# Save model to models folder
save_model(model, 'models/best_model.joblib')

print("Model saved!")