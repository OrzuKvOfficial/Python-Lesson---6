# NumPy va Pandas bilan ishlash
import numpy as np
import pandas as pd

# Ma'lumotlarni vizualizatsiya qilish
import matplotlib.pyplot as plt
import seaborn as sns

# Ma'lumotlarni bo'lish
from sklearn.model_selection import train_test_split

# Model yaratish
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Ma'lumotlarni o'qish
data = pd.read_csv('data.csv')

# Ma'lumotlarni tahlil qilish va tayyorlash
X = data[['feature1', 'feature2']]
y = data['target']

# Ma'lumotlarni o'quv va test to'plamlariga ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelni yaratish va o'qitish
model = LinearRegression()
model.fit(X_train, y_train)

# Modelni baholash
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"Mean Squared Error: {mse}")

# Natijalarni vizualizatsiya qilish
plt.scatter(y_test, predictions)
plt.xlabel('Haqiqiy qiymatlar')
plt.ylabel('Bashorat qilingan qiymatlar')
plt.show()
