import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

mtcars = pd.read_csv("28.Oct/mtcars.csv")

X = pd.DataFrame(mtcars['wt'])
Y = pd.DataFrame(mtcars['mpg'])

model = LinearRegression()
model.fit(X, Y)

k = model.coef_[0][0]
b = model.intercept_[0]

plt.scatter(X, Y, color="blue", label="Data Points")
plt.plot(X, model.predict(X), color="red", label=f"Y = {k:.2f}X + {b:.2f}")
plt.xlabel("Vehicle Weight (wt)")
plt.ylabel("Vehicle Gas Mileage (mpg)")
plt.title("Linear Regression: Vehicle Gas Mileage vs Weight")
plt.legend()
plt.show()

print("Slope (k):", k)
print("Intercept (b):", b)
