from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# x = np.array([1, 2, 3, 4, 5])
# y = np.array([1, 4, 9, 16, 25])

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

model = LinearRegression()
model.fit(x, y)

print(model.coef_)
print(model.intercept_)
# model.predict(x).reshape(-1,1)-(x*model.coef_ + model.intercept_)
plt.scatter(x, y)
plt.plot(x, x*model.coef_ + model.intercept_, color='red')
plt.scatter(x,model.predict(x), color='blue')
plt.show()

# use polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
print(x_poly)
model = LinearRegression()
model.fit(x_poly, y)
print(model.coef_)
print(model.intercept_)
plt.scatter(x, y)
plt.plot(x, model.predict(x_poly), color='red')
plt.show()

# (x_poly*model.coef_ + model.intercept_).sum(axis=1)

