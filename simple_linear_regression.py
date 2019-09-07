import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_predict = regressor.predict(x_test)

plt.style.use('seaborn')
plt.scatter(x_train, y_train, color="orange")
plt.plot(x_train, regressor.predict(x_train))
plt.xlabel('Year of Experience')
plt.ylabel('Salary')

plt.tight_layout()

plt.show()
