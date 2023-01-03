import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plot

data = pd.read_csv('Position_Salaries.csv')

def lineal(x_name,y_name,datos):
    x = datos[x_name].values.reshape(-1,1)
    y = datos[y_name].values.reshape(-1,1)
    plot.scatter(x,y)
    print(datos.keys())

    model = LinearRegression()
    model.fit(x,y)
    y_pred = model.predict(x)
    plot.plot(x,y_pred,color='r')
    plot.show()

    rmse = np.sqrt(mean_squared_error(x,y_pred))
    r2 = r2_score(y,y_pred)
    print('RMSE',rmse)
    print('R2',r2)

#lineal('Level','Salary',data)


def polinomial(datos, x_name,y_name,degree):
    x = datos[x_name].values.reshape(-1,1)
    y = datos[y_name].values.reshape(-1,1)
    poly = PolynomialFeatures(degree = degree,include_bias =False)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly,y)
    y_pred = model.predict(x_poly)
    plot.scatter(x,y)
    plot.plot(x,y_pred,color='r')
    plot.show()
    rmse = np.sqrt(mean_squared_error(x,y_pred))
    r2 = r2_score(y,y_pred)
    print('RMSE',rmse)
    print('R2',r2)

polinomial(data,'Level','Salary',3)