import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#%matplotlib inline

hs=pd.read_csv('housePrice.csv')
#print(hs.describe())
#print(hs.head())
x_data,y_data=(hs['Room'].values,hs['Price'].values)
plt.plot(x_data,y_data, 'ro')
#plt.plot.ylable('Area')
#plt.plot.xlable('price')
plt.show()
#print(sns.histplot(hs("Price")))
msk=np.random.rand(len(hs))<0.8
train=hs[msk]
test=hs[~msk]
train_x=np.asanyarray(train[['Room']])
train_y=np.asanyarray(train[['Price']])
test_x=np.asanyarray(test[['Room']])
test_y=np.asanyarray(test[['Price']])
poly=PolynomialFeatures(degree=2)
train_x_poly=poly.fit_transform(train_x)

hs=linear_model.LinearRegression()
train_y_=hs.fit(train_x_poly,train_y)
#y=teta+x1=+x2
print('coefficients:',hs.coef_)
print('Intercept:',hs.intercept_)

plt.scatter(train.Room ,train.Price, color='blue')
xx=np.arange(0.0,10.0,0.1)
yy=hs.intercept_[0]+hs.coef_[0][1]*xx+hs.coef_[0][2]*np.power(xx,2)
plt.plot(xx,yy,'_r')
plt.xlabel('room')
plt.ylabel('price')  
plt.show()

from sklearn.metrics import r2_score

test_x_poly=poly.fit_transform(test_x)
test_y_=hs.predict(test_x_poly)
print("mean absolute error:%.2f" %np.mean(np.mean(np.absolute(test_y_ - test_y))))
print("residual sum of squares(MSE):%.2F"%np.mean((test_y_ - test_y)**2))
print("r2_score:%2f"% r2_score(test_y,test_y_))