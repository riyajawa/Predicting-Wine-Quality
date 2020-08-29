import numpy as np
import pandas as pd
df= pd.read_csv('wineQualityWhites.csv',sep=',')

X=df[list(df.columns)[:-1]]
Y=df['quality']

X=np.append(arr=np.ones((X.shape[0],1)), values =X,axis=1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11]]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

#Scalint the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#building a model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#prediction
predictions = regressor.predict(X_test)

#evaulating using metrics
from sklearn.metrics import r2_score
r2_score(Y_test,predictions)

#backwards elimantion
import statsmodels.api as sm
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
regressor_OLS= sm.OLS(endog =Y, exog=X_opt).fit()
regressor_OLS.summary()

#Displaying the results
import matplotlib.pylab as plt
plt.scatter(Y_test,predictions,c='g')
plt.xlabel('True Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted Quality Against True Quality')
plt.show()

