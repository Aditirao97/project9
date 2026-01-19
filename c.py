import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,r2_score

data =pd.read_csv('data.csv')
print(data.head(2))

data.drop(columns=['date','street','statezip', 'country','waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement','yr_built', 'yr_renovated', 'city', 'sqft_living', 'sqft_lot', 'floors'],inplace=True)
lb=LabelEncoder()
print(data.info())
print(data.columns)
x=data[['bedrooms', 'bathrooms']]
y=data[['price']]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

pipeline=Pipeline([
    ('model',RandomForestRegressor(n_estimators=100, random_state=42))
])
print(X_test.head(2))
pipeline.fit(X_train,y_train)

predi=pipeline.predict(X_test)

rmse=root_mean_squared_error(y_test,predi)
print('rmse',rmse)
r2=r2_score(y_test,predi)
print(r2)











