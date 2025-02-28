import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error,r2_score
def calculate_seconds(x):
    temp=x.split(":")
    return int(temp[0]) * 3600 + int(temp[1])*60 + int(temp[2])
def generateFeatures(df):
    features = ['day','dayofweek','month','quarter','year','dayofyear','weekofyear']
    for col in features:
        df[col] = getattr(df['Date'].dt,col) * 1
def train():
    df= pd.read_csv("./app_usage_data.csv")
    df['DateTime']= pd.to_datetime(df['Date'] +" " +df['Time'],format='%m/%d/%Y %H:%M:%S')
    df['Date']= pd.to_datetime(df['Date'],format='%m/%d/%Y')
    df=df.sort_values(['DateTime'])
    redundant_activities=['Screen on (unlocked)','Screen off (locked)','Screen on (locked)', 'Screen off','Permission controller','System UI','Package installer',
    'Device shutdown','Call Management']
    df=df[~df['App'].isin(redundant_activities)]
    df['TotalSeconds']=df['Duration'].apply(lambda x:calculate_seconds(x))
    df=df.groupby(['Date','App']).sum().reset_index()
    df['TotalMinutes'] = df['TotalSeconds'] // 60
    generateFeatures(df)
    df.drop(["Date",'TotalSeconds','year'],axis=1, inplace=True)
    df=pd.get_dummies(df,drop_first=True)
    train_length=int(0.90*df.shape[0])
    train=df[0:train_length]
    test=df[train_length:].reset_index(drop=True)
    y_test=test['TotalMinutes']
    X_test=test.drop(['TotalMinutes'],axis=1)
    y=train['TotalMinutes']
    X=train.drop(['TotalMinutes'],axis=1)
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.15, random_state=42)
    rf=RandomForestRegressor()
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

    #print(random_grid)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs=1)
    rf_random.fit(X_train.values,y_train.values)
    best_rf_model= rf_random.best_estimator_
    y_pred = best_rf_model.predict(X_test.values)
    print("r2 score is:",r2_score(y_test.values,y_pred))
    print("mean squared error  is:",mean_squared_error(y_test.values,y_pred))
    print("root mean squared error is:",np.sqrt(mean_squared_error(y_test.values, y_pred)))
    print("training completed!")

if __name__=="__main__":
    train()
