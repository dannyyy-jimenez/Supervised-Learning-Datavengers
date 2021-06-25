import pandas as pd
import numpy as np
from datetime import datetime 

original_df_train = pd.read_csv('data/churn_train.csv')
original_df_test = pd.read_csv('data/churn_test.csv')

df_train = original_df_train.copy()
df_test = original_df_test.copy()


df_train = df_train[['avg_dist', 'avg_rating_of_driver',
       'city', 'last_trip_date', 'phone', 'surge_pct',
       'trips_in_first_30_days', 'luxury_car_user', 'weekday_pct']]


dummies = pd.get_dummies(df_train['city'])
df_train = df_train.join(dummies)


df_train_dummies = df_train[['avg_dist', 'avg_rating_of_driver', 'last_trip_date', 'phone',
       'surge_pct', 'trips_in_first_30_days', 'luxury_car_user', 'weekday_pct',
       'Astapor', "King's Landing", 'Winterfell']]



df_train_dummies['last_trip_date'] = pd.to_datetime(df_train_dummies['last_trip_date'])

dummies_phone = pd.get_dummies(df_train_dummies['phone'])
df_train_dummies = df_train_dummies.join(dummies_phone)
df_train_dummies.drop(columns=['phone'], inplace=True)




X_train = df_train_dummies

y_train = (X_train['last_trip_date'] > datetime(2014, 6, 1)).astype(int) 
print(X_train)
print(y_train)
print(np.count_nonzero(y_train))