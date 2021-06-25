import pandas as pd
import numpy as np
from datetime import datetime 
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_data(csv_file_name):
    '''
    Loads entries of a csv file into a dataframe.
    Parameters
    ----------
    csv_file_name : string
        The name of the csv file you want to load in.
    Returns
    -------
    dataframe_
        A dataframe made from the values in the csv file.
    '''
    file_name = os.path.join('.', 'data', ''.join([csv_file_name, '.csv']))
    dataframe_ = pd.read_csv(file_name)
    return dataframe_



df_train = load_data('churn_train')
df_test = load_data('churn_test')


def clean_columns(data):
    df_train = load_data('churn_train')
    dummies = pd.get_dummies(df_train['city'])
    df_train = df_train.join(dummies)
    df_train['last_trip_date'] = pd.to_datetime(df_train['last_trip_date'])         
    dummies_phone = pd.get_dummies(df_train['phone'])
    df_train = df_train.join(dummies_phone)
    df_train['avg_rating_of_driver'] = df_train['avg_rating_of_driver'].fillna(0)
    df_train['luxury_car_user'] = df_train['luxury_car_user'].astype(int)
    df_train.drop(columns=['avg_surge'], inplace=True)
    df_train.drop(columns=['avg_rating_by_driver'], inplace=True)
    df_train.drop(columns=['city'], inplace=True)
    df_train.drop(columns=['phone'], inplace=True)
    df_train.drop(columns=['iPhone'], inplace=True)
    
    X = df_train
    X['Churn'] = (X['last_trip_date'] > datetime(2014, 6, 1)).astype(int) 
    y= (X['last_trip_date'] > datetime(2014, 6, 1)).astype(int) 
    
    X.drop(columns=['last_trip_date'], inplace=True)


    return X, y

print(clean_columns(load_data('churn_train')))


