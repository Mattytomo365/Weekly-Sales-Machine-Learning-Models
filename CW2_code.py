#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:49:24 2025

@author: matthewtomlinson
"""

import pandas as pd #importing all the different modules and packages and renaming them for ease of use
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingRegressor



'''
Loading the datasets
'''
train = pd.read_csv('./train.csv') #specifying the location of the train dataset so the program can read it
test = pd.read_csv('./test.csv')#specifying the location of the test dataset so the program can read it
stores = pd.read_csv('./stores.csv')#specifying the location of the stores dataset so the program can read it
features = pd.read_csv('./features.csv')#specifying the location of the features dataset so the program can read it

print("Datasets loaded")



'''
Merging the datasets
'''
train_merged = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left') #Centralising all the data
train_merged = train_merged.merge(stores, on=['Store'], how='left')

test_merged = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
test_merged = test_merged.merge(stores, on=['Store'], how='left')

print("Datasets merged")
print(train_merged)



'''
Categorical feature processing
'''
l1 = LabelEncoder() #calling the LabelEncoder function for each varaible created
l2 = LabelEncoder()
l3 = LabelEncoder()
l4 = LabelEncoder()


l1.fit(train_merged['IsHoliday']) #assigning a column of the dataset to each variable
l2.fit(train_merged['Type'])

l3.fit(test_merged['IsHoliday'])
l4.fit(test_merged['Type'])

train_merged.IsHoliday = l1.transform(train_merged.IsHoliday) #converting all values in the column to numerical values
train_merged.Type = l2.transform(train_merged.Type)

test_merged.IsHoliday = l3.transform(test_merged.IsHoliday)
test_merged.Type = l4.transform(test_merged.Type)

print("Categorical feature processing completed")



'''
Handle missing values
'''
train_merged['MarkDown1'] = train_merged['MarkDown1'].fillna(train_merged['MarkDown1'].mean()) #filling missing values with the mean of each column
train_merged['MarkDown2'] = train_merged['MarkDown2'].fillna(train_merged['MarkDown2'].mean())
train_merged['MarkDown3'] = train_merged['MarkDown3'].fillna(train_merged['MarkDown3'].mean())
train_merged['MarkDown4'] = train_merged['MarkDown4'].fillna(train_merged['MarkDown4'].mean())
train_merged['MarkDown5'] = train_merged['MarkDown5'].fillna(train_merged['MarkDown5'].mean())
train_merged['CPI'] = train_merged['CPI'].fillna(train_merged['CPI'].mean())
train_merged['Unemployment'] = train_merged['Unemployment'].fillna(train_merged['Unemployment'].mean())

test_merged['MarkDown1'] = test_merged['MarkDown1'].fillna(test_merged['MarkDown1'].mean())
test_merged['MarkDown2'] = test_merged['MarkDown2'].fillna(test_merged['MarkDown2'].mean())
test_merged['MarkDown3'] = test_merged['MarkDown3'].fillna(test_merged['MarkDown3'].mean())
test_merged['MarkDown4'] = test_merged['MarkDown4'].fillna(test_merged['MarkDown4'].mean())
test_merged['MarkDown5'] = test_merged['MarkDown5'].fillna(test_merged['MarkDown5'].mean())
test_merged['CPI'] = test_merged['CPI'].fillna(test_merged['CPI'].mean())
test_merged['Unemployment'] = test_merged['Unemployment'].fillna(test_merged['Unemployment'].mean())

print("Missing values handled")
print(train_merged)



'''
Feature engineering
'''
train_merged['Date'] = pd.to_datetime(train_merged['Date'], errors='coerce') #Converting the Date column type to datetime format

def add_date_features(df): #Splitting the Date column into Year, Month, and Week features
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  #Ensure 'Date' is datetime
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    return df

def add_holiday_features(df): #adds holiday specific features, improving the predictive accuracy of the models
    #Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    super_bowl_dates = ['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08']
    labor_day_dates = ['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06']
    thanksgiving_dates = ['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29']
    christmas_dates = ['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27']
    
    #Convert holiday lists to datetime, if they aren't already
    super_bowl_dates = pd.to_datetime(super_bowl_dates)
    labor_day_dates = pd.to_datetime(labor_day_dates)
    thanksgiving_dates = pd.to_datetime(thanksgiving_dates)
    christmas_dates = pd.to_datetime(christmas_dates)
    
    #Add holiday flag columns
    df['Is_SuperBowl'] = df['Date'].isin(super_bowl_dates).astype(int)
    df['Is_LaborDay'] = df['Date'].isin(labor_day_dates).astype(int)
    df['Is_Thanksgiving'] = df['Date'].isin(thanksgiving_dates).astype(int)
    df['Is_Christmas'] = df['Date'].isin(christmas_dates).astype(int)
    return df

train_merged = add_date_features(train_merged)
test_merged = add_date_features(test_merged)

train_merged = add_holiday_features(train_merged)
test_merged = add_holiday_features(test_merged)

print(train_merged)



'''
Splitting the data
'''
X = train_merged.drop(['Weekly_Sales', 'Date'], axis = 1)
y = train_merged['Weekly_Sales']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

'''
Data is splitted into a validation dataset (X_val), 
and an array of Weekly_Sales values from the train dataset (y_val)
'''

print("Data splitted")



'''
Weighting for holidays
'''
def weighted_mae(y_val, y_pred, is_holiday):
    """
    Custom weighted mean absolute error that gives 5x weight to holiday weeks
    y_val: Actual sales values
    y_pred: Predicted sales values
    is_holiday: Array or Series indicating holiday weeks (1 if holiday, 0 otherwise)
    return: Weighted MAE
    """
    weights = 1 + 4 * is_holiday  #5x weight for holidays
    weighted_errors = weights * np.abs(y_val - y_pred)
    return np.mean(weighted_errors)



'''
Training/testing the random forest model
'''
random_forest = RandomForestRegressor(n_estimators = 10, random_state = 42)
random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_val) #Takes input features from X_val dataset and predicts their corresponding Weekly_Sales value (y_pred)
#y_pred is a numpy array of predicted values of Weekly_Sales for each row in the X_val dataset

print("Random Forest Regressor (RF) predicted values: ", y_pred_rf) #outputs the initial predicted values of RF



'''
Training/testing the gradient boosting model
'''
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # Initializing GBR
gbr.fit(X_train, y_train)  #Train the model

#Make predictions
y_pred_gbr = gbr.predict(X_val)  #Predict on the validation set

print("Gradient Boosting Regressor (GBR) predicted values: ", y_pred_gbr)



'''
Evaluate random forest model performance
'''
#Calculates mean absolute error by comparing the predicted values (y_pred) with the actual values (y_val)
mae_weighted_rf = weighted_mae(y_val, y_pred_rf, X_val['IsHoliday']) #Pass the holiday flag
 
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf)) #Calculates root mean squared error
r2_rf = r2_score(y_val, y_pred_rf) #Calculates r^2 score

#Generation of holiday and non-holiday specific MAE's
holiday_mask = X_val['IsHoliday'] == 1
mae_holiday_rf = mean_absolute_error(y_val[holiday_mask], y_pred_rf[holiday_mask]) #holiday MAE
mae_non_holiday_rf = mean_absolute_error(y_val[~holiday_mask], y_pred_rf[~holiday_mask]) #non-holiday MAE

#Ouputs all the evaluation metrics of RF
print("Initial Random Forest Regressor performance:")
print("Weighted mean absolute error (WMAE): ", mae_weighted_rf) 
print("Holiday mean absolute error (MAE): ", mae_holiday_rf) 
print("Non-holiday mean absolute error (MAE): ", mae_non_holiday_rf)
print("Root mean squared error (RMSE): ", rmse_rf)
print("R^2 score: ", r2_rf)



'''
Evaluate gradient boosting model performance
'''
mae_weighted_gbr = weighted_mae(y_val, y_pred_gbr, X_val['IsHoliday'])  #Use custom weighted MAE

rmse_gbr = np.sqrt(mean_squared_error(y_val, y_pred_gbr))
r2_gbr = r2_score(y_val, y_pred_gbr)

holiday_mask = X_val['IsHoliday'] == 1
mae_holiday_gbr = mean_absolute_error(y_val[holiday_mask], y_pred_gbr[holiday_mask])
mae_non_holiday_gbr = mean_absolute_error(y_val[~holiday_mask], y_pred_gbr[~holiday_mask])

#Ouputs all the evaluation metrics of GBR
print("Initial Gradient Boosting Regressor performance:")
print("Weighted mean absolute error (WMAE): ", mae_weighted_gbr)
print("Holiday mean absolute error (MAE): ", mae_holiday_gbr) 
print("Non-holiday mean absolute error (MAE): ", mae_non_holiday_gbr)
print("Root mean squared error (RMSE): ", rmse_gbr)
print("R^2 Score: ", r2_gbr)



'''
Hyperparameter tuning for random forest model
'''
param_grid_rf = { #Dictionary specifying the hyperparameters to tune
    'n_estimators' : [10, 20, 50], #number of trees
    'max_depth' : [5, 10, 20], #maximum depth of the trees
    'min_samples_split' : [2, 5, 10] #minimum number of samples required to split an internal node
}

weighted_mae_scorer_rf = make_scorer( #Creating a custom scorer, prioritising the holiday weeks
    lambda y_true, y_pred_rf: weighted_mae(y_true, y_pred_rf, X_train.loc[y_true.index, 'IsHoliday']),
    greater_is_better=False #Since GridSearchCV minimizes the score
)

grid_search_rf = GridSearchCV(estimator = RandomForestRegressor(random_state = 42),
                           param_grid = param_grid_rf,
                           scoring = weighted_mae_scorer_rf, #Uses custom scorer, allowing model to perform better during holiday periods
                           cv = 2, #2-fold cross-validation
                           verbose = 2, #Verbose of 2 means more information is displayed
                           n_jobs = -1 #Uses all cores
                             )

grid_search_rf.fit(X_train, y_train) #Train on training data

best_params_rf = grid_search_rf.best_params_ #Gathers the best parameters returned from GridSearchCV
best_score_rf = -grid_search_rf.best_score_ #Gathers the best score by converting the MAE to positive

print("Best parameters for RF: ", best_params_rf) #Displayed the best hyperparameters
print("Best MAE (validation) for RF: ", best_score_rf) #Displayes the performance of the hyperparameters



'''
Hyperparameter tuning for gradient boosting model
'''
param_grid_gbr = {
    'n_estimators': [10, 20, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 10, 20]
}

weighted_mae_scorer_gbr = make_scorer( #Creating a custom scorer, prioritising the holiday weeks
    lambda y_true, y_pred_gbr: weighted_mae(y_true, y_pred_gbr, X_train.loc[y_true.index, 'IsHoliday']),
    greater_is_better=False #Since GridSearchCV minimizes the score
)

grid_search_gbr = GridSearchCV(
    estimator = GradientBoostingRegressor(random_state = 42),
    param_grid = param_grid_gbr,
    scoring = weighted_mae_scorer_gbr,  #Use custom weighted MAE scorer
    cv = 2,
    verbose = 2,
    n_jobs = -1
)

grid_search_gbr.fit(X_train, y_train)

best_params_gbr = grid_search_gbr.best_params_
best_score_gbr = -grid_search_gbr.best_score_

print("Best parameters for GBR: ", best_params_gbr)
print("Best Weighted MAE for GBR: ", best_score_gbr)



'''
Re-training/testing the random forest model with the best parameters
'''
best_rf = RandomForestRegressor(**best_params_rf, random_state = 42) #uses the best parameters to train the model
best_rf.fit(X_train, y_train)

#Make predictions
y_pred_best_rf = best_rf.predict(X_val)  #Predict on the validation set

print("Re-training/testing for RF model complete")



'''
Re-training/testing the gradient boosting model with the best parameters
'''
best_gbr = GradientBoostingRegressor(**best_params_gbr, random_state=42)
best_gbr.fit(X_train, y_train)

#Make predictions
y_pred_best_gbr = best_gbr.predict(X_val)  #Predict on the validation set

print("Re-training/testing for GBR model complete")



'''
Making final predictions for RF
'''
X_test_rf = test_merged.drop(['Date'], axis = 1) #Dropping unecessary columns in the test_merged dataset

test_predictions_rf = best_rf.predict(X_test_rf) #Making the final predictions of the Weekly_Sales column

test_output_rf = pd.DataFrame({ #Constructing a portion of the test_merged dataset to output, including the newly predicted Weekly_Sales column
    'Store' : test_merged['Store'],
    'Dept' : test_merged['Dept'],
    'Date' : test_merged['Date'],
    'IsHoliday' : test_merged['IsHoliday'],
    'Weekly_Sales' : test_predictions_rf
})

#Modifications are only made to the test dataset the model is optimised

print(test_output_rf)



'''
Making final predictions for GBR
'''
X_test_gbr = test_merged.drop(['Date'], axis = 1) #Dropping unecessary columns in the test_merged dataset

test_predictions_gbr = best_gbr.predict(X_test_gbr) #Making the final predictions of the Weekly_Sales column

test_output_gbr = pd.DataFrame({ #Constructing a portion of the test_merged dataset to output, including the newly predicted Weekly_Sales column
    'Store' : test_merged['Store'],
    'Dept' : test_merged['Dept'],
    'Date' : test_merged['Date'],
    'IsHoliday' : test_merged['IsHoliday'],
    'Weekly_Sales' : test_predictions_gbr
})

print(test_output_gbr)



'''
Evaluate tuned RF model's performance
'''
#Calculates mean absolute error by comparing the predicted values (y_pred) with the actual values (y_val)
mae_weighted_best_rf = weighted_mae(y_val, y_pred_best_rf, X_val['IsHoliday']) #Pass the holiday flag

rmse_best_rf = np.sqrt(mean_squared_error(y_val, y_pred_best_rf))
r2_best_rf = r2_score(y_val, y_pred_best_rf)

holiday_mask = X_val['IsHoliday'] == 1
mae_holiday_best_rf = mean_absolute_error(y_val[holiday_mask], y_pred_best_rf[holiday_mask])
mae_non_holiday_best_rf = mean_absolute_error(y_val[~holiday_mask], y_pred_best_rf[~holiday_mask])

#Outputs all the evaluation metrics of the tuned model
print("Tuned Random Forest Regressor performance:")
print("Weighted MAE", mae_weighted_best_rf)
print("Holiday MAE: ", mae_holiday_best_rf) 
print("Non-holiday MAE: ", mae_non_holiday_best_rf)
print("RMSE: ", rmse_best_rf)
print("R^2 Score: ", r2_best_rf)



'''
Evaluate the tuned GBR model's performance
'''
#Calculates mean absolute error by comparing the predicted values (y_pred) with the actual values (y_val)
mae_weighted_best_gbr = weighted_mae(y_val, y_pred_best_gbr, X_val['IsHoliday']) #Pass the holiday flag

rmse_best_gbr = np.sqrt(mean_squared_error(y_val, y_pred_best_gbr))
r2_best_gbr = r2_score(y_val, y_pred_best_gbr)

holiday_mask = X_val['IsHoliday'] == 1 #Separation of MAE into holiday and non-holiday
mae_holiday_best_gbr = mean_absolute_error(y_val[holiday_mask], y_pred_best_gbr[holiday_mask])
mae_non_holiday_best_gbr = mean_absolute_error(y_val[~holiday_mask], y_pred_best_gbr[~holiday_mask])

#Outputs all the evaluation metrics of the tuned model
print("Tuned Gradient Boosting Regressor performance:")
print("Weighted MAE", mae_weighted_best_gbr)
print("Holiday MAE: ", mae_holiday_best_gbr) 
print("Non-holiday MAE: ", mae_non_holiday_best_gbr)
print("RMSE: ", rmse_best_gbr)
print("R^2 Score: ", r2_best_gbr)



'''
Feature importance for RF
'''
#Shows how much certain features contributed to the predictions of the model
feature_importances_rf = best_rf.feature_importances_
important_features_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances_rf
}).sort_values(by='Importance', ascending=False)

print(important_features_rf)



'''
Feature importance for GBR
'''
#Shows how much certain features contributed to the predictions of the model
feature_importances_gbr = best_gbr.feature_importances_
important_features_gbr = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances_gbr
}).sort_values(by='Importance', ascending=False)

print(important_features_gbr)