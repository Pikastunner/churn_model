from pprint import pprint 
from collections import defaultdict
import numpy as np
import pandas as pd
import regex as re
import seaborn as sns
import matplotlib.pyplot as mtp
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from matplotlib.colors import ListedColormap
from sklearn import linear_model
import sys 
from patsy import dmatrices
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def vif_df(data):
    model_string = ' + '.join(data.columns)
    y, x = dmatrices(f'Churn ~ {model_string}', data=data, return_type = 'dataframe')
    vif_df = pd.DataFrame()
    vif_df['Variable'] = x.columns
    vif_df['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif_df

def clean_data(data):

    # FIX
    check = []
    check.append(len(data))
    string_formats = []
    for column in data:
        data[column].replace('(^[\s]+$|^$)', np.nan, inplace=True, regex=True)
    #data = data.astype({'TotalCharges':'float64'})


    # Check for heavily null columns and remove
    to_remove = []
    num_rows = len(data)
    for col in data.columns:
        if data[col].isna().sum() > num_rows / 2:
            to_remove.append(col)
    data = data.drop(to_remove)

    # Replace nulls that only appear once in a row with the mean of that column
    to_replace = {}
    for index, row in data.iterrows():
        null_rows = row.isna().sum()
        if (null_rows == 1):
            curr_row = [str(i) for i in list(row)]
            to_replace.update({index:  curr_row.index('nan')})
    for i, j  in to_replace.items():
        data.iloc[[i], j] = data.loc[:, list(data.columns)[j]].mean()
    data = data.dropna()


    to_hot_encode = []
    for i in list(data.columns):
        unique_vals = data[i].value_counts()
        number_format = str(data[i].iloc[0]).replace(".", "")
        if not str(number_format).isnumeric():
            ord_enc = OrdinalEncoder() 
            data[i] = ord_enc.fit_transform(data[[i]])
            """
            if len(list(unique_vals)) == 2:
                ord_enc = OrdinalEncoder() 
                dataset[i] = ord_enc.fit_transform(dataset[[i]])
            else:
                print(i, repr((dataset[i]).iloc[0]))
                to_hot_encode.append(i)
            """
    if len(to_hot_encode) > 0:
        data = pd.get_dummies(data, columns = to_hot_encode)
        #print(data)
    old_column_names = [old_cols for old_cols in data.columns]
    new_column_names = [re.sub("[()-\s]", "", new_cols) for new_cols in old_column_names]
    replace_names = {old_column_names[i]: new_column_names[i] for i in range(len(old_column_names))}
    data = data.rename(columns = replace_names)







    mean_of_cols = {column: np.mean(data[column]) for column in data.columns}
    sd_of_cols = {column: np.std(data[column]) for column in data.columns}
    num_cols = len(data.columns)
    to_remove = []
    for index, row in data.iterrows():
        outliers = []
        for column in data.columns:
            data_mean, data_std = mean_of_cols[column], sd_of_cols[column]
            threshold = data_std * 3
            lower_bound, upper_bound = data_mean - threshold, data_mean + threshold
            if (row[column] <= lower_bound or row[column] >= upper_bound):
                outliers.append(column)
        if len(outliers) == 0:
            continue
        elif len(outliers) > num_cols / 2:
            to_remove.append(data["customerID"])
        else:
            for cols in outliers:
                data.iloc[index][cols] = mean_of_cols[cols]
    for id in to_remove:
        data = data[data.CustomerID != id]
    # Remove the first column in the dataset which corresponds with the customerID
    data.pop(data.columns[0])






    scaler = MinMaxScaler()
    to_add = data['Churn']
    sliced = data.loc[:, data.columns != 'Churn']
    sliced = scaler.fit_transform(sliced)
    data = pd.DataFrame(sliced, columns=data.columns[:-1], index=data.index)
    data["Churn"] = to_add 

    to_purge = []
    for column in data:
        val_dis = list(data[column].value_counts())
        if len(val_dis) == 2:
            if min(val_dis) < 0.9 * max(val_dis):
                to_purge.append(column)
    data.drop(to_purge, axis=1)




    model_string = ' + '.join(data.columns)
    y, x = dmatrices(f'Churn ~ {model_string}', data=data, return_type = 'dataframe')
    vif_df = pd.DataFrame()
    vif_df['Variable'] = x.columns
    vif_df['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    high_vif = vif_df.loc[vif_df['VIF'] >= 10]
    to_drop = [i for i in (high_vif['Variable']) if i != 'Intercept']
    data.drop(to_drop, axis=1)

    return data



# THERE MAY BE A PROBLEM WITH THIS FUNCTION SINCE CHURN MAY BE REMOVED ALREADY
def feat_sel1(data, x, x_train, x_test, y_train, y_test):
    # Checking which variables are significant using ANOVA f-test Feature Selection
    bestfeatures = SelectKBest(score_func = f_classif, k = 'all')
    bestfeatures.fit(x_train, y_train)
    x_train_fs = bestfeatures.transform(x_train)
    x_test_fs = bestfeatures.transform(x_test)
    model_by_Anova = {}
    for i in range(len(bestfeatures.scores_)):
        if (bestfeatures.pvalues_[i] <= 0.05):
            model_by_Anova.update({x.columns[i]: i}) 
        print(f"Feature: {i, bestfeatures.pvalues_[i]}")
    return [bestfeatures.scores_, model_by_Anova]
    #pyplot.bar([i for i in range(len(bestfeatures.scores_))], bestfeatures.scores_)

def feat_sel2(data, x, x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)   
    sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
    sfs.fit(x_train, y_train)
    ret = {}
    stor = sfs.get_support()
    for i, j in enumerate(stor):
        ret.update({x.columns[j]: j})
    return ret


def get_cm(data, x, x_train, x_test, y_train, y_test, classifier):
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    return [cm, classifier, y_pred]

def get_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "f1 Score": f1}
    df = pd.DataFrame()
    df["Metrics"] = metrics.keys()
    df["Scores"] = metrics.values()
    return df