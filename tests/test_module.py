import pytest
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

from cafspy import ICAFS
from cafspy import CAFS

def test_icafs_cacao():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')
   
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1]).to_frame()
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())

    lr_algo = KNeighborsClassifier(n_neighbors=3)
    scores_list,feature_list = ICAFS(X_algarrobo,y_algarrobo,t=2,T=10,lr=lr_algo,print_logs=True)

    assert len(feature_list) == 11
    assert len(scores_list) == 11

def test_icafs_dataframe_are_not_pandas():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')
 
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())
    
    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)

def test_icafs_y_has_two_columns():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')

    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = "test"

    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)


def test_icafs_y_has_two_columns():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')
 
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = "test"

    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)

def test_icafs_lr_is_classification():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')

    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = 1

    with pytest.raises(TypeError):
        lr_algo = LogisticRegression()
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)

def test_icafs_contain_only_numbers():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')

    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    X_algarrobo["R"] = "A"
    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)

def test_cafs_cacao():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')
    covering_array  = np.loadtxt('data/coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1]).to_frame()
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())

    lr_algo = KNeighborsClassifier(n_neighbors=3)
    scores_list,feature_list = CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)

    assert len(feature_list) == 11
    assert len(scores_list) == 11

def test_cafs_dataframe_are_not_pandas():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')
    covering_array  = np.loadtxt('data/coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())
    
    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)

def test_cafs_y_has_two_columns():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')
    covering_array  = np.loadtxt('data/coveringArray.csv', delimiter=",", dtype=int)
    
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = "test"

    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)


def test_cafs_y_has_two_columns():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')
    covering_array  = np.loadtxt('data/coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = "test"

    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        CAFS(X_algarrobo,y_algarrobo,10,lr_algo,True)

def test_cafs_lr_is_classification():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')
    covering_array  = np.loadtxt('data/coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = 1

    with pytest.raises(TypeError):
        lr_algo = LogisticRegression()
        CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)

def test_cafs_contain_only_numbers():
    df_algarrobo = pd.read_csv('data/algarrobo.csv')
    covering_array  = np.loadtxt('data/coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    X_algarrobo["R"] = "A"
    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)