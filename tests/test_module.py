import pytest
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

# Optional - to run the code from source code, without installing cafspy using pip...
# Add the path of the module to run directly the test file on "tests" location 
import sys
sys.path.append("./../src")

# Add data location to all functions, in order to run the test_module directly as a script.
datapath = "../data/"

from cafspy import ICAFS
from cafspy import CAFS
# Opt-in to the new behavior to disable silent downcasting
pd.set_option('future.no_silent_downcasting', True)

def test_icafs_cacao():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')
   
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1]).to_frame()
    y_algarrobo = y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())

    lr_algo = KNeighborsClassifier(n_neighbors=3)
    scores_list,feature_list = ICAFS(X_algarrobo,y_algarrobo,t=2,T=10,lr=lr_algo,print_logs=True)

    assert len(feature_list) == 11
    assert len(scores_list) == 11

def test_icafs_dataframe_are_not_pandas():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')
 
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())
    
    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)

def test_icafs_y_has_two_columns():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')

    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = "test"

    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)


def test_icafs_y_has_two_columns():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')
 
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = "test"

    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)

def test_icafs_lr_is_classification():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')

    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = 1

    with pytest.raises(TypeError):
        lr_algo = LogisticRegression()
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)

def test_icafs_contain_only_numbers():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')

    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    X_algarrobo["R"] = "A"
    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        ICAFS(X_algarrobo,y_algarrobo,2,10,lr_algo,True)

def test_cafs_cacao():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')
    covering_array  = np.loadtxt(datapath + 'coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1]).to_frame()
    y_algarrobo = y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())
    lr_algo = KNeighborsClassifier(n_neighbors=3)
    scores_list,feature_list = CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)

    assert len(feature_list) == 11
    assert len(scores_list) == 11

def test_cafs_dataframe_are_not_pandas():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')
    covering_array  = np.loadtxt(datapath + 'coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())

    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)

def test_cafs_y_has_two_columns():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')
    covering_array  = np.loadtxt(datapath + 'coveringArray.csv', delimiter=",", dtype=int)
    
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = "test"

    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)


def test_cafs_y_has_two_columns():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')
    covering_array  = np.loadtxt(datapath + 'coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = "test"

    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        CAFS(X_algarrobo,y_algarrobo,10,lr_algo,True)

def test_cafs_lr_is_classification():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')
    covering_array  = np.loadtxt(datapath + 'coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1]).to_frame()
    y_algarrobo = y_algarrobo.astype(int)
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    y_algarrobo["Test"] = 1

    with pytest.raises(TypeError):
        lr_algo = LogisticRegression()
        CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)

def test_cafs_contain_only_numbers():
    df_algarrobo = pd.read_csv(datapath + 'algarrobo.csv')
    covering_array  = np.loadtxt(datapath + 'coveringArray.csv', delimiter=",", dtype=int)
    unique_names_algarrobo = df_algarrobo['Labels'].unique()
    algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']
    y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1])
    y_algarrobo.astype('int64')
    X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min()).to_frame()
    X_algarrobo["R"] = "A"
    with pytest.raises(TypeError):
        lr_algo = KNeighborsClassifier(n_neighbors=3)
        CAFS(covering_array,X_algarrobo,y_algarrobo,10,lr_algo,True)



# Modify here to run test function that will be used
print("CAFS-Cacao")
test_cafs_cacao()

print("iCAFS-Cacao")
test_icafs_cacao()
