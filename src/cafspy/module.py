
import pandas as pd
import numpy as np
import random
from numpy import inf
from sklearn.base import is_classifier
from testflows.combinatorics import Covering
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import cross_val_score

def ICAFS(X, y, t, T,lr, seed = 42, shuffle =False,print_logs=False):
  
  """Iterative Covering Array Feature Selection Algorithm.

  This Mehtod runs ICAFS, a covering array beased algorithm for feture selection

  Args:
      X (pd.DataFrame): a dataset that forcefully must include headers as string.
      y (pd.DataFrame): a dataste of classes that forcefully must include headers as string.
      t (int): interaction level between features.
      T (int): number of iteration.
      lr(sklearn): a sklearn classification algorithms.   
      seed (int,optional): a seed
      shuffle(bool,optional): if true ti will perform shuffle based on the seed
      print_logs(bool,optional):print the selected features at each iteration
  Returns:
      pd.DataFrame: A pandas DataFrame containing the processed data.

  Raises:
      FileNotFoundError: If the file at `filepath` does not exist.
      ValueError: If `output_format` is not a supported format.
  """

  _validate_dataframe_model(X,y,lr)
  score_list = []
  featur_list = []
  
  v_variable = [0, 1]
  best_f1_score = float('-inf')
  best_header = X.columns.values.copy()
  
  score = cross_val_score(lr, X.values, y.values.ravel(), cv=5, scoring='f1_macro')
  score_list.append(score.mean())
  featur_list.append(X.shape[1])

  for  it in range(0,T):
      dict_parameters = {}
      partial_score = 0
      
      if shuffle:
        random.seed(seed)
        random.shuffle(best_header)

      for colum_key in best_header:
          dict_parameters[colum_key] = v_variable
      generate_covering_array = Covering(dict_parameters, strength=t)
      for test in generate_covering_array.array:
          list_attributes_to_consider = []

          check_for_all_cero = True
          for (test_key, test_value) in test.items():
              if test_value == 1:
                  check_for_all_cero = False
                  list_attributes_to_consider.append(test_key)

          if check_for_all_cero:
              continue  
          X_prime = X[list_attributes_to_consider]
          score = cross_val_score(lr,X_prime.values, y.values.ravel(), cv=5, scoring='f1_macro')
          if  score.mean() >= partial_score :
              partial_score = score.mean()
              best_header = list_attributes_to_consider.copy()
      best_f1_score = partial_score
      if print_logs:
        print(f"best f1 score= {best_f1_score}, iteration:{it+1}, numbers features selected ={ len(best_header)},best features selected={', '.join(best_header)}" )
      score_list.append(best_f1_score)
      featur_list.append(len(best_header))
  return score_list,featur_list


def CAFS(ca,X,y, T ,lr,print_logs=False):

  _validate_dataframe_model(X,y,lr) 
  num_rows = ca.shape[0]
  if len(X.columns) < ca.shape[1] :
    num_colums = len(X.columns)
  else:
    num_colums = ca.shape[1]
  best_f1_score = float('-inf')
  best_headers = X.columns.values.copy()
  featur_list = []
  score_list = []

  score = cross_val_score(lr, X.values, y.values.ravel(), cv=5, scoring='f1_macro')
  score_list.append(score.mean())
  featur_list.append(X.shape[1])

  for  it in range(0,T):
     partial_score = 0
     partial_best_header = []
     for i in range(0,num_rows):
        lst_headers_to_select = []
        for j in range(0,num_colums):
          if ca[i][j] == 1 :
              lst_headers_to_select.append(best_headers[j])
        if len(lst_headers_to_select) == 0:
            continue
        X_prime = X[lst_headers_to_select]
        score = cross_val_score(lr,X_prime.values, y.values.ravel(), cv=5, scoring='f1_macro')
        if score.mean() > partial_score :
            partial_score = score.mean()
            partial_best_header = lst_headers_to_select.copy()
     best_f1_score = partial_score
     best_headers = partial_best_header
     if print_logs:
         print(f"best f1 score= {best_f1_score}, iteration:{it+1}, numbers features selected ={len(best_headers)},best features selected={', '.join(best_headers)}" )

     num_colums  = len(best_headers)
     score_list.append(best_f1_score)
     featur_list.append(len(best_headers))
 
  return score_list,featur_list

# Private methods
def _validate_classifier(model_object):
    """Checks if an object is a scikit-learn classifier and prints the result."""
    model_name = model_object.__class__.__name__
    if is_classifier(model_object) == False:
        raise TypeError(f"❌ FAILURE: '{model_name}' is NOT a scikit-learn classifier.")
    
def _validate_all_columns_are_integer_type(df):
    """Checks if the dtype of every column is an integer type."""
    is_all_int_dtype = df.dtypes.apply(is_numeric_dtype).all()    
    if not is_all_int_dtype:
        raise TypeError("Validation FAILED: One or more columns do not have an integer dtype")
    
def _validate_dataframe_model(X,y,lr):
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame): 
      raise TypeError("Either X or y must be pandas DataFrame")
    if y.shape[1] > 1: 
      raise TypeError("Target must have only one column")
    _validate_classifier(lr)
    _validate_all_columns_are_integer_type(X)
    _validate_all_columns_are_integer_type(y)
