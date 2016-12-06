import pandas as pd
import numpy as np 

'''
def normalize_df(df):
  values = df.values
  means = np.mean(values, axis=0)
  stds = np.std(values, axis=0)
  normalized_values = (values - means) / stds
  normalized_df = pd.DataFrame(columns=df.columns)
  normalized_df.values = normalized_values
  return(normalized_values)
'''

def compute_cluster_averages(df, csv_filename=None, save_csv=None):
  if csv_filename is not None:
    df = pd.read_csv(csv_filename, index_col=0)
  rownames = list(df.index)
  cluster_names = [rowname.split("_")[0] for rowname in rownames]
  df.index = cluster_names
  df_averages = pd.DataFrame(columns=df.columns.values)
  for i, cluster_name in enumerate(cluster_names): 
    cluster_df = df.loc[cluster_name]
    means = cluster_df.apply(np.mean)
    df_averages.loc[cluster_name] = means

  if save_csv is not None:
    df_averages.to_csv(save_csv)
  return(df_averages)
'''
def regress_docking_onto_tica(docking_csv_filename, aggregate_docking_filename, tica_csv_filename, aggregate_tica_filename):
  with open(docking_df_filename, "rb") as f:
    docking_df = pickle.load(f)

  docking_df = normalize_df(docking_df)
  docking_df = compute_cluster_averages(docking_df)

  with open(aggregate_docking_filename, "wb") as f:
    pickle.dump(docking_df, f)

  tica_df = load_csv_from_pickle(tica_csv_filename)
  tica_df = normalize_df(tica_df)
  tica_df = compute_cluster_averages(tica_df)

  join_dfs(docking_df, tica_df)

  call_lasso_function_docking_vs_tica()

  return s

def compute_auc_docking_active():
  do_analysis()
  save_plot()
'''


'''
Ideas for OOP refatoring. 

Class:

class Analyzer

class Dataset
'''