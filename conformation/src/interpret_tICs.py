from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import gzip
import mdtraj as md
from copy import deepcopy
from io_functions import *
import pandas as pd
import os
import copy
import multiprocessing as mp 
import pickle 
from residue import *
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import lasso_path, LogisticRegressionCV
#import matplotlib
#from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import roc_auc_score
#sns.style.use('fivethirtyeight')
#rcParams.update({'figure.autolayout': True})

'''Pseudocode:

compute_random_forests_fxn:
  Load and concat all features
  Load and concat tIC projections

  for each column in tIC array:
    do random forest regression from all feature space onto that tIC
    save resulting feature importances. 

interpret_tIC_rf:
  for each tIC:
    load feature importances 
    get feature names, combine and sort (use old code)
    plot a bar graph. 
    print out a matlines (or equivalent) plot for top 10 residues that matter.
    figure out a way to superimpose these on top of tIC heatmaps

'''

def compute_random_forests(features_dir, projected_tica_coords, rf_dir, R=True, n_trees=500, 
  n_tica_components=25, start_tIC=0):

  features = np.concatenate(load_file_list(None, directory = features_dir, ext = ".dataset"))
  tics = np.concatenate(load_file(projected_tica_coords))

  for j in range(start_tIC, np.shape(tics)[1]):
    #os.system("rm %s/tIC.%d_rfr.pkl" %(rf_dir, j))
    rfr = RandomForestRegressor(n_estimators=n_trees, max_features="sqrt",
                                n_jobs=8, verbose=1)
    print(("Fitting tree for tIC %d" %(j+1)))
    rfr.fit(features, tics[:,j])
    print("Done fitting RF model. Saving now...")

    with open("%s/tIC.%d_rfr.pkl" %(rf_dir, j), "wb") as f:
      pickle.dump(rfr.feature_importances_, f)
    print("Saved random forest feature importances")

  return

def get_feature_list(feature_residues_csv, structure_file):
  feature_names = generate_features(feature_residues_csv)
  structure = md.load_frame(structure_file, index=0)
  all_resSeq = [r.resSeq for r in structure.topology.residues if r.is_protein]
  all_res = [str(r).title() for r in structure.topology.residues if r.is_protein]

  feature_list = []
  for i, feature_name in enumerate(feature_names):
    try:
      res_i = int(feature_name[0])
      res_j = int(feature_name[1])
    except:
      res_i = int(feature_name[0][1])
      res_j = int(feature_name[1][1])

    res_i_idx = all_resSeq.index(res_i)
    res_i_name = all_res[res_i_idx]

    res_j_idx = all_resSeq.index(res_j)
    res_j_name = all_res[res_j_idx]

    feature_list.append((res_i_name, res_j_name))

  return(feature_list)

def plot_importance_df(df, column_name, ylabel, title, analysis_type, save_string, j, save_dir):
  n_rows = df.shape[0]
  bar_width = [0.4 for i in range(0,n_rows)]
  opacity = 0.8
  index = np.arange(n_rows)
  df_copy = copy.deepcopy(df)
  df = df.reindex(df_copy.importance.abs().order(ascending=False).index)
  df = df[:n_rows]
  print(df)
  with plt.style.context(('fivethirtyeight')):
    print("Using dark_background")
    plt.barh(bottom=np.arange(n_rows), width=df['importance'].values, height=bar_width, 
             alpha=opacity, color='b',label='Feature importance')
    plt.ylabel(ylabel)
    plt.xlabel('Overall Importance')
    plt.title(title)
    plt.yticks(index + bar_width, df[column_name].tolist(), fontsize=8)
    pp = PdfPages("%s/%s%d_%s.pdf" %(save_dir, analysis_type, (j+1), save_string))
    pp.savefig()
    pp.close()
    plt.clf()

def plot_df_bar(df, ylabel, xlabel, title, save_dir):
  pass

def compute_residue_importances(df, percentile=95):
  residue_importances = {}
  for row in df.iterrows():
    row = row[1]
    res_i = row['res_i']
    if res_i not in list(residue_importances.keys()):
      residue_importances[res_i] = []
    residue_importances[res_i].append(np.abs(row['importance']))

    res_j = row['res_j']
    if res_j not in list(residue_importances.keys()):
      residue_importances[res_j] = []
    residue_importances[res_j].append(np.abs(row['importance']))

  for residue in residue_importances.keys():
    if len(residue_importances[residue]) == 0:
      residue_importances[residue] = 0.
    elif len(residue_importances[residue]) == 1:
      residue_importances[residue] = residue_importances[residue][0]
    else:
      residue_importances[residue] = np.percentile(residue_importances[residue], percentile)

  return(residue_importances)

def merge_importances_features(feature_importances_file, feature_residues_map, importances=None):
  feature_objects = compat_verboseload(feature_residues_map)

  if importances is None:
    try:
      with open(feature_importances_file) as f:
        importances = pickle.load(f)
    except:
      with gzip.open(feature_importances_file) as f:
        importances = pickle.load(f)

  feature_importances = list(zip(feature_objects, importances.tolist()))
  rows = []
  for i, feature_importance in enumerate(feature_importances):
    if np.abs(feature_importance[1]) > 0.0:
      if hasattr(feature_importance[0], "residue_i"):
        row = [feature_importance[0].__repr__().title(), 
                     feature_importance[0].residue_i.__repr__().title(),
                     feature_importance[0].residue_j.__repr__().title(), 
                     feature_importance[1], feature_importance[0]]
      else:
        row = [feature_importance[0].__repr__().title(), 
                     feature_importance[0].residue.__repr__().title(),
                     feature_importance[0].residue.__repr__().title(), 
                     feature_importance[1], feature_importance[0]]      
      rows.append(row)
    
  df = pd.DataFrame(rows, columns=('feature_name', 'res_i', 'res_j', 
                                   'importance', 'feature'))

  return df

def compute_per_residue_importance(df, percentile):
  net_df = pd.DataFrame(columns=['residue', 'importance'])
  for _, row in df.iterrows():
    res_i = row['res_i']
    res_j = row['res_j']

    importance = row["importance"]
    if res_i not in net_df.index:
      net_df.loc[res_i] = [res_i, [importance]]
    else:
      net_df.loc[res_i]["importance"].append(importance)

    if res_j not in net_df.index:
      net_df.loc[res_j] = [res_j, [importance]]
    else:
      net_df.loc[res_j]["importance"].append(importance)

  net_df["importance"] = net_df["importance"].apply(np.percentile, q=percentile)

  return net_df

def interpret_tIC_components(tica_filename, save_dir, feature_residues_pkl, n_tica_components=25, percentile=95):
  tica_object = compat_verboseload(tica_filename)
  tica_components = tica_object.components_
  features_with_non_zero_importances = []
  features_per_tIC = []

  for j in range(0, np.shape(tica_components)[0]):
    if j >= n_tica_components: break

    components = tica_components[j,:]
    print(("Interpreting tIC %d" %(j+1)))
    feature_importances_df = merge_importances_features(None, feature_residues_pkl, components)
    residue_importances_df = compute_per_residue_importance(feature_importances_df, percentile)
    features_with_non_zero_importances += [feature for feature in feature_importances_df['feature'].tolist()]
    features_per_tIC.append([feature for feature in feature_importances_df['feature'].tolist()])

    print("feature_importances_df.shape")
    print(feature_importances_df.shape)
    print("residue_importances_df.shape")
    print(residue_importances_df.shape)

    plot_importance_df(feature_importances_df, "feature_name", "Contact", 
                       "tIC %d Contact Component" % (j+1),
                       "tIC", 
                       "per_contact_importances", j, save_dir)

    plot_importance_df(residue_importances_df, "residue", "Residue", 
                       "tIC %d Residue Importances" % (j+1), 
                       "tIC",
                       "per_residue_importances", j, save_dir)

    with open("%s/tIC.%d_feature_importance_df.pkl" %(save_dir, j), "wb") as f:
      pickle.dump(feature_importances_df, f)

    with open("%s/tIC.%d_residue_importance_df.pkl" %(save_dir, j), "wb") as f:
      pickle.dump(residue_importances_df, f)
  return list(set(features_with_non_zero_importances)), features_per_tIC

def interpret_tIC_rf(rf_dir, feature_residues_pkl, n_tica_components=25, percentile=95):
  for j in range(0, n_tica_components):
    print(("Interpreting tIC %d" %(j+1)))
    importances_file = "%s/tIC.%d_rfr.pkl" %(rf_dir, j)
    if os.path.exists(importances_file):
      feature_importances_df = merge_importances_features(importances_file, feature_residues_pkl)
      residue_importances_df = compute_per_residue_importance(feature_importances_df, percentile)

      plot_importance_df(feature_importances_df, "feature_name", "Contact", 
                         "tIC %d Contact Importances" % (j+1),
                         "tIC", 
                         "per_contact_importances", j, rf_dir)

      plot_importance_df(residue_importances_df, "residue", "Residue", 
                         "tIC %d Residue Importances" % (j+1), 
                         "tIC",
                         "per_residue_importances", j, rf_dir)

      with open("%s/tIC.%d_rfr_feature_importance_df.pkl" %(rf_dir, j), "wb") as f:
        pickle.dump(feature_importances_df, f)

      with open("%s/tIC.%d_rfr_residue_importance_df.pkl" %(rf_dir, j), "wb") as f:
        pickle.dump(residue_importances_df, f)


'''
def interpret_tIC_rf(rf_dir, feature_residues_csv, structure_file, n_tica_components=25):
  df = pd.DataFrame(columns=('index', 'res_i', 'res_j', 'feature_name', 'importance'))
  feature_list = get_feature_list(feature_residues_csv, structure_file)
  for i, feature in enumerate(feature_list):
    df.loc[i] = [i, feature[0], feature[1], "", 0.0]

  for j in range(0,n_tica_components):
    print("Interpreting tIC %d" %(j+1))
    df_j = deepcopy(df)
    with open("%s/tIC.%d_rfr.pkl" %(rf_dir, j), "rb") as f:
      feature_importances = pickle.load(f)

    df_j['importance'] = feature_importances
    concatenated_feature_names = zip(df['res_i'].tolist(), df['res_j'].tolist())
    concatenated_feature_names = ["%s-%s" %(res_i, res_j) for res_i, res_j in concatenated_feature_names]
    df_j['feature_name'] = concatenated_feature_names
    with open("%s/tIC.%d_rfr_feature_importance_df.pkl" %(rf_dir, j), "wb") as f:
      pickle.dump(df_j, f)
    plot_importance_df(df_j, save_string="per_feature_rf_importance", j=j, save_dir=rf_dir)

    residue_df = pd.DataFrame(columns=('feature_name', 'importance'))
    residue_importances = compute_residue_importances(df_j)
    for i, (key, value) in enumerate(residue_importances.iteritems()):
      residue_df.loc[i] = [key, value]
    plot_importance_df(residue_df, save_string="per_residue_rf_importance", j=j, save_dir=rf_dir)

    with open("%s/tIC.%d_rfr_residue_importance_df.pkl" %(rf_dir, j), "wb") as f:
      pickle.dump(residue_df, f)
'''

def plot_top_features_per_tIC(projected_tica_coords_file, features_dir, features_ext, 
                              rf_dir, n_tica_components, normalize=False, n_features=5):
  features = np.concatenate(load_file_list(get_trajectory_files(features_dir, features_ext)))
  projected_tica_coords = np.concatenate(load_file(projected_tica_coords_file))

  for j in range(0, n_tica_components):
    print(("Examining tIC %d" %(j+1)))
    df = compat_verboseload("%s/tIC.%d_rfr_feature_importance_df.pkl" %(rf_dir, j))
    df.sort(columns='importance', inplace=True, ascending=0)
    indices = df[:n_features]['index'].tolist()
    top_features_normalized = features[:,indices]
    if normalize:
      top_features_normalized = preprocessing.scale(top_features_normalized)
    top_features_normalized = top_features_normalized[projected_tica_coords[:,j].argsort(),:]
    projected_tica_coords_sorted = projected_tica_coords[projected_tica_coords[:,j].argsort(),:]
    
    df = pd.DataFrame(top_features_normalized, 
                      index=projected_tica_coords_sorted[:,j], 
                      columns=df[:n_features]['feature_name'])

    plt.figure()
    pd.rolling_mean(df, 1000).plot()
    plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
    plt.title("Feature Behavior Over Course of tIC %d" %(j+1))
    plt.xlabel("tIC %d Projection" %(j+1))
    plt.ylabel("Distance (nm)")
    pp = PdfPages("%s/tIC%d_feature_behavior.pdf" %(rf_dir, (j+1)))
    pp.savefig(bbox_inches='tight')
    pp.close()
    plt.clf()



def top_residues_per_tIC():
  return

def rank_tICs_by_docking_rf(docking_csv, tica_coords_csv, analysis_dir, n_trees=500):
  docking = pd.read_csv(docking_csv, header=0, index_col=0)
  tica_coords = pd.read_csv(tica_coords_csv, header=0, index_col=0)
  tica_coords = tica_coords.loc[docking.index]
  tica_coords = preprocessing.scale(tica_coords)

  error_rate = []

  # Range of `n_estimators` values to explore.
  min_estimators = 25
  max_estimators = 500

  for n_estimators in range(min_estimators, max_estimators + 1,15):
    rfr = RandomForestRegressor(n_estimators=n_estimators, 
                                max_features='sqrt', n_jobs=-1, oob_score=True)
    rfr.fit(tica_coords, docking)

    # Record the OOB error for each `n_estimators=i` setting.
    oob_error = 1 - rfr.oob_score_
    error_rate.append((n_estimators, oob_error))

  # Generate the "OOB error rate" vs. "n_estimators" plot.

  n_estimators = [tup[0] for tup in error_rate]
  errors = [tup[1] for tup in error_rate]

  plt.plot(n_estimators, errors)

  plt.xlim(min_estimators, max_estimators)
  plt.xlabel("n_estimators")
  plt.ylabel("OOB error rate")
  plt.title("Random Forest Regression Model Selection")
  plt.show()

  pp = PdfPages("%s/docking_tica_rf_oob_plot.pdf" % analysis_dir)
  pp.savefig(bbox_inches='tight')
  pp.close()
  plt.clf()

  best_n_estimators = n_estimators[np.argmin(errors)]
  print("best_n_estimators")
  print(best_n_estimators)
  final_model = RandomForestRegressor(n_estimators=best_n_estimators, 
                                      max_features='sqrt', n_jobs=-1, oob_score=True)
  final_model.fit(tica_coords, docking)

  df_rows = []
  for j in range(0, np.shape(tica_coords)[1]):
    df_rows.append(["tIC %d" % (j+1), final_model.feature_importances_[j]])
  df = pd.DataFrame(df_rows, columns=('tIC', 'importance'))

  plot_importance_df(df, "tIC", "Gini Importance of tIC", "Gini Importance of tICs in Predicting Docking Score", "docking_vs_tICA_rf", "", 0, analysis_dir)


def rank_tICs_by_docking_logistic(docking, tica, analysis_dir, docking_csv=None, tica_coords_csv=None):
  if docking_csv is not None:
    docking = pd.read_csv(docking_csv, header=0, index_col=0)
    drugs = docking.columns.values
    tica_coords = pd.read_csv(tica_coords_csv, header=0, index_col=0)
  tica_coords = tica_coords.loc[docking.index][tica_coords.columns.values]

  tica_coords = preprocessing.scale(tica_coords)
  docking = preprocessing.MinMaxScaler(feature_range=(0.01, 0.99)).fit_transform(docking)

  logit = np.log(docking / (1.0-docking))

  eps = 5e-3  # the smaller it is the longer is the path
  for d, drug in enumerate(drugs):
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(tica_coords, docking[:,d], eps, fit_intercept=False)

    print(np.shape(alphas_lasso))
    print(np.shape(coefs_lasso))

    df = pd.DataFrame(coefs_lasso.T, index=-np.log10(alphas_lasso), columns=["tIC %d" %(j+1) for j in range(0,np.shape(tica_coords)[1])])
    plt.figure()
    df.plot(colormap='gist_rainbow')

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso and Elastic-Net Paths, %s' %drug)
    plt.axis('tight')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pp = PdfPages("%s/%s_docking_tica_lasso_coefs_plot.pdf" % (analysis_dir, drug))
    pp.savefig(bbox_inches='tight')
    pp.close()
    plt.clf()

def rank_tICs_by_docking_mord(docking_csv, tica_coords_csv, analysis_dir, n_trees=500):
  docking = pd.read_csv(docking_csv, header=0, index_col=0)
  tica_coords = pd.read_csv(tica_coords_csv, header=0, index_col=0)
  tica_coords = tica_coords.loc[docking.index]

  tica_coords = preprocessing.scale(tica_coords)

  from mord import OrdinalRidge

  c_range = [float(i)/10000. for i in range(10600,10900)]
  print(c_range)
  coefs = np.zeros((len(c_range), np.shape(tica_coords)[1]))

  for j, C in enumerate(range(1,10)):
    lad = OrdinalRidge(alpha=C, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto')
    lad.fit(tica_coords, docking)
    coefs[j,:] = lad.coef_

  print(coefs)
  print(np.shape(coefs))
  df = pd.DataFrame(coefs, index=c_range, columns=["tIC %d" %(j+1) for j in range(0,np.shape(tica_coords)[1])])
  plt.figure()
  df.plot(colormap='gist_rainbow')

  plt.xlabel('-Log(alpha)')
  plt.ylabel('coefficients')
  plt.title('Lasso and Elastic-Net Paths')
  plt.axis('tight')
  #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  pp = PdfPages("%s/docking_tica_lasso_mord_coefs_plot.pdf" % analysis_dir)
  pp.savefig(bbox_inches='tight')
  pp.close()
  plt.clf()

#from http://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
from sklearn.metrics import mutual_info_score

def fd_rule(x):
  return np.ceil((np.max(x) - np.min(x)) / (2. * (np.percentile(x,75)-np.percentile(x,25)) * np.power(len(x), -0.3333)))

def calc_MI(x, y):
  bin_x, bin_y = fd_rule(x), fd_rule(y)
  print((bin_x, bin_y))
  c_xy = np.histogram2d(x, y, [bin_x, bin_y])[0]
  mi = mutual_info_score(None, None, contingency=c_xy)
  print(mi)
  return mi

def compute_MI_matrix(data_i, data_j):
  MI_matrix = np.zeros((np.shape(data_i)[1], np.shape(data_j)[1]))
  for i in range(0, np.shape(data_i)[1]):
    for j in range(0, np.shape(data_j)[1]):
      MI_matrix[i][j] = calc_MI(data_i[:,i], data_j[:,j])
  return MI_matrix

from scipy.stats import pearsonr

def compute_pearson_matrix(data_i, data_j):
  p_matrix = np.zeros((np.shape(data_i)[1], np.shape(data_j)[1]))
  for i in range(0, np.shape(data_i)[1]):
    for j in range(0, np.shape(data_j)[1]):
      p_matrix[i][j] = pearsonr(data_i[:,i], data_j[:,j])[0]
  return p_matrix

import entropy_estimators as ee

from scipy.stats import ks_2samp

def compute_ks_matrix(data_i, data_j):
  ks_matrix = np.zeros((np.shape(data_i)[1], np.shape(data_j)[1]))
  for i in range(0, np.shape(data_i)[1]):
    for j in range(0, np.shape(data_j)[1]):
      ks_matrix[i][j] = ks_2samp(data_i[:,i], data_j[:,j])[0]
      print(ks_matrix[i][j])
  return ks_matrix


from scipy.stats import spearmanr

def compute_sr_matrix(data_i, data_j):
  sr_matrix = np.zeros((np.shape(data_i)[1], np.shape(data_j)[1]))
  for i in range(0, np.shape(data_i)[1]):
    for j in range(0, np.shape(data_j)[1]):
      sr_matrix[i][j] = spearmanr(data_i[:,i], data_j[:,j])[0]
      print(sr_matrix[i][j])
  return sr_matrix

from scipy.stats import ranksums

def compute_rs_matrix(data_i, data_j):
  rs_matrix = np.zeros((np.shape(data_i)[1], np.shape(data_j)[1]))
  for i in range(0, np.shape(data_i)[1]):
    for j in range(0, np.shape(data_j)[1]):
      rs_matrix[i][j] = ranksums(data_i[:,i], data_j[:,j])[0]
      print(rs_matrix[i][j])
  return rs_matrix


def find_non_zero_features(important_contact_features, feature_residues):
  important_contact_features_sorted = [feature for feature in important_contact_features]
  important_contact_features_pruned = []
  for feature in important_contact_features_sorted:
      if feature not in important_contact_features_pruned:
          important_contact_features_pruned.append(feature)
  important_contact_features_indices = []
  for feature in important_contact_features_pruned:
      try:
        index = feature_residues.index(feature)
      except: 
        index = feature_residues.index(list(feature))
      important_contact_features_indices.append(index)
  return important_contact_features_pruned, important_contact_features_indices
  

def subsample(filename, indices, names):
  features = load_file(filename)
  subsampled_features = pd.DataFrame(features[:, indices], columns=names)
  return subsampled_features
    
from functools import partial
def subsample_features(features_dir, indices, names, save_file, features=None, worker_pool=None):
  feature_files = get_trajectory_files(features_dir, ".dataset")
  names = [str(name) for name in names]
  if features is not None:
    subsampled_features = [pd.DataFrame(f[:, indices], columns=names) for f in features]

  else:
    subsample_partial = partial(subsample, indices=indices, names=names)
    if worker_pool is not None:
      subsampled_features = worker_pool.map_sync(subsample_partial, feature_files)
    else:
      pool = mp.Pool(mp.cpu_count())
      subsampled_features = pool.map(subsample_partial, feature_files)
      pool.terminate()

  with open(save_file, "wb") as f:
    pickle.dump(subsampled_features, f)

def compute_single_model(j, data_i, data_j, task, model_type, n_trees, n_folds, max_depth, symmetric):
  print("Examining response variable %d out of %d" %(j, np.shape(data_j)[1]))
  if symmetric:
    X = data_i[:,list(range(0,j)) + list(range(j+1,data_i.shape[1]))]
  else:
    X = data_i
  y = data_j[:,j]
  score = []
  importance = []
  for k in range(0, n_folds):
    if task == "regression":
      rfm = RandomForestRegressor(n_estimators = n_trees, max_depth=max_depth, max_features = 'sqrt', n_jobs=-1)
      X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
      rfm.fit(X_train, y_train)
      score.append(rfm.score(X_test, y_test))
      importance.append(rfm.feature_importances_)
    else:
      if "rf" in model_type:
        rfm = RandomForestClassifier(n_estimators = n_trees, max_depth=max_depth, max_features = 'sqrt', n_jobs=-1)
        try:
          X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)
          rfm.fit(X_train, y_train)
          y_test_matrix = np.eye(len(set(y_test.tolist())))[y_test.tolist()]
          score.append(roc_auc_score(y_test_matrix,rfm.predict_proba(X_test)))
          importance.append(rfm.feature_importances_)
        except:
          score.append(0.)
          importance.append(np.zeros(np.shape(X)[1]))
      else:
        rfm = LogisticRegressionCV()
        try:
          X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)
          rfm.fit(X_train, y_train)
          y_test_matrix = np.eye(len(set(y_test.tolist())))[y_test.tolist()]
          score.append(roc_auc_score(y_test_matrix,rfm.predict_proba(X_test)))
          importance.append(rfm.coef_)
        except:
          score.append(0.)
          importance.append(np.zeros(np.shape(X)[1]))

  score = np.mean(score)
  importance = np.mean(np.vstack(importance), axis=0)
  if symmetric:
    arr = np.zeros(data_i.shape[1])
    arr[:j] = importance[:j]
    arr[(j+1):] = importance[j:]
    importance = arr

  return (score, importance)

def compute_sl_matrix(data_i, data_j, task="regression", model_type="rfr",
                      n_trees=500, n_folds=5, max_depth=3, symmetric=False, worker_pool=None, 
                      parallel=False):
  importances_matrix = np.zeros((np.shape(data_i)[1], np.shape(data_j)[1]))
  
  compute_single_model_partial = partial(compute_single_model, data_i=data_i, data_j=data_j,
                                         task=task, model_type=model_type, n_trees=n_trees,
                                         n_folds=n_folds, max_depth=max_depth, symmetric=symmetric)

  if worker_pool is not None:
    score_importance_tuples = worker_pool.map_sync(compute_single_model_partial, range(0, np.shape(data_j)[1]))
  elif parallel:
    pool = mp.Pool(mp.cpu_count())
    score_importance_tuples = pool.map(compute_single_model_partial, range(0, np.shape(data_j)[1]))
    pool.terminate()
  else:
    score_importance_tuples = []
    for j in range(0, np.shape(data_j)[1]):
      score_importance_tuples.append(compute_single_model_partial(j))

  scores = [t[0] for t in score_importance_tuples]
  importances = [t[1] for t in score_importance_tuples]

  for j, importance in enumerate(importances):
    importances_matrix[:,j] = importance

  return scores, importances_matrix

def calculate_cluster_averages_per_feature(clusterer, features):
  n_clusters = clusterer.n_clusters 
  concatenated_clusters = np.concatenate(clusterer.labels_)
  concatenated_features = np.concatenate(features)
  cluster_averages = np.zeros((n_clusters, concatenated_features.shape[1]))
  for i in range(0, n_clusters):
    rows = np.where(concatenated_clusters == i)[0]
    means = np.mean(concatenated_features[rows,:], axis=0)
    cluster_averages[i,:] = means
  return cluster_averages

