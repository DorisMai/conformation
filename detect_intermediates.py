from sklearn import mixture
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from msmbuilder.dataset import dataset, _keynat, NumpyDirDataset
from msmbuilder.utils import verbosedump, verboseload
import time
import random
import os
import multiprocessing as mp
import csv
from io_functions import *
from interpret_tICs import plot_importance_df, compute_residue_importances, merge_importances_features, compute_per_residue_importance
from functools import partial
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import gzip, pickle
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import operator
from sklearn import preprocessing
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, lasso_path, LogisticRegressionCV
import seaborn as sns
from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV
from scipy.signal import argrelextrema
from scipy import stats
from scipy import spatial
from matplotlib.colors import ListedColormap
from msmbuilder.utils import KDTree
from matplotlib.pyplot import cm
from sklearn import mixture


def select_model(X, tic_j, max_components, save_dir):
  num_trials = 0
  possible_components = []
  for trial in range(0,num_trials):
    train_indices = random.sample(list(range(0,len(X))), len(X)/2)
    test_indices = list(set(range(0,len(X))) - set(train_indices))
    X_train = X[train_indices,:]
    X_test = X[test_indices,:]

    bics = []
    aics = []
    test_likelihoods = []
    models = []
    for n_components in range(max_components, max_components):
        print(("For tIC %d looking at GMM model with %d components" %(tic_j, n_components)))
        g = mixture.DPGMM(n_components=10)
        #g = mixture.GMM(n_components = n_components, n_init = 3, min_covar = 1e-1, params='mc')
        g.fit(X_train)
        bics.append(g.bic(X_train))
        aics.append(g.aic(X_train))
        test_likelihoods.append(sum(g.score(X_test)))
        models.append(g)
    #plt.plot(range(1,max_components),aics)
    #minbic = bics.index(min(bics)) + 1
    #minaic = aics.index(min(aics)) + 1
    max_likelihood = test_likelihoods.index(max(test_likelihoods)) + 1
    #if(minbic) > 1: min_likelihood = test_likelihoods[1:].index(max(test_likelihoods[1:])) + 2
    #selected_components = min(minbic,minaic,min_likelihood)
    possible_components.append(max_likelihood)

  #num_components = min(possible_components)
  #g = mixture.GMM(n_components = num_components, n_init=5, tol =1e-5, min_covar = 1e-1, params='mc')
  g = mixture.DPGMM(n_components=5, alpha=10.0)
  g.fit(X)



  '''
  pickle.dump(bics, open("%s/tIC%d_gmm_bics.pkl" %(save_dir, tic_j), "wb"))
  plt.scatter(range(1,max_components),bics)
  pp = PdfPages("%s/tIC%d_gmm_bics.pdf" %(save_dir, tic_j))
  pp.savefig()
  pp.close()
  plt.clf()

  pickle.dump(aics, open("%s/tIC%d_gmm_aics.pkl" %(save_dir, tic_j), "wb"))
  plt.scatter(range(1,max_components),aics)
  pp = PdfPages("%s/tIC%d_gmm_aics.pdf" %(save_dir, tic_j))
  pp.savefig()
  pp.close()
  plt.clf()

  pickle.dump(test_likelihoods, open("%s/tIC%d_gmm_test_likelihoods.pkl" %(save_dir, tic_j), "wb"))
  plt.scatter(range(1,max_components),test_likelihoods)
  pp = PdfPages("%s/tIC%d_gmm_test_likelihoods.pdf" %(save_dir, tic_j))
  pp.savefig()
  pp.close()
  plt.clf()
  '''

  return(g)

def compute_gmm(tic_j_x_tuple, max_components, save_dir):
  print(("Analyzing tIC %d" %(tic_j_x_tuple[0])))
  model = select_model(tic_j_x_tuple[1].reshape(-1,1), tic_j_x_tuple[0], max_components, save_dir)
  with gzip.open("%s/tIC%d_gmm.pkl.gz" %(save_dir, tic_j_x_tuple[0]), "wb") as f:
    pickle.dump(model, f)
  return

def compute_gmms(projected_tica_coords, save_dir, max_components):
  tics = np.concatenate(load_file(projected_tica_coords))
  compute_gmm_partial = partial(compute_gmm, max_components = max_components, save_dir = save_dir)
  #for pair in [(j, tics[:,j]) for j in range(0,np.shape(tics)[1])]:
   # compute_gmm_partial(pair)
  pool = mp.Pool(mp.cpu_count())
  pool.map(compute_gmm_partial, [(j, tics[:,j]) for j in range(0,np.shape(tics)[1])])
  pool.terminate()
  return

def compute_gmm_R(tIC_j_x_tuple, max_components, save_dir):
  j = tIC_j_x_tuple[0]
  tIC = tIC_j_x_tuple[1]
  #gmm = r['compute.tIC.mixture.model'](tIC, j, save_dir, max_components=max_components, num_repeats=5)
  #gmm_dict = { key : np.array(gmm.rx2(key)) for key in gmm.names}
  #return(gmm_dict)

def compute_gmms_R(projected_tica_coords, max_components, save_dir, max_j=10):
  tics = np.concatenate(load_file(projected_tica_coords))
  #tics = tics[range(0,np.shape(tics)[0],10),:]
  compute_gmm_partial = partial(compute_gmm_R, max_components = max_components, save_dir = save_dir)
  #for pair in [(j, tics[:,j]) for j in range(0,np.shape(tics)[1])]:
   # compute_gmm_partial(pair)
  pool = mp.Pool(mp.cpu_count())
  gmms = pool.map(compute_gmm_partial, [(j, tics[:,j]) for j in range(0,max_j)])
  pool.terminate()

  with gzip.open("%s/all_gmms.pkl.gz" %save_dir, "wb") as f:
    pickle.dump(gmms, f)

  for j, gmm in enumerate(gmms):
    gmm_means = gmm['mu']
    with open("%s/tIC.%d_means.pkl" %(save_dir, j), "wb") as f:
      pickle.dump(gmm_means, f)

    gmm_classes = gmm['classes']
    with gzip.open("%s/tIC.%d_classes.pkl.gz" %(save_dir, j), "wb") as f:
      pickle.dump(gmm_classes, f)

  return(gmms)

def compute_rf_model(features, gmm):
  return

def plot_importances(feature_importances, save_dir, i, percentile=True):
  bar_width = 0.4
  opacity = 0.8
  index = np.arange(50)

  feature_importance_dict = {}

  if percentile:
    for feature_importance in feature_importances:
      residue_i = feature_importance[0].split("_")[0]
      residue_j = feature_importance[0].split("_")[1]
      importance = feature_importance[1]
      if residue_i not in list(feature_importance_dict.keys()):
        feature_importance_dict[residue_i] = []
      if residue_j not in list(feature_importance_dict.keys()):
        feature_importance_dict[residue_j] = []

      feature_importance_dict[residue_i].append(importance)
      feature_importance_dict[residue_j].append(importance)
    for residue in list(feature_importance_dict.keys()):
      feature_importance_dict[residue] = np.percentile(feature_importance_dict[residue], 95.0)

    feature_importances = [(key, value) for key, value in feature_importance_dict.items()]
    feature_importances.sort(key=operator.itemgetter(1),reverse=True)


  plt.barh(np.arange(50), [f[1] for f in feature_importances[0:50]], bar_width, alpha=opacity, color='b',label='Feature importance')
  plt.ylabel('Feature')
  plt.xlabel('Overall Importance')
  plt.title("Random-Forest + GMM Model of tIC %d Feature Importance" %(i+1))
  plt.yticks(index + bar_width, [f[0] for f in feature_importances[0:50]],fontsize=8)
  pp = PdfPages("%s/tIC%d_overall_importances.pdf" %(save_dir, i))
  pp.savefig()
  pp.close()
  plt.clf()

def plot_overall_rf_importances(rf_dir, feature_residues_map):
  filename = "%s/tIC%d_overall_rf.pkl" %(rf_dir, 0)
  j = 0
  feature_names = generate_features(feature_residues_map)
  feature_names = ["%d_%d" %(f[0], f[1]) for f in feature_names]

  while(os.path.exists(filename)):
    print(("Examining tIC %d" %(j+1)))
    with gzip.open(filename, "rb") as f:
      rf = pickle.load(f)

    feature_importances = list(zip(feature_names, rf.feature_importances_.astype(float).tolist()))
    feature_importances.sort(key=operator.itemgetter(1),reverse=True)
    print((feature_importances[0:10]))
    plot_importances(feature_importances, rf_dir, j)
    j += 1
    filename = "%s/tIC%d_overall_rf.pkl" %(rf_dir, j)


def plot_component_importances(df, save_dir, tic_i, component):
  df = df.sort_values(by="feature_importance")
  top_n = 50
  top_df = df.iloc[0:top_n]
  for i in range(0,top_n):
    if top_df.iloc[i]["component_mean"] < top_df.iloc[i]["non_component_mean"]:
        top_df.iloc[i][feature_importance] = top_df.iloc[i][feature_importance] * -1.0


  bar_width = 0.2
  plt.barh(np.arange(50), top_df["feature_importance"].tolist(), bar_width, alpha=opacity, color='b',label='Feature importance')
  bar_width = 0.2
  plt.xlabel('Feature')
  plt.ylabel('Overall Importance')
  plt.title("Random-Forest + GMM Model of tIC %d Feature Importance" %(tic_i+1))
  plt.xticks(index + bar_width, top_df["feature_name"].tolist(), rotation='vertical')
  pp = PdfPages("%s/tIC%d_c%d_vs_all_importances.pdf" %(save_dir, (tic_i+1), component))
  pp.savefig()
  pp.close()
  plt.clf()


def plot_importances_df(df, title, save_file):
  df = df.sort_values(by="feature_importance")
  bar_width = 0.2
  index = np.arange(start=1, stop=len(df.keys())+1, step=1)
  plt.barh(index, df["feature_importance"].tolist(), bar_width, alpha=opacity, color='b',label='Feature importance')
  plt.ylabel('tIC')
  plt.xlabel('Overall Importance')
  plt.title(title)
  plt.xticks(index + bar_width, top_df["feature_name"].tolist(), rotation='vertical')
  pp = PdfPages(save_file)
  pp.savefig()
  pp.close()
  plt.clf()


def plot_column_pair(i, num_columns, save_dir, titles, data, gmm_means, refcoords):
  for j in range(i+1, num_columns):
    plt.hexbin(data[:,i],  data[:,j], bins = 'log', mincnt=1)
    print(gmm_means)
    print((gmm_means[i]))
    print((gmm_means[j]))
    for mean in gmm_means[i]:
      plt.axvline(x=mean,color='k',ls='dashed')
    for mean in gmm_means[j]:
      plt.axhline(y=mean,color='k',ls='dashed')
    if refcoords is not None:
      plt.scatter([refcoords[0,i]], [refcoords[0,j]], marker = 's', c='w',s=15)
      plt.scatter([refcoords[1,i]], [refcoords[1,j]], marker = 'v', c='k',s=15)
    if titles is not None:
      pp = PdfPages("%s/%s_%s.pdf" %(save_dir, titles[i], titles[j]))
      plt.xlabel(titles[i])
      plt.ylabel(titles[j])
      pp.savefig()
      pp.close()
      plt.clf()
    else:
      pp = PdfPages("%s/tIC.%d_tIC.%d.pdf" %(save_dir, i+1, j+1))
      plt.xlabel("tIC.%d" %(i+1))
      plt.ylabel("tIC.%d" %(j+1))
      pp.savefig()
      pp.close()
      plt.clf()

def plot_tics_gmm(save_dir, data_file, gmm_dir, R=True, titles = None, tICA = False, scale = 1.0, refcoords_file = None):
  data = np.concatenate(load_file(data_file))

  if(refcoords_file is not None and os.path.exists(refcoords_file)):
    refcoords = load_file(refcoords_file)
  else:
    refcoords = None
  print((np.shape(refcoords)))
  print(refcoords)

  gmm_means = []
  if not R:
    for j in range(0,np.shape(data)[1]):
      with gzip.open("%s/tIC%d_gmm.pkl.gz" %(gmm_dir, j)) as f:
        gmm = pickle.load(f)
      gmm_means.append(gmm.means_)
  else:
    for j in range(0,np.shape(data)[1]):
      with open("%s/tIC.%d_means.pkl" %(gmm_dir, j)) as f:
        means = pickle.load(f)
      gmm_means.append(means)

  num_columns = np.shape(data)[1]
  plot_column_pair_partial = partial(plot_column_pair, num_columns = num_columns, save_dir = save_dir, titles = titles,
    data = data, gmm_means = gmm_means, refcoords = refcoords)
  #for i in range(0,num_columns):
  #  plot_column_pair_partial(i)
  pool = mp.Pool(mp.cpu_count())
  pool.map(plot_column_pair_partial, list(range(0,num_columns)))
  pool.terminate()

  print("Done plotting columns")
  return

def plot_tics_gmm_R(save_dir, data_file, gmm_dir, titles = None, tICA = False, scale = 1.0, refcoords_file = None):
  data = verboseload(data_file)
  data = np.concatenate(data)
  data[:,0] *= scale

  if(refcoords_file is not None):
    refcoords = load_file(refcoords_file)
  else:
    refcoords = None
  print((np.shape(refcoords)))
  print(refcoords)

  gmm_means = []
  for j in range(0,np.shape(data)[1]):
    with gzip.open("%s/tIC%d_gmm.pkl.gz" %(gmm_dir, j)) as f:
      gmm = pickle.load(f)
    gmm_means.append(gmm.means_)

  num_columns = np.shape(data)[1]
  plot_column_pair_partial = partial(plot_column_pair, num_columns = num_columns, save_dir = save_dir, titles = titles,
    data = data, gmm_means = gmm_means, refcoords = refcoords)
  #for i in range(0,num_columns):
  #  plot_column_pair_partial(i)
  pool = mp.Pool(mp.cpu_count())
  pool.map(plot_column_pair_partial, list(range(0,num_columns)))
  pool.terminate()

  print("Done plotting columns")
  return


def compute_one_vs_all_rf_models(features_dir, projected_tica_coords, gmm_dir, save_dir, R=True, n_trees = 10, n_tica_components=25, feature_residues_map=""):
  features = np.concatenate(load_file_list(None, directory = features_dir, ext = ".dataset"))
  tics = np.concatenate(load_file(projected_tica_coords))
  feature_names = generate_features(feature_residues_map)
  feature_names = ["%d_%d" %(f[0], f[1]) for f in feature_names]

  for i in range(0, n_tica_components):
    print(("Computing random forest model for tIC.%d" %(i+1)))

    if R:
      with gzip.open("%s/tIC.%d_classes.pkl.gz" %(gmm_dir, i), "rb") as f:
        Y = pickle.load(f)
    else:
      with gzip.open("%s/tIC%d_gmm.pkl.gz" %(gmm_dir, i), "rb") as f:
        gmm = pickle.load(f)
      Y = gmm.predict(tics[:,i].reshape(-1,1))

    n_components = len(np.unique(Y).tolist())
    print((np.unique(Y).tolist()))
    print(("n_components %d" %n_components))

    for component in np.unique(Y).tolist():
      print(("Analyzing component %d" %component))
      all_indices = list(range(0,np.shape(tics)[0]))
      component_indices = [k for k in all_indices if int(Y[k]) == component]
      non_component_indices = list(set(all_indices)-set(component_indices))
      non_component_indices.sort()
      print((len(component_indices)))
      print((len(non_component_indices)))
      print((len(component_indices)+len(non_component_indices)))
      print("Found indices")
      Z = copy.deepcopy(Y)
      Z[component_indices] = 0
      Z[non_component_indices] = 1
      rf = RandomForestClassifier(max_features="sqrt",bootstrap=True,n_estimators=n_trees,n_jobs=-1)
      print("fitting random forest model")
      r = rf.fit(features, Z)
      print("fit random forest model, saving now.")
      with gzip.open("%s/tIC%d_c%d_vs_all_rf.pkl.gz" %(save_dir, i, component), "wb") as f:
        pickle.dump(rf.feature_importances_.astype(float), f)
      print("Finished saving")

      feature_component_means = np.mean(features[component_indices,:], axis=0)
      print((np.shape(feature_component_means)))
      print((feature_component_means[0:100]))
      feature_non_component_means = np.mean(features[non_component_indices,:], axis=0)

      feature_importances = list(zip(feature_names, rf.feature_importances_.astype(float).tolist()))
      pickle.dump(feature_importances, open("%s/tIC%d_c%d_importances_list.pkl" %(save_dir, i, component), "wb"))
      '''
      df = pd.DataFrame(columns=('feature_name', 'feature_importance', 'component_mean', 'non_component_mean'))

      for i, feature_importance in enumerate(feature_importances):
        df.iloc[k] = [feature_importance[0], feature_importance[1], feature_component_means[i], feature_non_component_means[i]]

      pickle.dump(df, open("%s/tIC%d_c%d_vs_all_df.pkl" %(save_dir, (i+1), component), "wb"))
      try:
        plot_component_importances(df, save_dir, i, component)
      except:
        continue
    '''


  return

def compute_overall_rf_models(features_dir, projected_tica_coords, gmm_dir, save_dir, R=True, n_trees = 10, n_tica_components=25, feature_residues_map = ""):
  '''
  if not os.path.exists("%s/feature_subset.h5" %save_dir):
    features = np.concatenate(load_file_list(None, directory = features_dir, ext = ".dataset"))
    features = features[range(0,np.shape(features)[0],100),:]
    verbosedump(features, "%s/feature_subset.h5" %save_dir)
  else:
    features = verboseload("%s/feature_subset.h5" %save_dir)
  '''

  features = np.concatenate(load_file_list(None, directory = features_dir, ext = ".dataset"))

  tics = np.concatenate(load_file(projected_tica_coords))
  feature_names = generate_features(feature_residues_map)
  feature_names = ["%d_%d" %(f[0], f[1]) for f in feature_names]

  for i in range(0, n_tica_components):
    print(("Computing random forest model for tIC.%d" %(i+1)))

    if R:
      with gzip.open("%s/tIC.%d_classes.pkl.gz" %(gmm_dir, i), "rb") as f:
        Y = pickle.load(f)
    else:
      with gzip.open("%s/tIC%d_gmm.pkl.gz" %(gmm_dir, i), "rb") as f:
        gmm = pickle.load(f)
      Y = gmm.predict(tics[:,i].reshape(-1,1))
      if len(gmm.means_) == 1: continue

    rf = RandomForestClassifier(max_features="sqrt",bootstrap=True,n_estimators=n_trees,n_jobs=-1)
    r = rf.fit(features, Y)

    with gzip.open("%s/tIC%d_overall_rf.pkl" %(save_dir, i), "wb") as f:
      pickle.dump(rf, f)
    print("Saved RF model")

    feature_importances = list(zip(feature_names, rf.feature_importances_.astype(float).tolist()))
    feature_importances.sort(key=operator.itemgetter(1),reverse=True)

    print((feature_importances[0:20]))
    try:
      plot_importances(feature_importances, save_dir, i)
    except:
      continue




def plot_coef_path(df, filename):
  plt.figure()
  df[list(df.keys())[list(range(0,11))]].plot(colormap='gist_rainbow')
  plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
  pp = PdfPages(filename)
  pp.savefig(bbox_inches='tight')
  pp.close()
  plt.clf()

def plot_df_list_rolling(dfs, filename, return_fig=True, smoothing=100, include_original=False,
                         ref_df=None, xlabel="Time (nanoseconds)", custom_y_bounds=None, color="rainbow",
                         subplots=True):
  plt.clf()
  min_length = min([df.shape[0] for df in dfs])
  dfs = [df.iloc[:min_length] for df in dfs]
  with plt.style.context(('seaborn-whitegrid')):
    if subplots:
      fig, axes = plt.subplots(nrows=len(dfs[0].columns.values), ncols=1, figsize=(8.0, len(dfs[0].columns.values)*3.0))
      plt.ylabel("Distance (Angstroms)")

      for i,var in enumerate(dfs[0].columns.values):
        if ref_df is not None:
          try:
            dot_df = pd.DataFrame(ref_df[var])
            dot_df = pd.concat([dot_df.transpose()]*dfs[0].shape[0], axis=0)
            dot_df.index = dfs[0].index
            dot_df[dot_df.columns.values[0]].plot(ax=axes[i], c="blue", linestyle='dashed', label="", title="")
            dot_df[dot_df.columns.values[1]].plot(ax=axes[i], c="green", linestyle='dashed', label="", title="")
          except:
            print("Feature does not have reference crystal values.")
        if include_original:
          color=iter(cm.inferno(np.linspace(0.2,0.6,len(dfs))))
          for df in dfs:
            c = next(color)
            if custom_y_bounds is None:
              df[var].plot(ax=axes[i], title="", label="", c=c, alpha=0.2)
            else:
              df[var].plot(ax=axes[i], title="", label="", c=c, alpha=0.2, ylim=custom_y_bounds[i])
        
        color=iter(cm.inferno(np.linspace(0.2, 0.6,len(dfs))))

        for df in dfs:
          c = next(color)
          pd.rolling_mean(df, smoothing, center=True, min_periods=None)[var].plot(ax=axes[i], linewidth=2.5, title=var, c=c)


        axes[i].legend(loc='center left', bbox_to_anchor=(1.0,0.5))
        axes[i].set_title("")

      #axes[0].set_ylabel('Distance in Angstroms')
      #pd.rolling_mean(df, 100).plot(colormap='gist_rainbow', subplots=True, ax=f.gca(), layout=(1,5))
    else:
      fig = plt.figure()

      for df in dfs:
        pd.rolling_mean(df, smoothing).plot(colormap='gist_rainbow', subplots=True, ax=fig.gca())

    #plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
    #plt.rc('text', usetex=True)
    plt.xlabel(xlabel)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    #pp = PdfPages(filename)
    #pp.savefig(bbox_inches='tight')
    #pp.close()
    if return_fig:
      return

    plt.clf()


def plot_df_rolling(df, filename, return_fig=True, subplots=True, smoothing=100, include_original=False, min_periods=1,
                    ref_df=None, color="rainbow", xlabel="Time (microseconds)", custom_y_bounds=None):

  plt.clf()
  with plt.style.context(('seaborn-whitegrid')):
    if subplots:
      if color == 'rainbow':
        n_colors_i = np.floor(len(df.columns.values) / 2.)
        n_colors_j = np.ceil(len(df.columns.values)/2.)
        color = iter(cm.gist_rainbow(np.concatenate([np.linspace(0., 0.4, n_colors_i), np.linspace(.6, 1.0, n_colors_j)])))
      else:
        color=iter(ListedColormap(sns.hls_palette(8, l=.3, s=.8))(np.linspace(0.2,0.8,len(df.columns.values))))
        #color=iter(cm.get_cmap(color)(np.linspace(0.2,0.8,len(df.columns.values))))
      fig, axes = plt.subplots(nrows=len(df.columns.values), ncols=1, figsize=(8.0, len(df.columns.values)*3.0))
      plt.ylabel("Distance (Angstroms)")

      for i,var in enumerate(df.columns.values):
        c = next(color)
        if ref_df is not None:
          try:
            dot_df = pd.DataFrame(ref_df[var])
            dot_df = pd.concat([dot_df.transpose()]*df.shape[0], axis=0)
            dot_df.index = df.index
            dot_df[dot_df.columns.values[0]].plot(ax=axes[i], c="blue", linestyle='dashed', label="", title="")
            dot_df[dot_df.columns.values[1]].plot(ax=axes[i], c="green", linestyle='dashed', label="", title="")
          except:
            print("Feature does not have reference crystal values.")
        if include_original:
          if custom_y_bounds is None:
            df[var].plot(ax=axes[i], title="", label="", c=c, alpha=0.2, ylim=[df[var].values.min(), df[var].values.max()])
          else:
            df[var].plot(ax=axes[i], title="", label="", c=c, alpha=0.2, ylim=custom_y_bounds[i])

        pd.rolling_mean(df, smoothing, center=True, min_periods=None)[var].plot(ax=axes[i], linewidth=2.5, title=var, c=c)


        axes[i].legend(loc='center left', bbox_to_anchor=(1.0,0.5))
        axes[i].set_title("")

      #axes[0].set_ylabel('Distance in Angstroms')
      #pd.rolling_mean(df, 100).plot(colormap='gist_rainbow', subplots=True, ax=f.gca(), layout=(1,5))
    else:
      fig = plt.figure()
      pd.rolling_mean(df, smoothing).plot(colormap='gist_rainbow', subplots=True, ax=fig.gca())

    #plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
    #plt.rc('text', usetex=True)
    plt.xlabel(xlabel)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    #pp = PdfPages(filename)
    #pp.savefig(bbox_inches='tight')
    #pp.close()
    if return_fig:
      return

    plt.clf()



def compute_docking_cutoff(docking_csv, plot_file):
  docking = pd.read_csv(docking_csv, header=0, index_col=0)
  plt.figure()
  docking.plot(kind='hist', bins=50)
  plt.xlabel("Docking DeltaG")
  plt.ylabel("Frequency")
  plt.title("Histogram of Docking Scores")
  pp = PdfPages(plot_file)
  pp.savefig()
  pp.close()
  plt.clf()


'''ref: http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#example-linear-model-plot-lasso-model-selection-py'''
def rank_tICs_by_logistic_lasso(docking_csv, tica_coords_csv, plot_file, cutoff):
  docking = pd.read_csv(docking_csv, header=0, index_col=0)
  tica = pd.read_csv(tica_coords_csv, header=0, index_col=0)
  tica = tica.loc[docking.index]

  tica_names = list(tica.keys())
  for i, name in enumerate(tica_names):
    tica_names[i] = "tIC%d" % (i+1)

  docking_values = docking.values
  tica_values = tica.values
  #docking_values[docking_values <= cutoff] = 0.0
  #docking_values[docking_values > cutoff] = 1.0
  #print(docking_values)
  #print(np.max(docking_values))
  tica_values = preprocessing.normalize(tica.values)

  X = diabetes.data

  y = diabetes.target

  X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

  # Compute paths

  eps = 5e-3  # the smaller it is the longer is the path

  print("Computing regularization path using the lasso...")
  alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)

  print("Computing regularization path using the positive lasso...")
  alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
      X, y, eps, positive=True, fit_intercept=False)
  print("Computing regularization path using the elastic net...")
  alphas_enet, coefs_enet, _ = enet_path(
      X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)

  print("Computing regularization path using the positve elastic net...")
  alphas_positive_enet, coefs_positive_enet, _ = enet_path(
      X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

  # Display results

  plt.figure(1)
  ax = plt.gca()
  ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
  l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
  l2 = plt.plot(-np.log10(alphas_enet), coefs_enet.T, linestyle='--')

  plt.xlabel('-Log(alpha)')
  plt.ylabel('coefficients')
  plt.title('Lasso and Elastic-Net Paths')
  plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
  plt.axis('tight')


def compute_one_vs_all_rf_models_MSM(features_dir, projected_tica_coords, clusterer_dir, msm_rf_dir, feature_residues_map, n_trees=10, states_to_analyze=None):
  feature_names = generate_features(feature_residues_map)
  feature_names = ["%s_%s" %(f[0].res_name.title(), f[1].res_name.title())
                                for f in feature_names]
  feature_files = get_trajectory_files(features_dir, ext=".dataset")
  #features = load_file(feature_files[0])
  features = np.concatenate(load_file_list(None, directory = features_dir, ext = ".dataset"))
  clusterer = verboseload(clusterer_dir)
  Y = np.concatenate(verboseload(clusterer_dir).labels_)
  #Y = verboseload(clusterer_dir).labels_[0]

  print("Y has shape:")
  print((np.shape(Y)))

  if states_to_analyze is None:
    states_to_analyze = list(range(0, cluster.n_clusters))
  for state in states_to_analyze:
    print(("Computing random forest model for state %d" %(state)))
    all_indices = list(range(0,np.shape(Y)[0]))
    state_indices = [k for k in all_indices if int(Y[k]) == state]
    non_state_indices = list(set(all_indices)-set(state_indices))
    non_state_indices.sort()
    Z = copy.deepcopy(Y)
    Z[state_indices] = 0
    Z[non_state_indices] = 1
    rf = RandomForestClassifier(max_features="sqrt",bootstrap=True,n_estimators=n_trees,n_jobs=-1,verbose=1)
    print("fitting random forest model")
    r = rf.fit(features, Z)
    print("fit random forest model, saving now.")
    with gzip.open("%s/msm_state%d_vs_all_rf.pkl.gz" %(msm_rf_dir, state), "wb") as f:
      pickle.dump(rf.feature_importances_.astype(float), f)
    print("Finished saving")

    feature_state_means = np.mean(features[state_indices,:], axis=0)
    feature_non_state_means = np.mean(features[non_state_indices,:], axis=0)

    feature_importances = list(zip(feature_names, rf.feature_importances_.astype(float).tolist()))
    pickle.dump(feature_importances, open("%s/msm_state%d_importances_list.pkl" %(msm_rf_dir, state), "wb"))

    df = pd.DataFrame(columns=('feature_name', 'feature_importance', 'component_mean', 'non_component_mean'))

    for i, feature_importance in enumerate(feature_importances):
      df.loc[i] = [feature_importance[0], feature_importance[1], feature_state_means[i], feature_non_state_means[i]]

    pickle.dump(df, open("%s/msm_state_%d_vs_all_df.pkl" %(msm_rf_dir, state), "wb"))
    '''
    try:
      plot_component_importances(df, save_dir, i, component)
    except:
      continue
    '''


  return

def interpret_msm_rf(msm_dir, feature_residues_pkl, n_msm_states=25, percentile=99.9999):
  for j in range(0, n_msm_states):
    print(("Interpreting MSM state %d" %(j+1)))
    importances_file = "%s/msm_state%d_vs_all_rf.pkl.gz" %(msm_dir, j)
    if os.path.exists(importances_file):
      feature_importances_df = merge_importances_features(importances_file, feature_residues_pkl)
      residue_importances_df = compute_per_residue_importance(feature_importances_df, percentile)

      plot_importance_df(feature_importances_df, "feature_name", "Contact",
                         "MSM %d Contact Importances" % (j+1),
                         "msm_state",
                         "per_contact_importances", j, msm_dir)

      plot_importance_df(residue_importances_df, "residue", "Residue",
                         "MSM %d Residue Importances" % (j+1),
                         "msm_state",
                         "per_residue_importances", j, msm_dir)

      feature_means_file = "%s/msm_state_%d_vs_all_df.pkl" %(msm_dir, j)
      if os.path.exists(feature_means_file):
        with open(feature_means_file, "rb") as f:
          feature_means_df = pickle.load(f)
        feature_importances_new = []
        for index, row in feature_means_df.iterrows():
          if row["component_mean"] < row["non_component_mean"]:
            feature_importances_new.append(-1.0 * row["feature_importance"])
          else:
            feature_importances_new.append(row["feature_importance"])
        feature_means_df["feature_importance"] = feature_importances_new
        feature_means_df.columns = ["contact", "importance", "state_mean", "non_state_mean"]
        plot_importance_df(feature_means_df, "contact", "Contact",
                           "MSM State %d Contact Changes" % (j+1),
                           "msm_state",
                           "per_contact_changes", j, msm_dir)

      with open("%s/MSM_state%d_rfr_feature_importance_df.pkl" %(msm_dir, j), "wb") as f:
        pickle.dump(feature_importances_df, f)

      with open("%s/MSM_state%d_rfr_residue_importance_df.pkl" %(msm_dir, j), "wb") as f:
        pickle.dump(residue_importances_df, f)


def plot_tICs_vs_docking(docking_csv, tica_coords_csv, plot_file, chosen_ligand="docking", max_tIC=10):
  docking = pd.read_csv(docking_csv, header=0, index_col=0)
  docking.columns = [c.rstrip().lstrip() for c in docking.columns]
  tica = pd.read_csv(tica_coords_csv, header=0, index_col=0)
  tica = tica.loc[docking.index]
  tica[list(tica.keys())] = preprocessing.normalize(tica.values)

  tica_names = list(tica.keys())
  for i, name in enumerate(tica_names):
    tica_names[i] = "tIC%d" % (i+1)

  print(docking)
  print((list(docking.keys())))
  if chosen_ligand is not "docking":
    print((docking[chosen_ligand]))
    docking = pd.DataFrame(docking[chosen_ligand].values, columns=[chosen_ligand])
  index = pd.Index(docking.values).astype(np.float64)
  df = pd.DataFrame(np.hstack((tica.values,docking.values)), columns=tica_names+[chosen_ligand])
  df.sort(columns=chosen_ligand, inplace=True)
  df = pd.DataFrame(df[tica_names].values[:,list(range(0,max_tIC))], index=df[chosen_ligand], columns=list(range(1,max_tIC + 1)))
  plot_df_rolling(df, plot_file)


def compute_and_plot_single_kde(data, title, xlabel,
                                fig_file=None, custom_bounds=None,
                                custom_y_bounds=None, n_points=200,
                                convert_to_energy=False, crystal_values=None):
  try:
    with plt.style.context(('seaborn-whitegrid')):
      kde = stats.gaussian_kde(data)
      if custom_bounds is not None:
        x = np.linspace(custom_bounds[0], custom_bounds[1], n_points)
      else:
        x = np.linspace(0.8*data.min(), 1.2*data.max(),n_points)
      dx = kde(x)
      if convert_to_energy:
        dx *= -0.61
      plt.plot(x,dx,color='black')
      print("plotted")
      plt.xlim(x.min(), x.max())
      if custom_y_bounds is not None:
        plt.ylim(custom_y_bounds[0], custom_y_bounds[1])
      plt.fill_between(x,0,dx, alpha=.75,color='#5673E0')
      plt.xlabel(xlabel)
      if convert_to_energy:
        plt.ylabel("delta G")
      else:
        plt.ylabel("Estimated Equilibrium Population")
      plt.title(title)
      if fig_file is not None:
        plt.savefig(fig_file)
      if crystal_values is not None:
        plt.axvline(x=crystal_values[0], color='b', ls='dashed')
        plt.axvline(x=crystal_values[1], color='g', ls='dashed')
      plt.show()
  except:
    print("Couldn't plot, likely due to singular matrix.")

def compute_kde_difference(data_i, data_j, custom_bounds=None, n_points=200):
  kde_i = stats.gaussian_kde(data_i)
  kde_j = stats.gaussian_kde(data_j)
  if custom_bounds is not None:
    x = np.linspace(custom_bounds[0], custom_bounds[1], n_points)
  else:
    x = np.linspace(0.8*min(data_i.min(), data_j.min()), 1.2*(max(data_j.max(), data_i.max())),200)
  dx_i = kde_i(x)
  dx_j = kde_j(x)
  dx = dx_i - dx_j
  return x, dx

def compute_and_plot_kde_difference(data_i, data_j, title, xlabel,
                                    fig_file=None, custom_bounds=None,
                                    custom_y_bounds=None,
                                    crystal_values=None,
                                    show_plot=True):
  try:
    with plt.style.context(('seaborn-whitegrid')):
      kde_i = stats.gaussian_kde(data_i)
      kde_j = stats.gaussian_kde(data_j)
      if custom_bounds is not None:
        x = np.linspace(custom_bounds[0], custom_bounds[1], 200)
      else:
        x = np.linspace(0.8*data_i.min(), 1.2*data_j.max(),200)
      dx_i = kde_i(x)
      dx_j = kde_j(x)

      print(dx_i.shape)
      print(dx_j.shape)

      dx = dx_i - dx_j

      print(dx.shape)

      plt.plot(x,dx,color='black')
      plt.xlim(x.min(), x.max())
      if custom_y_bounds is not None:
        plt.ylim(custom_y_bounds[0], custom_y_bounds[1])
      plt.fill_between(x,0,dx, alpha=.75,color='#5673E0')
      plt.xlabel(xlabel)
      plt.ylabel("Estimated Equilibrium Population")
      plt.title(title)

      if crystal_values is not None:
        plt.axvline(x=crystal_values[0], color='b', ls='dashed')
        plt.axvline(x=crystal_values[1], color='g', ls='dashed')

      if fig_file is not None:
        plt.savefig(fig_file)
      if show_plot:
        plt.show()
  except:
    print("Couldn't plot, likely due to singular matrix.")

def get_ori_feature_name(feature_name):
  ori_feature = copy.deepcopy(feature_name)
  if ori_feature.count("<") == 2:
    ori_feature = ori_feature.split(" < ")[1]
  elif ori_feature.count("<") == 1:
    ori_feature = ori_feature.split(" < ")[0]
  elif ">" in ori_feature:
    ori_feature = ori_feature.split(" > ")[0]
  return ori_feature

def plot_lp_pp_model_outcomes(importances_df, scores, save_dir, traj_features_df, onehot_features_df, n_pp_features=25,
                              cutoff=5.,
                              n_lp_features=25, protein_features=None, ref_features_df=None,
                              show_plot=False, redo=True):
  if protein_features is None:
    order = np.argsort(np.abs(scores))
    pp_features = importances_df.columns.values[order]
    pp_features = pp_features[:min(25, len(pp_features))]
  else:
    pp_features = []
    for protein_feature in protein_features:
      pp_features += [n for n in importances_df.columns.values.tolist() if protein_feature in n]
    print(pp_features)


  for j, pp_feature in enumerate(pp_features):
    print(pp_feature)
    ori_pp_feature = get_ori_feature_name(pp_feature)
    print(ori_pp_feature)

    custom_bounds = [0.8*traj_features_df[ori_pp_feature].min(),
                     1.2*traj_features_df[ori_pp_feature].max()]

    pp_feature_dir = "%s/%s" %(save_dir, pp_feature)
    if not os.path.exists(pp_feature_dir):
      os.makedirs(pp_feature_dir)

    df = importances_df[pp_feature].abs().sort(inplace=False, ascending=False)
    df = importances_df[pp_feature].loc[[n for n in 
                                        df.index.values.tolist() if "<" in n]].iloc[:n_lp_features]
    df = df.loc[df > 0.]
    print(df)

    if ref_features_df is not None:
      crystal_values = ref_features_df[ori_pp_feature].values.flatten()

    for i in range(0, df.shape[0]):
      lig_feature = df.index.values.tolist()[i].replace("[", "").replace("]", "")
      print(lig_feature)
      #lig_feature = get_ori_feature_name(lig_feature)

      save_pdf = "%s/%s_difference.pdf" %(pp_feature_dir, lig_feature)
      save_png = "%s/%s_difference.png" %(pp_feature_dir, lig_feature)

      if os.path.exists(save_pdf) and not redo:
        continue


      data_i = traj_features_df.loc[onehot_features_df[lig_feature] == 1][ori_pp_feature].values
      data_j = traj_features_df.loc[onehot_features_df[lig_feature] == 0][ori_pp_feature].values

      print(ori_pp_feature)

      title = lig_feature.replace("Lig900", "BU72")
      title = lig_feature.replace("Lig1", "SUF")

      title_j = "Not %s" %df.index.values[i].replace("Lig900", "BU72").replace("Lig1", "SUF")
          #compute_and_plot_single_kde(data_i, title_i,
          #                            "Phe289 to Asn150 Distance, Angstroms", "%s/%s.pdf" %(analysis_dir,title_i), custom_bounds=[5,16], custom_y_bounds=[0,.6])
          #compute_and_plot_single_kde(data_j, title_j,
          #                            "Phe289 to Asn150 Distance, Angstroms", "%s/%s.pdf" %(analysis_dir,title_j), custom_bounds=[5,16], custom_y_bounds=[0,.6])
      plt.clf()
      compute_and_plot_kde_difference(data_i, data_j, title, 
                                      "%s Distance, Angstroms" %ori_pp_feature, 
                                      save_pdf,
                                      custom_bounds=custom_bounds, custom_y_bounds=[-0.7,0.7],
                                      crystal_values=crystal_values,
                                      show_plot=show_plot)

def sample_tIC_regions(tica_coords, save_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  print(np.shape(tica_coords))
  for j in range(0, np.shape(tica_coords)[1]):
    print("Fitting KDE to tIC %d" % (j+1))
    grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1, 1.0, 10)},
                    cv=5, n_jobs=-1) # 20-fold cross-validation
    tic_coords = tica_coords[:,j].reshape(-1, 1)
    grid.fit(tic_coords)
    print(grid.best_params_)
    kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_params_["bandwidth"]).fit(tic_coords)
    verbosedump(kde, os.path.join(save_dir, "kde_model_tIC_%d.h5" % (j+1)))

    s = np.linspace(np.min(tic_coords), np.max(tic_coords), 100)
    e = kde.score_samples(s.reshape(-1,1))

    plt.plot(s, e)
    pp = PdfPages(os.path.join(save_dir, "tIC_%d_kde.pdf" %(j+1)))
    plt.title("Kernel Density Estimator for tIC %d" % (j+1))
    pp.savefig()
    pp.close()
    plt.clf()

def sample_tIC_regions_silverman(tica_coords, save_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  print(np.shape(tica_coords))
  for j in range(0, np.shape(tica_coords)[1]):
    print("Fitting KDE to tIC %d" % (j+1))
    tic_coords = tica_coords[:,j]
    bandwidth = 1.06 * np.std(tic_coords) * np.power(np.shape(tic_coords)[0], -.2)
    tic_coords = tica_coords[:,j].reshape(-1, 1)
    print("bandwidth = %f" % bandwidth)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(tic_coords)
    verbosedump(kde, os.path.join(save_dir, "kde_model_tIC_%d.h5" % (j+1)))

    s = np.linspace(np.min(tic_coords), np.max(tic_coords), 100)
    e = kde.score_samples(s.reshape(-1,1))

    plt.plot(s, e)
    pp = PdfPages(os.path.join(save_dir, "tIC_%d_kde.pdf" %(j+1)))
    plt.title("Kernel Density Estimator for tIC %d" % (j+1))
    pp.savefig()
    pp.close()
    plt.clf()

#thanks to this for code suggestion:
#http://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-one-1d-clustering-method-in-sci
def get_kde_mins_and_maxes(tica_coords, kde_dir):
  print(np.shape(tica_coords))

  j = 0
  kde_file = os.path.join(kde_dir, "kde_model_tIC_%d.h5" % (j+1))
  while(os.path.exists(kde_file)):
    print("Finding maxima for tIC %d" % (j+1))
    tic_coords = tica_coords[:,j].reshape(-1,1)
    s = np.linspace(np.min(tic_coords), np.max(tic_coords), 1000)
    kde = verboseload(kde_file)
    e = kde.score_samples(s.reshape(-1,1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    minima = s[mi]
    maxima = s[ma]

    print("minima:")
    print(minima)
    print("maxima")
    print(maxima)

    verbosedump(minima, os.path.join(kde_dir, "kde_model_tIC_%d_minima.h5" % (j+1)))
    verbosedump(maxima, os.path.join(kde_dir, "kde_model_tIC_%d_maxima.h5" % (j+1)))
    j += 1
    kde_file = os.path.join(kde_dir, "kde_model_tIC_%d.h5" % (j+1))

def multi_binarizer(data, boundaries):
  new_data = np.zeros(data.shape)
  new_data[np.where(data < boundaries[0])[0]] = 0.
  new_data[np.where(data > boundaries[len(boundaries)-1])[0]] = len(boundaries)
  for i, minimum in enumerate(boundaries):
    if i == (len(boundaries)-1): continue
    new_data[np.where((data > boundaries[i]) & (data < boundaries[i+1]))[0]] = i + 1
  return new_data

def name_bin(old_name, boundaries, binary_keep_one=True):
  names = []
  names.append("%s < %f" %(old_name, boundaries[0]))
  if len(boundaries) == 1 and binary_keep_one:
    return names 

  for i in range(0, len(boundaries)-1):
    names.append("%f < %s < %f" %(boundaries[i], old_name, boundaries[i+1]))
  names.append("%s > %f" %(old_name, boundaries[len(boundaries)-1]))
  return names

def multi_onehot(data, boundaries, binary_keep_one=True):
  new_data = np.zeros((data.shape[0], len(boundaries)+1))
  new_data[np.where(data < boundaries[0])[0], 0] = 1.
  if len(boundaries) == 1 and binary_keep_one:
    return new_data[:,0].reshape((-1,1))

  new_data[np.where(data > boundaries[len(boundaries)-1])[0], len(boundaries)] = 1.
  for i, minimum in enumerate(boundaries):
    if i == (len(boundaries)-1): continue
    new_data[np.where((data > boundaries[i]) & (data < boundaries[i+1]))[0], i+1] = 1.
  return new_data

def get_kde_and_multi_onehot(data, names=None, binary_keep_one=True):
  new_data = []
  new_names = []
  for j in range(0, data.shape[1]):
    print(j)
    column = data[:,j]
    kde_mins = get_kde_mins(column)
    new_data.append(column, kde_mins)
    if names is not None:
      new_names.append(name_bin(names[j], kde_mins, binary_keep_one))
  return new_data, new_names

def multi_onehot_trajectories(data, names, custom_bounds=None,
                              subsample=1, binary_keep_one=True,
                              bandwidth=None):
  data_conc = np.concatenate(data)[::subsample]
  new_data = []
  new_names = []
  all_bounds = []
  for j in range(0, data_conc.shape[1]):
    if custom_bounds is not None:
      bounds = custom_bounds
      all_bounds.append(bounds)
    else:
      print(j)
      column = data_conc[:,j]
      bounds = get_kde_mins(column, bandwidth=bandwidth)
      if len(bounds) == 0:
        bounds = [np.average(column)]
      all_bounds.append(bounds)
    if names is not None:
      new_names += name_bin(names[j], bounds, binary_keep_one)

  for traj in data:
    new_traj = [multi_onehot(traj[:,j], all_bounds[j], binary_keep_one) for j in range(0, data_conc.shape[1])]
    new_traj = np.concatenate(new_traj, axis=1)
    new_data.append(new_traj)

  new_dfs = [pd.DataFrame(t, columns=new_names) for t in new_data]

  return new_data, new_names, new_dfs

def multi_onehot_trajectories_gmm(data, names, subsample=1, n_components=4):
  data_conc = np.concatenate(data)[::subsample]
  new_data = []
  new_names = []
  all_means = []
  gmms = []
  for j in range(0, data_conc.shape[1]):
    print(j)
    column = data_conc[:,j]
    gmm_model = mixture.GMM(n_components=n_components)
    gmm_model.fit(column.reshape((-1,1)))
    gmms.append(gmm_model)
    means = gmm_model.means_
    all_means.append(means)
    for mean in means:
      if names is not None:
        new_names.append("%s_mean%.2f" %(names[j], mean))
      else:
        new_names.append("")

  for traj in data:
    new_traj = []
    for j in range(0, data_conc.shape[1]):
      transformed_traj = gmms[j].predict(traj[:,j].reshape((-1,1)))
      transformed_traj = convert_to_n_class(transformed_traj, n_classes=len(all_means[j]))
      new_traj.append(transformed_traj)
    new_traj = np.concatenate(new_traj, axis=1)
    new_data.append(new_traj)

  new_dfs = [pd.DataFrame(t, columns=new_names) for t in new_data]

  return new_data, new_names, new_dfs


def make_edge_list(scores, importances_df, cutoff_score=0.8, cutoff_importance=0.):
    importances_arr = importances_df.values
    edge_tuples = []
    x_names = importances_df.index.values
    y_names = importances_df.columns.values
    for j, score in enumerate(scores):
        print(score)
        if score > cutoff_score:
            print(j)
            importances = importances_arr[:,j]
            for i, importance in enumerate(importances):
                if np.abs(importance) > cutoff_importance:
                    edge_tuples.append((x_names[i], y_names[j], importances[i]*score))
    edge_df = pd.DataFrame(edge_tuples, columns=["feature_i", "feature_j", "importance"])
    return edge_df

def get_kde_mins(data, bandwidth=None):
  kde = stats.gaussian_kde(data, bw_method=bandwidth)
  x = np.linspace(data.min(), data.max(),200)
  dx = kde(x)
  mi, ma = argrelextrema(dx, np.less)[0], argrelextrema(dx, np.greater)[0]
  minima = x[mi]
  return minima

def sample_kde_maxima(tica_coords, kde_dir, trajs):
  print(np.shape(tica_coords))

  j = 0
  kde_maxima_file = os.path.join(kde_dir, "kde_model_tIC_%d_maxima.h5" % (j+1))
  while(os.path.exists(kde_maxima_file)):
    tic_coords = [tic_coord[:,j].reshape(-1,1) for tic_coord in tica_coords]
    print(np.shape(tic_coords))
    print(np.shape(tic_coords[0]))
    maxima = verboseload(kde_maxima_file)
    #point = np.zeros(np.shape(tica_coords[0])[1])
    for i in range(0, np.shape(maxima)[0]):
      maximum = maxima[i]
      #point[j] = maximum
      point = [maximum]
      distances, indices = KDTree(tic_coords).query(point, 5)
      print("Indices:")
      print(indices)
      for ind, index in enumerate(indices):
        print(index)
        sample = md.load_frame(trajs[index[0]], index=index[1])
        sample.save(os.path.join(kde_dir, "tIC_%d_%s_%d.pdb" %((j+1), str(maximum), ind)))


    j += 1
    kde_maxima_file = os.path.join(kde_dir, "kde_model_tIC_%d_maxima.h5" % (j+1))

def calculate_cluster_averages_per_feature(clusterer, features):
  """
  Concatenate clusters into one long list
  Concatenate features
  for each cluster:
    np.where
    then get those features
    np.mean over axis=0
    store in new matrix
  """
  n_clusters = clusterer.n_clusters
  concatenated_clusters = np.concatenate(clusterer.labels_)
  concatenated_features = np.concatenate(features)
  cluster_averages = np.zeros((n_clusters, concatenated_features.shape[1]))
  for i in range(0, n_clusters):
    rows = np.where(concatenated_clusters == i)[0]
    means = np.mean(concatenated_features[rows,:], axis=0)
    cluster_averages[i,:] = means
  return cluster_averages

def find_most_populated_intermediates(msm_object, intermediates):
  msm_intermediates = np.concatenate(msm_object.partial_transform(intermediates))
  order = np.argsort(msm_object.populations_[msm_intermediates])
  print(np.sum(msm_object.populations_[msm_intermediates][order]))
  print(intermediates[order])
  return(intermediates[order])

def convert_to_n_class(values, n_classes=None):
  if n_classes is None:
    return np.eye(len(set(values.tolist())))[values.tolist()]
  else:
    return np.eye(n_classes)[values.tolist()]



