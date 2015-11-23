from sklearn import mixture
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import matplotlib.pyplot as plt
from msmbuilder.dataset import dataset, _keynat, NumpyDirDataset
from msmbuilder.utils import verbosedump, verboseload
import time
import random
import os
import multiprocessing as mp
import csv
from io_functions import *
from functools import partial
import gzip, pickle
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import operator

from pandas import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com
from rpy2.robjects import r
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()
base = get_base()
R_functions = "%s/conformation/analysis.R" %base
R_analysis = "%s/conformation/b2ar_analysis.R" %base
ro.r.source(R_functions)
ro.r.source(R_analysis)


def select_model(X, tic_j, max_components, save_dir):
  num_trials = 0
  possible_components = []
  for trial in range(0,num_trials):
    train_indices = random.sample(range(0,len(X)), len(X)/2)
    test_indices = list(set(range(0,len(X))) - set(train_indices))
    X_train = X[train_indices,:]
    X_test = X[test_indices,:]
    
    bics = []
    aics = []
    test_likelihoods = []
    models = []
    for n_components in range(max_components, max_components):
        print("For tIC %d looking at GMM model with %d components" %(tic_j, n_components))
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
  print("Analyzing tIC %d" %(tic_j_x_tuple[0]))
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
  gmm = r['compute.tIC.mixture.model'](tIC, j, save_dir, max_components=10, num_repeats=5)
  return(gmm)

def compute_gmms_R(projected_tica_coords, max_components, save_dir, max_j=10):
  tics = np.concatenate(load_file(projected_tica_coords))
  tics = tics[range(0,np.shape(tics)[0],100),:]
  compute_gmm_partial = partial(compute_gmm_R, max_components = max_components, save_dir = save_dir)
  #for pair in [(j, tics[:,j]) for j in range(0,np.shape(tics)[1])]:
   # compute_gmm_partial(pair)
  pool = mp.Pool(mp.cpu_count())
  gmms = pool.map(compute_gmm_partial, [(j, tics[:,j]) for j in range(0,max_j)])
  pool.terminate()

  with gzip.open("%s/all_gmms.pkl.gz" %save_dir, "wb") as f:
    pickle.dump(gmms, f)

  for j, gmm in enumerate(gmms):
    gmm_dict = { key : classes.rx2(key) for key in gmm.names}
    gmm_means = np.array(gmm_dict['mu'])
    print("Means for component %d" %j)
    print(gmm_means)
    with open("%s/tIC.%d_means.pkl" %(save_dir, j), "wb") as f:
      pickle.dump(gmm_means, f)

  return(gmms)

def compute_rf_model(features, gmm):
  return

def plot_importances(feature_importances, save_dir, i):
  plt.bar(np.arange(50), [f[1] for f in feature_importances[0:50]], bar_width, alpha=opacity, color='b',label='Feature importance')
  bar_width = 0.2
  plt.xlabel('Feature')
  plt.ylabel('Overall Importance')
  plt.title('Random-Forest + GMM Model of tIC Feature Importance')
  plt.xticks(index + bar_width, [f[0] for f in feature_importances[0:50]], rotation='vertical')
  pp = PdfPages("%s/tIC%d_overall_importances.pdf" %(save_dir, i))
  pp.savefig()
  pp.close()
  plt.clf()

def plot_component_importances(df, save_dir, tic_i, component):
  df = df.sort_values(by="feature_importance")
  top_n = 50
  top_df = df.iloc[0:top_n]
  for i in range(0,top_n):
    if top_df.iloc[i]["component_mean"] < top_df.iloc[i]["non_component_mean"]:
        top_df.iloc[i][feature_importance] = top_df.iloc[i][feature_importance] * -1.0


  bar_width = 0.2
  plt.bar(np.arange(50), top_df["feature_importance"].tolist(), bar_width, alpha=opacity, color='b',label='Feature importance')
  bar_width = 0.2
  plt.xlabel('Feature')
  plt.ylabel('Overall Importance')
  plt.title('Random-Forest + GMM Model of tIC Feature Importance')
  plt.xticks(index + bar_width, top_df["feature_name"].tolist(), rotation='vertical')
  pp = PdfPages("%s/tIC%d_c%d_vs_all_importances.pdf" %(save_dir, (tic_i+1), component))
  pp.savefig()
  pp.close()
  plt.clf()

def plot_column_pair(i, num_columns, save_dir, titles, data, gmm_means, refcoords):
  for j in range(i+1, num_columns):
    plt.hexbin(data[:,i],  data[:,j], bins = 'log', mincnt=1)
    print(gmm_means)
    print(gmm_means[i])
    print(gmm_means[j])
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
  data = verboseload(data_file)
  data = np.concatenate(data)
  data[:,0] *= scale

  if(refcoords_file is not None):
    refcoords = load_file(refcoords_file)
  else:
    refcoords = None
  print(np.shape(refcoords))
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
  pool.map(plot_column_pair_partial, range(0,num_columns))
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
  print(np.shape(refcoords))
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
  pool.map(plot_column_pair_partial, range(0,num_columns))
  pool.terminate()

  print("Done plotting columns")
  return


def compute_one_vs_all_rf_models(features_dir, projected_tica_coords, gmm_dir, save_dir, n_trees = 10, n_tica_components=25):
  features = np.concatenate(load_file_list(None, directory = features_dir, ext = ".dataset"))
  tics = np.concatenate(load_file(projected_tica_coords))
  feature_names = generate_features("/home/enf/b2ar_analysis/featuresreimaged_notrajfix_tm_residues_under_cutoff1nm/feature_residues_map.csv")
  for i in range(0, n_tica_components):
    print("Computing random forest model for tIC.%d" %(i+1))
    gmm = pickle.load(gzip.open("%s/tIC%d_gmm.pkl.gz" %(gmm_dir, i), "rb"))

    if len(gmm.means_) == 1: continue
    Y = gmm.predict(tics[:,i].reshape(-1,1))
    for component in range(0,len(gmm.means_)):
      print("Analyzing component %d" %component)
      all_indices = range(0,np.shape(tics)[0])
      component_indices = [k for k in all_indices if Y[k] == component]
      non_component_indices = list(set(all_indices)-set(component_indices)).sort()
      print("Found indices")
      Z = copy.deepcopy(Y)
      Z[component_indices] = 0
      Z[non_component_indices] = 1
      rf = RandomForestClassifier(max_features="sqrt",bootstrap=True,n_estimators=n_trees,n_jobs=-1)
      print("fitting random forest model")
      r = rf.fit(features, Z)
      print("fit random forest model, dumping now.")
      with gzip.open("%s/tIC%d_c%d_vs_all_rf.pkl.gz" %(save_dir, i, component), "wb") as f:
        pickle.dump(rf, f)

      feature_component_means = np.mean(features[component_indices,:], axis=0)
      print(np.shape(feature_component_means))
      print(feature_component_means[0:100])
      feature_non_component_means = np.mean(features[non_component_indices,:], axis=0)

      feature_importances = [[i, feature_names[i], rf.feature_importances_[i]] for i in range(0,len(rf.feature_importances_))]
      feature_importances = feature_importances.sort(key=operator.itemgetter(2))
      pickle.dump(feature_importances, open("%s/tIC%d_c%d_importances_list.pkl" %(save_dir, i, component), "wb"))
      print(feature_importances[0:20])

      df = pd.DataFrame(columns=('feature_name', 'feature_index', 'feature_importance', 'component_mean', 'non_component_mean'))

      for feature_importance in feature_importances:
        print("Adding to df: %s" %feature_importance)
        df.iloc[k] = [feature_importance[1], feature_importance[0], feature_importance[2], feature_component_means[feature_importance[0]], feature_non_component_means[feature_importance[0]]]

      pickle.dump(df, open("%s/tIC%d_c%d_vs_all_df.pkl" %(save_dir, (i+1), component), "wb"))
      plot_component_importances(df, save_dir, i, component)


  return

def compute_overall_rf_models(features_dir, projected_tica_coords, gmm_dir, save_dir, R=True, n_trees = 10, n_tica_components=25):
  features = np.concatenate(load_file_list(None, directory = features_dir, ext = ".dataset"))
  tics = np.concatenate(load_file(projected_tica_coords))
  feature_names = generate_features("/home/enf/b2ar_analysis/featuresreimaged_notrajfix_tm_residues_under_cutoff1nm/feature_residues_map.csv")

  for i in range(0, n_tica_components):
    print("Computing random forest model for tIC.%d" %(i+1))
    with gzip.open("%s/tIC%d_gmm.pkl.gz" %(gmm_dir, i), "wb") as f:
      gmm = pickle.load(f)
    if len(gmm.means_) == 1: continue
    Y = gmm.predict(tics[:,i].reshape(-1,1))
    rf = RandomForestClassifier(max_features="sqrt",bootstrap=True,n_estimators=n_trees,n_jobs=-1)
    r = rf.fit(features, Y)

    with gzip.open("%s/tIC%d_overall_rf.pkl" %(save_dir, i), "wb") as f:
      pickle.dump(rf, f)

    feature_importances = [(feature_names[i],rf.feature_importances_[i]) for i in range(0,len(rf.feature_importances_))]
    feature_importances = feature_importances.sort(key=operator.itemgetter(1))
    print(feature_importances[0:20])
    try:
      plot_importances(feature_importances, save_dir, i)
    except:
      continue



