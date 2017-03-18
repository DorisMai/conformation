from plots import *
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from interpret_tICs import *
import pickle
from msm_resampled import *
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as preprocessing
from sklearn.metrics import auc as calculate_auc
import statsmodels
import scipy
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import label_binarize, scale, StandardScaler
from scipy.stats import pearsonr
from msmbuilder.utils import verbosedump, verboseload
from analysis import *
from grids import *
from docking_analysis import *
from custom_msm import *
from custom_clusterer import *
from subsampling import *
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import scale
from random import shuffle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from detect_intermediates import *
from interpret_tICs import *
from custom_tica import *
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from pandas.tools.plotting import table
from msm_resampled import *
import random
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcROC
import pybel as pb
import pubchempy as pc

matplotlib.style.use('ggplot')


def make_importances_df(importances, titles, scaled=False):
    if scaled:
        return pd.DataFrame(np.mean(np.vstack(list(importances)), axis=0), index = titles + ["%s_scaled" %n for n in titles], columns=["importance"]).sort("importance", inplace=False, ascending=False)
    else: 
        return pd.DataFrame(np.mean(np.vstack(list(importances)), axis=0), index = titles, columns=["importance"]).sort("importance", inplace=False, ascending=False)
        

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


def get_sample_coords(sample_indices, coords):
    sample_coords = []
    for cluster in range(0, np.shape(sample_indices)[0]):
        print("Looking at cluster %d" %cluster)
        cluster_coords = []
        if sample_indices[cluster][0].shape[0] == 1:
            traj_index_frame_tuples = [sample_indices[cluster]]
        else:
            traj_index_frame_tuples = sample_indices[cluster]
        for traj_index_frame_tuple in traj_index_frame_tuples:
            traj_index = traj_index_frame_tuple[0]
            frame = traj_index_frame_tuple[1]
            cluster_coords.append(coords[traj_index][frame])
        cluster_coords = np.vstack(cluster_coords)
        sample_coords.append(cluster_coords)
    return sample_coords

def analyze_docking_results_in_dir(docking_dir, ligands_dir, precision="SP", redo=False, write_to_disk=False, ligands=None):
  summary = "%s/all_docking_summary.csv" %docking_dir
  df = analyze_docking_results_multiple(docking_dir, precision=precision, 
                                        summary=summary, ligands=None,
                                        redo=redo, write_to_disk=write_to_disk)
  return df


def initialize_analysis(clusterer_dir, user_defined_coords, user_defined_names, biased_agonist_dir, agonist_dir, inverse_agonist_dir, docking_dir,
                        precision, docking_multiple_ligands, aggregate_docking, feature_residues_pkl, n_components, top_features,
                        lag_time, n_clusters, projected_features_dir, traj_dir, traj_ext, tica_dir,
                        prior_counts, msm_object, analysis_dir, n_samples):
  clusterer = compat_verboseload(clusterer_dir)
  cluster_averages = calculate_cluster_averages_per_feature(clusterer, user_defined_coords)
  cluster_averages = pd.DataFrame(cluster_averages, columns=user_defined_names)
  active_clusters = cluster_averages.loc[(cluster_averages["rmsd_npxxy_active"] < 0.5) & (cluster_averages["tm6_tm3_dist"] > 12.) & (cluster_averages["tm6_tm3_dist"] < 15.)]
  inactive_clusters = cluster_averages.loc[(cluster_averages["rmsd_npxxy_active"] > 0.5) & (cluster_averages["tm6_tm3_dist"] <10.)]

  biased_ligands = get_ligands(biased_agonist_dir)
  agonist_ligands = get_ligands(agonist_dir)  
  inverse_ligands = get_ligands(inverse_agonist_dir)

  all_ligands = get_ligands("/home/enf/b2ar_analysis/all_ligands")
  if not os.path.exists(docking_multiple_ligands):
    analyze_docking_results_multiple(docking_dir, precision = precision, ligands = all_ligands, summary = docking_multiple_ligands, redo = True)
  c = compute_cluster_averages(None, csv_filename=docking_multiple_ligands, save_csv=aggregate_docking)


  with open(feature_residues_pkl, "rb") as f:
      feature_residues = pickle.load(f)

  tica_coords = compat_verboseload(projected_features_dir)

  pp_n_components = n_components
  apriori_dfs = []
  for array in user_defined_coords:
      apriori_dfs.append(pd.DataFrame(array, columns=user_defined_names))
  tica_dfs = []
  for array in tica_coords:
      tica_dfs.append(pd.DataFrame(array, columns=["tIC.%d" %i for i in range(1,n_components+1)]))

  cluster_pnas_averages = calculate_cluster_averages_per_feature(clusterer, user_defined_coords)
  cluster_pnas_averages = pd.DataFrame(cluster_pnas_averages, columns=user_defined_names)

  cluster_tica_averages = calculate_cluster_averages_per_feature(clusterer, tica_coords)
  cluster_tica_averages = pd.DataFrame(cluster_tica_averages, columns=["tIC.%d" %i for i in range(1, n_components+1)])
  cluster_tica_pnas = pd.concat([cluster_pnas_averages, cluster_tica_averages], axis=1).dropna()

  clusters_map = make_clusters_map(clusterer)
  tica_resampled_file = os.path.join(tica_dir, "tica_msm_lag-time%d_clusters%d_resampled.h5" %(lag_time, n_clusters))
  projected_features = compat_verboseload(projected_features_dir)

  num_trajs = len(get_trajectory_files(traj_dir, traj_ext))

  def reweight_features_by_msm(msm_object):
      total_samples = 10000
      resampled_traj_to_frames_file = os.path.join(tica_dir, "msm_lag-time%d_prior-counts%s_clusters%d_resampled_%d.h5" %(lag_time, str(prior_counts), n_clusters, total_samples))
      resampled_traj_to_frames = resample_by_msm(total_samples, msm_object, clusters_map, num_trajs, resampled_traj_to_frames_file)

      resample_features_by_msm_equilibirum_pop(projected_features, resampled_traj_to_frames, tica_resampled_file)
      tica_resampled = compat_verboseload(tica_resampled_file)
      pnas_resampled_file = os.path.join(tica_dir, "pnas_resampled.h5")
      resample_features_by_msm_equilibirum_pop(user_defined_coords, resampled_traj_to_frames, pnas_resampled_file)
      pnas_resampled = compat_verboseload(pnas_resampled_file)

      resampled_traj_index_pairs = []
      for traj in resampled_traj_to_frames.keys():
          [resampled_traj_index_pairs.append((traj, frame)) for frame in resampled_traj_to_frames[traj]]


      features_eq = resample_features_by_msm_trajectory(top_features, resampled_traj_index_pairs)*10.
      tica_eq = pd.DataFrame(tica_resampled, columns=["tIC.%d" %i for i in range(1,n_components+1)])
      pnas_eq = pd.DataFrame(pnas_resampled, columns=user_defined_names)
      features_eq = pd.concat([features_eq, tica_eq, pnas_eq], axis=1)
      features_eq.columns = [str(f) for f in features_eq.columns.values.tolist()]

      f0 = pd.concat([f for f in top_features], axis=0)
      f2 = pd.concat([f for f in tica_dfs])
      f3 = pd.concat([f for f in apriori_dfs])
      prot_lig_features = pd.concat([f0,f2,f3],axis=1)
      all_traj_features = [pd.concat([top_features[i]*10., tica_dfs[i], apriori_dfs[i]], axis=1) for i in range(0, len(tica_dfs))]
      return features_eq, all_traj_features


  n_steps = 10000
  save_file = "%s/msm_traj_index_pairs.h5" % (tica_dir)
  #msm_traj_index_pairs = generate_msm_traj_index_series(msm_object, random.choice(active_clusters.index.values.tolist()), n_steps, bu72_pp_clusters_map, save_file)
  #msm_traj_index_pairs = compat_verboseload(save_file)
  features_eq, all_traj_features = reweight_features_by_msm(msm_object)


  samples_indices_file = "%s/samples_indices.h5" %analysis_dir
  samples_dir = "%s/clusterer_%dclusters_%dsamples" %(tica_dir, n_clusters, n_samples)
  if not os.path.exists(samples_dir):
      os.makedirs(samples_dir)
      sample_from_clusterer(clusterer_dir, projected_features_dir, get_trajectory_files(traj_dir, ".h5"), 
                            n_samples, samples_dir, samples_indices_file, structure=None,
                            residue_cutoff=10000, parallel=True,
                            worker_pool=None)
      clusters_map = make_clusters_map(compat_verboseload(clusterer_dir))
  
  with open(feature_residues_pkl, "rb") as f:
    feature_names = pickle.load(f)

  samples_indices = compat_verboseload(samples_indices_file)
  tica_coords = compat_verboseload(projected_features_dir)
  
  samples_tica = []
  samples_pnas = []
  samples_features = []

  samples_tica_file = "%s/clusterer_%dclusters_%dsamples_samples_kdtree_tica.h5" %(tica_dir, n_clusters, n_samples)
  if not os.path.exists(samples_tica_file):
    samples_tica = get_sample_coords(samples_indices, tica_coords)
    verbosedump(samples_tica, samples_tica_file)
  else:
    samples_tica = compat_verboseload(samples_tica_file)
  samples_tica_avg_df = pd.DataFrame([np.mean(t, axis=0) for t in samples_tica], index=["cluster%d" %i for i in range(0,n_clusters)], columns=["tIC.%d" %i for i in range(1, n_components+1)])

  samples_pnas_file = "%s/clusterer_%dclusters_%dsamples_samples_kdtree_pnas.h5" %(tica_dir, n_clusters, n_samples)
  #if not os.path.exists(samples_pnas_file):
  samples_pnas = get_sample_coords(samples_indices, user_defined_coords)
  verbosedump(samples_pnas, samples_pnas_file)
  #else:
  #  samples_pnas = compat_verboseload(samples_pnas_file)
  samples_pnas_avg_df = pd.DataFrame([np.mean(t, axis=0) for t in samples_pnas], index=["cluster%d" %i for i in range(0,n_clusters)], columns=user_defined_names)

  samples_features_file = "%s/clusterer_%dclusters_%dsamples_samples_kdtree_features.h5" %(tica_dir, n_clusters, n_samples)
  #if not os.path.exists(samples_features_file):
  samples_features = get_sample_coords(samples_indices, [x.values for x in top_features])
    #verbosedump(samples_features, samples_features_file)
  #else:
  #  samples_features = compat_verboseload(samples_features_file)
  samples_features_avg_df = pd.DataFrame([np.mean(t, axis=0) for t in samples_features], index=["cluster%d" %i for i in range(0,n_clusters)], columns=[str(f) for f in top_features[0].columns.values.tolist()])

  """
  samples_normalized_features_file = "%s/clusterer_%dclusters_%dsamples_samples_kdtree_features_normalized.h5" %(tica_dir, n_clusters, n_samples)
  if not os.path.exists(samples_normalized_features_file):
    features = load_file_list(get_trajectory_files(features_dir, ".dataset"), directory = None, ext = None)
    n = StandardScaler()
    n.fit(np.concatenate(features))
    normalized_features = [n.transform(f) for f in features]
    samples_normalized_features = get_sample_coords(samples_indices, normalized_features)
    verbosedump(samples_normalized_features, samples_normalized_features_file)
  else:
    samples_normalized_features = compat_verboseload(samples_normalized_features_file)
    samples_normalized_features_avg_df = pd.DataFrame([np.mean(t, axis=0) for t in samples_normalized_features], index=["cluster%d" %i for i in range(0,n_clusters)], columns=[str(f) for f in feature_names])
  """

  feature_strings = [str(feature_name) for feature_name in feature_names]
  #samples_normalized_features_averages = [np.mean(f, axis=0) for f in samples_normalized_features]
  #samples_normalized_features_averages_df = pd.DataFrame(samples_normalized_features_averages, columns=feature_strings)
  samples_normalized_features_avg_df =  pd.DataFrame(StandardScaler().fit_transform(samples_features_avg_df.values), columns=samples_features_avg_df.columns, index=samples_features_avg_df.index)

  samples_pnas_tica = pd.concat([samples_pnas_avg_df, samples_tica_avg_df], axis=1)
  samples_pnas_avg_df.sort("rmsd_npxxy_active", inplace=False)

  docking_multiple_ligands = "%s/all_docking_scores.csv" % docking_dir
  aggregate_docking = "%s/aggregate_docking.csv" % docking_dir

  if not os.path.exists(docking_multiple_ligands):
    analyze_docking_results_multiple(docking_dir, precision = precision, ligands = all_ligands, summary = docking_multiple_ligands, redo = True)

  #if not os.path.exists(aggregate_docking):
  #  print(docking_multiple_ligands)
  #  compute_cluster_averages(None, csv_filename=docking_multiple_ligands, save_csv=aggregate_docking)

  reference_docking_dir = "/home/enf/b2ar_analysis/reference_docking/docking_%s" %precision
  reference_ligand_docking = "%s/all_docking_scores.csv" % reference_docking_dir

  if not os.path.exists(reference_ligand_docking):
    analyze_docking_results_multiple(reference_docking_dir, precision = precision, ligands = all_ligands, summary = reference_ligand_docking, redo = True)

  reference_docking = pd.read_csv(reference_ligand_docking, index_col=0).dropna()
  reference_docking.columns = [''.join(e for e in lig if e.isalnum() or e=='-' or e=='_') for lig in reference_docking.columns.values]
  reference_docking.loc["null_scores"] = reference_docking.iloc[1].subtract(reference_docking.iloc[0])                             


  return [clusterer, cluster_averages, active_clusters, inactive_clusters, biased_ligands, agonist_ligands, inverse_ligands, all_ligands, c, feature_residues, tica_coords, user_defined_coords, pp_n_components, apriori_dfs, tica_dfs,
         cluster_pnas_averages, cluster_tica_averages, cluster_tica_pnas, top_features, clusters_map, tica_resampled_file, projected_features, num_trajs, features_eq, all_traj_features, samples_indices_file, samples_dir,
         samples_tica_avg_df, samples_pnas_avg_df, samples_features_avg_df, samples_normalized_features_avg_df, feature_names, feature_strings, samples_pnas_tica, reference_docking]

def msm_reweighted_features_per_ligand(feature_dfs, 
                                       ligand_populations_df,
                                       total_samples,
                                       clusters_map,
                                       msm_object,
                                       save_dir="",
                                       ligand_subset=None,
                                       redo=False):
  num_trajs = len(feature_dfs)
  lig_features_eq = {}
  lig_features_eq_filename = "%s/lig_features_eq.h5" %save_dir
  if os.path.exists(lig_features_eq_filename) and not redo:
    return(compat_verboseload(lig_features_eq_filename))

  for ligand in ligand_populations_df.index.values.tolist():
    if ligand_subset is not None:
      if ligand not in ligand_subset:
        continue
    lig_msm_resampled_file = "%s/%s_msm_eq_resampled.h5" %(save_dir, ligand)
    eq_pops = ligand_populations_df.loc[ligand][list(range(0, ligand_populations_df.shape[1]))].values
    new_msm = copy.deepcopy(msm_object)
    new_msm.populations_ = eq_pops

    lig_traj_to_frames = resample_by_msm(total_samples,
                                         msm_object=new_msm,
                                         clusters_map=clusters_map,
                                         num_trajs=num_trajs,
                                         save_file=None,
                                         equilibrium_populations=new_msm.populations_)

    lig_features_eq[ligand] = resample_features_by_msm_equilibirum_pop(feature_dfs,
                                                                       lig_traj_to_frames, None)

  verbosedump(lig_features_eq, lig_features_eq_filename)

  return lig_features_eq

def compute_docking_ddg(full_docking_df, md_lig_name, msm_object):
  docking_df = copy.deepcopy(full_docking_df)
  col_inds = []
  cols = []
  msm_state_ids = []
  for i, cluster in enumerate(docking_df.columns.values.tolist()):
    if "state" in cluster.lower():
      cluster_id = int(cluster[5:])
      print(cluster_id)
      try:
        print(msm_object.mapping_[cluster_id])
        msm_state_ids.append(msm_object.mapping_[cluster_id])
        col_inds.append(i)
        cols.append(cluster)
      except:
        continue
  msm_docking_df = docking_df[cols]
  eq_pops = msm_object.populations_[msm_state_ids]
  dg_md = np.log(eq_pops) / (-0.61)

  Boltzmann_per_state = np.exp(-0.61 * (dg_md - (msm_docking_df.loc[md_lig_name].values)))
  Z_apo = np.sum(Boltzmann_per_state)
  dg_apo = np.log(Boltzmann_per_state / Z_apo) / (-0.61)

  Boltzmanns_per_ligand = np.exp(-0.61*(dg_apo + msm_docking_df.values))
  Z_per_ligand = np.sum(Boltzmanns_per_ligand, axis=1)
  eq_pops_ligs = Boltzmanns_per_ligand / Z_per_ligand.reshape((-1,1))
  dg = np.log(eq_pops_ligs) / (-0.61)

  dg_dg_apo = np.vstack([dg, dg_apo.reshape((1, -1))])
  all_eq_pops = np.exp(dg_dg_apo * (-0.61))
 
  eq_pops_df = pd.DataFrame(all_eq_pops,
                            index=full_docking_df.index.values.tolist() + ["apo"],
                            columns=msm_state_ids)
  print(msm_state_ids)

  ddg = dg - dg_apo 

  docking_df[cols] = ddg

  return(docking_df, eq_pops_df)

def compute_docking_dg(docking_cluster_averages, msm_object, samples_tica_avg_df, samples_pnas_avg_df,
                       samples_normalized_features_avg_df, important_contact_features, traj_dir, traj_ext, 
                       tica_dir, ligands, reference_docking, clusters_map, feature_dfs, save_dir): 
  df_agg = docking_cluster_averages
  #df_agg = pd.read_csv(aggregate_docking, index_col=0).dropna()
  #df_agg = pd.read_csv(docking_multiple_ligands, index_col=0).dropna()
  df_agg.index = [n.split("_")[0] for n in df_agg.index.values]


  df_agg.columns = [''.join(e for e in lig if e.isalnum() or e=='-' or e=='_') for lig in df_agg.columns.values]
  msm_obj =msm_object

  msm_clusters = msm_obj.mapping_.keys()
  msm_cluster_names = []
  msm_cluster_eq_pops = []
  for cluster_id in msm_clusters:
      cluster_name = "cluster%d" %cluster_id
      if cluster_name in df_agg.index.values:
          state_id = msm_obj.mapping_[cluster_id]
          msm_cluster_eq_pops.append(msm_obj.populations_[state_id])
          msm_cluster_names.append(cluster_name)
  msm_cluster_eq_pops = np.array(msm_cluster_eq_pops)
  msm_cluster_deltaG = -0.61 * np.log(msm_cluster_eq_pops)
  msm_cluster_eq_pops_df = pd.DataFrame(msm_cluster_eq_pops, index=msm_cluster_names)
  aggregate_docking_msm = df_agg.loc[msm_cluster_names]

  samples_tica_avg_df = samples_tica_avg_df.loc[msm_cluster_names]
  samples_pnas_avg_df = samples_pnas_avg_df.loc[msm_cluster_names]
  samples_top_features_avg_df = samples_normalized_features_avg_df.loc[msm_cluster_names]
  print(aggregate_docking_msm.columns)

  ligand = "3p0g_lig"

  apo_deltaG = msm_cluster_deltaG - (-1.0 * aggregate_docking_msm[ligand].values)

  apo_populations = np.exp(-1.0*apo_deltaG / 0.61)
  Z_apo = np.sum(apo_populations)
  apo_populations = apo_populations / Z_apo
  apo_eq_pops_df = copy.deepcopy(msm_cluster_eq_pops_df)
  apo_eq_pops_df[apo_eq_pops_df.columns] = apo_populations.reshape((-1,1))
  apo_deltaG = -.61 * np.log(apo_populations)

  msm_cluster_eq_pops = apo_populations
  msm_cluster_deltaG = apo_deltaG
  msm_cluster_eq_pops_df = apo_eq_pops_df

  new_populations = copy.deepcopy(aggregate_docking_msm)
  for ligand in aggregate_docking_msm.columns.values:
      new_populations[ligand] = np.exp(-1.0*(-1.0*aggregate_docking_msm[ligand].values+msm_cluster_deltaG)/0.61)

  Z = np.sum(new_populations.values, axis=0)
  for j, ligand in enumerate(aggregate_docking_msm.columns.values):
      new_populations[ligand] = new_populations[ligand].values / Z[j]
  population_deltas = copy.deepcopy(new_populations)
  for ligand in aggregate_docking_msm.columns.values:
      population_deltas[ligand] = population_deltas[ligand].values / msm_cluster_eq_pops
  new_energies = copy.deepcopy(new_populations)
  for ligand in aggregate_docking_msm.columns.values:
      new_energies[ligand] = -.61 * np.log(new_populations[ligand])
  delta_delta_g = copy.deepcopy(new_energies)
  for ligand in aggregate_docking_msm.columns.values:
      delta_delta_g[ligand] = new_energies[ligand].values - msm_cluster_deltaG


  docking_normalized = copy.deepcopy(aggregate_docking_msm)
  #docking_normalized[docking_normalized.columns.values] = scale(docking_normalized.values)

  ddg_scaled = copy.deepcopy(delta_delta_g)
  #ddg_scaled[delta_delta_g.columns.values] = scale(delta_delta_g.values)
      
  deltas_tica = pd.concat([delta_delta_g, samples_tica_avg_df, samples_pnas_avg_df, samples_top_features_avg_df], axis=1)

  total_samples = 10000
  bi_msm = msm_obj
  num_trajs = len(get_trajectory_files(traj_dir, traj_ext))

  lig_features_eq = {}
  #lig_features_eq = msm_reweighted_features_per_ligand(feature_dfs, new_populations, bi_msm, 
  #                                                     total_samples, clusters_map, num_trajs, apo_populations, save_dir)

  features = delta_delta_g.transpose()
  null_features = reference_docking.transpose().loc[features.index]

  classes = pd.read_csv("/home/enf/b2ar_analysis/b2ar_antagonists_agonists3.csv", header=None)

  agonists = classes.iloc[1].dropna().values.tolist()
  agonists = [''.join(e for e in lig if e.isalnum() or e=='-' or e=='_') for lig in agonists]

  antagonists = classes.iloc[0].dropna().values.tolist()
  antagonists = [''.join(e for e in lig if e.isalnum() or e=='-' or e=='_') for lig in antagonists]


  labels = np.zeros((features.shape[0], 1), dtype=object)
  for agonist in agonists:
      try:
          labels[features.index.values.tolist().index(agonist), 0] = "agonist"
      except:
          print(agonist)
          continue
  for agonist in antagonists:
      try:
          labels[features.index.values.tolist().index(agonist), 0] = "antagonist"
      except:
          print(agonist)
          continue
  non_zero_inds = np.where(labels != 0)[0]
  X = features.values[non_zero_inds,:]
  N = -1.0 * null_features.values[non_zero_inds,2]
  C = N
  y = labels[non_zero_inds, :]
  y = label_binarize(y, ["agonist", "antagonist"])


  return apo_populations, df_agg, aggregate_docking_msm, docking_normalized, ddg_scaled, deltas_tica, delta_delta_g, lig_features_eq, new_populations, bi_msm, num_trajs, features, null_features, classes, agonists, antagonists, labels, X, N, C, y

def make_clustermap(delta_delta_g, tica_dir, n_clusters, msm_lag_time, n_components, precision, null_features):
  """
  samples_tica = pd.read_csv(tica_coords_csv, index_col=0)
  samples_docking = pd.read_csv(docking_multiple_ligands, index_col=0)
  common_indices = list(set(samples_docking.index.values).intersection(samples_tica.index.values))
  samples_tica = samples_tica.loc[common_indices]
  samples_docking = samples_docking.loc[common_indices]


  pearson_matrix = np.zeros((samples_docking.shape[1], samples_tica.shape[1]))
  for i in range(0, pearson_matrix.shape[0]):
      for j in range(0, pearson_matrix.shape[1]):
          pearson_matrix[i][j] = pearsonr(samples_docking.values[:,i], samples_tica.values[:,j])[0]
  MI_matrix = np.abs(compute_sr_matrix(samples_docking.values, samples_tica.values))
  """
  plt.clf()
  #first_entries = ["nebivolol", "s-carvedilol", "s-carazolol", "s-atenolol", "xamoterol", "3p0g_lig", "isoetharine", "ethylnorepinephrine", "salbutamol", "norepinephrine"]
  secret_compounds = [c for c in delta_delta_g.columns.values if "Compound" in c]
  #drug_order = first_entries + list(set(delta_delta_g.columns.values).difference(set(first_entries)).difference(set(secret_compounds)))
  #delta_delta_g = delta_delta_g[drug_order]
  #delta_delta_g.sort("nebivolol", inplace=True)

  #plot_heatmap(scale(delta_delta_g.values).T, delta_delta_g.columns.values, delta_delta_g.index.values, save_file="%s/msm_n-clusters%d_lag-time%d_n-heatmap.eps" %(tica_dir, n_clusters, msm_lag_time))
  #plot_heatmap(MI_matrix, samples_docking.columns.values, samples_tica.columns.values, save_file="%s/msm_n-clusters%d_lag-time%d_tICs%d.eps" %(tica_dir, n_clusters, msm_lag_time, n_components))
  ddg_scaled = copy.deepcopy(delta_delta_g)
  ddg_scaled[delta_delta_g.columns.values] = scale(delta_delta_g.values)
  #ddg_scaled.index = [n.split("cluster")[1] for n in ddg_scaled.index.values]


  #plot_clustermap(docking_normalized[["nebivolol", "terbutaline", "s-carvedilol", "Ici118551", "s-atenolol", "propranolol", "bisoprolol", "s-carazolol", "timolol", "procaterol", "r_isopreterenol", "norepinephrine", "r_epinephrine", "ethylnorepinephrine", "isoetharine", "N-Cyclopentylbutanephrine", "3p0g_lig", "fenoterol", "formoterol"]].loc[["cluster80", "cluster16", "cluster99", "cluster90", "cluster43", "cluster62", "cluster9", "cluster89", "cluster58", "cluster74", "cluster6"]].transpose(), save_file="%s/msm_n-clusters%d_lag-time%d_tICs%d.eps" %(tica_dir, n_clusters, msm_lag_time, n_components), method='average')
  #plot_clustermap(ddg_scaled[["s-carvedilol", "s-carazolol", "alprenalol", "norepinephrine", "nebivolol", "clenbuterol", "Tulobuterol", "r_isopreterenol", "isoetharine", "formoterol", "r_epinephrine", "ethylnorepinephrine", "N-Cyclopentylbutanephrine"]].loc[importances_df.index.values.tolist()[:10]].transpose(), save_file="%s/msm_n-clusters%d_lag-time%d_tICs%d.eps" %(tica_dir, n_clusters, msm_lag_time, n_components), method='average')
  plot_clustermap(pd.concat([ddg_scaled.transpose(), null_features], axis=1), save_file="%s/msm_n-clusters%d_lag-time%d_tICs%d_%s.eps" %(tica_dir, n_clusters, msm_lag_time, n_components, precision), method='average', z_score=1)
  return secret_compounds, ddg_scaled



#print(deltas_tica.iloc[0:10])

"""
docking_normalized[docking_normalized.columns.values] = scale(population_deltas.values)

train_biased_antagonists = ["s-carvedilol", "nebivolol"] 
train_inverse_agonists = [] #["s-carazolol", "Ici118551"]"

train_arrestin_agonists = ["isoetharine", "3p0g_lig"]
train_gprot_agonists = ["procaterol"]

train_agonists = ["r_isopreterenol"] + train_arrestin_agonists + train_gprot_agonists

indices = []
for biased_antagonist in (train_biased_antagonists):# + train_arrestin_agonists):
    for inverse_agonist in train_inverse_agonists:
        bias_antagonist_minus_antagonists = delta_delta_g[biased_antagonist].values - delta_delta_g[inverse_agonist].values
        #bias_antagonist_minus_antagonists = scale(bias_antagonist_minus_antagonists)
        indices.append(set(np.where(bias_antagonist_minus_antagonists < -0.)[0]))
    indices.append(set(np.where(scale(delta_delta_g[biased_antagonist].values) <-1.)[0]))

#if train_gprot_agonists is not None:
#    for biased_antagonist in train_arrestin_agonists:
#        for inverse_agonist in (train_gprot_agonists):
#            bias_antagonist_minus_antagonists = delta_delta_g[biased_antagonist].values - delta_delta_g[inverse_agonist].values
#            #bias_antagonist_minus_antagonists = scale(bias_antagonist_minus_antagonists)
#            indices.append(set(np.where(bias_antagonist_minus_antagonists < -0.)[0]))
#        indices.append(set(np.where(delta_delta_g[biased_antagonist].values <0)[0]))


indices = set.intersection(*indices)
#bias_antagonist_minus_agonists = deltas_tica[[" 3p0g_lig"]].mean(axis=1).values - deltas_tica[train_agonists].mean(axis=1).values
#bias_antagonist_minus_agonists = scale(bias_antagonist_minus_agonists)
#indices = list(set(np.where(bias_antagonist_minus_antagonists < -.5)[0]))#.tolist()).intersection(set(np.where(bias_antagonist_minus_antagonists > 1.)[0].tolist())))
biased_antagonist_states = deltas_tica.iloc[list(indices)]#.intersection(set(np.where(np.max(scale(deltas_tica[train_biased_antagonists].values),axis=1) < -.5)[0])))]
print("biased antagonist states")
print(indices)

#biased_antagonist_states = biased_antagonist_states.loc[biased_antagonist_states["tm6_tm3_dist"] > 12.]

indices = []

for biased_antagonist in train_agonists:
    for inverse_agonist in (train_inverse_agonists):
        bias_antagonist_minus_antagonists = delta_delta_g[biased_antagonist].values - delta_delta_g[inverse_agonist].values
        bias_antagonist_minus_antagonists = scale(bias_antagonist_minus_antagonists)
        indices.append(set(np.where(bias_antagonist_minus_antagonists < -0.)[0]))
        #indices.append(set(np.where(delta_delta_g[inverse_antagonist].values > -.5)[0]))
    indices.append(set(np.where(delta_delta_g[biased_antagonist].values <-0.)[0]))
indices = set.intersection(*indices)
agonist_states = deltas_tica.iloc[list(indices)]#.intersection(set(np.where(np.max(scale(deltas_tica[train_biased_antagonists].values),axis=1) < -.5)[0])))]
#agonist_states = agonist_states.loc[agonist_states["tm6_tm3_dist"] > 12.]
print("agonist states:")
print(indices)

indices = []
for biased_antagonist in train_arrestin_agonists:
    for inverse_agonist in (train_gprot_agonists):
        bias_antagonist_minus_antagonists = agonist_states[biased_antagonist].values - agonist_states[inverse_agonist].values
        bias_antagonist_minus_antagonists = scale(bias_antagonist_minus_antagonists)
        indices.append(set(np.where(bias_antagonist_minus_antagonists < -0.)[0]))
        #indices.append(set(np.where(delta_delta_g[inverse_antagonist].values > -.5)[0]))
    indices.append(set(np.where(agonist_states[biased_antagonist].values <-0.)[0]))
indices = set.intersection(*indices)
arrestin_agonist_states = agonist_states.iloc[list(indices)]#.intersection(set(np.where(np.max(scale(deltas_tica[train_biased_antagonists].values),axis=1) < -.5)[0])))]
print("arrestin agonist states:")
print(indices)

indices = []
for biased_antagonist in train_gprot_agonists:
    for inverse_agonist in (train_arrestin_agonists):
        bias_antagonist_minus_antagonists = agonist_states[biased_antagonist].values - agonist_states[inverse_agonist].values
        bias_antagonist_minus_antagonists = scale(bias_antagonist_minus_antagonists)
        indices.append(set(np.where(bias_antagonist_minus_antagonists < -0.)[0]))
        #indices.append(set(np.where(delta_delta_g[inverse_antagonist].values > -.5)[0]))
    indices.append(set(np.where(agonist_states[biased_antagonist].values <-0.)[0]))
indices = set.intersection(*indices)
gprot_agonist_states = agonist_states.iloc[list(indices)]#.intersection(set(np.where(np.max(scale(deltas_tica[train_biased_antagonists].values),axis=1) < -.5)[0])))]
print("gprot agonist states:")
print(indices)
"""

def make_ligand_plots(lig_features_eq):
  thr_x = np.linspace(5, 50, 500)
  thr_kde1 = stats.gaussian_kde(lig_features_eq["norepinephrine"]["Thr66_Leu266"])
  thr_kde2 = stats.gaussian_kde(lig_features_eq["ethylnorepinephrine"]["Thr66_Leu266"])
  thr_dx1 = thr_kde1(thr_x)
  thr_dx2 = thr_kde2(thr_x)
  plt.plot(thr_x,thr_dx1-thr_dx2)

  thr_x = np.linspace(5, 50, 500)
  thr_kde1 = stats.gaussian_kde(lig_features_eq["r_epinephrine"]["Asn148_Leu266"])
  thr_kde2 = stats.gaussian_kde(lig_features_eq["r_isopreterenol"]["Asn148_Leu266"])
  thr_dx1 = thr_kde1(thr_x)
  thr_dx2 = thr_kde2(thr_x)
  plt.plot(thr_x,thr_dx1-thr_dx2)

  from scipy import stats
  thr_x = np.linspace(0,3.,500)
  thr_kde1 = stats.gaussian_kde(lig_features_eq["r_isopreterenol"]["rmsd_npxxy_active"])
  thr_kde2 = stats.gaussian_kde(lig_features_eq["s-carazolol"]["rmsd_npxxy_active"])
  thr_dx1 = thr_kde1(thr_x)
  thr_dx2 = thr_kde2(thr_x)
  plt.plot(thr_x,thr_dx1-thr_dx2)

  from scipy import stats
  thr_x = np.linspace(5, 50, 500)
  thr_kde1 = stats.gaussian_kde(lig_features_eq["isoetharine"]["Asn148_Leu266"])
  thr_kde2 = stats.gaussian_kde(lig_features_eq["r_isopreterenol"]["Asn148_Leu266"])
  thr_dx1 = thr_kde1(thr_x)
  thr_dx2 = thr_kde2(thr_x)
  plt.plot(thr_x,thr_dx1-thr_dx2)

  from scipy import stats
  thr_x = np.linspace(5, 50, 500)
  thr_kde1 = stats.gaussian_kde(lig_features_eq["N-Cyclopentylbutanephrine"]["Asn148_Leu266"])
  thr_kde2 = stats.gaussian_kde(lig_features_eq["r_isopreterenol"]["Asn148_Leu266"])
  thr_dx1 = thr_kde1(thr_x)
  thr_dx2 = thr_kde2(thr_x)
  plt.plot(thr_x,thr_dx1-thr_dx2)

  from scipy import stats
  thr_x = np.linspace(5, 50, 500)
  thr_kde1 = stats.gaussian_kde(lig_features_eq["ethylnorepinephrine"]["Asn148_Leu266"])
  thr_kde2 = stats.gaussian_kde(lig_features_eq["r_isopreterenol"]["Asn148_Leu266"])
  thr_dx1 = thr_kde1(thr_x)
  thr_dx2 = thr_kde2(thr_x)
  plt.plot(thr_x,thr_dx1-thr_dx2)

  from scipy import stats
  thr_x = np.linspace(5, 50, 500)
  thr_kde1 = stats.gaussian_kde(lig_features_eq["salbutamol"]["Asn148_Leu266"])
  thr_kde2 = stats.gaussian_kde(lig_features_eq["r_isopreterenol"]["Asn148_Leu266"])
  thr_dx1 = thr_kde1(thr_x)
  thr_dx2 = thr_kde2(thr_x)
  plt.plot(thr_x,thr_dx1-thr_dx2)

  from scipy import stats
  thr_x = np.linspace(5, 20, 500)
  thr_kde1 = stats.gaussian_kde(lig_features_eq["N-Cyclopentylbutanephrine"]["tm6_tm3_dist"])
  thr_kde2 = stats.gaussian_kde(lig_features_eq["r_isopreterenol"]["tm6_tm3_dist"])
  thr_dx1 = thr_kde1(thr_x)
  thr_dx2 = thr_kde2(thr_x)
  plt.plot(thr_x,thr_dx1-thr_dx2)

  from scipy import stats
  thr_x = np.linspace(5, 50, 500)
  thr_kde1 = stats.gaussian_kde(lig_features_eq["3p0g_lig"]["Thr66_Leu266"])
  thr_kde2 = stats.gaussian_kde(lig_features_eq["r_isopreterenol"]["Asn148_Leu266"])
  thr_dx1 = thr_kde1(thr_x)
  thr_dx2 = thr_kde2(thr_x)
  plt.plot(thr_x,thr_dx1-thr_dx2)

def custom_lim_finder(values):
    mins = np.min(values, axis=0)
    maxs = np.max(values, axis=0)
    stds = np.std(values, axis=0)
    custom_lims = [[mins[i] - 0.5*stds[i], maxs[i] + 0.5*stds[i]] for i in range(0,len(mins))]
    return custom_lims

def construct_difference_plots():
  for measurement in ["Ala59_Leu266", "Thr66_Leu266", "Asn148_Leu266"]:
      """
      iso = lig_features_eq["r_isopreterenol"][measurement]
      x = np.linspace(np.min(iso.values), np.max(iso.values), 500)
      kde2 = stats.gaussian_kde(iso)
      dx2 = kde2(x)
      plt.clf()
      plt.plot(x, dx2)
      plt.title("Isopreterenol Eq. Population Frequency")
      plt.xlabel("%s closest heavy atom distance" %str(measurement))
      plt.ylabel("Eq. Population")
      save_file = "%s/%s_isopreterenol_kde.eps" %(analysis_dir, measurement)
      plt.savefig(save_file)
      
      plt.clf()
      plt.hist(iso, range=[np.min(iso.values), np.max(iso.values)], bins=100)
      plt.title("Isopreterenol Eq. Population Frequency")
      plt.xlabel("%s closest heavy atom distance" %str(measurement))
      plt.ylabel("Eq. Population")
      save_file = "%s/%s_isopreterenol_hist.eps" %(analysis_dir, measurement)
      plt.savefig(save_file)
      """
      cara = lig_features_eq["s-carazolol"][measurement]
      c = np.linspace(np.min(cara.values), np.max(cara.values), 500)
      kde2 = stats.gaussian_kde(cara)
      dc2 = kde2(c)
      
      """
      plt.clf()
      plt.plot(c, dc2)
      plt.title("s-carazolol Eq. Population Frequency")
      plt.xlabel("%s closest heavy atom distance" %str(measurement))
      plt.ylabel("Eq. Population")
      save_file = "%s/%s_carazolol_kde.eps" %(analysis_dir, measurement)
      plt.savefig(save_file)
      
      plt.clf()
      plt.hist(cara, range=[np.min(iso.values), np.max(iso.values)], bins=100)
      plt.title("Carazolol Eq. Population Frequency")
      plt.xlabel("%s closest heavy atom distance" %str(measurement))
      plt.ylabel("Eq. Population")
      save_file = "%s/%s_carazolol_hist.eps" %(analysis_dir, measurement)
      plt.savefig(save_file)
      """
      
      for ligand in ["3p0g_lig", "salbutamol", "salmeterol", "s-carvedilol", "isoetharine", "norepinephrine", "r_epinephrine", "ethylnorepinephrine", "nebivolol", "N-Cyclopentylbutanephrine"]:
          save_file = "%s/%s_%s_minus_carazolol_frequency.eps" %(analysis_dir, measurement, ligand)
          if os.path.exists(save_file):
              continue
              
          plt.clf()
          print(ligand)
          print(measurement)
          
          kde1 = stats.gaussian_kde(lig_features_eq[ligand][measurement].dropna())
          
          dc1 = kde1(c)
          
          plt.plot(c,dc1-dc2)
          if ligand == "3p0g_lig":
              lig_title = "BI"
          else:
              lig_title = ligand
          plt.title("%s Frequency minus Carazolol Frequency" %lig_title)
          plt.xlabel("%s closest heavy atom distance" %str(measurement))
          plt.ylabel("Equilibrium Population Change")
          plt.savefig(save_file)

def plot_ligand_observable_difference_kde(lig_features_eq, reference_lig, observable, save_dir):
  return 

def hi():
  return


def construct_2d_distance_plots():
  #deer_distances = ["Ala59_Leu266", "Thr66_Leu266", "Asn148_Leu266", "tm6_tm3_dist", "rmsd_npxxy_active"]
  ligands = ["3p0g_lig", "salbutamol", "salmeterol", "s-carvedilol", "isoetharine", "norepinephrine", "r_epinephrine", "ethylnorepinephrine", "nebivolol", "N-Cyclopentylbutanephrine"]
  deer_distances = ["tm6_tm3_dist", "rmsd_npxxy_active", "rmsd_npxxy_inactive", "Ala59_Leu266", "Thr66_Leu266", "Asn148_Leu266"]
  #deer_distances = ["tm6_tm3_dist", "rmsd_npxxy_active"]
  all_apo_data = lig_features_eq["s-carazolol"][deer_distances].values

  for ligand in ligands:
      jointplots(lig_features_eq[ligand][deer_distances].values, analysis_dir, titles = deer_distances, main = "%s Minus Carazolol" %ligand, refcoords = None, refcoords_j=None,
              axes=None, reshape=True, data_j=None, titles_j=None, max_tIC=100, min_density=None, 
              custom_lims=custom_lim_finder(all_apo_data), max_diff=2.5, tpt_paths=None, tpt_paths_j=None,
               n_levels=10, worker_pool=None, parallel=True, n_pts=200j, all_apo_data=all_apo_data)

def convert_compound_on_disk_to_smiles(filename):
  if 1==1:
    if ".sdf" in filename or ".mol" in filename:
      return(''.join(convert_sdf_to_smiles(filename).replace('\'', '').replace("\\n", "")))
    elif ".smi" in filename:
      with open(filename, "r") as f:
        smiles = ''.join(f.read().replace('\n', '').replace('\'', '').replace("\\n", ''))
      return(smiles)
  else:
    return("")

def convert_compound_in_dir_to_smiles(compound_name, directory):
  sdf_file = "%s/%s.sdf" %(directory, compound_name)
  mol_file = "%s/%s.mol" %(directory, compound_name)
  smi_file = "%s/%s.smi" %(directory, compound_name)

  if os.path.exists(sdf_file):
    return(convert_compound_on_disk_to_smiles(sdf_file))
  elif os.path.exists(mol_file):
    return(convert_compound_on_disk_to_smiles(mol_file))
  elif os.path.exists(smi_file):
    return(convert_compound_on_disk_to_smiles(smi_file))
  else:
    return("")

def convert_compounds_in_dir_to_smiles(compound_names, directory, parallel=False, worker_pool=None):
  convert_compound_in_dir_to_smiles_partial = partial(convert_compound_in_dir_to_smiles,
                                                      directory=directory)

  smiles_strings = function_mapper(convert_compound_in_dir_to_smiles_partial,
                                   parallel=parallel,
                                   worker_pool=worker_pool,
                                   var_list=compound_names)
  return(smiles_strings)


def convert_sdf_to_smiles(sdf_file):
  try:
    for mol in pb.readfile("sdf", sdf_file):
      return(mol.write("can"))
  except:
    return("")

def convert_sdfs_to_smiles(sdfs, parallel=False, worker_pool=None):
  smiles_list = function_mapper(convert_sdf_to_smiles, worker_pool, parallel, sdfs)
  return(smiles_list)

def convert_smiles_to_compound(smiles):
  try: 
    c = pc.get_compounds(smiles, namespace='smiles')
    pc_smiles = c[0].canonical_smiles
    c2 = pc.get_compounds(pc_smiles, namespace='smiles')[0]
    cid = c[0].cid
    return((c[0].synonyms[0], c[0].canonical_smiles, c2.synonyms[0], cid))
  except:
    return(("", "", "", ""))

def convert_smiles_to_compounds(smiles, parallel=False, worker_pool=None):
  compound_names_smiles = function_mapper(convert_smiles_to_compound, worker_pool, parallel, smiles)
  return(compound_names_smiles)

def write_smiles_to_disk():
  return

def convert_sdf_to_compound(sdf):
  smiles = convert_sdf_to_smiles(sdf)
  name, pc_smiles, pc_name, cid = convert_smiles_to_compound(smiles)
  return((smiles, name, pc_smiles, pc_name, cid))

def convert_sdfs_to_compounds(sdfs, parallel=False, worker_pool=None):
  print("Getting SMILES from SDFs...")
  print("Done. Now getting compound names from SMILES...")
  results = function_mapper(convert_sdf_to_compound, worker_pool, parallel, sdfs)
  smiles_list = [t[0] for t in results]
  names = [t[1] for t in results]
  pc_smiles = [t[2] for t in results]
  pc_names = [t[3] for t in results]
  cids = [t[4] for t in results]
  print("Done. returning compound names.")
  return(smiles_list, names, pc_smiles, pc_names, cids) 

"""
adapted from RDKit since there is no AUC
calculator for BedROC
"""

def calc_auc_from_roc(roc, scores, col):
  TNR = roc[0] 
  precision = roc[1]    
  numMol = len(scores) 
  AUC = 0 

   # loop over score list 
  for i in range(0, numMol-1): 
    AUC += (TNR[i+1]-TNR[i]) * (precision[i+1]+precision[i]) 

  return 0.5*AUC 

def compute_auc(y_train, y_score):
    fdr, precision, _ = roc_curve(y_train, y_score[:,1])
    roc_auc = calculate_auc(fdr, precision)
    log_auc = logauc2(fdr, precision)
    return roc_auc, log_auc

def b(x, y, i):
    return y[i+1] - x[i+1] * (y[i+1] - y[i]) / (x[i+1] - x[i])

def logauc(x, y, lam=0.001):
    num = 0.
    for i in range(0, len(x)-1):
        if x[i] >= lam:
            num += ((y[i+1]-y[i])/np.log(10) + b(x, y, i) * (np.log10(x[i+1]) - np.log10(x[i])))
    return num / (np.log10(1./lam))

def logauc2(x, y, lam=0.001):
    num = 0.
    for i in range(0, len(x)-1):
        if x[i] >= lam:
            num += (np.log10(x[i+1]) - np.log10(x[i])) * (y[i+1]+y[i]) /2.
    return num / (np.log10(1./lam))

def do_regression_experiment(features, y, feature_names, n_trials, 
                             train_size=0.8, regularize=False, model="rfr",
                             normalize=False, normalize_axis0=True, worker_pool=None,
                             parallel=True):
    test_r2s = []
    feature_importances = []
    results_dict = {}

    do_single_regression_experiment_partial = partial(do_single_regression_experiment, features=features, 
                                                      y=y, n_estimators=1000, train_size=train_size,
                                                      model=model, normalize=normalize, normalize_axis0=normalize_axis0)

    model_results = function_mapper(do_single_regression_experiment_partial, worker_pool, parallel, list(range(0,n_trials)))
    
    print("Finished fitting models")

    results_dict['feature_importances'] = [t[0] for t in model_results]
    results_dict['test_r2s'] = [t[1] for t in model_results]
    
    return results_dict

def do_single_regression_experiment(trial, features, y,
                                    n_estimators, 
                                    train_size=0.8,
                                    regularize=False,
                                    model="rfr",
                                    normalize=True,
                                    normalize_axis0=True):
  features_y = features + [y]
  train_test_arrays = train_test_split(*features_y, train_size=train_size) 

  y_train = train_test_arrays[2*len(features)]
  y_test = train_test_arrays[2*len(features) + 1]
  feature_importance = []

  test_r2s = []
  kendall_coefficients = []
  kendall_pvalues = []
  feature_importances = []
  kendall_scores = []
  r2_scores = []


  for i in range(0, len(features)):
      X_train = train_test_arrays[2*i]
      X_test = train_test_arrays[2*i+1]

      if normalize:
        sc = StandardScaler()
        sc.fit(X_train)
      else:
        sc = None
      X_train = custom_normalize(X_train, sc, normalize_axis0)
      X_test = custom_normalize(X_test, sc, normalize_axis0)

      if model == "rfr":
        rfr = RandomForestRegressor(n_estimators=100, max_depth=None, max_features='sqrt', n_jobs=-1)
        rfr.fit(X_train, y_train)
        feature_importance.append(rfr.feature_importances_)
      elif model == "LassoCV":
        rfr = linear_model.LassoCV(n_jobs=-1)
        rfr.fit(X_train, y_train)
        feature_importance.append(rfr.coef_)
      elif model == "RidgeCV":
        rfr = linear_model.RidgeCV()
        rfr.fit(X_train, y_train)
        feature_importance.append(rfr.coef_)
      elif model == "SVR":
        rfr = SVR()
        rfr.fit(X_train, y_train)
        feature_importance.append([0. for i in range(0, X_train.shape[1])])

         
      coef, pval = scipy.stats.kendalltau(rfr.predict(X_test).ravel(), y_test.ravel())
      kendall_scores.append(coef)
      kendall_pvalues.append(pval)
      r2_score = rfr.score(X_test, y_test)
      r2_scores.append(r2_score)
          

      feature_importances.append(feature_importance)
    
  return((feature_importances, r2_scores))


class RegularizedModel(object):

  def __init__(self, sklearn_model, input_transformer=None,
               normalize_axis0=False, retained_features=None):
    self.sklearn_model = sklearn_model
    self.retained_features = retained_features
    self.input_transformer = input_transformer
    self.normalize_axis0 = normalize_axis0

  def predict(self, X):
    X = self.pre_regularize(X)
    y_pred = self.sklearn_model.predict(X)
    return(y_pred)

  def predict_proba(self, X):
    X = self.pre_regularize(X)
    proba = self.sklearn_model.predict_proba(X)
    return(proba)

  def pre_regularize(self, X):
    X = custom_normalize(X, self.input_transformer, self.normalize_axis0)
    if self.retained_features is not None:
      X = X[:, self.retained_features]
    return X

class CustomSplitter(object):

  def __init__(self, antagonist_inds, a_agonist_inds,
               b_agonist_inds,
               proportion=0.9):
    self.antagonist_inds = antagonist_inds
    self.a_agonist_inds = a_agonist_inds
    self.b_agonist_inds = b_agonist_inds
    self.proportion = proportion

  def split(self, array_list):
      random.shuffle(self.antagonist_inds)
      random.shuffle(self.a_agonist_inds)
      n_ligands = len(self.antagonist_inds) + len(self.a_agonist_inds) + len(self.b_agonist_inds)
      antagonist_prop = float(len(self.antagonist_inds)) / float(n_ligands)

      n_test_ligands = float(len(self.b_agonist_inds))/(1.-antagonist_prop)
      n_test_antagonists = int(np.round(n_test_ligands * antagonist_prop))

      test_inds = self.b_agonist_inds + self.antagonist_inds[:n_test_antagonists]
      train_inds = self.antagonist_inds[n_test_antagonists:] + self.a_agonist_inds

      print("train inds:")
      print(train_inds,end="")

      print("test_inds:")
      print(test_inds, end="")

      split_list = []
      for arr in array_list:
          split_list.append(arr[train_inds,:])
          split_list.append(arr[test_inds, :])
      return split_list

def generate_or_load_model(features, y, featurizer_names, 
                           n_trials, train_test_split_p,
                           manual_regularize, model_name, filename,
                           n_estimators=1000, max_depth=3,
                           redo=True, parallel=False, worker_pool=None,
                           criterion='gini', normalize=True, splitter=None,
                           normalize_axis0=False, loocv=False, penalty='l1'):
    
  if os.path.exists(filename) and not redo:  
    print("Already fit model, loading now.")
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return(model)
  else:
    print("Aobut to fit model(s).")
    model = do_classification_experiment(features=features, y=y, feature_names=featurizer_names,
                                         n_trials=n_trials, train_size=train_test_split_p, 
                                         regularize=manual_regularize,
                                         model=model_name, parallel=parallel, worker_pool=worker_pool,
                                         n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
                                         normalize=normalize, splitter=splitter, normalize_axis0=normalize_axis0,
                                         loocv=loocv, penalty=penalty)
    with open(filename, "wb") as f:
        pickle.dump(model, f, protocol=2)
    return(model)

def precision_threshold_full_recall(recall_curve):
  precision, recall, thresholds = recall_curve
  for i, threshold in enumerate(thresholds):
    if recall[threshold] == 1.:
      return((precision[i], threshold))

def do_single_classification_experiment(trial, features_y=[], y=None,
                                        features=None, model=None,
                                        regularize=None, train_size=None,
                                        max_depth=None, n_estimators=1000,
                                        splitter=None, normalize=True,
                                        normalize_axis0=False,
                                        loocv=False, penalty='l1'):
  aucs = []
  log_aucs = []
  roc_aucs = []
  class_b_aucs = []
  bedrocs = []
  fdrs = []
  precisions = []
  class_b_precisions = []
  class_b_fdrs = []

  recalls = []
  class_b_recalls = []

  precisions_at_full_recall = []
  thresholds_at_full_recall = []

  fdr_means = []
  precision_means = []
  recall_means = []

  accuracies = []

  if loocv:
    train_test_arrays = []
    indices = list(range(0,features_y[0].shape[0]))
    train_inds = [n for n in indices if n != trial]
    test_ind = [trial]
    for arr in features_y:
      if "pandas" in str(type(arr)).lower():
        arr = arr.values
      arr_train = arr[train_inds, :]
      arr_test = arr[test_ind, :]
      train_test_arrays.append(arr_train)
      train_test_arrays.append(arr_test)
  elif splitter is None:
    train_test_arrays = train_test_split(*features_y, train_size=train_size, stratify=y) 
  else:
    train_test_arrays = splitter.split(features_y)
  y_train = train_test_arrays[2*len(features)]
  y_test = train_test_arrays[2*len(features) + 1]
  feature_importance = []

  for i in range(0, len(features)):
      X_train = train_test_arrays[2*i]
      X_test = train_test_arrays[2*i+1]

      if normalize:
        sc = StandardScaler()
        sc.fit(X_train)
      else:
        sc = None
      X_train = custom_normalize(X_train, sc, normalize_axis0)
      X_test = custom_normalize(X_test, sc, normalize_axis0)

      if model == "rfr":
        rfr = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features='sqrt', oob_score=False)
        rfr.fit(X_train, y_train)
        if not regularize:
            feature_importance.append(rfr.feature_importances_)
        else:
            f = np.zeros(X_train.shape[1])
            top_indices = np.argsort(rfr.feature_importances_*-1.)[:regularize]
            rfr = RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt', oob_score=False)
            try:
              X_train = X_train[:, top_indices]
              X_test = X_test[:, top_indices]
            except:
              X_train = X_train.values[:, top_indices]
              X_test = X_test.values[:, top_indices]
            rfr.fit(X_train, y_train)
            f[top_indices] = rfr.feature_importances_
            feature_importance.append(f)
      elif model=="logistic_cv":
        if penalty == 'l2':
          rfr = linear_model.LogisticRegressionCV()
        else:
          rfr = linear_model.LogisticRegressionCV(penalty='l1', Cs=100, solver='liblinear')
        rfr.fit(X_train, y_train)
        #print(rfr.coef_)
        if not regularize:
            feature_importance.append(rfr.coef_)
        else:
            f = np.zeros(X_train.shape[1])
            top_indices = np.argsort(np.abs(rfr.coef_)*-1.)[:regularize].flatten()
            rfr = linear_model.LogisticRegressionCV()
            X_train = X_train[:, top_indices]
            X_test = X_test[:, top_indices]
            rfr.fit(X_train, y_train)
            f[top_indices] = rfr.coef_
            feature_importance.append(f)
      if train_size == 1.:
          X_test = X_train
          y_test = y_train
      
      y_pred = rfr.predict(X_test)
      y_score = rfr.predict_proba(X_test)

      y_probb = y_score[:,1]

      n_model_positives = len(np.where(y_pred == 1.)[0])
      n_model_negatives = len(np.where(y_pred == 0.)[0])

      positive_indices = np.where(y_test == 1.)[0].tolist()
      negative_indices = np.where(y_test == 0.)[0].tolist()

      tp_indices = set(np.where(y_test == 1.)[0].tolist()).intersection(set(np.where(y_pred == 1.)[0].tolist()))
      fp_indices = set(np.where(y_test == 0.)[0].tolist()).intersection(set(np.where(y_pred == 1.)[0].tolist()))

      try:
        recall_mean = np.mean(y_score[positive_indices][:,1])/(float(len(positive_indices))/float(len(y_test)))
      except:
        recall_mean = 0.
      try:
        fdr_mean = np.mean(y_score[negative_indices][:,1])/(float(n_model_positives)/float(y_score.shape[0]))
      except:
        fdr_mean = 0.
      try:
        precision_mean = np.mean(y_score[positive_indices][:,1])/(float(n_model_positives)/float(y_score.shape[0]))
      except:
        precision_mean = 0.

      recall_means.append(recall_mean)
      precision_means.append(precision_mean)
      fdr_means.append(fdr_mean)

      tp = float(len(set(np.where(y_test == 1.)[0].tolist()).intersection(set(np.where(y_pred == 1.)[0].tolist()))))
      fp = float(len(set(np.where(y_test == 0.)[0].tolist()).intersection(set(np.where(y_pred == 1.)[0].tolist()))))

      P = float(len(np.where(y_test == 1.)[0]))
      try: 
        mean_recall = recall_score(y_test, y_pred)
        class_b_recall = recall_score(y_test, y_pred, average=None)[1]
        recalls.append(mean_recall)
        class_b_recalls.append(class_b_recall)

        #print("class_b_recall:")
        #print(class_b_recall)
      except:
        recalls.append(0.)
        class_b_recalls.append(0.)

      try:
        curve = precision_recall_curve(y_test, y_probb, pos_label=1)
        min_precision, min_threshold = precision_threshold_full_recall(curve)
        precisions_at_full_recall.append(min_precision)
        thresholds_at_full_recall.append(min_threshold)
      except:
        precisions_at_full_recall.append(0.)
        thresholds_at_full_recall.append(0.)

      try:
        precision = precision_score(y_test, y_pred)
        fdr = 1. - precision
        precisions.append(precision)
        fdrs.append(fdr)

        class_b_precision = precision_score(y_test, y_pred, average=None)[1]
        class_b_fdr = 1. - class_b_precision

        class_b_precisions.append(class_b_precision)
        #print("class_b_fdr:")
        #$print(class_b_fdr)

        #print("class_b_precision")
        #print(class_b_precision)
        class_b_fdrs.append(class_b_fdr)
      except:
        precisions.append(0.)
        fdrs.append(1.)
        class_b_precisions.append(0.)
        class_b_fdrs.append(1.)

      y_score = rfr.predict_proba(X_test)


      try:
        auc, logauc = compute_auc(y_test, y_score)
        aucs.append(auc)
        log_aucs.append(logauc)  
      except:
        aucs.append(0.)
        log_aucs.append(0.)
      try:
        roc_aucs.append(roc_auc_score(convert_to_n_class(y_test[:,0]), y_score))
        class_b_aucs.append(roc_auc_score(convert_to_n_class(y_test[:,0]), y_score, average=None)[1])
      except:
        roc_aucs.append(0.)
        class_b_aucs.append(0.)

      try:
        scores_conc = np.hstack([y_test, y_score[:,1].reshape((-1,1))])
        score_df = pd.DataFrame(scores_conc, columns=["label"] + [str(i) for i in range(0, scores_conc.shape[1]-1)])
        score_df.sort("0", inplace=True, ascending=False)
        bedroc_auc = CalcBEDROC(score_df.values, col=0, alpha=20)
        #bedroc_auc = calc_auc_from_roc(CalcBEDROC(score_df.values, col=0, alpha=20), score_df.values, col=0)
        bedrocs.append(bedroc_auc)
      except:
        bedrocs.append(0.)

      accuracies.append(accuracy_score(y_test, y_pred))

  return((feature_importance, aucs, log_aucs, roc_aucs, bedrocs, precisions,
          fdrs, recalls, precision_means, fdr_means, recall_means,
          class_b_aucs, class_b_recalls, class_b_precisions, class_b_fdrs, accuracies,
          thresholds_at_full_recall, precisions_at_full_recall))

def custom_normalize(X, normalizer, normalize_axis0=False):
  if normalizer is not None:
    if not normalize_axis0:
      X = normalizer.transform(X)
    else:
      X = np.hstack([normalizer.transform(X), preprocessing.normalize(X, axis=1)])#, return_norm=True)
  elif normalize_axis0:
    X = np.hstack([X, preprocessing.normalize(X, axis=1)])#, return_norm=True)
    X = np.hstack([X, preprocessing.normalize(X, axis=1)])#, return_norm=True)
  return X

def do_classification_experiment(features, y, feature_names,
                                 n_trials, train_size=0.8,
                                 regularize=False, model="rfr",
                                 parallel=False, worker_pool=None,
                                 n_estimators=1000, max_depth=3,
                                 splitter=None, criterion='gini',
                                 normalize=False, normalize_axis0=False,
                                 loocv=False, penalty='l1'):
    test_aucs = []
    test_log_aucs = []
    test_roc_aucs = []
    feature_importances = []

    features_y = copy.deepcopy(features)
    features_y.append(y)

    results_dict = {} 

    print("Fitting models over all data...")
    for feature_ind, X_train in enumerate(features):
      if normalize:
        sc = StandardScaler()
        sc.fit(X_train)
      else:
        sc = None
      X_train = custom_normalize(X_train, sc, normalize_axis0)

      y_train = copy.deepcopy(y)
      if model == "rfr":
        rfr = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features='sqrt', n_jobs=-1, oob_score=False, criterion=criterion)
        rfr.fit(X_train, y_train)
        if not regularize:
            final_rfr = RegularizedModel(sklearn_model=rfr, input_transformer=sc, normalize_axis0=normalize_axis0)
            results_dict[feature_names[feature_ind]] = (final_rfr, rfr.feature_importances_)
        else:
            f = np.zeros(X_train.shape[1])
            top_indices = np.argsort(rfr.feature_importances_*-1.)[:regularize].tolist()
            rfr = RandomForestClassifier(n_estimators=500, max_depth=3, max_features='sqrt', n_jobs=-1, oob_score=False)
            try:
              X_train = X_train[:, top_indices]
            except:
              X_train = X_train.values[:, top_indices]
            rfr.fit(X_train, y_train)
            final_rfr = RegularizedModel(sklearn_model=rfr, retained_features=top_indices, input_transformer=sc, normalize_axis0=normalize_axis0)
            f[top_indices] = rfr.feature_importances_
            results_dict[feature_names[feature_ind]] = (final_rfr, f)

      elif model=="logistic_cv":
        if penalty=='l2':
          rfr = linear_model.LogisticRegressionCV()
        else:
          rfr = linear_model.LogisticRegressionCV(penalty='l1', Cs=100, solver='liblinear')
        rfr.fit(X_train, y_train)
        if not regularize:
            final_rfr = RegularizedModel(sklearn_model=rfr, input_transformer=sc, normalize_axis0=normalize_axis0)
            results_dict[feature_names[feature_ind]] = (final_rfr, rfr.coef_)
        else:
            f = np.zeros(X_train.shape[1])  
            top_indices = np.argsort(np.abs(rfr.coef_)*-1.)[:regularize].flatten()
            rfr = linear_model.LogisticRegressionCV()
            X_train = X_train[:, top_indices]
            rfr.fit(X_train, y_train)
            final_rfr = RegularizedModel(sklearn_model=rfr, retained_features=top_indices, input_transformer=sc, normalize_axis0=normalize_axis0)
            f[top_indices] = rfr.coef_
            results_dict[feature_names[feature_ind]] = (final_rfr, f)


    print("Fitting models over split train data...")
    if loocv:
      trial_indices = list(range(0, features[0].shape[0]))
    else:
      trial_indices = list(range(0,n_trials))

    do_single_classification_experiment_single_partial = partial(do_single_classification_experiment, features_y=features_y,
                                                                                             y=y, features=features, model=model,
                                                                                             regularize=regularize, train_size=train_size, 
                                                                                             n_estimators=n_estimators, max_depth=max_depth,
                                                                                             normalize=normalize, splitter=splitter,
                                                                                             normalize_axis0=normalize_axis0,
                                                                                             loocv=loocv, penalty=penalty)
    model_results = function_mapper(do_single_classification_experiment_single_partial, worker_pool, parallel, trial_indices)
    
    print("Finished fitting models")

    results_dict['feature_importances'] = [t[0] for t in model_results]
    results_dict['test_aucs'] = [t[1] for t in model_results]
    results_dict['test_log_aucs'] = [t[2] for t in model_results]
    results_dict['test_roc_aucs'] =  [t[3] for t in model_results]
    results_dict['bedrocs'] =  [t[4] for t in model_results]
    results_dict['precisions'] =  [t[5] for t in model_results]
    results_dict['fdrs'] =  [t[6] for t in model_results]
    results_dict['recalls'] =  [t[7] for t in model_results]
    results_dict['precision_means'] =  [t[8] for t in model_results]
    results_dict['fdr_means'] =  [t[9] for t in model_results]
    results_dict['recall_means'] =  [t[10] for t in model_results]
    results_dict['class_b_aucs'] =  [t[11] for t in model_results]
    results_dict['class_b_recalls'] =  [t[12] for t in model_results]
    results_dict['class_b_precisions'] =  [t[13] for t in model_results]
    results_dict['class_b_fdrs'] =  [t[14] for t in model_results]  
    results_dict['accuracies'] =  [t[15] for t in model_results]    
    results_dict['thresholds_at_full_recall'] =  [t[16] for t in model_results]    
    results_dict['precisions_at_full_recall'] =  [t[17] for t in model_results]    

    return results_dict

def analyze_regression_experiment(results_dict, feature_names,
                                  top_clusters, common_agonists, experiment_name, save_dir):
    test_r2s = results_dict["test_r2s"]
    feature_importances = results_dict["feature_importances"]

    auc_df = pd.DataFrame(np.array(test_r2s), columns=feature_names)    
    plt.style.use('ggplot')
    plt.figure(figsize=(5, 5))
    sns.set_style("darkgrid")
    #g = (auc_df
    #    .pipe((sns.violinplot, 'data'), orient='v', cut=0.))
    g = (auc_df
        .pipe((sns.boxplot, 'data'), orient='v', showfliers=False))
    g.set_xticklabels(auc_df.columns.values, rotation=90)
    sns.despine()
    plt.title(experiment_name)
    plt.ylabel("Frequency of R^2 over Random Splits")
    plt.xlabel("Featurization")
    plt.tight_layout()
    plt.show()
    plt.savefig("%s/%s_r2s.eps" %(save_dir, experiment_name))
    plt.clf()
    
    docking_importances = [f[2] for f in feature_importances]
    importances_df = make_importances_df(docking_importances, top_clusters)
    importances_df.iloc[0:25].plot(kind='barh')
    print(importances_df)
    plt.xlabel("Feature Importance")
    plt.ylabel("MSM State")
    plt.title(experiment_name)
    plt.tight_layout()


    plt.savefig("%s/%s_feature_importances.eps" %(save_dir, experiment_name))
    plt.clf()

    cs = np.logspace(-3., 20.)
    print("Computing regularization path ...")
    clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, binarize(y, 0.2))
        coefs_.append(clf.coef_.ravel().copy())
    
    max_features = 5
    coefs_ = pd.DataFrame(np.array(coefs_), columns=top_clusters, index=np.log10(cs))
    coefs_[importances_df.iloc[:max_features].index].plot(colormap='RdYlBu')
    plt.xlabel("Lasso Parameter")
    plt.ylabel("Coefficient")
    plt.title(experiment_name)
    plt.tight_layout()


    plt.savefig("%s/%s_lasso.eps" %(save_dir, experiment_name))
    plt.clf()
    
    #plot_clustermap(X_df[common_agonists].loc[importances_df.index.values.tolist()[:max_features]].transpose(), save_file="%s/%s_ligands_vs_msm_states_ddg.eps" %(save_dir, experiment_name), method='average', z_score=1)
    
    test_r2s = np.array(test_r2s)
    delta_r2s = np.zeros((test_r2s.shape[0], len(feature_names)-1))
    results_rows = [[np.median(test_r2s[:,0]), 0., (0, 0.)]]
    for i, name in enumerate(feature_names):
        if i == 0: continue
        delta_r2s[:,i-1] = test_r2s[:,i] - test_r2s[:,0]    
        n_successes = len(np.where(delta_r2s[:,i-1] > 0.)[0])
        nobs = delta_r2s.shape[0]
        confint = statsmodels.stats.proportion.proportion_confint(count=n_successes, nobs=nobs, alpha=0.01, method='wilson')
        results_rows.append([np.median(test_r2s[:,i]), np.median(delta_r2s[:,i-1]), confint])
    results_df = pd.DataFrame(results_rows, columns=["Median R^2", "Median delta R^2", "Sign Test 99% CI"], index=feature_names)
    
    return importances_df, results_df

def make_auc_df(test_aucs, featurizer_names, exp_title, save_dir):
  auc_df = pd.DataFrame(np.array(test_aucs), columns=featurizer_names)    
  plt.style.use('ggplot')
  fig = plt.figure(figsize=(5, 5))
  fig.patch.set_alpha(0.)
  ax = fig.add_subplot(111)
  #ax.patch.set_alpha(0.)
  sns.set_style("darkgrid")
  g = (auc_df
      .pipe((sns.violinplot, 'data'), orient='v', cut=0.))
  #g = (auc_df
  #    .pipe((sns.boxplot, 'data'), orient='v', showfliers=True))
  g.set_xticklabels(auc_df.columns.values, rotation=90)
  sns.despine()
  plt.title("ROC AUC Performance Comparison")
  plt.ylabel("Frequency of AUCs over Random Splits")
  plt.xlabel("Featurization")
  plt.tight_layout()
  plt.savefig("%s/%s%s_aucs.eps" %(save_dir, exp_title, str(featurizer_names)))#, transparent=True)
  plt.show()

  if auc_df.shape[1] > 1:
    new_auc_df = auc_df.subtract(auc_df[auc_df.columns.values[0]], axis=0)
    auc_df = new_auc_df[new_auc_df.columns.values[1:]]
    fig = plt.figure(figsize=(5, 5))
    fig.patch.set_alpha(0.)
    ax = fig.add_subplot(111)
    #ax.patch.set_alpha(0.)
    sns.set_style("darkgrid")
    g = (auc_df
        .pipe((sns.violinplot, 'data'), orient='v', cut=0.))
    #g = (auc_df
    #    .pipe((sns.boxplot, 'data'), orient='v', showfliers=True))
    g.set_xticklabels(auc_df.columns.values, rotation=90)
    sns.despine()
    plt.title("ROC AUC Performance Comparison")
    plt.ylabel("Change in AUC vs. Crystal Structures")
    plt.xlabel("Featurization")
    plt.tight_layout()
    plt.savefig("%s/%s%s_auc_deltas.eps" %(save_dir, exp_title, str(featurizer_names)))#, transparent=True)
    plt.show()


"""
Following function takes helpful code from:
http://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
Thanks!
"""
def test_auc_significance(test_aucs, featurizer_names, exp_title, save_dir, fxn=np.median):
  test_aucs = np.vstack(test_aucs)
  print(test_aucs.shape)
  delta_aucs = np.zeros((test_aucs.shape[0], len(featurizer_names)-1))
  results_rows = [[fxn(test_aucs[:,0]), 0., (0, 0.)]]
  for i, name in enumerate(featurizer_names):
      if i == 0: continue
      delta_aucs[:,i-1] = test_aucs[:,i] - test_aucs[:,0]    
      n_successes = len(np.where(delta_aucs[:,i-1] > 0.)[0])
      nobs = delta_aucs.shape[0]
      confint = statsmodels.stats.proportion.proportion_confint(count=n_successes, nobs=nobs, alpha=0.01, method='wilson')
      results_rows.append([fxn(test_aucs[:,i]), fxn(delta_aucs[:,i-1]), confint])
  results_df = pd.DataFrame(results_rows, columns=["Median AUC", "Median delta AUC", "Sign Test 99% CI"], index=featurizer_names)
  print(results_df)
  
  fig, ax = plt.subplots(figsize=(8,6)) # no visible frame
  fig.patch.set_visible(False)
  #ax.axis('off')

  ax.xaxis.set_visible(False)  # hide the x axis
  ax.yaxis.set_visible(False)  # hide the y axis

  table(ax, results_df, fontsize=36.)  # where df is your data frame#

  plt.savefig("%s/%s_%s_auc_significance.eps" %(save_dir, exp_title, str(featurizer_names)))#, transparent=True)
  plt.show()



def plot_clustermap(corr_df, save_file, method='single', row_cluster=True, col_cluster=True, xtick_labelsize=8, ytick_labelsize=8, z_score=0):
  sns.set_style("darkgrid", {"figure.facecolor": "white"})
  plt.rcParams['xtick.labelsize'] = xtick_labelsize
  plt.rcParams['ytick.labelsize'] = ytick_labelsize
  ratio = float(corr_df.shape[0]) / float(corr_df.shape[1])
  if ratio > 1.:
    figsize=(8./ratio , 8.)
  else:
    figsize = (8., 8./ratio)

  #fig = plt.figure(figsize)
  #fig.patch.set_alpha(0.)
  #ax = fig.add_subplot(111)
  g = sns.clustermap(corr_df, z_score=z_score, method=method, row_cluster=row_cluster, col_cluster=col_cluster)
  plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  sns.set(font_scale=0.5)
  
  g.savefig(save_file, facecolor='w', edgecolor='w')
  #plt.show()


def analyze_multiclass_experiment(results_dict, featurizer_names, 
                                  all_feature_names, drug_names, save_dir, 
                                  class_names, X, coef_name="Importance", 
                                  exp_title="", remake=True, fxn=np.mean):

  X_df = pd.DataFrame(X, columns=all_feature_names[1], index=drug_names)

  if not os.path.exists("%s/ligands_vs_msm_states_ddg.eps" %(save_dir)) or remake:
    plot_clustermap(standardize_df(X_df), 
                    save_file="%s/ligands_vs_msm_states_ddg.eps" %(save_dir), 
                    method='average', z_score=None)

  """
  Takes a dictionary mapping model result type (feature importances, AUC's, etc.)
    to the results across many trials.

    For key "feature_importances" or "coefs": list of arrays of feature_importances 
     with one array per trial. The function then finds average feature_importance
     for each feature for each class among all the trials
  """
  print("analyzing precision at full recall:")
  test_auc_significance(results_dict["precisions_at_full_recall"],
                        featurizer_names, "%s_Precision" %exp_title, save_dir, fxn=fxn)

  print("analyzing threshold at full recall:") 
  test_auc_significance(results_dict["thresholds_at_full_recall"],
                        featurizer_names, "%s_Threshold" %exp_title, save_dir, fxn=fxn)


  print("analyzing Accuracy:")
  test_auc_significance(results_dict["accuracies"],
                        featurizer_names, "%s_Accuracy" %exp_title, save_dir, fxn=fxn)

  make_auc_df(results_dict["accuracies"], 
              featurizer_names, "%s_Accuracy" %exp_title, save_dir)

  print("analyzing Recall:")
  test_auc_significance(results_dict["class_b_recalls"],
                        featurizer_names, "%s_Recall" %exp_title, save_dir, fxn=fxn)

  make_auc_df(results_dict["class_b_recalls"], 
              featurizer_names, "%s_Recall" %exp_title, save_dir)

  print("analyzing precision:")
  test_auc_significance(results_dict["class_b_precisions"],
                        featurizer_names, "%s_precision" %exp_title, save_dir, fxn=fxn)

  make_auc_df(results_dict["class_b_precisions"], 
              featurizer_names, "%s_precision" %exp_title, save_dir)

  print("analyzing fdr:")
  test_auc_significance(results_dict["class_b_fdrs"],
                        featurizer_names, "%s_fdr" %exp_title, save_dir, fxn=fxn)

  make_auc_df(results_dict["class_b_fdrs"], 
              featurizer_names, "%s_fdr" %exp_title, save_dir)

  print("analyzing Recall:")
  test_auc_significance(results_dict["recalls"],
                        featurizer_names, "%s_Recall" %exp_title, save_dir, fxn=fxn)

  make_auc_df(results_dict["recalls"], 
              featurizer_names, "%s_Recall" %exp_title, save_dir)

  print("analyzing precision:")
  test_auc_significance(results_dict["precisions"],
                        featurizer_names, "%s_precision" %exp_title, save_dir, fxn=fxn)

  make_auc_df(results_dict["precisions"], 
              featurizer_names, "%s_precision" %exp_title, save_dir)

  print("analyzing fdr:")
  test_auc_significance(results_dict["fdrs"],
                        featurizer_names, "fdr", save_dir, fxn=fxn)

  make_auc_df(results_dict["fdrs"], 
              featurizer_names, "fdr", save_dir)

  print("analyzing class B ROC AUC:")
  test_auc_significance(results_dict["class_b_aucs"],
                        featurizer_names, "%s: Class B AUC" %exp_title, save_dir, fxn=fxn)

  make_auc_df(results_dict["class_b_aucs"], 
              featurizer_names, "%s: Class B AUC" %exp_title, save_dir)

  print("analyzing ROC AUC:")
  test_auc_significance(results_dict["test_roc_aucs"],
                        featurizer_names, "%s AUC" %exp_title, save_dir, fxn=fxn)

  make_auc_df(results_dict["test_roc_aucs"], 
              featurizer_names, "%s AUC" %exp_title, save_dir)

  try:
    print("analyzing BedROC")
    test_auc_significance(results_dict["bedrocs"],
                          featurizer_names, "BedROC", save_dir, fxn=fxn)
    make_auc_df(results_dict["bedrocs"], 
                featurizer_names, "BedROCs", save_dir)
  except:
    print("/n")

  try:
    print("analyzing LogAUC")
    test_auc_significance(results_dict["test_log_aucs"],
                          featurizer_names, "LogAUC", save_dir, fxn=fxn)
    make_auc_df(results_dict["test_log_aucs"], 
                featurizer_names, "LogAUCs", save_dir)
  except:
    print("/n")



  importance_dfs = {}
  for k, featurizer_name in enumerate(featurizer_names):
    feature_names = all_feature_names[k]

    feature_importances = results_dict["feature_importances"]
    avg = np.zeros(feature_importances[0][k].shape)
    print(feature_importances[0][k].shape)
    for feature_importance in feature_importances:
      avg += feature_importance[k]
    avg /= len(feature_importances)
    if len(avg.shape) == 1:
      avg = avg.reshape((1,-1))
    print(avg.shape)
    for class_id in range(0, avg.shape[0]):
      importance_df = pd.DataFrame(avg[class_id,:], 
                                   index=feature_names,
                                   columns=["Importance"]).sort(["Importance"],
                                   inplace=False)

      importance_df["abs"] = importance_df["Importance"].abs() 
      importance_df.sort("abs", inplace=True, ascending=False)
      importance_df = importance_df.groupby(importance_df.index).first().sort(["Importance"], inplace=False, ascending=False).drop("abs", 1)



      if class_id == 0:
        importance_dfs[featurizer_name] = importance_df
                                     
      if importance_df.shape[0] > 30:
        importance_df = pd.concat([importance_df.iloc[:20], 
                                   importance_df.iloc[range(importance_df.shape[0]-20,
                                   importance_df.shape[0])]], axis=0)

      fig = plt.figure()
      ax = fig.add_subplot(111)
      importance_df.plot(kind='barh', ax=ax)
      fig.patch.set_alpha(0.)
      plt.xlabel("Feature Importance")
      plt.ylabel("Feature")
      if avg.shape[0] < 2:
        class_id=1
      title = "Feature Importances: %s" %str(class_names[class_id])
      plt.title(title)
      plt.tight_layout()
      plt.savefig("%s/%s_%s_%s.eps" %(save_dir, exp_title, featurizer_name, title))#, transparent=True)
      plt.show()

  return(importance_dfs)

def standardize_df(df, columns=None):
  new_df = copy.deepcopy(df)
  if columns is not None:
    new_df[columns] = StandardScaler().fit_transform(new_df[columns].values)
    return(new_df)
  else:
    return(pd.DataFrame(StandardScaler().fit_transform(df.values), columns=df.columns, index=df.index))


def analyze_classification_experiment(test_aucs, feature_importances, feature_names,
                        X, y, X_df, top_clusters, common_agonists, experiment_name, save_dir):
    
    auc_df = pd.DataFrame(np.array(test_aucs), columns=feature_names)    
    plt.style.use('ggplot')
    plt.figure(figsize=(5, 5))
    sns.set_style("darkgrid")
    g = (auc_df
        .pipe((sns.violinplot, 'data'), orient='v', cut=0.))
    #g = (auc_df
    #    .pipe((sns.boxplot, 'data'), orient='v', showfliers=True))
    g.set_xticklabels(auc_df.columns.values, rotation=90)
    sns.despine()
    plt.title(experiment_name)
    plt.ylabel("Frequency AUCs over Random Splits")
    plt.xlabel("Featurization")
    plt.tight_layout()

    plt.savefig("%s/%s_aucs.eps" %(save_dir, experiment_name))
    plt.show()
    plt.clf()
    
    docking_importances = [f[1] for f in feature_importances]
    importances_df = make_importances_df(docking_importances, top_clusters)
    importances_df.iloc[0:25].plot(kind='barh')
    print(importances_df)
    plt.xlabel("Feature Importance")
    plt.ylabel("MSM State")
    plt.title(experiment_name)
    plt.tight_layout()


    plt.savefig("%s/%s_feature_importances.eps" %(save_dir, experiment_name))
    plt.clf()

    cs = np.logspace(-3., 20.)
    print("Computing regularization path ...")
    clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, y)
        coefs_.append(clf.coef_.ravel().copy())
    
    max_features = 5
    coefs_ = pd.DataFrame(np.array(coefs_), columns=top_clusters, index=np.log10(cs))
    coefs_[importances_df.iloc[:max_features].index].plot(colormap='RdYlBu')
    plt.xlabel("Lasso Parameter")
    plt.ylabel("Coefficient")
    plt.title(experiment_name)
    plt.tight_layout()


    plt.savefig("%s/%s_lasso.eps" %(save_dir, experiment_name))
    plt.clf()
    

    plot_clustermap(X_df[common_agonists].loc[importances_df.index.values.tolist()[:max_features]].transpose(), save_file="%s/%s_ligands_vs_msm_states_ddg.eps" %(save_dir, experiment_name), method='average', z_score=1)
    
    test_aucs = np.array(test_aucs)
    delta_aucs = np.zeros((test_aucs.shape[0], len(feature_names)-1))
    results_rows = [[np.median(test_aucs[:,0]), 0., (0, 0.)]]
    for i, name in enumerate(feature_names):
        if i == 0: continue
        delta_aucs[:,i-1] = test_aucs[:,i] - test_aucs[:,0]    
        n_successes = len(np.where(delta_aucs[:,i-1] > 0.)[0])
        nobs = delta_aucs.shape[0]
        confint = statsmodels.stats.proportion.proportion_confint(count=n_successes, nobs=nobs, alpha=0.01, method='wilson')
        results_rows.append([np.median(test_aucs[:,i]), np.median(delta_aucs[:,i-1]), confint])
    results_df = pd.DataFrame(results_rows, columns=["Median AUC", "Median delta AUC", "Sign Test 99% CI"], index=feature_names)
    
    
    return importances_df, results_df

def compare_feature_to_apo(lig_features_eq, ligands, reference_ligand, feature, save_file=None):
  #kde_array = np.zeros((len(ligands), lig_features_eq[lig_features_eq.keys()[0]].shape[0]))
  ref_data = lig_features_eq[reference_ligand][feature].values
  custom_bounds = [0.8*ref_data.min(), 1.2*ref_data.max()]
  dx_rows = []
  for i, ligand in enumerate(ligands):
    x, dx =  compute_kde_difference(lig_features_eq[ligand][feature].values,
                                    ref_data, 
                                    custom_bounds=custom_bounds,
                                    n_points=1000)
    dx_rows.append(dx)
  kde_array = np.array(dx_rows).T
  print(kde_array.shape)
  df = pd.DataFrame(kde_array, columns=ligands, index=x)
  df.plot(colormap='rainbow', linewidth=3.)
  plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
  plt.xlabel("%s Distance, Angstroms" %feature)
  plt.ylabel("Change compared to %s" %reference_ligand)
  if save_file is not None:
    plt.savefig(save_file)
def plot_overall_kde(lig_features_eq, ligands, reference_ligand, feature, save_file=None):
  #kde_array = np.zeros((len(ligands), lig_features_eq[lig_features_eq.keys()[0]].shape[0]))
  custom_bounds = [0.8*lig_features_eq[reference_ligand][feature].values.min(), 1.2*lig_features_eq[reference_ligand][feature].values.max()]
  dx_rows = []
  for i, ligand in enumerate(ligands):
    kde = stats.gaussian_kde(lig_features_eq[ligand][feature].values)
    x = np.linspace(custom_bounds[0], custom_bounds[1])
    dx = kde(x)
    dx_rows.append(dx)
  kde_array = np.array(dx_rows).T
  print(kde_array.shape)
  df = pd.DataFrame(kde_array, columns=ligands, index=x)
  df.plot(colormap='rainbow', linewidth=3.)
  plt.xlabel("%s Distance, Angstroms" %feature)
  plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
  if save_file is not None:
    plt.savefig(save_file)

def plot_observable_ligand(cluster_features_averages, docking_df):
  docking_df = docking_df[cluster_features_averages.index]
  for observable in cluster_features_averages.columns.values:
    x = cluster_features_averages[observable].values
    docking_df.columns = x
    new_df = standardize_df(docking_df.transpose())
    new_df.sort_index(inplace=False).plot()
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()

def reweight_observables_with_ddg(observable_df, ddg_df):
  reweighted_array = np.zeros((ddg_df.shape[0], observable_df.shape[0]))
  boltzmann_arr = np.exp(ddg_df.values*0.6)
  observable_arr = observable_df.values
  for i, ligand in enumerate(ddg_df.index.values.tolist()):
    denominator = np.sum(boltzmann_arr[i,:])
    for j in range(0, observable_arr.shape[0]):
      numerator = np.dot(boltzmann_arr[i,:], observable_arr[j,:])
      average = numerator / denominator
      reweighted_array[i][j] = average
  reweighted_df = pd.DataFrame(reweighted_array, index=ddg_df.index, columns=observable_df.index)
  return(reweighted_df)



