import numpy as np
from msmbuilder.utils import verbosedump, verboseload
import msmbuilder.tpt as tpt 
import msmbuilder.msm as msm 
import random 
import pandas as pd

def resample_by_msm(total_samples, msm_object, clusters_map, num_trajs, save_file, equilibrium_populations=None):
  if equilibrium_populations is None:
    equilibrium_populations = msm_object.populations_

  num_to_sample_per_cluster = {}
  for cluster_id in msm_object.mapping_.keys():
    state_id = msm_object.mapping_[cluster_id]
    num_to_sample_per_cluster[cluster_id] = np.rint(equilibrium_populations[state_id] * total_samples)

  print("Found number to sample per cluster based on equilibrium proporrtions.")
  sample_pairs = []
  for cluster_id in msm_object.mapping_.keys():
    traj_index_pairs = list(clusters_map[cluster_id])
    if len(traj_index_pairs) == 0:
      continue
    num_to_sample = num_to_sample_per_cluster[cluster_id]
    random_indices = np.random.choice(range(0, len(traj_index_pairs)), size=num_to_sample, replace=True)
    clusters_sample_pairs = [traj_index_pairs[i] for i in random_indices]
    sample_pairs += clusters_sample_pairs

  print("Obtained random (trajectory, frame) pairs based on equilibrium populations")
  #if there exists some fancy numpy way to index a 3d array by 2d tuples, then great, else:
  traj_to_frames = {}
  for i in range(0, num_trajs):
    traj_to_frames[i] = []

  for sample_pair in sample_pairs:
    traj_to_frames[sample_pair[0]].append(sample_pair[1])

  print("Rearranged equilibrium sampled frames based on trajectories")

  if save_file is not None:
    verbosedump(traj_to_frames, save_file)
  return traj_to_frames

def traj_index_series_from_dict(traj_index_dict):
  resampled_traj_index_pairs = []
  for traj in traj_index_dict.keys():
      [resampled_traj_index_pairs.append((traj, frame)) for frame in traj_index_dict[traj]]
  return resampled_traj_index_pairs

def resample_features_by_msm_equilibirum_pop(features, traj_to_frames, save_file=None):
  resampled_features = []
  for traj_index, frames in traj_to_frames.items():
    if isinstance(features[0], pd.DataFrame):
      resampled_features.append(features[traj_index].iloc[frames])
    else:
      resampled_features.append(features[traj_index][frames, :])
  
  if isinstance(features[0], pd.DataFrame):
    resampled_features = pd.concat(resampled_features, axis=0)
  else:
    resampled_features = np.concatenate(resampled_features)

  if save_file is not None:
    verbosedump(resampled_features, save_file)
  else:
    return resampled_features

def generate_msm_traj_index_series(msm_object, start_cluster, n_steps, clusters_map, save_file=None):
  inv_map = {v: k for k, v in msm_object.mapping_.items()}
  msm_trajectory = msm_object.sample_discrete(state=start_cluster, n_steps=n_steps)

  traj_index_pairs = []
  clusters = []
  for state in msm_trajectory:
    cluster = state #inv_map[state]
    clusters.append(cluster)
    traj_index_pair = random.choice(list(clusters_map[cluster]))
    traj_index_pairs.append(traj_index_pair)

  if save_file is not None:
    verbosedump(traj_index_pairs, save_file)
  return traj_index_pairs, clusters

def generate_tpt_traj_index_series(msm_object, sources, sinks, clusters_map, num_paths, remove_path, save_file):
  net_flux = tpt.net_fluxes(sources, sinks, msm_object)
  tpt_paths = tpt.paths(sources, sinks, net_flux, remove_path=remove_path, 
                        num_paths=num_paths, flux_cutoff=0.5)

  inv_map = {v: k for k, v in msm_object.mapping_.items()}

  print(tpt_paths)
  traj_index_pairs_list = []
  for path in tpt_paths[0]:
    print("path = %s" %(str(path)))
    traj_index_pairs = []
    for state in path:
      cluster = inv_map[state]
      traj_index_pair = random.choice(list(clusters_map[cluster]))
      traj_index_pairs.append(traj_index_pair)
    traj_index_pairs_list.append(traj_index_pairs)

  verbosedump(traj_index_pairs_list, save_file)

  inv_tpt_paths = []
  for tpt_path in tpt_paths[0]:
    inv_tpt_paths.append([inv_map[i] for i in tpt_path])
  return tpt_paths[0], inv_tpt_paths, traj_index_pairs_list

def resample_features_by_msm_trajectory(features, traj_index_pairs):
  msm_featurized_traj = np.zeros((len(traj_index_pairs), features[0].shape[1]))
  df = False
  for i, traj_index_pair in enumerate(traj_index_pairs):
    try:
      msm_featurized_traj[i,:] = features[traj_index_pair[0]][traj_index_pair[1],:]
    except:
      msm_featurized_traj[i,:] = features[traj_index_pair[0]].values[traj_index_pair[1],:] 
      df = True
  if df: 
    msm_featurized_traj = pd.DataFrame(msm_featurized_traj, columns=features[0].columns.values)     
  return msm_featurized_traj
  