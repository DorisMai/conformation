import mdentropy
from mdentropy.core.information import ncmutinf
import numpy as np
import multiprocessing as mp
from functools import partial


"""
Compute Transfer Entropy between all pairs of features
from a list of timeseries datasets.
----------
Parameters:

featurized_timeseries: list
  List of Numpy arrays, each with shape (n_frames, n_features).
lag_time: int
  Lag time to be used in computation of Transfer Entropy
feature_names: list of str
  If your features have names (e.g., "Arg325" or "Temp. in Kansas"),
  you can optionally include them here.

Returns:
tentropy_pairs: Numpy array of shape (1, n_features^2-n_features), containing
  the transfer entropy between each possible pair of features.
tentropy_pairs_id_tuples: List of tuples of ints. 
 In same order as columns of tentropy_pairs.
 Each tuple contains two ints describing the ids of the features for which
 tentropy was calculated. 
tentropy_pairs_names: List of tuples of strings.

Caveats:
Implemented weighted mean is not optimal and may not be numerically stable.
This is an area for improvement.
"""

def compute_tentropy_pairs(featurized_timeseries, lag_time,
                           feature_names=None):

  total_frames = 0.
  n_vars = featurized_timeseries[0].shape[1]
  n_tent_pairs = n_vars ** 2 - n_vars
  tentropy_pairs = np.zeros((1, n_tent_pairs))
  tentropy_pairs_names = []
  tentropy_pairs_id_tuples = []

  for t_id, t in enumerate(featurized_timeseries):
    print("Examining trajectory %d" %t_id)
    k = 0
    t = t[::lag_time]
    
    for i in range(0, t.shape[1]):
      print(i)
      for j in range(0, t.shape[1]):
        if i == j: 
          continue
        x = t[1:,i].reshape((-1,1))
        y = t[:-1,j].reshape((-1,1))
        z = t[:-1,i].reshape((-1,1))
        n_frames = x.shape[0]
        #n_bins = np.floor(np.sqrt(n_frames / 5.))
        n_bins=len(set(t[:,i].astype(int).tolist())) * len(set(t[:,j].astype(int).tolist()))        
        tent = np.nan_to_num(ncmutinf(n_bins, x, y, z, method='grassberger'))

        tentropy_pairs[0, k] += tent * n_frames
        if k==0:
          total_frames += n_frames

        k += 1

        if t_id == 0 and feature_names is not None:
          tentropy_pairs_names.append((feature_names[i], 
                                       feature_names[j]))

  tentropy_pairs /= total_frames

  return tentropy_pairs, tentropy_pairs_id_tuples, tentropy_pairs_names


"""
Called by compute_tentropy_between. Computes transfer entropy between 
all pairs of features between timeseries_i[t_id] and timeseries_j[t_id] at 
lag_time.
"""
def compute_tentropy_between_singletraj(timeseries_tuple, lag_time):
  t_i, t_j = timeseries_tuple
  tentropy_pairs_single = []

  t_i = t_i[::lag_time]
  t_j = t_j[::lag_time]

  k = 0

  for i in range(0, t_i.shape[1]):
    print("Examining, within timeseries, variable %d out of %d" %(i, t_i.shape[1]))
    for j in range(0, t_j.shape[1]):
      x = t_i[1:,i].reshape((-1,1))
      y = t_j[:-1,j].reshape((-1,1))
      z = t_i[:-1,i].reshape((-1,1))
      n_frames = x.shape[0]
      n_bins = None 
      tent = np.nan_to_num(ncmutinf(n_bins, x, y, z, method='grassberger'))

      tentropy_pairs_single.append(tent * n_frames)

      k += 1  

  return np.array(tentropy_pairs_single)

"""
Compute Transfer Entropy between all pairs of features
between two separate lists of featurized timeseries data.
----------
Parameters:

timeseries_i: list
  List of Numpy arrays, each with shape (n_frames, n_features).
timeseries_j: list
  Second list of Numpy arrays, each with shape (n_frames, n_features).
lag_time: int
  Lag time to be used in computation of Transfer Entropy
titles_i: list of str
  If your features have names (e.g., "Arg325" or "Temp. in Kansas"),
  you can optionally include them here for timeseries_i list.
titles_j: list of str
  If your features have names (e.g., "Arg325" or "Temp. in Kansas"),
  you can optionally include them here for timeseries_j list.
worker_pool: object from  iPyParallel module
parallel: bool. If True, will use multiprocessing module 
  to map over all trajectories in parallel.

Returns:
tentropy_array: Numpy array of shape (len(timeseries_i) * len(timeseries_j)), containing
  the transfer entropy between each possible pair of features.
tentropy_pairs_id_tuples: List of tuples of ints. 
 In same order as entries of tentropy_array.
 Each tuple contains two ints describing the ids of the features for which
 tentropy was calculated. 
tentropy_pairs_names: List of tuples of strings.

Caveats:
Implemented weighted mean is not optimal and may not be numerically stable.
This is an area for improvement.
"""

def compute_tentropy_between(timeseries_tuples, lag_time,
                             titles_i=None, titles_j=None, worker_pool=None,
                             parallel=False):
  total_frames = 0.
  n_tent_pairs = timeseries_tuples[0][0].shape[1] * timeseries_tuples[0][1].shape[1]

  tentropy_array = np.zeros(n_tent_pairs)
  tentropy_pairs_names = []
  tentropy_pairs_id_tuples = []

  compute_tentropy_between_singletraj_partial = partial(compute_tentropy_between_singletraj,
                                                        lag_time=lag_time)

  if worker_pool is not None:
    tentropy_pairs = worker_pool.map_sync(compute_tentropy_between_singletraj_partial, timeseries_tuples)
  elif parallel:
    pool = mp.Pool(mp.cpu_count())
    tentropy_pairs = pool.map(compute_tentropy_between_singletraj_partial, timeseries_tuples)
    pool.terminate() 
  else:
    tentropy_pairs = []
    for timeseries_tuple in timeseries_tuples:
      tentropy_pairs.append(compute_tentropy_between_singletraj_partial(timeseries_tuple))

  for arr in tentropy_pairs:
    tentropy_array += arr


  for t_id, timeseries_tuple in enumerate(timeseries_tuples):
    t_i, t_j = timeseries_tuple
    t_i = t_i[::lag_time]
    t_j = t_j[::lag_time]
    total_frames += (t_i.shape[0] - 1)
    if t_id == 0:
      for i in range(0, t_i.shape[1]):
        for j in range(0, t_j.shape[1]):
          if titles_i is not None:
            tentropy_pairs_names.append((titles_i[i], titles_j[j]))
          tentropy_pairs_id_tuples.append((i, j))

  tentropy_array /= float(total_frames)

  return tentropy_array, tentropy_pairs_id_tuples, tentropy_pairs_names
