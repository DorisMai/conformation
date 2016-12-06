from io_functions import *
import os
from functools import partial
import multiprocessing as mp

def subsample_feature(feature_file, subsampled_dir):
	traj_name = os.path.basename(feature_file)
	traj_name = os.path.splitext(traj_name)[0]
	print(traj_name)
	feature = load_file(feature_file)
	subsampled = feature[::5,:]
	print(np.shape(subsampled))
	print("Saving to:")
	print(os.path.join(subsampled_dir, "%s.npy" % traj_name))
	np.save(os.path.join(subsampled_dir, "%s.npy" % traj_name), subsampled)

def subsample_features(features_dir, feature_ext):
	subsampled_dir = os.path.join(features_dir, "subsampled")
	if not os.path.exists(subsampled_dir):
		os.makedirs(subsampled_dir)

	feature_files = get_trajectory_files(features_dir, feature_ext)

	subsample_feature_parallel = partial(subsample_feature, subsampled_dir=subsampled_dir)
	pool = mp.Pool(mp.cpu_count())
	pool.map(subsample_feature_parallel, feature_files)
	pool.terminate()
	#for feature_file in feature_files:
	#	subsample_feature_parallel(feature_file)
	"""
	for feature_file in feature_files:
		traj_name = os.path.basename(feature_file)
		traj_name = os.path.splitext(traj_name)[0]
		print(traj_name)
		feature = load_file(feature_file)
		subsampled = []
		for traj_feature in feature:
			subsampled.append(traj_feature[::5])
		np.save(os.path.join(subsampled_dir, "%s.npy" % traj_name), subsampled)
	"""