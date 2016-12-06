from msmbuilder.decomposition import tICA, SparseTICA
from sklearn.kernel_approximation import Nystroem as ScikitNystroem
from LandmarkNystroem import Nystroem as LandmarkNystroem
from msmbuilder.decomposition.base import MultiSequenceDecompositionMixin
from msmbuilder.utils import verbosedump, verboseload
import numpy as np
import multiprocessing as mp
from io_functions import *
import json
import inspect
from scipy.linalg import svd
import types
from sklearn.metrics.pairwise import pairwise_kernels
import tables


class Nystroem(MultiSequenceDecompositionMixin, LandmarkNystroem):
	pass

def ktica(features, landmarks, projected_data_filename, nystroem_data_filename, fit_model_filename, sparse = False, shrinkage = 0.05, wolf = True, rho = 0.01,
					n_components=25, lag_time=5, refcoords_csv=None):
	if not sparse:
		if shrinkage is None:
			tica_model = tICA(n_components = n_components, lag_time = lag_time)
		else:
			if wolf:
				tica_model = tICA(n_components = n_components, lag_time = lag_time, shrinkage = shrinkage)
			else:
				tica_model = tICA(n_components = n_components, lag_time = lag_time, gamma = shrinkage)

		
	else:
		if shrinkage is None:
			tica_model = SparseTICA(n_components = n_components, lag_time = lag_time, rho = rho)
		else:
			tica_model = SparseTICA(n_components = n_components, lag_time = lag_time, rho = rho, shrinkage = shrinkage)


	if not os.path.exists(nystroem_data_filename):
		nys = Nystroem(n_components = np.shape(landmarks)[0], basis = landmarks)#np.shape(landmarks)[0])# basis=landmarks)
		nyx = nys.fit_transform(features)
		print("Computed Nystroem.")
		del features
		del landmarks
		try:
			save_dataset(nyx, nystroem_data_filename)
		except:
			os.system("rm -rf %s" %nystroem_data_filename)
			save_dataset(nyx, nystroem_data_filename)
	else:
		nyx = load_dataset(nystroem_data_filename)
		print("Loaded Nystroem")

	if not os.path.exists(fit_model_filename):
		print("Fitting Kernel tICA model")
		fit_model = tica_model.fit(nyx)
		verbosedump(fit_model, fit_model_filename)
		transformed_data = fit_model.transform(nyx)
		del(nyx)
		try:
			save_dataset(transformed_data, projected_data_filename)
		except:
			os.system("rm -rf %s" %projected_data_filename)
			save_dataset(transformed_data, projected_data_filename)
	else:
		fit_model = verboseload(fit_model_filename)
		transformed_data = fit_model.transform(nyx)
		os.system("rm -rf %s" %projected_data_filename)
		save_dataset(transformed_data, projected_data_filename)
		if refcoords_csv is not None:
			np.savetxt(refcoords_csv, transformed_data, delimiter=",")
	return

def landmark_ktica(features_dir, combined_features_file=None, ktica_dir="", feature_ext="dataset", use_clusters_as_landmarks=True, clusters_map_file = "", 
	landmarks_dir = "", nystroem_components=1000, n_components=10, lag_time=5, nystroem_data_filename = "", 
	fit_model_filename = "", projected_data_filename = "", landmark_subsample=10, 
	sparse = False, shrinkage = 0.05, wolf = False, rho = 0.01, refcoords_csv=None):
	'''
	features_dir: string, directory where your featurized trajectories are kept. 
	combined_features_dir: if you have a file containing all featurized trajectories in one file, i.e. as a list of np arrays, this is it.
	feature_ext: if instead of a combined file of features they are in separate files, what is the extension of your feature files? 
	use_clusters_as_landmarks: this is if you are doing a composition of tICA --> clustering --> Nystroem --> tICA. this is what I do. 
		if true, you need to feed it a json file containing a dictionary that maps cluster name --> list of 2-tuples, where each tuple has 
		(trajectory_id, frame_number pairs). So this way, instead of choosing landmark points at random in the Nystroem approximation, you
		are using regular linear tICA-driven clustering to choose your landmark points more efficiently. 
	landmarks_dir: directory where you will save the landmarks. this should be a file containing a list of 1d np arrays or a 2d array
	nystroem_components: the number of landmarks to use. 
	n_components: the number of ktICA components to compute.
	lag_time: lag time of tICA 
	nystroem_data_filename: where you will save Nystroem object
	fit_model_filename: the filename of the ktICA object to save.
	projected_data_filename: where you will save the features projected with kernel tICA 
	landmark_subsample= how frequently to subsample the landmarks if you are doing use_clusters_as_landmarks.
	sparse: set to False. 
	shrinkage: same as gamma in old version of tICA. you might want to mess with this. 
	wolf = False: keep this as true unless you're using Robert's branch of msmbuilder
	rho = Ignore this. 

	'''
	print(landmark_subsample)
	if not os.path.exists(ktica_dir): os.makedirs(ktica_dir)


	if not os.path.exists(nystroem_data_filename):
		if combined_features_file is not None and os.path.exists(combined_features_file): 
			features = verboseload(combined_features_file)
		else:
			features = load_file_list(get_trajectory_files(features_dir, ext = feature_ext))

		if os.path.exists(landmarks_dir):
			landmarks = verboseload(landmarks_dir)
			print((np.shape(landmarks)))
		else:
			if use_clusters_as_landmarks:
				print("Using cluster centers as landmarks")
				with open(clusters_map_file) as f:
					clusters_map = json.load(f)
					clusters_map = {int(k):v for k,v in list(clusters_map.items())}
					landmarks = []
					for cluster_id,sample_list in list(clusters_map.items()):
						for sample in sample_list:
							traj = sample[0]
							frame = sample[1]
							landmark = features[traj][frame]
							landmarks.append(landmark)
					landmarks = [landmarks[i] for i in range(0,np.shape(landmarks)[0]) if i%landmark_subsample==0] #%landmark_subsample == 0]

					print("Landmarks have shape: ")
					print((len(landmarks)))
					print((np.shape(landmarks)))
					verbosedump(landmarks, landmarks_dir)
			else: 
				n = np.shape(features)[0]
				indices = np.random.choice(n, nystroem_components)
				features_concatenated = np.concatenate(features)
				landmarks = features_concatenated[indices,:]
				verbosedump(landmarks, landmarks_dir)
	else:
		if combined_features_file is not None and os.path.exists(combined_features_file): 
			features = verboseload(combined_features_file)
		else:
			features = load_file_list(get_trajectory_files(features_dir, ext = feature_ext))
		
		print("np.shape(features)")
		print(np.shape(features))

		landmarks = verboseload(landmarks_dir)
		print((np.shape(landmarks)))

	ktica(features, landmarks, projected_data_filename, nystroem_data_filename, fit_model_filename, sparse, shrinkage, wolf, rho,
				n_components=n_components, lag_time=lag_time, refcoords_csv=refcoords_csv)


def landmark_ktica_ticaTraj(tica_dir, clusterer_dir, ktica_dir, clusters_map_file = "", landmarks_dir = "", nystroem_components=1000, n_components=10, lag_time=5, nystroem_data_filename = "", fit_model_filename = "", projected_data_filename = "", landmark_subsample=1, sparse = False, wolf = True, rho = 0.01, shrinkage = None):
	if not os.path.exists(ktica_dir): os.makedirs(ktica_dir)
	
	if not sparse:
		if shrinkage is None:
			tica_model = tICA(n_components = n_components, lag_time = lag_time)
		else:
			tica_model = tICA(n_components = n_components, lag_time = lag_time, shrinkage = shrinkage)
		
	else:
		if shrinkage is None:
			tica_model = SparseTICA(n_components = n_components, lag_time = lag_time, rho = rho)
		else:
			tica_model = SparseTICA(n_components = n_components, lag_time = lag_time, rho = rho, shrinkage = shrinkage)

	if not os.path.exists(nystroem_data_filename):
		clusterer = verboseload(clusterer_dir)
		tica = verboseload(tica_dir)
		features = tica
		clusters = clusterer.cluster_centers_
		landmarks = clusters

		print("here's what goes into the combined class:")
		#print(np.shape(features))
		print((np.shape(landmarks)))
		print((type(landmarks)))
		nys = Nystroem(n_components = np.shape(landmarks)[0], basis = landmarks)#np.shape(landmarks)[0])# basis=landmarks)
		nyx = nys.fit_transform(features)
		del features
		del landmarks
		try:
			save_dataset(nyx, nystroem_data_filename)
		except:
			os.system("rm -rf %s" %nystroem_data_filename)
			save_dataset(nyx, nystroem_data_filename)
	else:
		nyx = load_dataset(nystroem_data_filename)

	print((np.shape(nyx)))
	print((dir(nyx)))

	if not os.path.exists(projected_data_filename):
		fit_model = tica_model.fit(nyx)
		verbosedump(fit_model, fit_model_filename)
		transformed_data = fit_model.transform(nyx)
		del(nyx)
		try:
			save_dataset(transformed_data, projected_data_filename)
		except:
			os.system("rm -rf %s" %projected_data_filename)
			save_dataset(transformed_data, projected_data_filename)
	else:
		print("Already performed landmark kernel tICA.")
	

def ktica_test(features_dir, tica_dir, landmark_indices = None, nystroem_components=1000, tica_components=10, lag_time=5, nystroem_data_filename = "", fit_model_filename = "", projected_data_filename = ""):
	nys = Nystroem(n_components=nystroem_components)
	tica_model = tICA(n_components = tica_components, lag_time = lag_time)
	feature_files = get_trajectory_files(features_dir, ext = ".h5")[0:3]

	#if os.path.exists(nystroem_data_filename):
	#	nyx = verboseload(nystroem_data_filename)
	#else:
	features = load_file_list(feature_files)
	nyx = nys.fit_transform(features)
	verbosedump(nyx, nystroem_data_filename)

	print((np.shape(nyx)))
	print((dir(nyx)))

	fit_model = tica_model.fit(nyx)
	verbosedump(fit_model, fit_model_filename)
	transformed_data = fit_model.transform(nyx)
	verbosedump(transformed_data, projected_data_filename)

	return
	
	#features = 





