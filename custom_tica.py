import os
from msmbuilder.decomposition import tICA, SparseTICA
from io_functions import *
import multiprocessing as mp
import glob
from sklearn.preprocessing import StandardScaler
from io_functions import compat_verboseload

def fit_normalizer(features_directory, features=None):
	save_file = "%s/normalizer.h5" %features_directory
	n = StandardScaler()
	if features is None:
		for feature_file in get_trajectory_files(features_directory, ".dataset"):
			n.partial_fit(load_file(feature_file))
	else:
		for feature in features:
			n.partial_fit(feature)

	verbosedump(n, save_file)


def fit_and_transform(features_directory, model_dir, stride=5, lag_time=10, 
					  n_components = 5, wolf = True, shrinkage = None, rho = 0.05,
					  parallel=True, sparse = True, traj_ext = ".h5", normalize=True, 
					  partial_fit=True, subsample=1, recompute_tica=False, features=None):
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	projected_data_filename = "%s/phi_psi_chi2_allprot_projected.h5" %model_dir
	fit_model_filename  = "%s/phi_psi_chi2_allprot_tica_coords.h5" %model_dir
	normalizer = "%s/normalizer.h5" %features_directory
	n = compat_verboseload(normalizer)
	#active_pdb_file = "/scratch/users/enf/b2ar_analysis/renamed_topologies/A-00.pdb"

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

	if not os.path.exists(projected_data_filename) or recompute_tica:
		print("loading feature files")
		feature_files = get_trajectory_files(features_directory, ext = ".dataset")

		if partial_fit:
			transformed_data = []

			for i, feature_file in enumerate(feature_files):
				print("fitting tICA model to %s" %feature_file)
				if features is None:
					featurized_traj = load_file(feature_file)
				else:
					featurized_traj = features[i]
				normalized_featurized_traj = n.transform(featurized_traj)
				tica_model.partial_fit(normalized_featurized_traj)

			print("Finished computing tICA model. Now transforming.")

			for i, feature_file in enumerate(feature_files):
				print("Transforming %s" %feature_file)
				if features is None:
					featurized_traj = load_file(feature_file)
				else:
					featurized_traj = features[i]
				normalized_featurized_traj = n.transform(featurized_traj)
				transformed_data.append(tica_model.partial_transform(n.transform(featurized_traj)))

			fit_model = tica_model


		else:
			if features is None:
				if not parallel:
					features = []
					for feature_file in feature_files:
						#if "A-00" not in feature_file and "A-01" not in feature_file: continue
						#print("Loading feature files one at a time")
						print("loading %s" %feature_file)
						#if sparse: 
						#	features.append(load_features(feature_file)[0:1000,0:10])
						#else:
						
						features.append(load_file(feature_file)[::subsample,:])
				else:
					pool = mp.Pool(mp.cpu_count())
					features = pool.map(load_file, feature_files)
					pool.terminate()



			transpose = False
			for i in range(0, len(features)):
				if np.shape(features[0])[1] != np.shape(features[i])[1]:
					transpose = True
					break 	
			if transpose: 
				for i in range(0, len(features)):
					features[i] = np.transpose(features[i])
			print(np.shape(features[0]))
			#print np.shape(features[1])
			print((features[0][0][0:10]))
			#print(features[1][0][0:10])
			print((np.shape(features)))

			if normalize:
				features = [n.transform(f) for f in features]

			print("fitting data to tICA model")
			fit_model = tica_model.fit(features)

			if subsample == 1:
				transformed_data = fit_model.transform(features)
			else:
				transformed_data = [fit_model.transform(n.transform(load_file(f))) for f in feature_files]
			print("transformed data with tICA model")

		print((fit_model.summarize()))
		#print(dir(fit_model))
		#save_dataset(fit_model, fit_model_filename)
		
		verbosedump(fit_model, fit_model_filename)
		print("saved tICA model")
		verbosedump(transformed_data, projected_data_filename)
		print("saved data projected onto tICA coords")

	else:
		print("already computed tICA model")
		#fit_model = load_file(fit_model_filename)
		#transformed_data = load_file(projected_data_filename)

	#print fit_model.summarize()

	#active_pdb = md.load(active_pdb_file)
	#top = active_pdb.topology
	#atom_indices = atom_indices = [a.index for a in top.atoms if str(a.residue)[0:3] != "POP" and not a.residue.is_water and str(a.residue)[0:2] != "NA" and str(a.residue)[0:2] != "CL"]
	#active_pdb = md.load(active_pdb_file, atom_indices=atom_indices)
	#read_and_featurize_custom(active_pdb_file, condition = "A-00_custom_features", location = "/scratch/users/enf/b2ar_analysis")
	#active_features = [np.transpose(compat_verboseload("/scratch/users/enf/b2ar_analysis/A-00_custom_features.h5"))]
	#active_pdb_projected = fit_model.transform(active_features)
	#print(active_pdb_projected)

def transform(existing_model, features_directory, tica_dir, normalizer):
	model = compat_verboseload(existing_model)
	feature_files = get_trajectory_files(features_directory, ext = ".dataset")
	features = load_file_list(feature_files)
	features = [normalizer.transform(feature) for feature in features]
	tica_coords = model.transform(features)
	tica_coords = np.concatenate(tica_coords)

	if not os.path.exists(tica_dir): os.makedirs(tica_dir)
	np.savetxt("%s/refcoords.csv" %tica_dir, tica_coords, delimiter=",")
	return

	#load features into list
	#transform it with model
	#save to dataset or just a csv file
	#use to make new tica coord plots with inactive and active structures 

def transform_features(feature_filename, model, normalizer):
	return model.partial_transform(normalizer.transform(load_file(feature_filename)))

def transform_to_h5(existing_model_filename, features_directory,
					projected_data_filename, normalizer,
					worker_pool=None, parallel=False):
	model = compat_verboseload(existing_model_filename)
	transform_partial = partial(transform_features, model=model, normalizer=normalizer)
	feature_files = get_trajectory_files(features_directory, ext=".dataset")
	if worker_pool is not None:
		transformed_features = worker_pool.map_sync(transform_partial, feature_files)
	elif parallel:
		pool = mp.Pool(mp.cpu_count()/4)
		transformed_features = pool.map(transform_partial, feature_files)
		pool.terminate()
	else:
		transformed_features = [transform_partial(f) for f in feature_files]
	verbosedump(transformed_features, projected_data_filename)


def check_tica_vs_features(tica_coords_dir, feature_dir):
	tica_coords = compat_verboseload(tica_coords_dir)
	tica_coords = np.concatenate(tica_coords)
	print((np.shape(tica_coords)))
	feature_files = get_trajectory_files(feature_dir, ext = ".h5")
	if len(feature_files) == 0: feature_files = get_trajectory_files(feature_dir, ext = ".dataset")
	pool = mp.Pool(mp.cpu_count())
	features = pool.map(load_features, feature_files)
	pool.terminate()
	if np.shape(features[0])[1] != np.shape(features[1])[1]:
		for i in range(0, len(features)):
			features[i] = np.transpose(features[i])
	features = np.concatenate(features)
	print((np.shape(features)))
	print((np.shape(tica_coords)))

#print out all random forest GINI decreases:
