from msmbuilder.utils import verbosedump, verboseload
from io_functions import *
from functools import partial
from analysis import *
import multiprocessing as mp
import mdtraj as md
#from msmbuilder.cluster import KMeans
#from msmbuilder.cluster import KCenters
from msmbuilder.cluster import MiniBatchKMeans
import random
import json
from sklearn import mixture
from msmbuilder.cluster import GMM
from custom_msm import *

def cluster(data_dir, traj_dir, n_clusters, lag_time):
	clusterer_dir = "/scratch/users/enf/b2ar_analysis/clusterer_%d_t%d.h5" %(n_clusters, lag_time)
	if (os.path.exists(clusterer_dir)):
		print("Already clustered")
	else:
		try:
			reduced_data = verboseload(data_dir)
		except:
			reduced_data = load_dataset(data_dir)
		trajs = np.concatenate(reduced_data)
		clusterer = MiniBatchKMedoids(n_clusters = n_clusters)
		clusterer.fit_transform(reduced_data)
		verbosedump(clusterer, "/scratch/users/enf/b2ar_analysis/clusterer_%d_t%d.h5" %(n_clusters, lag_time))	

def recompute_cluster_means(means, tICs):
	return

def cluster_minikmeans(tica_dir, data_dir, traj_dir, n_clusters, clusterer_dir=None,tICs=None):
	if (os.path.exists(clusterer_dir)):
		reduced_data = load_file(data_dir)
		clusterer = verboseload(clusterer_dir)
		clusterer.labels_ = clusterer.transform(reduced_data)
		verbosedump(clusterer, clusterer_dir)
	else:
		print("Clustering by KMeans")
		try:
			reduced_data = verboseload(data_dir)
		except:
			reduced_data = load_dataset(data_dir)
		if tICs is not None:
			X = []
			for traj in reduced_data:
				X.append(traj[:,tICs])
		else:
			X = reduced_data

		clusterer = MiniBatchKMeans(n_clusters = n_clusters, n_init=10)
		clusterer.fit_transform(X)
		verbosedump(clusterer, clusterer_dir)

def cluster_gmm(projected_features_file, model_file, tICs=None, n_components=25):
	try:
		projected_features = verboseload(projected_features_file)
	except:
		projected_features = load_dataset(projected_features_file)
	X = np.concatenate(projected_features)

	X = []
	for traj in projected_features:
		X.append(traj[:,tICs])

	gmm = GMM(n_components=n_components, covariance_type='diag')
	print("Now fitting GMM model")
	gmm.fit(X)
	labels = gmm.predict(X)
	print("Completed GMM model. Saving now.")

	msmb_gmm = MSMB_GMM(labels, n_components, gmm.means_)
	verbosedump(msmb_gmm, model_file)

class MSMB_GMM(object):
	def __init__(self, labels, n_clusters, centers):
		self.labels_ = labels 
		self.n_clusters = n_clusters
		self.cluster_centers_ = centers

def cluster_kcenters(tica_dir, data_dir, traj_dir, n_clusters, lag_time):
	clusterer_dir = "%s/kcenters_clusterer_%dclusters.h5" %(tica_dir, n_clusters)
	if (os.path.exists(clusterer_dir)):
		print("Already clustered")
	else:
		print("Clustering by KMeans")
		reduced_data = verboseload(data_dir)
		trajs = np.concatenate(reduced_data)
		clusterer = KClusters(n_clusters = n_clusters)
		clusterer.fit_transform(reduced_data)
		verbosedump(clusterer, clusterer_dir)	

def make_clusters_map(clusterer):
	n_clusters = clusterer.n_clusters
	labels = clusterer.labels_
	clusters_map = {}

	for i in range(0,n_clusters):
		clusters_map[i] = set()

	for i in range(0, len(labels)):
		trajectory = labels[i]
		for j in range(0, len(trajectory)):
			cluster = trajectory[j]
			clusters_map[cluster].add((i,j))

	for cluster in list(clusters_map.keys()):
		print(len(clusters_map[cluster]))

	return clusters_map

def cos_to_means(clusterer_dir, features_dir):
	clusterer = verboseload(clusterer_dir)
	clusters_map = make_clusters_map(clusterer)

	features = verboseload(features_dir)
	feature_distances = {}

	for i in range(0, len(list(clusters_map.keys()))):
		indices = clusters_map[i]
		k_mean = clusterer.cluster_centers_[i]
		print(k_mean)
		find_cos_partial = partial(find_cos, k_mean=k_mean, features = features)
		feature_distances_i = list(map(find_cos_partial, indices))
		feature_distances[i] = feature_distances_i

	print((feature_distances[0][0:10]))
	sorted_map = {}

	print((list(feature_distances.keys())))
	print((len(list(feature_distances.keys()))))

	for i in range(0, len(list(feature_distances.keys()))):
		sorted_features = sorted(feature_distances[i], key = lambda x: x[2], reverse = True)
		sorted_map[i] = sorted_features

	print(sorted_map[0][0:10])
	return sorted_map

def find_dist(index, k_mean, features, tICs=None):
		traj = index[0]
		frame = index[1]
		conformation = features[traj][frame]
		a = conformation
		if tICs is not None:
			a = a[tICs]
		b = k_mean
		return (traj, frame, np.linalg.norm(b-a))

def dist_to_means(clusterer_dir, features_dir, n_samples = False, n_components = False, tica_coords_csv = False, kmeans_csv = False, tICs=None):
	clusterer = verboseload(clusterer_dir)
	clusters_map = make_clusters_map(clusterer)

	try: 
		features = verboseload(features_dir)
	except:
		features = load_dataset(features_dir)
	feature_distances = {}

	for i in range(0, len(list(clusters_map.keys()))):
		indices = clusters_map[i]
		k_mean = clusterer.cluster_centers_[i]
		print(k_mean)
		find_dist_partial = partial(find_dist, k_mean=k_mean, features = features, tICs=tICs)
		feature_distances_i = list(map(find_dist_partial, indices))
		feature_distances[i] = feature_distances_i

	print((feature_distances[0][0:10]))
	sorted_map = {}

	print((list(feature_distances.keys())))
	print((len(list(feature_distances.keys()))))

	for i in range(0, len(list(feature_distances.keys()))):
		sorted_features = sorted(feature_distances[i], key = lambda x: x[2], reverse = False)
		sorted_map[i] = sorted_features

	if n_samples is not False and n_components is not False and tica_coords_csv is not False:
		tica_coords_map = {}
		for cluster_id in list(sorted_map.keys()):
			for j in range(0, n_samples):
				sample = "cluster%d_sample%d" %(cluster_id, j)
				sample_tuple = sorted_map[cluster_id][j][0:2]
				sample_coords = features[sample_tuple[0]][sample_tuple[1]]
				tica_coords_map[sample] = sample_coords
		titles = ["sample"]
		for k in range(0, n_components):
			titles.append("component_%d" %k)
		print((list(tica_coords_map.keys())[0]))
		print((tica_coords_map[list(tica_coords_map.keys())[0]]))
		write_map_to_csv(tica_coords_csv, tica_coords_map, titles)

	if kmeans_csv is not False:
		kmeans_map = {}
		for cluster in range(0,clusterer.n_clusters):
			k_mean = clusterer.cluster_centers_[cluster]
			cluster_id = "cluster%d" %cluster
			kmeans_map[cluster_id] = k_mean
		titles = ["cluster"]
		for k in range(0, n_components):
			titles.append("component_%d" %k)
		write_map_to_csv(kmeans_csv, kmeans_map, titles)			


	print(sorted_map[0][0:10]) 
	return sorted_map


def save_md_snapshot(traj, frame, save_dir, lig_name="", save_string="", structure=None, residue_cutoff=10000):
	if structure is None:
		traj_frame = md.load_frame(traj, index=frame)
	else:
		traj_frame = md.load_frame(traj, index=frame, top=structure)
	top = traj_frame.topology
	prot_atoms = [a.index for a in top.atoms if a.residue.is_protein or lig_name in str(a.residue).upper()]
	water_atoms = [a.index for a in top.atoms if a.residue.is_water]

	atom_indices = [a.index for a in top.atoms if str(a.residue)[0:3] != "SOD" and str(a.residue)[0:3] != "CLA" and a.residue.resSeq < residue_cutoff and str(a.residue)[0:3] != "POP" and not a.residue.is_water]
	for idx in atom_indices:
		if idx not in atom_indices:
			atom_indices.append(idx)
	#try: 
	if 1==1:
	#print indices
		if structure is None:
			conformation = md.load_frame(traj, index=frame, atom_indices=sorted(atom_indices))
		else:
			conformation = md.load_frame(traj, index=frame, atom_indices=sorted(atom_indices), top=structure)		
		conformation.save_pdb("%s/%s.pdb" %(save_dir, save_string))
	#except:
	#	print("can't find sample for cluster")

def find_snapshots_within_feature_range(feature_dfs, feature, bounds, 
																				trajectory_filenames, save_dir,
																				save_string, n_save, lig_name="NON", 
																				structure=None):
	traj_frame_pairs = []
	for traj_id, feature_df in enumerate(feature_dfs):
		traj_frames = feature_df.loc[(feature_df[feature] > bounds[0]) & (feature_df[feature] < bounds[1])].index.values.tolist()
		traj_frame_pairs += [(traj_id, traj_frame) for traj_frame in traj_frames]

	for i, traj_frame_pair in enumerate(traj_frame_pairs):
		print(traj_frame_pair)
		if i == n_save: break
		traj = traj_frame_pair[0]
		traj_filename = trajectory_filenames[traj]
		frame = traj_frame_pair[1]
		print(traj_filename)
		print(frame)
		save_md_snapshot(traj_filename, frame, save_dir, lig_name, "%s_%d" %(save_string, i), structure)

def get_sample(traj_frame_cluster_sample, trajectories, structure=None, residue_cutoff=10000, save_dir="", lig_name="UNK", reseed_dir=None):
	traj_id, frame, cluster, sample = traj_frame_cluster_sample
	print(traj_frame_cluster_sample)
	print(traj_id)
	traj = trajectories[traj_id]

	if structure is None:
		traj_frame = md.load_frame(traj, index=frame)
	else:
		traj_frame = md.load_frame(traj, index=frame, top=structure)
	top = traj_frame.topology
	prot_atoms = [a.index for a in top.atoms if a.residue.is_protein or lig_name in str(a.residue).upper()]
	water_atoms = [a.index for a in top.atoms if a.residue.is_water]
	import itertools
	#distances_to_measure = list(itertools.product(prot_atoms,water_atoms))

	#distances = md.compute_distances(frame, distances_to_measure)[0]
	#waters_to_keep = []
	#for i, distance in enumerate(distances):
	#	if distance < 0.5:
	#		water = distances_to_measure[i][1]
	#		if water not in waters_to_keep:
	#			waters_to_keep.append(water)

	atom_indices = [a.index for a in top.atoms if str(a.residue)[0:3] != "SOD" and str(a.residue)[0:3] != "CLA" and a.residue.resSeq < residue_cutoff and str(a.residue)[0:3] != "POP" and not a.residue.is_water]
	for idx in atom_indices:
		if idx not in atom_indices:
			atom_indices.append(idx)

	try: 
	#print indices
		if structure is None:
			conformation = md.load_frame(traj, index=frame, atom_indices=sorted(atom_indices))
			if reseed_dir is not None: 
				conformation2 = md.load_frame(traj, index=frame)
		else:
			conformation = md.load_frame(traj, index=frame, atom_indices=sorted(atom_indices), top=structure)
			if reseed_dir is not None:
				conformation2 = md.load_frame(traj, index=frame, top=structure)

		conformation.save_pdb("%s/cluster%d_sample%d.pdb" %(save_dir, cluster, sample))
		if reseed_dir is not None: 
			conformation2.save_pdb("%s/cluster%d_sample%d.pdb" %(reseed_dir, cluster, sample))
			conformation2.save("%s/cluster%d_sample%d.rst7" %(reseed_dir, cluster, sample)) 
	except:
		print("can't find sample for cluster")


def get_samples(cluster, trajectories, clusters_map, clusterer_dir, features_dir, traj_dir, save_dir, n_samples, method, structure=None, residue_cutoff=10000, save_later=False, lig_name="UNK", reseed_dir=None):
	num_configurations = len(clusters_map[cluster])
	if method == "random":
		try:
			indices = random.sample(list(range(num_configurations)), n_samples)
		except:
			return(list(range(0, min(n_samples, num_configurations))))
		#print indices
	else:
		indices = list(range(0, min(n_samples, num_configurations)))
	if not save_later:
		for s in range(0, n_samples):
			if s == len(clusters_map[cluster]): return(indices[0:s])
			if method != "random":
				k = s
			else:
				k = indices[s]
			sample = clusters_map[cluster][k]
			traj_id = sample[0]
			frame = sample[1]

			traj = trajectories[traj_id]
			print(("cluster %d sample %d" %(cluster, k)))
			#print traj

			#traj_obj = md.load(traj)
			#print traj_obj
			#print frame

			if structure is None:
				top = md.load_frame(traj, index=frame).topology
			else:
				top = md.load_frame(traj, index=frame, top=structure).topology

			atom_indices = [a.index for a in top.atoms if str(a.residue)[0:3] != "SOD" and str(a.residue)[0:3] != "CLA" and a.residue.resSeq < residue_cutoff and str(a.residue)[0:3] != "POP" and not a.residue.is_water]
			#print indices
			if structure is None:
				conformation = md.load_frame(traj, index=frame, atom_indices=sorted(atom_indices))
			else:
				conformation = md.load_frame(traj, index=frame, atom_indices=sorted(atom_indices), top=structure)

			conformation.save_pdb("%s/cluster%d_sample%d.pdb" %(save_dir, cluster, s))
	
	return indices


def sample_clusters(clusterer_dir, features_dir, traj_dir, traj_ext, save_dir, n_samples, method, clusters_map_file = "", tICs=None, structure=None, worker_pool=None):
	if method == "cos":	
		clusters_map = cos_to_means(clusterer_dir, features_dir)
	elif method == "dist":
		clusters_map = dist_to_means(clusterer_dir, features_dir)
	elif method == "random":
		clusters_map = dist_to_means(clusterer_dir, features_dir, tICs=tICs)
	clusters = list(range(0, len(list(clusters_map.keys()))))
	if not os.path.exists(save_dir): os.makedirs(save_dir)
	
	trajectories = get_trajectory_files(traj_dir, traj_ext)
	
	sampler = partial(get_samples, trajectories = trajectories, clusters_map = clusters_map, clusterer_dir = clusterer_dir, features_dir = features_dir, traj_dir = traj_dir, save_dir = save_dir, n_samples = n_samples, method = method, structure=structure, reseed_dir=reseed_dir)
	if worker_pool is not None:
		list_of_indices = worker_pool.map_sync(sampler, clusters)
	else:
		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		list_of_indices = pool.map(sampler, clusters)
		pool.terminate()
	print("Done sampling, now saving clusters map")
	for i in clusters:
		print(i)
		indices = list_of_indices[i]
		if method == "random":
			#print(len(indices))
			#print(indices[0:5])
			clusters_map[i] = [clusters_map[i][j] for j in indices]
	if method == "random":
		with open(clusters_map_file, 'w') as f:
			json.dump(clusters_map, f)


	
	#for cluster in clusters:
	#	if cluster != 118: continue
	#	sampler(cluster)
	#	print cluster





def get_pnas(cluster, clusters_map, pnas_coords, tica_coords, feature_coords, n_samples):
		distances = []
		coords = []
		tica_coords_list = []
		feature_coords_list = []
		print("cluster = %d" %cluster)
		for s in range(0, n_samples):
			print("sample  = %d" %s)
			if s == len(clusters_map[cluster]): return
			sample = clusters_map[cluster][s]
			print(sample)
			traj_id = sample[0]
			frame = sample[1]
			pnas_coord = pnas_coords[traj_id][frame]
			tica_coord = tica_coords[traj_id][frame]
			if feature_coords is not None: 
				feature_coord = feature_coords[traj_id][frame]
				feature_coords_list.append(feature_coord)
			coords.append(pnas_coord)
			tica_coords_list.append(tica_coord)
		return [coords, tica_coords_list, feature_coords_list]



def cluster_pnas_distances(clusterer_dir, features_dir, pnas_coords_dir, projected_features_dir, traj_dir, traj_ext, pnas_coords_csv, tica_coords_csv, feature_coords_csv, n_samples, method, coord_names, clusters_map_file = None):
	if method == "cos":	
		clusters_map = cos_to_means(clusterer_dir, features_dir)
	elif method == "random":
		with open(clusters_map_file) as f:
			clusters_map = json.load(f)
			clusters_map = {int(k):v for k,v in list(clusters_map.items())}
			print((list(clusters_map.keys())))
	elif method == "dist":
		clusters_map = dist_to_means(clusterer_dir, features_dir)
	else: 
		print("method not recognized")
		return
	clusters = list(range(0, len(list(clusters_map.keys()))))

	trajectories = get_trajectory_files(traj_dir, traj_ext)

	pnas_coords = verboseload(pnas_coords_dir)
	try:
		tica_coords = verboseload(projected_features_dir)
	except:
		tica_coords = load_dataset(projected_features_dir)
	feature_coords = None
	if features_dir is not None: feature_coords = load_file_list(None, features_dir, ".dataset")

	sampler = partial(get_pnas, clusters_map = clusters_map, pnas_coords = pnas_coords, tica_coords = tica_coords, feature_coords = feature_coords, n_samples = n_samples)
	num_workers = mp.cpu_count()
	#pool = mp.Pool(num_workers)
	pnas_feature = []
	for cluster in clusters: 
		pnas_feature.append(sampler(cluster))
	#pnas_feature = pool.map(sampler, clusters)
	#pool.terminate()

	pnas_distance_map = {}
	pnas_coords_map = {}
	tica_coords_map = {}
	feature_coords_map = {}

	for i in range(0, len(list(clusters_map.keys()))):
		try:
			pnas_coord = pnas_feature[i][0]
			print(pnas_coord)
			tica_coord = pnas_feature[i][1]
			if features_dir is not None: feature_coord = pnas_feature[i][2]
		except:
			continue
		for j in range(0, len(pnas_coord)):
			pnas_coords_map["cluster%d_sample%d" %(i,j)] = pnas_coord[j]
			tica_coords_map["cluster%d_sample%d" %(i,j)] = tica_coord[j]
			if features_dir is not None: feature_coords_map["cluster%d_sample%d" %(i,j)] = feature_coord[j]

	n_components = len(tica_coords_map[list(tica_coords_map.keys())[0]])

	write_map_to_csv(pnas_coords_csv, pnas_coords_map, ["sample"] + coord_names)
	tic_names = []
	for i in range(0, n_components):
		tic_names.append("tIC_%d" %i)
	write_map_to_csv(tica_coords_csv, tica_coords_map, ["sample"] + tic_names)
	if features_dir is not None: write_map_to_csv(feature_coords_csv, feature_coords_map, [])

def sample_features(clusterer_dir, features_dir, features_ext, n_samples, method, clusters_map_file = None):
	if method == "cos":	
		clusters_map = cos_to_means(clusterer_dir, features_dir)
	elif method == "random":
		with open(clusters_map_file) as f:
			clusters_map = json.load(f)
			clusters_map = {int(k):v for k,v in list(clusters_map.items())}
			print((list(clusters_map.keys())))
	elif method == "dist":
		clusters_map = dist_to_means(clusterer_dir, features_dir)
	else: 
		print("method not recognized")
		return
	clusters = list(range(0, len(list(clusters_map.keys()))))

	features = load_file_list(None, features_dir, features_ext)

	sampler = partial(get_pnas, clusters_map = clusters_map, pnas_coords = pnas_coords, tica_coords = tica_coords, n_samples = n_samples)
	num_workers = mp.cpu_count()
	#pool = mp.Pool(num_workers)
	pnas_feature = []
	for cluster in clusters: 
		pnas_feature.append(sampler(cluster))
	#pnas_feature = pool.map(sampler, clusters)
	#pool.terminate()

	pnas_distance_map = {}
	pnas_coords_map = {}
	tica_coords_map = {}

	for i in range(0, len(list(clusters_map.keys()))):
		try:
			pnas_distance = pnas_feature[i][0]
			print(pnas_distance)
			pnas_coord = pnas_feature[i][1]
			print(pnas_coord)
			tica_coord = pnas_feature[i][2]
		except:
			continue
		for j in range(0, len(pnas_distance)):
			pnas_distance_map["cluster%d_sample%d" %(i, j)] = [pnas_distance[j]]
			pnas_coords_map["cluster%d_sample%d" %(i,j)] = pnas_coord[j]
			tica_coords_map["cluster%d_sample%d" %(i,j)] = tica_coord[j]

	n_components = len(tica_coords_map[list(tica_coords_map.keys())[0]])

	write_map_to_csv(active_pnas_csv, pnas_distance_map, ["sample", "active_pnas_distance"])
	write_map_to_csv(pnas_coords_csv, pnas_coords_map, ["sample", "tm6_tm3_dist", "npxxy_rmsd_inactive", "npxxy_rmsd_active", "connector_rmsd_inactive", "connector_rmsd_active"])
	tic_names = []
	for i in range(0, n_components):
		tic_names.append("tIC_%d" %i)
	write_map_to_csv(tica_coords_csv, tica_coords_map, ["sample"] + tic_names)


def reseed_from_clusterer(clusterer_file, main, tica_dir, projected_features_dir, traj_files): 
	clusterer = verboseload(clusterer_file)
	n_clusters = len(clusterer.cluster_centers_)
	print(n_clusters)
	clusters_map = make_clusters_map(verboseload(clusterer_file))
	count_tuples = []
	for i in range(0,n_clusters):
	    count_tuples.append((i, len(clusters_map[i])))
	count_tuples.sort(key=operator.itemgetter(1))
	min_populated_clusters = [count_tuples[i][0] for i in range(0,16)]
	print(min_populated_clusters)
	plot_all_tics_and_clusters(tica_dir, projected_features_dir, clusterer_file, None, tic_range = [0], main=main, label = "cluster_id", active_cluster_ids = min_populated_clusters)

	traj_index_frame_pairs = list(find_closest_indices_to_cluster_center(projected_features_dir, clusterer_file))
	traj_index_frame_pairs = [tuple(pair) for pair in traj_index_frame_pairs]

	for i, traj_index_frame_pair in enumerate(traj_index_frame_pairs):
	    traj_index, frame = traj_index_frame_pair
	    if i in min_populated_clusters:
	        print("Looking at cluster %d" % i)
	        print("Snapshot in: %s" %str(traj_index_frame_pair))
	        snapshot = md.load_frame(traj_files[traj_index], index=frame)
	        snapshot.save("%s/%smincount_snapshot_cluster%d.rst7" % (tica_dir, main, i))
	        snapshot.save("%s/%smincount_snapshot_cluster%d.pdb" % (tica_dir, main, i))
	        protein_indices = [a.index for a in snapshot.topology.atoms if a.residue.is_protein or "LIG" in str(a.residue)]
	        snapshot_protein = snapshot.atom_slice(protein_indices)
	        snapshot_protein.save("%s/%smincount_snapshot_cluster%d_protein.pdb" %(tica_dir, main, i))

	return(min_populated_clusters)
  

def sample_cluster(traj_index_frame_pairs, trajectories, structure, residue_cutoff, save_dir, lig_name, reseed_dir, cluster):
	cluster_tuples = traj_index_frame_pairs[cluster]
	print(cluster_tuples[0].shape)
	if len(cluster_tuples[0].shape) < 1:
		print("Converting to list")
		cluster_tuples = [cluster_tuples]

	for sample, cluster_tuple in enumerate(cluster_tuples):
		traj_index = cluster_tuple[0]
		frame = cluster_tuple[1]
		traj_frame_cluster_sample = [traj_index, frame, cluster, sample]
		get_sample(traj_frame_cluster_sample, trajectories, structure=None, residue_cutoff=10000, save_dir=save_dir, lig_name=lig_name, reseed_dir=reseed_dir)

def sample_from_clusterer(clusterer_file, projected_features_dir, traj_files, 
						  n_samples, save_dir, samples_indices_file, structure=None,
						  residue_cutoff=10000, parallel=False,
						  worker_pool=None, lig_name="UNK", reseed_dir=None): 
	clusterer = verboseload(clusterer_file)
	n_clusters = len(clusterer.cluster_centers_)
	
	traj_index_frame_pairs = find_closest_indices_to_cluster_center(projected_features_dir, clusterer_file, k=n_samples)
	print(traj_index_frame_pairs)
	print(len(traj_index_frame_pairs))
	sample_cluster_partial = partial(sample_cluster, traj_index_frame_pairs, traj_files, structure, residue_cutoff, save_dir, lig_name, reseed_dir)
	if worker_pool is not None:
		worker_pool.map_sync(sample_cluster_partial, range(0, n_clusters))
	elif parallel:
		pool = mp.Pool(mp.cpu_count())
		pool.map(sample_cluster_partial, range(0, n_clusters))
		pool.terminate()
	else:
		for cluster in range(0, n_clusters):
			sample_cluster_partial(cluster)

	verbosedump(traj_index_frame_pairs, samples_indices_file)


