import mdtraj as md
import os
import numpy as np
import h5py
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import KMeans
from msmbuilder.msm import MarkovStateModel
from sklearn.pipeline import Pipeline
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.utils import verbosedump, verboseload
from msmbuilder.dataset import dataset
from matplotlib import pyplot as plt
from msmbuilder.cluster import KMedoids
import datetime
import multiprocessing as mp
import glob
 	

def generateData(files):
	for data in files:
		print(data)
		yield verboseload(data)

def generateTraj(files, top=None):
	for traj in files:
		if top == None:
			yield md.load(file)
		else:
			yield md.load(file, top=top)

def get_trajectory_files(traj_dir):
	traj_files = []
	for traj in os.listdir(traj_dir):
			if traj.endswith(".dcd") or traj.endswith(".h5"):
				traj_files.append("%s/%s" %(traj_dir,traj))
	return sorted(traj_files)

def read_trajectory(directory, filename, stride=10):
	print(("reading %s" %(filename)))
	traj = md.load(filename, stride=stride, top="/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-H-05-all/system.pdb")
	return traj

def read_and_save_traj(traj, stride=10):
	directory = traj.split("/")
	simulation = directory[len(directory)-2]
	dcd_file = directory[len(directory)-1]
	print(("analyzing simulation %s file %s" %(simulation, dcd_file)))
	top_file = directory[0:len(directory)-2]
	#print(top_file)
	top_file.append("system.pdb")
	#print(top_file)
	top_file = '/'.join(top_file)
	#print("topology file = %s" %(top_file))
	#print("traj = %s" %traj)
	traj = md.load(traj,stride=stride,top=top_file)
	#print("loaded trajectory")
	new_file = "%s_stride%d.h5" %(dcd_file.rsplit( ".", 1 )[ 0 ] , stride)
	new_root_dir = "/home/enf/b2ar_analysis/subsampled"

	condition = "%s-%s" %(simulation.split('-')[1], simulation.split('-')[2])

	new_condition_dir = "%s/%s" %(new_root_dir, condition)

	if not os.path.exists(new_condition_dir):
		os.makedirs(new_condition_dir)

	new_file_full = "%s/%s/%s" %(new_root_dir, condition, new_file)
	print(("saving trajectory as %s" %new_file_full))
	traj.save(new_file_full)


def read_and_featurize(filename, dihedrals=['phi','psi','chi2'], stride=10):
	print(("reading and featurizing %s" %(filename)))
	traj = md.load(filename).select('chain A and protein')
	featurizer = DihedralFeaturizer(types = dihedrals)
	features = featurizer.transform(traj_list = traj)
	print("finished featurizing")

	directory = filename.split("/")
	condition = directory[len(directory)-2]
	dcd_file = directory[len(directory)-1]
	new_file = "%s_features_stride%d.h5" %(dcd_file.rsplit( ".", 1 )[ 0 ] , stride)
	new_root_dir = "/home/enf/b2ar_analysis/subsampled_features/"
	new_condition_dir = "%s/%s" %(new_root_dir, condition)

	if not os.path.exists(new_condition_dir):
		os.makedirs(new_condition_dir)

	new_file_full = "%s/%s/%s" %(new_root_dir, condition, new_file)
	print(("saving features as %s" %new_file_full))

	verbosedump(features, new_file_full)
	return features

def featurize(directory):
	simulations = os.listdir(directory)
	for simulation in simulations:
		sim_dir = "%s/%s" %(directory, simulation)
		
		new_dir = "%s/%s" %("/home/enf/b2ar_analysis/subsampled_features", simulation)
		if os.path.exists(new_dir):
			print("we have already featurized this simulation")
			continue

		print(("currently analyzing %s " %sim_dir))
		trajs = get_trajectory_files(sim_dir)
		

		print(("there are %d cpus" %(mp.cpu_count())))
		pool = mp.Pool(mp.cpu_count())
		features_i = pool.map(read_and_featurize, trajs)
		features = [np.concatenate(np.concatenate(features_i))]
		combined_dir = "/home/enf/b2ar_analysis/combined_features"
		new_file_name = "%s_combined.h5" %(simulation)
		new_file = "%s/%s" %(combined_dir, new_file_name)
		print(("saving concatenated features for %s as %s" %(simulation, new_file)))
		verbosedump(features, new_file)

def fit_and_transform(directory):
	print("fitting data to tICA model")

	tica_model = tICA(n_components=4)

	features = generateData(get_trajectory_files(directory))
	for data in features:
		print((np.shape(data[0])))
		tica_model.partial_fit(data[0])
		print("Fitting: ")
		print(data)

	transformed_data = []
	for data in features:
		print("Transforming: ")
		print(data)
		transformed_data.append(tica_model.partial_transform(data))
		
	verbosedump(transformed_data, "/home/enf/b2ar_analysis/phi_psi_chi_stride10_projected.h5")
	trajs = np.concatenate(transformed_data)
	plt.hexbin(trajs[:,0], trajs[:,1], bins='log', mincnt=1)
	plt.show()
	

def subsample(directory):
	conditions = os.listdir(directory)
	for condition in conditions: 
		subdirectory = "%s/%s" %(directory, condition)
		traj_dir = condition.split("_")[1]

		new_root_dir = "/home/enf/b2ar_analysis/subsampled"
		sim_dir = "%s-%s" %(traj_dir.split('-')[1], traj_dir.split('-')[2])
		new_condition_dir = "%s/%s" %(new_root_dir, sim_dir)
		#print(new_condition_dir)
		if os.path.exists(new_condition_dir):
			print("We have already subsampled this simulation")
			continue
		traj_dir = "%s/%s" %(subdirectory, traj_dir)
		traj_files = get_trajectory_files(traj_dir)
		#print("Subsampling %d trajectories in condition %s" %(len(traj_files), condition))

		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers // 2)
		chunksize = 8
		#print("dividing %d traj_files into %d chunks" %(len(traj_files), chunksize))
		pool.map(read_and_save_traj, traj_files, chunksize = 8)
				


#subsample("/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs")
#featurize("/home/enf/b2ar_analysis/subsampled")
fit_and_transform("/home/enf/b2ar_analysis/combined_features")

'''
For each subdirectory i in DESRES:
	For each subdir j in i: 
		get list of all dcd files
		split subdirectory to find the correct directory to save the file
		apply function read_and_save_traj on each dcd file
'''

'''

traj_dir = "/home/harrigan/data/gpcr/DESRES/DESRES-Trajectory_pnas2011b-H-05-all/pnas2011b-H-05-all"
traj_files = get_trajectory_files(traj_dir)[0:5]

a = datetime.datetime.now().replace(microsecond=0)
trajectories = map(read_trajectory, traj_files)
#trajectories = []
#for traj_file in traj_files:
#	trajectories.append(read_trajectory(traj_file))
b = datetime.datetime.now().replace(microsecond=0)
print(b-a)
'''


'''
dataset = []
trajs = []

traj_dir = "/home/harrigan/data/gpcr/DESRES/DESRES-Trajectory_pnas2011b-H-05-all/pnas2011b-H-05-all"
traj_files = []

if not (os.path.isfile("/home/enf/b2ar_analysis/H-05/%s" %("combined_traj_stride10.h5"))):
	print("traj not loaded yet")
	for traj in os.listdir(traj_dir):
		if traj.endswith(".dcd"):
			traj_files.append("%s/%s" %(traj_dir,traj))
	traj_files.sort()
	traj = md.load(traj_files, top = "/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-H-05-all/system.pdb", stride=10)
	traj = traj[0].join(traj[1:])
	traj.save("/home/enf/b2ar_analysis/H-05/%s" %("combined_traj_stride10.h5"))
else:
	print("loading h5 traj")
	traj = md.load("/home/enf/b2ar_analysis/H-05/%s" %("combined_traj_stride10.h5"))

'''
'''
if not (os.path.isfile("phi_psi_chi2_features_vd_stride10.h5")):
	print("featurizing")
	phi_psi_chi2 = DihedralFeaturizer(types=['phi','psi','chi2'])
	features = phi_psi_chi2.transform(traj_list = traj)
	print("finished featurizing")
	verbosedump(features, "phi_psi_chi2_features_vd_stride10.h5")
else:
	print("loading existing features")
	features = verboseload("phi_psi_chi2_features_vd_stride10.h5")

if not (os.path.isfile("reduced_phi_psi_chi_stride10.h5")):
	print("Fitting tICA model")
	tica_model = tICA(n_components=4)
	fitted_model = tica_model.fit(features)
	reduced_data = fitted_model.transform(features)
	verbosedump(reduced_data, "reduced_phi_psi_chi_stride10.h5")
	print(tica_model.summarize())
else:
	reduced_data = verboseload("reduced_phi_psi_chi_stride10.h5")

clusterer = KMedoids(n_clusters=9)

clusters = clusterer.fit_transform(reduced_data)[0]

center_locations = []

for i in range(0, len(clusters)):
	print i
	for j in range(0, len(clusterer.cluster_centers_)):
		if np.linalg.norm(reduced_data[0][i] - clusterer.cluster_centers_[j]) < 0.001:
			print("found match")
			center_locations.append(i)

print(center_locations)

for center in center_locations:
	frame = md.load_frame("combined_traj_stride10.h5", index=center)
	frame.save_pdb("frame_%d.pdb" %(center))






#trajs = np.concatenate(reduced_data)
#plt.hexbin(trajs[:,0], trajs[:,1], bins='log', mincnt=1)
#plt.show()
'''


