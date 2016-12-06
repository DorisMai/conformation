#from PDB_Order_Fixer import PDB_Order_Fixer
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
from msmbuilder.cluster import MiniBatchKMedoids
import datetime
import multiprocessing as mp
import glob
import copy
import gc
from functools import partial 
import itertools

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

def subsample_traj(traj, stride=5, top=None):
	directory = traj.split("/")
	simulation = directory[len(directory)-2]
	dcd_file = directory[len(directory)-1]
	condition = "%s-%s" %(simulation.split('-')[1], simulation.split('-')[2])
	print(("analyzing simulation %s file %s" %(simulation, dcd_file)))
	top_file = top

	top = md.load_frame(traj, 0, top=top_file).topology
	atom_indices = [a.index for a in top.atoms if a.residue.is_protein and a.residue.resSeq != 341 and a.residue.name[0:2] != "HI" and a.residue.resSeq != 79 and a.residue.resSeq != 296 and a.residue.resSeq != 269 and a.residue.resSeq != 178 and a.residue.resSeq != 93 and a.residue.name != "NMA" and a.residue.name != "NME" and a.residue.name != "ACE"]

	traj = md.load(traj, stride=stride, top=top_file, atom_indices=atom_indices)

	new_file = "%s_stride%d.h5" %(dcd_file.rsplit( ".", 1 )[ 0 ] , stride)
	new_root_dir = "/scratch/users/enf/b2ar_analysis/subsampled"

	

	new_condition_dir = "%s/%s" %(new_root_dir, condition)

	new_file_full = "%s/%s/%s" %(new_root_dir, condition, new_file)
	print(("saving trajectory as %s" %new_file_full))
	traj.save(new_file_full)


def subsample(directory, stride=5):
	conditions = sorted(os.listdir(directory))
	for condition in conditions: 
		subdirectory = "%s/%s" %(directory, condition)
		traj_dir = condition.split("_")[1]
		agonist_residues = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

		if traj_dir.split('-')[1] not in agonist_residues:
			print("skipping simulation")
			continue
		#if traj_dir.split('-')[2] != "00": continue

		new_root_dir = "/scratch/users/enf/b2ar_analysis/subsampled"
		sim_dir = "%s-%s" %(traj_dir.split('-')[1], traj_dir.split('-')[2])
		new_condition_dir = "%s/%s" %(new_root_dir, sim_dir)
		#print(new_condition_dir)
		if os.path.exists(new_condition_dir):
			print("We have already subsampled this simulation")
			continue
		else:
			os.makedirs(new_condition_dir)

		top_file = "%s/system.pdb" %subdirectory
		'''
		new_top_file = "/scratch/users/enf/b2ar_analysis/topology/%s_reordered.pdb" %sim_dir

		if os.path.exists(new_top_file):
			print("already fixed pdb file")
			top_file = new_top_file
		else:
			pdbfixer = PDB_Order_Fixer(top_file, new_top_file)
			pdbfixer.fix_pdb()
			top_file = new_top_file
			print("fixed and saved pdb")
		'''

		traj_dir = "%s/%s" %(subdirectory, traj_dir)
		traj_files = get_trajectory_files(traj_dir)

		#print("Subsampling %d trajectories in condition %s" %(len(traj_files), condition))

		subsample_parallel = partial(subsample_traj, top = top_file, stride=stride)

		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		#print("dividing %d traj_files into %d chunks" %(len(traj_files), chunksize))
		pool.map(subsample_parallel, traj_files)
		pool.terminate()
		gc.collect()

		subsampled_trajs = get_trajectory_files("/scratch/users/enf/b2ar_analysis/subsampled/%s" %sim_dir)

		subsampled_trajs = md.load(subsampled_trajs)
		combined_traj = subsampled_trajs[0].join(subsampled_trajs[1:len(subsampled_trajs)])
		combined_traj.save("/scratch/users/enf/b2ar_analysis/subsampled_combined/%s.h5" %sim_dir)
		print("trajectory has been subsampled, combined, and saved")


def read_and_featurize_divided(filename, dihedrals=['phi', 'psi', 'chi2'], stride=10):
	#print("reading and featurizing %s" %(filename))

	traj_top = md.load_frame(filename,0).topology
	atom_indices = [a.index for a in traj_top.atoms if a.residue.name[0:2] != "HI"]

	traj = md.load(filename,atom_indices=atom_indices)
	#print("got traj")
	featurizer = DihedralFeaturizer(types = dihedrals)
	features = featurizer.transform(traj_list = traj)
	#print(np.shape(features))
	#print("finished featurizing")

	directory = filename.split("/")
	condition = directory[len(directory)-2]
	dcd_file = directory[len(directory)-1]
	new_file = "%s_features_stride%d.h5" %(dcd_file.rsplit( ".", 1 )[ 0 ] , stride)
	new_root_dir = "/scratch/users/enf/b2ar_analysis/subsampled_features"
	new_condition_dir = "%s/%s" %(new_root_dir, condition)

	new_file_full = "%s/%s/%s" %(new_root_dir, condition, new_file)
	#print("saving features as %s" %new_file_full)

	verbosedump(features, new_file_full)
	return features

def featurize_divided(directory):
	simulations = os.listdir(directory)
	for simulation in simulations:
		if simulation[0] not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
			continue

		sim_dir = "%s/%s" %(directory, simulation)
		
		new_dir = "%s/%s" %("/scratch/users/enf/b2ar_analysis/subsampled_features", simulation)
		if os.path.exists(new_dir):
			print("we have already featurized this simulation")
			continue
		else:
			os.makedirs(new_dir)

		print(("currently analyzing %s " %sim_dir))
		trajs = get_trajectory_files(sim_dir)[0:3]
		print(trajs)

		#print("there are %d cpus" %(mp.cpu_count()))
		pool = mp.Pool(mp.cpu_count())
		features_i = pool.map(read_and_featurize, trajs)
		#print(features_i)
		features = [np.concatenate(np.concatenate(features_i))]
		print((np.shape(features[0])))
		combined_dir = "/scratch/users/enf/b2ar_analysis/combined_features"
		new_file_name = "%s_combined.h5" %(simulation)
		new_file = "%s/%s" %(combined_dir, new_file_name)
		#print("saving concatenated features for %s as %s" %(simulation, new_file))
		verbosedump(features, new_file)

def fix_topology(topology):
	'''pseudocode: 
		for each chain:
			create a dictionary that will map residue strings to a list of residue objects
			for each key in dictionary:
				if length of value is greater than one:
					add all atoms to res > 1 to res = 1 and delete atoms from res > 1

	'''

	'''
	new_top = md.Topology()
	resi_dict = {}
	new_chain = new_top.add_chain()
	for resname in residues.keys():
			new_residue = new_top.add_residue(residues[resname][0].name, new_chain, residues[resname][0].resSeq)
			resi_dict[resname] = new_residue
	for i in range(0,len(list(topology.atoms))):
		atom = topology.atom(i)
		res_obj = resi_dict[str(atom.residue)]
		new_atom = new_top.add_atom(atom.name, atom.element, res_obj)
		new_atom.index = atom.index
		new_atom.serial = atom.index
	#for bond in topology.bonds:
	#	new_top.add_bond(new_top.atom(bond[0].index), new_top.atom(bond[1].index))
	
	new_top = new_top.subset([a.index for a in topology.atoms])

	print "fixed topology"

	print [(a.index, a.name) for a in topology.atoms if a.residue.resSeq == 342]
	print [(a.index, a.name) for a in new_top.atoms if a.residue.resSeq == 342]
	'''

	new_top = topology.copy()

	residues = {}
	for chain in new_top.chains:
		print(chain)
		for residue in chain.residues:
			resname = str(residue)
			if resname in list(residues.keys()):
				residues[resname].append(residue)
			else:
				residues[resname] = [residue]

	for resname in list(residues.keys()):
		fragments = residues[resname]
		if len(fragments) > 1:
			main_fragment = fragments[0]
			new_atom_list = []
			new_atom_list += main_fragment._atoms
			for i in range(1,len(fragments)):
				fragment = fragments[i]
				for atom in fragment.atoms:
					atom.residue = main_fragment
				new_atom_list += fragment._atoms
				fragment._atoms = []
				fragment.chain = main_fragment.chain
			main_fragment._atoms = new_atom_list

	print(topology)
	print([(a.index, a.name, a.residue.chain.index) for a in new_top.atoms if a.residue.index == 71])

	
	print(new_top)
	print([(a.index, a.name, a.residue.chain.index) for a in topology.atoms if a.residue.index == 71])
	return new_top


def read_and_featurize(filename, dihedrals=['phi', 'psi', 'chi2'], stride=10):
	print(("reading and featurizing %s" %(filename)))

	traj = md.load(filename)
	#test_traj_init = md.load_frame(filename,5)
	#test_traj_init.save_pdb("/scratch/users/enf/b2ar_analysis/test_init.pdb")

	#traj.topology = fix_topology(traj.topology)

	#traj[-1].save_pdb("/scratch/users/enf/b2ar_analysis/test_fixed.pdb")
	#traj.save_dcd("/scratch/users/enf/b2ar_analysis/test_fixed.dcd")

	#print("got traj")
	featurizer = DihedralFeaturizer(types = dihedrals)
	features = featurizer.transform(traj_list = traj)
	#print("finished featurizing")

	directory = filename.split("/")
	traj_file = directory[len(directory)-1]
	condition = traj_file.split("_")[0].split(".")[0]

	print(("Condition %s has features of shape %s" %(condition, np.shape(features))))

	new_file = "/scratch/users/enf/b2ar_analysis/combined_features/%s_features.h5" %condition
	verbosedump(features, new_file)


def featurize(directory):
	agonist_bound = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
	all_trajs = get_trajectory_files(directory)
	trajs = []
	for fulltraj in all_trajs:
		traj = fulltraj.split("/")
		filename = traj[len(traj)-1]
		if filename[0] in agonist_bound:
			trajs.append(fulltraj)
	#read_and_featurize(trajs[0])
	pool = mp.Pool(mp.cpu_count())
	pool.map(read_and_featurize, trajs)
	pool.terminate()
	print("Completed featurizing")

def load_features(filename):
	return np.concatenate(verboseload(filename))

def fit_and_transform(directory, stride=5):
	
	projected_data_filename = "/scratch/users/enf/b2ar_analysis/phi_psi_chi_stride%d_projected.h5" %stride
	fit_model_filename  = "/scratch/users/enf/b2ar_analysis/phi_psi_chi2_stride%s_tica_coords.h5" %stride
	#active_pdb_file = "/scratch/users/enf/b2ar_analysis/3P0G_pymol_prepped.pdb"
	active_pdb_file = "/scratch/users/enf/b2ar_analysis/system_B.pdb"

	tica_model = tICA(n_components=4)

	if not os.path.exists(projected_data_filename):
		print("loading feature files")
		feature_files = get_trajectory_files(directory)
		pool = mp.Pool(mp.cpu_count())
		features = pool.map(load_features, feature_files)
		pool.terminate()
		if not os.path.exists(fit_model_filename):
			print("fitting data to tICA model")
			fit_model = tica_model.fit(features)
			verbosedump(fit_model, fit_model_filename)
			transformed_data = fit_model.transform(features)
			verbosedump(transformed_data, projected_data_filename)
		else:
			print("loading tICA model")
			fit_model = verboseload(fit_model_filename)
			transformed_data = fit_model.transform(features)
			verbosedump(transformed_data, projected_data_filename)
	else:
		fit_model = verboseload(fit_model_filename)
		transformed_data = verboseload(projected_data_filename)

	active_pdb = md.load(active_pdb_file)
	top = active_pdb.topology
	atom_indices = [a.index for a in top.atoms if a.residue.is_protein and a.residue.resSeq != 341 and a.residue.name[0:2] != "HI" and a.residue.resSeq != 79 and a.residue.resSeq != 296 and a.residue.resSeq != 269 and a.residue.resSeq != 178 and a.residue.resSeq != 93 and a.residue.name != "NMA" and a.residue.name != "NME" and a.residue.name != "ACE"]
	active_pdb = md.load(active_pdb_file, atom_indices=atom_indices)
	featurizer = DihedralFeaturizer(types=['phi', 'psi', 'chi2'])
	active_pdb_features = featurizer.transform(active_pdb)
	active_pdb_projected = fit_model.transform(active_pdb_features)
	print((active_pdb_projected[0:4]))
	
def cluster(data_dir, traj_dir, n_clusters):
	reduced_data = verboseload(data_dir)
	trajs = np.concatenate(reduced_data)
	plt.hexbin(trajs[:,0], trajs[:,1], bins='log', mincnt=1)

	clusterer = MiniBatchKMedoids(n_clusters = n_clusters)
	clusterer.fit_transform(reduced_data)
	
	centers = clusterer.cluster_centers_
	for i in range(0, np.shape(centers)[0]):
		center = centers[i,:]
		plt.scatter(center[0],center[1])
		plt.annotate('C%d' %i, xy=(center[0],center[1]),xytext=(center[0]+0.1,center[1]+0.1), arrowprops=dict(facecolor='black',shrink=0.05))

		location = clusterer.cluster_ids_[i,:]
		print(location)
		traj = get_trajectory_files(traj_dir)[location[0]]
		print(("traj = %s" %traj))
		print(("frame = %d" %location[1]))
		conformation = md.load_frame(traj, location[1])
		conformation.save_pdb("/scratch/users/enf/b2ar_analysis/cluster_%d.pdb" %i)


	plt.show()



#subsample("/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs", stride=5)
#featurize("/scratch/users/enf/b2ar_analysis/subsampled_combined")
#fit_and_transform("/scratch/users/enf/b2ar_analysis/combined_features")
cluster("/scratch/users/enf/b2ar_analysis/phi_psi_chi_stride5_projected.h5", "/scratch/users/enf/b2ar_analysis/subsampled_combined", n_clusters=100)

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

if not (os.path.isfile("/scratch/users/enf/b2ar_analysis/H-05/%s" %("combined_traj_stride10.h5"))):
	print("traj not loaded yet")
	for traj in os.listdir(traj_dir):
		if traj.endswith(".dcd"):
			traj_files.append("%s/%s" %(traj_dir,traj))
	traj_files.sort()
	traj = md.load(traj_files, top = "/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-H-05-all/system.pdb", stride=10)
	traj = traj[0].join(traj[1:])
	traj.save("/scratch/users/enf/b2ar_analysis/H-05/%s" %("combined_traj_stride10.h5"))
else:
	print("loading h5 traj")
	traj = md.load("/scratch/users/enf/b2ar_analysis/H-05/%s" %("combined_traj_stride10.h5"))

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


