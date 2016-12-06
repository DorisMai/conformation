from PDB_Order_Fixer import PDB_Order_Fixer
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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from msmbuilder.cluster import KMedoids
import datetime
import multiprocessing as mp
import glob
import copy
import gc
from functools import partial 
import itertools
import operator
from mdtraj.geometry import dihedral as ManualDihedral
import time
import fileinput
from msmbuilder.cluster import MiniBatchKMedoids
from matplotlib.backends.backend_pdf import PdfPages
from msmbuilder.msm import implied_timescales
import networkx as nx
import pytraj.io as mdio
from pytraj import adict
import random 


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

def get_trajectory_files(traj_dir, ext = ".h5"):
	traj_files = []
	for traj in os.listdir(traj_dir):
			if traj.endswith(ext):
				traj_files.append("%s/%s" %(traj_dir,traj))
	return sorted(traj_files)

def get_traj_no_palm(traj_dir):
	trajs = get_trajectory_files(traj_dir)
	non_palm = []

	for i in range(0, len(trajs)):
		traj = trajs[i]
		traj_name = traj.split("/")[len(traj.split("/"))-1]
		if traj_name[0] not in ['C', 'F', 'H', 'J']:
			non_palm.append(i)

	return non_palm


def convert_mae_to_pdb(traj_dir):
	pdbs = get_trajectory_files(traj_dir)

	for pdb in pdbs:
		mae = pdb.rsplit(".", 1)[0]
		mae = "%s.mae" %mae
		bashCommand = "$SCHRODINGER/utilities/structconvert -ipdb %s -omae %s" %(pdb, mae)
		os.system(bashCommand)

def remove_ter(pdb_dir):
	pdb_files = get_trajectory_files(pdb_dir)

	for pdb_file in pdb_files:
		pdb = file(pdb_file, "rb")
		lines = pdb.readlines()
		new_pdb = file(pdb_file, "wb")
		for line in lines: 
			if "TER" in line:
				continue
			else:
				new_pdb.write(line)
		new_pdb.close()

def reorder(pdb_dir):
	pdb_files = get_trajectory_files(pdb_dir, ext = ".pdb")

	for pdb_file in pdb_files:
		fixer = PDB_Order_Fixer(pdb_file, pdb_file)
		fixer.fix_pdb()

def get_condition(filename):
	print(filename)
	filename = filename.split('/')[len(filename.split('/'))-1]
	pieces = filename.split('-')
	condition = "%s-%s" %(pieces[0], pieces[1])
	return condition 

def read_trajectory(directory, filename, stride=10):
	print(("reading %s" %(filename)))
	traj = md.load(filename, stride=stride, top="/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-H-05-all/system.pdb")
	return traj

def alias_pdb(filename, condition):
	aliases = [("ASH", "ASP"), ("GLH", "GLU"), ("HIP", "HIS"), ("HIE", "HIS"), ("HID", "HIS"), ("HSE", "HIS"), ("HSD", "HIS"), ("CYP", "CYS")]
	new_pdb = "/scratch/users/enf/b2ar_analysis/renamed_topologies/%s.pdb" %condition
	if(os.path.exists(new_pdb)):
		top_file = new_pdb
	else:
		old_file = open(filename, "rb")
		new_file = open(new_pdb, "wb")
		lines = old_file.readlines()
		for line in lines:
			new_line = copy.deepcopy(line)
			for alias in aliases:
				new_line = new_line.replace(alias[0], alias[1])
			new_file.write(new_line)
		new_file.close()
	return new_pdb

def subsample_traj(traj, stride=5, top=None):
	directory = traj.split("/")
	simulation = directory[len(directory)-2]
	dcd_file = directory[len(directory)-1]
	condition = "%s-%s" %(simulation.split('-')[1], simulation.split('-')[2])
	print(("analyzing simulation %s file %s" %(simulation, dcd_file)))
	top_file = top

	top = md.load_frame(traj, 0, top=top_file).topology
	atom_indices = [a.index for a in top.atoms if str(a.residue)[0:3] != "POP" and not a.residue.is_water and str(a.residue)[0:2] != "NA" and str(a.residue)[0:2] != "CL"]

	traj = md.load(traj, stride=stride, top=top_file, atom_indices=atom_indices)
	print("traj loaded")

	new_file = "%s_stride%d.h5" %(dcd_file.rsplit( ".", 1 )[ 0 ] , stride)
	new_root_dir = "/scratch/users/enf/b2ar_analysis/subsampled_allprot"

	

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
		#	if traj_dir.split('-')[2] != "00": continue

		new_root_dir = "/scratch/users/enf/b2ar_analysis/subsampled_allprot"
		sim_dir = "%s-%s" %(traj_dir.split('-')[1], traj_dir.split('-')[2])
		new_condition_dir = "%s/%s" %(new_root_dir, sim_dir)
		if os.path.exists(new_condition_dir):
			print("We have already subsampled_allprot this simulation")
			continue
		else:
			os.makedirs(new_condition_dir)

		top_file = "%s/system.pdb" %subdirectory
		top_file = alias_pdb(top_file,sim_dir)
		

		traj_dir = "%s/%s" %(subdirectory, traj_dir)
		traj_files = get_trajectory_files(traj_dir)

		subsample_parallel = partial(subsample_traj, top = top_file, stride=stride)

		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		pool.map(subsample_parallel, traj_files)
		pool.terminate()
		gc.collect()
		#subsample_parallel(traj_files[0])

		subsampled_allprot_trajs = get_trajectory_files("/scratch/users/enf/b2ar_analysis/subsampled_allprot/%s" %sim_dir)

		combined_traj = md.load(subsampled_allprot_trajs)
		combined_traj.save("/scratch/users/enf/b2ar_analysis/subsampled_allprot_combined/%s.h5" %sim_dir)
		first_frame = md.load_frame("/scratch/users/enf/b2ar_analysis/subsampled_allprot_combined/%s.h5" %sim_dir, index=1)
		first_frame.save_pdb("/scratch/users/enf/b2ar_analysis/%s_firstframe.pdb" %condition)
		print("trajectory has been subsampled_allprot, combined, and saved")


def fix_topology(topology):
	
	new_top = topology.copy()

	residues = {}
	for chain in new_top.chains:
		#print chain
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

	return new_top

def phi_indices(top, residues = None):
	graph = top.to_bondgraph()

	if residues is None:
		c_atoms = [(a, a.residue.resSeq) for a in top.atoms if a.name == "C"]
	else:
		for i in len(residues):
			residues[i] -= 1
		c_atoms = [(a, a.residue.resSeq) for a in top.atoms if a.name == "C" and a.residue.resSeq in residues]

	c_atoms.sort(key=operator.itemgetter(1))
	c_atoms = [c_atom[0] for c_atom in c_atoms]
	#print("%d C atoms" %len(c_atoms))

	phi_tuples = []

	for c in c_atoms:
		n = None
		ca = None
		next_c = None

		c_index = c.index
		c_neighbors = list(graph.edge[c].keys())
		for c_neighbor in c_neighbors:
			if c_neighbor.name == "N":
				n = c_neighbor
				break
		
		if n != None:
			n_neighbors = list(graph.edge[n].keys())
			for n_neighbor in n_neighbors:
				if n_neighbor.name == "CA":
						ca = n_neighbor
						break
		if ca != None:
			ca_neighbors = list(graph.edge[ca].keys())
			for ca_neighbor in ca_neighbors:
				if ca_neighbor.name == "C":
					next_c = ca_neighbor
					break
		if n != None and ca != None and next_c != None:
			phi_tuples.append((c.index, n.index, ca.index, next_c.index))

	#print("phi angles = %d" %len(phi_tuples))
	return phi_tuples




def psi_indices(top, residues = None):

	graph = top.to_bondgraph()
	if residues is None:
		n_atoms = [(a, a.residue.resSeq) for a in top.atoms if a.name == "N"]
	else:
		n_atoms = [(a, a.residue.resSeq) for a in top.atoms if a.name == "N" and a.residue.resSeq in residues]

	n_atoms.sort(key=operator.itemgetter(1))
	n_atoms = [n_atom[0] for n_atom in n_atoms]

	psi_tuples = []

	for n in n_atoms:
		c = None
		ca = None
		next_n = None

		n_index = n.index
		n_neighbors = list(graph.edge[n].keys())
		for n_neighbor in n_neighbors:
			if n_neighbor.name == "CA":
				ca = n_neighbor
				break
		
		if ca != None:	
			ca_neighbors = list(graph.edge[ca].keys())
			for ca_neighbor in ca_neighbors:
				if ca_neighbor.name == "C":
					c = ca_neighbor
					break

		if c != None:
			c_neighbors = list(graph.edge[c].keys())
			for c_neighbor in c_neighbors:
				if c_neighbor.name == "N":
					next_n = c_neighbor
					break

		if c != None and ca != None and next_n != None:
			psi_tuples.append((n.index, c.index, ca.index, next_n.index))

	#print("psi angles = %d " %len(psi_tuples))
	return psi_tuples

def phi_indices_resSeq(top):
	'''
	for i in residues
		residue_i = residues[i]
		residue_ip1 = residues[i+1]
		if residue_i.resSeq == residue_ip1.resSeq - 1:
			N = bla
			C = bla
			CA = 
			N_next 
	'''
	return


def chi2_indices(top, specified_residues = None):
	seq1 = ('CA', 'CB', 'CG', 'CD')
	seq2 = ('CA', 'CB', 'CG', 'OD1')
	seq3 = ('CA', 'CB', 'CG', 'ND1')
	seq4 = ('CA', 'CB', 'CG1', 'CD1')
	seq5 = ('CA', 'CB', 'CG,' 'SD')
	term_4 = ('CD', 'OD1', 'ND1', 'CD1', 'SD')

	top = fix_topology(top)
	if residues is None:
		residues = [(res, res.resSeq) for res in top.residues]
	else:
		residues = [(res, res.resSeq) for res in top.residues if res.resSeq in specified_residues]

	residues.sort(key=operator.itemgetter(1))
	residues = [res[0] for res in residues]
	chi2_tuples = []

	for residue in residues:
		dihedral = [None, None, None, None]
		for atom in residue.atoms:
			if atom.name == 'CA': dihedral[0] = atom.index
			if atom.name == 'CB': dihedral[1] = atom.index
			if atom.name == 'CG' or atom.name == 'CG1': dihedral[2] = atom.index
			if atom.name in term_4: dihedral[3] = atom.index
		if None not in dihedral:
			dihedral = tuple(dihedral)
			chi2_tuples.append(dihedral)

	return chi2_tuples


def read_and_featurize_custom(traj_file, condition=None, location=None, dihedral_residues = None, distance_residues = None):
	top = md.load_frame(traj_file,index = 0).topology
	#atom_indices = [a.index for a in top.atoms if a.residue.resSeq != 130]
	atom_indices = [a.index for a in top.atoms]
	traj = md.load(traj_file, atom_indices=atom_indices)
	print(traj_file)
	#print traj
	#print("loaded trajectory")

	'''
	a = time.time()
	featurizer = DihedralFeaturizer(types = ['phi', 'psi', 'chi2'])
	features = featurizer.transform(traj)
	b = time.time()
	#print(b-a)
	print("original features has dim")
	print(np.shape(features))
	'''
	a = time.time()

	
	phi_tuples = phi_indices(traj.topology, dihedral_residues)
	psi_tuples = psi_indices(traj.topology, dihedral_residues)
	chi2_tuples = chi2_indices(traj.topology, dihedral_residues)

	#if distance_residues is not None:

	

	#print("new features has dim %d" %(2*len(phi_tuples) + 2*len(psi_tuples) + 2*len(chi2_tuples)))

	#print("feauturizing manually:")

	phi_angles = np.transpose(ManualDihedral.compute_dihedrals(traj=traj,indices=phi_tuples))
	psi_angles = np.transpose(ManualDihedral.compute_dihedrals(traj=traj,indices=psi_tuples))
	chi2_angles = np.transpose(ManualDihedral.compute_dihedrals(traj=traj,indices=chi2_tuples))
	
	manual_features = np.concatenate([np.sin(phi_angles), np.cos(phi_angles), np.sin(psi_angles), np.cos(psi_angles), np.sin(chi2_angles), np.cos(chi2_angles)])
	b = time.time()
	#print(b-a)

	print("new features has shape: ")
	print((np.shape(manual_features)))

	if condition is None:
		condition = get_condition(traj_file)

	if location is None:
		location = "/scratch/users/enf/b2ar_analysis/features_allprot"

	verbosedump(manual_features, "%s/%s.h5" %(location, condition))


def featurize_custom(directory, dihedral_residues = None, dihedral_types = None, distance_residues = None):
	agonist_bound = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	all_trajs = get_trajectory_files(directory)
	trajs = []
	for fulltraj in all_trajs:
		traj = fulltraj.split("/")
		filename = traj[len(traj)-1]
		if filename[0] in agonist_bound:
			condition = get_condition(fulltraj)
			if os.path.exists("/scratch/users/enf/b2ar_analysis/features_allprot/%s.h5" %(condition)):
				print("already featurized")
			else:
				trajs.append(fulltraj)
	
	featurize_partial = partial(read_and_featurize_custom, dihedral_residues = dihedral_residues, distance_residues = distance_residues)
	pool = mp.Pool(mp.cpu_count())
	pool.map(featurize_partial, trajs)
	pool.terminate()
	print("Completed featurizing")


def load_features(filename):
	return np.transpose(verboseload(filename))

def fit_and_transform(features_directory, model_dir, stride=5, lag_time=10, n_components = 5):
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	projected_data_filename = "%s/phi_psi_chi2_allprot_projected.h5" %model_dir
	fit_model_filename  = "%s/phi_psi_chi2_allprot_tica_coords.h5" %model_dir
	#active_pdb_file = "/scratch/users/enf/b2ar_analysis/renamed_topologies/A-00.pdb"

	tica_model = tICA(n_components = n_components, lag_time = lag_time)

	if not os.path.exists(projected_data_filename):
		print("loading feature files")
		feature_files = get_trajectory_files(features_directory, ext = ".h5")
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
			print("transforming")
			transformed_data = fit_model.transform(features)
			verbosedump(transformed_data, projected_data_filename)
	else:
		fit_model = verboseload(fit_model_filename)
		transformed_data = verboseload(projected_data_filename)

	print(fit_model.summarize())

	#active_pdb = md.load(active_pdb_file)
	#top = active_pdb.topology
	#atom_indices = atom_indices = [a.index for a in top.atoms if str(a.residue)[0:3] != "POP" and not a.residue.is_water and str(a.residue)[0:2] != "NA" and str(a.residue)[0:2] != "CL"]
	#active_pdb = md.load(active_pdb_file, atom_indices=atom_indices)
	#read_and_featurize_custom(active_pdb_file, condition = "A-00_custom_features", location = "/scratch/users/enf/b2ar_analysis")
	#active_features = [np.transpose(verboseload("/scratch/users/enf/b2ar_analysis/A-00_custom_features.h5"))]
	#active_pdb_projected = fit_model.transform(active_features)
	#print(active_pdb_projected)


def plot_hex(transformed_data_file):
	transformed_data = verboseload(transformed_data_file)
	trajs = np.concatenate(transformed_data)
	plt.hexbin(trajs[:,0], trajs[:,1], bins='log', mincnt=1)
	plt.show()


def save_pdb(traj_dir, clusterer, i):
	location = clusterer.cluster_ids_[i,:]
	traj = get_trajectory_files(traj_dir)[location[0]]
	print(("traj = %s, frame = %d" %(traj, location[1])))
	conformation = md.load_frame(traj, location[1])
	conformation.save_pdb("/scratch/users/enf/b2ar_analysis/clusters_1000_allprot/%d.pdb" %i)
	return None

def get_cluster_centers(clusterer_dir, traj_dir):
	clusterer = verboseload(clusterer_dir)

	centers = clusterer.cluster_centers_

	save_pdb_partial = partial(save_pdb, traj_dir = traj_dir, clusterer = clusterer)
	indices = list(range(0, np.shape(centers)[0]))
	pool = mp.Pool(mp.cpu_count())
	pool.map(save_pdb_partial, indices)
	pool.terminate()
	return


def plot_tica(transformed_data_dir, lag_time):
	transformed_data = verboseload(transformed_data_dir)
	trajs = np.concatenate(transformed_data)
	plt.hexbin(trajs[:,0], trajs[:,1], bins='log', mincnt=1)
	pp = PdfPages("/scratch/users/enf/b2ar_analysis/tica_phi_psi_chi2_t%d.pdf" %lag_time)
	pp.savefig()
	pp.close()


def plot_tica_and_clusters(tica_dir, transformed_data_dir, clusterer_dir, lag_time, component_i = 0, component_j = 1):
	transformed_data = verboseload(transformed_data_dir)
	clusterer = verboseload(clusterer_dir)

	trajs = np.concatenate(transformed_data)
	plt.hexbin(trajs[:,component_i], trajs[:,component_j], bins='log', mincnt=1)

	centers = clusterer.cluster_centers_
	for i in range(0, np.shape(centers)[0]):
		center = centers[i,:]
		plt.annotate('%d' %i, xy=(center[0],center[1]), xytext=(center[0], center[1]),size=6)

	pp = PdfPages("%s/c%d_c%d_clusters%d.pdf" %(tica_dir, component_i, component_j, np.shape(centers)[0]))
	pp.savefig()
	pp.close()

def cluster(data_dir, traj_dir, n_clusters, lag_time):
	clusterer_dir = "/scratch/users/enf/b2ar_analysis/clusterer_%d_t%d.h5" %(n_clusters, lag_time)
	if (os.path.exists(clusterer_dir)):
		print("Already clustered")
	else:
		reduced_data = verboseload(data_dir)
		trajs = np.concatenate(reduced_data)
		clusterer = MiniBatchKMedoids(n_clusters = n_clusters)
		clusterer.fit_transform(reduced_data)
		verbosedump(clusterer, "/scratch/users/enf/b2ar_analysis/clusterer_%d_t%d.h5" %(n_clusters, lag_time))	

def cluster_kmeans(tica_dir, data_dir, traj_dir, n_clusters, lag_time):
	clusterer_dir = "%s/clusterer_%dclusters.h5" %(tica_dir, n_clusters)
	if (os.path.exists(clusterer_dir)):
		print("Already clustered")
	else:
		print("Clustering by KMeans")
		reduced_data = verboseload(data_dir)
		trajs = np.concatenate(reduced_data)
		clusterer = KMeans(n_clusters = n_clusters, n_jobs=-1)
		clusterer.fit_transform(reduced_data)
		verbosedump(clusterer, clusterer_dir)	

def compare_clusters(cluster_dir):
	return

def plot_timescales(clusterer_dir, n_clusters, lag_time):
	clusterer = verboseload(clusterer_dir)
	sequences = clusterer.labels_
	lag_times = list(np.arange(1,150,5))
	n_timescales = 5

	msm_timescales = implied_timescales(sequences, lag_times, n_timescales=n_timescales, msm=MarkovStateModel(verbose=False))
	print(msm_timescales)

	for i in range(n_timescales):
		plt.plot(lag_times, msm_timescales[:,i])
	plt.semilogy()
	pp = PdfPages("/scratch/users/enf/b2ar_analysis/kmeans_%d_%d_implied_timescales.pdf" %(n_clusters, lag_time))
	pp.savefig()
	pp.close()

def align(traj_dir, target_dir):
	new_dir = "%s_aligned" %(traj_dir)
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)


def reimage(traj_dir):
	new_dir = "%s_reimaged" %(traj_dir)
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)

	files = get_trajectory_files(traj_dir, ext = ".pdb")

	for i in range(0, len(files)):
		print(i)
		pdb = files[i]
		name = pdb.split("/")[len(pdb.split("/"))-1]
		conformation = mdio.load(pdb, pdb)
		f0 = conformation[0]
		adict['autoimage']("", f0, conformation.top)
		new_file = "%s/%s" %(new_dir, name)
		mdio.save(new_file, traj = f0, top = conformation.top)

	return new_dir

def remove_palm(traj_dir):
	pdb_files = get_trajectory_files(traj_dir)

	for pdb_file in pdb_files:
		print(pdb_file)
		top = md.load(pdb_file).topology
		indices = [a.index for a in top.atoms]
		pdb = md.load(pdb_file, atom_indices = indices)
		pdb.save_pdb(pdb_file)

def build_msm(clusterer_dir, lag_time):
	clusterer = verboseload(clusterer_dir)
	n_clusters = np.shape(clusterer.cluster_centers_)[0]
	labels = clusterer.labels_
	msm_modeler = MarkovStateModel(lag_time=lag_time)
	print(("fitting msm to trajectories with %d clusters and lag_time %d" %(n_clusters, lag_time)))
	msm_modeler.fit_transform(labels)
	verbosedump(msm_modeler, "/scratch/users/enf/b2ar_analysis/msm_model_%d_clusters_t%d" %(n_clusters, lag_time))
	print(("fitted msm to trajectories with %d states" %(msm_modeler.n_states_)))
	#np.savetxt("/scratch/users/enf/b2ar_analysis/msm_%d_clusters_t%d_transmat.csv" %(n_clusters, lag_time), msm_modeler.transmat_, delimiter=",")
	#G = nx.from_numpy_matrix(msm_modeler.transmat_)
	#nx.write_edgelist(G, "/scratch/users/enf/b2ar_analysis/msm_%d_clusters_t%d_edgelist" %(n_clusters, lag_time), msm_modeler.transmat_, delimiter=",")
	transmat = msm_modeler.transmat_

	mapping = msm_modeler.mapping_

	edges = open("/scratch/users/enf/b2ar_analysis/msm_%d_clusters_t%d_edgelist.csv" %(n_clusters, lag_time), "wb")
	for i in range(0, msm_modeler.n_states_):
		if i == 0:
			for j in range(0, msm_modeler.n_states_):
				edges.write(";")
				edges.write("%d" %mapping[j])
			edges.write("\n")

		edges.write("%d" %(mapping[i]))
		for j in range(0, msm_modeler.n_states_):
			prob = transmat[i][j]
			edges.write(";")
			if prob > 0.000001:
				edges.write("%f" %prob)
			else:
				edges.write("0")
		edges.write("\n")
	edges.close()

def construct_graph(msm_modeler_dir, n_clusters, tica_lag_time, msm_lag_time, inactive = None, active = None):
	graph = nx.DiGraph()
	msm_modeler = verboseload(msm_modeler_dir)
	mapping = msm_modeler.mapping_
	inv_mapping = {v: k for k, v in list(mapping.items())}
	transmat = msm_modeler.transmat_

	for i in range(0, msm_modeler.n_states_):
		for j in range(0, msm_modeler.n_states_):
			prob = transmat[i][j]
			if prob > 0.000001:
				original_i = inv_mapping[i]
				original_j = inv_mapping[j]
				graph.add_edge(original_i, original_j, prob = float(prob), inverse_prob = 1.0 / float(prob))

	print((graph.number_of_nodes()))

	if inactive is not None:
		rmsd_file = open(inactive,"rb")
		rmsd_lines = rmsd_file.readlines()
		for line in rmsd_lines:
			line = line.split(";")
			original_id = int(line[0])
			if original_id in list(mapping.keys()):
				rmsd = line[1]
				graph.node[original_id]["rmsd_to_inactive"] = float(rmsd)

	if active is not None:
		rmsd_file = open(active,"rb")
		rmsd_lines = rmsd_file.readlines()
		for line in rmsd_lines:
			line = line.split(";")
			original_id = int(line[0])
			if original_id in list(mapping.keys()):
				rmsd = line[1]
				graph.node[original_id]["rmsd_to_active"] = float(rmsd)

	nx.write_graphml(graph, "/scratch/users/enf/b2ar_analysis/msm_%d_clusters_tica_t%d_msm_t%d_graph.graphml" %(n_clusters, tica_lag_time, msm_lag_time))



def rmsd_to_structure(clusters_dir, ref_dir, text):
	pdbs = get_trajectory_files(clusters_dir)

	ref = md.load_frame(ref_dir, index=0)
	rmsds = np.zeros(shape=(len(pdbs),2))

	for i in range(0,len(pdbs)):
		print(i) 
		pdb_file = pdbs[i]
		pdb = md.load_frame(pdb_file, index=0)
		rmsd = md.rmsd(pdb, ref, 0)
		rmsds[i,0] = i
		rmsds[i,1] = rmsd[0]

	rmsd_file = "%s/%s_rmsds.csv" %(clusters_dir, text)
	np.savetxt(rmsd_file, rmsds, delimiter=",")

def rmsd_pymol(pdb_dir, ref_dir, script_dir, rmsd_dir):
	script = open(script_dir, "rb")
	lines = script.readlines()

	new_script = open(script_dir, "wb")

	for line in lines:
		if line[0:7] == "pdb_dir": 
			print ("found pdb line")
			line = "pdb_dir = '%s'\n" %pdb_dir
		elif line[0:7] == "ref_dir": line = "ref_dir = '%s'\n" %ref_dir
		elif line[0:8] == "rmsd_dir": line = "rmsd_dir = '%s'\n" %rmsd_dir
		new_script.write(line)

	new_script.close()
	command = "/scratch/users/enf/pymol/pymol %s" %script_dir
	print(command)
	os.system(command)

def convert_csv_to_map(filename):
	rmsds = open(filename, "rb")
	lines = rmsds.readlines()
	rmsd_map = {}
	for line in lines:
		line = line.split()
		cluster = line[0]
		rmsd = line[1]

		if cluster in list(rmsd_map.keys()):
			rmsd_map[cluster].append(rmsd)
		else:
			rmsd_map[cluster] = [rmsd]
	return rmsd_map

def calc_mean_and_stdev(rmsd_map):
	stat_map = {}
	for key in list(rmsd_map.keys()):
		rmsds = np.array(rmsd_map[key])
		mean = np.mean(rmsds, axis = 0)
		stdev = numpy.std(rmsds, axis = 0)
		stat_map[key] = (mean, stdev)
	return stats_map

def analyze_rmsds(inactive_rmsd_file, active_rmsd_file, analysis_file):
	inactive_rmsd_map = convert_csv_to_map(inactive_rmsd_file)
	inactive_stats_map = calc_mean_and_stdev(inactive_rmsd_map)

	active_rmsd_map = convert_csv_to_map(active_rmsd_file)
	active_stats_map = calc_mean_and_stdev(active_rmsd_map)

	new_file = open(analysis_file, "wb")
	new_file.write("cluster, inactive_rmsd, inactive_stdev, active_rmsd, active_stdev")

	for key in sorted(inactive_rmsd_map.keys()):
		new_file.write("%s, %f, %f, %f, %f" %(inactive_stats_map[key][0], inactive_stats_map[key][1], active_stats_map[key][0], active_stats_map[key][1]))
	new_file.close()
	
	return [inactive_stats_map, active_stats_map]



def reorder_pdbs(pdb_dir):
	new_dir = "%s_reordered" %pdb_dir
	if not os.path.exists(new_dir): os.makedirs(new_dir)
	pdbs = get_trajectory_files(pdb_dir)
	for pdb in pdbs: 
		name = pdb.split("/")[len(pdb.split("/"))-1]
		new_name = "%s/%s" %(new_dir, name)
		pdb_fixer = PDB_Order_Fixer(pdb, new_name)
		pdb_fixer.fix_pdb()

def pprep_prot(pdb):
	pdb_name = pdb.split("/")[len(pdb.split("/"))-1]
	new_pdb = pdb_name.rsplit( ".", 1 )[ 0 ]
	new_pdb = "%s.mae" %(new_pdb)
	ref = "/scratch/users/enf/b2ar_analysis/3P0G_pymol_prepped.pdb"

	command = "$SCHRODINGER/utilities/prepwizard -disulfides -fix -noepik -noimpref -noprotassign -reference_st_file %s -NOLOCAL %s %s" %(ref, pdb_name, new_pdb)
	print(command)
	os.system(command)

def pprep(pdb_dir):
	pdbs = get_trajectory_files(pdb_dir, ext = ".pdb")
	os.chdir(pdb_dir)
	
	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	pool.map(pprep_prot, pdbs)
	pool.terminate()

def generate_grid(mae, grid_center, tica_dir, n_clusters, n_samples):
	mae_name = mae_name.rsplit( ".", 1)[0]
	grid_job = "%s.in" %mae_name

	mae_last_name = mae_name.split("/")[len(mae_name.splot("/"))-1]

	output_dir = "%s/grids_n_clusters%d_n_samples%d" %(tica_dir, n_clusters, n_samples)

	gridfile = open(grid_job, "wb")
	gridfile.write("GRIDFILE   %s.zip" %mae_last_name)
	gridfile.write("OUTPUTDIR   %s" %output_dir)
	gridfile.write("GRID_CENTER   %s\n" %grid_center)
	gridfile.write("INNERBOX   10,\n")
	gridfile.write("OUTERBOX   25.0,\n")
	gridfile.write("RECEP_FILE   %s" %mae_name)
	gridfile.write("RECEP_VSCALE   1.0")
	gridfile.write("WRITEZIP   TRUE")
	gridfile.close()


def generate_grids(mae_dir, grid_center, tica_dir, n_clusters, n_samples):
	maes = get_trajectory_files(mae_dir, ".mae")

	generate = partial(generate_grid, grid_center = grid_center, tica_dir = tica_dir, n_clusters = n_clusters, n_samples = n_samples)

	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	pool.map(generate, maes)
	pool.terminate()



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

def dist_to_means(clusterer_dir, features_dir):
	clusterer = verboseload(clusterer_dir)
	clusters_map = make_clusters_map(clusterer)

	features = verboseload(features_dir)
	feature_distances = {}

	def find_cos(index, k_mean):
		traj = index[0]
		frame = index[1]
		conformation = features[traj][frame]
		a = conformation
		b = k_mean
		return (traj, frame, np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))

	for i in range(0, len(list(clusters_map.keys()))):
		indices = clusters_map[i]
		k_mean = clusterer.cluster_centers_[i]
		print(k_mean)
		find_cos_partial = partial(find_cos, k_mean=k_mean)
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

def sample_clusters(clusterer_dir, features_dir, traj_dir, save_dir, n_samples):
	clusters_map = dist_to_means(clusterer_dir, features_dir)
	if not os.path.exists(save_dir): os.makedirs(save_dir)
	
	#non_palm = get_traj_no_palm(traj_dir)

	trajectories = get_trajectory_files(traj_dir)

	for cluster in range(0, len(list(clusters_map.keys()))):
		for s in range(0, n_samples):
			sample = clusters_map[cluster][s]
			traj_id = sample[0]
			frame = sample[1]
			traj = trajectories[traj_id]

			top = md.load_frame(traj, index=frame).topology
			indices = [a.index for a in top.atoms if str(a.residue)[0:3] != "SOD" and str(a.residue)[0:3] != "CLA" and a.residue.resSeq < 341]

			conformation = md.load_frame(traj, index=frame, atom_indices=indices)
			conformation.save_pdb("%s/cluster%d_sample%d.pdb" %(save_dir, cluster, s))
	
	remove_ter(save_dir)
	reorder(save_dir)
	#remove_palm(save_dir)
	new_dir = reimage(save_dir)
	#pprep(new_dir)

def generate_grids(pdb_dir):
	return
					


def output_msm_graph(msm):
	return



n_clusters = 500
lag_time = 10
msm_lag_time = 20
n_components = 5
n_samples = 20

#clusterer_dir = "/scratch/users/enf/b2ar_analysis/kmeans_clusterer_%d_t%d.h5" %(n_clusters, lag_time)
traj_dir = "/scratch/users/enf/b2ar_analysis/subsampled_allprot_combined"
#projected_features_dir = "/scratch/users/enf/b2ar_analysis/phi_psi_chi2_allprot_stride5_t%d_projected.h5" %lag_time
#features_dir = "/scratch/users/enf/b2ar_analysis/features_allprot"
#model_dir = "/scratch/users/enf/b2ar_analysis/tICA_t%d_n_components%d" %(lag_time, n_components)

#subsample("/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs", stride=5)
#featurize_custom("/scratch/users/enf/b2ar_analysis/subsampled_allprot_combined")
#fit_and_transform("/scratch/users/enf/b2ar_analysis/features_allprot", stride=5, lag_time=100)
#plot_tica("/scratch/users/enf/b2ar_analysis/phi_psi_chi2_allprot_stride5_t%d_projected.h5" %lag_time, lag_time=lag_time)
#plot_hex("/scratch/users/enf/b2ar_analysis/phi_psi_chi2_allprot_stride5_projected.h5")
#cluster("/scratch/users/enf/b2ar_analysis/phi_psi_chi2_allprot_stride5_t%d_projected.h5" %lag_time, "/scratch/users/enf/b2ar_analysis/subsampled_allprot_combined",n_clusters = n_clusters, lag_time = lag_time)
#cluster_kmeans("/scratch/users/enf/b2ar_analysis/phi_psi_chi2_allprot_stride5_t%d_projected.h5" %lag_time, "/scratch/users/enf/b2ar_analysis/subsampled_allprot_combined",n_clusters = n_clusters, lag_time = lag_time)


#plot_tica_and_clusters("/scratch/users/enf/b2ar_analysis/phi_psi_chi2_allprot_stride5_t%d_projected.h5" %lag_time, "/scratch/users/enf/b2ar_analysis/kmeans_clusterer_%d_t%d.h5" %(n_clusters, lag_time), lag_time)

#plot_timescales("/scratch/users/enf/b2ar_analysis/kmeans_clusterer_%d_t%d.h5" %(n_clusters, lag_time), n_clusters, lag_time)

#eaturize_custom("/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-A-00-all/pnas2011b-A-00-all/pnas2011b-A-00-all-000.dcd", "/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-A-00-all/system.pdb")
#featurize_custom("/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-B-00-all/pnas2011b-B-00-all/pnas2011b-B-00-all-000.dcd", "/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-B-00-all/system.pdb")

#extract_centers("/scratch/users/enf/b2ar_analysis/kmeans_clusterer_%d_t%d.h5" %(n_clusters, lag_time), "/scratch/users/enf/b2ar_analysis/subsampled_allprot_combined", lag_time=lag_time)

#build_msm("/scratch/users/enf/b2ar_analysis/kmeans_clusterer_%d_t%d.h5" %(n_clusters, lag_time), msm_lag_time)

#reimage("/scratch/users/enf/b2ar_analysis/%d_clusters_t%d" %(n_clusters, lag_time))

#rmsd_to_structure("/scratch/users/enf/b2ar_analysis/%d_clusters_t%d_reimaged_pymol" %(n_clusters, lag_time), "/scratch/users/enf/b2ar_analysis/3P0G_pymol_prepped.pdb")

#construct_graph("/scratch/users/enf/b2ar_analysis/msm_model_%d_clusters_t%d" %(n_clusters, msm_lag_time), n_clusters = n_clusters, tica_lag_time = lag_time, msm_lag_time = msm_lag_time, inactive = "/scratch/users/enf/b2ar_analysis/%d_clusters_t%d_reimaged_pymol/inactive_rmsds.csv" %(n_clusters, lag_time), active = "/scratch/users/enf/b2ar_analysis/%d_clusters_t%d_reimaged_pymol/active_rmsds.csv" %(n_clusters, lag_time))

#reorder_pdbs("/scratch/users/enf/b2ar_analysis/%d_clusters_t%d_reimaged_pymol" %(n_clusters, lag_time))

#remove_ter("/scratch/users/enf/b2ar_analysis/100clusters_t100/test_remove_ter")
#reorder("/scratch/users/enf/b2ar_analysis/100clusters_t100/test_remove_ter")

#sample_clusters(clusterer_dir, traj_dir, lag_time, n_samples=1)

#get_traj_no_palm(traj_dir)

#dist_to_means(clusterer_dir, projected_features_dir)

tica_dir = "/scratch/users/enf/b2ar_analysis/tICA_t%d_n_components%d" %(lag_time, n_components)
clusterer_dir = "%s/clusterer_%dclusters.h5" %(tica_dir, n_clusters)
features_dir = "/scratch/users/enf/b2ar_analysis/features_allprot"
model_dir = "/scratch/users/enf/b2ar_analysis/tICA_t%d_n_components%d" %(lag_time, n_components)
projected_features_dir = "%s/phi_psi_chi2_allprot_projected.h5" %tica_dir
save_dir = "%s/clusters%d_n_components%d_n_samples%d" %(tica_dir, n_clusters, n_components, n_samples)
reimaged_dir = "%s/clusters%d_n_components%d_n_samples%d_reimaged" %(tica_dir, n_clusters, n_components, n_samples)
active_ref_dir = "/scratch/users/enf/b2ar_analysis/3P0G_pymol_prepped.pdb"
inactive_ref_dir = "/scratch/users/enf/b2ar_analysis/2RH1_prepped.pdb"
script_dir = "/scratch/users/enf/b2ar_analysis/pymol_rmsd.py"
active_rmsd_dir =  "%s/active_rmsds.csv" %reimaged_dir
inactive_rmsd_dir = "%s/inactive_rmsd.csv" %reimaged_dir
analysis_file = "%s/rmsd_analyzed.csv" %reimaged_dir
#fit_and_transform(features_directory = features_dir, model_dir = model_dir, stride=5, lag_time = lag_time, n_components = n_components)
#cluster_kmeans(tica_dir, projected_features_dir, traj_dir, n_clusters, lag_time)
#plot_tica_and_clusters(tica_dir, projected_features_dir, clusterer_dir, lag_time)
#sample_clusters(clusterer_dir, projected_features_dir, traj_dir, save_dir, n_samples)

#generate_grids(mae_dir = "/scratch/users/enf/b2ar_analysis/100_clusters_t100_take2_reimaged", grid_center = "64.042, 17.531, 13.12", tica_dir = "/scratch/users/enf/b2ar_analysis", n_clusters = n_clusters, n_samples = n_samples)

#specific_residues = [130, 131, 208, 211, 219, 268, 272, 286, 288, 316, 322, 323, 326]

#rmsd_pymol(reimaged_dir, inactive_ref_dir, script_dir, inactive_rmsd_dir)
#rmsd_pymol(reimaged_dir, active_ref_dir, script_dir, active_rmsd_dir)
analyze_rmsds(inactive_rmsd_dir, active_rmsd_dir, analysis_file)

'''important residues for GPCRs:
Asp3.49	-- Asp130
Arg3.50	--	Arg131
Phe5.47	--	Phe208
Pro5.50	--	Pro211
Tyr5.58	--	Tyr219
Glu6.30	--	Glu268
Thr6.34	--	Thr272
Trp6.48	--	Trp286
Pro6.50	--	Pro288
Lys7.43	--	Lys316
Asn7.49	--	Asn322
Pro7.50	--	Pro323
Tyr7.53	--	Tyr326	
'''
'''
B2AR binding pocket residues:

Met82
Val86
His93
Cyx106
Trp109
Thr110
Asp113
Val114
Val117
Thr118
Thr164
Cyx191
Asp192
Phe193
Thr195
Tyr199
Ala200
Ser203
Val206
Ser208
Trp286
Phe289
Phe290
Asn293
Lys305
Tyr308
Ile309
Asn312
Tyr316



'''




'''
def extract_centers(clusterer_dir, traj_dir, lag_time):
	clusterer = verboseload(clusterer_dir)
	n_clusters = clusterer.n_clusters
	labels = clusterer.labels_
	sample_conformations = []
	visited_clusters = set()
	all_visited = False

	for i in range(0, len(labels)):
		trajectory = labels[i]
		for j in range(0, len(trajectory)):
			label = trajectory[j]
			if label not in visited_clusters:
				sample_conformations.append((label,i,j))
				visited_clusters.add(label)
				if len(visited_clusters) == n_clusters:
					print("sampled all clusters")
					all_visited = True
					break
		if all_visited == True: break

	trajectories = get_trajectory_files(traj_dir)

	for cluster in sample_conformations:
		cluster_id = cluster[0]
		traj = trajectories[cluster[1]]
		frame = cluster[2]

		conformation = md.load_frame(traj,index=frame)

		save_dir = "/scratch/users/enf/b2ar_analysis/%d_clusters_t%d" %(n_clusters, lag_time)
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		conformation.save_pdb("%s/%d.pdb" %(save_dir, cluster_id))
'''

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


