import fileinput
import os
import copy
import csv
from msmbuilder.utils import verbosedump, verboseload
import numpy as np
import mdtraj as md
import multiprocessing as mp
from msmbuilder.dataset import dataset, _keynat, NumpyDirDataset
from functools import partial
import scipy.io as sio
import pickle
import sys
import subprocess

def compat_verboseload(filename):
	with open(filename, 'rb') as f:
	    d = pickle.load(f, encoding='latin1') 
	return d

def get_base():
	sherlock_base = "/scratch/users/enf/b2ar_analysis"
	biox3_base = "/home/enf/b2ar_analysis"

	if os.path.exists(sherlock_base):
		print("we are operating on sherlock")
		base = sherlock_base
	elif os.path.exists(biox3_base):
		print("we are operating on biox3")
		base = biox3_base
	else:
		print("WHERE ARE WE?")
		base = ""
	return(base)

def save_dataset(data, path): 
	if os.path.exists(path):
		cmd = "rm -rf %s" %path
		subprocess.call(cmd, shell=True)
	ds = dataset(path, 'w', 'dir-npy')
	for i in range(0,len(data)):
		ds[i] = data[i]
	ds.close()

def load_dataset(path):
	ds = dataset(path, 'r', 'dir-npy')
	data = np.array(ds[:])
	return(data)

def load_npz(filename):
	nyx = np.load(filename)
	nyx = [nyx[key] for key in list(nyx.keys())][0]
	return(nyx)

def load_file(filename):
	print(("loading %s" %filename))
	if filename.split(".")[1] == "h5":
		return np.nan_to_num(np.transpose(compat_verboseload(filename)))
	elif filename.split(".")[1] == "dataset":
		return np.nan_to_num(np.array(load_dataset(filename)))
	elif filename.split(".")[1] == "csv":
		csv = np.nan_to_num(np.genfromtxt(filename, delimiter=","))
		return(np.nan_to_num(csv))
	elif filename.split(".")[1] == "npy":
		return(np.nan_to_num(np.load(filename)))
	elif filename.split(".")[1] == "npz":
		return(np.nan_to_num(np.array(load_dataset(filename))))
	elif filename.split(".")[1] == "pkl":
		return compat_verboseload(filename)

def load_file_list(files, directory = None, ext = None):
	print(directory)
	print(ext)
	if directory != None and ext != None:
		files = get_trajectory_files(directory, ext)
	print(files)
	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	features = pool.map(load_file, files)
	pool.terminate()
	return(features)

def load_features(filename):
	print(("loading %s" %filename))
	if filename.split(".")[1] == ".h5":
		return np.nan_to_num(np.transpose(compat_verboseload(filename)))
	else:
		return np.nan_to_num(np.transpose(np.array(load_dataset(filename))))

def get_trajectory_files(traj_dir, ext = ".pdb"):
	traj_files = []
	for traj in os.listdir(traj_dir):
			if traj.endswith(ext):
				traj_files.append("%s/%s" %(traj_dir,traj))
	return sorted(traj_files)

def convert_feature_to_ext(feature_file, save_ext):
	print(("Converting %s" %feature_file))
	feature = load_file(feature_file)
	name = feature_file.split(".")[0]
	if save_ext == ".csv":
		np.savetxt("%s.csv" %name, feature, delimiter=",")
	elif save_ext == ".npy":
		np.save("%s.npy" %name, feature)
	elif save_ext == ".mat":
		sio.savemat("%s.mat" %name, mdict={'arr':feature})
	else:
		print("ext not recognized")

#def save_mat(dictionary, filename):
#	sio.savemat(filename, dictionary)
	
def convert_features_to_ext(features_dir, load_ext, save_ext):
	files = get_trajectory_files(features_dir, load_ext)
	convert_partial = partial(convert_feature_to_ext, save_ext = save_ext)
	pool = mp.Pool(mp.cpu_count()/4)
	features = pool.map(convert_partial, files)
	pool.terminate()
	#for feature_file in files:
	#	print("converting %s" %feature_file)
		

def get_trajectory_files_conditions(traj_dir, ext, condition_1, condition_2):
	trajs = get_trajectory_files(traj_dir, ext)
	traj_1 = [t for t in trajs if condition_1 in t]
	traj_2 = [t for t in trajs if condition_2 in t]
	traj_1 = sorted(traj_1)
	traj_2 = sorted(traj_2)
	print((len(traj_1)))
	print((len(traj_2)))
	print((len(traj_1+traj_2)))
	return(traj_1 + traj_2)

def get_ligands(lig_dir, ext= ".sdf"):
	ligands = get_trajectory_files(lig_dir, ext = ext)
	ligs = []
	for ligand in ligands:
		lig_last_name = ligand.split("/")[len(ligand.split("/"))-1]
		lig = lig_last_name.split(".")[0]
		ligs.append(lig)
	return ligs

def write_map_to_csv(filename, data_map, titles):
	csvfile = open(filename, "wb")
	i = 0
	if len(titles) > 0:
		for title in titles:
			if i < (len(titles) - 1):
				csvfile.write("%s, " %title)
			else:
				csvfile.write("%s \n" %title)
			i += 1

	for key in sorted(data_map.keys()):
		csvfile.write("%s, " %key)
		i = 0
		for value in data_map[key]:
			if i < (len(data_map[key])-1):
				csvfile.write("%s, " %str(value))
			else:
				csvfile.write("%s \n" %str(value))
			i += 1
	return

def generateData(files):
	for data in files:
		print(data)
		yield compat_verboseload(data)

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

def convert_csv_to_list(filename):
	with open(filename, "rb") as f:
		reader = csv.reader(f)
		lines = list(reader)
		i = 0
		lines.pop(0)
		for line in lines:
			for i in range(1, len(line)):
				line[i] = float(line[i])
		return lines

def convert_csv_to_map_nocombine(filename):
	print(filename)
	rmsds = open(filename, "rb")
	lines = [line.strip() for line in rmsds.readlines()]
	#if "docking" in filename:
	#	print lines
	rmsd_map = {}
	i = 0
	for line in lines:
		if i == 0:
			i += 1
			continue
		if ';' in line:
			line = line.split(';')
		elif ',' in line:
			line = line.split(',')
		else:
			line = line.split()

		#print line
		cluster = line[0].split('.')[0]
		#print cluster
		#print line
		for i in range(1,len(line)):
			try:
				rmsd = float(line[i].split('\\')[0])
			except:
				continue
			#print rmsd
			#if rmsd > 3.0: print "%s %f" %(cluster, rmsd)

			if cluster in list(rmsd_map.keys()):
				rmsd_map[cluster].append(rmsd)
			else:
				rmsd_map[cluster] = [rmsd]
	return rmsd_map

def get_titles(filename):
	csv_file = open(filename, "rb")
	firstline = csv_file.readlines()[0].split("\n")[0].split(",")
	firstline = [f.strip() for f in firstline]
	return firstline

def convert_csv_to_joined_map(filename, new_filename = False):
	rmsds = open(filename, "rb")
	lines = rmsds.readlines()
	rmsd_map = {}
	i = 0
	for line in lines:
		if i == 0:
			i += 1
			continue
		if ';' in line:
			line = line.split(';')
		elif ',' in line:
			line = line.split(',')
		else:
			line = line.split()



		#print line
		cluster = line[0].split('_')[0]
		rmsd = float(line[1].split('\\')[0])

		if cluster in list(rmsd_map.keys()):
			rmsd_map[cluster].append(rmsd)
		else:
			rmsd_map[cluster] = [rmsd]

	titles = []
	num_entries = len(rmsd_map[list(rmsd_map.keys())[0]])
	titles.append("cluster")
	for i in range(0, num_entries):
		titles.append("sample%d" %i)

	if new_filename is not False:
		write_map_to_csv(new_filename, rmsd_map, titles)

	return [rmsd_map, titles]

def combine_map(map_i, map_j):
	map_j_val_length = len(map_j[list(map_j.keys())[0]])
	placeholder = []
	for i in range(0, map_j_val_length):
		placeholder.append(0.0)

	for key in list(map_i.keys()):
		if key in list(map_j.keys()):
			map_i[key] += map_j[key]
		else:
			#map_i[key] += placeholder
			map_i.pop(key)
	return map_i

def combine_maps(map_list):
	combined_map = copy.deepcopy(map_list[0])
	for i in range(1, len(map_list)):
		combined_map = combine_map(combined_map, map_list[i])
	return combined_map

def combine_csv_list(csv_list, new_csv_filename):
	map_list = []
	combined_map = {}
	combined_map_titles = ["cluster"]

	for csv in csv_list:
		map_list.append(convert_csv_to_map_nocombine(csv))
		csv_file = open(csv, "rb")
		csv_titles = csv_file.readlines()[0].split("\n")[0].split(",")
		csv_titles = csv_titles[1:len(csv_titles)]
		print(csv_titles)
		combined_map_titles += csv_titles

	combined_map = combine_maps(map_list)
	print(combined_map_titles)
	write_map_to_csv(new_csv_filename, combined_map, combined_map_titles)
	return combined_map

def get_condition(filename):
	filename = filename.split('/')[len(filename.split('/'))-1]
	pieces = filename.split('-')
	condition = "%s-%s" %(pieces[0], pieces[1])
	return condition 

def read_trajectory(directory, filename, stride=10):
	print(("reading %s" %(filename)))
	traj = md.load(filename, stride=stride, top="/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-H-05-all/system.pdb")
	return traj

def reverse_sign_csv(csv_file):
	with open(csv_file, 'rb') as f:
		reader = csv.reader(f)
		lines = list(reader)

	new_csv = open(csv_file, "wb")
	line_num = 0
	for line in lines:
		if line_num > 0:
			for i in range(1, len(line)):
				line[i] = str(-1.0 * (float(line[i])))
			print(line)
			new_csv.write(",".join(line))
			new_csv.write(" \n")
		else:
			new_csv.write(",".join(line))
			new_csv.write(" \n")
			line_num += 1
	new_csv.close()

def convert_matrix_to_map(matrix_file, traj_dir, ext, header, csv_file):
	trajs = get_trajectory_files(traj_dir, ext = ext)
	matrix = np.vstack(compat_verboseload(matrix_file))
	values_map = {}
	for i in range(0, np.shape(matrix)[0]):
		traj = trajs[i]
		traj_lastname = traj.split("/")[len(traj.split("/"))-1]
		traj_name = traj_lastname.split(".")[0]
		values = tuple(matrix[i,:])
		values_map[traj_name] = values
		write_map_to_csv(csv_file, values_map, header)
	return values_map


def convert_matrix_list_to_list(np_file, csv_file):
	matrix_list = compat_verboseload(np_file)
	all_values = np.concatenate(matrix_list)
	np.savetxt(csv_file, all_values, delimiter=",")
	return all_values	 

def find_missing_features(traj_dir, features_dir):
	trajs = get_trajectory_files(traj_dir, ".lh5")
	trajs = [t.split("/")[len(t.split("/"))-1].split(".")[0] for t in trajs]
	trajs = set(trajs)
	features = get_trajectory_files(features_dir, ".h5")
	features = [f.split("/")[len(f.split("/"))-1].split(".")[0] for f in features]
	features = set(features)
	print((trajs - features))

def generate_features(features_file):
	if features_file.split(".")[1] == "csv":
		reader = csv.reader(open(features_file, "rb"))
		features = []
		for line in reader:
			try:
				try:
					features.append(((int(line[1]), int(line[2]), str(line[3])), (int(line[4]), int(line[5]), str(line[6]))))
				except:
					features.append(((int(line[1]), int(line[2]))))
			except:
				continue
	elif features_file.split(".")[1] == "pkl":
		print("Loading pickle file of features.")
		print(features_file)
		with open(features_file, "rb") as f:
			features = pickle.load(f)
	else:
		print("Extension is not recognized for features file.")
	return(features)

def generate_residues_map(csv_map):
	reader = csv.reader(open(csv_map, "rb"))
	residues_map = {}
	for line in reader:
		residues_map[int(line[0])] = int(line[1])
	return residues_map

def map_residues(residues_map, residues):
	new_residues = []
	for residue in residues:
		try:
			new_residues.append(residues_map[residue])
		except:
			print(("residue %d not in receptor" %residue))
	return new_residues 

def test_residues_map(traj_file_1, traj_file_2, residues, residues_map):
	traj_1 = md.load_frame(traj_file_1, index = 0)
	traj_2 = md.load_frame(traj_file_2, index = 0)
	top1 = traj_1.topology
	top2 = traj_2.topology
	for residue in residues:
		new_residue = residues_map[residue]
		print("Original residues:")
		residues = [r for r in top1.residues if r.resSeq == residue and r.is_protein]
		print((residues[0]))
		print("New residues:")
		residues = [r for r in top2.residues if r.resSeq == new_residue and r.is_protein]
		print((residues[0]))
	return

def test_residues_map_num_atoms(traj_file_1, traj_file_2, residues, residues_map):
	traj_1 = md.load_frame(traj_file_1, index = 0)
	traj_2 = md.load_frame(traj_file_2, index = 0)
	top1 = traj_1.topology
	top2 = traj_2.topology
	for residue in residues:
		new_residue = residues_map[residue]
		atoms = [a.index for a in top1.atoms if a.residue.resSeq == residue and a.residue.is_protein]
		len1 = len(atoms)
		atoms = [a.index for a in top2.atoms if a.residue.resSeq == new_residue and a.residue.is_protein]
		len2 = len(atoms)
		if (len1 != len2) or (len1 == len2):
			print(("Atom number %d %d doesn't match for residue %d" %(len1, len2, residue)))
	return


def map_residues_universal(residues, save):
	pdb_file = "/home/enf/b2ar_analysis_sherlock_all/exacycle_data/alignment_universal_pdb.txt"
	lh5_file = "/home/enf/b2ar_analysis_sherlock_all/exacycle_data/alignment_universal_lh5.txt"

	pdb_lines = []
	lh5_lines = []

	pdb = open(pdb_file, "rb")
	lh5 = open(lh5_file, "rb")
	for line in pdb.readlines():
		if "TER" not in line and "END" not in line:
			pdb_lines.append(line.split())

	for line in lh5.readlines():
		if "TER" not in line and "END" not in line:
			lh5_lines.append(line.split())

	if len(pdb_lines) != len(lh5_lines):
		print("Alignemnt no goood")
		sys.exit()

	residues_map = {}
	new_residues = []

	for i in range(0,len(pdb_lines)):
		if pdb_lines[i][4] == lh5_lines[i][4]:
			residues_map[int(pdb_lines[i][5])] = int(lh5_lines[i][5])

	for residue in residues:
		new_residues.append(residues_map[residue])

	if 281 in list(residues_map.keys()):
		print(("Residue 281 is now Residue %d" %residues_map[281]))

	print(("residue 272 is now residue %d" %residues_map[272]))

	pdb.close()
	lh5.close()

	if save != False:
		writer = csv.writer(open(save, "wb"))
		for key, value in list(residues_map.items()):
			writer.writerow([key,value])

	return new_residues

def map_residues_condition(residues, condition):
	if condition == "2rh1":
		pdb_file = "/home/enf/b2ar_analysis_sherlock_all/exacycle_data/align_2rh1_pdb.txt"
		lh5_file = "/home/enf/b2ar_analysis_sherlock_all/exacycle_data/align_2rh1_lh5.txt"
	elif condition == "3p0g":
		pdb_file = "/home/enf/b2ar_analysis_sherlock_all/exacycle_data/align_3p0g_pdb.txt"
		lh5_file = "/home/enf/b2ar_analysis_sherlock_all/exacycle_data/align_3p0g_lh5.txt"

	pdb_lines = []
	lh5_lines = []

	pdb = open(pdb_file, "rb")
	lh5 = open(lh5_file, "rb")
	for line in pdb.readlines():
		if "TER" not in line and "END" not in line:
			pdb_lines.append(line.split())

	for line in lh5.readlines():
		if "TER" not in line and "END" not in line:
			lh5_lines.append(line.split())

	if len(pdb_lines) != len(lh5_lines):
		print("Alignemnt no goood")
		sys.exit()

	residues_map = {}
	new_residues = []

	for i in range(0,len(pdb_lines)):
		if pdb_lines[i][4] == lh5_lines[i][4]:
			residues_map[int(pdb_lines[i][5])] = int(lh5_lines[i][5])

	for residue in residues:
		new_residues.append(residues_map[residue])

	if 281 in list(residues_map.keys()):
		print(("Residue 281 is now Residue %d" %residues_map[281]))

	print(("residue 272 is now residue %d" %residues_map[272]))

	pdb.close()
	lh5.close()

	if condition == "2rh1":
		writer = csv.writer(open("/home/enf/b2ar_analysis_sherlock_all/exacycle_data/residues_map_2rh1.csv", "wb"))
		for key, value in list(residues_map.items()):
			writer.writerow([key,value])
	if condition == "3p0g":
		writer = csv.writer(open("/home/enf/b2ar_analysis_sherlock_all/exacycle_data/residues_map_3p0g.csv", "wb"))
		for key, value in list(residues_map.items()):
			writer.writerow([key,value])

	return new_residues

def get_cluster_ids(active_clusters_csv, intermediate_clusters_csv, inactive_clusters_csv):
	with open(active_clusters_csv, 'rb') as f:
	    reader = csv.reader(f)
	    active_clusters = list(reader)[0]
	active_clusters = [int(c[7:]) for c in active_clusters]
	print(active_clusters)
	with open(intermediate_clusters_csv, 'rb') as f:
	    reader = csv.reader(f)
	    intermediate_clusters = list(reader)[0]
	intermediate_clusters = [int(c[7:]) for c in intermediate_clusters]
	print(intermediate_clusters)
	print((intermediate_clusters[0:10]))
	with open(inactive_clusters_csv, 'rb') as f:
	    reader = csv.reader(f)
	    inactive_clusters = list(reader)[0]
	inactive_clusters = [int(c[7:]) for c in inactive_clusters]
	print(inactive_clusters)
	return active_clusters, intermediate_clusters, inactive_clusters

