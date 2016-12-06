import mdtraj as md
import numpy as np
from analysis import *
from msmbuilder.utils import verbosedump, verboseload
import time
from mdtraj.geometry import dihedral as ManualDihedral
import itertools
import sys

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

def fix_traj(traj):
	time0 = time.time()
	new_traj = copy.deepcopy(traj)
	topology = new_traj.topology 

	new_top = fix_topology(topology)
	topology = new_top
	new_traj.topology = new_top 

	new_atom_sequence = [a for a in topology.atoms]
	new_index_sequence = [a.index for a in topology.atoms]
	
	for i in range(0, np.shape(traj.xyz)[0]):
		new_traj.xyz[i] = new_traj.xyz[i][new_index_sequence]

	for i in range(0, len(new_index_sequence)):
		new_atom_sequence[i].index = i

	time1 = time.time()
	print(time1 - time0)
	return new_traj

def phi_indices(top, residues = None):
	residues = copy.deepcopy(residues)
	graph = top.to_bondgraph()

	if residues is None:
		c_atoms = [(a, a.residue.resSeq) for a in top.atoms if a.name == "C"]
	else:
		for i in range(0,len(residues)):
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
		else:
			print("No phi found for %s " %c.name)

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
		else:
			print("No phs found for %s " %c.name)

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

def chi1_indices(top, specified_residues = None):
	term_4 = ('CG', 'CG1', 'OG1', 'SG', 'OG')
	chi1_residues = ["Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"]
	chi1_residues = [a.upper() for a in chi1_residues]

	top = fix_topology(top)
	if specified_residues is None:
		residues = [(res, res.resSeq) for res in top.residues]
	else:
		residues = [(res, res.resSeq) for res in top.residues if res.resSeq in specified_residues]

	residues.sort(key=operator.itemgetter(1))
	residues = [res[0] for res in residues]
	chi1_tuples = []

	#print "CHI1: \n"
	for residue in residues:
		dihedral = [None, None, None, None]
		for atom in residue.atoms:
			if atom.name == 'N': dihedral[0] = atom.index
			if atom.name == 'CA': dihedral[1] = atom.index
			if atom.name == 'CB': dihedral[2] = atom.index
			if atom.name in term_4: dihedral[3] = atom.index
		if None not in dihedral:
			dihedral = tuple(dihedral)
			chi1_tuples.append(dihedral)
			#print residue.resSeq
		elif dihedral != [None, None, None, None] and str(residue.name)[0:3] in chi1_residues:
			print("no chi1 found for %s" %str(residue))	
	return chi1_tuples



def chi2_indices(top, specified_residues = None):
	seq1 = ('CA', 'CB', 'CG', 'CD')
	seq2 = ('CA', 'CB', 'CG', 'OD1')
	seq3 = ('CA', 'CB', 'CG', 'ND1')
	seq4 = ('CA', 'CB', 'CG1', 'CD1')
	seq5 = ('CA', 'CB', 'CG,' 'SD')

	chi2_residues = ["Arg", "Asn", "Asp", "Gln", "Glu", "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Trp", "Tyr"]
	chi2_residues = [a.upper() for a in chi2_residues]

	term_4 = ('CD', 'OD1', 'ND1', 'CD1', 'SD')

	top = fix_topology(top)
	if specified_residues is None:
		residues = [(res, res.resSeq) for res in top.residues]
	else:
		residues = [(res, res.resSeq) for res in top.residues if res.resSeq in specified_residues]

	residues.sort(key=operator.itemgetter(1))
	residues = [res[0] for res in residues]
	chi2_tuples = []

	#print "CHI2: \n"
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
			#print residue.resSeq
		elif dihedral != [None, None, None, None] and str(residue.name)[0:3] in chi2_residues:
			print("no chi2 found for %s" %str(residue))		


	return chi2_tuples


def read_and_featurize_custom(traj_file, features_dir = None, condition=None, dihedral_types = ["phi", "psi", "chi1", "chi2"], dihedral_residues = None, contact_residues = None):
	#if "23" not in traj_file and "24" not in traj_file: return
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
	dihedral_indices = []
	residue_order = []
	if len(dihedral_residues) > 0:
		for dihedral_type in dihedral_types:
			if dihedral_type == "phi": dihedral_indices.append(phi_indices(fix_topology(top), dihedral_residues))
			if dihedral_type == "psi": dihedral_indices.append(psi_indices(fix_topology(top), dihedral_residues))
			if dihedral_type == "chi1": dihedral_indices.append(chi1_indices(fix_topology(top), dihedral_residues))
			if dihedral_type == "chi2": dihedral_indices.append(chi2_indices(fix_topology(top), dihedral_residues))

		#print("new features has dim %d" %(2*len(phi_tuples) + 2*len(psi_tuples) + 2*len(chi2_tuples)))

		#print("feauturizing manually:")
		dihedral_angles = []

		for dihedral_type in dihedral_indices:
			angles = np.transpose(ManualDihedral.compute_dihedrals(traj=traj,indices=dihedral_type))
			dihedral_angles.append(np.sin(angles))
			dihedral_angles.append(np.cos(angles))

		manual_features = np.transpose(np.concatenate(dihedral_angles))

	if len(contact_residues) > 0:
		fixed_traj = fix_traj(traj)
		fixed_top = fixed_traj.topology
		distance_residues = []
		res_objects = [r for r in fixed_top.residues]
		for r in contact_residues:
			for res in res_objects:
				if res.resSeq == r and len(res._atoms) > 5:
					#print res._atoms
					distance_residues.append(res.index)
		if len(contact_residues) != len(distance_residues):
			print("Residues are missing")
			print(len(contact_residues))
			print(len(distance_residues))
			#sys.exit()
			#return None
		
		combinations = itertools.combinations(distance_residues, 2)
		pairs = [c for c in combinations]
		#print pairs
		contact_features = md.compute_contacts(traj, contacts = pairs, scheme = 'closest-heavy', ignore_nonprotein=False)[0]
		#print contact_features
		#print(np.shape(contact_features))
		if len(dihedral_residues) > 0: 
			manual_features = np.column_stack((manual_features, contact_features))
		else:
			manual_features = contact_features


	b = time.time()

	print(("new features %s has shape: " %traj_file))
	print((np.shape(manual_features)))

	if condition is None:
		condition = get_condition(traj_file)

	verbosedump(manual_features, "%s/%s.h5" %(features_dir, condition))


def read_and_featurize_iter(traj_file, features_dir = None, condition=None, dihedral_types = ["phi", "psi", "chi1", "chi2"], dihedral_residues = None, contact_residues = None):

	a = time.time()
	dihedral_indices = []
	residue_order = []
	if len(dihedral_residues) > 0:
		for dihedral_type in dihedral_types:
			if dihedral_type == "phi": dihedral_indices.append(phi_indices(fix_topology(top), dihedral_residues))
			if dihedral_type == "psi": dihedral_indices.append(psi_indices(fix_topology(top), dihedral_residues))
			if dihedral_type == "chi1": dihedral_indices.append(chi1_indices(fix_topology(top), dihedral_residues))
			if dihedral_type == "chi2": dihedral_indices.append(chi2_indices(fix_topology(top), dihedral_residues))

		#print("new features has dim %d" %(2*len(phi_tuples) + 2*len(psi_tuples) + 2*len(chi2_tuples)))

		#print("feauturizing manually:")
		dihedral_angles = []

		for dihedral_type in dihedral_indices:
			angles = np.transpose(ManualDihedral.compute_dihedrals(traj=traj,indices=dihedral_type))
			dihedral_angles.append(np.sin(angles))
			dihedral_angles.append(np.cos(angles))

		manual_features = np.transpose(np.concatenate(dihedral_angles))

	if len(contact_residues) > 0:
		contact_features = []
		for chunk in md.iterload(traj_file, chunk = 10000):
			
			fixed_traj = fix_traj(chunk)
			fixed_top = fixed_traj.topology
			distance_residues = []
			res_objects = [r for r in fixed_top.residues]
			for r in contact_residues:
				for res in res_objects:
					if res.resSeq == r and len(res._atoms) > 5:
						#print res._atoms
						distance_residues.append(res.index)
			if len(contact_residues) != len(distance_residues):
				print("Residues are missing")
				print(len(contact_residues))
				print(len(distance_residues))
				#sys.exit()
				#return None
			
			combinations = itertools.combinations(distance_residues, 2)
			pairs = [c for c in combinations]
			#print pairs
			
			contact_features.append(md.compute_contacts(fixed_traj, contacts = pairs, scheme = 'closest-heavy', ignore_nonprotein=False)[0])
		
		contact_features = np.concatenate(contact_features)

		if len(dihedral_residues) > 0: 
			manual_features = np.column_stack((manual_features, contact_features))
		else:
			manual_features = contact_features


	b = time.time()

	print(("new features %s has shape: " %traj_file))
	print((np.shape(manual_features)))

	if condition is None:
		condition = get_condition(traj_file)

	verbosedump(manual_features, "%s/%s.h5" %(features_dir, condition))

def featurize_custom(traj_dir, features_dir, traj_ext, dihedral_residues, dihedral_types, contact_residues, residues_map):
	if not os.path.exists(features_dir): os.makedirs(features_dir)

	all_trajs = get_trajectory_files(traj_dir, traj_ext)
	trajs = []
	for fulltraj in all_trajs:
		#if "clone0.lh5" not in fulltraj: continue
		traj = fulltraj.split("/")
		filename = traj[len(traj)-1]
		#if agonist_bound is not False and filename[0] not in agonist_bound: continue
		filename_noext = filename.split(".")[0]
		if os.path.exists("%s/%s.h5.h5" %(features_dir, filename_noext)):
			print("already featurized")	
		else:
			trajs.append(fulltraj)

	pool = mp.Pool(mp.cpu_count()/4)

	if residues_map is not None:
		dihedral_residues = map_residues(residues_map, dihedral_residues)
		contact_residues = map_residues(residues_map, contact_residues)

	print(contact_residues)	

	featurize_partial = partial(read_and_featurize_iter, features_dir = features_dir, dihedral_residues = dihedral_residues, dihedral_types = dihedral_types, contact_residues = contact_residues)
	#pool.map(featurize_partial, trajs)
	#pool.terminate()
	for traj in trajs:
		featurize_partial(traj)

	print("Completed featurizing")


def featurize_known_traj(traj_dir, inactive, features_dir):
	print(("currently featurizing %s" %traj_dir.split("/")[len(traj_dir.split("/"))-1]))
	traj = md.load(traj_dir)
	rmsds = rmsd_npxxy(traj, inactive)
	helix6_helix3_distances = helix6_helix3_dist(traj)
	features = np.transpose(np.concatenate([[rmsds], [np.concatenate(helix6_helix3_distances)]]))
	print(np.shape(features))

	filename = "%s/%s" %(features_dir, traj_dir.split("/")[len(traj_dir.split("/"))-1])
	verbosedump(features, filename)

def featurize_known(directory, inactive_dir, active_dir):
	features_dir = "/scratch/users/enf/b2ar_analysis/features_known"
	if not os.path.exists(features_dir): os.makedirs(features_dir)

	ianctive = md.load(inactive_dir)

	agonist_bound = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	all_trajs = get_trajectory_files(directory)
	trajs = []
	for fulltraj in all_trajs:
		traj = fulltraj.split("/")
		filename = traj[len(traj)-1]
		if filename[0] in agonist_bound:
			condition = get_condition(fulltraj)
			if os.path.exists("%s/%s.h5" %(features_dir, condition)):
				print("already featurized")
				trajs.append(fulltraj)
			else:
				trajs.append(fulltraj)
	
	featurize_partial = partial(featurize_known_traj, inactive_dir = inactive_dir, features_dir = features_dir)
	#pool = mp.Pool(mp.cpu_count()-1)
	#pool.map(featurize_partial, trajs)
	#pool.terminate()
	featurize_partial(trajs[0])

	print("Completed featurizing")

def compute_pnas_coords_and_distance(traj_file, inactive, active, scale = 7.14, residues_map = None):
	print("featurizing %s" %traj_file)
	traj = md.load(traj_file)
	inactive_tuple = np.array([helix6_helix3_dist(inactive) / scale, rmsd_npxxy(inactive, inactive)])
	active_tuple = np.array([helix6_helix3_dist(active) / scale, rmsd_npxxy(active, inactive)])
	traj_coords = [helix6_helix3_dist(traj, residues_map) / scale, rmsd_npxxy(traj, inactive, residues_map), rmsd_npxxy(traj, active, residues_map), rmsd_connector(traj, inactive, residues_map), rmsd_connector(traj, active, residues_map)]
	traj_coords = np.transpose(np.vstack(traj_coords))
	active_vectors = traj_coords[:,[0,1]] - np.transpose(active_tuple)
	inactive_vectors = traj_coords[:,[0,1]] - np.transpose(inactive_tuple)

	inactive_distances = np.linalg.norm(inactive_vectors, axis = 1)
	active_distances = np.linalg.norm(active_vectors, axis = 1)
	distances = [inactive_distances, active_distances]
	#print distances[1]
	return [traj_coords, distances]

def convert_np_to_map(data):
	data_map = {}
	for i in range(0, len(data)):
		traj_data = data[i]
		for j in range(0, np.shape(traj_data)[0]):
			try:
				data_map["traj%d_frame%d" %(i,j)] = traj_data[j,:]
			except:
				data_map["traj%d_frame%d" %(i,j)] = [traj_data[j]]
	return data_map

def featurize_pnas_distance_traj(traj_dir, ianctive, active, features_dir):
	#pnas_distances = 
	return

def featurize_pnas_distance(traj_dir, features_dir, ext, inactive_dir, active_dir, inactive_distances_dir, active_distances_dir, coords_dir, inactive_distances_csv, active_distances_csv, coords_csv, scale = 7.14, residues_map = None):
	if not os.path.exists(features_dir): os.makedirs(features_dir)

	inactive = md.load(inactive_dir)
	active = md.load(active_dir)

	agonist_bound = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	trajs = get_trajectory_files(traj_dir, ext = ext)
	#trajs = [t for t in trajs if "clone0.lh5" in t]
	#traj_objs = md.load(trajs)
	featurize_partial = partial(compute_pnas_coords_and_distance, inactive = inactive, active = active, scale = scale, residues_map = residues_map)
	pool = mp.Pool(16)
	features = pool.map(featurize_partial, trajs)
	#for traj in trajs:
	#	featurize_partial(traj)
	pool.terminate()
	

	coords = [f[0] for f in features]
	inactive_distances = [f[1][0] for f in features]
	active_distances = [f[1][1] for f in features]

	verbosedump(coords, coords_dir)
	verbosedump(inactive_distances, inactive_distances_dir)
	verbosedump(active_distances, active_distances_dir)

	write_map_to_csv(coords_csv, convert_np_to_map(coords), ["frame", "tm3_tm6_dist", "rmsd_npxxy_inactive", "rmsd_npxxy_active", "rmsd_connector_inactive", "rmsd_connector_active"])
	write_map_to_csv(active_distances_csv, convert_np_to_map(active_distances), ["frame", "pnas_distance_active"])
	print("Completed featurizing")
def load_pdb_traj(pdb_file):
	print(pdb_file)
	return md.load_frame(pdb_file, index = 0)

def featurize_pnas_distance_pdbs(traj_dir, new_filename, features_dir, inactive_dir, active_dir, inactive_distances_dir, active_distances_dir, coords_dir):
	#if not os.path.exists(features_dir): os.makedirs(features_dir)

	inactive = md.load(inactive_dir)
	active = md.load(active_dir)

	agonist_bound = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	samples = get_trajectory_files(traj_dir, ext = ".pdb")
	pool = mp.Pool(mp.cpu_count())
	trajs = pool.map(load_pdb_traj, samples)
	trajs_joined = trajs[0].join(trajs[1:])

	trajs_joined.save_hdf5(new_filename)

	features = compute_pnas_coords_and_distance(new_filename, inactive, active)

	coords = [f[0] for f in features]
	inactive_distances = [f[1][0] for f in features]
	active_distances = [f[1][1] for f in features]

	verbosedump(coords, coords_dir)
	verbosedump(inactive_distances, inactive_distances_dir)
	verbosedump(active_distances, active_distances_dir)

	print("Completed featurizing")
	

def load_features(filename):
	return np.transpose(verboseload(filename))