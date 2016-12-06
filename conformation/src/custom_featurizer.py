__author__ = "Evan Feinberg"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import mdtraj as md
import numpy as np
from analysis import *
from msmbuilder.utils import verbosedump, verboseload
import time 
from pdb_editing import *
from mdtraj.geometry import dihedral as ManualDihedral
import sys
from msmbuilder.featurizer import DihedralFeaturizer
import itertools
from numpy import random
import json
import pickle
from residue import ContactFeature
from residue import AromaticFeature
from scipy.stats import gamma 
from sklearn.metrics.pairwise import pairwise_distances
from residue import *
from aromatic_featurizer import compute_pi_interactions
import subprocess
import random

def swap_chains_in_traj(traj_file, chains, return_traj=False):
    t = md.load(traj_file)
    for idx, chain in enumerate(t.topology.chains):
        if chain.id == chains[0]:
            chain.id = chains[1]
        elif chain.id == chains[1]:
            chain.id = chains[0]
    t.save(traj_file, force_overwrite=True)
    if return_traj:
        return t
    else:
        return


def compute_average_min_distance(traj, residues_i, residues_j):

  atoms_i = []
  atoms_j = []

  for residue in residues_i:
    atoms_i.append([a.index for a in traj.topology.atoms if residue.is_mdtraj_res_equivalent(a.residue) and a.name == "CA"][0])
  for residue in residues_j:
    atoms_j.append([a.index for a in traj.topology.atoms if residue.is_mdtraj_res_equivalent(a.residue) and a.name == "CA"][0])

  traj_i = traj.atom_slice(atoms_i)
  traj_j = traj.atom_slice(atoms_j)

  xyz_i = traj_i.xyz
  xyz_j = traj_j.xyz
  
  distances = np.zeros(traj.n_frames)
  for frame in range(0, np.shape(xyz_i)[0]):	
    distances[frame] = np.mean(np.min(pairwise_distances(
    	X=xyz_i[frame,:], Y=xyz_j[frame,:], metric='euclidean'), axis=0))

  avg_min_distances = distances * 10.

  return avg_min_distances

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
			print("No psis found for %s " %c.residue)

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

	#top = fix_topology(top)
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

	#top = fix_topology(top)
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
		if (None not in dihedral) and (str(residue.name)[0:3] in chi2_residues):
			dihedral = tuple(dihedral)
			chi2_tuples.append(dihedral)
			#print residue
		elif dihedral != [None, None, None, None] and str(residue.name)[0:3] in chi2_residues:
			print("no chi2 found for %s" %str(residue))		


	return chi2_tuples

def find_binding_pocket_residues(ligand_resobj, protein_resobj_list, protein_file, cutoff=0.66):
	protein = md.load(protein_file)
	protein_resids = [convert_residue_to_mdtraj_index(protein.topology, res) for res in protein_resobj_list]
	ligand_resid = convert_residue_to_mdtraj_index(protein.topology, ligand_resobj)
	ligand_protein_pairs = [(ligand_resid, protein_resid) for protein_resid in protein_resids]
	distances = md.compute_contacts(protein, contacts=ligand_protein_pairs, scheme='closest-heavy')[0][0]

	bp_resids = []
	bp_res_objs = []
	for i in range(0, len(distances)):
		distance = distances[i]
		if distance < cutoff:
			bp_resids.append(protein_resids[i])
			bp_res_objs.append(protein_resobj_list[i])

	return bp_resids, bp_res_objs

def compute_atom_residue_pairs_under_cutoff(atom_objects, residue_objects, protein_file, all_lig_atoms=False, cutoff=0.66):
	protein = md.load(protein_file)
	resids = [convert_residue_to_mdtraj_index(protein.topology, res) for res in residue_objects]
	atom_ids = [convert_atom_to_mdtraj_index(protein.topology, atom) for atom in atom_objects]

	atom_residue_pairs = []
	atom_id_resid_pairs = []
	for a, _ in enumerate(atom_ids):
		for r, _ in enumerate(resids):
			atom_residue_pairs.append((atom_objects[a], residue_objects[r]))
			atom_id_resid_pairs.append((atom_ids[a], resids[r]))

	distances, contacts = compute_atom_residue_pair_distances(protein, atom_id_resid_pairs, scheme='closest-heavy')

	res_to_atom_dist = {}
	res_atom_dict = {}
	for res in residue_objects: 
		res_to_atom_dist[res] = 1000.
		res_atom_dict[res] = []

	bp_atom_residue_pairs = []
	for i in range(0, np.shape(distances)[1]):
		res = atom_residue_pairs[i][1]
		distance = distances[0][i]
		if not all_lig_atoms:
			if distance < res_to_atom_dist[res]:
				res_to_atom_dist[res] = distance 
				res_atom_dict[res] = atom_residue_pairs[i]
		else:
			if distance < cutoff:
				bp_atom_residue_pairs.append(atom_residue_pairs[i])

	if not all_lig_atoms:
		bp_atom_residue_pairs = res_atom_dict.values()
	
	return bp_atom_residue_pairs

def apply_gamma_pdf_parameters(contact_features, gamma_pdf_parameters):
	gamma_features = []

	for param in gamma_pdf_parameters:
		param_features = np.apply_along_axis(gamma.pdf, 0, contact_features, a=param[0], loc=param[1], scale=param[2])
		gamma_features.append(param_features)

	gamma_features = np.hstack(gamma_features)
	return gamma_features

def custom_compute_distances(residue_pairs, scheme, gamma_pdf_parameters, traj_file, structure, iterative):
	if len(residue_pairs) > 0:
		if structure is not None:
			top = md.load_frame(traj_file, index=0, top=structure).topology
		else:
			top = md.load_frame(traj_file, index=0).topology
		chains = [c for c in  top.chains]
		if chains[0].id == 'A' and chains[0].n_residues < 2:
			chains[0].id = 'B'
			chains[1].id = 'A'
			chains[2].id = 'D'
			chains[3].id = 'C'

		residue_pairs = convert_residue_pairs_to_mdtraj_indices(top, residue_pairs)
		contact_features = []
		if iterative:
			try:
				for chunk in md.iterload(traj_file, chunk = 5000):
					chunk_features = md.compute_contacts(chunk, contacts = residue_pairs, scheme = scheme, ignore_nonprotein=False)[0]
					contact_features.append(chunk_features)
				contact_features = np.concatenate(contact_features)
			except Exception as e:
				print(str(e))
				print("Failed")
				return
		else:
			try:
				if structure is not None:
					traj = md.load(traj_file, top=structure)
				else:
					traj = md.load(traj_file)
				contact_features = md.compute_contacts(traj, contacts = residue_pairs, scheme = scheme, ignore_nonprotein=False)[0]
			except Exception as e:
				print(str(e))
				print("Failed for traj")
				return

		if gamma_pdf_parameters is not None:
			contact_features = apply_gamma_pdf_parameters(contact_features, gamma_pdf_parameters)

		return contact_features	

def read_and_featurize(traj_file, features_dir = None, condition=None, 
											 dihedral_types = ["phi", "psi", "chi1", "chi2"], 
											 dihedral_residues = None, contact_residue_pairs = [],
											 ca_residue_pairs = [], atom_residue_pairs = [], 
											 iterative = True, structure = None,
											 gamma_pdf_parameters=None, anton=False,
											 binarize=0.5):
	if structure is not None:
		traj = md.load_frame(traj_file, index=0, top=structure)
		top = traj.topology

	else:
		traj = md.load_frame(traj_file, index=0)
		top = traj.topology 

	dihedral_indices = []
	residue_order = []
	manual_features = []
	if dihedral_residues is not None:
		if not anton:
			atom_indices = convert_residues_to_mdtraj_atom_indices(top, dihedral_residues)
			traj = traj.atom_slice(atom_indices)
			dihedral_types = [s.lower() for s in dihedral_types]
			if "phi" in dihedral_types:
				phi_indices, _ = md.compute_phi(traj)
				dihedral_indices.append(phi_indices)
			if "psi" in dihedral_types:
				psi_indices, _ = md.compute_psi(traj)
				dihedral_indices.append(psi_indices)
			if "chi2" in dihedral_types:
				chi2_indices, _ = md.compute_chi2(traj)
				dihedral_indices.append(chi2_indices)
			if "chi1" in dihedral_types:
				chi1_indices, _ = md.compute_chi1(traj)
				dihedral_indices.append(chi1_indices)
			dihedral_indices = np.vstack(dihedral_indices)
			dihedrals = md.compute_dihedrals(md.load(traj_file).atom_slice(atom_indices), dihedral_indices)
			manual_features.append(np.cos(dihedrals))
			manual_features.append(np.sin(dihedrals))
		else:
			for dihedral_type in dihedral_types:
				if dihedral_type == "phi": dihedral_indices.append(phi_indices(fix_topology(top), dihedral_residues))
				if dihedral_type == "psi": dihedral_indices.append(psi_indices(fix_topology(top), dihedral_residues))
				if dihedral_type == "chi1": dihedral_indices.append(chi1_indices(fix_topology(top), dihedral_residues))
				if dihedral_type == "chi2": dihedral_indices.append(chi2_indices(fix_topology(top), dihedral_residues))

			dihedral_angles = []

			for dihedral_type in dihedral_indices:
				try:
					angles = np.transpose(ManualDihedral.compute_dihedrals(traj=traj,indices=dihedral_type))
				except:
					print("Cannot compute dihedrals for %s" %traj_file)
					return
				dihedral_angles.append(np.sin(angles))
				dihedral_angles.append(np.cos(angles))

			manual_features = np.transpose(np.concatenate(dihedral_angles))


	if len(contact_residue_pairs) > 0:
		manual_features.append(custom_compute_distances(contact_residue_pairs, 'closest-heavy', gamma_pdf_parameters, traj_file, structure, iterative))
	if len(ca_residue_pairs) > 0:
		manual_features.append(custom_compute_distances(ca_residue_pairs, 'CA', gamma_pdf_parameters, traj_file, structure, iterative))
	if len(contact_residue_pairs) > 0 or len(ca_residue_pairs) > 0:
		manual_features = np.hstack(manual_features)


	if len(atom_residue_pairs) > 0:
		
		chains = [c for c in  top.chains]
		if chains[0].id == 'A' and chains[0].n_residues < 2:
			chains[0].id = 'B'
			chains[1].id = 'A'
			chains[2].id = 'D'
			chains[3].id = 'C'

		chain_r = [chain for chain in chains if chain.id=='R']
		if len(chain_r) == 0:
			print("No Chain R!!")
			for i, chain in enumerate(chains):
				if len([r for r in chain.residues if r.is_protein]) > 10:
					top.chains(i).id = 'R'

		atom_residue_pairs = convert_atom_residue_pairs_to_mdtraj_indices(top, atom_residue_pairs)
		print(atom_residue_pairs)
		contact_features = []
		if iterative:
			try:
				for chunk in md.iterload(traj_file, chunk = 5000):
					chunk_features = compute_atom_residue_pair_distances(chunk, contacts=atom_residue_pairs, scheme='closest-heavy')[0]
					contact_features.append(chunk_features)
				contact_features = np.concatenate(contact_features)
			except Exception as e:
				print(str(e))
				print("Failed")
				return
		else:
			if 1==1:
				if structure is not None:
					traj = md.load(traj_file, top=structure)
				else:
					traj = md.load(traj_file)
					contact_features = compute_atom_residue_pair_distances(traj, contacts=atom_residue_pairs, scheme='closest-heavy')[0]
					#print(contact_features)
					print(np.shape(contact_features))
			else:
				print("Failed for traj")
				return

		if gamma_pdf_parameters is not None:
			contact_features = apply_gamma_pdf_parameters(contact_features, gamma_pdf_parameters)		

		if len(manual_features) > 0:
			manual_features = np.column_stack((manual_features, contact_features))
		else:
			manual_features = contact_features



	print(("new features %s has shape: " %traj_file))
	print((np.shape(manual_features)))

	if condition is None:
		condition = traj_file.split("/")[len(traj_file.split("/"))-1]
		condition = condition.split(".")[0]

	filename = "%s/%s.dataset" %(features_dir, condition)
	if os.path.exists(filename):
		subprocess.call("rm -rf %s" %filename, shell=True)

	if binarize is not None:
		from sklearn.preprocessing import binarize
		manual_features = binarize(manual_features, threshold=binarize)
		manual_features = manual_features.astype('uint4')
		print(manual_features.shape)

	save_dataset(manual_features, filename)

def compute_atom_residue_pair_distances(traj, contacts, scheme='closest-heavy'):
	"""
	Computes the closest distance between a given atom and a given residue, and repeats
	for each such (atom_index, residue_index) tuple in list `contacts`

	Example input:
	traj = rep1.h5
	contacts = [(10, 30), (7, 24), (3, 89)]
	"""
	if scheme == 'closest':
	    residue_membership = [[atom.index for atom in residue.atoms]
	                          for residue in traj.topology.residues]
	elif scheme == 'closest-heavy':
	    # then remove the hydrogens from the above list
	    residue_membership = [[atom.index for atom in residue.atoms if "H" not in atom.name]
	                          for residue in traj.topology.residues]

	residue_lens = [len(ainds) for ainds in residue_membership]

	atom_pairs = []
	n_atom_pairs_per_atom_residue_pair = []
	for pair in contacts:
	    atom_pairs.extend(list(itertools.product([pair[0]], residue_membership[pair[1]])))
	    n_atom_pairs_per_atom_residue_pair.append(residue_lens[pair[1]])

	atom_distances = md.compute_distances(traj, atom_pairs)

	# now squash the results based on residue membership
	n_atom_residue_pairs = len(contacts)
	distances = np.zeros((len(traj), n_atom_residue_pairs), dtype=np.float32)

	for i in range(n_atom_residue_pairs):
	    index = int(np.sum(n_atom_pairs_per_atom_residue_pair[:i]))
	    n = n_atom_pairs_per_atom_residue_pair[i]
	    distances[:, i] = atom_distances[:, index : index + n].min(axis=1)

	return distances, contacts


def compute_contacts_below_cutoff(traj_file_frame, cutoff = 100000.0, contact_residues = [], 
																	anton = False, structure=None, within_turn=True):
	traj_file = traj_file_frame[0]
	print("structure")
	print(structure)
	if structure is not None:
		frame = md.load_frame(traj_file, index=0, top=structure)
	else:
		frame = md.load_frame(traj_file, index = 0)
	#frame = fix_traj(frame)
	top = frame.topology
	
	distance_residues = []
	res_indices = []
	residue_to_mdtraj_index = {}
	residue_full_infos = []

	for i in range(0, len(contact_residues)):
		residue = contact_residues[i]
		indices = [res.index for res in top.residues if residue.is_mdtraj_res_equivalent(res)]
		if len(indices) == 0:
			print(("No residues in trajectory for residue %d chain %s" %(residue.resSeq, residue.chain_id)))
			continue
		else:
			ind = indices[0]
			for j in indices:
				if j != ind: 
					#print("Warning: multiple res objects for residue %d " %residue)
					matching_residues = [res for res in top.residues if res.index==ind]
					matching_atoms = []
					for matching_residue in matching_residues:
						matching_atoms += [str(a) for a in matching_residue.atoms]
					if "CB" in matching_atoms:
						ind = j
			res_indices.append(ind)
			residue.res_name = str(top.residue(ind))
			distance_residues.append(residue)
			residue_to_mdtraj_index[residue] = ind
			

	
	residue_combinations = itertools.combinations(distance_residues, 2)
	if within_turn:
		residue_pairs = [c for c in residue_combinations]
	else:
		residue_pairs = [c for c in residue_combinations if abs(int(c[0].resSeq - c[1].resSeq)) > 4.]
	mdtraj_index_combinations = []
	contact_features = []
	for combination in residue_pairs:
		res0 = combination[0]
		res1 = combination[1]
		mdtraj_index0 = residue_to_mdtraj_index[res0]
		mdtraj_index1 = residue_to_mdtraj_index[res1]
		mdtraj_index_combinations.append((mdtraj_index0, mdtraj_index1))
		pair = [res0, res1]
		contact_features.append(pair)

	
	print("mdraj_index_combinations[0:10]")
	print(mdtraj_index_combinations[0:10])
	print("contact_features[0:10]")
	print(contact_features[0:10])	
	final_residue_pairs = []
	final_mdtraj_index_pairs = []

	print(("About to compute %d features" %len(mdtraj_index_combinations)))
	distances = md.compute_contacts(frame, contacts = mdtraj_index_combinations, scheme = 'closest-heavy', ignore_nonprotein=False)[0]
	#print(distances)
	print((np.shape(distances)))
	print("cutoff")
	print(cutoff)
	print("distances[0:10]")
	print(distances[0:10])
	for i in range(0, len(distances[0])):
		distance = distances[0][i]
		#print(distance)
		if distance < cutoff:
			final_mdtraj_index_pairs.append(mdtraj_index_combinations[i])
			final_residue_pairs.append(contact_features[i])

	print(("There are %d residue-residue contacts below cutoff in structure." %len(final_residue_pairs)))
	
	return final_residue_pairs


def which_trajs_to_featurize(traj_dir, traj_ext, features_dir, excluded_trajs, redo=False):
	all_trajs = get_trajectory_files(traj_dir, traj_ext)
	trajs = []
	for fulltraj in all_trajs:
		#if "H-05" not in fulltraj and "A-00" not in fulltraj: continue
		traj = fulltraj.split("/")
		filename = traj[len(traj)-1]
		#if agonist_bound is not False and filename[0] not in agonist_bound: continue
		filename_noext = filename.split(".")[0]
		if not redo:
			if os.path.exists("%s/%s.dataset" %(features_dir, filename_noext)):
				print("already featurized")	
				continue 

		include = True
		for excluded_traj in excluded_trajs:
			if excluded_traj in filename_noext:
				include = False
		if not include: continue

		trajs.append(fulltraj)

	return(trajs)	

def fix_chain_names_single(traj_file):
	traj_changed = False

	traj = md.load(traj_file)
	top = traj.topology
	chains = top.chains
	chains_r = [c for c in chains if c.id=='R']
	if len(chains_r) == 0:
		traj_changed = True
		for i, chain in enumerate(traj.topology.chains):
			if len([r for r in chain.residues if r.is_protein]) > 10:
				traj.topology.chain(i).id = 'R'

	ligand_residues = [r for r in top.residues if "lig" in str(r).lower()]
	chains_to_change_to_r = []
	for ligand_residue in ligand_residues:
		if ligand_residue.chain.id != 'R':
			if ligand_residue.chain.id not in chains_to_change_to_r:
				chains_to_change_to_r.append(ligand_residue.chain.id)

	if len(chains_to_change_to_r) > 0:
		traj_changed = True
		for i, chain in enumerate(traj.topology.chains):
			if chain.id in chains_to_change_to_r:
				traj.topology.chain(i).id = 'L'

	if traj_changed:
		traj.save(traj_file)
	else:
		return

def fix_chain_names(trajectories, worker_pool=None, parallel=False):
	if worker_pool is not None:
		worker_pool.map_sync(fix_chain_names_single, trajectories)
	elif parallel:
		pool = mp.Pool(mp.cpu_count())
		pool.map(fix_chain_names_single, trajectories)
		pool.terminate()
	else:
		for t in trajectories:
			fix_chain_names_single(t)

def convert_mdtraj_atom_to_residue_object(mdtraj_atom_index, top, dihedral_type=None):
	mdtraj_residue = top.atom(mdtraj_atom_index).residue
	chain_id = mdtraj_residue.chain.id
	resSeq = mdtraj_residue.resSeq
	res_name = mdtraj_residue.name
	res_obj = Residue(resSeq, chain_id=chain_id, res_name="%s%d" %(str(res_name).title(), resSeq))
	if dihedral_type is not None:
		dihedral_feature = DihedralFeature(res_obj, dihedral_type)
		return res_obj, dihedral_feature
	else:
		return res_obj

def convert_dihedral_indices_to_residue_objects(dihedral_indices, dihedral_type, top):
	res_objs = []
	dihedral_features = []
	for i in range(0, dihedral_indices.shape[0]):
		res_obj, dihedral_feature = convert_mdtraj_atom_to_residue_object(dihedral_indices[i][0], top, dihedral_type)
		res_objs.append(res_obj) 
		dihedral_features.append(dihedral_feature)
	return res_objs, dihedral_features

def find_dihedral_residues(trajectory, dihedral_types, residues):
	top = trajectory.topology
	atom_indices = convert_residues_to_mdtraj_atom_indices(top, residues)
	traj = trajectory.atom_slice(atom_indices)
	top = traj.topology 
	dihedral_types = [s.lower() for s in dihedral_types]
	res_objs = []
	dihedral_features = []
	if "phi" in dihedral_types:
		phi_indices, _ = md.compute_phi(traj)
		phi_res_objs, phi_dihedral_features = convert_dihedral_indices_to_residue_objects(phi_indices, "phi", top)
		res_objs += phi_res_objs
		dihedral_features += phi_dihedral_features
	if "psi" in dihedral_types:
		psi_indices, _ = md.compute_psi(traj)
		psi_res_objs, psi_dihedral_features = convert_dihedral_indices_to_residue_objects(psi_indices, "psi", top)
		res_objs += psi_res_objs
		dihedral_features += psi_dihedral_features
	if "chi2" in dihedral_types:
		chi2_indices, _ = md.compute_chi2(traj)
		chi2_res_objs, chi2_dihedral_features = convert_dihedral_indices_to_residue_objects(chi2_indices, "chi2", top)
		res_objs += chi2_res_objs
		dihedral_features += chi2_dihedral_features
	if "chi1" in dihedral_types:
		chi1_indices, _ = md.compute_chi1(traj)
		chi1_res_objs, chi1_dihedral_features = convert_dihedral_indices_to_residue_objects(chi1_indices, "chi1", top)
		res_objs += chi1_res_objs
		dihedral_features += chi1_dihedral_features
	return res_objs, dihedral_features


def featurize_contacts_custom(traj_dir, features_dir, traj_ext, structures, 
															traj_top_structure = None,
															contact_residue_pairs_file = None, 
															dihedral_residues = [], 
															dihedral_types = None, 
															contact_residues = None, 
															agonist_bound = False, 
															residues_map = None, 
															contact_cutoff = None, 
															user_specified_contact_residue_pairs = [],
															user_specified_atom_residue_pairs = [],
															gamma_pdf_parameters=None,
															parallel = False, exacycle = False,
															iterative=True,
															load_from_file=False,
															worker_pool=None,
															redo=False,
															schemes=[],
															excluded_trajs=[],
															binarize=0.5,
															within_turn=True,
															simulation_reference_filename=""):
	'''
	Nb: The input to this function, either contact_residues or contact_residue_pairs_file, must contain instances 
	of object Residue(). The .resSeq attribute of each such instance must refer to residue numbering in your reference
	structure/PDB. This is to standardize it across multiple simulation conditions. The residues_map must be given to map 
	the ref structure/PDB residue ID numbers to the "resSeq" attributes that mdtraj requires.

	you can also input a residue_map, a dictionary that maps residue_object --> residue_object. The reason for this is that there
		is not a consensus residue numbering for the same protein.
	'''
	trajs = which_trajs_to_featurize(traj_dir, traj_ext, features_dir, excluded_trajs, redo)
	#trajs = get_trajectory_files(traj_dir, ".pdb")
	if not os.path.exists(features_dir): os.makedirs(features_dir)
	if load_from_file:
		contact_residue_pairs = generate_features(contact_residue_pairs_file)
		print("Which contacts to measure already chosen.")
		if exacycle: contact_residue_pairs = [residues_map[key] for key in contact_residue_pairs]
	else:
		final_feature_objects = []

		contact_residue_pairs = []
		ca_residue_pairs = []
		for structure in structures:
			print("structure")
			print(structure) 
			structure_contact_residue_pairs = compute_contacts_below_cutoff([structure,0], cutoff = contact_cutoff, contact_residues = contact_residues, anton = False, structure = traj_top_structure, within_turn=within_turn)
			for pair in structure_contact_residue_pairs:
				if sorted(pair) not in contact_residue_pairs: contact_residue_pairs.append(sorted(pair))

		contact_residue_pairs = sorted([ContactFeature(pair[0], pair[1]) for pair in contact_residue_pairs])
		dihedral_residues = []
		#for pair in contact_residue_pairs:
	#		if pair.residue_i not in dihedral_residues:
#				dihedral_residues.append(pair.residue_i)#
#			if pair.residue_j not in dihedral_residues:#
#				dihedral_residues.append(pair.residue_j)
		print("dihedral_residues:")
		print(dihedral_residues)
		if dihedral_types is not None:
			dihedral_res_objs, dihedral_features = find_dihedral_residues(md.load_frame(trajs[0], index=0), dihedral_types, dihedral_residues)
			cos_dihedral_features = copy.deepcopy(dihedral_features)
			sin_dihedral_features = copy.deepcopy(dihedral_features)
			for d in cos_dihedral_features:
				d.trig = "cos"
			for d in sin_dihedral_features:
				d.trig = "sin"
			final_feature_objects += cos_dihedral_features
			final_feature_objects += sin_dihedral_features

			print(cos_dihedral_features[0])

		if "CA" in schemes or "ca" in schemes:
			old_pairs = copy.deepcopy(contact_residue_pairs)
			t0 = md.load_frame(simulation_reference_filename, index=0)
			residue_pairs = convert_residue_pairs_to_mdtraj_indices(t0.topology, contact_residue_pairs)
			ca_pairs = md.compute_contacts(t0, contacts = residue_pairs, scheme = 'CA', ignore_nonprotein=False)[1]
			ca_pairs = [sorted((t0.topology.residue(pair[0]).resSeq, t0.topology.residue(pair[1]).resSeq)) for pair in ca_pairs]
			for pair in old_pairs:
				if sorted((pair.residue_i.resSeq, pair.residue_j.resSeq)) in ca_pairs:
					pair.residue_i.CA = True 
					pair.residue_j.CA = True
					ca_residue_pairs.append(pair)
			ca_residue_pairs = sorted(ca_residue_pairs)
		#ordering = np.argsort([r[0].resSeq for r in contact_residue_pairs]).tolist()
		#contact_residue_pairs = [contact_residue_pairs[i] for i in ordering]
		top = md.load_frame(simulation_reference_filename, index=0).topology
		for residue_pair in user_specified_contact_residue_pairs:
			res_i, res_j = residue_pair.residue_i, residue_pair.residue_j
			mdtraj_i = convert_residue_to_mdtraj_index(top, res_i)
			res_i.res_name = top.residue(mdtraj_i).__repr__().title()
			mdtraj_j = convert_residue_to_mdtraj_index(top, res_j)
			res_j.res_name = top.residue(mdtraj_j).__repr__().title()

		for atom_residue_pair in user_specified_atom_residue_pairs:
			atom_i, res_j = atom_residue_pair.residue_i, atom_residue_pair.residue_j
			mdtraj_i = convert_atom_to_mdtraj_index(top, atom_i)
			atom_i.mdtraj_rep = top.atom(mdtraj_i).__repr__().title()
			mdtraj_j = convert_residue_to_mdtraj_index(top, res_j)
			res_j.res_name = top.residue(mdtraj_j).__repr__().title()

		contact_residue_pairs += user_specified_contact_residue_pairs
		final_features = contact_residue_pairs + ca_residue_pairs

		if gamma_pdf_parameters is not None:
			base_features = copy.deepcopy(final_features)
			final_features = []
			for param in gamma_pdf_parameters:
				param_features = copy.deepcopy(base_features)
				for feature in param_features:
					feature[0].mean = param[0]
					feature[0].loc = param[1]
					feature[0].scale = param[2]
					feature[1].mean = param[0]
					feature[1].loc = param[1]
					feature[1].scale = param[2]
				final_features += param_features

		final_contact_features = final_features

		final_feature_objects += final_contact_features + user_specified_atom_residue_pairs

		print(("There are %d features to be used in featurization." % len(final_feature_objects)))
		print("Saving contact feature residue pairs to disk.")
		with open(contact_residue_pairs_file, "wb") as f:
			pickle.dump(final_feature_objects, f)


	print("About to featurize trajectories based on the chosen featurization scheme.")
	print(final_feature_objects)
	#print("atom_residue_pairs=")
	featurize_partial = partial(read_and_featurize, features_dir = features_dir,
								contact_residue_pairs = contact_residue_pairs,
								ca_residue_pairs = ca_residue_pairs,
								atom_residue_pairs=user_specified_atom_residue_pairs, iterative=iterative, structure=traj_top_structure,
								gamma_pdf_parameters=gamma_pdf_parameters, binarize=binarize)

	if worker_pool is not None:
		random.shuffle(trajs)
		worker_pool.map_sync(featurize_partial, trajs)
	elif parallel:
		pool = mp.Pool(mp.cpu_count())
		pool.map(featurize_partial, trajs)
		pool.terminate()
	else:
		for traj in trajs:
			print("Featurizing %s" % traj)
			featurize_partial(traj)

	print("Completed featurizing")

def save_feature_residues_pkl(traj_dir, features_dir, traj_ext, structures, contact_residue_pairs_file = None, dihedral_residues = None, dihedral_types = None, contact_residues = None, agonist_bound = False, residues_map = None, contact_cutoff = None, parallel = False, exacycle = False):
	if residues_map is not None:
		contact_residues = [r for r in contact_residues if r in list(residues_map.keys())]
		if exacycle: contact_residues = [residues_map[key] for key in contact_residues]	

	if contact_residue_pairs_file == "" or (not os.path.exists(contact_residue_pairs_file)):
		contact_residue_pairs = []
		for structure in structures: 
			contact_residue_pairs.append(compute_contacts_below_cutoff([structure,0], cutoff = contact_cutoff, contact_residues = contact_residues, anton = False)[0])
		contact_residue_pairs = sorted(list(set(contact_residue_pairs)))
		with open(contact_residue_pairs_file, "wb") as f:
			pickle.dump(contact_residue_pairs, f)
	else:
		print("Features already computed")
		contact_residue_pairs = generate_features(contact_residue_pairs_file)
		if exacycle: contact_residue_pairs = [residues_map[key] for key in contact_residue_pairs]
	print(contact_residue_pairs)
	print(("Number of contact pairs = %d" %len(contact_residue_pairs)))

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
			condition = get_condition(fulltraj).split(".h5")[0]
			if os.path.exists("%s/%s" %(features_dir, condition)):
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

def compute_pnas_coords_and_distance(traj_file, inactive, active, scale = 7.14, residues_map = None, structure=None, 
	connector_residues=[], npxxy_residues=[], tm6_tm3_residues=[]):
	print("featurizing %s" %traj_file)
	if structure is not None:
		traj = md.load(traj_file, top=structure)
	else:
		traj = md.load(traj_file)
	inactive_tuple = np.array([helix6_helix3_dist(inactive, residues=tm6_tm3_residues) / scale, rmsd_npxxy(inactive, inactive, residues=npxxy_residues)])
	active_tuple = np.array([helix6_helix3_dist(active, residues=tm6_tm3_residues) / scale, rmsd_npxxy(active, inactive, residues=npxxy_residues)])
	traj_coords = [helix6_helix3_dist(traj, residues=tm6_tm3_residues) / scale, rmsd_npxxy(traj, inactive, residues=npxxy_residues), rmsd_npxxy(traj, active, residues=npxxy_residues), rmsd_connector(traj, inactive, residues=connector_residues), rmsd_connector(traj, active, residues=connector_residues)]
	traj_coords = np.transpose(np.vstack(traj_coords))
	active_vectors = traj_coords[:,[0,1]] - np.transpose(active_tuple)
	inactive_vectors = traj_coords[:,[0,1]] - np.transpose(inactive_tuple)

	inactive_distances = np.linalg.norm(inactive_vectors, axis = 1)
	active_distances = np.linalg.norm(active_vectors, axis = 1)
	distances = [inactive_distances, active_distances]
	#print distances[1]
	print("traj_coords")
	print(traj_coords)
	print("distances")
	print(distances)
	return [traj_coords, distances]

def compute_user_defined_features(traj_file, inactive, active, structure=None,
									feature_name_residues_dict = {}, save_dir=None, overwrite=True):
	import shutil
	
	traj_basename = os.path.basename(traj_file)
	traj_basename = os.path.splitext(traj_basename)[0]
	save_file = "%s/%s.dataset" %(save_dir, traj_basename)
	print("save_file = %s" %save_file)
	if os.path.exists(save_file) and overwrite is False:
		features = load_file(save_file)
		return features
	else:
		print("featurizing %s" %traj_file)
		if structure is not None:
			traj = md.load(traj_file, top=structure)
		else:
			traj = md.load(traj_file)

		top = traj.topology 

		chains = [c for c in  top.chains]
		if chains[0].id == 'A' and chains[0].n_residues < 2:
			chains[0].id = 'B'
			chains[1].id = 'A'
			chains[2].id = 'D'
			chains[3].id = 'C'

		features = []
		for name in sorted(feature_name_residues_dict.keys()):
			#print name
			residues = feature_name_residues_dict[name]
			if name == "tm6_tm3_dist":
				features.append(helix6_helix3_dist(traj, residues=residues))	
			elif name == "rmsd_npxxy_active":
				features.append(rmsd_npxxy(traj, active, residues=residues))
			elif name == "rmsd_npxxy_inactive":
				features.append(rmsd_npxxy(traj, inactive, residues=residues))
			elif name == "rmsd_connector_active":
				features.append(rmsd_connector(traj, active, residues=residues))
			elif name == "rmsd_connector_inactive":
				features.append(rmsd_connector(traj, inactive, residues=residues))
			elif "ca_dist" in name:
				features.append(helix6_helix3_dist(traj, residues=residues))
			elif "closest_dist" in name:
				features.append(compute_closest_heavy(traj, residues=residues))
			elif "packing" in name:
				features.append(compute_average_min_distance(traj, residues[0], residues[1]))
			elif "stacking" in name:
				features.append(compute_pi_interactions(traj, ))
			elif "rmsd" in name:
				if "inactive" in name:
					features.append(compute_distance(traj, inactive, residues=residues))
				else:
					features.append(compute_distance(traj, active, residues=residues))

		features = np.transpose(np.array(features))

		try:
			save_dataset(features, save_file)
		except:
			shutil.rmtree(save_file)
			save_dataset(features, save_file)
		return features

def compute_user_defined_features_wrapper(traj_dir, traj_ext, inactive_file, active_file, structure,
										  feature_name_residues_dict, save_file, save_dir, worker_pool=None, 
										  parallel=True, trajs=None, overwrite=True):
	inactive = md.load(inactive_file)
	active = md.load(active_file)
	compute_user_defined_features_partial = partial(compute_user_defined_features, 
													inactive=inactive, active=active,
													structure=structure, 
													feature_name_residues_dict=feature_name_residues_dict,
													save_dir=save_dir, overwrite=overwrite)
	if trajs is None:
		trajs = get_trajectory_files(traj_dir, traj_ext)

	if worker_pool is not None:
		features = worker_pool.map_sync(compute_user_defined_features_partial, trajs)
	elif parallel:
		pool = mp.Pool(mp.cpu_count())
		features = pool.map(compute_user_defined_features_partial, trajs)
		pool.terminate()
	else:
		features = []
		for traj in trajs:
			traj_features = compute_user_defined_features_partial(traj)
			features.append(traj_features)	
	verbosedump(features, save_file)

def get_aromatic_feature_names(pi_pi_ring_pairs, cation_pi_pairs):
	features = []
	for ring_pair in pi_pi_ring_pairs:
		features.append(AromaticFeature(ring_pair[0][0], ring_pair[1][0], "Pi Parallel"))
	for ring_pair in pi_pi_ring_pairs:
		features.append(AromaticFeature(ring_pair[0][0], ring_pair[1][0], "Pi T"))
	for ring_pair in cation_pi_pairs:
		features.append(AromaticFeature(ring_pair[0], ring_pair[1][0], "Cation-Pi"))
	return features

def compute_pi_interactions_traj(traj_file, pi_pi_ring_pairs, cation_pi_pairs,
																 save_dir, structure=None):
	traj_basename = os.path.basename(traj_file)
	traj_basename = os.path.splitext(traj_basename)[0]
	save_file = "%s/%s.dataset" %(save_dir, traj_basename)
	print("save_file = %s" %save_file)
	if os.path.exists(save_file):
		features = load_file(save_file)
		return features
	else:
		print("featurizing %s" %traj_file)
		if structure is not None:
			traj = md.load(traj_file, top=structure)
		else:
			traj = md.load(traj_file)

		top = traj.topology 

		chains = [c for c in  top.chains]
		if chains[0].id == 'A' and chains[0].n_residues < 2:
			chains[0].id = 'B'
			chains[1].id = 'A'
			chains[2].id = 'D'
			chains[3].id = 'C'

	features = compute_pi_interactions(traj, pi_pi_ring_pairs, cation_pi_pairs)
	save_dataset(features, save_file)

	return features

def compute_aromatic_features_wrapper(traj_dir, traj_ext, structure, pi_pi_ring_pairs,
																		  cation_pi_pairs, save_file, save_dir, 
																		  worker_pool=None, parallel=True, traj_files=None):

	compute_pi_interactions_partial = partial(compute_pi_interactions_traj,
																						pi_pi_ring_pairs=pi_pi_ring_pairs,
																						cation_pi_pairs=cation_pi_pairs,
																						save_dir=save_dir,
																						structure=structure)
	if traj_files is None:
		traj_files = get_trajectory_files(traj_dir, traj_ext)

	if worker_pool is not None:
		features = worker_pool.map_sync(compute_pi_interactions_partial, traj_files)
	elif parallel:
		pool = mp.Pool(mp.cpu_count())
		features = pool.map(compute_pi_interactions_partial, traj_files)
		pool.terminate()
	else:
		features = []
		for traj_file in traj_files:
			traj_features = compute_pi_interactions_partial(traj_file)
			features.append(traj_features)
	save_dataset(features, save_file)



def retain_features_within_range(features_dir, feature_residues, cutoff,
								 frame_proportion, new_features_file,
								 new_features_pkl, worker_pool=None):

	if worker_pool is not None:
		features = worker_pool.map_sync(load_file, get_trajectory_files(features_dir, ".dataset"))
	else:
		pool = mp.Pool(mp.cpu_count())
		features = pool.map(load_file, get_trajectory_files(features_dir, ".dataset"))
		pool.terminate()
	conc_features = np.concatenate(features)
	new_feature_indices = []

	for feature in range(0, conc_features.shape[1]):
		indices = np.nonzero(conc_features[:,feature] < cutoff)[0]
		if len(indices) > frame_proportion * conc_features.shape[0]:
			new_feature_indices.append(feature)

	new_features = [f[:,new_feature_indices] for f in features]
	
	new_feature_residues = []
	for new_feature_index in new_feature_indices:
		new_feature_residues.append(feature_residues[new_feature_index])

	with open(new_features_pkl, "wb") as f:
		pickle.dump(new_feature_residues, f)

	save_dataset(new_features, new_features_file)

#credit: http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
def is_consecutive(data, stepsize=1, min_length=5):
    consecutives = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    #print(consecutives)
    for consecutive in consecutives:
    	if len(consecutive) >= min_length:
    		return True

    return False

def reaction_coordinate_sampler(traj_files, traj_ext, rxn_coords, 
								titles, coords_bounds_dict, 
								save_file):

	traj_files = [os.path.basename(t) for t in traj_files]
	coord_to_trajectories_dict = {}

	for coord in coords_bounds_dict.keys():
		bounds = coords_bounds_dict[coord]
		print(("Analyzing %s" % coord))
		column = titles.index(coord)
		coord_to_trajectories_dict[coord] = []
		for i, projected_traj in enumerate(rxn_coords):
			#print("Analyzing trajectory %s" % traj_files[i])
			projection = projected_traj[:,column]
			span = True
			for bound in bounds:
				indices = np.nonzero((projection > bound[0]) & (projection < bound[1]))[0]
				#print(indices)
				if not is_consecutive(indices):
					span=False
			if span: coord_to_trajectories_dict[coord].append((i, traj_files[i]))

	print(coord_to_trajectories_dict)
	if save_file is not None:
		write_map_to_csv(save_file, coord_to_trajectories_dict, [])

	return coord_to_trajectories_dict

def featurize_pnas_distance_traj(traj_dir, ianctive, active, features_dir):
	#pnas_distances = 
	return

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

def featurize_pnas_distance(traj_dir, features_dir, ext, inactive_dir, 
														active_dir, inactive_distances_dir, 
														active_distances_dir, coords_dir, 
														inactive_distances_csv, active_distances_csv, 
														coords_csv, scale = 7.14, residues_map = None, 
														structure=None, connector_residues=[], 
														npxxy_residues=[], tm6_tm3_residues=[]):
	if not os.path.exists(features_dir): os.makedirs(features_dir)

	inactive = md.load(inactive_dir)
	active = md.load(active_dir)

	agonist_bound = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	trajs = get_trajectory_files(traj_dir, ext = ext)

	featurize_partial = partial(compute_pnas_coords_and_distance, inactive = inactive, 
															active = active, scale = scale, residues_map = residues_map, structure=structure, 
															connector_residues=connector_residues, npxxy_residues=npxxy_residues, tm6_tm3_residues=tm6_tm3_residues)
	#features = []
	#for traj in trajs:
	#	features.append(featurize_partial(traj))
	pool = mp.Pool(mp.cpu_count())
	features = pool.map(featurize_partial, trajs)
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

def featurize_pnas_distance_pdbs(traj_dir, new_filename, features_dir, inactive_dir, active_dir, inactive_distances_dir, active_distances_dir, coords_dir, scale = 7.14):
	#if not os.path.exists(features_dir): os.makedirs(features_dir)

	inactive = md.load(inactive_dir)
	active = md.load(active_dir)

	agonist_bound = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	samples = get_trajectory_files(traj_dir, ext = ".pdb")[0:1]
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

def featurize_sasa_individual(traj_file, bp_residues, anton, stride):
	print(traj_file)
	traj = md.load(traj_file, stride = stride)
	top = traj.topology
	protein_atoms = [a.index for a in top.atoms if a.residue.is_protein]
	traj = traj.atom_slice(protein_atoms)
	
	print(traj)

	#try:
	traj = fix_traj(traj)
	top = traj.topology
	sasa = md.shrake_rupley(traj, mode = 'atom')
	#except:
	#	return []

	bp_atoms = [a.index for a in top.atoms if a.residue.resSeq in bp_residues]
	print((np.shape(sasa)))
	sasa = sasa[:, bp_atoms]
	total_sasa = sasa.sum(axis=1)
	return(total_sasa)

def featurize_sasa(traj_dir, traj_ext, bp_residues, sasa_file, residues_map = None, anton = False, skip = 1, stride = 0):
	trajs = get_trajectory_files(traj_dir, traj_ext)

	
	print(("therer are initially %d bp residues" %len(bp_residues)))
	if residues_map is not None:
		bp_residues = map_residues(residues_map, bp_residues)
	print(("therer are now %d bp residues" %len(bp_residues)))

	featurize_partial = partial(featurize_sasa_individual, bp_residues = bp_residues, anton = anton, stride = stride)
	pool = mp.Pool(mp.cpu_count())
	sasa = pool.map(featurize_partial, [trajs[i] for i in range(0,len(trajs),skip)])
	pool.terminate()
	#sasa = []
	#for traj in trajs[0:5]:
	#	sasa.append(featurize_partial(traj))
	sasa = np.concatenate(sasa)
	#print("SASA is:")
	#print(sasa)
	sample_names = [sample.split(".")[0] for sample in trajs]
	sasa_map = {}
	for i in range(0, len(sasa)):
		sasa_map[sample_names[i]] = [sasa[i]]
	write_map_to_csv(sasa_file, sasa_map, ["sample", "sasa"])
	#print(sasa)
	np.savetxt(sasa_file, sasa, delimiter=",")
	#plt.hist(sasa, bins=50)
	

	print("Completed featurizing")


#Compute standardized values for each feature 
def standardize_features(features_dir, features_ext, standardized_features_dir):
	if not os.path.exists(standardized_features_dir): os.makedirs(standardized_features_dir)
	feature_files = get_trajectory_files(features_dir, features_ext)
	features = load_file_list(feature_files)
	concatenated_features = np.concatenate(features)
	means = np.mean(concatenated_features, axis = 0)
	stdevs = np.std(concatenated_features, axis = 0)
	standardized_features = []
	for X in features: 
		X -= means
		X /= stdevs 
		standardized_features.append(X)

	print("Finished standardizing features")
	for i in range(0, len(feature_files)):
		filename = feature_files[i].split("/")[len(feature_files[i].split("/"))-1]
		new_filename = "%s/%s" %(standardized_features_dir, filename)
		verbosedump(standardized_features[i], new_filename)
	print("Finished saving all standardized features")
	return 

def sample_tIC_extremes(tica_projected_coords_dir, features_dir, standardized_features_dir, tica_extremes_dir, ext, percentile):
	if not os.path.exists(tica_extremes_dir): os.makedirs(tica_extremes_dir)

	tica_coords = verboseload(tica_projected_coords_dir)
	tica_concatenated = np.concatenate(tica_coords)
	print("np shape of tica_concatenated is")
	print((np.shape(tica_concatenated)))
	del(tica_coords)

	standardized_features = load_file_list(get_trajectory_files(standardized_features_dir, ext))
	standardized_features_concatenated = np.concatenate(standardized_features)
	print("np shape of standardized_features_concatenated is ")
	print((np.shape(standardized_features_concatenated)))
	del(standardized_features)

	features = load_file_list(get_trajectory_files(features_dir, ext))
	features_concatenated = np.concatenate(features)
	del(features)
	print("np shape of features_concatenated is ")
	print((np.shape(features_concatenated)))

	print("Loaded standardized features, finding extrema now")

	for j in range(0,np.shape(tica_concatenated)[1]):
		print(("Analyzing tIC%d" %j))
		tIC = tica_concatenated[:,j]
		print((np.shape(tIC)))
		low_percentile = np.percentile(tIC, percentile)
		high_percentile = np.percentile(tIC, (100. - percentile))
		low_indices = np.where(tIC < low_percentile)[0]
		high_indices = np.where(tIC > high_percentile)[0]
		low_features = standardized_features_concatenated[low_indices,:]
		print(low_features)
		print((np.shape(low_features)))
		high_features = standardized_features_concatenated[high_indices,:]
		np.savetxt("%s/tIC.%d_standardized_low_values.csv" %(tica_extremes_dir, (j+1)), low_features,  delimiter=",")
		np.savetxt("%s/tIC.%d_standardized_high_values.csv" %(tica_extremes_dir, (j+1)),high_features,  delimiter=",")
		
		low_features = features_concatenated[low_indices,:]
		high_features = features_concatenated[high_indices,:]
		np.savetxt("%s/tIC.%d_low_values.csv" %(tica_extremes_dir, (j+1)),low_features,  delimiter=",")
		np.savetxt("%s/tIC.%d_high_values.csv" %(tica_extremes_dir, (j+1)) ,high_features,  delimiter=",")

	print("Finished finding extreme values")



#Saves coefficients in each tIC for each feature (i.e., pair of residues)
def find_most_important_residues_in_tIC(traj_file, tica_object, tic_features_csv, contact_residues,tic_residue_csv, feature_coefs_csv, duplicated_feature_coefs_csv, cutoff):
	try:
		tica = verboseload(tica_object)
	except:
		tica = load_dataset(tica_object)
	print(traj_file)
	traj = md.load_frame(traj_file, 0)
	#traj = fix_traj(traj)
	top = traj.topology 
	#residue_pairs = compute_contacts_below_cutoff([traj_file, [0]], cutoff = cutoff, contact_residues = contact_residues, anton = True)
	residue_pairs = generate_features(tic_features_csv)
	new_residue_pairs = []
	for pair in residue_pairs:
		try:
			new_residue_pairs.append(("%s%d.%d" %(pair.residue_i[2], pair.residue_i[1], pair.residue_i[0])), ("%s%d.%d" %(pair.residue_j[2], pair.residue_j[1], pair.residue_j[0])))
		except:
			new_residue_pairs.append(pair)
	residue_pairs = new_residue_pairs
	#print traj_file

	
	top_indices_per_tIC = {}
	feature_coefs_per_tIC = {}
	duplicated_feature_coefs_per_tIC = {}


	#for each tIC:
		#for each feature, get the absolute component value
		#add to feature_coefs_per_tIC dictionary the absolute coefficient for that tIC
		#duplicate them for the analysis where we look at residues individually
		#sort by absolute coefficient value

	#for each tIC:
		#

	for i in range(0, np.shape(tica.components_)[0]):
		print(i)
		index_components = [(j,abs(tica.components_[i][j])) for j in range(0,np.shape(tica.components_)[1])]
		feature_coefs_per_tIC[i] = [component[1] for component in index_components]
		duplicated_feature_coefs_per_tIC[i] = [j for k in feature_coefs_per_tIC[i] for j in (k, k)] 
		index_components = sorted(index_components, key= lambda x: x[1],reverse=True)
		print((index_components[0:10]))
		list_i = [index_components[j][0] for j in range(0,len(index_components))]
		top_indices_per_tIC[i] = list_i
	
	top_residues_per_tIC = {}
	for i in range(0, np.shape(tica.components_)[0]):
		top_residues_per_tIC[i] = []
		for index in top_indices_per_tIC[i]:
			residues = residue_pairs[index]
			top_residues_per_tIC[i].append(residues)
		top_residues_per_tIC[i] = [item for sublist in top_residues_per_tIC[i] for item in sublist]

	residue_list = residue_pairs

	feature_coefs_per_tIC["residues_0"] = [pair.residue_i for pair in residue_list]
	feature_coefs_per_tIC["residues_1"] = [pair.residue_j for pair in residue_list]
	duplicated_feature_coefs_per_tIC["residues"] = [residue for residue_pair in residue_list for residue in residue_pair]

	write_map_to_csv(tic_residue_csv, top_residues_per_tIC, [])
	write_map_to_csv(feature_coefs_csv, feature_coefs_per_tIC, [])
	write_map_to_csv(duplicated_feature_coefs_csv, duplicated_feature_coefs_per_tIC, [])
	return
	#print(top_residues_per_tIC)

def save_features_to_residues_map(traj_file, residue_pairs, feature_residues_csv, cutoff, residues_map = None, exacycle = False):
	if residues_map is not None:
		contact_residues = [r for r in contact_residues if r in list(residues_map.keys())]
		if exacycle: contact_residues = [residues_map[key] for key in contact_residues]

	traj = md.load_frame(traj_file, 0)
	#traj = fix_traj(traj)
	top = traj.topology 
	residue_pairs, residue_infos = compute_contacts_below_cutoff([traj_file, [0]], cutoff = cutoff, contact_residues = contact_residues, anton = False)
	if exacycle:
		reverse_residues_map = {v: k for k, v in list(residues_map.items())}
		new_residue_pairs = []
		for residue_pair in residue_pairs:
			new_residue_pair = [reverse_residues_map[residue_pair.residue_i], reverse_residues_map[residue_pair.residue_j]]
			new_residue_pairs.append(new_residue_pair)
		residue_pairs = new_residue_pairs

		new_reisdue_infos = []
		for residue_info in residue_infos:
			new_residue_info = [(reverse_residues_map[residue_info[0][0]], residue_info[0][1], residue_info[0][2]), (reverse_residues_map[residue_info[1][0]], residue_info[1][1], residue_info[1][2])]
			new_residue_infos.append(new_residue_info)
		residue_infos = new_reisdue_infos

	print(("There are: %d residue pairs" %len(residue_pairs)))
	f = open(feature_residues_csv, "wb")
	f.write("feature, residue.1.resSeq, residue.2.resSeq, residue.1.res, residue.1.chain, residue.2.res, residue.2.chain,\n")
	for i in range(0, len(residue_infos)):
		f.write("%d, %d, %d, %d, %d, %d, %d,\n" %(i, residue_infos[i][0][0], residue_infos[i][1][0], residue_infos[i][0][1], residue_infos[i][0][2], residue_infos[i][1][1], residue_infos[i][1][2]))
	f.close()
	return 

