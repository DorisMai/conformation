

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