
def read_and_featurize_iter(traj_file, features_dir = None, condition=None, dihedral_types = ["phi", "psi", "chi1", "chi2"], dihedral_residues = None, resSeq_pairs = None):

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

	if len(resSeq_pairs) > 0:
		top = md.load_frame(traj_file, index=0).topology
		resIndex_pairs = convert_resSeq_to_resIndex(top, resSeq_pairs)
		contact_features = []
		try:
			for chunk in md.iterload(traj_file, chunk = 1000):
			#	chunk = fix_traj(chunk)
			#chunk = md.load(traj_file,stride=1000)
			#print(resIndex_pairs[0:10])
				chunk_features = md.compute_contacts(chunk, contacts = resIndex_pairs, scheme = 'closest-heavy', ignore_nonprotein=False)[0]
				print((np.shape(chunk_features)))
				contact_features.append(chunk_features)
			contact_features = np.concatenate(contact_features)
		except Exception as e:
			print(str(e))
			print("Failed")
			return
			#traj = md.load(traj_file)
			#contact_features = md.compute_contacts(chunk, contacts = contact_residue_pairs, scheme = 'closest-heavy', ignore_nonprotein=False)[0]

		if len(dihedral_residues) > 0: 
			manual_features = np.column_stack((manual_features, contact_features))
		else:
			manual_features = contact_features


	b = time.time()

	print(("new features %s has shape: " %traj_file))
	print((np.shape(manual_features)))

	if condition is None:
		condition = traj_file.split("/")[len(traj_file.split("/"))-1]
		condition = condition.split(".")[0]

	save_dataset(manual_features, "%s/%s.dataset" %(features_dir, condition))

def compute_contacts_below_cutoff(traj_file_frame, cutoff = 100000.0, contact_residues = [], anton = False):
	traj_file = traj_file_frame[0]
	frame = md.load_frame(traj_file, index = 0)
	#frame = fix_traj(frame)
	top = frame.topology
	
	distance_residues = []
	res_indices = []
	resSeq_to_resIndex = {}

	for i in range(0, len(contact_residues)):
		residue = contact_residues[i]
		indices = [r.index for r in top.residues if r.resSeq == residue and not r.is_water]
		if len(indices) == 0:
			print(("No residues in trajectory for residue %d" %residue))
			continue
		else:
			ind = indices[0]
			for j in indices:
				if j != ind: 
					#print("Warning: multiple res objects for residue %d " %residue)
					if "CB" in [str(a) for a in r.atoms for r in top.residues if r.index == ind]:
						ind = j
			res_indices.append(ind)
			distance_residues.append(residue)
			resSeq_to_resIndex[residue] = ind
	
	resSeq_combinations = itertools.combinations(distance_residues, 2)
	res_index_combinations = []
	resSeq_pairs = [c for c in resSeq_combinations]
	for combination in resSeq_pairs:
		res0 = combination[0]
		res1 = combination[1]
		res_index0 = resSeq_to_resIndex[res0]
		res_index1 = resSeq_to_resIndex[res1]
		res_index_combinations.append((res_index0, res_index1))


	final_resSeq_pairs = []
	final_resIndex_pairs = []

	distances = md.compute_contacts(frame, contacts = res_index_combinations, scheme = 'closest-heavy', ignore_nonprotein=False)[0]
	#print(distances)
	print((np.shape(distances)))
	for i in range(0, len(distances[0])):
		distance = distances[0][i]
		#print(distance)
		if distance < cutoff:
			final_resIndex_pairs.append(res_index_combinations[i])
			final_resSeq_pairs.append(resSeq_pairs[i])

	print((len(final_resSeq_pairs)))
	print((len(final_resIndex_pairs)))
	return(final_resSeq_pairs)



def featurize_custom_anton(traj_dir, features_dir, traj_ext, featurized_traj_and_frame, contact_residue_pairs_csv, dihedral_residues = None, dihedral_types = None, contact_residues = None, agonist_bound = False, residues_map = None, contact_cutoff = None):
	if not os.path.exists(features_dir): os.makedirs(features_dir)

	all_trajs = get_trajectory_files(traj_dir, traj_ext)
	trajs = []
	for fulltraj in all_trajs:
		#if "H-05" not in fulltraj and "A-00" not in fulltraj: continue
		traj = fulltraj.split("/")
		filename = traj[len(traj)-1]
		#if agonist_bound is not False and filename[0] not in agonist_bound: continue
		filename_noext = filename.split(".")[0]
		if os.path.exists("%s/%s.dataset" %(features_dir, filename_noext)):
			print("already featurized")	
		else:
			trajs.append(fulltraj)

	pool = mp.Pool(mp.cpu_count()/4)

	if residues_map is not None:
		contact_residues = [r for r in contact_residues if r in list(residues_map.keys())]
		dihedral_residues = [d for d in dihedral_residues if d in list(residues_map.keys())]		

	#print contact_residues	
	

	print(featurized_traj_and_frame)
	if featurized_traj_and_frame is None: featurized_traj_and_frame = [all_trajs[0], 0]
	if contact_residue_pairs_csv is None: 
		contact_residue_pairs = compute_contacts_below_cutoff(featurized_traj_and_frame, cutoff = contact_cutoff, contact_residues = contact_residues, anton = False)
	else:
		contact_residue_pairs = generate_features(contact_residue_pairs_csv)
	print(("Number of contact pairs = %d" %len(contact_residue_pairs)))


	featurize_partial = partial(read_and_featurize_iter, features_dir = features_dir, dihedral_residues = dihedral_residues, dihedral_types = dihedral_types, resSeq_pairs = contact_residue_pairs)
	#pool.map(featurize_partial, trajs)
	#pool.terminate()
	for traj in trajs:
		print(traj)
		featurize_partial(traj)


	print("Completed featurizing")
