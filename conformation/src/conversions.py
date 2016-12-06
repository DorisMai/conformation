
def convert_mae_to_pdb(structures):
	pdb = structures[0]
	mae = structures[1]
	bashCommand = "$SCHRODINGER/utilities/structconvert -ipdb %s -omae %s" %(pdb, mae)	
	subprocess.call(bashCommand, shell = True)

def convert_mae_to_pdbs(traj_dir):
	pdbs = get_trajectory_files(traj_dir, ext = ".pdb")
	maes = []
	structures = []

	for pdb in pdbs:
		mae = pdb.rsplit(".", 1)[0]
		mae = "%s.mae" %mae
		structures.append((pdb, mae))
	
	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	pool.map(convert_mae_to_pdb, structures)
	pool.terminate()
	time.sleep(10)