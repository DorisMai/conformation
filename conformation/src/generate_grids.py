import os

def get_trajectory_files(traj_dir):
	traj_files = []
	for traj in os.listdir(traj_dir):
			if traj.endswith(".pdb"):
				traj_files.append("%s/%s" %(traj_dir,traj))
	return sorted(traj_files)

def convert_to_mae(pdb_dir):
	pdbs = get_trajectory_files(pbd_dir)

	for pdb in pdbs:
		stem = pdb.rsplit( ".", 1 )[ 0 ]
		mae_name = "%s.mae" %stem
		if os.path.exists(mae_name):
			print "already converted"
			continue
		else:
			command = "$SCHRODINGER/utilities/structconvert -ipdb %s -omae %s" %(pdb, mae_name)

def generate_grid_inputs(pdb_dir, outerbox, innerbox):
	