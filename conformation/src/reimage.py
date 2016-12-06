import mdtraj as md
import os
import h5py
import multiprocessing as mp
from functools import partial 
import pytraj.io as mdio

def get_trajectory_files(traj_dir, ext = ".h5"):
	traj_files = []
	for traj in os.listdir(traj_dir):
			if traj.endswith(ext):
				traj_files.append("%s/%s" %(traj_dir,traj))
	return sorted(traj_files)

def reimage_traj(traj_file, save_dir):
	traj = md.load(traj_file)
	topology = md.load_frame(traj_file,index=0)

	traj_pytraj = mdio.load_mdtraj(traj)
	traj_pytraj.autoimage()
	traj.xyz[:] = traj_pytraj.xyz / 10.
	filename = traj_file.split("/")[len(traj_file.split("/"))-1]
	filename = filename.split(".")[0]
	h5_filename = "%s.h5" %filename
	new_h5_file = "%s/%s" %(save_dir, h5_filename)
	print(new_h5_file)
	traj.save(new_h5_file)
	return

def reimage_trajs(traj_dir):
	new_dir = "%s_reimaged" %traj_dir

	if not os.path.exists(new_dir): os.makedirs(new_dir)

	trajs = get_trajectory_files(traj_dir)
	
	reimage = partial(reimage_traj, save_dir = new_dir)

	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers / 2)
	pool.map(reimage, trajs)
	pool.terminate()
	return

traj_dir = "/scratch/users/enf/b2ar_analysis/subsampled_allprot_combined"
reimage_trajs(traj_dir)
print("done")
