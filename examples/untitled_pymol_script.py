import pymol
import os

def get_trajectory_files(traj_dir):
	traj_files = []
	for traj in os.listdir(traj_dir):
			if traj.endswith(".dcd") or traj.endswith(".h5") or traj.endswith(".pdb"):
				traj_files.append("%s/%s" %(traj_dir,traj))
	return sorted(traj_files)

n_clusters = 100
lag_time = 100

traj_dir = "/Users/evan/scratch_enf/b2ar_analysis/%d_clusters_t%d_reimaged" %(n_clusters, lag_time)
new_dir = "/Users/evan/scratch_enf/b2ar_analysis/%d_clusters_t%d_reimaged_pymol" %(n_clusters, lag_time)
if not os.path.exists(new_dir): os.makedirs(new_dir)

trajs = get_trajectory_files(traj_dir)

cmd.load("/Users/evan/scratch_enf/b2ar_analysis/3P0G_pymol_prepped.pdb", "3P0G")

for i in range(0,len(trajs)):
	print(i)
	filename = trajs[i]
	print(filename)
	cmd.load(filename, str(i))
	cmd.remove("resn SOD")
	cmd.remove("resn CLA")
	cmd.remove("hydro")
	cmd.h_add()
	cmd.align(str(i), "3P0G")
	cmd.set("pdb_conect_all", "on")
	new_filename = "%s/%d.pdb" %(new_dir, i)
	cmd.save(new_filename, str(i))
	cmd.delete(str(i))

