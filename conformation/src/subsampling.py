import mdtraj as md
from io_functions import *
import subprocess
import multiprocessing as mp
import os
from glob import glob
from functools import partial 

def reimage(traj_file):
	print("Examining %s" %traj_file)
	traj = md.load(traj_file)
	#traj = fix_traj(traj)
	top = traj.topology
	traj_name = traj_file.split(".h5")[0]
	traj_last_name = traj_file.split("/")[len(traj_file.split("/"))-1]
	print(traj_name)
	directory = "/".join(traj_file.split("/")[0:len(traj_file.split("/"))-2])
	parent_directory = "/".join(traj_file.split("/")[0:len(traj_file.split("/"))-3])
	new_directory = "%s/subsampled_reimaged_amber" %directory
	traj.save_dcd("%s.dcd" %traj_name)
	traj[0].save_pdb("%s.pdb" %traj_name)
	del(traj)
	del(top)
	f = open("%s.in" %traj_name, 'w')
	f.write("parm %s\n" %("%s.pdb" %traj_name))
	f.write("trajin %s\n" %("%s.dcd" %traj_name))
	f.write("autoimage\n")
	f.write("trajout %s\n" %("%s_reimaged.dcd" %traj_name))
	f.write("trajout %s onlyframes 1\n" %("%s_reimaged.pdb" %traj_name))
	f.write("go")
	f.close()
	os.chdir(directory)
	os.system("ml load amber/14-intel")
	subprocess.call("cpptraj %s %s.in" %("%s.pdb" %traj_name, traj_name), shell=True)
	print("loading reimaged trajectory")
	traj = md.load("%s_reimaged.dcd" %traj_name, top = "%s_reimaged.pdb" %traj_name)
	traj.save("%s/%s" %(new_directory, traj_last_name))


def reimage_amber(traj_dir):
	trajs = get_trajectory_files(traj_dir, ext = ".h5")
	trajs_to_reimage = trajs
	#for traj in trajs:
	#	if "reimage" not in traj: trajs_to_reimage.append(traj)
	print(trajs_to_reimage)
	pool = mp.Pool(mp.cpu_count())
	#for traj in trajs:
	#	reimage(traj)
	pool.map(reimage, trajs_to_reimage)
	pool.close()
	#map(reimage, trajs)

def subsample_traj(traj, stride=5, top=None):
	directory = traj.split("/")
	simulation = directory[len(directory)-2]
	dcd_file = directory[len(directory)-1]
	condition = "%s-%s" %(simulation.split('-')[1], simulation.split('-')[2])
	print("analyzing simulation %s file %s" %(simulation, dcd_file))
	top_file = top

	top = md.load_frame(traj, 0, top=top_file).topology
	atom_indices = [a.index for a in top.atoms if str(a.residue)[0:3] != "POP" and not a.residue.is_water and str(a.residue)[0:2] != "NA" and str(a.residue)[0:2] != "CL"]

	traj = md.load(traj, stride=stride, top=top_file, atom_indices=atom_indices)
	print("traj loaded")

	new_file = "%s_stride%d.h5" %(dcd_file.rsplit( ".", 1 )[ 0 ] , stride)
	new_root_dir = "/scratch/users/enf/b2ar_analysis/subsampled_allprot"

	

	new_condition_dir = "%s/%s" %(new_root_dir, condition)

	new_file_full = "%s/%s/%s" %(new_root_dir, condition, new_file)
	print("saving trajectory as %s" %new_file_full)
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

def combine_subsample_copy(traj_directory, new_directory, topology, stem_directory, new_prefix=""):
	head = os.path.basename(traj_directory.strip("/"))
	print(head)
	new_filename = "%s/%s_%s.nc" %(new_directory, new_prefix, head)
	new_h5_filename = "%s/%s_%s.h5" %(new_directory, new_prefix, head)
	if os.path.exists(new_filename): return

	subprocess.call("cd %s" %traj_directory, shell=True)
	os.chdir(traj_directory)
	subprocess.call("cp /home/enf/xstream_scratch/md_simulations/input_files/auto_reimage_vsp.in %s" % traj_directory, shell=True)
	print("Finished copying auto reimager")
	#subprocess.call("cp %s/system.prmtop ./" % stem_directory, shell=True)
	subprocess.call("bash auto_reimage_vsp.in", shell=True)
	print("finished reimaging")
	files = glob('1_thru_*_skip_10_reimaged.nc')
	max_nc = [path.split("_")[2] for path in files]
	latest_traj = files[np.array(max_nc).argmin()]
	
	traj = md.load(latest_traj, top=topology)
	print("loaded traj")
	traj.save(new_h5_filename)
	print("saved h5 traj")
	subprocess.call("cp %s %s" %(latest_traj, new_filename), shell=True)
	print("finished copying traj")

	return

def subsample_amber(stem_directory, new_directory, new_prefix=""):
	if not os.path.exists(new_directory): os.makedirs(new_directory)
	
	paths = glob("%s/rep_*/" % stem_directory)
	print(paths)
	topology = "%s/system.pdb" % stem_directory
	combine_subsample_copy_partial = partial(combine_subsample_copy, new_directory=new_directory, topology = topology, stem_directory = stem_directory, new_prefix=new_prefix)
	#for path in paths:
	#	print("path to reimage")
	#	print(path)
	#	combine_subsample_copy_partial(path)
	pool = mp.Pool(mp.cpu_count()/4)
	pool.map(combine_subsample_copy_partial, paths)
	pool.terminate()
	return 



