import os
from glob import glob
import subprocess

def reimage(stem_directory, trajectory, topology):
	current_dir = os.getcwd()
	os.chdir(stem_directory)
	trajectory_basename = os.path.basename(trajectory)
	trajectory_basename_noext = os.path.splitext(trajectory_basename)[0]
	prmtop_basename = os.path.basename(topology)
	subprocess.call("reimage_once.bash %s %s" %(trajectory_basename_noext, prmtop_basename), shell=True)
	os.chdir(current_dir)

	return

def subsample_amber(stem_directory, prmtop_file):	
	#stem_directory: your directory with the xtc files in it
	#prmtop_file: your topology file. a .prmtop but i guess it could be a .pdb if need be
	trajectories = glob("%s/*.xtc" %stem_directory)
	topology = prmtop_file

	for trajectory in trajectories:
		reimage(stem_directory, trajectory, topology)

	return 