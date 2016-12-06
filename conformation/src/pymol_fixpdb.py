import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI

import pymol
import os
from optparse import OptionParser
import optparse
import sys
import multiprocessing as mp
from functools import partial

pymol.finish_launching()

def get_trajectory_files(traj_dir):
	traj_files = []
	for traj in os.listdir(traj_dir):
			if traj.endswith(".pdb"):
				traj_files.append("%s/%s" %(traj_dir,traj))
	return sorted(traj_files)

def fix_pdb(pdb_dir):
	cmd.feedback("disable","all","actions")
	cmd.feedback("disable","all","results")
	cmd.set('suspend_updates', 'on')

	pdbs = get_trajectory_files(pdb_dir)

	for i in range(0,len(pdbs)):
		print(i) 
		pdb_file = pdbs[i]
		pdb_name = pdb_file.split("/")[len(pdb_file.split("/"))-1]
		cmd.load(pdb_file, str(i))
		cmd.save(pdb_file, str(i))
		cmd.set("pdb_conect_all", "on")
		cmd.delete(str(i))


pdb_dir = '/scratch/users/enf/b2ar_analysis/tICA_t10_n_components5_switches_npxx_tm6_bp/clusters1000_n_components5_n_samples10'

fix_pdb(pdb_dir)

pymol.cmd.quit()