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

def calc_rmsds(pdb_dir, ref_dir, rmsd_file):
	cmd.feedback("disable","all","actions")
	cmd.feedback("disable","all","results")
	cmd.set('suspend_updates', 'on')
	n_clusters = 100
	lag_time = 100

	pdbs = get_trajectory_files(pdb_dir)

	cmd.load(ref_dir, "ref")

	new_file = open(rmsd_file, "wb")
	rmsds = []
	for i in range(0,len(pdbs)):
		print(i) 
		pdb_file = pdbs[i]
		pdb_name = pdb_file.split("/")[len(pdb_file.split("/"))-1]
		cmd.load(pdb_file, str(i))
		rmsd = cmd.align(str(i), "ref")
		print(rmsd[0])
		new_file.write("%s;%f\n" %(pdb_name, rmsd[0]))
		rmsds.append(rmsd[0])
		cmd.delete(str(i))
	new_file.close()
	return rmsds

def calc_rmsds_parallel(pdb_dir, ref_dir, rmsd_file):

	def calc_rmsd(pdb_file, ref_file):
		pdb_name = pdb_file.split("/")[len(pdb_file.split("/"))-1]
		cmd.load(pdb_file, pdb_name)
		rmsd = cmd.align(pdb_name, "ref")
		print(rmsd[0])
		cmd.delete(pdb_name)
		return rmsd[0]

	calc_rmsd_partial = partial(calc_rmsd, ref_file = ref_dir)

	pdbs = get_trajectory_files(pdb_dir)
	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	rmsds = pool.map(calc_rmsd_partial, pdbs)
	print("HELLO!")
	pool.terminate()

	new_file = open(rmsd_file, "wb")

	for i in range(0, len(pdbs)):
		print(i)
		rmsd = rmsds[i]
		pdb_file = pdbs[i]
		pdb_name = pdb_file.split("/")[len(pdb_file.split("/"))-1]
		new_file.write("%s, %f \n" %(pdb_name, rmsd))

	print("HELLO?")
	new_file.close()
	return

pdb_dir = '/scratch/users/enf/b2ar_analysis/tICA_t10_n_components5_switches_npxx_tm6_bp/clusters1000_n_components5_n_samples10_reimaged'
ref_dir = '/scratch/users/enf/b2ar_analysis/3P0G_pymol_prepped.pdb'
rmsd_dir = '/scratch/users/enf/b2ar_analysis/tICA_t10_n_components5_switches_npxx_tm6_bp/clusters1000_n_components5_n_samples10_reimaged/active_rmsds.csv'

calc_rmsds(pdb_dir, ref_dir, rmsd_dir)
pymol.cmd.quit()