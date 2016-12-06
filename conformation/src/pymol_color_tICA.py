import __main__
#__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI

import pymol
from pymol import cmd
import os
from optparse import OptionParser
import optparse
import sys
import multiprocessing as mp
from functools import partial
import csv
import time
sys.path.append("/Users/Evan/vsp/b2ar_analysis/conformation")
from tica_variables import *
import numpy as np

#pymol.finish_launching()

def get_trajectory_files(traj_dir, ext = ".pdb"):
	traj_files = []
	for traj in os.listdir(traj_dir):
			if traj.endswith(ext):
				traj_files.append("%s/%s" %(traj_dir,traj))
	return sorted(traj_files)

def change_color(object_name, residue_importances, ginis):
	cmd.alter("%s" %(object_name), "b=0.0")
	for resid in residue_importances.keys():
		print(resid)
		cmd.alter("resid %d" %resid, "b=%f" %(residue_importances[resid]))
	cmd.spectrum("b", "blue green red", object_name,min(ginis), max(ginis))

def interpret_tIC(pdb_dir, pdb_list_file, ref_dir, importance_file, save_dir, tic_j, load = True, method = "mean"):
	#cmd.feedback("disable","all","actions")
	#cmd.feedback("disable","all","results")
	#cmd.set('suspend_updates', 'on')
	n_clusters = 100
	lag_time = 100

	pdbs = []
	if pdb_list_file is None:
		#pdbs = get_trajectory_files(pdb_dir)
		pdbs = []
	else:
		with open(pdb_list_file, 'rb') as f:
			reader = csv.reader(f)
			pdbs = list(reader)
	#pdbs = [ref_dir]

	if(load):
		cmd.load(ref_dir[0], "inactive_ref")
		cmd.load(ref_dir[1], "active_ref")
		cmd.align("active_ref", "inactive_ref")

	with open(importance_file, 'rb') as f:
		reader = csv.reader(f)
		importances = list(reader)

	residue_importances = {}
	ginis = []
	if method == "mean":
		for line in importances:
			if 'R' in line[0]:
				resid0 = int(line[0].split('_')[1])
				imp0 = float(line[1])
				resid1 = int(line[0].split('_')[2])
				imp1 = float(line[1])
				if resid0 in iter(residue_importances.keys()):
					residue_importances[resid0].append(imp0)
				else:
					residue_importances[resid0] = [imp0]

				if resid1 in iter(residue_importances.keys()):
					residue_importances[resid1].append(imp1)
				else:
					residue_importances[resid1] = [imp1]
		for key in residue_importances.keys():
			residue_importances[key] = np.percentile(np.array(residue_importances[key]),99.99)
			ginis.append(residue_importances[key])

	else:
		for line in importances:
			if 'R' in line[0]:
				resid0 = int(line[0].split('_')[1])
				imp0 = float(line[1])
				resid1 = int(line[0].split('_')[2])
				imp1 = float(line[1])
				if resid0 in iter(residue_importances.keys()):
					if imp0 > residue_importances[resid0]: residue_importances[resid0] = imp0
				else:
					residue_importances[resid0] = imp0

				if resid1 in iter(residue_importances.keys()):
					if imp1 > residue_importances[resid1]: residue_importances[resid1] = imp1
				else:
					residue_importances[resid1] = imp1
				ginis.append(imp0)


	for i in range(0,len(pdbs)):
		print(i) 
		if(pdb_list_file is not None):
			if i == 0: continue
			pdb_file = "%s/%s.pdb" %(pdb_dir,pdbs[i][2])
			pdb_name = "%s_rep%d" %(pdbs[i][1], i%5)
			pdb_name = pdb_name.replace(" ", "")
			pdb_name = pdb_name.replace(",", "_")
			pdb_name = pdb_name.replace(">", "_gr_")
			pdb_name = pdb_name.replace("<", "_le_")
			print(pdb_name)
			cmd.load(pdb_file, pdb_name)
		else:
			pdb_file = pdbs[i]
			pdb_name = pdb_file.split("/")[len(pdb_file.split("/"))-1]
			pdb_name = pdb_name.split(".")[0]
			cmd.load(pdb_file, pdb_name)
		cmd.align(pdb_name, "inactive_ref")
		change_color(pdb_name, residue_importances, ginis)

	change_color("inactive_ref", residue_importances, ginis)
	change_color("active_ref", residue_importances, ginis)


	print(save_dir)
	print(tic_j)
	#cmd.deselect()
	#cmd.hide("all")
	#cmd.cartoon("loop")
	#cmd.orient("inactive_ref")
	cmd.save("%s/tIC%d_interpretation.pse" %(save_dir, tic_j))
	#
		#new_file.write("%s;%f\n" %(pdb_name, rmsd[0]))
		#rmsds.append(rmsd[0])
		#cmd.delete(str(i))
	#time.sleep(100)	

#pdb_dir = "/Users/Evan/vsp/b2ar_analysis/tICA_t10_n_components5_switches_npxx_tm6_bp/clusters1000_n_components5_n_samples10_reimaged"
#ref_dir = "/Users/Evan/vsp/b2ar_analysis/3P0G_pymol_prepped.pdb"
#rmsd_dir = '/scratch/users/enf/b2ar_analysis/tICA_t10_n_components5_switches_npxx_tm6_bp/clusters1000_n_components5_n_samples10_reimaged/active_rmsds.csv'
#importance_file = "/Users/Evan/vsp/b2ar_analysis/tICA_t5_n_components25reimaged_notrajfix_tm_residues_under_cutoff1nm_regularization0pt5/analysis_n_clusters1000_random/tIC7rf_importance.csv"

print("HI")

simulation_ref_dir = "/Users/Evan/vsp/b2ar_analysis/A-00_protein_BIA.pdb"
dock_pose = "/Users/Evan/vsp/b2ar_analysis/tICA_t5_n_components25reimaged_notrajfix_tm_residues_under_cutoff1nm_regularization0pt5/docking_n_clusters1000_n_samples10_random_SP/3p0g_lig/cluster507_sample3_pv.maegz"
#interpret_tIC(save_dir, tica_samples_csv, [inactive_ref_dir, simulation_ref_dir], "%s/tIC7rf.csv" %analysis_dir, analysis_dir, 7)
interpret_tIC(save_dir, None, [inactive_ref_dir, simulation_ref_dir, dock_pose], "%s/tIC7rf.csv" %analysis_dir, analysis_dir, 7, load=True, method="mean")
#calc_rmsds(pdb_dir, ref_dir, importance_file)