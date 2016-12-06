import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI

import pymol
import os
from optparse import OptionParser
import optparse
import sys

pymol.finish_launching()

def get_trajectory_files(traj_dir):
	traj_files = []
	for traj in os.listdir(traj_dir):
			if traj.endswith(".pdb"):
				traj_files.append("%s/%s" %(traj_dir,traj))
	return sorted(traj_files)

def calc_rmsds(pdb_dir, ref_dir, rmsd_file):
	n_clusters = 100
	lag_time = 100

	pdbs = get_trajectory_files(pdb_dir)

	cmd.load(ref_dir, "ref")

	new_file = open(rmsd_file, "wb")

	for i in range(0,len(pdbs)):
		print(i) 
		pdb_file = pdbs[i]
		pdb_name = pdb_file.split("/")[len(pdb_file.split("/"))-1]
		cmd.load(pdb_file, str(i))
		rmsd = cmd.align(str(i), "ref")
		print(rmsd[0])
		new_file.write("%s;%f\n" %(pdb_name, rmsd[0]))
		cmd.delete(str(i))
	new_file.close()

parser = OptionParser()
parser.add_option("-p", "--pdb_dir", type = "string", help = "dir to pdbs")
parser.add_option("-r", "--ref_dir", type = "string", help = "dir to reference pdb")
parser.add_option("-f", "--rmsd_dir", type = "string", help = "rmsd file")
print("HELLO")
opts, args = parser.parse_args()
pdb_dir = opts.pdb_dir
ref_dir = opts.ref_dir
rmsd_dir = opts.rmsd_dir

calc_rmsds(pdb_dir, ref_dir, rmsd_dir)
pymol.cmd.quit()