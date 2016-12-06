import os 
from io_functions import *
import multiprocessing as mp
from PDB_Order_Fixer import PDB_Order_Fixer

def remove_ter_individual(pdb_file):
	print("removing ter lines in %s" %pdb_file)
	pdb = file(pdb_file, "rb")
	lines = pdb.readlines()
	new_pdb = file(pdb_file, "wb")
	for line in lines: 
		if "TER" in line:
			continue
		else:
			new_pdb.write(line)
	new_pdb.close()
def remove_ter(pdb_dir):
	pdb_files = get_trajectory_files(pdb_dir)
	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	pool.map(remove_ter_individual, pdb_files)
	pool.terminate()

def reorder_individual(pdb_file):
	print("reordering %s " %pdb_file)
	fixer = PDB_Order_Fixer(pdb_file, pdb_file)
	fixer.fix_pdb()

def reorder(pdb_dir):
	pdb_files = get_trajectory_files(pdb_dir, ext = ".pdb")
	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	pool.map(reorder_individual, pdb_files)
	pool.terminate()


def alias_pdb(filename, condition):
	aliases = [("ASH", "ASP"), ("GLH", "GLU"), ("HIP", "HIS"), ("HIE", "HIS"), ("HID", "HIS"), ("HSE", "HIS"), ("HSD", "HIS"), ("CYP", "CYS")]
	new_pdb = "/scratch/users/enf/b2ar_analysis/renamed_topologies/%s.pdb" %condition
	if(os.path.exists(new_pdb)):
		top_file = new_pdb
	else:
		old_file = open(filename, "rb")
		new_file = open(new_pdb, "wb")
		lines = old_file.readlines()
		for line in lines:
			new_line = copy.deepcopy(line)
			for alias in aliases:
				new_line = new_line.replace(alias[0], alias[1])
			new_file.write(new_line)
		new_file.close()
	return new_pdb


def remove_palm(traj_dir):
	pdb_files = get_trajectory_files(traj_dir)

	for pdb_file in pdb_files:
		print(pdb_file)
		top = md.load(pdb_file).topology
		indices = [a.index for a in top.atoms]
		pdb = md.load(pdb_file, atom_indices = indices)
		pdb.save_pdb(pdb_file)


def pymol_fixpdb(pdb_dir, script_dir):
	script = open(script_dir, "rb")
	lines = script.readlines()

	new_script = open(script_dir, "wb")

	for line in lines:
		if line[0:7] == "pdb_dir": 
			print ("found pdb line")
			line = "pdb_dir = '%s'\n" %pdb_dir
		new_script.write(line)

	new_script.close()
	command = "/scratch/users/enf/pymol/pymol %s" %script_dir
	print(command)
	os.system(command)	



def reorder_pdbs(pdb_dir):
	new_dir = "%s_reordered" %pdb_dir
	if not os.path.exists(new_dir): os.makedirs(new_dir)
	pdbs = get_trajectory_files(pdb_dir)
	for pdb in pdbs: 
		name = pdb.split("/")[len(pdb.split("/"))-1]
		new_name = "%s/%s" %(new_dir, name)
		pdb_fixer = PDB_Order_Fixer(pdb, new_name)
		pdb_fixer.fix_pdb()