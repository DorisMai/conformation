from PDB_Order_Fixer import PDB_Order_Fixer
import mdtraj as md
import os
import numpy as np
import h5py
import datetime
import multiprocessing as mp
import copy
import gc
from functools import partial 
import time
import fileinput
import subprocess
from subprocess import Popen
import sys
from io_functions import *

import signal
import zipfile

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

'''
If simulation was run under periodic boundary conditions, this will reimage the trajectory.
It takes as input the trajectory file (traj_file), the directory that that trajectory is in (traj_dir),
the directory to which you would like to save the new trajectory, and the extension (ext) of the file. 

In the docking pipeline, I do this to all PDB files containing receptors to which I would like to dock. 
You can skip this step if it already has been reimaged. 

It requires that Pytraj be installed. This can be very annoying to do. It was easy to install on 
Sherlock but not on Biox3. 

'''



'''
The following two functions take as input a directory containing PDB files containing structures to which you would like to dock.
It will prepare the protein with Schrodinger's tools (add hydrogens, SS bonds (no, not that SS!), bond orders, etc.) and then save
an .mae file, which is required for docking.
'''

def pprep_prot(pdb, ref, extension = ".mae"):
	pdb_noext = os.path.splitext(pdb)[0]
	mae_filename =  "%s.mae" % pdb_noext
	if os.path.exists(mae_filename): 
		print("already prepped and mae'd protein")
		return
	current_directory = os.getcwd()
	os.chdir(os.path.dirname(pdb))
	mae_filename = os.path.basename(mae_filename)
	command = "$SCHRODINGER/utilities/prepwizard -WAIT -disulfides -fix -noepik -noimpref -noprotassign -reference_st_file %s -NOLOCAL %s %s" %(ref, pdb, mae_filename)
	print(command)
	print(os.getcwd())
	subprocess.call(command, shell=True)
	os.chdir(current_directory)
	return

def remove_path_and_extension(directory):
	filename = directory.split("/")[len(directory.split("/"))-1]
	filename_no_ext = filename.split(".")[0]
	filename_no_pv = filename_no_ext.split("_pv")[0]
	return(filename_no_pv)

def pprep(pdb_dir, ref, indices = None, chosen_receptors = None, extension = ".mae", worker_pool=None, parallel=False):
	pdbs = get_trajectory_files(pdb_dir, ext = ".pdb")
	"""
	print((len(chosen_receptors)))
	print((len(pdbs)))
	if indices is not None:
		pdbs = pdbs[indices[0] : indices[1]]
	elif chosen_receptors is not None:
		print((remove_path_and_extension(pdbs[0])))
		pdbs = [pdb for pdb in pdbs if remove_path_and_extension(pdb) in chosen_receptors]
	print((len(pdbs)))
	os.chdir(pdb_dir)
	"""
	pprep_partial = partial(pprep_prot, ref = ref, extension = extension)
	print(len(pdbs))
	print(pdbs[0:3])
	if worker_pool is not None:
		worker_pool.map_sync(pprep_partial, pdbs)
	elif parallel:
		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		pool.map(pprep_partial, pdbs)
		pool.terminate()
	else:
		for pdb in pdbs:
			pprep_partial(pdb)
	print("Done prepping proteins")
	#time.sleep(10)


'''
The f ollowing two functions take as input the directory contianing the ligands you would like to dock to your receptors, 
and prepares them with Schrodinger LigPrep, and then saves them in .maegz format, required for the actual docking. 
You can change the settings listed in "ligfile.write" lines. Perhaps we should add this instead as optional inputs
in the function definition. 
'''

def prepare_ligand(lig, lig_dir, n_ring_conf, n_stereoisomers, force_field, verbose=True):
	os.chdir(lig_dir)
	lig_last_name = lig.split("/")[len(lig.split("/"))-1]
	lig_no_ext = lig_last_name.split(".")[0]
	lig_ext = lig_last_name.split(".")[1]

	lig_input = "%s/%s.inp" %(lig_dir, lig_no_ext)
	lig_output = "%s-out.maegz" %lig_no_ext
	#if os.path.exists("%s/%s_bmout-bad.maegz" %(lig_dir, lig_no_ext)):
	#	print("Already failed preparing ligand.")

	if os.path.exists("%s/%s" %(lig_dir,lig_output)):
		return

	if "mae" not in lig_ext:
		lig_mae = "%s.mae" %(lig_no_ext)
		intermediate_file = "%s/%s" %(lig_dir, lig_mae)
		cmd = "unset PYTHONPATH; $SCHRODINGER/utilities/sdconvert -isd %s -omae %s" %(lig, intermediate_file)
		subprocess.call(cmd, shell=True)

	ligfile = open(lig_input, "wb")
	ligfile.write("INPUT_FILE_NAME   %s \n" %lig_mae)
	ligfile.write("OUT_MAE   %s \n" %lig_output)
	ligfile.write("FORCE_FIELD   %d \n" %force_field)
	ligfile.write("PH   7.4 \n")
	ligfile.write("EPIK   yes \n")
	ligfile.write("DETERMINE_CHIRALITIES   no \n")
	ligfile.write("IGNORE_CHIRALITIES   no \n")
	ligfile.write("NUM_STEREOISOMERS   %d \n" %n_stereoisomers)
	ligfile.write("NUM_RING_CONF   %d \n" %n_ring_conf)
	ligfile.close()

	cmd = "unset PYTHONPATH; $SCHRODINGER/ligprep -WAIT -inp %s" %lig_input
	subprocess.call(cmd, shell=True)


def prepare_ligands(lig_dir, exts = [".mae"], n_ring_conf=1, n_stereoisomers=1, force_field=16, worker_pool=None, parallel=True, redo=False):
	ligs = []
	for ext in exts:
		ligs += get_trajectory_files(lig_dir, ext)

	print("Examining %d ligands" %len(ligs))

	"""
	if not redo: 
		unfinished_ligs = []
		for lig in ligs: 
			lig_no_ext = lig.split(".")[0]
			lig_mae = "%s-out.maegz" %lig_no_ext
			if not os.path.exists(lig_mae):
				unfinished_ligs.append(lig)
		ligs = unfinished_ligs

	print("Preparing %d ligands" %(len(ligs)))
	"""

	lig_partial = partial(prepare_ligand, lig_dir = lig_dir, n_ring_conf=n_ring_conf, n_stereoisomers=n_stereoisomers, force_field=force_field)
	if worker_pool is not None:
		worker_pool.map_sync(lig_partial, ligs)
	elif parallel:
		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		pool.map(lig_partial, ligs)
		pool.terminate()
	else:
		for lig in ligs:
			lig_partial(lig)
	print("finished preparing ligands")

'''
To dock, Schrodinger has to generate grid files (in .zip format) for each receptor. This needs as input the (x,y,z) coordinates 
for the center of the grid, and parameters for the size of the box surrounding that point in which Glide will try to dock your ligand(s).
ALl you need to do is pass to "generate_grids() the following: 
mae_dir, a directory containing mae files of the receptors to which you will dock 
grid_center, a *string* containing the x,y,z coords of the center of the grid, e.g: grid_center = "64.4, 16.9, 11.99"
grid_dir: the directory where you want Schrodinger to save the .zip grid files
remove_lig: if there is a co-crystallized or docked ligand already in your .mae files, you will need to remove it first. to automatically
do this, set remove_lig to the 3-letter upper case string residue name denoting that ligand. for B2AR PDB ID: 3P0G, I would pass: remove_lig = "BIA"

"
'''

def generate_grid_input(mae, grid_center, grid_dir, remove_lig = None, outer_box=25.):
	mae_name = mae.rsplit( ".", 1)[0]
	mae_last_name = mae_name.split("/")[len(mae_name.split("/"))-1]

	output_dir = grid_dir
	
	new_mae = "%s/%s.mae" %(output_dir, mae_last_name)

	grid_job = "%s/%s.in" %(output_dir, mae_last_name)
	grid_file = "%s/%s.zip" %(output_dir, mae_last_name)

	if (os.path.exists(grid_job) and os.path.exists(new_mae)) or (os.path.exists(grid_file)):
		print("Already created that grid job, skipping")
		return

	if remove_lig == None:
		if not os.path.exists(new_mae):
			cmd = "cp %s %s" %(mae, new_mae)
			print(cmd)
			subprocess.call(cmd, shell=True)
	else:
		cmd = "$SCHRODINGER/run $SCHRODINGER/mmshare-v3.3/python/common/delete_atoms.py -asl \"res.pt %s \" %s %s" %(remove_lig, mae, new_mae)
		print(cmd)
		subprocess.call(cmd, shell=True)

	gridfile = open(grid_job, "wb")
	gridfile.write("GRIDFILE   %s.zip \n" %mae_last_name)
	gridfile.write("OUTPUTDIR   %s \n" %output_dir)
	gridfile.write("GRID_CENTER   %s \n" %grid_center)
	gridfile.write("INNERBOX   10, \n")
	gridfile.write("OUTERBOX   %d, \n" %outer_box)
	gridfile.write("RECEP_FILE   %s \n" %new_mae)
	#gridfile.write("RECEP_VSCALE   1.0 \n")
	#gridfile.write("LIGAND_MOLECULE  1 \n")
	#gridfile.write("WRITEZIP   TRUE \n")
	gridfile.close()

def generate_grid(grid_file, grid_dir):
	grid_zip = grid_file.rsplit( ".", 1)[0]
	grid_zip = "%s.zip" %grid_zip
	if os.path.exists(grid_zip):
		print("already generated grid; skipping")
		return

	os.chdir(grid_dir)
	grid_command = "$SCHRODINGER/glide %s -OVERWRITE -WAIT" %grid_file

	subprocess.call(grid_command, shell = True)
	print("completed grid generation job")
	return 

def unzip(zip_file):
	output_folder = "/".join(zip_file.split("/")[0:len(zip_file.split("/"))-1])
	os.chdir(output_folder)
	zip_file_last_name = zip_file.split("/")[len(zip_file.split("/"))-1].split(".")[0]
	if os.path.exists("%s/%s.grd" %(output_folder, zip_file_last_name)): 
		print("Already unzipped grid files")
		return

	cmd = "unzip -u %s" %zip_file
	subprocess.call(cmd, shell = True)
	return

def unzip_file(filename_grid_dir):
	filename = filename_grid_dir[0]
	grid_dir = filename_grid_dir[1]
	gridname = filename.split("/")[len(filename.split("/"))-1]
	print("unzipping %s" %gridname)
	try:
		fh = open(filename, 'rb')
		z = zipfile.ZipFile(fh)
		for name in z.namelist():
		    outpath = grid_dir
		    z.extract(name, outpath)
		fh.close()
	except:
		unzip(filename)
	return

def unzip_receptors(grid_dir, receptors, worker_pool=None):
	print("Unzipping selected grid files")
	grids = ["%s/%s.zip" %(grid_dir,receptor) for receptor in receptors if not os.path.exists("%s/%s.grd" %(grid_dir, receptor))]
	print(grids)
	if worker_pool is not None:
		worker_pool.map_sync(unzip, grids)
	else:
		pool = mp.Pool(mp.cpu_count())
		pool.map(unzip, grids)
		pool.terminate()

	#filename_grid_dirs = [(grid, grid_dir) for grid in grids]
	#num_workers = mp.cpu_count()
	#pool = mp.Pool(num_workers)
	#pool.map(unzip_file, filename_grid_dirs)
	#pool.terminate()
	print("Finishing unzipping grid files")
	return


def generate_grids(mae_dir, grid_center, grid_dir, remove_lig = None, indices = None, chosen_receptors = None, outer_box=10., worker_pool=None):
	print(grid_dir)
	if not os.path.exists(grid_dir): os.makedirs(grid_dir)

	maes = get_trajectory_files(mae_dir, ".mae")
	if chosen_receptors is not None:
		maes = [mae for mae in maes if remove_path_and_extension(mae) in chosen_receptors]

	generate_grid_input_partial = partial(generate_grid_input, grid_dir = grid_dir, grid_center = grid_center, remove_lig = remove_lig, outer_box=outer_box)
	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	pool.map(generate_grid_input_partial, maes)
	pool.terminate()

	grid_files = get_trajectory_files(grid_dir, ".in")

	grid_partial = partial(generate_grid, grid_dir = grid_dir)

	if indices is not None:
		grid_files = grid_files[indices[0] : indices[1]]

	if worker_pool is not None:
		worker_pool.map_sync(grid_partial, grid_files)
	else:
		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		pool.map(grid_partial, grid_files)
		#for grid_file in grid_files:
		#	grid_partial(grid_file)
		pool.terminate() 

	#zips = get_trajectory_files(grid_dir, ".zip")
	#pool = mp.Pool(num_workers)
	#pool.map(unzip, zips)
	#pool.terminate()

	return grid_dir

'''
the function, dock_conformations() is to dock a single ligand to many conformations. you will probably  be using the function,
dock_ligands_and_receptors(), however.
'''

def run_command(cmd):
	subprocess.call(cmd, shell = True)

def dock(dock_job):
	signal.alarm(300)
	docking_dir = os.path.dirname(dock_job)
	os.chdir(docking_dir)
	cmd = "$SCHRODINGER/glide %s -OVERWRITE -WAIT -strict" %dock_job
	print(cmd)
	try:
		run_command(cmd)
		os.chdir("/home/enf/b2ar_analysis/conformation")
	except TimeoutException:
		print("Docking job timed out")
		os.chdir("/home/enf/b2ar_analysis/conformation")
		return
	else:
		os.chdir("/home/enf/b2ar_analysis/conformation")
		signal.alarm(0)
	return

def dock_conformations(grid_dir, docking_dir, ligand_dir, precision = "SP", chosen_jobs = False,
					   parallel = False, grid_ext = ".zip", worker_pool=None,
					   return_jobs=False):
	if not os.path.exists(docking_dir): os.makedirs(docking_dir)

	#grid_subdirs = [x[0] for x in os.walk(grid_dir)]
	#grid_subdirs = grid_subdirs[1:]
	#unzip_receptors(grid_dir, chosen_jobs, worker_pool)
	grid_files = get_trajectory_files(grid_dir, grid_ext)
	dock_jobs = []
	for grid_file in grid_files:
		grid_filename = grid_file.split("/")[len(grid_file.split("/"))-1]
		grid_file_no_ext = grid_filename.split(".")[0]

		if chosen_jobs is not False:
			if grid_file_no_ext not in chosen_jobs:
				#print "%s not in chosen jobs " %grid_file_no_ext
				continue
		#print grid_file_no_ext
		maegz_name = "%s/%s_pv.maegz" %(docking_dir, grid_file_no_ext)
		log_name = "%s/%s.log" %(docking_dir, grid_file_no_ext)
		log_size = 0
		if os.path.exists(log_name): log_size = os.stat(log_name).st_size
		if os.path.exists(maegz_name):# and log_size > 3000:
			#print("already docked %s" %grid_file_no_ext)
			continue
		dock_job_name = "%s/%s.in" %(docking_dir, grid_file_no_ext)
		dock_jobs.append(dock_job_name)

		dock_job_input = open(dock_job_name, "wb")
		dock_job_input.write("GRIDFILE  %s \n" %grid_file)
		dock_job_input.write("LIGANDFILE   %s \n" %ligand_dir)
		if precision == "XP":
			dock_job_input.write("POSTDOCK_XP_DELE   0.5 \n")
		dock_job_input.write("PRECISION   %s \n" %precision)
		if precision == "XP":
			dock_job_input.write("WRITE_XP_DESC   False \n")
		dock_job_input.write("OUTPUTDIR   %s \n" %docking_dir)
		dock_job_input.close()

	print("Written all docking job input files")
	#print dock_jobs
	if return_jobs:
		return dock_jobs

	if worker_pool is not None:
		print("MAPPING OVER WORKER POOL")
		worker_pool.map_sync(dock, dock_jobs)
	elif parallel:
		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		pool.map(dock, dock_jobs)
		pool.terminate()
	else:
		print("DOCKING IN SERIES")
		os.chdir(docking_dir)
		for job in dock_jobs:
			dock(job)

	print("Done docking.")

def failed(log_file):
	log = open(log_file, "rb")
	conformation = log_file.rsplit(".", 1)[0]
	conformation = conformation.split("/")[len(conformation.split("/"))-1 ]
	score = 0.0
	xp_score = None
	lines = log.readlines()
	for line in lines:
		line = line.split()
		if len(line) >= 3:
			if (line[0] == "Best" and line[1] == "XP" and line[2] == "pose:"):
				xp_score = float(line[6])
				#print "%f, %f" %(xp_score, score)
				if xp_score < score: score = xp_score
			elif  (line[0] == "Best" and line[1] == "Emodel="):
				xp_score = float(line[8])
				#print "%f, %f" %(xp_score, score)
				if xp_score < score: score = xp_score
	if score == 0: return False 
	return True
def failed_docking_jobs(docking_dir, ligand, precision):
	logs = get_trajectory_files(docking_dir, ext = ".log")
	failed_jobs = []

	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	job_results = pool.map(failed, logs)
	pool.terminate()

	for i in range(0,len(logs)):
		if job_results[i] == False:
			failed_jobs.append(logs[i])

	failed_jobs = [j.split("/")[len(j.split("/"))-1].split(".")[0] for j in failed_jobs]
	return failed_jobs



def dock_helper(args):
	dock_conformations(*args)

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess

'''
This is the function for docking multiple ligands to multiple receptors. 

grid_dir: the directory where the .zip files for each receptor to whicih you would like to dock is located. 
docking_dir = the directory to which you would like Glide to save all the results of docking 
ligands_dir = the directory containing the .maegz files containing the LigPrep prepared ligands for docking 
precision --> each Glide docking job can be done in SP or XP level of precision. XP is more accurate but takes about 7 times as long as each SP calculations
	you can change this to precision = "XP" if you would like to try that. Literature shows that it is in fact a little more accurate.
chosen_ligands --> if, in your ligands_dir directory you only want to dock particular ligands, pass here a list of strings of the ligand names,
	and it will only dock those ligands. in the example folder provided, for example, if you pass ["procaterol", "ta-2005"], it will only dock
	those two ligands
chosen_receptors --> same as chosen_ligands. if you pass ["cluster301_sample0", "cluster451_sample5"] it will only use those two receptors for docking
parallel --> if you set it to "both" it will run in parallel over both ligands and receptors. I don't recommend this generally. 
	if you pass "ligand": it will parallelize over all ligands. Recommened if n_liagnds > n_receptors
	if you pass "receptor": it will parallelize over receptors. Recommedned if n_receptors > n_ligands
'''

def dock_ligands_and_receptors(grid_dir, docking_dir, ligands_dir, precision = "SP",
							   ext = "-out.maegz", chosen_ligands = False, chosen_receptors = False,
							   parallel = False, grid_ext = ".zip", worker_pool=None,
							   ligand_dirs=None):
	ligands = get_trajectory_files(ligands_dir, ext = ext)

	if parallel == "both":
		lig_dirs = []
		docking_dirs = []
		args = []
		for ligand in ligands:
			lig_last_name = ligand.split("/")[len(ligand.split("/"))-1]
			lig_no_ext = lig_last_name.split("-out.")[0]
			if chosen_ligands is not False:
				if lig_no_ext not in chosen_ligands: continue
			lig_dir = "%s/%s" %(docking_dir, lig_no_ext)
			if not os.path.exists(lig_dir): os.makedirs(lig_dir)
			docking_dirs.append(lig_dir)
			lig_dirs.append(ligand)
			args.append((grid_dir, lig_dir, ligand, precision, chosen_receptors, True, grid_ext))
		
		num_workers = 5
		pool = MyPool(num_workers)
		pool.map(dock_helper, args)
		pool.terminate()

	elif parallel == "ligand":
		print("parallelize over ligands.")
		lig_dirs = []
		docking_dirs = []
		args = []
		for ligand in ligands:
			lig_last_name = ligand.split("/")[len(ligand.split("/"))-1]
			lig_no_ext = lig_last_name.split("-out.")[0]
			lig_dir = "%s/%s" %(docking_dir, lig_no_ext)
			if not os.path.exists(lig_dir): os.makedirs(lig_dir)
			docking_dirs.append(lig_dir)
			lig_dirs.append(ligand)
			args.append((grid_dir, lig_dir, ligand, precision, chosen_receptors, False, grid_ext))
		
		print(args)
		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		pool.map(dock_helper, args)
		pool.terminate()

	elif parallel == "receptor":
		lig_dirs = []
		docking_dirs = []
		args = []
		for ligand in ligands:
			lig_last_name = ligand.split("/")[len(ligand.split("/"))-1]
			lig_no_ext = lig_last_name.split("-out.")[0]
			if chosen_ligands is not False:
				if lig_no_ext not in chosen_ligands: continue
			lig_dir = "%s/%s" %(docking_dir, lig_no_ext)
			if not os.path.exists(lig_dir): os.makedirs(lig_dir)
			docking_dirs.append(lig_dir)
			lig_dirs.append(ligand)
			args.append((grid_dir, lig_dir, ligand, precision, chosen_receptors, True, grid_ext))
		
		for arg in args:
			dock_helper(arg)

	else:
		dock_jobs = []
		for ligand in ligands:
			lig_last_name = ligand.split("/")[len(ligand.split("/"))-1]
			lig_no_ext = lig_last_name.split("-out.")[0]
			if chosen_ligands is not False:
				if lig_no_ext not in chosen_ligands: continue
			lig_dir = "%s/%s" %(docking_dir, lig_no_ext)
			if not os.path.exists(lig_dir): os.makedirs(lig_dir)
			dock_jobs += dock_conformations(grid_dir, lig_dir, ligand, precision = precision,
							   chosen_jobs = chosen_receptors, grid_ext=grid_ext, 
							   worker_pool=worker_pool, return_jobs=True)
		if worker_pool is not None:
			worker_pool.map_sync(dock, dock_jobs)
		elif parallel:
			pool = mp.Pool(4)
			pool.map(dock, dock_jobs)
			pool.terminate()
		else:
			for dock_job in dock_jobs:
				dock(dock_job)


'''

Identical as above functions for docking, but for MM-GBSA calculations 
'''

def mmgbsa_individual(job):
	cmd = "$SCHRODINGER/prime_mmgbsa -WAIT %s" %job
	subprocess.call(cmd, shell = True)
	print("Completed mmgbsa job %s" %job)
	return

def mmgbsa(docking_dir, mmgbsa_dir, chosen_jobs = False):
	if not os.path.exists(mmgbsa_dir): os.makedirs(mmgbsa_dir)
	os.chdir(mmgbsa_dir)

	dock_files = get_trajectory_files(docking_dir, "pv.maegz")
	mmgbsa_jobs = []
	for dock_file in dock_files:
		dock_filename = dock_file.split("/")[len(dock_file.split("/"))-1]
		dock_file_no_ext = dock_filename.rsplit(".", 1)[0]
		dock_file_no_pv = dock_file_no_ext.split("_pv")[0]
		if chosen_jobs is not False:
			if dock_file_no_pv not in chosen_jobs:
				continue
		mmgbsa_out_name = "%s/%s-out.maegz" %(mmgbsa_dir, dock_file_no_pv)
		log_name = "%s/%s.log" %(mmgbsa_dir, dock_file_no_pv)
		log_size = 0
		if os.path.exists(log_name): log_size = os.stat(log_name).st_size

		if os.path.exists(mmgbsa_out_name) and log_size > 2599:
			print("Already ran mmgbsa with %s" %dock_file_no_pv)
			continue
		cmd = "cp %s %s" %(dock_file, mmgbsa_dir)
		os.system(cmd)
		job_name = "%s/%s.inp" %(mmgbsa_dir, dock_file_no_ext)
		mmgbsa_jobs.append(job_name)
		job_input = open(job_name, "wb")
		job_input.write("STRUCT_FILE %s \n" %dock_filename)
		job_input.write("OUT_TYPE COMPLEX \n")
		job_input.write("FLEXDIST 5.0 \n")
		job_input.write("OVERWRITE \n")
		job_input.close()

	print("Written all mmgbsa input files")

	num_workers = mp.cpu_count()
	pool = mp.Pool(num_workers)
	pool.map(mmgbsa_individual, mmgbsa_jobs)
	pool.terminate()
	print("Done with MM GBSA calculations")

def convert_maegz_file_to_pdb(maegz):
	current_dir = os.getcwd()
	os.chdir(os.path.dirname(maegz))
	filename_noext = os.path.splitext(maegz)[0]
	new_filename = "%s.pdb" %filename_noext
	command = "$SCHRODINGER/utilities/pdbconvert -imae %s -opdb %s" %(maegz, new_filename)
	subprocess.call(command, shell=True)
	os.chdir(current_dir)
	return


def convert_maegz_files_to_pdb(maegz_dir, ext, worker_pool=None):
	maegz_files = get_trajectory_files(maegz_dir, ext)
	if worker_pool is not None:
		worker_pool.map_sync(convert_maegz_file_to_pdb, maegz_files)
	else:
		for maegz_file in maegz_files:
			convert_maegz_file_to_pdb(maegz_file)

def mmgbsa_ligands_and_receptors(docking_dir, mmgbsa_dir, ligands, chosen_receptors = False):
	for ligand in ligands:
		lig_dir = "%s/%s" %(docking_dir, ligand)
		lig_mmgbsa_dir = "%s/%s" %(mmgbsa_dir, ligand)
		mmgbsa(lig_dir, lig_mmgbsa_dir, chosen_jobs = chosen_receptors)

import pybel
def save_sdf(mol, save_dir):
	mol.write("sdf", "%s/%s.sdf" % (save_dir, mol.title))

def fast_split_sdf(sdf_file, save_dir):
	#mols = []
	i = 0
	for i, mol in enumerate(pybel.readfile("sdf", sdf_file)):
		print(i)
		print(mol.title)
		mol.write("sdf", "%s/%s_%d.sdf" % (save_dir, mol.title, i))


	#print("converting %d sdfs" %i)
	
	#save_sdf_partial = partial(save_sdf, save_dir=save_dir)
	#pool = mp.Pool(mp.cpu_count())
	#pool.map(save_sdf, mols)
	#pool.terminate()




