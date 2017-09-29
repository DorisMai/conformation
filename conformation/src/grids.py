__author__ = "Evan N. Feinberg"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

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
import random
import signal
import zipfile
from conformation.src.io_functions import *
from conformation.src.analysis import function_mapper
from pubchempy import get_compounds, Compound, Substance
from chembl_webresource_client import CompoundResource
import pubchempy as pcp
import pybel
import pandas as pd

def convert_maegz_to_mol2_single(filename, save_dir):
  try:
    job_name = filename.split("/")[-1].split(".")[0]
    print(job_name)
    savename = "%s/%s.mol2" %(save_dir, job_name)
    if not os.path.exists(savename):
      cmd = "$SCHRODINGER/utilities/structconvert -imae -omol2 %s %s" %(filename, savename)
      subprocess.call(cmd, shell=True)
    return
  except:
    return(None)


class TimeoutException(Exception):	 # Custom exception class
		pass

def timeout_handler(signum, frame):	 # Custom signal handler
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

def pprep_prot(pdb, ref=None, extension = ".mae"):
	pdb_noext = os.path.splitext(pdb)[0]
	mae_filename =	"%s.mae" % pdb_noext
	if os.path.exists(mae_filename): 
		print("already prepped and mae'd protein")
		return

	protein = md.load_frame(pdb, index=0)
	protein = protein.atom_slice([a.index for a in protein.topology.atoms if a.residue.is_protein or a.residue.is_water])
	protein.save(pdb, force_overwrite=True)

	current_directory = os.getcwd()
	os.chdir(os.path.dirname(pdb))
	mae_filename = os.path.basename(mae_filename)
	if ref is not None:
		command = "$SCHRODINGER/utilities/prepwizard -WAIT -disulfides -fix -noepik -noimpref -noprotassign -reference_st_file %s -NOLOCAL %s %s" %(ref, pdb, mae_filename)
	else:
		command = "$SCHRODINGER/utilities/prepwizard -WAIT -disulfides -fix -noepik -noimpref -noprotassign -NOLOCAL %s %s" %(pdb, mae_filename)
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

def pprep(pdb_dir, ref=None, indices = None, chosen_receptors = None, extension = ".mae", worker_pool=None, parallel=False):
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

def convert_ligand_files_to_smiles(lig_dir, lig_names):
	smiles = []
	for idx, lig_name in enumerate(lig_names):
		mol_file = "%s/%s.mol" %(lig_dir, lig_name)
		sdf_file = "%s/%s.sdf" %(lig_dir, lig_name)
		if os.path.exists(mol_file):
			lig_file = mol_file
		elif os.path.exists(sdf_file):
			lig_file = sdf_file
		else:
			smiles.append(None)
			continue
		mol = Chem.MolFromMolFile(lig_file)
		smi = Chem.MolToSmiles(mol)
		smiles.append(smi)
	return(smiles)


'''
The f ollowing two functions take as input the directory contianing the ligands you would like to dock to your receptors, 
and prepares them with Schrodinger LigPrep, and then saves them in .maegz format, required for the actual docking. 
You can change the settings listed in "ligfile.write" lines. Perhaps we should add this instead as optional inputs
in the function definition. 
'''

def prepare_ligand_job(lig_job):
	os.chdir(os.path.dirname(lig_job))
	cmd = "unset PYTHONPATH; $SCHRODINGER/ligprep -WAIT -inp %s" %lig_job
	subprocess.call(cmd, shell=True)
	return

def prepare_ligand(lig, lig_dir, n_ring_conf, n_stereoisomers, force_field, verbose=True):
	os.chdir(lig_dir)
	lig_last_name = lig.split("/")[len(lig.split("/"))-1]
	lig_no_ext = lig_last_name.split(".")[0]
	lig_ext = lig_last_name.split(".")[1]

	lig_input = "%s/%s.inp" %(lig_dir, lig_no_ext)
	lig_output = "%s-out.maegz" %lig_no_ext
	if "_bad" in lig_no_ext or "failed" in lig_no_ext:
		return

	if os.path.exists("%s/%s_bmout-bad.maegz" %(lig_dir, lig_no_ext)):
		return

	if os.path.exists("%s/%s-failed.mae" %(lig_dir, lig_no_ext)):
		return
	#	print("Already failed preparing ligand.")

	if os.path.exists("%s/%s" %(lig_dir,lig_output)):
		return

	if "sdi" in lig_ext or "sdf" in lig_ext or ("mol" in lig_ext and "mol2" not in lig_ext):
		lig_mae = "%s.mae" %(lig_no_ext)
		intermediate_file = "%s/%s" %(lig_dir, lig_mae)
		cmd = "unset PYTHONPATH; $SCHRODINGER/utilities/sdconvert -isd %s -omae %s" %(lig, intermediate_file)
		subprocess.call(cmd, shell=True)
	elif "smi" in lig_ext:
		lig_mae = "%s.mae" %(lig_no_ext)
		intermediate_file = "%s/%s" %(lig_dir, lig_mae)
		cmd = "unset PYTHONPATH; $SCHRODINGER/utilities/smiles_to_mae %s %s" %(lig, intermediate_file)
		subprocess.call(cmd, shell=True)
	else:
		lig_mae = lig_last_name	

	ligfile = open(lig_input, "w")
	ligfile.write("INPUT_FILE_NAME	 %s \n" %lig_mae)
	ligfile.write("OUT_MAE	 %s \n" %lig_output)
	ligfile.write("FORCE_FIELD	 %d \n" %force_field)
	ligfile.write("PH	 7.4 \n")
	ligfile.write("EPIK	 yes \n")
	ligfile.write("DETERMINE_CHIRALITIES	 no \n")
	ligfile.write("IGNORE_CHIRALITIES	 no \n")
	ligfile.write("NUM_STEREOISOMERS	 %d \n" %n_stereoisomers)
	ligfile.write("NUM_RING_CONF	 %d \n" %n_ring_conf)
	ligfile.close()

	return(lig_input)
	#cmd = "unset PYTHONPATH; $SCHRODINGER/ligprep -WAIT -inp %s" %lig_input
	#subprocess.call(cmd, shell=True)


def write_smiles_files(smiles_df, lig_dir):
	smiles_df = smiles_df[["ligand", "smiles"]].dropna()
	lig_names = smiles_df["ligand"].values.tolist()
	smiles_strings = smiles_df["smiles"].values.tolist()
	name_list = []
	name_smiles_tuples = []

	for name, smiles in zip(lig_names, smiles_strings):
		if name not in name_list:
			name_list.append(name)
			name_smiles_tuples.append((name,smiles))

	print("Writing SMILES files now.")
	for name, smiles in name_smiles_tuples:
		with open("%s/%s.smi" %(lig_dir, name), "w") as f:
			f.write(smiles)
	print("Finished writing SMILES files.")

def get_smiles(idx, names=None, cids=None, sids=None, binding_db=None):
	try: 
		cid = int(cids[idx])
		smiles_string = binding_db.loc[binding_db['PubChem CID'] == cid]["Ligand SMILES"].values[0]
	except:
		try:
			sid = int(sids[idx])
			smiles_string = binding_db.loc[binding_db['PubChem SID'] == sid]["Ligand SMILES"].values[0]
		except:
			try:
				cid = int(cids[idx])
				smiles_string = str(Compound.from_cid(int(cid)).isomeric_smiles)
			except:
				try:
					chembl_id = Substance.from_sid(sids[idx]).source_id
					print(chembl_id)
					try:
						compounds = CompoundResource()
						c = compounds.get(chembl_id)
						smiles_string = c["smiles"]
					except:
						smiles_string = binding_db.loc[binding_db["ChEMBL ID of Ligand"] == chembl_id]["Ligand SMILES"].values[0]
				except:
					try:
						name = names[idx]
						cs = get_compounds(name, 'name')
						smiles_string = cs[0].isomeric_smiles
					except:
						smiles_string = np.nan 
	return(smiles_string)

def add_smiles_column(cid_df, names=None, cids=None, sids=None, binding_db=None,
											worker_pool=None, parallel=False):

	indices = range(0, cid_df.shape[0])

	get_smiles_partial = partial(get_smiles, names=names, cids=cids,
															 sids=sids, binding_db=binding_db)
	
	smiles_strings = function_mapper(get_smiles_partial,
																	 worker_pool,
																	 parallel,
																	 indices)

	cid_df["smiles"] = smiles_strings
	return(cid_df)

"""
download SDF from PubChem 
cid: integer 
save_dir: string. cannot have slash at end. 
"""
def download_sdf_from_cid(cid, save_dir):
	try:
		save_file =	"%s/CID_%d.sdf" %(save_dir, cid)
		if not os.path.exists(save_file):
			pcp.download('SDF', save_file, cid, 'cid')
		return
	except:
		return

def download_sdfs_from_cids(cids, save_dir, worker_pool=None, parallel=False):
	download_sdf_from_cid_partial = partial(download_sdf_from_cid, save_dir=save_dir)
	function_mapper(download_sdf_from_cid_partial, worker_pool, parallel, cids)
	return

def convert_name_to_cid(name):
	try:
		cid = get_compounds(name, 'name')[0].cid
	except:
		cid = None
	return(cid)

def convert_names_to_cids(names, worker_pool=None, parallel=False):
	return(function_mapper(convert_name_to_cid, worker_pool, parallel, names))


def prepare_ligands(lig_dir, exts = [".mae"],
										n_ring_conf=1, n_stereoisomers=1,
										force_field=16, worker_pool=None,
										parallel=True, redo=False,
										smiles_df=None, cid_df=None,
										binding_db=None,
										return_df=True):
	if smiles_df is not None or cid_df is not None:
		if ".smi" not in exts:
			exts.append(".smi")
	
	if smiles_df is not None:
		write_smiles_files(smiles_df, lig_dir)

	if cid_df is not None:
		if "CID" in cid_df.columns.values.tolist():
			cids = cid_df["CID"].values
			cid_df["ligand"] = ["CID_%d" %cid for cid in cids.tolist()]
		else:
			cids = None
		if "ID" in cid_df.columns.values.tolist():
			sids = cid_df["SID"].values
			cid_df["ligand"] = ["SID_%d" %sid for sid in sids.tolist()]
		else:
			sids = None
		if "name" in cid_df.columns.values.tolist():
			names = cid_df["name"].values
			new_names = []
			for name in names:
				new_names.append(name.replace("β", "beta").replace("α", "alpha"))
			cid_df["ligand"] = names
		else:
			names = None

		indices = range(0, cid_df.shape[0])

		get_smiles_partial = partial(get_smiles, names=names, cids=cids,
																 sids=sids, binding_db=binding_db)
		
		smiles_strings = function_mapper(get_smiles_partial,
																		 worker_pool,
																		 parallel,
																		 indices)

		cid_df["smiles"] = smiles_strings
		write_smiles_files(cid_df, lig_dir)


	ligs = []
	for ext in exts:
		ligs += get_trajectory_files(lig_dir, ext)

	for lig in ligs:
		if "#" in lig:
			new_name = lig.replace("#", "")
			subprocess.call("mv %s %s" %(lig, new_name), shell=True)

	ligs = []
	for ext in exts:
		ligs += get_trajectory_files(lig_dir, ext)

	print("Examining %d ligands" %len(ligs))


	lig_partial = partial(prepare_ligand, lig_dir = lig_dir, 
												n_ring_conf=n_ring_conf,
												n_stereoisomers=n_stereoisomers,
												force_field=force_field)

	lig_jobs = [lig_partial(lig) for lig in ligs]
	lig_jobs = [job for job in lig_jobs if job is not None]

	print("About to prepare %d ligands" %len(lig_jobs))

	if worker_pool is not None:
		random.shuffle(lig_jobs)
		worker_pool.map_async(prepare_ligand_job, lig_jobs)
	elif parallel:
		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		pool.map(prepare_ligand_job, lig_jobs)
		pool.terminate()
	else:
		for lig in ligs:
			prepare_ligand_job(lig_jobs)
	print("finished preparing ligands")
	if return_df:
		return(cid_df)

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

	if not os.path.exists(new_mae):
		cmd = "cp %s %s" %(mae, new_mae)
		print(cmd)
		subprocess.call(cmd, shell=True)
	else:
		#cmd = "$SCHRODINGER/run $SCHRODINGER/mmshare-v3.3/python/common/delete_atoms.py -asl \"(atom.i_m_pdb_convert_problem 4)\" %s %s" %(mae, new_mae)
		#cmd = "$SCHRODINGER/run $SCHRODINGER/mmshare-v3.3/python/common/delete_atoms.py -asl \"res.pt %s \" %s %s" %(remove_lig, mae, new_mae)
		#print(cmd)
		#subprocess.call(cmd, shell=True)
		print("")

	gridfile = open(grid_job, "w")
	gridfile.write("GRIDFILE	 %s.zip \n" %mae_last_name)
	gridfile.write("OUTPUTDIR	 %s \n" %output_dir)
	gridfile.write("GRID_CENTER	 %s \n" %grid_center)
	gridfile.write("INNERBOX	 10, \n")
	gridfile.write("OUTERBOX	 %d, \n" %outer_box)
	gridfile.write("RECEP_FILE	 %s \n" %new_mae)
	#gridfile.write("RECEP_VSCALE	 1.0 \n")
	#gridfile.write("LIGAND_MOLECULE	1 \n")
	#gridfile.write("WRITEZIP	 TRUE \n")
	gridfile.close()

def generate_grid(grid_file, grid_dir):
	grid_zip = grid_file.rsplit( ".", 1)[0]
	grid_zip = "%s.zip" %grid_zip
	if os.path.exists(grid_zip):
		print("already generated grid; skipping")
		return

	try:
		os.chdir(grid_dir)
		grid_command = "$SCHRODINGER/glide %s -OVERWRITE -WAIT" %grid_file
		print(grid_command)
		subprocess.call(grid_command, shell = True)
		print("completed grid generation job")
	except:
		print("ERROR WITH GRID GENERATION")
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


def generate_grids(mae_dir, grid_center, grid_dir, remove_lig = None,
									 indices = None, chosen_receptors = None, outer_box=25.,
									 worker_pool=None, parallel=False):
	print(grid_dir)
	if not os.path.exists(grid_dir): os.makedirs(grid_dir)

	maes = get_trajectory_files(mae_dir, ".mae")
	if chosen_receptors is not None:
		maes = [mae for mae in maes if remove_path_and_extension(mae) in chosen_receptors]

	generate_grid_input_partial = partial(generate_grid_input, grid_dir = grid_dir, grid_center = grid_center, remove_lig = remove_lig, outer_box=outer_box)
	grid_partial = partial(generate_grid, grid_dir = grid_dir)

	if indices is not None:
		grid_files = grid_files[indices[0] : indices[1]]

	if worker_pool is not None:
		worker_pool.map_sync(generate_grid_input_partial, maes)
		grid_files = get_trajectory_files(grid_dir, ".in")
		worker_pool.map_sync(grid_partial, grid_files)
	elif parallel:
		num_workers = mp.cpu_count()
		pool = mp.Pool(num_workers)
		pool.map(generate_grid_input_partial, maes)
		grid_files = get_trajectory_files(grid_dir, ".in")
		pool.map(grid_partial, grid_files)
		pool.terminate() 
	else:
		for mae in maes:
			generate_grid_input_partial(mae)		
		grid_files = get_trajectory_files(grid_dir, ".in")
		for grid_file in grid_files:
			grid_partial(grid_file)

	#zips = get_trajectory_files(grid_dir, ".zip")
	#pool = mp.Pool(num_workers)
	#pool.map(unzip, zips)
	#pool.terminate()

	return grid_dir

'''
the function, dock_conformations() is to dock a single ligand to many conformations. you will probably	be using the function,
dock_ligands_and_receptors(), however.
'''

from subprocess import STDOUT, check_output

def run_command(cmd):
	subprocess.call(cmd, shell = True)

def dock(dock_job, timeout=600):
	try:
		docking_dir = os.path.dirname(dock_job)
		os.chdir(docking_dir)
		log_file = "%s.log" %(dock_job.split("/")[-1].split(".")[0])
		cmd = "timeout %d $SCHRODINGER/glide %s -OVERWRITE -WAIT -NOJOBID -strict > %s" %(timeout, dock_job, log_file)
		print(cmd)
		run_command(cmd)
		#p = subprocess.Popen(cmd, shell=True)
		#try:
		#	p.wait(timeout=5)
		#except:
		#	print("Docking job timed out")
		#	p.terminate()
		#except:

		#	print("Docking job timed out.")

		os.chdir("/home/enf/b2ar_analysis/conformation")
	except:
		os.chdir("/home/enf/b2ar_analysis/conformation")
		print("docking job timed out or failed.")
	return


def dock_conformations(docking_ligand_dir_tuple, grid_dir="", precision = "SP",
											 chosen_jobs=None,
											 grid_ext = ".zip",
											 return_jobs=False, retry_after_failed=False,
											 redo=False):
	docking_dir, ligand_dir = docking_ligand_dir_tuple
	if not os.path.exists(docking_dir): os.makedirs(docking_dir)

	#grid_subdirs = [x[0] for x in os.walk(grid_dir)]
	#grid_subdirs = grid_subdirs[1:]
	#unzip_receptors(grid_dir, chosen_jobs, worker_pool)
	grid_files = get_trajectory_files(grid_dir, grid_ext)
	dock_jobs = []
	for grid_file in grid_files:
		grid_filename = grid_file.split("/")[len(grid_file.split("/"))-1]
		grid_file_no_ext = grid_filename.split(".")[0]

		if chosen_jobs is not None:
			if grid_file_no_ext not in chosen_jobs:
				#print "%s not in chosen jobs " %grid_file_no_ext
				continue
		if not redo:
			#print grid_file_no_ext
			maegz_name = "%s/%s_pv.maegz" %(docking_dir, grid_file_no_ext)
			lib_name = "%s/%s_lib.maegz" %(docking_dir, grid_file_no_ext)
			log_name = "%s/%s.log" %(docking_dir, grid_file_no_ext)
			log_size = 0
			if os.path.exists(log_name): log_size = os.stat(log_name).st_size
			if (os.path.exists(maegz_name) or os.path.exists(lib_name)):
				continue
			elif os.path.exists(log_name):
				with open(log_name, 'r') as f:
					read_log = f.read().replace('\n', '')
					if "cannot write PoseViewer" in read_log or "No Ligand Poses were written" in read_log:
						if not retry_after_failed:
							continue


			#if not retry_after_failed:
		#		if os.path.exists(log_name):
		#			conformation, score, best_pose = analyze_log_file(log_name)
	#				if score == 0.0:
#						continue

		dock_job_name = "%s/%s.in" %(docking_dir, grid_file_no_ext)
		dock_jobs.append(dock_job_name)

		with open(dock_job_name, "w") as dock_job_input:
			dock_job_input.write("GRIDFILE	%s \n" %grid_file)
			dock_job_input.write("LIGANDFILE	 %s \n" %ligand_dir)
			if precision == "XP":
				dock_job_input.write("POSTDOCK_XP_DELE	 0.5 \n")
			dock_job_input.write("PRECISION	 %s \n" %precision)
			if precision == "XP":
				dock_job_input.write("WRITE_XP_DESC	 False \n")
			dock_job_input.write("OUTPUTDIR	 %s \n" %docking_dir)
			dock_job_input.write("POSE_OUTTYPE ligandlib \n")

	#print("Written all docking job input files")
	#print dock_jobs
	if return_jobs:
		return dock_jobs
	"""
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
	"""

def dock_ligand_conformation_tuples(ligand_conformation_tuples, docking_dir,
																		precision="XP", worker_pool=None,
																		timeout=300):
	dock_jobs = []
	for tup in ligand_conformation_tuples:
		lig_filename, grid_filename = tup

		lig_name = lig_filename.split(".")[0].split("/")[-1]

		prot_name = grid_filename.split(".")[0].split("/")[-1]

		dock_job_name = "%s/%s_%s.in" %(docking_dir, lig_name, prot_name)
		outfile = "%s/%s_%s_lib.maegz" %(docking_dir, lig_name, prot_name)
		if not os.path.exists(outfile):
			dock_jobs.append(dock_job_name)

		if not os.path.exists(dock_job_name):
			with open(dock_job_name, "w") as dock_job_input:
				dock_job_input.write("GRIDFILE	%s \n" %grid_filename)
				dock_job_input.write("LIGANDFILE	 %s \n" %lig_filename)
				if precision == "XP":
					dock_job_input.write("POSTDOCK_XP_DELE	 0.5 \n")
				dock_job_input.write("PRECISION	 %s \n" %precision)
				if precision == "XP":
					dock_job_input.write("WRITE_XP_DESC	 False \n")
				dock_job_input.write("OUTPUTDIR	 %s \n" %docking_dir)
				dock_job_input.write("POSE_OUTTYPE ligandlib \n")


	print("About to run %d docking jobs" %len(dock_jobs))
	if worker_pool is None:
		for dock_job in dock_jobs:
			dock(dock_job, timeout=timeout)
	else:
		dock_partial = partial(dock, timeout=timeout)
		random.shuffle(dock_jobs)
		worker_pool.map_async(dock_partial, dock_jobs)



def failed(log_file):
	log = open(log_file, "r")
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
			elif	(line[0] == "Best" and line[1] == "Emodel="):
				xp_score = float(line[8])
				#print "%f, %f" %(xp_score, score)
				if xp_score < score: score = xp_score
	if score == 0: return False 
	log.close()
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
								 ext = "-out.maegz", chosen_ligands = False, chosen_receptors = None,
								 parallel = False, grid_ext = ".zip", worker_pool=None,
								 ligand_dirs=None, retry_after_failed=False, timeout=600,
								 redo=False):
	ligands = get_trajectory_files(ligands_dir, ext = ext)

	dock_jobs = []
	ligand_dirs = []

	ligand_dir_tuples = []

	print("Creating new directories for each ligand.")
	for ligand in ligands:
		lig_last_name = ligand.split("/")[len(ligand.split("/"))-1]
		lig_no_ext = lig_last_name.split("-out.")[0]
		if chosen_ligands is not False:
			if lig_no_ext not in chosen_ligands: continue
		lig_dir = "%s/%s" %(docking_dir, lig_no_ext)
		if not os.path.exists(lig_dir): os.makedirs(lig_dir)
		ligand_dir_tuples.append((lig_dir, ligand))

	print("Done creating directories. Determining which docking jobs to conduct.")
	dock_conformations_partial = partial(dock_conformations, grid_dir=grid_dir,
																			 precision=precision,
																			 chosen_jobs=chosen_receptors, grid_ext=grid_ext,
																			 return_jobs=True,
																			 retry_after_failed=retry_after_failed, redo=redo)
	print(mp.cpu_count())	
	a = time.time()
	pool = mp.Pool(mp.cpu_count()-1)
	dock_jobs = pool.map_async(dock_conformations_partial, ligand_dir_tuples)
	#dock_jobs = worker_pool.map(dock_conformations_partial, ligand_dir_tuples)
	#dock_jobs.wait_interactive()
	dock_jobs.wait()
	dock_jobs = dock_jobs.get()
	pool.terminate()
	print(time.time()-a)

	print(len(dock_jobs))

	#for i, ldt in enumerate(ligand_dir_tuples):
#		if i % 100 == 0: print(i)
	#	dock_jobs.append(dock_conformations_partial(ldt))


	#dock_jobs = function_mapper(dock_conformations_partial,
#								worker_pool, parallel,
#									ligand_dir_tuples)

	dock_jobs = [job for ligand in dock_jobs for job in ligand]

	partial_docker = partial(dock, timeout=timeout)
	print("About to do %d Docking computations." %len(dock_jobs))
	
	if worker_pool is not None:
		random.shuffle(dock_jobs)
		docked_jobs = worker_pool.map(partial_docker, dock_jobs)
		docked_jobs.wait()
		#docked_jobs.wait_interactive()
	elif parallel:
		pool = mp.Pool(4)
		pool.map(partial_docker, dock_jobs)
		pool.terminate()
	else:
		for dock_job in dock_jobs[:2]:
			partial_docker(dock_job)
	
	print("Completed docking.")


'''

Identical as above functions for docking, but for MM-GBSA calculations 
'''

#def split_and_run_dock_jobs(dock_jobs, n_nodes):
#	random.shuffle(dock_jobs)
#	for i in range(dock_)

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
		job_input = open(job_name, "w")
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

def get_lig_names(docking_dir):
  subdirs = [x[0] for x in os.walk(docking_dir)]
  lig_names = []

  for subdir in subdirs:
    lig_name = subdir.split("/")[len(subdir.split("/"))-1]
    lig_names.append(lig_name)

  return lig_names

def listdirs(folder):
    #return [
    #    d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
    #    if os.path.isdir(d)
    #]
    #from glob import glob
    return(["%s/%s" %(folder, f) for f in os.listdir(folder)])
    #return([d.path for d in os.scandir(folder) if d.is_dir()])

def get_arg_tuple(subdir, ligands=None, precision="SP", redo=False, reread=True, write_to_disk=False):
  lig_name = subdir.split("/")[len(subdir.split("/"))-1]
  if ligands is not None:
    if lig_name not in ligands:
      return None
  docking_summary = "%s/docking_summary.csv" %subdir
  return([subdir, lig_name, precision, docking_summary, reread, write_to_disk])

def parse_log_file(result):
  ligand, result = result
  keep_columns = [c for c in result.columns.values.tolist() if "score" in c]
  result.index = result['sample'].values

  docking_pose = result["best_pose_id"]
  result = result[keep_columns]

  return (ligand, docking_pose, result)

def analyze_log_file(log_file):
	log = open(log_file, "rb")
	conformation = log_file.rsplit(".", 1)[0]
	conformation = conformation.split("/")[len(conformation.split("/"))-1 ]
	score = 0.0
	xp_score = None
	lines = log.readlines()
	current_pose = 0
	best_pose = 0
	for line in lines:
		line = [w.decode("utf-8") for w in line.split()]
		if len(line) >= 3:
			if (line[0] == "Best" and line[1] == "XP" and line[2] == "pose:"):
				current_pose += 1
				xp_score = float(line[6])
				#print "%f, %f" %(xp_score, score)
				if xp_score < score: 
					score = xp_score
					best_pose = current_pose
			elif  (line[0] == "Best" and line[1] == "Emodel="):
				current_pose += 1
				xp_score = float(line[8])
				#print("%f, %f" %(xp_score, score))
				if xp_score < score:
					score = xp_score
					best_pose = current_pose

	score = -1.0*score
	log.close()
	return (conformation, score, best_pose)

def analyze_log_files_in_dir(path, parallel=False, worker_pool=None):
	log_files = get_trajectory_files(path, ".log")
	if worker_pool is not None:
		results = worker_pool.map_sync(analyze_log_file, log_files)
	elif parallel:
		pool = mp.Pool(mp.cpu_count())
		results = pool.map(analyze_log_file, log_files)
		pool.terminate()
	else:
		results = [analyze_log_file(log_file) for log_file in log_files]

	scores = np.array([r[1] for r in results]).reshape((-1,1))
	index = [f.split("/")[-1].split(".")[0] for f in log_files]
	df = pd.DataFrame(scores, index=index)
	return(df)


def analyze_docking_results(docking_dir, ligand, precision, docking_summary, reread=True, write_to_disk=False):
  try:
    results_file = docking_summary
    n_clusters = len(get_trajectory_files(docking_dir, ext = ".in"))

    if os.path.exists(results_file):
      if reread:
        df = pd.read_csv(results_file)
        if df.shape[0] >= n_clusters:
          return (ligand, df)

    logs = get_trajectory_files(docking_dir, ext = ".log")
    scores_list = []

    for log in logs:
      scores_list.append(analyze_log_file(log))

    scores_df = pd.DataFrame(scores_list, columns=["sample", "%s" %("%s_%s_score" %(ligand, precision)), "best_pose_id"])
    if write_to_disk:
      scores_df.to_csv(results_file)
    #scores_map = {score[0] : score[1] for score in scores_list}
    #titles = ["sample", "%s" %("%s_%s_score" %(ligand, precision))]
    #write_map_to_csv(results_file, scores_map, titles)
    #merged_results = merge_samples(scores_map)
    return (ligand, scores_df)
  except:
    return None

def analyze_docking_results_wrapper(args):
  return analyze_docking_results(*args)

def analyze_docking_results_multiple(docking_dir, precision, summary,
                                     ligands=None, poses_summary=None, redo=False, reread=True,
                                     write_to_disk=False, parallel=False, worker_pool=None):

  """Produces Pandas DataFrame where each row is a ligand, each column
     is a protein conformation, and each entry is the maximum Glide docking score
     for that (ligand_i, protein_j) combination
  Parameters
  ----------
  docking_dir: directory organized by ligand subdirectories
    in each ligand subdirectory will be Glide .log files of
    the docking for all protein conformations to that ligand
  precision: "SP" or "XP"
  summary: provide a pkl_file for saving the dataframe
  parallel: will use multiprocessing to try to speed up analysis
  """

  if os.path.exists(summary) and not redo:
    with open(summary, "rb") as f:
      df = pickle.load(f)
    return(df, None)
    #df = pd.read_csv(summary, index_col=0).transpose()
    return df, None

  print("Analyzing docking results")
  print(docking_dir)
  subdirs = listdirs(docking_dir)
  #print subdirs
  results_list = []
  lig_names = []
  arg_tuples = []

  print("Obtaining docking scores now...")

  get_arg_tuple_partial = partial(get_arg_tuple,
                    ligands=ligands,
                    precision=precision,
                    redo=redo, reread=reread,
                    write_to_disk=write_to_disk)

  pool = mp.Pool(mp.cpu_count()-1)
  arg_tuples = pool.map_async(get_arg_tuple_partial,
                  subdirs)
  arg_tuples.wait()
  arg_tuples = arg_tuples.get()
  arg_tuples = [t for t in arg_tuples if t is not None]
  pool.terminate()

  #arg_tuples = function_mapper(get_arg_tuple_partial,
  #              worker_pool,
  #                parallel, subdirs)


  print("Obtained ligand arguments.")

  #results_list = function_mapper(analyze_docking_results_wrapper, worker_pool, parallel, arg_tuples)
  a = time.time()
  pool = mp.Pool(mp.cpu_count()-1)
  results_list = pool.map_async(analyze_docking_results_wrapper, arg_tuples)
  results_list.wait()
  results_list = results_list.get()
  pool.terminate()

  results_list = [r for r in results_list if r is not None]
  print(time.time()-a)
  print("Examined all ligands.")

  a = time.time()
  pool = mp.Pool(mp.cpu_count()-1)
  results = pool.map_async(parse_log_file, results_list)
  results.wait()
  results = results.get()
  pool.terminate()

  all_docking_results = [t[2] for t in results]
  all_docking_poses = [t[1] for t in results]
  lig_names = [t[0] for t in results]
  print(time.time()-a)
  print("Parsed all log files.")


  all_docking_poses = pd.concat(all_docking_poses, axis=1)
  all_docking_poses.columns = lig_names
  if poses_summary is not None:
    all_docking_poses.to_csv(poses_summary)

  all_docking_results = pd.concat(all_docking_results, axis=1)
  all_docking_results.columns = lig_names

  all_docking_results = all_docking_results.transpose()

  with open(summary, "wb") as f:
    pickle.dump(all_docking_results, f, protocol=2)
  #all_docking_results.to_csv(summary)
  return all_docking_results, all_docking_poses

  #combined_map = combine_maps(results_list)
  #combined_filename = summary
  #write_map_to_csv(combined_filename, combined_map, ["sample"] + lig_names)

	#print("converting %d sdfs" %i)
	
	#save_sdf_partial = partial(save_sdf, save_dir=save_dir)
	#pool = mp.Pool(mp.cpu_count())
	#pool.map(save_sdf, mols)
	#pool.terminate()




