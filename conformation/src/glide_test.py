from grids import *
from io_functions import get_ligands
from analysis import analyze_docking_results_multiple

base = "/scratch/users/enf/b2ar_analysis/glide_test_1"
receptors_dir = "%s/receptors" %base	#where your receptor files in pdb format live
ligands_dir = "%s/ligands" %base		#where your ligand files in SDF format live (probably trivial to make it mol2 compatible )
#reimaged_dir = "%s/receptors_reimaged" %base	#if you reimage, where your receptors will be reimaged to. OTHERWISE just set it to your original receptors dir
reimaged_dir = receptors_dir
grids_dir = "%s/grids" %base			#where the docking grids will be saved
docking_dir = "%s/docking" %base 		#where docking results willbe saved
mae_dir = reimaged_dir					#where the mae files will be saved after  the protein prep stage 
docking_summary = "%s/docking_summary.csv" %base

''' grid_center:
Must be user-specified for now. There is an easy GUI way to do this in Schrodinger.
Open your reference receptor file, i.e. the file containing the receptor to which you will align all
other receptors and the one containing the co-crystallized or pre-docked/ligand. In Schrodinger, go to 
File --> Import your reference receptor. Run protein prep wizard. Then go to Tasks --> docking --> grid generation
In "Receptor" tab, make sure you have checked "Pick to identify ligand" and "Show markers". Then, go to "Site" tab.
Copy the listed X, Y, Z coordinates. Boom.
'''
active_ref_dir = "%s/3P0G_pymol_prepped.pdb" %base			#pdb or mae file containgin the reference receptor to which all 
															#other receptors will be aligned, important for choosing the right
															#xyz centroid for the binding box
grid_center = "64.4, 16.9, 11.99"							#note how this is a string, not a python tuple

#reimage_trajs(receptors_dir, ext = ".pdb")	requires PyTraj, do only if you have to reimage your receptors
pprep(reimaged_dir, ref = active_ref_dir)					#runs schrodinger protein prep wizard, gets aligned and prepped mae file per receptor

#if there is a ligand you need to remove before docking, set remove_lig to a 3-letter string denoting its residue name
#e.g. for B2AR PDB 3P0G I would pass remove_lig = "BIA"
generate_grids(mae_dir, grid_center, grids_dir, remove_lig = None)			#generates docking grids for each receptor

inverse_agonist_ligands = get_ligands(ligands_dir)			
prepare_ligands(ligands_dir, ext = ".sdf")					#invoke Schrodinger LigPrep to prepare ligands for docking

precision = "SP"

#if n_ligands > n_receptors, set parallel = "ligand". if n_receptors > n_ligands, set parallel = "receptor"
dock_ligands_and_receptors(grids_dir, docking_dir, ligands_dir, precision = precision, ext = "-out.maegz", parallel = "ligand")

analyze_docking_results_multiple(docking_dir, precision = "SP", ligands = inverse_agonist_ligands, summary = docking_summary)

