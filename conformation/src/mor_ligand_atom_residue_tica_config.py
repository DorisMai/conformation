import os
import sys
import csv

exacycle = False


n_clusters = 100
lag_time = 5
if exacycle:
  lag_time *= 2

msm_lag_time = 5
n_components = 10
n_samples = 10
n_macrostates = 25
n_trees = 100

precision = "SP"

sparse = True
wolf = True

is_sparse = ""
if(sparse): is_sparse = "sparse-"

shrinkage_string = ""
rho_string = ""
if wolf and not sparse:
  wolf_string = "_wolf_"
  shrinkage = None
  shrinkage_string = "autoShrinkage"
  rho = None
  rho_string = ""
elif wolf and sparse:
  wolf_string = "_wolf_"
  shrinkage = None
  shrinkage_string = "autoShrinkage"
  rho = 0.005
  rho_string = "_rho0pt005"
else:
  wolf_string = ""
  shrinkage = 0.001
  shrinkage_string = "0pt001"
  rho = None
  rho_string = ""

traj_ext = ".h5"
base = "/home/enf/md_simulations/MOR/ligand_binding/h5_new"	
traj_dir = "/home/enf/md_simulations/MOR/ligand_binding/h5_new"
structure = None
pnas_features_dir =  "/home/enf/md_simulations/MOR/ligand_binding/h5_new/pnas_features"
if not os.path.exists(pnas_features_dir): os.makedirs(pnas_features_dir)
iterative = False
featurize_parallel = True
inactive_dir = "/home/enf/md_simulations/MOR/4dkl_R_for_conformation.pdb"
active_dir = "/home/enf/md_simulations/MOR/5c1m.pdb"
simulation_structure = "/home/enf/quintin/Post_Process/GPCR/MOR/LIG_path/BU_path/Rep_5-0/ionized.pdb"

sampling_method = "random"
precision = "SP"
