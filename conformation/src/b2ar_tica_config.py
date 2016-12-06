import os
import sys
import csv
from get_variable_names import get_common_residues_pkl, find_common_residues

exacycle = False
parallel_featurize = False
load_feature_parallel = True


n_clusters = 1000
lag_time = 5
if exacycle:
  lag_time *= 2

msm_lag_time = 5
n_components = 2
n_samples = 10
n_macrostates = 25
n_trees = 100

sampling_method = "random"

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
  rho = 0.01
  rho_string = "_rho0pt01"
else:
  wolf_string = ""
  shrinkage = 0.001
  shrinkage_string = "0pt001"
  rho = None
  rho_string = ""

traj_ext = ".h5"
base = "/home/enf/b2ar_analysis"
traj_dir = "/home/enf/b2ar_analysis/subsampled_reimaged_amber"
structure = None
pnas_features_dir = "/home/enf/b2ar_analysis/all_pnas_features"
if not os.path.exists(pnas_features_dir): os.makedirs(pnas_features_dir)
inactive_dir = "/home/enf/b2ar_analysis/2RH1_prepped.pdb"
active_dir = "/home/enf/b2ar_analysis/3P0G_pymol_prepped.pdb"

#contact_residues = find_common_residues([inactive_dir, active_dir, simulation_structure], common_residues_pkl)


