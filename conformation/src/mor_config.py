from feature_types import *
import os
import sys
import csv

exacycle = False


n_clusters = 1000
lag_time = 5
if exacycle:
  lag_time *= 2

msm_lag_time = 5
n_components = 25
n_samples = 10
n_macrostates = 25
n_trees = 100

sparse = False
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
  rho = 0.1
  rho_string = "_rho0pt1"
else:
  wolf_string = ""
  shrinkage = 0.001
  shrinkage_string = "0pt001"
  rho = None
  rho_string = ""

if not os.path.exists(analysis_dir): os.makedirs(analysis_dir)

traj_dir = "/home/enf/MOR/mor_active_apo_crystalwaters/reimaged"
struture = "/home/enf/MOR/mor_active_apo_crystalwaters/system.pdb"
pnas_features_dir = "/home/enf/MOR/mor_active_apo_crystalwaters/pnas_features"
if not os.path.exists(pnas_features_dir): os.makedirs(pnas_features_dir)
inactive_dir = "/home/enf/MOR/4dkl_A.pdb"
active_dir = "/home/enf/MOR/5c1m.pdb"