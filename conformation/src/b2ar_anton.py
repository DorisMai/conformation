from PDB_Order_Fixer import PDB_Order_Fixer
import mdtraj as md
import os
import numpy as np
import h5py

import datetime
import glob
import copy
from functools import partial 
import operator
import time

import random 
import subprocess
from subprocess import Popen
import sys
from io_functions import *
from custom_clusterer import *
from custom_tica import *
from custom_featurizer import *
from pdb_editing import *
from analysis import *
from io_functions import *
#from topology_fixing import *
from subsampling import *
from conversions import *
from custom_msm import *
from grids import *
from landmark_kernel_tica import *
from docking_analysis import *
from plots import *
#from pymol_color_tICA import *

import scipy as sp
from pandas import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com
from rpy2.robjects import r
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()
from detect_intermediates import *
from interpret_tICs import *

from ipyparallel import Client


base = get_base()

R_functions = "%s/conformation/analysis.R" %base
R_analysis = "%s/conformation/b2ar_analysis.R" %base
ro.r.source(R_functions)
ro.r.source(R_analysis)

grid_center = "64.4, 16.9, 11.99"

base = get_base()

#reimage_traj_new("%s/A-00.h5" %traj_dir, base, "", ".h5")
#featurize_pnas_distance(base, base, "-00.h5", inactive_ref_dir, active_ref_dir, "%s/pnas_inactive_dist_test.csv" %base, "%s/pnas_active_dist_test.csv" %base, "%s/pnas_coords_dir.csv" %base, None, "%s/pnas_active_dist_test.csv" %base, "%s/pnas_all_dist_test.csv" %base, scale = 7.14, residues_map = None)

from b2ar_feature_types import contact_residues, feature_name, cutoff, feature_name_residues_dict #, tm6_tm3_residues, npxxy_residues, connector_residues, feature_name_residues_dict, coords_bounds_dict
from get_variable_names import *
residues_map_csv =  get_residues_map_csv(base)



#do tICA
from b2ar_tica_config import *

(active_ref_dir, inactive_ref_dir, simulation_ref_dir, scripts_dir,
          ligand_dir, agonist_dir, inverse_agonist_dir, biased_agonist_dir, ref_receptors_dir, whole_trajectory_pnas,
          sasa_file) = get_base_files(base)

tica_dir = get_tica_dir(base, is_sparse, lag_time, n_components, feature_name, 
                                 wolf_string, shrinkage_string, rho_string)
ori_tica_dir = copy.deepcopy(tica_dir)
features_dir = get_features_dir(base, feature_name)

landmarks_dir = get_landmarks_dir(tica_dir)
analysis_dir = get_analysis_dir(tica_dir, n_clusters, sampling_method)
gmm_dir = get_gmm_dir(tica_dir)
rf_dir = get_rf_dir(tica_dir)


ref_tica_dir, ref_tica_coords = get_ref_tica_dirs(tica_dir)

graph_file = get_graph_file(tica_dir, msm_lag_time, n_clusters)

pnas_titles =  ["tm6_tm3_dist", "rmsd_npxxy_inactive", "rmsd_npxxy_active", "rmsd_connector_inactive", "rmsd_connector_active"]
pnas_features_dir = analysis_dir


(clusterer_dir, msm_model_dir, macrostate_dir, features_known, model_dir, projected_features_dir,
         projection_operator_dir, ktica_fit_model_filename, ktica_projected_data_filename, nystroem_data_filename,
         mutual_information_csv, pearson_csv) = get_tica_files(base, tica_dir, n_clusters, msm_lag_time, n_macrostates)

(standardized_features_dir, feature_residues_csv, feature_residues_pkl,
          contact_csv, ref_features_dir) = get_feature_files(features_dir)

(kmeans_csv, tica_coords_csv, features_csv, active_rmsd_dir, inactive_rmsd_dir, active_pnas_dir, inactive_pnas_joined, active_pnas_joined,
        clusters_map_file, ktica_clusters_map_file, analysis_file, combined_file, docking_summary, docking_joined, docking_z_scores_csv,
        aggregate_docking, aggregate_docking_joined, docking_pnas_joined, aggregate_docking_pnas, aggregate_docking_pnas_joined, docking_multiple_ligands,
        docking_distances_file, docking_pdf, mmgbsa_docking_distances, pnas_coords, mmgbsa_dir, mmgbsa_csv, mmgbsa_pdf, aggregate_mmgbsa,
        aggregate_mmgbsa_joined, aggregate_mmgbsa_pnas_joined, mmgbsa_z_scores_csv, active_clusters_csv, intermediate_clusters_csv,
        inactive_clusters_csv, pnas_clusters_averages, tica_clusters_averages, tica_classes_csv, tica_samples_csv, subgraph_save_base,
        degree_save_base, degree_map_csv, degree_z_map_csv, aggregate_docking_pnas_degree_z_joined, tic_residue_csv, feature_coefs_csv,
        duplicated_feature_coefs_csv) = get_analysis_files(analysis_dir, n_clusters, tica_dir, tica_dir, sampling_method, n_samples, precision,
                                                           msm_lag_time)

(inactive_pnas_distances_dir, active_pnas_distances_dir, active_pnas_all_distances_dir,
          inactive_pnas_distances_new_csv, active_pnas_distances_new_csv, active_pnas_joined, active_pnas_means, pnas_coords_dir,
          pnas_coords_csv, pnas_all_coords_csv, pnas_coords_hexbin_dir, pnas_coords_co_crystallized_docking_dir,
          pnas_coords_active_colors_dir, user_defined_features_file, reaction_coordinates_trajs_file) = get_pnas_files(whole_trajectory_pnas, pnas_features_dir)

features_dir = get_features_dir(base, feature_name)



graph_file = get_graph_file(tica_dir, msm_lag_time, n_clusters)
(scripts_dir, pymol_fixpdb_dir) = get_script_dir(scripts_dir)
(save_dir, reimaged_dir, mae_dir, combined_reimaged_dir, grid_dir, docking_dir) = get_docking_dirs(tica_dir, n_clusters, n_components, n_samples, sampling_method, precision)

contact_residues = get_common_residues(residues_map_csv, contact_residues)

print("BEGINNING tICA ANALYSIS")
'''
featurize_pnas_distance(traj_dir, pnas_features_dir, traj_ext, inactive_dir,
                            active_dir, inactive_pnas_distances_dir,
                            active_pnas_distances_dir, pnas_coords_dir,
                            inactive_pnas_distances_new_csv, active_pnas_distances_new_csv,
                            pnas_coords_csv, scale = 7.14, residues_map = None,
                            structure=structure, connector_residues=connector_residues, 
                            npxxy_residues=npxxy_residues, tm6_tm3_residues=tm6_tm3_residues)
'''
#plot_columns(analysis_dir, pnas_coords_dir, titles = pnas_titles, tICA = False, scale = 7.14, refcoords_file = None)

#compute_user_defined_features_wrapper(traj_dir, traj_ext, inactive_dir, active_dir, structure,
#                                          feature_name_residues_dict, user_defined_features_file)
#plot_columns(pnas_features_dir, user_defined_features_file, titles = feature_name_residues_dict.keys(), tICA=False, scale=1.0, refcoords_file=None)
#reaction_coordinate_sampler(traj_dir, traj_ext, user_defined_features_file, 
 #                               feature_name_residues_dict, coords_bounds_dict, reaction_coordinates_trajs_file)

#save_feature_residues_pkl(traj_dir, features_dir = features_dir, traj_ext = traj_ext, contact_residue_pairs_file = feature_residues_pkl, structures = [active_ref_dir, inactive_ref_dir], dihedral_residues =  [], dihedral_types = ["phi", "psi", "chi1", "chi2"], contact_residues =  contact_residues, residues_map = residues_map, contact_cutoff = cutoff, parallel = parallel, exacycle = exacycle)

#featurize_contacts_custom(traj_dir, features_dir = features_dir, traj_ext = traj_ext, contact_residue_pairs_file = feature_residues_pkl, structures = [active_ref_dir, inactive_ref_dir], dihedral_residues =  [], dihedral_types = ["phi", "psi", "chi1", "chi2"], contact_residues =  contact_residues, residues_map = None, contact_cutoff = cutoff, parallel = False, exacycle = exacycle, load_from_file=True)

#featurize_contacts_custom(ref_receptors_dir, features_dir = ref_features_dir, traj_ext = ".pdb", structures = [active_ref_dir, inactive_ref_dir], contact_residue_pairs_file = feature_residues_pkl, dihedral_residues =  [], dihedral_types = ["phi", "psi", "chi1", "chi2"], contact_residues =  contact_residues, residues_map = None, contact_cutoff = cutoff, exacycle = False, load_from_file=True)

#fit_and_transform(features_directory = features_dir, model_dir = tica_dir, stride=5, lag_time = lag_time, n_components = n_components, sparse = sparse, wolf = wolf, rho = rho, shrinkage = shrinkage, parallel=load_feature_parallel, traj_ext = traj_ext)

#timescales_plot_file = get_timescales_plot_file(tica_dir)
#plot_timescales(projection_operator_dir, timescales_plot_file, "tICA Timescales")

#corner_plot_file = get_corner_plot_file(tica_dir)
#plot_corner(projected_features_dir, corner_plot_file, "tICA", "tIC")

#interpret_tIC_components(projection_operator_dir, tica_dir, feature_residues_pkl, n_tica_components=25, percentile=95)
#plot_pnas_vs_tics(pnas_coords_dir, projected_features_dir, ["tm6_tm3_dist", "rmsd_npxxy_inactive", "rmsd_npxxy_active", "rmsd_connector_inactive", "rmsd_connector_active"], tica_dir)


#transform(existing_model = projection_operator_dir, features_directory = ref_features_dir, tica_dir = ref_tica_dir)

#plot_columns(tica_dir, projected_features_dir, titles = None, tICA = True, scale = 1.0, refcoords_file = ref_tica_coords)



#cluster_minikmeans(tica_dir, projected_features_dir, traj_dir, n_clusters=n_clusters, clusterer_dir=clusterer_dir)

rc = Client()
dview = rc[:]
dview.map(os.chdir, ['/home/enf/b2ar_analysis/conformation']*len(rc.ids))

#sample_clusters(clusterer_dir, projected_features_dir, traj_dir, traj_ext, save_dir, n_samples, method = sampling_method, clusters_map_file = clusters_map_file, worker_pool=dview)

#compute_user_defined_features_wrapper(traj_dir, traj_ext, inactive_dir, active_dir, structure,
#                                          feature_name_residues_dict, user_defined_features_file)
#plot_columns(pnas_features_dir, user_defined_features_file, titles = feature_name_residues_dict.keys(), tICA=False, scale=1.0, refcoords_file=None)

#cluster_pnas_distances(clusterer_dir, features_dir, user_defined_features_file, projected_features_dir, traj_dir, traj_ext, pnas_coords_csv, tica_coords_csv, features_csv, n_samples, sampling_method, list(feature_name_residues_dict.keys()), clusters_map_file = clusters_map_file)


#cluster_pnas_distances(clusterer_dir, features_dir, pnas_coords_dir, projected_features_dir, traj_dir, traj_ext, pnas_coords_csv, tica_coords_csv, features_csv, n_samples, sampling_method, pnas_titles, clusters_map_file = clusters_map_file)
#cluster_pnas_distances(clusterer_dir, None, active_pnas_distances_dir, pnas_coords_dir, projected_features_dir, traj_dir, traj_ext, active_pnas_distances_new_csv, pnas_coords_csv, tica_coords_csv, features_csv, n_samples, sampling_method, clusters_map_file = clusters_map_file)
#cluster_pnas_distances(clusterer_dir, None, pnas_coords_dir, projected_features_dir, traj_dir, traj_ext, pnas_coords_csv, tica_coords_csv, features_csv, n_samples, sampling_method, ["tm6_tm3_dist", "rmsd_npxxy_inactive", "rmsd_npxxy_active", "rmsd_connector_inactive", "rmsd_connector_active"], clusters_map_file =  clusters_map_file)

#r['do.analysis'](tica_dir, analysis_dir, pnas_coords_csv, tica_coords_csv, features_dir, "")

#active_clusters, intermediate_clusters, inactive_clusters = get_cluster_ids(active_clusters_csv, intermediate_clusters_csv, inactive_clusters_csv)

#plot_all_tics_and_clusters(tica_dir, projected_features_dir, clusterer_dir, lag_time, label = "dot", active_cluster_ids = active_clusters, intermediate_cluster_ids = intermediate_clusters, inactive_cluster_ids = inactive_clusters)

#find_correlation(features_dir, projected_features_dir, mutual_information_csv, pearson_csv, bins=50, exacycle = exacycle)
#r['analyze.tic.feature.correlations'](pearson_csv, feature_residues_pkl, tica_dir, "pearson_tic_feature_coefficients", " :: Feature Pearson Correlation Coefficients")
#r['analyze.tic.feature.correlations'](mutual_information_csv, feature_residues_pkl, tica_dir, "MI_tic_feature_coefficients", " :: Feature Mutual Information Score")

#plot_timescales(clusterer_dir, n_clusters, tica_dir)
#build_msm(clusterer_dir, msm_lag_time)
#macrostate_pcca(msm_model_dir, clusterer_dir, n_macrostates, macrostate_dir)
#construct_graph(msm_model_dir, clusterer_dir, n_clusters, lag_time, msm_lag_time, graph_file, inactive = None,active = active_pnas_joined, pnas_clusters_averages = pnas_clusters_averages, tica_clusters_averages = tica_clusters_averages, docking = None, macrostate = macrostate_dir) #docking=aggregate_docking_joined)


#compute_gmms_R(projected_features_dir, max_components=10, save_dir=gmm_dir, max_j=n_components)
#plot_tics_gmm(gmm_dir, projected_features_dir, gmm_dir, R=True, titles = None, tICA = False, scale = 1.0, refcoords_file = ref_tica_coords)

#compute_one_vs_all_rf_models(features_dir, projected_features_dir, gmm_dir, rf_dir, n_trees = 500, n_tica_components=10, feature_residues_map=feature_residues_pkl)
#compute_overall_rf_models(features_dir, projected_features_dir, gmm_dir, rf_dir, n_trees = 500, n_tica_components=10,feature_residues_map=feature_residues_pkl)
#plot_overall_rf_importances(rf_dir, feature_residues_pkl)

print("Performing docking and analysis of docking.")

#dview.map(os.chdir, ['/home/enf/b2ar_analysis/conformation']*len(rc.ids))

reimaged_dir = save_dir
mae_dir = reimaged_dir
#remove_ter(reimaged_dir)
#reorder(reimaged_dir)
indices = [0,1000]
chosen_receptors = []
for i in range(indices[0], indices[1]):
  for j in range(0, n_samples):
    chosen_receptors.append("cluster%d_sample%d" %(i, j))
#pprep(mae_dir, ref = active_ref_dir, chosen_receptors = chosen_receptors, worker_pool=dview)
#generate_grids(mae_dir, grid_center, grid_dir, remove_lig = "BIA", chosen_receptors = chosen_receptors, worker_pool=dview)

inverse_ligands = get_ligands(inverse_agonist_dir)
agonist_ligands = get_ligands(agonist_dir)
agonist_ligands = [a for a in agonist_ligands if "TA" not in a]
biased_ligands = get_ligands(biased_agonist_dir)

#dock_ligands_and_receptors(grid_dir, docking_dir, agonist_dir, precision = precision, ext = "-out.maegz", chosen_ligands = agonist_ligands, chosen_receptors = chosen_receptors, parallel = None, grid_ext = ".grd", worker_pool=dview)
dock_ligands_and_receptors(grid_dir, docking_dir, inverse_agonist_dir, precision = precision, ext = "-out.maegz", chosen_ligands = inverse_ligands, chosen_receptors = chosen_receptors, parallel = None, grid_ext = ".grd", worker_pool=dview)
dock_ligands_and_receptors(grid_dir, docking_dir,  biased_agonist_dir, precision = precision, ext = "-out.maegz", chosen_ligands = biased_ligands, chosen_receptors = chosen_receptors, parallel = None, grid_ext = ".grd", worker_pool=dview)


#analyze_docking_results_multiple(docking_dir, precision = "SP", ligands = [a for a in agonist_ligands if "ta" not in a], summary = docking_multiple_ligands, redo = True)
#compute_aggregate_scores(docking_multiple_ligands, inverse_agonists = inverse_ligands, summary = aggregate_docking, z_scores_csv = docking_z_scores_csv)
#plot_tICs_vs_docking(aggregate_docking, tica_coords_csv, "%s/tICA_vs_docking.pdf" % docking_dir)
#rank_tICs_by_docking_rf(aggregate_docking, tica_coords_csv, analysis_dir)
#rank_tICs_by_docking_mord(aggregate_docking, tica_coords_csv, analysis_dir)
#combine_csv_list([docking_joined, active_pnas_joined], aggregate_docking_pnas_joined)
#aggregate_docking_joined_map = convert_csv_to_joined_map(aggregate_docking, aggregate_docking_joined)[0]
##aggregate_docking_means = calc_mean(aggregate_docking_joined_map)
#write_map_to_csv(aggregate_docking_joined, aggregate_docking_means, ["cluster", "mean_aggregate_docking_z_score"])
#combine_csv_list([aggregate_docking_joined, active_pnas_joined], aggregate_docking_pnas_joined)

#LANDMARK Kernel tICA
"""
from b2ar_ktica_config import *
print("BEGINNING kernel tICA ANALYSIS")

print(feature_name)
tica_dir = get_ktica_dir(tica_dir, is_sparse, lag_time, n_components, feature_name, 
                 wolf_string, shrinkage_string, rho_string)
print(tica_dir)
ref_tica_dir, ref_tica_coords, ref_nystroem, ref_ktica_projected_data_filename = get_ref_ktica_dirs(tica_dir)
features_dir = get_features_dir(base, feature_name)

landmarks_dir = get_landmarks_dir(tica_dir)
analysis_dir = get_analysis_dir(tica_dir, n_clusters, sampling_method)
gmm_dir = get_gmm_dir(tica_dir)
rf_dir = get_rf_dir(tica_dir)

graph_file = get_graph_file(tica_dir, msm_lag_time, n_clusters)

pnas_titles =  ["tm6_tm3_dist", "rmsd_npxxy_inactive", "rmsd_npxxy_active", "rmsd_connector_inactive", "rmsd_connector_active"]
pnas_features_dir = analysis_dir


(clusterer_dir, msm_model_dir, macrostate_dir, features_known, model_dir, projected_features_dir,
         projection_operator_dir, ktica_fit_model_filename, ktica_projected_data_filename, nystroem_data_filename,
         mutual_information_csv, pearson_csv) = get_tica_files(base, tica_dir, n_clusters, msm_lag_time, n_macrostates)

(standardized_features_dir, feature_residues_csv, feature_residues_pkl,
          contact_csv, ref_features_dir) = get_feature_files(features_dir)

(kmeans_csv, tica_coords_csv, features_csv, active_rmsd_dir, inactive_rmsd_dir, active_pnas_dir, inactive_pnas_joined, active_pnas_joined,
        clusters_map_file, ktica_clusters_map_file, analysis_file, combined_file, docking_summary, docking_joined, docking_z_scores_csv,
        aggregate_docking, aggregate_docking_joined, docking_pnas_joined, aggregate_docking_pnas, aggregate_docking_pnas_joined, docking_multiple_ligands,
        docking_distances_file, docking_pdf, mmgbsa_docking_distances, pnas_coords, mmgbsa_dir, mmgbsa_csv, mmgbsa_pdf, aggregate_mmgbsa,
        aggregate_mmgbsa_joined, aggregate_mmgbsa_pnas_joined, mmgbsa_z_scores_csv, active_clusters_csv, intermediate_clusters_csv,
        inactive_clusters_csv, pnas_clusters_averages, tica_clusters_averages, tica_classes_csv, tica_samples_csv, subgraph_save_base,
        degree_save_base, degree_map_csv, degree_z_map_csv, aggregate_docking_pnas_degree_z_joined, tic_residue_csv, feature_coefs_csv,
        duplicated_feature_coefs_csv) = get_analysis_files(analysis_dir, n_clusters, ori_tica_dir, tica_dir, sampling_method, n_samples, precision,
                                                           msm_lag_time)

(inactive_pnas_distances_dir, active_pnas_distances_dir, active_pnas_all_distances_dir,
          inactive_pnas_distances_new_csv, active_pnas_distances_new_csv, active_pnas_joined, active_pnas_means, pnas_coords_dir,
          pnas_coords_csv, pnas_all_coords_csv, pnas_coords_hexbin_dir, pnas_coords_co_crystallized_docking_dir,
          pnas_coords_active_colors_dir, user_defined_features_file, reaction_coordinates_trajs_file) = get_pnas_files(whole_trajectory_pnas, pnas_features_dir)

features_dir = get_features_dir(base, feature_name)
print(features_dir)
print(feature_residues_pkl)



graph_file = get_graph_file(tica_dir, msm_lag_time, n_clusters)
(scripts_dir, pymol_fixpdb_dir) = get_script_dir(scripts_dir)
(save_dir, reimaged_dir, mae_dir, combined_reimaged_dir, grid_dir, docking_dir) = get_docking_dirs(tica_dir, n_clusters, n_components, n_samples, sampling_method, precision)

landmark_ktica(features_dir, None, tica_dir, clusters_map_file = clusters_map_file, landmarks_dir = landmarks_dir, nystroem_components=1000, n_components=n_components, lag_time=lag_time, nystroem_data_filename = nystroem_data_filename, fit_model_filename = ktica_fit_model_filename, projected_data_filename = ktica_projected_data_filename, landmark_subsample=landmark_subsample, sparse = sparse, wolf = wolf, rho = rho, shrinkage = shrinkage)
plot_pnas_vs_tics(pnas_coords_dir, ktica_projected_data_filename, ["tm6_tm3_dist", "rmsd_npxxy_inactive", "rmsd_npxxy_active", "rmsd_connector_inactive", "rmsd_connector_active"], tica_dir)
#timescales_plot_file = get_timescales_plot_file(tica_dir)
#plot_timescales(ktica_fit_model_filename, timescales_plot_file, "Kernel tICA Timescales")
#corner_plot_file = get_corner_plot_file(tica_dir)
#plot_seaborn(ktica_projected_data_filename, corner_plot_file, "kernel tICA", "k-tIC", chosen_columns=range(0,10))

print("ref_tica_coords")
print(ref_tica_coords)
#landmark_ktica(ref_features_dir, None, tica_dir, clusters_map_file = clusters_map_file, landmarks_dir = landmarks_dir, nystroem_components=1000, n_components=n_components, lag_time=lag_time, nystroem_data_filename = ref_nystroem, fit_model_filename = ktica_fit_model_filename, projected_data_filename = ref_ktica_projected_data_filename, landmark_subsample=landmark_subsample, sparse = sparse, wolf = wolf, rho = rho, shrinkage = shrinkage, refcoords_csv = ref_tica_coords)

#plot_columns(tica_dir, projected_features_dir, titles = None, tICA = True, scale = 1.0, refcoords_file = ref_tica_coords)
##plot_all_tics(tica_dir, ktica_projected_data_filename, lag_time)
cluster_minikmeans(tica_dir, ktica_projected_data_filename, traj_dir, n_clusters=n_clusters, clusterer_dir=clusterer_dir)

sample_clusters(clusterer_dir, ktica_projected_data_filename, traj_dir, traj_ext, save_dir, n_samples, method = sampling_method, clusters_map_file = ktica_clusters_map_file)

cluster_pnas_distances(clusterer_dir, None, active_pnas_distances_dir, pnas_coords_dir, ktica_projected_data_filename, traj_dir, traj_ext, active_pnas_distances_new_csv, pnas_coords_csv, tica_coords_csv, features_csv, n_samples, sampling_method, clusters_map_file = ktica_clusters_map_file)

#r['do.analysis'](tica_dir, analysis_dir, pnas_coords_csv, tica_coords_csv, features_dir, docking_multiple_ligands)

#active_clusters, intermediate_clusters, inactive_clusters = get_cluster_ids(active_clusters_csv, intermediate_clusters_csv, inactive_clusters_csv)



#plot_all_tics_and_clusters(tica_dir, ktica_projected_data_filename, clusterer_dir, lag_time, label = "dot", inactive_subsample=1, intermediate_subsample=1, active_cluster_ids=active_clusters, intermediate_cluster_ids=intermediate_clusters, inactive_cluster_ids=inactive_clusters)

#find_correlation(features_dir, ktica_projected_data_filename, mutual_information_csv, pearson_csv, bins=50, exacycle = exacycle)
#r['analyze.tic.feature.correlations'](pearson_csv, feature_residues_pkl, tica_dir, "pearson_tic_feature_coefficients", " :: Feature Pearson Correlation Coefficients")
#r['analyze.tic.feature.correlations'](mutual_information_csv, feature_residues_pkl, tica_dir, "MI_tic_feature_coefficients", " :: Feature Mutual Information Score")



compute_random_forests(features_dir, ktica_projected_data_filename, rf_dir, n_trees=500, 
                        n_tica_components=25, start_tIC=1)

interpret_tIC_rf(rf_dir, feature_residues_pkl, n_tica_components=25, percentile=95)

plot_top_features_per_tIC(ktica_projected_data_filename, features_dir, ".dataset", 
                              rf_dir, n_components, normalize=False, n_features=10)
"""
'''
print("Performing docking and analysis of docking.")
reimaged_dir = save_dir
mae_dir = reimaged_dir
#remove_ter(reimaged_dir)
#reorder(reimaged_dir)
indices = [750,1000]
chosen_receptors = []
for i in range(indices[0], indices[1]):
  for j in range(0, n_samples):
    chosen_receptors.append("cluster%d_sample%d" %(i, j))
pprep(mae_dir, ref = active_ref_dir, chosen_receptors = chosen_receptors)
generate_grids(mae_dir, grid_center, grid_dir, remove_lig = "BIA", chosen_receptors = chosen_receptors)

inverse_ligands = get_ligands(inverse_agonist_dir)
agonist_ligands = get_ligands(agonist_dir)
agonist_ligands = [a for a in agonist_ligands if "TA" not in a]
dock_ligands_and_receptors(grid_dir, docking_dir, agonist_dir, precision = precision, ext = "-out.maegz", chosen_ligands = agonist_ligands, chosen_receptors = chosen_receptors, parallel = "receptor", grid_ext = ".grd")

#analyze_docking_results_multiple(docking_dir, precision = "SP", ligands = agonist_ligands, summary = docking_multiple_ligands, redo = True)
#compute_aggregate_scores(docking_multiple_ligands, inverse_agonists = inverse_ligands, summary = aggregate_docking, z_scores_csv = docking_z_scores_csv)
#combine_csv_list([docking_joined, active_pnas_joined], aggregate_docking_pnas_joined)
#aggregate_docking_joined_map = convert_csv_to_joined_map(aggregate_docking, aggregate_docking_joined)[0]
##aggregate_docking_means = calc_mean(aggregate_docking_joined_map)
#write_map_to_csv(aggregate_docking_joined, aggregate_docking_means, ["cluster", "mean_aggregate_docking_z_score"])
#combine_csv_list([aggregate_docking_joined, active_pnas_joined], aggregate_docking_pnas_joined)


n_clusters=25
n_samples=100

tics_to_cluster = [1,10]
clusterer_dir = "%s/clusterer_%dclusters_tICs-2-11.h5" %  (tica_dir, n_clusters)
analysis_dir = get_analysis_dir(tica_dir, n_clusters, sampling_method)

(save_dir, reimaged_dir, mae_dir, combined_reimaged_dir, grid_dir, docking_dir) = get_docking_dirs(tica_dir, n_clusters, 2, n_samples, sampling_method, precision)
(kmeans_csv, tica_coords_csv, features_csv, active_rmsd_dir, inactive_rmsd_dir, active_pnas_dir, inactive_pnas_joined, active_pnas_joined,
        clusters_map_file, ktica_clusters_map_file, analysis_file, combined_file, docking_summary, docking_joined, docking_z_scores_csv,
        aggregate_docking, aggregate_docking_joined, docking_pnas_joined, aggregate_docking_pnas, aggregate_docking_pnas_joined, docking_multiple_ligands,
        docking_distances_file, docking_pdf, mmgbsa_docking_distances, pnas_coords, mmgbsa_dir, mmgbsa_csv, mmgbsa_pdf, aggregate_mmgbsa,
        aggregate_mmgbsa_joined, aggregate_mmgbsa_pnas_joined, mmgbsa_z_scores_csv, active_clusters_csv, intermediate_clusters_csv,
        inactive_clusters_csv, pnas_clusters_averages, tica_clusters_averages, tica_classes_csv, tica_samples_csv, subgraph_save_base,
        degree_save_base, degree_map_csv, degree_z_map_csv, aggregate_docking_pnas_degree_z_joined, tic_residue_csv, feature_coefs_csv,
        duplicated_feature_coefs_csv) = get_analysis_files(analysis_dir, n_clusters, ori_tica_dir, tica_dir, sampling_method, n_samples, precision,
                                                           msm_lag_time)

pnas_features_dir = analysis_dir
(inactive_pnas_distances_dir, active_pnas_distances_dir, active_pnas_all_distances_dir,
          inactive_pnas_distances_new_csv, active_pnas_distances_new_csv, active_pnas_joined, active_pnas_means, pnas_coords_dir,
          pnas_coords_csv, pnas_all_coords_csv, pnas_coords_hexbin_dir, pnas_coords_co_crystallized_docking_dir,
          pnas_coords_active_colors_dir, user_defined_features_file, reaction_coordinates_trajs_file) = get_pnas_files(whole_trajectory_pnas, pnas_features_dir)

#cluster_minikmeans(tica_dir, ktica_projected_data_filename, traj_dir, n_clusters=n_clusters, clusterer_dir=clusterer_dir,tICs=tics_to_cluster)
#cluster_gmm(ktica_projected_data_filename, clusterer_dir, tICs=[1,10], n_components=25)
#sample_clusters(clusterer_dir, ktica_projected_data_filename, traj_dir, traj_ext, save_dir, n_samples=n_samples, method = sampling_method, clusters_map_file = ktica_clusters_map_file, tICs=[1,10])
#cluster_pnas_distances(clusterer_dir, None, active_pnas_distances_dir, pnas_coords_dir, ktica_projected_data_filename, traj_dir, traj_ext, active_pnas_distances_new_csv, pnas_coords_csv, tica_coords_csv, features_csv, n_samples, sampling_method, clusters_map_file = ktica_clusters_map_file)
#r['do.analysis'](tica_dir, analysis_dir, pnas_coords_csv, tica_coords_csv, features_dir)
#active_clusters, intermediate_clusters, inactive_clusters = get_cluster_ids(active_clusters_csv, intermediate_clusters_csv, inactive_clusters_csv)
#plot_tica_and_clusters(component_j=10, transformed_data=load_file(ktica_projected_data_filename), clusterer=verboseload(clusterer_dir), lag_time=5, component_i=1, label = "dot", active_cluster_ids = active_clusters, intermediate_cluster_ids = intermediate_clusters, inactive_cluster_ids = inactive_clusters, inactive_subsample=1, intermediate_subsample=1, tica_dir = tica_dir, center_i=0, center_j=1)
#plot_tica_and_clusters(component_j=10, transformed_data=load_file(ktica_projected_data_filename), clusterer=verboseload(clusterer_dir), lag_time=5, component_i=1, label = "cluster_id", active_cluster_ids = active_clusters, intermediate_cluster_ids = intermediate_clusters, inactive_cluster_ids = inactive_clusters, inactive_subsample=1, intermediate_subsample=1, tica_dir = tica_dir, center_i=0, center_j=1)

#plot_timescales(clusterer_dir, n_clusters, tica_dir)
msm_lag_time=25
(clusterer_dir, msm_model_dir, macrostate_dir, features_known, model_dir, projected_features_dir,
         projection_operator_dir, ktica_fit_model_filename, ktica_projected_data_filename, nystroem_data_filename,
         mutual_information_csv, pearson_csv) = get_tica_files(base, tica_dir, n_clusters, msm_lag_time, n_macrostates)
clusterer_dir = "%s/clusterer_%dclusters_tICs-2-11.h5" %  (tica_dir, n_clusters)
#build_msm(clusterer_dir, lag_time=25, msm_model_dir=msm_model_dir)
#construct_graph(msm_model_dir, clusterer_dir, n_clusters, lag_time, msm_lag_time, graph_file, inactive = None, active = None, pnas_clusters_averages = pnas_clusters_averages, tica_clusters_averages = tica_clusters_averages, docking=None, macrostate = None)
msm_analysis_dir = get_msm_dir(tica_dir, msm_lag_time, n_clusters)
#compute_one_vs_all_rf_models_MSM(features_dir, ktica_projected_data_filename, clusterer_dir, msm_analysis_dir, feature_residues_pkl, n_trees=500, states_to_analyze=range(6,25))
#interpret_msm_rf(msm_analysis_dir, feature_residues_pkl, n_msm_states=25, percentile=95)

(model_file, means_file, labels_file) = get_gmm_clusterer_files(tica_dir, n_clusters)
#cluster_gmm(projected_features_file, tICs=[1,10], n_components=25,
#                model_file, means_file, labels_file):


print("Performing docking and analysis of docking.")
candidate_dir = "%s/candidates" % biased_agonist_dir
#prepare_ligands(candidate_dir, ".sdf")
candidate_ligands = get_ligands(candidate_dir)
print("candidate_ligands")
print(candidate_ligands)


biased_ligands = get_ligands(biased_agonist_dir)
print("biased_ligands")
print(biased_ligands)
reimaged_dir = save_dir
mae_dir = reimaged_dir
#remove_ter(reimaged_dir)
#reorder(reimaged_dir)
indices = [0,5]
chosen_receptors = []
for i in range(indices[0], indices[1]):
  for j in range(0, n_samples):
    chosen_receptors.append("cluster%d_sample%d" %(i, j))
#pprep(mae_dir, ref = active_ref_dir, chosen_receptors = chosen_receptors)
#generate_grids(mae_dir, grid_center, grid_dir, remove_lig = "BIA", chosen_receptors = chosen_receptors)

inverse_ligands = get_ligands(inverse_agonist_dir)
agonist_ligands = get_ligands(agonist_dir)

agonist_ligands = [a for a in agonist_ligands if "TA" not in a]
#dock_ligands_and_receptors(grid_dir, docking_dir, candidate_dir, precision = precision, ext = "-out.maegz", chosen_ligands = candidate_ligands, chosen_receptors = chosen_receptors, parallel = "receptor", grid_ext = ".grd")

#analyze_docking_results_multiple(docking_dir, precision = "SP", ligands = biased_ligands, summary = docking_multiple_ligands, redo = True)
#compute_cluster_averages(None, csv_filename=docking_multiple_ligands, save_csv=aggregate_docking)

#compute_aggregate_scores(docking_multiple_ligands, inverse_agonists = inverse_ligands, summary = aggregate_docking, z_scores_csv = docking_z_scores_csv)
#aggregate_docking_joined_map = convert_csv_to_joined_map(aggregate_docking, aggregate_docking_joined)[0]
#aggregate_docking_means = calc_mean(aggregate_docking_joined_map)
#write_map_to_csv(aggregate_docking_joined, aggregate_docking_means, ["cluster", "mean_aggregate_docking_z_score"])
#r['do.analysis'](tica_dir, analysis_dir, pnas_coords_csv, tica_coords_csv, features_dir, docking_multiple_ligands)
tics_vs_docking_file = "%s/tICA_vs_docking.pdf" % analysis_dir
plot_tICs_vs_docking(docking_multiple_ligands, tica_coords_csv, tics_vs_docking_file, chosen_ligand="s-carvedilol")



#plot_timescales(clusterer_dir, n_clusters, tica_dir)
#build_msm(clusterer_dir, msm_lag_time)
#macrostate_pcca(msm_model_dir, clusterer_dir, n_macrostates, macrostate_dir)
#construct_graph(msm_model_dir, clusterer_dir, n_clusters, lag_time, msm_lag_time, graph_file, inactive = None,active = active_pnas_joined, pnas_clusters_averages = pnas_clusters_averages, tica_clusters_averages = tica_clusters_averages, docking = None, macrostate = macrostate_dir) #docking=aggregate_docking_joined)


#compute_gmms_R(ktica_projected_data_filename, max_components=10, save_dir=gmm_dir, max_j=n_components)
#plot_tics_gmm(gmm_dir, ktica_projected_data_filename, gmm_dir, R=True, titles = None, tICA = False, scale = 1.0, refcoords_file = ref_tica_coords)

#compute_one_vs_all_rf_models(features_dir, ktica_projected_data_filename gmm_dir, rf_dir, n_trees = 10, n_tica_components=25)
#compute_overall_rf_models(features_dir, ktica_projected_data_filename, gmm_dir, rf_dir, n_trees = 500, n_tica_components=10,feature_residues_map=feature_residues_pkl)
#plot_overall_rf_importances(rf_dir, feature_residues_pkl)

#compute_overall_rf_models(features_dir, projected_features_dir, gmm_dir, rf_dir, n_trees = n_trees, n_tica_components=25)



#while(not os.path.exists(pearson_csv)):
#	print("Waiting")
#	time.sleep(100)

#time.sleep(60*130)

#featurize_pnas_distance("%s/reference_receptors" %base, "%s/reference_receptors" %base, ".pdb", inactive_ref_dir, active_ref_dir, "%s/reference_receptors/inactive_pnas_distances_ref.h5" %base, "%s/reference_receptors/active_pnas_distances_ref.h5" %base, "%s/reference_receptors/ref_coords.h5" %base, "%s/reference_receptors/inactive_distances.csv" %base, "%s/reference_receptors/active_distances.csv" %base, "%s/reference_receptors/ref_coords.csv" %base, scale = 1.0)
#convert_matrix_to_map("%s/reference_receptors/ref_coords.h5" %base, "%s/reference_receptors" %base, ".pdb", "%s/reference_receptors/ref_coords.csv" %base)

#featurize_pnas_distance(reimaged_dir, features_dir, ".pdb", inactive_ref_dir, active_ref_dir, inactive_pnas_distances_dir, active_pnas_distances_dir, pnas_coords_dir, scale = 7.14)
#convert_matrix_to_map(active_pnas_distances_dir, reimaged_dir, ".pdb",active_pnas_distances_new_csv)

#pymol_fixpdb(save_dir, pymol_fixpdb_dir)
#reorder(save_dir)
#reimage_trajs(save_dir, ext = ".pdb")
reimaged_dir = save_dir
mae_dir = reimaged_dir
#remove_ter(reimaged_dir)
#reorder(reimaged_dir)
#rmsd_pymol(reimaged_dir, inactive_ref_dir, script_dir, inactive_rmsd_dir)
#rmsd_pymol(reimaged_dir, active_ref_dir, script_dir, active_rmsd_dir)
#analyze_docking_results(docking_dir)
#pnas_distances(reimaged_dir, inactive_ref_dir, active_ref_dir)
#analyze_rmsds(inactive_rmsd_dir, active_rmsd_dir, inactive_pnas_dir, active_pnas_dir, combined_file, analysis_file)
#plot_pnas_vs_docking(docking_summary, pnas_coords, "%s/pnas_vs_docking.pdf" %docking_dir)

#pnas_distance(traj_file, inactive_file, active_file)


#plot_all_tics_and_clusters(tica_dir, projected_features_dir, clusterer_dir, lag_time)
indices = [0,250]
chosen_receptors = []
for i in range(indices[0], indices[1]):
	for j in range(0, n_samples):
		chosen_receptors.append("cluster%d_sample%d" %(i, j))
#pprep(mae_dir, ref = active_ref_dir, chosen_receptors = chosen_receptors)
#generate_grids(mae_dir, grid_center, grid_dir, remove_lig = "BIA", chosen_receptors = chosen_receptors)
#dock_conformations(grid_dir = grid_dir, docking_dir = docking_dir, ligand_dir = ligand_dir, chosen_jobs = False, precision = precision)
#analyze_docking_results(docking_dir, "BI", "SP", docking_summary)
#combine_csv_list([docking_summary, active_pnas_dir], docking_distances_file)

#docking_joined_map = convert_csv_to_joined_map(docking_summary, docking_joined)[0]
#pnas_joined_map = convert_csv_to_joined_map(active_pnas_distances_new_csv, active_pnas_joined)[0]
#inactive_pnas_joined_map = convert_csv_to_joined_map(inactive_pnas_dir, )[0]

#docking_averages = calc_mean(docking_joined_map)
#pnas_averages = calc_mean(pnas_joined_map)

#write_map_to_csv(docking_joined, docking_averages, ["cluster", "mean_docking_score"])
#write_map_to_csv(active_pnas_means, pnas_averages, ["cluster", "pnas_averages"])

#top_n =  top_n_scoring_samples(active_pnas_means, score_type = "pnas_averages", n = 10, n_samples = 1)
#print top_n
#plot_all_tics_and_clusters(tica_dir, projected_features_dir, clusterer_dir, lag_time, cluster_ids = [692, 349, 311, 356, 705, 866, 527, 763, 0, 132, 799, 685, 754])


#combine_csv_list([docking_joined, active_pnas_joined], docking_pnas_joined)

#top_n =  top_n_scoring_samples(aggregate_docking_joined, score_type = "mean_aggregate_docking_z_score", n = n_mmgbsa, n_samples = n_samples)
#print top_n



#rmsd_pymol(reimaged_dir, inactive_ref_dir, script_dir, inactive_rmsd_dir)
#rmsd_pymol(reimaged_dir, active_ref_dir, script_dir, active_rmsd_dir)
#active_pnas_map = convert_csv_to_map(active_pnas_dir)
#active_pnas_stats = calc_mean_and_stdev(active_pnas_map)
#write_map_to_csv("%s/active_pnas_stats.csv" %reimaged_dir, active_pnas_stats, ["cluster, average_pnas, stdev_pnas"])
#analyze_rmsds(inactive_rmsd_dir, active_rmsd_dir, inactive_pnas_dir, active_pnas_dir, combined_file, analysis_file)
#combine_docking_distances(docking_summary, combined_file, docking_dir)

#pnas_values = convert_matrix_to_map(active_pnas_distances_dir, reimaged_dir, ".pdb", active_pnas_distances_new_csv)
#compute_means_ligands(docking_dir, active_pnas_joined, inverse_ligands + agonist_ligands)
#convert_matrix_list_to_list("%s/features_pnas/active_pnas_distances.h5" %base, "%s/features_pnas/all_pnas_distances.csv" %base)
#unzip_receptors(grid_dir, chosen_receptors)
inverse_ligands = get_ligands(inverse_agonist_dir)
agonist_ligands = get_ligands(agonist_dir)
agonist_ligands = [a for a in agonist_ligands if "ta" not in a]
#prepare_ligands(inverse_agonist_dir, ext = ".sdf")

#dock_ligands_and_receptors(grid_dir, docking_dir, agonist_dir, precision = precision, ext = "-out.maegz", chosen_ligands = agonist_ligands, chosen_receptors = chosen_receptors, parallel = "receptor", grid_ext = ".grd")
#dock_ligands_and_receptors(grid_dir, docking_dir, inverse_agonist_dir, precision = precision, ext = "-out.maegz", chosen_ligands = inverse_ligands, chosen_receptors = chosen_receptors, parallel = "receptor", grid_ext = ".grd")

#analyze_docking_results_multiple(docking_dir, precision = "SP", ligands = agonist_ligands, summary = docking_multiple_ligands, redo = True)
#compute_aggregate_scores(docking_multiple_ligands, inverse_agonists = inverse_ligands, summary = aggregate_docking, z_scores_csv = docking_z_scores_csv)
#combine_csv_list([docking_joined, active_pnas_joined], aggregate_docking_pnas_joined)

#aggregate_docking_joined_map = convert_csv_to_joined_map(aggregate_docking, aggregate_docking_joined)[0]
#aggregate_docking_means = calc_mean(aggregate_docking_joined_map)
#write_map_to_csv(aggregate_docking_joined, aggregate_docking_means, ["cluster", "mean_aggregate_docking_z_score"])
r['do.analysis'](tica_dir, analysis_dir, pnas_coords_csv, tica_coords_csv, features_dir, docking_multiple_ligands)


#combine_csv_list([aggregate_docking_joined, active_pnas_joined], aggregate_docking_pnas_joined)

#mmgbsa(docking_dir, mmgbsa_dir, chosen_jobs = top_n)
#mmgbsa_ligands_and_receptors(docking_dir, mmgbsa_dir, inverse_ligands, chosen_receptors = top_n)
#analyze_mmgbsa_results_multiple(mmgbsa_dir, summary = mmgbsa_csv , ligands = inverse_ligands + agonist_ligands, chosen_receptors = top_n)
#compute_aggregate_scores(mmgbsa_csv, inverse_agonists = inverse_ligands, summary = aggregate_mmgbsa, z_scores_csv = mmgbsa_z_scores_csv)
#aggregate_mmgbsa_joined_map = convert_csv_to_joined_map(aggregate_mmgbsa, aggregate_mmgbsa_joined)[0]
#aggregate_mmgbsa_means = calc_mean(aggregate_mmgbsa_joined_map)
#write_map_to_csv(aggregate_mmgbsa_joined, aggregate_mmgbsa_means, ["cluster", "mean_aggregate_mmgbsa_z_score"])
#combine_csv_list([aggregate_mmgbsa_joined, active_pnas_joined], aggregate_mmgbsa_pnas_joined)

#analyze_mmgbsa_results(mmgbsa_dir, mmgbsa_csv)
#combine_docking_mmgbsa(docking_distances_file, mmgbsa_csv, mmgbsa_dir, mmgbsa_docking_distances)
#plot_pnas_vs_docking(docking_summary, pnas_coords, docking_pdf, selected = to_dock)

#plot_pnas_vs_docking(docking_summary, pnas_coords, "%s/pnas_vs_docking.pdf" %docking_dir)

#reimage_trajs(save_dir)

#featurize_known(traj_dir, inactive_ref_dir)

#plot_hex("%s/features_known/A-00.h5")


#### Dock SP, Find top n_s and do Dock XP on those, Find top n_x and MM_GBSA on those #####
#generate_grids(mae_dir, grid_center, tica_dir, n_clusters, n_samples)
#precision = "SP"
#to_dock = False
#dock_conformations(grid_dir, tica_dir, precision = precision, to_dock = to_dock)
#analyze_docking_results_multiple()

####


important residues for GPCRs:
Asp3.49	-- Asp130
Arg3.50	--	Arg131
Phe5.47	--	Phe208
Pro5.50	--	Pro211
Tyr5.58	--	Tyr219
Glu6.30	--	Glu268
Thr6.34	--	Thr272
Trp6.48	--	Trp286
Pro6.50	--	Pro288
Lys7.43	--	Lys316
Asn7.49	--	Asn322
Pro7.50	--	Pro323
Tyr7.53	--	Tyr326	
'''



'''
B2AR binding pocket residues:

Met82
Val86
His93
Cyx106
Trp109
Thr110
Asp113
Val114
Val117
Thr118
Thr164
Cyx191
Asp192
Phe193
Thr195
Tyr199
Ala200
Ser203
Val206
Ser208
Trp286
Phe289
Phe290
Asn293
Lys305
Tyr308
Ile309
Asn312
Tyr316



'''

'''
http://pubs.acs.org/doi/pdf/10.1021/jm800710x
Full aognist list:
procaterol
R-isoproterenol
R-epinephrine
TA-2005
http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2693451/
http://scicurve.com/paper/9249239

inverse agonist list:
carazolol 
atenolol
timolol
alprenolol

http://www.ncbi.nlm.nih.gov/pubmed/7908406

'''
