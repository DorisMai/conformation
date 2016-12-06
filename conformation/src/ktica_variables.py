from io_functions import get_base, generate_residues_map
from feature_types import *
from tica_variables import *
import os

base = get_base()
residues_map_csv = "%s/exacycle_data/residues_map.csv" %base


ktica = True
#Use tICA projection as the features for Nystroem approximation, DO NOT DO THIS UNDER MOST CIRCUMSTANCES:
ktica_ticaTraj = False

k_tica_components = 25
landmark_subsample=5
n_components = 25

if exacycle:
	traj_dir = "%s/exacycle_data/b2ar3p0g2rh1_bi/Trajectories" %base
	pnas_features_dir = "%s/exacycle_data/features_pnas" %base
	traj_ext = ".lh5"
	feature_parallel = True
else:
	traj_dir = "%s/subsampled_reimaged_amber" %base
	pnas_features_dir = "%s/features_pnas" %base
	traj_ext = ".h5"
	feature_parallel = False 


n_clusters = 1000
if exacycle:
	lag_time *= 2
#tica_regularization = 1000.0    
#tica_regularization_string = "1000pt0"


if(exacycle): base = "%s/exacycle_data" %base
if(exacycle):
	parallel = True
else:
	parallel = False


sparse = False
is_sparse = ""
if(sparse): is_sparse = "sparse-"

shrinkage_string = ""
rho_string = ""

wolf = True
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


if ktica: tica_dir = "%s/%sktICA_n_components%d_random_specified_regularization%s%s%s" %(ori_tica_dir, is_sparse, k_tica_components, wolf_string, shrinkage_string, rho_string)
if ktica_ticaTraj: tica_dir = "%s/%sktICA_n_components%d_ticaTraj_regularization%s%s%s" %(ori_tica_dir, is_sparse, k_tica_components, wolf_string, shrinkage_string, rho_string)
tica_extremes_dir = "%s/tica_extreme_features" %tica_dir 
percentile = 0.01
#r.assign('tica.dir',tica_dir)
landmarks_dir = "%s/landmarks.h5" %tica_dir
#r.assign('landmarks.dir', 'landmarks_dir')
if not os.path.exists(tica_dir): os.makedirs(tica_dir)
analysis_dir = "%s/analysis_n_clusters%d_%s" %(tica_dir, n_clusters, sampling_method)
gmm_dir = "%s/gmm" %tica_dir
if not os.path.exists(gmm_dir): os.makedirs(gmm_dir)
rf_dir = "%s/rf" %tica_dir
if not os.path.exists(rf_dir): os.makedirs(rf_dir)
#r.assign('analysis.dir', analysis_dir)
ori_clusterer_dir = "%s/clusterer_%dclusters.h5" %(ori_tica_dir, n_clusters)
clusterer_dir = "%s/clusterer_%dclusters.h5" %(tica_dir, n_clusters)
kmeans_csv = "%s/kmeans_csv" %analysis_dir
#features_dir = "%s/features_allprot"
features_dir = "%s/features%s" %(base,feature_types)
standardized_features_dir = "%s_standardized" %features_dir 
feature_residues_csv = "%s/feature_residues_map.csv" %features_dir 
contact_csv = "%s/contact_residue_pairs.csv" %features_dir
combined_features_dir = "%s/combined_features.h5" %ori_tica_dir
msm_model_dir = "%s/msm_model_%d_clusters_t%d.h5" %(tica_dir, n_clusters, msm_lag_time)
macrostate_dir = "%s/msm_modeL%d_clusters_t%d_macrostate_states%d" %(tica_dir, n_clusters, msm_lag_time,  n_macrostates)
features_known = "%s/features_known" %base
model_dir = tica_dir
projected_features_dir = "%s/phi_psi_chi2_allprot_projected.h5" %(ori_tica_dir)
projection_operator_dir = "%s/phi_psi_chi2_allprot_tica_coords.h5" %(ori_tica_dir)
ktica_fit_model_filename = "%s/ktica_model.h5" %tica_dir
ktica_projected_data_filename = "%s/ktica_random_specified_projected_coords.npz" %tica_dir
nystroem_data_filename = "%s/nystroem_random_specified.npz" %tica_dir
mutual_information_csv = "%s/mutual_information.csv" %tica_dir
pearson_csv = "%s/pearson.csv" %tica_dir
save_dir = "%s/clusters%d_n_components%d_n_samples%d_%s" %(tica_dir, n_clusters, n_components, n_samples, sampling_method)
#save_dir = "%s/reorder_test" %tica_dir
#reimaged_dir = save_dir
reimaged_dir = "%s/clusters%d_n_components%d_n_samples%d_%s_reimaged" %(tica_dir, n_clusters, n_components, n_samples, sampling_method)
mae_dir = reimaged_dir
combined_reimaged_dir = "%s/combined.h5" %reimaged_dir
tica_coords_csv = "%s/tica_coords.csv" %analysis_dir
#r.assign('tica.coords.csv', tica_coords_csv)
active_ref_dir = "%s/3P0G_pymol_prepped.pdb" %base
inactive_ref_dir = "%s/2RH1_prepped.pdb" %base
scripts_dir = "%s/scripts" %base
script_dir = "%s/pymol_rmsd.py" %scripts_dir
pymol_fixpdb_dir = "%s/pymol_fixpdb.py" %scripts_dir
active_rmsd_dir =  "%s/active_rmsds.csv" %analysis_dir
inactive_rmsd_dir = "%s/inactive_rmsd.csv" %analysis_dir
active_pnas_dir = "%s/active_pnas_distances.csv" %analysis_dir
inactive_pnas_dir = "%s/inactive_pnas_distances.csv" %analysis_dir
inactive_pnas_joined = "%s/inactive_pnas_joined.csv" %analysis_dir
active_pnas_joined = "%s/active_pnas_joined.csv" %analysis_dir
clusters_map_file = "%s/clusters_map_%d_clusters_%s" %(ori_tica_dir, n_clusters, sampling_method)
ktica_clusters_map_file = "%s/clusters_map_%d_clusters_%s" %(tica_dir, n_clusters, sampling_method)
analysis_file = "%s/rmsd_analyzed.csv" %analysis_dir
combined_file = "%s/rmsd_combined.csv" %analysis_dir
ligand_dir = "%s/ligprep_2/ligprep_2-out.maegz" %base
agonist_dir = "%s/b2ar_full_agonists" %base
inverse_agonist_dir = "%s/b2ar_inverse_agonists" %base
grid_dir = "%s/grids_n_clusters%d_n_samples%d_%s" %(tica_dir, n_clusters, n_samples, sampling_method)
docking_dir = "%s/docking_n_clusters%d_n_samples%d_%s_%s" %(tica_dir, n_clusters, n_samples, sampling_method, precision)
docking_summary = "%s/docking_summary.csv" %analysis_dir
docking_joined = "%s/docking_joined.csv" %analysis_dir
docking_z_scores_csv = "%s/docking_z_scores.csv" %analysis_dir 
aggregate_docking = "%s/aggregate_docking.csv" %analysis_dir
#r.assign('docking.aggregated.csv', aggregate_docking)
aggregate_docking_joined = "%s/aggregate_docking_joined.csv" %analysis_dir
docking_pnas_joined = "%s/docking_pnas_joined.csv" %analysis_dir
aggregate_docking_pnas = "%s/aggregate_docking_pnas.csv" %analysis_dir
aggregate_docking_pnas_joined = "%s/aggregate_docking_pnas_joined.csv" %analysis_dir
docking_multiple_ligands = "%s/all_docking_combined.csv" %analysis_dir
#r.assign('docking.csv', docking_multiple_ligands)
docking_distances_file = "%s/distances_docking.csv" %analysis_dir
docking_pdf = "%s/pnas_vs_docking.pdf" %analysis_dir
mmgbsa_docking_distances = "%s/mmgbsa_docking_distances.csv" %analysis_dir
pnas_coords = "%s/pnas_coords.csv" %analysis_dir
mmgbsa_dir = "%s/mmgbsa_n_clusters%d_n_samples%d_%s_%s" %(tica_dir, n_clusters, n_samples, sampling_method, precision)
mmgbsa_csv = "%s/mmgbsa_top_%d.csv" %(analysis_dir, n_mmgbsa)
mmgbsa_pdf = "%s/pnas_vs_mmgbsa.pdf" %analysis_dir
aggregate_mmgbsa = "%s/aggregate_mmgbsa.csv" %analysis_dir
aggregate_mmgbsa_joined = "%s/aggregate_mmgbsa_joined.csv" %analysis_dir
aggregate_mmgbsa_pnas_joined = "%s/aggregate_mmgbsa_pnas_joined.csv" %analysis_dir
mmgbsa_z_scores_csv = "%s/mmgbsa_z_scores.csv" %analysis_dir

ref_receptors_dir = "%s/reference_receptors" %base

subgraph_save_base = "%s/msm%d_graph_n_clusters%d_subgraph" %(analysis_dir, msm_lag_time, n_clusters)
degree_save_base = "%s/msm%d_graph_n_clusters%d_subgraph" %(analysis_dir, msm_lag_time, n_clusters)
degree_map_csv = "%s/msm%d_graph_n_clusters%d_degree_map.csv" %(analysis_dir, msm_lag_time, n_clusters)
degree_z_map_csv = "%s/msm%d_graph_n_clusters%d_degree_z_map.csv" %(analysis_dir, msm_lag_time, n_clusters)
aggregate_docking_pnas_degree_z_joined = "%s/aggregate_docking_pnas_msm%d_graph_n_clusters%d_degree_z_map.csv" %(analysis_dir, msm_lag_time, n_clusters)
#top_dock = top_n_scoring_clusters(aggregate_docking_joined, score_type = 1, n = 1000)
#print top_dock

ref_receptors_dir = "%s/reference_receptors" %base
ref_features_dir = "%s/reference_receptors" %features_dir
ref_tica_dir = "%s/reference_receptors" %tica_dir
ref_tica_coords = "%s/refcoords.csv" %ref_tica_dir
graph_file = "%s/msm%d_graph_n_clusters%d.graphml" %(tica_dir, msm_lag_time, n_clusters)

#compute_z_core_degrees_group(G = None, graph_file = graph_file, cluster_ids = top_dock, subgraph_save_base = subgraph_save_base, degree_save_base = degree_save_base, degree_map_csv = degree_map_csv, degree_z_map_csv = degree_z_map_csv)
#combine_csv_list([aggregate_docking_pnas_joined, degree_z_map_csv], aggregate_docking_pnas_degree_z_joined)

pnas_titles =  ["tm6_tm3_dist", "rmsd_npxxy_inactive", "rmsd_npxxy_active", "rmsd_connector_inactive", "rmsd_connector_active"]
pnas_features_dir = analysis_dir
#pnas_features_dir = "%s/pnas_reference_features" %base
whole_trajectory_pnas = "%s/all_pnas_features" %(base)
if not os.path.exists(whole_trajectory_pnas): os.makedirs(whole_trajectory_pnas)
if not os.path.exists(pnas_features_dir): os.makedirs(pnas_features_dir)
#whole_trajectory_pnas = pnas_features_dir
inactive_pnas_distances_dir = "%s/inactive_pnas_distances.h5" %pnas_features_dir
active_pnas_distances_dir = "%s/active_pnas_distances.h5" %whole_trajectory_pnas
active_pnas_all_distances_dir = "%s/active_pnas_all_distances.csv" %whole_trajectory_pnas
active_pnas_distances_new_csv = "%s/active_pnas_distances_new.csv" %pnas_features_dir
active_pnas_joined = "%s/active_pnas_joined.csv" %pnas_features_dir
active_pnas_means = "%s/active_pnas_means.csv" %pnas_features_dir
pnas_coords_dir = "%s/pnas_coords.h5" %whole_trajectory_pnas
pnas_coords_csv = "%s/pnas_coords_new.csv" %pnas_features_dir
#r.assign('pnas.coords.csv', pnas_coords_csv)
pnas_all_coords_csv = "%s/pnas_all_coords.csv" %whole_trajectory_pnas
pnas_coords_hexbin_dir = "%s/pnas_coords_figure.pdf" %pnas_features_dir
pnas_coords_co_crystallized_docking_dir = "%s/co_crystallized_docking.pdf" %pnas_features_dir
pnas_coords_active_colors_dir = "%s/pnas_coords_active_colors_figure.pdf" %pnas_features_dir
active_clusters_csv = "%s/active_clusters.csv" %analysis_dir
intermediate_clusters_csv = "%s/intermediate_clusters.csv" %analysis_dir
inactive_clusters_csv = "%s/inactive_clusters.csv" %analysis_dir
tic_residue_csv = "%s/tica_residues.csv" %tica_dir
feature_coefs_csv = "%s/feature_coefs.csv" %tica_dir
duplicated_feature_coefs_csv = "%s/duplicated_feature_coefs.csv" %tica_dir
#standard_clusters_map_file = "%s/tICA_t5_n_components25all_residues_under_cutoff1nm_regularization0pt001/clusters_map_%d_clusters_%s" %(base, n_clusters, sampling_method)
pnas_clusters_averages = "%s/pnas_coords_averages.csv" %analysis_dir
tica_clusters_averages = "%s/tica_coords_averages.csv" %analysis_dir

sasa_file = "%s/sasa_bp.csv" %base

if not os.path.exists(analysis_dir): os.makedirs(analysis_dir)