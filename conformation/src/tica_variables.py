from feature_types import *
import os
import sys
import csv

exacycle = False

def get_base(exacycle=False):
	possible_bases = ["/scratch/users/enf/b2ar_analysis",
										"/home/enf/b2ar_analysis", "/Users/evan/vsp/b2ar_analysis",
										"~/b2ar_analysis"]
	for base in possible_bases:
		if os.path.exists(base):
			if(exacycle): base = "%s/exacycle_data" %base
			return base 

	raise("Where are we?")

def generate_residues_map(csv_map):
	reader = csv.reader(open(csv_map, "rb"))
	residues_map = {}
	for line in reader:
		residues_map[int(line[0])] = int(line[1])
	return residues_map

def make_directory_if_not_exist(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def get_tica_dir(base, is_sparse, lag_time, n_components, feature_types, 
								 wolf_string, shrinkage_string, rho_string):
	tica_dir = "%s/%stICA_t%d_n_components%d%s_regularization%s%s%s" %(base, is_sparse,
																																 lag_time, n_components, 
																																 feature_types, wolf_string, 
																																 shrinkage_string, rho_string)
	make_directory_if_not_exist(tica_dir)
	return tica_dir

def get_ktica_dir(tica_dir, n_components, feature_types, 
									wolf_string, shrinkage_string):
	ktica_dir = "%s/ktICA_n_components%d_random_specified_regularization%s%s" % (tica_dir, 
								n_components, wolf_string, shrinkage_string)
	make_directory_if_not_exist(ktica_dir)
	return ktica_dir
	
def get_landmarks_dir(tica_dir):
	return "%s/landmarks.h5" %tica_dir

def get_analysis_dir(tica_dir, n_clusters, sampling_method):
	analysis_dir = "%s/analysis_n_clusters%d_%s" %(tica_dir, n_clusters, sampling_method)
	make_directory_if_not_exist(analysis_dir)
	return analysis_dir

def get_gmm_dir(tica_dir):
	gmm_dir = "%s/gmm" %tica_dir
	make_directory_if_not_exist(gmm_dir)
	return gmm_dir

def get_rf_dir(tica_dir):
	rf_dir = "%s/rf" %tica_dir
	make_directory_if_not_exist(rf_dir)
	return rf_dir

def get_tica_files(tica_dir, n_clusters):
	clusterer_dir = "%s/clusterer_%dclusters.h5" %(tica_dir, n_clusters)
	msm_model_dir = "%s/msm_model_%d_clusters_t%d.h5" %(tica_dir, n_clusters, msm_lag_time)
	macrostate_dir = "%s/msm_modeL%d_clusters_t%d_macrostate_states%d" %(tica_dir, n_clusters, msm_lag_time,  n_macrostates)
	features_known = "%s/features_known" %base
	model_dir = tica_dir
	projected_features_dir = "%s/phi_psi_chi2_allprot_projected.h5" %(tica_dir)
	projection_operator_dir = "%s/phi_psi_chi2_allprot_tica_coords.h5" %(tica_dir)
	ktica_fit_model_filename = "%s/ktica_model.h5" %tica_dir
	ktica_projected_data_filename = "%s/ktica_random_specified_projected_coords.npz" %tica_dir
	nystroem_data_filename = "%s/nystroem_random_specified.npz" %tica_dir
	mutual_information_csv = "%s/mutual_information.csv" %tica_dir
	pearson_csv = "%s/pearson.csv" %tica_dir

 return clusterer_dir

def get_feature_files(features_dir):
	standardized_features_dir = "%s_standardized" %features_dir 
	feature_residues_csv = "%s/feature_residues_map.csv" %features_dir 
	feature_residues_pkl = "%s/feature_residues.pkl" %features_dir
	contact_csv = "%s/contact_residue_pairs.csv" %features_dir
	ref_features_dir = "%s/reference_receptors" %features_dir

	return (standardized_features_dir, feature_residues_csv, feature_residues_pkl,
					contact_csv, ref_features_dir)

def get_analysis_files(analysis_dir):
	kmeans_csv = "%s/kmeans_csv" %analysis_dir
	tica_coords_csv = "%s/tica_coords.csv" %analysis_dir
	features_csv = "%s/features.csv" %analysis_dir
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
	active_clusters_csv = "%s/active_clusters.csv" %analysis_dir
	intermediate_clusters_csv = "%s/intermediate_clusters.csv" %analysis_dir
	inactive_clusters_csv = "%s/inactive_clusters.csv" %analysis_dir
	pnas_clusters_averages = "%s/pnas_coords_averages.csv" %analysis_dir
	tica_clusters_averages = "%s/tica_coords_averages.csv" %analysis_dir
	tica_classes_csv = "%s/tica_classes.csv" %analysis_dir
	tica_samples_csv = "%s/tica_samples.csv" %analysis_dir	
	subgraph_save_base = "%s/msm%d_graph_n_clusters%d_subgraph" %(analysis_dir, msm_lag_time, n_clusters)
	degree_save_base = "%s/msm%d_graph_n_clusters%d_subgraph" %(analysis_dir, msm_lag_time, n_clusters)
	degree_map_csv = "%s/msm%d_graph_n_clusters%d_degree_map.csv" %(analysis_dir, msm_lag_time, n_clusters)
	degree_z_map_csv = "%s/msm%d_graph_n_clusters%d_degree_z_map.csv" %(analysis_dir, msm_lag_time, n_clusters)
	aggregate_docking_pnas_degree_z_joined = "%s/aggregate_docking_pnas_msm%d_graph_n_clusters%d_degree_z_map.csv" %(analysis_dir, msm_lag_time, n_clusters)
	tic_residue_csv = "%s/tica_residues.csv" %tica_dir
	feature_coefs_csv = "%s/feature_coefs.csv" %tica_dir
	duplicated_feature_coefs_csv = "%s/duplicated_feature_coefs.csv" %tica_dir

def get_pnas_files(whole_trajectory_pnas, pnas_features_dir):
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

def get_features_dir(base, feature_types):
	features_dir = "%s/features%s" %(base,feature_types)
	make_directory_if_not_exist(features_dir)
	return features_dir

def get_base_files(base):
	active_ref_dir = "%s/3P0G_pymol_prepped.pdb" %base
	inactive_ref_dir = "%s/2RH1_prepped.pdb" %base
	simulation_ref_dir = "%s/A-00_protein_BIA.pdb" %base
	scripts_dir = "%s/scripts" %base
	ligand_dir = "%s/ligprep_2/ligprep_2-out.maegz" %base
	agonist_dir = "%s/b2ar_full_agonists" %base
	inverse_agonist_dir = "%s/b2ar_inverse_agonists" %base
	ref_receptors_dir = "%s/reference_receptors" %base
	whole_trajectory_pnas = "%s/all_pnas_features" %(base)
	sasa_file = "%s/sasa_bp.csv" %base

def get_graph_file(tica_dir, msm_lag_time, n_clusters):
	graph_file = "%s/msm%d_graph_n_clusters%d.graphml" %(tica_dir, msm_lag_time, n_clusters)
	return graph_file	

def get_script_dir(scripts_dir):s
	script_dir = "%s/pymol_rmsd.py" %scripts_dir
	pymol_fixpdb_dir = "%s/pymol_fixpdb.py" %scripts_dir

def get_docking_dirs(tica_dir, n_clusters, n_components, n_samples, sampling_method):
	save_dir = "%s/clusters%d_n_components%d_n_samples%d_%s" %(tica_dir, n_clusters, n_components, n_samples, sampling_method)
	reimaged_dir = "%s/clusters%d_n_components%d_n_samples%d_%s_reimaged" %(tica_dir, n_clusters, n_components, n_samples, sampling_method)
	mae_dir = reimaged_dir
	combined_reimaged_dir = "%s/combined.h5" %reimaged_dir
	grid_dir = "%s/grids_n_clusters%d_n_samples%d_%s" %(tica_dir, n_clusters, n_samples, sampling_method)
	docking_dir = "%s/docking_n_clusters%d_n_samples%d_%s_%s" %(tica_dir, n_clusters, n_samples, sampling_method, precision)

def get_ref_tica_dirs(tica_dir):
	ref_tica_dir = "%s/reference_receptors" %tica_dir
	make_directory_if_not_exist(ref_tica_dir)
	ref_tica_coords = "%s/refcoords.csv" %ref_tica_dir
	return ref_tica_dir, ref_tica_coords

def get_common_residues(residues_map_csv, contact_residues):
	residues_map = generate_residues_map(residues_map_csv)
	contact_residues = [res for res in contact_residues if res.resSeq in residues_map.keys()]

def get_residues_map_csv(base):
	residues_map_csv = "%s/exacycle_data/residues_map.csv" %base
	return(residues_map_csv)

def get_trajectory_info(exacycle, base):
	if exacycle:
	traj_dir = "%s/exacycle_data/b2ar3p0g2rh1_bi/Trajectories" %base
	traj_ext = ".lh5"
	feature_parallel = True
else:
	traj_dir = "%s/subsampled_reimaged_amber" %base
	traj_ext = ".h5"
	feature_parallel = False 


base = get_base(exacycle)

traj_dir, traj_ext, feature_parallel = get_trajectory_info(exacycle, base)



n_clusters = 1000
lag_time = 5
if exacycle:
	lag_time *= 2




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

ktica = False
k_tica_components = 25
landmark_subsample=5
k_tica_sparse = False
k_tica_shrinkage = 0.01
k_tica_shrinkage_string = ""
k_tica_wolf = True
if k_tica_wolf: 
	k_tica_wolf_string = "_wolf_"
	k_tica_shrinkage = None
	k_tica_shrinkage_string = ""
else:
	k_tica_wolf_string = ""
	k_tica_regularization = 0.25
	k_tica_regularization_string = "0pt25"

msm_lag_time = 5
n_components = 25
n_samples = 10
n_macrostates = 25
n_mmgbsa=100
gmm_max_components = 10
n_trees = 100

landmarks_dir = get_landmarks_dir(tica_dir)
analysis_dir = get_analysis_dir(tica_dir, n_clusters, sampling_method)
gmm_dir = get_gmm_dir(tica_dir)
rf_dir = get_rf_dir(tica_dir)

features_dir = get_features_dir(base, feature_types)

ref_tica_dir, ref_tica_coords = get_ref_tica_dirs(tica_dir)

graph_file = get_graph_file(tica_dir, msm_lag_time, n_clusters)

pnas_titles =  ["tm6_tm3_dist", "rmsd_npxxy_inactive", "rmsd_npxxy_active", "rmsd_connector_inactive", "rmsd_connector_active"]
pnas_features_dir = analysis_dir
#pnas_features_dir = ref_receptors_dir


#standard_clusters_map_file = "%s/tICA_t5_n_components25all_residues_under_cutoff1nm_regularization0pt001/clusters_map_%d_clusters_%s" %(base, n_clusters, sampling_method)



if not os.path.exists(analysis_dir): os.makedirs(analysis_dir)
