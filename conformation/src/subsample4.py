from PDB_Order_Fixer import PDB_Order_Fixer
import mdtraj as md
import os
import numpy as np
import h5py
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import KMeans

from sklearn.pipeline import Pipeline
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.dataset import dataset

from msmbuilder.cluster import KMedoids
import datetime
import multiprocessing as mp
import glob
import copy
import gc
from functools import partial 
import itertools
import operator
from mdtraj.geometry import dihedral as ManualDihedral
import time
import fileinput
from msmbuilder.cluster import MiniBatchKMedoids
from msmbuilder.cluster import MiniBatchKMeans

import random 
import subprocess
from subprocess import Popen
import sys
from grids import *
from io_functions import *
from custom_clusterer import *
from custom_tica import *
from custom_featurizer import *
from pdb_editing import *
from analysis import *
from io_functions import *
from grids import *
from topology_fixing import *
from subsampling import *
from conversions import *
from custom_msm import *


n_clusters = 1000
lag_time = 10
msm_lag_time = 10
n_components = 5
n_samples = 10
#feature_types = "_switches_tm6"
feature_types = "_switches_npxx_tm6_bp"
n_mmgbsa = 50
#feature_types = ""

switch_residues = [130, 131, 208, 211, 219, 268, 272, 286, 288, 316, 322, 323, 326]
switch_npxx = [130, 131, 208, 211, 219, 268, 272, 286, 288, 316] + list(range(322,328))
tm6_residues = list(range(269, 299))
bp_residues = [82, 86, 93, 106, 110, 113, 114, 117, 118, 164, 191, 192, 193, 195, 199, 200, 203, 206, 208, 286, 289, 290, 293, 305, 308, 309, 312, 316]
dihedral_residues = list(set(switch_npxx + tm6_residues))
sampling_method = "dist"
precision = "SP"

sherlock_base = "/scratch/users/enf/b2ar_analysis"
biox3_base = "/home/enf/b2ar_analysis_sherlock_all"

if os.path.exists(sherlock_base):
	print("we are operating on sherlock")
	base = sherlock_base
elif os.path.exists(biox3_base):
	print("we are operating on biox3")
	base = biox3_base
else:
	print("WHERE ARE WE?")
	sys.exit()

traj_dir = "%s/subsampled_allprot_combined_reimaged" %base
pnas_features_dir = "%s/features_pnas" %base

tica_dir = "%s/tICA_t%d_n_components%d%s" %(base, lag_time, n_components, feature_types)
analysis_dir = "%s/analysis_n_clusters%d" %(tica_dir, n_clusters)
clusterer_dir = "%s/clusterer_%dclusters.h5" %(tica_dir, n_clusters)
kmeans_csv = "%s/kmeans_csv" %analysis_dir
#features_dir = "%s/features_allprot"
features_dir = "%s/features%s" %(base,feature_types)
msm_model_dir = "%s/msm_model_%d_clusters_t%d.h5" %(tica_dir, n_clusters, msm_lag_time)
features_known = "%s/features_known" %base
model_dir = tica_dir
projected_features_dir = "%s/phi_psi_chi2_allprot_projected.h5" %(tica_dir)
save_dir = "%s/clusters%d_n_components%d_n_samples%d_%s" %(tica_dir, n_clusters, n_components, n_samples, sampling_method)
#save_dir = "%s/reorder_test" %tica_dir
reimaged_dir = "%s/clusters%d_n_components%d_n_samples%d_%s_reimaged" %(tica_dir, n_clusters, n_components, n_samples, sampling_method)
combined_reimaged_dir = "%s/combined.h5" %reimaged_dir
tica_coords_csv = "%s/tica_coords.csv" %analysis_dir
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
aggregate_docking_joined = "%s/aggregate_docking_joined.csv" %analysis_dir
docking_pnas_joined = "%s/docking_pnas_joined.csv" %analysis_dir
aggregate_docking_pnas = "%s/aggregate_docking_pnas.csv" %analysis_dir
aggregate_docking_pnas_joined = "%s/aggregate_docking_pnas_joined.csv" %analysis_dir
docking_multiple_ligands = "%s/all_docking_combined.csv" %analysis_dir
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

subgraph_save_base = "%s/msm%d_graph_n_clusters%d_subgraph" %(analysis_dir, msm_lag_time, n_clusters)
degree_save_base = "%s/msm%d_graph_n_clusters%d_subgraph" %(analysis_dir, msm_lag_time, n_clusters)
degree_map_csv = "%s/msm%d_graph_n_clusters%d_degree_map.csv" %(analysis_dir, msm_lag_time, n_clusters)
degree_z_map_csv = "%s/msm%d_graph_n_clusters%d_degree_z_map.csv" %(analysis_dir, msm_lag_time, n_clusters)
aggregate_docking_pnas_degree_z_joined = "%s/aggregate_docking_pnas_msm%d_graph_n_clusters%d_degree_z_map.csv" %(analysis_dir, msm_lag_time, n_clusters)
top_dock = top_n_scoring_clusters(aggregate_docking_joined, score_type = 1, n = 1000)
#print top_dock

graph_file = "%s/msm%d_graph_n_clusters%d.graphml" %(tica_dir, msm_lag_time, n_clusters)

compute_z_core_degrees_group(G = None, graph_file = graph_file, cluster_ids = top_dock, subgraph_save_base = subgraph_save_base, degree_save_base = degree_save_base, degree_map_csv = degree_map_csv, degree_z_map_csv = degree_z_map_csv)
combine_csv_list([aggregate_docking_pnas_joined, degree_z_map_csv], aggregate_docking_pnas_degree_z_joined)

pnas_features_dir = analysis_dir
inactive_pnas_distances_dir = "%s/inactive_pnas_distances.h5" %pnas_features_dir
active_pnas_distances_dir = "%s/active_pnas_distances.h5" %pnas_features_dir
active_pnas_distances_new_csv = "%s/active_pnas_distances.csv" %pnas_features_dir
pnas_coords_dir = "%s/pnas_coords.h5" %pnas_features_dir
pnas_coords_hexbin_dir = "%s/pnas_coords_figure.pdf" %pnas_features_dir
pnas_coords_co_crystallized_docking_dir = "%s/co_crystallized_docking.pdf" %pnas_features_dir
pnas_coords_active_colors_dir = "%s/pnas_coords_active_colors_figure.pdf" %pnas_features_dir

#mae_dir = "%s/docking_test" %tica_dir
mae_dir = reimaged_dir
grid_center = "64.4, 16.9, 11.99"

inverse_ligands = get_ligands(inverse_agonist_dir)
agonist_ligands = get_ligands(agonist_dir)

#inactive = md.load_frame(inactive_ref_dir, top = inactive_ref_dir, index = 0)
#active = md.load_frame(active_ref_dir, top = active_ref_dir, index = 0)

#compute_pnas_distance(traj_dir, inactive, active)

'''
precision = "SP"
tica_dir = "/scratch/users/enf/b2ar_analysis/reference_docking"
docking_dir = "%s/docking_%s" %(tica_dir, precision)
grid_dir = "%s/reference_grids" %tica_dir
ligands_dir = agonist_dir
docking_multiple_ligands = "%s/all_docking_combined.csv" %docking_dir
dock_ligands_and_receptors(grid_dir, docking_dir, ligands_dir, precision = precision, ext = "-out.maegz", chosen_ligands = False, chosen_receptors = False, parallel = True)
analyze_docking_results_multiple(docking_dir, precision = precision)
compute_aggregate_docking_scores(docking_multiple_ligands, docking_dir)
'''
'''
precision = "SP"
tica_dir = "/scratch/users/enf/b2ar_analysis/reference_docking"
docking_dir = "%s/docking_%s" %(tica_dir, precision)
grid_dir = "%s/reference_grids" %tica_dir
ligands_dir = inverse_agonist_dir
docking_multiple_ligands = "%s/all_docking_combined.csv" %docking_dir
dock_ligands_and_receptors(grid_dir, docking_dir, ligands_dir, precision = precision, ext = "-out.maegz", chosen_ligands = False, chosen_receptors = False, parallel = True)
analyze_docking_results_multiple(docking_dir, precision = precision, summary = docking_multiple_ligands)
#compute_aggregate_docking_scores(docking_multiple_ligands, docking_dir)
'''

'''
top_dock = "%s/top50_docking.txt" %tica_dir
to_dock = []
dockfile = open(top_dock, "rb")
for line in dockfile.readlines():
	line = line.split('\n')[0]
	to_dock.append(line)
print to_dock
'''
#featurize_pnas_distance("%s/reference_receptors" %base, "%s/reference_receptors" %base, ".pdb", inactive_ref_dir, active_ref_dir, "%s/reference_receptors/inactive_pnas_distances_ref.h5" %base, "%s/reference_receptors/active_pnas_distances_ref.h5" %base, "%s/reference_receptors/ref_coords.h5" %base, scale = 1.0)
#convert_matrix_to_map("%s/reference_receptors/ref_coords.h5" %base, "%s/reference_receptors" %base, ".pdb", "%s/reference_receptors/ref_coords.csv" %base)

####Featurize with PNAS distances and coords, 2D####
#featurize_pnas_distance(traj_dir, pnas_features_dir, inactive_ref_dir, active_ref_dir, inactive_pnas_distances_dir, active_pnas_distances_dir, pnas_coords_dir)
####
#featurize_pnas_distance(reimaged_dir, features_dir, ".pdb", inactive_ref_dir, active_ref_dir, inactive_pnas_distances_dir, active_pnas_distances_dir, pnas_coords_dir)
#plot_hex(pnas_coords_dir, pnas_coords_hexbin_dir)
#plot_col(pnas_coords_dir, pnas_coords_active_colors_dir, active_pnas_distances_dir)

#to_dock = ["cluster0_sample1", "cluster0_sample2", "cluster0_sample3"]

#featurize_custom(traj_dir, features_dir = features_dir, dihedral_residues = dihedral_residues, dihedral_types = ["phi", "psi", "chi1", "chi2"], contact_residues = bp_residues)
#fit_and_transform(features_directory = features_dir, model_dir = tica_dir, stride=5, lag_time = lag_time, n_components = n_components)
#cluster_minikmeans(tica_dir, projected_features_dir, traj_dir, n_clusters, lag_time)
#cluster_kmeans(tica_dir, projected_features_dir, traj_dir, n_clusters, lag_time)
#plot_tica_and_clusters(tica_dir, projected_features_dir, clusterer_dir, lag_time)
#sample_clusters(clusterer_dir, projected_features_dir, traj_dir, save_dir, n_samples, method = sampling_method)
#dist_to_means(clusterer_dir, projected_features_dir, n_samples = n_samples, n_components = n_components, tica_coords_csv = tica_coords_csv, kmeans_csv = kmeans_csv)
#reverse_sign_csv(docking_joined)
plot_all_tics(tica_dir, projected_features_dir, clusterer_dir, lag_time)

#pymol_fixpdb(save_dir, pymol_fixpdb_dir)
#reorder(save_dir)
#reimage_trajs(save_dir, ext = ".pdb")
#remove_ter(reimaged_dir)
#reorder(reimaged_dir)
#pprep(mae_dir)
#rmsd_pymol(reimaged_dir, inactive_ref_dir, script_dir, inactive_rmsd_dir)
#rmsd_pymol(reimaged_dir, active_ref_dir, script_dir, active_rmsd_dir)
#analyze_docking_results(docking_dir)
#pnas_distances(reimaged_dir, inactive_ref_dir, active_ref_dir)
#analyze_rmsds(inactive_rmsd_dir, active_rmsd_dir, inactive_pnas_dir, active_pnas_dir, combined_file, analysis_file)
#plot_pnas_vs_docking(docking_summary, pnas_coords, "%s/pnas_vs_docking.pdf" %docking_dir)

#pnas_distance(traj_file, inactive_file, active_file)


#plot_all_tics_and_clusters(tica_dir, projected_features_dir, clusterer_dir, lag_time)
#pprep(mae_dir)
#generate_grids(mae_dir, grid_center, tica_dir, n_clusters, n_samples, grid_dir)
#dock_conformations(grid_dir = grid_dir, docking_dir = docking_dir, ligand_dir = ligand_dir, chosen_jobs = False, precision = precision)
#analyze_docking_results(docking_dir, "BI", "SP", docking_summary)
#combine_csv_list([docking_summary, active_pnas_dir], docking_distances_file)

#docking_joined_map = convert_csv_to_joined_map(docking_summary, docking_joined)[0]
#pnas_joined_map = convert_csv_to_joined_map(active_pnas_dir, active_pnas_joined)[0]
#inactive_pnas_joined_map = convert_csv_to_joined_map(inactive_pnas_dir, inactive_pnas_joined)[0]

#docking_averages = calc_mean(docking_joined_map)
#pnas_averages = calc_mean(pnas_joined_map)

#write_map_to_csv(docking_joined, docking_averages, ["cluster", "mean_docking_score"])
#write_map_to_csv(active_pnas_joined, pnas_averages, ["cluster", "pnas_averages"])

#combine_csv_list([docking_joined, active_pnas_joined], docking_pnas_joined)

#top_n =  top_n_scoring_samples(aggregate_docking_joined, score_type = "mean_aggregate_docking_z_score", n = n_mmgbsa, n_samples = n_samples)
#print top_n

#plot_timescales(clusterer_dir, n_clusters, tica_dir)
#build_msm(clusterer_dir, msm_lag_time)
#construct_graph(msm_model_dir, clusterer_dir, n_clusters, lag_time, msm_lag_time, graph_file, inactive = inactive_pnas_joined, active = active_pnas_joined, docking=aggregate_docking_joined)


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

#inverse_ligands = get_ligands(inverse_agonist_dir)
#agonist_ligands = get_ligands(agonist_dir)[2:5]
#prepare_ligands(inverse_agonist_dir, ext = ".sdf")
#dock_ligands_and_receptors(grid_dir, docking_dir, agonist_dir, precision = precision, ext = "-out.maegz", chosen_ligands = False, chosen_receptors = False)
#analyze_docking_results_multiple(docking_dir, precision = "SP", ligands = (inverse_ligands + agonist_ligands), summary = docking_multiple_ligands)
#compute_aggregate_scores(docking_multiple_ligands, inverse_agonists = inverse_ligands, summary = aggregate_docking, z_scores_csv = docking_z_scores_csv)
#combine_csv_list([docking_joined, active_pnas_joined], aggregate_docking_pnas_joined)

#aggregate_docking_joined_map = convert_csv_to_joined_map(aggregate_docking, aggregate_docking_joined)[0]
#aggregate_docking_means = calc_mean(aggregate_docking_joined_map)
#write_map_to_csv(aggregate_docking_joined, aggregate_docking_means, ["cluster", "mean_aggregate_docking_z_score"])
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


'''important residues for GPCRs:
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



'''
def extract_centers(clusterer_dir, traj_dir, lag_time):
	clusterer = verboseload(clusterer_dir)
	n_clusters = clusterer.n_clusters
	labels = clusterer.labels_
	sample_conformations = []
	visited_clusters = set()
	all_visited = False

	for i in range(0, len(labels)):
		trajectory = labels[i]
		for j in range(0, len(trajectory)):
			label = trajectory[j]
			if label not in visited_clusters:
				sample_conformations.append((label,i,j))
				visited_clusters.add(label)
				if len(visited_clusters) == n_clusters:
					print("sampled all clusters")
					all_visited = True
					break
		if all_visited == True: break

	trajectories = get_trajectory_files(traj_dir)

	for cluster in sample_conformations:
		cluster_id = cluster[0]
		traj = trajectories[cluster[1]]
		frame = cluster[2]

		conformation = md.load_frame(traj,index=frame)

		save_dir = "%s/%d_clusters_t%d" %(n_clusters, lag_time)
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		conformation.save_pdb("%s/%d.pdb" %(save_dir, cluster_id))
'''

'''
For each subdirectory i in DESRES:
	For each subdir j in i: 
		get list of all dcd files
		split subdirectory to find the correct directory to save the file
		apply function read_and_save_traj on each dcd file
'''

'''

traj_dir = "/home/harrigan/data/gpcr/DESRES/DESRES-Trajectory_pnas2011b-H-05-all/pnas2011b-H-05-all"
traj_files = get_trajectory_files(traj_dir)[0:5]

a = datetime.datetime.now().replace(microsecond=0)
trajectories = map(read_trajectory, traj_files)
#trajectories = []
#for traj_file in traj_files:
#	trajectories.append(read_trajectory(traj_file))
b = datetime.datetime.now().replace(microsecond=0)
print(b-a)
'''


'''
dataset = []
trajs = []

traj_dir = "/home/harrigan/data/gpcr/DESRES/DESRES-Trajectory_pnas2011b-H-05-all/pnas2011b-H-05-all"
traj_files = []

if not (os.path.isfile("%s/H-05/%s" %("combined_traj_stride10.h5"))):
	print("traj not loaded yet")
	for traj in os.listdir(traj_dir):
		if traj.endswith(".dcd"):
			traj_files.append("%s/%s" %(traj_dir,traj))
	traj_files.sort()
	traj = md.load(traj_files, top = "/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-H-05-all/system.pdb", stride=10)
	traj = traj[0].join(traj[1:])
	traj.save("%s/H-05/%s" %("combined_traj_stride10.h5"))
else:
	print("loading h5 traj")
	traj = md.load("%s/H-05/%s" %("combined_traj_stride10.h5"))

'''
'''
if not (os.path.isfile("phi_psi_chi2_features_vd_stride10.h5")):
	print("featurizing")
	phi_psi_chi2 = DihedralFeaturizer(types=['phi','psi','chi2'])
	features = phi_psi_chi2.transform(traj_list = traj)
	print("finished featurizing")
	verbosedump(features, "phi_psi_chi2_features_vd_stride10.h5")
else:
	print("loading existing features")
	features = verboseload("phi_psi_chi2_features_vd_stride10.h5")

if not (os.path.isfile("reduced_phi_psi_chi_stride10.h5")):
	print("Fitting tICA model")
	tica_model = tICA(n_components=4)
	fitted_model = tica_model.fit(features)
	reduced_data = fitted_model.transform(features)
	verbosedump(reduced_data, "reduced_phi_psi_chi_stride10.h5")
	print(tica_model.summarize())
else:
	reduced_data = verboseload("reduced_phi_psi_chi_stride10.h5")

clusterer = KMedoids(n_clusters=9)

clusters = clusterer.fit_transform(reduced_data)[0]

center_locations = []

for i in range(0, len(clusters)):
	print i
	for j in range(0, len(clusterer.cluster_centers_)):
		if np.linalg.norm(reduced_data[0][i] - clusterer.cluster_centers_[j]) < 0.001:
			print("found match")
			center_locations.append(i)

print(center_locations)

for center in center_locations:
	frame = md.load_frame("combined_traj_stride10.h5", index=center)
	frame.save_pdb("frame_%d.pdb" %(center))






#trajs = np.concatenate(reduced_data)
#plt.hexbin(trajs[:,0], trajs[:,1], bins='log', mincnt=1)
#plt.show()
'''


