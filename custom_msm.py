from msmbuilder.msm import MarkovStateModel
from msmbuilder.msm import ContinuousTimeMSM
from msmbuilder.utils import verbosedump, verboseload
import networkx as nx
import numpy as np
from msmbuilder.msm import implied_timescales
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io_functions import *
from analysis import *
from msmbuilder import lumping
import re

def plot_timescales(clusterer_dir, n_clusters, tica_dir):
	clusterer = verboseload(clusterer_dir)
	print clusterer
	sequences = clusterer.labels_
	#print(sequences)
	#lag_times = list(np.arange(1,150,5))
	lag_times = [1, 4, 8, 12, 16]
	n_timescales = 10

	msm_timescales = implied_timescales(sequences, lag_times, n_timescales=n_timescales, msm=MarkovStateModel(verbose=True))
	print msm_timescales
	implied_timescales(sequences, lag_times, n_timescales=n_timescales, msm=MarkovStateModel(verbose=False))

	for i in range(n_timescales):
		plt.plot(lag_times, msm_timescales[:,i])
	plt.semilogy()
	pp = PdfPages("%s/n_clusters%d_implied_timescales.pdf" %(tica_dir, n_clusters))
	pp.savefig()
	pp.close()

def build_msm(clusterer_dir, lag_time):
	clusterer = verboseload(clusterer_dir)
	n_clusters = np.shape(clusterer.cluster_centers_)[0]
	labels = clusterer.labels_
	msm_modeler = MarkovStateModel(lag_time=lag_time)
	print("fitting msm to trajectories with %d clusters and lag_time %d" %(n_clusters, lag_time))
	msm_modeler.fit_transform(labels)
	verbosedump(msm_modeler, "/scratch/users/enf/b2ar_analysis/msm_model_%d_clusters_t%d" %(n_clusters, lag_time))
	print("fitted msm to trajectories with %d states" %(msm_modeler.n_states_))
	'''
	#np.savetxt("/scratch/users/enf/b2ar_analysis/msm_%d_clusters_t%d_transmat.csv" %(n_clusters, lag_time), msm_modeler.transmat_, delimiter=",")
	#G = nx.from_numpy_matrix(msm_modeler.transmat_)
	#nx.write_edgelist(G, "/scratch/users/enf/b2ar_analysis/msm_%d_clusters_t%d_edgelist" %(n_clusters, lag_time), msm_modeler.transmat_, delimiter=",")
	transmat = msm_modeler.transmat_

	mapping = msm_modeler.mapping_
	inv_mapping = {v: k for k, v in mapping.items()}

	edges = open("/scratch/users/enf/b2ar_analysis/msm_%d_clusters_t%d_edgelist.csv" %(n_clusters, lag_time), "wb")
	for i in range(0, msm_modeler.n_states_):
		if i == 0:
			for j in range(0, msm_modeler.n_states_):
				edges.write(";")
				edges.write("%d" %inv_mapping[j])
			edges.write("\n")

		edges.write("%d" %(inv_mapping[i]))
		for j in range(0, msm_modeler.n_states_):
			prob = transmat[i][j]
			edges.write(";")
			if prob > 0.000001:
				edges.write("%f" %prob)
			else:
				edges.write("0")
		edges.write("\n")
	edges.close()
	'''

def construct_graph(msm_modeler_dir, clusterer_dir, n_clusters, tica_lag_time, msm_lag_time, graph_file, inactive = None, active = None, pnas_clusters_averages = None, tica_clusters_averages = None, docking=None, macrostate = None):
	clusterer = verboseload(clusterer_dir)
	n_clusters = np.shape(clusterer.cluster_centers_)[0]
	labels = clusterer.labels_
	if not os.path.exists(msm_modeler_dir):
		msm_modeler = MarkovStateModel(lag_time=msm_lag_time, n_timescales = 5, sliding_window = True, verbose = True)
		print("fitting msm to trajectories with %d clusters and lag_time %d" %(n_clusters, msm_lag_time))
		msm_modeler.fit_transform(labels)
		verbosedump(msm_modeler, msm_modeler_dir)
	else:
		msm_modeler = verboseload(msm_modeler_dir)
	graph = nx.DiGraph()
	mapping = msm_modeler.mapping_
	inv_mapping = {v: k for k, v in mapping.items()}
	transmat = msm_modeler.transmat_

	for i in range(0, msm_modeler.n_states_):
		for j in range(0, msm_modeler.n_states_):
			prob = transmat[i][j]
			if prob > 0.0:
				if prob < 0.001: prob = 0.001
				original_i = inv_mapping[i]
				original_j = inv_mapping[j]
				graph.add_edge(original_i, original_j, prob = float(prob), inverse_prob = 1.0 / float(prob), weight = float(prob))

	print(graph.number_of_nodes())

	if inactive is not None:
		scores = convert_csv_to_map_nocombine(inactive)
		for cluster in scores.keys():
			cluster_id = int(cluster[7:len(cluster)])
			if cluster_id in graph.nodes():
				score = scores[cluster][0]
				graph.node[cluster_id]["inactive_pnas"] = score

	if active is not None:
		scores = convert_csv_to_map_nocombine(active)
		for cluster in scores.keys():
			cluster_id = int(re.search(r'\d+',cluster).group()) 
			if cluster_id in graph.nodes():
				score = scores[cluster][0]
				graph.node[cluster_id]["active_pnas"] = score

	if pnas_clusters_averages is not None:
		scores = convert_csv_to_map_nocombine(pnas_clusters_averages)
		for cluster in scores.keys():
			cluster_id = int(re.search(r'\d+',cluster).group()) 
			if cluster_id in graph.nodes():
				graph.node[cluster_id]["tm6_tm3_dist"] = scores[cluster][0]
				graph.node[cluster_id]["rmsd_npxxy_active"] = scores[cluster][2]
				graph.node[cluster_id]["rmsd_connector_active"] = scores[cluster][4]

	if tica_clusters_averages is not None:
		scores = convert_csv_to_map_nocombine(tica_clusters_averages)
		for cluster in scores.keys():
			cluster_id = int(re.search(r'\d+',cluster).group()) 
			if cluster_id in graph.nodes():
				for i in range(0,len(scores[cluster])):
					graph.node[cluster_id]["tIC%d" %(i+1)] = scores[cluster][i]

	if docking is not None:
		scores = convert_csv_to_map_nocombine(docking)
		for cluster in scores.keys():
			cluster_id = int(cluster[7:len(cluster)])
			if cluster_id in graph.nodes():
				score = scores[cluster][0]
				graph.node[cluster_id]["docking"] = score

	if macrostate is not None:
		macromodel = verboseload(macrostate)
		for cluster_id in range(0, n_clusters):
			if cluster_id in graph.nodes():
				microstate_cluster_id = mapping[cluster_id]
				macrostate_cluster_id = macromodel.microstate_mapping_[microstate_cluster_id]
				#print(macrostate_cluster_id)
				graph.node[cluster_id]["macrostate"] = int(macrostate_cluster_id)

	nx.write_graphml(graph, graph_file)

def compute_subgraphs(graph = None, graph_file = None, save_base = None):
	if graph is None:
		G = nx.read_graphml(graph)
	else:
		G = graph 

	subgraphs = nx.strongly_connected_component_subgraphs(G)
	subgraph_list = []
	for subgraph in subgraphs:
		subgraph_list.append(subgraph)
	subgraphs = subgraph_list
	if save_base is not None:
		for i in range(0,len(subgraphs)):
			subgraph = subgraphs[i]
			graph_file = "%s%d.graphml" %(save_base, i)
			nx.write_graphml(subgraph, graph_file)
	return subgraphs

def compute_z_score(value, mean, std):
	return (value - mean) / std

def find_degree_distributions(subgraphs, save_base):
	subgraph_degrees = []
	for i in range(0, len(subgraphs)):
		subgraph = subgraphs[i]
		in_degrees = np.array(list(subgraph.in_degree(weight = "prob").values()))
		out_degrees = np.array(list(subgraph.out_degree(weight = "prob").values()))
		degrees = in_degrees - out_degrees
		save_file = "%s%d_degrees.csv" %(save_base, i)
		np.savetxt(save_file, degrees, delimiter = ",")
		subgraph_degrees.append(degrees)
	return subgraph_degrees

def compute_z_score_degrees(subgraph, cluster_id):
	degrees = nx.degree(subgraph, weight = "prob")
	degrees = np.array(degrees)
	mean_degree = np.mean(degrees, axis=0)
	std_degree = np.std(degrees, axis = 0)
	return 


def find_subgraph(subgraphs, cluster_id):
	for subgraph in subgraphs:
		if cluster_id in subgraph.nodes():
			return subgraph
	print("That cluster is not in any subgraph!")

def remove_self_edges(G):
	for node in G.nodes():
		G.remove_edge(node, node)
	return G

def compute_z_core_degrees_group(G = None, graph_file = None, cluster_ids = None, subgraph_save_base = None, degree_save_base = None, degree_map_csv = None, degree_z_map_csv = None):
	G = nx.read_graphml(graph_file)
	G = remove_self_edges(G)
	subgraphs = compute_subgraphs(graph = G, save_base = subgraph_save_base)
	subgraph_degrees = find_degree_distributions(subgraphs, degree_save_base)
	degree_map = {}
	degree_z_map = {}

	print(subgraphs[0].nodes())
	if 'cluster' in cluster_ids[0]:
		cluster_ids = [s[7:len(s)] for s in cluster_ids]

	for i in range(0,len(subgraphs)):
		subgraph = subgraphs[i]
		print(subgraph_degrees[i][1:10])
		mean = np.mean(subgraph_degrees[i], axis = 0)
		print mean
		std = np.std(subgraph_degrees[i], axis = 0)
		print std
		for cluster_id in cluster_ids:
			if cluster_id in subgraph.nodes():
				degree = G.in_degree(nbunch = cluster_id, weight = "prob") - G.out_degree(nbunch = cluster_id, weight = "prob")
				degree_map["cluster%s" %cluster_id] = [degree]
				degree_z = (degree - mean) / std
				degree_z_map["cluster%s" %cluster_id] = [degree_z]

	write_map_to_csv(degree_map_csv, degree_map, ["cluster", "degree"])
	write_map_to_csv(degree_z_map_csv, degree_z_map, ["cluster", "z_degree"])


def macrostate_pcca(msm_file, clusterer_file, n_macrostates, macrostate_dir):

	msm = verboseload(msm_file)
	clusterer = verboseload(clusterer_file)

	#pcca = lumping.PCCAPlus.from_msm(msm = msm,n_macrostates = n_macrostates)
	#macrostate_model = MarkovStateModel()
	#macrostate_model.fit(pcca.transform(labels))

	pcca_object = lumping.PCCA(n_macrostates = 10)
	pcca_object.fit(sequences = clusterer.labels_)
	#pcca_object.transform(sequences = clusterer.labels_)
	#macrostate_model = pcca_object.from_msm(msm = msm, n_macrostates = n_macrostates)
	print(pcca_object)
	print(pcca_object.microstate_mapping_)
	verbosedump(pcca_object, macrostate_dir)



def macrostate_bace(msm_file, n_macrosates):
	return

def make_trajectory(msm_file, tica_dir):
	return