import mdtraj as md
import os
import numpy as np
import h5py
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import KMeans
from msmbuilder.msm import MarkovStateModel
from sklearn.pipeline import Pipeline
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.utils import verbosedump, verboseload
from msmbuilder.dataset import dataset
from matplotlib import pyplot as plt
from msmbuilder.cluster import KMedoids

def fit_and_plot(pipeline, trajectories):
    transformed = pipeline.fit_transform(trajectories)
    transformed = np.concatenate(transformed)

    print(('Eiegenvaue sum', pipeline.named_steps['tica'].eigenvalues_.sum()))

    x = transformed[:, 0]
    y = transformed[:, 1]

    plt.axes(axisbg='w')
    plt.grid(False)
    plt.hist2d(x, y, bins=100, cmap='hot_r', norm=LogNorm())
    plt.xlabel('1st tIC')
    plt.ylabel('2nd tIC')
    plt.title('tICA Heatmap (log color scale)')
    plt.colorbar()

dataset = []
trajs = []

traj_dir = "/home/harrigan/data/gpcr/DESRES/DESRES-Trajectory_pnas2011b-H-05-all/pnas2011b-H-05-all"
traj_files = []
'''
if not (os.path.isfile("/home/enf/b2ar_analysis/H-05/%s" %("combined_traj_stride10.h5"))):
	print("traj not loaded yet")
	for traj in os.listdir(traj_dir):
		if traj.endswith(".dcd"):
			traj_files.append("%s/%s" %(traj_dir,traj))
	traj_files.sort()
	traj = md.load(traj_files, top = "/home/harrigan/compute/wetmsm/gpcr/des/system_mae_to_pdb/des_trajs/DESRES-Trajectory_pnas2011b-H-05-all/system.pdb", stride=10)
	traj = traj[0].join(traj[1:])
	traj.save("/home/enf/b2ar_analysis/H-05/%s" %("combined_traj_stride10.h5"))
else:
'''
#print("loading h5 traj")
#traj = md.load("combined_traj_stride10.h5")
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
	features = [np.concatenate(features)]

if not (os.path.isfile("reduced_phi_psi_chi_stride10.h5")):
	print("Fitting tICA model")
	tica_model = tICA(n_components=4)
	fitted_model = tica_model.fit(features)
	reduced_data = fitted_model.transform(features)
	verbosedump(reduced_data, "reduced_phi_psi_chi_stride10.h5")
	print((tica_model.summarize()))
else:
	reduced_data = verboseload("reduced_phi_psi_chi_stride10.h5")

clusterer = KMedoids(n_clusters=9)

clusters = clusterer.fit_transform(reduced_data)[0]

center_locations = []

for i in range(0, len(clusters)):
	print(i)
	for j in range(0, len(clusterer.cluster_centers_)):
		if np.linalg.norm(reduced_data[0][i] - clusterer.cluster_centers_[j]) < 0.001:
			print("found match")
			center_locations.append(i)

print(center_locations)

for center in center_locations:
	frame = md.load_frame("combined_traj_stride10.h5", index=center)
	frame.save_pdb("frame_%d.pdb" %(center))


trajs = np.concatenate(reduced_data)
plt.hexbin(trajs[:,0], trajs[:,1], bins='log', mincnt=1)
plt.show()



