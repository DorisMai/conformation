import numpy as np
from residue import *

def _compute_centroid(coordinates):
  """given molecule, an instance of class PDB, compute the x,y,z centroid of that molecule"""

  centroid = np.mean(coordinates, axis=0)
  return(centroid)

"""following two functions adapted from:
http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
"""

def _unit_vector(vector):
  """ Returns the unit vector of the vector.  """
  return vector / np.linalg.norm(vector)


def _angle_between(vector_i, vector_j):
  """ Returns the angle in radians between vectors "vector_i" and "vector_j"::
      >>> _angle_between((1, 0, 0), (0, 1, 0))
      1.5707963267948966
      >>> _angle_between((1, 0, 0), (1, 0, 0))
      0.0
      >>> _angle_between((1, 0, 0), (-1, 0, 0))
      3.141592653589793
  """
  vector_i_u = _unit_vector(vector_i)
  vector_j_u = _unit_vector(vector_j)
  angle = np.arccos(np.dot(vector_i_u, vector_j_u))
  if np.isnan(angle):
    if (vector_i_u == vector_j_u).all():
      return 0.0
    else:
      return np.pi
  return angle

def _compute_ring_center(mol, ring):
  ring_xyz = np.zeros((len(ring._path), 3))
  for i, atom_idx in enumerate(ring._path):
    atom = mol.GetAtom(atom_idx)
    ring_xyz[i, :] = [atom.x(), atom.y(), atom.z()]
  ring_centroid = _compute_centroid(ring_xyz)
  return ring_centroid

def _ring_normal(xyz):
  v1 = xyz[1,:] - xyz[0,:]
  v2 = xyz[2,:] - xyz[0,:]
  normal = np.cross(v1, v2)
  return(normal)

def _compute_ring_normal(mol, ring):
  points = np.zeros((3,3))
  for i, atom_idx in enumerate(ring._path):
    if i == 3: break
    atom = mol.GetAtom(atom_idx)
    points[i,:] = [atom.x(), atom.y(), atom.z()]

  v1 = points[1,:] - points[0,:]
  v2 = points[2,:] - points[0,:]
  normal = np.cross(v1, v2)
  return normal

def _is_pi_parallel(xyz_i, xyz_j):
  center_i =_compute_centroid(xyz_i)
  center_j = _compute_centroid(xyz_j)

  normal_i = _ring_normal(xyz_i)
  normal_j = _ring_normal(xyz_j)

  dist = np.linalg.norm(center_i - center_j)
  angle = _angle_between(normal_i, normal_j) * 180 / np.pi
  if ((np.abs(angle) < 30.0) 
       or (np.abs(angle) > 150.0 and np.abs(angle) < 210.0)
       or (np.abs(angle) > 330.0 and np.abs(angle) < 360.0)):
    if dist < 0.44:
      return True
  return False

def _is_cation_pi(cation_xyz, ring_xyz):
  ring_center = _compute_centroid(ring_xyz)
  ring_normal = _ring_normal(ring_xyz)

  cation_to_ring_vec = cation_xyz - ring_center
  dist = np.linalg.norm(cation_to_ring_vec)
  angle = _angle_between(cation_to_ring_vec, ring_normal) * 180. / np.pi
  if dist < 0.65:
    if ((np.abs(angle) < 30.0) or
        (np.abs(angle) > 150.0 and np.abs(angle) < 210.0) or
        (np.abs(angle) > 330.0 and np.abs(angle) < 360.0)):
      return True 

  return False
    

def _is_pi_t(xyz_i, xyz_j):
  center_i =_compute_centroid(xyz_i)
  center_j = _compute_centroid(xyz_j)

  normal_i = _ring_normal(xyz_i)
  normal_j = _ring_normal(xyz_j)

  dist = np.linalg.norm(center_i - center_j)
  angle = _angle_between(normal_i, normal_j) * 180 / np.pi

  if ((np.abs(angle) > 60.0 and np.abs(angle) < 120.0) 
       or (np.abs(angle) > 240.0 and np.abs(angle) < 300.0)):
    if dist < 0.65:
      return True

  return False

def compute_cation_pi(cation_atom, ring_atoms, traj):
  pi_features = np.zeros(traj.n_frames)
  for f in range(0, traj.n_frames):
    cation_xyz = traj.xyz[f, cation_atom]
    ring_xyz = traj.xyz[f, ring_atoms]
    pi_features[f] = int(_is_cation_pi(cation_xyz, ring_xyz))
  return pi_features

def compute_pi_parallel(atoms_i, atoms_j, traj):
  pi_features = np.zeros(traj.n_frames)
  for f in range(0, traj.n_frames):
    xyz_i = traj.xyz[f, atoms_i]
    xyz_j = traj.xyz[f, atoms_j]
    pi_features[f] = int(_is_pi_parallel(xyz_i, xyz_j))
  return pi_features

def compute_pi_t(atoms_i, atoms_j, traj):
  pi_features = np.zeros(traj.n_frames)
  for f in range(0, traj.n_frames):
    xyz_i = traj.xyz[f, atoms_i]
    xyz_j = traj.xyz[f, atoms_j]
    pi_features[f] = int(_is_pi_t(xyz_i, xyz_j))
  return pi_features


def compute_pi_interactions(traj, pi_pi_ring_pairs, cation_pi_pairs):
  featurized_traj = np.zeros((traj.n_frames, len(pi_pi_ring_pairs)*2 
                              + len(cation_pi_pairs)))

  for p, ring_pair in enumerate(pi_pi_ring_pairs):
    atoms_i = [convert_atom_to_mdtraj_index(traj.topology, atom)
               for atom in ring_pair[0]]
    atoms_j = [convert_atom_to_mdtraj_index(traj.topology, atom)
               for atom in ring_pair[1]]

    pi_parallel_features = compute_pi_parallel(atoms_i, atoms_j, traj)
    pi_t_features = compute_pi_t(atoms_i, atoms_j, traj)
    
    featurized_traj[:,p] = pi_parallel_features
    featurized_traj[:,len(pi_pi_ring_pairs)+p] = pi_t_features

  for p, ring_pair in enumerate(cation_pi_pairs):
    cation_atom = convert_atom_to_mdtraj_index(traj.topology, ring_pair[0])
    ring_atoms = [convert_atom_to_mdtraj_index(traj.topology, atom)
                  for atom in ring_pair[1]]
    cation_pi_features = compute_cation_pi(cation_atom, ring_atoms, traj)
    featurized_traj[:, len(pi_pi_ring_pairs)*2 + p] = cation_pi_features

  return featurized_traj
 