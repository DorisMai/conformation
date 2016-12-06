import pandas as pd

def convert_residue_pairs_to_mdtraj_indices(top, residue_pairs):
  resIndices = []
  for pair in residue_pairs:
    indices = [r.index for r in top.residues if pair.residue_i.is_mdtraj_res_equivalent(r) and not r.is_water]
    if len(indices) == 0:
      print(("FATAL: No residues in trajectory for residue %d chain %s" % (pair.residue_i.resSeq, pair.residue_i.chain_id)))
      return None
    else:
      ind_i = indices[0]
      for j in indices:
        if j != ind_i: 
          #print("Warning: multiple res objects for residue %d " %resSeq0)
          test_atoms = []
          for r in top.residues:
            if r.index != ind_i:
              continue
            test_atoms.append([str(a) for a in r.atoms])
          if "CB" in test_atoms:
            ind_i = j

    indices = [r.index for r in top.residues if pair.residue_j.is_mdtraj_res_equivalent(r) and not r.is_water]
    if len(indices) == 0:
      print(("FATAL: No residues in trajectory for residue %d chain %s" % (pair.residue_j.resSeq, pair.residue_j.chain_id)))
      return None
    else:
      ind_j = indices[0]
      for j in indices:
        if j != ind_j: 
          test_atoms = []
          for r in top.residues:
            if r.index != ind_j:
              continue
            test_atoms.append([str(a) for a in r.atoms])
          if "CB" in test_atoms:
            ind_j = j

    resIndices.append((ind_i, ind_j))
  print(("looking at %d pairs for trajectory" %len(resIndices)))
  return(resIndices)

def convert_atom_to_mdtraj_index(top, atom):
  matching_atom = [a.index for a in top.atoms if atom.is_mdtraj_atom_equivalent(a)][0]
  return matching_atom

def convert_residue_to_mdtraj_index(top, residue):
  indices = [r.index for r in top.residues if residue.is_mdtraj_res_equivalent(r) and not r.is_water]
  if len(indices) == 0:
    print(("FATAL: No residues in trajectory for residue %d chain %s" % (residue.resSeq, residue.chain_id)))
    return None
  else:
    ind_i = indices[0]
    for j in indices:
      if j != ind_i: 
        #print("Warning: multiple res objects for residue %d " %resSeq0)
        test_atoms = []
        for r in top.residues:
          if r.index != ind_i:
            continue
          test_atoms.append([str(a) for a in r.atoms])
        if "CB" in test_atoms:
          ind_i = j
  return ind_i 

def convert_residues_to_mdtraj_atom_indices(top, residues):
  mdtraj_resids = [convert_residue_to_mdtraj_index(top, residue) for residue in residues]
  mdtraj_atom_indices = []
  for mdtraj_resid in mdtraj_resids:
    atom_ids = [a.index for a in top.residue(mdtraj_resid).atoms]
    mdtraj_atom_indices += atom_ids
  return mdtraj_atom_indices

def convert_atom_residue_pairs_to_mdtraj_indices(top, atom_residue_pairs):
  atom_residue_index_pairs = []
  for pair in atom_residue_pairs:
    atom_mdtraj_index = convert_atom_to_mdtraj_index(top, pair.residue_i)
    indices = [r.index for r in top.residues if pair.residue_j.is_mdtraj_res_equivalent(r) and not r.is_water]
    if len(indices) == 0:
      print(("FATAL: No residues in trajectory for residue %d chain %s" % (pair.residue_j.resSeq, pair.residue_j.chain_id)))
      return None
    else:
      ind_i = indices[0]
      for j in indices:
        if j != ind_i: 
          test_atoms = []
          for r in top.residues:
            if r.index != ind_i:
              continue
            test_atoms.append([str(a) for a in r.atoms])
          if "CB" in test_atoms:
            ind_i = j
    atom_residue_mdtraj_indices = (atom_mdtraj_index, ind_i)
    atom_residue_index_pairs.append(atom_residue_mdtraj_indices)
  return atom_residue_index_pairs


class Residue(object):
  def __init__(self, resSeq, chain_id=None, chain_name=None, res_name=None, ballosteros_weinstein=None, mean=None, CA=False):
    self.resSeq = resSeq
    self.chain_id = chain_id
    self.chain_name = chain_name
    self.ballosteros_weinstein = ballosteros_weinstein
    self.res_name = res_name
    self.mean = mean
    self.CA = CA

  def __repr__(self):
    name = None
    if self.res_name is not None:
      name = self.res_name
      if hasattr(self, "CA"):
        if self.CA:
          name = "%s_CA" %name
    elif self.chain_id is not None:
      name = "%s%d" % (self.chain_id, self.resSeq) 
    else:
      name = str(self.resSeq)
    if hasattr(self, "mean"):
      if self.mean is not None:
        name = "%s-mean%d" %(name, self.mean)


    return name

  def __eq__(self, other): 
    return self.__dict__ == other.__dict__

  def __hash__(self):
    return hash(self.__repr__())

  def __ne__(self, other):
    return (not self.__eq__(other))

  def __lt__(self, other):
    return str(self) < str(other)

  def is_mdtraj_res_equivalent(self, mdtraj_res):
    if self.resSeq != mdtraj_res.resSeq:
      return False

    if self.chain_id is not None and (str(self.chain_id) != str(mdtraj_res.chain.id)):
      return False

    if self.chain_name is not None and (str(self.chain_name) != str(mdtraj_res.chain.id)):
      return False 

    return True

class Atom(object):
  def __init__(self, atom_id=None, resSeq=None, chain_id=None, atom_name=None, res_name=None, mdtraj_rep=None):
    self.resSeq = resSeq
    self.chain_id = chain_id
    self.atom_name = atom_name
    self.atom_id = atom_id
    self.res_name = res_name
    self.mdtraj_rep = mdtraj_rep

  def __repr__(self):
    if self.mdtraj_rep is None:
      if self.resSeq is None:
        return "%s-%s" %(self.res_name, self.atom_name)
      else:
        return "%s%d-%s" %(self.res_name, self.resSeq, self.atom_name)
    else:
      return self.mdtraj_rep

  def __eq__(self, other):
    return self.__dict__ == other.__dict__

  def __hash__(self):
    return hash(self.__repr__())

  def __ne__(self, other):
    return (not self.__eq__(other))

  def __lt__(self, other):
    return str(self) < str(other)

  def is_mdtraj_atom_equivalent(self, mdtraj_atom):
    if self.resSeq is not None and self.resSeq != mdtraj_atom.residue.resSeq:
      return False
    if self.chain_id is not None and (self.chain_id != mdtraj_atom.residue.chain.id):
      return False
    if self.atom_name is not None and (self.atom_name != mdtraj_atom.name):
      return False
    if self.atom_id is not None and (self.atom_id != mdtraj_atom.index):
      return False
    if self.res_name is not None and (self.res_name != mdtraj_atom.residue.name):
      return False

    return True


class ContactFeature(object):
  def __init__(self, residue_i, residue_j):
    self.residue_i = residue_i
    self.residue_j = residue_j

  def __eq__(self, other): 
    return self.__dict__ == other.__dict__

  def __hash__(self):
    return hash(self.__repr__())

  def __ne__(self, other):
    return (not self.__eq__(other))

  def __lt__(self, other):
    return str(self) < str(other)

  def __repr__(self):
    name = "%s to %s" %(str(self.residue_i), str(self.residue_j))
    return name

class DihedralFeature(object):
  def __init__(self, residue, dihedral_type, trig=None):
    self.residue = residue
    self.dihedral_type = dihedral_type
    self.trig = trig

  def __eq__(self, other): 
    return self.__dict__ == other.__dict__

  def __hash__(self):
    return hash(self.__repr__())

  def __ne__(self, other):
    return (not self.__eq__(other))

  def __lt__(self, other):
    return str(self) < str(other)

  def __repr__(self):
    name = "%s: %s" %(str(self.residue), self.dihedral_type)
    return name

class AromaticFeature(object):
  def __init__(self, residue_i, residue_j, aromatic_type):
    self.residue_i = residue_i
    self.residue_j = residue_j
    self.aromatic_type = aromatic_type

  def __eq__(self, other): 
    return self.__dict__ == other.__dict__

  def __hash__(self):
    return hash(self.__repr__())

  def __lt__(self, other):
    return str(self) < str(other)

  def __ne__(self, other):
    return (not self.__eq__(other))

  def __repr__(self):
    name = "%s to %s: %s" %(str(self.residue_i), str(self.residue_j), self.aromatic_type)
    return name
