import mdtraj as md
from residue import Residue
from mor_h8_tica_config import inactive_dir, active_dir, base, simulation_structure
from get_variable_names import get_common_residues_pkl, find_common_residues

def convert_list_to_resobject_list(contact_residues):
  resobject_list = []
  for residue in contact_residues:
    new_residue = Residue(resSeq = residue[1], chain_id = residue[0])
    resobject_list.append(new_residue)
  return(resobject_list)

#CHOOSE RESIDUES:

bp_4dkl = md.load("/home/enf/md_simulations/MOR/4dkl_bp_6pt6.pdb")
bp_5c1m = md.load("/home/enf/md_simulations/MOR/5c1m_bp_6pt6.pdb")
bp_4dkl_residues = set([r.resSeq for r in bp_4dkl.topology.residues if r.is_protein])
bp_5c1m_residues = set([r.resSeq for r in bp_4dkl.topology.residues if r.is_protein])
bp_residues = sorted(list(bp_4dkl_residues | bp_5c1m_residues))
bp_residue_objects = convert_list_to_resobject_list([("R", i) for i in bp_residues])

cutoff = 0.66
feature_name = "bp_residues_4dkl_5c1m_under_cutoff%dA" %(int(10*cutoff))

common_residues_pkl = get_common_residues_pkl(base)
contact_residues = find_common_residues([inactive_dir, active_dir, simulation_structure], common_residues_pkl)
contact_resSeq = [r.resSeq for r in contact_residues]


bp_residue_objects = [res for res in bp_residue_objects if bp.resSeq in contact_resSeq]

