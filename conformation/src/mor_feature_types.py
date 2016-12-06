from residue import Residue
from mor_tica_config import inactive_dir, active_dir, base, mor_active_apo_crystalwaters_protein
from get_variable_names import get_common_residues_pkl, find_common_residues

def convert_list_to_resobject_list(contact_residues):
  resobject_list = []
  for residue in contact_residues:
    new_residue = Residue(resSeq = residue[1], chain_id = residue[0])
    resobject_list.append(new_residue)
  return(resobject_list)

#CHOOSE RESIDUES:


all_residues = list(range(29,340))

bp_residues = convert_list_to_resobject_list([("A", r) for r in bp_residues])

tm6_tm3_residues = convert_list_to_resobject_list([("A",279), ("A",165)])
print("tm6_tm3_residues")
print(tm6_tm3_residues)
npxxy_residues =  convert_list_to_resobject_list([("A", r) for r in range(332,337)])
npxxy_edf3_residues = list(range(332,337)) + [285,252,158]
npxxy_edf3 = convert_list_to_resobject_list([("A", r) for r in npxxy_edf3_residues])
connector_residues = convert_list_to_resobject_list([("A", 165), ("A", 252)])
triad_residues = convert_list_to_resobject_list([("A", 289), ("A", 244), ("A", 155)])


feature_name = "all_residues_4dkl_5c1m_under_cutoff%dnm" %(int(cutoff))

feature_name_residues_dict = {}
feature_name_residues_dict["tm6_tm3_dist"] = tm6_tm3_residues
feature_name_residues_dict["rmsd_npxxy_active"] = npxxy_residues
feature_name_residues_dict["rmsd_npxxy_inactive"] = npxxy_residues

contact_residues = find_common_residues([inactive_dir, active_dir, mor_active_apo_crystalwaters_protein], get_common_residues_pkl(base))


