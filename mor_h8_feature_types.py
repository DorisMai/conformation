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


all_residues = list(range(29,340))

tm6_tm3_residues = convert_list_to_resobject_list([("R",279), ("R",165)])
print("tm6_tm3_residues")
print(tm6_tm3_residues)
npxxy_residues =  convert_list_to_resobject_list([("R", r) for r in range(332,337)])
dry_network_residues = convert_list_to_resobject_list([("R", r) for r in [165, 279, 252]])
triad_residues = convert_list_to_resobject_list([("R", r) for r in [289, 244, 155]])

cutoff = 1.0
feature_name = "all_residues_4dkl_5c1m_under_cutoff%dA" %(int(10*cutoff))
#eature_name = "protein_ligand_atom_contacts"

feature_name_residues_dict = {}
feature_name_residues_dict["tm6_tm3_dist"] = tm6_tm3_residues
feature_name_residues_dict["rmsd_npxxy_active"] = npxxy_residues
feature_name_residues_dict["rmsd_npxxy_inactive"] = npxxy_residues
feature_name_residues_dict["rmsd_DRY_inactive"] = dry_network_residues
feature_name_residues_dict["rmsd_DRY_active"] = dry_network_residues
feature_name_residues_dict["rmsd_triad_inactive"] = triad_residues
feature_name_residues_dict["rmsd_triad_active"] = triad_residues


common_residues_pkl = get_common_residues_pkl(base)
contact_residues = find_common_residues([inactive_dir, active_dir, simulation_structure], common_residues_pkl)

#ligand_residue = Residue(resSeq = 900, chain_id = "L", res_name = "LIG")

#contact_residues += [ligand_residue]


