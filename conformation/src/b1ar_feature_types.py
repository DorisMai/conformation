from residue import Residue
from b1ar_tica_config import inactive_dir, active_dir, base, simulation_structure
from get_variable_names import get_common_residues_pkl, find_common_residues

def convert_list_to_resobject_list(contact_residues):
  resobject_list = []
  for residue in contact_residues:
    new_residue = Residue(resSeq = residue[1], chain_id = residue[0])
    resobject_list.append(new_residue)
  return(resobject_list)

#CHOOSE RESIDUES:


all_residues = list(range(1,277))


npxxy_residues =  convert_list_to_resobject_list([("P", r) for r in range(255,260)])
#dry_network_residues = convert_list_to_resobject_list([("P", r) for r in [165, 279, 252]])
#triad_residues = convert_list_to_resobject_list([("P", r) for r in [289, 244, 155]])
#tm6_tm3_residues_new = convert_list_to_resobject_list([("P", 277), ("P", 339)])
tm6_packing_residues = convert_list_to_resobject_list([("P", r) for r in range(206, 216)])
tm3_packing_residues = convert_list_to_resobject_list([("P", r) for r in range(95, 106)])
tm7_packing_residues = convert_list_to_resobject_list([("P", r) for r in range(250, 260)])
tm5_packing_residues = convert_list_to_resobject_list([("P", r) for r in range(181, 193)])

cutoff = .66
feature_name = "all_residues_5f8u_hom_under_cutoff%dA" %(int(10*cutoff))
#eature_name = "protein_ligand_atom_contacts"

feature_name_residues_dict = {}
feature_name_residues_dict["rmsd_npxxy_inactive"] = npxxy_residues
feature_name_residues_dict["rmsd_npxxy_active"] = npxxy_residues
feature_name_residues_dict["tm6_tm3_packing"] = [tm3_packing_residues, tm6_packing_residues]
feature_name_residues_dict["tm6_tm7_packing"] = [tm7_packing_residues, tm6_packing_residues]
feature_name_residues_dict["tm6_tm5_packing"] = [tm5_packing_residues, tm6_packing_residues]


common_residues_pkl = get_common_residues_pkl(base)
contact_residues = find_common_residues([inactive_dir, active_dir, simulation_structure], common_residues_pkl)

#ligand_residue = Residue(resSeq = 900, chain_id = "L", res_name = "LIG")

#contact_residues += [ligand_residue]


