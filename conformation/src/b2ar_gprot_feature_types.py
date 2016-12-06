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
helix_residues = {}
helix_residues["tm1"] = list(range(29,61))
helix_residues["icl1"] = list(range(61,67))
helix_residues["tm2"] = list(range(67,97))
helix_residues["ecl1"] = list(range(97,102))
helix_residues["tm3"] = list(range(102,137))
helix_residues["icl2"] = list(range(137,147))
helix_residues["tm4"] = list(range(147,171))
helix_residues["ecl2"] = list(range(171,197))
helix_residues["tm5"] = list(range(197,230))
helix_residues["icl3"] = list(range(230,267))
helix_residues["tm6"] = list(range(267,299))
helix_residues["ecl3"] = list(range(299,305))
helix_residues["tm7"] = list(range(305,330))

cutoff = 1.0


helix_residues["tm8"] = list(range(330,340))

switch_residues = [130, 131, 208, 211, 219, 268, 272, 286, 288, 316, 322, 323, 326]
switch_npxx = [130, 131, 208, 211, 219, 268, 272, 286, 288, 316] + list(range(322,328))
switch_pp_npxx = set(switch_npxx + [51, 79, 106, 113, 118, 121, 130, 131, 132, 141, 158, 208, 211, 219, 268, 272, 282, 285, 286, 288, 316, 318, 319, 320, 323, 326, 282])
tm6_residues = list(range(270,299))
bp_residues = sorted(list(set([236, 148, 300, 297, 151, 147, 143, 326, 322, 293, 297, 318])))
dihedral_residues = list(set(switch_npxx + tm6_residues))
skip_5_residues = list(range(30,340,5))
skip_3_residues = list(range(30,340,3))
skip5_switches_pp_npxx = list(set(skip_5_residues + list(switch_pp_npxx)))
skip5_switches_pp_npxx_ser = list(set(skip_5_residues + list(switch_pp_npxx) + [207]))
skip3_switches_pp_npxx = list(set(skip_3_residues + list(switch_pp_npxx)))
#print(len(skip5_switches_pp_npxx))
all_residues = list(range(29,340))
tm_residues = helix_residues["tm1"] + helix_residues["tm2"] + helix_residues["tm3"] + helix_residues["tm4"] + helix_residues["tm5"] + helix_residues["tm6"] + helix_residues["tm7"] + helix_residues["tm8"]
sampling_method = "random"
precision = "SP"

tm6_tm3_residues = convert_list_to_resobject_list([("S",131), ("S",272)])
print("tm6_tm3_residues")
print(tm6_tm3_residues)
npxxy_residues =  convert_list_to_resobject_list([("S", r) for r in range(322,327)])
connector_residues = convert_list_to_resobject_list([("S", 121), ("S", 282)])

feature_name = "tm_residues_2rh1_3sn6_under_cutoff%dnm" %(int(cutoff))

feature_name_residues_dict = {}
feature_name_residues_dict["tm6_tm3_dist"] = tm6_tm3_residues
feature_name_residues_dict["rmsd_npxxy_active"] = npxxy_residues
feature_name_residues_dict["rmsd_npxxy_inactive"] = npxxy_residues
feature_name_residues_dict["rmsd_connector_active"] = connector_residues
feature_name_residues_dict["rmsd_connector_inactive"] = connector_residues
#feature_name_residues_dict["rmsd_binding-pocket_inactive"] = bp_residues
#feature_name_residues_dict["rmsd_binding-pocket_active"] = bp_residues
#feature_name_residues_dict["rmsd_293-289_active"] = res293_289
#feature_name_residues_dict["rmsd_293-289_inactive"] = res293_289
#feature_name_residues_dict["rmsd_npxxy-edf3_active"] = npxxy_edf3
#feature_name_residues_dict["rmsd_npxxy-edf3_inactive"] = npxxy_edf3
#feature_name_residues_dict["rmsd_triad_inactive"] = triad_residues
#feature_name_residues_dict["rmsd_triad_active"] = triad_residues

coords_bounds_dict = {'tm6_tm3_dist': [8.0, 10.0, 14.0], 'rmsd_npxxy_inactive': [0.2, 0.4, 1.0]}

contact_residues = []
#contact_residues = find_common_residues([inactive_dir, active_dir, mor_active_apo_crystalwaters_protein], get_common_residues_pkl(base))
