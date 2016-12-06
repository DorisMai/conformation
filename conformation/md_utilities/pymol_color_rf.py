from pymol import cmd
import sys
import pandas as pd 
import pickle
import numpy as np

def change_color(object_name, residue_importances, ginis):
  cmd.alter("%s" %(object_name), "b=0.0")
  for resid in residue_importances.keys():
    print(resid)
    cmd.alter("resid %d" %resid, "b=%f" %(residue_importances[resid]))
  cmd.spectrum("b", "blue green red", object_name,min(ginis), max(ginis))


def color_protein(protein, df):
  #df["importance"] = np.abs(df["importance"].values)
  #df["importance"] = np.log(df["importance"].values)
  #df["importance"] = df["importance"].values/np.max(df["importance"].values)
  min_imp = min(df["importance"].values)
  max_imp = max(df["importance"].values)
  print(min_imp)
  print(max_imp)

  cmd.spectrum("b", "blue red", selection=protein, minimum=min_imp, maximum=max_imp)
  for index in df.index:
    print(df.loc[index])
    resid = int(df.loc[index]["resid"])
    net_importance = df.loc[index]["importance"]
    cmd.alter("resid %d & %s" % (resid, protein), "b=%f" %(net_importance))
    cmd.show("ribbon", "resi %d" % resid)
    if net_importance > np.percentile(df["importance"].values,95):
      cmd.show("sticks", "resi %d" % resid)
      #if "2" in protein:
      #  cmd.util.cbac("resi %d & sidechain & %s" % (resid, protein))
      #else:
      #  cmd.util.cbag("resi %d & sidechain & %s" % (resid, protein))
    print(resid)
    print(net_importance)
    


with open(feature_importances_file) as f:
  feature_importances_df = pickle.load(f)

proteins = cmd.get_object_list()
print(proteins)

for protein in proteins:
  cmd.alter(protein, "b=0")

for protein in proteins:
  #cmd.align(protein, "2RH1_prepped")
  color_protein(protein, feature_importances_df)

for protein in proteins:
  #cmd.align(protein, "2RH1_prepped")
  color_protein(protein, feature_importances_df)