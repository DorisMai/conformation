import os 
from pymol import cmd

print((os.getcwd()))
for pdb in os.listdir(os.getcwd()):
  #print(os.path.splitext(pdb))
  if os.path.splitext(pdb)[1] == ".pdb":
    name = os.path.splitext(pdb)[0]
    print(name)
    print(pdb)
    cmd.load(pdb)
    cmd.cealign("target", name)
    cmd.alter(name, "chain='R'")
    cmd.select("to_save", "gprot or %s" % name)
    save_file = "%s_aligned.pdb" % name
    cmd.set("retain_order", 1)
    cmd.save(save_file, "to_save")
    cmd.delete("to_save")
    cmd.delete(name)