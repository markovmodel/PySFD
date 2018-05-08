import subprocess
import shlex

cnd2mol = {}
cnd2mol["WT"]    = ["3PDO.c2", "3PDO.c3", "3PDO.c4", "DR1.c2", "DR1.c3", "DR1.c4"]
cnd2mol["bN82A"] = [ "bN82A_3PDO.c2", "bN82A_3PDO.c3", "bN82A_3PDO.c4", "bN82A_DR1.c2", "bN82A_DR1.c3", "bN82A_DR1.c4"]

numnodes = 27

cnd2l_mol_r = {}
for mycnd in cnd2mol:
    cnd2l_mol_r[mycnd] = []
    for mymol in cnd2mol[mycnd]:
        for r in range(numnodes):
            cnd2l_mol_r[mycnd].append("%s/r_%05d/%s.r_%05d.prot" % (mymol, r, mymol, r))

for mycnd in cnd2mol:
    for r in range(len(cnd2l_mol_r[mycnd])):
        subprocess.call(shlex.split("mkdir -p input_cnd/%s/r_%05d" % (mycnd, r)))
        #subprocess.call(shlex.split("ln -sf input/%s input_cnd/%s/r_%05d/%s.r_%05d.prot" % (cnd2l_mol_r[mycnd][r], mycnd, r, mymol, r)))
        subprocess.call(shlex.split("ln -sf ../../../input/%s.pdb input_cnd/%s/r_%05d/%s.r_%05d.prot.pdb" % (cnd2l_mol_r[mycnd][r], mycnd, r, mycnd, r)))
        subprocess.call(shlex.split("ln -sf ../../../input/%s.xtc input_cnd/%s/r_%05d/%s.r_%05d.prot.xtc" % (cnd2l_mol_r[mycnd][r], mycnd, r, mycnd, r)))
