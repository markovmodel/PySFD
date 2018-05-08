import numpy as np
import pandas as pd
import biopandas.pdb as bp

# install via
# "conda install biopython"
import Bio

bb2lbl = { 0 : "s", 1 : "b" }
idlevel = "residue"
inpdb = "input/WT.pcca2/r_00000/WT.pcca2.r_00000.noh.pdb"
df_pdb = bp.PandasPdb().read_pdb(inpdb).df["ATOM"][['chain_id', 'residue_number', 'residue_name', 'atom_name']]
df_pdb.columns = ["seg", "res", "rnm", "anm"]
if idlevel == "residue":
    df_pdb = df_pdb[["seg", "res", "rnm"]].drop_duplicates().query("seg == 'C'")
df_pdb = df_pdb.query("rnm not in ['ACE', 'NHE']")
df_pdb["bb"] = 0
df_tmp = df_pdb.copy()
df_tmp["bb"] = 1
df_pdb = pd.concat([df_pdb, df_tmp])
df_pdb = df_pdb.sort_index().reset_index(drop = True)
df_pdb["rnm"] = np.vectorize(Bio.SeqUtils.IUPACData.protein_letters_3to1.get)(df_pdb["rnm"].apply(lambda x : x.title()))
df_pdb["rgn"] = df_pdb["rnm"].astype(str) + df_pdb["res"].astype(str) + np.vectorize(bb2lbl.get)(df_pdb["bb"])
df_pdb.drop("rnm" ,axis = 1, inplace = True)
df_pdb["res"] = df_pdb["res"].apply(lambda x : [x])
df_pdb = pd.concat(
[pd.DataFrame({ "rgn" : 2 * [ "MHCII", "MHCII"], "seg" : 2 * ["A", "B"], "res" : 2 * [range(4, 185), range(4, 191)], "bb" : [0,0,1,1]}),
 df_pdb])
df_pdb = df_pdb[["rgn", "seg", "res", "bb"]]
df_pdb.to_csv("df_rgn_seg_res_bb.dat", sep = "\t", index = False)
