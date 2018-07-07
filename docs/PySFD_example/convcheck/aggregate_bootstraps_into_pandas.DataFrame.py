import pandas as pd
import subprocess
import numpy as np 

l_num_bs = list(range(25,151,25))+[162]
l_runID  = range(100)

#feature_func_name = "prf.distance.Ca2Ca.std_err"
#feature_func_name = "prf.distance.Ca2Ca.std_dev"
feature_func_name = "spbsf.distance.Ca2Ca.std_dev"
intrajtype        = "samplebatches"

l_lbl = ['seg1', 'res1', 'rnm1', 'seg2', 'res2', 'rnm2']

infilename = "output.bak/meta/162/%s/samplebatches/%s.samplebatches.bN82A_vs_WT.nsigma_1.000000.nfunit_0.000000.dat" % (feature_func_name, feature_func_name)
df_ref = pd.read_csv(infilename, sep = "\t", header=[0,1])
df_ref["score"] = ((np.sign(df_ref["score"])+1)/2).astype("int")
df_ref["lblens"] = df_ref.loc[:, l_lbl + ["score"]].astype(str).sum(axis=1)
df_ref["lblens"]

a_num_bs    = []
a_runID     = []
a_numsdf    = []
a_bsandref  = []
a_bsniref   = []
a_refnibs   = []

for num_bs in l_num_bs:
	for runID in l_runID:
		infilename = "output.%d/%d/meta/%s/%s/%s.%s.bN82A_vs_WT.nsigma_1.000000.nfunit_0.000000.dat" % (num_bs, runID, feature_func_name, intrajtype, feature_func_name, intrajtype)
		df_tmp = pd.read_csv(infilename, sep = "\t", header=[0,1])
		df_tmp["score"] = ((np.sign(df_tmp["score"])+1)/2).astype("int")
		df_tmp["lblens"] = df_tmp.loc[:, l_lbl + ["score"]].astype(str).sum(axis=1)
		a_numsdf.append(len(df_tmp))
		a_runID.append(runID)
		a_num_bs.append(num_bs)
		a_bsandref.append(len(np.intersect1d(df_tmp["lblens"],df_ref["lblens"])))
		a_bsniref.append(len(np.setdiff1d(df_tmp["lblens"],df_ref["lblens"])))
		a_refnibs.append(len(np.setdiff1d(df_ref["lblens"],df_tmp["lblens"])))

df_sdfcomps = pd.DataFrame({ "num_bs" : a_num_bs, "r" : a_runID, "numsdf" : a_numsdf, "bsandref" : a_bsandref, "bsniref" : a_bsniref, "refnibs" : a_refnibs})
df_sdfcomps = df_sdfcomps.groupby(["num_bs"])["numsdf", "bsandref", "bsniref", "refnibs" ].agg(["mean", "std"])
df_sdfcomps

df_sdfcomps.to_csv("df_sdfcomps.%s.dat" % (feature_func_name), sep = "\t")


