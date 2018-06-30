
# coding: utf-8

# In[1]:


import pysfd

import collections
import time
# for further optional processing of the resulting pandas.DataFrames in this notebook
import pandas as pd
# for specific dihedral features computed with mdtraj
import mdtraj as md

import warnings
warnings.simplefilter("always")

def show_full_df(x):
    with pd.option_context('display.max_rows', None):
        display(x)


# In[2]:


l_ens2numreplica = [("WT.pcca2", 3), ("bN82A.pcca2", 3), ("aT41A.pcca1", 3)]

intrajdatatype   = "samplebatches"

#
# coarse-graining (no coarse-graining, if df_rgn_seg_res_bb=None)
#
#df_rgn_seg_res_bb = pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
#                               'seg' : ["A", "A", "B", "B", "C"],
#                               'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
#
# coarse-graining by (seg, res) ID:
df_rgn_seg_res_bb     = pd.read_csv("scripts/df_rgn_seg_res_bb.dat", sep = "\t")
df_rgn_seg_res_bb.res = df_rgn_seg_res_bb.res.apply(lambda x : list(eval(x)))
#df_rgn_seg_res_bb.to_csv("scripts/df_rgn_seg_res_bb.dat", sep = "\t", index = False)
#df_rgn_seg_res_bb = None

# coarse-graining by (seg, res, bb) ID:
df_rgn_seg_res_bb_with_bb     = pd.read_csv("scripts/df_rgn_seg_res_bb_with_bb.dat", sep = "\t")
df_rgn_seg_res_bb_with_bb.res = df_rgn_seg_res_bb_with_bb.res.apply(lambda x : list(eval(x)))

cgtype2label   = { "" : None, ".cg_nobb" : df_rgn_seg_res_bb, ".cg_withbb" : df_rgn_seg_res_bb_with_bb }
# histgramming for single features
cgtype2hSRFs   = { ""           : pd.DataFrame( { "seg" : ["A", "A"],         "res" : [5, 10]}),
                   ".cg_nobb"   : pd.DataFrame( { "rgn" : ["a1_0", "a1_4"] }),
                   ".cg_withbb" : pd.DataFrame( { "rgn" : ["MHCII"] }) }
for mykey in cgtype2hSRFs:
    cgtype2hSRFs[mykey]["dbin"] = 0.1
# histgramming for pairwise residual features
cgtype2hPRFs   = { ""           : pd.DataFrame( { "seg1" : ["A", "A"],       "res1" : [5, 10], "seg2" : ["A", "A"], "res2" : [10, 15]}),
                   ".cg_nobb"   : pd.DataFrame( { "rgn1" : ["a1_0", "a1_0"], "rgn2" : ["a1_0", "a1_7"] }),
                   ".cg_withbb" : pd.DataFrame( { "rgn1" : ["MHCII"],        "rgn2" : ["MHCII"] }) }
for mykey in cgtype2hPRFs:
    cgtype2hPRFs[mykey]["dbin"] = 0.1
# histgramming for sparse pairwise backbone/sidechain features
cgtype2hsPBSFs = { ""           : pd.DataFrame( { "seg1" : ["A", "A"], "res1" : [4, 5], "bb1" : [1, 1], "seg2" : ["A", "B"], "res2" : [4, 17], "bb2" : [1, 1]}),
                   ".cg_nobb"   : pd.DataFrame( { "rgn1" : ["a1_0", "a1_0"], "rgn2" : ["a1_0", "a1_7"] }),
                   ".cg_withbb" : pd.DataFrame( { "rgn1" : ["MHCII"],        "rgn2" : ["MHCII"] }) }
for mykey in cgtype2hsPBSFs:
    cgtype2hsPBSFs[mykey]["dbin"] = 0.1


# In[3]:


#
# Instantiation of Feature Classes - just delete/comment out what feature type you won't need
#

l_FeatureType = []

#
# Single Residual Features (SRF)
#

#l_FeatureType += [pysfd.features.srf.ChemicalShift(error_type="std_err",
#                                               df_rgn_seg_res_bb=df_rgn_seg_res_bb,
#                                              label=mylbl)]
l_FeatureType += [pysfd.features.srf.CA_RMSF(error_type="std_err",
                                                 df_rgn_seg_res_bb=mydf,
                                                 label=mylbl) for mylbl, mydf in cgtype2label.items()]
l_FeatureType += [pysfd.features.srf.SASA_sr(error_type=stdtype,
                                            df_rgn_seg_res_bb=mydf,
                                            label=mylbl) for stdtype in ["std_err", "std_dev"]
                                                      for mylbl, mydf in cgtype2label.items()]
l_FeatureType += [pysfd.features.srf.RSASA_sr(error_type=stdtype,
                                              df_rgn_seg_res_bb=mydf,
                                              label=mylbl) for stdtype in ["std_err", "std_dev"]
                                                      for mylbl, mydf in cgtype2label.items()]

l_FeatureType += [pysfd.features.srf.Dihedral(circular_stats=None, error_type=stdtype,
                                                  df_rgn_seg_res_bb=mydf,
                                                  feat_subfunc=feat_subfunc,
                                                  label=mylbl) for mylbl, mydf in cgtype2label.items()
                                                            for stdtype in ["std_err", "std_dev"]
                                                            for feat_subfunc in [md.compute_phi,  md.compute_psi,  md.compute_omega, md.compute_chi1, md.compute_chi2, md.compute_chi3,  md.compute_chi4]]

l_FeatureType += [pysfd.features.srf.Scalar_Coupling(error_type=stdtype,
                                                  df_rgn_seg_res_bb=mydf,
                                                  feat_subfunc=feat_subfunc,
                                                  label=mylbl) for mylbl, mydf in cgtype2label.items()
                                                            for stdtype in ["std_err", "std_dev"]
                                                            for feat_subfunc in [md.compute_J3_HN_C, md.compute_J3_HN_CB, md.compute_J3_HN_HA]]

#
# Pairwise Residueal Features (PRF)
#

l_FeatureType += [pysfd.features.prf.Ca2Ca_Distance(error_type=stdtype,
                                               df_rgn_seg_res_bb=mydf,
                                               label=mylbl) for stdtype in ["std_err", "std_dev"]
                                                         for mylbl, mydf in cgtype2label.items()]

l_FeatureType += [pysfd.features.prf.CaPos_Correlation(partial_corr=is_partial_corr,
                                                       error_type="std_err",
                                                       df_rgn_seg_res_bb=mydf,
                                                       label=mylbl)
                                              for mylbl, mydf in cgtype2label.items()
                                              for is_partial_corr in [False, True]]

l_FeatureType += [pysfd.features.prf.Dihedral_Correlation(partial_corr=is_partial_corr,
                                                          error_type="std_err",
                                                          df_rgn_seg_res_bb=mydf,
                                                          feat_subfunc=md.compute_phi,
                                                          label=mylbl)
                                               for mylbl, mydf in cgtype2label.items()
                                               for is_partial_corr in [False, True]]

l_FeatureType += [pysfd.features.prf.Scalar_Coupling_Correlation(partial_corr=is_partial_corr,
                                                          error_type="std_err",
                                                          df_rgn_seg_res_bb=mydf,
                                                          feat_subfunc=md.compute_J3_HN_C,
                                                          label=mylbl)
                                               for mylbl, mydf in cgtype2label.items()
                                               for is_partial_corr in [False, True]]



#
# sparse Pairwise Backbone/Sidechain Feature (sPBSF)
#

l_FeatureType += [myClass(error_type="std_err",
                 df_rgn_seg_res_bb=mydf,
                 is_with_dwell_times=False,
                 label=mylbl) for myClass in [
                                           pysfd.features.spbsf.HBond_VMD,
                                           pysfd.features.spbsf.HBond_mdtraj,
#                                           pysfd.features.spbsf.HBond_HBPLUS,
#                                           pysfd.features.spbsf.HvvdwHB,
                                           pysfd.features.spbsf.Hvvdwdist_VMD]
                           for mylbl, mydf in cgtype2label.items()]

#
# Pairwise Pairwise Residual Features (PPRF)
#

l_FeatureType += [pysfd.features.pprf.Ca2Ca_Distance_Correlation(partial_corr=is_partial_corr,
                                                  error_type="std_err",
                                                  df_rgn_seg_res_bb=mydf,
                                                  label=mylbl,
                                                  atmselstr="name CA and index 0 to 50")
                                               for mylbl, mydf in cgtype2label.items()
                                               for is_partial_corr in [False, True]]

#
# Pairwise sparse Pairwise Backbone/Sidechain Features (PsPBSF)
#

mysPBSF_class = pysfd.features.spbsf.HBond_mdtraj(error_type="std_err",
                                                  is_correlation=True,
                                                  label="")
l_FeatureType += [pysfd.features.pspbsf.sPBSF_Correlation(partial_corr=is_partial_corr,
                                                          error_type="std_err",
                                                          df_rgn_seg_res_bb=mydf,
                                                          label=mylbl,
                                                          sPBSF_class=mysPBSF_class)
                                               for mylbl, mydf in cgtype2label.items()
                                               for is_partial_corr in [False, True]]

l_FeatureType += [pysfd.features.srf.IsDSSP_mdtraj(error_type=stdtype,
                                                   df_rgn_seg_res_bb=mydf,
                                                   DSSPpars=("H", True),
                                                   label=mylbl) for mylbl, mydf in cgtype2label.items()
                                                                for stdtype in ["std_err"]]

#
# with significant difference tests of higher moments and histogramming of specific features
#

l_FeatureType += [pysfd.features.srf.Scalar_Coupling(error_type=stdtype,
                                                     max_mom_ord=2,
                                                     df_rgn_seg_res_bb=mydf,
                                                     df_hist_feats=cgtype2hSRFs[mylbl],
                                                     feat_subfunc=feat_subfunc,
                                                     label=mylbl+"_2moms_fhists") for mylbl, mydf in cgtype2label.items()
                                                                                  for stdtype in ["std_err"]
                                                                                  for feat_subfunc in [md.compute_J3_HN_C]]

l_FeatureType += [pysfd.features.prf.Ca2Ca_Distance(error_type=stdtype,
                                                    df_rgn_seg_res_bb=mydf,
                                                    df_hist_feats=cgtype2hPRFs[mylbl],
                                                    label=mylbl+"_fhists") for stdtype in ["std_err", "std_dev"]
                                                                           for mylbl, mydf in list(cgtype2label.items())]


l_FeatureType += [myClass(error_type="std_err",
                  df_rgn_seg_res_bb=mydf,
                  max_mom_ord=2,
                  df_hist_feats=cgtype2hsPBSFs[mylbl],
                  is_with_dwell_times=False,
                  label=mylbl+"_fhists") for myClass in [pysfd.features.spbsf.HBond_mdtraj]
                                         for mylbl, mydf in list(cgtype2label.items())]


# In[4]:

#
# Instantiate PySFD object
#

mybenchmark = {}
mymaxworkers = (2, 3)
mySFD     = pysfd.PySFD(l_ens2numreplica,
                        l_FeatureType[0],
                        intrajdatatype,
                        intrajformat = "xtc")

#
# Compute features and significant feature differences
#

for myFeatureType in l_FeatureType:
    print(myFeatureType)
    starttime = time.time()
    # compute features
    mySFD.feature_func = myFeatureType
    mySFD.comp_features(max_workers = mymaxworkers)
    if mySFD.feature_func_name in mySFD.df_fhists:
        mySFD.plot_feature_hists()
    mySFD.write_features()
    # compute feature differences
    mySFD.comp_feature_diffs(num_sigma=2, num_funit = 0.0)
    #mySFD.comp_feature_diffs_with_dwells(num_sigma=4)
    #mySFD.comp_feature_diffs_with_dwells(num_sigma=4)
    mySFD.write_feature_diffs()
    endtime = time.time()
    mybenchmark[mySFD.feature_func_name] = endtime - starttime
    
print(mybenchmark)


# In[5]:

#
# Display some results from above 
#

featuretype_labels = list(mySFD.df_features.keys())
print(featuretype_labels)
print(mySFD.feature_func_name)
print(mySFD.df_features[featuretype_labels[0]].head())
ens_comps = list(mySFD.df_feature_diffs[featuretype_labels[0]].keys())
print(ens_comps)
print(mySFD.df_feature_diffs[featuretype_labels[0]][ens_comps[0]].head())


# In[6]:
#
# compute common signficant feature differences
#

df_common = mySFD.comp_and_write_common_feature_diffs(feature_func_name = "spbsf.HBond_VMD.std_err",
                                                      l_sda_pair = [('bN82A.pcca2', 'WT.pcca2'),
                                                       ('aT41A.pcca1', 'WT.pcca2')],
                                                      l_sda_not_pair = [('aT41A.pcca1', 'bN82A.pcca2')])
df_common.head()


# In[7]:

#
# reload features and compute common signficant feature differences
#

#mySFD.reload_features(feature_func_name="pbsi.Hvvdwdist_VMD.std_dev",
mySFD.reload_features(feature_func_name="spbsf.HBond_VMD.std_err",
                      intrajdatatype="samplebatches",
                      l_ens=None)
mySFD.comp_feature_diffs(num_sigma=2, num_funit = 0.0)
print(list(mySFD.df_features.keys()))
print(mySFD.feature_func_name)

df_common = mySFD.comp_and_write_common_feature_diffs(feature_func_name = "spbsf.HBond_VMD.std_err",
                                                      l_sda_pair = [('bN82A.pcca2', 'WT.pcca2'),
                                                       ('aT41A.pcca1', 'WT.pcca2')],
                                                      l_sda_not_pair = [('aT41A.pcca1', 'bN82A.pcca2')])
print(df_common.head())

# In[8]:

#
# Compute Feature Redundancies via Correlations
#

mybenchmark = {}
mymaxworkers = (2, 3)
mySFD     = pysfd.PySFD(l_ens2numreplica,
                        l_FeatureType[0],
                        intrajdatatype,
                        intrajformat = "xtc")



l_featuretype = ['srf.RSASA_sr.std_err', 'srf.SASA_sr.std_err', 'spbsf.Hvvdwdist_VMD.std_err', 'srf.CA_RMSF.std_err', 'srf.phi.std_err', 'srf.psi.std_err', 'srf.omega.std_err', 'srf.J3_HN_C.std_err', 'srf.J3_HN_CB.std_err', 'srf.J3_HN_HA.std_err', 'srf.chi1.std_err', 'srf.chi2.std_err', 'srf.chi3.std_err', 'srf.chi4.std_err']

#l_featuretype = ['spbsf.HBond_mdtraj.std_err']
l_featuretype
for myfeaturetype in l_featuretype:
    mySFD.reload_features(feature_func_name = myfeaturetype,
                          intrajdatatype = "samplebatches",
                          l_ens=None)

columns_bak = mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].columns
mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].columns = mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].columns.droplevel(level = 1)
mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"]         = mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].query("rnm2 == 'WAT'")
#mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].set_index(mySFD.l_lbl["spbsf.Hvvdwdist_VMD.std_err"], inplace = True)
#mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"] = mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].mask(mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"] < 0.1)
#mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].reset_index(inplace = True)
mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].columns = columns_bak
mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].rename(columns = { "seg1" : "seg", "res1" : "res", "rnm1" : "rnm", "bb1" : "bb" }, level = 0, inplace = True)
mySFD.df_features["spbsf.Hvvdwdist_VMD.std_err"].drop(columns = ["seg2", "res2", "rnm2", "bb2"], level = 0, inplace = True)
mySFD.df_features["srf.Hvvdwdist_H2O.std_err"]           = mySFD.df_features.pop("spbsf.Hvvdwdist_VMD.std_err")
mySFD.l_lbl["srf.Hvvdwdist_H2O.std_err"]                 = mySFD.l_lbl.pop("spbsf.Hvvdwdist_VMD.std_err")
mySFD.l_lbl["srf.Hvvdwdist_H2O.std_err"]                 = mySFD.l_lbl[l_featuretype[0]]
#display(mySFD.df_features["srf.Hvvdwdist_H2O.std_err"])
   
#corrmethod = "kendall"
corrmethod = "spearman"
#corrmethod = "pearson"
#corrmethod = "circcorr"
print(corrmethod)


l_featuretype = [ 'srf.CA_RMSF.std_err', 'srf.RSASA_sr.std_err', 'srf.SASA_sr.std_err', 'srf.Hvvdwdist_H2O.std_err', 'srf.phi.std_err', 'srf.psi.std_err', 'srf.omega.std_err', 'srf.chi1.std_err', 'srf.chi2.std_err', 'srf.chi3.std_err', 'srf.chi4.std_err', 'srf.J3_HN_C.std_err', 'srf.J3_HN_CB.std_err', 'srf.J3_HN_HA.std_err' ]
l_radcol = ['phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4']
rnm2df_feature_corr, df_mean_features = mySFD.featuretype_redundancies(l_featuretype = l_featuretype,
                                                                       corrmethod = corrmethod,
                                                                       l_radcol = l_radcol,
                                                                       withrnmlevel = False,
                                                                       withplots = True)

# In[9]:

#
# Plot two feature types against each other
#

import numpy as np
import matplotlib
# do not interact with matplotlib plots:
matplotlib.use('agg')
import matplotlib.pyplot as plt

def J3_HN_CB(phi):
    return md.nmr.scalar_couplings._J3_function(phi, **md.nmr.scalar_couplings.J3_HN_CB_coefficients["Bax2007"])

l_philin   = np.linspace(-np.pi, np.pi, 100)
l_phi      = df_mean_features["phi"]
l_J3_HN_CB = df_mean_features["J3_HN_CB"]

mydf = pd.DataFrame({ "phi" : l_phi, "J3_HN_CB" : l_J3_HN_CB } )
mydf.dropna(inplace = True)

for mycmap in [ "Blues" ]:
    plt.figure()
    plt.plot(l_philin / np.pi * 180, J3_HN_CB(l_philin), color = "k", linewidth = 0.5)
    mymap = plt.hist2d(mydf.phi / np.pi * 180, mydf.J3_HN_CB, bins = 25, normed = False,
           cmin = 1, cmap = plt.get_cmap(mycmap), norm = matplotlib.colors.LogNorm())
    #plt.scatter(mydf.phi, mydf.J3_HN_CB)
    plt.colorbar(ticks=[9.375, 18.75, 37.5, 75, 150, 300], format = '%d')

    #plt.xlim((-np.pi, np.pi))
    plt.xlim((-180, 180))
    plt.xticks(np.linspace(-180, 180, 7))
    plt.ylim((0, 4.5))
    plt.xlabel(r"$\phi$ [°]")
    plt.ylabel("J3_HN_CB")
    plt.savefig("output/figures/feature_type_correlations/J3_HN_CB_vs_phi.pdf")

print(pd.Series(l_phi).corr(pd.Series(J3_HN_CB(l_phi)), method = "spearman"))
print(pd.Series(l_phi).corr(pd.Series(l_J3_HN_CB),      method = "spearman"))



l_RSASA_sr       = df_mean_features["RSASA_sr"]
l_Hvvdwdist_H2O = df_mean_features["Hvvdwdist_H2O"]

mydf = pd.DataFrame({ "RSASA_sr" : l_RSASA_sr, "Hvvdwdist_H2O" : l_Hvvdwdist_H2O } )
mydf.dropna(inplace = True)

for mycmap in [ "Blues" ]:
    plt.figure()
    mymap = plt.hist2d(mydf.RSASA_sr, mydf.Hvvdwdist_H2O, bins = 25, normed = False,
           cmin = 1, cmap = plt.get_cmap(mycmap), norm = matplotlib.colors.LogNorm())
    #plt.scatter(mydf.phi, mydf.J3_HN_CB)
    #plt.colorbar(ticks=[9.375, 18.75, 37.5, 75, 150, 300], format = '%d')
    plt.colorbar()

    #plt.xlim((-np.pi, np.pi))
    #plt.xlim((-180, 180))
    #plt.xticks(np.linspace(-180, 180, 7))
    #plt.ylim((0, 4.5))
    plt.xlabel(r"RSASA_sr [$\AA$]")
    plt.ylabel("Hvvdwdist_H2O")
    #plt.savefig("output/figures/feature_type_correlations/J3_HN_CB_vs_phi.pdf")

print(pd.Series(l_RSASA_sr).corr(pd.Series(l_Hvvdwdist_H2O),      method = "spearman"))

# # in the following are some useful snippets for post-processing

# In[10]:

mySFD.reload_features(feature_func_name="spbsf.HBond_VMD.std_err",
                      intrajdatatype="samplebatches",
                      l_ens=None)

abc = mySFD.df_features['spbsf.HBond_VMD.std_err']
abc.loc[(abc[("seg1", "")] == 'B') & (abc[("res1", "")] == 15) & (abc[("res2", "")] == 79), :]
abc.head()


# In[11]:


abc = mySFD.df_features['spbsf.HBond_VMD.std_err']
#abc.loc[(abc[("seg1", "")] == 'B') & (abc[("res1", "")] == 15) & (abc[("res2", "")] == 79), :]
#abc.loc[(abc[("WT.pcca2", "mf")] == 1) , :]
abc.head()

