
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

import sys
FeatureTypeInd = int(sys.argv[1])
num_bs         = int(sys.argv[2])
runID          = int(sys.argv[3])

# In[2]:


l_ens2numreplica = [("WT", 162), ("bN82A", 162)]

#intrajdatatype   = "samplebatches"
intrajdatatype   = "convcheck"

l_FeatureType = [pysfd.features.prf.Ca2Ca_Distance(error_type=stdtype,
                                               df_rgn_seg_res_bb=mydf,
                                               label=mylbl) for stdtype in ["std_err", "std_dev"]
                                                         for mylbl, mydf in [("", None)]]

l_FeatureType += [myClass(error_type="std_err",
                 df_rgn_seg_res_bb=mydf,
                 is_with_dwell_times=False,
                 label=mylbl) for myClass in [ pysfd.features.spbsf.HBond_mdtraj ]
                           for mylbl, mydf in [("", None)]]

if FeatureTypeInd == 0:
    print(len(l_FeatureType))
    for abc in l_FeatureType:
        print(abc)

#import sys
#sys.exit()

# In[4]:

#
# Instantiate PySFD object
#

mybenchmark = {}
mymaxworkers = (1, 5)
mySFD     = pysfd.PySFD(l_ens2numreplica,
                        l_FeatureType[0],
                        intrajdatatype,
                        num_bs=num_bs,
                        intrajformat = "xtc")

#
# Compute features and significant feature differences
#

for myFeatureType in [l_FeatureType[FeatureTypeInd]]:
    print(myFeatureType)
    starttime = time.time()
    # compute features
    mySFD.feature_func = myFeatureType
    mySFD.comp_features(max_workers = mymaxworkers)
    if mySFD.feature_func_name in mySFD.df_fhists:
        mySFD.plot_feature_hists()
    mySFD.write_features(outdir="output.%d/%d/meta/%s/%s" % (mySFD.num_bs, runID,
                                                             mySFD.feature_func_name, mySFD.intrajdatatype))
    # compute feature differences
    mySFD.comp_feature_diffs(num_sigma=1, num_funit = 0.0)
    #mySFD.comp_feature_diffs_with_dwells(num_sigma=4)
    #mySFD.comp_feature_diffs_with_dwells(num_sigma=4)
    mySFD.write_feature_diffs(outdir="output.%d/%d/meta/%s/%s" % (mySFD.num_bs, runID,
                                                                   mySFD.feature_func_name, mySFD.intrajdatatype))
    endtime = time.time()
    mybenchmark[mySFD.feature_func_name] = endtime - starttime
    
print(mybenchmark)

sys.exit()
