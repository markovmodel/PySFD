# This file is part of PySFD.
#
# Copyright (c) 2018 Sebastian Stolzenberg,
# Computational Molecular Biology Group,
# Freie Universitaet Berlin (GER)
#
# for any feedback or questions, please contact the author:
# Sebastian Stolzenberg <ss629@cornell.edu>
#
# PySFD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

r"""
=======================================
PySFD - Significant Feature Differences Analyzer for Python
        Sparse Pairwise Backbone Sidechain Features (contact frequencies and dwell times)
=======================================
"""

# only necessary for Python 2
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import warnings as _warnings
import numpy as _np
import mdtraj as _md
import pandas as _pd
import subprocess as _subprocess
import shlex as _shlex
import glob as _glob
import itertools as _itertools
#import pickle as _pickle
#import os as _os

from pysfd.features import _feature_agent


class _sPBSF(_feature_agent.FeatureAgent):
    """
    ######################################
    Parent Class _sPBSF
    ######################################
    
    Intermediary class between a particular _sPBSF-derived feature class and
    _feature_agent.FeatureAgent
    in order to bundle common tasks

    If coarse-graining (via df_rgn_seg_res_bb, see below) into regions,
    by default aggregate via rgn_agg_func = "sum"

    Parameters:
    -----------
    * error_type   : str, default="std_err"
        compute feature errors as ...
        | "std_err" : ... standard errors
        | "std_dev" : ... mean standard deviations

    * max_mom_ord   : int, default: 1
                      maximum ordinal of moment to compute
                      if max_mom_ord > 1, this will add additional entries
                      "mf.2", "sf.2", ..., "mf.%d" % max_mom_ord, "sf.%d" % max_mom_ord
                      to the feature tables

    * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                          regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
      df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                      'seg' : ["A", "A", "B", "B", "C"],
                                      'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                          if None, no coarse-graining is performed

    * rgn_agg_func  : function or str for coarse-graining, default = "sum"
                      function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                      this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                      - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                      - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                        such as "mean", "std"

    * df_hist_feats : pandas.DataFrame, default=None
                      data frame of features, for which to compute histograms.
                      .columns are self.l_lbl[self.feature_func_name] + ["dbin"], e.g.:
                      df_hist_feats = pd.DataFrame( { "seg1" : ["A", "A"],
                                                      "res1" : [5, 10],
                                                      "seg2" : ["A", "A"],
                                                      "res2" : [10, 15],
                                                      "dbin" : [0.1, 0.1] })
                      dbin is the histogram binning resolution in units of the feature type.
                      Only dbin values are allowed, which
                      sum exactly to the next significant digit's unit, e.g.:
                      for dbin = 0.02 = 2*10^-2 exists an n = 10, so that
                      n * dbin = 0.1  = 1*10^-1
                      Currently - for simplicity - dbin values have to be
                      the same for each feature.
                      If df_hist_feats == dbin (i.e. an int or float), 
                      compute histograms for all features with
                      uniform histogram binning resolution dbin.

    * is_with_dwell_times : bool, default=False
        compute binary pairwise interactions with mean dwell times (t_on, t_off)?

    * label        : string, user-specific label for feature_name 
    """

    def __init__(self, feature_name, error_type, max_mom_ord, df_rgn_seg_res_bb, rgn_agg_func, is_with_dwell_times,
                 label, df_hist_feats = None, **params):
        if rgn_agg_func is None:
            rgn_agg_func = "sum"
        params["is_with_dwell_times"] = is_with_dwell_times
        params["_finish_traj_df"]     = self._finish_traj_df
        s_coarse = ""
        if df_rgn_seg_res_bb is not None:
            s_coarse = "coarse."
        super(_sPBSF, self).__init__(feature_name      = feature_name + s_coarse + error_type + label,
                                     error_type        = error_type,
                                     max_mom_ord       = max_mom_ord,
                                     df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                     rgn_agg_func      = rgn_agg_func,
                                     df_hist_feats     = df_hist_feats,
                                     **params)

    @staticmethod
    def _finish_traj_df(fself, l_lbl, traj_df, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats, is_with_dwell_times, is_correlation, r):
        """
        helper function of _feature_func_engine:
        finishes processing of traj_df in each of
        the _feature_func_engine() in the spbsf module

        Parameters
        ----------
        * fself          : self pointer to foreign master PySFD object

        * l_lbl          : list of str, feature label types, see below

        * traj_df        : pandas.DataFrame containing feature labels 

        * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                              regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
          df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                          'seg' : ["A", "A", "B", "B", "C"],
                                          'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                              if None, no coarse-graining is performed

        * rgn_agg_func  : function or str for coarse-graining
                          function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                          this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                          - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                          - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                            such as "mean", "std"

        * df_hist_feats : pandas.DataFrame, default=None
                          data frame of features, for which to compute histograms.
                          .columns are self.l_lbl[self.feature_func_name] + ["dbin"], e.g.:
                          df_hist_feats = pd.DataFrame( { "seg1" : ["A", "A"],
                                                          "res1" : [5, 10],
                                                          "seg2" : ["A", "A"],
                                                          "res2" : [10, 15],
                                                          "dbin" : [0.1, 0.1] })
                          dbin is the histogram binning resolution in units of the feature type.
                          Only dbin values are allowed, which
                          sum exactly to the next significant digit's unit, e.g.:
                          for dbin = 0.02 = 2*10^-2 exists an n = 10, so that
                          n * dbin = 0.1  = 1*10^-1
                          Currently - for simplicity - dbin values have to be
                          the same for each feature.
                          If df_hist_feats == dbin (i.e. an int or float), 
                          compute histograms for all features with
                          uniform histogram binning resolution dbin.

        * is_with_dwell_times : bool
            compute binary pairwise interactions with mean dwell times (t_on, t_off)?

        * is_correlation : bool, optional, whether or not to output feature values
                           for a subsequent correlation analysis (e.g. pff.Feature_Correlation())

        * r              : int, replica index

        Returns
        -------
        * traj_df        : pandas.DataFrame, contains all the feature values accumulated for this replica

        * dataflags      : dict, contains flags with more information about the data in traj_df
        """

        dataflags = { "error_type" : fself.error_type[fself._feature_func_name],
                      "is_with_dwell_times" : is_with_dwell_times }
        if fself._feature_func_name in fself.max_mom_ord:
            dataflags["max_mom_ord"] = fself.max_mom_ord[fself._feature_func_name]

        if is_with_dwell_times and is_correlation:
            raise ValueError("both is_with_dwell_times and is_correlation are True!")

        def _comp_mean_dwell_times(a, is_std, mylen):
            """
            Computes mean dwell times in units of input frames
            (equivalent to mean first passage times between the "on" and "off"
            states of an interaction
            
            Parameters:
            ----------
            a      : sparse input array, including 0-indexed frames
                    in which the pairwise interaction exists
            is_std : using standard deviations (True) or standard errors (False)
            mylen  : trajectory length

            Returns:
            ----------
            avg_t_on, avg_t_off
            """
        
            if len(a) in [0, mylen]:
                return None, None
            x = _np.in1d(_np.arange(mylen), a).astype(int)
            dx = x[1:] != x[:-1]
            l = _np.append(_np.where(dx), len(x) - 1)
            m = _np.diff(_np.append(-1, l))[:-1]
            m1 = _np.mean(m[1::2]) if len(m[1::2]) > 0 else None
            m2 = _np.mean(m[::2])  if len(m[::2])  > 0 else None
            if is_std:
                ms1 = _np.std(m[1::2]) if len(m[1::2]) > 0 else None
                ms2 = _np.std(m[::2])  if len(m[::2])  > 0 else None
                if x[0] == 0:
                    return m1, ms1, m2, ms2
                elif x[0] == 1:
                    return m2, ms2, m1, ms1
            else:
                if x[0] == 0:
                    return m1, m2
                elif x[0] == 1:
                    return m2, m1

        numframes = traj_df['frame'].max() + 1
        if is_correlation:
            traj_df["feature"] = traj_df["seg1"]             + "_" + \
                                 traj_df["res1"].astype(str) + "_" + \
                                 traj_df["bb1"].astype(str)  + "_" + \
                                 traj_df["seg2"]             + "_" + \
                                 traj_df["res2"].astype(str) + "_" + \
                                 traj_df["bb2"].astype(str)
            for mycol in ['seg1', 'rnm1', 'res1', 'bb1', 'seg2', 'rnm2', 'res2', 'bb2']:
                del traj_df[mycol]

            def sparse2full01(x, numframes):
                b = _np.zeros(numframes)
                b[x.values] = 1
                # mark contacts with all "1" for deletion
                if _np.all(b == 1):
                    return float('NaN')
                else:
                    return list(b)
            traj_df = traj_df.groupby("feature").agg({"frame" : lambda x : sparse2full01(x, numframes)}).dropna()
            traj_df = _pd.DataFrame(_np.array(list(traj_df["frame"].values)), index = traj_df.index)
            ##traj_df_old = traj_df.copy()
            #traj_df.set_index(["bspair", "frame"], inplace = True)
            #traj_df = traj_df.unstack(fill_value=0)
            #traj_df.columns = traj_df.columns.get_level_values(1)
            #del traj_df.columns.name
            ## delete all rows that have only "1" entries
            #traj_df = traj_df.loc[traj_df.sum(axis=1) != traj_df.shape[1]]
            ##print("len(full_traj):", len(full_traj), "len(traj_df_old):", len(traj_df_old), "len(traj_df):", len(traj_df))
            ##import pickle
            ##with open("test.pickle.%d.dat" % (r), "bw") as fout:
            ##    pickle.dump([traj_df_old, traj_df, full_traj], fout)
            dataflags["df_rgn_seg_res_bb"] = df_rgn_seg_res_bb
            dataflags["l_lbl"]             = ['seg1', 'res1', 'bb1', 'seg2', 'res2', 'bb2']
            return traj_df, dataflags

        if is_with_dwell_times:
            if fself.error_type[fself._feature_func_name] == "std_dev":
                traj_df = traj_df.groupby(l_lbl).agg({
                    'f' : 'sum',
                    'frame' : lambda x: _comp_mean_dwell_times(x, True, traj_df['frame'].max())})
                traj_df['f'] /= 1. * numframes
                traj_df[['ton', 'ston', 'tof', 'stof']] = traj_df['frame'].apply(_pd.Series)
                traj_df['sf'] = _np.sqrt(traj_df['f'] * (1 - traj_df['f']) * numframes / (numframes - 1))
            elif fself.error_type[fself._feature_func_name] == "std_err":
                traj_df = traj_df.groupby(l_lbl).agg({
                    'f' : 'sum',
                    'frame' : lambda x: _comp_mean_dwell_times(x, False, traj_df['frame'].max())})
                traj_df['f'] /= 1. * numframes
                traj_df[['ton', 'tof']] = traj_df['frame'].apply(_pd.Series)
            del traj_df['frame']
        else:
            def myhist(a_data, dbin):
                if _np.any(a_data.isnull()):
                    return _np.float("NaN")
                else:
                    prec = len(str(dbin).partition(".")[2])+1
                    a_data = _np.concatenate((a_data, (numframes-len(a_data)) * [0]))
                    a_bins =_np.arange(_np.floor(min(0,a_data.min()) / dbin),
                                       _np.ceil(a_data.max() / dbin) + 1, 1) * dbin
                    a_hist = _np.histogram(a_data, bins = a_bins, density = True)
                    return tuple(list(a_hist))

            if df_rgn_seg_res_bb is None:
                # if include ALL feature entries for histogramming:
                if isinstance(df_hist_feats, (int, float)):
                    dbin = df_hist_feats
                    traj_df_hist = traj_df.reset_index()
                # elif include NO feature entries for histogramming:
                elif df_hist_feats is None:
                    dbin = None
                    traj_df_hist = None
                # else (if include SOME feature entries for histogramming):
                elif isinstance(df_hist_feats, _pd.DataFrame):
                    dbin = df_hist_feats["dbin"][0]
                    traj_df_hist = traj_df.merge(df_hist_feats.drop(columns = "dbin"), how = "right")
                if traj_df_hist is not None:
                    traj_df_hist = traj_df_hist.groupby(l_lbl).agg( { "f" : lambda x: myhist(x, dbin ) } ).reset_index()
                    traj_df_hist.rename(columns = { "f" : "fhist" }, inplace = True)

                traj_df.drop_duplicates(inplace = True)
                traj_df = traj_df.groupby(l_lbl).agg({ 'f' : 'sum' })
                traj_df.rename(columns = { 'f' : 'N' }, inplace = True)
                traj_df['f'] = traj_df['N'] / numframes
 
                l_flbl = ['f']
                #import scipy.stats as _scipy_stats
                #import numpy as np
                #numframes = 4
                #print(numframes)
                #a_data = np.random.randint(0,2,numframes)
                #print(a_data)
                #N = a_data.sum()
                #print(N)
                #f = a_data.mean()
                #print(f)
                #mymom = 4
                #mymoment = (              N * (1 - f)**mymom + \
                #             (numframes - N) *      (-f)**mymom) / \
                #                          (numframes)
                #print(mymoment)
                #print(_scipy_stats.moment(a_data, moment = mymom))
                for mymom in range(2, fself.max_mom_ord[fself._feature_func_name]+1):
                    l_flbl += ['f.%d' % mymom]
                    #traj_df['f.%d' % mymom] = _scipy_stats.moment(a_f, axis=0, moment = mymom)
                    traj_df['f.%d' % mymom] = (traj_df['N'] * (1 - traj_df['f'])**mymom + \
                                              (numframes - traj_df['N']) * (-traj_df['f'])**mymom) / \
                                              (numframes)
                traj_df = traj_df[l_flbl].copy()
                traj_df.reset_index(inplace = True)
                if fself.error_type[fself._feature_func_name] == "std_dev":
                    traj_df['sf'] = _np.sqrt(traj_df['f'] * (1 - traj_df['f']) * numframes / (numframes - 1))
                if traj_df_hist is not None:
                    traj_df = traj_df.merge(traj_df_hist, how = "outer")
                traj_df.reset_index(drop = True, inplace = True)

            elif df_rgn_seg_res_bb is not None:
                if "bb" in df_rgn_seg_res_bb.columns:
                    traj_df_seg1_res1 = traj_df[["seg1", "res1", "bb1"]].drop_duplicates()
                    traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res", "bb1" : "bb"}, inplace = True)
                    traj_df_seg2_res2 = traj_df[["seg2", "res2", "bb2"]].drop_duplicates()
                    traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res", "bb2" : "bb"}, inplace = True)
                    df_rgn1_seg1_res1 = df_rgn_seg_res_bb.copy()
                    df_rgn1_seg1_res1.columns = ["rgn1", "seg1", "res1", "bb1"]
                    df_rgn2_seg2_res2 = df_rgn_seg_res_bb.copy()
                    df_rgn2_seg2_res2.columns = ["rgn2", "seg2", "res2", "bb2"]
                else:
                    traj_df_seg1_res1 = traj_df[["seg1", "res1"]].drop_duplicates()
                    traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res"}, inplace = True)
                    traj_df_seg2_res2 = traj_df[["seg2", "res2"]].drop_duplicates()
                    traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res"}, inplace = True)
                    df_rgn1_seg1_res1 = df_rgn_seg_res_bb.copy()
                    df_rgn1_seg1_res1.columns = ["rgn1", "seg1", "res1"]
                    df_rgn2_seg2_res2 = df_rgn_seg_res_bb.copy()
                    df_rgn2_seg2_res2.columns = ["rgn2", "seg2", "res2"]
                traj_df_seg_res = _pd.concat([traj_df_seg1_res1, traj_df_seg2_res2]).drop_duplicates()
                df_merge = traj_df_seg_res.merge(df_rgn_seg_res_bb, how = "outer", copy = False, indicator = True)
                df_merge = df_merge.query("_merge == 'right_only'")
                if len(df_merge) > 0:
                    warnstr = "df_rgn_seg_res_bb, your coarse-graining definition, has resID entries that are not in your feature list:\n%s" % df_merge
                    _warnings.warn(warnstr)
                traj_df = traj_df.merge(df_rgn1_seg1_res1, copy = False)
                traj_df = traj_df.merge(df_rgn2_seg2_res2, copy = False)

                #rgn1="a1H2"
                #rgn2="a1H2"
                #print(traj_df.query("rgn1 == '%s' and rgn2 == '%s'" % (rgn1, rgn2)))
                traj_df = traj_df.groupby(["rgn1", "rgn2", "frame"]).agg( { "f" : rgn_agg_func } )

                # if include ALL feature entries for histogramming:
                if isinstance(df_hist_feats, (int, float)):
                    dbin = df_hist_feats
                    traj_df_hist = traj_df.reset_index()
                # elif include NO feature entries for histogramming:
                elif df_hist_feats is None:
                    dbin = None
                    traj_df_hist = None
                # else (if include SOME feature entries for histogramming):
                elif isinstance(df_hist_feats, _pd.DataFrame):
                    dbin = df_hist_feats["dbin"][0]
                    traj_df_hist = traj_df.reset_index().merge(df_hist_feats.drop(columns = "dbin"), how = "right")

                if traj_df_hist is not None:
                    traj_df_hist = traj_df_hist.groupby(["rgn1", "rgn2"]).agg( { "f" : lambda x: myhist(x, dbin ) } )
                    traj_df_hist.rename(columns = { "f" : "fhist" }, inplace = True)

                if fself.error_type[fself._feature_func_name] == "std_err":
                    traj_df = traj_df.groupby(['rgn1', 'rgn2']).agg( { 'f' : 'sum' } )
                    #print(traj_df.query("rgn1 == '%s' and rgn2 == '%s'" % (rgn1, rgn2)))
                    traj_df.rename(columns = { 'f' : 'N' }, inplace = True)
                    traj_df['f'] = traj_df['N'] / numframes
                    l_flbl = ['f']
                    for mymom in range(2, fself.max_mom_ord[fself._feature_func_name]+1):
                        l_flbl += ['f.%d' % mymom]
                        #traj_df['f.%d' % mymom] = _scipy_stats.moment(a_f, axis=0, moment = mymom)
                        traj_df['f.%d' % mymom] = (traj_df['N'] * (1 - traj_df['f'])**mymom + \
                                                  (numframes - traj_df['N']) * (-traj_df['f'])**mymom) / \
                                                  (numframes)
                    traj_df = traj_df[l_flbl].copy()

                elif fself.error_type[fself._feature_func_name] == "std_dev":
                    #traj_df.set_index(['rgn1', 'rgn2'] + l_lbl + ['frame'], inplace = True)
                    #rgn1="a1H2"
                    #rgn2="a1H2"
                    #print(traj_df.query("rgn1 == '%s' and rgn2 == '%s'" % (rgn1, rgn2)))
                    # to sum up all the contacts between two regions:
                    #traj_df = traj_df.groupby(['rgn1', 'rgn2', 'frame']).agg( { 'f' : rgn_agg_func } )
                    #print(traj_df.query("rgn1 == '%s' and rgn2 == '%s'" % (rgn1, rgn2)))
                    # the following unusal way to compute mean/std frequencies
                    # for each pairwise interaction
                    # accounts for missing contact entries of
                    # zero-contact frames
                    mygroup = traj_df.groupby(['rgn1', 'rgn2'])
                    traj_df['mf'] = 1. * mygroup.transform('sum') / numframes
                    traj_df['sf'] = (traj_df['f'] - traj_df['mf']) ** 2
                    traj_df['sfcount'] = 1
                    traj_df = traj_df.groupby(['rgn1', 'rgn2']).agg({'mf'      : lambda g: g.iloc[0],
                                                                     'sf'      : _np.sum,
                                                                     'sfcount' : _np.sum})
                    # add contributions from missing contact entries,
                    # i.e. frames with zero contacts
                    traj_df['sf'] += (numframes - traj_df['sfcount']) * traj_df['mf'] ** 2
                    traj_df['sf'] = _np.sqrt(1. * traj_df['sf'] / (numframes - 1))
                    del traj_df['sfcount']
                    traj_df.rename(columns={'mf': 'f'}, inplace = True)
                    #print(traj_df.query("rgn1 == '%s' and rgn2 == '%s'" % (rgn1, rgn2)))
                if traj_df_hist is not None:
                    traj_df = traj_df.merge(traj_df_hist, left_index = True, right_index = True, how = "outer")
                traj_df.reset_index(inplace = True)
        traj_df['r'] = r
        #traj_df.reset_index(inplace = True)
        return traj_df, dataflags


class HBond_mdtraj(_sPBSF):
    """
    Computes hydrogen bonds via mdtraj.baker_hubbard()
    (which is slow compared to hbond_vmd)
    for a particular simulation with replica index r

    Parameters
    ----------
    (see in _sPBSF parent class docstring below)

    """
    __doc__ = __doc__ + _sPBSF.__doc__

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = None, df_hist_feats = None, is_with_dwell_times = False, label = ""):
        super(HBond_mdtraj, self).__init__(feature_name        = "spbsf.HBond_mdtraj.",
                                           error_type          = error_type,
                                           max_mom_ord         = max_mom_ord,
                                           df_rgn_seg_res_bb   = df_rgn_seg_res_bb,
                                           rgn_agg_func        = rgn_agg_func,
                                           df_hist_feats       = df_hist_feats,
                                           is_with_dwell_times = is_with_dwell_times,
                                           label               = label)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes hydrogen bonds via mdtraj.baker_hubbard() (slow)
        for a particular simulation with replica index r
    
        Parameters
        ----------
        * args   : tuple (fself, myens, r):
            * fself        : self pointer to foreign master PySFD object
            * myens        : string
                             Name of simulated ensemble
            * r            : int, replica index

        * params : dict, extra parameters as keyword arguments
            * error_type     : str
                compute feature errors as ...
                | "std_err"  : ... standard errors
                | "std_dev"  : ... mean standard deviations

            * max_mom_ord   : int, default: 1
                              maximum ordinal of moment to compute
                              if max_mom_ord > 1, this will add additional entries
                              "mf.2", "sf.2", ..., "mf.%d" % max_mom_ord, "sf.%d" % max_mom_ord
                              to the feature tables
    
            * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                                  regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
              df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                              'seg' : ["A", "A", "B", "B", "C"],
                                              'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                                  if None, no coarse-graining is performed

            * rgn_agg_func  : function or str for coarse-graining
                              function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                              this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                              - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                              - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                                such as "mean", "std"

            * df_hist_feats : pandas.DataFrame, default=None
                              data frame of features, for which to compute histograms.
                              .columns are self.l_lbl[self.feature_func_name] + ["dbin"], e.g.:
                              df_hist_feats = pd.DataFrame( { "seg1" : ["A", "A"],
                                                              "res1" : [5, 10],
                                                              "seg2" : ["A", "A"],
                                                              "res2" : [10, 15],
                                                              "dbin" : [0.1, 0.1] })
                              dbin is the histogram binning resolution in units of the feature type.
                              Only dbin values are allowed, which
                              sum exactly to the next significant digit's unit, e.g.:
                              for dbin = 0.02 = 2*10^-2 exists an n = 10, so that
                              n * dbin = 0.1  = 1*10^-1
                              Currently - for simplicity - dbin values have to be
                              the same for each feature.
                              If df_hist_feats == dbin (i.e. an int or float), 
                              compute histograms for all features with
                              uniform histogram binning resolution dbin.
       """
        l_lbl1 = ["seg1", "res1", "rnm1", "bb1", "anm1"]
        l_lbl2 = ["seg2", "res2", "rnm2", "bb2", "anm2"]
        l_lbl = l_lbl1[:-1] + l_lbl2[:-1]
   
        fself, myens, r = args
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        is_with_dwell_times                         = params["is_with_dwell_times"]
        is_correlation                              = params.get("is_correlation", False)
        _finish_traj_df                             = params["_finish_traj_df"]

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        mytraj = _md.load('%s.%s' % (instem, fself.intrajformat), top='%s.pdb' % instem)
        a_rnm  = fself._get_raw_topology_ids('%s.pdb' % instem, "atom").rnm.values
        a_atom = list(mytraj.topology.atoms)
        a_seg = [a.segment_id for a in a_atom]
        a_res = [a.residue.resSeq for a in a_atom]
        #a_rnm = [a.residue.name for a in a_atom]
        a_anm = [a.name for a in a_atom]
        a_bb = [fself.is_bb(a.name) for a in a_atom]
        df_lbl1 = _pd.DataFrame(data={'seg1': a_seg, 'res1': a_res, 'rnm1': a_rnm, 'bb1': a_bb, 'anm1': a_anm},
                                columns=l_lbl1)
        df_lbl2 = _pd.DataFrame(data={'seg2': a_seg, 'res2': a_res, 'rnm2': a_rnm, 'bb2': a_bb, 'anm2': a_anm},
                                columns=l_lbl2)
        traj_df = []
        #picklefname = "output/tmp/%s.%s/%s/r_%05d/%s.%s.%s.r_%05d.pickle.dat" % (fself.feature_func_name, fself.intrajdatatype, myens, r, fself.feature_func_name, fself.intrajdatatype, myens, r)
        #if not _os.path.isfile(picklefname):
        if True:
            for i in range(len(mytraj)):
                pai_inds = _md.baker_hubbard(mytraj[i], exclude_water=False, periodic=False)
                a = df_lbl1.iloc[pai_inds[:, 0]].reset_index(drop=True)
                b = df_lbl2.iloc[pai_inds[:, 2]].reset_index(drop=True)
                traj_df.append(_pd.concat([a, b], axis=1))
                traj_df[-1]['frame'] = i
                # if (i % 10) == 0:
                #    print(i)
            traj_df = _pd.concat(traj_df, copy=False)
            #_subprocess.Popen(_shlex.split("mkdir -p output/tmp/%s.%s/%s/r_%05d" % (fself.feature_func_name, fself.intrajdatatype, myens, r))).wait()
            #with open(picklefname, "wb") as f:
            #    _pickle.dump(traj_df, f)
        #else:
        #    with open(picklefname, "rb") as f:
        #        print("reloading %s" % picklefname)
        #        traj_df = _pickle.load(f)
        if fself.maxnumframes > 0:
            traj_df = traj_df.query("frame < %d" % fself.maxnumframes).copy()
  
        for mycol in ['anm1', 'anm2']:
            del traj_df[mycol]
        traj_df.drop_duplicates(inplace = True)
        traj_df = traj_df[l_lbl + ['frame']]
        #traj_df['f'] = 1. / len(mytraj)
        # now, number of contacts, later mean of number of contacts = frequency
        traj_df['f'] = 1
        return _finish_traj_df(fself, l_lbl, traj_df, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats, is_with_dwell_times, is_correlation, r)


class Hvvdwdist_VMD(_sPBSF):
    """
    For a particular simulation with replica index r, compute non-covalent,
    heavy atom van der Waals radius contacts with VMD
    (see features/scripts/b1.1.a.compute_sPBSFs.hvvdwdist.tcl and
    Venkatakrishnan, A., Deupi, X., Lebon, G., Tate, C. G., Schertler, G. F., and Babu, M. M. (2013)
    Molecular signatures of g-protein-coupled receptors. Nature, 494(7436), 185–194.
    for details)

    Parameters
    ----------
    * l_solv_rnm : optional list of additional solvent residue names (str,   in VMD: "resname")

    * l_anm      : optional list of additional solvent atom names    (str,   in VMD: "name")

    * l_rad      : optional list of additional solvent vdW radii     (float, in VMD: "name"), corresponds to l_anm

    * solv_seg   : str, segID name that will represent all solvent molecules

    (see more in _sPBSF parent class docstring below)

    """
    __doc__ = __doc__ + _sPBSF.__doc__

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = None, df_hist_feats = None, is_with_dwell_times = False, l_solv_rnm = None,
                 l_anm = None, l_rad = None, solv_seg = "X", label = ""):
        if l_solv_rnm is None:
            l_solv_rnm = ["\\\"Cl-\\\"", "\\\"Na\\\\+\\\"", "WAT"]
        if l_anm is None:
            l_anm = ["\\\"Cl-\\\"", "\\\"Na\\\\+\\\""]
        if l_rad is None:
            l_rad = [1.9, 1.5]

        super(Hvvdwdist_VMD, self).__init__(feature_name        = "spbsf.Hvvdwdist_VMD.",
                                            error_type          = error_type,
                                            max_mom_ord         = max_mom_ord,
                                            df_rgn_seg_res_bb   = df_rgn_seg_res_bb,
                                            rgn_agg_func        = rgn_agg_func,
                                            df_hist_feats       = df_hist_feats,
                                            is_with_dwell_times = is_with_dwell_times,
                                            label               = label,
                                            l_solv_rnm          = l_solv_rnm,
                                            l_anm               = l_anm,
                                            l_rad               = l_rad,
                                            solv_seg            = solv_seg)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        For a particular simulation with replica index r, compute non-covalent,
        heavy atom van der Waals radius contacts with VMD
        (see features/scripts/b1.1.a.compute_sPBSFs.hvvdwdist.tcl and
        Venkatakrishnan, A., Deupi, X., Lebon, G., Tate, C. G., Schertler, G. F., and Babu, M. M. (2013)
        Molecular signatures of g-protein-coupled receptors. Nature, 494(7436), 185–194.
        for details)

        Parameters
        ----------
        * args   : tuple (fself, myens, r)
            * fself      : self pointer to foreign master PySFD object

            * myens      : string, Name of simulated ensemble

            * r          : int, replica index

        * params : dict, extra parameters as keyword arguments
            * l_solv_rnm : optional list of additional solvent residue names (str,   in VMD: "resname")

            * l_anm      : optional list of additional solvent atom names    (str,   in VMD: "name")

            * l_rad      : optional list of additional solvent vdW radii     (float, in VMD: "name"),
                           corresponds to l_anm

            * solv_seg   : str, segID name that will represent all solvent molecules

            * error_type   : str
                compute feature errors as ...
                | "std_err" : ... standard errors
                | "std_dev" : ... mean standard deviations

            * max_mom_ord   : int, default: 1
                              maximum ordinal of moment to compute
                              if max_mom_ord > 1, this will add additional entries
                              "mf.2", "sf.2", ..., "mf.%d" % max_mom_ord, "sf.%d" % max_mom_ord
                              to the feature tables

            * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                                  regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
              df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                              'seg' : ["A", "A", "B", "B", "C"],
                                              'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                                  if None, no coarse-graining is performed

            * rgn_agg_func  : function or str for coarse-graining
                              function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                              this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                              - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                              - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                                such as "mean", "std"

            * df_hist_feats : pandas.DataFrame, default=None
                              data frame of features, for which to compute histograms.
                              .columns are self.l_lbl[self.feature_func_name] + ["dbin"], e.g.:
                              df_hist_feats = pd.DataFrame( { "seg1" : ["A", "A"],
                                                              "res1" : [5, 10],
                                                              "seg2" : ["A", "A"],
                                                              "res2" : [10, 15],
                                                              "dbin" : [0.1, 0.1] })
                              dbin is the histogram binning resolution in units of the feature type.
                              Only dbin values are allowed, which
                              sum exactly to the next significant digit's unit, e.g.:
                              for dbin = 0.02 = 2*10^-2 exists an n = 10, so that
                              n * dbin = 0.1  = 1*10^-1
                              Currently - for simplicity - dbin values have to be
                              the same for each feature.
                              If df_hist_feats == dbin (i.e. an int or float), 
                              compute histograms for all features with
                              uniform histogram binning resolution dbin.
       """
        l_lbl1 = ["seg1", "res1", "rnm1", "bb1", "anm1"]
        l_lbl2 = ["seg2", "res2", "rnm2", "bb2", "anm2"]
        l_lbl = l_lbl1[:-1] + l_lbl2[:-1]
    
        fself, myens, r                             = args
        l_solv_rnm                                  = params["l_solv_rnm"]
        l_anm                                       = params["l_anm"]
        l_rad                                       = params["l_rad"]
        solv_seg                                    = params["solv_seg"]
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        is_with_dwell_times                         = params["is_with_dwell_times"]
        is_correlation                              = params.get("is_correlation", False)
        _finish_traj_df                             = params["_finish_traj_df"]

        s_solv_seg   = "_".join(solv_seg)
        s_solv_rnm    = "_".join(l_solv_rnm)
        s_anm         = "_".join(l_anm)
        s_rad        = "_".join([str(f) for f in l_rad])
    
        indir = "input/%s/r_%05d" % (myens, r)
        instem = "%s.r_%05d.noh" % (myens, r)
        outdir = "output/%s/r_%05d/%s/%s" % (myens, r, fself.feature_func_name, fself.intrajdatatype)
        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()
        mycmd = "vmd -dispdev text -e %s/features/scripts/compute_sPBSF.hvvdwdist.tcl -args %s %s %s %s %s %s %s %s" \
                % (fself.pkg_dir, indir, instem, fself.intrajformat, outdir, s_solv_seg, s_solv_rnm, s_anm, s_rad)
        outfile = open("%s/log.compute_sPBSFs.hvvdwdist.tcl.log" % outdir, "w")
        myproc = _subprocess.Popen(_shlex.split(mycmd), stdout=outfile, stderr=outfile)
        myproc.wait()
        outfile.close()
        traj_df = _pd.read_csv("%s/%s.sPBSF.hvvdwdist.dat" % (outdir, instem), sep=' ',
                               names=['frame'] + l_lbl)
        #traj_df = traj_df.query("not ((seg1 == seg2) and (abs(res2 - res1) <= 4))").copy()

        def order(df, blockpair):
            # order contact pair IDs blockwise: (seg1,res1,rnm1,bb1)<->(seg2,res2,rnm2,bb2)
            # by successively ordering as seg1<seg2, if "seg1==seg2": res1<res2,
            #                                        if "seg1==seg2" and "res1==res2": rnm1<rnm2,...
            blockpair_01 = [ x for y in blockpair       for x in y ]
            blockpair_10 = [ x for y in blockpair[::-1] for x in y ]
            pairs = _np.transpose(blockpair)
            cumboolmask = _pd.Series(_np.ones(len(df), dtype="bool"))
            for mypair in pairs:
                boolmask    = _np.logical_and((df[mypair[0]]   >= df[mypair[1]]), cumboolmask)
                if not boolmask.any():
                    break
                #boolmask    = _np.logical_and((df[mypair[0]]   >  df[mypair[1]]), cumboolmask)
                df.loc[boolmask, blockpair_01] = df.loc[boolmask, blockpair_10].values
                cumboolmask = _np.logical_and((df[mypair[0]]  == df[mypair[1]]), cumboolmask)
        blockpair    = [ l_lbl1[:-1], l_lbl2[:-1] ]
        order(traj_df, blockpair)

        #traj_df['f'] = 1. / (traj_df['frame'].max() + 1)
        # now, number of contacts, later mean of number of contacts = frequency
        traj_df['f'] = 1.
        return _finish_traj_df(fself, l_lbl, traj_df, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats, is_with_dwell_times, is_correlation, r)


class HBond_VMD(_sPBSF):
    """
    Computes hydrogen bonds via VMD using standard parameters
    (see features/scripts/b1.1.a.compute_sPBSFs.hbonds.tcl for details)

    Parameters
    ----------
    * cutoff_dist  : optional, float, VMD hbond distance cutoff (in Angstrom)

    * cutoff_angle : optional, float, VMD hbond distance cutoff (in Angstrom)

    (see more in _sPBSF parent class docstring below)

    """
    __doc__ = __doc__ + _sPBSF.__doc__

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = None, df_hist_feats = None, is_with_dwell_times = False, cutoff_dist = 3.0, cutoff_angle = 20, label = ""):
        super(HBond_VMD, self).__init__(feature_name        = "spbsf.HBond_VMD.",
                                        error_type          = error_type,
                                        max_mom_ord         = max_mom_ord,
                                        df_rgn_seg_res_bb   = df_rgn_seg_res_bb,
                                        rgn_agg_func        = rgn_agg_func,
                                        df_hist_feats       = df_hist_feats,
                                        is_with_dwell_times = is_with_dwell_times,
                                        label               = label,
                                        cutoff_dist         = cutoff_dist,
                                        cutoff_angle        = cutoff_angle)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes hydrogen bonds via VMD using standard parameters
        (see features/scripts/b1.1.a.compute_sPBSFs.hbonds.tcl for details)
    
        Parameters
        ----------
        * args   : tuple (fself, myens, r):
            * fself        : self pointer to foreign master PySFD object

            * myens        : string
                             Name of simulated ensemble

            * r            : int
                             replica index

        * params : dict, extra parameters as keyword arguments
            * cutoff_dist  : optional, float, VMD hbond distance cutoff (in Angstrom)

            * cutoff_angle : optional, float, VMD hbond distance cutoff (in angles)

            * error_type   : str
                compute feature errors as ...
                | "std_err" : ... standard errors
                | "std_dev" : ... mean standard deviations

            * max_mom_ord   : int, default: 1
                              maximum ordinal of moment to compute
                              if max_mom_ord > 1, this will add additional entries
                              "mf.2", "sf.2", ..., "mf.%d" % max_mom_ord, "sf.%d" % max_mom_ord
                              to the feature tables

            * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                                  regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
              df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                              'seg' : ["A", "A", "B", "B", "C"],
                                              'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                                  if None, no coarse-graining is performed

            * rgn_agg_func  : function or str for coarse-graining
                              function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                              this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                              - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                              - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                                such as "mean", "std"
 
            * df_hist_feats : pandas.DataFrame, default=None
                              data frame of features, for which to compute histograms.
                              .columns are self.l_lbl[self.feature_func_name] + ["dbin"], e.g.:
                              df_hist_feats = pd.DataFrame( { "seg1" : ["A", "A"],
                                                              "res1" : [5, 10],
                                                              "seg2" : ["A", "A"],
                                                              "res2" : [10, 15],
                                                              "dbin" : [0.1, 0.1] })
                              dbin is the histogram binning resolution in units of the feature type.
                              Only dbin values are allowed, which
                              sum exactly to the next significant digit's unit, e.g.:
                              for dbin = 0.02 = 2*10^-2 exists an n = 10, so that
                              n * dbin = 0.1  = 1*10^-1
                              Currently - for simplicity - dbin values have to be
                              the same for each feature.
                              If df_hist_feats == dbin (i.e. an int or float), 
                              compute histograms for all features with
                              uniform histogram binning resolution dbin.
      """
        l_lbl1 = ["seg1", "res1", "rnm1", "bb1", "anm1"]
        l_lbl2 = ["seg2", "res2", "rnm2", "bb2", "anm2"]
        l_lbl = l_lbl1[:-1] + l_lbl2[:-1]
    
        fself, myens, r = args
        cutoff_dist    = params["cutoff_dist"]
        cutoff_angle   = params["cutoff_angle"]
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        is_with_dwell_times                         = params["is_with_dwell_times"]
        is_correlation                              = params.get("is_correlation", False)
        _finish_traj_df                             = params["_finish_traj_df"]
        indir = "input/%s/r_%05d" % (myens, r)
        instem = "%s.r_%05d.prot" % (myens, r)
        outdir = "output/%s/r_%05d/%s/%s" % (myens, r, fself.feature_func_name, fself.intrajdatatype)
        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()
        _subprocess.Popen(_shlex.split("rm -rf %s/hbplus" % outdir)).wait()
        mycmd = "vmd -dispdev text -e %s/features/scripts/compute_sPBSF.hbond.tcl -args %s %s %s %s %f %f" \
                % (fself.pkg_dir, indir, instem, fself.intrajformat, outdir, cutoff_dist, cutoff_angle)
        outfile = open("%s/log.compute_sPBSFs.hbond.tcl.log" % outdir, "w")
        myproc = _subprocess.Popen(_shlex.split(mycmd), stdout=outfile, stderr=outfile)
        myproc.wait()
        outfile.close()
        traj_df = _pd.read_csv("%s/%s.sPBSF.hbond.vmd.dat" % (outdir, instem), sep=' ',
                               names=['frame'] + l_lbl)
    
        #traj_df['f'] = 1. / (traj_df['frame'].max() + 1)
        # now, number of contacts, later mean of number of contacts = frequency
        traj_df['f'] = 1.
        return _finish_traj_df(fself, l_lbl, traj_df, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats, is_with_dwell_times, is_correlation, r)


class HBond_HBPLUS(_sPBSF):
    """
    Computes hydrogren bonds via the HBPLUS program
    (more refined than VMD or mdtraj, i.e. more hard-wired parameters
    used to infer hydrogen bonds):
    I.K. McDonald and J.M. Thornton (1994), "Satisfying Hydrogen Bonding Potential in Proteins", JMB 238:777-793.
    how to obtain HBPLUS:
    http://www.ebi.ac.uk/thornton-srv/software/HBPLUS

    Before you instantiate this feature class, please double-check
    the path in the "hbdir" environment variable in
    scripts/compute_PI.hbplus.sh
    , e.g.,
    export hbdir=/home/sstolzen/mypackages/hbplus
    , and make this file executable:
    chmod +x scripts/compute_PI.hbplus.sh

    Parameters
    ----------
    (see in _sPBSF parent class docstring below)

    """
    __doc__ = __doc__ + _sPBSF.__doc__

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = None, df_hist_feats = None, is_with_dwell_times = False, label = ""):
        super(HBond_HBPLUS, self).__init__(feature_name        = "spbsf.HBond_HBPLUS.",
                                           error_type          = error_type,
                                           max_mom_ord         = max_mom_ord,
                                           df_rgn_seg_res_bb   = df_rgn_seg_res_bb,
                                           rgn_agg_func        = rgn_agg_func,
                                           df_hist_feats       = df_hist_feats,
                                           is_with_dwell_times = is_with_dwell_times,
                                           label               = label)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes hydrogren bonds via the HBPLUS program (more refined than VMD or mdtraj:
        I.K. McDonald and J.M. Thornton (1994), "Satisfying Hydrogen Bonding Potential in Proteins", JMB 238:777-793.
        how to obtain HBPLUS:
        http://www.ebi.ac.uk/thornton-srv/software/HBPLUS
 
        Parameters
        ----------
        * args   : tuple (fself, myens, r)
            * fself      : self pointer to foreign master PySFD object

            * myens      : string, Name of simulated ensemble

            * r          : int, replica index
        * params : dict, extra parameters as keyword arguments
            * error_type    : str
                compute feature errors as ...
                | "std_err" : ... standard errors
                | "std_dev" : ... mean standard deviations

            * max_mom_ord   : int, default: 1
                              maximum ordinal of moment to compute
                              if max_mom_ord > 1, this will add additional entries
                              "mf.2", "sf.2", ..., "mf.%d" % max_mom_ord, "sf.%d" % max_mom_ord
                              to the feature tables

            * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                                  regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
              df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                              'seg' : ["A", "A", "B", "B", "C"],
                                              'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                                  if None, no coarse-graining is performed
    
            * rgn_agg_func  : function or str for coarse-graining
                              function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                              this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                              - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                              - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                                such as "mean", "std"

            * df_hist_feats : pandas.DataFrame, default=None
                              data frame of features, for which to compute histograms.
                              .columns are self.l_lbl[self.feature_func_name] + ["dbin"], e.g.:
                              df_hist_feats = pd.DataFrame( { "seg1" : ["A", "A"],
                                                              "res1" : [5, 10],
                                                              "seg2" : ["A", "A"],
                                                              "res2" : [10, 15],
                                                              "dbin" : [0.1, 0.1] })
                              dbin is the histogram binning resolution in units of the feature type.
                              Only dbin values are allowed, which
                              sum exactly to the next significant digit's unit, e.g.:
                              for dbin = 0.02 = 2*10^-2 exists an n = 10, so that
                              n * dbin = 0.1  = 1*10^-1
                              Currently - for simplicity - dbin values have to be
                              the same for each feature.
                              If df_hist_feats == dbin (i.e. an int or float), 
                              compute histograms for all features with
                              uniform histogram binning resolution dbin.
       """
        l_lbl1 = ["seg1", "res1", "rnm1", "bb1", "anm1"]
        l_lbl2 = ["seg2", "res2", "rnm2", "bb2", "anm2"]
        l_lbl = l_lbl1[:-1] + l_lbl2[:-1]
    
        fself, myens, r = args
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        is_with_dwell_times                         = params["is_with_dwell_times"]
        is_correlation                              = params.get("is_correlation", False)
        _finish_traj_df                             = params["_finish_traj_df"]

        indir = "input/%s/r_%05d" % (myens, r)
        instem = "%s.r_%05d.prot" % (myens, r)
        outdir = "output/%s/r_%05d/%s/%s" % (myens, r, fself.feature_func_name, fself.intrajdatatype)
        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()
        _subprocess.Popen(_shlex.split("rm -rf %s/hbplus" % outdir)).wait()
        mycmd = "%s/features/scripts/compute_PI.hbplus.sh %s %s %s %s %s" % (
            fself.pkg_dir, fself.pkg_dir, indir, instem, fself.intrajformat, outdir)
        myproc = _subprocess.Popen(_shlex.split(mycmd))
        myproc.wait()
   
        df_pdb = fself._get_raw_topology_ids('%s/%s.pdb' % (indir, instem), "residue")
        df_pdb1 = df_pdb.copy()
        df_pdb1.columns = ["seg1", "res1", "rnm12"]
        df_pdb2 = df_pdb.copy()
        df_pdb2.columns = ["seg2", "res2", "rnm22"]
        numframes = len(_glob.glob("%s/hbplus/hbplus/*.hb2" % outdir))
        traj_df = []
        for f in range(numframes):
            inhb = "%s/hbplus/hbplus/%s_tmp.%05d.hb2" % (outdir, instem, f)
            mycmd = "awk '/^[A-Z][0-9]/ {print substr($0,1,1)\" \"substr($0,2,4)\" \"substr($0,7,3)\" \"$2\" \
\"substr($0,15,1)\" \"substr($0,16,4)\" \"substr($0,21,3)\" \"$4}' %s" % (
                inhb)
            myproc = _subprocess.Popen(_shlex.split(mycmd), stdout=_subprocess.PIPE)
            myproc.wait()
            traj_df.append(
                _pd.read_table(myproc.stdout, sep=" ", header=None, names=l_lbl))
            myproc.stdout.close()
            traj_df[-1][['bb1', 'bb2']] = _np.vectorize(fself.is_bb)(traj_df[-1][['bb1', 'bb2']])
            #traj_df[-1][['rnm1', 'rnm2']] = _np.vectorize(fself.rnm2pdbrnm.__getitem__)(traj_df[-1][['rnm1', 'rnm2']])
            traj_df[-1]['frame'] = f
            traj_df[-1].drop_duplicates(inplace = True)
            #if "bb" in df_rgn_seg_res_bb.columns:
            #    traj_df_seg1_res1 = traj_df[["seg1", "res1", "bb1"]].drop_duplicates()
            #    traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res", "bb1" : "bb"}, inplace = True)
            #    traj_df_seg2_res2 = traj_df[["seg2", "res2", "bb2"]].drop_duplicates()
            #    traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res", "bb2" : "bb"}, inplace = True)
            #else:
            #    traj_df_seg1_res1 = traj_df[["seg1", "res1"]].drop_duplicates()
            #    traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res"}, inplace = True)
            #    traj_df_seg2_res2 = traj_df[["seg2", "res2"]].drop_duplicates()
            #    traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res"}, inplace = True)
            #traj_df_seg_res = _pd.concat([traj_df_seg1_res1, traj_df_seg2_res2]).drop_duplicates()
            #df_merge = traj_df_seg_res.merge(df_pdb, how = "outer", copy = False) 
            #df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
            #if len(df_merge) > 0:
            #    warnstr = "residue name mismatch!\n%s" % df_merge
            #    _warnings.warn(warnstr) 
            traj_df[-1] = traj_df[-1].merge(df_pdb1, copy = False)
            traj_df[-1] = traj_df[-1].merge(df_pdb2, copy = False)
            traj_df[-1].drop(["rnm1", "rnm2"], axis=1, inplace = True)
            traj_df[-1].rename(columns={'rnm12': 'rnm1', 'rnm22': 'rnm2'}, inplace = True)
    
        traj_df = _pd.concat(traj_df, copy=False)
        traj_df = traj_df[l_lbl + ['frame']]
        #traj_df['f'] = 1. / numframes
        # now, number of contacts, later mean of number of contacts = frequency
        traj_df['f'] = 1.
        return _finish_traj_df(fself, l_lbl, traj_df, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats, is_with_dwell_times, is_correlation, r)


class HvvdwHB(_sPBSF):
    """
    For a particular simulation with replica index r, computes
    HvvdwHB contacts, which are defined to exist (versus not exist)
    in a simulation frame, if this contact is
    a hydrogen bond     computed with HBond_HBPLUS() or
    a Hvvdwdist contact computed with Hvvdwdist_VMD()
    (see above feature classes for documentation)

    Hvvdwdist contacts whose contact partners are
    within four residue positions of the same chain ID
    are ignored here, i.e. filtered out using the
    "traj_df = traj_df.query("not ((seg1 == seg2) and (abs(res2 - res1) <= 4))")"
    command (see below), because such contacts are not as sensitive as
    corresponding hydrogen bonds

    Parameters
    ----------
    * l_solv_rnm : optional list of additional solvent residue names (str,   in VMD: "resname")

    * l_anm      : optional list of additional solvent atom names    (str,   in VMD: "name")

    * l_rad      : optional list of additional solvent vdW radii     (float, in VMD: "name"), corresponds to l_anm

    * solv_seg   : str, segID name that will represent all solvent molecules

    * label      : string, user-specific label for feature_name

    (see more in _sPBSF parent class docstring below)

    """
    __doc__ = __doc__ + _sPBSF.__doc__

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = None, df_hist_feats = None, is_with_dwell_times = False,
                 l_solv_rnm = None, l_anm = None, l_rad = None, solv_seg = "X", label = ""):
        if l_solv_rnm is None:
            l_solv_rnm = ["\\\"Cl-\\\"", "\\\"Na\\\\+\\\"", "WAT"]
        if l_anm is None:
            l_anm = ["\\\"Cl-\\\"", "\\\"Na\\\\+\\\""]
        if l_rad is None:
            l_rad = [1.9, 1.5]

        super(HvvdwHB, self).__init__(feature_name        = "spbsf.HvvdwHB.",
                                      error_type          = error_type,
                                      max_mom_ord         = max_mom_ord,
                                      df_rgn_seg_res_bb   = df_rgn_seg_res_bb,
                                      rgn_agg_func        = rgn_agg_func,
                                      df_hist_feats       = df_hist_feats,
                                      is_with_dwell_times = is_with_dwell_times,
                                      label               = label,
                                      l_solv_rnm          = l_solv_rnm,
                                      l_anm               = l_anm,
                                      l_rad               = l_rad,
                                      solv_seg            = solv_seg)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        For a particular simulation with replica index r, compute
        HvvdwHB contacts, which are defined to exist (versus not exist)
        in a simulation frame, if this contact is
        a hydrogen bond     computed with HBond_HBPLUS() or
        a Hvvdwdist contact computed with Hvvdwdist_VMD()
        (see above feature classes for documentation)
    
        Hvvdwdist contacts whose contact partners are
        within four residue positions of the same chain ID
        are ignored here, i.e. filtered out using the
        "traj_df = traj_df.query("not ((seg1 == seg2) and (abs(res2 - res1) <= 4))")"
        command (see below), because such contacts are not as sensitive as
        corresponding hydrogen bonds

        Parameters
        ----------
        * args   : tuple (fself, myens, r)
            * fself      : self pointer to foreign master PySFD object

            * myens      : string, Name of simulated ensemble

            * r          : int, replica index

        * params : dict, extra parameters as keyword arguments
            * l_solv_rnm : optional list of additional solvent residue names (str,   in VMD: "resname")

            * l_anm      : optional list of additional solvent atom names    (str,   in VMD: "name")

            * l_rad      : optional list of additional solvent vdW radii     (float, in VMD: "name"),
                           corresponds to l_anm

            * solv_seg   : str, segID name that will represent all solvent molecules

            * error_type   : str
                compute feature errors as ...
                | "std_err" : ... standard errors
                | "std_dev" : ... mean standard deviations

            * max_mom_ord   : int, default: 1
                              maximum ordinal of moment to compute
                              if max_mom_ord > 1, this will add additional entries
                              "mf.2", "sf.2", ..., "mf.%d" % max_mom_ord, "sf.%d" % max_mom_ord
                              to the feature tables

            * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                                  regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
              df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                              'seg' : ["A", "A", "B", "B", "C"],
                                              'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                                  if None, no coarse-graining is performed

            * rgn_agg_func  : function or str for coarse-graining
                              function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                              this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                              - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                              - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                                such as "mean", "std"

            * df_hist_feats : pandas.DataFrame, default=None
                              data frame of features, for which to compute histograms.
                              .columns are self.l_lbl[self.feature_func_name] + ["dbin"], e.g.:
                              df_hist_feats = pd.DataFrame( { "seg1" : ["A", "A"],
                                                              "res1" : [5, 10],
                                                              "seg2" : ["A", "A"],
                                                              "res2" : [10, 15],
                                                              "dbin" : [0.1, 0.1] })
                              dbin is the histogram binning resolution in units of the feature type.
                              Only dbin values are allowed, which
                              sum exactly to the next significant digit's unit, e.g.:
                              for dbin = 0.02 = 2*10^-2 exists an n = 10, so that
                              n * dbin = 0.1  = 1*10^-1
                              Currently - for simplicity - dbin values have to be
                              the same for each feature.
                              If df_hist_feats == dbin (i.e. an int or float), 
                              compute histograms for all features with
                              uniform histogram binning resolution dbin.
       """
        l_lbl1 = ["seg1", "res1", "rnm1", "bb1", "anm1"]
        l_lbl2 = ["seg2", "res2", "rnm2", "bb2", "anm2"]
        l_lbl = l_lbl1[:-1] + l_lbl2[:-1]
    
        fself, myens, r                             = args
        l_solv_rnm                                  = params["l_solv_rnm"]
        l_anm                                       = params["l_anm"]
        l_rad                                       = params["l_rad"]
        solv_seg                                    = params["solv_seg"]
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        is_with_dwell_times                         = params["is_with_dwell_times"]
        is_correlation                              = params.get("is_correlation", False)
        _finish_traj_df                             = params["_finish_traj_df"]

        s_solv_seg   = "_".join(solv_seg)
        s_solv_rnm   = "_".join(l_solv_rnm)
        s_anm        = "_".join(l_anm)
        s_rad        = "_".join([str(f) for f in l_rad])
    
        indir = "input/%s/r_%05d" % (myens, r)
        instem = "%s.r_%05d.noh" % (myens, r)
        outdir = "output/%s/r_%05d/%s/%s" % (myens, r, fself.feature_func_name, fself.intrajdatatype)

        df_pdb = fself._get_raw_topology_ids('%s/%s.pdb' % (indir, instem), "residue")
        df_pdb1 = df_pdb.copy()
        df_pdb1.columns = ["seg1", "res1", "rnm12"]
        df_pdb2 = df_pdb.copy()
        df_pdb2.columns = ["seg2", "res2", "rnm22"]
        # Hvvdwdist()
        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()
        mycmd = "vmd -dispdev text -e %s/features/scripts/compute_sPBSF.hvvdwdist.tcl -args %s %s %s %s %s %s %s %s" \
                % (fself.pkg_dir, indir, instem, fself.intrajformat, outdir, s_solv_seg, s_solv_rnm, s_anm, s_rad)
        print(mycmd)
        outfile = open("%s/log.compute_sPBSFs.hvvdwdist.tcl.log" % outdir, "w")
        myproc = _subprocess.Popen(_shlex.split(mycmd), stdout=outfile, stderr=outfile)
        myproc.wait()
        outfile.close()
        traj_df_hv = _pd.read_csv("%s/%s.sPBSF.hvvdwdist.dat" % (outdir, instem), sep=' ',
                                  names=['frame'] + l_lbl)[l_lbl + ['frame']]
        traj_df_hv = traj_df_hv.query("not ((seg1 == seg2) and (abs(res2 - res1) <= 4))").copy()

        def order(df, blockpair):
            # order contact pair IDs blockwise: (seg1,res1,rnm1,bb1)<->(seg2,res2,rnm2,bb2)
            # by successively ordering as seg1<seg2, if "seg1==seg2": res1<res2,
            #                                        if "seg1==seg2" and "res1==res2": rnm1<rnm2,...
            blockpair_01 = [ x for y in blockpair       for x in y ]
            blockpair_10 = [ x for y in blockpair[::-1] for x in y ]
            pairs = _np.transpose(blockpair)
            cumboolmask = _pd.Series(_np.ones(len(df), dtype="bool"))
            for mypair in pairs:
                boolmask    = _np.logical_and((df[mypair[0]]   >= df[mypair[1]]), cumboolmask)
                if not boolmask.any():
                    break
                df.loc[boolmask, blockpair_01] = df.loc[boolmask, blockpair_10].values
                cumboolmask = _np.logical_and((df[mypair[0]]  == df[mypair[1]]), cumboolmask)
        blockpair    = [ l_lbl1[:-1], l_lbl2[:-1] ]
        order(traj_df_hv, blockpair)

        # HBPLUS hydrogen bond
        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()
        _subprocess.Popen(_shlex.split("rm -rf %s/hbplus" % outdir)).wait()
        mycmd = "%s/features/scripts/compute_PI.hbplus.sh %s %s %s %s %s" % (
            fself.pkg_dir, fself.pkg_dir, indir, instem, fself.intrajformat, outdir)
        myproc = _subprocess.Popen(_shlex.split(mycmd))
        myproc.wait()
    
        numframes = len(_glob.glob("%s/hbplus/hbplus/*.hb2" % outdir))
        if numframes != (traj_df_hv['frame'].max() + 1):
            raise ValueError("HBPLUS numframes not equal Hvvdwdist numframes!")
        traj_df_hb = []
        for f in range(numframes):
            inhb = "%s/hbplus/hbplus/%s_tmp.%05d.hb2" % (outdir, instem, f)
            mycmd = "awk '/^[A-Z][0-9]/ {print substr($0,1,1)\" \"substr($0,2,4)\" \"substr($0,7,3)\" \"$2\" \
\"substr($0,15,1)\" \"substr($0,16,4)\" \"substr($0,21,3)\" \"$4}' %s" % (
                inhb)
            myproc = _subprocess.Popen(_shlex.split(mycmd), stdout=_subprocess.PIPE)
            myproc.wait()
            traj_df_hb.append(
                _pd.read_table(myproc.stdout, sep=" ", header=None, names=l_lbl))
            myproc.stdout.close()
            traj_df_hb[-1][['bb1', 'bb2']] = _np.vectorize(fself.is_bb)(traj_df_hb[-1][['bb1', 'bb2']])
            #traj_df_hb[-1][['rnm1', 'rnm2']] = _np.vectorize(
            # fself.rnm2pdbrnm.__getitem__)(traj_df_hb[-1][['rnm1', 'rnm2']])
            traj_df_hb[-1]['frame'] = f
            traj_df_hb[-1].drop_duplicates(inplace = True)
            #if "bb" in df_rgn_seg_res_bb.columns:
            #    traj_df_seg1_res1 = traj_df[["seg1", "res1", "bb1"]].drop_duplicates()
            #    traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res", "bb1" : "bb"}, inplace = True)
            #    traj_df_seg2_res2 = traj_df[["seg2", "res2", "bb2"]].drop_duplicates()
            #    traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res", "bb2" : "bb"}, inplace = True)
            #else:
            #    traj_df_seg1_res1 = traj_df[["seg1", "res1"]].drop_duplicates()
            #    traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res"}, inplace = True)
            #    traj_df_seg2_res2 = traj_df[["seg2", "res2"]].drop_duplicates()
            #    traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res"}, inplace = True)
            #traj_df_seg_res = _pd.concat([traj_df_seg1_res1, traj_df_seg2_res2]).drop_duplicates()
            #df_merge = traj_df_seg_res.merge(df_pdb, how = "outer", copy = False) 
            #df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
            #if len(df_merge) > 0:
            #    warnstr = "residue name mismatch!\n%s" % df_merge
            #    _warnings.warn(warnstr) 
            traj_df_hb[-1] = traj_df_hb[-1].merge(df_pdb1, copy = False)
            traj_df_hb[-1] = traj_df_hb[-1].merge(df_pdb2, copy = False)
            traj_df_hb[-1].drop(["rnm1", "rnm2"], axis=1, inplace = True)
            traj_df_hb[-1].rename(columns={'rnm12': 'rnm1', 'rnm22': 'rnm2'}, inplace = True)

        traj_df_hb = _pd.concat(traj_df_hb, copy=False)[l_lbl + ['frame']].reset_index(drop=True)
        order(traj_df_hb, blockpair)

        traj_df = _pd.concat([traj_df_hv, traj_df_hb])
        traj_df.drop_duplicates(inplace = True)

        #traj_df['f'] = 1. / numframes
        # now, number of contacts, later mean of number of contacts = frequency
        traj_df['f'] = 1.
        return _finish_traj_df(fself, l_lbl, traj_df, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats, is_with_dwell_times, is_correlation, r)
