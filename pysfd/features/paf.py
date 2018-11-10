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
        pairwise atomic features (paf)
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
import itertools as _itertools

# for circular statistics in circcorr in Dihedral_Correlation
import scipy.stats as _scipy_stats

from pysfd.features import _feature_agent


class _PAF(_feature_agent.FeatureAgent):
    """
    Pairwise Atomic Feature (PAF):
    Intermediary class between a particular feature class in this module and
    _feature_agent.FeatureAgent
    in order to bundle common tasks
    """
    def __init__(self, feature_name, error_type, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats = None, max_mom_ord = 1, **params):
        if df_rgn_seg_res_bb is not None:
            if "bb" in df_rgn_seg_res_bb.columns:
                df_rgn_seg_res_bb.drop(columns = "bb", inplace = True)
        super(_PAF, self).__init__(feature_name      = feature_name,
                                   error_type        = error_type,
                                   max_mom_ord       = max_mom_ord,
                                   df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                   rgn_agg_func      = rgn_agg_func,
                                   df_hist_feats     = df_hist_feats,
                                   **params)

    def get_feature_func(self):
        def f(args):
            return self._feature_func_engine(self._myf, args, self.params)
        f.__name__ = self.feature_name
        return f


class _PAF_Distance(_PAF):
    """
    Pairwise Atomic Feature Distance:
    Intermediary class between a particular feature class in this module and
    _feature_agent.FeatureAgent
    in order to bundle common tasks
    """
    def __init__(self, feature_name, error_type, max_mom_ord, df_sel, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats, label, **params):
        s_coarse = ""
        if df_rgn_seg_res_bb is not None:
            s_coarse = "coarse."
        params["df_sel"] = df_sel
        super(_PAF_Distance, self).__init__(
                                  feature_name      = feature_name + s_coarse + error_type + label,
                                  error_type        = error_type,
                                  max_mom_ord       = max_mom_ord,
                                  df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                  rgn_agg_func      = rgn_agg_func,
                                  df_hist_feats     = df_hist_feats,
                                  **params)

    @staticmethod
    def _feature_func_engine(myf, args, params):
        """
        Computes feature-feature distances (e.g. Ca-to-Ca distances)
        for a particular simulation with replica index r
 
        Parameters
        ----------
        * args   : tuple (fself, myens, r):
            * fself        : self pointer to foreign master PySFD object

            * myens        : string
                             Name of simulated ensemble

            * r            : int
                             replica index

        * params : dict, extra parameters as keyword arguments
            * error_type   : str
                compute feature errors as ...
                | "std_err" : ... standard errors
                | "std_dev" : ... mean standard deviations

            * max_mom_ord   : int, default: 1
                              maximum ordinal of moment to compute
                              if max_mom_ord > 1, this will add additional entries
                              "mf.2", "sf.2", ..., "mf.%d" % max_mom_ord, "sf.%d" % max_mom_ord
                              to the feature tables

            * df_sel : pandas.DataFrame, optional, default = None
                       if not None, distances are only computed between atom pairs listed in this DataFrame
                       df_sel.columns = ["seg1", "res1", "anm1", "seg2", "res2", "anm2"]

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

            * is_correlation : bool, optional, whether or not to output feature values
                               for a subsequent correlation analysis (e.g. pff.Feature_Correlation())

        * myf          : function, with which to compute feature-to-feature distance

        Returns
        -------
        * traj_df        : pandas.DataFrame, contains all the feature values accumulated for this replica

        * dataflags      : dict, contains flags with more information about the data in traj_df
        """
        fself, myens, r = args
        error_type                                    = params["error_type"]
        max_mom_ord                                   = params["max_mom_ord"]
        df_rgn_seg_res_bb                             = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                  = params["rgn_agg_func"]
        df_hist_feats                                 = params["df_hist_feats"]
        df_sel                                        = params["df_sel"]

        dataflags = { "error_type"  : error_type, "max_mom_ord" : max_mom_ord }

        mytraj = _md.load('input/%s/r_%05d/%s.r_%05d.prot.%s' % (myens, r, myens, r, fself.intrajformat),
                          top = 'input/%s/r_%05d/%s.r_%05d.prot.pdb' % (myens, r, myens, r))
        l_lbl = ['seg1', 'res1', 'rnm1', 'anm1', 'seg2', 'res2', 'rnm2', 'anm2']

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        a_rnm = fself._get_raw_topology_ids('%s.pdb' % instem, "atom").rnm.values
        a_f, traj_df = myf(mytraj, a_rnm, df_sel)

        def myhist(a_data, dbin):
            prec = len(str(dbin).partition(".")[2])+1
            a_bins =_np.arange(_np.floor(a_data.min() / dbin),
                               _np.ceil(a_data.max() / dbin) + 1, 1) * dbin
            a_hist = _np.histogram(a_data, bins = a_bins, density = True)
            return tuple(list(a_hist))

        if df_rgn_seg_res_bb is None:
            if "is_correlation" in params:
                if params["is_correlation"] == True:
                    traj_df["feature"] = traj_df["seg1"].astype(str) + "_" + \
                                         traj_df["res1"].astype(str) + "_" + \
                                         traj_df["rnm1"].astype(str) + "_" + \
                                         traj_df["anm1"].astype(str) + "_" + \
                                         traj_df["seg2"].astype(str) + "_" + \
                                         traj_df["res2"].astype(str) + "_" + \
                                         traj_df["rnm2"].astype(str) + "_" + \
                                         traj_df["anm2"].astype(str)
                    traj_df.drop(columns = l_lbl, inplace = True)
                    traj_df.set_index("feature", inplace = True)
                    traj_df = _pd.DataFrame(a_f.transpose(), index = traj_df.index)
                    return traj_df, None 

            traj_df['f'] = _np.mean(a_f, axis=0)
            # if include ALL feature entries for histogramming:
            if isinstance(df_hist_feats, (int, float)):
                dbin = df_hist_feats
                traj_df['fhist'] = True
                # label used below to include "fhist" entry:
                l_flbl = ['fhist', 'f']
            # elif include NO feature entries for histogramming:
            elif df_hist_feats is None:
                dbin = None
                traj_df['fhist'] = False
                # label used below to NOT include "fhist" entry:
                l_flbl = ['f']
            # else (if include SOME feature entries for histogramming):
            elif isinstance(df_hist_feats, _pd.DataFrame):
                dbin = df_hist_feats["dbin"][0]
                df_hist_feats['fhist'] = True
                # label used below to include "fhist" entry:
                l_flbl = ['fhist', 'f']
                traj_df = traj_df.merge(df_hist_feats.drop(columns = "dbin"), how = "outer")
                if _np.any(traj_df.loc[traj_df.fhist == True].isnull()):
                    df_error_tmp = traj_df.loc[traj_df.fhist == True] 
                    raise ValueError("ERROR: df_hist_feats is of type _pd.DataFrame, but some \
                    of your feature identifier entries (%s) in df_hist_feats do not seem to \
                    match those in your input trajectories!\n\
                    non-matching entries:\n%s" % (",".join(l_lbl),
                                     df_error_tmp.loc[_np.any(df_error_tmp.isnull(), axis = 1), :]))

            # work-around, since this does not work:
            # traj_df.loc[traj_df.fhist == True, "fhist"] = list(_np.apply_along_axis(lambda x: myhist(x, df_hist_feats["dbin"][0]), axis = 0, arr = a_f[:, traj_df.fhist == True]).transpose())
            mytmp = traj_df.loc[traj_df.fhist == True, "fhist"].copy().to_frame()
            if len(mytmp) > 0:
                mytmp["fhist"] = list(_np.apply_along_axis(lambda x: myhist(x, dbin), axis = 0, arr = a_f[:, traj_df.fhist == True]).transpose())
                traj_df.loc[traj_df.fhist == True, "fhist"] = mytmp["fhist"]
            for mymom in range(2, max_mom_ord+1):
                traj_df['f.%d' % mymom] = _scipy_stats.moment(a_f, axis=0, moment = mymom)
                l_flbl += ['f.%d' % mymom]
            traj_df = traj_df[l_lbl + l_flbl].copy()
            if error_type == "std_dev":
                # correction factor to convert numpy.std into pandas.std
                std_factor = _np.sqrt(_np.shape(a_f)[0] / (_np.shape(a_f)[0] - 1.))
                traj_df['sf'] = _np.std(a_f, axis=0) * std_factor
        elif df_rgn_seg_res_bb is not None:
            traj_df_seg1_res1 = traj_df[["seg1", "res1"]].drop_duplicates()
            traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res"}, inplace = True)
            traj_df_seg2_res2 = traj_df[["seg2", "res2"]].drop_duplicates()
            traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res"}, inplace = True)
            traj_df_seg_res = _pd.concat([traj_df_seg1_res1, traj_df_seg2_res2]).drop_duplicates()
            df_merge = traj_df_seg_res.merge(df_rgn_seg_res_bb, how = "outer", copy = False)
            df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
            if len(df_merge) > 0:
                warnstr = "not-defined resIDs in df_rgn_seg_res_bb (your definition for coarse-graining):\n%s" % df_merge
                _warnings.warn(warnstr)
            df_rgn1_seg1_res1 = df_rgn_seg_res_bb.copy()
            df_rgn1_seg1_res1.columns = ['rgn1', 'seg1', 'res1']
            df_rgn2_seg2_res2 = df_rgn_seg_res_bb.copy()
            df_rgn2_seg2_res2.columns = ['rgn2', 'seg2', 'res2']

            traj_df = traj_df[l_lbl]
            traj_df = _pd.concat([traj_df, _pd.DataFrame(_np.transpose(a_f))], axis = 1, copy = False)
            traj_df = traj_df.merge(df_rgn1_seg1_res1, copy = False)
            traj_df = traj_df.merge(df_rgn2_seg2_res2, copy = False)
            traj_df.set_index(["rgn1", "rgn2"] + l_lbl, inplace = True)
            #print(traj_df.query("rgn1 == 'a1L1' and rgn2 == 'a1L1'"))
            traj_df = traj_df.stack()
            traj_df = traj_df.to_frame().reset_index()
            traj_df.columns = ["rgn1", "rgn2"] + l_lbl + ["frame", "f"]
            traj_df.set_index(["rgn1", "rgn2", "frame"] + l_lbl, inplace = True)
            #print(traj_df.query("rgn1 == 'a1L1' and rgn2 == 'a1L1'"))
            # computes the mean distance between two regions in each frame:
            traj_df = traj_df.groupby(["rgn1", "rgn2", "frame"]).agg( { "f" : rgn_agg_func } )
            #print(traj_df.query("rgn1 == 'a1L1' and rgn2 == 'a1L1'"))
            if "is_correlation" in params:
                if params["is_correlation"] == True:
                    traj_df = traj_df.unstack()
                    traj_df.columns = traj_df.columns.get_level_values(1)
                    traj_df.columns.name = None
                    traj_df.reset_index(inplace = True)
                    traj_df["feature"] = traj_df["rgn1"] + "_" + traj_df["rgn2"]
                    traj_df.drop(columns =["rgn1", "rgn2"], inplace = True)
                    traj_df.set_index("feature", inplace = True)
                    return traj_df, None 

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
                if _np.any(traj_df_hist.isnull()):
                    raise ValueError("ERROR: df_hist_feats is of type _pd.DataFrame, but some \
                    of your feature identifier entries (%s) in df_hist_feats do not seem to \
                    match those in df_rgn_seg_res_bb or your input trajectories!\n \
                    non-matching entries:\n%s" % (",".join(["rgn1", "rgn2"]),
                                                  traj_df_hist.loc[_np.any(traj_df_hist.isnull(), axis = 1), :]))
            if traj_df_hist is not None:
                traj_df_hist = traj_df_hist.groupby(["rgn1", "rgn2"]).agg( { "f" : lambda x: myhist(x, dbin ) } )
                traj_df_hist.rename(columns = { "f" : "fhist" }, inplace = True)

            if error_type == "std_err":
                l_func = ['mean'] + [ lambda x: _scipy_stats.moment(x, moment = mymom) for mymom in range(2, max_mom_ord+1)]
                l_lbl = ['f'] + [ 'f.%d' % mymom for mymom in range(2, max_mom_ord+1)]
                traj_df = traj_df.groupby(["rgn1", "rgn2"]).agg( l_func )
                traj_df.columns = l_lbl
            elif error_type == "std_dev":
                # mean/std over the frames...: 
                traj_df = traj_df.groupby(["rgn1", "rgn2"]).agg( ["mean", "std"])
                traj_df.columns = traj_df.columns.droplevel(level = 0)
                traj_df.columns = ["f", "sf"]
            if traj_df_hist is not None:
                traj_df = traj_df.merge(traj_df_hist, left_index = True, right_index = True, how = "outer")
            traj_df.reset_index(inplace = True, drop = False)
        traj_df['r'] = r
        #traj_df['r'] = _pd.Series(len(traj_df) * [r])
        #traj_df.reset_index(inplace = True)
        return traj_df, dataflags


class _PAF_Correlation(_PAF):
    """
    Pairwise Atomic Feature Correlation:
    Intermediary class between a particular feature class in this module and
    _feature_agent.FeatureAgent
    in order to bundle common tasks
    """
    def __init__(self, feature_name, partial_corr, error_type, df_sel,
                 df_rgn_seg_res_bb, rgn_agg_func, label, **params):
        s_coarse = ""
        if df_rgn_seg_res_bb is not None:
            s_coarse = "coarse."
        params["partial_corr"] = partial_corr
        params["df_sel"] = df_sel

        super(_PAF_Correlation, self).__init__(feature_name      = feature_name + s_coarse + error_type + label,
                                               error_type        = error_type,
                                               df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                               rgn_agg_func      = rgn_agg_func,
                                               **params)

    @staticmethod
    def _feature_func_engine(myf, args, params):
        """
        Computes pairwise residual feature (partial) correlations
        for a particular simulation with replica index r
 
        Parameters
        ----------
        * args   : tuple (fself, myens, r):
            * fself        : self pointer to foreign master PySFD object

            * myens        : string
                             Name of simulated ensemble

            * r            : int
                             replica index

        * params : dict, extra parameters as keyword arguments
            * error_type   : str
                compute feature errors as ...
                | "std_err" : ... standard errors
                | "std_dev" : ... mean standard deviations

            * df_sel : pandas.DataFrame, optional, default = None
                       if not None, distances are only computed between atom pairs listed in this DataFrame
                       df_sel.columns = ["seg1", "res1", "anm1", "seg2", "res2", "anm2"]

            * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                                  regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
              df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                              'seg' : ["A", "A", "B", "B", "C"],
                                              'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                                  if None, no coarse-graining is performed
                                  !!!
                                  Note: of course, coarse-graining cannot be performed here in
                                        individual frames, but over correlation coefficients
                                  !!! 
    
            * rgn_agg_func  : function or str for coarse-graining
                              function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                              this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                              - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                              - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                                such as "mean", "std"

        * myf              : function, with which to compute feature-to-feature distance

        Returns
        -------
        * traj_df        : pandas.DataFrame, contains all the feature values accumulated for this replica
        * dataflags      : dict, contains flags with more information about the data in traj_df
        """
        fself, myens, r = args
        if params["error_type"] == "std_dev":
            print("WARNING: error_type \"std_dev\" not defined in _PAF_Correlation!"
                  " Falling back to \"std_err\" instead ...")
            params["error_type"] = "std_err"
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.partial_corr                          = params["partial_corr"]
        df_sel                                      = params["df_sel"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]

        dataflags = { "error_type" : fself.error_type[fself._feature_func_name] }

        mytraj = _md.load('input/%s/r_%05d/%s.r_%05d.prot.%s' % (myens, r, myens, r, fself.intrajformat),
                          top='input/%s/r_%05d/%s.r_%05d.prot.pdb' % (myens, r, myens, r))

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        a_rnm = fself._get_raw_topology_ids('%s.pdb' % instem, "atom").rnm.values
        if "feat_subfunc" in params:
            traj_df, corr, a_0ind1, a_0ind2 = myf(mytraj, a_rnm, df_sel, params["feat_subfunc"])
        else:
            traj_df, corr, a_0ind1, a_0ind2 = myf(mytraj, a_rnm, df_sel)

        if fself.partial_corr:
            cinv      = _np.linalg.pinv(corr)
            cinv_diag = _np.diag(cinv)
            # square root of self inverse correlations
            scinv     = _np.sqrt(_np.repeat([cinv_diag], len(cinv_diag), axis = 0))
            #pcorr     = - cinv[i,j] / _np.sqrt(cinv[i,i] * cinv[j,j])
            corr      = - cinv / scinv / scinv.transpose()
        a_f    = corr[a_0ind1, a_0ind2] 
        l_lbl = ['seg1', 'res1', 'rnm1', 'anm1', 'seg2', 'res2', 'rnm2', 'anm2']

        if df_rgn_seg_res_bb is None:
            traj_df['f'] = a_f
            traj_df = traj_df[l_lbl + ['f']].copy()
        elif df_rgn_seg_res_bb is not None:
            traj_df_seg1_res1 = traj_df[["seg1", "res1"]].drop_duplicates()
            traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res"}, inplace = True)
            traj_df_seg2_res2 = traj_df[["seg2", "res2"]].drop_duplicates()
            traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res"}, inplace = True)
            traj_df_seg_res = _pd.concat([traj_df_seg1_res1, traj_df_seg2_res2])
            df_merge = traj_df_seg_res.merge(df_rgn_seg_res_bb, how = "outer", copy = False)
            df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
            if len(df_merge) > 0:
                warnstr = "not-defined resIDs in df_rgn_seg_res_bb (your definition for coarse-graining):\n%s" % df_merge
                _warnings.warn(warnstr)
            df_rgn1_seg1_res1 = df_rgn_seg_res_bb.copy()
            df_rgn1_seg1_res1.columns = ['rgn1', 'seg1', 'res1']
            df_rgn2_seg2_res2 = df_rgn_seg_res_bb.copy()
            df_rgn2_seg2_res2.columns = ['rgn2', 'seg2', 'res2']
            if fself.error_type[fself._feature_func_name] == "std_err":
                traj_df['f'] = a_f
                traj_df = traj_df[l_lbl + ['f']]
                traj_df = traj_df.merge(df_rgn1_seg1_res1, copy = False)
                traj_df = traj_df.merge(df_rgn2_seg2_res2, copy = False)
                traj_df.set_index(["rgn1", "rgn2"] + l_lbl, inplace = True)
                #print(traj_df.query("rgn1 == 'a1L2' and rgn2 == 'a1L2'"))
                #print(traj_df.query("rgn1 == 'a1L1' and rgn2 == 'a1L1'"))
                traj_df = traj_df.groupby(["rgn1", "rgn2"]).agg({ 'f' : rgn_agg_func })
                #print(traj_df.query("rgn1 == 'a1L2' and rgn2 == 'a1L2'"))
                #print(traj_df.query("rgn1 == 'a1L1' and rgn2 == 'a1L1'"))
        traj_df.reset_index(inplace = True)
        traj_df['r'] = r
        #traj_df.reset_index(inplace = True)
        return traj_df, dataflags


class Atm2Atm_Distance(_PAF_Distance):
    """
    Computes atom-to-atom distances (in units of nm)
    for a particular simulation with replica index r

    If coarse-graining (via df_rgn_seg_res_bb, see below) into regions,
    by default aggregate via rgn_agg_func = "mean"

    Parameters
    ----------
    * error_type   : str, default="std_err"
        compute feature errors as ...
        | "std_err" : ... standard errors
        | "std_dev" : ... mean standard deviations

    * max_mom_ord   : int, default: 1
                      maximum ordinal of moment to compute
                      if max_mom_ord > 1, this will add additional entries
                      "mf.2", "sf.2", ..., "mf.%d" % max_mom_ord, "sf.%d" % max_mom_ord
                      to the feature tables

    * df_sel : pandas.DataFrame, optional, default = None
               if not None, distances are only computed between atom pairs listed in this DataFrame
               df_sel.columns = ["seg1", "res1", "anm1", "seg2", "res2", "anm2"]

    * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                          regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
      df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                      'seg' : ["A", "A", "B", "B", "C"],
                                      'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                          if None, no coarse-graining is performed
                          !!!
                          Note: of course, coarse-graining cannot be performed here in
                                individual frames, but over correlation coefficients
                          !!! 

    * rgn_agg_func  : function or str for coarse-graining, default = "mean"
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

    * label        : string, user-specific label

    ---------
    you can verify the results from this code, e.g., in VMD via:
    
    mol load pdb WT.pcca3.r_00000.prot.pdb
    animate delete all
    animate read xtc WT.pcca3.r_00000.prot.xtc
    set mysel1 [atomselect top "name CA and chain A and resid 4"]
    set mysel2 [atomselect top "name CA and chain A and resid 6"]
    set l_dist [measure bond [list [$mysel1 get index] [$mysel2 get index]] frame all]
    vecmean $l_dist 
    # standard deviation in VMD is not using "1/sqrt(N-1)" correction
    vecstddev $l_dist 
    """

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_sel = None, df_rgn_seg_res_bb = None, rgn_agg_func = "mean", df_hist_feats = None, label = ""):
        super(Atm2Atm_Distance, self).__init__(
            feature_name      = "paf.distance.Atm2Atm.",
            error_type        = error_type,
            max_mom_ord       = max_mom_ord,
            df_sel            = df_sel,
            df_rgn_seg_res_bb = df_rgn_seg_res_bb,
            rgn_agg_func      = rgn_agg_func,
            df_hist_feats     = df_hist_feats,
            label             = label)

    @staticmethod
    def _myf	(mytraj, a_rnm, df_sel):
        if df_sel is None:
            atmselstr = "name CA"
            a_index = mytraj.topology.select(atmselstr)
            a_pairs = _np.array(list(_itertools.combinations(a_index, 2)))
            a_ind1  = a_pairs[:,0]
            a_ind2  = a_pairs[:,1]
        else:
            df_top  = mytraj.topology.to_dataframe()[0].loc[:, ["segmentID", "resSeq", "name"]].reset_index()
            df_top.columns = ["ind1", "seg1", "res1", "anm1"]
            print(df_top.head())
            print(df_sel.head())
            newdf_sel  = df_sel.merge(df_top)
            df_top.columns = ["ind2", "seg2", "res2", "anm2"]
            newdf_sel  = newdf_sel.merge(df_top)
            if len(newdf_sel) != len(df_sel):
                df_err = df_sel.merge(newdf_sel, indicator=True, how='outer')
                df_err = df_err.query("_merge != 'both'")
                warnstr = "check your entries in df_sel: not all match with input topology!\n%s\ncontinuing with common entries ..." % df_err
                _warnings.warn(warnstr)
            a_ind1  = newdf_sel.ind1.values
            a_ind2  = newdf_sel.ind2.values
            a_pairs = newdf_sel.loc[:, ["ind1", "ind2"]].values
           
        a_atom = list(mytraj.topology.atoms)
        a_seg  = _np.array([a.segment_id for a in a_atom])
        a_res  = _np.array([a.residue.resSeq for a in a_atom])
        a_anm  = _np.array([a.name for a in a_atom])
        #a_rnm  = _np.array([a.residue.name for a in a_atom])
        a_f     = _md.compute_distances(mytraj, atom_pairs = a_pairs, periodic = True, opt = True)
        traj_df  = _pd.DataFrame(data={'seg1': a_seg[a_ind1], 'rnm1': a_rnm[a_ind1], 'res1': a_res[a_ind1], 'anm1' : a_anm[a_ind1],
                                       'seg2': a_seg[a_ind2], 'rnm2': a_rnm[a_ind2], 'res2': a_res[a_ind2], 'anm2' : a_anm[a_ind2] })
        return a_f, traj_df


class AtmPos_Correlation(_PAF_Correlation):
    """
    Computes pairwise atom position (partial) correlations
    for a particular simulation with replica index r

    If coarse-graining (via df_rgn_seg_res_bb, see below) into regions,
    by default aggregate via rgn_agg_func = "mean"

    Parameters
    ----------
    * error_type   : str, default="std_err"
        compute feature errors as ...
        | "std_err" : ... standard errors
        | "std_dev" : ... mean standard deviations

    * df_sel : pandas.DataFrame, optional, default = None
               if not None, distances are only computed between atom pairs listed in this DataFrame
               df_sel.columns = ["seg1", "res1", "anm1", "seg2", "res2", "anm2"]

    * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                          regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
      df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                      'seg' : ["A", "A", "B", "B", "C"],
                                      'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                          if None, no coarse-graining is performed
                          !!!
                          Note: of course, coarse-graining cannot be performed here in
                                individual frames, but over correlation coefficients
                          !!! 

    * rgn_agg_func  : function or str for coarse-graining, default = "mean"
                      function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                      this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                      - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                      - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                        such as "mean", "std"

    * label        : string, user-specific label
    """

    def __init__(self, partial_corr = False, error_type = "std_err", df_sel = None, df_rgn_seg_res_bb = None, rgn_agg_func = "mean", label = ""):
        s_pcorr = "partial_" if partial_corr else ""
        super(AtmPos_Correlation, self).__init__(
            feature_name      = "paf." + s_pcorr + "correlation.AtmPos.",
            partial_corr      = partial_corr,
            error_type        = error_type,
            df_sel            = df_sel,
            df_rgn_seg_res_bb = df_rgn_seg_res_bb,
            rgn_agg_func      = rgn_agg_func,
            label             = label)

    @staticmethod
    def _myf(mytraj, a_rnm, df_sel):
        a_atom    = list(mytraj.topology.atoms)
        a_seg     = _np.array([a.segment_id for a in a_atom])
        a_res     = _np.array([a.residue.resSeq for a in a_atom])
        a_anm     = _np.array([a.name for a in a_atom])
        #a_rnm    = _np.array([a.residue.name for a in a_atom])
        if df_sel is None:
            atmselstr = "name CA"
            a_index   = mytraj.topology.select(atmselstr)
            a_pairs   = _np.array(list(_itertools.combinations(a_index, 2)))
            a_ind1    = a_pairs[:,0]
            a_ind2    = a_pairs[:,1]
            a_0pairs  = _np.array(list(_itertools.combinations(range(len(a_index)), 2)))
            a_0ind1   = a_0pairs[:,0]
            a_0ind2   = a_0pairs[:,1]
        else:
            df_top  = mytraj.topology.to_dataframe()[0].loc[:, ["segmentID", "resSeq", "name"]].reset_index()
            df_top.columns = ["ind1", "seg1", "res1", "anm1"]
            newdf_sel  = df_sel.merge(df_top)
            df_top.columns = ["ind2", "seg2", "res2", "anm2"]
            newdf_sel  = newdf_sel.merge(df_top)
            if len(newdf_sel) != len(df_sel):
                df_err = df_sel.merge(newdf_sel, indicator=True, how='outer')
                df_err = df_err.query("_merge != 'both'")
                warnstr = "check your entries in df_sel: not all match with input topology!\n%s\ncontinuing with common entries ..." % df_err
                _warnings.warn(warnstr)
            a_ind1   = newdf_sel.ind1.values
            a_ind2   = newdf_sel.ind2.values
            a_pairs  = newdf_sel.loc[:, ["ind1", "ind2"]].values
            a_index  = _np.unique(a_pairs)
            # give me the indices in a_index of the atomic indices in a_ind1, a_ind2:
            a_0ind1  = _np.nonzero(a_ind1[:, None] == a_index)[1]
            a_0ind2  = _np.nonzero(a_ind2[:, None] == a_index)[1]
            a_0pairs = _np.array([a_0ind1, a_0ind2])

        xyz       = mytraj.xyz[:, a_index, :]
        x         = _np.transpose(xyz[:,:,0])
        y         = _np.transpose(xyz[:,:,1])
        z         = _np.transpose(xyz[:,:,2])
        cov_x     = _np.cov(x)
        cov_y     = _np.cov(y)
        cov_z     = _np.cov(z)
        # self covariance
        scov_x = _np.diag(cov_x)
        scov_y = _np.diag(cov_y)
        scov_z = _np.diag(cov_z)
        # square root of self covariances
        scov   = _np.sqrt(_np.repeat([scov_x + scov_y + scov_z], len(scov_x), axis = 0))
        corr   = (cov_x + cov_y + cov_z) / scov / scov.transpose()
        traj_df  = _pd.DataFrame(data={'seg1': a_seg[a_ind1], 'rnm1': a_rnm[a_ind1], 'res1': a_res[a_ind1], 'anm1' : a_anm[a_ind1],
                                       'seg2': a_seg[a_ind2], 'rnm2': a_rnm[a_ind2], 'res2': a_res[a_ind2], 'anm2' : a_anm[a_ind2] })
        return traj_df, corr, a_0ind1, a_0ind2

