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
        Pairwise sparse Pairwise Backbone/Sidechain Features (PsPBSF)
=======================================
"""

# only necessary for Python 2
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import warnings as _warnings
import numpy as _np
import pandas as _pd
import itertools as _itertools

from pysfd.features import _feature_agent


class _PsPBSF(_feature_agent.FeatureAgent):
    """
    Pairwise sparse Pairwise Backbone/Sidechain Feature (PsPBSF):
    Intermediary class between a particular feature class in this module and
    _feature_agent.FeatureAgent
    in order to bundle common tasks
    """
    def __init__(self, feature_name, error_type, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats = None, max_mom_ord = 1, **params):
        super(_PsPBSF, self).__init__(feature_name      = feature_name,
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


class _PsPBSF_Correlation(_PsPBSF):
    """
    Pairwise Backbone/Sidechain Feature Correlation:
    Intermediary class between a particular feature class in this module and
    _feature_agent.FeatureAgent
    in order to bundle common tasks
    """
    def __init__(self, feature_name, partial_corr, error_type,
                 df_rgn_seg_res_bb, rgn_agg_func, label, **params):
        s_coarse = ""
        if df_rgn_seg_res_bb is not None:
            s_coarse = "coarse."
        params["partial_corr"] = partial_corr
        super(_PsPBSF_Correlation, self).__init__(feature_name      = feature_name + s_coarse + error_type + label,
                                                  error_type        = error_type,
                                                  df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                                  rgn_agg_func      = rgn_agg_func,
                                                  **params)

    @staticmethod
    def _feature_func_engine(myf, args, params):
        """
        Computes Pairwise sparse Backbone/Sidechain Feature (partial) correlations
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

        * myf          : function, with which to compute feature-to-feature distance

        Returns
        -------
        * traj_df        : pandas.DataFrame, contains all the feature values accumulated for this replica

        * dataflags      : dict, contains flags with more information about the data in traj_df
        """

        fself, myens, r = args
        if params["error_type"] == "std_dev":
            print("WARNING: error_type \"std_dev\" not defined in _PsPBSF_Correlation !"
                  " Falling back to \"std_err\" instead ...")
            params["error_type"] = "std_err"
        fself.error_type[fself._feature_func_name] = params["error_type"]
        fself.partial_corr                         = params["partial_corr"]
        df_rgn_seg_res_bb                          = params["df_rgn_seg_res_bb"]
        rgn_agg_func                               = params["rgn_agg_func"]
        #traj_df, corr, a_0ind1, a_0ind2            = myf(params["sPBSF_class"], args, params)
        full_traj, sub_dataflags = myf(params["sPBSF_class"], args)
        l_lbl = sub_dataflags["l_lbl"]
        traj_df = full_traj.transpose().corr()

        dataflags = { "error_type" : fself.error_type[fself._feature_func_name] }

        if fself.partial_corr:
            cinv      = _np.linalg.pinv(traj_df.values)
            cinv_diag = _np.diag(cinv)
            # square root of self inverse correlations
            scinv     = _np.sqrt(_np.repeat([cinv_diag], len(cinv_diag), axis = 0))
            #pcorr    = - cinv[i,j] / _np.sqrt(cinv[i,i] * cinv[j,j])
            #corr      = - cinv / scinv / scinv.transpose()
            traj_df   = _pd.DataFrame(- cinv / scinv / scinv.transpose(), index = traj_df.index, columns = traj_df.columns)

        traj_df.index.name = "bspair1"
        traj_df.columns.name = "bspair2"
        traj_df = traj_df.stack(dropna = False).to_frame().reset_index()
        traj_df.columns = ["bspair1", "bspair2", "f"]
        if df_rgn_seg_res_bb is None:
            traj_df.columns = ['bspair1', 'bspair2', 'f']
        elif df_rgn_seg_res_bb is not None:
            if sub_dataflags["df_rgn_seg_res_bb"] is not None:
                raise TypeError("cannot coarse-grain coarse-grained PBSF feature!")
            df_rgn_seg_res_bb["res"] = df_rgn_seg_res_bb["res"].astype("str")
            if "bb" in df_rgn_seg_res_bb.columns:
                df_rgn1_seg1_res1 = df_rgn_seg_res_bb.copy()
                df_rgn1_seg1_res1.columns = ["rgn1", "seg1", "res1", "bb1"]
                df_rgn2_seg2_res2 = df_rgn_seg_res_bb.copy()
                df_rgn2_seg2_res2.columns = ["rgn2", "seg2", "res2", "bb2"]
                df_rgn3_seg3_res3 = df_rgn_seg_res_bb.copy()
                df_rgn3_seg3_res3.columns = ["rgn3", "seg3", "res3", "bb3"]
                df_rgn4_seg4_res4 = df_rgn_seg_res_bb.copy()
                df_rgn4_seg4_res4.columns = ["rgn4", "seg4", "res4", "bb4"]
            else:
                df_rgn1_seg1_res1 = df_rgn_seg_res_bb.copy()
                df_rgn1_seg1_res1.columns = ["rgn1", "seg1", "res1"]
                df_rgn2_seg2_res2 = df_rgn_seg_res_bb.copy()
                df_rgn2_seg2_res2.columns = ["rgn2", "seg2", "res2"]
                df_rgn3_seg3_res3 = df_rgn_seg_res_bb.copy()
                df_rgn3_seg3_res3.columns = ["rgn3", "seg3", "res3"]
                df_rgn4_seg4_res4 = df_rgn_seg_res_bb.copy()
                df_rgn4_seg4_res4.columns = ["rgn4", "seg4", "res4"]

            if fself.error_type[fself._feature_func_name] == "std_err":
                df_tmp = traj_df["bspair1"].str.split('_', 6, expand = True)
                df_tmp.columns = ["seg1", "res1", "bb1", "seg2", "res2", "bb2"]
                traj_df = _pd.concat([traj_df, df_tmp], axis = 1, copy = True)
                df_tmp = traj_df["bspair2"].str.split('_', 6, expand = True)
                df_tmp.columns = ["seg3", "res3", "bb3", "seg4", "res4", "bb4"]
                traj_df = _pd.concat([traj_df, df_tmp], axis = 1)
                if "bb" in df_rgn_seg_res_bb.columns:
                    traj_df_seg1_res1 = traj_df[["seg1", "res1", "bb1"]].drop_duplicates()
                    traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res", "bb1" : "bb"}, inplace = True)
                    traj_df_seg2_res2 = traj_df[["seg2", "res2", "bb2"]].drop_duplicates()
                    traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res", "bb2" : "bb"}, inplace = True)
                    traj_df_seg3_res3 = traj_df[["seg3", "res3", "bb3"]].drop_duplicates()
                    traj_df_seg3_res3.rename(columns={"seg3" : "seg", "res3" : "res", "bb3" : "bb"}, inplace = True)
                    traj_df_seg4_res4 = traj_df[["seg4", "res4", "bb4"]].drop_duplicates()
                    traj_df_seg4_res4.rename(columns={"seg4" : "seg", "res4" : "res", "bb4" : "bb"}, inplace = True)
                    df_rgn1_seg1_res1 = df_rgn_seg_res_bb.copy()
                    df_rgn1_seg1_res1.columns = ["rgn1", "seg1", "res1", "bb1"]
                    df_rgn2_seg2_res2 = df_rgn_seg_res_bb.copy()
                    df_rgn2_seg2_res2.columns = ["rgn2", "seg2", "res2", "bb2"]
                    df_rgn3_seg3_res3 = df_rgn_seg_res_bb.copy()
                    df_rgn3_seg3_res3.columns = ["rgn3", "seg3", "res3", "bb3"]
                    df_rgn4_seg4_res4 = df_rgn_seg_res_bb.copy()
                    df_rgn4_seg4_res4.columns = ["rgn4", "seg4", "res4", "bb4"]
                else:
                    traj_df_seg1_res1 = traj_df[["seg1", "res1"]].drop_duplicates()
                    traj_df_seg1_res1.rename(columns={"seg1" : "seg", "res1" : "res"}, inplace = True)
                    traj_df_seg2_res2 = traj_df[["seg2", "res2"]].drop_duplicates()
                    traj_df_seg2_res2.rename(columns={"seg2" : "seg", "res2" : "res"}, inplace = True)
                    traj_df_seg3_res3 = traj_df[["seg3", "res3"]].drop_duplicates()
                    traj_df_seg3_res3.rename(columns={"seg3" : "seg", "res3" : "res"}, inplace = True)
                    traj_df_seg4_res4 = traj_df[["seg4", "res4"]].drop_duplicates()
                    traj_df_seg4_res4.rename(columns={"seg4" : "seg", "res4" : "res"}, inplace = True)
                    df_rgn1_seg1_res1 = df_rgn_seg_res_bb.copy()
                    df_rgn1_seg1_res1.columns = ["rgn1", "seg1", "res1"]
                    df_rgn2_seg2_res2 = df_rgn_seg_res_bb.copy()
                    df_rgn2_seg2_res2.columns = ["rgn2", "seg2", "res2"]
                    df_rgn3_seg3_res3 = df_rgn_seg_res_bb.copy()
                    df_rgn3_seg3_res3.columns = ["rgn3", "seg3", "res3"]
                    df_rgn4_seg4_res4 = df_rgn_seg_res_bb.copy()
                    df_rgn4_seg4_res4.columns = ["rgn4", "seg4", "res4"]
                traj_df_seg_res = _pd.concat([traj_df_seg1_res1,
                                              traj_df_seg2_res2,
                                              traj_df_seg3_res3,
                                              traj_df_seg4_res4]).drop_duplicates()
                df_merge = traj_df_seg_res.merge(df_rgn_seg_res_bb, how = "outer", copy = False)
                df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
                if len(df_merge) > 0:
                    warnstr = "not-defined resIDs in df_rgn_seg_res_bb " \
                              "(your definition for coarse-graining):\n%s" % df_merge
                    _warnings.warn(warnstr)
                traj_df = traj_df.merge(df_rgn1_seg1_res1, copy = False)
                traj_df = traj_df.merge(df_rgn2_seg2_res2, copy = False)
                traj_df = traj_df.merge(df_rgn3_seg3_res3, copy = False)
                traj_df = traj_df.merge(df_rgn4_seg4_res4, copy = False)
                traj_df.set_index(["rgn1", "rgn2", "rgn3", "rgn4"] + l_lbl, inplace = True)
                traj_df = traj_df.groupby(["rgn1", "rgn2", "rgn3", "rgn4"]).agg({ 'f' : rgn_agg_func })
            traj_df.reset_index(inplace = True)
        traj_df['r'] = r
        #traj_df.reset_index(inplace = True)
        return traj_df, dataflags


class sPBSF_Correlation(_PsPBSF_Correlation):
    """
    Computes (partial) correlations of sPBSF() (see spbsf module)
    for a particular simulation with replica index r

    If coarse-graining (via df_rgn_seg_res_bb, see below) into regions,
    by default aggregate via rgn_agg_func = "mean"

    Parameters
    ----------
    * error_type   : str, default="std_err"
        compute feature errors as ...
        | "std_err" : ... standard errors
        | "std_dev" : ... mean standard deviations

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

    def __init__(self, partial_corr = False, error_type = "std_err",
                 df_rgn_seg_res_bb = None, rgn_agg_func = "mean", label = "", sPBSF_class = None):
        if sPBSF_class is None:
            raise ValueError("sPBSF_class not defined")
        s_pcorr = "partial_" if partial_corr else ""
        super(sPBSF_Correlation, self).__init__(
            feature_name      = "pspbsf." + s_pcorr + "correlation." + sPBSF_class.feature_name + ".",
            partial_corr      = partial_corr,
            error_type        = error_type,
            df_rgn_seg_res_bb = df_rgn_seg_res_bb,
            rgn_agg_func      = rgn_agg_func,
            label             = label,
            sPBSF_class       = sPBSF_class)

    @staticmethod
    def _myf(sPBSF_class, args):
        sPBSF_class.params["is_correlation"] = True
        myfeature_func = sPBSF_class.get_feature_func()
        return myfeature_func(args)
