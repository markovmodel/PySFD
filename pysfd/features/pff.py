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
        Pairwise Feature Features (PFFs)
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


class _PFF(_feature_agent.FeatureAgent):
    """
    Pairwise Feature Feature (PFF):
    Intermediary class between a particular feature class in this module and
    _feature_agent.FeatureAgent
    in order to bundle common tasks
    """
    def __init__(self, feature_name, error_type, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats = None, max_mom_ord = 1, **params):
        super(_PFF, self).__init__(feature_name      = feature_name,
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


class _Feature_Correlation(_PFF):
    """
    Feature (partial) Correlation:
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
        super(_Feature_Correlation, self).__init__(feature_name      = feature_name + s_coarse + error_type + label,
                                                   error_type        = error_type,
                                                   df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                                   rgn_agg_func      = rgn_agg_func,
                                                   **params)

    @staticmethod
    def _feature_func_engine(myf, args, params):
        """
        Computes Feature (partial) correlations
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
            print("WARNING: error_type \"std_dev\" not defined in _Feature_Correlation !"
                  " Falling back to \"std_err\" instead ...")
            params["error_type"] = "std_err"
        fself.error_type[fself._feature_func_name] = params["error_type"]
        fself.partial_corr                         = params["partial_corr"]
        df_rgn_seg_res_bb                          = params["df_rgn_seg_res_bb"]
        rgn_agg_func                               = params["rgn_agg_func"]
        traj_df1, _ = myf(params["Feature_class1"], args)
        traj_df1.index.name = "feature1"
        traj_df1.index = "f1_" + traj_df1.index
        if not isinstance(traj_df1, _pd.DataFrame):
            raise TypeError("Feature_class1 returns wrong type! Is is_correlation=True implemented yet for Feature_class1?")
        if params["Feature_class1"] == params["Feature_class2"]:
            df_corr = traj_df1.transpose().corr()
        else:
            traj_df2, _ = myf(params["Feature_class2"], args)
            traj_df2.index.name = "feature2"
            traj_df2.index = "f2_" + traj_df2.index
            if not isinstance(traj_df2, _pd.DataFrame):
                raise TypeError("Feature_class2 returns wrong type! Is is_correlation=True implemented yet for Feature_class2?")
            df_corr = _pd.concat([traj_df1, traj_df2]).transpose().corr()

        dataflags = { "error_type" : fself.error_type[fself._feature_func_name] }

        if fself.partial_corr:
            cinv      = _np.linalg.pinv(df_corr.values)
            cinv_diag = _np.diag(cinv)
            # square root of self inverse correlations
            scinv     = _np.sqrt(_np.repeat([cinv_diag], len(cinv_diag), axis = 0))
            #pcorr    = - cinv[i,j] / _np.sqrt(cinv[i,i] * cinv[j,j])
            df_corr   = _pd.DataFrame(- cinv / scinv / scinv.transpose(), index = df_corr.index, columns = df_corr.columns)
        traj_df = df_corr.stack(dropna = False).to_frame().reset_index()
        #if params["Feature_class1"] == params["Feature_class2"]:
        #    a_pairs   = _np.array(list(_itertools.combinations(traj_df1.index, 2)))
        #    a_ind1    = a_pairs[:,0]
        #    a_ind2    = a_pairs[:,1]
        #    a_0pairs  = _np.array(list(_itertools.combinations(range(len(traj_df1)), 2)))
        #    a_0ind1   = a_0pairs[:,0]
        #    a_0ind2   = a_0pairs[:,1]
        #    traj_df   = _pd.DataFrame(data={'feature1': a_ind1, 'feature2': a_ind2, 'f' : corr[a_0ind1, a_0ind2] })
        #else:
        #    traj_df = _pd.DataFrame(corr, index = myindex, columns = mycolumns).stack()
        #    print(traj_df.head())
        l_lbl = ['feature1', 'feature2']
        traj_df['r'] = r
        traj_df.columns = l_lbl + ["f", "r"]
        return traj_df, dataflags


class Feature_Correlation(_Feature_Correlation):
    """
    Computes Feature (partial) correlations
    for a particular simulation with replica index r

    Note: Here, df_rgn_seg_res_bb is not implemented yet, since this would also require
    the possibility to coarse-grain coarse-grained features.

    Parameters
    ----------
    * error_type   : str, default="std_err"
        compute feature errors as ...
        | "std_err" : ... standard errors
        | "std_dev" : ... mean standard deviations

    * label        : string, user-specific label
    """

    def __init__(self, partial_corr = False, error_type = "std_err",
                 label = "", Feature_class1 = None, Feature_class2 = None):
        if Feature_class1 is None:
            raise ValueError("Feature_class1 not defined")
        if Feature_class2 is None:
            Feature_class2 = Feature_class1

        s_pcorr = "partial_" if partial_corr else ""
        
        super(Feature_Correlation, self).__init__(
            feature_name      = "pff." + s_pcorr + "correlation." + Feature_class1.feature_name + "_vs_" + Feature_class2.feature_name,
            partial_corr      = partial_corr,
            error_type        = error_type,
            df_rgn_seg_res_bb = None,
            rgn_agg_func      = None,
            label             = label,
            Feature_class1    = Feature_class1,
            Feature_class2    = Feature_class2)

    @staticmethod
    def _myf(Feature_class, args):
        Feature_class.params["is_correlation"] = True
        myfeature_func = Feature_class.get_feature_func()
        return myfeature_func(args)
