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
        Feature Agent (main class from which feature classes derive)
=======================================
"""

# only necessary for Python 2
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import pandas as _pd
import numpy  as _np


class FeatureAgent(object):
    """
    FeatureAgent main class

    Prepares the function with customized parameters for PySFD analysis

    Parameters
    ----------
    * feature_name  : str, feature name

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

    * rgn_agg_func  : function or string for coarse-graining
                      function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                      this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                      - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                      - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                        such as "mean", "std"

    * df_hist_feats : pandas.DataFrame, default=None
                      data frame of features, for which to compute histograms.
                      .columns are that of self.l_lbl[self.feature_func_name]
                      (i.e. l_lbl, see below)

    * params         dict, extra parameters as keyword arguments to customize self._feature_func_engine
    """

    def __init__(self, feature_name, error_type, max_mom_ord, df_rgn_seg_res_bb, rgn_agg_func, df_hist_feats, **params):
        self.feature_name = feature_name
        if error_type not in ['std_err', 'std_dev']:
            raise ValueError("ERROR: error_type = %s is not in ['std_err', 'std_dev']" % error_type)
        if (max_mom_ord > 1) and (error_type != 'std_err'):
            print("WARNING: error_type=\"%s\" not defined for max_mom_ord > 1 ! Falling back to \"std_err\" instead ..." % error_type)
            error_type = "std_err"
        params["error_type"]  = error_type
        params["max_mom_ord"] = max_mom_ord

        # prepare input df_rgn_seg_res_bb ...
        if df_rgn_seg_res_bb is not None:
            lens = [len(item) for item in df_rgn_seg_res_bb['res']]
            df_tmp = _pd.DataFrame( {"rgn" : _np.repeat(df_rgn_seg_res_bb['rgn'].values, lens),
                                     "seg" : _np.repeat(df_rgn_seg_res_bb['seg'].values, lens),
                                     "res" : _np.concatenate(df_rgn_seg_res_bb['res'].values)})
            df_tmp = df_tmp[["rgn", "seg", "res"]]
            if "bb" in df_rgn_seg_res_bb.columns:
                df_tmp["bb"] = _np.repeat(df_rgn_seg_res_bb['bb'].values, lens)
            df_rgn_seg_res_bb = df_tmp
        params["df_rgn_seg_res_bb"] = df_rgn_seg_res_bb
        params["rgn_agg_func"] = rgn_agg_func
        if df_hist_feats is not None:
            if isinstance(df_hist_feats, _pd.DataFrame):
                if (len(df_hist_feats.dbin.unique()) > 1):
                    print("WARNING: dbin values in df_hist_feats not unique! Taking the dbin value of the first row ..." % error_type)
                    df_hist_feats.dbin = df_hist_feats.dbin[0]
        params["df_hist_feats"] = df_hist_feats
        self.params = params
        
    def get_feature_func(self):
        def f(args):
            return self._feature_func_engine(args, self.params)
        f.__name__ = self.feature_name
        return f

    @staticmethod
    def _feature_func_engine(args, params):
        """ this function computes particular features from the simulated ensembles """
        # dummy
        return args, params
