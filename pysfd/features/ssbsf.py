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
        Single Backbone Sidechain Features (ssbsf)
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
# for circular statistics in srf.Dihedral
import scipy.stats as _scipy_stats

from pysfd.features import _feature_agent


class _SBSF(_feature_agent.FeatureAgent):
    """
    Intermediary class between a particular feature class in this module and
    _feature_agent.FeatureAgent
    in order to bundle common tasks
    """
    def __init__(self, feature_name, error_type, subsel, df_rgn_seg_res_bb, rgn_agg_func, label, df_hist_feats = None, max_mom_ord = 1, **params):
        params["_finish_traj_df"] = self._finish_traj_df
        params["subsel"] = subsel
        s_coarse = ""
        if df_rgn_seg_res_bb is not None:
            s_coarse = "coarse."
        super(_SBSF, self).__init__(feature_name      = feature_name + s_coarse + error_type + label,
                                   error_type        = error_type,
                                   max_mom_ord       = max_mom_ord,
                                   df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                   rgn_agg_func      = rgn_agg_func,
                                   df_hist_feats     = df_hist_feats,
                                   **params)

    @staticmethod
    def _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r, params, df_hist_feats = None, circular_stats = None):
        """
        helper function of _feature_func_engine:
        finishes processing of traj_df in each of
        the _feature_func_engine() in the ssbsf module

        Parameters
        ----------
        * fself          : self pointer to foreign master PySFD object

        * l_lbl          : list of str, feature label types

        * traj_df        : pandas.DataFrame containing feature labels 

        * a_feat         : numpy.array containing feature information
                           numpy.shape(a_feat) = (number_of_frames, number_of_feature_labels)
                           !!! Note: !!!
                           if None (as for class CA_RMSF), feature values are already averaged over frames and are found in traj_df
                           only in this case, when corase-graining, we are averaging first over frames,
                           and then over the coarse-grained "rgn" entries

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

        * r             : int, replica index

        * params        : extra parameters from initial feature class initiation, possibly
                          containing, e.g., "is_correlation"
                          * is_correlation : bool, optional, whether or not to output feature values
                                             for a subsequent correlation analysis (e.g. pff.Feature_Correlation())

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

        * circular_stats : str, default=None
            whether to use circular statistics in combination with
            fself.error_type[fself._feature_func_name]
                | None  : use regular, arithmetic (ignores periodic boundary conditions)
                | "csd" : use circular mean and standard deviations

        Returns
        -------
        * traj_df        : pandas.DataFrame, contains all the feature values accumulated for this replica

        * dataflags      : dict, contains flags with more information about the data in traj_df
        """

        dataflags = { "error_type"     : fself.error_type[fself._feature_func_name],
                      "circular_stats" : circular_stats }
        if fself._feature_func_name in fself.max_mom_ord:
            dataflags["max_mom_ord"] = fself.max_mom_ord[fself._feature_func_name]

        def mycircmean(x):
            return _scipy_stats.circmean(x, low = -_np.pi, high = _np.pi)
        def mycircstd(x):
            return _scipy_stats.circstd(x, low = -_np.pi, high = _np.pi)

        def myhist(a_data, dbin):
            prec = len(str(dbin).partition(".")[2])+1
            a_bins =_np.arange(_np.floor(a_data.min() / dbin),
                               _np.ceil(a_data.max() / dbin) + 1, 1) * dbin
            if len(a_bins) == 1:
                a_bins = _np.array([a_bins[0], a_bins[0] + dbin])
            a_hist = _np.histogram(a_data, bins = a_bins, density = True)
            return tuple(list(a_hist))

        # if a_feat == None (as for class CA_RMSF), feature values are already averaged over frames and are found in traj_df
        if (df_rgn_seg_res_bb is None) and (a_feat is not None):
            if "is_correlation" in params:
                if params["is_correlation"] == True:
                    traj_df["feature"] = traj_df["seg"].astype(str) + "_" + traj_df["res"].astype(str) + "_" + traj_df["rnm"].astype(str) + "_" + traj_df["bb"].astype(str)
                    traj_df.drop(columns = l_lbl, inplace = True)
                    traj_df.set_index("feature", inplace = True)
                    traj_df = _pd.DataFrame(a_feat.transpose(), index = traj_df.index)
                    return traj_df, None 

            # if include ALL feature entries for histogramming:
            if isinstance(df_hist_feats, (int, float)):
                dbin = df_hist_feats
                traj_df['fhist'] = True
                # label used below to include "fhist" entry:
                l_flbl = ['fhist']
            # elif include NO feature entries for histogramming:
            elif df_hist_feats is None:
                dbin = None
                traj_df['fhist'] = False
                # label used below to NOT include "fhist" entry:
                l_flbl = []
            # else (if include SOME feature entries for histogramming):
            elif isinstance(df_hist_feats, _pd.DataFrame):
                dbin = df_hist_feats["dbin"][0]
                df_hist_feats['fhist'] = True
                # label used below to include "fhist" entry:
                l_flbl = ['fhist']
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
                mytmp["fhist"] = list(_np.apply_along_axis(lambda x: myhist(x, dbin), axis = 0, arr = a_feat[:, traj_df.fhist == True]).transpose())
                traj_df.loc[traj_df.fhist == True, "fhist"] = mytmp["fhist"]
            # correction factor to convert numpy.std into pandas.std
            if _np.shape(a_feat)[0] > 1:
                std_factor = _np.sqrt(_np.shape(a_feat)[0] / (_np.shape(a_feat)[0] - 1.))
            else:
                std_factor = 0
            if circular_stats is None:
                traj_df['f']  = _np.mean(a_feat, axis = 0)
                l_flbl += ['f']
                for mymom in range(2, fself.max_mom_ord[fself._feature_func_name]+1):
                    traj_df['f.%d' % mymom] = _scipy_stats.moment(a_feat, axis = 0, moment = mymom)
                    l_flbl += ['f.%d' % mymom]
                traj_df = traj_df[l_lbl + l_flbl].copy()
                if fself.error_type[fself._feature_func_name] == "std_dev":
                    traj_df['sf'] = _np.std(a_feat, axis=0) * std_factor
            elif circular_stats == "csd":
                traj_df['f']  = _scipy_stats.circmean(a_feat, low = -_np.pi, high = _np.pi, axis = 0)
                l_flbl += ['f']
                traj_df = traj_df[l_lbl + l_flbl].copy()
                if fself.error_type[fself._feature_func_name] == "std_dev":
                    traj_df['sf'] = _scipy_stats.circstd(a_feat, low = -_np.pi, high = _np.pi, axis=0)
            #traj_df.set_index(l_lbl)
        # coarse-graining:
        elif df_rgn_seg_res_bb is not None:
            #if (fself.error_type[fself._feature_func_name] == "std_err") and (circular_stats is None):
            #    #
            #    # averaging over "rgn" should be done before over frames, but
            #    # I suspect it may be more efficient to do the reverse (over frame before over "rgn")
            #    # but doesn't matter for _np.mean() here
            #    #
            #    if a_feat is not None:
            #        traj_df['f'] = _np.mean(a_feat, axis = 0)
            #    traj_df_seg_res = traj_df[["seg", "res"]].copy().drop_duplicates()
            #    df_merge = traj_df_seg_res.merge(df_rgn_seg_res_bb, how = "outer", copy = False)
            #    df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
            #    if len(df_merge) > 0:
            #        warnstr = "not-defined resIDs in df_rgn_seg_res_bb (your definition for coarse-graining):\n%s" % \
            #                  df_merge
            #        _warnings.warn(warnstr)
            #    traj_df = traj_df.merge(df_rgn_seg_res_bb, copy = False)
            #    #rgn="a1L2"
            #    #print(traj_df.query("rgn == '%s'" % rgn))
            #    traj_df = traj_df.groupby(["rgn"]).agg({ 'f' : aggtype })
            #elif (fself.error_type[fself._feature_func_name] == "std_err") and (circular_stats == "csd"):
            if (fself.error_type[fself._feature_func_name] == "std_err"):
                traj_df_seg_res_bb = traj_df[["seg", "res", "bb"]].copy().drop_duplicates()
                df_merge = traj_df_seg_res_bb.merge(df_rgn_seg_res_bb, how = "outer", copy = False, indicator = True)
                df_merge = df_merge.query("_merge == 'right_only'")
                if len(df_merge) > 0:
                    warnstr = "df_rgn_seg_res_bb, your coarse-graining definition, has resID entries that are not in your feature list:\n%s" % df_merge
                    _warnings.warn(warnstr)
                # if a_feat == None (as for class CA_RMSF), feature values are already averaged over frames and are found in traj_df
                if a_feat is None:
                    traj_df = traj_df.merge(df_rgn_seg_res_bb, copy = False)
                    traj_df = traj_df.groupby(["rgn"]).agg( { "f" : rgn_agg_func } )
                    traj_df_hist = None
                else:
                    traj_df.reset_index(drop = True, inplace = True)
                    traj_df = _pd.concat([traj_df, _pd.DataFrame(_np.transpose(a_feat))], axis = 1, copy = False)
                    traj_df = traj_df.merge(df_rgn_seg_res_bb, copy = False)
                    traj_df.set_index(["rgn"] + l_lbl, inplace = True)
                    traj_df = traj_df.stack()
                    traj_df = traj_df.to_frame().reset_index()
                    #rgn="a1L2"
                    #print(traj_df.query("rgn == '%s'" % rgn))
                    traj_df.columns = ["rgn"] + l_lbl + ["frame", "f"]
                    traj_df.set_index(["rgn", "frame"] + l_lbl, inplace = True)
                    # computes the rgn_agg_func, e.g., mean over a region "rgn" in each frame:
                    traj_df = traj_df.groupby(["rgn", "frame"]).agg( { "f" : rgn_agg_func } )
                    if "is_correlation" in params:
                        if params["is_correlation"] == True:
                            traj_df = traj_df.unstack()
                            traj_df.index.name = "feature"
                            traj_df.columns = traj_df.columns.get_level_values(1)
                            traj_df.columns.name = None
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
                            non-matching entries:\n%s" % (",".join(["rgn"]),
                                                          traj_df_hist.loc[_np.any(traj_df_hist.isnull(), axis = 1), :]))
                    if traj_df_hist is not None:
                        traj_df_hist = traj_df_hist.groupby(["rgn"]).agg( { "f" : lambda x: myhist(x, dbin ) } )
                        traj_df_hist.rename(columns = { "f" : "fhist" }, inplace = True)

                    # mean/std over the frames...: 
                    if circular_stats is None:
                        l_func = ['mean'] + [ lambda x: _scipy_stats.moment(x, moment = mymom) for mymom in range(2, fself.max_mom_ord[fself._feature_func_name]+1)]
                        l_lbl = ['f'] + [ 'f.%d' % mymom for mymom in range(2, fself.max_mom_ord[fself._feature_func_name]+1)]
                        traj_df = traj_df.groupby(["rgn"]).agg( l_func )
                        traj_df.columns = l_lbl
                    elif circular_stats == "csd":
                        traj_df = traj_df.groupby(["rgn"]).agg( { "f" : mycircmean } )
                        traj_df.columns = ["f"]
            elif fself.error_type[fself._feature_func_name] == "std_dev":
                traj_df_seg_res = traj_df[["seg", "res", "bb"]].copy().drop_duplicates()
                df_merge = traj_df_seg_res.merge(df_rgn_seg_res_bb, how = "outer", copy = False, indicator = True)
                df_merge = df_merge.query("_merge == 'right_only'")
                if len(df_merge) > 0:
                    warnstr = "df_rgn_seg_res_bb, your coarse-graining definition, has resID entries that are not in your feature list:\n%s" % df_merge
                    _warnings.warn(warnstr)
                traj_df.reset_index(drop = True, inplace = True)
                traj_df = _pd.concat([traj_df, _pd.DataFrame(_np.transpose(a_feat))], axis = 1, copy = False)
                traj_df = traj_df.merge(df_rgn_seg_res_bb, copy = False)
                traj_df.set_index(["rgn"] + l_lbl, inplace = True)
                traj_df = traj_df.stack()
                traj_df = traj_df.to_frame().reset_index()
                #rgn="a1L2"
                #print(traj_df.query("rgn == '%s'" % rgn))
                traj_df.columns = ["rgn"] + l_lbl + ["frame", "f"]
                traj_df.set_index(["rgn", "frame"] + l_lbl, inplace = True)
                # computes the mean over a region in each frame:
                traj_df = traj_df.groupby(["rgn", "frame"]).agg( { "f" : rgn_agg_func } )

                if "is_correlation" in params:
                    if params["is_correlation"] == True:
                        traj_df = traj_df.unstack()
                        traj_df.index.name = "feature"
                        traj_df.columns = traj_df.columns.get_level_values(1)
                        traj_df.columns.name = None
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
                        non-matching entries:\n%s" % (",".join(["rgn"]),
                                                      traj_df_hist.loc[_np.any(traj_df_hist.isnull(), axis = 1), :]))
                if traj_df_hist is not None:
                    traj_df_hist = traj_df_hist.groupby(["rgn"]).agg( { "f" : lambda x: myhist(x, dbin ) } )
                    traj_df_hist.rename(columns = { "f" : "fhist" }, inplace = True)

                # mean/std over the frames...: 
                if circular_stats is None:
                    traj_df = traj_df.groupby(["rgn"]).agg( ["mean", "std"] )
                elif circular_stats == "csd":
                    traj_df = traj_df.groupby(["rgn"]).agg( [mycircmean, mycircstd] )
                traj_df.columns = traj_df.columns.droplevel(level=0)
                traj_df.columns = ["f", "sf"]

            if traj_df_hist is not None:
                traj_df = traj_df.merge(traj_df_hist, left_index = True, right_index = True, how = "outer")
            traj_df.reset_index(inplace=True)
        traj_df['r'] = r
        return traj_df, dataflags


class SASA_sr(_SBSF):
    """
    Computes backbone/sidechain SASA values via mdtraj.shrake_rupley() 
    for a particular simulation with replica index r

    If coarse-graining (via df_rgn_seg_res_bb, see below) into regions,
    by default aggregate via rgn_agg_func = "sum"

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

    * subsel : str, optional, default = "all"
               sub-selection of residues for which to compute features
               subsel is an atom selection string as used in MDTraj
               distances between all possible combinations of atoms defined in subsel
               example: "name CA and within 15 of chain A and resid 82"

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

    * label        : string, user-specific label for feature_name
    """

    def __init__(self, error_type = "std_err", max_mom_ord = 1, subsel = "all", df_rgn_seg_res_bb = None, rgn_agg_func = "sum", df_hist_feats = None, label = ""):
        super(SASA_sr, self).__init__(feature_name      = "ssbsf.SASA_sr.",
                                      error_type        = error_type,
                                      max_mom_ord       = max_mom_ord,
                                      subsel            = subsel,
                                      df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                      rgn_agg_func      = rgn_agg_func,
                                      df_hist_feats     = df_hist_feats,
                                      label             = label)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes backbone/sidechain SASA values via mdtraj.shrake_rupley() 
        for a particular simulation with replica index r

        Parameters
        ----------
        * args   : tuple (fself, myens, r):
            * fself        : self pointer to foreign master PySFD object

            * myens        : string
                             Name of simulated ensemble

            * r            : int, replica index
        * params : dict, extra parameters as keyword arguments
            * error_type   : str
                compute feature errors as ...
                | "std_err" : ... standard errors
                | "std_dev" : ... mean standard deviations

            * subsel : str, optional, default = "all"
                       sub-selection of residues for which to compute features
                       subsel is an atom selection string as used in MDTraj
                       distances between all possible combinations of atoms defined in subsel
                       example: "name CA and within 15 of chain A and resid 82"

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
        fself, myens, r = args
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        subsel                                      = params["subsel"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        _finish_traj_df                             = params["_finish_traj_df"]

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        mytraj = _md.load('%s.%s' % (instem, fself.intrajformat), top='%s.pdb' % instem)
        traj_df_seg_res_bb = mytraj.topology.to_dataframe()[0].loc[:, [ "segmentID", "resSeq", "name" ]].drop_duplicates()
        traj_df_seg_res_bb.columns = [ "seg", "res", "bb" ]
        df_pdb = fself._get_raw_topology_ids( '%s.pdb' % instem, "residue" )
        df_merge = traj_df_seg_res_bb.merge(df_pdb, how = "outer", copy = False)
        df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
        if len(df_merge) > 0:
            warnstr = "residue name mismatch!:\n%s" % df_merge
            _warnings.warn(warnstr)

        if isinstance(subsel, str):
            mytraj.atom_slice(mytraj.topology.select(subsel), inplace = True)
        else:
            raise ValueError("subsel has to be of instance str!")
        traj_df = mytraj.topology.to_dataframe()[0].loc[:, [ "segmentID", "resSeq", "name" ]].drop_duplicates()
        traj_df.columns = [ "seg", "res", "bb" ]
        traj_df = traj_df.merge(df_pdb)
        #a_rnm  = fself._get_raw_topology_ids('%s.pdb' % instem, "residue").rnm.values
        #a_residue = list(mytraj.topology.residues)
        #a_seg = [a.segment_id for a in a_residue]
        #a_res = [a.resSeq for a in a_residue]
        ##a_rnm = [fself.rnm2pdbrnm(a.name) for a in a_residue]
        ##a_rnm = [a.name for a in a_residue]
        l_lbl = [ 'seg', 'res', 'rnm', 'bb' ]
        #traj_df = _pd.DataFrame(data={'seg': a_seg, 'rnm': a_rnm, 'res': a_res })
        traj_df["bb"] = _np.in1d(traj_df["bb"], [ "N", "CA", "C", "O", "H", "HA" ]).astype(int)
        a_feat = _md.shrake_rupley(mytraj, mode="atom")
        traj_df = _pd.concat([traj_df[l_lbl],
                              _pd.DataFrame(_np.transpose(a_feat))], axis = 1, copy = False)
        traj_df = traj_df.groupby(l_lbl).agg("sum")
        a_feat = traj_df.values.transpose()
        traj_df = traj_df.reset_index(drop = False)[l_lbl]
        return _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r, params, df_hist_feats)
