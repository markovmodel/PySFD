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
        Single Residue Feature (srf)
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


class _SRF(_feature_agent.FeatureAgent):
    """
    Intermediary class between a particular feature class in this module and
    _feature_agent.FeatureAgent
    in order to bundle common tasks
    """
    def __init__(self, feature_name, error_type, df_rgn_seg_res_bb, rgn_agg_func, label, df_hist_feats = None, max_mom_ord = 1, **params):
        params["_finish_traj_df"] = self._finish_traj_df
        s_coarse = ""
        if df_rgn_seg_res_bb is not None:
            s_coarse = "coarse."
            if "bb" in df_rgn_seg_res_bb.columns:
                df_rgn_seg_res_bb.drop(columns = "bb", inplace = True)
        super(_SRF, self).__init__(feature_name      = feature_name + s_coarse + error_type + label,
                                   error_type        = error_type,
                                   max_mom_ord       = max_mom_ord,
                                   df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                   rgn_agg_func      = rgn_agg_func,
                                   df_hist_feats     = df_hist_feats,
                                   **params)

    @staticmethod
    def _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r, df_hist_feats = None, circular_stats = None):
        """
        helper function of _feature_func_engine:
        finishes processing of traj_df in each of
        the _feature_func_engine() in the srf module

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

        * r              : int, replica index

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
            std_factor = _np.sqrt(_np.shape(a_feat)[0] / (_np.shape(a_feat)[0] - 1.)) 
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
                traj_df_seg_res = traj_df[["seg", "res"]].copy().drop_duplicates()
                df_merge = traj_df_seg_res.merge(df_rgn_seg_res_bb, how = "outer", copy = False)
                df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
                if len(df_merge) > 0:
                    warnstr = "not-defined resIDs in df_rgn_seg_res_bb (your definition for coarse-graining):\n%s" % \
                            df_merge
                    _warnings.warn(warnstr)
                traj_df = traj_df.merge(df_rgn_seg_res_bb, copy = False)
                # if a_feat == None (as for class CA_RMSF), feature values are already averaged over frames and are found in traj_df
                if a_feat is None:
                    traj_df = traj_df.groupby(["rgn"]).agg( { "f" : rgn_agg_func } )
                    traj_df_hist = None
                else:
                    traj_df = _pd.concat([traj_df, _pd.DataFrame(_np.transpose(a_feat))], axis = 1, copy = False)
                    traj_df.set_index(["rgn"] + l_lbl, inplace = True)
                    traj_df = traj_df.stack()
                    traj_df = traj_df.to_frame().reset_index()
                    #rgn="a1L2"
                    #print(traj_df.query("rgn == '%s'" % rgn))
                    traj_df.columns = ["rgn"] + l_lbl + ["frame", "f"]
                    traj_df.set_index(["rgn", "frame"] + l_lbl, inplace = True)
                    # computes the rgn_agg_func, e.g., mean over a region "rgn" in each frame:
                    traj_df = traj_df.groupby(["rgn", "frame"]).agg( { "f" : rgn_agg_func } )
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
                traj_df_seg_res = traj_df[["seg", "res"]].copy().drop_duplicates()
                df_merge = traj_df_seg_res.merge(df_rgn_seg_res_bb, how = "outer", copy = False)
                df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
                if len(df_merge) > 0:
                    warnstr = "not-defined resIDs in df_rgn_seg_res_bb (your definition for coarse-graining):\n%s" % \
                            df_merge
                    _warnings.warn(warnstr)
                traj_df = traj_df.merge(df_rgn_seg_res_bb, copy = False)
                traj_df = _pd.concat([traj_df, _pd.DataFrame(_np.transpose(a_feat))], axis = 1, copy = False)
                traj_df.set_index(["rgn"] + l_lbl, inplace = True)
                traj_df = traj_df.stack()
                traj_df = traj_df.to_frame().reset_index()
                #rgn="a1L2"
                #print(traj_df.query("rgn == '%s'" % rgn))
                traj_df.columns = ["rgn"] + l_lbl + ["frame", "f"]
                traj_df.set_index(["rgn", "frame"] + l_lbl, inplace = True)
                # computes the mean over a region in each frame:
                traj_df = traj_df.groupby(["rgn", "frame"]).agg( { "f" : rgn_agg_func } )

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


class ChemicalShift(_SRF):
    """
    Computes residual chemical shifts using mdtraj.chemical_shifts() 
    for a particular simulation with replica index r

    If coarse-graining (via df_rgn_seg_res_bb, see below) into regions,
    by default aggregate via rgn_agg_func = "mean"

    CURRENT LIMITATIONS:
    - apparently, "shiftx2" can only compute
      chemical shifts of single-chain proteins and
    - only in single-process mode

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

    * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                          regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
      df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                      'seg' : ["A", "A", "B", "B", "C"],
                                      'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                          if None, no coarse-graining is performed

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

   * label        : string, user-specific label for feature_name
    """

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = "mean", df_hist_feats = None, label = ""):
        super(ChemicalShift, self).__init__(feature_name      = "srf.chemical_shift.",
                                            error_type        = error_type,
                                            max_mom_ord       = max_mom_ord,
                                            df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                            rgn_agg_func      = rgn_agg_func,
                                            df_hist_feats     = df_hist_feats,
                                            label             = label)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes residual chemical shifts using mdtraj.chemical_shifts() 
        for a particular simulation with replica index r
    
        (CURRENT LIMITATIONS:
        apparently, "shiftx2" can only compute
        chemical shifts of single-chain proteins and
        only very slowly)

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
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        _finish_traj_df                             = params["_finish_traj_df"]

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        mytraj = _md.load('%s.%s' % (instem, fself.intrajformat), top='%s.pdb' % instem)
        df_pdb = fself._get_raw_topology_ids('%s.pdb' % instem, "residue")
        df_pdb = df_pdb.query("seg == '%s'" % mytraj.topology.chain(0).atom(0).segment_id)
        df_pdb.rename({"rnm" : "rnm_old"}, inplace = True)
        traj_df = _md.compute_chemical_shifts(mytraj).reset_index()
        a_feat = traj_df.copy().drop(["resSeq", "name"], axis = 1).as_matrix().transpose()
        traj_df = traj_df[["resSeq", "name"]]
        traj_df.columns = ['res', "anm"]
        traj_df["seg"] = mytraj.topology.chain(0).atom(0).segment_id

        traj_df_seg_res = traj_df[["seg", "res"]].drop_duplicates()
        df_merge = traj_df_seg_res.merge(df_pdb, how = "outer", copy = False)
        df_merge = df_merge.loc[df_merge.isnull().values.sum(axis=1) > 0].drop_duplicates()
        if len(df_merge) > 0:
            warnstr = "residue name mismatch!:\n%s" % df_merge
            _warnings.warn(warnstr)
        traj_df = traj_df.merge(df_pdb, copy = False)
        traj_df.drop(["rnm"], axis=1, inplace=True)
        traj_df.rename(columns={'rnm_old': 'rnm'}, inplace=True)
        l_lbl = ["seg", "res", "rnm", "anm"]
        traj_df = traj_df[l_lbl]
        return _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r, df_hist_feats)


class CA_RMSF(_SRF):
    """
    Computes CA root mean square fluctations via VMD
    (see features/scripts/compute_carmsf.vmd.tcl for details)

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
                                individual frames, but over RMSF values
                          !!! 

    * rgn_agg_func  : function or str for coarse-graining, default = "mean"
                      function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                      this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                      - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                      - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                        such as "mean", "std"

    * label        : string, user-specific label for feature_name
    """

    def __init__(self, error_type = "std_err", df_rgn_seg_res_bb = None, rgn_agg_func = "mean", label = ""):
        if error_type == "std_dev":
            print("WARNING: error_type \"std_dev\" not defined in CA_RMSF ! Falling back to \"std_err\" instead ...")
            error_type = "std_err"
        super(CA_RMSF, self).__init__(
                                        feature_name      = "srf.CA_RMSF.",
                                        error_type        = "std_err",
                                        df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                        rgn_agg_func      = rgn_agg_func,
                                        label             = label)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes CA root mean square fluctations via VMD
        (see features/scripts/compute_carmsf.vmd.tcl for details)
    
        Parameters
        ----------
        * args   : tuple (fself, myens, r):

            * fself      : self pointer to foreign master PySFD object

            * myens        : string
                             Name of simulated ensemble

            * r            : int, replica index

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
    
            * rgn_agg_func  : function or str for coarse-graining
                              function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                              this function uses the coarse-graining mapping defined in df_rgn_seg_res_bb
                              - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                              - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                                such as "mean", "std"
        """

        l_lbl = ["seg", "res", "rnm"]
    
        fself, myens, r = args
        fself.error_type[fself._feature_func_name] = params["error_type"]
        df_rgn_seg_res_bb                          = params["df_rgn_seg_res_bb"]
        rgn_agg_func                               = params["rgn_agg_func"]
        _finish_traj_df                            = params["_finish_traj_df"]

        indir  = "input/%s/r_%05d" % (myens, r)
        instem = "%s.r_%05d.prot" % (myens, r)
        outdir = "output/%s/r_%05d/%s/%s" % (myens, r, fself.feature_func_name, fself.intrajdatatype)
        _subprocess.Popen(_shlex.split("rm -rf %s" % outdir)).wait()
        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()
        mycmd = "vmd -dispdev text -e %s/features/scripts/compute_carmsf.vmd.tcl -args %s %s %s %s" \
                % (fself.pkg_dir, indir, instem, fself.intrajformat, outdir)
        outfile = open("%s/log.compute_carmsf.vmd.tcl.log" % outdir, "w")
        myproc = _subprocess.Popen(_shlex.split(mycmd), stdout=outfile, stderr=outfile)
        myproc.wait()
        outfile.close()
        traj_df = _pd.read_csv("%s/%s.caRMSF.vmd.dat" % (outdir, instem), sep=' ', names=l_lbl + ["f"])
        a_feat = None
        return _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r)


class SASA_sr(_SRF):
    """
    Computes residual SASA values via mdtraj.shrake_rupley() 
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

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = "sum", df_hist_feats = None, label = ""):
        super(SASA_sr, self).__init__(feature_name      = "srf.SASA_sr.",
                                      error_type        = error_type,
                                      max_mom_ord       = max_mom_ord,
                                      df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                      rgn_agg_func      = rgn_agg_func,
                                      df_hist_feats     = df_hist_feats,
                                      label             = label)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes residual SASA values via mdtraj.shrake_rupley() 
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
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        _finish_traj_df                             = params["_finish_traj_df"]

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        mytraj = _md.load('%s.%s' % (instem, fself.intrajformat), top='%s.pdb' % instem)
        a_rnm  = fself._get_raw_topology_ids('%s.pdb' % instem, "residue").rnm.values

        a_residue = list(mytraj.topology.residues)
        a_seg = [a.segment_id for a in a_residue]
        a_res = [a.resSeq for a in a_residue]
        #a_rnm = [fself.rnm2pdbrnm(a.name) for a in a_residue]
        #a_rnm = [a.name for a in a_residue]
        l_lbl = ['seg', 'res', 'rnm']
        traj_df = _pd.DataFrame(data={'seg': a_seg, 'rnm': a_rnm, 'res': a_res })
        traj_df = traj_df[l_lbl]
        a_feat = _md.shrake_rupley(mytraj, mode="residue")
        return _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r, df_hist_feats)

class RSASA_sr(_SRF):
    """
    Computes residual relative SASA values via mdtraj.shrake_rupley() 
    for a particular simulation with replica index r
    relative SASAs are SASAs normalized by their residue-specific maximum SASA.

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

    * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                          regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
      df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                      'seg' : ["A", "A", "B", "B", "C"],
                                      'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                          if None, no coarse-graining is performed

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

    * label        : string, user-specific label for feature_name
    """
 
    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = "mean", df_hist_feats = None, label = ""):
      
        super(RSASA_sr, self).__init__(feature_name      ="srf.RSASA_sr.",
                                       error_type        = error_type,
                                       max_mom_ord       = max_mom_ord,
                                       df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                       rgn_agg_func      = rgn_agg_func,
                                       df_hist_feats     = df_hist_feats,
                                       label             = label)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes residual relative SASA values via mdtraj.shrake_rupley() 
        for a particular simulation with replica index r
        relative SASAs are SASAs normalized by their residue-specific maximum SASA.
 
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

        # maximum accessible surface area values (in units of A^2, divided by 100 => nm^2) 
        # Tien, M. Z.; Meyer, A. G.; Sydykova, D. K.; Spielman, S. J.; Wilke, C. O. (2013).
        # "Maximum allowed solvent accessibilites of residues in proteins".
        # PLoS ONE. 8 (11): e80635. PMID 24278298. doi:10.1371/journal.pone.0080635.
        #
        # Miller, S.; Janin, J.; Lesk, A. M.; Chothia, C. (1987).
        # "Interior and surface of monomeric proteins". J. Mol. Biol. 196: 641-656.
        # Rose, G. D.; Geselowitz, A. R.; Lesser, G. J.; Lee, R. H.; Zehfus, M. H. (1985).
        # "Hydrophobicity of amino acid residues in globular proteins". Science. 229: 834-838.
        _df_max_rsasa = _pd.DataFrame(
            data={
                "tien_th"  : _np.array([129, 274, 195, 193, 167, 223, 225, 104, 224, 197,
                                        201, 236, 224, 240, 159, 155, 172, 285, 263, 174]) / 100.0,
                "tien_emp" : _np.array([121, 265, 187, 187, 148, 214, 214, 97, 216, 195,
                                        191, 230, 203, 228, 154, 143, 163, 264, 255, 165]) / 100.0,
                "miller"   : _np.array([113, 241, 158, 151, 140, 183, 189, 85, 194, 182,
                                        180, 211, 204, 218, 143, 122, 146, 259, 229, 160]) / 100.0,
                "rose"     : _np.array([118.1, 256, 165.5, 158.7, 146.1, 186.2, 193.2, 88.1,
                                        202.5, 181, 193.1, 225.8, 203.4, 222.8, 146.8,
                                        129.8, 152.5, 266.3, 236.8, 164.5]) / 100.0 },

            index = _np.array(["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS",
                               "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]))
 
        fself, myens, r = args
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        _finish_traj_df                             = params["_finish_traj_df"]

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        mytraj = _md.load('%s.%s' % (instem, fself.intrajformat), top='%s.pdb' % instem)
        a_rnm  = fself._get_raw_topology_ids('%s.pdb' % instem, "residue").rnm.values

        a_residue = list(mytraj.topology.residues)
        a_seg = [a.segment_id for a in a_residue]
        a_res = [a.resSeq for a in a_residue]
        #a_rnm = [fself.rnm2pdbrnm(a.name) for a in a_residue]
        #a_rnm = [a.name for a in a_residue]
        _df_max_rsasa = _df_max_rsasa.reindex(_np.unique(_np.concatenate((_df_max_rsasa.index.values, a_rnm))))
        a_maxrsasa = _df_max_rsasa.loc[[fself.rnm2pdbrnm(resname) for resname in a_rnm], "tien_emp"]
        l_lbl = ['seg', 'res', 'rnm']
        traj_df = _pd.DataFrame(data={ 'seg': a_seg, 'rnm': a_rnm, 'res': a_res })
        traj_df = traj_df[l_lbl]
        a_feat = _np.apply_along_axis(lambda x: x / a_maxrsasa,
                                      1,
                                      _md.shrake_rupley(mytraj, mode="residue"))
        return _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r, df_hist_feats)


class Dihedral(_SRF):
    """
    Computes a type of dihedral as defined by feat_subfunc

    If coarse-graining (via df_rgn_seg_res_bb, see below) into regions,
    by default aggregate via rgn_agg_func = "mean" or mycircmean(),
    depending on circular_stats (see below). 

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

    * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                          regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
      df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                      'seg' : ["A", "A", "B", "B", "C"],
                                      'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                          if None, no coarse-graining is performed

    * rgn_agg_func  : function or str for coarse-graining, 
                      default is rgn_agg_func = "mean" or mycircmean(), depending on circular_stats (see below)
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

    * circular_stats : str, default="csd"
        whether to use circular statistics in combination with error_type
            | None  : use regular statistics on the dihedral (ignores periodic boundary conditions)
            | "csd" : use circular mean and standard deviations

    * feat_subfunc : function returning (indices, values), such as
         with options "periodic", "opt", see mdtraj documentation:
             - mdtraj.compute_chi1
             - mdtraj.compute_chi2
             - mdtraj.compute_chi3
             - mdtraj.compute_chi4
             - mdtraj.compute_omega
             - mdtraj.compute_phi
             - mdtraj.compute_psi
         if you would like to use option values that are differ from the default,
         just create a new function:
         def mychi1(traj):
             return mdtraj.compute_chi1(traj, periodic=False, opt=False)

         PySFD.features.Dihedral(error_type="std_dev", feat_subfunc=mychi1, label="")]

    * label        : string, user-specific label for feature_name
    """

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = None, df_hist_feats = None, label = "", circular_stats = "csd", feat_subfunc = None):
        if circular_stats not in [None, "csd"]:
            raise ValueError("circular_stats not in [None, \"csd\"]")

        if feat_subfunc is None:
            raise ValueError("feat_subfunc not defined")
        super(Dihedral, self).__init__(
                                        feature_name      = "srf." + feat_subfunc.__name__.split("compute_")[-1] + ".",
                                        error_type        = error_type,
                                        max_mom_ord       = max_mom_ord,
                                        df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                        rgn_agg_func      = rgn_agg_func,
                                        df_hist_feats     = df_hist_feats,
                                        label             = label,
                                        circular_stats    = circular_stats,
                                        feat_subfunc      = feat_subfunc)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes a type of dihedral as defined by feat_subfunc
    
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
 
            * circular_stats : str
                whether to use circular statistics in combination with error_type
                    | None  : use regular statistics on the dihedral (ignores periodic boundary conditions)
                    | "csd" : use circular mean and standard deviations

            * feat_subfunc : function returning (indices, values), such as
                 with options "periodic", "opt", see mdtraj documentation:
                     - mdtraj.compute_chi1
                     - mdtraj.compute_chi2
                     - mdtraj.compute_chi3
                     - mdtraj.compute_chi4
                     - mdtraj.compute_omega
                     - mdtraj.compute_phi
                     - mdtraj.compute_psi
                 if you would like to use option values that are differ from the default,
                 just create a new function:
                 def mychi1(traj):
                     return mdtraj.compute_chi1(traj, periodic=False, opt=False)
        
                 PySFD.features.Dihedral(error_type="std_dev", feat_subfunc=mychi1, label="")]
        """

        fself, myens, r                             = args
        circular_stats                              = params["circular_stats"]
        feat_subfunc                                = params["feat_subfunc"]
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        _finish_traj_df                             = params["_finish_traj_df"]

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        mytraj = _md.load('%s.%s' % (instem, fself.intrajformat), top='%s.pdb' % instem)
        a_rnm  = fself._get_raw_topology_ids('%s.pdb' % instem, "atom").rnm.values
        a_atom = list(mytraj.topology.atoms)
        a_seg  = _np.array([a.segment_id for a in a_atom])
        a_res  = _np.array([a.residue.resSeq for a in a_atom])
        #df_rnm_rnm2 = _pd.DataFrame({ 'rnm' : a_rnm, 'rnm2' : _np.array([a.residue.name for a in a_atom])})
        #print(df_rnm_rnm2.query('rnm != rnm2').drop_duplicates())
        a_result  = feat_subfunc(mytraj)
        a_indices = a_result[0][:, 1]

        l_lbl = ['seg', 'res', 'rnm']
        traj_df   = _pd.DataFrame(data={'seg': a_seg[a_indices], 'rnm': a_rnm[a_indices], 'res': a_res[a_indices] })
        traj_df = traj_df[l_lbl]

        a_feat = a_result[1]
        if circular_stats == None:
            rgn_agg_func = "mean"
        elif circular_stats == "csd":
            def mycircmean(x):
                return _scipy_stats.circmean(x, low = -_np.pi, high = _np.pi)
            rgn_agg_func = mycircmean
        traj_df, dataflags = _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r, df_hist_feats = df_hist_feats, circular_stats = circular_stats)

        if df_rgn_seg_res_bb is None:
            a_residue = list(mytraj.topology.residues)
            a_seg = [a.segment_id for a in a_residue]
            a_rnm  = fself._get_raw_topology_ids('%s.pdb' % instem, "residue").rnm.values
            a_res = [a.resSeq for a in a_residue]
            l_lbl = ['seg', 'res', 'rnm']
            traj_df_allres = _pd.DataFrame(data={'seg': a_seg, 'rnm': a_rnm, 'res': a_res })
            traj_df_allres = traj_df_allres[l_lbl]
            traj_df = traj_df_allres.merge(traj_df, how = "outer")
        return traj_df, dataflags

class Scalar_Coupling(_SRF):
    """
    Computes a type of scalar coupling as defined by feat_subfunc

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

    * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                          regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
      df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                      'seg' : ["A", "A", "B", "B", "C"],
                                      'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                          if None, no coarse-graining is performed

    * rgn_agg_func  : function or str for coarse-graininga, default = "mean"
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
 
    * feat_subfunc : function returning (indices, values), such as
         with option "model", see mdtraj documentation:
                 - mdtraj.compute_J3_HN_C
                 - mdtraj.compute_J3_HN_CB
                 - mdtraj.compute_J3_HN_HA
         if you would like to use option values that are differ from the default,
         just create a new function:
         def myscalar(traj):
             return mdtraj.compute_chi1(traj, periodic=False, opt=False)

         PySFD.features.Scalar_Coupling(error_type="std_dev", feat_func=myscalar, label="")]

    * label        : string, user-specific label for feature_name
    """

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = "mean", df_hist_feats = None, label = "", feat_subfunc = None):
        if feat_subfunc is None:
            raise ValueError("feat_subfunc not defined")
        super(Scalar_Coupling, self).__init__(
                                        feature_name      = "srf." + feat_subfunc.__name__.split("compute_")[-1] + ".",
                                        error_type        = error_type,
                                        max_mom_ord       = max_mom_ord,
                                        df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                        rgn_agg_func      = rgn_agg_func,
                                        df_hist_feats     = df_hist_feats,
                                        label             = label,
                                        feat_subfunc      = feat_subfunc)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes a type of scalar coupling as defined by feat_subfunc
    
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

            * feat_subfunc : function returning (indices, values), such as
                with option "model", see mdtraj documentation:
                        - mdtraj.compute_J3_HN_C
                        - mdtraj.compute_J3_HN_CB
                        - mdtraj.compute_J3_HN_HA
                if you would like to use option values that are differ from the default,
                just create a new function:
                def myscalar(traj):
                    return mdtraj.compute_chi1(traj, periodic=False, opt=False)
        
                PySFD.features.Scalar_Coupling(error_type="std_dev", feat_func=myscalar, label="")]
        """

        fself, myens, r                             = args
        feat_subfunc                                = params["feat_subfunc"]
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        _finish_traj_df                             = params["_finish_traj_df"]

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        mytraj = _md.load('%s.%s' % (instem, fself.intrajformat), top='%s.pdb' % instem)
        a_rnm  = fself._get_raw_topology_ids('%s.pdb' % instem, "atom").rnm.values
        a_atom = list(mytraj.topology.atoms)
        a_seg  = _np.array([a.segment_id for a in a_atom])
        a_res  = _np.array([a.residue.resSeq for a in a_atom])
        #df_rnm_rnm2 = _pd.DataFrame({ 'rnm' : a_rnm, 'rnm2' : _np.array([a.residue.name for a in a_atom])})
        #print(df_rnm_rnm2.query('rnm != rnm2').drop_duplicates())
        a_result  = feat_subfunc(mytraj)
        a_indices = a_result[0][:, 1]

        l_lbl = ['seg', 'res', 'rnm']
        traj_df   = _pd.DataFrame(data={'seg': a_seg[a_indices], 'rnm': a_rnm[a_indices], 'res': a_res[a_indices] })
        traj_df = traj_df[l_lbl]
        a_feat = a_result[1]
        return _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r, df_hist_feats)



class IsDSSP_mdtraj(_SRF):
    """
    Computes binary secondary structure assignments (DSSP), e.g.,
    whether or not a residual helix ("H") is formed (depending on DSSPpars, see below)

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

    * df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                          regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
      df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                      'seg' : ["A", "A", "B", "B", "C"],
                                      'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                          if None, no coarse-graining is performed

    * rgn_agg_func  : function or str for coarse-graininga, default = "mean"
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

    * DSSPpars: tuple, default = (whatDSSP, issimplified) = ('H', True)
      - whatDSSP: string, defining what DSSP value to check for.
        Possible values (double-check doc string of mdtraj.compute_dssp):
        The DSSP assignment codes are:
                ‘H’ : Alpha helix
                ‘B’ : Residue in isolated beta-bridge
                ‘E’ : Extended strand, participates in beta ladder
                ‘G’ : 3-helix (3/10 helix)
                ‘I’ : 5 helix (pi helix)
                ‘T’ : hydrogen bonded turn
                ‘S’ : bend
                ‘ ‘ : Loops and irregular elements
        The simplified DSSP codes are:
                ‘H’ : Helix. Either of the ‘H’, ‘G’, or ‘I’ codes.
                ‘E’ : Strand. Either of the ‘E’, or ‘B’ codes.
                ‘C’ : Coil. Either of the ‘T’, ‘S’ or ‘ ‘ codes.
        
        A special ‘NA’ code will be assigned to each ‘residue’ in the topology which isn’t actually a protein residue
        (does not contain atoms with the names ‘CA’, ‘N’, ‘C’, ‘O’),
        such as water molecules that are listed as ‘residue’s in the topology.

      - simplified : bool, default=True
        Use the simplified 3-category assignment scheme. Otherwise the original 8-category scheme is used.

    * label        : string, user-specific label for feature_name
    """

    def __init__(self, error_type = "std_err", max_mom_ord = 1, df_rgn_seg_res_bb = None, rgn_agg_func = "mean", df_hist_feats = None, label = "", DSSPpars = None):
        if DSSPpars is None:
            DSSPpars = ('H', True)
        DSSPlabel = "isDSSPeq" + DSSPpars[0]
        if DSSPpars[1] == True:
            DSSPlabel += "simplified"

        super(IsDSSP_mdtraj, self).__init__(
                                        feature_name      = "srf." + DSSPlabel + ".",
                                        error_type        = error_type,
                                        max_mom_ord       = max_mom_ord,
                                        df_rgn_seg_res_bb = df_rgn_seg_res_bb,
                                        rgn_agg_func      = rgn_agg_func,
                                        df_hist_feats     = df_hist_feats,
                                        label             = label,
                                        DSSPpars          = DSSPpars)

    @staticmethod
    def _feature_func_engine(args, params):
        """
        Computes binary secondary structure assignments (DSSP), e.g., 
        whether or not a residual helix ("H") is formed (depending on DSSPpars)
    
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

            * DSSPpars: tuple, default = (whatDSSP, issimplified) = ('H', True)
              - whatDSSP: string, defining what DSSP value to check for.
                Possible values (double-check doc string of mdtraj.compute_dssp):
                The DSSP assignment codes are:
                        ‘H’ : Alpha helix
                        ‘B’ : Residue in isolated beta-bridge
                        ‘E’ : Extended strand, participates in beta ladder
                        ‘G’ : 3-helix (3/10 helix)
                        ‘I’ : 5 helix (pi helix)
                        ‘T’ : hydrogen bonded turn
                        ‘S’ : bend
                        ‘ ‘ : Loops and irregular elements
                The simplified DSSP codes are:
                        ‘H’ : Helix. Either of the ‘H’, ‘G’, or ‘I’ codes.
                        ‘E’ : Strand. Either of the ‘E’, or ‘B’ codes.
                        ‘C’ : Coil. Either of the ‘T’, ‘S’ or ‘ ‘ codes.
                
                A special ‘NA’ code will be assigned to each ‘residue’ in the topology which isn’t actually a protein residue
                (does not contain atoms with the names ‘CA’, ‘N’, ‘C’, ‘O’),
                such as water molecules that are listed as ‘residue’s in the topology.
        
              - simplified : bool, default=True
                Use the simplified 3-category assignment scheme. Otherwise the original 8-category scheme is used.
        """

        fself, myens, r                             = args
        DSSPpars                                    = params["DSSPpars"]
        fself.error_type[fself._feature_func_name]  = params["error_type"]
        fself.max_mom_ord[fself._feature_func_name] = params["max_mom_ord"]
        df_rgn_seg_res_bb                           = params["df_rgn_seg_res_bb"]
        rgn_agg_func                                = params["rgn_agg_func"]
        df_hist_feats                               = params["df_hist_feats"]
        _finish_traj_df                             = params["_finish_traj_df"]

        instem = 'input/%s/r_%05d/%s.r_%05d.prot' % (myens, r, myens, r)
        mytraj = _md.load('%s.%s' % (instem, fself.intrajformat), top='%s.pdb' % instem)

        a_rnm     = fself._get_raw_topology_ids('%s.pdb' % instem, "residue").rnm.values
        a_residue = list(mytraj.topology.residues)
        a_seg     = [a.segment_id for a in a_residue]
        a_res     = [a.resSeq for a in a_residue]
        #a_rnm = [fself.rnm2pdbrnm(a.name) for a in a_residue]
        #a_rnm = [a.name for a in a_residue]
        l_lbl = ['seg', 'res', 'rnm']
        traj_df = _pd.DataFrame(data={'seg': a_seg, 'rnm': a_rnm, 'res': a_res })
        traj_df = traj_df[l_lbl]

        a_feat = (_md.compute_dssp(mytraj, simplified = DSSPpars[1]) == DSSPpars[0]).astype("int")
        return _finish_traj_df(fself, l_lbl, traj_df, a_feat, df_rgn_seg_res_bb, rgn_agg_func, r, df_hist_feats)
