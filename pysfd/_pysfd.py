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
=======================================
"""

# only necessary for Python 2
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import warnings as _warnings
import numpy as _np
import pandas as _pd
import biopandas.pdb as _bp
import subprocess as _subprocess
import shlex as _shlex
import os as _os
import time as _time
import sys as _sys
import signal as _signal
import itertools as _itertools
# multi-processing module able to pickle, e.g, lambda functions
import pathos as _pathos
import traceback as _traceback
# for circular statistics in srf.Dihedral
import scipy.stats as _scipy_stats
import scipy.special as _scipy_special
import matplotlib as _matplotlib
# do not interact with matplotlib plots:
_matplotlib.use('Agg')
import matplotlib.pyplot as _plt
from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
import pickle as _pickle
#import socket as _socket

import seaborn as _sns; _sns.set()

if _sys.version_info[0] < 3:
    # Define a non-daemon pathos process pool
    # see
    # https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    class _NoDaemonProcess(_pathos.multiprocessing.Pool.Process):
        # make 'daemon' attribute always return False
        def _get_daemon(self):
            return False

        def _set_daemon(self, value):
            pass

        daemon = property(_get_daemon, _set_daemon)

    class _NoDaemonPool(_pathos.multiprocessing.Pool):
        Process = _NoDaemonProcess
elif _sys.version_info[0] >= 3:
    # for a minimal working example on how to implement non-daemon processes
    # carefully executing daemon subprocesses, see
    # https://stackoverflow.com/questions/47574860/python-pathos-process-pool-non-daemonic
    #
    import multiprocess.context as _context

    class _NoDaemonProcess(_context.Process):
        def _get_daemon(self):
            return False

        def _set_daemon(self, value):
            pass

        daemon = property(_get_daemon, _set_daemon)

    class _NoDaemonPool(_pathos.multiprocessing.Pool):
        def Process(self, *args, **kwds):
            return _NoDaemonProcess(*args, **kwds)




class PySFD(object):
    """
    PySFD main class

    Input Trajectory Files should be organized as:
    input/%s/r_%05d/%s.r_%05d.%s.%s' % (myens, r, myens, r, subsys, intrajformat)
    each with a PDB File containing the topology information:
    'input/%s/r_%05d/%s.r_%05d.%s.pdb' % (myens, r, myens, r, subsys)
    , where
    * myens is the name of the simulated ensemble
    * r is the replica index
    * subsys is a trajectory label for the subset of atoms considered (e.g. "prot", "noh")
    * intrajformat is the trajectory format

    Parameters
    ----------
    * l_ens_numreplica : list of lists of [str, int], i.e.
        [ensemble label, number of replica], e.g.,
        l_ens_numreplica = [["WT", 401], ["MT1", 390], ["MT2", 410]]
        Significant feature differences are computed along this list as
        "right with respect to left", e.g.:
        MT2 wrt. MT1
        MT2 wrt. WT
        MT1 wrt. WT

    * FeatureObj : object derived from
                   PySFD.features._feature_agent._FeatureAgent
                   whose _feature_func_engine() computes a particular feature's statistics

    * intrajdatatype : string, default="samplebatches"
        * 'samplebatches'   : trajectories each containing frames sampled from
                              a stationary distribution of, e.g.,
                              a trajectory-bootstrapped MSM or a bayesian MSM sample
                              of bootstrapped frames drawn, e.g., from
                              a meta-stable set of a Markov State Model
        * 'raw'             : plain simulation trajectories, whose feature statistics 
                              - means or (means and standard deviations) - 
                              are further bootstrapped (with replacement) on the trajectory level
                              (with num_bs bootstraps, see below)
        * 'convcheck'       : for convergence checks of the significant feature differences:
                              samples a single bootstrap (with replacement) from
                              the input trajectories containing num_bs (<= numreplica)
                              bootstrapped trajectories;
                              for the actual convergence check, you need to run PySFD
                              with the parameter value 'convcheck' multiple times with
                              varying parameter values of *num_bs*
                              (see Fig. S5 in the manuscript and
                              PySFD/docs/PySFD_example/convcheck folder for example scripts)
        * 'bootstraps'      : creates num_bs bootstraps (with replacement) from input trajectories
                              for later processing
                              TODO: implement SFD computations

    * intrajformat : string, default = "xtc"
        Input trajectory format
        |  "xtc" : gromacs xtc format (additional pdb file for topolgy information required, see example)
        |  "dcd" :         dcd format (additional pdb file for topolgy information required, see example)
        |  "pdb" :         pdb format

    * num_bs: int, optional if intrajdatatype != "samplebatches", default = 10
        Number of bootstrapped trajectory sets to create

    * maxnumframes: int, optional, default = -1
        Maximum number of frames to analyze in each trajectory,
        if not > 0, then all frames in each trajectory will be analyzed
        (currently only implemented for spbsf.HBond_mdtraj)

    * rnm2pdbrnm : dict, optional, default = None
            Maps topology residue name to PDB residue name, e.g.,
            rnm2pdbrnm = {"ASH": "ASP", "GLH": "GLU", ... }
            If None, use the default hard-coded dict
            so far, this is only used in features.srf.RSASA_sr()

    * l_bb_atomnames : list, optional, default = None
        List of atom names of the protein backbone, e.g.,
        l_bb_atomnames = ["N", "CA", "C", "O"]
        If None, use the default, hard-coded
        l_bb_atomnames = ["N", "CA", "C", "O"]

    Special Note:
    This package uses the pandas module, whose std() function applied
    on an array-like of length N uses \sqrt(N - 1) and not
    \sqrt(N) - as in numpy - as the denominator.
    Therefore, all manual standard deviation implementations in PySFD
    consistently use \sqrt(N - 1).

    In case of timing different multiprocessings, please keep in mind
    that comp_features() and run_ens() use time.sleep() commands that
    you may want to temporarily disable
    """

    def __init__(self, l_ens_numreplica = None, FeatureObj = None, intrajdatatype = "samplebatches",
                 intrajformat = "xtc", num_bs = 10, maxnumframes = -1, rnm2pdbrnm = None, l_bb_atomnames = None):
        param2possible_values = dict(intrajdatatype = ['samplebatches', 'raw', 'convcheck', 'bootstraps'], intrajformat = ['xtc', 'dcd', 'pdb'])
        for myparam in param2possible_values:
            if eval(myparam) not in param2possible_values[myparam]:
                raise ValueError("parameter %s with value %s not in %s" % (myparam, eval(myparam),
                                                                           param2possible_values[myparam]))

        self.l_ens               = _np.transpose(l_ens_numreplica)[0] if l_ens_numreplica is not None else None
        self.l_ens_numreplica    = dict(l_ens_numreplica) if l_ens_numreplica is not None else None
        self.intrajdatatype      = intrajdatatype
        
        self.init_FeatureObj(FeatureObj)

        self.l_bb_atomnames      = l_bb_atomnames
        self.intrajformat        = intrajformat
        self.pkg_dir             = _os.path.dirname(_os.path.abspath(__file__))
        self.num_bs              = num_bs
        self.maxnumframes        = maxnumframes
        if rnm2pdbrnm is None:
            self._rnm2pdbrnm     = {
                                   "ASH": "ASP",
                                   "CYX": "CYS",
                                   "GLH": "GLU",
                                   "HIE": "HIS",
                                   "HID": "HIS",
                                   "HIP": "HIS",
                                   "HSD": "HIS",
                                   "HSE": "HIS"}
        else:
            self._rnm2pdbrnm = rnm2pdbrnm
        if l_bb_atomnames is None:
            self.l_bb_atomnames  = ["N", "CA", "C", "O"]
        else:
            self.l_bb_atomnames  = l_bb_atomnames

        self.l_lbl               = {}

        # Preliminarily as of 10/2017, error_type is inherently defined in
        # feat_func(), and updated accordingly in
        # self.error_type[self._feature_func_name]
        # in PySFD.run_ens() based on
        # "if not "sf" in l_traj_df[0].columns:" (see below)
        #
        # Alternatively, one could directly define error_type as self.error_type
        # and make feat_func() read it, but then looping through a list of 
        # feat_func() items would require
        # updating self.error_type for every iteration
        self.error_type          = {}
        self.max_mom_ord         = {}
        self.df_features         = {}
        self.df_fhists           = {}
        self.df_feature_diffs    = {}
        self.is_bb               = lambda x: 1 if x in ["N", "CA", "C", "O"] else 0

    def init_FeatureObj(self, FeatureObj):
        # use feature_func setter for proper initialization
        self._feature_func       = FeatureObj.get_feature_func() if FeatureObj is not None else None
        self._feature_func_name  = self._feature_func.__name__   if FeatureObj is not None else None
        self._num_sigma_funit    = { self._feature_func_name : None }

    def rnm2pdbrnm(self, x):
        if x not in self._rnm2pdbrnm:
            return x
        else:
            return self._rnm2pdbrnm[x]

    @property
    def feature_func(self):
        return self._feature_func

    @feature_func.setter
    def feature_func(self, value):
        self._feature_func      = value.get_feature_func()
        self._feature_func_name = self._feature_func.__name__
        if self._feature_func_name not in self._num_sigma_funit:
            self._num_sigma_funit[self._feature_func_name] = None

    @property
    def feature_func_name(self):
        return self._feature_func_name

    @feature_func_name.setter
    def feature_func_name(self, value):
        pass

    @property
    def num_sigma_funit(self):
        return self._num_sigma_funit[self._feature_func_name]

    @num_sigma_funit.setter
    def num_sigma_funit(self, value):
        pass

    def _get_raw_topology_ids(self, inpdb, idlevel = "atom"):
        """
        obtains the original topology identifiers
        (chainID, Residue Name, Residue Number, Atom Name) for idlevel="atom"
        or
        (chainID, Residue Name, Residue Number)            for idlevel="residue"
        from the "ATOM" entries of a PDB file
        (this function is necessary because mdtraj (<=1.9.1) normalizes these
        residue names (e.g. from "ASH"->"ASP"), and therefore removes
        information about, e.g., different protonation states)

        Parameters:
        ---------
        * inpdb   : str, input pdb file name
        * idlevel : str, identifier level:
                    "atom"    : extract IDs per atom
                    "residue" : extract IDs per residue

        Returns:
        --------
        * df_pdb : pandas DataFrame containing toplogy IDs of
                  (chainID, Residue Name, Residue Number, Atom Name) for idlevel="atom"
                  or
                  (chainID, Residue Name, Residue Number)            for idlevel="residue"
        """

        if idlevel not in ["residue", "atom"]:
            raise ValueError("Warning: idlevel not in [\"residue\", \"atom\"]")
        df_pdb = _bp.PandasPdb().read_pdb(inpdb).df["ATOM"][['chain_id', 'residue_number', 'residue_name', 'atom_name']]
        df_pdb.columns = ["seg", "res", "rnm", "anm"]
        if idlevel == "residue":
            df_pdb = df_pdb[["seg", "res", "rnm"]].drop_duplicates()
        return df_pdb

    def run_ens(self, args):
        """ 
        Computes feature data for an MD-simulated ensemble
        (same protein topology across all replica)

        Parameters in args:
        ----------
        * myens : string
            Name of simulated ensemble
        * numreplica : int
            Number of replica
        * maxworkers : int
            Maximum Number of workers for each simulated ensemble

        Returns
        ----------
        * myensdf, pandas.DataFrame
            Dataframe containing average and standard errors of interaction observables (e.g. frequencies)
            for all relevant pairs of residue backbone/sidechain entities
        """
        def mycircmean(x):
            return _scipy_stats.circmean(x, low = -_np.pi, high = _np.pi)
        def mycircstd(x):
            return _scipy_stats.circstd(x, low = -_np.pi, high = _np.pi)

        myens, numreplica, max_workers = args
        l_args = [(self, myens, r) for r in range(numreplica)]

        pool = _pathos.pools.ProcessPool(max_workers)
        pool.restart(force=True)
        try:
            if self.intrajdatatype != "convcheck":
                feature_func_results = pool.amap(self.feature_func, l_args)
                counter_i = 0
                while not feature_func_results.ready():
                    _time.sleep(2)
                    if counter_i % 30 == 0:
                        print('Waiting for child processes running in pool.amap() in run_ens( {} )'.format(myens))
                    counter_i += 1
                l_traj_df, dataflags = list(zip(*feature_func_results.get()))
            else:
                picklefname = "output/tmp/%s.%s/%s/%s.%s.%s.mnf_%d.pickle.dat" % (self.feature_func_name, self.intrajdatatype, myens, self.feature_func_name, self.intrajdatatype, myens, self.maxnumframes)
                if not _os.path.isfile(picklefname): 
                    _subprocess.Popen(_shlex.split("mkdir -p output/tmp/%s.%s/%s" % (self.feature_func_name, self.intrajdatatype, myens))).wait()
                    feature_func_results = pool.amap(self.feature_func, l_args) 
                    counter_i = 0 
                    while not feature_func_results.ready(): 
                        _time.sleep(2) 
                        if counter_i % 30 == 0: 
                            print('Waiting for child processes running in pool.amap() in run_ens( {} )'.format(myens)) 
                        counter_i += 1 
                    l_traj_df, dataflags = list(zip(*feature_func_results.get()))
                    with open(picklefname, "wb") as f:
                        _pickle.dump((l_traj_df, dataflags), f)
                else:
                    with open(picklefname, "rb") as f:
                        print("reloading %s" % picklefname)
                        l_traj_df, dataflags = _pickle.load(f)

            pool.close()
            pool.join()
            l_traj_df = list(l_traj_df)
            dataflags = list(dataflags)
            if len([x for x in dataflags if str(x) != str(dataflags[0])]) > 0:
                raise ValueError("elements in dataflags are not the same across difference ensemble replica!")
            dataflags = dataflags[0]
        except:
            _sys.stdout.flush()
            _traceback.print_exc()
            _sys.stderr.flush()
            print('\nAttempting to kill session after exception: {} run_ens( {} )\n'.format(_sys.exc_info(), myens))
            _sys.stdout.flush()

        self.error_type[self._feature_func_name]  = dataflags.get("error_type")
        self.max_mom_ord[self._feature_func_name] = dataflags.get("max_mom_ord", 1)
        if self.error_type[self._feature_func_name] is None:
            raise ValueError("error_type is None !")
        self.df_fhists[self.feature_func_name]    = dataflags.get("df_fhists", None)
        is_with_dwell_times                       = dataflags.get("is_with_dwell_times", False)
        circular_stats                            = dataflags.get("circular_stats")
        
        if self.error_type[self._feature_func_name] == 'std_err':
            if is_with_dwell_times:
                l_obs = ['f', 'ton', 'tof']
                dict_groups = {'mf'      : lambda g: g.iloc[0],
                               'mton'    : lambda g: g.iloc[0],
                               'mtof'    : lambda g: g.iloc[0],
                               'sf'      : _np.sum,
                               'ston'    : _np.sum,
                               'stof'    : _np.sum,
                               'sfcount' : _np.sum}
            else:
                l_obs = ['f'] + ['f.%d' % mymom for mymom in range(2, self.max_mom_ord[self._feature_func_name]+1)]
                dict_groups = [('mf',      lambda g: g.iloc[0]),
                               ('sf',      _np.sum),
                               ('sfcount', _np.sum)]
                dict_groups += [('mf.%d' % mymom, lambda g: g.iloc[0]) for mymom in range(2, self.max_mom_ord[self._feature_func_name]+1)]
                dict_groups += [('sf.%d' % mymom, _np.sum)             for mymom in range(2, self.max_mom_ord[self._feature_func_name]+1)]
                dict_groups = dict(dict_groups)
        elif self.error_type[self._feature_func_name] == 'std_dev':
            if is_with_dwell_times:
                l_obs    = ['f', 'ton', 'tof', 'sf', 'ston', 'stof']
                l_rename = ['f', 'ton', 'tof']
                dict_groups = {'mf'      : _np.sum,
                               'mton'    : _np.sum,
                               'mtof'    : _np.sum,
                               'sf'      : _np.sum,
                               'ston'    : _np.sum,
                               'stof'    : _np.sum}
            else:
                l_obs    = ['f', 'sf']
                l_rename = ['f']
                dict_groups = {'mf'      : _np.sum,
                               'sf'      : _np.sum}

        self.l_lbl[self.feature_func_name] = [ l for l in l_traj_df[0].columns if l not in ["r", 'fhist'] + l_obs]

        def mark_MTS(x):
            return 'MTS' if len(_np.unique(x)) > 1 else x.iloc[0]
        if 'rnm' in self.l_lbl[self.feature_func_name]:
            dict_groups['rnm'] = mark_MTS
        elif ('rnm1' in self.l_lbl[self.feature_func_name]) and ('rnm2' in self.l_lbl[self.feature_func_name]):
            dict_groups['rnm1'] = mark_MTS
            dict_groups['rnm2'] = mark_MTS
        l_lbl_no_rnm = [x for x in self.l_lbl[self.feature_func_name] if x not in ['rnm', 'rnm1', 'rnm2']]
        if self.intrajdatatype == "samplebatches":
            l_myensdf = [_pd.concat(l_traj_df, copy=False)]
            #myensdf.to_csv("df_myensdf.%s.dat" % myens, sep = "\t")
        elif self.intrajdatatype == "raw":
            # bootstrap regular simulation trajectories for statistical significance
            l_bsensdf = []
            for r in range(self.num_bs):
                myinds = _np.random.choice(range(len(l_traj_df)), size=numreplica, replace=True)
                myensdf = _pd.concat([l_traj_df[i] for i in myinds], copy=False)
                for myobs in l_obs:
                    myensdf[myobs] = 1. * myensdf.groupby(
                        self.l_lbl[self.feature_func_name])[myobs].transform("sum") / numreplica
                myensdf["r"] = r
                l_bsensdf.append(myensdf)
            l_myensdf = [_pd.concat(l_bsensdf, copy=False)]
        elif self.intrajdatatype == "convcheck":
            myinds = _np.random.choice(range(len(l_traj_df)), size=self.num_bs, replace=True)
            myensdf = [l_traj_df[i].copy() for i in myinds]
            for myensind in range(len(myensdf)):
                myensdf[myensind]["r"] = myensind
            self.l_ens_numreplica = { i : self.num_bs for i,j in self.l_ens_numreplica.items() }
            numreplica = self.num_bs
            l_myensdf = [_pd.concat(myensdf, copy=False)]
        elif self.intrajdatatype == "bootstraps":
            l_myensdf = []
            for bs in range(self.num_bs):
                myinds = _np.random.choice(range(len(l_traj_df)), size=numreplica, replace=True)
                myensdf = [l_traj_df[i].copy() for i in myinds]
                for myensind in range(len(myensdf)):
                    myensdf[myensind]["r"] = myensind
                l_myensdf.append(_pd.concat(myensdf, copy=False))

        def myfunc(x, numframes):
            if _np.any(x.isnull()):
                return _np.float("NaN")
            else:
                dbin = x.iloc[0][1]
                dbin = dbin[1] - dbin[0]
                prec = len(str(dbin).partition(".")[2])
                myzerohist = tuple([_np.array([1/dbin]), _np.array([0, dbin])])
                l_dfhist  = [_pd.DataFrame({i : x.iloc[i][0]},  index = _np.round(x.iloc[i][1][:-1],  prec).astype("str")) for i in range(len(x))]
                l_dfhist += [_pd.DataFrame({i : myzerohist[0]}, index = _np.round(myzerohist[1][:-1], prec).astype("str")) for i in range(len(x), numframes)]
                df_hist = _pd.concat(l_dfhist, axis = 1, sort = True, join = "outer")
                del l_dfhist
                df_hist.fillna(0, inplace = True)
                df_hist = _pd.DataFrame({ "hmean" : df_hist.apply("mean", axis = 1), "hstd" : df_hist.apply("std", axis = 1)}, index = df_hist.index)
                allbins = _pd.DataFrame( index = _np.round(_np.arange(_np.round(df_hist.index.astype("float").min()/dbin), _np.round(df_hist.index.astype("float").max()/dbin+1), 1)*dbin, prec).astype("str"))
                df_hist = _pd.concat([df_hist, allbins], axis = 1, sort = True, join = "outer")
                df_hist["bin"] = df_hist.index.astype("float")
                df_hist.sort_values(by = "bin", inplace = True)
                df_hist.fillna(0, inplace = True)
                a_bin = df_hist.bin.values
                a_bin = _np.round(_np.append(a_bin, a_bin[-1]+dbin), prec)
                return tuple([a_bin, df_hist.hmean.values, df_hist.hstd.values])

        for myind, myensdf in enumerate(l_myensdf):
            if 'fhist' in myensdf.columns:
                numframes = len(myensdf.r.unique())
                my_dict_groups = { 'fhist' : lambda x: myfunc(x, numframes) }
                if 'rnm' in self.l_lbl[self.feature_func_name]:
                    my_dict_groups['rnm'] = mark_MTS
                    df_hist = myensdf.loc[(~_pd.isnull(myensdf.fhist))|(_pd.isnull(myensdf.f))]
                    df_hist = df_hist.groupby(l_lbl_no_rnm)[['rnm', 'fhist']]
                    df_hist = df_hist.agg(my_dict_groups).to_frame()
                    df_hist = df_hist[self.l_lbl[self.feature_func_name] + ['fhist']]
                elif ('rnm1' in self.l_lbl[self.feature_func_name]) and ('rnm2' in self.l_lbl[self.feature_func_name]):
                    my_dict_groups['rnm1'] = mark_MTS
                    my_dict_groups['rnm2'] = mark_MTS
                    df_hist = myensdf.loc[(~_pd.isnull(myensdf.fhist))|(_pd.isnull(myensdf.f))]
                    df_hist = df_hist.groupby(l_lbl_no_rnm)[['rnm1', 'rnm2', 'fhist']]
                    df_hist = df_hist.agg(my_dict_groups).to_frame()
                    df_hist = df_hist[self.l_lbl[self.feature_func_name] + ['fhist']]
                else:
                    df_hist = myensdf.loc[(~_pd.isnull(myensdf.fhist))|(_pd.isnull(myensdf.f))]
                    df_hist = df_hist.groupby(l_lbl_no_rnm)['fhist']
                    df_hist = df_hist.agg(my_dict_groups).to_frame()
                #df_hist = myensdf.loc[~_pd.isnull(myensdf.fhist)].groupby(self.l_lbl[self.feature_func_name])['fhist'].agg(lambda x: myfunc(x, numframes))
                #df_hist = myensdf.loc[(~_pd.isnull(myensdf.fhist))|(_pd.isnull(myensdf.f))].groupby(self.l_lbl[self.feature_func_name])['fhist'].agg(lambda x: myfunc(x, numframes)).to_frame()
                myensdf = myensdf.loc[~_pd.isnull(myensdf.f)]
                myensdf.drop(columns = "fhist", inplace = True)
            else:
                df_hist = None
            if self.error_type[self._feature_func_name] == 'std_err':
                #mygroup = myensdf.groupby(l_lbl_no_rnm)
                # check for circular statistics
                if circular_stats == "csd":
                    #dict_groups = dict([('f', [mycircmean, mycircstd])] + [('f.%d' % mymom, [mycircmean, mycircstd]) for mymom in range(2, self.max_mom_ord[self._feature_func_name]+1)])
                    dict_groups = dict([('f', [mycircmean, mycircstd])])
                    if 'rnm' in self.l_lbl[self.feature_func_name]:
                        dict_groups['rnm'] = mark_MTS
                    elif ('rnm1' in self.l_lbl[self.feature_func_name]) and ('rnm2' in self.l_lbl[self.feature_func_name]):
                        dict_groups['rnm1'] = mark_MTS
                        dict_groups['rnm2'] = mark_MTS
                    myensdf = myensdf.groupby(l_lbl_no_rnm).agg(dict_groups)
                    myensdf.rename(columns = { 'mycircmean' : 'm', 'mycircstd' : 's' }, level = 1, inplace = True)
                    myensdf.columns = [myensdf.columns.map('{0[1]}{0[0]}'.format)]
                else:
                    # the following unusual way to compute mean/std frequencies for each feature
                    # accounts for missing frequency entries of zero-frequency trajectories in sPBSF features
                    for myobs in l_obs:
                        mygroup = myensdf.groupby(l_lbl_no_rnm)
                        myensdf['m' + myobs] = 1. * mygroup[myobs].transform('sum') / numreplica
                        myensdf['s' + myobs] = (myensdf[myobs] - myensdf['m' + myobs]) ** 2
                    del mygroup
                    myensdf['sfcount'] = 1
                    myensdf = myensdf.groupby(l_lbl_no_rnm).agg(dict_groups)
                    for myobs in l_obs:
                        # add contributions from missing frequency entries, i.e. for which myensdf["freq"] = 0
                        myensdf['s' + myobs] += (numreplica - myensdf['sfcount']) * myensdf['m' + myobs] ** 2
                        myensdf['s' + myobs] = _np.sqrt(1. * myensdf['s' + myobs] / (numreplica - 1))
                    del myensdf["sfcount"]
            elif self.error_type[self._feature_func_name] == 'std_dev':
                if circular_stats == "csd":
                    dict_groups = { 'f' : mycircmean, 'sf' : mycircmean }
                    if 'rnm' in self.l_lbl[self.feature_func_name]:
                        dict_groups['rnm'] = mark_MTS
                    elif ('rnm1' in self.l_lbl[self.feature_func_name]) and ('rnm2' in self.l_lbl[self.feature_func_name]):
                        dict_groups['rnm1'] = mark_MTS
                        dict_groups['rnm2'] = mark_MTS
                    myensdf = myensdf.groupby(l_lbl_no_rnm).agg(dict_groups)
                    myensdf.rename(columns = { 'f' : 'mf' }, inplace = True)
                else:
                    for myobs in l_rename:
                        myensdf.rename(columns = { myobs : 'm' + myobs }, inplace = True)
                    myensdf  = myensdf.groupby(l_lbl_no_rnm).agg(dict_groups)
                    myensdf /= 1. * numreplica
            if df_hist is not None:
                myensdf = myensdf.merge(df_hist, left_index = True, right_index = True, how = "outer")
            myensdf = myensdf.reset_index()[self.l_lbl[self.feature_func_name] + [x for x in myensdf.columns if x not in self.l_lbl[self.feature_func_name]]].set_index(self.l_lbl[self.feature_func_name])
            l_myensdf[myind] = _pd.concat([myensdf], axis=1, keys=[myens]).reset_index(drop = False)
        return [l_myensdf, self.l_lbl, self.error_type, self.max_mom_ord]

    def comp_features(self, max_workers=(1, 1)):
        """
        Computes a features table for all simulated ensembles

        Parameters
        ----------
        * max_workers : tuple of ints
            (i,j), where i is the maximum number of processes that simultaneously   for all simulated ensembles and
                         j is the number of assigned sub-processes for all replicas in a simulated ensemble
            in total, i * j cores will be used simultaneously
        """
        l_args = [(myens, self.l_ens_numreplica[myens], max_workers[1]) for myens in self.l_ens]

        if (max_workers[0] * max_workers[1]) > _pathos.multiprocessing.cpu_count():
            warnstr = "Only %d CPUs available, but with parameter 'max_workers', \
                       %d * %d = %d CPUs were requested!" % (_pathos.multiprocessing.cpu_count(),
                                                       max_workers[0],
                                                       max_workers[1],
                                                       max_workers[0] * max_workers[1])
            _warnings.warn(warnstr)
        pool = _NoDaemonPool(max_workers[0])
        #pool.restart(force=True)
        try:
            #_subprocess.Popen(_shlex.split("mkdir -p logs/%s" % _socket.gethostname())).wait()
            #with open("logs/%s/top.out" % _socket.gethostname(), "wb") as outf:
            #    _subprocess.Popen(_shlex.split("top -u stolzen -c -d 5 -n 30 -b"), stdout=outf, stderr=outf)
            results = pool.map_async(self.run_ens, l_args)
            counter_i = 0
            while not results.ready():
                if counter_i % 30 == 0:
                    print("Waiting for child processes running in pool.map_async() in comp_features()")
                _time.sleep(2)
                counter_i += 1
            results = results.get()
            pool.close()
            pool.join()
        except:
            _sys.stdout.flush()
            _traceback.print_exc()
            _sys.stderr.flush()
            print('\nAttempting to kill session after exception: {} in comp_features()\n'.format(_sys.exc_info()))
            _sys.stdout.flush()
            _os.killpg(_os.getpgid(0), _signal.SIGTERM)

        if self.intrajdatatype == "bootstraps":
            l_df_features           = [None] * self.num_bs
            l_df_features_segresrnm = [None] * self.num_bs
        else:
            l_df_features           = [None]
            l_df_features_segresrnm = [None]

        for myens, myresults in zip(self.l_ens, results):
            l_myensdf, self.l_lbl, self.error_type, self.max_mom_ord = tuple(myresults)
            print(myens)
            # distinguish whether "rnm", "rnm1"/"rnm2", or not any are in the labels
            # (used later to turn any different residue names into "MTS", see below)
            if 'rnm' in self.l_lbl[self.feature_func_name]:
                is_rnm = 'rnm'
            elif ('rnm1' in self.l_lbl[self.feature_func_name]) and ('rnm2' in self.l_lbl[self.feature_func_name]):
                is_rnm = 'rnm12'
            else:
                is_rnm = False

            for myind, myensdf in enumerate(l_myensdf):
                # prepare a DataFrame df_myens_segresrnm containing segresrnm ID information
                if is_rnm == 'rnm':
                    myensdf['segres'] = myensdf['seg'].astype(str) + '_' + myensdf['res'].astype(str)
                    df_myens_segresrnm = _pd.DataFrame({
                     'segres' : myensdf['segres'],
                     'rnm'    : myensdf['rnm']}).drop_duplicates()
                elif is_rnm == 'rnm12':
                    myensdf['segres1'] = myensdf['seg1'].astype(str) + '_' + myensdf['res1'].astype(str)
                    myensdf['segres2'] = myensdf['seg2'].astype(str) + '_' + myensdf['res2'].astype(str)
                    df_myens_segresrnm = _pd.DataFrame({
                     'segres' : _np.concatenate((myensdf['segres1'], myensdf['segres2'])),
                     'rnm' : _np.concatenate((myensdf['rnm1'], myensdf['rnm2']))}).drop_duplicates()
                # for any additional ensemble (myensdf), check for differing residue names
                # that are not NaN and rename them as "MTS"
                if l_df_features[myind] is None:
                    l_df_features[myind] = myensdf
                    if is_rnm in ['rnm', 'rnm12']:
                        l_df_features_segresrnm[myind] = df_myens_segresrnm.copy()
                else:
                    if is_rnm in ['rnm', 'rnm12']:
                        df_tmp = l_df_features_segresrnm[myind].merge(df_myens_segresrnm, how = "outer", on = ['segres'])
                        df_tmp = df_tmp.loc[df_tmp.rnm_x != df_tmp.rnm_y]
                        df_tmp = df_tmp.loc[df_tmp.rnm_x == df_tmp.rnm_x]
                        df_tmp = df_tmp.loc[df_tmp.rnm_y == df_tmp.rnm_y].reset_index(drop = True)
                        if is_rnm == 'rnm':
                            l_df_features[myind].loc[_np.in1d(l_df_features[myind].segres,  df_tmp.segres), 'rnm'] = 'MTS'
                            myensdf.loc[    _np.in1d(myensdf.segres,      df_tmp.segres), 'rnm'] = 'MTS'
                        else:
                            l_df_features[myind].loc[_np.in1d(l_df_features[myind].segres1, df_tmp.segres), 'rnm1'] = 'MTS'
                            l_df_features[myind].loc[_np.in1d(l_df_features[myind].segres2, df_tmp.segres), 'rnm2'] = 'MTS'
                            myensdf.loc[    _np.in1d(myensdf.segres1,     df_tmp.segres), 'rnm1'] = 'MTS'
                            myensdf.loc[    _np.in1d(myensdf.segres2,     df_tmp.segres), 'rnm2'] = 'MTS'
    
                    # update l_df_features_segresrnm[myind]
                    l_df_features[myind] = l_df_features[myind].merge(myensdf, how="outer")
                    if is_rnm == 'rnm':
                        l_df_features_segresrnm[myind] = _pd.DataFrame({
                         'segres' : l_df_features[myind]['segres'],
                         'rnm'    : l_df_features[myind]['rnm']}).drop_duplicates()
                    elif is_rnm == 'rnm12':
                        l_df_features_segresrnm[myind] = _pd.DataFrame({
                         'segres' : _np.concatenate((l_df_features[myind]['segres1'], l_df_features[myind]['segres2'])),
                         'rnm'    : _np.concatenate((l_df_features[myind]['rnm1'],    l_df_features[myind]['rnm2']))}).drop_duplicates()

        self.df_features[self.feature_func_name] = []
        if "fhist" in l_df_features[-1].columns.get_level_values(1):
            self.df_fhists[self.feature_func_name] = []

        for myind in range(len(l_myensdf)):
            if "fhist" in l_df_features[myind].columns.get_level_values(1):
                l_obs = ["fhist", "mf", "sf"] + ['%s.%d' % (flbl, mymom) for mymom in range(2, self.max_mom_ord[self._feature_func_name]+1) for flbl in ['mf', 'sf'] ]
            else:
                l_obs = ["mf", "sf"] + ['%s.%d' % (flbl, mymom) for mymom in range(2, self.max_mom_ord[self._feature_func_name]+1) for flbl in ['mf', 'sf'] ]
            nanentries = l_df_features[myind].loc[_np.all(l_df_features[myind].loc[:,l_df_features[myind].columns.get_level_values(1) == "mf"].isnull(), axis = 1)]
            if len(nanentries) > 0:
                warnstr = "The following feature labels defined in df_hist_feats do not match with any simulated ensemble:\n%s" % (nanentries)
                _warnings.warn(warnstr)
            l_df_features[myind].iloc[:, l_df_features[myind].columns.get_level_values(1).isin(l_obs)] = \
                l_df_features[myind].iloc[:, l_df_features[myind].columns.get_level_values(1).isin(l_obs)].fillna(0)
            if is_rnm == 'rnm':
                del l_df_features[myind]['segres']
            elif is_rnm == 'rnm12':
                del l_df_features[myind]['segres1']
                del l_df_features[myind]['segres2']
            l_df_features[myind] = l_df_features[myind].sort_values(by=self.l_lbl[self.feature_func_name])
            if "fhist" in l_df_features[myind].columns.get_level_values(1):
                # only consider features for histogramming, i.e. from df_hist_feats
                self.df_fhists[self.feature_func_name].append(
                l_df_features[myind].loc[
                    ~_np.all(l_df_features[myind].loc[:,l_df_features[myind].columns.get_level_values(1) == "fhist"] == 0, axis = 1),
                    (l_df_features[myind].columns.get_level_values(1)=="fhist") |
                    (_np.in1d(l_df_features[myind].columns.get_level_values(0), self.l_lbl[self.feature_func_name] ))])
                #.reset_index(drop=True)
                self.df_fhists[self.feature_func_name][-1].columns = self.df_fhists[self.feature_func_name][-1].columns.droplevel(1)
            self.df_features[self.feature_func_name].append(l_df_features[myind].drop(columns = "fhist", level = 1).reset_index(drop=True))
        
        if self.intrajdatatype in ["bootstraps"]:
            if ((len(self.df_features[self.feature_func_name]) == 1)):
                raise ValueError("self.intrajdatatype is in [\"bootstraps\"] but len(self.df_features[%s]) is zero!" % (self.df_features[self.feature_func_name]))
        else:
            self.df_features[self.feature_func_name] = self.df_features[self.feature_func_name][0]
            if self.feature_func_name in self.df_fhists:
                self.df_fhists[self.feature_func_name]   = self.df_fhists[self.feature_func_name][0]

    def comp_feature_diffs_old(self, num_sigma=2, num_funit=0):
        """
        Computes statistically significant differences in
        mean features between pairs of simulated ensembles

        Parameters
        ----------
        * num_sigma : float, level of significance, measured in multiples of joint uncertainty (see paper)
        * num_funit : float, level of biological significance, measured in multiples of feature units (see paper)
        (Note: significance is defined by both statistical AND biological significance !)

        Stores
        ------
        * self.df_feature_diffs[self.feature_func_name]:
          "sdiff":
              significant difference, see paper
          "score":
              if self.error_type = "std_err":
                  a two-sided Z-score
              if self.error_type = "std_dev":
                  similar to an effect size of a two-populations difference
          "pval":
              if self.error_type = "std_err":
                  p values for two-sided Z-test
        
        """
        self._num_sigma_funit[self.feature_func_name] = (num_sigma, num_funit)
        df_feature_diffs = {}
        for i in range(len(self.l_ens)):
            ens_i = self.l_ens[i]
            for j in range(i + 1, len(self.l_ens)):
                ens_j = self.l_ens[j]
                diff_f = self.df_features[
                        self.feature_func_name][ens_j]["mf"] - self.df_features[self.feature_func_name][ens_i]["mf"]
                absdiff   = _np.abs(diff_f)
                deltadiff = _np.sqrt(
                        self.df_features[self.feature_func_name][ens_i]["sf"] ** 2 +
                        self.df_features[self.feature_func_name][ens_j]["sf"] ** 2)
                dfsel = ( absdiff - _np.maximum( num_sigma * deltadiff, _np.ones(len(diff_f)) * num_funit)).values
                df_feature_diffs[(ens_j, ens_i)] = self.df_features[self.feature_func_name].loc[
                    dfsel>0,
                    [(u, "") for u in self.l_lbl[self.feature_func_name]] +
                    [(u, d) for u in [ ens_j, ens_i ] for d in ["mf", "sf"]]]
                df_feature_diffs[(ens_j, ens_i)]["sdiff"] = _np.sign(diff_f[dfsel>0]) * dfsel[dfsel>0]
                df_feature_diffs[(ens_j, ens_i)]["score"] = diff_f[dfsel>0] / deltadiff[dfsel>0]
                if self.error_type[self.feature_func_name] == "std_err":
                    df_feature_diffs[(ens_j, ens_i)]["pval"]  = _scipy_special.cdf(_np.abs(df_feature_diffs[(ens_j, ens_i)]["score"])) * 2
                df_feature_diffs[(ens_j, ens_i)] = df_feature_diffs[(ens_j, ens_i)].sort_values(
                    by=self.l_lbl[self.feature_func_name]).reset_index(drop=True)
        self.df_feature_diffs[self.feature_func_name] = df_feature_diffs

    def comp_feature_diffs(self, num_sigma=2, num_funit=0):
        """
        Computes statistically significant differences in
        mean features between pairs of simulated ensembles

        Parameters
        ----------
        * num_sigma : float, level of significance, measured in multiples of joint uncertainty (see paper)
        * num_funit : float, level of biological significance, measured in multiples of feature units (see paper)
        (Note: significance is defined by both statistical AND biological significance !)

        Stores
        ------
        * self.df_feature_diffs[self.feature_func_name]:
          "sdiff":
              significant difference, see paper
          "score":
              if self.error_type = "std_err":
                  a two-sided Z-score
              if self.error_type = "std_dev":
                  similar to an effect size of a two-populations difference
          "pval":
              if self.error_type = "std_err":
                  p values for two-sided Z-test
        
        """
        if self.intrajdatatype in ["bootstraps"]:
            raise NotImplementedError("comp_feature_diffs() not implemented yet for self.intrajdatatype in [\"bootstraps\"]")
        self._num_sigma_funit[self.feature_func_name] = (num_sigma, num_funit)
        df_feature_diffs = {}
        for i in range(len(self.l_ens)):
            ens_i = self.l_ens[i]
            for j in range(i + 1, len(self.l_ens)):
                ens_j = self.l_ens[j]
                df_tmp = self.df_features[self.feature_func_name][self.l_lbl[self.feature_func_name] + [ens_j, ens_i]].copy()
                df_tmp["lsdm"] = 0
                for mymom in range(1, self.max_mom_ord[self._feature_func_name]+1):
                    if mymom == 1:
                       mlbl = ""
                    else:
                       mlbl = ".%d" % mymom
                    df_tmp["diff%s" % mlbl]  = df_tmp[ens_j]["mf%s" % mlbl] - df_tmp[ens_i]["mf%s" % mlbl]
                    df_tmp["ddiff%s" % mlbl] = _np.sqrt(
                                               df_tmp[ens_i]["sf%s" % mlbl] ** 2 +
                                               df_tmp[ens_j]["sf%s" % mlbl] ** 2)
                    df_tmp["sdiff%s" % mlbl] = _np.abs(df_tmp["diff%s" % mlbl]) - \
                                               _np.maximum( num_sigma * df_tmp["ddiff%s" % mlbl],
                                                            _np.ones(len(df_tmp)) * num_funit)
                    df_tmp.loc[df_tmp["sdiff%s" % mlbl]<=0, "sdiff%s" % mlbl] = _np.float("NaN")
                    df_tmpbool    = (df_tmp["lsdm"]==0) & (df_tmp["sdiff%s" % mlbl]>0)
                    df_tmp.loc[df_tmpbool, "lsdm"] = mymom

                l_flbl = [ "mf", "sf" ] + ['%sf.%d' % (flbl, mymom) for mymom in range(2, self.max_mom_ord[self._feature_func_name]+1) for flbl in ["m", "s"]]
                df_tmp = df_tmp.loc[df_tmp.lsdm>0]
                for mymom in range(1, self.max_mom_ord[self._feature_func_name]+1):
                    if mymom == 1:
                       mlbl = ""
                    else:
                       mlbl = ".%d" % mymom
                    df_tmp["sdiff%s" % mlbl] = _np.sign(df_tmp["diff%s" % mlbl]) * df_tmp["sdiff%s" % mlbl] 
                    df_tmp["score%s" % mlbl] = df_tmp["diff%s" % mlbl] / df_tmp["ddiff%s" % mlbl]
                    if self.error_type[self.feature_func_name] == "std_err":
                        #https://stackoverflow.com/questions/3496656/convert-z-score-z-value-standard-score-to-p-value-for-normal-distribution-in
                        df_tmp["pval%s" % mlbl] = (1 - _scipy_special.ndtr(_np.abs(df_tmp["score%s" % mlbl]))) * 2
                    df_tmp.drop(columns = ["diff%s" % mlbl, "ddiff%s" % mlbl], level = 0, inplace = True)                    
                df_feature_diffs[(ens_j, ens_i)] = df_tmp.sort_values(
                    by=["lsdm"] + self.l_lbl[self.feature_func_name]).reset_index(drop=True)
        self.df_feature_diffs[self.feature_func_name] = df_feature_diffs

    def comp_feature_diffs_with_dwells(self, num_sigma=2):
        """
        Compute statistically significant differences in
        pairwise interaction frequencies/dwell times between
        pairs of simulated ensembles

        !!!!!!

        Currently, no significant difference in dwell time has been observed in real MD trajectories
        that cannot already be explained by a significant difference in interaction frequency

        If you find such a significant difference, please inform the author (Sebastian Stolzenberg) at
        ss629@cornell.edu
        to claim your complimentary beer / chocolate bar!

        To scan your own MD simulations for such differences, just type in:
        ############
        mySDA = PySFD.PySFD(...)
        mySDA.comp_features(...)
        ...
        mySDA.comp_feature_diffs_with_dwells(num_sigma=2)
        abc = mySDA.df_feature_diffs['pbsi.HBond_VMD.std_err'][('bN82A.pcca1', 'WT.pcca3')]
        # significant differences in 'ton' ("on"  dwell time) that is NOT significant in 'f' (interaction frequency)
        print(abc[(-1, 1, 0)])
        # significant differences in 'tof' ("off" dwell time) that is NOT significant in 'f' (interaction frequency)
        print(abc[(-1, 0, 1)])
        ############

        !!!!!!

        Parameters
        ----------
        * num_sigma: int, level of statistical significance, measure in multiples of standard errors
            (i,j), where i is the number of assigned workers for all simulated ensemble and
                         j is the number of assigned workers for all replicas in a simulated ensemble

        Stores
        -------
        * df_feature_diffs[(ens_j, ens_i)][(is_['f'], is_['ton'], is_['tof'])]
          into self.df_feature_diffs[self.feature_func_name],
          - ens_j, ens_i are the compared ensembles
          - is_['f'], is_['ton'], is_['tof'] are indicators for significance of
            interaction frequency f, avg. interaction "on" dwell time, and
            avg. interaction "off" dwell time, e.g.,
            -1 means NOT significantly different features
             1 means     significantly different features
             0 any                               feature
        """

        self._num_sigma_funit[self.feature_func_name] = (num_sigma, 0.0)

        def myf(x, y):
            return { -1 : _np.logical_not(x), 0 : _np.ones(len(x)).astype('bool'), 1 : x  }[y]

        df_feature_diffs = {}
        for i in range(len(self.l_ens)):
            ens_i = self.l_ens[i]
            for j in range(i + 1, len(self.l_ens)):
                ens_j = self.l_ens[j]
                df_feature_diffs[(ens_j, ens_i)] = {}
                diff_obs = {}
                diffsel = {}
                #nansel = {}
                for myobs in ['f', 'ton', 'tof']:
                    diff_obs[myobs] = self.df_features[
                        self.feature_func_name][ens_j]['m' + myobs] - \
                        self.df_features[self.feature_func_name][ens_i]['m' + myobs]
                    diffsel[myobs]  = (_np.abs(diff_obs[myobs]) - num_sigma * _np.sqrt(
                                    self.df_features[self.feature_func_name][ens_i]['s' + myobs] ** 2 +
                                    self.df_features[self.feature_func_name][ens_j]['s' + myobs] ** 2))
                    #nansel[myobs]   = (diffsel[myobs] != diffsel[myobs])
                    diffsel[myobs]  = (diffsel[myobs] > 0.0)
                is_ = {}
                for is_['f'], is_['ton'], is_['tof'] in _itertools.product(range(-1, 2), range(-1, 2), range(-1, 2)):
                    #mydiffsel = _np.array([ _np.logical_or(myf(diffsel[myobs], is_[myobs]), nansel[myobs])
                    mydiffsel = _np.array([ myf(diffsel[myobs], is_[myobs]) for myobs in ['f', 'ton', 'tof']])
                    mydiffsel = _np.logical_and.reduce(mydiffsel)
                    df_feature_diffs[(ens_j, ens_i)][(is_['f'], is_['ton'], is_['tof'])] = \
                        self.df_features[self.feature_func_name].loc[
                            mydiffsel,
                            [(u, "") for u in self.l_lbl[self.feature_func_name]] +
                            [(u, d) for u in [ ens_j, ens_i ] for d in ["mf", "sf", "mton", "ston", "mtof", "stof"]]]
                    df_feature_diffs[(ens_j, ens_i)][(is_['f'], is_['ton'], is_['tof'])] = \
                        df_feature_diffs[(ens_j, ens_i)][(is_['f'], is_['ton'], is_['tof'])].reset_index(drop=True)
        self.df_feature_diffs[self.feature_func_name] = df_feature_diffs

    def comp_and_write_common_feature_diffs(self, feature_func_name=None, l_sda_pair=None, l_sda_not_pair=None, outdir=None):
        """
        Computes and writes statistically significant differences in
        mean features of "feature_func_name" that are
        common among the pairs of simulated ensembles defined in l_sda_pair, but that are
        not significantly different among all pairs defined in l_sda_not_pair

        Parameters
        ----------
        * feature_func_name: name of the feature function, for which
                             significant differences have been computed
        * l_sda_pair:        list of 2-d tuples of str
                             pairs of simulated ensembles compaired by
                             comp_feature_diffs
        * l_sda_not_pair:    list of 2-d tuples of str
                             pairs of simulated ensembles compaired by
                             comp_feature_diffs
        * outdir : string, optional, default = "output/meta/%s/%s/common" % (feature_func_name,
                                                                             self.intrajdatatype)
                   output path

        Returns
        -------
        * df_merged : Data Frame containing list of common feature difference
        """

        if l_sda_pair is None:
            raise ValueError("l_sda_pair is not defined")
        if feature_func_name is None:
            raise ValueError("feature_func_name is not defined")
        if outdir is None:
            outdir = "output/meta/%s/%s/common" % (feature_func_name, self.intrajdatatype)

        df_merged = None
        # aggregate common feature differences among l_sda_pair
        for mykey in l_sda_pair:
            df_tmp = self.df_feature_diffs[feature_func_name][mykey].copy()
            df_tmp.drop(columns = [mylbl for mylbl in df_tmp.columns.get_level_values(0) if ("score" in mylbl) or ("pval" in mylbl)], level = 0, inplace = True)
            df_tmp.drop(columns = [myens for myens in self.l_ens], level = 0, inplace = True)
            df_tmp[[mylbl for mylbl in df_tmp.columns.get_level_values(0) if ("sdiff" in mylbl)]] = _np.sign(
            df_tmp[[mylbl for mylbl in df_tmp.columns.get_level_values(0)  if ("sdiff" in mylbl)]])
            # use of "column_bak" is just to avoid a PerformanceWarning by pandas.merge
            column_bak = df_tmp.columns.copy()
            df_tmp.columns = df_tmp.columns.droplevel(1)
            if df_merged is None:
                df_merged = df_tmp
            else:
                df_merged = df_merged.merge(
                    df_tmp,
                    on = self.l_lbl[feature_func_name] + ["lsdm"] + [mylbl for mylbl in df_tmp.columns.get_level_values(0) if ("sdiff" in mylbl)])
                    #[self.l_lbl[feature_func_name] + ["lsdm"] + [mylbl for mylbl in df_tmp.columns.get_level_values(0) if ("sdiff" in mylbl)]]
        df_merged.columns = column_bak

        df_merged["lbl"]  = df_merged.loc[:, self.l_lbl[feature_func_name] + [mylbl for mylbl in df_tmp.columns.get_level_values(0) if ("sdiff" in mylbl)]].astype(str).sum(axis=1)
        # exclude all feature difference among l_sda_not_pair
        if l_sda_not_pair is not None:
            for mykey in l_sda_not_pair:
                df_tmp = self.df_feature_diffs[feature_func_name][mykey].drop(["mf", "sf"], level = 1, axis = 1)
                df_tmp[[mylbl for mylbl in df_tmp.columns.get_level_values(0) if ("sdiff" in mylbl)]] = _np.sign(df_tmp[[mylbl for mylbl in df_tmp.columns.get_level_values(0) if ("sdiff" in mylbl)]])
                df_tmp["lbl"] = df_tmp.loc[:, self.l_lbl[feature_func_name] + [mylbl for mylbl in df_tmp.columns.get_level_values(0) if ("sdiff" in mylbl)]].astype(str).sum(axis=1)
                df_merged = df_merged.loc[_np.in1d(df_merged["lbl"], _np.setdiff1d(df_merged["lbl"], df_tmp["lbl"])), :]
        del df_merged["lbl"]
        s_sda_pair = "_and_".join(["_vs_".join(x) for x in l_sda_pair])
        if l_sda_not_pair is not None:
            s_sda_pair += "_not_" + "_and_".join(["_vs_".join(x) for x in l_sda_not_pair])
        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()
        
        try:
            df_merged.to_csv("%s/%s.%s.%s.nsigma_%f.nfunit_%f.dat" % (
                outdir,
                feature_func_name, self.intrajdatatype,
                s_sda_pair,
                self.num_sigma_funit[0],
                self.num_sigma_funit[1]),
                sep="\t", float_format="%.4f", index=False)
        except OSError as e:
            if e.errno == 36:
                warnstr = "output file name for common SFDs too long, shortening by renaming s_sda_pair to \"common_SFDs\""
                _warnings.warn(warnstr)
                s_sda_pair = "common_SFDs"
                df_merged.to_csv("%s/%s.%s.%s.nsigma_%f.nfunit_%f.dat" % (
                    outdir,
                    feature_func_name, self.intrajdatatype,
                    s_sda_pair,
                    self.num_sigma_funit[0],
                    self.num_sigma_funit[1]),
                    sep="\t", float_format="%.4f", index=False)
            else:
                raise
            
        return df_merged

    def write_features(self, outdir=None):
        """
        Writes out feature table for all ensembles

        Parameters
        ----------
        * outdir : string, optional, default = "output/meta/%s/%s" % (self.feature_func_name,
                                                                      self.intrajdatatype)
                   output path
        """

        if self.feature_func_name not in self.df_features:
            raise ValueError("ERROR: self.df_features[%s] does not exist, run comp_features first!" %
                             self.feature_func_name)
        if outdir is None:
            outdir = "output/meta/%s/%s" % (self.feature_func_name, self.intrajdatatype)

        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()

        s_l_ens = "_".join(self.l_ens)
        if len(s_l_ens) > 150:
            s_l_ens = "MD_ensembles"
            warnstr = "output string for l_ens is too long (> 150), renaming to \"MD_ensembles\""
            _warnings.warn(warnstr)

        if self.intrajdatatype in ["bootstraps"]:
            for myind, mydf in enumerate(self.df_features[self.feature_func_name]):
                mydf.to_csv("%s/%s.%s.%s.bs_%05d.dat" % (
                    outdir,
                    self.feature_func_name,
                    self.intrajdatatype,
                    s_l_ens,
                    myind),
                    sep="\t", float_format="%.4f", index=False)
        elif self.intrajdatatype in ["convcheck"]:
            self.df_features[self.feature_func_name].to_csv("%s/%s.%s.mnf_%d.nbst_%d.%s.dat" % (
                    outdir,
                    self.feature_func_name,
                    self.intrajdatatype,
                    self.maxnumframes,
                    self.num_bs,
                    s_l_ens),
                    sep="\t", float_format="%.4f", index=False)

        else:
            self.df_features[self.feature_func_name].to_csv("%s/%s.%s.%s.dat" % (
                outdir,
                self.feature_func_name,
                self.intrajdatatype,
                s_l_ens),
                sep="\t", float_format="%.4f", index=False)
            if self.feature_func_name in self.df_fhists:
                self.df_fhists[self.feature_func_name].to_csv("%s/%s.%s.%s.fhists.dat" % (
                    outdir,
                    self.feature_func_name,
                    self.intrajdatatype,
                    s_l_ens),
                    sep="\t", float_format="%.4f", index=False)


    def plot_feature_hists(self, outdir=None):
        """
        Plots feature histograms for all ensembles

        Parameters
        ----------
        * outdir : string, optional, default = "output/meta/%s/%s" % (self.feature_func_name,
                                                                      self.intrajdatatype)
                   output path
        """

        if self.feature_func_name not in self.df_fhists:
            raise ValueError("ERROR: self.df_fhists[%s] does not exist! Have you run comp_features with correctly defined df_hist_feats?" %
                             self.feature_func_name)
        if self.df_fhists[self.feature_func_name] is None:
            raise ValueError("ERROR: self.df_fhists[%s] is None! Maybe %s does not support feature histogramming?" %
                             (self.feature_func_name, self.feature_func_name))
        if outdir is None:
            outdir = "output/meta/%s/%s" % (self.feature_func_name, self.intrajdatatype)

        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()

        s_l_ens = "_".join(self.l_ens)
        if len(s_l_ens) > 150:
            s_l_ens = "MD_ensembles"
            warnstr = "output string for l_ens is too long (> 150), renaming to \"MD_ensembles\""
            _warnings.warn(warnstr)

        mypdf = "%s/%s.%s.%s.fhists.pdf" % (outdir,
                                            self.feature_func_name,
                                            self.intrajdatatype,
                                            s_l_ens)
        mydf_fhists = self.df_fhists[self.feature_func_name].copy()
        mydf_fhists.columns = _pd.MultiIndex.from_arrays(
                              [mydf_fhists.columns,
                               _np.concatenate(
                              (_np.repeat("", len(self.l_lbl[self.feature_func_name])), 
                               _np.repeat("fhist", len(self.l_ens)))),
                              ])
        mydf_fhists =  mydf_fhists.merge(self.df_features[self.feature_func_name]).set_index(self.l_lbl[self.feature_func_name])
        mydf_fhists.sort_index(axis = 1, inplace = True)
        with _PdfPages(mypdf) as pdf:
            for index, row in mydf_fhists.iterrows():
                #f,(ax1,ax2)=plt.subplots(1, 2, sharey=True,figsize=(15,5))
                _plt.figure()
                #_plt.title("%s\n" % self.feature_func_name + "/".join(["_".join([str(y) for y in x])
                #                                                    for x in [index[:len(index)//2], index[len(index)//2:]]]))
                _plt.title("%s\n" % self.feature_func_name + "_".join([str(x) for x in index]))
                _plt.xlabel("feature value [feature unit]")
                _plt.ylabel("probability density")
                abc = row.loc[row.index.get_level_values(1) == "fhist"]
                dbin = abc.loc[abc != 0][0][0]
                dbin = dbin[1] - dbin[0]
                prec = len(str(dbin).partition(".")[2])+1
                l_col = _plt.get_cmap('rainbow', len(self.l_ens))
                for i in range(len(self.l_ens)):
                    myens = self.l_ens[i]
                    if row[(myens, "fhist")] == 0:
                        row[(myens, "fhist")] = ([0, dbin], [1/dbin], [0])
                    row[(myens, "fhist")] = list(row[(myens, "fhist")])
                    row[(myens, "fhist")][0] = row[(myens, "fhist")][0][:-1] + dbin/2
                    _plt.errorbar(row[(myens, "fhist")][0],
                                  row[(myens, "fhist")][1],
                                  yerr  = row[(myens, "fhist")][2],
                                  label = '{0}: {1:.{2}f}({3:.{4}f})+-{5:.{6}f}'.format(myens,
                                                                row[(myens, "mf")], prec,
                                                                row[(myens, "sf")]*10**prec, 0,
                                                                row[(myens, "sf")], prec),
                                  color = l_col(i))
                    #_plt.vlines(row[(myens, "mf")], 0, 1, color = l_col(i))
                _plt.ylim(0, _plt.ylim()[1])
                for i in range(len(self.l_ens)):
                    myens = self.l_ens[i]
                    _plt.vlines(row[(myens, "mf")], _plt.ylim()[0], (i+1) / 10 / len(self.l_ens)*_plt.ylim()[1], color = l_col(i))
                    #_plt.hlines(0, _plt.xlim()[0], _plt.xlim()[1], color = l_col(i))
                    _plt.errorbar(row[(myens, "mf")], (i+1) / 10 / len(self.l_ens)*_plt.ylim()[1], xerr = row[(myens, "sf")], color = l_col(i))
                _plt.legend()
                pdf.savefig()

    def write_feature_diffs(self, outdir=None):
        """
        Writes out pairwise interaction table for all ensembles

        Parameters
        ----------
        * outdir : string, optional, default = "output/meta/%s/%s" % (self.feature_func_name,
                                                                      self.intrajdatatype)
                   output path
        """

        if self.feature_func_name not in self.df_feature_diffs:
            raise ValueError(
                "ERROR: self.df_feature_diffs[%s] does not exist, run comp_feature_diffs first!" %
                self.feature_func_name)
        if outdir is None:
            outdir = "output/meta/%s/%s" % (self.feature_func_name, self.intrajdatatype)

        _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()
        for mykey in self.df_feature_diffs[self.feature_func_name].keys():
            if isinstance(self.df_feature_diffs[self.feature_func_name][mykey], _pd.DataFrame):
                self.df_feature_diffs[self.feature_func_name][mykey].to_csv(
                    "%s/%s.%s.%s.nsigma_%f.nfunit_%f.dat" % (
                        outdir,
                        self.feature_func_name, self.intrajdatatype, "_vs_".join(mykey),
                        self.num_sigma_funit[0],
                        self.num_sigma_funit[1]),
                    sep="\t", float_format="%.4f", index=False)

    def reload_features(self, feature_func_name=None, intrajdatatype=None, l_ens=None, outdir=None):
        """
        Reloads already computed features table for all ensembles

        Parameters
        ----------
        * feature_func_name     : name of function that computed a particular feature,

        * intrajdatatype : string, default="samplebatches"
            * 'samplebatches'   : trajectories each containing frames sampled from
                                  a stationary distribution of, e.g.,
                                  a trajectory-bootstrapped MSM or a bayesian MSM sample
                                  of bootstrapped frames drawn, e.g., from
                                  a meta-stable set of a Markov State Model
            * 'raw'             : plain simulation trajectories, whose feature statistics 
                                  - means or (means and standard deviations) - 
                                  are further bootstrapped on the trajectory level
                                  (with num_bs bootstraps)
        * l_ens                 : as in _np.transpose(l_ens_numreplica)[0] (see init() above)
                                  if None (default), just reload data from self.l_ens

        * outdir : string, optional, default = "output/meta/%s/%s" % (self.feature_func_name,
                                                                      self.intrajdatatype)
                   output path
        """

        def f(x, y):
            print(x, y)
            raise ValueError("ERROR: feature_func not defined because feature table was reloaded (i.e. not computed)")

        if feature_func_name is not None:
            self._feature_func = f
            self._feature_func_name = feature_func_name
            self.error_type[self._feature_func_name] = feature_func_name.split(".")[-1]
        if intrajdatatype is not None:
            self.intrajdatatype = intrajdatatype
        if l_ens is not None:
            self.l_ens               = l_ens
            self.l_ens_numreplica    = None
        if outdir is None:
            outdir = "output/meta/%s/%s" % (self.feature_func_name, self.intrajdatatype)

        s_l_ens = "_".join(self.l_ens)
        if len(s_l_ens) > 150:
            s_l_ens = "MD_ensembles"
            warnstr = "output string for l_ens is too long (> 150), renaming to \"MD_ensembles\""
            _warnings.warn(warnstr)

        indat = "%s/%s.%s.%s" % (outdir, self.feature_func_name, self.intrajdatatype, s_l_ens)
        df_features = _pd.read_csv(indat+".dat", header=[0,1], sep = "\t")
        df_features.rename(columns = { key: "" for key in [i for i in df_features.columns.get_level_values(1) if "Unnamed: " in i ] }, inplace = True)
        self.l_lbl[self.feature_func_name] = list(df_features.columns.get_level_values(0)[df_features.columns.get_level_values(1)==''])
        self.max_mom_ord[self.feature_func_name] = len(df_features.columns.get_level_values(1).unique())//2
        self.df_features[self.feature_func_name] = df_features

        if _os.path.isfile(indat+".fhists.dat"):
            self.df_fhists[self.feature_func_name] = _pd.read_csv(indat+".fhists.dat", header=[0], sep = "\t")
            self.df_fhists[self.feature_func_name][self.l_ens] = self.df_fhists[self.feature_func_name][self.l_ens].applymap(lambda x: eval(x, None, {'array': _np.array }))

    def view_feature_diffs(self, coarse_grain_type, l_SDApair, l_SDA_not_pair = None, is_only_water_access=None, outdir=None):
        """
        Views (common) significant features in a separate PyMOL session.
        This function uses the IPyMOL module, in essence a PyMOL API.

        Prior to running this function, please make sure to have successfully run
        self.comp_feature_diffs(),
        at which state the following required parameters will have been set:
        - self.feature_func_name
        - self.intrajdatatype
        - self.num_sigma_funit

        Parameters
        ----------
        * coarse_grain_type : string, defining what coarse-graining has been applied to
                              the current feature type, possible values:
                              - None        : no feature coarse-graining used
                                              in the feature type
                              - "cg_nobb"   : coarse-graining on seg-res level only, i.e.
                                              as defined in
                                              scripts/df_rgn_seg_res_bb.dat
                              - "cg_withbb" : coarse-graining on seg-res-bb level, i.e.
                                              as defined in
                                              scripts/df_rgn_seg_res_bb_with_bb.dat

        * l_SDApair, l_SDA_not_pair: two lists defining what ensembles to compare,
                                     see examples below
                                     (if l_SDA_not_pair is None, set l_SDA_not_pair=[] )
            # show significant differences between ensembles 'bN82A.pcca2' and 'WT.pcca2'
            l_SDApair            = [('bN82A.pcca2', 'WT.pcca2')]
            l_SDA_not_pair       = []

            # show significant differences between ensembles
            # 'bN82A.pcca2' and 'WT.pcca2' ...
            l_SDApair            = [('bN82A.pcca2', 'WT.pcca2')]
            # ... which are not significantly different between
            # 'aT41A.pcca1' and 'WT.pcca2':
            l_SDApair            = [('aT41A.pcca1', 'WT.pcca2')]
            
            # show significant differences common to
            # multiple ensemble comparisons ...
            l_SDApair            = [('bN82A.pcca2', 'WT.pcca2'),
                                    ('aT41A.pcca1', 'WT.pcca2')]
            # ... which are not significantly different among
            # the following comparisons
            l_SDA_not_pair            = [('aT41A.pcca1', 'bN82A.pcca2')]

        * is_only_water_access : whether or not to only show significant differences in
                                 interaction frequency differences only with water 
                                 (for spbsf, in particular
                                 spbsf.Hvvdwdist_VMD and spbsf.HvvdwHB)

        * outdir            : string, optional,
                              default = "output/meta" % (feature_func_name,
                                                         self.intrajdatatype)
                              output path (passed in PyMOL, but currently not used)
        """
        from ipymol import viewer as pymol
        import re

        for myvarlbl in ["self.feature_func_name",
                         "self.intrajdatatype",
                         "self.num_sigma_funit"]:
            if eval(myvarlbl) is None:
                raise ValueError("%s is None, i.e. not defined!" % myvarlbl)

        if l_SDA_not_pair is None:
            l_SDA_not_pair = []
        
        d_locals                         = {}
        d_locals["l_SDApair"]            = l_SDApair
        d_locals["l_SDA_not_pair"]       = l_SDA_not_pair
        d_locals["feature_func_name"]    = self.feature_func_name
        d_locals["stattype"]             = self.intrajdatatype                  # e.g., "samplebatches"
        d_locals["nsigma"]               = self.num_sigma_funit[0]              # e.g., 2.000000
        d_locals["nfunit"]               = self.num_sigma_funit[1]              # e.g., 0.000000
        d_locals["intrajformat"]         = self.intrajformat                    # e.g., "xtc"
        d_locals["outdir"]               = outdir                               # 
        d_locals["coarse_grain_type"]    = coarse_grain_type                    # e.g., "cg_nobb"

        # spbsf
        if is_only_water_access is None:
            d_locals["is_only_water_access"] = False
        else:
            d_locals["is_only_water_access"] = is_only_water_access
        
        # Start PyMOL RPC server, if not started yet 
        if not hasattr(pymol, "_server"):
            pymol.start()   # Start PyMOL RPC server

        is_pymol_connected = False
        # keep trying until PyMol server is connected
        while not is_pymol_connected:
            try:
                pymol._server.do("pass")
                is_pymol_connected = True
            except ConnectionRefusedError:
                _time.sleep(1)
       
        pymol._server.do("d_locals = %s" % d_locals)
        pymol._server.do("run VisFeatDiffs/PyMOL/PyMOL_VisFeatDiffs_%s.py" % self.feature_func_name.split(".")[0])


    def featuretype_redundancies(self, l_featuretype = None, corrmethod = "spearman", 
                                 l_radcol = None, withrnmlevel = False, withplots = False, outdir = None ):
        '''
        Detects feature type redundancies via correlations from
        already computed features

        Parameters
        ----------
        * l_featuretype : list of str, list of feature type names
        * corrmethod    : str, name of correlation method
                          'pearson', 'spearman', 'kendall' : see pandas DataFrame.corr() method,
                          'circcorr' : circular correlations, as defined in
                                       S. R. Jammalamadaka and A. SenGupta:
                                       "Topics In Circular Statistics. Series on Multivariate Analysis",
                                       World Scientific (2001),
                                       see also:
                                       https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Circular_Data_Correlation.pdf
                                       (is slow for withrnmlevel == True)
                          default: 'spearman'
        * l_radcol      : list of radial feature types
                          must be defined, if corrmethod != 'corccorr'
        * withrnmlevel  : bool, whether or not to compute feature type correlations also for 
                          individual residue names
                          currently implemented for single residual features (SRF)
        * withplots     : bool, whether or not to plot feature type correlations
        * outdir        : string, optional, default = "output/figures/feature_type_correlations"
                          output path
                   
        Returns:
        -----------
        * rnm2df_feature_corr : dictionary of pandas.DataFrame objects; keys:
                                "all" :             feature type correlations overi
                                                    all residue types and ensemble averages
                                "ALA", "VAL", ... : feature type correlations over
                                                    all ensemble averages for each
                                                    residue name, if withrnmlevel == True
        * df_mean_features    : pandas.DataFrame of mean features
                                (can be used for further processing, e.g.,
                                to plot different feature types against each other)
        '''

        # output directory for feature type correlation plots
        if outdir is None:
            outdir = "output/figures/feature_type_correlations"

        def circcorr(df):
            ''' 
            circular correlation, as defined in 
            S. R. Jammalamadaka and A. SenGupta:
            "Topics In Circular Statistics. Series on Multivariate Analysis",
            World Scientific (2001)
            see also:
            https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Circular_Data_Correlation.pdf
            '''
            df = df.dropna()
            x = df.iloc[:,0]
            y = df.iloc[:,1]
            dsinx = _np.sin(x - _scipy_stats.circmean(x, low = -_np.pi, high = _np.pi))
            dsiny = _np.sin(y - _scipy_stats.circmean(y, low = -_np.pi, high = _np.pi))
            return _np.sum(dsinx * dsiny) / _np.sqrt(_np.sum(dsinx**2) * _np.sum(dsiny**2))

        if l_featuretype is None:
            l_featuretype = list(self.df_features.keys())
        if corrmethod != "circcorr":
            if l_radcol is None:
                raise ValueError("need definition of l_radcol because corrmethod != 'circcorr'")        
        if withplots:
            _subprocess.Popen(_shlex.split("mkdir -p %s" % outdir)).wait()

        df_mean_features    = None
        rnm2df_feature_corr = {}
        # gather featuretype data frames
        l_shortfeaturetype = []
        for myfeaturetype in l_featuretype:
            df_myfeature = self.df_features[myfeaturetype].copy()
            df_myfeature.set_index(self.l_lbl[myfeaturetype], inplace = True)
            df_myfeature.drop('sf', axis=1, level=1, inplace = True)
            df_myfeature.columns = df_myfeature.columns.droplevel(1)
            # display(df_myfeature)
            # seg res rnm ensemble1 ensemble2 ensemble3 ...
            # ...
            df_myfeature = _pd.DataFrame(df_myfeature.stack()).reset_index()
            df_myfeature.rename(columns = {'level_3' : 'ens', 0 : myfeaturetype.split(".")[1]}, inplace = True)
            l_shortfeaturetype.append(myfeaturetype.split(".")[1])
            # display(df_myfeature.head())
            # seg res rnm ens feature (e.g., "srf.RSASA_sr.std_err" -> "RSASA_sr")
            # ...
            if df_mean_features is None:
                df_mean_features = df_myfeature.copy()
            else:
                df_mean_features = df_mean_features.merge(df_myfeature, how = "outer")
        
        # display(df_mean_features.head())
        if corrmethod != 'circcorr':
            # adjust radian values for minimum variance (to deal with periodic boundaries of radians)
            for mycol in l_radcol:
                df_tmp = (df_mean_features[mycol] + 2 * _np.pi) % (2 * _np.pi)
                if df_mean_features[mycol].var(skipna = True) > df_tmp.var(skipna = True):
                    df_mean_features[mycol] = df_tmp

        # compute correlations between feature types along (feature) and (ensemble average)
        if corrmethod == 'circcorr':
            df_tmp = df_mean_features.drop(self.l_lbl[myfeaturetype] + ["ens"], axis = 1)
            df_tmp = _pd.DataFrame([[circcorr(df_tmp[[i,j]]) for i in df_tmp] for j in df_tmp], 
                                   columns = df_tmp.columns, index = df_tmp.columns)
        else:
            df_tmp = df_mean_features.drop(self.l_lbl[myfeaturetype] + ["ens"], axis = 1).corr(method = corrmethod, 
                                                                                               min_periods = 1)
        df_tmp = _pd.DataFrame(df_tmp.stack()).reset_index()
        df_tmp.rename(columns = {'level_0' : 'ftype1', 'level_1' : 'ftype2', 0 : "corr"}, inplace = True)
        df_tmp.sort_values(by = ['ftype1', 'ftype2'], inplace = True)
        df_tmp = df_tmp.set_index(['ftype1', 'ftype2'])
        df_tmp.columns = [""]
        df_tmp.index.names = ["", ""]
        rnm2df_feature_corr["all"] = df_tmp.unstack()
        rnm2df_feature_corr["all"].columns = rnm2df_feature_corr["all"].columns.droplevel(level = 0)
        rnm2df_feature_corr["all"] = rnm2df_feature_corr["all"].loc[l_shortfeaturetype, l_shortfeaturetype]
        l_rnm = []
        if withrnmlevel:
            # compute correlations between feature types for each (feature) along (ensemble average)
            if corrmethod == 'circcorr':
                def myfunc(df_tmp):
                    df_tmp = df_tmp.drop(mySDA.l_lbl[myfeaturetype] + ["ens"], axis = 1)
                    return _pd.DataFrame([[circcorr(df_tmp[[i,j]]) for i in df_tmp] for j in df_tmp], 
                                           columns = df_tmp.columns, index = df_tmp.columns)
                df_feature_corr = df_mean_features.groupby(self.l_lbl[myfeaturetype]).apply(myfunc)
            else:
                df_feature_corr = df_mean_features.groupby(self.l_lbl[myfeaturetype]).corr(method = corrmethod, min_periods = 1)

            df_feature_corr = _pd.DataFrame(df_feature_corr.stack()).reset_index()
            df_feature_corr.rename(columns = {'level_3' : 'ftype1', 'level_4' : 'ftype2', 0 : "corr"}, inplace = True)
            rnm2df_feature_corr["byres"] = df_feature_corr.groupby(['rnm', 'ftype1', 'ftype2'])["corr"].agg("mean").unstack()
            rnm2df_feature_corr["byres"].columns.name = ""
            rnm2df_feature_corr["byres"].index.name   = ""
            l_rnm = list(df_mean_features.rnm.unique())

        for myrnm in ["all"] + l_rnm:
            if myrnm == "all":
                mydf_feature_corr = rnm2df_feature_corr[myrnm]
            else:
                mydf_feature_corr = rnm2df_feature_corr["byres"].loc[myrnm, :]
                df_tmp = rnm2df_feature_corr["all"].copy()
                df_tmp[:] = _np.float("NaN")
                df_tmp.loc[mydf_feature_corr.index, mydf_feature_corr.columns] = mydf_feature_corr
                mydf_feature_corr = df_tmp
                rnm2df_feature_corr[myrnm] = mydf_feature_corr.copy()

            if withplots:
                fig = _plt.figure()
                _sns.heatmap(mydf_feature_corr, cmap = _plt.get_cmap("bwr"), vmin = -1, vmax = 1,
                             cbar_kws={"ticks" : _np.linspace(-1, 1, 9)}, annot=True, fmt=".1f")
                _plt.yticks(_np.arange(0.5, len(mydf_feature_corr.index), 1), mydf_feature_corr.index)
                _plt.xticks(_np.arange(0.5, len(mydf_feature_corr.columns), 1),
                           mydf_feature_corr.columns, rotation=90)
                _plt.title(myrnm)
                _plt.savefig("%s/feature_type_correlation.%s.srf.%s.pdf" % (outdir,
                                                                            corrmethod,
                                                                            myrnm))
        return rnm2df_feature_corr, df_mean_features
