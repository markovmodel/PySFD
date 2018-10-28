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

#
# PyMOL_VisFeatDiffs
#
from pymol.cgo import *

import numpy  as np
import pandas as pd

class PyMOL_VisFeatDiffs(object):
    '''
    Visualizes Significant Feature Differences that are
    common among the pairs of simulated ensembles defined in l_SDApair, but that are
    not significantly different among all pairs defined in l_SDA_not_pair

    Parameters
    ----------
    * l_SDApair            : list of 2-d tuples of str
                             pairs of simulated ensembles compaired by
                             comp_feature_diffs
    * l_SDA_not_pair       : list of 2-d tuples of str
                             pairs of simulated ensembles compaired by
                             comp_feature_diffs
    * feature_func_name    : name of the feature function used, e.g. "srf.chi1.std_err"
    * stattype             : type of statistics, i.e. either "samplebatches" or "raw"
    * num_sigma            : float, level of statistical significance, measured in multiples of standard errors
    * num_funits           : float, level of biological significance, measured in multiples of feature units
                             (Note: significance is defined by both statistical AND biological significance !)
    * intrajformat         : is the trajectory format
    * df_rgn_seg_res_bb    : pandas DataFrame (default: None) that defines
                             regions by segIDs and resIDs for coarse-grained results, e.g.:
      df_rgn_seg_res_bb = pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                     'seg' : ["A", "A", "B", "B", "C"],
                                     'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                             if None, just use non-coarse-grained results
    * outdir               : output directory path (not needed right now)

    * myview               : specific PyMOL view to save visualizations, e.g.,
                             "\
                               -0.635461390,    0.528695464,   -0.562786579,\
                               -0.764068067,   -0.535845578,    0.359358341,\
                               -0.111567713,    0.658326149,    0.744472742,\
                                0.000470890,    0.000115473,  -42.329708099,\
                                0.563536644,   -9.008704185,   -3.723406792,\
                             -116.611633301,  200.864410400,    0.000000000"
    '''

    def __init__(self, l_SDApair, l_not_SDApair, feature_func_name, stattype, nsigma, nfunit, intrajformat, df_rgn_seg_res_bb=None, VisFeatDiffsDir=None, outdir=None, myview=None):
        self.l_SDApair         = l_SDApair
        self.l_not_SDApair     = l_not_SDApair
        self.feature_func_name = feature_func_name
        self.stattype          = stattype 
        self.nsigma            = nsigma
        self.nfunit            = nfunit
        self.intrajformat      = intrajformat
        self.df_rgn_seg_res_bb = df_rgn_seg_res_bb
        self.VisFeatDiffsDir   = VisFeatDiffsDir
        self.outdir            = outdir

        if (df_rgn_seg_res_bb is not None) and (not isinstance(df_rgn_seg_res_bb, pd.DataFrame)):
            print(df_rgn_seg_res_bb)
            raise ValueError("df_rgn_seg_res_bb has to be either None or a pandas DataFrame!")
        if myview is None:
            self.myview  = "\
                      0.652665138,    0.205351248,   -0.729285538,\
                      0.711609006,    0.164262310,    0.683099866,\
                      0.260070741,   -0.964803815,   -0.038922187,\
                     -0.000003442,   -0.000028193, -158.983245850,\
                     -1.527315855,    1.568387151,  -11.586393356,\
                    104.869468689,  213.097259521,   20.000000000"
        else:
            self.myview = myview
        # coloring for feature difference visualizations
        self.l_r         = [1.0,0.0]
        self.l_g         = [0.0,0.0]
        self.l_b         = [0.0,1.0]
        self.sgn2col     = ["less","more"]

    def _add_vis(row):
        raise ValueError("Don't run _add_vis() from the parent class!")

    def vis_feature_diffs( self ):
        # define colors for bars indicating feature differences
        cmd.set_color("less", [self.l_r[0], self.l_g[0], self.l_b[0]])
        cmd.set_color("more", [self.l_r[1], self.l_g[1], self.l_b[1]])
    
        # prepare display
        cmd.bg_color("white")
        cmd.set("defer_builds_mode", 3)
        cmd.set("async_builds", 1)
        cmd.set("cartoon_cylindrical_helices", 0)
        cmd.set("ray_shadow", 0)
        cmd.set("ray_opaque_background", "on")
        cmd.set("valence", 0)
        cmd.do("run %s/center_of_mass.py" % self.VisFeatDiffsDir)

        # load in diff data
        s_SDApairs      = "_and_".join(["_vs_".join(x) for x in l_SDApair])
        s_SDA_not_pairs = "_and_".join(["_vs_".join(x) for x in l_SDA_not_pair])
        if self.df_rgn_seg_res_bb is None:
            cmd.set("cartoon_color", "white")
        instem="output/meta/%s/%s" % (feature_func_name, stattype)
        if (len(l_SDApair)>1) or (len(l_SDA_not_pair)>0):
            instem += "/common"
        if (len(l_SDA_not_pair)>0):
            s_SDApairs += "_not_" + s_SDA_not_pairs  
        infilename="%s/%s.%s.%s.nsigma_%.6f.nfunit_%.6f.dat" % (instem,
                                                                feature_func_name,
                                                                stattype,
                                                                s_SDApairs,
                                                                nsigma,
                                                                nfunit)
        with open(infilename) as infile:
            l_lbl1  = next(infile).split()
            l_lbl2  = next(infile).split()
        numdifflbls = len([x for x in l_lbl1 if x in ["sdiff", "score", "pval"]])
        numenscols  = len(l_lbl2) // 2
        numlblcols  = len(l_lbl1) - 2 * numenscols - numdifflbls
        newcols = l_lbl1[:numlblcols] + l_lbl1[numlblcols:-numdifflbls:2] + l_lbl1[-numdifflbls:]
        df_features = pd.read_csv(infilename, skiprows = 2, header=None, delim_whitespace = True)
        df_features.columns = pd.MultiIndex(levels=[newcols, ['', 'mf', 'sf']],
                           labels=[list(range(numlblcols)) + [(numlblcols + i) for i in range(numenscols) for j in range(2)] + list(range(len(newcols)-numdifflbls, len(newcols))),
                                  numlblcols * [0] +  numenscols * [1,2] + numdifflbls * [0] ])
        df_features.drop(["mf", "sf"], level = 1, axis = 1, inplace=True)
        df_features.columns = df_features.columns.droplevel(1)
        
        # load reference structure for structural alignments
        l_mol    = np.unique([y for x in l_SDApair + l_SDA_not_pair for y in x])
        refmol   = "ref.%s" % l_mol[0]
        cmd.load("input/%s/r_00000/%s.r_00000.prot.pdb" % (l_mol[0], l_mol[0]), refmol)
    
        # load in conformations and add visualizations
        for mymol in l_mol:
            cmd.load("input/%s/r_00000/%s.r_00000.prot.pdb" % (mymol, mymol), mymol)
            cmd.load_traj("input/%s/r_00000/%s.r_00000.prot.%s" % (mymol, mymol, intrajformat), object=mymol, interval=1, start=1, stop=1, state=1)
            cmd.cealign(refmol, mymol)
    
            cmd.hide("everything",mymol)
            cmd.show("cartoon"  ,"/%s" % (mymol))
            cmd.color("white"   ,"/%s" % (mymol))
    
            l_cgo = []
            for index, row in df_features.iterrows():
                if self._add_vis(mymol, row, l_cgo) is None:
                    continue
            if len(l_cgo) != 0:
                cmd.load_cgo(l_cgo,"%s.diffs" % (mymol))
        cmd.delete(refmol)
        if self.myview is not None:
            cmd.set_view(self.myview)
        #cmd.ray()
        #cmd.save("" % (outdir))


