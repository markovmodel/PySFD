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
# PyMOL_VisFeatDiffs_srf
#
from pymol import cmd, stored
import os
#VisFeatDiffsDir = os.environ['PYSFDPATH'] + "/VisFeatDiffs/PyMOL"
VisFeatDiffsDir = "VisFeatDiffs/PyMOL"
cmd.do("run %s/PyMOL_VisFeatDiffs.py" % (VisFeatDiffsDir))

class PyMOL_VisFeatDiffs_srf(PyMOL_VisFeatDiffs):
    '''
    '''
    __doc__ = PyMOL_VisFeatDiffs.__doc__ + __doc__

    def __init__(self, l_SDApair, l_SDA_not_pair, l_seg, featuregroup, featuretype, stdtype, stattype, nsigma, nfunit, intrajformat, df_rgn_seg_res_bb=None, VisFeatDiffsDir=None, outdir=None, myview=None):
        super(PyMOL_VisFeatDiffs_srf, self).__init__(l_SDApair,
                                             l_SDA_not_pair,
                                             l_seg,
                                             featuregroup,
                                             featuretype,
                                             stdtype,
                                             stattype,
                                             nsigma,
                                             nfunit,
                                             intrajformat,
                                             df_rgn_seg_res_bb,
                                             VisFeatDiffsDir,
                                             outdir,
                                             myview)
 
    def _add_vis(self, mymol, row, l_cgo):
        mycolind = int((np.sign(row.sdiff) + 1) / 2)
        if self.df_rgn_seg_res_bb is None:
            cmd.select("sel","/%s/%s and i. %d and not h." % (mymol, row.seg, row.res))
        else:
            df_unique_rgnsegbb = self.df_rgn_seg_res_bb[["seg"]].drop_duplicates()
            for myrgn in [row.rgn]:
                selstr = "/%s and (" % (mymol)
                for sindex, srow in df_unique_rgnsegbb.iterrows():
                    df_sel = self.df_rgn_seg_res_bb.query("rgn == '%s' and seg == '%s'" % (myrgn, srow.seg))
                    if df_sel.shape[0] > 0:
                        selstr += "(c. %s and i. %s) or" % (srow.seg,
                                                            "+".join([str(x) for x in df_sel.res.values[0]]))
                selstr = selstr[:-3] + ")"
                cmd.select("%s" % myrgn, selstr)
            cmd.set_name("%s" % row.rgn, "sel")
        count = cmd.count_atoms("sel")
        if count == 0:
            if self.df_rgn_seg_res_bb is None:
                raise ValueError("count for %s.%d.%s.%d is zero!" % (row.seg, row.res, row.rnm))
            else:
                raise ValueError("count for %s is zero!" % (row.rgn))
        else:
            if self.df_rgn_seg_res_bb is None:
                cmd.show("sticks", "sel and (sidechain or name ca)")
                cmd.color(self.sgn2col[mycolind], "sel and (sidechain or name ca)")
            else:
                #cmd.show("sticks", "sel")
                cmd.color(self.sgn2col[mycolind], "sel")
              
                
        cmd.delete("sel")

# show significant differences for single ensemble comparison
l_SDApair            = [('bN82A.pcca2', 'WT.pcca2')]
#l_SDA_not_pair       = [('aT41A.pcca1', 'WT.pcca2')]
l_SDA_not_pair       = []

## show significant differences common to multiple ensemble comparisons ...
#l_SDApair            = [('bN82A.pcca2', 'WT.pcc23'), ('aT41A.pcca1', 'WT.pcca2')]
## ... which are not significant different among the following comparisons
#l_SDA_not_pair            = [('aT41A.pcca1', 'bN82A.pcca2')]
##l_SDA_not_pair       = []

l_seg                = ["A", "B", "C"]
featuregroup         = "srf"
#featuretype          = "SASA_sr"
featuretype          = "chi1"
stdtype              = "std_err"
stattype             = "samplebatches"
nsigma               = 2.000000
nfunit               = 0.000000
intrajformat         = "xtc"
outdir               = None

# df_rgn_seg_res_bb is already defined in PyMOL_VisFeatDiffs.py
# if not-coarse-grained data should be visualized, use:
# df_rgn_seg_res_bb = None
#df_rgn_seg_res_bb = None

MyVis = PyMOL_VisFeatDiffs_srf( l_SDApair, l_SDA_not_pair, l_seg, featuregroup, featuretype, stdtype, stattype, nsigma, nfunit, intrajformat, df_rgn_seg_res_bb=df_rgn_seg_res_bb, VisFeatDiffsDir=VisFeatDiffsDir, outdir=outdir, myview=None)
MyVis.vis_feature_diffs()
