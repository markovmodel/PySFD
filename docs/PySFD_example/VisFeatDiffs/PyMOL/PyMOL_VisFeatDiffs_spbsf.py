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
# PyMOL_VisFeatDiffs_spbsf
#

from pymol import cmd, stored
import os
#VisFeatDiffsDir = os.environ['PYSFDPATH'] + "/VisFeatDiffs/PyMOL"
VisFeatDiffsDir = "VisFeatDiffs/PyMOL"
cmd.do("run %s/PyMOL_VisFeatDiffs.py" % (VisFeatDiffsDir))

class PyMOL_VisFeatDiffs_spbsf(PyMOL_VisFeatDiffs):
    '''
    * is_only_water_access: bool, if True, only show differences in solvent-accessibility
                            (e.g. for water-accessibility difference with "Hvvdwdist_VMD")
    * linewidth            : float, default 15
                             width of lines representing pairwise feature differences
    * sphererad            : float, default 2
                             sphere radius  representing single   feature differences

    '''
    __doc__ = PyMOL_VisFeatDiffs.__doc__ + __doc__

    def __init__(self, l_SDApair, l_SDA_not_pair, l_seg, featuregroup, featuretype, stdtype, stattype, nsigma, nfunit, intrajformat, df_rgn_seg_res_bb=None, VisFeatDiffsDir=None, outdir=None, is_only_water_access=False, myview=None, linewidth=15, sphererad=2):
        super(PyMOL_VisFeatDiffs_spbsf, self).__init__(l_SDApair,
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
        self.linewidth = linewidth
        self.sphererad = sphererad
        self.is_only_water_access = is_only_water_access
        # whether or not feature difference is with backbone
        # of a residue or not
        self.bb2selstr = { 0 : "not name c+o+n", 1 : "name c+o+n+ca" }
 
    def _add_vis(self, mymol, row, l_cgo):
        if (self.is_only_water_access and (not "WAT" in [row.rnm1, row.rnm2])):
            return None

        mycolind = int((np.sign(row.sdiff) + 1) / 2)
        if self.df_rgn_seg_res_bb is None:
            cmd.select("sel1","/%s/%s and i. %d and %s and not h." % (mymol, row.seg1, row.res1, self.bb2selstr[row.bb1]))
            cmd.select("sel2","/%s/%s and i. %d and %s and not h." % (mymol, row.seg2, row.res2, self.bb2selstr[row.bb2]))
        else:
            if "bb" in self.df_rgn_seg_res_bb:
                df_unique_rgnsegbb = self.df_rgn_seg_res_bb[["seg", "bb"]].drop_duplicates()
                for myrgn in [row.rgn1, row.rgn2]:
                    selstr = "/%s and (" % (mymol)
                    for sindex, srow in df_unique_rgnsegbb.iterrows():
                        df_sel = self.df_rgn_seg_res_bb.query("rgn == '%s' and seg == '%s' and bb == %d" % (myrgn, srow.seg, srow.bb))
                        if df_sel.shape[0] > 0:
                            selstr += "(c. %s and i. %s and %s) or" % (srow.seg,
                                                                       "+".join([str(x) for x in df_sel.res.values[0]]),
                                                                       srow.bb)
                    selstr = selstr[:-3] + ")"
                    cmd.select("%s" % myrgn, selstr)
            else:
                df_unique_rgnsegbb = self.df_rgn_seg_res_bb[["seg"]].drop_duplicates()
                for myrgn in [row.rgn1, row.rgn2]:
                    selstr = "/%s and (" % (mymol)
                    for sindex, srow in df_unique_rgnsegbb.iterrows():
                        df_sel = self.df_rgn_seg_res_bb.query("rgn == '%s' and seg == '%s'" % (row.rgn1, srow.seg))
                        if df_sel.shape[0] > 0:
                            selstr += "(c. %s and i. %s) or" % (srow.seg,
                                                                "+".join([str(x) for x in df_sel.res.values[0]]))
                    selstr = selstr[:-3] + ")"
                    cmd.select("%s" % myrgn, selstr)
            cmd.set_name("%s" % row.rgn1, "sel1")
            cmd.set_name("%s" % row.rgn2, "sel2")
        count1 = cmd.count_atoms("sel1")
        count2 = cmd.count_atoms("sel2")
        if count1 == 0:
            if count2 == 0:
                raise ValueError("counts of both %s.%d.%s.%d and %s.%d.%s.%d are zero!" % (row.seg1, row.res1, row.rnm1, row.bb1, row.seg2, row.res2, row.rnm2, row.bb2))
            else:
                if self.df_rgn_seg_res_bb is None:
                    cmd.show("sticks", "sel2")
                    #cmd.color(self.sgn2col[mycolind], "sel2")
                else:
                    com("sel2",state=1)
                    coord2=cmd.get_model("sel2_COM",state=1).atom[0].coord
                    cmd.delete("sel2_COM")
                    cmd.color(self.sgn2col[mycolind], "sel2")
                    l_cgo += [ \
                        COLOR, self.l_r[mycolind], self.l_g[mycolind], self.l_b[mycolind], \
                        SPHERE,   coord2[0], coord2[1],coord2[2], self.sphererad \
                    ]
        elif count2 == 0:
                if self.df_rgn_seg_res_bb is None:
                    cmd.show("sticks", "sel1")
                    #cmd.color(self.sgn2col[mycolind], "sel1")
                else:
                    com("sel1",state=1)
                    cmd.color(self.sgn2col[mycolind], "sel1")
                    coord1=cmd.get_model("sel1_COM",state=1).atom[0].coord
                    cmd.delete("sel1_COM")
                    l_cgo += [ \
                        COLOR, self.l_r[mycolind], self.l_g[mycolind], self.l_b[mycolind], \
                        SPHERE,   coord1[0], coord1[1],coord1[2], self.sphererad \
                    ]
        else:
            com("sel1",state=1)
            coord1=cmd.get_model("sel1_COM",state=1).atom[0].coord
            cmd.delete("sel1_COM")
            com("sel2",state=1)
            coord2=cmd.get_model("sel2_COM",state=1).atom[0].coord
            cmd.delete("sel2_COM")
            l_cgo += [ \
                LINEWIDTH, self.linewidth, \
                BEGIN, LINES, \
                COLOR, self.l_r[mycolind], self.l_g[mycolind], self.l_b[mycolind], \
                VERTEX,   coord1[0], coord1[1],coord1[2], \
                VERTEX,   coord2[0], coord2[1],coord2[2], \
                END \
            ]
            if self.df_rgn_seg_res_bb is None:
                cmd.show("sticks", "sel1")
                cmd.show("sticks", "sel2")
                util.cbag("sel1")
                util.cbag("sel2")
            else:
                cmd.color(self.sgn2col[mycolind], "sel1")
                cmd.color(self.sgn2col[mycolind], "sel2")
        cmd.delete("sel1")
        cmd.delete("sel2")

## show significant differences for single ensemble comparison
l_SDApair            = [('bN82A.pcca2', 'WT.pcca2')]
#l_SDApair            = [('aT41A.pcca1', 'WT.pcca2')]
l_SDA_not_pair       = []

# show significant differences common to multiple ensemble comparisons ...
#l_SDApair            = [('bN82A.pcca2', 'WT.pcca2'), ('aT41A.pcca1', 'WT.pcca2')]
## ... which are not significant different among the following comparisons
#l_SDA_not_pair            = [('aT41A.pcca1', 'bN82A.pcca2')]
##l_SDA_not_pair       = []

l_seg                = ["A", "B", "C"]
featuregroup         = "spbsf"
featuretype          = "HBond_VMD"
#featuretype          = "Hvvdwdist_VMD"
stdtype              = "std_err"
stattype             = "samplebatches"
nsigma               = 2.000000
nfunit               = 0.000000
intrajformat         = "xtc"
outdir               = None
is_only_water_access = False

# df_rgn_seg_res_bb is already defined in PyMOL_VisFeatDiffs.py
# if not-coarse-grained data should be visualized, use:
# df_rgn_seg_res_bb = None
df_rgn_seg_res_bb = None

MyVis = PyMOL_VisFeatDiffs_spbsf( l_SDApair, l_SDA_not_pair, l_seg, featuregroup, featuretype, stdtype, stattype, nsigma, nfunit, intrajformat, df_rgn_seg_res_bb=df_rgn_seg_res_bb, VisFeatDiffsDir=VisFeatDiffsDir, outdir=outdir, is_only_water_access=is_only_water_access, myview=None)
MyVis.vis_feature_diffs()
