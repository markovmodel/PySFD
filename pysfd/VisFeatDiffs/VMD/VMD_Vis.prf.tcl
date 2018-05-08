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
# VMD_Vis:
#
# Maps significant feature differences as lines or spheres
# onto the a frame of each simulated ensemble
#
# distance:
# colored lines represent atom-to-atom or region-to-region distances
# that are higher/lower, respectively, in the reference ensemble (l_ref)
# than in the comparison (l_cmp) ensembles
#

#set VisFeatDiffsDir	$::env(PYSFDPATH)/VisFeatDiffs
set featuregroup	prf
set featuretype		distance.Ca2Ca
#set is_coarse           True
set is_coarse           False
set stdtype		std_dev
set statstype		samplebatches
set nsigma		2.000000
set nfunit		0.000000

# show significant differences for single ensemble comparison
set l_cmp		[list bN82A.pcca2]
set l_ref		[list WT.pcca2]
#set l_not_cmp		[list aT41A.pcca1]
#set l_not_ref		[list WT.pcca2]
set l_not_cmp		[list ]
set l_not_ref		[list ]

## show significant differences common to multiple ensemble comparisons ...
#set l_cmp		[list bN82A.pcca2 aT41A.pcca1]
#set l_ref		[list WT.pcca2 WT.pcca2]
## ... which are not significant different among the following comparisons
#set l_not_cmp		[list aT41A.pcca1]
#set l_not_ref		[list bN82A.pcca2]

set intrajformat	xtc
set outdir		output/meta/$featuregroup.$featuretype.$stdtype/$statstype/map2pdb
# color coding for the lines representing significantly different features
set sign2col(-1) 3
set sign2col(1)  7

# distance_atoms specific options:
set linewidth 6
if { $is_coarse == "True" } {
	set s_coarse ".coarse"
	# load in coarse-graining "seg,res"->"rgn" mappings
	source scripts/rgn2segresbb.tcl
} elseif { $is_coarse == "False" } {
	set s_coarse ""
} else {
	puts "ERROR \$is_coarse must be either \"True\" or \"False\"!"
}

proc add_vis { }  {
	global sign2col
	global linewidth
	global rgn2segresbb
	global is_coarse
	upvar 1 l_ l_

	for { set i 0 } { $i < [llength $l_(sdiff,)] } { incr i } {
		if { $is_coarse == "True" } {
			set l_segresbb(1) [split $rgn2segresbb([lindex $l_(rgn1,) $i]) "+"]
			set l_segresbb(2) [split $rgn2segresbb([lindex $l_(rgn2,) $i]) "+"]
			foreach selind { 1 2 } {
	                        set selstr($selind) "noh and ("
        	                foreach segresbb $l_segresbb($selind) {
					set segresbb [split $segresbb ","]
					set seg [lindex $segresbb 0]
					set res [lindex $segresbb 1]
					set selstr($selind) "$selstr($selind) (chain $seg and resid $res and name CA) or"
				}
				set selstr($selind) "[string range $selstr($selind) 0 end-3])"
			}
			set mysel1 [atomselect top $selstr(1)]
			set mysel2 [atomselect top $selstr(2)]
		} elseif { $is_coarse == "False" } {
			set mysel1 [atomselect top "noh and chain [lindex $l_(seg1,) $i] and resid [lindex $l_(res1,) $i] and name CA"]
			set mysel2 [atomselect top "noh and chain [lindex $l_(seg2,) $i] and resid [lindex $l_(res2,) $i] and name CA"]
		}
		# sets the color for the line representing the significantly different pairwise interaction
		graphics top color $sign2col([sgn [lindex $l_(sdiff,) $i]])
		if { ([$mysel1 num]==0) || ([$mysel2 num]==0) } {
			puts "ERROR: either \[\$mysel1 num\]==0 ([$mysel1 num]) or \[\$mysel2 num\]==0 ([$mysel2 num]) mysel1str=[$mysel1 text],mysel2str=[$mysel2 text]"
		} else {
			draw line [measure center $mysel1] [measure center $mysel2] width $linewidth
		}
		$mysel1 delete
		$mysel2 delete
	}
}

if { ($is_coarse == "True") || ($is_coarse == "False")  } {
	source $VisFeatDiffsDir/VMD/VMD_Vis.tcl
}
