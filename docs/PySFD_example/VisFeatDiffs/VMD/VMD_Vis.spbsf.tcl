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
# pbsi (pairwise backbone sidechain interactions):
# colored lines&spheres represent PBSIs that are more/less frequent,
# respectively, in the reference ensemble (l_ref) than
# in the comparison (l_cmp) ensembles
#

#set VisFeatDiffsDir	$::env(PYSFDPATH)/VisFeatDiffs
set featuregroup	spbsf
set featuretype		HBond_VMD
#set featuretype		Hvvdwdist_VMD
set is_coarse           False
#set is_coarse           True
set stdtype		std_err
#set stdtype		std_dev
set statstype		samplebatches
#set nsigma		2.000000
set nsigma		2.000000
set nfunit		0.000000

# show significant differences for single ensemble comparison
set l_cmp		[list bN82A.pcca2]
#set l_cmp		[list aT41A.pcca1]
set l_ref		[list WT.pcca2]
#set l_not_cmp		[list aT41A.pcca2]
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

# show only spbsf differences with water?
set is_only_water_access False

set linewidth 6
set sphererad 1
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
	global is_only_water_access
	global sign2col
	global linewidth
	global sphererad
	global rgn2segresbb
	global is_coarse
	upvar 1 l_ l_

	set bb2selstr(0) "sidechain"
	set bb2selstr(1) "backbone"

	for { set i 0 } { $i < [llength $l_(sdiff,)] } { incr i } {
		if { $is_only_water_access == "True" } {
			if { ([lindex $l_(rnm1,) $i] != "WAT") && ([lindex $l_(rnm2,) $i] != "WAT") } {
				continue
			}
		}
		if { $is_coarse == "True" } {
			set l_segresbb(1) [split $rgn2segresbb([lindex $l_(rgn1,) $i]) "+"]
			set l_segresbb(2) [split $rgn2segresbb([lindex $l_(rgn2,) $i]) "+"]
			foreach selind { 1 2 } {
	                        set selstr($selind) "noh and ("
        	                foreach segresbb $l_segresbb($selind) {
					set segresbb [split $segresbb ","]
					set seg [lindex $segresbb 0]
					set res [lindex $segresbb 1]
					if { [llength $segresbb] == 3 } {
					        set bb [lindex $segresbb 2]
					        set selstr($selind) "$selstr($selind) (chain $seg and resid $res and $bb2selstr($bb)) or"
					} else {
					        set selstr($selind) "$selstr($selind) (chain $seg and resid $res) or"
					}
				}
				set selstr($selind) "[string range $selstr($selind) 0 end-3])"
			}
			# notation:
			# "donsel" = donor atom selection, "accsel" = acceptor atom selection, i.e. for H-Bonds,
			# the first interaction partner is the donor, and the second is the acceptor.
			# for regular contacts, this notation is meaningless and can be ignored
		        set donsel [atomselect top $selstr(1)]
		        set accsel [atomselect top $selstr(2)]
			
			graphics top color $sign2col([sgn [lindex $l_(sdiff,) $i]])
			if {[$donsel num]==0} {
				if {[$accsel num]==0} {
					puts "ERROR: both \[\$donsel num\]==0 and \[\$accsel num\]==0 ! donselstr=[$donsel text],accselstr=[$accsel text]"
				} else {
					draw sphere [measure center $accsel] radius $sphererad
				}
			} else {
				if {[$accsel num]==0} {
					draw sphere [measure center $donsel] radius $sphererad
				} else {
					draw line [measure center $donsel] [measure center $accsel] width $linewidth
				}
			}
			$donsel delete
			$accsel delete
		} else {
			# collect ID info for involved residue to be shown explicitly
			if { [info exists seg2res([lindex $l_(seg1,) $i])] } {
				lappend seg2res([lindex $l_(seg1,) $i]) [lindex $l_(res1,) $i]
			} else {
				set seg2res([lindex $l_(seg1,) $i]) [list [lindex $l_(res1,) $i] ]
			}
			if { [info exists seg2res([lindex $l_(seg2,) $i])] } {
				lappend seg2res([lindex $l_(seg2,) $i]) [lindex $l_(res2,) $i]
			} else {
				set seg2res([lindex $l_(seg2,) $i]) [list [lindex $l_(res2,) $i] ]
			}
			# notation:
			# "donsel" = donor atom selection, "accsel" = acceptor atom selection, i.e. for H-Bonds,
			# the first interaction partner is the donor, and the second is the acceptor.
			# for regular contacts, this notation is meaningless and can be ignored
			set donsel [atomselect top "noh and chain [lindex $l_(seg1,) $i] and resid [lindex $l_(res1,) $i] and $bb2selstr([lindex $l_(bb1,) $i])"]
			set accsel [atomselect top "noh and chain [lindex $l_(seg2,) $i] and resid [lindex $l_(res2,) $i] and $bb2selstr([lindex $l_(bb2,) $i])"]
		   
			# sets the color for the line representing the significantly different pairwise interaction
			graphics top color $sign2col([sgn [lindex $l_(sdiff,) $i]])
			if {[$donsel num]==0} {
				if {[$accsel num]==0} {
					puts "ERROR: both \[\$donsel num\]==0 and \[\$accsel num\]==0 ! donselstr=[$donsel text],accselstr=[$accsel text]"
				} else {
					draw sphere [measure center $accsel] radius $sphererad
				}
			} else {
				if {[$accsel num]==0} {
					draw sphere [measure center $donsel] radius $sphererad
				} else {
					draw line [measure center $donsel] [measure center $accsel] width $linewidth
				}
			}
			$donsel delete
			$accsel delete
		}
	}
	if { $is_coarse == "False" } {
		if { ! [array exists seg2res] } {
			puts "WARNING: array seg2res does not exist"
		} else {
			foreach myseg [array names seg2res] {
				# mol delrep 3 top
				mol representation Licorice
				mol color Name
				mol selection "noh and chain $myseg and resid $seg2res($myseg)"
				mol material Opaque
				mol addrep top
			}
		}
	}
}

if { ($is_coarse == "True") || ($is_coarse == "False")  } {
	source $VisFeatDiffsDir/VMD/VMD_Vis.tcl
}
