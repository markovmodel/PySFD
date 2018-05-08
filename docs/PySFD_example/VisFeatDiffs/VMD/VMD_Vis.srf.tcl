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
# srf (single residue feature):
# colored residues represent SRFs that are significantly
# higher/lower, respectively, in the reference ensemble (l_ref) than
# in the comparison (l_cmp) ensemble
#

#set VisFeatDiffsDir	$::env(PYSFDPATH)/VisFeatDiffs
set featuregroup	srf
#set featuretype		CA_RMSF_VMD
#set featuretype		SASA_sr
set featuretype		chi1
#set featuretype		phi
#set featuretype		SASA_sr
#set is_coarse           True
set is_coarse           False
set stdtype		std_err
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
	global rgn2segresbb
	global is_coarse
	upvar 1 l_ l_

	if { $is_coarse == "True" } {
		for { set i 0 } { $i < [llength $l_(sdiff,)] } { incr i } {
			set l_segresbb(1) [split $rgn2segresbb([lindex $l_(rgn,) $i]) "+"]
			foreach selind { 1 } {
	                        set selstr($selind) "noh and ("
        	                foreach segresbb $l_segresbb($selind) {
					set segresbb [split $segresbb ","]
					set seg [lindex $segresbb 0]
					set res [lindex $segresbb 1]
					set selstr($selind) "$selstr($selind) (chain $seg and resid $res) or"
				}
				set selstr($selind) "[string range $selstr($selind) 0 end-3])"
			}
			set sel  [atomselect top $selstr(1)]
			set sign [sgn [lindex $l_(sdiff,) $i]]
			set segsign "$seg,$sign"
			if { [info exists segsign2selstr($segsign)] } {
				set segsign2selstr($segsign) "$segsign2selstr($segsign) $res"
			} else {
				set segsign2selstr($segsign) "chain $seg and resid $res"
			}
			if {[$sel num]==0} {
				puts "ERROR: \[\$sel num\]==0 ! myselstr=[$sel text]"
			}
			$sel delete
		}
		if { ! [array exists segsign2selstr] } {
			puts "WARNING: array segsign2selstr does not exist in molID [molinfo top] !"
		} else {
			foreach mykey [array names segsign2selstr] {
				set l_key  [split $mykey ","]
				set myseg  [lindex $l_key 0]
				set mysign [lindex $l_key 1]
				mol representation NewCartoon
				mol color ColorID $sign2col($mysign)
				mol selection $segsign2selstr($mykey)
				mol material Opaque
				mol addrep top
			}
		}
	} elseif { $is_coarse == "False" } {
		for { set i 0 } { $i < [llength $l_(sdiff,)] } { incr i } {
			set mysegsign "[lindex $l_(seg,) $i],[sgn [lindex $l_(sdiff,) $i]]"
			set myres [lindex $l_(res,) $i]
			if { [info exists segsign2res($mysegsign)] } {
				lappend segsign2res($mysegsign) $myres
			} else {
				set segsign2res($mysegsign) [list $myres ]
			}
			set mysel [atomselect top "noh and chain [lindex $l_(seg,) $i] and resid $myres"]
			if {[$mysel num]==0} {
				puts "ERROR: \[\$mysel num\]==0 ! myselstr=[$mysel text]"
			}
			$mysel delete
		}
		if { ! [array exists segsign2res] } {
			puts "WARNING: array segsign2res does not exist in molID [molinfo top] !"
		} else {
			foreach mykey [array names segsign2res] {
				set l_key  [split $mykey ","]
				set myseg  [lindex $l_key 0]
				set mysign [lindex $l_key 1]
				mol representation Licorice
				mol color ColorID $sign2col($mysign)
				mol selection "noh and chain $myseg and resid $segsign2res($mykey)"
				mol material Opaque
				mol addrep top
			}
		}
	}
}

if { ($is_coarse == "True") || ($is_coarse == "False")  } {
	source $VisFeatDiffsDir/VMD/VMD_Vis.tcl
}
