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

proc VMD_visualize_feature_diffs { featuregroup featuretype s_coarse stdtype statstype nsigma nfunit l_cmp l_ref l_not_cmp l_not_ref intrajformat outdir } {
	# prepare display
	display cachemode On
	color Display {Background} white
	display projection Orthographic

	# various labels for simulated ensembles
	set l_mol {}
	set s_mol ""
	foreach mycmp $l_cmp myref $l_ref {
		lappend l_mol c.$mycmp r.$myref
		set s_mol "$s_mol\_and_$mycmp\_vs_$myref"
	}
	set l_mol [lsort -unique $l_mol]
	set s_mol [string range $s_mol 5 end]

	set l_not_mol {}
	set s_not_mol ""
	foreach mycmp $l_not_cmp myref $l_not_ref {
		lappend l_not_mol not_c.$mycmp not_r.$myref
		set s_not_mol "$s_not_mol\_and_$mycmp\_vs_$myref"
	}
	set l_not_mol [lsort -unique $l_not_mol]
	set s_not_mol [string range $s_not_mol 5 end]

	# account for "common" feature differences
	if { (([llength $l_cmp] > 1) && ([llength $l_ref] > 1)) || (([llength $l_not_cmp] > 0) && ([llength $l_not_ref] > 0)) } {
		set instem "output/meta/$featuregroup.${featuretype}${s_coarse}.$stdtype/$statstype/common"
		if { [ string length $s_not_mol] > 0 } {
			set infile $instem/$featuregroup.${featuretype}${s_coarse}.$stdtype.$statstype.$s_mol\_not_$s_not_mol.nsigma_$nsigma.nfunit_$nfunit.dat
		} else {
			set infile $instem/$featuregroup.${featuretype}${s_coarse}.$stdtype.$statstype.$s_mol.nsigma_$nsigma.nfunit_$nfunit.dat
		}
	# account for individual feature differences
	} else {
		set instem "output/meta/$featuregroup.${featuretype}${s_coarse}.$stdtype/$statstype"
		set infile $instem/$featuregroup.${featuretype}${s_coarse}.$stdtype.$statstype.$mycmp\_vs_$myref.nsigma_$nsigma.nfunit_$nfunit.dat
	}
        puts "infile: $infile"
        flush stdout

	# load in a frame for each simulated ensemble
	foreach mollbl [concat $l_mol $l_not_mol] {
		if { [string range $mollbl 0 3] == "not_" } {
			set mymol [string range $mollbl 6 end]
		} else {
			set mymol [string range $mollbl 2 end]
		}
		mol load pdb input/$mymol/r_00000/$mymol.r_00000.prot.pdb
		animate delete all
		animate read $intrajformat input/$mymol/r_00000/$mymol.r_00000.prot.$intrajformat beg last end last waitfor all
		mol rename top $mollbl
	
		mol delrep 0 top
		mol representation NewCartoon
		#mol color Chain
		mol color Name
		mol selection all
		mol material Opaque
		mol addrep top
		set molID($mollbl) [molinfo top]
	}
	# structurally align the frames
	set refsel [atomselect $molID([lindex $l_mol 1]) "name CA"]
	for {set i 0} { $i < [llength $l_mol] } { incr i } {
		set cmpsel [atomselect $molID([lindex $l_mol $i]) "name CA"]
		set movsel [atomselect $molID([lindex $l_mol $i]) all]
		$movsel move [measure fit $cmpsel $refsel]
		$cmpsel delete
		$movsel delete
	}
	$refsel delete
	
	#
	# load in PySFD dat file
	#
	set indata [split [exec cat $infile | sed -e "s/ \\+/ /g" -e "s/^ //g" -e "s/\t/ /g"] "\n"]
	set l_lbl1 [split [lindex $indata 0] " "]
	set l_lbl2 [split [lindex $indata 1] " "]
	for {set i 0} {$i<[llength $l_lbl1]} {incr i} {
		set l_([lindex $l_lbl1 $i],[lindex $l_lbl2 $i]) {}
	}
	foreach line [lrange $indata 2 end] {
		for {set i 0} {$i<[llength $l_lbl1]} {incr i} {
			lappend l_([lindex $l_lbl1 $i],[lindex $l_lbl2 $i]) [lindex $line $i]
		}
	}
	if { [llength $l_(sdiff,)] == 0 } {
		puts "WARNING: no feature differences found!"
	}

	# add visualization for each ensemble	
	graphics top delete all
	foreach m [molinfo list]  {
		mol top $m
		add_vis
	}
}

# define cursor keys "a" and "d" to toggle
# between the molecular representations
set c 0
user add key a {
	set l_molid [molinfo list]
	set c [expr max($c-1,0)]
	foreach i $l_molid { mol off $i }
	mol on [lindex $l_molid $c]
	mol top [lindex $l_molid $c]
}
user add key d {
	set l_molid [molinfo list]
	set c [expr min($c+1,[llength $l_molid]-1)]
	foreach i $l_molid { mol off $i }
	mol on [lindex $l_molid $c]
	mol top [lindex $l_molid $c]
}

proc sgn {a} {expr {[string range $a 0 0]=="-" ? -1 : 1 }}
# see also http://wiki.tcl.tk/819#pagetoc42fa2740 for further definitions of sgn

VMD_visualize_feature_diffs $featuregroup $featuretype $s_coarse $stdtype $statstype $nsigma $nfunit $l_cmp $l_ref $l_not_cmp $l_not_ref $intrajformat $outdir
