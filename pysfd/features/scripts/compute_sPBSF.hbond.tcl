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
# Computes VMD Hydrogen Bonds
#
# Standard hbond cutoffs in VMD
#set cutoff(dist) 3.0
#set cutoff(angle) 20

set indir	  [lindex $argv 0]
set instem	  [lindex $argv 1]
set intype	  [lindex $argv 2]   
set outdir	  [lindex $argv 3]
set cutoff(dist)  [lindex $argv 4]
set cutoff(angle) [lindex $argv 5]

if { $intype == "xtc" } {
	mol load pdb $indir/$instem.pdb
	animate delete all
	animate read xtc $indir/$instem.xtc waitfor all
} elseif { $intype == "dcd" } {
	mol load pdb $indir/$instem.pdb
	animate delete all
	animate read dcd $indir/$instem.dcd waitfor all
} else {
	puts "ERROR: Unknown/missing intype variable"
	exit
}

puts "numframes: [molinfo top get numframes]"
set totframes [molinfo top get numframes] 

set mysel(all) [atomselect top all]
foreach myinfo { chain segid resid residue resname backbone } {
	set mysel($myinfo) [$mysel(all) get $myinfo]
}

set outfile [open "$outdir/$instem.sPBSF.hbond.vmd.dat" w]
set sel1 [atomselect top "protein"]
puts "totframes $totframes"
for {set f 0} {$f<$totframes} {incr f} {
	$sel1 frame $f
	set l_contact {}
	set mycontacts [measure hbonds $cutoff(dist) $cutoff(angle) $sel1]
	puts "len mycontacts [llength [lindex $mycontacts 0]]"
	for {set k 0} {$k<[llength [lindex $mycontacts 0]]} {incr k} {
		#lappend l_contact [format "%05d %s %s %3d %d %4s %4s %3d %d" $f [lindex $mysel(chain) [lindex $mycontacts 0 $k]] [lindex $mysel(resname) [lindex $mycontacts 0 $k]] [lindex $mysel(resid) [lindex $mycontacts 0 $k]] [lindex $mysel(backbone) [lindex $mycontacts 0 $k]] [lindex $mysel(chain) [lindex $mycontacts 1 $k]] [lindex $mysel(resname) [lindex $mycontacts 1 $k]] [lindex $mysel(resid) [lindex $mycontacts 1 $k]] [lindex $mysel(backbone) [lindex $mycontacts 1 $k]]]
		lappend l_contact [format "%d %s %d %s %d %s %d %s %d" $f [lindex $mysel(chain) [lindex $mycontacts 0 $k]] [lindex $mysel(resid) [lindex $mycontacts 0 $k]] [lindex $mysel(resname) [lindex $mycontacts 0 $k]] [lindex $mysel(backbone) [lindex $mycontacts 0 $k]] [lindex $mysel(chain) [lindex $mycontacts 1 $k]] [lindex $mysel(resid) [lindex $mycontacts 1 $k]] [lindex $mysel(resname) [lindex $mycontacts 1 $k]] [lindex $mysel(backbone) [lindex $mycontacts 1 $k]]]
	}
	puts $outfile [join [lsort -unique $l_contact] "\n"]
}
close $outfile

exit
