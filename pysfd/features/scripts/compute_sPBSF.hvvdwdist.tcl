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
# considers non-covalent contacts between atom listed in $type2rad
# Pairwise residue contacts are calculated according to the residue
# contact defined previously (see reference below) (i.e. if the distance between any two heavy
# atoms from two residues is smaller than the sum of their van der Waals
# radii plus 0.6 Angstrom, these two residues are considered “in contact”).
# reference: Venkatakrishnan, A. J., Deupi, X., Lebon, G., Tate, C. G., Schertler, G. F., and Babu, M. M. (2013) Molecular signatures of G-protein-coupled re- ceptors. Nature 494, 185–194

set indir	[lindex $argv 0]
set instem	[lindex $argv 1]
set intype	[lindex $argv 2]   
set outdir	[lindex $argv 3]
set s_solv_seg  [lindex $argv 4]
set s_solv_rnm  [string map { "_" " " } [lindex $argv 5]]
set s_anm       [split [lindex $argv 6] "_"]
set s_rad       [split [lindex $argv 7] "_"]

# the following is used to ignore any coordinate with x, y, z == $far_factor
# can be useful in different situtations
set trash_coord  -999.99

#puts $s_solv_rnm
#set s_solv_rnm  "\"Cl-\" \"Na\\+\" WAT"
#puts $s_solv_rnm
#set s_solv_rnm  [lindex $argv 4]

# these are standard van der Waals radii taken from VMD (when loading PDB files) 
# consistent with values listed on www.wikipedia.org
array set type2rad {
"carbon"			1.70 \
"nitrogen"			1.55 \
"oxygen"			1.52 \
"sulfur"			1.80 \
}

## these are default van der Waals radius values when loading in Amber pdb files
#array set type2rad {
#"carbon"			1.50 \
#"nitrogen"			1.40 \
#"oxygen"			1.30 \
#"sulfur"			1.90 \
#}

for {set i 0} {$i < [llength $s_rad] } {incr i} {
	set mynewkey "name [lindex $s_anm $i]"
	set type2rad($mynewkey) [lindex $s_rad $i]
}
set t2rkeys [array names type2rad]
puts $t2rkeys
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

set buffer 0.0
set allsel [atomselect top all]
$allsel set radius 0.0
foreach mykey $t2rkeys {
	set mysel [atomselect top $mykey]
	$mysel set radius [expr $type2rad($mykey)+$buffer]
	$mysel delete
}
set unq_rad [lindex [lsort -real [$allsel get radius]] 0]
if { $unq_rad == 0 } {
	set $mysel [atomselect top "radius 0.0"]
	puts "ERROR: atoms with name [lsort -unqiue [$mysel get name]] have zero vdW radius!"
	$mysel delete
	exit
}
$allsel delete

set bufferdist 0.6

puts "numframes: [molinfo top get numframes]"
set totframes [molinfo top get numframes] 

# treat solvent residues here
set l_solv_rnm [split $s_solv_rnm " "]
set solv_sel   [atomselect top "resname $s_solv_rnm"]
$solv_sel set segid $s_solv_seg
$solv_sel set chain $s_solv_seg
$solv_sel delete
for {set i 0} {$i < [llength $l_solv_rnm]} {incr i} {
	set solv_sel [atomselect top "resname [lindex $l_solv_rnm $i]"]
	$solv_sel set resid $i
	$solv_sel delete
}
foreach mysel [atomselect list] { $mysel delete }
unset mysel

set mysel(all) [atomselect top all]
foreach myinfo { chain segid resid residue resname backbone } {
	set mysel($myinfo) [$mysel(all) get $myinfo]
}

set outfile [open "$outdir/$instem.sPBSF.hvvdwdist.dat" w]
for {set f 0} {$f<$totframes} {incr f} {
	puts "frame $f"
	animate goto $f
	set l_contact {}
	for {set i 0} {$i<[llength $t2rkeys]} {incr i} {
		set sel1 [atomselect top "[lindex $t2rkeys $i] and not (x \"$trash_coord\" and y \"$trash_coord\" and z \"$trash_coord\")"]
		for {set j $i} {$j<[llength $t2rkeys]} {incr j} {
			set sel2 [atomselect top "[lindex $t2rkeys $j] and not (x \"$trash_coord\" and y \"$trash_coord\" and z \"$trash_coord\")"]
			set mycontacts [measure contacts [expr $type2rad([lindex $t2rkeys $i])+$type2rad([lindex $t2rkeys $j])+$bufferdist] $sel1 $sel2]
			for {set k 0} {$k<[llength [lindex $mycontacts 0]]} {incr k} {
				if { [lindex $mysel(residue) [lindex $mycontacts 1 $k]]>[lindex $mysel(residue) [lindex $mycontacts 0 $k]] } {
					lappend l_contact [format "%d %s %d %s %d %s %d %s %d" $f [lindex $mysel(chain) [lindex $mycontacts 0 $k]] [lindex $mysel(resid) [lindex $mycontacts 0 $k]] [lindex $mysel(resname) [lindex $mycontacts 0 $k]] [lindex $mysel(backbone) [lindex $mycontacts 0 $k]] [lindex $mysel(chain) [lindex $mycontacts 1 $k]] [lindex $mysel(resid) [lindex $mycontacts 1 $k]] [lindex $mysel(resname) [lindex $mycontacts 1 $k]] [lindex $mysel(backbone) [lindex $mycontacts 1 $k]]]
				} else {
					lappend l_contact [format "%d %s %d %s %d %s %d %s %d" $f [lindex $mysel(chain) [lindex $mycontacts 1 $k]] [lindex $mysel(resid) [lindex $mycontacts 1 $k]] [lindex $mysel(resname) [lindex $mycontacts 1 $k]] [lindex $mysel(backbone) [lindex $mycontacts 1 $k]] [lindex $mysel(chain) [lindex $mycontacts 0 $k]] [lindex $mysel(resid) [lindex $mycontacts 0 $k]] [lindex $mysel(resname) [lindex $mycontacts 0 $k]] [lindex $mysel(backbone) [lindex $mycontacts 0 $k]]]
				}
			}
			$sel2 delete
		}
		$sel1 delete
	}
	puts $outfile [join [lsort -unique $l_contact] "\n"]
}
close $outfile

exit
