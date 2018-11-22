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
# considers contacts between atom listed in $type2rad
# that are more than four residue positions in protein sequence apart
# Pairwise residue interactions are calculated according to the residue
# contact defined previously (22) (i.e. if the distance between any two heavy
# atoms from two residues is smaller than the sum of their van der Waals
# radii plus 0.6 Angstrom, these two residues are considered “in contact”).
# However, we exclude from this calculation contact pairs that are within 4
# residues in sequence, because the van der Waals interactions between a
# residue and its immediate neighbors are not sensitive
# to conformational rearrangements. 
# reference 22. Venkatakrishnan, A. J., Deupi, X., Lebon, G., Tate, C. G., Schertler, G. F., and Babu, M. M. (2013) Molecular signatures of G-protein-coupled re- ceptors. Nature 494, 185–194

set indir	[lindex $argv 0]
set instem	[lindex $argv 1]
set intype	[lindex $argv 2]   
set outdir	[lindex $argv 3]
set subsel 	[lindex $argv 4]

if { $intype == "xtc" } {
	mol load pdb $indir/$instem.pdb
	animate delete all
	animate read xtc $indir/$instem.xtc waitfor all
} elseif { $intype == "dcd" } {
	mol load pdb $indir/$instem.pdb
	animate delete all
	animate read dcd $indir/$instem.dcd waitfor all
} elseif { $intype == "pdb" } {
	mol load pdb $indir/$instem.pdb
} else {
	puts "ERROR: Unknown/missing intype variable"
	exit
}

# used for the MHCII system:
set fitselstr  "name CA and not ((chain A and resid 4 to 78) or (chain B and resid 6 to 86) or (chain C and resid 105 to 117))"
#set fitselstr  "name CA"
#set rmsfselstr "((chain A and resid 4 to 78) or (chain B and resid 6 to 86) or (chain C and resid 105 to 117))"

set refsel [atomselect top $fitselstr frame 0]
set cmpsel [atomselect top $fitselstr]
set movsel [atomselect top all]
for {set i 1} { $i < [molinfo top get numframes] } { incr i } {
	$cmpsel frame $i
	$movsel frame $i
	$movsel move [measure fit $cmpsel $refsel]
}
$refsel delete
$cmpsel delete
$movsel delete

puts "using RMSFs with atom selection string: \"($subsel) and name CA\""
set mysel [atomselect top "($subsel) and name CA"]
foreach myinfo { chain segid resid residue resname backbone } {
	set l_($myinfo) [$mysel get $myinfo]
}
set l_(rmsf) [measure rmsf $mysel]

set outfile [open "$outdir/$instem.caRMSF.vmd.dat" w]
for {set i 0} { $i < [llength $l_(rmsf)] } { incr i } {
	puts $outfile [format "%s %d %s %f" [lindex $l_(chain) $i] [lindex $l_(resid) $i] [lindex $l_(resname) $i] [lindex $l_(rmsf) $i] ]
}
close $outfile

exit
