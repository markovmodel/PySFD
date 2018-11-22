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
# measure SASA values

set indir	[lindex $argv 0]
set instem	[lindex $argv 1]
set intype	[lindex $argv 2]   
set outdir	[lindex $argv 3]

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

puts "numframes: [molinfo top get numframes]"
set totframes [molinfo top get numframes] 

set allsel [atomselect top all]
set mysel(residue) [lsort -unique [$allsel get residue]]
set outfile [open "$outdir/$instem.rSASA.vmd.dat" w]

#set sel1 [atomselect top "protein"]
puts "totframes $totframes"

foreach myresidue $mysel(residue) {
	set ressel [atomselect top "residue $myresidue"]
	for {set f 0} {$f<$totframes} {incr f} {
		$allsel frame $f
		$ressel frame $f
		set rsasa [measure sasa 1.4 $allsel -restrict $ressel]
		puts $outfile [format "%d %s %s %d %f" $f [lsort -unique [$ressel get chain]] [lsort -unique [$ressel get resname]] [lsort -unique [$ressel get resid]] $rsasa]
	}
	$ressel delete
}
close $outfile

exit
