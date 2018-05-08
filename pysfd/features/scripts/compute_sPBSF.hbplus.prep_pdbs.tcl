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

# VMD script to prepare MD trajectories
# as PDB files to be read by the HBPLUS program

set pkg_dir	[lindex $argv 0]
set indir	[lindex $argv 1]
set instem	[lindex $argv 2]
set intrajformat	[lindex $argv 3]   
set outdir	[lindex $argv 4]

if { $intrajformat == "xtc" } {
	mol load pdb $indir/$instem.pdb
	animate delete all
	animate read xtc $indir/$instem.xtc waitfor all
} elseif { $intrajformat == "dcd" } {
	mol load pdb $indir/$instem.pdb
	animate delete all
	animate read dcd $indir/$instem.dcd waitfor all
} else {
	puts "ERROR: Unknown/missing intrajformat variable"
	exit
}

set molID(traj)	[molinfo top]
puts "loaded [molinfo top get numframes] frames"

source $pkg_dir/features/scripts/MD2hbpluspdb.tcl
set mynumframes [molinfo top get numframes] 
set molID(traj) [molinfo top]

set l_nm2 [MD2hbpluspdb]
set mysel [atomselect top "(protein) and (noh or $l_nm2)"]

for {set i 0} {$i<$mynumframes} {incr i} {
	$mysel frame $i
	$mysel writepdb [format "$outdir/$instem\_tmp.%05d.pdb" $i]
}

exit
