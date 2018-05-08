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

# "CHARMM to HBPLUS-PDB" Atom Name Converter
# HBPLUS-PDB is the PDB format used by HBPLUS as input
#
# works for standard amino-acids only:
# ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
#
proc MD2hbpluspdb { {mol top} } {

# caveat: hydrogens from the N-terminal residues are not kept for HBPLUS input, i.e. HBPLUS will build these hydrogens itself
# note: HBPLUS treats HIS as double-protonated (in CHARMM: HSP)

#
# convert residue names
#
set l_rnm1 { "HSD HSE HSP HIP HIE HID" "GLH" "ASH" "CYX" }
set l_rnm2 {                      HIS  GLU  ASP     CYS  }

foreach rnm1 $l_rnm1 {
	set sel($rnm1) [atomselect top "resname $rnm1"]
}
foreach rnm1 $l_rnm1 rnm2 $l_rnm2 {
	$sel($rnm1) set resname $rnm2
	$sel($rnm1) delete
}

#
# convert atom names for specific residues
#
#
# note: "no change" conversions (e.g. HE->HE) here are to keep the corresponding hydrogens for HBPLUS
set l_rnm {  ARG  ARG  ARG  ARG  ASN  ASN CYS  GLN  GLN ILE LYS LYS LYS SER  ARG HIS HIS THR TRP TYR PRO }
set l_nm1 { HH11 HH12 HH21 HH22 HD21 HD22 HG1 HE21 HE22  CD HZ1 HZ2 HZ3 HG1   HE HD1 HE2 HG1 HE1  HH  1H }
set l_nm2 { 1HH1 2HH1 1HH2 2HH2 1HD2 2HD2  HG 1HE2 2HE2 CD1 1HZ 2HZ 3HZ  HG   HE HD1 HE2 HG1 HE1  HH  1H }

foreach rnm $l_rnm nm1 $l_nm1 {
	#puts "$rnm $nm1"
	set sel($rnm,$nm1) [atomselect top "resname $rnm and name $nm1"]
}
set outselstr ""
foreach rnm $l_rnm nm1 $l_nm1 nm2 $l_nm2 {
	#puts "$rnm $nm1 $nm2"
	$sel($rnm,$nm1) set name $nm2
	$sel($rnm,$nm1) delete
	set outselstr "$outselstr or (resname $rnm and name $nm2)"
}

#
# convert atom names for all residues
#
set l_nm1 { HN OT1 OT2}
set l_nm2 {  H   O OXT}

foreach nm1 $l_nm1 {
	#puts "$nm1"
	set sel($nm1) [atomselect top "name $nm1"]
}
foreach nm1 $l_nm1 nm2 $l_nm2 {
	#puts "$nm1 $nm2"
	$sel($nm1) set name $nm2
	$sel($nm1) delete
}

return "((name $l_nm2) $outselstr)"
}
