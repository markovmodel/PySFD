#!/bin/bash

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
# computes PAIs with timetraces (with plots) for 
# each entire trajectory (mysample=””), or for 
# last samplesize (default:500) frames
# of each trajectory (mysample=” .last_sampled_frames”)
#

pkg_dir=$1
indir=$2
instem=$3
intype=$4
outdir=$5

# define environment variables for HBPLUS
export hbdir=/home/mi/sstolzen/local/stow/hbplus.old
#export hbdir=/home/sstolzen/mypackages/hbplus.old
alias hbplus='$hbdir/hbplus'
alias clean='$hbdir/clean'
alias chkqnh='$hbdir/chkqnh'
alias accall='$hbdir/accall'

echo "*********************"
echo $instem
echo "*********************"

#
# prepare pdb files for HBPLUS
# make sure to update e.g. "myselstr" in the tcl script
#
mkdir -p $outdir/hbplus/tmps
vmd -dispdev text -e $pkg_dir/features/scripts/compute_sPBSF.hbplus.prep_pdbs.tcl -args $pkg_dir $indir $instem $intype $outdir/hbplus/tmps > $outdir/hbplus/tmps/log.compute_sPBSF.hbplus.prep_pdbs.tcl.log

if [ "1" == "1" ]
then
#
# compute hbplus PAIs (pairwise atomic interactions)
#
cd $outdir/hbplus
mkdir -p clndpdbs hbplus
cd tmps
for file in $( ls $instem\_tmp*.pdb ); do
file=(`echo $file | sed -e 's/.pdb//g'`)
echo $file
cd ../clndpdbs
$hbdir/clean &> log.$file.new.log << fin
../tmps/$file.pdb
fin
cd ../hbplus
# note the -R option here which allows to detect aromatic hydrogen bonds
# in general, please read
# http://www.csb.yale.edu/userguides/datamanip/hbplus/command_options.txt
# e.g. to take care of disulphide-bridges etc.
$hbdir/hbplus -R ../clndpdbs/$file.new ../tmps/$file.pdb &> $file.hbplus.log
done
cd ../../../../../../../
fi

##
## compute sPBSF (pairwise backbone/sidechain interaction) timetraces (necessary for computation of sPBSF frequencies,dwell times,correlations,... in later scripts)
##
#~/mypackages/R-2.15.0/bin/R --no-save --args $mymol $outdir < scripts/b1.1.b.sPBSF_timetraces.hbplus.R &> $outdir/log.b1.1.b.sPBSF_timetraces.hbplus.R.log
#outdir=output/$mymol/observables/PRI/hbplus
#mkdir -p $outdir
#~/mypackages/R-2.15.0/bin/R --no-save --args $mymol $outdir < scripts/b1.1.b.PRI_timetraces.hbplus.R &> $outdir/log.b1.1.b.PRI_timetraces.hbplus.R.log

#echo 
#echo "done. If log files look good, please compress"
#echo "output/\$yourdir/observables/sPBSF/hbplus/tmp"  
#echo "output/\$yourdir/observables/sPBSF/hbplus/hbplus"  
#echo "output/\$yourdir/observables/sPBSF/hbplus/clndpdbs"  
#echo "e.g. via"
#echo "tar -czf \$yourdir.tar.gz \$yourdir; rm -r \$yourdir"
#echo "to avoid huge number of files on disk"
#echo

