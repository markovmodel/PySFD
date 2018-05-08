## PySFD - Significant Feature Differences Analyzer for Python

PySFD computes and visualizes any significant differences in features, e.g., distances, contacts, angles, dihedrals, 
among different sets of molecular dynamics-simulated ensembles.

For an overview, pleas read the following documentation.
For further details, please read
- the doc strings throughout the PySFD package
- example jupyter notebook and python files located in `PySFD/docs/PySFD_example`

### Citation
If you find this tool useful, please cite:
```
S. Stolzenberg: "PySFD: Molecular Mechanistic Clues from Significant Features Differences detected among Simulated Ensembles" (submitted)
```

### Installation
PySFD is a Python package available for Python 2 and 3, should in principle run on all common platforms, 
but is currently only supported for Linux and MacOS.

#### Required Packages

First, make sure you have the folllowing python packages installed,
e.g. via Miniconda https://conda.io/miniconda.html
```sh
conda install jupyter numpy pathos pandas biopandas mdtraj scipy matplotlib seaborn
```
to install SHIFTX2 (only works for Python 3):
```sh
conda install -c omnia/label/dev shiftx2
```

For visualization of significant feature differences, please install the latest versions of:

- PyMOL (https://pymol.org), e.g., via
```sh
conda install -c schrodinger pymol
conda install Pmw
```
- VMD (http://www.ks.uiuc.edu/Research/vmd)

#### PySFD

Then just download the PySFD package, e.g., via GitHub
```sh
git clone https://github.com/markovmodel/PySFD.git
```
and from within the downloaded "PySFD" directory,
install PySFD into the Python Path of your environment via:
```sh
python setup.py install
```
To uninstall, simply type:
```sh
pip uninstall pysfd
```

### Features
Pre-defined features are stored in `PySFD.features` and currently include the following modules
and feature classes:

[srf.py] : Single Residue Feature (SRF)
- `CA_RMSF_VMD`:      CA atom root mean square flucatuations (RMSF), computed with VMD
- `ChemicalShift`:    predicted NMR Chemical Shifts, computed via mdtraj and shiftx2
- `Dihedral`:         dihedral angles computed with mdtraj, one of the following methods is passed to the `Dihedral` class:
    - `mdtraj.compute_chi1`
    - `mdtraj.compute_chi2`
    - `mdtraj.compute_chi3`
    - `mdtraj.compute_chi4`
    - `mdtraj.compute_omega`
    - `mdtraj.compute_phi`
    - `mdtraj.compute_psi`
- `Scalar_Coupling`:  scalar couplings computed with mdtraj, one of the following methods is passed to the `Scalar_Coupling` class:
    - `mdtraj.compute_J3_HN_C`
    - `mdtraj.compute_J3_HN_CB`
    - `mdtraj.compute_J3_HN_HA`
- `SASA_sr`  : solvent accessibility surface areas (SASAs) via mdtraj.shrake_rupley
- `RSASA_sr` : relative SASA, i.e. normalized to the total SASA of a particular residue, computed via mdtraj.shrake_rupley

[prf.py] : Pairwise Residual Features (PRF)
- `Ca2Ca_Distance`:       distance between CA atoms
- `CaPos_Correlation`:    (partial) correlations between Ca positions
- `Dihedral_Correlation`: (partial) correlations between Dihedral angles (see srf module)
- `Scalar_Coupling_Correlation`: (partial) correlations between Dihedral angles (see srf module)

[spbsf.py]  : sparse Pairwise Backbone Sidechain Features (sPBSF) (contact frequencies and dwell times)
- `HBond_mdtraj`: hydrogen bonds via mdtraj
- `HBond_VMD`:    hydrogen bonds via VMD
- `HBond_HBPLUS`: hydrogen bonds via HBPLUS
    - link to HBPLUS program:
      https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/
    - after installation, please update "export hbdir=..."
      in PySFD/features/scripts/compute_PI.hbplus.sh
- `Hvvdwdist_VMD`: heavy atom van der Waals radii contacts between residues that are > 4 residue positions apart, computed via VMD
- `HvvdwHB`: contact, if `Hvvdwdist_VMD` or `HBond_HBPLUS` contact in a simulation frame
`Hvvdwdist_VMD` contacts can be transformed into a single residual feature (SRF, see above), e.g., by only considering the contact frequencies of any residual backbone or side-chain with water (`Hvvdwdist_H2O`).

[pprf.py]   : Pairwise Pairwise Residual Features (PPRF)
- `Ca2Ca_Distance_Correlation`: pairwise (partial) correlations between Ca-to-Ca distances

[pspbsf.py] : Pairwise sparse Pairwise Backbone/Sidechain Features (PsPBSF)
- `sPBSF_Correlation`: (partial) correlations between sPBSF features (see spbsf module)

Each of these feature classes contained in each module is derived
from `PySFD.features._feature_agent.FeatureAgent`.
Each such feature class is instantiated and passed as a `FeatureObj` object into
an `PySFD.PySFD` object, which in turn extracts the essential feature function from `FeatureObj` via
`FeatureObj.get_feature_func()`.
These features classes can be exented by adding further feature classes in each existing feature module,
or adding further feature modules (make sure then to update `PySFD/features/__init__.py`).

An important functionality of PySFD is that it allows the histogramming of specific (or all) computed features of a particular type.
The features to be histogrammed are initially listed in the `df_hist_feats` parameter, a pandas.DataFrame passed in a particular feature class (see docstrings).
For each ensemble, the average feature histograms (with standard deviations) over ensemble trajectories are then computed, and stored into the `PySFD.df_fhists` dictionary.

### Feature Coarse-Graining
Each of these features can be further coarse-grained for each simulation frame by "region" into regional features by providing
`df_rgn_seg_res_bb`, a user-defined pandas Dataframe, and
`rgn_agg_func`, user-defined function or string, 
to the specific feature class.

`df_rgn_seg_res_bb` maps from the residual (or sub-residual, i.e. backbone/sidechain) level to the regional level via 
the columns `rgn` (region ID), `seg` (SegID), `res` (ResID), and optionally `bb` (is backbone (1)? is sidechain (0)?).
```python
'''
* df_rgn_seg_res_bb : optional pandas.DataFrame for coarse-graining that defines
                      regions by segIDs and resIDs, and optionally backbone/sidechain, e.g.
  df_rgn_seg_res_bb = _pd.DataFrame({'rgn' : ["a1", "a2", "b1", "b2", "c"],
                                     'seg' : ["A", "A", "B", "B", "C"],
                                     'res' : [range(4,83), range(83,185), range(4,95), range(95,191), range(102,121)]})
                      if None, no coarse-graining is performed
'''
```

`rgn_agg_func` defines the function used in the coarse-graining, e.g. mean, sum, ...
```python
'''
* rgn_agg_func  : function or str for coarse-graining
                  function that defines how to aggregate from residues (backbone/sidechain) to regions in each frame
                  this function uses the coarse-graining mapping defined in `df_rgn_seg_res_bb`
                  - if a function, it has to be vectorized (i.e. able to be used by, e.g., a 1-dimensional numpy array)
                  - if a string, it has to be readable by the aggregate function of a pandas Data.Frame,
                    such as "mean", "std"
'''
```

Each feature class then coarse-grains these features
- in each simulation frame, if feature values are computed by frame
- otherwise over feature values, e.g., if the underlying feature type describes an entire trajectory (correlation coefficients, RMSF, ...)

Each feature type has a default for `rgn_agg_func` (usually "mean" or "sum", see doc strings)

For help on how to semi-automatically define `df_rgn_seg_res_bb`
see:

`PySFD/docs/notes/how_to_define_rgn_to_seg_res_ranges.txt`

For example definitions of `df_rgn_seg_res_bb` data frames, see

`PySFD/docs/PySFD_example/scripts`

### Feature and Significant Feature Differences
These are computed by the following methods (see example jupyter notebooks in `PySFD/docs/PySFD_example`):
1) `PySFD.comp_features()`,
2) `PySFD.comp_feature_diffs()`, or `PySFD.comp_feature_diffs_with_dwells()`, and
3) `PySFD.comp_and_write_common_feature_diffs()`

(further notes below)

Multiple features types and differences can be computed within the same PySFD instance and
are stored, respectively, into the dictionaries
`PySFD.df_features[PySFD.feature_func_name]` and
`PySFD.df_feature_diffs[PySFD.feature_func_name]`, where
`PySFD.feature_func_name` is the name of the
currently selected feature function `PySFD.feature_func`.

Further Notes:

1\. `PySFD.comp_features()`

Features are computed as means over ensemble trajectory means (and optionally higher statistical moments with respect to the mean, see parameter `max_mom_ord` in the docstrings),
i.e. each feature as a mean of the feature's mean (or higher statistical moment with respect to the mean) along each trajectory
(circular/arithmetic means for circular/linear feature types).
The uncertainties of these feature means (and optionally higher moments) can be computed either as

"standard errors"     (`PySFD.error_type[PySFD.feature_func_name] = "std_err"`) (statistical uncertainty)

or

"standard deviations" (`PySFD.error_type[PySFD.feature_func_name] = "std_dev"`) ("exclusive"(/"effect size") uncertainty).

(optional higher moments, i.e. `max_mom_ord>1`, are defined only for "standard errors")

If these uncertainties are computed as "standard errors", then each as
a (circular) standard deviation over ensemble trajectory means.

If these uncertainties are computed as "standard deviations", then each as
a mean of (circular) standard deviations over ensemble trajectories.

`PySFD.error_type[PySFD.feature_func_name]` is implicitly defined in each `FeatureObj` object
passed into PySFD.

(

`PySFD.error_type[PySFD.feature_func_name]` gets updated
in the beginning of `PySFD.run_ens()`:
```python
self.error_type[self._feature_func_name] = dataflags.get("error_type")
```

Alternatively, one could directly define error_type as a universal PySFD.error_type
and make PySFD.PyASDA.feat_func() read it, but then looping through a list of 
feature functions items would require
updating self.error_type for every iteration, e.g. via an external dictionary.

)

Reloading of already computed features (not differences) is invoked via
`PySFD.reload_features()`
and writing features to disk via
`PySFD.write_features()`.

2\. `PySFD.comp_feature_diffs()`, or `PySFD.comp_feature_diffs_with_dwells()`

To determine significant differences among pairs of ensembles, the individual ensemble
uncertainties are added geometrically (to form "sigma", i.e. the sqrt(...) in the manuscript)
and scaled by `num_sigma`.
An individual feature is significantly different between two ensembles, if its mean (optionally a higher moment with respect to the mean) differs in absolute value by
more than both `num_funit` and num_sigma * sigma:

![equation](http://latex.codecogs.com/gif.latex?%7Babs%7D%5Cleft%28%5Cbar%7Bf%7D_%7Ba%7D-%5Cbar%7Bf%7D_%7Bb%7D%5Cright%29-%5Cmax%5Cleft%28n_%7B%5Csigma%7D%5Ccdot%5Csqrt%7B%5CDelta_%7Bf_%7Ba%7D%7D%5E%7B2%7D&plus;%5CDelta_%7Bf_%7Bb%7D%7D%5E%7B2%7D%7D%2Cn_%7Bf%7D%5Cright%29%3E0)

(f_a, f_b could either be a mean or optionally a higher statistical moment with respect to the mean)

If significant differences are computed also in regard to higher statistical moments, then significant differences in the m-th moment (for all 1 < m <= `max_mom_ord`)
are selected, for which all n-th moments (n\<m, "mean" for n=1) are not significantly different.

Significant feature difference (d_s in the manuscript) are written to disk via

`PySFD.write_feature_diffs()`

`PySFD.comp_feature_diffs_with_dwells()` computes significant differences in
sPBSF features *and* dwell times (for this, `PySFD.is_with_dwell_times` must be set to True)
sPBSF dwell times are defined as the average simulation time (in units of frames) a sPBSF spends
in the "on" ("off") state before switching to the "off" ("on") state.
Currently, for sPBSF dwell time computations, the corresponding ensemble trajectories have to be
plain MD simulation trajectories (i.e. time-dependent, *not* "samplebatches" trajectories).

!!!!!!

Currently, no significant difference in dwell time has been observed in real MD trajectories
that cannot already be explained by a significant difference in interaction frequency.

If you are the first to find such a significant difference, please inform the author 
to claim your complimentary beer / chocolate bar ! ;-)

To scan your own MD simulations for such peculiar differences, just type in, e.g.:
```python
mySDA = PySFD(...)
mySDA.comp_features(...)
...
mySDA.comp_feature_diffs_with_dwells(num_sigma=2)
abc = mySDA.df_feature_diffs['pbsi.HBond_VMD.std_err'][('bN82A.pcca1', 'WT.pcca3')]
# significant differences in 'ton' ("on"  dwell time) that is NOT significant in 'f' (interaction frequency)
print(abc[(-1, 1, 0)])
# significant differences in 'tof' ("off" dwell time) that is NOT significant in 'f' (interaction frequency)
print(abc[(-1, 0, 1)])
```
!!!!!!

3\. `PySFD.comp_and_write_common_feature_diffs()`

`PySFD.comp_and_write_common_feature_diffs()` computes and writes out significant feature differences that
are common among different pairwise ensemble comparisons, and *optionally* that are NOT significantly different among
 a different set of pairwise ensemble comparisons. These common differences may indicate general mechanistic elements that trigger
conformational changes in the simulated system.
    
### Multiprocessed Feature Computation
`PySFD.comp_features()` uses a two-layer multiprocessing (using the pathos module)
to compute features. In this multiprocessing, i*j CPU cores on a single node simultaneously
compute features of i ensembles and j trajectories (replica). If the features of more than
i ensembles and/or j replica are to be computed, these features will be computed subsequently on
the respectice level (of i or of j).
In particular, `PySFD.comp_features()` spawns i non-deamon processes each executing
`PySFD.run_ens()`. Each `PySFD.run_ens()` then spawns j deamom processes, each executing
`PySFD.feature_func()`, where the latter is the current feature function.

### Simulation Input
Features are computed from input simulation trajectories that are to be organized
with respect to your current working directory as:

`'input/%s/r_%05d/%s.r_%05d.prot.%s' % (myens, r, myens, r, intrajtype)`

each with a PDB File containing the topology information:

`'input/%s/r_%05d/%s.r_%05d.prot.pdb' % (myens, r, myens, r)`, where
* `myens` is the name of the simulated ensemble
* `r` is the replica index running from 0 to PySFD.num_bs
* `intrajtype` is the trajectory format

These input trajectories can be one of two different types:
```sh
PySFD.intrajdatatype : string, default="samplebatches"
        * 'samplebatches'   : trajectories each containing frames sampled from
                              a stationary distribution of, e.g.,
                              a trajectory-bootstrapped MSM or a bayesian MSM sample
                              of bootstrapped frames drawn, e.g., from
                              a meta-stable set of a Markov State Model
        * 'raw'             : plain simulation trajectories, whose feature statistics 
                              - means or (means and standard deviations) - 
                              are further bootstrapped on the trajectory level
                              (with *num_bs* bootstraps)
```

and one of two different formats:
```sh
PySFD.intrajformat : string, default = "xtc"
        Input trajectory format
        |  "xtc" : gromacs xtc format
        |  "dcd" :         dcd format
```
The user is responsible for the correct naming and numbering of residues and atoms
among different ensembles trajectories (e.g. among different mutants)

### Significant Feature Difference Visualization
(Common) significant feature differences can be visualized with
PyMOL and VMD scripts provided in 
`$PYSFDPATH/VisFeatDiffs`
, where $PYSFDPATH, e.g., can be the path to your GitHub-downloaded (or already locally installed) PySFD directory.
These scripts are executed directly either within a PyMOL or VMD session, respectively,
and read in the (common) significant feature difference tables.
You should copy `$PYSFDPATH/VisFeatDiffs` into your current working directory
(see `PySFD/docs/PySFD_example`).

#### PyMOL
[PyMOLVisFeatDiffs.py]: contains the parent PyMOLVisFeatDiffs class

Each of

[PyMOLVisFeatDiffs_prf.py],

[PyMOLVisFeatDiffs_spbsf.py], and

[PyMOLVisFeatDiffs_srf.py]

contains all the feature-specific parameter values and an `_add_vis()` method that
is executed by `PyMOLVisFeatDiffs.vis_feature_diffs`

These feature-specific python files are executed within PyMOL, e.g., via
```sh
run $CURRENT_WORKDIR/VisFeatDiffs/PyMOL/PyMOLVisFeatDiffs_spbsf.py
```
(also see `VisFeatDiffs/PyMOL/readme.txt`).

Before visualizing coarse-grained significant differences,
the coresponding "seg,res"->"rgn" mappings have to be defined in 
`$CURRENT_WORKDIR/scripts/df_rgn_seg_res_bb.dat`
see
`PySFD/docs/notes/how_to_define_rgn_to_seg_res_ranges.txt`
for help on how to create this file from "df_rgn_seg_res_bb" defined in PySFD

Currently, PyMOL visualizations for significant feature differences are implemented only for: 
- Single Residual Features (SRF)
- Pairwise Residual Features (PRF)
- sparse Pairwise Backbone/Sidechain Features (sPBSF)

#### VMD
[VMD_Vis.tcl] : main VMD Visualizer function `VMD_visualize_feature_diffs`

Each of

[VMD_Vis.prf.tcl],

[VMD_Vis.spbsf.tcl], and

[VMD_Vis.srf.tcl]
contains all the feature-specific parameter values and an `add_vis()` function that
is executed by VMD_visualize_feature_diffs (contained in [VMD_Vis.tcl]).

These feature-specific tcl files are executed within VMD, e.g., via
```tcl
set VisFeatDiffsDir $CURRENT_WORKDIR/VisFeatDiffs
source $VisFeatDiffsDir/VMD/VMD_Vis.spbsf.tcl
```
(also see `VisFeatDiffs/VMD/readme.txt`)

Before visualizing coarse-grained significant differences,
the coresponding "seg,res"->"rgn" mappings have to be defined in 
`$CURRENT_WORKDIR/scripts/rgn2segres.tcl`
see `PySFD/docs/notes/how_to_define_rgn_to_seg_res_ranges.txt`
for help on how to create this file from `df_rgn_seg_res_bb` defined in PySFD

Currently, VMD visualizations for significant feature differences are implemented only for: 
- Single Residual Features (SRF)
- Pairwise Residual Features (PRF)
- sparse Pairwise Backbone/Sidechain Features (sPBSF)

[srf.py]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/features/srf.py>
[prf.py]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/features/prf.py>
[spbsf.py]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/features/spbsf.py>
[pprf.py]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/features/pprf.py>
[pspbsf.py]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/features/pspbsf.py>
[VMD_Vis.tcl]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/VisFeatDiffs/VMD/VMD_Vis.tcl>
[VMD_Vis.prf.tcl]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/VisFeatDiffs/VMD/VMD_Vis.prf.tcl>
[VMD_Vis.spbsf.tcl]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/VisFeatDiffs/VMD/VMD_Vis.spbsf.tcl>
[VMD_Vis.srf.tcl]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/VisFeatDiffs/VMD/VMD_Vis.srf.tcl>
[PyMOLVisFeatDiffs.py]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/VisFeatDiffs/PyMOL/PyMOL_VisFeatDiffs.py>
[PyMOLVisFeatDiffs_prf.py]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/VisFeatDiffs/PyMOL/PyMOL_VisFeatDiffs_prf.py>
[PyMOLVisFeatDiffs_spbsf.py]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/VisFeatDiffs/PyMOL/PyMOL_VisFeatDiffs_spbsf.py>
[PyMOLVisFeatDiffs_srf.py]: <https://github.com/markovmodel/PySFD/blob/master/pysfd/VisFeatDiffs/PyMOL/PyMOL_VisFeatDiffs_srf.py>

