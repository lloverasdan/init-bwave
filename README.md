# Procedure for running idealized baroclinic-wave simulations in WRF
## Methodology from Lloveras and Durran (2023)

### Step 1 - Download and compile the numerical model:
⋅⋅* WRF version 3.6.1: https://www2.mmm.ucar.edu/wrf/users/download/get_sources.html

### Step 2 - Zonally uniform jet:
⋅⋅* Run "make_jet.py" with parameter options at beginning of script
⋅⋅* Creates netCDF file "wrfin_start" containing input fields for WRF

### Step 3 - Run zonally uniform jet through WRF:
⋅⋅* Copy "wrfin_start" (rename to "wrfinput_d01") into WRF directory
⋅⋅* Run WRF (wrf.exe) for 36 hours
⋅⋅* Use "namelist.input_init" when running (rename to "namelist.input")
⋅⋅* Creates netCDF file (rename to "wrfout_osc")

### Step 4 - Average over oscillations:
⋅⋅* Copy "wrfin_start" into new file "wrfin_avg"
⋅⋅* Run "avg_oscillations.py"
⋅⋅* Script averages fields in "wrfout_osc" and places new fields into "wrfin_avg"
⋅⋅* Repeat steps 3+4 as many times as desired to obtain steady jet

### Step 5 - Add cyclogenetic perturbations and/or moisture:
⋅⋅* Copy "wrfin_avg" into new file "wrfin"
⋅⋅* Run "add_qgpv_moist.py" with parameter options at beginning of script
⋅⋅* Script computes new fields and places them into "wrfin"

### Step 6 - Run WRF:
⋅⋅* Copy "wrfin"(rename to "wrfinput_d01") into WRF directory
⋅⋅* Run WRF (wrf.exe) for desired length
⋅⋅* Use "namelist.input_dry_20km" for dry runs
⋅⋅* Use "namelist.input_moist_20km" for moist runs with parameterized convection
⋅⋅* Use "namelist.input_moist_4km" for moist runs with explicit convection
⋅⋅* Rename namelist file to "namelist.input"
⋅⋅* Creates netCDF file (rename to "wrfout")

### Optional: To obtain initial conditions for runs with finer grid spacing, run "interp_20_to_4km.py"

### Python packages required:
NumPy: https://numpy.org/
netCDF4-Python: https://unidata.github.io/netcdf4-python/
Ambiance: https://pypi.org/project/ambiance/
WRF-Python: https://wrf-python.readthedocs.io/en/latest/
Numba: https://numba.pydata.org/
