"""
Written by Daniel J. Lloveras
"""

### Modules
import numpy as np
import netCDF4 as nc
from bwave_ideal_wrf import epv_jet, write_wrfinputd01

### Constants
G = 9.81  # gravitational acceleration, in m/s^2
T0 = 300.  # reference potential temperature, in K
P0 = 1.0e5  # reference pressure, in Pa
CP = 1004.  # specific heat at constant pressure, in J/(K*kg)       
CV = 717.  # specific heat at constant volume, in J/(K*kg)
RD = 287.  # ideal gas constant for dry air, in J/(K*kg)
RV = 461.6 # ideal gas constant for water vapor, in J/(K*kg)
F0 = 1.0e-4  # Coriolis parameter, in s^-1
SVPT0 = 273.15
GAMMA = CP/CV
KAPPA = RD/CP

### WRF file strings
file_name = 'wrfin_start_path_here'
netcdf_type = 'NETCDF3_64BIT_OFFSET'
title_str = "OUTPUT FROM IDEAL V3.6.1 PREPROCESSOR"
time_str = "2021-01-01_00:00:00"

### Grid parameters
nx = 400 # number of grid points in x direction
ny = 360 # number of grid points in y direction
nz = 100 # number of grid points in z direction
hres = 20. # horizontal grid spacing in km
zl = 20. # model top in km
p_bot = 101200. # bottom pressure for WRF in Pa
p_top = 5000. # top pressure for WRF in Pa
pibot = 1008. # bottom pi for EPV inversion
pitop = 424. # top pi for EPV inversion
npi_h = 180 # number of grid points in pi for high-resolution EPV inversion
npi_l = 40 # number of grid points in pi for low-resolution EPV inversion
ny_l = 30 # number of grid points in y for low-resolution EPV inversion
nit = 50000  # number of iterations for EPV inversion
om = 1.8  # successive over-relaxation coefficient for EPV inversion

### Background jet parameters
pvt = 0.2e-6 # tropospheric PV in (m^2*K)/(s*kg)
pvs = 7.0e-6 # stratospheric PV in (m^2*K)/(s*kg)
pim = 680. # mean height of tropopause in pi
dpitr = 30. # max displacement from mean tropopause height in pi
dpipv = 15. # tropopause depth in pi over which strongest PV change occurs
thtop = 535. # mean top potential temperature in K
dth = 10. # max displacement from mean top potential temperature in K
dyth = 1.0e6 # meridional scale for top potential temperature transition in m
dytr = 1.0e6  # meridional scale for tropopause height transition in m

### Horizontal shear option
shear_type = 'none' # choices are baro_cyc, baro_anticyc, surf_cyc, surf_anticyc

### Barotropic shear parameters
c = 10. # maximum additional wind speed in m/s
dyu = 1.0e6 # meridional scale for shear transition in m

### Surface-based shear parameters
dph = 750. # half displacement from bottom phi in m^2/s^2
dyph = 1.0e6 # meridional scale for bottom phi transition in m

### Set up grids
ny_h = ny
ly = ny_h*hres*1000
dy_l, ylvls_l, dy_h, ylvls_h = epv_jet.y_grid(ly, ny_l, ny_h)
dpi_l, pilvls_l, dpi_h, pilvls_h, p_pi_l, p_pi_h = epv_jet.pi_grid(pibot, pitop, npi_l, npi_h)
znw, znu, dnw, rdnw, dn, rdn, fnp, fnm, cof1, cof2, cf1,\
    cf2, cf3, cfn, cfn1, dx, dy, rdx, rdy = epv_jet.eta_grid(nz, hres)

### Low-resolution run
f_l = np.zeros((npi_l,ny_l))
pv_l, f_l, u_l, theta_l, pv_out_l = epv_jet.solve_PV_inversion(ly, ny_l, \
    dy_l, dytr, pim, dpitr, pvt, pvs, dpipv, npi_l, pibot, pitop, dpi_l, \
    dyth, thtop, dth, f_l, om, nit, dyph, dph, shear_type)

### Interpolate onto high-resolution grid
f_temp = np.zeros((npi_h,ny_l))
for j in range(ny_l):
    f_temp[:,j] = epv_jet.interp_decreasing(pilvls_l,pilvls_h,f_l[:,j],npi_l,npi_h)

f_interp = np.zeros((npi_h,ny_h))
for k in range(npi_h):
    f_interp[k,:] = epv_jet.interp_increasing(ylvls_l,ylvls_h,f_temp[k,:],ny_l,ny_h)

### High-resolution run
pv_h, f_h, u_h, theta_h, pv_out_h = epv_jet.solve_PV_inversion(ly, ny_h, \
    dy_h, dytr, pim, dpitr, pvt, pvs, dpipv, npi_h, pibot, pitop, dpi_h, \
    dyth, thtop, dth, f_interp, om, nit, dyph, dph, shear_type)

### Barotropic shear
if shear_type == 'baro_cyc' or shear_type == 'baro_anticyc':
    u_h, f_h = epv_jet.barotropic_shear(u_h, f_h, ly, ny_h, npi_h, dy_h, dpi_h, c, dyu, shear_type)

### Unstaggered pressure for WRF levels
p_unstag = znu*(p_bot - p_top) + p_top

### Interpolate from pi to WRF levels
u_eta = np.zeros((nz,ny))
theta_eta = np.zeros((nz,ny))
ph_eta = np.zeros((nz,ny))
p_eta = np.zeros((nz,ny))
for j in range(ny):
    u_eta[:,j] = epv_jet.interp_decreasing(p_pi_h,p_unstag,u_h[:,j],npi_h,nz)
    theta_eta[:,j] = epv_jet.interp_decreasing(p_pi_h,p_unstag,theta_h[:,j],npi_h,nz)
    ph_eta[:,j] = epv_jet.interp_decreasing(p_pi_h,p_unstag,f_h[:,j],npi_h,nz)
    p_eta[:,j] = p_unstag
    
### Bottom and top pressure for shear cases
if shear_type == 'baro_cyc' or shear_type == 'baro_anticyc' or shear_type == 'surf_cyc' or shear_type == 'surf_anticyc':
    p_bot_inc = np.zeros(ny)
    p_top_inc = np.zeros(ny)
    rho_eta = P0/(RD*theta_eta)*(p_eta/P0)**(CV/CP)
    u_bot = 1.5*u_eta[0,:] - 0.5*u_eta[1,:]
    u_top = 1.5*u_eta[-1,:] - 0.5*u_eta[-2,:]
    rho_bot = 1.5*rho_eta[0,:] - 0.5*rho_eta[1,:]
    rho_top = 1.5*rho_eta[-1,:] - 0.5*rho_eta[-2,:]

    pb_y = p_bot
    pt_y = p_top
    for i in range(int(ny/2. + 1)):
        pb_y = +F0*u_bot[int(ny/2. - i)]*rho_bot[int(ny/2. - i)]*hres*1000. + pb_y
        p_bot_inc[int(ny/2. - i)] = pb_y
        pt_y = +F0*u_top[int(ny/2. - i)]*rho_top[int(ny/2. - i)]*hres*1000. + pt_y
        p_top_inc[int(ny/2. - i)] = pt_y

    pb_y = p_bot
    pt_y = p_top
    for i in range(int(ny/2.),ny):
        pb_y = -F0*u_bot[i]*rho_bot[i]*hres*1000. + pb_y
        p_bot_inc[i] = pb_y
        pt_y = -F0*u_top[i]*rho_top[i]*hres*1000. + pt_y
        p_top_inc[i] = pt_y
        
else:
    p_bot_inc = np.ones(ny)*p_bot
    p_top_inc = np.ones(ny)*p_top
    
### Extend into x
u_jet = np.zeros((nz,ny,nx))
v_jet = np.zeros((nz,ny,nx))
theta_jet = np.zeros((nz,ny,nx))
ph_jet = np.zeros((nz,ny,nx))
p_jet = np.zeros((nz,ny,nx))
pbot = np.zeros((ny,nx))
ptop = np.zeros((ny,nx))

for i in range(nx):
    u_jet[:,:,i] = u_eta
    theta_jet[:,:,i] = theta_eta
    ph_jet[:,:,i] = ph_eta
    p_jet[:,:,i] = p_eta
    pbot[:,i] = p_bot_inc
    ptop[:,i] = p_top_inc

### Compute WRF base variables
qv = np.zeros((nz,ny,nx))
mub, pb, t_init, alb, phb = epv_jet.base_wrf(ph_jet, p_jet, theta_jet, pbot, ptop,\
                                             nx, ny, nz, znu, znw, dn, dnw)

### Compute WRF perturbation variables
u, v, ph, p, mu, t, tsk = epv_jet.pert_wrf(u_jet, v_jet, ph_jet, p_jet, theta_jet,\
                                           pb, alb, mub, pbot, ptop, nx, ny, nz, \
                                           znu, znw, dn, dnw)

hres_m = hres*1000.
ncfile = nc.Dataset(file_name,'w',format=netcdf_type)
ncfile = write_wrfinputd01.write(ncfile, nx, ny, nz, hres_m, title_str, \
        time_str, u, v, t, ph, phb, t_init, mu, mub, p, pb, \
        fnm, fnp, rdnw, rdn, dnw, dn, cfn, cfn1, rdx, rdy, cf1, \
        cf2, cf3, qv, znw, znu, p_top, tsk)
ncfile.close()
