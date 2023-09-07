"""
Written by Daniel J. Lloveras
"""

import numpy as np
from netCDF4 import Dataset
from bwave_ideal_wrf import epv_jet, qgpv_pert
from wrf import destagger, getvar
from ambiance import Atmosphere

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

### Grid parameters
nx = 400 # number of grid points in x direction
ny = 360 # number of grid points in y direction
nz = 100 # number of grid points in z direction
hres = 20. # horizontal grid resolution in km
zl = 20. # model top in km
dz = 200. # z grid spacing in m
ly = hres*ny*1000. # length in y direction in m

### Tropopause anomaly parameters
up_pert = True # tropopause anomaly option
up_pert_mag = 1.25e-4 # magnitude of tropopause anomaly in s^-1
x_up_pert = 140 # x center gridpoint for tropopause anomaly
y_up_pert = 165 # y center gridpoint for tropopause anomaly
z_up_pert = 40 # z center gridpoint for tropopause anomaly
ax_up_pert = 200. # x decay scale of tropopause anomaly in km
ay_up_pert = 600. # y decay scale of tropopause anomaly in km
az_up_pert = 1.5 # z decay scale of tropopause anomaly in km

### Surface anomaly parameters
surf_pert = True # surface anomaly option
surf_pert_mag = 4.0 # magnitude of surface theta anomaly in K
x_surf_pert = 200 # x center gridpoint for surface anomaly
y_surf_pert = 135 # y center gridpoint for surface anomaly
ax_surf_pert = 600. # x decay scale for surface anomaly in km
ay_surf_pert = 200. # y decay scale for surface anomaly in km

### Moisture parameters
moist = True # moisture option
rh_0 = 0.85 # surface relative humidity
zrh = 8000. # height decay scale in m
delta = 1.25 # height decay parameter
nit = 5 # number of iterations for virtual temperature calculation

### Read in the file
nc_in = Dataset('wrfin_path_here','r+')

### Base variables
znu = np.asarray(nc_in.variables['ZNU'][0,:])
znw = np.asarray(nc_in.variables['ZNW'][0,:])
dnw = np.asarray(nc_in.variables['DNW'][0,:])
dn = np.asarray(nc_in.variables['DN'][0,:])
phb = np.asarray(nc_in.variables['PHB'][0,:,:,:])
pb = np.asarray(nc_in.variables['PB'][0,:,:,:])
mub = np.asarray(nc_in.variables['MUB'][0,:,:])
t_init = np.asarray(nc_in.variables['T_INIT'][0,:,:])
alb = (RD/P0)*(t_init + T0)*((pb/P0)**(-1/GAMMA))

### Full variables
u_jet = destagger(np.asarray(nc_in.variables['U'][0,:,:,:]),-1)
v_jet = destagger(np.asarray(nc_in.variables['V'][0,:,:,:]),-2)
th_jet = np.asarray(nc_in.variables['T'][0,:,:,:]) + T0
ph_jet = np.asarray(nc_in.variables['PH'][0,:,:,:]) + phb
p_jet = np.asarray(nc_in.variables['P'][0,:,:,:]) + pb
mu_jet = np.asarray(nc_in.variables['MU'][0,:,:]) + mub
z_jet = destagger(ph_jet/G,0)
rho_jet = P0/(RD*th_jet)*(p_jet/P0)**(CV/CP)

### Set up grids
xl, yl, x, y, xg, yg, dz, zp = qgpv_pert.cartesian_mesh(nx, ny, nz, hres, zl)
kmax, lmax, facx, facy, dxg, dyg = qgpv_pert.spectral_mesh(nx, ny, xl, yl)

### Compute reference static stability using standard atmosphere
atm_ref = Atmosphere(zp)
t_ref = atm_ref.temperature
p_ref = atm_ref.pressure
th_ref = (P0/p_ref)**(RD/CP)*t_ref

n_ref = np.zeros(nz-1)
dtdz_ref = np.zeros(nz-1)
for k in range(1,nz-1):
    dtdz_ref[k] = (th_ref[k+1] - th_ref[k-1])/(zp[k+1] - zp[k-1])
    n_ref[k] = np.sqrt(G/th_ref[k]*dtdz_ref[k])
    
dtdz_ref[0] = dtdz_ref[1]
n_ref[0] = n_ref[1]

### Tropopause anomaly
if up_pert:
    
    ### Initialize
    pvxy, bcxy, pvsp, b_fac = qgpv_pert.trop_anom(up_pert_mag,\
        x_up_pert, y_up_pert, z_up_pert, az_up_pert, ax_up_pert,\
        ay_up_pert, x, y, zp, xg, yg, nx, ny, nz, n_ref, dz, dtdz_ref)
    
    ### Invert
    fsp = qgpv_pert.qgpv_inversion(nx, ny, nz, b_fac, facx, facy,\
        kmax, lmax, pvsp, dz)

    ### Compute perturbations
    u_up_pert, v_up_pert, theta_up_pert, rho_up_pert, fxy_up = qgpv_pert.qgpv_solver(fsp,\
        nx, ny, nz, b_fac, dxg, dyg, dz, bcxy, dtdz_ref)

    p_up_pert = fxy_up*F0
    
else:
    u_up_pert = np.zeros((nz,ny,nx))
    v_up_pert = np.zeros((nz,ny,nx))
    theta_up_pert = np.zeros((nz,ny,nx))
    rho_up_pert = np.zeros((nz,ny,nx))
    p_up_pert = np.zeros((nz,ny,nx))

### Surface anomaly
if surf_pert:
    
    ### Initialize
    pvs, bcs, pvsps, b_facs = qgpv_pert.surf_anom(surf_pert_mag,\
        x_surf_pert, y_surf_pert, ax_surf_pert, ay_surf_pert,\
        x, y, zp, xg, yg, nx, ny, nz, n_ref, dz, dtdz_ref)
    
    ### Invert
    fsps = qgpv_pert.qgpv_inversion(nx, ny, nz, b_facs, facx, facy,\
        kmax, lmax, pvsps, dz)
    
    ### Compute perturbations
    u_surf_pert, v_surf_pert, theta_surf_pert, rho_surf_pert, fxy_surf = qgpv_pert.qgpv_solver(fsps,\
        nx, ny, nz, b_facs, dxg, dyg, dz, bcs, dtdz_ref)

    p_surf_pert = fxy_surf*F0

else:
    u_surf_pert = np.zeros((nz,ny,nx))
    v_surf_pert = np.zeros((nz,ny,nx))
    theta_surf_pert = np.zeros((nz,ny,nx))
    rho_surf_pert = np.zeros((nz,ny,nx))
    p_surf_pert = np.zeros((nz,ny,nx))

### Add perturbations together
u_pert = u_up_pert + u_surf_pert
v_pert = v_up_pert + v_surf_pert
theta_pert = theta_up_pert + theta_surf_pert
rho_pert = rho_up_pert + rho_surf_pert
p_pert = p_up_pert + p_surf_pert

### Interpolate onto eta levels
up_eta = np.zeros((nz,ny,nx))
vp_eta = np.zeros((nz,ny,nx))
thetap_eta = np.zeros((nz,ny,nx))
rhop_eta = np.zeros((nz,ny,nx))
presp_eta = np.zeros((nz,ny,nx))
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            thetap_eta[k,j,i] = qgpv_pert.interp_0(theta_pert[:,j,i], zp, z_jet[k,j,i], nz)
            rhop_eta[k,j,i] = qgpv_pert.interp_0(rho_pert[:,j,i], zp, z_jet[k,j,i], nz)            
            up_eta[k,j,i] = qgpv_pert.interp_0(u_pert[:,j,i], zp, z_jet[k,j,i], nz)
            vp_eta[k,j,i] = qgpv_pert.interp_0(v_pert[:,j,i], zp, z_jet[k,j,i], nz)
            presp_eta[k,j,i] = qgpv_pert.interp_0(p_pert[:,j,i], zp, z_jet[k,j,i], nz)            

u_eta = u_jet + up_eta
v_eta = v_jet + vp_eta
rho_eta = rho_jet + rhop_eta
theta_eta = th_jet + thetap_eta
p_eta = p_jet + presp_eta
ph_eta = z_jet*G

### Compute new perturbation variables
p_bot = 1.5*p_eta[0,:,:] - 0.5*p_eta[1,:,:]
p_top = 1.5*p_eta[-1,:,:] - 0.5*p_eta[-2,:,:]
u, v, ph, p, mu, t, tsk = epv_jet.pert_wrf(u_eta, v_eta, ph_eta, p_eta, theta_eta, pb, alb, mub,\
                                           p_bot, p_top, nx, ny, nz, znu, znw, dn, dnw)
        
### Add moisture and update theta
if moist:
    rh = np.zeros((nz,ny,nx))
    for k in range(nz):
        z = k*dz
        if z < zrh: rh[k,:,:] = np.ones((ny,nx))*rh_0*(1 - 0.9*(z/zrh)**delta)
        else: rh[k,:,:] = np.ones((ny,nx))*rh_0*0.1

    theta_new = np.zeros((nit,nz,ny,nx))
    theta_new[0,:,:,:] = t + T0
    for it in range(nit-1):
        temp = theta_new[it,:,:,:]*((p + pb)/P0)**(RD/CP)
        es = 611.2*np.exp(17.67*(temp - SVPT0)/(temp - 29.65))
        qv = rh*(RD/RV)*(es/((p + pb) - es))
        theta_new[it+1,:,:,:] = (t + T0)/(1 + 0.61*qv)

    t_moist = theta_new[nit-1,:,:,:] - T0
    tsk_moist = theta_new[nit-1,0,:,:]*(((p + pb)[0,:,:]/P0)**(RD/CP))
    
else:
    t_moist = t
    tsk_moist = tsk
    qv = np.zeros((nz,ny,nx))

### Write new variables to the input file
nc_in.variables['U'][0,:,:,:] = u
nc_in.variables['V'][0,:,:,:] = v
nc_in.variables['PH'][0,:,:,:] = ph
nc_in.variables['MU'][0,:,:] = mu
nc_in.variables['P'][0,:,:,:] = p
nc_in.variables['T'][0,:,:,:] = t_moist
nc_in.variables['TSK'][0,:,:] = tsk_moist
nc_in.variables['QVAPOR'][0,:,:,:] = qv
nc_in.close()
