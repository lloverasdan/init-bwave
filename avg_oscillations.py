"""
Written by Daniel J. Lloveras
"""

from netCDF4 import Dataset
from wrf import getvar
import numpy as np

nt = 35
nz = 100
ny = 360
nx = 400
nc_out = Dataset('wrfout_osc_path_here')

u = np.zeros((nt,nz,ny,nx+1))
v = np.zeros((nt,nz,ny+1,nx))
ph = np.zeros((nt,nz+1,ny,nx))
t = np.zeros((nt,nz,ny,nx))
mu = np.zeros((nt,ny,nx))
p = np.zeros((nt,nz,ny,nx))
tsk = np.zeros((nt,ny,nx))
for ti in range(nt):
    u[ti,:,:,:] = np.asarray(getvar(nc_out,'U',timeidx=ti+1))
    v[ti,:,:,:] = np.asarray(getvar(nc_out,'V',timeidx=ti+1))
    ph[ti,:,:,:] = np.asarray(getvar(nc_out,'PH',timeidx=ti+1))
    t[ti,:,:,:] = np.asarray(getvar(nc_out,'T',timeidx=ti+1))
    mu[ti,:,:] = np.asarray(getvar(nc_out,'MU',timeidx=ti+1))
    p[ti,:,:,:] = np.asarray(getvar(nc_out,'P',timeidx=ti+1))
    tsk[ti,:,:] = np.asarray(getvar(nc_out,'TSK',timeidx=ti+1))
    
u_avg = np.mean(u,axis=0)
v_avg = np.mean(v,axis=0)
ph_avg = np.mean(ph,axis=0)
t_avg = np.mean(t,axis=0)
mu_avg = np.mean(mu,axis=0)
p_avg = np.mean(p,axis=0)
tsk_avg = np.mean(tsk,axis=0)

nc_in = Dataset('wrfin_avg_path_here','r+')

nc_in.variables['U'][0,:,:,:] = u_avg
nc_in.variables['V'][0,:,:,:] = v_avg
nc_in.variables['PH'][0,:,:,:] = ph_avg
nc_in.variables['T'][0,:,:,:] = t_avg
nc_in.variables['MU'][0,:,:] = mu_avg
nc_in.variables['P'][0,:,:,:] = p_avg
nc_in.variables['TSK'][0,:,:] = tsk_avg

nc_in.close()
