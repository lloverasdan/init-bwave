#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Written by Daniel J. Lloveras
"""

import numpy as np
import netCDF4 as nc
from bwave_ideal_wrf import qgpv_pert, write_wrfinputd01
from wrf import destagger

# DEFINE

nc_in = nc.Dataset('wrfincoarse_path_here')
netcdf_type = 'NETCDF3_64BIT_OFFSET'
title_str = "OUTPUT FROM IDEAL V3.6.1 PREPROCESSOR"
time_str = "2021-01-01_00:00:00"

nx_in = 400
ny_in = 360
dx_in = 20.
hres_m_in = dx_in*1000.

nx_out = 2000
ny_out = 1800
dx_out = 4.
hres_m_out = dx_out*1000.
rdx_out = 1/hres_m_out
rdy_out = 1/hres_m_out
ptop = 5000.
nz = 100

# READ

u_in = destagger(np.array(nc_in.variables['U']),stagger_dim=3)[0,:,:,:]
v_in = destagger(np.array(nc_in.variables['V']),stagger_dim=2)[0,:,:,:]
t_in = np.array(nc_in.variables['T'])[0,:,:,:]
ph_in = np.array(nc_in.variables['PH'])[0,:,:,:]
phb_in = np.array(nc_in.variables['PHB'])[0,:,:,:]
p_in = np.array(nc_in.variables['P'])[0,:,:,:]
pb_in = np.array(nc_in.variables['PB'])[0,:,:,:]
t_init_in = np.array(nc_in.variables['T_INIT'])[0,:,:,:]
mu_in = np.array(nc_in.variables['MU'])[0,:,:]
mub_in = np.array(nc_in.variables['MUB'])[0,:,:]
moist_in = np.array(nc_in.variables['QVAPOR'])[0,:,:,:]
tsk_in = np.array(nc_in.variables['TSK'])[0,:,:]

fnm = np.array(nc_in.variables['FNM'])[0,:]
fnp = np.array(nc_in.variables['FNP'])[0,:]
rdnw = np.array(nc_in.variables['RDNW'])[0,:]
rdn = np.array(nc_in.variables['RDN'])[0,:]
dnw = np.array(nc_in.variables['DNW'])[0,:]
dn = np.array(nc_in.variables['DN'])[0,:]
cfn = float(np.asarray(nc_in.variables['CFN'])[0])
cfn1 = float(np.asarray(nc_in.variables['CFN1'])[0])
cf1 = float(np.asarray(nc_in.variables['CF1'])[0])
cf2 = float(np.asarray(nc_in.variables['CF2'])[0])
cf3 = float(np.asarray(nc_in.variables['CF3'])[0])
znw = np.array(nc_in.variables['ZNW'])[0,:]
znu = np.array(nc_in.variables['ZNU'])[0,:]

# INTERPOLATE

x_in = np.linspace(int(dx_in),int(nx_in*dx_in),int(nx_in))
x_out = np.linspace(int(dx_out),int(nx_out*dx_out),int(nx_out))

u_temp = np.zeros([nz, ny_in, nx_out+1])
v_temp = np.zeros([nz, ny_in+1, nx_out])
t_temp = np.zeros([nz, ny_in, nx_out])
ph_temp = np.zeros([nz+1, ny_in, nx_out])
phb_temp = np.zeros([nz+1, ny_in, nx_out])
p_temp = np.zeros([nz, ny_in, nx_out])
pb_temp = np.zeros([nz, ny_in, nx_out])
t_init_temp = np.zeros([nz, ny_in, nx_out])
mu_temp = np.zeros([ny_in, nx_out])
mub_temp = np.zeros([ny_in, nx_out])
moist_temp = np.zeros([nz, ny_in, nx_out])
tsk_temp = np.zeros([ny_in, nx_out])

for j in range(ny_in):
    for i in range(nx_out):
        ph_temp[nz,j,i] = qgpv_pert.interp_0(ph_in[nz,j,:], x_in, x_out[i], nx_in)
        phb_temp[nz,j,i] = qgpv_pert.interp_0(phb_in[nz,j,:], x_in, x_out[i], nx_in)
        for k in range(nz):
            u_temp[k,j,i] = qgpv_pert.interp_0(u_in[k,j,:], x_in, x_out[i], nx_in)
            v_temp[k,j,i] = qgpv_pert.interp_0(v_in[k,j,:], x_in, x_out[i], nx_in)
            t_temp[k,j,i] = qgpv_pert.interp_0(t_in[k,j,:], x_in, x_out[i], nx_in)
            ph_temp[k,j,i] = qgpv_pert.interp_0(ph_in[k,j,:], x_in, x_out[i], nx_in)
            phb_temp[k,j,i] = qgpv_pert.interp_0(phb_in[k,j,:], x_in, x_out[i], nx_in)
            p_temp[k,j,i] = qgpv_pert.interp_0(p_in[k,j,:], x_in, x_out[i], nx_in)
            pb_temp[k,j,i] = qgpv_pert.interp_0(pb_in[k,j,:], x_in, x_out[i], nx_in)
            t_init_temp[k,j,i] = qgpv_pert.interp_0(t_init_in[k,j,:], x_in, x_out[i], nx_in)
            mu_temp[j,i] = qgpv_pert.interp_0(mu_in[j,:], x_in, x_out[i], nx_in)
            mub_temp[j,i] = qgpv_pert.interp_0(mub_in[j,:], x_in, x_out[i], nx_in)
            moist_temp[k,j,i] = qgpv_pert.interp_0(moist_in[k,j,:], x_in, x_out[i], nx_in)
            tsk_temp[j,i] = qgpv_pert.interp_0(tsk_in[j,:], x_in, x_out[i], nx_in)
            
u_temp[:,:,-1] = u_temp[:,:,-2]
v_temp[:,-1,:] = v_temp[:,-1,:]

y_in = np.linspace(int(dx_in),int(ny_in*dx_in),int(ny_in))
y_out = np.linspace(int(dx_out),int(ny_out*dx_out),int(ny_out))

u_out = np.zeros([nz, ny_out, nx_out+1])
v_out = np.zeros([nz, ny_out+1, nx_out])
t_out = np.zeros([nz, ny_out, nx_out])
ph_out = np.zeros([nz+1, ny_out, nx_out])
phb_out = np.zeros([nz+1, ny_out, nx_out])
p_out = np.zeros([nz, ny_out, nx_out])
pb_out = np.zeros([nz, ny_out, nx_out])
t_init_out = np.zeros([nz, ny_out, nx_out])
mu_out = np.zeros([ny_out, nx_out])
mub_out = np.zeros([ny_out, nx_out])
moist_out = np.zeros([nz, ny_out, nx_out])
tsk_out = np.zeros([ny_out, nx_out])

for j in range(ny_out):
    for i in range(nx_out):
        ph_out[nz,j,i] = qgpv_pert.interp_0(ph_temp[nz,:,i], y_in, y_out[j], ny_in)
        phb_out[nz,j,i] = qgpv_pert.interp_0(phb_temp[nz,:,i], y_in, y_out[j], ny_in)
        for k in range(nz):
            u_out[k,j,i] = qgpv_pert.interp_0(u_temp[k,:,i], y_in, y_out[j], ny_in)
            v_out[k,j,i] = qgpv_pert.interp_0(v_temp[k,:,i], y_in, y_out[j], ny_in)
            t_out[k,j,i] = qgpv_pert.interp_0(t_temp[k,:,i], y_in, y_out[j], ny_in)
            ph_out[k,j,i] = qgpv_pert.interp_0(ph_temp[k,:,i], y_in, y_out[j], ny_in)
            phb_out[k,j,i] = qgpv_pert.interp_0(phb_temp[k,:,i], y_in, y_out[j], ny_in)
            p_out[k,j,i] = qgpv_pert.interp_0(p_temp[k,:,i], y_in, y_out[j], ny_in)
            pb_out[k,j,i] = qgpv_pert.interp_0(pb_temp[k,:,i], y_in, y_out[j], ny_in)
            t_init_out[k,j,i] = qgpv_pert.interp_0(t_init_temp[k,:,i], y_in, x_out[j], ny_in)
            mu_out[j,i] = qgpv_pert.interp_0(mu_temp[:,i], y_in, y_out[j], ny_in)
            mub_out[j,i] = qgpv_pert.interp_0(mub_temp[:,i], y_in, y_out[j], ny_in)
            moist_out[k,j,i] = qgpv_pert.interp_0(moist_temp[k,:,i], y_in, y_out[j], ny_in)
            tsk_out[j,i] = qgpv_pert.interp_0(tsk_temp[:,i], y_in, y_out[j], ny_in)
            
u_out[:,:,-1] = u_out[:,:,-2]
v_out[:,-1,:] = v_out[:,-1,:]

# WRITE
nc_out = nc.Dataset('wrfinfine_path_here','w',format=netcdf_type)
nc_out = write_wrfinputd01.write(nc_out, nx_out, ny_out, nz, hres_m_out, title_str, \
        time_str, u_out, v_out, t_out, ph_out, phb_out, t_init_out, mu_out, mub_out, p_out, pb_out, \
        fnm, fnp, rdnw, rdn, dnw, dn, cfn, cfn1, rdx_out, rdy_out, cf1, \
        cf2, cf3, moist_out, znw, znu, ptop, tsk_out)
nc_out.close()
