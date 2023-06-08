#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Written by Daniel J. Lloveras
"""

import numpy as np
from numba import njit

#%% Constants

F0 = 1.0e-4  # Coriolis parameter, in s^-1
G = 9.81  # gravitational acceleration, in m/s^2
T0 = 300.  # reference potential temperature, in K
P0 = 1.0e5  # reference pressure, in Pa
CP = 1004.  # specific heat at constant pressure, in J/(K*kg)       
CV = 717.  # specific heat at constant volume, in J/(K*kg)
RD = 287.  # ideal gas constant for dry air, in J/(K*kg)
GAMMA = CP/CV
KAPPA = RD/CP

#%% Grids

@njit
def y_grid(ly, ny_l, ny_h):
    
    dy_l = ly/(ny_l - 1)  # grid res in y for low-res run
    ylvls_l = np.zeros(ny_l)  # low-res y grid values
    for j in range(ny_l):
        ylvls_l[j] = j*dy_l
        
    dy_h = ly/(ny_h - 1)  # grid res in y for high-res run
    ylvls_h = np.zeros(ny_h)  # high-res y grid values
    for j in range(ny_h):
        ylvls_h[j] = j*dy_h
        
    return dy_l, ylvls_l, dy_h, ylvls_h

@njit
def pi_grid(pibot, pitop, npi_l, npi_h):
    
    dpi_l = (pibot - pitop)/(npi_l - 1)  # grid res in pi for low-res run
    pilvls_l = np.zeros(npi_l)  # low-res pi grid values
    p_pi_l = np.zeros(npi_l)
    for k in range(npi_l):
        pilvls_l[k] = pibot - k*dpi_l
        p_pi_l[k] = P0*(pilvls_l[k]/CP)**(1/KAPPA)
        
    dpi_h = (pibot - pitop)/(npi_h - 1)  # grid res in pi for high-res run  
    pilvls_h = np.zeros(npi_h)  # high-res pi grid values
    p_pi_h = np.zeros(npi_h)
    for k in range(npi_h):
        pilvls_h[k] = pibot - k*dpi_h
        p_pi_h[k] = P0*(pilvls_h[k]/CP)**(1/KAPPA)
        
    return dpi_l, pilvls_l, dpi_h, pilvls_h, p_pi_l, p_pi_h

@njit
def eta_grid(nz, hres):
    
    znw = np.zeros(nz+1)
    for k in range(nz+1):
        znw[k] = (np.exp(-2*k/float(nz)) - np.exp(-2))/(1 - np.exp(-2))
    
    znu = np.zeros(nz)
    for k in range(nz):
        znu[k] = 0.5*(znw[k+1] + znw[k])
        
    dnw = np.zeros(nz)
    rdnw = np.zeros(nz)
    for k in range(nz):
        dnw[k] = znw[k+1] - znw[k]
        rdnw[k] = 1./dnw[k]

    dn = np.zeros(nz)
    rdn = np.zeros(nz)
    fnp = np.zeros(nz)
    fnm = np.zeros(nz)
    for k in range(1,nz):
        dn[k] = 0.5*(dnw[k] + dnw[k-1])
        rdn[k] = 1./dn[k]
        fnp[k] = 0.5*dnw[k]/dn[k]
        fnm[k] = 0.5*dnw[k-1]/dn[k]

    cof1 = (2.*dn[1] + dn[2])/(dn[1] + dn[2])*dnw[0]/dn[1]
    cof2 = dn[1]/(dn[1] + dn[2])*dnw[0]/dn[2]
    cf1 = fnp[1] + cof1
    cf2 = fnm[1] - cof1 - cof2
    cf3 = cof2
    cfn = (0.5*dnw[nz-1] + dn[nz-1])/dn[nz-1]
    cfn1 = -0.5*dnw[nz-1]/dn[nz-1]
    dx = hres*1000.
    dy = hres*1000.
    rdx = 1./dx
    rdy = 1./dy    
        
    return znw, znu, dnw, rdnw, dn, rdn, fnp, fnm, cof1, cof2, cf1, cf2, cf3, cfn, cfn1, dx, dy, rdx, rdy

#%% Interpolation functions

@njit
def interp_0(v_in, z_in, z_val, nz_in):
    
    v_val = 0.
    height = z_val
    if (z_in[nz_in-1] > z_in[0]):
        if (height > z_in[nz_in-1]):
            w2 = (z_in[nz_in-1] - height)/(z_in[nz_in-1] - z_in[nz_in-2])
            w1 = 1. - w2
            v_val = w1*v_in[nz_in-1] + w2*v_in[nz_in-2]
        
        elif (height < z_in[0]):
            w2 = (z_in[1] - height)/(z_in[1] - z_in[0])
            w1 = 1. - w2
            v_val = w1*v_in[1] + w2*v_in[0]
            
        else:
            interp = False
            kp = nz_in-1
            while (interp == False and kp >= 1):
                if (z_in[kp] >= height and z_in[kp-1] <= height):
                    w2 = (height - z_in[kp])/(z_in[kp-1] - z_in[kp])
                    w1 = 1. - w2
                    v_val = w1*v_in[kp] + w2*v_in[kp-1]
                    interp = True

                kp = kp - 1                    
    
    else:
        if (height < z_in[nz_in-1]):
            w2 = (z_in[nz_in-1] - height)/(z_in[nz_in-1] - z_in[nz_in-2])
            w1 = 1. - w2
            v_val = w1*v_in[nz_in-1] + w2*v_in[nz_in-2]
        
        elif (height > z_in[0]):
            w2 = (z_in[1] - height)/(z_in[1] - z_in[0])
            w1 = 1. - w2
            v_val = w1*v_in[1] + w2*v_in[0]
            
        else:
            interp = False
            kp = nz_in-1
            while (interp == False and kp >= 1):
                if (z_in[kp] <= height and z_in[kp-1] >= height):
                    w2 = (height - z_in[kp])/(z_in[kp-1] - z_in[kp])
                    w1 = 1. - w2
                    v_val = w1*v_in[kp] + w2*v_in[kp-1]
                    interp = True

                kp = kp - 1          
            
    return v_val

### Increasing grid values
@njit    
def interp_increasing(x_in, x_out, y_in, n_in, n_out):
    
    y_out = np.zeros(n_out)
    i_in = 0
    for i in range(n_out):
        while True:
            if (x_out[i] >= x_in[i_in] and x_out[i] <= x_in[i_in+1]):
                dx = x_out[i] - x_in[i_in]
                dx_in = x_in[i_in+1] - x_in[i_in]
                y_out[i] = y_in[i_in] + (y_in[i_in+1] - y_in[i_in])*dx/dx_in
                
                break
            
            else:
                i_in = i_in+1
                if (i_in > n_in-1):
                    break
                
        else:
            continue
        
        if (i_in > n_in-1):
            break

    return y_out

### Decreasing grid values
@njit    
def interp_decreasing(x_in, x_out, y_in, n_in, n_out):
    
    y_out = np.zeros(n_out)
    i_in = 0
    for i in range(n_out):
        while True:
            if (x_out[i] <= x_in[i_in] and x_out[i] >= x_in[i_in+1]):
                dx = x_out[i] - x_in[i_in]
                dx_in = x_in[i_in+1] - x_in[i_in]
                y_out[i] = y_in[i_in] + (y_in[i_in+1] - y_in[i_in])*dx/dx_in
                
                break
            
            else:
                i_in = i_in+1
                if (i_in > n_in-1):
                    break
                
        else:
            continue
        
        if (i_in > n_in-1):
            break

    return y_out

#%% Theta at top of atmosphere
    
@njit
def theta_top(ly, ny, dy, dyth, thtop, dth):

    thtopline = np.zeros(ny)
    for j in range(ny):
        a2 = 1.5*(j*dy - ly/2.)/dyth
        if (np.abs(a2) < np.pi/2.):
            thtopline[j] = thtop + dth*np.sin(a2)
        elif (a2 > np.pi/2.):
            thtopline[j] = thtop + dth
        else:
            thtopline[j] = thtop - dth

    return thtopline

#%% Phi at bottom of atmosphere
    
@njit
def phi_bot(ly, ny, dy, dyph, phbot, dph, shear_type):

    if shear_type == 'cyc':
        phbotline = np.zeros(ny)
        for j in range(ny):
            a3 = 1.5*(j*dy - ly/2.)/dyph
            if (np.abs(a3) < np.pi):
                phbotline[j] = phbot - dph*np.cos(a3)
            else:
                phbotline[j] = phbot + dph

    elif shear_type == 'anticyc':
        phbotline = np.zeros(ny)
        for j in range(ny):
            a3 = 1.5*(j*dy - ly/2.)/dyph
            if (np.abs(a3) < np.pi):
                phbotline[j] = phbot + dph*np.cos(a3)
            else:
                phbotline[j] = phbot - dph           
                
    else:
        phbotline = np.ones(ny)*phbot
        
    return phbotline

#%% Tropopause shape
    
@njit
def trop_shape(ly, ny, dy, dytr, pim, dpitr):
    
    pitp = np.zeros(ny)
    for j in range(ny):
        a1 = 2.*(j*dy - ly/2.)/dytr
        if (np.abs(a1) <= np.pi/2.):
            pitp[j] = pim + dpitr*np.sin(a1)
        elif (a1 > np.pi/2.):
            pitp[j] = pim + dpitr
        else:
            pitp[j] = pim - dpitr
    
    return pitp
    
#%% PV distribution

@njit
def pv_dist(pvt, pvs, dpipv, ny, npi, pibot, pitop, dpi, pitp):
    
    pv = np.zeros((npi,ny))
    for j in range(ny):
        for k in range(npi):
            pilev = pibot - k*dpi
            gamma_val = (pibot - pilev)/(pibot - pitp[j])
            beta_val = (pitp[j] - pilev)/(pitp[j] - pitop)
            if (pitp[j] > pilev):
                gamma_val = 0.
            else:
                beta_val = 0.
                
            a = 1. + 5.0*(gamma_val**2.)
            b = 1. + 3.0*(beta_val**3.)               
            pv[k,j] = (pvt*a + pvs*b)/2. + \
                        (pvt*a - pvs*b)/2.*(np.tanh(2.*(pilev - pitp[j])/dpipv))
      
    return pv

#%% PV inversion
    
@njit
def pv_inv(pv, f, ny, pibot, dy, npi, dpi, om):    

    fct = ((dy*dpi)**2.)*F0*P0/(G*KAPPA*(CP**(1/KAPPA)))
    for k in range(1,npi-1):
        pilev = pibot - k*dpi
        for j in range(1,ny-1):
            b = -0.5*(f[k,j+1] + f[k,j-1] + f[k+1,j] + f[k-1,j] + (F0*dy)**2.)
            c = 0.25*(-fct*pilev**(-(1 - (1/KAPPA)))*pv[k,j] \
                - (1/16.)*((f[k+1,j+1] - f[k+1,j-1] - f[k-1,j+1] + f[k-1,j-1])**2.) \
                + (f[k,j+1] + f[k,j-1])*(f[k+1,j] + f[k-1,j]) \
                + ((F0*dy)**2.)*(f[k+1,j] + f[k-1,j]))
            d = np.abs(b**2. - 4.*c)
            r = 0.5*(-b - np.sqrt(d)) - f[k,j]
            f[k,j] = f[k,j] + om*r
    
    return f

#%% U, Theta, and PV computations
    
@njit
def u_calc(f, ny, npi, dy):    

    u = np.zeros((npi,ny))
    for j in range(1,ny-1):
        u[:,j] = -(1/F0)*(f[:,j+1] - f[:,j-1])/(2.*dy)   
    
    u[:,0] = -(1/F0)*(f[:,1] - f[:,0])/dy
    u[:,ny-1] = -(1/F0)*(f[:,ny-1] - f[:,ny-2])/dy     

    return u

@njit
def theta_calc(f, ny, npi, dpi):

    theta = np.zeros((npi,ny))
    for k in range(1,npi-1):
        theta[k,:] = (f[k+1,:] - f[k-1,:])/(2.*dpi)

    theta[0,:] = (f[1,:] - f[0,:])/dpi 
    theta[npi-1,:] = (f[npi-1,:] - f[npi-2,:])/dpi        

    return theta
    
@njit
def pv_calc(f, ny, dy, npi, dpi, pibot):

    pv_out = np.zeros((npi,ny))
    for k in range(1,npi-1):
        pilev = pibot - k*dpi
        for j in range(1,ny-1):
            pv_out[k,j] = (G*KAPPA*(CP**(1/KAPPA))/P0)*pilev**(1 - (1/KAPPA)) \
                        *((F0*(f[k+1,j] - 2*f[k,j] + f[k-1,j])/(dpi**2.)) \
                        - ((1/F0)*(1/(4*dy*dpi)**2.)*(f[k+1,j+1] - f[k+1,j-1] \
                        - f[k-1,j+1] + f[k-1,j-1])**2) \
                        + ((f[k,j+1] - 2*f[k,j] + f[k,j-1])/(F0*dy*dy)) \
                        *((f[k+1,j] - 2*f[k,j] + f[k-1,j])/(dpi**2.)))
                        
    pv_out[0,:] = pv_out[1,:]
    pv_out[npi-1,:] = pv_out[npi-2,:]
    pv_out[:,0] = pv_out[:,1]
    pv_out[:,ny-1] = pv_out[:,ny-2]
    
    return pv_out
    
#%% Iteration
 
@njit    
def solve_PV_inversion(ly, ny, dy, dytr, pim, dpitr, pvt, pvs, dpipv, npi, \
                       pibot, pitop, dpi, dyth, thtop, dth, f, om, nit, \
                       dyph, phbot, dph, shear_type):
    
    ### Compute tropopause shape, PV distribution, and top theta
    pitp = trop_shape(ly, ny, dy, dytr, pim, dpitr)
    pv = pv_dist(pvt, pvs, dpipv, ny, npi, pibot, pitop, dpi, pitp)
        
    thtopline = theta_top(ly, ny, dy, dyth, thtop, dth)
    phbotline = phi_bot(ly, ny, dy, dyph, phbot, dph, shear_type)
    
    ### Run PV inversion iterations and apply boundary conditions  
    for i in range(nit):
        f = pv_inv(pv, f, ny, pibot, dy, npi, dpi, om)
        f[0,:] = phbotline
        f[npi-1,:] = f[npi-2,:] + thtopline[:]*dpi
        f[:,0] = f[:,1]
        f[:,ny-1] = f[:,ny-2]

    ### Compute u, theta, and PV given solution for phi
    u = u_calc(f, ny, npi, dy)
    theta = theta_calc(f, ny, npi, dpi)
    pv_out = pv_calc(f, ny, dy, npi, dpi, pibot)
    
    return pv, f, u, theta, pv_out

@njit
def base_wrf(ph_in, p_in, theta_in, p_bot, p_top, nx, ny, nz, znu, znw, dn, dnw):
    
    ip = int(nx/2) - 1
    jp = int(ny/2) - 1

    p_ex = np.append(p_bot[jp,ip],p_in[:,jp,ip])
    z_ex = np.append(0.,ph_in[:,jp,ip]/G)
    mub = np.zeros((ny,nx))
    pb = np.zeros((nz,ny,nx))
    t_init = np.zeros((nz,ny,nx))
    alb = np.zeros((nz,ny,nx))
    phb = np.zeros((nz+1,ny,nx))
    for j in range(ny):
        for i in range(nx):
            ps = interp_0(p_ex, z_ex, 0., nz+1)
            mub[j,i] = ps - p_top[jp,ip]
            for k in range(nz):
                pb[k,j,i] = znu[k]*(ps - p_top[jp,ip]) + p_top[jp,ip]
                t_init[k,j,i] = interp_0(theta_in[:,jp,ip], p_in[:,jp,ip], pb[k,j,i], nz) - T0
                alb[k,j,i] = (RD/P0)*(t_init[k,j,i] + T0)*((pb[k,j,i]/P0)**(-1/GAMMA))
                
            for k in range(1,nz+1):    
                phb[k,j,i] = phb[k-1,j,i] - dnw[k-1]*mub[j,i]*alb[k-1,j,i]
    
    return mub, pb, t_init, alb, phb

@njit
def pert_wrf(u_in, v_in, ph_in, p_in, theta_in, pb, alb, mub, p_bot, p_top, nx, ny, nz, znu, znw, dn, dnw):
    
    mu = np.zeros((ny,nx))
    t = np.zeros((nz,ny,nx))
    p = np.zeros((nz,ny,nx))
    alt = np.zeros((nz,ny,nx))
    al = np.zeros((nz,ny,nx))
    ph = np.zeros((nz+1,ny,nx))
    u = np.zeros((nz,ny,nx+1))
    v = np.zeros((nz,ny+1,nx))
    for j in range(ny):
        for i in range(nx):
            z1 = ph_in[:,j,i]/G
            pd1 = p_in[:,j,i]
            th1 = theta_in[:,j,i]
            u1 = u_in[:,j,i]
            v1 = v_in[:,j,i]

            ### Sounding
            p_ex = np.append(p_bot[j,i],pd1[:])
            z_ex = np.append(0.,z1[:])
            pd_surf = interp_0(p_ex, z_ex, 0., nz+1)

            ### Interpolate
            for k in range(nz):
                p_level = znu[k]*(pd_surf - p_top[j,i]) + p_top[j,i]
                t[k,j,i] = interp_0(th1, pd1, p_level, nz) - T0

            ### Compute fields at top of atmosphere
            mu[j,i] = (pd_surf - p_top[j,i]) - mub[j,i]
            p[nz-1,j,i] = -0.5*mu[j,i]*dnw[nz-1]
            alt[nz-1,j,i] = (RD/P0)*(t[nz-1,j,i] + T0)*(((p[nz-1,j,i] + pb[nz-1,j,i])/P0)**(-1/GAMMA))
            al[nz-1,j,i] = alt[nz-1,j,i] - alb[nz-1,j,i]        

            ### Compute fields down the column
            for k in range(nz-2,-1,-1):
                p[k,j,i] = p[k+1,j,i] - mu[j,i]*dn[k+1]
                alt[k,j,i] = (RD/P0)*(t[k,j,i] + T0)*(((p[k,j,i] + pb[k,j,i])/P0)**(-1/GAMMA))
                al[k,j,i] = alt[k,j,i] - alb[k,j,i]

            ### Compute geopotential
            for k in range(1,nz+1):
                ph[k,j,i] = ph[k-1,j,i] - dnw[k-1]*((mub[j,i] + mu[j,i])*al[k-1,j,i] + \
                        mu[j,i]*alb[k-1,j,i])

            ### Interpolate u and v    
            p_surf_val = interp_0(p_ex, z_ex, 0, nz+1)
            for k in range(nz):
                p_level = znu[k]*(p_surf_val - p_top[j,i]) + p_top[j,i]
                u[k,j,i] = interp_0(u1, pd1, p_level, nz)
                v[k,j,i] = interp_0(v1, pd1, p_level, nz)

    ### Zonal periodicity      
    for j in range(ny+1):
        for k in range(0,nz):
            v[k,j,nx-1] = v[k,j,0]

    for j in range(ny):
        for k in range(nz):
            u[k,j,nx] = u[k,j,0]         

    tsk = np.zeros((ny,nx))
    for j in range(0,ny):
        for i in range(0,nx):
            thtmp = t[0,j,i] + T0
            ptmp = p[0,j,i] + pb[0,j,i]
            tsk[j,i] = thtmp*((ptmp/P0)**(RD/CP))

    return u, v, ph, p, mu, t, tsk
