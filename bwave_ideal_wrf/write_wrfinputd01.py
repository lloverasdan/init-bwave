#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Written by Daniel J. Lloveras
"""

import numpy as np

#%% Constants
RD = 287
FF = 1e-4

def write(ncfile, nx, ny, nz, hres_m, title_str, time_str, u, v, t, ph, phb,\
          t_init, mu, mub, p, pb, fnm, fnp, rdnw, rdn, dnw, dn,\
          cfn, cfn1, rdx, rdy, cf1, cf2, cf3, moist, znw, znu, ptop, tsk):
    
    ### Define dimensions
    ncfile.createDimension('Time',None)
    ncfile.createDimension('DateStrLen',19)
    ncfile.createDimension('west_east',nx)
    ncfile.createDimension('south_north',ny)
    ncfile.createDimension('bottom_top',nz)
    ncfile.createDimension('west_east_stag',nx+1)
    ncfile.createDimension('south_north_stag',ny+1)
    ncfile.createDimension('bottom_top_stag',nz+1)
    
    ### Define attributes
    ncfile.TITLE = title_str
    ncfile.START_DATE = time_str
    ncfile.SIMULATION_START_DATE = time_str
    ncfile.WEST_EAST_GRID_DIMENSION = nx+1
    ncfile.SOUTH_NORTH_GRID_DIMENSION = ny+1
    ncfile.BOTTOM_TOP_GRID_DIMENSION = nz+1
    ncfile.DX = hres_m
    ncfile.DY = hres_m
    ncfile.WEST_EAST_PATCH_START_UNSTAG = 1
    ncfile.WEST_EAST_PATCH_END_UNSTAG = nx
    ncfile.WEST_EAST_PATCH_START_STAG = 1
    ncfile.WEST_EAST_PATCH_END_STAG = nx+1
    ncfile.SOUTH_NORTH_PATCH_START_UNSTAG = 1
    ncfile.SOUTH_NORTH_PATCH_END_UNSTAG = ny
    ncfile.SOUTH_NORTH_PATCH_START_STAG = 1
    ncfile.SOUTH_NORTH_PATCH_END_STAG = ny+1
    ncfile.BOTTOM_TOP_PATCH_START_UNSTAG = 1
    ncfile.BOTTOM_TOP_PATCH_END_UNSTAG = nz
    ncfile.BOTTOM_TOP_PATCH_START_STAG = 1
    ncfile.BOTTOM_TOP_PATCH_END_STAG = nz+1
    ncfile.MMINLU = "USGS"
    ncfile.ISWATER = 0
    ncfile.ISLAKE = 0
    ncfile.ISICE = 0
    ncfile.ISURBAN = 0
    ncfile.ISOILWATER = 0
    
    ### Rename attributes to include hyphens
    ncfile.renameAttribute('WEST_EAST_GRID_DIMENSION','WEST-EAST_GRID_DIMENSION')
    ncfile.renameAttribute('WEST_EAST_PATCH_START_UNSTAG','WEST-EAST_PATCH_START_UNSTAG')
    ncfile.renameAttribute('WEST_EAST_PATCH_END_UNSTAG','WEST-EAST_PATCH_END_UNSTAG')
    ncfile.renameAttribute('WEST_EAST_PATCH_START_STAG','WEST-EAST_PATCH_START_STAG')
    ncfile.renameAttribute('WEST_EAST_PATCH_END_STAG','WEST-EAST_PATCH_END_STAG')
    ncfile.renameAttribute('SOUTH_NORTH_GRID_DIMENSION','SOUTH-NORTH_GRID_DIMENSION')
    ncfile.renameAttribute('SOUTH_NORTH_PATCH_START_UNSTAG','SOUTH-NORTH_PATCH_START_UNSTAG')
    ncfile.renameAttribute('SOUTH_NORTH_PATCH_END_UNSTAG','SOUTH-NORTH_PATCH_END_UNSTAG')
    ncfile.renameAttribute('SOUTH_NORTH_PATCH_START_STAG','SOUTH-NORTH_PATCH_START_STAG')
    ncfile.renameAttribute('SOUTH_NORTH_PATCH_END_STAG','SOUTH-NORTH_PATCH_END_STAG')
    ncfile.renameAttribute('BOTTOM_TOP_GRID_DIMENSION','BOTTOM-TOP_GRID_DIMENSION')
    ncfile.renameAttribute('BOTTOM_TOP_PATCH_START_UNSTAG','BOTTOM-TOP_PATCH_START_UNSTAG')
    ncfile.renameAttribute('BOTTOM_TOP_PATCH_END_UNSTAG','BOTTOM-TOP_PATCH_END_UNSTAG')
    ncfile.renameAttribute('BOTTOM_TOP_PATCH_START_STAG','BOTTOM-TOP_PATCH_START_STAG')
    ncfile.renameAttribute('BOTTOM_TOP_PATCH_END_STAG','BOTTOM-TOP_PATCH_END_STAG')
    
    ### Define variables
    
    ### Time
    Times = ncfile.createVariable('Times','S1',('Time','DateStrLen'))
    
    ### Land use category
    LU_INDEX = ncfile.createVariable('LU_INDEX','f4',('Time','south_north','west_east'))
    LU_INDEX.FieldType = 104
    LU_INDEX.MemoryOrder = "XY "
    LU_INDEX.description = "LAND USE CATEGORY"
    LU_INDEX.units = ""
    LU_INDEX.stagger = ""
    LU_INDEX.coordinates = "XLONG XLAT"
    
    ### Eta values on half (mass) levels
    ZNU = ncfile.createVariable('ZNU','f4',('Time','bottom_top'))
    ZNU.FieldType = 104
    ZNU.MemoryOrder = "Z  "
    ZNU.description = "eta values on half (mass) levels"
    ZNU.units = ""
    ZNU.stagger = ""
    
    ### Eta values on full (w) levels
    ZNW = ncfile.createVariable('ZNW','f4',('Time','bottom_top_stag'))
    ZNW.FieldType = 104
    ZNW.MemoryOrder = "Z  "
    ZNW.description = "eta values on full (w) levels"
    ZNW.units = ""
    ZNW.stagger = "Z"

    ### X-wind component
    U = ncfile.createVariable('U','f4',('Time','bottom_top','south_north','west_east_stag'))
    U.FieldType = 104
    U.MemoryOrder = "XYZ"
    U.description = "x-wind component"
    U.units = "m s-1"
    U.stagger = "X"
    U.coordinates = "XLONG_U XLAT_U"
    
    ### Y-wind component
    V = ncfile.createVariable('V','f4',('Time','bottom_top','south_north_stag','west_east'))
    V.FieldType = 104
    V.MemoryOrder = "XYZ"
    V.description = "y-wind component"
    V.units = "m s-1"
    V.stagger = "Y"
    V.coordinates = "XLONG_V XLAT_V"
    
    ### Perturbation geopotential
    PH = ncfile.createVariable('PH','f4',('Time','bottom_top_stag','south_north','west_east'))
    PH.FieldType = 104
    PH.MemoryOrder = "XYZ"
    PH.description = "perturbation geopotential"
    PH.units = "m2 s-2"
    PH.stagger = "Z"
    PH.coordinates = "XLONG XLAT"
    
    ### Base-state geopotential
    PHB = ncfile.createVariable('PHB','f4',('Time','bottom_top_stag','south_north','west_east'))
    PHB.FieldType = 104
    PHB.MemoryOrder = "XYZ"
    PHB.description = "base-state geopotential"
    PHB.units = "m2 s-2"
    PHB.stagger = "Z"
    PHB.coordinates = "XLONG XLAT"
    
    ### Perturbation potential temperature (theta - t0)
    T = ncfile.createVariable('T','f4',('Time','bottom_top','south_north','west_east'))
    T.FieldType = 104
    T.MemoryOrder = "XYZ"
    T.description = "perturbation potential temperature (theta-t0)"
    T.units = "K"
    T.stagger = ""
    T.coordinates = "XLONG XLAT"
    
    ### Initial potential temperature
    T_INIT = ncfile.createVariable('T_INIT','f4',('Time','bottom_top','south_north','west_east'))
    T_INIT.FieldType = 104
    T_INIT.MemoryOrder = "XYZ"
    T_INIT.description = "initial potential temperature"
    T_INIT.units = "K"
    T_INIT.stagger = ""
    T_INIT.coordinates = "XLONG XLAT"
    
    ### Perturbation dy air mass in column
    MU = ncfile.createVariable('MU','f4',('Time','south_north','west_east'))
    MU.FieldType = 104
    MU.MemoryOrder = "XY "
    MU.description = "perturbation dry air mass in column"
    MU.units = "Pa"
    MU.stagger = ""
    MU.coordinates = "XLONG XLAT"
    
    ### Base state dry air mass in column
    MUB = ncfile.createVariable('MUB','f4',('Time','south_north','west_east'))
    MUB.FieldType = 104
    MUB.MemoryOrder = "XY "
    MUB.description = "base state dry air mass in column"
    MUB.units = "Pa"
    MUB.stagger = ""
    MUB.coordinates = "XLONG XLAT"
    
    ### Perturbation pressure
    P = ncfile.createVariable('P','f4',('Time','bottom_top','south_north','west_east'))
    P.FieldType = 104
    P.MemoryOrder = "XYZ"
    P.description = "perturbation pressure"
    P.units = "Pa"
    P.stagger = ""
    P.coordinates = "XLONG XLAT"
    
    ### Base state pressure
    PB = ncfile.createVariable('PB','f4',('Time','bottom_top','south_north','west_east'))
    PB.FieldType = 104
    PB.MemoryOrder = "XYZ"
    PB.description = "BASE STATE PRESSURE"
    PB.units = "Pa"
    PB.stagger = ""
    PB.coordinates = "XLONG XLAT"
    
    ### Upper weight for vertical stretching
    FNM = ncfile.createVariable('FNM','f4',('Time','bottom_top'))
    FNM.FieldType = 104
    FNM.MemoryOrder = "Z  "
    FNM.description = "upper weight for vertical stretching"
    FNM.units = ""
    FNM.stagger = ""
    
    ### Lower weight for vertical stretching
    FNP = ncfile.createVariable('FNP','f4',('Time','bottom_top'))
    FNP.FieldType = 104
    FNP.MemoryOrder = "Z  "
    FNP.description = "lower weight for vertical stretching"
    FNP.units = ""
    FNP.stagger = ""
    
    ### Inverse d(eta) values between full (w) levels
    RDNW = ncfile.createVariable('RDNW','f4',('Time','bottom_top'))
    RDNW.FieldType = 104
    RDNW.MemoryOrder = "Z  "
    RDNW.description = "inverse d(eta) values between full (w) levels"
    RDNW.units = ""
    RDNW.stagger = ""
    
    ### Inverse d(eta) values between half (mass) levels
    RDN = ncfile.createVariable('RDN','f4',('Time','bottom_top'))
    RDN.FieldType = 104
    RDN.MemoryOrder = "Z  "
    RDN.description = "inverse d(eta) values between half (mass) levels"
    RDN.units = ""
    RDN.stagger = ""
    
    ### d(eta) values between full (w) levels
    DNW = ncfile.createVariable('DNW','f4',('Time','bottom_top'))
    DNW.FieldType = 104
    DNW.MemoryOrder = "Z  "
    DNW.description = "d(eta) values between full (w) levels"
    DNW.units = ""
    DNW.stagger = ""
    
    ### d(eta) values between half (mass) levels
    DN = ncfile.createVariable('DN','f4',('Time','bottom_top'))
    DN.FieldType = 104
    DN.MemoryOrder = "Z  "
    DN.description = "d(eta) values between half (mass) levels"
    DN.units = ""
    DN.stagger = ""
    
    ### Inverse x grid length
    RDX = ncfile.createVariable('RDX','f4','Time')
    RDX.FieldType = 104
    RDX.MemoryOrder = "0  "
    RDX.description = "INVERSE X GRID LENGTH"
    RDX.units = ""
    RDX.stagger = ""
    
    ### Inverse y grid length
    RDY = ncfile.createVariable('RDY','f4','Time')
    RDY.FieldType = 104
    RDY.MemoryOrder = "0  "
    RDY.description = "INVERSE Y GRID LENGTH"
    RDY.units = ""
    RDY.stagger = ""
    
    ### Extrapolation constant
    CFN = ncfile.createVariable('CFN','f4','Time')
    CFN.FieldType = 104
    CFN.MemoryOrder = "0  "
    CFN.description = "extrapolation constant"
    CFN.units = ""
    CFN.stagger = ""
    
    ### 1 - extrapolation constant
    CFN1 = ncfile.createVariable('CFN1','f4','Time')
    CFN1.FieldType = 104
    CFN1.MemoryOrder = "0  "
    CFN1.description = "extrapolation constant"
    CFN1.units = ""
    CFN1.stagger = ""
    
    ### 2nd order extrapolation constant
    CF1 = ncfile.createVariable('CF1','f4','Time')
    CF1.FieldType = 104
    CF1.MemoryOrder = "0  "
    CF1.description = "2nd order extrapolation constant"
    CF1.units = ""
    CF1.stagger = ""
    
    ### 2nd order extrapolation constant
    CF2 = ncfile.createVariable('CF2','f4','Time')
    CF2.FieldType = 104
    CF2.MemoryOrder = "0  "
    CF2.description = "2nd order extrapolation constant"
    CF2.units = ""
    CF2.stagger = ""
    
    ### 2nd order extrapolation constant
    CF3 = ncfile.createVariable('CF3','f4','Time')
    CF3.FieldType = 104
    CF3.MemoryOrder = "0  "
    CF3.description = "2nd order extrapolation constant"
    CF3.units = ""
    CF3.stagger = ""
    
    ### Water vapor mixing ratio
    QVAPOR = ncfile.createVariable('QVAPOR','f4',('Time','bottom_top','south_north','west_east'))
    QVAPOR.FieldType = 104
    QVAPOR.MemoryOrder = "XYZ"
    QVAPOR.description = "Water vapor mixing ratio"
    QVAPOR.units = "kg kg-1"
    QVAPOR.stagger = ""
    QVAPOR.coordinates = "XLONG XLAT"
    
    ### Map scale factor on mass grid
    MAPFAC_M = ncfile.createVariable('MAPFAC_M','f4',('Time','south_north','west_east'))
    MAPFAC_M.FieldType = 104
    MAPFAC_M.MemoryOrder = "XY "
    MAPFAC_M.description = "Map scale factor on mass grid"
    MAPFAC_M.units = ""
    MAPFAC_M.stagger = ""
    MAPFAC_M.coordinates = "XLONG XLAT"
    
    ### Map scale factor on u-grid
    MAPFAC_U = ncfile.createVariable('MAPFAC_U','f4',('Time','south_north','west_east_stag'))
    MAPFAC_U.FieldType = 104
    MAPFAC_U.MemoryOrder = "XY "
    MAPFAC_U.description = "Map scale factor on u-grid"
    MAPFAC_U.units = ""
    MAPFAC_U.stagger = "X"
    MAPFAC_U.coordinates = "XLONG_U XLAT_U"
    
    ### Map scale factor on v-grid
    MAPFAC_V = ncfile.createVariable('MAPFAC_V','f4',('Time','south_north_stag','west_east'))
    MAPFAC_V.FieldType = 104
    MAPFAC_V.MemoryOrder = "XY "
    MAPFAC_V.description = "Map scale factor on v-grid"
    MAPFAC_V.units = ""
    MAPFAC_V.stagger = "Y"
    MAPFAC_V.coordinates = "XLONG_V XLAT_V"
    
    ### Map scale factor on mass grid, x direction
    MAPFAC_MX = ncfile.createVariable('MAPFAC_MX','f4',('Time','south_north','west_east'))
    MAPFAC_MX.FieldType = 104
    MAPFAC_MX.MemoryOrder = "XY "
    MAPFAC_MX.description = "Map scale factor on mass grid, x direction"
    MAPFAC_MX.units = ""
    MAPFAC_MX.stagger = ""
    MAPFAC_MX.coordinates = "XLONG XLAT"
    
    ### Map scale factor on mass grid, y direction
    MAPFAC_MY = ncfile.createVariable('MAPFAC_MY','f4',('Time','south_north','west_east'))
    MAPFAC_MY.FieldType = 104
    MAPFAC_MY.MemoryOrder = "XY "
    MAPFAC_MY.description = "Map scale factor on mass grid, y direction"
    MAPFAC_MY.units = ""
    MAPFAC_MY.stagger = ""
    MAPFAC_MY.coordinates = "XLONG XLAT"
    
    ### Map scale factor on u-grid, x direction
    MAPFAC_UX = ncfile.createVariable('MAPFAC_UX','f4',('Time','south_north','west_east_stag'))
    MAPFAC_UX.FieldType = 104
    MAPFAC_UX.MemoryOrder = "XY "
    MAPFAC_UX.description = "Map scale factor on u-grid, x direction"
    MAPFAC_UX.units = ""
    MAPFAC_UX.stagger = "X"
    MAPFAC_UX.coordinates = "XLONG_U XLAT_U"
    
    ### Map scale factor on u-grid, y direction
    MAPFAC_UY = ncfile.createVariable('MAPFAC_UY','f4',('Time','south_north','west_east_stag'))
    MAPFAC_UY.FieldType = 104
    MAPFAC_UY.MemoryOrder = "XY "
    MAPFAC_UY.description = "Map scale factor on u-grid, x direction"
    MAPFAC_UY.units = ""
    MAPFAC_UY.stagger = "X"
    MAPFAC_UY.coordinates = "XLONG_U XLAT_U"
    
    ### Map scale factor on v-grid, x direction
    MAPFAC_VX = ncfile.createVariable('MAPFAC_VX','f4',('Time','south_north_stag','west_east'))
    MAPFAC_VX.FieldType = 104
    MAPFAC_VX.MemoryOrder = "XY "
    MAPFAC_VX.description = "Map scale factor on v-grid, x direction"
    MAPFAC_VX.units = ""
    MAPFAC_VX.stagger = "Y"
    MAPFAC_VX.coordinates = "XLONG_V XLAT_V"
    
    ### Inverse map scale factor on v-grid, x direction
    MF_VX_INV = ncfile.createVariable('MF_VX_INV','f4',('Time','south_north_stag','west_east'))
    MF_VX_INV.FieldType = 104
    MF_VX_INV.MemoryOrder = "XY "
    MF_VX_INV.description = "Inverse map scale factor on v-grid, x direction"
    MF_VX_INV.units = ""
    MF_VX_INV.stagger = "Y"
    MF_VX_INV.coordinates = "XLONG_V XLAT_V"
    
    ### Map scale factor on v-grid, x direction
    MAPFAC_VY = ncfile.createVariable('MAPFAC_VY','f4',('Time','south_north_stag','west_east'))
    MAPFAC_VY.FieldType = 104
    MAPFAC_VY.MemoryOrder = "XY "
    MAPFAC_VY.description = "Map scale factor on v-grid, x direction"
    MAPFAC_VY.units = ""
    MAPFAC_VY.stagger = "Y"
    MAPFAC_VY.coordinates = "XLONG_V XLAT_V"
    
    ### Coriolis sine latitude term
    F = ncfile.createVariable('F','f4',('Time','south_north','west_east'))
    F.FieldType = 104
    F.MemoryOrder = "XY "
    F.description = "Coriolis sine latitude term"
    F.units = "s-1"
    F.stagger = ""
    F.coordinates = "XLONG XLAT"
    
    ### Pressure top of the model
    P_TOP = ncfile.createVariable('P_TOP','f4','Time')
    P_TOP.FieldType = 104
    P_TOP.MemoryOrder = "0  "
    P_TOP.description = "PRESSURE TOP OF THE MODEL"
    P_TOP.units = "Pa"
    P_TOP.stagger = ""
    
    ### Surface skin temperature
    TSK = ncfile.createVariable('TSK','f4',('Time','south_north','west_east'))
    TSK.FieldType = 104
    TSK.MemoryOrder = "XY "
    TSK.description = "SURFACE SKIN TEMPERATURE"
    TSK.units = "K"
    TSK.stagger = ""
    TSK.coordinates = "XLONG XLAT"
    
    ### Land mask, x direction
    XLAND = ncfile.createVariable('XLAND','f4',('Time','south_north','west_east'))
    XLAND.FieldType = 104
    XLAND.MemoryOrder = "XY "
    XLAND.description = "LAND MASK (1 FOR LAND, 2 FOR WATER)"
    XLAND.units = ""
    XLAND.stagger = ""
    XLAND.coordinates = "XLONG XLAT"
    
    ### Hydrostatic pressure
    P_HYD = ncfile.createVariable('P_HYD','f4',('Time','bottom_top','south_north','west_east'))
    P_HYD.FieldType = 104
    P_HYD.MemoryOrder = "XYZ"
    P_HYD.description = "hydrostatic pressure"
    P_HYD.units = "Pa"
    P_HYD.stagger = ""
    P_HYD.coordinates = "XLONG XLAT"

    ### Assign values to variables
    
    time_char = list(time_str)
    Times[0,:] = time_char[:]
    LU_INDEX[0,:,:] = np.ones((ny,nx))*16
    ZNU[0,:] = znu[:] 
    ZNW[0,:] = znw[:] 
    U[0,:,:,:] = u[:,:,:]
    V[0,:,:,:] = v[:,:,:]
    PH[0,:,:,:] = ph
    PHB[0,:,:,:] = phb
    T[0,:,:,:] = t
    T_INIT[0,:,:,:] = t_init
    MU[0,:,:] = mu
    MUB[0,:,:] = mub
    P[0,:,:,:] = p
    PB[0,:,:,:] = pb
    FNM[0,:] = fnm
    FNP[0,:] = fnp
    RDNW[0,:] = rdnw
    RDN[0,:] = rdn
    DNW[0,:] = dnw
    DN[0,:] = dn
    CFN[0] = cfn
    CFN1[0] = cfn1
    RDX[0] = rdx
    RDY[0] = rdy
    CF1[0] = cf1
    CF2[0] = cf2
    CF3[0] = cf3
    QVAPOR[0,:,:,:] = moist
    MAPFAC_M[0,:,:] = np.ones((ny,nx))
    MAPFAC_U[0,:,:] = np.ones((ny,nx+1))
    MAPFAC_V[0,:,:] = np.ones((ny+1,nx))
    MAPFAC_MX[0,:,:] = np.ones((ny,nx))
    MAPFAC_MY[0,:,:] = np.ones((ny,nx))
    MAPFAC_UX[0,:,:] = np.ones((ny,nx+1))
    MAPFAC_UY[0,:,:] = np.ones((ny,nx+1))
    MAPFAC_VX[0,:,:] = np.ones((ny+1,nx))
    MF_VX_INV[0,:,:] = np.ones((ny+1,nx))
    MAPFAC_VY[0,:,:] = np.ones((ny+1,nx))
    F[0,:,:] = np.ones((ny,nx))*FF
    P_TOP[0] = int(ptop)
    TSK[0,:,:] = tsk[:,:]
    XLAND[0,:,:] = np.ones((ny,nx))*2
    P_HYD[0,:,:,:] = pb

    return ncfile
