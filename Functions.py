import os
import sys
import tempfile
import argparse
import subprocess
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time

''' --- Pseudospectral Method for the right hand side of the Cahn Hilliard equation --- '''
def ps(c, k, D, k2, kappa, a, b, R2, mu2, B):
    ck = np.fft.fft2(c)
    c3k = np.fft.fft2(c*c*c)
    result_k_Diff = D * ((- kappa * k2 * k2 + b * k2) * ck - a * k2 * c3k)
    result_k_React = - k * kappa * k2 * ck
    result_x_React = - k * ( a * (c - c_c) * (c - c_c) * (c - c_c) - b * (c - c_c) + B) + R2 * mu2
    result = np.fft.ifft2(result_k_Diff + result_k_React).real + result_x_React
    return result

''' --- Non-Dimensional Form of the Cahn Hilliard equation (still with Ginzburg-Landau functional) --- '''
def psDL(c, lamtot, lam2, k2, mu2, B, c_c):
    ck = np.fft.fft2(c)
    c3k = np.fft.fft2(c*c*c)
    result_k_Diff = ((- (1. / 8.) * k2 * k2 + (1. / 4.) * k2) * ck - k2 * c3k)
    result_k_React = - (1. / 8.) * k2 * ck
    result_x_React = - lamtot * ( c * c * c - (1. / 4.) * c + B) + lam2 * mu2
    result = np.fft.ifft2(result_k_Diff).real + lamtot * np.fft.ifft2(result_k_React).real + result_x_React
    return result

''' --- Non-Dimensional Form of the Reaction Diffusion equation --- '''
def RDDL(c, kin, kout, cin, cout, gammain, gammaout, k2, a, c_cin, c_cout, c_c):
    ci = c - c_c
    ck = np.fft.fft2(ci)
    c3k = np.fft.fft2(ci*ci*ci)
    result_k = - k2 * ck
    result_k_Diff = ((- (1. / 8.) * k2 * k2 + (1. / 4.) * k2) * ck - k2 * c3k)
    condlist = [c < c_cout, (c >= c_cout) & (c <= c_cin),c > c_cin]
    funclist = [lambda x: gammaout - kout * (x - cout), lambda x: a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3, lambda x: -gammain - kin * (x - cin)]
    s = np.piecewise(c, condlist, funclist)
    result = np.fft.ifft2(result_k_Diff).real + s
    return result, s

''' --- pseudospectral method for the right hand side of the Florence-Huggins Free energy --- '''
def psFH(c, k, k2, gamma, Chi):
    lnck = np.fft.fft2((np.log(c / (1-c))))
    ck = np.fft.fft2(c)
    result_k = (-kappa * k2 * k2 - 2. * Chi * k2) * ck
    result = np.fft.ifft2(result_k).real
    return result

''' --- Matrix form for the finite difference scheme --- '''
def FDMatrix(c, A, L, Nx, Ny, dx, dy):
    np.reshape(c, (Nx * Ny))
    result = np.zeros(( Nx, Ny))
    
    d2xin = 1. / (dx * dx)
    d2yin = 1. / (dy * dy)
    d4xin = d2xin * d2xin
    d4yin = d2yin * d2yin
    d2c1 = (c * A) * d2xin + (c * A)
    d2c3 = np.dot((c * c * c),A) * d2xin
    d4c = 1

''' --- Interpolates a Cubic function, used for the approximate Reaction Rates between the equilibrium concentrations --- '''
def Spline(nuin, nuout, kin, kout, cin, cout, c_cin, c_cout):
    a = nuout - kout * (c_cout - cout)
    b = -nuin - kin * (c_cin - cin)
    c = -kout
    d = -kin
    print(a,b,c,d)
    a0 = b - 0.5625 * c - 0.1875 * d
    a1 = 9. * (a - b) + 3.75 * c + 1.75 * d
    a2 = -24. * (a - b) - 7. * d - 5. * d
    a3 = 16. * (a - b) + 4. * (c + d)
    return a0,a1,a2,a3

''' --- defines the interpolation for the Reaction rates between the two equilibrium concentrations for the whole model --- ''' 
def Lambda(c, a, b, k, l):
    result1 = a * c + b
    result2 = k * c + l
    return result1, result2

''' --- Calculates the Volume fraction of the interface depending on the set threshold --- '''
def Interface(c, Nx, Ny, c0, th):
    result = np.zeros((Nx,Ny))
    IntFrac = 0
    for i in range(Nx):
        for j in range(Ny):
            if ((abs(c[i][j]) < (c0 * th))):
                result[i][j] = 1
                IntFrac = IntFrac + result[i][j]
    IntFrac = float(IntFrac) / float((Nx * Ny))
    return result, IntFrac              

''' --- calculate the flux of a scalar field --- '''
def flux(c, Nx, Ny, dx, dy):
    resultx, resulty=np.zeros((Nx-1,Ny-1)), np.zeros((Nx-1,Ny-1))
    for x in range (Nx-1):
        for y in range (Ny-1):
            resultx[x][y]=-(c[x+1][y]-c[x][y])/dx
            resulty[x][y]=-(c[x][y+1]-c[x][y])/dy
    return resultx, resulty

''' --- calculates the second derivative of the concentration in Fourier space to get the chemical potential --- '''
def mu(c, k2, kappa, a, b, B, c_c):
    ck = np.fft.fft2(c)
    result_k =  (1. / 8.) * k2 * ck
    resultgrad = np.fft.ifft2(result_k).real
    result = c * c * c - (1. / 4.) * c + B
    return result, resultgrad

''' --- Convert the continous concentration field to a binary field, depending on the threshold --- '''
def Filter(c, th):
    condlist = [c < (th*np.amax(c)), c >= (th*np.amax(c))]
    funclist = [lambda x: 0, lambda x: 1]
    result = np.piecewise(c, condlist, funclist)
    return result