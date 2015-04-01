# -*- coding: utf-8 -*-
"""
This module contains functions for loading ands plotting data.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

Re_list = [8e4, 1.1e5, 1.3e5, 1.6e5, 1.9e5, 2.1e5, 2.4e5, 2.7e5, 2.9e5, 3.2e5,
           3.4e5]

def load(Re, foil="0020"):
    """Loads airfoil data for a given Reynolds number."""
    fname = "NACA {}_T1_Re{:.3f}_M0.00_N9.0.dat".format(foil, Re/1e6)
    fpath = "data/{}/{}".format(foil, fname)
    aoa, cl, cd = np.loadtxt(fpath, skiprows=14, unpack=True)
    aoa = np.append([0], aoa[:-1])
    cl = np.append([0], cl[:-1])
    cd = np.append(cd[0], cd[:-1])
    df = pd.DataFrame()
    df["aoa"] = aoa
    df["cl"] = cl
    df["cd"] = cd
    return df
    
def plot_cl_cd(Re, foil="0020", newfig=True):
    """Plots lift over drag ratio for a given Reynolds number."""
    if newfig:
        plt.figure()
    df = load(Re, foil)
    plt.plot(df.aoa, df.cl/df.cd)
    plt.xlabel("Angle of attack (deg)")
    plt.ylabel("$C_L/C_D$")
    
def plot_cl_cd_all(foil="0020"):
    plt.figure()
    for Re in Re_list:
        plot_cl_cd(Re, foil, newfig=False)
    plt.xlim((0,30))
        
def plot_max_cl():
    max_cl = []
    for Re in Re_list:
        max_cl.append(load(Re)["cl"].max())
    plt.figure()
    plt.plot(Re_list, max_cl, "-o")
    plt.xlabel("$Re_c$")
    plt.ylabel(r"$C_{l, \mathrm{max}}$")
#    plt.ylim((1.1,1.22))
    
def plot_min_cd():
    min_cd = []
    for Re in Re_list:
        min_cd.append(load(Re)["cd"].min())
    plt.figure()
    plt.plot(Re_list, min_cd, "-o")
    plt.xlabel("$Re_c$")
    plt.ylabel(r"$C_{d, \mathrm{min}}$")
#    plt.ylim((0.005,0.03))
    
def plot_max_cl_cd():
    vals = []
    for Re in Re_list:
        df = load(Re)
        vals.append(np.max(df.cl/df.cd))
    plt.figure()
    plt.plot(Re_list, vals, "-o")
    plt.xlabel("$Re_c$")
    plt.ylabel(r"$(C_l/C_d)_\mathrm{max}$")
#    plt.ylim((25,60))
    ax = plt.gca()
    ax.xaxis.major.formatter.set_powerlimits((0,0)) 
    plt.tight_layout()
    
def plot_aoa_max_cl_cd():
    """Plots angle of attack at which max C_l/C_d occurs."""
    vals = []
    for Re in Re_list:
        df = load(Re)
        clcd = df.cl/df.cd
        ind = np.where(clcd==clcd.max())[0][0]
        vals.append(df.aoa[ind])
    plt.figure()
    plt.plot(Re_list, vals, "-o")
    plt.xlabel("$Re_c$")
    plt.ylabel(r"$\alpha_{C_l/C_d, \mathrm{max}}$")
#    plt.ylim((6,10))
    
def plot_ct(Re, foil="0020", newfig=True):
    """Plots tangential coefficient for a given Reynolds number."""
    if newfig:
        plt.figure()
    df = load(Re, foil)
    aoa_rad = df.aoa*np.pi/180
    ct = np.sqrt(df.cl**2 + df.cd**2)*np.sin(aoa_rad - np.arctan2(df.cd,df.cl))
    # Equivalent expression below!
    # ct = df.cl*np.sin(aoa_rad) - df.cd*np.cos(aoa_rad)
    plt.plot(df.aoa, ct)
    plt.xlabel("Angle of attack (deg)")
    plt.ylabel("$C_T$")
    
def plot_ct_all(foil="0020"):
    plt.figure()
    for Re in Re_list:
        plot_ct(Re, foil, newfig=False)
    plt.xlim((0,30))
    
if __name__ == "__main__":
    plot_cl_cd_all("4520")
#    plot_max_cl()
#    plot_min_cd()
#    plot_max_cl_cd()
#    plot_aoa_max_cl_cd()
#    plot_ct(1.1e5)
#    plot_cl_cd(1.1e5)
    plot_ct_all("0020")
    plot_ct_all("4520")