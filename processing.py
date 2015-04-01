# -*- coding: utf-8 -*-
"""
This module contains functions for loading ands plotting data.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.style.use("arial")

Re_list = [8e4, 1.1e5, 1.3e5, 1.6e5, 1.9e5, 2.1e5, 2.4e5, 2.7e5, 2.9e5, 3.2e5,
           3.4e5]

def load(Re, foil="0020"):
    """Loads airfoil data for a given Reynolds number."""
    fname = "NACA {}_T1_Re{:.3f}_M0.00_N9.0.dat".format(foil, Re/1e6)
    fpath = "data/{}/{}".format(foil, fname)
    aoa, cl, cd = np.loadtxt(fpath, skiprows=14, unpack=True)
    if aoa[0] != 0.0:
        aoa = np.append([0.0], aoa[:-1])
        cl = np.append([0.0], cl[:-1])
        cd = np.append(cd[0.0], cd[:-1])
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
        
def plot_max_cl(foil="0020"):
    max_cl = []
    for Re in Re_list:
        max_cl.append(load(Re, foil=foil)["cl"].max())
    plt.figure()
    plt.plot(Re_list, max_cl, "-o")
    plt.xlabel("$Re_c$")
    plt.ylabel(r"$C_{l, \mathrm{max}}$")
#    plt.ylim((1.1,1.22))
    
def plot_min_cd(foil="0020"):
    min_cd = []
    for Re in Re_list:
        min_cd.append(load(Re, foil=foil)["cd"].min())
    plt.figure()
    plt.plot(Re_list, min_cd, "-o")
    plt.xlabel("$Re_c$")
    plt.ylabel(r"$C_{d, \mathrm{min}}$")
#    plt.ylim((0.005,0.03))
    
def plot_max_cl_cd(foil="0020"):
    vals = []
    for Re in Re_list:
        df = load(Re, foil=foil)
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
    
def lookup(aoa_deg, Re, foil="0020"):
    aoa_deg = np.asarray(aoa_deg)
    df = load(Re, foil)
    df["aoa_rad"] = df.aoa/180*np.pi
    f_cl = interp1d(df.aoa, df.cl)
    f_cd = interp1d(df.aoa, df.cd)
    f_ct = interp1d(df.aoa, df.cl*np.sin(df.aoa_rad) - df.cd*np.cos(df.aoa_rad))
    cl, cd, ct = f_cl(aoa_deg), f_cd(aoa_deg), f_ct(aoa_deg)
    return {"cl": cl, "cd": cd, "ct": ct}
    
def calc_cft_ctorque(Re, tsr=1.9, chord=0.14, R=0.5, foil="0020"):
    U_infty = 1.0
    omega = tsr*U_infty/R
    theta_blade_deg = np.arange(0, 181)
    theta_blade_rad = theta_blade_deg/180.0*np.pi
    blade_vel_mag = omega*R
    blade_vel_x = blade_vel_mag*np.cos(theta_blade_rad)
    blade_vel_y = blade_vel_mag*np.sin(theta_blade_rad)
    u = U_infty # No induction
    rel_vel_mag = np.sqrt((blade_vel_x + u)**2 + blade_vel_y**2)
    rel_vel_x = u + blade_vel_x
    rel_vel_y = blade_vel_y
    relvel_dot_bladevel = (blade_vel_x*rel_vel_x + blade_vel_y*rel_vel_y)
    alpha_rad = np.arccos(relvel_dot_bladevel/(rel_vel_mag*blade_vel_mag))
    alpha_deg = alpha_rad*180/np.pi
    ctorque = lookup(alpha_deg, Re, foil)["ct"]*chord/(2*R)*rel_vel_mag**2/U_infty**2
    df = pd.DataFrame()
    df["theta"] = theta_blade_deg
    df["alpha_deg"] = alpha_deg
    df["rel_vel_mag"] = rel_vel_mag
    df["ctorque"] = ctorque
    return df
    
def calc_cft_re_dep(tsr=1.9, chord=0.14, R=0.5, foil="0020"):
    max_ctorque = []
    min_ctorque = []
    for Re in Re_list:
        df = calc_cft_ctorque(Re, tsr, chord, R, foil)
        max_ctorque.append(df.ctorque.max())
        min_ctorque.append(df.ctorque.min())
    return max_ctorque
        
def plot_cft_re_dep(tsr=1.9, chord=0.14, R=0.5, foil="0020", newfig=True,
                    fmt="-ok"):
    max_ctorque = calc_cft_re_dep(tsr, chord, R, foil)
    if newfig:
        plt.figure()
    plt.plot(Re_list, max_ctorque, fmt, label="NACA {}".format(foil))
    plt.grid()
    ax = plt.gca()
    ax.xaxis.major.formatter.set_powerlimits((0,0))
    plt.xlabel(r"$Re_c$")
    plt.ylabel(r"Max geometric torque coeff.")
    plt.tight_layout()
    
def plot_cft_re_dep_all(tsr=1.9, chord=0.14, R=0.5):
    plt.figure()
    for foil, fmt in zip(["0020", "2520", "4520"], ["-ok", "-sk", "-^k"]):
        plot_cft_re_dep(tsr=tsr, chord=chord, R=R, foil=foil, newfig=False,
                        fmt=fmt)
    plt.legend(loc="best")
    
def plot_cft_ctorque(Re, tsr=1.9, chord=0.14, R=0.5, foil="0020"):
    df = calc_cft_ctorque(Re, tsr, chord, R, foil)
    plt.figure(figsize=(11,3.25))
    plt.subplot(1, 3, 1)
    plt.plot(df.theta, df.alpha_deg, "k")
    plt.xlabel("Azimuthal angle (degrees)")
    plt.ylabel("Angle of attack (degrees)")
    plt.xticks(np.arange(0, 181, 30))
    plt.grid()
    plt.subplot(1, 3, 2)
    plt.plot(df.theta, df.rel_vel_mag**2, "k")
    plt.xlabel("Azimuthal angle (degrees)")
    plt.ylabel(r"$|U_{\mathrm{rel}}|^2/U_\infty^2$")
    plt.xticks(np.arange(0, 181, 30))
    plt.grid()
    plt.subplot(1, 3, 3)
    plt.plot(df.theta, df.ctorque, "k")
    plt.xlabel("Azimuthal angle (degrees)")
    plt.ylabel("Torque coefficient")
    plt.xticks(np.arange(0, 181, 30))
    plt.grid()
    plt.tight_layout(pad=0.2)
    plt.show()
    
if __name__ == "__main__":
    foil = "2520"
#    plot_cl_cd_all("4520")
    plot_max_cl(foil)
    plot_min_cd(foil)
    plot_max_cl_cd(foil)
#    plot_aoa_max_cl_cd()
#    plot_ct(1.1e5)
#    plot_cl_cd(1.1e5)
    plot_ct_all("4520")
    plot_cft_ctorque(2.1e5, foil=foil)
#    plot_cft_re_dep(foil=foil)
    plot_cft_re_dep_all()