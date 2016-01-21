# -*- coding: utf-8 -*-
"""This module contains functions for loading and plotting data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pxl.styleplot import set_sns
import os

Re_list = [8e4, 1.1e5, 1.3e5, 1.6e5, 1.9e5, 2.1e5, 2.4e5, 2.7e5, 2.9e5, 3.2e5,
           3.4e5]

def load(Re, foil="0020"):
    """Load airfoil data for a given Reynolds number."""
    fname = "NACA {}_T1_Re{:.3f}_M0.00_N9.0.dat".format(foil, Re/1e6)
    fpath = "data/{}/{}".format(foil, fname)
    aoa, cl, cd = np.loadtxt(fpath, skiprows=14, unpack=True)
    if aoa[0] != 0.0:
        aoa = np.append([0.0], aoa[:-1])
        cl = np.append([0.0], cl[:-1])
        cd = np.append(cd[0], cd[:-1])
    df = pd.DataFrame()
    df["aoa"] = aoa
    df["cl"] = cl
    df["cd"] = cd
    return df

def plot_cl(Re, foil="0020", newfig=True):
    """Plot lift coefficient for a given Reynolds number and profile."""
    if newfig:
        plt.figure()
    df = load(Re, foil)
    plt.plot(df.aoa, df.cl, label="{:.1e}".format(Re))
    plt.xlabel("Angle of attack (deg)")
    plt.ylabel("$C_l$")

def plot_cl_all(foil="0020"):
    plt.figure()
    for Re in Re_list:
        plot_cl(Re, foil, newfig=False)
    plt.xlim((0,30))

def plot_cd(Re, foil="0020", newfig=True):
    """
    Plot drag coefficient for a given Reynolds number and profile.
    """
    if newfig:
        plt.figure()
    df = load(Re, foil)
    plt.plot(df.aoa, df.cd)
    plt.xlabel("Angle of attack (deg)")
    plt.ylabel("$C_d$")

def plot_cd_all(foil="0020"):
    plt.figure()
    for Re in Re_list:
        plot_cd(Re, foil, newfig=False)
    plt.xlim((0,30))


def plot_cl_cd(Re, foil="0020", newfig=True):
    """Plot lift over drag ratio for a given Reynolds number."""
    if newfig:
        plt.figure()
    df = load(Re, foil)
    plt.plot(df.aoa, df.cl/df.cd)
    plt.xlabel("Angle of attack (deg)")
    plt.ylabel("$C_l/C_d$")


def plot_cl_cd_all(foil="0020"):
    plt.figure()
    for Re in Re_list:
        plot_cl_cd(Re, foil, newfig=False)
    plt.xlim((0,30))


def plot_max_cl(foil="0020", normalize=False, newfig=True, **kwargs):
    max_cl = []
    ylab = r"$C_{l_{\max}}$"
    for Re in Re_list:
        max_cl.append(load(Re, foil=foil)["cl"].max())
    if newfig:
        plt.figure()
    if normalize:
        max_cl = np.asarray(max_cl)
        max_cl /= max_cl[5]
        ylab += " (normalized)"
    plt.plot(Re_list, max_cl, label="NACA " + foil, **kwargs)
    plt.grid(True)
    plt.xlabel("$Re_c$")
    plt.ylabel(ylab)


def plot_max_cl_all(save=False, newfig=True, legend=True):
    if newfig:
        plt.figure()
    for foil, marker in zip(["0020", "2520", "4520"], ["v", "s", "^"]):
        plot_max_cl(foil=foil, newfig=False, normalize=True, marker=marker)
    if legend:
        plt.legend(loc="best")
    if save:
        plt.savefig("figures/foils_max_cl.pdf")


def plot_min_cd(foil="0020", newfig=True, normalize=False, **kwargs):
    ylab = r"$C_{d_{\min}}$"
    min_cd = []
    for Re in Re_list:
        min_cd.append(load(Re, foil=foil)["cd"].min())
    if newfig:
        plt.figure()
    if normalize:
        ylab += " (normalized)"
        min_cd = np.asarray(min_cd)
        min_cd /= min_cd[5]
    plt.plot(Re_list, min_cd, label="NACA " + foil, **kwargs)
    plt.xlabel("$Re_c$")
    plt.ylabel(ylab)
    plt.grid(True)


def plot_min_cd_all(save=False, newfig=True, legend=True):
    if newfig:
        plt.figure()
    for foil, marker in zip(["0020", "2520", "4520"], ["v", "s", "^"]):
        plot_min_cd(foil=foil, newfig=False, normalize=True, marker=marker)
    if legend:
        plt.legend(loc="best")
    if save:
        plt.savefig("figures/foils_min_cd.pdf")


def plot_max_cl_cd(foil="0020", newfig=True, normalize=False, **kwargs):
    ylab = r"$(C_l/C_d)_\mathrm{max}$"
    vals = []
    for Re in Re_list:
        df = load(Re, foil=foil)
        vals.append(np.max(df.cl/df.cd))
    if newfig:
        plt.figure()
    if normalize:
        ylab += " (normalized)"
        vals = np.asarray(vals)
        vals /= vals[5]
    plt.plot(Re_list, vals, label="NACA " + foil, **kwargs)
    plt.xlabel("$Re_c$")
    plt.ylabel(ylab)
    plt.grid(True)
    plt.tight_layout()


def plot_max_cl_cd_all(save=False, newfig=True, legend=True):
    if newfig:
        plt.figure()
    for foil, marker in zip(["0020", "2520", "4520"], ["v", "s", "^"]):
        plot_max_cl_cd(foil=foil, newfig=False, normalize=True, marker=marker)
    if legend:
        plt.legend(loc="best")
    if save:
        plt.savefig("figures/foils_max_cl_cd.pdf")


def plot_aoa_max_cl_cd():
    """Plot angle of attack at which max C_l/C_d occurs."""
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
    foil_coeffs = lookup(alpha_deg, Re, foil)
    ctorque = foil_coeffs["ct"]*chord/(2*R)*rel_vel_mag**2/U_infty**2
    cdx = -foil_coeffs["cd"]*np.sin(np.pi/2 - alpha_rad + theta_blade_rad)
    clx = foil_coeffs["cl"]*np.cos(np.pi/2 - alpha_rad - theta_blade_rad)
    df = pd.DataFrame()
    df["theta"] = theta_blade_deg
    df["alpha_deg"] = alpha_deg
    df["rel_vel_mag"] = rel_vel_mag
    df["ctorque"] = ctorque
    df["cdrag"] = clx + cdx
    return df

def calc_aft_ctorque(Re, foil="0020"):
    """
    Calculates the approximate torque coefficient for an AFT blade.
    """
    alpha = np.linspace(0, 22, num=100)
    pitch = np.linspace(-90, 90, num=361)
    alpha_rad = alpha*np.pi/180.0
    pitch_rad = pitch*np.pi/180.0
    ctorque = np.zeros((len(alpha), len(pitch)))
    for n, ai in enumerate(alpha_rad):
        coeffs = lookup(ai/np.pi*180.0, Re, foil=foil)
        ctorque[n, :] = coeffs["cl"]*np.sin(pitch_rad + ai) \
                      - coeffs["cd"]*np.cos(pitch_rad + ai)
    # df = pd.DataFrame(data=ctorque, index=alpha, columns=pitch)
    ind = np.where(ctorque==ctorque.max())
    i, j = ind[0][0], ind[1][0]
    print(alpha[i], pitch[j])
    tsr = 1/np.tan(alpha_rad[i] + pitch_rad[j])
#    print(tsr)
    return ctorque.max()

def calc_cft_re_dep(tsr=1.9, chord=0.14, R=0.5, foil="0020"):
    max_ctorque = []
    min_ctorque = []
    max_cdrag = []
    for Re in Re_list:
        df = calc_cft_ctorque(Re, tsr, chord, R, foil)
        max_ctorque.append(df.ctorque.max())
        min_ctorque.append(df.ctorque.min())
        max_cdrag.append(df.cdrag.max())
    return {"max_ctorque": np.asarray(max_ctorque),
            "max_cdrag": np.asarray(max_cdrag)}

def calc_aft_re_dep(foil="0020"):
    ctorque = []
    for Re in Re_list:
        ctorque.append(calc_aft_ctorque(Re, foil))
    return np.asarray(ctorque)

def plot_cft_re_dep(tsr=1.9, chord=0.14, R=0.5, foil="0020", newfig=True,
                    **kwargs):
    d = calc_cft_re_dep(tsr, chord, R, foil)
    max_ctorque = d["max_ctorque"]
    max_cdrag = d["max_cdrag"]
    if newfig:
        plt.figure()
    plt.plot(Re_list, max_ctorque/max_ctorque[5], label="NACA {}".format(foil),
             **kwargs)
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.major.formatter.set_powerlimits((0,0))
    plt.xlabel(r"$Re_c$")
    plt.ylabel(r"$C_{T_\mathrm{max}}$ (normalized)")
    plt.tight_layout()

def plot_aft_re_dep(foil="0020", fmt="-ok", newfig=True):
    ct = calc_aft_re_dep(foil)
    if newfig:
        plt.figure()
    plt.plot(Re_list, ct/ct[5], fmt, label="NACA {}".format(foil),
             markerfacecolor="none")
    plt.grid(True)
    plt.xlabel(r"$Re_c$")
    plt.ylabel(r"$C_{T_\mathrm{max}}$ (normalized)")
    plt.tight_layout()

def plot_cft_re_dep_all(tsr=1.9, chord=0.14, R=0.5, RVAT=True, save=False):
    plt.figure()
    for foil, marker in zip(["0020", "2520", "4520"], ["v", "s", "^"]):
        plot_cft_re_dep(tsr=tsr, chord=chord, R=R, foil=foil, newfig=False,
                        marker=marker)
    if RVAT:
        plot_rvat_re_dep(marker="o")
    plt.legend(loc="best")
    if save:
        plt.savefig("figures/cft_re_dep_foils.pdf")


def plot_cft_ctorque(Re, tsr=1.9, chord=0.14, R=0.5, foil="0020", save=False):
    df = calc_cft_ctorque(Re, tsr, chord, R, foil)
    plt.figure(figsize=(7.5, 2.5))
    plt.subplot(1, 3, 1)
    plt.plot(df.theta, df.alpha_deg)
    plt.xlabel("Azimuthal angle (deg.)")
    plt.ylabel("Angle of attack (deg.)")
    plt.xticks(np.arange(0, 181, 30))
    plt.grid(True)
    label_subplot(text="(a)")
    plt.subplot(1, 3, 2)
    plt.plot(df.theta, df.rel_vel_mag)
    plt.xlabel("Azimuthal angle (deg.)")
    plt.ylabel(r"$|U_{\mathrm{rel}}|/U_\infty$")
    plt.xticks(np.arange(0, 181, 30))
    plt.grid(True)
    label_subplot(text="(b)")
    plt.subplot(1, 3, 3)
    plt.plot(df.theta, df.ctorque)
    plt.xlabel("Azimuthal angle (deg.)")
    plt.ylabel("Torque coefficient")
    plt.xticks(np.arange(0, 181, 30))
    plt.grid(True)
    label_subplot(text="(c)")
    plt.tight_layout(pad=0.2)
    if save:
        plt.savefig("figures/foil_kinematics_ct.pdf")


def plot_rvat_re_dep(newfig=False, normalize=True, **kwargs):
    if newfig:
        plt.figure()
    fp = "C:/Users/Pete/Research/Experiments/RVAT Re dep/Data/Processed/Perf-tsr_0.csv"
    df = pd.read_csv(fp)
    cp = df.mean_cp
    if normalize:
        cp /= cp[5]
    plt.plot(df.Re_c_ave, cp, label="UNH-RVAT exp.", **kwargs)


def plot_all_foils_re_dep(save=False):
    plt.figure(figsize=(7.5, 2.65))
    plt.subplot(1, 3, 1)
    plot_max_cl_all(newfig=False, legend=True)
    label_subplot(text="(a)")
    plt.subplot(1, 3, 2)
    plot_min_cd_all(newfig=False, legend=False)
    label_subplot(text="(b)")
    plt.subplot(1, 3, 3)
    plot_max_cl_cd_all(newfig=False, legend=False)
    label_subplot(text="(c)")
    plt.tight_layout(pad=0.2)
    if save:
        plt.savefig("figures/all_foils_re_dep.pdf")
    plt.show()


def label_subplot(ax=None, x=0.5, y=-0.28, text="(a)", **kwargs):
    """Create a subplot label."""
    if ax is None:
        ax = plt.gca()
    ax.text(x=x, y=y, s=text, transform=ax.transAxes,
            horizontalalignment="center", verticalalignment="top", **kwargs)


if __name__ == "__main__":
    if not os.path.isdir("figures"):
        os.mkdir("figures")
    set_sns()
    foil = "0020"
    save = True
    # plot_cl_all(foil)
    # plt.legend()
    # plot_cd_all(foil)
    # plot_max_cl(foil)
    # plot_min_cd(foil)
    # plot_max_cl_cd(foil)
    # plot_aoa_max_cl_cd()
    # plot_ct(1.1e5)
    # plot_cl_cd(1.1e5)
    # plot_ct_all("4520")
    plot_cft_ctorque(2.1e5, foil=foil, save=save)
    # plot_cft_re_dep(foil=foil)
    plot_cft_re_dep_all(save=save)
    # plot_min_cd_all()
    # plot_max_cl_all()
    # plot_max_cl_cd_all()
    plot_all_foils_re_dep(save=save)
    # plot_aft_re_dep()
    plt.show()
