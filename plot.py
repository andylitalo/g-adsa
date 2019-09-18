# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:09:14 2019
Functions for plotting in G-ADSA analysis iPython Jupyter notebooks.
@author: Andy
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_line(x, y, ax=None, xlabel='', ylabel='', title='', marker='.', lw=0,
              ax_fs=16, t_fs=20, color='b', label=None, xlog=False, ylog=False,
              ls='-'):
    """Plot a single line."""
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(x, y, marker=marker, color=color, label=label, lw=lw, ls=ls)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel, fontsize=ax_fs)
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel, fontsize=ax_fs)
    if len(title) > 0:
        ax.set_title(title, fontsize=t_fs)

    return ax


def plot_errorbars_ads_des(x, y, yerr, p_set_arr, T, ax=None, xlabel='', ylabel='',
                           title='', ax_fs=16, t_fs=20, color='b', label_tag='',
                           xlog=False, ylog=False, xlim=[], ylim=[]):
    """Plots errorbars on points."""
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # find midpoint where adsorption switches to desorption
    midpt = np.where(p_set_arr == np.max(p_set_arr))[0][0]
    # adsorption
    ax.errorbar(x[:midpt], y[:midpt], yerr=yerr[:midpt], color=color, marker='o',
                fillstyle='none', ls='', label='%.1f C (ads) %s' % (T, label_tag))
    ax.errorbar(x[midpt:], y[midpt:], yerr=yerr[midpt:], color=color, marker='x',
                ls='', label='%.1f C (des) %s' % (T, label_tag))
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel, fontsize=ax_fs)
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel, fontsize=ax_fs)
    if len(title) > 0:
        ax.set_title(title, fontsize=t_fs)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if len(xlim) == 2:
        ax.set_xlim(xlim)
    if len(ylim) == 2:
        ax.set_ylim(ylim)
    plt.legend(loc='best')

    return ax


def plot_two_axes(x, y1, y2, x2=[], markers=['o', '^'], labels=['1', '2'],
                  colors=['r', 'b'], ms=8, ax=None, xlabel='', ylabels=[],
                           title='', ax_fs=16, t_fs=20, tk_fs=14, lw=0,
                           figsize=(6,4)):
    """Plots with two axes."""
    # create plot for pressure
    fig, ax1 = plt.subplots(figsize=figsize)
    # plot first axis
    ax1.plot(x, y1, marker=markers[0], ms=ms, label=labels[0], color=colors[0],
             lw=lw)
    # labels
    if len(xlabel) > 0:
        ax1.set_xlabel(xlabel, fontsize=ax_fs)
    if len(ylabels) == 2:
        ax1.set_ylabel(ylabels[0], color=colors[0], fontsize=ax_fs)
    ax1.tick_params('y', colors=colors[0])
    if len(title) > 0:
        ax1.set_title(title, fontsize=t_fs)

    # separate axis for gas mass
    ax2 = ax1.twinx()
    # plot second axis
    if len(x2) > 0:
        ax2.plot(x2, y2, marker=markers[1], ms=ms, label=labels[1],
                 color=colors[1], lw=lw)
    else:
        ax2.plot(x, y2, marker=markers[1], ms=ms, label=labels[1],
                 color=colors[1], lw=lw)
    # labels
    if len(ylabels) == 2:
        ax2.set_ylabel(ylabels[1], color=colors[1], fontsize=ax_fs)
    ax2.tick_params('y', colors=colors[1])

    # increase font size of tick labels
    ax1.tick_params(axis='both', which='major', labelsize=tk_fs)
    ax2.tick_params(axis='y', which='major', labelsize=tk_fs)
    fig.tight_layout()

    return ax1
