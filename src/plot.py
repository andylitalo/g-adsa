# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:09:14 2019
Functions for plotting in G-ADSA analysis iPython Jupyter notebooks.
@author: Andy
"""

# standard libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# custom libraries
import dataproc



def dft(d_dft, x, y, xlabel, ylabel, title, cmap_type='brg', cmap_max=300):
    tag_list = list(d_dft.keys())
    # creates color list from blue to red
    cmap = cm.get_cmap(cmap_type)
    skip = int( cmap_max / max(len(tag_list), 1) ) # max prevents divide by zero
    colors = [cmap(i) for i in range(0, cmap_max, skip)]
    ax = None
    for i, tag in enumerate(tag_list):
        ax = plot_line(d_dft[tag][x], d_dft[tag][y], ax=ax, xlabel=xlabel,
                            ylabel=ylabel, title=title, 
                            color=colors[i], label=tag, marker='', lw=2)
    plt.legend(fontsize=14)
    
    return ax


def diffusivity_sqrt(i, p_set_arr, t_mp1, t_fit, t0, w_gas_act, w_fit, w0, a, polyol, T):
    """Plots diffusivity scatterplot with square-root fit."""
    p_set = p_set_arr[i]
    # identify whether system was adsorbing or desorbing CO2
    is_adsorbing = i <= np.argmax(p_set_arr) and p_set_arr[i] != 0
    sign = 2*(is_adsorbing-0.5)
    stage = 'Adsorption' if is_adsorbing else 'Desorption'
    # plot data translated such that first point is 0,0 and data increases (so t^1/2 looks like a straight line on log-log)
    ax = plot_line(t_mp1-t0, sign*(w_gas_act-w0), marker='^', label='data', xlog=True, ylog=True, xlabel='t [s]',
                  ylabel=r'$\Delta w_{CO2}$ [g]', title=stage + ' of CO2 in %s polyol at p = %d kPa, %d C' % (polyol, p_set, T))
    plot_line(t_fit-t0, sign*(w_fit-w0), ax=ax, lw=2, color='r', marker=None,
              label='{a:.1e}(t-{t0:.1e})^(1/2) + {w0:.1e}'.format(a=a, w0=w0, t0=t0))
    plt.legend(loc='best')

    
def diffusivity_exp(i, p_set_arr, t_mp1, t_fit, i0, w_gas_2_plot, w_fit_2_plot,
                    a, b, c, polyol, T, is_adsorbing):
    """Plots diffusivity scatterplot with exponential fit."""
    p_set = p_set_arr[i]
    # identify whether system was adsorbing or desorbing CO2
    stage = 'Adsorption' if is_adsorbing else 'Desorption'
    # plot data translated such that first point is 0,0 and data increases (so t^1/2 looks like a straight line on log-log)
    ax = plot_line(t_mp1, w_gas_2_plot, marker='^', label='data', ylog=True, xlabel='t [s]',
                  ylabel=r'$M_{\infty} - M_{CO2} / M_{\infty} - M_0$ [g]',
                  title=stage + ' of CO2 in %s polyol at p = %d kPa, %d C' % (polyol, p_set, T))
    plot_line(t_fit+t_mp1[i0], w_fit_2_plot, ax=ax, lw=2, color='r', marker=None,
              label=r'{a:.1e}exp({b:.1e}t); $M_\infty=${c:.1e}' \
                          .format(a=a, b=b, c=c))
    plt.legend(loc='best')


def get_f(tag):
    f, mw = get_f_mw(tag)
    return f


def get_f_mw(tag):
    # extracts molecular weight (different if written explicitly before '-' or written with 'k')
    if '-' in tag:
        i_dash = tag.find('-')
        i__ = tag[:i_dash].rfind('_')
        mw = int(tag[i__+1:i_dash])
    elif 'p' == tag[0]:
        i_k = tag.find('k')
        i__ = tag.find('_')
        mw = 1000*int(tag[i__+1:i_k])
    else:
        i_k = tag.find('k')
        mw = 1000*int(tag[:i_k]) # converts from kg/mol -> g/mol
       
    # extracts functionality
    i_f = tag.rfind('f')
    f = int(tag[i_f-1:i_f])
    
    return f, mw


def get_f_per_mw(tag):
    f, mw = get_f_mw(tag)    
    # computes functionality per molecular weight
    f_per_mw = f / mw
    
    return f_per_mw



def get_mw(tag):
    f, mw = get_f_mw(tag)
    return mw


def henrys_const(d, csv_file_list, cmap, d_marker, force_origin=True,
                             p_thresh_ideal=1000, ax=None, by_wt=False,
                             save_path=None, n_entries_col=12, ms=8,
                             d_fill={}, x_label='f/Mw', color_var='T'):
    """
    Plots Henry's constant vs. functionality per molecular weight.
    
    """
    # defines dictionaries of x axis functions based on the label
    d_x = {'f/Mw':get_f_per_mw, 'f':get_f, 'Mw':get_mw}
    # calculates range of temperatures for determining coloring
    T_list = [d[tag]['T'] for tag in csv_file_list]
    T_min = np.min(T_list)
    T_max = np.max(T_list)
    # calculates range of functionality for determining coloring
    f_list = [d_x['f'](tag) for tag in csv_file_list]
    f_min = 2 # np.min(f_list)
    f_max = 6 #np.max(f_list)
    
    # determines units
    if by_wt:
        units = 'w/(w.Pa)'
    else:
        units = r'mmol/($m^3$ Pa)'

    # computes Henry's constant under each condition
    ct = 0
    for tag in csv_file_list:
        # raises to the minimum pressure if it is higher than the prescribed ideal threshold 
        p_thresh = max(p_thresh_ideal, np.nanmin(d[tag]['p']))
        try:
            # increase threshold for pressure to 1000 kPa for fitting of Henry's constant
            H, s_H = dataproc.compute_henrys_const(d[tag]['p'], 
                                                   d[tag]['solub'], 
                                                   d[tag]['spec_vol'], 
                                                   p_thresh=p_thresh, 
                                                  s_solub=d[tag]['s_solub'], 
                                                  s_spec_vol=d[tag]['s_spec_vol'], 
                                                  force_origin=force_origin, 
                                                  by_wt=by_wt)
        except:
            print('Henry''s constant computation failed for {0:s}'.format(tag))
            continue
        
        # convert to mmol if not by weight (i.e., by mmol/m^3.Pa)
        if not by_wt:
            H *= 1000
            s_H *= 1000
        
        # plot
        x = d_x[x_label](tag)
        T = d[tag]['T']
        # determines color
        if color_var == 'T':
            color = cmap((T - T_min)/(T_max - T_min))
        elif color_var == 'f':
            f = d_x['f'](tag)
            color = cmap((f - f_min)/(f_max - f_min))
        else:
            print('{0:s} is an invalid color_var'.format(color_var))
            color='r'
            
        polyol = d[tag]['polyol']
        marker = d_marker[polyol]
        if polyol in d_fill:
            fillstyle = d_fill[polyol]
        else:
            fillstyle = 'none'
        if ax is not None:
            ax = plot_errorbars(x, H, s_H, color=color, 
                                marker=marker, ms=ms, fillstyle=fillstyle,
                                xlabel=x_label, ylabel=r'$H$ [' + units + ']', 
                                label=tag, ax=ax,
                                title='Henry\'s Constant vs. ' + x_label)
   
        else:
            ax = plot_errorbars(x, H, s_H, color=color, 
                                marker=marker, ms=ms, fillstyle=fillstyle,
                                label=tag, ax=ax)
            
        ct += 1
                
    # put legend outside of plot box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    legend_x = 1
    legend_y = 0.5
    # arranges legend into columns
    ncol = int(ct/n_entries_col + 1)
    plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y), 
               ncol=ncol)
    ax.set_xscale('log')
    
    # saves figure
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        
    return ax
    
    
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


def plot_errorbars(x, y, yerr, ax=None, xlabel='', ylabel='', marker='o',
                   title='', tk_fs=14, ax_fs=16, t_fs=20, color='b', ms=8,
                   fillstyle='none', label='', xlog=False, ylog=False, 
                   xlim=[], ylim=[]):
    """Plots errorbars on points."""
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # adsorption
    ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, ms=ms,
                fillstyle=fillstyle, ls='', label=label)
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
    ax.tick_params(axis='both', labelsize=tk_fs)
    plt.legend(loc='best')

    return ax


def plot_errorbars_ads_des(x, y, yerr, p_set_arr, T, ax=None, xlabel='', ylabel='',
                           title='', ax_fs=16, t_fs=20, color='b', label_tag='',
                           xlog=False, ylog=False, xlim=[], ylim=[], ms=6, m_ads='o', m_des='x'):
    """Plots errorbars on points."""
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # find midpoint where adsorption switches to desorption
    midpt = np.where(p_set_arr == np.max(p_set_arr))[0][0]
    # adsorption
    ax.errorbar(x[:midpt], y[:midpt], yerr=yerr[:midpt], color=color, marker=m_ads, ms=ms,
                fillstyle='none', ls='', label='%.1f C (ads) %s' % (T, label_tag))
    # desorption
    ax.errorbar(x[midpt:], y[midpt:], yerr=yerr[midpt:], color=color, marker=m_des, ms=ms,
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


def plot_errorbars_ads_des_stat_sys(x, y, yerr_stat, yerr_sys, p_set_arr, T, ax=None,
                            xlabel='', ylabel='',
                           title='', ax_fs=16, t_fs=20, color='b', label_tag='',
                           xlog=False, ylog=False, xlim=[], ylim=[]):
    """Plots errorbars on points."""
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # find midpoint where adsorption switches to desorption
    midpt = np.where(p_set_arr == np.max(p_set_arr))[0][0]
    # plot adsorption
    yerr_ads = [yerr_stat[:midpt], yerr_sys[:midpt]]
    ax.errorbar(x[:midpt], y[:midpt], yerr=yerr_ads, color=color, marker='o',
                fillstyle='none', ls='', label='%.1f C (ads) %s' % (T, label_tag),
                capsize=2)
    # plot desorption
    yerr_des = [yerr_stat[midpt:], yerr_sys[midpt:]]
    ax.errorbar(x[midpt:], y[midpt:], yerr=yerr_des, color=color, marker='x',
                ls='', label='%.1f C (des) %s' % (T, label_tag), capsize=2)
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


def species_solub(d, csv_file_list, species_tag, species_name, data_folder,
                  cmap, ms=8, ax=None):
    """
    Plots solubility of a given species.
    """
    
    T_list = [d[tag]['T'] for tag in csv_file_list]
    T_min = np.min(T_list)
    T_max = np.max(T_list)
    for tag in csv_file_list:
        # only considers data for the given species
        if species_tag not in tag:
            continue
        T = d[tag]['T']
        color = cmap((T_max - T)/(T_max - T_min))
        if ax is None:
            ax = plot_errorbars(d[tag]['p'], d[tag]['solub'], d[tag]['s_solub'], 
                                color=color, xlabel='p [kPa]', ms=ms,
                                ylabel='solubility [w/w]', 
                                title='Solubility vs. p, CO2 in {0:s}'.format(species_name),
                                label='{0:d} C'.format(int(d[tag]['T'])))
        else:
            ax = plot_errorbars(d[tag]['p'], d[tag]['solub'], d[tag]['s_solub'], 
                                    color=color, ax=ax, ms=ms,
                                    label='{0:d} C'.format(int(d[tag]['T'])))
    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])
    plt.legend(fontsize=12)# save figure
    
    # save figure
    if data_folder is not None:
        plt.savefig(data_folder + 'solub_{0:s}.pdf'.format(species_tag), bbox_inches='tight')
    
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


def sensitivity_manual(d, d_dft, tag_meas, prop, param_varied, param_fit_val, data_folder, fig_name, tags_2_plot, 
                       solub=False, cmap_max=300, cmap_type='brg', param_split_char='~', ax=None,
                       val_split_char='_', dec_pt_char='-', color='k', ms=6, m_ads='o', m_des='x', lw=1):
    """
    Plots the changes in DFT predictions due to variations of a PC-SAFT parameter
    and compares to experimental results. Must manually choose the parameters to plot.
    
    split_char is character that splits names of multiple varied parameters.
    
    """
        
    # labeling dictionaries
    y_label_dict = {'if_tension':'Interfacial Tension [mN/m]', 'solub':'Solubility [w/w]', 'spec_vol':'Specific Volume [mL/g]'}
    title_dict = {'if_tension':'Interfacial Tension', 'solub':'Solubility', 'spec_vol':'Specific Volume'}
    legend_dict = {'epsn':r'$\epsilon$', 'sigma':r'$\sigma$', 'N':r'$N$', 'T':r'$T$'}
    str_code_dict = {'epsn':'.1f', 'sigma':'.2f', 'N':'.0f', 'T':'.2f'}
    
    def lgd_label(tag):
        """Creates the legend label for the given tag."""
        lgd = ''
        # splits tag into the individual variables that are varied
        param_list = tag.split(param_split_char)
        for param in param_list:
            name, val_str = param.split(val_split_char)
            ones, dec = val_str.split(dec_pt_char)
            val = float(ones) + float(dec)/10**len(dec)
            lgd += ''.join((legend_dict[name], ' = {0:', str_code_dict[name], '} ')).format(val)

        return lgd

    # determines if plot should have solubility or pressure as x axis
    x = 'solub' if solub else 'p'
    x_label = r'$w_{CO2}$ [w/w]' if solub else 'pressure [kPa]'
    # plots the measured interfacial tension first in black markers
    ax = plot_errorbars_ads_des(d[tag_meas][x], d[tag_meas][prop], d[tag_meas]['s_' + prop], d[tag_meas]['p'], d[tag_meas]['T'], 
                                             color=color, xlabel=x_label, ylabel=y_label_dict[prop], ax=ax,
                                             title='Sensitivity of ' + title_dict[prop], label_tag=tag_meas[:4],
                               ms=ms, m_ads=m_ads, m_des=m_des)
    
    # plots DFT prediction with PC-SAFT parameters fitted to measurements of solubility in black line
    lgd = ''.join((legend_dict[param_varied], ' = {0:', str_code_dict[param_varied], '} (fit)')).format(param_fit_val)
    ax.plot(d_dft[tag_meas][x], d_dft[tag_meas][prop], color=color, label=lgd, lw=lw)
    
    # parses the tags into values
    tag_lgd_list = [(tag, lgd_label(tag)) for tag in tags_2_plot]
    # creates color list from blue to red
    cmap = cm.get_cmap(cmap_type)
    skip = int( cmap_max / max(len(tag_lgd_list), 1) ) # prevents divide by zero
    colors = [cmap(i) for i in range(0, cmap_max, skip)]
    # pairs up colors with tag and number
    tag_lgd_color_list = zip(tag_lgd_list, colors)
    
    # plots perturbed values
    for tag_lgd_color in tag_lgd_color_list:
        # separately parses tag and legend label because they are in a tuple of their own
        tag_lgd, color = tag_lgd_color
        tag, lgd = tag_lgd
        ax.plot(d_dft[tag][x], d_dft[tag][prop], color=color, label=lgd, lw=lw)

    # puts legend outside of plot box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.65, box.height])
    legend_x = 1
    legend_y = 0.5
    plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))
    
    # save figure
    if len(fig_name) > 0:
        plt.savefig(data_folder + fig_name, bbox_inches='tight')
        
    return ax


def sensitivity(d, d_dft, tag_meas, prop, param_varied, param_fit_val, data_folder, fig_name, solub=False, 
                cmap_max=300, cmap_type='brg', param_split_char='~', val_split_char='_', dec_pt_char='-'):
    """Helper function that automatically selects the tags to plot. Only plots tags with one parameter varied."""
    # gets tags for the variable to plot (ignores cases where two or more variables were varied, indicated by split_char)
    tags_2_plot = [tag for tag in d_dft.keys() if ((param_varied in tag) and (param_split_char not in tag))]
    
    return sensitivity_manual(d, d_dft, tag_meas, prop, param_varied, param_fit_val, data_folder, fig_name, tags_2_plot, 
                       solub=solub, cmap_max=cmap_max, cmap_type=cmap_type, param_split_char=param_split_char,
                       val_split_char=val_split_char, dec_pt_char=dec_pt_char)