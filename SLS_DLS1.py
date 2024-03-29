#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:18:26 2019
@author: chiappisil
"""
from math import radians, exp
from scipy.special import gamma
from scipy.stats import linregress
import os as os
import shutil
import numpy as np
import sys
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from lmfit import minimize, Parameters, fit_report
import SLS_DLS2 as dls


RR = 1.35e-5 #cm-1, Rayleigh ratio of toluene, the standard used for renormalizing. 

def extract_data_file(data_path, filename):
    ''' Extracts from each file the values of temperature, angle, countrates and monitor intensity'''
    # T = '' #Sample temperature in Kelvin
    # Angle = '' #Measurement angle in degree
    # CR0 = '' #Mean countrate of channel 0
    # CR1 = '' #Mean countrate of channel 1
    # Imon = '' #Value of the monitor diode
    # print(sample_info)
    # data_path = sample_info['data_path']
    with open(os.path.join(data_path, filename), "r", encoding='latin1') as f:
        for index, line in enumerate(f):
            if line.startswith('Temp'):
                T = float(line.split(':')[1])
            if line.startswith('Date'):
                Date = line.split(':')[1].split('"')[1].split('"')[0]
            if line.startswith('Time'):
                Time = line.split(' :')[1].split('"')[1].split('"')[0]
            if line.startswith('Angle'):
                Angle = float(line.split(':')[1]) 
            if line.startswith('MeanCR0'):
                CR0 = float(line.split(':')[1])
            if line.startswith('MeanCR1'):    
                CR1 = float(line.split(':')[1])
            if line.startswith('Wavelength'):    
                Wavelength = float(line.split(':')[1])
            if line.startswith('Monitor'):
                Imon = float(line.split('Diode')[1])
                Monitor_diode_line = index
            if line.startswith('"Count Rate"'):
                Count_rate_line = index  
            if line.startswith('"Correlation"'):
                Correlation_line = index
                
    # print(os.path.join(data_path, filename))     
    CRs = np.loadtxt(os.path.join(data_path, filename), skiprows=Count_rate_line+2, encoding='latin1', usecols=(0,1,2), max_rows=Monitor_diode_line-Count_rate_line-3)
    mean_CR0 = np.average(CRs[:,1])
    mean_CR1 = np.average(CRs[:,2])
    temp = np.loadtxt(os.path.join(data_path, filename), skiprows=Correlation_line+1, encoding='latin1', usecols=(0,1,2), max_rows=Count_rate_line-Correlation_line-3)
    g2s = pd.DataFrame(temp, columns=['tau', 'g2_1','g2_2']).set_index('tau')
    g2s['g2_average'] = np.average([g2s['g2_1'].values, g2s['g2_2'].values], axis=0)
    g2s['g2_std'] = np.std([g2s['g2_1'].values, g2s['g2_2'].values], axis=0)
    
    
    return {'Filename': filename,
            'T': T, 
            'Angle': Angle,
            'Imon': Imon, 
            'CR0': CR0,
            'CR1':CR1,
            'Date_Time': Date + ' ' + Time,
#            'Time': Time,
#            'Date': Date,
            'Wavelength': Wavelength,
            'mean_CR0':  mean_CR0,
            'mean_CR1':  mean_CR1,
            'CRs': CRs,
            'g2s':g2s}


def group_data_files_by_temperature(df, tolerance):
    """
    Groups data files in a folder based on their temperature and moves them to new folders.

    Args:
        df (pandas.DataFrame): DataFrame containing experimental parameters, including temperature and filenames.
        tolerance (float): Tolerance value used for grouping data files.

    Returns:
        None
    """
    # Create a set of unique temperatures in the DataFrame
    temperatures = set(df['T'])

    # Create a new folder for each unique temperature
    for temp in temperatures:
        # Round the temperature to the nearest multiple of the tolerance value
        rounded_temp = round(temp / tolerance) * tolerance
        
        # Create a new folder for the rounded temperature
        folder_name = f"{rounded_temp:.1f}"
        os.makedirs(folder_name, exist_ok=True)

        # Get a list of filenames corresponding to the current temperature
        filenames = df[df['T'] == temp]['Filename']

        # Move each file to the new folder
        for filename in filenames:
            shutil.move(filename, folder_name)

    

def extract_data(sample_info):
    ''' Extracts all information from the datafiles contained in the folder data_path, and returns
    a dictionary where each element is a dictionary will all the informations conatained for each
    file in the folder. '''
    
    data_path = sample_info['data_path']
    data = {}
    for file in sorted(os.listdir(data_path)):
        if file.endswith(".ASC"):
            data[file] =  extract_data_file(data_path, file)    
    
    parameters = ['Filename', 'T', 'Angle', 'Imon', 'CR0', 'CR1', 'mean_CR0', 'mean_CR1', 'Date_Time', 'Wavelength'] #parameters found in the summary pandas dataframe
    data_summary =  pd.DataFrame(columns=parameters)
    # print(data_summary)      
    
    for key in data:
        temp_dict = {requested_value : data[key][requested_value] for requested_value in parameters}
        data_summary.loc[key] = temp_dict
    
    data_summary['q'] = 4*np.pi/data_summary['Wavelength']*sample_info['refractive_index']*np.sin(np.radians(data_summary['Angle']/2))
    
    data_summary['Date_Time']= pd.to_datetime(data_summary['Date_Time'])  
    data_summary = data_summary.drop(columns=['Filename'])
    
    data_average = data_summary.groupby(['Angle']).agg([np.mean, np.std])
    print('Data from {} imported correcty.'.format(data_path))
    
    
    # data_solvent = {}
    # if sample_info['data_path_solvent']: #if the solvent data path has been provided
    #     solvent_path = sample_info['data_path_solvent']
    #     for file in sorted(os.listdir(solvent_path)):
    #         if file.endswith(".ASC"):
    #             data_solvent[file] =  extract_data_file(solvent_path, file)  
    #     parameters = ['T', 'Angle', 'Imon', 'CR0', 'CR1', 'mean_CR0', 'mean_CR1', 'Date_Time', 'Wavelength'] #parameters found in the summary pandas dataframe
    #     solvent_summary =  pd.DataFrame(columns=parameters)
    #     for key in data_solvent:
    #         temp_dict = {requested_value : data_solvent[key][requested_value] for requested_value in parameters}
    #         solvent_summary.loc[key] = temp_dict
    
    #     solvent_summary['q'] = 4*np.pi/data_summary['Wavelength']*sample_info['refractive_index']*np.sin(np.radians(data_summary['Angle']/2))
    #     solvent_summary['Date_Time']= pd.to_datetime(solvent_summary['Date_Time'])        
    #     print('Data from {} imported correcty.'.format(solvent_path))

    
    
    return data, data_summary, data_average





def plot_raw_intensity(data, title, path):
    ''' Here all count rate traces for each file are plotted. The main goal is 
    to be able to rapidly spot problems in the measurements. 
    '''
    def applyPlotStyle(title, ax):
        ax.set_xlabel('time / s')
        ax.set_ylabel('Count rate / a.u.')
        ax.set_title(title)
        # ax.legend(loc='upper left')
    
    
    def plot(ax, data, file):
        ax.plot(data[file]['CRs'][:,0], data[file]['CRs'][:,1], '-', linewidth=1, label='CR0')
        ax.plot(data[file]['CRs'][:,0], data[file]['CRs'][:,2], '-', linewidth=1, label='CR1')
        
    N = len(data)
    rows = int(np.ceil(np.sqrt(N/2)))
    cols = int(np.ceil(N/rows)) if rows > 0 else 1
    gs = gridspec.GridSpec(rows, cols)
    
    fig = plt.figure(figsize=(cols*3.5, rows*3.5))

    i = 0
    for file in sorted(data):
        ax = fig.add_subplot(gs[i])
        applyPlotStyle(file, ax)
        plot(ax, data, file)
        i+= 1
    
    gs.tight_layout(fig)
    plt.savefig(os.path.join(path, '{}_raw_CR.pdf'.format(title)))
    plt.close(fig)
    print('Countrates of {} correctly plotted.'.format(title))
    return None


def plot_all_g2s(data, title, path):
    ''' Here all intensity correlation functions for each file are plotted. The main goal is 
    to be able to rapidly spot problems in the measurements. 
    '''
    def applyPlotStyle(title, ax):
        ax.set_xlabel('$\\tau$/ ms')
        ax.set_ylabel('g$^{(\\tau)}$ / a.u.')
        ax.set_xscale('log')
        ax.set_title(title)
      
    def plot(ax, data, file):
        g2_t0 = np.average(data[file]['g2s']['g2_average'][:20])
        if g2_t0 > 0.4:
            ax.spines['bottom'].set_color('red')
            ax.spines['top'].set_color('red')
            ax.xaxis.label.set_color('red')
            ax.tick_params(axis='x', colors='red')
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('red')
            ax.yaxis.label.set_color('red')
            ax.tick_params(axis='y', colors='red')    
            print('File {} has too large intercept, check the file and delete it if neccesary'.format(file))
            
            
        
            
        ax.plot(data[file]['g2s'].index.to_numpy(), data[file]['g2s']['g2_1'].to_numpy(), '-', linewidth=0.5)
        ax.plot(data[file]['g2s'].index.to_numpy(), data[file]['g2s']['g2_2'].to_numpy(), '-', linewidth=0.5)
        ax.plot(data[file]['g2s'].index.to_numpy(), data[file]['g2s']['g2_average'].to_numpy(), '-', linewidth=1.5)
            # ax.plot(data[file]['g2s'][:,0], data[file]['g2s'][:,2], '-', linewidth=0.5)
        # ax.plot(data[file]['g2s'][:,0], data[file]['g2s'][:,3], '-', linewidth=1.5)
        
    
    N = len(data)
    rows = int(np.ceil(np.sqrt(N/2)))
    cols = int(np.ceil(N/rows)) if rows > 0 else 1
    gs = gridspec.GridSpec(rows, cols)
    
    fig = plt.figure(figsize=(cols*3.5, rows*3.5))

    i = 0
    for file in sorted(data):
        ax = fig.add_subplot(gs[i])
        applyPlotStyle(file, ax)
        plot(ax, data, file)
        i+= 1
    
    gs.tight_layout(fig)
    plt.savefig(os.path.join(path, '{}_raw_g2s.pdf'.format(title)))
    print('Correlation functions of {} correctly plotted.'.format(title))
    plt.close(fig)
    
    return None

def toluene_normalization(static_tol, sample, path):
    '''In this function, the toluene static intensities are plotted. The intensity
    is calculated as the average of the countrates CR0 and CR1, and normalized by the
    monitor intensity. The angle dependent intensity is fitted with the function:
        I = A/sin(angle). 
    This function is used to normalize the scattering intensity of the sample measurements. 
    '''
    
    def model(x, A):
        return A/np.sin(np.radians(x))
    
    Inten = ((static_tol['CR0']['mean'] + static_tol['CR1']['mean'])/static_tol['Imon']['mean']).tolist() #intensity calculated from the angle averaged values of CR0, CR1, and Imon
    Inten_std = (static_tol['CR0']['std'] + static_tol['CR1']['std'])/(static_tol['CR0']['mean'] + static_tol['CR1']['mean']) #standard deviation from error propagation.
    Inten_std += static_tol['Imon']['std']/static_tol['Imon']['mean']#standard deviation from error propagation.
    Inten_std *= Inten #standard deviation from error propagation.
    Inten_std = Inten_std.tolist()
    A, Aerr = opt.curve_fit(model, static_tol.index.tolist(), Inten, p0 = [1.5e-5])
    
    #opt_Inten = model(static_tol['Angle'].tolist(), A)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(static_tol.index.tolist(), Inten, yerr=Inten_std,  fmt='o')
    ax.set_xlabel('Angle / deg')
    ax.set_ylabel('CR0+CR1 / Imon / a.u.')
    ax.plot(static_tol.index.tolist(), model(static_tol.index.tolist(), A))
    
    fig.savefig(os.path.join(path, 'toluene.pdf'))
    plt.close(fig)
    
    for key in sample:
        sample[key]['Tol_int'] = A[0]
        sample[key]['Tol_int_err'] = Aerr[0][0]
    return A, Aerr

def solvent_intensity(sample_info):
    return None




def sample_intensity(sample_info):
    sample = sample_info['sample_summary']
    sample['Inten'] = (sample['CR0']+sample['CR1'])/sample['Imon']
    I_tol = sample_info['Tol_int']/np.sin(np.radians(sample['Angle'].tolist()))
    sample['I_q'] = sample['Inten']/I_tol*RR #cm-1
    # sample['I_q'] = sample['Inten']/I_tol*RR #cm-1
    # print(sample_info)
    NA = 6.022e23 #mol-1
    wl = sample_info['sample_summary']['Wavelength'].mean()
    KL = 4*np.pi**2*sample_info['refractive_index']**2*sample_info['dndc']**2/NA/(wl*1e-7)**4  #in cm2/g2/mol
    sample['KcR'] = sample_info['conc']*KL/sample['I_q']
    sample['KcR'] = sample['I_q']/sample['I_q']*sample['KcR']
    sample_info['sample_average']['I_q', 'mean'] = sample.groupby(['Angle']).mean()['I_q'] 
    
    sample_info['sample_average']['I_q', 'std'] = sample.groupby(['Angle']).std()['I_q'] 
    sample_info['sample_average']['KcR', 'mean'] = sample.groupby(['Angle']).mean()['KcR']
    sample_info['sample_average']['KcR', 'std'] = sample.groupby(['Angle']).std()['KcR']
#    print(sample_info['sample_average'])
    return None

def export_intensity(sample_info):
    sample = sample_info['sample_average']
#    print(sample)
    static_to_be_exported = pd.DataFrame()
    static_to_be_exported['T'] = sample['T','mean']
    static_to_be_exported['q'] = sample['q','mean']
    static_to_be_exported['Imon_mean'] = sample['Imon','mean']
    static_to_be_exported['Imon_std'] = sample['Imon','std']
    static_to_be_exported['Iq_mean'] = sample['I_q','mean']
    static_to_be_exported['Iq_std'] = sample['I_q','std']
    static_to_be_exported['KcR_mean'] = sample['KcR','mean']
    static_to_be_exported['KcR_std'] = sample['KcR','std']
    try:
        Iq = sample_info['I0']*np.exp(-1/3*sample_info['Rg']**2*sample['q', 'mean']**2)
        static_to_be_exported['Guinier_fit'] = Iq
    except:
        None
#    print(static_to_be_exported)
#    static_to_be_exported['Angle'] = sample.index.tolist()
#    columns_to_be_exported = 
    filename = os.path.join(sample_info['data_path'], 'SLS_params.csv')
    with open(filename, 'w+') as f:
        header = '#Units are K for the Temperature, cm^-1 for the intensity, 1/nm for the scattering vector, and mol/g for KcR\n'
        f.write(header)
    
    static_to_be_exported.to_csv(filename, mode='a')
    
    return None

def plot_intensity(sample_info):
    ''' Function where the static intensities are plotted. Two plots will be generated:
    in the first, the static intensity is reported as a function of the
        scattering vector q, in a log-log representation.
    in the second, Kl*C/R is reported as a function of the sinus of the 
        scattering angle.'''
        
    sample = sample_info['sample_average']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0,4.5))
    ax1.set_xlabel('q / nm-1')
    ax1.set_ylabel('I(q) / cm$^{-1}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    #defines the limits to plot the data analyzed by guinier fit. 
    try:
        minq = 0.0 if sample_info['qmin'] is None else sample_info['qmin']
    except:
        minq = 0.0
    try:
        maxq = 10.0 if sample_info['qmax'] is None else sample_info['qmax']
    except:
        maxq = 10.0
    
    
    mask_1 =  sample['q', 'mean'].to_numpy() > minq
    mask_2 =  sample['q', 'mean'].to_numpy() < maxq
    mask =  np.logical_and(mask_1, mask_2)
    
    scatter = ax1.errorbar(sample['q', 'mean'][mask],  sample['I_q', 'mean'][mask], color='C0', marker='s', yerr=sample['I_q', 'std'][mask],  label=sample_info['name'])
    mask = [x for x in mask == False]
    ax1.errorbar(sample['q', 'mean'][mask],  sample['I_q', 'mean'][mask], color='C0', marker='s', yerr=sample['I_q', 'std'][mask], alpha=0.5)
    ax1.set_ylim(ax1.get_ylim())

    try:
        Iq = sample_info['I0']*np.exp(-1/3*sample_info['Rg']**2*sample['q', 'mean']**2)
        fit = ax1.plot(sample['q', 'mean'].to_numpy(), Iq.to_numpy(), label='Guinier fit', color='C2')
        s1 = 'I(0) = {:.2f} $\pm$ {:.2f} $cm^{{-1}}$'.format(sample_info['I0'],  sample_info['I0_err'])
        s2 = 'R$_g$ = {:.0f} $\pm$ {:.0f} nm'.format(sample_info['Rg'],  sample_info['Rg_err'])
    except:
        s1, s2 = '', ''
        print('Could not plot guinier fits')

    
    # extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=s1))
    handles.append(mpatches.Patch(color='none', label=s2))
    ax1.legend(handles=handles)
    
    
    # plt.legend([scatter, fit, extra, extra],[scatter.get_label(), fit.get_label(), s1, s2], loc='best')
    
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.set_xlabel(r'$sin(\theta/2$)')
    ax2.set_ylabel('K$_L$c/R / mol g$^{-1}$')    # 
    ax2.errorbar(np.sin(np.radians(sample.index/2)),  sample['KcR', 'mean'], marker='s', yerr=sample['KcR', 'std'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(sample_info['data_path'], str(sample_info['name'])+'_static.pdf'))
    # plt.close()
    return None

    
def analyze_static_intensity(sample_info):
    ''' Description here'''
    sample = sample_info['sample_average']
    
    def Guinier(sample):
        ''' In this function, the scattering intensity is described as:
            I(q) = I(0)exp(-q^2Rg^2/3). '''
            
        #initialization of the fit routine for the guinier fit.
        fit_params = Parameters()
        fit_params.add('I0', value = sample['I_q', 'mean'].to_numpy()[0], min=1e-6, vary=True)
        fit_params.add('Rg', value = 15.0, min=5.0, max=400, vary=True)
        
        
        def f2min(params):
            vals = params.valuesdict()
            Iq = np.log(vals['I0']) - 1/3*(vals['Rg']**2)*(sample['q', 'mean']**2)

            #the function is evaluated only between qmin and qmax, if provided.            
            try:
                minq = 0.0 if sample_info['qmin'] is None else sample_info['qmin']
            except:
                minq = 0.0
            try:
                maxq = 10.0 if sample_info['qmax'] is None else sample_info['qmax']
            except:
                maxq = 10.0
           
           
            mask_1 =  sample['q', 'mean'].to_numpy() > minq
            mask_2 =  sample['q', 'mean'].to_numpy() < maxq
            mask =  np.logical_and(mask_1, mask_2)
 
            if np.isnan(sample['I_q', 'std'].to_numpy()).any():
                res = (np.log(sample['I_q', 'mean']) - Iq)
            else:
                res = ( (np.log(sample['I_q', 'mean']) - Iq) / (sample['I_q', 'std']/sample['I_q', 'mean'])).to_numpy()

            return res[mask]
        
        out = minimize(f2min, fit_params, xtol=1e-6)
        print(fit_report(out))
        return out.params['I0'].value, out.params['I0'].stderr, out.params['Rg'].value, out.params['Rg'].stderr
            
    sample_info['I0'],  sample_info['I0_err'], sample_info['Rg'], sample_info['Rg_err'] = Guinier(sample)
    plot_intensity(sample_info)
    print('Guinier analysis of {} sample performed.'.format(sample_info['name']))
    return None



def plot_analyzed_correlations_functions(sample_info, dls_methods):
    ''' Function where all correlation functions are plotted, together with the
    best fitting functions and the residuals. '''


    
    num_true = sum(1 for condition in dls_methods.values() if condition is True)  #counts the number of analyses performed.
    
    if num_true == 0: #if no analysis is performed, the function is terminated here. 
        return None
    else:
        for sample in sample_info:
            data = sample_info[sample] 
            for run in data['sample_data']:
                fig = plt.figure(figsize=(3.5*num_true, 5.5))
                plt.suptitle(run)
                outer = gridspec.GridSpec(1, num_true, wspace=0.2, hspace=0.1)
                counter = 0
                
                if dls_methods['Cumulant'] is True:
                    gs0 = gridspec.GridSpecFromSubplotSpec(2, 1,  subplot_spec=outer[counter], hspace=0.0, height_ratios=[2,1])
                    
                    ax1 = fig.add_subplot(gs0[0])
                    ax2 = fig.add_subplot(gs0[1], sharex=ax1)
                    
                    ax1.set_xscale("log", nonpositive='clip')
                    ax2.set_xscale("log", nonpositive='clip')
                    
                    ax1.set_ylabel('$g^{(2)}(\\tau$)')
                    ax2.set_xlabel('$\\tau$ / ms')
                    ax2.set_ylabel('Residual')     

                    
                    decay = dls_methods['Cumulant_decay']*data['sample_summary'].loc[run,'CM_A']
                                        
                    tau = data['sample_data'][run]['g2s'].index.to_numpy()
                    g2_cumulant = data['sample_data'][run]['g2s']['g2_Cumulant'].to_numpy()
                    g2s = data['sample_data'][run]['g2s']['g2_average'].to_numpy()
                    
                    tau = tau[g2s>decay]
                    g2_cumulant = g2_cumulant[g2s>decay]
                    g2s = g2s[g2s>decay]
                    residual = g2s - g2_cumulant
            
                    ax1.set_title('Cumulant Analysis')
                    ax1.plot( data['sample_data'][run]['g2s'].index.to_list(), data['sample_data'][run]['g2s']['g2_average'].to_list(), 'o')
                    ax1.plot(tau, g2_cumulant)
                    ax2.plot(tau, residual)
                    counter += 1
                    
                    
                if dls_methods['Frisken'] is True:
                    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1,  subplot_spec=outer[counter], hspace=0.0, height_ratios=[2,1])
                    
                    ax1 = fig.add_subplot(gs1[0])
                    ax2 = fig.add_subplot(gs1[1], sharex=ax1)
                    
                    ax1.set_xscale("log", nonpositive='clip')
                    ax2.set_xscale("log", nonpositive='clip')
                    
                    ax1.set_title('Frisken Analysis')
                    ax1.set_ylabel('$g^{(2)}(\\tau$)')
                    ax2.set_xlabel('$\\tau$ / ms')
                    ax2.set_ylabel('Residual')                
                    ax1.plot( data['sample_data'][run]['g2s'].index.to_list(), data['sample_data'][run]['g2s']['g2_average'].to_list(), 'o')
                    ax1.plot( data['sample_data'][run]['g2s'].index.to_list(), data['sample_data'][run]['g2s']['g2_Frisken'].to_list())
                    residual = data['sample_data'][run]['g2s']['g2_average'].to_numpy() - data['sample_data'][run]['g2s']['g2_Frisken'].to_numpy()
                    ax2.plot( data['sample_data'][run]['g2s'].index.to_list(), residual)
                    
                    counter += 1 
 
                if dls_methods['Double_exponential'] is True:
                    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1,  subplot_spec=outer[counter], hspace=0.0, height_ratios=[2,1])
                    
                    ax1 = fig.add_subplot(gs1[0])
                    ax2 = fig.add_subplot(gs1[1], sharex=ax1)
                    
                    ax1.set_xscale("log", nonpositive='clip')
                    ax2.set_xscale("log", nonpositive='clip')
                    
                    ax1.set_title('Double_exponential')
                    ax1.set_ylabel('$g^{(2)}(\\tau$)')
                    ax2.set_xlabel('$\\tau$ / ms')
                    ax2.set_ylabel('Residual')                
                    ax1.plot( data['sample_data'][run]['g2s'].index.to_list(), data['sample_data'][run]['g2s']['g2_average'].to_list(), 'o')
                    ax1.plot( data['sample_data'][run]['g2s'].index.to_list(), data['sample_data'][run]['g2s']['g2_Double_Exponential'].to_list())
                    residual = data['sample_data'][run]['g2s']['g2_average'].to_numpy() - data['sample_data'][run]['g2s']['g2_Double_Exponential'].to_numpy()
                    ax2.plot( data['sample_data'][run]['g2s'].index.to_list(), residual)
                    
                    counter += 1 
                    
                if dls_methods['Stretched_exponential'] is True:
                    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1,  subplot_spec=outer[counter], hspace=0.0, height_ratios=[2,1])
                    
                    ax1 = fig.add_subplot(gs1[0])
                    ax2 = fig.add_subplot(gs1[1], sharex=ax1)
                    
                    ax1.set_xscale("log", nonpositive='clip')
                    ax2.set_xscale("log", nonpositive='clip')
                    
                    ax1.set_title('Stretched exponential Analysis')
                    ax1.set_ylabel('$g^{(2)}(\\tau$)')
                    ax2.set_xlabel('$\\tau$ / ms')
                    ax2.set_ylabel('Residual')                
                    ax1.plot( data['sample_data'][run]['g2s'].index.to_list(), data['sample_data'][run]['g2s']['g2_average'].to_list(), 'o')
                    ax1.plot( data['sample_data'][run]['g2s'].index.to_list(), data['sample_data'][run]['g2s']['g2_Stretched_exp'].to_list())
                    residual = data['sample_data'][run]['g2s']['g2_average'].to_numpy() - data['sample_data'][run]['g2s']['g2_Stretched_exp'].to_numpy()
                    ax2.plot( data['sample_data'][run]['g2s'].index.to_list(), residual)
                    
                    counter += 1             
                    
                #plt.tight_layout()
                plt.savefig(os.path.join(data['data_path'], str(run).split('.ASC')[0] + '.png'))
                plt.close()
            
            if dls_methods['Cumulant'] is True:
                print('Cumulant analysis on sample {} performed'.format(sample))
            if dls_methods['Stretched_exponential'] is True:
                print('Stretched exponential fit on sample {} performed'.format(sample))
            if dls_methods['Frisken'] is True:
                print('Frisken fit on sample {} performed'.format(sample))
            if dls_methods['Double_exponential'] is True:
                print('Double Exponential fit on sample {} performed'.format(sample))                
    return None
   
def plot_dls_results(sample_info, dls_methods):
    ''' Description here'''
    for sample in sample_info:
       data = sample_info[sample]['sample_summary']
       
       
       if sample_info[sample]['time_series'] is True:
            plt.figure()
            plt.ylabel('D$_{app}$ = $\Gamma$/q$^2$ / $\mu$m^2 s$^{-1}$')
            plt.xlabel('Date and Time')
           
            if dls_methods['Frisken'] is True:
               D_app_Frisken = data['FR_Gamma']/(data['q'])**2 * 1000 / 1e6 #from nm^2/ms to mu2/s
               plt.plot(data['Date_Time'], D_app_Frisken, 'bo', alpha=0.75, label='Frisken fit')
             
                
            if dls_methods['Cumulant'] is True:
               D_app_cumulant = data['CM_Gamma']/(data['q'])**2 * 1000 / 1e6 #from nm^2/ms to mu2/s
               plt.plot(data['Date_Time'], D_app_cumulant, 'rs',  alpha=0.75, label='Cumulant fit')
    
            if dls_methods['Stretched_exponential'] is True:
               D_app_stretched = data['SE_Gamma']/data['SE_beta']*gamma(1/data['SE_beta'])/(data['q'])**2 * 1000 / 1e6 #from nm^2/ms to mu2/s
               plt.plot(data['Date_Time'], D_app_stretched, 'mv',  alpha=0.75, label='Stretched exp. fit')             
           
           
       else:
           fig = plt.figure()
           ax = fig.add_subplot(111)
           plt.ylabel('D$_{app}$ = $\Gamma$/q$^2$ / $\mu$m^2 s$^{-1}$')
           plt.xlabel('q$^2$ / 10$^{-4}$ nm$^{-2}$')
           
           if dls_methods['Frisken'] is True:
               D_app_Frisken = data['FR_Gamma']/(data['q'])**2 * 1000 / 1e6 #from nm^2/ms to mu2/s
               ax.plot(data['q'].to_numpy()**2*1e4, D_app_Frisken.to_numpy(), 'bo', alpha=0.75, label='Frisken fit')

               linfit = linregress(data['q']**2*1e4, D_app_Frisken)
               Dapp_fit = data['q']**2*1e4*linfit.slope + linfit.intercept
               ax.plot(data['q'].to_numpy()**2*1e4, Dapp_fit.to_numpy())
               try:
                   sample_info[sample]['D0_Frisken'], sample_info[sample]['D0_Frisken_err'] = linfit.intercept, linfit.intercept_stderr
                   s1 = 'D(0)$_{{Frisken}}$ = {:.2f} $\pm$ {:.2f} $\mu$m^2 s$^{{-1}}$'.format(sample_info[sample]['D0_Frisken'],  sample_info[sample]['D0_Frisken_err'])
               except:
                   sample_info[sample]['D0_Frisken'], sample_info[sample]['D0_Frisken_err'] = linfit.intercept, np.nan
                   s1 = 'D(0)$_{{Frisken}}$ = {:.2f} $\mu$m^2 s$^{{-1}}$'.format(sample_info[sample]['D0_Frisken'])
            
               ax.plot([],[], ' ', label=s1)
    
           if dls_methods['Cumulant'] is True:
               D_app_cumulant = data['CM_Gamma']/(data['q'])**2 * 1000 / 1e6 #from nm^2/ms to mu2/s
               ax.plot(data['q'].to_numpy()**2*1e4, D_app_cumulant.to_numpy(), 'rs',  alpha=0.75, label='Cumulant fit')
               
               linfit = linregress(data['q']**2*1e4, D_app_cumulant)
               Dapp_fit = data['q']**2*1e4*linfit.slope + linfit.intercept
               ax.plot(data['q'].to_numpy()**2*1e4, Dapp_fit.to_numpy(), color='red')
               try:
                   sample_info[sample]['D0_Cumulant'], sample_info[sample]['D0_Cumulant_err'] = linfit.intercept, linfit.intercept_stderr
                   s1 = 'D(0)$_{{Cumulant}}$ = {:.2f} $\pm$ {:.2f} $\mu$m^2 s$^{{-1}}$'.format(sample_info[sample]['D0_Cumulant'],  sample_info[sample]['D0_Cumulant_err'])
               except:
                   sample_info[sample]['D0_Cumulant'], sample_info[sample]['D0_Cumulant_err'] = linfit.intercept, np.nan
                   s1 = 'D(0)$_{{Cumulant}}$ = {:.2f} $\mu$m^2 s$^{{-1}}$'.format(sample_info[sample]['D0_Cumulant'])
            
               ax.plot([],[], ' ', label=s1)
    
           if dls_methods['Stretched_exponential'] is True:
               D_app_stretched = data['SE_Gamma']/data['SE_beta']*gamma(1/data['SE_beta'])/(data['q'])**2 * 1000 / 1e6 #from nm^2/ms to mu2/s
               plt.plot(data['q'].to_numpy()**2*1e4, D_app_stretched.to_numpy(), 'mv',  alpha=0.75, label='Stretched exp. fit')        

               linfit = linregress(data['q']**2*1e4, D_app_stretched)
               Dapp_fit = data['q']**2*1e4*linfit.slope + linfit.intercept
               ax.plot(data['q'].to_numpy()**2*1e4, Dapp_fit.to_numpy(), color='magenta')
               try:
                   sample_info[sample]['D0_stretched'], sample_info[sample]['D0_stretched_err'] = linfit.intercept, linfit.intercept_stderr
                   s1 = 'D(0)$_{{stretched}}$ = {:.2f} $\pm$ {:.2f} $\mu$m^2 s$^{{-1}}$'.format(sample_info[sample]['D0_stretched'],  sample_info[sample]['D0_stretched_err'])
               except:
                   sample_info[sample]['D0_stretched'], sample_info[sample]['D0_stretched_err'] = linfit.intercept, np.nan
                   s1 = 'D(0)$_{{stretched}}$ = {:.2f} $\mu$m^2 s$^{{-1}}$'.format(sample_info[sample]['D0_stretched'])
            
               ax.plot([],[], ' ', label=s1)
               



       ax.legend()
       plt.tight_layout()
       plt.savefig(os.path.join(sample_info[sample]['data_path'], str(sample) + 'Dapp.pdf'))
       plt.show()
       
    return None

def analyze_correlation_function(sample_info, dls_methods):
    ''' Description here'''
    #sample = sample_info['sample_average']   
    for sample in sample_info:
#       sys.stdout.write('*'); sys.stdout.flush(); 
       data = sample_info[sample]  
       #analyse non-averaged curves. 
       if dls_methods['Frisken'] is True:
            frisken_temp = pd.DataFrame(columns=['run', 'FR_A', 'FR_A_err', 'FR_Gamma', 'FR_Gamma_err', 'FR_mu2', 'FR_mu2_err'])
       if dls_methods['Cumulant'] is True:
            cumulant_temp = pd.DataFrame(columns=['run', 'CM_A', 'CM_A_err', 'CM_Gamma', 'CM_Gamma_err', 'CM_mu2', 'CM_mu2_err'])
       if dls_methods['Stretched_exponential'] is True:
            stretched_temp = pd.DataFrame(columns=['run', 'SE_A', 'SE_A_err', 'SE_Gamma', 'SE_Gamma_err', 'SE_beta', 'SE_beta_err'])
       if dls_methods['Double_exponential'] is True:
           double_exp_temp = pd.DataFrame(columns=['run', 'DE_A', 'DE_A_err', 'DE_Gamma_1', 'DE_Gamma_1_err',  'DE_Gamma_2', 'DE_Gamma_2_err',  'DE_alpha', 'DE_alpha_err'])
       
       for run in data['sample_data']:
           g2_tau = np.array([data['sample_data'][run]['g2s'].index.to_list(), data['sample_data'][run]['g2s']['g2_average'].to_list()])
                  
           if dls_methods['Frisken'] is True:
               Frisken_params, data['sample_data'][run]['g2s']['g2_Frisken'] = dls.Frisken(g2_tau)
               frisken_pars =  {'run':run,
                                'FR_A': Frisken_params.params['A'].value,
                                'FR_A_err': Frisken_params.params['A'].stderr, 
                                'FR_Gamma': Frisken_params.params['Gamma'].value, 
                                'FR_Gamma_err': Frisken_params.params['Gamma'].stderr, 
                                'FR_mu2':Frisken_params.params['mu2'].value, 
                                'FR_mu2_err':Frisken_params.params['mu2'].stderr}
               
               # frisken_temp = frisken_temp.append(frisken_pars, ignore_index=True)
               frisken_temp = pd.concat([frisken_temp,  pd.DataFrame([frisken_pars])], ignore_index=True)
 
               
               

           if dls_methods['Cumulant'] is True: 
               Cumulant_params, data['sample_data'][run]['g2s']['g2_Cumulant'] = dls.Cumulant(g2_tau, dls_methods['Cumulant_decay'])
               cumulant_pars = {'run':run,
                                'CM_A': Cumulant_params.params['A'].value,
                                'CM_A_err': Cumulant_params.params['A'].stderr,
                                'CM_Gamma': Cumulant_params.params['Gamma'].value, 
                                'CM_Gamma_err': Cumulant_params.params['Gamma'].stderr, 
                                'CM_mu2': Cumulant_params.params['mu2'].value, 
                                'CM_mu2_err': Cumulant_params.params['mu2'].stderr}
               cumulant_temp = pd.concat([cumulant_temp,  pd.DataFrame([cumulant_pars])], ignore_index=True)
#               cumulant_temp = cumulant_temp.append(cumulant_pars, ignore_index=True)

           if dls_methods['Stretched_exponential'] is True: 
               Stretched_params, data['sample_data'][run]['g2s']['g2_Stretched_exp'] = dls.Stretched_exponential(g2_tau)
               stretched_pars = {'run':run,
                                'SE_A': Stretched_params.params['A'].value,
                                'SE_A_err': Stretched_params.params['A'].stderr, 
                                'SE_Gamma': Stretched_params.params['Gamma'].value, 
                                'SE_Gamma_err': Stretched_params.params['Gamma'].stderr, 
                                'SE_beta': Stretched_params.params['beta'].value, 
                                'SE_beta_err': Stretched_params.params['beta'].stderr}
               # stretched_temp = stretched_temp.append(stretched_pars, ignore_index=True)
               stretched_temp = pd.concat([stretched_temp,  pd.DataFrame([stretched_pars])], ignore_index=True)
               
           if dls_methods['Double_exponential'] is True: 
               Double_exp_params, data['sample_data'][run]['g2s']['g2_Double_Exponential'] = dls.Double_Exponential(g2_tau)
               double_exponential_pars =  {'run':run,
                                'DE_A': Double_exp_params.params['A'].value,
                                'DE_A_err': Double_exp_params.params['A'].stderr, 
                                'DE_Gamma_1': Double_exp_params.params['Gamma_1'].value, 
                                'DE_Gamma_1_err': Double_exp_params.params['Gamma_1'].stderr, 
                                'DE_Gamma_2': Double_exp_params.params['Gamma_2'].value, 
                                'DE_Gamma_2_err': Double_exp_params.params['Gamma_2'].stderr, 
                                'DE_alpha': Double_exp_params.params['alpha'].value, 
                                'DE_alpha_err':Double_exp_params.params['alpha'].stderr}
               
               
               double_exp_temp = pd.concat([double_exp_temp,  pd.DataFrame([double_exponential_pars])], ignore_index=True)
               
               
           if dls_methods['Contin'] is True:
               # data['sample_data'][run]['g2s']['g2_Contin'] = dls.Contin(g2_tau)
               alldata =dls.Contin(g2_tau, dls_methods['Contin_pars'])
               distr_function = [alldata[-1][1][:, 2], alldata[-1][1][:, 0], alldata[-1][1][:, 1]] #distribution function of decay times
               # print(alldata[-1][0])
               
               testxdata = alldata[-1][1][:, 2]
               testydata = alldata[-1][1][:, 0]
               testyerr = alldata[-1][1][:, 1]
               
               plt.xscale('log')
               plt.errorbar(testxdata, testydata, yerr=testyerr, fmt='rs')

               plt.show()
               
       if dls_methods['Frisken'] is True:
           frisken_temp.set_index('run',inplace=True) 
           data['sample_summary'] = pd.concat([data['sample_summary'], frisken_temp], axis=1)
       
       if dls_methods['Cumulant'] is True:
           cumulant_temp.set_index('run',inplace=True) 
           data['sample_summary'] = pd.concat([data['sample_summary'], cumulant_temp], axis=1)
           
       if dls_methods['Stretched_exponential'] is True:
           stretched_temp.set_index('run',inplace=True) 
           data['sample_summary'] = pd.concat([data['sample_summary'], stretched_temp], axis=1)
        
       if dls_methods['Double_exponential'] is True:
           double_exp_temp.set_index('run',inplace=True) 
           data['sample_summary'] = pd.concat([data['sample_summary'], double_exp_temp], axis=1)
           

       # print(frisken_temp)
       # print(data['sample_summary'])

            
def export_DLS_parameters(sample_info, dls_methods):
    for sample in sample_info:
        data = sample_info[sample]
#       print(data['sample_summary'])
#       print(data['data_path'])
        filename = os.path.join(data['data_path'], 'exp_SLS_DLS_params.csv')
        with open(filename, 'w') as f:
            s = '# The units are: q -- 1/nm; Gamma  -- 1/ms\n'
            f.write(s)
        # print(data['sample_summary'])
        data['sample_summary'].sort_index().to_csv(filename, mode='a')

    
    
    
    return None

def export_results(sample_info):
    try:
        results = pd.read_csv('results.csv')
    except:
        results = pd.DataFrame(columns=['Sample', 'Tol_int','Tol_int_err','I0','I0_err','Rg','Rg_err','D0_Frisken','D0_Frisken_err','D0_Cumulant','D0_Cumulant_err','D0_stretched','D0_stretched_err'])
        
    
    for sample in sample_info:
        s = sample_info[sample]
        params = {'Sample': sample}
        for pars in results:
            if pars == 'Sample':
                None
            else:
                try:
                    params[pars] = s[pars]
                except:
                    params[pars] = np.nan
        # print(params)
        results = pd.concat([results, pd.DataFrame([params])], ignore_index=True)
        # results = results.append(params, ignore_index=True)
        results.set_index('Sample', inplace=True)
    
    results.to_csv('results.csv')
            
            
    
    
    