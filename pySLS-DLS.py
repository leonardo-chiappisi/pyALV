#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:46:14 2019
@author: Leonardo Chiappisi
Script for the analysis of static and dynamic light scattering data from the ALV DLS/SLS machine. 
2021.03.02: Bug correction in the calculation of the scattering intensity. 
2021.03.03: Included export option for fitted paramters. See function export_results. 
2021.07.09: Minor bug correction in export function.
2021.11.08: Added the double exponential fit analysis of the DLS data. 
2021.11.15: Minor bug correction
2023.03.25: Minor improvements in raw-data plotting
2024.01.26: Changes to keep compatibility with pandas 2.0
"""

version = '0.2.5'
date = '2024.01.26'

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="overflow encountered in exp")

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
#import SLS-DLS1 as sls
from SLS_DLS1 import (extract_data, plot_raw_intensity, plot_all_g2s,
                    toluene_normalization, sample_intensity, plot_intensity,
                    analyze_static_intensity, analyze_correlation_function,
                    plot_analyzed_correlations_functions, plot_dls_results,
                    export_DLS_parameters, export_intensity, export_results,
                    group_data_files_by_temperature)

plot_raw_data = True
analyze_static = True
analyze_dynamic = True
# contin_parameters = {'LAST':1,
#                      'TIME': [1e-4, 1e2],   #range of decay times analyzed.
#                      'IWT':2,                                     
#                      'NERFIT':0,                                       
#                      'NINTT':-1,
#                      'NLINF':1,
#                      'IFORMY':'(1E11.4)',
#                      'IFORMT':'(1E11.4)',                                                                  
#                      'DOUSNQ':1,                                         
#                      'IUSER':[10,2],                                          
#                      'RUSER':[10,-1],
#                      'NONNEG':1
#                      }



#methods used to analyze the DLS data. 
dls_methods = {'Cumulant': True, #cumulant analsis of the data with a cutoff defined by the Cumulat_decay parameter is performed. 
               'Cumulant_decay': 0.5, #The correlation function is analyzed until it has decay to xxx of the initial value. 
               'Frisken': True, #correlation curve fitted with the Frisken method
               'Double_exponential': True, #correlation curve fitted with two exponentials. 
               'Stretched_exponential': False, #The correlation function is analysed with a stretched exponential decay. 
               'Contin': False,#not_yet_implemented
               # 'Contin_pars': contin_parameters #dictionary containing all the parameters needed to perfom the contin analysis.
               }

toluene = {'name': 'toluene',
           'data_path': 'toluene', #relative path where toluene *.ASC files are saved.
           'data_path_solvent': '', #leave empty. 
           'refractive_index': 1.496}


sample_info = {} #dictionary where all informations on the sample are stored. All the sample defined therein will be analysed. 
# sample_info['Sample2PScommDil0p12'] = {'name': 'Sample2PScommDil0p12', #the name of the sample
#                                     'data_path': 'rawdata/Sample2PScommDil0p12',  #the datapath where all the ASC files are stored, can be relative or absolute. 
#                                     'data_path_solvent': '',  #the datapath where all the ASC files relative to the solvent are stored, can be relative or absolute. 
#                                     'conc': 0.1, #in gram/cm3
#                                     'dndc': 0.15, #in cm3/gram
#                                     'refractive_index': 1.332,
#                                     'qmin': None,  #minimum q-value used for the analysis of the sls data
#                                     'qmax': None, #maximum q-value used for the analysis of the sls data
#                                     'time_series': False, #True if the experiment was performed at one angle as a function of time
#                                     'Temperature_series': False, #were data measured at different temperature and should they be sorted according to temperature?
#                                     'temperature_tolerance': 0.2, #float, groups all temperatures according to the mean temperature given this tolerance.
#                                     }

sample_info['DOPA'] = {'name': 'DOPA', #the name of the sample
                                    'data_path': 'rawdata/Anastasia/DOPA',  #the datapath where all the ASC files are stored, can be relative or absolute. 
                                    'data_path_solvent': '',  #the datapath where all the ASC files relative to the solvent are stored, can be relative or absolute. 
                                    'conc': 0.1, #in gram/cm3
                                    'dndc': 0.15, #in cm3/gram
                                    'refractive_index': 1.332,
                                    'qmin': None,  #minimum q-value used for the analysis of the sls data
                                    'qmax': None, #maximum q-value used for the analysis of the sls data
                                    'time_series': False, #True if the experiment was performed at one angle as a function of time
                                    'Temperature_series': False, #were data measured at different temperature and should they be sorted according to temperature?
                                    'temperature_tolerance': 0.2, #float, groups all temperatures according to the mean temperature given this tolerance.
                                    }



# sample_info['mPVC_mc_plast'] = {'name': 'mPVC_mc_plast', #the name of the sample
#                                     'data_path': 'rawdata/Anastasia/mPVC_mc_plast',  #the datapath where all the ASC files are stored, can be relative or absolute. 
#                                     'data_path_solvent': '',  #the datapath where all the ASC files relative to the solvent are stored, can be relative or absolute. 
#                                     'conc': 0.1, #in gram/cm3
#                                     'dndc': 0.15, #in cm3/gram
#                                     'refractive_index': 1.332,
#                                     'qmin': None,  #minimum q-value used for the analysis of the sls data
#                                     'qmax': None, #maximum q-value used for the analysis of the sls data
#                                     'time_series': False, #True if the experiment was performed at one angle as a function of time
#                                     'Temperature_series': False, #were data measured at different temperature and should they be sorted according to temperature?
#                                     'temperature_tolerance': 0.2, #float, groups all temperatures according to the mean temperature given this tolerance.
#                                     }




# sample_info['PolyDB_2h_irr'] = {'name': 'PolyDB_2h_irr', #the name of the sample
#                                     'data_path': 'rawdata/Anastasia/PolyDB_2h_irr',  #the datapath where all the ASC files are stored, can be relative or absolute. 
#                                     'data_path_solvent': '',  #the datapath where all the ASC files relative to the solvent are stored, can be relative or absolute. 
#                                     'conc': 0.1, #in gram/cm3
#                                     'dndc': 0.15, #in cm3/gram
#                                     'refractive_index': 1.332,
#                                     'qmin': None,  #minimum q-value used for the analysis of the sls data
#                                     'qmax': None, #maximum q-value used for the analysis of the sls data
#                                     'time_series': False, #True if the experiment was performed at one angle as a function of time
#                                     'Temperature_series': False, #were data measured at different temperature and should they be sorted according to temperature?
#                                     'temperature_tolerance': 0.2, #float, groups all temperatures according to the mean temperature given this tolerance.
#                                     }

# # sample_info['PolyDB_16h_irr'] = {'name': 'PolyDB_16h_irr', #the name of the sample
# #                                     'data_path': 'rawdata/Anastasia/PolyDB_16h_irr',  #the datapath where all the ASC files are stored, can be relative or absolute. 
# #                                     'data_path_solvent': '',  #the datapath where all the ASC files relative to the solvent are stored, can be relative or absolute. 
# #                                     'conc': 0.1, #in gram/cm3
# #                                     'dndc': 0.15, #in cm3/gram
# #                                     'refractive_index': 1.332,
# #                                     'qmin': None,  #minimum q-value used for the analysis of the sls data
# #                                     'qmax': None, #maximum q-value used for the analysis of the sls data
# #                                     'time_series': False, #True if the experiment was performed at one angle as a function of time
# #                                     'Temperature_series': False, #were data measured at different temperature and should they be sorted according to temperature?
# #                                     'temperature_tolerance': 0.2, #float, groups all temperatures according to the mean temperature given this tolerance.
# #                                     }

# sample_info['polyDB_alone'] = {'name': 'polyDB_alone', #the name of the sample
#                                     'data_path': 'rawdata/Anastasia/polyDB_alone',  #the datapath where all the ASC files are stored, can be relative or absolute. 
#                                     'data_path_solvent': '',  #the datapath where all the ASC files relative to the solvent are stored, can be relative or absolute. 
#                                     'conc': 0.1, #in gram/cm3
#                                     'dndc': 0.15, #in cm3/gram
#                                     'refractive_index': 1.332,
#                                     'qmin': None,  #minimum q-value used for the analysis of the sls data
#                                     'qmax': None, #maximum q-value used for the analysis of the sls data
#                                     'time_series': False, #True if the experiment was performed at one angle as a function of time
#                                     'Temperature_series': False, #were data measured at different temperature and should they be sorted according to temperature?
#                                     'temperature_tolerance': 0.2, #float, groups all temperatures according to the mean temperature given this tolerance.
#                                     }















#Read all the data files. 
toluene_data, toluene_summary, toluene_average = extract_data(toluene)
for sample in sample_info:
    s = sample_info[sample]
    s['sample_data'], s['sample_summary'], s['sample_average'] = extract_data(s)
    
    if s['Temperature_series'] is True:
        group_data_files_by_temperature(s['sample_summary'],s['temperature_tolerance'])
        
#####################################

#plots the raw data
if plot_raw_data == True:
    plot_raw_intensity(toluene_data, 'toluene', path=toluene['data_path'])
    plot_all_g2s(toluene_data, 'toluene', path=toluene['data_path'])
    for sample in sample_info:
        s = sample_info[sample]
        plot_raw_intensity(s['sample_data'], s['name'], path=s['data_path'])
        plot_all_g2s(s['sample_data'], s['name'], path=s['data_path'])
        if s['data_path_solvent']:
            plot_raw_intensity(s['sample_data'], s['name'], path=s['data_path'])
            plot_all_g2s(s['sample_data'], s['name'], path=s['data_path'])
############################################

#Analysis of the static data by Guinier analysis. 
if analyze_static is True:
    toluene_normalization(toluene_average, sample_info, path=toluene['data_path'])
    for sample in sample_info:
        sample_intensity(sample_info[sample])
        analyze_static_intensity(sample_info[sample])
        export_intensity(sample_info[sample])
#######################################################

#Analysis of the correlation function according to the models specified in dls_methods. 
if analyze_dynamic is True:
    analyze_correlation_function(sample_info, dls_methods)
    plot_analyzed_correlations_functions(sample_info, dls_methods)
    plot_dls_results(sample_info, dls_methods)
    export_DLS_parameters(sample_info, dls_methods)
#######################################################

export_results(sample_info)

        

