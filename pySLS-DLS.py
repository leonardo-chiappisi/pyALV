#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:46:14 2019
@author: Leonardo Chiappisi
Script for the analysis of static and dynamic light scattering data from the ALV DLS/SLS machine. 
2021.03.02 = Bug correction in the calculation of the scattering intensity. 
"""

version = '0.1'
date = '2021.03.02'

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
                    export_DLS_parameters, export_intensity)

plot_raw_data = True
analyze_static = True
analyze_dynamic = True

contin_parameters = {'LAST':1,  
                     'TIME': [1e-4, 1e2],   #range of decay times analyzed.
                     'IWT':2,                                     
                     'NERFIT':0,                                       
                     'NINTT':-1,
                     'NLINF':1,
                     'IFORMY':'(1E11.4)',
                     'IFORMT':'(1E11.4)',                                                                  
                     'DOUSNQ':1,                                         
                     'IUSER':[10,2],                                          
                     'RUSER':[10,-1],
                     'NONNEG':1
                     }



#methods used to analyze the DLS data. 
dls_methods = {'Cumulant': False, #cumulant analsis of the data with a cutoff defined by the Cumulat_decay parameter is performed. 
               'Cumulant_decay': 0.5, #The correlation function is analyzed until it has decay to xxx of the initial value. 
               'Frisken': True, #correlation curve fitted with the Frisken method
               'Double_exponential': False, #not_yet_implemented
               'Stretched_exponential': False, #The correlation function is analysed with a stretched exponential decay. 
               'Contin': False, #not_yet_implemented
               'Contin_pars': contin_parameters #dictionary containing all the parameters needed to perfom the contin analysis.
               }

toluene = {'name': 'toluene',
           'data_path': 'toluene', #relative path where toluene *.ASC files are saved.
           'data_path_solvent': '', #leave empty. 
           'refractive_index': 1.496}


sample_info = {} #dictionary where all informations on the sample are stored. All the sample defined therein will be analysed. 
sample_info['AY38_G4-2.0-typeB'] = {'name': 'AY38_G4-2.0-typeB', #the name of the sample
                         'data_path': 'rawdata',  #the datapath where all the ASC files are stored, can be relative or absolute. 
                         'data_path_solvent': '',  #the datapath where all the ASC files relative to the solvent are stored, can be relative or absolute. 
                         'conc': 0.1, #in gram/cm3
                         'dndc': 0.15, #in cm3/gram
                         'refractive_index': 1.332,
                         'qmin': None,  #minimum q-value used for the analysis of the sls data
                         'qmax': None, #maximum q-value used for the analysis of the sls data
                         'time_series': False #True if the experiment was performed at one angle as a function of time
                         }




#Read all the data files. 
toluene_data, toluene_summary, toluene_average = extract_data(toluene)
for sample in sample_info:
    s = sample_info[sample]
    s['sample_data'], s['sample_summary'], s['sample_average'] = extract_data(s)
#####################################

#plots the raw data
if plot_raw_data == True:
    plot_raw_intensity(toluene_data, 'toluene')
    plot_all_g2s(toluene_data, 'toluene')
    for sample in sample_info:
        s = sample_info[sample]
        plot_raw_intensity(s['sample_data'], s['name'])
        plot_all_g2s(s['sample_data'], s['name'])
############################################

#Analysis of the static data by Guinier analysis. 
if analyze_static is True:
    toluene_normalization(toluene_average, sample_info)
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
