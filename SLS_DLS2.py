#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:39:25 2020

@author: leonardo
"""

from lmfit import minimize, Parameters, fit_report
import numpy as np
from math import log
import os as os
from CONTINWrapper import runCONTINfit

def Frisken(g2_tau):
    ''' In this function, the correlation function is analyzed as:
        g2(tau) = A*exp(-2*Gamma*tau)*(1 + mu2/2*tau**2) '''
                    
    fit_params = Parameters()
    fit_params.add('A', value = 0.3, vary=True)
    fit_params.add('Gamma', value = 0.01, min=0.001, vary=True)
    fit_params.add('mu2', value = 0.01, vary=True)
                    
    tau, g2 = g2_tau[0,:], g2_tau[1,:]
                
    def f2min(params):
        vals = params.valuesdict()
        model = vals['A']*(np.exp(-2*vals['Gamma']*tau))*(1+vals['mu2']/2*tau**2)
        return (g2 - model)
        
    out = minimize(f2min, fit_params, xtol=1e-6)
    
    g2_calc = out.params['A'].value*(np.exp(-2*out.params['Gamma'].value*tau))*(1+out.params['mu2'].value/2*tau**2)
    
    # print(fit_report(out))

    return out, g2_calc



def Double_Exponential(g2_tau):
    ''' In this function, the field correlation function is analyzed as:
        g1(tau) = A*[alpha*exp(-Gamma1*tau)+(1-alpha)*exp(-Gamma2*tau)]'''
        
    tau, g2 = g2_tau[0,:], g2_tau[1,:]
    
    fit_params = Parameters()
    fit_params.add('A', value = 0.3, vary=True)
    fit_params.add('Gamma_1', value = 0.01, min=0.001, vary=True)
    fit_params.add('beta', value = 15, min=2, vary=True)
    fit_params.add('Gamma_2', expr='Gamma_1*beta')
    fit_params.add('alpha', value=0.5, min=0.01, max=0.99, vary=True)
    
    def f2min(params):
        vals = params.valuesdict()
        model = vals['A']*(vals['alpha']*np.exp(-vals['Gamma_1']*tau)+(1-vals['alpha'])*np.exp(-vals['Gamma_2']*tau))**2
        return (g2 - model)
    
    out = minimize(f2min, fit_params, xtol=1e-6)
    
    g2_calc = out.params['A'].value*(out.params['alpha'].value*np.exp(-out.params['Gamma_1'].value*tau)+(1-out.params['alpha'].value)*np.exp(-out.params['Gamma_2'].value*tau))**2
    
    return out, g2_calc

def Cumulant(g2_tau,decay):
    ''' In this function, the correlation function is analyzed  untill it has decayed by 20% as:
        ln g2(tau) = ln(A) -2*Gamma*tau) + mu2/2*tau**2'''
    
    tau, g2 = g2_tau[0,:], g2_tau[1,:]
    Intercept = np.average(g2[:15]) #the initial value of the correlation function is calculated.
    new_tau = tau[g2>Intercept*decay]
    new_g2 = g2[g2>Intercept*decay]

    fit_params = Parameters()
    fit_params.add('A', value = Intercept, min = 1e-10, vary=True)
    fit_params.add('Gamma', value = 0.01, min=0.001, vary=True)
    fit_params.add('mu2', value = 0.01, vary=True)
    
    def f2min(params):
        vals = params.valuesdict()
        model = log(vals['A']) - 2*vals['Gamma']*new_tau +vals['mu2']/2*new_tau**2
        res = (np.log(new_g2) - model)
        # print(type(res), res)
        return res[~np.isnan(res)]
        
    out = minimize(f2min, fit_params, xtol=1e-6)
    
    # print(fit_report(out))
    g2_calc = out.params['A'].value*np.exp(-2*out.params['Gamma'].value*tau + out.params['mu2'].value/2*tau**2)
    
    return out, g2_calc

def Stretched_exponential(g2_tau):
    ''' In this function, the correlation function is analyzed as:
        g2(tau) = A*exp(-2*(Gamma*tau)**beta) '''
                    
    fit_params = Parameters()
    fit_params.add('A', value = 0.3, vary=True)
    fit_params.add('Gamma', value = 0.01, min=0.001, vary=True)
    fit_params.add('beta', value = 1.00, vary=True)
                    
    tau, g2 = g2_tau[0,:], g2_tau[1,:]
                
    def f2min(params):
        vals = params.valuesdict()
        model = vals['A']*(np.exp(-2*(vals['Gamma']*tau)**vals['beta']))
        return (g2 - model)
        
    out = minimize(f2min, fit_params, xtol=1e-6)
    
    g2_calc = out.params['A'].value*(np.exp(-2*(out.params['Gamma'].value*tau)**out.params['beta'].value))
    
    # print(fit_report(out))

    return out, g2_calc


def create_contin_parameter_file(parameters):
    '''function which creates the contin_parameter file to be used for the
    contin analsys'''
    
    s = '  TEST DATA SET 1 (inverse laplace transform) \n'
    s += 'LAST,,{}\n'.format(parameters['LAST'])
    s += 'GMNMX,1,{}\n'.format(parameters['TIME'][0])
    s += 'GMNMX,2,{}\n'.format(parameters['TIME'][1])
    s += 'IWT,,{}\n'.format(parameters['IWT'])
    s += 'NERFIT,,{}\n'.format(parameters['NERFIT'])
    s += 'NINTT,,{}\n'.format(parameters['NINTT'])
    s += 'NLINF,,{}\n'.format(parameters['NLINF'])
    s += 'IFORMY,,{}\n'.format(parameters['IFORMY'])
    s += 'IFORMT,,{}\n'.format(parameters['IFORMT'])
    s += 'DOUSNQ,,{}\n'.format(parameters['DOUSNQ'])
    s += 'IUSER,{},{}\n'.format(parameters['IUSER'][0], parameters['IUSER'][1])  
    s += 'RUSER,{},{}\n'.format(parameters['RUSER'][0], parameters['RUSER'][1])
    s += 'NONNEG,,{}\n'.format(parameters['NONNEG'])
    
    param_file = os.path.join('contin', 'contin_parameter_template.txt')
    
    with open(param_file, "w") as text_file:
        text_file.write(s)
    
    return None
    

    
def Contin(g2_tau, contin_pars):
    create_contin_parameter_file(contin_pars)
    parameterFile = os.path.join('contin',"contin_parameter_template.txt") #the parameters for the contin program are stored in this file. 
    # print(parameterFile)
    alldata = runCONTINfit(g2_tau[0,:], g2_tau[1,:], parameterFile) #, continInputFile=None, continOutputFile=None)
    # print('\n\n\n\n contin data', data_temp)
    return alldata
    