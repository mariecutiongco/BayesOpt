#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 12:18:53 2018

This iteration of the code does not split the dataset into train and test, and shows the optimization
protocol individual datapoints!
This iteration of the code does not use a grid search!

@author: Marie
"""
import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from copy import deepcopy
import pandas as pd

def bayesian_optimization_nogrid(x, y, ind_active_bayesopt, ind_active_rnd, kernel_bayesopt, n_offset_bayesopt, iter): #kernel_rnd):

        #convert all np arrays into pd dataframes
        n_offset_bayesopt = int(n_offset_bayesopt)
        x = pd.DataFrame(x)
        x = x.iloc[:,0:len(x.columns)]
        x.columns = range(x.shape[1])
        y = pd.DataFrame(y)
        y.columns = range(y.shape[1])

        #create index
        ind_bayesopt = np.array(x.index)
        ind_rnd = np.array(x.index)

        #get the dataset that we have already chosen previously
        x_active_bayesopt       = x.iloc[ind_active_bayesopt,:]
        y_active_bayesopt       = y.iloc[ind_active_bayesopt,:]

        #get the index of the dataset that we need to query
        ind_candidates_bayesopt   = list(set(ind_bayesopt) - set(ind_active_bayesopt))
        ind_candidates_rnd        = list(set(ind_rnd) - set(ind_active_rnd))

        #get the dataset that we need to query
        x_candidates_bayesopt       = x.iloc[ind_candidates_bayesopt,:]


        ######### Bayes opt procedure scenario 1: we start everything from scratch and we have not done any experiments so we show the experimental points one by one. No grid search apart from what we already have measured real PCR values for.


        # Update Bayes opt active set with one point selected via EI (of we have exceded the initial offset period)

        if (iter > n_offset_bayesopt):
            model_bayesopt = gp.GaussianProcessRegressor(kernel=kernel_bayesopt, alpha=0.01, n_restarts_optimizer=5, optimizer=None, normalize_y=True)
            model_bayesopt.fit(x_active_bayesopt, y_active_bayesopt)

            # Find the currently best value (based on the model, not the active data itself as there could be a tiny difference)
            mu_active_bayesopt, sigma_active_bayesopt = model_bayesopt.predict(x_active_bayesopt, return_std=True)
            ind_loss_optimum = np.argmax(mu_active_bayesopt)
            mu_max_active_bayesopt = mu_active_bayesopt[ind_loss_optimum,:] #is this just one value?

            # Predict the values for all the possible candidates in the candidate set using the fitted model_bayesopt
            mu_candidates, sigma_candidates = model_bayesopt.predict(x_candidates_bayesopt, return_std=True)

            # Compute the expected improvement for all the candidates
            Z  = ((mu_candidates - mu_max_active_bayesopt[0])/sigma_candidates)[0]
            ei = ((mu_candidates - mu_max_active_bayesopt) * norm.cdf(Z) + sigma_candidates * norm.pdf(Z))[0]
            ei[sigma_candidates == 0.0] = 0.0   # Make sure to account for the case where sigma==0 to avoid nummerical issues (would be NaN otherwise)

            # Find the candidate with the largest expected imrpovement... and choose that one to query/include
            eimax = np.max(ei)           # maximum expected improvement
            iii   = np.argwhere(eimax==ei) # find all points with the same maximum value of ei in case there are more than one (often the case in the beginning)
            iiii = np.random.choice(iii[:,0], size=1)
            ind_new_bayesopt = [ind_candidates_bayesopt[iiii[0].astype(int)]]
            ind_active_bayesopt   = list(set(ind_active_bayesopt).union(set(ind_new_bayesopt)))
            model_bayesopt.kernel_.k1
            kernel_bayesopt = deepcopy(model_bayesopt.kernel_)

        else:
            ind_new_bayesopt = np.random.choice(ind_candidates_bayesopt, size=1)
            ind_active_bayesopt   = list(set(ind_active_bayesopt).union(set(ind_new_bayesopt)))

            model_bayesopt = gp.GaussianProcessRegressor(kernel=kernel_bayesopt, alpha=0.01, n_restarts_optimizer=5, optimizer=None, normalize_y=True)
            model_bayesopt.fit(x_active_bayesopt, y_active_bayesopt)

            mu_active_bayesopt, sigma_active_bayesopt = model_bayesopt.predict(x_active_bayesopt, return_std=True)

    ###################### Random procedure, i.e. choose points randomly to be included

        ind_new_rnd = np.random.choice(ind_candidates_rnd, size=1)
        ind_active_rnd        = list(set(ind_active_rnd).union(set(ind_new_rnd)))

        y_best_bayesopt = np.max(y.iloc[ind_active_bayesopt,:])
        y_best_rnd = np.max(y.iloc[ind_active_rnd,:])

        return ind_active_bayesopt, ind_active_rnd, ind_new_bayesopt, ind_new_rnd, np.squeeze(y_best_bayesopt), np.squeeze(y_best_rnd), kernel_bayesopt, x, y, model_bayesopt, mu_active_bayesopt, sigma_active_bayesopt
