#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 12:18:53 2018

This iteration of the code does not split the dataset into train and test, and shows the optimization
protocol individual datapoints!
This iteration of the code does not use the empirical x data as we assume that those information have already been utilized in a predictive functionself.
This iteration of the code uses a pure grid search!

@author: Marie
"""
import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from copy import deepcopy
import pandas as pd


def bayesian_optimization_withgrid_iter2(gridx, ally_bayesopt, ally_rnd, all_activeind_bayesopt, all_activeind_rnd, kernel_bayesopt, kernel_rnd, n_offset_bayesopt, counter):

        #convert all np arrays into pd dataframes
        n_offset_bayesopt = int(n_offset_bayesopt)

        all_x = deepcopy(gridx)
        all_x = pd.DataFrame(all_x)
        all_x = pd.DataFrame(all_x.iloc[:,0:len(all_x.columns)])
        all_x.columns = range(all_x.shape[1])

        #combine datasets first!
        ind_all_x = np.array(all_x.index)

        #get the x data that has already been seen
        allx_bayesopt = all_x.iloc[all_activeind_bayesopt,:]
        allx_rnd = all_x.iloc[all_activeind_rnd,:]

        #get the index of the dataset that we need to query
        ind_candidates_bayesopt   = list(set(ind_all_x) - set(all_activeind_bayesopt))
        ind_candidates_rnd        = list(set(ind_all_x) - set(all_activeind_rnd))

        #get the dataset that we need to query
        #we only need gridx data here because gridy candidates will be predicted from the bayes opt GP model
        x_candidates_bayesopt       = all_x.iloc[ind_candidates_bayesopt,0:len(all_x.columns)]
        x_candidates_rnd            = all_x.iloc[ind_candidates_rnd,0:len(all_x.columns)]


        # create a model based on all data that we have already seen! not doing this one by one anymore, because it doesnt matter!

        #first we initialize the models that we will need for later on
        kernel_init = gp.kernels.ConstantKernel(0.2)*gp.kernels.RBF(length_scale=0.05)
        model_bayesopt = gp.GaussianProcessRegressor(kernel=kernel_init, alpha=0.01, n_restarts_optimizer=5, optimizer=None, normalize_y=True)
        model_bayesopt.fit(allx_bayesopt, ally_bayesopt)
        model_bayesopt.kernel_.k1
        kernel_bayesopt = deepcopy(model_bayesopt.kernel_)

        model_rnd = gp.GaussianProcessRegressor(kernel=kernel_init, alpha=0.01, n_restarts_optimizer=5, optimizer=None, normalize_y=True)
        model_rnd.fit(allx_rnd, ally_rnd)
        model_rnd.kernel_.k1
        kernel_rnd = deepcopy(model_rnd.kernel_)

        #because we havent done anything, we will just select random datapoints from the grid to query next

        if (1 < counter < n_offset_bayesopt+1):

            ind_new_bayesopt = np.random.choice(ind_candidates_bayesopt, size=1)
            all_activeind_bayesopt   = list(set(all_activeind_bayesopt).union(set(ind_new_bayesopt)))

            ind_new_rnd = np.random.choice(ind_candidates_rnd, size=1)
            all_activeind_rnd        = list(set(all_activeind_rnd).union(set(ind_new_rnd)))

            model_bayesopt = gp.GaussianProcessRegressor(kernel=kernel_bayesopt, alpha=0.01, n_restarts_optimizer=5, optimizer=None, normalize_y=True)
            model_bayesopt.fit(allx_bayesopt, ally_bayesopt)
            model_bayesopt.kernel_.k1
            kernel_bayesopt = deepcopy(model_bayesopt.kernel_)

            # Find the currently best value (based on the model, not the active data itself as there could be a tiny difference)
            mu_active_bayesopt, sigma_active_bayesopt = model_bayesopt.predict(allx_bayesopt, return_std=True)

        else:

            #now we can add the optimizer to the models!

            model_bayesopt = gp.GaussianProcessRegressor(kernel=kernel_bayesopt, alpha=0.01, n_restarts_optimizer=5, optimizer=None, normalize_y=True)
            model_bayesopt.fit(allx_bayesopt, ally_bayesopt)
            model_bayesopt.kernel_.k1
            kernel_bayesopt = deepcopy(model_bayesopt.kernel_)

            # Find the currently best value (based on the model, not the active data itself as there could be a tiny difference)
            mu_active_bayesopt, sigma_active_bayesopt = model_bayesopt.predict(allx_bayesopt, return_std=True)
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
            all_activeind_bayesopt   = list(set(all_activeind_bayesopt).union(set(ind_new_bayesopt)))
            model_bayesopt.kernel_.k1
            kernel_bayesopt = deepcopy(model_bayesopt.kernel_)

            model_rnd = gp.GaussianProcessRegressor(kernel=kernel_rnd, alpha=0.01, n_restarts_optimizer=5, optimizer=None, normalize_y=True)
            model_rnd.fit(allx_rnd, ally_rnd)
            model_rnd.kernel_.k1
            kernel_rnd = deepcopy(model_rnd.kernel_)

            mu_candidates_rnd, sigma_candidates_rnd = model_rnd.predict(x_candidates_rnd, return_std=True)
            ind_new_rnd = np.random.choice(ind_candidates_rnd, size=1)
            all_activeind_rnd  = list(set(all_activeind_rnd).union(set(ind_new_rnd)))

        #now we have to merge all the y data that we have
        y_new_bayesopt = model_bayesopt.predict(all_x.iloc[ind_new_bayesopt,], return_std=False)
        y_new_bayesopt = pd.DataFrame(y_new_bayesopt)
        y_new_rnd = model_rnd.predict(all_x.iloc[ind_new_rnd,], return_std=False)
        y_new_rnd = pd.DataFrame(y_new_rnd)

        ypredicted_candidates_bayesopt = pd.concat([ally_bayesopt, y_new_bayesopt], ignore_index=True)
        ypredicted_candidates_rnd = pd.concat([ally_rnd, y_new_rnd], ignore_index=True)
        y_best_bayesopt = ypredicted_candidates_bayesopt.max()
        y_best_rnd = ypredicted_candidates_rnd.max()


        return all_activeind_bayesopt, all_activeind_rnd, ind_new_bayesopt, ind_new_rnd, y_best_bayesopt, y_best_rnd, kernel_bayesopt, kernel_rnd, y_new_bayesopt, y_new_rnd, model_bayesopt, mu_active_bayesopt, sigma_active_bayesopt


    #you need to predict the new y for the expected best x using the model that you have on R. Then maybe you should retrain the model on R before proceeding. That's how it would be in real life!
