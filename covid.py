#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:06:31 2020

@author: slimane
"""
import numpy as np

from scipy import stats

import  scipy.integrate as integ


import scipy.optimize as op#import minimize



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import csv as csv 


import pandas as pd   

from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split 
from sklearn import linear_model




import os
from pymcmcstat.MCMC import MCMC
from pymcmcstat.plotting import MCMCPlotting
import pymcmcstat
print(pymcmcstat.__version__)
np.seterr(over='ignore');

from pymcmcstat import propagation as up



# In[2]: Data input 


class data:   
    def __init__(self, contry='TUN',N=11791749, t_change =14):
        # INIT de la dabase TUN
        self.contry=contry
     
        # self.popNumbers={'Italy':60.4e6/250, 'Tunisia':630000,'France':67e6/250,'Spain':46.5e6/250}
        #Popxna=1e9/1000
        self.dataset_contry =pd.read_excel(r'/home/slimane/biologie/corona/dataBase/TUNdataBase.xlsx')       
        
        #self.dataset_contry =pd.read_csv('/home/slimane/biologie/corona/dataBase/TUNdataBase.csv')       
         
        self.population=N
        
        a= self.dataset_contry['Nb_cas_journalier'].replace(np.nan, 0).values
        self.newCases=a[np.argmin(a==False):]
#        self.newCases=np.flip(self.dataset_contry['cases'].values,0)

        b= self.dataset_contry['retablie_journalier'].replace(np.nan, 0).values
        self.newRecover=b[np.argmin(a==False):]
        self.cumRecover=np.cumsum(self.newRecover)

        

     
        b=self.dataset_contry['Décès_journalier'].replace(np.nan, 0).values
        self.newDeath=b[np.argmin(a==False):]
        self.Death=np.cumsum(self.newDeath)
        
    
        self.cumCases = np.cumsum(self.newCases,0)
        self.Cases = np.cumsum(self.newCases-self.newDeath-self.newRecover,0)
        
        self.ratio=self.newCases[1:len(self.newCases)]/self.newCases[0:len(self.newCases)-1]

        self.date = self.dataset_contry['Dates']
        #self.Deaths= self.dataset_contry['Deaths']

        self.t_change = t_change
        
    def get_popnumberDatabase(self):    
        y=self.population
        return(y)
    
    def plotData(self):
        recov = self.Total - (self.Cases -self.Deaths)
        plt.plot(self.Cases,"b-")
        plt.plot(self.Deaths,"r-")
        plt.plot(recov,"g-")
        plt.legend(["Anctive cases","Death","Recovered"])
        plt.show()
        
    def CR(self,t, x1, x2, x3):
            res = [0]*len(t)
            for i in range(len(t)):
                res[i] = x1 * np.exp(x2*t[i])-x3
            return res
        
    def estimation(self):
        t_change = self.t_change
        obs = self.cumCases[:-t_change]
        t_measured = list(range(len(obs)))

        popt, pcov = op.curve_fit(self.CR, t_measured, obs)

        x1 = popt[0]
        x2 = popt[1]
        x3 = popt[2]
        t0 = (np.log(x3)-np.log(x1))/x2
        
        return [x1,x2,x3,t0]
        
    def estimates_goodness(self):
        t_change = self.t_change
        obs = self.Total[:-t_change]
        t_measured = list(range(len(obs)))
        est = self.estimation()
        x1 = est[0]
        x2 = est[1]
        x3 = est[2]
        t0 = est[3]
        CR_fit = self.CR(t_measured,x1,x2,x3)
        print("The CR exponential growth parameters")
        print("x1=%.3f"%x1)
        print("x2=%.3f"%x2)
        print("x3=%.3f"%x3)
        print("t0=%.3f"%t0)
        print("Goodness of fit measures")
        print("R2 = ", r2_score(CR_fit, obs))
        print("explained variance score",explained_variance_score(obs,CR_fit))
        print("MSE = ", mean_squared_error(obs,CR_fit))
        
    def regression_report(self):
        t_change = self.t_change
        obs = self.Total[:-t_change]
        t_measured = list(range(len(obs)))
        est = self.estimation()
        x3 = est[2]
        y = np.log(obs-x3)
        X = pd.DataFrame(t_measured)
        lm = linear_model.LinearRegression()
        lm.fit(X,y)
        params = np.append(lm.intercept_,lm.coef_)
        predictions = lm.predict(X)
        newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
        MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))
        var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params/ sd_b
        p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
        sd_b = np.round(sd_b,3)
        ts_b = np.round(ts_b,3)
        #p_values = np.round(p_values,3)
        params = np.round(params,4)
        myDF3 = pd.DataFrame()
        myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["P-values"] = [params,sd_b,ts_b,p_values]
        print(myDF3)
        
    def get_fit(self):
        t_change = self.t_change
        obs = self.Total[:-t_change]
        t_measured = list(range(len(obs)))
        est = self.estimation()
        x1 = est[0]
        x2 = est[1]
        x3 = est[2]
        t0 = est[3]
        t0 = int(round(t0))

        if t0>0:
            obs1 = [0] * (len(t_measured) - t0)
            for i in range(len(t_measured)-t0):
                obs1[i] = obs[i+t0]
        elif t0<0:
            obs1 = [0] * (len(t_measured) - t0)
            for i in range(len(t_measured)):
                obs1[i-t0] = obs[i]
        else:
            obs1=obs
        
        t_model = list(range(len(obs1)))

        fit = [0] * len(t_model)
        for i in range(len(t_model)):
            fit[i] = x1 * np.exp(x2*(t_model[i]+t0))-x3
        
        return t_model,fit,obs1
        
    def plot_CR(self):
        t_model,fit,obs1 = self.get_fit()
        plt.plot(t_model, fit, '-', linewidth=2, label='fitted CR')
        plt.scatter(t_model, obs1, marker='*', color='orange', label='Observations')
        plt.legend()
        plt.xlim([0, max(t_model)])
        plt.ylim([0, 1.1 * max(fit)])
        plt.show()

    def plot_logCR(self):
        est = self.estimation()
        x3 = est[2]
        t_model,fit,obs1 = self.get_fit()
        log_fit = [0] * len(t_model)
        for i in range(len(t_model)):
            log_fit[i] = np.log(fit[i] +x3)

        plt.plot(t_model, log_fit, '-', linewidth=2,  label='fitted log CR')
        plt.scatter(t_model, np.log(obs1), marker='*', color='orange', label='log Observations')
        plt.legend()
        plt.xlim([0, max(t_model)])
        plt.ylim([0, 1.1 * max(log_fit)])
        plt.show()
    
    def preds_CR(self):
        t_change = self.t_change
        obs = self.Total[:-t_change]
        t_measured = list(range(len(obs)))
        date = self.date[len(obs):]
        test = self.Total[len(obs):]
        pred = [0] * len(test)
        est = self.estimation()
        x1 = est[0]
        x2 = est[1]
        x3 = est[2]
        for i in range(len(test)):
            pred[i] = x1 * np.exp(x2*(i + len(t_measured)))-x3
    
        df = {'date': date, 'predictions': pred,'observations': test}
        df = pd.DataFrame(df)
        return df
        
        
        






# In[94]:
class SAsIQR:
    def __init__(self, beta=1/6, tau1=0.251, tau2=0.778, gama=1./12, mu=1/25, alpha1=.2,f=3,confinement_hour=12,efficacite_Conf=0.3,start_conf=np.infty) :
        self.beta=beta#1./10#0.01 #taux de passage de l'etat asymptomatique a infecte
        self.tau1=tau1#0.0 # taux de mise en quarantaine
        self.tau2=tau2#0.0 # taux de mise en quarantaine
   
        self.gama=gama#0.5 # taux de gerison des quaranteme
        self.mu=mu#0.4 # taux de mortalité des quarantene 
        
        self.f=f
        self.alpha=alpha1
        self.alpha2=f*alpha1
        
        self.confinement_hour=confinement_hour
        self.start_conf=start_conf
        self.efficacite_Conf=efficacite_Conf # %des personne qui ne respetent pas le confinement
        
    ###################################
    def paramEstimation(self,N,x1,x2,x3):
        ## in this par we evaluate the parameter of the model using the method of @article{Liu2020}
        #author = {Liu, Zhihua and Magal, Pierre and Seydi, Ousmane and Webb, Glenn},
        #journal = {Biology},
        #number = {3},
        #title = {{Understanding unreported cases in the COVID-19 epidemic outbreak in Wuhan, xna, and the importance of major public health interventions}},
        #volume = {9},
        #year = {2020}
        tau=self.tau2/self.tau1
        gama1=self.gama+self.mu   
        h0=x2*x3/self.tau1  
        ass0=h0*(x2+self.tau2+gama1)/(x2+self.tau2+self.beta*tau+gama1)
        i0= self.beta*h0/(x2+self.tau2+self.beta*tau+gama1)
        q0=0#tau1*h0/(x2+gama1)
        d0=0
        r0=0
        s0=N-(ass0+i0+q0+d0+r0)   
        self.alpha1=(x2+self.beta+self.tau1)/((self.f*self.beta/(x2+self.tau2+gama1)+1)*s0)
        self.alpha2=self.f*self.alpha1
        #################
        #param=(beta,tau1,tau2) 
        y0=[s0,ass0,i0,q0,d0,r0]
        return(y0)

    #####################   
    def f1(self,s,ass,i,t): # taux d'infecté par un assym 
       # start_conf=21 # the begining of the confinement in day
        end_conf=np.infty # end of the confinment in day 
        tt=t-np.floor(t)
        if t>=self.start_conf and t<end_conf: # first day of confinement
            if ( (tt>=(1-self.confinement_hour/24))):
                b=self.alpha1*(1-self.efficacite_Conf)   
            else:
                  b=self.alpha1*0.15
        else:
            b=self.alpha1
        y=float(b*s)
        return y#a*s**2/(0.0*(i+ass)+s)

    def f2(self,s,ass,i,t): # taux d'infecté par un infecté
        k=self.alpha2/self.alpha1
        return k*self.f1(s,ass,i,t) 

    ####################################""
    def func(self,y,t):#S,beta,T_l,T_n): #Matrice de chaque stade
        s,ass,i,q,d,r=y # param=[si,beta(si),T_l(si), T_n(si)] 
        dsdt=-i*self.f2(s,ass,i,t)-ass*self.f1(s,ass,i,t)
        dassdt=-dsdt - ass*(self.beta+self.tau1)
        didt= self.beta*ass-(self.mu+self.tau2+self.gama)*i  
        dqdt=self.tau1*ass+self.tau2*i-(self.mu+self.gama)*q
        dddt=self.mu*(q+i)
        drdt=-(dsdt+dassdt+didt+dqdt+dddt) #gama1*qq+gama2*ii #    
        return([dsdt,dassdt,didt,dqdt,dddt,drdt])
    
    ####################################""
    def calSol(self,timeSet,y0):
       # print(ddata.popNumber)
        sol = integ.odeint(self.func, y0, timeSet)
        self.s=sol[:,0]    
        self.ass=sol[:,1]
        self.i=sol[:,2]
        self.q=sol[:,3]
        self.d=sol[:,4]
        self.r=sol[:,5]
        return timeSet, sol
    ####################################""
    def R0(self,s0):
        y=(self.alpha1+self.alpha2*self.beta/(self.tau2+self.gama+self.mu))*s0/(self.beta+self.tau1)
        return(y)

   ##############################################
    def plotSAsQI(self, data,timeSet):
    #Fig 1:  données + i + assy  
        fig, ax = plt.subplots()
        line1,= plt.plot(data.date,data.Cases,c='orange',marker='*',label='it interpolation')
        line2,=ax.plot(timeSet,self.q,c='orange',label='q')
        line3,=ax.plot(timeSet,self.ass,label='ass')
        line4,=ax.plot(timeSet,self.i,label='i')
      #  line3,=ax.plot(timeSet,self.i+self.ass,label='i+ass')     
        plt.xlabel(r'time in day',fontsize=12)
        plt.ylabel(r"population density",fontsize=12)
        #plt.title()
        ax.legend(loc='upper right')
        #plt.savefig('../results/fig111.png', transparent=True)
        plt.show()
        print ('Epidemimiological pic = %d, I+Q=%.d, Death num D=%.d \n'% (timeSet[np.argmax(self.i+self.q)],max(self.i+self.q),np.max(self.d)))

    ###################################"
    def plotRatios(self, data,timeSet):
    # Fig 2: ratio Ass/I
        fig, ax = plt.subplots()
        
        ax.plot(timeSet,self.i/self.ass, label='i/ass')
        ax.plot(timeSet,self.r/self.q,label='r/q')
    
        plt.xlabel(r'time in  day',fontsize=12)
        plt.ylabel(r"ratio",fontsize=12)
        #plt.title()
        ax.legend(loc='upper right')
        #plt.savefig('../results/fig111.png', transparent=True)
        plt.show()
    
    # ############################################
    def plotRD(self, data,timeSet):
   # Fig 3:  R et D
        fig, ax = plt.subplots()
        line1,= plt.plot(data.date,data.cumDeath,c='orange',marker='*' ,label='death')
        line2,= plt.plot(data.date,data.cumRecover,c='green',marker='+' ,label='recovered')
        line5,=ax.plot(timeSet,self.d,c='orange',label='d')
        line6,=ax.plot(timeSet,self.r,c='green',label='r')
        plt.xlabel('time in  day',fontsize=12)
        plt.ylabel(r"$density$",fontsize=12)
        ax.legend(loc='upper right')
        #plt.savefig('../results/fig112.png', transparent=True)
        plt.show()


# In[94]:
# class opti:
#      def __init__(self, model='corona', param=param, data=data):
#         self.beta=beta#1./10#0.01 #taux de passage de l'etat asymptomatique a infecte
#         self.tau1=tau1#0.0 # taux de mise en quarantaine
#         self.tau2=tau2#0.0 # taux de mise en quarantaine
   
#         self.gama=gama#0.5 # taux de gerison des quaranteme
#         self.mu=mu#0.4 # taux de mortalité des quarantene 
        
#         self.f=f
#         self.alpha=alpha1
#         self.alpha2=f*alpha1
        
#         self.confinement_hour=confinement_hour
#         self.start_conf=start_conf
#         self.efficacite_Conf=efficacite_Conf # %des personne qui ne respetent pas le confinement
        
        
        
#     def optFuncDim2(param,data):
#         res=np.zeros(len(data.shape))
#         ndp, nbatch = data.shape[0]
#         time = data.xdata[0][:,0]
#         y1data = data.ydata[0][:, 0] # cases
#       #  y2data = data.ydata[1][:, 0] # death
#       #  y3data = data.ydata[2][:, 0] # recovery
#         xdata = data.xdata[0][:,0]#data.user_defined_object[0]
#         #par=(param[0],param[1],param[2])
#         corona.tau1=param[0]
#         corona.tau2=param[1]
    
    
#         corona.calSol(time,y0)
    
#         bb=1#param[4]
#         ## l'optimisation est sur la base des i+ass+q
#         # aa=q[(np.floor(timeSet)==timeSet)*(timeSet<len(ddata.date))]+ass[(np.floor(timeSet)==timeSet)*(timeSet<len(ddata.date))]+i[(np.floor(timeSet)==timeSet)*(timeSet<len(ddata.date))]
#           ## l'optimisation est sur la base des i+q
#         aa=corona.q#+corona.i
#         dd=corona.d
#         rr=corona.r
#         res[0]= np.linalg.norm(aa-y1data)
#        # res[1]=np.linalg.norm(dd-y2data)
#      #   res[2]=np.linalg.norm(rr-y3data)
#         return res
    
#     def optim(self)
#         # initialize MCMC object
#         mcstat = MCMC()
#         # initialize data structure 
#         #OPtFuncDim1
#         # ydata=np.column_stack((ddata.Cases,ddata.cumDeath))
#         # mcstat.data.add_data_set(x=ddata.date, y=ydata)#, user_defined_object=ddata.date)
        
#         mcstat.data.add_data_set(x=ddata.date[:27], y=ddata.Cases[:27],weight=10)#, user_defined_object=ddata.date)
#         #mcstat.data.add_data_set(x=ddata.date[:27], y=ddata.cumDeath[:27],weight=0)#, user_defined_object=ddata.date)
#         #mcstat.data.add_data_set(x=ddata.date[:27], y=ddata.cumRecover[:27],weight=0)#, user_defined_object=ddata.date)
#         # initialize parameter array
        
#         # add model parameters
#         mcstat.parameters.add_model_parameter(name='tau1', theta0=tau1, minimum=0.25,maximum=0.26, prior_mu=tau1, prior_sigma=0.1)
#         mcstat.parameters.add_model_parameter(name='tau2', theta0=tau2, minimum=0.75,maximum=0.8,prior_mu=tau2,prior_sigma=0.01)
#         #mcstat.parameters.add_model_parameter(name='efficacite_Conf', theta0=efficacite_Conf, minimum=efficacite_Conf*0.8,maximum=efficacite_Conf*1.2)
         
                                         
#         # # initial values for the model states
#         # mcstat.parameters.add_model_parameter(name='S0', theta0=0.77, minimum=0,
#         #                                       maximum=np.inf, prior_mu=0.77,
#         #                                       prior_sigma=2)
#         # mcstat.parameters.add_model_parameter(name='I0', theta0=1.3, minimum=0,
#         #                                       maximum=np.inf, prior_mu=1.3,
#         #                                       prior_sigma=2)
#         # mcstat.parameters.add_model_parameter(name='R0', theta0=10, minimum=0,
#         #                                       maximum=np.inf, prior_mu=10,
#         #                                       prior_sigma=2)
        
#         # Generate options
#         mcstat.simulation_options.define_simulation_options(
#             nsimu=1.5e+5, updatesigma=True)
#         # Define model object:
#         mcstat.model_settings.define_model_settings(
#             sos_function=optFuncDim2)#,
#          #   sigma2=0.05**2)#,
#            # S20=np.array([1,1]))
#            # N0=np.array([4,4,4]))
#         mcstat.run_simulation()
#     # Rerun starting from results of previous run
#     #mcstat.simulation_options.nsimu = int(3e+4)
#     #mcstat.run_simulation(use_previous_results=True)
#     results = mcstat.simulation_results.results
#     burnin = int(results['nsimu']/2)
#     chain = results['chain'][burnin:, :]
#     s2chain = results['s2chain'][burnin:, :]
#     names = results['names'] # parameter names
    
#     # display chain stats
#     mcstat.chainstats(chain, results)
    
#     from pymcmcstat import mcmcplot as mcp
#     settings = dict(
#         fig=dict(figsize=(7, 6))
#     )
#     # plot chain panel
#     mcp.plot_chain_panel(chain, names, settings)
#     # plot density panel
#     mcp.plot_density_panel(chain, names, settings)
#     # pairwise correlation
#     f = mcp.plot_pairwise_correlation_panel(chain, names, settings)





    
#     def predmodelfun(data,param):
#         obj = data.xdata
#         time = obj[0][:,0]
#         xdata = obj
#         # last 3 parameters are the initial states
#         #y0 = np.array(q[-3:])
    
#         corona.tau1=param[0]
#         corona.tau2=param[1]
    
#         # evaluate model
#         #ymodel = np.zeros([time.size, 3])
#         tmodel, ymodel = corona.calSol(time,y0)
#         return ymodel
    
#     def plotResult()
    
#         mcstat.PI.setup_prediction_interval_calculation(
#             results=results,
#             data=mcstat.data,
#             modelfunction=predmodelfun)
        
#         mcstat.PI.generate_prediction_intervals(
#             nsample=500,
#             calc_pred_int=True,
#             waitbar=True)
        
        
#         # plot prediction intervals
#         fighandle, axhandle = mcstat.PI.plot_prediction_intervals(
#             adddata=False,
#             addlegend=False,
#             figsizeinches=[7.5, 8])
#         for ii in range(1):
#             axhandle[ii].plot(mcstat.data.ydata[0][:, 0],
#                             #  mcstat.data.ydata[0][:, ii + 1],
#                               'ko', mfc='none', label='data')
#             axhandle[ii].set_ylabel('')
#             #axhandle[ii].set_title(ylbls[ii + 1][0])
#             axhandle[ii].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#         axhandle[-1].set_xlabel('days');
    
#      # plor result    
#         tmax=40
#         timeSet=np.arange(ddata.t0,tmax,dt)
        
#         param=[[0.25,0.77],[0.251,0.775],[0.252,0.78]]
#         #print('(beta, efficacite_Conf,tau2)= (%.4f,%.4f,%.4f)' % (results['mean'][0],results['mean'][1],results['mean'][2]))
#         print('(alpha, beta)= (%.4f,%.4f)' % (results['mean'][0],results['mean'][1]))
        
#         ress=[]
        
#         for i in np.arange(len(param)):
#             tau1=param[i][0]#0.0 # taux de mise en quarantaine pour les infectes
#             tau2=param[i][1]
#             tau=tau2/tau1
    
#             p=0.2
#             mu=p*mu
#             gama=(1-p)*mu+gama
        
#             h0=ddata.chi2*ddata.chi3/tau1  
        
#             ass0=h0*(ddata.chi2+tau2+gama1)/(ddata.chi2+tau2+beta*tau+gama1)
        
#             i0= beta*h0/(ddata.chi2+tau2+beta*tau+gama1)
#             q0=0#tau1*h0/(ddata.chi2+gama1)
#             d0=0
#             r0=0
#             s0=ddata.popNumber-(ass0+i0+q0+d0+r0)#1.-(ass0+i0+q0+d0+r0) #if s<0.2 else 0
        
#             corona.alpha=(ddata.chi2+beta+tau1)/((3*beta/(ddata.chi2+tau2+gama1)+1)*s0)
#             y0=[s0,ass0,i0,q0,d0,r0]
            
#             time,res= corona.calSol(timeSet,y0)
#             ress.append(np.array([param[i][0],param[i][1],corona.q]))
            
        
#         fig, ax = plt.subplots()
#         ax.grid(True)
#         line1,= plt.plot(ddata.date[:tmax],ddata.Cases[:tmax],c='orange',marker='*',label='TUN interpolation')
#         #line2,=ax.plot(timeSet,res[:,2])
#         #line3,=ax.plot(timeSet,res1[:,2])
#         ax.fill_between(timeSet,ress[2][2],ress[0][2],alpha=0.2)
#         line3,=ax.plot(timeSet,ress[1][2],label='q(t)')
        
#         #plt.axvline(timeSet[(timeSet<19+dt) * (timeSet>19)], color='black', linestyle='-.', alpha=.5)
#         #plt.axvline(timeSet[(timeSet<24+dt) * (timeSet>24)], color='black', linestyle='-.', alpha=.5)
        
#         ax.text( 1, 6,
#                 '6PM-6AM curfew ',
#                   rotation=90,
#                   horizontalalignment='center',
#                   verticalalignment='top',
#                   multialignment='center',
#               size=10)
#         plt.xlabel(r'time in day',fontsize=12)
#         plt.ylabel(r"population density",fontsize=12)
#         #plt.title()
#         ax.legend(loc='upper left')
#         #plt.savefig('../results/fig111.png', transparent=True)
#         plt.show()
            
