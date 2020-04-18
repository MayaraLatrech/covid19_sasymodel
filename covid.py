#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:06:31 2020

@author: slimane ben miled & mayara latrach
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

from lmfit import Parameters

# In[2]: Data input 


class data:   
    def __init__(self,data,N=11791749, t_change =25):
        # INIT de la dabase TUN
        #self.contry=contry
     
        # self.popNumbers={'Italy':60.4e6/250, 'Tunisia':630000,'France':67e6/250,'Spain':46.5e6/250}
        #Popxna=1e9/1000
        #self.dataset_contry =pd.read_excel(r'/home/slimane/biologie/corona/dataBase/TUNdataBase.xlsx')       
        
        #self.dataset_contry =pd.read_csv('/home/slimane/biologie/corona/dataBase/TUNdataBase.csv')       
         
        
        #a= self.dataset_contry['Nb_cas_journalier'].replace(np.nan, 0).values
        #self.newconfirmed=a[np.argmin(a==False):]
#        self.newCases=np.flip(self.dataset_contry['cases'].values,0)

        #b= self.dataset_contry['retablie_journalier'].replace(np.nan, 0).values
        #self.newRecover=b[np.argmin(a==False):]
        #self.cumRecover=np.cumsum(self.newRecover)

        

     
        #b=self.dataset_contry['Décès_journalier'].replace(np.nan, 0).values
        #self.newDeath=b[np.argmin(a==False):]
        #self.Death=np.cumsum(self.newDeath)
        self.population=N
        self.date = data['date']
        self.confirmed = data['confirmed']
        self.deaths = data['deaths']
        self.recovered = data['recovered']
        
        self.t_change = t_change
        #self.cumCases = np.cumsum(self.newCases,0)
        #self.Cases = np.cumsum(self.newCases-self.newDeath-self.newRecover,0)
        
        #lf.ratio=self.newCases[1:len(self.newCases)]/self.newCases[0:len(self.newCases)-1]

        #self.date = self.dataset_contry['Dates']
        #self.Deaths= self.dataset_contry['Deaths']


        
    def get_popnumberDatabase(self):    
        y=self.population
        return(y)
    
    def plotData(self):
        #recov = self.Total - (self.Cases -self.Deaths)
        plt.plot(self.confirmed,"b-")
        plt.plot(self.deaths,"r-")
        plt.plot(self.recovered,"g-")
        plt.legend(["Confirmed cases","Deaths","Recovered"])
        plt.show()
        
    def CR(self,t, x1, x2, x3):
            res = [0]*len(t)
            for i in range(len(t)):
                res[i] = x1 * np.exp(x2*t[i])-x3
            return res
        
    def estimation(self):
        t_change = self.t_change
        obs = self.confirmed[:t_change]
        t_measured = list(range(len(obs)))

        popt, pcov = op.curve_fit(self.CR, t_measured, obs)

        x1 = popt[0]
        x2 = popt[1]
        x3 = popt[2]
        t0 = (np.log(x3)-np.log(x1))/x2
        
        return [x1,x2,x3,t0]
        
    def estimates_goodness(self):
        t_change = self.t_change
        obs = self.confirmed[:t_change]
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
        obs = self.confirmed[:t_change]
        t_measured = list(range(len(obs)))
        est = self.estimation()
        x3 = est[2]
        y = [0]*len(obs)
        ob = obs+x3
        for i in range(len(y)):
            if ob[i] >0:
                y[i] = np.log(ob[i])
            else:
                y[i] = 0
        #y = np.log(obs-x3)
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
        obs = self.confirmed[:t_change]
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
        log_obs = [0]*len(obs1)   
        for i in range(len(t_model)):
            log_fit[i] = np.log(fit[i] +x3)
            log_obs[i] = np.log(obs1[i])
        plt.plot(t_model, log_fit, '-', linewidth=2,  label='fitted log CR')
        plt.scatter(t_model, log_obs , marker='*', color='orange', label='log Observations')
        plt.legend()
        plt.xlim([0, max(t_model)])
        plt.ylim([0, 1.1 * max(log_fit)])
        plt.show()
    
    def preds_CR(self):
        t_change = self.t_change
        obs = self.confirmed[:t_change]
        t_measured = list(range(len(obs)))
        date = self.date[len(obs):]
        test = self.confirmed[len(obs):]
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
        
        
        
class SAsIQR:
    def __init__(self, beta=1/6, tau1=0.251, tau2=0.778, gama=1./12, mu=1/25, alpha1=.2,f=3,confinement_hour=12,efficacite_Conf=0.3,start_conf=np.infty) :
        self.beta=beta#1./10#0.01 #taux de passage de l'etat asymptomatique a infecte
        self.tau1=tau1#0.0 # taux de mise en quarantaine
        self.tau2=tau2#0.0 # taux de mise en quarantaine
   
        self.gama=gama#0.5 # taux de gerison des quaranteme
        self.mu=mu#0.4 # taux de mortalité des quarantene 
        
        self.f=f
        self.alpha1=alpha1
        self.alpha2=f*alpha1
        
        self.confinement_hour=confinement_hour
        self.start_conf=start_conf
        self.efficacite_Conf=efficacite_Conf # %des personne qui ne respetent pas le confinement
        
    ###################################
    def calc_params(self,database, t_change):
        ## in this par we evaluate the parameter of the model using the method of @article{Liu2020}
        #author = {Liu, Zhihua and Magal, Pierre and Seydi, Ousmane and Webb, Glenn},
        #journal = {Biology},
        #number = {3},
        #title = {{Understanding unreported cases in the COVID-19 epidemic outbreak in Wuhan, xna, and the importance of major public health interventions}},
        #volume = {9},
        #year = {2020}
        #database.t_change=t_change
        [x1,x2,x3,t0] = database.estimation()
        tau=self.tau2/self.tau1
        gama1=self.gama+self.mu   
        h0=x2*x3/self.tau1  
        ass0=h0*(x2+self.tau2+gama1)/(x2+self.tau2+self.beta*tau+gama1)
        i0= self.beta*h0/(x2+self.tau2+self.beta*tau+gama1)
        q0=self.tau1*h0/(x2+gama1)
        d0=0
        r0=0
        s0=database.population-(ass0+i0+q0+d0+r0)   
        self.alpha1=(x2+self.beta+self.tau1)/((self.f*self.beta/(x2+self.tau2+gama1)+1)*s0)
        self.alpha2=self.f*self.alpha1
        #################
        #param=(beta,tau1,tau2) 
        y0=[s0,ass0,i0,q0,d0,r0]
        res=[self.alpha1]
        return y0, res

    #####################   
    def f1(self,s,ass,i,t): # taux d'infecté par un assym 
       # start_conf=21 # the begining of the confinement in day       
        end_conf=np.infty # end of the confinment in day 
        tt=t-np.floor(t)
        if t>=self.start_conf and t<end_conf: # first day of confinement
           # b=self.alpha1*np.exp(-self.efficacite_Conf*(t-self.start_conf))   
            if ( (tt>=(1-self.confinement_hour/24))):
               b=self.alpha1*(1-self.efficacite_Conf)   
            else:
               b=self.alpha1*0.01
        else:
            b=self.alpha1
        y=float(b*s)
        return y#a*s**2/(0.0*(i+ass)+s)

    def f2(self,s,ass,i,t): # taux d'infecté par un infecté
        return self.f*self.f1(s,ass,i,t) 

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
        plt.grid(axis='x', color='0.95')
        line1,= plt.plot(data.date,data.confirmed,c='orange',marker='*',label='it interpolation')
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



# In[93]:        
class SIRU:
    def __init__(self,dataset, S0=11791749,f=0.8,nu=1/7,eta=1/7, t_change =25):
        self.S0 = S0
        self.f = f
        self.nu = nu
        self.eta = eta
        self.t_change = t_change
        self.dataset =dataset
        
    def calc_params(self):
        dat = data(self.dataset,self.t_change)
        [x1,x2,x3,t0] = dat.estimation()
        nu1 = self.f * self.nu
        nu2 = (1-self.f) * self.nu
        I0 = x3 *x2 /(self.f*self.eta)
        tau = (x2+self.eta)*(self.nu+x2)/(self.S0*(nu2+self.eta+x2))
        U0 =nu2*I0/(self.eta+x2)
        R0 = 0
        y0 = [self.S0,I0,R0,U0]
        pars = [tau,self.nu,nu1,nu2,self.eta]
        return y0, pars
        
    def model(self,y, t, paras):

        S = y[0]
        I = y[1]
        R = y[2]
        U = y[3]
        
        try:
            tau = paras['tau'].value
            nu = paras['nu'].value
            nu1 = paras['nu1'].value
            nu2 = paras['nu2'].value
            eta = paras['eta'].value
            mu = paras['mu'].value          
        except KeyError:
            tau, nu, nu1, nu2, eta, mu = paras
                
        def tauu(t,tau0):
            if t<24:
                to = tau0
            else:
                to= tau0*np.exp(-mu*(t-24))
            return to
        to = tauu(t,tau)
        # the model equations
        dS = -to * S * (I+U)
        dI = to * S * (I+U) - nu * I
        dR = nu1 * I - eta * R
        dU = nu2 * I - eta * U
        return [dS,dI,dR,dU]
    
    def g(self,t, y0, paras):
        sol = integ.odeint(self.model, y0, t, args=(paras,))
        return sol
    def residual(self, paras , t, data):
        y0, paras = self.calc_params()
        mod = self.g(t, y0, paras)
        x2_model = mod[:, 2]
        return (x2_model - data).ravel()
    def params_model(self):
        y0,pars = self.calc_params()
        params = Parameters()
        params.add('tau', value=pars[0], vary=False)
        params.add('nu', value=pars[1], vary=False)
        params.add('nu1', value=pars[2], vary=False)
        params.add('nu2', value=pars[3], vary=False)
        params.add('eta', value=pars[4], vary=False)
        params.add('mu', value=0.05)
        return params
    def integrate(self):
        y0,pars = self.calc_params()
        params = self.params_model()
        dat = data(self.dataset,self.t_change)
        [x1,x2,x3,t0] = dat.estimation()
        t0 = int(round(t0))
        t_int = list(range(len(dat.confirmed)-t0))
        t = list(range(200))
        dat = data(self.dataset,self.t_change)
        t_model,fit,obs1 =dat.get_fit()
        result = op.minimize(self.residual, params, args=(t_int, obs1), method='L-BFGS-B')  
        data_fitted = self.g(t, y0, result.params)
        return data_fitted , t , t_int
    
    def plot_fit_data(self):
        data_fitted, t, t_int = self.integrate()
        dat = data(self.dataset,self.t_change)
        [x1,x2,x3,t0] = dat.estimation()
        t0 = int(round(t0))
        dd = dat.confirmed
        obs1 = [0] * len(t_int)
        if t0>0:  
            for i in range(len(dd)-t0):
                obs1[i] = dd[i+t0]
        elif t0<0:
            for i in range(len(dd)):
                obs1[i-t0] = dd[i]
        else:
            obs1=dd
        
        plt.scatter(t_int, obs1, marker='o', color='b', label='Observations')
        plt.plot(t_int, data_fitted[:len(t_int),2] , '-', linewidth=2, color='black', label='fitted reported symptomatic')
        plt.scatter(t_int, data_fitted[:len(t_int),3] , marker='o', color='r', label='unreported symptomatic')
        plt.legend()
        plt.xlim([0, max(t_int)])
        plt.show()
    
    def plot_pred(self):
        data_fitted, t, t_int = self.integrate()
        plt.plot(t, data_fitted[:,2] , '-', linewidth=2, color='black', label='fitted reported symptomatic')
        plt.plot(t, data_fitted[:,3] , '-', linewidth=2, color='r', label='unreported symptomatic')
        plt.legend()
        plt.xlim([0, max(t)])
        plt.show()      
        
