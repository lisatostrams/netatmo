# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:52:02 2016

@author: Lisa Tostrams
"""

import datetime
import numpy as np   
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata
import pandas as pd
import scipy as sp
import geopy.distance as gp
import copy
import time
import random




def get_timestamps_temps(data_i):
    datesdata = data_i['thermo_module']['valid_datetime']
    tempdata = data_i['thermo_module']['temperature']
    thermo_mod = []
    i = 0
    for date in datesdata:   
        thermo_mod.append(((datetime.datetime.fromtimestamp(date['$date']/1000)-datetime.timedelta(minutes=120)).strftime('%Y-%m-%d %H:%M:%S'), tempdata[i]))
        i+=1
    thermo_mod = sorted(thermo_mod, key=lambda date: date[0])  
    index = pd.DatetimeIndex([t[0] for t in thermo_mod])
    series = [t[1] for t in thermo_mod]
    time_series = pd.Series(series, index=index)
    return time_series

def create_thermo_module_dataset():
    from data import data    
    Thermo_Module = []
    for i in range(0,len(data)):
        Thermo_Module.append(({'name': data[i]['_id'],'longitude': data[i]['longitude'],
                               'latitude':data[i]['latitude'], 'measurements': get_timestamps_temps(data[i]) }))
    #data = pd.DataFrame(Thermo_Module)
    return Thermo_Module
    
def plot_thermo_mods(ther_data, plot_hours, m, name):
    import matplotlib.dates as mdates
    days = mdates.DayLocator()
    daysFMT = mdates.DateFormatter('%d') 
    hours = mdates.HourLocator()
    daysFMT = mdates.DateFormatter('%d') 
    hoursFMT = mdates.DateFormatter('%H')
    fig,ax= plt.subplots() 
    for i in range(0, len(ther_data)):
        curr = ther_data[i]
        #time = make_time_plottable(curr)
        #if len(curr[1]) == len(time): 
        ax.plot_date(curr['measurements'].index, curr['measurements'].tolist(), '-.',lw = 0.8, c='darkcyan')
    ax.plot_date(m[0],m[1], '-',color = 'black', linewidth = 1.2, label='CWS mean')
    ax.plot(knmi[0],knmi[1], '-', color ='red', linewidth = 1.2, label='KNMI measurements')
    plt.xlabel('Day in month')
    plt.ylabel('Temperature in degrees celsius')
    fig.set_size_inches(6,4)
 #   ax.set_ylim(0,40)
  #  plt.legend()
    plt.legend(loc=3, borderaxespad=0.)
    ax.set_ylim(-10,40)
    if plot_hours:
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hoursFMT)
    else:
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(daysFMT)
        datemin = datetime.date(2016, 4, 1)
        datemax = datetime.date(2016, 5, 1)
        ax.set_xlim(datemin, datemax)
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
   # plt.grid() 
    plt.title('{}'.format(name))
    plt.savefig('figures/All temp/{} with knmi.png'.format(name), dpi=600)
    plt.show()
          
def plot_station(station, plot_hours, m, name):
    import matplotlib.dates as mdates
    days = mdates.DayLocator()
    daysFMT = mdates.DateFormatter('%d') 
    hours = mdates.HourLocator()
    daysFMT = mdates.DateFormatter('%d') 
    hoursFMT = mdates.DateFormatter('%H')
    fig,ax= plt.subplots() 
    ax.plot(station['measurements'], '-',lw = 1.5, c='darkcyan')
  #  ax.plot_date(m[0],m[1], '-',color = 'black', linewidth = 2, label='CWS mean')
    ax.plot(knmi[0],knmi[1], '-', color ='red', linewidth = 2, label='KNMI measurements')
    plt.xlabel('Day in month')
    plt.ylabel('Temperature in degrees celsius')
    fig.set_size_inches(6,4)
 #   ax.set_ylim(0,40)
  #  plt.legend()
    plt.legend(loc=3, borderaxespad=0.)
    ax.set_ylim(-10,40)
    if plot_hours:
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hoursFMT)
    else:
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(daysFMT)
        datemin = datetime.date(2016, 4, 2)
        datemax = datetime.date(2016, 4, 15)
        ax.set_xlim(datemin, datemax)
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
   # plt.grid() 
    plt.title('{}'.format(name))
    plt.savefig('figures/All temp/station {} with knmi.png'.format(name), dpi=600)
    plt.show()
    
def resample(thermo_dataset):
    new_thermo_dataset = []  
    nanpct = [] 
    for i in range(0,len(thermo_dataset)):
        print(i)
        current_station = thermo_dataset[i]
        temp_series = current_station['measurements']
        
        if(len(temp_series.tolist()) > 0):
            series = temp_series.tolist()
            not_nan_pct = np.sum(~np.isnan(np.asarray(series)))/len(series)
            nanpct.append(not_nan_pct)
            print(not_nan_pct)
            if not_nan_pct > 0.25:
                
                res_temp= temp_series.resample('10T').mean().bfill()
                dates = pd.date_range('2016-03-31 23:30:00', '2016-05-01 05:00:00', freq = '10Min')
                d = pd.DatetimeIndex(dates)
                res_re_temp = res_temp.reindex(d, method='nearest')
                new_station = copy.copy(current_station)
                new_station['measurements'] = res_re_temp
                new_thermo_dataset.append(new_station)
            
    return new_thermo_dataset, nanpct
    
def plot_points_slice(day, time):
    fig = plt.figure()
    date = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    temp_slice = [d['measurements'][time] for d in day]
    for i in range(0, len(day)):
        station = day[i]
        temp = station['measurements'][date]
        temp_slice.append(temp)
    plt.scatter([s['longitude'] for s in day], 
             [s['latitude'] for s in day], 
            c = temp_slice, 
            s=30,
            lw = 0.1,
            cmap = 'rainbow')   
    plt.colorbar(label='temperature in degrees Celsius')
    date_time = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S')
    plt.title('station locations and measurements on {} April, at {}:{}0'.format(date_time.day, date_time.hour, date_time.minute))
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.savefig('figures/Spatial analysis/locations_slice_filtered_data_{}_{}.png'.format(date_time.day, date_time.hour), dpi=600)
    plt.close()
    return ([s['longitude'] for s in day], [s['latitude'] for s in day], temp_slice), temp_slice

def plot_points_average(data, name):
    fig = plt.figure()
    temp_mean = []
    for d in data:
        temp_mean.append(np.nanmean(d['measurements']))
    plt.scatter([s['longitude'] for s in data], 
             [s['latitude'] for s in data], 
            c = temp_mean, 
            s=30,
            lw =0.1,
            cmap = 'rainbow', vmin=9.2, vmax = 10.7) 
    plt.colorbar(label='temperature in degrees Celsius')
    plt.title('station locations and average measurements')
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.savefig('figures/Spatial analysis/locations_average_{}.png'.format(name), dpi=600)
    plt.close()
    return ([s['longitude'] for s in data], [s['latitude'] for s in data], temp_mean )
     

def get_mean_faster(thermo_data):
    dates = thermo_data[0]['measurements'].index
    temp_data = [d['measurements'].tolist() for d in thermo_data]
    array_data = np.asarray(temp_data)
    temp_means = np.nanmean(array_data, axis=0)
    times_list = dates
    temp_std = np.nanstd(array_data, axis=0)
    print(len(temp_means))
    return (times_list, temp_means, temp_std)
            
def remove_outliers(thermo_data, m):
    t = m[0]
    means = m[1]
    stds = m[2] 
    new_data = []
    var_knmi = np.nanstd(knmi[1])    
    correlations = []
    z_scores = []
    for i in range(0, len(thermo_data)):
        station = thermo_data[i]
        temp_series = station['measurements'] 
        #times = station[0]
        temps = temp_series.tolist()
        new_temps = []        
        var_station = np.nanstd(temps)
        corr = temp_series.corr(knmi_timeseries)
        correlations.append(corr)
        if  corr > 0.3 and var_station > 2 :  #some var needed to prevent runtime errors..
            print(corr)
            temps = []
            temps = temps+temp_series['20160331'].tolist()
            
            for i in range(1,31):
                date = datetime.datetime(2016,4,i)
                td = datetime.datetime.strftime(date, '%Y%m%d')
                today = temp_series[td]
                today_knmi = knmi_timeseries[td]
                corr_today = today.corr(today_knmi)
                if corr_today < 0.3 or np.isnan(corr_today):
                    print('on day {} r={}'.format(i,corr_today))
                    today[td] = np.nan
                temps = temps+today.tolist()
            temps = temps + temp_series['20160501'].tolist()
            z = []
            for j in range(0, len(t)):
                c= j
                #c = times.index(t[j])
                Z = (temps[c]-means[j])/stds[j]
                z.append(Z)
                if temps[c] > means[j]+(3*stds[j]) or temps[c] < means[j]-(3*stds[j]):
                    new_temps.append(np.nan)
                elif j > 3:
                    if temps[c] is temps[c-1] and temps[c] is temps[c-2]:
                        new_temps.append(np.nan)
                    else:
                        new_temps.append(temps[c])
                else:
                    new_temps.append(temps[c])
            new_station = copy.copy(station)
            new_station['measurements'] = pd.Series(new_temps, index=temp_series.index)
            if sum(new_station['measurements'].isnull())/len(new_station['measurements']) < 0.5:
                new_data.append(new_station)
            z_scores.append(np.nanmean(z))
    return new_data, correlations, z_scores
       
  
def pca(data, name):
    import scipy.linalg as la
    explained_var = np.zeros([len(data), 30])
    new_data = [{'latitude':d['latitude'], 'longitude':d['longitude'], 'name':d['name'], 'measurements': d['measurements']['2016-03-31']} for d in data]
    for j in range(1,31): 
        date = datetime.datetime(2016,4,j)
        td = datetime.datetime.strftime(date, '%Y%m%d')
        X =[d['measurements'][td].tolist() for d in data]
        X = np.mat(np.asarray(X))
        wherenans = np.isnan(X)
        w = sum(wherenans.T)>0
        w = w.T
        idx = [i for i, k in enumerate(w) if k == True]
        X = np.delete(X, (idx), axis = 0)
        m = np.nanmean(X, axis=0)
    
        
        C = np.cov(X, rowvar=False)  # calc C
        S,V = la.eig(C)             # calc eigenvalues S and eigenvectors V.T
          
        V = V.T
        rho = abs((S)/(S).sum())
        sum_rho = np.cumsum(rho)    # explained var
        
        if PLOT:
            fig,ax = plt.subplots()
            ax.plot(rho, label = 'percentage per pc')
            ax.plot(sum_rho, label = 'summed percentage')
            plt.ylabel('Percentage explained')
            plt.xlabel('Principal component')
            ax.set_xlim(0,12)
            plt.title('Variance explained by principal components')
            plt.legend()
            plt.savefig('figures/PCA/rho_rhosum_{}.png'.format(name), dpi=600)
            plt.close()
            num = [2,10,25,50,100]
      
            days = mdates.DayLocator()
            daysFMT = mdates.DateFormatter('%d') 
            for i in num:
                Z = X * V[0:i-1,:].T
                W = Z* V[0:i-1,:]
                fig,ax= plt.subplots() 
                ax.plot(data[0]['measurements'][td].index, W.T, '--')
                ax.plot(data[0]['measurements'][td].index,m.T, 'k', label='mean')
                ax.plot(knmi_timeseries[td], 'r', label='KNMI station')
                ax.xaxis.set_major_locator(days)
                ax.xaxis.set_major_formatter(daysFMT)
        
            
                ax.set_ylim(-10,40)
                for label in ax.xaxis.get_ticklabels()[::2]:
                    label.set_visible(False)
                plt.ylabel('Temperature')
                plt.xlabel('Day in month')
                plt.legend()
                plt.grid()
                plt.title('Data reconstructed from first {} PC ({:.5f}% var explained)'.format(i,100*abs(sum_rho[i-1])))
                plt.savefig('figures/PCA/PCA_first_{}_pc_{}_data_knmi_{}.png'.format(i,name, j), dpi=600)  
                plt.close()
                
                fig,ax=plt.subplots()
                
                ax.plot(knmi_timeseries[td].tolist(), W.T, '.', c='seagreen', ms=5)
                min_ax = min(ax.get_xlim()[0], ax.get_ylim()[0])
                max_ax = max(ax.get_xlim()[1], ax.get_ylim()[1])
                fig.set_size_inches(5,5)
                ax.plot([min_ax, max_ax], [min_ax, max_ax], ls="--", color='k', lw = 2.0 )
                
                plt.ylabel('Measurements CWS')
                plt.xlabel('Measurement KNMI')
                plt.title('CWS vs KNMI measurements reconstruction {} PCs'.format(i))
                plt.savefig('figures/PCA/kcloud_{}_pcs_{}_{}.png'.format(i,name,j), dpi=300)
                plt.close()
        p =  sum(sum_rho < 0.999999)
        Z = X * V[0:p-1,:].T
        W = Z* V[0:p-1,:]
        idx = 0
        daymean = pd.Series(np.nanmean(W, axis=0).tolist()[0], index=data[0]['measurements'][td].index)
        for c in range(len(w)):
            if w[c] == False:
    
                s = pd.Series(W[idx,:].tolist()[0], index = data[c]['measurements'][td].index)
                new_data[c]['measurements'] = new_data[c]['measurements'].append(s)
                idx = idx+1
                explained_var[c][j-1] = daymean.corr(s)**2
            else:
                new_data[c]['measurements'] = new_data[c]['measurements'].append(data[c]['measurements'][td])
                explained_var[c][j-1] = daymean.corr(data[c]['measurements'][td])**2
  
    return new_data, explained_var
    
def kalman(station, model, m, i):
    #    
    n_iter = len(station['measurements'].tolist())
    sz = (n_iter,) # size of array
    x = model[0] # truth value (typo??? in example intro kalman paper at top of p. 13 calls this z)
    z = station['measurements'].tolist() # observations 
    N = knmi_timeseries.diff()
    Q =  0.1 # process variance

    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    #R =  np.var(model) # estimate of measurement variance
    #R = abs(station['measurements'].diff()) #changing variation 
    R = abs(np.asarray(z)-np.asarray(knmi[1]))
    # intial guesses
    xhat[0] = x
    P[0] = 1.0
    SE = []
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        #xhatminus[k] = xhat[k-1] + delta_m
        Pminus[k] = P[k-1]+Q  #static Q
    
        # measurement update
      #  K[k] = Pminus[k]/( Pminus[k]+R ) #static R
        
        if np.isnan(z[k]):
            xhat[k] = xhatminus[k]+N[k]
            P[k] = np.nan
        else:
            if np.isnan(R[k]) or R[k]+Pminus[k] < 0.3:
                R[k] = 1
            if np.isnan(xhatminus[k]):
                xhatminus[k] = z[k]
            if np.isnan(Pminus[k]):
                Pminus[k] = np.nanmean(P)
            K[k] = Pminus[k]/(Pminus[k]+R[k]) #dynamic R
            xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])

            P[k] = (1-K[k])*Pminus[k]
        SE.append((xhatminus[k]-z[k])**2)

   
    days = mdates.DayLocator()
    daysFMT = mdates.DateFormatter('%d') 
    idx = np.isnan(np.array(P))
    xhat[idx] = np.nan
    if PLOT:
        fig,ax= plt.subplots()
        ax.plot(station['measurements'].index,z,'k+',label='measurements')
        ax.plot(station['measurements'].index,xhat,'b-',label='a posteri estimate')
        ax.plot(knmi[0], knmi[1], 'r', label='KNMI station')
        ax.fill_between(station['measurements'].index, xhat-2*np.sqrt(P), xhat+2*np.sqrt(P), facecolor='blue', alpha=0.3)
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(daysFMT)
        datemin = datetime.date(2016, 4, 9)
        datemax = datetime.date(2016, 4, 15)
        ax.set_xlim(datemin, datemax)
        plt.legend(loc=4)
        plt.title('Estimate vs. iteration step', fontweight='bold')
        plt.xlabel('Day')
        plt.ylabel('Temperature')
        plt.savefig('figures/Kalman/kalman_model_mean_knmiplot_station_{}_R_{}.png'.format(Q,i,np.mean(R)), dpi=600)
        plt.close()
    return xhat,P, SE
    
def kalman_f(data):
    new_resampled_outlier = []
    i = 0
    m = get_mean_faster(data)
    mse = []
    for station in data:
        x,P, SE = kalman(station, knmi[1],m, i)
        new_station = copy.copy(station)
        new_station['measurements'] = pd.Series(x, index=station['measurements'].index)
       # diff = (new_station['measurements'].subtract(station['measurements']))**2
        new_resampled_outlier.append(new_station)
        i = i+1
        mse.append(np.nanmean(SE))
    
    return new_resampled_outlier,mse
    
def create_knmi():
    from read_knmi_obs import df
    dates = pd.date_range('2016-04-01 00:00:00', periods = len(df.value), freq='10Min')
    d = pd.DatetimeIndex(dates)
    series = df.value.tolist()
    temp_series = pd.Series(series,index=dates)
    dates = pd.date_range('2016-03-31 23:30:00', '2016-05-01 05:00:00', freq = '10Min')
    d = pd.DatetimeIndex(dates)
    res_re_temp = temp_series.reindex(d, method = 'nearest')
    t = res_re_temp.tolist()
    time = [dt_i.strftime('%Y-%m-%d %H:%M:%S') for dt_i in res_re_temp.index]
    dates = [datetime.datetime.strptime(t,'%Y-%m-%d %H:%M:%S') for t in time]

    return (dates,t), res_re_temp

def plot_data(thermo_data, name):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    data = [d['measurements']['2016-04'].tolist() for d in thermo_data]
    import numpy as np
    array_data = np.asarray(data)
    corr_data = []
    array_data = array_data.round(1)
    for station in thermo_data:
        c = station['measurements'].corr(knmi_timeseries)
        corr_data.append(c)
    corr_data = np.nan_to_num(corr_data)
    ax.plot(knmi_timeseries['2016-04'].tolist(), array_data.T, '.', ms=3, color='seagreen')
    min_ax = max(ax.get_xlim()[1], ax.get_ylim()[1])-3
    fig.set_size_inches(5,5)
    ax.plot([0, min_ax], [0, min_ax], ls="--", color='k', lw = 2.0, label='perfect correlation' )
    ax.set_xlim(0,min_ax)
    ax.set_ylim(0, min_ax)
    cor = np.nanmean(corr_data)
    ax.text(min_ax/1.4, 14, r'$r = {0:.3f}$'.format(cor), fontsize=15, color='darkred')
    plt.legend(loc =4)
    plt.ylabel('Measurements CWS')
    plt.xlabel('Measurement KNMI')
    plt.title('CWS vs KNMI measurements {}'.format(name))
    plt.savefig('figures/All temp/cloud {}.png'.format(name), dpi=300)
    plt.close()
    return corr_data
      
    
def fourier_transform(new_data):
    import scipy.signal
    for d in new_data:
        f, pxx = scipy.signal.welch(d['measurements']['2016-04'].tolist(), fs=144, nperseg=2048)
        plt.plot(f, pxx)
        
    f, pxx = scipy.signal.welch(knmi_timeseries['2016-04'].tolist(), fs=144, nperseg=2048)
    plt.plot(f, pxx, 'red', lw=2, label='KNMI data')
    plt.xlim(0,3.5)
    plt.xlabel('Frequency (per day)')   
    plt.ylabel('Power spectrum density')
    plt.savefig('figures/All temp/fouriertransform.png', dpi=600)
    plt.show()
    
def semivariance_as_function_of_distance(data, resolution, time, Ydist, max_dist):
    import itertools as it
    var_of_difference = [abs(x['measurements'][time] - y['measurements'][time])**2 for x,y in it.combinations(data,2) ]
    M = []
    r = int(resolution > 1)
    for i in linspace(0, max_dist, np.ceil(max_dist)*resolution):
       # print(np.round(i,1))
        M.append(0.5*np.nanmean([var_of_difference[j] for j in range(len(Ydist)) if np.round(Ydist[j],r) == np.round(i,r)]))
        
#    plt.plot(M, '.')
#    plt.title('sample semivariance, res = {} km'.format(1/resolution))
#    plt.xlabel('Lag in {} km'.format(1/resolution))
#    plt.ylabel('semivariance {}'.format(time))
#    plt.savefig('figures/Spatial analysis/semivariance as a function of distance res = {} km on {}.png'.format(1/resolution,'2016-04-12 noon'),dpi=600)
    return var_of_difference, M
    
    
def corr_as_function_of_distance(data, name, resolution, time, Ydist, max_dist):
    import itertools as it
    correlation_with_stations = [x['measurements'][time].corr(y['measurements'][time]) for x,y in it.combinations(data,2) ]
    M = []
    r = int(resolution > 1)
    for i in linspace(0, max_dist, np.ceil(max_dist)*resolution):
       # print(np.round(i,1))
        M.append(np.nanmean([correlation_with_stations[j] for j in range(len(Ydist)) if np.round(Ydist[j],r) == np.round(i,r)]))
        
    plt.plot(M)
    plt.title('correlation between stations as a function of distance, res = {} km'.format(1/resolution))
    plt.xlabel('Distance in {} km'.format(1/resolution))
    plt.ylabel('average correlation coeff on {}'.format(time))
    plt.savefig('figures/Spatial analysis/corr as a function of distance res = {} km on {} {}.png'.format(1/resolution,time, name),dpi=600)
    return M
        
def gamma_spherical(c_null, c, r, h):
    if h < r:
        return c_null + c*( 1.5*h/r - 0.5*(h/r)**3.0 )
    else:
        return c_null + c

def gamma_exponential(c_null, c, a, h):
    return c_null + c*(1-np.exp(-(h/a))) 
    
def gamma_gaussian(c_null, c, a, h):
    return c_null + c*(1-np.exp(-(h**2/a**2))) 
 
    
def fit_gamma(Mn, Ydist, res, model='Spherical'):
    hvals = linspace(0, max_dist, np.ceil(max_dist)*res)
    if model == 'Exponential' or model == 'Gaussian':
         c_null = 2#np.nanvar(np.asarray([d['measurements'] for d in data]))
    else :
         c_null =  (1.2-np.nanmean(M)) *np.nanstd(np.asarray([d['measurements'] for d in data]))
    c = np.nanmean(Mn)        
    r = max_dist
    gamma_h = []
    i = 0
    for h in hvals:
        if model == 'Exponential':
            gamma_h.append(gamma_exponential(c_null, c, r, h))
        elif model == 'Gaussian':
            gamma_h.append(gamma_gaussian(c_null, c, r, h))
        else:
            gamma_h.append(gamma_spherical(c_null, c, r, h))
            model = 'Spherical'
        i = i+1
    fig, ax = plt.subplots()
    ax.plot(gamma_h, 'r',lw=2, label = '{} model'.format(model))
    ax.plot(Mn, '.',label= 'Experimental variogram')
    ax.set_xticks((50, 100, 150, 200, 250, 300, 350, 400, 450))
    ax.set_xticklabels(('5','10', '15', '20', '25', '30', '35', '40','45'))
    plt.xlabel('Lag [km]')
    plt.ylabel('Semivariance')
    plt.title('Experimental variogram and fitted model')
    plt.legend(loc=0)
    plt.savefig('figures/Spatial analysis/variogram with {} model fitted, res = {}.png'.format(model, 1/res), dpi=600)
    plt.show()
    return gamma_h
    
def gamma(h, Mn):
    c_null = (1-np.nanmean(M)) * np.nanvar(np.asarray([d['measurements'] for d in data]))
    c = np.nanmean(Mn)
    r = max_dist
    return gamma_gaussian(c_null, c, r, h)


def SVh( P, h, bw , pd):
    '''
    Experimental semivariogram
    '''

    N = pd.shape[0]
    Z = list()
    for i in range(N):
        for j in range(i+1,N):
            if( pd[i,j] >= h-bw )and( pd[i,j] <= h+bw ):
                Z.append( ( P[i,2] - P[j,2] )**2.0 )
    return np.nansum( Z ) / ( 2.0 * len( Z ) )
 
def SV( P, hs, bw ):
    '''
    Experimental variogram
    '''
    import scipy.spatial
    sv = list()
    pd = sp.spatial.distance.squareform(sp.spatial.distance.pdist(P[:,:2], lambda u,v: gp.distance(u,v).meters))
    for h in hs:
        sv.append( SVh( P, h, bw, pd ) )
    sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] >= 0 ]
    return np.array( sv ).T
 
def C( P, h, bw ):
    '''
    calc uncorrelated var
    '''
    c0 = np.nanvar( P[:,2] )
    if h == 0:
        return c0
    return c0 - SVh( P, h, bw )
  

def spherical( h, a, C0 ):
    '''
    Spherical model of the semivariogram
    '''
    if type(h) == np.float64:
        if h <= a:
            return C0*( 1.5*h/a - 0.5*(h/a)**3.0 )
        else:
            return C0
    else:
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        return list( map( spherical, h, a, C0 ))
        
def gaussian(h, a, C0):
    '''
    Gaussian model of the semivariogram
    '''
    if type(h) == np.float64:
        if h <= a:
            return C0*(1-np.exp(-(h**2/a**2))) 
        else:
            return C0
    else:
        a = np.ones(h.size) * a
        C0 = np.ones(h.size) * C0
        return list(map(gaussian, h, a, C0))

def exponential(h, a, C0):
    if type(h) == np.float64:
        if h <= a:
            return C0*(1-np.exp(-(h/a))) 
        else:
            return C0
    else:
        a = np.ones(h.size) * a
        C0 = np.ones(h.size) * C0
        return list(map(gaussian, h, a, C0))
    
        
def cvmodel( P, model, hs, bw ):
    '''
    P           = ndarray, data
    model       =  modeling function
                      - spherical
                      - exponential
                      - gaussian
    hs          = lags
    bw          = bin width
    Output: (covfct) function modeling the covariance
    '''


    # calculate the sill
    C0 = C( P, hs[0], bw )

    # calculate the optimal parameters
    #param = opt( model, sv[0], sv[1], C0 )
    #print(C0, param)
    # return a covariance function
    covfct = lambda h, a=43000: C0 - model( h, a, C0 )
    return covfct  
    
def krige( P, hs, bw, u, N, covfct, mu):
    '''
    P           = data
    hs          = lags
    bw          = bin width
    u           = location to estimate
    covfct      = cov model
    mu          = mean
    
    '''
    import scipy.spatial
    d = np.array([gp.distance([p[0], p[1]], u).meters for p in P])
    P = np.vstack(( P.T, d )).T
    P = P[d.argsort()[:N]]
    k = covfct( P[:,3] )
    k = np.matrix( k ).T
    K = sp.spatial.distance.squareform(sp.spatial.distance.pdist(P[:,:2], lambda u,v: gp.distance(u,v).meters))
    K = covfct( K.ravel() )
    K = np.array( K )
    K = K.reshape(N,N)
    K = np.matrix( K )
    n = [random.random() for k in range(N)]
    K = K+(n*np.identity(N))*0.01
    K = K.round(8)
    while np.linalg.det(K) == 0:
        n = [random.random() for k in range(N)]
        K = K+(n*np.identity(N))*0.1
    weights = np.linalg.inv( K ) * k
    weights = np.array( weights )
    residuals = P[:,2] - mu
    estimation = np.dot( weights.T, residuals ) + mu
    error = abs(np.nanmean(weights*(estimation-P[:,2])**2))
    return float( estimation ), error
    
def kriging_fun(sliced, hs, bw, covfct, name, mu):
    '''
    sliced      = locations and values at timestep
    hs          = lags
    bw          = bin width
    covfct      = covariance model
    name        = name to save images with
    save        = wether to save images
    mu          = mean 
    '''
    P = np.array(sliced)
 
    
    if PLOT:
        sv = SV( P, hs, bw )  
        fig, ax = plt.subplots()
        ax.plot( sv[0], sv[1], '.' )
        ax.plot( sv[0], covfct(sv[0][0])+abs(covfct(sv[0][0])-covfct(sv[0])), 'r' ) 
        plt.title('Semivariogram')
        plt.ylabel('Semivariance')
        plt.xlabel('Lag [m]')
        plt.savefig('figures/Spatial analysis/semivariogram_{}.png'.format(name),fmt='png',dpi=300)
        plt.close()
    Z = []
    err = []
   

    for i in range(len(P)):
        p = P[i,:]
        Pn = np.delete(P, (i), axis=0)
        idx = np.isnan(Pn[:,2])
        Pn = Pn[~idx]
        estimate, error = krige(Pn, hs, bw, (p[0], p[1]), 10, covfct, mu)
        if np.isnan(p[2]):
            Z.append((p[0], p[1], estimate, error, np.nan))
        else:
            Z.append((p[0], p[1], estimate, error, abs(estimate-p[2])/np.sqrt(error)))
        err.append((estimate-p[2])**2)

    mse = np.nanmean(err)
    Z = pd.DataFrame(Z, columns = ['x', 'y', 'est', 'error', 'conf' ] )
    if PLOT:
        fig, ax = plt.subplots()
        plt.scatter(Z.x, Z.y, c=Z.conf, s=25,lw=0.1, cmap='Blues')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('Kriging estimates compared to actual measurements')
        plt.colorbar(label='abs(measurement - estimate) / sqrt(error var)')
        plt.savefig('figures/Spatial analysis/confidence_estimates_{}.png'.format(name), dpi=300)
        plt.close()
   
    return Z, mse, err
    
def spatial(data):
    dates = pd.date_range('2016-04-01 00:00:00', '2016-04-30 23:50:00', freq = '10Min')
    new_data = [{'latitude':d['latitude'], 'longitude':d['longitude'], 'name':d['name'], 'measurements': d['measurements']} for d in data]
    Zs = []
    mse_time = []
    mse_station = []
    conf = []
    ##### binwidth, plus or minus 500 meters
    bw = 500
    ###### lags in 1000 meter increments from zero to 43,000
    hs = np.arange(0,43000,bw)
    
    ##### to improve runtime.. trying to calc average covfct
    average = plot_points_average(new_data, 'Average measurements')
    covfct = cvmodel( np.array(average).T, gaussian, hs, bw )  
    
  
    m = get_mean_faster(data)
    mean = pd.Series(m[1], index = data[0]['measurements'].index) 
    
    for date in dates:
        start_time = time.time()
        dstr = datetime.datetime.strftime(date, '%Y-%m-%d %H:%M:%S')
        print(dstr)
        sliced = [(d['longitude'], d['latitude'], d['measurements'][dstr]) for d in data]

        mu = mean[dstr]
        Z, mse, err = kriging_fun(sliced, hs, bw, covfct, 'day {} hour {}_00'.format(date.day, date.hour), mu)
        mse_time.append(mse)
        mse_station.append(err)
        conf.append(Z.conf)
        for i in range(len(data)):
            if Z.conf[i] > 6 or np.isnan(Z.conf[i]):
                new_data[i]['measurements'][dstr] = Z.est[i]
        print("--- %s seconds ---" % (time.time() - start_time))
#    for s in range(len(new_data)):
#        series = pd.Series(Z[s,:], index = dates)
#        new_data[s]['measurements'] = new_data[s]['measurements'].append(series)
    return mse_time, mse_station, conf, new_data

def run_analysis():        
    raw =create_thermo_module_dataset()
    raw, nonnan = resample(raw)
    outlier, corr, z_scores = remove_outliers(raw, get_mean_faster(raw))
    filtered, mse =   kalman_f(outlier)
    pc, exp_var = pca(filtered, 'bla')
    var = np.nanmean(exp_var, axis=1)
    idx = var < 0.76
    idx2 = np.array(mse) > 1
    data = [pc[i] for i in range(len(idx)) if idx[i]==False and idx2[i]==False]
    mse_time, mse_station, conf, Z = spatial(data)
    
    plt.figure()         
    plt.hist(Ydist, 430, lw=0.1, facecolor='green')
    plt.title('Number of pairs of stations at a certain lag')
    plt.xlabel('Lag [km]')
    plt.ylabel('Number of pairs of stations')
    plt.savefig('figures/Quality measures/hist.png', dpi=600)
    
    plt.figure()         
    plt.hist(nonnan, 50, facecolor='green')
    plt.title('Histogram of valid measurements in stations')
    plt.xlabel('Percentage of valid measurements')
    plt.ylabel('Number of stations')
    plt.savefig('figures/Quality measures/nonnan.png', dpi=600)
      
    plt.figure()         
    plt.hist(np.array(corr)[~np.isnan(corr)], 50, facecolor='green')
    plt.title('Histogram of average correlation coefficients in stations')
    plt.xlabel('Correlation coefficient with KNMI')
    plt.ylabel('Number of stations')
    plt.savefig('figures/Quality measures/corr.png', dpi=600)      
    
    plt.figure()         
    plt.hist(z_scores, 50, facecolor='green')
    plt.title('Histogram of average zscore of measurements in stations')
    plt.xlabel('Average zscore')
    plt.ylabel('Number of stations')
    plt.savefig('figures/Quality measures/zscores.png', dpi=600)      
    
    plt.figure()         
    plt.hist(mse, 50, facecolor='green')
    plt.title('Histogram of MSE Kalman filter in stations')
    plt.xlabel('Mean squared error')
    plt.ylabel('Number of stations')
    plt.savefig('figures/Quality measures/mse_kalman.png', dpi=600)      
    
    plt.figure()         
    plt.hist(np.nanmean(exp_var, axis=1), 50, facecolor='green')
    plt.title('Histogram of PCA variance explained in stations')
    plt.xlabel('Percentage of variance explained')
    plt.ylabel('Number of stations')
    plt.savefig('figures/Quality measures/pca.png', dpi=600)      
    
    plt.figure()         
    plt.hist(np.nanmean(mse_station, axis=0), 50, facecolor='green')
    plt.title('Histogram MSE Kriging in stations')
    plt.xlabel('Mean square error')
    plt.ylabel('Number of stations')
    plt.savefig('figures/Quality measures/kriging_mse.png', dpi=600)   
    
    c = np.nanmean(conf, axis=0)
    plt.figure()         
    plt.hist(c, 50, facecolor='green')
    plt.title('Histogram confidence score Kriging in stations')
    plt.xlabel('Confidence score')
    plt.ylabel('Number of stations')
    plt.savefig('figures/Quality measures/kriging_conf.png', dpi=600)   
         
       
    days = mdates.HourLocator()
    daysFMT = mdates.DateFormatter('%H') 
    fig,ax= plt.subplots() 
    ax.plot(knmi_timeseries[td].index, np.nanmean(summed, axis=0), 'g')
    plt.xlabel('Hour of day')
    plt.ylabel('Mean squared error')
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(daysFMT)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.title('MSE Kriging during the day')
    plt.savefig('figures/Quality measures/kriging_day.png', dpi=600)   
    
    
    
    plot_thermo_mods(raw, False, get_mean_faster(raw), 'Raw data')
    cor_raw = plot_data(raw, 'Raw data')

    plot_thermo_mods(outlier, False, get_mean_faster(outlier), 'Preprocessed data')
    cor_outl = plot_data(outlier, 'Preprocessed data')
    idx = np.array(corr) > 0.9241
    corrbest =  [raw[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(corrbest, False, get_mean_faster(raw), 'Best {} corr stations r 0.9241 cutoff'.format(len(corrbest)))

    idx = np.array(corr) < 0.29
    corrworst =  [raw[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(corrworst, False, get_mean_faster(raw), 'Worst {} corr stations r 0.29 cutoff'.format(len(corrworst)))

    
    plot_thermo_mods(filtered, False, get_mean_faster(filtered), 'Kalman filtered data')
    cor_kal = plot_data(filtered, 'Kalman filtered data')
    idx = np.array(mse) < 0.07
    kalmanbest_12 = [outlier[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(kalmanbest_12, False, get_mean_faster(outlier), 'Best 10 stations MSE=0.07 cutoff')
    
    idx = np.array(mse) > 1
    kalman_12 = [outlier[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(kalman_12, False, get_mean_faster(outlier), 'Worst 10 stations MSE=1 cutoff')
    

    plot_thermo_mods(filtered, False, get_mean_faster(pc), 'PCA processed data')
    cor_pc = plot_data(pc, 'PCA processed data')
  
    
    idx = var < 0.7
    pcworst = [filtered[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(pcworst, False, get_mean_faster(filtered), 'Worst 10 stations per=0.72 cutoff')
    
    idx = var > 0.96
    pcbest = [filtered[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(pcbest, False, get_mean_faster(filtered), 'Best 10 stations per= 0.97 cutoff')
    
    plot_thermo_mods(data, False, get_mean_faster(data), 'Temporal processed data')
    corr_data = plot_data(data, 'Temporal processed data')
    
    plot_thermo_mods(new_data, False, get_mean_faster(new_data), 'Measurements after Kriging correction')
    corr_krig = plot_data(new_data, 'Kriging processed data')
    mse_per_station = np.nanmean(mse_station, axis=0) 
    
    idx = np.array(mse_per_station) >5.99132
    krigeworst = [data[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(krigeworst, False, get_mean_faster(filtered), 'Worst 10 Kriging mse stations mse 6 cutoff')
    
    idx = np.array(mse_per_station) < 0.6
    krigebest = [data[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(krigebest, False, get_mean_faster(filtered), 'Best 10 Kriging mse stations mse 0.6 cutoff')
    
       
    idx = np.array(c) > 6
    krigeworstc = [data[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(krigeworstc, False, get_mean_faster(filtered), 'Worst 10 Kriging conf stations conf 6 cutoff')
    
    idx = np.array(c) < 1.42
    krigebestc = [data[i] for i in range(len(idx)) if idx[i]==True]
    plot_thermo_mods(krigebestc, False, get_mean_faster(filtered), 'Best 10 Kriging conf stations conf  cutoff')
    
PLOT = False
knmi, knmi_timeseries = create_knmi()    
    


mean_data = get_mean_faster(data)
mean_dataseries = pd.Series(mean_data[1], index=data[0]['measurements'].index)
diff_knmi_post = mean_dataseries.subtract(knmi_timeseries)
summed = []

for j in range(1,31): 
    date = datetime.datetime(2016,11,j)
    td = datetime.datetime.strftime(date, '%Y%m%d')
    summed.append(diff_knmi_post[td])
mean_diff = np.nanmean(summed, axis=0)


days = mdates.HourLocator()
daysFMT = mdates.DateFormatter('%H') 
fig,ax= plt.subplots() 
ax.plot(data[0]['measurements'][td].index, mean_diff)
ax.set_ylim(-0.2,3)
plt.xlabel('Hour of day')
plt.ylabel('Difference in degrees Celsius')
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(daysFMT)
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
plt.title('Average difference during the day')
plt.savefig('figures/averagedifferenceknmiprocessed.png', dpi=600)   



#make googlemap plot
#import gmplot
#lats = [s['latitude'] for s in max_distances]
#longs = [s['longitude'] for s in max_distances]
#gmap = gmplot.GoogleMapPlotter(np.mean(lats), np.mean(longs), 12)
#gmap.scatter(lats, longs, color='b',size=200, marker=False)
#
#gmap.draw("maxdist.html")
#


##Spatial stuff

#corr_new = plot_data(new_resampled_outlier, 'new data')
#data = [new_resampled_outlier[i] for i in range(len(corr_new)) if corr_new[i] > 0.8]
points_locations = [[s['longitude'], s['latitude']] for s in data]
X = np.asarray(points_locations)
Ydist = sp.spatial.distance.pdist(X, lambda u,v: gp.distance(u,v).kilometers)
###max_dist = max(Ydist)
