# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 12:52:02 2016

@author: Lima
"""

import datetime
import numpy as np   
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata

def get_timestamps_temps(data_i):
    datesdata = data_i['thermo_module']['valid_datetime']
    tempdata = data_i['thermo_module']['temperature']
    thermo_mod = []
    i = 0
    for date in datesdata:   
        thermo_mod.append(((datetime.datetime.fromtimestamp(date['$date']/1000)-datetime.timedelta(minutes=120)).strftime('%Y-%m-%d %H:%M:%S'), tempdata[i]))
        i+=1
    thermo_mod = sorted(thermo_mod, key=lambda date: date[0])  
    return ([t[0] for t in thermo_mod], [t[1] for t in thermo_mod])

def make_time_plottable(thermo_mod):
    import matplotlib as mpl
    times = []
    for t in thermo_mod[0]:
        times.append(datetime.datetime.strptime(t,'%Y-%m-%d %H:%M:%S'))
    return mpl.dates.date2num(times)    

def create_thermo_module_dataset():
    from data import data    
    Thermo_Module = []
    for i in range(0,len(data)):
        Thermo_Module.append(({'name': data[i]['_id'],'longitude': data[i]['longitude'], 'latitude':data[i]['latitude']}, 
        get_timestamps_temps(data[i])))
    return Thermo_Module
    
def plot_thermo_mods(ther_data, plot_hours, m):
    import matplotlib.dates as mdates
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    daysFMT = mdates.DateFormatter('%d') 
    hoursFMT = mdates.DateFormatter('%H')
    fig,ax= plt.subplots() 
    for i in range(0, len(ther_data)):
        curr = ther_data[i][1]
        time = make_time_plottable(curr)
        if len(curr[1]) == len(time): 
            ax.plot_date(time, curr[1], '-.')
    ax.plot_date(m[0],m[1], '-',color = 'k', linewidth = 3.0)
    if plot_hours:
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hoursFMT)
    else:
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(daysFMT)
        datemin = datetime.date(2016, 4, 1)
        datemax = datetime.date(2016, 5, 1)
        ax.set_xlim(datemin, datemax)
    plt.grid() 
    plt.show()

def chop_into_days(thermo_dataset):
    days = []
    for d in range(1,31):
        stations_today = []
        for i in range(0,len(thermo_dataset)):
            current_station = thermo_dataset[i]
            name_location = current_station[0]
            temps_times = current_station[1]        
            today_ti = []
            today_tm = []
            for i in range(0, len(temps_times[0])): 
                if datetime.datetime.strptime(temps_times[0][i],'%Y-%m-%d %H:%M:%S').day == d and datetime.datetime.strptime(temps_times[0][i], '%Y-%m-%d %H:%M:%S').year == 2016:
                    today_ti.append(temps_times[0][i])
                    today_tm.append(temps_times[1][i])
            stations_today.append((name_location, (today_ti, today_tm)))
        days.append(stations_today)
        print(d)
    return days
            
def resample(thermo_dataset):
    new_thermo_dataset = []
    import pandas as pd    
    for i in range(0,len(thermo_dataset)):
        print(i)
        current_station = thermo_dataset[i]
        temp_df = current_station[1]
        if(len(temp_df[1]) > 0):
            series = temp_df[1]
            not_nan_pct = np.sum(~np.isnan(np.asarray(series)))/len(series)
            print(not_nan_pct)
            if not_nan_pct > 0:
                index = [datetime.datetime.strptime(t,'%Y-%m-%d %H:%M:%S') for t in temp_df[0]]
                temp_series = pd.Series(series,index=index)
                res_temp= temp_series.resample('10T').mean().bfill()
                dates = pd.date_range('2016-03-31 23:30:00', '2016-05-01 05:00:00', freq = '10Min')
                d = pd.DatetimeIndex(dates)
                res_re_temp = res_temp.reindex(d, method='nearest')
                t = res_re_temp.tolist()
                time = [dt_i.strftime('%Y-%m-%d %H:%M:%S') for dt_i in res_re_temp.index]
    #            df = pd.DataFrame(t, index = res_temp.index)
    #            dates = pd.date_range('2016-03-31 23:30:00', '2016-05-01 05:00:00', freq = '10Min')
    #            df.reindex(dates)
    #            temp = df.values.T.tolist()
                new_temp_df = (time,t)
                new_station = (current_station[0], new_temp_df)            
                new_thermo_dataset.append(new_station)
            
    return new_thermo_dataset
    
def plot_points_slice(day, time):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    temp_slice = []
    for i in range(0, len(day)):
        station = day[i]
        times = station[1][0]
        if time in times:        
            i = times.index(time)
            temperatures = station[1][1]
            temperature = temperatures[i]
            temp_slice.append(temperature)
        else:
            temp_slice.append(float('NaN'))
    plt.scatter([s[0]['longitude'] for s in day], 
             [s[0]['latitude'] for s in day], 
            c = temp_slice, 
            s=50,
            cmap = 'rainbow')   
    plt.colorbar()
    plt.show() 
    return ([s[0]['longitude'] for s in day], [s[0]['latitude'] for s in day], temp_slice)

def get_mean_faster(thermo_data):
    import pandas as pd
    dates = pd.date_range('2016-03-31 23:30:00', '2016-05-01 05:00:00', freq = '10Min')
    temp_data = [d[1][1] for d in thermo_data]
    array_data = np.asarray(temp_data)
    temp_means = np.nanmean(array_data, axis=0)
    times_list = [date.strftime('%Y-%m-%d %H:%M:%S') for date in dates]
    temp_std = np.nanstd(array_data, axis=0)
    print(len(temp_means))
    return (times_list, temp_means, temp_std)
    
def get_mean(thermo_data):
    import pandas as pd   
    import numpy as np
    dates = pd.date_range('2016-03-31 23:30:00', '2016-05-01 05:00:00', freq = '10Min')
    temp_means = []
    times_list = []
    temp_std = []
    for timestamp in dates:
        temp_slice = []
        time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        times_list.append(time)
        count = 0
       # print(timestamp)
        for i in range(0,len(thermo_data)):
            station = thermo_data[i][1]
            times = station[0]
            temps = station[1]
            if time in times:
                count = count+1
                i = times.index(time)
                temp_slice.append(temps[i])
        if len(temp_slice) > 0:
            temp_means.append(np.nanmean(temp_slice))
            temp_std.append(np.nanstd(temp_slice))
        else:
            temp_means.append(np.nan)
            temp_std.append(np.nan)
    print(len(times_list))
    print(len(temp_means))
    return (times_list, temp_means, temp_std)                
            
def remove_outliers(thermo_data, m):
    t = m[0]
    means = m[1]
    stds = m[2] 
    new_data = []
    import numpy as np
    for i in range(0, len(thermo_data)):
        station = thermo_data[i][1]
        times = station[0]
        temps = station[1]
        new_temps = []
        for j in range(0, len(t)):        
            if t[j] in times:
                c = times.index(t[j])
                if temps[c] > means[j]+(3*stds[j]) or temps[c] < means[j]-(3*stds[j]):
                    new_temps.append(np.nan)
                else:
                    new_temps.append(temps[c])
            else:
                new_temps.append(np.nan)
        new_station = (thermo_data[i][0], (times, new_temps))
        new_data.append(new_station)
    return new_data
    
def plot_thermo_mods_box(ther_data, plot_hours):
    import matplotlib.dates as mdates
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    daysFMT = mdates.DateFormatter('%d') 
    hoursFMT = mdates.DateFormatter('%H')
    fig,ax= plt.subplots() 
    ax.boxplot([curr[1][1] for curr in ther_data])
    m = get_mean(ther_data)
    ax.plot(m[0],m[1], '-',color = 'k', linewidth = 3.0)
    if plot_hours:
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hoursFMT)
    else:
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(daysFMT)
        datemin = datetime.date(2016, 4, 1)
        datemax = datetime.date(2016, 5, 1)
        ax.set_xlim(datemin, datemax)
    plt.grid() 
    plt.show()

#from pylab import *
def pca(data):
    import matplotlib as mpl
    import scipy.linalg as la
    temp_data = [d[1][1] for d in data]
    X = np.mat(np.asarray(temp_data))
    N = len(X)
    m = get_mean_faster(data)
    Y = X - np.ones((N,1))*np.mat(m[1])
    where_nans = np.isnan(Y)
    

    
    C = np.cov(Y)
    C = np.nan_to_num(C)
    S,V = la.eig(C)
    rho = abs((S*S)/(S*S).sum())
    import matplotlib.pyplot as plt
    sum_rho = np.cumsum(rho)
    plt.plot(rho)
    plt.plot(sum_rho)
    plt.close()
    V = V.T
    Y[where_nans] = 1
    #nrs = 'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentyfour twentyfive'
    #num = nrs.split()
    num = [1,2,5,10,25,35,50,70,80,90,100]
    times = []
    for t in m[0]:
        times.append(datetime.datetime.strptime(t,'%Y-%m-%d %H:%M:%S'))
    times =  mpl.dates.date2num(times)   
    import matplotlib.dates as mdates
    days = mdates.DayLocator()
    daysFMT = mdates.DateFormatter('%d') 

    for i in num:
        Z = V[:,0:i].T * Y
        W = np.ones((N,1))* np.mat(m[1]) + (np.mat(Z).T * np.mat(V[:,0:i]).T).T
        fig,ax= plt.subplots() 
        ax.plot(times,W.T, '-.')
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(daysFMT)
        datemin = datetime.date(2016, 4, 1)
        datemax = datetime.date(2016, 5, 1)
        ax.set_xlim(datemin, datemax)
        plt.ylabel('Temperature')
        plt.xlabel('Day in month')
        plt.grid()
        plt.title('Data (z<3) reconstructed from first {} PC ({:.4f}% var explained)'.format(i,100*abs(sum_rho[i-1])))
        plt.savefig('figures/PCA/PCA_first_{}_pc_outlier_removed_resampled_data.pdf'.format(i))  
        plt.close()
    return rho, W
    
def kalman(station, model, m):
    
#    x = []
#    A = 1
#    H = 1
#    Q = 0.1
#    K = 0
#    P = np.cov(model)
#    B = 0
#    u = 0
#    R = np.cov(station[1][1])
#    
    n_iter = len(station[1][1])
    sz = (n_iter,) # size of array
    x = model[0] # truth value (typo in example at top of p. 13 calls this z)
    z = station[1][1] # observations (normal about x, sigma=0.1)

    Q =  0.1#np.sqrt(np.var(model)) # process variance
    print(Q)
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R =  np.var(model) # estimate of measurement variance, change to see effect
    print(R)
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]

    import matplotlib as mpl
    import matplotlib.dates as mdates
    days = mdates.DayLocator()
    daysFMT = mdates.DateFormatter('%d') 
    times = []
    for t in m[0]:
        times.append(datetime.datetime.strptime(t,'%Y-%m-%d %H:%M:%S'))
    times =  mpl.dates.date2num(times)   
    fig,ax= plt.subplots()
    ax.plot(times,z,'k+',label='noisy measurements')
    ax.plot(times,xhat,'b-',label='a posteri estimate')
    ax.plot(times,model,color='g',label='model')
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(daysFMT)
    datemin = datetime.date(2016, 4, 1)
    datemax = datetime.date(2016, 5, 1)
    ax.set_xlim(datemin, datemax)
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Temperature')
    plt.savefig('figures/Kalman/kalman_model_mean_Q_{}_R_{}.pdf'.format(Q,R))

    return xhat
    
    
thermo_mod_dataset = create_thermo_module_dataset()
#plot_thermo_mods(thermo_mod_dataset, False)
resampled = resample(thermo_mod_dataset)
m = get_mean_faster(resampled)
#resampled_outlier = remove_outliers(resampled, m)
#m = []
#plot_thermo_mods(resampled_outlier, False, m)
#rho,W = pca(resampled_outlier)
station = resampled[1]
#plot_thermo_mods([station], False, m)
x = kalman(station, m[1],m)

