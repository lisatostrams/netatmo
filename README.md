## Initialize

Add data in json format to Data folder.
Use the data.py and read_knmi_obs.py to load files.

## Run all analysis
in april.py or november.py, simply run the run_analysis() function to perform all analysis and plot all results.

Alternatively..

## Create datasets

use create_thermo_module_dataset() to create the dataset
use plot_thermo_mods() to plot temp measurements

knmi, knmi_timeseries = create_knmi()  


## Preprocessing
data = resample(data)
data, _ = remove_outliers(data, get_mean_faster(data))

## Kalman filter
data, _ = kalman_f(data)

## Principal Component Analysis
data,_ = pca(data, 'name')

## Kriging
_,_,_, data = spatial(data)
