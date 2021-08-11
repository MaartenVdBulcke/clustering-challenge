# challenge-clustering


# Description
  This was an assignment during our training at BeCode.  
  The goal was to get a feel for clustering in machine learning.
  For this we used a self-created dataframe with features extracted from a database from <a href="https://www.kaggle.com/isaienkov/bearing-classification" target="_blank">Kaggle</a>, on testing bearings.  
  Our job was to cluster the data so as to identify different types of failures of the bearings. 
  
  
# Installation
## Python version
* Python 3.9

## Databases
I made use of two databases that were created by fellow-students and/or myself in a previous project: 
* link to first
* link to second

## Packages used
* pandas
* numpy
* itertools
* matplotlib.pyplot
* seaborn
* plotly
* sklearn

# Usage
| File                        | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| main.py                   | File containing Python code. The whole proces a manipulating the dataframes, investigating which features to combine and how many clusters are ideal. Also model evaluation |
| utils/model.py              | File containing Python code, using ML - Random Forest.   <br>Fitting our data to the model and use to it make predictions. |
| utils/data_manipulation.py | File containing Python code.<br>Functions made for ease of use in a team enviroment. |
| utils/plotting.py           | File containing Python code.<br>Used for getting to know the data.<br>Made plots to find correlations between features. |
| csv_output                  | Folder containing some of the csv-files we used for our coding.<br>Not all of our outputted files are in here,   <br>since Github has a file limit of 100MB. |
| visuals                     | Folder containing plots we deemed interesting and helped us gain   <br>insights on the data. |

# Which features to combine for effective clustering
## two features
| Column name of feature | Change made                  | Reason                                                                                        |
|------------------------|------------------------------|-----------------------------------------------------------------------------------------------|
| timestamp              | Only keeping rows above 0,25 | We found some outliers where the "rpm" and "hz" values spiked in the first parts of the test.  <br>With the use of plotting, we discovered a cut off point. |

![](visuals/Exp_24_RPM_reading_error.png)

### How many clusters

## three features
### How many clusters


## six features
### How many clusters



# Extra visuals

## The bearings test
![bearing test machine, image to be pushed](visuals/bearing_test_machine_set_up.jpg)

## Plot showing the min-max-difference of every axis, on every bearing.

![](visuals/vibration_spread_differences_on_all_axes.png)

## Plot that gave us the idea to look into the first seconds.
![](visuals/control_vs_good_vs_bad_Y_Speed_Hz.png)

## Plot that showed possible clusters
Ready for future exploration
![](visuals/scatter_cluster_ready.png)

# Links 
I made use of dataframes created in two other projects: 

* <a href="https://github.com/ltadrummond/challenge-clustering" target="_blank"> project from collegues </a> that extracted a dataframe with 54 features extracted from the original bearing dataset
* <a href="https://github.com/Roldan87/challenge-classification" target="_blank"> project from our group </a> that extracted a dataframe with 6 features extracted from the original bearing dataset

# Contributor
| Name                  | Github                                 |
|-----------------------|----------------------------------------|
| Maarten Van den Bulcke           | https://github.com/MaartenVdBulcke       |


# Timeline
09/08/2021 - 11/08/2021