#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
car_details = pd.read_csv('C:/Users/dell/Desktop/capstone/CAR DETAILS.csv')

#Exploring data and Performing Data Cleaning and Pre-Processing
car_details.head() #Showing the first five rows
car_details.shape # Showing the shape of the Dataset
car_details.dtypes # Checking Datatypes of all column
car_details.info() # Check ratings info
car_details.duplicated().sum() # Check Duplicates
car_details.columns #show all columms
car_details.isnull().sum() # Check the presence of missing values
import datetime
date_time = datetime.datetime.now()
car_details['Age']=date_time.year - car_details['year']
car_details.drop('year',axis=1,inplace=True)
