# Rating Products

# Import necessary libraries
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# Set display options for pandas dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Load the dataset into a pandas dataframe
df = pd.read_csv(r"Measurement_Problems\datasets\course_reviews.csv")

# Print the first 5 rows of the dataframe to check if it loaded correctly
df.head()

# Print the shape of the dataframe to get the number of rows and columns
df.shape

# Count the number of occurrences of each value in the "Rating" column
df["Rating"].value_counts()

# Count the number of occurrences of each value in the "Questions Asked" column
df["Questions Asked"].value_counts()

# Group the dataframe by the "Questions Asked" column and aggregate the count of values and the mean of "Rating"
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})

# Main Goal: 
# User and Time Weighted Course Score Calculation

# Method 1. 
# AVERAGE

# Find the mean rating of the dataframe column 'Rating'
df["Rating"].mean() # 4.764284061993986

# Method 2.
# TIME-BASED WEIGHTED AVERAGE

# Calculate the weighted average rating of the dataframe column 'Rating' based on the time of the rating
# Convert the 'Timestamp' column to datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Set a current date as a reference point
current_date = pd.to_datetime('2021-02-10 0:0:0')

# Create a new column 'days' by calculating the number of days between the 'current_date' and the 'Timestamp' column
df["days"] = (current_date - df["Timestamp"]).dt.days

# Calculate the mean of rating for each time period
df.loc[df["days"] <= 30, "Rating"].mean()
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()
df.loc[(df["days"] > 180), "Rating"].mean()

# Calculate the time-based weighted average rating using the mean ratings and corresponding weights
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100
# 4.765025682267194

# Define a function to calculate the time-based weighted average rating with customizable weights
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

# Calculate the time-based weighted average rating using the defined function
time_based_weighted_average(df)

# Calculate the time-based weighted average rating with customized weights using the defined function
time_based_weighted_average(df, 30, 26, 22, 22)

# Method 3.
# USER-BASED WEIGHTED AVERAGE

# Calculate the mean rating for each progress level using groupby
df.groupby("Progress").agg({"Rating": "mean"})

# Calculate the user-based weighted average rating
def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100

# Call the user_based_weighted_average function with custom weights
user_based_weighted_average(df, 20, 24, 26, 30)
# 4.803286469062915


# Method 4.
# WEIGHTED RATING

# Calculate the time-based weighted average rating
def time_based_weighted_average(dataframe, w1=30, w2=35, w3=35):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 60), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 60), "Rating"].mean() * w3 / 100

# Calculate the course weighted rating
def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

# Call the course_weighted_rating function with default weights
course_weighted_rating(df) # 4.782329008765954

# Call the course_weighted_rating function with custom weights
course_weighted_rating(df, time_w=40, user_w=60) # 4.785914747947271
