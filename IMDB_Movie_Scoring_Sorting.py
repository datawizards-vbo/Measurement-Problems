# import the required libraries
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# read the movies metadata csv file
df = pd.read_csv("datasets/movies_metadata.csv", low_memory=False)

# select only the required columns
df = df[["title", "vote_average", "vote_count"]]

# display the first five rows of the dataframe
df.head()

# display the number of rows and columns in the dataframe
df.shape

# 1. Sort Movies by Vote Average

# sort the dataframe by vote_average in descending order and display the first 20 rows
df.sort_values("vote_average", ascending=False).head(20)

# display the descriptive statistics of the vote_count column
df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

# select only the rows where vote_count is greater than 400, sort the dataframe by vote_average in descending order and display the first 20 rows
df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)

# scale the vote_count column between 1 and 10 and add the result to a new column called vote_count_score
df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)).fit(df[["vote_count"]]).transform(df[["vote_count"]])

# 2. Sort Movies by (vote average) * (vote count)
# multiply vote_average and vote_count_score and add the result to a new column called average_count_score
df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

# sort the dataframe by average_count_score in descending order and display the first 20 rows
df.sort_values("average_count_score", ascending=False).head(20)

# 3. Sort Movies by IMDB Weighted Rating

# Calculate weighted rating for each movie and add it to dataframe
M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)

# Print top 10 movies by average count score and weighted rating
print(df.sort_values("average_count_score", ascending=False).head(10))
print(df.sort_values("weighted_rating", ascending=False).head(10))

# 4. Sort Movies by Bayesian Average Rating
# Define a function to calculate the Bayesian average rating score for a given set of ratings
def calculate_bayesian_average_rating(rating_counts, confidence_level=0.95):
    # If there are no ratings, return 0
    if sum(rating_counts) == 0:
        return 0
    
    # Calculate the expected value of the rating distribution
    num_scores = len(rating_counts)  # get the number of possible scores
    z = st.norm.ppf(1 - (1 - confidence_level) / 2)  # calculate the z-score for the given confidence level
    total_ratings = sum(rating_counts)  # calculate the total number of ratings
    expected_value = 0.0  # initialize the expected value of the rating distribution
    expected_value_squared = 0.0  # initialize the squared expected value of the rating distribution
    for score, count in enumerate(rating_counts):
        probability = (count + 1) / (total_ratings + num_scores)  # calculate the probability of a given score
        expected_value += (score + 1) * probability  # update the expected value based on the score and its probability
        expected_value_squared += (score + 1) ** 2 * probability  # update the squared expected value based on the score and its probability
    
    # Calculate the variance of the rating distribution
    variance = (expected_value_squared - expected_value ** 2) / (total_ratings + num_scores + 1)
    
    # Calculate the Bayesian average score
    bayesian_average = expected_value - z * math.sqrt(variance)
    return bayesian_average  # return the Bayesian average score

# Apply the bayesian_average_rating function to each row of the dataframe using lambda function and create a new column "bar_score"
df["bar_score"] = df.apply(lambda x: calculate_bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)

# Sort the dataframe by "bar_score" in descending order and display the top 20 rows
df.sort_values("bar_score", ascending=False).head(20)
