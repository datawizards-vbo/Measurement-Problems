# Sorting Products

# Import necessary libraries
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# Set display options for pandas dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Load the dataset into a pandas dataframe
df = pd.read_csv(r"Measurement_Problems\datasets\course_reviews.csv")

# Print the first 5 rows of the dataframe to check if it loaded correctly
df.head()

# Print the shape of the dataframe to get the number of rows and columns
df.shape

# Main Goal: Sorting Courses

# Method 1.
# SORTING BY RATING

# Sort the dataset by rating in descending order and display the top 20 products
df.sort_values("rating", ascending=False).head(20)


# Method 2.
# SORTING BY COMMENT COUNT OR PURCHASE COUNT

# Sort the dataset by purchase_count in descending order and display the top 20 products
df.sort_values("purchase_count", ascending=False).head(20)

# Sort the dataset by commment_count in descending order and display the top 20 products
df.sort_values("commment_count", ascending=False).head(20)


# Method 3.
# SORTIN BY RATING, COMMENT AND PURCHASE

# Scale the purchase_count values to a range of 1 to 5
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

# Scale the commment_count values to a range of 1 to 5
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

# Calculate the weighted sorting score for each product using the scaled values and the specified weights
df["weighted_sorting_score"] = (df["comment_count_scaled"] * 32 / 100 +
                                df["purchase_count_scaled"] * 26 / 100 +
                                df["rating"] * 42 / 100)

# Sort the dataset by the weighted sorting score in descending order and display the top 20 products
df.sort_values("weighted_sorting_score", ascending=False).head(20)

# Filter the dataset to include only products that contain "Veri Bilimi" in their course_name
# Sort the filtered dataset by the weighted sorting score in descending order and display the top 20 products
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)

# Method 4.
# SORTING BY BAYESIAN AVERAGE RATING SCORE

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


# Calculate the Bayesian average rating score for each product using their respective ratings
df["bar_score"] = df.apply(lambda x: calculate_bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

# Sorting the DataFrame by "weighted_sorting_score" column in descending order, and selecting the top 20 rows
df.sort_values("weighted_sorting_score", ascending=False).head(20)

# Sorting the DataFrame by "bar_score" column in descending order, and selecting the top 20 rows
df.sort_values("bar_score", ascending=False).head(20)

# Selecting rows from the DataFrame where the "course_name" column index is either 5 or 1, and then sorting by "bar_score" column in descending order
df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)

# Method 5.
# HYBRID SORTING

# Defining a function to calculate a weighted sorting score for each row in the DataFrame
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    # Calculating the weighted sorting score for each row by combining three factors:
    # the scaled comment count, the scaled purchase count, and the rating.
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)

# Defining a function to calculate the hybrid sorting score for each row in the DataFrame.
def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    # Calculating the Bayesian average rating score for each row using the "bayesian_average_rating" function.
    bar_score = dataframe.apply(lambda x: calculate_bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    # Calculating the weighted sorting score for each row using the "weighted_sorting_score" function.
    wss_score = weighted_sorting_score(dataframe)

    # Calculating the hybrid sorting score by combining the Bayesian average rating score and the weighted sorting score
    # with the given weight coefficients.
    return bar_score*bar_w/100 + wss_score*wss_w/100

# Adding a new column "hybrid_sorting_score" to the DataFrame by calling the "hybrid_sorting_score" function.
df["hybrid_sorting_score"] = hybrid_sorting_score(df)

# Sorting the DataFrame by "hybrid_sorting_score" column in descending order, 
# and selecting the top 20 rows
df.sort_values("hybrid_sorting_score", ascending=False).head(20)

# Selecting rows from the DataFrame where the "course_name" column contains the string "Veri Bilimi",
# and then sorting those rows by "hybrid_sorting_score" column in descending order,
# and selecting the top 20 rows.
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)
