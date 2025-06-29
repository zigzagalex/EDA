# EDA
A function toolset for exploratory data analysis in python optimized for bigger datasets using polars. 

### Why? 
In every project, I have to do some EDA to understand the data more closely and decide on how to clean and transform it. I don't want to set it up every time from scratch. So to do this efficiently for data with up to 500 mio rows we use pandas and some parallel processing with joblib for plotting, to get 80% of the relevant analysis done fast. The other 20% are often domain specific anyways... (yes I'm pleading the pareto principle here) 

### How does this work?
There are three variables that have to be adjusted: 
1. ```parquet_path``` which is the path to the stored data
2. ```cat_cols``` a list of the categorical columns in the data
3. ```num_cols``` a list of the numerical columns in the data

Then run ```main.py``` and get:
* Size, count and null analysis of data
* Frequency analysis of categorical columns
* Simple statistical analysis of numerical columns (min, max, median, mean, std, var, skew, kurtosis, correlation matrix)
* Plots of numerical columns (box plot, histogram and qq plot for normal distribution) all saved in a pdf

### What could be improved?
1. New feature: EDA for text columns
2. New feature: EDA for image columns
3. Long tail counts for numerical and categorical variables e.g. 5th and 95th percentile counts of outliers. 
4. Join multiple parquet files
5. Add other data ingestion methods
6. Make a CLI tool out of this
7. Package it well

### Testing
A good public dataset to test it out and get a feel for the speed is this [kaggle flight delay](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022) dataset. It has approx 4 mio rows per parquet file, which is not much but with ~1.5gb it already starts posing difficulties for pandas. 