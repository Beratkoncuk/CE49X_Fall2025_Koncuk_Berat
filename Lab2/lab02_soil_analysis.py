# CE 49X - Lab 2: Soil Test Data Analysis

# Student Name: _____Berat Koncuk___________  
# Student ID: ___2021403234___  
# Date: ___11.10.2025___

import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the soil test dataset from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    # TODO: Implement data loading with error handling
    try: # if an error occurs python wont stop working
        df = pd.read_csv(file_path) # Reading CSV into a pandas DataFrame
        print("Data loaded successfully.") # Inform the user on success
        return df # Return the DataFrame to the caller
    except FileNotFoundError: # if the file cannot be found it gives a warning
        print(f"Error: File not found. Ensure the file exists at the specified path: {file_path}")
        return None # none returns and it stops the program
    except Exception as e: # for other errors
        print(f"Error loading data: {e}")
        return None   # safely exits the function without crashing the program
        # 'Exception' is the base class for most Python errors.
        
def clean_data(df):
    """
    Clean the dataset by handling missing values and removing outliers from 'soil_ph'.
    
    For each column in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
    - Missing values are filled with the column mean.
    
    Additionally, remove outliers in 'soil_ph' that are more than 3 standard deviations from the mean.
    
    Parameters:
        df (pd.DataFrame): The raw DataFrame.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy() # We want it to Work on a copy to avoid changing the original DataFrame and distrupt the og version
    
    # TODO: Fill missing values in each specified column with the column mean
    for col in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
        if df_cleaned[col].isnull().any(): # # Missing values in the column are marked as true/false by isnull function. Then the any function returns even if there is a single true value in it.
            mean_val = df_cleaned[col].mean() # Computes mean of column (ignores NaNs by default)
            df_cleaned[col].fillna(mean_val, inplace=True) # fillna function replaces NaNs with the mean (in-place)
            print(f"Filled missing values in '{col}' with mean value {mean_val:.2f}")
    
    # TODO: Remove outliers in 'soil_ph': values more than 3 standard deviations from the mean
    ph_mean = df_cleaned['soil_ph'].mean() # it computes the mean of soil ph from the copy we created 
    ph_std = df_cleaned['soil_ph'].std() # it computes the standard deviation
    lower_bound = ph_mean - 3 * ph_std # we accept the ones smaller than mean-3std as outlier
    upper_bound = ph_mean + 3 * ph_std # we accept the ones bigger than mean+3std as outlier
    df_cleaned = df_cleaned[(df_cleaned['soil_ph'] >= lower_bound) & (df_cleaned['soil_ph'] <= upper_bound)] # only keeps the ones we accept in these interval. ( mean +- 3std ).  [] means take only the true statements which means it is for filtration 
    
    print(f"After cleaning, 'soil_ph' values are within the range [{lower_bound:.2f}, {upper_bound:.2f}].")
    print(df_cleaned.head()) # shows first 5 row of clean version as default 
    return df_cleaned

def compute_statistics(df, column):
    """
    Compute and print descriptive statistics for the specified column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute statistics.
    """
    # TODO: Calculate minimum value
    min_val = df[column].min() # calcs. min value of the column taken 
    
    # TODO: Calculate maximum value
    max_val = df[column].max() # calcs. max value
    
    # TODO: Calculate mean value
    mean_val = df[column].mean() # calcs. mean
    
    # TODO: Calculate median value
    median_val = df[column].median()
    
    # TODO: Calculate standard deviation
    std_val = df[column].std()
    
    print(f"\nDescriptive statistics for '{column}':")
    print(f"  Minimum: {min_val}")
    print(f"  Maximum: {max_val}")
    print(f"  Mean: {mean_val:.2f}") # prints mean value with 2 decimals
    print(f"  Median: {median_val:.2f}")
    print(f"  Standard Deviation: {std_val:.2f}")

def main():
    # TODO: Update the file path to point to your soil_test.csv file
    file_path=r'C:\Users\Berat Koncuk\Downloads\soil_test.csv' # Path of the CSV file
    # The r, means "raw string" and it tells Python to read the backslashes (\) as normal characters.
    # Without 'r', Python would treat things like '\n' or '\t' as special characters

    # TODO: Load the dataset using the load_data function
    df = load_data(file_path)
    if df is None:
        return     # Stops if file not loaded properly 
    
    # TODO: Clean the dataset using the clean_data function
    df_clean = clean_data(df)  # Cleans the data (fill missing, remove outliers) as we defined
    
    # TODO: Compute and display statistics for the 'soil_ph' column
    compute_statistics(df_clean, 'soil_ph')  #  Shows stats for soil_ph
    compute_statistics(df_clean, 'nitrogen') # Shows stats for nitrogen 
    compute_statistics(df_clean, 'phosphorus') #shows stats for phosphorus
    compute_statistics(df_clean, 'moisture') #shows stats for moisture
    
    # TODO: (Optional) Compute statistics for other columns
    # compute_statistics(df_clean, 'nitrogen')
    # compute_statistics(df_clean, 'phosphorus')
    # compute_statistics(df_clean, 'moisture')
    
if __name__ == '__main__':
    main()   # If this file is imported by another script, this block will not run automatically.
    #Checks if this file is being run directly or imported and if imported this line won't work. 

# =============================================================================
# REFLECTION QUESTIONS
# =============================================================================
# Answer these questions in comments below:

# 1. What was the most challenging part of this lab?
# Answer: Understanding how to detect and remove outliers in the dataset was the most challenging part.

# 2. How could soil data analysis help civil engineers in real projects?
# Answer: It helps engineers evaluate ground conditions and choose safe and efficient foundation designs.

# 3. What additional features would make this soil analysis tool more useful?
# Answer: Adding graphs and correlation analysis would make the tool more informative and practical.

# 4. How did error handling improve the robustness of your code?
# Answer: It prevented the program from crashing and provided clear feedback when something went wrong.
