
"""
Lab 3: ERA5 Weather Data Analysis
---------------------------------
Analyzes ERA5 wind data for Berlin and Munich with a 'timestamp' column.
Generates monthly, seasonal, and diurnal statistics and visualizations.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD ERA5 DATA WITH "timestamp" COLUMN

def load_era5_data(file_path:str):

    df = pd.read_csv(file_path) 
    
    
    # 'timestamp' must be in "Year-Month-Day Hour:Minute:Second" format.
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') # Convert the 'timestamp' column from string to datetime type
    # 'errors="coerce"' means any invalid date values will be replaced with NaT (Not a Time)
    df.set_index('timestamp', inplace=True) # Instead of numeric index (0, 1, 2) makes the 'timestamp' column as index
    df.sort_index(inplace=True) #orders chronologically
    
    return df


# 2. PATHS OF DATA FILES

BERLIN_FILE = r"C:\Users\Berat Koncuk\OneDrive - Arup\Masaüstü\GitHub\CE49X_Fall2025_Koncuk_Berat\Lab3\berlin_era5_wind_20241231_20241231.csv"
MUNICH_FILE = r"C:\Users\Berat Koncuk\OneDrive - Arup\Masaüstü\GitHub\CE49X_Fall2025_Koncuk_Berat\Lab3\munich_era5_wind_20241231_20241231.csv"



# 3. MAIN EXECUTION 

def main():
    
    df_berlin = load_era5_data(BERLIN_FILE) # Calling the previously defined function for Berlin
    df_munich = load_era5_data(MUNICH_FILE)

    print("----- Berlin Dataset -----")
    print(df_berlin.head()) #Print first 5 rows (as default) of Berlin dataset to see the structure and check values

    print("\n----- Munich Dataset -----")
    print(df_munich.head())

    # ---------------------------------------------------------------------
    # 3.1 CHECKING AND CLEANING MISSING DATA
    # ---------------------------------------------------------------------
    
    # Checks if Berlin DataFrame has any missing values
    berlin_has_missing = df_berlin.isna().any().any()
    """first any() checks each column if is there at least one True (missing value) in this column and
    the 2nd one checks all columns whether is there at least one True in the entire DataFrame?"""

    # Checks if Munich DataFrame has any missing values
    munich_has_missing = df_munich.isna().any().any()

    # Removes any rows that contain missing values
    df_berlin.dropna(inplace=True) #'inplace=True' means we update the original DataFrame instead of creating a new one
    df_munich.dropna(inplace=True)
    
    # ---------------------------------------------------------------------
    # 3.2 WIND SPEED CALCULATION STARTS
    # ---------------------------------------------------------------------
    def calculate_wind_speed(u: pd.Series, v: pd.Series):
        #Computes wind speed from wind components u(E-W) and v(N-S).

        return np.sqrt(u**2 + v**2) # applies the formula for all values

    df_berlin['wind_speed'] = calculate_wind_speed(df_berlin['u10m'], df_berlin['v10m']) # Our csv has columns as u10m and v10m. 
    # Passes the 'u10m' and 'v10m' Series from the DataFrame into the function
    df_munich['wind_speed'] = calculate_wind_speed(df_munich['u10m'], df_munich['v10m'])

    # ---------------------------------------------------------------------
    # 3.3 TEMPORAL AGGREGATIONS
    # ---------------------------------------------------------------------
    def monthly_average(df: pd.DataFrame, column ):
        return df.groupby(df.index.month)[column].mean() # first groups all rows by month and [column].mean() computes the mean of the selected column for each month

    berlin_monthly_wind = monthly_average(df_berlin, 'wind_speed') # Computes monthly average wind speed for Berlin
    munich_monthly_wind = monthly_average(df_munich, 'wind_speed')

    def get_season(month: int):
        
        # 1 = Winter, 2 = Spring, 3 = Summer, 4 = Autumn
        if month in [12, 1, 2]:
            return 1 # returns as winter
        elif month in [3, 4, 5]:
            return 2
        elif month in [6, 7, 8]:
            return 3
        else:
            return 4

    # Assign seasons
    df_berlin['season'] = df_berlin.index.month.map(get_season) # df.index.month extracts the month and map() takes a function and applies it to all elements 
    df_munich['season'] = df_munich.index.month.map(get_season) # .map(get_season) converts it to a season code (1–4)

    berlin_seasonal_wind = df_berlin.groupby('season')['wind_speed'].mean() # Groups the datas by season and compute the mean wind speed for each season
    munich_seasonal_wind = df_munich.groupby('season')['wind_speed'].mean()

    # ---------------------------------------------------------------------
    # 3.4 STATISTICAL ANALYSIS
    # ---------------------------------------------------------------------
    
    # Resample the DataFrame to daily frequency ('D') and compute daily mean
    df_berlin_daily = df_berlin.resample('D').mean(numeric_only=True) # only takes numerical data
    df_munich_daily = df_munich.resample('D').mean(numeric_only=True) # This converts hourly data into daily averages

    print("\n=== Top 5 Extreme Wind Speed Days (Berlin) ===")
    print(df_berlin_daily['wind_speed'].nlargest(5)) # Find the top 5 days with highest wind speeds 
    # nlargest(5) returns the 5 largest values in the 'wind_speed' column
    print("\n=== Top 5 Extreme Wind Speed Days (Munich) ===")
    print(df_munich_daily['wind_speed'].nlargest(5))

    df_berlin['hour'] = df_berlin.index.hour # df.index.hour gives us the hour (0–23) of each row
    df_munich['hour'] = df_munich.index.hour

    berlin_hourly_pattern = df_berlin.groupby('hour')['wind_speed'].mean() # Computes the average wind speed for each hour of the day
    munich_hourly_pattern = df_munich.groupby('hour')['wind_speed'].mean() # groupby('hour') groups rows by the hour, mean() calculates average wind_speed

    # ---------------------------------------------------------------------
    # 3.5 VISUALIZATIONS
    # ---------------------------------------------------------------------
    # Optional improvements to default Matplotlib appearance:
    plt.rcParams['figure.facecolor'] = 'gray'  # Make the figure background white
    plt.rcParams['axes.facecolor']   = 'gray'  # Make the area behind the axes white too
    plt.rcParams.update({
        'axes.grid'        : True,   # Show grid lines behind the data
        'grid.alpha'       : 0.3,    # Make grid lines light and less distracting
        'lines.linewidth'  : 2.3,    # Make plot lines thicker 
        'lines.markersize' : 7,      # Larger markers
        'font.size'        : 13,     # Increase default font size 
        })

    # Monthly Average Wind Speed
    plt.figure(figsize=(10, 6)) # Creates a new figure with size (10, 6)
    plt.plot(berlin_monthly_wind.index, berlin_monthly_wind.values, marker='o', label='Berlin')
    plt.plot(munich_monthly_wind.index, munich_monthly_wind.values, marker='o', label='Munich')
    plt.title("Monthly Average Wind Speed (2024)", fontsize=16, pad=10) # Add title and axis labels with custom font sizes
    plt.xlabel("Month", fontsize=15)
    plt.ylabel("Wind Speed (m/s)", fontsize=14)
    plt.xticks(range(1, 13)) # Sets x-axis  to show months as numbers 1 through 12
    plt.legend(fontsize=12) # Add a lejant to differentiate Berlin and Munich lines
    plt.show()

    # Seasonal Comparison
    season_labels = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'} # Keys are numeric codes (1–4) and values are their string names

    plt.figure(figsize=(8, 5)) # Create a new figure with a specific size (width=8, height=5)
    plt.bar(berlin_seasonal_wind.index - 0.15, berlin_seasonal_wind.values, width=0.3, label='Berlin') # The "- 0.15" shifts Berlin’s bars slightly to the left for better visibility
    plt.bar(munich_seasonal_wind.index + 0.15, munich_seasonal_wind.values, width=0.3, label='Munich')
    plt.title("Seasonal Average Wind Speed (2024)", fontsize=16, pad=10) # pad leaves a space betwen graph and title
    plt.xlabel("Season", fontsize=14) # Label the x-axes
    plt.ylabel("Wind Speed (m/s)", fontsize=14) # Label the y-axes
    plt.xticks([1, 2, 3, 4], [season_labels[s] for s in [1, 2, 3, 4]], fontsize=12) # Replaces the x-axis numbers (1–4) with text labels ('Winter', 'Spring', etc.)
    plt.legend(fontsize=12)
    plt.show()

    # Hourly Pattern
    plt.figure(figsize=(10, 6)) # Creates a new figure with width=10 and height=6

    # marker='o' adds small circles at each data point
    plt.plot(berlin_hourly_pattern.index, berlin_hourly_pattern.values, marker='o', label='Berlin')
    plt.plot(munich_hourly_pattern.index, munich_hourly_pattern.values, marker='o', label='Munich')
    plt.title("Average Diurnal (Hourly) Wind Speed", fontsize=16, pad=10) # Adds a title to the graph
    plt.xlabel("Hour of the Day", fontsize=14)
    plt.ylabel("Wind Speed (m/s)", fontsize=14)
    plt.xticks(range(0, 24)) # Sets the x-axis ticks from 0 to 23 (representing 24 hours)
    plt.legend(fontsize=12)
    plt.show()


    print("\nDone! All calculations and plots use the 'timestamp' column as DatetimeIndex.")


if __name__ == "__main__":  # Runs main() only if this script is executed directly, not imported
    main()
# Eğer dosya doğrudan çalıştırılıyorsa → __name__ == "__main__"
#Eğer dosya başka bir dosyadan import edildiyse → __name__ == "dosya_adı"