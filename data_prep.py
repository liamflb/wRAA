from pybaseball import cache
cache.purge()
from pybaseball import statcast
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#######load data#######

RAW_CACHE_2024  = "statcast_2024-03-28_2024-09-30.pkl"
RAW_CACHE_2023  = "statcast_2023-03-30_2023-10-01.pkl"
RAW_CACHE_2022  = "statcast_2022-04-07_2022-10-02.pkl"

if os.path.exists(RAW_CACHE_2024):
    df_2024 = pd.read_pickle(RAW_CACHE_2024)
else:
    df_2024 = statcast(start_dt="2024-03-28", end_dt="2024-09-30")
    df_2024.to_pickle(RAW_CACHE_2024)
    print(f"Saved raw data to {RAW_CACHE_2024!r}")

if os.path.exists(RAW_CACHE_2023):
    df_2023 = pd.read_pickle(RAW_CACHE_2023)
else:
    df_2023 = statcast(start_dt="2023-03-30", end_dt="2023-10-02")
    df_2023.to_pickle(RAW_CACHE_2023)
    print(f"Saved raw data to {RAW_CACHE_2023!r}")

if os.path.exists(RAW_CACHE_2022):
    df_2022 = pd.read_pickle(RAW_CACHE_2022)
else:
    df_2022 = statcast(start_dt="2022-04-27", end_dt="2022-10-02")
    df_2022.to_pickle(RAW_CACHE_2022)
    print(f"Saved raw data to {RAW_CACHE_2022!r}")

sprint_speeds = pd.read_csv("sprint_speed.csv")

######combine dataframes######

data_2024 = df_2024.copy()
data_2023 = df_2023.copy()
data_2022 = df_2022.copy()
data_2024['year'] = 2024
data_2023['year'] = 2023
data_2022['year'] = 2022
data = pd.concat([data_2024, data_2023, data_2022])
data = data.reset_index(drop=True)
data.rename(columns={'batter': 'player_id'}, inplace=True)

print(data['events'].value_counts())


#######create total bases field############

condensed_data = data[['player_id', 'woba_value', 'estimated_woba_using_speedangle', 'iso_value',
                      'launch_speed', 'launch_angle', 'events', 'year']]

walk_hbp = condensed_data[condensed_data['events'].isin(["walk", "hit_by_pitch"])]
strikeouts = condensed_data[condensed_data['events'].isin(["strikeout", "strikeout_double_play"])]


merged_data = condensed_data.merge(sprint_speeds, on="player_id", how='left')
merged_data['total_bases'] =  merged_data.apply(
             lambda row: row['iso_value'] + 1 if pd.notna(row['woba_value']) and row['woba_value'] > 0 else 0,
             axis=1)

### visualize ###

merged_data.plot(kind='hexbin', x="launch_speed", y="launch_angle",
                C="total_bases", colormap=plt.get_cmap('jet'), colorbar=True)

plt.show()

######split into train and test######

train_set, test_set = train_test_split(merged_data, test_size=0.2, random_state=42)

######create pipeline to clean data######
    
class RowDropper(BaseEstimator, TransformerMixin):
    def fit (self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        condition = X['woba_value'].notna() & X['launch_speed'].notna()
        return X.loc[condition]
    
num_attributes = ['launch_speed', 'launch_angle', 'iso_value', 'woba_value', 'sprint_speed','total_bases']

cat_attributes = ["events", 'estimated_woba_using_speedangle', 'player_id', 'year']
        
num_pipeline= Pipeline([
    ("std_scaler", StandardScaler())
])    

pre_pipline = ColumnTransformer([
    ("num", num_pipeline, num_attributes)],
    remainder="passthrough"
)

full_pipeline = Pipeline([
    ("row_drop", RowDropper()),
    ("pre", pre_pipline)
])

######Apply pipeline to training and testing data######

prepared_data = full_pipeline.fit_transform(train_set)
prepared_test_data = full_pipeline.fit_transform(test_set)

prepared_df =  pd.DataFrame(
    prepared_data
    )

prepared_test_df = pd.DataFrame(
    prepared_test_data
    )

prepared_df = prepared_df.rename(columns={
    0:"launch_speed",
    1:"launch_angle",
    2:"iso_value",
    3:"woba_value",
    4:"sprint_speed",
    5:"total_bases",
    6: "player_id",
    7: "estimated_woba_using_speedangle",
    8: "events",
    9: "year",
    10: "name"
})

prepared_test_df = prepared_test_df.rename(columns={
    0:"launch_speed",
    1:"launch_angle",
    2:"iso_value",
    3:"woba_value",
    4:"sprint_speed",
    5:"total_bases",
    6: "player_id",
    7: "estimated_woba_using_speedangle",
    8: "events",
    9: "year",
    10: "name"
})

desired_columns = ['player_id',
                   'woba_value',
                   'estimated_woba_using_speedangle',
                   'iso_value',
                   'launch_speed',
                   'launch_angle',
                   'events',
                   'total_bases',
                   'year']


###### save data to csvs ######

walk_hbp['total_bases'] = 1
strikeouts['total_bases'] = 0

prepared_df = prepared_df[desired_columns]
prepared_test_df = prepared_test_df[desired_columns]
walk_hbp = walk_hbp[desired_columns]

prepared_df.to_csv("model_stats.csv", index=False)
prepared_test_df.to_csv("test_stats.csv", index=False)
walk_hbp.to_csv("walk_hbp.csv", index=False)
strikeouts.to_csv("strikeouts.csv", index=False)

