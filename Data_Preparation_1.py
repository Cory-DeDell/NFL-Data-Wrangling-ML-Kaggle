# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:48:01 2022

@author: Cory
"""
import pandas as pd
import numpy as np
from Data_Preparation_Functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", None)
train_df = pd.read_csv('raw_data.csv', low_memory=False)

# fix label error
map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in train_df['PossessionTeam'].unique():
    map_abbr[abb] = abb
   
train_df['PossessionTeam'] = train_df['PossessionTeam'].map(map_abbr)
train_df['HomeTeamAbbr'] = train_df['HomeTeamAbbr'].map(map_abbr)
train_df['VisitorTeamAbbr'] = train_df['VisitorTeamAbbr'].map(map_abbr)
train_df['FieldPosition'] = train_df['FieldPosition'].map(map_abbr)

# create indicator for team column, home = 1 away = 0 
train_df['Team'] = train_df['Team'].apply(lambda x: x == 'home')

# create column for play direction to the left
train_df['ToLeft'] = train_df.PlayDirection == "left"

# create column to indicate the ball carrier
train_df['IsBallCarrier'] = train_df.NflId == train_df.NflIdRusher

# create indicator for player on offense or defense
train_df['TeamOnOffense'] = 1 # for home team placeholder
# if home team is not possessing the ball TeamOnOffense = 0 (away team is possessing ball)
train_df.loc[train_df.PossessionTeam != train_df.HomeTeamAbbr, 'TeamOnOffense'] = 0 # away team placeholder
# if team = 1 and TeamOnOffense = 1, home team is on offense
# if team = 0 and TeamOnOffense = 0, away team is on offense
train_df['IsOnOffense'] = (train_df.Team == train_df.TeamOnOffense).astype(int) 
# remove placeholder column
train_df.drop(columns=['TeamOnOffense'], inplace=True)

# standardize player coordinates such that the play is always going left to right
# if the play direction is going to the left, map to new coordinates
train_df['X_std'] = train_df.X
train_df.loc[train_df.ToLeft, 'X_std'] = 120 - train_df.loc[train_df.ToLeft, 'X'] 
train_df['Y_std'] = train_df.Y
train_df.loc[train_df.ToLeft, 'Y_std'] = 160/3 - train_df.loc[train_df.ToLeft, 'Y'] 

# if play direction is to the left, reorient angle for standardized left to right play direction
train_df['Orientation_std'] = train_df.Orientation
train_df.loc[train_df.ToLeft, 'Orientation_std'] = np.mod(180 + train_df.loc[train_df.ToLeft, 'Orientation_std'], 360)
train_df['Dir_std'] = train_df.Dir
train_df.loc[train_df.ToLeft, 'Dir_std'] = np.mod(180 + train_df.loc[train_df.ToLeft, 'Dir_std'], 360)

# clean up weather text
train_df['GameWeather'] = train_df.GameWeather.apply(clean_up_weather_text)    

# clean up player positions
train_df['Position'] = train_df.apply(lambda row : clean_up_positions(row['PossessionTeam'], row['VisitorTeamAbbr'],
                      row['HomeTeamAbbr'], row['Team'], row['Position']), axis = 1)

# encode turf type natural/artificial
train_df['Turf'] = train_df['Turf'].map(Turf)

# convert gameclock time to minutes left in quarter
train_df.GameClock = train_df.GameClock.apply(convert_gameclock_to_minutes_elapsed_quarter)
train_df['TimeLeft'] = (train_df.Quarter*15) - train_df.GameClock

# drop gameclock and quarter columns
train_df.drop(columns=['Quarter', 'GameClock'], inplace=True)

# FEATURE ENGINEERING
# add yards from line of scrimmage to endzone
train_df['Yards_to_endzone'] = train_df.apply(lambda row : yards_to_goal_line(row['PossessionTeam'], 
                      row['FieldPosition'], row['YardLine']), axis = 1)
# drop yardline column
train_df.drop(columns=['YardLine'], inplace=True)

# FEATURE ENGINEERING
# encode point diferential. Create a function that returns the current point differential
# between the team in possesion and the team defending (team_pos_score - team_defending_score)
train_df['Point_differential'] = train_df.apply(lambda row : encode_point_differential(row['HomeTeamAbbr'],
                      row['PossessionTeam'], row['HomeScoreBeforePlay'], row['VisitorScoreBeforePlay']), axis = 1)

# FEATURE ENGINEERING
# create column for total time elapsed from snap to handoff
train_df['time_elapsed'] = (pd.to_datetime(train_df.TimeHandoff)-pd.to_datetime(train_df.TimeSnap)).dt.total_seconds()
# remove TimeHandoff and TimeSnap columns
train_df.drop(columns=['TimeHandoff', 'TimeSnap'], inplace=True)

# # encode defensive team
# train_df['DefenseTeam'] = train_df.apply(lambda row : encode_defensive_team(row['PossessionTeam'], 
#                         row['HomeTeamAbbr'], row['VisitorTeamAbbr']), axis = 1)

# encode defensive personnel count
train_df[['num_DL','num_LB','num_DB']] = train_df['DefensePersonnel'].apply(encode_defense_personnel_count)
# remove DefensePersonnel column
train_df.drop(columns=['DefensePersonnel'], inplace=True)

# encode offense personnel count
train_df[['num_OL','num_RB','num_WR','num_TE','num_QB']] = train_df['OffensePersonnel'].apply(encode_offense_personnel_count)
# remove OffensePersonnel column
train_df.drop(columns=['OffensePersonnel'], inplace=True)

# convert player height to inches
train_df['PlayerHeight'] = train_df.PlayerHeight.apply(player_height_to_inches)

# convert player age to seconds from January 1, 1970
train_df['PlayerBirthTime'] = train_df.PlayerBirthDate.apply(player_birthday_to_birth_in_seconds_from_1970)
# remove PlayerBirthDate column
train_df.drop(columns=['PlayerBirthDate'], inplace=True)

# convert wind speeds to integers and fill missing values with mean
train_df['WindSpeed'] = train_df.WindSpeed.apply(convert_wind_speed)   
mean_wind = train_df['WindSpeed'].mean()
train_df['WindSpeed'].fillna(mean_wind, inplace=True)

# fill missing temperature values with mean
mean_temp = train_df['Temperature'].mean()
train_df['Temperature'].fillna(mean_temp, inplace=True)

# fill missing humidity values with mean
mean_hum = train_df['Humidity'].mean()
train_df['Humidity'].fillna(mean_hum, inplace=True)

# encode one-hot features...curenttly do not have possession or or team on defense encoded
train_df = pd.get_dummies(train_df, columns=['OffenseFormation', 'Position', 'GameWeather', 'Turf'],
                          prefix=['OffenseFormation', 'Position', 'weather', 'Turf'])

# sort the dataframe such that similar players will be oriented the same in the feature vector
train_df = train_df.sort_values(by=['PlayId', 'IsOnOffense', 'IsBallCarrier', 'JerseyNumber']).reset_index(drop=True)


remove_columns = ['GameId', 'NflId', 'NflIdRusher', 'PlayerCollegeName', 'DisplayName', 'Stadium', 'Location',
                  'StadiumType', 'Season', 'JerseyNumber', 'WindDirection', 'HomeTeamAbbr', 'FieldPosition',
                  'VisitorTeamAbbr', 'Team', 'PossessionTeam', 'ToLeft', 'X', 'Y', 'Orientation', 'Dir',
                  'PlayDirection', 'IsBallCarrier']

train_df.drop(columns=remove_columns, inplace=True)

# remove plays with missing direction data for players
null_list = np.where(train_df.isnull())[0]
null_play_ids = np.array(train_df['PlayId'].iloc[null_list])
train_df = train_df.loc[(~train_df['PlayId'].isin(null_play_ids))]
train_df.reset_index(drop=True, inplace=True)

# remove play id column
train_df.drop(columns=['PlayId'], inplace=True)

train_df.to_csv('raw_data_modified.csv', index=False)