# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:18:21 2022

@author: Cory
"""
import re
from datetime import datetime

# data analysis and wrangling
import pandas as pd
import numpy as np
import random
import statistics

# wrangling data

######################################
######################################

# encode stadium as one hot

def apply_ordinal_encoder_stadium(txt):
    '''
    if a game was help outside or with an open roof, return 0
    if a game was held in a dome, return 1
    assume games labeled as 'retractable roof' had no significant influencial weather in stadium, return 1
    '''
    txt = txt.lower().replace(' ','')
    if 'ou' in txt or 'open' in txt:
        return 0
    elif 'heinz' in txt:
        return 0
    elif 'cloudy' in txt:
        return 0
    elif 'bowl' in txt:
        return 0
    elif 'roof' in txt and 'open' in txt:
        return 0
    elif 'indoor' in txt and 'open' in txt:
        return 0
    elif 'indoor' in txt and 'open' not in txt:
        return 1
    elif 'retractableroof' in txt:
        return 1
    elif 'roof' in txt and 'close' in txt:
        return 1
    elif 'indoor' in txt and 'close' in txt:
        return 1
    elif 'dome' in txt and 'close' in txt:
        return 1
    elif 'dome' in txt and 'close' not in txt and 'open' not in txt:
        return 1
    else:
        return np.nan
    
# train_df.StadiumType = train_df.StadiumType.apply(apply_ordinal_encoder_stadium)

######################################
######################################

# clean up weather text and encode one hot (good/bad/unknown)

def clean_up_weather_text(txt):
    
    if pd.isna(txt):
        return 'unknown conditions'
    
    txt = txt.lower().replace(' ','')
    
    if ('cloudy' in txt or 'sun' in txt or 'clear' in txt or 'indoor' in txt or\
    'controlledclimate' in txt or 'clouidy' in txt or'overcast' in txt or 'fair' in txt\
    or 'coudy' in txt) and ('rain' not in txt) and ('snow' not in txt) and ('cold' not in txt)\
    and ('cool' not in txt) and ('chance' not in txt) and ('showers' not in txt) and ('change' not in txt):
        return 'good conditions'
    
    elif ('rain' in txt or 'snow' in txt or 'cold' in txt or 'cool' in txt or 'showers' in txt)\
    and ('indoor' not in txt) and ('controlledclimate' not in txt) and\
    ('clouidy' not in txt) and ('overcast' not in txt) and ('fair' not in txt)\
    and ('chance' not in txt) and ('change' not in txt):
        return 'bad conditions'
    
    else:
        return 'unknown conditions'

######################################
######################################

# encode turf type natural/artificial

Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural',
        'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial',
        'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 'UBU Speed Series-S5-M':'Artificial',
        'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural',
        'grass':'Natural', 'Natural':'Natural', 'Artifical':'Artificial',
        'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial',
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'}


######################################
######################################

# clean up positions

def clean_up_positions(team_pos, team_away, team_home, team, position):
    # create list of positions that are already ok
    acceptable_positions = ['WR', 'TE', 'RB', 'QB', 'LB', 'FB', 'DL']
    
    if position in acceptable_positions:
        return position
    
    linebackers = ['ILB', 'MLB', 'OLB']
    offensive_line = ['C', 'OT', 'OG', 'G']
    defensive_line = ['DT', 'DE', 'NT']
    safeties_corners = ['SAF', 'FS', 'SS','CB']
    
    # determine if player is on offense or deffense
    if team == 1 and team_pos == team_home:
        player_pos = 'offense'
    elif team == 1 and team_pos != team_home:
        player_pos = 'defense'
    elif team == 0 and team_pos == team_away:
        player_pos = 'offense'
    elif team == 0 and team_pos != team_away:
        player_pos = 'defense'
    
    # encode correct tackle position
    if position == 'T' and player_pos == 'offense':
        return 'OL'
    elif position == 'T' and player_pos == 'defense':
        return 'DL'
    
    if position in offensive_line:
        return 'OL'
    
    if position in defensive_line:
        return 'DL'
    
    if position in linebackers:
        return 'LB'
    
    if position in safeties_corners:
        return 'S/CB'
    
    if position == 'HB':
        return 'RB'
    
######################################
######################################

# encode jersey number as typical position then encode as one-hot
def encode_jersey_number(num):
    '''
    1-9 Quarter Backs / Kickers / Punters
    10-19 Quarter Backs / Wide Receivers /Kickers / Punters
    20-29 Running Backs /Corner Backs / Safeties
    30-39 Running Backs / Corner Backs / Safeties
    40-49 Running Backs / Tight Ends / Corner Backs / Safeties
    50-59 Offensive Linemen /Defensive Linemen / Line Backer
    60-69 Offensive Linemen / Defensive Linemen
    70-79 Offensive Linemen / Defensive Linemen
    80-89 Wide Receiver / Tight End
    90-99 Defensive Linemen / Line Backer
    '''
    if 1 <= num <= 9:
        return 'QB/K/P'
    elif 10 <= num <= 19:
        return 'QB/WR/K/P'
    elif 20 <= num <= 39:
        return 'RB/CB/S'
    elif 40 <= num <= 49:
        return 'RB/TE/CB/S'
    elif 50 <= num <= 59:
        return 'OL/DL/LB'
    elif 60 <= num <= 79:
        return 'OL/DL'        
    elif 80 <= num <= 89:
        return 'WR/TE'            
    elif 90 <= num <= 99:
        return 'DL/LB'          
    
# train_df.JerseyNumber = train_df.JerseyNumber.apply(encode_jersey_number)
# train_df = pd.get_dummies(train_df, columns=['JerseyNumber'], prefix='pos')

######################################
######################################

# convert gameclock time to minutes left in quarter

def convert_gameclock_to_minutes_elapsed_quarter(txt):
    '''
    converts gameclock time to time elapsed in quarter
    '''
    txt = txt.split(':')
    x = list(map(int,txt))
    x[1] = x[1] / 60
    return 15 - sum(x)

######################################
######################################

# create function that returns yards to goal-line of team on defense from line of scrimmage

def yards_to_goal_line(team_possession, team_field_position, line_of_scrimmage):
    if team_possession == team_field_position:
        return 100 - line_of_scrimmage
    else:
        return line_of_scrimmage
    
######################################
######################################

# encode defensive team

def encode_defensive_team(team_pos, team_home, team_away):
    '''
    returns the team that is on the defensive side of the ball. This will
    be used to one-hot encode the team on defense

    Parameters
    ----------
    team_pos : team on offense
    team_home : home team
    team_away : away team
    
    '''
    if team_pos == team_home:
        return team_home
    elif team_pos == team_away:
        return team_away

######################################
######################################

def encode_point_differential(home_team_txt, pos_team_txt, team_home_score, team_away_score):
    if home_team_txt == pos_team_txt:
        return team_home_score - team_away_score
    else:
        return team_away_score - team_home_score
    
######################################
######################################

def encode_defense_personnel_count(txt):
    dict_ = {'DL':0,'LB':0,'DB':0}
    txt = txt.split(',')
    for t in txt:
        num = int(re.findall('[0-9]+', t)[0])
        pos = re.findall('[a-zA-Z]+', t)[0]
        if pos not in dict_.keys():
            pass
        else:
            dict_[pos] = num
    return pd.Series([dict_['DL'], dict_['LB'], dict_['DB']])
    
######################################
######################################

def encode_offense_personnel_count(txt):
    dict_ = {'OL':0,'RB':0,'WR':0,'TE':0,'QB':0}
    txt = txt.split(',')
    for t in txt:
        num = int(re.findall('[0-9]+', t)[0])
        pos = re.findall('[a-zA-Z]+', t)[0]
        if pos not in dict_.keys():
            pass
        else:
            dict_[pos] = num
    if dict_['QB'] == 0:
        dict_['QB'] = 1
    num_players = sum(dict_.values())
    if dict_['OL'] == 0:
        dict_['OL'] = 11 - num_players
    return pd.Series([dict_['OL'], dict_['RB'], dict_['WR'], dict_['TE'], dict_['QB']])

######################################
######################################

def player_height_to_inches(txt):
    '''
    convert player height from feet-inches to inches

    '''
    x = txt.split('-')
    a = int(x[0])*12
    b = int(x[1])
    return a + b

######################################
######################################

def player_birthday_to_birth_in_seconds_from_1970(txt):
    '''
    convert player birthday to birth time in seconds from January 1, 1970
    
    '''
    birthday = txt.split('/')
    month = int(birthday[0])
    day = int(birthday[1])
    year = int(birthday[2])
    return datetime(year, month, day).timestamp()

######################################
######################################

def convert_wind_speed(txt):
    '''
    if range of values is provided, return the mean.
    '''
    if pd.isnull(txt):
        return np.nan
    if txt == 'Calm':
        return 0
    if txt.isalpha():
        return np.nan
    if txt.isnumeric():
        return int(txt)
    if '-' in txt:
        txt = txt.split('-')
        return (int(txt[0]) + int(txt[1])) / 2
    txt = txt[0:2]
    num = int(re.findall('[0-9]+', txt)[0])
    return num

