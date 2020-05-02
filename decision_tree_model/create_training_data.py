# Processing the NBA data set obtained from https://www.kaggle.com/nathanlauga/nba-games/data

import pandas as pd

team_df = pd.read_csv('../datasets/teams.csv') # Obtained from https://www.kaggle.com/nathanlauga/nba-games/data
games_df = pd.read_csv('../datasets/games.csv') # Obtained from https://www.kaggle.com/nathanlauga/nba-games/data
stats_df = pd.read_csv('../datasets/teamstats.csv') # Obtained from Basketball Reference

# Processing Team Identification Data
team_ids = {}
for index, row in team_df.iterrows():
    team_ids[row['TEAM_ID']] = row['CITY'] + ' ' + row['NICKNAME']

# Removing asterisks from end of team names in the stats data frame
for index, row in stats_df.iterrows():
    if row['Team'].endswith('*'):
        stats_df.at[index, 'Team'] = row['Team'][:-1]

# Extracting the game data and team stat data by season
# Getting 2014-2015
mask = (games_df['SEASON'] == 2014)
games_2014 = games_df.loc[mask]
stats_mask = (stats_df['Year'] == 2014)
stats_2014 = stats_df.loc[stats_mask]

# Getting 2015-2016
mask = (games_df['SEASON'] == 2015)
games_2015 = games_df.loc[mask]
stats_mask = (stats_df['Year'] == 2015)
stats_2015 = stats_df.loc[stats_mask]

# Getting 2016-2017
mask = (games_df['SEASON'] == 2016)
games_2016 = games_df.loc[mask]
stats_mask = (stats_df['Year'] == 2016)
stats_2016 = stats_df.loc[stats_mask]

# Getting 2017-2018
mask = (games_df['SEASON'] == 2017)
games_2017 = games_df.loc[mask]
stats_mask = (stats_df['Year'] == 2017)
stats_2017 = stats_df.loc[stats_mask]

# Getting 2018-2019
mask = (games_df['SEASON'] == 2018)
games_2018 = games_df.loc[mask]
stats_mask = (stats_df['Year'] == 2018)
stats_2018 = stats_df.loc[stats_mask]

# Processing each game in the seasons by subtracting away team averages from the home team averages in the following categories and putting them in dataframes
categories = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W']

# games = games_xxxx dataframes
# stats = stats_xxxx dataframes
def get_processed_dataframe(games, stats):
    processed = []
    for index, row in games.iterrows():
        home_team = team_ids[row['HOME_TEAM_ID']]
        away_team = team_ids[row['VISITOR_TEAM_ID']]
        home_team_stats = stats.loc[stats['Team'] == home_team]
        away_team_stats = stats.loc[stats['Team'] == away_team]
        temp = {}

        #temp['Top_Team'] = home_team
        #temp['Bottom_Team'] = away_team
        # Getting differences in each of the desired categories
        for category in categories:
            temp[category] = round(home_team_stats[category].iloc[0] - away_team_stats[category].iloc[0], 2)

        if row['HOME_TEAM_WINS'] == 1:
            temp['Top_Team_Won'] = True
        else: 
            temp['Top_Team_Won'] = False

        processed.append(temp)

    df = pd.DataFrame(processed)
    return df

processed_2014 = get_processed_dataframe(games_2014, stats_2014)
processed_2015 = get_processed_dataframe(games_2015, stats_2015)
processed_2016 = get_processed_dataframe(games_2016, stats_2016)
processed_2017 = get_processed_dataframe(games_2017, stats_2017)
processed_2018 = get_processed_dataframe(games_2018, stats_2018)

processed_2014.to_csv('2014_2015_training_data.txt', encoding='utf-8', index=False)
processed_2015.to_csv('2015_2016_training_data.txt', encoding='utf-8', index=False)
processed_2016.to_csv('2016_2017_training_data.txt', encoding='utf-8', index=False)
processed_2017.to_csv('2017_2018_training_data.txt', encoding='utf-8', index=False)
processed_2018.to_csv('2018_2019_training_data.txt', encoding='utf-8', index=False)