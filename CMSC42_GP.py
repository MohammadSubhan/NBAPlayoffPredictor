import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Preprocessing
# ==============================================================================================================================
nba_df = pd.read_csv('games.csv')
nba_df['GAME_DATE_EST'] = pd.to_datetime(nba_df['GAME_DATE_EST'])
nba_df.drop(['GAME_ID', 'GAME_STATUS_TEXT', 'TEAM_ID_home', 'TEAM_ID_away'], axis=1, inplace=True)
team_df = pd.read_csv('teams.csv')

# Change IDs in the NBA game df to actual name
ids = nba_df['HOME_TEAM_ID'].unique()
for i in ids:
    nba_df.loc[(nba_df.HOME_TEAM_ID == i), 'HOME_TEAM_ID'] = team_df.loc[(team_df.TEAM_ID == i), 'ABBREVIATION'].iloc[0]
    nba_df.loc[(nba_df.VISITOR_TEAM_ID == i), 'VISITOR_TEAM_ID'] = team_df.loc[(team_df.TEAM_ID == i), 'ABBREVIATION'].iloc[0]

# Extract data from different NBA season
df_14 = nba_df.loc[nba_df['SEASON'] == 2014]
df_15 = nba_df.loc[nba_df['SEASON'] == 2015]
df_16 = nba_df.loc[nba_df['SEASON'] == 2016]
df_17 = nba_df.loc[nba_df['SEASON'] == 2017]
df_18 = nba_df.loc[nba_df['SEASON'] == 2018]

# train represents preseason + regular season games
# test represents playoffs + finals
train18, test18 = df_18.loc[df_18['GAME_DATE_EST'] < '2019-4-13'], df_18.loc[df_18['GAME_DATE_EST'] >= '2019-4-13']
train17, test17 = df_17.loc[df_17['GAME_DATE_EST'] < '2018-4-14'], df_17.loc[df_17['GAME_DATE_EST'] >= '2018-4-14']
train16, test16 = df_16.loc[df_16['GAME_DATE_EST'] < '2017-4-15'], df_16.loc[df_16['GAME_DATE_EST'] >= '2017-4-15']
train15, test15 = df_15.loc[df_15['GAME_DATE_EST'] < '2016-4-16'], df_15.loc[df_15['GAME_DATE_EST'] >= '2016-4-16']
train14, test14 = df_14.loc[df_14['GAME_DATE_EST'] < '2015-4-18'], df_14.loc[df_14['GAME_DATE_EST'] >= '2015-4-18']
train_dataset = pd.concat([train18, train17, train16, train15, train14], ignore_index=True)

# Brackets for each NBA season
# 14 represents 14-15 season
# Reference: Wikipedia
# (top, bottom) 1 represent top wons, 0 top loss
bracket14 = [[('ATL', 'BKN'), ('TOR', 'WAS'), ('CHI', 'MIL'), ('CLE', 'BOS'), ('GSW', 'NOP'), ('POR', 'MEM'), ('LAC', 'SAS'),
              ('HOU', 'DAL'), ('ATL', 'WAS'), ('CHI', 'CLE'), ('GSW', 'MEM'), ('LAC', 'HOU'), ('ATL', 'CLE'), ('GSW', 'HOU'),
              ('CLE', 'GSW')], [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]]
bracket15 = [[('CLE', 'DET'), ('ATL', 'BOS'), ('MIA', 'CHA'), ('TOR', 'IND'), ('GSW', 'HOU'), ('LAC', 'POR'), ('OKC', 'DAL'),
              ('SAS', 'MEM'), ('CLE', 'ATL'), ('MIA', 'TOR'), ('GSW', 'POR'), ('OKC', 'SAS'), ('CLE', 'TOR'), ('GSW', 'OKC'),
              ('CLE', 'GSW')], [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]]
bracket16 = [[('BOS', 'CHI'), ('WAS', 'ATL'), ('TOR', 'MIL'), ('CLE', 'IND'), ('GSW', 'POR'), ('LAC', 'UTA'), ('HOU', 'OKC'),
              ('SAS', 'MEM'), ('BOS', 'WAS'), ('TOR', 'CLE'), ('GSW', 'UTA'), ('HOU', 'SAS'), ('BOS', 'CLE'), ('GSW', 'SAS'),
              ('CLE', 'GSW')], [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]]
bracket17 = [[('TOR', 'WAS'), ('CLE', 'IND'), ('PHI', 'MIA'), ('BOS', 'MIL'), ('HOU', 'MIN'), ('OKC', 'UTA'), ('POR', 'NOP'),
              ('GSW', 'SAS'), ('TOR', 'CLE'), ('PHI', 'BOS'), ('HOU', 'UTA'), ('NOP', 'GSW'), ('CLE', 'BOS'), ('HOU', 'GSW'),
              ('CLE', 'GSW')], [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]]
bracket18 = [[('MIL', 'DET'), ('BOS', 'IND'), ('PHI', 'BKN'), ('TOR', 'ORL'), ('GSW', 'LAC'), ('HOU', 'UTA'), ('POR', 'OKC'),
              ('DEN', 'SAS'), ('MIL', 'BOS'), ('PHI', 'TOR'), ('GSW', 'HOU'), ('POR', 'DEN'), ('MIL', 'TOR'), ('GSW', 'POR'),
              ('TOR', 'GSW')], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]]


# Get testing datasets
#======================================================================================================================
# Input: a training dataframe of the NBA season you want (e.g. train18) to calculate average performance of each team
# in the given season base on the performance of preseason and regular season games (which is the training datasets)
# Return: average performance of each team when they play at home, as visitor and total average
def get_team_avg(df):
    team_home_avg = pd.DataFrame(columns= ['TEAM', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB'])
    team_away_avg = pd.DataFrame(columns= ['TEAM', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB'])
    teams = np.unique(df[['HOME_TEAM_ID', 'VISITOR_TEAM_ID']].values)
    for t in teams:
        home_team_stat = df.loc[df['HOME_TEAM_ID'] == t]
        away_team_stat = df.loc[df['VISITOR_TEAM_ID'] == t]
        team_home_avg = team_home_avg.append(
            pd.Series([t, home_team_stat['PTS_home'].mean(), home_team_stat['FG_PCT_home'].mean(),
                       home_team_stat['FT_PCT_home'].mean(), home_team_stat['FG3_PCT_home'].mean(),
                       home_team_stat['AST_home'].mean(),
                       home_team_stat['REB_home'].mean()], index=team_home_avg.columns), ignore_index=True)
        team_away_avg = team_away_avg.append(
            pd.Series([t, away_team_stat['PTS_away'].mean(), away_team_stat['FG_PCT_away'].mean(),
                       away_team_stat['FT_PCT_away'].mean(), away_team_stat['FG3_PCT_away'].mean(),
                       away_team_stat['AST_away'].mean(),
                       away_team_stat['REB_away'].mean()], index=team_away_avg.columns), ignore_index=True)
    team_home_avg = team_home_avg.set_index('TEAM')
    team_away_avg = team_away_avg.set_index('TEAM')
    team_avg = team_home_avg.add(team_away_avg, axis=0) / 2
    return team_home_avg, team_away_avg, team_avg

# Input: team_home_avg df, team_away_avg and a df which can contain multiple games or a single game
# Return: testX if you want to predict a single game instead of bracket of games
def testX_single(team_home_avg, team_away_avg, test_df):
    tX = pd.DataFrame(columns= ['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB'])
    for index, row in test_df.iterrows():
        diff = team_home_avg.loc[row['HOME_TEAM_ID'], :] - team_away_avg.loc[row['VISITOR_TEAM_ID'], :]
        tX = tX.append(diff, ignore_index=True)
    return tX

# input: a team_avg table and a bracket
# return: testX for bracket of games
def testX_bracket(team_avg, bracket):
    x = bracket[0]
    tX = pd.DataFrame(columns=['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB'])
    for (top, bottom) in x:
        diff = team_avg.loc[top, :] - team_avg.loc[bottom, :]
        tX = tX.append(diff, ignore_index=True)
    return tX


# Get training datasets
#========================================================================================================================
# stats_difference = home stats - visitor stats
def stats_difference(df):
    home_stats = df.iloc[:, 4:10].rename(columns={'PTS_home':'PTS', 'FG_PCT_home':'FG_PCT', 'FT_PCT_home':'FT_PCT',
                                                  'FG3_PCT_home':'FG3_PCT', 'AST_home':'AST', 'REB_home':'REB'})

    visitor_stats = df.iloc[:, 10:-1].rename(columns={'PTS_away':'PTS', 'FG_PCT_away':'FG_PCT', 'FT_PCT_away':'FT_PCT',
                                                      'FG3_PCT_away':'FG3_PCT', 'AST_away':'AST', 'REB_away':'REB'})
    diff_df = home_stats - visitor_stats
    return diff_df


# Get y for training or testing dataset
def get_y(df):
    return df.HOME_TEAM_WINS


def test_summary(py, y):
    y = np.array(y)
    loss = 0
    total_obs = len(py)
    for i in range(len(py)):
        if py[i] != y[i]:
            loss += 1
    return total_obs, loss, 1 - loss/total_obs

# Start training process
#=======================================================================================================================
# Logistic Model
logreg = LogisticRegression(solver='liblinear')
X = stats_difference(train_dataset)
y = get_y(train_dataset)

logreg.fit(X, y)


# Testing
# =========================================================================================================================
def test_all_bracket(brackets, trainings):
    season = 14
    average_accuracy = 0
    total_games = len(brackets) * len(brackets[0][0])
    for i in range(len(brackets)):
        team_home_avg, team_away_avg, team_avg = get_team_avg(trainings[i])
        t_y = brackets[i][1]
        t_X = testX_bracket(team_avg, brackets[i])
        predict_y = logreg.predict(t_X)
        total_obs, loss, accuracy = test_summary(predict_y, t_y)
        average_accuracy += accuracy
        print('Season {}: total_obs = {}, loss = {}, accuracy = {}'.format(season, total_obs, loss, accuracy))
        season += 1
    average_accuracy = average_accuracy/len(brackets)
    print('Total_games = {} \nAverage_accuracy = {}'.format(total_games, average_accuracy))

all_brackets = [bracket14, bracket15, bracket16, bracket17, bracket18]
all_trainings = [train14, train15, train16, train17, train18]
test_all_bracket(all_brackets, all_trainings)

