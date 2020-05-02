import pandas as pd

playoffs_2014 = pd.read_csv('../datasets/2014_2015_playoffs.csv')
playoffs_2015 = pd.read_csv('../datasets/2015_2016_playoffs.csv')
playoffs_2016 = pd.read_csv('../datasets/2016_2017_playoffs.csv')
playoffs_2017 = pd.read_csv('../datasets/2017_2018_playoffs.csv')
playoffs_2018 = pd.read_csv('../datasets/2018_2019_playoffs.csv')

stats_df = pd.read_csv('../datasets/teamstats.csv') # Obtained from Basketball Reference
categories = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W']

# Removing asterisks from end of team names in the stats data frame
for index, row in stats_df.iterrows():
    if row['Team'].endswith('*'):
        stats_df.at[index, 'Team'] = row['Team'][:-1]

# Getting 2014-2015 stats
stats_mask = (stats_df['Year'] == 2014)
stats_2014 = stats_df.loc[stats_mask]
# Getting 2015-2016 stats
stats_mask = (stats_df['Year'] == 2015)
stats_2015 = stats_df.loc[stats_mask]
# Getting 2016-2017 stats
stats_mask = (stats_df['Year'] == 2016)
stats_2016 = stats_df.loc[stats_mask]
# Getting 2017-2018 stats
stats_mask = (stats_df['Year'] == 2017)
stats_2017 = stats_df.loc[stats_mask]
# Getting 2018-2019 stats
stats_mask = (stats_df['Year'] == 2018)
stats_2018 = stats_df.loc[stats_mask]

def get_processed_test_dataframe(playoff_games, stats):
    processed = []
    for index, row in playoff_games.iterrows():
        top_team = row['Top_Team']
        bottom_team = row['Bottom_Team']
        top_team_stats = stats.loc[stats['Team'] == top_team]
        bottom_team_stats = stats.loc[stats['Team'] == bottom_team]
        temp = {}

        # Getting differences in each of the desired categories
        for category in categories:
            temp[category] = round(top_team_stats[category].iloc[0] - bottom_team_stats[category].iloc[0], 2)

        if row['Winner'] == top_team:
            temp['Top_Team_Won'] = True
        else: 
            temp['Top_Team_Won'] = False

        processed.append(temp)

    df = pd.DataFrame(processed)
    return df

processed_2014 = get_processed_test_dataframe(playoffs_2014, stats_2014)
processed_2015 = get_processed_test_dataframe(playoffs_2015, stats_2015)
processed_2016 = get_processed_test_dataframe(playoffs_2016, stats_2016)
processed_2017 = get_processed_test_dataframe(playoffs_2017, stats_2017)
processed_2018 = get_processed_test_dataframe(playoffs_2018, stats_2018)

processed_2014.to_csv('2014_2015_testing_data.csv', encoding='utf-8', index=False)
processed_2015.to_csv('2015_2016_testing_data.csv', encoding='utf-8', index=False)
processed_2016.to_csv('2016_2017_testing_data.csv', encoding='utf-8', index=False)
processed_2017.to_csv('2017_2018_testing_data.csv', encoding='utf-8', index=False)
processed_2018.to_csv('2018_2019_testing_data.csv', encoding='utf-8', index=False)
