# Processing the NBA data set obtained from https://www.kaggle.com/nathanlauga/nba-games/data

import pandas as pd
from datetime import date

# Processing Team Identification Data
team_df = pd.read_csv('./teamstats.csv')
team_ids = {}


