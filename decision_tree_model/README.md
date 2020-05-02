Focusing on 5 NBA seasons:

* 2014 - 2015
* 2015 - 2016
* 2016 - 2017
* 2017 - 2018
* 2018 - 2019

The training data was created by subtracting the away team averages from the home team averages in the following categories (left to right):
FG, FGA, FG%, 3P, 3PA, 3P%, 2P, 2PA, 2P%, FT, FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, W

This was done for every game in those respective seasons.

The testing data was made by observing playoff match ups for each of these seasons.

The notation of `top_team` refers to the home team in the training data, and in the testing data it refers to the team with the lower seed in the playoffs of that season.

The results come from using the C4.5 Decision Tree algorithm through the use of the Weka Learn software.
