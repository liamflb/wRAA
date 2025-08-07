# xwoba
Mimicking xwOBA

My goal was to create my own xwOBA statistic using an algorithm with exit velocity and launch angle, trained against total bases. I used batted ball data from Statcast, scraped with pybaseball, and used FanGraphs for wOBA constants and information. I was inspired to try this myself by this article: https://medium.com/@thomasjamesnestico/modelling-xwoba-with-knn-9b004e93861a. I decided to first recreate xwOBA with K-Nearest Neighbors, similar to what MLB does for their stat. 

I've developed the KNN model using exit velocity and launch angle, and written a function to look up an individual player's xwOBA. I trained on a random sample of data from the 2022-2024 seasons. I settled on a KNN model with 15 neighbors after selecting using a grid search. My next goal is to implement sprint speed for softly hit ground balls that faster players are able to turn into hits, somethign which MLB includes in their own calculation. My initial R^2 = 0.98 for qualified hitters in the 2024 season, between my xwOBA and MLB's xwOBA, although my metric is consistently greater than MLB's, with a mean of .320 as opposed to MLB's .312. Becuase of this fact, I thought that it would be better to represent my statistic as Weighted Runs Above Average, which is essentially my xWOBA scaled by plate appearances and the league average, giving it an average of 0 for all plate appearances, and an average of about 3 for qualified hitters in the 2024 season.


