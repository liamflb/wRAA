# xwoba
Mimicking xwOBA

My goal is to create my own xwOBA statistic using an algorithm with exit velocity, launch angle, and sprint speed. I used batted ball data from Statcast, scraped with pybaseball, and used FanGraphs for wOBA constants and information. I was inspired to try this myself by this article: https://medium.com/@thomasjamesnestico/modelling-xwoba-with-knn-9b004e93861a. I decided to first recreate xwOBA with K-Nearest Neighbors, similar to what MLB does for their stat. 

I've developed the KNN model using only exit velocity and launch angle, and written a function to look up an individual player's xwOBA. My next goal is to implement sprint speed for softly hit ground balls that faster players are able to turn into hits. I'd also like to try a random forest classifier and possibly other models. Initial R^2 = 0.89 (between my xwoba and mlb xwoba)



