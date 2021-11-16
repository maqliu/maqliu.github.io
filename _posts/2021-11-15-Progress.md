# Predicting Movie Success using Analytics

## Team Members
Rochisman Datta, Joe Palumbo, Parisa Davoodi, Manqiu Liu, Nicholas Gonzalez

<!-- TABLE OF CONTENTS -->
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#background">Background</a></li>
    <li><a href="#problem-definition">Problem Definition</a></li>
    <li><a href="#data-collection-feature-engineering">Data Collection & Feature Engineering</a></li>
    <li><a href="#method">Method</a></li>
    <li><a href="#Results-and-Discussions">Results and Discussions</a>
        <ul>
          <li><a href="#Dimensionality-Reduction">Dimensionality Reduction</a></li>
          <li><a href="#supervised-model">Supervised Model</a></li>
      </ul>
    </li>
    <li><a href="#proposal-timeline">Proposal Timeline</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Background

Companies in the content distribution and creation space are always on the lookout for the next big hit. What if we told you that House of Cards, which made Netflix a household name, was informed by data analytics? Data suggested the pairing of David Fincher (director) and Kevin Spacey (actor) would “bring in big audiences”[<sup>1</sup>](https://sofy.tv/blog/big-data-helped-netflix-series-house-cards-become-blockbuster/).  

A model which could predict the probability of a successful tv show/movie/game would be invaluable to any content creation company. 

## Problem Definition

The choice of movie components – director, genre, cast, run-time – determine the success of a movie. An improper combination of these factors could lead to box office flops.  Despite its highly anticipated release, why did Blade Runner 2049 fall short of public expectations[<sup>2</sup>](https://www.indiewire.com/2017/10/box-office-blade-runner-2049-ryan-gosling-1201884827/)?Would casting Harry Styles as the next James Bond be a recipe for disaster or a worthwhile endeavor?

We believe that a combination of features could be manipulated to set up a movie up for success in the  box office  - these combinations would help drive the movie making process. Our goal is to create a classifier that determines whether a movie will be successful based on its features.

## Data Collection Feature Engineering
Our dataset consisted of approximately 5,000 movies from IMDB’s API. The data was initially stored in two different datasets: (i) Movies (ii) Credits. Actors of each movie were stored in a string literal located in the ‘cast’ column of the Movies dataset. These string literals were formatted as a list of dictionaries that could be easily converted into a separate pandas data frame for each movie. This allowed our team to extract the top 5 actors from each movie, along with any actor that had more than 20 occurrences throughout the entire dataset. Similar methods were used to also pull directors, production companies, and genres.

The final data frame used was constructed by one-hot-encoding our extracted movie features. This matrix listed each movie as row values and each extracted actor, director, production company, and genre as column values. For any given row, a value of 1 listed under a column signifies that a specific movie attribute does relate to that movie. For example, the movie “Avatar” would contain a value of 1 under the column “actor_Sigourney_Weaver” because Sigourney Weaver stars in that movie. However, a value of 0 would be listed under the column “director_Quentin_Tarantino” because Quentin Tarantino did not direct that specific movie.

As far as dealing with missing values, the “budget” and “runtime” features were zero for about 4% of the total observations. We imputed the mean of each feature to populate these values. We inspected the distributions of these two numeric fields to identify potential outliers. We then normalized these features before running PCA. Lastly, we converted our target variable “vote average” (explained in Method section below) from a continuous variable to a binary variable by setting the threshold at 6 on the scale of 0 to 10. 


## Method

We first split the data into training (90%) and test data (10%). We then used Principal Component Analysis (unsupervised) to reduce the dimensionality of our features. We toggled the number of principle components used based on retaining 75% of the variance. 

Next, we built a supervised random forest classification model to classify movies as successful or not successful. We tested maximum tree depth values from 15 to 25 and determined that 20 yielded the best results. Our target variable in this case was a vote average rating from IMDB, bucketed into a categorical variable. 
To explore the predictive capability of our dataset, we also built a LightGBM classification model to see how much our prediction could improve when using an ensemble of weak learners to fortify the accuracy. We intend to discuss the improvement and limitation by comparing to our random forest model, and to determine our next step in feature engineering.

We are still considering running a K-Means or Gaussian Mixture Model clustering algorithm to form clusters of similar movies to explore any underlying patterns. Potentially, we will run a new random forest classification model on each of these clusters instead of having one model for the entire dataset.


## Results and Discussions

### Dimensionality Reduction

The PCA analysis reduced the dimensionality of our dataset from 13303 columns to 1299 columns. For the clustering aspect, we anticipate that the unsupervised models will output clusters that we could analyze to glean similarities. We expect to get clusters with movies that are obviously similar (i.e., common actors, genres), but also other clusters that lack clear commonalities and require further exploration.
From the line chart below, we could see the relationship between number of components taken and the explained variance.

<figure>
  <p align="center">
    <kbd>
      <img src="https://github.com/maqliu/maqliu.github.io-lucky13/blob/master/images/PCA%20New.png?raw=true" width="700"/>
    </kbd>
  </p>
  <p align="center">PCA Explained Variance Cumulative Sum Curve</p>
</figure>

### Supervised Model 

The random forest classification model yielded a ROC-AUC score of 0.644, which is plotted below. This means that our model has a 64.4% chance of classifying successful movies as successful. We look further into the confusion matrix, and we find the accuracy is 0.630 and the precision is 0.638. 

<figure>
  <p align="center">
    <kbd>
      <img src="https://github.com/maqliu/maqliu.github.io-lucky13/blob/master/images/rf.png?raw=true" width="400"/>
    </kbd>
  </p>
  <p align="center">Random Forest Model ROC Curve</p>
</figure>

When we ran the Light GBM model, we improved ROC-AUC score to 0.754, which is plotted below. This improved the chance of classifying successful movies as successful by about 10%. The accuracy is 0.704 and precision is 0.703, which improved by about 7% when compared to the random forest model.

<figure>
  <p align="center">
    <kbd>
      <img src="https://github.com/maqliu/maqliu.github.io-lucky13/blob/master/images/lgbm.png?raw=true" width="400"/>
    </kbd>
  </p>
  <p align="center">LightGBM Model ROC Curve</p>
</figure>

Based on the analysis above, the ensemble model proved to be better at classifying successful movies as successful. As we approach our final report, we plan to continue tuning the hyperparameter of the random forest classification model. Upon development of several supervised methods, we will use cross-validation to compare the performance across models. Finally, we will implement clustering methods for our unsupervised model.



## Proposal Timeline

- Video for Proposal – Oct 3rd 

  - PowerPoint – Parisa, Nick 

  - Voice-over – Joe and Manqiu 

  - Video Editing – Rochisman  

- Submit proposal – Oct 7th 

  - Data Cleaning (Everyone) – Oct 15th  

  - Feature Engineering (Parisa, Manqiu) – Oct 22nd 

  - Model Creation and Implementation (Joe, Rochisman, Nick) – Oct 29th 

  - First Draft of Mid-term report (Everyone)– Nov 7th  

  - Final Draft of Mid-term report (Everyone)– Nov 14th  

- Mid-point report – Nov 16th 

  - Tweaking and optimizing the models (Parisa, Nick, Joe) – Nov 23rd   

  - Drawing conclusions based on findings (Rochisman Manqiu) – Dec 3rd  

  - First draft of final report (Everyone) – Dec 5th  

- Final Project – Dec 7th  



## References

1. Davies, Aran. “How Big Data Helped Netflix Series House of Cards Become a Blockbuster?” Sofy.tv - Blog, 6 Sept. 2019, https://sofy.tv/blog/big-data-helped-netflix-series-house-cards-become-blockbuster/
2. Brueggemann, Tom. “'Blade Runner' Box Office Deja Vu AS '2049' Starring Ryan Gosling Falls Short.” IndieWire, IndieWire, 8 Oct. 2017, https://www.indiewire.com/2017/10/box-office-blade-runner-2049-ryan-gosling-1201884827/
3. Jolliffe, Ian T, and Jorge Cadima. “Principal component analysis: a review and recent developments.” Philosophical transactions. Series A, Mathematical, physical, and engineering sciences vol. 374,2065 (2016): 20150202. doi:10.1098/rsta.2015.0202 
