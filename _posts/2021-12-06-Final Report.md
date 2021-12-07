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

Our goal is to create a classifier that determines whether a movie will be successful based on its features. We believe that this classifier could help drive the movie making and casting processes and set up a movie for success before it is even filmed. 

## Data Collection and Feature Engineering

Our dataset consisted of approximately 5,000 movies from a TMDB’s Kaggle dataset. The data was initially stored in two different datasets: (i) Movies (ii) Credits. Actors of each movie were stored in a string literal located in the ‘cast’ column of the Movies dataset. These string literals were formatted as a list of dictionaries that could be easily converted into a separate pandas data frame for each movie. This allowed our team to extract the top from each movie, along with any actor that had more than 20 occurrences throughout the entire dataset. Similar methods were used to also pull directors, production companies, and genres. 

The final data frame used was constructed by one-hot-encoding our extracted movie features. This matrix listed each movie as row values and each extracted 3 actors, director, production company, and genre as column values. For any given row, a value of 1 listed under a column signifies that a specific movie attribute does relate to that movie. For example, the movie “Avatar” would contain a value of 1 under the column “actor_Sigourney_Weaver” because Sigourney Weaver stars in that movie. However, a value of 0 would be listed under the column “director_Quentin_Tarantino” because Quentin Tarantino did not direct that specific movie. 

As far as dealing with missing values, the “budget” and “runtime” features were zero for about 20% of the total observations. We were able to cross reference the IMDB API and get the budget and runtime for approximately 50% of the missing data and ended up dropping the remaining observations where “budget” and “runtime” were null.  

We inspected the distributions of these two numeric fields to identify potential outliers and then scaled the data between zero and one before running PCA. Lastly, we converted our target variable “vote average” (explained in Method section below) from a continuous variable to a binary variable by setting the threshold at 6.5 on the scale of 0 to 10. 


## Method

We first split the data into training (80%), validation (10%), and test data (10%). We then used Principal Component Analysis (“PCA”; unsupervised method) to reduce the dimensionality of our features. We toggled the number of principle components used based on retaining 80% of the variance. 

Next, we built the following supervised models to classify movies as successful or not successful; our target variable in all these cases was vote average rating. Below we describe the models used and method for tuning the parameters 

1. Random Forest - We experimented with different combinations of values to optimize the following hyperparameters for this tree model: 

   * n_estimators = 100 

     * The number of trees in the forest 

   * max_depth = 20 

     * The maximum depth of a tree 

   * max_features = 100 

     * The number of features to consider when finding the optimal split 

2. XG Boost - We experimented with different combinations of values to optimize the following hyperparameters of this ensemble method: 

   * reg_lambda = 0.05 

     * L2 regularization term on weights 

   * subsample = 0.8 

     * Ratio of training instance subsamples 

   * learning_rate = 0.1 

     * Weighting factor for the corrections by new trees when added to the model 

   * max_depth = 7 

3. LightGBM - We used a grid search to optimize the following hyperparameters for this next ensemble method: 

   * reg_alpha = 0.03 

     * L1 regularization term on weights 

   * num_leaves = 80 

     * Maximum tree leaves for base learners 

   * min_child_samples = 50 

     * Minimum number of data points required in a leaf 

   * learning_rate = 0.05 

     * max_depth = 20 

3. Neural Network - We built a 2-layer Neural network with two dense layers, 110 neurons (1st layer) and 30 neurons (2nd layer). We used the “Relu” activation function. We used Dropout layers to mitigate the effect of overfitting. We tuned hyper parameters by running a for loop and finding the highest accuracy scores given by each parameter. 

We then chose the best supervised model from the above methods based on performance on the validation dataset, and calculated final performance based on the test dataset.  

Our next step was to cluster our dataset so that we could group movies with similar components. We identified actors and production companies which performed better than the average movies in their respective cluster. For each cluster, we observed the percentage difference the best actors made to the cluster’s mean vote average to inform our movie casting decisions. 

This clustering analysis was performed on the entire dataset and using the PCA components (80% variance retained). We used the elbow method to pick the optimal number of clusters (10) in our K-Means algorithm. 

## Results and Discussions

### Dimensionality Reduction

The PCA analysis reduced the dimensionality of our dataset from 9613 columns to 1015 columns. From the line chart below, we could see the relationship between the number of components taken and the explained variance. 
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

The random forest classification model yielded a ROC-AUC score of 0.754, which is plotted below. This means that our model has a 75.4% chance of classifying successful movies as successful. As we look further into the confusion matrix, we determine that the model’s accuracy is 0.707 and the precision is 0.671. 

When we ran the XG Boost model, we improved the ROC-AUC score to 0.783, which is plotted below. This improved the classification of successful movies by about 5%. The model’s accuracy is 0.721 and the precision is 0.642; while accuracy improves by about 1.5%, precision 

When we ran the Light GBM model, we improved the ROC-AUC score to 0.77, which is plotted below. This did not improve upon the XG Boost model classification. The accuracy is 0.721 and precision is 0.652, which are comparable to those of the XG Boost model. 

When we executed the Neural Network, we improved the ROC-AUC score to 0.784, which is plotted below. With an accuracy of 0.709 and precision of 0.738, we selected this model as our final model since it performed the best on the validation dataset.  

Finally, we ran the neural network model on our test dataset, which yielded a ROC-AUC score of 0.812 on the test dataset.  

Based on the analysis above, the ensemble methods improved upon the initial Random Forest classification model. However, the Neural Network proved to be the best at classifying successful movies as successful. 

### Unsupervised Model 

#### Clustering

As we investigated each cluster, we determined that they were approximately segregated by movie genre. We calculated the following metrics to characterize a cluster: 

   * Cluster_perc: percentage of movies in a cluster that fall within a genre 

   * Total_perc: total percentage of a genre within the original dataset 

   * Compare: ratio of cluster_perc to total_perc 

   * Cluster_in_total_perc: ratio of the occurrences of a genre in a cluster to the total occurrence of that genre within the original dataset 



To analyze the actors that appear in a cluster, we calculated the following metrics: 

   * Actor_vote_avg: average vote average for the movies in which an actor was present in a cluster 

   * Cluster_vote_avg: total average vote average of all movies in a cluster 

   * Vote_avg_diff_pct: percentage difference between actor_vote_avg and cluster_vote_avg 



From this example, if we were to make a new science fiction movie, we could cast Sigourney Weaver because her science fiction movies perform better on average. If Weaver cannot be cast, Arnold Schwarzenegger would be the next ideal choice. 

To further test our model, we ran 29 movies from 2017 through our Neural Network model. Our dataset involved movies up to 2016, so we chose 2017 to simulate predicting the success of new movies. The model predicted 72% of the movies correctly, which is displayed in the tables below. For each movie, the tables show the ground truths for vote average and the associated binary classifier. The column “Predict Prob” shows the derived likelihood that the movie is successful. Finally, we used a probability threshold of 0.50 to classify our predictions as successful or unsuccessful. 





## References

1. Davies, Aran. “How Big Data Helped Netflix Series House of Cards Become a Blockbuster?” Sofy.tv - Blog, 6 Sept. 2019, https://sofy.tv/blog/big-data-helped-netflix-series-house-cards-become-blockbuster/
2. Brueggemann, Tom. “'Blade Runner' Box Office Deja Vu AS '2049' Starring Ryan Gosling Falls Short.” IndieWire, IndieWire, 8 Oct. 2017, https://www.indiewire.com/2017/10/box-office-blade-runner-2049-ryan-gosling-1201884827/
3. Jolliffe, Ian T, and Jorge Cadima. “Principal component analysis: a review and recent developments.” Philosophical transactions. Series A, Mathematical, physical, and engineering sciences vol. 374,2065 (2016): 20150202. doi:10.1098/rsta.2015.0202 
