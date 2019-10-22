# Movie Recommender Case Study

### Daniel Kim

<img src="images/grouplens.png" /> 

## The data
MovieLens is a recommender dataset created by GroupLens of the University of Minnesota. The data is composed of over 27 Million ratings from 280,000 users. Reviews of 58,000 movies include a numerical score and may include free-text entries from users as tags. For this case-study, a redacted portion of the dataset was used for development of the recommender.

The reduced dataset consists of 100,000 user reviews from 671 user on over 9,000 movies. While substantial, this data represents a very sparse matrix, with only 1.64% data density in the original data and 1.42% and 0.61% density in the train and test splits, respectively. As a result, a recommender on this system will need to accommodate the sparsity of data--which eliminates several recommender approaches. 

Within the data, there is a non-normal distribution of of ratings of movies, with the majority of ratings falling between 3 - 4 on a scale of 1 - 5 (worst - best). Additionally, an increment of 0.5 within ratings provide a discrete distribution.

<img src="images/mean_rating_movie.png" width= "600" /> 

Surprisingly, very few movies receive a large number of very-low or very-high ratings.

Most users provided fewer than 50 movie ratings which contributes to why the matrix is so sparse. As seen in the below distribution plot, there is a concentration of users who rated 200 or few movies with a much smaller population of users that that rated 200+ movies.

<img src="images/user_rating_count2.png" width= "600" /> 

When looking at the average rating given by user, the distribution is fairly normal, showing some left skew, meaning users are more likely to provide favorable ratings on average.

<img src="images/user_rating_avg.png" width= "600" /> 


## Approach
--------------
Given the nature of the dataset, two different approaches could be used for a recommender. A content-based recommendation system could be used to recommend movies based on similar features. Movie information and user-entered tags allow for a large corpus that could be used in latent feature analysis. Recommendations would be based on similarities between movies based on latent features, and correlated to user reviews of movies--in order to understand user preferences.

<img src="images/content_based_filtering.png" width= "300" height= "300" /> 


