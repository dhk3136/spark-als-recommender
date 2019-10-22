import pandas as pd
import numpy as np
from databricks import koalas as ks



def create_user_rating_df(ratings):
    new_df = pd.DataFrame()
    user_id_list = ratings['userId'].unique().tolist()
    for idx,user in enumerate(user_id_list):
        print(f"Building user {idx+1}/{len(user_id_list)}")
        temp_info = dict()
        temp_info['user'] = user
        df = ratings[ratings['userId'] == user].copy()
        movies = df['movieId'].tolist()
        movie_ratings = df['rating'].tolist()
        for m,r in list(zip(movies, movie_ratings)):
            temp_info[m] = r
        temp_df = pd.DataFrame(columns= movies, index= [idx])
        temp_df.loc[idx] = pd.Series(temp_info)
        new_df = pd.concat([new_df, temp_df], axis= 0, join= 'outer', sort= False, copy= False)
    return new_df


if __name__ == '__main__':
    
    # load in split data
    full_data = pd.read_csv('data/movies/ratings.csv', index_col= 0).drop('timestamp', axis= 1).reset_index()
#     ratings_train = pd.read_csv('data/movies/ratings_train.csv', index_col= 0)
#     ratings_test = pd.read_csv('data/movies/ratings_test.csv', index_col= 0)
    
    # create embeddings matrix
#     train_matrix = create_user_rating_df(ratings_train)
#     test_matrix = create_user_rating_df(ratings_test)
    full_matrix = create_user_rating_df(full_data)
    
    # save embeddings matrix
#     train_matrix.to_csv('data/movies/train_matrix.csv')
#     test_matrix.to_csv('data/movies/test_matrix.csv')
    full_matrix.to_csv('data/movies/full_matrix.csv')
    
    exit()