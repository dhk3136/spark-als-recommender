#!/usr/bin/env python

import sys
import numpy as np
from surprise import AlgoBase, Dataset, evaluate

class GlobalMean(AlgoBase):
    def train(self, trainset):

        AlgoBase.train(self, trainset)

        # Computing the average rating
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

    def estimate(self, u, i):

        return self.the_mean


class MeanofMeans(AlgoBase):
    def train(self, trainset):

        # calling base method first
        AlgoBase.train(self, trainset)

        users = np.array([u for (u, _, _) in self.trainset.all_ratings()])
        items = np.array([i for (_, i, _) in self.trainset.all_ratings()])
        ratings = np.array([r for (_, _, r) in self.trainset.all_ratings()])

        user_means,item_means = {},{}
        for user in np.unique(users):
            user_means[user] = ratings[users==user].mean()
        for item in np.unique(items):
            item_means[item] = ratings[items==item].mean()

        self.global_mean = ratings.mean()    
        self.user_means = user_means
        self.item_means = item_means
                            
    # returning 'mean of means'
    def estimate(self, u, i):
       
        if u not in self.user_means:
            return(np.mean([self.global_mean,
                            self.item_means[i]]))

        if i not in self.item_means:
            return(np.mean([self.global_mean,
                            self.user_means[u]]))

        return(np.mean([self.global_mean,
                        self.user_means[u],
                        self.item_means[i]]))


if __name__ == "__main__":

    data = Dataset.load_builtin('ml-100k')
    print("\nGlobal Mean...")
    algo = GlobalMean()
    evaluate(algo, data)

    print("\nMeanOfMeans...")
    algo = MeanofMeans()
    evaluate(algo, data)
