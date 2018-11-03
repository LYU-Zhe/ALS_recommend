#coding: utf-8
import sys
import pickle
import time
import itertools
import csv
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
import json
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parseRating(line):
    """
        userID ,movieID, rating, timestamp
    """
    fields = line.strip().split(",")
    return(int(fields[0]), int(fields[1]), float(fields[2]))


def computeRmse(model, data, n):
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

if __name__ == "__main__":

    #configuration
    conf = SparkConf().setAppName("SeriesRecommendALS")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    #load file an convert it to RDD
    ratings = sc.textFile("hdfs://mycluster/user/zhangyang8/ratings_movie.csv").map(parseRating)
    list_rating = ratings.map(lambda x: x[2]).collect()
    print type(list_rating)
    print len(list_rating)
    plt.hist(list_rating, bins=5, edgecolor='k', facecolor='red')
    plt.savefig('movie_rating_distribution.png')    
    numRatings = ratings.count()
    numUsers = ratings.map(lambda r: r[0]).distinct().count()
    numMovies = ratings.map(lambda r: r[1]).distinct().count()

    print "Got %d ratings from %d users on %d series." % (numRatings, numUsers, numMovies)

    #training:validation:test=6:2:2
    (training, validation, test) = ratings.randomSplit([0.6,0.2,0.2])
    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)

    #iterate parameter to find the best model
    ranks = [10]#[8, 10, 12, 15]
    lambdas = [0.001, 0.01, 0.1, 0.5, 1.0]
    numIters = [30]#[10, 20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1
    start_train = time.time()
    for rk, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(ratings=training,rank=rk,iterations=numIter,lambda_=lmbda)
        validationRmse = computeRmse(model, validation, numValidation)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %f, and numIter = %d." % (rk, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rk
            bestLambda = lmbda
            bestNumIter = numIter
    end_train = time.time()
    print "train time is: %fs" % (end_train-start_train)
    testRmse = computeRmse(bestModel, test, numTest)

    #print the parameters of the best model
    print "The best model was trained with rank = %d and lambda = %.2f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)

    #print the rmse improvement
    meanRating = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print "The best model improves the baseline by %.2f" % (improvement) + "%."
   
    #predict for all users 
    start_predict = time.time() 
    print "length of test: %d" % len(test.collect())
    predictions_test = bestModel.predictAll(test.map(lambda x: (x[0], x[1])))
    end_predict = time.time()
    print "predict time for all test users: %fs" % (end_predict-start_predict)
    predictionsAndRatings = predictions_test.map(lambda x: ((x[0], x[1]), round(x[2],2))).join(test.map(lambda x: ((x[0], x[1]), x[2])))
    print predictionsAndRatings.collect()[:10]

    sc.stop()
