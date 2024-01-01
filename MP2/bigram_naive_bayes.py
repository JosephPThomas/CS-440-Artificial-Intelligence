# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
import numpy as np
from collections import Counter
from collections import defaultdict


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=0.999, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
        # Initialize the list to store predictions
    # Calculate unigram and bigram probabilities with Laplace smoothing
    if bigram_lambda == 1.0:
        bigram_lambda = 0.999
    
    positive_unigram_prob, positive_unigram_calc_unknprob = calculate_unigram_probability(train_set, train_labels, 1, unigram_laplace)
    negative_unigram_prob, negative_unigram_calc_unknprob = calculate_unigram_probability(train_set, train_labels, 0, unigram_laplace)
    
    positive_bigram_prob, positive_bigram_calc_unknprob = calculate_bigram_probability(train_set, train_labels, 1, bigram_laplace)
    negative_bigram_prob, negative_bigram_calc_unknprob = calculate_bigram_probability(train_set, train_labels, 0, bigram_laplace)

    y_hats = []

    for doc in tqdm(dev_set, disable=silently):
        positive_likelyhood = np.log(pos_prior)  # Log likelihood for positive
        negative_likelyhood = np.log(1 - pos_prior)  # Log likelihood for negative

        # Calculate unigram probabilities
        for word in doc:
            # Calculate log likelihood for positive
            if word in positive_unigram_prob:
                positive_likelyhood += np.log(positive_unigram_prob[word])
            else:
                positive_likelyhood += np.log(positive_unigram_calc_unknprob)

            # Calculate log likelihood for negative
            if word in negative_unigram_prob:
                negative_likelyhood += np.log(negative_unigram_prob[word])
            else:
                negative_likelyhood += np.log(negative_unigram_calc_unknprob)

        # Calculate bigram probabilities
        for i in range(len(doc) - 1):
            bigram = f"{doc[i]} {doc[i+1]}"
            
            # Calculate log likelihood for positive using bigram
            if bigram in positive_bigram_prob:
                positive_likelyhood += np.log(bigram_lambda * positive_bigram_prob[bigram] + (1.0 - bigram_lambda) * positive_unigram_prob.get(doc[i], positive_unigram_calc_unknprob))
            else:
                positive_likelyhood += np.log((1.0 - bigram_lambda) * positive_unigram_prob.get(doc[i], positive_unigram_calc_unknprob))
            
            # Calculate log likelihood for negative using bigram
            if bigram in negative_bigram_prob:
                negative_likelyhood += np.log(bigram_lambda * negative_bigram_prob[bigram] + (1.0 - bigram_lambda) * negative_unigram_prob.get(doc[i], negative_unigram_calc_unknprob))
            else:
                negative_likelyhood += np.log((1.0 - bigram_lambda) * negative_unigram_prob.get(doc[i], negative_unigram_calc_unknprob))

        # If positive likelihood is greater than negative likelihood, 1 is added to the list
        if positive_likelyhood > negative_likelyhood:
            y_hats.append(1)
        # If negative likelihood is greater than positive likelihood, 0 is added to the list
        else:
            y_hats.append(0)
    
    return y_hats

def calculate_unigram_probability(train_set, train_labels, pn_label, laplace_smoothing):
    w_tab = defaultdict(int)#dictionary for unigram word count
    total = 0#total unigram word count
    #for loop to loop through training data and labels
    for i in range(len(train_labels)):
        if train_labels[i] == pn_label:
            for word in train_set[i]:
                #Counting occurence of each word
                w_tab[word] += 1
                total += 1

    dict_wprob = {} #Dictionary for word probability
    total_types = len(w_tab)#Unique word count
    #Calculates the word probability using the laplace smoothing

    for word in w_tab:
        prob = (w_tab[word] + laplace_smoothing) / (total + laplace_smoothing * total_types)
        dict_wprob[word] = prob

    calc_unknprob = laplace_smoothing / (total + laplace_smoothing * total_types)

    return dict_wprob, calc_unknprob

def calculate_bigram_probability(train_set, train_labels, pn_label, laplace_smoothing):
    bigram_tab = defaultdict(int)#dictionary for bigram word count
    total_bigrams = 0#total bigram word count
    #for loop to loop through training data and labels and add bigram values to dictionary
    for i in range(len(train_labels)):
        if train_labels[i] == pn_label:
            for j in range(len(train_set[i]) - 1):
                bigram = f"{train_set[i][j]} {train_set[i][j+1]}"
                bigram_tab[bigram] += 1
                total_bigrams += 1

    dict_bigram_wprob = {}#Dictionary for bigram word probability
    total_bigram_types = len(bigram_tab)
    #Calculates the bigram word probability using the laplace smoothing
    for bigram in bigram_tab:
        prob = (bigram_tab[bigram] + laplace_smoothing) / (total_bigrams + laplace_smoothing * total_bigram_types)
        dict_bigram_wprob[bigram] = prob

    calc_unknprob = laplace_smoothing / (total_bigrams + laplace_smoothing * total_bigram_types)

    return dict_bigram_wprob, calc_unknprob



