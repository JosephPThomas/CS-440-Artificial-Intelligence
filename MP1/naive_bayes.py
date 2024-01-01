# naive_bayes.py
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
import numpy as np
from tqdm import tqdm
from collections import Counter
from collections import defaultdict


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
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
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=8.0, pos_prior=0.8, silently=False):
    print_values(laplace,pos_prior)
    #Calls function to calculates positive/negetive probability and positive/negetive unknown probability
    positive_probability, positive_calc_unknprobability = calculate_probability(train_set, train_labels, 1, laplace)
    negative_probability, negative_calc_unknprobability = calculate_probability(train_set, train_labels, 0, laplace)
    #list to store predicted label
    y_hats = []
    
    for doc in tqdm(dev_set, disable=silently):
        positive_likelyhood = np.log(pos_prior)#log likelyhood for positive
        negative_likelyhood = np.log(1 - pos_prior)#log likelyhood for negative
        
        for word in doc:
            #Caulculate log likelyhood for positive
            if word in positive_probability:
                positive_likelyhood += np.log(positive_probability[word])
            else:
                positive_likelyhood += np.log(positive_calc_unknprobability)
            #Caulculate log likelyhood for negative
            if word in negative_probability:
                negative_likelyhood += np.log(negative_probability[word])
            else:
                negative_likelyhood += np.log(negative_calc_unknprobability)
        #If positive likelyhood is greater than negative likelyhood, 1 is added to the list
        if positive_likelyhood > negative_likelyhood:
            y_hats.append(1)
        #If negative likelyhood is greater than positive likelyhood, 0 is added to the list
        else:
            y_hats.append(0)
    #returns the predicted labels
    return y_hats

def calculate_probability(train_set, train_labels, pn_label, laplace_smoothing):
    w_tab = defaultdict(int)#dictionary for word count
    total = 0#total word count
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
