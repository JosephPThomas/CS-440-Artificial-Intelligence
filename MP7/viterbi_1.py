"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag_pointer1:{tag_pointer2: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    tag_dict = defaultdict(lambda: defaultdict(int))

    for sentence in sentences:
        for word, tag in sentence:
            tag_dict[tag][word] += 1

    tag_group = tag_dict.keys()

    for tag in tag_group:
        sum_of_tag_values = sum(tag_dict[tag].values())
        len_of_tag = len(tag_dict[tag])

        for word in tag_dict[tag]:
            word_count = tag_dict[tag][word]
            emit_prob[tag][word] = (word_count + emit_epsilon) / (sum_of_tag_values + emit_epsilon * (len_of_tag + 1))
            if emit_prob[tag][word] == 0:
                emit_prob[tag][word] = emit_epsilon
        emit_prob[tag]['UNK'] = emit_epsilon / (sum_of_tag_values + emit_epsilon * (len_of_tag + 1))

    tag_group_count = defaultdict(lambda: defaultdict(int))

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            tag_pointer1 = sentence[i][1]
            tag_pointer2 = sentence[i + 1][1]

            tag_group_count[tag_pointer1][tag_pointer2] += 1

    for tag_pointer1 in tag_group:
        if tag_pointer1 not in tag_group_count:
            trans_prob[tag_pointer1] = {tag_pointer2: epsilon_for_pt for tag_pointer2 in tag_group}
            continue

        sum_of_tag_values = sum(tag_group_count[tag_pointer1].values())
        len_of_tag = len(tag_group_count[tag_pointer1])

        for tag_pointer2 in tag_group_count[tag_pointer1]:
            tag_count = tag_group_count[tag_pointer1][tag_pointer2]
            trans_prob[tag_pointer1][tag_pointer2] = (tag_count + epsilon_for_pt) / (sum_of_tag_values + epsilon_for_pt * (len_of_tag + 1))

        missing_probability = epsilon_for_pt / (sum_of_tag_values + epsilon_for_pt * (len_of_tag + 1))

        for tag in tag_group:
            if tag not in tag_group_count[tag_pointer1]:
                trans_prob[tag_pointer1][tag] = missing_probability

    for tag in tag_group:
        if tag == 'START':
            init_prob[tag] = 1
        else:
            init_prob[tag] = epsilon_for_pt

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    if i == 0:
        log_prob[ i ] = { }

        for tag in prev_prob:
            log_prob[i][tag] = prev_prob[tag] + math.log(emit_prob[tag].get(word, emit_prob[tag]['UNK']))


        prev_predict_tag_seq[i] = {tag: tag for tag in prev_prob}


        return log_prob, prev_predict_tag_seq
    
    tag_group = prev_prob[i - 1].keys()
    prev_prob[i] = {tag: None for tag in tag_group}
    prev_predict_tag_seq[i] = {tag: None for tag in tag_group}

 
    for tag_obj in tag_group:
        high_probability, high_tag_obj = 0, ''

        high_probability, high_tag_obj = max(
            ((prev_prob[i - 1][tag_alt_obj] + math.log(trans_prob[tag_alt_obj][tag_obj])
            + math.log(emit_prob[tag_obj].get(word, emit_prob[tag_obj]['UNK'])), tag_alt_obj)
            for tag_alt_obj in tag_group),
            key=lambda x: x[0])


        prev_prob[i][tag_obj], prev_predict_tag_seq[i][tag_obj] = high_probability, high_tag_obj


    log_prob, predict_tag_seq = prev_prob, prev_predict_tag_seq


    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag_pointer2), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag_pointer2), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        column = log_prob[ len( log_prob ) - 1 ]
        topTag, maxTagValue = max(column.items(), key=lambda x: x[1])


        order_of_tag = [ ]
        value_of_tag= topTag
        for i in reversed( range(  length ) ):
            order_of_tag.append( ( sentence[ i ], value_of_tag) )
            value_of_tag= predict_tag_seq[ i ][ value_of_tag]

        order_of_tag = order_of_tag[ :: -1]
        predicts.append( order_of_tag )

    return predicts