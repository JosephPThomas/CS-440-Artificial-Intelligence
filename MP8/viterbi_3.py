"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
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


    all_tag_dict = defaultdict(dict)
    word_dict = defaultdict(int)
    tag_dict = defaultdict(lambda: defaultdict(int))
    tag_dict_count = defaultdict(int)
    tag_count = 0

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            present_word, present_tag = sentence[i]
            next_word, upcoming_tag = sentence[i + 1]

            if present_tag not in all_tag_dict:
                all_tag_dict[present_tag] = {}
            all_tag_dict[present_tag][upcoming_tag] = all_tag_dict[present_tag].get(upcoming_tag, 0) + 1

            word_dict[present_word] += 1
            tag_dict[present_tag][present_word] += 1

            if word_dict[present_word] == 1:
                tag_dict_count[present_tag] += 1
                tag_count += 1

    probability = {tag: emit_epsilon * (tag_dict_count.get(tag, 0) + emit_epsilon) / (tag_count + emit_epsilon * (len(tag_dict) - 2)) for tag in tag_dict}

    for tag in tag_dict:
        if tag == 'START':
            init_prob[tag] = 1
        else:
            init_prob[tag] = epsilon_for_pt

    for tag in tag_dict:
        len_of_tag = len(tag_dict[tag])
        sum_of_tag_values = sum(tag_dict[tag].values())

        scale = emit_epsilon * probability[tag]
        emit_prob[tag]["UNK"] = scale / (sum_of_tag_values + emit_epsilon * len_of_tag)

        for word in tag_dict[tag]:
            Pe = (tag_dict[tag][word] + emit_epsilon) / (sum_of_tag_values + emit_epsilon * len_of_tag)
            emit_prob[tag][word] = 0.0001 if Pe == 0 else Pe

    for tag in tag_dict:
        len_of_tag = len(all_tag_dict[tag])
        sum_of_tag_values = sum(all_tag_dict[tag].values())

        for repeat_tag in all_tag_dict[tag]:
            Pt = (all_tag_dict[tag][repeat_tag] + epsilon_for_pt) / (sum_of_tag_values + epsilon_for_pt * len_of_tag)
            trans_prob[tag][repeat_tag] = 0.0001 if Pt == 0 else Pt

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

    if i >= 0:  

        for unique_tag in emit_prob:
            high_probability = float("-inf")
            high_tag = None

            for tags in prev_prob:
                calculate_probability = prev_prob[tags]

                emit_prob_for_word = emit_prob[unique_tag].get(word, emit_prob[unique_tag]["UNK"])

                transition_prob = trans_prob.get(tags, {}).get(unique_tag, 0.0001)

                calculate_probability += math.log(emit_prob_for_word) + math.log(transition_prob)

                if calculate_probability > high_probability:
                    high_probability = calculate_probability
                    high_tag = tags

            log_prob[unique_tag] = high_probability
            predict_tag_seq[unique_tag] = prev_predict_tag_seq[high_tag] + [unique_tag]

    return log_prob, predict_tag_seq

def viterbi_3(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    prediction = []
    
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
            

        high_tag = None
        high_probability = float("-inf")

        for tags in log_prob:
            if log_prob[tags] > high_probability:
                high_tag = tags
                high_probability = log_prob[tags]

        prediction = [(sentence[num], predict_tag_seq[high_tag][num]) for num in range(len(sentence))]
        predicts.append(prediction)

    return predicts
