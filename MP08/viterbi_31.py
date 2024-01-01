"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
from collections import Counter
import math

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word, tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    predicts = []

    word_tag_counts = dict()
    tag_counts = Counter()
    tag_initial_counts = Counter()
    tag_transition_counts = dict()
    hapax_counts = Counter()
    hapax_total_count = 0

    k = 0.00001

    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag_counts:
                word_tag_counts[word] = Counter()
            word_tag_counts[word].update([tag])

    word_tag_counts['UNKNOWN-WORD'] = Counter()

    temp_tag_counter = Counter()
    for sentence in train:
        for word, tag in sentence:
            tag_counts.update([tag])

    for sentence in train:
        for index, (word, tag) in enumerate(sentence):
            if index == 0:
                tag_initial_counts.update([tag])

    for previous_tag in tag_counts:
        tag_transition_counts[previous_tag] = Counter()

    for sentence in train:
        tags_list = [tuple_[1] for tuple_ in sentence]
        for i in range(len(tags_list) - 1):
            previous_tag = tags_list[i]
            next_tag = tags_list[i + 1]
            tag_transition_counts[previous_tag].update([next_tag])
            temp_tag_counter.update([previous_tag])

    for tag in tag_counts:
        for word in word_tag_counts:
            if word_tag_counts[word][tag] == 1:
                hapax_counts.update([tag])
                hapax_total_count += 1

    vocab_size = len(word_tag_counts) - 1
    no_of_tags = len(tag_counts) + 1

    emission_probabilities = dict()
    transition_probabilities = dict()
    initial_probabilities = dict()
    hapax_probabilities = dict()

    for tag in tag_counts:
        probability = (hapax_counts[tag] + k) / (hapax_total_count + k * no_of_tags)
        hapax_probabilities[tag] = probability

    for word in word_tag_counts:
        emission_probabilities[word] = dict()
        for tag in tag_transition_counts:
            probability = (word_tag_counts[word][tag] + (k * hapax_probabilities[tag])) / (
                        tag_counts[tag] + ((k * hapax_probabilities[tag]) * (abs(vocab_size + 1)))
            )  # Numerator will be k for 'UNKNOWN'
            emission_probabilities[word][tag] = probability

    for previous_tag in tag_transition_counts:
        transition_probabilities[previous_tag] = dict()
        for next_tag in tag_transition_counts:
            transition_probabilities[previous_tag][next_tag] = (tag_transition_counts[previous_tag][next_tag] + k) / (
                        temp_tag_counter[previous_tag] + k * no_of_tags
            )

    for tag in tag_transition_counts:
        initial_probabilities[tag] = (tag_initial_counts[tag] + k) / (len(train) + k * no_of_tags)

    for sentence in test:
        prev_word_tag_nodes = []
        for index, word in enumerate(sentence):
            curr_word_tag_nodes = []

            if word in word_tag_counts:
                if index == 0:
                    for tag in word_tag_counts[word]:
                        p = math.log10(initial_probabilities[tag]) + math.log10(emission_probabilities[word][tag])
                        curr_word_tag_nodes.append({"probability": p, "backpointer": None, "tag": tag, "word": word})
                else:
                    for tag in word_tag_counts[word]:
                        p = math.log10(emission_probabilities[word][tag]) + max(
                            [node["probability"] + math.log10(transition_probabilities[node["tag"]][tag]) for node in
                             prev_word_tag_nodes])
                        max_node = max(prev_word_tag_nodes, key=lambda node_: node_["probability"] + math.log10(
                            transition_probabilities[node_["tag"]][tag]))  # for back pointer
                        curr_word_tag_nodes.append({"probability": p, "backpointer": max_node, "tag": tag, "word": word})
            else:
                if index == 0:
                    for tag in transition_probabilities:
                        p = math.log10(initial_probabilities[tag]) + math.log10(
                            emission_probabilities['UNKNOWN-WORD'][tag])
                        curr_word_tag_nodes.append({"probability": p, "backpointer": None, "tag": tag, "word": word})
                else:
                    for tag in transition_probabilities:
                        p = math.log10(emission_probabilities['UNKNOWN-WORD'][tag]) + max(
                            [node["probability"] + math.log10(transition_probabilities[node["tag"]][tag]) for node in
                             prev_word_tag_nodes])
                        max_node = max(prev_word_tag_nodes, key=lambda node_: node_["probability"] + math.log10(
                            transition_probabilities[node_["tag"]][tag]))  # for back pointer
                        curr_word_tag_nodes.append({"probability": p, "backpointer": max_node, "tag": tag, "word": word})

            prev_word_tag_nodes = curr_word_tag_nodes

        temp_reverse_sentence = []
        curr_node = max(prev_word_tag_nodes, key=lambda node_: node_["probability"])

        while curr_node != None:
            temp_reverse_sentence.append((curr_node["word"], curr_node["tag"]))
            curr_node = curr_node["backpointer"]

        temp_reverse_sentence.reverse()

        predicts.append(temp_reverse_sentence)

    return predicts
