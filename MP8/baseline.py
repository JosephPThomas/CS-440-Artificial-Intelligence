"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    list_word_tag = {}
    for sentence in train:
        for word, tag in sentence:
            if word not in list_word_tag:
                list_word_tag[word] = {}
            list_word_tag[word][tag] = list_word_tag[word].get(tag, 0) + 1

    word_tag = {}
    for word in list_word_tag:
        maximum_tag = max(list_word_tag[word], key=list_word_tag[word].get)
        word_tag[word] = maximum_tag

    tags_count = {}
    frequent_tag = None
    frequent_tag_count = -1
    for word, tag in word_tag.items():
        tags_count[tag] = tags_count.get(tag, 0) + 1
        if tags_count[tag] > frequent_tag_count:
            frequent_tag = tag
            frequent_tag_count = tags_count[tag]

    list_of_word_tag_pair = []
    for sentence in test:
        word_tag_pair = [(word, word_tag.get(word, frequent_tag)) for word in sentence]
        list_of_word_tag_pair.append(word_tag_pair)

    return list_of_word_tag_pair