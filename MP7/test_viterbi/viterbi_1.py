"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log
import pdb

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect
alpha = emit_epsilon  # Set Smoothing Constant used, but set here for accessibility if we need to change it.

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    # Input "sentences" comes in format as a list of list of    #
    # tuples, with the tuples containing (word, TAG).           #

    # ------------ CALCULATE INTIAL PROBABILITIES ------------- #
    # Count occurrences of each tag that occur at the beginning #
    # of a sentence, i.e. after the START tag.                  #
    initialTagsCount = { }


    # Since 'START' is our only valid start to the sentence,    #
    # its probability is 1. We set the probability of all other #
    # tags to some arbitrarily small number, since if we set to #
    # 0 it might mess up our math and cause errors.             #
    init_prob[ 'START' ] = 1
    init_prob[ 'UNK' ] = epsilon_for_pt 

    # We don't want to hardcode 'START' as our start indicator, #
    # so to avoid such an error we need to count the number of  #
    # tags



    # ------------ CALCULATE EMISSION PROBABILITIES ----------- #
    # Emission Probability is Probability of a Word given a Tag #
    # for each position k in the tag sequence. We use the       #
    # method from Naive Bayes, that is:                         #
    # P( W | C ) = [ count( W ) + a ] / [ n + a( V + 1 ) ]      #
    # Where:                                                    #
    # W is our Word - note that count( W ) only include counts  #
    #   for the specific class we are calculating for.          #
    # C is our Tag                                              #
    # a is our Smoothing Constant                               #
    # n is the Total Number of Words in training data for class #
    #   C, which in this case is our Tag                        #
    # V is the Number of Unique Words seen in training data for #
    #   class C, which in this case is our Tag                  #
    # We also have the probability for our UNKNOWN word, which  #
    # is just:                                                  #
    # P( UNK | C ) = a / [ n + a( V + 1 ) ]                     #
    # So we need to:                                            #
    # - Count the number of unique words seen in training data  #
    #   for each Tag                                            #
    # - Count the number of times a word appears with tag T for #
    #   each Tag                                                #
    # - Count the total number of times the tag T occurs        #
    tagWordCounts = { }
    for sentence in sentences:
        for tagPair in sentence:
            word = tagPair[ 0 ]
            tag  = tagPair[ 1 ]

            # Check if tag in dictionary, add entry if needed.  #
            if tag not in tagWordCounts:
                tagWordCounts[ tag ] = { }
            
            # Check if word was seen for tag already, add entry #
            # if needed.                                        #
            if word not in tagWordCounts[ tag ]:
                tagWordCounts[ tag ][ word ] = 0
            # Increment count for word in dictionary for tag.   #
            tagWordCounts[ tag ][ word ] += 1

    tagSet = tagWordCounts.keys( )

    # Calculate P( W | T ) in emit_prob, which has format       #
    # {tag: {word: # }                                          #
    # We store as logs to avoid underflow.                      #
    # Our Smoothing Constant is Epsilon.                        #
    for tag in tagSet:
        # Get the total number of words in training data for    #
        # Tag T and number of unique words for Tag T.           #
        n = sum( tagWordCounts[ tag ].values( ) )
        v = len( tagWordCounts[ tag ].keys( ) )

        # Iterate through words of tag and apply formula.       #
        for word in tagWordCounts[ tag ]:
            countW = tagWordCounts[ tag ][ word ]
            emit_prob[ tag ][ word ] = ( countW + emit_epsilon ) / ( n + emit_epsilon * ( v + 1 ) )

        # Also compute UNKNOWN probability for tag.             #
        emit_prob[ tag ][ 'UNK' ] = emit_epsilon / ( n + emit_epsilon * ( v + 1 ) )


    # ---------- CALCULATE TRANSITION PROBABILITIES ----------- #
    # We can estimate the transtition probability by counting   #
    # how often a 2-tag sequence occurs and dividing by how     #
    # often the first tag occurs.                               #
    # However, we need to apply smoothing like we did for the   #
    # emission probabilities. Should be simple - we just use    #
    # our counts of pairs in place of words, and apply the same #
    # concept.                                                  #
    tagPairsCount = { }

    # Count number of occurrences for each pair of tags.        #
    for sentence in sentences:
        for i in range( len( sentence ) - 1 ):
            # Get tags we will be analyzing.                    #
            tag0 = sentence[   i   ][ 1 ]
            tag1 = sentence[ i + 1 ][ 1 ]
            
            # Check if tag0 in dictionary, add entry if needed. #
            if tag0 not in tagPairsCount:
                tagPairsCount[ tag0 ] = { }

            # Check if following tag has been seen already, add #
            # entry if needed.                                  #
            if tag1 not in tagPairsCount[ tag0 ]:
                tagPairsCount[ tag0 ][ tag1 ] = 0
            # Increment count for pair in dictionary for tag.   #
            tagPairsCount[ tag0 ][ tag1 ] += 1

    # Calculate probability similar to like we did for emission #
    # probabilities.                                            #
    for tag0 in tagSet:
        # Handle 'END' specifically, since it has no            #
        # transitions to other tags. Set its values as epsilon. #
        if tag0 not in tagPairsCount:
            trans_prob[ tag0 ] = { }
            for tag1 in tagSet:
                trans_prob[ tag0 ][ tag1 ] = epsilon_for_pt
            continue

        # Get the total number of occurrences of the pair in    #
        # the training data and the number of unique follow-up  #
        # tags for Tag T.                                       #
        n = sum( tagPairsCount[ tag0 ].values( ) )
        v = len( tagPairsCount[ tag0 ].keys( ) )

        # Iterate through following tags of tag T and apply     #
        # formula.                                              #
        for tag1 in tagPairsCount[ tag0 ]:
            countT = tagPairsCount[ tag0 ][ tag1 ] 
            trans_prob[ tag0 ][ tag1 ] = ( countT + epsilon_for_pt ) / ( n + epsilon_for_pt * ( v + 1 ) )

        # We also need to compute probabilites for tags we      #
        # don't counts for, but can just use the UNKNOWN        #
        # probability.                                          #
        probUNKNOWN = epsilon_for_pt / ( n + epsilon_for_pt * ( v + 1 ) )

        # tagPairsCount should have seen every tag possible in  #
        # the tag set, assuming that the provided training data #
        # encountered every tag in the target tag set. Check    #
        # for any tags that were not encountered in follow-ups  #
        # for each tag and set their probability to be nonzero. #
        for tag in tagSet:
            if tag not in tagPairsCount[ tag0 ]:
                trans_prob[ tag0 ][ tag ] = probUNKNOWN

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice. Note: is recalculated with each step. This is our array v.
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column. Note: is recalculated with each step. This is our array b.
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i). This is our array v.
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i). This is our array b.

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    
    # Suppose that our input word sequence is w1...wn.      #
    # The basic data structure is an  n by m array v, where #
    # n is the length of the input word sequence and m is   #
    # the is the number of different (unique) tags.         #
    # Each cell (k, t) in the array contains the            #
    # probability v(k, t) of the best sequence of tags for  #
    # w1...wk that ends with tag t.                         #
    # We also store b(k, t), which is the previous best tag #
    # in that best sequence.                                #
    # Given a word and index, compute the tag with the      #
    # maximum probability based on the previously           #
    # predicted tag.                                        #

    # ------------------ INITIALIZATION ------------------- #
    # Check if first column, i.e. i == 0. We have to        #
    # fill out the first column differently.                #
    # Specifically, the cell for tag t get the value:       #
    # v( 1, t ) = Ps( t ) * Pe( w1 | t )                    #
    # where:                                                #
    # t is the tag                                          #
    # Ps( t ) is the initial probability of t               #
    # Pe( w1 | t ) is the emission probability of word w1,  #
    #   given tag t.                                        #
    # 'START' will always be the first word we see at the   #
    # beginning of a sentence, so it's probability will be  #
    # 1, but we still have to account for other words, in   #
    # which case we will use an extremely small number to   #
    # avoid math errors.                                    #
    if i == 0:
        # We expect 'START' at the beginning of our         #
        # sentence. However, in the test_viterbi test,      #
        # we don't get the 'START' and 'END' delimiters.    #
        # So, we should account for such for now.           #
        
        # We format log_prob as dict of dicts, so that we   #
        # can keep track of probabilities for each tag, for #
        # each index.                                       #
        log_prob[ i ] = { }

        # On first iteration, log_prob holds our initial    #
        # probabilities. Iterate through tags in initial    #
        # probs and add probabilities. Should already have  #
        # log applied to probability.                       #
        for tag in prev_prob:
            log_prob[ i ][ tag ] = prev_prob[ tag ]
            log_prob[ i ][ tag ] += math.log( emit_prob[ tag ][ word ] )

        # Also add an entry to prev_predict_tag_seq. We     #
        # expect the previous to be 'START'.                #
        prev_predict_tag_seq[ i ] = { }
        for tag in prev_prob:
            prev_predict_tag_seq[ i ][ tag ] = tag

        return log_prob, prev_predict_tag_seq
    
    # -------------- MOVING FORWARDS IN TIME -------------- #
    # Use the values in column k to fill column k + 1.      #
    # Specifically:                                         #
    # For each tagB:                                        #
    #   v(k+1, tagB) =                                      
    #       max_tagA v(k,tagA)*PT(tagB|tagA)*Pe(w_k+1|tagB) 
    #   b(k+1, tagB) =
    #       argmax_tagA v(k,tagA)*PT(tagB|tagA)*Pe(w_k+1|tagB)
    # That is, we compute the above for all possible tags   #
    # tagA. The maximum value goes into trellis cell        #
    # v(k+1,tagB) and the corresponding value of tagA is    #
    # stored in b(k+1,tagB)                                 #         
    # Calculate equation for CURRENT column based on        #
    # PREVIOUS column, as opposed to above which calculates #
    # for the NEXT column. This is due to the code's        #
    # implementation.                                       #
    
    # Get tagset, which is the keys to our column.          #
    tagSet = prev_prob[ i - 1 ].keys( )
    # Also initialize our dictionary objects.               #
    prev_prob[ i ] = { }
    prev_predict_tag_seq[ i ] = { }

    # Calculate probability for each tag in column.         #
    for tagB in tagSet:
        # Compute for all possible tags tagA. Find tagA     #
        # s.t. probability is maximized.                    #
        maxProb = 0
        maxTagA = ''

        # Calculate probability of tagA, for all tags. tagA #
        # is our first tag, followed by tagB.               #
        for tagA in tagSet:

            # prev_prob should already have log applied.    #
            prob = prev_prob[ i - 1 ][ tagA ]
            # Add transmission probability. Takes format of #
            # [ first tag ][ following tag ].               #
            prob += math.log( trans_prob[ tagA ][ tagB ] )
            # Add emission probablity. We need to check if  #
            # the word was seen in the training data.       #
            if word in emit_prob[ tagB ]:
                prob += math.log( emit_prob[ tagB ][ word ] )
            # Else, add with value in 'UNKNOWN' entry.      #
            else:
                prob += math.log( emit_prob[ tagB ][ 'UNK' ] )

            # Check if probability is higher than           #
            # previously seen values.                       #
            if maxProb == 0 or maxProb < prob:
                maxProb = prob
                maxTagA = tagA

        # Set probability for tagB at time i. Also set its  #
        # predicted sequence.                               #
        prev_prob[ i ][ tagB ] = maxProb
        prev_predict_tag_seq[ i ][ tagB ] = maxTagA

    # Set log_prob and predict_tag_seq to properly return   #
    # our output.                                           #
    log_prob = prev_prob 
    predict_tag_seq = prev_predict_tag_seq

    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    print( init_prob )
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
        # Pick the best tag in the final column and trace   #
        # backwards from T, using the values in b, to       #
        # produce the output tag sequence.                  #
        # Find max value in log_prob.                       #
        column = log_prob[ len( log_prob ) - 1 ]
        maxTagValue = 0
        maxTag = ''
        for tag in column:
            if maxTagValue == 0 or maxTagValue < column[ tag ]:
                maxTagValue = column[ tag ]
                maxTag = tag

        print( 'maxTag:', maxTag )
        print( 'maxTagValue:', maxTagValue )

        # Tag at end with maximum probability found, now    #
        # backtrack.                                        #
        tagSequence = [ ]
        tag = maxTag
        for i in reversed( range(  length ) ):
            word = sentence[ i ]
            tagSequence.append( ( word, tag ) )
            tag = predict_tag_seq[ i ][ tag ]

        # Reverse tagSequence to get the right order.       #
        tagSequence = tagSequence[ :: -1]
        predicts.append( tagSequence )

        print( 'log_prob:', log_prob )
        print( 'predict_tag_seq:', predict_tag_seq )

    return predicts