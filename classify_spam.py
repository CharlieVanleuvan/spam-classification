'''
  This program shell reads email data for the spam classification problem.
  The input to the program is the path to the Email directory "corpus" and a limit number.
  The program reads the first limit number of ham emails and the first limit number of spam.
  It creates an "emaildocs" variable with a list of emails consisting of a pair
    with the list of tokenized words from the email and the label either spam or ham.
  It prints a few example emails.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classify_spam.py  <corpus directory path> <limit number> <feature set>
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
import pandas as pd 
import math
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

#PRECISION, RECALL, AND F-MEASURE
def eval_measures(gold,predicted):
  #get a list of labels
  labels = list(set(gold))

  recall_list = []
  precision_list = []
  f_measure_list = []

  for lab in labels:
    #for each label, compare gold and predicted lists and compute value
    tp = fp = fn = tn = 0
    for i, val in enumerate(gold):
      if val == lab and predicted[i] == lab:
        tp += 1
      if val == lab and predicted[i] != lab:
        fn += 1
      if val != lab and predicted[i] == lab:
        fp += 1
      if val != lab and predicted[i] != lab:
        tn += 1
    
    try:
      recall = tp / (tp + fp)
    except ZeroDivisionError:
      recall = 0
    
    try:
      precision = tp / (tp + fn)
    except ZeroDivisionError:
      precision = 0

    recall_list.append(recall)
    precision_list.append(precision)

    try:
      f_measure_list.append(2 *(recall * precision)/ (recall + precision))
    except ZeroDivisionError:
      f_measure_list.append(0)
  
  print('\tPrecision\tRecall\t\tF')
  for i, lab in enumerate(labels):
    print(lab,'\t', "{:10.3f}".format(precision_list[i]),\
      "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(f_measure_list[i]))

#CROSS VALIDATION ACCURACY FUNCTION
def cross_validation_accuracy(num_folds, featuresets):
  subset_size = int(len(featuresets) / num_folds)

  print('Each fold size: ',subset_size)
  print('Accuracies:')

  accuracy_list = []

  #iterate over the folds
  for i in range(num_folds):
    #split test and training data based on entered num_folds
    test_this_round = featuresets[(i*subset_size):][:subset_size]
    train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]

    #run NB classifier
    classifier = nltk.NaiveBayesClassifier.train(train_this_round)

    #create gold standard and predicted classification lists
    goldList = []
    predictedList = []
    for (features,label) in test_this_round:
      goldList.append(label)
      predictedList.append(classifier.classify(features))

    #calculate accuracy
    accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
    print('\n',i, accuracy_this_round)
    eval_measures(goldList,predictedList)

    #append accuracy to list
    accuracy_list.append(accuracy_this_round)
  print('Average Accuracy from k-fold CV: ', sum(accuracy_list) / num_folds)

# define a feature definition function here

#Unigram features / Bag of words features. Creates a boolean feature label
def bag_of_words_features(document,word_features):
  #creates a feature for each word and returns a boolean TRUE or FALSE if that doc contains that word
  document_words = set(document)
  features = {}

  #loop through the keyword list, and create key in features dict
  for word in word_features:
    features['V_{}'.format(word)] = (word in document_words)
  return(features)

#Bigram features extraction
def bigram_document_features(document, word_features,bigram_features):
  #create a set of unique words and bigrams from the document
  document_words = set(document)
  document_bigrams = nltk.bigrams(document)

  #append the boolean value of each unigram and bigram to the features dict
  features = {}
  for word in word_features:
    features['V_{}'.format(word)] = (word in document_words)
  for bigram in bigram_features:
    features['B_{}_{}'.format(bigram[0],bigram[1])] = (bigram in document_bigrams)
  return(features)

#POS feature extraction. Counts the numbers of tags of nouns, verbs, etc. in a doc
def POS_features(document, word_features):
  document_words = set(document)
  tagged_words = nltk.pos_tag(document)

  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  
  numNoun = 0
  numVerb = 0
  numAdj = 0
  numAdverb = 0
  for (word,tag) in tagged_words:
    if tag.startswith('N'):
      numNoun += 1
    if tag.startswith('V'):
      numVerb += 1
    if tag.startswith('J'):
      numAdj += 1
    if tag.startswith('R'):
      numAdverb += 1
  
  features['nouns'] = numNoun
  features['verbs'] = numVerb
  features['adjectives'] = numAdj 
  features['adverbs'] = numAdverb
  return(features)

#Punctuation frequency feature extraction
def punctuation_freq_features(document, word_items):
  #for word in word items
    #get freq of word
    #create key in features and set = freq of word
  features = {}
  for (word,freq) in word_items:
    features['V_{}'.format(word)] = freq
  return(features)

#Tf-Idf feature extraction
def computeTF(document, word_features):
  #compute Term Frequency for each word. Each document will have its own term frequency for a word
  num_words_in_doc = len(document)
  tfDict = {}
  for word in word_features:
    freq = sum(1 for i in document if i == word)
    try:
      tfDict[word] = freq / num_words_in_doc
    except ZeroDivisionError:
      tfDict[word] = 0
  return(tfDict)

def computeIDF(documents, word_features):
  #compute inverse document frequency for all docs at once
  #log(number of documents / number of documents that contain word w)

  #number of documents
  N = len(documents)

  #number of documents that contain word n
  idfDict = dict.fromkeys(word_features,0)
  for text,label in documents:
    for word in word_features:
      if word in text:
        idfDict[word] += 1
  
  #take the log of the two frequencies
  for word,val in idfDict.items():
    idfDict[word] = math.log(N/float(val))
  
  return(idfDict)

def computeTFIDF(tf,idfs):
  tfidf = {}

  for word,val in tf.items():
    tfidf[word] = val * idfs[word]
  return(tfidf)

# function to read spam and ham files, train and test a classifier. [(tokenized text, classification)]
def processspamham(dirPath,limitStr,feature_set):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  # start lists for spam and ham email texts
  hamtexts = []
  spamtexts = []
  os.chdir(dirPath)
  # process all files in directory that end in .txt up to the limit
  #    assuming that the emails are sufficiently randomized
  for file in os.listdir("./spam"):
    if (file.endswith(".txt")) and (len(spamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./spam/"+file, 'r', encoding="latin-1")
      spamtexts.append (f.read())
      f.close()
  for file in os.listdir("./ham"):
    if (file.endswith(".txt")) and (len(hamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./ham/"+file, 'r', encoding="latin-1")
      hamtexts.append (f.read())
      f.close()
  
  # print number emails read
  print ("Number of spam files:",len(spamtexts))
  print ("Number of ham files:",len(hamtexts))
  
  # create list of mixed spam and ham email documents as (list of words, label)
  emaildocs = []
  # add all the spam
  for spam in spamtexts:
    #default tokenization
    tokens = nltk.word_tokenize(spam)
    
    #experiment, select only the punctuation
    #tokenizer = RegexpTokenizer(r"[^a-zA-Z\d\s:]")
    #tokens = tokenizer.tokenize(spam)

    emaildocs.append((tokens, 'spam'))
  # add all the regular emails
  for ham in hamtexts:
    #default tokenization
    tokens = nltk.word_tokenize(ham)

    #experiment, select only the punctuation
    #tokenizer = RegexpTokenizer(r"[^a-zA-Z\d\s:]")
    #tokens = tokenizer.tokenize(ham)

    emaildocs.append((tokens, 'ham'))
  
  # randomize the list
  random.shuffle(emaildocs)
  
    
  #Filter tokens for stop words
  stopwords = nltk.corpus.stopwords.words('english')

  # Get all words in list and get features

  #UNIGRAM FEATURES (WITHOUT STOPWORDS)
  #create word features list(this is a list of the words and their frequencies in emaildocs). 
  all_words_list = [word for (words,label) in emaildocs for word in words if word not in stopwords]
  all_words = nltk.FreqDist(all_words_list)
  word_items = all_words.most_common(1500)
  word_features = [word for (word,freq) in word_items]

  
  #BIGRAM FEATURES
  bigram_measures = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(all_words_list)
  bigram_features = finder.nbest(bigram_measures.chi_sq,500)
  
  
  
  # GET FEATURE SETS FROM FEATURE DEFINITION FUNCTION
  #call feature functions here
  
  if feature_set == 'unigram':
    #bag of word featureset
    bag_of_word_featureset = [(bag_of_words_features(d,word_features = word_features),c) for (d,c) in emaildocs]
    #TRAIN A CLASSIFIER AND SHOW PERFORMANCE IN CROSS VALIDATION
    cross_validation_accuracy(num_folds = 10, featuresets = tf_idf_featurese

  if feature_set == 'bigram':
    #bigram featureset
    bigram_featureset = [(bigram_document_features(document = d,word_features = word_features, bigram_features = bigram_features),c) for (d,c) in emaildocs]
    #TRAIN A CLASSIFIER AND SHOW PERFORMANCE IN CROSS VALIDATION
    cross_validation_accuracy(num_folds = 10, featuresets = bigram_features)

  if feature_set == 'POS':
    #POS featureset
    pos_featureset = [(POS_features(d, word_features=word_features),c) for (d,c) in emaildocs]
    # TRAIN A CLASSIFIER AND SHOW PERFORMANCE IN CROSS VALIDATION
    cross_validation_accuracy(num_folds = 10, featuresets = pos_featureset)

  if feature_set = "TF-IDF":
    #TF-IDF featureset
    idfs = computeIDF(emaildocs,word_features=word_features)
    tf_idf_featureset = [(computeTFIDF(computeTF(d,word_features=word_features),idfs=idfs),c) for (d,c) in emaildocs]
    # TRAIN A CLASSIFIER AND SHOW PERFORMANCE IN CROSS VALIDATION
    cross_validation_accuracy(num_folds = 10, featuresets = tf_idf_featureset)

  #Punctuation frequency featureset. Experiment only. Uncoment the above punctuation tokenizer to run this part.
  #punct_featureset = [(punctuation_freq_features(d,word_items=word_items),c) for (d,c) in emaildocs]



"""
commandline interface takes a directory name with ham and spam subdirectories
   and a limit to the number of emails read each of ham and spam
It then processes the files and trains a spam detection classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 4):
        print ('usage: python classifySPAM.py <corpus-dir> <limit> <feature-set>')
        sys.exit(0)
    processspamham(sys.argv[1], sys.argv[2], sys.argv[3])
        
