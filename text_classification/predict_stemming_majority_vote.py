import numpy as np
import Levenshtein
from sklearn import svm, grid_search
import os
import math

# Word endings
endings = ["fy", "di", "crjuj", ""]

# Load training data
training = np.loadtxt("../handout/training.csv", delimiter=",", dtype=np.dtype(str))
train_city_code = training[:,1].astype(np.dtype(float))
train_country_code = training[:,2].astype(np.dtype(float))
city_names = training[:,0]

# Load validation data
validation = np.loadtxt("../handout/validation.csv", delimiter=",", dtype=np.dtype(str))
validation_city_names = validation

# Load testing data
testing = np.loadtxt("../handout/testing.csv", delimiter=",", dtype=np.dtype(str))
testing_city_names = testing

unique_countries = np.unique(train_country_code)

# Make all city names lowercase
city_names = np.core.defchararray.lower(city_names)
validation_city_names = np.core.defchararray.lower(validation_city_names)
testing_city_names = np.core.defchararray.lower(testing_city_names)

all_city_names = np.concatenate((city_names, validation_city_names))
all_city_names = np.concatenate((all_city_names, testing_city_names))

# Build a dictionary of all the words in the city names
dictionary = list()

def similarity(word1, word2):
    maxlen = max(len(word1), len(word2))
    return (maxlen-Levenshtein.distance(word1,word2))/float(maxlen);

def filterForDictionary (word):
    # Filter 1: Levenshtein distance
    if len(word) > 2:
        if len([p for p in dictionary if similarity(word,p) > 0.60]) != 0: # There exists a word in the dictionary with levenshtein distance 1 or less
            return False
        else:
            return True
            # Check stemming
            #stem = word[:-2]
            #if len([p for p in dictionary if p == stem]) != 0:
            #    return False
            #else:
            #    return True
    else:
        return True

for city_name in all_city_names:
    for word in city_name.split(' '):
        if not (word in dictionary) and filterForDictionary(word):
            dictionary.append(word)

print "Dictionary size: "+str(len(dictionary))+"\n\n"

# def closestDictionaryWordGivenWord(word):
#     # Check for exact match
#     if word in dictionary:
#         return word
#     else:
#         # Take stem of word and

# Compute bag of word features for training
features = np.zeros(shape=(len(city_names), len(dictionary)), dtype=np.dtype(int))
i = 0
for city_name in city_names:
    for word in city_name.split(" "):

        if word in dictionary:
            index = dictionary.index(word)
            features[i,index] += 1
        else:
            candidate_words = [(p, -1*similarity(word,p)) for p in dictionary if similarity(word,p) > 0.60]
            if len(candidate_words) > 0:
                minimal_distance = map(sorted, zip(*candidate_words))[1][0]
                minimal_word_index = zip(*candidate_words)[1].index(minimal_distance)
                minimal_word = candidate_words[minimal_word_index][0]
                index = dictionary.index(minimal_word)
                features[i,index] += 1

    i += 1

#Compute bag of word features for validation
validation_features = np.zeros(shape=(len(validation_city_names), len(dictionary)), dtype=np.dtype(int))
i = 0
for city_name in validation_city_names:
    for word in city_name.split(" "):

        if word in dictionary:
            index = dictionary.index(word)
            validation_features[i,index] += 1
        else:
            candidate_words = [(p, -1*similarity(word,p)) for p in dictionary if similarity(word,p) > 0.60]
            if len(candidate_words) > 0:
                minimal_distance = map(sorted, zip(*candidate_words))[1][0]
                minimal_word_index = zip(*candidate_words)[1].index(minimal_distance)
                minimal_word = candidate_words[minimal_word_index][0]
                index = dictionary.index(minimal_word)
                validation_features[i,index] += 1
    i += 1

#Compute bag of word features for testing
testing_features = np.zeros(shape=(len(testing_city_names), len(dictionary)), dtype=np.dtype(int))
i = 0
for city_name in testing_city_names:
    for word in city_name.split(" "):

        if word in dictionary:
            index = dictionary.index(word)
            testing_features[i,index] += 1
        else:
            candidate_words = [(p, -1*similarity(word,p)) for p in dictionary if similarity(word,p) > 0.60]
            if len(candidate_words) > 0:
                minimal_distance = map(sorted, zip(*candidate_words))[1][0]
                minimal_word_index = zip(*candidate_words)[1].index(minimal_distance)
                minimal_word = candidate_words[minimal_word_index][0]
                index = dictionary.index(minimal_word)
                testing_features[i,index] += 1
    i += 1


#######
# Majority voting for city_code
#######

for country in unique_countries:
    country_code_features = features[np.where(train_country_code == country)] #Magic to get the features for the given country code
    city_code_labels = train_city_code[np.where(train_country_code == country)] #Magic to get the labels for the given country code

    for feature_row in country_code_features:
        city_code_indexes = np.where((country_code_features == feature_row).all(axis=1))
        city_codes = city_code_labels[city_code_indexes]
        bincount = np.bincount(city_codes.astype(np.dtype(int)))
        majority_city_code = bincount.argmax()

        #print "\n\ncity_codes:"+str(city_codes)+"\nmajority_city_code: "+str(majority_city_code)+"\nNumber of city_codes: "+str(len(city_codes))

        # We only set the majority code if the frequency of it is at least 2/3
        if float(bincount[majority_city_code])/len(city_codes) > 1:
            city_code_labels[city_code_indexes] = majority_city_code

    # Write back to main training array
    train_city_code[np.where(train_country_code == country)] = city_code_labels


#####
# One SVM per country code
#####

# # Train SVM for the country codes
# params = {'C': np.arange(0.01,1,0.1)}
# svr = svm.LinearSVC()
# clf = grid_search.GridSearchCV(svr, params, cv=5, n_jobs=4)
# clf.fit(features, train_country_code)

# print "\n\n\nCountry code SVM"
# print "TRAINING====="
# print "Best parameters: {0}\n\n Best score: {1}".format(clf.best_estimator_, clf.best_score_)
# print "Grid scores on development set:\n\n"
# for params, mean_score, scores in clf.grid_scores_:
#     print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)

# country_code_svm = clf.best_estimator_

# city_code_svms = list()
# for country in unique_countries:
#     country_code_features = features[np.where(train_country_code == country)] #Magic to get the features for the given country code
#     country_code_labels = train_city_code[np.where(train_country_code == country)] #Magic to get the labels for the given country c
   
#     # Train SVM and perform cross validation
#     params = {'C': np.arange(0.01,1,0.1)}
#     svr = svm.LinearSVC()
#     clf = grid_search.GridSearchCV(svr, params, cv=5, n_jobs=4)
#     clf.fit(country_code_features, country_code_labels)
    
#     print "\n\n\nCountry code: "+str(country)
#     print "TRAINING====="
#     print "Best parameters: {0}\n\n Best score: {1}".format(clf.best_estimator_, clf.best_score_)
#     print "Grid scores on development set:\n\n"
#     for params, mean_score, scores in clf.grid_scores_:
#         print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)

#     city_code_svms.append(clf.best_estimator_) # This saves the object describing the trained svm in the list

# # Predict country code for each validation entry
# country_code_pred = country_code_svm.predict(validation_features)

# final_labels = list()
# i = 0
# for country_code in country_code_pred:
#     feature_row = validation_features[i,:]
    
#     # Get svm for prediction
#     index = np.where(unique_countries == country_code)
#     index = index[0][0]
#     city_code_svm = city_code_svms[index]
    
#     city_code_pred = city_code_svm.predict(feature_row)
    
#     final_labels.append(str(int(city_code_pred[0]))+","+str(int(country_code)))
    
#     i += 1

# if os.path.exists("y-pred.txt"):
#     os.remove("y-pred.txt")

# with open("y-pred.txt", "w") as f:
#     for pred in final_labels:
#         f.write(pred+"\n")


####
# One classifiying SVM
####

#appending the country and city labels
append_labels = list()
i = 0
for c in train_country_code:
    append_labels.append(int(str(int(train_city_code[i]))+str(int(c))))
    i+=1

# train svm
params = {'C': np.arange(0.01,1,0.1)}
svr = svm.LinearSVC()
clf = grid_search.GridSearchCV(svr, params, cv=5, n_jobs=4)
clf.fit(features, append_labels)

print "TRAINING====="
print "Best parameters: {0}\n\n Best score: {1}".format(clf.best_estimator_, clf.best_score_)
print "Grid scores on development set:\n\n"
for params, mean_score, scores in clf.grid_scores_:
    print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)

ypred_validation = clf.best_estimator_.predict(validation_features)
ypred_testing = clf.best_estimator_.predict(testing_features)


if os.path.exists("y-pred.txt"):
    os.remove("y-pred.txt")
if os.path.exists("y-pred-testing.txt"):
    os.remove("y-pred-testing.txt")

with open("y-pred.txt", "w") as f:
    for pred in ypred_validation:
        string = str(pred)
        city = string[0:6]
        country = string[6:9]
        f.write(city+","+country+"\n")

with open("y-pred-testing.txt", "w") as f:
    for pred in ypred_testing:
        string = str(pred)
        city = string[0:6]
        country = string[6:9]
        f.write(city+","+country+"\n")

