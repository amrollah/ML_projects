import numpy as np
import Levenshtein

# Load training data
training = np.loadtxt("../handout/training.csv", delimiter=",", dtype=np.dtype(str))
city_code = training[:,1].astype(np.dtype(float))
country_code = training[:,2].astype(np.dtype(float))
city_names = training[:,0]

# Make all city names lowercase
city_names = np.core.defchararray.lower(city_names)

# Build a dictionary of all the words in the city names
dictionary = list()
for city_name in city_names:
    for word in city_name.split(' '):
        if not (word in dictionary) and ((len([p for p in dictionary if Levenshtein.distance(word,p) <= 2]) == 0) or len(word) <= 3) and word != '"' :
            dictionary.append(word)

# Compute bag of word features
features = np.zeros(shape=(len(city_names), len(dictionary)), dtype=np.dtype(int))
i = 0
for city_name in city_names:
    for word in city_name.split(" "):

        if word in dictionary:
            index = dictionary.index(word)
            features[i,index] += 1
        else:
            candidate_words = [(p, Levenshtein.distance(word,p)) for p in dictionary if Levenshtein.distance(word,p) <= 2]
            if len(candidate_words) > 0:
                minimal_distance = map(sorted, zip(*candidate_words))[1][0]
                minimal_word_index = zip(*candidate_words)[1].index(minimal_distance)
                minimal_word = candidate_words[minimal_word_index][0]
                index = dictionary.index(minimal_word)
                features[i,index] += 1

    i += 1

# Happy features here!
np.savetxt("features_training.txt", features, delimiter=',')
np.savetxt("labels_city_training.txt", city_code, delimiter=',')
np.savetxt("labels_country_training.txt", country_code, delimiter=',')