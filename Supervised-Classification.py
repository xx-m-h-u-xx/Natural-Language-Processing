''' Feature Extraction prog '''
''' Classification is the task of choosing the correct class label for a given input '''


"""The first step in creating a classifier is deciding what features of the input are relevant, 
and how to encode those features. The following feature extractor function builds a dictionary
containing relevant information about a given name:"""

# Gender Identification 	
def gender_features(word):
    return {'last_letter': word[-1]}

""" The returned dictionary, known as a feature set, maps from feature names to its values. 
- Feature names are case-sensitive strings that typically provide a short human-readable description of the feature (i.e. 'last_letter')
- Feature values are values with simple types, such as booleans, numbers, and strings  
# gender_features('Shrek')
# {'last_letter': 'k'}


""" Prepare a list of examples & corresponding class labels """

from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                    [(name, 'female') for name in names.words('female.txt')])
import random
random.shuffle(labeled_names)


""" Feature extractor then processes the names data; Divides resulting list of feature sets into training set and test set. 
The training set is used to train a new "naive Bayes" classifier """

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)


""" Tests alternative names not appeared in training data:

classifier.classify(gender_features('Neo'))
# 'male'

classifier.classify(gender_features('Trinity'))
# 'female'


""" Systematically evaluates the classifier on much larger quantity of unseen data: """

print(nltk.classify.accuracy(classifier, test_set))
# 0.77


""" Examines the classifier to determine which features it found MOST EFFECTIVE for DISTINGUSIHING names' genders:

classifier.show_most_informative_features(5)
''' Most Informative Features
             last_letter = 'a'            female : male   =     33.2 : 1.0
             last_letter = 'k'              male : female =     32.6 : 1.0
             last_letter = 'p'              male : female =     19.7 : 1.0
             last_letter = 'v'              male : female =     18.6 : 1.0
             last_letter = 'f'              male : female =     17.3 : 1.0  '''


