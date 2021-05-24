texts = [
    "blue car and blue window",
    "black crow in the window",
    "i see my reflection in the window"
]


""" Binary encoding """

vocab = sorted(set(word for sentence in texts for word in sentence.split()))
print(len(vocab), vocab)

# 12 ['and', 'black', 'blue', 'car', 'crow', 'i', 'in', 'my', 'reflection', 'see', 'the', 'window']
# We have 12 distinct words in our entire corpus. So our vocabulary contains 12 words. 
# After transforming, each document will be a vector of size 12.


import numpy as np
def binary_transform(text):
    # create a vector with all entries as 0
    output = np.zeros(len(vocab))
    # tokenize the input
    words = set(text.split())
    # for every word in vocab check if the doc contains it
    for i, v in enumerate(vocab):
        output[i] = v in words 
    return output

print(binary_transform("i saw crow"))

# [0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0.]


""" CountVectorizer class to transform a collection of documents into the feature matrix """
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(binary=True)
vec.fit(texts)
print([w for w in sorted(vec.vocabulary_.keys())])

['and', 'black', 'blue', 'car', 'crow', 'in', 'my', 'reflection', 'see', 'the', 'window']


# Visualising the transformation in a table. 
# The columns are each word in the vocabulary and the rows represent the documents

import pandas as pd
pd.DataFrame(vec.transform(texts).toarray(), columns=sorted(vec.vocabulary_.keys()))



""" Counting """
# Counting is another approach to represent text as a numeric feature. 
# it is similar to Binary scheme - but instead of just checking if a word exists or not, 
# it also checks how many times a word appeared. In sklearn, CountVectorizer can be used to transform the text.

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(binary=False) # we cound ignore binary=False argument since it is default
vec.fit(texts)

import pandas as pd
pd.DataFrame(vec.transform(texts).toarray(), columns=sorted(vec.vocabulary_.keys()))

  #	and	black	blue	car	crow	in	my	reflection	see	the	window
# 0	1	0	2	1	0	0	0	0	0	0	1
# 1	0	1	0	0	1	1	0	0	0	1	1
# 2	0	0	0	0	0	1	1	1	1	1	1
