""" Computing TF-IDF ratings """
# TF-IDF assigns more weight to less frequently occurring words
# rather than frequently occurring ones. It is based on the assumption that less frequently occurring words are more important.

# TF-IDF consists of two parts:
# Term frequency which is same as Counting method we saw before
# Inverse document frequency: This is responsible for reducing the weights of words that occur frequently 
# and increasing the weights of words that occur rarely.

""" NLP Text analytical techniques to transform text into a numeric feature space """
texts = [
    "blue car and blue window",
    "black crow in the window",
    "i see my reflection in the window"
]


from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
vec.fit(texts)

import pandas as pd
pd.DataFrame(vec.transform(texts).toarray(), columns=sorted(vec.vocabulary_.keys()))


  # 	and	black	blue	car	crow	in	my	reflection	see	the	window
# 0	0.396875	0.000000	0.793749	0.396875	0.000000	0.000000	0.00000	0.00000	0.00000	0.000000	0.234400
# 1	0.000000	0.534093	0.000000	0.000000	0.534093	0.406192	0.00000	0.00000	0.00000	0.406192	0.315444
# 2	0.000000	0.000000	0.000000	0.000000	0.000000	0.358291	0.47111	0.47111	0.47111	0.358291	0.278245
