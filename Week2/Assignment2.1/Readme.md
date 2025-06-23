Manual TF-IDF gives the scores

Sentence 1:

TF-IDF for word 'the': 0.0

TF-IDF for word 'is': 0.08109302162163289

TF-IDF for word 'star': 0.21972245773362198

TF-IDF for word 'a': 0.08109302162163289

TF-IDF for word 'sun': 0.08109302162163289

Sentence 2:

TF-IDF for word 'the': 0.0

TF-IDF for word 'is': 0.08109302162163289

TF-IDF for word 'satellite': 0.21972245773362198

TF-IDF for word 'a': 0.08109302162163289

TF-IDF for word 'moon': 0.08109302162163289

Sentence 3:

TF-IDF for word 'and': 0.15694461266687282

TF-IDF for word 'the': 0.0

TF-IDF for word 'are': 0.15694461266687282

TF-IDF for word 'moon': 0.05792358687259491

TF-IDF for word 'bodies': 0.15694461266687282

TF-IDF for word 'celestial': 0.15694461266687282

TF-IDF for word 'sun': 0.05792358687259491



While the Countvectorizer and TF-IDF together gives the scores in the form of a 

TF-IDF Output using Scikit-Learn:

['and'      'are'      'bodies'   'celestial' 'is'       'moon'    'satellite' 'star'     'sun'      'the']

[[0.         0.         0.         0.         0.4804584  0.         0.         0.63174505 0.4804584  0.37311881]

 [0.         0.         0.         0.         0.4804584  0.4804584  0.63174505 0.         0.         0.37311881]
 
 [0.4261835  0.4261835  0.4261835  0.4261835  0.         0.32412354 0.         0.         0.32412354 0.25171084]]


The scores for the words 'the' differ as in our normal logarithmic formula to calculate IDF there is no smoothing term, the smoothing is something which helps preventing division by zero inside the log
Moreover the values calculated manually are scaled down, the scikit-learn model follows normalization of the values

