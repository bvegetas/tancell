# tancell
Numpy implementation of Tree Augmented Naïve Bayes algorithm
## Features:
Supports sampling from trained TAN model, especially with user-specified random number generators;
Supports computation of conditional probabilities

## Basic usage:
model=tancell() #create a tancell instance

model.fit(X)    #fit the TAN model to given discrete data, where each row of X is a sample and each column an attribute

model.predict(X)  #Returns the log-likelihood of each sample within X

model.score(X)    #Returns the log-likelihood of the whole dataset X


## Advanced usage:
model.predict_pödkget(X,L,D)    #Returns the conditional log-likelihood of each sample within X, given that the observation of D-th attribute being L
