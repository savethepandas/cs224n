#!/usr/bin/env python

import numpy as np
import random

from cs224.assignment1.q1_softmax import softmax
from cs224.assignment1.q2_gradcheck import gradcheck_naive
from cs224.assignment1.q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    norm_factor = np.sqrt((x ** 2).sum(axis=1, keepdims=True))
    x = x / norm_factor 
    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print(x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    scoreVectors = outputVectors.dot(predicted)  # step 3/4: generate sccore vectors. (V x h) * (h x 1) = (V x 1)
    scoreVectorsExp = np.exp(scoreVectors)
    outputProbs = scoreVectorsExp / scoreVectorsExp.sum()  # step 4/5: convert to prob
    targetProb = outputProbs[target]  # yhat_o
    cost = -1 * np.log(targetProb)  # step 5/6: see how well it matches actual word
    gradPred = (outputVectors * scoreVectorsExp).sum(axis=0) / scoreVectorsExp.sum() - outputVectors[target]
    gradPred = np.reshape(gradPred, (-1, 1))

    grad = predicted.T * outputProbs
    grad[target, :] = predicted.T * (targetProb - 1)  # (1 x h) ele* (V x 1) = (V x h)

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
           newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    scoreVectors = outputVectors.dot(predicted)  # (K x h) * (h x 1) = (K x 1)
    negSampleScoreVectors = scoreVectors[indices, :]
    negSampleScoreVectorsSig = sigmoid(negSampleScoreVectors)
    cost = -1 * np.log(negSampleScoreVectorsSig).sum()
    
    gradPred = -(outputVectors[indices, :] * (1 - negSampleScoreVectorsSig)).sum(axis=0)
    gradPred = np.reshape(gradPred, (-1, 1))

    grad = np.zeros(outputVectors.shape)
    predicted = np.ravel(predicted)
    for i in indices:
        grad[i, :] += -predicted * (1 - sigmoid(scoreVectors[i, :]))
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    wordVector = np.zeros((len(tokens), 1))
    wordVector[tokens[currentWord]] = 1  # step 1: generate one-hot for center word
    hiddenLayer = inputVectors.T.dot(wordVector)  # step 2: get embedded word vector. (H x V) x (V x 1) = (H x 1)

    totalGradIn = np.zeros(hiddenLayer.shape)
    for contextWord in contextWords:  # for each context word, we calc prob of each word appearing and compare
        costSingleWord, gradPred, grad = word2vecCostAndGradient(hiddenLayer, tokens[contextWord], outputVectors, dataset)
        cost += costSingleWord
        totalGradIn += gradPred
        gradOut += grad

    gradIn[tokens[currentWord], :] = totalGradIn.T
    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    wordVectors = np.zeros((len(tokens), len(contextWords)))
    for i, contextWord in enumerate(contextWords):
        wordVectors[tokens[contextWord], i] = 1  # step 1: convert context words to one-hot
    hiddenLayer = inputVectors.T.dot(wordVectors)  # step 2: get embedded word vectors. (H x V) x (V x C) = (H x C)
    predictedVector = hiddenLayer.sum(axis=1, keepdims=True)  # step 3: average the context vectors. (H x 1)

    cost, gradPred, gradOut = word2vecCostAndGradient(predictedVector, tokens[currentWord], outputVectors, dataset)
    gradPred = np.ravel(gradPred)
    for contextWord in contextWords:
        gradIn[tokens[contextWord], :] += gradPred
    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    """
    Inputs:
    word2vecModel: either skipgram or cbow
    tokens: dictionary 
    wordVectors: first half of rows represents inputVectors (V) -> hiddenLayer
                 second half is outputVectors (U) -> scoreVector
    datset: ?
    C: length of context around word (one each side)
    
    Outputs:
    cost: uses word2vecCostAndGradient (either softmax or neg sample)
    grad: gradient wrt wordVectors (inputVectors and outputVectors)
    """

    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N//2,:]
    outputVectors = wordVectors[N//2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N//2, :] += gin / batchsize / denom
        grad[N//2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        """ randomly picks one word to be center and 2*C words to be context
            the "tokens" are the vocabulary set """
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    assert gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
