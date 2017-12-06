# backpropagation class used for nomine name generation
#
# THIS SCRIPT WAS BASED OFF THE GITHUB REPO
# https://github.com/lazyprogrammer/machine_learning_examplesthon
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import string
np.random.seed(1)
chararray = list(string.ascii_lowercase) + ['']
class NameNN:

    def __init__(self, inputSize,hiddenSize, learning_rate = 1e-2):
        self.I = int(inputSize + 1)
        self.H = int(hiddenSize)
        self.O = 27
        self.learning_rate = learning_rate
        self.lettersAccepted = inputSize // 26
        self.words = []
        self.createLayers()

    def createLayers(self):
        self.W1 = np.random.randn(self.I, self.H)
        self.b1 = np.random.randn(self.H)
        self.W2 = np.random.randn(self.H, self.O)
        self.b2 = np.random.randn(self.O)

    def forward(self,X):
        hidden = 1 / (1 + np.exp(-X.dot(self.W1) - self.b1))
        A = hidden.dot(self.W2) + self.b2
        expA = np.exp(A)
        output = expA / expA.sum(axis=0, keepdims=True)
        return output, hidden

    def updateWeights(self,inputL, hidden, output,true):
        difference = true - output
        # print("Hidden:",hidden)
        DW2= np.outer(hidden,difference)
        Db2 = difference.sum(axis=0)
        DHidden = (difference).dot(self.W2.T) * hidden * (1 - hidden)
        Db1 = DHidden.sum(axis=0)
        DW1 =  np.outer(inputL,DHidden)
        self.W2 += self.learning_rate * DW2
        self.b2 += self.learning_rate * Db2
        self.W1 += self.learning_rate * DW1
        self.b1 += self.learning_rate * Db1

    def cost(self,T, Y):
        tot = T * (T-Y)
        return tot.sum()

    def batch(self, word):
        for i in range(len(word)):
            # print('W:',word)
            # print('i:',i)
            inputVec = np.zeros(self.I)
            realOut = np.zeros(self.O)
            inputVec[self.I-1]=i
            if i == len(word) -1:
                index = self.O - 1
            else:
                index = string.ascii_lowercase.index(word[i+1])
            realOut[index] = 1

            for j in range(self.lettersAccepted):
                if not ((i-j) + 1):
                    break
                index = string.ascii_lowercase.index(word[i-j])
                inputVec[26*j+ index] = 1
            # print('I:',inputVec)
            # print('R:',realOut)
            output, hidden = self.forward(inputVec)
            # print('O:',output)
            self.updateWeights(inputVec,hidden,output,realOut)
        return self.cost(realOut,output)

    def addWords(self,words):
        self.words += words

    def getLetter(self, word,length):
        inputVec = np.zeros(self.I)
        inputVec[self.I-1]=length
        for j in range(self.lettersAccepted):
            if not ((len(word)-j)):
                break
            index = string.ascii_lowercase.index(word[-j-1])
            inputVec[26*j+ index] = 1
        # print('W:',word)
        # print('I:',inputVec)
        output, hidden = self.forward(inputVec)
        # print('O:',output)
        letter = np.random.choice(chararray,1,p = output)
        if letter != '':
            # print(letter)
            return letter[0]
        else:
            return None

    def getWord(self, base = None):
        if base is None:
            base = ''
        rLetter = self.getLetter(base,len(base))
        while rLetter:
            base += rLetter
            rLetter = self.getLetter(base,len(base))
        return base

    def runNTimes(self,N):
        for i in range(N):
            randword = np.random.randint(0,len(self.words))
            word = self.words[randword]
            cost = self.batch(word)
            if not i % 50:
                print(word)
                print("Iteration:",i)
                print("Cost:",cost)






# determine the classification rate
# num correct / num total
# def classification_rate(Y, P):
#     n_correct = 0
#     n_total = 0
#     for i in range(len(Y)):
#         n_total += 1
#         if Y[i] == P[i]:
#             n_correct += 1
#     return float(n_correct) / n_total






def main():
    # create the data
    Nclass = 500
    D = 2 # dimensionality of input
    M = 3 # hidden layer size
    K = 3 # number of classes

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)
    # turn Y into an indicator matrix for training
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    # let's see what it looks like
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()

    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 1e-3
    costs = []
    for epoch in range(1000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print("cost:", c, "classification_rate:", r)
            costs.append(c)

        # this is gradient ASCENT, not DESCENT
        # be comfortable with both!
        # oldW2 = W2.copy()
        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.show()
def getNames(filename):
    with open(filename,"r") as f:
        wordlist = [r.lower().replace("\n",'') for r in f]
    return wordlist


if __name__ == '__main__':
    wordlist = ['abababab','bababab','cdcdcdcd','dcdcdcdc','efefefefe','fefefefe']
    wordlist2 = ['mark', 'jason', 'nick','will','kevin','sam','eathon','ben', 'allen']
    inputSize = 26 * 3
    hiddenSize = 26 * 1.5
    nn = NameNN(inputSize,hiddenSize)
    wordlist3 = getNames('malenames.txt')
    nn.addWords(wordlist3)
    nn.runNTimes(10000)
    for i in chararray:
        print(nn.getWord(base = i))
