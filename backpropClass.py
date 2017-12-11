# backpropagation class used for nomine name generation
#
# THIS SCRIPT WAS BASED OFF THE GITHUB REPO
# https://github.com/lazyprogrammer/machine_learning_examplesthon
from __future__ import print_function, division
from builtins import range
import csv
import re
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import string
np.random.seed(1)
chararray = list(string.ascii_lowercase) + ['']

class NameNN:
    def __init__(self, inputSize,hiddenSize, learning_rate = 1e-2):
        self.I = int(inputSize + 1)#input layer size added one to keep track of current word length
        self.H = int(hiddenSize)# hiddden layer size
        self.O = 27#output layer size
        self.learning_rate = learning_rate#duh
        self.lettersAccepted = inputSize // 26#how many letters to look back
        self.words = []#database of words
        self.createLayers()#randomly assign values to all weights

    def createLayers(self):
        self.W1 = np.random.randn(self.I, self.H)#weights between input and hidden
        self.b1 = np.random.randn(self.H)#bias for hidden layer
        self.W2 = np.random.randn(self.H, self.O)  #weights between hidden adn output
        self.b2 = np.random.randn(self.O)#bias for output layers

    def forward(self,X):
        hidden = 1 / (1 + np.exp(-X.dot(self.W1) - self.b1))#calculate hidden node values
        A = hidden.dot(self.W2) + self.b2#calculate output node values
        expA = np.exp(A)#take exp
        output = expA / expA.sum(axis=0, keepdims=True)#make sure everything adds up to 1
        return output, hidden

    def updateWeights(self,inputL, hidden, output,true):
        difference = true - output#error for each output node
        # print("Hidden:",hidden)
        DW2= np.outer(hidden,difference)#change W2 weights according to error and hideen values
        Db2 = difference.sum(axis=0)#change the bias according to error
        DHidden = (difference).dot(self.W2.T) * hidden * (1 - hidden) #calculate error of the hidden nodes
        Db1 = DHidden.sum(axis=0)#update biases for hidden nodes based on hidden error
        DW1 =  np.outer(inputL,DHidden)#finally update weights for hidden nodes based on input values
        self.W2 += self.learning_rate * DW2#actually change everything
        self.b2 += self.learning_rate * Db2
        self.W1 += self.learning_rate * DW1
        self.b1 += self.learning_rate * Db1

    def cost(self,T, Y):
        tot = T * (T-Y)#calculate cost(somewhat meaningless for word generation)
        return tot.sum()

    def batch(self, word):
        '''Update the model according to a word'''
        for i in range(len(word)):#loop through letters of the word
            inputVec = np.zeros(self.I)#init input and True values
            realOut = np.zeros(self.O)
            inputVec[self.I-1]=i#set last value to the length of the word
            if i == len(word) -1:#if last letter
                index = self.O - 1#set output to node meaning end the word
            else:
                index = string.ascii_lowercase.index(word[i+1])#set node output to letter index
            realOut[index] = 1

            for j in range(self.lettersAccepted):#for each letter accepted
                if not ((i-j) + 1):#break if at the beginning of the word
                    break
                index = string.ascii_lowercase.index(word[i-j])#set the correct index in the input to 1
                inputVec[26*j+ index] = 1
            output, hidden = self.forward(inputVec)#calculate letter distribution
            self.updateWeights(inputVec,hidden,output,realOut)#BACKPROP
        return self.cost(realOut,output)#and return cost

    def addWords(self,words):
        '''update repository of words'''
        self.words += words

    def getLetter(self, word,length):
        '''use NN to choose next letter'''
        inputVec = np.zeros(self.I)#this is copy and pasted from above to
        #determine what the correct input vector is
        inputVec[self.I-1]=length
        for j in range(self.lettersAccepted):
            if not ((len(word)-j)):
                break
            index = string.ascii_lowercase.index(word[-j-1])
            inputVec[26*j+ index] = 1
        output, hidden = self.forward(inputVec)

        letter = np.random.choice(chararray,1,p = output)#based on the output,
        #select random letter from output distribution
        if letter != '':
            return letter[0]
        else:
            return None

    def getWord(self, base = None):
        '''iteratively call get letter to build a word'''
        if base is None:#set start of word if one is given
            base = ''
        rLetter = self.getLetter(base,len(base))
        while rLetter:#get letters and add them to the word
            base += rLetter
            rLetter = self.getLetter(base,len(base))
        return base

    def runNTimes(self,N):
        for i in range(N):#train the model n iterations
            randword = np.random.randint(0,len(self.words))
            word = self.words[randword]
            cost = self.batch(word)
            if not i % 50:
                print(word)
                print("Iteration:",i)
                print("Cost:",cost)

def getNamestxt(filename):
    '''build database from text file where it only takes first word from
    each line'''
    with open(filename,"r") as f:
        wordlist = [r.lower().replace("\n",'') for r in f]
    return wordlist

def getNamescsv(filename):
    '''build database from csv where it only takes the first word ff first column'''
    wordlist = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            f = str(row[0])
            # print(type(f))
            f = re.split("[, '\-!?:.$12368&]+", f)
            # print(f)

            if len(f[0]) > 2: wordlist.append(f[0].lower())
            # for i in f[0].lower():
            #     if i not in string.ascii_lowercase:
            #         print(i)
    return wordlist


if __name__ == '__main__':
    wordlist = ['abababab','bababab','cdcdcdcd','dcdcdcdc','efefefefe','fefefefe']
    wordlist2 = ['mark', 'jason', 'nick','will','kevin','sam','ethan','ben', 'allen']
    inputSize = 26 * 3
    hiddenSize = 26 * 1.5
    nn = NameNN(inputSize,hiddenSize, learning_rate = 1e-2)
    # wordlist3 = getNamestxt('femalenames.txt')
    wordlist3 = getNamescsv('dognames.csv')
    print(wordlist3)
    nn.addWords(wordlist3)
    nn.runNTimes(400000)
    for i in chararray:
        print(nn.getWord(base = i))
        print(nn.getWord(base = i))
        print(nn.getWord(base = i))
