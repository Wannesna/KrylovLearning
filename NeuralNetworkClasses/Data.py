import numpy as np
from math import floor

class Data:
    def __init__(self, inputs, labels):
        self.x = inputs # Inputs for each sample given in a MxN matrix
        self.y = labels # Labels for each sample given as a row vector of len M
        self.sampleSize = len(self.y) # M
        self.inputSize = np.size(self.x, 0) # N
       
    def info(self):
        print("The number of items in the data set is " + str(self.sampleSize) + "\n" + "inputsize is " + str(self.inputSize))

    def splitData(self, testPerc=0.3, method='random'):
        # Split the data into training and test data based on the given percentage and method default 30% randomly chosen data becomes test data, rest is training data
        if method == 'random':
            testSamples = np.sort(np.random.choice(self.x.shape[0], size=floor(testPerc*self.sampleSize),  replace=False))
            trainingSamples = [x for x in range(self.sampleSize) if x not in testSamples]
            trainingData = Data(self.x[trainingSamples, :], self.y[trainingSamples])
            testData = Data(self.x[testSamples, :], self.y[testSamples])

        return trainingData, testData


inputs = np.random.randint(5, size=(10,3))
labels = np.random.randint(2, size=(10,1))
p1 = Data(inputs, labels)

p2, p3 = p1.splitData()
