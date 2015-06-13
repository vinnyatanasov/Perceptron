#
# Handles reading features, processing reviews and plotting graph
# works with the perceptron classifier
#
# Vincent Atanasov, m4va
#

import numpy
import matplotlib.pyplot as plot

class Process(object):
    """
    class containing methods concerned with reading/creating data for perceptron algorithm
    """
    # initialised to 0, will be overwritten
    feature_space_size = 0
    # folder containing data files
    data_loc = "data/"
    
    @staticmethod
    def read_features(file):
        """
        creates a set of features from a given file
        returns: set of features
        """
        path = Process.data_loc + file
        features = set()
        
        # read file and add each word to set
        with open(path, "r") as f:
            for line in f:
                for w in line.strip().split():
                    features.add(w)
        
        return features
    
    @staticmethod
    def process_features():
        """
        creates feature space from train and test files
        returns: a dictionary of unique features
        """
        features = Process.read_features("train.positive")
        features = features.union(Process.read_features("train.negative"))
        features = features.union(Process.read_features("test.positive"))
        features = features.union(Process.read_features("test.negative"))
        
        # update feature space size variable
        Process.feature_space_size = len(features)
        
        # convert set to dictionary for efficiency of indexing
        features = list(features)
        features_index = {}
        for (fid, fval) in enumerate(features):
            features_index[fval] = fid
        
        return features_index
    
    @staticmethod
    def make_vectors(file, features, label):
        """
        converts reviews to vectors consisting of 0s and 1s - the last space
        in the vector will hold the label of the instance
        returns: a data set of vectors
        """
        path = Process.data_loc + file
        data = []
        # + 1 for the label
        dimensions = Process.feature_space_size + 1
        
        # create vector representation for each line in file
        with open(path, "r") as f:
            for line in f:
                x = numpy.zeros(dimensions)
                
                # find each word in features, and add 1 to corresponding position
                for word in line.strip().split():
                    x[features[word]] = 1
                
                # last space reserved for label, then append to list
                x[-1] = label
                data.append(x)
        
        return data
    
    @staticmethod
    def plot_results(train, test):
        """
        plots a graph of the train and test results using pyplot
        args: train and test are both arrays of the same length, containing their respective results
        """
        # x axis
        xaxis = xrange(1, len(train)+1, 1)
        plot.xlabel("Number of iterations")
        plot.xticks([x for x in xaxis])
        
        # y axis
        plot.ylabel("Error rate %")
        plot.ylim([0, 35])
        
        # plot graphs
        plot.plot(xaxis, train, label="Train error rate", linewidth=3)
        plot.plot(xaxis, test, label="Test error rate", linewidth=3)
        plot.title("Error rates with averaged weight vector and bias")
        plot.legend()
        plot.grid(True)
        plot.show()
        pass
    