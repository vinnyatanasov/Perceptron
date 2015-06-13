#
# Perceptron classifier
# uses process class to handle data 
#
# Vincent Atanasov, m4va
#

import numpy
import random
from process import Process

class Perceptron(object):
    """
    class contains all methods for the perceptron classifier
    """
    
    @staticmethod
    def activation(w, b, instance):
        """
        computes activation for given instance, using given weight vector and bias
        """
        return (numpy.dot(w, instance) + b)
    
    @staticmethod
    def run(run_on_loop, num_iterations):
        """
        handles running perceptron in two ways, on a loop or just once
        preliminary setup work is done independent of the mode
        args: run_on_loop is a boolean indicating whether to loop or not; num_iterations is how many times to train
        """
        # create feature space
        features = Process.process_features()
        
        # convert train data to vectors
        train_data = Process.make_vectors("train.positive", features, 1)
        train_data.extend(Process.make_vectors("train.negative", features, -1))
        
        # convert test data to vectors
        test_data = Process.make_vectors("test.positive", features, 1)
        test_data.extend(Process.make_vectors("test.negative", features, -1))
        
        # train perceptron - returns weights, biases and errors from each iteration
        print "Training..."
        train_results = Perceptron.train(train_data, num_iterations)
        print "End of training."
        print "----"
        
        # grab the information returned by the training
        weights_arr = train_results.get("weights")
        bias_arr = train_results.get("bias")
        train_errors = train_results.get("errors")
        
        test_errors = []
        print "Testing..."
        
        if (run_on_loop):
            # loop and graph mode, runs perceptron until reaches num_iterations
            # each iteration uses a different set of weights and bias, corresponding to those produced in training
            n = 0
            while n < num_iterations:
                print "Iteration [{0}]".format(n+1)
                # run test perceptron and save results in array for plotting
                test_errors.append(Perceptron.test(test_data, weights_arr[n], bias_arr[n]))
                n += 1
            
            # plot graph of train and test errors
            Process.plot_results(train_errors, test_errors)
        else:
            # simple mode, runs perceptron once
            # uses only the final weight vector and bias
            test_errors = Perceptron.test(test_data, weights_arr[-1], bias_arr[-1])
        
        print "End of testing."
        print "----"
        pass
    
    @staticmethod
    def train(dataset, max_iterations):
        """
        trains perceptron using dataset supplied
        args: dataset is the train dataset; max_iterations is number of times to train
        returns: a dictionary containing weights, bias and errors
        """
        # initialise weight vector and bias, and cached weights vector and bias
        w = numpy.zeros(Process.feature_space_size)
        cached_w = numpy.zeros(Process.feature_space_size)
        b = 0
        cached_b = 0
        count = 0
        
        # arrays to hold weights, biases and errors
        weights_arr = []
        bias_arr = []
        train_errors = []
        
        # for each iteration until the max specified
        for i in xrange(max_iterations):
            print "Iteration [{0}]".format(i+1)
            
            # shuffle data on each iteration
            random.shuffle(dataset)
            
            error_count = 0
            
            # for each train instance
            for data in dataset:
                # first increment counter
                count += 1
                
                # label is last item in array, vector is the rest
                vector = data[:-1]
                label = data[-1]
                
                # compute activation
                a = Perceptron.activation(w, b, vector)
                
                # if error, update weights and biases (normal and cached)
                if (label * a <= 0):
                    w += vector * label
                    b += label
                    cached_w += vector * label * count
                    cached_b += label * count
                    error_count += 1
            
            # compute average weight vector and bias using cached values and count, then add to arrays
            avg_w = w - ((1 / float(count)) * cached_w)
            avg_b = b - ((1 / float(count)) * float(cached_b))
            weights_arr.append(avg_w.copy())
            bias_arr.append(avg_b)
            
            # compute error percentage, store and print
            error_percentage = (float(error_count) / float(count)) * 100
            train_errors.append(error_percentage)
            print str(error_count) + " misclassifications"
            print "Train error rate: {:.2f}%".format(error_percentage)
        
        # return weights, bias and errors arrays
        return {"weights":weights_arr,
                "bias":bias_arr,
                "errors":train_errors}
    
    @staticmethod
    def test(dataset, w, b):
        """
        tests perceptron with dataset supplied and prints results
        args: dataset is the test dataset; w is the weight vector; b is the bias
        returns: percentage of errors
        """
        pos_error_count = 0
        neg_error_count = 0
        total_pos = 0
        
        # for each test instance
        for data in dataset:
            # label is last item in array, vector is the rest
            vector = data[:-1]
            label = data[-1]
            
            # compute activation and take sign as prediction
            a = Perceptron.activation(w, b, vector)
            prediction = numpy.sign(a)
            
            # keep track of errors as we progress
            # if positive instance
            if label == 1:
                total_pos += 1
                if label != prediction: pos_error_count += 1
            # else negative instance
            else:
                if label != prediction: neg_error_count += 1
        
        # compute and print error percentages
        total = len(dataset)
        pos_percentage = (float(pos_error_count) / float(total_pos)) * 100
        neg_percentage = (float(neg_error_count) / float(total - total_pos)) * 100
        error_percentage = (float(pos_error_count + neg_error_count) / float(total)) * 100
        print "Positive test error rate: {:.2f}%".format(pos_percentage)
        print "Negative test error rate: {:.2f}%".format(neg_percentage)
        print "Combined test error rate: {:.2f}%".format(error_percentage)
        
        # return just overall error percentage
        return (error_percentage)
    