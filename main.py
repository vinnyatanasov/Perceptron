#
# Perceptron classifier main file
#
# Vincent Atanasov, m4va
#

import argparse
from perceptron import Perceptron

if __name__ == "__main__":
    # setup command line arguments for iterations and loop mode
    parser = argparse.ArgumentParser(description="Perceptron", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--num_iterations", help="number of iterations the algorithm will run for", default=15, type=int)
    parser.add_argument("-l", "--loop", help="use this mode to loop the program, incrementing the number of iterations each time and plotting the results using numpy", action="store_true")
    args = parser.parse_args()
    
    print "--PERCEPTRON CLASSIFIER--"
    
    # run perceptron, passing in boolean indicating loop or not and number of iterations
    Perceptron.run(args.loop, args.num_iterations)