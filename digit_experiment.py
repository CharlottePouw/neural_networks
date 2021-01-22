import numpy # for mathematical functions
import scipy.special # for the sigmoid function expit()
import sys # to specify files in the command line
from script_neural_network import neuralNetwork # import neuralNetwork class

def main():
    
    argv = sys.argv
    trainingfile = argv[1]
    testfile = argv[2]
    
    ### INITIALIZE THE NEURAL NETWORK ###
    
    # choose number of input, hidden and output nodes
    input_nodes = 784 # we have 28 x 28 pixels, so 784 inputs in total
    hidden_nodes = 100 # value between 784 and 10, such that the network will summarize key features
    output_nodes = 10 # we have 10 possible labels: the digits 0 up to 9

    # choose learning rate
    learning_rate = 0.1

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    ### TRAIN THE NEURAL NETWORK ###
    
    # open the MNIST training data file and load the training instances into a list
    training_data_file = open(trainingfile, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # epochs is the number of times the training data set is used for training (multiple runs improves performance)
    epochs = 5

    for e in range(epochs):
        # go through all records in the training data list
        for record in training_data_list:

            # split the record based on commas
            all_values = record.split(',')

            # rescale input colour values to range 0.01 - 1.0, such that they match the output of the logistic function
            # the range is 0 - 255 at first, so we divide by 255 to get the range 0 - 1
            # we then multiply by 0.99 to get the range 0 - 0.99
            # finally, we add 0.01 to avoid zero values, so we end up with the range 0.01 - 1.0
            # we skip the first value, since that is the training target label
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # create an array of the length of output_nodes (i.e. 10) with 0.01 as its values
            targets = numpy.zeros(output_nodes) + 0.01

            # convert first element of the record (i.e. the training target label) into integer
            # and set the value of the matching index of the 'targets' array to 0.99
            targets[int(all_values[0])] = 0.99

            # train the neural network with the inputs and targets
            n.train(inputs, targets)
            pass
        pass
    
    ### TEST THE NEURAL NETWORK ###
    
    # open the MNIST test data file and load the test instances into a list
    test_data_file = open(testfile, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:

        # split the record based on commas
        all_values = record.split(',')

        # correct answer is first value
        correct_label = int(all_values[0])

        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # query the network
        outputs = n.query(inputs)

        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)

        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass
        pass
    
    ### EVALUATE NEURAL NETWORK ###
    
    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print ("performance =", scorecard_array.sum() / scorecard_array.size)

if __name__ == '__main__':
    main()