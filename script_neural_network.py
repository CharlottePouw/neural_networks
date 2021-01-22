import numpy # for mathematical functions
import scipy.special # for the sigmoid function expit()

# neural network class definition
class neuralNetwork:
    
    # initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        
        # set the number of nodes in the input, hidden and output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # define the learning rate
        self.lr = learningrate

        # initialize link weight matrices with random numbers using a normal distribution
        # 'wih' is the matrix with weights between the input and hidden layer
        # 'who' is the matrix with weights between the hidden and output layer
        # the weigths inside the arrays are w_i_j,
        # where the link is from node i in one layer to node j in the next layer 
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
                    
        # our activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)              
        pass
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        
        # convert list of inputs and targets (i.e. the desired outputs) to 2d arrays
        # these arrays are then transposed, so that they are "vertical"
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # get hidden inputs by calculating the dot product of the 'wih' matrix and the array of inputs
        hidden_inputs = numpy.dot(self.wih, inputs)
        
        # get hidden outputs by applying the activation function to the hidden inputs
        hidden_outputs = self.activation_function(hidden_inputs)
                    
        # get inputs for the final output layer by calculating the dot product of the 'who' matrix and the hidden outputs
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # get outputs from the final output layer by applying the activation function to the final inputs
        final_outputs = self.activation_function(final_inputs)
        
        # error is the (desired target output - actual output of the network)
        output_errors = targets - final_outputs
        
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers (i.e. the 'who' matrix)
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
                                        
        # update the weights for the links between the input and hidden layers (i.e. the 'wih' matrix)
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass
    
    # query the neural network
    def query(self, inputs_list):
                    
        # convert list of inputs to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
                    
        # get hidden inputs by calculating the dot product of the 'wih' matrix and the array of inputs
        hidden_inputs = numpy.dot(self.wih, inputs)
                    
        # get hidden outputs by applying the activation function to the hidden inputs
        hidden_outputs = self.activation_function(hidden_inputs)
                    
        # get inputs for the final output layer by calculating the dot product of the 'who' matrix and the hidden outputs
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # get outputs from the final output layer by applying the activation function to the final inputs
        final_outputs = self.activation_function(final_inputs)

        return final_outputs