from script_neural_network import neuralNetwork # import neuralNetwork class
import sys # to specify files in the command line
import gensim
from gensim.models import KeyedVectors # to load the language model
import csv # to read in conll files
import numpy # for mathematical functions
from sklearn import preprocessing # to use label encoder
from sklearn.preprocessing import minmax_scale # to rescale word embedding vectors
import operator # to calculate the best performance
import pandas as pd # to put results in a dataframe

# needed for evaluation of neural network
from collections import Counter, defaultdict
from basic_evaluation import obtain_counts, get_tp_fp_fn, calculate_precision_recall_fscore, provide_confusion_matrix, provide_output_tables

def extract_embeddings_as_features_and_gold(trainingfile, word_embedding_model):
    '''
    Function that extracts features and gold labels from file with training data using word embeddings
    
    :param trainingfile: path to training file
    :param word_embedding_model: a pretrained word embedding model
    
    :type trainingfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return gold_labels: list of gold labels
    '''
    features = []
    gold_labels = []
    
    # read in conll file as csvreader
    conllinput = open(trainingfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    
    for row in csvreader:
        if len(row) != 0:
            if row[0] in word_embedding_model:
                vector = word_embedding_model[row[0]]
                # rescale the vector to match the output of the sigmoid function
                # Fina posted her code on piazza, I found this function there (https://piazza.com/class/khd82wacg9i28g?cid=68)
                rescaled_vector = minmax_scale(vector, feature_range=(0.01, 0.99))
            else:
                rescaled_vector = [0.01]*300
            features.append(rescaled_vector)
        
            # gold labels are in the third column
            gold_labels.append(row[3])
        
    return features, gold_labels

def encode_labels(labels):
    '''
    Function that encodes string labels into numeric values
    
    :param labels: list of string labels
    :type labels: list (elements are strings)
    :return encoded_labels: list of encoded labels (elements are integers)
    '''
    
    # initialize label encoder
    label_encoder = preprocessing.LabelEncoder()
    
    # feed list of labels to encoder
    label_encoder.fit(labels)
    
    # transform labels into numeric values
    encoded_labels = label_encoder.transform(labels)

    return encoded_labels

def decode_labels(labels, encoded_labels):
    '''
    Function that decodes numerical labels into original string labels
    
    :param labels: list of original string labels
    :param encoded_labels: list of numerical labels
    
    :type labels: list (elements are strings)
    :type encoded_labels: list (elements are integers)
    
    :return decoded_labels: list of decoded labels (elements are strings)
    '''
    
    # initialize label encoder
    label_encoder = preprocessing.LabelEncoder()
    
    # feed list of labels to encoder
    label_encoder.fit(labels)
    
    # transform labels into numeric values
    decoded_labels = label_encoder.inverse_transform(encoded_labels)

    return decoded_labels
        

def main():
    
    argv = sys.argv
    trainingfile = argv[1]
    testfile = argv[2]
    language_model = argv[3]
    
    ### PREPARE INPUT FOR NEURAL NETWORK ###
    
    # Load word embedding model
    language_model = gensim.models.KeyedVectors.load_word2vec_format(language_model, binary=True)
        
    # extract the training tokens as word embeddings, and also extract the training labels
    training_features, training_labels = extract_embeddings_as_features_and_gold(trainingfile, language_model)
    
    # encode the training labels into numeric values
    encoded_training_labels = encode_labels(training_labels[1:])

    # extract the test tokens as word embeddings, also extract matching labels
    test_features, test_labels = extract_embeddings_as_features_and_gold(testfile, language_model)

    # encode the test labels into numeric values
    encoded_test_labels = encode_labels(test_labels[1:])
    
    ### INITIALIZE THE NETWORK ###
        
    # choose number of input, hidden and output nodes
    input_nodes = 300 # word embeddings are of 300 dimensions, so we have 300 inputs each time
    hidden_nodes = 100 # optimal value given hyperparameter tuning
    output_nodes = 5 # we have 5 labels: LOC, PER, ORG, MISC and O

    # choose learning rate
    learning_rate = 0.1 # optimal value given hyperparameter tuning

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    ### TRAIN THE NEURAL NETWORK ###

    # epochs is the number of times the training data set is used for training (multiple runs improves performance)
    epochs = 15 # optimal value given hyperparameter tuning

    for e in range(epochs):
        # iterate over all the training instances and corresponding training labels
        for embedding, label in zip(training_features[1:], encoded_training_labels):

            # the embedding is a list of 300 inputs for the network
            inputs = embedding

            # create an array of the length of output_nodes (i.e. 5) with 0.01 as its values
            targets = numpy.zeros(output_nodes) + 0.01

            # set the index matching the label in the 'targets' array to 0.99
            targets[label] = 0.99

            # train the neural network with the inputs and targets
            n.train(inputs, targets)

    ### TEST THE NEURAL NETWORK ###
        
    # needed for confusion matrix
    evaluation_counts = defaultdict(Counter)
    
    # to keep score of the network's predicted labels
    networks_labels = []
    
    # iterate over all the test instances and corresponding test labels
    for embedding, correct_label in zip(test_features[1:], encoded_test_labels):

        # the embedding is a list of 300 inputs for the network
        inputs = embedding

        # let the network calculate outputs
        outputs = n.query(inputs)

        # the index of the highest value in the 'outputs' array corresponds to the network's predicted label
        networks_label = numpy.argmax(outputs)
        
        # add network's prediction to list
        networks_labels.append(networks_label)

    ### EVALUATE NEURAL NETWORK ###
    
    # decode network's predicted labels
    decoded_labels = decode_labels(test_labels[1:], networks_labels)
    
    # provide evaluation table and confusion matrix
    evaluation_counts = obtain_counts(test_labels[1:], decoded_labels)
    evaluations = calculate_precision_recall_fscore(evaluation_counts)
    provide_output_tables(evaluations)
    provide_confusion_matrix(evaluation_counts)

if __name__ == '__main__':
    main()