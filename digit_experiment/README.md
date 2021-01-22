README: Description of code for digit experiment
----------------------------------------------------------------

### The following scripts are included in the "digit_experiment" folder:
- digit_experiment.py --> to train and test a 3-layer neural network for the task of recognizing handwritten digits.
- script_neural_network.py --> class definition of the neural network.

### The following files should be placed in this same directory "digit_experiment":
- mnist_train.csv --> the training data (60,000 training instances)
- mnist_test.csv --> the test data (10,000 test instances)

HOW TO RUN THE CODE FROM THE COMMAND LINE (assuming that we are in the directory "digit_experiment"):

1) Train and test the neural network by typing the following command:

python digit.experiment.py mnist_train.csv mnist_test.csv

This command should give the performance of the neural network as output.
