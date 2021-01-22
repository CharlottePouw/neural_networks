README: Description of code for NER experiment
----------------------------------------------------------------

### The following scripts are included in the "ner_experiment" folder:
- ner_experiment.py --> to train and test a 3-layer neural network for the task of Named Entity Recognition.
- script_neural_network.py --> class definition of the neural network.

### The following files are included in the "preprocessed_data" folder:
- reuters-train-tab-stripped-preprocessed_with_features.en --> The training data set
- gold_stripped-preprocessed_with_features.conll --> The test data set

### The following language model should be placed in this same directory "digit_experiment":
- GoogleNews-vectors-negative300.bin.gz

HOW TO RUN THE CODE FROM THE COMMAND LINE (assuming that we are in the directory "digit_experiment"):

1) Train and test the neural network by typing the following command:

python ner.experiment.py preprocessed_data/reuters-train-tab-stripped-preprocessed_with_features.en preprocessed_data/gold_stripped-preprocessed_with_features.conll GoogleNews-vectors-negative300.bin.gz

This command should give the performance of the neural network as output.
