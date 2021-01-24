README: Description of code for NER experiment
----------------------------------------------------------------

### The following scripts are included in the "ner_experiment" folder:
- ner_experiment.py --> to train and test a 3-layer neural network for the task of Named Entity Recognition.
- script_neural_network.py --> class definition of the neural network.

### The following files are included in the "preprocessed_data" folder:
- reuters-train-tab-stripped-preprocessed_with_features.en --> The full training data set
- gold_stripped-preprocessed_with_features.conll --> The test data set

### "Preprocessed_data" also contains a subdirectory "batches":
- The files in this directory are batches of the training data set, which can be used for 10-fold cross validation.

### The following language model should be placed in this same directory "digit_experiment":
- GoogleNews-vectors-negative300.bin.gz

HOW TO RUN THE CODE FROM THE COMMAND LINE (assuming that we are in the directory "digit_experiment"):

1) Train and test the neural network by typing the following command:

python ner.experiment.py preprocessed_data/reuters-train-tab-stripped-preprocessed_with_features.en preprocessed_data/gold_stripped-preprocessed_with_features.conll GoogleNews-vectors-negative300.bin.gz

- First argument = training data file, second argument = test data file, third argument = language model
- When performing 10-fold cross validation, use "ner_dataset/batches/all_but_batch{n}.txt" as the training data and "ner_dataset/batches/batch{n}.txt" as the test data.
- The script gives the performance of the network, resulting from different hyperparameter combinations, as output.
