README: Description of code for NER experiment
----------------------------------------------------------------

### The following scripts are included in the "ner_experiment" folder:
- ner_experiment.py --> to train and test a 3-layer neural network for the task of Named Entity Recognition.
- ner_experiment_hyperparameter_tuning.py --> to tune the network's hyperparameters for the NER task.
- script_neural_network.py --> class definition of the neural network.
- basic_evaluation.py --> contains functions that are imported to ner_experiment.py, needed for evaluation of the neural network.

### The following files are included in the "preprocessed_data" folder:
- reuters-train-tab-stripped-preprocessed_with_features.en --> The full training data set
- gold_stripped-preprocessed_with_features.conll --> The test data set

### "Preprocessed_data" also contains a subdirectory "batches":
- The files in this directory are batches of the training data set, which can be used for 10-fold cross validation.

### The following language model should be placed in the main directory "ner_experiment":
- GoogleNews-vectors-negative300.bin.gz

HOW TO RUN THE CODE FROM THE COMMAND LINE (assuming that we are in the directory "ner_experiment"):

1) Tune the network's hyperparameters by typing the following command:

python ner.experiment.py preprocessed_data/batches/all_but_batch{n}.txt preprocessed_data/batches/batch{n}.txt GoogleNews-vectors-negative300.bin.gz

- First argument = training data file, second argument = test data file, third argument = language model
- For {n}, any number between 1 and 10 needs to be specified.
- The script gives the performance of the network, resulting from different hyperparameter combinations, as output.

2) Train and test the neural network by typing the following command:

python ner.experiment.py preprocessed_data/reuters-train-tab-stripped-preprocessed_with_features.en preprocessed_data/gold_stripped-preprocessed_with_features.conll GoogleNews-vectors-negative300.bin.gz

- First argument = training data file, second argument = test data file, third argument = language model
- The script gives the network's overall performance (accuracy), an evaluation table, and a confusion matrix as output.
- It also creates an outputfile called "results.csv", which ends up in the main directory "ner_experiment". This file contains the tokens in combination with the gold labels and the network's predicted labels, which can be used for an error analysis.
