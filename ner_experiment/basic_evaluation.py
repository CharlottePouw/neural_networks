import csv
import sys
import pandas as pd
from collections import defaultdict, Counter

def extract_annotations(inputfile, annotationcolumn, delimiter='\t'):
    '''
    This function extracts annotations represented in the conll format from a file
    
    :param inputfile: the path to the conll file
    :param annotationcolumn: the name of the column in which the target annotation is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type annotationcolumn: string
    :type delimiter: string
    :returns: the annotations as a list
    '''
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    conll_input = pd.read_csv(inputfile, sep=delimiter)
    annotations = conll_input[annotationcolumn].tolist()
    return annotations

def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output
    
    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations
    
    :returns: a countainer providing the counts for each predicted and gold class pair
    '''
    # TIP on how to get the counts for each class
    # https://stackoverflow.com/questions/49393683/how-to-count-items-in-a-nested-dictionary, last accessed 22.10.2020
    evaluation_counts = defaultdict(Counter)
    
    for gold_annotation, machine_annotation in zip(goldannotations, machineannotations):
        evaluation_counts[gold_annotation][machine_annotation] += 1
        
    return evaluation_counts

def get_tp_fp_fn(evaluation_counts):
    '''
    Get the true positives, false positives and false negatives per class and return them in a dictionary
    '''
    tp_fp_fn_per_class = defaultdict(Counter)
    
    # Calculate true positives
    for annotation in evaluation_counts.keys():
        true_positives = evaluation_counts[annotation][annotation]
        tp_fp_fn_per_class[annotation]['tp'] += true_positives
    
        # Calculate false positives
        for other_annotation in evaluation_counts.keys():
            if other_annotation != annotation:
                false_positives = evaluation_counts[other_annotation][annotation]
                tp_fp_fn_per_class[annotation]['fp'] += false_positives
                
        # Calculate false negatives
        for predicted_label in evaluation_counts[annotation].keys():
            if predicted_label != annotation:
                false_negatives = evaluation_counts[annotation][predicted_label]
                tp_fp_fn_per_class[annotation]['fn'] += false_negatives
        
        # If there are no true positives, false positives or false negatives, we set the value to 0
        if 'tp' not in tp_fp_fn_per_class[annotation].keys():
            true_positives = 0
            tp_fp_fn_per_class[annotation]['tp'] = 0
            
        if 'fp' not in tp_fp_fn_per_class[annotation].keys():
            false_positives = 0
            tp_fp_fn_per_class[annotation]['fp'] = 0
            
        if 'fn' not in tp_fp_fn_per_class[annotation].keys():
            false_negatives = 0
            tp_fp_fn_per_class[annotation]['fn'] = 0
            
    return tp_fp_fn_per_class

def calculate_precision_recall_fscore(evaluation_counts):
    '''
    Calculate precision, recall and fscore for each class and return them in a dictionary
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns the precision, recall and f-score of each class in a container
    '''
    precision_recall_fscore_per_class = dict()
    
    # get the true positives, false positives and false negatives per class in a dictionary
    tp_fp_fn_per_class = get_tp_fp_fn(evaluation_counts)
    
    for annotation, values in tp_fp_fn_per_class.items():
        
        # Get the true positives, false positives and false negatives from our dictionary
        tp = values['tp']
        fp = values['fp']
        fn = values['fn']
        
        # Calculate precision, recall and f-score only if there are more than 0 true positives, 
        # such that we avoid division by 0
        if tp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            fscore = 2 * ((precision * recall) / (precision + recall))
            
        # Otherwise, set the values to 0
        else:
            precision = 0
            recall = 0
            fscore = 0
        precision_recall_fscore_per_class[annotation] = {"precision": precision, "recall": recall, "f-score": fscore}

    return precision_recall_fscore_per_class

def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :prints out a confusion matrix
    '''
    confusion_matrix = pd.DataFrame.from_dict({i: evaluation_counts[i]
                                              for i in evaluation_counts.keys()},
                                              orient='index')
    print(confusion_matrix)
    
def carry_out_evaluation(gold_annotations, systemfile, systemcolumn, delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)
    
    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    
    returns evaluation information for this specific system
    '''
    system_annotations = extract_annotations(systemfile, systemcolumn, delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)
    
    return evaluation_outcome

def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems
    
    :param evaluations: the outcome of evaluating one or more systems
    '''
    #https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    evaluations_pddf = pd.DataFrame.from_dict({(i,j): evaluations[i][j]
                                              for i in evaluations.keys()
                                              for j in evaluations[i].keys()},
                                             orient='index')
    print(evaluations_pddf)
    print(evaluations_pddf.to_latex())
    
def run_evaluations(goldfile, goldcolumn, systems):
    '''
    Carry out standard evaluation for one or more system outputs
    
    :param goldfile: path to file with goldstandard
    :param goldcolumn: indicator of column in gold file where gold labels can be found
    :param systems: required information to find and process system output
    :type goldfile: string
    :type goldcolumn: integer
    :type systems: list (providing file name, information on tab with system output and system name for each element)
    
    :returns the evaluations for all systems
    '''
    evaluations = {}
    #not specifying delimiters here, since it corresponds to the default ('\t')
    gold_annotations = extract_annotations(goldfile, goldcolumn)
    for system in systems:
        sys_evaluation = carry_out_evaluation(gold_annotations, system[0], system[1])
        evaluations[system[2]] = sys_evaluation
        
    return evaluations

def identify_evaluation_value(system, class_label, value_name, evaluations):
    '''
    Return the outcome of a specific value of the evaluation
    
    :param system: the name of the system
    :param class_label: the name of the class for which the value should be returned
    :param value_name: the name of the score that is returned
    :param evaluations: the overview of evaluations
    
    :returns the requested value
    '''
    return evaluations[system][class_label][value_name]

def create_system_information(system_information):
    '''
    Takes system information in the form that it is passed on through sys.argv or via a settingsfile
    and returns a list of elements specifying all the needed information on each system output file to carry out the evaluation.
    
    :param system_information is the input as from a commandline or an input file
    '''
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    systems_list = [system_information[i:i + 3] for i in range(0, len(system_information), 3)]
    return systems_list

def main():
    
    my_args = sys.argv # arguments are gold file, gold column, system file, system column and system name
    
    system_info = create_system_information(my_args[3:])
    evaluations = run_evaluations(my_args[1], my_args[2], system_info)
    provide_output_tables(evaluations)

if __name__ == '__main__':
    main()