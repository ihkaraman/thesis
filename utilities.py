#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import preprocess
import similarities
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sentence_transformers import util, SentenceTransformer
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.multiclass import OneVsRestClassifier


balance_ratio = 0.5
random_state = 1
starting_index = 100000
np.random.seed(random_state)

# In[ ]:


def read_data(path):
    df = pd.read_csv(path)
    df['text'] = df['text'].apply(preprocess.preprocess_text)
    return df


# In[ ]:


def vectorize_data(text, model_name='stsb-roberta-large'):
    
    model = SentenceTransformer(model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vectors = model.encode(text, convert_to_tensor=False, device=device)
    
    return vectors


# In[ ]:


def calculating_class_weights(y_true):
        
        number_dim = np.shape(y_true)[1]
        weights = []
        for i in range(number_dim):
            weights.append(dict(zip([0,1], compute_class_weight('balanced', [0.,1.], y_true[:, i]))))
            # weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])))
        return weights


# In[ ]:

def binary_classifier(X_train, y_train, X_test, y_test):

    # class_weights = calculating_class_weights(y_train.values)
    
    # Linear SVM
    model = LinearSVC(class_weight='balanced')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print('+ '*50)
    print("\033[1m" + 'Binary Classifier Results' + "\033[0m")
    print("\033[1m" + 'LinearSVM' + "\033[0m")
    print('-'*30)
    acc_score = accuracy_score(y_test, preds)
    print('Exact Match Ratio: {:.2f}'.format(acc_score))
    print('-'*30)
    print("\033[1m" + 'Classification Report' + "\033[0m")
    print(classification_report(y_test, preds, target_names=list(y_test.columns)))
    print('+ '*50)
    
    return f1_score(y_test, preds)
    
    
def multilabel_classifier(X_train, y_train, X_test, y_test):

    # class_weights = calculating_class_weights(y_train.values)
    
    # Linear SVM
    model = OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1)
    model.fit(X_train, y_train.values)
    preds = model.predict(X_test)
    
    print('* '*50)
    print("\033[1m" + 'Multilabel Classifier Results' + "\033[0m")
    print("\033[1m" + 'LinearSVM' + "\033[0m")
    print('-'*30)
    hamLoss = hamming_loss(y_test.values, preds)
    print('hamLoss: {:.2f}'.format(hamLoss))
    acc_score = accuracy_score(y_test.values, preds)
    print('Exact Match Ratio: {:.2f}'.format(acc_score))
    print('-'*30)
    print("\033[1m" + 'Classification Report' + "\033[0m")
    print(classification_report(y_test.values, preds, target_names=list(y_test.columns)))
    print('* '*50)


# In[ ]:


def calculate_imb_ratio(y):

    class_ratios = (y.sum() / y.shape[0]).values
    return class_ratios


# In[ ]:

def find_batches(batch_size, num_ins):
    
    # if batching is not used
    if batch_size == -1:
        return [num_ins]
    if num_ins <= 0:
        return []
    if num_ins <= batch_size:
        return [num_ins]
    else:
        epoch_num = int(num_ins/batch_size)
        epochs = [batch_size]*epoch_num
        remain = num_ins-batch_size*epoch_num
        if remain > 0:
            epochs.append(remain)
        return epochs


def calculate_balancing_num_instance_binary(n_samples, n_total_samples, balance_ratio=balance_ratio):
    
    if n_samples/n_total_samples > balance_ratio:
        print("Be careful! Given balancing ratio is lower than the class' imbalance ratio")
        
    return int((n_total_samples*balance_ratio - n_samples)*2)


# In[ ]:


def calculate_balancing_num_instance_multiclass(y, balance_ratio):
    
    oversampling_counts = {}
    n_samples = y.shape[0]
    
    for col in y.columns:
        oversampling_counts[col] = calculate_balancing_num_instance_binary(y[col].sum(), n_samples, balance_ratio)
    
    return oversampling_counts


# In[ ]:


def find_new_instances(X_labeled, X_unlabeled, class_similarity, batch_size):
    
    # finds new instances from unlabeled set 
    # compares vector-class similarity with avg. class_sim to assign labels
    new_instances = []
    
    for idx, instance in X_unlabeled.iteritems():
        ins_sim = similarities.calculate_similarity_between_vector_and_class(instance, X_labeled)
        if ins_sim > class_similarity:
            new_instances.append(idx)
            if len(new_instances) >= batch_size:
                break
            
    return new_instances


def find_similar_columns(instance, X_labeled, y_labeled, other_columns):
    
    other_similarities = {}
    
    for col_name in other_columns:
        
        indexes = (y_labeled[col_name] == 1).index
        other_similarities[col_name]  = similarities.calculate_similarity_between_vector_and_class(instance, X_labeled.loc[indexes])
    
    return other_similarities


def find_other_labels(instance_X, X_labeled, y_labeled, class_similarities, col_name, processed_columns):
                
    # defining all labels as 0s
    new_labels = {c:0 for c in y_labeled.columns}
    # changing col_name's label as 1
    new_labels[col_name] = 1

     ### finding other labels
    other_columns = [i for i in y_labeled.columns if i not in processed_columns]
    other_similarities = find_similar_columns(instance_X, X_labeled, y_labeled, other_columns)
    for col, sim in other_similarities.items():
        if sim > class_similarities[col]:
            new_labels[col] = 1

    return new_labels


def add_instance(instance_X, instance_index, new_labels, X_labeled, y_labeled, X_unlabeled, y_unlabeled):
# can be improved
                   
    ### appending data to unlabeled set and removing it from unlabeled set
    # starting index of new instances from a big number
    instance_new_index = max(starting_index, max(X_labeled.index)) + 1
    instance_X_series = pd.Series([instance_X], index=[instance_new_index])
    instance_new_labels =pd.DataFrame(new_labels, index=[instance_new_index])
    # adding new instance to labeled set
    X_labeled = pd.concat([X_labeled, instance_X_series])
    y_labeled = pd.concat([y_labeled, instance_new_labels])
    # removing new instance from unlabeled set
    X_unlabeled = X_unlabeled.drop(instance_index)
    y_unlabeled = y_unlabeled.drop(instance_index) # note: this is for test case

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled


def prepare_new_instances(new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, class_similarities, col_name, processed_columns):
# can be improved
    validation = []
    
    for instance_index in new_instances:

        instance_X = X_unlabeled.loc[instance_index]
        instance_y = y_unlabeled.loc[instance_index] # note: this is for test case

        new_labels = find_other_labels(instance_X, X_labeled, y_labeled, class_similarities, col_name, processed_columns)
        X_labeled, y_labeled, X_unlabeled, y_unlabeled = add_instance(instance_X, instance_index, new_labels, X_labeled, y_labeled, X_unlabeled, y_unlabeled)
        # validation
        validation.append((col_name, instance_index, instance_X, (instance_y), new_labels))
    
    return validation, X_labeled, y_labeled, X_unlabeled, y_unlabeled
        
        
def oversample_dataset(num_of_new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test, sim_calculation_type, batch_size):
    
    # 1. sort required # of new instances
    # 2. calculate class similarities
    # 3. iterate over columns that requieres most # of instances to balance with batches
    # 4. calculate batches accordiing to given batch_size (-1 means no batching)
    # 5. find batch_size of instances in the unlabeled set by using the similarities
    # 6. look for other labels that can be assigned to the found instances
    # 7. the ones that have higher similarity than class similarity are assigned to that class
    # 8. add the new instances to the labeled instances for X_labeled and y_labeled
    # 9. remove the new instances from the unlabeled set
    
    # giving priority to mostly imbalanced classes
    num_of_new_instances = {k: v for k, v in sorted(num_of_new_instances.items(), key=lambda item: item[1], reverse=True)}
    
    class_similarities = similarities.calculate_overall_class_similarities(X_labeled, y_labeled, sim_calculation_type)
    
    processed_columns = []
    validation = []
    
    for col_name, num_instance in num_of_new_instances.items():
        
        processed_columns.append(col_name)
        
        # if no need to add instance, skip that column
        if num_instance <= 0:
            continue
       
        print('*'*50)
        print("\033[1m" + col_name + "\033[0m")
        f1_before = binary_classifier(X_labeled, y_labeled[col_name], X_test, y_test)
        
        indexes = (y_labeled[y_labeled[col_name] == 1]).index
        batches = find_batches(batch_size, num_instance)
        
        for batch in batches:
            
            new_instances = find_new_instances(X_labeled.loc[indexes], X_unlabeled, class_similarities[col_name], batch) 
            val_new, X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new = prepare_new_instances(new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, class_similarities, col_name, processed_columns)
            
            # check results after every batch
            f1_after = binary_classifier(X_labeled_new, y_labeled_new[col_name], X_test, y_test)
            
            if f1_after > f1_before:
                
                X_labeled, y_labeled = X_labeled_new, y_labeled_new
                validation.extend(val_new)
                f1_before = f1_after
            
            X_unlabeled, y_unlabeled = X_unlabeled_new, y_unlabeled_new
            
            print('Shapes --------------')
            print(X_labeled.shape, X_unlabeled.shape)
                
                
               
    return validation, X_labeled, y_labeled, X_unlabeled, y_unlabeled 
