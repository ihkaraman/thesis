#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import preprocess
import similarities
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import hamming_loss, accuracy_score, classification_report
from sentence_transformers import util, SentenceTransformer
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


def classifier(X_train, y_train, X_test, y_test):

    # class_weights = calculating_class_weights(y_train.values)
    
    # Linear SVM
    model = OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1)
    model.fit(X_train, y_train.values)
    preds = model.predict(X_test)
    
    print("\033[1m" + 'LinearSVM results: ' + "\033[0m")
    print('-'*30)
    hamLoss = hamming_loss(y_test.values, preds)
    print('hamLoss: {:.2f}'.format(hamLoss))
    acc_score = accuracy_score(y_test.values, preds)
    print('Exact Match Ratio: {:.2f}'.format(acc_score))
    print('-'*30)
    print("\033[1m" + 'Classification Report' + "\033[0m")
    print(classification_report(y_test.values, preds, target_names=list(y_test.columns)))


# In[ ]:


def calculate_imb_ratio(y):

    class_ratios = (y.sum() / y.shape[0]).values
    return class_ratios


# In[ ]:


def calculate_balancing_num_instance_binary(n_samples, n_total_samples, balance_ratio=balance_ratio):
    
    if n_samples/n_total_samples > balance_ratio:
        print("Be careful! Given balancing ratio is lower than the class' imbalance ratio")
        
    return int((n_total_samples*balance_ratio - n_samples)*2)


# In[ ]:


def calculate_balancing_num_instance_multiclass(y, balance_ratio):
    
    oversampling_counts = {}
    n_samples = y.shape[0]
    n_classes = y.shape[1]
    
    for col in y.columns:
        oversampling_counts[col] = calculate_balancing_num_instance_binary(y[col].sum(), n_samples, balance_ratio)
    
    return oversampling_counts


# In[ ]:


def find_new_instances(X_labeled, X_unlabeled, class_similarity):
    
    new_instances = []
    
    for idx, instance in X_unlabeled.iteritems():
        avg_sim = similarities.calculate_similarity_between_vector_and_class(instance, X_labeled)
        if avg_sim > class_similarity:
            new_instances.append(idx)
            
    return new_instances


# In[ ]:


def find_similar_columns(instance, X_labeled, y_labeled, other_columns):
    
    other_similarities = {}
    
    for col_name in other_columns:
        
        indexes = (y_labeled[col_name] == 1).index
        
        other_similarities[col_name]  = similarities.calculate_similarity_between_vector_and_class(instance, X_labeled.loc[indexes])
    
    return other_similarities


# In[ ]:


def oversample_dataset_all_possible_instances(num_of_new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, similarity_factor):
    
    
    # 1. sort required # of new instances
    # 2. calculate class similarities
    # 3. iterate over columns that requieres most # of instances to balance
    # 4. find the all posible instances that can be found in the unlabeled set by using the similarities
    # 5. look for other labels that can be assigned to the found instances
    # 6. the ones that have higher similarity than average class similarity times similarity factor are assigned to that class
    # 7. add the new instances to the labeled instances for X_labeled and y_labeled
    # 8. remove the new instances from the unlabeled set# 
    
    # giving priority to mostly imbalanced classes
    num_of_new_instances = {k: v for k, v in sorted(num_of_new_instances.items(), key=lambda item: item[1], reverse=True)}
    
    class_similarities = similarities.calculate_overall_class_similarities(X_labeled, y_labeled)
    
    processed_columns = []
    
    validation = {}
    val_idx = 0
    
    for col_name, num_instance in num_of_new_instances.items():
        
        # note: we didnt use num_instance
        # the instances will be added should not exceed num_instance
        
        processed_columns.append(col_name)
        
        if num_instance == 0:
            continue
        
        indexes = (y_labeled[y_labeled[col_name] == 1]).index
        new_instances = find_new_instances(X_labeled.loc[indexes], X_unlabeled, class_similarities[col_name]*similarity_factor)
        
        for instance_index in new_instances:
            
            instance_X = X_unlabeled.loc[instance_index]
            instance_y = y_unlabeled.loc[instance_index] # note: this is for test case
            
            # defining all labels as 0s
            new_labels = {c:0 for c in y_labeled.columns}
            # changing col_name's label as 1
            new_labels[col_name] = 1
            
            ### finding other labels
            other_columns = [i for i in y_labeled.columns if i not in processed_columns]
            other_similarities = find_similar_columns(instance_X, X_labeled, y_labeled, other_columns)
            for col, sim in other_similarities.items():
                if sim > class_similarities[col]*similarity_factor:
                    new_labels[col] = 1
            
            ### appending data to unlabeled set and removing it from unlabeled set
            # starting index of new instances from a big number
            instance_new_index = max(starting_index, max(X_labeled.index)) + 1
            instance_X_series = pd.Series([instance_X], index=[instance_new_index])
            instance_new_labels =pd.DataFrame(new_labels, index=[instance_new_index])
            # adding new instance to labeled set
            X_labeled = pd.concat([X_labeled, instance_X_series])
            y_labeled = pd.concat([y_labeled, instance_new_labels])
            # removing new instance from unlabeled set
            X_unlabeled.drop(instance_index, inplace=True)
            y_unlabeled.drop(instance_index, inplace=True) # note: this is for test case
            
            # validation
            validation[val_idx] = (col_name, instance_index, instance_X, (instance_y), new_labels)
            val_idx += 1
    
    return validation, X_labeled, y_labeled, X_unlabeled, y_unlabeled 


def oversample_dataset_with_batches(num_of_new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, similarity_factor, batch_size):
    
    
    # 1. sort required # of new instances
    # 2. calculate class similarities
    # 3. iterate over columns that requieres most # of instances to balance
    # 4. find the all posible instances that can be found in the unlabeled set by using the similarities
    # 5. look for other labels that can be assigned to the found instances
    # 6. the ones that have higher similarity than average class similarity times similarity factor are assigned to that class
    # 7. add the new instances to the labeled instances for X_labeled and y_labeled
    # 8. remove the new instances from the unlabeled set# 
    
    
    # giving priority to mostly imbalanced classes
    num_of_new_instances = {k: v for k, v in sorted(num_of_new_instances.items(), key=lambda item: item[1], reverse=True)}
    
    class_similarities = similarities.calculate_overall_class_similarities(X_labeled, y_labeled)
    
    processed_columns = []
    
    validation = {}
    val_idx = 0
    
    for col_name, num_instance in num_of_new_instances.items():
        
        # note: we didnt use num_instance
        # the instances will be added should not exceed num_instance
        
        processed_columns.append(col_name)
        
        if num_instance == 0:
            continue
        
        indexes = (y_labeled[y_labeled[col_name] == 1]).index
        new_instances = find_new_instances(X_labeled.loc[indexes], X_unlabeled, class_similarities[col_name]*similarity_factor)
        
        for instance_index in new_instances:
            
            instance_X = X_unlabeled.loc[instance_index]
            instance_y = y_unlabeled.loc[instance_index] # note: this is for test case
            
            # defining all labels as 0s
            new_labels = {c:0 for c in y_labeled.columns}
            # changing col_name's label as 1
            new_labels[col_name] = 1
            
            ### finding other labels
            other_columns = [i for i in y_labeled.columns if i not in processed_columns]
            other_similarities = find_similar_columns(instance_X, X_labeled, y_labeled, other_columns)
            for col, sim in other_similarities.items():
                if sim > class_similarities[col]*similarity_factor:
                    new_labels[col] = 1
            
            ### appending data to unlabeled set and removing it from unlabeled set
            # starting index of new instances from a big number
            instance_new_index = max(starting_index, max(X_labeled.index)) + 1
            instance_X_series = pd.Series([instance_X], index=[instance_new_index])
            instance_new_labels =pd.DataFrame(new_labels, index=[instance_new_index])
            # adding new instance to labeled set
            X_labeled = pd.concat([X_labeled, instance_X_series])
            y_labeled = pd.concat([y_labeled, instance_new_labels])
            # removing new instance from unlabeled set
            X_unlabeled.drop(instance_index, inplace=True)
            y_unlabeled.drop(instance_index, inplace=True) # note: this is for test case
            
            # validation
            validation[val_idx] = (col_name, instance_index, instance_X, (instance_y), new_labels)
            val_idx += 1
    
    return validation, X_labeled, y_labeled, X_unlabeled, y_unlabeled 
