#!/usr/bin/env python
# coding: utf-8

import torch
import random
import parameters
import preprocess
import similarities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import hamming_loss, accuracy_score, f1_score, classification_report, coverage_error 
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, roc_auc_score
from sklearn.metrics import brier_score_loss, precision_score, recall_score, zero_one_loss, label_ranking_loss, log_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.multiclass import OneVsRestClassifier



# classifier_object = SVC(kernel='linear',probability=True, random_state=1)
# LinearSVC(class_weight='balanced', random_state=1)
classifier_object = LogisticRegression(random_state=1)

balance_ratio = parameters.balance_ratio
satisfying_threshold = parameters.satisfying_threshold
random_state = parameters.random_state
starting_index = parameters.starting_index
metric_weighting_type = parameters.metric_weighting_type
np.random.seed(random_state)



def read_data(path):
    df = pd.read_csv(path)
    df['text'] = df['text'].apply(preprocess.preprocess_text)
    return df

def vectorize_data(text, model_name='stsb-roberta-large'):
           
    if model_name in parameters.huggingface_embeddings:
          
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        vectors = model.encode(list(text), convert_to_tensor=False, device=device)
        
    elif model_name in parameters.openai_embeddings:
        
        import openai
        import config
        openai.api_key = config.openai_api_key
        
        vecs = []
        batch_num = int(len(text) / 1_000) + (len(text) % 1_000 > 0)
        for batch in np.array_split(text, batch_num):
            vectors = openai.Embedding.create(input=list(batch), engine=model_name)
            vectors = [vec['embedding'] for vec in vectors['data']]
            vecs.extend(vectors)
        vectors = np.array(vecs)   
        
    elif model_name in parameters.google_embeddings:
        
        if model_name == 'universal-sentence-encoder':
            model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

        import tensorflow_hub as hub
        model = hub.load(model_url)
        
        vecs = []
        batch_num = int(len(text) / 10_000) + (len(text) % 10_000 > 0)
        for batch in np.array_split(text, batch_num):
            vecs.extend(model(batch).numpy())
        vectors = np.array(vecs)   
        
    else:
        raise ValueError('Undefined embedding type!')
        
    if type(text)== pd.core.series.Series: 
        vectors = pd.Series([np.squeeze(i) for i in vectors], index=text.index)
        
    return vectors


def visualize_data(X, typ='pca'):
    
    matrix = np.array([np.squeeze(i) for i in X])
    
    if typ == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_ = pca.fit_transform(X)
    elif type == 'tsn':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto')
        X_ = tsne.fit_transform(matrix)
    
    x = X_[:,0]
    y = X_[:,1]
    plt.scatter(x, y)
    plt.title("2d Visualization")


def calculating_class_weights(y_true):
        
        number_dim = np.shape(y_true)[1]
        weights = []
        for i in range(number_dim):
            weights.append(dict(zip([0,1], compute_class_weight('balanced', [0.,1.], y_true[:, i]))))
            # weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])))
        return weights


def binary_classifier(X_train, y_train, X_test, y_test):

    # class_weights = calculating_class_weights(y_train.values)
    
    # Linear SVM
    model = LinearSVC(class_weight='balanced')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    #print('+ '*50)
    #print("\033[1m" + 'Binary Classifier Results' + "\033[0m")
    #print("\033[1m" + 'LinearSVM' + "\033[0m")
    #print('-'*30)
    #print("\033[1m" + 'Classification Report' + "\033[0m")
    #print(classification_report(y_test, preds))
    
    return f1_score(y_test, preds, average='binary')

def multilabel_brier_loss(y_true, y_pred, weighting):
    
    if weighting == 'average':
        b_scores = []
        for x, y in zip(y_true, y_pred):
            b_scores.append(brier_score_loss(x, y))
        return sum(b_scores)/len(b_scores)
    
    if weighting == 'macro':
        
        scores = []
        for i in range(len(y_true[0])):
            mask = y_true[:,i]==1
            b_scores = []
            for x, y in zip(y_true[mask], y_pred[mask]):
                b_scores.append(brier_score_loss(x, y))
            scores.append(sum(b_scores)/len(b_scores))

        return sum(scores)/len(scores)

def metric_function(success_metric, y_test, test_preds, test_scores):
    
    if success_metric.startswith('col'):
        success_metric = success_metric.replace('col_', '')
        clf_report = classification_report(y_test, test_preds, target_names=list(y_test.columns), output_dict=True)
        output_metric = {col:clf_report[col][success_metric] for col in y_test.columns}  
    elif success_metric=='accuracy':
        output_metric = accuracy_score(y_test, test_preds)
    elif success_metric=='f1_score':
        output_metric = f1_score(y_test, test_preds, average='macro')
    elif success_metric == 'coverage':
        output_metric = coverage_error(y_test, test_scores)
    elif success_metric == 'label_ranking':
        output_metric = label_ranking_loss(y_test, test_scores)
    elif success_metric == 'roc_auc_score':
        output_metric = roc_auc_score(y_test, test_scores)
    elif success_metric == 'log_loss':
        output_metric = log_loss(y_test, test_scores)   
    elif success_metric == 'average_precision':
        output_metric = average_precision_score(y_test, test_scores)
    elif success_metric == 'brier_loss':
        output_metric = multilabel_brier_loss(y_test, test_scores, 'macro') 
    elif success_metric == 'hamming_loss':
        output_metric = hamming_loss(y_test, test_preds)
    elif success_metric=='precision':
        output_metric = precision_score(y_test, test_preds, average='macro')
    elif success_metric=='recall':
        output_metric = recall_score(y_test, test_preds, average='macro')
    elif success_metric=='zero_one_loss':
        output_metric = zero_one_loss(y_test, test_preds)  
    elif success_metric == 'label_ranking_average_precision' :
        output_metric = label_ranking_average_precision_score(y_test, test_scores)
    return output_metric

      
def multilabel_classifier(X_train, y_train, X_test, y_test, success_metric, classifier_object=classifier_object, print_results=False):

    '''
    success_metric:
        x_y format
        x: type, [col, single]
        y: metric, [precision, recall, f1-score]
    '''
    # class_weights = calculating_class_weights(y_train.values)
    
    # Linear SVM
    model = OneVsRestClassifier(classifier_object, n_jobs=-1)
    model.fit(X_train, y_train.values)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_scores = model.predict_proba(X_train)
    test_scores = model.predict_proba(X_test)
    
    hamLoss_train = hamming_loss(y_train.values, train_preds)
    hamLoss_test = hamming_loss(y_test.values, test_preds)
    acc_score_train = accuracy_score(y_train.values, train_preds)
    acc_score_test = accuracy_score(y_test.values, test_preds)   
    f1_score_train = f1_score(y_train.values, train_preds, average=metric_weighting_type)
    f1_score_test = f1_score(y_test.values, test_preds, average=metric_weighting_type)
    coverage_train = coverage_error(y_train.values, train_scores)
    coverage_test = coverage_error(y_test.values, test_scores)
    rankLoss_train = label_ranking_loss(y_train.values, train_scores)
    rankLoss_test = label_ranking_loss(y_test.values, test_scores)
    
    if print_results:
        print("\033[1m" + 'Multilabel Classifier Results' + "\033[0m")
        print("\033[1m" + type(classifier_object).__name__ + "\033[0m")
        print('-'*30)
        print('Hamming Loss')
        print(f'Training : {hamLoss_train:.2f}')
        print(f'Test     : {hamLoss_test:.2f}')
        print('Exact Match Ratio')
        print(f'Training : {acc_score_train:.2f}')
        print(f'Test     : {acc_score_test:.2f}')
        print('Macro F1-Score')
        print(f'Training : {f1_score_train:.2f}')
        print(f'Test     : {f1_score_test:.2f}')
        print('Coverage Error')
        print(f'Training : {coverage_train:.2f}')
        print(f'Test     : {coverage_test:.2f}')
        print('Ranking Loss Error')
        print(f'Training : {rankLoss_train:.2f}')
        print(f'Test     : {rankLoss_test:.2f}')
        print('-'*30)
        print("\033[1m" + 'Classification Report' + "\033[0m")
        print(classification_report(y_test.values, test_preds, target_names=list(y_test.columns)))
        print('* '*50)
    
    
    
    if type(success_metric) == str:
        
        output_metric = metric_function(success_metric, y_test, test_preds, test_scores)
            
        return output_metric
    
    elif type(success_metric) == list:
        
        output_metric = []
        for metric in success_metric:
            output_metric.append(metric_function(metric, y_test, test_preds, test_scores))
                      
        return output_metric


def calculate_imb_ratio(y):
    
    class_ratios = (y.sum() / y.shape[0]).values
        
    return class_ratios


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


def calculate_balancing_num_instance_binary(n_samples, n_total_samples, balance_ratio=0.5, calculation_type=None, success_metric=0.0):
    
    if n_samples/n_total_samples > balance_ratio:
        print("Be careful! Given balancing ratio is lower than the class' imbalance ratio")
    
    balance_num = (n_total_samples*balance_ratio - n_samples)*2
    
    if calculation_type=='metric_based':
        if success_metric > 1.0 or success_metric < 0.0 :
            raise ValueError('Success metric should be between 0 and 1 !') 
        balance_num *= (1-success_metric)
    
    return int(balance_num)


def calculate_balancing_num_instance_multiclass(y, balance_ratio, calculation_type, s_metrics):
    
    oversampling_counts = {}
    n_samples = y.shape[0]
    
    for col in y.columns:
        s_metric = s_metrics[col]
        oversampling_counts[col] = calculate_balancing_num_instance_binary(y[col].sum(), n_samples, balance_ratio, 
                                                                           calculation_type, s_metric)
    
    return oversampling_counts


def find_new_instance_batches(X_labeled, X_unlabeled, class_similarity, batch_size):
    
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


def calculate_all_similarities(X_labeled, X_unlabeled, sort=False):
    
    # calculates all similarities for unlabeled set
    # if sort: sort them according to similarity in descending order
    all_similarities = {}
    
    for idx, instance in X_unlabeled.iteritems():
        ins_sim = similarities.calculate_similarity_between_vector_and_class(instance, X_labeled)
        all_similarities[idx] = ins_sim
    
    if sort:
        all_similarities = {k: v for k, v in sorted(all_similarities.items(), key=lambda item: item[1], reverse=True)}
    
    return all_similarities


def find_similar_columns(instance, X_labeled, y_labeled, other_columns):
    
    other_similarities = {}
    
    for col_name in other_columns:
        
        indexes = (y_labeled[col_name] == 1).index
        other_similarities[col_name]  = similarities.calculate_similarity_between_vector_and_class(instance, X_labeled.loc[indexes])
    
    return other_similarities


def find_labels(instance_X, X_labeled, y_labeled, class_similarities, col_name, processed_columns, 
                similarity_factors=None):
                
    # defining all labels as 0s
    new_labels = {c:0 for c in y_labeled.columns}
    # changing col_name's label as 1
    new_labels[col_name] = 1
    
    if not similarity_factors:
        similarity_factors = {k:1 for k,v in class_similarities.items()}    
    ### finding other labels
    other_columns = [i for i in y_labeled.columns if i not in processed_columns]
    other_similarities = find_similar_columns(instance_X, X_labeled, y_labeled, other_columns)
    for col, sim in other_similarities.items():
        if sim > class_similarities[col]*similarity_factors[col]:
            new_labels[col] = 1

    return new_labels


def find_all_labels(instance_X, X_labeled, y_labeled, class_similarities, col_name):
                
    # defining all labels as 0s
    new_labels = {c:0 for c in y_labeled.columns}
    # changing col_name's label as 1
    new_labels[col_name] = 1

     ### finding other labels
    other_columns = y_labeled.columns
    other_columns = other_columns.remove(col_name)
    
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

def check_for_improvement(single_score, general_score_after, general_score_before):
    
    if single_score in ['single_f1-score']:
        if general_score_after > general_score_before:
            return True
        else:
            return False
    elif single_score in ['coverage', 'label_ranking']:
        if general_score_after < general_score_before:
            return True
        else:
            return False

def update_similarity_factor(similarity_factor, update_type):

    if update_type == 'increase':
        similarity_factor = similarity_factor + ((1-similarity_factor)**2) * similarity_factor
    elif update_type == 'decrease':
        similarity_factor = similarity_factor - ((1-similarity_factor)**2) * similarity_factor
    
    return similarity_factor


def prepare_new_instances(new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, class_similarities, 
                          col_name, processed_columns, similarity_factors=None):
    # can be improved
    validation = []
    
    for instance_index in new_instances:

        instance_X = X_unlabeled.loc[instance_index]
        instance_y = y_unlabeled.loc[instance_index] # note: this is for test case

        new_labels = find_labels(instance_X, X_labeled, y_labeled, class_similarities, col_name, 
                                       processed_columns, similarity_factors)
        X_labeled, y_labeled, X_unlabeled, y_unlabeled = add_instance(instance_X, instance_index, new_labels, 
                                                                      X_labeled, y_labeled, X_unlabeled, y_unlabeled)
        # validation
        validation.append((col_name, instance_index, instance_X, (instance_y), new_labels))
    
    return validation, X_labeled, y_labeled, X_unlabeled, y_unlabeled
        
        
def oversample_dataset_v1(num_of_new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test, sim_calculation_type, batch_size):
    
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
       
        print("\033[1m" + '-'*15 + col_name + '-'*15 +"\033[0m")
        print('='*50)
        f1_before = binary_classifier(np.vstack(X_labeled.values), y_labeled[col_name], np.vstack(X_test.values), y_test[col_name])
                        
        indexes = (y_labeled[y_labeled[col_name] == 1]).index
        batches = find_batches(batch_size, num_instance)
        
        for batch in batches:
            
            new_instances = find_new_instance_batches(X_labeled.loc[indexes], X_unlabeled, class_similarities[col_name], batch) 
            val_new, X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new = prepare_new_instances(new_instances, X_labeled, 
                                                                                                            y_labeled, X_unlabeled, 
                                                                                                            y_unlabeled, class_similarities, 
                                                                                                            col_name, processed_columns)
            
            # check results after every batch
            f1_after = binary_classifier(np.vstack(X_labeled_new.values), y_labeled_new[col_name], np.vstack(X_test.values), y_test[col_name])
            
            if f1_after > f1_before:
             
                X_labeled, y_labeled = X_labeled_new, y_labeled_new
                validation.extend(val_new)
                # f1_before = f1_after if we assign f1_after to f1_before we may not catch the 
            
            X_unlabeled, y_unlabeled = X_unlabeled_new, y_unlabeled_new
            
    print('Shapes --------------')
    print(X_labeled.shape, X_unlabeled.shape)            
               
    return validation, X_labeled, y_labeled, X_unlabeled, y_unlabeled 


def oversample_dataset_v2(num_of_new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test, sim_calculation_type, batch_size):
        
    # giving priority to mostly imbalanced classes
    num_of_new_instances = {k: v for k, v in sorted(num_of_new_instances.items(), key=lambda item: item[1], reverse=True)}
    
    class_similarities = similarities.calculate_overall_class_similarities(X_labeled, y_labeled, sim_calculation_type)
    
    similarity_factors = similarities.calculate_similarity_factors(class_similarities)
            
    
    processed_columns = []
    validation = []
    
    for col_name, num_instance in num_of_new_instances.items():
        
        processed_columns.append(col_name)
        
        # if no need to add instance, skip that column
        if num_instance <= 0:
            continue
       
        print("\033[1m" + '-'*15 + col_name + '-'*15 +"\033[0m")
        print('='*50)
        
        f1_before = binary_classifier(np.vstack(X_labeled.values), y_labeled[col_name], np.vstack(X_test.values), y_test[col_name])
        
        indexes = (y_labeled[y_labeled[col_name] == 1]).index
        
        similarity_factor = similarity_factors[col_name]
        
        sorted_similarities = calculate_all_similarities(X_labeled.loc[indexes], X_unlabeled, sort=True)

        keys, values = list(sorted_similarities.keys()), list(sorted_similarities.values())

        for start_index in range(0, len(sorted_similarities), batch_size):

            new_instance_keys = keys[start_index:start_index+batch_size]
            new_instance_values = values[start_index:start_index+batch_size]

            # calculate batch's average similarity
            avg_batch_sim = sum(new_instance_values)/len(new_instance_values)
            
            num_of_failed_iter = 0
            
            if avg_batch_sim >= class_similarities[col_name]*similarity_factor:
                
                num_of_failed_iter = 0
                
                val_new, X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new = \
                prepare_new_instances(new_instance_keys, X_labeled, y_labeled, X_unlabeled, y_unlabeled, \
                                      class_similarities, col_name, processed_columns)

                # check results after every batch
                f1_after = binary_classifier(np.vstack(X_labeled_new.values), y_labeled_new[col_name], \
                                             np.vstack(X_test.values), y_test[col_name])

                
                if f1_after > f1_before:
                    
                    X_labeled, y_labeled = X_labeled_new, y_labeled_new
                    validation.extend(val_new)
                    # f1_before = f1_after if we assign f1_after to f1_before we may not catch the 
                    X_unlabeled, y_unlabeled = X_unlabeled_new, y_unlabeled_new

                    similarity_factor = update_similarity_factor(similarity_factor, 'increase')
       
                else:
                    similarity_factor = update_similarity_factor(similarity_factor, 'decrease')
            else:
                num_of_failed_iter += 1
            
            if num_of_failed_iter > 2:
                break          
            
    print('Shapes --------------')
    print(X_labeled.shape, X_unlabeled.shape)            
               
    return validation, X_labeled, y_labeled, X_unlabeled, y_unlabeled 


def oversample_dataset_v3(num_of_new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test, sim_calculation_type, batch_size, n_iter, single_score):
    
    
    # giving priority to mostly imbalanced classes
    num_of_new_instances = {k: v for k, v in sorted(num_of_new_instances.items(), key=lambda item: item[1], reverse=True)}
    
    class_similarities = similarities.calculate_overall_class_similarities(X_labeled, y_labeled, sim_calculation_type)
    similarity_factors = similarities.calculate_similarity_factors(class_similarities)
    processed_columns = []
    validation = []
    
    for col_name, num_instance in num_of_new_instances.items():
        
        processed_columns.append(col_name)
        
        # if no need to add instance, skip that column
        if num_instance <= 0:
            continue
        
        print("\033[1m" + '-'*int(25-len(col_name)/2) + col_name + '-'*int(25-len(col_name)/2) +"\033[0m")
        print('='*50)
        
        
        indexes = (y_labeled[y_labeled[col_name] == 1]).index
        
        similarity_factor = similarity_factors[col_name]
        
        all_similarities = calculate_all_similarities(X_labeled.loc[indexes], X_unlabeled, sort=False)
        iter_num = 0
        
        stopping_condition = True
        
        while stopping_condition or iter_num < n_iter:
              
            print(iter_num, ' iteration ...')              
            # filtering the instances that have greater similarity than similarity factor
            potential_instances = {k:v for k, v in all_similarities.items() if v>class_similarities[col_name]*similarity_factor}
            
            print(len(potential_instances))
            if len(potential_instances) == 0:
                break
            
            # shuffling the potential instances
            potential_instance_keys = list(potential_instances.keys())
            random.shuffle(potential_instance_keys)
            # potential_instances = {k:potential_instances[k] for k in ins_keys}
            
            binary_score_before = binary_classifier(np.vstack(X_labeled.values), y_labeled[col_name], np.vstack(X_test.values), y_test[col_name])
            general_score_before = multilabel_classifier(X_labeled, y_labeled, X_test, y_test, success_metric=single_score)
            
            candidate_instances = []
            
            for idx in potential_instance_keys():
                                    
                # check results for each instance,
                print('____shapes____')
                print(X_labeled.values.shape, X_labeled.values.append(X_unlabeled.loc[idx]).shape)
                print(y_labeled[col_name].values.shape, y_labeled[col_name].values.append([1]).shape)
                binary_score_after = binary_classifier(np.vstack(X_labeled.values.append(X_unlabeled.loc[idx])),
                                                                 y_labeled[col_name].values,
                                                                 np.vstack(X_test.values), y_test[col_name])
            
                if binary_score_after >= binary_score_before:
                    
                    if binary_score_after > satisfying_threshold:
                        stopping_condition = False
                        
                    candidate_instances.append(idx)
                    
                    if len(candidate_instances) >= batch_size:
                        
                        val_new, X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new = \
                        prepare_new_instances(candidate_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                                              class_similarities, col_name, processed_columns)
                        
                        general_score_after = multilabel_classifier(X_labeled_new, y_labeled_new, X_test, y_test, 
                                                                    success_metric=single_score)
                        
                        if check_for_improvement(single_score, general_score_after, general_score_before):
                            
                            X_labeled, y_labeled = X_labeled_new, y_labeled_new
                            X_unlabeled, y_unlabeled = X_unlabeled_new, y_unlabeled_new
                            validation.extend(val_new)
                            
                            # increasing similarity factor
                            similarity_factor = update_similarity_factor(similarity_factor, 'increase')  
                            
                        else:
                            # decreasing similarity factor
                            similarity_factor = update_similarity_factor(similarity_factor, 'decrease')

                        # emptying the list of candidate isntances
                        candidate_instances.clear()

    print('Shapes --------------')
    print(X_labeled.shape, X_unlabeled.shape)            
               
    return validation, X_labeled, y_labeled, X_unlabeled, y_unlabeled 


def oversample_dataset_v4(num_of_new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test, sim_calculation_type, 
                          batch_size, n_iter, balance_ratio, success_metric, single_score):
    
     
    class_similarities = similarities.calculate_overall_class_similarities(X_labeled, y_labeled, sim_calculation_type)
    similarity_factors = similarities.calculate_similarity_factors(class_similarities)
    
    validation = []
    
    # an initial classification
    col_metrics, general_score_before = multilabel_classifier(np.vstack(X_labeled), y_labeled, np.vstack(X_test), y_test, 
                                               success_metric=[success_metric, single_score], print_results=False)
    
    
    iter_num = 0
    stopping_condition = True
    
    while stopping_condition and iter_num < n_iter:
        
        num_of_new_instances = calculate_balancing_num_instance_multiclass(y_labeled, balance_ratio, 
                                                                                 calculation_type='metric_based', 
                                                                                 s_metrics=col_metrics)
        # calculating selection probabilities by num of required instances
        selection_probabilities = {k:max(0, v/sum(num_of_new_instances.values())) for k,v in num_of_new_instances.items()}
        # normalizing probabilities
        selection_probabilities = {k:v/sum(selection_probabilities.values()) for k,v in selection_probabilities.items()}
        # selecting a random class with selection_probabilities
        col_name = random.choices(list(selection_probabilities.keys()), weights=selection_probabilities.values())[0]
        
        # print("\033[1m" + '-'*int(25-len(col_name)/2) + col_name + '-'*int(25-len(col_name)/2) +"\033[0m")
        
        # find the indexes that belong to the chosen class
        indexes = (y_labeled[y_labeled[col_name] == 1]).index
        
        # find new instances from Unlabeled set by using similarity and similarity factor
        new_instances = find_new_instance_batches(X_labeled.loc[indexes], X_unlabeled, 
                                                     class_similarities[col_name]*similarity_factors[col_name], batch_size)
        # print('new_instances : ', new_instances)  
   
        val_new, X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new = \
        prepare_new_instances(new_instances, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                              class_similarities, col_name, [col_name], similarity_factors)

        col_metrics_tmp, general_score_after = multilabel_classifier(np.vstack(X_labeled_new), y_labeled_new, np.vstack(X_test), y_test, 
                                                    success_metric=[success_metric, single_score])

                
                
        if check_for_improvement(single_score, general_score_after, general_score_before):
            
            X_labeled, y_labeled = X_labeled_new, y_labeled_new
            validation.extend(val_new)
            
            # increasing similarity factor
            #similarity_factors[col_name] = update_similarity_factor(similarity_factors[col_name], 'increase')  
            
            # update column metrics after adding new instances
            col_metrics = col_metrics_tmp
            # updating general score to add better instances
            general_score_before = general_score_after
            
        else:
            # decreasing similarity factor
            similarity_factors[col_name] = update_similarity_factor(similarity_factors[col_name], 'decrease')
        
        X_unlabeled, y_unlabeled = X_unlabeled_new, y_unlabeled_new
        iter_num += 1
           
    print('Shapes --------------')
    print(X_labeled.shape, X_unlabeled.shape)            
               
    return validation, X_labeled, y_labeled, X_unlabeled, y_unlabeled 