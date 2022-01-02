import os
import pandas as pd
import numpy as np
from glob import glob
from ast import literal_eval


def load(path):
    if not os.path.exists(path+'/opp115.csv'):
        generate_dataset(path).to_csv(path+'/opp115.csv', sep=',', index=False)
    return pd.read_csv(path+'/opp115.csv', sep=',', header=0)

def load_policies(path):
    policies = []

    for file in glob(path+'/sanitized_policies/*.html'):
        with open(file, 'r') as policy:
            
            text = policy.read()
            segments = text.split('|||')

            tmp_policy = pd.DataFrame(columns=['policy_id', 'segment_id', 'text'])
            tmp_policy['segment_id'] = np.arange(len(segments))
            policy_url = file.split('\\')[-1].split('_')
            tmp_policy['policy_id'] = policy_url[0]
            tmp_policy['url'] = policy_url[1].replace('.html', '')
            tmp_policy['text'] = segments

            policies.append(tmp_policy)

    all_policies = pd.concat(policies)
    all_policies.reset_index(inplace=True, drop=True)
    
    return all_policies

def load_annotations(path):        
    annotations = []

    for file in glob(path+'/annotations/*.csv'): 
        tmp_annotation = pd.read_csv(file, sep=',', header=None, names=['annotation_id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id', 'data_practice', 'attributes', 'date', 'url'])
        tmp_annotation['policy_id'] = file.split('\\')[-1].split('_')[0]
        tmp_annotation.drop(['annotation_id', 'batch_id', 'date', 'url', 'attributes'], axis=1, inplace=True)
        annotations.append(tmp_annotation)

    all_annotations = pd.concat(annotations)
    all_annotations.reset_index(inplace=True, drop=True)
    
    return all_annotations

def generate_dataset(path):
    
    print('Generating dataset...')
    policies = load_policies(path)
    annotations = load_annotations(path)
    
    merged = pd.merge(annotations, policies, on=['policy_id', 'segment_id'], how='outer')
    merged = pd.concat([merged, pd.get_dummies(merged['data_practice'])], axis=1)
    merged = merged.groupby(['policy_id', 'segment_id', 'annotator_id']).max()
    merged = merged.groupby(['policy_id', 'segment_id']).max().reset_index()

    return merged