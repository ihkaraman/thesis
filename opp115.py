import os
import pandas as pd
import numpy as np
from glob import glob
from ast import literal_eval

def load(path):
    if not os.path.exists('./opp115.csv'):
        generate_dataset(path).to_csv('./opp115.csv', sep=',', index=False)

    return pd.read_csv('./opp115.csv', sep=',', header=0)

def generate_dataset(path):
    print('generating dataset...')

    p = load_policies(path)
    a = load_annotations(path)
    print(a.columns)
    print(p.columns)
    merged = pd.merge(a, p, on=['policy_id', 'segment_id'], how='outer')
    mode = merged.groupby(['policy_id', 'segment_id']).agg(lambda x: x.value_counts().index[0])
    mode.reset_index(inplace=True)

    return mode

def load_policies(path):
    policies = []

    for f in glob(path+'/sanitized_policies/*.html'):
        with open(f, 'r') as policy:
            text = policy.read()
            segments = text.split('|||')

            p = pd.DataFrame(columns=['policy_id', 'segment_id', 'text'])
            p['segment_id'] = np.arange(len(segments))
            policy_url = f.split('\\')[-1].split('_')
            p['policy_id'] = policy_url[0]
            p['url'] = policy_url[1].replace('.html', '')
            p['text'] = segments

            policies.append(p)

    p = pd.concat(policies)
    p.reset_index(inplace=True, drop=True)
    
    return p

def load_annotations(path):        
    annotations = []

    for f in glob(path+'/annotations/*.csv'): 
        a = pd.read_csv(f, sep=',', header=None, names=['annotation_id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id', 'data_practice', 'attributes', 'date', 'url'])
        a['policy_id'] = f.split('\\')[-1].split('_')[0]
        a.drop(['annotation_id', 'batch_id', 'annotator_id', 'date', 'url'], axis=1, inplace=True)
        annotations.append(a)

    a = pd.concat(annotations)
    a.reset_index(inplace=True, drop=True)
    
    return a

def attribute_counts(data):
    attributes = data['attributes'].to_list()
    counts = {}

    for a in attributes:
        d = literal_eval(a)

        for k, v in d.items():
            if not k in counts:
                counts[k] = {}
            elif not v['value'] in counts[k]:
                counts[k][v['value']] = 1
            else:
                counts[k][v['value']] += 1

    return counts