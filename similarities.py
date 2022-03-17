#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from itertools import combinations
import numpy as np
from scipy.spatial.distance import jensenshannon


sim_type = 'cosine'
# In[ ]:


def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0:
        norm1 += 0.00001
    if norm2 == 0:
        norm2 += 0.00001   
    return np.dot(vec1, vec2)/(norm1*norm2)


# In[ ]:


def minkowski_similarity(u, v, p=2):
    # minkowski distance is a distance measure but we need a similarity function
    if p <= 0:
        raise ValueError("p must be greater than 0")
    u_v = u - v
    dist = np.linalg.norm(u_v, ord=p)
    if dist == 0:
        dist += 0.0001
        
    return 1/dist #converting a distance to similarity


# In[ ]:


def vector_similarity(vec1, vec2, sim_type='cosine'):
    
    if sim_type == 'cosine':
        similarity = cosine_similarity(vec1, vec2)
    if sim_type == 'euclidean':
        similarity = minkowski_similarity(vec1, vec2, 2)
    if sim_type == 'manhattan':
        similarity = minkowski_similarity(vec1, vec2, 1)
    if sim_type == 'chebychev':
        similarity = minkowski_similarity(vec1, vec2, np.inf)
    if sim_type.startswith('minkowski'):
        similarity = minkowski_similarity(vec1, vec2, int(sim_type[-1]))
    if sim_type == 'JS':
        similarity = jensenshannon(vec1, vec2)
        
    return similarity


# In[ ]:


def calculate_within_class_similarity(vecs, sim_type='cosine'):
    
    similarities = []
    
    for i,j in list(combinations(vecs.index, 2)):
        similarities.append(vector_similarity(vecs.loc[i], vecs.loc[j], sim_type))    
            
    try:
        avg_similarity = sum(similarities)/len(similarities)
    except AssertionErrors:
        print('Error occured')
        
    return avg_similarity 


# In[ ]:


def calculate_similarity_between_vector_and_class(vec, class_vecs, sim_type='cosine'):
    
    similarities = []
    
    for c_vec in class_vecs:
        similarities.append(vector_similarity(vec, c_vec, sim_type))
    
    try:
        avg_similarity = sum(similarities)/len(similarities)
    except AssertionErrors:
        print('Error occured')
        
    return avg_similarity 


# In[ ]:


def calculate_overall_class_similarities(X, y):
    
    class_similarities = {}
    for col in y.columns:
        indexes = (y[col] == 1).index
        class_similarities[col] = calculate_within_class_similarity(X.loc[indexes]) 
        
    return class_similarities

