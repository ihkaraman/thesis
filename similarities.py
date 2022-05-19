#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from itertools import combinations
import numpy as np
from scipy.spatial.distance import jensenshannon

e = 0.00001
# In[ ]:

def convert_distance_to_similarity(distance):
    
    if distance == 0:
        similarity = 1/(1+distance)
        
    return similarity

def convert_similarity_to_distance(similarity):
    
    if similarity == 0:
        distance = (1/similarity)-1
        
    return distance

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0:
        norm1 += e
    if norm2 == 0:
        norm2 += e  
    return np.dot(vec1, vec2)/(norm1*norm2)


# In[ ]:


def minkowski_similarity(u, v, p=2):
    # minkowski distance is a distance measure but we need a similarity function
    if p <= 0:
        raise ValueError("p must be greater than 0")
    u_v = u - v
    dist = np.linalg.norm(u_v, ord=p)
        
    return convert_distance_to_similarity(dist) #converting a distance to similarity


# In[ ]:

def js_similarity(vec1, vec2):
    
    dist = jensenshannon(vec1, vec2)

    return convert_distance_to_similarity(dist)
    
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
        similarity = js_similarity(vec1, vec2)
        
    return similarity


# In[ ]:

def degrade_vector_to_scalar(similarities, sim_calculation_type):
    
    if sim_calculation_type == 'average':
        
        try:
            class_similarity = sum(similarities)/len(similarities)
        except AssertionError:
            print('Error occured')
            
    elif sim_calculation_type == 'safe_interval':
        class_similarity = np.percentile(similarities, 75)
   
    return class_similarity
'''
def calculate_within_class_similarity(vecs, sim_calculation_type, sim_type='cosine'):
    
    similarities = []
    
    for i,j in list(combinations(vecs.index, 2)):
        similarities.append(vector_similarity(vecs.loc[i], vecs.loc[j], sim_type))    
      
    return degrade_vector_to_scalar(similarities, sim_calculation_type) 
'''
def calculate_within_class_similarity(vecs, sim_calculation_type, sim_type='cosine'):
       
    #import ray
    #ray.init()
    
    #@ray.remote
    def run(vec1, vec2, sim_type):
        sim = vector_similarity(vec1, vec2, sim_type)
        return sim
    similarities = [run(vecs.loc[i], vecs.loc[j], sim_type) for i,j in list(combinations(vecs.index, 2))]
    #futures = [run.remote(vecs.loc[i], vecs.loc[j], sim_type) for i,j in list(combinations(vecs.index, 2))]
    #similarities = ray.get(futures) 
    
    #ray.shutdown()
     
    return degrade_vector_to_scalar(similarities, sim_calculation_type) 
'''
def calculate_similarity_between_vector_and_class(vec, class_vecs, sim_calculation_type='average', sim_type='cosine'):
    
    similarities = []
    
    for c_vec in class_vecs:
        similarities.append(vector_similarity(vec, c_vec, sim_type))
        
    return degrade_vector_to_scalar(similarities, sim_calculation_type) 
'''
def calculate_similarity_between_vector_and_class(vec, class_vecs, sim_calculation_type='average', sim_type='cosine'):
    
    #import ray
    #ray.init()
    
    #@ray.remote
    def run(vec1, vec2, sim_type):
        sim = vector_similarity(vec1, vec2, sim_type)
        return sim
    similarities = [run(vec, c_vec, sim_type) for c_vec in class_vecs]
    #futures = [run.remote(vec, c_vec, sim_type) for c_vec in class_vecs]
    #similarities = ray.get(futures) 
    
    #ray.shutdown()
        
    return degrade_vector_to_scalar(similarities, sim_calculation_type) 

def calculate_similarity_factors(class_similarities):
    
    return {k:(1/v)**0.5 for k,v in class_similarities.items()}


def calculate_overall_class_similarities(X, y, sim_calculation_type):
    
    class_similarities = {}
    for col in y.columns:
        indexes = y[y[col] == 1].index
        class_similarities[col] = calculate_within_class_similarity(X.loc[indexes], sim_calculation_type) 
        
    return class_similarities

