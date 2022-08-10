#!/usr/bin/env python
# coding: utf-8


from itertools import combinations, product
import numpy as np
from scipy.spatial.distance import jensenshannon

e = 0.00001

def convert_distance_to_similarity(distance):
    
    similarity = 1/(1+distance)
        
    return similarity

def convert_similarity_to_distance(similarity):
    
    if similarity == 0:
        similarity += 0.000001
    
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


def minkowski_similarity(u, v, p=2):
    # minkowski distance is a distance measure but we need a similarity function
    if p <= 0:
        raise ValueError("p must be greater than 0")
    u_v = u - v
    dist = np.linalg.norm(u_v, ord=p)
        
    return convert_distance_to_similarity(dist) #converting a distance to similarity


def js_similarity(vec1, vec2):
    
    dist = jensenshannon(vec1, vec2)

    return convert_distance_to_similarity(dist)
    

def vector_similarity(vec1, vec2, sim_type):
    
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


def degrade_vector_to_scalar(similarities, sim_calculation_type):
    
    if sim_calculation_type == 'average':
        class_similarity = np.mean(similarities)
            
    elif sim_calculation_type == 'safe_interval':
        class_similarity = np.percentile(similarities, 75)
        
    return class_similarity

def calculate_similarity_between_vector_and_class(vec, class_vecs, sim_calculation_type, sim_type):
    
    similarities = [vector_similarity(vec, c_vec, sim_type) for c_vec in class_vecs]
        
    return degrade_vector_to_scalar(similarities, sim_calculation_type) 


def calculate_similarity_within_classes(vecs, sim_calculation_type, sim_type):
    
    if sim_type=='cosine':
        vectors_product = np.array(list(combinations(vecs, 2)))
        first_part = vectors_product[:,0,:]
        second_part = vectors_product[:,1,:]
        sims = np.sum(np.multiply(first_part, second_part), axis=1)/(np.multiply(np.linalg.norm(first_part, axis=1), 
                                                                                 np.linalg.norm(second_part, axis=1)))
    else:
        sims = []
        for vec1, vec2 in list(combinations(vecs, 2)):
            sims.append(vector_similarity(vec1, vec2, sim_type))
        
    if sim_calculation_type:
        return degrade_vector_to_scalar(sims, sim_calculation_type)  
    else:
        return list(sims)


def calculate_similarity_factors(class_similarities):
    
    return {k:(1/v)**0.5 for k,v in class_similarities.items()}


def calculate_similarity_between_classes(vecs1, vecs2, sim_calculation_type=None, sim_type='cosine'):
    
    vectors_product = np.array(list(product(vecs1, vecs2)))
    first_part = vectors_product[:,0,:]
    second_part = vectors_product[:,1,:]
    sims = np.sum(np.multiply(first_part, second_part), axis=1)/(np.multiply(np.linalg.norm(first_part, axis=1), np.linalg.norm(second_part, axis=1)))
    
    if sim_calculation_type:
        return degrade_vector_to_scalar(sims, sim_calculation_type)  
    else:
        return list(sims)


def calculate_overall_class_similarities(X, y, sim_calculation_type, sim_type):
    
    class_similarities = {}
    for col in y.columns:
        indexes = y[y[col] == 1].index
        class_similarities[col] = calculate_similarity_within_classes(X.loc[indexes], sim_calculation_type, sim_type) 
        
    return class_similarities

