
metric_weighting_type = 'macro avg'

threshold_factor = 1.5
test_size = 0.2

balance_ratio = 0.5
satisfying_threshold = 0.9
random_state = 1
starting_index = 100000

unlabaled_ratios = {'opp115':0.88, 'ohsumed':0.95, 'reuters':0.9}

sim_calculation_type='average'
sim_type = 'cosine'
success_metric = 'col_f1-score' # single_f1-score
embedding_method = 'distiluse-base-multilingual-cased-v1' # try different embeddings and find proper one


data_paths = {'opp115'   : r'C:\Users\IsmailKaraman\workspace\GitHub\thesis\data\opp-115.csv',
              'ohsumed'  : r'C:\Users\IsmailKaraman\workspace\GitHub\thesis\data\ohsumed.csv',
              'reuters'  : r'C:\Users\IsmailKaraman\workspace\GitHub\thesis\data\Reuters21578.csv'}
              

huggingface_embeddings = ['stsb-roberta-large',
                        'all-MiniLM-L6-v2',
                        'all-MiniLM-L12-v2',
                        'all-mpnet-base-v1',
                        'all-mpnet-base-v2',
                        'all-roberta-large-v1',
                        'all-distilroberta-v1',
                        'albert-base-v2',
                        'ALBERT-xlarge',
                        'ALBERT-xxlarg',
                        'bert-base-nli-mean-tokens',
                        'distiluse-base-multilingual-cased-v1',
                        'multi-qa-mpnet-base-dot-v1',
                        'all-distilroberta-v1',
                        'all-roberta-large-v1',
                        'bert-base-uncased',
                        'bert-base-nli-mean-tokens',
                        'distiluse-base-multilingual-cased-v1',
                        'distilbert-base-nli-mean-tokens',
                        'multi-qa-mpnet-base-dot-v1',
                        'nlpaueb/legal-bert-base-uncased',
                        'paraphrase-multilingual-MiniLM-L12-v2',
                        'paraphrase-mpnet-base-v2',
                        'paraphrase-MiniLM-L6-v2',
                        'paraphrase-xlm-r-multilingual-v1',
                        'saibo/legal-roberta-base',
                        'sentence-t5-large',
                        'sentence-transformers/average_word_embeddings_glove.6B.300d',
                        'sentence-transformers/average_word_embeddings_glove.840B.300d']
                        
openai_embeddings = ['text-similarity-babbage-001',
                    'text-similarity-ada-001',
                    'text-similarity-curie-001',
                    'text-similarity-davinci-001']
                    
google_embeddings = ['universal-sentence-encoder']              