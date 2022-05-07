
metric_weighting_type = 'macro avg'

balance_ratio = 0.5
satisfying_threshold = 0.9
random_state = 1
starting_index = 100000


data_paths = {'opp115'   : r'C:\Users\IsmailKaraman\workspace\data\privacy_policy_data\OPP-115_v2\majority.csv',
              'ohsumed'  : r'C:\Users\IsmailKaraman\workspace\GitHub\thesis\data\ohsumed.csv',
              'reuters'  : r'C:\Users\IsmailKaraman\workspace\GitHub\thesis\data\Reuters21578.csv'}
              

huggingface_embeddings = ['stsb-roberta-large',
                        'all-MiniLM-L6-v2',
                        'all-MiniLM-L12-v2',
                        'all-mpnet-base-v1',
                        'all-mpnet-base-v2',
                        'all-roberta-large-v1',
                        'all-distilroberta-v1',
                        'bert-base-nli-mean-tokens',
                        'distiluse-base-multilingual-cased-v1',
                        'multi-qa-mpnet-base-dot-v1',
                        'all-distilroberta-v1',
                        'all-roberta-large-v1',
                        'bert-base-nli-mean-tokens',
                        'distiluse-base-multilingual-cased-v1',
                        'distilbert-base-nli-mean-tokens',
                        'multi-qa-mpnet-base-dot-v1',
                        'paraphrase-multilingual-MiniLM-L12-v2',
                        'paraphrase-mpnet-base-v2',
                        'paraphrase-MiniLM-L6-v2',
                        'paraphrase-xlm-r-multilingual-v1',
                        'sentence-t5-large']
                        
openai_embeddings = ['text-similarity-babbage-001',
                    'text-similarity-ada-001',
                    'text-similarity-curie-001',
                    'text-similarity-davinci-001']