import os 
import spacy
import re
import math
import random
from conll_utils import get_tweets_content, get_annotations, check_offsets
from entities_utils import simplify_nested_entities, filter_crossing_entities
from tqdm.auto import tqdm
from collections import defaultdict


nlp = spacy.load("es_core_news_lg")

def format_data(output_directory, seed):
    # Seed for reproducibility
    random.seed(seed)
    directory = os.getcwd()

    data_directory = os.path.join(directory, 'training-validation-data/train-valid-txt-files/')
    ann_directory = os.path.join(directory, 'training-validation-data/mentions.tsv')
    
    training_tweets_directory = os.path.join(data_directory, 'training/')
    validation_tweets_directory = os.path.join(data_directory, 'validation/')
    

    training_tweets_content = get_tweets_content(training_tweets_directory)
    validation_tweets_content = get_tweets_content(validation_tweets_directory)


    # Creating dictionary with ids of documents for training and validation.
    ids = {'training': [], 'validation': []}
    
    for id in open(f'{data_directory}/ids_train_set.txt', 'r', encoding='UTF-8').read().splitlines():
        ids['training'].append(str(id))
    
    
    for id in open(f'{data_directory}/ids_dev_set.txt', 'r',encoding='UTF-8').read().splitlines():
        ids['validation'].append(str(id))
    
    # Get annotations for each partition
    training_tweets_annotations, validation_tweets_annotations = get_annotations(ann_directory, ids)

    training_tweets_annotations_filtered = defaultdict(list)
    for filename in ids['training']:
        if filename in training_tweets_annotations:
            training_tweets_annotations_filtered[filename]=training_tweets_annotations[filename]
        else:
            training_tweets_annotations_filtered[filename]=[]        
    training_tweets_annotations = dict(sorted(training_tweets_annotations_filtered.items()))

    validation_tweets_annotations_filtered = defaultdict(list)
    for filename in ids['validation']:
        if filename in validation_tweets_annotations:
            validation_tweets_annotations_filtered[filename]=validation_tweets_annotations[filename]
        else:
            validation_tweets_annotations_filtered[filename]=[]        
    validation_tweets_annotations = dict(sorted(validation_tweets_annotations_filtered.items()))

    


    check_offsets(training_tweets_content, training_tweets_annotations)
    check_offsets(validation_tweets_content, validation_tweets_annotations)


    create_conll_file(training_tweets_content, training_tweets_annotations, output_directory, 'training_full_annotations')
    create_conll_file(validation_tweets_content, validation_tweets_annotations, output_directory, 'validation_full_annotations')
    

    create_partitions(output_directory, 'training_full_annotations')

def create_conll_file(documents, annotations, output_directory, entity_type):

    output_file = open(f'{output_directory}/{entity_type}.conll', 'w', encoding='UTF-8')
    original_entities_count = sum(len(v) for k, v in annotations.items())
    annotations = filter_crossing_entities(annotations)
    flat_annotations = simplify_nested_entities(annotations)
    flat_entities_count = sum(len(v) for k, v in flat_annotations.items())
    print(f'Original number of entities in partition: {original_entities_count}. {original_entities_count - flat_entities_count} deleted by nestings and crossing entities.')

    
    for filename, content in tqdm(documents.items()):
        #output_file.write(filename+'\n')
        doc = nlp(content)
        entities = flat_annotations[filename]
        entities_annotated = []
        entities_added = []
        cnt = 0
        for sent in doc.sents:
            cnt+=len(sent)
            for token in sent:

                
                if not token.text or '\n' in token.text or '\t' in token.text or token.text.strip()=='':
                    continue
                
                token_tag = 'O'
                token_start = token.idx
                token_end = token.idx + len(token)

                for entity in entities:
                    if token_start == entity["start_index"]:
                        token_tag = f'B-{entity["label"]}'
                        entities_annotated.append(entity)
                        entities_added.append(entity)
                        break
                    elif token_start > entity["start_index"] and token_end <= entity["end_index"]:
                        token_tag=f'I-{entity["label"]}'
                        break
              
                
                output_file.write(f'{token.text}\t{token_tag}\n')
           
            output_file.write('\n')
                

      
            
            

    print(len(entities_annotated))
   


def create_partitions(output_directory, entity_type):

    f = open(f'{output_directory}/{entity_type}.conll', 'r', encoding='UTF-8').read()
    f = re.sub(r'\n\s*\n', '\n\n', f)
    annotations = f.split('\n\n')
    random.shuffle(annotations)
    n_examples = len(annotations)
    n_train = math.floor(n_examples*0.50)
    n_val =  math.floor(n_examples*0.25)
    n_test=  math.floor(n_examples*0.25)
    train = open(f'{output_directory}/{entity_type}_train.iob2', 'w', encoding='UTF-8')
    for i in range(0, n_train):
        if i!=n_train-1: train.write(annotations[i] + "\n\n")
        else: train.write(annotations[i])
    train.close() 

    dev = open(f'{output_directory}/{entity_type}_valid.iob2', 'w', encoding='UTF-8')
    for i in range(n_train, n_train+n_val):
        if i!=n_train+n_val-1: dev.write(annotations[i] +"\n\n")
        else: dev.write(annotations[i])
    dev.close()

    test = open(f'{output_directory}/{entity_type}_test.iob2', 'w', encoding='UTF-8')
    for i in range(n_train+n_val, n_examples):
        if i!=n_examples-1: test.write(annotations[i] +"\n\n")  
        else: test.write(annotations[i])  
    test.close() 
