import os 
import pandas as pd
from collections import defaultdict

def get_tweets_content(directory):
    content = {}
    enc_directory = os.fsencode(directory) 
    for file in os.listdir(enc_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            full_path = os.path.join(directory, filename)
            name = filename.split('.')[0]
            content[name] = open(full_path, 'r', encoding = 'UTF-8').read()
    
    content = dict(sorted(content.items()))
    print(f'Number of documents found: {len(content)}')
    return content

def get_annotations(directory, ids):

    annotation_dict = {'training': defaultdict(list), 'validation': defaultdict(list)}
 
    df = pd.read_csv(directory, sep='\t', encoding='UTF-8')
    for index, row in df.iterrows():
        filename = str(row[0])
        start_index = int(row[1])
        end_index = int(row[2])
        label = row[3]
        span = row[4]

        if filename in ids['training']:
            annotation_dict['training'][filename].append({"start_index": start_index, "end_index": end_index, "label": label, "span": span})
        if filename in ids['validation']:
            annotation_dict['validation'][filename].append({"start_index": start_index, "end_index": end_index, "label": label, "span": span})

    annotation_dict['training'] = dict(sorted(annotation_dict['training'].items()))
    annotation_dict['validation'] = dict(sorted(annotation_dict['validation'].items()))
    return annotation_dict['training'], annotation_dict['validation']

def check_offsets(documents, annotations):
    for filename, content in documents.items():
        file_annotations = annotations[filename]
        for entity in file_annotations:
       
            start_index = entity["start_index"]
            end_index = entity["end_index"]
            span = entity["span"]
            original_span = content[start_index:end_index]
            if span != original_span:
                print("There is an inconsistency between the annotation indexes and the original text.")
                print("Filename: {}".format(filename))
                print("Original text: {} does not match with annotated entity {}".format(original_span, span))
                