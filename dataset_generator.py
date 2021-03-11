import gzip
import json
import argparse
import struct
from nltk import word_tokenize
from re import sub
import tensorflow as tf
from tensorflow.core.example import example_pb2

def generate_dataset(src_path, des_path):
    dataset = load_dataset(src_path)

    with open(des_path, 'wb') as w:
        for item in dataset:
            mid = '<s>' + item['mid'] + '</s>'
            abstract = item['abstract']
            abstract = abstract.lower()
            abstract = ' '.join(word_tokenize(abstract))

            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([abstract.encode('utf-8')])
            tf_example.features.feature['abstract'].bytes_list.value.extend([mid.encode('utf-8')])

            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            w.write(struct.pack('q', str_len))
            w.write(struct.pack('%ds' % str_len, tf_example_str))

def load_dataset(filepath):
    with gzip.GzipFile(filename=filepath, mode='r') as reader:
        json_bytes = reader.read() 

    json_str = json_bytes.decode('utf-8')            
    dataset = json.loads(json_str)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset generator script")
    parser.add_argument("-s",
                        dest="src_path", 
                        required=True,
                        help="File path for source json dataset (default: None).")
    parser.add_argument("-d",
                        dest="des_path", 
                        required=True,
                        help="Destination file path (default: None).")
    args = parser.parse_args()
    
    generate_dataset(args.src_path, args.des_path)