import os
import sys
import yaml
import itertools
import logging
import argparse
import pandas as pd
import numpy as np
import typing
from typing import Any, List, Iterable
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model


def prepare_from_npy(filepathIn=None):
        '''
        Retrieves data from RELISH npy files, separating each column into their own respective list.

        Parameters
        ----------
        filepathIn: str
                The filepath of the RELISH or TREC input npy file.

        Returns
        -------
        list of str
                All pubmed ids associated to the paper.
        list of str
                All words within the title.
        list of str
                All words within the abstract.
        '''
        if not isinstance(filepathIn, str):
                logging.alert("Wrong parameter type for prepareFromTSV.")
                sys.exit("filepathIn needs to be of type string")
        else:
                doc = np.load(filepathIn, allow_pickle=True)
                pmids = []
                titles = []
                abstracts = []
                docs = []
                for line in doc:
                    pmids.append(int(line[0]))
                    if isinstance(line[1], (np.ndarray, np.generic)):
                        titles.append(np.ndarray.tolist(line[1]))
                        abstracts.append(np.ndarray.tolist(line[2]))
                        docs.append(np.ndarray.tolist(
                            line[1]) + np.ndarray.tolist(line[2]))
                    else:
                        titles.append(line[1])
                        abstracts.append(line[2])
                        docs.append(line[1] + line[2])
                return (pmids, titles, abstracts, docs)

def generate_param_combinations(params):
    param_keys = []
    param_values = []
    
    for key, value in params.items():
        if 'values' in value:  # Check if 'values' exist in this parameter
            param_keys.append(key)
            param_values.append(value['values'])
        else:
            param_keys.append(key)
            param_values.append([value['value']])  # Use the single value as a list
    
    param_combinations = [dict(zip(param_keys, combination)) 
                          for combination in itertools.product(*param_values)]
    
    return param_combinations


def load_pretrained_model(model_filepath: str):
    """
    Loads the pre-trained model.
    Parameters
    ----------
    model_filepath : str
        Filepath of the downloaded pre-trained model.
    """    
    model = load_facebook_model(model_filepath)
    return model


def create_fasttext_model(pmids: List[str], docs: List[List[str]], params: dict) -> FastText:
    """
    Create and train the fastText model using Gensim for the documents 
    in the corpus.

    Parameters
    ----------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    params: dict
            Dictionary containing the parameters for the fastText model.
    Returns
    -------
    model: fastText
            fastText model.
    """
    model = FastText(**params)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

    return model


def save_model(model: FastText, output_file: str) -> None:
    """
    Saves the fastText model.

    Parameters
    ----------
    model: fastText
            fastText model.
    output_file: str
            File path of the fastText model generated.
    """
    model.save(output_file)


def create_document_embeddings(pmids: list, documents: list, model: FastText, iteration: int, output_dir_path: str) -> None:
    """
    Generates document embeddings from the generated Word2Vec model.
    Parameters
    ----------
    accessions : list
        List of accession numbers.
    documents : list
        List of function comments.
    model : fastText
        Pretrained FastText model.
    iteration: int
        Hyperparameter configuration number.
    output_dir_path: str
        File path for the generated embeddings.
    """
    document_embeddings = []

    for index in range(len(pmids)):
        embeddings_list = []
        for word in documents[index]:
            try:
                embeddings_list.append(model.wv[word])
            except:
                continue
        #  Generate document embeddings from word embeddings
        first = True
        document = []
        for embedding in embeddings_list:
            if first:
                for dimension in embedding:
                    document.append(0.0)
                first = False
            doc_dimension = 0
            for dimension in embedding:
                document[doc_dimension] += dimension
                doc_dimension += 1
        doc_dimension = 0
        for dimension in document:
            # Get the average of each dimension of the embeddings and store it in the document list
            document[doc_dimension] = (dimension / len(embeddings_list))
            doc_dimension += 1
        document_embeddings.append(document)


    df = pd.DataFrame(list(zip((pmids), document_embeddings)), columns =['pmids', 'embeddings'])
    df = df.sort_values('pmids')
    os.makedirs(f"{output_dir_path}", exist_ok=True)
    df.to_pickle(f'{output_dir_path}/embeddings_{iteration}.pkl') 
    print("Embeddings Generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to input RELISH tokenized .npy file")
    parser.add_argument("-o", "--output", type=str, help="Path to save embeddings pickle file")                 
    parser.add_argument("-p", "--params", type=str, help="Path to hyperparameter yaml file.")
    args = parser.parse_args()

    params = []
    with open(args.params, "r") as file:
        content = yaml.safe_load(file)
        params = content['params']

    param_combinations = generate_param_combinations(params)
    model_output_file_base = "./data/models/fasttext_model"
    model_output_dir = os.path.dirname(model_output_file_base)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    pmids, titles, abstracts, docs = prepare_from_npy(args.input)
    for i, param_set in enumerate(param_combinations):
        print(f"Training model with hyperparameters: {param_set}")
        model = create_fasttext_model(pmids, docs, param_set)
        model_output_file = f"{model_output_file_base}_{i}"
        save_model(model, model_output_file)
        create_document_embeddings(pmids, docs, model, i, args.output)
