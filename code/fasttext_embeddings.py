import sys
import logging
import argparse
import pandas as pd
import numpy as np
import typing
from typing import Any, Iterable
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
                import numpy as np
                doc = np.load(filepathIn, allow_pickle=True)
                pmids = []
                titles = []
                abstracts = []
                docs = []
                for line in doc:
                        pmids.append(np.ndarray.tolist(line[0]))
                        titles.append(np.ndarray.tolist(line[1]))
                        abstracts.append(np.ndarray.tolist(line[2]))
                        docs.append(np.ndarray.tolist(line[1]) + np.ndarray.tolist(line[2]))
                return (pmids, titles, abstracts, docs)


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


def create_document_embeddings(pmids: list, documents: list, model, output_dir_path: str) -> None:
    """
    Generates document embeddings from the generated Word2Vec model.
    Parameters
    ----------
    accessions : list
        List of accession numbers.
    functions : list
        List of function comments.
    model : 
        Pretraine Fasttext model.
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
    df.to_pickle(output_dir_path)
    print("Embeddings Generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input", type=str,
                       help="Path to input RELISH tokenized .npy file")
        parser.add_argument("-m", "--model", type=str,
                       help="Path to pre-trained model")               
        parser.add_argument("-o", "--output", type=str,
                       help="Path to save embeddings pickle file")                 
        args = parser.parse_args()
    pmids, titles, abstracts, docs = prepare_from_npy(args.input)
    model = load_pretrained_model(args.model)
    create_document_embeddings(pmids, docs, model, args.output)