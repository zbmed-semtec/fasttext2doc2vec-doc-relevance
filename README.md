# FastText2Doc2Vec-Doc-relevance

This repository focuses on an approach that uses the [FastText library](https://fasttext.cc/docs/en/support.htm) for generating word embeddings and subsequently employs a centroid aggregration technique to produce document-level embeddings. This process involves calculating the centroids of word embeddings found in the titles and abstracts of each document. The approach is applied to evaluate literature-based document-to-document recommendations using the RELISH dataset.

## Input Data
The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the RELISH.npy file, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, text is converted to lowercase, stop words are removed and finally, tokenization is employed, resulting in arrays of individual words.

## Downloading the pre-trained model

We employ a pretrained FastText model to generate word embeddings. To download this model using the command line, follow these steps:

1. Clone the FastText repository:

 ``` 
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
 ``` 

2. Once the FastText is successfully built, navigate to the FastText directory and execute the following command to download the English model:

``` 
$ ./download_model.py en
``` 

3. After the download is complete, you will find the **'cc.en.300.bin.gz'** model file located in the FastText directory, accessible at the following path:

``` 
'./fastText/cc.en.300.bin.gz'
``` 

## Code Execution

This script loads the preprocessed input file, extracting the PubMed IDs, titles, and abstracts. It then loads the pre-trained FastText model and generates document embeddings through FastText and centroid calculations using the title and abstract and saves the embeddings as a pickle file.

To execute this script, please use the following command:

`python3 code/fasttext_embeddings.py --input data/RELISH_tokenized.npy --model fastText/cc.en.300.bin.gz --output data/embeddings.pkl`

