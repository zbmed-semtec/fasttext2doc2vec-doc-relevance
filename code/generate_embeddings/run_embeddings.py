import os
import argparse
import itertools
import logging
import embeddings as em


log_file = "fasttext_trained.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def run(params_dict: dict, input_file: str):
    """
    Wrapper function to create Tagged Documents, generate & train fastText models, generate embeddings and
    stores embeddings in pickle file.
    Parameters
    ----------
    params_dict : dict
        Parameter dictionary consisting of all hyperparameters, combinations of which are used to generate fastText models.
    input_file : str
        Input RELISH tokenized npy file.
    """
    all_combinations = list(itertools.product(*params_dict.values()))

    param_combinations = [{key: value for key, value in zip(params_dict.keys(), combination)} for combination in all_combinations]
    # Retrieves cleaned data from .npy file 
    pmids, titles, abstracts, docs = em.prepare_from_npy(input_file)
    logging.info("Retrieved RELISH Cleaned Data")

    for idx, params in enumerate(param_combinations):
        logging.info(f"Combination {idx + 1}/{len(all_combinations)}")
        # Create and train Doc2Vec model
        model = em.create_fasttext_model(pmids, docs, params)
        logging.info("RELISH fastText Model Generated")

        # Define a directory for storing models
        models_directory = f"data/models/"

        # Ensure the directory exists or create it
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)

        # Save the model generated
        em.save_model(model, os.path.join(models_directory, f"fasttext_{idx}.model"))
        logging.info("RELISH Fasttext Model Saved")

        # Define a directory for storing embeddings
        embeddings_directory = f"data/embeddings/"

        # Ensure the directory exists or create it
        if not os.path.exists(embeddings_directory):
            os.makedirs(embeddings_directory)
            
        embeddings_filename = f"embeddings_fasttext_{idx}.pkl"

        # Generate the embeddings
        em.create_document_embeddings(pmids, docs, model, os.path.join(embeddings_directory, embeddings_filename))
        logging.info("RELISH Embeddings Generated and Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                       help="Path to input RELISH tokenized .npy file")                 
    args = parser.parse_args()

    params_dict = {
    "sg": [0, 1],
    "vector_size": [200, 300, 400],
    "window": [5, 6, 7],
    "min_count": [5],
    "epochs": [15],
    "workers": [8]
    }

    run(params_dict, args.input)