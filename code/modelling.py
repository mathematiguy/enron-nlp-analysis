import click
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet, stopwords
from nltk import ngrams

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import re
import string
import joblib
import logging
import pyprojroot
from tqdm import tqdm

import itertools as it
import concurrent.futures
from multiprocessing import cpu_count

proj_root = pyprojroot.find_root(pyprojroot.has_file(".git"))
nltk.data.path.append("data/nltk_data/")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set numpy random seed
np.random.seed(42)


class TextPreprocessor:
    def __init__(
        self,
        use_lemmatize=False,
        use_stemmer=False,
        ngrams=1,
        remove_stopwords=True,
        random_state=42,
    ):
        """
        Initializes the TextPreprocessor with specified options.

        Args:
            use_lemmatize (bool, optional): Use lemmatization if True. Defaults to True.
            use_stem (bool, optional): Use stemming if True. Defaults to False.
            use_bigram (bool, optional): Generate bigrams if True, else unigrams. Defaults to False.
            remove_stopwords (bool, optional): Remove stopwords if True. Defaults to True.
        """
        if use_lemmatize and use_stemmer:
            raise ValueError("Cannot use both lemmatize and stem. Choose one.")

        self.use_lemmatize = use_lemmatize
        self.use_stemmer = use_stemmer
        self.ngrams = ngrams
        self.remove_stopwords = remove_stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words("english")) if remove_stopwords else set()
        self.random_state = random_state

    def preprocess(self, text):
        """
        Preprocesses the given text based on the initialization parameters.

        Args:
            text (str): The text to preprocess.

        Returns:
            list: List of preprocessed tokens or n-grams.
        """
        # Tokenize and clean
        tokens = [
            token.lower()
            for token in word_tokenize(text)
            if not re.findall("[“”'’`\d]+", token)
        ]

        # Apply lemmatization or stemming
        if self.use_lemmatize:
            pos_tags = pos_tag(tokens)
            tokens = [
                self.lemmatizer.lemmatize(token, pos=self.pos_normalizer(pos))
                for token, pos in pos_tags
            ]
        elif self.use_stemmer:
            tokens = [self.stemmer.stem(word) for word in tokens]

        # Remove stopwords and punctuation
        tokens = [
            token
            for token in tokens
            if token not in self.stopwords and token not in string.punctuation
        ]

        # Generate n-grams
        return [" ".join(gram) for gram in ngrams(tokens, self.ngrams)]

    @staticmethod
    def pos_normalizer(nltk_tag):
        """
        Normalizes the NLTK POS tags to WordNet POS tags.

        Args:
            nltk_tag (str): NLTK POS tag.

        Returns:
            str: WordNet POS tag.
        """
        if nltk_tag.startswith("J"):
            return wordnet.ADJ
        elif nltk_tag.startswith("V"):
            return wordnet.VERB
        elif nltk_tag.startswith("N"):
            return wordnet.NOUN
        elif nltk_tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN


def identity_fn(x):
    return x


class TextModelFactory:
    def __init__(
        self, preprocess_func, model_type="naive", hyperparam=1, random_state=42
    ):
        """
        Initializes the TextModelFactory with specified preprocessing function, model type, and hyperparameters.

        Args:
            preprocess_func (function): Function to preprocess the text data.
            model_type (str, optional): Type of model to train. Options: "naive", "svm", "logistic". Defaults to "naive".
            hyperparam (int, optional): Hyperparameter for the model. Defaults to 1.
        """
        self.preprocess_func = preprocess_func
        self.model_type = model_type
        self.hyperparam = hyperparam
        self.vectorizer = TfidfVectorizer(analyzer=identity_fn)
        self.classifier = None
        self.random_state = random_state

    def train_model(self, X_train, y_train):
        """
        Trains the model on the given training set.

        Args:
            X_train (list): List of training data.
            y_train (list): List of training labels.
        """
        # Check if preprocess_func is a method of an instance (e.g., TextPreprocessor)
        # and call its preprocess method.
        if isinstance(self.preprocess_func, TextPreprocessor):
            X_processed = [self.preprocess_func.preprocess(x) for x in X_train]
        else:
            # If it's a standalone function
            X_processed = [self.preprocess_func(x) for x in X_train]

        # Fit vectorizer
        X_vectorized = self.vectorizer.fit_transform(X_processed)

        # Initialize the classifier
        if self.model_type == "naive":
            self.classifier = ComplementNB(alpha=self.hyperparam)
        elif self.model_type == "svm":
            self.classifier = SVC(C=self.hyperparam, random_state=self.random_state)
        elif self.model_type == "logistic":
            self.classifier = LogisticRegression(
                C=self.hyperparam, random_state=self.random_state
            )

        # Fit the model
        self.classifier.fit(X_vectorized, y_train)

    def predict(self, X):
        """
        Predicts labels for the given data.

        Args:
            X (list): List of data to predict.

        Returns:
            ndarray: Predicted labels.
        """
        if isinstance(self.preprocess_func, TextPreprocessor):
            X_processed = [self.preprocess_func.preprocess(x) for x in X]
        else:
            # If it's a standalone function
            X_processed = [self.preprocess_func(x) for x in X]
        X_vectorized = self.vectorizer.transform(X_processed)
        return self.classifier.predict(X_vectorized)

    def compute_f1(self, X, y_true):
        """
        Computes the F1 score for the given data and true labels.

        Args:
            X (list): List of data.
            y_true (list): True labels.

        Returns:
            float: F1 score.
        """
        y_pred = self.predict(X)
        return precision_recall_fscore_support(y_true, y_pred, average="weighted")[2]

    def compute_accuracy(self, X, y_true):
        """
        Computes the accuracy for the given data and true labels.

        Args:
            X (list): List of data.
            y_true (list): True labels.

        Returns:
            float: Accuracy score.
        """
        if isinstance(self.preprocess_func, TextPreprocessor):
            X_processed = [self.preprocess_func.preprocess(x) for x in X]
        else:
            # If it's a standalone function
            X_processed = [self.preprocess_func(x) for x in X]
        X_vectorized = self.vectorizer.transform(X_processed)
        return self.classifier.score(X_vectorized, y_true)

    def save_model(self, filename):
        """
        Saves the trained model and vectorizer to disk.

        Args:
            filename (str): The base filename to save the model.
                            The method will append different suffixes for the model and vectorizer.
        """
        # Save the classifier
        classifier_filename = f"{filename}_classifier.joblib"
        joblib.dump(self.classifier, classifier_filename)
        print(f"Model saved to {classifier_filename}")

        # Save the vectorizer
        vectorizer_filename = f"{filename}_vectorizer.joblib"
        joblib.dump(self.vectorizer, vectorizer_filename)
        print(f"Vectorizer saved to {vectorizer_filename}")

        # Save preprocessor configuration
        preprocessor_config_filename = f"{filename}_preprocessor_config.joblib"
        preprocessor_config = {
            "use_lemmatize": self.preprocess_func.use_lemmatize,
            "use_stemmer": self.preprocess_func.use_stemmer,
            "ngrams": self.preprocess_func.ngrams,
            "remove_stopwords": self.preprocess_func.remove_stopwords,
        }
        joblib.dump(preprocessor_config, preprocessor_config_filename)
        print(f"Preprocessor config saved to {preprocessor_config_filename}")

    @staticmethod
    def load_model(filename):
        """
        Loads the model, vectorizer, and preprocessor config from disk.

        Returns:
            Tuple containing the classifier, vectorizer, and a TextPreprocessor instance.
        """
        classifier = joblib.load(f"{filename}_classifier.joblib")
        vectorizer = joblib.load(f"{filename}_vectorizer.joblib")
        preprocessor_config = joblib.load(f"{filename}_preprocessor_config.joblib")

        preprocessor = TextPreprocessor(**preprocessor_config)
        return classifier, vectorizer, preprocessor


def run_experiment(
    processor,
    model_type,
    hyperparam,
    X_train,
    y_train,
    X_valid,
    y_valid,
):
    processor_name = processor[0]
    processor = processor[1]
    preprocessor = TextPreprocessor(**processor)

    current_model = TextModelFactory(
        model_type=model_type, hyperparam=hyperparam, preprocess_func=preprocessor
    )

    current_model.train_model(X_train, y_train)
    f1_score = current_model.compute_f1(X_valid, y_valid)
    acc = current_model.compute_accuracy(X_valid, y_valid)
    model_path = f"data/model_runs/p={processor_name}-m={model_type}-h={hyperparam}"
    current_model.save_model(model_path)
    return (model_path, processor_name, model_type, hyperparam, f1_score, acc)


# CLI entry point
@click.command()
@click.option(
    "--train_file", default="train_set.csv", help="Path to the training set CSV file."
)
@click.option(
    "--valid_file", default="valid_set.csv", help="Path to the validation set CSV file."
)
@click.option(
    "--test_file", default="test_set.csv", help="Path to the test set CSV file."
)
def main(train_file, valid_file, test_file):
    """
    Processes email datasets for feature extraction and analysis.
    """
    logging.info("Loading datasets...")
    df_train = pd.read_csv(train_file, index_col="Original Index")
    df_valid = pd.read_csv(valid_file, index_col="Original Index")
    df_test = pd.read_csv(test_file, index_col="Original Index")

    X_train = df_train["Masked Email"]
    y_train = df_train["POI"]

    X_valid = df_valid["Masked Email"]
    y_valid = df_valid["POI"]

    X_test = df_test["Masked Email"]
    y_test = df_test["POI"]

    logging.info(
        f"Training: {X_train.shape[0]}, Validation: {X_valid.shape[0]}, Test: {X_test.shape[0]}"
    )

    # Mapping preprocessors to their descriptions
    preprocessors = {
        "lemmatize+unigram": {
            "use_lemmatize": True,
            "remove_stopwords": True,
        },
        "stem+unigram": {"use_stemmer": True, "remove_stopwords": True},
        "lemmatize+bigram": {
            "use_lemmatize": True,
            "ngrams": 2,
            "remove_stopwords": True,
        },
        "stem+bigram": {
            "use_stemmer": True,
            "ngrams": 2,
            "remove_stopwords": True,
        },
    }
    model_types = ["naive", "logistic"]
    hyperparams = [0.5, 1.0, 10.0]

    best_model = {
        "preprocessor": "lemmatize_unigram_preprocessor",
        "model": "naive",
        "hyperparam": 0.5,
        "f1_score": 0,
        "model": None,
    }

    def run_experiments_in_parallel(n_series=0):
        num_workers = cpu_count()

        results = []  # List to store the results of experiments
        experiments = list(it.product(preprocessors.items(), model_types, hyperparams))

        # Process the remaining experiments in parallel
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            future_to_experiment = {
                executor.submit(
                    run_experiment, *exp, X_train, y_train, X_valid, y_valid
                ): exp
                for exp in experiments[n_series:]
            }

            futures = concurrent.futures.as_completed(future_to_experiment)
            for future in tqdm(
                futures,
                total=len(experiments) - n_series,
                desc="Processing Experiments",
            ):
                experiment = future_to_experiment[future]
                try:
                    # Retrieve the results of the experiment
                    result = future.result()
                    results.append(result)
                    (
                        model_path,
                        processor_name,
                        model_type,
                        hyperparam,
                        f1_score,
                        acc,
                    ) = result
                    logging.info(
                        f"Processor={processor_name}, model_type={model_type}, hyperparam={hyperparam}, f1_score={f1_score}, acc={acc}"
                    )
                except Exception as exc:
                    processor, model_type, hyperparam = experiment
                    logging.error(
                        f"{processor}, {model_type}, {hyperparam} generated an exception: {exc}"
                    )

        return results

    results = run_experiments_in_parallel(n_series=1)

    experiment_data = pd.DataFrame(
        results,
        columns=[
            "model_path",
            "preprocessor",
            "model_type",
            "hyperparam",
            "val_f1_score",
            "val_acc",
        ],
    ).sort_values("val_f1_score", ascending=False)

    # Save experiment data
    experiment_data.to_csv("data/experiment_data.csv", index=False)
    print(experiment_data)


if __name__ == "__main__":
    main()
