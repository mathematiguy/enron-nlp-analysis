import click
import pandas as pd
import re
import spacy
import logging
from tqdm import tqdm
from multiprocessing import cpu_count
from utils import parallel_batch_apply

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_nlp_model():
    """
    Loads the spaCy NLP model and sets up a custom EntityRuler with predefined patterns.

    Returns:
        Loaded spaCy NLP model with custom EntityRuler.
    """
    # Loading spaCy NLP model and setting up EntityRuler...
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = (
        2000000  # Increase the character limit to 2 million characters, for example
    )
    name_ruler = nlp.add_pipe("entity_ruler")
    patterns = [
        {"label": "PERSON", "pattern": [{"lower": "delainey"}]},
        {"label": "PERSON", "pattern": [{"lower": "jmf"}]},
        {"label": "PERSON", "pattern": [{"lower": "dave"}]},
        {"label": "PERSON", "pattern": [{"lower": "forney"}]},
        {"label": "PERSON", "pattern": [{"lower": "lloyd"}]},
        {"label": "PERSON", "pattern": [{"lower": "phillip"}]},
        {"label": "PERSON", "pattern": [{"lower": "tj"}]},
        {"label": "ORG", "pattern": [{"lower": "ercot"}]},
    ]
    name_ruler.add_patterns(patterns)
    return nlp


def change_ents(doc):
    """
    Replaces named entities in a document based on predefined replacements.

    Args:
        doc (spaCy Doc): A spaCy document object.

    Returns:
        str: The text of the document with entities replaced.
    """
    # Entity replacements mapping
    ent_replacements = {
        "PERSON": "Steve",
        "ORG": "Apple",
        "GPE": "Cupertino",
    }

    # Compile regex patterns for entity replacements
    regex_ent_replacements = {key: "" for key in ent_replacements}
    for ent in doc.ents:
        if ent.label_ in ent_replacements:
            text = re.escape(re.sub(r"\(.*|\).*|\+.*", "", ent.text))
            regex_ent_replacements[ent.label_] += f"|{text}"
    regex_ent_replacements = {
        key: val[1:] for key, val in regex_ent_replacements.items() if val
    }

    # Replace entities in the text
    new_text = doc.text
    for ent_label, regex in regex_ent_replacements.items():
        try:
            new_text = re.sub(
                regex, ent_replacements[ent_label], new_text, flags=re.IGNORECASE
            )
        except re.error as e:
            logging.error(f"Regex error with pattern {regex}: {e}")
            continue
    new_text = re.sub(r"[ \n\t]+", " ", new_text)
    return new_text.strip()


def process_texts(batch):
    """
    Processes a batch of texts using the spaCy NLP model and replaces named entities.

    Args:
        batch (list): A list of texts to process.

    Returns:
        list: A list of processed texts with entities replaced.
    """
    nlp = load_nlp_model()
    return [change_ents(text) for text in nlp.pipe(batch)]


def main():
    """
    Main function to process emails from a file and save the results to an output file.

    Args:
        email_file (str): Path to the input email file.
        output (str): Path for the output file.
    """
    for dataset in ["norm", "exec", "poi"]:
        email_file = f"data/{dataset}_emails.csv"
        logging.info(f"Reading email data from {email_file}...")
        df = pd.read_csv(email_file, index_col="Original Index")

        logging.info(f"Starting {dataset} email processing...")
        replaced_emails = parallel_batch_apply(
            df["Email"],
            process_texts,
            batch_size=len(df) // 200,
            cores=min([16, cpu_count()]),
        )

        df["Masked Email"] = replaced_emails

        output = f"data/{dataset}_masked_emails.csv"
        logging.info(f"Saving masked emails to {output}...")
        df.to_csv(output)

        logging.info("Masking complete. Output saved.")


if __name__ == "__main__":
    main()
