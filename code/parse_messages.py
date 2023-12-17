import click
import numpy as np
import pandas as pd

import os
import re
import sys
import yaml
import codecs
import logging
import datetime
import pyprojroot

import nltk
from tqdm import tqdm
from utils import parallel_apply

proj_root = pyprojroot.find_root(pyprojroot.has_file(".git"))
nltk.data.path.append("data/nltk_data/")

from nltk.tokenize import word_tokenize

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_emails(df_emails_path):
    return pd.read_parquet(df_emails_path)


def remove_fw_re_subjects(df_emails):
    re_emails = df_emails["Subject"].str.lower().str.contains("re:")
    fw_emails_1 = df_emails["Subject"].str.lower().str.contains("fw:")
    fw_emails_2 = df_emails["Subject"].str.lower().str.contains("fwd:")
    return df_emails[~(re_emails | fw_emails_1 | fw_emails_2)]


def trim_reply_emails(email):
    patterns = [
        "[- ]*Original Message",
        "[- ]*Forwarded ",
        "From:\t",
        "To:\t",
        "To: ",
        "Do You Yahoo!?",
        "[- ]*Sent from my BlackBerry",
    ]
    for pattern in patterns:
        match = re.search(pattern, email)
        if match:
            email = email[: match.start()].strip()
    return email


def process_emails(df_emails):
    re_emails = df_emails["Subject"].str.lower().str.contains("re:")
    fw_emails_1 = df_emails["Subject"].str.lower().str.contains("fw:")
    fw_emails_2 = df_emails["Subject"].str.lower().str.contains("fwd:")
    fw_emails = fw_emails_1 | fw_emails_2

    # trim the emails and ignore the forwards
    good_emails = parallel_apply(df_emails[~fw_emails]["Content"], trim_reply_emails)

    # drop any null emails
    good_emails = good_emails[good_emails.str.len() > 0]

    df_emails = (
        good_emails.rename("Email Trimmed")
        .to_frame()
        .join(df_emails[[col for col in df_emails if col != "Email"]], how="left")
    )
    return df_emails


def find_possible_email_addresses(df_emails, names):
    addresses = {}
    for name in names:
        pattern = name.lower()
        found_addresses = df_emails[df_emails["From"].str.contains(pattern, na=False)][
            "From"
        ].unique()
        addresses[name] = found_addresses
    return addresses


def assign_labels_and_sender(df_emails, person_info):
    # Assign POI labels
    df_emails["POI"] = df_emails["From"].isin(sum(person_info["POI"].values(), []))

    # Assign Exec 200 labels
    exec_200_addrs = sum(person_info["Executives"]["Over200k"].values(), [])
    df_emails["Exec 200"] = df_emails["From"].isin(exec_200_addrs)

    # Assign Exec 300 labels
    exec_300_addrs = sum(person_info["Executives"]["Over300k"].values(), [])
    df_emails["Exec 300"] = df_emails["From"].isin(exec_300_addrs)

    # Initialize the 'Sender' column with a string data type instead of NaN
    df_emails["Sender"] = pd.NA

    for name, addresses in {**person_info["POI"], **person_info["Executives"]}.items():
        df_emails.loc[df_emails["From"].isin(addresses), "Sender"] = name

    return df_emails


def get_sender(e):
    """
    Add sender information for the normal people
    """
    if "@enron.com" in e:
        i = e.split("@")[0]
        if "." in i:
            return i.split(".")[1].capitalize()
    return np.nan


def ascii_conversion(s):
    return codecs.encode(s, "ascii", "ignore").decode()


def check_min_word_count(s, l=5):
    return len(word_tokenize(s)) > l


def secretary_filtering(name, messages):
    if name == "Lay":
        return messages[~messages.apply(lambda s: "Rosie" in s or "Rosalee" in s)]
    elif name == "Skilling":
        return messages[
            ~messages.apply(lambda s: "Sherri" in s or "Joannie" in s or "SRS" in s)
        ]
    return messages


def filter_secretaries(df_emails, name, processing_func):
    """
    Apply additional processing for emails associated with a specific name.

    Args:
        df_emails (DataFrame): DataFrame containing email data.
        name (str): The name to apply specific processing to.
        processing_func (function): The function to apply for processing.

    Returns:
        DataFrame: Processed DataFrame for the specific name.
    """
    logging.info(f"Applying additional processing for {name}")
    return processing_func(df_emails.loc[df_emails["Sender"] == name, :])


def filter_emails(df_processed):
    """
    Filters emails based on specific criteria.

    Args:
        df_processed (DataFrame): The DataFrame to be filtered.

    Returns:
        DataFrame: The filtered DataFrame.
    """
    # Drop emails less than 5 words long
    logging.info("Dropping emails with less than 5 words")
    df_filtered = df_processed[
        parallel_apply(df_processed["Email Trimmed"], check_min_word_count)
    ].copy()

    # Filter out emails sent before 1999
    logging.info("Filtering out emails sent before 1999")
    date_format = "%a, %d %b %Y %H:%M:%S "
    df_filtered["Datetime"] = pd.to_datetime(
        df_filtered["Date"].apply(lambda x: x.split("-")[0]),
        format=date_format,
        errors="coerce",
    )
    df_filtered = df_filtered[
        (df_filtered["Datetime"] > datetime.datetime(1998, 12, 31))
        & (~df_filtered["Datetime"].isna())
    ]

    return df_filtered


def clean_and_format_data(df_emails, df_processed):
    """
    Cleans and formats the DataFrame by joining and renaming columns.

    Args:
        df_emails (DataFrame): Original DataFrame of emails.
        df_processed (DataFrame): Processed DataFrame of emails.

    Returns:
        DataFrame: The cleaned and formatted DataFrame.
    """
    # Join with original dataframe and drop duplicates
    processed_df = (
        df_processed["Email Trimmed"]
        .to_frame()
        .join(
            df_emails[[col for col in df_emails.columns if col != "Email Trimmed"]],
            how="left",
        )
    )
    processed_df = processed_df[
        ["Email Trimmed", "Sender", "POI", "Exec 200", "Exec 300", "Date"]
    ].drop_duplicates()
    processed_df = processed_df.rename(columns={"Email Trimmed": "Email"})
    return processed_df.sort_index()


def preprocess_emails(df, normal_person_names):
    """
    Preprocess a DataFrame of emails by applying specific filtering and cleaning rules.
    """

    logging.info("Starting preprocessing of emails.")

    # Define processing functions for specific names

    # Remove Rosalee Fleming emails; Lay's secretary
    lay_processing = lambda messages: messages[
        ~messages.apply(lambda s: ("Rosie" in s) or ("Rosalee" in s))
    ]
    # Remove Sherri Sera, Joannie Williamson, SRS; Skilling's secretary
    skilling_processing = lambda messages: messages[
        ~messages.apply(lambda s: ("Sherri" in s) or ("Joannie" in s) or ("SRS" in s))
    ]
    secretary_filter = {"Lay": lay_processing, "Skilling": skilling_processing}

    # Process emails for specific names
    processed_emails = []
    for name in normal_person_names:
        if name in secretary_filter:
            filtered_emails = filter_secretaries(df, name, secretary_filter[name])
            processed_emails.append(filtered_emails)
        else:
            processed_emails.append(df.loc[df["Sender"] == name, :])
    df_processed = pd.concat(processed_emails) if processed_emails else df.copy()

    # Filter and clean the DataFrame
    df_filtered = filter_emails(df_processed)
    processed_df = clean_and_format_data(df_processed, df_filtered)

    logging.info("Email preprocessing completed.")
    return processed_df


def save_emails(df_emails, filepath):
    df_emails.to_csv(filepath, index_label="Original Index")


@click.command()
@click.option(
    "--emails_file",
    default="data/enron_emails.parquet",
    help="Path to parquet emails file",
)
@click.option(
    "--person_yaml",
    default="data/person_info.yaml",
    help="Path to person info yaml file",
)
@click.option(
    "--output_dir",
    default="data/",
    help="Folder to save the processed emails.",
)
def main(emails_file, person_yaml, output_dir):
    """
    Main function to process emails.

    Parameters:
    emails_file (str): Path to the emails file.
    person_yaml (str): Path to the YAML file containing person information.
    output (str): Path for the output CSV file.
    """

    # Step 1: Read emails from the dataset
    logging.info(f"Reading emails from {emails_file}")
    df_emails = read_emails(emails_file)

    # Step 2: Remove forwarded and replied emails
    logging.info("Removing forwarded and replied emails")
    df_emails = remove_fw_re_subjects(df_emails)

    # Step 3: Process emails (trimming replies and forwards)
    logging.info("Processing emails (trimming replies and forwards)")
    df_emails = process_emails(df_emails)

    # Load person info from YAML file
    logging.info(f"Loading person info from {person_yaml}")
    try:
        person_info = yaml.load(open(person_yaml), Loader=yaml.FullLoader)
    except Exception as e:
        logging.error(f"Failed to load person info from {person_yaml}: {e}")
        return

    # Step 4: Assign labels and sender information
    logging.info("Assigning labels and sender information")
    df_emails = assign_labels_and_sender(df_emails, person_info)

    # Step 5: Additional processing for POI, Execs, and Normal people
    logging.info("Separating DataFrame slices for POI, Execs, and Normal people")
    df_poi = df_emails[df_emails["POI"]].copy()
    df_exec = df_emails[df_emails["Exec 200"]].copy()
    df_norm = df_emails[(~df_emails["POI"]) & (~df_emails["Exec 200"])].copy()

    # Add sender information for normal people
    logging.info("Adding sender information for normal emails")
    df_norm["Sender"] = df_norm["From"].apply(lambda x: get_sender(x))

    # Process POI emails (ASCII conversion, filtering, etc.)
    logging.info("Processing POI emails")
    df_poi["Email Trimmed"] = df_poi["Email Trimmed"].apply(ascii_conversion)

    # Step 6: Preprocess and filter emails for normal people
    logging.info("Preprocessing and filtering normal people emails")
    df_norm = preprocess_emails(df_norm, person_info["Names"])
    df_exec = preprocess_emails(df_exec, [])
    df_poi = preprocess_emails(df_poi, [])

    # Step 7: Save the processed emails to CSV
    norm_path = os.path.join(output_dir, "norm_emails.csv")
    logging.info(f"Saving norm emails to {norm_path}")
    save_emails(df_norm, norm_path)

    exec_path = os.path.join(output_dir, "exec_emails.csv")
    logging.info(f"Saving exec emails to {exec_path}")
    save_emails(df_exec, exec_path)

    poi_path = os.path.join(output_dir, "poi_emails.csv")
    logging.info(f"Saving poi emails to {poi_path}")
    save_emails(df_poi, poi_path)

    logging.info("Email processing completed successfully.")


if __name__ == "__main__":
    main()
