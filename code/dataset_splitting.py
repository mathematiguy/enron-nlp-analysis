import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
import sys
import logging
import pyprojroot

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_email_samples(df, email_cap=4000, max_sample_size=700):
    """
    Generate a sample of emails from a DataFrame.

    Args:
        df (DataFrame): DataFrame containing emails.
        email_cap (int): Maximum number of emails to sample.
        max_sample_size (int): Maximum sample size per unique sender.

    Returns:
        DataFrame: Sampled subset of emails.
    """
    df_senders_list = []
    total_emails = 0
    for sender in df["Sender"].unique():
        df_tmp = df[df["Sender"] == sender]
        sample_size = np.min([max_sample_size, df_tmp.shape[0]])
        df_tmp = df_tmp.sample(sample_size)
        df_senders_list.append(df_tmp)
        total_emails += sample_size
        if total_emails > email_cap:
            break
    return pd.concat(df_senders_list)


def date_standardizer(s):
    """
    Standardize date format in a string.

    Args:
        s (str): Date string.

    Returns:
        str: Standardized date string.
    """
    s = s.strip()
    m = re.search("[\d]+", s)
    if (m.end() - m.start()) == 1:
        return re.sub(f" {m.group()} ", f" 0{m.group()} ", s)
    return s


@click.command()
@click.option(
    "--poi_file", default="data/poi_emails.csv", help="Path to the POI emails CSV file."
)
@click.option(
    "--exec_file",
    default="data/exec_emails.csv",
    help="Path to the Exec emails CSV file.",
)
@click.option(
    "--norm_file",
    default="data/normal_emails.csv",
    help="Path to the Normal emails CSV file.",
)
@click.option(
    "--output_dir",
    default=".",
    help="Directory to save the train, validation, and test sets.",
)
def main(poi_file, exec_file, norm_file, output_dir):
    """
    Processes email datasets and splits them into training, validation, and test sets.
    """
    logging.info("Loading email datasets...")
    df_poi = pd.read_csv(poi_file, index_col="Original Index")
    df_exec = pd.read_csv(exec_file, index_col="Original Index")
    df_norm = pd.read_csv(norm_file, index_col="Original Index")

    # Sample emails
    logging.info("Sampling emails...")
    email_sample_size = 700
    num_exec_emails = 3000
    num_norm_emails = 9000
    df_exec_samples = get_email_samples(
        df_exec, email_cap=num_exec_emails, max_sample_size=email_sample_size
    )
    df_norms_samples = get_email_samples(
        df_norm, email_cap=num_norm_emails, max_sample_size=email_sample_size
    )

    # Concatenate and sort datasets
    df = pd.concat([df_poi, df_exec_samples, df_norms_samples]).sort_index()

    # Standardize date and split datasets
    logging.info("Standardizing dates and splitting dataset...")
    df["Date"] = df["Date"].apply(lambda x: date_standardizer(x))
    df["Datetime"] = pd.to_datetime(
        df["Date"].str.split("-").apply(lambda x: x[0]), format="%a, %d %b %Y %H:%M:%S "
    )
    y_full = df["POI"]
    X_full = df[[col for col in df.columns if col != "POI"]]

    # Splitting the dataset
    test_size = 3310
    valid_size = 1986
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, stratify=y_full
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=valid_size, stratify=y_train
    )

    # Saving datasets
    logging.info("Saving datasets...")
    X_train["POI"] = y_train
    X_valid["POI"] = y_valid
    X_test["POI"] = y_test
    X_train.to_csv(os.path.join(output_dir, "train_set.csv"))
    X_valid.to_csv(os.path.join(output_dir, "valid_set.csv"))
    X_test.to_csv(os.path.join(output_dir, "test_set.csv"))
    logging.info("Datasets saved successfully in the specified output directory.")


if __name__ == "__main__":
    main()
