import os
import pandas as pd
import click
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import chardet
import logging

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def decode_email(fp):
    """
    Decodes the content of an email file.

    Args:
    fp (str): File path of the email file.

    Returns:
    str: Decoded email content.
    """
    with open(fp, "rb") as f:
        raw_data = f.read()

        # Detect and use the correct encoding
        detected_encoding = chardet.detect(raw_data)["encoding"]
        if detected_encoding is None:
            detected_encoding = (
                "us-ascii"  # Default to 'us-ascii' if encoding is undetected
            )

        try:
            text = raw_data.decode(detected_encoding)
        except UnicodeDecodeError:
            text = raw_data.decode("us-ascii", errors="replace")

    return text.replace("\r", "")


def read_email(fp):
    """
    Reads and parses the content of an email.

    Args:
    fp (str): File path of the email file.

    Returns:
    dict: Parsed email content in key-value pairs.
    """
    text = decode_email(fp)

    header, content = text.split("\n\n", 1)

    # Define the fields we are interested in
    fields = [
        "Message-ID",
        "Date",
        "From",
        "Subject",
        "X-FileName",
        "X-Origin",
        "X-Folder",
        "X-bcc",
        "X-cc",
        "X-To",
        "X-From",
        "Content-Transfer-Encoding",
        "Content-Type",
        "Mime-Version",
        "To",
        "Cc",
        "Bcc",
        "Content",
    ]

    email_dict = {field: "" for field in fields}
    email_dict["Content"] = content

    current_key = None
    lines = header.strip().split("\n")
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()

            if key in email_dict or current_key is None:
                email_dict[key] = value.strip()
                current_key = key
            else:
                email_dict[current_key] += " " + line.strip()
        elif current_key:
            email_dict[current_key] += " " + line.strip()

    return email_dict


def collect_paths(directory):
    """
    Collects file paths of all files in a given directory.

    Args:
    directory (str): Directory to search in.

    Returns:
    list: List of file paths.
    """
    paths = []
    for root, dirs, files in os.walk(directory, followlinks=True):
        for f in files:
            paths.append(os.path.join(root, f))
    return paths


def process_emails(paths):
    """
    Processes a list of email file paths to extract their content.

    Args:
    paths (list): List of email file paths.

    Returns:
    DataFrame: DataFrame containing the paths and parsed content of emails.
    """
    logging.info(f"Processing {len(paths)} emails...")
    enron_data = pd.DataFrame({"path": paths})

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_email, fp): fp for fp in paths}
        emails = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            emails.append(future.result())

    enron_data["email"] = emails
    return enron_data


def generate_parquet(enron_data, output_file):
    """
    Collects fields from the processed emails and saves it to parquet.

    Args:
    enron_data (DataFrame): DataFrame containing the parsed emails.
    output_file (str): Path to the output file.
    """
    logging.info(f"Generating report...")
    fields = pd.json_normalize(enron_data.email)
    enron_df = pd.concat([enron_data.loc[:, ["path"]], fields], axis=1)
    enron_df.to_parquet(output_file)
    logging.info(f"Report saved to {output_file}")


@click.command()
@click.option(
    "--input_dir", default="data/maildir", help="Directory containing the emails."
)
@click.option(
    "--output",
    default="data/enron_emails.parquet",
    help="Output file for the processed emails.",
)
def main(input_dir, output):
    """
    Main function to process emails and generate a report.

    Args:
    input_dir (str): Directory containing the emails.
    output (str): Path for the output file.
    """
    logging.info(f"Starting email processing...")
    paths = collect_paths(input_dir)
    enron_data = process_emails(paths)
    generate_parquet(enron_data, output)
    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
