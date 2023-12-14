import requests
from bs4 import BeautifulSoup
import pandas as pd
import click
import logging

# Setting up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def fetch_and_parse(url):
    """
    Fetch and parse the HTML content from a given URL.

    Args:
    url (str): URL of the webpage to fetch.

    Returns:
    BeautifulSoup: Parsed HTML content of the page.
    """
    logging.info(f"Fetching webpage content from {url}")
    response = requests.get(url)
    webpage = response.content
    soup = BeautifulSoup(webpage, "html.parser")
    return soup


def extract_table(soup):
    """
    Extract and process the table from the parsed HTML content.

    Args:
    soup (BeautifulSoup): Parsed HTML content of the webpage.

    Returns:
    DataFrame: Processed data in a pandas DataFrame.
    """
    logging.info("Extracting and processing the table...")
    table = soup.find("table")

    def process_row(row, is_header=False):
        row_data = []
        for cell in row.find_all(["td", "th"]):
            if cell.find("b") and not is_header:
                name = cell.find("b").get_text(strip=True)
                title = cell.get_text(strip=True).replace(name, "", 1).strip()
                row_data.append({"Name": name, "Title": title})
            else:
                row_data.append(cell.get_text(strip=True))
        return row_data

    header_row = table.find("tr")
    headers = process_row(header_row, is_header=True)
    data_rows = [process_row(row) for row in table.find_all("tr")[1:]]

    convictions_df = pd.DataFrame(data_rows, columns=headers)
    convictions_df["Name"] = convictions_df["Name /Title"].apply(
        lambda x: x["Name"] if isinstance(x, dict) else None
    )
    convictions_df["Title"] = convictions_df["Name /Title"].apply(
        lambda x: x["Title"] if isinstance(x, dict) else None
    )
    convictions_df.drop("Name /Title", axis=1, inplace=True)
    convictions_df.rename(
        columns={convictions_df.columns[0]: "Employee Level"}, inplace=True
    )

    current_level = None
    for index, row in convictions_df.iterrows():
        if row["Employee Level"] and row["Name"] is None:
            current_level = row["Employee Level"]
        else:
            convictions_df.at[index, "Employee Level"] = current_level

    convictions_df = convictions_df[convictions_df["Name"].notna()]

    convictions_df["First Name"] = (
        convictions_df.Name.str.replace("F.", "")
        .str.strip()
        .apply(lambda x: x.split(" ", 1)[0])
    )
    convictions_df["Last Name"] = (
        convictions_df.Name.str.replace("Jr.", "")
        .str.strip()
        .apply(lambda x: x.rsplit(" ", 1)[-1])
    )
    convictions_df["Email"] = (
        convictions_df["First Name"] + "." + convictions_df["Last Name"] + "@enron.com"
    ).str.lower()

    return convictions_df[
        [
            "Employee Level",
            "Name",
            "Title",
            "Pleaded Guilty",
            "Convicted",
            "Sentence",
            "Status",
            "Charges",
            "First Name",
            "Last Name",
            "Email",
        ]
    ]


@click.command()
@click.option(
    "--url",
    default="https://archive.nytimes.com/www.nytimes.com/packages/html/national/20061023_ENRON_TABLE/index.html",
    help="URL of the webpage to scrape.",
)
@click.option("--output", default="convictions.csv", help="Output CSV file name.")
def main(url, output):
    """
    Main function to scrape a table from a webpage and save it as a CSV.

    Args:
    url (str): URL of the webpage to scrape.
    output (str): Name of the output CSV file.
    """
    soup = fetch_and_parse(url)
    convictions_df = extract_table(soup)
    convictions_df.to_csv(output, index=False)
    logging.info(f"Data extracted and saved to {output}")


if __name__ == "__main__":
    main()
