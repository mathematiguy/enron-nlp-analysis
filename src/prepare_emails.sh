#! /bin/bash

echo "Downloading enron email dataset"
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz -O data/enron_mail_20150507.tar.gz
echo "Unzipping..."
tar -xf data/enron_mail_20150507.tar.gz -C data/
echo "Converting raw emails to parquet"
python code/convert_to_parquet.py --input_dir data/maildir --output data/enron_emails.parquet
echo "Deleting raw files"
rm -rf data/enron_mail_20150507.tar.gz data/maildir
