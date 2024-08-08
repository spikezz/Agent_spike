import os
import pandas as pd
import csv
import click

pd.set_option('display.max_rows', None)

# Function to clean and properly format the string fields
def clean_quotes(value):
    if isinstance(value, str):
        # Remove extra quotes and strip leading/trailing spaces
        value = value.strip().replace('""', '"')
    return value

def convert_parquet_to_csv_func(parquet_dir, csv_dir):
    """Convert all Parquet files in a directory to CSV format."""
    for file_name in os.listdir(parquet_dir):
        if file_name.endswith('.parquet'):
            parquet_file = os.path.join(parquet_dir, file_name)
            csv_file = os.path.join(csv_dir, file_name.replace('.parquet', '.csv'))
            # Load the Parquet file
            df = pd.read_parquet(parquet_file)
            # Clean quotes in string fields
            for column in df.select_dtypes(include=['object']).columns:
                df[column] = df[column].apply(clean_quotes)
            # Save to CSV
            new_csv_file=""
            df.to_csv(csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
            if file_name=="create_summarized_entities.parquet" or \
                file_name=="create_base_entity_graph.parquet" or \
                file_name=="create_base_extracted_entities.parquet":
                with open(csv_file, 'r') as f:
                    for line in f:
                        new_csv_file+=line.replace('""','"')
                with open(csv_file, 'w') as f:
                    f.write(new_csv_file)
            else:
                with open(csv_file, 'r') as f:
                    for line in f:
                        new_csv_file+=line.replace('""\\','').replace('\\""','""')
                with open(csv_file, 'w') as f:
                    f.write(new_csv_file)
            print(f"Converted {parquet_file} to {csv_file} successfully.")
    print("All Parquet files have been converted to CSV.")

@click.command()
@click.option('--parquet_dir', help='Directory containing Parquet files.')
@click.option('--csv_dir', help='Directory to save CSV files.')
def convert_parquet_to_csv(parquet_dir, csv_dir):
    """Convert all Parquet files in a directory to CSV format."""
    for file_name in os.listdir(parquet_dir):
        if file_name.endswith('.parquet'):
            parquet_file = os.path.join(parquet_dir, file_name)
            csv_file = os.path.join(csv_dir, file_name.replace('.parquet', '.csv'))
            # Load the Parquet file
            df = pd.read_parquet(parquet_file)
            # Clean quotes in string fields
            for column in df.select_dtypes(include=['object']).columns:
                df[column] = df[column].apply(clean_quotes)
            # Save to CSV
            new_csv_file=""
            df.to_csv(csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
            if file_name=="create_summarized_entities.parquet" or \
                file_name=="create_base_entity_graph.parquet" or \
                file_name=="create_base_extracted_entities.parquet":
                with open(csv_file, 'r') as f:
                    for line in f:
                        new_csv_file+=line.replace('""','"')
                with open(csv_file, 'w') as f:
                    f.write(new_csv_file)
            else:
                with open(csv_file, 'r') as f:
                    for line in f:
                        new_csv_file+=line.replace('""\\','').replace('\\""','""')
                with open(csv_file, 'w') as f:
                    f.write(new_csv_file)
            print(f"Converted {parquet_file} to {csv_file} successfully.")
    print("All Parquet files have been converted to CSV.")

if __name__ == '__main__':
    convert_parquet_to_csv()