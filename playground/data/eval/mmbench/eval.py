import pandas as pd
import argparse
import os
import sys 

parser = argparse.ArgumentParser()
parser.add_argument('--upload-dir', default = "", type = str, help = "Excel file with answers and predictions")
parser.add_argument('--output-file', default = "", type = str)
args = parser.parse_args()

df = pd.read_excel(args.upload_dir)

# Check if df['prediction'] is longer than one letter
df['prediction'] = df['prediction'].apply(lambda x: x[0] if len(x) > 1 else x)
# df['prediction'] = df['prediction'].str.upper()

df['correct'] = (df['prediction'] == df['answer'])
accuracy = df['correct'].mean() * 100

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(f"{args.output_file}", "w") as f:
    # Redirect the standard output to the file
    original_stdout = sys.stdout
    sys.stdout = f
    print(f"Accuracy: {accuracy:.2f}%")
sys.stdout = sys.__stdout__

