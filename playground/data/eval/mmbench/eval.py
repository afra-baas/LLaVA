import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--upload-dir', default = "", type = str, help = "Excel file with answers and predictions")
args = parser.parse_args()

df = pd.read_excel(args.upload_dir)

# Check if df['prediction'] is longer than one letter
df['prediction'] = df['prediction'].apply(lambda x: x[0] if len(x) > 1 else x)
# df['prediction'] = df['prediction'].str.upper()

df['correct'] = (df['prediction'] == df['answer'])
accuracy = df['correct'].mean() * 100
print(f"Accuracy: {accuracy:.2f}%")



