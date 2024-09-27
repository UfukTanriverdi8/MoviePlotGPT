import pandas as pd

# Read the CSV file
df = pd.read_csv('dataset/the office/the-office-lines.csv')

# Filter out the rows where 'deleted' is False
filtered_df = df[df["deleted"] == False]

# Open a new text file in write mode
with open('office_lines.txt', 'w', encoding='utf-8') as file:
    # Iterate through the filtered DataFrame
    for index, row in filtered_df.iterrows():
        # Write the line text and speaker to the text file
        file.write(f"{row['speaker']}: {row['line_text']}\n")