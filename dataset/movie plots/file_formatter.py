import pandas as pd
from unidecode import unidecode

# Function to clean and normalize text
def clean_text(text):
    text = text.replace('\n\n', '\n').strip()
    return unidecode(text)

data = pd.read_csv('dataset/movie plots/wiki_movie_plots_deduped.csv')
all_countries = data["Origin/Ethnicity"].unique()
print(all_countries)
# english_countries = ["American", "British", "Australian", "Canadian"]
# data = data[data["Origin/Ethnicity"].isin(english_countries)]
print(len(data))

with open('dataset/movie plots/all_plots.txt', 'w', encoding="utf-8") as f:
    for index, row in data.iterrows():
        f.write(f"The plot summary of the movie named '{clean_text(row["Title"])}' is:" + "\n")
        f.write(clean_text(row["Plot"]) + "\nEND_OF_PLOT\n")  
print("Done")
