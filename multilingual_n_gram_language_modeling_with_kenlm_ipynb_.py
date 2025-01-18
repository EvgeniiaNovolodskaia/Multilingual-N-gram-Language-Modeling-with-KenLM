
import tensorflow_datasets as tfds
from itertools import islice
import random
from polyglot.text import Text
import re
import matplotlib.pyplot as plt
import seaborn as sns
import kenlm.kenlm as kenlm
from tqdm import tqdm

ds = tfds.load("wiki40b/fr", split="test", data_dir="gs://tfds-data/datasets")
sample = [ex["text"].numpy().decode("utf-8")
for ex in islice(ds.shuffle(buffer_size=10_000), 100)]

def get_sentences(language, nb_sentences) -> list[Text]:
  """
  Extract a list of unique sentences.
  Arguments:
    - language: language code;
    - nb_sentences: number of sentences to extract (train + test).
  Returns:
    - a list of tokenized unique sentences.
  """
  sentences = set()
  print(f"Processing and tokenizing {language}")

  # Load and shuffer the dataset for some language
  ds = tfds.load(f"wiki40b/{language}", split="test", data_dir="gs://tfds-data/datasets")
  ds = [ex for ex in islice(ds.shuffle(buffer_size=10_000), nb_sentences)]

  for article in ds:
    text = article["text"].numpy().decode("utf-8")
    # Remove unwanted markers
    for markers in ["_START_ARTICLE_\n", "_START_SECTION_\n", "_START_PARAGRAPH_\n", "\n", "_NEWLINE_"]:
      text = text.replace(markers, " ")
    # Use Polyglot to split the text into sentences and detect language
    segmented_text = Text(text, hint_language_code=language)
    # Extract sentences and tokenize them, store them in the set
    for sentence in segmented_text.sentences:
      sentences.add(" ".join(sentence.words))

  return list(sentences)

def clean_text(sentence):
  """
  Remove punctuation and all numbers from the sentence
  """
  # Punctuation symbols surrounded by spaces or with space on one side to be removed
  punctuations = r'\s([?.!,;:"\-()\/|\'“”"\[\]،–„’«»।。、%，・])\s?'

  # Remove punctuation
  cleaned_sentence = re.sub(punctuations, ' ', sentence)

  # Remove all numbers
  cleaned_sentence = re.sub(r'\d+', '', cleaned_sentence)

  # Remove extra spaces
  cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()

  return cleaned_sentence

def save_to_file(sentences, language, file_name):
  """
  Save tokenized sentences to a file
  Arguments:
    - sentences: list of tokenized sentences,
    - language: language code,
    - file_name: train or test.
  """
  with open(f'/content/{language}_{file_name}.txt', 'w') as file:
    for sentence in sentences:
      # Remove punctuation and numbers
      cleaned_sentence = clean_text(sentence)

      # Write the cleaned sentence to the file
      file.write(cleaned_sentence + "\n")

l_codes = ["ar", "bg", "ca", "cs", "da", "de", "el", "en", "es", "et", "fa",
          "fi", "fr", "he", "hi", "hr", "hu", "id", "it", "ja", "ko", "lt",
          "lv", "ms", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sr",
          "sv", "th", "tl", "tr", "uk", "vi", "zh-cn", "zh-tw"]

for l in l_codes:
  sentences = get_sentences(l, nb_sentences=43_000)

  train_set = sentences[:43_000]
  test_set = sentences[40_000:43_000]

  save_to_file(train_set, l, "train")
  save_to_file(test_set, l, "test")

def read_sentences(language, file_name):
  with open(f'/content/{language}_{file_name}.txt', 'r') as f:
    sentences = f.readlines()
  return [sentence.strip() for sentence in sentences]

ttr_results = []
def compute_ttr(language):
  """
  Compute TTR
  Arguments:
    - language: language code
  Returns:
    - TTR values
  """
  # Read train and test sentences
  train_sentences = read_sentences(language, "train")
  test_sentences = read_sentences(language, "test")
  all_sentences = train_sentences + test_sentences

  tokens = []
  for sentence in all_sentences:
    tokens.extend(sentence.split())  # Split each sentence into tokens (words)

  types = set(tokens)  # Unique words (types)
  ttr = len(types) / len(tokens)

  ttr_results.append({'Language': language, 'TTR': ttr})

# Compute_ttr for each language
for l in l_codes:
  compute_ttr(l)

import pandas as pd

# Create a dataframe
df = pd.DataFrame(ttr_results)

# Optionally, save the DataFrame to a CSV file
df.to_csv('/content/language_ttr_results.csv', index=False)

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# List of language codes
l_codes = ["ar", "bg", "ca", "cs", "da", "de", "el", "en", "es", "et", "fa",
           "fi", "fr", "he", "hi", "hr", "hu", "id", "it", "ja", "ko", "lt",
           "lv", "ms", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sr",
           "sv", "th", "tl", "tr", "uk", "vi", "zh-cn", "zh-tw"]

# Create a directory in Google Drive to store the files
drive_dir = '/content/drive/MyDrive/wiki40b_language_files/'
os.makedirs(drive_dir, exist_ok=True)

# Copy train and test files for each language to Google Drive
for lang in l_codes:
    train_file = f'/content/{lang}_train.txt'
    test_file = f'/content/{lang}_test.txt'

    # Check if the files exist before copying
    if os.path.exists(train_file) and os.path.exists(test_file):
        # Copy the files to Google Drive
        !cp {train_file} {drive_dir}
        !cp {test_file} {drive_dir}
    else:
        print(f"Files for {lang} not found.")

print("All files have been copied to Google Drive.")

l_typology = {
    "isolating": ["id", "ms", "th", "tl", "vi", "zh-cn", "zh-tw"],
    "fusional": ["bg", "ca", "cs", "da", "de", "el", "en", "es", "fa",
        "fr", "hi", "hr", "it", "lt", "lv", "nl", "no", "pl",
        "pt", "ro", "ru", "sk", "sl", "sr", "sv", "uk"],
    "introflexive": ["ar", "he"],
    "agglutinative": ["et", "fi", "hu", "ja", "ko", "tr"]
}

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/kpu/kenlm.git
# %cd kenlm
!python setup.py develop
!mkdir -p build
# %cd build
!cmake ..
!make -j 4

import os
os.makedirs("/content/models", exist_ok=True)

for l in l_codes:
  print(f"Training {l}")
  os.system(f"/content/kenlm/build/bin/lmplz -o 5 </content/{l}_train.txt >/content/models/{l}_model.arpa")

results = []

for l in tqdm(l_codes):
  print(f"Estimating perplexity for {l}")

  # Load the trained model for each language
  m = kenlm.Model(f'/content/models/{l}_model.arpa')

  # Read the sentences from test set
  sentences = read_sentences(l, 'test')

  # Skip if no sentences are found
  if not sentences:
    print(f"No sentences found for {l}, skipping.")
    continue

  # Calculate total perplexity for the sentences
  total_perplexity = 0
  sentence_count = 0

  for sentence in sentences:
    # Calculate perplexity for each sentence
    perplexity = m.perplexity(sentence)
    total_perplexity += perplexity
    sentence_count += 1

  # Store the average perplexity for each language
  if sentence_count > 0:
    avg_perplexity = total_perplexity / sentence_count
    results.append({'Language': l, 'Perplexity': avg_perplexity})
    print(f"Perplexity for {l}: {avg_perplexity}")
  else:
    print(f"No valid sentences for perplexity calculation for {l}.")

df_perplexity = pd.DataFrame(results)

# Merging the dataframes with TTR and with perplexity
df_merged = pd.merge(df, df_perplexity, on='Language')

# Map language to typology
def get_typology(language):
  for typology, langs in l_typology.items():
    if language in langs:
      return typology
  return 'unknown'

df_merged['Typology'] = df_merged['Language'].apply(get_typology)

typology_colors = {
    "isolating": "blue",
    "fusional": "orange",
    "agglutinative": "green",
    "introflexive": "red"
}

plt.figure(figsize=(9, 5))

sns.scatterplot(data=df_merged,
                x='TTR',
                y='Perplexity',
                hue='Typology',
                palette=typology_colors,
                marker='o',
                s=100)

# Annotations for each language
for i, row in df_merged.iterrows():
  plt.text(row['TTR'] + 0.002, row['Perplexity'], row['Language'], fontsize=10)

plt.xlabel('TTR')
plt.ylabel('Perplexity')
plt.title('Perplexity in relation to the TTR')

plt.show()