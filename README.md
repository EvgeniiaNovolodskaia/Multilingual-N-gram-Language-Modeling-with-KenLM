# Multilingual-N-gram-Language-Modeling-with-KenLM
This project explores multilingual text processing and analysis using **KenLM**, an efficient library for building and evaluating n-gram language models. The main goal is to evaluate the relationship between linguistic typology, morphological complexity, and the performance of language models.

## **Features**
1. **Data Preparation:**
   - Text data is sourced from the **Wiki40B** corpus using `tensorflow_datasets`.
   - Sentences are tokenized and cleaned using `polyglot` and `re`.

2. **Language Model Training:**
   - **KenLM** is used to train n-gram language models for multiple languages.
   - The models are stored in `.arpa` format for further evaluation.

3. **Evaluation of Models:**
   - The trained models are evaluated on test data by calculating **perplexity** for each language.
   - The relationship between **Perplexity** and **Type-Token Ratio (TTR)** is analyzed.

4. **Visualization:**
   - Results are visualized using **Matplotlib** and **Seaborn** to show the correlation between TTR, Perplexity, and linguistic typology.
  

### Installing KenLM

KenLM must be installed from source. Follow these steps:

```bash
git clone https://github.com/kpu/kenlm.git
cd kenlm
python setup.py develop
mkdir -p build
cd build
cmake ..
make -j4
