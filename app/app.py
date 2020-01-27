
import itertools
import spacy
from spacy.lang.en import English, stop_words
import pandas 
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt 

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#from pprint import pprint
import pprint
import random

#nlp = English()

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
#data_path = "/data/cleaned data - At-Risk Population Surveys.csv"
data_path = "/data/How to Engineer Impact with Data.csv"

def calc_age(df):

    valid_data = df[df["age"] > 0]
    results = {}

    ages = valid_data['age']
    results["mean"] = ages.mean()
    results["median"] = ages.median()
    results["min"] = ages.min()
    results["max"] = ages.max()
    
    return results

def calc_affirmative_percentage(df, column_name, affirmative_token):

    info = df[df[column_name] == affirmative_token]
    
    affirmative_count = info[column_name].count()

    percentage = affirmative_count / df[column_name].count()

    return percentage

def get_tokens(text):

    if type(text) != str:
        return " "

    text = text.replace("`", " ")
    my_tokens= nlp(text)


    # lemmatization (find the root words where necessary)
    my_tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in my_tokens ]

    punctuations = string.punctuation
    # Removing stop words and punctuations
    my_tokens = [ word for word in my_tokens if word not in stop_words.STOP_WORDS and word not in punctuations and "u0" not in word and word != "t" ]

    if "self" in my_tokens and "check" in my_tokens:
        my_tokens.remove("self")
        my_tokens.remove("check")
        my_tokens.append("selfcheck")
    if "data" in my_tokens:
        print("data")
        my_tokens.remove("data")

    if "science" in my_tokens:
        print("science")
        my_tokens.remove("science")

    if "machine" in my_tokens and "learning" in my_tokens:
        my_tokens.remove("machine")
        my_tokens.remove("learning")
        my_tokens.append("machine learning")


    #print(my_tokens)
    return my_tokens


def generate_word_cloud(final_tokens):

    dummy_list = list()
    for key, value in final_tokens.items():
        dummy_list.extend([key] * value)

    random.shuffle(dummy_list)
    funk = ", ".join(dummy_list)
  
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(funk) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.savefig("/results/word_cloud.png")


def parse_lda_results(lda_model):
    '''

    '''
    final = {}
    lda_results = lda_model.show_topics()[0][1]

    split_results = lda_results.split("+")

    for result in split_results:
        weight,entry  = result.split('*"')#[1].replace('"', " ").strip()
        entry = entry.replace('"', " ").strip()
        weight = int(float(weight)* 1000)

        if weight < 1:
            weight = 1

        final[entry] = weight
      
     

    return final


def depreciated():
    df = pandas.read_csv(data_path)

    #results = calc_age(df)

  #  results["do you have dermatologist"] = calc_affirmative_percentage(df, "Do you have a dermatologist?", "yes")
    #print(results["do you have dermatologist"])

    # token_df = df["How do you currently keep track of skin changes?"].apply(get_tokens) 
   # token_df = df["Where would you go to learn about Melanoma?"].apply(get_tokens) 
    token_df = df["What does Data Science mean to you?"].apply(get_tokens) 

    joined_tokens = list(itertools.chain.from_iterable(token_df.tolist()))

    #final_tokens = " ".join(joined_tokens)

    words = corpora.Dictionary([joined_tokens])

   # print(joined_tokens)

    corpus = [words.doc2bow(doc) for doc in [joined_tokens]]


    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=words,
                                           num_topics=5, 
                                           random_state=2,
                                           update_every=1,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

    pprint(lda_model.print_topics(num_words=10))
    lda_results = parse_lda_results(lda_model)

    final_results = []
    for token in joined_tokens:
        if token in lda_results:
            final_results.append(token)

    generate_word_cloud(" ".join(final_results))


def create_bigram_trigram_models(data_words):
    '''
    data_words, list of lists, every element is an "article"
    '''

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=2, threshold=50) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

 

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return bigram_mod, trigram_mod

def tokenize(text):
    # Removing stop words and punctuations
    #my_tokens = [ word for word in my_tokens if word not in stop_words.STOP_WORDS and word not in punctuations and "u0" not in word and word != "t" ]

    tokens = gensim.utils.simple_preprocess(text, deacc=True)
 
    return tokens

def clean_text(text):
    no_stopwords = remove_stopwords(text)
    bigrams = make_bigrams(no_stopwords)

    lemmatized = lemmatization(bigrams)
    
    return lemmatized

def remove_stopwords(tokens):
    return [ word for word in tokens if word not in stop_words.STOP_WORDS and "u0" not in word and word != "t" and word != "andme" and word != "base" ]
    #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(tokens):
    return bigram_mod[tokens]

def make_trigrams(tokens):
    return trigram_mod[bigram_mod[tokens]]

def lemmatization(tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    # for sent in texts:
    doc = nlp(" ".join(tokens)) 
    
    return [token.lemma_ for token in doc if token.lemma_ != "datum"] 

def process_topics(all_data, articles):

    # Create Dictionary
    id2word = corpora.Dictionary(articles)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in articles]  

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    #pprint(lda_model.print_topics())
    return lda_model



def process_survey():
    df = pandas.read_csv(data_path)

    ds_meaning_df = df["What does Data Science mean to you?"].apply(tokenize) 

    ds_examples_df =  df["What is an example of Data Science?"].apply(tokenize) 

    all_data_corpus = ds_examples_df.tolist()
    all_data_corpus.extend(ds_meaning_df.tolist())

    bigram_mod, trigram_mod = create_bigram_trigram_models(all_data_corpus)

    # not the most efficient but easier to read
    cleaned_all_data_corpus = list()
    for article in all_data_corpus:
        cleaned_article = clean_text(article)
        cleaned_all_data_corpus.append(cleaned_article)

    # process ds examples 
    cleaned_ds_examples = ds_meaning_df.apply(clean_text)

    lda_model = process_topics(cleaned_all_data_corpus, cleaned_ds_examples.tolist())

    pprint(lda_model.print_topics())
    p_r = parse_lda_results(lda_model)

    #print(p_r)
    generate_word_cloud(p_r)
    # # Compute Perplexity
    # print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # # Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)

def process_pdf_text():
    data_path = "/results/parsed_pdf.csv"
    df = pandas.read_csv(data_path)
    print(df.columns)

    ds_text_df = df["text "].apply(tokenize) 

    all_data_corpus = ds_text_df.tolist() 

    bigram_mod, trigram_mod = create_bigram_trigram_models(all_data_corpus)

    # not the most efficient but easier to read
    cleaned_all_data_corpus = list()
    for article in all_data_corpus:
        cleaned_article = clean_text(article )
        cleaned_all_data_corpus.append(cleaned_article)

    # process ds examples 
    cleaned_ds_examples = ds_text_df.apply(clean_text)

    lda_model = process_topics(cleaned_all_data_corpus, cleaned_ds_examples.tolist())

    pprint(lda_model.print_topics())
    p_r = parse_lda_results(lda_model)

    #print(p_r)
    generate_word_cloud(p_r)    

if __name__ == "__main__":
    #process_survey()
    # process_pdf_text()
    data_path = "/results/parsed_pdf.csv"
    df = pandas.read_csv(data_path)
    print(df.columns)

    ds_text_df = df["text "].apply(tokenize) 

    all_data_corpus = ds_text_df.tolist() 

    bigram_mod, trigram_mod = create_bigram_trigram_models(all_data_corpus)

    # not the most efficient but easier to read
    cleaned_all_data_corpus = list()
    for article in all_data_corpus:
        cleaned_article = clean_text(article )
        cleaned_all_data_corpus.append(cleaned_article)

    # process ds examples 
    cleaned_ds_examples = ds_text_df.apply(clean_text)

    lda_model = process_topics(cleaned_all_data_corpus, cleaned_ds_examples.tolist())

    #pprint(lda_model.print_topics())
    with open("/results/pdf_topics.txt", "w") as fh:
        
        fh.write(pprint.pformat(lda_model.print_topics()))
    p_r = parse_lda_results(lda_model)

    #print(p_r)
    generate_word_cloud(p_r)    