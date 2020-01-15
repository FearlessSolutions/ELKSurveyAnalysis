
import itertools
from spacy.lang.en import English, stop_words
import pandas 
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt 

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from pprint import pprint

nlp = English()

data_path = "/data/cleaned data - At-Risk Population Surveys.csv"

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

    #print(my_tokens)
    return my_tokens


def generate_word_cloud(final_tokens):

    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(final_tokens) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.savefig("/results/word_cloud.png")


def parse_lda_results(lda_model):
    final = []
    lda_results = lda_model.show_topics()[0][1]

    split_results = lda_results.split("+")

    for result in split_results:
        entry = result.split('*"')[1].replace('"', " ").strip()
        final.append(entry)
     

    return final

if __name__ == "__main__":

    df = pandas.read_csv(data_path)

    results = calc_age(df)

    results["do you have dermatologist"] = calc_affirmative_percentage(df, "Do you have a dermatologist?", "yes")
    #print(results["do you have dermatologist"])

    # token_df = df["How do you currently keep track of skin changes?"].apply(get_tokens) 
    token_df = df["Where would you go to learn about Melanoma?"].apply(get_tokens) 


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

#    pprint(lda_model.print_topics(num_words=10))
    lda_results = parse_lda_results(lda_model)

    final_results = []
    for token in joined_tokens:
        if token in lda_results:
            final_results.append(token)

    generate_word_cloud(" ".join(final_results))
