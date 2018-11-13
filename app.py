import pandas as pd
import re
from flask import Flask, jsonify
app = Flask(__name__)

def split_pairs(x):
    """Split a list into a list of 2-tuples."""
    output = []
    for i in range(0, len(x)):
        if i == 0:
            output.append(tuple(['_START', x[i]]))
        output.append(tuple(x[i:i+2]))
    return output


def build_model():
    print('Rebuilding the model.')
    print('Reading some reviews...')
    reviews = pd.read_csv('data/winemag-data-130k-v2.csv')
    wines = reviews[['title','description']]

    print('Cleaning data...')
    # Clean descriptions - remove certain punctuation
    wines = wines.assign(
        clean_desc=wines.description.str.replace('[^ ,;\.\?\-—!…\w\d]', '')
    )
    # Clean descriptions - remove multiple white spaces
    wines = wines.assign(clean_desc=wines.clean_desc.str.replace('\s', ' '))
    # Clean descriptions - remove multiple punctuation
    wines = wines.assign(clean_desc=wines.clean_desc.str.replace('\.{3}', '…'))
    # TODO: Clean descriptions - remove white spaces around commas and other punctuation

    wines = wines.assign(words=wines.clean_desc.str.split(pat=' '))

    print('Unnesting data...')
    # Unnest pairs of words
    wines = wines.assign(
        pairs=wines.apply(lambda row: split_pairs(row.words), axis=1)
    )
    words = wines['pairs'].apply(pd.Series).stack().to_frame(name='pairs')

    punctuation = ['.', '?', '!']
    words = words.assign(
        word_1=words['pairs'].apply(
            lambda x: '_START' if any([x[0].endswith(p) for p in punctuation]) else x[0]
        ),
        word_2=words['pairs'].apply(lambda x: x[1] if len(x) > 1 else '_END')
    )

    words = words.reset_index()
    words = words[['pairs', 'word_1', 'word_2']]

    # Remove any empty words, filter out pairs that occur infrequently to make it faster
    words = words[
        (words.word_1 != '') &
        (words.word_2 != '') &
        (words.word_2 != '_END')
    ]
    print('Counting frequencies & calculating weights...')
    # Count frequencies
    w1 = words.groupby('word_1').agg('size').reset_index(name='w1_freq')
    w2 = words.groupby(['word_1', 'word_2']).agg('size').reset_index(name='n')
    wordcounts = w2.merge(w1, on='word_1', how='inner')

    # Calculate weights
    wordcounts = wordcounts.assign(weight=wordcounts.n/wordcounts.w1_freq)

    return wordcounts


def update_model():
    wordcounts = build_model()
    wordcounts.to_csv('data/wordcounts.csv', index=False)

def read_model():
    wordcounts = pd.read_csv('data/wordcounts.csv')
    return wordcounts


@app.route('/')
def home():
    wordcounts = read_model()

    punctuation = ['.', '!', '?']
    reviews = []

    # Generate 10 reviews
    for i in range(10):
        prev_word = '_START'
        review = ''
        keep_speaking = True

        while(keep_speaking):
            prev_word = wordcounts[wordcounts.word_1 == prev_word] \
                .sample(1, weights=wordcounts.weight) \
                .iloc[0, 1]
            review += ' ' + prev_word

            # if it's the end of a sentence, start over
            if any([prev_word.endswith(x) for x in punctuation]):
                prev_word = '_START'

            # if we've generated enough text, take a break
            pattern = '|'.join(punctuation).replace('.', '\.').replace('?', '\?')
            num_sentences = len(re.findall(pattern, review))
            if len(review) >= 40 and num_sentences >= 2 and prev_word == '_START':
                keep_speaking = False

        reviews.append(review)

    return jsonify({
        'reviews': reviews
    })
