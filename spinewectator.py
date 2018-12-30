import pandas as pd
import re
import tweepy
import yaml

PUNCTUATION = ['.', '!', '?']

def split_pairs(x):
    """Split a list into a list of 2-tuples."""
    output = []
    for i in range(0, len(x)):
        if i == 0:
            output.append(tuple(['_START', x[i]]))
        output.append(tuple(x[i:i+2]))
    return output

def fetch_wines():
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
    return wines

def fetch_titles():
    print('Reading some reviews...')
    reviews = pd.read_csv('data/winemag-data-130k-v2.csv')
    titles = reviews[['title']]

    # Clean titles - remove everything in parentheses
    titles = titles.assign(clean_title=titles.title.str.replace('\(.*\)', ''))
    titles = titles.assign(clean_title=titles.clean_title.str.strip())
    titles = titles.assign(words=titles.clean_title.str.split(pat=' '))
    return titles


def build_model(df):
    """Takes a DataFrame with a 'words' column"""
    print('Rebuilding the model.')
    print('Unnesting data...')
    df = df.assign(
        pairs=df.apply(lambda row: split_pairs(row.words), axis=1)
    )
    words = df['pairs'].apply(pd.Series).stack().to_frame(name='pairs')

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
    wines = fetch_wines()
    wordcounts = build_model(wines)
    wordcounts.to_csv('data/wordcounts.csv', index=False)


def update_title_model():
    titles = fetch_titles()
    wordcounts = build_model(titles)
    wordcounts.to_csv('data/title_wordcounts.csv', index=False)


def read_model():
    wordcounts = pd.read_csv('data/wordcounts.csv')
    return wordcounts


def read_title_model():
    wordcounts = pd.read_csv('data/title_wordcounts.csv')
    return wordcounts


def write_review():
    wordcounts = read_model()
    punctuation = ['.', '!', '?']
    prev_word = '_START'
    review = []
    keep_speaking = True

    while(keep_speaking):
        prev_word = wordcounts[wordcounts.word_1 == prev_word] \
            .sample(1, weights=wordcounts.weight) \
            .iloc[0, 1]
        review.append(prev_word)

        # if it's the end of a sentence, start over
        if any([prev_word.endswith(x) for x in punctuation]):
            prev_word = '_START'

        # if we've generated enough text, take a break
        # pattern = '|'.join(punctuation).replace('.', '\.').replace('?', '\?')
        # num_sentences = len(re.findall(pattern, review))
        num_sentences = sum([w.endswith(p) for w in review for p in punctuation])
        if len(review) >= 40 and num_sentences >= 2 and prev_word == '_START':
            keep_speaking = False

    return ' '.join(review)


def write_title():
    wordcounts = read_title_model()
    prev_word = '_START'
    title = []

    for i in range(6):
        prev_word = wordcounts[wordcounts.word_1 == prev_word] \
            .sample(1, weights=wordcounts.weight) \
            .iloc[0, 1]
        title.append(prev_word)

    # Cut the title short if it contains 2 dates
    date_locs = [re.match('^\d+$', t) is not None for t in title]
    if date_locs.count(True) > 1:
        date_locs.reverse()
        index = date_locs.index(True)
        title = title[:-1 * (index + 1)]
    return ' '.join(title)



def tweet(cred_path='secret.yml'):
    # Read credentials file
    print('Authorizing...')
    with open(cred_path, 'r') as stream:
        try:
            creds = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    auth = tweepy.OAuthHandler(creds['consumer_key'], creds['consumer_secret'])
    auth.set_access_token(creds['access_token'], creds['access_token_secret'])

    api = tweepy.API(auth)

    print('Writing review...')
    title = write_title()
    review = write_review()

    tweet = '{0}:\n{1}'.format(title, review)
    # If the tweet is too long, crop it
    while len(tweet) > 280:
        print('Tweet is too long, cropping...')
        sentences = tweet.split('. ')
        sentences.pop()
        tweet = '. '.join(sentences) + '.'
    print('Tweeting: ')
    print(tweet)
    response = api.update_status(tweet)

    return response
