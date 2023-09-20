import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yelpapi import YelpAPI
from collections import Counter
import string
import pandas as pd


api_key = "hy8_mc7NcYkgqLW2pfIeJMN3BO51gZCsNvBKbJfMfo3qWlVFet1y5L1C6eH1Gv4lOaYwsdPHiCx5hTShxqMBwKA4WT-XK4KbUcaZ3PhPTN84QNcwGXBURWLIwb82ZHYx"
yelp_api = YelpAPI(api_key)


# Search Query
searh_term = "ramen"
search_location = "New York, NY"
search_by_rating = "rating"
search_limit = 50

search_results = yelp_api.search_query(term=searh_term, location=search_location, 
                                       sort_by=search_by_rating, limit=search_limit, offset=40)

## Create DataFrame ##

results_df = pd.DataFrame.from_dict(search_results['businesses'])
print(results_df)

## Alias's ##
'''
0                         izakaya-ajimi-brooklyn
1          kyuramen-columbia-university-new-york
2                              dokodemo-new-york
3                 kyuramen-union-square-new-york
4                               okonomi-brooklyn
5     kyuramen-long-island-city-long-island-city
6                          pasta-ramen-montclair
7                         tenho-ramen-new-york-4
8                           tamashii-gold-queens
9           shinka-ramen-and-sake-bar-new-york-3
10                 kailani-shave-ice-new-milford
11                        cha-menya-douglaston-2
12                     maison-kintaro-new-york-2
13                        toribro-ramen-new-york
14             kinya-ramen-sushi-bar-west-orange
15                   fukurou-brooklyn-new-york-2
16                             ni-ramen-lynbrook
17                        beat-mazesoba-queens-2
18                    kohoku-ku-ramen-new-york-2
19                  kiko-ramen-garden-city-south
20         bedford-gardens-restaurant-brooklyn-2
21                        fuyu-ramen-sunnyside-2
22                     shinjuku-ramen-new-york-2
23              7-doors-down-ramen-co-bloomfield
24                       teppen-ramen-new-york-2
25                                 rokc-new-york
26                  menya-and-izakaya-brooklyn-2
27             blackbeard-ramen-cliffside-park-2
28                      8-ramen-rockville-centre
29                      hakata-tonton-new-york-7
30                      jeju-noodle-bar-new-york
31                  domodomo-new-york-new-york-4
32                              wanpaku-brooklyn
33                       menya-sandaime-fort-lee
34                          ramen-danbo-brooklyn
35                             tabetomo-new-york
36                    kyuramen-flushing-flushing
37                              tonchin-brooklyn
38                          nippon-cha-bayside-2
39                          shaku-ramen-flushing
40                      ippudo-westside-new-york
41                         senshi-ramen-new-york
42           cocoron-and-goemon-curry-new-york-2
43                     sakai-ramen-staten-island
44        kogane-ramen-brooklyn-heights-brooklyn
45                      ichiran-midtown-new-york
46                            karakatta-new-york
47                         yuji-ramen-brooklyn-3
48                          ippudo-ny-new-york-7
49                    nori-izakaya-staten-island
'''
## Getting reviews from 5 businesses ##
length = int(input("\nEnter the number of iterations: "))
review_lists = [[] for _ in range(length)]

## Adding vader sentiment and stopwords##
analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

## Identifying business wanting to analyze and perferming Sentiment, stop words, Tokenizing and POS ##
for i in range(length):
    id_for_review = input("\nEnter the business alias: ").strip()

    try:
        reviews_results = yelp_api.reviews_query(id=id_for_review)
        review_df = pd.DataFrame.from_dict(reviews_results['reviews'])

        review_texts = review_df["text"].tolist()
        for text in review_texts:
            tokens = nltk.word_tokenize(text)
            tokens = [token for token in tokens if token not in string.punctuation]
            tokens_without_stopwords = [token for token in tokens if token.lower() not in stop_words]
            pos_tags = pos_tag(tokens_without_stopwords)
            review_lists[i].extend(pos_tags)

            sentiment = analyzer.polarity_scores(text)
            print(f"Restaurant: {id_for_review}")
            print(f"Review Text: {text}")
            print("\n")
            print(f"Tokens: {tokens}")
            print(f"Tokens without Stop Words: {tokens_without_stopwords}")
            print(f"POS Tags: {pos_tags}")
            print("\n")

            print(f"Sentiment - Positive: {sentiment['pos']}, Negative: {sentiment['neg']}, Neutral: {sentiment['neu']}, Compound: {sentiment['compound']}")
            print()

    except Exception as e:
        print("Error:", e)
        print("Skipping to next iteration")
        continue

## Creating lists with Token and tag ##
for i, review_list in enumerate(review_lists, start=1):
    print(f"\nReview List {i}:", review_list)
    
    all_tokens = [token for token, tag in pos_tags]
    word_counts = Counter(all_tokens)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_words[:5]

    print("Top words and their occurrences")
    for word, count in top_words:
        print(f"{word}: {count} occurences")
        