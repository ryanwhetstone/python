import tweepy
from textblob import TextBlob
import pandas as pd
from IPython.display import display

# Step 1 - Authenticate
consumer_key= 'dNltCHVvGCMe1tbOsd0uVyDzM'
consumer_secret= '6DDT94jq42cgokvXLYuvjOvUXtFLLG9OzDCPr40TmY5XkXy6C1'

access_token='2826897272-M9jhGCfKo3zjvZaya2XSnYzU9cLmbkHMrPJZMI1'
access_token_secret='E7PxcT8jriMkHBwE92kz5uzYYtMZQ84Yf3UnFd6EWjrH1'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 3 - Retrieve Tweets
public_tweets = api.search('$amzn')
# public_tweets = api.user_timeline(screen_name = 'realDonaldTrump', count = 200, include_rts = False)

polarities = []
tweets = []
analysis_results = []

for tweet in public_tweets:
    print("")
    # print(tweet.text)
    print("")
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    polarity = analysis.polarity
    polarities.append(polarity)
    # print(polarity)
    print("")
    if polarity > .15:
        # print("positive")
        sentiment = "positive"
    elif polarity < -.1:
        # print("negative")
        sentiment = "negative"
    else:
        # print("neutral")
        sentiment = "neutral"
    print("")
    analysis_results.append((tweet.text, polarity, sentiment))
    # analysis_results.append(analysis_result)


ap = average_polarity = sum(polarities) / len(polarities)
if(ap <= -.15):
    polarity_description = "negative"
elif(ap > -.15 and ap <= 0):
    polarity_description = "fairly negative"
elif(ap > 0 and ap <= 0.15):
    polarity_description = "fairly positive"
else:
    polarity_description = "positive"
print("The average polarity (sentiment) is " +  str(round(average_polarity,2)) + ", which is " + polarity_description)




df = pd.DataFrame(analysis_results, columns= ['Tweet', 'Polarity', 'Sentiment'])
# df.to_csv('/Users/ryanwhetstone/GoogleDrive/Python/test.csv',index=False)

display(df.head(10))