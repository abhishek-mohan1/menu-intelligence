from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def score_sentiment(df):
    df['sentiment_score'] = df['processed_text'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )

    def label(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df['sentiment_score'].apply(label)

    return df