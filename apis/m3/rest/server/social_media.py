from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/social_media/social_media.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of tweets in a specific language
@app.get("/v1/social_media/tweet_count_by_language", operation_id="get_tweet_count_by_language", summary="Retrieves the total number of tweets in a specified language from the social media database.")
async def get_tweet_count_by_language(lang: str = Query(..., description="Language of the tweet")):
    cursor.execute("SELECT COUNT(TweetID) AS tweet_number FROM twitter WHERE Lang = ?", (lang,))
    result = cursor.fetchone()
    if not result:
        return {"tweet_number": []}
    return {"tweet_number": result[0]}

# Endpoint to get the text of reshared tweets
@app.get("/v1/social_media/reshared_tweets_text", operation_id="get_reshared_tweets_text", summary="Retrieves the text content of tweets that have been reshared. The operation filters tweets based on a specified reshare status, allowing users to view the text of either reshared or non-reshared tweets.")
async def get_reshared_tweets_text(is_reshare: str = Query(..., description="Whether the tweet is a reshare (TRUE or FALSE)")):
    cursor.execute("SELECT text FROM twitter WHERE IsReshare = ?", (is_reshare,))
    result = cursor.fetchall()
    if not result:
        return {"text": []}
    return {"text": [row[0] for row in result]}

# Endpoint to get the count of tweets with reach greater than a specified number
@app.get("/v1/social_media/tweet_count_by_reach", operation_id="get_tweet_count_by_reach", summary="Retrieves the total number of tweets that have a reach greater than the specified minimum reach.")
async def get_tweet_count_by_reach(reach: int = Query(..., description="Minimum reach of the tweet")):
    cursor.execute("SELECT COUNT(TweetID) AS tweet_number FROM twitter WHERE Reach > ?", (reach,))
    result = cursor.fetchone()
    if not result:
        return {"tweet_number": []}
    return {"tweet_number": result[0]}

# Endpoint to get the count of tweets with sentiment greater than a specified value on a specific weekday
@app.get("/v1/social_media/tweet_count_by_sentiment_weekday", operation_id="get_tweet_count_by_sentiment_weekday", summary="Retrieves the total number of tweets that have a sentiment score exceeding a specified value, filtered by a particular weekday. This operation is useful for analyzing the volume of tweets with a certain sentiment level on a specific day of the week.")
async def get_tweet_count_by_sentiment_weekday(sentiment: int = Query(..., description="Minimum sentiment value of the tweet"), weekday: str = Query(..., description="Weekday of the tweet (e.g., 'Thursday')")):
    cursor.execute("SELECT COUNT(TweetID) AS tweet_number FROM twitter WHERE Sentiment > ? AND Weekday = ?", (sentiment, weekday))
    result = cursor.fetchone()
    if not result:
        return {"tweet_number": []}
    return {"tweet_number": result[0]}

# Endpoint to get the text of the tweet with the maximum likes
@app.get("/v1/social_media/tweet_text_with_max_likes", operation_id="get_tweet_text_with_max_likes", summary="Retrieves the text content of the tweet that has garnered the highest number of likes. This operation provides a quick way to identify the most popular tweet based on user engagement.")
async def get_tweet_text_with_max_likes():
    cursor.execute("SELECT text FROM twitter WHERE Likes = ( SELECT MAX( Likes) FROM twitter )")
    result = cursor.fetchone()
    if not result:
        return {"text": []}
    return {"text": result[0]}

# Endpoint to get the cities in a specific country
@app.get("/v1/social_media/cities_by_country", operation_id="get_cities_by_country", summary="Retrieves a list of cities within a specified country. The operation filters the location data to exclude null city values and returns only those cities that belong to the provided country.")
async def get_cities_by_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT City FROM location WHERE City IS NOT NULL AND Country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of tweets from a specific country
@app.get("/v1/social_media/tweet_count_by_country", operation_id="get_tweet_count_by_country", summary="Retrieves the total number of tweets originating from a specified country. The operation filters tweets based on the provided country name and returns the count of matching tweets.")
async def get_tweet_count_by_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.Country = ? LIMIT 1", (country,))
    result = cursor.fetchone()
    if not result:
        return {"tweet_count": []}
    return {"tweet_count": result[0]}

# Endpoint to get the city with the most tweets in a specific country
@app.get("/v1/social_media/top_city_by_tweets_in_country", operation_id="get_top_city_by_tweets_in_country", summary="Retrieves the city with the highest number of tweets in a specified country. The operation filters tweets based on the provided country name and aggregates them by city. The city with the most tweets is then returned.")
async def get_top_city_by_tweets_in_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT T2.City FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.Country = ? GROUP BY T2.City ORDER BY COUNT(T1.TweetID) DESC LIMIT 1", (country,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the count of reshared tweets in a specific city
@app.get("/v1/social_media/reshared_tweets_count_by_city", operation_id="get_reshared_tweets_count_by_city", summary="Retrieves the total number of reshared tweets originating from a specified city. The operation considers a tweet as a reshare based on the provided boolean value.")
async def get_reshared_tweets_count_by_city(city: str = Query(..., description="City name"), is_reshare: str = Query(..., description="Whether the tweet is a reshare (TRUE or FALSE)")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.City = ? AND T1.IsReshare = ?", (city, is_reshare))
    result = cursor.fetchone()
    if not result:
        return {"tweet_count": []}
    return {"tweet_count": result[0]}

# Endpoint to get the text of tweets with sentiment greater than a specified value in a specific city
@app.get("/v1/social_media/tweets_text_by_sentiment_city", operation_id="get_tweets_text_by_sentiment_city", summary="Retrieve the text of tweets with a sentiment score exceeding a specified value from a particular city. The sentiment and city parameters are used to filter the results.")
async def get_tweets_text_by_sentiment_city(sentiment: int = Query(..., description="Minimum sentiment value of the tweet"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.text FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T1.Sentiment > ? AND T2.City = ?", (sentiment, city))
    result = cursor.fetchall()
    if not result:
        return {"text": []}
    return {"text": [row[0] for row in result]}

# Endpoint to get the country with the highest number of likes on a tweet
@app.get("/v1/social_media/top_country_by_likes", operation_id="get_top_country_by_likes", summary="Retrieves the country with the highest number of likes on a tweet. This operation identifies the top country by analyzing the number of likes on tweets, considering the location of each tweet. The result is determined by ordering the countries based on the total likes and selecting the top one.")
async def get_top_country_by_likes():
    cursor.execute("SELECT T2.Country FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID ORDER BY T1.Likes DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the top country by tweet count with positive sentiment from specified countries
@app.get("/v1/social_media/top_country_by_tweet_count", operation_id="get_top_country_by_tweet_count", summary="Retrieves the country with the highest number of positive tweets from the provided list of countries. The sentiment of each tweet is compared against a specified minimum value to determine positivity. The result is limited to the top-ranked country.")
async def get_top_country_by_tweet_count(country1: str = Query(..., description="First country"), country2: str = Query(..., description="Second country"), min_sentiment: int = Query(..., description="Minimum sentiment value")):
    cursor.execute("SELECT T2.Country FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.Country IN (?, ?) AND T1.Sentiment > ? GROUP BY T2.Country ORDER BY COUNT(T1.TweetID) DESC LIMIT 1", (country1, country2, min_sentiment))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of tweets from a specific city on a specific weekday
@app.get("/v1/social_media/tweet_count_city_weekday", operation_id="get_tweet_count_city_weekday", summary="Retrieves the total number of tweets originating from a specified city on a given weekday. The operation requires the city and weekday as input parameters to filter the data accordingly.")
async def get_tweet_count_city_weekday(city: str = Query(..., description="City name"), weekday: str = Query(..., description="Weekday name")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.City = ? AND T1.Weekday = ?", (city, weekday))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of tweets with more than a specified number of likes from users of a specific gender
@app.get("/v1/social_media/tweet_count_likes_gender", operation_id="get_tweet_count_likes_gender", summary="Retrieves the total number of tweets that have surpassed a specified threshold of likes, originating from users of a particular gender. The response is based on a comparison of the number of likes against a provided minimum value and the gender of the user.")
async def get_tweet_count_likes_gender(min_likes: int = Query(..., description="Minimum number of likes"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T1.Likes > ? AND T2.Gender = ?", (min_likes, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of tweets from users of a specific gender
@app.get("/v1/social_media/tweet_count_gender", operation_id="get_tweet_count_gender", summary="Retrieves the total number of tweets posted by users of a specified gender. The gender parameter is used to filter the results.")
async def get_tweet_count_gender(gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T2.Gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the gender of the user with the highest reach
@app.get("/v1/social_media/top_gender_by_reach", operation_id="get_top_gender_by_reach", summary="Retrieves the gender of the user who has the highest reach on Twitter. The reach is determined by the number of followers and the user's influence on the platform. This operation does not require any input parameters and returns a single value representing the gender of the user with the highest reach.")
async def get_top_gender_by_reach():
    cursor.execute("SELECT T2.Gender FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID ORDER BY T1.Reach DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the count of tweets from users of a specific gender in a specific country
@app.get("/v1/social_media/tweet_count_gender_country", operation_id="get_tweet_count_gender_country", summary="Retrieves the total number of tweets posted by users of a specific gender in a given country. The operation requires the gender and country as input parameters to filter the data accordingly.")
async def get_tweet_count_gender_country(gender: str = Query(..., description="Gender of the user"), country: str = Query(..., description="Country name")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID INNER JOIN user AS T3 ON T1.UserID = T3.UserID WHERE T3.Gender = ? AND T2.Country = ?", (gender, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the text of tweets from a specific city by users of a specific gender
@app.get("/v1/social_media/tweet_text_city_gender", operation_id="get_tweet_text_city_gender", summary="Retrieves the text content of tweets originating from a specified city, authored by users of a particular gender. This operation enables targeted data extraction based on user demographics and location.")
async def get_tweet_text_city_gender(city: str = Query(..., description="City name"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT T1.text FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID INNER JOIN user AS T3 ON T1.UserID = T3.UserID WHERE T2.City = ? AND T3.Gender = ?", (city, gender))
    result = cursor.fetchall()
    if not result:
        return {"texts": []}
    return {"texts": [row[0] for row in result]}

# Endpoint to get the average percentage of tweets from a specific city in a specific country
@app.get("/v1/social_media/avg_tweets_city_country", operation_id="get_avg_tweets_city_country", summary="Retrieves the average percentage of tweets originating from a specific city within a given country. The calculation is based on the total number of tweets from the specified city divided by the total number of tweets from the specified country.")
async def get_avg_tweets_city_country(city: str = Query(..., description="City name"), country: str = Query(..., description="Country name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.City = ? THEN 1.0 ELSE 0 END) / COUNT(T1.TweetID) AS avg FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.Country = ?", (city, country))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of tweets from users of a specific gender with positive sentiment
@app.get("/v1/social_media/percentage_tweets_gender_sentiment", operation_id="get_percentage_tweets_gender_sentiment", summary="Retrieves the proportion of tweets from users of a specified gender that have a sentiment score above a given threshold. This operation calculates the ratio of qualifying tweets to the total number of tweets, providing insights into the positivity of tweets from users of a certain gender.")
async def get_percentage_tweets_gender_sentiment(gender: str = Query(..., description="Gender of the user"), min_sentiment: int = Query(..., description="Minimum sentiment value")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Gender = ? THEN 1.0 ELSE 0 END) / COUNT(T1.TweetID) AS per FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T1.Sentiment > ?", (gender, min_sentiment))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of users based on gender
@app.get("/v1/social_media/user_count_by_gender", operation_id="get_user_count_by_gender", summary="Retrieves the total number of users of a specific gender from the user database.")
async def get_user_count_by_gender(gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(UserID) AS user_number FROM user WHERE Gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"user_number": []}
    return {"user_number": result[0]}

# Endpoint to get the count of states in a specific country
@app.get("/v1/social_media/state_count_by_country", operation_id="get_state_count_by_country", summary="Retrieves the total number of states within a specified country. The operation requires the country name as an input parameter to filter the data and provide the accurate count.")
async def get_state_count_by_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT COUNT(State) AS State_number FROM location WHERE Country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"State_number": []}
    return {"State_number": result[0]}

# Endpoint to get distinct state codes based on state
@app.get("/v1/social_media/distinct_state_codes", operation_id="get_distinct_state_codes", summary="Retrieves a unique list of state codes associated with a specified state. This operation is useful for obtaining a comprehensive set of state codes for a given state, which can be used for further data analysis or filtering.")
async def get_distinct_state_codes(state: str = Query(..., description="State name")):
    cursor.execute("SELECT DISTINCT StateCode FROM location WHERE State = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"state_codes": []}
    return {"state_codes": [row[0] for row in result]}

# Endpoint to get distinct location IDs based on state
@app.get("/v1/social_media/distinct_location_ids", operation_id="get_distinct_location_ids", summary="Retrieves a unique set of location identifiers for a specified state. This operation allows you to obtain a comprehensive list of distinct location IDs within a given state, providing a concise overview of the available locations.")
async def get_distinct_location_ids(state: str = Query(..., description="State name")):
    cursor.execute("SELECT DISTINCT LocationID FROM location WHERE State = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"location_ids": []}
    return {"location_ids": [row[0] for row in result]}

# Endpoint to get the count of tweets based on state and reshare status
@app.get("/v1/social_media/tweet_count_by_state_reshare", operation_id="get_tweet_count_by_state_reshare", summary="Retrieves the total number of tweets originating from a specified state that have been reshared or not. The state is identified by its name, and the reshare status is determined by a boolean value.")
async def get_tweet_count_by_state_reshare(state: str = Query(..., description="State name"), is_reshare: str = Query(..., description="Reshare status (TRUE or FALSE)")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.State = ? AND T1.IsReshare = ?", (state, is_reshare))
    result = cursor.fetchone()
    if not result:
        return {"tweet_count": []}
    return {"tweet_count": result[0]}

# Endpoint to get the country of tweets based on reach
@app.get("/v1/social_media/country_by_reach", operation_id="get_country_by_reach", summary="Retrieves the country associated with tweets that have a specified reach. The reach parameter is used to filter the tweets and determine the corresponding country.")
async def get_country_by_reach(reach: int = Query(..., description="Reach of the tweet")):
    cursor.execute("SELECT T2.Country FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T1.Reach = ?", (reach,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the count of tweets based on sentiment and state
@app.get("/v1/social_media/tweet_count_by_sentiment_state", operation_id="get_tweet_count_by_sentiment_state", summary="Retrieves the total number of tweets with a sentiment score surpassing a specified value, originating from a particular state. The sentiment and state parameters are used to filter the results.")
async def get_tweet_count_by_sentiment_state(sentiment: float = Query(..., description="Sentiment value"), state: str = Query(..., description="State name")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T1.Sentiment > ? AND T2.State = ?", (sentiment, state))
    result = cursor.fetchone()
    if not result:
        return {"tweet_count": []}
    return {"tweet_count": result[0]}

# Endpoint to get the top tweet text based on Klout score in a specific state
@app.get("/v1/social_media/top_tweet_by_klout_state", operation_id="get_top_tweet_by_klout_state", summary="Retrieves the top-rated tweet text from a specific state, based on the Klout score of the tweet's author. The Klout score is a measure of social media influence, and the tweet returned is the one with the highest score in the specified state.")
async def get_top_tweet_by_klout_state(state: str = Query(..., description="State name")):
    cursor.execute("SELECT T1.text FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T2.State = ? ORDER BY T1.Klout DESC LIMIT 1", (state,))
    result = cursor.fetchone()
    if not result:
        return {"tweet_text": []}
    return {"tweet_text": result[0]}

# Endpoint to get the count of likes based on state and gender
@app.get("/v1/social_media/like_count_by_state_gender", operation_id="get_like_count_by_state_gender", summary="Retrieves the total number of likes for tweets originating from a specified state, filtered by the gender of the users who liked the tweets.")
async def get_like_count_by_state_gender(state: str = Query(..., description="State name"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.Likes) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID INNER JOIN user AS T3 ON T3.UserID = T1.UserID WHERE T2.State = ? AND T3.Gender = ?", (state, gender))
    result = cursor.fetchone()
    if not result:
        return {"like_count": []}
    return {"like_count": result[0]}

# Endpoint to get the gender of the user based on tweet ID
@app.get("/v1/social_media/user_gender_by_tweet_id", operation_id="get_user_gender_by_tweet_id", summary="Retrieves the gender of the user who posted the tweet identified by the provided Tweet ID. This operation requires a valid Tweet ID to be specified as an input parameter.")
async def get_user_gender_by_tweet_id(tweet_id: str = Query(..., description="Tweet ID")):
    cursor.execute("SELECT T2.Gender FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T1.TweetID = ?", (tweet_id,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the city of a tweet based on its text
@app.get("/v1/social_media/get_city_by_tweet_text", operation_id="get_city_by_tweet_text", summary="Retrieves the city associated with a specific tweet text. The operation searches for the tweet text in the database and returns the corresponding city from the location table.")
async def get_city_by_tweet_text(text: str = Query(..., description="Text of the tweet")):
    cursor.execute("SELECT T2.City FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T1.text = ?", (text,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the gender of a user based on the retweet count of their tweet
@app.get("/v1/social_media/get_gender_by_retweet_count", operation_id="get_gender_by_retweet_count", summary="Retrieves the gender of a user who has a specific number of retweets on their tweet. The operation uses the provided retweet count to identify the user and return their gender.")
async def get_gender_by_retweet_count(retweet_count: int = Query(..., description="Retweet count of the tweet")):
    cursor.execute("SELECT T2.Gender FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T1.RetweetCount = ?", (retweet_count,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the gender of the user with the highest Klout score on a specific weekday
@app.get("/v1/social_media/get_gender_by_weekday_highest_klout", operation_id="get_gender_by_weekday_highest_klout", summary="Retrieves the gender of the user who has the highest Klout score on a specific weekday. The weekday is provided as an input parameter.")
async def get_gender_by_weekday_highest_klout(weekday: str = Query(..., description="Weekday (e.g., 'Wednesday')")):
    cursor.execute("SELECT T2.Gender FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T1.Weekday = ? ORDER BY T1.Klout DESC LIMIT 1", (weekday,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the gender of the user with the highest number of likes on their tweet
@app.get("/v1/social_media/get_gender_by_highest_likes", operation_id="get_gender_by_highest_likes", summary="Retrieves the gender of the user who has received the highest number of likes on a single tweet. This operation does not require any input parameters and returns the gender of the user with the most popular tweet.")
async def get_gender_by_highest_likes():
    cursor.execute("SELECT T2.Gender FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID ORDER BY T1.Likes DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the count of tweets on a specific weekday in a specific state
@app.get("/v1/social_media/get_tweet_count_by_weekday_state", operation_id="get_tweet_count_by_weekday_state", summary="Retrieves the total number of tweets posted on a specific day of the week in a particular state. The operation requires the day of the week and the state as input parameters.")
async def get_tweet_count_by_weekday_state(weekday: str = Query(..., description="Weekday (e.g., 'Thursday')"), state: str = Query(..., description="State (e.g., 'Michigan')")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T1.Weekday = ? AND T2.State = ?", (weekday, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the state code of a tweet based on its TweetID
@app.get("/v1/social_media/get_state_code_by_tweet_id", operation_id="get_state_code_by_tweet_id", summary="Retrieves the state code associated with a specific tweet, identified by its unique TweetID. This operation fetches the state code from the location table, which is linked to the tweet via a common location identifier.")
async def get_state_code_by_tweet_id(tweet_id: str = Query(..., description="TweetID of the tweet")):
    cursor.execute("SELECT T2.StateCode FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE T1.TweetID = ?", (tweet_id,))
    result = cursor.fetchone()
    if not result:
        return {"state_code": []}
    return {"state_code": result[0]}

# Endpoint to get the percentage of male users in a specific state
@app.get("/v1/social_media/get_percentage_male_users_by_state", operation_id="get_percentage_male_users_by_state", summary="Retrieves the proportion of male users among all users who have tweeted from a specified state. The state is provided as an input parameter.")
async def get_percentage_male_users_by_state(state: str = Query(..., description="State (e.g., 'Florida')")):
    cursor.execute("SELECT SUM(CASE WHEN T3.Gender = 'Male' THEN 1.0 ELSE 0 END) / COUNT(T1.TweetID) AS percentage FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID INNER JOIN user AS T3 ON T3.UserID = T1.UserID WHERE T2.State = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of positive sentiment tweets in a specific state
@app.get("/v1/social_media/get_percentage_positive_sentiment_by_state", operation_id="get_percentage_positive_sentiment_by_state", summary="Retrieves the proportion of tweets with positive sentiment originating from a specified state. The calculation is based on the total number of tweets from the state.")
async def get_percentage_positive_sentiment_by_state(state: str = Query(..., description="State (e.g., 'California')")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Sentiment > 0 THEN 1.0 ELSE 0 END) / COUNT(T1.TweetID) AS percentage FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID WHERE State = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the weekday of a tweet based on its TweetID
@app.get("/v1/social_media/get_weekday_by_tweet_id", operation_id="get_weekday_by_tweet_id", summary="Retrieves the weekday associated with a specific tweet, identified by its unique TweetID.")
async def get_weekday_by_tweet_id(tweet_id: str = Query(..., description="TweetID of the tweet")):
    cursor.execute("SELECT Weekday FROM twitter WHERE TweetID = ?", (tweet_id,))
    result = cursor.fetchone()
    if not result:
        return {"weekday": []}
    return {"weekday": result[0]}

# Endpoint to get the reach of a tweet based on its text
@app.get("/v1/social_media/get_reach_by_tweet_text", operation_id="get_reach_by_tweet_text", summary="Retrieves the reach of a specific tweet by searching for the exact match of the provided tweet text. The reach is a measure of the potential audience size for the tweet.")
async def get_reach_by_tweet_text(text: str = Query(..., description="Text of the tweet")):
    cursor.execute("SELECT Reach FROM twitter WHERE text = ?", (text,))
    result = cursor.fetchone()
    if not result:
        return {"reach": []}
    return {"reach": result[0]}

# Endpoint to get the count of distinct tweet IDs based on language
@app.get("/v1/social_media/count_distinct_tweet_ids_by_lang", operation_id="get_count_distinct_tweet_ids_by_lang", summary="Retrieves the total number of unique tweets written in a specified language. This operation provides a count of distinct tweet IDs, offering insights into the volume of original content in the requested language.")
async def get_count_distinct_tweet_ids_by_lang(lang: str = Query(..., description="Language of the tweet")):
    cursor.execute("SELECT COUNT(DISTINCT TweetID) FROM twitter WHERE Lang = ?", (lang,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the location ID of a tweet based on tweet ID
@app.get("/v1/social_media/get_location_id_by_tweet_id", operation_id="get_location_id_by_tweet_id", summary="Retrieves the location identifier associated with a specific tweet. The operation requires the unique identifier of the tweet as input and returns the corresponding location identifier.")
async def get_location_id_by_tweet_id(tweet_id: str = Query(..., description="Tweet ID")):
    cursor.execute("SELECT LocationID FROM twitter WHERE TweetID = ?", (tweet_id,))
    result = cursor.fetchone()
    if not result:
        return {"location_id": []}
    return {"location_id": result[0]}

# Endpoint to get the count of tweets based on the day of the week
@app.get("/v1/social_media/count_tweets_by_weekday", operation_id="get_count_tweets_by_weekday", summary="Retrieves the total number of tweets that were posted on a specific day of the week. The day of the week is provided as an input parameter.")
async def get_count_tweets_by_weekday(weekday: str = Query(..., description="Day of the week (e.g., 'Wednesday')")):
    cursor.execute("SELECT COUNT(TweetID) FROM twitter WHERE Weekday = ?", (weekday,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get tweet texts based on the day of the week
@app.get("/v1/social_media/get_tweets_by_weekday", operation_id="get_tweets_by_weekday", summary="Retrieves the text content of tweets posted on a specified day of the week. The input parameter determines the day for which tweet texts are returned.")
async def get_tweets_by_weekday(weekday: str = Query(..., description="Day of the week (e.g., 'Thursday')")):
    cursor.execute("SELECT text FROM twitter WHERE Weekday = ?", (weekday,))
    result = cursor.fetchall()
    if not result:
        return {"tweets": []}
    return {"tweets": [row[0] for row in result]}

# Endpoint to get tweet texts based on user gender
@app.get("/v1/social_media/get_tweets_by_user_gender", operation_id="get_tweets_by_user_gender", summary="Retrieves the text content of tweets posted by users of a specified gender. The operation filters tweets based on the gender of the user who posted them, enabling targeted data extraction.")
async def get_tweets_by_user_gender(gender: str = Query(..., description="Gender of the user (e.g., 'Unknown')")):
    cursor.execute("SELECT T1.text FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T2.Gender = ?", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"tweets": []}
    return {"tweets": [row[0] for row in result]}

# Endpoint to get the gender with the highest number of tweets in a specific language
@app.get("/v1/social_media/get_top_gender_by_tweet_count_and_lang", operation_id="get_top_gender_by_tweet_count_and_lang", summary="Retrieves the gender with the highest number of tweets in a specified language. The operation filters tweets by the provided language and calculates the total count for each gender. The gender with the highest count is then returned.")
async def get_top_gender_by_tweet_count_and_lang(lang: str = Query(..., description="Language of the tweet")):
    cursor.execute("SELECT T.Gender FROM ( SELECT T2.Gender, COUNT( text) AS num FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T1.Lang = ? GROUP BY T2.Gender ) T ORDER BY T.num DESC LIMIT 1", (lang,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get distinct genders of users whose tweets have been retweeted more than a specified number of times
@app.get("/v1/social_media/get_distinct_genders_by_retweet_count", operation_id="get_distinct_genders_by_retweet_count", summary="Retrieves the unique genders of users who have had their tweets retweeted more than a specified number of times. This operation allows you to analyze the distribution of genders among users with high retweet counts, providing insights into the demographics of popular content creators on the platform. The input parameter specifies the minimum number of retweets required for a user to be included in the results.")
async def get_distinct_genders_by_retweet_count(retweet_count: int = Query(..., description="Minimum number of retweets")):
    cursor.execute("SELECT DISTINCT T2.Gender FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T1.RetweetCount > ?", (retweet_count,))
    result = cursor.fetchall()
    if not result:
        return {"genders": []}
    return {"genders": [row[0] for row in result]}

# Endpoint to get the count of users who reshared tweets based on gender
@app.get("/v1/social_media/count_users_reshare_by_gender", operation_id="count_users_reshare_by_gender", summary="Retrieves the total number of users who have reshared tweets, filtered by a specified gender. The operation considers whether the tweets were original posts or reshares.")
async def count_users_reshare_by_gender(gender: str = Query(..., description="Gender of the user (e.g., 'Female')"), is_reshare: str = Query(..., description="Whether the tweet is a reshare (e.g., 'TRUE')")):
    cursor.execute("SELECT COUNT(T1.UserID) FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T2.Gender = ? AND T1.IsReshare = ?", (gender, is_reshare))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country of a tweet based on TweetID
@app.get("/v1/social_media/tweet_country", operation_id="get_tweet_country", summary="Retrieves the country associated with a specific tweet, identified by its unique TweetID. This operation fetches the location data linked to the tweet and returns the corresponding country.")
async def get_tweet_country(tweet_id: str = Query(..., description="TweetID of the tweet")):
    cursor.execute("SELECT T2.Country FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.TweetID = ?", (tweet_id,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get tweet texts from a specific country
@app.get("/v1/social_media/tweets_by_country", operation_id="get_tweets_by_country", summary="Retrieves the text content of tweets originating from a specified country. The operation filters tweets based on their associated location and returns the text of those that match the provided country.")
async def get_tweets_by_country(country: str = Query(..., description="Country of the tweet")):
    cursor.execute("SELECT T1.text FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T2.Country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"tweets": []}
    return {"tweets": [row[0] for row in result]}

# Endpoint to get tweet texts from a specific city, state, and country
@app.get("/v1/social_media/tweets_by_location", operation_id="get_tweets_by_location", summary="Retrieves the text content of tweets originating from a specific city, state, and country. The operation filters tweets based on the provided location parameters, enabling users to obtain tweets from a particular geographical area.")
async def get_tweets_by_location(city: str = Query(..., description="City of the tweet"), state: str = Query(..., description="State of the tweet"), country: str = Query(..., description="Country of the tweet")):
    cursor.execute("SELECT T1.text FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T2.City = ? AND T2.State = ? AND T2.Country = ?", (city, state, country))
    result = cursor.fetchall()
    if not result:
        return {"tweets": []}
    return {"tweets": [row[0] for row in result]}

# Endpoint to get distinct languages of tweets from a specific country
@app.get("/v1/social_media/languages_by_country", operation_id="get_languages_by_country", summary="Retrieves a unique set of languages used in tweets originating from a specified country. The operation filters tweets based on the provided country and returns the distinct languages used in those tweets.")
async def get_languages_by_country(country: str = Query(..., description="Country of the tweet")):
    cursor.execute("SELECT DISTINCT T1.Lang FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T2.Country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the country with the highest positive sentiment
@app.get("/v1/social_media/top_positive_sentiment_country", operation_id="get_top_positive_sentiment_country", summary="Retrieves the country with the highest cumulative positive sentiment from Twitter data. The sentiment value is calculated as the sum of individual sentiment scores, which must exceed the provided minimum positive sentiment threshold. The data is filtered and aggregated by country.")
async def get_top_positive_sentiment_country(min_sentiment: int = Query(..., description="Minimum sentiment value to consider as positive")):
    cursor.execute("SELECT T.Country FROM ( SELECT T2.Country, SUM(T1.Sentiment) AS num FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.Sentiment > ? GROUP BY T2.Country ) T ORDER BY T.num DESC LIMIT 1", (min_sentiment,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the total likes of tweets in a specific language and gender
@app.get("/v1/social_media/total_likes_by_language_gender", operation_id="get_total_likes_by_language_gender", summary="Retrieves the total number of likes for tweets in a specified language and gender. The operation calculates the sum of likes from the Twitter database, filtering by the given language and the gender of the user who posted the tweets.")
async def get_total_likes_by_language_gender(lang: str = Query(..., description="Language of the tweet"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT SUM(T1.Likes) FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T1.Lang = ? AND T2.Gender = ?", (lang, gender))
    result = cursor.fetchone()
    if not result:
        return {"total_likes": []}
    return {"total_likes": result[0]}

# Endpoint to get the average number of tweets per user per day for a specific gender and date range
@app.get("/v1/social_media/avg_tweets_per_user_per_day", operation_id="get_avg_tweets_per_user_per_day", summary="Retrieves the average number of unique tweets per user per week for a specific gender within a given date range. The calculation is based on the count of distinct tweets divided by the count of distinct users, then divided by 7 to obtain the weekly average.")
async def get_avg_tweets_per_user_per_day(gender: str = Query(..., description="Gender of the user"), start_day: int = Query(..., description="Start day of the date range"), end_day: int = Query(..., description="End day of the date range")):
    cursor.execute("SELECT COUNT(DISTINCT T1.TweetID) / COUNT(DISTINCT T1.UserID) / 7 AS avg FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T2.Gender = ? AND T1.Day BETWEEN ? AND ?", (gender, start_day, end_day))
    result = cursor.fetchone()
    if not result:
        return {"avg_tweets_per_user_per_day": []}
    return {"avg_tweets_per_user_per_day": result[0]}

# Endpoint to get the count of tweets with Klout score greater than a specified value
@app.get("/v1/social_media/tweet_count_by_klout", operation_id="get_tweet_count_by_klout", summary="Get the count of tweets with Klout score greater than a specified value")
async def get_tweet_count_by_klout(min_klout: int = Query(..., description="Minimum Klout score")):
    cursor.execute("SELECT COUNT(DISTINCT T1.TweetID) FROM twitter WHERE Klout > ?", (min_klout,))
    result = cursor.fetchone()
    if not result:
        return {"tweet_count": []}
    return {"tweet_count": result[0]}

# Endpoint to get tweet texts in languages other than a specified language
@app.get("/v1/social_media/tweets_excluding_language", operation_id="get_tweets_excluding_language", summary="Retrieves the text content of tweets that are not written in the specified language. The operation filters out tweets in the provided language, returning only those in other languages.")
async def get_tweets_excluding_language(lang: str = Query(..., description="Language to exclude")):
    cursor.execute("SELECT text FROM twitter WHERE Lang != ?", (lang,))
    result = cursor.fetchall()
    if not result:
        return {"tweets": []}
    return {"tweets": [row[0] for row in result]}

# Endpoint to get the user with the most tweets
@app.get("/v1/social_media/top_user_by_tweets", operation_id="get_top_user_by_tweets", summary="Retrieves the user who has posted the highest number of unique tweets. The operation returns the user's unique identifier.")
async def get_top_user_by_tweets():
    cursor.execute("SELECT UserID FROM twitter GROUP BY UserID ORDER BY COUNT(DISTINCT TweetID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"top_user": []}
    return {"top_user": result[0]}

# Endpoint to get the count of distinct tweets based on weekday and reshare status
@app.get("/v1/social_media/count_distinct_tweets_weekday_reshare", operation_id="get_count_distinct_tweets_weekday_reshare", summary="Retrieves the total number of unique tweets that were posted on a specific day of the week and have a particular reshare status. The weekday and reshare status are provided as input parameters.")
async def get_count_distinct_tweets_weekday_reshare(weekday: str = Query(..., description="Weekday of the tweet"), is_reshare: str = Query(..., description="Reshare status of the tweet")):
    cursor.execute("SELECT COUNT(DISTINCT TweetID) FROM twitter WHERE Weekday = ? AND IsReshare = ?", (weekday, is_reshare))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get tweet texts ordered by reach
@app.get("/v1/social_media/tweet_texts_by_reach", operation_id="get_tweet_texts_by_reach", summary="Retrieves the text of the most widely-reached tweets, up to a specified limit. The tweets are ordered by their reach in descending order, providing the most popular tweets first.")
async def get_tweet_texts_by_reach(limit: int = Query(..., description="Number of tweets to return")):
    cursor.execute("SELECT text FROM twitter ORDER BY Reach DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"texts": []}
    return {"texts": [row[0] for row in result]}

# Endpoint to get the count of distinct tweets based on reshare status and likes
@app.get("/v1/social_media/count_distinct_tweets_reshare_likes", operation_id="get_count_distinct_tweets_reshare_likes", summary="Retrieves the count of unique tweets that have been reshared and have more than the specified number of likes. The reshare status and minimum number of likes are provided as input parameters.")
async def get_count_distinct_tweets_reshare_likes(is_reshare: str = Query(..., description="Reshare status of the tweet"), likes: int = Query(..., description="Minimum number of likes")):
    cursor.execute("SELECT COUNT(DISTINCT TweetID) FROM twitter WHERE IsReshare = ? AND Likes > ?", (is_reshare, likes))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct tweets based on user gender and weekday
@app.get("/v1/social_media/count_distinct_tweets_gender_weekday", operation_id="get_count_distinct_tweets_gender_weekday", summary="Retrieves the number of unique tweets posted by users of a specific gender on a particular day of the week. The gender and weekday are provided as input parameters.")
async def get_count_distinct_tweets_gender_weekday(gender: str = Query(..., description="Gender of the user"), weekday: str = Query(..., description="Weekday of the tweet")):
    cursor.execute("SELECT COUNT(DISTINCT T1.TweetID) FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T2.Gender = ? AND T1.Weekday = ?", (gender, weekday))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get tweet texts based on user gender and language
@app.get("/v1/social_media/tweet_texts_gender_language", operation_id="get_tweet_texts_gender_language", summary="Retrieves tweet texts based on the specified user gender and language. The operation filters tweets by the gender of the user who posted them and the language of the tweet, then returns the corresponding text content.")
async def get_tweet_texts_gender_language(gender: str = Query(..., description="Gender of the user"), lang: str = Query(..., description="Language of the tweet")):
    cursor.execute("SELECT T1.text FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T2.Gender = ? AND T1.Lang = ?", (gender, lang))
    result = cursor.fetchall()
    if not result:
        return {"texts": []}
    return {"texts": [row[0] for row in result]}

# Endpoint to get the count of distinct tweets based on language and country
@app.get("/v1/social_media/count_distinct_tweets_language_country", operation_id="get_count_distinct_tweets_language_country", summary="Retrieves the number of unique tweets written in a specific language and originating from a particular country. The language and country are provided as input parameters.")
async def get_count_distinct_tweets_language_country(lang: str = Query(..., description="Language of the tweet"), country: str = Query(..., description="Country of the tweet location")):
    cursor.execute("SELECT COUNT(DISTINCT T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.Lang = ? AND T2.Country = ?", (lang, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of tweets based on country, user gender, and sentiment
@app.get("/v1/social_media/count_tweets_country_gender_sentiment", operation_id="get_count_tweets_country_gender_sentiment", summary="Retrieves the total number of tweets originating from a specified country, posted by users of a certain gender, and having a sentiment score greater than a given threshold. This operation provides a quantitative measure of tweet activity based on geographical location, user demographics, and sentiment analysis.")
async def get_count_tweets_country_gender_sentiment(country: str = Query(..., description="Country of the tweet location"), gender: str = Query(..., description="Gender of the user"), sentiment: int = Query(..., description="Minimum sentiment value")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T2.LocationID = T1.LocationID INNER JOIN user AS T3 ON T3.UserID = T1.UserID WHERE T2.Country = ? AND T3.Gender = ? AND T1.Sentiment > ?", (country, gender, sentiment))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in count of tweets based on sentiment and user gender
@app.get("/v1/social_media/sentiment_diff_gender", operation_id="get_sentiment_diff_gender", summary="Retrieves the difference in the count of tweets with a positive sentiment and those with a neutral sentiment, filtered by the gender of the user.")
async def get_sentiment_diff_gender(positive_sentiment: int = Query(..., description="Positive sentiment value"), neutral_sentiment: int = Query(..., description="Neutral sentiment value"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Sentiment > ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.Sentiment = ? THEN 1 ELSE 0 END) AS diff FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T2.Gender = ?", (positive_sentiment, neutral_sentiment, gender))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get the city with the highest retweet count
@app.get("/v1/social_media/top_city_by_retweet_count", operation_id="get_top_city_by_retweet_count", summary="Retrieves the top city with the highest retweet count from the social media data. The number of cities returned can be specified, with the default being one.")
async def get_top_city_by_retweet_count(limit: int = Query(..., description="Number of cities to return")):
    cursor.execute("SELECT T2.City FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID ORDER BY T1.RetweetCount DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of tweets from specific cities
@app.get("/v1/social_media/tweet_count_by_cities", operation_id="get_tweet_count_by_cities", summary="Retrieves the total number of tweets originating from two specified cities. The response includes separate counts for each city, providing a comparative view of tweet activity between the two locations.")
async def get_tweet_count_by_cities(city1: str = Query(..., description="First city"), city2: str = Query(..., description="Second city")):
    cursor.execute("SELECT SUM(CASE WHEN T2.City = ? THEN 1 ELSE 0 END) AS bNum , SUM(CASE WHEN T2.City = ? THEN 1 ELSE 0 END) AS cNum FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T2.City IN (?, ?)", (city1, city2, city1, city2))
    result = cursor.fetchone()
    if not result:
        return {"counts": []}
    return {"counts": {"bNum": result[0], "cNum": result[1]}}

# Endpoint to get the count of tweets from a specific day, state, and country
@app.get("/v1/social_media/tweet_count_by_day_state_country", operation_id="get_tweet_count", summary="Retrieves the total number of tweets from a specific day, state, and country. The operation filters tweets based on the provided day, state, and country, and returns the count of matching tweets.")
async def get_tweet_count(day: int = Query(..., description="Day of the month"), state: str = Query(..., description="State"), country: str = Query(..., description="Country")):
    cursor.execute("SELECT COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.Day = ? AND T2.State = ? AND T2.Country = ?", (day, state, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top cities with the most tweets in a specific country
@app.get("/v1/social_media/top_cities_by_tweet_count", operation_id="get_top_cities", summary="Retrieves a list of the top cities in a specified country with the highest number of tweets, sorted in descending order. The list is limited to a specified number of cities.")
async def get_top_cities(country: str = Query(..., description="Country"), limit: int = Query(..., description="Limit of top cities to return")):
    cursor.execute("SELECT T.City FROM ( SELECT T2.City, COUNT(T1.TweetID) AS num FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T2.Country = ? GROUP BY T2.City ) T ORDER BY T.num DESC LIMIT ?", (country, limit))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get distinct cities with a specific sentiment
@app.get("/v1/social_media/distinct_cities_by_sentiment", operation_id="get_distinct_cities", summary="Retrieves a list of unique cities where the specified sentiment is found in Twitter data. The sentiment parameter determines the emotional tone to filter the cities by.")
async def get_distinct_cities(sentiment: int = Query(..., description="Sentiment value")):
    cursor.execute("SELECT DISTINCT T2.City FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE Sentiment = ?", (sentiment,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the most liked tweet text from a specific country and gender
@app.get("/v1/social_media/most_liked_tweet_by_country_gender", operation_id="get_most_liked_tweet", summary="Retrieves the text of the most liked tweet from a specific country and gender, based on the provided limit. The operation considers the user's gender and location to determine the most popular tweet.")
async def get_most_liked_tweet(country: str = Query(..., description="Country"), gender: str = Query(..., description="Gender"), limit: int = Query(..., description="Limit of top tweets to return")):
    cursor.execute("SELECT T2.text FROM user AS T1 INNER JOIN twitter AS T2 ON T1.UserID = T2.UserID INNER JOIN location AS T3 ON T2.LocationID = T3.LocationID WHERE T3.Country = ? AND T1.Gender = ? ORDER BY T2.Likes DESC LIMIT ?", (country, gender, limit))
    result = cursor.fetchall()
    if not result:
        return {"tweets": []}
    return {"tweets": [row[0] for row in result]}

# Endpoint to get the average likes per tweet for a specific gender and weekday
@app.get("/v1/social_media/average_likes_by_gender_weekday", operation_id="get_average_likes", summary="Retrieves the average number of likes per tweet for a specific gender on a given weekday. The gender and weekday are provided as input parameters.")
async def get_average_likes(gender: str = Query(..., description="Gender"), weekday: str = Query(..., description="Weekday")):
    cursor.execute("SELECT SUM(T1.Likes) / COUNT(T1.TweetID) FROM twitter AS T1 INNER JOIN user AS T2 ON T1.UserID = T2.UserID WHERE T2.Gender = ? AND T1.Weekday = ?", (gender, weekday))
    result = cursor.fetchone()
    if not result:
        return {"average_likes": []}
    return {"average_likes": result[0]}

# Endpoint to get the average likes per tweet for two specific cities
@app.get("/v1/social_media/average_likes_by_cities", operation_id="get_average_likes_by_cities", summary="Retrieves the average number of likes per tweet for two specified cities. The operation calculates the average by summing the total likes for each city and dividing by the total number of tweets in each city. The input parameters are used to specify the two cities for which the average likes are calculated.")
async def get_average_likes_by_cities(city1: str = Query(..., description="First city"), city2: str = Query(..., description="Second city")):
    cursor.execute("SELECT SUM(CASE WHEN T2.City = ? THEN Likes ELSE NULL END) / COUNT(CASE WHEN T2.City = ? THEN 1 ELSE 0 END) AS bNum , SUM(CASE WHEN City = ? THEN Likes ELSE NULL END) / COUNT(CASE WHEN City = ? THEN TweetID ELSE NULL END) AS cNum FROM twitter AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T2.City IN (?, ?)", (city1, city1, city2, city2, city1, city2))
    result = cursor.fetchone()
    if not result:
        return {"average_likes": []}
    return {"average_likes": {"city1": result[0], "city2": result[1]}}

api_calls = [
    "/v1/social_media/tweet_count_by_language?lang=en",
    "/v1/social_media/reshared_tweets_text?is_reshare=TRUE",
    "/v1/social_media/tweet_count_by_reach?reach=1000",
    "/v1/social_media/tweet_count_by_sentiment_weekday?sentiment=0&weekday=Thursday",
    "/v1/social_media/tweet_text_with_max_likes",
    "/v1/social_media/cities_by_country?country=Argentina",
    "/v1/social_media/tweet_count_by_country?country=Argentina",
    "/v1/social_media/top_city_by_tweets_in_country?country=Argentina",
    "/v1/social_media/reshared_tweets_count_by_city?city=Buenos%20Aires&is_reshare=TRUE",
    "/v1/social_media/tweets_text_by_sentiment_city?sentiment=0&city=Buenos%20Aires",
    "/v1/social_media/top_country_by_likes",
    "/v1/social_media/top_country_by_tweet_count?country1=Argentina&country2=Australia&min_sentiment=0",
    "/v1/social_media/tweet_count_city_weekday?city=Buenos%20Aires&weekday=Thursday",
    "/v1/social_media/tweet_count_likes_gender?min_likes=10&gender=Male",
    "/v1/social_media/tweet_count_gender?gender=Male",
    "/v1/social_media/top_gender_by_reach",
    "/v1/social_media/tweet_count_gender_country?gender=Male&country=Argentina",
    "/v1/social_media/tweet_text_city_gender?city=Buenos%20Aires&gender=Male",
    "/v1/social_media/avg_tweets_city_country?city=Buenos%20Aires&country=Argentina",
    "/v1/social_media/percentage_tweets_gender_sentiment?gender=Male&min_sentiment=0",
    "/v1/social_media/user_count_by_gender?gender=Unknown",
    "/v1/social_media/state_count_by_country?country=United%20Kingdom",
    "/v1/social_media/distinct_state_codes?state=Gwynedd",
    "/v1/social_media/distinct_location_ids?state=West%20Sussex",
    "/v1/social_media/tweet_count_by_state_reshare?state=Texas&is_reshare=TRUE",
    "/v1/social_media/country_by_reach?reach=547851",
    "/v1/social_media/tweet_count_by_sentiment_state?sentiment=0&state=Ha%20Noi",
    "/v1/social_media/top_tweet_by_klout_state?state=Connecticut",
    "/v1/social_media/like_count_by_state_gender?state=Wisconsin&gender=Female",
    "/v1/social_media/user_gender_by_tweet_id?tweet_id=tw-715909161071091712",
    "/v1/social_media/get_city_by_tweet_text?text=One%20of%20our%20favorite%20stories%20is%20%40FINRA_News%27s%20move%20to%20the%20cloud%20with%20AWS%20Enterprise%20Support!%20https://amp.twimg.com/v/991837f1-4815-4edc-a88f-e68ded09a02a",
    "/v1/social_media/get_gender_by_retweet_count?retweet_count=535",
    "/v1/social_media/get_gender_by_weekday_highest_klout?weekday=Wednesday",
    "/v1/social_media/get_gender_by_highest_likes",
    "/v1/social_media/get_tweet_count_by_weekday_state?weekday=Thursday&state=Michigan",
    "/v1/social_media/get_state_code_by_tweet_id?tweet_id=tw-685681052912873473",
    "/v1/social_media/get_percentage_male_users_by_state?state=Florida",
    "/v1/social_media/get_percentage_positive_sentiment_by_state?state=California",
    "/v1/social_media/get_weekday_by_tweet_id?tweet_id=tw-682712873332805633",
    "/v1/social_media/get_reach_by_tweet_text?text=Happy%20New%20Year%20to%20all%20those%20AWS%20instances%20of%20ours!",
    "/v1/social_media/count_distinct_tweet_ids_by_lang?lang=en",
    "/v1/social_media/get_location_id_by_tweet_id?tweet_id=tw-682714048199311366",
    "/v1/social_media/count_tweets_by_weekday?weekday=Wednesday",
    "/v1/social_media/get_tweets_by_weekday?weekday=Thursday",
    "/v1/social_media/get_tweets_by_user_gender?gender=Unknown",
    "/v1/social_media/get_top_gender_by_tweet_count_and_lang?lang=en",
    "/v1/social_media/get_distinct_genders_by_retweet_count?retweet_count=30",
    "/v1/social_media/count_users_reshare_by_gender?gender=Female&is_reshare=TRUE",
    "/v1/social_media/tweet_country?tweet_id=tw-682723090279841798",
    "/v1/social_media/tweets_by_country?country=Australia",
    "/v1/social_media/tweets_by_location?city=Rawang&state=Selangor&country=Malaysia",
    "/v1/social_media/languages_by_country?country=Brazil",
    "/v1/social_media/top_positive_sentiment_country?min_sentiment=0",
    "/v1/social_media/total_likes_by_language_gender?lang=ru&gender=Male",
    "/v1/social_media/avg_tweets_per_user_per_day?gender=Male&start_day=1&end_day=31",
    "/v1/social_media/tweet_count_by_klout?min_klout=50",
    "/v1/social_media/tweets_excluding_language?lang=en",
    "/v1/social_media/top_user_by_tweets",
    "/v1/social_media/count_distinct_tweets_weekday_reshare?weekday=Monday&is_reshare=TRUE",
    "/v1/social_media/tweet_texts_by_reach?limit=3",
    "/v1/social_media/count_distinct_tweets_reshare_likes?is_reshare=TRUE&likes=100",
    "/v1/social_media/count_distinct_tweets_gender_weekday?gender=Male&weekday=Monday",
    "/v1/social_media/tweet_texts_gender_language?gender=Male&lang=fr",
    "/v1/social_media/count_distinct_tweets_language_country?lang=fr&country=Australia",
    "/v1/social_media/count_tweets_country_gender_sentiment?country=Australia&gender=Male&sentiment=0",
    "/v1/social_media/sentiment_diff_gender?positive_sentiment=0&neutral_sentiment=0&gender=Male",
    "/v1/social_media/top_city_by_retweet_count?limit=1",
    "/v1/social_media/tweet_count_by_cities?city1=Bangkok&city2=Chiang%20Mai",
    "/v1/social_media/tweet_count_by_day_state_country?day=31&state=Santa&country=Argentina",
    "/v1/social_media/top_cities_by_tweet_count?country=Canada&limit=3",
    "/v1/social_media/distinct_cities_by_sentiment?sentiment=0",
    "/v1/social_media/most_liked_tweet_by_country_gender?country=Argentina&gender=Male&limit=1",
    "/v1/social_media/average_likes_by_gender_weekday?gender=Male&weekday=Monday",
    "/v1/social_media/average_likes_by_cities?city1=Bangkok&city2=Chiang%20Mai"
]
