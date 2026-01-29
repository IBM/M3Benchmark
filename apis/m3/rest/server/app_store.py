from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/app_store/app_store.sqlite')
cursor = conn.cursor()

# Endpoint to get distinct translated reviews for apps updated within a specific date range and having a specific sentiment
@app.get("/v1/app_store/distinct_translated_reviews", operation_id="get_distinct_translated_reviews", summary="Get distinct translated reviews for apps updated within a specific date range and having a specific sentiment")
async def get_distinct_translated_reviews(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), sentiment: str = Query(..., description="Sentiment of the review")):
    cursor.execute("SELECT DISTINCT Translated_Review FROM user_reviews WHERE App IN ( SELECT App FROM playstore WHERE `Last Updated` BETWEEN ? AND ? ) AND Sentiment = ?", (start_date, end_date, sentiment))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [row[0] for row in result]}

# Endpoint to get the count of sentiment polarity and last updated date for a specific app within a polarity range
@app.get("/v1/app_store/count_sentiment_polarity", operation_id="get_count_sentiment_polarity", summary="Get the count of sentiment polarity and last updated date for a specific app within a polarity range")
async def get_count_sentiment_polarity(app: str = Query(..., description="Name of the app"), min_polarity: float = Query(..., description="Minimum sentiment polarity"), max_polarity: float = Query(..., description="Maximum sentiment polarity")):
    cursor.execute("SELECT COUNT(T2.Sentiment_Polarity), T1.\"Last Updated\" FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ? AND T2.Sentiment_Polarity BETWEEN ? AND ?", (app, min_polarity, max_polarity))
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": [{"count": row[0], "last_updated": row[1]} for row in result]}

# Endpoint to get the count of ratings and the rating for a specific app with a specific sentiment
@app.get("/v1/app_store/count_ratings", operation_id="get_count_ratings", summary="Retrieves the total count of ratings and the specific rating value for a given app, filtered by a specified sentiment. This operation provides insights into the app's popularity and user feedback based on the selected sentiment.")
async def get_count_ratings(app: str = Query(..., description="Name of the app"), sentiment: str = Query(..., description="Sentiment of the review")):
    cursor.execute("SELECT COUNT(T1.Rating), T1.Rating FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ? AND T2.Sentiment = ?", (app, sentiment))
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": [{"count": row[0], "rating": row[1]} for row in result]}

# Endpoint to get distinct apps and their categories for a specific sentiment polarity
@app.get("/v1/app_store/distinct_apps_categories", operation_id="get_distinct_apps_categories", summary="Retrieves a list of unique applications and their respective categories based on a specified sentiment polarity. This operation filters the data from the user reviews and returns only the distinct applications and their categories that match the given sentiment polarity.")
async def get_distinct_apps_categories(sentiment_polarity: str = Query(..., description="Sentiment polarity of the review")):
    cursor.execute("SELECT DISTINCT T1.App, T1.Category FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment_Polarity = ?", (sentiment_polarity,))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": [{"app": row[0], "category": row[1]} for row in result]}

# Endpoint to get the average sentiment polarity and content rating for a specific app
@app.get("/v1/app_store/avg_sentiment_polarity", operation_id="get_avg_sentiment_polarity", summary="Retrieves the average sentiment polarity of user reviews and the content rating for a specified app. The sentiment polarity is calculated based on the reviews associated with the app, providing a measure of overall user sentiment. The content rating represents the app's maturity level as defined by the app store.")
async def get_avg_sentiment_polarity(app: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT AVG(T2.Sentiment_Polarity), T1.\"Content Rating\" FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ?", (app,))
    result = cursor.fetchall()
    if not result:
        return {"averages": []}
    return {"averages": [{"avg_polarity": row[0], "content_rating": row[1]} for row in result]}

# Endpoint to get the minimum sentiment polarity and installs for a specific app
@app.get("/v1/app_store/min_sentiment_polarity", operation_id="get_min_sentiment_polarity", summary="Retrieves the lowest sentiment polarity score and the corresponding install count for a specified application. The sentiment polarity score reflects the overall sentiment of user reviews for the app, with lower scores indicating more negative sentiment. The install count represents the total number of times the app has been installed.")
async def get_min_sentiment_polarity(app: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT MIN(T2.Sentiment_Polarity), T1.Installs FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ?", (app,))
    result = cursor.fetchall()
    if not result:
        return {"min_polarity": []}
    return {"min_polarity": [{"min_polarity": row[0], "installs": row[1]} for row in result]}

# Endpoint to get the count of sentiment subjectivity and current version for a specific app with subjectivity below a threshold
@app.get("/v1/app_store/count_sentiment_subjectivity", operation_id="get_count_sentiment_subjectivity", summary="Get the count of sentiment subjectivity and current version for a specific app with subjectivity below a threshold")
async def get_count_sentiment_subjectivity(app: str = Query(..., description="Name of the app"), max_subjectivity: float = Query(..., description="Maximum sentiment subjectivity")):
    cursor.execute("SELECT COUNT(T2.Sentiment_Subjectivity), T1.\"Current Ver\" FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ? AND T2.Sentiment_Subjectivity < ?", (app, max_subjectivity))
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": [{"count": row[0], "current_ver": row[1]} for row in result]}

# Endpoint to get the count of apps with a specific rating
@app.get("/v1/app_store/count_apps_by_rating", operation_id="get_count_apps_by_rating", summary="Retrieves the total number of applications that have a specified rating in the app store. The rating parameter is used to filter the count of apps.")
async def get_count_apps_by_rating(rating: int = Query(..., description="Rating of the app")):
    cursor.execute("SELECT COUNT(App) FROM playstore WHERE Rating = ?", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get apps with a specific price ordered by installs in descending order with a limit
@app.get("/v1/app_store/apps_by_price_ordered_by_installs", operation_id="get_apps_by_price_ordered_by_installs", summary="Get apps with a specific price ordered by installs in descending order with a limit")
async def get_apps_by_price_ordered_by_installs(price: float = Query(..., description="Price of the app"), limit: int = Query(..., description="Limit of the number of apps to return")):
    cursor.execute("SELECT App FROM playstore WHERE Price = ? ORDER BY CAST(REPLACE(REPLACE(Installs, ',', ''), '+', '') AS INTEGER) DESC LIMIT ?", (price, limit))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": [row[0] for row in result]}

# Endpoint to get distinct apps ordered by reviews in descending order with a limit
@app.get("/v1/app_store/distinct_apps_ordered_by_reviews", operation_id="get_distinct_apps_ordered_by_reviews", summary="Retrieves a distinct set of apps, ordered by their review count in descending order, up to a specified limit. This operation is useful for obtaining a ranked list of unique applications based on their review popularity.")
async def get_distinct_apps_ordered_by_reviews(limit: int = Query(..., description="Limit of the number of apps to return")):
    cursor.execute("SELECT DISTINCT App FROM playstore ORDER BY Reviews DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": [row[0] for row in result]}

# Endpoint to get the count of reviews and category for a specific app and sentiment
@app.get("/v1/app_store/review_count_category_by_app_sentiment", operation_id="get_review_count_category", summary="Retrieves the total number of reviews and associated category for a specified application and sentiment. The sentiment parameter filters the reviews based on the specified sentiment, while the app parameter identifies the application in question.")
async def get_review_count_category(app: str = Query(..., description="Name of the app"), sentiment: str = Query(..., description="Sentiment of the review")):
    cursor.execute("SELECT COUNT(T2.App), T1.Category FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ? AND T2.Sentiment = ?", (app, sentiment))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "category": []}
    return {"count": result[0], "category": result[1]}

# Endpoint to get distinct apps and their installs based on sentiment polarity
@app.get("/v1/app_store/distinct_apps_installs_by_sentiment_polarity", operation_id="get_distinct_apps_installs", summary="Retrieve a unique list of applications and their respective install counts, filtered by a specified sentiment polarity threshold. This operation provides insights into the popularity of apps that meet the sentiment polarity criteria.")
async def get_distinct_apps_installs(sentiment_polarity: float = Query(..., description="Sentiment polarity threshold")):
    cursor.execute("SELECT DISTINCT T1.App, T1.Installs FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment_Polarity > ?", (sentiment_polarity,))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": result}

# Endpoint to get apps and their translated reviews based on rating
@app.get("/v1/app_store/apps_translated_reviews_by_rating", operation_id="get_apps_translated_reviews", summary="Retrieves a list of apps and their translated user reviews that match the specified rating. The rating parameter is used to filter the results.")
async def get_apps_translated_reviews(rating: float = Query(..., description="Rating of the app")):
    cursor.execute("SELECT T1.App, T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Rating = ?", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get distinct sentiments for apps based on Android version
@app.get("/v1/app_store/distinct_sentiments_by_android_version", operation_id="get_distinct_sentiments", summary="Get distinct sentiments for apps based on Android version")
async def get_distinct_sentiments(android_version: str = Query(..., description="Android version")):
    cursor.execute("SELECT DISTINCT Sentiment FROM user_reviews WHERE App IN ( SELECT App FROM playstore WHERE `Android Ver` = ? )", (android_version,))
    result = cursor.fetchall()
    if not result:
        return {"sentiments": []}
    return {"sentiments": result}

# Endpoint to get the sum of sentiment subjectivity for apps with more than a specified number of genres
@app.get("/v1/app_store/sum_sentiment_subjectivity_by_genres", operation_id="get_sum_sentiment_subjectivity", summary="Retrieves the total sentiment subjectivity for apps that belong to more than a specified number of genres. The sentiment subjectivity is calculated based on user reviews. The input parameter determines the minimum number of genres an app must have to be included in the calculation.")
async def get_sum_sentiment_subjectivity(genres_count: int = Query(..., description="Number of genres")):
    cursor.execute("SELECT SUM(T2.Sentiment_Subjectivity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Genres > ?", (genres_count,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct apps and their sentiments based on the last updated year
@app.get("/v1/app_store/distinct_apps_sentiments_by_last_updated", operation_id="get_distinct_apps_sentiments", summary="Retrieves a list of unique applications and their corresponding sentiments from the user reviews, where the applications were last updated before the specified year. The input parameter 'last_updated_year' is used to filter the applications based on their last updated year.")
async def get_distinct_apps_sentiments(last_updated_year: int = Query(..., description="Last updated year")):
    cursor.execute("SELECT DISTINCT App, Sentiment FROM user_reviews WHERE App IN ( SELECT App FROM playstore WHERE CAST(SUBSTR('Last Updated', -4, 4) AS INTEGER) < ? )", (last_updated_year,))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": result}

# Endpoint to get the sum of installs and translated reviews for apps with a specific content rating
@app.get("/v1/app_store/sum_installs_translated_reviews_by_content_rating", operation_id="get_sum_installs_translated_reviews", summary="Retrieves the total number of installs and translated reviews for apps that are rated 'Adults only 18+'. The content rating parameter is used to filter the results.")
async def get_sum_installs_translated_reviews(content_rating: str = Query(..., description="Content rating of the app")):
    cursor.execute("SELECT SUM(T1.Installs), T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.\"Content Rating\" = ?", (content_rating,))
    result = cursor.fetchall()
    if not result:
        return {"installs": [], "reviews": []}
    return {"installs": result[0][0], "reviews": result[0][1]}

# Endpoint to get the app with the highest revenue based on price and installs
@app.get("/v1/app_store/highest_revenue_app", operation_id="get_highest_revenue_app", summary="Retrieves the application with the highest revenue, calculated by multiplying the price by the number of installs. The sentiment polarity of user reviews for the app is also returned.")
async def get_highest_revenue_app():
    cursor.execute("SELECT T1.App, T2.Sentiment_Polarity FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App ORDER BY T1.Price * CAST(REPLACE(REPLACE(Installs, ',', ''), '+', '') AS INTEGER) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"app": [], "sentiment_polarity": []}
    return {"app": result[0], "sentiment_polarity": result[1]}

# Endpoint to get the average rating and count of positive sentiments for a specific category
@app.get("/v1/app_store/avg_rating_positive_sentiment_count_by_category", operation_id="get_avg_rating_positive_sentiment_count", summary="Retrieves the average rating and count of positive sentiments for a specified app category. The sentiment parameter filters the reviews to be considered for the positive sentiment count. The category parameter determines the app category for which the average rating and positive sentiment count are calculated.")
async def get_avg_rating_positive_sentiment_count(category: str = Query(..., description="Category of the app"), sentiment: str = Query(..., description="Sentiment of the review")):
    cursor.execute("SELECT AVG(T1.Rating) , COUNT(CASE WHEN T2.Sentiment = ? THEN 1 ELSE NULL END) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Category = ?", (sentiment, category))
    result = cursor.fetchone()
    if not result:
        return {"avg_rating": [], "positive_sentiment_count": []}
    return {"avg_rating": result[0], "positive_sentiment_count": result[1]}

# Endpoint to get the rating of a specific app
@app.get("/v1/app_store/app_rating", operation_id="get_app_rating", summary="Retrieves the rating of a specified application from the app store. The operation requires the name of the app as an input parameter to identify the correct application and return its corresponding rating.")
async def get_app_rating(app: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT Rating FROM playstore WHERE APP = ?", (app,))
    result = cursor.fetchone()
    if not result:
        return {"rating": []}
    return {"rating": result[0]}

# Endpoint to get the count of reviews for a specific app where the review is translated
@app.get("/v1/app_store/count_translated_reviews", operation_id="get_count_translated_reviews", summary="Retrieves the total number of translated reviews for a specified application. The input parameter determines the application for which the count is calculated.")
async def get_count_translated_reviews(app: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT COUNT(App) FROM user_reviews WHERE App = ? AND Translated_Review IS NOT NULL", (app,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top 5 apps in a specific genre
@app.get("/v1/app_store/top_apps_by_genre", operation_id="get_top_apps_by_genre", summary="Retrieves the top 5 most popular applications within a specified genre. The genre is used to filter the results, and the applications are ranked based on their popularity, which is determined by the number of occurrences in the database.")
async def get_top_apps_by_genre(genre: str = Query(..., description="Genre of the app")):
    cursor.execute("SELECT DISTINCT App FROM playstore WHERE Genres = ? GROUP BY App ORDER BY COUNT(App) DESC LIMIT 5", (genre,))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": [row[0] for row in result]}

# Endpoint to get the count of reviews for a specific app with a specific sentiment
@app.get("/v1/app_store/count_reviews_by_sentiment", operation_id="get_count_reviews_by_sentiment", summary="Retrieves the total number of reviews for a specified application that match a given sentiment. The sentiment parameter allows for filtering the reviews based on their sentiment, such as positive, negative, or neutral.")
async def get_count_reviews_by_sentiment(app: str = Query(..., description="Name of the app"), sentiment: str = Query(..., description="Sentiment of the review")):
    cursor.execute("SELECT COUNT(App) FROM user_reviews WHERE App = ? AND Sentiment = ?", (app, sentiment))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct apps with a specific number of installs
@app.get("/v1/app_store/distinct_apps_by_installs", operation_id="get_distinct_apps_by_installs", summary="Retrieves a list of unique applications that have been installed a specified number of times. The input parameter determines the minimum number of installs required for an app to be included in the response.")
async def get_distinct_apps_by_installs(installs: str = Query(..., description="Number of installs (e.g., '5,000+')")):
    cursor.execute("SELECT DISTINCT App FROM playstore WHERE Installs = ?", (installs,))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": [row[0] for row in result]}

# Endpoint to get translated reviews for a specific app with a specific sentiment
@app.get("/v1/app_store/translated_reviews_by_sentiment", operation_id="get_translated_reviews_by_sentiment", summary="Retrieves translated reviews for a specified application based on the provided sentiment. The sentiment parameter filters the reviews to only include those with the specified sentiment. The app parameter is used to identify the application for which reviews are being retrieved.")
async def get_translated_reviews_by_sentiment(app: str = Query(..., description="Name of the app"), sentiment: str = Query(..., description="Sentiment of the review")):
    cursor.execute("SELECT Translated_Review FROM user_reviews WHERE App = ? AND Sentiment = ?", (app, sentiment))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [row[0] for row in result]}

# Endpoint to get the top app by negative sentiment for a specific app type
@app.get("/v1/app_store/top_app_by_negative_sentiment", operation_id="get_top_app_by_negative_sentiment", summary="Retrieves the app with the highest number of negative reviews for a given app type. The app type and sentiment are required as input parameters to filter the results.")
async def get_top_app_by_negative_sentiment(app_type: str = Query(..., description="Type of the app (e.g., 'Free')"), sentiment: str = Query(..., description="Sentiment of the review (e.g., 'Negative')")):
    cursor.execute("SELECT T1.App FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Type = ? AND T2.Sentiment = ? GROUP BY T1.App ORDER BY COUNT(T2.Sentiment) DESC LIMIT 1", (app_type, sentiment))
    result = cursor.fetchone()
    if not result:
        return {"app": []}
    return {"app": result[0]}

# Endpoint to get the count of negative sentiments for apps with a specific number of installs
@app.get("/v1/app_store/count_negative_sentiments_by_installs", operation_id="get_count_negative_sentiments_by_installs", summary="Retrieves the count of negative user reviews for apps that have a specified number of installs. The operation filters apps based on their install count and sentiment of the reviews, providing a quantitative measure of negative feedback for apps with a given install range.")
async def get_count_negative_sentiments_by_installs(installs: str = Query(..., description="Number of installs (e.g., '100,000,000+')"), sentiment: str = Query(..., description="Sentiment of the review (e.g., 'Negative')")):
    cursor.execute("SELECT COUNT(T2.Sentiment) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Installs = ? AND T2.Sentiment = ?", (installs, sentiment))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct content ratings for apps with reviews containing a specific phrase
@app.get("/v1/app_store/distinct_content_ratings_by_review_phrase", operation_id="get_distinct_content_ratings_by_review_phrase", summary="Retrieve the unique content ratings of apps that have reviews containing a specified phrase. The phrase is searched within the translated review text of each app. This operation is useful for identifying the content rating distribution of apps based on a particular review phrase.")
async def get_distinct_content_ratings_by_review_phrase(review_phrase: str = Query(..., description="Phrase to search in the translated review (e.g., '%gr8%')")):
    cursor.execute("SELECT DISTINCT T1.`Content Rating` FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Translated_Review LIKE ?", (review_phrase,))
    result = cursor.fetchall()
    if not result:
        return {"content_ratings": []}
    return {"content_ratings": [row[0] for row in result]}

# Endpoint to get the sum of sentiment polarity for the most expensive app
@app.get("/v1/app_store/sum_sentiment_polarity_most_expensive_app", operation_id="get_sum_sentiment_polarity_most_expensive_app", summary="Retrieves the cumulative sentiment polarity for the most expensive application in the app store. This operation calculates the total sentiment polarity based on user reviews, providing insights into the overall sentiment towards the highest-priced app.")
async def get_sum_sentiment_polarity_most_expensive_app():
    cursor.execute("SELECT SUM(T2.Sentiment_Polarity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Price = ( SELECT MAX(Price) FROM playstore )")
    result = cursor.fetchone()
    if not result:
        return {"sum_polarity": []}
    return {"sum_polarity": result[0]}

# Endpoint to get the rating and translated reviews for a specific app
@app.get("/v1/app_store/rating_and_reviews_by_app", operation_id="get_rating_and_reviews_by_app", summary="Retrieves the average rating and translated user reviews for a specified application. The operation requires the name of the app as an input parameter to filter the results.")
async def get_rating_and_reviews_by_app(app: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT T1.Rating, T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ?", (app,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [{"rating": row[0], "review": row[1]} for row in result]}

# Endpoint to get the top app by sentiment subjectivity in a specific genre
@app.get("/v1/app_store/top_app_by_sentiment_subjectivity", operation_id="get_top_app_by_sentiment_subjectivity", summary="Retrieves the top-rated application in a specified genre based on the cumulative sentiment subjectivity of user reviews. The sentiment subjectivity is a measure of how subjective or objective a review is, with higher values indicating a more subjective review. The genre parameter is used to filter the results to a specific category of applications.")
async def get_top_app_by_sentiment_subjectivity(genre: str = Query(..., description="Genre of the app")):
    cursor.execute("SELECT T1.App FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Genres = ? GROUP BY T1.App ORDER BY SUM(T2.Sentiment_Subjectivity) DESC LIMIT 1", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"app": []}
    return {"app": result[0]}

# Endpoint to get the top translated review for apps with a specific content rating
@app.get("/v1/app_store/top_translated_review_by_content_rating", operation_id="get_top_translated_review_by_content_rating", summary="Retrieves the top translated review for apps that have a 'Mature 17+' content rating. The review is selected based on the highest rating of the app. The content rating parameter is used to filter the apps.")
async def get_top_translated_review_by_content_rating(content_rating: str = Query(..., description="Content rating of the app")):
    cursor.execute("SELECT T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.\"Content Rating\" = ? ORDER BY T1.Rating LIMIT 1", (content_rating,))
    result = cursor.fetchone()
    if not result:
        return {"review": []}
    return {"review": result[0]}

# Endpoint to get the app with the highest sentiment polarity
@app.get("/v1/app_store/top_app_by_sentiment_polarity", operation_id="get_top_app_by_sentiment_polarity", summary="Retrieves the app with the highest overall sentiment polarity, based on aggregated user reviews. The sentiment polarity is calculated as the sum of individual review polarities. The app's total installs are also returned.")
async def get_top_app_by_sentiment_polarity():
    cursor.execute("SELECT T1.Installs FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App GROUP BY T1.App ORDER BY SUM(T2.Sentiment_Polarity) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"installs": []}
    return {"installs": result[0]}

# Endpoint to get the count of sentiments for apps in a specific genre
@app.get("/v1/app_store/count_sentiments_by_genre", operation_id="get_count_sentiments_by_genre", summary="Retrieves the total count of a specific sentiment type for apps within a given genre. This operation provides a quantitative measure of user sentiment towards apps in a particular genre, enabling data-driven insights into user preferences and app performance.")
async def get_count_sentiments_by_genre(genre: str = Query(..., description="Genre of the app"), sentiment: str = Query(..., description="Sentiment type")):
    cursor.execute("SELECT COUNT(T2.Sentiment) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Genres = ? AND T2.Sentiment = ?", (genre, sentiment))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top app by translated review count with specific installs
@app.get("/v1/app_store/top_app_by_translated_review_count", operation_id="get_top_app_by_translated_review_count", summary="Retrieves the top app based on the count of a specific translated review, filtered by a minimum install count of 1,000,000. The app with the highest number of occurrences of the specified translated review is returned.")
async def get_top_app_by_translated_review_count(installs: str = Query(..., description="Installs count of the app"), translated_review: str = Query(..., description="Translated review text")):
    cursor.execute("SELECT T1.App FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Installs = ? AND T2.Translated_Review = ? GROUP BY T1.App ORDER BY COUNT(T2.Translated_Review) DESC LIMIT 1", (installs, translated_review))
    result = cursor.fetchone()
    if not result:
        return {"app": []}
    return {"app": result[0]}

# Endpoint to get the rating and total sentiment subjectivity for a specific app
@app.get("/v1/app_store/rating_and_sentiment_subjectivity", operation_id="get_rating_and_sentiment_subjectivity", summary="Retrieves the average rating and cumulative sentiment subjectivity for a specified application. The sentiment subjectivity is calculated by aggregating individual user review scores.")
async def get_rating_and_sentiment_subjectivity(app: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT T1.Rating, SUM(T2.Sentiment_Subjectivity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ?", (app,))
    result = cursor.fetchone()
    if not result:
        return {"rating": [], "sentiment_subjectivity": []}
    return {"rating": result[0], "sentiment_subjectivity": result[1]}

# Endpoint to get the percentage of apps with a specific content rating and translated review
@app.get("/v1/app_store/percentage_content_rating_translated_review", operation_id="get_percentage_content_rating_translated_review", summary="Retrieves the percentage of apps that have a specified content rating and a translated review. The content rating parameter filters the apps, and the translated review parameter is used to include only those apps with a specific translated review text in their user reviews.")
async def get_percentage_content_rating_translated_review(content_rating: str = Query(..., description="Content rating of the app"), translated_review: str = Query(..., description="Translated review text")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.\"Content Rating\" = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.App) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Translated_Review = ?", (content_rating, translated_review))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct apps with a specific rating
@app.get("/v1/app_store/distinct_apps_by_rating", operation_id="get_distinct_apps_by_rating", summary="Retrieves a list of unique applications that have been assigned a specific rating. The rating is provided as an input parameter, allowing the user to filter the results based on their desired rating value.")
async def get_distinct_apps_by_rating(rating: float = Query(..., description="Rating of the app")):
    cursor.execute("SELECT DISTINCT App FROM playstore WHERE Rating = ?", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": [row[0] for row in result]}

# Endpoint to get distinct apps with more than a specific number of reviews and a specific content rating
@app.get("/v1/app_store/distinct_apps_by_reviews_content_rating", operation_id="get_distinct_apps_by_reviews_content_rating", summary="Get distinct apps with more than a specific number of reviews and a specific content rating")
async def get_distinct_apps_by_reviews_content_rating(reviews: int = Query(..., description="Number of reviews"), content_rating: str = Query(..., description="Content rating of the app")):
    cursor.execute("SELECT DISTINCT App FROM playstore WHERE Reviews > ? AND \`Content Rating\` = ?", (reviews, content_rating))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": [row[0] for row in result]}

# Endpoint to get genres of apps with a specific number of installs
@app.get("/v1/app_store/genres_by_installs", operation_id="get_genres_by_installs", summary="Retrieves the genres of apps that have surpassed a billion installs. The input parameter specifies the install count threshold for the apps to be considered.")
async def get_genres_by_installs(installs: str = Query(..., description="Installs count of the app")):
    cursor.execute("SELECT Genres FROM playstore WHERE Installs = ? GROUP BY Genres", (installs,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the average price of apps in a specific genre
@app.get("/v1/app_store/average_price_by_genre", operation_id="get_average_price_by_genre", summary="Retrieves the average price of apps belonging to a specified genre. The genre is provided as an input parameter, allowing the calculation of the average price for that particular category.")
async def get_average_price_by_genre(genre: str = Query(..., description="Genre of the app")):
    cursor.execute("SELECT AVG(Price) FROM playstore WHERE Genres = ?", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the average number of installs for apps in a specific category and size
@app.get("/v1/app_store/average_installs_by_category_and_size", operation_id="get_average_installs_by_category_and_size", summary="Retrieves the average number of installs for apps that belong to a specified category and are smaller than a given size. The category and size parameters are used to filter the apps for which the average install count is calculated.")
async def get_average_installs_by_category_and_size(category: str = Query(..., description="Category of the app"), size: str = Query(..., description="Size of the app in the format 'X.XM'")):
    cursor.execute("SELECT AVG(CAST(REPLACE(REPLACE(Installs, ',', ''), '+', '') AS INTEGER)) FROM playstore WHERE Category = ? AND Size < ?", (category, size))
    result = cursor.fetchone()
    if not result:
        return {"average_installs": []}
    return {"average_installs": result[0]}

# Endpoint to get the average number of reviews for apps with a specific rating
@app.get("/v1/app_store/average_reviews_by_rating", operation_id="get_average_reviews_by_rating", summary="Retrieves the average number of reviews for apps that have received a specified rating. This operation provides a statistical overview of user engagement for apps with a particular rating, enabling comparative analysis and informed decision-making.")
async def get_average_reviews_by_rating(rating: float = Query(..., description="Rating of the app")):
    cursor.execute("SELECT AVG(Reviews) FROM playstore WHERE Rating = ?", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"average_reviews": []}
    return {"average_reviews": result[0]}

# Endpoint to get genres of apps with specific sentiment and sentiment polarity
@app.get("/v1/app_store/genres_by_sentiment_and_polarity", operation_id="get_genres_by_sentiment_and_polarity", summary="Retrieves the genres of apps that have received reviews with a specified sentiment and sentiment polarity, sorted by sentiment polarity in descending order. The number of apps returned is limited by the provided limit parameter.")
async def get_genres_by_sentiment_and_polarity(sentiment: str = Query(..., description="Sentiment of the review"), sentiment_polarity: float = Query(..., description="Sentiment polarity of the review"), limit: int = Query(..., description="Limit of the number of apps")):
    cursor.execute("SELECT Genres FROM playstore WHERE App IN ( SELECT App FROM user_reviews WHERE Sentiment = ? AND Sentiment_Polarity > ? ORDER BY Sentiment_Polarity DESC LIMIT ? )", (sentiment, sentiment_polarity, limit))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the percentage of apps with more positive than negative reviews for a specific rating
@app.get("/v1/app_store/percentage_positive_reviews_by_rating", operation_id="get_percentage_positive_reviews_by_rating", summary="Retrieves the percentage of apps that have received more positive than negative reviews for a given rating. The rating parameter is used to filter the apps and calculate the percentage based on the sentiment of their reviews.")
async def get_percentage_positive_reviews_by_rating(rating: float = Query(..., description="Rating of the app")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN ( SELECT COUNT(CASE WHEN Sentiment = 'Positive' THEN 1 ELSE NULL END) - COUNT(CASE WHEN Sentiment = 'Negative' THEN 1 ELSE NULL END) FROM user_reviews GROUP BY App ) > 0 THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.Sentiment) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Rating = ?", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average sentiment polarity for apps excluding a specific sentiment
@app.get("/v1/app_store/average_sentiment_polarity_excluding_sentiment", operation_id="get_average_sentiment_polarity_excluding_sentiment", summary="Retrieves the average sentiment polarity for apps, excluding a specific sentiment. The sentiment polarity is calculated by averaging the sentiment polarity scores of user reviews for each app, excluding the specified sentiment. This operation provides a filtered view of the overall sentiment polarity for apps in the app store.")
async def get_average_sentiment_polarity_excluding_sentiment(sentiment: str = Query(..., description="Sentiment to exclude")):
    cursor.execute("SELECT T1.App, AVG(T2.Sentiment_Polarity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment != ? GROUP BY T1.App", (sentiment,))
    result = cursor.fetchall()
    if not result:
        return {"average_sentiment_polarity": []}
    return {"average_sentiment_polarity": [{"app": row[0], "polarity": row[1]} for row in result]}

# Endpoint to get the percentage difference between positive and negative reviews for apps updated after a specific year
@app.get("/v1/app_store/percentage_difference_reviews_by_year", operation_id="get_percentage_difference_reviews_by_year", summary="Get the percentage difference between positive and negative reviews for apps updated after a specific year")
async def get_percentage_difference_reviews_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST((( SELECT COUNT(*) Po FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE SUBSTR(T1.\"Last Updated\", -4, 4) > ? AND T2.Sentiment = 'Positive' ) - ( SELECT COUNT(*) Ne FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE SUBSTR(T1.\"Last Updated\", -4, 4) > ? AND T2.Sentiment = 'Negative' )) AS REAL) * 100 / ( SELECT COUNT(*) NUM FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE SUBSTR(T1.\"Last Updated\", -4, 4) > ? )", (year, year, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the percentage of apps updated after a specific year with a specific type and minimum rating
@app.get("/v1/app_store/percentage_updated_apps_by_type_and_rating", operation_id="get_percentage_updated_apps_by_type_and_rating", summary="Retrieves the percentage of apps of a specific type and minimum rating that were updated after a given year. The calculation is based on the total number of apps that meet the specified type and rating criteria.")
async def get_percentage_updated_apps_by_type_and_rating(year: str = Query(..., description="Year in 'YYYY' format"), type: str = Query(..., description="Type of the app"), rating: float = Query(..., description="Minimum rating of the app")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN SUBSTR('Last Updated', -4) > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(App) PER FROM playstore WHERE Type = ? AND Rating >= ?", (year, type, rating))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct genres for a specific app
@app.get("/v1/app_store/distinct_genres_by_app", operation_id="get_distinct_genres_by_app", summary="Retrieves a unique set of genres associated with a specific application. The operation filters the genres based on the provided application name.")
async def get_distinct_genres_by_app(app: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT DISTINCT Genres FROM playstore WHERE App = ?", (app,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get distinct ratings for a specific app
@app.get("/v1/app_store/distinct_ratings_by_app", operation_id="get_distinct_ratings_by_app", summary="Retrieves the unique ratings given to a specific application in the app store. The operation filters the ratings based on the provided app name.")
async def get_distinct_ratings_by_app(app: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT DISTINCT Rating FROM playstore WHERE App = ?", (app,))
    result = cursor.fetchall()
    if not result:
        return {"ratings": []}
    return {"ratings": [row[0] for row in result]}

# Endpoint to get the average price of apps based on content rating and genre
@app.get("/v1/app_store/average_price_by_content_rating_and_genre", operation_id="get_average_price", summary="Retrieves the average price of apps that are rated for users aged 10 and above and belong to a specific genre. The genre is provided as an input parameter.")
async def get_average_price(content_rating: str = Query(..., description="Content rating of the app"), genre: str = Query(..., description="Genre of the app")):
    cursor.execute("SELECT AVG(Price) FROM playstore WHERE 'Content Rating' = ? AND Genres = ?", (content_rating, genre))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the size and count of apps based on app name and sentiment polarity
@app.get("/v1/app_store/app_size_and_count_by_name_and_sentiment", operation_id="get_app_size_and_count", summary="Retrieves the total size and count of apps that match a specific name and meet a minimum sentiment polarity threshold. The sentiment polarity is determined by user reviews.")
async def get_app_size_and_count(app_name: str = Query(..., description="Name of the app"), sentiment_polarity: float = Query(..., description="Minimum sentiment polarity")):
    cursor.execute("SELECT T1.Size, COUNT(T1.App) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ? AND T2.Sentiment_Polarity >= ?", (app_name, sentiment_polarity))
    result = cursor.fetchone()
    if not result:
        return {"size": [], "count": []}
    return {"size": result[0], "count": result[1]}

# Endpoint to get distinct apps and their installs based on sentiment polarity
@app.get("/v1/app_store/distinct_apps_and_installs_by_sentiment", operation_id="get_distinct_apps_and_installs", summary="Retrieves a unique list of applications and their respective install counts based on a specified sentiment polarity. The sentiment polarity is used to filter the applications and their install counts.")
async def get_distinct_apps_and_installs(sentiment_polarity: float = Query(..., description="Sentiment polarity")):
    cursor.execute("SELECT DISTINCT T1.App, T1.Installs FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment_Polarity = ?", (sentiment_polarity,))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": [{"app": row[0], "installs": row[1]} for row in result]}

# Endpoint to get the average sentiment polarity and rating of an app
@app.get("/v1/app_store/average_sentiment_and_rating_by_app", operation_id="get_average_sentiment_and_rating", summary="Retrieves the average sentiment polarity and rating of a specific app from the app store. The sentiment polarity is calculated based on user reviews, while the rating is sourced from the app's details. The app name is required as an input parameter to identify the app.")
async def get_average_sentiment_and_rating(app_name: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT AVG(T2.Sentiment_Polarity), T1.Rating FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ?", (app_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_sentiment": [], "rating": []}
    return {"average_sentiment": result[0], "rating": result[1]}

# Endpoint to get the count of apps with a specific sentiment, ordered by rating
@app.get("/v1/app_store/app_count_by_sentiment_ordered_by_rating", operation_id="get_app_count_by_sentiment", summary="Retrieves the count of applications that have a specified sentiment in their reviews, ordered by their respective ratings. The number of results returned is limited by a specified value.")
async def get_app_count_by_sentiment(sentiment: str = Query(..., description="Sentiment of the reviews"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.App, COUNT(T1.App) COUNTNUMBER FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment = ? GROUP BY T1.App ORDER BY T1.Rating LIMIT ?", (sentiment, limit))
    result = cursor.fetchall()
    if not result:
        return {"apps": []}
    return {"apps": [{"app": row[0], "count": row[1]} for row in result]}

# Endpoint to get the percentage of positive reviews for a specific app and version
@app.get("/v1/app_store/positive_review_percentage_by_app_and_version", operation_id="get_positive_review_percentage", summary="Get the percentage of positive reviews for a specific app and version")
async def get_positive_review_percentage(app_name: str = Query(..., description="Name of the app"), current_ver: str = Query(..., description="Current version of the app")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Sentiment = 'Positive' THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN T2.Sentiment = 'Negative' THEN 1 ELSE 0 END), T1.`Current Ver` FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ? AND T1.`Current Ver` = ?", (app_name, current_ver))
    result = cursor.fetchone()
    if not result:
        return {"percentage": [], "current_ver": []}
    return {"percentage": result[0], "current_ver": result[1]}

# Endpoint to get the installs and percentage of positive reviews for a specific app
@app.get("/v1/app_store/installs_and_positive_review_percentage_by_app", operation_id="get_installs_and_positive_review_percentage", summary="Retrieves the total number of installs and the percentage of positive reviews for a specified app. The app is identified by its name, and the percentage of positive reviews is calculated based on the sentiment analysis of user reviews.")
async def get_installs_and_positive_review_percentage(app_name: str = Query(..., description="Name of the app")):
    cursor.execute("SELECT T1.Installs , CAST(SUM(CASE WHEN T2.Sentiment = 'Positive' THEN 1 ELSE 0 END) * 100 / SUM(CASE WHEN T2.Sentiment IS NOT NULL THEN 1.0 ELSE 0 END) AS REAL) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ?", (app_name,))
    result = cursor.fetchone()
    if not result:
        return {"installs": [], "percentage": []}
    return {"installs": result[0], "percentage": result[1]}

# Endpoint to get the maximum sentiment polarity and genres for a specific app
@app.get("/v1/app_store/max_sentiment_polarity_and_genres_by_app", operation_id="get_max_sentiment_polarity_and_genres", summary="Retrieves the maximum sentiment polarity and associated genres for a specified app, considering only sentiment polarities greater than a provided threshold. This operation filters the app's genres and identifies the highest sentiment polarity for each genre that surpasses the given threshold.")
async def get_max_sentiment_polarity_and_genres(app_name: str = Query(..., description="Name of the app"), sentiment_polarity: float = Query(..., description="Minimum sentiment polarity")):
    cursor.execute("SELECT MAX(T2.Sentiment_Polarity), T1.Genres FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ? AND T2.Sentiment_Polarity > ? GROUP BY T1.Genres", (app_name, sentiment_polarity))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [{"max_sentiment_polarity": row[0], "genre": row[1]} for row in result]}

# Endpoint to get the rating and count of sentiment polarities for a specific app
@app.get("/v1/app_store/rating_and_sentiment_polarity_count_by_app", operation_id="get_rating_and_sentiment_polarity_count", summary="Get the rating and count of sentiment polarities for a specific app with sentiment polarity less than a specified value")
async def get_rating_and_sentiment_polarity_count(app_name: str = Query(..., description="Name of the app"), sentiment_polarity: int = Query(..., description="Maximum sentiment polarity")):
    cursor.execute("SELECT T1.Rating, COUNT(T2.Sentiment_Polarity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = ? AND CAST(Sentiment_Polarity AS INTEGER) < ?", (app_name, sentiment_polarity))
    result = cursor.fetchone()
    if not result:
        return {"rating": [], "count": []}
    return {"rating": result[0], "count": result[1]}

# Endpoint to get the app and translated review for a specific category, ordered by rating
@app.get("/v1/app_store/app_and_translated_review_by_category", operation_id="get_app_and_translated_review", summary="Retrieves the top-rated app and its translated user review for a specified category. The app and review are sourced from the playstore and user_reviews tables, respectively, and are joined based on the app's unique identifier. The result is ordered by rating in ascending order and limited to a single record.")
async def get_app_and_translated_review(category: str = Query(..., description="Category of the app")):
    cursor.execute("SELECT T1.App, T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Category = ? GROUP BY T1.App, T2.Translated_Review ORDER BY T1.Rating ASC LIMIT 1", (category,))
    result = cursor.fetchone()
    if not result:
        return {"app": [], "translated_review": []}
    return {"app": result[0], "translated_review": result[1]}

# Endpoint to get app details and translated reviews based on app type and category
@app.get("/v1/app_store/app_details_translated_reviews", operation_id="get_app_details_translated_reviews", summary="Retrieves the details of apps and their translated reviews, filtered by a specific app type and category. The operation returns a list of apps and their corresponding translated reviews, based on the provided app type and category.")
async def get_app_details_translated_reviews(app_type: str = Query(..., description="Type of the app (e.g., 'Free')"), category: str = Query(..., description="Category of the app (e.g., 'SPORTS')")):
    cursor.execute("SELECT T1.App, T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Type = ? AND T1.Category = ?", (app_type, category))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the count of apps and average sentiment polarity based on content rating and genre
@app.get("/v1/app_store/app_count_avg_sentiment_polarity", operation_id="get_app_count_avg_sentiment_polarity", summary="Retrieves the total number of apps and the average sentiment polarity of user reviews for apps that match a specific content rating and genre. The content rating and genre are provided as input parameters.")
async def get_app_count_avg_sentiment_polarity(content_rating: str = Query(..., description="Content rating of the app (e.g., 'Teen')"), genre: str = Query(..., description="Genre of the app (e.g., 'Role Playing')")):
    cursor.execute("SELECT COUNT(T1.App), AVG(T2.Sentiment_Polarity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.\"Content Rating\" = ? AND T1.Genres = ?", (content_rating, genre))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "avg_sentiment_polarity": []}
    return {"count": result[0], "avg_sentiment_polarity": result[1]}

# Endpoint to get the average rating and percentage of positive sentiments based on genre
@app.get("/v1/app_store/avg_rating_positive_sentiment_percentage", operation_id="get_avg_rating_positive_sentiment_percentage", summary="Retrieves the average rating and the percentage of positive sentiments for apps belonging to a specific genre. The genre is provided as an input parameter, and the operation calculates the average rating from the 'playstore' table and the percentage of positive sentiments from the 'user_reviews' table.")
async def get_avg_rating_positive_sentiment_percentage(genre: str = Query(..., description="Genre of the app (e.g., 'Racing')")):
    cursor.execute("SELECT AVG(T1.Rating), CAST(COUNT(CASE WHEN T2.Sentiment = 'Positive' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.Sentiment) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Genres = ?", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"avg_rating": [], "positive_sentiment_percentage": []}
    return {"avg_rating": result[0], "positive_sentiment_percentage": result[1]}

api_calls = [
    "/v1/app_store/distinct_translated_reviews?start_date=January%201%2C%202018&end_date=January%2031%2C%202018&sentiment=Positive",
    "/v1/app_store/count_sentiment_polarity?app=7%20Minute%20Workout&min_polarity=0&max_polarity=0.5",
    "/v1/app_store/count_ratings?app=HTC%20Weather&sentiment=Neutral",
    "/v1/app_store/distinct_apps_categories?sentiment_polarity=-1.0",
    "/v1/app_store/avg_sentiment_polarity?app=Cooking%20Fever",
    "/v1/app_store/min_sentiment_polarity?app=Basketball%20Stars",
    "/v1/app_store/count_sentiment_subjectivity?app=Akinator&max_subjectivity=0.5",
    "/v1/app_store/count_apps_by_rating?rating=5",
    "/v1/app_store/apps_by_price_ordered_by_installs?price=0&limit=5",
    "/v1/app_store/distinct_apps_ordered_by_reviews?limit=10",
    "/v1/app_store/review_count_category_by_app_sentiment?app=10%20Best%20Foods%20for%20You&sentiment=Neutral",
    "/v1/app_store/distinct_apps_installs_by_sentiment_polarity?sentiment_polarity=0",
    "/v1/app_store/apps_translated_reviews_by_rating?rating=3.9",
    "/v1/app_store/distinct_sentiments_by_android_version?android_version=8.0%20and%20up",
    "/v1/app_store/sum_sentiment_subjectivity_by_genres?genres_count=1",
    "/v1/app_store/distinct_apps_sentiments_by_last_updated?last_updated_year=2015",
    "/v1/app_store/sum_installs_translated_reviews_by_content_rating?content_rating=Adults%20only%2018%2B",
    "/v1/app_store/highest_revenue_app",
    "/v1/app_store/avg_rating_positive_sentiment_count_by_category?category=COMICS&sentiment=Positive",
    "/v1/app_store/app_rating?app=Draw%20A%20Stickman",
    "/v1/app_store/count_translated_reviews?app=Brit%20%2B%20Co",
    "/v1/app_store/top_apps_by_genre?genre=Shopping",
    "/v1/app_store/count_reviews_by_sentiment?app=Dino%20War%3A%20Rise%20of%20Beasts&sentiment=Neutral",
    "/v1/app_store/distinct_apps_by_installs?installs=5%2C000%2B",
    "/v1/app_store/translated_reviews_by_sentiment?app=Dog%20Run%20-%20Pet%20Dog%20Simulator&sentiment=Negative",
    "/v1/app_store/top_app_by_negative_sentiment?app_type=Free&sentiment=Negative",
    "/v1/app_store/count_negative_sentiments_by_installs?installs=100%2C000%2C000%2B&sentiment=Negative",
    "/v1/app_store/distinct_content_ratings_by_review_phrase?review_phrase=%25gr8%25",
    "/v1/app_store/sum_sentiment_polarity_most_expensive_app",
    "/v1/app_store/rating_and_reviews_by_app?app=Garden%20Coloring%20Book",
    "/v1/app_store/top_app_by_sentiment_subjectivity?genre=Photography",
    "/v1/app_store/top_translated_review_by_content_rating?content_rating=Mature%2017%2B",
    "/v1/app_store/top_app_by_sentiment_polarity",
    "/v1/app_store/count_sentiments_by_genre?genre=Weather&sentiment=Neutral",
    "/v1/app_store/top_app_by_translated_review_count?installs=1%2C000%2C000%2B&translated_review=nan",
    "/v1/app_store/rating_and_sentiment_subjectivity?app=Onefootball%20-%20Soccer%20Scores",
    "/v1/app_store/percentage_content_rating_translated_review?content_rating=Teen&translated_review=nan",
    "/v1/app_store/distinct_apps_by_rating?rating=5",
    "/v1/app_store/distinct_apps_by_reviews_content_rating?reviews=75000000&content_rating=Teen",
    "/v1/app_store/genres_by_installs?installs=1%2C000%2C000%2C000%2B",
    "/v1/app_store/average_price_by_genre?genre=Dating",
    "/v1/app_store/average_installs_by_category_and_size?category=ENTERTAINMENT&size=1.0M",
    "/v1/app_store/average_reviews_by_rating?rating=5",
    "/v1/app_store/genres_by_sentiment_and_polarity?sentiment=Positive&sentiment_polarity=0.5&limit=3",
    "/v1/app_store/percentage_positive_reviews_by_rating?rating=4.7",
    "/v1/app_store/average_sentiment_polarity_excluding_sentiment?sentiment=Negative",
    "/v1/app_store/percentage_difference_reviews_by_year?year=2015",
    "/v1/app_store/percentage_updated_apps_by_type_and_rating?year=2018&type=Free&rating=4.5",
    "/v1/app_store/distinct_genres_by_app?app=Honkai%20Impact%203rd",
    "/v1/app_store/distinct_ratings_by_app?app=Learn%20C%2B%2B",
    "/v1/app_store/average_price_by_content_rating_and_genre?content_rating=Everyone%2010%2B&genre=Arcade",
    "/v1/app_store/app_size_and_count_by_name_and_sentiment?app_name=Browser%204G&sentiment_polarity=0.5",
    "/v1/app_store/distinct_apps_and_installs_by_sentiment?sentiment_polarity=0.3",
    "/v1/app_store/average_sentiment_and_rating_by_app?app_name=Golf%20GPS%20Rangefinder:%20Golf%20Pad",
    "/v1/app_store/app_count_by_sentiment_ordered_by_rating?sentiment=Negative&limit=5",
    "/v1/app_store/positive_review_percentage_by_app_and_version?app_name=Fate/Grand%20Order%20(English)&current_ver=1.18.0",
    "/v1/app_store/installs_and_positive_review_percentage_by_app?app_name=FREEDOME%20VPN%20Unlimited%20anonymous%20Wifi%20Security",
    "/v1/app_store/max_sentiment_polarity_and_genres_by_app?app_name=Honkai%20Impact%203rd&sentiment_polarity=0.5",
    "/v1/app_store/rating_and_sentiment_polarity_count_by_app?app_name=Dragon%20Ball%20Legends&sentiment_polarity=-0.5",
    "/v1/app_store/app_and_translated_review_by_category?category=EDUCATION",
    "/v1/app_store/app_details_translated_reviews?app_type=Free&category=SPORTS",
    "/v1/app_store/app_count_avg_sentiment_polarity?content_rating=Teen&genre=Role%20Playing",
    "/v1/app_store/avg_rating_positive_sentiment_percentage?genre=Racing"
]
