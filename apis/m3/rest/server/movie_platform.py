from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/movie_platform/movie_platform.sqlite')
cursor = conn.cursor()

# Endpoint to get the most popular movie title from a specific release year
@app.get("/v1/movie_platform/most_popular_movie_by_year", operation_id="get_most_popular_movie_by_year", summary="Retrieves the title of the most popular movie released in a specified year. The popularity is determined by a predefined metric, and the result is limited to a single movie title. The release year is provided as an input parameter.")
async def get_most_popular_movie_by_year(movie_release_year: int = Query(..., description="Release year of the movie")):
    cursor.execute("SELECT movie_title FROM movies WHERE movie_release_year = ? ORDER BY movie_popularity DESC LIMIT 1", (movie_release_year,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the most popular movie details
@app.get("/v1/movie_platform/most_popular_movie_details", operation_id="get_most_popular_movie_details", summary="Retrieves the title, release year, and director of the most popular movie, as determined by popularity rankings.")
async def get_most_popular_movie_details():
    cursor.execute("SELECT movie_title, movie_release_year, director_name FROM movies ORDER BY movie_popularity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie_details": []}
    return {"movie_details": {"movie_title": result[0], "movie_release_year": result[1], "director_name": result[2]}}

# Endpoint to get the movie with the longest popularity string
@app.get("/v1/movie_platform/movie_with_longest_popularity_string", operation_id="get_movie_with_longest_popularity_string", summary="Retrieves the movie with the longest popularity string, based on the length of the popularity attribute. The response includes the movie's title and release year.")
async def get_movie_with_longest_popularity_string():
    cursor.execute("SELECT movie_title, movie_release_year FROM movies ORDER BY LENGTH(movie_popularity) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie_details": []}
    return {"movie_details": {"movie_title": result[0], "movie_release_year": result[1]}}

# Endpoint to get the most frequently occurring movie title
@app.get("/v1/movie_platform/most_frequent_movie_title", operation_id="get_most_frequent_movie_title", summary="Retrieves the movie title that appears most frequently in the database. This operation returns the single most common movie title, providing insights into the most popular movie on the platform.")
async def get_most_frequent_movie_title():
    cursor.execute("SELECT movie_title FROM movies GROUP BY movie_title ORDER BY COUNT(movie_title) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the average popularity of movies directed by a specific director
@app.get("/v1/movie_platform/average_popularity_by_director", operation_id="get_average_popularity_by_director", summary="Retrieves the average popularity rating of movies directed by a specified director. The director's name is provided as an input parameter.")
async def get_average_popularity_by_director(director_name: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT AVG(movie_popularity) FROM movies WHERE director_name = ?", (director_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_popularity": []}
    return {"average_popularity": result[0]}

# Endpoint to get the average rating score of a specific movie
@app.get("/v1/movie_platform/average_rating_by_movie", operation_id="get_average_rating_by_movie", summary="Retrieves the average rating score for a specific movie. The operation calculates the average rating score by joining the movies and ratings tables on the movie_id field and filtering the results based on the provided movie title.")
async def get_average_rating_by_movie(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT AVG(T2.rating_score) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the latest rating details of a specific user
@app.get("/v1/movie_platform/latest_rating_details_by_user", operation_id="get_latest_rating_details_by_user", summary="Retrieves the most recent rating details for a specific user, including the user's avatar image URL and the date of the latest rating. The operation filters the data based on the provided user ID.")
async def get_latest_rating_details_by_user(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T3.user_avatar_image_url, T3.rating_date_utc FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id INNER JOIN ratings_users AS T3 ON T3.user_id = T2.user_id WHERE T3.user_id = ? ORDER BY T3.rating_date_utc DESC LIMIT 1", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"rating_details": []}
    return {"rating_details": {"user_avatar_image_url": result[0], "rating_date_utc": result[1]}}

# Endpoint to get the percentage of subscribers among users who rated movies
@app.get("/v1/movie_platform/percentage_of_subscribers", operation_id="get_percentage_of_subscribers", summary="Retrieves the percentage of subscribers among users who have rated movies. This operation calculates the proportion of subscribers by dividing the total number of subscribers who have rated movies by the total number of users who have rated movies.")
async def get_percentage_of_subscribers():
    cursor.execute("SELECT CAST(SUM(CASE WHEN user_subscriber = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings")
    result = cursor.fetchone()
    if not result:
        return {"percentage_of_subscribers": []}
    return {"percentage_of_subscribers": result[0]}

# Endpoint to get movie titles rated by trial users in a specific month
@app.get("/v1/movie_platform/movies_rated_by_trial_users", operation_id="get_movies_rated_by_trial_users", summary="Retrieves a list of movie titles that have been rated by trial users during a specific month. The month is determined by the provided rating timestamp in the 'YYYY-MM%' format.")
async def get_movies_rated_by_trial_users(rating_timestamp_utc: str = Query(..., description="Rating timestamp in 'YYYY-MM%' format")):
    cursor.execute("SELECT T1.movie_title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_trialist = 1 AND T2.rating_timestamp_utc LIKE ?", (rating_timestamp_utc,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get user IDs who rated a specific movie with a specific score
@app.get("/v1/movie_platform/user_ids_by_movie_and_score", operation_id="get_user_ids_by_movie_and_score", summary="Retrieves the user IDs of individuals who have rated a specific movie with a particular score. The operation requires the title of the movie and the corresponding rating score as input parameters.")
async def get_user_ids_by_movie_and_score(movie_title: str = Query(..., description="Title of the movie"), rating_score: int = Query(..., description="Rating score")):
    cursor.execute("SELECT T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.rating_score = ?", (movie_title, rating_score))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get distinct movie titles and their popularity based on a specific rating score
@app.get("/v1/movie_platform/distinct_movies_by_rating_score", operation_id="get_distinct_movies_by_rating_score", summary="Retrieves a list of unique movie titles along with their respective popularity scores, filtered by a specified rating score. This operation allows users to identify distinct movies that have been rated with a particular score, providing insights into the popularity of these movies.")
async def get_distinct_movies_by_rating_score(rating_score: int = Query(..., description="Rating score to filter movies")):
    cursor.execute("SELECT DISTINCT T2.movie_title, T2.movie_popularity FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = ?", (rating_score,))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": result}

# Endpoint to get movie titles rated in a specific year and after a specific month
@app.get("/v1/movie_platform/movies_rated_in_year_after_month", operation_id="get_movies_rated_in_year_after_month", summary="Retrieves a list of movie titles that were rated in the specified year and after the specified month. The year and month are provided as input parameters to filter the results.")
async def get_movies_rated_in_year_after_month(year: int = Query(..., description="Year to filter ratings (YYYY)"), month: int = Query(..., description="Month to filter ratings (MM)")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE CAST(SUBSTR(T1.rating_timestamp_utc, 1, 4) AS INTEGER) = ? AND CAST(SUBSTR(T1.rating_timestamp_utc, 6, 2) AS INTEGER) > ?", (year, month))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": result}

# Endpoint to get movie titles, user IDs, rating scores, and critics for movies rated by critics
@app.get("/v1/movie_platform/movies_rated_by_critics", operation_id="get_movies_rated_by_critics", summary="Retrieves a list of movies that have been rated by critics, along with the user IDs of the raters, their respective rating scores, and the critic status. This operation fetches data from the 'ratings' and 'movies' tables, filtering for movies that have been rated by critics.")
async def get_movies_rated_by_critics():
    cursor.execute("SELECT T2.movie_title, T1.user_id, T1.rating_score, T1.critic FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.critic IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": result}

# Endpoint to get the percentage of a specific rating score for a given movie title
@app.get("/v1/movie_platform/percentage_rating_score_for_movie", operation_id="get_percentage_rating_score_for_movie", summary="Retrieves the percentage of a specific rating score for a given movie title. This operation calculates the proportion of a particular rating score out of the total ratings for the specified movie. The input parameters include the rating score to calculate the percentage and the movie title to filter the ratings.")
async def get_percentage_rating_score_for_movie(rating_score: int = Query(..., description="Rating score to calculate percentage"), movie_title: str = Query(..., description="Movie title to filter ratings")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.rating_score = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = ?", (rating_score, movie_title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of movies released in a specific year
@app.get("/v1/movie_platform/percentage_movies_released_in_year", operation_id="get_percentage_movies_released_in_year", summary="Retrieves the percentage of movies released in a specific year, based on the total count of movies that have ratings. The year is provided as a four-digit input parameter.")
async def get_percentage_movies_released_in_year(movie_release_year: int = Query(..., description="Year to filter movie releases (YYYY)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.movie_release_year = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id", (movie_release_year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the director name for a specific movie title
@app.get("/v1/movie_platform/director_by_movie_title", operation_id="get_director_by_movie_title", summary="Retrieves the name of the director associated with a given movie title. The input parameter specifies the movie title for which the director's name is sought.")
async def get_director_by_movie_title(movie_title: str = Query(..., description="Movie title to get the director name")):
    cursor.execute("SELECT director_name FROM movies WHERE movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"director": []}
    return {"director": result[0]}

# Endpoint to get the most followed list
@app.get("/v1/movie_platform/most_followed_list", operation_id="get_most_followed_list", summary="Retrieves the title of the most followed list from the movie platform. The list is determined by the number of followers, with the list having the highest number of followers being returned.")
async def get_most_followed_list():
    cursor.execute("SELECT list_title FROM lists ORDER BY list_followers DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"list_title": []}
    return {"list_title": result[0]}

# Endpoint to get the most recently updated list URL within a specific year and follower range
@app.get("/v1/movie_platform/recent_list_url_by_year_followers", operation_id="get_recent_list_url_by_year_followers", summary="Retrieves the URL of the most recently updated list that falls within the specified year and follower range. The list is selected based on its latest update timestamp and the number of followers it has, which must be between the provided minimum and maximum values.")
async def get_recent_list_url_by_year_followers(year: str = Query(..., description="Year to filter list updates (YYYY)"), min_followers: int = Query(..., description="Minimum number of followers"), max_followers: int = Query(..., description="Maximum number of followers")):
    cursor.execute("SELECT list_url FROM lists WHERE list_update_timestamp_utc LIKE ? AND list_followers BETWEEN ? AND ? ORDER BY list_update_timestamp_utc DESC LIMIT 1", (year + '%', min_followers, max_followers))
    result = cursor.fetchone()
    if not result:
        return {"list_url": []}
    return {"list_url": result[0]}

# Endpoint to get the oldest list ID for a specific user
@app.get("/v1/movie_platform/oldest_list_id_by_user", operation_id="get_oldest_list_id_by_user", summary="Retrieves the ID of the oldest list associated with a given user. The list is determined by the earliest creation date.")
async def get_oldest_list_id_by_user(user_id: int = Query(..., description="User ID to get the oldest list ID")):
    cursor.execute("SELECT list_id FROM lists_users WHERE user_id = ? ORDER BY list_creation_date_utc ASC LIMIT 1", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"list_id": []}
    return {"list_id": result[0]}

# Endpoint to get the count of ratings for a specific movie with certain criteria
@app.get("/v1/movie_platform/count_ratings_by_criteria", operation_id="get_count_ratings_by_criteria", summary="Retrieves the total count of ratings for a specific movie, filtered by a maximum rating score, user eligibility for trial, and whether the user has a payment method. This operation provides a quantitative measure of user engagement and satisfaction for a given movie, considering various user attributes.")
async def get_count_ratings_by_criteria(movie_id: int = Query(..., description="Movie ID to filter ratings"), max_rating_score: int = Query(..., description="Maximum rating score"), user_eligible_for_trial: int = Query(..., description="User eligibility for trial (1 for eligible, 0 for not eligible)"), user_has_payment_method: int = Query(..., description="User has payment method (1 for yes, 0 for no)")):
    cursor.execute("SELECT COUNT(*) FROM ratings WHERE movie_id = ? AND rating_score <= ? AND user_eligible_for_trial = ? AND user_has_payment_method = ?", (movie_id, max_rating_score, user_eligible_for_trial, user_has_payment_method))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get movie title and popularity based on release year and director name
@app.get("/v1/movie_platform/movie_details_by_year_director", operation_id="get_movie_details", summary="Retrieves the title and popularity of a movie directed by a specific individual and released in a particular year. The operation requires the year of release and the director's name as input parameters.")
async def get_movie_details(movie_release_year: int = Query(..., description="Release year of the movie"), director_name: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT movie_title, movie_popularity FROM movies WHERE movie_release_year = ? AND director_name = ?", (movie_release_year, director_name))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": result}

# Endpoint to get the earliest movie release year and director name
@app.get("/v1/movie_platform/earliest_movie_release", operation_id="get_earliest_movie", summary="Retrieves the year of the earliest movie release and the name of its director from the movie database.")
async def get_earliest_movie():
    cursor.execute("SELECT movie_release_year, director_name FROM movies WHERE movie_release_year IS NOT NULL ORDER BY movie_release_year ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie": []}
    return {"movie": result}

# Endpoint to get user IDs of subscribers with a minimum span of list creation years
@app.get("/v1/movie_platform/subscriber_user_ids_by_year_span", operation_id="get_subscriber_user_ids", summary="Retrieves the user IDs of subscribers who have created lists spanning at least the specified number of years. The operation groups users by their IDs and filters those with a minimum difference of 10 years between the earliest and latest list creation dates.")
async def get_subscriber_user_ids(year_span: int = Query(..., description="Minimum span of list creation years")):
    cursor.execute("SELECT user_id FROM lists_users WHERE user_subscriber = 1 GROUP BY user_id HAVING MAX(SUBSTR(list_creation_date_utc, 1, 4)) - MIN(SUBSTR(list_creation_date_utc, 1, 4)) >= ?", (year_span,))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": result}

# Endpoint to get the count of users who rated a specific movie with a specific score
@app.get("/v1/movie_platform/user_count_by_movie_rating", operation_id="get_user_count_by_movie_rating", summary="Retrieves the number of users who have assigned a particular rating to a specific movie. The operation requires the title of the movie and the corresponding rating score as input parameters.")
async def get_user_count_by_movie_rating(movie_title: str = Query(..., description="Title of the movie"), rating_score: int = Query(..., description="Rating score")):
    cursor.execute("SELECT COUNT(T2.user_id) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = ? AND T2.rating_score = ?", (movie_title, rating_score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user eligibility for trial and list followers based on list title
@app.get("/v1/movie_platform/user_eligibility_list_followers", operation_id="get_user_eligibility_list_followers", summary="Retrieves the eligibility status for a user trial and lists the followers associated with a specific list title. The list title is used to filter the results.")
async def get_user_eligibility_list_followers(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT T2.user_eligible_for_trial, T1.list_followers FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T1.user_id AND T1.list_id = T2.list_id WHERE T1.list_title = ?", (list_title,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get movie release year and user ID based on director name and rating score
@app.get("/v1/movie_platform/movie_year_user_id_by_director_rating", operation_id="get_movie_year_user_id", summary="Retrieves the release year of a movie and the ID of a user who rated it, based on a specified director's name and rating score. The movie is the second oldest film directed by the given director. The user has given the specified rating score to this movie.")
async def get_movie_year_user_id(director_name: str = Query(..., description="Name of the director"), rating_score: int = Query(..., description="Rating score")):
    cursor.execute("SELECT T2.movie_release_year, T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_id = ( SELECT movie_id FROM movies WHERE director_name = ? ORDER BY movie_release_year ASC LIMIT 2, 1 ) AND T1.rating_score = ?", (director_name, rating_score))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get director URL based on user ID and critic likes
@app.get("/v1/movie_platform/director_url_by_user_critic_likes", operation_id="get_director_url", summary="Retrieves the URL of the director associated with the movie that a specific user has rated with a certain number of critic likes. The user is identified by their unique user ID, and the number of critic likes indicates the user's rating for the movie.")
async def get_director_url(user_id: int = Query(..., description="User ID"), critic_likes: int = Query(..., description="Number of critic likes")):
    cursor.execute("SELECT T2.director_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = ? AND T1.critic_likes = ?", (user_id, critic_likes))
    result = cursor.fetchall()
    if not result:
        return {"director_urls": []}
    return {"director_urls": result}

# Endpoint to get average rating score and director name based on movie title
@app.get("/v1/movie_platform/avg_rating_director_by_movie_title", operation_id="get_avg_rating_director", summary="Retrieves the average rating score and the name of the director associated with a specific movie title. The input parameter is the title of the movie.")
async def get_avg_rating_director(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT AVG(T1.rating_score), T2.director_name FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the latest list movie number and user payment method
@app.get("/v1/movie_platform/latest_list_movie_number_payment_method", operation_id="get_latest_list_movie_number", summary="Retrieves the most recent list of movies and the associated payment method used by the user. The data is fetched from the 'lists' and 'lists_users' tables, joined on the 'list_id' field, and sorted in descending order by the 'list_movie_number'. Only the top result is returned.")
async def get_latest_list_movie_number():
    cursor.execute("SELECT T1.list_movie_number, T2.user_has_payment_method FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id ORDER BY T1.list_movie_number DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the movie title with the highest critic likes
@app.get("/v1/movie_platform/top_movie_by_critic_likes", operation_id="get_top_movie_by_critic_likes", summary="Retrieves the title of the movie that has received the most positive reviews from professional critics. This operation fetches the movie with the highest number of critic likes from the ratings table and joins it with the movies table to obtain the corresponding movie title.")
async def get_top_movie_by_critic_likes():
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.critic_likes DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the maximum movie popularity and minimum rating timestamp for movies released between specific years with a specific rating score and user payment method
@app.get("/v1/movie_platform/max_min_popularity_timestamp", operation_id="get_max_min_popularity_timestamp", summary="Retrieves the highest movie popularity and earliest timestamp of a specific rating score for movies released within a specified range of years, considering whether the user has a payment method.")
async def get_max_min_popularity_timestamp(start_year: int = Query(..., description="Start year of the movie release range"), end_year: int = Query(..., description="End year of the movie release range"), rating_score: int = Query(..., description="Rating score"), user_has_payment_method: int = Query(..., description="User has payment method (1 for true, 0 for false)")):
    cursor.execute("SELECT MAX(T2.movie_popularity), MIN(T1.rating_timestamp_utc) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year BETWEEN ? AND ? AND T1.rating_score = ? AND T1.user_has_payment_method = ?", (start_year, end_year, rating_score, user_has_payment_method))
    result = cursor.fetchone()
    if not result:
        return {"max_popularity": [], "min_timestamp": []}
    return {"max_popularity": result[0], "min_timestamp": result[1]}

# Endpoint to get the count of movie titles and critic information for movies directed by a specific director and having a popularity greater than a specified value
@app.get("/v1/movie_platform/count_movies_director_popularity", operation_id="get_count_movies_director_popularity", summary="Retrieve the number of movies and associated critic information for films directed by a specific director that surpass a given popularity threshold.")
async def get_count_movies_director_popularity(director_name: str = Query(..., description="Director name"), movie_popularity: int = Query(..., description="Movie popularity threshold")):
    cursor.execute("SELECT COUNT(T2.movie_title), T1.critic FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.director_name = ? AND T2.movie_popularity > ?", (director_name, movie_popularity))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "critic": []}
    return {"count": result[0], "critic": result[1]}

# Endpoint to get the user avatar image URL for a specific user ID, rating score, and rating date
@app.get("/v1/movie_platform/user_avatar_image_url", operation_id="get_user_avatar_image_url", summary="Retrieves the avatar image URL of a user who has rated a movie with a specific score on a given date. The user is identified by their unique ID.")
async def get_user_avatar_image_url(user_id: int = Query(..., description="User ID"), rating_score: int = Query(..., description="Rating score"), rating_date_utc: str = Query(..., description="Rating date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.user_avatar_image_url FROM ratings AS T1 INNER JOIN ratings_users AS T2 ON T1.user_id = T2.user_id WHERE T2.user_id = ? AND rating_score = ? AND T2.rating_date_utc = ?", (user_id, rating_score, rating_date_utc))
    result = cursor.fetchone()
    if not result:
        return {"user_avatar_image_url": []}
    return {"user_avatar_image_url": result[0]}

# Endpoint to get the list followers and user subscriber status for a specific user ID, ordered by list followers in descending order
@app.get("/v1/movie_platform/list_followers_subscriber", operation_id="get_list_followers_subscriber", summary="Retrieves the number of followers for each list associated with a specific user, along with a boolean indicating whether the user is subscribed to each list. The results are ordered by the number of followers in descending order, with a limit of 1. The user ID is required as an input parameter.")
async def get_list_followers_subscriber(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T1.list_followers, T2.user_subscriber = 1 FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id AND T2.list_id = T2.list_id WHERE T2.user_id = ? ORDER BY T1.list_followers DESC LIMIT 1", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"list_followers": [], "user_subscriber": []}
    return {"list_followers": result[0], "user_subscriber": result[1]}

# Endpoint to get distinct movie release years and titles for movies with the highest number of releases in a year and a specific rating score
@app.get("/v1/movie_platform/distinct_movies_highest_releases", operation_id="get_distinct_movies_highest_releases", summary="Retrieves the distinct movie titles and their respective release years for movies that had the highest number of releases in a year, filtered by a specific rating score.")
async def get_distinct_movies_highest_releases(rating_score: int = Query(..., description="Rating score")):
    cursor.execute("SELECT DISTINCT T1.movie_release_year, T1.movie_title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_release_year = ( SELECT movie_release_year FROM movies GROUP BY movie_release_year ORDER BY COUNT(movie_id) DESC LIMIT 1 ) AND T2.rating_score = ?", (rating_score,))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": [{"release_year": row[0], "title": row[1]} for row in result]}

# Endpoint to get the count of user IDs for movies released in a specific year, directed by a specific director, with a specific rating score and user payment method
@app.get("/v1/movie_platform/count_user_ids_movie_criteria", operation_id="get_count_user_ids_movie_criteria", summary="Retrieves the count of unique users who have rated movies released in a specific year, directed by a specific director, and with a specific rating score. The count is further filtered by whether the users have a payment method on file. This endpoint provides insights into user engagement and preferences based on movie release year, director, and rating score.")
async def get_count_user_ids_movie_criteria(movie_release_year: int = Query(..., description="Movie release year"), director_name: str = Query(..., description="Director name"), rating_score: int = Query(..., description="Rating score"), user_has_payment_method: int = Query(..., description="User has payment method (1 for true, 0 for false)")):
    cursor.execute("SELECT COUNT(T2.user_id) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_release_year = ? AND T1.director_name = ? AND T2.rating_score = ? AND T2.user_has_payment_method = ?", (movie_release_year, director_name, rating_score, user_has_payment_method))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average list movie number and user avatar image URL for a specific user ID
@app.get("/v1/movie_platform/avg_list_movie_number_avatar", operation_id="get_avg_list_movie_number_avatar", summary="Retrieves the average number of movies listed and the associated user avatar image URL for a given user ID. This operation calculates the average from the user's movie lists and fetches the corresponding avatar image URL.")
async def get_avg_list_movie_number_avatar(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT AVG(T1.list_movie_number), T2.user_avatar_image_url FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T2.user_id = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"avg_list_movie_number": [], "user_avatar_image_url": []}
    return {"avg_list_movie_number": result[0], "user_avatar_image_url": result[1]}

# Endpoint to get the count of user IDs and rating URL for a specific movie title and rating score threshold
@app.get("/v1/movie_platform/count_user_ids_rating_url", operation_id="get_count_user_ids_rating_url", summary="Retrieves the count of unique user IDs and their associated rating URLs for a specified movie title, given a rating score threshold. This operation provides insights into the popularity and rating distribution of a movie.")
async def get_count_user_ids_rating_url(movie_title: str = Query(..., description="Movie title"), rating_score: int = Query(..., description="Rating score threshold")):
    cursor.execute("SELECT COUNT(T2.user_id), T2.rating_url FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = ? AND T2.rating_score <= ?", (movie_title, rating_score))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "rating_url": []}
    return {"count": result[0], "rating_url": result[1]}

# Endpoint to get the list followers for lists created within a specific date range and user eligibility for trial
@app.get("/v1/movie_platform/list_followers_date_range", operation_id="get_list_followers_date_range", summary="Retrieves a list of followers for movie lists created within a specified date range. The operation also checks the eligibility of the user for a trial. The date range is defined by the start and end dates provided in 'YYYY-MM-DD' format. The user's eligibility for trial is determined by a boolean value (1 for true, 0 for false).")
async def get_list_followers_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), user_eligible_for_trial: int = Query(..., description="User eligibility for trial (1 for true, 0 for false)")):
    cursor.execute("SELECT T1.list_followers FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id AND T1.list_id = T2.list_id WHERE T2.list_creation_date_utc BETWEEN ? AND ? AND T2.user_eligible_for_trial = ?", (start_date, end_date, user_eligible_for_trial))
    result = cursor.fetchall()
    if not result:
        return {"list_followers": []}
    return {"list_followers": [row[0] for row in result]}

# Endpoint to get the rating URL for a specific user ID, rating score, and movie title
@app.get("/v1/movie_platform/rating_url_user_movie", operation_id="get_rating_url_user_movie", summary="Retrieves the URL of a specific rating given by a user for a particular movie. The rating is identified by the user's unique ID, the rating score, and the title of the movie.")
async def get_rating_url_user_movie(user_id: int = Query(..., description="User ID"), rating_score: int = Query(..., description="Rating score"), movie_title: str = Query(..., description="Movie title")):
    cursor.execute("SELECT T2.rating_url FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_id = ? AND T2.rating_score = ? AND T1.movie_title = ?", (user_id, rating_score, movie_title))
    result = cursor.fetchone()
    if not result:
        return {"rating_url": []}
    return {"rating_url": result[0]}

# Endpoint to get directors who have directed more than a specified number of movies within a given year range
@app.get("/v1/movie_platform/directors_by_year_range", operation_id="get_directors_by_year_range", summary="Retrieve the names of directors who have directed more than a specified number of movies within a given year range. The operation filters movies based on their release year and counts the number of movies directed by each director. It then returns the names of directors who meet the specified minimum number of movies directed within the provided year range.")
async def get_directors_by_year_range(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), min_movies: int = Query(..., description="Minimum number of movies directed")):
    cursor.execute("SELECT T2.director_name FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year BETWEEN ? AND ? GROUP BY T2.director_name HAVING COUNT(T2.movie_id) > ?", (start_year, end_year, min_movies))
    result = cursor.fetchall()
    if not result:
        return {"directors": []}
    return {"directors": [row[0] for row in result]}

# Endpoint to get the critic likes for a specific movie rated with a specific score by non-trialist users
@app.get("/v1/movie_platform/critic_likes_by_movie_rating", operation_id="get_critic_likes_by_movie_rating", summary="Retrieve the number of critic likes for a specific movie, rated by non-trialist users with a particular score. The operation filters results based on the movie title, user trialist status, and rating score.")
async def get_critic_likes_by_movie_rating(user_trialist: int = Query(..., description="User trialist status (0 or 1)"), rating_score: int = Query(..., description="Rating score"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.critic_likes FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_trialist = ? AND T2.rating_score = ? AND T1.movie_title = ?", (user_trialist, rating_score, movie_title))
    result = cursor.fetchall()
    if not result:
        return {"critic_likes": []}
    return {"critic_likes": [row[0] for row in result]}

# Endpoint to get the average rating score for movies directed by a specific director
@app.get("/v1/movie_platform/avg_rating_by_director", operation_id="get_avg_rating_by_director", summary="Retrieves the average rating score for movies directed by the director of a specified movie. The input parameter is the title of the movie, which is used to identify the director and calculate the average rating of their movies.")
async def get_avg_rating_by_director(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT AVG(T2.rating_score), T1.director_name FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"avg_rating": [], "director_name": []}
    return {"avg_rating": result[0], "director_name": result[1]}

# Endpoint to get the earliest movie release year for the director with the most movies in a given year range
@app.get("/v1/movie_platform/earliest_movie_year_by_director", operation_id="get_earliest_movie_year_by_director", summary="Retrieves the release year of the earliest movie directed by the filmmaker with the most movies within a specified year range. The year range is defined by the provided start and end years.")
async def get_earliest_movie_year_by_director(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT MIN(movie_release_year) FROM movies WHERE director_name = ( SELECT T2.director_name FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year BETWEEN ? AND ? GROUP BY T2.director_name ORDER BY COUNT(T2.director_name) DESC LIMIT 1 )", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"earliest_year": []}
    return {"earliest_year": result[0]}

# Endpoint to get the movie titles and their highest rating scores within a given popularity range
@app.get("/v1/movie_platform/max_rating_by_popularity", operation_id="get_max_rating_by_popularity", summary="Retrieve the titles of movies and their highest rating scores that fall within a specified range of popularity. The popularity range is defined by the minimum and maximum popularity values provided as input parameters.")
async def get_max_rating_by_popularity(min_popularity: int = Query(..., description="Minimum movie popularity"), max_popularity: int = Query(..., description="Maximum movie popularity")):
    cursor.execute("SELECT T1.movie_title, MAX(T2.rating_score) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_popularity BETWEEN ? AND ? GROUP BY T1.movie_title", (min_popularity, max_popularity))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": [{"title": row[0], "max_rating": row[1]} for row in result]}

# Endpoint to get the rating URL for a specific movie rated by a specific user with a specific number of critic likes
@app.get("/v1/movie_platform/rating_url_by_user_movie", operation_id="get_rating_url_by_user_movie", summary="Retrieves the URL of a specific rating given by a user to a movie, based on the user's ID, the movie's title, and the number of critic likes the rating received.")
async def get_rating_url_by_user_movie(user_id: int = Query(..., description="User ID"), movie_title: str = Query(..., description="Title of the movie"), critic_likes: int = Query(..., description="Number of critic likes")):
    cursor.execute("SELECT T2.rating_url FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_id = ? AND T1.movie_title = ? AND T2.critic_likes = ?", (user_id, movie_title, critic_likes))
    result = cursor.fetchone()
    if not result:
        return {"rating_url": []}
    return {"rating_url": result[0]}

# Endpoint to get the average movie popularity for movies directed by a specific director
@app.get("/v1/movie_platform/avg_popularity_by_director", operation_id="get_avg_popularity_by_director", summary="Retrieves the average popularity rating of movies directed by a specific director. The input parameter specifies the director's name, which is used to filter the movies and calculate the average popularity.")
async def get_avg_popularity_by_director(director_name: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT AVG(T2.movie_popularity) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.director_name = ?", (director_name,))
    result = cursor.fetchone()
    if not result:
        return {"avg_popularity": []}
    return {"avg_popularity": result[0]}

# Endpoint to get movie titles rated by users within a specific time range and included in a specific list
@app.get("/v1/movie_platform/movie_titles_by_rating_time_list", operation_id="get_movie_titles_by_rating_time_list", summary="Retrieve a list of movie titles that have been rated by users within a specified date range and are included in a particular user-defined list. The start and end dates should be provided in 'YYYY-MM-DD' format, along with the title of the list.")
async def get_movie_titles_by_rating_time_list(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists AS T3 ON T3.user_id = T1.user_id WHERE T1.rating_timestamp_utc BETWEEN ? AND ? AND T3.list_title = ?", (start_date, end_date, list_title))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the average rating score and release year for a specific movie
@app.get("/v1/movie_platform/avg_rating_and_year_by_movie", operation_id="get_avg_rating_and_year_by_movie", summary="Retrieves the average rating score and release year of a specified movie. The operation calculates the average rating score from the ratings table and fetches the release year from the movies table, based on the provided movie title.")
async def get_avg_rating_and_year_by_movie(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT AVG(T1.rating_score), T2.movie_release_year FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"avg_rating": [], "release_year": []}
    return {"avg_rating": result[0], "release_year": result[1]}

# Endpoint to get the count of lists where the difference between update and creation year is greater than a specified number of years
@app.get("/v1/movie_platform/list_count_by_year_difference", operation_id="get_list_count_by_year_difference", summary="Retrieves the count of lists where the time elapsed between the last update and creation is greater than the specified number of years. This operation is useful for understanding the lifespan and activity of lists on the platform.")
async def get_list_count_by_year_difference(year_difference: int = Query(..., description="Difference in years between list update and creation timestamps")):
    cursor.execute("SELECT COUNT(*) FROM lists WHERE SUBSTR(list_update_timestamp_utc, 1, 4) - SUBSTR(list_creation_timestamp_utc, 1, 4) > ?", (year_difference,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the description of a list by its title
@app.get("/v1/movie_platform/list_description_by_title", operation_id="get_list_description_by_title", summary="Retrieves the description of a specific movie list based on its title. The provided title is used to locate the corresponding list and return its description.")
async def get_list_description_by_title(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT list_description FROM lists WHERE list_title = ?", (list_title,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the URL of a list by its title
@app.get("/v1/movie_platform/list_url_by_title", operation_id="get_list_url_by_title", summary="Retrieves the URL associated with a specific list, identified by its title. The operation returns the URL of the list that matches the provided title.")
async def get_list_url_by_title(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT list_url FROM lists WHERE list_title = ?", (list_title,))
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get the count of lists with more than a specified number of followers and updated after a specified date
@app.get("/v1/movie_platform/list_count_by_followers_and_update_date", operation_id="get_list_count_by_followers_and_update_date", summary="Retrieves the count of movie lists that have surpassed a certain number of followers and were updated after a specific date. The input parameters define the minimum number of followers and the date after which the lists were updated.")
async def get_list_count_by_followers_and_update_date(min_followers: int = Query(..., description="Minimum number of followers"), update_date: str = Query(..., description="Update date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(*) FROM lists WHERE list_followers > ? AND list_update_timestamp_utc > ?", (min_followers, update_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of list users by user ID and subscription status
@app.get("/v1/movie_platform/list_user_count_by_user_id_and_subscription", operation_id="get_list_user_count_by_user_id_and_subscription", summary="Retrieves the total number of users who have a specific subscription status for a given user ID. The user ID and subscription status are provided as input parameters.")
async def get_list_user_count_by_user_id_and_subscription(user_id: int = Query(..., description="User ID"), user_subscriber: int = Query(..., description="Subscription status (1 for subscribed, 0 for not subscribed)")):
    cursor.execute("SELECT COUNT(*) FROM lists_users WHERE user_id = ? AND user_subscriber = ?", (user_id, user_subscriber))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the release year of a movie by its title
@app.get("/v1/movie_platform/movie_release_year_by_title", operation_id="get_movie_release_year_by_title", summary="Retrieves the release year of a specific movie by its title. The input parameter is used to specify the title of the movie for which the release year is sought.")
async def get_movie_release_year_by_title(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT movie_release_year FROM movies WHERE movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"release_year": []}
    return {"release_year": result[0]}

# Endpoint to get the URL of a movie by its title
@app.get("/v1/movie_platform/movie_url_by_title", operation_id="get_movie_url_by_title", summary="Retrieves the URL of a specific movie by providing its title. The title is used to search the movie database and return the corresponding URL.")
async def get_movie_url_by_title(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT movie_url FROM movies WHERE movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get the most popular movie title from a list of titles
@app.get("/v1/movie_platform/most_popular_movie_by_titles", operation_id="get_most_popular_movie_by_titles", summary="Retrieves the most popular movie from a given pair of movie titles. The popularity is determined by a predefined metric, and the result is limited to a single movie title.")
async def get_most_popular_movie_by_titles(movie_title_1: str = Query(..., description="First movie title"), movie_title_2: str = Query(..., description="Second movie title")):
    cursor.execute("SELECT movie_title FROM movies WHERE movie_title = ? OR movie_title = ? ORDER BY movie_popularity DESC LIMIT 1", (movie_title_1, movie_title_2))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the count of movies directed by a specific director
@app.get("/v1/movie_platform/movie_count_by_director", operation_id="get_movie_count_by_director", summary="Retrieves the total number of movies directed by a given director. The director's name is used to filter the movies and calculate the count.")
async def get_movie_count_by_director(director_name: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT COUNT(movie_id) FROM movies WHERE director_name = ?", (director_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the trialist status of users for a specific list title
@app.get("/v1/movie_platform/user_trialist_by_list_title", operation_id="get_user_trialist_by_list_title", summary="Retrieves the trialist status of users associated with a specific list title. The list title is used to identify the relevant users and their corresponding trialist statuses.")
async def get_user_trialist_by_list_title(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT T2.user_trialist FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.list_title = ?", (list_title,))
    result = cursor.fetchone()
    if not result:
        return {"user_trialist": []}
    return {"user_trialist": result[0]}

# Endpoint to get list titles for a specific user who is eligible for a trial
@app.get("/v1/movie_platform/list_titles_user_eligible_for_trial", operation_id="get_list_titles_user_eligible_for_trial", summary="Retrieves a list of titles that a specific user is eligible to access during a trial period. The operation filters the titles based on the user's ID and their eligibility for the trial. The user's eligibility is determined by a binary flag, with 1 indicating eligibility and 0 indicating ineligibility.")
async def get_list_titles_user_eligible_for_trial(user_id: int = Query(..., description="User ID"), user_eligible_for_trial: int = Query(..., description="User eligibility for trial (1 for eligible, 0 for not eligible)")):
    cursor.execute("SELECT T1.list_title FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.user_id = ? AND T2.user_eligible_for_trial = ?", (user_id, user_eligible_for_trial))
    result = cursor.fetchall()
    if not result:
        return {"list_titles": []}
    return {"list_titles": [row[0] for row in result]}

# Endpoint to get the count of lists for a specific user with a certain number of movies and a payment method
@app.get("/v1/movie_platform/count_lists_user_movies_payment_method", operation_id="get_count_lists_user_movies_payment_method", summary="Retrieve the total count of lists that a specific user has, where each list contains more than a certain number of movies and the user has a payment method set up. The user is identified by their unique ID, and the minimum number of movies in a list is also specified.")
async def get_count_lists_user_movies_payment_method(user_id: int = Query(..., description="User ID"), list_movie_number: int = Query(..., description="Number of movies in the list"), user_has_payment_method: int = Query(..., description="User has payment method (1 for yes, 0 for no)")):
    cursor.execute("SELECT COUNT(*) FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.user_id = ? AND T1.list_movie_number > ? AND T2.user_has_payment_method = ?", (user_id, list_movie_number, user_has_payment_method))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the list description for a specific user with the highest number of followers
@app.get("/v1/movie_platform/list_description_user_highest_followers", operation_id="get_list_description_user_highest_followers", summary="Retrieves the description of the list with the highest number of followers for a given user. The user is identified by the provided user_id.")
async def get_list_description_user_highest_followers(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T1.list_description FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.user_id = ? ORDER BY T1.list_followers DESC LIMIT 1", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"list_description": []}
    return {"list_description": result[0]}

# Endpoint to get the latest update date of a list with a specific title
@app.get("/v1/movie_platform/latest_update_date_list_title", operation_id="get_latest_update_date_list_title", summary="Retrieves the most recent update date for a list identified by its title. The list must be associated with a specific user. The response includes the date and time of the latest update in UTC format.")
async def get_latest_update_date_list_title(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT T2.list_update_date_utc FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.list_title = ? ORDER BY T2.list_update_date_utc DESC LIMIT 1", (list_title,))
    result = cursor.fetchone()
    if not result:
        return {"latest_update_date": []}
    return {"latest_update_date": result[0]}

# Endpoint to get the user avatar image URL for a specific list title
@app.get("/v1/movie_platform/user_avatar_image_url_list_title", operation_id="get_user_avatar_image_url_list_title", summary="Retrieves the avatar image URL of the user associated with a specific list title. The list title is used to identify the user and their corresponding avatar image URL.")
async def get_user_avatar_image_url_list_title(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT T2.user_avatar_image_url FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.list_title = ?", (list_title,))
    result = cursor.fetchall()
    if not result:
        return {"user_avatar_image_urls": []}
    return {"user_avatar_image_urls": [row[0] for row in result]}

# Endpoint to get the count of list IDs for a specific list title
@app.get("/v1/movie_platform/count_list_ids_list_title", operation_id="get_count_list_ids_list_title", summary="Retrieves the total number of unique list IDs associated with a given list title. This operation provides a count of lists that a user has created with the specified title.")
async def get_count_list_ids_list_title(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT COUNT(list_id) FROM lists_users WHERE user_id = ( SELECT user_id FROM lists WHERE list_title = ? )", (list_title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users who rated a specific movie with a specific score
@app.get("/v1/movie_platform/count_users_movie_rating_score", operation_id="get_count_users_movie_rating_score", summary="Retrieves the total number of users who have assigned a specific rating score to a particular movie. The operation requires the movie title and the desired rating score as input parameters.")
async def get_count_users_movie_rating_score(movie_title: str = Query(..., description="Title of the movie"), rating_score: int = Query(..., description="Rating score")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.rating_score = ?", (movie_title, rating_score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the critics who rated a specific movie
@app.get("/v1/movie_platform/critics_movie_title", operation_id="get_critics_movie_title", summary="Retrieve a list of critics who have rated a specific movie. The operation uses the provided movie title to identify the relevant critics from the ratings database.")
async def get_critics_movie_title(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T1.critic FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ?", (movie_title,))
    result = cursor.fetchall()
    if not result:
        return {"critics": []}
    return {"critics": [row[0] for row in result]}

# Endpoint to get the count of ratings for a specific movie with a certain number of critic likes
@app.get("/v1/movie_platform/count_ratings_movie_critic_likes", operation_id="get_count_ratings_movie_critic_likes", summary="Retrieves the total number of ratings for a specified movie that has received more than a certain number of likes from critics. The movie is identified by its title, and the minimum number of critic likes is provided as input.")
async def get_count_ratings_movie_critic_likes(movie_title: str = Query(..., description="Title of the movie"), critic_likes: int = Query(..., description="Number of critic likes")):
    cursor.execute("SELECT COUNT(*) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.critic_likes > ?", (movie_title, critic_likes))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the user IDs who commented on a specific movie
@app.get("/v1/movie_platform/user_ids_movie_critic_comments", operation_id="get_user_ids_movie_critic_comments", summary="Retrieve the user IDs of individuals who have provided a specified number of critic comments on a particular movie. This operation fetches data from the 'ratings' and 'movies' tables, filtering based on the movie title and the count of critic comments.")
async def get_user_ids_movie_critic_comments(movie_title: str = Query(..., description="Title of the movie"), critic_comments: int = Query(..., description="Number of critic comments")):
    cursor.execute("SELECT T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.critic_comments = ?", (movie_title, critic_comments))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the rating score for a specific movie and user
@app.get("/v1/movie_platform/rating_score_by_movie_and_user", operation_id="get_rating_score", summary="Retrieves the rating score assigned by a specific user to a particular movie. The operation requires the movie title and user ID as input parameters to identify the relevant rating score.")
async def get_rating_score(movie_title: str = Query(..., description="Title of the movie"), user_id: int = Query(..., description="ID of the user")):
    cursor.execute("SELECT T1.rating_score FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.user_id = ?", (movie_title, user_id))
    result = cursor.fetchone()
    if not result:
        return {"rating_score": []}
    return {"rating_score": result[0]}

# Endpoint to get the rating URL for a specific movie and user
@app.get("/v1/movie_platform/rating_url_by_movie_and_user", operation_id="get_rating_url", summary="Retrieves the URL of the rating provided by a specific user for a given movie. The operation requires the title of the movie and the ID of the user as input parameters.")
async def get_rating_url(movie_title: str = Query(..., description="Title of the movie"), user_id: int = Query(..., description="ID of the user")):
    cursor.execute("SELECT T1.rating_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.user_id = ?", (movie_title, user_id))
    result = cursor.fetchone()
    if not result:
        return {"rating_url": []}
    return {"rating_url": result[0]}

# Endpoint to get the trialist status for a specific movie and user
@app.get("/v1/movie_platform/user_trialist_by_movie_and_user", operation_id="get_user_trialist", summary="Retrieves the trialist status of a user for a specific movie. This operation requires the movie title and user ID as input parameters. The trialist status indicates whether the user has tried the movie or not.")
async def get_user_trialist(movie_title: str = Query(..., description="Title of the movie"), user_id: int = Query(..., description="ID of the user")):
    cursor.execute("SELECT T1.user_trialist FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.user_id = ?", (movie_title, user_id))
    result = cursor.fetchone()
    if not result:
        return {"user_trialist": []}
    return {"user_trialist": result[0]}

# Endpoint to get the count of trialist users for a specific movie
@app.get("/v1/movie_platform/count_trialist_users_by_movie", operation_id="get_count_trialist_users", summary="Retrieves the number of users who are currently in the trial period and have rated a specific movie. The movie is identified by its title, and the trialist status is determined by the provided user_trialist parameter.")
async def get_count_trialist_users(movie_title: str = Query(..., description="Title of the movie"), user_trialist: int = Query(..., description="Trialist status of the user (1 for trialist)")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.user_trialist = ?", (movie_title, user_trialist))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the rating URL for a specific movie where critic is not null
@app.get("/v1/movie_platform/rating_url_by_movie_with_critic", operation_id="get_rating_url_with_critic", summary="Get the rating URL for a specific movie where critic is not null")
async def get_rating_url_with_critic(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T1.rating_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.critic IS NOT NULL", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"rating_url": []}
    return {"rating_url": result[0]}

# Endpoint to get the count of ratings for the most popular movie
@app.get("/v1/movie_platform/count_ratings_most_popular_movie", operation_id="get_count_ratings_most_popular_movie", summary="Retrieves the total number of ratings for the most popular movie, as determined by its popularity score.")
async def get_count_ratings_most_popular_movie():
    cursor.execute("SELECT COUNT(rating_id) FROM ratings WHERE movie_id = ( SELECT movie_id FROM movies ORDER BY movie_popularity DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the movie titles for a specific user with specific critic likes and comments
@app.get("/v1/movie_platform/movie_titles_by_user_critic_likes_comments", operation_id="get_movie_titles", summary="Retrieve the titles of movies that a specific user has rated, based on the number of critic likes and comments associated with those ratings. This operation requires the user's ID, the number of critic likes, and the number of critic comments as input parameters.")
async def get_movie_titles(user_id: int = Query(..., description="ID of the user"), critic_likes: int = Query(..., description="Number of critic likes"), critic_comments: int = Query(..., description="Number of critic comments")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = ? AND T1.critic_likes = ? AND T1.critic_comments = ?", (user_id, critic_likes, critic_comments))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the count of trialist users for a specific movie with a specific rating score
@app.get("/v1/movie_platform/count_trialist_users_by_movie_rating_score", operation_id="get_count_trialist_users_by_rating_score", summary="Retrieves the number of trialist users who have rated a specific movie with a specific rating score. The movie is identified by its title, and the rating score is a numerical value. The trialist status of the user is also considered, with 1 indicating a trialist user.")
async def get_count_trialist_users_by_rating_score(movie_title: str = Query(..., description="Title of the movie"), rating_score: int = Query(..., description="Rating score"), user_trialist: int = Query(..., description="Trialist status of the user (1 for trialist)")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.rating_score = ? AND T1.user_trialist = ?", (movie_title, rating_score, user_trialist))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of ratings for a specific movie after a specific date
@app.get("/v1/movie_platform/count_ratings_by_movie_after_date", operation_id="get_count_ratings_by_movie_after_date", summary="Retrieves the total number of ratings received by a specific movie after a given date. The operation requires the movie title and the date (in 'YYYY-MM-DD' format) as input parameters. The date parameter is used to filter ratings that were submitted on or after the specified date.")
async def get_count_ratings_by_movie_after_date(movie_title: str = Query(..., description="Title of the movie"), rating_timestamp_utc: str = Query(..., description="Rating timestamp in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.rating_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.rating_timestamp_utc >= ?", (movie_title, rating_timestamp_utc))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the rating score for a specific movie and list title
@app.get("/v1/movie_platform/rating_score_by_movie_and_list_title", operation_id="get_rating_score_by_list_title", summary="Retrieves the rating score for a specific movie from a user's list. The operation requires the title of the movie and the title of the list as input parameters. The rating score is determined by matching the provided movie title with the movie title in the movies table and the provided list title with the list title in the lists table.")
async def get_rating_score_by_list_title(movie_title: str = Query(..., description="Title of the movie"), list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT T1.rating_score FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists AS T3 ON T3.user_id = T1.user_id WHERE T2.movie_title = ? AND T3.list_title = ?", (movie_title, list_title))
    result = cursor.fetchone()
    if not result:
        return {"rating_score": []}
    return {"rating_score": result[0]}

# Endpoint to get movie titles from a specific list
@app.get("/v1/movie_platform/movie_titles_from_list", operation_id="get_movie_titles_from_list", summary="Retrieves the titles of movies that belong to a specific user-defined list. The list is identified by its title, which is provided as an input parameter.")
async def get_movie_titles_from_list(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists AS T3 ON T3.user_id = T1.user_id WHERE T3.list_title = ?", (list_title,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the average rating score of a specific movie
@app.get("/v1/movie_platform/average_rating_score", operation_id="get_average_rating_score", summary="Retrieves the average rating score for a specific movie. The operation calculates the average rating score based on the provided movie title, considering all associated ratings in the database.")
async def get_average_rating_score(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT AVG(T1.rating_score) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"average_rating_score": []}
    return {"average_rating_score": result[0]}

# Endpoint to get the percentage of 1-star ratings for a specific movie
@app.get("/v1/movie_platform/percentage_one_star_ratings", operation_id="get_percentage_one_star_ratings", summary="Retrieves the percentage of 1-star ratings given to a specific movie. The calculation is based on the total number of ratings for the movie. The movie is identified by its title.")
async def get_percentage_one_star_ratings(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.rating_score = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_one_star_ratings": []}
    return {"percentage_one_star_ratings": result[0]}

# Endpoint to get the difference in average ratings between two movies
@app.get("/v1/movie_platform/rating_difference_between_movies", operation_id="get_rating_difference_between_movies", summary="Retrieves the difference in average ratings between two specified movies. The operation calculates the average rating for each movie by summing the rating scores and dividing by the total number of ratings. The difference between these averages is then returned.")
async def get_rating_difference_between_movies(movie_title_1: str = Query(..., description="Title of the first movie"), movie_title_2: str = Query(..., description="Title of the second movie")):
    cursor.execute("SELECT SUM(CASE WHEN T2.movie_title = ? THEN T1.rating_score ELSE 0 END) / SUM(CASE WHEN T2.movie_title = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.movie_title = ? THEN T1.rating_score ELSE 0 END) / SUM(CASE WHEN T2.movie_title = ? THEN 1 ELSE 0 END) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id", (movie_title_1, movie_title_1, movie_title_2, movie_title_2))
    result = cursor.fetchone()
    if not result:
        return {"rating_difference": []}
    return {"rating_difference": result[0]}

# Endpoint to get the count of movies released in a specific year
@app.get("/v1/movie_platform/count_movies_by_release_year", operation_id="get_count_movies_by_release_year", summary="Retrieves the total number of movies released in a specified year. The year is provided as an input parameter.")
async def get_count_movies_by_release_year(movie_release_year: int = Query(..., description="Release year of the movie")):
    cursor.execute("SELECT COUNT(*) FROM movies WHERE movie_release_year = ?", (movie_release_year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies directed by a specific director
@app.get("/v1/movie_platform/count_movies_by_director", operation_id="get_count_movies_by_director", summary="Retrieves the total number of movies directed by a given director. The director's name is used to filter the movies and calculate the count.")
async def get_count_movies_by_director(director_name: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT COUNT(movie_title) FROM movies WHERE director_name = ?", (director_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most popular movie directed by a specific director
@app.get("/v1/movie_platform/most_popular_movie_by_director", operation_id="get_most_popular_movie_by_director", summary="Retrieves the most popular movie directed by a given director. The popularity of a movie is determined by its ranking in the database. The director's name is used to filter the movies and the most popular one is returned.")
async def get_most_popular_movie_by_director(director_name: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT movie_title FROM movies WHERE director_name = ? ORDER BY movie_popularity DESC LIMIT 1", (director_name,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the director ID of a movie by its title
@app.get("/v1/movie_platform/director_id_by_movie_title", operation_id="get_director_id_by_movie_title", summary="Retrieves the unique identifier of the director associated with a specific movie, based on the provided movie title.")
async def get_director_id_by_movie_title(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT director_id FROM movies WHERE movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"director_id": []}
    return {"director_id": result[0]}

# Endpoint to get the user ID of the most followed list
@app.get("/v1/movie_platform/most_followed_list_user_id", operation_id="get_most_followed_list_user_id", summary="Retrieves the user ID associated with the most followed list on the movie platform. The list with the highest number of followers is identified and its corresponding user ID is returned.")
async def get_most_followed_list_user_id():
    cursor.execute("SELECT user_id FROM lists ORDER BY list_followers DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"user_id": []}
    return {"user_id": result[0]}

# Endpoint to get the list title with the most comments
@app.get("/v1/movie_platform/most_commented_list_title", operation_id="get_most_commented_list_title", summary="Retrieves the title of the list that has received the most comments from users. This operation provides a quick way to identify the most actively discussed list on the platform.")
async def get_most_commented_list_title():
    cursor.execute("SELECT list_title FROM lists GROUP BY list_title ORDER BY COUNT(list_comments) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"list_title": []}
    return {"list_title": result[0]}

# Endpoint to get the highest rated movie title from a specific year
@app.get("/v1/movie_platform/highest_rated_movie_by_year", operation_id="get_highest_rated_movie_by_year", summary="Retrieves the title of the movie with the highest rating from a specified release year. The operation filters movies by the provided release year and sorts them by their rating scores in descending order. The title of the top-rated movie is then returned.")
async def get_highest_rated_movie_by_year(movie_release_year: int = Query(..., description="Year the movie was released")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year = ? ORDER BY T1.rating_score DESC LIMIT 1", (movie_release_year,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the top 3 movie titles by critic likes
@app.get("/v1/movie_platform/top_3_movies_by_critic_likes", operation_id="get_top_3_movies_by_critic_likes", summary="Retrieves the top three movie titles based on the highest number of critic likes. The operation fetches the data from the ratings and movies tables, joining them on the movie_id field. The results are ordered in descending order by the number of critic likes and limited to the top three.")
async def get_top_3_movies_by_critic_likes():
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.critic_likes DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the count of users in lists with more than a specified number of followers and created in a specific year
@app.get("/v1/movie_platform/user_count_by_list_followers_and_year", operation_id="get_user_count_by_list_followers_and_year", summary="Retrieves the total number of users who are part of lists with more than the specified minimum number of followers and were created in the given year. The year should be provided in 'YYYY' format.")
async def get_user_count_by_list_followers_and_year(min_followers: int = Query(..., description="Minimum number of list followers"), creation_year: str = Query(..., description="Year the list was created in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_followers > ? AND T1.list_creation_date_utc LIKE ?", (min_followers, creation_year + '%'))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users who rated a specific movie with a specific score
@app.get("/v1/movie_platform/user_count_by_rating_and_movie_title", operation_id="get_user_count_by_rating_and_movie_title", summary="Retrieves the number of users who have assigned a specific rating to a particular movie. The operation requires the rating score and the title of the movie as input parameters.")
async def get_user_count_by_rating_and_movie_title(rating_score: int = Query(..., description="Rating score"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = ? AND T2.movie_title = ?", (rating_score, movie_title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user cover image URLs from lists with a specific title
@app.get("/v1/movie_platform/user_cover_image_urls_by_list_title", operation_id="get_user_cover_image_urls_by_list_title", summary="Retrieves the cover image URLs of users associated with lists that have a specified title. The title parameter is used to filter the lists and return the corresponding user cover image URLs.")
async def get_user_cover_image_urls_by_list_title(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT T1.user_cover_image_url FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_title LIKE ?", (list_title,))
    result = cursor.fetchall()
    if not result:
        return {"user_cover_image_urls": []}
    return {"user_cover_image_urls": [row[0] for row in result]}

# Endpoint to get the total number of list followers for a specific user avatar image URL
@app.get("/v1/movie_platform/total_list_followers_by_user_avatar", operation_id="get_total_list_followers_by_user_avatar", summary="Retrieves the cumulative count of followers for all lists associated with a specific user avatar image URL. The operation calculates the total by summing the number of followers for each list linked to the provided user avatar image URL.")
async def get_total_list_followers_by_user_avatar(user_avatar_image_url: str = Query(..., description="URL of the user avatar image")):
    cursor.execute("SELECT SUM(T2.list_followers) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T1.user_avatar_image_url = ?", (user_avatar_image_url,))
    result = cursor.fetchone()
    if not result:
        return {"total_followers": []}
    return {"total_followers": result[0]}

# Endpoint to get movie titles rated with a specific score by a specific user
@app.get("/v1/movie_platform/movie_titles_by_rating_and_user", operation_id="get_movie_titles_by_rating_and_user", summary="Retrieves the titles of movies that a specific user has rated with a particular score. The operation requires the rating score and the user's unique identifier as input parameters.")
async def get_movie_titles_by_rating_and_user(rating_score: int = Query(..., description="Rating score"), user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = ? AND T1.user_id = ?", (rating_score, user_id))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get movie titles based on release year and user ID
@app.get("/v1/movie_platform/movie_titles_by_year_and_user", operation_id="get_movie_titles_by_year_and_user", summary="Retrieves a list of movie titles that were released in a specific year and rated by a particular user. The operation filters movies based on the provided release year and user ID, and returns the corresponding movie titles.")
async def get_movie_titles_by_year_and_user(movie_release_year: int = Query(..., description="Release year of the movie"), user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year = ? AND T1.user_id = ?", (movie_release_year, user_id))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the top-rated movie title
@app.get("/v1/movie_platform/top_rated_movie", operation_id="get_top_rated_movie", summary="Retrieves the title of the movie with the highest average rating score. This operation calculates the average rating score for each movie by summing all rating scores and dividing by the total number of ratings. The movie with the highest calculated average rating score is then returned.")
async def get_top_rated_movie():
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id GROUP BY T2.movie_title ORDER BY SUM(T1.rating_score) / COUNT(T1.rating_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get movie titles based on the number of critic comments
@app.get("/v1/movie_platform/movie_titles_by_critic_comments", operation_id="get_movie_titles_by_critic_comments", summary="Retrieves a list of movie titles, ordered by the number of critic comments in descending order. The number of results can be limited by specifying the 'limit' parameter.")
async def get_movie_titles_by_critic_comments(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.critic_comments DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the list title and user avatar image URL based on user ID
@app.get("/v1/movie_platform/list_title_and_avatar_by_user", operation_id="get_list_title_and_avatar_by_user", summary="Retrieves the title of the most recently created list and the avatar image URL of the user associated with the provided user ID. The list is selected based on the user's ID and ordered by its creation timestamp in descending order. Only the first matching list is returned.")
async def get_list_title_and_avatar_by_user(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T2.list_title, T1.user_avatar_image_url FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T1.user_id = ? ORDER BY T2.list_creation_timestamp_utc LIMIT 1", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"list_title": [], "user_avatar_image_url": []}
    return {"list_title": result[0], "user_avatar_image_url": result[1]}

# Endpoint to get the most rated movie title in a specific year
@app.get("/v1/movie_platform/most_rated_movie_by_year", operation_id="get_most_rated_movie_by_year", summary="Retrieves the title of the movie with the highest number of ratings in a specified year. The year is provided in the 'YYYY' format.")
async def get_most_rated_movie_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_timestamp_utc LIKE ? GROUP BY T2.movie_title ORDER BY COUNT(T2.movie_title) DESC LIMIT 1", (year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the average rating score for a specific movie title
@app.get("/v1/movie_platform/average_rating_by_movie_title", operation_id="get_average_rating_by_movie_title", summary="Retrieves the average rating score for a given movie title. The operation calculates the average rating score from the ratings table, which is joined with the movies table based on the movie_id. The movie_title parameter is used to filter the results.")
async def get_average_rating_by_movie_title(movie_title: str = Query(..., description="Movie title")):
    cursor.execute("SELECT AVG(T1.rating_score) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title LIKE ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get movie titles based on user ID and critic comments
@app.get("/v1/movie_platform/movie_titles_by_user_and_critic_comments", operation_id="get_movie_titles_by_user_and_critic_comments", summary="Retrieves a list of movie titles that a specific user has rated and that have received a certain number of critic comments. The operation filters movies based on the provided user ID and the count of critic comments associated with each movie.")
async def get_movie_titles_by_user_and_critic_comments(user_id: int = Query(..., description="User ID"), critic_comments: int = Query(..., description="Number of critic comments")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = ? AND T1.critic_comments = ?", (user_id, critic_comments))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get movie titles based on the number of critic likes
@app.get("/v1/movie_platform/movie_titles_by_critic_likes", operation_id="get_movie_titles_by_critic_likes", summary="Retrieves a list of movie titles that have received more critic likes than the specified threshold. The input parameter determines the minimum number of critic likes required for a movie to be included in the results.")
async def get_movie_titles_by_critic_likes(critic_likes: int = Query(..., description="Number of critic likes")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.critic_likes > ?", (critic_likes,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the average rating score for a specific movie title in a specific year
@app.get("/v1/movie_platform/average_rating_by_year_and_movie_title", operation_id="get_average_rating_by_year_and_movie_title", summary="Retrieves the average rating score for a specific movie title in a given year. The operation calculates the average rating by summing up all rating scores for the specified movie title and year, then dividing by the total number of ratings. The year should be provided in 'YYYY' format, and the movie title should be the exact title of the movie.")
async def get_average_rating_by_year_and_movie_title(year: str = Query(..., description="Year in 'YYYY' format"), movie_title: str = Query(..., description="Movie title")):
    cursor.execute("SELECT SUM(T1.rating_score) / COUNT(T1.rating_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_timestamp_utc LIKE ? AND T2.movie_title LIKE ?", (year + '%', movie_title))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the percentage of ratings above 3 for a specific movie title
@app.get("/v1/movie_platform/rating_percentage_above_3", operation_id="get_rating_percentage_above_3", summary="Retrieves the percentage of ratings above 3 for a specific movie. The calculation is based on the total number of ratings for the given movie title. The movie title is provided as an input parameter.")
async def get_rating_percentage_above_3(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.rating_score > 3 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.rating_score) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title LIKE ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the highest rated movie by a specific director
@app.get("/v1/movie_platform/highest_rated_movie_by_director", operation_id="get_highest_rated_movie_by_director", summary="Retrieves the movie with the highest average rating directed by a specified director. The input parameter is the director's name, which is used to filter the results and identify the top-rated movie.")
async def get_highest_rated_movie_by_director(director_name: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.director_name = ? GROUP BY T2.movie_title ORDER BY SUM(T1.rating_score) / COUNT(T1.rating_id) DESC LIMIT 1", (director_name,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the year with the most movie releases
@app.get("/v1/movie_platform/most_movie_releases_year", operation_id="get_most_movie_releases_year", summary="Retrieves the year in which the highest number of movies were released. The operation calculates the total number of movies released each year and returns the year with the maximum count.")
async def get_most_movie_releases_year():
    cursor.execute("SELECT movie_release_year FROM movies GROUP BY movie_release_year ORDER BY COUNT(movie_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the director with the most movies
@app.get("/v1/movie_platform/most_prolific_director", operation_id="get_most_prolific_director", summary="Retrieves the unique identifier of the director who has directed the most number of movies. The result is determined by counting the number of movies directed by each director and sorting them in descending order. The identifier of the director with the highest count is returned.")
async def get_most_prolific_director():
    cursor.execute("SELECT director_id FROM movies GROUP BY director_id ORDER BY COUNT(movie_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"director_id": []}
    return {"director_id": result[0]}

# Endpoint to get the number of movies by the most popular director
@app.get("/v1/movie_platform/movie_count_most_popular_director", operation_id="get_movie_count_most_popular_director", summary="Retrieves the total count of movies directed by the most popular director, based on movie popularity.")
async def get_movie_count_most_popular_director():
    cursor.execute("SELECT COUNT(movie_id) FROM movies WHERE director_id = ( SELECT director_id FROM movies ORDER BY movie_popularity DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of subscribers with a payment method after a specific date
@app.get("/v1/movie_platform/subscriber_count_with_payment_method", operation_id="get_subscriber_count_with_payment_method", summary="Retrieves the total number of subscribers who have a registered payment method and have been active since a specified date. The date should be provided in the 'YYYY%' format.")
async def get_subscriber_count_with_payment_method(rating_date_utc: str = Query(..., description="Date in 'YYYY%' format")):
    cursor.execute("SELECT COUNT(user_subscriber) FROM ratings_users WHERE user_has_payment_method = 1 AND rating_date_utc > ?", (rating_date_utc,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the user ID of the first non-subscriber who created a list
@app.get("/v1/movie_platform/first_non_subscriber_list_creator", operation_id="get_first_non_subscriber_list_creator", summary="Retrieves the unique identifier of the first user who created a list without being a subscriber, sorted by the date of list creation in UTC.")
async def get_first_non_subscriber_list_creator():
    cursor.execute("SELECT user_id FROM lists_users WHERE user_subscriber = 0 ORDER BY list_creation_date_utc LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"user_id": []}
    return {"user_id": result[0]}

# Endpoint to get the total number of list followers for the user with the most lists
@app.get("/v1/movie_platform/total_list_followers_most_lists", operation_id="get_total_list_followers_most_lists", summary="Retrieves the total count of followers for the lists owned by the user who has created the most lists. This operation does not require any input parameters and returns a single integer value representing the total number of followers.")
async def get_total_list_followers_most_lists():
    cursor.execute("SELECT SUM(T1.list_followers) FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id GROUP BY T1.user_id ORDER BY COUNT(T1.list_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"total_followers": []}
    return {"total_followers": result[0]}

# Endpoint to get the total number of list followers for lists with a specific title
@app.get("/v1/movie_platform/total_list_followers_by_title", operation_id="get_total_list_followers_by_title", summary="Retrieves the total count of followers for all lists that share a specified title. The input parameter is the title of the list.")
async def get_total_list_followers_by_title(list_title: str = Query(..., description="Title of the list")):
    cursor.execute("SELECT SUM(T2.list_followers) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_title LIKE ?", (list_title,))
    result = cursor.fetchone()
    if not result:
        return {"total_followers": []}
    return {"total_followers": result[0]}

# Endpoint to get movie titles with a specific rating score
@app.get("/v1/movie_platform/movie_titles_by_rating_score", operation_id="get_movie_titles_by_rating_score", summary="Retrieves the titles of movies that have been assigned a specific rating score. The rating score is provided as an input parameter, allowing the user to filter the results based on their desired rating.")
async def get_movie_titles_by_rating_score(rating_score: int = Query(..., description="Rating score of the movie")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = ?", (rating_score,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the most commented movie title
@app.get("/v1/movie_platform/most_commented_movie_title", operation_id="get_most_commented_movie_title", summary="Retrieves the title of the movie that has received the highest number of critic comments. This operation identifies the movie with the most critic comments by joining the ratings and movies tables, grouping by movie title, and ordering by the count of critic comments in descending order. The title of the top-ranked movie is then returned.")
async def get_most_commented_movie_title():
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id GROUP BY T2.movie_title ORDER BY COUNT(T1.critic_comments) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get user avatar image URLs based on rating timestamp
@app.get("/v1/movie_platform/user_avatar_image_urls_by_rating_timestamp", operation_id="get_user_avatar_image_urls_by_rating_timestamp", summary="Retrieves the avatar image URLs of users who have rated movies, filtered by a specific rating timestamp. The timestamp should be provided in the 'YYYY-MM-DD HH:MM:SS' format.")
async def get_user_avatar_image_urls_by_rating_timestamp(rating_timestamp_utc: str = Query(..., description="Rating timestamp in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT T2.user_avatar_image_url FROM ratings AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id WHERE T1.rating_timestamp_utc LIKE ?", (rating_timestamp_utc,))
    result = cursor.fetchall()
    if not result:
        return {"user_avatar_image_urls": []}
    return {"user_avatar_image_urls": [row[0] for row in result]}

# Endpoint to get user avatar image URLs based on list title
@app.get("/v1/movie_platform/user_avatar_image_urls_by_list_title", operation_id="get_user_avatar_image_urls_by_list_title", summary="Retrieves the avatar image URLs of users associated with a specific list title. The list title is used to filter the results.")
async def get_user_avatar_image_urls_by_list_title(list_title: str = Query(..., description="list title")):
    cursor.execute("SELECT T1.user_avatar_image_url FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_title LIKE ?", (list_title,))
    result = cursor.fetchall()
    if not result:
        return {"user_avatar_image_urls": []}
    return {"user_avatar_image_urls": [row[0] for row in result]}

# Endpoint to get user payment methods for the list with the maximum movie number
@app.get("/v1/movie_platform/user_payment_methods_max_movie_number", operation_id="get_user_payment_methods_max_movie_number", summary="Retrieves the payment methods associated with users who have the list containing the maximum number of movies. This operation does not require any input parameters and returns a list of user payment methods.")
async def get_user_payment_methods_max_movie_number():
    cursor.execute("SELECT T1.user_has_payment_method FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_movie_number = ( SELECT MAX(list_movie_number) FROM lists )")
    result = cursor.fetchall()
    if not result:
        return {"user_payment_methods": []}
    return {"user_payment_methods": [row[0] for row in result]}

# Endpoint to get user avatar image URLs based on rating score
@app.get("/v1/movie_platform/user_avatar_image_urls_by_rating_score", operation_id="get_user_avatar_image_urls_by_rating_score", summary="Retrieves the avatar image URLs of users who have given a specific rating score. The rating score is provided as an input parameter.")
async def get_user_avatar_image_urls_by_rating_score(rating_score: int = Query(..., description="Rating score")):
    cursor.execute("SELECT T2.user_avatar_image_url FROM ratings AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id WHERE T1.rating_score = ?", (rating_score,))
    result = cursor.fetchall()
    if not result:
        return {"user_avatar_image_urls": []}
    return {"user_avatar_image_urls": [row[0] for row in result]}

# Endpoint to get the count of critics for the most popular movie
@app.get("/v1/movie_platform/critic_count_most_popular_movie", operation_id="get_critic_count_most_popular_movie", summary="Retrieves the total number of critics who have rated the most popular movie. The popularity of a movie is determined by its highest rating count.")
async def get_critic_count_most_popular_movie():
    cursor.execute("SELECT COUNT(T1.critic) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_popularity = ( SELECT MAX(movie_popularity) FROM movies )")
    result = cursor.fetchone()
    if not result:
        return {"critic_count": []}
    return {"critic_count": result[0]}

# Endpoint to get user IDs based on rating score, timestamp, and movie title
@app.get("/v1/movie_platform/user_ids_by_rating_score_timestamp_title", operation_id="get_user_ids_by_rating_score_timestamp_title", summary="Retrieves a list of user IDs who have rated a specific movie with a given score at a certain time. The search is based on the provided rating score, timestamp, and movie title.")
async def get_user_ids_by_rating_score_timestamp_title(rating_score: int = Query(..., description="Rating score"), rating_timestamp_utc: str = Query(..., description="Rating timestamp in 'YYYY-MM-DD HH:MM:SS' format"), movie_title: str = Query(..., description="Movie title")):
    cursor.execute("SELECT T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE rating_score = ? AND rating_timestamp_utc LIKE ? AND T2.movie_title LIKE ?", (rating_score, rating_timestamp_utc, movie_title))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get movie URLs based on rating score and timestamp
@app.get("/v1/movie_platform/movie_urls_by_rating_score_timestamp", operation_id="get_movie_urls_by_rating_score_timestamp", summary="Retrieve a list of movie URLs that have been rated with a specific score at a given UTC timestamp. The rating score and timestamp are provided as input parameters to filter the results.")
async def get_movie_urls_by_rating_score_timestamp(rating_score: int = Query(..., description="Rating score"), rating_timestamp_utc: str = Query(..., description="Rating timestamp in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT T2.movie_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE rating_score = ? AND rating_timestamp_utc LIKE ?", (rating_score, rating_timestamp_utc))
    result = cursor.fetchall()
    if not result:
        return {"movie_urls": []}
    return {"movie_urls": [row[0] for row in result]}

# Endpoint to get the count of movie titles with a specific rating score and release year
@app.get("/v1/movie_platform/movie_title_count_by_rating_score_release_year", operation_id="get_movie_title_count_by_rating_score_release_year", summary="Retrieves the count of the most popular movie title with a specified rating score and release year. The count is determined by matching the rating score and release year in the respective tables, and then ordering the results by movie popularity in descending order. The top result is returned.")
async def get_movie_title_count_by_rating_score_release_year(rating_score: int = Query(..., description="Rating score"), movie_release_year: int = Query(..., description="Movie release year")):
    cursor.execute("SELECT COUNT(T2.movie_title) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = ? AND T2.movie_release_year = ? ORDER BY T2.movie_popularity DESC LIMIT 1", (rating_score, movie_release_year))
    result = cursor.fetchone()
    if not result:
        return {"movie_title_count": []}
    return {"movie_title_count": result[0]}

# Endpoint to get the movie title with the highest rating score for movies with popularity greater than a specified value
@app.get("/v1/movie_platform/movie_title_by_popularity_and_rating", operation_id="get_movie_title_by_popularity_and_rating", summary="Retrieves the title of the highest-rated movie that surpasses a given popularity threshold. The popularity threshold is a user-defined value that determines the minimum popularity a movie must have to be considered for the highest rating. The operation returns the title of the movie with the highest rating score among those that meet the popularity threshold.")
async def get_movie_title_by_popularity_and_rating(movie_popularity: int = Query(..., description="Movie popularity threshold")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_popularity > ? ORDER BY T1.rating_score LIMIT 1", (movie_popularity,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the count of users who rated a specific movie and have a payment method
@app.get("/v1/movie_platform/user_count_movie_payment", operation_id="get_user_count_movie_payment", summary="Retrieves the number of users who have rated a specific movie and possess a payment method. The operation filters users based on the provided movie title and payment method status.")
async def get_user_count_movie_payment(movie_title: str = Query(..., description="Title of the movie"), user_has_payment_method: int = Query(..., description="User has payment method (1 for true, 0 for false)")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN ratings_users AS T3 ON T1.user_id = T3.user_id WHERE T2.movie_title = ? AND T3.user_has_payment_method = ?", (movie_title, user_has_payment_method))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users in lists with more than a specified number of followers and have a payment method
@app.get("/v1/movie_platform/user_count_lists_followers_payment", operation_id="get_user_count_lists_followers_payment", summary="Retrieves the count of users who are part of lists with a follower count exceeding a specified threshold and have a registered payment method. This operation is useful for understanding user engagement and payment trends within the platform.")
async def get_user_count_lists_followers_payment(list_followers: int = Query(..., description="Number of list followers threshold"), user_has_payment_method: int = Query(..., description="User has payment method (1 for true, 0 for false)")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_followers > ? AND T1.user_has_payment_method = ?", (list_followers, user_has_payment_method))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of ratings for movies with a specific release year, popularity, and rating score below a threshold
@app.get("/v1/movie_platform/rating_count_by_year_popularity_score", operation_id="get_rating_count_by_year_popularity_score", summary="Retrieve the total number of ratings for the most popular movie released in a specific year, with ratings below a given score threshold.")
async def get_rating_count_by_year_popularity_score(rating_score: int = Query(..., description="Rating score threshold"), movie_release_year: int = Query(..., description="Movie release year"), movie_popularity: int = Query(..., description="Movie popularity")):
    cursor.execute("SELECT COUNT(T1.rating_score) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score < ? AND T2.movie_release_year = ? AND T2.movie_popularity = ( SELECT MAX(movie_popularity) FROM movies WHERE movie_release_year = ? )", (rating_score, movie_release_year, movie_popularity))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of users who rated a specific movie with a specific score
@app.get("/v1/movie_platform/percentage_users_rating_movie", operation_id="get_percentage_users_rating_movie", summary="Retrieves the percentage of users who have assigned a specific rating to a particular movie. The operation calculates this percentage by comparing the number of users who have given the specified rating to the total number of users who have rated the movie. The movie is identified by its title.")
async def get_percentage_users_rating_movie(rating_score: int = Query(..., description="Rating score"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.rating_score = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ?", (rating_score, movie_title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of subscribers who rated a specific movie
@app.get("/v1/movie_platform/percentage_subscribers_rating_movie", operation_id="get_percentage_subscribers_rating_movie", summary="Retrieves the percentage of subscribers who have rated a specific movie. The operation calculates this percentage by comparing the count of subscribers who have rated the movie to the total number of users who have rated the movie. The user subscriber status and movie title are required as input parameters.")
async def get_percentage_subscribers_rating_movie(user_subscriber: int = Query(..., description="User subscriber status (1 for true, 0 for false)"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.user_subscriber = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists_users AS T3 ON T1.user_id = T3.user_id WHERE T2.movie_title = ?", (user_subscriber, movie_title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of users with a payment method who rated a specific movie
@app.get("/v1/movie_platform/percentage_users_payment_method_rating_movie", operation_id="get_percentage_users_payment_method_rating_movie", summary="Retrieves the percentage of users who have a payment method and have rated a specific movie. The calculation is based on the total number of users who have rated the movie. The input parameters include a boolean value indicating whether the user has a payment method and the title of the movie.")
async def get_percentage_users_payment_method_rating_movie(user_has_payment_method: int = Query(..., description="User has payment method (1 for true, 0 for false)"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.user_has_payment_method = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists_users AS T3 ON T1.user_id = T3.user_id WHERE T2.movie_title = ?", (user_has_payment_method, movie_title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the list titles for a specific user ID
@app.get("/v1/movie_platform/list_titles_by_user_id", operation_id="get_list_titles_by_user_id", summary="Retrieves a list of titles associated with a specific user. The user is identified by a unique user ID. The list includes all titles linked to the user's account.")
async def get_list_titles_by_user_id(user_id: str = Query(..., description="User ID")):
    cursor.execute("SELECT list_title FROM lists WHERE user_id LIKE ?", (user_id,))
    result = cursor.fetchall()
    if not result:
        return {"list_titles": []}
    return {"list_titles": [row[0] for row in result]}

# Endpoint to get the most recently updated list title for a specific year
@app.get("/v1/movie_platform/recent_list_title_by_year", operation_id="get_recent_list_title_by_year", summary="Retrieves the most recent list title that was updated in the specified year. The year should be provided in 'YYYY' format.")
async def get_recent_list_title_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT list_title FROM lists WHERE strftime('%Y', list_update_timestamp_utc) = ? ORDER BY list_update_timestamp_utc DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"list_title": []}
    return {"list_title": result[0]}

# Endpoint to get the percentage of subscribers in lists_users
@app.get("/v1/movie_platform/percentage_subscribers", operation_id="get_percentage_subscribers", summary="Retrieves the percentage of users who are subscribers from the user lists. The operation calculates the ratio of subscribers to the total number of users in the lists, providing a clear overview of the subscriber base.")
async def get_percentage_subscribers(user_subscriber: int = Query(..., description="User subscriber status (1 for subscriber, 0 for non-subscriber)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN user_subscriber = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(list_id) FROM lists_users", (user_subscriber,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct list IDs for subscribers
@app.get("/v1/movie_platform/distinct_list_ids_for_subscribers", operation_id="get_distinct_list_ids_for_subscribers", summary="Retrieves a unique set of list IDs associated with subscribers. The operation filters lists based on the user's subscription status, returning only those lists that belong to subscribers.")
async def get_distinct_list_ids_for_subscribers(user_subscriber: int = Query(..., description="User subscriber status (1 for subscriber, 0 for non-subscriber)")):
    cursor.execute("SELECT DISTINCT T2.list_id FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T1.user_subscriber = ?", (user_subscriber,))
    result = cursor.fetchall()
    if not result:
        return {"list_ids": []}
    return {"list_ids": [row[0] for row in result]}

# Endpoint to get distinct list titles for users eligible for trial
@app.get("/v1/movie_platform/distinct_list_titles_for_trial_eligible", operation_id="get_distinct_list_titles_for_trial_eligible", summary="Retrieves a unique list of movie titles that are available for users who are eligible for a trial. The eligibility status is determined by the provided input parameter.")
async def get_distinct_list_titles_for_trial_eligible(user_eligible_for_trial: int = Query(..., description="User eligibility for trial (1 for eligible, 0 for not eligible)")):
    cursor.execute("SELECT DISTINCT T2.list_title FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T1.user_eligible_for_trial = ?", (user_eligible_for_trial,))
    result = cursor.fetchall()
    if not result:
        return {"list_titles": []}
    return {"list_titles": [row[0] for row in result]}

# Endpoint to get the count of list IDs for subscribers with a minimum number of followers
@app.get("/v1/movie_platform/count_list_ids_subscribers_followers", operation_id="get_count_list_ids_subscribers_followers", summary="Retrieves the total count of distinct list IDs associated with subscribers who have at least a specified number of followers. The count is determined by filtering lists based on the minimum number of followers and the subscriber status of the user.")
async def get_count_list_ids_subscribers_followers(list_followers: int = Query(..., description="Minimum number of list followers"), user_subscriber: int = Query(..., description="User subscriber status (1 for subscriber, 0 for non-subscriber)")):
    cursor.execute("SELECT COUNT(T1.list_id) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_followers >= ? AND T1.user_subscriber = ?", (list_followers, user_subscriber))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average number of list followers for lists with more than a specified number of movies
@app.get("/v1/movie_platform/average_list_followers", operation_id="get_average_list_followers", summary="Retrieves the average count of followers for movie lists that contain more than a specified number of movies. The input parameter determines the minimum number of movies a list must have to be included in the calculation.")
async def get_average_list_followers(list_movie_number: int = Query(..., description="Number of movies in the list")):
    cursor.execute("SELECT AVG(list_followers) FROM lists WHERE list_movie_number > ?", (list_movie_number,))
    result = cursor.fetchone()
    if not result:
        return {"average_followers": []}
    return {"average_followers": result[0]}

# Endpoint to get distinct list titles for subscribers with fewer than a specified number of movies
@app.get("/v1/movie_platform/distinct_list_titles_subscribers_movies", operation_id="get_distinct_list_titles_subscribers_movies", summary="Retrieves a unique list of titles for lists that belong to subscribers and contain fewer movies than the specified number. This operation is useful for identifying popular lists with limited content, which can help in curating personalized recommendations for subscribers.")
async def get_distinct_list_titles_subscribers_movies(list_movie_number: int = Query(..., description="Number of movies in the list"), user_subscriber: int = Query(..., description="User subscriber status (1 for subscriber, 0 for non-subscriber)")):
    cursor.execute("SELECT DISTINCT T2.list_title FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_movie_number < ? AND T1.user_subscriber = ?", (list_movie_number, user_subscriber))
    result = cursor.fetchall()
    if not result:
        return {"list_titles": []}
    return {"list_titles": [row[0] for row in result]}

# Endpoint to get the most recently updated list title and the time since its last update
@app.get("/v1/movie_platform/recently_updated_list", operation_id="get_recently_updated_list", summary="Retrieves the title of the most recently updated list and the time elapsed since its last update. The list is identified based on the latest update timestamp, and the time elapsed is calculated using the current local time and the list's update timestamp in UTC.")
async def get_recently_updated_list():
    cursor.execute("SELECT list_title, datetime(CURRENT_TIMESTAMP, 'localtime') - datetime(list_update_timestamp_utc) FROM lists ORDER BY list_update_timestamp_utc LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"list_title": [], "time_since_update": []}
    return {"list_title": result[0], "time_since_update": result[1]}

# Endpoint to get user IDs and subscriber status for users associated with a specific list title
@app.get("/v1/movie_platform/user_ids_subscriber_status", operation_id="get_user_ids_subscriber_status", summary="Retrieves the user IDs and subscriber statuses of users associated with a specified list title. The list title can be searched using wildcard characters for a broader match. The response includes a list of user IDs and their corresponding subscriber statuses.")
async def get_user_ids_subscriber_status(list_title: str = Query(..., description="list title to search for (use %% for wildcard)")):
    cursor.execute("SELECT T1.user_id, T1.user_subscriber FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_title LIKE ?", (list_title,))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": [], "subscriber_status": []}
    return {"user_ids": [row[0] for row in result], "subscriber_status": [row[1] for row in result]}

# Endpoint to get list titles and their age in days for lists with more than a specified number of followers
@app.get("/v1/movie_platform/list_titles_age_followers", operation_id="get_list_titles_age_followers", summary="Retrieves a list of titles along with their respective ages in days for lists that have more than the specified number of followers. The age of a list is calculated based on the difference between the current date and the list's creation timestamp.")
async def get_list_titles_age_followers(list_followers: int = Query(..., description="Number of list followers")):
    cursor.execute("SELECT list_title, 365 * (strftime('%Y', 'now') - strftime('%Y', list_creation_timestamp_utc)) + 30 * (strftime('%m', 'now') - strftime('%m', list_creation_timestamp_utc)) + strftime('%d', 'now') - strftime('%d', list_creation_timestamp_utc) FROM lists WHERE list_followers > ?", (list_followers,))
    result = cursor.fetchall()
    if not result:
        return {"list_titles": [], "age_in_days": []}
    return {"list_titles": [row[0] for row in result], "age_in_days": [row[1] for row in result]}

# Endpoint to get the percentage of ratings with missing movie IDs
@app.get("/v1/movie_platform/percentage_missing_movie_ids", operation_id="get_percentage_missing_movie_ids", summary="Retrieves the percentage of ratings that do not have a corresponding movie ID in the movies table. This operation calculates the ratio of ratings with missing movie IDs to the total number of ratings, providing insights into data integrity and potential data loss.")
async def get_percentage_missing_movie_ids():
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.movie_id IS NULL THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.movie_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get movie ratings for a specific user
@app.get("/v1/movie_platform/user_movie_ratings", operation_id="get_user_movie_ratings", summary="Retrieves a list of movies rated by a specific user, along with the corresponding rating scores and timestamps. The user is identified by the provided user_id.")
async def get_user_movie_ratings(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T2.movie_title, T1.rating_timestamp_utc, T1.rating_score FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = ?", (user_id,))
    result = cursor.fetchall()
    if not result:
        return {"ratings": []}
    return {"ratings": result}

# Endpoint to get the count of movies released between specific years and with a certain popularity
@app.get("/v1/movie_platform/movie_count_by_year_popularity", operation_id="get_movie_count_by_year_popularity", summary="Retrieves the total number of movies released between the specified start and end years, inclusive, that have a popularity rating greater than the provided minimum. This operation is useful for analyzing movie trends and popularity over time.")
async def get_movie_count_by_year_popularity(start_year: int = Query(..., description="Start year (inclusive)"), end_year: int = Query(..., description="End year (inclusive)"), min_popularity: int = Query(..., description="Minimum popularity")):
    cursor.execute("SELECT COUNT(movie_id) FROM movies WHERE movie_release_year BETWEEN ? AND ? AND movie_popularity > ?", (start_year, end_year, min_popularity))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies released in a specific year and directed by a specific director
@app.get("/v1/movie_platform/movie_count_by_year_director", operation_id="get_movie_count_by_year_director", summary="Retrieves the total number of movies released in a given year and directed by a specific director. The operation requires the release year and director's name as input parameters.")
async def get_movie_count_by_year_director(release_year: int = Query(..., description="Release year"), director_name: str = Query(..., description="Director name")):
    cursor.execute("SELECT COUNT(movie_id) FROM movies WHERE movie_release_year = ? AND director_name LIKE ?", (release_year, director_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the director URL for a specific movie title
@app.get("/v1/movie_platform/director_url_by_movie_title", operation_id="get_director_url_by_movie_title", summary="Retrieves the URL of the director associated with a given movie title. The movie title is used to search for a match in the movies database.")
async def get_director_url_by_movie_title(movie_title: str = Query(..., description="Movie title")):
    cursor.execute("SELECT director_url FROM movies WHERE movie_title LIKE ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"director_url": []}
    return {"director_url": result[0]}

# Endpoint to get the title of the most recently updated list
@app.get("/v1/movie_platform/most_recent_list_title", operation_id="get_most_recent_list_title", summary="Retrieves the title of the most recently updated list in the movie platform database. This operation identifies the list with the latest update timestamp and returns its title.")
async def get_most_recent_list_title():
    cursor.execute("SELECT list_title FROM lists WHERE list_update_timestamp_utc = ( SELECT list_update_timestamp_utc FROM lists ORDER BY list_update_timestamp_utc DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"list_title": []}
    return {"list_title": result[0]}

# Endpoint to get user IDs based on the number of list comments
@app.get("/v1/movie_platform/user_ids_by_list_comments", operation_id="get_user_ids_by_list_comments", summary="Retrieves a list of user IDs who have created lists with a specified number of comments. This operation allows you to identify users based on their list comment activity, providing insights into user engagement and content popularity.")
async def get_user_ids_by_list_comments(list_comments: int = Query(..., description="Number of list comments")):
    cursor.execute("SELECT user_id FROM lists WHERE list_comments = ?", (list_comments,))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the director with the most movies in a specific year range
@app.get("/v1/movie_platform/top_director_by_year_range", operation_id="get_top_director_by_year_range", summary="Retrieve the director with the highest number of movies within a specified year range. The response includes the director's name and the average rating of their movies. The year range is inclusive of the start and end years provided as input parameters.")
async def get_top_director_by_year_range(start_year: int = Query(..., description="Start year (inclusive)"), end_year: int = Query(..., description="End year (inclusive)")):
    cursor.execute("SELECT T2.director_name, T1.rating_score FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year BETWEEN ? AND ? GROUP BY T2.director_id ORDER BY COUNT(T2.movie_id) DESC LIMIT 1", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"director_name": [], "rating_score": []}
    return {"director_name": result[0], "rating_score": result[1]}

# Endpoint to get the count of users who rated a specific movie with specific criteria
@app.get("/v1/movie_platform/user_count_by_movie_rating_criteria", operation_id="get_user_count_by_movie_rating_criteria", summary="Retrieve the number of users who have rated a particular movie based on specific rating criteria, including the rating score, user trialist status, and the time period in which the rating was given.")
async def get_user_count_by_movie_rating_criteria(movie_title: str = Query(..., description="Movie title"), rating_score: int = Query(..., description="Rating score"), user_trialist: int = Query(..., description="User trialist status (0 or 1)"), start_timestamp: str = Query(..., description="Start timestamp in 'YYYY%' format"), end_timestamp: str = Query(..., description="End timestamp in 'YYYY%' format")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = ? AND T1.rating_score = ? AND T1.user_trialist = ? AND T1.rating_timestamp_utc BETWEEN ? AND ?", (movie_title, rating_score, user_trialist, start_timestamp, end_timestamp))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of user ratings and movie image URLs within a specific date range
@app.get("/v1/movie_platform/user_ratings_count_image_url", operation_id="get_user_ratings_count_image_url", summary="Retrieves the total number of user ratings and associated movie image URLs for a specified time period. The start and end dates, provided in 'YYYY-MM-DD HH:MM:SS' format, determine the range of ratings to be considered. The response includes the count of user ratings and the URLs of the corresponding movie images.")
async def get_user_ratings_count_image_url(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD HH:MM:SS' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT COUNT(T1.user_id), T2.movie_image_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE datetime(T1.rating_timestamp_utc) BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"ratings_count_image_url": []}
    return {"ratings_count_image_url": result}

# Endpoint to get the average list movie number and the count of top ratings for a specific user
@app.get("/v1/movie_platform/user_list_movie_number_top_ratings", operation_id="get_user_list_movie_number_top_ratings", summary="Retrieves the average number of movies listed and the total count of top-rated movies for a specific user. The user is identified by the provided user_id.")
async def get_user_list_movie_number_top_ratings(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT AVG(T3.list_movie_number), SUM(CASE WHEN T1.rating_score = 5 THEN 1 ELSE 0 END) FROM ratings AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id INNER JOIN lists AS T3 ON T2.user_id = T3.user_id WHERE T1.user_id = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"list_movie_number_top_ratings": []}
    return {"list_movie_number_top_ratings": result}

# Endpoint to get the director name, movie release year, and average rating score for trialist users
@app.get("/v1/movie_platform/director_movie_year_avg_rating", operation_id="get_director_movie_year_avg_rating", summary="Retrieves the name of the director, the release year of the movie, and the average rating score for movies watched by trialist users. The results are ordered by movie popularity in descending order and limited to a specified number of entries.")
async def get_director_movie_year_avg_rating(user_trialist: int = Query(..., description="User trialist status (1 for trialist)"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.director_name, T1.movie_release_year, SUM(T2.rating_score) / COUNT(T2.user_id) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_trialist = ? ORDER BY T1.movie_popularity DESC LIMIT ?", (user_trialist, limit))
    result = cursor.fetchone()
    if not result:
        return {"director_movie_year_avg_rating": []}
    return {"director_movie_year_avg_rating": result}

# Endpoint to get the movie title for a specific user's latest rating
@app.get("/v1/movie_platform/user_latest_movie_title", operation_id="get_user_latest_movie_title", summary="Retrieves the title of the most recent movie rated by a specific user, with an option to limit the number of results. The operation returns the movie title based on the user's rating timestamp, sorted in descending order.")
async def get_user_latest_movie_title(user_id: int = Query(..., description="User ID"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = ? ORDER BY T1.rating_timestamp_utc DESC LIMIT ?", (user_id, limit))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get movie details and average rating score ordered by rating timestamp
@app.get("/v1/movie_platform/movie_details_avg_rating", operation_id="get_movie_details_avg_rating", summary="Retrieves a list of movies along with their average rating scores, sorted by the most recent rating timestamp. The list is limited to the specified number of results. Additional details include the director's name and the movie's release year.")
async def get_movie_details_avg_rating(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.movie_id, AVG(T1.rating_score), T2.director_name, T2.movie_release_year FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.rating_timestamp_utc ASC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"movie_details_avg_rating": []}
    return {"movie_details_avg_rating": result}

api_calls = [
    "/v1/movie_platform/most_popular_movie_by_year?movie_release_year=1945",
    "/v1/movie_platform/most_popular_movie_details",
    "/v1/movie_platform/movie_with_longest_popularity_string",
    "/v1/movie_platform/most_frequent_movie_title",
    "/v1/movie_platform/average_popularity_by_director?director_name=Stanley%20Kubrick",
    "/v1/movie_platform/average_rating_by_movie?movie_title=When%20Will%20I%20Be%20Loved",
    "/v1/movie_platform/latest_rating_details_by_user?user_id=41579158",
    "/v1/movie_platform/percentage_of_subscribers",
    "/v1/movie_platform/movies_rated_by_trial_users?rating_timestamp_utc=2020-04%25",
    "/v1/movie_platform/user_ids_by_movie_and_score?movie_title=Love%20Will%20Tear%20Us%20Apart&rating_score=1",
    "/v1/movie_platform/distinct_movies_by_rating_score?rating_score=5",
    "/v1/movie_platform/movies_rated_in_year_after_month?year=2020&month=4",
    "/v1/movie_platform/movies_rated_by_critics",
    "/v1/movie_platform/percentage_rating_score_for_movie?rating_score=5&movie_title=Welcome%20to%20the%20Dollhouse",
    "/v1/movie_platform/percentage_movies_released_in_year?movie_release_year=2021",
    "/v1/movie_platform/director_by_movie_title?movie_title=Sex%2C%20Drink%20and%20Bloodshed",
    "/v1/movie_platform/most_followed_list",
    "/v1/movie_platform/recent_list_url_by_year_followers?year=2012&min_followers=1&max_followers=2",
    "/v1/movie_platform/oldest_list_id_by_user?user_id=85981819",
    "/v1/movie_platform/count_ratings_by_criteria?movie_id=1269&max_rating_score=2&user_eligible_for_trial=1&user_has_payment_method=1",
    "/v1/movie_platform/movie_details_by_year_director?movie_release_year=2021&director_name=Steven%20Spielberg",
    "/v1/movie_platform/earliest_movie_release",
    "/v1/movie_platform/subscriber_user_ids_by_year_span?year_span=10",
    "/v1/movie_platform/user_count_by_movie_rating?movie_title=Pavee%20Lackeen:%20The%20Traveller%20Girl&rating_score=4",
    "/v1/movie_platform/user_eligibility_list_followers?list_title=World%20War%202%20and%20Kids",
    "/v1/movie_platform/movie_year_user_id_by_director_rating?director_name=Quentin%20Tarantino&rating_score=4",
    "/v1/movie_platform/director_url_by_user_critic_likes?user_id=2452551&critic_likes=39",
    "/v1/movie_platform/avg_rating_director_by_movie_title?movie_title=When%20Will%20I%20Be%20Loved",
    "/v1/movie_platform/latest_list_movie_number_payment_method",
    "/v1/movie_platform/top_movie_by_critic_likes",
    "/v1/movie_platform/max_min_popularity_timestamp?start_year=1920&end_year=1929&rating_score=1&user_has_payment_method=1",
    "/v1/movie_platform/count_movies_director_popularity?director_name=Francis%20Ford%20Coppola&movie_popularity=1000",
    "/v1/movie_platform/user_avatar_image_url?user_id=1103&rating_score=5&rating_date_utc=2020-04-19",
    "/v1/movie_platform/list_followers_subscriber?user_id=4208563",
    "/v1/movie_platform/distinct_movies_highest_releases?rating_score=1",
    "/v1/movie_platform/count_user_ids_movie_criteria?movie_release_year=1924&director_name=Erich%20von%20Stroheim&rating_score=5&user_has_payment_method=1",
    "/v1/movie_platform/avg_list_movie_number_avatar?user_id=8516503",
    "/v1/movie_platform/count_user_ids_rating_url?movie_title=The%20Magnificent%20Ambersons&rating_score=2",
    "/v1/movie_platform/list_followers_date_range?start_date=2016-02-01&end_date=2016-02-29&user_eligible_for_trial=1",
    "/v1/movie_platform/rating_url_user_movie?user_id=22030372&rating_score=5&movie_title=Riff-Raff",
    "/v1/movie_platform/directors_by_year_range?start_year=1960&end_year=1985&min_movies=10",
    "/v1/movie_platform/critic_likes_by_movie_rating?user_trialist=0&rating_score=5&movie_title=Apocalypse%20Now",
    "/v1/movie_platform/avg_rating_by_director?movie_title=The%20Crowd",
    "/v1/movie_platform/earliest_movie_year_by_director?start_year=1960&end_year=1985",
    "/v1/movie_platform/max_rating_by_popularity?min_popularity=400&max_popularity=500",
    "/v1/movie_platform/rating_url_by_user_movie?user_id=45579900&movie_title=The%20Vertical%20Ray%20of%20the%20Sun&critic_likes=20",
    "/v1/movie_platform/avg_popularity_by_director?director_name=Christopher%20Nolan",
    "/v1/movie_platform/movie_titles_by_rating_time_list?start_date=2013-01-01&end_date=2013-12-31&list_title=100%20Greatest%20Living%20American%20Filmmakers",
    "/v1/movie_platform/avg_rating_and_year_by_movie?movie_title=Pavee%20Lackeen:%20The%20Traveller%20Girl",
    "/v1/movie_platform/list_count_by_year_difference?year_difference=10",
    "/v1/movie_platform/list_description_by_title?list_title=Short%20and%20pretty%20damn%20sweet",
    "/v1/movie_platform/list_url_by_title?list_title=Short%20and%20pretty%20damn%20sweet",
    "/v1/movie_platform/list_count_by_followers_and_update_date?min_followers=200&update_date=2010-01-01",
    "/v1/movie_platform/list_user_count_by_user_id_and_subscription?user_id=83373278&user_subscriber=1",
    "/v1/movie_platform/movie_release_year_by_title?movie_title=La%20Antena",
    "/v1/movie_platform/movie_url_by_title?movie_title=La%20Antena",
    "/v1/movie_platform/most_popular_movie_by_titles?movie_title_1=The%20General&movie_title_2=Il%20grido",
    "/v1/movie_platform/movie_count_by_director?director_name=Hong%20Sang-soo",
    "/v1/movie_platform/user_trialist_by_list_title?list_title=250%20Favourite%20Films",
    "/v1/movie_platform/list_titles_user_eligible_for_trial?user_id=32172230&user_eligible_for_trial=1",
    "/v1/movie_platform/count_lists_user_movies_payment_method?user_id=85981819&list_movie_number=100&user_has_payment_method=1",
    "/v1/movie_platform/list_description_user_highest_followers?user_id=85981819",
    "/v1/movie_platform/latest_update_date_list_title?list_title=250%20Favourite%20Films",
    "/v1/movie_platform/user_avatar_image_url_list_title?list_title=250%20Favourite%20Films",
    "/v1/movie_platform/count_list_ids_list_title?list_title=250%20Favourite%20Films",
    "/v1/movie_platform/count_users_movie_rating_score?movie_title=A%20Way%20of%20Life&rating_score=5",
    "/v1/movie_platform/critics_movie_title?movie_title=A%20Way%20of%20Life",
    "/v1/movie_platform/count_ratings_movie_critic_likes?movie_title=Imitation%20of%20Life&critic_likes=1",
    "/v1/movie_platform/user_ids_movie_critic_comments?movie_title=When%20Will%20I%20Be%20Loved&critic_comments=2",
    "/v1/movie_platform/rating_score_by_movie_and_user?movie_title=A%20Way%20of%20Life&user_id=39115684",
    "/v1/movie_platform/rating_url_by_movie_and_user?movie_title=A%20Way%20of%20Life&user_id=39115684",
    "/v1/movie_platform/user_trialist_by_movie_and_user?movie_title=A%20Way%20of%20Life&user_id=39115684",
    "/v1/movie_platform/count_trialist_users_by_movie?movie_title=When%20Will%20I%20Be%20Loved&user_trialist=1",
    "/v1/movie_platform/rating_url_by_movie_with_critic?movie_title=A%20Way%20of%20Life",
    "/v1/movie_platform/count_ratings_most_popular_movie",
    "/v1/movie_platform/movie_titles_by_user_critic_likes_comments?user_id=58149469&critic_likes=1&critic_comments=2",
    "/v1/movie_platform/count_trialist_users_by_movie_rating_score?movie_title=When%20Will%20I%20Be%20Loved&rating_score=1&user_trialist=1",
    "/v1/movie_platform/count_ratings_by_movie_after_date?movie_title=A%20Way%20of%20Life&rating_timestamp_utc=2012-01-01",
    "/v1/movie_platform/rating_score_by_movie_and_list_title?movie_title=Innocence%20Unprotected&list_title=250%20Favourite%20Films",
    "/v1/movie_platform/movie_titles_from_list?list_title=250%20Favourite%20Films",
    "/v1/movie_platform/average_rating_score?movie_title=A%20Way%20of%20Life",
    "/v1/movie_platform/percentage_one_star_ratings?movie_title=When%20Will%20I%20Be%20Loved",
    "/v1/movie_platform/rating_difference_between_movies?movie_title_1=Innocence%20Unprotected&movie_title_2=When%20Will%20I%20Be%20Loved",
    "/v1/movie_platform/count_movies_by_release_year?movie_release_year=2007",
    "/v1/movie_platform/count_movies_by_director?director_name=%C3%85ke%20Sandgren",
    "/v1/movie_platform/most_popular_movie_by_director?director_name=%C3%85ke%20Sandgren",
    "/v1/movie_platform/director_id_by_movie_title?movie_title=It's%20Winter",
    "/v1/movie_platform/most_followed_list_user_id",
    "/v1/movie_platform/most_commented_list_title",
    "/v1/movie_platform/highest_rated_movie_by_year?movie_release_year=2008",
    "/v1/movie_platform/top_3_movies_by_critic_likes",
    "/v1/movie_platform/user_count_by_list_followers_and_year?min_followers=100&creation_year=2009",
    "/v1/movie_platform/user_count_by_rating_and_movie_title?rating_score=5&movie_title=White%20Night%20Wedding",
    "/v1/movie_platform/user_cover_image_urls_by_list_title?list_title=Georgia%20related%20films",
    "/v1/movie_platform/total_list_followers_by_user_avatar?user_avatar_image_url=https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214",
    "/v1/movie_platform/movie_titles_by_rating_and_user?rating_score=5&user_id=94978",
    "/v1/movie_platform/movie_titles_by_year_and_user?movie_release_year=2003&user_id=2941",
    "/v1/movie_platform/top_rated_movie",
    "/v1/movie_platform/movie_titles_by_critic_comments?limit=3",
    "/v1/movie_platform/list_title_and_avatar_by_user?user_id=85981819",
    "/v1/movie_platform/most_rated_movie_by_year?year=2020",
    "/v1/movie_platform/average_rating_by_movie_title?movie_title=Versailles%20Rive-Gauche",
    "/v1/movie_platform/movie_titles_by_user_and_critic_comments?user_id=59988436&critic_comments=21",
    "/v1/movie_platform/movie_titles_by_critic_likes?critic_likes=20",
    "/v1/movie_platform/average_rating_by_year_and_movie_title?year=2019&movie_title=The%20Fall%20of%20Berlin",
    "/v1/movie_platform/rating_percentage_above_3?movie_title=Patti%20Smith:%20Dream%20of%20Life",
    "/v1/movie_platform/highest_rated_movie_by_director?director_name=Abbas%20Kiarostami",
    "/v1/movie_platform/most_movie_releases_year",
    "/v1/movie_platform/most_prolific_director",
    "/v1/movie_platform/movie_count_most_popular_director",
    "/v1/movie_platform/subscriber_count_with_payment_method?rating_date_utc=2014%25",
    "/v1/movie_platform/first_non_subscriber_list_creator",
    "/v1/movie_platform/total_list_followers_most_lists",
    "/v1/movie_platform/total_list_followers_by_title?list_title=Non-American%20Films%20about%20World%20War%20II",
    "/v1/movie_platform/movie_titles_by_rating_score?rating_score=5",
    "/v1/movie_platform/most_commented_movie_title",
    "/v1/movie_platform/user_avatar_image_urls_by_rating_timestamp?rating_timestamp_utc=2019-10-17%2001:36:36",
    "/v1/movie_platform/user_avatar_image_urls_by_list_title?list_title=Vladimir%20Vladimirovich%20Nabokov",
    "/v1/movie_platform/user_payment_methods_max_movie_number",
    "/v1/movie_platform/user_avatar_image_urls_by_rating_score?rating_score=5",
    "/v1/movie_platform/critic_count_most_popular_movie",
    "/v1/movie_platform/user_ids_by_rating_score_timestamp_title?rating_score=4&rating_timestamp_utc=2013-05-04%2006:33:32&movie_title=Freaks",
    "/v1/movie_platform/movie_urls_by_rating_score_timestamp?rating_score=5&rating_timestamp_utc=2013-05-03%2005:11:17",
    "/v1/movie_platform/movie_title_count_by_rating_score_release_year?rating_score=4&movie_release_year=1998",
    "/v1/movie_platform/movie_title_by_popularity_and_rating?movie_popularity=13000",
    "/v1/movie_platform/user_count_movie_payment?movie_title=One%20Flew%20Over%20the%20Cuckoo%27s%20Nest&user_has_payment_method=1",
    "/v1/movie_platform/user_count_lists_followers_payment?list_followers=3000&user_has_payment_method=1",
    "/v1/movie_platform/rating_count_by_year_popularity_score?rating_score=3&movie_release_year=1995&movie_popularity=1995",
    "/v1/movie_platform/percentage_users_rating_movie?rating_score=5&movie_title=Go%20Go%20Tales",
    "/v1/movie_platform/percentage_subscribers_rating_movie?user_subscriber=1&movie_title=G.I.%20Jane",
    "/v1/movie_platform/percentage_users_payment_method_rating_movie?user_has_payment_method=1&movie_title=A%20Shot%20in%20the%20Dark",
    "/v1/movie_platform/list_titles_by_user_id?user_id=4208563",
    "/v1/movie_platform/recent_list_title_by_year?year=2016",
    "/v1/movie_platform/percentage_subscribers?user_subscriber=1",
    "/v1/movie_platform/distinct_list_ids_for_subscribers?user_subscriber=1",
    "/v1/movie_platform/distinct_list_titles_for_trial_eligible?user_eligible_for_trial=1",
    "/v1/movie_platform/count_list_ids_subscribers_followers?list_followers=1&user_subscriber=1",
    "/v1/movie_platform/average_list_followers?list_movie_number=200",
    "/v1/movie_platform/distinct_list_titles_subscribers_movies?list_movie_number=50&user_subscriber=1",
    "/v1/movie_platform/recently_updated_list",
    "/v1/movie_platform/user_ids_subscriber_status?list_title=Sound%20and%20Vision",
    "/v1/movie_platform/list_titles_age_followers?list_followers=200",
    "/v1/movie_platform/percentage_missing_movie_ids",
    "/v1/movie_platform/user_movie_ratings?user_id=39115684",
    "/v1/movie_platform/movie_count_by_year_popularity?start_year=1970&end_year=1980&min_popularity=11000",
    "/v1/movie_platform/movie_count_by_year_director?release_year=1976&director_name=Felipe%20Cazals",
    "/v1/movie_platform/director_url_by_movie_title?movie_title=Red%20Blooded%20American%20Girl",
    "/v1/movie_platform/most_recent_list_title",
    "/v1/movie_platform/user_ids_by_list_comments?list_comments=142",
    "/v1/movie_platform/top_director_by_year_range?start_year=1970&end_year=1979",
    "/v1/movie_platform/user_count_by_movie_rating_criteria?movie_title=The%20Secret%20Life%20of%20Words&rating_score=3&user_trialist=0&start_timestamp=2010%25&end_timestamp=2020%25",
    "/v1/movie_platform/user_ratings_count_image_url?start_date=2017-01-01%2000:00:00&end_date=2017-12-31%2000:00:00",
    "/v1/movie_platform/user_list_movie_number_top_ratings?user_id=8516503",
    "/v1/movie_platform/director_movie_year_avg_rating?user_trialist=1&limit=1",
    "/v1/movie_platform/user_latest_movie_title?user_id=57756708&limit=1",
    "/v1/movie_platform/movie_details_avg_rating?limit=10"
]
