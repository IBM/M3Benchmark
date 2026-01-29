from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/movielens/movielens.sqlite')
cursor = conn.cursor()

# Endpoint to get distinct director IDs for movies with a rating greater than a specified value
@app.get("/v1/movielens/distinct_director_ids_by_rating", operation_id="get_distinct_director_ids", summary="Retrieve a unique set of director IDs associated with movies that have a rating higher than the provided minimum rating.")
async def get_distinct_director_ids(min_rating: float = Query(..., description="Minimum rating of the movie")):
    cursor.execute("SELECT DISTINCT T2.directorid FROM u2base AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.rating > ?", (min_rating,))
    result = cursor.fetchall()
    if not result:
        return {"director_ids": []}
    return {"director_ids": [row[0] for row in result]}

# Endpoint to get the count of users who rated a movie with a specific rating and gender
@app.get("/v1/movielens/user_count_by_rating_and_gender", operation_id="get_user_count", summary="Retrieve the number of users who have assigned a specific rating to a movie and belong to a certain gender category. The operation considers the user's gender and the rating they have given to a movie.")
async def get_user_count(rating: int = Query(..., description="Rating of the movie"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.userid) FROM u2base AS T1 INNER JOIN users AS T2 ON T1.userid = T2.userid WHERE T1.rating = ? AND T2.u_gender = ?", (rating, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the genre of movies for a specific actor
@app.get("/v1/movielens/genre_by_actor", operation_id="get_genre_by_actor", summary="Retrieves the genre(s) of movies in which a specific actor has appeared. The operation uses the provided actor ID to search for corresponding movie IDs and then identifies the genre(s) associated with those movies.")
async def get_genre_by_actor(actor_id: int = Query(..., description="ID of the actor")):
    cursor.execute("SELECT T2.genre FROM movies2actors AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid INNER JOIN actors AS T3 ON T1.actorid = T3.actorid WHERE T3.actorid = ?", (actor_id,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the count of movies from a specific country with a rating below a specified value
@app.get("/v1/movielens/movie_count_by_country_and_rating", operation_id="get_movie_count", summary="Get the count of movies from a specific country with a rating below a specified value")
async def get_movie_count(country: str = Query(..., description="Country of the movie"), max_rating: float = Query(..., description="Maximum rating of the movie")):
    cursor.execute("SELECT COUNT(T1.movieid) FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? AND T1.rating < ?", (country, max_rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies from a specific country and genre
@app.get("/v1/movielens/movie_count_by_country_and_genre", operation_id="get_movie_count_by_country_and_genre", summary="Retrieves the total number of movies from a specified country and genre. The operation uses the provided country and genre parameters to filter the movies and calculate the count.")
async def get_movie_count_by_country_and_genre(country: str = Query(..., description="Country of the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT COUNT(T1.movieid) FROM movies2directors AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? AND T1.genre = ?", (country, genre))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average occupation of users who rated movies below a specified rating
@app.get("/v1/movielens/average_occupation_by_rating", operation_id="get_average_occupation", summary="Get the average occupation of users who rated movies below a specified rating")
async def get_average_occupation(max_rating: float = Query(..., description="Maximum rating of the movie")):
    cursor.execute("SELECT AVG(T2.occupation) FROM u2base AS T1 INNER JOIN users AS T2 ON T1.userid = T2.userid WHERE T1.rating < ?", (max_rating,))
    result = cursor.fetchone()
    if not result:
        return {"average_occupation": []}
    return {"average_occupation": result[0]}

# Endpoint to get the top-rated movies from a specific country
@app.get("/v1/movielens/top_rated_movies_by_country", operation_id="get_top_rated_movies", summary="Retrieve the top-rated movies from a specified country, ranked by average user rating. The operation returns a limited number of movies based on the provided limit parameter.")
async def get_top_rated_movies(country: str = Query(..., description="Country of the movie"), limit: int = Query(..., description="Number of top-rated movies to return")):
    cursor.execute("SELECT T1.movieid FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? GROUP BY T1.movieid ORDER BY AVG(T1.rating) DESC LIMIT ?", (country, limit))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get the average number of cast members in movies from a specific country
@app.get("/v1/movielens/average_cast_num_by_country", operation_id="get_average_cast_num", summary="Retrieves the average number of cast members in movies produced in a specified country. The calculation is based on the data from the movies and movies2actors tables, considering only movies from the provided country.")
async def get_average_cast_num(country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT AVG(T2.cast_num) FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid WHERE T1.country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"average_cast_num": []}
    return {"average_cast_num": result[0]}

# Endpoint to get distinct movie IDs from a specific country and language
@app.get("/v1/movielens/distinct_movie_ids_by_country_and_language", operation_id="get_distinct_movie_ids", summary="Retrieves a limited number of unique movie IDs from a specific country and language. The operation filters movies based on the provided country and language, and returns a distinct set of movie IDs up to the specified limit.")
async def get_distinct_movie_ids(country: str = Query(..., description="Country of the movie"), is_english: str = Query(..., description="Whether the movie is in English (T/F)"), limit: int = Query(..., description="Number of distinct movie IDs to return")):
    cursor.execute("SELECT DISTINCT T1.movieid FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? AND T2.isEnglish = ? LIMIT ?", (country, is_english, limit))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get the count of actors with a specific quality and movie rating
@app.get("/v1/movielens/actor_count_by_quality_and_rating", operation_id="get_actor_count", summary="Retrieve the number of actors who possess a specified quality and have appeared in movies with a certain rating. This operation considers the quality of the actor and the rating of the movies they have been in to provide an accurate count.")
async def get_actor_count(a_quality: int = Query(..., description="Quality of the actor"), rating: int = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT COUNT(T1.actorid) FROM actors AS T1 INNER JOIN movies2actors AS T2 ON T1.actorid = T2.actorid INNER JOIN u2base AS T3 ON T2.movieid = T3.movieid WHERE T1.a_quality = ? AND T3.rating = ?", (a_quality, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average rating of movies from a specific country and year
@app.get("/v1/movielens/average_rating_by_country_year", operation_id="get_average_rating", summary="Retrieves the average rating of movies produced in a specified country and year. The operation calculates the mean rating from a dataset that combines user ratings and movie details, filtered by the provided country and year parameters.")
async def get_average_rating(country: str = Query(..., description="Country of the movie"), year: int = Query(..., description="Year of the movie")):
    cursor.execute("SELECT AVG(T1.rating) FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? AND T2.year = ?", (country, year))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the count of movies from a specific country with a running time less than a specified value and a specific rating
@app.get("/v1/movielens/count_movies_by_country_runningtime_rating", operation_id="get_count_movies", summary="Retrieves the number of movies from a specified country that have a running time less than a given duration and a particular rating. This operation is useful for analyzing movie distribution based on country, running time, and rating.")
async def get_count_movies(country: str = Query(..., description="Country of the movie"), runningtime: int = Query(..., description="Running time of the movie"), rating: int = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT COUNT(T1.movieid) FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? AND T2.runningtime < ? AND T1.rating = ?", (country, runningtime, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user IDs based on movie country, running time, rating, and user gender
@app.get("/v1/movielens/user_ids_by_movie_criteria_and_gender", operation_id="get_user_ids", summary="Retrieves a list of user IDs who have rated movies from a specific country, with a certain running time, below a given rating, and are of a particular gender.")
async def get_user_ids(country: str = Query(..., description="Country of the movie"), runningtime: int = Query(..., description="Running time of the movie"), rating: int = Query(..., description="Rating of the movie"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT T1.userid FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid INNER JOIN users AS T3 ON T1.userid = T3.userid WHERE T2.country = ? AND T2.runningtime = ? AND T1.rating < ? AND T3.u_gender = ?", (country, runningtime, rating, gender))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the count of users based on actor quality, movie rating, and user gender
@app.get("/v1/movielens/count_users_by_actor_quality_rating_gender", operation_id="get_count_users", summary="Retrieves the number of users who have rated movies featuring actors of a specified quality, with a rating above a certain threshold, and who are of a particular gender.")
async def get_count_users(actor_quality: int = Query(..., description="Quality of the actor"), rating: int = Query(..., description="Rating of the movie"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.userid) FROM u2base AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid INNER JOIN actors AS T3 ON T2.actorid = T3.actorid INNER JOIN users AS T4 ON T1.userid = T4.userid WHERE T3.a_quality = ? AND T1.rating > ? AND T4.u_gender = ?", (actor_quality, rating, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the user ID with the highest number of movies rated with a specific rating
@app.get("/v1/movielens/top_user_by_rating", operation_id="get_top_user", summary="Retrieve the user ID of the individual who has rated the most movies with a specific rating. The rating is provided as an input parameter.")
async def get_top_user(rating: int = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT userid FROM u2base WHERE rating = ? GROUP BY userid ORDER BY COUNT(movieid) DESC LIMIT 1", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"user_id": []}
    return {"user_id": result[0]}

# Endpoint to get actor IDs and genres based on director ID
@app.get("/v1/movielens/actor_genres_by_director", operation_id="get_actor_genres", summary="Retrieves a list of actor IDs and their associated genres based on the provided director ID. This operation returns data from the movies, movies2actors, actors, and movies2directors tables, where the director ID matches the input parameter.")
async def get_actor_genres(directorid: int = Query(..., description="Director ID")):
    cursor.execute("SELECT T2.actorid, T4.genre FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid INNER JOIN actors AS T3 ON T2.actorid = T3.actorid INNER JOIN movies2directors AS T4 ON T1.movieid = T4.movieid WHERE T4.directorid = ?", (directorid,))
    result = cursor.fetchall()
    if not result:
        return {"actor_genres": []}
    return {"actor_genres": [{"actorid": row[0], "genre": row[1]} for row in result]}

# Endpoint to get actor IDs and director IDs based on genre
@app.get("/v1/movielens/actor_director_by_genre", operation_id="get_actor_director", summary="Retrieves the IDs of actors and directors associated with a specific genre. The genre is provided as an input parameter, and the operation returns a list of actor and director IDs that have worked on movies within that genre.")
async def get_actor_director(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT T2.actorid, T1.directorid FROM movies2directors AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid WHERE T1.genre = ?", (genre,))
    result = cursor.fetchall()
    if not result:
        return {"actor_directors": []}
    return {"actor_directors": [{"actorid": row[0], "directorid": row[1]} for row in result]}

# Endpoint to get the count of actors based on gender and movie year
@app.get("/v1/movielens/count_actors_by_gender_year", operation_id="get_count_actors", summary="Retrieves the total number of actors who have appeared in movies of a specific year, filtered by gender. The operation requires the gender and the year as input parameters.")
async def get_count_actors(gender: str = Query(..., description="Gender of the actor"), year: int = Query(..., description="Year of the movie")):
    cursor.execute("SELECT COUNT(T2.actorid) FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid INNER JOIN actors AS T3 ON T2.actorid = T3.actorid WHERE T3.a_gender = ? AND T1.year = ?", (gender, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get actor IDs based on actor quality, movie country, and language
@app.get("/v1/movielens/actor_ids_by_quality_country_language", operation_id="get_actor_ids", summary="Retrieves a list of actor IDs who meet the specified quality criteria and have appeared in movies from the given country and language. The quality of the actor, the country of the movie, and the language of the movie are used as filters to determine the relevant actor IDs.")
async def get_actor_ids(quality: int = Query(..., description="Quality of the actor"), country: str = Query(..., description="Country of the movie"), isEnglish: str = Query(..., description="Language of the movie (T for English)")):
    cursor.execute("SELECT T2.actorid FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid INNER JOIN actors AS T3 ON T2.actorid = T3.actorid WHERE T3.a_quality = ? AND T1.country = ? AND T1.isEnglish = ?", (quality, country, isEnglish))
    result = cursor.fetchall()
    if not result:
        return {"actor_ids": []}
    return {"actor_ids": [row[0] for row in result]}

# Endpoint to get movie IDs based on running time and director's average revenue
@app.get("/v1/movielens/movie_ids_by_runningtime_revenue", operation_id="get_movie_ids", summary="Retrieves a list of movie IDs that match the specified running time and the average revenue of their respective directors. This operation allows you to filter movies based on their duration and the financial success of the directors who worked on them.")
async def get_movie_ids(runningtime: int = Query(..., description="Running time of the movie"), avg_revenue: int = Query(..., description="Average revenue of the director")):
    cursor.execute("SELECT T1.movieid FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid INNER JOIN directors AS T3 ON T2.directorid = T3.directorid WHERE T1.runningtime = ? AND T3.avg_revenue = ?", (runningtime, avg_revenue))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get genres of movies from a specific country
@app.get("/v1/movielens/genres_by_country", operation_id="get_genres_by_country", summary="Retrieves the genres of movies originating from a specified country. The operation filters movies based on the provided country and returns the corresponding genres.")
async def get_genres_by_country(country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT T2.genre FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the most popular genre among users of a specific age
@app.get("/v1/movielens/most_popular_genre_by_age", operation_id="get_most_popular_genre_by_age", summary="Retrieves the most popular genre among users of a specific age. This operation identifies the genre with the highest number of movies watched by users of the specified age. The age parameter is used to filter the user data and determine the most popular genre.")
async def get_most_popular_genre_by_age(age: int = Query(..., description="Age of the user")):
    cursor.execute("SELECT T1.genre FROM movies2directors AS T1 INNER JOIN u2base AS T2 ON T1.movieid = T2.movieid INNER JOIN users AS T3 ON T2.userid = T3.userid WHERE T3.age = ? GROUP BY T1.genre ORDER BY COUNT(T1.movieid) DESC LIMIT 1", (age,))
    result = cursor.fetchone()
    if not result:
        return {"genre": []}
    return {"genre": result[0]}

# Endpoint to get the actor ID with the most users of a specific occupation
@app.get("/v1/movielens/actor_with_most_users_by_occupation", operation_id="get_actor_with_most_users_by_occupation", summary="Retrieves the ID of the actor who has the most users with a specific occupation. This operation identifies the movie with the highest number of users in a given occupation, then returns the actor who starred in that movie. The input parameter specifies the occupation to consider.")
async def get_actor_with_most_users_by_occupation(occupation: int = Query(..., description="Occupation of the user")):
    cursor.execute("SELECT T3.actorid FROM users AS T1 INNER JOIN u2base AS T2 ON T1.userid = T2.userid INNER JOIN movies2actors AS T3 ON T2.movieid = T3.movieid WHERE T1.occupation = ? GROUP BY T2.movieid ORDER BY COUNT(T1.userid) DESC LIMIT 1", (occupation,))
    result = cursor.fetchone()
    if not result:
        return {"actorid": []}
    return {"actorid": result[0]}

# Endpoint to get distinct ages of users who gave a specific rating
@app.get("/v1/movielens/distinct_ages_by_rating", operation_id="get_distinct_ages_by_rating", summary="Retrieve the unique ages of users who have assigned a particular rating to a movie. The operation allows you to specify a rating, and it will return a list of distinct ages of users who have given that rating.")
async def get_distinct_ages_by_rating(rating: int = Query(..., description="Rating given by the user")):
    cursor.execute("SELECT DISTINCT T2.age FROM u2base AS T1 INNER JOIN users AS T2 ON T1.userid = T2.userid WHERE T1.rating = ?", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"ages": []}
    return {"ages": [row[0] for row in result]}

# Endpoint to get the country with the most movies of a specific genre
@app.get("/v1/movielens/country_with_most_movies_by_genre", operation_id="get_country_with_most_movies_by_genre", summary="Retrieves the country with the highest number of movies in a specified genre. The genre is provided as an input parameter. The operation returns the country name.")
async def get_country_with_most_movies_by_genre(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT T3.country FROM movies2directors AS T1 INNER JOIN directors AS T2 ON T1.directorid = T2.directorid INNER JOIN movies AS T3 ON T1.movieid = T3.movieid WHERE T1.genre = ? GROUP BY T3.country ORDER BY COUNT(T3.country) DESC LIMIT 1", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the top 5 movies based on average actor quality
@app.get("/v1/movielens/top_movies_by_actor_quality", operation_id="get_top_movies_by_actor_quality", summary="Retrieves the top 5 movies that have the highest average quality of actors. The ranking is determined by calculating the average quality of actors associated with each movie and then selecting the top 5 movies with the highest average quality.")
async def get_top_movies_by_actor_quality():
    cursor.execute("SELECT T2.movieid FROM actors AS T1 INNER JOIN movies2actors AS T2 ON T1.actorid = T2.actorid GROUP BY T2.actorid ORDER BY AVG(T1.a_quality) DESC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"movieids": []}
    return {"movieids": [row[0] for row in result]}

# Endpoint to get movie IDs based on country, genre, and language
@app.get("/v1/movielens/movies_by_country_genre_language", operation_id="get_movies_by_country_genre_language", summary="Retrieves a list of up to five movie IDs that match the specified country, genre, and language. The language is indicated as English (T) or non-English (F).")
async def get_movies_by_country_genre_language(country: str = Query(..., description="Country of the movie"), genre: str = Query(..., description="Genre of the movie"), isEnglish: str = Query(..., description="Language of the movie (T for English, F for non-English)")):
    cursor.execute("SELECT T1.movieid FROM movies2directors AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? AND T1.genre = ? AND T2.isEnglish = ? LIMIT 5", (country, genre, isEnglish))
    result = cursor.fetchall()
    if not result:
        return {"movieids": []}
    return {"movieids": [row[0] for row in result]}

# Endpoint to get the percentage of female users who gave a specific rating
@app.get("/v1/movielens/percentage_female_users_by_rating", operation_id="get_percentage_female_users_by_rating", summary="Retrieves the percentage of female users who have rated a specific value. This operation calculates the proportion of female users who have assigned a particular rating, providing insights into user demographics and preferences.")
async def get_percentage_female_users_by_rating(rating: int = Query(..., description="Rating given by the user")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.u_gender = 'F', 1, 0)) AS REAL) * 100 / COUNT(T2.userid) FROM u2base AS T1 INNER JOIN users AS T2 ON T1.userid = T2.userid WHERE T1.rating = ?", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in the number of female and male users for a specific genre
@app.get("/v1/movielens/gender_difference_by_genre", operation_id="get_gender_difference_by_genre", summary="Retrieve the gender disparity in user engagement for a specific movie genre. This operation calculates the difference between the number of female and male users who have engaged with movies of the specified genre. The genre is provided as an input parameter.")
async def get_gender_difference_by_genre(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT SUM(IIF(T2.u_gender = 'F', 1, 0)) - SUM(IIF(T2.u_gender = 'M', 1, 0)) FROM u2base AS T1 INNER JOIN users AS T2 ON T1.userid = T2.userid INNER JOIN movies2directors AS T3 ON T3.movieid = T1.movieid WHERE T3.genre = ?", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get genres of movies from a specific year and language
@app.get("/v1/movielens/genres_by_year_language", operation_id="get_genres_by_year_language", summary="Retrieves the genres of movies released in a specific year and language. The operation filters movies based on the provided year and language, and returns the genres associated with the matching movies. The language is represented as a boolean value, with True indicating English and False indicating non-English.")
async def get_genres_by_year_language(year: int = Query(..., description="Year of the movie"), isEnglish: str = Query(..., description="Language of the movie (T for English, F for non-English)")):
    cursor.execute("SELECT T2.genre FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.year = ? AND T1.isEnglish = ?", (year, isEnglish))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the count of movies based on country, language, and genre
@app.get("/v1/movielens/count_movies_by_country_language_genre", operation_id="get_count_movies_by_country_language_genre", summary="Retrieves the total number of movies from a specified country, language, and genre. The operation filters movies based on the provided country, language (English or non-English), and genre, then returns the count of movies that match the criteria.")
async def get_count_movies_by_country_language_genre(country: str = Query(..., description="Country of the movie"), is_english: str = Query(..., description="Whether the movie is in English (T/F)"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT COUNT(T1.movieid) FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.country = ? AND T1.isEnglish = ? AND T2.genre = ?", (country, is_english, genre))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies rated by users based on rating, age, and gender
@app.get("/v1/movielens/count_movies_by_rating_age_gender", operation_id="get_count_movies_by_rating_age_gender", summary="Retrieve the total number of movies that have been rated by users within a specific age range and gender, who have given a particular rating. This operation provides insights into movie popularity based on user demographics and rating preferences.")
async def get_count_movies_by_rating_age_gender(rating: int = Query(..., description="Rating of the movie"), age: int = Query(..., description="Age of the user"), u_gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.movieid) FROM u2base AS T1 INNER JOIN users AS T2 ON T1.userid = T2.userid WHERE T1.rating = ? AND T2.age < ? AND T2.u_gender = ?", (rating, age, u_gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the movie ID with the highest number of ratings by users of a specific gender
@app.get("/v1/movielens/top_movie_by_gender", operation_id="get_top_movie_by_gender", summary="Retrieves the movie ID that has received the most ratings from users of a specific gender. The gender is provided as an input parameter.")
async def get_top_movie_by_gender(u_gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT T1.movieid FROM u2base AS T1 INNER JOIN users AS T2 ON T1.userid = T2.userid WHERE T2.u_gender = ? GROUP BY T1.movieid ORDER BY COUNT(T2.userid) DESC LIMIT 1", (u_gender,))
    result = cursor.fetchone()
    if not result:
        return {"movieid": []}
    return {"movieid": result[0]}

# Endpoint to get the count of distinct movies from a specific country with a specific rating
@app.get("/v1/movielens/count_distinct_movies_by_country_rating", operation_id="get_count_distinct_movies_by_country_rating", summary="Retrieves the number of unique movies produced in a specified country that have received a particular rating. The operation filters movies based on the provided country and rating parameters.")
async def get_count_distinct_movies_by_country_rating(country: str = Query(..., description="Country of the movie"), rating: int = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT COUNT(DISTINCT T1.movieid) FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? AND T1.rating = ?", (country, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct movie IDs based on year and rating
@app.get("/v1/movielens/distinct_movies_by_year_rating", operation_id="get_distinct_movies_by_year_rating", summary="Retrieve a unique list of movie identifiers that match the specified year and rating. This operation filters movies based on the provided year and rating, ensuring that only distinct movie IDs are returned.")
async def get_distinct_movies_by_year_rating(year: int = Query(..., description="Year of the movie"), rating: int = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT DISTINCT T1.movieid FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.year = ? AND T1.rating = ?", (year, rating))
    result = cursor.fetchall()
    if not result:
        return {"movieids": []}
    return {"movieids": [row[0] for row in result]}

# Endpoint to get the count of distinct movies based on year, rating, and language
@app.get("/v1/movielens/count_distinct_movies_by_year_rating_language", operation_id="get_count_distinct_movies_by_year_rating_language", summary="Retrieves the count of unique movies released in a specific year, with a particular rating, and in a specified language. The language is determined by the 'is_english' parameter, where 'true' indicates English and 'false' indicates non-English.")
async def get_count_distinct_movies_by_year_rating_language(year: int = Query(..., description="Year of the movie"), rating: int = Query(..., description="Rating of the movie"), is_english: str = Query(..., description="Whether the movie is in English (T/F)")):
    cursor.execute("SELECT COUNT(DISTINCT T1.movieid) FROM movies AS T1 INNER JOIN u2base AS T2 ON T1.movieid = T2.movieid WHERE T1.year = ? AND T2.rating = ? AND T1.isEnglish = ?", (year, rating, is_english))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct users based on gender and movie country
@app.get("/v1/movielens/count_distinct_users_by_gender_country", operation_id="get_count_distinct_users_by_gender_country", summary="Retrieves the count of unique users who have watched movies from a specific country, filtered by gender. This operation considers the user's gender and the country of the movie to provide an accurate count of distinct users.")
async def get_count_distinct_users_by_gender_country(u_gender: str = Query(..., description="Gender of the user"), country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT COUNT(DISTINCT T2.userid) FROM users AS T1 INNER JOIN u2base AS T2 ON T1.userid = T2.userid INNER JOIN movies AS T3 ON T2.movieid = T3.movieid WHERE T1.u_gender = ? AND T3.country = ?", (u_gender, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct directors based on genre and quality
@app.get("/v1/movielens/count_distinct_directors_by_genre_quality", operation_id="get_count_distinct_directors_by_genre_quality", summary="Retrieves the number of unique directors who have worked on movies of a specific genre and quality. The genre and director quality are provided as input parameters.")
async def get_count_distinct_directors_by_genre_quality(genre: str = Query(..., description="Genre of the movie"), d_quality: int = Query(..., description="Quality of the director")):
    cursor.execute("SELECT COUNT(DISTINCT T2.directorid) FROM movies2directors AS T2 INNER JOIN directors AS T3 ON T2.directorid = T3.directorid WHERE T2.genre = ? AND T3.d_quality = ?", (genre, d_quality))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get genres of movies directed by directors with a specific average revenue
@app.get("/v1/movielens/genres_by_director_revenue", operation_id="get_genres_by_director_revenue", summary="Retrieves the genres of movies directed by directors who have a specified average revenue. The average revenue is provided as an input parameter.")
async def get_genres_by_director_revenue(avg_revenue: int = Query(..., description="Average revenue of the director")):
    cursor.execute("SELECT T2.genre FROM directors AS T1 INNER JOIN movies2directors AS T2 ON T1.directorid = T2.directorid WHERE T1.avg_revenue = ?", (avg_revenue,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the count of distinct actors based on language, gender, and quality
@app.get("/v1/movielens/count_distinct_actors_by_language_gender_quality", operation_id="get_count_distinct_actors_by_language_gender_quality", summary="Retrieves the count of unique actors who meet the specified language, gender, and quality criteria. The language filter determines whether the movies associated with the actors are in English or not. The gender filter narrows down the actors based on their gender. The quality filter further refines the selection based on the actor's quality rating.")
async def get_count_distinct_actors_by_language_gender_quality(is_english: str = Query(..., description="Whether the movie is in English (T/F)"), a_gender: str = Query(..., description="Gender of the actor"), a_quality: int = Query(..., description="Quality of the actor")):
    cursor.execute("SELECT COUNT(DISTINCT T1.actorid) FROM actors AS T1 INNER JOIN movies2actors AS T2 ON T1.actorid = T2.actorid INNER JOIN movies AS T3 ON T2.movieid = T3.movieid WHERE T3.isEnglish = ? AND T1.a_gender = ? AND T1.a_quality = ?", (is_english, a_gender, a_quality))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the countries of movies based on actor gender and quality
@app.get("/v1/movielens/countries_by_actor_gender_quality", operation_id="get_countries_by_actor_gender_quality", summary="Retrieves a list of countries where movies featuring actors of a specified gender and quality have been produced. The gender and quality of the actors are used as filters to determine the relevant movies and their respective countries of origin.")
async def get_countries_by_actor_gender_quality(a_gender: str = Query(..., description="Gender of the actor"), a_quality: int = Query(..., description="Quality of the actor")):
    cursor.execute("SELECT T3.country FROM actors AS T1 INNER JOIN movies2actors AS T2 ON T1.actorid = T2.actorid INNER JOIN movies AS T3 ON T2.movieid = T3.movieid WHERE T1.a_gender = ? AND T1.a_quality = ?", (a_gender, a_quality))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the highest average rating of movies based on country and genre
@app.get("/v1/movielens/highest_avg_rating_by_country_genre", operation_id="get_highest_avg_rating_by_country_genre", summary="Retrieves the highest average rating of movies from a specific country and genre. The operation calculates the average rating of each movie that matches the provided country and genre, then returns the highest average rating.")
async def get_highest_avg_rating_by_country_genre(country: str = Query(..., description="Country of the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT AVG(T2.rating) FROM movies AS T1 INNER JOIN u2base AS T2 ON T1.movieid = T2.movieid INNER JOIN movies2directors AS T3 ON T1.movieid = T3.movieid WHERE T1.country = ? AND T3.genre = ? GROUP BY T1.movieid ORDER BY AVG(T2.rating) DESC LIMIT 1", (country, genre))
    result = cursor.fetchone()
    if not result:
        return {"avg_rating": []}
    return {"avg_rating": result[0]}

# Endpoint to get the count of distinct movies directed by directors with a specific quality and average rating
@app.get("/v1/movielens/count_distinct_movies_by_director_quality_rating", operation_id="get_count_distinct_movies_by_director_quality_rating", summary="Retrieve the number of unique movies directed by directors of a specified quality, with an average rating exceeding a given threshold.")
async def get_count_distinct_movies_by_director_quality_rating(d_quality: int = Query(..., description="Quality of the director"), avg_rating: float = Query(..., description="Average rating of the movie")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT DISTINCT T2.movieid FROM directors AS T1 INNER JOIN movies2directors AS T2 ON T1.directorid = T2.directorid INNER JOIN u2base AS T3 ON T2.movieid = T3.movieid WHERE T1.d_quality = ? GROUP BY T2.movieid HAVING AVG(T3.rating) > ? ) AS T1", (d_quality, avg_rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the movie ID with the highest average rating for a specific genre
@app.get("/v1/movielens/highest_rated_movie_by_genre", operation_id="get_highest_rated_movie_by_genre", summary="Retrieves the ID of the movie with the highest average user rating within a specified genre. The genre is provided as an input parameter.")
async def get_highest_rated_movie_by_genre(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT T1.movieid FROM movies2directors AS T1 INNER JOIN u2base AS T2 ON T1.movieid = T2.movieid WHERE T1.genre = ? GROUP BY T1.movieid ORDER BY AVG(T2.rating) DESC LIMIT 1", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"movieid": []}
    return {"movieid": result[0]}

# Endpoint to get the count of users based on user ID and gender
@app.get("/v1/movielens/count_users_by_userid_gender", operation_id="get_count_users_by_userid_gender", summary="Retrieves the total number of users with a specific user ID and gender. The user ID and gender are provided as input parameters to filter the count.")
async def get_count_users_by_userid_gender(userid: int = Query(..., description="User ID"), u_gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.userid) FROM users AS T1 INNER JOIN u2base AS T2 ON T1.userid = T2.userid WHERE T2.userid = ? AND T1.u_gender = ?", (userid, u_gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct movie IDs based on running time and maximum rating
@app.get("/v1/movielens/distinct_movies_by_runningtime_max_rating", operation_id="get_distinct_movies_by_runningtime_max_rating", summary="Retrieves a list of unique movie IDs that have a specified running time and the highest user rating. The operation filters movies based on their running time and selects those with the maximum rating from the user base.")
async def get_distinct_movies_by_runningtime_max_rating(runningtime: int = Query(..., description="Running time of the movie")):
    cursor.execute("SELECT DISTINCT T1.movieid FROM movies AS T1 INNER JOIN u2base AS T2 ON T1.movieid = T2.movieid WHERE T1.runningtime = ? AND T2.rating = ( SELECT MAX(rating) FROM u2base )", (runningtime,))
    result = cursor.fetchall()
    if not result:
        return {"movieids": []}
    return {"movieids": [row[0] for row in result]}

# Endpoint to get actor IDs and quality based on movie ID
@app.get("/v1/movielens/actors_by_movieid", operation_id="get_actors_by_movieid", summary="Retrieves a list of actors associated with a specific movie, along with their respective quality ratings. The operation requires the movie's unique identifier (movie ID) as input and returns a collection of actor IDs and their corresponding quality ratings.")
async def get_actors_by_movieid(movieid: int = Query(..., description="Movie ID")):
    cursor.execute("SELECT T1.actorid, T1.a_quality FROM actors AS T1 INNER JOIN movies2actors AS T2 ON T1.actorid = T2.actorid WHERE T2.movieid = ?", (movieid,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": [{"actorid": row[0], "a_quality": row[1]} for row in result]}

# Endpoint to get the director ID with the most movies based on director quality
@app.get("/v1/movielens/top_director_by_quality", operation_id="get_top_director_by_quality", summary="Retrieves the ID of the director with the highest number of movies, filtered by a specified director quality level.")
async def get_top_director_by_quality(d_quality: int = Query(..., description="Quality of the director")):
    cursor.execute("SELECT T1.directorid FROM directors AS T1 INNER JOIN movies2directors AS T2 ON T1.directorid = T2.directorid WHERE T1.d_quality = ? GROUP BY T1.directorid ORDER BY COUNT(T2.movieid) DESC LIMIT 1", (d_quality,))
    result = cursor.fetchone()
    if not result:
        return {"directorid": []}
    return {"directorid": result[0]}

# Endpoint to get the count of distinct movies based on genre and rating
@app.get("/v1/movielens/count_distinct_movies_by_genre_rating", operation_id="get_count_distinct_movies_by_genre_rating", summary="Retrieves the number of unique movies that match a specified genre and rating. The genre and rating are provided as input parameters, allowing for a targeted count of distinct movies.")
async def get_count_distinct_movies_by_genre_rating(genre: str = Query(..., description="Genre of the movie"), rating: int = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT COUNT(DISTINCT T2.movieid) FROM u2base AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T2.genre = ? AND T1.rating = ?", (genre, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies based on rating and user age range
@app.get("/v1/movielens/count_movies_by_rating_age_range", operation_id="get_count_movies_by_rating_age_range", summary="Retrieves the total number of movies that have been rated by users within a specified age range. The rating value and the age range are provided as input parameters. This operation is useful for understanding the distribution of movie ratings across different age groups.")
async def get_count_movies_by_rating_age_range(rating: int = Query(..., description="Rating of the movie"), min_age: int = Query(..., description="Minimum age of the user"), max_age: int = Query(..., description="Maximum age of the user")):
    cursor.execute("SELECT COUNT(T1.movieid) FROM u2base AS T1 INNER JOIN users AS T2 ON T1.userid = T2.userid WHERE T1.rating = ? AND T2.age BETWEEN ? AND ?", (rating, min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get movie IDs based on rating and genre
@app.get("/v1/movielens/movie_ids_by_rating_genre", operation_id="get_movie_ids_by_rating_genre", summary="Retrieves a list of movie IDs that match a specified rating and genre. The operation filters movies based on the provided rating and genre, and returns the corresponding movie IDs.")
async def get_movie_ids_by_rating_genre(rating: int = Query(..., description="Rating of the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT T1.movieid FROM u2base AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.rating = ? AND T2.genre = ?", (rating, genre))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get director IDs based on country
@app.get("/v1/movielens/director_ids_by_country", operation_id="get_director_ids_by_country", summary="Retrieves a list of director IDs associated with movies produced in a specified country. The operation filters movies based on the provided country and returns the corresponding director IDs.")
async def get_director_ids_by_country(country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT T2.directorid FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"director_ids": []}
    return {"director_ids": [row[0] for row in result]}

# Endpoint to get user IDs and ages based on movie ID
@app.get("/v1/movielens/user_ids_ages_by_movie_id", operation_id="get_user_ids_ages_by_movie_id", summary="Retrieves a list of user IDs and their corresponding ages who have rated a specific movie. The operation requires the movie ID as input to filter the relevant users.")
async def get_user_ids_ages_by_movie_id(movieid: int = Query(..., description="Movie ID")):
    cursor.execute("SELECT T2.userid, T2.age FROM u2base AS T1 INNER JOIN users AS T2 ON T1.userid = T2.userid WHERE T1.movieid = ?", (movieid,))
    result = cursor.fetchall()
    if not result:
        return {"user_data": []}
    return {"user_data": [{"userid": row[0], "age": row[1]} for row in result]}

# Endpoint to get the most common genre for non-English movies
@app.get("/v1/movielens/most_common_genre_non_english", operation_id="get_most_common_genre_non_english", summary="Retrieves the genre that is most frequently associated with non-English movies. The operation filters movies based on their language and identifies the genre with the highest occurrence. The input parameter determines whether the movies are considered English or non-English.")
async def get_most_common_genre_non_english(is_english: str = Query(..., description="Is the movie in English (T/F)")):
    cursor.execute("SELECT T2.genre FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.isEnglish = ? GROUP BY T2.genre ORDER BY COUNT(T1.movieid) DESC LIMIT 1", (is_english,))
    result = cursor.fetchone()
    if not result:
        return {"genre": []}
    return {"genre": result[0]}

# Endpoint to get actor and director IDs based on movie ID
@app.get("/v1/movielens/actor_director_ids_by_movie_id", operation_id="get_actor_director_ids_by_movie_id", summary="Retrieves the IDs of actors and directors associated with a specific movie. The operation requires the movie's unique identifier as input and returns a list of actor and director IDs.")
async def get_actor_director_ids_by_movie_id(movieid: int = Query(..., description="Movie ID")):
    cursor.execute("SELECT T1.actorid, T2.directorid FROM movies2actors AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.movieid = ?", (movieid,))
    result = cursor.fetchall()
    if not result:
        return {"actor_director_ids": []}
    return {"actor_director_ids": [{"actorid": row[0], "directorid": row[1]} for row in result]}

# Endpoint to get the percentage of movies with actors of a certain quality in a specific country
@app.get("/v1/movielens/percentage_movies_actor_quality_country", operation_id="get_percentage_movies_actor_quality_country", summary="Retrieves the percentage of movies in a specified country that feature actors with a quality rating equal to or above a given threshold. The calculation is based on the total count of movies in the specified country.")
async def get_percentage_movies_actor_quality_country(a_quality: int = Query(..., description="Actor quality threshold"), country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.a_quality >= ?, 1, 0)) AS REAL) * 100 / COUNT(T1.movieid) FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid INNER JOIN actors AS T3 ON T2.actorid = T3.actorid WHERE T1.country = ?", (a_quality, country))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of movies where the average revenue exceeds director quality in a specific genre
@app.get("/v1/movielens/percentage_movies_revenue_exceeds_director_quality", operation_id="get_percentage_movies_revenue_exceeds_director_quality", summary="Retrieves the percentage of movies in a specified genre where the average revenue surpasses the director's quality. The genre is provided as an input parameter.")
async def get_percentage_movies_revenue_exceeds_director_quality(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.avg_revenue > T2.d_quality, 1, 0)) AS REAL) * 100 / COUNT(T1.movieid) FROM movies2directors AS T1 INNER JOIN directors AS T2 ON T1.directorid = T2.directorid WHERE T1.genre = ?", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get actor IDs based on movie year
@app.get("/v1/movielens/actor_ids_by_movie_year", operation_id="get_actor_ids_by_movie_year", summary="Retrieves a list of actor IDs who have appeared in movies released in a specified year. The year of the movie is provided as an input parameter.")
async def get_actor_ids_by_movie_year(year: int = Query(..., description="Year of the movie")):
    cursor.execute("SELECT T1.actorid FROM movies2actors AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"actor_ids": []}
    return {"actor_ids": [row[0] for row in result]}

# Endpoint to get actor IDs based on movie running time and language
@app.get("/v1/movielens/actor_ids_by_running_time_language", operation_id="get_actor_ids_by_running_time_language", summary="Retrieves the IDs of actors who have appeared in movies with a specified running time and language. The running time is provided in minutes, and the language is indicated as either English or non-English.")
async def get_actor_ids_by_running_time_language(runningtime: int = Query(..., description="Running time of the movie"), is_english: str = Query(..., description="Is the movie in English (T/F)")):
    cursor.execute("SELECT T2.actorid FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid WHERE T1.runningtime = ? AND T1.isEnglish = ?", (runningtime, is_english))
    result = cursor.fetchall()
    if not result:
        return {"actor_ids": []}
    return {"actor_ids": [row[0] for row in result]}

# Endpoint to get actor IDs based on country and minimum number of movies
@app.get("/v1/movielens/actor_ids_by_country_min_movies", operation_id="get_actor_ids_by_country_min_movies", summary="Retrieves a list of actor IDs who have acted in at least a specified number of movies from a given country. The country and the minimum number of movies are provided as input parameters.")
async def get_actor_ids_by_country_min_movies(country: str = Query(..., description="Country of the movie"), min_movies: int = Query(..., description="Minimum number of movies")):
    cursor.execute("SELECT T2.actorid FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid WHERE T1.country = ? GROUP BY T2.actorid HAVING COUNT(T1.movieid) > ?", (country, min_movies))
    result = cursor.fetchall()
    if not result:
        return {"actor_ids": []}
    return {"actor_ids": [row[0] for row in result]}

# Endpoint to get the count of actors in movies from a specific country with a cast number greater than a specified value
@app.get("/v1/movielens/count_actors_by_country_cast_num", operation_id="get_count_actors_by_country_cast_num", summary="Retrieves the total number of actors who have appeared in movies from a specified country, where their cast number in the movie exceeds a given value. This operation is useful for understanding the distribution of actors across different countries based on their cast numbers.")
async def get_count_actors_by_country_cast_num(country: str = Query(..., description="Country of the movie"), cast_num: int = Query(..., description="Cast number greater than this value")):
    cursor.execute("SELECT COUNT(T2.actorid) FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid WHERE T1.country = ? AND T2.cast_num > ?", (country, cast_num))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct movie IDs based on year and cast number
@app.get("/v1/movielens/distinct_movie_ids_by_year_cast_num", operation_id="get_distinct_movie_ids_by_year_cast_num", summary="Retrieves a unique set of movie identifiers that match the specified year and cast number. This operation filters movies based on the provided year and the number of cast members, returning only those that meet both criteria.")
async def get_distinct_movie_ids_by_year_cast_num(year: int = Query(..., description="Year of the movie"), cast_num: int = Query(..., description="Cast number")):
    cursor.execute("SELECT DISTINCT T1.movieid FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid WHERE T1.year = ? AND T2.cast_num = ?", (year, cast_num))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get the count of actors in movies from either of two specified countries
@app.get("/v1/movielens/count_actors_by_two_countries", operation_id="get_count_actors_by_two_countries", summary="Retrieves the total number of actors who have appeared in movies produced in either of the two specified countries. The input parameters represent the countries of the movies.")
async def get_count_actors_by_two_countries(country1: str = Query(..., description="First country of the movie"), country2: str = Query(..., description="Second country of the movie")):
    cursor.execute("SELECT COUNT(T1.actorid) FROM movies2actors AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? OR T2.country = ?", (country1, country2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of directors based on average revenue and genre
@app.get("/v1/movielens/count_directors_by_revenue_genre", operation_id="get_count_directors_by_revenue_genre", summary="Retrieve the number of directors who have an average revenue equal to the provided value and have directed movies in the specified genres.")
async def get_count_directors_by_revenue_genre(avg_revenue: int = Query(..., description="Average revenue of the director"), genre1: str = Query(..., description="First genre of the movie"), genre2: str = Query(..., description="Second genre of the movie")):
    cursor.execute("SELECT COUNT(T1.directorid) FROM directors AS T1 INNER JOIN movies2directors AS T2 ON T1.directorid = T2.directorid WHERE T1.avg_revenue = ? AND (T2.genre = ? OR T2.genre = ?)", (avg_revenue, genre1, genre2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get director IDs based on quality and number of movies directed
@app.get("/v1/movielens/director_ids_by_quality_movie_count", operation_id="get_director_ids_by_quality_movie_count", summary="Retrieves a list of director IDs who have directed a specified minimum number of movies and have a quality rating equal to or above a given threshold. The operation filters directors based on their quality and the number of movies they have directed, returning only those who meet the specified criteria.")
async def get_director_ids_by_quality_movie_count(d_quality: int = Query(..., description="Director quality"), movie_count: int = Query(..., description="Number of movies directed")):
    cursor.execute("SELECT T1.directorid FROM directors AS T1 INNER JOIN movies2directors AS T2 ON T1.directorid = T2.directorid WHERE T1.d_quality >= ? GROUP BY T1.directorid HAVING COUNT(T2.movieid) >= ?", (d_quality, movie_count))
    result = cursor.fetchall()
    if not result:
        return {"director_ids": []}
    return {"director_ids": [row[0] for row in result]}

# Endpoint to get the count of movies from a specific country and genre
@app.get("/v1/movielens/count_movies_by_country_genre", operation_id="get_count_movies_by_country_genre", summary="Retrieves the total number of movies from a specified country and genre. The operation uses the provided country and genre parameters to filter the movies and calculate the count.")
async def get_count_movies_by_country_genre(country: str = Query(..., description="Country of the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT COUNT(T1.movieid) FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.country = ? AND T2.genre = ?", (country, genre))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies from a specific year and genres
@app.get("/v1/movielens/count_movies_by_year_genres", operation_id="get_count_movies_by_year_genres", summary="Retrieve the total number of movies released in a specific year that belong to one or more selected genres. The operation accepts the year of release and a list of genres as input parameters, and returns the count of movies that match the provided criteria.")
async def get_count_movies_by_year_genres(year: int = Query(..., description="Year of the movie"), genre1: str = Query(..., description="First genre of the movie"), genre2: str = Query(..., description="Second genre of the movie")):
    cursor.execute("SELECT COUNT(T1.movieid) FROM movies2directors AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.year = ? AND T1.genre IN (?, ?)", (year, genre1, genre2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get movie IDs based on running time and genre
@app.get("/v1/movielens/movie_ids_by_running_time_genre", operation_id="get_movie_ids_by_running_time_genre", summary="Retrieves a list of movie IDs that meet the specified running time and genre criteria. The running time must be equal to or greater than the provided value, and the genre must match exactly. This operation does not return the movies themselves, only their unique identifiers.")
async def get_movie_ids_by_running_time_genre(runningtime: int = Query(..., description="Running time of the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT T1.movieid FROM movies2directors AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.runningtime >= ? AND T1.genre = ?", (runningtime, genre))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get the percentage of movies with a specific rating from a specific country
@app.get("/v1/movielens/percentage_movies_by_rating_country", operation_id="get_percentage_movies_by_rating_country", summary="Retrieves the percentage of movies from a specified country that have a particular rating. This operation calculates the proportion of movies with the given rating from the total number of movies produced in the specified country.")
async def get_percentage_movies_by_rating_country(rating: int = Query(..., description="Rating of the movie"), country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.rating = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.movieid) FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ?", (rating, country))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of distinct movies with a specific rating and genre
@app.get("/v1/movielens/count_distinct_movies_by_rating_genre", operation_id="get_count_distinct_movies_by_rating_genre", summary="Retrieve the number of unique movies that have a specified rating and belong to a particular genre.")
async def get_count_distinct_movies_by_rating_genre(rating: int = Query(..., description="Rating of the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT COUNT(DISTINCT T1.movieid) FROM movies2directors AS T1 INNER JOIN u2base AS T2 ON T1.movieid = T2.movieid WHERE T2.rating = ? AND T1.genre = ?", (rating, genre))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct average revenue of directors based on quality
@app.get("/v1/movielens/directors_avg_revenue", operation_id="get_directors_avg_revenue", summary="Retrieves the unique average revenue of directors who meet a specified quality standard. The quality standard is defined by the input parameter.")
async def get_directors_avg_revenue(d_quality: int = Query(..., description="Quality of the director")):
    cursor.execute("SELECT DISTINCT T1.avg_revenue FROM directors AS T1 INNER JOIN movies2directors AS T2 ON T1.directorid = T2.directorid WHERE T1.d_quality = ?", (d_quality,))
    result = cursor.fetchall()
    if not result:
        return {"avg_revenue": []}
    return {"avg_revenue": [row[0] for row in result]}

# Endpoint to get the count of movies from a specific country with the highest rating
@app.get("/v1/movielens/count_movies_highest_rating", operation_id="get_count_movies_highest_rating", summary="Retrieve the number of movies from a specified country that have the highest user rating. The operation filters movies based on the provided country and identifies those with the maximum rating, then returns the count of such movies.")
async def get_count_movies_highest_rating(country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT COUNT(movieid) FROM movies WHERE country = ? AND movieid IN ( SELECT movieid FROM u2base WHERE rating = ( SELECT MAX(rating) FROM u2base ) )", (country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most popular movie among users of a specific age
@app.get("/v1/movielens/most_popular_movie_by_age", operation_id="get_most_popular_movie_by_age", summary="Retrieves the most popular movie among users of a specific age. The popularity is determined by the number of users of the specified age who have rated the movie. The movie with the highest number of ratings from users of the given age is returned.")
async def get_most_popular_movie_by_age(age: int = Query(..., description="Age of the user")):
    cursor.execute("SELECT T2.movieid FROM users AS T1 INNER JOIN u2base AS T2 ON T1.userid = T2.userid WHERE T1.age = ? GROUP BY T2.movieid ORDER BY COUNT(T1.userid) DESC LIMIT 1", (age,))
    result = cursor.fetchone()
    if not result:
        return {"movieid": []}
    return {"movieid": result[0]}

# Endpoint to get the count of distinct users who rated movies from a specific country and age
@app.get("/v1/movielens/count_users_country_age", operation_id="get_count_users_country_age", summary="Retrieve the number of unique users who have rated movies from a specific country and age group. This operation considers the country of origin for the movies and the age of the users to provide a precise count.")
async def get_count_users_country_age(country: str = Query(..., description="Country of the movie"), age: int = Query(..., description="Age of the user")):
    cursor.execute("SELECT COUNT(DISTINCT T2.userid) FROM movies AS T1 INNER JOIN u2base AS T2 ON T1.movieid = T2.movieid INNER JOIN users AS T3 ON T2.userid = T3.userid WHERE T1.country = ? AND T3.age = ?", (country, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user details who rated a specific movie with a specific rating
@app.get("/v1/movielens/user_details_movie_rating", operation_id="get_user_details_movie_rating", summary="Retrieves the age and user ID of users who have rated a specific movie with a particular rating. The operation requires the movie ID and the rating as input parameters.")
async def get_user_details_movie_rating(movieid: str = Query(..., description="Movie ID"), rating: int = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT T1.userid, T1.age FROM users AS T1 INNER JOIN u2base AS T2 ON T1.userid = T2.userid WHERE T2.movieid = ? AND T2.rating = ?", (movieid, rating))
    result = cursor.fetchall()
    if not result:
        return {"user_details": []}
    return {"user_details": [{"userid": row[0], "age": row[1]} for row in result]}

# Endpoint to get distinct movie IDs based on rating and year
@app.get("/v1/movielens/distinct_movie_ids_rating_year", operation_id="get_distinct_movie_ids_rating_year", summary="Retrieves a unique list of movie IDs that match the specified rating and release year. The operation filters movies based on the provided rating and year, ensuring that only movies with the exact rating and year are included in the result set.")
async def get_distinct_movie_ids_rating_year(rating: int = Query(..., description="Rating of the movie"), year: int = Query(..., description="Year of the movie")):
    cursor.execute("SELECT DISTINCT T1.movieid FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T1.rating = ? AND T2.year = ?", (rating, year))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get distinct movie IDs based on country and rating
@app.get("/v1/movielens/distinct_movie_ids_country_rating", operation_id="get_distinct_movie_ids_country_rating", summary="Retrieves a unique set of movie IDs that match the specified country and rating. The operation filters movies based on the provided country and rating, ensuring that only distinct movie IDs are returned.")
async def get_distinct_movie_ids_country_rating(country: str = Query(..., description="Country of the movie"), rating: int = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT DISTINCT T1.movieid FROM u2base AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid WHERE T2.country = ? AND T1.rating = ?", (country, rating))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get the count of actors in movies from specific countries
@app.get("/v1/movielens/count_actors_movies_countries", operation_id="get_count_actors_movies_countries", summary="Retrieve the total number of actors who have appeared in movies produced in the specified countries. This operation accepts two input parameters, each representing a country of the movie. The result is a count of unique actors who have acted in movies from these countries.")
async def get_count_actors_movies_countries(country1: str = Query(..., description="First country of the movie"), country2: str = Query(..., description="Second country of the movie")):
    cursor.execute("SELECT COUNT(T2.actorid) FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid WHERE T1.country IN (?, ?)", (country1, country2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct actors in movies with a specific rating
@app.get("/v1/movielens/count_distinct_actors_rating", operation_id="get_count_distinct_actors_rating", summary="Retrieves the total number of unique actors who have appeared in movies with a specified rating. The rating parameter is used to filter the movies and calculate the count of distinct actors.")
async def get_count_distinct_actors_rating(rating: int = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT COUNT(DISTINCT T2.actorid) FROM u2base AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid WHERE T1.rating = ?", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the movie ID with the highest average rating in a specific genre
@app.get("/v1/movielens/highest_rated_movie_genre", operation_id="get_highest_rated_movie_genre", summary="Retrieves the movie ID of the film with the highest average user rating within a specified genre. The genre is provided as an input parameter.")
async def get_highest_rated_movie_genre(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT T2.movieid FROM u2base AS T2 INNER JOIN movies2directors AS T3 ON T2.movieid = T3.movieid WHERE T3.genre = ? GROUP BY T2.movieid ORDER BY AVG(T2.rating) LIMIT 1", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"movieid": []}
    return {"movieid": result[0]}

# Endpoint to get the ratio of male to female actors in movies from a specific country
@app.get("/v1/movielens/male_to_female_actor_ratio", operation_id="get_male_to_female_actor_ratio", summary="Retrieves the ratio of male to female actors in movies produced in a specified country. The operation calculates the ratio based on the number of male and female actors in the movies from the given country.")
async def get_male_to_female_actor_ratio(country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.a_gender = 'M', 1, 0)) AS REAL) / SUM(IIF(T3.a_gender = 'F', 1, 0)) FROM movies AS T1 INNER JOIN movies2actors AS T2 ON T1.movieid = T2.movieid INNER JOIN actors AS T3 ON T2.actorid = T3.actorid WHERE T1.country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of users based on movie country, running time, rating, and year
@app.get("/v1/movielens/user_count_by_country_runningtime_rating_year", operation_id="get_user_count_by_criteria", summary="Retrieves the total number of users who have rated movies from a specific country, with a particular running time, rating, and year. This operation provides insights into user preferences and engagement based on these criteria.")
async def get_user_count_by_criteria(country: str = Query(..., description="Country of the movie"), runningtime: int = Query(..., description="Running time of the movie"), rating: int = Query(..., description="Rating of the movie"), year: int = Query(..., description="Year of the movie")):
    cursor.execute("SELECT COUNT(T2.userid) FROM movies AS T1 INNER JOIN u2base AS T2 ON T1.movieid = T2.movieid WHERE T1.country = ? AND T1.runningtime = ? AND T2.rating = ? AND T1.year = ?", (country, runningtime, rating, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of actors with specific cast number and quality in a movie
@app.get("/v1/movielens/percentage_actors_by_cast_num_quality", operation_id="get_percentage_actors", summary="Retrieves the percentage of actors, filtered by a specific cast number, quality, and gender, who have appeared in a given movie.")
async def get_percentage_actors(cast_num: int = Query(..., description="Cast number of the actor"), a_quality: int = Query(..., description="Quality of the actor"), movieid: int = Query(..., description="Movie ID"), a_gender: str = Query(..., description="Gender of the actor")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.cast_num = ? AND T1.a_quality = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.actorid) FROM actors AS T1 INNER JOIN movies2actors AS T2 ON T1.actorid = T2.actorid WHERE T2.movieid = ? AND T1.a_gender = ?", (cast_num, a_quality, movieid, a_gender))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of male to female actors based on quality
@app.get("/v1/movielens/male_to_female_actor_ratio_by_quality", operation_id="get_male_to_female_actor_ratio_by_quality", summary="Retrieves the ratio of male to female actors based on the specified quality. The quality parameter is used to filter the actors and calculate the ratio accordingly.")
async def get_male_to_female_actor_ratio_by_quality(a_quality: int = Query(..., description="Quality of the actor")):
    cursor.execute("SELECT CAST(SUM(IIF(a_gender = 'M', 1, 0)) AS REAL) / SUM(IIF(a_gender = 'F', 1, 0)) FROM actors WHERE a_quality = ?", (a_quality,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the actor ID with the most movies
@app.get("/v1/movielens/top_actor_by_movie_count", operation_id="get_top_actor_by_movie_count", summary="Retrieves the ID of the actor who has appeared in the highest number of movies. The operation ranks actors based on the count of movies they have acted in and returns the top actor's ID.")
async def get_top_actor_by_movie_count():
    cursor.execute("SELECT actorid FROM movies2actors GROUP BY actorid ORDER BY COUNT(movieid) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"actor_id": []}
    return {"actor_id": result[0]}

# Endpoint to get the genre with the most movies
@app.get("/v1/movielens/top_genre_by_movie_count", operation_id="get_top_genre_by_movie_count", summary="Retrieves the genre with the highest number of associated movies. The operation returns the genre that appears most frequently in the movies2directors table, providing insights into the most popular genre based on movie count.")
async def get_top_genre_by_movie_count():
    cursor.execute("SELECT genre FROM movies2directors GROUP BY genre ORDER BY COUNT(movieid) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"genre": []}
    return {"genre": result[0]}

# Endpoint to get the top genre by movie count for directors with a specific quality
@app.get("/v1/movielens/top_genre_by_director_quality", operation_id="get_top_genre_by_director_quality", summary="Retrieves the genre with the highest number of movies directed by individuals of a specified quality. The quality of a director is determined by the provided input parameter.")
async def get_top_genre_by_director_quality(d_quality: int = Query(..., description="Quality of the director")):
    cursor.execute("SELECT T2.genre FROM directors AS T1 INNER JOIN movies2directors AS T2 ON T1.directorid = T2.directorid WHERE T1.d_quality = ? GROUP BY T2.genre ORDER BY COUNT(T2.movieid) DESC LIMIT 1", (d_quality,))
    result = cursor.fetchone()
    if not result:
        return {"genre": []}
    return {"genre": result[0]}

# Endpoint to get the top-rated movie ID for non-English movies from a specific country
@app.get("/v1/movielens/top_rated_non_english_movie", operation_id="get_top_rated_non_english_movie", summary="Retrieves the ID of the highest-rated non-English movie from a specified country. The movie's language and country are used as filtering criteria to determine the top-rated movie.")
async def get_top_rated_non_english_movie(isEnglish: str = Query(..., description="Whether the movie is in English ('F' for non-English)"), country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT T2.movieid FROM movies AS T1 INNER JOIN u2base AS T2 ON T1.movieid = T2.movieid WHERE T1.isEnglish = ? AND T1.country = ? ORDER BY T2.rating LIMIT 1", (isEnglish, country))
    result = cursor.fetchone()
    if not result:
        return {"movie_id": []}
    return {"movie_id": result[0]}

# Endpoint to get the ratio of directors with specific quality and revenue to the total number of movies
@app.get("/v1/movielens/director_quality_revenue_ratio", operation_id="get_director_quality_revenue_ratio", summary="Retrieves the proportion of movies directed by individuals with a specified quality level and average revenue. This is calculated by dividing the count of directors who meet the specified criteria by the total number of movies in the database.")
async def get_director_quality_revenue_ratio(d_quality: int = Query(..., description="Director quality"), avg_revenue: int = Query(..., description="Average revenue")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.d_quality = ? AND T1.avg_revenue = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(T2.movieid) FROM directors AS T1 INNER JOIN movies2directors AS T2 ON T1.directorid = T2.directorid", (d_quality, avg_revenue))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get movie IDs based on user gender, occupation, and rating
@app.get("/v1/movielens/movie_ids_by_user_criteria", operation_id="get_movie_ids_by_user_criteria", summary="Retrieves a list of movie IDs that match the specified user gender, occupation, and rating. The operation filters movies based on the provided user criteria and returns the corresponding movie IDs.")
async def get_movie_ids_by_user_criteria(u_gender: str = Query(..., description="User gender"), occupation: int = Query(..., description="User occupation"), rating: int = Query(..., description="Rating")):
    cursor.execute("SELECT T2.movieid FROM users AS T1 INNER JOIN u2base AS T2 ON T1.userid = T2.userid INNER JOIN movies AS T3 ON T2.movieid = T3.movieid WHERE T1.u_gender = ? AND T1.occupation = ? AND T2.rating = ?", (u_gender, occupation, rating))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get the sum of actors by gender, country, and running time for a specific movie
@app.get("/v1/movielens/actor_gender_sum_by_country_runningtime", operation_id="get_actor_gender_sum_by_country_runningtime", summary="Retrieves the total count of actors, categorized by their gender, country of origin, and the running time of a specific movie. The operation filters actors based on the provided movie ID and gender, then groups the results by country and running time.")
async def get_actor_gender_sum_by_country_runningtime(a_gender: str = Query(..., description="Actor gender"), movieid: int = Query(..., description="Movie ID")):
    cursor.execute("SELECT SUM(IIF(T1.a_gender = ?, 1, 0)) , T3.country, T3.runningtime FROM actors AS T1 INNER JOIN movies2actors AS T2 ON T1.actorid = T2.actorid INNER JOIN movies AS T3 ON T2.movieid = T3.movieid WHERE T2.movieid = ? GROUP BY T3.country, T3.runningtime", (a_gender, movieid))
    result = cursor.fetchall()
    if not result:
        return {"actor_sum": []}
    return {"actor_sum": [{"sum": row[0], "country": row[1], "runningtime": row[2]} for row in result]}

# Endpoint to get the count of movies by genre and director quality
@app.get("/v1/movielens/movie_count_by_genre_director_quality", operation_id="get_movie_count_by_genre_director_quality", summary="Retrieves the total number of movies that belong to a specific genre and are directed by individuals of a certain quality level. The genre and director quality are provided as input parameters.")
async def get_movie_count_by_genre_director_quality(genre: str = Query(..., description="Genre of the movie"), d_quality: int = Query(..., description="Director quality")):
    cursor.execute("SELECT COUNT(T1.movieid) FROM movies2directors AS T1 INNER JOIN movies AS T2 ON T1.movieid = T2.movieid INNER JOIN directors AS T3 ON T1.directorid = T3.directorid WHERE T1.genre = ? AND T3.d_quality = ?", (genre, d_quality))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get movie genres based on running time, language, and country
@app.get("/v1/movielens/movie_genres_by_criteria", operation_id="get_movie_genres_by_criteria", summary="Retrieves a list of genres for movies that meet the specified criteria. The criteria include a maximum running time, language (English or not), and country of origin. The endpoint returns the genres of movies that satisfy all the provided conditions.")
async def get_movie_genres_by_criteria(runningtime: int = Query(..., description="Running time of the movie"), isEnglish: str = Query(..., description="Language of the movie (T for English)"), country: str = Query(..., description="Country of the movie")):
    cursor.execute("SELECT T2.genre FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.runningtime <= ? AND T1.isEnglish = ? AND T1.country = ?", (runningtime, isEnglish, country))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get distinct movie IDs based on multiple criteria
@app.get("/v1/movielens/distinct_movie_ids_by_criteria", operation_id="get_distinct_movie_ids_by_criteria", summary="Retrieve a unique set of movie IDs that match the specified criteria, including country, genre, running time, rating, user age range, and language. The endpoint filters movies based on the provided parameters and returns the distinct movie IDs that meet all the criteria.")
async def get_distinct_movie_ids_by_criteria(country: str = Query(..., description="Country of the movie"), genre: str = Query(..., description="Genre of the movie"), runningtime: int = Query(..., description="Running time of the movie"), rating: int = Query(..., description="Rating of the movie"), min_age: int = Query(..., description="Minimum age of the user"), max_age: int = Query(..., description="Maximum age of the user"), isEnglish: str = Query(..., description="Language of the movie (T for English)")):
    cursor.execute("SELECT DISTINCT T1.movieid FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid INNER JOIN u2base AS T3 ON T1.movieid = T3.movieid INNER JOIN users AS T4 ON T3.userid = T4.userid WHERE T1.country = ? AND T2.genre = ? AND T1.runningtime = ? AND T3.rating = ? AND T4.age BETWEEN ? AND ? AND T1.isEnglish = ?", (country, genre, runningtime, rating, min_age, max_age, isEnglish))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get the percentage difference of English movies in a specific country and year
@app.get("/v1/movielens/english_movie_percentage_difference", operation_id="get_english_movie_percentage_difference", summary="Retrieves the percentage difference between English and non-English movies produced in a specific country and year. This operation calculates the difference by subtracting the count of non-English movies from the count of English movies, dividing the result by the total number of movies, and multiplying by 100 to obtain a percentage. The input parameters specify the language of the movies, the country of production, and the year of release.")
async def get_english_movie_percentage_difference(isEnglishTrue: str = Query(..., description="Language of the movie (T for English)"), isEnglishFalse: str = Query(..., description="Language of the movie (F for non-English)"), country: str = Query(..., description="Country of the movie"), year: int = Query(..., description="Year of the movie")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.isEnglish = ?, 1, 0)) - SUM(IIF(T1.isEnglish = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.movieid) FROM movies AS T1 INNER JOIN movies2directors AS T2 ON T1.movieid = T2.movieid WHERE T1.country = ? AND T1.year = ?", (isEnglishTrue, isEnglishFalse, country, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get actor statistics based on gender, quality, and movie ID
@app.get("/v1/movielens/actor_statistics_by_movie", operation_id="get_actor_statistics_by_movie", summary="Retrieves statistics about actors in a specific movie, categorized by gender and quality. The response includes the total number of male and female actors, the percentage of high-quality and low-quality actors, the director's ID, and the movie genre. The movie is identified by its unique ID.")
async def get_actor_statistics_by_movie(a_gender_male: str = Query(..., description="Actor gender (M for male)"), a_gender_female: str = Query(..., description="Actor gender (F for female)"), a_quality_high: int = Query(..., description="Actor quality (5 for high quality)"), a_quality_low: int = Query(..., description="Actor quality (0 for low quality)"), movieid: int = Query(..., description="Movie ID")):
    cursor.execute("SELECT SUM(IIF(a_gender = ?, 1, 0)) , SUM(IIF(a_gender = ?, 1, 0)) , CAST(SUM(IIF(a_quality = ?, 1, 0)) AS REAL) * 100 / COUNT(*) , CAST(SUM(IIF(a_quality = ?, 1, 0)) AS REAL) * 100 / COUNT(*), ( SELECT directorid FROM movies2directors WHERE movieid = ? ) , ( SELECT genre FROM movies2directors WHERE movieid = ? ) FROM actors WHERE actorid IN ( SELECT actorid FROM movies2actors WHERE movieid = ? )", (a_gender_male, a_gender_female, a_quality_high, a_quality_low, movieid, movieid, movieid))
    result = cursor.fetchone()
    if not result:
        return {"actor_statistics": []}
    return {"actor_statistics": {"sum_male": result[0], "sum_female": result[1], "percentage_high_quality": result[2], "percentage_low_quality": result[3], "directorid": result[4], "genre": result[5]}}

api_calls = [
    "/v1/movielens/distinct_director_ids_by_rating?min_rating=4",
    "/v1/movielens/user_count_by_rating_and_gender?rating=5&gender=M",
    "/v1/movielens/genre_by_actor?actor_id=851",
    "/v1/movielens/movie_count_by_country_and_rating?country=USA&max_rating=3",
    "/v1/movielens/movie_count_by_country_and_genre?country=France&genre=drama",
    "/v1/movielens/average_occupation_by_rating?max_rating=2",
    "/v1/movielens/top_rated_movies_by_country?country=USA&limit=10",
    "/v1/movielens/average_cast_num_by_country?country=USA",
    "/v1/movielens/distinct_movie_ids_by_country_and_language?country=other&is_english=F&limit=5",
    "/v1/movielens/actor_count_by_quality_and_rating?a_quality=5&rating=5",
    "/v1/movielens/average_rating_by_country_year?country=france&year=4",
    "/v1/movielens/count_movies_by_country_runningtime_rating?country=UK&runningtime=2&rating=5",
    "/v1/movielens/user_ids_by_movie_criteria_and_gender?country=France&runningtime=2&rating=3&gender=M",
    "/v1/movielens/count_users_by_actor_quality_rating_gender?actor_quality=0&rating=3&gender=F",
    "/v1/movielens/top_user_by_rating?rating=5",
    "/v1/movielens/actor_genres_by_director?directorid=22397",
    "/v1/movielens/actor_director_by_genre?genre=Action",
    "/v1/movielens/count_actors_by_gender_year?gender=F&year=4",
    "/v1/movielens/actor_ids_by_quality_country_language?quality=3&country=USA&isEnglish=T",
    "/v1/movielens/movie_ids_by_runningtime_revenue?runningtime=3&avg_revenue=1",
    "/v1/movielens/genres_by_country?country=UK",
    "/v1/movielens/most_popular_genre_by_age?age=18",
    "/v1/movielens/actor_with_most_users_by_occupation?occupation=5",
    "/v1/movielens/distinct_ages_by_rating?rating=3",
    "/v1/movielens/country_with_most_movies_by_genre?genre=Action",
    "/v1/movielens/top_movies_by_actor_quality",
    "/v1/movielens/movies_by_country_genre_language?country=UK&genre=Adventure&isEnglish=F",
    "/v1/movielens/percentage_female_users_by_rating?rating=2",
    "/v1/movielens/gender_difference_by_genre?genre=horror",
    "/v1/movielens/genres_by_year_language?year=4&isEnglish=T",
    "/v1/movielens/count_movies_by_country_language_genre?country=USA&is_english=F&genre=Action",
    "/v1/movielens/count_movies_by_rating_age_gender?rating=5&age=18&u_gender=M",
    "/v1/movielens/top_movie_by_gender?u_gender=F",
    "/v1/movielens/count_distinct_movies_by_country_rating?country=UK&rating=5",
    "/v1/movielens/distinct_movies_by_year_rating?year=4&rating=1",
    "/v1/movielens/count_distinct_movies_by_year_rating_language?year=1&rating=1&is_english=T",
    "/v1/movielens/count_distinct_users_by_gender_country?u_gender=F&country=France",
    "/v1/movielens/count_distinct_directors_by_genre_quality?genre=Action&d_quality=4",
    "/v1/movielens/genres_by_director_revenue?avg_revenue=4",
    "/v1/movielens/count_distinct_actors_by_language_gender_quality?is_english=T&a_gender=M&a_quality=5",
    "/v1/movielens/countries_by_actor_gender_quality?a_gender=F&a_quality=0",
    "/v1/movielens/highest_avg_rating_by_country_genre?country=USA&genre=Action",
    "/v1/movielens/count_distinct_movies_by_director_quality_rating?d_quality=5&avg_rating=3.5",
    "/v1/movielens/highest_rated_movie_by_genre?genre=Adventure",
    "/v1/movielens/count_users_by_userid_gender?userid=2462959&u_gender=F",
    "/v1/movielens/distinct_movies_by_runningtime_max_rating?runningtime=0",
    "/v1/movielens/actors_by_movieid?movieid=1722327",
    "/v1/movielens/top_director_by_quality?d_quality=5",
    "/v1/movielens/count_distinct_movies_by_genre_rating?genre=drama&rating=3",
    "/v1/movielens/count_movies_by_rating_age_range?rating=5&min_age=25&max_age=35",
    "/v1/movielens/movie_ids_by_rating_genre?rating=1&genre=Horror",
    "/v1/movielens/director_ids_by_country?country=France",
    "/v1/movielens/user_ids_ages_by_movie_id?movieid=1695219",
    "/v1/movielens/most_common_genre_non_english?is_english=F",
    "/v1/movielens/actor_director_ids_by_movie_id?movieid=1949144",
    "/v1/movielens/percentage_movies_actor_quality_country?a_quality=3&country=UK",
    "/v1/movielens/percentage_movies_revenue_exceeds_director_quality?genre=Action",
    "/v1/movielens/actor_ids_by_movie_year?year=4",
    "/v1/movielens/actor_ids_by_running_time_language?runningtime=2&is_english=T",
    "/v1/movielens/actor_ids_by_country_min_movies?country=France&min_movies=2",
    "/v1/movielens/count_actors_by_country_cast_num?country=USA&cast_num=1",
    "/v1/movielens/distinct_movie_ids_by_year_cast_num?year=1&cast_num=0",
    "/v1/movielens/count_actors_by_two_countries?country1=USA&country2=UK",
    "/v1/movielens/count_directors_by_revenue_genre?avg_revenue=4&genre1=Adventure&genre2=Action",
    "/v1/movielens/director_ids_by_quality_movie_count?d_quality=3&movie_count=2",
    "/v1/movielens/count_movies_by_country_genre?country=USA&genre=comedy",
    "/v1/movielens/count_movies_by_year_genres?year=4&genre1=Action&genre2=drama",
    "/v1/movielens/movie_ids_by_running_time_genre?runningtime=2&genre=Horror",
    "/v1/movielens/percentage_movies_by_rating_country?rating=1&country=USA",
    "/v1/movielens/count_distinct_movies_by_rating_genre?rating=1&genre=comedy",
    "/v1/movielens/directors_avg_revenue?d_quality=5",
    "/v1/movielens/count_movies_highest_rating?country=France",
    "/v1/movielens/most_popular_movie_by_age?age=25",
    "/v1/movielens/count_users_country_age?country=UK&age=35",
    "/v1/movielens/user_details_movie_rating?movieid=2409051&rating=2",
    "/v1/movielens/distinct_movie_ids_rating_year?rating=5&year=1",
    "/v1/movielens/distinct_movie_ids_country_rating?country=France&rating=1",
    "/v1/movielens/count_actors_movies_countries?country1=France&country2=USA",
    "/v1/movielens/count_distinct_actors_rating?rating=5",
    "/v1/movielens/highest_rated_movie_genre?genre=Crime",
    "/v1/movielens/male_to_female_actor_ratio?country=UK",
    "/v1/movielens/user_count_by_country_runningtime_rating_year?country=UK&runningtime=2&rating=1&year=2",
    "/v1/movielens/percentage_actors_by_cast_num_quality?cast_num=2&a_quality=2&movieid=1672580&a_gender=F",
    "/v1/movielens/male_to_female_actor_ratio_by_quality?a_quality=0",
    "/v1/movielens/top_actor_by_movie_count",
    "/v1/movielens/top_genre_by_movie_count",
    "/v1/movielens/top_genre_by_director_quality?d_quality=0",
    "/v1/movielens/top_rated_non_english_movie?isEnglish=F&country=USA",
    "/v1/movielens/director_quality_revenue_ratio?d_quality=4&avg_revenue=4",
    "/v1/movielens/movie_ids_by_user_criteria?u_gender=F&occupation=3&rating=5",
    "/v1/movielens/actor_gender_sum_by_country_runningtime?a_gender=F&movieid=2312852",
    "/v1/movielens/movie_count_by_genre_director_quality?genre=horror&d_quality=0",
    "/v1/movielens/movie_genres_by_criteria?runningtime=2&isEnglish=T&country=other",
    "/v1/movielens/distinct_movie_ids_by_criteria?country=UK&genre=Comedy&runningtime=3&rating=5&min_age=45&max_age=50&isEnglish=T",
    "/v1/movielens/english_movie_percentage_difference?isEnglishTrue=T&isEnglishFalse=F&country=other&year=3",
    "/v1/movielens/actor_statistics_by_movie?a_gender_male=M&a_gender_female=F&a_quality_high=5&a_quality_low=0&movieid=1684910"
]
