from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/movie/movie.sqlite')
cursor = conn.cursor()

# Endpoint to get character names from a specific movie title
@app.get("/v1/movie/character_names_by_title", operation_id="get_character_names_by_title", summary="Retrieves the names of characters appearing in a specified movie. The operation filters the character list based on the provided movie title.")
async def get_character_names_by_title(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.`Character Name` FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID WHERE T1.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"character_names": []}
    return {"character_names": [row[0] for row in result]}

# Endpoint to get the character with the most screen time from a specific movie title
@app.get("/v1/movie/character_most_screen_time", operation_id="get_character_most_screen_time", summary="Retrieves the character who has the longest screen time in a specified movie. The movie is identified by its title.")
async def get_character_most_screen_time(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.`Character Name` FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID WHERE T1.Title = ? ORDER BY T2.screentime DESC LIMIT 1", (title,))
    result = cursor.fetchone()
    if not result:
        return {"character_name": []}
    return {"character_name": result[0]}

# Endpoint to get actor names based on movie title and character name
@app.get("/v1/movie/actor_name_by_title_and_character", operation_id="get_actor_name_by_title_and_character", summary="Retrieves the name of the actor who played a specific character in a given movie. The operation requires the title of the movie and the name of the character as input parameters. It returns the name of the actor who portrayed the character in the specified movie.")
async def get_actor_name_by_title_and_character(title: str = Query(..., description="Title of the movie"), character_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT T3.Name FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ? AND T2.`Character Name` = ?", (title, character_name))
    result = cursor.fetchall()
    if not result:
        return {"actor_names": []}
    return {"actor_names": [row[0] for row in result]}

# Endpoint to get actor names based on movie title
@app.get("/v1/movie/actor_names_by_title", operation_id="get_actor_names_by_title", summary="Retrieves the names of all actors who have appeared in a movie with the specified title. The input parameter is the title of the movie.")
async def get_actor_names_by_title(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T3.Name FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"actor_names": []}
    return {"actor_names": [row[0] for row in result]}

# Endpoint to get movie titles based on character name
@app.get("/v1/movie/titles_by_character_name", operation_id="get_titles_by_character_name", summary="Retrieves a list of movie titles that feature a specific character. The character's name is provided as an input parameter, and the operation returns the corresponding movie titles.")
async def get_titles_by_character_name(character_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT T1.Title FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID WHERE T2.`Character Name` = ?", (character_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get movie titles based on actor name
@app.get("/v1/movie/titles_by_actor_name", operation_id="get_titles_by_actor_name", summary="Retrieves a list of movie titles in which a specific actor has appeared. The operation filters movies based on the provided actor's name.")
async def get_titles_by_actor_name(actor_name: str = Query(..., description="Actor name")):
    cursor.execute("SELECT T1.Title FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.Name = ?", (actor_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of movies based on actor name and MPAA rating
@app.get("/v1/movie/count_by_actor_and_rating", operation_id="get_count_by_actor_and_rating", summary="Retrieves the total number of movies associated with a specific actor and a given MPAA rating. The operation filters movies based on the provided actor's name and the desired MPAA rating, then returns the count of matching movies.")
async def get_count_by_actor_and_rating(actor_name: str = Query(..., description="Actor name"), mpaa_rating: str = Query(..., description="MPAA rating")):
    cursor.execute("SELECT COUNT(*) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.Name = ? AND T1.`MPAA Rating` = ?", (actor_name, mpaa_rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest rated movie of an actor
@app.get("/v1/movie/highest_rated_movie_by_actor", operation_id="get_highest_rated_movie_by_actor", summary="Retrieves the highest rated movie of a specified actor. The operation uses the provided actor's name to search for their highest rated movie, considering the movie's rating and the actor's involvement in the movie.")
async def get_highest_rated_movie_by_actor(actor_name: str = Query(..., description="Actor name")):
    cursor.execute("SELECT T1.Title FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.Name = ? ORDER BY T1.Rating DESC LIMIT 1", (actor_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get character names based on actor name and movie title
@app.get("/v1/movie/character_names_by_actor_and_title", operation_id="get_character_names_by_actor_and_title", summary="Retrieves the names of characters played by a specific actor in a given movie. The operation requires the actor's name and the movie title as input parameters to filter the results.")
async def get_character_names_by_actor_and_title(actor_name: str = Query(..., description="Actor name"), title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.`Character Name` FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.Name = ? AND T1.Title = ?", (actor_name, title))
    result = cursor.fetchall()
    if not result:
        return {"character_names": []}
    return {"character_names": [row[0] for row in result]}

# Endpoint to get character names based on actor name
@app.get("/v1/movie/character_names_by_actor", operation_id="get_character_names_by_actor", summary="Retrieves the names of characters portrayed by a specified actor. The operation filters the characters based on the provided actor name and returns a list of character names.")
async def get_character_names_by_actor(actor_name: str = Query(..., description="Actor name")):
    cursor.execute("SELECT T1.`Character Name` FROM characters AS T1 INNER JOIN actor AS T2 ON T1.ActorID = T2.ActorID WHERE T2.Name = ?", (actor_name,))
    result = cursor.fetchall()
    if not result:
        return {"character_names": []}
    return {"character_names": [row[0] for row in result]}

# Endpoint to get the tallest actor in a specific movie
@app.get("/v1/movie/tallest_actor_in_movie", operation_id="get_tallest_actor_in_movie", summary="Retrieves the name of the tallest actor who has a role in the specified movie. The operation filters the movie by its title and identifies the actor with the greatest height, returning their name.")
async def get_tallest_actor_in_movie(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T3.Name FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ? ORDER BY T3.`Height (Inches)` DESC LIMIT 1", (title,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of characters based on gender and ethnicity
@app.get("/v1/movie/count_characters_by_gender_ethnicity", operation_id="get_count_characters_by_gender_ethnicity", summary="Retrieves the total number of characters associated with a specific gender and ethnicity. The gender and ethnicity are provided as input parameters, allowing for a targeted count of characters.")
async def get_count_characters_by_gender_ethnicity(gender: str = Query(..., description="Gender of the actor"), ethnicity: str = Query(..., description="Ethnicity of the actor")):
    cursor.execute("SELECT COUNT(*) FROM characters AS T1 INNER JOIN actor AS T2 ON T1.ActorID = T2.ActorID WHERE T2.Gender = ? AND T2.Ethnicity = ?", (gender, ethnicity))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average rating of movies featuring a specific actor
@app.get("/v1/movie/average_rating_by_actor", operation_id="get_average_rating_by_actor", summary="Retrieves the average rating of movies in which a specified actor has appeared. The actor's name is used to identify the relevant movies and calculate the average rating.")
async def get_average_rating_by_actor(actor_name: str = Query(..., description="Name of the actor")):
    cursor.execute("SELECT AVG(T1.Rating) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.Name = ?", (actor_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the percentage difference in screentime for a specific movie
@app.get("/v1/movie/screentime_percentage_difference", operation_id="get_screentime_percentage_difference", summary="Retrieves the percentage difference between the maximum and minimum screentime of characters in a specific movie. The calculation is based on the provided movie title.")
async def get_screentime_percentage_difference(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT (MAX(CAST(SUBSTR(T2.screentime, 3, 2) AS REAL)) - MIN(CAST(SUBSTR(T2.screentime, 3, 2) AS REAL))) * 100 / MIN(CAST(SUBSTR(T2.screentime, 3, 2) AS REAL)) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the movie with the highest budget
@app.get("/v1/movie/highest_budget_movie", operation_id="get_highest_budget_movie", summary="Retrieves the title of the movie with the highest budget from the database.")
async def get_highest_budget_movie():
    cursor.execute("SELECT Title FROM movie ORDER BY Budget DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the MPAA rating of movies featuring a specific character
@app.get("/v1/movie/mpaa_rating_by_character", operation_id="get_mpaa_rating_by_character", summary="Retrieve the MPAA rating of movies in which a specified character appears. The character's name is used to identify the relevant movies.")
async def get_mpaa_rating_by_character(character_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT T1.`MPAA Rating` FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID WHERE T2.`Character Name` = ?", (character_name,))
    result = cursor.fetchone()
    if not result:
        return {"mpaa_rating": []}
    return {"mpaa_rating": result[0]}

# Endpoint to get the highest-rated character in a specific genre with a specific credit order
@app.get("/v1/movie/highest_rated_character_by_genre_credit_order", operation_id="get_highest_rated_character_by_genre_credit_order", summary="Retrieves the character with the highest movie rating in a specified genre and credit order. The character's name is returned. The genre and credit order are used to filter the results.")
async def get_highest_rated_character_by_genre_credit_order(credit_order: str = Query(..., description="Credit order of the character"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT T2.`Character Name` FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID WHERE T2.creditOrder = ? AND T1.Genre = ? ORDER BY T1.Rating DESC LIMIT 1", (credit_order, genre))
    result = cursor.fetchone()
    if not result:
        return {"character_name": []}
    return {"character_name": result[0]}

# Endpoint to get the actor with the most screentime in a specific movie
@app.get("/v1/movie/actor_most_screentime_in_movie", operation_id="get_actor_most_screentime_in_movie", summary="Retrieves the name of the actor who has the most screentime in a specified movie. The input parameter is the title of the movie.")
async def get_actor_most_screentime_in_movie(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.Name FROM characters AS T1 INNER JOIN actor AS T2 ON T1.ActorID = T2.ActorID INNER JOIN movie AS T3 ON T3.MovieID = T1.MovieID WHERE T3.Title = ? ORDER BY T1.screentime DESC LIMIT 1", (title,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of characters played by the richest actor
@app.get("/v1/movie/count_characters_by_richest_actor", operation_id="get_count_characters_by_richest_actor", summary="Retrieves the total number of characters played by the actor with the highest net worth. The operation calculates this count by joining the characters and actor tables, filtering for the actor with the maximum net worth, and then aggregating the count of characters associated with this actor.")
async def get_count_characters_by_richest_actor():
    cursor.execute("SELECT COUNT(*) FROM characters AS T1 INNER JOIN actor AS T2 ON T1.ActorID = T2.ActorID WHERE CAST(REPLACE(REPLACE(T2.NetWorth, ',', ''), '$', '') AS REAL) = ( SELECT MAX(CAST(REPLACE(REPLACE(NetWorth, ',', ''), '$', '') AS REAL)) FROM actor)")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the actor name for a specific character
@app.get("/v1/movie/actor_name_by_character", operation_id="get_actor_name_by_character", summary="Retrieves the name of the actor associated with a specific character. The operation uses the provided character name to search for a match in the characters table and returns the corresponding actor's name from the actor table.")
async def get_actor_name_by_character(character_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT T2.Name FROM characters AS T1 INNER JOIN actor AS T2 ON T1.ActorID = T2.ActorID WHERE T1.`Character Name` = ?", (character_name,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the date of birth of an actor based on character name
@app.get("/v1/movie/actor_dob_by_character_name", operation_id="get_actor_dob_by_character_name", summary="Retrieves the date of birth of the actor associated with the provided character name. The character name is used to identify the actor from the characters table, which is then linked to the actor table to obtain the date of birth.")
async def get_actor_dob_by_character_name(character_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT T2.`Date of Birth` FROM characters AS T1 INNER JOIN actor AS T2 ON T1.ActorID = T2.ActorID WHERE T1.`Character Name` = ?", (character_name,))
    result = cursor.fetchone()
    if not result:
        return {"date_of_birth": []}
    return {"date_of_birth": result[0]}

# Endpoint to get the birth city of an actor based on character name
@app.get("/v1/movie/actor_birth_city_by_character_name", operation_id="get_actor_birth_city_by_character_name", summary="Retrieves the birth city of the actor who portrayed a specific character. The character's name is required as an input parameter to identify the corresponding actor and their birth city.")
async def get_actor_birth_city_by_character_name(character_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT T2.`Birth City` FROM characters AS T1 INNER JOIN actor AS T2 ON T1.ActorID = T2.ActorID WHERE T1.`Character Name` = ?", (character_name,))
    result = cursor.fetchone()
    if not result:
        return {"birth_city": []}
    return {"birth_city": result[0]}

# Endpoint to get the biography of an actor based on character name
@app.get("/v1/movie/actor_biography_by_character_name", operation_id="get_actor_biography_by_character_name", summary="Retrieves the biography of the actor who portrayed a specific character. The character's name is used to identify the actor and fetch their biographical information.")
async def get_actor_biography_by_character_name(character_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT T2.Biography FROM characters AS T1 INNER JOIN actor AS T2 ON T1.ActorID = T2.ActorID WHERE T1.`Character Name` = ?", (character_name,))
    result = cursor.fetchone()
    if not result:
        return {"biography": []}
    return {"biography": result[0]}

# Endpoint to get the height of an actor based on character name
@app.get("/v1/movie/actor_height_by_character_name", operation_id="get_actor_height_by_character_name", summary="Retrieves the height of the actor who portrayed a specific character. The character name is used to identify the actor and retrieve their height from the database.")
async def get_actor_height_by_character_name(character_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT T2.`Height (Inches)` FROM characters AS T1 INNER JOIN actor AS T2 ON T1.ActorID = T2.ActorID WHERE T1.`Character Name` = ?", (character_name,))
    result = cursor.fetchone()
    if not result:
        return {"height": []}
    return {"height": result[0]}

# Endpoint to get the character name based on movie title and credit order
@app.get("/v1/movie/character_name_by_movie_title_and_credit_order", operation_id="get_character_name_by_movie_title_and_credit_order", summary="Retrieves the name of a character from a specific movie based on the character's credit order. The operation requires the title of the movie and the credit order of the character as input parameters. The character's name is returned as the result.")
async def get_character_name_by_movie_title_and_credit_order(movie_title: str = Query(..., description="Title of the movie"), credit_order: str = Query(..., description="Credit order of the character")):
    cursor.execute("SELECT T2.`Character Name` FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID WHERE T1.Title = ? AND T2.creditOrder = ?", (movie_title, credit_order))
    result = cursor.fetchone()
    if not result:
        return {"character_name": []}
    return {"character_name": result[0]}

# Endpoint to get the actor name based on movie title and credit order
@app.get("/v1/movie/actor_name_by_movie_title_and_credit_order", operation_id="get_actor_name_by_movie_title_and_credit_order", summary="Retrieves the name of the actor who played a character in a specific movie, based on the character's credit order. The operation requires the movie title and the character's credit order as input parameters.")
async def get_actor_name_by_movie_title_and_credit_order(movie_title: str = Query(..., description="Title of the movie"), credit_order: str = Query(..., description="Credit order of the character")):
    cursor.execute("SELECT T3.Name FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ? AND T2.creditOrder = ?", (movie_title, credit_order))
    result = cursor.fetchone()
    if not result:
        return {"actor_name": []}
    return {"actor_name": result[0]}

# Endpoint to get the actor name based on movie release date and credit order
@app.get("/v1/movie/actor_name_by_release_date_and_credit_order", operation_id="get_actor_name_by_release_date_and_credit_order", summary="Retrieves the name of the actor who has a specific credit order in a movie released on a given date. The input parameters include the movie's release date and the character's credit order.")
async def get_actor_name_by_release_date_and_credit_order(release_date: str = Query(..., description="Release date of the movie in 'YYYY-MM-DD' format"), credit_order: str = Query(..., description="Credit order of the character")):
    cursor.execute("SELECT T3.Name FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.`Release Date` = ? AND T2.creditOrder = ?", (release_date, credit_order))
    result = cursor.fetchone()
    if not result:
        return {"actor_name": []}
    return {"actor_name": result[0]}

# Endpoint to get the percentage of actors born in a specific country for a given movie title
@app.get("/v1/movie/percentage_actors_by_birth_country_and_movie_title", operation_id="get_percentage_actors_by_birth_country_and_movie_title", summary="Retrieves the percentage of actors born in a specified country who have appeared in a given movie. The calculation is based on the total number of actors in the movie.")
async def get_percentage_actors_by_birth_country_and_movie_title(birth_country: str = Query(..., description="Birth country of the actor"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.`Birth Country` = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.`Birth Country`) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ?", (birth_country, movie_title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of actors born after a specific date for a given movie title
@app.get("/v1/movie/percentage_actors_by_dob_and_movie_title", operation_id="get_percentage_actors_by_dob_and_movie_title", summary="Retrieves the percentage of actors in a specific movie who were born after a given date. The calculation is based on the total number of actors in the movie. The date of birth and movie title are required as input parameters.")
async def get_percentage_actors_by_dob_and_movie_title(dob: str = Query(..., description="Date of birth in 'YYYY-MM-DD' format"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.`Date of Birth` > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.`Date of Birth`) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ?", (dob, movie_title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get movie IDs based on rating range and budget
@app.get("/v1/movie/movie_ids_by_rating_and_budget", operation_id="get_movie_ids_by_rating_and_budget", summary="Retrieves a list of movie IDs that fall within a specified rating range and have a specific budget. The minimum and maximum rating values, along with the budget, are used to filter the results.")
async def get_movie_ids_by_rating_and_budget(min_rating: float = Query(..., description="Minimum rating"), max_rating: float = Query(..., description="Maximum rating"), budget: int = Query(..., description="Budget of the movie")):
    cursor.execute("SELECT MovieID FROM movie WHERE Rating BETWEEN ? AND ? AND Budget = ?", (min_rating, max_rating, budget))
    result = cursor.fetchall()
    if not result:
        return {"movie_ids": []}
    return {"movie_ids": [row[0] for row in result]}

# Endpoint to get the count of movies based on MPAA rating and release date
@app.get("/v1/movie/count_movies_by_rating_and_release_date", operation_id="get_count_movies_by_rating_and_release_date", summary="Retrieves the total number of movies that match a given MPAA rating and a specified release date pattern. The release date pattern should be provided in 'YYYY-MM%' format.")
async def get_count_movies_by_rating_and_release_date(mpaa_rating: str = Query(..., description="MPAA rating of the movie"), release_date_pattern: str = Query(..., description="Release date pattern in 'YYYY-MM%' format")):
    cursor.execute("SELECT COUNT(*) FROM movie WHERE `MPAA Rating` = ? AND `Release Date` LIKE ?", (mpaa_rating, release_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get actor name based on ActorID, gender, and ethnicity
@app.get("/v1/movie/actor_name_by_id_gender_ethnicity", operation_id="get_actor_name_by_id_gender_ethnicity", summary="Retrieves the name of a specific actor based on the provided ActorID, gender, and ethnicity. This operation returns the name of the actor that matches the given criteria.")
async def get_actor_name_by_id_gender_ethnicity(actor_id: int = Query(..., description="ActorID of the actor"), gender: str = Query(..., description="Gender of the actor"), ethnicity: str = Query(..., description="Ethnicity of the actor")):
    cursor.execute("SELECT Name FROM actor WHERE ActorID = ? AND Gender = ? AND Ethnicity = ?", (actor_id, gender, ethnicity))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get movie genres based on actor's birth city and movie rating
@app.get("/v1/movie/genres_by_birth_city_and_rating", operation_id="get_genres_by_birth_city_and_rating", summary="Retrieves the genres of movies that meet the specified criteria. The criteria include the birth city of the actor and a minimum movie rating. The operation returns a list of genres that match the provided parameters.")
async def get_genres_by_birth_city_and_rating(birth_city: str = Query(..., description="Birth city of the actor"), rating: float = Query(..., description="Minimum rating of the movie")):
    cursor.execute("SELECT T1.Genre FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.`Birth City` = ? AND T1.Rating > ?", (birth_city, rating))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the count of movies based on genre and actor name
@app.get("/v1/movie/count_movies_by_genre_and_actor_name", operation_id="get_count_movies_by_genre_and_actor_name", summary="Retrieves the total number of movies that belong to a specific genre and feature a particular actor. The genre and actor name are provided as input parameters.")
async def get_count_movies_by_genre_and_actor_name(genre: str = Query(..., description="Genre of the movie"), actor_name: str = Query(..., description="Name of the actor")):
    cursor.execute("SELECT COUNT(*) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Genre = ? AND T3.Name = ?", (genre, actor_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get actor height and net worth based on movie title
@app.get("/v1/movie/actor_height_networth_by_movie_title", operation_id="get_actor_height_networth_by_movie_title", summary="Retrieves the height and net worth of actors who have appeared in a specified movie. The movie is identified by its title.")
async def get_actor_height_networth_by_movie_title(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T3.`Height (Inches)`, T3.NetWorth FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ?", (movie_title,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": [{"height": row[0], "net_worth": row[1]} for row in result]}

# Endpoint to get the top genre based on MPAA rating and actor net worth
@app.get("/v1/movie/top_genre_by_mpaa_rating_and_net_worth", operation_id="get_top_genre_by_mpaa_rating_and_net_worth", summary="Retrieves the genre of the movie with the highest net worth actor for a given MPAA rating. The genre is determined by considering all movies with the specified MPAA rating and selecting the one with the actor who has the highest net worth. The net worth is calculated by removing commas and dollar signs from the actor's net worth value and converting it to a real number.")
async def get_top_genre_by_mpaa_rating_and_net_worth(mpaa_rating: str = Query(..., description="MPAA rating of the movie")):
    cursor.execute("SELECT T1.Genre FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.`MPAA Rating` = ? ORDER BY CAST(REPLACE(REPLACE(T3.NetWorth, ',', ''), '$', '') AS REAL) DESC LIMIT 1", (mpaa_rating,))
    result = cursor.fetchone()
    if not result:
        return {"genre": []}
    return {"genre": result[0]}

# Endpoint to get actor net worth based on movie title, height range, and gender
@app.get("/v1/movie/actor_networth_by_movie_title_height_gender", operation_id="get_actor_networth_by_movie_title_height_gender", summary="Retrieves the net worth of actors who have appeared in a specified movie, within a given height range, and of a particular gender. The response includes the net worth of each actor who meets the provided criteria.")
async def get_actor_networth_by_movie_title_height_gender(movie_title: str = Query(..., description="Title of the movie"), min_height: int = Query(..., description="Minimum height in inches"), max_height: int = Query(..., description="Maximum height in inches"), gender: str = Query(..., description="Gender of the actor")):
    cursor.execute("SELECT T3.NetWorth FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ? AND T3.`Height (Inches)` BETWEEN ? AND ? AND T3.Gender = ?", (movie_title, min_height, max_height, gender))
    result = cursor.fetchall()
    if not result:
        return {"net_worths": []}
    return {"net_worths": [row[0] for row in result]}

# Endpoint to get the count of movies based on title, gender, and birth country
@app.get("/v1/movie/count_movies_by_title_gender_birth_country", operation_id="get_count_movies_by_title_gender_birth_country", summary="Retrieves the total number of movies that meet the specified criteria, including a particular title, actor gender, and actor birth country. This operation provides a count of movies that match the given parameters, offering insights into the distribution of movies based on these attributes.")
async def get_count_movies_by_title_gender_birth_country(movie_title: str = Query(..., description="Title of the movie"), gender: str = Query(..., description="Gender of the actor"), birth_country: str = Query(..., description="Birth country of the actor")):
    cursor.execute("SELECT COUNT(*) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ? AND T3.Gender = ? AND T3.`Birth Country` = ?", (movie_title, gender, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top movie by budget for a specific actor
@app.get("/v1/movie/top_movie_by_actor_budget", operation_id="get_top_movie_by_actor_budget", summary="Retrieves the top movie, sorted by budget in descending order, in which a specific actor has appeared. The response includes the movie's title and MPAA rating.")
async def get_top_movie_by_actor_budget(actor_name: str = Query(..., description="Name of the actor")):
    cursor.execute("SELECT T1.`MPAA Rating`, T1.Title FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.Name = ? ORDER BY T1.Budget DESC LIMIT 1", (actor_name,))
    result = cursor.fetchone()
    if not result:
        return {"movie": []}
    return {"movie": {"mpaa_rating": result[0], "title": result[1]}}

# Endpoint to get actor net worth and date of birth based on movie title and height range
@app.get("/v1/movie/actor_networth_dob_by_movie_title_height", operation_id="get_actor_networth_dob_by_movie_title_height", summary="Retrieves the net worth and date of birth of actors who have appeared in a specified movie and whose heights fall within a given range. The movie is identified by its title, and the height range is defined by a minimum and maximum value in inches.")
async def get_actor_networth_dob_by_movie_title_height(movie_title: str = Query(..., description="Title of the movie"), min_height: int = Query(..., description="Minimum height in inches"), max_height: int = Query(..., description="Maximum height in inches")):
    cursor.execute("SELECT T3.NetWorth, T3.`Date of Birth` FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Title = ? AND T3.`Height (Inches)` BETWEEN ? AND ?", (movie_title, min_height, max_height))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": [{"net_worth": row[0], "date_of_birth": row[1]} for row in result]}

# Endpoint to get the runtime of movies based on actor ethnicity and date of birth
@app.get("/v1/movie/runtime_by_ethnicity_dob", operation_id="get_runtime_by_ethnicity_dob", summary="Retrieves the runtime of movies featuring actors of a specific ethnicity born on a given date. The runtime is the duration of the movie in minutes.")
async def get_runtime_by_ethnicity_dob(ethnicity: str = Query(..., description="Ethnicity of the actor"), date_of_birth: str = Query(..., description="Date of birth of the actor in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.Runtime FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.Ethnicity = ? AND T3.`Date of Birth` = ?", (ethnicity, date_of_birth))
    result = cursor.fetchall()
    if not result:
        return {"runtime": []}
    return {"runtime": [row[0] for row in result]}

# Endpoint to get actor names based on movie gross, character name, and genre
@app.get("/v1/movie/actor_names_by_gross_character_genre", operation_id="get_actor_names_by_gross_character_genre", summary="Retrieves the names of actors who have played a specific character in movies of a certain genre that have grossed a specified amount.")
async def get_actor_names_by_gross_character_genre(gross: int = Query(..., description="Gross amount of the movie"), character_name: str = Query(..., description="Character name in the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT T3.Name FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Gross = ? AND T2.`Character Name` = ? AND T1.Genre = ?", (gross, character_name, genre))
    result = cursor.fetchall()
    if not result:
        return {"actor_names": []}
    return {"actor_names": [row[0] for row in result]}

# Endpoint to get the total gross of movies based on actor net worth, movie rating, and genre
@app.get("/v1/movie/total_gross_by_networth_rating_genre", operation_id="get_total_gross_by_networth_rating_genre", summary="Retrieves the cumulative gross earnings of movies that meet specific criteria: the actor's net worth exceeds a certain threshold, the movie's rating is below a given value, and the movie belongs to a particular genre.")
async def get_total_gross_by_networth_rating_genre(net_worth: float = Query(..., description="Net worth of the actor"), rating: float = Query(..., description="Rating of the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT SUM(T1.Gross) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE CAST(REPLACE(REPLACE(T3.NetWorth, ',', ''), '$', '') AS REAL) > ? AND T1.Rating < ? AND T1.Genre = ?", (net_worth, rating, genre))
    result = cursor.fetchone()
    if not result:
        return {"total_gross": []}
    return {"total_gross": result[0]}

# Endpoint to get the runtime of movies based on actor name and movie rating
@app.get("/v1/movie/runtime_by_actor_rating", operation_id="get_runtime_by_actor_rating", summary="Retrieves the runtime of movies featuring a specific actor and a rating above a given threshold. The operation filters movies by the provided actor's name and the minimum rating, then returns the runtime of the matching movies.")
async def get_runtime_by_actor_rating(actor_name: str = Query(..., description="Name of the actor"), rating: float = Query(..., description="Rating of the movie")):
    cursor.execute("SELECT T1.Runtime FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.Name = ? AND T1.Rating > ?", (actor_name, rating))
    result = cursor.fetchall()
    if not result:
        return {"runtime": []}
    return {"runtime": [row[0] for row in result]}

# Endpoint to get the difference in count of actors based on net worth and movie genre
@app.get("/v1/movie/networth_difference_by_genre", operation_id="get_networth_difference_by_genre", summary="Retrieves the difference in the count of actors with a net worth above and below a specified threshold for a given movie genre. The net worth threshold and genre are provided as input parameters.")
async def get_networth_difference_by_genre(net_worth_threshold: float = Query(..., description="Net worth threshold of the actor"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT SUM(CASE WHEN CAST(REPLACE(REPLACE(T3.NetWorth, ',', ''), '$', '') AS REAL) > ? THEN 1 ELSE 0 END) - SUM(CASE WHEN CAST(REPLACE(REPLACE(T3.NetWorth, ',', ''), '$', '') AS REAL) < ? THEN 1 ELSE 0 END) FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T1.Genre = ?", (net_worth_threshold, net_worth_threshold, genre))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get actor names based on gender, movie title, birth city, and height
@app.get("/v1/movie/actor_names_by_gender_title_city_height", operation_id="get_actor_names_by_gender_title_city_height", summary="Get actor names based on gender, movie title, birth city, and height")
async def get_actor_names_by_gender_title_city_height(gender: str = Query(..., description="Gender of the actor"), title: str = Query(..., description="Title of the movie"), birth_city: str = Query(..., description="Birth city of the actor"), height_inches: float = Query(..., description="Height of the actor in inches")):
    cursor.execute("SELECT T3.Name FROM movie AS T1 INNER JOIN characters AS T2 ON T1.MovieID = T2.MovieID INNER JOIN actor AS T3 ON T3.ActorID = T2.ActorID WHERE T3.Gender = ? AND T1.Title = ? AND T3.`Birth City` = ? AND T3.`Height (Inches)` * 100 > ( SELECT AVG(`Height (Inches)`) FROM actor ) * 50", (gender, title, birth_city, height_inches))
    result = cursor.fetchall()
    if not result:
        return {"actor_names": []}
    return {"actor_names": [row[0] for row in result]}

api_calls = [
    "/v1/movie/character_names_by_title?title=Look%20Who's%20Talking",
    "/v1/movie/character_most_screen_time?title=Batman",
    "/v1/movie/actor_name_by_title_and_character?title=Batman&character_name=Joker",
    "/v1/movie/actor_names_by_title?title=Batman",
    "/v1/movie/titles_by_character_name?character_name=Dr.%20Archibald%20'Moonlight'%20Graham",
    "/v1/movie/titles_by_actor_name?actor_name=Tom%20Cruise",
    "/v1/movie/count_by_actor_and_rating?actor_name=Morgan%20Freeman&mpaa_rating=PG",
    "/v1/movie/highest_rated_movie_by_actor?actor_name=Tom%20Cruise",
    "/v1/movie/character_names_by_actor_and_title?actor_name=Tom%20Cruise&title=Born%20on%20the%20Fourth%20of%20July",
    "/v1/movie/character_names_by_actor?actor_name=Tom%20Cruise",
    "/v1/movie/tallest_actor_in_movie?title=Batman",
    "/v1/movie/count_characters_by_gender_ethnicity?gender=Male&ethnicity=African%20American",
    "/v1/movie/average_rating_by_actor?actor_name=Tom%20Cruise",
    "/v1/movie/screentime_percentage_difference?title=Batman",
    "/v1/movie/highest_budget_movie",
    "/v1/movie/mpaa_rating_by_character?character_name=Peter%20Quill",
    "/v1/movie/highest_rated_character_by_genre_credit_order?credit_order=1&genre=Thriller",
    "/v1/movie/actor_most_screentime_in_movie?title=Batman",
    "/v1/movie/count_characters_by_richest_actor",
    "/v1/movie/actor_name_by_character?character_name=Chanice%20Kobolowski",
    "/v1/movie/actor_dob_by_character_name?character_name=Sully",
    "/v1/movie/actor_birth_city_by_character_name?character_name=Gabriel%20Martin",
    "/v1/movie/actor_biography_by_character_name?character_name=Michael%20Moscovitz",
    "/v1/movie/actor_height_by_character_name?character_name=Lurch",
    "/v1/movie/character_name_by_movie_title_and_credit_order?movie_title=G.I.%20Joe:%20The%20Rise%20of%20Cobra&credit_order=3",
    "/v1/movie/actor_name_by_movie_title_and_credit_order?movie_title=American%20Hustle&credit_order=2",
    "/v1/movie/actor_name_by_release_date_and_credit_order?release_date=2015-10-26&credit_order=1",
    "/v1/movie/percentage_actors_by_birth_country_and_movie_title?birth_country=USA&movie_title=Mrs.%20Doubtfire",
    "/v1/movie/percentage_actors_by_dob_and_movie_title?dob=1970-01-01&movie_title=Dawn%20of%20the%20Planet%20of%20the%20Apes",
    "/v1/movie/movie_ids_by_rating_and_budget?min_rating=7&max_rating=8&budget=15000000",
    "/v1/movie/count_movies_by_rating_and_release_date?mpaa_rating=PG&release_date_pattern=1990-06%",
    "/v1/movie/actor_name_by_id_gender_ethnicity?actor_id=439&gender=Male&ethnicity=White",
    "/v1/movie/genres_by_birth_city_and_rating?birth_city=New%20York%20City&rating=5",
    "/v1/movie/count_movies_by_genre_and_actor_name?genre=Romance&actor_name=John%20Travolta",
    "/v1/movie/actor_height_networth_by_movie_title?movie_title=Three%20Men%20and%20a%20Little%20Lady",
    "/v1/movie/top_genre_by_mpaa_rating_and_net_worth?mpaa_rating=PG",
    "/v1/movie/actor_networth_by_movie_title_height_gender?movie_title=Misery&min_height=60&max_height=70&gender=Male",
    "/v1/movie/count_movies_by_title_gender_birth_country?movie_title=Ghost&gender=Male&birth_country=USA",
    "/v1/movie/top_movie_by_actor_budget?actor_name=Leonardo%20DiCaprio",
    "/v1/movie/actor_networth_dob_by_movie_title_height?movie_title=Die%20Hard%202&min_height=60&max_height=65",
    "/v1/movie/runtime_by_ethnicity_dob?ethnicity=African%20American&date_of_birth=1954-12-28",
    "/v1/movie/actor_names_by_gross_character_genre?gross=136766062&character_name=Don%20Altobello&genre=Drama",
    "/v1/movie/total_gross_by_networth_rating_genre?net_worth=375000000&rating=7&genre=Comedy",
    "/v1/movie/runtime_by_actor_rating?actor_name=Jackie%20Chan&rating=7",
    "/v1/movie/networth_difference_by_genre?net_worth_threshold=400000000&genre=Drama",
    "/v1/movie/actor_names_by_gender_title_city_height?gender=Female&title=Godzilla&birth_city=Sherman%20Oaks&height_inches=50"
]
