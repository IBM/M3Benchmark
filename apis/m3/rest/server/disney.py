from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/disney/disney.sqlite')
cursor = conn.cursor()

# Endpoint to get the difference in total revenue between two years
@app.get("/v1/disney/revenue_difference", operation_id="get_revenue_difference", summary="Retrieves the difference in total revenue between two specified years. This operation compares the total revenue for the first year and the second year, both provided as input parameters, and returns the difference. The result offers a clear view of the revenue change between the two years.")
async def get_revenue_difference(year1: int = Query(..., description="First year for comparison"), year2: int = Query(..., description="Second year for comparison")):
    cursor.execute("SELECT SUM(CASE WHEN `Year` = ? THEN Total ELSE 0 END) - SUM(CASE WHEN `Year` = ? THEN Total ELSE 0 END) FROM revenue", (year1, year2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to compare two segments and return the greater one for a specific year
@app.get("/v1/disney/compare_segments", operation_id="compare_segments", summary="This endpoint compares the revenue of two specified segments for a given year and returns the segment with the higher revenue. The segments to be compared and the year for the comparison are provided as input parameters.")
async def compare_segments(segment1: str = Query(..., description="First segment to compare"), segment2: str = Query(..., description="Second segment to compare"), year: int = Query(..., description="Year for comparison")):
    cursor.execute("SELECT CASE WHEN ? > ? THEN ? ELSE ? END FROM revenue WHERE `Year` = ?", (segment1, segment2, segment1, segment2, year))
    result = cursor.fetchone()
    if not result:
        return {"segment": []}
    return {"segment": result[0]}

# Endpoint to get the director of a specific movie
@app.get("/v1/disney/director_by_movie", operation_id="get_director_by_movie", summary="Retrieves the director associated with a specific movie, identified by its name.")
async def get_director_by_movie(name: str = Query(..., description="Name of the movie")):
    cursor.execute("SELECT director FROM director WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"director": []}
    return {"director": result[0]}

# Endpoint to get the villains from movies directed by a specific director
@app.get("/v1/disney/villains_by_director", operation_id="get_villains_by_director", summary="Retrieve a list of villain characters from movies directed by the specified director. The operation filters out any non-villain characters and returns only those that are explicitly identified as villains in the characters table.")
async def get_villains_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T2.villian FROM director AS T1 INNER JOIN characters AS T2 ON T1.name = T2.movie_title WHERE T1.director = ? AND T2.villian IS NOT NULL", (director,))
    result = cursor.fetchall()
    if not result:
        return {"villains": []}
    return {"villains": [row[0] for row in result]}

# Endpoint to get the count of movies released in a specific month and directed by a specific director
@app.get("/v1/disney/movie_count_by_month_director", operation_id="get_movie_count_by_month_director", summary="Retrieve the number of movies directed by a specific director and released in a given month. The month should be provided in 'MMM' format. The director's name is also required to filter the results.")
async def get_movie_count_by_month_director(month: str = Query(..., description="Month of release in 'MMM' format"), director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT COUNT(movie_title) FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE SUBSTR(release_date, INSTR(release_date, '-') + 1, 3) = ? AND T2.director = ?", (month, director))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the director of a movie containing a specific song
@app.get("/v1/disney/director_by_song", operation_id="get_director_by_song", summary="Retrieves the director of a movie that features a specified song. The operation searches for the movie title associated with the given song and returns the director of that movie.")
async def get_director_by_song(song: str = Query(..., description="Name of the song")):
    cursor.execute("SELECT T2.director FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T1.song = ?", (song,))
    result = cursor.fetchone()
    if not result:
        return {"director": []}
    return {"director": result[0]}

# Endpoint to get voice actors for characters or villains in a specific movie
@app.get("/v1/disney/voice_actors_by_movie", operation_id="get_voice_actors_by_movie", summary="Retrieves the voice actors associated with characters or villains in a specified movie. The operation accepts a character name pattern, a villain name pattern, and the movie title as input parameters. It returns the voice actors who have voiced characters or villains that match the provided patterns in the given movie.")
async def get_voice_actors_by_movie(character: str = Query(..., description="Character name pattern"), villain: str = Query(..., description="Villain name pattern"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T1.`voice-actor` FROM `voice-actors` AS T1 INNER JOIN characters AS T2 ON T1.movie = T2.movie_title WHERE T1.character LIKE ? OR T2.villian LIKE ? AND T2.movie_title = ?", (character, villain, movie_title))
    result = cursor.fetchall()
    if not result:
        return {"voice_actors": []}
    return {"voice_actors": [row[0] for row in result]}

# Endpoint to get the release dates of movies featuring a specific voice actor
@app.get("/v1/disney/release_dates_by_voice_actor", operation_id="get_release_dates_by_voice_actor", summary="Retrieve the release dates of movies in which a specified voice actor has performed. The input parameter is the name of the voice actor.")
async def get_release_dates_by_voice_actor(voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT T2.release_date FROM `voice-actors` AS T1 INNER JOIN characters AS T2 ON T1.movie = T2.movie_title WHERE T1.`voice-actor` = ?", (voice_actor,))
    result = cursor.fetchall()
    if not result:
        return {"release_dates": []}
    return {"release_dates": [row[0] for row in result]}

# Endpoint to get the count of movies featuring a specific voice actor and released after a specific day of the month
@app.get("/v1/disney/movie_count_by_voice_actor_day", operation_id="get_movie_count_by_voice_actor_day", summary="Retrieves the number of movies featuring a specified voice actor that were released after a certain day of the month. The operation filters the movies based on the provided voice actor's name and the day of the month, and returns the count of movies that meet these criteria.")
async def get_movie_count_by_voice_actor_day(voice_actor: str = Query(..., description="Name of the voice actor"), day: int = Query(..., description="Day of the month")):
    cursor.execute("SELECT COUNT(T2.movie) FROM characters AS T1 INNER JOIN `voice-actors` AS T2 ON T1.movie_title = T2.movie WHERE T2.`voice-actor` = ? AND SUBSTR(release_date, INSTR(release_date, '-') + 5) > ?", (voice_actor, day))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies directed by a specific director and belonging to a specific genre
@app.get("/v1/disney/movie_count_by_director_genre", operation_id="get_movie_count_by_director_genre", summary="Retrieves the total number of movies directed by a specific director and belonging to a specific genre. The operation requires the director's name and the genre as input parameters to filter the results.")
async def get_movie_count_by_director_genre(director: str = Query(..., description="Name of the director"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT COUNT(T3.name) FROM ( SELECT T2.name FROM `movies_total_gross` AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T2.director = ? AND T1.genre = ? GROUP BY T2.name ) T3", (director, genre))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest grossing movie directed by a specific director
@app.get("/v1/disney/highest_grossing_movie_by_director", operation_id="get_highest_grossing_movie_by_director", summary="Retrieves the title of the highest grossing movie directed by the specified director. The operation uses the director's name to identify the relevant movie and returns the title of the movie with the highest total gross revenue.")
async def get_highest_grossing_movie_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T2.movie_title FROM director AS T1 INNER JOIN movies_total_gross AS T2 ON T1.name = T2.movie_title WHERE T1.director = ? ORDER BY T2.total_gross DESC LIMIT 1", (director,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get movies with a specific MPAA rating directed by a specific director
@app.get("/v1/disney/movies_by_rating_and_director", operation_id="get_movies_by_rating_and_director", summary="Retrieves a list of movies with a specified MPAA rating that were directed by a specific director. The operation filters the movies based on the provided rating and director, and returns the titles of the matching movies.")
async def get_movies_by_rating_and_director(mpaa_rating: str = Query(..., description="MPAA rating of the movie"), director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T1.movie_title FROM `movies_total_gross` AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T1.MPAA_rating = ? AND T2.director = ?", (mpaa_rating, director))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the villain from the highest grossing movie
@app.get("/v1/disney/villain_from_highest_grossing_movie", operation_id="get_villain_from_highest_grossing_movie", summary="Retrieves the name of the villain from the movie that has earned the highest total gross revenue. The operation uses an inner join to match the movie title from the highest grossing movie with the corresponding villain in the characters table.")
async def get_villain_from_highest_grossing_movie():
    cursor.execute("SELECT T2.villian FROM `movies_total_gross` AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title ORDER BY T1.total_gross DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"villain": []}
    return {"villain": result[0]}

# Endpoint to get the genre of movies featuring a specific villain
@app.get("/v1/disney/genre_by_villain", operation_id="get_genre_by_villain", summary="Retrieve the genre of movies in which a specified villain appears. The operation filters movies based on the provided villain's name and returns the corresponding genre.")
async def get_genre_by_villain(villain: str = Query(..., description="Name of the villain")):
    cursor.execute("SELECT T2.genre FROM characters AS T1 INNER JOIN movies_total_gross AS T2 ON T2.movie_title = T1.movie_title WHERE T1.villian = ?", (villain,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the villain from a specific movie
@app.get("/v1/disney/villain_by_movie", operation_id="get_villain_by_movie", summary="Retrieves the villain character from a specified Disney movie. The operation requires the title of the movie as input and returns the corresponding villain character.")
async def get_villain_by_movie(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT villian FROM characters WHERE movie_title = ?", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"villain": []}
    return {"villain": result[0]}

# Endpoint to get movies featuring a specific hero
@app.get("/v1/disney/movies_by_hero", operation_id="get_movies_by_hero", summary="Retrieves a list of movie titles featuring the specified hero. The hero's name is used to filter the results.")
async def get_movies_by_hero(hero: str = Query(..., description="Name of the hero")):
    cursor.execute("SELECT movie_title FROM characters WHERE hero = ?", (hero,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get movies featuring a specific song
@app.get("/v1/disney/movies_by_song", operation_id="get_movies_by_song", summary="Retrieves a list of Disney movies that feature a specified song. The song title is used to search for relevant movies.")
async def get_movies_by_song(song: str = Query(..., description="Title of the song")):
    cursor.execute("SELECT movie_title FROM characters WHERE song = ?", (song,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the voice actor for a specific character
@app.get("/v1/disney/voice_actor_by_character", operation_id="get_voice_actor_by_character", summary="Retrieves the voice actor associated with a given character in the Disney universe. The character's name is required as an input parameter to identify the corresponding voice actor.")
async def get_voice_actor_by_character(character: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT `voice-actor` FROM `voice-actors` WHERE character = ?", (character,))
    result = cursor.fetchone()
    if not result:
        return {"voice_actor": []}
    return {"voice_actor": result[0]}

# Endpoint to get the hero from a movie with a specific total gross
@app.get("/v1/disney/hero_by_total_gross", operation_id="get_hero_by_total_gross", summary="Retrieve the name of the hero from a Disney movie that has earned a specific total gross. The total gross must be provided in the format '$XXX,XXX,XXX'. This operation returns the hero's name by joining the characters and movies_total_gross tables on the movie title and filtering for the specified total gross.")
async def get_hero_by_total_gross(total_gross: str = Query(..., description="Total gross of the movie in the format '$XXX,XXX,XXX'")):
    cursor.execute("SELECT T1.hero FROM characters AS T1 INNER JOIN movies_total_gross AS T2 ON T2.movie_title = T1.movie_title WHERE T2.total_gross = ?", (total_gross,))
    result = cursor.fetchone()
    if not result:
        return {"hero": []}
    return {"hero": result[0]}

# Endpoint to get the top song from movies released in a specific decade
@app.get("/v1/disney/top_song_by_decade", operation_id="get_top_song_by_decade", summary="Retrieve the top-performing song from Disney movies released between the specified start and end years of a decade. The song is determined by the highest total gross of the movie it is featured in. The start and end years are inclusive and should be provided in the format YYYY.")
async def get_top_song_by_decade(start_year: int = Query(..., description="Start year of the decade (e.g., 1970)"), end_year: int = Query(..., description="End year of the decade (e.g., 1979)")):
    cursor.execute("SELECT T2.song FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title WHERE CAST(SUBSTR(T1.release_date, INSTR(T1.release_date, ', ') + 1) AS int) BETWEEN ? AND ? ORDER BY CAST(REPLACE(SUBSTR(T1.total_gross, 2), ',', '') AS float) DESC LIMIT 1", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"song": []}
    return {"song": result[0]}

# Endpoint to get the hero of a movie directed by a specific director
@app.get("/v1/disney/hero_by_director", operation_id="get_hero_by_director", summary="Retrieves the hero of a movie directed by the specified director. The input parameter is used to identify the director, and the operation returns the hero's name.")
async def get_hero_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T1.hero FROM characters AS T1 INNER JOIN director AS T2 ON T2.name = T1.movie_title WHERE T2.director = ?", (director,))
    result = cursor.fetchall()
    if not result:
        return {"heroes": []}
    return {"heroes": [row[0] for row in result]}

# Endpoint to get the voice actor of a hero in a specific movie
@app.get("/v1/disney/voice_actor_by_movie_and_hero", operation_id="get_voice_actor_by_movie_and_hero", summary="Get the voice actor of a hero in a specific movie")
async def get_voice_actor_by_movie_and_hero(movie_title: str = Query(..., description="Title of the movie"), hero: str = Query(..., description="Name of the hero")):
    cursor.execute("SELECT T2.`voice-actor` FROM characters AS T1 INNER JOIN `voice-actors` AS T2 ON T2.movie = T1.movie_title WHERE T1.movie_title = ? AND T2.character = T1.hero", (movie_title, hero))
    result = cursor.fetchall()
    if not result:
        return {"voice_actors": []}
    return {"voice_actors": [row[0] for row in result]}

# Endpoint to get the director of a movie based on a specific character and voice actor
@app.get("/v1/disney/director_by_character_and_voice_actor", operation_id="get_director_by_character_and_voice_actor", summary="Retrieves the director of a movie in which a specified character was voiced by a particular voice actor. The operation requires the name of the character and the voice actor as input parameters.")
async def get_director_by_character_and_voice_actor(character: str = Query(..., description="Name of the character"), voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT T1.director FROM director AS T1 INNER JOIN `voice-actors` AS T2 ON T2.movie = T1.name WHERE T2.character = ? AND T2.`voice-actor` = ?", (character, voice_actor))
    result = cursor.fetchall()
    if not result:
        return {"directors": []}
    return {"directors": [row[0] for row in result]}

# Endpoint to get the release date of a movie based on a specific character and voice actor
@app.get("/v1/disney/release_date_by_character_and_voice_actor", operation_id="get_release_date_by_character_and_voice_actor", summary="Retrieves the release date of a movie featuring a specified character voiced by a particular actor. The operation requires the character's name and the voice actor's name as input parameters.")
async def get_release_date_by_character_and_voice_actor(character: str = Query(..., description="Name of the character"), voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT T1.release_date FROM characters AS T1 INNER JOIN `voice-actors` AS T2 ON T2.movie = T1.movie_title WHERE T2.character = ? AND T2.`voice-actor` = ?", (character, voice_actor))
    result = cursor.fetchall()
    if not result:
        return {"release_dates": []}
    return {"release_dates": [row[0] for row in result]}

# Endpoint to get the director of a movie based on genre and release date
@app.get("/v1/disney/director_by_genre_and_release_date", operation_id="get_director_by_genre_and_release_date", summary="Retrieves the director of a movie that matches the specified genre and release date. The genre and release date are provided as input parameters, allowing the user to filter the results and obtain the desired director.")
async def get_director_by_genre_and_release_date(genre: str = Query(..., description="Genre of the movie"), release_date: str = Query(..., description="Release date of the movie in 'MMM DD, YYYY' format")):
    cursor.execute("SELECT T1.director FROM director AS T1 INNER JOIN movies_total_gross AS T2 ON T2.movie_title = T1.name WHERE T2.genre = ? AND T2.release_date = ?", (genre, release_date))
    result = cursor.fetchall()
    if not result:
        return {"directors": []}
    return {"directors": [row[0] for row in result]}

# Endpoint to get the hero of a movie based on genre and release date
@app.get("/v1/disney/hero_by_genre_and_release_date", operation_id="get_hero_by_genre_and_release_date", summary="Retrieves the hero of a movie that matches the specified genre and release date. The genre and release date are used to filter the movies, and the hero of the first matching movie is returned. The release date should be provided in 'DD-MMM-YY' format.")
async def get_hero_by_genre_and_release_date(genre: str = Query(..., description="Genre of the movie"), release_date: str = Query(..., description="Release date of the movie in 'DD-MMM-YY' format")):
    cursor.execute("SELECT T1.hero FROM characters AS T1 INNER JOIN movies_total_gross AS T2 ON T2.movie_title = T1.movie_title WHERE T2.genre = ? AND T1.release_date = ?", (genre, release_date))
    result = cursor.fetchall()
    if not result:
        return {"heroes": []}
    return {"heroes": [row[0] for row in result]}

# Endpoint to get the highest grossing movie featuring a specific hero
@app.get("/v1/disney/highest_grossing_movie_by_hero", operation_id="get_highest_grossing_movie_by_hero", summary="Retrieves the title of the highest grossing movie featuring a specified hero. The hero's name is used to filter the movies and identify the one with the highest total gross revenue. The result is returned in descending order of total gross revenue, with the highest grossing movie listed first.")
async def get_highest_grossing_movie_by_hero(hero: str = Query(..., description="Name of the hero")):
    cursor.execute("SELECT T1.movie_title FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title WHERE T2.hero = ? ORDER BY CAST(REPLACE(SUBSTR(total_gross, 2), ',', '') AS REAL) DESC LIMIT 1", (hero,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the count of movies directed by a specific director
@app.get("/v1/disney/count_movies_by_director", operation_id="get_count_movies_by_director", summary="Retrieve the total number of movies directed by a given director. The operation requires the director's name as input and returns the corresponding count.")
async def get_count_movies_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT COUNT(name) FROM director WHERE director = ?", (director,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the director with the most movies
@app.get("/v1/disney/top_director", operation_id="get_top_director", summary="Retrieves the director who has directed the most movies in the database. The result is determined by counting the number of movies each director has directed and sorting them in descending order. The director with the highest count is returned.")
async def get_top_director():
    cursor.execute("SELECT director FROM director GROUP BY director ORDER BY COUNT(name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"director": []}
    return {"director": result[0]}

# Endpoint to get the count of movies based on MPAA rating, genre, and release date range
@app.get("/v1/disney/movie_count_by_rating_genre_release_date", operation_id="get_movie_count", summary="Retrieves the number of movies that match a specified MPAA rating, genre, and release date range. The response includes the total count of movies that meet the provided criteria.")
async def get_movie_count(mpaa_rating: str = Query(..., description="MPAA rating of the movie"), genre: str = Query(..., description="Genre of the movie"), start_year: int = Query(..., description="Start year of the release date range"), end_year: int = Query(..., description="End year of the release date range")):
    cursor.execute("SELECT COUNT(movie_title) FROM movies_total_gross WHERE MPAA_rating = ? AND genre = ? AND CAST(SUBSTR(release_date, INSTR(release_date, ', ') + 1) AS int) BETWEEN ? AND ?", (mpaa_rating, genre, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get characters based on voice actor
@app.get("/v1/disney/characters_by_voice_actor", operation_id="get_characters", summary="Retrieves a list of characters voiced by a specified voice actor. The operation filters the characters based on the provided voice actor's name.")
async def get_characters(voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT character FROM `voice-actors` WHERE 'voice-actor' = ?", (voice_actor,))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get total gross of movies based on a specific song
@app.get("/v1/disney/total_gross_by_song", operation_id="get_total_gross_by_song", summary="Retrieves the cumulative box office earnings of all movies that include a specified song in their soundtrack. The song title is provided as an input parameter.")
async def get_total_gross_by_song(song: str = Query(..., description="Name of the song")):
    cursor.execute("SELECT T1.total_gross FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T2.movie_title = T1.movie_title WHERE T2.song = ?", (song,))
    result = cursor.fetchall()
    if not result:
        return {"total_gross": []}
    return {"total_gross": [row[0] for row in result]}

# Endpoint to get MPAA ratings of movies based on a specific villain
@app.get("/v1/disney/mpaa_rating_by_villain", operation_id="get_mpaa_rating_by_villain", summary="Retrieve the MPAA ratings of movies in which a specified villain appears. The operation filters movies based on the provided villain's name and returns the corresponding MPAA ratings.")
async def get_mpaa_rating_by_villain(villain: str = Query(..., description="Name of the villain")):
    cursor.execute("SELECT T1.MPAA_rating FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T2.movie_title = T1.movie_title WHERE T2.villian = ?", (villain,))
    result = cursor.fetchall()
    if not result:
        return {"mpaa_ratings": []}
    return {"mpaa_ratings": [row[0] for row in result]}

# Endpoint to get the count of movies based on MPAA rating and voice actor
@app.get("/v1/disney/movie_count_by_rating_voice_actor", operation_id="get_movie_count_by_rating_voice_actor", summary="Retrieves the total number of movies that have a specified MPAA rating and feature a particular voice actor. The response provides a count of movies that meet both criteria.")
async def get_movie_count_by_rating_voice_actor(mpaa_rating: str = Query(..., description="MPAA rating of the movie"), voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT COUNT(T.movie) FROM ( SELECT T1.movie FROM `voice-actors` AS T1 INNER JOIN movies_total_gross AS T2 ON T1.movie = T2.movie_title WHERE MPAA_rating = ? AND `voice-actor` = ? GROUP BY T1.movie ) AS T", (mpaa_rating, voice_actor))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top director based on voice actor
@app.get("/v1/disney/top_director_by_voice_actor", operation_id="get_top_director_by_voice_actor", summary="Retrieves the director who has most frequently collaborated with the specified voice actor. The result is determined by counting the number of movies in which the voice actor and director have worked together.")
async def get_top_director_by_voice_actor(voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT director FROM director AS T1 INNER JOIN `voice-actors` AS T2 ON T1.name = T2.movie WHERE T2.`voice-actor` = ? GROUP BY director ORDER BY COUNT(director) DESC LIMIT 1", (voice_actor,))
    result = cursor.fetchone()
    if not result:
        return {"director": []}
    return {"director": result[0]}

# Endpoint to get the top movie by total gross based on director
@app.get("/v1/disney/top_movie_by_director", operation_id="get_top_movie_by_director", summary="Retrieves the top-grossing movie directed by a specific director. The operation uses the provided director's name to search for the movie with the highest total gross. The result is returned in descending order, with the top movie listed first.")
async def get_top_movie_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T2.name FROM movies_total_gross AS T1 INNER JOIN director AS T2 ON T2.name = T1.movie_title WHERE T2.director = ? ORDER BY CAST(REPLACE(SUBSTR(total_gross, 2), ',', '') AS int) DESC LIMIT 1", (director,))
    result = cursor.fetchone()
    if not result:
        return {"movie": []}
    return {"movie": result[0]}

# Endpoint to get voice actors based on director and release date
@app.get("/v1/disney/voice_actors_by_director_release_date", operation_id="get_voice_actors_by_director_release_date", summary="Retrieve a list of voice actors who have collaborated with a specified director on a movie released on a particular date. The provided director's name and movie release date in 'Month DD, YYYY' format are used to filter the results. The response includes unique voice actors, excluding those with 'None' as their name.")
async def get_voice_actors_by_director_release_date(director: str = Query(..., description="Name of the director"), release_date: str = Query(..., description="Release date of the movie in 'Month DD, YYYY' format")):
    cursor.execute("SELECT T2.`voice-actor` FROM director AS T1 INNER JOIN `voice-actors` AS T2 INNER JOIN movies_total_gross AS T3 ON T1.name = T2.movie AND T2.movie = T3.movie_title WHERE T1.director = ? AND T3.release_date = ? AND T2.`voice-actor` != 'None' GROUP BY T2.`voice-actor`", (director, release_date))
    result = cursor.fetchall()
    if not result:
        return {"voice_actors": []}
    return {"voice_actors": [row[0] for row in result]}

# Endpoint to get the count of movies based on director, MPAA rating, and genre
@app.get("/v1/disney/movie_count_by_director_rating_genre", operation_id="get_movie_count_by_director_rating_genre", summary="Retrieves the total number of movies directed by a specific director, filtered by a given MPAA rating and genre. This operation provides a quantitative overview of a director's work within a particular rating and genre category.")
async def get_movie_count_by_director_rating_genre(director: str = Query(..., description="Name of the director"), mpaa_rating: str = Query(..., description="MPAA rating of the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT COUNT(*) FROM director AS T1 INNER JOIN movies_total_gross AS T2 ON T1.name = T2.movie_title WHERE T1.director = ? AND T2.MPAA_rating = ? AND T2.genre = ?", (director, mpaa_rating, genre))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies based on genre
@app.get("/v1/disney/movie_count_by_genre", operation_id="get_movie_count_by_genre", summary="Retrieves the total number of movies belonging to a specified genre. The genre is provided as an input parameter.")
async def get_movie_count_by_genre(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT COUNT(movie_title) FROM `movies_total_gross` WHERE genre = ?", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the year with the highest revenue within a range
@app.get("/v1/disney/top_revenue_year", operation_id="get_top_revenue_year", summary="Retrieves the year with the highest revenue from the Studio Entertainment sector within the specified range. The range is defined by the start and end years provided as input parameters.")
async def get_top_revenue_year(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT `Year` FROM revenue WHERE `Year` BETWEEN ? AND ? ORDER BY `Studio Entertainment[NI 1]` DESC LIMIT 1", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get songs from movies of a specific genre
@app.get("/v1/disney/songs_by_genre", operation_id="get_songs_by_genre", summary="Retrieves a list of unique songs from Disney movies that belong to a specified genre. The genre is provided as an input parameter.")
async def get_songs_by_genre(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT song FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title WHERE T1.genre = ? GROUP BY song", (genre,))
    result = cursor.fetchall()
    if not result:
        return {"songs": []}
    return {"songs": [row[0] for row in result]}

# Endpoint to get voice actors for heroes in movies
@app.get("/v1/disney/voice_actors_for_heroes", operation_id="get_voice_actors_for_heroes", summary="Retrieves the voice actors associated with the heroes in a specific movie. This operation returns a list of voice actors who have provided their talents for the heroes in a given movie. The voice actors are identified by matching the hero characters with their respective voice actors in the same movie.")
async def get_voice_actors_for_heroes():
    cursor.execute("SELECT T2.`voice-actor` FROM characters AS T1 INNER JOIN `voice-actors` AS T2 ON T2.character = T1.hero WHERE T2.movie = T1.movie_title")
    result = cursor.fetchall()
    if not result:
        return {"voice_actors": []}
    return {"voice_actors": [row[0] for row in result]}

# Endpoint to get directors of movies released within a specific year range
@app.get("/v1/disney/directors_by_year_range", operation_id="get_directors_by_year_range", summary="Retrieves a list of directors who have directed movies released within a specified year range. The range is defined by the start and end years provided as input parameters. The operation returns a distinct list of directors, grouped by their names.")
async def get_directors_by_year_range(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT T2.director FROM movies_total_gross AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name AND CAST(SUBSTR(release_date, INSTR(release_date, ', ') + 1) AS int) BETWEEN ? AND ? GROUP BY T2.director", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"directors": []}
    return {"directors": [row[0] for row in result]}

# Endpoint to get voice actors for villains in a specific movie
@app.get("/v1/disney/voice_actors_for_villains", operation_id="get_voice_actors_for_villains", summary="Retrieves the voice actors who have portrayed villains in a specified movie. The operation filters the voice actors based on the provided movie title and returns the corresponding voice actor names.")
async def get_voice_actors_for_villains(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T1.`voice-actor` FROM `voice-actors` AS T1 INNER JOIN characters AS T2 ON T2.movie_title = T1.movie WHERE T2.movie_title = ? AND T1.character = T2.villian", (movie_title,))
    result = cursor.fetchall()
    if not result:
        return {"voice_actors": []}
    return {"voice_actors": [row[0] for row in result]}

# Endpoint to get movies and voice actors directed by a specific director
@app.get("/v1/disney/movies_and_voice_actors_by_director", operation_id="get_movies_and_voice_actors_by_director", summary="Retrieves a list of movies and their respective voice actors directed by the specified director. The director's name is provided as an input parameter.")
async def get_movies_and_voice_actors_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T1.name, T2.`voice-actor` FROM director AS T1 INNER JOIN `voice-actors` AS T2 ON T1.name = T2.movie WHERE T1.director = ?", (director,))
    result = cursor.fetchall()
    if not result:
        return {"movies_and_voice_actors": []}
    return {"movies_and_voice_actors": [{"movie": row[0], "voice_actor": row[1]} for row in result]}

# Endpoint to get distinct characters from movies with a specific MPAA rating
@app.get("/v1/disney/characters_by_mpaa_rating", operation_id="get_characters_by_mpaa_rating", summary="Retrieve a unique list of characters that appear in movies with a specified MPAA rating. The operation filters movies based on the provided MPAA rating and extracts the distinct characters from the filtered movies.")
async def get_characters_by_mpaa_rating(mpaa_rating: str = Query(..., description="MPAA rating of the movie")):
    cursor.execute("SELECT DISTINCT T2.character FROM movies_total_gross AS T1 INNER JOIN `voice-actors` AS T2 ON T1.movie_title = T2.movie WHERE T1.MPAA_rating = ?", (mpaa_rating,))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get the movie title with the highest total gross where the song is null
@app.get("/v1/disney/highest_gross_movie_without_song", operation_id="get_highest_gross_movie_without_song", summary="Retrieves the title of the movie with the highest total gross that does not feature any songs. The operation filters out movies with songs and sorts the remaining movies by their total gross in descending order, returning the top result.")
async def get_highest_gross_movie_without_song():
    cursor.execute("SELECT T1.movie_title FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T2.movie_title = T1.movie_title WHERE T2.song IS NULL ORDER BY CAST(REPLACE(trim(T1.total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the director with the most distinct voice actors
@app.get("/v1/disney/director_with_most_voice_actors", operation_id="get_director_with_most_voice_actors", summary="Retrieves the director who has worked with the most unique voice actors. This operation calculates the count of distinct voice actors each director has collaborated with and returns the director with the highest count.")
async def get_director_with_most_voice_actors():
    cursor.execute("SELECT T2.director, COUNT(DISTINCT T1.`voice-actor`) FROM `voice-actors` AS T1 INNER JOIN director AS T2 ON T1.movie = T2.name GROUP BY T2.director ORDER BY COUNT(DISTINCT T1.`voice-actor`) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"director": [], "count": []}
    return {"director": result[0], "count": result[1]}

# Endpoint to get voice actors from movies released on a specific date
@app.get("/v1/disney/voice_actors_by_release_date", operation_id="get_voice_actors_by_release_date", summary="Retrieves a list of voice actors who have contributed to movies released on a specific date. The date should be provided in the 'MMM DD, YYYY' format. This operation does not require any authentication.")
async def get_voice_actors_by_release_date(release_date: str = Query(..., description="Release date of the movie in 'MMM DD, YYYY' format")):
    cursor.execute("SELECT T2.`voice-actor` FROM movies_total_gross AS T1 INNER JOIN `voice-actors` AS T2 ON T1.movie_title = T2.movie WHERE T1.release_date = ?", (release_date,))
    result = cursor.fetchall()
    if not result:
        return {"voice_actors": []}
    return {"voice_actors": [row[0] for row in result]}

# Endpoint to get directors of movies with songs
@app.get("/v1/disney/directors_with_songs", operation_id="get_directors_with_songs", summary="Retrieves a list of directors who have directed movies that include songs. The data is obtained by joining the 'characters' and 'director' tables on the 'movie_title' and 'name' fields, respectively. The query filters out movies without songs and groups the results by the 'director' field.")
async def get_directors_with_songs():
    cursor.execute("SELECT T2.director FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T1.song IS NOT NULL GROUP BY T2.director")
    result = cursor.fetchall()
    if not result:
        return {"directors": []}
    return {"directors": [row[0] for row in result]}

# Endpoint to get the highest grossing movie of a specific voice actor
@app.get("/v1/disney/highest_gross_movie_by_voice_actor", operation_id="get_highest_gross_movie_by_voice_actor", summary="Retrieves the title of the highest grossing movie associated with a specified voice actor. The voice actor's name is used to filter the results, and the movie with the highest total gross revenue is returned.")
async def get_highest_gross_movie_by_voice_actor(voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT T2.movie_title FROM `voice-actors` AS T1 INNER JOIN movies_total_gross AS T2 ON T2.movie_title = T1.movie WHERE T1.`voice-actor` = ? ORDER BY CAST(REPLACE(trim(T2.total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1", (voice_actor,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the highest grossing movie directed by a specific director
@app.get("/v1/disney/highest_gross_movie_by_director", operation_id="get_highest_gross_movie_by_director", summary="Retrieves the title of the highest grossing movie directed by the specified director. The operation filters movies by the provided director's name and sorts them by their total gross earnings in descending order. The movie with the highest gross earnings is then returned.")
async def get_highest_gross_movie_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T2.movie_title FROM director AS T1 INNER JOIN movies_total_gross AS T2 ON T1.name = T2.movie_title WHERE T1.director = ? ORDER BY CAST(REPLACE(trim(T2.total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1", (director,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the average total gross of movies for a specific voice actor
@app.get("/v1/disney/average_gross_by_voice_actor", operation_id="get_average_gross_by_voice_actor", summary="Retrieves the average total gross of movies in which a specified voice actor has participated. The calculation is based on the sum of the total gross of all movies associated with the voice actor, divided by the total number of movies.")
async def get_average_gross_by_voice_actor(voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT SUM(CAST(REPLACE(trim(T2.total_gross, '$'), ',', '') AS REAL)) / COUNT(T2.movie_title) FROM `voice-actors` AS T1 INNER JOIN movies_total_gross AS T2 ON T1.movie = T2.movie_title WHERE T1.`voice-actor` = ?", (voice_actor,))
    result = cursor.fetchone()
    if not result:
        return {"average_gross": []}
    return {"average_gross": result[0]}

# Endpoint to get the percentage of movies with songs
@app.get("/v1/disney/percentage_movies_with_songs", operation_id="get_percentage_movies_with_songs", summary="Retrieves the percentage of Disney movies that have songs. This operation calculates the ratio of movies with songs to the total number of movies, providing a quantitative measure of the prevalence of songs in Disney movies.")
async def get_percentage_movies_with_songs():
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.song IS NOT NULL THEN T2.movie_title ELSE NULL END) AS REAL) * 100 / COUNT(T2.movie_title) FROM characters AS T1 INNER JOIN movies_total_gross AS T2 ON T1.movie_title = T2.movie_title")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get movie titles and genres released in a specific year
@app.get("/v1/disney/movies_by_year", operation_id="get_movies_by_year", summary="Retrieves the titles and genres of Disney movies released in a specified year. The year parameter is used to filter the results.")
async def get_movies_by_year(year: str = Query(..., description="Year of release in 'YYYY' format")):
    cursor.execute("SELECT movie_title, genre FROM movies_total_gross WHERE SUBSTR(release_date, LENGTH(release_date) - 3, LENGTH(release_date)) = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": [{"movie_title": row[0], "genre": row[1]} for row in result]}

# Endpoint to get the name of the director
@app.get("/v1/disney/director_name", operation_id="get_director_name", summary="Retrieves the full name of a specified director from the Disney database.")
async def get_director_name(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT name FROM director WHERE director = ?", (director,))
    result = cursor.fetchall()
    if not result:
        return {"director_names": []}
    return {"director_names": [row[0] for row in result]}

# Endpoint to get the movie with the highest total gross
@app.get("/v1/disney/highest_grossing_movie", operation_id="get_highest_grossing_movie", summary="Retrieves the movie with the highest total gross from the Disney catalog. The response includes the movie title and the ratio of its inflation-adjusted gross to its total gross.")
async def get_highest_grossing_movie():
    cursor.execute("SELECT movie_title, CAST(REPLACE(trim(inflation_adjusted_gross, '$'), ',', '') AS REAL) / CAST(REPLACE(trim(total_gross, '$'), ',', '') AS REAL) FROM movies_total_gross ORDER BY CAST(REPLACE(trim(total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie": []}
    return {"movie": result[0], "ratio": result[1]}

# Endpoint to get movies by MPAA rating and genre
@app.get("/v1/disney/movies_by_rating_genre", operation_id="get_movies_by_rating_genre", summary="Retrieves a list of movies that match the specified MPAA rating and genre. The response includes the movie title and release date for each matching movie.")
async def get_movies_by_rating_genre(mpaa_rating: str = Query(..., description="MPAA rating of the movie"), genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT movie_title, release_date FROM movies_total_gross WHERE MPAA_rating = ? AND genre = ?", (mpaa_rating, genre))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": [{"title": row[0], "release_date": row[1]} for row in result]}

# Endpoint to get movies and characters by voice actor
@app.get("/v1/disney/movies_characters_by_voice_actor", operation_id="get_movies_characters_by_voice_actor", summary="Get movies and characters by voice actor")
async def get_movies_characters_by_voice_actor(voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT movie, character FROM `voice-actors` WHERE `voice-actor` = ?", (voice_actor,))
    result = cursor.fetchall()
    if not result:
        return {"movies_characters": []}
    return {"movies_characters": [{"movie": row[0], "character": row[1]} for row in result]}

# Endpoint to get movie titles and songs by director
@app.get("/v1/disney/movie_songs_by_director", operation_id="get_movie_songs_by_director", summary="Retrieves a list of movie titles and their associated songs directed by the specified director. The director's name is provided as an input parameter.")
async def get_movie_songs_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T1.movie_title, T1.song FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T2.director = ?", (director,))
    result = cursor.fetchall()
    if not result:
        return {"movie_songs": []}
    return {"movie_songs": [{"movie_title": row[0], "song": row[1]} for row in result]}

# Endpoint to get movie titles, heroes, and songs by director and release year
@app.get("/v1/disney/movie_heroes_songs_by_director_year", operation_id="get_movie_heroes_songs_by_director_year", summary="Retrieves a list of movie titles, their respective heroes, and associated songs directed by a specific director in a given year. The director's name and the four-digit release year are required as input parameters.")
async def get_movie_heroes_songs_by_director_year(director: str = Query(..., description="Name of the director"), release_year: str = Query(..., description="Release year in 'YYYY' format")):
    cursor.execute("SELECT T1.movie_title, T2.hero, T2.song FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title INNER JOIN director AS T3 ON T1.movie_title = T3.name WHERE T3.director = ? AND SUBSTR(T1.release_date, LENGTH(T1.release_date) - 3, LENGTH(T1.release_date)) = ?", (director, release_year))
    result = cursor.fetchall()
    if not result:
        return {"movie_heroes_songs": []}
    return {"movie_heroes_songs": [{"movie_title": row[0], "hero": row[1], "song": row[2]} for row in result]}

# Endpoint to get movie titles and directors by hero
@app.get("/v1/disney/movie_directors_by_hero", operation_id="get_movie_directors_by_hero", summary="Retrieves a list of movie titles and their respective directors based on the provided hero's name. The hero's name is used to filter the results, ensuring that only movies featuring the specified hero are returned.")
async def get_movie_directors_by_hero(hero: str = Query(..., description="Name of the hero")):
    cursor.execute("SELECT T1.movie_title, T2.director FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T1.hero = ?", (hero,))
    result = cursor.fetchall()
    if not result:
        return {"movie_directors": []}
    return {"movie_directors": [{"movie_title": row[0], "director": row[1]} for row in result]}

# Endpoint to get hero, director, and release date by movie title
@app.get("/v1/disney/hero_director_release_date_by_movie", operation_id="get_hero_director_release_date_by_movie", summary="Retrieves the hero, director, and release date associated with a specific movie title. The provided movie title is used to search for corresponding records in the characters and director tables, which are then returned as a single result.")
async def get_hero_director_release_date_by_movie(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T1.hero, T2.director, T1.release_date FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T1.movie_title = ?", (movie_title,))
    result = cursor.fetchall()
    if not result:
        return {"hero_director_release_date": []}
    return {"hero_director_release_date": [{"hero": row[0], "director": row[1], "release_date": row[2]} for row in result]}

# Endpoint to get movie title, total gross, and MPAA rating by hero
@app.get("/v1/disney/movie_gross_rating_by_hero", operation_id="get_movie_gross_rating_by_hero", summary="Retrieves the title, total gross, and MPAA rating of movies featuring a specific hero. The hero's name is used to filter the results.")
async def get_movie_gross_rating_by_hero(hero: str = Query(..., description="Name of the hero")):
    cursor.execute("SELECT T1.movie_title, T1.total_gross, T1.MPAA_rating FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title INNER JOIN director AS T3 ON T3.name = T1.movie_title WHERE T2.hero = ?", (hero,))
    result = cursor.fetchall()
    if not result:
        return {"movie_gross_rating": []}
    return {"movie_gross_rating": [{"movie_title": row[0], "total_gross": row[1], "MPAA_rating": row[2]} for row in result]}

# Endpoint to get movie details based on voice actor
@app.get("/v1/disney/movie_details_by_voice_actor", operation_id="get_movie_details_by_voice_actor", summary="Retrieves detailed information about movies in which a specified voice actor has participated. The response includes the movie title, director, and release date. This operation is useful for obtaining comprehensive data on a voice actor's filmography.")
async def get_movie_details_by_voice_actor(voice_actor: str = Query(..., description="Voice actor's name")):
    cursor.execute("SELECT T1.movie, T3.director, T2.release_date FROM `voice-actors` AS T1 INNER JOIN characters AS T2 ON T1.movie = T2.movie_title INNER JOIN director AS T3 ON T3.name = T2.movie_title WHERE T1.`voice-actor` = ?", (voice_actor,))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": result}

# Endpoint to get movie titles and total gross based on voice actor and inflation-adjusted gross ratio
@app.get("/v1/disney/movie_gross_by_voice_actor_and_ratio", operation_id="get_movie_gross_by_voice_actor_and_ratio", summary="Retrieves movie titles and their total gross earnings for a specified voice actor, filtered by a given inflation-adjusted gross ratio. The ratio is calculated as the inflation-adjusted gross divided by the total gross.")
async def get_movie_gross_by_voice_actor_and_ratio(voice_actor: str = Query(..., description="Voice actor's name"), ratio: float = Query(..., description="Inflation-adjusted gross ratio")):
    cursor.execute("SELECT T1.movie_title, T1.total_gross FROM movies_total_gross AS T1 INNER JOIN `voice-actors` AS T2 ON T1.movie_title = T2.movie WHERE T2.`voice-actor` = ? AND CAST(REPLACE(trim(T1.inflation_adjusted_gross, '$'), ',', '') AS REAL) * 1.0 / CAST(REPLACE(trim(T1.total_gross, '$'), ',', '') AS REAL) * 1.0 < ?", (voice_actor, ratio))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": result}

# Endpoint to get the director of the movie with the highest total gross
@app.get("/v1/disney/top_grossing_movie_director", operation_id="get_top_grossing_movie_director", summary="Retrieves the director of the movie that has earned the highest total gross revenue. The operation fetches the movie with the highest total gross from the movies_total_gross table and then identifies the corresponding director from the director table using an inner join. The total gross is calculated by converting the gross revenue string to a real number, removing any commas and dollar signs.")
async def get_top_grossing_movie_director():
    cursor.execute("SELECT T2.director FROM movies_total_gross AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name ORDER BY CAST(REPLACE(trim(T1.total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"director": []}
    return {"director": result[0]}

# Endpoint to get voice actors and villains based on movie title
@app.get("/v1/disney/voice_actors_villains_by_movie", operation_id="get_voice_actors_villains_by_movie", summary="Retrieves the voice actors and their corresponding villains from a specific movie. The operation requires the movie title as an input parameter to filter the results.")
async def get_voice_actors_villains_by_movie(movie_title: str = Query(..., description="Movie title")):
    cursor.execute("SELECT T1.`voice-actor`, T2.villian FROM `voice-actors` AS T1 INNER JOIN characters AS T2 ON T1.movie = T2.movie_title WHERE T2.movie_title = ?", (movie_title,))
    result = cursor.fetchall()
    if not result:
        return {"voice_actors_villains": []}
    return {"voice_actors_villains": result}

# Endpoint to get voice actors based on movie title and character
@app.get("/v1/disney/voice_actors_by_movie_and_character", operation_id="get_voice_actors_by_movie_and_character", summary="Retrieves the voice actors associated with a specific character in a given Disney movie. The operation requires the movie title and character name as input parameters to accurately identify and return the corresponding voice actors.")
async def get_voice_actors_by_movie_and_character(movie_title: str = Query(..., description="Movie title"), character: str = Query(..., description="Character name")):
    cursor.execute("SELECT T1.`voice-actor` FROM `voice-actors` AS T1 INNER JOIN characters AS T2 ON T1.movie = T2.movie_title WHERE T2.movie_title = ? AND T1.character = ?", (movie_title, character))
    result = cursor.fetchall()
    if not result:
        return {"voice_actors": []}
    return {"voice_actors": result}

# Endpoint to get directors and MPAA ratings based on genre and release year
@app.get("/v1/disney/directors_mpaa_ratings_by_genre_year", operation_id="get_directors_mpaa_ratings_by_genre_year", summary="Retrieves a list of directors and their corresponding MPAA ratings for a specified genre and release year. The genre and release year are used to filter the results, providing a focused view of the data.")
async def get_directors_mpaa_ratings_by_genre_year(genre: str = Query(..., description="Genre of the movie"), release_year: str = Query(..., description="Release year in 'YYYY' format")):
    cursor.execute("SELECT T2.director, T1.MPAA_rating FROM movies_total_gross AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T1.genre = ? AND SUBSTR(T1.release_date, LENGTH(T1.release_date) - 3, LENGTH(T1.release_date)) = ?", (genre, release_year))
    result = cursor.fetchall()
    if not result:
        return {"directors_mpaa_ratings": []}
    return {"directors_mpaa_ratings": result}

# Endpoint to get the percentage of movies in a specific genre within a date range
@app.get("/v1/disney/genre_percentage_by_date_range", operation_id="get_genre_percentage_by_date_range", summary="Retrieve the percentage of movies belonging to a specific genre, along with their titles and directors, released within a specified date range. The genre, start year, and end year are required input parameters.")
async def get_genre_percentage_by_date_range(genre: str = Query(..., description="Genre of the movie"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.genre = ? THEN T1.movie_title ELSE NULL END) AS REAL) * 100 / COUNT(T1.movie_title), group_concat(T1.movie_title), group_concat(T2.director) FROM movies_total_gross AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE SUBSTR(T1.release_date, LENGTH(T1.release_date) - 3, LENGTH(T1.release_date)) BETWEEN ? AND ?", (genre, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"genre_percentage": []}
    return {"genre_percentage": result}

# Endpoint to get movie names and directors based on release date range and gross ratio
@app.get("/v1/disney/movies_directors_by_date_range_gross_ratio", operation_id="get_movies_directors_by_date_range_gross_ratio", summary="Retrieve the names and directors of movies that meet a specified gross ratio within a given release date range. The gross ratio is calculated as the movie's total gross divided by the average total gross of all movies within the same date range. The input parameters include the start and end years of the date range, as well as the desired gross ratio threshold.")
async def get_movies_directors_by_date_range_gross_ratio(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), gross_ratio: float = Query(..., description="Gross ratio")):
    cursor.execute("SELECT T2.name, T2.director FROM movies_total_gross AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE SUBSTR(T1.release_date, LENGTH(T1.release_date) - 3, LENGTH(T1.release_date)) BETWEEN ? AND ? AND CAST(REPLACE(trim(T1.total_gross, '$'), ',', '') AS REAL) / ( SELECT SUM(CAST(REPLACE(trim(T3.total_gross, '$'), ',', '') AS REAL)) / COUNT(T3.movie_title) AS avg_gross FROM movies_total_gross AS T3 INNER JOIN director AS T4 ON T3.movie_title = T4.name WHERE SUBSTR(T3.release_date, LENGTH(T3.release_date) - 3, LENGTH(T3.release_date)) BETWEEN ? AND ? ) - 1 > ?", (start_year, end_year, start_year, end_year, gross_ratio))
    result = cursor.fetchall()
    if not result:
        return {"movies_directors": []}
    return {"movies_directors": result}

# Endpoint to get the count of voice actors based on movie
@app.get("/v1/disney/voice_actor_count_by_movie", operation_id="get_voice_actor_count_by_movie", summary="Retrieves the total number of voice actors associated with a specific movie. The operation requires the movie title as input and returns the count of voice actors who have contributed to that movie.")
async def get_voice_actor_count_by_movie(movie: str = Query(..., description="Movie title")):
    cursor.execute("SELECT COUNT('voice-actor') FROM `voice-actors` WHERE movie = ?", (movie,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get movies by voice actor
@app.get("/v1/disney/movies_by_voice_actor", operation_id="get_movies_by_voice_actor", summary="Get movies by a specific voice actor")
async def get_movies_by_voice_actor(voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT movie FROM `voice-actors` WHERE `voice-actor` = ?", (voice_actor,))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": [row[0] for row in result]}

# Endpoint to get count of movies released between specific years
@app.get("/v1/disney/count_movies_released_between_years", operation_id="get_count_movies_released_between_years", summary="Retrieves the total number of Disney movies released between the specified years. The input parameters define the range of years to consider, using the last two digits of the release year.")
async def get_count_movies_released_between_years(start_year: str = Query(..., description="Start year (last two digits)"), end_year: str = Query(..., description="End year (last two digits)")):
    cursor.execute("SELECT COUNT(movie_title) FROM characters WHERE SUBSTR(release_date, LENGTH(release_date) - 1, LENGTH(release_date)) BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get songs by director
@app.get("/v1/disney/songs_by_director", operation_id="get_songs_by_director", summary="Retrieves a list of songs from movies directed by the specified director. The director's name is used to filter the results.")
async def get_songs_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T1.song FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T2.director = ?", (director,))
    result = cursor.fetchall()
    if not result:
        return {"songs": []}
    return {"songs": [row[0] for row in result]}

# Endpoint to get release date by director and movie title
@app.get("/v1/disney/release_date_by_director_and_movie_title", operation_id="get_release_date_by_director_and_movie_title", summary="Retrieves the release date of a movie directed by a specific director. The operation requires the director's name and the title of the movie as input parameters. It returns the release date of the movie if a match is found in the database.")
async def get_release_date_by_director_and_movie_title(director: str = Query(..., description="Name of the director"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T1.release_date FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T2.director = ? AND T1.movie_title = ?", (director, movie_title))
    result = cursor.fetchall()
    if not result:
        return {"release_dates": []}
    return {"release_dates": [row[0] for row in result]}

# Endpoint to get villains by voice actor
@app.get("/v1/disney/villains_by_voice_actor", operation_id="get_villains_by_voice_actor", summary="Retrieves a list of villains from Disney movies voiced by a specific voice actor. The voice actor's name is used to filter the results.")
async def get_villains_by_voice_actor(voice_actor: str = Query(..., description="Name of the voice actor")):
    cursor.execute("SELECT T1.villian FROM characters AS T1 INNER JOIN `voice-actors` AS T2 ON T1.movie_title = T2.movie WHERE T2.`voice-actor` = ?", (voice_actor,))
    result = cursor.fetchall()
    if not result:
        return {"villains": []}
    return {"villains": [row[0] for row in result]}

# Endpoint to get movies without villains by director
@app.get("/v1/disney/movies_without_villains_by_director", operation_id="get_movies_without_villains_by_director", summary="Retrieves a list of movies directed by a specific director that do not feature any villain characters. The director's name is provided as an input parameter.")
async def get_movies_without_villains_by_director(director: str = Query(..., description="Name of the director")):
    cursor.execute("SELECT T1.movie_title FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T2.director = ? AND T1.villian IS NULL", (director,))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": [row[0] for row in result]}

# Endpoint to get movies by director and release year
@app.get("/v1/disney/movies_by_director_and_release_year", operation_id="get_movies_by_director_and_release_year", summary="Retrieves a list of movies directed by a specific director and released before a certain year. The director's name and the last two digits of the release year are used as input parameters to filter the results. The response includes the titles of the movies that meet the specified criteria.")
async def get_movies_by_director_and_release_year(director: str = Query(..., description="Name of the director"), release_year: str = Query(..., description="Release year (last two digits)")):
    cursor.execute("SELECT T1.movie_title FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T2.director = ? AND SUBSTR(T1.release_date, LENGTH(T1.release_date) - 1, LENGTH(T1.release_date)) < ?", (director, release_year))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": [row[0] for row in result]}

# Endpoint to get directors of high-grossing movies
@app.get("/v1/disney/directors_of_high_grossing_movies", operation_id="get_directors_of_high_grossing_movies", summary="Retrieve the unique list of directors who have directed movies that have grossed more than the specified total gross amount. The total gross amount is provided in dollars and does not include commas or dollar signs.")
async def get_directors_of_high_grossing_movies(total_gross: float = Query(..., description="Total gross amount (in dollars)")):
    cursor.execute("SELECT DISTINCT T2.director FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name INNER JOIN movies_total_gross AS T3 ON T1.movie_title = T3.movie_title WHERE CAST(REPLACE(trim(T3.total_gross, '$'), ',', '') AS REAL) > ?", (total_gross,))
    result = cursor.fetchall()
    if not result:
        return {"directors": []}
    return {"directors": [row[0] for row in result]}

# Endpoint to get the song from the highest-grossing movie
@app.get("/v1/disney/song_from_highest_grossing_movie", operation_id="get_song_from_highest_grossing_movie", summary="Retrieves the song from the movie that has earned the highest total gross revenue. The song is obtained from the characters table, which is joined with the movies_total_gross table based on the movie title. The total gross revenue is converted to a numeric value for accurate comparison, and the result is ordered in descending order to identify the highest-grossing movie. The song from this movie is then returned.")
async def get_song_from_highest_grossing_movie():
    cursor.execute("SELECT T2.song FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title ORDER BY CAST(REPLACE(trim(T1.total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"song": []}
    return {"song": result[0]}

# Endpoint to get the director of the highest grossing movie within a specified release date range
@app.get("/v1/disney/director_highest_grossing_movie", operation_id="get_director_highest_grossing_movie", summary="Retrieve the director of the highest grossing movie released within a specified date range. The date range is determined by the provided start and end years. The result is based on the total gross earnings of the movies, with the highest earning movie's director being returned.")
async def get_director_highest_grossing_movie(start_year: str = Query(..., description="Start year of the release date range (e.g., '1937')"), end_year: str = Query(..., description="End year of the release date range (e.g., '1990')")):
    cursor.execute("SELECT T2.director FROM characters AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name INNER JOIN movies_total_gross AS T3 ON T3.movie_title = T1.movie_title WHERE SUBSTR(T3.release_date, LENGTH(T3.release_date) - 3, LENGTH(T3.release_date)) BETWEEN ? AND ? ORDER BY CAST(REPLACE(trim(T3.total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"director": []}
    return {"director": result[0]}

# Endpoint to get the hero from movies of a specific genre
@app.get("/v1/disney/hero_by_genre", operation_id="get_hero_by_genre", summary="Retrieves the main hero character from movies belonging to a specified genre. The genre is provided as an input parameter, and the operation returns the hero's name.")
async def get_hero_by_genre(genre: str = Query(..., description="Genre of the movie (e.g., 'Comedy')")):
    cursor.execute("SELECT T2.hero FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title WHERE T1.genre = ?", (genre,))
    result = cursor.fetchall()
    if not result:
        return {"heroes": []}
    return {"heroes": [row[0] for row in result]}

# Endpoint to get the hero and voice actor for movies directed by a specific director
@app.get("/v1/disney/hero_voice_actor_by_director", operation_id="get_hero_voice_actor_by_director", summary="Retrieves the hero and corresponding voice actor for movies directed by a specified director. The director's name is used to filter the results.")
async def get_hero_voice_actor_by_director(director: str = Query(..., description="Director of the movie (e.g., 'Wolfgang Reitherman')")):
    cursor.execute("SELECT T2.hero, T1.`voice-actor` FROM `voice-actors` AS T1 INNER JOIN characters AS T2 ON T1.movie = T2.movie_title INNER JOIN director AS T3 ON T3.name = T2.movie_title WHERE T3.director = ?", (director,))
    result = cursor.fetchall()
    if not result:
        return {"hero_voice_actors": []}
    return {"hero_voice_actors": [{"hero": row[0], "voice_actor": row[1]} for row in result]}

# Endpoint to get the genre of movies featuring a specific hero
@app.get("/v1/disney/genre_by_hero", operation_id="get_genre_by_hero", summary="Retrieves the genre of movies in which a specified hero appears. The hero's name is used to search for corresponding movies and their associated genres.")
async def get_genre_by_hero(hero: str = Query(..., description="Hero of the movie (e.g., 'Taran')")):
    cursor.execute("SELECT T1.genre FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title WHERE T2.hero = ?", (hero,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the voice actor and director for movies featuring a specific hero
@app.get("/v1/disney/voice_actor_director_by_hero", operation_id="get_voice_actor_director_by_hero", summary="Retrieves the voice actor and director associated with movies featuring a specified hero. The hero parameter is used to filter the results and must be provided in the request.")
async def get_voice_actor_director_by_hero(hero: str = Query(..., description="Hero of the movie (e.g., 'Elsa')")):
    cursor.execute("SELECT T1.`voice-actor`, T3.director FROM `voice-actors` AS T1 INNER JOIN characters AS T2 ON T1.movie = T2.movie_title INNER JOIN director AS T3 ON T2.movie_title = T3.name WHERE T2.hero = ?", (hero,))
    result = cursor.fetchall()
    if not result:
        return {"voice_actor_directors": []}
    return {"voice_actor_directors": [{"voice_actor": row[0], "director": row[1]} for row in result]}

# Endpoint to get the percentage of directors with movies grossing over a specified amount
@app.get("/v1/disney/percentage_directors_high_grossing_movies", operation_id="get_percentage_directors_high_grossing_movies", summary="Retrieves the percentage of directors who have directed at least one movie that grossed more than the specified minimum amount. The calculation is based on the total gross earnings of each movie and the distinct count of directors.")
async def get_percentage_directors_high_grossing_movies(min_gross: int = Query(..., description="Minimum gross amount (e.g., 100000000)")):
    cursor.execute("SELECT CAST(COUNT(DISTINCT CASE WHEN CAST(REPLACE(trim(T1.total_gross, '$'), ',', '') AS REAL) > ? THEN T3.director ELSE NULL END) AS REAL) * 100 / COUNT(DISTINCT T3.director) FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title INNER JOIN director AS T3 ON T1.movie_title = T3.name", (min_gross,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of voice actors in movies of a specific genre
@app.get("/v1/disney/percentage_voice_actors_by_genre", operation_id="get_percentage_voice_actors_by_genre", summary="Retrieves the percentage of voice actors who have contributed to movies of a specified genre. This operation calculates the ratio of voice actors in the given genre to the total number of voice actors across all movies. The genre is provided as an input parameter.")
async def get_percentage_voice_actors_by_genre(genre: str = Query(..., description="Genre of the movie (e.g., 'Drama')")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.genre = ? THEN T3.`voice-actor` ELSE NULL END) AS REAL) * 100 / COUNT(T3.`voice-actor`) FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title INNER JOIN `voice-actors` AS T3 ON T3.movie = T1.movie_title", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the movie title with the earliest release year
@app.get("/v1/disney/earliest_movie_title", operation_id="get_earliest_movie_title", summary="Retrieves the title of the Disney movie with the earliest release year. The operation sorts the movies by the last two digits of their release year in ascending order and returns the title of the first movie in the sorted list.")
async def get_earliest_movie_title():
    cursor.execute("SELECT movie_title FROM characters ORDER BY SUBSTR(release_date, LENGTH(release_date) - 1, LENGTH(release_date)) ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the villain from the movie with the latest release year
@app.get("/v1/disney/latest_villain", operation_id="get_latest_villain", summary="Retrieves the villain from the most recently released Disney movie. The operation sorts the movies by their release year and selects the villain from the latest one.")
async def get_latest_villain():
    cursor.execute("SELECT villian FROM characters ORDER BY SUBSTR(release_date, LENGTH(release_date) - 1, LENGTH(release_date)) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"villain": []}
    return {"villain": result[0]}

# Endpoint to get the top movie title by total gross in a specific genre
@app.get("/v1/disney/top_movie_by_genre", operation_id="get_top_movie_by_genre", summary="Retrieves the title of the highest-grossing movie in a specified genre. The genre is provided as an input parameter, and the movie title is determined by the total gross earnings, with the highest-earning movie returned.")
async def get_top_movie_by_genre(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT movie_title FROM movies_total_gross WHERE genre = ? ORDER BY CAST(REPLACE(trim(total_gross, '$'), ',', '') AS REAL) DESC LIMIT 1", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"movie_title": []}
    return {"movie_title": result[0]}

# Endpoint to get the total revenue between two years
@app.get("/v1/disney/total_revenue_between_years", operation_id="get_total_revenue_between_years", summary="Retrieves the total revenue generated between the specified start and end years, inclusive. The operation calculates the sum of all revenues within the given year range.")
async def get_total_revenue_between_years(start_year: int = Query(..., description="Start year (inclusive)"), end_year: int = Query(..., description="End year (inclusive)")):
    cursor.execute("SELECT SUM(Total) FROM revenue WHERE `Year` BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"total_revenue": []}
    return {"total_revenue": result[0]}

# Endpoint to get the top hero by inflation-adjusted gross in a specific genre
@app.get("/v1/disney/top_hero_by_genre", operation_id="get_top_hero_by_genre", summary="Retrieves the top hero by inflation-adjusted gross in a specified genre. The genre is provided as an input parameter. The hero is determined by selecting the movie with the highest inflation-adjusted gross in the given genre and then identifying the main character from that movie.")
async def get_top_hero_by_genre(genre: str = Query(..., description="Genre of the movie")):
    cursor.execute("SELECT T2.hero FROM movies_total_gross AS T1 INNER JOIN characters AS T2 ON T1.movie_title = T2.movie_title WHERE T1.genre = ? ORDER BY CAST(REPLACE(trim(T1.inflation_adjusted_gross, '$'), ',', '') AS REAL) DESC LIMIT 1", (genre,))
    result = cursor.fetchone()
    if not result:
        return {"hero": []}
    return {"hero": result[0]}

# Endpoint to get the director of the movie with the lowest total gross
@app.get("/v1/disney/director_lowest_total_gross", operation_id="get_director_lowest_total_gross", summary="Retrieves the director of the movie with the lowest total gross from the Disney catalog. The operation sorts the movies by their total gross in ascending order and returns the director of the movie with the lowest value.")
async def get_director_lowest_total_gross():
    cursor.execute("SELECT T2.director FROM movies_total_gross AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name ORDER BY CAST(REPLACE(trim(T1.total_gross, '$'), ',', '') AS REAL) ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"director": []}
    return {"director": result[0]}

# Endpoint to get the percentage of revenue from Walt Disney Parks and Resorts in a specific year
@app.get("/v1/disney/revenue_percentage_parks_resorts", operation_id="get_revenue_percentage_parks_resorts", summary="Retrieves the proportion of total revenue generated by Walt Disney Parks and Resorts in a specified year. The calculation is based on the sum of revenue from Walt Disney Parks and Resorts divided by the total revenue for the given year, expressed as a percentage.")
async def get_revenue_percentage_parks_resorts(year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT SUM(`Walt Disney Parks and Resorts`) / SUM(Total) * 100 FROM revenue WHERE year = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average total gross for movies with a specific MPAA rating
@app.get("/v1/disney/average_total_gross_by_mpaa_rating", operation_id="get_average_total_gross_by_mpaa_rating", summary="Retrieves the average total gross revenue for Disney movies with a specified MPAA rating. The calculation is based on the sum of total gross revenue divided by the count of movies with the given rating.")
async def get_average_total_gross_by_mpaa_rating(mpaa_rating: str = Query(..., description="MPAA rating of the movie")):
    cursor.execute("SELECT SUM(CAST(REPLACE(trim(total_gross, '$'), ',', '') AS REAL)) / COUNT(movie_title) FROM movies_total_gross WHERE MPAA_rating = ?", (mpaa_rating,))
    result = cursor.fetchone()
    if not result:
        return {"average_total_gross": []}
    return {"average_total_gross": result[0]}

# Endpoint to get the count of distinct voice actors in a specific movie
@app.get("/v1/disney/count_distinct_voice_actors_by_movie", operation_id="get_count_distinct_voice_actors_by_movie", summary="Get the count of distinct voice actors in a specific movie")
async def get_count_distinct_voice_actors_by_movie(movie: str = Query(..., description="Movie title")):
    cursor.execute("SELECT COUNT(DISTINCT `voice-actor`) FROM `voice-actors` WHERE movie = ?", (movie,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of inflation-adjusted gross to total gross for movies released in a specific year
@app.get("/v1/disney/gross_ratio_by_year", operation_id="get_gross_ratio_by_year", summary="Retrieves the ratio of inflation-adjusted gross to total gross for movies released in a specific year. The year is provided as a parameter, allowing for a targeted calculation of the ratio. This operation is useful for analyzing the financial performance of movies in a given year, taking into account inflation.")
async def get_gross_ratio_by_year(year: str = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT SUM(CAST(REPLACE(trim(inflation_adjusted_gross, '$'), ',', '') AS REAL)) / SUM(CAST(REPLACE(trim(total_gross, '$'), ',', '') AS REAL)) FROM movies_total_gross WHERE SUBSTR(release_date, LENGTH(release_date) - 3, LENGTH(release_date)) = ? GROUP BY SUBSTR(release_date, LENGTH(release_date) - 3, LENGTH(release_date))", (year,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the inflation-adjusted gross for two specific movies
@app.get("/v1/disney/inflation_adjusted_gross_for_movies", operation_id="get_inflation_adjusted_gross_for_movies", summary="Retrieves the total inflation-adjusted gross for two specified movies from the Disney catalog. The endpoint accepts the titles of the movies as input parameters and returns the respective inflation-adjusted gross amounts.")
async def get_inflation_adjusted_gross_for_movies(movie1: str = Query(..., description="First movie title"), movie2: str = Query(..., description="Second movie title")):
    cursor.execute("SELECT SUM(CASE WHEN movie_title = ? THEN CAST(REPLACE(trim(inflation_adjusted_gross, '$'), ',', '') AS REAL) ELSE 0 END), SUM(CASE WHEN movie_title = ? THEN CAST(REPLACE(trim(inflation_adjusted_gross, '$'), ',', '') AS REAL) ELSE 0 END) FROM movies_total_gross", (movie1, movie2))
    result = cursor.fetchone()
    if not result:
        return {"inflation_adjusted_gross": []}
    return {"inflation_adjusted_gross": {"movie1": result[0], "movie2": result[1]}}

# Endpoint to get movie details based on director
@app.get("/v1/disney/movie_details_by_director", operation_id="get_movie_details_by_director", summary="Retrieves detailed information about movies directed by a specific director. The response includes the movie title, latest release date, and highest inflation-adjusted gross. The director's name is used to filter the results.")
async def get_movie_details_by_director(director: str = Query(..., description="Director of the movie")):
    cursor.execute("SELECT T1.movie_title, MAX(T1.release_date), MAX(T1.inflation_adjusted_gross) FROM movies_total_gross AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T2.director = ?", (director,))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": result}

# Endpoint to get voice actors who have acted in more than a specified number of movies
@app.get("/v1/disney/voice_actors_by_movie_count", operation_id="get_voice_actors_by_movie_count", summary="Retrieves a list of voice actors who have contributed to more than the specified number of movies. This operation allows you to filter voice actors based on their movie participation count, providing a targeted list of prolific voice actors.")
async def get_voice_actors_by_movie_count(movie_count: int = Query(..., description="Number of movies the voice actor has acted in")):
    cursor.execute("SELECT 'voice-actor' FROM `voice-actors` GROUP BY 'voice-actor' HAVING COUNT(movie) > ?", (movie_count,))
    result = cursor.fetchall()
    if not result:
        return {"voice_actors": []}
    return {"voice_actors": result}

# Endpoint to get the percentage of total gross exceeding a specified amount
@app.get("/v1/disney/percentage_total_gross_exceeding", operation_id="get_percentage_total_gross_exceeding", summary="Retrieves the percentage of movies whose inflation-adjusted gross exceeds a specified amount. The calculation is based on the total inflation-adjusted gross of all movies in the dataset.")
async def get_percentage_total_gross_exceeding(gross_amount: float = Query(..., description="Gross amount to compare against")):
    cursor.execute("SELECT SUM(CASE WHEN CAST(REPLACE(trim(inflation_adjusted_gross, '$'), ',', '') AS REAL) > ? THEN CAST(REPLACE(trim(inflation_adjusted_gross, '$'), ',', '') AS REAL) ELSE 0 END) * 100 / SUM(CAST(REPLACE(trim(inflation_adjusted_gross, '$'), ',', '') AS REAL)) FROM movies_total_gross", (gross_amount,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of movies exceeding a specified gross amount directed by a specific director
@app.get("/v1/disney/percentage_movies_exceeding_gross_by_director", operation_id="get_percentage_movies_exceeding_gross_by_director", summary="Retrieves the percentage of movies directed by a specific director that have earned more than a given gross amount. The calculation is based on the total gross earnings of each movie, excluding any commas or dollar signs. The result is expressed as a percentage of the total number of movies directed by the specified director.")
async def get_percentage_movies_exceeding_gross_by_director(gross_amount: float = Query(..., description="Gross amount to compare against"), director: str = Query(..., description="Director of the movie")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN CAST(REPLACE(trim(T1.total_gross, '$'), ',', '') AS REAL) > ? THEN T1.movie_title ELSE NULL END) AS REAL) * 100 / COUNT(T1.movie_title) FROM movies_total_gross AS T1 INNER JOIN director AS T2 ON T1.movie_title = T2.name WHERE T2.director = ?", (gross_amount, director))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/disney/revenue_difference?year1=1998&year2=1997",
    "/v1/disney/compare_segments?segment1=Studio%20Entertainment[NI%201]&segment2=Disney%20Media%20Networks&year=1998",
    "/v1/disney/director_by_movie?name=Pinocchio",
    "/v1/disney/villains_by_director?director=Wolfgang%20Reitherman",
    "/v1/disney/movie_count_by_month_director?month=Dec&director=Wolfgang%20Reitherman",
    "/v1/disney/director_by_song?song=Once%20Upon%20a%20Dream",
    "/v1/disney/voice_actors_by_movie?character=%25&villain=%25&movie_title=Alice%20in%20Wonderland",
    "/v1/disney/release_dates_by_voice_actor?voice_actor=Alan%20Tudyk",
    "/v1/disney/movie_count_by_voice_actor_day?voice_actor=Alan%20Tudyk&day=12",
    "/v1/disney/movie_count_by_director_genre?director=Wolfgang%20Reitherman&genre=Comedy",
    "/v1/disney/highest_grossing_movie_by_director?director=Wolfgang%20Reitherman",
    "/v1/disney/movies_by_rating_and_director?mpaa_rating=G&director=Wolfgang%20Reitherman",
    "/v1/disney/villain_from_highest_grossing_movie",
    "/v1/disney/genre_by_villain?villain=Commander%20Rourke",
    "/v1/disney/villain_by_movie?movie_title=Beauty%20and%20the%20Beast",
    "/v1/disney/movies_by_hero?hero=Robin%20Hood",
    "/v1/disney/movies_by_song?song=I%20Thought%20I%20Lost%20You",
    "/v1/disney/voice_actor_by_character?character=Binkie%20Muddlefoot",
    "/v1/disney/hero_by_total_gross?total_gross=%24222%2C527%2C828",
    "/v1/disney/top_song_by_decade?start_year=1970&end_year=1979",
    "/v1/disney/hero_by_director?director=Will%20Finn",
    "/v1/disney/voice_actor_by_movie_and_hero?movie_title=The%20Little%20Mermaid&hero=Ariel",
    "/v1/disney/director_by_character_and_voice_actor?character=Aunt%20Sarah&voice_actor=Verna%20Felton",
    "/v1/disney/release_date_by_character_and_voice_actor?character=Hyacinth%20Hippo&voice_actor=Tress%20MacNeille",
    "/v1/disney/director_by_genre_and_release_date?genre=Adventure&release_date=Mar%2030,%202007",
    "/v1/disney/hero_by_genre_and_release_date?genre=Adventure&release_date=4-Mar-16",
    "/v1/disney/highest_grossing_movie_by_hero?hero=Donald%20Duck",
    "/v1/disney/count_movies_by_director?director=Wolfgang%20Reitherman",
    "/v1/disney/top_director",
    "/v1/disney/movie_count_by_rating_genre_release_date?mpaa_rating=R&genre=Horror&start_year=1990&end_year=2015",
    "/v1/disney/characters_by_voice_actor?voice_actor=Frank%20Welker",
    "/v1/disney/total_gross_by_song?song=Little%20Wonders",
    "/v1/disney/mpaa_rating_by_villain?villain=Turbo",
    "/v1/disney/movie_count_by_rating_voice_actor?mpaa_rating=PG&voice_actor=Bill%20Thompson",
    "/v1/disney/top_director_by_voice_actor?voice_actor=Bill%20Thompson",
    "/v1/disney/top_movie_by_director?director=Ron%20Clements",
    "/v1/disney/voice_actors_by_director_release_date?director=Ben%20Sharpsteen&release_date=Feb%209,%201940",
    "/v1/disney/movie_count_by_director_rating_genre?director=Ron%20Clements&mpaa_rating=PG&genre=Adventure",
    "/v1/disney/movie_count_by_genre?genre=Horror",
    "/v1/disney/top_revenue_year?start_year=2000&end_year=2010",
    "/v1/disney/songs_by_genre?genre=Drama",
    "/v1/disney/voice_actors_for_heroes",
    "/v1/disney/directors_by_year_range?start_year=1990&end_year=2000",
    "/v1/disney/voice_actors_for_villains?movie_title=The%20Rescuers",
    "/v1/disney/movies_and_voice_actors_by_director?director=Wolfgang%20Reitherman",
    "/v1/disney/characters_by_mpaa_rating?mpaa_rating=PG",
    "/v1/disney/highest_gross_movie_without_song",
    "/v1/disney/director_with_most_voice_actors",
    "/v1/disney/voice_actors_by_release_date?release_date=Nov%2024,%202010",
    "/v1/disney/directors_with_songs",
    "/v1/disney/highest_gross_movie_by_voice_actor?voice_actor=Jim%20Cummings",
    "/v1/disney/highest_gross_movie_by_director?director=Ron%20Clements",
    "/v1/disney/average_gross_by_voice_actor?voice_actor=Sterling%20Holloway",
    "/v1/disney/percentage_movies_with_songs",
    "/v1/disney/movies_by_year?year=2016",
    "/v1/disney/director_name?director=Jack%20Kinney",
    "/v1/disney/highest_grossing_movie",
    "/v1/disney/movies_by_rating_genre?mpaa_rating=PG-13&genre=Romantic%20Comedy",
    "/v1/disney/movies_characters_by_voice_actor?voice_actor=Bill%20Thompson",
    "/v1/disney/movie_songs_by_director?director=Ron%20Clements",
    "/v1/disney/movie_heroes_songs_by_director_year?director=Wolfgang%20Reitherman&release_year=1977",
    "/v1/disney/movie_directors_by_hero?hero=Donald%20Duck",
    "/v1/disney/hero_director_release_date_by_movie?movie_title=Mulan",
    "/v1/disney/movie_gross_rating_by_hero?hero=Elsa",
    "/v1/disney/movie_details_by_voice_actor?voice_actor=Freddie%20Jones",
    "/v1/disney/movie_gross_by_voice_actor_and_ratio?voice_actor=Frank%20Welker&ratio=2",
    "/v1/disney/top_grossing_movie_director",
    "/v1/disney/voice_actors_villains_by_movie?movie_title=Cinderella",
    "/v1/disney/voice_actors_by_movie_and_character?movie_title=Lion%20King&character=Lion%20King",
    "/v1/disney/directors_mpaa_ratings_by_genre_year?genre=Musical&release_year=1993",
    "/v1/disney/genre_percentage_by_date_range?genre=Comedy&start_year=1991&end_year=2000",
    "/v1/disney/movies_directors_by_date_range_gross_ratio?start_year=2001&end_year=2005&gross_ratio=1",
    "/v1/disney/voice_actor_count_by_movie?movie=Aladdin",
    "/v1/disney/movies_by_voice_actor?voice_actor=Jeff%20Bennett",
    "/v1/disney/count_movies_released_between_years?start_year=37&end_year=50",
    "/v1/disney/songs_by_director?director=Ben%20Sharpsteen",
    "/v1/disney/release_date_by_director_and_movie_title?director=Roger%20Allers&movie_title=The%20Lion%20King",
    "/v1/disney/villains_by_voice_actor?voice_actor=Scott%20Weinger%20Brad%20Kane",
    "/v1/disney/movies_without_villains_by_director?director=Wolfgang%20Reitherman",
    "/v1/disney/movies_by_director_and_release_year?director=Jack%20Kinney&release_year=47",
    "/v1/disney/directors_of_high_grossing_movies?total_gross=100000000",
    "/v1/disney/song_from_highest_grossing_movie",
    "/v1/disney/director_highest_grossing_movie?start_year=1937&end_year=1990",
    "/v1/disney/hero_by_genre?genre=Comedy",
    "/v1/disney/hero_voice_actor_by_director?director=Wolfgang%20Reitherman",
    "/v1/disney/genre_by_hero?hero=Taran",
    "/v1/disney/voice_actor_director_by_hero?hero=Elsa",
    "/v1/disney/percentage_directors_high_grossing_movies?min_gross=100000000",
    "/v1/disney/percentage_voice_actors_by_genre?genre=Drama",
    "/v1/disney/earliest_movie_title",
    "/v1/disney/latest_villain",
    "/v1/disney/top_movie_by_genre?genre=Action",
    "/v1/disney/total_revenue_between_years?start_year=2010&end_year=2016",
    "/v1/disney/top_hero_by_genre?genre=Adventure",
    "/v1/disney/director_lowest_total_gross",
    "/v1/disney/revenue_percentage_parks_resorts?year=2010",
    "/v1/disney/average_total_gross_by_mpaa_rating?mpaa_rating=PG-13",
    "/v1/disney/count_distinct_voice_actors_by_movie?movie=Bambi",
    "/v1/disney/gross_ratio_by_year?year=1995",
    "/v1/disney/inflation_adjusted_gross_for_movies?movie1=Cars&movie2=Cars%202",
    "/v1/disney/movie_details_by_director?director=Chris%20Buck",
    "/v1/disney/voice_actors_by_movie_count?movie_count=5",
    "/v1/disney/percentage_total_gross_exceeding?gross_amount=1236035515",
    "/v1/disney/percentage_movies_exceeding_gross_by_director?gross_amount=100000000&director=Gary%20Trousdale"
]
