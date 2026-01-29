from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/video_games/video_games.sqlite')
cursor = conn.cursor()

# Endpoint to get the difference in sales between two game platforms in a specific region
@app.get("/v1/video_games/sales_difference_by_platform_region", operation_id="get_sales_difference", summary="Retrieve the difference in sales, in units of 100,000, between two specified game platforms within a given region. This operation calculates the total sales for each platform and subtracts the lower value from the higher one, providing a clear comparison of their performance in the selected region.")
async def get_sales_difference(platform_id_1: int = Query(..., description="First game platform ID"), platform_id_2: int = Query(..., description="Second game platform ID"), region_id: int = Query(..., description="Region ID")):
    cursor.execute("SELECT (SUM(CASE WHEN T.game_platform_id = ? THEN T.num_sales ELSE 0 END) - SUM(CASE WHEN T.game_platform_id = ? THEN T.num_sales ELSE 0 END)) * 100000 AS nums FROM region_sales AS T WHERE T.region_id = ?", (platform_id_1, platform_id_2, region_id))
    result = cursor.fetchone()
    if not result:
        return {"nums": []}
    return {"nums": result[0]}

# Endpoint to get games with the same genre as a specified game
@app.get("/v1/video_games/games_by_genre_of_game", operation_id="get_games_by_genre", summary="Retrieves a list of video games that share the same genre as the specified game. The genre is determined by the provided game name.")
async def get_games_by_genre(game_name: str = Query(..., description="Name of the game to match genre")):
    cursor.execute("SELECT T1.game_name FROM game AS T1 WHERE T1.genre_id = ( SELECT T.genre_id FROM game AS T WHERE T.game_name = ? )", (game_name,))
    result = cursor.fetchall()
    if not result:
        return {"games": []}
    return {"games": [row[0] for row in result]}

# Endpoint to get the count of games in a specific genre
@app.get("/v1/video_games/count_games_by_genre", operation_id="get_count_games_by_genre", summary="Retrieves the total number of video games that belong to a specified genre. The genre is identified by its name.")
async def get_count_games_by_genre(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT COUNT(T1.id) FROM game AS T1 INNER JOIN genre AS T2 ON T1.genre_id = T2.id WHERE T2.genre_name = ?", (genre_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the genre of a specific game
@app.get("/v1/video_games/genre_of_game", operation_id="get_genre_of_game", summary="Retrieves the genre of a specific video game. The operation requires the name of the game as input and returns the genre associated with it.")
async def get_genre_of_game(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T2.genre_name FROM game AS T1 INNER JOIN genre AS T2 ON T1.genre_id = T2.id WHERE T1.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"genre": []}
    return {"genre": result[0]}

# Endpoint to get the publisher of a specific game
@app.get("/v1/video_games/publisher_of_game", operation_id="get_publisher_of_game", summary="Retrieves the publisher of a specific video game. The operation requires the name of the game as input and returns the name of the publisher associated with the game.")
async def get_publisher_of_game(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T3.publisher_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id WHERE T1.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"publisher": []}
    return {"publisher": result[0]}

# Endpoint to get games published by a specific publisher
@app.get("/v1/video_games/games_by_publisher", operation_id="get_games_by_publisher", summary="Retrieves a list of video games published by the specified publisher. The publisher is identified by its name, which is provided as an input parameter.")
async def get_games_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT T1.game_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id WHERE T3.publisher_name = ?", (publisher_name,))
    result = cursor.fetchall()
    if not result:
        return {"games": []}
    return {"games": [row[0] for row in result]}

# Endpoint to get the count of games in a specific genre published by a specific publisher
@app.get("/v1/video_games/count_games_by_genre_and_publisher", operation_id="get_count_games_by_genre_and_publisher", summary="Retrieve the total number of video games that belong to a specific genre and were published by a particular publisher.")
async def get_count_games_by_genre_and_publisher(genre_name: str = Query(..., description="Name of the genre"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT COUNT(T1.id) FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id INNER JOIN genre AS T4 ON T1.genre_id = T4.id WHERE T4.genre_name = ? AND T3.publisher_name = ?", (genre_name, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get games in a specific genre published by a specific publisher
@app.get("/v1/video_games/games_by_genre_and_publisher", operation_id="get_games_by_genre_and_publisher", summary="Retrieves a list of video games that belong to a specified genre and are published by a specific publisher. The operation requires the name of the publisher and the genre as input parameters.")
async def get_games_by_genre_and_publisher(publisher_name: str = Query(..., description="Name of the publisher"), genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT T1.game_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id INNER JOIN genre AS T4 ON T1.genre_id = T4.id WHERE T3.publisher_name = ? AND T4.genre_name = ?", (publisher_name, genre_name))
    result = cursor.fetchall()
    if not result:
        return {"games": []}
    return {"games": [row[0] for row in result]}

# Endpoint to get the publisher with the most games
@app.get("/v1/video_games/top_publisher", operation_id="get_top_publisher", summary="Retrieves the name of the publisher with the highest number of distinct video games. This operation identifies the publisher with the most games by counting the unique game IDs associated with each publisher and then selecting the top publisher based on this count.")
async def get_top_publisher():
    cursor.execute("SELECT T.publisher_name FROM ( SELECT T2.publisher_name, COUNT(DISTINCT T2.id) FROM game_publisher AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id GROUP BY T1.publisher_id ORDER BY COUNT(T2.id) DESC LIMIT 1 ) t")
    result = cursor.fetchone()
    if not result:
        return {"publisher": []}
    return {"publisher": result[0]}

# Endpoint to get the platform of a specific game released in a specific year
@app.get("/v1/video_games/platform_of_game_by_year", operation_id="get_platform_of_game_by_year", summary="Retrieves the platform on which a specific video game was released in a given year. The operation requires the name of the game and its release year as input parameters. It returns the name of the platform associated with the specified game and year.")
async def get_platform_of_game_by_year(game_name: str = Query(..., description="Name of the game"), release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT T5.platform_name FROM game_publisher AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN game AS T3 ON T1.game_id = T3.id INNER JOIN game_platform AS T4 ON T1.id = T4.game_publisher_id INNER JOIN platform AS T5 ON T4.platform_id = T5.id WHERE T3.game_name = ? AND T4.release_year = ?", (game_name, release_year))
    result = cursor.fetchone()
    if not result:
        return {"platform": []}
    return {"platform": result[0]}

# Endpoint to get the release year of a game on a specific platform
@app.get("/v1/video_games/release_year_by_game_and_platform", operation_id="get_release_year", summary="Retrieves the release year of a specific video game on a given platform. The operation requires the game's name and the platform's name as input parameters to accurately locate the release year in the database.")
async def get_release_year(game_name: str = Query(..., description="Name of the game"), platform_name: str = Query(..., description="Name of the platform")):
    cursor.execute("SELECT T4.release_year FROM game_publisher AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN game AS T3 ON T1.game_id = T3.id INNER JOIN game_platform AS T4 ON T1.id = T4.game_publisher_id INNER JOIN platform AS T5 ON T4.platform_id = T5.id WHERE T3.game_name = ? AND T5.platform_name = ?", (game_name, platform_name))
    result = cursor.fetchone()
    if not result:
        return {"release_year": []}
    return {"release_year": result[0]}

# Endpoint to get distinct publisher names for a specific genre
@app.get("/v1/video_games/distinct_publishers_by_genre", operation_id="get_distinct_publishers", summary="Retrieves a list of unique publisher names associated with a specific video game genre. The genre is specified as an input parameter.")
async def get_distinct_publishers(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT DISTINCT T3.publisher_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id INNER JOIN genre AS T4 ON T1.genre_id = T4.id WHERE T4.genre_name = ?", (genre_name,))
    result = cursor.fetchall()
    if not result:
        return {"publishers": []}
    return {"publishers": [row[0] for row in result]}

# Endpoint to get the count of publishers with more than a specified number of games in a specific genre
@app.get("/v1/video_games/count_publishers_by_genre_and_game_count", operation_id="get_count_publishers", summary="Retrieve the number of publishers who have published more than a specified number of games in a given genre. This operation requires the genre name and the minimum number of games as input parameters.")
async def get_count_publishers(genre_name: str = Query(..., description="Name of the genre"), min_game_count: int = Query(..., description="Minimum number of games")):
    cursor.execute("SELECT COUNT(T.publisher_name) FROM ( SELECT T3.publisher_name, COUNT(DISTINCT T1.id) FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id INNER JOIN genre AS T4 ON T1.genre_id = T4.id WHERE T4.genre_name = ? GROUP BY T3.publisher_name HAVING COUNT(DISTINCT T1.id) > ? ) t", (genre_name, min_game_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of games in a specific genre for a given publisher
@app.get("/v1/video_games/percentage_games_by_genre_and_publisher", operation_id="get_percentage_games", summary="Retrieves the percentage of games in a specific genre that a given publisher has released. The operation calculates this percentage by counting the number of games in the specified genre that the publisher has released and dividing it by the total number of games released by the publisher.")
async def get_percentage_games(genre_name: str = Query(..., description="Name of the genre"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T4.genre_name = ? THEN T1.id ELSE NULL END) AS REAL) * 100/ COUNT(T1.id) FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id INNER JOIN genre AS T4 ON T1.genre_id = T4.id WHERE T3.publisher_name = ?", (genre_name, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total sales for a specific platform
@app.get("/v1/video_games/total_sales_by_platform", operation_id="get_total_sales", summary="Retrieves the total sales volume for a specified video game platform, calculated as the sum of all sales multiplied by 100,000 and divided by 4. The platform is identified by its name.")
async def get_total_sales(platform_name: str = Query(..., description="Name of the platform")):
    cursor.execute("SELECT SUM(T1.num_sales) * 100000 / 4 FROM region_sales AS T1 INNER JOIN game_platform AS T2 ON T1.game_platform_id = T2.id INNER JOIN platform AS T3 ON T2.platform_id = T3.id WHERE T3.platform_name = ?", (platform_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get distinct game IDs for a specific publisher
@app.get("/v1/video_games/distinct_game_ids_by_publisher", operation_id="get_distinct_game_ids", summary="Retrieve a unique set of game IDs associated with a specific publisher. This operation allows you to identify the distinct games published by a particular publisher, providing a concise overview of their portfolio.")
async def get_distinct_game_ids(publisher_id: int = Query(..., description="ID of the publisher")):
    cursor.execute("SELECT DISTINCT T.game_id FROM game_publisher AS T WHERE T.publisher_id = ?", (publisher_id,))
    result = cursor.fetchall()
    if not result:
        return {"game_ids": []}
    return {"game_ids": [row[0] for row in result]}

# Endpoint to get the genre ID of a specific game
@app.get("/v1/video_games/genre_id_by_game_name", operation_id="get_genre_id", summary="Retrieves the genre ID associated with a specific video game. The operation requires the game's name as input and returns the corresponding genre ID from the database.")
async def get_genre_id(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T.genre_id FROM game AS T WHERE T.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"genre_id": []}
    return {"genre_id": result[0]}

# Endpoint to get the region ID by region name
@app.get("/v1/video_games/region_id_by_name", operation_id="get_region_id", summary="Retrieves the unique identifier of a specific region based on its name. The operation uses the provided region name to search the database and return the corresponding region ID.")
async def get_region_id(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T.id FROM region AS T WHERE T.region_name = ?", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"region_id": []}
    return {"region_id": result[0]}

# Endpoint to get the game platform ID with the highest sales in a specific region
@app.get("/v1/video_games/top_game_platform_by_region", operation_id="get_top_game_platform", summary="Retrieves the game platform with the highest sales in a specified region. The operation calculates the total sales for each game platform in the given region and returns the platform with the highest sales.")
async def get_top_game_platform(region_id: int = Query(..., description="ID of the region")):
    cursor.execute("SELECT T1.game_platform_id FROM ( SELECT T.game_platform_id, SUM(T.num_sales) FROM region_sales AS T WHERE T.region_id = ? GROUP BY T.game_platform_id ORDER BY SUM(T.num_sales) DESC LIMIT 1 ) T1", (region_id,))
    result = cursor.fetchone()
    if not result:
        return {"game_platform_id": []}
    return {"game_platform_id": result[0]}

# Endpoint to get the genre ID with the highest number of games
@app.get("/v1/video_games/top_genre_by_game_count", operation_id="get_top_genre_by_game_count", summary="Retrieves the genre identifier associated with the highest number of games. This operation calculates the count of games for each genre and returns the genre with the maximum count.")
async def get_top_genre_by_game_count():
    cursor.execute("SELECT genre_id FROM ( SELECT T.genre_id, COUNT(T.id) FROM game AS T GROUP BY T.genre_id ORDER BY COUNT(T.id) DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"genre_id": []}
    return {"genre_id": result[0]}

# Endpoint to get the count of distinct games on a specific platform and release year
@app.get("/v1/video_games/count_distinct_games_by_platform_and_year", operation_id="get_count_distinct_games_by_platform_and_year", summary="Retrieves the number of unique games released on a specific platform during a given year. The operation requires the platform name and the release year as input parameters to filter the results.")
async def get_count_distinct_games_by_platform_and_year(platform_name: str = Query(..., description="Name of the platform"), release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT COUNT(DISTINCT T3.game_id) FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN game_publisher AS T3 ON T2.game_publisher_id = T3.id WHERE T1.platform_name = ? AND T2.release_year = ?", (platform_name, release_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the publisher name of a specific game by its name
@app.get("/v1/video_games/publisher_name_by_game_name", operation_id="get_publisher_name_by_game_name", summary="Retrieves the name of the publisher associated with a specific video game, based on the provided game name.")
async def get_publisher_name_by_game_name(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T1.publisher_name FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id INNER JOIN game AS T3 ON T2.game_id = T3.id WHERE T3.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the game platform ID with the highest sales in a specific region
@app.get("/v1/video_games/top_game_platform_by_region_sales", operation_id="get_top_game_platform_by_region_sales", summary="Retrieves the game platform ID that has achieved the highest sales in a specified region. The region is identified by its name, which is provided as an input parameter.")
async def get_top_game_platform_by_region_sales(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T.game_platform_id FROM ( SELECT T2.game_platform_id, MAX(T2.num_sales) FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id WHERE T1.region_name = ? ) t", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"game_platform_id": []}
    return {"game_platform_id": result[0]}

# Endpoint to get the count of platforms for a specific game by its name
@app.get("/v1/video_games/count_platforms_by_game_name", operation_id="get_count_platforms_by_game_name", summary="Retrieves the total number of platforms associated with a specific video game, identified by its name. This operation provides a count of unique platforms that the game is available on, offering insights into its distribution and availability across different platforms.")
async def get_count_platforms_by_game_name(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT COUNT(T2.id) FROM game_platform AS T1 INNER JOIN platform AS T2 ON T1.platform_id = T2.id INNER JOIN game_publisher AS T3 ON T1.game_publisher_id = T3.id INNER JOIN game AS T4 ON T3.game_id = T4.id WHERE T4.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the publisher name of a specific game by its ID
@app.get("/v1/video_games/publisher_name_by_game_id", operation_id="get_publisher_name_by_game_id", summary="Retrieves the name of the publisher associated with a specific video game, identified by its unique game_id. This operation fetches the publisher's name from the publisher table, which is linked to the game_publisher table via the publisher_id. The game_id is used to pinpoint the exact game for which the publisher's name is required.")
async def get_publisher_name_by_game_id(game_id: int = Query(..., description="ID of the game")):
    cursor.execute("SELECT T2.publisher_name FROM game_publisher AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T1.game_id = ?", (game_id,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the publisher name with the highest number of distinct games
@app.get("/v1/video_games/top_publisher_by_game_count", operation_id="get_top_publisher_by_game_count", summary="Retrieves the name of the publisher with the highest number of distinct video games in the catalog. This operation identifies the publisher with the most diverse game portfolio by counting the unique game IDs associated with each publisher and ranking them in descending order. The top-ranked publisher's name is returned as the result.")
async def get_top_publisher_by_game_count():
    cursor.execute("SELECT T.publisher_name FROM ( SELECT T2.publisher_name, COUNT(DISTINCT T1.game_id) FROM game_publisher AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id GROUP BY T2.publisher_name ORDER BY COUNT(DISTINCT T1.game_id) DESC LIMIT 1 ) t")
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the difference in the count of games between two genres
@app.get("/v1/video_games/game_count_difference_by_genres", operation_id="get_game_count_difference_by_genres", summary="Retrieve the difference in the number of games between two specified genres. This operation compares the count of games in the first genre with the count in the second genre, providing a numerical difference as the result.")
async def get_game_count_difference_by_genres(genre_name_1: str = Query(..., description="Name of the first genre"), genre_name_2: str = Query(..., description="Name of the second genre")):
    cursor.execute("SELECT COUNT(CASE WHEN T1.genre_name = ? THEN T2.id ELSE NULL END) - COUNT(CASE WHEN T1.genre_name = ? THEN T2.id ELSE NULL END) FROM genre AS T1 INNER JOIN game AS T2 ON T1.id = T2.genre_id", (genre_name_1, genre_name_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the sales of a game in a specific region and platform
@app.get("/v1/video_games/sales_by_region_and_platform", operation_id="get_sales_by_region_and_platform", summary="Retrieves the total sales of a video game in a specific region and platform, expressed in hundreds of thousands. The operation requires the name of the region and the ID of the game platform as input parameters.")
async def get_sales_by_region_and_platform(region_name: str = Query(..., description="Name of the region"), game_platform_id: int = Query(..., description="ID of the game platform")):
    cursor.execute("SELECT T2.num_sales * 100000 FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id WHERE T1.region_name = ? AND T2.game_platform_id = ?", (region_name, game_platform_id))
    result = cursor.fetchone()
    if not result:
        return {"sales": []}
    return {"sales": result[0]}

# Endpoint to get the count of games published by a specific publisher
@app.get("/v1/video_games/count_games_by_publisher", operation_id="get_count_games_by_publisher", summary="Retrieves the total number of video games published by a specific publisher. The operation requires the publisher's name as input and returns the count of games associated with the provided publisher.")
async def get_count_games_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT COUNT(T2.game_id) FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id WHERE T1.publisher_name = ?", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the platform name of the earliest released game
@app.get("/v1/video_games/earliest_platform", operation_id="get_earliest_platform", summary="Retrieves the name of the platform associated with the earliest released video game. The platform is determined by finding the game with the lowest release year and then identifying its associated platform.")
async def get_earliest_platform():
    cursor.execute("SELECT T2.platform_name FROM game_platform AS T1 INNER JOIN platform AS T2 ON T1.platform_id = T2.id ORDER BY T1.release_year ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"platform_name": []}
    return {"platform_name": result[0]}

# Endpoint to get the percentage of games in a specific genre published by a specific publisher
@app.get("/v1/video_games/percentage_genre_by_publisher", operation_id="get_percentage_genre_by_publisher", summary="Retrieves the percentage of games in a specified genre that were published by a specific publisher. This operation calculates the ratio of games in the given genre to the total games published by the specified publisher.")
async def get_percentage_genre_by_publisher(genre_name: str = Query(..., description="Name of the genre"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T4.genre_name = ? THEN T1.id ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id INNER JOIN genre AS T4 ON T1.genre_id = T4.id WHERE T3.publisher_name = ?", (genre_name, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of games published by two different publishers
@app.get("/v1/video_games/ratio_games_by_publishers", operation_id="get_ratio_games_by_publishers", summary="Retrieves the ratio of games published by two specified publishers. This operation calculates the proportion of games published by the first publisher relative to the second publisher. The input parameters are used to identify the two publishers.")
async def get_ratio_games_by_publishers(publisher_name_1: str = Query(..., description="Name of the first publisher"), publisher_name_2: str = Query(..., description="Name of the second publisher")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.publisher_name = ? THEN T2.game_id ELSE NULL END) AS REAL) / COUNT(CASE WHEN T1.publisher_name = ? THEN T2.game_id ELSE NULL END) FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id", (publisher_name_1, publisher_name_2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of distinct games published by a specific publisher
@app.get("/v1/video_games/count_distinct_games_by_publisher", operation_id="get_count_distinct_games_by_publisher", summary="Retrieves the total number of unique games published by a specific publisher. The operation requires the publisher's name as input and returns the count of distinct games associated with the provided publisher.")
async def get_count_distinct_games_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT COUNT(DISTINCT T2.game_id) FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id WHERE T1.publisher_name = ?", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the platform name of games released in a specific year by a specific publisher
@app.get("/v1/video_games/platform_by_year_and_publisher", operation_id="get_platform_by_year_and_publisher", summary="Get the platform name of games released in a specific year by a specific publisher")
async def get_platform_by_year_and_publisher(release_year: int = Query(..., description="Release year of the game"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT T4.platform_name FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id INNER JOIN game_platform AS T3 ON T2.id = T3.game_publisher_id INNER JOIN platform AS T4 ON T3.platform_id = T4.id WHERE T3.release_year = ? AND T1.publisher_name = ?", (release_year, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"platform_name": []}
    return {"platform_name": result[0]}

# Endpoint to get the distinct publisher name of the earliest released game
@app.get("/v1/video_games/earliest_publisher", operation_id="get_earliest_publisher", summary="Retrieves the name of the publisher that released the earliest game. This operation identifies the publisher responsible for the game with the earliest release year, considering all available games and their respective platforms. The result is a single, distinct publisher name.")
async def get_earliest_publisher():
    cursor.execute("SELECT DISTINCT T3.publisher_name FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id ORDER BY T1.release_year LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the count of game publishers for a specific platform, release year, and region
@app.get("/v1/video_games/count_game_publishers_platform_year_region", operation_id="get_count_game_publishers_platform_year_region", summary="Retrieve the number of game publishers associated with a specific platform, release year, and region. This operation provides a count of publishers based on the provided platform name, release year, and region name.")
async def get_count_game_publishers_platform_year_region(platform_name: str = Query(..., description="Name of the platform"), release_year: int = Query(..., description="Release year"), region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT COUNT(T3.game_publisher_id) FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id INNER JOIN platform AS T4 ON T3.platform_id = T4.id WHERE T4.platform_name = ? AND T3.release_year = ? AND T1.region_name = ?", (platform_name, release_year, region_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the platform with the highest sales in a specific region
@app.get("/v1/video_games/highest_sales_platform_region", operation_id="get_highest_sales_platform_region", summary="Retrieves the name of the platform with the highest total sales in the specified region. The operation calculates the total sales for each platform in the given region and returns the platform with the highest sales.")
async def get_highest_sales_platform_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T.platform_name FROM ( SELECT T4.platform_name, SUM(T2.num_sales) FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id INNER JOIN platform AS T4 ON T3.platform_id = T4.id WHERE T1.region_name = ? ORDER BY T2.num_sales DESC LIMIT 1 ) t", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"platform_name": []}
    return {"platform_name": result[0]}

# Endpoint to get the release year with the highest number of distinct games for a specific platform
@app.get("/v1/video_games/highest_distinct_games_year_platform", operation_id="get_highest_distinct_games_year_platform", summary="Retrieves the year with the most distinct video games released for a specified platform. The platform is identified by its name.")
async def get_highest_distinct_games_year_platform(platform_name: str = Query(..., description="Name of the platform")):
    cursor.execute("SELECT T.release_year FROM ( SELECT T2.release_year, COUNT(DISTINCT T3.game_id) FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN game_publisher AS T3 ON T2.game_publisher_id = T3.id WHERE T1.platform_name = ? GROUP BY T2.release_year ORDER BY COUNT(DISTINCT T3.game_id) DESC LIMIT 1 ) t", (platform_name,))
    result = cursor.fetchone()
    if not result:
        return {"release_year": []}
    return {"release_year": result[0]}

# Endpoint to get the publisher names with exactly one distinct game
@app.get("/v1/video_games/single_game_publishers", operation_id="get_single_game_publishers", summary="Retrieves the names of publishers who have released exactly one unique video game. This operation identifies publishers with a single distinct game title in their portfolio, providing insights into their publishing focus.")
async def get_single_game_publishers():
    cursor.execute("SELECT T.publisher_name FROM ( SELECT T2.publisher_name, COUNT(DISTINCT T1.game_id) FROM game_publisher AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id GROUP BY T2.publisher_name HAVING COUNT(DISTINCT T1.game_id) = 1 ) t")
    result = cursor.fetchall()
    if not result:
        return {"publisher_names": []}
    return {"publisher_names": [row[0] for row in result]}

# Endpoint to get the game IDs for a specific genre ID
@app.get("/v1/video_games/game_ids_by_genre", operation_id="get_game_ids_by_genre", summary="Retrieves the unique identifiers of video games that belong to a specified genre. The genre is identified by its unique ID.")
async def get_game_ids_by_genre(genre_id: int = Query(..., description="ID of the genre")):
    cursor.execute("SELECT T.id FROM game AS T WHERE T.genre_id = ?", (genre_id,))
    result = cursor.fetchall()
    if not result:
        return {"game_ids": []}
    return {"game_ids": [row[0] for row in result]}

# Endpoint to get the release years for games within a specific ID range
@app.get("/v1/video_games/release_years_by_id_range", operation_id="get_release_years_by_id_range", summary="Retrieves the release years of video games that fall within a specified ID range. The range is defined by a starting and ending ID, both of which are required as input parameters. This operation returns a list of release years for the games that meet the ID range criteria.")
async def get_release_years_by_id_range(start_id: int = Query(..., description="Starting ID of the range"), end_id: int = Query(..., description="Ending ID of the range")):
    cursor.execute("SELECT T.release_year FROM game_platform AS T WHERE T.id BETWEEN ? AND ?", (start_id, end_id))
    result = cursor.fetchall()
    if not result:
        return {"release_years": []}
    return {"release_years": [row[0] for row in result]}

# Endpoint to get game publisher IDs based on platform ID
@app.get("/v1/video_games/game_publisher_id_by_platform_id", operation_id="get_game_publisher_id_by_platform_id", summary="Retrieves the IDs of all game publishers associated with a specific platform. The platform is identified by its unique ID.")
async def get_game_publisher_id_by_platform_id(platform_id: int = Query(..., description="ID of the platform")):
    cursor.execute("SELECT T.game_publisher_id FROM game_platform AS T WHERE T.platform_id = ?", (platform_id,))
    result = cursor.fetchall()
    if not result:
        return {"game_publisher_ids": []}
    return {"game_publisher_ids": [row[0] for row in result]}

# Endpoint to get game platform IDs based on release year range
@app.get("/v1/video_games/game_platform_ids_by_release_year_range", operation_id="get_game_platform_ids_by_release_year_range", summary="Retrieves the unique identifiers of game platforms that were released within a specified range of years. The range is defined by the start and end years provided as input parameters. This operation does not return any other information about the game platforms.")
async def get_game_platform_ids_by_release_year_range(start_year: int = Query(..., description="Start year of the release year range"), end_year: int = Query(..., description="End year of the release year range")):
    cursor.execute("SELECT T.id FROM game_platform AS T WHERE T.release_year BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"game_platform_ids": []}
    return {"game_platform_ids": [row[0] for row in result]}

# Endpoint to get sales numbers based on region ID and game platform ID
@app.get("/v1/video_games/sales_by_region_and_game_platform", operation_id="get_sales_by_region_and_game_platform", summary="Retrieves the total sales volume, in hundreds of thousands, for a specific game platform in a given region. The operation requires the unique identifiers for the region and the game platform as input parameters.")
async def get_sales_by_region_and_game_platform(region_id: int = Query(..., description="ID of the region"), game_platform_id: int = Query(..., description="ID of the game platform")):
    cursor.execute("SELECT T.num_sales * 100000 FROM region_sales AS T WHERE T.region_id = ? AND T.game_platform_id = ?", (region_id, game_platform_id))
    result = cursor.fetchall()
    if not result:
        return {"sales": []}
    return {"sales": [row[0] for row in result]}

# Endpoint to get distinct platform names based on release year
@app.get("/v1/video_games/distinct_platform_names_by_release_year", operation_id="get_distinct_platform_names_by_release_year", summary="Retrieve a list of unique platform names associated with a specific release year. This operation allows you to discover the various platforms that have released games in a particular year, providing insights into the gaming landscape for that year.")
async def get_distinct_platform_names_by_release_year(release_year: int = Query(..., description="Release year")):
    cursor.execute("SELECT DISTINCT T1.platform_name FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id WHERE T2.release_year = ?", (release_year,))
    result = cursor.fetchall()
    if not result:
        return {"platform_names": []}
    return {"platform_names": [row[0] for row in result]}

# Endpoint to get release years based on game name
@app.get("/v1/video_games/release_years_by_game_name", operation_id="get_release_years_by_game_name", summary="Retrieves the release years for a specific video game. The operation filters the game platform data based on the provided game name and returns the corresponding release years.")
async def get_release_years_by_game_name(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T1.release_year FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id INNER JOIN game AS T3 ON T2.game_id = T3.id WHERE T3.game_name = ?", (game_name,))
    result = cursor.fetchall()
    if not result:
        return {"release_years": []}
    return {"release_years": [row[0] for row in result]}

# Endpoint to get average sales in Japan
@app.get("/v1/video_games/average_sales_in_japan", operation_id="get_average_sales_in_japan", summary="Retrieves the average sales of video games in Japan, calculated by aggregating sales data from the region_sales table and joining it with the region table using the region_id. The result is multiplied by 100,000 to provide a more readable value.")
async def get_average_sales_in_japan(region_name: str = Query(..., description="Name of the region (must be 'Japan')")):
    cursor.execute("SELECT AVG(T2.num_sales) * 100000 AS avg_japan FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id WHERE T1.region_name = ?", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_sales": []}
    return {"average_sales": result[0]}

# Endpoint to get distinct genre names based on release year range
@app.get("/v1/video_games/distinct_genre_names_by_release_year_range", operation_id="get_distinct_genre_names_by_release_year_range", summary="Retrieves a list of unique genre names for video games released within a specified range of years. The operation accepts a start year and an end year to define the range, and returns the distinct genre names that fall within this period.")
async def get_distinct_genre_names_by_release_year_range(start_year: int = Query(..., description="Start year of the release year range"), end_year: int = Query(..., description="End year of the release year range")):
    cursor.execute("SELECT DISTINCT T4.genre_name FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id INNER JOIN game AS T3 ON T2.game_id = T3.id INNER JOIN genre AS T4 ON T3.genre_id = T4.id WHERE T1.release_year BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"genre_names": []}
    return {"genre_names": [row[0] for row in result]}

# Endpoint to get distinct region names based on sales threshold
@app.get("/v1/video_games/distinct_region_names_by_sales_threshold", operation_id="get_distinct_region_names_by_sales_threshold", summary="Retrieve a list of unique region names where the total sales surpass the specified threshold. The threshold is provided in units of 100,000.")
async def get_distinct_region_names_by_sales_threshold(sales_threshold: int = Query(..., description="Sales threshold (in units of 100,000)")):
    cursor.execute("SELECT DISTINCT T1.region_name FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id WHERE T2.num_sales * 100000 > ?", (sales_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"region_names": []}
    return {"region_names": [row[0] for row in result]}

# Endpoint to get the top publisher by sales in a specific region
@app.get("/v1/video_games/top_publisher_by_region", operation_id="get_top_publisher_by_region", summary="Retrieves the publisher with the highest sales in the specified region. The operation calculates the total sales for each publisher in the given region and returns the publisher with the highest sales. The region is determined by the provided region name.")
async def get_top_publisher_by_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T.publisher_name FROM ( SELECT T5.publisher_name, SUM(T2.num_sales) * 100000 FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id INNER JOIN game_publisher AS T4 ON T3.game_publisher_id = T4.id INNER JOIN publisher AS T5 ON T4.publisher_id = T5.id WHERE T1.region_name = ? GROUP BY T5.publisher_name ORDER BY SUM(T2.num_sales) * 100000 DESC LIMIT 1 ) t", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the release year of games with specific sales in a specific region
@app.get("/v1/video_games/release_year_by_sales_region", operation_id="get_release_year_by_sales_region", summary="Retrieve the release year of video games that have achieved a specified sales volume in a particular region. The operation requires the sales amount and the name of the region as input parameters.")
async def get_release_year_by_sales_region(sales: int = Query(..., description="Sales amount in the specified region"), region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T3.release_year FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id WHERE T2.num_sales * 100000 = ? AND T1.region_name = ?", (sales, region_name))
    result = cursor.fetchall()
    if not result:
        return {"release_years": []}
    return {"release_years": [row[0] for row in result]}

# Endpoint to get the platform names for a specific game
@app.get("/v1/video_games/platform_names_by_game", operation_id="get_platform_names_by_game", summary="Retrieves the names of all platforms on which a specific game is available. The game is identified by its name, which is provided as an input parameter.")
async def get_platform_names_by_game(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T1.platform_name FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN game_publisher AS T3 ON T2.game_publisher_id = T3.id INNER JOIN game AS T4 ON T3.game_id = T4.id WHERE T4.game_name = ?", (game_name,))
    result = cursor.fetchall()
    if not result:
        return {"platform_names": []}
    return {"platform_names": [row[0] for row in result]}

# Endpoint to get the top game by sales in a specific region
@app.get("/v1/video_games/top_game_by_region", operation_id="get_top_game_by_region", summary="Retrieves the name of the video game with the highest sales in a specified region. The region is identified by its name.")
async def get_top_game_by_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T.game_name FROM ( SELECT T5.game_name FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id INNER JOIN game_publisher AS T4 ON T3.game_publisher_id = T4.id INNER JOIN game AS T5 ON T4.game_id = T5.id WHERE T1.region_name = ? ORDER BY T2.num_sales DESC LIMIT 1 ) t", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"game_name": []}
    return {"game_name": result[0]}

# Endpoint to get distinct publisher names with sales above a certain threshold in a specific region
@app.get("/v1/video_games/distinct_publishers_above_sales_threshold", operation_id="get_distinct_publishers_above_sales_threshold", summary="Retrieves a list of unique publisher names that have surpassed the average sales threshold in a specified region. The region is identified by its name, and the sales threshold is calculated as 90% of the average sales in that region.")
async def get_distinct_publishers_above_sales_threshold(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT DISTINCT T5.publisher_name FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id INNER JOIN game_publisher AS T4 ON T3.game_publisher_id = T4.id INNER JOIN publisher AS T5 ON T4.publisher_id = T5.id WHERE T2.num_sales * 10000000 > ( SELECT AVG(T2.num_sales) * 100000 * 90 FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id WHERE T1.region_name = ? )", (region_name,))
    result = cursor.fetchall()
    if not result:
        return {"publisher_names": []}
    return {"publisher_names": [row[0] for row in result]}

# Endpoint to get the percentage of games released on a specific platform in a specific year
@app.get("/v1/video_games/percentage_games_by_platform_year", operation_id="get_percentage_games_by_platform_year", summary="Retrieves the percentage of games released on a specified platform during a given year. This operation calculates the ratio of games released on the specified platform to the total number of games released in the specified year.")
async def get_percentage_games_by_platform_year(platform_name: str = Query(..., description="Name of the platform"), release_year: int = Query(..., description="Release year of the games")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.platform_name = ? THEN T3.game_id ELSE NULL END) AS REAL) * 100 / COUNT(T3.game_id) FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN game_publisher AS T3 ON T2.game_publisher_id = T3.id WHERE T2.release_year = ?", (platform_name, release_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of game platforms released in a specific year
@app.get("/v1/video_games/count_game_platforms_by_year", operation_id="get_count_game_platforms_by_year", summary="Retrieves the total number of game platforms that were released in a specified year. The year is provided as an input parameter.")
async def get_count_game_platforms_by_year(release_year: int = Query(..., description="Release year of the game platforms")):
    cursor.execute("SELECT COUNT(T.id) FROM game_platform AS T WHERE T.release_year = ?", (release_year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total sales for a specific game platform
@app.get("/v1/video_games/total_sales_by_game_platform", operation_id="get_total_sales_by_game_platform", summary="Retrieves the total sales volume, in units of 100,000, for a specific game platform. The platform is identified by its unique ID.")
async def get_total_sales_by_game_platform(game_platform_id: int = Query(..., description="ID of the game platform")):
    cursor.execute("SELECT SUM(T.num_sales) * 100000 FROM region_sales AS T WHERE T.game_platform_id = ?", (game_platform_id,))
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get the publisher ID based on the publisher name
@app.get("/v1/video_games/publisher_id_by_name", operation_id="get_publisher_id_by_name", summary="Retrieves the unique identifier of a video game publisher based on the provided publisher name.")
async def get_publisher_id_by_name(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT T.id FROM publisher AS T WHERE T.publisher_name = ?", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"id": []}
    return {"id": result[0]}

# Endpoint to get the genre name based on the genre ID
@app.get("/v1/video_games/genre_name_by_id", operation_id="get_genre_name_by_id", summary="Retrieves the name of a specific video game genre using its unique identifier. The genre ID is required as an input parameter to identify the genre and return its corresponding name.")
async def get_genre_name_by_id(id: int = Query(..., description="ID of the genre")):
    cursor.execute("SELECT T.genre_name FROM genre AS T WHERE T.id = ?", (id,))
    result = cursor.fetchone()
    if not result:
        return {"genre_name": []}
    return {"genre_name": result[0]}

# Endpoint to get the game platform ID based on the release year
@app.get("/v1/video_games/game_platform_id_by_release_year", operation_id="get_game_platform_id_by_release_year", summary="Get the game platform ID based on the release year")
async def get_game_platform_id_by_release_year(release_year: int = Query(..., description="Release year of the game platform")):
    cursor.execute("SELECT T.id FROM game_platform AS T WHERE T.release_year = ?", (release_year,))
    result = cursor.fetchone()
    if not result:
        return {"id": []}
    return {"id": result[0]}

# Endpoint to get the release year of a game based on the game name
@app.get("/v1/video_games/release_year_by_game_name", operation_id="get_release_year_by_game_name", summary="Retrieves the release year of a specific video game by its name. The operation searches for the game in the database and returns the year it was released. The game name is provided as an input parameter.")
async def get_release_year_by_game_name(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T3.release_year FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN game_platform AS T3 ON T2.id = T3.game_publisher_id WHERE T1.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"release_year": []}
    return {"release_year": result[0]}

# Endpoint to get the game names based on the platform name
@app.get("/v1/video_games/game_names_by_platform", operation_id="get_game_names_by_platform", summary="Retrieves a list of game names associated with the specified platform. The platform name is used to filter the results.")
async def get_game_names_by_platform(platform_name: str = Query(..., description="Name of the platform")):
    cursor.execute("SELECT T4.game_name FROM game_platform AS T1 INNER JOIN platform AS T2 ON T1.platform_id = T2.id INNER JOIN game_publisher AS T3 ON T1.game_publisher_id = T3.id INNER JOIN game AS T4 ON T3.game_id = T4.id WHERE T2.platform_name = ?", (platform_name,))
    result = cursor.fetchall()
    if not result:
        return {"game_names": []}
    return {"game_names": [row[0] for row in result]}

# Endpoint to get the total sales in a specific region and platform
@app.get("/v1/video_games/total_sales_by_region_and_platform", operation_id="get_total_sales_by_region_and_platform", summary="Retrieves the aggregated sales data for a specific video game platform in a given region. The operation calculates the total sales by summing the individual sales figures and converting them to a more manageable scale. The input parameters determine the region and platform for which the sales data is retrieved.")
async def get_total_sales_by_region_and_platform(region_name: str = Query(..., description="Name of the region"), platform_name: str = Query(..., description="Name of the platform")):
    cursor.execute("SELECT SUM(T1.num_sales * 100000) FROM region_sales AS T1 INNER JOIN region AS T2 ON T1.region_id = T2.id INNER JOIN game_platform AS T3 ON T1.game_platform_id = T3.id INNER JOIN platform AS T4 ON T3.platform_id = T4.id WHERE T2.region_name = ? AND T4.platform_name = ?", (region_name, platform_name))
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get game names based on release year
@app.get("/v1/video_games/game_names_by_release_year", operation_id="get_game_names_by_release_year", summary="Retrieves the names of video games released in a specific year. The operation filters games based on the provided release year and returns a list of corresponding game names.")
async def get_game_names_by_release_year(release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT T3.game_name FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id INNER JOIN game AS T3 ON T2.game_id = T3.id WHERE T1.release_year = ?", (release_year,))
    result = cursor.fetchall()
    if not result:
        return {"game_names": []}
    return {"game_names": [row[0] for row in result]}

# Endpoint to get the count of games on a specific platform
@app.get("/v1/video_games/count_games_by_platform", operation_id="get_count_games_by_platform", summary="Retrieves the total number of video games available on a specified gaming platform. The platform is identified by its name.")
async def get_count_games_by_platform(platform_name: str = Query(..., description="Name of the platform")):
    cursor.execute("SELECT COUNT(T1.id) FROM game_platform AS T1 INNER JOIN platform AS T2 ON T1.platform_id = T2.id WHERE T2.platform_name = ?", (platform_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get game names and release years based on genre
@app.get("/v1/video_games/game_names_release_years_by_genre", operation_id="get_game_names_release_years_by_genre", summary="Retrieves a list of up to five video game names and their respective release years, filtered by a specified genre. The genre is provided as an input parameter.")
async def get_game_names_release_years_by_genre(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT T3.game_name, T1.release_year FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id INNER JOIN game AS T3 ON T2.game_id = T3.id INNER JOIN genre AS T4 ON T3.genre_id = T4.id WHERE T4.genre_name = ? LIMIT 5", (genre_name,))
    result = cursor.fetchall()
    if not result:
        return {"game_details": []}
    return {"game_details": [{"game_name": row[0], "release_year": row[1]} for row in result]}

# Endpoint to get platform names based on game name
@app.get("/v1/video_games/platform_names_by_game_name", operation_id="get_platform_names_by_game_name", summary="Retrieves the names of all platforms on which a specific video game is available. The game is identified by its name, which is provided as an input parameter.")
async def get_platform_names_by_game_name(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T4.platform_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN game_platform AS T3 ON T2.id = T3.game_publisher_id INNER JOIN platform AS T4 ON T3.platform_id = T4.id WHERE T1.game_name = ?", (game_name,))
    result = cursor.fetchall()
    if not result:
        return {"platform_names": []}
    return {"platform_names": [row[0] for row in result]}

# Endpoint to get the percentage difference in sales for North America compared to the average
@app.get("/v1/video_games/sales_percentage_difference_north_america", operation_id="get_sales_percentage_difference_north_america", summary="Retrieves the percentage difference in sales for a specific game platform in North America compared to the average sales across all regions. The input parameter is the ID of the game platform.")
async def get_sales_percentage_difference_north_america(game_platform_id: int = Query(..., description="ID of the game platform")):
    cursor.execute("SELECT (SUM(CASE WHEN T2.region_name = 'North America' THEN T1.num_sales ELSE 0 END) - AVG(T1.num_sales)) * 100.0 / AVG(T1.num_sales) FROM region_sales AS T1 INNER JOIN region AS T2 ON T1.region_id = T2.id WHERE T1.game_platform_id = ?", (game_platform_id,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the game with the longest name
@app.get("/v1/video_games/longest_game_name", operation_id="get_longest_game_name", summary="Retrieves the video game with the longest name from the database. The operation returns the name of the game with the highest character count.")
async def get_longest_game_name():
    cursor.execute("SELECT T.game_name FROM game AS T ORDER BY LENGTH(T.game_name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"game_name": []}
    return {"game_name": result[0]}

# Endpoint to get the total sales in dollars
@app.get("/v1/video_games/total_sales_in_dollars", operation_id="get_total_sales_in_dollars", summary="Retrieves the total sales of video games in dollars, calculated by summing up the sales across all regions and multiplying the result by 100,000.")
async def get_total_sales_in_dollars():
    cursor.execute("SELECT SUM(T.num_sales) * 100000 FROM region_sales t")
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get the average number of games per publisher
@app.get("/v1/video_games/average_games_per_publisher", operation_id="get_average_games_per_publisher", summary="Retrieves the average number of games per publisher by calculating the total number of games and dividing it by the number of unique publishers.")
async def get_average_games_per_publisher():
    cursor.execute("SELECT CAST(COUNT(T.game_id) AS REAL) / COUNT(DISTINCT T.publisher_id) FROM game_publisher AS T")
    result = cursor.fetchone()
    if not result:
        return {"average_games_per_publisher": []}
    return {"average_games_per_publisher": result[0]}

# Endpoint to get the minimum release year of games
@app.get("/v1/video_games/min_release_year", operation_id="get_min_release_year", summary="Retrieves the earliest release year of video games across all platforms. This operation provides a single integer value representing the year when the first game was released, based on the available data.")
async def get_min_release_year():
    cursor.execute("SELECT MIN(T.release_year) FROM game_platform t")
    result = cursor.fetchone()
    if not result:
        return {"min_release_year": []}
    return {"min_release_year": result[0]}

# Endpoint to get the game name with the least number of genres
@app.get("/v1/video_games/game_with_least_genres", operation_id="get_game_with_least_genres", summary="Retrieves the name of the video game that is associated with the least number of genres. This operation identifies the game with the fewest genre tags, providing a unique perspective on the diversity of game categories.")
async def get_game_with_least_genres():
    cursor.execute("SELECT T.game_name FROM ( SELECT T2.game_name, COUNT(T2.id) FROM genre AS T1 INNER JOIN game AS T2 ON T1.id = T2.genre_id GROUP BY T2.game_name ORDER BY COUNT(T2.id) ASC LIMIT 1 ) t")
    result = cursor.fetchone()
    if not result:
        return {"game_name": []}
    return {"game_name": result[0]}

# Endpoint to get the platform names ordered by the number of distinct games published
@app.get("/v1/video_games/platforms_by_game_count", operation_id="get_platforms_by_game_count", summary="Retrieve a list of platform names, sorted by the number of unique games published on each platform. The list is grouped by the release year of the games.")
async def get_platforms_by_game_count():
    cursor.execute("SELECT T1.platform_name FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN game_publisher AS T3 ON T2.game_publisher_id = T3.id GROUP BY T2.release_year, T1.platform_name ORDER BY COUNT(DISTINCT T3.game_id) DESC")
    result = cursor.fetchall()
    if not result:
        return {"platform_names": []}
    return {"platform_names": [row[0] for row in result]}

# Endpoint to get the count of sales records with zero sales in a specific region
@app.get("/v1/video_games/zero_sales_count_by_region", operation_id="get_zero_sales_count_by_region", summary="Retrieve the count of sales records that have no sales in a specified region. The operation requires the name of the region and the number of sales as input parameters. The number of sales should be set to zero to get the desired result.")
async def get_zero_sales_count_by_region(region_name: str = Query(..., description="Name of the region"), num_sales: int = Query(..., description="Number of sales")):
    cursor.execute("SELECT COUNT(*) FROM region_sales AS T1 INNER JOIN region AS T2 ON T1.region_id = T2.id WHERE T2.region_name = ? AND T1.num_sales = ?", (region_name, num_sales))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the genre name for a specific game
@app.get("/v1/video_games/genre_by_game_name", operation_id="get_genre_by_game_name", summary="Retrieves the genre name associated with a specific video game. The operation requires the game's name as input and returns the corresponding genre name from the database.")
async def get_genre_by_game_name(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T1.genre_name FROM genre AS T1 INNER JOIN game AS T2 ON T1.id = T2.genre_id WHERE T2.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"genre_name": []}
    return {"genre_name": result[0]}

# Endpoint to get the publisher with the most distinct games
@app.get("/v1/video_games/publisher_with_most_games", operation_id="get_publisher_with_most_games", summary="Retrieves the name of the publisher with the highest number of distinct video games in the catalog. This operation does not require any input parameters and returns the publisher's name as a string.")
async def get_publisher_with_most_games():
    cursor.execute("SELECT T.publisher_name FROM ( SELECT T1.publisher_name, COUNT(DISTINCT T2.game_id) FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id GROUP BY T1.publisher_name ORDER BY COUNT(DISTINCT T2.game_id) DESC LIMIT 1 ) t")
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the latest release year for a specific platform
@app.get("/v1/video_games/latest_release_year_by_platform", operation_id="get_latest_release_year_by_platform", summary="Retrieves the most recent release year for a specified video game platform. The operation filters games by platform name and returns the latest release year, providing a snapshot of the platform's latest game releases.")
async def get_latest_release_year_by_platform(platform_name: str = Query(..., description="Name of the platform")):
    cursor.execute("SELECT T2.release_year FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id WHERE T1.platform_name = ? ORDER BY T2.release_year DESC LIMIT 1", (platform_name,))
    result = cursor.fetchone()
    if not result:
        return {"release_year": []}
    return {"release_year": result[0]}

# Endpoint to get the count of distinct publishers for games with a specific name pattern
@app.get("/v1/video_games/count_publishers_by_game_name", operation_id="get_count_publishers_by_game_name", summary="Retrieve the number of unique publishers associated with games that match a specified name pattern. The input parameter allows for a wildcard search to broaden or narrow the search criteria.")
async def get_count_publishers_by_game_name(game_name_pattern: str = Query(..., description="Pattern to match game names (use % for wildcard)")):
    cursor.execute("SELECT COUNT(DISTINCT T1.publisher_id) FROM game_publisher AS T1 INNER JOIN game AS T2 ON T1.game_id = T2.id WHERE T2.game_name LIKE ?", (game_name_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of games in a specific genre
@app.get("/v1/video_games/percentage_games_by_genre", operation_id="get_percentage_games_by_genre", summary="Retrieves the percentage of video games that belong to a specific genre. The genre is specified as an input parameter, and the result is calculated by comparing the count of games in the specified genre to the total count of games in the database.")
async def get_percentage_games_by_genre(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.genre_name = ? THEN T2.id ELSE NULL END) AS REAL) * 100 / COUNT(T2.id) FROM genre AS T1 INNER JOIN game AS T2 ON T1.id = T2.genre_id", (genre_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of sales between two regions
@app.get("/v1/video_games/sales_ratio_between_regions", operation_id="get_sales_ratio_between_regions", summary="Retrieves the ratio of total video game sales between two specified regions. The operation calculates the sum of sales for each region and returns the ratio of the first region's sales to the second region's sales.")
async def get_sales_ratio_between_regions(region_name_1: str = Query(..., description="Name of the first region"), region_name_2: str = Query(..., description="Name of the second region")):
    cursor.execute("SELECT SUM(CASE WHEN T2.region_name = ? THEN T1.num_sales ELSE 0 END) / SUM(CASE WHEN T2.region_name = ? THEN T1.num_sales ELSE 0 END) FROM region_sales AS T1 INNER JOIN region AS T2 ON T1.region_id = T2.id", (region_name_1, region_name_2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the year with the most game platform releases
@app.get("/v1/video_games/year_with_most_game_platform_releases", operation_id="get_year_with_most_game_platform_releases", summary="Retrieves the year in which the highest number of video game platforms were released. This operation calculates the total number of game platforms released each year and returns the year with the maximum count.")
async def get_year_with_most_game_platform_releases():
    cursor.execute("SELECT T1.release_year FROM ( SELECT T.release_year, COUNT(id) FROM game_platform AS T GROUP BY T.release_year ORDER BY COUNT(T.id) DESC LIMIT 1 ) T1")
    result = cursor.fetchone()
    if not result:
        return {"release_year": []}
    return {"release_year": result[0]}

# Endpoint to get the count of publishers with a specific name pattern
@app.get("/v1/video_games/count_publishers_by_name_pattern", operation_id="get_count_publishers_by_name_pattern", summary="Retrieves the total number of publishers whose names match the provided pattern. The pattern can include wildcard characters to broaden the search. This operation is useful for understanding the distribution of publishers based on their names.")
async def get_count_publishers_by_name_pattern(publisher_name_pattern: str = Query(..., description="Pattern to match publisher names (use % for wildcard)")):
    cursor.execute("SELECT COUNT(T.id) FROM publisher AS T WHERE T.publisher_name LIKE ?", (publisher_name_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top 2 platforms by sales in a specific region
@app.get("/v1/video_games/top_platforms_by_region", operation_id="get_top_platforms_by_region", summary="Retrieves the top two video game platforms with the highest sales in a specified region. The region is identified by its name.")
async def get_top_platforms_by_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T4.platform_name FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id INNER JOIN platform AS T4 ON T3.platform_id = T4.id WHERE T1.region_name = ? ORDER BY T2.num_sales DESC LIMIT 2", (region_name,))
    result = cursor.fetchall()
    if not result:
        return {"platform_names": []}
    return {"platform_names": [row[0] for row in result]}

# Endpoint to get the count of distinct games released in a specific year
@app.get("/v1/video_games/count_games_by_release_year", operation_id="get_count_games_by_release_year", summary="Retrieves the total number of unique video games released in a specified year. The operation considers games from all publishers and platforms, ensuring an accurate count of distinct titles.")
async def get_count_games_by_release_year(release_year: int = Query(..., description="Year of release")):
    cursor.execute("SELECT COUNT(DISTINCT T2.game_id) FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id INNER JOIN game_platform AS T3 ON T2.id = T3.game_publisher_id WHERE T3.release_year = ?", (release_year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top publisher by the number of distinct games released in a specific year
@app.get("/v1/video_games/top_publisher_by_release_year", operation_id="get_top_publisher_by_release_year", summary="Retrieves the publisher with the highest number of distinct video games released in a specified year. The operation filters games by the provided release year, groups them by publisher, and counts the distinct games per publisher. The publisher with the highest count is returned.")
async def get_top_publisher_by_release_year(release_year: int = Query(..., description="Year of release")):
    cursor.execute("SELECT T3.publisher_name FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id INNER JOIN publisher AS T3 ON T2.publisher_id = T3.id WHERE T1.release_year = ? GROUP BY T3.publisher_name ORDER BY COUNT(DISTINCT T2.game_id) DESC LIMIT 1", (release_year,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the count of publishers for a specific game
@app.get("/v1/video_games/publisher_count_by_game", operation_id="get_publisher_count_by_game", summary="Retrieves the total number of publishers associated with a specific video game. The operation requires the name of the game as an input parameter to accurately determine the count.")
async def get_publisher_count_by_game(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT COUNT(T2.publisher_id) FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id WHERE T1.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top publisher by number of games in a specific genre
@app.get("/v1/video_games/top_publisher_by_genre", operation_id="get_top_publisher_by_genre", summary="Retrieves the publisher with the highest number of games in a specified genre. The genre is determined by the provided genre name.")
async def get_top_publisher_by_genre(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT T.publisher_name FROM ( SELECT T4.publisher_name, COUNT(DISTINCT T2.id) FROM genre AS T1 INNER JOIN game AS T2 ON T1.id = T2.genre_id INNER JOIN game_publisher AS T3 ON T2.id = T3.game_id INNER JOIN publisher AS T4 ON T3.publisher_id = T4.id WHERE T1.genre_name = ? GROUP BY T4.publisher_name ORDER BY COUNT(DISTINCT T2.id) DESC LIMIT 1 ) t", (genre_name,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the count of games in a specific genre by a specific publisher
@app.get("/v1/video_games/game_count_by_genre_and_publisher", operation_id="get_game_count_by_genre_and_publisher", summary="Retrieves the total number of video games belonging to a specific genre and published by a specific publisher. The genre and publisher are provided as input parameters.")
async def get_game_count_by_genre_and_publisher(genre_name: str = Query(..., description="Name of the genre"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT COUNT(T3.id) FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id INNER JOIN game AS T3 ON T2.game_id = T3.id INNER JOIN genre AS T4 ON T3.genre_id = T4.id WHERE T4.genre_name = ? AND T1.publisher_name = ?", (genre_name, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct games on a specific platform in a specific region with sales greater than a specified number
@app.get("/v1/video_games/game_count_by_platform_region_sales", operation_id="get_game_count_by_platform_region_sales", summary="Retrieve the number of unique games available on a specified platform in a given region, where the sales of each game surpass a certain threshold. The platform and region are identified by their respective names, while the minimum sales value is provided as a numerical input.")
async def get_game_count_by_platform_region_sales(platform_name: str = Query(..., description="Name of the platform"), region_name: str = Query(..., description="Name of the region"), min_sales: int = Query(..., description="Minimum number of sales")):
    cursor.execute("SELECT COUNT(DISTINCT T2.id) FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN region_sales AS T3 ON T1.id = T3.game_platform_id INNER JOIN region AS T4 ON T3.region_id = T4.id WHERE T1.platform_name = ? AND T4.region_name = ? AND T3.num_sales > ?", (platform_name, region_name, min_sales))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of games in a specific genre
@app.get("/v1/video_games/game_count_by_genre", operation_id="get_game_count_by_genre", summary="Retrieves the total number of video games that belong to a specified genre. The genre is identified by its name, which is provided as an input parameter.")
async def get_game_count_by_genre(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT COUNT(CASE WHEN T1.genre_name = ? THEN T2.id ELSE NULL END) FROM genre AS T1 INNER JOIN game AS T2 ON T1.id = T2.genre_id", (genre_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common genre
@app.get("/v1/video_games/most_common_genre", operation_id="get_most_common_genre", summary="Retrieves the genre that appears most frequently in the video game database. The genre is determined by counting the number of games associated with each genre and selecting the one with the highest count.")
async def get_most_common_genre():
    cursor.execute("SELECT T2.genre_name FROM game AS T1 INNER JOIN genre AS T2 ON T2.id = T1.genre_id GROUP BY T2.genre_name ORDER BY COUNT(T1.genre_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"genre_name": []}
    return {"genre_name": result[0]}

# Endpoint to get the number of sales in EUR for a specific game platform in a specific region
@app.get("/v1/video_games/sales_in_eur_by_platform_region", operation_id="get_sales_in_eur_by_platform_region", summary="Retrieve the total sales in EUR for a specific video game platform in a given region. The operation requires the ID of the game platform and the name of the region as input parameters. The result is calculated by multiplying the number of sales by 100,000.")
async def get_sales_in_eur_by_platform_region(game_platform_id: int = Query(..., description="ID of the game platform"), region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T2.num_sales * 100000 AS nums_eur FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id WHERE T2.game_platform_id = ? AND T1.region_name = ?", (game_platform_id, region_name))
    result = cursor.fetchone()
    if not result:
        return {"nums_eur": []}
    return {"nums_eur": result[0]}

# Endpoint to get the count of games with a specific substring in their name
@app.get("/v1/video_games/count_games_by_name_substring", operation_id="get_count_games_by_name_substring", summary="Retrieves the total number of video games that have a specified substring in their names. The substring is case-insensitive and can appear anywhere within the game name.")
async def get_count_games_by_name_substring(name_substring: str = Query(..., description="Substring to search in game names")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T.game_name FROM game AS T WHERE T.game_name LIKE ? )", ('%' + name_substring + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top game platforms by total sales
@app.get("/v1/video_games/top_game_platforms_by_sales", operation_id="get_top_game_platforms_by_sales", summary="Retrieves the top game platforms ranked by their total sales volume. The number of platforms returned can be specified using the provided input parameter.")
async def get_top_game_platforms_by_sales(limit: int = Query(..., description="Number of top game platforms to return")):
    cursor.execute("SELECT T.game_platform_id, SUM(T.num_sales) * 100000 FROM region_sales AS T GROUP BY game_platform_id ORDER BY SUM(T.num_sales) * 100000 DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"top_game_platforms": []}
    return {"top_game_platforms": result}

# Endpoint to get the earliest release year of game platforms
@app.get("/v1/video_games/earliest_release_year", operation_id="get_earliest_release_year", summary="Retrieves the earliest release year(s) of game platforms, sorted in ascending order. The number of years returned is determined by the provided limit parameter.")
async def get_earliest_release_year(limit: int = Query(..., description="Number of earliest release years to return")):
    cursor.execute("SELECT T.release_year FROM game_platform AS T ORDER BY T.release_year ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"earliest_release_year": []}
    return {"earliest_release_year": result[0]}

# Endpoint to get publishers with a specific substring in their name
@app.get("/v1/video_games/publishers_by_name_substring", operation_id="get_publishers_by_name_substring", summary="Retrieves a list of video game publishers whose names contain a specified substring. The substring is case-insensitive and can appear anywhere within the publisher name.")
async def get_publishers_by_name_substring(name_substring: str = Query(..., description="Substring to search in publisher names")):
    cursor.execute("SELECT T.publisher_name FROM publisher AS T WHERE T.publisher_name LIKE ?", ('%' + name_substring + '%',))
    result = cursor.fetchall()
    if not result:
        return {"publishers": []}
    return {"publishers": result}

# Endpoint to get games by platform name
@app.get("/v1/video_games/games_by_platform", operation_id="get_games_by_platform", summary="Retrieves a list of video game names that are available on the specified platform. The platform is identified by its name.")
async def get_games_by_platform(platform_name: str = Query(..., description="Platform name")):
    cursor.execute("SELECT T1.game_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN game_platform AS T3 ON T2.id = T3.game_publisher_id INNER JOIN platform AS T4 ON T3.platform_id = T4.id WHERE T4.platform_name = ?", (platform_name,))
    result = cursor.fetchall()
    if not result:
        return {"games": []}
    return {"games": result}

# Endpoint to get games by region name
@app.get("/v1/video_games/games_by_region", operation_id="get_games_by_region", summary="Retrieves a list of video game names that are available in the specified region. The operation filters games based on the provided region name and returns the corresponding game names.")
async def get_games_by_region(region_name: str = Query(..., description="Region name")):
    cursor.execute("SELECT T1.game_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN game_platform AS T3 ON T2.id = T3.game_publisher_id INNER JOIN region_sales AS T4 ON T3.id = T4.game_platform_id INNER JOIN region AS T5 ON T4.region_id = T5.id WHERE T5.region_name = ?", (region_name,))
    result = cursor.fetchall()
    if not result:
        return {"games": []}
    return {"games": result}

# Endpoint to get genres by publisher name
@app.get("/v1/video_games/genres_by_publisher", operation_id="get_genres_by_publisher", summary="Retrieves the genres of video games published by the specified publisher. The operation filters video games by publisher name and returns the corresponding genres.")
async def get_genres_by_publisher(publisher_name: str = Query(..., description="Publisher name")):
    cursor.execute("SELECT T4.genre_name FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id INNER JOIN game AS T3 ON T2.game_id = T3.id INNER JOIN genre AS T4 ON T3.genre_id = T4.id WHERE T1.publisher_name = ?", (publisher_name,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": result}

# Endpoint to get the count of games excluding specific genres
@app.get("/v1/video_games/count_games_excluding_genres", operation_id="get_count_games_excluding_genres", summary="Retrieves the total count of video games that do not belong to the provided genre(s). This operation excludes games from the specified genre(s) and returns the remaining count. The input parameters are used to define the genre(s) to be excluded from the count.")
async def get_count_games_excluding_genres(genre1: str = Query(..., description="Genre name to exclude"), genre2: str = Query(..., description="Genre name to exclude"), genre3: str = Query(..., description="Genre name to exclude")):
    cursor.execute("SELECT COUNT(T2.id) FROM genre AS T1 INNER JOIN game AS T2 ON T1.id = T2.genre_id WHERE T1.genre_name NOT IN (?, ?, ?)", (genre1, genre2, genre3))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the region with the highest sales for a platform
@app.get("/v1/video_games/top_region_by_sales", operation_id="get_top_region_by_sales", summary="Retrieves the region with the highest total sales for a specific video game platform. The operation calculates the total sales for each region across all games on the platform, then returns the region with the highest cumulative sales.")
async def get_top_region_by_sales():
    cursor.execute("SELECT T.region_name FROM ( SELECT T1.platform_name, T4.region_name, SUM(T3.num_sales) FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN region_sales AS T3 ON T1.id = T3.game_platform_id INNER JOIN region AS T4 ON T3.region_id = T4.id GROUP BY T1.platform_name, T4.region_name ORDER BY SUM(T3.num_sales) DESC LIMIT 1 ) t")
    result = cursor.fetchone()
    if not result:
        return {"region_name": []}
    return {"region_name": result[0]}

# Endpoint to get the game with the highest sales
@app.get("/v1/video_games/top_game_by_sales", operation_id="get_top_game_by_sales", summary="Retrieves the name of the video game with the highest sales across all platforms and regions. This operation does not require any input parameters and returns the game title as a string.")
async def get_top_game_by_sales():
    cursor.execute("SELECT T.game_name FROM ( SELECT T1.game_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN game_platform AS T3 ON T2.id = T3.game_publisher_id INNER JOIN region_sales AS T4 ON T3.id = T4.game_platform_id ORDER BY T4.num_sales LIMIT 1 ) t")
    result = cursor.fetchone()
    if not result:
        return {"game_name": []}
    return {"game_name": result[0]}

# Endpoint to get the regions where a specific game was sold
@app.get("/v1/video_games/regions_by_game_name", operation_id="get_regions_by_game_name", summary="Retrieve the regions where a specified video game was sold. This operation requires the game's name as input and returns a list of corresponding region names.")
async def get_regions_by_game_name(game_name: str = Query(..., description="Game name")):
    cursor.execute("SELECT T5.region_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN game_platform AS T3 ON T2.id = T3.game_publisher_id INNER JOIN region_sales AS T4 ON T3.id = T4.game_platform_id INNER JOIN region AS T5 ON T4.region_id = T5.id WHERE T1.game_name = ?", (game_name,))
    result = cursor.fetchall()
    if not result:
        return {"region_names": []}
    return {"region_names": [row[0] for row in result]}

# Endpoint to get the games released in a specific year
@app.get("/v1/video_games/games_by_release_year", operation_id="get_games_by_release_year", summary="Retrieve a list of video game titles released in a specific year. The operation filters games based on the provided release year, which should be in 'YYYY' format. The result includes games from all publishers and platforms.")
async def get_games_by_release_year(release_year: str = Query(..., description="Release year in 'YYYY' format")):
    cursor.execute("SELECT T1.game_name FROM game AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.game_id INNER JOIN game_platform AS T3 ON T2.id = T3.game_publisher_id WHERE T3.release_year = ?", (release_year,))
    result = cursor.fetchall()
    if not result:
        return {"game_names": []}
    return {"game_names": [row[0] for row in result]}

# Endpoint to get the average sales per game for a specific platform
@app.get("/v1/video_games/average_sales_per_game_by_platform", operation_id="get_average_sales_per_game_by_platform", summary="Retrieves the average sales per game for a specific video game platform. The platform is identified by its name, and the calculation considers the total sales across all regions for each game on the platform.")
async def get_average_sales_per_game_by_platform(platform_name: str = Query(..., description="Platform name")):
    cursor.execute("SELECT SUM(T3.num_sales * 100000) / COUNT(T1.id) FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN region_sales AS T3 ON T2.id = T3.game_platform_id WHERE T1.platform_name = ?", (platform_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_sales": []}
    return {"average_sales": result[0]}

# Endpoint to get the percentage of games published by a specific publisher
@app.get("/v1/video_games/percentage_games_by_publisher", operation_id="get_percentage_games_by_publisher", summary="Retrieves the percentage of games published by a specific publisher. This operation calculates the ratio of games associated with the given publisher to the total number of games in the database. The result is expressed as a percentage.")
async def get_percentage_games_by_publisher(publisher_name: str = Query(..., description="Publisher name")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.publisher_name = ? THEN T2.game_id ELSE NULL END) AS REAL) * 100 / COUNT(T2.game_id) FROM publisher AS T1 INNER JOIN game_publisher AS T2 ON T1.id = T2.publisher_id", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total sales for a specific region
@app.get("/v1/video_games/total_sales_by_region", operation_id="get_total_sales_by_region", summary="Retrieves the total sales volume, in hundreds of thousands, for a specific region identified by its unique region ID.")
async def get_total_sales_by_region(region_id: int = Query(..., description="Region ID")):
    cursor.execute("SELECT SUM(T.num_sales * 100000) FROM region_sales AS T WHERE T.region_id = ?", (region_id,))
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get the top platform by sales in a specific region
@app.get("/v1/video_games/top_platform_by_region", operation_id="get_top_platform_by_region", summary="Retrieve the name of the platform with the highest sales in the specified region. The operation fetches the top-selling platform by aggregating sales data from the region and game platform associations.")
async def get_top_platform_by_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T.platform_name FROM ( SELECT T4.platform_name FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id INNER JOIN platform AS T4 ON T3.platform_id = T4.id WHERE T1.region_name = ? ORDER BY T2.num_sales DESC LIMIT 1 ) t", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"platform_name": []}
    return {"platform_name": result[0]}

# Endpoint to get the publisher of a specific game
@app.get("/v1/video_games/publisher_by_game_name", operation_id="get_publisher_by_game_name", summary="Retrieves the publisher of a specific video game by its name. The operation uses the provided game name to search for a match in the game database and returns the corresponding publisher's name.")
async def get_publisher_by_game_name(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T2.publisher_name FROM game_publisher AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN game AS T3 ON T1.game_id = T3.id WHERE T3.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the platform of a specific game
@app.get("/v1/video_games/platform_by_game_name", operation_id="get_platform_by_game_name", summary="Retrieves the platform associated with a specific video game. The operation uses the provided game name to search for the corresponding platform in the database. The result is the name of the platform on which the game is available.")
async def get_platform_by_game_name(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T2.platform_name FROM game_platform AS T1 INNER JOIN platform AS T2 ON T1.platform_id = T2.id INNER JOIN game_publisher AS T3 ON T1.game_publisher_id = T3.id INNER JOIN game AS T4 ON T3.game_id = T4.id WHERE T4.game_name = ?", (game_name,))
    result = cursor.fetchone()
    if not result:
        return {"platform_name": []}
    return {"platform_name": result[0]}

# Endpoint to get the top region by total sales for a specific platform
@app.get("/v1/video_games/top_region_by_platform", operation_id="get_top_region_by_platform", summary="Retrieves the region with the highest total sales for a specific video game platform. The operation calculates the total sales for each region across all games on the specified platform and returns the region with the highest sales.")
async def get_top_region_by_platform():
    cursor.execute("SELECT T.region_name FROM ( SELECT T2.region_name, SUM(T1.num_sales) FROM region_sales AS T1 INNER JOIN region AS T2 ON T1.region_id = T2.id INNER JOIN game_platform AS T3 ON T1.game_platform_id = T3.id INNER JOIN platform AS T4 ON T3.platform_id = T4.id GROUP BY T4.platform_name ORDER BY SUM(T1.num_sales) DESC LIMIT 1 ) t")
    result = cursor.fetchone()
    if not result:
        return {"region_name": []}
    return {"region_name": result[0]}

# Endpoint to get distinct release years for games with sales above a certain threshold in a specific region
@app.get("/v1/video_games/release_years_by_sales_threshold_and_region", operation_id="get_release_years_by_sales_threshold_and_region", summary="Retrieves unique release years for video games that have surpassed a specified sales threshold in a given region. The sales threshold is provided in units of 100,000, and the region is identified by its name.")
async def get_release_years_by_sales_threshold_and_region(sales_threshold: int = Query(..., description="Sales threshold (in units of 100,000)"), region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT DISTINCT T3.release_year FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id WHERE T2.num_sales * 100000 > ? AND T1.region_name = ?", (sales_threshold, region_name))
    result = cursor.fetchall()
    if not result:
        return {"release_years": []}
    return {"release_years": [row[0] for row in result]}

# Endpoint to get the count of games for a specific platform and release year
@app.get("/v1/video_games/count_games_platform_release_year", operation_id="get_count_games_platform_release_year", summary="Retrieves the total number of games released on a specific platform during a given year. The operation requires the platform name and the release year as input parameters to filter the games and calculate the count.")
async def get_count_games_platform_release_year(platform_name: str = Query(..., description="Name of the platform"), release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT COUNT(T3.game_id) FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN game_publisher AS T3 ON T2.game_publisher_id = T3.id WHERE T1.platform_name = ? AND T2.release_year = ?", (platform_name, release_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of games released on a specific platform in a given year
@app.get("/v1/video_games/percentage_games_platform_year", operation_id="get_percentage_games_platform_year", summary="Retrieves the percentage of games released on a specified platform during a given year. This operation calculates the proportion of games released on the input platform in the provided year, relative to the total number of games released in that year.")
async def get_percentage_games_platform_year(platform_name: str = Query(..., description="Name of the platform"), release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.platform_name = ? THEN T3.game_id ELSE NULL END) AS REAL) * 100 / COUNT(T3.game_id) FROM game_platform AS T1 INNER JOIN platform AS T2 ON T1.platform_id = T2.id INNER JOIN game_publisher AS T3 ON T1.game_publisher_id = T3.id WHERE T1.release_year = ?", (platform_name, release_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total sales for a specific region
@app.get("/v1/video_games/total_sales_region", operation_id="get_total_sales_region", summary="Retrieves the total sales volume for a specified region, expressed in hundreds of thousands.")
async def get_total_sales_region(region_id: int = Query(..., description="ID of the region")):
    cursor.execute("SELECT SUM(T.num_sales) * 100000 FROM region_sales AS T WHERE T.region_id = ?", (region_id,))
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get the game platform IDs for a specific region
@app.get("/v1/video_games/game_platform_ids_region", operation_id="get_game_platform_ids_region", summary="Retrieves the unique identifiers of game platforms that have sales data in the specified region. The operation filters the sales data by the provided region ID and returns the corresponding game platform IDs.")
async def get_game_platform_ids_region(region_id: int = Query(..., description="ID of the region")):
    cursor.execute("SELECT T.game_platform_id FROM region_sales AS T WHERE T.region_id = ?", (region_id,))
    result = cursor.fetchall()
    if not result:
        return {"game_platform_ids": []}
    return {"game_platform_ids": [row[0] for row in result]}

# Endpoint to get the difference in sales between two regions
@app.get("/v1/video_games/sales_difference_regions", operation_id="get_sales_difference_regions", summary="Retrieve the difference in total sales between two specified regions. This operation compares the sales data of the regions identified by the provided region IDs and returns the net difference.")
async def get_sales_difference_regions(region_id_1: int = Query(..., description="ID of the first region"), region_id_2: int = Query(..., description="ID of the second region")):
    cursor.execute("SELECT SUM(CASE WHEN T.region_id = ? THEN T.num_sales ELSE 0 END) - SUM(CASE WHEN T.region_id = ? THEN T.num_sales ELSE 0 END) FROM region_sales T", (region_id_1, region_id_2))
    result = cursor.fetchone()
    if not result:
        return {"sales_difference": []}
    return {"sales_difference": result[0]}

# Endpoint to get distinct platform IDs for a specific release year
@app.get("/v1/video_games/distinct_platform_ids_release_year", operation_id="get_distinct_platform_ids_release_year", summary="Retrieves a unique set of platform identifiers associated with games released in a specified year. This operation allows you to identify the distinct platforms on which games were released in a particular year, providing insights into platform popularity and game distribution trends.")
async def get_distinct_platform_ids_release_year(release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT DISTINCT T.platform_id FROM game_platform AS T WHERE T.release_year = ?", (release_year,))
    result = cursor.fetchall()
    if not result:
        return {"platform_ids": []}
    return {"platform_ids": [row[0] for row in result]}

# Endpoint to get the count of game publishers for a specific release year
@app.get("/v1/video_games/count_game_publishers_release_year", operation_id="get_count_game_publishers_release_year", summary="Retrieves the total number of unique game publishers that have released games in a specified year. The year of release is provided as an input parameter.")
async def get_count_game_publishers_release_year(release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT COUNT(T.game_publisher_id) FROM game_platform AS T WHERE T.release_year = ?", (release_year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of game publishers for a specific platform
@app.get("/v1/video_games/count_game_publishers_by_platform", operation_id="get_count_game_publishers_by_platform", summary="Retrieves the total number of game publishers associated with a specific gaming platform. The platform is identified by its name.")
async def get_count_game_publishers_by_platform(platform_name: str = Query(..., description="Name of the platform")):
    cursor.execute("SELECT COUNT(T1.game_publisher_id) FROM game_platform AS T1 INNER JOIN platform AS T2 ON T1.platform_id = T2.id WHERE T2.platform_name = ?", (platform_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct platform names for games released in a specific year
@app.get("/v1/video_games/distinct_platforms_by_release_year", operation_id="get_distinct_platforms_by_release_year", summary="Retrieve a list of unique platform names for video games released in a specified year. The operation filters games based on the provided release year and returns the distinct platform names associated with those games.")
async def get_distinct_platforms_by_release_year(release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT DISTINCT T2.platform_name FROM game_platform AS T1 INNER JOIN platform AS T2 ON T1.platform_id = T2.id WHERE T1.release_year = ?", (release_year,))
    result = cursor.fetchall()
    if not result:
        return {"platforms": []}
    return {"platforms": [row[0] for row in result]}

# Endpoint to get the difference in the count of game publishers between two platforms
@app.get("/v1/video_games/difference_in_game_publishers_between_platforms", operation_id="get_difference_in_game_publishers_between_platforms", summary="Retrieve the difference in the number of unique game publishers between two specified platforms. This operation compares the count of distinct game publishers associated with each platform, providing a numerical contrast between the two.")
async def get_difference_in_game_publishers_between_platforms(platform_name_1: str = Query(..., description="Name of the first platform"), platform_name_2: str = Query(..., description="Name of the second platform")):
    cursor.execute("SELECT COUNT(CASE WHEN T2.platform_name = ? THEN T1.game_publisher_id ELSE NULL END) - COUNT(CASE WHEN T2.platform_name = ? THEN T1.game_publisher_id ELSE NULL END) FROM game_platform AS T1 INNER JOIN platform AS T2 ON T1.platform_id = T2.id", (platform_name_1, platform_name_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get game IDs for a specific publisher
@app.get("/v1/video_games/game_ids_by_publisher", operation_id="get_game_ids_by_publisher", summary="Retrieves a list of game IDs associated with a specific video game publisher. The operation filters games based on the provided publisher name and returns their unique identifiers.")
async def get_game_ids_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT T1.game_id FROM game_publisher AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T2.publisher_name = ?", (publisher_name,))
    result = cursor.fetchall()
    if not result:
        return {"game_ids": []}
    return {"game_ids": [row[0] for row in result]}

# Endpoint to get game names for a specific genre
@app.get("/v1/video_games/game_names_by_genre", operation_id="get_game_names_by_genre", summary="Retrieves a list of game names that belong to the specified genre. The genre is identified by its name, which is provided as an input parameter.")
async def get_game_names_by_genre(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT T1.game_name FROM game AS T1 INNER JOIN genre AS T2 ON T1.genre_id = T2.id WHERE T2.genre_name = ?", (genre_name,))
    result = cursor.fetchall()
    if not result:
        return {"game_names": []}
    return {"game_names": [row[0] for row in result]}

# Endpoint to get distinct genre names for games published by a specific publisher
@app.get("/v1/video_games/distinct_genres_by_publisher", operation_id="get_distinct_genres_by_publisher", summary="Retrieve a unique set of genre names for video games published by a specific publisher. The operation requires the publisher's ID as input to filter the results.")
async def get_distinct_genres_by_publisher(publisher_id: int = Query(..., description="ID of the publisher")):
    cursor.execute("SELECT DISTINCT T2.genre_name FROM game AS T1 INNER JOIN genre AS T2 ON T1.genre_id = T2.id INNER JOIN game_publisher AS T3 ON T1.id = T3.game_id WHERE T3.publisher_id = ?", (publisher_id,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the total sales for a specific release year
@app.get("/v1/video_games/total_sales_by_release_year", operation_id="get_total_sales_by_release_year", summary="Retrieves the total sales of video games released in a specific year. The operation calculates the sum of sales across all regions for the given release year.")
async def get_total_sales_by_release_year(release_year: int = Query(..., description="Release year")):
    cursor.execute("SELECT SUM(T1.num_sales) FROM region_sales AS T1 INNER JOIN game_platform AS T2 ON T1.game_platform_id = T2.id WHERE T2.release_year = ?", (release_year,))
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get the difference in sales between two release years
@app.get("/v1/video_games/sales_difference_between_years", operation_id="get_sales_difference_between_years", summary="Retrieve the difference in total sales between two specified release years for video games. This operation compares the total sales of video games released in the first year with those released in the second year, providing a numerical difference.")
async def get_sales_difference_between_years(year1: int = Query(..., description="First release year"), year2: int = Query(..., description="Second release year")):
    cursor.execute("SELECT SUM(CASE WHEN T2.release_year = ? THEN T1.num_sales ELSE 0 END) - SUM(CASE WHEN T2.release_year = ? THEN T1.num_sales ELSE 0 END) FROM region_sales AS T1 INNER JOIN game_platform AS T2 ON T1.game_platform_id = T2.id", (year1, year2))
    result = cursor.fetchone()
    if not result:
        return {"sales_difference": []}
    return {"sales_difference": result[0]}

# Endpoint to get the average sales for a specific region
@app.get("/v1/video_games/average_sales_by_region", operation_id="get_average_sales_by_region", summary="Retrieves the average sales volume, in hundreds of thousands, for a specific region. The region is identified by its unique ID.")
async def get_average_sales_by_region(region_id: int = Query(..., description="Region ID")):
    cursor.execute("SELECT AVG(T.num_sales * 100000) FROM region_sales AS T WHERE T.region_id = ?", (region_id,))
    result = cursor.fetchone()
    if not result:
        return {"average_sales": []}
    return {"average_sales": result[0]}

# Endpoint to get the release year for a specific game platform and publisher
@app.get("/v1/video_games/release_year_by_game_platform_and_publisher", operation_id="get_release_year_by_game_platform_and_publisher", summary="Retrieves the release year of a video game based on the provided game publisher and platform IDs. This operation allows you to find the release year of a specific game by specifying the publisher and platform it belongs to.")
async def get_release_year_by_game_platform_and_publisher(game_publisher_id: int = Query(..., description="Game publisher ID"), game_platform_id: int = Query(..., description="Game platform ID")):
    cursor.execute("SELECT T.release_year FROM game_platform AS T WHERE T.game_publisher_id = ? AND T.id = ?", (game_publisher_id, game_platform_id))
    result = cursor.fetchone()
    if not result:
        return {"release_year": []}
    return {"release_year": result[0]}

# Endpoint to get the game name for a specific game ID
@app.get("/v1/video_games/game_name_by_id", operation_id="get_game_name_by_id", summary="Retrieves the name of a specific video game using its unique identifier. The game ID is required as an input parameter to identify the game.")
async def get_game_name_by_id(game_id: int = Query(..., description="Game ID")):
    cursor.execute("SELECT T.game_name FROM game AS T WHERE T.id = ?", (game_id,))
    result = cursor.fetchone()
    if not result:
        return {"game_name": []}
    return {"game_name": result[0]}

# Endpoint to get the count of distinct games based on genre and release year
@app.get("/v1/video_games/count_distinct_games_by_genre_and_year", operation_id="get_count_distinct_games", summary="Retrieve the number of unique games categorized by a specific genre and released in a particular year. The genre and year are provided as input parameters.")
async def get_count_distinct_games(genre_name: str = Query(..., description="Genre name of the game"), release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT COUNT(DISTINCT T3.id) FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id INNER JOIN game AS T3 ON T2.game_id = T3.id INNER JOIN genre AS T4 ON T3.genre_id = T4.id WHERE T4.genre_name = ? AND T1.release_year = ?", (genre_name, release_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get game platform IDs and region names based on sales threshold
@app.get("/v1/video_games/game_platform_region_by_sales", operation_id="get_game_platform_region", summary="Retrieve the platform IDs and corresponding region names for video games that have achieved sales equal to or below the specified sales threshold (in millions).")
async def get_game_platform_region(sales_threshold: int = Query(..., description="Sales threshold in millions")):
    cursor.execute("SELECT T2.game_platform_id, T1.region_name FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id WHERE T2.num_sales * 100000 <= ?", (sales_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"game_platform_region": []}
    return {"game_platform_region": [{"game_platform_id": row[0], "region_name": row[1]} for row in result]}

# Endpoint to get game names based on publisher name and release year
@app.get("/v1/video_games/game_names_by_publisher_and_year", operation_id="get_game_names", summary="Retrieves the names of video games published by a specific publisher in a given year. The operation requires the publisher's name and the release year as input parameters to filter the results.")
async def get_game_names(publisher_name: str = Query(..., description="Publisher name"), release_year: int = Query(..., description="Release year of the game")):
    cursor.execute("SELECT T3.game_name FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id INNER JOIN game AS T3 ON T2.game_id = T3.id INNER JOIN publisher AS T4 ON T2.publisher_id = T4.id WHERE T4.publisher_name = ? AND T1.release_year = ?", (publisher_name, release_year))
    result = cursor.fetchall()
    if not result:
        return {"game_names": []}
    return {"game_names": [row[0] for row in result]}

# Endpoint to get genre name based on game ID
@app.get("/v1/video_games/genre_name_by_game_id", operation_id="get_genre_name", summary="Retrieves the genre name associated with a specific video game, identified by its unique game ID. This operation fetches the genre name from the genre table, which is linked to the game table via the genre ID. The game ID is used to locate the corresponding genre ID, and subsequently, the genre name.")
async def get_genre_name(game_id: int = Query(..., description="ID of the game")):
    cursor.execute("SELECT T2.genre_name FROM game AS T1 INNER JOIN genre AS T2 ON T1.genre_id = T2.id WHERE T1.id = ?", (game_id,))
    result = cursor.fetchone()
    if not result:
        return {"genre_name": []}
    return {"genre_name": result[0]}

# Endpoint to get platform names based on game ID
@app.get("/v1/video_games/platform_names_by_game_id", operation_id="get_platform_names", summary="Retrieves the names of all platforms associated with a specific video game, identified by its unique game_id. This operation fetches the platform names from the database by joining the game_publisher, game_platform, and platform tables using their respective IDs.")
async def get_platform_names(game_id: int = Query(..., description="ID of the game")):
    cursor.execute("SELECT T3.platform_name FROM game_publisher AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.game_publisher_id INNER JOIN platform AS T3 ON T2.platform_id = T3.id WHERE T1.game_id = ?", (game_id,))
    result = cursor.fetchall()
    if not result:
        return {"platform_names": []}
    return {"platform_names": [row[0] for row in result]}

# Endpoint to get genre names based on a list of game names
@app.get("/v1/video_games/genre_names_by_game_names", operation_id="get_genre_names_by_game_names", summary="Retrieves the genre names associated with the provided list of game names. The operation accepts up to three game names as input parameters and returns the corresponding genre names from the database.")
async def get_genre_names_by_game_names(game_name_1: str = Query(..., description="First game name"), game_name_2: str = Query(..., description="Second game name"), game_name_3: str = Query(..., description="Third game name")):
    cursor.execute("SELECT T2.genre_name FROM game AS T1 INNER JOIN genre AS T2 ON T1.genre_id = T2.id WHERE T1.game_name IN (?, ?, ?)", (game_name_1, game_name_2, game_name_3))
    result = cursor.fetchall()
    if not result:
        return {"genre_names": []}
    return {"genre_names": [row[0] for row in result]}

# Endpoint to get distinct publisher names in a specific region with sales below a certain threshold
@app.get("/v1/video_games/publishers_by_region_and_sales", operation_id="get_publishers_by_region_and_sales", summary="Retrieve a list of unique video game publishers from a specified region, whose sales are below a certain threshold. The operation returns a limited number of results based on the provided limit parameter.")
async def get_publishers_by_region_and_sales(region_name: str = Query(..., description="Name of the region"), sales_threshold: int = Query(..., description="Sales threshold in units of 100,000"), limit: int = Query(..., description="Limit of results to return")):
    cursor.execute("SELECT T.publisher_name FROM ( SELECT DISTINCT T5.publisher_name FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id INNER JOIN game_publisher AS T4 ON T3.game_publisher_id = T4.id INNER JOIN publisher AS T5 ON T4.publisher_id = T5.id WHERE T1.region_name = ? AND T2.num_sales * 100000 < ? LIMIT ? ) t", (region_name, sales_threshold, limit))
    result = cursor.fetchall()
    if not result:
        return {"publishers": []}
    return {"publishers": [row[0] for row in result]}

# Endpoint to get platform IDs for a specific game
@app.get("/v1/video_games/platform_ids_by_game", operation_id="get_platform_ids_by_game", summary="Retrieves the platform IDs associated with a specific video game. The operation filters the game_platform table based on the provided game name, joining it with the game_publisher and game tables to ensure accurate results.")
async def get_platform_ids_by_game(game_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T1.platform_id FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id INNER JOIN game AS T3 ON T2.game_id = T3.id WHERE T3.game_name = ?", (game_name,))
    result = cursor.fetchall()
    if not result:
        return {"platform_ids": []}
    return {"platform_ids": [row[0] for row in result]}

# Endpoint to get the release year of a specific game
@app.get("/v1/video_games/release_year_by_game", operation_id="get_release_year_by_game", summary="Retrieves the release year of a specific game by its unique identifier. This operation fetches the release year from the game_platform table, which is linked to the game_publisher table via the game_publisher_id. The game_id is used to filter the results and return the corresponding release year.")
async def get_release_year_by_game(game_id: int = Query(..., description="ID of the game")):
    cursor.execute("SELECT T1.release_year FROM game_platform AS T1 INNER JOIN game_publisher AS T2 ON T1.game_publisher_id = T2.id WHERE T2.game_id = ?", (game_id,))
    result = cursor.fetchone()
    if not result:
        return {"release_year": []}
    return {"release_year": result[0]}

# Endpoint to get the difference in game counts between two platforms for a specific publisher
@app.get("/v1/video_games/game_count_difference_by_platforms_and_publisher", operation_id="get_game_count_difference_by_platforms_and_publisher", summary="Retrieve the difference in the number of games available on two specified platforms for a given publisher. This operation compares the game counts and returns the disparity, providing insights into the publisher's game distribution across platforms.")
async def get_game_count_difference_by_platforms_and_publisher(platform_name_1: str = Query(..., description="Name of the first platform"), platform_name_2: str = Query(..., description="Name of the second platform"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT COUNT(CASE WHEN T1.platform_name = ? THEN T3.game_id ELSE NULL END) - COUNT(CASE WHEN T1.platform_name = ? THEN T3.game_id ELSE NULL END) FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN game_publisher AS T3 ON T2.game_publisher_id = T3.id INNER JOIN publisher AS T4 ON T3.publisher_id = T4.id WHERE T4.publisher_name = ?", (platform_name_1, platform_name_2, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"game_count_difference": []}
    return {"game_count_difference": result[0]}

# Endpoint to get the percentage of games released in a specific year for a specific platform
@app.get("/v1/video_games/percentage_games_by_year_and_platform", operation_id="get_percentage_games_by_year_and_platform", summary="Retrieves the percentage of games released in a specific year for a given platform. This operation calculates the proportion of games released in the provided year relative to the total number of games available for the specified platform. The input parameters include the release year of the games and the name of the platform.")
async def get_percentage_games_by_year_and_platform(release_year: int = Query(..., description="Release year of the game"), platform_name: str = Query(..., description="Name of the platform")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.release_year = ? THEN T3.game_id ELSE NULL END) AS REAL) * 100 / COUNT(T3.game_id) FROM platform AS T1 INNER JOIN game_platform AS T2 ON T1.id = T2.platform_id INNER JOIN game_publisher AS T3 ON T2.game_publisher_id = T3.id WHERE T1.platform_name = ?", (release_year, platform_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct platform IDs in a specific region with sales below a certain threshold
@app.get("/v1/video_games/platform_ids_by_region_and_sales_threshold", operation_id="get_platform_ids_by_region_and_sales_threshold", summary="Retrieve the unique identifiers of platforms that have sales below 30% of the average sales in a specified region. This operation filters platforms based on the provided region and the sales threshold, ensuring that only those with sales under the calculated average are returned.")
async def get_platform_ids_by_region_and_sales_threshold(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT DISTINCT T3.platform_id FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id INNER JOIN game_platform AS T3 ON T2.game_platform_id = T3.id WHERE T1.region_name = ? AND T2.num_sales * 100 * 100000 < ( SELECT AVG(T2.num_sales * 100000) * 30 FROM region AS T1 INNER JOIN region_sales AS T2 ON T1.id = T2.region_id WHERE T1.region_name = ? )", (region_name, region_name))
    result = cursor.fetchall()
    if not result:
        return {"platform_ids": []}
    return {"platform_ids": [row[0] for row in result]}

api_calls = [
    "/v1/video_games/sales_difference_by_platform_region?platform_id_1=50&platform_id_2=51&region_id=1",
    "/v1/video_games/games_by_genre_of_game?game_name=3D%20Lemmings",
    "/v1/video_games/count_games_by_genre?genre_name=Action",
    "/v1/video_games/genre_of_game?game_name=3D%20Lemmings",
    "/v1/video_games/publisher_of_game?game_name=3D%20Lemmings",
    "/v1/video_games/games_by_publisher?publisher_name=10TACLE%20Studios",
    "/v1/video_games/count_games_by_genre_and_publisher?genre_name=Puzzle&publisher_name=10TACLE%20Studios",
    "/v1/video_games/games_by_genre_and_publisher?publisher_name=10TACLE%20Studios&genre_name=Puzzle",
    "/v1/video_games/top_publisher",
    "/v1/video_games/platform_of_game_by_year?game_name=Panzer%20Tactics&release_year=2007",
    "/v1/video_games/release_year_by_game_and_platform?game_name=Panzer%20Tactics&platform_name=DS",
    "/v1/video_games/distinct_publishers_by_genre?genre_name=Puzzle",
    "/v1/video_games/count_publishers_by_genre_and_game_count?genre_name=Puzzle&min_game_count=3",
    "/v1/video_games/percentage_games_by_genre_and_publisher?genre_name=Sports&publisher_name=Nintendo",
    "/v1/video_games/total_sales_by_platform?platform_name=DS",
    "/v1/video_games/distinct_game_ids_by_publisher?publisher_id=352",
    "/v1/video_games/genre_id_by_game_name?game_name=Pro%20Evolution%20Soccer%202012",
    "/v1/video_games/region_id_by_name?region_name=Japan",
    "/v1/video_games/top_game_platform_by_region?region_id=2",
    "/v1/video_games/top_genre_by_game_count",
    "/v1/video_games/count_distinct_games_by_platform_and_year?platform_name=X360&release_year=2010",
    "/v1/video_games/publisher_name_by_game_name?game_name=ModNation%20Racers",
    "/v1/video_games/top_game_platform_by_region_sales?region_name=Japan",
    "/v1/video_games/count_platforms_by_game_name?game_name=Pro%20Evolution%20Soccer%202016",
    "/v1/video_games/publisher_name_by_game_id?game_id=10031",
    "/v1/video_games/top_publisher_by_game_count",
    "/v1/video_games/game_count_difference_by_genres?genre_name_1=Sports&genre_name_2=Simulation",
    "/v1/video_games/sales_by_region_and_platform?region_name=Europe&game_platform_id=3871",
    "/v1/video_games/count_games_by_publisher?publisher_name=Ascaron%20Entertainment%20GmbH",
    "/v1/video_games/earliest_platform",
    "/v1/video_games/percentage_genre_by_publisher?genre_name=Adventure&publisher_name=Namco%20Bandai%20Games",
    "/v1/video_games/ratio_games_by_publishers?publisher_name_1=Atari&publisher_name_2=Athena",
    "/v1/video_games/count_distinct_games_by_publisher?publisher_name=Electronic%20Arts",
    "/v1/video_games/platform_by_year_and_publisher?release_year=2004&publisher_name=Codemasters",
    "/v1/video_games/earliest_publisher",
    "/v1/video_games/count_game_publishers_platform_year_region?platform_name=X360&release_year=2011&region_name=Japan",
    "/v1/video_games/highest_sales_platform_region?region_name=Europe",
    "/v1/video_games/highest_distinct_games_year_platform?platform_name=PC",
    "/v1/video_games/single_game_publishers",
    "/v1/video_games/game_ids_by_genre?genre_id=2",
    "/v1/video_games/release_years_by_id_range?start_id=1&end_id=10",
    "/v1/video_games/game_publisher_id_by_platform_id?platform_id=15",
    "/v1/video_games/game_platform_ids_by_release_year_range?start_year=2000&end_year=2003",
    "/v1/video_games/sales_by_region_and_game_platform?region_id=2&game_platform_id=9615",
    "/v1/video_games/distinct_platform_names_by_release_year?release_year=2016",
    "/v1/video_games/release_years_by_game_name?game_name=3DS%20Classic%20Collection",
    "/v1/video_games/average_sales_in_japan?region_name=Japan",
    "/v1/video_games/distinct_genre_names_by_release_year_range?start_year=2000&end_year=2002",
    "/v1/video_games/distinct_region_names_by_sales_threshold?sales_threshold=300000",
    "/v1/video_games/top_publisher_by_region?region_name=North%20America",
    "/v1/video_games/release_year_by_sales_region?sales=350000&region_name=North%20America",
    "/v1/video_games/platform_names_by_game?game_name=Counter%20Force",
    "/v1/video_games/top_game_by_region?region_name=Japan",
    "/v1/video_games/distinct_publishers_above_sales_threshold?region_name=Japan",
    "/v1/video_games/percentage_games_by_platform_year?platform_name=PSP&release_year=2004",
    "/v1/video_games/count_game_platforms_by_year?release_year=1981",
    "/v1/video_games/total_sales_by_game_platform?game_platform_id=9658",
    "/v1/video_games/publisher_id_by_name?publisher_name=1C%20Company",
    "/v1/video_games/genre_name_by_id?id=3",
    "/v1/video_games/game_platform_id_by_release_year?release_year=2017",
    "/v1/video_games/release_year_by_game_name?game_name=Adventure%20Island",
    "/v1/video_games/game_names_by_platform?platform_name=SCD",
    "/v1/video_games/total_sales_by_region_and_platform?region_name=North%20America&platform_name=PS4",
    "/v1/video_games/game_names_by_release_year?release_year=2011",
    "/v1/video_games/count_games_by_platform?platform_name=Wii",
    "/v1/video_games/game_names_release_years_by_genre?genre_name=Sports",
    "/v1/video_games/platform_names_by_game_name?game_name=Panzer%20Tactics",
    "/v1/video_games/sales_percentage_difference_north_america?game_platform_id=9577",
    "/v1/video_games/longest_game_name",
    "/v1/video_games/total_sales_in_dollars",
    "/v1/video_games/average_games_per_publisher",
    "/v1/video_games/min_release_year",
    "/v1/video_games/game_with_least_genres",
    "/v1/video_games/platforms_by_game_count",
    "/v1/video_games/zero_sales_count_by_region?region_name=Europe&num_sales=0",
    "/v1/video_games/genre_by_game_name?game_name=Mario%20vs.%20Donkey%20Kong",
    "/v1/video_games/publisher_with_most_games",
    "/v1/video_games/latest_release_year_by_platform?platform_name=WiiU",
    "/v1/video_games/count_publishers_by_game_name?game_name_pattern=Marvel%",
    "/v1/video_games/percentage_games_by_genre?genre_name=Sports",
    "/v1/video_games/sales_ratio_between_regions?region_name_1=North%20America&region_name_2=Japan",
    "/v1/video_games/year_with_most_game_platform_releases",
    "/v1/video_games/count_publishers_by_name_pattern?publisher_name_pattern=%Interactive%",
    "/v1/video_games/top_platforms_by_region?region_name=North%20America",
    "/v1/video_games/count_games_by_release_year?release_year=2012",
    "/v1/video_games/top_publisher_by_release_year?release_year=2007",
    "/v1/video_games/publisher_count_by_game?game_name=Minecraft",
    "/v1/video_games/top_publisher_by_genre?genre_name=Action",
    "/v1/video_games/game_count_by_genre_and_publisher?genre_name=Sports&publisher_name=Nintendo",
    "/v1/video_games/game_count_by_platform_region_sales?platform_name=DS&region_name=Other&min_sales=0",
    "/v1/video_games/game_count_by_genre?genre_name=Strategy",
    "/v1/video_games/most_common_genre",
    "/v1/video_games/sales_in_eur_by_platform_region?game_platform_id=26&region_name=Europe",
    "/v1/video_games/count_games_by_name_substring?name_substring=Box",
    "/v1/video_games/top_game_platforms_by_sales?limit=3",
    "/v1/video_games/earliest_release_year?limit=1",
    "/v1/video_games/publishers_by_name_substring?name_substring=Entertainment",
    "/v1/video_games/games_by_platform?platform_name=SCD",
    "/v1/video_games/games_by_region?region_name=Japan",
    "/v1/video_games/genres_by_publisher?publisher_name=Agatsuma%20Entertainment",
    "/v1/video_games/count_games_excluding_genres?genre1=Role-Playing&genre2=Shooter&genre3=Simulation",
    "/v1/video_games/top_region_by_sales",
    "/v1/video_games/top_game_by_sales",
    "/v1/video_games/regions_by_game_name?game_name=Pengo",
    "/v1/video_games/games_by_release_year?release_year=2010",
    "/v1/video_games/average_sales_per_game_by_platform?platform_name=PS2",
    "/v1/video_games/percentage_games_by_publisher?publisher_name=Brash%20Entertainment",
    "/v1/video_games/total_sales_by_region?region_id=1",
    "/v1/video_games/top_platform_by_region?region_name=Europe",
    "/v1/video_games/publisher_by_game_name?game_name=2002%20FIFA%20World%20Cup",
    "/v1/video_games/platform_by_game_name?game_name=3Xtreme",
    "/v1/video_games/top_region_by_platform",
    "/v1/video_games/release_years_by_sales_threshold_and_region?sales_threshold=200000&region_name=Japan",
    "/v1/video_games/count_games_platform_release_year?platform_name=PS3&release_year=2010",
    "/v1/video_games/percentage_games_platform_year?platform_name=PS4&release_year=2014",
    "/v1/video_games/total_sales_region?region_id=4",
    "/v1/video_games/game_platform_ids_region?region_id=1",
    "/v1/video_games/sales_difference_regions?region_id_1=2&region_id_2=3",
    "/v1/video_games/distinct_platform_ids_release_year?release_year=2007",
    "/v1/video_games/count_game_publishers_release_year?release_year=1984",
    "/v1/video_games/count_game_publishers_by_platform?platform_name=X360",
    "/v1/video_games/distinct_platforms_by_release_year?release_year=2000",
    "/v1/video_games/difference_in_game_publishers_between_platforms?platform_name_1=PS3&platform_name_2=X360",
    "/v1/video_games/game_ids_by_publisher?publisher_name=Bethesda%20Softworks",
    "/v1/video_games/game_names_by_genre?genre_name=Racing",
    "/v1/video_games/distinct_genres_by_publisher?publisher_id=464",
    "/v1/video_games/total_sales_by_release_year?release_year=2000",
    "/v1/video_games/sales_difference_between_years?year1=2000&year2=1990",
    "/v1/video_games/average_sales_by_region?region_id=3",
    "/v1/video_games/release_year_by_game_platform_and_publisher?game_publisher_id=6657&game_platform_id=19",
    "/v1/video_games/game_name_by_id?game_id=44",
    "/v1/video_games/count_distinct_games_by_genre_and_year?genre_name=Adventure&release_year=2005",
    "/v1/video_games/game_platform_region_by_sales?sales_threshold=20000",
    "/v1/video_games/game_names_by_publisher_and_year?publisher_name=505%20Games&release_year=2006",
    "/v1/video_games/genre_name_by_game_id?game_id=119",
    "/v1/video_games/platform_names_by_game_id?game_id=178",
    "/v1/video_games/genre_names_by_game_names?game_name_1=Airlock&game_name_2=Airline%20Tycoon&game_name_3=Airblade",
    "/v1/video_games/publishers_by_region_and_sales?region_name=North%20America&sales_threshold=10000&limit=5",
    "/v1/video_games/platform_ids_by_game?game_name=Airborne%20Troops:%20Countdown%20to%20D-Day",
    "/v1/video_games/release_year_by_game?game_id=156",
    "/v1/video_games/game_count_difference_by_platforms_and_publisher?platform_name_1=SNES&platform_name_2=DS&publisher_name=Culture%20Brain",
    "/v1/video_games/percentage_games_by_year_and_platform?release_year=2007&platform_name=Wii",
    "/v1/video_games/platform_ids_by_region_and_sales_threshold?region_name=Europe"
]
