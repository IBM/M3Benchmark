from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/olympics/olympics.sqlite')
cursor = conn.cursor()

# Endpoint to get the games names for a specific competitor
@app.get("/v1/olympics/games_names_by_competitor", operation_id="get_games_names_by_competitor", summary="Retrieves the names of the Olympic Games in which a specified competitor has participated. The competitor is identified by their full name.")
async def get_games_names_by_competitor(full_name: str = Query(..., description="Full name of the competitor")):
    cursor.execute("SELECT T1.games_name FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.full_name = ?", (full_name,))
    result = cursor.fetchall()
    if not result:
        return {"games_names": []}
    return {"games_names": [row[0] for row in result]}

# Endpoint to get the games names for a specific competitor at a specific age
@app.get("/v1/olympics/games_names_by_competitor_age", operation_id="get_games_names_by_competitor_age", summary="Retrieve the names of the games in which a specific competitor, identified by their full name, participated at a specified age.")
async def get_games_names_by_competitor_age(full_name: str = Query(..., description="Full name of the competitor"), age: int = Query(..., description="Age of the competitor")):
    cursor.execute("SELECT T1.games_name FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.full_name = ? AND T2.age = ?", (full_name, age))
    result = cursor.fetchall()
    if not result:
        return {"games_names": []}
    return {"games_names": [row[0] for row in result]}

# Endpoint to get the age of a specific competitor in a specific game
@app.get("/v1/olympics/competitor_age_in_game", operation_id="get_competitor_age_in_game", summary="Retrieves the age of a specific competitor in a given game. The operation requires the full name of the competitor and the name of the game as input parameters. It returns the age of the competitor in the specified game.")
async def get_competitor_age_in_game(full_name: str = Query(..., description="Full name of the competitor"), games_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T2.age FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.full_name = ? AND T1.games_name = ?", (full_name, games_name))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the count of persons from a specific region
@app.get("/v1/olympics/count_persons_by_region", operation_id="get_count_persons_by_region", summary="Retrieves the total number of persons associated with a specified region. The region is identified by its name, which is used to filter the data and calculate the count.")
async def get_count_persons_by_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT COUNT(T1.person_id) FROM person_region AS T1 INNER JOIN noc_region AS T2 ON T1.region_id = T2.id WHERE T2.region_name = ?", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the full names of persons from a specific region
@app.get("/v1/olympics/full_names_by_region", operation_id="get_full_names_by_region", summary="Retrieves the full names of individuals associated with a specified region. The operation filters the data based on the provided region name, returning a list of corresponding full names.")
async def get_full_names_by_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT T3.full_name FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ?", (region_name,))
    result = cursor.fetchall()
    if not result:
        return {"full_names": []}
    return {"full_names": [row[0] for row in result]}

# Endpoint to get the region names of a specific person
@app.get("/v1/olympics/region_names_by_person", operation_id="get_region_names_by_person", summary="Retrieves the region names associated with a specific individual. The operation uses the provided full name to search for corresponding records in the database and returns the region names linked to that person.")
async def get_region_names_by_person(full_name: str = Query(..., description="Full name of the person")):
    cursor.execute("SELECT T1.region_name FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.full_name = ?", (full_name,))
    result = cursor.fetchall()
    if not result:
        return {"region_names": []}
    return {"region_names": [row[0] for row in result]}

# Endpoint to get the NOC of the tallest person of a specific gender
@app.get("/v1/olympics/tallest_person_noc_by_gender", operation_id="get_tallest_person_noc_by_gender", summary="Retrieve the National Olympic Committee (NOC) of the tallest person of a specified gender. The operation filters persons by gender and sorts them by height in descending order, returning the NOC of the tallest individual.")
async def get_tallest_person_noc_by_gender(gender: str = Query(..., description="Gender of the person (M or F)")):
    cursor.execute("SELECT T1.noc FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.gender = ? ORDER BY T3.height DESC LIMIT 1", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"noc": []}
    return {"noc": result[0]}

# Endpoint to get the count of persons from a specific region and gender
@app.get("/v1/olympics/count_persons_by_region_gender", operation_id="get_count_persons_by_region_gender", summary="Retrieves the total count of individuals from a specified region and gender. The operation filters the data based on the provided region name and gender, and returns the corresponding count.")
async def get_count_persons_by_region_gender(region_name: str = Query(..., description="Name of the region"), gender: str = Query(..., description="Gender of the person (M or F)")):
    cursor.execute("SELECT COUNT(T3.id) FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ? AND T3.gender = ?", (region_name, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city names of a specific game
@app.get("/v1/olympics/city_names_by_game", operation_id="get_city_names_by_game", summary="Retrieves the names of cities that have hosted a specific Olympic game. The operation filters the list of cities based on the provided game name.")
async def get_city_names_by_game(games_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT T2.city_name FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T3.games_name = ?", (games_name,))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get the games names held in a specific city
@app.get("/v1/olympics/games_names_by_city", operation_id="get_games_names_by_city", summary="Retrieve the names of the Olympic games that took place in a specified city. The operation requires the city's name as input and returns a list of corresponding game names.")
async def get_games_names_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T3.games_name FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = ?", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"games_names": []}
    return {"games_names": [row[0] for row in result]}

# Endpoint to get the year of the first game held in a specific city
@app.get("/v1/olympics/first_game_year_by_city", operation_id="get_first_game_year_by_city", summary="Retrieves the year of the first Olympic Games held in the specified city. The operation filters the games by city name and returns the earliest year in which the Games were held in that city.")
async def get_first_game_year_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T3.games_year FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = ? ORDER BY T3.games_year LIMIT 1", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the count of games held in a specific city
@app.get("/v1/olympics/count_games_by_city", operation_id="get_count_games_by_city", summary="Retrieves the total number of Olympic Games held in a specified city. The operation requires the name of the city as an input parameter.")
async def get_count_games_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT COUNT(T1.games_id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id WHERE T2.city_name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average height of athletes from a specific region and gender
@app.get("/v1/olympics/avg_height_by_region_gender", operation_id="get_avg_height_by_region_gender", summary="Retrieves the average height of athletes from a specified region and gender. The operation calculates the mean height by querying the database for athletes from the given region and gender, then aggregating their heights.")
async def get_avg_height_by_region_gender(region_name: str = Query(..., description="Name of the region"), gender: str = Query(..., description="Gender of the athlete (M/F)")):
    cursor.execute("SELECT AVG(T3.height) FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ? AND T3.gender = ?", (region_name, gender))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the percentage of athletes from a specific region in a specific game
@app.get("/v1/olympics/percentage_athletes_by_region_game", operation_id="get_percentage_athletes_by_region_game", summary="Retrieves the percentage of athletes from a specified region who participated in a particular game. The calculation is based on the total number of athletes in the game and the count of athletes from the given region.")
async def get_percentage_athletes_by_region_game(region_name: str = Query(..., description="Name of the region"), games_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T5.region_name = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T3.id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id INNER JOIN person_region AS T4 ON T3.id = T4.person_id INNER JOIN noc_region AS T5 ON T4.region_id = T5.id WHERE T1.games_name = ?", (region_name, games_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the sport name based on a partial event name
@app.get("/v1/olympics/sport_name_by_event", operation_id="get_sport_name_by_event", summary="Retrieves the name of the sport associated with a given partial event name. The operation searches for events that contain the provided partial name and returns the corresponding sport name. The input parameter is used to specify the partial event name.")
async def get_sport_name_by_event(event_name: str = Query(..., description="Partial event name")):
    cursor.execute("SELECT T1.sport_name FROM sport AS T1 INNER JOIN event AS T2 ON T1.id = T2.sport_id WHERE T2.event_name LIKE ?", (event_name,))
    result = cursor.fetchall()
    if not result:
        return {"sport_names": []}
    return {"sport_names": [row[0] for row in result]}

# Endpoint to get the count of events in a specific sport
@app.get("/v1/olympics/count_events_by_sport", operation_id="get_count_events_by_sport", summary="Retrieves the total number of events associated with a specified sport. The operation requires the name of the sport as an input parameter and returns the count of events linked to that sport.")
async def get_count_events_by_sport(sport_name: str = Query(..., description="Name of the sport")):
    cursor.execute("SELECT COUNT(T2.event_name) FROM sport AS T1 INNER JOIN event AS T2 ON T1.id = T2.sport_id WHERE T1.sport_name = ?", (sport_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the game IDs based on a specific athlete's full name
@app.get("/v1/olympics/game_ids_by_athlete", operation_id="get_game_ids_by_athlete", summary="Retrieves the IDs of all games in which a specified athlete has participated. The athlete is identified by their full name.")
async def get_game_ids_by_athlete(full_name: str = Query(..., description="Full name of the athlete")):
    cursor.execute("SELECT T2.games_id FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id WHERE T1.full_name = ?", (full_name,))
    result = cursor.fetchall()
    if not result:
        return {"game_ids": []}
    return {"game_ids": [row[0] for row in result]}

# Endpoint to get the sport with the most events
@app.get("/v1/olympics/sport_with_most_events", operation_id="get_sport_with_most_events", summary="Retrieves the sport with the highest number of events, up to a specified limit. This operation returns the name of the sport that has the most events associated with it in the database. The limit parameter allows you to restrict the number of results returned.")
async def get_sport_with_most_events(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.sport_name FROM sport AS T1 INNER JOIN event AS T2 ON T1.id = T2.sport_id GROUP BY T1.sport_name ORDER BY COUNT(T2.event_name) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"sport_name": []}
    return {"sport_name": result[0]}

# Endpoint to get the tallest person from a specific region
@app.get("/v1/olympics/tallest_person_by_region", operation_id="get_tallest_person_by_region", summary="Retrieves the tallest person from a specified region, with the option to limit the number of results returned. The operation uses the region name to identify the relevant data and returns the full name of the tallest person(s) based on height.")
async def get_tallest_person_by_region(region_name: str = Query(..., description="Name of the region"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.full_name FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ? ORDER BY T3.height DESC LIMIT ?", (region_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the person with the most competitions
@app.get("/v1/olympics/person_with_most_competitions", operation_id="get_person_with_most_competitions", summary="Retrieves the person who has participated in the most competitions, based on the provided limit. The response is sorted in descending order, with the person having the highest number of competitions appearing first.")
async def get_person_with_most_competitions(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.full_name FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id GROUP BY T2.person_id ORDER BY COUNT(T2.person_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the full name of a competitor by their ID
@app.get("/v1/olympics/competitor_full_name_by_id", operation_id="get_competitor_full_name_by_id", summary="Retrieves the full name of a specific competitor based on their unique identifier. The competitor's ID is used to locate their full name in the database, which is then returned as the response.")
async def get_competitor_full_name_by_id(competitor_id: int = Query(..., description="ID of the competitor")):
    cursor.execute("SELECT T1.full_name FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id WHERE T2.id = ?", (competitor_id,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the count of competitions for a person by their full name
@app.get("/v1/olympics/count_competitions_by_full_name", operation_id="get_count_competitions_by_full_name", summary="Retrieves the total number of competitions a person has participated in, based on their full name.")
async def get_count_competitions_by_full_name(full_name: str = Query(..., description="Full name of the person")):
    cursor.execute("SELECT COUNT(T2.id) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id WHERE T1.full_name = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the BMI of a competitor by their ID
@app.get("/v1/olympics/competitor_bmi_by_id", operation_id="get_competitor_bmi_by_id", summary="Retrieves the Body Mass Index (BMI) of a specific competitor, calculated based on their weight and height. The competitor is identified by their unique ID.")
async def get_competitor_bmi_by_id(competitor_id: int = Query(..., description="ID of the competitor")):
    cursor.execute("SELECT CAST(T1.weight AS REAL) / (T1.height * T1.height) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id WHERE T2.id = ?", (competitor_id,))
    result = cursor.fetchone()
    if not result:
        return {"bmi": []}
    return {"bmi": result[0]}

# Endpoint to get the percentage of male competitors from a specific region
@app.get("/v1/olympics/percentage_male_competitors_by_region", operation_id="get_percentage_male_competitors_by_region", summary="Retrieves the percentage of male competitors from a specified region. This operation calculates the ratio of male competitors to the total number of competitors in the given region. The region is identified by its name.")
async def get_percentage_male_competitors_by_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.gender = 'M' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.person_id) FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ?", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the youngest competitor
@app.get("/v1/olympics/youngest_competitor", operation_id="get_youngest_competitor", summary="Retrieves the youngest competitor(s) from the database, up to the specified limit. The operation returns the full name of the youngest competitor(s) based on their age at the time of the competition.")
async def get_youngest_competitor(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.full_name FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id ORDER BY T2.age LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the count of competitors in a specific event with a specific age
@app.get("/v1/olympics/count_competitors_by_event_and_age", operation_id="get_count_competitors_by_event_and_age", summary="Retrieves the total number of competitors participating in a specified event and age group. The event name can be partially matched using wildcard characters. The age parameter is used to filter competitors by their age.")
async def get_count_competitors_by_event_and_age(event_name: str = Query(..., description="Name of the event (use %% for wildcard)"), age: int = Query(..., description="Age of the competitors")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM competitor_event AS T1 INNER JOIN games_competitor AS T2 ON T1.competitor_id = T2.id INNER JOIN event AS T3 ON T1.event_id = T3.id WHERE T3.event_name LIKE ? AND T2.age = ?", (event_name, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get event names for a specific sport
@app.get("/v1/olympics/event_names_by_sport", operation_id="get_event_names_by_sport", summary="Retrieves the names of all events associated with a specified sport. The operation filters events based on the provided sport name and returns a list of corresponding event names.")
async def get_event_names_by_sport(sport_name: str = Query(..., description="Name of the sport")):
    cursor.execute("SELECT T2.event_name FROM sport AS T1 INNER JOIN event AS T2 ON T1.id = T2.sport_id WHERE T1.sport_name = ?", (sport_name,))
    result = cursor.fetchall()
    if not result:
        return {"event_names": []}
    return {"event_names": [row[0] for row in result]}

# Endpoint to get the count of medals for a specific person and medal type
@app.get("/v1/olympics/medal_count_by_person_and_medal", operation_id="get_medal_count_by_person_and_medal", summary="Retrieves the total number of a specific type of medal won by a given person. The operation requires the full name of the person and the name of the medal as input parameters. It returns the count of medals that match the provided criteria.")
async def get_medal_count_by_person_and_medal(full_name: str = Query(..., description="Full name of the person"), medal_name: str = Query(..., description="Name of the medal")):
    cursor.execute("SELECT COUNT(T1.id) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id INNER JOIN competitor_event AS T3 ON T2.id = T3.competitor_id INNER JOIN medal AS T4 ON T3.medal_id = T4.id WHERE T1.full_name = ? AND T4.medal_name = ?", (full_name, medal_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the event with the most gold medals
@app.get("/v1/olympics/event_with_most_gold_medals", operation_id="get_event_with_most_gold_medals", summary="Retrieves the event that has the highest number of gold medals. The operation filters the results based on the specified medal name, which is provided as an input parameter.")
async def get_event_with_most_gold_medals(medal_name: str = Query(..., description="Name of the medal")):
    cursor.execute("SELECT T2.event_name FROM competitor_event AS T1 INNER JOIN event AS T2 ON T1.event_id = T2.id INNER JOIN medal AS T3 ON T1.medal_id = T3.id WHERE T3.medal_name = ? GROUP BY T2.id ORDER BY COUNT(T1.event_id) DESC LIMIT 1", (medal_name,))
    result = cursor.fetchone()
    if not result:
        return {"event_name": []}
    return {"event_name": result[0]}

# Endpoint to get the count of persons from a specific region
@app.get("/v1/olympics/person_count_by_region", operation_id="get_person_count_by_region", summary="Retrieves the total number of persons associated with a specified region. The region is identified by its name.")
async def get_person_count_by_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id WHERE T1.region_name = ?", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get cities that have hosted a minimum number of games
@app.get("/v1/olympics/cities_hosting_minimum_games", operation_id="get_cities_hosting_minimum_games", summary="Retrieves a list of cities that have hosted at least the specified minimum number of games. The response includes the names of these cities, which are determined by querying the database for cities that meet the minimum hosting requirement.")
async def get_cities_hosting_minimum_games(min_games: int = Query(..., description="Minimum number of games hosted")):
    cursor.execute("SELECT T2.city_name FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id GROUP BY T2.id HAVING COUNT(T1.games_id) >= ?", (min_games,))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get the count of games in a specific city and season
@app.get("/v1/olympics/game_count_by_city_and_season", operation_id="get_game_count_by_city_and_season", summary="Retrieves the total number of games held in a specified city during a particular season. The operation requires the city name and the season as input parameters to filter the games and calculate the count.")
async def get_game_count_by_city_and_season(city_name: str = Query(..., description="Name of the city"), season: str = Query(..., description="Season of the games")):
    cursor.execute("SELECT COUNT(T3.id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = ? AND T3.season = ?", (city_name, season))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the region with the most persons
@app.get("/v1/olympics/region_with_most_persons", operation_id="get_region_with_most_persons", summary="Retrieves the name of the region with the highest number of persons. This operation aggregates and counts the number of persons associated with each region, then returns the region with the maximum count. The result is determined by joining the person_region and noc_region tables on the region_id field.")
async def get_region_with_most_persons():
    cursor.execute("SELECT T2.region_name FROM person_region AS T1 INNER JOIN noc_region AS T2 ON T1.region_id = T2.id GROUP BY T2.region_name ORDER BY COUNT(T1.person_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"region_name": []}
    return {"region_name": result[0]}

# Endpoint to get the city of the earliest game
@app.get("/v1/olympics/city_of_earliest_game", operation_id="get_city_of_earliest_game", summary="Retrieves the name of the city that hosted the earliest Olympic Games. This operation fetches the city name from the database by joining the 'games_city', 'city', and 'games' tables, and then orders the results by the year of the games. The city with the earliest game year is returned.")
async def get_city_of_earliest_game():
    cursor.execute("SELECT T2.city_name FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id ORDER BY T3.games_year LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the game with the youngest competitor
@app.get("/v1/olympics/game_with_youngest_competitor", operation_id="get_game_with_youngest_competitor", summary="Retrieves the name of the Olympic game that had the youngest competitor. The operation fetches the game's name from the games table and joins it with the games_competitor table to determine the age of the competitors. The game with the youngest competitor is then returned.")
async def get_game_with_youngest_competitor():
    cursor.execute("SELECT T1.games_name FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id ORDER BY T2.age LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"games_name": []}
    return {"games_name": result[0]}

# Endpoint to get the count of competitors in a specific game
@app.get("/v1/olympics/competitor_count_by_game", operation_id="get_competitor_count_by_game", summary="Retrieves the total number of competitors who participated in a specific Olympic game. The game is identified by its name, which is provided as an input parameter.")
async def get_competitor_count_by_game(games_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.games_name = ?", (games_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of events and percentage of a specific medal type for a given athlete
@app.get("/v1/olympics/athlete_event_count_medal_percentage", operation_id="get_athlete_event_count_medal_percentage", summary="Retrieves the total number of events and the percentage of a specific medal type that an athlete has won. The athlete is identified by their full name, and the medal type is determined by its unique ID. This operation provides insights into an athlete's performance and medal distribution across events.")
async def get_athlete_event_count_medal_percentage(medal_id: int = Query(..., description="ID of the medal"), full_name: str = Query(..., description="Full name of the athlete")):
    cursor.execute("SELECT COUNT(T3.event_id) , CAST(COUNT(CASE WHEN T4.id = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T4.id) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id INNER JOIN competitor_event AS T3 ON T2.id = T3.competitor_id INNER JOIN medal AS T4 ON T3.medal_id = T4.id WHERE T1.full_name = ?", (medal_id, full_name))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "percentage": []}
    return {"count": result[0], "percentage": result[1]}

# Endpoint to get the season with the highest difference in competitor counts between two specified games
@app.get("/v1/olympics/season_max_competitor_difference", operation_id="get_season_max_competitor_difference", summary="Retrieves the season with the maximum difference in the number of competitors between two specified games. The operation compares the competitor counts for each season across the provided games and identifies the season with the highest disparity.")
async def get_season_max_competitor_difference(game_name_1: str = Query(..., description="Name of the first game"), game_name_2: str = Query(..., description="Name of the second game")):
    cursor.execute("SELECT P1 , ( SELECT MAX(P2) - MIN(P2) FROM ( SELECT COUNT(T2.person_id) AS P2 FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.games_name IN (?, ?) GROUP BY T1.season ) ORDER BY P2 DESC LIMIT 1 ) FROM ( SELECT T1.season AS P1, COUNT(T2.person_id) AS P2 FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.games_name IN (?, ?) GROUP BY T1.season ) ORDER BY P2 DESC LIMIT 1", (game_name_1, game_name_2, game_name_1, game_name_2))
    result = cursor.fetchone()
    if not result:
        return {"season": [], "difference": []}
    return {"season": result[0], "difference": result[1]}

# Endpoint to get the most common age of competitors
@app.get("/v1/olympics/most_common_competitor_age", operation_id="get_most_common_competitor_age", summary="Retrieves the age that is most frequently observed among competitors in the database. The result is determined by grouping competitors by their age and then ordering the groups by the count of competitors in each group in descending order. The age from the group with the highest count is returned.")
async def get_most_common_competitor_age():
    cursor.execute("SELECT age FROM games_competitor GROUP BY age ORDER BY COUNT(person_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get city names that start with a specific letter
@app.get("/v1/olympics/city_names_starting_with", operation_id="get_city_names_starting_with", summary="Retrieves a list of city names that start with a specified letter. The input parameter determines the initial letter of the city names to be returned.")
async def get_city_names_starting_with(letter: str = Query(..., description="Letter the city name starts with")):
    cursor.execute("SELECT city_name FROM city WHERE city_name LIKE ?", (letter + '%',))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get city names for specified games
@app.get("/v1/olympics/city_names_for_games", operation_id="get_city_names_for_games", summary="Retrieves the names of cities that have hosted the specified Olympic Games. The operation accepts the names of two Olympic Games as input parameters and returns a list of city names that have hosted either of the specified games.")
async def get_city_names_for_games(game_name_1: str = Query(..., description="Name of the first game"), game_name_2: str = Query(..., description="Name of the second game")):
    cursor.execute("SELECT T2.city_name FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T3.games_name IN (?, ?)", (game_name_1, game_name_2))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get medal names for a specific athlete
@app.get("/v1/olympics/athlete_medal_names", operation_id="get_athlete_medal_names", summary="Retrieves the names of all medals won by a specific athlete. The athlete is identified by their full name, which is used to search across multiple tables: person, games_competitor, competitor_event, and medal. The operation returns a list of medal names associated with the athlete.")
async def get_athlete_medal_names(full_name: str = Query(..., description="Full name of the athlete")):
    cursor.execute("SELECT T4.medal_name FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id INNER JOIN competitor_event AS T3 ON T2.id = T3.competitor_id INNER JOIN medal AS T4 ON T3.medal_id = T4.id WHERE T1.full_name = ?", (full_name,))
    result = cursor.fetchall()
    if not result:
        return {"medal_names": []}
    return {"medal_names": [row[0] for row in result]}

# Endpoint to get the athlete with the most medals excluding a specific medal type
@app.get("/v1/olympics/athlete_most_medals_excluding", operation_id="get_athlete_most_medals_excluding", summary="Retrieves the name of the athlete who has won the most medals, excluding a specific medal type. The operation filters out the specified medal type and calculates the total medals won by each athlete. The athlete with the highest count is returned.")
async def get_athlete_most_medals_excluding(medal_id: int = Query(..., description="ID of the medal to exclude")):
    cursor.execute("SELECT T1.full_name FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id INNER JOIN competitor_event AS T3 ON T2.id = T3.competitor_id INNER JOIN medal AS T4 ON T3.medal_id = T4.id WHERE T4.id != ? GROUP BY T1.full_name ORDER BY COUNT(T4.id) DESC LIMIT 1", (medal_id,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get distinct sport names for a specific athlete
@app.get("/v1/olympics/athlete_sport_names", operation_id="get_athlete_sport_names", summary="Retrieves a unique list of sport names in which a specific athlete has participated. The athlete is identified by their full name.")
async def get_athlete_sport_names(full_name: str = Query(..., description="Full name of the athlete")):
    cursor.execute("SELECT DISTINCT T1.sport_name FROM sport AS T1 INNER JOIN event AS T2 ON T1.id = T2.sport_id INNER JOIN competitor_event AS T3 ON T2.id = T3.event_id INNER JOIN games_competitor AS T4 ON T3.competitor_id = T4.id INNER JOIN person AS T5 ON T4.person_id = T5.id WHERE T5.full_name = ?", (full_name,))
    result = cursor.fetchall()
    if not result:
        return {"sport_names": []}
    return {"sport_names": [row[0] for row in result]}

# Endpoint to get the oldest competitor
@app.get("/v1/olympics/oldest_competitor", operation_id="get_oldest_competitor", summary="Retrieves the full name of the oldest competitor in the Olympics. The operation returns the oldest competitor based on their age at the time of participation, as recorded in the games_competitor table. The result is determined by joining the person and games_competitor tables and sorting the records in descending order by age.")
async def get_oldest_competitor():
    cursor.execute("SELECT T1.full_name FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id ORDER BY T2.age DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the game name with the highest number of female competitors
@app.get("/v1/olympics/game_with_most_female_competitors", operation_id="get_game_with_most_female_competitors", summary="Retrieves the name of the Olympic game that has the highest number of female competitors. The gender of the competitors is specified as an input parameter.")
async def get_game_with_most_female_competitors(gender: str = Query(..., description="Gender of the competitors (e.g., 'F' for female)")):
    cursor.execute("SELECT T1.games_name FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.gender = ? GROUP BY T1.games_name ORDER BY COUNT(T2.person_id) DESC LIMIT 1", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"game_name": []}
    return {"game_name": result[0]}

# Endpoint to get the count of male competitors from a specific region
@app.get("/v1/olympics/count_male_competitors_by_region", operation_id="get_count_male_competitors_by_region", summary="Retrieves the total number of male competitors from a specified region. The operation filters competitors based on their region and gender, then aggregates the count of male competitors from the selected region.")
async def get_count_male_competitors_by_region(region_name: str = Query(..., description="Name of the region"), gender: str = Query(..., description="Gender of the competitors (e.g., 'M' for male)")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ? AND T3.gender = ?", (region_name, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of competitors in a specific city
@app.get("/v1/olympics/count_competitors_by_city", operation_id="get_count_competitors_by_city", summary="Retrieves the total number of competitors who participated in the Olympics held in a specific city. The city is identified by its name.")
async def get_count_competitors_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT COUNT(T1.person_id) FROM games_competitor AS T1 INNER JOIN games_city AS T2 ON T1.games_id = T2.games_id INNER JOIN city AS T3 ON T2.city_id = T3.id WHERE T3.city_name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of competitors in a specific event
@app.get("/v1/olympics/count_competitors_by_event", operation_id="get_count_competitors_by_event", summary="Retrieves the total number of competitors participating in a specific event. The event is identified by its name, which is provided as an input parameter.")
async def get_count_competitors_by_event(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT COUNT(T1.competitor_id) FROM competitor_event AS T1 INNER JOIN event AS T2 ON T1.event_id = T2.id INNER JOIN sport AS T3 ON T2.sport_id = T3.id WHERE T2.event_name = ?", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct events in a specific sport
@app.get("/v1/olympics/count_distinct_events_by_sport", operation_id="get_count_distinct_events_by_sport", summary="Retrieves the total number of unique events associated with a specific sport. The operation requires the name of the sport as an input parameter to filter the results.")
async def get_count_distinct_events_by_sport(sport_name: str = Query(..., description="Name of the sport")):
    cursor.execute("SELECT COUNT(DISTINCT T2.event_name) FROM sport AS T1 INNER JOIN event AS T2 ON T1.id = T2.sport_id WHERE T1.sport_name = ?", (sport_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of female competitors in a specific event
@app.get("/v1/olympics/percentage_female_competitors_by_event", operation_id="get_percentage_female_competitors_by_event", summary="Retrieves the percentage of competitors of a specified gender in a given event. The operation calculates this percentage by counting the number of competitors of the specified gender and dividing it by the total number of competitors in the event. The result is then multiplied by 100 to express it as a percentage.")
async def get_percentage_female_competitors_by_event(gender: str = Query(..., description="Gender of the competitors (e.g., 'F' for female)"), event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.gender = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id INNER JOIN competitor_event AS T3 ON T2.id = T3.competitor_id INNER JOIN event AS T4 ON T3.event_id = T4.id WHERE T4.event_name = ?", (gender, event_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average age of competitors in a specific game
@app.get("/v1/olympics/average_age_by_game", operation_id="get_average_age_by_game", summary="Retrieves the average age of competitors who participated in a specified game. The operation calculates the mean age of all competitors in the given game, providing a statistical overview of the age distribution.")
async def get_average_age_by_game(games_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT AVG(T2.age) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.games_name = ?", (games_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the years with only one season type
@app.get("/v1/olympics/years_with_single_season", operation_id="get_years_with_single_season", summary="Retrieves the years in which only a single season type of Olympic games was held. The season type is specified as an input parameter, and the operation returns the years that meet the criteria.")
async def get_years_with_single_season(season: str = Query(..., description="Season type (e.g., 'Winter')")):
    cursor.execute("SELECT games_year FROM games WHERE season != ? GROUP BY games_year HAVING COUNT(season) = 1", (season,))
    result = cursor.fetchall()
    if not result:
        return {"years": []}
    return {"years": [row[0] for row in result]}

# Endpoint to get the count of games in a specific year range
@app.get("/v1/olympics/count_games_in_year_range", operation_id="get_count_games_in_year_range", summary="Retrieves the total number of games that took place between the specified start and end years, inclusive.")
async def get_count_games_in_year_range(start_year: int = Query(..., description="Start year (e.g., 1990)"), end_year: int = Query(..., description="End year (e.g., 1999)")):
    cursor.execute("SELECT COUNT(games_year) FROM games WHERE games_year BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of competitors from a specific region who won a medal
@app.get("/v1/olympics/competitor_count_by_region_and_medal", operation_id="get_competitor_count_by_region_and_medal", summary="Retrieves the total number of competitors from a specified region who have won any medal, excluding a particular medal type. The operation filters competitors based on their region and medal records, then returns the count of unique competitors who meet the criteria.")
async def get_competitor_count_by_region_and_medal(region_id: int = Query(..., description="ID of the region"), medal_id: int = Query(..., description="ID of the medal to exclude")):
    cursor.execute("SELECT COUNT(T3.person_id) FROM competitor_event AS T1 INNER JOIN games_competitor AS T2 ON T1.competitor_id = T2.id INNER JOIN person_region AS T3 ON T2.person_id = T3.person_id WHERE T3.region_id = ? AND T1.medal_id != ?", (region_id, medal_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of competitors from a specific region in a specific game
@app.get("/v1/olympics/competitor_count_by_game_and_region", operation_id="get_competitor_count_by_game_and_region", summary="Retrieves the total number of competitors from a specified region who participated in a particular Olympic game. The operation requires the name of the game and the region as input parameters.")
async def get_competitor_count_by_game_and_region(games_name: str = Query(..., description="Name of the game"), region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT COUNT(T3.id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id INNER JOIN person_region AS T4 ON T3.id = T4.person_id INNER JOIN noc_region AS T5 ON T4.region_id = T5.id WHERE T1.games_name = ? AND T5.region_name = ?", (games_name, region_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of competitors in a specific event who won a specific medal
@app.get("/v1/olympics/competitor_count_by_event_and_medal", operation_id="get_competitor_count_by_event_and_medal", summary="Retrieves the total number of competitors who participated in a specified event and won a particular medal. The event is identified by its name, and the medal is determined by its unique ID.")
async def get_competitor_count_by_event_and_medal(event_name: str = Query(..., description="Name of the event"), medal_id: int = Query(..., description="ID of the medal")):
    cursor.execute("SELECT COUNT(T2.competitor_id) FROM event AS T1 INNER JOIN competitor_event AS T2 ON T1.id = T2.event_id WHERE T1.event_name LIKE ? AND T2.medal_id = ?", (event_name, medal_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the region with the highest number of competitors who won a specific medal
@app.get("/v1/olympics/top_region_by_medal", operation_id="get_top_region_by_medal", summary="Retrieves the region with the most competitors who have won any medal, excluding the one specified by the input parameter. The operation returns the name of the region with the highest count of medal-winning competitors.")
async def get_top_region_by_medal(medal_id: int = Query(..., description="ID of the medal to exclude")):
    cursor.execute("SELECT T5.region_name FROM medal AS T1 INNER JOIN competitor_event AS T2 ON T1.id = T2.medal_id INNER JOIN games_competitor AS T3 ON T2.competitor_id = T3.id INNER JOIN person_region AS T4 ON T3.person_id = T4.person_id INNER JOIN noc_region AS T5 ON T4.region_id = T5.id WHERE T1.id != ? GROUP BY T5.region_name ORDER BY COUNT(T2.competitor_id) DESC LIMIT 1", (medal_id,))
    result = cursor.fetchone()
    if not result:
        return {"region_name": []}
    return {"region_name": result[0]}

# Endpoint to get the distinct names of competitors who won a specific medal in a specific sport
@app.get("/v1/olympics/competitor_names_by_sport_and_medal", operation_id="get_competitor_names_by_sport_and_medal", summary="Retrieve the unique names of competitors who have won a specific medal in a given sport. The operation requires the sport's name and the medal's ID as input parameters.")
async def get_competitor_names_by_sport_and_medal(sport_name: str = Query(..., description="Name of the sport"), medal_id: int = Query(..., description="ID of the medal")):
    cursor.execute("SELECT DISTINCT T5.full_name FROM event AS T1 INNER JOIN competitor_event AS T2 ON T1.id = T2.event_id INNER JOIN games_competitor AS T3 ON T2.competitor_id = T3.id INNER JOIN sport AS T4 ON T1.sport_id = T4.id INNER JOIN person AS T5 ON T3.person_id = T5.id WHERE T4.sport_name = ? AND T2.medal_id = ?", (sport_name, medal_id))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the distinct medal names won by a specific competitor
@app.get("/v1/olympics/medals_by_competitor", operation_id="get_medals_by_competitor", summary="Retrieve the unique medal names won by a specific competitor, excluding a particular medal. The operation filters the medals by the competitor's full name and the ID of the medal to be excluded.")
async def get_medals_by_competitor(full_name: str = Query(..., description="Full name of the competitor"), medal_id: int = Query(..., description="ID of the medal to exclude")):
    cursor.execute("SELECT DISTINCT T1.medal_name FROM medal AS T1 INNER JOIN competitor_event AS T2 ON T1.id = T2.medal_id INNER JOIN games_competitor AS T3 ON T2.competitor_id = T3.id INNER JOIN person AS T4 ON T3.person_id = T4.id WHERE T4.full_name = ? AND T2.medal_id <> ?", (full_name, medal_id))
    result = cursor.fetchall()
    if not result:
        return {"medal_names": []}
    return {"medal_names": [row[0] for row in result]}

# Endpoint to get the percentage of female competitors under a certain age in a specific game
@app.get("/v1/olympics/female_percentage_by_game_and_age", operation_id="get_female_percentage_by_game_and_age", summary="Retrieves the percentage of female competitors under a specified age in a given game. The operation calculates this percentage by counting the number of female competitors under the specified age and dividing it by the total number of competitors in the game.")
async def get_female_percentage_by_game_and_age(games_name: str = Query(..., description="Name of the game"), age: int = Query(..., description="Age threshold")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.gender = 'F' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.person_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.games_name = ? AND T2.age < ?", (games_name, age))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of male to female competitors in a specific game
@app.get("/v1/olympics/male_to_female_ratio_by_game", operation_id="get_male_to_female_ratio_by_game", summary="Retrieves the ratio of male to female competitors participating in a specified game. The operation calculates this ratio by counting the number of male and female competitors in the given game and returns the result as a real number.")
async def get_male_to_female_ratio_by_game(games_name: str = Query(..., description="Name of the game")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.gender = 'M' THEN 1 ELSE NULL END) AS REAL) / COUNT(CASE WHEN T3.gender = 'F' THEN 1 ELSE NULL END) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.games_name = ?", (games_name,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the games with the highest and lowest number of competitors in a specific season
@app.get("/v1/olympics/highest_and_lowest_competitor_games_by_season", operation_id="get_highest_and_lowest_competitor_games_by_season", summary="Retrieves the games with the highest and lowest number of competitors for a specified season. The season parameter is used to filter the games, and the results are ordered by the count of competitors to determine the games with the highest and lowest number of participants.")
async def get_highest_and_lowest_competitor_games_by_season(season: str = Query(..., description="Season of the games")):
    cursor.execute("SELECT ( SELECT T1.games_name FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.season = ? GROUP BY T1.games_year ORDER BY COUNT(T2.person_id) DESC LIMIT 1 ) AS HIGHEST , ( SELECT T1.games_name FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.season = ? GROUP BY T1.games_year ORDER BY COUNT(T2.person_id) LIMIT 1 ) AS LOWEST", (season, season))
    result = cursor.fetchone()
    if not result:
        return {"highest": [], "lowest": []}
    return {"highest": result[0], "lowest": result[1]}

# Endpoint to get the count of medalists from a specific region excluding a specific medal ID
@app.get("/v1/olympics/medalist_count_region_exclude_medal", operation_id="get_medalist_count", summary="Retrieves the total number of medalists from a specified region, excluding those who have won a particular medal. The operation filters the results based on the provided region name and the ID of the medal to be excluded.")
async def get_medalist_count(region_name: str = Query(..., description="Name of the region"), exclude_medal_id: int = Query(..., description="Medal ID to exclude")):
    cursor.execute("SELECT COUNT(T3.person_id) FROM medal AS T1 INNER JOIN competitor_event AS T2 ON T1.id = T2.medal_id INNER JOIN games_competitor AS T3 ON T2.competitor_id = T3.id INNER JOIN person_region AS T4 ON T3.person_id = T4.person_id INNER JOIN noc_region AS T5 ON T4.region_id = T5.id WHERE T5.region_name = ? AND T1.id != ?", (region_name, exclude_medal_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of competitors in a specific sport and games with a specific BMI range
@app.get("/v1/olympics/competitor_count_sport_games_bmi", operation_id="get_competitor_count", summary="Retrieves the count of competitors who participated in a specific sport and games, within a defined BMI range. The BMI range is determined by the minimum and maximum BMI values provided as input parameters. The operation considers the weight and height of each competitor to calculate their BMI and filter the results accordingly.")
async def get_competitor_count(sport_name: str = Query(..., description="Name of the sport"), games_name: str = Query(..., description="Name of the games"), bmi_min: float = Query(..., description="Minimum BMI"), bmi_max: float = Query(..., description="Maximum BMI")):
    cursor.execute("SELECT COUNT(T5.id) FROM sport AS T1 INNER JOIN event AS T2 ON T1.id = T2.sport_id INNER JOIN competitor_event AS T3 ON T2.id = T3.event_id INNER JOIN games_competitor AS T4 ON T3.competitor_id = T4.id INNER JOIN person AS T5 ON T4.person_id = T5.id INNER JOIN games AS T6 ON T4.games_id = T6.id WHERE T1.sport_name = ? AND T6.games_name = ? AND T5.weight * 10000.0 / (T5.height * T5.height) BETWEEN ? AND ?", (sport_name, games_name, bmi_min, bmi_max))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average height of competitors in a specific sport and gender
@app.get("/v1/olympics/avg_height_sport_gender", operation_id="get_avg_height", summary="Retrieves the average height of competitors in a specified sport and gender. The operation calculates the average height by considering the height data of all competitors who have participated in the given sport and gender. The sport and gender are provided as input parameters.")
async def get_avg_height(sport_name: str = Query(..., description="Name of the sport"), gender: str = Query(..., description="Gender of the competitor")):
    cursor.execute("SELECT AVG(T5.height) FROM sport AS T1 INNER JOIN event AS T2 ON T1.id = T2.sport_id INNER JOIN competitor_event AS T3 ON T2.id = T3.event_id INNER JOIN games_competitor AS T4 ON T3.competitor_id = T4.id INNER JOIN person AS T5 ON T4.person_id = T5.id WHERE T1.sport_name = ? AND T5.gender = ?", (sport_name, gender))
    result = cursor.fetchone()
    if not result:
        return {"avg_height": []}
    return {"avg_height": result[0]}

# Endpoint to get the youngest age of a specific person in games
@app.get("/v1/olympics/youngest_age_by_person", operation_id="get_youngest_age", summary="Retrieve the youngest age at which a specific person, identified by their full name, has competed in games. The operation returns the minimum age of the person in the games they have participated in.")
async def get_youngest_age(full_name: str = Query(..., description="Full name of the person")):
    cursor.execute("SELECT T2.age FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id WHERE T1.full_name = ? ORDER BY T2.age LIMIT 1", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the count of persons from the same region as a specific person
@app.get("/v1/olympics/person_count_same_region", operation_id="get_person_count_same_region", summary="Retrieves the total number of individuals hailing from the same region as the specified person. The person is identified by their full name.")
async def get_person_count_same_region(full_name: str = Query(..., description="Full name of the person")):
    cursor.execute("SELECT COUNT(person_id) FROM person_region WHERE region_id = ( SELECT T1.region_id FROM person_region AS T1 INNER JOIN person AS T2 ON T1.person_id = T2.id WHERE T2.full_name = ? )", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of events a specific person competed in with a specific event name pattern
@app.get("/v1/olympics/event_count_person_event_name", operation_id="get_event_count", summary="Retrieve the total number of events a specific individual has participated in, based on a specified event name pattern. This operation filters the results by the full name of the person and the event name pattern, providing a count of matching events.")
async def get_event_count(full_name: str = Query(..., description="Full name of the person"), event_name_pattern: str = Query(..., description="Pattern of the event name")):
    cursor.execute("SELECT COUNT(T1.id) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id INNER JOIN competitor_event AS T3 ON T2.id = T3.competitor_id INNER JOIN event AS T4 ON T3.event_id = T4.id WHERE T1.full_name = ? AND T4.event_name LIKE ?", (full_name, event_name_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of medals a specific person won in events with a specific name pattern
@app.get("/v1/olympics/medal_count_person_event_name", operation_id="get_medal_count", summary="Retrieve the total number of medals a specific individual has won in events that match a given name pattern. The operation requires the full name of the person, a pattern for the event name, and the ID of the medal type. The result reflects the count of medals won by the person in events that meet the specified criteria.")
async def get_medal_count(full_name: str = Query(..., description="Full name of the person"), event_name_pattern: str = Query(..., description="Pattern of the event name"), medal_id: int = Query(..., description="Medal ID")):
    cursor.execute("SELECT COUNT(T1.id) FROM event AS T1 INNER JOIN competitor_event AS T2 ON T1.id = T2.event_id INNER JOIN games_competitor AS T3 ON T2.competitor_id = T3.id INNER JOIN person AS T4 ON T3.person_id = T4.id WHERE T4.full_name = ? AND T1.event_name LIKE ? AND T2.medal_id = ?", (full_name, event_name_pattern, medal_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city names where a specific person competed
@app.get("/v1/olympics/city_names_by_person", operation_id="get_city_names", summary="Retrieves the names of cities where a specific individual has participated in competitions. The operation requires the full name of the person as input to filter the results.")
async def get_city_names(full_name: str = Query(..., description="Full name of the person")):
    cursor.execute("SELECT T4.city_name FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id INNER JOIN games_city AS T3 ON T2.games_id = T3.games_id INNER JOIN city AS T4 ON T3.city_id = T4.id WHERE T1.full_name = ?", (full_name,))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get the game name with the most competitors in a specific city
@app.get("/v1/olympics/game_with_most_competitors_in_city", operation_id="get_game_with_most_competitors", summary="Retrieves the name of the game with the highest number of competitors in a specified city. The operation considers all games held in the city and counts the number of competitors for each game. The game with the most competitors is then returned.")
async def get_game_with_most_competitors(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.games_name FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN games_city AS T3 ON T2.games_id = T3.games_id INNER JOIN city AS T4 ON T3.city_id = T4.id WHERE T4.city_name = ? GROUP BY T1.id ORDER BY COUNT(T2.person_id) DESC LIMIT 1", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"game_name": []}
    return {"game_name": result[0]}

# Endpoint to get the average age of competitors in a specific game from a specific region
@app.get("/v1/olympics/average_age_by_game_and_region", operation_id="get_average_age", summary="Retrieves the average age of competitors who participated in a specific Olympic game from a particular region. The operation requires the name of the game and the region as input parameters.")
async def get_average_age(games_name: str = Query(..., description="Name of the game"), region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT AVG(T2.age) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person_region AS T3 ON T2.person_id = T3.person_id INNER JOIN noc_region AS T4 ON T3.region_id = T4.id WHERE T1.games_name = ? AND T4.region_name = ?", (games_name, region_name))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the region name based on NOC code
@app.get("/v1/olympics/region_name_by_noc", operation_id="get_region_name_by_noc", summary="Retrieves the region name associated with the provided National Olympic Committee (NOC) code. The operation uses the NOC code to look up the corresponding region name in the noc_region table.")
async def get_region_name_by_noc(noc: str = Query(..., description="NOC code")):
    cursor.execute("SELECT region_name FROM noc_region WHERE noc = ?", (noc,))
    result = cursor.fetchone()
    if not result:
        return {"region_name": []}
    return {"region_name": result[0]}

# Endpoint to get the sport name based on sport ID
@app.get("/v1/olympics/sport_name_by_id", operation_id="get_sport_name_by_id", summary="Retrieves the name of a specific sport by its unique identifier. The operation uses the provided sport ID to look up the corresponding sport name in the database.")
async def get_sport_name_by_id(sport_id: int = Query(..., description="Sport ID")):
    cursor.execute("SELECT sport_name FROM sport WHERE id = ?", (sport_id,))
    result = cursor.fetchone()
    if not result:
        return {"sport_name": []}
    return {"sport_name": result[0]}

# Endpoint to get the event ID based on event name
@app.get("/v1/olympics/event_id_by_name", operation_id="get_event_id_by_name", summary="Retrieves the unique identifier (ID) of a specific event from the database, based on the provided event name.")
async def get_event_id_by_name(event_name: str = Query(..., description="Event name")):
    cursor.execute("SELECT id FROM event WHERE event_name = ?", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"event_id": []}
    return {"event_id": result[0]}

# Endpoint to get the sport ID based on sport name
@app.get("/v1/olympics/sport_id_by_name", operation_id="get_sport_id_by_name", summary="Retrieves the unique identifier of a sport, given its name. The operation searches for a sport with the provided name and returns its corresponding ID.")
async def get_sport_id_by_name(sport_name: str = Query(..., description="Sport name")):
    cursor.execute("SELECT id FROM sport WHERE sport_name = ?", (sport_name,))
    result = cursor.fetchone()
    if not result:
        return {"sport_id": []}
    return {"sport_id": result[0]}

# Endpoint to get the weight of a person based on their full name
@app.get("/v1/olympics/person_weight_by_name", operation_id="get_person_weight_by_name", summary="Retrieves the weight of a specific individual, identified by their full name. The full name is a required input parameter.")
async def get_person_weight_by_name(full_name: str = Query(..., description="Full name of the person")):
    cursor.execute("SELECT weight FROM person WHERE full_name = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"weight": []}
    return {"weight": result[0]}

# Endpoint to get the city ID based on city name
@app.get("/v1/olympics/city_id_by_name", operation_id="get_city_id_by_name", summary="Retrieves the unique identifier of a city based on its name. The operation uses the provided city name to search for a match in the database and returns the corresponding city ID.")
async def get_city_id_by_name(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT id FROM city WHERE city_name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"city_id": []}
    return {"city_id": result[0]}

# Endpoint to get the count of competitors who won a specific medal in a specific event
@app.get("/v1/olympics/competitor_count_event_medal", operation_id="get_competitor_count_event_medal", summary="Retrieves the total number of competitors who have won a specific medal in an event matching the provided name pattern. The event name pattern supports wildcard characters for broader search criteria. The medal name must be an exact match.")
async def get_competitor_count_event_medal(event_name: str = Query(..., description="Event name pattern (use %% for wildcard)"), medal_name: str = Query(..., description="Medal name")):
    cursor.execute("SELECT COUNT(T1.competitor_id) FROM competitor_event AS T1 INNER JOIN event AS T2 ON T1.event_id = T2.id INNER JOIN medal AS T3 ON T1.medal_id = T3.id WHERE T2.event_name LIKE ? AND T3.medal_name = ?", (event_name, medal_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the games year for a specific city
@app.get("/v1/olympics/games_year_city", operation_id="get_games_year_city", summary="Retrieves the year of the Olympic Games held in the specified city. The operation uses the provided city name to search for a match in the database and returns the corresponding year of the Games.")
async def get_games_year_city(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT T3.games_year FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"games_year": []}
    return {"games_year": result[0]}

# Endpoint to get the count of competitors of a specific age in a specific games
@app.get("/v1/olympics/competitor_count_games_age", operation_id="get_competitor_count_games_age", summary="Retrieves the total number of competitors of a specified age who participated in a particular Olympic Games. The operation requires the name of the Games and the age of the competitors as input parameters.")
async def get_competitor_count_games_age(games_name: str = Query(..., description="Games name"), age: int = Query(..., description="Age of the competitor")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.games_name = ? AND T2.age = ?", (games_name, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of games a specific person has competed in
@app.get("/v1/olympics/games_count_person", operation_id="get_games_count_person", summary="Retrieves the total number of games a specific individual has participated in. The operation requires the full name of the person as an input parameter to identify the individual and calculate the count of games they have competed in.")
async def get_games_count_person(full_name: str = Query(..., description="Full name of the person")):
    cursor.execute("SELECT COUNT(T2.games_id) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id WHERE T1.full_name = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of competitors older than a specific age in a specific games
@app.get("/v1/olympics/competitor_count_games_age_greater", operation_id="get_competitor_count_games_age_greater", summary="Retrieves the total number of competitors who participated in a specific Olympic Games and are older than a given age. The input parameters include the name of the Olympic Games and the minimum age of the competitors.")
async def get_competitor_count_games_age_greater(games_name: str = Query(..., description="Games name"), age: int = Query(..., description="Minimum age of the competitor")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.games_name = ? AND T2.age > ?", (games_name, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the heaviest person from a specific region
@app.get("/v1/olympics/heaviest_person_region", operation_id="get_heaviest_person_region", summary="Retrieves the full name of the heaviest person from the specified region. The operation filters the person data based on the provided region name and returns the name of the heaviest individual.")
async def get_heaviest_person_region(region_name: str = Query(..., description="Region name")):
    cursor.execute("SELECT T3.full_name FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ? ORDER BY T3.weight DESC LIMIT 1", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the tallest person from a specific region
@app.get("/v1/olympics/tallest_person_region", operation_id="get_tallest_person_region", summary="Retrieves the height of the tallest person from the specified region. The operation filters the data by the provided region name and returns the maximum height value.")
async def get_tallest_person_region(region_name: str = Query(..., description="Region name")):
    cursor.execute("SELECT T3.height FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ? ORDER BY T3.height DESC LIMIT 1", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"height": []}
    return {"height": result[0]}

# Endpoint to get the percentage of athletes taller than a certain height from a specific region
@app.get("/v1/olympics/percentage_tall_athletes_by_region", operation_id="get_percentage_tall_athletes_by_region", summary="Retrieves the percentage of athletes from a specified region who are taller than a given height. The operation calculates this percentage by counting the number of athletes in the region who exceed the provided height threshold and dividing it by the total number of athletes in the region.")
async def get_percentage_tall_athletes_by_region(height: int = Query(..., description="Height threshold in cm"), region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.height > ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.person_id) FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ?", (height, region_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average weight of athletes from a specific region and gender
@app.get("/v1/olympics/average_weight_by_region_and_gender", operation_id="get_average_weight_by_region_and_gender", summary="Retrieves the average weight of athletes from a specified region and gender. The operation calculates the average weight by joining data from the noc_region, person_region, and person tables based on the provided region name and gender.")
async def get_average_weight_by_region_and_gender(region_name: str = Query(..., description="Name of the region"), gender: str = Query(..., description="Gender of the athletes (M or F)")):
    cursor.execute("SELECT AVG(T3.weight) FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.region_name = ? AND T3.gender = ?", (region_name, gender))
    result = cursor.fetchone()
    if not result:
        return {"average_weight": []}
    return {"average_weight": result[0]}

# Endpoint to get the count of games in a specific city within a year range
@app.get("/v1/olympics/count_games_by_city_and_year_range", operation_id="get_count_games_by_city_and_year_range", summary="Retrieves the total number of games held in a specified city within a given year range. The operation requires the city name and the start and end years of the range as input parameters.")
async def get_count_games_by_city_and_year_range(city_name: str = Query(..., description="Name of the city"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT COUNT(T3.id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = ? AND T3.games_year BETWEEN ? AND ?", (city_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city with the most games
@app.get("/v1/olympics/city_with_most_games", operation_id="get_city_with_most_games", summary="Retrieves the name of the city that has hosted the most Olympic Games. This operation returns the city with the highest count of hosted games, as determined by the data in the games_city and city tables. The result is based on the aggregation and ordering of the data, with the city name being the primary output.")
async def get_city_with_most_games():
    cursor.execute("SELECT T2.city_name FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id GROUP BY T2.city_name ORDER BY COUNT(T2.city_name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the games name for a specific city and year
@app.get("/v1/olympics/games_name_by_city_and_year", operation_id="get_games_name_by_city_and_year", summary="Retrieves the name of the Olympic games held in a specific city during a given year. The operation requires the city name and the year of the games as input parameters.")
async def get_games_name_by_city_and_year(city_name: str = Query(..., description="Name of the city"), games_year: int = Query(..., description="Year of the games")):
    cursor.execute("SELECT T3.games_name FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = ? AND T3.games_year = ?", (city_name, games_year))
    result = cursor.fetchone()
    if not result:
        return {"games_name": []}
    return {"games_name": result[0]}

# Endpoint to get the percentage of medalists older than a certain age
@app.get("/v1/olympics/percentage_medalists_older_than", operation_id="get_percentage_medalists_older_than", summary="Retrieves the percentage of medalists who are older than a specified age. This operation calculates the ratio of medalists above the given age threshold to the total number of medalists. The medalists are filtered by a specific medal ID.")
async def get_percentage_medalists_older_than(age: int = Query(..., description="Age threshold"), medal_id: int = Query(..., description="Medal ID")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.age > ? THEN 1 END) AS REAL) * 100 / COUNT(T2.person_id) FROM competitor_event AS T1 INNER JOIN games_competitor AS T2 ON T1.competitor_id = T2.id WHERE T1.medal_id = ?", (age, medal_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the age of a competitor in a specific game by their full name
@app.get("/v1/olympics/competitor_age_by_game_and_name", operation_id="get_competitor_age", summary="Retrieves the age of a specific competitor in a given game. The operation requires the full name of the competitor and the name of the game as input parameters. The age is determined by querying the games, games_competitor, and person tables, and matching the provided game name and competitor full name.")
async def get_competitor_age(games_name: str = Query(..., description="Name of the games"), full_name: str = Query(..., description="Full name of the competitor")):
    cursor.execute("SELECT T2.age FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.games_name = ? AND T3.full_name = ?", (games_name, full_name))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the count of competitors in a specific game by gender
@app.get("/v1/olympics/competitor_count_by_game_and_gender", operation_id="get_competitor_count_by_gender", summary="Retrieves the total number of competitors who participated in a specific game, categorized by gender. The operation requires the name of the game and the gender as input parameters to filter the count accordingly.")
async def get_competitor_count_by_gender(games_name: str = Query(..., description="Name of the games"), gender: str = Query(..., description="Gender of the competitor")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.games_name = ? AND T3.gender = ?", (games_name, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the full names of competitors in a specific game
@app.get("/v1/olympics/competitor_names_by_game", operation_id="get_competitor_names", summary="Retrieves the full names of all competitors who participated in a specified Olympic game. The game is identified by its name, which is provided as an input parameter.")
async def get_competitor_names(games_name: str = Query(..., description="Name of the games")):
    cursor.execute("SELECT T3.full_name FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.games_name = ?", (games_name,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the percentage of female competitors taller than a specific height in a specific year
@app.get("/v1/olympics/percentage_female_competitors_by_height_and_year", operation_id="get_percentage_female_competitors", summary="Retrieves the percentage of female competitors who are taller than a specified height in a given year. The calculation is based on the total number of competitors in the specified year.")
async def get_percentage_female_competitors(gender: str = Query(..., description="Gender of the competitor"), height: int = Query(..., description="Height of the competitor"), games_year: int = Query(..., description="Year of the games")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.gender = ? AND T3.height > ? THEN 1 END) AS REAL) * 100 / COUNT(T2.person_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T1.games_year = ?", (gender, height, games_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of competitors older than a specific age in a specific season
@app.get("/v1/olympics/percentage_competitors_by_age_and_season", operation_id="get_percentage_competitors_by_age_and_season", summary="Retrieves the percentage of competitors who are older than a specified age during a particular season. This operation calculates the ratio of competitors above the given age to the total number of competitors in the specified season.")
async def get_percentage_competitors_by_age_and_season(age: int = Query(..., description="Age of the competitor"), season: str = Query(..., description="Season of the games")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.age > ? AND T1.season = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.games_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id", (age, season))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the region ID of a person by their full name
@app.get("/v1/olympics/region_id_by_person_name", operation_id="get_region_id_by_person_name", summary="Retrieves the region ID associated with a specific person, identified by their full name. This operation returns the region ID from the person_region table, which is linked to the person table via the person_id field. The person's full name is used to filter the results.")
async def get_region_id_by_person_name(full_name: str = Query(..., description="Full name of the person")):
    cursor.execute("SELECT T1.region_id FROM person_region AS T1 INNER JOIN person AS T2 ON T1.person_id = T2.id WHERE T2.full_name = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"region_id": []}
    return {"region_id": result[0]}

# Endpoint to get the height of persons in a specific region
@app.get("/v1/olympics/person_height_by_region", operation_id="get_person_height_by_region", summary="Retrieves the height of individuals associated with a specified region. The operation requires the unique identifier of the region as input and returns the height data of the corresponding individuals.")
async def get_person_height_by_region(region_id: int = Query(..., description="ID of the region")):
    cursor.execute("SELECT T2.height FROM person_region AS T1 INNER JOIN person AS T2 ON T1.person_id = T2.id WHERE T1.region_id = ?", (region_id,))
    result = cursor.fetchall()
    if not result:
        return {"heights": []}
    return {"heights": [row[0] for row in result]}

# Endpoint to get the city names for a specific games ID
@app.get("/v1/olympics/city_names_by_games_id", operation_id="get_city_names_by_games_id", summary="Retrieves the names of cities that hosted a specific Olympic Games, identified by its unique games_id. The response includes a list of city names associated with the provided games_id.")
async def get_city_names_by_games_id(games_id: int = Query(..., description="ID of the games")):
    cursor.execute("SELECT T2.city_name FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id WHERE T1.games_id = ?", (games_id,))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get the games IDs for a specific city name
@app.get("/v1/olympics/games_ids_by_city_name", operation_id="get_games_ids_by_city_name", summary="Retrieves the unique identifiers of all Olympic Games held in a specified city. The operation filters the games based on the provided city name and returns their corresponding IDs.")
async def get_games_ids_by_city_name(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.games_id FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id WHERE T2.city_name = ?", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"games_ids": []}
    return {"games_ids": [row[0] for row in result]}

# Endpoint to get the count of competitors under a certain age in a specific season
@app.get("/v1/olympics/competitor_count_by_season_age", operation_id="get_competitor_count_by_season_age", summary="Retrieves the total number of competitors who are younger than a specified age in a given season. The season and maximum age are provided as input parameters.")
async def get_competitor_count_by_season_age(season: str = Query(..., description="Season of the games"), max_age: int = Query(..., description="Maximum age of the competitors")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.season = ? AND T2.age < ?", (season, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct games names for a specific games ID
@app.get("/v1/olympics/distinct_games_names_by_games_id", operation_id="get_distinct_games_names_by_games_id", summary="Retrieve a list of unique game names associated with a specific game ID. This operation fetches the distinct game names from the games table, based on the provided game ID. The game ID is used to filter the results, ensuring that only relevant game names are returned.")
async def get_distinct_games_names_by_games_id(games_id: int = Query(..., description="ID of the games")):
    cursor.execute("SELECT DISTINCT T1.games_name FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T2.games_id = ?", (games_id,))
    result = cursor.fetchall()
    if not result:
        return {"games_names": []}
    return {"games_names": [row[0] for row in result]}

# Endpoint to get the average age of competitors in a specific season
@app.get("/v1/olympics/average_age_by_season", operation_id="get_average_age_by_season", summary="Retrieves the average age of competitors who participated in a specified season of the games. The season is provided as an input parameter.")
async def get_average_age_by_season(season: str = Query(..., description="Season of the games")):
    cursor.execute("SELECT AVG(T2.age) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.season = ?", (season,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the percentage of competitors under a certain age in a specific season
@app.get("/v1/olympics/percentage_competitors_by_season_age", operation_id="get_percentage_competitors_by_season_age", summary="Retrieves the percentage of competitors under a specified age during a particular season. This operation calculates the ratio of competitors below the given age to the total number of competitors in the specified season. The season and maximum age are provided as input parameters.")
async def get_percentage_competitors_by_season_age(season: str = Query(..., description="Season of the games"), max_age: int = Query(..., description="Maximum age of the competitors")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.age < ? THEN 1 END) AS REAL) * 100 / COUNT(T2.games_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.season = ?", (max_age, season))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct medal names for a specific competitor
@app.get("/v1/olympics/distinct_medal_names_by_competitor", operation_id="get_distinct_medal_names_by_competitor", summary="Retrieve a unique list of medal names awarded to a specific competitor. The operation requires the competitor's ID as input to filter the results.")
async def get_distinct_medal_names_by_competitor(competitor_id: int = Query(..., description="ID of the competitor")):
    cursor.execute("SELECT DISTINCT T1.medal_name FROM medal AS T1 INNER JOIN competitor_event AS T2 ON T1.id = T2.medal_id WHERE T2.competitor_id = ?", (competitor_id,))
    result = cursor.fetchall()
    if not result:
        return {"medal_names": []}
    return {"medal_names": [row[0] for row in result]}

# Endpoint to get the event IDs for a specific medal name
@app.get("/v1/olympics/event_ids_by_medal_name", operation_id="get_event_ids_by_medal_name", summary="Retrieves the event IDs associated with a specific medal. The operation filters events based on the provided medal name and returns their corresponding IDs.")
async def get_event_ids_by_medal_name(medal_name: str = Query(..., description="Name of the medal")):
    cursor.execute("SELECT T2.event_id FROM medal AS T1 INNER JOIN competitor_event AS T2 ON T1.id = T2.medal_id WHERE T1.medal_name = ?", (medal_name,))
    result = cursor.fetchall()
    if not result:
        return {"event_ids": []}
    return {"event_ids": [row[0] for row in result]}

# Endpoint to get the full name of the heaviest person
@app.get("/v1/olympics/heaviest_person", operation_id="get_heaviest_person", summary="Retrieves the full name of the heaviest individual, as determined by their recorded weight.")
async def get_heaviest_person():
    cursor.execute("SELECT full_name FROM person ORDER BY weight DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the city name of the games held in a specific year
@app.get("/v1/olympics/city_name_by_year", operation_id="get_city_name_by_year", summary="Get the city name of the games held in a specific year")
async def get_city_name_by_year(games_year: int = Query(..., description="Year of the games")):
    cursor.execute("SELECT T2.city_name FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T3.games_year = ?", (games_year,))
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the percentage of bronze medals in a specific event
@app.get("/v1/olympics/percentage_bronze_medals", operation_id="get_percentage_bronze_medals", summary="Retrieves the percentage of a specific type of medal (e.g., bronze) awarded in a given event. This operation calculates the ratio of the count of a particular medal to the total number of competitors in the specified event.")
async def get_percentage_bronze_medals(medal_name: str = Query(..., description="Name of the medal"), event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T4.medal_name = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.person_id) FROM competitor_event AS T1 INNER JOIN games_competitor AS T2 ON T1.competitor_id = T2.id INNER JOIN event AS T3 ON T1.event_id = T3.id INNER JOIN medal AS T4 ON T1.medal_id = T4.id WHERE T3.event_name LIKE ?", (medal_name, event_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the NOC of people with a specific weight
@app.get("/v1/olympics/noc_by_weight", operation_id="get_noc_by_weight", summary="Retrieves the National Olympic Committee (NOC) of athletes who weigh a specific amount. The operation filters athletes based on the provided weight and returns the corresponding NOC.")
async def get_noc_by_weight(weight: int = Query(..., description="Weight of the person")):
    cursor.execute("SELECT T1.noc FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.weight = ?", (weight,))
    result = cursor.fetchall()
    if not result:
        return {"noc": []}
    return {"noc": [row[0] for row in result]}

# Endpoint to get the city of the oldest competitor
@app.get("/v1/olympics/city_of_oldest_competitor", operation_id="get_city_of_oldest_competitor", summary="Retrieves the city name of the oldest competitor in the Olympics. This operation fetches the city information by joining the competitor, games, and city tables, and then orders the results by the competitor's age in descending order. The city of the oldest competitor is returned as the result.")
async def get_city_of_oldest_competitor():
    cursor.execute("SELECT T4.city_name FROM games_competitor AS T1 INNER JOIN games AS T2 ON T1.games_id = T2.id INNER JOIN games_city AS T3 ON T1.games_id = T3.games_id INNER JOIN city AS T4 ON T3.city_id = T4.id ORDER BY T1.age DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the NOC and region name of the heaviest person
@app.get("/v1/olympics/heaviest_person_noc_region", operation_id="get_heaviest_person_noc_region", summary="Retrieves the National Olympic Committee (NOC) and region name of the heaviest individual, as determined by their weight. This operation involves joining multiple tables to gather the necessary data, and the result is sorted by weight in descending order to identify the heaviest person.")
async def get_heaviest_person_noc_region():
    cursor.execute("SELECT T1.noc, T1.region_name FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id ORDER BY T3.weight DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"noc": [], "region_name": []}
    return {"noc": result[0], "region_name": result[1]}

# Endpoint to get the games year and season for a specific person
@app.get("/v1/olympics/games_year_season_by_full_name", operation_id="get_games_year_season_by_full_name", summary="Get the games year and season for a specific person by their full name")
async def get_games_year_season_by_full_name(full_name: str = Query(..., description="Full name of the person")):
    cursor.execute("SELECT T1.games_year, T1.season FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.full_name = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"games_year": [], "season": []}
    return {"games_year": result[0], "season": result[1]}

# Endpoint to get the average weight of medal winners
@app.get("/v1/olympics/average_weight_by_medal", operation_id="get_average_weight_by_medal", summary="Retrieves the average weight of athletes who have won a specific medal. The operation calculates the average weight based on the provided medal name, considering the weight of all athletes who have won that medal in the Olympics.")
async def get_average_weight_by_medal(medal_name: str = Query(..., description="Name of the medal")):
    cursor.execute("SELECT AVG(T1.weight) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id INNER JOIN competitor_event AS T3 ON T2.id = T3.competitor_id INNER JOIN medal AS T4 ON T3.medal_id = T4.id WHERE T4.medal_name = ?", (medal_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_weight": []}
    return {"average_weight": result[0]}

# Endpoint to get distinct seasons for competitors with specific height and weight
@app.get("/v1/olympics/distinct_seasons_by_height_weight", operation_id="get_distinct_seasons_by_height_weight", summary="Retrieves the unique seasons in which competitors of a specific height and weight have participated. This operation filters the games data based on the provided height and weight, and returns the distinct seasons in which these competitors have been involved.")
async def get_distinct_seasons_by_height_weight(height: int = Query(..., description="Height of the person"), weight: int = Query(..., description="Weight of the person")):
    cursor.execute("SELECT DISTINCT T1.season FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.height = ? AND T3.weight = ?", (height, weight))
    result = cursor.fetchall()
    if not result:
        return {"seasons": []}
    return {"seasons": [row[0] for row in result]}

# Endpoint to get distinct full names of medal winners
@app.get("/v1/olympics/distinct_full_names_by_medal", operation_id="get_distinct_full_names_by_medal", summary="Retrieves a unique list of full names of individuals who have won a specific type of medal in the Olympics. The type of medal is specified as an input parameter.")
async def get_distinct_full_names_by_medal(medal_name: str = Query(..., description="Name of the medal")):
    cursor.execute("SELECT DISTINCT T1.full_name FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id INNER JOIN competitor_event AS T3 ON T2.id = T3.competitor_id INNER JOIN medal AS T4 ON T3.medal_id = T4.id WHERE T4.medal_name = ?", (medal_name,))
    result = cursor.fetchall()
    if not result:
        return {"full_names": []}
    return {"full_names": [row[0] for row in result]}

# Endpoint to get the average height of competitors within a specific age range
@app.get("/v1/olympics/average_height_by_age_range", operation_id="get_average_height_by_age_range", summary="Retrieves the average height of competitors who fall within a specified age range. The age range is defined by the provided minimum and maximum age values. This operation calculates the average height from the person table, considering only those individuals who have participated in games and whose ages are within the specified range.")
async def get_average_height_by_age_range(min_age: int = Query(..., description="Minimum age of the competitor"), max_age: int = Query(..., description="Maximum age of the competitor")):
    cursor.execute("SELECT AVG(T1.height) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id WHERE T2.age BETWEEN ? AND ?", (min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the age of the tallest person
@app.get("/v1/olympics/age_of_tallest_person", operation_id="get_age_of_tallest_person", summary="Retrieves the age of the tallest person in the database. The operation identifies the tallest person by comparing the heights of all individuals and then returns the age of the tallest person.")
async def get_age_of_tallest_person():
    cursor.execute("SELECT T2.age FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id ORDER BY T1.height DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the difference in count of competitors above and below a specific weight for a given age
@app.get("/v1/olympics/weight_difference_count_by_age", operation_id="get_weight_difference_count_by_age", summary="Retrieve the count difference of competitors with weights above and below a specified threshold for individuals under a certain age. This operation provides insights into the distribution of competitor weights within a specific age range.")
async def get_weight_difference_count_by_age(weight: int = Query(..., description="Weight threshold"), max_age: int = Query(..., description="Maximum age of the competitor")):
    cursor.execute("SELECT COUNT(CASE WHEN T1.weight > ? THEN 1 ELSE NULL END) - COUNT(CASE WHEN T1.weight < ? THEN 1 ELSE NULL END) FROM person AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.person_id WHERE T2.age < ?", (weight, weight, max_age))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the percentage of competitors of a specific age in a given game
@app.get("/v1/olympics/competitor_age_percentage", operation_id="get_competitor_age_percentage", summary="Retrieves the percentage of competitors of a specific age in a given game. This operation calculates the ratio of competitors of the specified age to the total number of competitors in the selected game. The age and game name are provided as input parameters.")
async def get_competitor_age_percentage(age: int = Query(..., description="Age of the competitor"), games_name: str = Query(..., description="Name of the games")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.age = ? THEN 1 END) AS REAL) * 100 / COUNT(T2.person_id) FROM games AS T1 INNER JOIN games_competitor AS T2 ON T1.id = T2.games_id WHERE T1.games_name = ?", (age, games_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct region names of male competitors taller than a specific height percentage
@app.get("/v1/olympics/distinct_region_names_tall_males", operation_id="get_distinct_region_names_tall_males", summary="Retrieves a list of unique region names where male competitors taller than a specified height percentage threshold have participated. The height percentage is calculated based on the average height of all competitors of the same gender.")
async def get_distinct_region_names_tall_males(gender: str = Query(..., description="Gender of the competitor"), height_percentage: int = Query(..., description="Height percentage threshold")):
    cursor.execute("SELECT DISTINCT T1.region_name FROM noc_region AS T1 INNER JOIN person_region AS T2 ON T1.id = T2.region_id INNER JOIN person AS T3 ON T2.person_id = T3.id WHERE T3.gender = ? AND T3.height * 100 > ( SELECT AVG(height) FROM person WHERE gender = ? ) * ?", (gender, gender, height_percentage))
    result = cursor.fetchall()
    if not result:
        return {"regions": []}
    return {"regions": [row[0] for row in result]}

api_calls = [
    "/v1/olympics/games_names_by_competitor?full_name=John%20Aalberg",
    "/v1/olympics/games_names_by_competitor_age?full_name=John%20Aalberg&age=31",
    "/v1/olympics/competitor_age_in_game?full_name=John%20Aalberg&games_name=1994%20Winter",
    "/v1/olympics/count_persons_by_region?region_name=Finland",
    "/v1/olympics/full_names_by_region?region_name=Finland",
    "/v1/olympics/region_names_by_person?full_name=John%20Aalberg",
    "/v1/olympics/tallest_person_noc_by_gender?gender=M",
    "/v1/olympics/count_persons_by_region_gender?region_name=Finland&gender=F",
    "/v1/olympics/city_names_by_game?games_name=1992%20Summer",
    "/v1/olympics/games_names_by_city?city_name=London",
    "/v1/olympics/first_game_year_by_city?city_name=London",
    "/v1/olympics/count_games_by_city?city_name=London",
    "/v1/olympics/avg_height_by_region_gender?region_name=Finland&gender=M",
    "/v1/olympics/percentage_athletes_by_region_game?region_name=Finland&games_name=1994%20Winter",
    "/v1/olympics/sport_name_by_event?event_name=Shooting%20Women%25s%20Trap",
    "/v1/olympics/count_events_by_sport?sport_name=Swimming",
    "/v1/olympics/game_ids_by_athlete?full_name=Jessica%20Carolina%20Aguilera%20Aguilera",
    "/v1/olympics/sport_with_most_events?limit=1",
    "/v1/olympics/tallest_person_by_region?region_name=Sweden&limit=1",
    "/v1/olympics/person_with_most_competitions?limit=1",
    "/v1/olympics/competitor_full_name_by_id?competitor_id=90991",
    "/v1/olympics/count_competitions_by_full_name?full_name=Martina%20Kohlov",
    "/v1/olympics/competitor_bmi_by_id?competitor_id=147420",
    "/v1/olympics/percentage_male_competitors_by_region?region_name=Estonia",
    "/v1/olympics/youngest_competitor?limit=1",
    "/v1/olympics/count_competitors_by_event_and_age?event_name=Basketball%20Men%25s%20Basketball&age=24",
    "/v1/olympics/event_names_by_sport?sport_name=Art%20Competitions",
    "/v1/olympics/medal_count_by_person_and_medal?full_name=Henk%20Jan%20Zwolle&medal_name=Gold",
    "/v1/olympics/event_with_most_gold_medals?medal_name=Gold",
    "/v1/olympics/person_count_by_region?region_name=Australia",
    "/v1/olympics/cities_hosting_minimum_games?min_games=3",
    "/v1/olympics/game_count_by_city_and_season?city_name=Stockholm&season=Summer",
    "/v1/olympics/region_with_most_persons",
    "/v1/olympics/city_of_earliest_game",
    "/v1/olympics/game_with_youngest_competitor",
    "/v1/olympics/competitor_count_by_game?games_name=1928%20Summer",
    "/v1/olympics/athlete_event_count_medal_percentage?medal_id=1&full_name=Michael%20Fred%20Phelps%2C%20II",
    "/v1/olympics/season_max_competitor_difference?game_name_1=1988%20Winter&game_name_2=1988%20Summer",
    "/v1/olympics/most_common_competitor_age",
    "/v1/olympics/city_names_starting_with?letter=M",
    "/v1/olympics/city_names_for_games?game_name_1=1976%20Summer&game_name_2=1976%20Winter",
    "/v1/olympics/athlete_medal_names?full_name=Coleen%20Dufresne%20(-Stewner)",
    "/v1/olympics/athlete_most_medals_excluding?medal_id=4",
    "/v1/olympics/athlete_sport_names?full_name=Chin%20Eei%20Hui",
    "/v1/olympics/oldest_competitor",
    "/v1/olympics/game_with_most_female_competitors?gender=F",
    "/v1/olympics/count_male_competitors_by_region?region_name=Belgium&gender=M",
    "/v1/olympics/count_competitors_by_city?city_name=Sapporo",
    "/v1/olympics/count_competitors_by_event?event_name=Sailing%20Mixed%2012%20metres",
    "/v1/olympics/count_distinct_events_by_sport?sport_name=Modern%20Pentathlon",
    "/v1/olympics/percentage_female_competitors_by_event?gender=F&event_name=Equestrianism%20Mixed%20Three-Day%20Event%2C%20Individual",
    "/v1/olympics/average_age_by_game?games_name=1992%20Summer",
    "/v1/olympics/years_with_single_season?season=Winter",
    "/v1/olympics/count_games_in_year_range?start_year=1990&end_year=1999",
    "/v1/olympics/competitor_count_by_region_and_medal?region_id=151&medal_id=4",
    "/v1/olympics/competitor_count_by_game_and_region?games_name=2016%20Summer&region_name=China",
    "/v1/olympics/competitor_count_by_event_and_medal?event_name=Ice%20Hockey%20Men%25s%20Ice%20Hockey&medal_id=1",
    "/v1/olympics/top_region_by_medal?medal_id=4",
    "/v1/olympics/competitor_names_by_sport_and_medal?sport_name=Cycling&medal_id=1",
    "/v1/olympics/medals_by_competitor?full_name=Lee%20Chong%20Wei&medal_id=4",
    "/v1/olympics/female_percentage_by_game_and_age?games_name=2002%20Winter&age=20",
    "/v1/olympics/male_to_female_ratio_by_game?games_name=2012%20Summer",
    "/v1/olympics/highest_and_lowest_competitor_games_by_season?season=Summer",
    "/v1/olympics/medalist_count_region_exclude_medal?region_name=Malaysia&exclude_medal_id=4",
    "/v1/olympics/competitor_count_sport_games_bmi?sport_name=Canoeing&games_name=2008%20Summer&bmi_min=25.0&bmi_max=30",
    "/v1/olympics/avg_height_sport_gender?sport_name=Basketball&gender=M",
    "/v1/olympics/youngest_age_by_person?full_name=Michael%20Fred%20Phelps%2C%20II",
    "/v1/olympics/person_count_same_region?full_name=Clara%20Hughes",
    "/v1/olympics/event_count_person_event_name?full_name=Ian%20James%20Thorpe&event_name_pattern=Swimming%20Men%25s%20200%20metres%20Freestyle",
    "/v1/olympics/medal_count_person_event_name?full_name=Larysa%20Semenivna%20Latynina%20(Diriy-)&event_name_pattern=Gymnastics%20Women%25s%20Individual%20All-Around&medal_id=1",
    "/v1/olympics/city_names_by_person?full_name=Carl%20Lewis%20Borack",
    "/v1/olympics/game_with_most_competitors_in_city?city_name=Los%20Angeles",
    "/v1/olympics/average_age_by_game_and_region?games_name=2016%20Summer&region_name=USA",
    "/v1/olympics/region_name_by_noc?noc=COL",
    "/v1/olympics/sport_name_by_id?sport_id=19",
    "/v1/olympics/event_id_by_name?event_name=Shooting%20Mixed%20Skeet",
    "/v1/olympics/sport_id_by_name?sport_name=Hockey",
    "/v1/olympics/person_weight_by_name?full_name=Dagfinn%20Sverre%20Aarskog",
    "/v1/olympics/city_id_by_name?city_name=Rio%20de%20Janeiro",
    "/v1/olympics/competitor_count_event_medal?event_name=Rowing%20Women%25s%20Coxed%20Eights&medal_name=Gold",
    "/v1/olympics/games_year_city?city_name=Roma",
    "/v1/olympics/competitor_count_games_age?games_name=1984%20Summer&age=20",
    "/v1/olympics/games_count_person?full_name=Prithipal%20Singh",
    "/v1/olympics/competitor_count_games_age_greater?games_name=1984%20Summer&age=50",
    "/v1/olympics/heaviest_person_region?region_name=Russia",
    "/v1/olympics/tallest_person_region?region_name=Portugal",
    "/v1/olympics/percentage_tall_athletes_by_region?height=175&region_name=Vanuatu",
    "/v1/olympics/average_weight_by_region_and_gender?region_name=Tonga&gender=M",
    "/v1/olympics/count_games_by_city_and_year_range?city_name=London&start_year=1900&end_year=1992",
    "/v1/olympics/city_with_most_games",
    "/v1/olympics/games_name_by_city_and_year?city_name=Beijing&games_year=2008",
    "/v1/olympics/percentage_medalists_older_than?age=30&medal_id=1",
    "/v1/olympics/competitor_age_by_game_and_name?games_name=2012%20Summer&full_name=A%20Lamusi",
    "/v1/olympics/competitor_count_by_game_and_gender?games_name=1948%20Summer&gender=M",
    "/v1/olympics/competitor_names_by_game?games_name=1936%20Summer",
    "/v1/olympics/percentage_female_competitors_by_height_and_year?gender=F&height=170&games_year=1988",
    "/v1/olympics/percentage_competitors_by_age_and_season?age=24&season=Winter",
    "/v1/olympics/region_id_by_person_name?full_name=Christine%20Jacoba%20Aaftink",
    "/v1/olympics/person_height_by_region?region_id=7",
    "/v1/olympics/city_names_by_games_id?games_id=3",
    "/v1/olympics/games_ids_by_city_name?city_name=London",
    "/v1/olympics/competitor_count_by_season_age?season=Summer&max_age=30",
    "/v1/olympics/distinct_games_names_by_games_id?games_id=13",
    "/v1/olympics/average_age_by_season?season=Winter",
    "/v1/olympics/percentage_competitors_by_season_age?season=Summer&max_age=35",
    "/v1/olympics/distinct_medal_names_by_competitor?competitor_id=9",
    "/v1/olympics/event_ids_by_medal_name?medal_name=Gold",
    "/v1/olympics/heaviest_person",
    "/v1/olympics/city_name_by_year?games_year=1992",
    "/v1/olympics/percentage_bronze_medals?medal_name=Bronze&event_name=Basketball%20Men%25s%20Basketball",
    "/v1/olympics/noc_by_weight?weight=77",
    "/v1/olympics/city_of_oldest_competitor",
    "/v1/olympics/heaviest_person_noc_region",
    "/v1/olympics/games_year_season_by_full_name?full_name=Sohail%20Abbas",
    "/v1/olympics/average_weight_by_medal?medal_name=Silver",
    "/v1/olympics/distinct_seasons_by_height_weight?height=180&weight=73",
    "/v1/olympics/distinct_full_names_by_medal?medal_name=Gold",
    "/v1/olympics/average_height_by_age_range?min_age=22&max_age=28",
    "/v1/olympics/age_of_tallest_person",
    "/v1/olympics/weight_difference_count_by_age?weight=70&max_age=24",
    "/v1/olympics/competitor_age_percentage?age=28&games_name=2014%20Winter",
    "/v1/olympics/distinct_region_names_tall_males?gender=M&height_percentage=87"
]
