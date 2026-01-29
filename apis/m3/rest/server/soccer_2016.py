from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/soccer_2016/soccer_2016.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of players born after a specific year
@app.get("/v1/soccer_2016/player_count_by_year", operation_id="get_player_count_by_year", summary="Retrieves the total number of soccer players born after the specified year. The year parameter is used to filter the players by their birth year.")
async def get_player_count_by_year(year: int = Query(..., description="Year of birth (YYYY)")):
    cursor.execute("SELECT COUNT(Player_Id) FROM Player WHERE SUBSTR(DOB, 1, 4) > ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches in a specific year and month
@app.get("/v1/soccer_2016/match_count_by_year_month", operation_id="get_match_count_by_year_month", summary="Retrieves the total number of matches that took place in a given year and month. The year and month are provided as input parameters, allowing for a targeted count of matches within the specified time frame.")
async def get_match_count_by_year_month(year: str = Query(..., description="Year of the match (YYYY)"), month: str = Query(..., description="Month of the match (M)")):
    cursor.execute("SELECT COUNT(Match_Id) FROM `Match` WHERE SUBSTR(Match_Date, 1, 4) = ? AND SUBSTR(Match_Date, 7, 1) = ?", (year, month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches where a specific player was the man of the match
@app.get("/v1/soccer_2016/match_count_by_man_of_the_match", operation_id="get_match_count_by_man_of_the_match", summary="Retrieves the total number of matches in which a particular player was awarded the man of the match title. The player is identified by their unique ID.")
async def get_match_count_by_man_of_the_match(man_of_the_match: int = Query(..., description="Player ID of the man of the match")):
    cursor.execute("SELECT COUNT(Match_Id) FROM `Match` WHERE Man_of_the_Match = ?", (man_of_the_match,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get match IDs in a specific year
@app.get("/v1/soccer_2016/match_ids_by_year", operation_id="get_match_ids_by_year", summary="Retrieves a list of match IDs that took place in a specified year. The year should be provided in the YYYY format.")
async def get_match_ids_by_year(year: str = Query(..., description="Year of the match (YYYY)")):
    cursor.execute("SELECT Match_Id FROM `Match` WHERE SUBSTR(Match_Date, 1, 4) = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"match_ids": []}
    return {"match_ids": [row[0] for row in result]}

# Endpoint to get the count of players from a specific country
@app.get("/v1/soccer_2016/player_count_by_country", operation_id="get_player_count_by_country", summary="Retrieves the total number of soccer players from a specified country in the 2016 dataset. The operation filters players based on the provided country name and returns the count.")
async def get_player_count_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT COUNT(CASE WHEN T2.Country_Name = ? THEN T1.Player_Id ELSE NULL END) FROM Player AS T1 INNER JOIN Country AS T2 ON T1.Country_Name = T2.Country_Id", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country name of the oldest player
@app.get("/v1/soccer_2016/oldest_player_country", operation_id="get_oldest_player_country", summary="Retrieves the name of the country with the oldest registered soccer player from the 2016 dataset. The operation considers only players with a valid country association and returns the country name of the player with the earliest date of birth.")
async def get_oldest_player_country():
    cursor.execute("SELECT T1.Country_Name FROM Country AS T1 INNER JOIN Player AS T2 ON T2.Country_Name = T1.Country_Id WHERE T2.Country_Name IS NOT NULL ORDER BY T2.DOB LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the bowling skill of a specific player
@app.get("/v1/soccer_2016/bowling_skill_by_player", operation_id="get_bowling_skill_by_player", summary="Retrieves the bowling skill level of a specific soccer player from the 2016 season. The operation requires the player's name as input and returns the corresponding bowling skill level.")
async def get_bowling_skill_by_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T1.Bowling_Skill FROM Bowling_Style AS T1 INNER JOIN Player AS T2 ON T2.Bowling_skill = T1.Bowling_Id WHERE T2.Player_Name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"bowling_skill": []}
    return {"bowling_skill": result[0]}

# Endpoint to get the count of players with a specific batting hand born after a specific year
@app.get("/v1/soccer_2016/player_count_by_batting_hand_and_year", operation_id="get_player_count_by_batting_hand_and_year", summary="Retrieves the total number of players who were born after a specified year and use a particular batting hand. The response is based on the aggregated data from the Player and Batting_Style tables.")
async def get_player_count_by_batting_hand_and_year(year: int = Query(..., description="Year of birth (YYYY)"), batting_hand: str = Query(..., description="Batting hand of the player")):
    cursor.execute("SELECT SUM(CASE WHEN SUBSTR(T1.DOB, 1, 4) > ? THEN 1 ELSE 0 END) FROM Player AS T1 INNER JOIN Batting_Style AS T2 ON T1.Batting_hand = T2.Batting_Id WHERE T2.Batting_Hand = ?", (year, batting_hand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get player names from a specific country with a specific batting hand
@app.get("/v1/soccer_2016/player_names_by_country_and_batting_hand", operation_id="get_player_names_by_country_and_batting_hand", summary="Retrieves the names of soccer players from a specified country who use a particular batting hand. The operation filters players based on the provided country name and batting hand, then returns a list of matching player names.")
async def get_player_names_by_country_and_batting_hand(country_name: str = Query(..., description="Name of the country"), batting_hand: str = Query(..., description="Batting hand of the player")):
    cursor.execute("SELECT T2.Player_Name FROM Country AS T1 INNER JOIN Player AS T2 ON T2.Country_Name = T1.Country_id INNER JOIN Batting_Style AS T3 ON T2.Batting_hand = T3.Batting_Id WHERE T1.Country_Name = ? AND T3.Batting_Hand = ?", (country_name, batting_hand))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get bowling skills of players from a specific country
@app.get("/v1/soccer_2016/bowling_skills_by_country", operation_id="get_bowling_skills_by_country", summary="Retrieves the distribution of bowling skills among soccer players from a specified country. The operation returns a list of unique bowling skills and their respective counts, providing insights into the prevalence of different bowling styles within the selected country's soccer player population.")
async def get_bowling_skills_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Bowling_Skill FROM Player AS T1 INNER JOIN Bowling_Style AS T2 ON T1.Bowling_skill = T2.Bowling_Id INNER JOIN Country AS T3 ON T1.Country_Name = T3.Country_Id WHERE T3.Country_Name = ? GROUP BY T2.Bowling_Skill", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"bowling_skills": []}
    return {"bowling_skills": [row[0] for row in result]}

# Endpoint to get the minimum date of birth of players with a specific bowling skill
@app.get("/v1/soccer_2016/min_dob_by_bowling_skill", operation_id="get_min_dob_by_bowling_skill", summary="Retrieves the earliest date of birth among players who possess a specified bowling skill. The input parameter determines the bowling skill to filter the players.")
async def get_min_dob_by_bowling_skill(bowling_skill: str = Query(..., description="Bowling skill of the player")):
    cursor.execute("SELECT MIN(T1.DOB) FROM Player AS T1 INNER JOIN Bowling_Style AS T2 ON T1.Bowling_skill = T2.Bowling_Id WHERE T2.Bowling_Skill = ?", (bowling_skill,))
    result = cursor.fetchone()
    if not result:
        return {"min_dob": []}
    return {"min_dob": result[0]}

# Endpoint to get the most common bowling skill among players
@app.get("/v1/soccer_2016/most_common_bowling_skill", operation_id="get_most_common_bowling_skill", summary="Retrieves the most frequently occurring bowling skill among soccer players in the 2016 season. The operation returns the skill with the highest count, based on the number of players who possess it.")
async def get_most_common_bowling_skill():
    cursor.execute("SELECT T1.Bowling_Skill FROM Bowling_Style AS T1 INNER JOIN Player AS T2 ON T2.Bowling_skill = T1.Bowling_Id GROUP BY T1.Bowling_Skill ORDER BY COUNT(T1.Bowling_Skill) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"bowling_skill": []}
    return {"bowling_skill": result[0]}

# Endpoint to get the man of the match for a specific match date
@app.get("/v1/soccer_2016/man_of_the_match_by_date", operation_id="get_man_of_the_match_by_date", summary="Retrieves the name of the player who was awarded the man of the match for a specific match date. The date must be provided in the 'YYYY-MM-DD' format.")
async def get_man_of_the_match_by_date(match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.Player_Name FROM Match AS T1 INNER JOIN Player AS T2 ON T2.Player_Id = T1.Man_of_the_Match WHERE T1.Match_Date = ?", (match_date,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the count of matches where a player had a specific role
@app.get("/v1/soccer_2016/count_matches_by_role_and_player", operation_id="get_count_matches_by_role_and_player", summary="Retrieves the total number of matches in which a specific player had a particular role. The operation requires the role description and the player's name as input parameters.")
async def get_count_matches_by_role_and_player(role_desc: str = Query(..., description="Role description"), player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT SUM(CASE WHEN T3.Role_Desc = ? THEN 1 ELSE 0 END) FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Rolee AS T3 ON T2.Role_Id = T3.Role_Id WHERE T1.Player_Name = ?", (role_desc, player_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the role ID of a player in a specific match
@app.get("/v1/soccer_2016/role_id_by_player_and_match_date", operation_id="get_role_id_by_player_and_match_date", summary="Retrieves the role ID of a specific player in a given match. The operation requires the player's name and the match date as input parameters. The role ID corresponds to the player's role in the match, which is determined by joining relevant tables in the database.")
async def get_role_id_by_player_and_match_date(player_name: str = Query(..., description="Player name"), match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.Role_Id FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Rolee AS T3 ON T2.Role_Id = T3.Role_Id INNER JOIN Match AS T4 ON T2.Match_Id = T4.Match_Id WHERE T1.Player_Name = ? AND T4.Match_Date = ?", (player_name, match_date))
    result = cursor.fetchone()
    if not result:
        return {"role_id": []}
    return {"role_id": result[0]}

# Endpoint to get the maximum win margin of matches played by a specific player
@app.get("/v1/soccer_2016/max_win_margin_by_player", operation_id="get_max_win_margin_by_player", summary="Retrieves the highest win margin achieved by a specific player in their matches. The player is identified by their name.")
async def get_max_win_margin_by_player(player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT MAX(T3.Win_Margin) FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Match AS T3 ON T2.Match_Id = T3.Match_Id WHERE T1.Player_Name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"max_win_margin": []}
    return {"max_win_margin": result[0]}

# Endpoint to get the average win margin of matches played by a specific player
@app.get("/v1/soccer_2016/avg_win_margin_by_player", operation_id="get_avg_win_margin_by_player", summary="Retrieves the average win margin of matches played by a specified player. The player's name is used to filter the matches and calculate the average win margin.")
async def get_avg_win_margin_by_player(player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT CAST(SUM(T3.Win_Margin) AS REAL) / COUNT(*) FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Match AS T3 ON T2.Match_Id = T3.Match_Id WHERE T1.Player_Name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"avg_win_margin": []}
    return {"avg_win_margin": result[0]}

# Endpoint to get the percentage of players with a specific batting hand born after a certain year
@app.get("/v1/soccer_2016/percentage_batting_hand_by_year", operation_id="get_percentage_batting_hand_by_year", summary="Retrieves the percentage of players with a specified batting hand born after a given year. This operation calculates the ratio of players with the provided batting hand to the total number of players born after the specified year.")
async def get_percentage_batting_hand_by_year(batting_hand: str = Query(..., description="Batting hand of the player"), year: int = Query(..., description="Year of birth")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Batting_Hand = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.Player_Id) FROM Player AS T1 INNER JOIN Batting_Style AS T2 ON T1.Batting_hand = T2.Batting_Id WHERE SUBSTR(T1.DOB, 1, 4) > ?", (batting_hand, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the youngest player
@app.get("/v1/soccer_2016/youngest_player", operation_id="get_youngest_player", summary="Retrieves the name of the youngest player in the 2016 soccer roster. The operation sorts the players by their date of birth in descending order and returns the name of the first player in the sorted list.")
async def get_youngest_player():
    cursor.execute("SELECT Player_Name FROM Player ORDER BY DOB DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the number of matches won by a specific team in the toss
@app.get("/v1/soccer_2016/toss_wins_by_team", operation_id="get_toss_wins_by_team", summary="Retrieves the total number of matches won by a specified team in the toss. The team is identified by its name, which is provided as an input parameter.")
async def get_toss_wins_by_team(team_name: str = Query(..., description="Team name")):
    cursor.execute("SELECT SUM(CASE WHEN Toss_Winner = ( SELECT Team_Id FROM Team WHERE Team_Name = ? ) THEN 1 ELSE 0 END) FROM `Match`", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"toss_wins": []}
    return {"toss_wins": result[0]}

# Endpoint to get the player name based on match, over, ball, and innings details
@app.get("/v1/soccer_2016/player_name_by_match_over_ball_innings", operation_id="get_player_name_by_match_over_ball_innings", summary="Retrieves the name of the player who played a specific ball in a given match, over, and innings. The operation requires the match ID, over ID, ball ID, and innings number to identify the exact ball and the corresponding player.")
async def get_player_name_by_match_over_ball_innings(match_id: int = Query(..., description="Match ID"), over_id: int = Query(..., description="Over ID"), ball_id: int = Query(..., description="Ball ID"), innings_no: int = Query(..., description="Innings number")):
    cursor.execute("SELECT T2.Player_Name FROM Ball_by_Ball AS T1 INNER JOIN Player AS T2 ON T1.Striker = T2.Player_Id WHERE T1.Match_Id = ? AND T1.Over_Id = ? AND T1.Ball_Id = ? AND T1.Innings_No = ?", (match_id, over_id, ball_id, innings_no))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the venue name based on win margin
@app.get("/v1/soccer_2016/venue_name_by_win_margin", operation_id="get_venue_name_by_win_margin", summary="Retrieves the name of the venue where a match was played with a specific win margin. The win margin is provided as an input parameter.")
async def get_venue_name_by_win_margin(win_margin: int = Query(..., description="Win margin")):
    cursor.execute("SELECT T2.Venue_Name FROM `Match` AS T1 INNER JOIN Venue AS T2 ON T1.Venue_Id = T2.Venue_Id WHERE T1.Win_Margin = ?", (win_margin,))
    result = cursor.fetchall()
    if not result:
        return {"venue_names": []}
    return {"venue_names": [row[0] for row in result]}

# Endpoint to get the player name based on match date
@app.get("/v1/soccer_2016/player_name_by_match_date", operation_id="get_player_name_by_match_date", summary="Retrieves the name of the player who was the man of the match on the specified date. The date must be provided in the 'YYYY-MM-DD' format.")
async def get_player_name_by_match_date(match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Match AS T2 ON T1.Player_Id = T2.Man_of_the_Match WHERE T2.Match_Date = ?", (match_date,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the player name based on match ID and role description
@app.get("/v1/soccer_2016/player_name_by_match_role", operation_id="get_player_name_by_match_role", summary="Retrieves the name of the player who played a specific role in a given match. The match is identified by its unique ID, and the role is specified by its description. This operation returns the player's name, providing a direct link between the match, the role, and the player.")
async def get_player_name_by_match_role(match_id: str = Query(..., description="Match ID"), role_desc: str = Query(..., description="Role description")):
    cursor.execute("SELECT T3.Player_Name FROM Player_Match AS T1 INNER JOIN Rolee AS T2 ON T1.Role_Id = T2.Role_Id INNER JOIN Player AS T3 ON T1.Player_Id = T3.Player_Id WHERE T1.Match_Id = ? AND T2.Role_Desc = ?", (match_id, role_desc))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the player name based on season year
@app.get("/v1/soccer_2016/player_name_by_season_year", operation_id="get_player_name_by_season_year", summary="Retrieves the name of the player who was the man of the series for a given season year in the 2016 soccer season. The operation requires the season year as an input parameter.")
async def get_player_name_by_season_year(season_year: int = Query(..., description="Season year")):
    cursor.execute("SELECT T2.Player_Name FROM Season AS T1 INNER JOIN Player AS T2 ON T1.Man_of_the_Series = T2.Player_Id WHERE T1.Season_Year = ?", (season_year,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the date of birth of the player based on season year and orange cap
@app.get("/v1/soccer_2016/player_dob_by_season_year_orange_cap", operation_id="get_player_dob_by_season_year_orange_cap", summary="Retrieves the date of birth of the player who was the man of the series in the specified season year. The season year must have an orange cap winner. The response includes the date of birth of the player.")
async def get_player_dob_by_season_year_orange_cap(season_year: int = Query(..., description="Season year")):
    cursor.execute("SELECT T2.DOB FROM Season AS T1 INNER JOIN Player AS T2 ON T1.Man_of_the_Series = T2.Player_Id WHERE T1.Season_Year = ? AND T1.Orange_Cap IS NOT NULL", (season_year,))
    result = cursor.fetchall()
    if not result:
        return {"dobs": []}
    return {"dobs": [row[0] for row in result]}

# Endpoint to get the country name based on season ID and purple cap
@app.get("/v1/soccer_2016/country_name_by_season_id_purple_cap", operation_id="get_country_name_by_season_id_purple_cap", summary="Retrieves the name of the country associated with a specific season ID, provided that the season has a Purple Cap winner. The operation uses the season ID to identify the relevant season and then retrieves the country name of the player who was the Man of the Series for that season.")
async def get_country_name_by_season_id_purple_cap(season_id: int = Query(..., description="Season ID")):
    cursor.execute("SELECT T3.Country_Name FROM Season AS T1 INNER JOIN Player AS T2 ON T1.Man_of_the_Series = T2.Player_Id INNER JOIN Country AS T3 ON T2.Country_Name = T3.Country_Id WHERE T1.Season_Id = ? AND T1.Purple_Cap IS NOT NULL", (season_id,))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get the country name based on city name
@app.get("/v1/soccer_2016/country_name_by_city_name", operation_id="get_country_name_by_city_name", summary="Retrieves the name of the country associated with the specified city. The operation uses the provided city name to search for a match in the City table, then retrieves the corresponding country name from the Country table using the matched Country_Id.")
async def get_country_name_by_city_name(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT T2.Country_Name FROM City AS T1 INNER JOIN Country AS T2 ON T1.Country_Id = T2.Country_Id WHERE T1.City_Name = ?", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get the count of cities in a specific country
@app.get("/v1/soccer_2016/count_cities_by_country", operation_id="get_count_cities_by_country", summary="Retrieves the total number of cities in a specified country. The operation calculates the sum of cities based on the provided country name.")
async def get_count_cities_by_country(country_name: str = Query(..., description="Country name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Country_Name = ? THEN 1 ELSE 0 END) FROM City AS T1 INNER JOIN Country AS T2 ON T1.Country_Id = T2.Country_Id", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city with the most venues
@app.get("/v1/soccer_2016/city_with_most_venues", operation_id="get_city_with_most_venues", summary="Retrieves the name of the city that hosts the highest number of venues. The operation identifies the city with the most venues by joining the City and Venue tables, grouping by city, and ordering the results in descending order based on the count of venues. The top result is returned.")
async def get_city_with_most_venues():
    cursor.execute("SELECT T1.City_Name FROM City AS T1 INNER JOIN Venue AS T2 ON T1.City_Id = T2.City_Id GROUP BY T1.City_Id ORDER BY COUNT(T2.Venue_Id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the batting hand of a player by name
@app.get("/v1/soccer_2016/player_batting_hand", operation_id="get_player_batting_hand", summary="Retrieves the batting hand of a specific soccer player from the 2016 season. The player is identified by their name, which is provided as an input parameter. The operation returns the batting hand of the player, which is determined based on the batting style associated with the player in the database.")
async def get_player_batting_hand(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T2.Batting_hand FROM Player AS T1 INNER JOIN Batting_Style AS T2 ON T1.Batting_hand = T2.Batting_Id WHERE T1.Player_Name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"batting_hand": []}
    return {"batting_hand": result[0]}

# Endpoint to get the ratio of cities in two countries
@app.get("/v1/soccer_2016/city_ratio_by_countries", operation_id="get_city_ratio_by_countries", summary="Retrieves the ratio of cities in two specified countries. The operation calculates the proportion of cities in the first country relative to the second country. The input parameters are used to identify the countries for comparison.")
async def get_city_ratio_by_countries(country_name_1: str = Query(..., description="Name of the first country"), country_name_2: str = Query(..., description="Name of the second country")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Country_Name = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.Country_Name = ? THEN 1 ELSE 0 END) FROM City AS T1 INNER JOIN Country AS T2 ON T1.Country_Id = T2.Country_Id", (country_name_1, country_name_2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the difference in the number of matches played at two venues
@app.get("/v1/soccer_2016/match_venue_difference", operation_id="get_match_venue_difference", summary="Retrieves the difference in the total number of matches played at two specified venues. The operation compares the number of matches held at the first venue with the number of matches held at the second venue, and returns the difference.")
async def get_match_venue_difference(venue_name_1: str = Query(..., description="Name of the first venue"), venue_name_2: str = Query(..., description="Name of the second venue")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Venue_Name = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.Venue_Name = ? THEN 1 ELSE 0 END) FROM `Match` AS T1 INNER JOIN Venue AS T2 ON T1.Venue_Id = T2.Venue_Id", (venue_name_1, venue_name_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the number of matches played in a specific month of a specific year
@app.get("/v1/soccer_2016/matches_in_month_year", operation_id="get_matches_in_month_year", summary="Retrieves the total number of soccer matches played in a specific month of a given year. The month is represented by a single digit, and the year is provided in the YYYY format. This operation is useful for analyzing match frequency trends over time.")
async def get_matches_in_month_year(month: str = Query(..., description="Month (single digit)"), year: str = Query(..., description="Year (YYYY format)")):
    cursor.execute("SELECT SUM(CASE WHEN SUBSTR(Match_Date, 7, 1) = ? THEN 1 ELSE 0 END) FROM `Match` WHERE SUBSTR(Match_Date, 1, 4) = ?", (month, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players born between two dates
@app.get("/v1/soccer_2016/player_count_by_dob_range", operation_id="get_player_count_by_dob_range", summary="Retrieves the total number of soccer players born within a specified date range. The start and end dates of the range are provided as input parameters, allowing for a customizable time frame.")
async def get_player_count_by_dob_range(start_date: str = Query(..., description="Start date (YYYY-MM-DD format)"), end_date: str = Query(..., description="End date (YYYY-MM-DD format)")):
    cursor.execute("SELECT COUNT(Player_Id) AS cnt FROM Player WHERE DOB BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the number of matches involving a specific team in a specific year
@app.get("/v1/soccer_2016/matches_involving_team_in_year", operation_id="get_matches_involving_team_in_year", summary="Retrieves the total count of matches in which a specific team has participated during a given year. The team is identified by its unique ID, and the year is specified in the YYYY format.")
async def get_matches_involving_team_in_year(team_id: int = Query(..., description="Team ID"), year: str = Query(..., description="Year (YYYY format)")):
    cursor.execute("SELECT SUM(CASE WHEN Team_1 = ? OR Team_2 = ? THEN 1 ELSE 0 END) FROM `Match` WHERE SUBSTR(Match_Date, 1, 4) = ?", (team_id, team_id, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the Orange Cap winners who have won more than a specified number of times
@app.get("/v1/soccer_2016/orange_cap_winners_by_count", operation_id="get_orange_cap_winners_by_count", summary="Retrieve a list of Orange Cap winners who have won the title more than a specified number of times. The input parameter determines the minimum number of wins required for a player to be included in the response.")
async def get_orange_cap_winners_by_count(min_count: int = Query(..., description="Minimum number of times won")):
    cursor.execute("SELECT Orange_Cap FROM Season GROUP BY Orange_Cap HAVING COUNT(Season_Year) > ?", (min_count,))
    result = cursor.fetchall()
    if not result:
        return {"orange_cap_winners": []}
    return {"orange_cap_winners": [row[0] for row in result]}

# Endpoint to get the count of matches in a specific season
@app.get("/v1/soccer_2016/match_count_by_season", operation_id="get_match_count_by_season", summary="Retrieves the total number of matches played in a specified season.")
async def get_match_count_by_season(season_id: int = Query(..., description="Season ID")):
    cursor.execute("SELECT COUNT(Match_Id) FROM `Match` WHERE Season_Id = ?", (season_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of umpires from a specific country
@app.get("/v1/soccer_2016/umpire_count_by_country", operation_id="get_umpire_count_by_country", summary="Retrieves the total number of umpires from a specified country. The operation calculates the sum of umpires based on the provided country name, using an inner join between the Country and Umpire tables.")
async def get_umpire_count_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Country_Name = ? THEN 1 ELSE 0 END) FROM Country AS T1 INNER JOIN Umpire AS T2 ON T1.Country_ID = T2.Umpire_Country", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top player based on the number of Man of the Match awards
@app.get("/v1/soccer_2016/top_player_by_man_of_the_match", operation_id="get_top_player_by_man_of_the_match", summary="Retrieves the top player(s) who have received the most Man of the Match awards, up to the specified limit.")
async def get_top_player_by_man_of_the_match(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Match AS T2 ON T1.Player_Id = T2.Man_of_the_Match GROUP BY T2.Man_of_the_Match ORDER BY COUNT(T2.Man_of_the_Match) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the top country based on the number of players
@app.get("/v1/soccer_2016/top_country_by_player_count", operation_id="get_top_country_by_player_count", summary="Retrieves the country with the highest number of players, allowing the user to limit the number of results returned. The operation ranks countries based on the count of associated players and returns the top-ranked country.")
async def get_top_country_by_player_count(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Country_Name FROM Country AS T1 INNER JOIN Player AS T2 ON T1.Country_Id = T2.Country_Name GROUP BY T2.Country_Name ORDER BY COUNT(T2.Country_Name) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the count of a specific player's Orange Cap awards
@app.get("/v1/soccer_2016/player_orange_cap_count", operation_id="get_player_orange_cap_count", summary="Retrieves the total number of Orange Cap awards won by a specific player in the 2016 soccer season.")
async def get_player_orange_cap_count(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Player_Name = ? THEN 1 ELSE 0 END) AS cnt FROM Player AS T1 INNER JOIN Season AS T2 ON T1.Player_Id = T2.Orange_Cap", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top season based on the number of matches in a specific venue
@app.get("/v1/soccer_2016/top_season_by_venue", operation_id="get_top_season_by_venue", summary="Retrieve the season with the highest number of matches held at a specified venue. The operation allows you to limit the number of results returned. The data is sorted in descending order based on the count of matches per season.")
async def get_top_season_by_venue(venue_name: str = Query(..., description="Name of the venue"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Season_Id FROM `Match` AS T1 INNER JOIN Venue AS T2 ON T1.Venue_Id = T2.Venue_Id WHERE T2.Venue_Name = ? GROUP BY T1.Season_Id ORDER BY COUNT(T1.Season_Id) DESC LIMIT ?", (venue_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"season_id": []}
    return {"season_id": result[0]}

# Endpoint to get the top team based on the number of matches won in a specific season
@app.get("/v1/soccer_2016/top_team_by_season", operation_id="get_top_team_by_season", summary="Retrieves the team that has won the most matches in a specified season. The operation allows you to limit the number of results returned. The team is determined by counting the number of matches won and ranking them in descending order.")
async def get_top_team_by_season(season_id: int = Query(..., description="ID of the season"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Team_Name FROM Team WHERE Team_Id = ( SELECT Match_Winner FROM `Match` WHERE season_Id = ? GROUP BY Match_Winner ORDER BY COUNT(Match_Winner) DESC LIMIT ? )", (season_id, limit))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the top venue based on the number of matches played by a specific team
@app.get("/v1/soccer_2016/top_venue_by_team", operation_id="get_top_venue_by_team", summary="Retrieves the venue where a specified team has played the most matches, up to a defined limit. The response is sorted in descending order based on the number of matches played at each venue.")
async def get_top_venue_by_team(team_name: str = Query(..., description="Name of the team"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.Venue_Name FROM Team AS T1 INNER JOIN Match AS T2 ON T1.Team_Id = T2.Team_1 INNER JOIN Venue AS T3 ON T2.Venue_Id = T3.Venue_Id WHERE T1.Team_Name = ? GROUP BY T3.Venue_Id ORDER BY COUNT(T3.Venue_Id) DESC LIMIT ?", (team_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"venue_name": []}
    return {"venue_name": result[0]}

# Endpoint to get the top team based on the number of matches lost
@app.get("/v1/soccer_2016/top_team_by_losses", operation_id="get_top_team_by_losses", summary="Retrieves the team that has lost the most matches, up to a specified limit. The team is determined by counting the number of matches where the team was not the winner. The result is ordered in descending order based on the number of losses.")
async def get_top_team_by_losses(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Team_Name FROM Team AS T1 INNER JOIN ( SELECT COUNT(Team_1) AS a, Team_1 FROM Match WHERE Team_1 <> Match_Winner GROUP BY Team_1 UNION SELECT COUNT(Team_2) AS a, Team_2 FROM Match WHERE Team_2 <> Match_Winner GROUP BY Team_2 ORDER BY a DESC LIMIT ? ) AS T2 ON T1.Team_Id = T2.Team_1", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the first Man of the Match based on match date
@app.get("/v1/soccer_2016/first_man_of_the_match", operation_id="get_first_man_of_the_match", summary="Retrieves the first Man of the Match based on the earliest match date. The operation allows you to limit the number of results returned. This endpoint is useful for identifying the earliest player to receive the Man of the Match award in the 2016 soccer season.")
async def get_first_man_of_the_match(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Player_Name FROM Player WHERE Player_Id = ( SELECT Man_of_the_Match FROM `Match` ORDER BY match_date ASC LIMIT ? )", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the first match date for a specific team
@app.get("/v1/soccer_2016/first_match_date_by_team", operation_id="get_first_match_date_by_team", summary="Retrieves the earliest match date for a specified team. The operation returns the date of the first match involving the team, sorted in ascending order. The number of results can be limited by providing a value for the 'limit' parameter.")
async def get_first_match_date_by_team(team_name: str = Query(..., description="Name of the team"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Match_Date FROM `Match` WHERE team_1 = ( SELECT Team_Id FROM Team WHERE Team_Name = ? ) OR Team_2 = ( SELECT Team_Id FROM Team WHERE Team_Name = ? ) ORDER BY Match_Date ASC LIMIT ?", (team_name, team_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"match_date": []}
    return {"match_date": result[0]}

# Endpoint to get the count of players with a specific batting hand from a specific country
@app.get("/v1/soccer_2016/player_count_by_batting_hand_and_country", operation_id="get_player_count_by_batting_hand_and_country", summary="Retrieves the total number of players from a specified country who use a particular batting hand. The operation filters players based on their batting hand and country, then aggregates the count.")
async def get_player_count_by_batting_hand_and_country(batting_hand: str = Query(..., description="Batting hand of the player"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Batting_hand = ? THEN 1 ELSE 0 END) AS cnt FROM Batting_Style AS T1 INNER JOIN Player AS T2 ON T1.Batting_Id = T2.Batting_hand INNER JOIN Country AS T3 ON T2.Country_Name = T3.Country_Id WHERE T3.Country_Name = ?", (batting_hand, country_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the captain of a specific team
@app.get("/v1/soccer_2016/team_captain", operation_id="get_team_captain", summary="Retrieves the name of the player who most frequently served as the captain of a specific team, based on the team's name, ID, and the role description and ID associated with the captain's position.")
async def get_team_captain(team_name: str = Query(..., description="Name of the team"), team_id: int = Query(..., description="ID of the team"), role_desc: str = Query(..., description="Description of the role"), role_id: int = Query(..., description="ID of the role")):
    cursor.execute("SELECT T4.Player_Name FROM Team AS T1 INNER JOIN Player_Match AS T2 ON T1.Team_id = T2.Team_id INNER JOIN Rolee AS T3 ON T2.Role_Id = T3.Role_Id INNER JOIN Player AS T4 ON T2.Player_Id = T4.Player_Id WHERE T1.Team_Name = ? AND T1.Team_Id = ? AND T3.Role_Desc = ? AND T3.Role_Id = ? GROUP BY T4.Player_Id ORDER BY COUNT(T3.Role_Id) DESC LIMIT 1", (team_name, team_id, role_desc, role_id))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the percentage of players with a specific batting hand
@app.get("/v1/soccer_2016/batting_hand_percentage", operation_id="get_batting_hand_percentage", summary="Retrieves the percentage of soccer players in the 2016 dataset who use a specific batting hand. The batting hand is provided as an input parameter, and the result is calculated by comparing the count of players with the specified batting hand to the total number of players in the dataset.")
async def get_batting_hand_percentage(batting_hand: str = Query(..., description="Batting hand of the player")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Batting_hand = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.Player_Id) FROM Batting_Style AS T1 INNER JOIN Player AS T2 ON T2.Batting_hand = T1.Batting_Id", (batting_hand,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get player names based on date of birth
@app.get("/v1/soccer_2016/player_names_by_dob", operation_id="get_player_names_by_dob", summary="Retrieves the names of soccer players born on a specific date. The date of birth must be provided in the 'YYYY-MM-DD' format.")
async def get_player_names_by_dob(dob: str = Query(..., description="Date of birth in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT Player_name FROM Player WHERE DOB = ?", (dob,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the count of matches played by a specific player
@app.get("/v1/soccer_2016/match_count_by_player", operation_id="get_match_count_by_player", summary="Retrieves the total number of matches played by a specific soccer player in the 2016 season. The player is identified by their unique ID.")
async def get_match_count_by_player(player_id: int = Query(..., description="ID of the player")):
    cursor.execute("SELECT SUM(CASE WHEN Player_Id = ? THEN 1 ELSE 0 END) FROM Player_Match", (player_id,))
    result = cursor.fetchone()
    if not result:
        return {"match_count": []}
    return {"match_count": result[0]}

# Endpoint to get the team with the highest win margin
@app.get("/v1/soccer_2016/team_with_highest_win_margin", operation_id="get_team_with_highest_win_margin", summary="Retrieves the name of the team with the highest win margin from the 2016 soccer season. The operation calculates the win margin for each team based on their match results and returns the team with the highest margin.")
async def get_team_with_highest_win_margin():
    cursor.execute("SELECT T2.Team_Name FROM Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Team_1 ORDER BY T1.Win_Margin DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the country name based on venue name
@app.get("/v1/soccer_2016/country_by_venue", operation_id="get_country_by_venue", summary="Retrieves the name of the country where the specified venue is located. The venue is identified by its name, which is provided as an input parameter.")
async def get_country_by_venue(venue_name: str = Query(..., description="Name of the venue")):
    cursor.execute("SELECT T3.Country_Name FROM Venue AS T1 INNER JOIN City AS T2 ON T2.City_Id = T1.City_Id INNER JOIN Country AS T3 ON T3.Country_Id = T2.Country_id WHERE T1.Venue_Name = ?", (venue_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the team name based on match ID and team name
@app.get("/v1/soccer_2016/team_name_by_match_id", operation_id="get_team_name_by_match_id", summary="Retrieves the name of a specific team that participated in a given match. The operation requires the match ID and the team name as input parameters to accurately identify the team. The team name is returned as the result.")
async def get_team_name_by_match_id(match_id: int = Query(..., description="ID of the match"), team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT T3.Team_Name FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T2.Player_Id = T1.Player_Id INNER JOIN Team AS T3 ON T3.Team_Id = T2.Team_Id WHERE T2.Match_Id = ? AND T3.Team_Name = ? GROUP BY T3.Team_Name", (match_id, team_name))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the team name based on match date and win margin
@app.get("/v1/soccer_2016/team_name_by_match_date_win_margin", operation_id="get_team_name_by_match_date_win_margin", summary="Retrieves the name of the team that won a match on a specific date with a given win margin. The match date should be provided in 'YYYY-MM-DD' format, and the win margin is the difference in goals between the winning and losing teams.")
async def get_team_name_by_match_date_win_margin(match_date: str = Query(..., description="Date of the match in 'YYYY-MM-DD' format"), win_margin: int = Query(..., description="Win margin")):
    cursor.execute("SELECT T1.Team_Name FROM Team AS T1 INNER JOIN Match AS T2 ON T1.Team_Id = T2.Match_Winner WHERE T2.Match_Date = ? AND T2.Win_Margin = ?", (match_date, win_margin))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the count of matches with a specific outcome type
@app.get("/v1/soccer_2016/match_count_by_outcome_type", operation_id="get_match_count_by_outcome_type", summary="Retrieves the total number of matches that resulted in a specific outcome type. The outcome type is provided as an input parameter.")
async def get_match_count_by_outcome_type(outcome_type: str = Query(..., description="Type of the outcome")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Outcome_Type = ? THEN 1 ELSE 0 END) FROM Match AS T1 INNER JOIN Outcome AS T2 ON T2.Outcome_Id = T1.Outcome_type", (outcome_type,))
    result = cursor.fetchone()
    if not result:
        return {"match_count": []}
    return {"match_count": result[0]}

# Endpoint to get the city names based on country name
@app.get("/v1/soccer_2016/city_names_by_country", operation_id="get_city_names_by_country", summary="Retrieves a list of city names associated with the specified country in the 2016 soccer dataset. The operation filters the cities based on the provided country name and returns the corresponding city names.")
async def get_city_names_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.City_Name FROM City AS T1 INNER JOIN Country AS T2 ON T2.Country_Id = T1.Country_id WHERE T2.Country_Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get the sum of matches won by a specific team
@app.get("/v1/soccer_2016/sum_matches_won_by_team", operation_id="get_sum_matches_won_by_team", summary="Retrieves the total number of matches won by a specific team. The team is identified by its name, which is provided as an input parameter.")
async def get_sum_matches_won_by_team(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Team_Name = ? THEN 1 ELSE 0 END) FROM Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Match_Winner", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the team name of the match winner for a specific match date and match ID
@app.get("/v1/soccer_2016/team_name_by_match_date_and_id", operation_id="get_team_name_by_match_date_and_id", summary="Retrieves the name of the team that won the match on a specific date, given the match ID. The match date should be provided in 'YYYY%' format.")
async def get_team_name_by_match_date_and_id(match_date: str = Query(..., description="Match date in 'YYYY%' format"), match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT T2.Team_Name FROM Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Match_Winner WHERE T1.Match_Date LIKE ? AND T1.Match_Id = ?", (match_date, match_id))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the role description of a player in a specific match
@app.get("/v1/soccer_2016/role_description_by_player_and_match", operation_id="get_role_description_by_player_and_match", summary="Retrieves the role description of a specific player in a given match. The operation requires the match ID and the player's name as input parameters to accurately identify the role description.")
async def get_role_description_by_player_and_match(match_id: int = Query(..., description="Match ID"), player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT T3.Role_Desc FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T2.Player_Id = T1.Player_Id INNER JOIN Rolee AS T3 ON T3.Role_Id = T2.Role_Id WHERE T2.Match_Id = ? AND T1.Player_Name = ?", (match_id, player_name))
    result = cursor.fetchone()
    if not result:
        return {"role_description": []}
    return {"role_description": result[0]}

# Endpoint to get the sum of cities in a specific country
@app.get("/v1/soccer_2016/sum_cities_by_country", operation_id="get_sum_cities_by_country", summary="Retrieves the total number of cities in a specified country from the 2016 soccer database. The operation calculates the sum based on the provided country name.")
async def get_sum_cities_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Country_Name = ? THEN 1 ELSE 0 END) FROM City AS T1 INNER JOIN Country AS T2 ON T2.Country_Id = T1.Country_id", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of matches played at a specific venue
@app.get("/v1/soccer_2016/sum_matches_by_venue", operation_id="get_sum_matches_by_venue", summary="Retrieves the total number of matches played at a specified venue in the 2016 soccer season.")
async def get_sum_matches_by_venue(venue_name: str = Query(..., description="Name of the venue")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Venue_Name = ? THEN 1 ELSE 0 END) FROM Match AS T1 INNER JOIN Venue AS T2 ON T2.Venue_Id = T1.Venue_Id", (venue_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the win margin of matches between two teams on a specific date
@app.get("/v1/soccer_2016/win_margin_by_teams_and_date", operation_id="get_win_margin_by_teams_and_date", summary="Retrieve the win margin for matches played between two specified teams on a given date. The operation requires the names of both teams and the date of the match in 'YYYY-MM-DD' format.")
async def get_win_margin_by_teams_and_date(team_1: str = Query(..., description="Name of the first team"), team_2: str = Query(..., description="Name of the second team"), match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.Win_Margin FROM Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Team_1 INNER JOIN Team AS T3 ON T3.Team_Id = T1.Team_2 WHERE (T2.Team_Name = ? AND T3.Team_Name = ? AND T1.Match_Date = ?) OR (T2.Team_Name = ? AND T3.Team_Name = ? AND T1.Match_Date = ?)", (team_1, team_2, match_date, team_2, team_1, match_date))
    result = cursor.fetchone()
    if not result:
        return {"win_margin": []}
    return {"win_margin": result[0]}

# Endpoint to get distinct team names with win margins below 30% of the average win margin for a specific year
@app.get("/v1/soccer_2016/teams_below_avg_win_margin", operation_id="get_teams_below_avg_win_margin", summary="Retrieve the unique names of teams that have a win margin less than 30% of the average win margin for a given year. The year is specified by the 'match_date' parameter in 'YYYY%' format.")
async def get_teams_below_avg_win_margin(match_date: str = Query(..., description="Match date in 'YYYY%' format")):
    cursor.execute("SELECT DISTINCT CASE WHEN T1.Win_Margin < ( SELECT AVG(Win_Margin) * 0.3 FROM Match WHERE Match_Date LIKE ? ) THEN T2.Team_Name END, CASE WHEN T1.Win_Margin < ( SELECT AVG(Win_Margin) * 0.3 FROM Match WHERE Match_Date LIKE ? ) THEN T3.Team_Name END FROM Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Team_1 INNER JOIN Team AS T3 ON T3.Team_Id = T1.Team_2 WHERE T1.Match_Date LIKE ?", (match_date, match_date, match_date))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": result}

# Endpoint to get the percentage of players with a specific role born in a specific year
@app.get("/v1/soccer_2016/percentage_players_by_role_and_dob", operation_id="get_percentage_players_by_role_and_dob", summary="Retrieves the percentage of players with a specific role born in a particular year. The role is identified by its description, and the year is determined by the player's date of birth. The calculation is based on the total count of players with the specified role and the total count of players born in the given year.")
async def get_percentage_players_by_role_and_dob(role_desc: str = Query(..., description="Role description"), dob: str = Query(..., description="Date of birth in 'YYYY%' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Role_Desc = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.Role_Id) FROM Rolee AS T1 INNER JOIN Player_Match AS T2 ON T2.Role_Id = T1.Role_Id INNER JOIN Player AS T3 ON T3.Player_Id = T2.Player_Id WHERE T3.DOB LIKE ?", (role_desc, dob))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of overs in a specific match and innings
@app.get("/v1/soccer_2016/count_overs_by_match_and_innings", operation_id="get_count_overs_by_match_and_innings", summary="Retrieves the total number of overs played in a specific match and innings. The operation requires the unique identifier of the match and the innings number as input parameters.")
async def get_count_overs_by_match_and_innings(match_id: int = Query(..., description="Match ID"), innings_no: int = Query(..., description="Innings number")):
    cursor.execute("SELECT COUNT(Over_Id) FROM Ball_by_Ball WHERE Match_Id = ? AND Innings_No = ?", (match_id, innings_no))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest scoring batsman details in a specific match
@app.get("/v1/soccer_2016/highest_scoring_batsman_by_match", operation_id="get_highest_scoring_batsman_by_match", summary="Retrieves the details of the batsman who scored the most runs in a specific match. The match is identified by its unique ID. The returned data includes the over, ball, and innings information for the highest-scoring batsman.")
async def get_highest_scoring_batsman_by_match(match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT Over_Id, Ball_Id, Innings_No FROM Batsman_Scored WHERE Match_Id = ? ORDER BY Runs_Scored DESC LIMIT 1", (match_id,))
    result = cursor.fetchone()
    if not result:
        return {"batsman_details": []}
    return {"batsman_details": result}

# Endpoint to get match IDs based on over ID
@app.get("/v1/soccer_2016/match_ids_by_over_id", operation_id="get_match_ids_by_over_id", summary="Retrieves a list of up to 5 unique match IDs that correspond to the specified over ID. The matches are selected from the Ball_by_Ball table and grouped by their respective Match_Id.")
async def get_match_ids_by_over_id(over_id: int = Query(..., description="Over ID")):
    cursor.execute("SELECT Match_Id FROM Ball_by_Ball WHERE Over_Id = ? GROUP BY Match_Id LIMIT 5", (over_id,))
    result = cursor.fetchall()
    if not result:
        return {"match_ids": []}
    return {"match_ids": [row[0] for row in result]}

# Endpoint to get the sum of wickets taken in a specific innings for a given match ID
@app.get("/v1/soccer_2016/sum_wickets_taken_by_match_innings", operation_id="get_sum_wickets_taken_by_match_innings", summary="Retrieves the total number of wickets taken in a specific innings of a given match. The match is identified by its unique ID, and the innings is specified by its number.")
async def get_sum_wickets_taken_by_match_innings(match_id: int = Query(..., description="Match ID"), innings_no: int = Query(..., description="Innings number")):
    cursor.execute("SELECT SUM(CASE WHEN Match_Id = ? THEN 1 ELSE 0 END) FROM Wicket_Taken WHERE Innings_No = ?", (match_id, innings_no))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get match IDs based on match date
@app.get("/v1/soccer_2016/match_ids_by_date", operation_id="get_match_ids_by_date", summary="Retrieves a list of match IDs that took place on a specified date. The date should be provided in the 'YYYY-MM-DD' format.")
async def get_match_ids_by_date(match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT Match_Id FROM Match WHERE Match_Date LIKE ?", ('%' + match_date + '%',))
    result = cursor.fetchall()
    if not result:
        return {"match_ids": []}
    return {"match_ids": [row[0] for row in result]}

# Endpoint to get match IDs based on out type
@app.get("/v1/soccer_2016/match_ids_by_out_type", operation_id="get_match_ids_by_out_type", summary="Retrieves a list of match IDs where the specified out type was recorded. The out type is identified by its name.")
async def get_match_ids_by_out_type(out_name: str = Query(..., description="Out type name")):
    cursor.execute("SELECT T1.Match_Id FROM Wicket_Taken AS T1 INNER JOIN Out_Type AS T2 ON T2.Out_Id = T1.Kind_Out WHERE T2.Out_Name = ?", (out_name,))
    result = cursor.fetchall()
    if not result:
        return {"match_ids": []}
    return {"match_ids": [row[0] for row in result]}

# Endpoint to get the sum of wickets taken in a specific innings for a given out type
@app.get("/v1/soccer_2016/sum_wickets_taken_by_innings_out_type", operation_id="get_sum_wickets_taken_by_innings_out_type", summary="Retrieves the total number of wickets taken in a specified innings for a given out type. The operation calculates the sum based on the provided innings number and out type name.")
async def get_sum_wickets_taken_by_innings_out_type(innings_no: int = Query(..., description="Innings number"), out_name: str = Query(..., description="Out type name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Innings_No = ? THEN 1 ELSE 0 END) FROM Wicket_Taken AS T1 INNER JOIN Out_Type AS T2 ON T2.Out_Id = T1.Kind_Out WHERE T2.Out_Name = ?", (innings_no, out_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of matches where a specific player was man of the match
@app.get("/v1/soccer_2016/sum_man_of_the_match_by_player", operation_id="get_sum_man_of_the_match_by_player", summary="Retrieves the total number of matches in which a specified player was awarded the 'Man of the Match' title. The operation calculates this sum by comparing the provided player's name with the 'Man of the Match' field in the 'Match' table, which is linked to the 'Player' table via the 'Player_Id' field.")
async def get_sum_man_of_the_match_by_player(player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Player_Name = ? THEN 1 ELSE 0 END) FROM Match AS T1 INNER JOIN Player AS T2 ON T2.Player_Id = T1.Man_of_the_Match", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get player names and DOBs based on country and year of birth
@app.get("/v1/soccer_2016/player_names_dobs_by_country_dob", operation_id="get_player_names_dobs_by_country_dob", summary="Retrieves the names and dates of birth of soccer players born in a specific year and country. The operation filters players based on the provided year of birth and country name.")
async def get_player_names_dobs_by_country_dob(dob_year: str = Query(..., description="Year of birth in 'YYYY' format"), country_name: str = Query(..., description="Country name")):
    cursor.execute("SELECT T2.Player_Name, T2.DOB FROM Country AS T1 INNER JOIN Player AS T2 ON T2.Country_Name = T1.Country_Id WHERE T2.DOB LIKE ? AND T1.Country_Name = ?", (dob_year + '%', country_name))
    result = cursor.fetchall()
    if not result:
        return {"player_info": []}
    return {"player_info": [{"player_name": row[0], "dob": row[1]} for row in result]}

# Endpoint to get player names who were man of the match in a specific season
@app.get("/v1/soccer_2016/player_names_man_of_the_match_by_season", operation_id="get_player_names_man_of_the_match_by_season", summary="Retrieves the names of players who were awarded the 'Man of the Match' title during a specified season. The operation filters matches based on the provided season year and groups the results by player name.")
async def get_player_names_man_of_the_match_by_season(season_year: int = Query(..., description="Season year")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Match AS T2 ON T2.Man_of_the_Match = T1.Player_Id INNER JOIN Season AS T3 ON T3.Season_Id = T2.Season_Id WHERE T3.Season_Year = ? GROUP BY T1.Player_Name", (season_year,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the win percentage of a specific team
@app.get("/v1/soccer_2016/win_percentage_by_team", operation_id="get_win_percentage_by_team", summary="Retrieves the win percentage of a specified team in the 2016 soccer season. The operation calculates the percentage by comparing the number of matches won by the team to the total number of matches played by the team. The team is identified by its name, and the match winner is determined by the provided match winner ID.")
async def get_win_percentage_by_team(match_winner: int = Query(..., description="Match winner ID"), team_name: str = Query(..., description="Team name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Match_Winner = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.Match_Id) FROM Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Team_1 INNER JOIN Team AS T3 ON T3.Team_Id = T1.Team_2 WHERE T2.Team_Name = ? OR T3.Team_Name = ?", (match_winner, team_name, team_name))
    result = cursor.fetchone()
    if not result:
        return {"win_percentage": []}
    return {"win_percentage": result[0]}

# Endpoint to get player names and their countries based on team name and match date
@app.get("/v1/soccer_2016/player_country_by_team_match", operation_id="get_player_country_by_team_match", summary="Retrieve the names of players and their respective countries based on the provided team name and match date. The team name and match date are used to filter the results, ensuring that only relevant player and country information is returned.")
async def get_player_country_by_team_match(team_name: str = Query(..., description="Name of the team"), match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T4.Player_Name, T5.Country_Name FROM Player_Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Team_Id INNER JOIN Match AS T3 ON T3.Match_Id = T1.Match_Id INNER JOIN Player AS T4 ON T4.Player_Id = T1.Player_Id INNER JOIN Country AS T5 ON T5.Country_Id = T4.Country_Name WHERE T2.Team_Name = ? AND T3.Match_Date = ?", (team_name, match_date))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get player names and their dates of birth based on bowling skill
@app.get("/v1/soccer_2016/player_dob_by_bowling_skill", operation_id="get_player_dob_by_bowling_skill", summary="Retrieves the names and dates of birth of soccer players from the 2016 season who possess a specific bowling skill. The operation filters players based on the provided bowling skill and returns their names and dates of birth.")
async def get_player_dob_by_bowling_skill(bowling_skill: str = Query(..., description="Bowling skill of the player")):
    cursor.execute("SELECT T1.Player_Name, T1.DOB FROM Player AS T1 INNER JOIN Bowling_Style AS T2 ON T2.Bowling_Id = T1.Bowling_skill WHERE T2.Bowling_skill = ?", (bowling_skill,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get country names based on umpire name
@app.get("/v1/soccer_2016/country_by_umpire_name", operation_id="get_country_by_umpire_name", summary="Retrieves the names of countries associated with a specific umpire in the 2016 soccer dataset. The operation uses the provided umpire name to look up the corresponding country information.")
async def get_country_by_umpire_name(umpire_name: str = Query(..., description="Name of the umpire")):
    cursor.execute("SELECT T1.Country_Name FROM Country AS T1 INNER JOIN Umpire AS T2 ON T2.Umpire_Country = T1.Country_Id WHERE T2.Umpire_Name = ?", (umpire_name,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get player names based on match date, role description, and match winner
@app.get("/v1/soccer_2016/player_by_match_date_role_winner", operation_id="get_player_by_match_date_role_winner", summary="Get player names based on match date, role description, and match winner")
async def get_player_by_match_date_role_winner(match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format"), role_desc: str = Query(..., description="Role description"), match_winner: int = Query(..., description="Match winner team ID")):
    cursor.execute("SELECT T3.Player_Name FROM Player_Match AS T1 INNER JOIN Match AS T2 ON T2.Match_Id = T1.Match_Id INNER JOIN Player AS T3 ON T3.Player_Id = T1.Player_Id INNER JOIN Rolee AS T4 ON T4.Role_Id = T1.Role_Id WHERE T2.Match_Date = ? AND T4.Role_Desc = ? AND T2.Match_Winner = T1.Team_Id", (match_date, role_desc, match_winner))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get team names and match counts based on player name
@app.get("/v1/soccer_2016/team_match_count_by_player", operation_id="get_team_match_count_by_player", summary="Retrieves the names of teams and their respective match counts for a given player. The player's name is used to filter the results.")
async def get_team_match_count_by_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T3.Team_Name, COUNT(T2.Match_Id) FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T2.Player_Id = T1.Player_Id INNER JOIN Team AS T3 ON T3.Team_Id = T2.Team_Id WHERE T1.Player_Name = ?", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": result}

# Endpoint to get the percentage of matches played at a specific venue in a given city
@app.get("/v1/soccer_2016/venue_match_percentage", operation_id="get_venue_match_percentage", summary="Retrieves the percentage of matches played at a specified venue within a given city. This operation calculates the proportion of matches held at the venue by comparing the total number of matches at the venue to the total number of matches in the city.")
async def get_venue_match_percentage(venue_name: str = Query(..., description="Name of the venue"), city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Venue_Name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.Match_Id) FROM City AS T1 INNER JOIN Venue AS T2 ON T2.City_Id = T1.City_Id INNER JOIN Match AS T3 ON T3.Venue_Id = T2.Venue_Id WHERE T1.City_Name = ?", (venue_name, city_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of wickets taken by a specific out type in a given match
@app.get("/v1/soccer_2016/wicket_out_percentage", operation_id="get_wicket_out_percentage", summary="Retrieves the percentage of wickets taken by a specific out type in a given match. The operation calculates this percentage by comparing the total number of wickets taken by the specified out type to the overall number of wickets taken in the match. The input parameters include the name of the out type and the match ID.")
async def get_wicket_out_percentage(out_name: str = Query(..., description="Name of the out type"), match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Out_Name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.Player_Out) FROM Wicket_Taken AS T1 INNER JOIN Out_Type AS T2 ON T2.Out_Id = T1.Kind_Out WHERE T1.Match_Id = ?", (out_name, match_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of matches with a specific toss decision within a date range
@app.get("/v1/soccer_2016/toss_decision_percentage", operation_id="get_toss_decision_percentage", summary="Retrieve the percentage of matches with a specific toss decision outcome that occurred within a specified date range. The calculation is based on the total number of matches played during the given period.")
async def get_toss_decision_percentage(toss_name: str = Query(..., description="Name of the toss decision"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Toss_Name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.Toss_Id) FROM Match AS T1 INNER JOIN Toss_Decision AS T2 ON T2.Toss_Id = T1.Toss_Decide WHERE T1.Match_Date BETWEEN ? AND ?", (toss_name, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the toss winners based on toss decision
@app.get("/v1/soccer_2016/toss_winners_by_decision", operation_id="get_toss_winners_by_decision", summary="Retrieves the team that won the toss based on the specified toss decision. The toss decision is identified by its unique ID.")
async def get_toss_winners_by_decision(toss_decide: int = Query(..., description="Toss decision ID")):
    cursor.execute("SELECT Toss_Winner FROM Match WHERE Toss_Decide = ?", (toss_decide,))
    result = cursor.fetchall()
    if not result:
        return {"toss_winners": []}
    return {"toss_winners": result}

# Endpoint to get match IDs based on the man of the match
@app.get("/v1/soccer_2016/match_ids_by_man_of_the_match", operation_id="get_match_ids_by_man_of_the_match", summary="Retrieves the unique identifiers of all matches in which a specified player was named the man of the match. The player's name is used to filter the matches.")
async def get_match_ids_by_man_of_the_match(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T1.Match_Id FROM Match AS T1 INNER JOIN Player AS T2 ON T2.Player_Id = T1.Man_of_the_Match WHERE T2.Player_Name = ?", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"match_ids": []}
    return {"match_ids": result}

# Endpoint to get the dates of birth of players who were man of the match
@app.get("/v1/soccer_2016/man_of_the_match_dob", operation_id="get_man_of_the_match_dob", summary="Retrieves the birth dates of players who were awarded the 'Man of the Match' title in the 2016 soccer season. This operation provides a comprehensive list of these players' birth dates, offering insights into their age distribution and potential correlation with performance.")
async def get_man_of_the_match_dob():
    cursor.execute("SELECT T2.DOB FROM Match AS T1 INNER JOIN Player AS T2 ON T2.Player_Id = T1.Man_of_the_Match")
    result = cursor.fetchall()
    if not result:
        return {"dob": []}
    return {"dob": [row[0] for row in result]}

# Endpoint to get the names of teams that won the toss in matches within a given range of match IDs
@app.get("/v1/soccer_2016/toss_winners_by_match_id_range", operation_id="get_toss_winners_by_match_id_range", summary="Retrieve the names of teams that won the toss in matches falling within the specified range of match IDs. The range is defined by the minimum and maximum match IDs provided as input parameters.")
async def get_toss_winners_by_match_id_range(min_match_id: int = Query(..., description="Minimum match ID"), max_match_id: int = Query(..., description="Maximum match ID")):
    cursor.execute("SELECT T2.Team_Name FROM Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Toss_Winner WHERE T1.Match_Id BETWEEN ? AND ?", (min_match_id, max_match_id))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get the names of teams that played against a specific team
@app.get("/v1/soccer_2016/opponent_teams_by_team_name", operation_id="get_opponent_teams_by_team_name", summary="Retrieve the names of teams that have played against a specified team in the 2016 soccer season. The input parameter is the name of the team for which opponent teams are sought.")
async def get_opponent_teams_by_team_name(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT T2.Team_Name FROM Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Team_2 WHERE T1.Team_1 = ( SELECT Team_Id FROM Team WHERE Team_Name = ? ) GROUP BY T2.Team_Name", (team_name,))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get the name of the team that won a specific match
@app.get("/v1/soccer_2016/match_winner_by_match_id", operation_id="get_match_winner_by_match_id", summary="Retrieves the name of the team that won the match corresponding to the provided match ID. The match ID is used to identify the specific match and the winning team's name is returned.")
async def get_match_winner_by_match_id(match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT T2.Team_Name FROM Match AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Match_Winner WHERE T1.Match_Id = ?", (match_id,))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the match IDs of matches played at a specific venue
@app.get("/v1/soccer_2016/match_ids_by_venue_name", operation_id="get_match_ids_by_venue_name", summary="Retrieve the unique identifiers of matches that were played at a specified venue. The operation requires the venue's name as input and returns a list of corresponding match IDs.")
async def get_match_ids_by_venue_name(venue_name: str = Query(..., description="Name of the venue")):
    cursor.execute("SELECT T1.Match_Id FROM Match AS T1 INNER JOIN Venue AS T2 ON T2.Venue_Id = T1.Venue_Id WHERE T2.Venue_Name = ?", (venue_name,))
    result = cursor.fetchall()
    if not result:
        return {"match_ids": []}
    return {"match_ids": [row[0] for row in result]}

# Endpoint to get the names of venues where matches were played in a specific season
@app.get("/v1/soccer_2016/venues_by_season_id", operation_id="get_venues_by_season_id", summary="Retrieve the names of venues where matches were played during a specified season. The operation filters matches by the provided season ID and returns a unique list of venue names.")
async def get_venues_by_season_id(season_id: int = Query(..., description="Season ID")):
    cursor.execute("SELECT T2.Venue_Name FROM Match AS T1 INNER JOIN Venue AS T2 ON T2.Venue_Id = T1.Venue_Id WHERE T1.Season_Id = ? GROUP BY T2.Venue_Name", (season_id,))
    result = cursor.fetchall()
    if not result:
        return {"venue_names": []}
    return {"venue_names": [row[0] for row in result]}

# Endpoint to get the name of the city where a specific venue is located
@app.get("/v1/soccer_2016/city_name_by_venue_name", operation_id="get_city_name_by_venue_name", summary="Retrieves the name of the city where the specified venue is located. The operation requires the venue's name as input and returns the corresponding city name.")
async def get_city_name_by_venue_name(venue_name: str = Query(..., description="Name of the venue")):
    cursor.execute("SELECT T1.City_Name FROM City AS T1 INNER JOIN Venue AS T2 ON T2.City_Id = T1.City_Id WHERE T2.Venue_Name = ?", (venue_name,))
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the names of venues located in a specific city
@app.get("/v1/soccer_2016/venues_by_city_name", operation_id="get_venues_by_city_name", summary="Retrieves the names of all soccer venues situated in a specified city. The operation requires the name of the city as an input parameter.")
async def get_venues_by_city_name(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T2.Venue_Name FROM City AS T1 INNER JOIN Venue AS T2 ON T2.City_Id = T1.City_Id WHERE T1.City_Name = ?", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"venue_names": []}
    return {"venue_names": [row[0] for row in result]}

# Endpoint to get the names of match winners at venues with names starting with a specific prefix
@app.get("/v1/soccer_2016/match_winners_by_venue_name_prefix", operation_id="get_match_winners_by_venue_name_prefix", summary="Retrieve the names of match winners from games played at venues with names starting with a specified prefix. The prefix is used to filter the venues and subsequently identify the corresponding match winners.")
async def get_match_winners_by_venue_name_prefix(venue_name_prefix: str = Query(..., description="Prefix of the venue name")):
    cursor.execute("SELECT T2.Match_Winner FROM Venue AS T1 INNER JOIN Match AS T2 ON T1.Venue_Id = T2.Venue_Id WHERE T1.Venue_Name LIKE ?", (venue_name_prefix + '%',))
    result = cursor.fetchall()
    if not result:
        return {"match_winners": []}
    return {"match_winners": [row[0] for row in result]}

# Endpoint to get city names based on venue name pattern
@app.get("/v1/soccer_2016/city_names_by_venue_name", operation_id="get_city_names_by_venue_name", summary="Retrieves the names of cities that have venues matching the specified name pattern. The input pattern should include the '%' wildcard character to match any sequence of characters. For example, 'Stadium%' would return cities with venues starting with 'Stadium'. The operation performs a case-insensitive search.")
async def get_city_names_by_venue_name(venue_name_pattern: str = Query(..., description="Pattern to match venue names (use % for wildcard)")):
    cursor.execute("SELECT T2.City_Name FROM Venue AS T1 INNER JOIN City AS T2 ON T1.City_Id = T2.City_Id WHERE T1.Venue_Name LIKE ?", (venue_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get the sum of match wins for a specific team
@app.get("/v1/soccer_2016/sum_match_wins_by_team", operation_id="get_sum_match_wins_by_team", summary="Retrieves the total number of matches won by a specific soccer team in the 2016 season. The team is identified by its name, which is provided as an input parameter.")
async def get_sum_match_wins_by_team(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT SUM(T2.Match_Winner) FROM Team AS T1 INNER JOIN Match AS T2 ON T1.Team_Id = T2.Match_Winner WHERE T1.Team_Name = ?", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum_match_wins": []}
    return {"sum_match_wins": result[0]}

# Endpoint to get the sum of venue names for a specific city
@app.get("/v1/soccer_2016/sum_venue_names_by_city", operation_id="get_sum_venue_names_by_city", summary="Retrieves the total count of venues in a specified city. The operation calculates this count by summing the unique venue names associated with the provided city.")
async def get_sum_venue_names_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT SUM(T1.Venue_Name) FROM Venue AS T1 INNER JOIN City AS T2 ON T1.City_Id = T2.City_Id WHERE T2.City_Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum_venue_names": []}
    return {"sum_venue_names": result[0]}

# Endpoint to get the latest ball ID for a non-striker
@app.get("/v1/soccer_2016/latest_ball_id_for_non_striker", operation_id="get_latest_ball_id_for_non_striker", summary="Retrieves the most recent ball ID associated with a non-striker in the 2016 soccer season. The operation returns the highest ball ID from the Ball_by_Ball table where the non-striker matches the ball ID, providing the latest ball ID for a non-striker.")
async def get_latest_ball_id_for_non_striker():
    cursor.execute("SELECT Ball_Id FROM Ball_by_Ball WHERE Non_Striker = Ball_Id ORDER BY Ball_Id DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"latest_ball_id": []}
    return {"latest_ball_id": result[0]}

# Endpoint to get the percentage of runs scored in a specific over range for a given innings
@app.get("/v1/soccer_2016/percentage_runs_scored_in_over_range", operation_id="get_percentage_runs_scored_in_over_range", summary="Retrieves the percentage of runs scored between the specified minimum and maximum over IDs for a given innings. The calculation is based on the total runs scored in the innings.")
async def get_percentage_runs_scored_in_over_range(min_over: int = Query(..., description="Minimum over ID"), max_over: int = Query(..., description="Maximum over ID"), innings_no: int = Query(..., description="Innings number")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN ? < Over_Id AND Over_Id < ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(Runs_Scored) FROM Batsman_Scored WHERE Innings_No = ?", (min_over, max_over, innings_no))
    result = cursor.fetchone()
    if not result:
        return {"percentage_runs_scored": []}
    return {"percentage_runs_scored": result[0]}

# Endpoint to get the average innings number for a specific innings
@app.get("/v1/soccer_2016/average_innings_number", operation_id="get_average_innings_number", summary="Retrieves the average number of innings for a specific innings number in the 2016 soccer season. The input parameter determines the innings number for which the average is calculated.")
async def get_average_innings_number(innings_no: int = Query(..., description="Innings number")):
    cursor.execute("SELECT AVG(Innings_No) FROM Extra_Runs WHERE Innings_No = ?", (innings_no,))
    result = cursor.fetchone()
    if not result:
        return {"average_innings_number": []}
    return {"average_innings_number": result[0]}

# Endpoint to get the percentage of matches won by a margin greater than a specified value
@app.get("/v1/soccer_2016/percentage_matches_won_by_margin", operation_id="get_percentage_matches_won_by_margin", summary="Retrieves the percentage of matches won by a margin greater than the specified win margin. The calculation is based on the total number of matches in the database.")
async def get_percentage_matches_won_by_margin(win_margin: int = Query(..., description="Win margin")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN Win_Margin > ? THEN 1 ELSE 0 END) AS REAL) * 100 / TOTAL(Match_Id) FROM `Match`", (win_margin,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_matches_won": []}
    return {"percentage_matches_won": result[0]}

# Endpoint to get player names born between specific dates
@app.get("/v1/soccer_2016/player_names_by_dob_range", operation_id="get_player_names_by_dob_range", summary="Retrieves a list of player names who were born between the specified start and end dates, sorted in descending order by their date of birth. The start and end dates should be provided in 'YYYY-MM-DD' format.")
async def get_player_names_by_dob_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT Player_Name FROM Player WHERE DOB BETWEEN ? AND ? ORDER BY DOB DESC", (start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the sum of wickets taken with no fielders in a specific over
@app.get("/v1/soccer_2016/sum_wickets_no_fielders_by_over", operation_id="get_sum_wickets_no_fielders_by_over", summary="Retrieves the total number of wickets taken with no fielders present in a specific over of the 2016 soccer season.")
async def get_sum_wickets_no_fielders_by_over(over_id: int = Query(..., description="Over ID")):
    cursor.execute("SELECT SUM(CASE WHEN Fielders = '' THEN 1 ELSE 0 END) FROM Wicket_Taken WHERE Over_Id = ?", (over_id,))
    result = cursor.fetchone()
    if not result:
        return {"sum_wickets_no_fielders": []}
    return {"sum_wickets_no_fielders": result[0]}

# Endpoint to get the country with the highest number of umpires
@app.get("/v1/soccer_2016/country_with_most_umpires", operation_id="get_country_with_most_umpires", summary="Retrieves the country with the highest number of umpires in the 2016 soccer season. The operation calculates the total count of umpires for each country and returns the country with the maximum count.")
async def get_country_with_most_umpires():
    cursor.execute("SELECT T2.Country_Id, COUNT(T1.Umpire_Id) FROM Umpire AS T1 INNER JOIN Country AS T2 ON T2.Country_Id = T1.Umpire_Country GROUP BY T2.Country_Id ORDER BY COUNT(T1.Umpire_Id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country_with_most_umpires": []}
    return {"country_with_most_umpires": {"country_id": result[0], "umpire_count": result[1]}}

# Endpoint to get the percentage of players with a specific role description
@app.get("/v1/soccer_2016/percentage_players_role", operation_id="get_percentage_players_role", summary="Retrieves the percentage of players with a specific role description from the 2016 soccer dataset. The role description is provided as an input parameter, and the result is calculated based on the total number of players in the dataset.")
async def get_percentage_players_role(role_desc: str = Query(..., description="Role description of the player")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Role_Desc = ? THEN 1 ELSE 0 END) AS REAL) * 100 / TOTAL(T1.Player_Id) FROM Player_Match AS T1 INNER JOIN Rolee AS T2 ON T1.Role_Id = T2.Role_Id", (role_desc,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the player out based on the out type
@app.get("/v1/soccer_2016/player_out_by_out_type", operation_id="get_player_out_by_out_type", summary="Retrieves the details of the player who was out based on the specified out type. The out type is used to filter the results and return the relevant player out information.")
async def get_player_out_by_out_type(out_name: str = Query(..., description="Name of the out type")):
    cursor.execute("SELECT Player_Out FROM Wicket_Taken AS T1 INNER JOIN Out_Type AS T2 ON T1.Kind_Out = T2.Out_Id WHERE Out_Name = ?", (out_name,))
    result = cursor.fetchall()
    if not result:
        return {"player_out": []}
    return {"player_out": [row[0] for row in result]}

# Endpoint to get the percentage of players with a specific batting hand
@app.get("/v1/soccer_2016/percentage_players_batting_hand", operation_id="get_percentage_players_batting_hand", summary="Retrieves the percentage of soccer players in the 2016 dataset who use a specific batting hand. The batting hand is provided as an input parameter, and the result is calculated by comparing the count of players with the specified batting hand to the total number of players in the dataset.")
async def get_percentage_players_batting_hand(batting_hand: str = Query(..., description="Batting hand of the player")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Batting_hand = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.Country_Name) FROM Batting_Style AS T1 INNER JOIN Player AS T2 ON T1.Batting_id = T2.Batting_hand", (batting_hand,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of players with a specific bowling skill
@app.get("/v1/soccer_2016/percentage_players_bowling_skill", operation_id="get_percentage_players_bowling_skill", summary="Get the percentage of players with a specific bowling skill")
async def get_percentage_players_bowling_skill(bowling_skill: str = Query(..., description="Bowling skill of the player")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Bowling_skill = ? THEN 1 ELSE 0 END) AS REAL) * 100 / TOTAL(T1.Player_Id) FROM Player AS T1 INNER JOIN Bowling_Style AS T2 ON T1.Bowling_skill = T2.Bowling_Id", (bowling_skill,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of matches won by a specific type and margin
@app.get("/v1/soccer_2016/count_matches_win_type_margin", operation_id="get_count_matches_win_type_margin", summary="Retrieves the total number of matches won by a specific type and within a certain margin. The operation filters matches based on the provided win type and margin, counting only those that meet the criteria.")
async def get_count_matches_win_type_margin(win_type: str = Query(..., description="Type of win"), win_margin: int = Query(..., description="Win margin")):
    cursor.execute("SELECT COUNT(T2.Win_Id) FROM `Match` AS T1 INNER JOIN Win_By AS T2 ON T1.Win_Type = T2.Win_Id WHERE T2.Win_Type = ? AND T1.Win_Margin < ?", (win_type, win_margin))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of matches where the toss winner is the match winner
@app.get("/v1/soccer_2016/sum_matches_toss_winner_match_winner", operation_id="get_sum_matches_toss_winner_match_winner", summary="Retrieves the total count of matches in which the team that won the toss also won the match. The data is filtered by matches where the first team is the toss winner.")
async def get_sum_matches_toss_winner_match_winner():
    cursor.execute("SELECT SUM(CASE WHEN T1.Team_2 = T1.Match_Winner THEN 1 ELSE 0 END) FROM `Match` AS T1 INNER JOIN Venue AS T2 ON T1.Venue_Id = T2.Venue_Id WHERE T1.Team_1 = T1.Toss_Winner")
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the man of the series and country name for a specific season year
@app.get("/v1/soccer_2016/man_of_the_series_country", operation_id="get_man_of_the_series_country", summary="Retrieves the name of the man of the series and the corresponding country for a specified season year. The operation fetches this information by joining the Season, Player, and Country tables using the provided season year.")
async def get_man_of_the_series_country(season_year: int = Query(..., description="Year of the season")):
    cursor.execute("SELECT T2.Player_Name, T3.Country_Name FROM Season AS T1 INNER JOIN Player AS T2 ON T1.Man_of_the_Series = T2.Player_Id INNER JOIN Country AS T3 ON T2.Country_Name = T3.Country_Id WHERE T1.Season_Year = ?", (season_year,))
    result = cursor.fetchall()
    if not result:
        return {"man_of_the_series": []}
    return {"man_of_the_series": [{"player_name": row[0], "country_name": row[1]} for row in result]}

# Endpoint to get the venue with the most matches
@app.get("/v1/soccer_2016/most_matches_venue", operation_id="get_most_matches_venue", summary="Retrieves the name of the venue that hosted the highest number of matches in the 2016 soccer season. The operation returns the venue with the most matches based on the match records in the database.")
async def get_most_matches_venue():
    cursor.execute("SELECT T2.Venue_Name FROM `Match` AS T1 INNER JOIN Venue AS T2 ON T1.Venue_Id = T2.Venue_Id GROUP BY T2.Venue_Name ORDER BY COUNT(T2.Venue_Id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"venue_name": []}
    return {"venue_name": result[0]}

# Endpoint to get the city with the least matches of a specific win type
@app.get("/v1/soccer_2016/least_matches_city_win_type", operation_id="get_least_matches_city_win_type", summary="Retrieve the name of the city that has hosted the fewest matches resulting in a specific type of win. The win type is specified as an input parameter.")
async def get_least_matches_city_win_type(win_type: str = Query(..., description="Type of win")):
    cursor.execute("SELECT T4.City_Name FROM `Match` AS T1 INNER JOIN Win_By AS T2 ON T1.Win_Type = T2.Win_Id INNER JOIN Venue AS T3 ON T1.Venue_Id = T3.Venue_Id INNER JOIN City AS T4 ON T3.City_Id = T4.City_Id WHERE T2.Win_Type = ? GROUP BY T4.City_Id ORDER BY COUNT(T2.Win_Type) ASC LIMIT 1", (win_type,))
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the names of players who were man of the series more than a specified number of times
@app.get("/v1/soccer_2016/man_of_the_series_count", operation_id="get_man_of_the_series_count", summary="Retrieve the names of soccer players who have been awarded the 'Man of the Series' title more than a specified number of times. This operation requires the minimum count of 'Man of the Series' awards as an input parameter.")
async def get_man_of_the_series_count(man_of_the_series_count: int = Query(..., description="Number of times a player was man of the series")):
    cursor.execute("SELECT T2.Player_Name FROM Season AS T1 INNER JOIN Player AS T2 ON T1.Man_of_the_Series = T2.Player_Id WHERE T1.Man_of_the_Series > ?", (man_of_the_series_count,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get player names and their country names ordered by the count of out types
@app.get("/v1/soccer_2016/player_country_out_types", operation_id="get_player_country_out_types", summary="Retrieves a list of player names along with their respective country names, sorted in ascending order based on the count of their out types. This operation fetches data from the Player, Wicket_Taken, Out_Type, and Country tables, joining them based on the specified relationships.")
async def get_player_country_out_types():
    cursor.execute("SELECT T1.Player_Name, T4.Country_Name FROM Player AS T1 INNER JOIN Wicket_Taken AS T2 ON T1.Player_Id = T2.Fielders INNER JOIN Out_Type AS T3 ON T2.Kind_Out = T3.Out_Id INNER JOIN Country AS T4 ON T1.Country_Name = T4.Country_Id GROUP BY T1.Player_Name ORDER BY COUNT(T3.Out_Name) ASC")
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the percentage of matches won by the team that won the toss and chose to field
@app.get("/v1/soccer_2016/percentage_matches_won_by_toss_winner", operation_id="get_percentage_matches_won_by_toss_winner", summary="Retrieves the percentage of matches won by the team that won the toss and opted to field, based on the specified toss decision and win type.")
async def get_percentage_matches_won_by_toss_winner(toss_name: str = Query(..., description="Toss decision name"), win_type: str = Query(..., description="Win type")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.Team_1 = T1.Match_Winner = T1.Toss_Winner THEN 1 ELSE 0 END) AS REAL) * 100 / TOTAL(T1.Team_1) FROM `Match` AS T1 INNER JOIN Win_By AS T2 ON T1.Win_Type = T2.Win_Id INNER JOIN Toss_Decision AS T3 ON T1.Toss_Decide = T3.Toss_Id WHERE T3.Toss_Name = ? AND T2.Win_Type = ?", (toss_name, win_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average number of players out by a specific out type
@app.get("/v1/soccer_2016/average_players_out_by_type", operation_id="get_average_players_out_by_type", summary="Retrieves the average number of players out for a specific out type in the 2016 soccer season. The operation calculates the average based on the provided out type name.")
async def get_average_players_out_by_type(out_name: str = Query(..., description="Out type name")):
    cursor.execute("SELECT AVG(T1.Player_Out) FROM Wicket_Taken AS T1 INNER JOIN Out_Type AS T2 ON T1.Kind_Out = T2.Out_Id WHERE T2.Out_Name = ?", (out_name,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get distinct over IDs for a specific striker
@app.get("/v1/soccer_2016/distinct_over_ids_by_striker", operation_id="get_distinct_over_ids_by_striker", summary="Retrieves a unique set of over IDs associated with a specific striker in the 2016 soccer data.")
async def get_distinct_over_ids_by_striker(striker: int = Query(..., description="Striker ID")):
    cursor.execute("SELECT DISTINCT Over_Id FROM Ball_by_Ball WHERE Striker = ?", (striker,))
    result = cursor.fetchall()
    if not result:
        return {"over_ids": []}
    return {"over_ids": result}

# Endpoint to get the count of matches where the team that won the toss also won the match
@app.get("/v1/soccer_2016/count_matches_toss_winner_won", operation_id="get_count_matches_toss_winner_won", summary="Retrieves the total number of matches in which the team that won the toss (as indicated by the provided toss decision ID) also emerged victorious in the match.")
async def get_count_matches_toss_winner_won(toss_decide: int = Query(..., description="Toss decision ID")):
    cursor.execute("SELECT COUNT(Team_1) FROM `Match` WHERE Team_1 = Toss_Winner AND Toss_Decide = ?", (toss_decide,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches played in a specific month and year
@app.get("/v1/soccer_2016/count_matches_by_month_year", operation_id="get_count_matches_by_month_year", summary="Retrieves the total number of soccer matches played in a given month and year. The input parameter specifies the desired month and year in the 'YYYY-MM%' format. The response provides a count of matches that took place during the specified period.")
async def get_count_matches_by_month_year(match_date: str = Query(..., description="Match date in 'YYYY-MM%' format")):
    cursor.execute("SELECT SUM(CASE WHEN Match_Date LIKE ? THEN 1 ELSE 0 END) FROM `Match`", (match_date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players born before a specific date and not named a specific name
@app.get("/v1/soccer_2016/count_players_by_dob_and_name", operation_id="get_count_players_by_dob_and_name", summary="Retrieves the total number of players, excluding those with a specific name, who were born before a given date. The date of birth and player name are provided as input parameters.")
async def get_count_players_by_dob_and_name(dob: str = Query(..., description="Date of birth in 'YYYY-MM-DD' format"), player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT SUM(CASE WHEN DOB < ? THEN 1 ELSE 0 END) FROM Player WHERE Player_Name != ?", (dob, player_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches where a specific player was the man of the match
@app.get("/v1/soccer_2016/count_man_of_the_match", operation_id="get_count_man_of_the_match", summary="Retrieves the total number of matches in which a specified player was named the man of the match. The player is identified by their name.")
async def get_count_man_of_the_match(player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Player_Name = ? THEN 1 ELSE 0 END) FROM `Match` AS T1 INNER JOIN Player AS T2 ON T1.Man_of_the_Match = T2.Player_Id", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the player with the most roles in a specific team
@app.get("/v1/soccer_2016/top_player_by_roles_in_team", operation_id="get_top_player_by_roles_in_team", summary="Retrieves the name of the player who has played the most roles in a specific team. The team is identified by its name, which is provided as an input parameter. The player with the highest number of roles is determined by counting the distinct roles they have played in matches for the specified team.")
async def get_top_player_by_roles_in_team(team_name: str = Query(..., description="Team name")):
    cursor.execute("SELECT T3.Player_Name FROM Player_Match AS T1 INNER JOIN Team AS T2 ON T1.Team_Id = T2.Team_Id INNER JOIN Player AS T3 ON T1.Player_Id = T3.Player_Id WHERE T2.Team_Name = ? GROUP BY T3.Player_Name ORDER BY COUNT(T1.Role_Id) DESC LIMIT 1", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": result[0]}

# Endpoint to get the player with the most man of the series awards
@app.get("/v1/soccer_2016/top_player_by_man_of_the_series", operation_id="get_top_player_by_man_of_the_series", summary="Retrieves the name of the soccer player who has been awarded the 'Man of the Series' the most times in the 2016 season. The operation considers all matches and seasons, grouping by player name and ordering by the count of 'Man of the Series' awards in descending order. The top result is returned.")
async def get_top_player_by_man_of_the_series():
    cursor.execute("SELECT T3.Player_Name FROM Season AS T1 INNER JOIN Match AS T2 ON T1.Man_of_the_Series = T2.Man_of_the_Match INNER JOIN Player AS T3 ON T2.Man_of_the_Match = T3.Player_Id GROUP BY T3.Player_Name ORDER BY COUNT(T1.Man_of_the_Series) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": result[0]}

# Endpoint to get the season year and orange cap for a specific player
@app.get("/v1/soccer_2016/player_season_orange_cap", operation_id="get_player_season_orange_cap", summary="Retrieves the season year and corresponding orange cap for a specific player. The operation uses the provided player's name to filter the results and returns the season year and orange cap for each season the player has participated in.")
async def get_player_season_orange_cap(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T4.Season_Year, T4.Orange_Cap FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Match AS T3 ON T2.Match_Id = T3.Match_Id INNER JOIN Season AS T4 ON T3.Season_Id = T4.Season_Id WHERE T1.Player_Name = ? GROUP BY T4.Season_Year, T4.Orange_Cap", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the team name, orange cap, and purple cap for each team
@app.get("/v1/soccer_2016/team_caps", operation_id="get_team_caps", summary="Retrieves the team name along with the top scorer (orange cap) and top wicket-taker (purple cap) for each team in the 2016 soccer season. The data is aggregated from various tables including Season, Match, Player_Match, Player, and Team.")
async def get_team_caps():
    cursor.execute("SELECT T5.Team_Name, T1.Orange_Cap, T1.Purple_Cap FROM Season AS T1 INNER JOIN Match AS T2 ON T1.Season_Id = T2.Season_Id INNER JOIN Player_Match AS T3 ON T2.Match_Id = T3.Match_Id INNER JOIN Player AS T4 ON T3.Player_Id = T4.Player_Id INNER JOIN Team AS T5 ON T3.Team_Id = T5.Team_Id GROUP BY T5.Team_Name, T1.Orange_Cap, T1.Purple_Cap")
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get player names from a specific country
@app.get("/v1/soccer_2016/players_by_country", operation_id="get_players_by_country", summary="Retrieves the names of soccer players from a specified country in the 2016 season. The operation filters players based on the provided country name and returns a list of matching player names.")
async def get_players_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Country AS T2 ON T1.Country_Name = T2.Country_Id WHERE T2.Country_Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of players with a specific batting hand
@app.get("/v1/soccer_2016/count_players_by_batting_hand", operation_id="get_count_players_by_batting_hand", summary="Retrieves the total number of players who use a specific batting hand. The batting hand is provided as an input parameter, and the operation returns the count of players who use that hand.")
async def get_count_players_by_batting_hand(batting_hand: str = Query(..., description="Batting hand of the player")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Batting_hand = ? THEN 1 ELSE 0 END) FROM Player AS T1 INNER JOIN Batting_Style AS T2 ON T1.Batting_hand = T2.Batting_Id", (batting_hand,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches not won by a specific type
@app.get("/v1/soccer_2016/count_matches_not_won_by_type", operation_id="get_count_matches_not_won_by_type", summary="Retrieves the total count of matches that were not won by a specific win type. The win type is defined by the input parameter, which could represent various win conditions such as runs.")
async def get_count_matches_not_won_by_type(win_type: str = Query(..., description="Type of win (e.g., runs)")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Win_Type != ? THEN 1 ELSE 0 END) FROM `Match` AS T1 INNER JOIN Win_By AS T2 ON T1.Win_Type = T2.Win_Id", (win_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get umpire names from a specific country
@app.get("/v1/soccer_2016/umpires_by_country", operation_id="get_umpires_by_country", summary="Retrieves the names of all umpires from a specified country. The operation filters umpires based on the provided country name and returns a list of their names.")
async def get_umpires_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Umpire_Name FROM Umpire AS T1 INNER JOIN Country AS T2 ON T1.Umpire_Country = T2.Country_Id WHERE T2.Country_Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get country names for players with a specific bowling skill
@app.get("/v1/soccer_2016/countries_by_bowling_skill", operation_id="get_countries_by_bowling_skill", summary="Retrieves the names of countries with players possessing a specified bowling skill. The operation filters players based on their bowling skill and returns the corresponding country names.")
async def get_countries_by_bowling_skill(bowling_skill: str = Query(..., description="Bowling skill of the player")):
    cursor.execute("SELECT T3.Country_Name FROM Bowling_Style AS T1 INNER JOIN Player AS T2 ON T1.Bowling_Id = T2.Bowling_skill INNER JOIN Country AS T3 ON T2.Country_Name = T3.Country_Id WHERE T1.Bowling_skill = ?", (bowling_skill,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get venue names for matches played by a specific team
@app.get("/v1/soccer_2016/venues_by_team", operation_id="get_venues_by_team", summary="Retrieves a list of unique venue names where a specified team has played matches. The team is identified by its name.")
async def get_venues_by_team(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT T1.Venue_Name FROM Venue AS T1 INNER JOIN Match AS T2 ON T1.Venue_Id = T2.Venue_Id INNER JOIN Team AS T3 ON T2.Team_1 = T3.Team_Id WHERE T3.Team_Name = ? GROUP BY T1.Venue_Name", (team_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of runs scored by a specific team while batting
@app.get("/v1/soccer_2016/count_runs_scored_by_team", operation_id="get_count_runs_scored_by_team", summary="Get the count of runs scored by a specific team while batting")
async def get_count_runs_scored_by_team(team_batting_1: int = Query(..., description="First team batting number"), team_batting_2: int = Query(..., description="Second team batting number"), team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT COUNT(T1.Runs_Scored) FROM Batsman_Scored AS T1 INNER JOIN Ball_by_Ball AS T2 ON T1.Match_Id = T2.Match_Id INNER JOIN Match AS T3 ON T2.Match_Id = T3.Match_Id INNER JOIN Team AS T4 ON T3.Team_1 = T4.Team_Id WHERE (T2.Team_Batting = ? OR T2.Team_Batting = ?) AND T4.Team_Name = ?", (team_batting_1, team_batting_2, team_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average extra runs for a specific extra type
@app.get("/v1/soccer_2016/average_extra_runs", operation_id="get_average_extra_runs", summary="Retrieves the average number of extra runs associated with a specific extra type in the 2016 soccer season.")
async def get_average_extra_runs(extra_name: str = Query(..., description="Name of the extra type")):
    cursor.execute("SELECT AVG(T1.Extra_Runs) FROM Extra_Runs AS T1 INNER JOIN Extra_Type AS T2 ON T1.Extra_Type_Id = T2.Extra_Id WHERE T2.Extra_Name = ?", (extra_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_extra_runs": []}
    return {"average_extra_runs": result[0]}

# Endpoint to get top players based on bowling skill
@app.get("/v1/soccer_2016/top_players_by_bowling_skill", operation_id="get_top_players_by_bowling_skill", summary="Retrieves a list of the top soccer players from the 2016 season, ranked by their bowling skill. The number of players returned is determined by the provided limit parameter.")
async def get_top_players_by_bowling_skill(limit: int = Query(..., description="Number of top players to retrieve")):
    cursor.execute("SELECT Player_Id FROM Player ORDER BY Bowling_skill DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"player_ids": []}
    return {"player_ids": [row[0] for row in result]}

# Endpoint to get the count of players born before a certain date and with bowling skill below a certain level
@app.get("/v1/soccer_2016/player_count_by_dob_and_bowling_skill", operation_id="get_player_count_by_dob_and_bowling_skill", summary="Retrieves the number of players who were born before a specified date and have a bowling skill level below a certain threshold.")
async def get_player_count_by_dob_and_bowling_skill(dob: str = Query(..., description="Date of birth in 'YYYY-MM-DD' format"), bowling_skill: int = Query(..., description="Bowling skill level")):
    cursor.execute("SELECT COUNT(*) FROM Player WHERE DOB < ? AND Bowling_skill < ?", (dob, bowling_skill))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the man of the series for seasons within a specific year range
@app.get("/v1/soccer_2016/man_of_the_series_by_year_range", operation_id="get_man_of_the_series_by_year_range", summary="Retrieve the man of the series for seasons that fall within the specified year range. The operation accepts a start year and an end year as input parameters, and returns the man of the series for each season within this range.")
async def get_man_of_the_series_by_year_range(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT Man_of_the_Series FROM Season WHERE ? < Season_Year < ?", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"man_of_the_series": []}
    return {"man_of_the_series": [row[0] for row in result]}

# Endpoint to get the total runs scored in a specific match and innings
@app.get("/v1/soccer_2016/total_runs_scored_by_match_and_innings", operation_id="get_total_runs_scored_by_match_and_innings", summary="Retrieves the cumulative number of runs scored in a particular match and innings. The operation requires the unique identifier of the match and the innings number as input parameters.")
async def get_total_runs_scored_by_match_and_innings(match_id: int = Query(..., description="Match ID"), innings_no: int = Query(..., description="Innings number")):
    cursor.execute("SELECT SUM(Runs_Scored) FROM Batsman_Scored WHERE Match_Id = ? AND Innings_No = ?", (match_id, innings_no))
    result = cursor.fetchone()
    if not result:
        return {"total_runs": []}
    return {"total_runs": result[0]}

# Endpoint to get the count of runs scored above a certain threshold in specific matches, innings, overs, and balls
@app.get("/v1/soccer_2016/runs_scored_above_threshold", operation_id="get_runs_scored_above_threshold", summary="Retrieves the total count of instances where the number of runs scored surpasses a specified threshold in a defined range of matches, innings, overs, and balls.")
async def get_runs_scored_above_threshold(runs_threshold: int = Query(..., description="Runs threshold"), start_match_id: int = Query(..., description="Start match ID"), end_match_id: int = Query(..., description="End match ID"), innings_no: int = Query(..., description="Innings number"), over_id: int = Query(..., description="Over ID"), ball_id: int = Query(..., description="Ball ID")):
    cursor.execute("SELECT SUM(CASE WHEN Runs_Scored > ? THEN 1 ELSE 0 END) FROM Batsman_Scored WHERE ? < Match_Id < ? AND Innings_No = ? AND Over_Id = ? AND Ball_Id = ?", (runs_threshold, start_match_id, end_match_id, innings_no, over_id, ball_id))
    result = cursor.fetchone()
    if not result:
        return {"runs_count": []}
    return {"runs_count": result[0]}

# Endpoint to get match details for a specific venue
@app.get("/v1/soccer_2016/match_details_by_venue", operation_id="get_match_details_by_venue", summary="Retrieves match details, including match ID and date, for a specific venue. The operation requires the venue name as input and returns the corresponding match details.")
async def get_match_details_by_venue(venue_name: str = Query(..., description="Name of the venue")):
    cursor.execute("SELECT T1.Match_Id, T1.Match_Date FROM `Match` AS T1 INNER JOIN Venue AS T2 ON T1.Venue_Id = T2.Venue_Id WHERE T2.Venue_Name = ?", (venue_name,))
    result = cursor.fetchall()
    if not result:
        return {"match_details": []}
    return {"match_details": [{"match_id": row[0], "match_date": row[1]} for row in result]}

# Endpoint to get the count of matches played at a specific venue within a date range
@app.get("/v1/soccer_2016/match_count_by_venue_and_date_range", operation_id="get_match_count_by_venue_and_date_range", summary="Retrieve the total number of matches played at a specified venue between a given start and end date. The operation filters matches based on the provided venue name and date range, then calculates the sum of matches that meet the criteria.")
async def get_match_count_by_venue_and_date_range(venue_name: str = Query(..., description="Name of the venue"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(CASE WHEN Venue_Name = ? THEN 1 ELSE 0 END) FROM `Match` AS T1 INNER JOIN Venue AS T2 ON T1.Venue_Id = T2.Venue_Id WHERE Match_Date BETWEEN ? AND ?", (venue_name, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"match_count": []}
    return {"match_count": result[0]}

# Endpoint to get venue and city details for a specific match
@app.get("/v1/soccer_2016/venue_and_city_by_match", operation_id="get_venue_and_city_by_match", summary="Retrieves the venue name and city name associated with a specific match. The match is identified by its unique match ID.")
async def get_venue_and_city_by_match(match_id: str = Query(..., description="Match ID")):
    cursor.execute("SELECT T2.Venue_Name, T3.City_Name FROM `Match` AS T1 INNER JOIN Venue AS T2 ON T1.Venue_Id = T2.Venue_Id INNER JOIN City AS T3 ON T2.City_Id = T3.City_Id WHERE T1.Match_Id = ?", (match_id,))
    result = cursor.fetchone()
    if not result:
        return {"venue_and_city": []}
    return {"venue_and_city": {"venue_name": result[0], "city_name": result[1]}}

# Endpoint to get toss details for a specific match
@app.get("/v1/soccer_2016/toss_details_by_match", operation_id="get_toss_details_by_match", summary="Retrieves the toss details for a specific soccer match in 2016. The details include the toss name, the decision made during the toss, and the winner of the toss. The match is identified by its unique match ID.")
async def get_toss_details_by_match(match_id: str = Query(..., description="Match ID")):
    cursor.execute("SELECT T2.Toss_Name, T1.Toss_Decide, T1.Toss_Winner FROM `Match` AS T1 INNER JOIN Toss_Decision AS T2 ON T1.Toss_Decide = T2.Toss_Id WHERE T1.Match_Id = ?", (match_id,))
    result = cursor.fetchone()
    if not result:
        return {"toss_details": []}
    return {"toss_details": {"toss_name": result[0], "toss_decide": result[1], "toss_winner": result[2]}}

# Endpoint to get the count of players born before a specific date in a given country
@app.get("/v1/soccer_2016/count_players_born_before_date_in_country", operation_id="get_count_players_born_before_date_in_country", summary="Get the count of players born before a specific date in a given country")
async def get_count_players_born_before_date_in_country(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(CASE WHEN T1.DOB < ? THEN 1 ELSE 0 END) FROM Player AS T1 INNER JOIN Country AS T2 ON T1.Country_Name = T2.Country_Id WHERE T2.Country_Name = ?", (date, country_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get player names based on bowling skill
@app.get("/v1/soccer_2016/player_names_by_bowling_skill", operation_id="get_player_names_by_bowling_skill", summary="Retrieves the names of soccer players who possess a specific bowling skill. The operation filters players based on the provided bowling skill and returns their names.")
async def get_player_names_by_bowling_skill(bowling_skill: str = Query(..., description="Bowling skill")):
    cursor.execute("SELECT T2.Player_Name FROM Bowling_Style AS T1 INNER JOIN Player AS T2 ON T1.Bowling_Id = T2.Bowling_skill WHERE T1.Bowling_skill = ?", (bowling_skill,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the latest match date and role description based on player's date of birth
@app.get("/v1/soccer_2016/latest_match_date_role_by_dob", operation_id="get_latest_match_date_role_by_dob", summary="Retrieves the most recent match date and corresponding role description for the player with the latest date of birth. This operation does not require any input parameters and returns the specified data based on the internal database query.")
async def get_latest_match_date_role_by_dob():
    cursor.execute("SELECT T1.Match_Date, T4.Role_Desc FROM `Match` AS T1 INNER JOIN Player_Match AS T2 ON T1.Match_Id = T2.Match_Id INNER JOIN Player AS T3 ON T2.Player_Id = T3.Player_Id INNER JOIN Rolee AS T4 ON T2.Role_Id = T4.Role_Id ORDER BY T3.DOB DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"match_date": [], "role_desc": []}
    return {"match_date": result[0], "role_desc": result[1]}

# Endpoint to get the count of matches within a date range where the man of the match is from a specific country
@app.get("/v1/soccer_2016/count_matches_by_date_range_and_country", operation_id="get_count_matches_by_date_range_and_country", summary="Retrieve the total number of matches that took place within a specified date range and where the man of the match hails from a given country. The date range is defined by the start and end dates provided in 'YYYY-MM-DD' format, and the country is identified by its name.")
async def get_count_matches_by_date_range_and_country(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Match_Date BETWEEN ? AND ? THEN 1 ELSE 0 END) FROM `Match` AS T1 INNER JOIN Player AS T2 ON T2.Player_Id = T1.Man_of_the_Match INNER JOIN Country AS T3 ON T3.Country_Id = T2.Country_Name WHERE T3.Country_Name = ?", (start_date, end_date, country_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get player names who were man of the series and orange cap winner
@app.get("/v1/soccer_2016/player_names_man_of_series_orange_cap", operation_id="get_player_names_man_of_series_orange_cap", summary="Retrieves the names of players who have been awarded both the 'Man of the Series' and 'Orange Cap' titles in the 2016 soccer season.")
async def get_player_names_man_of_series_orange_cap():
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Season AS T2 ON T1.Player_Id = T2.Man_of_the_Series = T2.Orange_Cap")
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get match dates where a specific team was the winner
@app.get("/v1/soccer_2016/match_dates_by_winning_team", operation_id="get_match_dates_by_winning_team", summary="Retrieves the dates of matches where the specified team was the winner. The operation requires the team's name as an input parameter to filter the results.")
async def get_match_dates_by_winning_team(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT T1.Match_Date FROM `Match` AS T1 INNER JOIN Team AS T2 ON T1.Match_Winner = T2.Team_Id WHERE T2.Team_Name = ?", (team_name,))
    result = cursor.fetchall()
    if not result:
        return {"match_dates": []}
    return {"match_dates": [row[0] for row in result]}

# Endpoint to get umpire names and IDs from a specific country
@app.get("/v1/soccer_2016/umpire_names_ids_by_country", operation_id="get_umpire_names_ids_by_country", summary="Retrieves the names and unique identifiers of umpires from a specified country. The operation filters umpires based on the provided country name and returns a list of matching records.")
async def get_umpire_names_ids_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Umpire_Name, T1.Umpire_Id FROM Umpire AS T1 INNER JOIN Country AS T2 ON T1.Umpire_Country = T2.Country_Id WHERE T2.Country_Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"umpire_names_ids": []}
    return {"umpire_names_ids": [{"name": row[0], "id": row[1]} for row in result]}

# Endpoint to get the ratio of runs scored when batting first to the number of times fielding first, for a specific match and over
@app.get("/v1/soccer_2016/runs_ratio_batting_fielding_by_match_over", operation_id="get_runs_ratio_batting_fielding_by_match_over", summary="Retrieve the ratio of runs scored by the team batting first to the number of times the opposing team fielded first, for a specific match and over. The input parameters include the match ID, match date, and the number of times the opposing team fielded first. The output is a grouped result by over, providing insights into the runs scored and fielding frequency.")
async def get_runs_ratio_batting_fielding_by_match_over(match_id: int = Query(..., description="Match ID"), match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format"), fielding_count: int = Query(..., description="Number of times fielding first")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.Toss_Name = 'bat' THEN T3.Runs_Scored ELSE NULL END) AS REAL) / SUM(CASE WHEN T1.Toss_Name = 'field' THEN 1 ELSE 0 END) FROM Toss_Decision AS T1 INNER JOIN Match AS T2 ON T1.Toss_Id = T2.Toss_Decide INNER JOIN Batsman_Scored AS T3 ON T2.Match_Id = T3.Match_Id WHERE T2.Match_Id = ? AND T2.Match_Date = ? GROUP BY T3.Over_Id HAVING COUNT(T1.Toss_Name = 'field') = ?", (match_id, match_date, fielding_count))
    result = cursor.fetchall()
    if not result:
        return {"runs_ratio": []}
    return {"runs_ratio": [row[0] for row in result]}

# Endpoint to get the match ID of the match with the highest win margin
@app.get("/v1/soccer_2016/match_id_highest_win_margin", operation_id="get_match_id_highest_win_margin", summary="Retrieve the match ID of the game with the highest win margin, up to a specified limit. This operation allows you to identify the match with the most significant victory difference, providing insights into the most dominant performance in the 2016 soccer season.")
async def get_match_id_highest_win_margin(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Match_Id FROM `Match` ORDER BY Match_Winner DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"match_id": []}
    return {"match_id": result[0]}

# Endpoint to get the most common date of birth among players
@app.get("/v1/soccer_2016/most_common_dob", operation_id="get_most_common_dob", summary="Retrieves the most frequently occurring date of birth among soccer players in the 2016 dataset. The operation returns the top N dates of birth, where N is specified by the 'limit' input parameter. This endpoint is useful for identifying trends or patterns in player birth dates.")
async def get_most_common_dob(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DOB FROM Player GROUP BY DOB ORDER BY COUNT(DOB) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"dob": []}
    return {"dob": result[0]}

# Endpoint to get the match date with the highest win margin
@app.get("/v1/soccer_2016/match_date_highest_win_margin", operation_id="get_match_date_highest_win_margin", summary="Retrieve the date of the match with the highest win margin, with the option to limit the number of results returned.")
async def get_match_date_highest_win_margin(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Match_Date FROM `Match` ORDER BY Win_Margin DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"match_date": []}
    return {"match_date": result[0]}

# Endpoint to get the season with the most matches
@app.get("/v1/soccer_2016/season_most_matches", operation_id="get_season_most_matches", summary="Retrieves the season with the highest number of matches, up to the specified limit.")
async def get_season_most_matches(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Season_Id FROM `Match` GROUP BY Season_Id ORDER BY COUNT(Match_Id) LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"season_id": []}
    return {"season_id": result[0]}

# Endpoint to get the count of matches for players who have been man of the match at least a specified number of times
@app.get("/v1/soccer_2016/match_count_man_of_the_match", operation_id="get_match_count_man_of_the_match", summary="Retrieves the total number of matches in which players have been awarded the man of the match title at least a specified number of times. This operation allows you to filter the results based on the minimum count of man of the match awards received by a player.")
async def get_match_count_man_of_the_match(min_count: int = Query(..., description="Minimum number of times a player has been man of the match")):
    cursor.execute("SELECT COUNT(Match_Id) FROM `Match` GROUP BY Man_of_the_Match HAVING COUNT(Match_Id) >= ?", (min_count,))
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": [row[0] for row in result]}

# Endpoint to get the player name who was man of the match in a specified season, ordered by match date
@app.get("/v1/soccer_2016/player_man_of_the_match_season", operation_id="get_player_man_of_the_match_season", summary="Retrieve the name of the player who was awarded the man of the match title in a specified season, sorted by match date in descending order. The operation allows you to limit the number of results returned. The season is identified by its unique ID.")
async def get_player_man_of_the_match_season(season_id: int = Query(..., description="Season ID"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Player_name FROM Player AS T1 INNER JOIN Match AS T2 ON T1.Player_Id = T2.Man_of_the_Match WHERE T2.Season_Id = ? ORDER BY T2.Match_Date DESC LIMIT ?", (season_id, limit))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the team name that won the match in a specified season, ordered by match date
@app.get("/v1/soccer_2016/team_match_winner_season", operation_id="get_team_match_winner_season", summary="Retrieves the name of the team that won matches in a specified season, sorted by match date. The number of results can be limited by providing a value for the limit parameter.")
async def get_team_match_winner_season(season_id: int = Query(..., description="Season ID"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Team_Name FROM team AS T1 INNER JOIN Match AS T2 ON T1.Team_Id = T2.Match_Winner WHERE T2.Season_Id = ? ORDER BY T2.Match_Date LIMIT ?", (season_id, limit))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the count of cities in a specified country
@app.get("/v1/soccer_2016/count_cities_in_country", operation_id="get_count_cities_in_country", summary="Retrieves the total number of cities in a specified country. The operation calculates the sum of cities based on the provided country name, which is used to filter the data from the cities and countries tables.")
async def get_count_cities_in_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Country_Name = ? THEN 1 ELSE 0 END) FROM City AS T1 INNER JOIN country AS T2 ON T1.Country_id = T2.Country_id", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of umpires from a specified country
@app.get("/v1/soccer_2016/umpires_from_country", operation_id="get_umpires_from_country", summary="Retrieves the names of umpires who are from the specified country. The operation filters umpires based on the provided country name and returns a list of their names.")
async def get_umpires_from_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Umpire_Name FROM Umpire AS T1 INNER JOIN country AS T2 ON T2.Country_Id = T1.Umpire_Country WHERE T2.Country_Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"umpire_names": []}
    return {"umpire_names": [row[0] for row in result]}

# Endpoint to get the names of players with a specified bowling style
@app.get("/v1/soccer_2016/players_with_bowling_style", operation_id="get_players_with_bowling_style", summary="Retrieves the names of soccer players from the 2016 season who possess a specific bowling style. The operation requires the bowling style as an input parameter to filter the results.")
async def get_players_with_bowling_style(bowling_style: str = Query(..., description="Bowling style")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Bowling_Style AS T2 ON T1.Bowling_skill = T2.Bowling_Id WHERE T2.Bowling_skill = ?", (bowling_style,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the sum of matches for a specific season and team
@app.get("/v1/soccer_2016/sum_matches_season_team", operation_id="get_sum_matches_season_team", summary="Retrieves the total number of matches played by a specific team during a given season. The operation requires the team's name and the season's ID as input parameters.")
async def get_sum_matches_season_team(season_id: int = Query(..., description="Season ID"), team_name: str = Query(..., description="Team name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Season_Id = ? THEN 1 ELSE 0 END) FROM `Match` AS T1 INNER JOIN Team AS T2 ON T1.Team_1 = T2.Team_Id OR T1.Team_2 = T2.Team_Id WHERE T2.Team_Name = ?", (season_id, team_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the country name of a specific umpire
@app.get("/v1/soccer_2016/umpire_country", operation_id="get_umpire_country", summary="Retrieves the country name of a specified umpire from the 2016 soccer season. The operation requires the umpire's name as input and returns the corresponding country name.")
async def get_umpire_country(umpire_name: str = Query(..., description="Umpire name")):
    cursor.execute("SELECT T2.Country_Name FROM Umpire AS T1 INNER JOIN country AS T2 ON T2.Country_Id = T1.Umpire_Country WHERE T1.Umpire_Name = ?", (umpire_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the venue name for a specific city
@app.get("/v1/soccer_2016/venue_by_city", operation_id="get_venue_by_city", summary="Get the venue name for a specific city")
async def get_venue_by_city(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.Venue_Name FROM Venue AS T1 INNER JOIN City AS T2 ON T1.City_Id = T2.City_Id WHERE T2.City_Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"venue_name": []}
    return {"venue_name": result[0]}

# Endpoint to get the country name of the youngest player
@app.get("/v1/soccer_2016/youngest_player_country", operation_id="get_youngest_player_country", summary="Retrieves the name of the country with the youngest player in the 2016 soccer dataset. The operation identifies the youngest player based on their date of birth and returns the corresponding country name.")
async def get_youngest_player_country():
    cursor.execute("SELECT T1.Country_Name FROM Country AS T1 INNER JOIN Player AS T2 ON T1.Country_Id = T2.Country_Name ORDER BY T2.DOB DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the player name of the first match winner in a specific season
@app.get("/v1/soccer_2016/first_match_winner_player", operation_id="get_first_match_winner_player", summary="Retrieves the name of the player who won the first match in a specified season. The season is identified by its unique ID.")
async def get_first_match_winner_player(season_id: int = Query(..., description="Season ID")):
    cursor.execute("SELECT T3.Player_Name FROM `Match` AS T1 INNER JOIN Player_Match AS T2 ON T1.Match_Winner = T2.Team_Id INNER JOIN Player AS T3 ON T2.Player_Id = T3.Player_Id WHERE T1.Season_Id = ? ORDER BY T1.Match_Date LIMIT 1", (season_id,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the player name with the highest age difference between season year and birth year
@app.get("/v1/soccer_2016/highest_age_difference_player", operation_id="get_highest_age_difference_player", summary="Retrieves the name of the player with the largest age difference between their birth year and the season year. The player's age is calculated by subtracting the birth year from the season year. The result is sorted in descending order, and the name of the player with the highest age difference is returned.")
async def get_highest_age_difference_player():
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Season AS T2 ON T1.Player_Id = T2.Purple_Cap ORDER BY T2.Season_Year - SUBSTR(T1.DOB, 1, 4) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the venue, city, and country of the most recent match
@app.get("/v1/soccer_2016/most_recent_match_venue", operation_id="get_most_recent_match_venue", summary="Retrieves the venue, city, and country details of the most recent soccer match in the 2016 season. The data is obtained by querying the match records and joining relevant tables to gather comprehensive location information.")
async def get_most_recent_match_venue():
    cursor.execute("SELECT T1.Venue_Name, T2.City_Name, T3.Country_Name FROM Venue AS T1 INNER JOIN City AS T2 ON T1.City_Id = T2.City_Id INNER JOIN Country AS T3 ON T2.Country_Id = T3.Country_Id INNER JOIN Match AS T4 ON T1.Venue_Id = T4.Venue_Id ORDER BY T4.Match_Date DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"venue_name": [], "city_name": [], "country_name": []}
    return {"venue_name": result[0], "city_name": result[1], "country_name": result[2]}

# Endpoint to get the sum of innings for a specific match
@app.get("/v1/soccer_2016/sum_innings_by_match", operation_id="get_sum_innings_by_match", summary="Retrieve the total number of innings played in a specific match. The response includes separate counts for the first and second innings.")
async def get_sum_innings_by_match(match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT SUM(CASE WHEN Innings_No = 1 THEN 1 ELSE 0 END) AS IN1 , SUM(CASE WHEN Innings_No = 2 THEN 1 ELSE 0 END) AS IN2 FROM Ball_by_Ball WHERE Match_Id = ?", (match_id,))
    result = cursor.fetchone()
    if not result:
        return {"innings_1": [], "innings_2": []}
    return {"innings_1": result[0], "innings_2": result[1]}

# Endpoint to get the ball details for a specific match and over
@app.get("/v1/soccer_2016/ball_details_by_match_over", operation_id="get_ball_details_by_match_over", summary="Retrieves the details of the ball, including the ball ID, runs scored, and innings number, for a specific match and over. The match and over are identified using their respective IDs.")
async def get_ball_details_by_match_over(match_id: int = Query(..., description="Match ID"), over_id: int = Query(..., description="Over ID")):
    cursor.execute("SELECT Ball_Id, Runs_Scored, Innings_No FROM Batsman_Scored WHERE Match_Id = ? AND Over_Id = ?", (match_id, over_id))
    result = cursor.fetchall()
    if not result:
        return {"ball_details": []}
    return {"ball_details": result}

# Endpoint to get the count of matches in a specific year
@app.get("/v1/soccer_2016/count_matches_by_year", operation_id="get_count_matches_by_year", summary="Retrieves the total number of matches that took place in a specified year. The year should be provided in the 'YYYY' format.")
async def get_count_matches_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(Match_Id) FROM `Match` WHERE Match_Date LIKE ?", (year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the age of a player based on their name
@app.get("/v1/soccer_2016/player_age", operation_id="get_player_age", summary="Retrieves the current age of a specified soccer player from the 2016 season. The age is calculated based on the player's year of birth.")
async def get_player_age(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT 2022 - SUBSTR(DOB, 1, 4) FROM Player WHERE Player_Name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the win ratio of toss winners in a specific year
@app.get("/v1/soccer_2016/toss_win_ratio", operation_id="get_toss_win_ratio", summary="Retrieves the ratio of matches won by the toss winner in a specific year. The year is provided in the 'YYYY%' format. The result is calculated by dividing the total number of matches won by the toss winner by the total number of matches played in the given year.")
async def get_toss_win_ratio(year: str = Query(..., description="Year in 'YYYY%' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Toss_Winner = Match_Winner THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN Match_Date LIKE ? THEN 1 ELSE 0 END) FROM `Match`", (year,))
    result = cursor.fetchone()
    if not result:
        return {"win_ratio": []}
    return {"win_ratio": result[0]}

# Endpoint to get the count of matches in a specific year with a win margin less than a specified value
@app.get("/v1/soccer_2016/match_count_year_win_margin", operation_id="get_match_count_year_win_margin", summary="Retrieves the total number of matches played in a given year where the winning margin was less than a specified value. The year is provided in 'YYYY%' format and the win margin is a numerical value.")
async def get_match_count_year_win_margin(year: str = Query(..., description="Year in 'YYYY%' format"), win_margin: int = Query(..., description="Win margin")):
    cursor.execute("SELECT COUNT(Match_Id) FROM `Match` WHERE Match_Date LIKE ? AND Win_Margin < ?", (year, win_margin))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get player names who played in a specific year and month
@app.get("/v1/soccer_2016/players_year_month", operation_id="get_players_year_month", summary="Retrieves the names of up to two players who participated in matches during a specified year and month. The year should be provided in 'YYYY' format, and the month in 'M' format.")
async def get_players_year_month(year: str = Query(..., description="Year in 'YYYY' format"), month: str = Query(..., description="Month in 'M' format")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Match AS T3 ON T2.Match_Id = T3.Match_Id WHERE SUBSTR(T3.Match_Date, 1, 4) = ? AND SUBSTR(T3.Match_Date, 7, 1) = ? LIMIT 2", (year, month))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get the total number of matches played by a specific player
@app.get("/v1/soccer_2016/total_matches_player", operation_id="get_total_matches_player", summary="Retrieves the total count of matches a particular player has participated in. The operation requires the player's name as input and returns the cumulative number of matches played by that player.")
async def get_total_matches_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Player_Name = ? THEN 1 ELSE 0 END) FROM Player_Match AS T1 INNER JOIN Player AS T2 ON T1.Player_Id = T2.Player_Id", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_matches": []}
    return {"total_matches": result[0]}

# Endpoint to get player names from a specific country born in a specific year
@app.get("/v1/soccer_2016/players_country_year", operation_id="get_players_country_year", summary="Retrieves the names of soccer players from a specific country who were born in a particular year. The operation filters players based on the provided country name and birth year, returning a list of matching player names.")
async def get_players_country_year(country_name: str = Query(..., description="Name of the country"), year: str = Query(..., description="Year in 'YYYY%' format")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Country AS T2 ON T1.Country_Name = T2.Country_Id WHERE T2.Country_Name = ? AND T1.DOB LIKE ?", (country_name, year))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get the win percentage of a specific team in a specific year
@app.get("/v1/soccer_2016/team_win_percentage", operation_id="get_team_win_percentage", summary="Retrieves the win percentage of a specific soccer team for a given year. The calculation is based on the team's match results, considering all matches in which the team participated. The year is specified in the 'YYYY%' format.")
async def get_team_win_percentage(team_name: str = Query(..., description="Name of the team"), year: str = Query(..., description="Year in 'YYYY%' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Match_Winner = T2.Team_Id THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.Match_Id) FROM `Match` AS T1 INNER JOIN Team AS T2 ON T1.Team_1 = T2.Team_Id OR T1.Team_2 = T2.Team_Id WHERE T2.Team_Name = ? AND T1.Match_Date LIKE ?", (team_name, year))
    result = cursor.fetchone()
    if not result:
        return {"win_percentage": []}
    return {"win_percentage": result[0]}

# Endpoint to get the ratio of left-hand batsmen to right-hand batsmen
@app.get("/v1/soccer_2016/batting_hand_ratio", operation_id="get_batting_hand_ratio", summary="Retrieves the ratio of left-handed to right-handed batsmen in the 2016 soccer season. The operation calculates this ratio based on the batting hand of each player, which is provided as input parameters. The result is a single value representing the proportion of left-handed batsmen to right-handed batsmen.")
async def get_batting_hand_ratio(left_hand: str = Query(..., description="Batting hand for left-hand batsmen"), right_hand: str = Query(..., description="Batting hand for right-hand batsmen")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Batting_hand = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.Batting_hand = ? THEN 1 ELSE 0 END) FROM Player AS T1 INNER JOIN Batting_Style AS T2 ON T1.Batting_hand = T2.Batting_Id", (left_hand, right_hand))
    result = cursor.fetchone()
    if not result:
        return {"batting_hand_ratio": []}
    return {"batting_hand_ratio": result[0]}

# Endpoint to get the oldest player and their country
@app.get("/v1/soccer_2016/oldest_player", operation_id="get_oldest_player", summary="Retrieves the name of the oldest soccer player and their country of origin from the 2016 dataset. The player's age is determined by their date of birth, and the result is limited to a single record.")
async def get_oldest_player():
    cursor.execute("SELECT T1.Player_Name, T2.Country_Name FROM Player AS T1 INNER JOIN Country AS T2 ON T1.Country_Name = T2.Country_Id ORDER BY T1.DOB LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"player": [], "country": []}
    return {"player": result[0], "country": result[1]}

# Endpoint to get the bowling skills of players from a specific country
@app.get("/v1/soccer_2016/bowling_skills_country", operation_id="get_bowling_skills_country", summary="Retrieves the bowling skills of soccer players from a specified country. The operation filters players based on their country of origin and returns their corresponding bowling skills.")
async def get_bowling_skills_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Bowling_skill FROM Bowling_Style AS T1 INNER JOIN Player AS T2 ON T1.Bowling_Id = T2.Bowling_skill INNER JOIN Country AS T3 ON T2.Country_Name = T3.Country_Id WHERE T3.Country_Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"bowling_skills": []}
    return {"bowling_skills": [row[0] for row in result]}

# Endpoint to get umpire details based on country name
@app.get("/v1/soccer_2016/umpire_details_by_country", operation_id="get_umpire_details", summary="Retrieves the details of umpires associated with a specific country. The operation returns a list of umpire IDs and names for the given country.")
async def get_umpire_details(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Umpire_Id, T1.Umpire_Name FROM Umpire AS T1 INNER JOIN Country AS T2 ON T1.Umpire_Country = T2.Country_Id WHERE T2.Country_Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"umpires": []}
    return {"umpires": result}

# Endpoint to get player names based on team name and role description
@app.get("/v1/soccer_2016/player_names_by_team_and_role", operation_id="get_player_names", summary="Retrieves the names of soccer players who belong to a specific team and have a particular role. The team is identified by its name, and the role is determined by its description. The operation returns a list of unique player names that meet the specified criteria.")
async def get_player_names(team_name: str = Query(..., description="Name of the team"), role_desc: str = Query(..., description="Description of the role")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Team AS T3 ON T2.Team_Id = T3.Team_Id INNER JOIN Rolee AS T4 ON T2.Role_Id = T4.Role_Id WHERE T3.Team_Name = ? AND T4.Role_Desc = ? GROUP BY T1.Player_Name", (team_name, role_desc))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": result}

# Endpoint to get the sum of matches won by a team in a specific year
@app.get("/v1/soccer_2016/sum_matches_won_by_team_year", operation_id="get_sum_matches_won", summary="Retrieves the total number of matches won by a specific team in a given year. The operation requires the team's name and the year as input parameters. The year should be provided in the 'YYYY' format.")
async def get_sum_matches_won(year: str = Query(..., description="Year in 'YYYY' format"), team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT SUM(CASE WHEN Match_Date LIKE ? THEN 1 ELSE 0 END) FROM `Match` AS T1 INNER JOIN Team AS T2 ON T1.Match_Winner = T2.Team_Id WHERE T2.Team_Name = ?", (year + '%', team_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get match IDs based on extra type name
@app.get("/v1/soccer_2016/match_ids_by_extra_type", operation_id="get_match_ids", summary="Retrieves a list of match IDs associated with a specific extra type. The extra type is identified by its name, which is provided as an input parameter. The operation performs a database query to fetch the relevant match IDs.")
async def get_match_ids(extra_name: str = Query(..., description="Name of the extra type")):
    cursor.execute("SELECT T1.Match_Id FROM Extra_Runs AS T1 INNER JOIN Extra_Type AS T2 ON T1.Extra_Type_Id = T2.Extra_Id WHERE T2.Extra_Name = ?", (extra_name,))
    result = cursor.fetchall()
    if not result:
        return {"match_ids": []}
    return {"match_ids": result}

# Endpoint to get team name based on match year and win type
@app.get("/v1/soccer_2016/team_name_by_year_win_type", operation_id="get_team_name", summary="Retrieves the name of the team that won a match in a specific year, based on the provided win type. The operation filters matches by the given year and win type, and returns the name of the first team that meets the criteria.")
async def get_team_name(year: str = Query(..., description="Year in 'YYYY' format"), win_type: str = Query(..., description="Type of win")):
    cursor.execute("SELECT T1.Team_Name FROM Team AS T1 INNER JOIN Match AS T2 ON T1.Team_Id = T2.Team_1 OR T1.Team_Id = T2.Team_2 INNER JOIN Win_By AS T3 ON T2.Win_Type = T3.Win_Id WHERE SUBSTR(T2.Match_Date, 1, 4) = ? AND T3.Win_Type = ? LIMIT 1", (year, win_type))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get wicket statistics based on innings number and out type
@app.get("/v1/soccer_2016/wicket_stats_by_innings_out_type", operation_id="get_wicket_stats", summary="Retrieves the average number of wickets taken per match and the total count of a specific out type, filtered by the innings number. The operation calculates the average by dividing the total number of wickets taken by the number of matches. It also sums the occurrences of the specified out type in the given innings.")
async def get_wicket_stats(innings_no: int = Query(..., description="Innings number"), out_name: str = Query(..., description="Name of the out type")):
    cursor.execute("SELECT CAST(COUNT(T1.Player_Out) AS REAL) / COUNT(T1.Match_Id), SUM(CASE WHEN T2.Out_Name = ? THEN 1 ELSE 0 END) FROM Wicket_Taken AS T1 INNER JOIN Out_Type AS T2 ON T1.Kind_Out = T2.Out_Id WHERE T1.Innings_No = ?", (out_name, innings_no))
    result = cursor.fetchone()
    if not result:
        return {"wicket_stats": []}
    return {"wicket_stats": result}

# Endpoint to get the count of matches based on innings number
@app.get("/v1/soccer_2016/match_count_by_innings", operation_id="get_match_count_by_innings", summary="Retrieves the total number of matches that occurred during a specific innings. The innings number is provided as an input parameter.")
async def get_match_count_by_innings(innings_no: int = Query(..., description="Innings number")):
    cursor.execute("SELECT COUNT(Match_Id) FROM Wicket_Taken WHERE innings_no = ?", (innings_no,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get country name based on city name
@app.get("/v1/soccer_2016/country_name_by_city", operation_id="get_country_name", summary="Retrieves the name of the country associated with the specified city. The operation uses the provided city name to search for a match in the city database and returns the corresponding country name.")
async def get_country_name(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.Country_Name FROM Country AS T1 INNER JOIN city AS T2 ON T1.Country_Id = T2.Country_Id WHERE city_name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the sum of matches won by a specific win type
@app.get("/v1/soccer_2016/sum_matches_by_win_type", operation_id="get_sum_matches_by_win_type", summary="Retrieves the total number of matches won under a specific win type. The win type is provided as an input parameter, which determines the matches to be counted.")
async def get_sum_matches_by_win_type(win_type: str = Query(..., description="Type of win")):
    cursor.execute("SELECT SUM(CASE WHEN T2.win_type = ? THEN 1 ELSE 0 END) FROM `Match` AS T1 INNER JOIN Win_By AS T2 ON T1.Win_Type = T2.Win_Id", (win_type,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get team names based on win margin and match date
@app.get("/v1/soccer_2016/team_names_by_win_margin_and_date", operation_id="get_team_names_by_win_margin_and_date", summary="Retrieves the names of soccer teams that achieved a specific win margin on a given match date in the 2016 season. The win margin and match date are provided as input parameters.")
async def get_team_names_by_win_margin_and_date(win_margin: int = Query(..., description="Win margin"), match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.Team_Name FROM Team AS T1 INNER JOIN Match AS T2 ON T1.Team_Id = T2.Team_1 WHERE T2.win_margin = ? AND match_date = ?", (win_margin, match_date))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get team names based on match ID and player name
@app.get("/v1/soccer_2016/team_names_by_match_id_and_player_name", operation_id="get_team_names_by_match_id_and_player_name", summary="Retrieves the team names associated with a specific match and player. The operation uses the provided match ID and player name to identify the relevant team(s).")
async def get_team_names_by_match_id_and_player_name(match_id: int = Query(..., description="Match ID"), player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT T1.Team_Name FROM Team AS T1 INNER JOIN Player_Match AS T2 ON T1.Team_Id = T2.Team_Id INNER JOIN Player AS T3 ON T2.Player_Id = T3.Player_Id WHERE T2.match_id = ? AND T3.player_name = ?", (match_id, player_name))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get the count of venues based on country and city name
@app.get("/v1/soccer_2016/venue_count_by_country_and_city", operation_id="get_venue_count_by_country_and_city", summary="Retrieves the total number of venues located in a specific city within a given country. The operation requires the country and city names as input parameters to filter the venues accordingly.")
async def get_venue_count_by_country_and_city(country_name: str = Query(..., description="Country name"), city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(T1.Venue_name) FROM Venue AS T1 INNER JOIN City AS T2 ON T1.City_Id = T2.City_Id INNER JOIN Country AS T3 ON T2.Country_Id = T3.Country_Id WHERE T3.country_name = ? AND T2.city_name = ?", (country_name, city_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of match winners based on team name and match date pattern
@app.get("/v1/soccer_2016/match_winner_count_by_team_and_date", operation_id="get_match_winner_count_by_team_and_date", summary="Get the count of match winners based on team name and match date pattern")
async def get_match_winner_count_by_team_and_date(team_name: str = Query(..., description="Team name"), match_date_pattern: str = Query(..., description="Match date pattern in 'YYYY%' format")):
    cursor.execute("SELECT COUNT(T1.Match_Winner) FROM `Match` AS T1 INNER JOIN Team AS T2 ON T2.Team_Id = T1.Team_1 OR T2.Team_Id = T1.Team_2 WHERE T2.team_name = ? AND T1.Match_Date LIKE ?", (team_name, match_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the match ID with the highest win margin for a specific team and match date pattern
@app.get("/v1/soccer_2016/highest_win_margin_match_id", operation_id="get_highest_win_margin_match_id", summary="Get the match ID with the highest win margin for a specific team and match date pattern")
async def get_highest_win_margin_match_id(team_name: str = Query(..., description="Team name"), match_date_pattern: str = Query(..., description="Match date pattern in 'YYYY%' format")):
    cursor.execute("SELECT T2.match_id FROM Team AS T1 INNER JOIN Match AS T2 ON T1.team_id = T2.match_winner WHERE T1.team_name = ? AND T2.match_date LIKE ? ORDER BY T2.win_margin DESC LIMIT 1", (team_name, match_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"match_id": []}
    return {"match_id": result[0]}

# Endpoint to get the count of matches played by a specific player in a specific role
@app.get("/v1/soccer_2016/match_count_by_player_and_role", operation_id="get_match_count_by_player_and_role", summary="Retrieves the total number of matches a specific player has played in a designated role. The operation requires the player's name and the role's ID as input parameters.")
async def get_match_count_by_player_and_role(player_name: str = Query(..., description="Player name"), role_id: int = Query(..., description="Role ID")):
    cursor.execute("SELECT COUNT(T1.Match_Id) FROM Player_Match AS T1 INNER JOIN Player AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Rolee AS T3 ON T1.Role_Id = T3.Role_Id WHERE T2.Player_Name = ? AND T3.Role_Id = ?", (player_name, role_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average win margin for matches played at a specific venue
@app.get("/v1/soccer_2016/average_win_margin_by_venue", operation_id="get_average_win_margin_by_venue", summary="Retrieves the average win margin for soccer matches played at a specified venue in the 2016 season. The venue is identified by its name, which is provided as an input parameter.")
async def get_average_win_margin_by_venue(venue_name: str = Query(..., description="Venue name")):
    cursor.execute("SELECT AVG(T1.win_margin) FROM Match AS T1 INNER JOIN Venue AS T2 ON T1.venue_id = T2.venue_id WHERE T2.venue_name = ?", (venue_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_win_margin": []}
    return {"average_win_margin": result[0]}

# Endpoint to get the team name of the losing team for a specific match
@app.get("/v1/soccer_2016/losing_team_name_by_match_id", operation_id="get_losing_team_name_by_match_id", summary="Retrieve the name of the team that lost a specific match. The match is identified by its unique match_id. The endpoint returns the team name of the losing team for the provided match_id.")
async def get_losing_team_name_by_match_id(match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT Team_Name FROM Team WHERE Team_Id = ( SELECT CASE WHEN Team_1 = Match_Winner THEN Team_2 ELSE Team_1 END FROM Match WHERE match_id = ? )", (match_id,))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the venue name for a specific match
@app.get("/v1/soccer_2016/venue_name_by_match_id", operation_id="get_venue_name_by_match_id", summary="Retrieves the name of the venue where a specific match was played. The match is identified by its unique match_id.")
async def get_venue_name_by_match_id(match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT T1.Venue_Name FROM Venue AS T1 INNER JOIN Match AS T2 ON T1.venue_id = T2.venue_id WHERE T2.match_id = ?", (match_id,))
    result = cursor.fetchone()
    if not result:
        return {"venue_name": []}
    return {"venue_name": result[0]}

# Endpoint to get the team name with the highest win margin as the away team
@app.get("/v1/soccer_2016/highest_win_margin_away_team", operation_id="get_highest_win_margin_away_team", summary="Retrieves the name of the team that achieved the highest win margin while playing as the away team in the 2016 soccer season.")
async def get_highest_win_margin_away_team():
    cursor.execute("SELECT T1.team_name FROM Team AS T1 INNER JOIN Match AS T2 ON T1.team_id = T2.team_2 ORDER BY T2.win_margin LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the percentage of matches won by a specific team in a given year
@app.get("/v1/soccer_2016/percentage_matches_won_by_team", operation_id="get_percentage_matches_won_by_team", summary="Retrieves the percentage of matches won by a specific team in a given year. The calculation is based on the total number of matches played by the team in the specified year. The year is provided as a four-digit input parameter.")
async def get_percentage_matches_won_by_team(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Match_Winner = 7 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.Match_Winner) FROM Team AS T1 INNER JOIN Match AS T2 ON T1.Team_Id = T2.Match_Winner WHERE T2.Match_Date LIKE ?", (year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in the number of roles with Role_Id 1 and roles with Role_Id greater than 1 for a specific player
@app.get("/v1/soccer_2016/role_difference_by_player", operation_id="get_role_difference_by_player", summary="Retrieves the difference in the number of roles between the primary role (Role_Id 1) and all other roles for a specific player in the 2016 soccer season. The input parameter is the player's name.")
async def get_role_difference_by_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT SUM(CASE WHEN T3.Role_Id = 1 THEN 1 ELSE 0 END) - SUM(CASE WHEN T3.Role_Id > 1 THEN 1 ELSE 0 END) FROM Player_Match AS T1 INNER JOIN Player AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Rolee AS T3 ON T1.Role_Id = T3.Role_Id WHERE T2.Player_Name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of players with bowling skill greater than a specified value
@app.get("/v1/soccer_2016/count_players_by_bowling_skill", operation_id="get_count_players_by_bowling_skill", summary="Retrieves the total number of players who possess a bowling skill level surpassing the provided threshold. The input parameter determines the minimum skill level for the count.")
async def get_count_players_by_bowling_skill(bowling_skill: int = Query(..., description="Bowling skill value")):
    cursor.execute("SELECT COUNT(Player_Name) FROM Player WHERE Bowling_skill > ?", (bowling_skill,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players born in a specific year
@app.get("/v1/soccer_2016/count_players_by_dob", operation_id="get_count_players_by_dob", summary="Retrieves the total number of soccer players who were born in a specified year. The year should be provided in the 'YYYY' format.")
async def get_count_players_by_dob(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(Player_Name) FROM Player WHERE DOB LIKE ?", (year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players born in a specific decade with a specific bowling skill
@app.get("/v1/soccer_2016/count_players_by_dob_and_bowling_skill", operation_id="get_count_players_by_dob_and_bowling_skill", summary="Retrieves the total number of soccer players born in a specific decade who possess a particular bowling skill. The decade is specified in 'YYYY' format, and the bowling skill is indicated by a predefined value.")
async def get_count_players_by_dob_and_bowling_skill(decade: str = Query(..., description="Decade in 'YYYY' format"), bowling_skill: int = Query(..., description="Bowling skill value")):
    cursor.execute("SELECT COUNT(Player_Name) FROM Player WHERE DOB LIKE ? AND Bowling_skill = ?", (decade + '%', bowling_skill))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches with a winner in a specific year
@app.get("/v1/soccer_2016/count_matches_with_winner_by_year", operation_id="get_count_matches_with_winner_by_year", summary="Retrieves the total number of matches that have a declared winner in a specified year. The year should be provided in the 'YYYY' format.")
async def get_count_matches_with_winner_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(Match_Id) FROM Match WHERE Match_Date LIKE ? AND Match_Winner IS NOT NULL", (year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country ID of a specific city
@app.get("/v1/soccer_2016/country_id_by_city", operation_id="get_country_id_by_city", summary="Retrieves the unique identifier of the country associated with the specified city in the 2016 soccer dataset. The operation requires the name of the city as input and returns the corresponding country ID.")
async def get_country_id_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT Country_id FROM City WHERE City_Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_id": []}
    return {"country_id": result[0]}

# Endpoint to get the age of a player in the year 2008
@app.get("/v1/soccer_2016/player_age_in_2008", operation_id="get_player_age_in_2008", summary="Retrieves the age of a specified player in the year 2008. The operation calculates the age based on the player's date of birth (DOB) and returns the result.")
async def get_player_age_in_2008(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT 2008 - strftime('%Y', DOB) FROM Player WHERE Player_Name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get player names based on batting hand
@app.get("/v1/soccer_2016/player_names_by_batting_hand", operation_id="get_player_names_by_batting_hand", summary="Retrieves the names of soccer players who use the specified batting hand. The batting hand is a parameter that filters the results to include only players who use that particular hand for batting.")
async def get_player_names_by_batting_hand(batting_hand: str = Query(..., description="Batting hand of the player")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Batting_Style AS T2 ON T1.Batting_hand = T2.Batting_Id WHERE T2.Batting_hand = ?", (batting_hand,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get player names based on country name
@app.get("/v1/soccer_2016/player_names_by_country", operation_id="get_player_names_by_country", summary="Retrieves the names of all soccer players from a specified country in the 2016 dataset. The operation filters the player list based on the provided country name.")
async def get_player_names_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Country AS T2 ON T1.Country_Name = T2.Country_ID WHERE T2.Country_Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get venue names based on city name
@app.get("/v1/soccer_2016/venue_names_by_city", operation_id="get_venue_names_by_city", summary="Retrieves the names of all venues located in a specified city. The operation requires the name of the city as an input parameter.")
async def get_venue_names_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.Venue_Name FROM Venue AS T1 INNER JOIN City AS T2 ON T1.City_ID = T2.City_ID WHERE T2.City_Name = ?", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"venue_names": []}
    return {"venue_names": [row[0] for row in result]}

# Endpoint to get player names based on season year
@app.get("/v1/soccer_2016/player_names_by_season_year", operation_id="get_player_names_by_season_year", summary="Retrieves the names of players who were awarded 'Man of the Match' during a specified season. The operation filters players based on the provided season year and returns a distinct list of player names.")
async def get_player_names_by_season_year(season_year: int = Query(..., description="Year of the season")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Match AS T2 ON T1.Player_Id = T2.Man_of_the_Match INNER JOIN Player_Match AS T3 ON T3.Player_Id = T1.Player_Id INNER JOIN Season AS T4 ON T2.Season_Id = T4.Season_Id WHERE T4.Season_Year = ? GROUP BY T1.Player_Name", (season_year,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get player names based on runs scored
@app.get("/v1/soccer_2016/player_names_by_runs_scored", operation_id="get_player_names_by_runs_scored", summary="Retrieve the names of soccer players who scored fewer than the specified number of runs. This operation filters players based on their runs scored in matches and returns a list of unique player names.")
async def get_player_names_by_runs_scored(runs_scored: int = Query(..., description="Maximum runs scored")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Batsman_Scored AS T3 ON T2.Match_ID = T3.Match_ID WHERE T3.Runs_Scored < ? GROUP BY T1.Player_Name", (runs_scored,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get role descriptions based on player name
@app.get("/v1/soccer_2016/role_descriptions_by_player_name", operation_id="get_role_descriptions_by_player_name", summary="Retrieves the descriptions of roles associated with a specific soccer player from the 2016 season. The operation filters the roles based on the provided player's name and returns a distinct list of role descriptions.")
async def get_role_descriptions_by_player_name(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T3.Role_Desc FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Rolee AS T3 ON T2.Role_Id = T3.Role_Id WHERE T1.Player_Name = ? GROUP BY T3.Role_Desc", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"role_descriptions": []}
    return {"role_descriptions": [row[0] for row in result]}

# Endpoint to get player names based on role description
@app.get("/v1/soccer_2016/player_names_by_role_description", operation_id="get_player_names_by_role_description", summary="Retrieves a list of player names who have a specific role, as described by the provided role description. The operation groups the results by player name to ensure uniqueness.")
async def get_player_names_by_role_description(role_desc: str = Query(..., description="Role description")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Rolee AS T3 ON T2.Role_Id = T3.Role_Id WHERE T3.Role_Desc = ? GROUP BY T1.Player_Name", (role_desc,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get player names based on team ID
@app.get("/v1/soccer_2016/player_names_by_team_id", operation_id="get_player_names_by_team_id", summary="Retrieves the names of all players associated with a specific team. The team is identified by its unique ID. The response includes a list of distinct player names who have participated in matches for the team.")
async def get_player_names_by_team_id(team_id: int = Query(..., description="ID of the team")):
    cursor.execute("SELECT T1.Player_Name FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Team AS T3 ON T2.Team_Id = T3.Team_Id WHERE T3.Team_Id = ? GROUP BY T1.Player_Name", (team_id,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the count of players based on role description and match date
@app.get("/v1/soccer_2016/player_count_by_role_and_date", operation_id="get_player_count_by_role_and_date", summary="Retrieves the total number of players who have a specified role and played in matches during a given year. The role is identified by its description, and the year is determined by the match date.")
async def get_player_count_by_role_and_date(role_desc: str = Query(..., description="Role description"), match_date: str = Query(..., description="Match date in 'YYYY%' format")):
    cursor.execute("SELECT COUNT(T1.Player_Id) FROM Player_Match AS T1 INNER JOIN Match AS T2 ON T1.Match_Id = T2.Match_Id INNER JOIN Rolee AS T3 ON T1.Role_Id = T3.Role_Id WHERE T3.Role_Desc = ? AND T2.Match_Date LIKE ?", (role_desc, match_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the team names for a specific player in a given season
@app.get("/v1/soccer_2016/team_names_by_player_season", operation_id="get_team_names_by_player_season", summary="Retrieves the names of teams that a specific player has played for during a given season. The operation requires the player's name and the season year as input parameters.")
async def get_team_names_by_player_season(season_year: int = Query(..., description="Season year"), player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT T5.Team_Name FROM Player AS T1 INNER JOIN Match AS T2 ON T1.Player_Id = T2.Man_of_the_Match INNER JOIN Player_Match AS T3 ON T3.Player_Id = T1.Player_Id INNER JOIN Season AS T4 ON T2.Season_Id = T4.Season_Id INNER JOIN Team AS T5 ON T3.Team_Id = T5.Team_Id WHERE T4.Season_Year = ? AND T1.Player_Name = ? GROUP BY T5.Team_Name", (season_year, player_name))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get the win type for a specific match
@app.get("/v1/soccer_2016/win_type_by_match_id", operation_id="get_win_type_by_match_id", summary="Retrieves the type of win for a specific soccer match in the 2016 season. The match is identified by its unique ID, which is provided as an input parameter.")
async def get_win_type_by_match_id(match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT T2.Win_Type FROM Match AS T1 INNER JOIN Win_By AS T2 ON T1.Win_Type = T2.Win_Id WHERE T1.Match_Id = ?", (match_id,))
    result = cursor.fetchone()
    if not result:
        return {"win_type": []}
    return {"win_type": result[0]}

# Endpoint to get the country name for a specific player
@app.get("/v1/soccer_2016/country_name_by_player_name", operation_id="get_country_name_by_player_name", summary="Retrieves the country name associated with a given player in the 2016 soccer dataset. The operation requires the player's name as input and returns the corresponding country name.")
async def get_country_name_by_player_name(player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT T2.Country_Name FROM Player AS T1 INNER JOIN Country AS T2 ON T1.Country_Name = T2.Country_ID WHERE T1.Player_Name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the count of players with a specific bowling style
@app.get("/v1/soccer_2016/player_count_by_bowling_style", operation_id="get_player_count_by_bowling_style", summary="Retrieves the total number of players who have a specific bowling style. The operation requires the bowling style as an input parameter to filter the count of players accordingly.")
async def get_player_count_by_bowling_style(bowling_style: str = Query(..., description="Bowling style")):
    cursor.execute("SELECT COUNT(T1.Player_Id) FROM Player AS T1 INNER JOIN Bowling_Style AS T2 ON T1.Bowling_skill = T2.Bowling_Id WHERE T2.Bowling_skill = ?", (bowling_style,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the outcome type for a specific match
@app.get("/v1/soccer_2016/outcome_type_by_match_id", operation_id="get_outcome_type_by_match_id", summary="Retrieves the outcome type associated with a specific soccer match in the 2016 dataset. The outcome type is determined by querying the match and outcome tables using the provided match ID.")
async def get_outcome_type_by_match_id(match_id: str = Query(..., description="Match ID")):
    cursor.execute("SELECT T2.Outcome_Type FROM Match AS T1 INNER JOIN Outcome AS T2 ON T1.Outcome_type = T2.Outcome_Id WHERE T1.Match_Id = ?", (match_id,))
    result = cursor.fetchone()
    if not result:
        return {"outcome_type": []}
    return {"outcome_type": result[0]}

# Endpoint to get the city name of the youngest player
@app.get("/v1/soccer_2016/city_name_of_youngest_player", operation_id="get_city_name_of_youngest_player", summary="Retrieves the city name of the youngest player in the 2016 soccer dataset. The operation fetches the city name by joining the Player, Country, and City tables based on their respective IDs, and then orders the results by the player's date of birth (DOB) in ascending order. The city name of the youngest player is returned as the result.")
async def get_city_name_of_youngest_player():
    cursor.execute("SELECT T3.City_Name FROM Player AS T1 INNER JOIN Country AS T2 ON T1.Country_Name = T2.Country_Id INNER JOIN City AS T3 ON T2.Country_Id = T3.Country_Id ORDER BY T1.DOB LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the count of distinct matches won by a specific team in a given season
@app.get("/v1/soccer_2016/match_count_by_team_season", operation_id="get_match_count_by_team_season", summary="Retrieve the number of unique matches won by a specified team during a particular season. The operation requires the team's name and the year of the season as input parameters.")
async def get_match_count_by_team_season(team_name: str = Query(..., description="Team name"), season_year: int = Query(..., description="Season year")):
    cursor.execute("SELECT COUNT(DISTINCT T2.Match_Id) FROM Team AS T1 INNER JOIN Match AS T2 ON T1.team_id = T2.match_winner INNER JOIN Player_Match AS T3 ON T1.Team_Id = T3.Team_Id INNER JOIN Season AS T4 ON T2.Season_Id = T4.Season_Id WHERE T1.Team_Name = ? AND T4.Season_Year = ?", (team_name, season_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of seasons a specific team has won matches
@app.get("/v1/soccer_2016/season_count_by_team", operation_id="get_season_count_by_team", summary="Retrieve the number of seasons in which a specific soccer team has won matches. The operation filters matches by the provided team name and counts the distinct seasons in which the team has emerged victorious.")
async def get_season_count_by_team(team_name: str = Query(..., description="Team name")):
    cursor.execute("SELECT COUNT(T.Season_Year) FROM ( SELECT T4.Season_Year FROM Team AS T1 INNER JOIN Match AS T2 ON T1.team_id = T2.match_winner INNER JOIN Player_Match AS T3 ON T1.Team_Id = T3.Team_Id INNER JOIN Season AS T4 ON T2.Season_Id = T4.Season_Id WHERE T1.Team_Name = ? GROUP BY T4.Season_Year ) T", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the date of birth and role description for a specific player
@app.get("/v1/soccer_2016/player_dob_role_by_name", operation_id="get_player_dob_role_by_name", summary="Retrieves the date of birth and role description for a specific soccer player from the 2016 season. The operation requires the player's name as input and returns the corresponding date of birth and role description.")
async def get_player_dob_role_by_name(player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT T1.DOB, T3.Role_Desc FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Rolee AS T3 ON T2.Role_Id = T3.Role_Id WHERE T1.Player_Name = ? GROUP BY T1.DOB, T3.Role_Desc", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"player_info": []}
    return {"player_info": [{"dob": row[0], "role_desc": row[1]} for row in result]}

# Endpoint to get the count of man of the match awards for a specific player
@app.get("/v1/soccer_2016/man_of_the_match_count_by_player", operation_id="get_man_of_the_match_count_by_player", summary="Retrieves the total number of 'Man of the Match' awards received by a specific soccer player in the 2016 season. The player's name is provided as an input parameter.")
async def get_man_of_the_match_count_by_player(player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT COUNT(T2.Man_of_the_Match) FROM Player AS T1 INNER JOIN Match AS T2 ON T1.Player_Id = T2.Man_of_the_Match INNER JOIN Player_Match AS T3 ON T3.Player_Id = T1.Player_Id WHERE T1.Player_Name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches won by a specific team in a specific year
@app.get("/v1/soccer_2016/count_matches_won_by_team_year", operation_id="get_count_matches_won_by_team_year", summary="Retrieve the total number of matches won by a specific soccer team in a given year. The operation requires the team's name and the year of the matches as input parameters. The year should be provided in the 'YYYY%' format.")
async def get_count_matches_won_by_team_year(team_name: str = Query(..., description="Name of the team"), match_year: str = Query(..., description="Year of the match in 'YYYY%' format")):
    cursor.execute("SELECT COUNT(T.Match_Id) FROM ( SELECT T2.Match_Id FROM Team AS T1 INNER JOIN Match AS T2 ON T1.team_id = T2.match_winner INNER JOIN Player_Match AS T3 ON T1.Team_Id = T3.Team_Id WHERE T1.Team_Name = ? AND T2.Match_Date LIKE ? GROUP BY T2.Match_Id ) T", (team_name, match_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the team names that won a specific match
@app.get("/v1/soccer_2016/team_names_won_specific_match", operation_id="get_team_names_won_specific_match", summary="Retrieves the names of teams that won a specific match, identified by its unique match ID. The operation joins multiple tables to gather comprehensive data about the winning teams, including their names and the type of win. The results are grouped by team name for clarity and conciseness.")
async def get_team_names_won_specific_match(match_id: str = Query(..., description="ID of the match")):
    cursor.execute("SELECT T1.Team_Name FROM Team AS T1 INNER JOIN Match AS T2 ON T1.team_id = T2.match_winner INNER JOIN Player_Match AS T3 ON T1.Team_Id = T3.Team_Id INNER JOIN Win_By AS T4 ON T2.Win_Type = T4.Win_Id WHERE T2.Match_Id = ? GROUP BY T1.Team_Name", (match_id,))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get the count of matches won by a specific win type
@app.get("/v1/soccer_2016/count_matches_won_by_win_type", operation_id="get_count_matches_won_by_win_type", summary="Retrieves the total number of matches won by a specific win type. The win type is provided as an input parameter, allowing the user to filter the results accordingly. This operation is useful for analyzing the performance of teams or players based on different win types.")
async def get_count_matches_won_by_win_type(win_type: str = Query(..., description="Type of win (e.g., 'wickets')")):
    cursor.execute("SELECT COUNT(T1.Match_Id) FROM Match AS T1 INNER JOIN Win_By AS T2 ON T1.Win_Type = T2.Win_Id WHERE T2.Win_type = ?", (win_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the role description of a player in a specific season
@app.get("/v1/soccer_2016/player_role_description_season", operation_id="get_player_role_description_season", summary="Retrieves the role description of a specific player for a given season. The operation requires the player's name and the year of the season as input parameters. It returns the role description of the player based on their performance in the specified season.")
async def get_player_role_description_season(player_name: str = Query(..., description="Name of the player"), season_year: int = Query(..., description="Year of the season")):
    cursor.execute("SELECT T4.Role_Desc FROM Player AS T1 INNER JOIN Player_Match AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Match AS T3 ON T2.Match_Id = T3.Match_Id INNER JOIN Rolee AS T4 ON T2.Role_Id = T4.Role_Id INNER JOIN Season AS T5 ON T3.Season_Id = T5.Season_Id WHERE T1.Player_name = ? AND T5.Season_Year = ?", (player_name, season_year))
    result = cursor.fetchall()
    if not result:
        return {"role_descriptions": []}
    return {"role_descriptions": [row[0] for row in result]}

# Endpoint to get the player name if they were man of the match more than a specified number of times in a specific season
@app.get("/v1/soccer_2016/player_man_of_the_match_count_season", operation_id="get_player_man_of_the_match_count_season", summary="Retrieve the name of a player who was awarded the man of the match title more than a specified number of times in a given season. This operation requires the count of man of the match awards and the year of the season as input parameters.")
async def get_player_man_of_the_match_count_season(man_of_the_match_count: int = Query(..., description="Number of times the player was man of the match"), season_year: int = Query(..., description="Year of the season")):
    cursor.execute("SELECT CASE WHEN COUNT(T2.Man_of_the_Match) > ? THEN T1.Player_Name ELSE 0 END FROM Player AS T1 INNER JOIN Match AS T2 ON T1.Player_Id = T2.Man_of_the_Match INNER JOIN Player_Match AS T3 ON T3.Player_Id = T1.Player_Id INNER JOIN Season AS T4 ON T2.Season_Id = T4.Season_Id WHERE T4.Season_Year = ?", (man_of_the_match_count, season_year))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the proportion of players from a specific country born between specific years
@app.get("/v1/soccer_2016/proportion_players_country_birth_years", operation_id="get_proportion_players_country_birth_years", summary="Retrieve the proportion of players from a specified country who were born within a given range of years. The calculation is based on the total number of players in the database.")
async def get_proportion_players_country_birth_years(country_name: str = Query(..., description="Name of the country"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Country_Name = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(T1.Player_Id) FROM Player AS T1 INNER JOIN Country AS T2 ON T1.Country_Name = T2.Country_ID WHERE strftime('%Y', T1.DOB) BETWEEN ? AND ?", (country_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

# Endpoint to get the percentage of matches won by a specific win type
@app.get("/v1/soccer_2016/percentage_matches_win_type", operation_id="get_percentage_matches_win_type", summary="Retrieves the percentage of matches won by a specific win type. The win type is identified by its unique ID. The calculation is based on the total number of matches won by the specified win type and the total number of matches played.")
async def get_percentage_matches_win_type(win_type: int = Query(..., description="Win type ID (e.g., 1 for wickets)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.win_type = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.Win_Type) FROM Match AS T1 INNER JOIN Win_By AS T2 ON T1.Win_Type = T2.Win_Id", (win_type,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of matches with a specific win margin
@app.get("/v1/soccer_2016/count_matches_win_margin", operation_id="get_count_matches_win_margin", summary="Retrieves the total number of matches in the 2016 soccer season that ended with a specified win margin. The win margin is the difference in goals scored between the winning and losing teams.")
async def get_count_matches_win_margin(win_margin: int = Query(..., description="Win margin")):
    cursor.execute("SELECT COUNT(Match_Id) FROM Match WHERE win_margin = ?", (win_margin,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players born between specific years
@app.get("/v1/soccer_2016/count_players_birth_years", operation_id="get_count_players_birth_years", summary="Retrieves the total number of soccer players born between the specified start and end years. The start and end years are inclusive and should be provided in 'YYYY' format.")
async def get_count_players_birth_years(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(Player_Id) FROM Player WHERE strftime('%Y', DOB) BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get team names based on match ID
@app.get("/v1/soccer_2016/team_names_by_match_id", operation_id="get_team_names_by_match_id", summary="Retrieves the names of the teams that won the match with the specified ID. The match ID is used to identify the winning team, which is then used to retrieve the team name from the Team table. The win type is also considered to ensure accurate results.")
async def get_team_names_by_match_id(match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT T1.Team_Name FROM Team AS T1 INNER JOIN Match AS T2 ON T1.team_id = T2.match_winner INNER JOIN Win_By AS T3 ON T2.win_type = T3.win_id WHERE T2.Match_Id = ?", (match_id,))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get match ID based on venue name
@app.get("/v1/soccer_2016/match_id_by_venue_name", operation_id="get_match_id_by_venue_name", summary="Retrieves the ID of the match with the highest win margin that took place at the specified venue. The venue is identified by its name.")
async def get_match_id_by_venue_name(venue_name: str = Query(..., description="Venue name")):
    cursor.execute("SELECT T2.Match_Id FROM Venue AS T1 INNER JOIN Match AS T2 ON T1.venue_id = T2.venue_id WHERE T1.Venue_Name = ? ORDER BY T2.Win_Margin DESC LIMIT 1", (venue_name,))
    result = cursor.fetchone()
    if not result:
        return {"match_id": []}
    return {"match_id": result[0]}

# Endpoint to get player names based on role description
@app.get("/v1/soccer_2016/player_names_by_role", operation_id="get_player_names_by_role", summary="Retrieves a list of player names who have played a specific role in the 2016 soccer season. The role is determined by the provided role description.")
async def get_player_names_by_role(role_desc: str = Query(..., description="Role description")):
    cursor.execute("SELECT T2.Player_Name FROM Player_Match AS T1 INNER JOIN Player AS T2 ON T1.Player_Id = T2.Player_Id INNER JOIN Rolee AS T3 ON T1.Role_Id = T3.Role_Id WHERE T3.Role_Desc = ? GROUP BY T2.Player_Name", (role_desc,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get venue name and team name based on match ID
@app.get("/v1/soccer_2016/venue_and_team_by_match_id", operation_id="get_venue_and_team_by_match_id", summary="Retrieves the name of the venue and the name of the winning team for a specific soccer match in the 2016 season. The match is identified by its unique ID.")
async def get_venue_and_team_by_match_id(match_id: int = Query(..., description="Match ID")):
    cursor.execute("SELECT T1.Venue_Name, T3.Team_Name FROM Venue AS T1 INNER JOIN Match AS T2 ON T1.venue_id = T2.venue_id INNER JOIN Team AS T3 ON T2.match_winner = T3.Team_Id WHERE T2.Match_Id = ?", (match_id,))
    result = cursor.fetchall()
    if not result:
        return {"venue_and_team": []}
    return {"venue_and_team": [{"venue_name": row[0], "team_name": row[1]} for row in result]}

# Endpoint to get the percentage of wins by wickets for a given team
@app.get("/v1/soccer_2016/win_percentage_by_wickets", operation_id="get_win_percentage_by_wickets", summary="Retrieves the percentage of matches won by a specific team through wickets. The calculation is based on the total number of matches won by the team and the number of those wins that were achieved by wickets.")
async def get_win_percentage_by_wickets(team_name: str = Query(..., description="Team name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.Win_Type = 'wickets' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.Win_Type) FROM Team AS T1 INNER JOIN Match AS T2 ON T1.Team_Id = T2.Match_Winner INNER JOIN Win_By AS T3 ON T2.Win_Type = T3.Win_Id WHERE T1.Team_Name = ?", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"win_percentage": []}
    return {"win_percentage": result[0]}

api_calls = [
    "/v1/soccer_2016/player_count_by_year?year=1985",
    "/v1/soccer_2016/match_count_by_year_month?year=2008&month=5",
    "/v1/soccer_2016/match_count_by_man_of_the_match?man_of_the_match=41",
    "/v1/soccer_2016/match_ids_by_year?year=2008",
    "/v1/soccer_2016/player_count_by_country?country_name=Australia",
    "/v1/soccer_2016/oldest_player_country",
    "/v1/soccer_2016/bowling_skill_by_player?player_name=SC%20Ganguly",
    "/v1/soccer_2016/player_count_by_batting_hand_and_year?year=1985&batting_hand=Right-hand%20bat",
    "/v1/soccer_2016/player_names_by_country_and_batting_hand?country_name=Australia&batting_hand=Right-hand%20bat",
    "/v1/soccer_2016/bowling_skills_by_country?country_name=Australia",
    "/v1/soccer_2016/min_dob_by_bowling_skill?bowling_skill=Legbreak",
    "/v1/soccer_2016/most_common_bowling_skill",
    "/v1/soccer_2016/man_of_the_match_by_date?match_date=2008-04-18",
    "/v1/soccer_2016/count_matches_by_role_and_player?role_desc=Captain&player_name=SC%20Ganguly",
    "/v1/soccer_2016/role_id_by_player_and_match_date?player_name=SC%20Ganguly&match_date=2008-04-18",
    "/v1/soccer_2016/max_win_margin_by_player?player_name=SC%20Ganguly",
    "/v1/soccer_2016/avg_win_margin_by_player?player_name=SC%20Ganguly",
    "/v1/soccer_2016/percentage_batting_hand_by_year?batting_hand=Right-hand%20bat&year=1985",
    "/v1/soccer_2016/youngest_player",
    "/v1/soccer_2016/toss_wins_by_team?team_name=Sunrisers%20Hyderabad",
    "/v1/soccer_2016/player_name_by_match_over_ball_innings?match_id=419169&over_id=3&ball_id=2&innings_no=2",
    "/v1/soccer_2016/venue_name_by_win_margin?win_margin=138",
    "/v1/soccer_2016/player_name_by_match_date?match_date=2008-05-12",
    "/v1/soccer_2016/player_name_by_match_role?match_id=419117&role_desc=CaptainKeeper",
    "/v1/soccer_2016/player_name_by_season_year?season_year=2013",
    "/v1/soccer_2016/player_dob_by_season_year_orange_cap?season_year=2014",
    "/v1/soccer_2016/country_name_by_season_id_purple_cap?season_id=7",
    "/v1/soccer_2016/country_name_by_city_name?city_name=Ranchi",
    "/v1/soccer_2016/count_cities_by_country?country_name=India",
    "/v1/soccer_2016/city_with_most_venues",
    "/v1/soccer_2016/player_batting_hand?player_name=MK%20Pandey",
    "/v1/soccer_2016/city_ratio_by_countries?country_name_1=India&country_name_2=South%20Africa",
    "/v1/soccer_2016/match_venue_difference?venue_name_1=M%20Chinnaswamy%20Stadium&venue_name_2=Maharashtra%20Cricket%20Association%20Stadium",
    "/v1/soccer_2016/matches_in_month_year?month=5&year=2008",
    "/v1/soccer_2016/player_count_by_dob_range?start_date=1990-01-01&end_date=1999-12-31",
    "/v1/soccer_2016/matches_involving_team_in_year?team_id=10&year=2012",
    "/v1/soccer_2016/orange_cap_winners_by_count?min_count=1",
    "/v1/soccer_2016/match_count_by_season?season_id=7",
    "/v1/soccer_2016/umpire_count_by_country?country_name=South%20Africa",
    "/v1/soccer_2016/top_player_by_man_of_the_match?limit=1",
    "/v1/soccer_2016/top_country_by_player_count?limit=1",
    "/v1/soccer_2016/player_orange_cap_count?player_name=CH%20Gayle",
    "/v1/soccer_2016/top_season_by_venue?venue_name=M%20Chinnaswamy%20Stadium&limit=1",
    "/v1/soccer_2016/top_team_by_season?season_id=1&limit=1",
    "/v1/soccer_2016/top_venue_by_team?team_name=Kolkata%20Knight%20Riders&limit=1",
    "/v1/soccer_2016/top_team_by_losses?limit=1",
    "/v1/soccer_2016/first_man_of_the_match?limit=1",
    "/v1/soccer_2016/first_match_date_by_team?team_name=Chennai%20Super%20Kings&limit=1",
    "/v1/soccer_2016/player_count_by_batting_hand_and_country?batting_hand=Left-hand%20bat&country_name=India",
    "/v1/soccer_2016/team_captain?team_name=Deccan%20Chargers&team_id=8&role_desc=Captain&role_id=1",
    "/v1/soccer_2016/batting_hand_percentage?batting_hand=Right-hand%20bat",
    "/v1/soccer_2016/player_names_by_dob?dob=1981-07-07",
    "/v1/soccer_2016/match_count_by_player?player_id=2",
    "/v1/soccer_2016/team_with_highest_win_margin",
    "/v1/soccer_2016/country_by_venue?venue_name=St%20George's%20Park",
    "/v1/soccer_2016/team_name_by_match_id?match_id=335990&team_name=Mumbai%20Indians",
    "/v1/soccer_2016/team_name_by_match_date_win_margin?match_date=2009-05-07&win_margin=7",
    "/v1/soccer_2016/match_count_by_outcome_type?outcome_type=Superover",
    "/v1/soccer_2016/city_names_by_country?country_name=U.A.E",
    "/v1/soccer_2016/sum_matches_won_by_team?team_name=Pune%20Warriors",
    "/v1/soccer_2016/team_name_by_match_date_and_id?match_date=2015%25&match_id=829768",
    "/v1/soccer_2016/role_description_by_player_and_match?match_id=335992&player_name=K%20Goel",
    "/v1/soccer_2016/sum_cities_by_country?country_name=South%20Africa",
    "/v1/soccer_2016/sum_matches_by_venue?venue_name=Newlands",
    "/v1/soccer_2016/win_margin_by_teams_and_date?team_1=Mumbai%20Indians&team_2=Royal%20Challengers%20Bangalore&match_date=2008-05-28",
    "/v1/soccer_2016/teams_below_avg_win_margin?match_date=2011%25",
    "/v1/soccer_2016/percentage_players_by_role_and_dob?role_desc=Captain&dob=1977%25",
    "/v1/soccer_2016/count_overs_by_match_and_innings?match_id=335996&innings_no=1",
    "/v1/soccer_2016/highest_scoring_batsman_by_match?match_id=336004",
    "/v1/soccer_2016/match_ids_by_over_id?over_id=20",
    "/v1/soccer_2016/sum_wickets_taken_by_match_innings?match_id=548335&innings_no=1",
    "/v1/soccer_2016/match_ids_by_date?match_date=2015-04-18",
    "/v1/soccer_2016/match_ids_by_out_type?out_name=hit%20wicket",
    "/v1/soccer_2016/sum_wickets_taken_by_innings_out_type?innings_no=2&out_name=stumped",
    "/v1/soccer_2016/sum_man_of_the_match_by_player?player_name=Yuvraj%20Singh",
    "/v1/soccer_2016/player_names_dobs_by_country_dob?dob_year=1977&country_name=England",
    "/v1/soccer_2016/player_names_man_of_the_match_by_season?season_year=2010",
    "/v1/soccer_2016/win_percentage_by_team?match_winner=3&team_name=Chennai%20Super%20Kings",
    "/v1/soccer_2016/player_country_by_team_match?team_name=Gujarat%20Lions&match_date=2016-04-11",
    "/v1/soccer_2016/player_dob_by_bowling_skill?bowling_skill=Left-arm%20fast",
    "/v1/soccer_2016/country_by_umpire_name?umpire_name=BR%20Doctrove",
    "/v1/soccer_2016/player_by_match_date_role_winner?match_date=2008-06-01&role_desc=Captain&match_winner=1",
    "/v1/soccer_2016/team_match_count_by_player?player_name=CK%20Kapugedera",
    "/v1/soccer_2016/venue_match_percentage?venue_name=Wankhede%20Stadium&city_name=Mumbai",
    "/v1/soccer_2016/wicket_out_percentage?out_name=bowled&match_id=392187",
    "/v1/soccer_2016/toss_decision_percentage?toss_name=field&start_date=2010-01-01&end_date=2016-12-31",
    "/v1/soccer_2016/toss_winners_by_decision?toss_decide=2",
    "/v1/soccer_2016/match_ids_by_man_of_the_match?player_name=BB%20McCullum",
    "/v1/soccer_2016/man_of_the_match_dob",
    "/v1/soccer_2016/toss_winners_by_match_id_range?min_match_id=336010&max_match_id=336020",
    "/v1/soccer_2016/opponent_teams_by_team_name?team_name=Pune%20Warriors",
    "/v1/soccer_2016/match_winner_by_match_id?match_id=336000",
    "/v1/soccer_2016/match_ids_by_venue_name?venue_name=Brabourne%20Stadium",
    "/v1/soccer_2016/venues_by_season_id?season_id=2",
    "/v1/soccer_2016/city_name_by_venue_name?venue_name=M%20Chinnaswamy%20Stadium",
    "/v1/soccer_2016/venues_by_city_name?city_name=Mumbai",
    "/v1/soccer_2016/match_winners_by_venue_name_prefix?venue_name_prefix=St%20George",
    "/v1/soccer_2016/city_names_by_venue_name?venue_name_pattern=St%20George%25",
    "/v1/soccer_2016/sum_match_wins_by_team?team_name=Deccan%20Chargers",
    "/v1/soccer_2016/sum_venue_names_by_city?city_name=Pune",
    "/v1/soccer_2016/latest_ball_id_for_non_striker",
    "/v1/soccer_2016/percentage_runs_scored_in_over_range?min_over=1&max_over=25&innings_no=1",
    "/v1/soccer_2016/average_innings_number?innings_no=2",
    "/v1/soccer_2016/percentage_matches_won_by_margin?win_margin=100",
    "/v1/soccer_2016/player_names_by_dob_range?start_date=1970-01-01&end_date=1990-12-31",
    "/v1/soccer_2016/sum_wickets_no_fielders_by_over?over_id=3",
    "/v1/soccer_2016/country_with_most_umpires",
    "/v1/soccer_2016/percentage_players_role?role_desc=CaptainKeeper",
    "/v1/soccer_2016/player_out_by_out_type?out_name=hit%20wicket",
    "/v1/soccer_2016/percentage_players_batting_hand?batting_hand=Right-hand%20bat",
    "/v1/soccer_2016/percentage_players_bowling_skill?bowling_skill=Legbreak",
    "/v1/soccer_2016/count_matches_win_type_margin?win_type=wickets&win_margin=50",
    "/v1/soccer_2016/sum_matches_toss_winner_match_winner",
    "/v1/soccer_2016/man_of_the_series_country?season_year=2012",
    "/v1/soccer_2016/most_matches_venue",
    "/v1/soccer_2016/least_matches_city_win_type?win_type=NO%20Result",
    "/v1/soccer_2016/man_of_the_series_count?man_of_the_series_count=1",
    "/v1/soccer_2016/player_country_out_types",
    "/v1/soccer_2016/percentage_matches_won_by_toss_winner?toss_name=field&win_type=runs",
    "/v1/soccer_2016/average_players_out_by_type?out_name=lbw",
    "/v1/soccer_2016/distinct_over_ids_by_striker?striker=7",
    "/v1/soccer_2016/count_matches_toss_winner_won?toss_decide=2",
    "/v1/soccer_2016/count_matches_by_month_year?match_date=2010-03%",
    "/v1/soccer_2016/count_players_by_dob_and_name?dob=1990-06-29&player_name=Gurkeerat%20Singh",
    "/v1/soccer_2016/count_man_of_the_match?player_name=SR%20Watson",
    "/v1/soccer_2016/top_player_by_roles_in_team?team_name=Delhi%20Daredevils",
    "/v1/soccer_2016/top_player_by_man_of_the_series",
    "/v1/soccer_2016/player_season_orange_cap?player_name=SP%20Narine",
    "/v1/soccer_2016/team_caps",
    "/v1/soccer_2016/players_by_country?country_name=Zimbabwea",
    "/v1/soccer_2016/count_players_by_batting_hand?batting_hand=Left-hand%20bat",
    "/v1/soccer_2016/count_matches_not_won_by_type?win_type=runs",
    "/v1/soccer_2016/umpires_by_country?country_name=New%20Zealand",
    "/v1/soccer_2016/countries_by_bowling_skill?bowling_skill=Slow%20left-arm%20chinaman",
    "/v1/soccer_2016/venues_by_team?team_name=Kochi%20Tuskers%20Kerala",
    "/v1/soccer_2016/count_runs_scored_by_team?team_batting_1=1&team_batting_2=2&team_name=Delhi%20Daredevils",
    "/v1/soccer_2016/average_extra_runs?extra_name=noballs",
    "/v1/soccer_2016/top_players_by_bowling_skill?limit=5",
    "/v1/soccer_2016/player_count_by_dob_and_bowling_skill?dob=1975-10-16&bowling_skill=3",
    "/v1/soccer_2016/man_of_the_series_by_year_range?start_year=2011&end_year=2015",
    "/v1/soccer_2016/total_runs_scored_by_match_and_innings?match_id=335988&innings_no=2",
    "/v1/soccer_2016/runs_scored_above_threshold?runs_threshold=3&start_match_id=335989&end_match_id=337000&innings_no=1&over_id=1&ball_id=1",
    "/v1/soccer_2016/match_details_by_venue?venue_name=Kingsmead",
    "/v1/soccer_2016/match_count_by_venue_and_date_range?venue_name=MA%20Chidambaram%20Stadium&start_date=2009-05-09&end_date=2011-08-08",
    "/v1/soccer_2016/venue_and_city_by_match?match_id=336005",
    "/v1/soccer_2016/toss_details_by_match?match_id=336011",
    "/v1/soccer_2016/count_players_born_before_date_in_country?date=1980-04-11&country_name=South%20Africa",
    "/v1/soccer_2016/player_names_by_bowling_skill?bowling_skill=Legbreak",
    "/v1/soccer_2016/latest_match_date_role_by_dob",
    "/v1/soccer_2016/count_matches_by_date_range_and_country?start_date=2011%25&end_date=2012%25&country_name=Australia",
    "/v1/soccer_2016/player_names_man_of_series_orange_cap",
    "/v1/soccer_2016/match_dates_by_winning_team?team_name=Sunrisers%20Hyderabad",
    "/v1/soccer_2016/umpire_names_ids_by_country?country_name=England",
    "/v1/soccer_2016/runs_ratio_batting_fielding_by_match_over?match_id=335987&match_date=2008-04-18&fielding_count=17",
    "/v1/soccer_2016/match_id_highest_win_margin?limit=1",
    "/v1/soccer_2016/most_common_dob?limit=1",
    "/v1/soccer_2016/match_date_highest_win_margin?limit=1",
    "/v1/soccer_2016/season_most_matches?limit=1",
    "/v1/soccer_2016/match_count_man_of_the_match?min_count=5",
    "/v1/soccer_2016/player_man_of_the_match_season?season_id=9&limit=1",
    "/v1/soccer_2016/team_match_winner_season?season_id=1&limit=1",
    "/v1/soccer_2016/count_cities_in_country?country_name=U.A.E",
    "/v1/soccer_2016/umpires_from_country?country_name=England",
    "/v1/soccer_2016/players_with_bowling_style?bowling_style=Legbreak",
    "/v1/soccer_2016/sum_matches_season_team?season_id=8&team_name=Rajasthan%20Royals",
    "/v1/soccer_2016/umpire_country?umpire_name=TH%20Wijewardene",
    "/v1/soccer_2016/venue_by_city?city_name=Abu%20Dhabi",
    "/v1/soccer_2016/youngest_player_country",
    "/v1/soccer_2016/first_match_winner_player?season_id=1",
    "/v1/soccer_2016/highest_age_difference_player",
    "/v1/soccer_2016/most_recent_match_venue",
    "/v1/soccer_2016/sum_innings_by_match?match_id=336011",
    "/v1/soccer_2016/ball_details_by_match_over?match_id=335988&over_id=20",
    "/v1/soccer_2016/count_matches_by_year?year=2011",
    "/v1/soccer_2016/player_age?player_name=Ishan%20Kishan",
    "/v1/soccer_2016/toss_win_ratio?year=2012%25",
    "/v1/soccer_2016/match_count_year_win_margin?year=2009%25&win_margin=10",
    "/v1/soccer_2016/players_year_month?year=2014&month=6",
    "/v1/soccer_2016/total_matches_player?player_name=Mohammad%20Hafeez",
    "/v1/soccer_2016/players_country_year?country_name=South%20Africa&year=1984%25",
    "/v1/soccer_2016/team_win_percentage?team_name=Mumbai%20Indians&year=2009%25",
    "/v1/soccer_2016/batting_hand_ratio?left_hand=Left-hand%20bat&right_hand=Right-hand%20bat",
    "/v1/soccer_2016/oldest_player",
    "/v1/soccer_2016/bowling_skills_country?country_name=Zimbabwea",
    "/v1/soccer_2016/umpire_details_by_country?country_name=New%20Zealand",
    "/v1/soccer_2016/player_names_by_team_and_role?team_name=Rising%20Pune%20Supergiants&role_desc=CaptainKeeper",
    "/v1/soccer_2016/sum_matches_won_by_team_year?year=2013&team_name=Sunrisers%20Hyderabad",
    "/v1/soccer_2016/match_ids_by_extra_type?extra_name=penalty",
    "/v1/soccer_2016/team_name_by_year_win_type?year=2015&win_type=Tie",
    "/v1/soccer_2016/wicket_stats_by_innings_out_type?innings_no=2&out_name=lbw",
    "/v1/soccer_2016/match_count_by_innings?innings_no=2",
    "/v1/soccer_2016/country_name_by_city?city_name=Rajkot",
    "/v1/soccer_2016/sum_matches_by_win_type?win_type=wickets",
    "/v1/soccer_2016/team_names_by_win_margin_and_date?win_margin=38&match_date=2009-04-30",
    "/v1/soccer_2016/team_names_by_match_id_and_player_name?match_id=335989&player_name=T%20Kohli",
    "/v1/soccer_2016/venue_count_by_country_and_city?country_name=South%20Africa&city_name=Centurion",
    "/v1/soccer_2016/match_winner_count_by_team_and_date?team_name=Delhi%20Daredevils&match_date_pattern=2014%",
    "/v1/soccer_2016/highest_win_margin_match_id?team_name=Royal%20Challengers%20Bangalore&match_date_pattern=2012%",
    "/v1/soccer_2016/match_count_by_player_and_role?player_name=K%20Goel&role_id=3",
    "/v1/soccer_2016/average_win_margin_by_venue?venue_name=Newlands",
    "/v1/soccer_2016/losing_team_name_by_match_id?match_id=336039",
    "/v1/soccer_2016/venue_name_by_match_id?match_id=829768",
    "/v1/soccer_2016/highest_win_margin_away_team",
    "/v1/soccer_2016/percentage_matches_won_by_team?year=2013",
    "/v1/soccer_2016/role_difference_by_player?player_name=SC%20Ganguly",
    "/v1/soccer_2016/count_players_by_bowling_skill?bowling_skill=2",
    "/v1/soccer_2016/count_players_by_dob?year=1970",
    "/v1/soccer_2016/count_players_by_dob_and_bowling_skill?decade=198&bowling_skill=2",
    "/v1/soccer_2016/count_matches_with_winner_by_year?year=2008",
    "/v1/soccer_2016/country_id_by_city?city_name=East%20London",
    "/v1/soccer_2016/player_age_in_2008?player_name=SC%20Ganguly",
    "/v1/soccer_2016/player_names_by_batting_hand?batting_hand=Left-hand%20bat",
    "/v1/soccer_2016/player_names_by_country?country_name=England",
    "/v1/soccer_2016/venue_names_by_city?city_name=Bangalore",
    "/v1/soccer_2016/player_names_by_season_year?season_year=2008",
    "/v1/soccer_2016/player_names_by_runs_scored?runs_scored=3",
    "/v1/soccer_2016/role_descriptions_by_player_name?player_name=SC%20Ganguly",
    "/v1/soccer_2016/player_names_by_role_description?role_desc=Keeper",
    "/v1/soccer_2016/player_names_by_team_id?team_id=1",
    "/v1/soccer_2016/player_count_by_role_and_date?role_desc=Captain&match_date=2008%25",
    "/v1/soccer_2016/team_names_by_player_season?season_year=2008&player_name=SC%20Ganguly",
    "/v1/soccer_2016/win_type_by_match_id?match_id=336000",
    "/v1/soccer_2016/country_name_by_player_name?player_name=SB%20Joshi",
    "/v1/soccer_2016/player_count_by_bowling_style?bowling_style=Left-arm%20fast",
    "/v1/soccer_2016/outcome_type_by_match_id?match_id=392195",
    "/v1/soccer_2016/city_name_of_youngest_player",
    "/v1/soccer_2016/match_count_by_team_season?team_name=Kings%20XI%20Punjab&season_year=2008",
    "/v1/soccer_2016/season_count_by_team?team_name=Pune%20Warriors",
    "/v1/soccer_2016/player_dob_role_by_name?player_name=R%20Dravid",
    "/v1/soccer_2016/man_of_the_match_count_by_player?player_name=SC%20Ganguly",
    "/v1/soccer_2016/count_matches_won_by_team_year?team_name=Mumbai%20Indians&match_year=2008%25",
    "/v1/soccer_2016/team_names_won_specific_match?match_id=335993",
    "/v1/soccer_2016/count_matches_won_by_win_type?win_type=wickets",
    "/v1/soccer_2016/player_role_description_season?player_name=W%20Jaffer&season_year=2012",
    "/v1/soccer_2016/player_man_of_the_match_count_season?man_of_the_match_count=5&season_year=2008",
    "/v1/soccer_2016/proportion_players_country_birth_years?country_name=India&start_year=1975&end_year=1985",
    "/v1/soccer_2016/percentage_matches_win_type?win_type=1",
    "/v1/soccer_2016/count_matches_win_margin?win_margin=7",
    "/v1/soccer_2016/count_players_birth_years?start_year=1970&end_year=1975",
    "/v1/soccer_2016/team_names_by_match_id?match_id=419135",
    "/v1/soccer_2016/match_id_by_venue_name?venue_name=St%20George's%20Park",
    "/v1/soccer_2016/player_names_by_role?role_desc=Captain",
    "/v1/soccer_2016/venue_and_team_by_match_id?match_id=392194",
    "/v1/soccer_2016/win_percentage_by_wickets?team_name=Delhi%20Daredevils"
]
