from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/hockey/hockey.sqlite')
cursor = conn.cursor()

# Endpoint to get the first and last names of players born in a specific year and not in a specific country
@app.get("/v1/hockey/players_by_birth_year_and_country", operation_id="get_players_by_birth_year_and_country", summary="Retrieve the first and last names of hockey players born in a specific year and not from a specific country. The operation filters the player records based on the provided birth year and excludes those born in the specified country.")
async def get_players_by_birth_year_and_country(birth_year: int = Query(..., description="Birth year of the player"), birth_country: str = Query(..., description="Birth country of the player")):
    cursor.execute("SELECT firstName, lastName FROM Master WHERE birthYear = ? AND birthCountry != ?", (birth_year, birth_country))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the given names of players with a specific position and no shoot/catch preference
@app.get("/v1/hockey/players_by_position_no_shoot_catch", operation_id="get_players_by_position_no_shoot_catch", summary="Retrieves the first names of hockey players who play a specific position and have no preference for shoot or catch. The position is provided as an input parameter.")
async def get_players_by_position_no_shoot_catch(pos: str = Query(..., description="Position of the player")):
    cursor.execute("SELECT nameGiven FROM Master WHERE shootCatch IS NULL AND pos = ?", (pos,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the first and last names of players not in the Hall of Fame
@app.get("/v1/hockey/players_not_in_hall_of_fame", operation_id="get_players_not_in_hall_of_fame", summary="Retrieves the first and last names of hockey players who have not been inducted into the Hall of Fame.")
async def get_players_not_in_hall_of_fame():
    cursor.execute("SELECT firstName, lastName FROM Master WHERE hofID IS NULL")
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the most recently born player who is still alive
@app.get("/v1/hockey/most_recently_born_alive_player", operation_id="get_most_recently_born_alive_player", summary="Get the most recently born player who is still alive")
async def get_most_recently_born_alive_player():
    cursor.execute("SELECT nameGiven, birthYear, birthMon, birthDay FROM Master WHERE deathYear IS NULL ORDER BY birthYear DESC, birthMon DESC, birthDay DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": result}

# Endpoint to get the first and last names and age at death of players with no shoot/catch preference
@app.get("/v1/hockey/players_age_at_death_no_shoot_catch", operation_id="get_players_age_at_death_no_shoot_catch", summary="Retrieves the full names and calculated ages at death of hockey players who have no recorded shoot/catch preference and whose death years are known.")
async def get_players_age_at_death_no_shoot_catch():
    cursor.execute("SELECT firstName, lastName, deathYear - birthYear FROM Master WHERE shootCatch IS NULL AND deathYear IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the first and last names of goalies who played for more than a specified number of teams within a given year range
@app.get("/v1/hockey/goalies_by_year_range_and_team_count", operation_id="get_goalies_by_year_range_and_team_count", summary="Retrieve the first and last names of goalies who played for more than a specified number of teams within a given year range. This operation filters goalies based on the start and end year of the range, and the minimum number of teams played for. The result is a list of goalies who meet these criteria.")
async def get_goalies_by_year_range_and_team_count(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), min_teams: int = Query(..., description="Minimum number of teams played for")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.year >= ? AND T2.year <= ? GROUP BY T2.playerID HAVING COUNT(DISTINCT T2.tmID) > ?", (start_year, end_year, min_teams))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get the average weight of players taller than a specified height
@app.get("/v1/hockey/average_weight_by_height", operation_id="get_average_weight_by_height", summary="Retrieves the average weight of hockey players who are taller than the specified minimum height.")
async def get_average_weight_by_height(min_height: int = Query(..., description="Minimum height of the player")):
    cursor.execute("SELECT AVG(weight) FROM Master WHERE height > ?", (min_height,))
    result = cursor.fetchone()
    if not result:
        return {"average_weight": []}
    return {"average_weight": result[0]}

# Endpoint to get the distinct given names and birth countries of goalies who have played more than a specified number of minutes
@app.get("/v1/hockey/goalies_by_minutes_played", operation_id="get_goalies_by_minutes_played", summary="Retrieve the unique names and birth countries of goalies who have played more than the specified number of minutes. This operation provides a list of goalies who meet the minimum minutes played criteria, offering insights into their nationalities and given names.")
async def get_goalies_by_minutes_played(min_minutes: int = Query(..., description="Minimum number of minutes played")):
    cursor.execute("SELECT DISTINCT T1.nameGiven, T1.birthCountry FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID GROUP BY T1.nameGiven, T1.birthCountry HAVING SUM(T2.Min) > ?", (min_minutes,))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get the first and last names of goalies who have played in more than one specified league
@app.get("/v1/hockey/goalies_by_leagues", operation_id="get_goalies_by_leagues", summary="Retrieve the first and last names of goalies who have played in more than the specified minimum number of leagues. The operation accepts multiple league IDs as input parameters to filter the goalies. The result is a list of unique goalies who have played in at least the minimum number of distinct leagues.")
async def get_goalies_by_leagues(league1: str = Query(..., description="First league ID"), league2: str = Query(..., description="Second league ID"), min_leagues: int = Query(..., description="Minimum number of leagues played in")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.lgID IN (?, ?) GROUP BY T2.playerID HAVING COUNT(DISTINCT T2.lgID) > ?", (league1, league2, min_leagues))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get the goalie with the most minutes played in a single year who is deceased
@app.get("/v1/hockey/top_goalie_minutes_deceased", operation_id="get_top_goalie_minutes_deceased", summary="Retrieves the goalie who has played the most minutes in a single year, among those who are deceased. The response includes the goalie's ID, the year in which they played the most minutes, and the total minutes played in that year.")
async def get_top_goalie_minutes_deceased():
    cursor.execute("SELECT T1.playerID, T2.year, Min FROM Master AS T1 INNER JOIN Goalies AS T2 ON T2.playerID = T1.playerID WHERE T1.deathYear IS NOT NULL ORDER BY T2.Min DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"goalie": []}
    return {"goalie": result}

# Endpoint to get player details for goalies in a specific team and year range
@app.get("/v1/hockey/goalie_details_by_team_year_range", operation_id="get_goalie_details", summary="Retrieves the given name, height, weight, and age of goalies from a specific team within a specified year range. The team is identified by its unique ID, and the year range is defined by a start and end year. The data is grouped by player ID.")
async def get_goalie_details(tm_id: str = Query(..., description="Team ID"), start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT T1.nameGiven, T1.height , T1.weight, STRFTIME('%Y', CURRENT_TIMESTAMP) - birthYear FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.tmID = ? AND T2.year >= ? AND T2.year <= ? GROUP BY T1.playerID", (tm_id, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get goalie details based on minimum number of games played
@app.get("/v1/hockey/goalie_details_by_min_games", operation_id="get_goalie_details_by_min_games", summary="Retrieves the first name, last name, and year of goalies who have played a minimum number of games. The minimum number of games is specified as an input parameter.")
async def get_goalie_details_by_min_games(min_games: int = Query(..., description="Minimum number of games played")):
    cursor.execute("SELECT T1.firstName, T1.lastName , T2.year FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.ENG >= ?", (min_games,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the goalie with the best save percentage
@app.get("/v1/hockey/best_save_percentage_goalie", operation_id="get_best_save_percentage_goalie", summary="Retrieves the goalie with the highest save percentage from the hockey database. The save percentage is calculated as the ratio of goals against (GA) to shots against (SA). The result includes the goalie's first and last name, along with the year the save percentage was recorded.")
async def get_best_save_percentage_goalie():
    cursor.execute("SELECT T1.firstName, T1.lastName, T2.year FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE CAST(T2.GA AS REAL) / T2.SA IS NOT NULL ORDER BY CAST(T2.GA AS REAL) / T2.SA LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get goalie details based on year and height
@app.get("/v1/hockey/goalie_details_by_year_height", operation_id="get_goalie_details_by_year_height", summary="Retrieves unique details of hockey goalies who played in a specific year and are shorter than a given height. The response includes the first name, last name, and team name of each goalie.")
async def get_goalie_details_by_year_height(year: int = Query(..., description="Year"), max_height: int = Query(..., description="Maximum height in inches")):
    cursor.execute("SELECT DISTINCT T1.firstName, T1.lastName, T3.name FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID INNER JOIN Teams AS T3 ON T2.tmID = T3.tmID WHERE T2.year = ? AND T1.height < ?", (year, max_height))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get goalie details by player ID
@app.get("/v1/hockey/goalie_details_by_player_id", operation_id="get_goalie_details_by_player_id", summary="Retrieves the unique nickname, team name, and year of a goalie based on the provided player ID. The player ID is used to fetch the goalie's details from the Master and Goalies tables, and then the team details are obtained from the Teams table.")
async def get_goalie_details_by_player_id(player_id: str = Query(..., description="Player ID")):
    cursor.execute("SELECT DISTINCT T1.nameNick, T3.year, T3.name FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID INNER JOIN Teams AS T3 ON T2.tmID = T3.tmID WHERE T1.playerID = ?", (player_id,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the average minutes played by the goalie with the most games
@app.get("/v1/hockey/avg_minutes_most_games_goalie", operation_id="get_avg_minutes_most_games_goalie", summary="Retrieves the average minutes played per game by the goalie who has played the most games in a given year. The response includes the first name, last name, and year of the goalie.")
async def get_avg_minutes_most_games_goalie():
    cursor.execute("SELECT T1.firstName, T1.lastName, T2.year, AVG(T2.Min) FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T1.playerID = ( SELECT playerID FROM Goalies GROUP BY playerID ORDER BY COUNT(playerID) DESC LIMIT 1 ) GROUP BY T1.firstName, T1.lastName, T2.year")
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get goalie details based on shutout to goals against ratio
@app.get("/v1/hockey/goalie_details_by_sho_ga_ratio", operation_id="get_goalie_details_by_sho_ga_ratio", summary="Retrieves unique details of hockey goalies who have a shutout to goals against ratio greater than the provided value. The response includes the first name, last name, and the year of the goalie.")
async def get_goalie_details_by_sho_ga_ratio(sho_ga_ratio: float = Query(..., description="Shutout to goals against ratio")):
    cursor.execute("SELECT DISTINCT T1.firstName, T1.lastName, T2.year FROM Master AS T1 INNER JOIN ( SELECT playerID, year FROM Goalies WHERE CAST(SHO AS REAL) / GA > ? ) AS T2 ON T2.playerID = T1.playerID", (sho_ga_ratio,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get players who died in a different country from their birth country
@app.get("/v1/hockey/players_different_birth_death_country", operation_id="get_players_different_birth_death_country", summary="Retrieves a list of hockey players who passed away in a country different from their birth country. The players are sorted by their birth year in ascending order.")
async def get_players_different_birth_death_country():
    cursor.execute("SELECT firstName, lastName FROM Master WHERE birthCountry != deathCountry ORDER BY birthYear")
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get players who played in both NHL and WHA
@app.get("/v1/hockey/players_nhl_wha", operation_id="get_players_nhl_wha", summary="Retrieves the names and debut dates of players who have played in both the National Hockey League (NHL) and the World Hockey Association (WHA).")
async def get_players_nhl_wha():
    cursor.execute("SELECT nameGiven, firstNHL, firstWHA FROM Master WHERE firstNHL IS NOT NULL AND firstWHA IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get living players with a specific position
@app.get("/v1/hockey/living_players_by_position", operation_id="get_living_players_by_position", summary="Retrieves a list of living hockey players who play in the specified position(s). The position parameter accepts a single position or multiple positions using the %/% delimiter. The response includes the first and last names of the players.")
async def get_living_players_by_position(position: str = Query(..., description="Position (use %/% for multiple positions)")):
    cursor.execute("SELECT firstName, lastName, pos FROM Master WHERE deathYear IS NULL AND pos LIKE ?", (position,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the tallest player's nickname and last NHL season
@app.get("/v1/hockey/tallest_player", operation_id="get_tallest_player", summary="Get the tallest player's nickname and last NHL season")
async def get_tallest_player(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT nameNick, lastNHL FROM Master ORDER BY height DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": result}

# Endpoint to get the difference in average height between players born before and after a certain year
@app.get("/v1/hockey/height_difference_by_birth_year", operation_id="get_height_difference", summary="Retrieves the difference in average height between hockey players born before and after a specified year. This operation calculates the average height of players born before the input year and subtracts it from the average height of players born in the input year or later. The result provides insights into the height trends of hockey players across different birth years.")
async def get_height_difference(birth_year: int = Query(..., description="Year to compare birth years")):
    cursor.execute("SELECT AVG(IIF(birthYear < ?, height, NULL)) - AVG(IIF(birthYear >= ?, height, NULL)) FROM Master", (birth_year, birth_year))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get distinct goalies with specific shooting and position criteria
@app.get("/v1/hockey/goalies_by_shooting_position", operation_id="get_goalies_by_shooting_position", summary="Retrieves a list of unique goalies who meet the specified shooting hand and position criteria. The response includes the first name, last name, and team name of each goalie. The goalies are filtered based on their shooting hand and position, as well as their association with a coach and a team.")
async def get_goalies_by_shooting_position(shoot_catch: str = Query(..., description="Shooting hand of the goalie (e.g., 'L' for left)"), pos: str = Query(..., description="Position of the goalie (e.g., 'G' for goalie)")):
    cursor.execute("SELECT DISTINCT firstName, lastName, T3.name FROM Goalies AS T1 INNER JOIN Master AS T2 ON T2.playerID = T1.playerID INNER JOIN Teams AS T3 ON T1.lgID = T3.lgID WHERE T1.playerID IS NOT NULL AND T2.coachID IS NOT NULL AND T2.shootCatch = ? AND T2.pos = ?", (shoot_catch, pos))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get distinct goalies from a specific birth country and position who have passed away
@app.get("/v1/hockey/goalies_by_birth_country_position", operation_id="get_goalies_by_birth_country_position", summary="Retrieve a unique list of goalies who were born in a specific country and played a certain position, and who have since passed away. The response includes the first and last names of the goalies, as well as the name of the team they played for.")
async def get_goalies_by_birth_country_position(birth_country: str = Query(..., description="Birth country of the goalie (e.g., 'Canada')"), pos: str = Query(..., description="Position of the goalie (e.g., 'G' for goalie)")):
    cursor.execute("SELECT DISTINCT firstName, lastName, T3.name FROM Goalies AS T1 INNER JOIN Master AS T2 ON T2.playerID = T1.playerID INNER JOIN Teams AS T3 ON T1.lgID = T3.lgID WHERE T2.birthCountry = ? AND T2.deathYear IS NOT NULL AND T2.pos = ?", (birth_country, pos))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get goalies who played for a specific team and rank with a specific position and have passed away
@app.get("/v1/hockey/goalies_by_team_rank_position", operation_id="get_goalies_by_team_rank_position", summary="Retrieve the names and year of death of goalies who played for a specific team, held a certain rank, and played in a specific position. The team is identified by its name, the rank is a numerical value, and the position is a single-letter code.")
async def get_goalies_by_team_rank_position(team_name: str = Query(..., description="Name of the team (e.g., 'Boston Bruins')"), rank: int = Query(..., description="Rank of the team"), pos: str = Query(..., description="Position of the goalie (e.g., 'G' for goalie)")):
    cursor.execute("SELECT T1.firstName, T1.lastName, T3.year FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID INNER JOIN Teams AS T3 ON T2.year = T3.year AND T2.tmID = T3.tmID WHERE T1.deathYear IS NOT NULL AND T3.name = ? AND T3.rank = ? AND T1.pos = ?", (team_name, rank, pos))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get distinct team names for goalies who debuted before a certain year and have passed away
@app.get("/v1/hockey/teams_by_debut_year", operation_id="get_teams_by_debut_year", summary="Retrieves unique team names of deceased goalies who made their NHL debut before the specified year. The operation filters the data based on the year of NHL debut and returns a list of distinct team names.")
async def get_teams_by_debut_year(first_nhl: int = Query(..., description="Year of NHL debut (e.g., 1950)")):
    cursor.execute("SELECT DISTINCT T3.name FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID INNER JOIN Teams AS T3 ON T2.lgID = T3.lgID AND T2.year = T3.year WHERE T1.deathYear IS NOT NULL AND T1.firstNHL < ?", (first_nhl,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": result}

# Endpoint to get distinct coaches who were also players
@app.get("/v1/hockey/coaches_who_were_players", operation_id="get_coaches_who_were_players", summary="Retrieves a list of unique coaches who have also played on a team. The response includes each coach's first name, the team they coached, and the year they coached that team.")
async def get_coaches_who_were_players():
    cursor.execute("SELECT DISTINCT T2.nameGiven, T3.name, T3.year FROM Coaches AS T1 INNER JOIN Master AS T2 ON T2.coachID = T1.coachID INNER JOIN Teams AS T3 ON T1.lgID = T3.lgID WHERE T2.playerID IS NOT NULL AND T2.coachID IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"coaches": []}
    return {"coaches": result}

# Endpoint to get the coach with the highest win rate
@app.get("/v1/hockey/top_coach_by_win_rate", operation_id="get_top_coach_by_win_rate", summary="Get the coach with the highest win rate")
async def get_top_coach_by_win_rate(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T2.nameGiven, T3.name FROM Coaches AS T1 INNER JOIN Master AS T2 ON T2.coachID = T1.coachID INNER JOIN Teams AS T3 ON T1.lgID = T3.lgID WHERE T1.coachID IS NOT NULL ORDER BY CAST(T1.w AS REAL) / T1.g DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"coach": []}
    return {"coach": result}

# Endpoint to get the top coach by win rate with details
@app.get("/v1/hockey/top_coach_details_by_win_rate", operation_id="get_top_coach_details_by_win_rate", summary="Get the top coach by win rate with details")
async def get_top_coach_details_by_win_rate(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT CAST(T2.W AS REAL) / T2.G, T1.firstName, T1.lastName, T2.year FROM Master AS T1 INNER JOIN Coaches AS T2 ON T1.coachID = T2.coachID INNER JOIN ( SELECT coachID FROM Coaches ORDER BY CAST(w AS REAL) / g DESC LIMIT ? ) AS T3 ON T2.coachID = T3.coachID", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"coach": []}
    return {"coach": result}

# Endpoint to get the goalie with the most coaching experience for a specific team
@app.get("/v1/hockey/top_goalie_by_coaching_experience", operation_id="get_top_goalie_by_coaching_experience", summary="Retrieves the goalie with the most coaching experience for a specific team, based on the provided team ID. The response is sorted in descending order by the number of coaching experiences and limited to the specified number of results.")
async def get_top_goalie_by_coaching_experience(team_id: str = Query(..., description="Team ID (e.g., 'MTL')"), limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T2.nameGiven , T2.birthYear, T2.birthMon, T2.birthDay, T3.name FROM Goalies AS T1 INNER JOIN Master AS T2 ON T2.playerID = T1.playerID INNER JOIN Teams AS T3 ON T3.lgID = T1.lgID WHERE T3.tmID = ? GROUP BY T2.nameGiven, T2.birthYear, T2.birthMon, T2.birthDay, T3.name ORDER BY COUNT(T2.coachID) DESC LIMIT ?", (team_id, limit))
    result = cursor.fetchone()
    if not result:
        return {"goalie": []}
    return {"goalie": result}

# Endpoint to get distinct goalies with more losses than wins over multiple years
@app.get("/v1/hockey/goalies_with_more_losses_than_wins", operation_id="get_goalies_with_more_losses_than_wins", summary="Retrieve a list of unique goalies who have recorded more losses than wins over a specified minimum number of years. The position of the player is also a required input parameter.")
async def get_goalies_with_more_losses_than_wins(pos: str = Query(..., description="Position of the player"), min_years: int = Query(..., description="Minimum number of years")):
    cursor.execute("SELECT DISTINCT T1.firstName, T1.lastName, T3.name FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID INNER JOIN Teams AS T3 ON T2.year = T3.year AND T2.tmID = T3.tmID WHERE T1.pos = ? AND T2.L > T2.W GROUP BY T1.firstName, T1.lastName, T3.name HAVING COUNT(T3.year) > ?", (pos, min_years))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get goalies born in a specific year with null shootCatch
@app.get("/v1/hockey/goalies_born_in_year_with_null_shootCatch", operation_id="get_goalies_born_in_year_with_null_shootCatch", summary="Retrieves a list of goalies who were born in a specific year and have no recorded shootCatch preference. The response includes each goalie's first name, last name, birth year, and their win-loss ratio.")
async def get_goalies_born_in_year_with_null_shootCatch(birth_year: int = Query(..., description="Birth year of the player")):
    cursor.execute("SELECT T1.firstName, T1.lastName, T2.year, CAST(T2.W AS REAL) / T2.GP FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T1.birthYear = ? AND T1.shootCatch IS NULL", (birth_year,))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get average minutes per game for a specific player
@app.get("/v1/hockey/average_minutes_per_game_for_player", operation_id="get_average_minutes_per_game_for_player", summary="Retrieves the average number of minutes played per game for a specific hockey player, identified by their unique player ID.")
async def get_average_minutes_per_game_for_player(player_id: str = Query(..., description="Player ID")):
    cursor.execute("SELECT T1.nameGiven, CAST(SUM(T2.Min) AS REAL) / SUM(T2.GP) FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T1.playerID = ? GROUP BY T1.nameGiven", (player_id,))
    result = cursor.fetchall()
    if not result:
        return {"average_minutes": []}
    return {"average_minutes": result}

# Endpoint to get goalies with a specific average minutes per game who have died
@app.get("/v1/hockey/goalies_with_average_minutes_per_game_who_have_died", operation_id="get_goalies_with_average_minutes_per_game_who_have_died", summary="Retrieve a list of deceased goalies who have played an average of more than the specified minutes per game. The response includes the first and last names of the goalies.")
async def get_goalies_with_average_minutes_per_game_who_have_died(avg_minutes_per_game: float = Query(..., description="Average minutes per game")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T1.deathYear IS NOT NULL GROUP BY T1.playerID HAVING CAST(SUM(T2.Min) AS REAL) / SUM(T2.GP) > ?", (avg_minutes_per_game,))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get the count of non-null notes in AwardsMisc
@app.get("/v1/hockey/count_non_null_notes_in_awards_misc", operation_id="get_count_non_null_notes_in_awards_misc", summary="Retrieves the total count of non-empty notes in the AwardsMisc records. This operation provides a quantitative overview of the number of entries that contain additional information in the notes field.")
async def get_count_non_null_notes_in_awards_misc():
    cursor.execute("SELECT COUNT(note) FROM AwardsMisc WHERE note IS NOT NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players with a specific position in a specific year
@app.get("/v1/hockey/count_players_by_position_and_year", operation_id="get_count_players_by_position_and_year", summary="Retrieves the total number of players who played a specified position during a given year.")
async def get_count_players_by_position_and_year(pos: str = Query(..., description="Position of the player"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT COUNT(playerID) FROM AwardsPlayers WHERE pos = ? AND year = ?", (pos, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of coaches with specific notes in a specific year
@app.get("/v1/hockey/count_coaches_by_year_and_notes", operation_id="get_count_coaches_by_year_and_notes", summary="Retrieves the total number of coaches who have specific notes in a given year. The year and notes are provided as input parameters.")
async def get_count_coaches_by_year_and_notes(year: int = Query(..., description="Year"), notes: str = Query(..., description="Notes")):
    cursor.execute("SELECT COUNT(coachID) FROM Coaches WHERE year = ? AND notes = ?", (year, notes))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of combined shutouts in a specific year and role
@app.get("/v1/hockey/count_combined_shutouts_by_year_and_role", operation_id="get_count_combined_shutouts_by_year_and_role", summary="Retrieves the total number of combined shutouts in a given year and role. The year and role are provided as input parameters, allowing for a targeted count of combined shutouts.")
async def get_count_combined_shutouts_by_year_and_role(year: int = Query(..., description="Year"), role: str = Query(..., description="Role (R/P)")):
    cursor.execute("SELECT COUNT(year) FROM CombinedShutouts WHERE year = ? AND `R/P` = ?", (year, role))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of goalies with null ENG in a specific year
@app.get("/v1/hockey/count_goalies_with_null_eng_by_year", operation_id="get_count_goalies_with_null_eng_by_year", summary="Retrieves the total number of goalies in a given year who have no recorded data for the ENG attribute.")
async def get_count_goalies_with_null_eng_by_year(year: int = Query(..., description="Year")):
    cursor.execute("SELECT COUNT(tmID) FROM Goalies WHERE year = ? AND ENG IS NULL", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct years for goalies in a specific league with non-null SA
@app.get("/v1/hockey/distinct_years_for_goalies_in_league_with_non_null_sa", operation_id="get_distinct_years_for_goalies_in_league_with_non_null_sa", summary="Retrieves a list of unique years in which goalies from a specified league have recorded non-null save attempts (SA). The league is identified by its unique ID.")
async def get_distinct_years_for_goalies_in_league_with_non_null_sa(lg_id: str = Query(..., description="League ID")):
    cursor.execute("SELECT DISTINCT year FROM Goalies WHERE lgID = ? AND SA IS NOT NULL", (lg_id,))
    result = cursor.fetchall()
    if not result:
        return {"years": []}
    return {"years": result}

# Endpoint to get the distinct count of goalies where PostW equals PostL
@app.get("/v1/hockey/goalies_count_postw_eq_postl", operation_id="get_goalies_count_postw_eq_postl", summary="Retrieves the unique count of goalies who have the same position for both the left and right wings.")
async def get_goalies_count_postw_eq_postl():
    cursor.execute("SELECT DISTINCT COUNT(tmID) FROM Goalies WHERE PostW = PostL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of Hall of Fame inductees for a specific year
@app.get("/v1/hockey/hof_inductees_by_year", operation_id="get_hof_inductees_by_year", summary="Retrieves the names of individuals inducted into the Hall of Fame in a specified year.")
async def get_hof_inductees_by_year(year: int = Query(..., description="Year of induction into the Hall of Fame")):
    cursor.execute("SELECT name FROM HOF WHERE year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of Hall of Fame inductees by category
@app.get("/v1/hockey/hof_inductees_count_by_category", operation_id="get_hof_inductees_count_by_category", summary="Retrieves the total number of Hall of Fame inductees in a specific category. The category is provided as an input parameter.")
async def get_hof_inductees_count_by_category(category: str = Query(..., description="Category of the Hall of Fame inductee")):
    cursor.execute("SELECT COUNT(hofID) FROM HOF WHERE category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of Hall of Fame inductees by year and category
@app.get("/v1/hockey/hof_inductees_count_by_year_and_category", operation_id="get_hof_inductees_count_by_year_and_category", summary="Retrieves the total number of Hall of Fame inductees for a specific category and year. The year and category are provided as input parameters, allowing for a targeted count of inductees.")
async def get_hof_inductees_count_by_year_and_category(year: int = Query(..., description="Year of induction into the Hall of Fame"), category: str = Query(..., description="Category of the Hall of Fame inductee")):
    cursor.execute("SELECT COUNT(hofID) FROM HOF WHERE year > ? AND category = ?", (year, category))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the distinct nicknames of Hall of Fame inductees for a specific year
@app.get("/v1/hockey/hof_inductees_nicknames_by_year", operation_id="get_hof_inductees_nicknames_by_year", summary="Retrieve a unique set of nicknames for players inducted into the Hockey Hall of Fame in a specified year.")
async def get_hof_inductees_nicknames_by_year(year: int = Query(..., description="Year of induction into the Hall of Fame")):
    cursor.execute("SELECT DISTINCT T1.nameNick FROM Master AS T1 INNER JOIN HOF AS T2 ON T1.hofID = T2.hofID WHERE T2.year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"nicknames": []}
    return {"nicknames": [row[0] for row in result]}

# Endpoint to get the Hall of Fame induction status of the tallest player
@app.get("/v1/hockey/hof_status_tallest_player", operation_id="get_hof_status_tallest_player", summary="Retrieves the Hall of Fame induction status of the tallest player in the database. If the player is not in the Hall of Fame, the response will be 'NO'. Otherwise, the year of induction will be returned.")
async def get_hof_status_tallest_player():
    cursor.execute("SELECT CASE WHEN T1.hofID IS NULL THEN 'NO' ELSE T2.year END FROM Master AS T1 LEFT JOIN HOF AS T2 ON T1.hofID = T2.hofID WHERE T1.height = ( SELECT MAX(height) FROM Master )")
    result = cursor.fetchall()
    if not result:
        return {"status": []}
    return {"status": [row[0] for row in result]}

# Endpoint to get the distinct awards of coaches from a specific birth country
@app.get("/v1/hockey/coach_awards_by_birth_country", operation_id="get_coach_awards_by_birth_country", summary="Retrieve a unique list of awards won by coaches born in a specified country. The operation filters coaches based on their birth country and returns a distinct set of awards they have received.")
async def get_coach_awards_by_birth_country(birth_country: str = Query(..., description="Birth country of the coach")):
    cursor.execute("SELECT DISTINCT T2.award FROM Master AS T1 INNER JOIN AwardsCoaches AS T2 ON T1.coachID = T2.coachID WHERE T1.birthCountry = ?", (birth_country,))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": [row[0] for row in result]}

# Endpoint to get the count of coaches with more than a specified number of wins from a specific birth country
@app.get("/v1/hockey/coach_count_by_wins_and_birth_country", operation_id="get_coach_count_by_wins_and_birth_country", summary="Retrieves the count of coaches who have surpassed a specified number of wins and originate from a particular birth country. The operation filters coaches based on the provided win count and birth country, then returns the total count of coaches who meet these criteria.")
async def get_coach_count_by_wins_and_birth_country(wins: int = Query(..., description="Number of wins"), birth_country: str = Query(..., description="Birth country of the coach")):
    cursor.execute("SELECT COUNT(T2.coachID) FROM Master AS T1 INNER JOIN Coaches AS T2 ON T1.coachID = T2.coachID WHERE T2.W > ? AND T1.birthCountry = ?", (wins, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of coaches from a specific league and birth country
@app.get("/v1/hockey/coach_count_by_league_and_birth_country", operation_id="get_coach_count_by_league_and_birth_country", summary="Retrieves the total number of coaches from a specified league and birth country. The league is identified by its unique ID, and the birth country is provided as a string. This operation is useful for analyzing the distribution of coaches across different leagues and countries.")
async def get_coach_count_by_league_and_birth_country(league_id: str = Query(..., description="League ID"), birth_country: str = Query(..., description="Birth country of the coach")):
    cursor.execute("SELECT COUNT(T2.coachID) FROM Master AS T1 INNER JOIN Coaches AS T2 ON T1.coachID = T2.coachID WHERE T2.lgID = ? AND T1.birthCountry = ?", (league_id, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the awards of coaches born in a specific year
@app.get("/v1/hockey/coach_awards_by_birth_year", operation_id="get_coach_awards_by_birth_year", summary="Retrieves the awards won by coaches who were born in the specified year. The birth year is provided as an input parameter.")
async def get_coach_awards_by_birth_year(birth_year: int = Query(..., description="Birth year of the coach")):
    cursor.execute("SELECT T2.award FROM Master AS T1 INNER JOIN AwardsCoaches AS T2 ON T1.coachID = T2.coachID WHERE T1.birthYear = ?", (birth_year,))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": [row[0] for row in result]}

# Endpoint to get the count of coaches based on year and birth city
@app.get("/v1/hockey/count_coaches_by_year_birth_city", operation_id="get_count_coaches_by_year_birth_city", summary="Retrieves the total number of coaches who received an award in a specific year and were born in a particular city. The operation requires the year of the award and the birth city of the coach as input parameters.")
async def get_count_coaches_by_year_birth_city(year: int = Query(..., description="Year of the award"), birth_city: str = Query(..., description="Birth city of the coach")):
    cursor.execute("SELECT COUNT(T1.coachID) FROM Master AS T1 INNER JOIN AwardsCoaches AS T2 ON T1.coachID = T2.coachID WHERE T2.year = ? AND T1.birthCity = ?", (year, birth_city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of deceased coaches awarded after a specific year
@app.get("/v1/hockey/count_deceased_coaches_awarded_after_year", operation_id="get_count_deceased_coaches_awarded_after_year", summary="Retrieves the total number of deceased coaches who received an award after the specified year. The year parameter is used to filter the results.")
async def get_count_deceased_coaches_awarded_after_year(year: int = Query(..., description="Year after which the award was given")):
    cursor.execute("SELECT COUNT(T1.coachID) FROM Master AS T1 INNER JOIN AwardsCoaches AS T2 ON T1.coachID = T2.coachID WHERE T1.deathYear IS NOT NULL AND T2.year > ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct awards for deceased coaches in a specific league
@app.get("/v1/hockey/distinct_awards_deceased_coaches_league", operation_id="get_distinct_awards_deceased_coaches_league", summary="Retrieves a unique list of awards won by coaches who have passed away in a specified league. The league is identified by its unique ID.")
async def get_distinct_awards_deceased_coaches_league(lg_id: str = Query(..., description="League ID")):
    cursor.execute("SELECT DISTINCT T2.award FROM Master AS T1 INNER JOIN AwardsCoaches AS T2 ON T1.coachID = T2.coachID WHERE T1.deathYear IS NOT NULL AND T2.lgID = ?", (lg_id,))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": [row[0] for row in result]}

# Endpoint to get the count of distinct coaches in the Hall of Fame with weight greater than a specific value
@app.get("/v1/hockey/count_distinct_coaches_hof_weight", operation_id="get_count_distinct_coaches_hof_weight", summary="Retrieve the number of unique coaches in the Hall of Fame who have a weight greater than the provided value. The weight parameter is used to filter the coaches.")
async def get_count_distinct_coaches_hof_weight(weight: int = Query(..., description="Weight of the coach")):
    cursor.execute("SELECT COUNT(DISTINCT T1.coachID) FROM Master AS T1 INNER JOIN HOF AS T2 ON T1.hofID = T2.hofID WHERE T1.weight > ?", (weight,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct names of Hall of Fame members with unknown shoot/catch preference
@app.get("/v1/hockey/distinct_names_hof_unknown_shoot_catch", operation_id="get_distinct_names_hof_unknown_shoot_catch", summary="Retrieves a list of unique names of Hall of Fame members who have not specified their shoot/catch preference. This operation fetches the first and last names from the Master table, which are then cross-referenced with the Hall of Fame (HOF) table using the hofID field. Only records with a null value in the shootCatch field are considered.")
async def get_distinct_names_hof_unknown_shoot_catch():
    cursor.execute("SELECT DISTINCT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN HOF AS T2 ON T1.hofID = T2.hofID WHERE T1.shootCatch IS NULL")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"firstName": row[0], "lastName": row[1]} for row in result]}

# Endpoint to get the count of Hall of Fame members who are both players and coaches
@app.get("/v1/hockey/count_hof_players_coaches", operation_id="get_count_hof_players_coaches", summary="Retrieves the total number of individuals who have been inducted into the Hall of Fame as both players and coaches. This operation returns a count of members who have contributed to the sport in both capacities, providing a unique perspective on their achievements.")
async def get_count_hof_players_coaches():
    cursor.execute("SELECT COUNT(T1.playerID) FROM Master AS T1 INNER JOIN HOF AS T2 ON T1.hofID = T2.hofID WHERE T1.playerID IS NOT NULL AND T1.coachID IS NOT NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct birth cities of players awarded in a specific year
@app.get("/v1/hockey/distinct_birth_cities_players_awarded_year", operation_id="get_distinct_birth_cities_players_awarded_year", summary="Retrieves a unique list of birth cities for players who received an award in a specified year. The year of the award is a required input parameter.")
async def get_distinct_birth_cities_players_awarded_year(year: int = Query(..., description="Year of the award")):
    cursor.execute("SELECT DISTINCT T1.birthCity FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"birth_cities": []}
    return {"birth_cities": [row[0] for row in result]}

# Endpoint to get the count of players with a specific award and birth city
@app.get("/v1/hockey/count_players_award_birth_city", operation_id="get_count_players_award_birth_city", summary="Retrieves the total number of hockey players who have received a specific award and were born in a particular city. The operation requires the name of the award and the birth city as input parameters.")
async def get_count_players_award_birth_city(award: str = Query(..., description="Award name"), birth_city: str = Query(..., description="Birth city of the player")):
    cursor.execute("SELECT COUNT(T1.playerID) FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ? AND T1.birthCity = ?", (award, birth_city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of deceased players with a specific award
@app.get("/v1/hockey/count_deceased_players_award", operation_id="get_count_deceased_players_award", summary="Retrieves the total number of deceased players who have received a specific award. The award is specified as an input parameter.")
async def get_count_deceased_players_award(award: str = Query(..., description="Award name")):
    cursor.execute("SELECT COUNT(T1.playerID) FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ? AND T1.deathYear IS NOT NULL", (award,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct deceased players from a specific state
@app.get("/v1/hockey/count_distinct_deceased_players_state", operation_id="get_count_distinct_deceased_players_state", summary="Retrieve the unique count of deceased hockey players who passed away in a specified state. The state is provided as an input parameter.")
async def get_count_distinct_deceased_players_state(death_state: str = Query(..., description="State where the player died")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID WHERE T1.deathState = ?", (death_state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get awards of players who died in a specific city
@app.get("/v1/hockey/awards_by_death_city", operation_id="get_awards_by_death_city", summary="Retrieves the awards won by hockey players who passed away in a specified city. The operation filters the list of players based on the provided city of death and returns the corresponding awards.")
async def get_awards_by_death_city(death_city: str = Query(..., description="City where the player died")):
    cursor.execute("SELECT T2.award FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID WHERE T1.deathCity = ?", (death_city,))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": [row[0] for row in result]}

# Endpoint to get distinct nicknames of players with a specific award and birth month
@app.get("/v1/hockey/nicknames_by_award_birth_month", operation_id="get_nicknames_by_award_birth_month", summary="Retrieves unique nicknames of players who have been awarded a specific honor and were born in a particular month. The operation filters players based on the provided award and birth month, ensuring that only distinct nicknames are returned.")
async def get_nicknames_by_award_birth_month(award: str = Query(..., description="Award received by the player"), birth_month: int = Query(..., description="Birth month of the player")):
    cursor.execute("SELECT DISTINCT T1.nameNick FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ? AND T1.birthMon = ?", (award, birth_month))
    result = cursor.fetchall()
    if not result:
        return {"nicknames": []}
    return {"nicknames": [row[0] for row in result]}

# Endpoint to get the count of players in the Hall of Fame born in specific months
@app.get("/v1/hockey/hof_player_count_by_birth_months", operation_id="get_hof_player_count_by_birth_months", summary="Retrieve the number of Hall of Fame players born in the specified months. This operation allows you to compare the count of players inducted into the Hall of Fame based on their birth months. The input parameters represent the birth months of interest.")
async def get_hof_player_count_by_birth_months(birth_month1: int = Query(..., description="First birth month of the player"), birth_month2: int = Query(..., description="Second birth month of the player")):
    cursor.execute("SELECT COUNT(T1.playerID) FROM Master AS T1 INNER JOIN HOF AS T2 ON T1.hofID = T2.hofID WHERE T1.birthMon IN (?, ?)", (birth_month1, birth_month2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the birth month of the player with the most awards
@app.get("/v1/hockey/birth_month_most_awards", operation_id="get_birth_month_most_awards", summary="Retrieves the birth month of the player who has been honored with the highest number of awards. This operation identifies the player with the most awards by joining the Master and AwardsPlayers tables, grouping by playerID, and ordering by the count of awards in descending order. The birth month of the top-ranked player is then returned.")
async def get_birth_month_most_awards():
    cursor.execute("SELECT T1.birthMon FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID GROUP BY T2.playerID ORDER BY COUNT(T2.award) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"birth_month": []}
    return {"birth_month": result[0]}

# Endpoint to get the birth year of the player with the most awards
@app.get("/v1/hockey/birth_year_most_awards", operation_id="get_birth_year_most_awards", summary="Retrieves the birth year of the player who has been honored with the highest number of awards. This operation identifies the player with the most awards by joining the Master and AwardsPlayers tables, grouping by birth year, and ordering the results in descending order based on the count of awards. The birth year of the top-ranked player is then returned.")
async def get_birth_year_most_awards():
    cursor.execute("SELECT T1.birthYear FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID GROUP BY T1.birthYear ORDER BY COUNT(T2.award) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"birth_year": []}
    return {"birth_year": result[0]}

# Endpoint to get the birth country of the player with the most awards
@app.get("/v1/hockey/birth_country_most_awards", operation_id="get_birth_country_most_awards", summary="Retrieves the birth country of the player who has been honored with the highest number of awards in the hockey league. The data is obtained by aggregating and sorting the awards received by each player, grouped by their birth country.")
async def get_birth_country_most_awards():
    cursor.execute("SELECT T1.birthCountry FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID GROUP BY T1.birthCountry ORDER BY COUNT(T2.award) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"birth_country": []}
    return {"birth_country": result[0]}

# Endpoint to get the birth country of the Hall of Fame player with the most players
@app.get("/v1/hockey/birth_country_most_hof_players", operation_id="get_birth_country_most_hof_players", summary="Retrieves the birth country of the Hall of Fame player with the highest number of fellow inductees from the same country. The data is fetched from the Master and HOF tables, grouped by birth country, and ordered by the count of player IDs in descending order. The result is limited to the top entry.")
async def get_birth_country_most_hof_players():
    cursor.execute("SELECT T1.birthCountry FROM Master AS T1 INNER JOIN HOF AS T2 ON T1.hofID = T2.hofID GROUP BY T1.birthCountry ORDER BY COUNT(T1.playerID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"birth_country": []}
    return {"birth_country": result[0]}

# Endpoint to get distinct positions of players from a specific country with a specific award
@app.get("/v1/hockey/positions_by_country_award", operation_id="get_positions_by_country_award", summary="Retrieves the unique positions held by players born in a specified country who have been awarded a specific honor. The response is based on a combination of player data and award records.")
async def get_positions_by_country_award(birth_country: str = Query(..., description="Birth country of the player"), award: str = Query(..., description="Award received by the player")):
    cursor.execute("SELECT DISTINCT T1.pos FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID WHERE T1.birthCountry = ? AND T2.award = ?", (birth_country, award))
    result = cursor.fetchall()
    if not result:
        return {"positions": []}
    return {"positions": [row[0] for row in result]}

# Endpoint to get the average BMI of Hall of Fame players
@app.get("/v1/hockey/average_bmi_hof_players", operation_id="get_average_bmi_hof_players", summary="Retrieves the average Body Mass Index (BMI) of all Hall of Fame (HOF) players. The calculation is based on the sum of each player's weight divided by the square of their height, then divided by the total number of HOF players.")
async def get_average_bmi_hof_players():
    cursor.execute("SELECT SUM(T1.weight / (T1.height * T1.height)) / COUNT(T1.coachID) FROM Master AS T1 INNER JOIN HOF AS T2 ON T1.hofID = T2.hofID")
    result = cursor.fetchone()
    if not result:
        return {"average_bmi": []}
    return {"average_bmi": result[0]}

# Endpoint to get the percentage of Hall of Fame players from a specific country
@app.get("/v1/hockey/percentage_hof_players_by_country", operation_id="get_percentage_hof_players_by_country", summary="Retrieves the percentage of players from a specific country who have been inducted into the Hall of Fame. The calculation is based on the total number of players from the specified country and the total number of Hall of Fame inductees.")
async def get_percentage_hof_players_by_country(birth_country: str = Query(..., description="Birth country of the player")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.birthCountry = ? THEN T1.playerID ELSE NULL END) AS REAL) * 100 / COUNT(T1.playerID) FROM Master AS T1 INNER JOIN HOF AS T2 ON T1.hofID = T2.hofID", (birth_country,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of years a goalie played
@app.get("/v1/hockey/goalie_years_count", operation_id="get_goalie_years_count", summary="Retrieves the total number of years a specific goalie has played in the league. The goalie is identified by their unique playerID.")
async def get_goalie_years_count(playerID: str = Query(..., description="Player ID of the goalie")):
    cursor.execute("SELECT COUNT(year) FROM Goalies WHERE playerID = ?", (playerID,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the team ID of a goalie in a specific year
@app.get("/v1/hockey/goalie_team_id", operation_id="get_goalie_team_id", summary="Retrieves the team ID of a specific goalie for a given year. The goalie is identified by the provided player ID, and the year parameter determines the season for which the team ID is returned.")
async def get_goalie_team_id(playerID: str = Query(..., description="Player ID of the goalie"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT tmID FROM Goalies WHERE playerID = ? AND year = ?", (playerID, year))
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get the games played by a goalie in a specific year
@app.get("/v1/hockey/goalie_games_played", operation_id="get_goalie_games_played", summary="Retrieves the number of games played by a specific goalie in a given year. The goalie is identified by a unique player ID, and the year is specified as a four-digit integer. This operation provides a snapshot of a goalie's activity for a particular season.")
async def get_goalie_games_played(playerID: str = Query(..., description="Player ID of the goalie"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT GP FROM Goalies WHERE playerID = ? AND year = ?", (playerID, year))
    result = cursor.fetchone()
    if not result:
        return {"games_played": []}
    return {"games_played": result[0]}

# Endpoint to get the minutes played by a goalie in a specific year
@app.get("/v1/hockey/goalie_minutes_played", operation_id="get_goalie_minutes_played", summary="Retrieves the total minutes played by a specific goalie in a given year. The goalie is identified by a unique player ID, and the year is specified as a four-digit integer. This operation provides a precise measure of a goalie's playing time, which can be useful for performance analysis and comparisons.")
async def get_goalie_minutes_played(playerID: str = Query(..., description="Player ID of the goalie"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT Min FROM Goalies WHERE playerID = ? AND year = ?", (playerID, year))
    result = cursor.fetchone()
    if not result:
        return {"minutes_played": []}
    return {"minutes_played": result[0]}

# Endpoint to get the wins of a goalie in a specific year
@app.get("/v1/hockey/goalie_wins", operation_id="get_goalie_wins", summary="Retrieves the total number of wins for a specific hockey goalie in a given year. The operation requires the unique identifier of the goalie (playerID) and the year for which the wins are to be fetched.")
async def get_goalie_wins(playerID: str = Query(..., description="Player ID of the goalie"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT W FROM Goalies WHERE playerID = ? AND year = ?", (playerID, year))
    result = cursor.fetchone()
    if not result:
        return {"wins": []}
    return {"wins": result[0]}

# Endpoint to get the ties/overtime losses of a goalie in a specific year
@app.get("/v1/hockey/goalie_ties_overtime_losses", operation_id="get_goalie_ties_overtime_losses", summary="Retrieves the number of ties and overtime losses for a specific goalie in a given year. The goalie is identified by the provided playerID, and the year is specified by the year parameter.")
async def get_goalie_ties_overtime_losses(playerID: str = Query(..., description="Player ID of the goalie"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT `T/OL` FROM Goalies WHERE playerID = ? AND year = ?", (playerID, year))
    result = cursor.fetchone()
    if not result:
        return {"ties_overtime_losses": []}
    return {"ties_overtime_losses": result[0]}

# Endpoint to get the sum of wins by coaches who received a specific award in a given year
@app.get("/v1/hockey/sum_wins_coaches_award_year", operation_id="get_sum_wins_coaches_award_year", summary="Retrieves the total number of wins accumulated by coaches who were honored with a specific award in a given year. The operation requires the year and the name of the award as input parameters.")
async def get_sum_wins_coaches_award_year(year: int = Query(..., description="Year"), award: str = Query(..., description="Award")):
    cursor.execute("SELECT SUM(T1.W) FROM Coaches AS T1 INNER JOIN AwardsCoaches AS T2 ON T1.coachID = T2.coachID WHERE T2.year = ? AND T2.award = ?", (year, award))
    result = cursor.fetchone()
    if not result:
        return {"sum_wins": []}
    return {"sum_wins": result[0]}

# Endpoint to check if a player has a specific note in AwardsMisc
@app.get("/v1/hockey/check_player_note", operation_id="check_player_note", summary="This operation verifies if a specific note is associated with a player in the database. It uses the provided legendsID to identify the player and the note parameter to check for a match. The operation returns 'YES' if the note is found, and 'NO' otherwise.")
async def check_player_note(note: str = Query(..., description="Note to check"), legendsID: str = Query(..., description="Legends ID of the player")):
    cursor.execute("SELECT IIF(T1.note = ?, 'YES', 'NO') FROM AwardsMisc AS T1 RIGHT JOIN Master AS T2 ON T1.ID = T2.playerID WHERE T2.legendsID = ?", (note, legendsID))
    result = cursor.fetchone()
    if not result:
        return {"has_note": []}
    return {"has_note": result[0]}

# Endpoint to get the position of a player based on first and last name
@app.get("/v1/hockey/player_position", operation_id="get_player_position", summary="Retrieves the position of a specific hockey player based on their first and last name. The operation uses the provided names to search for a match in the Master and AwardsPlayers tables, returning the position of the player if a match is found.")
async def get_player_position(firstName: str = Query(..., description="First name of the player"), lastName: str = Query(..., description="Last name of the player")):
    cursor.execute("SELECT T1.pos FROM Master AS T1 INNER JOIN AwardsPlayers AS T2 ON T1.playerID = T2.playerID WHERE T1.firstName = ? AND T1.lastName = ?", (firstName, lastName))
    result = cursor.fetchone()
    if not result:
        return {"position": []}
    return {"position": result[0]}

# Endpoint to get the birth country of coaches based on year and notes
@app.get("/v1/hockey/coach_birth_country", operation_id="get_coach_birth_country", summary="Retrieves the birth country of coaches who were active in a specific year and have certain notes associated with them. The year and notes are used to filter the coaches and determine the birth country.")
async def get_coach_birth_country(year: int = Query(..., description="Year"), notes: str = Query(..., description="Notes")):
    cursor.execute("SELECT T1.birthCountry FROM Master AS T1 INNER JOIN Coaches AS T2 ON T1.coachID = T2.coachID WHERE T2.year = ? AND T2.notes = ?", (year, notes))
    result = cursor.fetchone()
    if not result:
        return {"birth_country": []}
    return {"birth_country": result[0]}

# Endpoint to get the first and last name of a goalie based on stint
@app.get("/v1/hockey/goalie_name_by_stint", operation_id="get_goalie_name_by_stint", summary="Retrieves the first and last name of the heaviest goalie who played during the specified stint. The stint is a period of time during which a goalie played for a particular team.")
async def get_goalie_name_by_stint(stint: int = Query(..., description="Stint of the goalie")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.stint = ? ORDER BY T1.weight DESC LIMIT 1", (stint,))
    result = cursor.fetchone()
    if not result:
        return {"goalie": []}
    return {"goalie": {"firstName": result[0], "lastName": result[1]}}

# Endpoint to get the first and last name of a goalie based on the sum of ENG
@app.get("/v1/hockey/goalie_name_by_eng_sum", operation_id="get_goalie_name_by_eng_sum", summary="Retrieves the full name of the tallest goalie with a total ENG value surpassing the provided threshold. The ENG value is a performance metric specific to goalies.")
async def get_goalie_name_by_eng_sum(eng_sum: int = Query(..., description="Sum of ENG for the goalie")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID GROUP BY T2.playerID, T1.height HAVING SUM(T2.ENG) > ? ORDER BY T1.height DESC LIMIT 1", (eng_sum,))
    result = cursor.fetchone()
    if not result:
        return {"goalie": []}
    return {"goalie": {"firstName": result[0], "lastName": result[1]}}

# Endpoint to get the shootCatch of a goalie based on the year
@app.get("/v1/hockey/goalie_shootcatch_by_year", operation_id="get_goalie_shootcatch_by_year", summary="Retrieves the shootCatch statistic of the goalie who had the highest total shots on goal (SHO) in the specified year. The data is obtained by joining the Master and Goalies tables on the playerID field and filtering for the given year. The results are then grouped by playerID and ordered in descending order based on the sum of SHO. The top result is returned.")
async def get_goalie_shootcatch_by_year(year: int = Query(..., description="Year of the goalie")):
    cursor.execute("SELECT T1.shootCatch FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? GROUP BY T2.playerID ORDER BY SUM(T2.SHO) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"shootCatch": []}
    return {"shootCatch": result[0]}

# Endpoint to get the first and last name of a goalie based on year and goals against
@app.get("/v1/hockey/goalie_name_by_year_and_ga", operation_id="get_goalie_name_by_year_and_ga", summary="Retrieves the full name of the goalie who had the most goals against in a specific year. The goalie's year and goals against are used as input parameters to filter the results. The data is sorted by the goalie's birth year, birth month, and birth day in descending order to ensure the most recent goalie is returned.")
async def get_goalie_name_by_year_and_ga(year: int = Query(..., description="Year of the goalie"), ga: int = Query(..., description="Goals against for the goalie")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? AND T2.GA > ? GROUP BY T2.playerID, T1.birthYear, T1.birthMon, T1.birthMon HAVING SUM(T2.GA) ORDER BY T1.birthYear DESC, T1.birthMon DESC, SUM(T1.birthDay) DESC LIMIT 1", (year, ga))
    result = cursor.fetchone()
    if not result:
        return {"goalie": []}
    return {"goalie": {"firstName": result[0], "lastName": result[1]}}

# Endpoint to get the first and last name of a goalie based on team ID
@app.get("/v1/hockey/goalie_name_by_team_id", operation_id="get_goalie_name_by_team_id", summary="Retrieves the first and last name of the goalie with the highest save percentage from the team specified by the provided team ID.")
async def get_goalie_name_by_team_id(tmID: str = Query(..., description="Team ID of the goalie")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.tmID = ? GROUP BY T2.playerID ORDER BY SUM(T2.SA - T2.GA) DESC LIMIT 1", (tmID,))
    result = cursor.fetchone()
    if not result:
        return {"goalie": []}
    return {"goalie": {"firstName": result[0], "lastName": result[1]}}

# Endpoint to get the team name based on the year
@app.get("/v1/hockey/team_name_by_year", operation_id="get_team_name_by_year", summary="Retrieves the name of the team that had the highest total of PostENG (post-engagement) in the specified year. The team name is obtained by joining the Goalies and Teams tables on the tmID field and filtering for the given year. The results are then ordered in descending order based on the sum of PostENG and limited to the top record.")
async def get_team_name_by_year(year: int = Query(..., description="Year of the team")):
    cursor.execute("SELECT T2.name FROM Goalies AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID WHERE T1.year = ? GROUP BY T2.name ORDER BY SUM(PostENG) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result[0]}

# Endpoint to get the sum of SHO based on the year
@app.get("/v1/hockey/sum_sho_by_year", operation_id="get_sum_sho_by_year", summary="Retrieves the total number of shutouts (SHO) for the top-performing team in a specified year. The calculation is based on the combined shutouts of all goalies in the team for that year.")
async def get_sum_sho_by_year(year: int = Query(..., description="Year of the goalie")):
    cursor.execute("SELECT SUM(T2.SHO) FROM Scoring AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? GROUP BY T2.tmID ORDER BY SUM(T2.PostSHO) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"sum_sho": []}
    return {"sum_sho": result[0]}

# Endpoint to get the coach ID and nameNick based on the year
@app.get("/v1/hockey/coach_id_name_nick_by_year", operation_id="get_coach_id_name_nick_by_year", summary="Retrieves the ID and nickname of the coach with the highest win-loss ratio for the specified year. The coach's data is fetched from the Master and Coaches tables, which are joined based on the coach ID. The results are ordered by the win-loss ratio in descending order, and only the top record is returned.")
async def get_coach_id_name_nick_by_year(year: int = Query(..., description="Year of the coach")):
    cursor.execute("SELECT T2.coachID, T1.nameNick FROM Master AS T1 INNER JOIN Coaches AS T2 ON T1.coachID = T2.coachID WHERE T2.year = ? ORDER BY CAST(T2.W AS REAL) / (T2.W + T2.L) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"coach": []}
    return {"coach": {"coachID": result[0], "nameNick": result[1]}}

# Endpoint to get the coach ID based on the year
@app.get("/v1/hockey/coach_id_by_year", operation_id="get_coach_id_by_year", summary="Retrieves the ID of the coach who led the team with the highest points per game (PPG) to points conceded (PPC) ratio in the specified year. The coach's team must be present in the Coaches and Teams tables, with a matching team ID (tmID) in both tables.")
async def get_coach_id_by_year(year: int = Query(..., description="Year of the coach")):
    cursor.execute("SELECT T1.coachID FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID WHERE T2.year = ? ORDER BY CAST(T2.PPG AS REAL) / T2.PPC DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"coachID": []}
    return {"coachID": result[0]}

# Endpoint to get the games played (GP) based on player ID and year
@app.get("/v1/hockey/games_played_by_player_id_and_year", operation_id="get_games_played_by_player_id_and_year", summary="Retrieves the number of games played by a specific hockey player in a given year. The operation requires the player's unique ID and the year to filter the results.")
async def get_games_played_by_player_id_and_year(playerID: str = Query(..., description="Player ID"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT GP FROM ScoringSC WHERE playerID = ? AND YEAR = ?", (playerID, year))
    result = cursor.fetchone()
    if not result:
        return {"GP": []}
    return {"GP": result[0]}

# Endpoint to get the count of years for a specific player
@app.get("/v1/hockey/count_years_by_player", operation_id="get_count_years_by_player", summary="Retrieves the total number of years a specific player has played, as determined by the provided playerID.")
async def get_count_years_by_player(playerID: str = Query(..., description="Player ID")):
    cursor.execute("SELECT COUNT(year) FROM ScoringSC WHERE playerID = ?", (playerID,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the goals for a specific player in a specific year
@app.get("/v1/hockey/goals_by_player_year", operation_id="get_goals_by_player_year", summary="Retrieves the total number of goals scored by a specific player during a particular year. The operation requires the player's unique identifier and the year of interest as input parameters.")
async def get_goals_by_player_year(playerID: str = Query(..., description="Player ID"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT G FROM ScoringSC WHERE playerID = ? AND year = ?", (playerID, year))
    result = cursor.fetchall()
    if not result:
        return {"goals": []}
    return {"goals": [row[0] for row in result]}

# Endpoint to get the player with the highest assists
@app.get("/v1/hockey/top_assists_player", operation_id="get_top_assists_player", summary="Retrieves the unique identifier of the player who has the highest number of assists in the ScoringSC table. The data is sorted in descending order based on the number of assists, and only the top result is returned.")
async def get_top_assists_player():
    cursor.execute("SELECT playerID FROM ScoringSC ORDER BY A DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"playerID": []}
    return {"playerID": result[0]}

# Endpoint to get the league ID for a specific player in a specific year
@app.get("/v1/hockey/league_id_by_player_year", operation_id="get_league_id_by_player_year", summary="Retrieves the league ID associated with a particular player for a given year. The operation requires the player's unique identifier and the year of interest as input parameters.")
async def get_league_id_by_player_year(playerID: str = Query(..., description="Player ID"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT lgID FROM ScoringSC WHERE playerID = ? AND year = ?", (playerID, year))
    result = cursor.fetchall()
    if not result:
        return {"leagueID": []}
    return {"leagueID": [row[0] for row in result]}

# Endpoint to get distinct positions for a specific player
@app.get("/v1/hockey/distinct_positions_by_player", operation_id="get_distinct_positions_by_player", summary="Retrieves the unique positions that a specific player has played in. The player is identified by the provided playerID.")
async def get_distinct_positions_by_player(playerID: str = Query(..., description="Player ID")):
    cursor.execute("SELECT DISTINCT pos FROM ScoringSC WHERE playerID = ?", (playerID,))
    result = cursor.fetchall()
    if not result:
        return {"positions": []}
    return {"positions": [row[0] for row in result]}

# Endpoint to get the sum of wins for teams with a specific number of distinct goalies in a specific year
@app.get("/v1/hockey/sum_wins_by_team_goalies_year", operation_id="get_sum_wins_by_team_goalies_year", summary="Retrieves the total number of wins for teams that had a specific number of unique goalies in a given year. The operation filters teams by the provided year and counts the distinct goalies for each team. It then calculates the sum of wins for teams that meet the specified count of unique goalies.")
async def get_sum_wins_by_team_goalies_year(year: int = Query(..., description="Year"), distinct_goalies: int = Query(..., description="Number of distinct goalies")):
    cursor.execute("SELECT SUM(T2.W) FROM Goalies AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID WHERE T2.year = ? GROUP BY T1.tmID HAVING COUNT(DISTINCT T1.playerID) = ?", (year, distinct_goalies))
    result = cursor.fetchall()
    if not result:
        return {"sum_wins": []}
    return {"sum_wins": [row[0] for row in result]}

# Endpoint to get the birth year of the goalie with the highest PostSA in a specific year
@app.get("/v1/hockey/birth_year_top_postsa_goalie", operation_id="get_birth_year_top_postsa_goalie", summary="Retrieves the birth year of the goalie who had the highest PostSA (Post Shot Attempts) in the specified year. The PostSA is a metric that measures the number of shots a goalie has faced, including those that missed the net or were blocked. The goalie with the highest PostSA in a given year is considered to have faced the most shots, which can be an indicator of their workload and performance.")
async def get_birth_year_top_postsa_goalie(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.birthYear FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? ORDER BY T2.PostSA DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"birthYear": []}
    return {"birthYear": result[0]}

# Endpoint to get the coaching career span for a specific coach
@app.get("/v1/hockey/coaching_career_span", operation_id="get_coaching_career_span", summary="Retrieves the total span of a coach's career in years. This operation calculates the difference between the earliest and latest years a coach has been active, based on the provided first and last names. The result provides a comprehensive view of the coach's career length.")
async def get_coaching_career_span(firstName: str = Query(..., description="First name of the coach"), lastName: str = Query(..., description="Last name of the coach")):
    cursor.execute("SELECT MAX(T2.year) - MIN(T2.year) FROM Master AS T1 INNER JOIN Coaches AS T2 ON T1.coachID = T2.coachID WHERE T1.firstName = ? AND T1.lastName = ?", (firstName, lastName))
    result = cursor.fetchone()
    if not result:
        return {"career_span": []}
    return {"career_span": result[0]}

# Endpoint to get the shoot catch of the goalie with the highest SHO in a specific year
@app.get("/v1/hockey/shoot_catch_top_sho_goalie", operation_id="get_shoot_catch_top_sho_goalie", summary="Retrieves the shoot catch technique of the goalie who had the highest number of shutouts (SHO) in a given year.")
async def get_shoot_catch_top_sho_goalie(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.shootCatch FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? ORDER BY T2.SHO DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"shootCatch": []}
    return {"shootCatch": result[0]}

# Endpoint to get the birth details of the goalie with the highest GA in a specific year
@app.get("/v1/hockey/birth_details_top_ga_goalie", operation_id="get_birth_details_top_ga_goalie", summary="Retrieve the birth details of the goalie who had the highest goals against (GA) in a specified year. The birth details include the birth year, month, and day.")
async def get_birth_details_top_ga_goalie(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.birthYear, T1.birthMon, birthDay FROM Master AS T1 INNER JOIN Goalies AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? ORDER BY T2.GA DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"birth_details": []}
    return {"birth_details": {"birthYear": result[0], "birthMon": result[1], "birthDay": result[2]}}

# Endpoint to get the player with the highest +/- score in a given year
@app.get("/v1/hockey/top_player_by_plus_minus", operation_id="get_top_player_by_plus_minus", summary="Retrieves the name of the player with the highest +/- score in the specified year. The +/- score is calculated as the difference between the number of goals scored by the player's team and the number of goals scored by the opposing team while the player is on the ice. The player's first and last names are returned.")
async def get_top_player_by_plus_minus(year: int = Query(..., description="Year to filter the players")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN Scoring AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? GROUP BY T2.playerID ORDER BY SUM(T2.`+/-`) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": {"firstName": result[0], "lastName": result[1]}}

# Endpoint to get the weight of the player with the highest PPG after a given year
@app.get("/v1/hockey/top_player_weight_by_ppg", operation_id="get_top_player_weight_by_ppg", summary="Retrieves the weight of the player with the highest points per game (PPG) in the years following the specified year. The data is filtered by year and aggregated by player and weight, then ordered by the sum of PPG in descending order. The result is limited to the top record.")
async def get_top_player_weight_by_ppg(year: int = Query(..., description="Year to filter the players")):
    cursor.execute("SELECT T1.weight FROM Master AS T1 INNER JOIN Scoring AS T2 ON T1.playerID = T2.playerID WHERE T2.year > ? GROUP BY T1.playerID, T1.weight ORDER BY SUM(T2.PPG) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"weight": []}
    return {"weight": result[0]}

# Endpoint to get the shoot/catch preference of players with a specific number of SHG in a given year
@app.get("/v1/hockey/shoot_catch_by_shg", operation_id="get_shoot_catch_by_shg", summary="Retrieves the preferred shooting and catching hand of players who scored a specific number of shorthanded goals (SHG) in a given year. The data is filtered by the provided year and the total number of SHG scored by each player.")
async def get_shoot_catch_by_shg(year: int = Query(..., description="Year to filter the players"), shg_count: int = Query(..., description="Number of SHG to filter the players")):
    cursor.execute("SELECT T1.shootCatch FROM Master AS T1 INNER JOIN Scoring AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? GROUP BY T2.playerID HAVING SUM(T2.SHG) = ?", (year, shg_count))
    result = cursor.fetchall()
    if not result:
        return {"shootCatch": []}
    return {"shootCatch": [row[0] for row in result]}

# Endpoint to get the player with the highest GWG in a given year
@app.get("/v1/hockey/top_player_by_gwg", operation_id="get_top_player_by_gwg", summary="Retrieves the top player with the highest number of game-winning goals (GWG) in a specified year. The player's first and last names are returned. The input parameter 'year' is used to filter the players.")
async def get_top_player_by_gwg(year: int = Query(..., description="Year to filter the players")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM Master AS T1 INNER JOIN Scoring AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? GROUP BY T2.playerID ORDER BY SUM(T2.GWG) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": {"firstName": result[0], "lastName": result[1]}}

# Endpoint to get the SOG of a specific player in a given year
@app.get("/v1/hockey/player_sog", operation_id="get_player_sog", summary="Retrieves the number of shots on goal (SOG) for a specific hockey player in a given year. The operation requires the player's first and last name, as well as the year to filter the results. The data is fetched from a database that contains player information and scoring statistics.")
async def get_player_sog(first_name: str = Query(..., description="First name of the player"), last_name: str = Query(..., description="Last name of the player"), year: int = Query(..., description="Year to filter the player")):
    cursor.execute("SELECT T2.SOG FROM Master AS T1 INNER JOIN Scoring AS T2 ON T1.playerID = T2.playerID WHERE T1.firstName = ? AND T1.lastName = ? AND T2.year = ?", (first_name, last_name, year))
    result = cursor.fetchone()
    if not result:
        return {"SOG": []}
    return {"SOG": result[0]}

# Endpoint to get the coach with the highest BenchMinor in a given year
@app.get("/v1/hockey/top_coach_by_bench_minor", operation_id="get_top_coach_by_bench_minor", summary="Retrieves the coach with the highest cumulative BenchMinor penalty minutes in the specified year. The operation filters coaches based on the provided year and returns the first name and last name of the coach with the highest total BenchMinor penalty minutes.")
async def get_top_coach_by_bench_minor(year: int = Query(..., description="Year to filter the coaches")):
    cursor.execute("SELECT DISTINCT T3.firstName, T3.lastName FROM Teams AS T1 INNER JOIN Coaches AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year INNER JOIN Master AS T3 ON T2.coachID = T3.coachID WHERE T1.year = ? GROUP BY T3.firstName, T3.lastName ORDER BY SUM(T1.BenchMinor) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"coach": []}
    return {"coach": {"firstName": result[0], "lastName": result[1]}}

# Endpoint to get the goalie with the highest GA in a given year and weight
@app.get("/v1/hockey/top_goalie_by_ga", operation_id="get_top_goalie_by_ga", summary="Retrieves the goalie with the highest goals against (GA) in a given year, considering only goalies who weigh more than a specified weight. The result is filtered by the provided year and weight parameters.")
async def get_top_goalie_by_ga(year: int = Query(..., description="Year to filter the goalies"), weight: int = Query(..., description="Weight to filter the goalies")):
    cursor.execute("SELECT T1.playerID FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T1.year = ? AND T2.weight > ? ORDER BY T1.GA DESC LIMIT 1", (year, weight))
    result = cursor.fetchone()
    if not result:
        return {"playerID": []}
    return {"playerID": result[0]}

# Endpoint to get the win percentage difference for a team between two years with a specific coach
@app.get("/v1/hockey/win_percentage_difference", operation_id="get_win_percentage_difference", summary="Retrieve the difference in win percentage for a specific team between two given years, under the guidance of a particular coach. The calculation considers the total wins and losses of the team in each year.")
async def get_win_percentage_difference(year1: int = Query(..., description="First year to compare"), year2: int = Query(..., description="Second year to compare"), team_name: str = Query(..., description="Name of the team"), coach_first_name: str = Query(..., description="First name of the coach"), coach_last_name: str = Query(..., description="Last name of the coach")):
    cursor.execute("SELECT SUM(CASE WHEN T1.year = ? THEN CAST(T1.W AS REAL) * 100 / (T1.W + T1.L) ELSE 0 END) - ( SELECT CAST(W AS REAL) * 100 / (W + L) FROM Teams WHERE year = ? AND name = ? ) FROM Teams AS T1 INNER JOIN Coaches AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year INNER JOIN Master AS T3 ON T2.coachID = T3.coachID WHERE T1.name = ? AND T3.firstName = ? AND T3.lastName = ?", (year1, year2, team_name, team_name, coach_first_name, coach_last_name))
    result = cursor.fetchone()
    if not result:
        return {"win_percentage_difference": []}
    return {"win_percentage_difference": result[0]}

# Endpoint to get the legendsID of the goalie with the highest save percentage in a given year
@app.get("/v1/hockey/top_goalie_by_save_percentage", operation_id="get_top_goalie_by_save_percentage", summary="Retrieves the unique identifier of the goalie with the highest save percentage in a specific year. The save percentage is calculated as the ratio of saves to shots against. The year parameter is used to filter the goalies.")
async def get_top_goalie_by_save_percentage(year: int = Query(..., description="Year to filter the goalies")):
    cursor.execute("SELECT T2.legendsID FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T1.year = ? ORDER BY 1 - CAST(T1.PostGA AS REAL) / T1.PostSA DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"legendsID": []}
    return {"legendsID": result[0]}

# Endpoint to get the count of players who are also coaches
@app.get("/v1/hockey/player_coach_count", operation_id="get_player_coach_count", summary="Retrieves the total number of individuals who are both players and coaches in the hockey league. This operation does not require any input parameters and returns a single integer value representing the count.")
async def get_player_coach_count():
    cursor.execute("SELECT COUNT(playerID) FROM Master WHERE playerID IS NOT NULL AND coachID IS NOT NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get team details with the highest T value
@app.get("/v1/hockey/team_details_highest_t", operation_id="get_team_details_highest_t", summary="Retrieves the team details of the team with the highest T value. The response includes the team's name and the number of bench minor penalties. The limit parameter can be used to restrict the number of results returned.")
async def get_team_details_highest_t(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT BenchMinor, name FROM Teams ORDER BY T DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the most awarded coach in a specific league
@app.get("/v1/hockey/most_awarded_coach", operation_id="get_most_awarded_coach", summary="Retrieves the coach with the most awards in a specific league. The operation accepts a league ID to filter the results and a limit parameter to restrict the number of awards returned. The data is sorted in descending order based on the number of awards received by each coach.")
async def get_most_awarded_coach(lgID: str = Query(..., description="League ID"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT award FROM Teams AS T1 INNER JOIN AwardsCoaches AS T2 ON T1.lgID = T2.lgID WHERE T1.lgID = ? GROUP BY T2.coachID, T2.award ORDER BY COUNT(T2.award) DESC LIMIT ?", (lgID, limit))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": result}

# Endpoint to get the team with the highest PPG to PPC ratio
@app.get("/v1/hockey/highest_ppg_ppc_ratio", operation_id="get_highest_ppg_ppc_ratio", summary="Retrieve the team with the highest ratio of points per game (PPG) to points per chance (PPC). The results are ordered by the number of losses in descending order. The limit parameter can be used to restrict the number of results returned.")
async def get_highest_ppg_ppc_ratio(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT CAST(SUM(PPG) AS REAL) * 100 / SUM(PPC) FROM Teams GROUP BY tmID ORDER BY SUM(L) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"ratio": []}
    return {"ratio": result}

# Endpoint to get the proportion of Hall of Fame inductees in a specific category and year range
@app.get("/v1/hockey/hof_inductees_proportion", operation_id="get_hof_inductees_proportion", summary="Retrieves the proportion of Hall of Fame inductees in a specific category within a given year range. The calculation divides the count of inductees by a provided divisor. The category and year range are also specified as input parameters.")
async def get_hof_inductees_proportion(divisor: int = Query(..., description="Divisor for the proportion calculation"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), category: str = Query(..., description="Category of the inductee")):
    cursor.execute("SELECT CAST(COUNT(name) AS REAL) / ? FROM HOF WHERE year BETWEEN ? AND ? AND category = ?", (divisor, start_year, end_year, category))
    result = cursor.fetchall()
    if not result:
        return {"proportion": []}
    return {"proportion": result}

# Endpoint to get the most common birth country and year combination
@app.get("/v1/hockey/most_common_birth_country_year", operation_id="get_most_common_birth_country_year", summary="Retrieves the most frequently occurring combination of birth country and year from the database. The results are ordered by frequency in descending order and can be limited by specifying the maximum number of results to return.")
async def get_most_common_birth_country_year(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT birthCountry, birthYear FROM Master GROUP BY birthCountry, birthYear ORDER BY COUNT(birthCountry) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"birth_details": []}
    return {"birth_details": result}

# Endpoint to get the team with the highest number of shutouts
@app.get("/v1/hockey/team_highest_shutouts", operation_id="get_team_highest_shutouts", summary="Retrieves the team with the highest total number of shutouts, up to the specified limit. This operation returns the team with the most games where no goals were conceded, providing a measure of their defensive performance.")
async def get_team_highest_shutouts(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT SUM(W) FROM Goalies GROUP BY tmID ORDER BY SUM(SHO) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"shutouts": []}
    return {"shutouts": result}

# Endpoint to get the team with the highest regular season wins in a specific year
@app.get("/v1/hockey/team_highest_regular_wins", operation_id="get_team_highest_regular_wins", summary="Retrieves the team with the highest number of regular season wins for a specified year. The operation allows limiting the number of results returned. The data is filtered by the provided year and sorted in descending order based on regular season wins.")
async def get_team_highest_regular_wins(year: int = Query(..., description="Year of the regular season"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT tmID FROM TeamSplits WHERE YEAR = ? ORDER BY rW DESC LIMIT ?", (year, limit))
    result = cursor.fetchall()
    if not result:
        return {"team_ids": []}
    return {"team_ids": result}

# Endpoint to get player positions ordered by birth date
@app.get("/v1/hockey/player_positions_by_birth_date", operation_id="get_player_positions_by_birth_date", summary="Retrieves the positions of hockey players, ordered by their birth dates. The operation returns a limited number of results based on the provided input parameter. This endpoint is useful for obtaining a ranked list of player positions based on their birth dates.")
async def get_player_positions_by_birth_date(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT pos FROM Master WHERE birthYear IS NOT NULL ORDER BY birthYear, birthMon, birthDay LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"positions": []}
    return {"positions": result}

# Endpoint to get the goalie with the highest post-game goals against in a specific league and year
@app.get("/v1/hockey/goalie_highest_post_ga", operation_id="get_goalie_highest_post_ga", summary="Retrieves the goalie with the highest cumulative post-game goals against in a specified league and year. The operation returns the goalie with the highest sum of post-game goals against, calculated by aggregating the PostGA column for the given league and year. The results can be limited to a specific number of goalies.")
async def get_goalie_highest_post_ga(lgID: str = Query(..., description="League ID"), year: int = Query(..., description="Year"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT SUM(PostGA) FROM Goalies WHERE lgID = ? AND year = ? GROUP BY playerID ORDER BY SUM(PostGA) DESC LIMIT ?", (lgID, year, limit))
    result = cursor.fetchall()
    if not result:
        return {"post_ga": []}
    return {"post_ga": result}

# Endpoint to get the team with the highest October losses in a specific year
@app.get("/v1/hockey/team_highest_october_losses", operation_id="get_team_highest_october_losses", summary="Retrieves the team with the highest number of losses in October for a given year. The operation allows you to limit the number of results returned. The team is identified by its unique ID.")
async def get_team_highest_october_losses(year: int = Query(..., description="Year"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT OctL, tmID FROM TeamSplits WHERE year = ? ORDER BY OctL DESC LIMIT ?", (year, limit))
    result = cursor.fetchall()
    if not result:
        return {"october_losses": []}
    return {"october_losses": result}

# Endpoint to get the count of players based on first NHL year
@app.get("/v1/hockey/count_players_first_nhl_year", operation_id="get_count_players_first_nhl_year", summary="Retrieves the total number of players who commenced their National Hockey League (NHL) careers in a specified year. The input parameter determines the year to consider for the count.")
async def get_count_players_first_nhl_year(first_nhl: str = Query(..., description="First NHL year of the player")):
    cursor.execute("SELECT COUNT(playerID) FROM Master WHERE shootCatch IS NULL AND firstNHL = ?", (first_nhl,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the year and age of the player with the highest goals in a season
@app.get("/v1/hockey/top_goal_scorer_year_age", operation_id="get_top_goal_scorer_year_age", summary="Retrieves the year and age of the player who scored the most goals in a single season. The data is obtained by aggregating and sorting the total goals scored by each player in a season, then selecting the top scorer. The age is calculated as the difference between the season year and the player's birth year.")
async def get_top_goal_scorer_year_age():
    cursor.execute("SELECT T1.year, T1.year - T2.birthYear FROM Scoring AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID GROUP BY T1.year, T1.year - T2.birthYear ORDER BY SUM(T1.G) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"year": [], "age": []}
    return {"year": result[0], "age": result[1]}

# Endpoint to get the average height of players in specific positions
@app.get("/v1/hockey/average_height_by_position", operation_id="get_average_height_by_position", summary="Retrieves the average height of hockey players who have been awarded and play in either the first or second specified position. The calculation is based on the sum of the heights of all qualifying players divided by their count.")
async def get_average_height_by_position(pos1: str = Query(..., description="First position"), pos2: str = Query(..., description="Second position")):
    cursor.execute("SELECT CAST(SUM(T2.height) AS REAL) / COUNT(*) FROM AwardsPlayers AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T2.height IS NOT NULL AND (T2.pos = ? OR T2.pos = ?)", (pos1, pos2))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the top assists scorer in a specific league
@app.get("/v1/hockey/top_assists_scorer", operation_id="get_top_assists_scorer", summary="Retrieves the player with the highest total assists in a specific league. The league is identified by its unique ID. The response includes the player's first and last name, along with the total number of assists they have made in the specified league.")
async def get_top_assists_scorer(lg_id: str = Query(..., description="League ID")):
    cursor.execute("SELECT SUM(T1.A), T2.firstName, T2.lastName FROM Scoring AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T1.lgID = ? GROUP BY T2.firstName, T2.lastName ORDER BY SUM(T1.A) DESC LIMIT 1", (lg_id,))
    result = cursor.fetchone()
    if not result:
        return {"assists": [], "first_name": [], "last_name": []}
    return {"assists": result[0], "first_name": result[1], "last_name": result[2]}

# Endpoint to get the top coach by wins and their award
@app.get("/v1/hockey/top_coach_by_wins", operation_id="get_top_coach_by_wins", summary="Retrieves the coach with the highest number of wins and the corresponding award. The coach's ID and award are returned, with the coach being selected based on the total sum of their wins.")
async def get_top_coach_by_wins():
    cursor.execute("SELECT DISTINCT T2.coachID, T1.award FROM AwardsCoaches AS T1 INNER JOIN Coaches AS T2 ON T1.coachID = T2.coachID GROUP BY T2.coachID, T1.award ORDER BY SUM(T2.w) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"coach_id": [], "award": []}
    return {"coach_id": result[0], "award": result[1]}

# Endpoint to get the team name based on series note and year
@app.get("/v1/hockey/team_name_by_series_note_year", operation_id="get_team_name_by_series_note_year", summary="Retrieves the name of the team that lost a specific series in a given year. The series is identified by its note, and the year is used to filter the results.")
async def get_team_name_by_series_note_year(note: str = Query(..., description="Series note"), year: str = Query(..., description="Year")):
    cursor.execute("SELECT T2.name FROM SeriesPost AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND tmIDLoser = tmID WHERE T1.note = ? AND T2.year = ?", (note, year))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the player with the lowest plus-minus rating
@app.get("/v1/hockey/lowest_plus_minus_player", operation_id="get_lowest_plus_minus_player", summary="Retrieves the player with the lowest cumulative plus-minus rating in the National Hockey League (NHL). The plus-minus rating is calculated based on the player's performance in each season, considering the number of goals scored and conceded while they were on the ice. The response includes the player's first and last names, given name, and the year they first played in the NHL.")
async def get_lowest_plus_minus_player():
    cursor.execute("SELECT DISTINCT T3.firstNHL - T1.year, T3.nameGiven , T3.firstName, T3.lastName FROM Scoring AS T1 INNER JOIN Teams AS T2 ON T2.tmID = T1.tmID INNER JOIN Master AS T3 ON T1.playerID = T3.playerID GROUP BY T3.firstName, T3.lastName, T3.nameGiven, T3.firstNHL - T1.year, T3.firstName, T3.lastName ORDER BY SUM(T1.`+/-`) ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"years_difference": [], "name_given": [], "first_name": [], "last_name": []}
    return {"years_difference": result[0], "name_given": result[1], "first_name": result[2], "last_name": result[3]}

# Endpoint to get the player with the most games played within a PIM and year range
@app.get("/v1/hockey/top_games_played_by_pim_year_range", operation_id="get_top_games_played_by_pim_year_range", summary="Retrieve the name of the player who has played the most games within the specified range of penalty minutes and years. The range is defined by the minimum and maximum penalty minutes, as well as the minimum and maximum years. The player with the highest number of games played within these constraints is returned.")
async def get_top_games_played_by_pim_year_range(min_pim: int = Query(..., description="Minimum penalty minutes"), max_pim: int = Query(..., description="Maximum penalty minutes"), min_year: int = Query(..., description="Minimum year"), max_year: int = Query(..., description="Maximum year")):
    cursor.execute("SELECT T2.nameGiven FROM Scoring AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID AND T1.PIM BETWEEN ? AND ? AND T1.year BETWEEN ? AND ? ORDER BY T1.GP DESC LIMIT 1", (min_pim, max_pim, min_year, max_year))
    result = cursor.fetchone()
    if not result:
        return {"name_given": []}
    return {"name_given": result[0]}

# Endpoint to get the goalie with the most goals against
@app.get("/v1/hockey/top_goalie_by_goals_against", operation_id="get_top_goalie_by_goals_against", summary="Retrieves the top goalie with the highest number of goals against, based on the sum of goals against (GA) in their career. The goalie's age during their last year in the NHL is also provided.")
async def get_top_goalie_by_goals_against():
    cursor.execute("SELECT T2.lastNHL - T2.birthYear FROM GoaliesSC AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T2.lastNHL IS NOT NULL GROUP BY T2.lastNHL, T2.birthYear ORDER BY SUM(GA) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"years_difference": []}
    return {"years_difference": result[0]}

# Endpoint to get the coach with the most awards
@app.get("/v1/hockey/top_coach_by_awards", operation_id="get_top_coach_by_awards", summary="Retrieves the top-ranked coach based on the number of awards received. The coach's position, name, and specific awards are returned. The data is sorted in descending order by the count of awards, with the coach having the highest number of awards listed first.")
async def get_top_coach_by_awards():
    cursor.execute("SELECT T1.pos, T2.award, T1.nameGiven, T1.lastName FROM Master AS T1 INNER JOIN AwardsCoaches AS T2 ON T2.coachID = T1.coachID GROUP BY T1.pos, T2.award, T1.nameGiven, T1.lastName ORDER BY COUNT(T2.award) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"position": [], "award": [], "name_given": [], "last_name": []}
    return {"position": result[0], "award": result[1], "name_given": result[2], "last_name": result[3]}

# Endpoint to get the sum of goals for coaches before their first 'Second Team All-Star' award
@app.get("/v1/hockey/sum_goals_before_first_award", operation_id="get_sum_goals_before_first_award", summary="Retrieves the total number of goals scored by coaches before they received their first specified award. The award type is used to filter the coaches and calculate the sum of their goals.")
async def get_sum_goals_before_first_award(award: str = Query(..., description="Award type")):
    cursor.execute("SELECT SUM(T1.g) FROM Coaches AS T1 INNER JOIN ( SELECT coachID, year FROM AwardsCoaches WHERE award = ? ORDER BY year LIMIT 1 ) AS T2 ON T1.coachID = T2.coachID AND T1.year < T2.year", (award,))
    result = cursor.fetchone()
    if not result:
        return {"sum_goals": []}
    return {"sum_goals": result[0]}

# Endpoint to get the count of distinct opponents for a team in a specific year
@app.get("/v1/hockey/count_distinct_opponents_by_year", operation_id="get_count_distinct_opponents_by_year", summary="Retrieves the number of unique opponents a team has faced in a given year, sorted by the total wins against those opponents in descending order. The result is limited to the top team with the most wins.")
async def get_count_distinct_opponents_by_year(year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT COUNT(DISTINCT oppID), T2.tmID, T2.oppID FROM Teams AS T1 INNER JOIN TeamVsTeam AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID WHERE T2.year = ? GROUP BY T2.tmID, T2.oppID ORDER BY SUM(T2.W) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count_distinct_opponents": []}
    return {"count_distinct_opponents": result[0], "tmID": result[1], "oppID": result[2]}

# Endpoint to get the player with the highest game-winning goals within a range of short-handed goals
@app.get("/v1/hockey/player_with_highest_gwg_in_shg_range", operation_id="get_player_with_highest_gwg_in_shg_range", summary="Retrieves the name of the player who has scored the highest number of game-winning goals within the specified range of short-handed goals. The range is defined by the minimum and maximum short-handed goals provided as input parameters.")
async def get_player_with_highest_gwg_in_shg_range(min_shg: int = Query(..., description="Minimum short-handed goals"), max_shg: int = Query(..., description="Maximum short-handed goals")):
    cursor.execute("SELECT T2.nameGiven, T2.lastName FROM Scoring AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T1.SHG BETWEEN ? AND ? ORDER BY T1.GWG DESC LIMIT 1", (min_shg, max_shg))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"nameGiven": result[0], "lastName": result[1]}

# Endpoint to get the sum of losses for interim coaches in a specific year and team
@app.get("/v1/hockey/sum_losses_interim_coaches", operation_id="get_sum_losses_interim_coaches", summary="Retrieves the total number of losses for interim coaches of a specific team in a given year. The operation filters coaches based on their notes and aggregates their losses. The team name and year are also used to narrow down the results.")
async def get_sum_losses_interim_coaches(notes: str = Query(..., description="Notes for the coach"), year: str = Query(..., description="Year in YYYY format"), team_name: str = Query(..., description="Team name")):
    cursor.execute("SELECT SUM(T1.l), T1.coachID FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.notes = ? AND T1.year = ? AND T2.name = ? GROUP BY T1.coachID", (notes, year, team_name))
    result = cursor.fetchone()
    if not result:
        return {"sum_losses": []}
    return {"sum_losses": result[0], "coachID": result[1]}

# Endpoint to get the team with the highest assists for players with a minimum number of short-handed assists
@app.get("/v1/hockey/team_with_highest_assists_min_sha", operation_id="get_team_with_highest_assists_min_sha", summary="Retrieves the team with the highest total assists from a given year, considering only players who have accumulated at least a specified minimum number of short-handed assists. The response includes the team's rank, year, and name.")
async def get_team_with_highest_assists_min_sha(min_sha: int = Query(..., description="Minimum short-handed assists")):
    cursor.execute("SELECT T2.rank, T2.year, T2.name FROM Scoring AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.SHA >= ? ORDER BY T1.A DESC LIMIT 1", (min_sha,))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"rank": result[0], "year": result[1], "name": result[2]}

# Endpoint to get the coach IDs for a specific year and team rank
@app.get("/v1/hockey/coach_ids_by_year_and_rank", operation_id="get_coach_ids_by_year_and_rank", summary="Retrieves the unique identifiers of coaches who were active in a given year and whose teams achieved a specified rank. The year should be provided in YYYY format.")
async def get_coach_ids_by_year_and_rank(year: int = Query(..., description="Year in YYYY format"), rank: int = Query(..., description="Team rank")):
    cursor.execute("SELECT T1.coachID FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? AND T2.rank = ?", (year, rank))
    result = cursor.fetchall()
    if not result:
        return {"coach_ids": []}
    return {"coach_ids": [row[0] for row in result]}

# Endpoint to get distinct team names for a specific half, rank, and year range
@app.get("/v1/hockey/distinct_team_names_by_half_rank_year_range", operation_id="get_distinct_team_names_by_half_rank_year_range", summary="Retrieves a unique list of team names that meet the specified half, rank, and year range criteria. The half parameter filters results to a specific half of the season, while the rank parameter narrows the list to teams with a particular rank. The year range is defined by the min_year and max_year parameters, ensuring that only teams from the specified years are included in the results.")
async def get_distinct_team_names_by_half_rank_year_range(half: int = Query(..., description="Half of the season"), rank: int = Query(..., description="Team rank"), min_year: int = Query(..., description="Minimum year in YYYY format"), max_year: int = Query(..., description="Maximum year in YYYY format")):
    cursor.execute("SELECT DISTINCT T2.name FROM TeamsHalf AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.half = ? AND T1.rank = ? AND T1.year BETWEEN ? AND ?", (half, rank, min_year, max_year))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get player details based on age range at first NHL game and team goals
@app.get("/v1/hockey/player_details_by_age_range_and_team_goals", operation_id="get_player_details_by_age_range_and_team_goals", summary="Retrieves player details for those who made their NHL debut within a specified age range and played for teams with fewer than a given number of goals. The response includes the player's first name, last name, birth year, birth month, birth day, and team ID.")
async def get_player_details_by_age_range_and_team_goals(min_age: int = Query(..., description="Minimum age at first NHL game"), max_age: int = Query(..., description="Maximum age at first NHL game"), max_team_goals: int = Query(..., description="Maximum team goals")):
    cursor.execute("SELECT T2.nameGiven, T2.lastName, T2.birthYear, birthMon, birthDay , T3.tmID FROM Scoring AS T1 INNER JOIN Master AS T2 ON T2.playerID = T1.playerID INNER JOIN Teams AS T3 ON T3.tmID = T1.tmID WHERE (T2.firstNHL - T2.birthYear) BETWEEN ? AND ? AND T3.G < ?", (min_age, max_age, max_team_goals))
    result = cursor.fetchall()
    if not result:
        return {"player_details": []}
    return {"player_details": [{"nameGiven": row[0], "lastName": row[1], "birthYear": row[2], "birthMon": row[3], "birthDay": row[4], "tmID": row[5]} for row in result]}

# Endpoint to get the average bench minor penalties for a specific team
@app.get("/v1/hockey/average_bench_minor_penalties", operation_id="get_average_bench_minor_penalties", summary="Retrieves the average number of bench minor penalties for a specific hockey team. The team is identified by its name, which is provided as an input parameter.")
async def get_average_bench_minor_penalties(team_name: str = Query(..., description="Team name")):
    cursor.execute("SELECT CAST(SUM(BenchMinor) AS REAL) / 2 FROM Teams WHERE name = ?", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_bench_minor_penalties": []}
    return {"average_bench_minor_penalties": result[0]}

# Endpoint to get the power play goal to power play chance ratio for a specific team and year
@app.get("/v1/hockey/ppg_to_ppc_ratio", operation_id="get_ppg_to_ppc_ratio", summary="Retrieves the ratio of power play goals to power play chances for a given team and year. This operation calculates the ratio by dividing the number of power play goals (PPG) by the total power play chances (PPC) for the specified team and year.")
async def get_ppg_to_ppc_ratio(year: int = Query(..., description="Year in YYYY format"), team_name: str = Query(..., description="Team name")):
    cursor.execute("SELECT CAST(PPG AS REAL) / PPC FROM Teams WHERE year = ? AND name = ?", (year, team_name))
    result = cursor.fetchone()
    if not result:
        return {"ppg_to_ppc_ratio": []}
    return {"ppg_to_ppc_ratio": result[0]}

# Endpoint to get the team with the highest points in a given year
@app.get("/v1/hockey/team_with_highest_points", operation_id="get_team_with_highest_points", summary="Retrieves the team with the highest total points accumulated in a specific year. The operation calculates the sum of points for each team in the given year and returns the team with the highest sum.")
async def get_team_with_highest_points():
    cursor.execute("SELECT SUM(Pts), year FROM Teams GROUP BY year, tmID ORDER BY SUM(Pts) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result}

# Endpoint to get the count of teams with more wins than losses and points greater than a specified value in a given year
@app.get("/v1/hockey/count_teams_with_more_wins_and_points", operation_id="get_count_teams_with_more_wins_and_points", summary="Retrieves the number of teams that have won more games than they have lost and have accumulated more points than the specified minimum in a given year.")
async def get_count_teams_with_more_wins_and_points(year: int = Query(..., description="Year of the team"), points: int = Query(..., description="Minimum points")):
    cursor.execute("SELECT COUNT(tmID) FROM Teams WHERE year = ? AND W > L AND Pts > ?", (year, points))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the team with the highest average bench minor penalties in a given year
@app.get("/v1/hockey/team_with_highest_avg_bench_minor", operation_id="get_team_with_highest_avg_bench_minor", summary="Retrieves the name of the team with the highest average bench minor penalties in a specified year. The calculation is based on the total sum of bench minor penalties divided by the number of games played by the team in the given year. The result is sorted in descending order and the team with the highest average is returned.")
async def get_team_with_highest_avg_bench_minor(year: int = Query(..., description="Year of the team")):
    cursor.execute("SELECT name FROM Teams WHERE year = ? GROUP BY tmID, name ORDER BY CAST(SUM(BenchMinor) AS REAL) / 2 DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result[0]}

# Endpoint to get the top 3 teams with the highest total penalty minutes in a given year
@app.get("/v1/hockey/top_3_teams_with_highest_pim", operation_id="get_top_3_teams_with_highest_pim", summary="Retrieves the names of the top three teams with the highest total penalty minutes in a specified year. The teams are ranked based on their cumulative penalty minutes, and only the top three are returned.")
async def get_top_3_teams_with_highest_pim(year: int = Query(..., description="Year of the team")):
    cursor.execute("SELECT name FROM Teams WHERE year = ? GROUP BY tmID, name ORDER BY SUM(PIM) DESC LIMIT 3", (year,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": [team[0] for team in result]}

# Endpoint to get the team with the highest power play conversion rate among specified teams in a given year
@app.get("/v1/hockey/team_with_highest_ppc_among_specified", operation_id="get_team_with_highest_ppc_among_specified", summary="Retrieves the team with the highest power play conversion rate among the specified teams in a given year. The operation considers the provided year and team names to determine the team with the highest conversion rate. The result is a single team name.")
async def get_team_with_highest_ppc_among_specified(year: int = Query(..., description="Year of the team"), team1: str = Query(..., description="Name of the first team"), team2: str = Query(..., description="Name of the second team"), team3: str = Query(..., description="Name of the third team")):
    cursor.execute("SELECT name FROM Teams WHERE year = ? AND name IN (?, ?, ?) ORDER BY PKC DESC LIMIT 1", (year, team1, team2, team3))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result[0]}

# Endpoint to get the names of teams with more wins than losses in a given year
@app.get("/v1/hockey/teams_with_more_wins_than_losses", operation_id="get_teams_with_more_wins_than_losses", summary="Retrieves the names of hockey teams that have won more games than they have lost in a specified year. The year parameter is used to filter the results.")
async def get_teams_with_more_wins_than_losses(year: int = Query(..., description="Year of the team")):
    cursor.execute("SELECT T2.name FROM TeamsSC AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? AND T1.W > T1.L", (year,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": [team[0] for team in result]}

# Endpoint to get the names of teams in a given year
@app.get("/v1/hockey/teams_in_year", operation_id="get_teams_in_year", summary="Retrieves the names of hockey teams that were active in a specified year. The year parameter is used to filter the teams.")
async def get_teams_in_year(year: int = Query(..., description="Year of the team")):
    cursor.execute("SELECT T2.name FROM TeamsSC AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": [team[0] for team in result]}

# Endpoint to get the count of teams with points greater than a specified value in a given year
@app.get("/v1/hockey/count_teams_with_points_greater_than", operation_id="get_count_teams_with_points_greater_than", summary="Retrieves the number of teams that have accumulated more points than a specified value in a given year. The operation filters teams based on the provided year and minimum points threshold, then returns the count of teams that meet these criteria.")
async def get_count_teams_with_points_greater_than(year: int = Query(..., description="Year of the team"), points: int = Query(..., description="Minimum points")):
    cursor.execute("SELECT COUNT(T1.tmID) FROM TeamsSC AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? AND T2.Pts > ?", (year, points))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the penalty minutes of teams with a specific rank in a given year
@app.get("/v1/hockey/penalty_minutes_by_rank", operation_id="get_penalty_minutes_by_rank", summary="Retrieve the total penalty minutes (PIM) for teams that achieved a specific rank in a given year. The operation requires the year and rank as input parameters to filter the results.")
async def get_penalty_minutes_by_rank(year: int = Query(..., description="Year of the team"), rank: int = Query(..., description="Rank of the team")):
    cursor.execute("SELECT T1.PIM FROM TeamsSC AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? AND T2.rank = ?", (year, rank))
    result = cursor.fetchall()
    if not result:
        return {"penalty_minutes": []}
    return {"penalty_minutes": [pim[0] for pim in result]}

# Endpoint to get the team with the highest total wins
@app.get("/v1/hockey/team_with_highest_total_wins", operation_id="get_team_with_highest_total_wins", summary="Retrieves the name of the team with the highest total wins across all seasons. The operation calculates the sum of wins for each team and returns the team with the highest aggregate wins.")
async def get_team_with_highest_total_wins():
    cursor.execute("SELECT T2.name FROM TeamsSC AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year GROUP BY T2.name ORDER BY SUM(T1.W) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result[0]}

# Endpoint to get the win count for a specific team against another team in a given year
@app.get("/v1/hockey/team_vs_team_win_count", operation_id="get_team_vs_team_win_count", summary="Retrieves the number of wins a specific team has against another team in a given year. The operation requires the year, the name of the team, and the name of the opponent team as input parameters. The result is a count of wins for the specified team against the specified opponent in the specified year.")
async def get_team_vs_team_win_count(year: int = Query(..., description="Year of the match"), team_name: str = Query(..., description="Name of the team"), opponent_name: str = Query(..., description="Name of the opponent team")):
    cursor.execute("SELECT T1.W FROM TeamVsTeam AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? AND T1.tmID = ( SELECT DISTINCT tmID FROM Teams WHERE name = ? ) AND T1.oppID = ( SELECT DISTINCT tmID FROM Teams WHERE name = ? )", (year, team_name, opponent_name))
    result = cursor.fetchone()
    if not result:
        return {"win_count": []}
    return {"win_count": result[0]}

# Endpoint to get team names based on year and tie status
@app.get("/v1/hockey/team_names_by_year_and_tie_status", operation_id="get_team_names_by_year_and_tie_status", summary="Retrieves the names of teams that played against each other in a specific year, with an option to filter by tie status. The response includes the names of both teams involved in the match.")
async def get_team_names_by_year_and_tie_status(year: int = Query(..., description="Year of the match"), tie_status: int = Query(..., description="Tie status (1 for tie, 0 for no tie)")):
    cursor.execute("SELECT T2.name, T3.name FROM TeamVsTeam AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year INNER JOIN Teams AS T3 ON T1.year = T3.year AND T1.oppID = T3.tmID WHERE T1.year = ? AND T1.T = ?", (year, tie_status))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": result}

# Endpoint to get distinct first names of coaches for a specific team
@app.get("/v1/hockey/coach_first_names_by_team", operation_id="get_coach_first_names_by_team", summary="Retrieves a list of unique first names of coaches associated with a specific hockey team. The team is identified by its name.")
async def get_coach_first_names_by_team(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT DISTINCT T3.firstName FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.name = ?", (team_name,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the count of distinct Hall of Fame IDs for coaches of a specific team
@app.get("/v1/hockey/count_hof_ids_by_team", operation_id="get_count_hof_ids_by_team", summary="Retrieves the number of unique Hall of Fame IDs associated with coaches of a specific team. The team is identified by its name.")
async def get_count_hof_ids_by_team(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT COUNT(DISTINCT hofID) FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.name = ?", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the tallest coach of a specific team
@app.get("/v1/hockey/tallest_coach_by_team", operation_id="get_tallest_coach_by_team", summary="Retrieves the height of the tallest coach associated with a specific hockey team. The operation filters coaches by team name and returns the maximum height value from the filtered set. The input parameter specifies the team name.")
async def get_tallest_coach_by_team(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT T3.height FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.name = ? AND T3.coachID IS NOT NULL ORDER BY T3.height DESC LIMIT 1", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"height": []}
    return {"height": result[0]}

# Endpoint to get first names of coaches for a specific year
@app.get("/v1/hockey/coach_first_names_by_year", operation_id="get_coach_first_names_by_year", summary="Retrieves the first names of coaches who were active in a specific year. The year is provided as an input parameter, and the response includes a list of first names for coaches who were active in that year.")
async def get_coach_first_names_by_year(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T3.firstName FROM Coaches AS T1 INNER JOIN TeamsSC AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the count of distinct coach IDs for a specific team and birth month
@app.get("/v1/hockey/count_coach_ids_by_team_and_birth_month", operation_id="get_count_coach_ids_by_team_and_birth_month", summary="Retrieves the number of unique coaches associated with a specific team and born in a particular month. The team is identified by its name, and the birth month is specified as a number between 1 and 12.")
async def get_count_coach_ids_by_team_and_birth_month(team_name: str = Query(..., description="Name of the team"), birth_month: int = Query(..., description="Birth month (1-12)")):
    cursor.execute("SELECT COUNT(DISTINCT T3.coachID) FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.name = ? AND T3.birthMon = ?", (team_name, birth_month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct coach IDs for a specific team and birth country
@app.get("/v1/hockey/count_coach_ids_by_team_and_birth_country", operation_id="get_count_coach_ids_by_team_and_birth_country", summary="Retrieves the number of unique coaches associated with a specific team and birth country. The operation requires the team name and the birth country as input parameters.")
async def get_count_coach_ids_by_team_and_birth_country(team_name: str = Query(..., description="Name of the team"), birth_country: str = Query(..., description="Birth country")):
    cursor.execute("SELECT COUNT(DISTINCT T3.coachID) FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.name = ? AND T3.birthCountry = ?", (team_name, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct coach IDs for a specific team where the coach has a recorded death year
@app.get("/v1/hockey/count_coach_ids_by_team_and_death_year", operation_id="get_count_coach_ids_by_team_and_death_year", summary="Retrieves the number of unique coaches who have a recorded death year for a specified team. The team is identified by its name.")
async def get_count_coach_ids_by_team_and_death_year(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT COUNT(DISTINCT T3.coachID) FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.name = ? AND T3.deathYear IS NOT NULL", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct nicknames of coaches for a specific team and birth country
@app.get("/v1/hockey/coach_nicknames_by_team_and_birth_country", operation_id="get_coach_nicknames_by_team_and_birth_country", summary="Retrieves the unique nicknames of coaches who have coached a specific team and were born in a certain country. The team is identified by its name, and the country is specified by its name.")
async def get_coach_nicknames_by_team_and_birth_country(team_name: str = Query(..., description="Name of the team"), birth_country: str = Query(..., description="Birth country")):
    cursor.execute("SELECT DISTINCT nameNick FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.name = ? AND T3.birthCountry = ?", (team_name, birth_country))
    result = cursor.fetchall()
    if not result:
        return {"nicknames": []}
    return {"nicknames": [row[0] for row in result]}

# Endpoint to get the count of distinct coaches based on year and birth country
@app.get("/v1/hockey/count_distinct_coaches", operation_id="get_count_distinct_coaches", summary="Retrieves the total number of unique coaches who have coached a team in a specific year and were born in a particular country. The year and birth country are provided as input parameters.")
async def get_count_distinct_coaches(year: int = Query(..., description="Year of the team"), birth_country: str = Query(..., description="Birth country of the coach")):
    cursor.execute("SELECT COUNT(DISTINCT T3.coachID) FROM Coaches AS T1 INNER JOIN TeamsSC AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.year = ? AND T3.birthCountry = ?", (year, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the team name with the highest total goals in a specific year against a specific opponent
@app.get("/v1/hockey/top_team_by_goals", operation_id="get_top_team_by_goals", summary="Retrieves the name of the team with the highest total goals scored in a specific year against a given opponent. The year and opponent team name are required as input parameters.")
async def get_top_team_by_goals(year: int = Query(..., description="Year of the match"), opponent_name: str = Query(..., description="Name of the opponent team")):
    cursor.execute("SELECT T3.name FROM TeamVsTeam AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.oppID = T2.tmID INNER JOIN Teams AS T3 ON T1.year = T3.year AND T1.tmID = T3.tmID WHERE T1.year = ? AND T2.name = ? GROUP BY T3.name ORDER BY SUM(T2.G) DESC LIMIT 1", (year, opponent_name))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get distinct team names that have played against a specific opponent
@app.get("/v1/hockey/distinct_team_names", operation_id="get_distinct_team_names", summary="Retrieves a list of unique team names that have played against a specified opponent team in a given year. The input parameter is used to identify the opponent team.")
async def get_distinct_team_names(opponent_name: str = Query(..., description="Name of the opponent team")):
    cursor.execute("SELECT DISTINCT T3.name FROM TeamVsTeam AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.oppID = T2.tmID INNER JOIN Teams AS T3 ON T1.year = T3.year AND T1.tmID = T3.tmID WHERE T2.name = ?", (opponent_name,))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get the penalty minutes for a specific team in a specific year
@app.get("/v1/hockey/penalty_minutes", operation_id="get_penalty_minutes", summary="Retrieves the total penalty minutes (PIM) for a specific hockey team in a given year. The operation requires the team's name and the year as input parameters to accurately fetch the penalty minutes data.")
async def get_penalty_minutes(team_name: str = Query(..., description="Name of the team"), year: int = Query(..., description="Year of the team")):
    cursor.execute("SELECT T2.PIM FROM Teams AS T1 INNER JOIN TeamsSC AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.name = ? AND T1.year = ?", (team_name, year))
    result = cursor.fetchone()
    if not result:
        return {"penalty_minutes": []}
    return {"penalty_minutes": result[0]}

# Endpoint to get the number of wins for a specific team with a specific number of penalty minutes
@app.get("/v1/hockey/wins_by_penalty_minutes", operation_id="get_wins_by_penalty_minutes", summary="Retrieves the number of wins for a specific hockey team based on the given penalty minutes. The operation uses the team name and penalty minutes as input parameters to filter the data and return the corresponding number of wins.")
async def get_wins_by_penalty_minutes(team_name: str = Query(..., description="Name of the team"), penalty_minutes: int = Query(..., description="Penalty minutes of the team")):
    cursor.execute("SELECT T2.W FROM Teams AS T1 INNER JOIN TeamsSC AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.name = ? AND T2.PIM = ?", (team_name, penalty_minutes))
    result = cursor.fetchone()
    if not result:
        return {"wins": []}
    return {"wins": result[0]}

# Endpoint to get the years a specific team had a specific number of penalty minutes
@app.get("/v1/hockey/years_by_penalty_minutes", operation_id="get_years_by_penalty_minutes", summary="Retrieve the years in which a specific hockey team incurred a specified number of penalty minutes. The operation requires the team's name and the number of penalty minutes as input parameters.")
async def get_years_by_penalty_minutes(team_name: str = Query(..., description="Name of the team"), penalty_minutes: int = Query(..., description="Penalty minutes of the team")):
    cursor.execute("SELECT T1.year FROM Teams AS T1 INNER JOIN TeamsSC AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.name = ? AND T2.PIM = ?", (team_name, penalty_minutes))
    result = cursor.fetchall()
    if not result:
        return {"years": []}
    return {"years": [row[0] for row in result]}

# Endpoint to get the first name of the earliest coach for a specific team
@app.get("/v1/hockey/earliest_coach_first_name", operation_id="get_earliest_coach_first_name", summary="Retrieves the first name of the earliest coach associated with a specific hockey team. The team is identified by its name, and the coach's first name is determined based on the earliest year of their coaching tenure.")
async def get_earliest_coach_first_name(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT T3.firstName FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T2.name = ? ORDER BY T1.year LIMIT 1", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"first_name": []}
    return {"first_name": result[0]}

# Endpoint to get the average win ratio for a specific team in a specific year
@app.get("/v1/hockey/average_win_ratio", operation_id="get_average_win_ratio", summary="Retrieves the average win ratio for a specific hockey team in a given year. The win ratio is calculated by summing the ratio of wins to games played for each opponent, then dividing by the total number of opponents. The team name and year are required as input parameters.")
async def get_average_win_ratio(team_name: str = Query(..., description="Name of the team"), year: int = Query(..., description="Year of the team")):
    cursor.execute("SELECT SUM(CAST(T2.W AS REAL) / T2.G) / COUNT(T1.oppID) FROM TeamVsTeam AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID WHERE T2.name = ? AND T1.year = ?", (team_name, year))
    result = cursor.fetchone()
    if not result:
        return {"average_win_ratio": []}
    return {"average_win_ratio": result[0]}

# Endpoint to get the average penalty minutes for a specific team
@app.get("/v1/hockey/average_penalty_minutes", operation_id="get_average_penalty_minutes", summary="Retrieves the average penalty minutes for a specific hockey team. The calculation is based on the sum of penalty minutes (PIM) divided by the count of PIM for the specified team in a given year. The team name is required as an input parameter.")
async def get_average_penalty_minutes(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT CAST(SUM(T2.PIM) AS REAL) / COUNT(T2.PIM) FROM Teams AS T1 INNER JOIN TeamsSC AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.name = ?", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_penalty_minutes": []}
    return {"average_penalty_minutes": result[0]}

# Endpoint to get the win ratio for a specific team
@app.get("/v1/hockey/win_ratio", operation_id="get_win_ratio", summary="Retrieves the win ratio for a specific hockey team. The win ratio is calculated as the sum of wins divided by the sum of games played and wins. The team's name is required as an input parameter to identify the team for which the win ratio is calculated.")
async def get_win_ratio(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT SUM(T2.W / T2.G) / SUM(T2.G + T2.W) FROM Teams AS T1 INNER JOIN TeamsSC AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.name = ?", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"win_ratio": []}
    return {"win_ratio": result[0]}

# Endpoint to get the win ratio for a specific team
@app.get("/v1/hockey/team_win_ratio", operation_id="get_team_win_ratio", summary="Retrieves the win ratio for a specific hockey team. The win ratio is calculated as the sum of the team's wins divided by the sum of the team's games played and wins. The team's name is required as an input parameter to identify the team for which the win ratio is calculated.")
async def get_team_win_ratio(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT SUM(CAST(T2.W AS REAL) / T2.G) / SUM(T2.G + T2.W) FROM Teams AS T1 INNER JOIN TeamsSC AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.name = ?", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"win_ratio": []}
    return {"win_ratio": result[0]}

# Endpoint to get the latest inductee in a specific category
@app.get("/v1/hockey/latest_inductee", operation_id="get_latest_inductee", summary="Retrieves the name of the most recent inductee in the specified category or categories. The operation supports multiple categories, allowing you to find the latest inductee across a range of categories.")
async def get_latest_inductee(category1: str = Query(..., description="First category"), category2: str = Query(..., description="Second category")):
    cursor.execute("SELECT name FROM HOF WHERE category IN (?, ?) ORDER BY year DESC LIMIT 1", (category1, category2))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get inductees in a specific category within a year range
@app.get("/v1/hockey/inductees_by_category_year_range", operation_id="get_inductees_by_category_year_range", summary="Retrieves the names and unique identifiers of Hall of Fame inductees belonging to a specified category, within a defined range of induction years. The category and year range are provided as input parameters.")
async def get_inductees_by_category_year_range(category: str = Query(..., description="Category of the inductee"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT name, hofID FROM HOF WHERE category = ? AND year BETWEEN ? AND ?", (category, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"inductees": []}
    return {"inductees": result}

# Endpoint to get the count of inductees in a specific category and year
@app.get("/v1/hockey/inductee_count_by_category_year", operation_id="get_inductee_count_by_category_year", summary="Retrieves the total number of inductees in a specified category and year from the Hall of Fame (HOF) records.")
async def get_inductee_count_by_category_year(category: str = Query(..., description="Category of the inductee"), year: int = Query(..., description="Year of induction")):
    cursor.execute("SELECT COUNT(hofID) FROM HOF WHERE category = ? AND year = ?", (category, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players who are also coaches and in the Hall of Fame
@app.get("/v1/hockey/player_coach_hof_count", operation_id="get_player_coach_hof_count", summary="Retrieves the total count of individuals who have been inducted into the Hall of Fame, served as both players and coaches in their careers.")
async def get_player_coach_hof_count():
    cursor.execute("SELECT COUNT(playerID) FROM Master WHERE hofID IS NOT NULL AND playerID IS NOT NULL AND coachID IS NOT NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players with a specific last name who are not coaches
@app.get("/v1/hockey/player_count_by_last_name", operation_id="get_player_count_by_last_name", summary="Retrieves the total number of players with a given last name who are not coaches. The operation filters out players who also serve as coaches.")
async def get_player_count_by_last_name(last_name: str = Query(..., description="Last name of the player")):
    cursor.execute("SELECT COUNT(playerID) FROM Master WHERE lastName = ? AND coachID IS NULL", (last_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get bench minor penalties for a specific coach in a specific year
@app.get("/v1/hockey/bench_minor_penalties", operation_id="get_bench_minor_penalties", summary="Retrieves the total number of bench minor penalties incurred by a specific coach in a given year. The operation requires the coach's first and last names, as well as the year of the season, to accurately identify the coach and the corresponding bench minor penalties.")
async def get_bench_minor_penalties(first_name: str = Query(..., description="First name of the coach"), last_name: str = Query(..., description="Last name of the coach"), year: int = Query(..., description="Year of the season")):
    cursor.execute("SELECT T2.BenchMinor FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T1.year = T2.year AND T1.tmID = T2.tmID INNER JOIN Master AS T3 ON T1.coachID = T3.coachID WHERE T3.firstName = ? AND T3.lastName = ? AND T1.year = ?", (first_name, last_name, year))
    result = cursor.fetchone()
    if not result:
        return {"bench_minor": []}
    return {"bench_minor": result[0]}

# Endpoint to get the tallest goalie with a specific number of even-strength goals against
@app.get("/v1/hockey/tallest_goalie_by_even_strength_goals", operation_id="get_tallest_goalie_by_even_strength_goals", summary="Retrieves the name of the tallest goalie who has conceded a specific number of even-strength goals. The response is sorted by height in descending order and limited to the top result.")
async def get_tallest_goalie_by_even_strength_goals(even_strength_goals: int = Query(..., description="Number of even-strength goals against")):
    cursor.execute("SELECT T2.firstName, T2.lastName FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T1.ENG = ? ORDER BY T2.height DESC LIMIT 1", (even_strength_goals,))
    result = cursor.fetchone()
    if not result:
        return {"goalie": []}
    return {"goalie": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get goalie statistics for a specific team and year
@app.get("/v1/hockey/goalie_stats_by_team_year", operation_id="get_goalie_stats_by_team_year", summary="Retrieves goalie statistics for a specific team and season, based on the number of games played. The data includes the shoot/catch preference and the first and last names of the goalies.")
async def get_goalie_stats_by_team_year(year: int = Query(..., description="Year of the season"), tmID: str = Query(..., description="Team ID"), gp: int = Query(..., description="Number of games played")):
    cursor.execute("SELECT T2.shootCatch, T2.firstName, T2.lastName FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID AND T1.year = ? WHERE T1.tmID = ? AND T1.GP = ?", (year, tmID, gp))
    result = cursor.fetchone()
    if not result:
        return {"goalie_stats": []}
    return {"goalie_stats": {"shoot_catch": result[0], "first_name": result[1], "last_name": result[2]}}

# Endpoint to get the count of distinct teams for a goalie with a specific last name
@app.get("/v1/hockey/team_count_by_goalie_last_name", operation_id="get_team_count_by_goalie_last_name", summary="Retrieves the number of unique teams that a goalie with a specific last name has played for. The last name of the goalie is provided as an input parameter.")
async def get_team_count_by_goalie_last_name(last_name: str = Query(..., description="Last name of the goalie")):
    cursor.execute("SELECT COUNT(DISTINCT T1.tmID) FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T2.lastName = ?", (last_name,))
    result = cursor.fetchone()
    if not result:
        return {"team_count": []}
    return {"team_count": result[0]}

# Endpoint to get goalies' names and whether they played for a specific team in a specific year
@app.get("/v1/hockey/goalies_names_team_year", operation_id="get_goalies_names_team_year", summary="Retrieves the first and last names of goalies who played in a specific year and whether they played for a given team. The team is identified by its ID, and the year is specified as a four-digit number. Only goalies who have a coach and are not in the Hall of Fame are included in the results.")
async def get_goalies_names_team_year(team_id: str = Query(..., description="Team ID (e.g., 'BOS')"), year: int = Query(..., description="Year (e.g., 1972)")):
    cursor.execute("SELECT T2.firstName, T2.lastName, IIF(T1.tmID = ?, 'YES', 'NO') FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T1.year = ? AND T1.tmID = ? AND T2.coachID IS NOT NULL AND T2.hofID IS NULL", (team_id, year, team_id))
    result = cursor.fetchall()
    if not result:
        return {"goalies": []}
    return {"goalies": result}

# Endpoint to get the sum of games played by goalies with a specific legends ID
@app.get("/v1/hockey/sum_games_played_legends_id", operation_id="get_sum_games_played_legends_id", summary="Retrieves the total number of games played by goalies associated with a specific legends ID. The legends ID is used to identify the goalies whose game statistics are aggregated.")
async def get_sum_games_played_legends_id(legends_id: str = Query(..., description="Legends ID (e.g., 'P196402')")):
    cursor.execute("SELECT SUM(T1.GP) FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T2.legendsID = ?", (legends_id,))
    result = cursor.fetchone()
    if not result:
        return {"sum_games_played": []}
    return {"sum_games_played": result[0]}

# Endpoint to get the goalie with the most minutes played for a specific team and position
@app.get("/v1/hockey/top_goalie_team_position", operation_id="get_top_goalie_team_position", summary="Retrieves the goalie with the highest total minutes played for a given team and position. The team name and position are required as input parameters to filter the results.")
async def get_top_goalie_team_position(team_name: str = Query(..., description="Team name (e.g., 'Quebec Bulldogs')"), position: str = Query(..., description="Position (e.g., 'D')")):
    cursor.execute("SELECT T2.firstName, T2.lastName FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID INNER JOIN Teams AS T3 ON T1.tmID = T3.tmID AND T1.year = T3.year WHERE T3.name = ? AND T2.pos = ? GROUP BY T1.playerID, T2.firstName, T2.lastName ORDER BY SUM(T1.Min) DESC LIMIT 1", (team_name, position))
    result = cursor.fetchone()
    if not result:
        return {"top_goalie": []}
    return {"top_goalie": result}

# Endpoint to get the count of distinct goalies for a specific team
@app.get("/v1/hockey/count_distinct_goalies_team", operation_id="get_count_distinct_goalies_team", summary="Retrieve the number of unique goalies who have played for a specific hockey team. The team is identified by its name, which is provided as an input parameter.")
async def get_count_distinct_goalies_team(team_name: str = Query(..., description="Team name (e.g., 'Calgary Flames')")):
    cursor.execute("SELECT COUNT(DISTINCT playerID) FROM Goalies AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T2.name = ?", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"count_distinct_goalies": []}
    return {"count_distinct_goalies": result[0]}

# Endpoint to get the goalie with the most goals against for a specific team
@app.get("/v1/hockey/top_goalie_goals_against_team", operation_id="get_top_goalie_goals_against_team", summary="Retrieves the goalie from a specific team who has allowed the most goals against. The team is identified by its name. The goalie's ID is returned.")
async def get_top_goalie_goals_against_team(team_name: str = Query(..., description="Team name (e.g., 'Minnesota North Stars')")):
    cursor.execute("SELECT playerID FROM Goalies AS T1 INNER JOIN Teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T2.name = ? GROUP BY T1.playerID ORDER BY SUM(T1.GA) DESC LIMIT 1", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"top_goalie": []}
    return {"top_goalie": result[0]}

# Endpoint to get the count of distinct goalies in the Hall of Fame for a specific team
@app.get("/v1/hockey/count_hof_goalies_team", operation_id="get_count_hof_goalies_team", summary="Retrieve the number of unique Hall of Fame goalies who have played for a specific team. The team is identified by its name, which must be provided as an input parameter.")
async def get_count_hof_goalies_team(team_name: str = Query(..., description="Team name (e.g., 'Haileybury Hockey Club')")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID INNER JOIN Teams AS T3 ON T1.tmID = T3.tmID AND T1.year = T3.year WHERE T3.name = ? AND T2.hofID IS NOT NULL", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"count_hof_goalies": []}
    return {"count_hof_goalies": result[0]}

# Endpoint to get the shoot/catch preference of the goalie with the most shutouts in a specific year
@app.get("/v1/hockey/top_shoot_catch_goalie_year", operation_id="get_top_shoot_catch_goalie_year", summary="Retrieves the preferred shoot/catch style of the goalie who achieved the highest number of shutouts in a given year. The year is specified as a parameter.")
async def get_top_shoot_catch_goalie_year(year: int = Query(..., description="Year (e.g., 2010)")):
    cursor.execute("SELECT T2.shootCatch FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T1.year = ? GROUP BY T2.shootCatch ORDER BY SUM(T1.SHO) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"top_shoot_catch": []}
    return {"top_shoot_catch": result[0]}

# Endpoint to get the percentage change in games played between two years for a specific goalie
@app.get("/v1/hockey/percentage_change_games_played", operation_id="get_percentage_change_games_played", summary="Retrieve the percentage change in games played by a specific goalie between two given years. The calculation is based on the total games played by the goalie in each year. The input parameters include the first and second years, as well as the first and last names of the goalie.")
async def get_percentage_change_games_played(year1: int = Query(..., description="First year (e.g., 2005)"), year2: int = Query(..., description="Second year (e.g., 2006)"), first_name: str = Query(..., description="First name of the goalie (e.g., 'David')"), last_name: str = Query(..., description="Last name of the goalie (e.g., 'Aebischer')")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T1.year = ? THEN T1.GP ELSE 0 END) - SUM(CASE WHEN T1.year = ? THEN T1.GP ELSE 0 END)) AS REAL) * 100 / SUM(CASE WHEN T1.year = ? THEN T1.GP ELSE 0 END) FROM Goalies AS T1 INNER JOIN Master AS T2 ON T1.playerID = T2.playerID WHERE T2.firstName = ? AND T2.lastName = ?", (year1, year2, year1, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result[0]}

# Endpoint to get player and coach IDs from the Master table
@app.get("/v1/hockey/player_coach_ids", operation_id="get_player_coach_ids", summary="Retrieves a list of unique player and coach identifiers from the Master table. This operation filters out any records with missing player or coach IDs, ensuring that only complete entries are returned.")
async def get_player_coach_ids():
    cursor.execute("SELECT playerID, coachID FROM Master WHERE playerID IS NOT NULL AND coachID IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"player_coach_ids": []}
    return {"player_coach_ids": result}

# Endpoint to get distinct player IDs with a specific average height
@app.get("/v1/hockey/player_ids_avg_height", operation_id="get_player_ids_avg_height", summary="Retrieves a list of unique player IDs from the Master table, grouped by those who have a specific average height. The average height is provided as an input parameter.")
async def get_player_ids_avg_height(avg_height: int = Query(..., description="Average height (e.g., 75)")):
    cursor.execute("SELECT DISTINCT playerID FROM Master GROUP BY playerID HAVING AVG(height) = ?", (avg_height,))
    result = cursor.fetchall()
    if not result:
        return {"player_ids": []}
    return {"player_ids": result}

# Endpoint to get player IDs ordered by weight in descending order with a limit
@app.get("/v1/hockey/player_ids_by_weight", operation_id="get_player_ids_by_weight", summary="Retrieves a specified number of player IDs, ordered by weight in descending order. This operation allows you to limit the number of results returned.")
async def get_player_ids_by_weight(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT playerID FROM Master ORDER BY weight DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"player_ids": []}
    return {"player_ids": [row[0] for row in result]}

# Endpoint to get distinct first and last names of players from a specific birth country
@app.get("/v1/hockey/player_names_by_birth_country", operation_id="get_player_names_by_birth_country", summary="Retrieve a unique list of first and last names of hockey players born in a specified country.")
async def get_player_names_by_birth_country(birth_country: str = Query(..., description="Birth country of the player")):
    cursor.execute("SELECT DISTINCT firstName, lastName FROM Master WHERE birthCountry = ?", (birth_country,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get distinct player IDs of deceased players
@app.get("/v1/hockey/deceased_player_ids", operation_id="get_deceased_player_ids", summary="Retrieves a unique list of player identifiers for players who are no longer living. This operation filters out any null or invalid player identifiers.")
async def get_deceased_player_ids():
    cursor.execute("SELECT DISTINCT playerID FROM Master WHERE deathYear IS NOT NULL AND playerID IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"player_ids": []}
    return {"player_ids": [row[0] for row in result]}

# Endpoint to get distinct first names of coaches who coached after a specific year
@app.get("/v1/hockey/coach_first_names_after_year", operation_id="get_coach_first_names_after_year", summary="Retrieves a list of unique first names of coaches who have coached in seasons after the specified year. This operation is useful for identifying the distinct first names of coaches who have coached after a certain year, providing insights into coaching trends and patterns.")
async def get_coach_first_names_after_year(year: int = Query(..., description="Year after which the coach coached")):
    cursor.execute("SELECT DISTINCT T1.firstName FROM Master AS T1 INNER JOIN Coaches AS T2 ON T1.coachID = T2.coachID WHERE T2.year > ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get height and weight of coaches who received awards in a specific year
@app.get("/v1/hockey/coach_height_weight_by_year", operation_id="get_coach_height_weight_by_year", summary="Retrieves the height and weight of coaches who were awarded in a specified year. The year parameter is used to filter the results.")
async def get_coach_height_weight_by_year(year: str = Query(..., description="Year when the coach received the award")):
    cursor.execute("SELECT T1.height, T1.weight FROM Master AS T1 INNER JOIN AwardsCoaches AS T2 ON T1.coachID = T2.coachID WHERE T2.year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"coach_details": []}
    return {"coach_details": [{"height": row[0], "weight": row[1]} for row in result]}

# Endpoint to get the sum of goals and age of a specific player
@app.get("/v1/hockey/player_goals_and_age", operation_id="get_player_goals_and_age", summary="Retrieves the total number of goals scored and the age of a specific hockey player. The player is identified by their unique ID. The age is calculated based on the current year and the player's birth year.")
async def get_player_goals_and_age(player_id: str = Query(..., description="Player ID")):
    cursor.execute("SELECT SUM(T2.G), STRFTIME('%Y', CURRENT_TIMESTAMP) - T1.birthyear FROM Master AS T1 INNER JOIN Scoring AS T2 ON T1.playerID = T2.playerID WHERE T1.playerID = ? GROUP BY T1.birthyear", (player_id,))
    result = cursor.fetchone()
    if not result:
        return {"player_details": []}
    return {"player_details": {"goals": result[0], "age": result[1]}}

# Endpoint to get distinct coach IDs of coaches who are not players
@app.get("/v1/hockey/coach_ids_not_players", operation_id="get_coach_ids_not_players", summary="Retrieves a unique list of coach identifiers for coaches who have not been players. This operation filters out coaches who have also played in the league, ensuring that only those who have exclusively served as coaches are included in the results.")
async def get_coach_ids_not_players():
    cursor.execute("SELECT DISTINCT T2.coachID FROM Master AS T1 INNER JOIN AwardsCoaches AS T2 ON T1.coachID = T2.coachID WHERE T1.playerID IS NULL")
    result = cursor.fetchall()
    if not result:
        return {"coach_ids": []}
    return {"coach_ids": [row[0] for row in result]}

# Endpoint to get distinct player IDs based on position and weight
@app.get("/v1/hockey/player_ids_by_position_weight", operation_id="get_player_ids_by_position_weight", summary="Retrieves a unique list of player IDs that match a specified position pattern, exceed a given weight threshold, and correspond to an exact position. This operation ensures that the returned player IDs are not null and correspond to the provided position.")
async def get_player_ids_by_position_weight(pos_like: str = Query(..., description="Position pattern to match"), weight: int = Query(..., description="Minimum weight of the player"), pos: str = Query(..., description="Exact position of the player")):
    cursor.execute("SELECT DISTINCT playerID FROM Master WHERE pos LIKE ? AND weight > ? AND playerID IS NOT NULL AND pos = ?", (pos_like, weight, pos))
    result = cursor.fetchall()
    if not result:
        return {"player_ids": []}
    return {"player_ids": [row[0] for row in result]}

# Endpoint to get the count of games played by players from a specific birth country
@app.get("/v1/hockey/games_played_by_birth_country", operation_id="get_games_played_by_birth_country", summary="Retrieves the total number of games played by players born in a specified country. The input parameter is the birth country of the players.")
async def get_games_played_by_birth_country(birth_country: str = Query(..., description="Birth country of the player")):
    cursor.execute("SELECT COUNT(T2.GP) FROM Master AS T1 INNER JOIN Scoring AS T2 ON T1.playerID = T2.playerID WHERE T1.birthCountry = ?", (birth_country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of points and coach ID for a specific team
@app.get("/v1/hockey/team_points_and_coach_id", operation_id="get_team_points_and_coach_id", summary="Retrieves the total points accumulated by a specific team and the ID of the team's coach. The team is identified by its unique ID.")
async def get_team_points_and_coach_id(team_id: str = Query(..., description="Team ID")):
    cursor.execute("SELECT SUM(T2.Pts), T1.coachID FROM Coaches AS T1 INNER JOIN Teams AS T2 ON T2.tmID = T1.tmID WHERE T2.tmID = ? GROUP BY T1.coachID", (team_id,))
    result = cursor.fetchall()
    if not result:
        return {"team_details": []}
    return {"team_details": [{"points": row[0], "coach_id": row[1]} for row in result]}

# Endpoint to get the sum of goals for a specific team in a specific division and year
@app.get("/v1/hockey/sum_goals_team_division_year", operation_id="get_sum_goals", summary="Retrieves the total number of goals scored by a specific team in a given division and year. The operation requires the division ID, team ID, and year as input parameters to accurately calculate the sum of goals.")
async def get_sum_goals(div_id: str = Query(..., description="Division ID"), tm_id: str = Query(..., description="Team ID"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT SUM(T2.G) FROM Teams AS T1 INNER JOIN Scoring AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.divID = ? AND T1.tmID = ? AND T1.year = ?", (div_id, tm_id, year))
    result = cursor.fetchone()
    if not result:
        return {"sum_goals": []}
    return {"sum_goals": result[0]}

# Endpoint to get the sum of losses and assists for a specific team in a specific year
@app.get("/v1/hockey/sum_losses_assists_team_year", operation_id="get_sum_losses_assists", summary="Retrieves the total number of losses and assists for a specific hockey team in a given year. The operation requires the team's unique identifier and the year to calculate the sums.")
async def get_sum_losses_assists(tm_id: str = Query(..., description="Team ID"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT SUM(T1.L), SUM(T2.A) FROM Teams AS T1 INNER JOIN Scoring AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.tmID = ? AND T1.year = ?", (tm_id, year))
    result = cursor.fetchone()
    if not result:
        return {"sum_losses": [], "sum_assists": []}
    return {"sum_losses": result[0], "sum_assists": result[1]}

# Endpoint to get the sum of weights for players in a specific team and year
@app.get("/v1/hockey/sum_weights_team_year", operation_id="get_sum_weights", summary="Retrieves the total weight of all players in a specific team for a given year. The operation requires the year and team ID as input parameters to filter the data accordingly.")
async def get_sum_weights(year: int = Query(..., description="Year"), tm_id: str = Query(..., description="Team ID")):
    cursor.execute("SELECT SUM(T1.weight) FROM Master AS T1 INNER JOIN Scoring AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? AND T2.tmID = ?", (year, tm_id))
    result = cursor.fetchone()
    if not result:
        return {"sum_weights": []}
    return {"sum_weights": result[0]}

# Endpoint to get the player ID and team ID of the shortest player within a specific year range
@app.get("/v1/hockey/shortest_player_year_range", operation_id="get_shortest_player", summary="Get the player ID and team ID of the shortest player within a specific year range")
async def get_shortest_player(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT T2.playerID, T2.tmID FROM ( SELECT playerID FROM Master WHERE height IS NOT NULL ORDER BY height ASC LIMIT 1 ) AS T1 INNER JOIN ( SELECT DISTINCT playerID, tmID FROM Scoring WHERE year BETWEEN ? AND ? ) AS T2 ON T1.playerID = T2.playerID", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"player_id": [], "team_id": []}
    return {"player_id": result[0], "team_id": result[1]}

# Endpoint to get the team ID and birth country of players from the team with the highest win ratio in a specific year
@app.get("/v1/hockey/team_birth_country_highest_win_ratio", operation_id="get_team_birth_country", summary="Retrieves the team ID and birth country of players from the team with the highest win ratio in a given year. The win ratio is calculated as the number of wins divided by the total number of games played (wins plus losses). The year is a required input parameter.")
async def get_team_birth_country(year: int = Query(..., description="Year")):
    cursor.execute("SELECT DISTINCT T3.tmID, T1.birthCountry FROM Master AS T1 INNER JOIN Scoring AS T2 ON T1.playerID = T2.playerID INNER JOIN ( SELECT year, tmID FROM Teams WHERE year = ? ORDER BY W / (W + L) DESC LIMIT 1 ) AS T3 ON T2.tmID = T3.tmID AND T2.year = T3.year", (year,))
    result = cursor.fetchall()
    if not result:
        return {"team_id": [], "birth_country": []}
    return {"team_id": [row[0] for row in result], "birth_country": [row[1] for row in result]}

# Endpoint to get the win ratio and player ID of the player with the highest goals for a specific team in a specific year
@app.get("/v1/hockey/win_ratio_player_highest_goals", operation_id="get_win_ratio_player", summary="Retrieves the win ratio and player ID of the player who scored the most goals for a specific team in a given year. The operation calculates the win ratio by dividing the number of wins by the total number of games played for the team in the specified year. The player with the highest total goals scored in the same year is then identified. The team and year are determined by the provided team ID and year parameters.")
async def get_win_ratio_player(tm_id: str = Query(..., description="Team ID"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT CAST(T1.W AS REAL) / T1.G, T2.playerID FROM Teams AS T1 INNER JOIN Scoring AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.tmID = ? AND T1.year = ? GROUP BY T1.W / T1.G, T2.playerID ORDER BY SUM(T2.G) DESC LIMIT 1", (tm_id, year))
    result = cursor.fetchone()
    if not result:
        return {"win_ratio": [], "player_id": []}
    return {"win_ratio": result[0], "player_id": result[1]}

api_calls = [
    "/v1/hockey/players_by_birth_year_and_country?birth_year=1990&birth_country=USA",
    "/v1/hockey/players_by_position_no_shoot_catch?pos=F",
    "/v1/hockey/players_not_in_hall_of_fame",
    "/v1/hockey/most_recently_born_alive_player",
    "/v1/hockey/players_age_at_death_no_shoot_catch",
    "/v1/hockey/goalies_by_year_range_and_team_count?start_year=2000&end_year=2005&min_teams=2",
    "/v1/hockey/average_weight_by_height?min_height=72",
    "/v1/hockey/goalies_by_minutes_played?min_minutes=5000",
    "/v1/hockey/goalies_by_leagues?league1=PCHA&league2=NHL&min_leagues=1",
    "/v1/hockey/top_goalie_minutes_deceased",
    "/v1/hockey/goalie_details_by_team_year_range?tm_id=COL&start_year=2000&end_year=2010",
    "/v1/hockey/goalie_details_by_min_games?min_games=10",
    "/v1/hockey/best_save_percentage_goalie",
    "/v1/hockey/goalie_details_by_year_height?year=2005&max_height=72",
    "/v1/hockey/goalie_details_by_player_id?player_id=aubinje01",
    "/v1/hockey/avg_minutes_most_games_goalie",
    "/v1/hockey/goalie_details_by_sho_ga_ratio?sho_ga_ratio=0.05",
    "/v1/hockey/players_different_birth_death_country",
    "/v1/hockey/players_nhl_wha",
    "/v1/hockey/living_players_by_position?position=%/%",
    "/v1/hockey/tallest_player?limit=1",
    "/v1/hockey/height_difference_by_birth_year?birth_year=1990",
    "/v1/hockey/goalies_by_shooting_position?shoot_catch=L&pos=G",
    "/v1/hockey/goalies_by_birth_country_position?birth_country=Canada&pos=G",
    "/v1/hockey/goalies_by_team_rank_position?team_name=Boston%20Bruins&rank=1&pos=G",
    "/v1/hockey/teams_by_debut_year?first_nhl=1950",
    "/v1/hockey/coaches_who_were_players",
    "/v1/hockey/top_coach_by_win_rate?limit=1",
    "/v1/hockey/top_coach_details_by_win_rate?limit=1",
    "/v1/hockey/top_goalie_by_coaching_experience?team_id=MTL&limit=1",
    "/v1/hockey/goalies_with_more_losses_than_wins?pos=G&min_years=2",
    "/v1/hockey/goalies_born_in_year_with_null_shootCatch?birth_year=1987",
    "/v1/hockey/average_minutes_per_game_for_player?player_id=aebisda01",
    "/v1/hockey/goalies_with_average_minutes_per_game_who_have_died?avg_minutes_per_game=0.5",
    "/v1/hockey/count_non_null_notes_in_awards_misc",
    "/v1/hockey/count_players_by_position_and_year?pos=G&year=1983",
    "/v1/hockey/count_coaches_by_year_and_notes?year=2007&notes=interim",
    "/v1/hockey/count_combined_shutouts_by_year_and_role?year=1977&role=R",
    "/v1/hockey/count_goalies_with_null_eng_by_year?year=2005",
    "/v1/hockey/distinct_years_for_goalies_in_league_with_non_null_sa?lg_id=NHL",
    "/v1/hockey/goalies_count_postw_eq_postl",
    "/v1/hockey/hof_inductees_by_year?year=1978",
    "/v1/hockey/hof_inductees_count_by_category?category=Builder",
    "/v1/hockey/hof_inductees_count_by_year_and_category?year=1980&category=Player",
    "/v1/hockey/hof_inductees_nicknames_by_year?year=2007",
    "/v1/hockey/hof_status_tallest_player",
    "/v1/hockey/coach_awards_by_birth_country?birth_country=Canada",
    "/v1/hockey/coach_count_by_wins_and_birth_country?wins=30&birth_country=USA",
    "/v1/hockey/coach_count_by_league_and_birth_country?league_id=NHL&birth_country=Canada",
    "/v1/hockey/coach_awards_by_birth_year?birth_year=1952",
    "/v1/hockey/count_coaches_by_year_birth_city?year=1940&birth_city=Toronto",
    "/v1/hockey/count_deceased_coaches_awarded_after_year?year=1940",
    "/v1/hockey/distinct_awards_deceased_coaches_league?lg_id=NHL",
    "/v1/hockey/count_distinct_coaches_hof_weight?weight=195",
    "/v1/hockey/distinct_names_hof_unknown_shoot_catch",
    "/v1/hockey/count_hof_players_coaches",
    "/v1/hockey/distinct_birth_cities_players_awarded_year?year=1970",
    "/v1/hockey/count_players_award_birth_city?award=All-Rookie&birth_city=Toronto",
    "/v1/hockey/count_deceased_players_award?award=All-Rookie",
    "/v1/hockey/count_distinct_deceased_players_state?death_state=MA",
    "/v1/hockey/awards_by_death_city?death_city=Kemptville",
    "/v1/hockey/nicknames_by_award_birth_month?award=All-Rookie&birth_month=3",
    "/v1/hockey/hof_player_count_by_birth_months?birth_month1=7&birth_month2=8",
    "/v1/hockey/birth_month_most_awards",
    "/v1/hockey/birth_year_most_awards",
    "/v1/hockey/birth_country_most_awards",
    "/v1/hockey/birth_country_most_hof_players",
    "/v1/hockey/positions_by_country_award?birth_country=Canada&award=All-Rookie",
    "/v1/hockey/average_bmi_hof_players",
    "/v1/hockey/percentage_hof_players_by_country?birth_country=USA",
    "/v1/hockey/goalie_years_count?playerID=healygl01",
    "/v1/hockey/goalie_team_id?playerID=roypa01&year=1992",
    "/v1/hockey/goalie_games_played?playerID=rutlewa01&year=1967",
    "/v1/hockey/goalie_minutes_played?playerID=valiqst01&year=2007",
    "/v1/hockey/goalie_wins?playerID=vanbijo01&year=1990",
    "/v1/hockey/goalie_ties_overtime_losses?playerID=vernomi01&year=1998",
    "/v1/hockey/sum_wins_coaches_award_year?year=1933&award=Second%20Team%20All-Star",
    "/v1/hockey/check_player_note?note=posthumous&legendsID=P194502",
    "/v1/hockey/player_position?firstName=Mike&lastName=Antonovich",
    "/v1/hockey/coach_birth_country?year=1998&notes=co-coach%20with%20Dave%20Lewis",
    "/v1/hockey/goalie_name_by_stint?stint=3",
    "/v1/hockey/goalie_name_by_eng_sum?eng_sum=10",
    "/v1/hockey/goalie_shootcatch_by_year?year=2010",
    "/v1/hockey/goalie_name_by_year_and_ga?year=2002&ga=150",
    "/v1/hockey/goalie_name_by_team_id?tmID=NJD",
    "/v1/hockey/team_name_by_year?year=2010",
    "/v1/hockey/sum_sho_by_year?year=1995",
    "/v1/hockey/coach_id_name_nick_by_year?year=2009",
    "/v1/hockey/coach_id_by_year?year=2011",
    "/v1/hockey/games_played_by_player_id_and_year?playerID=broadpu01&year=1922",
    "/v1/hockey/count_years_by_player?playerID=cleghsp01",
    "/v1/hockey/goals_by_player_year?playerID=dyeba01&year=1921",
    "/v1/hockey/top_assists_player",
    "/v1/hockey/league_id_by_player_year?playerID=adamsja01&year=1920",
    "/v1/hockey/distinct_positions_by_player?playerID=hartgi01",
    "/v1/hockey/sum_wins_by_team_goalies_year?year=2011&distinct_goalies=3",
    "/v1/hockey/birth_year_top_postsa_goalie?year=2008",
    "/v1/hockey/coaching_career_span?firstName=Don&lastName=Waddell",
    "/v1/hockey/shoot_catch_top_sho_goalie?year=1996",
    "/v1/hockey/birth_details_top_ga_goalie?year=1965",
    "/v1/hockey/top_player_by_plus_minus?year=1981",
    "/v1/hockey/top_player_weight_by_ppg?year=2000",
    "/v1/hockey/shoot_catch_by_shg?year=1989&shg_count=7",
    "/v1/hockey/top_player_by_gwg?year=1986",
    "/v1/hockey/player_sog?first_name=Cam&last_name=Neely&year=1990",
    "/v1/hockey/top_coach_by_bench_minor?year=2003",
    "/v1/hockey/top_goalie_by_ga?year=1978&weight=190",
    "/v1/hockey/win_percentage_difference?year1=2006&year2=2005&team_name=Vancouver%20Canucks&coach_first_name=Alain&coach_last_name=Vigneault",
    "/v1/hockey/top_goalie_by_save_percentage?year=2011",
    "/v1/hockey/player_coach_count",
    "/v1/hockey/team_details_highest_t?limit=1",
    "/v1/hockey/most_awarded_coach?lgID=NHL&limit=1",
    "/v1/hockey/highest_ppg_ppc_ratio?limit=1",
    "/v1/hockey/hof_inductees_proportion?divisor=30&start_year=1950&end_year=1980&category=Player",
    "/v1/hockey/most_common_birth_country_year?limit=1",
    "/v1/hockey/team_highest_shutouts?limit=1",
    "/v1/hockey/team_highest_regular_wins?year=2005&limit=1",
    "/v1/hockey/player_positions_by_birth_date?limit=8",
    "/v1/hockey/goalie_highest_post_ga?lgID=WCHL&year=1924&limit=1",
    "/v1/hockey/team_highest_october_losses?year=2006&limit=1",
    "/v1/hockey/count_players_first_nhl_year?first_nhl=2011",
    "/v1/hockey/top_goal_scorer_year_age",
    "/v1/hockey/average_height_by_position?pos1=LW&pos2=L/C",
    "/v1/hockey/top_assists_scorer?lg_id=NHL",
    "/v1/hockey/top_coach_by_wins",
    "/v1/hockey/team_name_by_series_note_year?note=EX&year=1912",
    "/v1/hockey/lowest_plus_minus_player",
    "/v1/hockey/top_games_played_by_pim_year_range?min_pim=200&max_pim=250&min_year=2003&max_year=2005",
    "/v1/hockey/top_goalie_by_goals_against",
    "/v1/hockey/top_coach_by_awards",
    "/v1/hockey/sum_goals_before_first_award?award=Second%20Team%20All-Star",
    "/v1/hockey/count_distinct_opponents_by_year?year=1915",
    "/v1/hockey/player_with_highest_gwg_in_shg_range?min_shg=1&max_shg=5",
    "/v1/hockey/sum_losses_interim_coaches?notes=interim&year=1997&team_name=Tampa%20Bay%20Lightning",
    "/v1/hockey/team_with_highest_assists_min_sha?min_sha=7",
    "/v1/hockey/coach_ids_by_year_and_rank?year=1969&rank=4",
    "/v1/hockey/distinct_team_names_by_half_rank_year_range?half=1&rank=1&min_year=1917&max_year=1920",
    "/v1/hockey/player_details_by_age_range_and_team_goals?min_age=18&max_age=24&max_team_goals=5",
    "/v1/hockey/average_bench_minor_penalties?team_name=St.%20Louis%20Blues",
    "/v1/hockey/ppg_to_ppc_ratio?year=2009&team_name=New%20York%20Rangers",
    "/v1/hockey/team_with_highest_points",
    "/v1/hockey/count_teams_with_more_wins_and_points?year=2006&points=100",
    "/v1/hockey/team_with_highest_avg_bench_minor?year=2006",
    "/v1/hockey/top_3_teams_with_highest_pim?year=2006",
    "/v1/hockey/team_with_highest_ppc_among_specified?year=1995&team1=Florida%20Panthers&team2=Edmonton%20Oilers&team3=Los%20Angeles%20Kings",
    "/v1/hockey/teams_with_more_wins_than_losses?year=1917",
    "/v1/hockey/teams_in_year?year=1922",
    "/v1/hockey/count_teams_with_points_greater_than?year=1922&points=20",
    "/v1/hockey/penalty_minutes_by_rank?year=1923&rank=2",
    "/v1/hockey/team_with_highest_total_wins",
    "/v1/hockey/team_vs_team_win_count?year=1985&team_name=Philadelphia%20Flyers&opponent_name=Boston%20Bruins",
    "/v1/hockey/team_names_by_year_and_tie_status?year=1909&tie_status=1",
    "/v1/hockey/coach_first_names_by_team?team_name=Montreal%20Canadiens",
    "/v1/hockey/count_hof_ids_by_team?team_name=Montreal%20Canadiens",
    "/v1/hockey/tallest_coach_by_team?team_name=Montreal%20Canadiens",
    "/v1/hockey/coach_first_names_by_year?year=1922",
    "/v1/hockey/count_coach_ids_by_team_and_birth_month?team_name=Philadelphia%20Flyers&birth_month=3",
    "/v1/hockey/count_coach_ids_by_team_and_birth_country?team_name=Philadelphia%20Flyers&birth_country=USA",
    "/v1/hockey/count_coach_ids_by_team_and_death_year?team_name=Buffalo%20Sabres",
    "/v1/hockey/coach_nicknames_by_team_and_birth_country?team_name=Buffalo%20Sabres&birth_country=USA",
    "/v1/hockey/count_distinct_coaches?year=1922&birth_country=USA",
    "/v1/hockey/top_team_by_goals?year=2000&opponent_name=Buffalo%20Sabres",
    "/v1/hockey/distinct_team_names?opponent_name=Buffalo%20Sabres",
    "/v1/hockey/penalty_minutes?team_name=Montreal%20Canadiens&year=1918",
    "/v1/hockey/wins_by_penalty_minutes?team_name=Montreal%20Canadiens&penalty_minutes=24",
    "/v1/hockey/years_by_penalty_minutes?team_name=Montreal%20Canadiens&penalty_minutes=49",
    "/v1/hockey/earliest_coach_first_name?team_name=Montreal%20Canadiens",
    "/v1/hockey/average_win_ratio?team_name=Buffalo%20Sabres&year=2000",
    "/v1/hockey/average_penalty_minutes?team_name=Montreal%20Canadiens",
    "/v1/hockey/win_ratio?team_name=Montreal%20Canadiens",
    "/v1/hockey/team_win_ratio?team_name=Montreal%20Canadiens",
    "/v1/hockey/latest_inductee?category1=Player&category2=Builder",
    "/v1/hockey/inductees_by_category_year_range?category=Builder&start_year=1970&end_year=1979",
    "/v1/hockey/inductee_count_by_category_year?category=Player&year=1958",
    "/v1/hockey/player_coach_hof_count",
    "/v1/hockey/player_count_by_last_name?last_name=Green",
    "/v1/hockey/bench_minor_penalties?first_name=Scotty&last_name=Bowman&year=1982",
    "/v1/hockey/tallest_goalie_by_even_strength_goals?even_strength_goals=10",
    "/v1/hockey/goalie_stats_by_team_year?year=1973&tmID=QUN&gp=32",
    "/v1/hockey/team_count_by_goalie_last_name?last_name=Young",
    "/v1/hockey/goalies_names_team_year?team_id=BOS&year=1972",
    "/v1/hockey/sum_games_played_legends_id?legends_id=P196402",
    "/v1/hockey/top_goalie_team_position?team_name=Quebec%20Bulldogs&position=D",
    "/v1/hockey/count_distinct_goalies_team?team_name=Calgary%20Flames",
    "/v1/hockey/top_goalie_goals_against_team?team_name=Minnesota%20North%20Stars",
    "/v1/hockey/count_hof_goalies_team?team_name=Haileybury%20Hockey%20Club",
    "/v1/hockey/top_shoot_catch_goalie_year?year=2010",
    "/v1/hockey/percentage_change_games_played?year1=2005&year2=2006&first_name=David&last_name=Aebischer",
    "/v1/hockey/player_coach_ids",
    "/v1/hockey/player_ids_avg_height?avg_height=75",
    "/v1/hockey/player_ids_by_weight?limit=5",
    "/v1/hockey/player_names_by_birth_country?birth_country=Finland",
    "/v1/hockey/deceased_player_ids",
    "/v1/hockey/coach_first_names_after_year?year=2000",
    "/v1/hockey/coach_height_weight_by_year?year=1930",
    "/v1/hockey/player_goals_and_age?player_id=aaltoan01",
    "/v1/hockey/coach_ids_not_players",
    "/v1/hockey/player_ids_by_position_weight?pos_like=%25L%25&weight=200&pos=L",
    "/v1/hockey/games_played_by_birth_country?birth_country=USA",
    "/v1/hockey/team_points_and_coach_id?team_id=ANA",
    "/v1/hockey/sum_goals_team_division_year?div_id=EW&tm_id=BIR&year=1976",
    "/v1/hockey/sum_losses_assists_team_year?tm_id=BOS&year=2010",
    "/v1/hockey/sum_weights_team_year?year=1997&tm_id=ANA",
    "/v1/hockey/shortest_player_year_range?start_year=1925&end_year=1936",
    "/v1/hockey/team_birth_country_highest_win_ratio?year=2000",
    "/v1/hockey/win_ratio_player_highest_goals?tm_id=CAR&year=1998"
]
