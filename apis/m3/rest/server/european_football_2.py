from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/european_football_2/european_football_2.sqlite')
cursor = conn.cursor()

# Endpoint to get the player API ID with the highest overall rating
@app.get("/v1/european_football_2/top_player_by_overall_rating", operation_id="get_top_player_by_overall_rating", summary="Retrieves the player with the highest overall rating, allowing the user to limit the number of results returned.")
async def get_top_player_by_overall_rating(limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT player_api_id FROM Player_Attributes ORDER BY overall_rating DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"player_api_id": []}
    return {"player_api_id": result[0]}

# Endpoint to get the player name with the highest height
@app.get("/v1/european_football_2/tallest_player", operation_id="get_tallest_player", summary="Retrieves the name of the tallest player(s) from the database, with the option to limit the number of results returned.")
async def get_tallest_player(limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT player_name FROM Player ORDER BY height DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the preferred foot of the player with the lowest potential
@app.get("/v1/european_football_2/preferred_foot_lowest_potential", operation_id="get_preferred_foot_lowest_potential", summary="Retrieve the preferred foot of the player with the lowest potential, with the option to limit the number of results returned.")
async def get_preferred_foot_lowest_potential(limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT preferred_foot FROM Player_Attributes WHERE potential IS NOT NULL ORDER BY potential ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"preferred_foot": []}
    return {"preferred_foot": result[0]}

# Endpoint to get the count of players with specific overall rating range and defensive work rate
@app.get("/v1/european_football_2/count_players_by_rating_and_work_rate", operation_id="get_count_players_by_rating_and_work_rate", summary="Retrieves the number of players within a specified range of overall ratings and a specific defensive work rate. The response includes a count of players who meet the provided criteria.")
async def get_count_players_by_rating_and_work_rate(min_rating: int = Query(..., description="Minimum overall rating"), max_rating: int = Query(..., description="Maximum overall rating"), defensive_work_rate: str = Query(..., description="Defensive work rate")):
    cursor.execute("SELECT COUNT(id) FROM Player_Attributes WHERE overall_rating BETWEEN ? AND ? AND defensive_work_rate = ?", (min_rating, max_rating, defensive_work_rate))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top players by crossing attribute
@app.get("/v1/european_football_2/top_players_by_crossing", operation_id="get_top_players_by_crossing", summary="Retrieve a list of top-performing players ranked by their crossing ability. The list is limited to the specified number of results.")
async def get_top_players_by_crossing(limit: int = Query(5, description="Limit the number of results")):
    cursor.execute("SELECT id FROM Player_Attributes ORDER BY crossing DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"player_ids": []}
    return {"player_ids": [row[0] for row in result]}

# Endpoint to get the league with the highest total goals in a specific season
@app.get("/v1/european_football_2/top_league_by_total_goals", operation_id="get_top_league_by_total_goals", summary="Retrieves the league with the highest total goals scored in a specific season. The operation calculates the total goals scored in each league for the given season and returns the league with the highest total. The number of results can be limited by specifying the desired limit.")
async def get_top_league_by_total_goals(season: str = Query(..., description="Season in 'YYYY/YYYY' format"), limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT t2.name FROM Match AS t1 INNER JOIN League AS t2 ON t1.league_id = t2.id WHERE t1.season = ? GROUP BY t2.name ORDER BY SUM(t1.home_team_goal + t1.away_team_goal) DESC LIMIT ?", (season, limit))
    result = cursor.fetchone()
    if not result:
        return {"league_name": []}
    return {"league_name": result[0]}

# Endpoint to get the team with the fewest home losses in a specific season
@app.get("/v1/european_football_2/team_with_fewest_home_losses", operation_id="get_team_with_fewest_home_losses", summary="Retrieve the team with the fewest home losses in a given season. This operation considers matches where the home team scored fewer goals than the away team. The results can be limited to a specific number of teams. The team names are returned in ascending order based on the number of home losses.")
async def get_team_with_fewest_home_losses(season: str = Query(..., description="Season in 'YYYY/YYYY' format"), limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT teamDetails.team_long_name FROM Match AS matchData INNER JOIN Team AS teamDetails ON matchData.home_team_api_id = teamDetails.team_api_id WHERE matchData.season = ? AND matchData.home_team_goal - matchData.away_team_goal < 0 GROUP BY matchData.home_team_api_id ORDER BY COUNT(*) ASC LIMIT ?", (season, limit))
    result = cursor.fetchone()
    if not result:
        return {"team_long_name": []}
    return {"team_long_name": result[0]}

# Endpoint to get the top players by penalties attribute
@app.get("/v1/european_football_2/top_players_by_penalties", operation_id="get_top_players_by_penalties", summary="Retrieve a list of top-performing football players based on their penalty performance. The list is sorted in descending order, with the most successful penalty takers appearing first. The number of players returned can be limited by specifying the desired limit.")
async def get_top_players_by_penalties(limit: int = Query(10, description="Limit the number of results")):
    cursor.execute("SELECT t2.player_name FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.id = t2.id ORDER BY t1.penalties DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the team with the most away wins in a specific league and season
@app.get("/v1/european_football_2/team_with_most_away_wins", operation_id="get_team_with_most_away_wins", summary="Retrieves the team with the highest number of away wins in a specified league and season. The league is identified by its name, and the season is denoted by a year range. The operation returns the team's full name. Optionally, the number of results can be limited.")
async def get_team_with_most_away_wins(league_name: str = Query(..., description="League name"), season: str = Query(..., description="Season in 'YYYY/YYYY' format"), limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT teamInfo.team_long_name FROM League AS leagueData INNER JOIN Match AS matchData ON leagueData.id = matchData.league_id INNER JOIN Team AS teamInfo ON matchData.away_team_api_id = teamInfo.team_api_id WHERE leagueData.name = ? AND matchData.season = ? AND matchData.away_team_goal - matchData.home_team_goal > 0 GROUP BY matchData.away_team_api_id ORDER BY COUNT(*) DESC LIMIT ?", (league_name, season, limit))
    result = cursor.fetchone()
    if not result:
        return {"team_long_name": []}
    return {"team_long_name": result[0]}

# Endpoint to get the teams with the lowest build-up play speed
@app.get("/v1/european_football_2/teams_with_lowest_build_up_play_speed", operation_id="get_teams_with_lowest_build_up_play_speed", summary="Retrieve a list of football teams with the lowest build-up play speed, ranked from slowest to fastest. The number of teams returned can be limited by specifying the desired limit.")
async def get_teams_with_lowest_build_up_play_speed(limit: int = Query(4, description="Limit the number of results")):
    cursor.execute("SELECT t1.buildUpPlaySpeed FROM Team_Attributes AS t1 INNER JOIN Team AS t2 ON t1.team_api_id = t2.team_api_id ORDER BY t1.buildUpPlaySpeed ASC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"build_up_play_speeds": []}
    return {"build_up_play_speeds": [row[0] for row in result]}

# Endpoint to get the league name with the most draws in a given season
@app.get("/v1/european_football_2/league_most_draws", operation_id="get_league_most_draws", summary="Retrieves the name of the league with the highest number of draws in a specified season. The season is provided in the 'YYYY/YYYY' format.")
async def get_league_most_draws(season: str = Query(..., description="Season in 'YYYY/YYYY' format")):
    cursor.execute("SELECT t2.name FROM Match AS t1 INNER JOIN League AS t2 ON t1.league_id = t2.id WHERE t1.season = ? AND t1.home_team_goal = t1.away_team_goal GROUP BY t2.name ORDER BY COUNT(t1.id) DESC LIMIT 1", (season,))
    result = cursor.fetchone()
    if not result:
        return {"league_name": []}
    return {"league_name": result[0]}

# Endpoint to get the age of players with a sprint speed above a certain threshold within a given date range
@app.get("/v1/european_football_2/player_age_sprint_speed", operation_id="get_player_age_sprint_speed", summary="Retrieve the ages of football players who have demonstrated a sprint speed above a specified threshold within a defined date range. The operation filters players based on their birthdays and attributes, considering the start and end years, and the minimum required sprint speed.")
async def get_player_age_sprint_speed(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), sprint_speed: int = Query(..., description="Minimum sprint speed")):
    cursor.execute("SELECT DISTINCT DATETIME() - T2.birthday age FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id WHERE STRFTIME('%Y',t1.`date`) >= ? AND STRFTIME('%Y',t1.`date`) <= ? AND t1.sprint_speed >= ?", (start_year, end_year, sprint_speed))
    result = cursor.fetchall()
    if not result:
        return {"ages": []}
    return {"ages": [row[0] for row in result]}

# Endpoint to get the league with the maximum number of matches
@app.get("/v1/european_football_2/league_max_matches", operation_id="get_league_max_matches", summary="Retrieves the league with the highest number of matches. The operation calculates the total number of matches for each league and identifies the league with the maximum count. The response includes the name of the league and the maximum number of matches.")
async def get_league_max_matches():
    cursor.execute("SELECT t2.name, t1.max_count FROM League AS t2 JOIN (SELECT league_id, MAX(cnt) AS max_count FROM (SELECT league_id, COUNT(id) AS cnt FROM Match GROUP BY league_id) AS subquery) AS t1 ON t1.league_id = t2.id")
    result = cursor.fetchone()
    if not result:
        return {"league_name": [], "max_matches": []}
    return {"league_name": result[0], "max_matches": result[1]}

# Endpoint to get the average height of players born between two years
@app.get("/v1/european_football_2/average_player_height", operation_id="get_average_player_height", summary="Retrieves the average height of football players born between the specified start and end years. The calculation is based on the sum of all player heights divided by the total number of players within the given year range.")
async def get_average_player_height(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT SUM(height) / COUNT(id) FROM Player WHERE SUBSTR(birthday, 1, 4) BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the player with the highest overall rating in a given year
@app.get("/v1/european_football_2/top_player_by_year", operation_id="get_top_player_by_year", summary="Retrieves the player with the highest overall rating from the given year. The year should be provided in 'YYYY' format.")
async def get_top_player_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT player_api_id FROM Player_Attributes WHERE SUBSTR(`date`, 1, 4) = ? ORDER BY overall_rating DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"player_api_id": []}
    return {"player_api_id": result[0]}

# Endpoint to get distinct team FIFA API IDs with build-up play speed within a given range
@app.get("/v1/european_football_2/team_build_up_play_speed", operation_id="get_team_build_up_play_speed", summary="Retrieve a unique list of team FIFA API identifiers that have a build-up play speed within the specified range. The minimum and maximum build-up play speed values are provided as input parameters to filter the results.")
async def get_team_build_up_play_speed(min_speed: int = Query(..., description="Minimum build-up play speed"), max_speed: int = Query(..., description="Maximum build-up play speed")):
    cursor.execute("SELECT DISTINCT team_fifa_api_id FROM Team_Attributes WHERE buildUpPlaySpeed > ? AND buildUpPlaySpeed < ?", (min_speed, max_speed))
    result = cursor.fetchall()
    if not result:
        return {"team_fifa_api_ids": []}
    return {"team_fifa_api_ids": [row[0] for row in result]}

# Endpoint to get distinct team long names with build-up play passing above the average in a given year
@app.get("/v1/european_football_2/team_build_up_play_passing", operation_id="get_team_build_up_play_passing", summary="Retrieves a list of unique team names that have demonstrated above-average build-up play passing in a specified year. The year is provided in 'YYYY' format.")
async def get_team_build_up_play_passing(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT t4.team_long_name FROM Team_Attributes AS t3 INNER JOIN Team AS t4 ON t3.team_api_id = t4.team_api_id WHERE SUBSTR(t3.`date`, 1, 4) = ? AND t3.buildUpPlayPassing > ( SELECT CAST(SUM(t2.buildUpPlayPassing) AS REAL) / COUNT(t1.id) FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE STRFTIME('%Y',t2.`date`) = ?)", (year, year))
    result = cursor.fetchall()
    if not result:
        return {"team_long_names": []}
    return {"team_long_names": [row[0] for row in result]}

# Endpoint to get the percentage of left-footed players born between two years
@app.get("/v1/european_football_2/percentage_left_footed_players", operation_id="get_percentage_left_footed_players", summary="Retrieves the percentage of left-footed players born between the specified start and end years. The calculation is based on the total number of players born within the given range and the subset of those who are left-footed.")
async def get_percentage_left_footed_players(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN t2.preferred_foot = 'left' THEN t1.id ELSE NULL END) AS REAL) * 100 / COUNT(t1.id) percent FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE SUBSTR(t1.birthday, 1, 4) BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the top 5 leagues with the fewest total goals
@app.get("/v1/european_football_2/leagues_fewest_goals", operation_id="get_leagues_fewest_goals", summary="Retrieves the names of the top 5 leagues with the lowest total number of goals scored in all matches. The data is aggregated from the match records and sorted in ascending order based on the total goals scored.")
async def get_leagues_fewest_goals():
    cursor.execute("SELECT t1.name, SUM(t2.home_team_goal) + SUM(t2.away_team_goal) FROM League AS t1 INNER JOIN Match AS t2 ON t1.id = t2.league_id GROUP BY t1.name ORDER BY SUM(t2.home_team_goal) + SUM(t2.away_team_goal) ASC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"leagues": []}
    return {"leagues": [{"name": row[0], "total_goals": row[1]} for row in result]}

# Endpoint to get the average number of long shots per date for a specific player
@app.get("/v1/european_football_2/average_long_shots_player", operation_id="get_average_long_shots_player", summary="Retrieves the average number of long shots per date for a specific football player. The player is identified by their name, which is provided as an input parameter. The operation calculates the average by summing the total number of long shots and dividing it by the number of dates the player has played.")
async def get_average_long_shots_player(player_name: str = Query(..., description="Player name")):
    cursor.execute("SELECT CAST(SUM(t2.long_shots) AS REAL) / COUNT(t2.`date`) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_long_shots": []}
    return {"average_long_shots": result[0]}

# Endpoint to get the top players by heading accuracy based on height
@app.get("/v1/european_football_2/top_players_by_heading_accuracy", operation_id="get_top_players_by_heading_accuracy", summary="Retrieves a list of top players with the highest heading accuracy, considering only those taller than the specified minimum height. The number of players returned is determined by the provided limit.")
async def get_top_players_by_heading_accuracy(min_height: int = Query(..., description="Minimum height of the player"), limit: int = Query(..., description="Number of top players to return")):
    cursor.execute("SELECT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.height > ? GROUP BY t1.id ORDER BY CAST(SUM(t2.heading_accuracy) AS REAL) / COUNT(t2.player_fifa_api_id) DESC LIMIT ?", (min_height, limit))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get teams with specific build-up play dribbling class and chance creation passing
@app.get("/v1/european_football_2/teams_by_build_up_play_dribbling_class", operation_id="get_teams_by_build_up_play_dribbling_class", summary="Retrieves a list of teams with a specified build-up play dribbling class and chance creation passing below the average for a given year. The teams are ordered by their chance creation passing in descending order.")
async def get_teams_by_build_up_play_dribbling_class(build_up_play_dribbling_class: str = Query(..., description="Build-up play dribbling class"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT t3.team_long_name FROM Team AS t3 INNER JOIN Team_Attributes AS t4 ON t3.team_api_id = t4.team_api_id WHERE t4.buildUpPlayDribblingClass = ? AND t4.chanceCreationPassing < ( SELECT CAST(SUM(t2.chanceCreationPassing) AS REAL) / COUNT(t1.id) FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t2.buildUpPlayDribblingClass = ? AND SUBSTR(t2.date, 1, 4) = ?) ORDER BY t4.chanceCreationPassing DESC", (build_up_play_dribbling_class, build_up_play_dribbling_class, year))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": [row[0] for row in result]}

# Endpoint to get leagues with more home goals than away goals in a specific season
@app.get("/v1/european_football_2/leagues_with_more_home_goals", operation_id="get_leagues_with_more_home_goals", summary="Retrieves a list of football leagues where the average number of goals scored by home teams surpasses the average number of goals scored by away teams in a specified season. The season is provided in the 'YYYY/YYYY' format.")
async def get_leagues_with_more_home_goals(season: str = Query(..., description="Season in 'YYYY/YYYY' format")):
    cursor.execute("SELECT t1.name FROM League AS t1 INNER JOIN Match AS t2 ON t1.id = t2.league_id WHERE t2.season = ? GROUP BY t1.name HAVING (CAST(SUM(t2.home_team_goal) AS REAL) / COUNT(DISTINCT t2.id)) - (CAST(SUM(t2.away_team_goal) AS REAL) / COUNT(DISTINCT t2.id)) > 0", (season,))
    result = cursor.fetchall()
    if not result:
        return {"leagues": []}
    return {"leagues": [row[0] for row in result]}

# Endpoint to get the short name of a team based on its long name
@app.get("/v1/european_football_2/team_short_name_by_long_name", operation_id="get_team_short_name_by_long_name", summary="Retrieves the abbreviated name of a football team using its full name as input. This operation is useful for obtaining the short name of a team when only its long name is known.")
async def get_team_short_name_by_long_name(team_long_name: str = Query(..., description="Long name of the team")):
    cursor.execute("SELECT team_short_name FROM Team WHERE team_long_name = ?", (team_long_name,))
    result = cursor.fetchone()
    if not result:
        return {"short_name": []}
    return {"short_name": result[0]}

# Endpoint to get players born in a specific month and year
@app.get("/v1/european_football_2/players_by_birth_month_year", operation_id="get_players_by_birth_month_year", summary="Retrieves a list of player names born in a specific month and year. The birth month and year should be provided in 'YYYY-MM' format.")
async def get_players_by_birth_month_year(birth_month_year: str = Query(..., description="Birth month and year in 'YYYY-MM' format")):
    cursor.execute("SELECT player_name FROM Player WHERE SUBSTR(birthday, 1, 7) = ?", (birth_month_year,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get distinct attacking work rates for a specific player
@app.get("/v1/european_football_2/attacking_work_rate_by_player", operation_id="get_attacking_work_rate_by_player", summary="Retrieve the unique attacking work rates associated with a specific player. The operation requires the player's name as input and returns a list of distinct attacking work rates for the player.")
async def get_attacking_work_rate_by_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT DISTINCT t2.attacking_work_rate FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"attacking_work_rates": []}
    return {"attacking_work_rates": [row[0] for row in result]}

# Endpoint to get distinct build-up play positioning classes for a specific team
@app.get("/v1/european_football_2/build_up_play_positioning_class_by_team", operation_id="get_build_up_play_positioning_class_by_team", summary="Retrieve the unique build-up play positioning classes associated with a specific team. The team is identified by its long name.")
async def get_build_up_play_positioning_class_by_team(team_long_name: str = Query(..., description="Long name of the team")):
    cursor.execute("SELECT DISTINCT t2.buildUpPlayPositioningClass FROM Team AS t1 INNER JOIN Team_attributes AS t2 ON t1.team_fifa_api_id = t2.team_fifa_api_id WHERE t1.team_long_name = ?", (team_long_name,))
    result = cursor.fetchall()
    if not result:
        return {"build_up_play_positioning_classes": []}
    return {"build_up_play_positioning_classes": [row[0] for row in result]}

# Endpoint to get heading accuracy for a specific player on a specific date
@app.get("/v1/european_football_2/heading_accuracy_by_player_date", operation_id="get_heading_accuracy_by_player_date", summary="Retrieves the heading accuracy of a specific football player on a given date. The operation requires the player's name and the date in 'YYYY-MM-DD' format as input parameters. The returned data is a single value representing the player's heading accuracy on the specified date.")
async def get_heading_accuracy_by_player_date(player_name: str = Query(..., description="Name of the player"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT t2.heading_accuracy FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ? AND SUBSTR(t2.date, 1, 10) = ?", (player_name, date))
    result = cursor.fetchone()
    if not result:
        return {"heading_accuracy": []}
    return {"heading_accuracy": result[0]}

# Endpoint to get overall rating for a specific player in a specific year
@app.get("/v1/european_football_2/overall_rating_by_player_year", operation_id="get_overall_rating_by_player_year", summary="Retrieves the overall rating of a specific football player for a given year. The operation requires the player's name and the year as input parameters. The player's name should match exactly, and the year should be provided in the 'YYYY' format. The returned overall rating is based on the player's attributes for the specified year.")
async def get_overall_rating_by_player_year(player_name: str = Query(..., description="Name of the player"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT t2.overall_rating FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ? AND strftime('%Y', t2.date) = ?", (player_name, year))
    result = cursor.fetchone()
    if not result:
        return {"overall_rating": []}
    return {"overall_rating": result[0]}

# Endpoint to get the count of matches in a specific league and season
@app.get("/v1/european_football_2/match_count_by_league_season", operation_id="get_match_count_by_league_season", summary="Retrieves the total number of matches played in a specific football league during a given season. The operation requires the season in 'YYYY/YYYY' format and the name of the league as input parameters.")
async def get_match_count_by_league_season(season: str = Query(..., description="Season in 'YYYY/YYYY' format"), league_name: str = Query(..., description="Name of the league")):
    cursor.execute("SELECT COUNT(t2.id) FROM League AS t1 INNER JOIN Match AS t2 ON t1.id = t2.league_id WHERE t2.season = ? AND t1.name = ?", (season, league_name))
    result = cursor.fetchone()
    if not result:
        return {"match_count": []}
    return {"match_count": result[0]}

# Endpoint to get the preferred foot of the most recently born player
@app.get("/v1/european_football_2/preferred_foot_most_recent_player", operation_id="get_preferred_foot_most_recent_player", summary="Retrieves the preferred foot of the youngest player in the database. The operation identifies the most recently born player and returns their preferred foot, as recorded in the Player_Attributes table.")
async def get_preferred_foot_most_recent_player():
    cursor.execute("SELECT t2.preferred_foot FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id ORDER BY t1.birthday DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"preferred_foot": []}
    return {"preferred_foot": result[0]}

# Endpoint to get the names of players with the highest potential
@app.get("/v1/european_football_2/players_with_highest_potential", operation_id="get_players_with_highest_potential", summary="Retrieves the names of football players with the highest potential rating. The operation identifies players with the maximum potential rating from the Player_Attributes table and returns their names from the Player table.")
async def get_players_with_highest_potential():
    cursor.execute("SELECT DISTINCT(t1.player_name) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.potential = (SELECT MAX(potential) FROM Player_Attributes)")
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the count of players with a specific weight and preferred foot
@app.get("/v1/european_football_2/count_players_weight_preferred_foot", operation_id="get_count_players_weight_preferred_foot", summary="Retrieves the count of distinct players who weigh less than the specified weight and have the specified preferred foot. The weight and preferred foot are provided as input parameters.")
async def get_count_players_weight_preferred_foot(weight: int = Query(..., description="Weight of the player"), preferred_foot: str = Query(..., description="Preferred foot of the player")):
    cursor.execute("SELECT COUNT(DISTINCT t1.id) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.weight < ? AND t2.preferred_foot = ?", (weight, preferred_foot))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the short names of teams with a specific chance creation passing class
@app.get("/v1/european_football_2/team_short_names_chance_creation_passing_class", operation_id="get_team_short_names_chance_creation_passing_class", summary="Retrieve the unique short names of football teams that exhibit a specified chance creation passing class. This operation filters teams based on their chance creation passing class, which is a measure of their passing ability in creating scoring opportunities.")
async def get_team_short_names_chance_creation_passing_class(chance_creation_passing_class: str = Query(..., description="Chance creation passing class of the team")):
    cursor.execute("SELECT DISTINCT t1.team_short_name FROM Team AS t1 INNER JOIN Team_attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t2.chanceCreationPassingClass = ?", (chance_creation_passing_class,))
    result = cursor.fetchall()
    if not result:
        return {"team_short_names": []}
    return {"team_short_names": [row[0] for row in result]}

# Endpoint to get the defensive work rate of a specific player
@app.get("/v1/european_football_2/defensive_work_rate_player", operation_id="get_defensive_work_rate_player", summary="Retrieves the unique defensive work rate value of a specific football player. The player is identified by their name, which is provided as an input parameter.")
async def get_defensive_work_rate_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT DISTINCT t2.defensive_work_rate FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"defensive_work_rate": []}
    return {"defensive_work_rate": [row[0] for row in result]}

# Endpoint to get the birthday of the player with the highest overall rating
@app.get("/v1/european_football_2/birthday_highest_overall_rating", operation_id="get_birthday_highest_overall_rating", summary="Retrieves the birthdate of the football player with the highest overall rating in the European league. The overall rating is determined by a combination of various performance metrics, and the player with the highest cumulative score is selected.")
async def get_birthday_highest_overall_rating():
    cursor.execute("SELECT t1.birthday FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id ORDER BY t2.overall_rating DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"birthday": []}
    return {"birthday": result[0]}

# Endpoint to get the names of leagues in a specific country
@app.get("/v1/european_football_2/league_names_by_country", operation_id="get_league_names_by_country", summary="Retrieves the names of football leagues associated with a specified country in the European football system. The operation requires the name of the country as an input parameter.")
async def get_league_names_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT t2.name FROM Country AS t1 INNER JOIN League AS t2 ON t1.id = t2.country_id WHERE t1.name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"league_names": []}
    return {"league_names": [row[0] for row in result]}

# Endpoint to get the average home team goals per match for a specific country and season
@app.get("/v1/european_football_2/average_home_team_goals_country_season", operation_id="get_average_home_team_goals_country_season", summary="Retrieves the average number of goals scored by home teams per match for a given country and season in European football. The calculation is based on the total sum of goals scored by home teams divided by the total number of matches played in the specified country and season.")
async def get_average_home_team_goals_country_season(country_name: str = Query(..., description="Name of the country"), season: str = Query(..., description="Season in 'YYYY/YYYY' format")):
    cursor.execute("SELECT CAST(SUM(t2.home_team_goal) AS REAL) / COUNT(t2.id) FROM Country AS t1 INNER JOIN Match AS t2 ON t1.id = t2.country_id WHERE t1.name = ? AND t2.season = ?", (country_name, season))
    result = cursor.fetchone()
    if not result:
        return {"average_goals": []}
    return {"average_goals": result[0]}

# Endpoint to get the average finishing of the tallest or shortest players
@app.get("/v1/european_football_2/average_finishing_extreme_heights", operation_id="get_average_finishing_extreme_heights", summary="Retrieves the average finishing score of the tallest or shortest players in the database. The operation calculates the average finishing score for the tallest and shortest players separately and returns the higher of the two averages.")
async def get_average_finishing_extreme_heights():
    cursor.execute("SELECT A FROM ( SELECT AVG(finishing) result, 'Max' A FROM Player AS T1 INNER JOIN Player_Attributes AS T2 ON T1.player_api_id = T2.player_api_id WHERE T1.height = ( SELECT MAX(height) FROM Player ) UNION SELECT AVG(finishing) result, 'Min' A FROM Player AS T1 INNER JOIN Player_Attributes AS T2 ON T1.player_api_id = T2.player_api_id WHERE T1.height = ( SELECT MIN(height) FROM Player ) ) ORDER BY result DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"average_finishing": []}
    return {"average_finishing": result[0]}

# Endpoint to get the names of players taller than a specific height
@app.get("/v1/european_football_2/players_taller_than", operation_id="get_players_taller_than", summary="Retrieves the names of European football players who are taller than the specified height in centimeters. This operation allows you to filter players based on their height, providing a list of those who exceed the given threshold.")
async def get_players_taller_than(height: int = Query(..., description="Height of the player in cm")):
    cursor.execute("SELECT player_name FROM Player WHERE height > ?", (height,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the count of players born after a specific year
@app.get("/v1/european_football_2/player_count_by_birth_year", operation_id="get_player_count_by_birth_year", summary="Retrieves the total number of players who were born after the specified year. The input parameter determines the year to be used as a reference point for the count.")
async def get_player_count_by_birth_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(id) FROM Player WHERE STRFTIME('%Y', birthday) > ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players with weight greater than a specific value and name starting with a specific string
@app.get("/v1/european_football_2/player_count_by_weight_and_name", operation_id="get_player_count_by_weight_and_name", summary="Retrieves the number of players who weigh more than the specified weight and have names starting with the provided string. This operation is useful for filtering players based on their weight and name.")
async def get_player_count_by_weight_and_name(weight: int = Query(..., description="Weight of the player"), name_start: str = Query(..., description="Starting string of the player name")):
    cursor.execute("SELECT COUNT(id) FROM Player WHERE weight > ? AND player_name LIKE ?", (weight, name_start + '%'))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct player names with overall rating greater than a specific value within a date range
@app.get("/v1/european_football_2/distinct_player_names_by_rating_and_date", operation_id="get_distinct_player_names_by_rating_and_date", summary="Retrieve a list of unique player names with an overall rating surpassing a specified value, within a defined date range. The operation filters players based on their overall rating and the year of their rating, returning only those who meet the criteria.")
async def get_distinct_player_names_by_rating_and_date(overall_rating: int = Query(..., description="Overall rating of the player"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.overall_rating > ? AND SUBSTR(t2.`date`, 1, 4) BETWEEN ? AND ?", (overall_rating, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the potential of a player by name
@app.get("/v1/european_football_2/player_potential_by_name", operation_id="get_player_potential_by_name", summary="Get the potential of a player by name")
async def get_player_potential_by_name(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT t2.potential FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"potential": []}
    return {"potential": result[0]}

# Endpoint to get distinct player IDs and names based on preferred foot
@app.get("/v1/european_football_2/distinct_players_by_preferred_foot", operation_id="get_distinct_players_by_preferred_foot", summary="Retrieves a list of unique players, identified by their IDs and names, who primarily use the specified foot for play. This operation filters players based on their preferred foot, ensuring that only those who match the provided preference are included in the results.")
async def get_distinct_players_by_preferred_foot(preferred_foot: str = Query(..., description="Preferred foot of the player")):
    cursor.execute("SELECT DISTINCT t1.id, t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.preferred_foot = ?", (preferred_foot,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [{"id": row[0], "player_name": row[1]} for row in result]}

# Endpoint to get distinct team long names based on build-up play speed class
@app.get("/v1/european_football_2/distinct_team_long_names_by_build_up_play_speed", operation_id="get_distinct_team_long_names_by_build_up_play_speed", summary="Retrieves a list of unique team names that match the specified build-up play speed class. The build-up play speed class is a measure of a team's speed in transitioning from defense to offense.")
async def get_distinct_team_long_names_by_build_up_play_speed(build_up_play_speed_class: str = Query(..., description="Build-up play speed class of the team")):
    cursor.execute("SELECT DISTINCT t1.team_long_name FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t2.buildUpPlaySpeedClass = ?", (build_up_play_speed_class,))
    result = cursor.fetchall()
    if not result:
        return {"team_long_names": []}
    return {"team_long_names": [row[0] for row in result]}

# Endpoint to get distinct build-up play passing classes based on team short name
@app.get("/v1/european_football_2/distinct_build_up_play_passing_classes_by_team_short_name", operation_id="get_distinct_build_up_play_passing_classes_by_team_short_name", summary="Retrieve the unique build-up play passing classes associated with a specific team, identified by its short name.")
async def get_distinct_build_up_play_passing_classes_by_team_short_name(team_short_name: str = Query(..., description="Short name of the team")):
    cursor.execute("SELECT DISTINCT t2.buildUpPlayPassingClass FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_short_name = ?", (team_short_name,))
    result = cursor.fetchall()
    if not result:
        return {"build_up_play_passing_classes": []}
    return {"build_up_play_passing_classes": [row[0] for row in result]}

# Endpoint to get distinct team short names based on build-up play passing rating
@app.get("/v1/european_football_2/distinct_team_short_names_by_build_up_play_passing", operation_id="get_distinct_team_short_names_by_build_up_play_passing", summary="Retrieve a list of unique team short names that have a build-up play passing rating greater than the provided value. This operation filters teams based on their build-up play passing rating and returns their respective short names.")
async def get_distinct_team_short_names_by_build_up_play_passing(build_up_play_passing: int = Query(..., description="Build-up play passing rating of the team")):
    cursor.execute("SELECT DISTINCT t1.team_short_name FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t2.buildUpPlayPassing > ?", (build_up_play_passing,))
    result = cursor.fetchall()
    if not result:
        return {"team_short_names": []}
    return {"team_short_names": [row[0] for row in result]}

# Endpoint to get the average overall rating of players with height greater than a specific value within a date range
@app.get("/v1/european_football_2/average_overall_rating_by_height_and_date", operation_id="get_average_overall_rating_by_height_and_date", summary="Retrieve the average overall rating of players who are taller than a specified height and played within a given date range. The calculation considers the sum of all overall ratings divided by the total number of players that meet the height and date criteria.")
async def get_average_overall_rating_by_height_and_date(height: int = Query(..., description="Height of the player"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(t2.overall_rating) AS REAL) / COUNT(t2.id) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.height > ? AND STRFTIME('%Y',t2.`date`) >= ? AND STRFTIME('%Y',t2.`date`) <= ?", (height, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"average_overall_rating": []}
    return {"average_overall_rating": result[0]}

# Endpoint to get the shortest player
@app.get("/v1/european_football_2/shortest_player", operation_id="get_shortest_player", summary="Retrieves the name of the shortest player in the database. The operation sorts all players by height in ascending order and returns the name of the first player in the sorted list.")
async def get_shortest_player():
    cursor.execute("SELECT player_name FROM player ORDER BY height ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the country name based on the league name
@app.get("/v1/european_football_2/country_name_by_league", operation_id="get_country_name_by_league", summary="Retrieves the name of the country associated with the specified league. The operation uses the provided league name to look up the corresponding country in the database.")
async def get_country_name_by_league(league_name: str = Query(..., description="Name of the league")):
    cursor.execute("SELECT t1.name FROM Country AS t1 INNER JOIN League AS t2 ON t1.id = t2.country_id WHERE t2.name = ?", (league_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get distinct team short names based on build-up play attributes
@app.get("/v1/european_football_2/team_short_names_by_buildup_play", operation_id="get_team_short_names_by_buildup_play", summary="Retrieve a unique list of team short names that match the specified build-up play attributes, including speed, dribbling, and passing. This operation filters teams based on their build-up play characteristics and returns the corresponding short names.")
async def get_team_short_names_by_buildup_play(build_up_play_speed: int = Query(..., description="Build-up play speed"), build_up_play_dribbling: int = Query(..., description="Build-up play dribbling"), build_up_play_passing: int = Query(..., description="Build-up play passing")):
    cursor.execute("SELECT DISTINCT t1.team_short_name FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t2.buildUpPlaySpeed = ? AND t2.buildUpPlayDribbling = ? AND t2.buildUpPlayPassing = ?", (build_up_play_speed, build_up_play_dribbling, build_up_play_passing))
    result = cursor.fetchall()
    if not result:
        return {"team_short_names": []}
    return {"team_short_names": [row[0] for row in result]}

# Endpoint to get the average overall rating of a player
@app.get("/v1/european_football_2/average_overall_rating_by_player", operation_id="get_average_overall_rating_by_player", summary="Retrieves the average overall rating of a specific football player. The operation calculates the average rating by summing up all the overall ratings of the player and dividing it by the total number of ratings available for the player. The player is identified by their name.")
async def get_average_overall_rating_by_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT CAST(SUM(t2.overall_rating) AS REAL) / COUNT(t2.id) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_overall_rating": []}
    return {"average_overall_rating": result[0]}

# Endpoint to get the count of matches in a specific league within a date range
@app.get("/v1/european_football_2/match_count_by_league_and_date_range", operation_id="get_match_count_by_league_and_date_range", summary="Retrieves the total number of matches played in a specific football league within a given date range. The league is identified by its name, and the date range is specified using the start and end dates in 'YYYY-MM' format.")
async def get_match_count_by_league_and_date_range(league_name: str = Query(..., description="Name of the league"), start_date: str = Query(..., description="Start date in 'YYYY-MM' format"), end_date: str = Query(..., description="End date in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(t2.id) FROM League AS t1 INNER JOIN Match AS t2 ON t1.id = t2.league_id WHERE t1.name = ? AND SUBSTR(t2.date, 1, 7) BETWEEN ? AND ?", (league_name, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"match_count": []}
    return {"match_count": result[0]}

# Endpoint to get team short names based on home team goals
@app.get("/v1/european_football_2/team_short_names_by_home_team_goals", operation_id="get_team_short_names_by_home_team_goals", summary="Retrieves the short names of teams that scored a specific number of goals at home. The number of home team goals is provided as an input parameter.")
async def get_team_short_names_by_home_team_goals(home_team_goal: int = Query(..., description="Number of home team goals")):
    cursor.execute("SELECT t1.team_short_name FROM Team AS t1 INNER JOIN Match AS t2 ON t1.team_api_id = t2.home_team_api_id WHERE t2.home_team_goal = ?", (home_team_goal,))
    result = cursor.fetchall()
    if not result:
        return {"team_short_names": []}
    return {"team_short_names": [row[0] for row in result]}

# Endpoint to get the player name with the highest balance for a given potential
@app.get("/v1/european_football_2/player_name_by_potential_and_balance", operation_id="get_player_name_by_potential_and_balance", summary="Retrieves the name of the player with the highest balance for a specified potential. The potential parameter is used to filter the results.")
async def get_player_name_by_potential_and_balance(potential: int = Query(..., description="Potential of the player")):
    cursor.execute("SELECT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.potential = ? ORDER BY t2.balance DESC LIMIT 1", (potential,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the difference in average ball control between two players
@app.get("/v1/european_football_2/ball_control_difference_between_players", operation_id="get_ball_control_difference_between_players", summary="Retrieve the difference in average ball control ratings between two specified players. The operation calculates the average ball control rating for each player by summing their individual ball control ratings and dividing by the total number of ratings. The difference between these averages is then returned.")
async def get_ball_control_difference_between_players(player_name_1: str = Query(..., description="Name of the first player"), player_name_2: str = Query(..., description="Name of the second player")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN t1.player_name = ? THEN t2.ball_control ELSE 0 END) AS REAL) / COUNT(CASE WHEN t1.player_name = ? THEN t2.id ELSE NULL END) - CAST(SUM(CASE WHEN t1.player_name = ? THEN t2.ball_control ELSE 0 END) AS REAL) / COUNT(CASE WHEN t1.player_name = ? THEN t2.id ELSE NULL END) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id", (player_name_1, player_name_1, player_name_2, player_name_2))
    result = cursor.fetchone()
    if not result:
        return {"ball_control_difference": []}
    return {"ball_control_difference": result[0]}

# Endpoint to get the long name of a team based on the short name
@app.get("/v1/european_football_2/team_long_name_by_short_name", operation_id="get_team_long_name_by_short_name", summary="Get the long name of a team based on the short name")
async def get_team_long_name_by_short_name(team_short_name: str = Query(..., description="Short name of the team")):
    cursor.execute("SELECT team_long_name FROM Team WHERE team_short_name = ?", (team_short_name,))
    result = cursor.fetchone()
    if not result:
        return {"team_long_name": []}
    return {"team_long_name": result[0]}

# Endpoint to get the youngest player between two given players
@app.get("/v1/european_football_2/youngest_player_between_two", operation_id="get_youngest_player_between_two", summary="Retrieves the name of the youngest player among the two provided players. The operation compares the birthdays of the specified players and returns the name of the one with the most recent birthday.")
async def get_youngest_player_between_two(player_name_1: str = Query(..., description="Name of the first player"), player_name_2: str = Query(..., description="Name of the second player")):
    cursor.execute("SELECT player_name FROM Player WHERE player_name IN (?, ?) ORDER BY birthday ASC LIMIT 1", (player_name_1, player_name_2))
    result = cursor.fetchone()
    if not result:
        return {"youngest_player": []}
    return {"youngest_player": result[0]}

# Endpoint to get the count of players based on preferred foot and attacking work rate
@app.get("/v1/european_football_2/player_count_by_preferred_foot_and_attacking_work_rate", operation_id="get_player_count_by_preferred_foot_and_attacking_work_rate", summary="Retrieve the number of players who share a specific preferred foot and attacking work rate. The preferred foot can be either left or right, and the attacking work rate ranges from low to high.")
async def get_player_count_by_preferred_foot_and_attacking_work_rate(preferred_foot: str = Query(..., description="Preferred foot of the player"), attacking_work_rate: str = Query(..., description="Attacking work rate of the player")):
    cursor.execute("SELECT COUNT(player_api_id) FROM Player_Attributes WHERE preferred_foot = ? AND attacking_work_rate = ?", (preferred_foot, attacking_work_rate))
    result = cursor.fetchone()
    if not result:
        return {"player_count": []}
    return {"player_count": result[0]}

# Endpoint to get the count of distinct players born before a certain year with a specific defensive work rate
@app.get("/v1/european_football_2/count_players_by_birth_year_and_defensive_work_rate", operation_id="get_count_players_by_birth_year_and_defensive_work_rate", summary="Retrieve the number of unique football players who were born before a specified year and have a particular defensive work rate. The birth year is provided in 'YYYY' format, and the defensive work rate is a specific attribute of the player.")
async def get_count_players_by_birth_year_and_defensive_work_rate(birth_year: str = Query(..., description="Year of birth in 'YYYY' format"), defensive_work_rate: str = Query(..., description="Defensive work rate")):
    cursor.execute("SELECT COUNT(DISTINCT t1.player_name) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE STRFTIME('%Y',t1.birthday) < ? AND t2.defensive_work_rate = ?", (birth_year, defensive_work_rate))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the player with the highest crossing skill among a list of players
@app.get("/v1/european_football_2/top_player_by_crossing_skill", operation_id="get_top_player_by_crossing_skill", summary="Retrieves the player with the highest crossing skill from a specified list of players. The list of players is provided as input parameters, and the player with the highest crossing skill is determined based on the corresponding attribute values. The operation returns the name of the top player and their crossing skill value.")
async def get_top_player_by_crossing_skill(player_name1: str = Query(..., description="Name of the first player"), player_name2: str = Query(..., description="Name of the second player"), player_name3: str = Query(..., description="Name of the third player")):
    cursor.execute("SELECT t1.player_name, t2.crossing FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name IN (?, ?, ?) ORDER BY t2.crossing DESC LIMIT 1", (player_name1, player_name2, player_name3))
    result = cursor.fetchone()
    if not result:
        return {"player_name": [], "crossing": []}
    return {"player_name": result[0], "crossing": result[1]}

# Endpoint to get the heading accuracy of a specific player
@app.get("/v1/european_football_2/heading_accuracy_by_player_name", operation_id="get_heading_accuracy_by_player_name", summary="Get the heading accuracy of a specific player")
async def get_heading_accuracy_by_player_name(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT t2.heading_accuracy FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"heading_accuracy": []}
    return {"heading_accuracy": result[0]}

# Endpoint to get the count of distinct players based on height and volleys
@app.get("/v1/european_football_2/count_players_by_height_and_volleys", operation_id="get_count_players_by_height_and_volleys", summary="Retrieves the count of unique football players who are taller than the specified height and have a volleys rating greater than the provided value. This operation is useful for understanding the distribution of players based on their height and volleys rating.")
async def get_count_players_by_height_and_volleys(height: int = Query(..., description="Height of the player in cm"), volleys: int = Query(..., description="Volleys rating of the player")):
    cursor.execute("SELECT COUNT(DISTINCT t1.id) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.height > ? AND t2.volleys > ?", (height, volleys))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct player names based on volleys and dribbling ratings
@app.get("/v1/european_football_2/player_names_by_volleys_and_dribbling", operation_id="get_player_names_by_volleys_and_dribbling", summary="Retrieves a list of unique player names who have volleys and dribbling ratings surpassing the provided thresholds. This operation filters players based on their volleys and dribbling skills, returning only those who meet the specified criteria.")
async def get_player_names_by_volleys_and_dribbling(volleys: int = Query(..., description="Volleys rating of the player"), dribbling: int = Query(..., description="Dribbling rating of the player")):
    cursor.execute("SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.volleys > ? AND t2.dribbling > ?", (volleys, dribbling))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the count of matches for a specific country and season
@app.get("/v1/european_football_2/count_matches_by_country_and_season", operation_id="get_count_matches_by_country_and_season", summary="Retrieves the total number of matches played in a specific country during a given season. The operation requires the name of the country and the season in 'YYYY/YYYY' format as input parameters.")
async def get_count_matches_by_country_and_season(country_name: str = Query(..., description="Name of the country"), season: str = Query(..., description="Season in 'YYYY/YYYY' format")):
    cursor.execute("SELECT COUNT(t2.id) FROM Country AS t1 INNER JOIN Match AS t2 ON t1.id = t2.country_id WHERE t1.name = ? AND t2.season = ?", (country_name, season))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the long passing skill of the youngest player
@app.get("/v1/european_football_2/long_passing_youngest_player", operation_id="get_long_passing_youngest_player", summary="Retrieves the long passing skill of the youngest player in the database. The operation sorts players by their birthdays in ascending order and returns the long passing attribute of the first player, who is the youngest.")
async def get_long_passing_youngest_player():
    cursor.execute("SELECT t2.long_passing FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id ORDER BY t1.birthday ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"long_passing": []}
    return {"long_passing": result[0]}

# Endpoint to get the count of matches in a specific league and month
@app.get("/v1/european_football_2/match_count_by_league_and_month", operation_id="get_match_count_by_league_and_month", summary="Retrieves the total number of matches played in a specific European football league during a given month. The league is identified by its name, and the month is specified in the 'YYYY-MM' format. This operation provides valuable insights into the match frequency within a league for a particular month.")
async def get_match_count_by_league_and_month(league_name: str = Query(..., description="Name of the league"), month_year: str = Query(..., description="Month and year in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(t2.id) FROM League AS t1 INNER JOIN Match AS t2 ON t1.id = t2.league_id WHERE t1.name = ? AND SUBSTR(t2.`date`, 1, 7) = ?", (league_name, month_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the league with the maximum number of matches in a specific season
@app.get("/v1/european_football_2/league_with_max_matches_in_season", operation_id="get_league_with_max_matches_in_season", summary="Retrieves the name of the league that has the highest number of matches in a given season. The season is specified in the 'YYYY/YYYY' format.")
async def get_league_with_max_matches_in_season(season: str = Query(..., description="Season in 'YYYY/YYYY' format")):
    cursor.execute("SELECT t1.name FROM League AS t1 JOIN Match AS t2 ON t1.id = t2.league_id WHERE t2.season = ? GROUP BY t1.name HAVING COUNT(t2.id) = (SELECT MAX(match_count) FROM (SELECT COUNT(t2.id) AS match_count FROM Match AS t2 WHERE t2.season = ? GROUP BY t2.league_id))", (season, season))
    result = cursor.fetchall()
    if not result:
        return {"leagues": []}
    return {"leagues": [row[0] for row in result]}

# Endpoint to get the average overall rating of players born before a specific year
@app.get("/v1/european_football_2/average_overall_rating_by_birth_year", operation_id="get_average_overall_rating_by_birth_year", summary="Get the average overall rating of players born before a specific year")
async def get_average_overall_rating_by_birth_year(birth_year: int = Query(..., description="Year of birth in 'YYYY' format")):
    cursor.execute("SELECT SUM(t2.overall_rating) / COUNT(t1.id) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE SUBSTR(t1.birthday, 1, 4) < ?", (birth_year,))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the percentage difference in overall ratings between two players
@app.get("/v1/european_football_2/percentage_difference_in_overall_ratings", operation_id="get_percentage_difference_in_overall_ratings", summary="Retrieves the percentage difference in overall ratings between two specified players. The calculation is based on the sum of the overall ratings of each player, comparing the first player to the second. The result is expressed as a percentage of the first player's total overall rating.")
async def get_percentage_difference_in_overall_ratings(player_name_1: str = Query(..., description="Name of the first player"), player_name_2: str = Query(..., description="Name of the second player")):
    cursor.execute("SELECT (SUM(CASE WHEN t1.player_name = ? THEN t2.overall_rating ELSE 0 END) * 1.0 - SUM(CASE WHEN t1.player_name = ? THEN t2.overall_rating ELSE 0 END)) * 100 / SUM(CASE WHEN t1.player_name = ? THEN t2.overall_rating ELSE 0 END) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id", (player_name_1, player_name_2, player_name_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the average build-up play speed of a specific team
@app.get("/v1/european_football_2/average_build_up_play_speed_by_team", operation_id="get_average_build_up_play_speed_by_team", summary="Retrieves the average build-up play speed for a specific football team. The team is identified by its long name, and the result is calculated by summing the build-up play speeds of all matches and dividing by the total number of matches played by the team.")
async def get_average_build_up_play_speed_by_team(team_long_name: str = Query(..., description="Long name of the team")):
    cursor.execute("SELECT CAST(SUM(t2.buildUpPlaySpeed) AS REAL) / COUNT(t2.id) FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_long_name = ?", (team_long_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_speed": []}
    return {"average_speed": result[0]}

# Endpoint to get the total crossing attribute of a specific player
@app.get("/v1/european_football_2/total_crossing_by_player", operation_id="get_total_crossing_by_player", summary="Retrieves the cumulative crossing attribute of a specific football player. The operation calculates the total crossing attribute by aggregating the individual crossing attribute values associated with the player's name.")
async def get_total_crossing_by_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT SUM(t2.crossing) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_crossing": []}
    return {"total_crossing": result[0]}

# Endpoint to get the chance creation passing attributes of a specific team
@app.get("/v1/european_football_2/chance_creation_passing_by_team", operation_id="get_chance_creation_passing_by_team", summary="Retrieves the chance creation passing attributes of a specific team, sorted in descending order. The team is identified by its long name, and the attributes include the chance creation passing value and its classification.")
async def get_chance_creation_passing_by_team(team_long_name: str = Query(..., description="Long name of the team")):
    cursor.execute("SELECT t2.chanceCreationPassing, t2.chanceCreationPassingClass FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_long_name = ? ORDER BY t2.chanceCreationPassing DESC LIMIT 1", (team_long_name,))
    result = cursor.fetchone()
    if not result:
        return {"chance_creation_passing": [], "chance_creation_passing_class": []}
    return {"chance_creation_passing": result[0], "chance_creation_passing_class": result[1]}

# Endpoint to get the distinct preferred foot of a specific player
@app.get("/v1/european_football_2/preferred_foot_by_player", operation_id="get_preferred_foot_by_player", summary="Retrieve the unique preferred foot of a specified player from the football database.")
async def get_preferred_foot_by_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT DISTINCT t2.preferred_foot FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"preferred_foot": []}
    return {"preferred_foot": [row[0] for row in result]}

# Endpoint to get the maximum overall rating of a specific player
@app.get("/v1/european_football_2/max_overall_rating_by_player", operation_id="get_max_overall_rating_by_player", summary="Retrieves the highest overall rating ever achieved by the specified player. The player's name is required as an input parameter.")
async def get_max_overall_rating_by_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT MAX(t2.overall_rating) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"max_overall_rating": []}
    return {"max_overall_rating": result[0]}

# Endpoint to get the average away team goals for a specific team in a specific country
@app.get("/v1/european_football_2/average_away_team_goals", operation_id="get_average_away_team_goals", summary="Retrieves the average number of goals scored by a specific away team in a specific country. The operation calculates this average by summing up the total goals scored by the away team in all matches and dividing it by the total number of matches played by the team in the specified country.")
async def get_average_away_team_goals(team_long_name: str = Query(..., description="Long name of the team"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT CAST(SUM(T1.away_team_goal) AS REAL) / COUNT(T1.id) FROM \"Match\" AS T1 INNER JOIN TEAM AS T2 ON T1.away_team_api_id = T2.team_api_id INNER JOIN Country AS T3 ON T1.country_id = T3.id WHERE T2.team_long_name = ? AND T3.name = ?", (team_long_name, country_name))
    result = cursor.fetchone()
    if not result:
        return {"average_goals": []}
    return {"average_goals": result[0]}

# Endpoint to get the player name with a specific overall rating on a specific date
@app.get("/v1/european_football_2/player_name_by_rating_date", operation_id="get_player_name_by_rating_date", summary="Retrieves the name of the oldest player with a specified overall rating on a given date. The date should be provided in 'YYYY-MM-DD' format, and the overall rating is a numerical value representing the player's skill level.")
async def get_player_name_by_rating_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), overall_rating: int = Query(..., description="Overall rating of the player")):
    cursor.execute("SELECT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE SUBSTR(t2.`date`, 1, 10) = ? AND t2.overall_rating = ? ORDER BY t1.birthday ASC LIMIT 1", (date, overall_rating))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the overall rating of a specific player on a specific date
@app.get("/v1/european_football_2/player_overall_rating", operation_id="get_player_overall_rating", summary="Retrieves the overall rating of a specific football player on a given date. The operation requires the player's name and the date in 'YYYY-MM-DD' format to accurately locate the desired rating.")
async def get_player_overall_rating(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT t2.overall_rating FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE SUBSTR(t2.`date`, 1, 10) = ? AND t1.player_name = ?", (date, player_name))
    result = cursor.fetchone()
    if not result:
        return {"overall_rating": []}
    return {"overall_rating": result[0]}

# Endpoint to get the potential of a specific player on a specific date
@app.get("/v1/european_football_2/player_potential", operation_id="get_player_potential", summary="Retrieve the potential of a specific football player on a given date. This operation requires the player's name and the date in 'YYYY-MM-DD' format. The potential value is determined by querying the player's attributes on the specified date.")
async def get_player_potential(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT t2.potential FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE SUBSTR(t2.`date`, 1, 10) = ? AND t1.player_name = ?", (date, player_name))
    result = cursor.fetchone()
    if not result:
        return {"potential": []}
    return {"potential": result[0]}

# Endpoint to get the attacking work rate of a specific player on a specific date
@app.get("/v1/european_football_2/player_attacking_work_rate", operation_id="get_player_attacking_work_rate", summary="Retrieves the attacking work rate of a specific football player on a given date. The operation requires the player's name and the date in 'YYYY-MM-DD%' format. The result is a single value representing the player's attacking work rate on the specified date.")
async def get_player_attacking_work_rate(date: str = Query(..., description="Date in 'YYYY-MM-DD%' format"), player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT t2.attacking_work_rate FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.`date` LIKE ? AND t1.player_name = ?", (date, player_name))
    result = cursor.fetchone()
    if not result:
        return {"attacking_work_rate": []}
    return {"attacking_work_rate": result[0]}

# Endpoint to get the defensive work rate of a specific player on a specific date
@app.get("/v1/european_football_2/player_defensive_work_rate", operation_id="get_player_defensive_work_rate", summary="Retrieves the defensive work rate of a specific football player on a given date. The operation requires the player's name and the date in 'YYYY-MM-DD' format to accurately locate the desired data.")
async def get_player_defensive_work_rate(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT t2.defensive_work_rate FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_fifa_api_id = t2.player_fifa_api_id WHERE SUBSTR(t2.`date`, 1, 10) = ? AND t1.player_name = ?", (date, player_name))
    result = cursor.fetchone()
    if not result:
        return {"defensive_work_rate": []}
    return {"defensive_work_rate": result[0]}

# Endpoint to get the date of the highest crossing value for a specific player
@app.get("/v1/european_football_2/player_highest_crossing_date", operation_id="get_player_highest_crossing_date", summary="Retrieves the most recent date on which a specific player achieved their highest crossing value. The operation filters player data based on the provided player name and identifies the date associated with the highest crossing value. The result is the most recent date among those with the highest crossing value.")
async def get_player_highest_crossing_date(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT `date` FROM ( SELECT t2.crossing, t2.`date` FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_fifa_api_id = t2.player_fifa_api_id WHERE t1.player_name = ? ORDER BY t2.crossing DESC) ORDER BY date DESC LIMIT 1", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the build-up play speed class of a specific team on a specific date
@app.get("/v1/european_football_2/team_build_up_play_speed_class", operation_id="get_team_build_up_play_speed_class", summary="Retrieve the build-up play speed class of a specific football team on a given date. The team is identified by its long name, and the date is provided in 'YYYY-MM-DD' format. This operation returns the speed class of the team's build-up play, which is a measure of their performance in setting up offensive plays.")
async def get_team_build_up_play_speed_class(team_long_name: str = Query(..., description="Long name of the team"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT t2.buildUpPlaySpeedClass FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_long_name = ? AND SUBSTR(t2.`date`, 1, 10) = ?", (team_long_name, date))
    result = cursor.fetchone()
    if not result:
        return {"build_up_play_speed_class": []}
    return {"build_up_play_speed_class": result[0]}

# Endpoint to get the build-up play dribbling class of a specific team on a specific date
@app.get("/v1/european_football_2/team_build_up_play_dribbling_class", operation_id="get_team_build_up_play_dribbling_class", summary="Retrieve the build-up play dribbling class of a specific football team on a given date. The team is identified by its short name, and the date is provided in the 'YYYY-MM-DD' format. This operation returns the build-up play dribbling class attribute of the team on the specified date.")
async def get_team_build_up_play_dribbling_class(team_short_name: str = Query(..., description="Short name of the team"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT t2.buildUpPlayDribblingClass FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_short_name = ? AND SUBSTR(t2.`date`, 1, 10) = ?", (team_short_name, date))
    result = cursor.fetchone()
    if not result:
        return {"build_up_play_dribbling_class": []}
    return {"build_up_play_dribbling_class": result[0]}

# Endpoint to get the build-up play passing class of a specific team on a specific date
@app.get("/v1/european_football_2/team_build_up_play_passing_class", operation_id="get_team_build_up_play_passing_class", summary="Retrieves the build-up play passing class of a specific football team on a given date. The team is identified by its long name, and the date is provided in the 'YYYY-MM-DD%' format. This operation returns the build-up play passing class attribute of the team on the specified date.")
async def get_team_build_up_play_passing_class(team_long_name: str = Query(..., description="Long name of the team"), date: str = Query(..., description="Date in 'YYYY-MM-DD%' format")):
    cursor.execute("SELECT t2.buildUpPlayPassingClass FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_long_name = ? AND t2.`date` LIKE ?", (team_long_name, date))
    result = cursor.fetchone()
    if not result:
        return {"build_up_play_passing_class": []}
    return {"build_up_play_passing_class": result[0]}

# Endpoint to get the chance creation passing class for a team on a specific date
@app.get("/v1/european_football_2/chance_creation_passing_class", operation_id="get_chance_creation_passing_class", summary="Retrieves the chance creation passing class for a specific team on a given date. The team is identified by its long name, and the date is provided in 'YYYY-MM-DD' format. This operation returns a single value representing the team's chance creation passing class on the specified date.")
async def get_chance_creation_passing_class(team_long_name: str = Query(..., description="Long name of the team"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT t2.chanceCreationPassingClass FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_long_name = ? AND SUBSTR(t2.`date`, 1, 10) = ?", (team_long_name, date))
    result = cursor.fetchone()
    if not result:
        return {"chanceCreationPassingClass": []}
    return {"chanceCreationPassingClass": result[0]}

# Endpoint to get the chance creation crossing class for a team on a specific date
@app.get("/v1/european_football_2/chance_creation_crossing_class", operation_id="get_chance_creation_crossing_class", summary="Retrieves the chance creation crossing class for a specific football team on a given date. The team is identified by its long name, and the date is provided in 'YYYY-MM-DD' format. This endpoint returns the chance creation crossing class value, which is a measure of the team's crossing ability in creating scoring opportunities.")
async def get_chance_creation_crossing_class(team_long_name: str = Query(..., description="Long name of the team"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT t2.chanceCreationCrossingClass FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_long_name = ? AND SUBSTR(t2.`date`, 1, 10) = ?", (team_long_name, date))
    result = cursor.fetchone()
    if not result:
        return {"chanceCreationCrossingClass": []}
    return {"chanceCreationCrossingClass": result[0]}

# Endpoint to get the chance creation shooting class for a team on a specific date
@app.get("/v1/european_football_2/chance_creation_shooting_class", operation_id="get_chance_creation_shooting_class", summary="Get the chance creation shooting class for a team on a specific date")
async def get_chance_creation_shooting_class(team_long_name: str = Query(..., description="Long name of the team"), date: str = Query(..., description="Date in 'YYYY-MM-DD%' format")):
    cursor.execute("SELECT t2.chanceCreationShootingClass FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_long_name = ? AND t2.`date` LIKE ?", (team_long_name, date))
    result = cursor.fetchone()
    if not result:
        return {"chanceCreationShootingClass": []}
    return {"chanceCreationShootingClass": result[0]}

# Endpoint to get the average overall rating of a player within a date range
@app.get("/v1/european_football_2/average_overall_rating", operation_id="get_average_overall_rating", summary="Retrieve the average overall rating of a specific football player within a given date range. The operation calculates the average rating based on the provided player's name and the specified start and end dates.")
async def get_average_overall_rating(player_name: str = Query(..., description="Name of the player"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(t2.overall_rating) AS REAL) / COUNT(t2.id) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_fifa_api_id = t2.player_fifa_api_id WHERE t1.player_name = ? AND SUBSTR(t2.`date`, 1, 10) BETWEEN ? AND ?", (player_name, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_overall_rating": []}
    return {"average_overall_rating": result[0]}

# Endpoint to get the percentage difference in overall ratings between two players on a specific date
@app.get("/v1/european_football_2/overall_rating_percentage_difference", operation_id="get_overall_rating_percentage_difference", summary="Retrieve the percentage difference in overall ratings between two specified players on a given date. The calculation is based on the sum of their individual overall ratings. The input parameters include the names of the two players and the date in 'YYYY-MM-DD' format.")
async def get_overall_rating_percentage_difference(player_name_1: str = Query(..., description="Name of the first player"), player_name_2: str = Query(..., description="Name of the second player"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT (SUM(CASE WHEN t1.player_name = ? THEN t2.overall_rating ELSE 0 END) * 1.0 - SUM(CASE WHEN t1.player_name = ? THEN t2.overall_rating ELSE 0 END)) * 100 / SUM(CASE WHEN t1.player_name = ? THEN t2.overall_rating ELSE 0 END) LvsJ_percent FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_fifa_api_id = t2.player_fifa_api_id WHERE SUBSTR(t2.`date`, 1, 10) = ?", (player_name_1, player_name_2, player_name_1, date))
    result = cursor.fetchone()
    if not result:
        return {"LvsJ_percent": []}
    return {"LvsJ_percent": result[0]}

# Endpoint to get the top N heaviest players
@app.get("/v1/european_football_2/heaviest_players", operation_id="get_heaviest_players", summary="Retrieves a list of the top N heaviest players, sorted in descending order by weight. The number of players returned is determined by the provided limit parameter.")
async def get_heaviest_players(limit: int = Query(..., description="Number of top heaviest players to retrieve")):
    cursor.execute("SELECT player_api_id FROM Player ORDER BY weight DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"player_api_id": []}
    return {"player_api_id": [row[0] for row in result]}

# Endpoint to get players older than a specified age
@app.get("/v1/european_football_2/players_older_than", operation_id="get_players_older_than", summary="Retrieves a list of players who are older than the specified minimum age. The age is calculated based on the current date and the player's birthday.")
async def get_players_older_than(age: int = Query(..., description="Minimum age of the players")):
    cursor.execute("SELECT player_name FROM Player WHERE CAST((JULIANDAY('now') - JULIANDAY(birthday)) AS REAL) / 365 >= ?", (age,))
    result = cursor.fetchall()
    if not result:
        return {"player_name": []}
    return {"player_name": [row[0] for row in result]}

# Endpoint to get the total home team goals for a specific player
@app.get("/v1/european_football_2/total_home_team_goals", operation_id="get_total_home_team_goals", summary="Retrieves the cumulative number of goals scored by a specific player when playing as the home team. The player's name is used to filter the results.")
async def get_total_home_team_goals(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT SUM(t2.home_team_goal) FROM Player AS t1 INNER JOIN match AS t2 ON t1.player_api_id = t2.away_player_9 WHERE t1.player_name = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_home_team_goals": []}
    return {"total_home_team_goals": result[0]}

# Endpoint to get the total away team goals for specified players
@app.get("/v1/european_football_2/total_away_team_goals", operation_id="get_total_away_team_goals", summary="Retrieves the total number of goals scored by the away team for the specified players. The operation accepts the names of two players as input parameters and returns the cumulative goals scored by these players when they were part of the away team.")
async def get_total_away_team_goals(player_name_1: str = Query(..., description="Name of the first player"), player_name_2: str = Query(..., description="Name of the second player")):
    cursor.execute("SELECT SUM(t2.away_team_goal) FROM Player AS t1 INNER JOIN match AS t2 ON t1.player_api_id = t2.away_player_5 WHERE t1.player_name IN (?, ?)", (player_name_1, player_name_2))
    result = cursor.fetchone()
    if not result:
        return {"total_away_team_goals": []}
    return {"total_away_team_goals": result[0]}

# Endpoint to get the sum of home team goals for players under a certain age
@app.get("/v1/european_football_2/sum_home_team_goals_by_age", operation_id="get_sum_home_team_goals_by_age", summary="Retrieves the total number of goals scored by home teams in matches where at least one player on the away team is under the specified age. The age is calculated based on the current date and the player's birthday.")
async def get_sum_home_team_goals_by_age(age: int = Query(..., description="Age of the player")):
    cursor.execute("SELECT SUM(t2.home_team_goal) FROM Player AS t1 INNER JOIN match AS t2 ON t1.player_api_id = t2.away_player_1 WHERE datetime(CURRENT_TIMESTAMP, 'localtime') - datetime(T1.birthday) < ?", (age,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the names of players with the highest overall rating
@app.get("/v1/european_football_2/highest_overall_rating_players", operation_id="get_highest_overall_rating_players", summary="Retrieves the names of football players who have achieved the highest overall rating in the European league. This operation identifies these top-rated players by comparing their overall ratings and selecting those with the maximum value. The result is a distinct list of player names.")
async def get_highest_overall_rating_players():
    cursor.execute("SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.overall_rating = (SELECT MAX(overall_rating) FROM Player_Attributes)")
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get the name of the player with the highest potential
@app.get("/v1/european_football_2/highest_potential_player", operation_id="get_highest_potential_player", summary="Retrieves the name of the football player with the highest potential from the European league. The operation identifies the player with the highest potential based on their attributes and returns their name.")
async def get_highest_potential_player():
    cursor.execute("SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id ORDER BY t2.potential DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": result[0]}

# Endpoint to get the names of players with a specific attacking work rate
@app.get("/v1/european_football_2/players_by_attacking_work_rate", operation_id="get_players_by_attacking_work_rate", summary="Retrieve a list of unique player names that match a specified attacking work rate. The attacking work rate is a measure of a player's effort and intensity in offensive play.")
async def get_players_by_attacking_work_rate(attacking_work_rate: str = Query(..., description="Attacking work rate of the player")):
    cursor.execute("SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.attacking_work_rate = ?", (attacking_work_rate,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get the name of the youngest player with a specific finishing rating
@app.get("/v1/european_football_2/youngest_player_by_finishing", operation_id="get_youngest_player_by_finishing", summary="Retrieves the name of the youngest player with the specified finishing rating. The operation filters players based on their finishing rating and returns the name of the youngest player who meets the criteria.")
async def get_youngest_player_by_finishing(finishing: int = Query(..., description="Finishing rating of the player")):
    cursor.execute("SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.finishing = ? ORDER BY t1.birthday ASC LIMIT 1", (finishing,))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": result[0]}

# Endpoint to get the names of players from a specific country
@app.get("/v1/european_football_2/players_by_country", operation_id="get_players_by_country", summary="Retrieve the names of football players who have played a match in a specified European country. The country is identified by its name.")
async def get_players_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT t3.player_name FROM Country AS t1 INNER JOIN Match AS t2 ON t1.id = t2.country_id INNER JOIN Player AS t3 ON t2.home_player_1 = t3.player_api_id WHERE t1.name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get the names of countries with players having a vision rating above a certain value
@app.get("/v1/european_football_2/countries_by_player_vision", operation_id="get_countries_by_player_vision", summary="Retrieve the names of countries that have football players with a vision rating surpassing the provided value. The vision rating is a measure of a player's ability to perceive and anticipate the game's flow.")
async def get_countries_by_player_vision(vision: int = Query(..., description="Vision rating of the player")):
    cursor.execute("SELECT DISTINCT t4.name FROM Player_Attributes AS t1 INNER JOIN Player AS t2 ON t1.player_api_id = t2.player_api_id INNER JOIN Match AS t3 ON t2.player_api_id = t3.home_player_8 INNER JOIN Country AS t4 ON t3.country_id = t4.id WHERE t1.vision > ?", (vision,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the name of the country with the highest average player weight
@app.get("/v1/european_football_2/country_with_highest_avg_player_weight", operation_id="get_country_with_highest_avg_player_weight", summary="Retrieves the name of the country with the highest average player weight. This operation calculates the average weight of players from each country based on their participation in matches and returns the country with the highest average. The result provides insights into the physical attributes of football players across different countries.")
async def get_country_with_highest_avg_player_weight():
    cursor.execute("SELECT t1.name FROM Country AS t1 INNER JOIN Match AS t2 ON t1.id = t2.country_id INNER JOIN Player AS t3 ON t2.home_player_1 = t3.player_api_id GROUP BY t1.name ORDER BY AVG(t3.weight) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the short names of teams with a specific chance creation passing class
@app.get("/v1/european_football_2/teams_by_chance_creation_passing", operation_id="get_teams_by_chance_creation_passing", summary="Retrieve the short names of football teams that belong to a specific chance creation passing class. The chance creation passing class is a metric that measures the team's ability to create scoring opportunities through passing. By specifying a class, you can filter the teams accordingly.")
async def get_teams_by_chance_creation_passing(chance_creation_passing: str = Query(..., description="Chance creation passing class of the team")):
    cursor.execute("SELECT DISTINCT t1.team_short_name FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t2.chanceCreationPassingClass = ?", (chance_creation_passing,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": [row[0] for row in result]}

# Endpoint to get player names based on height and limit
@app.get("/v1/european_football_2/player_names_by_height", operation_id="get_player_names_by_height", summary="Retrieves a list of player names who are taller than the specified minimum height, ordered alphabetically. The number of players returned is limited to the provided value.")
async def get_player_names_by_height(min_height: int = Query(..., description="Minimum height of the player"), limit: int = Query(..., description="Number of players to return")):
    cursor.execute("SELECT player_name FROM Player WHERE height > ? ORDER BY player_name LIMIT ?", (min_height, limit))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the count of players born after a specific date and with a name starting with a specific prefix
@app.get("/v1/european_football_2/player_count_by_birthday_and_name", operation_id="get_player_count_by_birthday_and_name", summary="Retrieves the total number of players who were born after a specified date and have a name that starts with a given prefix. The response is based on the provided date and name prefix parameters.")
async def get_player_count_by_birthday_and_name(birthday: str = Query(..., description="Birthday in 'YYYY-MM-DD' format"), name_prefix: str = Query(..., description="Prefix of the player name")):
    cursor.execute("SELECT COUNT(id) FROM Player WHERE birthday > ? AND player_name LIKE ?", (birthday, name_prefix + '%'))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in jumping attribute between two player IDs
@app.get("/v1/european_football_2/jumping_difference_by_player_ids", operation_id="get_jumping_difference", summary="Retrieve the difference in the jumping attribute between two specified players. The operation compares the jumping attribute of the first player ID with that of the second player ID and returns the difference.")
async def get_jumping_difference(player_id_1: int = Query(..., description="First player ID"), player_id_2: int = Query(..., description="Second player ID")):
    cursor.execute("SELECT SUM(CASE WHEN t1.id = ? THEN t1.jumping ELSE 0 END) - SUM(CASE WHEN t1.id = ? THEN t1.jumping ELSE 0 END) FROM Player_Attributes AS t1", (player_id_1, player_id_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get player IDs based on preferred foot and limit
@app.get("/v1/european_football_2/player_ids_by_preferred_foot", operation_id="get_player_ids_by_preferred_foot", summary="Retrieves a list of player IDs who primarily use a specified foot for play, sorted by their potential skill level in descending order. The number of player IDs returned is limited to the specified quantity.")
async def get_player_ids_by_preferred_foot(preferred_foot: str = Query(..., description="Preferred foot of the player"), limit: int = Query(..., description="Number of players to return")):
    cursor.execute("SELECT id FROM Player_Attributes WHERE preferred_foot = ? ORDER BY potential DESC LIMIT ?", (preferred_foot, limit))
    result = cursor.fetchall()
    if not result:
        return {"player_ids": []}
    return {"player_ids": [row[0] for row in result]}

# Endpoint to get the count of players with a specific preferred foot and maximum crossing attribute
@app.get("/v1/european_football_2/player_count_by_preferred_foot_and_max_crossing", operation_id="get_player_count_by_preferred_foot_and_max_crossing", summary="Retrieves the total number of players who predominantly use a specified foot and have the highest crossing ability. The preferred foot can be either 'Left' or 'Right'.")
async def get_player_count_by_preferred_foot_and_max_crossing(preferred_foot: str = Query(..., description="Preferred foot of the player")):
    cursor.execute("SELECT COUNT(t1.id) FROM Player_Attributes AS t1 WHERE t1.preferred_foot = ? AND t1.crossing = ( SELECT MAX(crossing) FROM Player_Attributes)", (preferred_foot,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of players with strength and stamina above specific values
@app.get("/v1/european_football_2/percentage_players_by_strength_and_stamina", operation_id="get_percentage_players_by_strength_and_stamina", summary="Retrieves the percentage of players who have strength and stamina values above the specified minimum thresholds. The calculation is based on the total number of players in the database.")
async def get_percentage_players_by_strength_and_stamina(min_strength: int = Query(..., description="Minimum strength value"), min_stamina: int = Query(..., description="Minimum stamina value")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN strength > ? AND stamina > ? THEN id ELSE NULL END) AS REAL) * 100 / COUNT(id) FROM Player_Attributes t", (min_strength, min_stamina))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get country names based on league name
@app.get("/v1/european_football_2/country_names_by_league", operation_id="get_country_names_by_league", summary="Retrieves the names of countries that participate in a specified league. The league is identified by its name.")
async def get_country_names_by_league(league_name: str = Query(..., description="Name of the league")):
    cursor.execute("SELECT name FROM Country WHERE id IN ( SELECT country_id FROM League WHERE name = ? )", (league_name,))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get match goals based on league name and date
@app.get("/v1/european_football_2/match_goals_by_league_and_date", operation_id="get_match_goals_by_league_and_date", summary="Retrieves the number of goals scored by both the home and away teams in matches played on a specific date within a given league. The league is identified by its name, and the date is provided in 'YYYY-MM-DD' format.")
async def get_match_goals_by_league_and_date(league_name: str = Query(..., description="Name of the league"), match_date: str = Query(..., description="Match date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT t2.home_team_goal, t2.away_team_goal FROM League AS t1 INNER JOIN Match AS t2 ON t1.id = t2.league_id WHERE t1.name = ? AND t2.`date` LIKE ?", (league_name, match_date + '%'))
    result = cursor.fetchall()
    if not result:
        return {"match_goals": []}
    return {"match_goals": [{"home_team_goal": row[0], "away_team_goal": row[1]} for row in result]}

# Endpoint to get player attributes based on player name
@app.get("/v1/european_football_2/player_attributes_by_name", operation_id="get_player_attributes_by_name", summary="Retrieves the sprint speed, agility, and acceleration attributes of a specific player. The player is identified by their name, which is provided as an input parameter.")
async def get_player_attributes_by_name(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT sprint_speed, agility, acceleration FROM Player_Attributes WHERE player_api_id IN ( SELECT player_api_id FROM Player WHERE player_name = ? )", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"player_attributes": []}
    return {"player_attributes": [{"sprint_speed": row[0], "agility": row[1], "acceleration": row[2]} for row in result]}

# Endpoint to get distinct build-up play speed classes for a specific team
@app.get("/v1/european_football_2/build_up_play_speed_class", operation_id="get_build_up_play_speed_class", summary="Retrieve the unique build-up play speed classes associated with a specific football team. The team is identified by its long name, which is provided as an input parameter.")
async def get_build_up_play_speed_class(team_long_name: str = Query(..., description="Long name of the team")):
    cursor.execute("SELECT DISTINCT t1.buildUpPlaySpeedClass FROM Team_Attributes AS t1 INNER JOIN Team AS t2 ON t1.team_api_id = t2.team_api_id WHERE t2.team_long_name = ?", (team_long_name,))
    result = cursor.fetchall()
    if not result:
        return {"build_up_play_speed_classes": []}
    return {"build_up_play_speed_classes": [row[0] for row in result]}

# Endpoint to get the count of matches for a specific league and season
@app.get("/v1/european_football_2/match_count_league_season", operation_id="get_match_count", summary="Retrieves the total number of matches played in a specific football league during a given season. The league is identified by its name, and the season is specified in the 'YYYY/YYYY' format.")
async def get_match_count(league_name: str = Query(..., description="Name of the league"), season: str = Query(..., description="Season in 'YYYY/YYYY' format")):
    cursor.execute("SELECT COUNT(t2.id) FROM League AS t1 INNER JOIN Match AS t2 ON t1.id = t2.league_id WHERE t1.name = ? AND t2.season = ?", (league_name, season))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum home team goals for a specific league
@app.get("/v1/european_football_2/max_home_team_goals", operation_id="get_max_home_team_goals", summary="Retrieves the highest number of goals scored by a home team in a specific league. The league is identified by its name.")
async def get_max_home_team_goals(league_name: str = Query(..., description="Name of the league")):
    cursor.execute("SELECT MAX(t2.home_team_goal) FROM League AS t1 INNER JOIN Match AS t2 ON t1.id = t2.league_id WHERE t1.name = ?", (league_name,))
    result = cursor.fetchone()
    if not result:
        return {"max_home_team_goals": []}
    return {"max_home_team_goals": result[0]}

# Endpoint to get player attributes for the heaviest player
@app.get("/v1/european_football_2/heaviest_player_attributes", operation_id="get_heaviest_player_attributes", summary="Retrieves the attributes of the heaviest player in the database. The attributes returned include the player's unique identifier, finishing skill, and curve skill.")
async def get_heaviest_player_attributes():
    cursor.execute("SELECT id, finishing, curve FROM Player_Attributes WHERE player_api_id = ( SELECT player_api_id FROM Player ORDER BY weight DESC LIMIT 1 ) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"attributes": []}
    return {"attributes": {"id": result[0], "finishing": result[1], "curve": result[2]}}

# Endpoint to get top 4 leagues by match count for a specific season
@app.get("/v1/european_football_2/top_leagues_by_match_count", operation_id="get_top_leagues_by_match_count", summary="Retrieves the top four European football leagues with the highest number of matches played in a specified season. The leagues are ranked in descending order based on the match count.")
async def get_top_leagues_by_match_count(season: str = Query(..., description="Season in 'YYYY/YYYY' format")):
    cursor.execute("SELECT t1.name FROM League AS t1 INNER JOIN Match AS t2 ON t1.id = t2.league_id WHERE t2.season = ? GROUP BY t1.name ORDER BY COUNT(t2.id) DESC LIMIT 4", (season,))
    result = cursor.fetchall()
    if not result:
        return {"leagues": []}
    return {"leagues": [row[0] for row in result]}

# Endpoint to get the team with the highest away team goals
@app.get("/v1/european_football_2/top_away_team_by_goals", operation_id="get_top_away_team_by_goals", summary="Retrieves the team with the highest number of goals scored in away matches. The team is identified by comparing the goals scored by each team in their respective away matches and selecting the team with the highest count.")
async def get_top_away_team_by_goals():
    cursor.execute("SELECT t2.team_long_name FROM Match AS t1 INNER JOIN Team AS t2 ON t1.away_team_api_id = t2.team_api_id ORDER BY t1.away_team_goal DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result[0]}

# Endpoint to get distinct player names with the highest overall rating
@app.get("/v1/european_football_2/top_rated_players", operation_id="get_top_rated_players", summary="Retrieves a list of unique player names who have achieved the highest overall rating in the game. This operation identifies these top-rated players by comparing their overall ratings and only returns those with the maximum value.")
async def get_top_rated_players():
    cursor.execute("SELECT DISTINCT t1.player_name FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t2.overall_rating = ( SELECT MAX(overall_rating) FROM Player_Attributes)")
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get the percentage of players with overall rating above a certain threshold and below a certain height
@app.get("/v1/european_football_2/player_rating_percentage", operation_id="get_player_rating_percentage", summary="Retrieves the percentage of players who have an overall rating above the specified threshold and are shorter than the provided height limit. The calculation is based on the total number of players in the database.")
async def get_player_rating_percentage(overall_rating: int = Query(..., description="Overall rating threshold"), height: int = Query(..., description="Height threshold in cm")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN t2.overall_rating > ? THEN t1.id ELSE NULL END) AS REAL) * 100 / COUNT(t1.id) percent FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_api_id = t2.player_api_id WHERE t1.height < ?", (overall_rating, height))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/european_football_2/top_player_by_overall_rating?limit=1",
    "/v1/european_football_2/tallest_player?limit=1",
    "/v1/european_football_2/preferred_foot_lowest_potential?limit=1",
    "/v1/european_football_2/count_players_by_rating_and_work_rate?min_rating=60&max_rating=65&defensive_work_rate=low",
    "/v1/european_football_2/top_players_by_crossing?limit=5",
    "/v1/european_football_2/top_league_by_total_goals?season=2015/2016&limit=1",
    "/v1/european_football_2/team_with_fewest_home_losses?season=2015/2016&limit=1",
    "/v1/european_football_2/top_players_by_penalties?limit=10",
    "/v1/european_football_2/team_with_most_away_wins?league_name=Scotland%20Premier%20League&season=2009/2010&limit=1",
    "/v1/european_football_2/teams_with_lowest_build_up_play_speed?limit=4",
    "/v1/european_football_2/league_most_draws?season=2015/2016",
    "/v1/european_football_2/player_age_sprint_speed?start_year=2013&end_year=2015&sprint_speed=97",
    "/v1/european_football_2/league_max_matches",
    "/v1/european_football_2/average_player_height?start_year=1990&end_year=1995",
    "/v1/european_football_2/top_player_by_year?year=2010",
    "/v1/european_football_2/team_build_up_play_speed?min_speed=50&max_speed=60",
    "/v1/european_football_2/team_build_up_play_passing?year=2012",
    "/v1/european_football_2/percentage_left_footed_players?start_year=1987&end_year=1992",
    "/v1/european_football_2/leagues_fewest_goals",
    "/v1/european_football_2/average_long_shots_player?player_name=Ahmed%20Samir%20Farag",
    "/v1/european_football_2/top_players_by_heading_accuracy?min_height=180&limit=10",
    "/v1/european_football_2/teams_by_build_up_play_dribbling_class?build_up_play_dribbling_class=Normal&year=2014",
    "/v1/european_football_2/leagues_with_more_home_goals?season=2009/2010",
    "/v1/european_football_2/team_short_name_by_long_name?team_long_name=Queens%20Park%20Rangers",
    "/v1/european_football_2/players_by_birth_month_year?birth_month_year=1970-10",
    "/v1/european_football_2/attacking_work_rate_by_player?player_name=Franco%20Zennaro",
    "/v1/european_football_2/build_up_play_positioning_class_by_team?team_long_name=ADO%20Den%20Haag",
    "/v1/european_football_2/heading_accuracy_by_player_date?player_name=Francois%20Affolter&date=2014-09-18",
    "/v1/european_football_2/overall_rating_by_player_year?player_name=Gabriel%20Tamas&year=2011",
    "/v1/european_football_2/match_count_by_league_season?season=2015/2016&league_name=Scotland%20Premier%20League",
    "/v1/european_football_2/preferred_foot_most_recent_player",
    "/v1/european_football_2/players_with_highest_potential",
    "/v1/european_football_2/count_players_weight_preferred_foot?weight=130&preferred_foot=left",
    "/v1/european_football_2/team_short_names_chance_creation_passing_class?chance_creation_passing_class=Risky",
    "/v1/european_football_2/defensive_work_rate_player?player_name=David%20Wilson",
    "/v1/european_football_2/birthday_highest_overall_rating",
    "/v1/european_football_2/league_names_by_country?country_name=Netherlands",
    "/v1/european_football_2/average_home_team_goals_country_season?country_name=Poland&season=2010/2011",
    "/v1/european_football_2/average_finishing_extreme_heights",
    "/v1/european_football_2/players_taller_than?height=180",
    "/v1/european_football_2/player_count_by_birth_year?year=1990",
    "/v1/european_football_2/player_count_by_weight_and_name?weight=170&name_start=Adam",
    "/v1/european_football_2/distinct_player_names_by_rating_and_date?overall_rating=80&start_year=2008&end_year=2010",
    "/v1/european_football_2/player_potential_by_name?player_name=Aaron%20Doran",
    "/v1/european_football_2/distinct_players_by_preferred_foot?preferred_foot=left",
    "/v1/european_football_2/distinct_team_long_names_by_build_up_play_speed?build_up_play_speed_class=Fast",
    "/v1/european_football_2/distinct_build_up_play_passing_classes_by_team_short_name?team_short_name=CLB",
    "/v1/european_football_2/distinct_team_short_names_by_build_up_play_passing?build_up_play_passing=70",
    "/v1/european_football_2/average_overall_rating_by_height_and_date?height=170&start_year=2010&end_year=2015",
    "/v1/european_football_2/shortest_player",
    "/v1/european_football_2/country_name_by_league?league_name=Italy%20Serie%20A",
    "/v1/european_football_2/team_short_names_by_buildup_play?build_up_play_speed=31&build_up_play_dribbling=53&build_up_play_passing=32",
    "/v1/european_football_2/average_overall_rating_by_player?player_name=Aaron%20Doran",
    "/v1/european_football_2/match_count_by_league_and_date_range?league_name=Germany%201.%20Bundesliga&start_date=2008-08&end_date=2008-10",
    "/v1/european_football_2/team_short_names_by_home_team_goals?home_team_goal=10",
    "/v1/european_football_2/player_name_by_potential_and_balance?potential=61",
    "/v1/european_football_2/ball_control_difference_between_players?player_name_1=Abdou%20Diallo&player_name_2=Aaron%20Appindangoye",
    "/v1/european_football_2/team_long_name_by_short_name?team_short_name=GEN",
    "/v1/european_football_2/youngest_player_between_two?player_name_1=Aaron%20Lennon&player_name_2=Abdelaziz%20Barrada",
    "/v1/european_football_2/player_count_by_preferred_foot_and_attacking_work_rate?preferred_foot=left&attacking_work_rate=low",
    "/v1/european_football_2/count_players_by_birth_year_and_defensive_work_rate?birth_year=1986&defensive_work_rate=high",
    "/v1/european_football_2/top_player_by_crossing_skill?player_name1=Alexis&player_name2=Ariel%20Borysiuk&player_name3=Arouna%20Kone",
    "/v1/european_football_2/heading_accuracy_by_player_name?player_name=Ariel%20Borysiuk",
    "/v1/european_football_2/count_players_by_height_and_volleys?height=180&volleys=70",
    "/v1/european_football_2/player_names_by_volleys_and_dribbling?volleys=70&dribbling=70",
    "/v1/european_football_2/count_matches_by_country_and_season?country_name=Belgium&season=2008/2009",
    "/v1/european_football_2/long_passing_youngest_player",
    "/v1/european_football_2/match_count_by_league_and_month?league_name=Belgium%20Jupiler%20League&month_year=2009-04",
    "/v1/european_football_2/league_with_max_matches_in_season?season=2008/2009",
    "/v1/european_football_2/average_overall_rating_by_birth_year?birth_year=1986",
    "/v1/european_football_2/percentage_difference_in_overall_ratings?player_name_1=Ariel%20Borysiuk&player_name_2=Paulin%20Puel",
    "/v1/european_football_2/average_build_up_play_speed_by_team?team_long_name=Heart%20of%20Midlothian",
    "/v1/european_football_2/total_crossing_by_player?player_name=Aaron%20Lennox",
    "/v1/european_football_2/chance_creation_passing_by_team?team_long_name=Ajax",
    "/v1/european_football_2/preferred_foot_by_player?player_name=Abdou%20Diallo",
    "/v1/european_football_2/max_overall_rating_by_player?player_name=Dorlan%20Pabon",
    "/v1/european_football_2/average_away_team_goals?team_long_name=Parma&country_name=Italy",
    "/v1/european_football_2/player_name_by_rating_date?date=2016-06-23&overall_rating=77",
    "/v1/european_football_2/player_overall_rating?date=2016-02-04&player_name=Aaron%20Mooy",
    "/v1/european_football_2/player_potential?date=2010-08-30&player_name=Francesco%20Parravicini",
    "/v1/european_football_2/player_attacking_work_rate?date=2015-05-01%25&player_name=Francesco%20Migliore",
    "/v1/european_football_2/player_defensive_work_rate?date=2013-02-22&player_name=Kevin%20Berigaud",
    "/v1/european_football_2/player_highest_crossing_date?player_name=Kevin%20Constant",
    "/v1/european_football_2/team_build_up_play_speed_class?team_long_name=Willem%20II&date=2011-02-22",
    "/v1/european_football_2/team_build_up_play_dribbling_class?team_short_name=LEI&date=2015-09-10",
    "/v1/european_football_2/team_build_up_play_passing_class?team_long_name=FC%20Lorient&date=2010-02-22%25",
    "/v1/european_football_2/chance_creation_passing_class?team_long_name=PEC%20Zwolle&date=2013-09-20",
    "/v1/european_football_2/chance_creation_crossing_class?team_long_name=Hull%20City&date=2010-02-22",
    "/v1/european_football_2/chance_creation_shooting_class?team_long_name=Hannover%2096&date=2015-09-10%",
    "/v1/european_football_2/average_overall_rating?player_name=Marko%20Arnautovic&start_date=2007-02-22&end_date=2016-04-21",
    "/v1/european_football_2/overall_rating_percentage_difference?player_name_1=Landon%20Donovan&player_name_2=Jordan%20Bowery&date=2013-07-12",
    "/v1/european_football_2/heaviest_players?limit=10",
    "/v1/european_football_2/players_older_than?age=35",
    "/v1/european_football_2/total_home_team_goals?player_name=Aaron%20Lennon",
    "/v1/european_football_2/total_away_team_goals?player_name_1=Daan%20Smith&player_name_2=Filipe%20Ferreira",
    "/v1/european_football_2/sum_home_team_goals_by_age?age=31",
    "/v1/european_football_2/highest_overall_rating_players",
    "/v1/european_football_2/highest_potential_player",
    "/v1/european_football_2/players_by_attacking_work_rate?attacking_work_rate=high",
    "/v1/european_football_2/youngest_player_by_finishing?finishing=1",
    "/v1/european_football_2/players_by_country?country_name=Belgium",
    "/v1/european_football_2/countries_by_player_vision?vision=89",
    "/v1/european_football_2/country_with_highest_avg_player_weight",
    "/v1/european_football_2/teams_by_chance_creation_passing?chance_creation_passing=Safe",
    "/v1/european_football_2/player_names_by_height?min_height=180&limit=3",
    "/v1/european_football_2/player_count_by_birthday_and_name?birthday=1990&name_prefix=Aaron",
    "/v1/european_football_2/jumping_difference_by_player_ids?player_id_1=6&player_id_2=23",
    "/v1/european_football_2/player_ids_by_preferred_foot?preferred_foot=right&limit=5",
    "/v1/european_football_2/player_count_by_preferred_foot_and_max_crossing?preferred_foot=left",
    "/v1/european_football_2/percentage_players_by_strength_and_stamina?min_strength=80&min_stamina=80",
    "/v1/european_football_2/country_names_by_league?league_name=Poland%20Ekstraklasa",
    "/v1/european_football_2/match_goals_by_league_and_date?league_name=Belgium%20Jupiler%20League&match_date=2008-09-24",
    "/v1/european_football_2/player_attributes_by_name?player_name=Alexis%20Blin",
    "/v1/european_football_2/build_up_play_speed_class?team_long_name=KSV%20Cercle%20Brugge",
    "/v1/european_football_2/match_count_league_season?league_name=Italy%20Serie%20A&season=2015/2016",
    "/v1/european_football_2/max_home_team_goals?league_name=Netherlands%20Eredivisie",
    "/v1/european_football_2/heaviest_player_attributes",
    "/v1/european_football_2/top_leagues_by_match_count?season=2015/2016",
    "/v1/european_football_2/top_away_team_by_goals",
    "/v1/european_football_2/top_rated_players",
    "/v1/european_football_2/player_rating_percentage?overall_rating=70&height=180"
]
