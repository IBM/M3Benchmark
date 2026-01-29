from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/ice_hockey_draft/ice_hockey_draft.sqlite')
cursor = conn.cursor()

# Endpoint to get the height in inches of a player by name
@app.get("/v1/ice_hockey_draft/player_height_in_inches", operation_id="get_player_height_in_inches", summary="Retrieves the height in inches of a specific ice hockey player, identified by their name. The player's height is fetched from a database table that maps height identifiers to their corresponding values in inches.")
async def get_player_height_in_inches(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T2.height_in_inch FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id WHERE T1.PlayerName = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"height_in_inches": []}
    return {"height_in_inches": result[0]}

# Endpoint to get distinct player names with height greater than a specified value
@app.get("/v1/ice_hockey_draft/players_above_height", operation_id="get_players_above_height", summary="Retrieve a list of unique player names who are taller than the specified height. The height should be provided in the format 'feet''inches'.")
async def get_players_above_height(height_in_inch: str = Query(..., description="Height in inches in the format 'feet''inches'")):
    cursor.execute("SELECT DISTINCT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id WHERE T2.height_in_inch > ?", (height_in_inch,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the count of players with height greater than a specified value and from a specific nation
@app.get("/v1/ice_hockey_draft/count_players_height_nation", operation_id="get_count_players_height_nation", summary="Retrieves the count of ice hockey players from a specific nation who are taller than a given height. The height is provided in inches, in the format 'feet''inches'. The nation is specified by its name.")
async def get_count_players_height_nation(height_in_inch: str = Query(..., description="Height in inches in the format 'feet''inches'"), nation: str = Query(..., description="Nation of the player")):
    cursor.execute("SELECT COUNT(T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id WHERE T2.height_in_inch > ? AND T1.nation = ?", (height_in_inch, nation))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the tallest player by height in centimeters
@app.get("/v1/ice_hockey_draft/tallest_player", operation_id="get_tallest_player", summary="Retrieves the name of the tallest ice hockey player based on their height in centimeters. The operation sorts players by height in descending order and returns the name of the player with the highest height value.")
async def get_tallest_player():
    cursor.execute("SELECT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id ORDER BY T2.height_in_cm DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the weight in kilograms of a player by name
@app.get("/v1/ice_hockey_draft/player_weight_in_kg", operation_id="get_player_weight_in_kg", summary="Retrieves the weight in kilograms of a specific ice hockey player, identified by their name. The operation fetches the weight information from the weight_info table, which is linked to the player's record in the PlayerInfo table via a weight identifier.")
async def get_player_weight_in_kg(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T2.weight_in_kg FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T1.PlayerName = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"weight_in_kg": []}
    return {"weight_in_kg": result[0]}

# Endpoint to get the count of players with weight greater than a specified value
@app.get("/v1/ice_hockey_draft/count_players_above_weight", operation_id="get_count_players_above_weight", summary="Retrieves the total number of ice hockey players who weigh more than the specified weight in kilograms. This operation is useful for understanding the distribution of player weights in the draft.")
async def get_count_players_above_weight(weight_in_kg: int = Query(..., description="Weight in kilograms")):
    cursor.execute("SELECT COUNT(T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T2.weight_in_kg > ?", (weight_in_kg,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of players with weight greater than a specified value and a specific position
@app.get("/v1/ice_hockey_draft/count_players_weight_position", operation_id="get_count_players_weight_position", summary="Retrieves the number of ice hockey players who weigh more than the specified weight (in kilograms) and play a particular position. This operation provides a statistical overview of the player roster based on weight and position.")
async def get_count_players_weight_position(weight_in_kg: int = Query(..., description="Weight in kilograms"), position_info: str = Query(..., description="Position of the player")):
    cursor.execute("SELECT COUNT(T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T2.weight_in_kg > ? AND T1.position_info = ?", (weight_in_kg, position_info))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the player with the maximum 7-year games played among players with weight greater than a specified value
@app.get("/v1/ice_hockey_draft/max_7yr_gp_above_weight", operation_id="get_max_7yr_gp_above_weight", summary="Retrieves the name of the ice hockey player who has played the most games over a 7-year period, among players weighing more than the specified weight in kilograms.")
async def get_max_7yr_gp_above_weight(weight_in_kg: int = Query(..., description="Weight in kilograms")):
    cursor.execute("SELECT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T2.weight_in_kg > ? AND T1.sum_7yr_GP = ( SELECT MAX(T1.sum_7yr_GP) FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T2.weight_in_kg > ? )", (weight_in_kg, weight_in_kg))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the weight in kilograms of the player with the maximum 7-year time on ice
@app.get("/v1/ice_hockey_draft/weight_max_7yr_toi", operation_id="get_weight_max_7yr_toi", summary="Retrieves the weight of the player who has accumulated the most time on ice over a 7-year period. The weight is returned in kilograms.")
async def get_weight_max_7yr_toi():
    cursor.execute("SELECT T2.weight_in_kg FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T1.sum_7yr_TOI = ( SELECT MAX(t.sum_7yr_TOI) FROM PlayerInfo t )")
    result = cursor.fetchone()
    if not result:
        return {"weight_in_kg": []}
    return {"weight_in_kg": result[0]}

# Endpoint to get the height difference in centimeters between two players
@app.get("/v1/ice_hockey_draft/height_difference", operation_id="get_height_difference", summary="Get the height difference in centimeters between two players")
async def get_height_difference(player_name_1: str = Query(..., description="Name of the first player"), player_name_2: str = Query(..., description="Name of the second player")):
    cursor.execute("( SELECT T2.height_in_cm FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id WHERE T1.PlayerName = ? ) - ( SELECT T2.height_in_cm FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id WHERE T1.PlayerName = ? )", (player_name_1, player_name_2))
    result = cursor.fetchone()
    if not result:
        return {"height_difference": []}
    return {"height_difference": result[0]}

# Endpoint to get the count of players based on weight and shooting hand
@app.get("/v1/ice_hockey_draft/count_players_by_weight_shoots", operation_id="get_count_players_by_weight_shoots", summary="Retrieves the total number of ice hockey players who weigh more than the specified weight (in kilograms) and shoot with the given hand. The weight is compared in kilograms, and the shooting hand is represented by a single character (e.g., 'R' for right).")
async def get_count_players_by_weight_shoots(weight_in_kg: int = Query(..., description="Weight in kilograms"), shoots: str = Query(..., description="Shooting hand (e.g., 'R' for right)")):
    cursor.execute("SELECT COUNT(T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T2.weight_in_kg > ? AND T1.shoots = ?", (weight_in_kg, shoots))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get player names based on weight and shooting hand
@app.get("/v1/ice_hockey_draft/player_names_by_weight_shoots", operation_id="get_player_names_by_weight_shoots", summary="Retrieves the names of ice hockey players who weigh more than the specified weight (in kilograms) and shoot with the provided hand. The weight is compared in kilograms, and the shooting hand is denoted by a single character (e.g., 'R' for right).")
async def get_player_names_by_weight_shoots(weight_in_kg: int = Query(..., description="Weight in kilograms"), shoots: str = Query(..., description="Shooting hand (e.g., 'R' for right)")):
    cursor.execute("SELECT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T2.weight_in_kg > ? AND T1.shoots = ?", (weight_in_kg, shoots))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the BMI of a player by name
@app.get("/v1/ice_hockey_draft/player_bmi_by_name", operation_id="get_player_bmi_by_name", summary="Retrieves the Body Mass Index (BMI) of a specific ice hockey player, calculated based on their weight and height. The player is identified by their name, which is provided as an input parameter.")
async def get_player_bmi_by_name(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT CAST(T2.weight_in_kg AS REAL) / (CAST(T3.height_in_cm AS REAL) / 100 * (CAST(T3.height_in_cm AS REAL) / 100)) FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id INNER JOIN height_info AS T3 ON T1.height = T3.height_id WHERE T1.PlayerName = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"bmi": []}
    return {"bmi": result[0]}

# Endpoint to get the average height of players by position
@app.get("/v1/ice_hockey_draft/average_height_by_position", operation_id="get_average_height_by_position", summary="Retrieves the average height of ice hockey players based on their position. The position is specified as an input parameter, with 'D' representing defense, for example. The calculation is performed by summing the heights of all players in the specified position and dividing by the total count of players in that position.")
async def get_average_height_by_position(position_info: str = Query(..., description="Position of the player (e.g., 'D' for defense)")):
    cursor.execute("SELECT CAST(SUM(T2.height_in_cm) AS REAL) / COUNT(T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id WHERE T1.position_info = ?", (position_info,))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the maximum weight in pounds of players
@app.get("/v1/ice_hockey_draft/max_weight_in_lbs", operation_id="get_max_weight_in_lbs", summary="Retrieves the maximum weight, in pounds, of all players in the database. This operation does not require any input parameters and returns a single value representing the heaviest weight recorded.")
async def get_max_weight_in_lbs():
    cursor.execute("SELECT MAX(T2.weight_in_lbs) FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id")
    result = cursor.fetchone()
    if not result:
        return {"max_weight": []}
    return {"max_weight": result[0]}

# Endpoint to get the count of players based on height and shooting hand
@app.get("/v1/ice_hockey_draft/count_players_by_height_shoots", operation_id="get_count_players_by_height_shoots", summary="Retrieves the total number of ice hockey players who have a specified height (in inches) and shooting hand. The height is provided in inches, and the shooting hand is represented by a single letter (e.g., 'R' for right).")
async def get_count_players_by_height_shoots(height_in_inch: str = Query(..., description="""Height in inches (e.g., '5''7"')"""), shoots: str = Query(..., description="Shooting hand (e.g., 'R' for right)")):
    cursor.execute("SELECT COUNT(T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id WHERE T2.height_in_inch = ? AND T1.shoots = ?", (height_in_inch, shoots))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the player name with the highest PIM and more than a specified number of games played in 7 years
@app.get("/v1/ice_hockey_draft/player_name_by_pim_and_games", operation_id="get_player_name_by_pim_and_games", summary="Retrieves the name of the ice hockey player who has accumulated the highest penalty minutes (PIM) and has played more than a specified number of games over a 7-year period.")
async def get_player_name_by_pim_and_games(sum_7yr_GP: int = Query(..., description="Sum of games played in 7 years")):
    cursor.execute("SELECT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.sum_7yr_GP > ? ORDER BY T2.PIM DESC LIMIT 1", (sum_7yr_GP,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the tallest player from a specific birthplace
@app.get("/v1/ice_hockey_draft/tallest_player_by_birthplace", operation_id="get_tallest_player_by_birthplace", summary="Retrieves the height of the tallest ice hockey player born in a specified location. The birthplace is provided as a parameter, and the response will contain the height in centimeters of the tallest player from that location.")
async def get_tallest_player_by_birthplace(birthplace: str = Query(..., description="Birthplace of the player (e.g., 'Edmonton, AB, CAN')")):
    cursor.execute("SELECT T2.height_in_cm FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id WHERE T1.birthplace = ? ORDER BY T2.height_in_cm DESC LIMIT 1", (birthplace,))
    result = cursor.fetchone()
    if not result:
        return {"height": []}
    return {"height": result[0]}

# Endpoint to get the count of distinct players based on overallby, draft year, and team
@app.get("/v1/ice_hockey_draft/count_distinct_players_by_overallby_draftyear_team", operation_id="get_count_distinct_players_by_overallby_draftyear_team", summary="Retrieve the number of unique players drafted by a specific team in a given year, based on their overall ranking.")
async def get_count_distinct_players_by_overallby_draftyear_team(overallby: str = Query(..., description="Overall by (e.g., 'Anaheim Ducks')"), draftyear: int = Query(..., description="Draft year (e.g., 2008)"), team: str = Query(..., description="Team (e.g., 'U.S. National U18 Team')")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.overallby = ? AND T1.draftyear = ? AND T2.TEAM = ?", (overallby, draftyear, team))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the weight of the player with the highest plus-minus rating
@app.get("/v1/ice_hockey_draft/weight_of_player_with_highest_plusminus", operation_id="get_weight_of_player_with_highest_plusminus", summary="Retrieves the weight, in kilograms, of the player who has the highest plus-minus rating in the season. The plus-minus rating is a metric that reflects a player's impact on the game, taking into account goals scored and conceded while the player is on the ice. The weight is obtained from the weight_info table, which is linked to the player's information.")
async def get_weight_of_player_with_highest_plusminus():
    cursor.execute("SELECT T3.weight_in_kg FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID INNER JOIN weight_info AS T3 ON T2.weight = T3.weight_id ORDER BY T1.PLUSMINUS DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"weight": []}
    return {"weight": result[0]}

# Endpoint to get the player name with the highest points in a specific season range and league
@app.get("/v1/ice_hockey_draft/top_player_by_season_range_and_league", operation_id="get_top_player_by_season_range_and_league", summary="Retrieves the name of the player with the highest points in a specified season range and league. The operation accepts the start and end seasons in 'YYYY' format, along with the league name. It returns the name of the player who has accumulated the most points within the provided season range and league.")
async def get_top_player_by_season_range_and_league(start_season: str = Query(..., description="Start season in 'YYYY' format"), end_season: str = Query(..., description="End season in 'YYYY' format"), league: str = Query(..., description="League name")):
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON BETWEEN ? AND ? AND T1.LEAGUE = ? ORDER BY T1.P DESC LIMIT 1", (start_season, end_season, league))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get player names based on season, game type, and team
@app.get("/v1/ice_hockey_draft/player_names_by_season_game_type_team", operation_id="get_player_names_by_season_game_type_team", summary="Retrieves the names of ice hockey players who played in a specific season, game type, and team. The season is specified in 'YYYY-YYYY' format, the game type is selected from available options, and the team name is provided. This operation returns a list of player names that meet the given criteria.")
async def get_player_names_by_season_game_type_team(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), game_type: str = Query(..., description="Game type"), team: str = Query(..., description="Team name")):
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.GAMETYPE = ? AND T1.TEAM = ?", (season, game_type, team))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the games played by the tallest player
@app.get("/v1/ice_hockey_draft/games_played_by_tallest_player", operation_id="get_games_played_by_tallest_player", summary="Get the games played by the tallest player")
async def get_games_played_by_tallest_player():
    cursor.execute("SELECT T1.GP FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.ELITEID = ( SELECT t.ELITEID FROM PlayerInfo t ORDER BY t.height DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"games_played": []}
    return {"games_played": result[0]}

# Endpoint to get the youngest player name in a specific season and league
@app.get("/v1/ice_hockey_draft/youngest_player_by_season_and_league", operation_id="get_youngest_player_by_season_and_league", summary="Retrieves the name of the youngest player who participated in a given season and league. The season is specified in 'YYYY-YYYY' format, and the league name is provided. The player's birthdate is used to determine their age and identify the youngest player.")
async def get_youngest_player_by_season_and_league(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), league: str = Query(..., description="League name")):
    cursor.execute("SELECT DISTINCT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.LEAGUE = ? ORDER BY T2.birthdate DESC LIMIT 1", (season, league))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the count of players with specific games played and shooting hand
@app.get("/v1/ice_hockey_draft/count_players_by_games_played_and_shooting_hand", operation_id="get_count_players_by_games_played_and_shooting_hand", summary="Retrieves the count of ice hockey players who have played a specific number of games and use a particular shooting hand. The input parameters define the number of games played and the shooting hand (L for left or R for right).")
async def get_count_players_by_games_played_and_shooting_hand(games_played: int = Query(..., description="Number of games played"), shooting_hand: str = Query(..., description="Shooting hand (L or R)")):
    cursor.execute("SELECT COUNT(T2.ELITEID) FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.GP = ? AND T2.shoots = ?", (games_played, shooting_hand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in goals between regular season and playoffs for a specific player
@app.get("/v1/ice_hockey_draft/goal_difference_regular_playoffs", operation_id="get_goal_difference_regular_playoffs", summary="Retrieve the difference in goals scored by a specific player between the regular season and playoffs for a given year. The operation requires the player's name, the season in 'YYYY-YYYY' format, and the respective game types for regular season and playoffs.")
async def get_goal_difference_regular_playoffs(player_name: str = Query(..., description="Player name"), season: str = Query(..., description="Season in 'YYYY-YYYY' format"), regular_game_type: str = Query(..., description="Regular game type"), playoff_game_type: str = Query(..., description="Playoff game type")):
    cursor.execute("SELECT T3.Rs_G - T4.Pf_G AS diff FROM ( SELECT T2.PlayerName, T1.G AS Rs_G FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.PlayerName = ? AND T1.SEASON = ? AND T1.GAMETYPE = ? ) AS T3 INNER JOIN ( SELECT T2.PlayerName, T1.G AS Pf_G FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.PlayerName = ? AND T1.SEASON = ? AND T1.GAMETYPE = ? ) AS T4 ON T3.PlayerName = T4.PlayerName", (player_name, season, regular_game_type, player_name, season, playoff_game_type))
    result = cursor.fetchone()
    if not result:
        return {"goal_difference": []}
    return {"goal_difference": result[0]}

# Endpoint to get the average weight of players with the highest CSS rank
@app.get("/v1/ice_hockey_draft/average_weight_highest_css_rank", operation_id="get_average_weight_highest_css_rank", summary="Retrieves the average weight of the top-ranked ice hockey players according to the CSS rank. The calculation is based on the sum of the weights of these players divided by their count.")
async def get_average_weight_highest_css_rank():
    cursor.execute("SELECT CAST(SUM(T2.weight_in_lbs) AS REAL) / COUNT(T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T1.CSS_rank = ( SELECT MAX(CSS_rank) FROM PlayerInfo )")
    result = cursor.fetchone()
    if not result:
        return {"average_weight": []}
    return {"average_weight": result[0]}

# Endpoint to get the percentage of teams with games played above a certain threshold in a specific season and game type
@app.get("/v1/ice_hockey_draft/percentage_teams_above_games_played", operation_id="get_percentage_teams_above_games_played", summary="Retrieves the percentage of teams that have played more than a specified number of games in a given season and game type. The calculation is based on the total number of teams that meet the minimum games played threshold, divided by the total number of teams in the specified season and game type.")
async def get_percentage_teams_above_games_played(min_games_played: int = Query(..., description="Minimum number of games played"), season: str = Query(..., description="Season in 'YYYY-YYYY' format"), game_type: str = Query(..., description="Game type")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN GP > ? THEN TEAM ELSE NULL END) AS REAL) * 100 / COUNT(TEAM) FROM SeasonStatus WHERE SEASON = ? AND GAMETYPE = ?", (min_games_played, season, game_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the player name with the highest goals in a specific season and league
@app.get("/v1/ice_hockey_draft/top_goal_scorer_by_season_and_league", operation_id="get_top_goal_scorer_by_season_and_league", summary="Retrieves the name of the player who scored the most goals in a given season and league. The season is specified in 'YYYY-YYYY' format, and the league name is provided as input. The operation returns the top goal scorer's name, based on the provided season and league.")
async def get_top_goal_scorer_by_season_and_league(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), league: str = Query(..., description="League name")):
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.LEAGUE = ? ORDER BY T1.G DESC LIMIT 1", (season, league))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get player names based on team and minimum points
@app.get("/v1/ice_hockey_draft/player_names_by_team_and_min_points", operation_id="get_player_names_by_team_and_min_points", summary="Retrieves the names of players from a specific team who have scored a minimum number of points. The team and minimum points are provided as input parameters.")
async def get_player_names_by_team_and_min_points(team: str = Query(..., description="Team name"), min_points: int = Query(..., description="Minimum points")):
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.TEAM = ? AND T1.P >= ?", (team, min_points))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get player names based on weight in kilograms
@app.get("/v1/ice_hockey_draft/player_names_by_weight", operation_id="get_player_names_by_weight", summary="Retrieves the names of ice hockey players who weigh a specified amount in kilograms. The operation filters players based on their weight and returns a list of matching player names.")
async def get_player_names_by_weight(weight_in_kg: int = Query(..., description="Weight in kilograms")):
    cursor.execute("SELECT T2.PlayerName FROM weight_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.weight_id = T2.weight WHERE T1.weight_in_kg = ?", (weight_in_kg,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get player names and heights based on a reference player's height
@app.get("/v1/ice_hockey_draft/player_names_heights_by_reference_player", operation_id="get_player_names_heights_by_reference_player", summary="Retrieves the names and heights of players who share the same height as the specified reference player. The reference player is identified by their name.")
async def get_player_names_heights_by_reference_player(reference_player_name: str = Query(..., description="Name of the reference player")):
    cursor.execute("SELECT T2.PlayerName, T1.height_in_cm FROM height_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.height_id = T2.height WHERE T2.height = ( SELECT height FROM PlayerInfo WHERE PlayerName = ? )", (reference_player_name,))
    result = cursor.fetchall()
    if not result:
        return {"player_info": []}
    return {"player_info": [{"player_name": row[0], "height_in_cm": row[1]} for row in result]}

# Endpoint to get player names and positions with the maximum penalty minutes
@app.get("/v1/ice_hockey_draft/player_names_positions_max_pim", operation_id="get_player_names_positions_max_pim", summary="Retrieves the names and positions of ice hockey players who have accumulated the highest number of penalty minutes in the current season. The data is sourced from the SeasonStatus and PlayerInfo tables, with the former providing penalty minute data and the latter supplying player information.")
async def get_player_names_positions_max_pim():
    cursor.execute("SELECT T2.PlayerName, T2.position_info FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.PIM = ( SELECT MAX(PIM) FROM SeasonStatus )")
    result = cursor.fetchall()
    if not result:
        return {"player_info": []}
    return {"player_info": [{"player_name": row[0], "position_info": row[1]} for row in result]}

# Endpoint to get player names with the maximum points
@app.get("/v1/ice_hockey_draft/player_names_max_points", operation_id="get_player_names_max_points", summary="Retrieves the names of ice hockey players who have scored the highest number of points in the current season. The data is fetched from the SeasonStatus and PlayerInfo tables, with the maximum points calculated based on the SeasonStatus table.")
async def get_player_names_max_points():
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.P = ( SELECT MAX(P) FROM SeasonStatus )")
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the percentage of players from specific nations drafted by a specific team
@app.get("/v1/ice_hockey_draft/percentage_players_by_nation", operation_id="get_percentage_players_by_nation", summary="Retrieves the percentage of players drafted by a specific team who originate from up to four specified nations. The calculation is based on the total number of players drafted by the team.")
async def get_percentage_players_by_nation(nation1: str = Query(..., description="First nation"), nation2: str = Query(..., description="Second nation"), nation3: str = Query(..., description="Third nation"), nation4: str = Query(..., description="Fourth nation"), overallby: str = Query(..., description="Team that drafted the players")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN nation IN (?, ?, ?, ?) THEN ELITEID ELSE NULL END) AS REAL) * 100 / COUNT(ELITEID) FROM PlayerInfo WHERE overallby = ?", (nation1, nation2, nation3, nation4, overallby))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the top player drafted by a specific team in a specific year
@app.get("/v1/ice_hockey_draft/top_player_by_team_year", operation_id="get_top_player_by_team_year", summary="Retrieves the name of the highest-ranked player drafted by a specific team in a given year. The team and year are provided as input parameters, with the team specified by the 'overallby' parameter and the year by the 'draftyear' parameter in 'YYYY' format. The result is the name of the top-ranked player based on the CSS ranking system.")
async def get_top_player_by_team_year(overallby: str = Query(..., description="Team that drafted the player"), draftyear: str = Query(..., description="Draft year in 'YYYY' format")):
    cursor.execute("SELECT PlayerName FROM PlayerInfo WHERE overallby = ? AND draftyear = ? ORDER BY CSS_rank DESC LIMIT 1", (overallby, draftyear))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the top player by points in a specific season, game type, and league
@app.get("/v1/ice_hockey_draft/top_player_by_season_gametype_league", operation_id="get_top_player_by_season_gametype_league", summary="Retrieves the top-performing player by points in a specified season, game type, and league. The player's name and team are returned. The season should be provided in 'YYYY-YYYY' format, while the game type and league are also required.")
async def get_top_player_by_season_gametype_league(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), gametype: str = Query(..., description="Game type"), league: str = Query(..., description="League")):
    cursor.execute("SELECT T2.PlayerName, T1.TEAM FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.GAMETYPE = ? AND T1.LEAGUE = ? ORDER BY T1.P DESC LIMIT 1", (season, gametype, league))
    result = cursor.fetchone()
    if not result:
        return {"player_info": []}
    return {"player_info": {"player_name": result[0], "team": result[1]}}

# Endpoint to get the count of players drafted by a specific team with more than a specified number of games played in 7 years
@app.get("/v1/ice_hockey_draft/count_players_by_team_games_played", operation_id="get_count_players_by_team_games_played", summary="Retrieves the total number of players drafted by a specific team who have played more than a specified number of games within a 7-year period. This operation provides a quantitative measure of a team's drafting performance based on the players' active participation in games.")
async def get_count_players_by_team_games_played(overallby: str = Query(..., description="Team that drafted the players"), sum_7yr_GP: int = Query(..., description="Number of games played in 7 years")):
    cursor.execute("SELECT COUNT(ELITEID) FROM PlayerInfo WHERE overallby = ? AND sum_7yr_GP > ?", (overallby, sum_7yr_GP))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the height of players based on season, team, and penalty minutes
@app.get("/v1/ice_hockey_draft/player_height_by_season_team_pim", operation_id="get_player_height_by_season_team_pim", summary="Retrieves the height of ice hockey players for a specific season, team, and penalty minutes. The height is returned in centimeters.")
async def get_player_height_by_season_team_pim(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), team: str = Query(..., description="Team"), pim: int = Query(..., description="Penalty minutes")):
    cursor.execute("SELECT T3.height_in_cm FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID INNER JOIN height_info AS T3 ON T2.height = T3.height_id WHERE T1.SEASON = ? AND T1.TEAM = ? AND T1.PIM = ?", (season, team, pim))
    result = cursor.fetchall()
    if not result:
        return {"heights": []}
    return {"heights": [row[0] for row in result]}

# Endpoint to get the percentage of goals scored by a specific player in a specific season and team
@app.get("/v1/ice_hockey_draft/percentage_goals_by_player_season_team", operation_id="get_percentage_goals_by_player_season_team", summary="Retrieves the percentage of total goals scored by a specific player in a given season and team. The calculation is based on the sum of goals scored by the player and the total goals scored by the team in the specified season.")
async def get_percentage_goals_by_player_season_team(player_name: str = Query(..., description="Player name"), season: str = Query(..., description="Season in 'YYYY-YYYY' format"), team: str = Query(..., description="Team")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.PlayerName = ? THEN T1.G ELSE 0 END) AS REAL) * 100 / SUM(T1.G) FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.TEAM = ?", (player_name, season, team))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of penalty minutes for a specific player in a given season and team
@app.get("/v1/ice_hockey_draft/penalty_minutes_percentage", operation_id="get_penalty_minutes_percentage", summary="Retrieves the percentage of penalty minutes a specific player has incurred in a given season and team. The calculation is based on the total penalty minutes (PIM) the player has accumulated compared to the total PIM of all players in the same season and team.")
async def get_penalty_minutes_percentage(player_name: str = Query(..., description="Name of the player"), season: str = Query(..., description="Season in 'YYYY-YYYY' format"), team: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.PlayerName = ? THEN T1.PIM ELSE 0 END) AS REAL) * 100 / SUM(T1.PIM) FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.TEAM = ?", (player_name, season, team))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the birthplace of a specific player
@app.get("/v1/ice_hockey_draft/player_birthplace", operation_id="get_player_birthplace", summary="Retrieves the birthplace of a specific ice hockey player. The operation requires the player's name as input and returns the birthplace information from the player's profile.")
async def get_player_birthplace(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT birthplace FROM PlayerInfo WHERE PlayerName = ?", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"birthplace": []}
    return {"birthplace": result[0]}

# Endpoint to get the names of players with a specific weight in pounds
@app.get("/v1/ice_hockey_draft/players_by_weight_lbs", operation_id="get_players_by_weight_lbs", summary="Retrieve the names of ice hockey players who weigh a specified number of pounds. The operation filters players based on their weight and returns a list of corresponding player names.")
async def get_players_by_weight_lbs(weight_lbs: int = Query(..., description="Weight in pounds")):
    cursor.execute("SELECT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id WHERE T2.weight_in_lbs = ?", (weight_lbs,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get the name of the heaviest player
@app.get("/v1/ice_hockey_draft/heaviest_player", operation_id="get_heaviest_player", summary="Retrieves the name of the heaviest player in the ice hockey draft. The operation compares the weights of all players and returns the name of the player with the highest weight. The weight information is obtained from the weight_info table and linked to the player information using the weight_id.")
async def get_heaviest_player():
    cursor.execute("SELECT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id ORDER BY T2.weight_in_kg DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": result[0]}

# Endpoint to get the percentage of players from a specific nation with weight above a certain value in pounds
@app.get("/v1/ice_hockey_draft/percentage_players_nation_weight", operation_id="get_percentage_players_nation_weight", summary="Retrieves the percentage of players from a specified nation who weigh more than a given value in pounds. This operation calculates the ratio of players from the specified nation with weights above the provided threshold to the total number of players in the database.")
async def get_percentage_players_nation_weight(nation: str = Query(..., description="Nation of the player"), weight_lbs: int = Query(..., description="Weight in pounds")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.nation = ? AND T2.weight_in_lbs > ? THEN T1.ELITEID ELSE NULL END) AS REAL) * 100 / COUNT(T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN weight_info AS T2 ON T1.weight = T2.weight_id", (nation, weight_lbs))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the distinct teams a specific player has played for
@app.get("/v1/ice_hockey_draft/player_teams", operation_id="get_player_teams", summary="Retrieve the unique teams that a particular player has been a part of. The operation requires the player's name as input to filter the results.")
async def get_player_teams(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT DISTINCT T1.TEAM FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.PlayerName = ?", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": [row[0] for row in result]}

# Endpoint to get the distinct seasons a specific player has played in
@app.get("/v1/ice_hockey_draft/player_seasons", operation_id="get_player_seasons", summary="Retrieve the unique seasons in which a specific ice hockey player has participated. The operation filters the seasons based on the provided player's name.")
async def get_player_seasons(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT DISTINCT T1.SEASON FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.PlayerName = ?", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"seasons": []}
    return {"seasons": [row[0] for row in result]}

# Endpoint to get the distinct game types a specific player has played in
@app.get("/v1/ice_hockey_draft/player_game_types", operation_id="get_player_game_types", summary="Retrieve the unique game types in which a specified player has participated. The operation filters the game types based on the provided player's name.")
async def get_player_game_types(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT DISTINCT T1.GAMETYPE FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.PlayerName = ?", (player_name,))
    result = cursor.fetchall()
    if not result:
        return {"game_types": []}
    return {"game_types": [row[0] for row in result]}

# Endpoint to get the distinct nations of players in a specific season
@app.get("/v1/ice_hockey_draft/nations_by_season", operation_id="get_nations_by_season", summary="Retrieve a unique list of nations represented by players in a specified season. The season is provided in the 'YYYY-YYYY' format.")
async def get_nations_by_season(season: str = Query(..., description="Season in 'YYYY-YYYY' format")):
    cursor.execute("SELECT DISTINCT T2.nation FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ?", (season,))
    result = cursor.fetchall()
    if not result:
        return {"nations": []}
    return {"nations": [row[0] for row in result]}

# Endpoint to get the highest points for a specific player
@app.get("/v1/ice_hockey_draft/highest_points_by_player", operation_id="get_highest_points_by_player", summary="Retrieves the highest points scored by a specific player in a season. The player's name is used to filter the results. The data is sourced from the SeasonStatus and PlayerInfo tables, which are joined on the ELITEID field. The results are ordered in descending order by points, and only the top record is returned.")
async def get_highest_points_by_player(player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T1.P FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.PlayerName = ? ORDER BY T1.P DESC LIMIT 1", (player_name,))
    result = cursor.fetchone()
    if not result:
        return {"points": []}
    return {"points": result[0]}

# Endpoint to get the shortest player from a specific nation
@app.get("/v1/ice_hockey_draft/shortest_player_by_nation", operation_id="get_shortest_player_by_nation", summary="Retrieves the name of the shortest ice hockey player from the specified nation. The operation considers the height information from the height_info table and the player details from the PlayerInfo table. The result is sorted by height in ascending order and limited to the shortest player.")
async def get_shortest_player_by_nation(nation: str = Query(..., description="Nation of the player")):
    cursor.execute("SELECT T2.PlayerName FROM height_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.height_id = T2.height WHERE T2.nation = ? ORDER BY T1.height_in_cm ASC LIMIT 1", (nation,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get players with a specific height in inches
@app.get("/v1/ice_hockey_draft/players_by_height_in_inches", operation_id="get_players_by_height_in_inches", summary="Get players with a specific height in inches")
async def get_players_by_height_in_inches(height_in_inch: str = Query(..., description="""Height in inches (format: '5''8"')""")):
    cursor.execute("SELECT T2.PlayerName FROM height_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.height_id = T2.height WHERE T1.height_in_inch = ?", (height_in_inch,))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the count of players taller than a specific height and born in a specific year
@app.get("/v1/ice_hockey_draft/count_players_by_height_and_birthyear", operation_id="get_count_players_by_height_and_birthyear", summary="Retrieves the number of ice hockey draft players who are taller than the specified height (in centimeters) and born in the given year. The response is based on the height and birthdate information of the players.")
async def get_count_players_by_height_and_birthyear(height_in_cm: int = Query(..., description="Height in centimeters"), birth_year: str = Query(..., description="Birth year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.ELITEID) FROM height_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.height_id = T2.height WHERE T1.height_in_cm > ? AND strftime('%Y', T2.birthdate) = ?", (height_in_cm, birth_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of players from a specific nation who are shorter than a specific height
@app.get("/v1/ice_hockey_draft/percentage_players_by_height_and_nation", operation_id="get_percentage_players_by_height_and_nation", summary="Retrieves the percentage of players from a specified nation who are shorter than a given height. The calculation is based on the total count of players from the specified nation and the count of players from the same nation who are shorter than the provided height.")
async def get_percentage_players_by_height_and_nation(height_in_cm: int = Query(..., description="Height in centimeters"), nation: str = Query(..., description="Nation of the player")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.height_in_cm < ? AND T2.nation = ? THEN T2.ELITEID ELSE NULL END) AS REAL) * 100 / COUNT(T2.ELITEID) FROM height_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.height_id = T2.height", (height_in_cm, nation))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the lightest player from a specific nation
@app.get("/v1/ice_hockey_draft/lightest_player_by_nation", operation_id="get_lightest_player_by_nation", summary="Retrieves the name of the lightest ice hockey player from the specified nation. The operation considers the weight information of all players and returns the name of the player with the lowest weight from the requested nation.")
async def get_lightest_player_by_nation(nation: str = Query(..., description="Nation of the player")):
    cursor.execute("SELECT T2.PlayerName FROM weight_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.weight_id = T2.weight WHERE T2.nation = ? ORDER BY T1.weight_in_lbs ASC LIMIT 1", (nation,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the player with the highest penalty minutes in a specific season
@app.get("/v1/ice_hockey_draft/highest_penalty_minutes_by_season", operation_id="get_highest_penalty_minutes_by_season", summary="Retrieves the name of the ice hockey player who accumulated the highest number of penalty minutes during a specified season. The season is provided in 'YYYY-YYYY' format.")
async def get_highest_penalty_minutes_by_season(season: str = Query(..., description="Season in 'YYYY-YYYY' format")):
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? ORDER BY T1.PIM DESC LIMIT 1", (season,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get distinct players from a specific team in a specific season
@app.get("/v1/ice_hockey_draft/distinct_players_by_team_and_season", operation_id="get_distinct_players_by_team_and_season", summary="Retrieves a list of unique player names from a specific team during a particular season. The operation filters the data based on the provided season and team name, ensuring that only distinct player names are returned.")
async def get_distinct_players_by_team_and_season(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), team: str = Query(..., description="Team name")):
    cursor.execute("SELECT DISTINCT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.TEAM = ?", (season, team))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the player with the highest penalty minutes drafted by a specific team in a specific year
@app.get("/v1/ice_hockey_draft/highest_penalty_minutes_by_draft_team_and_year", operation_id="get_highest_penalty_minutes_by_draft_team_and_year", summary="Retrieves the player who received the highest penalty minutes among those drafted by a specific team in a given year. The operation requires the team name and the draft year as input parameters to filter the results.")
async def get_highest_penalty_minutes_by_draft_team_and_year(overallby: str = Query(..., description="Team that drafted the player"), draftyear: int = Query(..., description="Draft year")):
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.overallby = ? AND T2.draftyear = ? ORDER BY T1.PIM DESC LIMIT 1", (overallby, draftyear))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the count of players drafted by a specific team with a specific height
@app.get("/v1/ice_hockey_draft/count_players_by_draft_team_and_height", operation_id="get_count_players_by_draft_team_and_height", summary="Retrieves the total number of players drafted by a specific team who have a certain height, measured in centimeters. This operation provides a count of players based on the team that drafted them and their height, offering insights into the team's drafting preferences and player characteristics.")
async def get_count_players_by_draft_team_and_height(overallby: str = Query(..., description="Team that drafted the player"), height_in_cm: int = Query(..., description="Height in centimeters")):
    cursor.execute("SELECT COUNT(T2.ELITEID) FROM height_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.height_id = T2.height WHERE T2.overallby = ? AND T1.height_in_cm = ?", (overallby, height_in_cm))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct player names based on season, team, and game type
@app.get("/v1/ice_hockey_draft/distinct_player_names_season_team_gametype", operation_id="get_distinct_player_names", summary="Retrieves a list of unique player names who played for a specific team during a given season and game type. The season is specified in 'YYYY-YYYY' format, the team name is provided, and the game type is indicated.")
async def get_distinct_player_names(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), team: str = Query(..., description="Team name"), game_type: str = Query(..., description="Game type")):
    cursor.execute("SELECT DISTINCT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.TEAM = ? AND T1.GAMETYPE = ?", (season, team, game_type))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the top player by points in a specific season and league
@app.get("/v1/ice_hockey_draft/top_player_by_points_season_league", operation_id="get_top_player_by_points", summary="Retrieves the name of the top-performing player based on points scored in a specific season and league. The operation filters data by the provided season and league, then sorts the results by points in descending order to identify the top player.")
async def get_top_player_by_points(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), league: str = Query(..., description="League name")):
    cursor.execute("SELECT DISTINCT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.LEAGUE = ? ORDER BY T1.P DESC LIMIT 1", (season, league))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the count of players with a specific weight and birth year
@app.get("/v1/ice_hockey_draft/count_players_weight_birthyear", operation_id="get_count_players_weight_birthyear", summary="Retrieves the total number of ice hockey players who have a specific weight (in pounds) and were born in a particular year. The response is based on the weight and birth year provided as input parameters.")
async def get_count_players_weight_birthyear(weight_in_lbs: int = Query(..., description="Weight in pounds"), birth_year: str = Query(..., description="Birth year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.ELITEID) FROM weight_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.weight_id = T2.weight WHERE T1.weight_in_lbs = ? AND strftime('%Y', T2.birthdate) = ?", (weight_in_lbs, birth_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top player by games played in a specific season and league
@app.get("/v1/ice_hockey_draft/top_player_by_games_played_season_league", operation_id="get_top_player_by_games_played", summary="Retrieves the name of the player who has played the most games in a specified season and league. The season is denoted by a range of years in 'YYYY-YYYY' format, and the league is identified by its name.")
async def get_top_player_by_games_played(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), league: str = Query(..., description="League name")):
    cursor.execute("SELECT DISTINCT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.LEAGUE = ? ORDER BY T1.GP DESC LIMIT 1", (season, league))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get player names based on season, league, team, and goals scored
@app.get("/v1/ice_hockey_draft/player_names_season_league_team_goals", operation_id="get_player_names_season_league_team_goals", summary="Retrieves the names of ice hockey players who played in a specific season, league, team, and scored a certain number of goals. The response includes a list of player names that meet the provided criteria.")
async def get_player_names_season_league_team_goals(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), league: str = Query(..., description="League name"), team: str = Query(..., description="Team name"), goals: int = Query(..., description="Number of goals scored")):
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.LEAGUE = ? AND T1.TEAM = ? AND T1.G = ?", (season, league, team, goals))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": [row[0] for row in result]}

# Endpoint to get the youngest player based on season, game type, and team
@app.get("/v1/ice_hockey_draft/youngest_player_season_gametype_team", operation_id="get_youngest_player", summary="Retrieves the name of the youngest player for a given season, game type, and team. The operation filters the SeasonStatus table based on the provided season, game type, and team, then joins it with the PlayerInfo table to obtain player details. The result is sorted by birthdate in ascending order, and the name of the youngest player is returned.")
async def get_youngest_player(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), game_type: str = Query(..., description="Game type"), team: str = Query(..., description="Team name")):
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.GAMETYPE = ? AND T1.TEAM = ? ORDER BY T2.birthdate ASC LIMIT 1", (season, game_type, team))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the top player by assists in a specific season, league, and game type
@app.get("/v1/ice_hockey_draft/top_player_by_assists_season_league_gametype", operation_id="get_top_player_by_assists", summary="Retrieves the name of the player with the highest number of assists in a specified season, league, and game type. The season is identified by a range of years, the league by its name, and the game type by its category. The result is determined by ordering the players in descending order of their assist count and selecting the top player.")
async def get_top_player_by_assists(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), league: str = Query(..., description="League name"), game_type: str = Query(..., description="Game type")):
    cursor.execute("SELECT T2.PlayerName FROM SeasonStatus AS T1 INNER JOIN PlayerInfo AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.SEASON = ? AND T1.LEAGUE = ? AND T1.GAMETYPE = ? ORDER BY T1.A DESC LIMIT 1", (season, league, game_type))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0]}

# Endpoint to get the count of teams for players drafted by a specific team, ordered by weight
@app.get("/v1/ice_hockey_draft/count_teams_drafted_by_team_ordered_by_weight", operation_id="get_count_teams_drafted_by_team", summary="Retrieves the total number of teams that have drafted players from a specific team, sorted by the weight of the players in descending order. The input parameter specifies the team for which the count is calculated. The result is limited to the heaviest player.")
async def get_count_teams_drafted_by_team(overallby: str = Query(..., description="Team that drafted the player")):
    cursor.execute("SELECT COUNT(T2.TEAM) FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID INNER JOIN weight_info AS T3 ON T1.weight = T3.weight_id WHERE T1.overallby = ? ORDER BY T3.weight_in_lbs DESC LIMIT 1", (overallby,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average weight of players drafted by a specific team
@app.get("/v1/ice_hockey_draft/average_weight_drafted_by_team", operation_id="get_average_weight_drafted_by_team", summary="Retrieves the average weight of players drafted by a specific ice hockey team. The team is identified by the provided overallby parameter, which is used to filter the data and calculate the average weight.")
async def get_average_weight_drafted_by_team(overallby: str = Query(..., description="Team that drafted the player")):
    cursor.execute("SELECT CAST(SUM(T1.weight_in_lbs) AS REAL) / COUNT(T2.ELITEID) FROM weight_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.weight_id = T2.weight WHERE T2.overallby = ?", (overallby,))
    result = cursor.fetchone()
    if not result:
        return {"average_weight": []}
    return {"average_weight": result[0]}

# Endpoint to get the average height of players in a specific team and game type
@app.get("/v1/ice_hockey_draft/average_height_team_gametype", operation_id="get_average_height_team_gametype", summary="Retrieves the average height of players from a specific team and game type. The calculation is based on the sum of all player heights in centimeters divided by the total number of players in the team for the specified game type.")
async def get_average_height_team_gametype(team: str = Query(..., description="Team name"), game_type: str = Query(..., description="Game type")):
    cursor.execute("SELECT CAST(SUM(T1.height_in_cm) AS REAL) / COUNT(T2.ELITEID) FROM height_info AS T1 INNER JOIN PlayerInfo AS T2 ON T1.height_id = T2.height INNER JOIN SeasonStatus AS T3 ON T2.ELITEID = T3.ELITEID WHERE T3.TEAM = ? AND T3.GAMETYPE = ?", (team, game_type))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the games played by a player in a specific season
@app.get("/v1/ice_hockey_draft/player_games_played", operation_id="get_player_games_played", summary="Get the games played by a player in a specific season")
async def get_player_games_played(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T2.GP FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.SEASON = ? AND T1.PlayerName = ?", (season, player_name))
    result = cursor.fetchone()
    if not result:
        return {"games_played": []}
    return {"games_played": result[0]}

# Endpoint to get players taller than a specific height
@app.get("/v1/ice_hockey_draft/players_taller_than", operation_id="get_players_taller_than", summary="")
async def get_players_taller_than(height_in_inch: str = Query(..., description="""Height in inches in 'X''Y"' format""")):
    cursor.execute("SELECT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN height_info AS T2 ON T1.height = T2.height_id WHERE T2.height_in_inch > ?", (height_in_inch,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [row[0] for row in result]}

# Endpoint to get the team of a player in a specific season
@app.get("/v1/ice_hockey_draft/player_team_in_season", operation_id="get_player_team_in_season", summary="Get the team of a player in a specific season")
async def get_player_team_in_season(season: str = Query(..., description="Season in 'YYYY-YYYY' format"), player_name: str = Query(..., description="Name of the player")):
    cursor.execute("SELECT T2.TEAM FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.SEASON = ? AND T1.PlayerName = ?", (season, player_name))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result[0]}

# Endpoint to get the team with the most players from a specific nation
@app.get("/v1/ice_hockey_draft/team_with_most_players_from_nation", operation_id="get_team_with_most_players_from_nation", summary="Retrieves the team with the highest number of distinct players from a specified nation. The nation is provided as an input parameter.")
async def get_team_with_most_players_from_nation(nation: str = Query(..., description="Nation of the players")):
    cursor.execute("SELECT T.TEAM FROM ( SELECT T2.TEAM, COUNT(DISTINCT T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.nation = ? GROUP BY T2.TEAM ORDER BY COUNT(DISTINCT T1.ELITEID) DESC LIMIT 1 ) AS T", (nation,))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result[0]}

# Endpoint to get the total games played by a player in a specific game type
@app.get("/v1/ice_hockey_draft/total_games_played_by_player", operation_id="get_total_games_played_by_player", summary="Retrieves the cumulative number of games played by a specific player in a designated game type. The operation requires the player's name and the type of game as input parameters.")
async def get_total_games_played_by_player(player_name: str = Query(..., description="Name of the player"), game_type: str = Query(..., description="Type of the game")):
    cursor.execute("SELECT SUM(T2.GP) FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T1.PlayerName = ? AND T2.GAMETYPE = ?", (player_name, game_type))
    result = cursor.fetchone()
    if not result:
        return {"total_games": []}
    return {"total_games": result[0]}

# Endpoint to get the top scorer in a team for a specific game type
@app.get("/v1/ice_hockey_draft/top_scorer_in_team", operation_id="get_top_scorer_in_team", summary="Retrieves the name of the top scorer in a specified team for a given game type. The operation considers the player's total goals scored in the season and returns the player with the highest count. The team and game type are provided as input parameters.")
async def get_top_scorer_in_team(team: str = Query(..., description="Name of the team"), game_type: str = Query(..., description="Type of the game")):
    cursor.execute("SELECT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.TEAM = ? AND T2.GAMETYPE = ? ORDER BY T2.G DESC LIMIT 1", (team, game_type))
    result = cursor.fetchone()
    if not result:
        return {"top_scorer": []}
    return {"top_scorer": result[0]}

# Endpoint to get the nation with the most players in a specific team
@app.get("/v1/ice_hockey_draft/nation_with_most_players_in_team", operation_id="get_nation_with_most_players_in_team", summary="Retrieves the nation with the highest number of players in a specified team. The team is identified by its name. The response provides the nation with the most players in the team.")
async def get_nation_with_most_players_in_team(team: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT T.nation FROM ( SELECT T1.nation, COUNT(T1.ELITEID) FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.TEAM = ? GROUP BY T1.nation ORDER BY COUNT(T1.ELITEID) DESC LIMIT 1 ) AS T", (team,))
    result = cursor.fetchone()
    if not result:
        return {"nation": []}
    return {"nation": result[0]}

# Endpoint to get the player with the most assists in a team for a specific season
@app.get("/v1/ice_hockey_draft/top_assist_player_in_team", operation_id="get_top_assist_player_in_team", summary="Retrieves the name of the player who has recorded the highest number of assists in a specific team during a given season. The team and season are provided as input parameters.")
async def get_top_assist_player_in_team(team: str = Query(..., description="Name of the team"), season: str = Query(..., description="Season in 'YYYY-YYYY' format")):
    cursor.execute("SELECT T1.PlayerName FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.TEAM = ? AND T2.SEASON = ? ORDER BY T2.A DESC LIMIT 1", (team, season))
    result = cursor.fetchone()
    if not result:
        return {"top_assist_player": []}
    return {"top_assist_player": result[0]}

# Endpoint to get the heights of players in a specific team
@app.get("/v1/ice_hockey_draft/player_heights_in_team", operation_id="get_player_heights_in_team", summary="Retrieves the heights of all players in a specified team, expressed in inches. The team is identified by its name.")
async def get_player_heights_in_team(team: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT T3.height_in_inch FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID INNER JOIN height_info AS T3 ON T1.height = T3.height_id WHERE T2.TEAM = ?", (team,))
    result = cursor.fetchall()
    if not result:
        return {"heights": []}
    return {"heights": [row[0] for row in result]}

# Endpoint to get the percentage of penalty minutes by players from a specific nation in a specific league
@app.get("/v1/ice_hockey_draft/percentage_pim_by_nation_league", operation_id="get_percentage_pim_by_nation", summary="Retrieves the percentage of penalty minutes incurred by players from a specified nation in a given league. The calculation is based on the total number of penalty minutes and the total count of players in the league.")
async def get_percentage_pim_by_nation(nation: str = Query(..., description="Nation name"), league: str = Query(..., description="League name")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.nation = ? THEN T2.PIM ELSE NULL END) AS REAL) * 100 / COUNT(*) FROM PlayerInfo AS T1 INNER JOIN SeasonStatus AS T2 ON T1.ELITEID = T2.ELITEID WHERE T2.LEAGUE = ?", (nation, league))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/ice_hockey_draft/player_height_in_inches?player_name=David%20Bornhammar",
    "/v1/ice_hockey_draft/players_above_height?height_in_inch=6%272%22",
    "/v1/ice_hockey_draft/count_players_height_nation?height_in_inch=6%272%22&nation=Sweden",
    "/v1/ice_hockey_draft/tallest_player",
    "/v1/ice_hockey_draft/player_weight_in_kg?player_name=David%20Bornhammar",
    "/v1/ice_hockey_draft/count_players_above_weight?weight_in_kg=90",
    "/v1/ice_hockey_draft/count_players_weight_position?weight_in_kg=90&position_info=D",
    "/v1/ice_hockey_draft/max_7yr_gp_above_weight?weight_in_kg=90",
    "/v1/ice_hockey_draft/weight_max_7yr_toi",
    "/v1/ice_hockey_draft/height_difference?player_name_1=David%20Bornhammar&player_name_2=Pauli%20Levokari",
    "/v1/ice_hockey_draft/count_players_by_weight_shoots?weight_in_kg=90&shoots=R",
    "/v1/ice_hockey_draft/player_names_by_weight_shoots?weight_in_kg=90&shoots=R",
    "/v1/ice_hockey_draft/player_bmi_by_name?player_name=David%20Bornhammar",
    "/v1/ice_hockey_draft/average_height_by_position?position_info=D",
    "/v1/ice_hockey_draft/max_weight_in_lbs",
    "/v1/ice_hockey_draft/count_players_by_height_shoots?height_in_inch=5%277%22&shoots=R",
    "/v1/ice_hockey_draft/player_name_by_pim_and_games?sum_7yr_GP=500",
    "/v1/ice_hockey_draft/tallest_player_by_birthplace?birthplace=Edmonton%2C%20AB%2C%20CAN",
    "/v1/ice_hockey_draft/count_distinct_players_by_overallby_draftyear_team?overallby=Anaheim%20Ducks&draftyear=2008&team=U.S.%20National%20U18%20Team",
    "/v1/ice_hockey_draft/weight_of_player_with_highest_plusminus",
    "/v1/ice_hockey_draft/top_player_by_season_range_and_league?start_season=2004&end_season=2005&league=QMJHL",
    "/v1/ice_hockey_draft/player_names_by_season_game_type_team?season=1998-1999&game_type=Regular%20Season&team=Acadie-Bathurst%20Titan",
    "/v1/ice_hockey_draft/games_played_by_tallest_player",
    "/v1/ice_hockey_draft/youngest_player_by_season_and_league?season=1997-1998&league=OHL",
    "/v1/ice_hockey_draft/count_players_by_games_played_and_shooting_hand?games_played=72&shooting_hand=L",
    "/v1/ice_hockey_draft/goal_difference_regular_playoffs?player_name=Pavel%20Brendl&season=1998-1999&regular_game_type=Regular%20Season&playoff_game_type=Playoffs",
    "/v1/ice_hockey_draft/average_weight_highest_css_rank",
    "/v1/ice_hockey_draft/percentage_teams_above_games_played?min_games_played=20&season=2007-2008&game_type=Playoffs",
    "/v1/ice_hockey_draft/top_goal_scorer_by_season_and_league?season=2007-2008&league=WHL",
    "/v1/ice_hockey_draft/player_names_by_team_and_min_points?team=Chilliwack%20Chiefs&min_points=100",
    "/v1/ice_hockey_draft/player_names_by_weight?weight_in_kg=120",
    "/v1/ice_hockey_draft/player_names_heights_by_reference_player?reference_player_name=Brian%20Gionta",
    "/v1/ice_hockey_draft/player_names_positions_max_pim",
    "/v1/ice_hockey_draft/player_names_max_points",
    "/v1/ice_hockey_draft/percentage_players_by_nation?nation1=Belarus&nation2=Czech%20Rep.&nation3=Slovakia&nation4=Ukraine&overallby=Toronto%20Maple%20Leafs",
    "/v1/ice_hockey_draft/top_player_by_team_year?overallby=Toronto%20Maple%20Leafs&draftyear=2008",
    "/v1/ice_hockey_draft/top_player_by_season_gametype_league?season=2006-2007&gametype=Playoffs&league=SuperElit",
    "/v1/ice_hockey_draft/count_players_by_team_games_played?overallby=Toronto%20Maple%20Leafs&sum_7yr_GP=300",
    "/v1/ice_hockey_draft/player_height_by_season_team_pim?season=2005-2006&team=Yale%20Univ.&pim=28",
    "/v1/ice_hockey_draft/percentage_goals_by_player_season_team?player_name=Ian%20Schultz&season=2007-2008&team=Calgary%20Hitmen",
    "/v1/ice_hockey_draft/penalty_minutes_percentage?player_name=Yevgeni%20Muratov&season=1999-2000&team=Ak%20Bars%20Kazan",
    "/v1/ice_hockey_draft/player_birthplace?player_name=Aaron%20Gagnon",
    "/v1/ice_hockey_draft/players_by_weight_lbs?weight_lbs=190",
    "/v1/ice_hockey_draft/heaviest_player",
    "/v1/ice_hockey_draft/percentage_players_nation_weight?nation=Denmark&weight_lbs=154",
    "/v1/ice_hockey_draft/player_teams?player_name=Andreas%20Jamtin",
    "/v1/ice_hockey_draft/player_seasons?player_name=Niklas%20Eckerblom",
    "/v1/ice_hockey_draft/player_game_types?player_name=Matthias%20Trattnig",
    "/v1/ice_hockey_draft/nations_by_season?season=1997-1998",
    "/v1/ice_hockey_draft/highest_points_by_player?player_name=Per%20Mars",
    "/v1/ice_hockey_draft/shortest_player_by_nation?nation=Italy",
    "/v1/ice_hockey_draft/players_by_height_in_inches?height_in_inch=5%27%278%22",
    "/v1/ice_hockey_draft/count_players_by_height_and_birthyear?height_in_cm=182&birth_year=1982",
    "/v1/ice_hockey_draft/percentage_players_by_height_and_nation?height_in_cm=200&nation=Russia",
    "/v1/ice_hockey_draft/lightest_player_by_nation?nation=USA",
    "/v1/ice_hockey_draft/highest_penalty_minutes_by_season?season=2000-2001",
    "/v1/ice_hockey_draft/distinct_players_by_team_and_season?season=2000-2001&team=Avangard%20Omsk",
    "/v1/ice_hockey_draft/highest_penalty_minutes_by_draft_team_and_year?overallby=Arizona%20Coyotes&draftyear=2000",
    "/v1/ice_hockey_draft/count_players_by_draft_team_and_height?overallby=Arizona%20Coyotes&height_in_cm=195",
    "/v1/ice_hockey_draft/distinct_player_names_season_team_gametype?season=2000-2001&team=Avangard%20Omsk&game_type=Playoffs",
    "/v1/ice_hockey_draft/top_player_by_points_season_league?season=2000-2001&league=International",
    "/v1/ice_hockey_draft/count_players_weight_birthyear?weight_in_lbs=185&birth_year=1980",
    "/v1/ice_hockey_draft/top_player_by_games_played_season_league?season=2000-2001&league=International",
    "/v1/ice_hockey_draft/player_names_season_league_team_goals?season=2000-2001&league=International&team=Czech%20Republic%20(all)&goals=0",
    "/v1/ice_hockey_draft/youngest_player_season_gametype_team?season=2000-2001&game_type=Regular%20Season&team=Avangard%20Omsk",
    "/v1/ice_hockey_draft/top_player_by_assists_season_league_gametype?season=2007-2008&league=OHL&game_type=Regular%20Season",
    "/v1/ice_hockey_draft/count_teams_drafted_by_team_ordered_by_weight?overallby=Arizona%20Coyotes",
    "/v1/ice_hockey_draft/average_weight_drafted_by_team?overallby=Arizona%20Coyotes",
    "/v1/ice_hockey_draft/average_height_team_gametype?team=Acadie-Bathurst%20Titan&game_type=Regular%20Season",
    "/v1/ice_hockey_draft/player_games_played?season=1997-1998&player_name=Pavel%20Patera",
    "/v1/ice_hockey_draft/players_taller_than?height_in_inch=5'9\"",
    "/v1/ice_hockey_draft/player_team_in_season?season=1997-1998&player_name=Niko%20Kapanen",
    "/v1/ice_hockey_draft/team_with_most_players_from_nation?nation=Sweden",
    "/v1/ice_hockey_draft/total_games_played_by_player?player_name=Per%20Mars&game_type=Playoffs",
    "/v1/ice_hockey_draft/top_scorer_in_team?team=Rimouski%20Oceanic&game_type=Playoffs",
    "/v1/ice_hockey_draft/nation_with_most_players_in_team?team=Plymouth%20Whalers",
    "/v1/ice_hockey_draft/top_assist_player_in_team?team=Plymouth%20Whalers&season=1999-2000",
    "/v1/ice_hockey_draft/player_heights_in_team?team=Oshawa%20Generals",
    "/v1/ice_hockey_draft/percentage_pim_by_nation_league?nation=Sweden&league=OHL"
]
