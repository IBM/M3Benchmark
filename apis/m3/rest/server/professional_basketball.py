from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/professional_basketball/professional_basketball.sqlite')
cursor = conn.cursor()

# Endpoint to get the percentage of home games won out of total games won
@app.get("/v1/professional_basketball/home_win_percentage", operation_id="get_home_win_percentage", summary="Retrieves the percentage of home games won by each team out of their total wins. The calculation is based on the number of home games won divided by the total number of games won, multiplied by 100.")
async def get_home_win_percentage():
    cursor.execute("SELECT CAST(homeWon AS REAL) * 100 / won FROM teams")
    result = cursor.fetchall()
    if not result:
        return {"percentage": []}
    return {"percentage": result}

# Endpoint to get team names where the percentage of games lost is greater than a specified value
@app.get("/v1/professional_basketball/teams_lost_percentage", operation_id="get_teams_lost_percentage", summary="Retrieve the names of professional basketball teams that have lost more than the specified percentage of their games. The input parameter determines the minimum percentage of games lost that a team must have to be included in the results.")
async def get_teams_lost_percentage(lost_percentage: float = Query(..., description="Percentage of games lost")):
    cursor.execute("SELECT name FROM teams WHERE CAST(lost AS REAL) * 100 / games > ?", (lost_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": result}

# Endpoint to get team names and wins for a given year where the team won more games than the previous year
@app.get("/v1/professional_basketball/teams_won_more_than_previous_year", operation_id="get_teams_won_more_than_previous_year", summary="Retrieves the names and wins of teams that have improved their performance in the current year compared to the previous year. The operation compares the number of wins for each team in the current year with the corresponding data from the previous year, and returns the teams that have surpassed their previous year's wins.")
async def get_teams_won_more_than_previous_year(previous_year: int = Query(..., description="Previous year"), current_year: int = Query(..., description="Current year")):
    cursor.execute("SELECT T1.name, T1.won FROM teams AS T1 INNER JOIN ( SELECT * FROM teams WHERE year = ? ) AS T2 on T1.tmID = T2.tmID WHERE T1.year = ? and T1.won > T2.won", (previous_year, current_year))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": result}

# Endpoint to get team names and offensive points where the home win-loss difference percentage is greater than a specified value
@app.get("/v1/professional_basketball/teams_home_win_loss_diff_percentage", operation_id="get_teams_home_win_loss_diff_percentage", summary="Retrieve the names and offensive points of professional basketball teams with a home win-loss difference percentage surpassing the provided threshold.")
async def get_teams_home_win_loss_diff_percentage(win_loss_diff_percentage: float = Query(..., description="Home win-loss difference percentage")):
    cursor.execute("SELECT name, o_pts FROM teams WHERE CAST((homeWon - homeLost) AS REAL) * 100 / games > ?", (win_loss_diff_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": result}

# Endpoint to get the percentage of teams that ranked first
@app.get("/v1/professional_basketball/percentage_teams_ranked_first", operation_id="get_percentage_teams_ranked_first", summary="Retrieves the percentage of professional basketball teams that have ranked first in their respective leagues. This operation calculates the total number of teams that have achieved a first-place ranking and divides it by the total number of teams in the database. The result is then multiplied by 100 to obtain the percentage.")
async def get_percentage_teams_ranked_first():
    cursor.execute("SELECT CAST(SUM(CASE WHEN rank = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(name) FROM teams")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get coach details for a specific year and team
@app.get("/v1/professional_basketball/coach_details_year_team", operation_id="get_coach_details_year_team", summary="Retrieves the details of a coach for a given year and team, including the number of games won and lost. The year and team ID are used to filter the results.")
async def get_coach_details_year_team(year: int = Query(..., description="Year"), tmID: str = Query(..., description="Team ID")):
    cursor.execute("SELECT coachID, won, lost FROM coaches WHERE year = ? AND tmID = ?", (year, tmID))
    result = cursor.fetchall()
    if not result:
        return {"coaches": []}
    return {"coaches": result}

# Endpoint to get the coach with the longest stint within a specified year range
@app.get("/v1/professional_basketball/longest_stint_coach_year_range", operation_id="get_longest_stint_coach_year_range", summary="Get the coach with the longest stint within a specified year range")
async def get_longest_stint_coach_year_range(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT coachID, tmID FROM coaches WHERE year BETWEEN ? AND ? ORDER BY stint DESC LIMIT 1", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"coach": []}
    return {"coach": result}

# Endpoint to get coach details and team wins for a specific year where the team won more than a specified number of games
@app.get("/v1/professional_basketball/coach_team_wins_year", operation_id="get_coach_team_wins_year", summary="Retrieves the coach's ID, name, and the number of games won by their team in a specific year, given that the team won more than a specified minimum number of games.")
async def get_coach_team_wins_year(year: int = Query(..., description="Year"), min_wins: int = Query(..., description="Minimum number of wins")):
    cursor.execute("SELECT T1.coachID, T2.name, T2.won FROM coaches AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? AND T2.won > ?", (year, min_wins))
    result = cursor.fetchall()
    if not result:
        return {"coaches": []}
    return {"coaches": result}

# Endpoint to get distinct coach and team details for a specified year range where the team lost more games than they won
@app.get("/v1/professional_basketball/coach_team_lost_more_year_range", operation_id="get_coach_team_lost_more_year_range", summary="Retrieve unique coach and team combinations for a given year range, where the team lost more games than they won. The operation requires a start and end year to filter the results.")
async def get_coach_team_lost_more_year_range(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT DISTINCT T1.coachID, T2.tmID, T1.year FROM coaches AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year BETWEEN ? AND ? AND T2.lost > T2.won", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"coaches": []}
    return {"coaches": result}

# Endpoint to get team details for a specific coach within a specified year range
@app.get("/v1/professional_basketball/team_details_coach_year_range", operation_id="get_team_details_coach_year_range", summary="Retrieves team details for a specific coach within a specified year range. The operation returns the team name, the year, and the average offensive points per game (o_pts) for the team during the specified year range. The coach is identified by the provided coachID, and the year range is defined by the start_year and end_year parameters.")
async def get_team_details_coach_year_range(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year"), coachID: str = Query(..., description="Coach ID")):
    cursor.execute("SELECT T2.name, T1.year, T2.o_pts FROM coaches AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year BETWEEN ? AND ? AND T1.coachID = ?", (start_year, end_year, coachID))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": result}

# Endpoint to get distinct coach details based on team win percentage
@app.get("/v1/professional_basketball/coach_details_by_win_percentage", operation_id="get_coach_details_by_win_percentage", summary="Retrieves unique coach details for teams with a win percentage surpassing the provided threshold. The response includes the coach's name, the year, and the coach's ID. The win percentage is calculated as the ratio of games won to total games played, multiplied by 100.")
async def get_coach_details_by_win_percentage(win_percentage: float = Query(..., description="Win percentage threshold")):
    cursor.execute("SELECT DISTINCT T2.name, T1.year, T1.coachID FROM coaches AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID WHERE CAST(T2.won AS REAL) * 100 / T2.games > ?", (win_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"coach_details": []}
    return {"coach_details": result}

# Endpoint to get the count of distinct players based on birth state, award, and year range
@app.get("/v1/professional_basketball/player_count_by_birth_state_award_year_range", operation_id="get_player_count_by_birth_state_award_year_range", summary="Retrieves the number of unique professional basketball players who were born in a specific state, received a particular award, and won it within a specified year range.")
async def get_player_count_by_birth_state_award_year_range(birth_state: str = Query(..., description="Birth state of the player"), award: str = Query(..., description="Award received by the player"), start_year: int = Query(..., description="Start year of the award"), end_year: int = Query(..., description="End year of the award")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T1.birthState = ? AND T2.award = ? AND T2.year BETWEEN ? AND ?", (birth_state, award, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct division IDs based on award year, award, and team year
@app.get("/v1/professional_basketball/division_ids_by_award_year_team_year", operation_id="get_division_ids_by_award_year_team_year", summary="Retrieve a unique set of division IDs associated with a specific award year, award type, and team year. This operation filters the awards given to coaches and their respective teams, returning only the distinct division IDs that meet the specified criteria.")
async def get_division_ids_by_award_year_team_year(award_year: int = Query(..., description="Year the award was given"), award: str = Query(..., description="Award received by the coach"), team_year: int = Query(..., description="Year of the team")):
    cursor.execute("SELECT DISTINCT T3.divID FROM awards_coaches AS T1 INNER JOIN coaches AS T2 ON T1.coachID = T2.coachID INNER JOIN teams AS T3 ON T2.tmID = T3.tmID WHERE T1.year = ? AND T1.award = ? AND T3.year = ?", (award_year, award, team_year))
    result = cursor.fetchall()
    if not result:
        return {"division_ids": []}
    return {"division_ids": result}

# Endpoint to get distinct coach IDs based on award, year range, team year, and team name
@app.get("/v1/professional_basketball/coach_ids_by_award_year_range_team_year_name", operation_id="get_coach_ids_by_award_year_range_team_year_name", summary="Retrieve a unique set of coach identifiers based on a specific award, a range of years for that award, a particular team year, and the team's name. This operation filters coaches by the provided award, award years, team year, and team name, and returns the distinct coach IDs that match the criteria.")
async def get_coach_ids_by_award_year_range_team_year_name(award: str = Query(..., description="Award received by the coach"), start_year: int = Query(..., description="Start year of the award"), end_year: int = Query(..., description="End year of the award"), team_year: int = Query(..., description="Year of the team"), team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT DISTINCT T2.coachID FROM coaches AS T1 INNER JOIN awards_coaches AS T2 ON T1.coachID = T2.coachID INNER JOIN teams AS T3 ON T3.tmID = T1.tmID WHERE T2.award = ? AND T2.year BETWEEN ? AND ? AND T1.year = ? AND T3.name = ?", (award, start_year, end_year, team_year, team_name))
    result = cursor.fetchall()
    if not result:
        return {"coach_ids": []}
    return {"coach_ids": result}

# Endpoint to get player nicknames based on blocks, conference, and season ID
@app.get("/v1/professional_basketball/player_nicknames_by_blocks_conference_season", operation_id="get_player_nicknames_by_blocks_conference_season", summary="Retrieves the nicknames of professional basketball players who have achieved a specific number of blocks in a given conference and season. The input parameters determine the number of blocks, the conference, and the season ID.")
async def get_player_nicknames_by_blocks_conference_season(blocks: int = Query(..., description="Number of blocks"), conference: str = Query(..., description="Conference of the player"), season_id: int = Query(..., description="Season ID")):
    cursor.execute("SELECT T2.nameNick FROM player_allstar AS T1 INNER JOIN players AS T2 ON T1.playerID = T2.playerID WHERE T1.blocks = ? AND T1.conference = ? AND T1.season_id = ?", (blocks, conference, season_id))
    result = cursor.fetchall()
    if not result:
        return {"nicknames": []}
    return {"nicknames": result}

# Endpoint to get the year of the winning team based on round and losing team ID
@app.get("/v1/professional_basketball/winning_team_year_by_round_loser", operation_id="get_winning_team_year_by_round_loser", summary="Retrieve the year of the highest-ranked winning team from a specific round, given the ID of the losing team. The response is based on the series post data and team rankings.")
async def get_winning_team_year_by_round_loser(round: str = Query(..., description="Round of the series"), tmIDLoser: str = Query(..., description="ID of the losing team")):
    cursor.execute("SELECT T2.year FROM series_post AS T1 INNER JOIN teams AS T2 ON T1.tmIDWinner = T2.tmID WHERE T1.round = ? AND T1.tmIDLoser = ? ORDER BY T2.rank ASC LIMIT 1", (round, tmIDLoser))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the birth date of the player with the most assists in a given season
@app.get("/v1/professional_basketball/player_birthdate_by_most_assists_season", operation_id="get_player_birthdate_by_most_assists_season", summary="Retrieves the birth date of the player who recorded the highest number of assists in the specified season. The season is identified by its unique ID.")
async def get_player_birthdate_by_most_assists_season(season_id: int = Query(..., description="Season ID")):
    cursor.execute("SELECT T1.birthDate FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T2.season_id = ? ORDER BY T2.assists DESC LIMIT 1", (season_id,))
    result = cursor.fetchone()
    if not result:
        return {"birthdate": []}
    return {"birthdate": result[0]}

# Endpoint to get distinct player names based on birth city, season ID range, and minutes played
@app.get("/v1/professional_basketball/player_names_by_birth_city_season_range_minutes", operation_id="get_player_names_by_birth_city_season_range_minutes", summary="Retrieve a unique list of player names who were born in a specific city and played in a specified season range, with a given number of minutes.")
async def get_player_names_by_birth_city_season_range_minutes(birth_city: str = Query(..., description="Birth city of the player"), start_season: int = Query(..., description="Start season ID"), end_season: int = Query(..., description="End season ID"), minutes: int = Query(..., description="Minutes played")):
    cursor.execute("SELECT DISTINCT T1.firstName, T1.middleName, T1.lastName FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.birthCity = ? AND T2.season_id BETWEEN ? AND ? AND T2.minutes = ?", (birth_city, start_season, end_season, minutes))
    result = cursor.fetchall()
    if not result:
        return {"player_names": []}
    return {"player_names": result}

# Endpoint to get the count of distinct players based on conference, minutes played, and college
@app.get("/v1/professional_basketball/player_count_by_conference_minutes_college", operation_id="get_player_count_by_conference_minutes_college", summary="Retrieves the number of unique players who played in a specific conference, played less than or equal to a certain number of minutes, and attended a particular college.")
async def get_player_count_by_conference_minutes_college(conference: str = Query(..., description="Conference of the player"), minutes: int = Query(..., description="Minutes played"), college: str = Query(..., description="College of the player")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T2.conference = ? AND T2.minutes <= ? AND T1.college = ?", (conference, minutes, college))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct players based on defensive rebound percentage and season ID range
@app.get("/v1/professional_basketball/player_count_by_defensive_rebound_percentage_season_range", operation_id="get_player_count_by_defensive_rebound_percentage_season_range", summary="Retrieves the number of unique players who have a defensive rebound percentage greater than the specified value within a given season range. The defensive rebound percentage is calculated as the ratio of defensive rebounds to total rebounds, multiplied by 100.")
async def get_player_count_by_defensive_rebound_percentage_season_range(defensive_rebound_percentage: float = Query(..., description="Defensive rebound percentage"), start_season: int = Query(..., description="Start season ID"), end_season: int = Query(..., description="End season ID")):
    cursor.execute("SELECT COUNT(DISTINCT playerID) FROM player_allstar WHERE CAST(d_rebounds AS REAL) * 100 / rebounds > ? AND season_id BETWEEN ? AND ?", (defensive_rebound_percentage, start_season, end_season))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the year of the best win-loss ratio for a specific coach
@app.get("/v1/professional_basketball/best_year_by_coach", operation_id="get_best_year_by_coach", summary="Retrieves the year in which a specific coach achieved their highest win-loss ratio in professional basketball. The coach is identified by their unique ID, and the year returned corresponds to the season with the highest ratio of games won to total games played.")
async def get_best_year_by_coach(coachID: str = Query(..., description="ID of the coach")):
    cursor.execute("SELECT year FROM coaches WHERE coachID = ? ORDER BY CAST(won AS REAL) / (won + lost) DESC LIMIT 1", (coachID,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the total minutes played by players with a specific birth city and nickname
@app.get("/v1/professional_basketball/total_minutes_by_city_nickname", operation_id="get_total_minutes_by_city_nickname", summary="Retrieves the cumulative minutes played by professional basketball players born in a specific city and having a particular nickname. The input parameters allow for filtering by birth city and nickname, with the latter supporting wildcard searches.")
async def get_total_minutes_by_city_nickname(birthCity: str = Query(..., description="Birth city of the player"), nameNick: str = Query(..., description="Nickname of the player (use %% for wildcard)")):
    cursor.execute("SELECT SUM(T2.minutes) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.birthCity = ? AND T1.nameNick LIKE ?", (birthCity, nameNick))
    result = cursor.fetchone()
    if not result:
        return {"total_minutes": []}
    return {"total_minutes": result[0]}

# Endpoint to get the team with the most home losses in a specific year for a specific award
@app.get("/v1/professional_basketball/team_most_home_losses_by_year_award", operation_id="get_team_most_home_losses_by_year_award", summary="Retrieves the team with the highest number of home losses in a given year for a specific award. The operation considers the coaches who have received the specified award in the provided year and identifies the team they were associated with. The team with the most home losses is then determined and returned.")
async def get_team_most_home_losses_by_year_award(year: int = Query(..., description="Year of the award"), award: str = Query(..., description="Name of the award")):
    cursor.execute("SELECT T3.tmID FROM awards_coaches AS T1 INNER JOIN coaches AS T2 ON T1.coachID = T2.coachID INNER JOIN teams AS T3 ON T3.tmID = T2.tmID WHERE T1.year = ? AND T1.award = ? GROUP BY T3.tmID ORDER BY SUM(T3.homeLost) DESC LIMIT 1", (year, award))
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get distinct team IDs based on specific criteria
@app.get("/v1/professional_basketball/distinct_team_ids_by_criteria", operation_id="get_distinct_team_ids_by_criteria", summary="Retrieve a unique list of team IDs that meet specific criteria. The criteria include a minimum number of wins, a specific year and round of the series, and the ID of the losing team. This operation is useful for identifying teams that have achieved a certain level of success under particular conditions.")
async def get_distinct_team_ids_by_criteria(won: int = Query(..., description="Number of wins"), year: int = Query(..., description="Year of the series"), round: str = Query(..., description="Round of the series"), tmIDLoser: str = Query(..., description="ID of the losing team")):
    cursor.execute("SELECT DISTINCT T2.tmID FROM series_post AS T1 INNER JOIN teams AS T2 ON T1.tmIDWinner = T2.tmID WHERE T2.won > ? AND T1.year = ? AND T1.round = ? AND T1.tmIDLoser = ?", (won, year, round, tmIDLoser))
    result = cursor.fetchall()
    if not result:
        return {"team_ids": []}
    return {"team_ids": [row[0] for row in result]}

# Endpoint to get league IDs based on player weight criteria
@app.get("/v1/professional_basketball/league_ids_by_weight_criteria", operation_id="get_league_ids_by_weight_criteria", summary="Retrieves the IDs of basketball leagues where the heaviest players, as determined by a given weight factor, are playing. The weight factor is a decimal value that represents the percentage of the maximum player weight to be considered. For example, a weight factor of 0.4 will return the IDs of leagues where the heaviest players weigh at least 60% of the maximum player weight.")
async def get_league_ids_by_weight_criteria(weight_factor: float = Query(..., description="Weight factor (e.g., 0.4)")):
    cursor.execute("SELECT T2.lgID FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID GROUP BY T2.lgID, T1.weight HAVING T1.weight = MAX(T1.weight) - MAX(T1.weight) * ?", (weight_factor,))
    result = cursor.fetchall()
    if not result:
        return {"league_ids": []}
    return {"league_ids": [row[0] for row in result]}

# Endpoint to get coach IDs who have coached more than a certain number of teams
@app.get("/v1/professional_basketball/coaches_by_team_count", operation_id="get_coaches_by_team_count", summary="Retrieves the unique identifiers of professional basketball coaches who have coached more than the specified number of teams. This operation allows you to identify coaches with extensive experience across multiple teams.")
async def get_coaches_by_team_count(team_count: int = Query(..., description="Number of teams coached")):
    cursor.execute("SELECT coachID FROM coaches GROUP BY coachID HAVING COUNT(DISTINCT tmID) > ?", (team_count,))
    result = cursor.fetchall()
    if not result:
        return {"coach_ids": []}
    return {"coach_ids": [row[0] for row in result]}

# Endpoint to get the coach with the most post-season wins
@app.get("/v1/professional_basketball/top_coach_by_post_wins", operation_id="get_top_coach_by_post_wins", summary="Retrieves the unique identifier of the coach with the highest number of post-season wins. This operation does not require any input parameters and returns the coachID of the top-performing coach in terms of post-season victories.")
async def get_top_coach_by_post_wins():
    cursor.execute("SELECT coachID FROM coaches ORDER BY post_wins DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"coach_id": []}
    return {"coach_id": result[0]}

# Endpoint to get the coach with the fewest post-season losses in a specific league
@app.get("/v1/professional_basketball/coach_fewest_post_losses", operation_id="get_coach_fewest_post_losses", summary="Retrieve the coach with the fewest post-season losses in a specific league, who has coached more than a given number of teams in that league. The league is identified by its unique ID, and the minimum number of teams coached is provided as input.")
async def get_coach_fewest_post_losses(lgID: str = Query(..., description="League ID"), team_count: int = Query(..., description="Number of teams coached")):
    cursor.execute("SELECT coachID FROM coaches WHERE lgID = ? AND post_wins != 0 AND post_losses != 0 AND coachID IN ( SELECT coachID FROM coaches WHERE lgID = ? GROUP BY coachID HAVING COUNT(tmID) > ? ) ORDER BY post_losses ASC LIMIT 1", (lgID, lgID, team_count))
    result = cursor.fetchone()
    if not result:
        return {"coach_id": []}
    return {"coach_id": result[0]}

# Endpoint to get the count of players in a specific league and position
@app.get("/v1/professional_basketball/player_count_by_league_position", operation_id="get_player_count_by_league_position", summary="Retrieves the total number of unique players in a specific league who play either of the two provided positions. The league is identified by its unique ID, and the positions are specified by their respective codes.")
async def get_player_count_by_league_position(lgID: str = Query(..., description="League ID"), pos1: str = Query(..., description="First position"), pos2: str = Query(..., description="Second position")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.lgID = ? AND (T1.pos = ? OR T1.pos = ?)", (lgID, pos1, pos2))
    result = cursor.fetchone()
    if not result:
        return {"player_count": []}
    return {"player_count": result[0]}

# Endpoint to get distinct first names of players in a specific league and position
@app.get("/v1/professional_basketball/distinct_first_names_by_league_position", operation_id="get_distinct_first_names_by_league_position", summary="Retrieve a unique list of first names of professional basketball players who play in the specified league and either of the two provided positions.")
async def get_distinct_first_names_by_league_position(pos1: str = Query(..., description="First position"), pos2: str = Query(..., description="Second position"), lgID: str = Query(..., description="League ID")):
    cursor.execute("SELECT DISTINCT T1.firstName FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE (T1.pos = ? OR T1.pos = ?) AND T2.lgID = ?", (pos1, pos2, lgID))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the count of distinct players from a specific high school city and conference
@app.get("/v1/professional_basketball/count_players_hs_city_conference", operation_id="get_count_players_hs_city_conference", summary="Retrieve the number of unique professional basketball players who attended high school in a specific city and played in a particular conference.")
async def get_count_players_hs_city_conference(hs_city: str = Query(..., description="High school city of the player"), conference: str = Query(..., description="Conference of the player")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.hsCity = ? AND T2.conference = ?", (hs_city, conference))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct colleges of players from a specific conference
@app.get("/v1/professional_basketball/distinct_colleges_conference", operation_id="get_distinct_colleges_conference", summary="Retrieves a unique list of colleges that have players in a specified conference. The response includes the names of colleges whose players have been selected for the all-star team in the given conference.")
async def get_distinct_colleges_conference(conference: str = Query(..., description="Conference of the player")):
    cursor.execute("SELECT DISTINCT T1.college FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T2.conference = ?", (conference,))
    result = cursor.fetchall()
    if not result:
        return {"colleges": []}
    return {"colleges": [row[0] for row in result]}

# Endpoint to get the count of distinct players from a specific birth city and league
@app.get("/v1/professional_basketball/count_players_birth_city_league", operation_id="get_count_players_birth_city_league", summary="Retrieve the number of unique professional basketball players born in a specific city and playing in a given league. The operation considers players who have played for at least one team in the specified league.")
async def get_count_players_birth_city_league(birth_city: str = Query(..., description="Birth city of the player"), lgID: str = Query(..., description="League ID of the player")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T1.birthCity = ? AND T2.lgID = ?", (birth_city, lgID))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the birth date of the player with the most awards
@app.get("/v1/professional_basketball/birth_date_most_awards", operation_id="get_birth_date_most_awards", summary="Retrieves the birth date of the professional basketball player who has won the specified award the most times. The award name is provided as an input parameter.")
async def get_birth_date_most_awards(award: str = Query(..., description="Award name")):
    cursor.execute("SELECT T1.birthDate FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ? GROUP BY T1.playerID, T1.birthDate ORDER BY COUNT(award) DESC LIMIT 1", (award,))
    result = cursor.fetchone()
    if not result:
        return {"birth_date": []}
    return {"birth_date": result[0]}

# Endpoint to get the count of distinct players with a specific award and birth city
@app.get("/v1/professional_basketball/count_players_award_birth_city", operation_id="get_count_players_award_birth_city", summary="Retrieve the number of unique professional basketball players who have received a specific award and were born in a particular city.")
async def get_count_players_award_birth_city(award: str = Query(..., description="Award name"), birth_city: str = Query(..., description="Birth city of the player")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ? AND T1.birthCity = ?", (award, birth_city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the height of the tallest player with a specific award
@app.get("/v1/professional_basketball/tallest_player_award", operation_id="get_tallest_player_award", summary="Retrieves the height of the tallest professional basketball player who has received a specific award. The award name must be provided as an input parameter.")
async def get_tallest_player_award(award: str = Query(..., description="Award name")):
    cursor.execute("SELECT T1.height FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ? ORDER BY T1.height DESC LIMIT 1", (award,))
    result = cursor.fetchone()
    if not result:
        return {"height": []}
    return {"height": result[0]}

# Endpoint to get the average weight of players with a specific award
@app.get("/v1/professional_basketball/avg_weight_award", operation_id="get_avg_weight_award", summary="Retrieves the average weight of professional basketball players who have received a specific award. The award is specified as an input parameter.")
async def get_avg_weight_award(award: str = Query(..., description="Award name")):
    cursor.execute("SELECT AVG(T1.weight) FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ?", (award,))
    result = cursor.fetchone()
    if not result:
        return {"avg_weight": []}
    return {"avg_weight": result[0]}

# Endpoint to get the average height of players with a specific award and birth city
@app.get("/v1/professional_basketball/avg_height_award_birth_city", operation_id="get_avg_height_award_birth_city", summary="Retrieves the average height of professional basketball players who have received a specific award and were born in a particular city.")
async def get_avg_height_award_birth_city(award: str = Query(..., description="Award name"), birth_city: str = Query(..., description="Birth city of the player")):
    cursor.execute("SELECT AVG(T1.height) FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ? AND T1.birthCity = ?", (award, birth_city))
    result = cursor.fetchone()
    if not result:
        return {"avg_height": []}
    return {"avg_height": result[0]}

# Endpoint to get the top 10 teams by total post points in a specific year
@app.get("/v1/professional_basketball/top_teams_by_post_points", operation_id="get_top_teams_by_post_points", summary="Retrieves the top 10 professional basketball teams with the highest total post points in a specified year. The teams are ranked based on the sum of their post points, providing a clear comparison of their performance in the post season.")
async def get_top_teams_by_post_points(year: int = Query(..., description="Year")):
    cursor.execute("SELECT tmID FROM players_teams WHERE year = ? GROUP BY tmID ORDER BY SUM(PostPoints) DESC LIMIT 10", (year,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": [row[0] for row in result]}

# Endpoint to get the names of teams with a win percentage below a specific threshold
@app.get("/v1/professional_basketball/teams_below_win_percentage", operation_id="get_teams_below_win_percentage", summary="Retrieve the names of professional basketball teams that have a win percentage lower than a specified threshold. The threshold is expressed as a whole number representing a percentage (e.g., 50 for 50%).")
async def get_teams_below_win_percentage(win_percentage: float = Query(..., description="Win percentage threshold (e.g., 50 for 50%)")):
    cursor.execute("SELECT name FROM teams WHERE CAST(won AS REAL) * 100 / (won + lost) < ?", (win_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": [row[0] for row in result]}

# Endpoint to get coach IDs with a winning percentage greater than a specified value
@app.get("/v1/professional_basketball/coach_ids_winning_percentage", operation_id="get_coach_ids_winning_percentage", summary="Retrieves the IDs of professional basketball coaches who have a winning percentage greater than a specified threshold. The winning percentage is calculated as the ratio of games won to the total number of games played.")
async def get_coach_ids_winning_percentage(winning_percentage: float = Query(..., description="Winning percentage threshold")):
    cursor.execute("SELECT coachID FROM coaches GROUP BY tmID, coachID, won, lost HAVING CAST(won AS REAL) * 100 / (won + lost) > ?", (winning_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"coach_ids": []}
    return {"coach_ids": [row[0] for row in result]}

# Endpoint to get coach IDs with a coaching tenure longer than a specified number of years in a given league
@app.get("/v1/professional_basketball/coach_ids_tenure_league", operation_id="get_coach_ids_tenure_league", summary="Retrieve the IDs of professional basketball coaches who have coached in a specific league for more than a certain number of years. The league is identified by its unique ID, and the minimum tenure duration is specified in years.")
async def get_coach_ids_tenure_league(lgID: str = Query(..., description="League ID"), tenure_years: int = Query(..., description="Minimum tenure years")):
    cursor.execute("SELECT coachID FROM coaches WHERE lgID = ? GROUP BY coachID HAVING MAX(year) - MIN(year) > ?", (lgID, tenure_years))
    result = cursor.fetchall()
    if not result:
        return {"coach_ids": []}
    return {"coach_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct team names with a specified award and minimum points
@app.get("/v1/professional_basketball/count_teams_award_points", operation_id="get_count_teams_award_points", summary="Retrieves the number of unique teams that have received a specific award and have accumulated at least a certain number of points. The award and minimum points are provided as input parameters.")
async def get_count_teams_award_points(award: str = Query(..., description="Award name"), min_points: int = Query(..., description="Minimum points")):
    cursor.execute("SELECT COUNT(DISTINCT T4.name) FROM ( SELECT T1.name, SUM(T2.points) FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year INNER JOIN awards_players AS T3 ON T2.playerID = T3.playerID WHERE T3.award = ? GROUP BY T1.name HAVING SUM(T2.points) >= ? ) AS T4", (award, min_points))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the tallest player in a specified team
@app.get("/v1/professional_basketball/tallest_player_team", operation_id="get_tallest_player_team", summary="Retrieves the full name of the tallest player in the specified team. The team is identified by its unique ID. The response includes the first, middle, and last names of the player.")
async def get_tallest_player_team(tmID: str = Query(..., description="Team ID")):
    cursor.execute("SELECT T1.firstName, T1.middleName, T1.lastName FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.tmID = ? ORDER BY T1.height DESC LIMIT 1", (tmID,))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": {"firstName": result[0], "middleName": result[1], "lastName": result[2]}}

# Endpoint to get the last names of players in a specified team
@app.get("/v1/professional_basketball/player_last_names_team", operation_id="get_player_last_names_team", summary="Retrieves the last names of all players associated with a specific team, identified by its unique team ID.")
async def get_player_last_names_team(tmID: str = Query(..., description="Team ID")):
    cursor.execute("SELECT T1.lastName FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.tmID = ?", (tmID,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get the count of distinct coach IDs within a specified year range
@app.get("/v1/professional_basketball/count_coach_ids_year_range", operation_id="get_count_coach_ids_year_range", summary="Retrieves the total number of unique coaches who have received awards within a specified year range. The year range is defined by the start_year and end_year parameters.")
async def get_count_coach_ids_year_range(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT COUNT(DISTINCT coachID) FROM awards_coaches WHERE year BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get coach IDs with a specified number of awards within a given year range
@app.get("/v1/professional_basketball/coach_ids_awards_year_range", operation_id="get_coach_ids_awards_year_range", summary="Retrieves the unique identifiers of professional basketball coaches who have received a specific number of awards within a defined year range. The operation filters the awards data by the provided start and end years, and groups the results by coach and award. It then returns the coach IDs that have the exact count of awards as specified.")
async def get_coach_ids_awards_year_range(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year"), award_count: int = Query(..., description="Number of awards")):
    cursor.execute("SELECT coachID FROM awards_coaches WHERE year BETWEEN ? AND ? GROUP BY coachID, award HAVING COUNT(award) = ?", (start_year, end_year, award_count))
    result = cursor.fetchall()
    if not result:
        return {"coach_ids": []}
    return {"coach_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct coach IDs with specific awards within a given year range
@app.get("/v1/professional_basketball/count_coach_ids_specific_awards_year_range", operation_id="get_count_coach_ids_specific_awards_year_range", summary="Retrieve the number of unique coaches who have received a specific award within a defined year range. The year range is specified by the start and end years, and the award is identified by the first and second award parameters. The count includes only those coaches who have received the same award within the specified year range.")
async def get_count_coach_ids_specific_awards_year_range(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year"), award1: str = Query(..., description="First award"), award2: str = Query(..., description="Second award")):
    cursor.execute("SELECT COUNT(DISTINCT coachID) FROM awards_coaches WHERE year BETWEEN ? AND ? AND award = ? AND coachID IN ( SELECT coachID FROM awards_coaches WHERE year BETWEEN ? AND ? AND award = ? )", (start_year, end_year, award1, start_year, end_year, award2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average points of players in a specified season
@app.get("/v1/professional_basketball/avg_points_season", operation_id="get_avg_points_season", summary="Retrieves the average points scored by players in a specific season. The season is identified by its unique ID. The calculation is based on the aggregated points of all players who participated in the given season.")
async def get_avg_points_season(season_id: int = Query(..., description="Season ID")):
    cursor.execute("SELECT AVG(T2.points) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T2.season_id = ?", (season_id,))
    result = cursor.fetchone()
    if not result:
        return {"avg_points": []}
    return {"avg_points": result[0]}

# Endpoint to get distinct player names with a height greater than a specified value
@app.get("/v1/professional_basketball/player_names_height", operation_id="get_player_names_height", summary="Retrieves a list of unique professional basketball player names with a height greater than the specified minimum height. The response includes the last and first names of the players.")
async def get_player_names_height(min_height: int = Query(..., description="Minimum height")):
    cursor.execute("SELECT DISTINCT T1.lastName, T1.firstName FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.height > ?", (min_height,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [{"lastName": row[0], "firstName": row[1]} for row in result]}

# Endpoint to get the minimum weight of players from a specific college
@app.get("/v1/professional_basketball/min_weight_by_college", operation_id="get_min_weight_by_college", summary="Retrieve the lightest weight of professional basketball players who attended a specified college. The operation filters players based on their college and identifies the player with the lowest weight from the filtered list.")
async def get_min_weight_by_college(college: str = Query(..., description="College of the player")):
    cursor.execute("SELECT MIN(T1.weight) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.college = ?", (college,))
    result = cursor.fetchone()
    if not result:
        return {"min_weight": []}
    return {"min_weight": result[0]}

# Endpoint to get the maximum weight of players from a specific birth country
@app.get("/v1/professional_basketball/max_weight_by_birth_country", operation_id="get_max_weight_by_birth_country", summary="Retrieves the maximum weight of professional basketball players born in a specified country. The input parameter is used to filter the players by their birth country.")
async def get_max_weight_by_birth_country(birth_country: str = Query(..., description="Birth country of the player")):
    cursor.execute("SELECT MAX(T1.weight) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.birthCountry = ?", (birth_country,))
    result = cursor.fetchone()
    if not result:
        return {"max_weight": []}
    return {"max_weight": result[0]}

# Endpoint to get the total points scored by players in a specific season range who are still alive
@app.get("/v1/professional_basketball/total_points_by_season_range", operation_id="get_total_points_by_season_range", summary="Retrieves the cumulative points scored by professional basketball players within a specified season range, considering only those players who are still alive.")
async def get_total_points_by_season_range(start_season: int = Query(..., description="Start season (inclusive)"), end_season: int = Query(..., description="End season (inclusive)"), death_date: str = Query(..., description="Death date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(T2.points) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T2.season_id BETWEEN ? AND ? AND T1.deathDate = ?", (start_season, end_season, death_date))
    result = cursor.fetchone()
    if not result:
        return {"total_points": []}
    return {"total_points": result[0]}

# Endpoint to get distinct player names born after a specific date with a certain percentage of offensive rebounds
@app.get("/v1/professional_basketball/distinct_players_by_birthdate_and_rebounds", operation_id="get_distinct_players_by_birthdate_and_rebounds", summary="Retrieves a list of unique professional basketball players who were born after a specified date and have a higher percentage of offensive rebounds than a given threshold. The response includes the last and first names of the players.")
async def get_distinct_players_by_birthdate_and_rebounds(birth_date: str = Query(..., description="Birth date in 'YYYY-MM-DD' format"), rebound_percentage: float = Query(..., description="Percentage of offensive rebounds")):
    cursor.execute("SELECT DISTINCT T1.lastName, T1.firstName FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.birthDate > ? AND CAST(T2.o_rebounds AS REAL) * 100 / T2.rebounds > ?", (birth_date, rebound_percentage))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the count of players who received a specific award within a year range
@app.get("/v1/professional_basketball/count_players_by_award_and_year_range", operation_id="get_count_players_by_award_and_year_range", summary="Retrieve the total number of professional basketball players who have been awarded a specific honor within a specified year range. The operation considers the inclusive start and end years, as well as the exact award name, to calculate the count.")
async def get_count_players_by_award_and_year_range(start_year: int = Query(..., description="Start year (inclusive)"), end_year: int = Query(..., description="End year (inclusive)"), award: str = Query(..., description="Award name")):
    cursor.execute("SELECT COUNT(playerID) FROM awards_players WHERE year BETWEEN ? AND ? AND award = ?", (start_year, end_year, award))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get team IDs with a minimum number of distinct all-star players
@app.get("/v1/professional_basketball/team_ids_by_min_allstar_players", operation_id="get_team_ids_by_min_allstar_players", summary="Retrieves the IDs of professional basketball teams that have at least a specified number of distinct all-star players. The input parameter determines the minimum count of all-star players required for a team to be included in the results.")
async def get_team_ids_by_min_allstar_players(min_allstar_players: int = Query(..., description="Minimum number of distinct all-star players")):
    cursor.execute("SELECT T1.tmID FROM players_teams AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID GROUP BY T1.tmID HAVING COUNT(DISTINCT T1.playerID) >= ?", (min_allstar_players,))
    result = cursor.fetchall()
    if not result:
        return {"team_ids": []}
    return {"team_ids": [row[0] for row in result]}

# Endpoint to get the maximum points scored by a team in a specific year range and rank
@app.get("/v1/professional_basketball/max_points_by_year_range_and_rank", operation_id="get_max_points_by_year_range_and_rank", summary="Retrieves the highest points scored by a team within a specified year range and rank. The operation considers the team's rank and the inclusive start and end years to determine the maximum points scored.")
async def get_max_points_by_year_range_and_rank(start_year: int = Query(..., description="Start year (inclusive)"), end_year: int = Query(..., description="End year (inclusive)"), rank: int = Query(..., description="Team rank")):
    cursor.execute("SELECT MAX(T2.points) FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year BETWEEN ? AND ? AND T1.rank = ?", (start_year, end_year, rank))
    result = cursor.fetchone()
    if not result:
        return {"max_points": []}
    return {"max_points": result[0]}

# Endpoint to get player names from teams that ranked first for three consecutive years within a specific year range
@app.get("/v1/professional_basketball/players_from_top_teams_by_year_range", operation_id="get_players_from_top_teams_by_year_range", summary="Retrieves the full names of professional basketball players who were part of teams that consistently ranked first for three consecutive years within a specified year range. The year range and team rank are provided as input parameters.")
async def get_players_from_top_teams_by_year_range(start_year: int = Query(..., description="Start year (inclusive)"), end_year: int = Query(..., description="End year (inclusive)"), rank: int = Query(..., description="Team rank")):
    cursor.execute("SELECT T5.lastName, T5.firstName FROM players_teams AS T4 INNER JOIN players AS T5 ON T4.playerID = T5.playerID WHERE T4.year BETWEEN ? AND ? AND T4.tmID IN ( SELECT DISTINCT T1.tmID FROM teams AS T1 INNER JOIN teams AS T2 INNER JOIN teams AS T3 ON T1.tmID = T2.tmID AND T2.tmID = T3.tmID AND T3.year - T2.year = 1 AND T2.year - T1.year = 1 WHERE T1.rank = ? AND T1.year BETWEEN ? AND ? )", (start_year, end_year, rank, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the team name with the highest points in a specific year range and rank
@app.get("/v1/professional_basketball/top_team_by_year_range_and_rank", operation_id="get_top_team_by_year_range_and_rank", summary="Retrieves the name of the team with the highest points within a specified rank and year range. The rank and year range are provided as input parameters. The team with the highest points is determined by considering all players who were part of the team during the specified years.")
async def get_top_team_by_year_range_and_rank(rank: int = Query(..., description="Team rank"), start_year: int = Query(..., description="Start year (inclusive)"), end_year: int = Query(..., description="End year (inclusive)")):
    cursor.execute("SELECT DISTINCT T1.name FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.rank = ? AND T1.year BETWEEN ? AND ? ORDER BY T2.points DESC LIMIT 1", (rank, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the count of teams with a minimum total points in a specific year
@app.get("/v1/professional_basketball/count_teams_by_min_points_and_year", operation_id="get_count_teams_by_min_points_and_year", summary="Retrieves the number of professional basketball teams that scored at least a specified minimum total points in a given year. The operation calculates the total points scored by each team in the specified year and returns the count of teams that meet or exceed the provided minimum points threshold.")
async def get_count_teams_by_min_points_and_year(year: int = Query(..., description="Year"), min_points: int = Query(..., description="Minimum total points")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T2.name, SUM(T1.points) FROM players_teams AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? GROUP BY T2.name HAVING SUM(points) >= ? ) AS T3", (year, min_points))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct players in teams with a win percentage greater than a specified value in a given year
@app.get("/v1/professional_basketball/count_distinct_players_win_percentage_year", operation_id="get_count_distinct_players", summary="Retrieve the number of unique players who have been part of teams with a win percentage surpassing a specified threshold in a given year. This operation considers the provided win percentage and year to filter the teams and count the distinct players.")
async def get_count_distinct_players(win_percentage: float = Query(..., description="Win percentage threshold"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players_teams AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE CAST(T2.won AS REAL) * 100 / CAST(T2.games AS REAL) > ? AND T1.year = ?", (win_percentage, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last names of players with a specific race and height greater than a specified value, ordered by height in ascending order
@app.get("/v1/professional_basketball/players_by_race_height", operation_id="get_players_by_race_height", summary="Retrieves the names of professional basketball players of a specified race who are taller than a given height, sorted by height in ascending order. The response is limited to a specified number of results.")
async def get_players_by_race_height(race: str = Query(..., description="Race of the player"), height: int = Query(..., description="Minimum height of the player"), limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT firstName, lastName FROM players WHERE race = ? AND height > ? ORDER BY height ASC LIMIT ?", (race, height, limit))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the count of distinct players drafted with a specific first name in a given year
@app.get("/v1/professional_basketball/count_drafted_players_by_name_year", operation_id="get_count_drafted_players", summary="Retrieves the total number of unique basketball players with a specified first name who were drafted in a particular year.")
async def get_count_drafted_players(first_name: str = Query(..., description="First name of the player"), draft_year: int = Query(..., description="Draft year")):
    cursor.execute("SELECT COUNT(DISTINCT playerID) FROM draft WHERE firstName = ? AND draftYear = ?", (first_name, draft_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of field goals made by a player with a specific first and last name within a range of seasons
@app.get("/v1/professional_basketball/count_fg_made_by_player_season_range", operation_id="get_count_fg_made", summary="Retrieve the total number of field goals made by a specific professional basketball player within a specified range of seasons. The player is identified by their first and last name, and the season range is defined by the start and end season parameters.")
async def get_count_fg_made(first_name: str = Query(..., description="First name of the player"), last_name: str = Query(..., description="Last name of the player"), start_season: int = Query(..., description="Start season"), end_season: int = Query(..., description="End season")):
    cursor.execute("SELECT COUNT(fg_made) FROM player_allstar WHERE first_name = ? AND last_name = ? AND season_id BETWEEN ? AND ?", (first_name, last_name, start_season, end_season))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the minimum and maximum BMI of players
@app.get("/v1/professional_basketball/min_max_bmi", operation_id="get_min_max_bmi", summary="Retrieves the minimum and maximum Body Mass Index (BMI) values of professional basketball players. The BMI is calculated based on the weight and height of each player in the database.")
async def get_min_max_bmi():
    cursor.execute("SELECT MIN(CAST(weight AS REAL) / (height * height)) , MAX(CAST(weight AS REAL) / (height * height)) FROM players")
    result = cursor.fetchone()
    if not result:
        return {"min_bmi": [], "max_bmi": []}
    return {"min_bmi": result[0], "max_bmi": result[1]}

# Endpoint to get the team with the highest home win percentage
@app.get("/v1/professional_basketball/team_highest_home_win_percentage", operation_id="get_team_highest_home_win_percentage", summary="Retrieves the team with the highest home win percentage from the professional basketball teams. The operation returns the name of the team with the highest ratio of home wins to total home games played. The number of results returned can be limited by specifying the 'limit' parameter.")
async def get_team_highest_home_win_percentage(limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT name FROM teams ORDER BY CAST(homeWon AS REAL) / (homeWon + homeLost) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result[0]}

# Endpoint to get the age of a player at the time of draft based on first name, last name, and draft round
@app.get("/v1/professional_basketball/player_age_at_draft", operation_id="get_player_age_at_draft", summary="Retrieves the age of a professional basketball player at the time of their draft, based on their first and last names and the round in which they were drafted. The age is calculated by subtracting the player's birth year from the draft year.")
async def get_player_age_at_draft(first_name: str = Query(..., description="First name of the player"), last_name: str = Query(..., description="Last name of the player"), draft_round: int = Query(..., description="Draft round")):
    cursor.execute("SELECT draftYear - strftime('%Y', birthDate) FROM draft AS T1 INNER JOIN players AS T2 ON T1.playerID = T2.playerID WHERE T1.firstName = ? AND T1.lastName = ? AND draftRound = ?", (first_name, last_name, draft_round))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the tallest player in a specific team after a given year
@app.get("/v1/professional_basketball/tallest_player_team_year", operation_id="get_tallest_player", summary="Retrieves the tallest player(s) from a specified team who played after a given year. The number of results returned can be limited by the user.")
async def get_tallest_player(team_name: str = Query(..., description="Name of the team"), year: int = Query(..., description="Year"), limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID INNER JOIN teams AS T3 ON T3.tmID = T2.tmID WHERE T3.name = ? AND T2.year > ? ORDER BY T1.height DESC LIMIT ?", (team_name, year, limit))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": result}

# Endpoint to get the player with the most awards who has passed away
@app.get("/v1/professional_basketball/most_awarded_deceased_player", operation_id="get_most_awarded_deceased_player", summary="Retrieves the professional basketball player who has received the most awards and is no longer living. The operation returns the top result based on the specified limit. The limit parameter determines the number of results to return.")
async def get_most_awarded_deceased_player(limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT T1.playerID FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE deathDate IS NOT NULL GROUP BY T1.playerID ORDER BY COUNT(award) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"playerID": []}
    return {"playerID": result[0]}

# Endpoint to get the team with the most players from a specific college
@app.get("/v1/professional_basketball/team_most_players_from_college", operation_id="get_team_most_players_from_college", summary="Retrieves the team with the highest number of players from a specified college. The operation returns the team name and is limited to the specified number of results. The input parameters include the college name and the maximum number of results to return.")
async def get_team_most_players_from_college(college: str = Query(..., description="Name of the college"), limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT T3.name FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID INNER JOIN teams AS T3 ON T3.tmID = T2.tmID WHERE T1.college = ? GROUP BY T3.name ORDER BY COUNT(DISTINCT T1.playerID) DESC LIMIT ?", (college, limit))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": result[0]}

# Endpoint to get the average BMI of all-star players
@app.get("/v1/professional_basketball/average_bmi_allstar_players", operation_id="get_average_bmi_allstar_players", summary="Retrieves the average Body Mass Index (BMI) of all professional basketball players who have been selected as all-stars. The BMI is calculated using the weight and height of each player, and the average is computed across all all-star players.")
async def get_average_bmi_allstar_players():
    cursor.execute("SELECT AVG(CAST(T1.weight AS REAL) / (T1.height * T1.height)) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID")
    result = cursor.fetchone()
    if not result:
        return {"average_bmi": []}
    return {"average_bmi": result[0]}

# Endpoint to get the team with the highest win rate improvement from one year to the next
@app.get("/v1/professional_basketball/team_highest_win_rate_improvement", operation_id="get_team_highest_win_rate_improvement", summary="Retrieves the team with the highest improvement in win rate from the previous year to the current year in a specified professional basketball league. The operation compares the win rates of teams in the provided leagues and years, and returns the team with the most significant increase in win rate.")
async def get_team_highest_win_rate_improvement(lgID_prev: str = Query(..., description="League ID for the previous year"), year_prev: int = Query(..., description="Year for the previous season"), lgID_current: str = Query(..., description="League ID for the current year"), year_current: int = Query(..., description="Year for the current season")):
    cursor.execute("SELECT T1.name FROM teams AS T1 INNER JOIN ( SELECT * FROM teams WHERE lgID = ? AND year = ? ) AS T2 ON T1.tmID = T2.tmID WHERE T1.lgID = ? AND T1.year = ? ORDER BY (CAST(T1.won AS REAL) / (T1.won + T1.lost) - (CAST(T2.won AS REAL) / (T2.won + T2.lost))) DESC LIMIT 1", (lgID_prev, year_prev, lgID_current, year_current))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the player with the most personal fouls in a specific league
@app.get("/v1/professional_basketball/player_most_personal_fouls", operation_id="get_player_most_personal_fouls", summary="Retrieves the first name of the player who has committed the most personal fouls in a specific league. The league is identified by its unique ID. The result is determined by counting the number of personal fouls (PF) for each player in the league and ordering them in descending order. The player with the highest count is returned.")
async def get_player_most_personal_fouls(lgID: str = Query(..., description="League ID")):
    cursor.execute("SELECT T1.firstName FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.lgID = ? GROUP BY T1.playerID, T1.firstName ORDER BY COUNT(PF) DESC LIMIT 1", (lgID,))
    result = cursor.fetchone()
    if not result:
        return {"player_first_name": []}
    return {"player_first_name": result[0]}

# Endpoint to get the average height of all-star players in a specific conference
@app.get("/v1/professional_basketball/average_height_allstar_players", operation_id="get_average_height_allstar_players", summary="Retrieves the average height of all-star players from a specified conference. The calculation is based on distinct heights to ensure accuracy and avoid duplicates. The conference is determined by the provided input parameter.")
async def get_average_height_allstar_players(conference: str = Query(..., description="Conference name")):
    cursor.execute("SELECT AVG(DISTINCT height) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE conference = ?", (conference,))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the coach with the most wins for a specific award
@app.get("/v1/professional_basketball/coach_most_wins_award", operation_id="get_coach_most_wins_award", summary="Retrieves the coach with the highest number of wins for a specified award. The award is provided as an input parameter. The operation returns the coach's unique identifier.")
async def get_coach_most_wins_award(award: str = Query(..., description="Award name")):
    cursor.execute("SELECT T1.coachID FROM coaches AS T1 INNER JOIN awards_coaches AS T2 ON T1.coachID = T2.coachID WHERE T2.award = ? GROUP BY T1.coachID, T1.won ORDER BY T1.won DESC LIMIT 1", (award,))
    result = cursor.fetchone()
    if not result:
        return {"coach_id": []}
    return {"coach_id": result[0]}

# Endpoint to get the team name for a specific award in a given year
@app.get("/v1/professional_basketball/team_name_award_year", operation_id="get_team_name_award_year", summary="Retrieves the name of the professional basketball team that won a specific award in a given year. The operation uses the provided year and award name to identify the team.")
async def get_team_name_award_year(year: int = Query(..., description="Year"), award: str = Query(..., description="Award name")):
    cursor.execute("SELECT name FROM teams AS T1 INNER JOIN coaches AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year INNER JOIN awards_coaches AS T3 ON T2.coachID = T3.coachID AND T2.year = T3.year WHERE T3.year = ? AND award = ?", (year, award))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the player with the highest field goal percentage in a specific year
@app.get("/v1/professional_basketball/player_highest_fg_percentage", operation_id="get_player_highest_fg_percentage", summary="Get the player with the highest field goal percentage in a specific year")
async def get_player_highest_fg_percentage(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE year = ? ORDER BY CAST(T2.fgMade AS REAL) / T2.fgAttempted DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"player_name": []}
    return {"player_name": result[0] + " " + result[1]}

# Endpoint to get the team name for a specific drafted player
@app.get("/v1/professional_basketball/team_name_drafted_player", operation_id="get_team_name_drafted_player", summary="Get the team name for a specific drafted player")
async def get_team_name_drafted_player(first_name: str = Query(..., description="First name of the drafted player"), last_name: str = Query(..., description="Last name of the drafted player")):
    cursor.execute("SELECT T1.name FROM teams AS T1 INNER JOIN draft AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.draftYear WHERE T2.firstName = ? AND T2.lastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"team_name": []}
    return {"team_name": result[0]}

# Endpoint to get the count of distinct players who won a specific award within a given year range and birth country
@app.get("/v1/professional_basketball/count_distinct_players_award_year_range", operation_id="get_count_distinct_players_award_year_range", summary="Retrieve the number of unique basketball players who have won a specific award within a specified range of years and were born in a particular country.")
async def get_count_distinct_players_award_year_range(award: str = Query(..., description="Award name"), birth_country: str = Query(..., description="Birth country"), start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT COUNT(DISTINCT T2.playerID) FROM awards_players AS T1 INNER JOIN players AS T2 ON T1.playerID = T2.playerID WHERE T1.award = ? AND T2.birthCountry = ? AND T1.year BETWEEN ? AND ?", (award, birth_country, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct all-star players from a specific college
@app.get("/v1/professional_basketball/distinct_allstar_players_college", operation_id="get_distinct_allstar_players_college", summary="Retrieve a unique list of professional basketball players who have been selected as all-stars and attended a specific college. The college name is provided as an input parameter.")
async def get_distinct_allstar_players_college(college: str = Query(..., description="College name")):
    cursor.execute("SELECT DISTINCT T1.firstName, T1.lastName FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.college = ?", (college,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of coaches who received awards between specified years and have more than a specified number of awards
@app.get("/v1/professional_basketball/count_coaches_awards_between_years", operation_id="get_count_coaches_awards_between_years", summary="Retrieves the count of professional basketball coaches who received awards between the specified start and end years, and have been awarded more than the specified minimum number of times.")
async def get_count_coaches_awards_between_years(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), min_awards: int = Query(..., description="Minimum number of awards received")):
    cursor.execute("SELECT COUNT(coachID) FROM awards_coaches WHERE year BETWEEN ? AND ? GROUP BY coachID HAVING COUNT(coachID) > ?", (start_year, end_year, min_awards))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct players who received a specific award between specified years
@app.get("/v1/professional_basketball/count_players_award_between_years", operation_id="get_count_players_award_between_years", summary="Retrieves the number of unique professional basketball players who were awarded a specific honor during a specified period. The period is defined by a start and end year, and the award is identified by its name.")
async def get_count_players_award_between_years(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), award: str = Query(..., description="Name of the award")):
    cursor.execute("SELECT COUNT(DISTINCT playerID) FROM awards_players WHERE year BETWEEN ? AND ? AND award = ?", (start_year, end_year, award))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct team names where the number of players born in a specific state exceeds a specified count
@app.get("/v1/professional_basketball/distinct_team_names_by_birth_state_count", operation_id="get_distinct_team_names_by_birth_state_count", summary="Retrieve a list of unique team names that have more than the specified minimum count of players born in the same state. This operation provides insights into the distribution of players across teams based on their birth state.")
async def get_distinct_team_names_by_birth_state_count(min_count: int = Query(..., description="Minimum count of players born in a specific state")):
    cursor.execute("SELECT DISTINCT name FROM teams WHERE tmID IN ( SELECT tmID FROM players_teams AS T1 INNER JOIN players AS T2 ON T1.playerID = T2.playerID WHERE T2.birthState IS NOT NULL GROUP BY T1.tmID, T2.birthState HAVING COUNT(*) > ? )", (min_count,))
    result = cursor.fetchall()
    if not result:
        return {"team_names": []}
    return {"team_names": [row[0] for row in result]}

# Endpoint to get the count of teams with more than a specified number of distinct all-star players in a given league
@app.get("/v1/professional_basketball/count_teams_allstar_players_league", operation_id="get_count_teams_allstar_players_league", summary="Retrieves the number of teams in a specified league that have more than a given number of unique all-star players. The league is identified by its unique ID, and the minimum count of all-star players is also provided.")
async def get_count_teams_allstar_players_league(league_id: str = Query(..., description="League ID"), min_allstars: int = Query(..., description="Minimum number of distinct all-star players")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT tmID FROM players_teams AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.lgID = ? GROUP BY T1.tmID HAVING COUNT(DISTINCT T1.playerID) > ? ) AS T3", (league_id, min_allstars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top birth states of all-star players
@app.get("/v1/professional_basketball/top_birth_states_allstar_players", operation_id="get_top_birth_states_allstar_players", summary="Retrieves a list of the top birth states for professional basketball all-star players, ranked by the number of unique all-star players born in each state. The list is limited to the specified number of top states.")
async def get_top_birth_states_allstar_players(limit: int = Query(..., description="Number of top birth states to return")):
    cursor.execute("SELECT T1.birthState FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID GROUP BY T1.birthState ORDER BY COUNT(DISTINCT T1.playerID) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"birth_states": []}
    return {"birth_states": [row[0] for row in result]}

# Endpoint to get the count of distinct players in a team with a specific rank in a given year
@app.get("/v1/professional_basketball/count_players_team_rank_year", operation_id="get_count_players_team_rank_year", summary="Retrieves the number of unique players who were part of a team that achieved a specific rank in a given year. The response is based on the provided year and team rank.")
async def get_count_players_team_rank_year(year: int = Query(..., description="Year"), rank: int = Query(..., description="Rank of the team")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID INNER JOIN teams AS T3 ON T3.tmID = T2.tmID WHERE T3.year = ? AND T3.rank = ?", (year, rank))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct players in teams with a win percentage above a specified threshold
@app.get("/v1/professional_basketball/count_players_win_percentage", operation_id="get_count_players_win_percentage", summary="Retrieves the number of unique players who have been part of teams with a win percentage exceeding the provided threshold. The win percentage is calculated as the ratio of total wins to the sum of total wins and losses.")
async def get_count_players_win_percentage(win_percentage: float = Query(..., description="Win percentage threshold")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID INNER JOIN teams AS T3 ON T3.tmID = T2.tmID WHERE CAST(T3.lost AS REAL) * 100 / (T3.lost + T3.won) < ?", (win_percentage,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of players born outside a specified country
@app.get("/v1/professional_basketball/players_born_outside_country", operation_id="get_players_born_outside_country", summary="Retrieves the full names of professional basketball players who were born outside the specified country. The response includes the first, middle, and last names of these players.")
async def get_players_born_outside_country(birth_country: str = Query(..., description="Country of birth")):
    cursor.execute("SELECT firstName, middleName, lastName FROM players WHERE birthCountry != ?", (birth_country,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [{"firstName": row[0], "middleName": row[1], "lastName": row[2]} for row in result]}

# Endpoint to get the top coach by win-loss difference in a specified year
@app.get("/v1/professional_basketball/top_coach_win_loss_year", operation_id="get_top_coach_win_loss_year", summary="Retrieves the top professional basketball coaches ranked by the difference between their wins and losses in a specified year. The response is limited to the number of top coaches provided as an input parameter.")
async def get_top_coach_win_loss_year(year: int = Query(..., description="Year"), limit: int = Query(..., description="Number of top coaches to return")):
    cursor.execute("SELECT coachID FROM coaches WHERE year = ? ORDER BY won - lost DESC LIMIT ?", (year, limit))
    result = cursor.fetchall()
    if not result:
        return {"coach_ids": []}
    return {"coach_ids": [row[0] for row in result]}

# Endpoint to get distinct team IDs of coaches who received a specific award in a given year
@app.get("/v1/professional_basketball/distinct_team_ids_coach_award_year", operation_id="get_distinct_team_ids_coach_award_year", summary="Retrieve a unique list of team IDs for coaches who were honored with a specific award in a particular year. The operation filters the coaches by the provided award and year, then identifies the distinct teams they belong to.")
async def get_distinct_team_ids_coach_award_year(year: int = Query(..., description="Year"), award: str = Query(..., description="Name of the award")):
    cursor.execute("SELECT DISTINCT T1.tmID FROM coaches AS T1 INNER JOIN awards_coaches AS T2 ON T1.coachID = T2.coachID WHERE T2.year = ? AND T2.award = ?", (year, award))
    result = cursor.fetchall()
    if not result:
        return {"team_ids": []}
    return {"team_ids": [row[0] for row in result]}

# Endpoint to get distinct player details who attempted more than a specified number of free throws and made all of them
@app.get("/v1/professional_basketball/players_free_throws", operation_id="get_players_free_throws", summary="Retrieves unique details of professional basketball players who have attempted more than the specified number of free throws and made all of them. The response includes the player's first name, last name, height, and weight.")
async def get_players_free_throws(ft_attempted: int = Query(..., description="Number of free throws attempted")):
    cursor.execute("SELECT DISTINCT T1.firstName, T1.lastName, T1.height, T1.weight FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T2.ft_attempted > ? AND ft_attempted = ft_made", (ft_attempted,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get distinct player details drafted from a specific city within a specified year range
@app.get("/v1/professional_basketball/players_drafted_from_city", operation_id="get_players_drafted_from_city", summary="Retrieve unique player profiles who were drafted from a particular city within a specified year range. The response includes the first name, last name, and team name of each player. The city and year range are provided as input parameters.")
async def get_players_drafted_from_city(draft_from: str = Query(..., description="City from which the player was drafted"), start_year: int = Query(..., description="Start year of the draft range"), end_year: int = Query(..., description="End year of the draft range")):
    cursor.execute("SELECT DISTINCT T1.firstName, T1.lastName, T3.name FROM players AS T1 INNER JOIN draft AS T2 ON T1.playerID = T2.playerID INNER JOIN teams AS T3 ON T2.tmID = T3.tmID WHERE T2.draftFrom = ? AND T2.draftYear BETWEEN ? AND ?", (draft_from, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get player details who won a specific award in a given year and their age in that year
@app.get("/v1/professional_basketball/players_award_year", operation_id="get_players_award_year", summary="Retrieves the first name, middle name, last name, and age of professional basketball players who won a specific award in a given year. The age is calculated based on the birth date of the player and the year of the award.")
async def get_players_award_year(award: str = Query(..., description="Award won by the player"), year: int = Query(..., description="Year the award was won")):
    cursor.execute("SELECT T1.firstName, T1.middleName, T1.lastName , ? - strftime('%Y', T1.birthDate) FROM awards_players AS T2 JOIN players AS T1 ON T2.playerID = T1.playerID WHERE T2.award = ? AND T2.year = ?", (year, award, year))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get distinct team names and years they won the finals within a specified year range
@app.get("/v1/professional_basketball/teams_finals_winners", operation_id="get_teams_finals_winners", summary="Retrieves unique team names and the years they emerged victorious in the finals, within a specified range of years. The operation filters results based on a specific round of the series. The round, start year, and end year of the range are required input parameters.")
async def get_teams_finals_winners(round: str = Query(..., description="Round of the series"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT DISTINCT T1.name, T2.year FROM teams AS T1 JOIN series_post AS T2 ON T1.tmID = T2.tmIDWinner WHERE T2.round = ? AND T2.year BETWEEN ? AND ?", (round, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": result}

# Endpoint to get distinct coach IDs and team names for coaches who won a specific round in a given year
@app.get("/v1/professional_basketball/coaches_winning_teams", operation_id="get_coaches_winning_teams", summary="Retrieve unique coach identifiers and team names for coaches who led their teams to victory in a specific round of a given year. The operation filters results based on the round and year provided as input parameters.")
async def get_coaches_winning_teams(round: str = Query(..., description="Round of the series"), year: int = Query(..., description="Year of the series")):
    cursor.execute("SELECT DISTINCT T1.coachID, T3.name FROM coaches AS T1 JOIN series_post AS T2 ON T1.tmID = T2.tmIDWinner JOIN teams AS T3 ON T3.tmID = T1.tmID WHERE T2.round = ? AND T2.year = ?", (round, year))
    result = cursor.fetchall()
    if not result:
        return {"coaches": []}
    return {"coaches": result}

# Endpoint to get distinct player details who were part of the winning team in a specific year and round
@app.get("/v1/professional_basketball/players_winning_team", operation_id="get_players_winning_team", summary="Retrieve unique player profiles who were members of the championship team in a specified year and round. The response includes the first, middle, and last names of the players.")
async def get_players_winning_team(year: int = Query(..., description="Year of the series"), round: str = Query(..., description="Round of the series")):
    cursor.execute("SELECT DISTINCT T3.firstName, T3.middleName, T3.lastName FROM series_post AS T1 INNER JOIN players_teams AS T2 ON T1.tmIDWinner = T2.tmID INNER JOIN players AS T3 ON T3.playerID = T2.playerID WHERE T1.year = ? AND T1.round = ?", (year, round))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the college of the player who won the most awards in a specific year
@app.get("/v1/professional_basketball/most_awards_player_college", operation_id="get_most_awards_player_college", summary="Retrieves the college of the professional basketball player who received the most awards in a specified year. The year parameter is used to filter the awards and determine the player with the highest number of accolades.")
async def get_most_awards_player_college(year: int = Query(..., description="Year of the awards")):
    cursor.execute("SELECT college FROM players WHERE playerID = ( SELECT playerID FROM awards_players WHERE year = ? GROUP BY playerID ORDER BY COUNT(award) DESC LIMIT 1 )", (year,))
    result = cursor.fetchone()
    if not result:
        return {"college": []}
    return {"college": result[0]}

# Endpoint to get the youngest player who won a specific award
@app.get("/v1/professional_basketball/youngest_award_winner", operation_id="get_youngest_award_winner", summary="Retrieves the youngest professional basketball player who won a specific award. The award is specified as an input parameter. The response includes the first, middle, and last names of the player.")
async def get_youngest_award_winner(award: str = Query(..., description="Award won by the player")):
    cursor.execute("SELECT T1.firstName, T1.middleName, T1.lastName FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ? ORDER BY T1.birthDate DESC LIMIT 1", (award,))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": result}

# Endpoint to get player details drafted in a specific round, from a country other than a specified one, in a given year
@app.get("/v1/professional_basketball/players_drafted_by_round_country", operation_id="get_players_drafted_by_round_country", summary="Retrieves the first, middle, and last names of professional basketball players drafted in a specific round, excluding those born in a specified country, during a given year.")
async def get_players_drafted_by_round_country(draft_round: int = Query(..., description="Draft round"), birth_country: str = Query(..., description="Country of birth"), draft_year: int = Query(..., description="Year of the draft")):
    cursor.execute("SELECT T1.firstName, T1.middleName, T1.lastName FROM players AS T1 INNER JOIN draft AS T2 ON T1.playerID = T2.playerID WHERE T2.draftRound = ? AND T1.birthCountry != ? AND T2.draftYear = ?", (draft_round, birth_country, draft_year))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the percentage of offensive rebounds out of total rebounds for a specific year
@app.get("/v1/professional_basketball/offensive_rebounds_percentage", operation_id="get_offensive_rebounds_percentage", summary="Retrieves the percentage of offensive rebounds out of total rebounds for a specific year. This operation calculates the ratio of offensive rebounds to total rebounds for all players in the given year, providing a comprehensive view of offensive rebounding performance.")
async def get_offensive_rebounds_percentage(year: int = Query(..., description="Year of the rebounds")):
    cursor.execute("SELECT CAST(SUM(T2.o_rebounds) AS REAL) * 100 / SUM(T2.rebounds) FROM players_teams AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.year = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct year, team name, and coach ID for coaches with a win rate greater than a specified value
@app.get("/v1/professional_basketball/coaches_win_rate", operation_id="get_coaches_win_rate", summary="Retrieves unique combinations of year, team name, and coach ID for coaches who have achieved a win rate surpassing the provided threshold. The win rate is calculated as the ratio of games won to the total number of games played.")
async def get_coaches_win_rate(win_rate: float = Query(..., description="Win rate threshold")):
    cursor.execute("SELECT DISTINCT T1.year, T2.name, T1.coachID FROM coaches AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID WHERE CAST(T1.won AS REAL) / CAST((T1.won + T1.lost) AS REAL) > ?", (win_rate,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct coach IDs for a specified team name
@app.get("/v1/professional_basketball/coaches_by_team", operation_id="get_coaches_by_team", summary="Retrieves a unique list of coach identifiers associated with a specific professional basketball team. The team is identified by its name.")
async def get_coaches_by_team(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT DISTINCT coachID FROM coaches AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID WHERE name = ?", (team_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of players with a specified offensive rebound percentage and year
@app.get("/v1/professional_basketball/player_count_orebounds_year", operation_id="get_player_count_orebounds_year", summary="Retrieves the number of professional basketball players who have achieved a specified offensive rebound percentage in a given year. The offensive rebound percentage is calculated as the ratio of offensive rebounds to defensive rebounds, multiplied by 100.")
async def get_player_count_orebounds_year(orebound_percentage: float = Query(..., description="Offensive rebound percentage"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT COUNT(playerID) FROM players_teams WHERE CAST(oRebounds AS REAL) * 100 / dRebounds <= ? AND year = ?", (orebound_percentage, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct all-star players within a specified season range and maximum steals
@app.get("/v1/professional_basketball/allstar_player_count", operation_id="get_allstar_player_count", summary="Retrieves the number of unique all-star players who played between the specified start and end seasons and had a maximum number of steals within the given range.")
async def get_allstar_player_count(start_season: int = Query(..., description="Start season"), end_season: int = Query(..., description="End season"), max_steals: int = Query(..., description="Maximum steals")):
    cursor.execute("SELECT COUNT(DISTINCT playerID) FROM player_allstar WHERE season_id BETWEEN ? AND ? AND steals <= ?", (start_season, end_season, max_steals))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get player IDs based on award, draft year, and draft round
@app.get("/v1/professional_basketball/player_ids_by_award_draft", operation_id="get_player_ids_by_award_draft", summary="Retrieves the IDs of professional basketball players who have received a specific award and were drafted in a particular year and round. The operation filters players based on the provided award, draft year, and draft round, and returns their unique player IDs.")
async def get_player_ids_by_award_draft(award: str = Query(..., description="Award name"), draft_year: int = Query(..., description="Draft year"), draft_round: int = Query(..., description="Draft round")):
    cursor.execute("SELECT T1.playerID FROM draft AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T2.award = ? AND T1.draftYear = ? AND T1.draftRound = ?", (award, draft_year, draft_round))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of distinct all-star players by season and race
@app.get("/v1/professional_basketball/allstar_player_count_by_season_race", operation_id="get_allstar_player_count_by_season_race", summary="Retrieves the number of unique all-star players for a specific season and race. The season is identified by its unique ID, and the race refers to the ethnicity of the players.")
async def get_allstar_player_count_by_season_race(season_id: int = Query(..., description="Season ID"), race: str = Query(..., description="Race of the player")):
    cursor.execute("SELECT COUNT(DISTINCT T1.playerID) FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T2.season_id = ? AND T1.race = ?", (season_id, race))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get team IDs based on post-season series year, round, and defensive points
@app.get("/v1/professional_basketball/team_ids_by_series_year_round_dpts", operation_id="get_team_ids_by_series_year_round_dpts", summary="Retrieves the team IDs of the winning teams from a specific post-season series, based on the provided year, round, and defensive points. This operation filters the series data to match the given year and round, then identifies the winning team's ID. It further narrows down the results by considering the defensive points of the team.")
async def get_team_ids_by_series_year_round_dpts(year: int = Query(..., description="Year"), round: str = Query(..., description="Round"), d_pts: int = Query(..., description="Defensive points")):
    cursor.execute("SELECT T2.tmID FROM series_post AS T1 INNER JOIN teams AS T2 ON T1.tmIDWinner = T2.tmID WHERE T1.year = ? AND T1.round = ? AND T2.d_pts = ?", (year, round, d_pts))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of games lost by a specified team
@app.get("/v1/professional_basketball/team_loss_percentage", operation_id="get_team_loss_percentage", summary="Retrieves the percentage of games lost by a specific professional basketball team. The team is identified by its name, which is provided as an input parameter. The calculation is based on the total number of games played and the number of games lost by the team.")
async def get_team_loss_percentage(team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT CAST(SUM(lost) AS REAL) * 100 / SUM(games) FROM teams WHERE name = ?", (team_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get player IDs based on award year, award name, and league ID
@app.get("/v1/professional_basketball/player_ids_by_award_year_league", operation_id="get_player_ids_by_award_year_league", summary="Retrieves a list of player IDs who have received an award in a specific year and league. The award name is used to filter the results. The year, award name, and league ID are required as input parameters.")
async def get_player_ids_by_award_year_league(year: int = Query(..., description="Year"), award: str = Query(..., description="Award name"), lgID: str = Query(..., description="League ID")):
    cursor.execute("SELECT playerID FROM awards_players WHERE year > ? AND award = ? AND lgID = ?", (year, award, lgID))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of distinct years a player played for a specific team within a specified year range
@app.get("/v1/professional_basketball/player_years_count_by_team_year_range", operation_id="get_player_years_count_by_team_year_range", summary="Retrieve the number of unique seasons a specific player was part of a given team, within a defined year range.")
async def get_player_years_count_by_team_year_range(tmID: str = Query(..., description="Team ID"), start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year"), playerID: str = Query(..., description="Player ID")):
    cursor.execute("SELECT COUNT(DISTINCT T2.year) FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.tmID = ? AND T2.year BETWEEN ? AND ? AND T1.playerID = ?", (tmID, start_year, end_year, playerID))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of a specific award in a given year
@app.get("/v1/professional_basketball/award_percentage_by_year", operation_id="get_award_percentage_by_year", summary="Retrieves the percentage of a specified award granted in a particular year. The calculation is based on the total number of awards given in that year.")
async def get_award_percentage_by_year(award: str = Query(..., description="Award name"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN award = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM awards_coaches WHERE year = ?", (award, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in win percentage between two years for a specific team
@app.get("/v1/professional_basketball/win_percentage_difference", operation_id="get_win_percentage_difference", summary="Retrieves the difference in win percentage for a specific team between two given years. The operation calculates the win percentage for each year by dividing the total number of games won by the team in that year by the total number of games played by the team in that year. The result is then subtracted to find the difference in win percentage between the two years.")
async def get_win_percentage_difference(year1: int = Query(..., description="First year"), year2: int = Query(..., description="Second year"), tmIDWinner: str = Query(..., description="Team ID")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN year = ? AND tmIDWinner = ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN year = ? THEN 1 ELSE 0 END) - CAST(SUM(CASE WHEN year = ? AND tmIDWinner = ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN year = ? THEN 1 ELSE 0 END) FROM series_post", (year1, tmIDWinner, year1, year2, tmIDWinner, year2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of awards for players from a specific high school city in a given year
@app.get("/v1/professional_basketball/award_count_by_high_school_city", operation_id="get_award_count_by_high_school_city", summary="Retrieves the total number of awards received by professional basketball players from a specific high school city in a given year. The operation requires the year and the city where the high school is located as input parameters.")
async def get_award_count_by_high_school_city(year: int = Query(..., description="Year"), hsCity: str = Query(..., description="High school city")):
    cursor.execute("SELECT COUNT(T1.award) FROM awards_players AS T1 INNER JOIN players AS T2 ON T1.playerID = T2.playerID WHERE T1.year = ? AND T2.hsCity = ?", (year, hsCity))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of a specific award for players from a specific college
@app.get("/v1/professional_basketball/award_percentage_by_college", operation_id="get_award_percentage_by_college", summary="Retrieves the percentage of a specified award earned by players from a given college. The calculation is based on the total number of players from the college and the count of those who have received the specified award.")
async def get_award_percentage_by_college(award: str = Query(..., description="Award name"), college: str = Query(..., description="College name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.award = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T1.college = ?", (award, college))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the top 5 players by steals in a specific season
@app.get("/v1/professional_basketball/top_players_by_steals", operation_id="get_top_players_by_steals", summary="Retrieves the top 5 players with the highest number of steals in a specified season. The season is identified by its unique season ID. The response includes the first, middle, and last names of the players.")
async def get_top_players_by_steals(season_id: int = Query(..., description="Season ID")):
    cursor.execute("SELECT DISTINCT T1.firstName, T1.middleName, T1.lastName FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T2.season_id = ? ORDER BY T2.steals DESC LIMIT 5", (season_id,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": result}

# Endpoint to get the colleges of players who won a specific award in a given year
@app.get("/v1/professional_basketball/colleges_by_award_year", operation_id="get_colleges_by_award_year", summary="Retrieves the colleges of professional basketball players who received a specified award in a given year. The operation requires the year and the name of the award as input parameters.")
async def get_colleges_by_award_year(year: int = Query(..., description="Year"), award: str = Query(..., description="Award name")):
    cursor.execute("SELECT T1.college FROM players AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? AND T2.award = ?", (year, award))
    result = cursor.fetchall()
    if not result:
        return {"colleges": []}
    return {"colleges": result}

# Endpoint to get player IDs with a specific free throw percentage in a given season
@app.get("/v1/professional_basketball/player_ids_by_ft_percentage", operation_id="get_player_ids_by_ft_percentage", summary="Retrieve the IDs of professional basketball players who achieved a specified free throw percentage in a given season. The operation filters players based on the provided season ID and free throw percentage, returning the corresponding player IDs.")
async def get_player_ids_by_ft_percentage(season_id: int = Query(..., description="Season ID"), ft_percentage: float = Query(..., description="Free throw percentage")):
    cursor.execute("SELECT playerID FROM player_allstar WHERE season_id = ? AND CAST(ft_made AS REAL) * 100 / ft_attempted > ?", (season_id, ft_percentage))
    result = cursor.fetchall()
    if not result:
        return {"player_ids": []}
    return {"player_ids": result}

# Endpoint to get player IDs with a specific three-point percentage within a range of years
@app.get("/v1/professional_basketball/player_ids_by_three_point_percentage", operation_id="get_player_ids_by_three_point_percentage", summary="Retrieve the unique identifiers of professional basketball players who have achieved a specified three-point shooting percentage within a given range of years. The operation filters players based on their all-star appearances and team memberships during the specified period, ensuring that only those with the required shooting efficiency are included in the results.")
async def get_player_ids_by_three_point_percentage(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year"), three_point_percentage: float = Query(..., description="Three-point percentage")):
    cursor.execute("SELECT DISTINCT T2.playerID FROM player_allstar AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.year BETWEEN ? AND ? AND T1.three_made / T1.three_attempted > ?", (start_year, end_year, three_point_percentage))
    result = cursor.fetchall()
    if not result:
        return {"player_ids": []}
    return {"player_ids": result}

# Endpoint to get coach IDs for a specific team and league
@app.get("/v1/professional_basketball/coach_ids_by_team_league", operation_id="get_coach_ids_by_team_league", summary="Retrieves unique coach identifiers for a specific team and league. The operation filters coaches based on the provided team and league identifiers, ensuring that only coaches associated with the specified team and league are returned.")
async def get_coach_ids_by_team_league(tmID: str = Query(..., description="Team ID"), lgID: str = Query(..., description="League ID")):
    cursor.execute("SELECT DISTINCT T2.coachID FROM coaches AS T1 INNER JOIN awards_coaches AS T2 ON T1.coachID = T2.coachID WHERE T1.tmID = ? AND T1.lgID = ?", (tmID, lgID))
    result = cursor.fetchall()
    if not result:
        return {"coach_ids": []}
    return {"coach_ids": result}

# Endpoint to get the count of distinct coach IDs for a specific team and award
@app.get("/v1/professional_basketball/count_coach_ids_by_team_award", operation_id="get_count_coach_ids_by_team_award", summary="Retrieves the number of unique coaches who have been awarded for a specific team. The operation requires the team ID and the name of the award as input parameters.")
async def get_count_coach_ids_by_team_award(tmID: str = Query(..., description="Team ID"), award: str = Query(..., description="Award name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.coachID) FROM coaches AS T1 INNER JOIN awards_coaches AS T2 ON T1.coachID = T2.coachID WHERE T1.tmID = ? AND T2.award = ?", (tmID, award))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last name of the top all-star player based on points within a specified year range
@app.get("/v1/professional_basketball/top_allstar_player_by_points", operation_id="get_top_allstar_player_by_points", summary="Retrieves the full name of the top all-star player who scored the most points within the specified year range. The operation filters players based on their all-star appearances and awards, then ranks them by points scored in descending order. The player with the highest point total within the given year range is returned.")
async def get_top_allstar_player_by_points(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT T3.firstname, T3.lastname FROM player_allstar AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID INNER JOIN draft AS T3 ON T1.playerID = T3.playerID WHERE T2.year BETWEEN ? AND ? ORDER BY T1.points DESC LIMIT 1", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": {"firstname": result[0], "lastname": result[1]}}

# Endpoint to get the names of players from a specific college where offensive rebounds are greater than defensive rebounds
@app.get("/v1/professional_basketball/players_by_college_rebounds", operation_id="get_players_by_college_rebounds", summary="Retrieve the full names of professional basketball players from a specified college who have more offensive rebounds than defensive rebounds in their career.")
async def get_players_by_college_rebounds(college: str = Query(..., description="College name")):
    cursor.execute("SELECT T1.firstName, T1.middleName, T1.lastName FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.college = ? AND T2.o_rebounds > T2.d_rebounds", (college,))
    result = cursor.fetchall()
    if not result:
        return {"players": []}
    return {"players": [{"firstName": row[0], "middleName": row[1], "lastName": row[2]} for row in result]}

# Endpoint to get the top player by blocks from a specific birth city
@app.get("/v1/professional_basketball/top_player_by_blocks_birth_city", operation_id="get_top_player_by_blocks_birth_city", summary="Retrieves the top professional basketball player with the most blocks, who was born in a specified city. The response includes the player's first and last name, as well as their team ID.")
async def get_top_player_by_blocks_birth_city(birth_city: str = Query(..., description="Birth city of the player")):
    cursor.execute("SELECT T1.firstName, T1.lastName, T2.tmID FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T1.birthCity = ? ORDER BY T2.blocks DESC LIMIT 1", (birth_city,))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": {"firstName": result[0], "lastName": result[1], "tmID": result[2]}}

# Endpoint to get the names of teams with a specific rank and league ID within a specified year range
@app.get("/v1/professional_basketball/teams_by_rank_league_year_range", operation_id="get_teams_by_rank_league_year_range", summary="Retrieve the names of basketball teams that have a rank lower than the specified value and belong to a league with an ID greater than the provided league ID. The teams must have played within the defined year range.")
async def get_teams_by_rank_league_year_range(rank: int = Query(..., description="Rank of the team"), league_id: int = Query(..., description="League ID"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT T1.name FROM teams AS T1 INNER JOIN series_post AS T2 ON T1.tmID = T2.tmIDLoser AND T1.year = T2.year WHERE T1.rank < ? AND T2.lgIDLoser > ? AND T2.year BETWEEN ? AND ?", (rank, league_id, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": [{"name": row[0]} for row in result]}

# Endpoint to get the team ID with the highest points per minute within a specified year range
@app.get("/v1/professional_basketball/top_team_by_points_per_minute", operation_id="get_top_team_by_points_per_minute", summary="Retrieves the team with the highest points per minute ratio within the specified year range. This operation considers all teams' performance data within the provided year range and returns the team ID of the top performer based on the points-to-minutes ratio. The input parameters define the start and end years of the range.")
async def get_top_team_by_points_per_minute(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT tmID FROM players_teams WHERE year BETWEEN ? AND ? ORDER BY CAST(points AS REAL) / minutes DESC LIMIT 1", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": {"tmID": result[0]}}

# Endpoint to get the difference in average age of players drafted in ABA and NBA within a specified year range
@app.get("/v1/professional_basketball/average_age_difference_aba_nba", operation_id="get_average_age_difference_aba_nba", summary="Retrieve the average age difference between players drafted in the American Basketball Association (ABA) and the National Basketball Association (NBA) within a specified year range. The calculation considers the age of players at the time of drafting, based on their birth dates and the draft year. The year range is defined by the start_year and end_year input parameters.")
async def get_average_age_difference_aba_nba(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.lgID = 'ABA', 1970 - strftime('%Y', T3.birthDate), 0)) AS REAL) / COUNT(IIF(T2.lgID = 'ABA', 1, 0)) - CAST(SUM(IIF(T2.lgID = 'NBA', 1970 - strftime('%Y', T3.birthDate), 0)) AS REAL) / COUNT(IIF(T2.lgID = 'NBA', 1, 0)) FROM draft AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID INNER JOIN players AS T3 ON T2.playerID = T3.playerID WHERE T1.draftYear BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the player ID with the most games played in a specific year
@app.get("/v1/professional_basketball/top_player_by_games_played", operation_id="get_top_player_by_games_played", summary="Retrieves the ID of the professional basketball player who played the most games in a specified year. The year is provided as an input parameter.")
async def get_top_player_by_games_played(year: int = Query(..., description="Year")):
    cursor.execute("SELECT playerID FROM players_teams WHERE year = ? ORDER BY GP DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": {"playerID": result[0]}}

# Endpoint to get the count of all-star players drafted in a specific year and round
@app.get("/v1/professional_basketball/count_allstar_players_drafted", operation_id="get_count_allstar_players_drafted", summary="Retrieves the total number of professional basketball players who were drafted in a specific year and round and have been selected as all-stars. The count is based on the provided draft year and round.")
async def get_count_allstar_players_drafted(draft_year: int = Query(..., description="Draft year"), draft_round: int = Query(..., description="Draft round")):
    cursor.execute("SELECT COUNT(T2.playerID) FROM draft AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T1.draftYear = ? AND T1.draftRound = ?", (draft_year, draft_round))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the team ID of the Finals MVP award winner in a specific year
@app.get("/v1/professional_basketball/finals_mvp_team", operation_id="get_finals_mvp_team", summary="Retrieve the team ID of the player who won the specified award in the given year. The operation returns the team ID of the player who was awarded the Finals MVP in the provided year. The input parameters include the year and the name of the award.")
async def get_finals_mvp_team(year: int = Query(..., description="Year"), award: str = Query(..., description="Award name")):
    cursor.execute("SELECT DISTINCT T3.tmID FROM players_teams AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID INNER JOIN teams AS T3 ON T1.tmID = T3.tmID AND T1.year = T3.year WHERE T2.year = ? AND T2.award = ? LIMIT 1", (year, award))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": {"tmID": result[0]}}

# Endpoint to get the number of wins for the team with the most offensive field goals made in a specific year
@app.get("/v1/professional_basketball/top_team_wins_by_offensive_fgm", operation_id="get_top_team_wins_by_offensive_fgm", summary="Retrieves the number of wins for the team that made the most offensive field goals in a given year. The team is identified by comparing the offensive field goal statistics across all teams for the specified year.")
async def get_top_team_wins_by_offensive_fgm(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T2.W FROM teams AS T1 INNER JOIN series_post AS T2 ON T1.tmID = T2.tmIDLoser AND T1.year = T2.year WHERE T2.year = ? ORDER BY T1.o_fgm DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"wins": []}
    return {"wins": result[0]}

# Endpoint to get the average turnovers for players in a specific year
@app.get("/v1/professional_basketball/average_turnovers_by_year", operation_id="get_average_turnovers", summary="Retrieves the average number of turnovers per player for the specified year, ordered by the highest number of assists. The result is limited to the top player with the most assists.")
async def get_average_turnovers(year: int = Query(..., description="Year to filter the players")):
    cursor.execute("SELECT AVG(T2.turnovers) FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? GROUP BY T1.playerID, T2.assists ORDER BY T2.assists DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"average_turnovers": []}
    return {"average_turnovers": result[0]}

# Endpoint to get the count of all-star players with a specific first and last name
@app.get("/v1/professional_basketball/count_allstar_players_by_name", operation_id="get_count_allstar_players", summary="Retrieves the total number of professional basketball all-star players who share a specific first and last name. The operation requires both the first and last name of the player as input parameters to accurately determine the count.")
async def get_count_allstar_players(first_name: str = Query(..., description="First name of the player"), last_name: str = Query(..., description="Last name of the player")):
    cursor.execute("SELECT COUNT(T1.playerID) FROM player_allstar AS T1 INNER JOIN awards_players AS T2 ON T1.playerID = T2.playerID WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in win rates between two years for teams with a specific draft round and year
@app.get("/v1/professional_basketball/win_rate_difference_by_year", operation_id="get_win_rate_difference", summary="Retrieve the difference in win rates between two specified years for teams that were drafted in a particular round and year. This operation calculates the win rate for each year and subtracts the second year's win rate from the first year's win rate. The input parameters include the two years to compare, the draft round, and the draft year.")
async def get_win_rate_difference(year1: int = Query(..., description="First year to compare"), year2: int = Query(..., description="Second year to compare"), draft_round: int = Query(..., description="Draft round"), draft_year: int = Query(..., description="Draft year")):
    cursor.execute("SELECT (CAST(SUM(CASE WHEN T1.year = ? THEN T1.won ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.year = ? THEN T1.won + T1.lost ELSE 0 END)) - (CAST(SUM(CASE WHEN T1.year = ? THEN T1.won ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.year = ? THEN T1.won + T1.lost ELSE 0 END)) FROM teams AS T1 INNER JOIN draft AS T2 ON T1.tmID = T2.tmID WHERE T2.draftRound = ? AND T2.draftYear = ?", (year1, year1, year2, year2, draft_round, draft_year))
    result = cursor.fetchone()
    if not result:
        return {"win_rate_difference": []}
    return {"win_rate_difference": result[0]}

# Endpoint to get the count of coaches awarded in a specific range of years for a specific team
@app.get("/v1/professional_basketball/count_coaches_awarded_by_year_range", operation_id="get_count_coaches_awarded", summary="Retrieve the total number of coaches who received a specific award within a defined year range for a given team. The operation requires the start and end years of the range, the name of the award, and the team's unique identifier.")
async def get_count_coaches_awarded(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), award: str = Query(..., description="Award name"), team_id: str = Query(..., description="Team ID")):
    cursor.execute("SELECT COUNT(T1.id) FROM awards_coaches AS T1 INNER JOIN teams AS T2 ON T1.year = T2.year WHERE T1.year BETWEEN ? AND ? AND T1.award = ? AND T2.tmID = ?", (start_year, end_year, award, team_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of games with steals for a specific team
@app.get("/v1/professional_basketball/percentage_games_with_steals", operation_id="get_percentage_games_with_steals", summary="Retrieves the percentage of games in which a specific team has recorded at least one steal. The calculation is based on the total number of games played by the team and the number of games in which the team has registered steals. The team is identified by its unique ID.")
async def get_percentage_games_with_steals(team_id: str = Query(..., description="Team ID")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.steals IS NOT NULL AND T1.tmID = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.tmID) FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year", (team_id,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the player ID with the highest steals in a specific year with no post-season games
@app.get("/v1/professional_basketball/top_steals_player_no_postseason", operation_id="get_top_steals_player", summary="Retrieves the ID of the professional basketball player who recorded the highest number of steals in a given year, excluding post-season games. The operation filters players based on the provided year and ranks them by their steals count, returning the top player's ID.")
async def get_top_steals_player(year: int = Query(..., description="Year to filter the players")):
    cursor.execute("SELECT T1.playerID FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.year = ? AND T2.PostGP = 0 ORDER BY T2.steals DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"player_id": []}
    return {"player_id": result[0]}

# Endpoint to get the player ID with the highest turnovers in a specific year with no post-season games
@app.get("/v1/professional_basketball/top_turnovers_player_no_postseason", operation_id="get_top_turnovers_player", summary="Retrieves the ID of the player who had the most turnovers in a given year, excluding post-season games. The year parameter is used to filter the results.")
async def get_top_turnovers_player(year: int = Query(..., description="Year to filter the players")):
    cursor.execute("SELECT T2.playerID FROM players_teams AS T1 INNER JOIN players AS T2 ON T1.playerID = T2.playerID WHERE T1.PostGP = 0 AND T1.year = ? ORDER BY T1.turnovers DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"player_id": []}
    return {"player_id": result[0]}

# Endpoint to get the team ID with the highest win rate in a specific year with no post-season games
@app.get("/v1/professional_basketball/top_win_rate_team_no_postseason", operation_id="get_top_win_rate_team", summary="Retrieves the team identifier with the highest win rate for a given year, excluding teams that played post-season games. The win rate is calculated as the ratio of games won to the total number of games played.")
async def get_top_win_rate_team(year: int = Query(..., description="Year to filter the teams")):
    cursor.execute("SELECT T2.tmID FROM players_teams AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.PostGP = 0 AND T1.year = ? ORDER BY CAST(T2.won AS REAL) / (T2.won + T2.lost) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get the team ID with the highest points in a specific year with no post-season games
@app.get("/v1/professional_basketball/top_points_team_no_postseason", operation_id="get_top_points_team", summary="Retrieves the unique identifier of the team that scored the highest number of points in a given year, excluding teams that played any post-season games. The year is specified as an input parameter.")
async def get_top_points_team(year: int = Query(..., description="Year to filter the teams")):
    cursor.execute("SELECT T2.tmID FROM players_teams AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? AND T1.PostGP = 0 ORDER BY T1.points DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get the team ID with the highest wins with no post-season games
@app.get("/v1/professional_basketball/top_wins_team_no_postseason", operation_id="get_top_wins_team", summary="Retrieves the unique identifier of the team that has the highest number of wins without having played any post-season games. The data is obtained by joining the 'players_teams' and 'teams' tables on matching team IDs and years, filtering out teams that have played post-season games, and then sorting the remaining teams by the number of wins in descending order. The team with the highest number of wins is then selected.")
async def get_top_wins_team():
    cursor.execute("SELECT T2.tmID FROM players_teams AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.PostGP = 0 ORDER BY T2.won DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get the team ID of the player with the highest rebounds in a specific year and post-game points
@app.get("/v1/professional_basketball/team_id_highest_rebounds", operation_id="get_team_id_highest_rebounds", summary="Retrieves the team ID of the player who had the highest number of rebounds in a given year, considering only those who scored a specific number of points after the game.")
async def get_team_id_highest_rebounds(post_gp: int = Query(..., description="Post-game points"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T2.tmID FROM players_teams AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.PostGP = ? AND T1.year = ? ORDER BY T1.rebounds DESC LIMIT 1", (post_gp, year))
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get the team ID of a player drafted in a specific round, selection, and year
@app.get("/v1/professional_basketball/team_id_draft_details", operation_id="get_team_id_draft_details", summary="Retrieves the team ID of a player drafted in a specific round, selection, and year. The operation uses the provided draft round, selection, and year to identify the team that drafted the player. The team ID is returned, providing information about the team that made the draft selection.")
async def get_team_id_draft_details(draft_round: int = Query(..., description="Draft round"), draft_selection: int = Query(..., description="Draft selection"), draft_year: int = Query(..., description="Draft year")):
    cursor.execute("SELECT T2.tmID FROM draft AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.draftYear = T2.year WHERE T1.draftRound = ? AND T1.draftSelection = ? AND T1.draftYear = ?", (draft_round, draft_selection, draft_year))
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get the home wins of a team for a player drafted in a specific round, selection, and year
@app.get("/v1/professional_basketball/home_wins_draft_details", operation_id="get_home_wins_draft_details", summary="Retrieves the number of home games won by a team for a player drafted in a specific round, selection, and year. The operation uses the draft round, selection, and year to identify the player and team, and then fetches the corresponding home wins data.")
async def get_home_wins_draft_details(draft_round: int = Query(..., description="Draft round"), draft_selection: int = Query(..., description="Draft selection"), draft_year: int = Query(..., description="Draft year")):
    cursor.execute("SELECT T2.homeWon FROM draft AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.draftYear = T2.year WHERE T1.draftRound = ? AND T1.draftSelection = ? AND T1.draftYear = ?", (draft_round, draft_selection, draft_year))
    result = cursor.fetchone()
    if not result:
        return {"home_wins": []}
    return {"home_wins": result[0]}

# Endpoint to get the weight and height of the player with the highest rebounds in all-star games
@app.get("/v1/professional_basketball/player_weight_height_highest_rebounds", operation_id="get_player_weight_height_highest_rebounds", summary="Retrieves the weight and height of the player who has achieved the highest number of rebounds in all-star games.")
async def get_player_weight_height_highest_rebounds():
    cursor.execute("SELECT T1.weight, T1.height FROM players AS T1 INNER JOIN player_allstar AS T2 ON T1.playerID = T2.playerID ORDER BY T2.rebounds DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"weight": [], "height": []}
    return {"weight": result[0], "height": result[1]}

# Endpoint to get the high school of the player with the highest rebounds in all-star games
@app.get("/v1/professional_basketball/player_high_school_highest_rebounds", operation_id="get_player_high_school_highest_rebounds", summary="Retrieves the high school of the professional basketball player who has recorded the highest number of rebounds in all-star games.")
async def get_player_high_school_highest_rebounds():
    cursor.execute("SELECT T2.highSchool FROM player_allstar AS T1 INNER JOIN players AS T2 ON T1.playerID = T2.playerID ORDER BY T1.rebounds DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"high_school": []}
    return {"high_school": result[0]}

# Endpoint to get the team ID of the player with the highest rebounds in a specific all-star season
@app.get("/v1/professional_basketball/team_id_highest_rebounds_allstar_season", operation_id="get_team_id_highest_rebounds_allstar_season", summary="Retrieves the team ID of the player who recorded the highest number of rebounds in a specified all-star season. The operation filters players based on the provided all-star season ID and ranks them by their rebounds count in descending order. The team ID of the top-ranked player is then returned.")
async def get_team_id_highest_rebounds_allstar_season(season_id: int = Query(..., description="All-star season ID")):
    cursor.execute("SELECT T2.tmID FROM players_teams AS T1 INNER JOIN teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year INNER JOIN player_allstar AS T3 ON T3.playerID = T1.playerID WHERE T3.season_id = ? ORDER BY T1.rebounds DESC LIMIT 1", (season_id,))
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get the total points of players in a specific position
@app.get("/v1/professional_basketball/total_points_by_position", operation_id="get_total_points_by_position", summary="Retrieves the total points scored by players in a specific position, aggregated by year. The position is specified as a parameter. The result is sorted by year in descending order and limited to the most recent year.")
async def get_total_points_by_position(position: str = Query(..., description="Player position (e.g., 'C-F-G')")):
    cursor.execute("SELECT SUM(T2.points) FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T1.pos = ? GROUP BY T2.playerID, T2.year ORDER BY T2.year DESC LIMIT 1", (position,))
    result = cursor.fetchone()
    if not result:
        return {"total_points": []}
    return {"total_points": result[0]}

# Endpoint to get the team ID of the youngest player in a specific position and league
@app.get("/v1/professional_basketball/team_id_youngest_player_position_league", operation_id="get_team_id_youngest_player_position_league", summary="Retrieves the team ID of the youngest player in a specific position and league. The operation considers the provided position and league ID to filter the players and teams, then orders the results by birth date in descending order to identify the youngest player. The team ID of this player is returned.")
async def get_team_id_youngest_player_position_league(position: str = Query(..., description="Player position (e.g., 'F-G')"), league_id: str = Query(..., description="League ID (e.g., 'NBA')")):
    cursor.execute("SELECT T1.tmID FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year INNER JOIN players AS T3 ON T2.playerID = T3.playerID WHERE T3.pos = ? AND T2.lgID = ? ORDER BY T3.birthDate DESC LIMIT 1", (position, league_id))
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get the name of the player with the most games played in a specific league and college
@app.get("/v1/professional_basketball/player_name_most_games_league_college", operation_id="get_player_name_most_games_league_college", summary="Retrieve the full name of the player who has played the most games in a specific league and college. The operation requires the league ID, the number of games played, and the college name as input parameters. The player with the highest number of games played, matching the provided league and college, will be returned.")
async def get_player_name_most_games_league_college(league_id: str = Query(..., description="League ID (e.g., 'PBLA')"), games_played: int = Query(..., description="Number of games played"), college: str = Query(..., description="College name (e.g., 'Central Missouri State')")):
    cursor.execute("SELECT T1.firstName, T1.middleName, T1.lastName FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.lgID = ? AND T2.GP = ? AND T1.college = ? GROUP BY T1.firstName, T1.middleName, T1.lastName ORDER BY COUNT(T2.id) DESC LIMIT 1", (league_id, games_played, college))
    result = cursor.fetchone()
    if not result:
        return {"first_name": [], "middle_name": [], "last_name": []}
    return {"first_name": result[0], "middle_name": result[1], "last_name": result[2]}

# Endpoint to get the team ID with the least post minutes in a specific year and games played
@app.get("/v1/professional_basketball/team_id_least_post_minutes", operation_id="get_team_id_least_post_minutes", summary="Retrieves the team ID with the least cumulative post minutes played in a given year, based on a specified number of games played. The team ID is determined by summing the post minutes of all players on each team for the specified year and games played, then selecting the team with the lowest total.")
async def get_team_id_least_post_minutes(games_played: int = Query(..., description="Number of games played"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.tmID FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T2.GP = ? AND T2.year = ? GROUP BY T1.tmID ORDER BY SUM(T2.PostMinutes) ASC LIMIT 1", (games_played, year))
    result = cursor.fetchone()
    if not result:
        return {"team_id": []}
    return {"team_id": result[0]}

# Endpoint to get the player with the highest turnovers in a specific year and games played
@app.get("/v1/professional_basketball/player_highest_turnovers", operation_id="get_player_highest_turnovers", summary="Retrieves the professional basketball player with the highest number of turnovers in a specified year, given a minimum number of games played. The response includes the player's first, middle, and last names.")
async def get_player_highest_turnovers(gp: int = Query(..., description="Games played"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.firstName, T1.middleName, T1.lastName FROM players AS T1 INNER JOIN players_teams AS T2 ON T1.playerID = T2.playerID WHERE T2.GP = ? AND T2.year = ? ORDER BY T2.turnovers DESC LIMIT 1", (gp, year))
    result = cursor.fetchone()
    if not result:
        return {"player": []}
    return {"player": {"firstName": result[0], "middleName": result[1], "lastName": result[2]}}

# Endpoint to get the team with the highest steals in a specific year where games played equals games started
@app.get("/v1/professional_basketball/team_highest_steals", operation_id="get_team_highest_steals", summary="Retrieves the team with the highest number of steals in a given year, considering only those teams where the number of games played equals the number of games started.")
async def get_team_highest_steals(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.tmID FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? AND T2.GP = T2.GS GROUP BY T1.tmID, T2.steals ORDER BY T2.steals DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": {"tmID": result[0]}}

# Endpoint to get distinct team IDs where games played equals games started
@app.get("/v1/professional_basketball/distinct_teams_gp_equals_gs", operation_id="get_distinct_teams_gp_equals_gs", summary="Retrieves a unique list of team IDs from a specific year where the total number of games played by a team equals the number of games started by the team. This operation considers the games played and games started by each player on the team.")
async def get_distinct_teams_gp_equals_gs():
    cursor.execute("SELECT DISTINCT T1.tmID FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T2.GP = T2.GS")
    results = cursor.fetchall()
    if not results:
        return {"teams": []}
    return {"teams": [{"tmID": result[0]} for result in results]}

# Endpoint to get the birth date of the player with the highest total rebounds in a specific year
@app.get("/v1/professional_basketball/player_birthdate_highest_rebounds", operation_id="get_player_birthdate_highest_rebounds", summary="Retrieves the birth date of the professional basketball player who accumulated the highest total rebounds in a given year. The year is specified as an input parameter.")
async def get_player_birthdate_highest_rebounds(year: int = Query(..., description="Year")):
    cursor.execute("SELECT birthDate FROM players WHERE playerID = ( SELECT playerID FROM players_teams WHERE year = ? GROUP BY playerID ORDER BY SUM(rebounds + dRebounds) DESC LIMIT 1 )", (year,))
    result = cursor.fetchone()
    if not result:
        return {"birthDate": []}
    return {"birthDate": result[0]}

# Endpoint to get the team name with the highest total rebounds in a specific year
@app.get("/v1/professional_basketball/team_highest_rebounds", operation_id="get_team_highest_rebounds", summary="Retrieves the name of the team with the highest total rebounds (including defensive rebounds) in a specified year. The team is determined by aggregating the rebounds of all its players in the given year.")
async def get_team_highest_rebounds(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.name FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T2.year = ? GROUP BY T1.name ORDER BY SUM(rebounds + dRebounds) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"team": []}
    return {"team": {"name": result[0]}}

# Endpoint to get team IDs with rebounds greater than a specified value in a specific year
@app.get("/v1/professional_basketball/teams_rebounds_greater_than", operation_id="get_teams_rebounds_greater_than", summary="Retrieves the IDs of professional basketball teams that have more rebounds than a specified value in a given year. The operation filters teams based on the provided year and minimum rebounds count, and returns the team IDs that meet these criteria.")
async def get_teams_rebounds_greater_than(year: int = Query(..., description="Year"), rebounds: int = Query(..., description="Rebounds") ):
    cursor.execute("SELECT T1.tmID FROM teams AS T1 INNER JOIN players_teams AS T2 ON T1.tmID = T2.tmID AND T1.year = T2.year WHERE T1.year = ? AND T2.rebounds > ?", (year, rebounds))
    results = cursor.fetchall()
    if not results:
        return {"teams": []}
    return {"teams": [{"tmID": result[0]} for result in results]}

api_calls = [
    "/v1/professional_basketball/home_win_percentage",
    "/v1/professional_basketball/teams_lost_percentage?lost_percentage=75",
    "/v1/professional_basketball/teams_won_more_than_previous_year?previous_year=2004&current_year=2005",
    "/v1/professional_basketball/teams_home_win_loss_diff_percentage?win_loss_diff_percentage=80",
    "/v1/professional_basketball/percentage_teams_ranked_first",
    "/v1/professional_basketball/coach_details_year_team?year=1950&tmID=BOS",
    "/v1/professional_basketball/longest_stint_coach_year_range?start_year=1970&end_year=1980",
    "/v1/professional_basketball/coach_team_wins_year?year=2000&min_wins=50",
    "/v1/professional_basketball/coach_team_lost_more_year_range?start_year=2000&end_year=2010",
    "/v1/professional_basketball/team_details_coach_year_range?start_year=1990&end_year=1995&coachID=adelmri01",
    "/v1/professional_basketball/coach_details_by_win_percentage?win_percentage=90",
    "/v1/professional_basketball/player_count_by_birth_state_award_year_range?birth_state=NY&award=All-Defensive%20First%20Team&start_year=1980&end_year=2000",
    "/v1/professional_basketball/division_ids_by_award_year_team_year?award_year=1977&award=NBA%20Coach%20of%20the%20Year&team_year=1976",
    "/v1/professional_basketball/coach_ids_by_award_year_range_team_year_name?award=NBA%20Coach%20of%20the%20Year&start_year=1970&end_year=1979&team_year=1981&team_name=Chicago%20Bulls",
    "/v1/professional_basketball/player_nicknames_by_blocks_conference_season?blocks=2&conference=West&season_id=2006",
    "/v1/professional_basketball/winning_team_year_by_round_loser?round=DSF&tmIDLoser=HSM",
    "/v1/professional_basketball/player_birthdate_by_most_assists_season?season_id=1985",
    "/v1/professional_basketball/player_names_by_birth_city_season_range_minutes?birth_city=Winter%20Haven&start_season=1980&end_season=1989&minutes=12",
    "/v1/professional_basketball/player_count_by_conference_minutes_college?conference=East&minutes=5&college=Illinois",
    "/v1/professional_basketball/player_count_by_defensive_rebound_percentage_season_range?defensive_rebound_percentage=75&start_season=1990&end_season=2007",
    "/v1/professional_basketball/best_year_by_coach?coachID=costela01",
    "/v1/professional_basketball/total_minutes_by_city_nickname?birthCity=Brooklyn&nameNick=%25Superman%25",
    "/v1/professional_basketball/team_most_home_losses_by_year_award?year=1994&award=NBA%20Coach%20of%20the%20Year",
    "/v1/professional_basketball/distinct_team_ids_by_criteria?won=60&year=1996&round=CSF&tmIDLoser=LAL",
    "/v1/professional_basketball/league_ids_by_weight_criteria?weight_factor=0.4",
    "/v1/professional_basketball/coaches_by_team_count?team_count=2",
    "/v1/professional_basketball/top_coach_by_post_wins",
    "/v1/professional_basketball/coach_fewest_post_losses?lgID=NBA&team_count=2",
    "/v1/professional_basketball/player_count_by_league_position?lgID=ABA&pos1=C&pos2=F-C",
    "/v1/professional_basketball/distinct_first_names_by_league_position?pos1=F&pos2=F-C&lgID=NBA",
    "/v1/professional_basketball/count_players_hs_city_conference?hs_city=Chicago&conference=West",
    "/v1/professional_basketball/distinct_colleges_conference?conference=East",
    "/v1/professional_basketball/count_players_birth_city_league?birth_city=Spencer&lgID=NBL",
    "/v1/professional_basketball/birth_date_most_awards?award=Most%20Valuable%20Player",
    "/v1/professional_basketball/count_players_award_birth_city?award=Most%20Valuable%20Player&birth_city=Houston",
    "/v1/professional_basketball/tallest_player_award?award=Rookie%20of%20the%20Year",
    "/v1/professional_basketball/avg_weight_award?award=Rookie%20of%20the%20Year",
    "/v1/professional_basketball/avg_height_award_birth_city?award=Most%20Valuable%20Player&birth_city=New%20York",
    "/v1/professional_basketball/top_teams_by_post_points?year=2000",
    "/v1/professional_basketball/teams_below_win_percentage?win_percentage=50",
    "/v1/professional_basketball/coach_ids_winning_percentage?winning_percentage=80",
    "/v1/professional_basketball/coach_ids_tenure_league?lgID=NBA&tenure_years=10",
    "/v1/professional_basketball/count_teams_award_points?award=Most%20Valuable%20Player&min_points=3800",
    "/v1/professional_basketball/tallest_player_team?tmID=AFS",
    "/v1/professional_basketball/player_last_names_team?tmID=BLB",
    "/v1/professional_basketball/count_coach_ids_year_range?start_year=1962&end_year=1975",
    "/v1/professional_basketball/coach_ids_awards_year_range?start_year=1970&end_year=1990&award_count=2",
    "/v1/professional_basketball/count_coach_ids_specific_awards_year_range?start_year=1962&end_year=2011&award1=ABA%20Coach%20of%20the%20Year&award2=NBA%20Coach%20of%20the%20Year",
    "/v1/professional_basketball/avg_points_season?season_id=1975",
    "/v1/professional_basketball/player_names_height?min_height=75",
    "/v1/professional_basketball/min_weight_by_college?college=UCLA",
    "/v1/professional_basketball/max_weight_by_birth_country?birth_country=USA",
    "/v1/professional_basketball/total_points_by_season_range?start_season=1960&end_season=1970&death_date=0000-00-00",
    "/v1/professional_basketball/distinct_players_by_birthdate_and_rebounds?birth_date=1950&rebound_percentage=30",
    "/v1/professional_basketball/count_players_by_award_and_year_range?start_year=1969&end_year=2010&award=Rookie%20of%20the%20Year",
    "/v1/professional_basketball/team_ids_by_min_allstar_players?min_allstar_players=3",
    "/v1/professional_basketball/max_points_by_year_range_and_rank?start_year=1950&end_year=1970&rank=1",
    "/v1/professional_basketball/players_from_top_teams_by_year_range?start_year=1937&end_year=1940&rank=1",
    "/v1/professional_basketball/top_team_by_year_range_and_rank?rank=3&start_year=1937&end_year=1940",
    "/v1/professional_basketball/count_teams_by_min_points_and_year?year=1937&min_points=500",
    "/v1/professional_basketball/count_distinct_players_win_percentage_year?win_percentage=75&year=1990",
    "/v1/professional_basketball/players_by_race_height?race=B&height=0&limit=3",
    "/v1/professional_basketball/count_drafted_players_by_name_year?first_name=Joe&draft_year=1970",
    "/v1/professional_basketball/count_fg_made_by_player_season_range?first_name=George&last_name=Mikan&start_season=1951&end_season=1953",
    "/v1/professional_basketball/min_max_bmi",
    "/v1/professional_basketball/team_highest_home_win_percentage?limit=1",
    "/v1/professional_basketball/player_age_at_draft?first_name=Alexis&last_name=Ajinca&draft_round=1",
    "/v1/professional_basketball/tallest_player_team_year?team_name=Denver%20Nuggets&year=1980&limit=1",
    "/v1/professional_basketball/most_awarded_deceased_player?limit=1",
    "/v1/professional_basketball/team_most_players_from_college?college=UCLA&limit=1",
    "/v1/professional_basketball/average_bmi_allstar_players",
    "/v1/professional_basketball/team_highest_win_rate_improvement?lgID_prev=ABA&year_prev=1972&lgID_current=ABA&year_current=1973",
    "/v1/professional_basketball/player_most_personal_fouls?lgID=NBL",
    "/v1/professional_basketball/average_height_allstar_players?conference=East",
    "/v1/professional_basketball/coach_most_wins_award?award=ABA%20Coach%20of%20the%20Year",
    "/v1/professional_basketball/team_name_award_year?year=1992&award=NBA%20Coach%20of%20the%20Year",
    "/v1/professional_basketball/player_highest_fg_percentage?year=1973",
    "/v1/professional_basketball/team_name_drafted_player?first_name=Mike&last_name=Lynn",
    "/v1/professional_basketball/count_distinct_players_award_year_range?award=Most%20Improved%20Player&birth_country=USA&start_year=1985&end_year=1990",
    "/v1/professional_basketball/distinct_allstar_players_college?college=California",
    "/v1/professional_basketball/count_coaches_awards_between_years?start_year=1950&end_year=1970&min_awards=1",
    "/v1/professional_basketball/count_players_award_between_years?start_year=1969&end_year=1975&award=Most%20Valuable%20Player",
    "/v1/professional_basketball/distinct_team_names_by_birth_state_count?min_count=5",
    "/v1/professional_basketball/count_teams_allstar_players_league?league_id=NBA&min_allstars=3",
    "/v1/professional_basketball/top_birth_states_allstar_players?limit=1",
    "/v1/professional_basketball/count_players_team_rank_year?year=1937&rank=6",
    "/v1/professional_basketball/count_players_win_percentage?win_percentage=20",
    "/v1/professional_basketball/players_born_outside_country?birth_country=USA",
    "/v1/professional_basketball/top_coach_win_loss_year?year=1988&limit=1",
    "/v1/professional_basketball/distinct_team_ids_coach_award_year?year=2010&award=NBA%20Coach%20of%20the%20Year",
    "/v1/professional_basketball/players_free_throws?ft_attempted=0",
    "/v1/professional_basketball/players_drafted_from_city?draft_from=Seattle&start_year=1965&end_year=1970",
    "/v1/professional_basketball/players_award_year?award=Finals%20MVP&year=2003",
    "/v1/professional_basketball/teams_finals_winners?round=F&start_year=1950&end_year=1960",
    "/v1/professional_basketball/coaches_winning_teams?round=QF&year=1946",
    "/v1/professional_basketball/players_winning_team?year=1970&round=F",
    "/v1/professional_basketball/most_awards_player_college?year=1970",
    "/v1/professional_basketball/youngest_award_winner?award=Rookie%20of%20the%20Year",
    "/v1/professional_basketball/players_drafted_by_round_country?draft_round=1&birth_country=USA&draft_year=1973",
    "/v1/professional_basketball/offensive_rebounds_percentage?year=2000",
    "/v1/professional_basketball/coaches_win_rate?win_rate=0.75",
    "/v1/professional_basketball/coaches_by_team?team_name=Oklahoma%20City%20Thunder",
    "/v1/professional_basketball/player_count_orebounds_year?orebound_percentage=50&year=1990",
    "/v1/professional_basketball/allstar_player_count?start_season=2000&end_season=2005&max_steals=10",
    "/v1/professional_basketball/player_ids_by_award_draft?award=Rookie%20of%20the%20Year&draft_year=1971&draft_round=2",
    "/v1/professional_basketball/allstar_player_count_by_season_race?season_id=1973&race=B",
    "/v1/professional_basketball/team_ids_by_series_year_round_dpts?year=1947&round=QF&d_pts=3513",
    "/v1/professional_basketball/team_loss_percentage?team_name=Houston%20Mavericks",
    "/v1/professional_basketball/player_ids_by_award_year_league?year=1990&award=Most%20Valuable%20Player&lgID=NBA",
    "/v1/professional_basketball/player_years_count_by_team_year_range?tmID=LAL&start_year=1975&end_year=1980&playerID=abdulka01",
    "/v1/professional_basketball/award_percentage_by_year?award=NBA%20Coach%20of%20the%20Year&year=1969",
    "/v1/professional_basketball/win_percentage_difference?year1=1947&year2=1946&tmIDWinner=CHS",
    "/v1/professional_basketball/award_count_by_high_school_city?year=2010&hsCity=Chicago",
    "/v1/professional_basketball/award_percentage_by_college?award=All-Defensive%20Second%20Team&college=Auburn",
    "/v1/professional_basketball/top_players_by_steals?season_id=1997",
    "/v1/professional_basketball/colleges_by_award_year?year=1990&award=Finals%20MVP",
    "/v1/professional_basketball/player_ids_by_ft_percentage?season_id=1996&ft_percentage=70",
    "/v1/professional_basketball/player_ids_by_three_point_percentage?start_year=1980&end_year=1983&three_point_percentage=0.6",
    "/v1/professional_basketball/coach_ids_by_team_league?tmID=STL&lgID=NBA",
    "/v1/professional_basketball/count_coach_ids_by_team_award?tmID=CHI&award=NBA%20Coach%20of%20the%20Year",
    "/v1/professional_basketball/top_allstar_player_by_points?start_year=1990&end_year=2000",
    "/v1/professional_basketball/players_by_college_rebounds?college=Wake%20Forest",
    "/v1/professional_basketball/top_player_by_blocks_birth_city?birth_city=Atlanta",
    "/v1/professional_basketball/teams_by_rank_league_year_range?rank=5&league_id=2&start_year=1980&end_year=2000",
    "/v1/professional_basketball/top_team_by_points_per_minute?start_year=1991&end_year=2000",
    "/v1/professional_basketball/average_age_difference_aba_nba?start_year=1970&end_year=1970",
    "/v1/professional_basketball/top_player_by_games_played?year=2011",
    "/v1/professional_basketball/count_allstar_players_drafted?draft_year=1996&draft_round=1",
    "/v1/professional_basketball/finals_mvp_team?year=1997&award=Finals%20MVP",
    "/v1/professional_basketball/top_team_wins_by_offensive_fgm?year=2001",
    "/v1/professional_basketball/average_turnovers_by_year?year=2003",
    "/v1/professional_basketball/count_allstar_players_by_name?first_name=Ray&last_name=Allen",
    "/v1/professional_basketball/win_rate_difference_by_year?year1=2004&year2=2003&draft_round=1&draft_year=2003",
    "/v1/professional_basketball/count_coaches_awarded_by_year_range?start_year=1971&end_year=1975&award=NBA%20Coach%20of%20the%20Year&team_id=POR",
    "/v1/professional_basketball/percentage_games_with_steals?team_id=LAL",
    "/v1/professional_basketball/top_steals_player_no_postseason?year=1996",
    "/v1/professional_basketball/top_turnovers_player_no_postseason?year=1988",
    "/v1/professional_basketball/top_win_rate_team_no_postseason?year=2000",
    "/v1/professional_basketball/top_points_team_no_postseason?year=1998",
    "/v1/professional_basketball/top_wins_team_no_postseason",
    "/v1/professional_basketball/team_id_highest_rebounds?post_gp=0&year=1997",
    "/v1/professional_basketball/team_id_draft_details?draft_round=1&draft_selection=6&draft_year=1976",
    "/v1/professional_basketball/home_wins_draft_details?draft_round=1&draft_selection=12&draft_year=1998",
    "/v1/professional_basketball/player_weight_height_highest_rebounds",
    "/v1/professional_basketball/player_high_school_highest_rebounds",
    "/v1/professional_basketball/team_id_highest_rebounds_allstar_season?season_id=1997",
    "/v1/professional_basketball/total_points_by_position?position=C-F-G",
    "/v1/professional_basketball/team_id_youngest_player_position_league?position=F-G&league_id=NBA",
    "/v1/professional_basketball/player_name_most_games_league_college?league_id=PBLA&games_played=10&college=Central%20Missouri%20State",
    "/v1/professional_basketball/team_id_least_post_minutes?games_played=82&year=2000",
    "/v1/professional_basketball/player_highest_turnovers?gp=82&year=1995",
    "/v1/professional_basketball/team_highest_steals?year=2011",
    "/v1/professional_basketball/distinct_teams_gp_equals_gs",
    "/v1/professional_basketball/player_birthdate_highest_rebounds?year=2001",
    "/v1/professional_basketball/team_highest_rebounds?year=1997",
    "/v1/professional_basketball/teams_rebounds_greater_than?year=2011&rebounds=600"
]
