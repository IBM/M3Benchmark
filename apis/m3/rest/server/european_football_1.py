from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/european_football_1/european_football_1.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of matches based on season, away team, and full-time result
@app.get("/v1/european_football_1/match_count_season_away_team_ftr", operation_id="get_match_count", summary="Retrieve the total number of matches played in a given season, by a specific away team, and with a particular full-time result. This operation provides a quantitative overview of matches based on the provided criteria.")
async def get_match_count(season: int = Query(..., description="Season of the match"), away_team: str = Query(..., description="Away team name"), ftr: str = Query(..., description="Full-time result (e.g., 'H', 'A', 'D')")):
    cursor.execute("SELECT COUNT(*) FROM matchs WHERE season = ? AND AwayTeam = ? AND FTR = ?", (season, away_team, ftr))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of divisions in a specific country
@app.get("/v1/european_football_1/division_percentage_by_country", operation_id="get_division_percentage", summary="Retrieves the percentage of football divisions located in a specified European country. The calculation is based on the total number of divisions in the database.")
async def get_division_percentage(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN country = ? THEN division ELSE NULL END) AS REAL) * 100 / COUNT(division) FROM divisions", (country,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of home, away, and draw results for a specific home team
@app.get("/v1/european_football_1/match_result_percentages_by_home_team", operation_id="get_match_result_percentages", summary="Retrieves the percentage of home, away, and draw results for a given home team in European football matches. The home team is specified as an input parameter.")
async def get_match_result_percentages(home_team: str = Query(..., description="Home team name")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN FTR = 'H' THEN 1 ELSE NULL END) / COUNT(HomeTeam) AS REAL) * 100, CAST(COUNT(CASE WHEN FTR = 'A' THEN 1 ELSE NULL END) AS REAL) / COUNT(HomeTeam), CAST(COUNT(CASE WHEN FTR = 'D' THEN 1 ELSE NULL END) AS REAL) / COUNT(HomeTeam) FROM matchs WHERE HomeTeam = ?", (home_team,))
    result = cursor.fetchone()
    if not result:
        return {"percentages": []}
    return {"home_win_percentage": result[0], "away_win_percentage": result[1], "draw_percentage": result[2]}

# Endpoint to get the most frequent away team for a specific home team, season, and full-time result
@app.get("/v1/european_football_1/most_frequent_away_team", operation_id="get_most_frequent_away_team", summary="Retrieves the away team that has played the most matches against a specific home team in a given season, considering only matches with a certain full-time result. The response will contain the name of the most frequent away team.")
async def get_most_frequent_away_team(home_team: str = Query(..., description="Home team name"), season: int = Query(..., description="Season of the match"), ftr: str = Query(..., description="Full-time result (e.g., 'H', 'A', 'D')")):
    cursor.execute("SELECT AwayTeam FROM matchs WHERE HomeTeam = ? AND season = ? AND FTR = ? GROUP BY AwayTeam ORDER BY COUNT(AwayTeam) DESC LIMIT 1", (home_team, season, ftr))
    result = cursor.fetchone()
    if not result:
        return {"away_team": []}
    return {"away_team": result[0]}

# Endpoint to get the percentage of matches in a specific division on a given date
@app.get("/v1/european_football_1/division_percentage_by_date", operation_id="get_division_percentage_by_date", summary="Retrieves the percentage of matches played in a specific division on a given date. The division is identified by its code, and the date is provided in 'YYYY-MM-DD' format. This endpoint calculates the percentage by dividing the number of matches in the specified division on the given date by the total number of matches on that date.")
async def get_division_percentage_by_date(division: str = Query(..., description="Division code (e.g., 'F1')"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Div = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(Div) FROM matchs WHERE Date = ?", (division, date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of matches involving a specific team with a given full-time result
@app.get("/v1/european_football_1/team_match_percentage_by_ftr", operation_id="get_team_match_percentage_by_ftr", summary="Retrieve the percentage of matches in which a specified team has been involved and resulted in a given full-time outcome. The full-time result can be a home win ('H'), an away win ('A'), or a draw ('D').")
async def get_team_match_percentage_by_ftr(team: str = Query(..., description="Team name"), ftr: str = Query(..., description="Full-time result (e.g., 'H', 'A', 'D')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN HomeTeam = ? OR AwayTeam = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(FTR) FROM matchs WHERE FTR = ?", (team, team, ftr))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of matches with a specific away team goal count in a given season
@app.get("/v1/european_football_1/away_team_goal_percentage_by_season", operation_id="get_away_team_goal_percentage", summary="Retrieve the percentage of matches in a specified season where the away team scored a certain number of goals. This operation provides insights into the performance of away teams in a given season, based on their goal count.")
async def get_away_team_goal_percentage(goals: int = Query(..., description="Number of away team goals"), season: int = Query(..., description="Season of the match")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN FTAG = ? THEN 1 ELSE 0 END) / COUNT(FTAG) AS REAL) * 100 FROM matchs WHERE season = ?", (goals, season))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get home and away teams for matches in a specific division and league name
@app.get("/v1/european_football_1/teams_by_division_and_league", operation_id="get_teams_by_division_and_league", summary="Retrieves the home and away teams for matches in a specified division and league. The division is identified by its code, while the league is identified by its name. This operation provides a comprehensive view of the teams participating in matches within the specified division and league.")
async def get_teams_by_division_and_league(league_name: str = Query(..., description="League name"), division: str = Query(..., description="Division code (e.g., 'E2')")):
    cursor.execute("SELECT T1.HomeTeam, T1.AwayTeam FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.Div = ?", (league_name, division))
    result = cursor.fetchall()
    if not result:
        return {"teams": []}
    return {"teams": result}

# Endpoint to get the count of distinct teams with a minimum number of goals in a specific country
@app.get("/v1/european_football_1/distinct_teams_goal_count_by_country", operation_id="get_distinct_teams_goal_count", summary="Retrieve the total count of unique football teams that have scored a minimum number of goals in a specified country. The operation considers both home and away matches.")
async def get_distinct_teams_goal_count(min_goals: int = Query(..., description="Minimum number of goals"), country: str = Query(..., description="Country name")):
    cursor.execute("SELECT COUNT(DISTINCT CASE WHEN T1.FTHG >= ? THEN HomeTeam ELSE NULL end) + COUNT(DISTINCT CASE WHEN T1.FTAG >= ? THEN AwayTeam ELSE NULL end) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.country = ?", (min_goals, min_goals, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches in a specific division, season, and with specific full-time scores
@app.get("/v1/european_football_1/match_count_by_division_season_scores", operation_id="get_match_count_by_division_season_scores", summary="Retrieve the total number of matches played in a specific football division and season, with a specific full-time score. The score is determined by the number of goals scored by the home and away teams. This operation allows you to analyze the frequency of particular scorelines in a given division and season.")
async def get_match_count_by_division_season_scores(season: int = Query(..., description="Season of the match"), division_name: str = Query(..., description="Division name"), away_goals: int = Query(..., description="Number of away team goals"), home_goals: int = Query(..., description="Number of home team goals")):
    cursor.execute("SELECT COUNT(T1.Div) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.season = ? AND T2.name = ? AND T1.FTAG = ? AND T1.FTHG = ?", (season, division_name, away_goals, home_goals))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get home teams from matches in a specific country with a specific full-time home goal count
@app.get("/v1/european_football_1/home_teams_by_country_and_goals", operation_id="get_home_teams_by_country_and_goals", summary="Retrieves the list of home teams that have played matches in a specified country and scored a certain number of full-time home goals. The operation filters matches based on the provided country and full-time home goal count, and returns the corresponding home teams.")
async def get_home_teams_by_country_and_goals(country: str = Query(..., description="Country of the division"), fthg: int = Query(..., description="Full-time home goal count")):
    cursor.execute("SELECT T1.HomeTeam FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.country = ? AND T1.FTHG = ?", (country, fthg))
    result = cursor.fetchall()
    if not result:
        return {"home_teams": []}
    return {"home_teams": [row[0] for row in result]}

# Endpoint to get home team win percentage in a specific division, country, and season
@app.get("/v1/european_football_1/home_team_win_percentage", operation_id="get_home_team_win_percentage", summary="Retrieves the home team win percentage for a specific division, country, and season. The calculation is based on the number of matches where the home team won, divided by the total number of matches played in the specified division, country, and season.")
async def get_home_team_win_percentage(division_name: str = Query(..., description="Name of the division"), country: str = Query(..., description="Country of the division"), season: int = Query(..., description="Season of the matches")):
    cursor.execute("SELECT T1.HomeTeam HWHT , CAST(COUNT(CASE WHEN T1.FTR = 'H' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(HomeTeam) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T2.country = ? AND T1.season = ?", (division_name, country, season))
    result = cursor.fetchall()
    if not result:
        return {"win_percentage": []}
    return {"win_percentage": [{"home_team": row[0], "percentage": row[1]} for row in result]}

# Endpoint to get the number and percentage of matches in a specific country and season
@app.get("/v1/european_football_1/matches_count_and_percentage", operation_id="get_matches_count_and_percentage", summary="Retrieves the total number and percentage of football matches played in a specific country and season. The data is grouped by division, providing a breakdown of matches per division within the specified country and season.")
async def get_matches_count_and_percentage(country: str = Query(..., description="Country of the division"), season: int = Query(..., description="Season of the matches")):
    cursor.execute("SELECT ( SELECT COUNT(T1.Div) AS total FROM matchs T1 INNER JOIN divisions T2 ON T2.division = T1.Div WHERE T2.country = ? AND T1.season = ? ) AS num , CASE WHEN 1 THEN T.result END AS percentage FROM ( SELECT 100.0 * COUNT(T1.Div) / ( SELECT COUNT(T1.Div) FROM matchs T1 INNER JOIN divisions T2 ON T2.division = T1.Div WHERE T2.country = ? AND T1.season = ? ) AS result FROM matchs T1 INNER JOIN divisions T2 ON T2.division = T1.Div WHERE T2.country = ? AND T1.season = ? GROUP BY T2.division ) AS T", (country, season, country, season, country, season))
    result = cursor.fetchall()
    if not result:
        return {"matches_info": []}
    return {"matches_info": [{"num": row[0], "percentage": row[1]} for row in result]}

# Endpoint to get away teams with the maximum goals in a specific season
@app.get("/v1/european_football_1/away_teams_max_goals", operation_id="get_away_teams_max_goals", summary="Get away teams with the maximum goals in a specific season")
async def get_away_teams_max_goals(season: int = Query(..., description="Season of the matches")):
    cursor.execute("SELECT ( SELECT MAX(MAX(FTAG), MAX(FTHG)) FROM matchs WHERE season = ? ) AS T1, AwayTeam FROM matchs WHERE season = ? AND FTHG = T1 OR FTAG = T1", (season, season))
    result = cursor.fetchall()
    if not result:
        return {"away_teams": []}
    return {"away_teams": [row[1] for row in result]}

# Endpoint to get the home team with the highest full-time home goals in a specific division and season
@app.get("/v1/european_football_1/top_home_team_by_goals", operation_id="get_top_home_team_by_goals", summary="Retrieve the home team that scored the most full-time home goals in a given division and season. The division and season are specified as input parameters.")
async def get_top_home_team_by_goals(division: str = Query(..., description="Division code"), season: int = Query(..., description="Season of the matches")):
    cursor.execute("SELECT HomeTeam FROM matchs WHERE Div = ? AND season = ? ORDER BY FTHG DESC LIMIT 1", (division, season))
    result = cursor.fetchone()
    if not result:
        return {"home_team": []}
    return {"home_team": result[0]}

# Endpoint to get the difference in home and away win percentages in a specific season
@app.get("/v1/european_football_1/win_percentage_difference", operation_id="get_win_percentage_difference", summary="Retrieves the difference in home and away win percentages for a specified season. The calculation is based on the total number of home and away wins divided by the total number of matches played in the given season.")
async def get_win_percentage_difference(season: int = Query(..., description="Season of the matches")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN FTR = 'H' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(FTR) - CAST(COUNT(CASE WHEN FTR = 'A' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(FTR) DIFFERENCE FROM matchs WHERE season = ?", (season,))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the division with the most draws in a specific season
@app.get("/v1/european_football_1/top_division_by_draws", operation_id="get_top_division_by_draws", summary="Retrieves the top division with the highest number of draws in a specified season. The division is determined by analyzing matches from the given season where the final result (FTR) is a draw (D). The results are grouped by division and ordered in descending order based on the count of draws. Only the top division is returned.")
async def get_top_division_by_draws(season: int = Query(..., description="Season of the matches")):
    cursor.execute("SELECT Div FROM matchs WHERE season = ? AND FTR = 'D' GROUP BY Div ORDER BY COUNT(FTR) DESC LIMIT 1", (season,))
    result = cursor.fetchone()
    if not result:
        return {"division": []}
    return {"division": result[0]}

# Endpoint to get home teams that won on a specific date in a specific division
@app.get("/v1/european_football_1/home_teams_won_on_date", operation_id="get_home_teams_won_on_date", summary="Retrieves the list of home teams that won their matches on a specific date in a given division. The division is identified by its code, and the date is provided in 'YYYY-MM-DD' format.")
async def get_home_teams_won_on_date(division: str = Query(..., description="Division code"), date: str = Query(..., description="Date of the match in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT HomeTeam FROM matchs WHERE Div = ? AND Date = ? AND FTR = 'H'", (division, date))
    result = cursor.fetchall()
    if not result:
        return {"home_teams": []}
    return {"home_teams": [row[0] for row in result]}

# Endpoint to get division names for matches on a specific date with specific home and away teams
@app.get("/v1/european_football_1/division_names_by_match_details", operation_id="get_division_names_by_match_details", summary="Retrieves the division names for football matches that took place on a specific date and involved the specified home and away teams. The operation filters matches based on the provided date, home team, and away team, and returns the corresponding division names.")
async def get_division_names_by_match_details(date: str = Query(..., description="Date of the match in 'YYYY-MM-DD' format"), home_team: str = Query(..., description="Home team name"), away_team: str = Query(..., description="Away team name")):
    cursor.execute("SELECT T2.name FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.Date = ? AND T1.HomeTeam = ? AND T1.AwayTeam = ?", (date, home_team, away_team))
    result = cursor.fetchall()
    if not result:
        return {"division_names": []}
    return {"division_names": [row[0] for row in result]}

# Endpoint to get the count of matches in a specific division within a range of seasons
@app.get("/v1/european_football_1/match_count_by_division_and_season_range", operation_id="get_match_count_by_division_and_season_range", summary="Retrieves the total number of matches played in a specified division within a given range of seasons. The division is identified by its name, and the range is defined by the start and end seasons.")
async def get_match_count_by_division_and_season_range(division_name: str = Query(..., description="Name of the division"), start_season: int = Query(..., description="Start season of the range"), end_season: int = Query(..., description="End season of the range")):
    cursor.execute("SELECT COUNT(T1.Div) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND (T1.season BETWEEN ? AND ?)", (division_name, start_season, end_season))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct divisions and countries for matches between specific home and away teams
@app.get("/v1/european_football_1/divisions_countries_home_away_teams", operation_id="get_divisions_countries_home_away_teams", summary="Retrieve unique combinations of divisions and countries where matches between the specified home and away teams have occurred. This operation provides a comprehensive overview of the leagues and nations in which the given teams have competed against each other.")
async def get_divisions_countries_home_away_teams(home_team: str = Query(..., description="Home team name"), away_team: str = Query(..., description="Away team name")):
    cursor.execute("SELECT DISTINCT T2.division, T2.country FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.HomeTeam = ? AND T1.AwayTeam = ?", (home_team, away_team))
    result = cursor.fetchall()
    if not result:
        return {"divisions_countries": []}
    return {"divisions_countries": result}

# Endpoint to get the away team with the highest FTAG in a specific division
@app.get("/v1/european_football_1/away_team_highest_ftag_division", operation_id="get_away_team_highest_ftag_division", summary="Retrieves the name of the away team that has scored the highest number of goals in a specific division. The division is identified by its name, which is provided as an input parameter.")
async def get_away_team_highest_ftag_division(division_name: str = Query(..., description="Division name")):
    cursor.execute("SELECT T1.AwayTeam FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? ORDER BY T1.FTAG DESC LIMIT 1", (division_name,))
    result = cursor.fetchone()
    if not result:
        return {"away_team": []}
    return {"away_team": result[0]}

# Endpoint to get the away teams for matches in a specific country
@app.get("/v1/european_football_1/away_teams_country", operation_id="get_away_teams_country", summary="Retrieves the names of the away teams that have played matches in a specific country. The operation returns a maximum of three team names. The country is specified as an input parameter.")
async def get_away_teams_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT T1.AwayTeam FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.country = ? LIMIT 3", (country,))
    result = cursor.fetchall()
    if not result:
        return {"away_teams": []}
    return {"away_teams": result}

# Endpoint to get the division with the most draws in a specific season
@app.get("/v1/european_football_1/division_most_draws_season", operation_id="get_division_most_draws_season", summary="Retrieves the name of the division with the highest number of draws in a specified season. The operation considers full-time results to identify draws and returns the division with the most occurrences of the specified result. The input parameters include the season year and the full-time result code for a draw.")
async def get_division_most_draws_season(season: int = Query(..., description="Season year"), ftr: str = Query(..., description="Full-time result (e.g., 'D' for draw)")):
    cursor.execute("SELECT T2.name FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.season = ? AND T1.FTR = ? GROUP BY T2.division ORDER BY COUNT(FTR) LIMIT 1", (season, ftr))
    result = cursor.fetchone()
    if not result:
        return {"division": []}
    return {"division": result[0]}

# Endpoint to get the count of home matches for a specific team in a specific division with a specific result
@app.get("/v1/european_football_1/count_home_matches_team_division_result", operation_id="get_count_home_matches_team_division_result", summary="Retrieves the total number of home matches played by a specific team in a given division that ended with a particular full-time result. The division, team, and full-time result are provided as input parameters.")
async def get_count_home_matches_team_division_result(division_name: str = Query(..., description="Division name"), home_team: str = Query(..., description="Home team name"), ftr: str = Query(..., description="Full-time result (e.g., 'H' for home win)")):
    cursor.execute("SELECT COUNT(T1.HomeTeam) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.HomeTeam = ? AND T1.FTR = ?", (division_name, home_team, ftr))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches with a specific result in a specific division
@app.get("/v1/european_football_1/count_matches_result_division", operation_id="get_count_matches_result_division", summary="Retrieves the total number of matches in a specified division that ended with a particular full-time result. The division is identified by its name, and the full-time result is represented by a single character code (e.g., 'D' for a draw).")
async def get_count_matches_result_division(division_name: str = Query(..., description="Division name"), ftr: str = Query(..., description="Full-time result (e.g., 'D' for draw)")):
    cursor.execute("SELECT COUNT(T1.FTR) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.FTR = ?", (division_name, ftr))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of divisions in a specific country
@app.get("/v1/european_football_1/count_divisions_country", operation_id="get_count_divisions_country", summary="Retrieves the total number of divisions in a specified country. The operation requires the country name as input and returns the count of divisions associated with that country.")
async def get_count_divisions_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT COUNT(division) FROM divisions WHERE country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of divisions in a specific country
@app.get("/v1/european_football_1/division_names_country", operation_id="get_division_names_country", summary="Retrieves the names of all football divisions in a specified country. The operation requires the country name as input and returns a list of division names that match the provided country.")
async def get_division_names_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT name FROM divisions WHERE country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"division_names": []}
    return {"division_names": result}

# Endpoint to get the winner of a match based on the full-time result
@app.get("/v1/european_football_1/match_winner", operation_id="get_match_winner", summary="Retrieves the winner of a football match between two specified teams on a given date. The result is determined by the full-time score, with the home team's name or the away team's name returned as the winner. The input parameters include the names of the home and away teams, the match date, and the names to be used for the home and away teams if they win.")
async def get_match_winner(home_team: str = Query(..., description="Home team name"), away_team: str = Query(..., description="Away team name"), date: str = Query(..., description="Match date in 'YYYY-MM-DD' format"), home_winner: str = Query(..., description="Home team name if they win"), away_winner: str = Query(..., description="Away team name if they win")):
    cursor.execute("SELECT CASE WHEN FTR = 'H' THEN ? ELSE ? END WINNER FROM matchs WHERE Date = ? AND HomeTeam = ? AND AwayTeam = ?", (home_winner, away_winner, date, home_team, away_team))
    result = cursor.fetchone()
    if not result:
        return {"winner": []}
    return {"winner": result[0]}

# Endpoint to get the full-time home and away goals for a specific match
@app.get("/v1/european_football_1/match_full_time_goals", operation_id="get_match_full_time_goals", summary="Retrieves the full-time home and away goals scored in a specific football match. The match is identified by the date it was played, the name of the home team, and the name of the away team. The response includes the number of goals scored by the home team and the away team at the end of the match.")
async def get_match_full_time_goals(date: str = Query(..., description="Match date in 'YYYY-MM-DD' format"), home_team: str = Query(..., description="Home team name"), away_team: str = Query(..., description="Away team name")):
    cursor.execute("SELECT FTHG, FTAG FROM matchs WHERE Date = ? AND HomeTeam = ? AND AwayTeam = ?", (date, home_team, away_team))
    result = cursor.fetchone()
    if not result:
        return {"full_time_goals": []}
    return {"full_time_goals": result}

# Endpoint to get the minimum date of matches where the total goals scored is greater than a specified number
@app.get("/v1/european_football_1/min_date_total_goals", operation_id="get_min_date_total_goals", summary="Retrieve the earliest date on which a football match was played with a total number of goals scored exceeding the specified threshold.")
async def get_min_date_total_goals(total_goals: int = Query(..., description="Total goals scored in the match")):
    cursor.execute("SELECT MIN(Date) FROM matchs WHERE FTHG + FTAG > ?", (total_goals,))
    result = cursor.fetchone()
    if not result:
        return {"min_date": []}
    return {"min_date": result[0]}

# Endpoint to get the winner of the match with the highest total goals in a specified division
@app.get("/v1/european_football_1/winner_highest_total_goals", operation_id="get_winner_highest_total_goals", summary="Retrieves the winning team from the match with the highest combined total of goals in a specified division.")
async def get_winner_highest_total_goals(division_name: str = Query(..., description="Name of the division")):
    cursor.execute("SELECT CASE WHEN T1.FTR = 'H' THEN T1.HomeTeam ELSE T1.AwayTeam END WINNER FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? ORDER BY T1.FTAG + T1.FTHG DESC LIMIT 1", (division_name,))
    result = cursor.fetchone()
    if not result:
        return {"winner": []}
    return {"winner": result[0]}

# Endpoint to get the count of matches with a specific result in a specified division on a given date
@app.get("/v1/european_football_1/count_matches_result_division_date", operation_id="get_count_matches_result_division_date", summary="Retrieves the total number of matches in a specific division that ended with a particular result on a given date. The division is identified by its name, the date is provided in 'YYYY-MM-DD' format, and the result is represented as 'H' for home win, 'A' for away win, or 'D' for draw.")
async def get_count_matches_result_division_date(division_name: str = Query(..., description="Name of the division"), match_date: str = Query(..., description="Date of the match in 'YYYY-MM-DD' format"), match_result: str = Query(..., description="Result of the match (H, A, D)")):
    cursor.execute("SELECT COUNT(T1.FTR) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.Date = ? AND T1.FTR = ?", (division_name, match_date, match_result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country with the highest total away goals
@app.get("/v1/european_football_1/country_highest_total_away_goals", operation_id="get_country_highest_total_away_goals", summary="Retrieves the country with the highest total number of away goals scored in matches. The operation calculates the sum of away goals for each country and returns the country with the highest total.")
async def get_country_highest_total_away_goals():
    cursor.execute("SELECT T2.country FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division GROUP BY T2.country ORDER BY SUM(T1.FTAG) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the division and name of matches with specific goals in a given season
@app.get("/v1/european_football_1/division_name_specific_goals_season", operation_id="get_division_name_specific_goals_season", summary="Retrieve the division and match name for games in a specified season where the home and away teams scored a certain number of goals. The season, home goals, and away goals are provided as input parameters.")
async def get_division_name_specific_goals_season(season: int = Query(..., description="Season of the match"), home_goals: int = Query(..., description="Home goals scored"), away_goals: int = Query(..., description="Away goals scored")):
    cursor.execute("SELECT T2.division, T2.name FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.season = ? AND T1.FTHG = ? AND T1.FTAG = ?", (season, home_goals, away_goals))
    result = cursor.fetchall()
    if not result:
        return {"divisions": []}
    return {"divisions": [{"division": row[0], "name": row[1]} for row in result]}

# Endpoint to get the division and name of the match with the highest total goals on a given date
@app.get("/v1/european_football_1/division_name_highest_total_goals_date", operation_id="get_division_name_highest_total_goals_date", summary="Retrieves the division and match name with the highest total goals scored on a specific date. The date and minimum total goals can be specified as input parameters to filter the results.")
async def get_division_name_highest_total_goals_date(match_date: str = Query(..., description="Date of the match in 'YYYY-MM-DD' format"), total_goals: int = Query(..., description="Total goals scored in the match")):
    cursor.execute("SELECT T2.division, T2.name FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.Date = ? AND T1.FTAG + T1.FTHG > ? ORDER BY T1.FTAG + T1.FTHG DESC LIMIT 1", (match_date, total_goals))
    result = cursor.fetchone()
    if not result:
        return {"division": [], "name": []}
    return {"division": result[0], "name": result[1]}

# Endpoint to get the division with the most matches with specific goals
@app.get("/v1/european_football_1/division_most_matches_specific_goals", operation_id="get_division_most_matches_specific_goals", summary="Retrieves the division that has the highest number of matches where the away team scored a specific number of goals and the home team scored a different specific number of goals.")
async def get_division_most_matches_specific_goals(away_goals: int = Query(..., description="Away goals scored"), home_goals: int = Query(..., description="Home goals scored")):
    cursor.execute("SELECT T2.name FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.FTAG = ? AND T1.FTHG = ? GROUP BY T2.division ORDER BY COUNT(T1.FTAG) DESC LIMIT 1", (away_goals, home_goals))
    result = cursor.fetchone()
    if not result:
        return {"division": []}
    return {"division": result[0]}

# Endpoint to get the count of matches in a specified division on a specific date with specific match details
@app.get("/v1/european_football_1/count_matches_division_date_details", operation_id="get_count_matches_division_date_details", summary="Retrieve the total number of matches played in a specified division on a particular date, given the match details such as the home and away team names, and the goals scored by each team.")
async def get_count_matches_division_date_details(division_name: str = Query(..., description="Name of the division"), home_goals: int = Query(..., description="Home goals scored"), away_goals: int = Query(..., description="Away goals scored"), home_team: str = Query(..., description="Home team name"), away_team: str = Query(..., description="Away team name")):
    cursor.execute("SELECT COUNT(T1.Date) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.Date = ( SELECT Date FROM matchs WHERE FTHG = ? AND FTAG = ? AND HomeTeam = ? AND AwayTeam = ? )", (division_name, home_goals, away_goals, home_team, away_team))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of divisions in a specified country with specific match goals
@app.get("/v1/european_football_1/count_divisions_country_goals", operation_id="get_count_divisions_country_goals", summary="Retrieves the total number of divisions in a specified country where matches have resulted in a specific number of home and away goals. The operation filters divisions based on the provided country and match goals, then returns the count of qualifying divisions.")
async def get_count_divisions_country_goals(country: str = Query(..., description="Country of the division"), home_goals: int = Query(..., description="Home goals scored"), away_goals: int = Query(..., description="Away goals scored")):
    cursor.execute("SELECT COUNT(T1.Div) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.country = ? AND T1.FTHG = ? AND T1.FTAG = ?", (country, home_goals, away_goals))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct countries where a specific team has played
@app.get("/v1/european_football_1/distinct_countries_by_team", operation_id="get_distinct_countries_by_team", summary="Retrieve a list of unique countries where the specified team has participated in matches, either as the home team or the away team.")
async def get_distinct_countries_by_team(team: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT DISTINCT T2.country FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.HomeTeam = ? OR T1.AwayTeam = ?", (team, team))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the count of distinct home teams in a specific division and season
@app.get("/v1/european_football_1/count_distinct_home_teams", operation_id="get_count_distinct_home_teams", summary="Retrieves the total number of unique home teams that have played in a specified division and season. The division is identified by its name, and the season is determined by the year.")
async def get_count_distinct_home_teams(division_name: str = Query(..., description="Name of the division"), season: int = Query(..., description="Season year")):
    cursor.execute("SELECT COUNT(DISTINCT T1.HomeTeam) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.season = ?", (division_name, season))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of home wins in a specific division and season
@app.get("/v1/european_football_1/percentage_home_wins", operation_id="get_percentage_home_wins", summary="Retrieves the percentage of home wins for a given division and season. The calculation is based on the total number of matches played in the specified division and season, with home wins identified by the 'H' result code. The returned value is a real number representing the percentage, rounded to two decimal places.")
async def get_percentage_home_wins(season: int = Query(..., description="Season year"), division_name: str = Query(..., description="Name of the division")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.FTR = 'H' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.FTR) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.season = ? AND T2.name = ?", (season, division_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of matches in a specific division with a given score
@app.get("/v1/european_football_1/percentage_matches_division_score", operation_id="get_percentage_matches_division_score", summary="Retrieves the percentage of matches in a specified division where the home team scored a certain number of goals and the away team scored another specific number of goals.")
async def get_percentage_matches_division_score(division_name: str = Query(..., description="Name of the division"), home_goals: int = Query(..., description="Home goals"), away_goals: int = Query(..., description="Away goals")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.name = ? THEN T1.Div ELSE NULL END) AS REAL) * 100 / COUNT(T1.Div) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.FTHG = ? AND T1.FTAG = ?", (division_name, home_goals, away_goals))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of matches in a specific division and season
@app.get("/v1/european_football_1/count_matches_division_season", operation_id="get_count_matches_division_season", summary="Retrieves the total number of matches played in a specific football division during a given season. The operation requires the season year and the name of the division as input parameters.")
async def get_count_matches_division_season(season: int = Query(..., description="Season year"), division_name: str = Query(..., description="Name of the division")):
    cursor.execute("SELECT COUNT(T1.Div) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.season = ? AND T2.name = ?", (season, division_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get home teams for matches on a specific date in a specific division
@app.get("/v1/european_football_1/home_teams_by_date_division", operation_id="get_home_teams_by_date_division", summary="Retrieves the home teams for matches scheduled on a specific date in a given division. The operation requires the date in 'YYYY-MM-DD' format and the name of the division as input parameters.")
async def get_home_teams_by_date_division(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), division_name: str = Query(..., description="Name of the division")):
    cursor.execute("SELECT T1.HomeTeam FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.Date = ? AND T2.name = ?", (date, division_name))
    result = cursor.fetchall()
    if not result:
        return {"home_teams": []}
    return {"home_teams": [row[0] for row in result]}

# Endpoint to get the winning team for matches on a specific date in a specific division
@app.get("/v1/european_football_1/winning_team_by_date_division", operation_id="get_winning_team_by_date_division", summary="Retrieves the winning team for matches played on a specific date in a given division. The operation requires a date in 'YYYY-MM-DD' format and the name of the division as input parameters. The result is the team that won the match on the specified date in the provided division.")
async def get_winning_team_by_date_division(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), division_name: str = Query(..., description="Name of the division")):
    cursor.execute("SELECT CASE WHEN T1.FTR = 'H' THEN T1.HomeTeam WHEN T1.FTR = 'A' THEN T1.AwayTeam END WINNER FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.Date = ? AND T2.name = ?", (date, division_name))
    result = cursor.fetchall()
    if not result:
        return {"winning_teams": []}
    return {"winning_teams": [row[0] for row in result]}

# Endpoint to get the top home team with the most wins in a specific division
@app.get("/v1/european_football_1/top_home_team_most_wins", operation_id="get_top_home_team_most_wins", summary="Retrieves the home team with the highest number of wins in a specified division. The division is identified by its name. The result is determined by counting the number of matches won at home by each team in the division and ordering them in descending order. The team with the most home wins is returned.")
async def get_top_home_team_most_wins(division_name: str = Query(..., description="Name of the division")):
    cursor.execute("SELECT T1.HomeTeam FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.FTR = 'H' GROUP BY T1.HomeTeam ORDER BY COUNT(T1.FTR) DESC LIMIT 1", (division_name,))
    result = cursor.fetchone()
    if not result:
        return {"top_home_team": []}
    return {"top_home_team": result[0]}

# Endpoint to get the count of away wins for a specific team in a specific division
@app.get("/v1/european_football_1/count_away_wins_team_division", operation_id="get_count_away_wins_team_division", summary="Retrieves the total number of away games won by a specific team in a given division. The operation requires the division name and the name of the away team as input parameters.")
async def get_count_away_wins_team_division(division_name: str = Query(..., description="Name of the division"), away_team: str = Query(..., description="Name of the away team")):
    cursor.execute("SELECT COUNT(T1.Div) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.AwayTeam = ? AND T1.FTR = 'A'", (division_name, away_team))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of away wins in a specific division and season
@app.get("/v1/european_football_1/count_away_wins_division_season", operation_id="get_count_away_wins_division_season", summary="Retrieves the total number of away wins for a specified division and season. The division is identified by its name, and the season is determined by the year. This operation provides a statistical overview of away team victories within a particular division and season.")
async def get_count_away_wins_division_season(division_name: str = Query(..., description="Name of the division"), season: int = Query(..., description="Season year")):
    cursor.execute("SELECT COUNT(T1.Div) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.FTR = 'A' AND T1.season = ?", (division_name, season))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of matches in a specific division with a specific full-time result
@app.get("/v1/european_football_1/count_matches_division_result", operation_id="get_count_matches_division_result", summary="Retrieves the total number of matches in a specified division that ended with a particular full-time result. The division is identified by its name, and the full-time result is represented by a single character code (e.g., 'D' for a draw).")
async def get_count_matches_division_result(division_name: str = Query(..., description="Name of the division"), full_time_result: str = Query(..., description="Full-time result (e.g., 'D' for draw)")):
    cursor.execute("SELECT COUNT(T1.Div) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.FTR = ?", (division_name, full_time_result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in match counts between two divisions in a specific season with a specific full-time result
@app.get("/v1/european_football_1/match_count_difference_divisions_season_result", operation_id="get_match_count_difference_divisions_season_result", summary="Retrieve the difference in the number of matches played between two specified divisions in a given season, filtered by a specific full-time result.")
async def get_match_count_difference_divisions_season_result(division_name_1: str = Query(..., description="Name of the first division"), division_name_2: str = Query(..., description="Name of the second division"), season: int = Query(..., description="Season (e.g., 2021)"), full_time_result: str = Query(..., description="Full-time result (e.g., 'H' for home win)")):
    cursor.execute("SELECT COUNT(CASE WHEN T2.name = ? THEN 1 ELSE NULL END) - COUNT(CASE WHEN T2.name = ? THEN 1 ELSE NULL END) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.season = ? AND T1.FTR = ?", (division_name_1, division_name_2, season, full_time_result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct home teams in a specific division and season with a specific full-time result
@app.get("/v1/european_football_1/distinct_home_teams_division_season_result", operation_id="get_distinct_home_teams_division_season_result", summary="Retrieve a unique list of home teams that have played in a specified division and season, with a specific full-time result. The division, season, and full-time result are provided as input parameters.")
async def get_distinct_home_teams_division_season_result(season: int = Query(..., description="Season (e.g., 2021)"), full_time_result: str = Query(..., description="Full-time result (e.g., 'H' for home win)"), division_name: str = Query(..., description="Name of the division")):
    cursor.execute("SELECT DISTINCT T1.HomeTeam FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.season = ? AND T1.FTR = ? AND T2.name = ?", (season, full_time_result, division_name))
    result = cursor.fetchall()
    if not result:
        return {"home_teams": []}
    return {"home_teams": [row[0] for row in result]}

# Endpoint to compare match counts between two home teams in a specific season with a specific full-time result
@app.get("/v1/european_football_1/compare_home_teams_season_result", operation_id="compare_home_teams_season_result", summary="This endpoint compares the number of matches won at home by two specified teams in a given season. It returns the name of the team with the higher number of home wins for the specified full-time result.")
async def compare_home_teams_season_result(home_team_1: str = Query(..., description="First home team"), home_team_2: str = Query(..., description="Second home team"), season: int = Query(..., description="Season (e.g., 2021)"), full_time_result: str = Query(..., description="Full-time result (e.g., 'H' for home win)")):
    cursor.execute("SELECT CASE WHEN COUNT(CASE WHEN T1.HomeTeam = ? THEN 1 ELSE NULL END) - COUNT(CASE WHEN T1.HomeTeam = ? THEN 1 ELSE NULL END) > 0 THEN ? ELSE ? END FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.season = ? AND T1.FTR = ?", (home_team_1, home_team_2, home_team_1, home_team_2, season, full_time_result))
    result = cursor.fetchone()
    if not result:
        return {"comparison_result": []}
    return {"comparison_result": result[0]}

# Endpoint to get the home team with the highest full-time home goals in a specific division and season
@app.get("/v1/european_football_1/top_home_team_goals_division_season", operation_id="get_top_home_team_goals_division_season", summary="Retrieve the home team that scored the most full-time home goals in a given division and season. The division and season are specified as input parameters.")
async def get_top_home_team_goals_division_season(division_name: str = Query(..., description="Name of the division"), season: int = Query(..., description="Season (e.g., 2021)")):
    cursor.execute("SELECT T1.HomeTeam FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.season = ? ORDER BY T1.FTHG DESC LIMIT 1", (division_name, season))
    result = cursor.fetchone()
    if not result:
        return {"home_team": []}
    return {"home_team": result[0]}

# Endpoint to get the total full-time home goals in a specific division and season
@app.get("/v1/european_football_1/total_home_goals_division_season", operation_id="get_total_home_goals_division_season", summary="Retrieves the total number of full-time home goals scored in a specified division and season. The division is identified by its name, and the season is denoted by a year. This operation aggregates the full-time home goals from all matches in the given division and season.")
async def get_total_home_goals_division_season(division_name: str = Query(..., description="Name of the division"), season: int = Query(..., description="Season (e.g., 2021)")):
    cursor.execute("SELECT SUM(T1.FTHG) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T2.name = ? AND T1.season = ?", (division_name, season))
    result = cursor.fetchone()
    if not result:
        return {"total_goals": []}
    return {"total_goals": result[0]}

# Endpoint to get the percentage of home and away wins for a specific team in a specific season
@app.get("/v1/european_football_1/win_percentage_team_season", operation_id="get_win_percentage_team_season", summary="Get the percentage of home and away wins for a specific team in a specific season")
async def get_win_percentage_team_season(season: int = Query(..., description="Season (e.g., 2021)"), team_name: str = Query(..., description="Name of the team")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.FTR = 'H' THEN 1 ELSE NULL END) + COUNT(CASE WHEN T1.FTR = 'A' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(t1.FTR) FROM matchs AS T1 INNER JOIN divisions AS T2 ON T1.Div = T2.division WHERE T1.season = ? AND (T1.AwayTeam = ? OR T1.HomeTeam = ?)", (season, team_name, team_name))
    result = cursor.fetchone()
    if not result:
        return {"win_percentage": []}
    return {"win_percentage": result[0]}

api_calls = [
    "/v1/european_football_1/match_count_season_away_team_ftr?season=2008&away_team=Ebbsfleet&ftr=D",
    "/v1/european_football_1/division_percentage_by_country?country=England",
    "/v1/european_football_1/match_result_percentages_by_home_team?home_team=Cittadella",
    "/v1/european_football_1/most_frequent_away_team?home_team=Caen&season=2010&ftr=A",
    "/v1/european_football_1/division_percentage_by_date?division=F1&date=2005-07-30",
    "/v1/european_football_1/team_match_percentage_by_ftr?team=Sassuolo&ftr=D",
    "/v1/european_football_1/away_team_goal_percentage_by_season?goals=2&season=2017",
    "/v1/european_football_1/teams_by_division_and_league?league_name=EFL%20League%20One&division=E2",
    "/v1/european_football_1/distinct_teams_goal_count_by_country?min_goals=4&country=Greece",
    "/v1/european_football_1/match_count_by_division_season_scores?season=2019&division_name=Scottish%20Championship&away_goals=2&home_goals=2",
    "/v1/european_football_1/home_teams_by_country_and_goals?country=Scotland&fthg=10",
    "/v1/european_football_1/home_team_win_percentage?division_name=LaLiga&country=Spain&season=2017",
    "/v1/european_football_1/matches_count_and_percentage?country=England&season=2012",
    "/v1/european_football_1/away_teams_max_goals?season=2021",
    "/v1/european_football_1/top_home_team_by_goals?division=P1&season=2021",
    "/v1/european_football_1/win_percentage_difference?season=2010",
    "/v1/european_football_1/top_division_by_draws?season=2008",
    "/v1/european_football_1/home_teams_won_on_date?division=EC&date=2008-01-20",
    "/v1/european_football_1/division_names_by_match_details?date=2009-09-13&home_team=Club%20Brugge&away_team=Genk",
    "/v1/european_football_1/match_count_by_division_and_season_range?division_name=Scottish%20Premiership&start_season=2006&end_season=2008",
    "/v1/european_football_1/divisions_countries_home_away_teams?home_team=Hearts&away_team=Hibernian",
    "/v1/european_football_1/away_team_highest_ftag_division?division_name=Bundesliga",
    "/v1/european_football_1/away_teams_country?country=Italy",
    "/v1/european_football_1/division_most_draws_season?season=2019&ftr=D",
    "/v1/european_football_1/count_home_matches_team_division_result?division_name=LaLiga&home_team=Valencia&ftr=H",
    "/v1/european_football_1/count_matches_result_division?division_name=Seria%20A&ftr=D",
    "/v1/european_football_1/count_divisions_country?country=England",
    "/v1/european_football_1/division_names_country?country=Netherlands",
    "/v1/european_football_1/match_winner?home_team=East%20Fife&away_team=Dumbarton&date=2009-10-10&home_winner=East%20Fife&away_winner=Dumbarton",
    "/v1/european_football_1/match_full_time_goals?date=2009-04-26&home_team=Bursaspor&away_team=Denizlispor",
    "/v1/european_football_1/min_date_total_goals?total_goals=10",
    "/v1/european_football_1/winner_highest_total_goals?division_name=Ligue%202",
    "/v1/european_football_1/count_matches_result_division_date?division_name=LaLiga%202&match_date=2016-03-27&match_result=A",
    "/v1/european_football_1/country_highest_total_away_goals",
    "/v1/european_football_1/division_name_specific_goals_season?season=2011&home_goals=1&away_goals=8",
    "/v1/european_football_1/division_name_highest_total_goals_date?match_date=2020-02-22&total_goals=5",
    "/v1/european_football_1/division_most_matches_specific_goals?away_goals=0&home_goals=0",
    "/v1/european_football_1/count_matches_division_date_details?division_name=Scottish%20League%20One&home_goals=5&away_goals=2&home_team=Pro%20Vercelli&away_team=Pescara",
    "/v1/european_football_1/count_divisions_country_goals?country=Greece&home_goals=5&away_goals=0",
    "/v1/european_football_1/distinct_countries_by_team?team=Bradford",
    "/v1/european_football_1/count_distinct_home_teams?division_name=Eredivisie&season=2008",
    "/v1/european_football_1/percentage_home_wins?season=2021&division_name=Bundesliga",
    "/v1/european_football_1/percentage_matches_division_score?division_name=Liga%20NOS&home_goals=1&away_goals=1",
    "/v1/european_football_1/count_matches_division_season?season=2021&division_name=Premier%20League",
    "/v1/european_football_1/home_teams_by_date_division?date=2020-10-02&division_name=Bundesliga",
    "/v1/european_football_1/winning_team_by_date_division?date=2020-10-02&division_name=Bundesliga",
    "/v1/european_football_1/top_home_team_most_wins?division_name=Bundesliga",
    "/v1/european_football_1/count_away_wins_team_division?division_name=Bundesliga&away_team=Werder%20Bremen",
    "/v1/european_football_1/count_away_wins_division_season?division_name=Bundesliga&season=2021",
    "/v1/european_football_1/count_matches_division_result?division_name=Bundesliga&full_time_result=D",
    "/v1/european_football_1/match_count_difference_divisions_season_result?division_name_1=Bundesliga&division_name_2=Premier%20League&season=2021&full_time_result=H",
    "/v1/european_football_1/distinct_home_teams_division_season_result?season=2021&full_time_result=H&division_name=Bundesliga",
    "/v1/european_football_1/compare_home_teams_season_result?home_team_1=Augsburg&home_team_2=Mainz&season=2021&full_time_result=H",
    "/v1/european_football_1/top_home_team_goals_division_season?division_name=Bundesliga&season=2021",
    "/v1/european_football_1/total_home_goals_division_season?division_name=Bundesliga&season=2021",
    "/v1/european_football_1/win_percentage_team_season?season=2021&team_name=Club%20Brugge"
]
