from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/law_episode/law_episode.sqlite')
cursor = conn.cursor()

# Endpoint to get keywords for a specific episode title
@app.get("/v1/law_episode/keywords_by_episode_title", operation_id="get_keywords_by_episode_title", summary="Retrieves a list of keywords linked to a specific episode, identified by its title. The operation searches for the episode using the provided title and returns the associated keywords.")
async def get_keywords_by_episode_title(title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT T2.keyword FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get the count of keywords for a specific season and episode
@app.get("/v1/law_episode/keyword_count_by_season_episode", operation_id="get_keyword_count_by_season_episode", summary="Retrieves the total count of keywords linked to a particular season and episode of a series. The operation requires the season and episode numbers as input parameters to accurately determine the keyword count.")
async def get_keyword_count_by_season_episode(season: int = Query(..., description="Season number"), episode: int = Query(..., description="Episode number")):
    cursor.execute("SELECT COUNT(T2.keyword) FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.season = ? AND T1.episode = ?", (season, episode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episode titles associated with a specific keyword
@app.get("/v1/law_episode/episode_titles_by_keyword", operation_id="get_episode_titles_by_keyword", summary="Retrieves the titles of episodes that are associated with a given keyword. The keyword is used to filter the episodes and return only those that match the specified keyword.")
async def get_episode_titles_by_keyword(keyword: str = Query(..., description="Keyword to search for")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T2.keyword = ?", (keyword,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get keywords for episodes with a rating greater than a specified value
@app.get("/v1/law_episode/keywords_by_rating", operation_id="get_keywords_by_rating", summary="Retrieve a list of keywords linked to episodes that have a rating surpassing the provided minimum value. This operation allows you to filter episodes based on their rating and extract the associated keywords for further analysis or categorization.")
async def get_keywords_by_rating(rating: float = Query(..., description="Minimum rating value")):
    cursor.execute("SELECT T2.keyword FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.rating > ?", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get votes for a specific episode title and star rating
@app.get("/v1/law_episode/votes_by_title_and_stars", operation_id="get_votes_by_title_and_stars", summary="Retrieves the total number of votes for a specific episode, based on the provided title and star rating. The operation filters episodes by title and votes by star rating, then returns the corresponding vote count.")
async def get_votes_by_title_and_stars(title: str = Query(..., description="Title of the episode"), stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T2.votes FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T2.stars = ?", (title, stars))
    result = cursor.fetchall()
    if not result:
        return {"votes": []}
    return {"votes": [row[0] for row in result]}

# Endpoint to get the total votes for a specific episode title
@app.get("/v1/law_episode/total_votes_by_title", operation_id="get_total_votes_by_title", summary="Retrieves the cumulative number of votes for a specific episode, identified by its title. The operation calculates the total votes by summing up all votes associated with the specified episode.")
async def get_total_votes_by_title(title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT SUM(T2.votes) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"total_votes": []}
    return {"total_votes": result[0]}

# Endpoint to get the top episode title with a specific star rating
@app.get("/v1/law_episode/top_episode_by_stars", operation_id="get_top_episode_by_stars", summary="Retrieve the title of the episode that has the highest number of votes for a given star rating. The episode is selected based on the specified star rating and the number of votes it has received.")
async def get_top_episode_by_stars(stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T2.stars = ? ORDER BY T2.votes DESC LIMIT 1", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get roles for a specific episode title, person name, and credited status
@app.get("/v1/law_episode/roles_by_title_name_credited", operation_id="get_roles_by_title_name_credited", summary="Retrieves the roles associated with a specific episode, identified by its title, and a particular person, identified by their name, considering their credited status. The operation returns a list of roles that meet the specified criteria.")
async def get_roles_by_title_name_credited(title: str = Query(..., description="Title of the episode"), name: str = Query(..., description="Name of the person"), credited: str = Query(..., description="Credited status (true or false)")):
    cursor.execute("SELECT T2.role FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T1.title = ? AND T3.name = ? AND T2.credited = ?", (title, name, credited))
    result = cursor.fetchall()
    if not result:
        return {"roles": []}
    return {"roles": [row[0] for row in result]}

# Endpoint to get the count of episodes with a specific title and credited status
@app.get("/v1/law_episode/episode_count_by_title_credited", operation_id="get_episode_count_by_title_credited", summary="Retrieves the total number of episodes that match a given title and credited status. The title and credited status are provided as input parameters, allowing for a targeted count of episodes that meet the specified criteria.")
async def get_episode_count_by_title_credited(title: str = Query(..., description="Title of the episode"), credited: str = Query(..., description="Credited status (true or false)")):
    cursor.execute("SELECT COUNT(T1.episode_id) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T2.credited = ?", (title, credited))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names of persons with a specific role in a specific episode title and credited status
@app.get("/v1/law_episode/person_names_by_title_role_credited", operation_id="get_person_names_by_title_role_credited", summary="Retrieves the names of individuals who have a specified role in a particular episode, based on the provided title and credited status. This operation filters the episode by title, role, and credited status, and returns the corresponding person names.")
async def get_person_names_by_title_role_credited(title: str = Query(..., description="Title of the episode"), credited: str = Query(..., description="Credited status (true or false)"), role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T3.name FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T1.title = ? AND T2.credited = ? AND T2.role = ?", (title, credited, role))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of persons credited in episodes based on credited status and person name
@app.get("/v1/law_episode/count_persons_credited_in_episodes", operation_id="get_count_persons_credited_in_episodes", summary="Retrieves the total count of persons with a specified credited status and name who have been credited in episodes. The count is determined by joining the Episode, Credit, and Person tables based on their respective IDs and filtering by the provided credited status and person name.")
async def get_count_persons_credited_in_episodes(credited: str = Query(..., description="Credited status (true or false)"), name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT COUNT(T3.person_id) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T2.credited = ? AND T3.name = ?", (credited, name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episode titles based on credited status and person name
@app.get("/v1/law_episode/episode_titles_by_credited_person", operation_id="get_episode_titles_by_credited_person", summary="Retrieves the titles of episodes in which a specified person has been credited or not credited, based on the provided credited status and person name.")
async def get_episode_titles_by_credited_person(credited: str = Query(..., description="Credited status (true or false)"), name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T2.credited = ? AND T3.name = ?", (credited, name))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get credited status based on person name and episode ID
@app.get("/v1/law_episode/credited_status_by_person_episode", operation_id="get_credited_status_by_person_episode", summary="Retrieves the credited status of a specific person for a given episode. The operation requires the person's name and the episode ID as input parameters. The result is a boolean value indicating whether the person is credited for the episode.")
async def get_credited_status_by_person_episode(name: str = Query(..., description="Name of the person"), episode_id: str = Query(..., description="Episode ID")):
    cursor.execute("SELECT T1.credited FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T2.name = ? AND T1.episode_id = ?", (name, episode_id))
    result = cursor.fetchone()
    if not result:
        return {"credited": []}
    return {"credited": result[0]}

# Endpoint to get the count of keywords for an episode based on its title
@app.get("/v1/law_episode/count_keywords_by_episode_title", operation_id="get_count_keywords_by_episode_title", summary="Retrieves the total number of keywords associated with a specific episode, identified by its title.")
async def get_count_keywords_by_episode_title(title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT COUNT(T2.keyword) FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top episode title based on star rating
@app.get("/v1/law_episode/top_episode_by_star_rating", operation_id="get_top_episode_by_star_rating", summary="Retrieves the title of the episode with the highest number of votes for a given star rating. The star rating is provided as an input parameter.")
async def get_top_episode_by_star_rating(stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T2.title FROM Vote AS T1 INNER JOIN Episode AS T2 ON T2.episode_id = T1.episode_id WHERE T1.stars = ? ORDER BY T1.votes DESC LIMIT 1", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of awards for an episode based on series, season, and episode number
@app.get("/v1/law_episode/count_awards_by_series_season_episode", operation_id="get_count_awards_by_series_season_episode", summary="Retrieves the total number of awards received by a specific episode in a given series and season. The episode is identified by its number within the season. The series, season, and episode number are required as input parameters.")
async def get_count_awards_by_series_season_episode(series: str = Query(..., description="Series name"), season: int = Query(..., description="Season number"), episode: int = Query(..., description="Episode number")):
    cursor.execute("SELECT COUNT(T2.award_id) FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T2.series = ? AND T1.season = ? AND T1.episode = ?", (series, season, episode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of roles for an episode based on series, season, episode number, and credited status
@app.get("/v1/law_episode/count_roles_by_series_season_episode_credited", operation_id="get_count_roles_by_series_season_episode_credited", summary="Retrieves the total number of roles for a specific episode in a series, based on the provided season, episode number, and whether the role is credited or not.")
async def get_count_roles_by_series_season_episode_credited(series: str = Query(..., description="Series name"), season: int = Query(..., description="Season number"), episode: int = Query(..., description="Episode number"), credited: str = Query(..., description="Credited status (true or false)")):
    cursor.execute("SELECT COUNT(T2.role) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.series = ? AND T1.season = ? AND T1.episode = ? AND T2.credited = ?", (series, season, episode, credited))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the summary of an episode based on award ID
@app.get("/v1/law_episode/episode_summary_by_award_id", operation_id="get_episode_summary_by_award_id", summary="Retrieves a summary of an episode associated with the specified award ID. The operation returns the summary of the episode that has been awarded the provided award ID.")
async def get_episode_summary_by_award_id(award_id: int = Query(..., description="Award ID")):
    cursor.execute("SELECT T1.summary FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T2.award_id = ?", (award_id,))
    result = cursor.fetchone()
    if not result:
        return {"summary": []}
    return {"summary": result[0]}

# Endpoint to get the roles of a person based on their name
@app.get("/v1/law_episode/roles_by_person_name", operation_id="get_roles_by_person_name", summary="Retrieves the roles associated with a specific person, identified by their name. The operation returns a list of roles that the person has been credited with in various episodes.")
async def get_roles_by_person_name(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.role FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T2.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"roles": []}
    return {"roles": [row[0] for row in result]}

# Endpoint to get the count of awards for a person based on their name
@app.get("/v1/law_episode/count_awards_by_person_name", operation_id="get_count_awards_by_person_name", summary="Retrieves the total number of awards associated with a specific individual, identified by their name.")
async def get_count_awards_by_person_name(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT COUNT(T2.award_id) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T1.name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the air date of the episode with the highest total votes
@app.get("/v1/law_episode/air_date_highest_votes", operation_id="get_air_date_highest_votes", summary="Retrieves the air date of the episode that has received the highest total number of votes, up to the specified limit. This operation returns the air date of the episode with the most votes, based on the aggregated votes from all users. The limit parameter allows you to restrict the number of results returned.")
async def get_air_date_highest_votes(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T2.air_date FROM Vote AS T1 INNER JOIN Episode AS T2 ON T2.episode_id = T1.episode_id GROUP BY T2.episode_id ORDER BY SUM(T1.votes) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"air_date": []}
    return {"air_date": result[0]}

# Endpoint to get the names of persons who won a specific award
@app.get("/v1/law_episode/person_names_by_award", operation_id="get_person_names_by_award", summary="Retrieves the names of individuals who have received a specific award, identified by its unique award_id.")
async def get_person_names_by_award(award_id: int = Query(..., description="Award ID")):
    cursor.execute("SELECT T1.name FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T2.award_id = ?", (award_id,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of roles for a specific person
@app.get("/v1/law_episode/role_count_by_person", operation_id="get_role_count_by_person", summary="Retrieves the total number of roles associated with a specific person. The operation requires the person's name as input and returns the count of roles linked to that person.")
async def get_role_count_by_person(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT COUNT(T1.role) FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T2.name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the votes for an episode with a specific title and star rating
@app.get("/v1/law_episode/votes_by_episode_title_and_stars", operation_id="get_votes_by_episode_title_and_stars", summary="Retrieves the total number of votes for a specific episode based on its title and the given star rating. The operation filters episodes by their title and star rating, then aggregates the corresponding votes.")
async def get_votes_by_episode_title_and_stars(stars: int = Query(..., description="Star rating"), title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT T2.votes FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T2.stars = ? AND T1.title = ?", (stars, title))
    result = cursor.fetchall()
    if not result:
        return {"votes": []}
    return {"votes": [row[0] for row in result]}

# Endpoint to get the ratio of episodes with a specific title to another title
@app.get("/v1/law_episode/episode_title_ratio", operation_id="get_episode_title_ratio", summary="Retrieves the proportion of episodes with the first specified title compared to the second specified title. This operation calculates the ratio by comparing the number of episodes with the first title to the number of episodes with the second title. The result is a real number representing the ratio.")
async def get_episode_title_ratio(title1: str = Query(..., description="First title for comparison"), title2: str = Query(..., description="Second title for comparison")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.title = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.title = ? THEN 1 ELSE 0 END) FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id", (title1, title2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the ratio of credited cast members in a range of episodes
@app.get("/v1/law_episode/credited_cast_ratio", operation_id="get_credited_cast_ratio", summary="Retrieves the proportion of credited cast members within a specified category, for a range of episodes. The calculation is based on the count of credits that match the provided category and credited status, divided by the total number of episodes in the range.")
async def get_credited_cast_ratio(category: str = Query(..., description="Category of the credit"), credited: str = Query(..., description="Credited status"), start_episode: int = Query(..., description="Starting episode number"), end_episode: int = Query(..., description="Ending episode number")):
    cursor.execute("SELECT CAST(COUNT(T1.episode_id) AS REAL) / (? - ? + 1) FROM Credit AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.category = ? AND T1.credited = ? AND T2.number_in_series BETWEEN ? AND ?", (end_episode, start_episode, category, credited, start_episode, end_episode))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the names of uncredited persons in a specific episode
@app.get("/v1/law_episode/uncredited_persons_by_episode", operation_id="get_uncredited_persons_by_episode", summary="Retrieves the names of individuals who were not credited in a specific episode. The operation requires the episode ID and the credited status as input parameters. The credited status is used to filter the results, ensuring that only individuals who were not credited are returned.")
async def get_uncredited_persons_by_episode(credited: str = Query(..., description="Credited status"), episode_id: str = Query(..., description="Episode ID")):
    cursor.execute("SELECT T2.name FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T1.credited = ? AND T1.episode_id = ?", (credited, episode_id))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of persons who won a minimum number of awards
@app.get("/v1/law_episode/person_count_by_min_awards", operation_id="get_person_count_by_min_awards", summary="Get the count of persons who won a minimum number of awards")
async def get_person_count_by_min_awards(result: str = Query(..., description="Award result"), min_awards: int = Query(..., description="Minimum number of awards")):
    cursor.execute("SELECT COUNT(T1.person_id) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T2.result = ? GROUP BY T1.person_id HAVING COUNT(T2.award_id) >= ?", (result, min_awards))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of persons with a specific role in a specific episode
@app.get("/v1/law_episode/person_names_by_role_and_episode", operation_id="get_person_names_by_role_and_episode", summary="Retrieves the names of individuals who have a specified role in a given episode. The operation requires the unique identifier of the episode and the role of the individual as input parameters.")
async def get_person_names_by_role_and_episode(episode_id: str = Query(..., description="Episode ID"), role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T2.name FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T1.episode_id = ? AND T1.role = ?", (episode_id, role))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of awards for a specific person with a specific result
@app.get("/v1/law_episode/award_count_by_person_and_result", operation_id="get_award_count_by_person_and_result", summary="Retrieves the total number of awards received by a specific individual for a particular award result. The operation requires the individual's name and the desired award result as input parameters.")
async def get_award_count_by_person_and_result(name: str = Query(..., description="Name of the person"), result: str = Query(..., description="Award result")):
    cursor.execute("SELECT COUNT(T2.award_id) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T1.name = ? AND T2.result = ?", (name, result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of the tallest person with a specific role
@app.get("/v1/law_episode/tallest_person_by_role", operation_id="get_tallest_person_by_role", summary="Retrieves the name of the tallest individual who has been credited with a specific role. The role is provided as an input parameter.")
async def get_tallest_person_by_role(role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T2.name FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T1.role = ? ORDER BY T2.height_meters DESC LIMIT 1", (role,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of persons from a specific country who won an award in a specific year
@app.get("/v1/law_episode/count_persons_by_country_and_year", operation_id="get_count_persons_by_country_and_year", summary="Retrieves the total number of individuals born in a specified country who received an award in a given year. The response is based on the provided year of the award and the birth country of the individual.")
async def get_count_persons_by_country_and_year(year: int = Query(..., description="Year of the award"), birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT COUNT(T1.person_id) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T2.year = ? AND T1.birth_country = ?", (year, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of episodes with a specific star rating
@app.get("/v1/law_episode/count_episodes_by_star_rating", operation_id="get_count_episodes_by_star_rating", summary="Retrieves the total number of episodes that have received a specified star rating. The star rating is provided as an input parameter.")
async def get_count_episodes_by_star_rating(stars: int = Query(..., description="Star rating of the episode")):
    cursor.execute("SELECT COUNT(T1.episode_id) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T2.stars = ?", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of votes with a specific star rating for a specific episode
@app.get("/v1/law_episode/percentage_votes_by_star_rating_and_episode", operation_id="get_percentage_votes_by_star_rating_and_episode", summary="Retrieves the percentage of votes with a specific star rating for a given episode. The operation calculates this percentage by comparing the number of votes with the specified star rating to the total number of votes for the episode. The input parameters include the star rating, episode title, and episode ID.")
async def get_percentage_votes_by_star_rating_and_episode(stars: int = Query(..., description="Star rating of the vote"), title: str = Query(..., description="Title of the episode"), episode_id: str = Query(..., description="ID of the episode")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.stars = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.episode_id) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T1.episode_id = ?", (stars, title, episode_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the title of the episode with the most keywords
@app.get("/v1/law_episode/episode_with_most_keywords", operation_id="get_episode_with_most_keywords", summary="Retrieves the title of the episode that has the highest number of associated keywords. This operation identifies the episode with the most keywords by joining the Episode and Keyword tables, grouping by episode_id, and ordering by the count of keywords in descending order. The title of the top episode is then returned.")
async def get_episode_with_most_keywords():
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id GROUP BY T1.episode_id ORDER BY COUNT(T2.keyword) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of episodes that aired in a specific year, won an award from a specific organization, and had a specific result
@app.get("/v1/law_episode/count_episodes_by_year_organization_result", operation_id="get_count_episodes_by_year_organization_result", summary="Retrieve the number of episodes that aired in a given year, received an award from a specified organization, and achieved a particular result.")
async def get_count_episodes_by_year_organization_result(year: str = Query(..., description="Year the episode aired, in 'YYYY' format"), organization: str = Query(..., description="Organization that awarded the episode"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT COUNT(T1.episode_id) FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE strftime('%Y', T1.air_date) = ? AND T2.organization = ? AND T2.result = ?", (year, organization, result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of awards won by an episode with a specific title and result
@app.get("/v1/law_episode/count_awards_by_episode_title_and_result", operation_id="get_count_awards_by_episode_title_and_result", summary="Retrieves the total number of awards won by a specific episode, based on its title and the outcome of the award. The response provides a quantitative measure of the episode's success in winning awards under the specified result category.")
async def get_count_awards_by_episode_title_and_result(title: str = Query(..., description="Title of the episode"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT COUNT(T2.award_id) FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T2.result = ?", (title, result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count and percentage of episodes in a specific season for a specific series and category
@app.get("/v1/law_episode/count_and_percentage_episodes_by_season_series_category", operation_id="get_count_and_percentage_episodes_by_season_series_category", summary="Retrieves the total count and percentage of episodes in a specified season for a given series and credit category. The calculation is based on the number of episodes in the provided season compared to the total episodes in the series.")
async def get_count_and_percentage_episodes_by_season_series_category(season: int = Query(..., description="Season number"), category: str = Query(..., description="Category of the credit"), series: str = Query(..., description="Title of the series")):
    cursor.execute("SELECT SUM(CASE WHEN T2.season = ? THEN 1 ELSE 0 END) AS num , CAST(SUM(CASE WHEN T2.season = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(T1.episode_id) FROM Credit AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.category = ? AND T2.series = ?", (season, season, category, series))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "percentage": []}
    return {"count": result[0], "percentage": result[1]}

# Endpoint to get the keyword of the episode with the second highest number of votes
@app.get("/v1/law_episode/keyword_of_second_highest_votes", operation_id="get_keyword_of_second_highest_votes", summary="Retrieves the keyword associated with the episode that has the second highest number of votes. This operation identifies the episode with the second most votes and returns the keyword linked to it. The keyword is obtained from the Keyword table, which is joined with the Episode table based on the episode_id. The result is ordered in descending order of votes and limited to one record.")
async def get_keyword_of_second_highest_votes():
    cursor.execute("SELECT T2.keyword FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.votes NOT IN ( SELECT MAX(T1.votes) FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id ) ORDER BY T1.votes DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"keyword": []}
    return {"keyword": result[0]}

# Endpoint to get the count of awards for a specific episode title and award result
@app.get("/v1/law_episode/count_awards_by_title_result", operation_id="get_count_awards_by_title_result", summary="Retrieves the total number of awards received by a specific episode, based on the provided episode title and award result. This operation does not return individual award details, but rather a count of awards that meet the specified criteria.")
async def get_count_awards_by_title_result(title: str = Query(..., description="Title of the episode"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT COUNT(T2.award) FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T2.result = ?", (title, result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of persons with a specific role in a specific episode title
@app.get("/v1/law_episode/get_person_names_by_title_role", operation_id="get_person_names_by_title_role", summary="Retrieves the names of individuals who have a specified role in a particular episode. The episode is identified by its title, and the role is defined by the provided role parameter. This operation returns a list of names that meet the specified criteria.")
async def get_person_names_by_title_role(title: str = Query(..., description="Title of the episode"), role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T3.name FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T1.title = ? AND T2.role = ?", (title, role))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the top organization for a specific person's awards
@app.get("/v1/law_episode/top_organization_by_person_name_result", operation_id="get_top_organization_by_person_name_result", summary="Retrieves the organization that has awarded a specific person the most times, based on the provided award result.")
async def get_top_organization_by_person_name_result(name: str = Query(..., description="Name of the person"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T2.organization FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T1.name = ? AND T2.result = ? GROUP BY T2.organization ORDER BY COUNT(T2.award_id) DESC LIMIT 1", (name, result))
    result = cursor.fetchone()
    if not result:
        return {"organization": []}
    return {"organization": result[0]}

# Endpoint to get the names of persons with a specific role in a specific episode number
@app.get("/v1/law_episode/get_person_names_by_episode_role", operation_id="get_person_names_by_episode_role", summary="Retrieves the names of individuals who have a specified role in a particular episode. The operation requires the episode number and the role of the individual as input parameters.")
async def get_person_names_by_episode_role(episode: int = Query(..., description="Episode number"), role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T3.name FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T1.episode = ? AND T2.role = ?", (episode, role))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of persons credited in a specific episode title
@app.get("/v1/law_episode/count_persons_by_title_credited", operation_id="get_count_persons_by_title_credited", summary="Retrieves the total number of individuals who have been credited or not credited for a specific episode, based on the provided episode title.")
async def get_count_persons_by_title_credited(title: str = Query(..., description="Title of the episode"), credited: str = Query(..., description="Credited status (true or false)")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T2.credited = ?", (title, credited))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top episode title by credit category
@app.get("/v1/law_episode/top_episode_by_credit_category", operation_id="get_top_episode_by_credit_category", summary="Retrieves the title of the episode with the highest number of credits in a specified category. The category is provided as an input parameter.")
async def get_top_episode_by_credit_category(category: str = Query(..., description="Category of the credit")):
    cursor.execute("SELECT T2.title FROM Credit AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.category = ? GROUP BY T2.episode_id ORDER BY COUNT(T1.category) DESC LIMIT 1", (category,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the top episode titles by votes and stars
@app.get("/v1/law_episode/top_episodes_by_votes_stars", operation_id="get_top_episodes_by_votes_stars", summary="Retrieves the top three episode titles that have received a minimum number of votes and a specific number of stars. The episodes are ordered by the number of votes in descending order.")
async def get_top_episodes_by_votes_stars(votes: int = Query(..., description="Minimum number of votes"), stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT T2.title FROM Vote AS T1 INNER JOIN Episode AS T2 ON T2.episode_id = T1.episode_id WHERE T1.votes >= ? AND T1.stars = ? ORDER BY T1.votes DESC LIMIT 3", (votes, stars))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the oldest person with a specific role
@app.get("/v1/law_episode/oldest_person_by_role", operation_id="get_oldest_person_by_role", summary="Retrieves the name of the oldest person who has a specified role in the database. The role is provided as an input parameter.")
async def get_oldest_person_by_role(role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T2.name FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T1.role = ? AND T2.birthdate IS NOT NULL ORDER BY T2.birthdate LIMIT 1", (role,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of episodes with a specific number of stars
@app.get("/v1/law_episode/count_episodes_by_episode_stars", operation_id="get_count_episodes_by_episode_stars", summary="Retrieves the total number of episodes that have a specified number of stars. The operation requires the episode number and the desired number of stars as input parameters.")
async def get_count_episodes_by_episode_stars(episode: int = Query(..., description="Episode number"), stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT COUNT(T1.episode_id) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T1.episode = ? AND T2.stars = ?", (episode, stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names of persons based on credit category
@app.get("/v1/law_episode/person_names_by_credit_category", operation_id="get_person_names_by_credit_category", summary="Retrieves the names of individuals associated with a specified credit category. The category parameter is used to filter the results.")
async def get_person_names_by_credit_category(category: str = Query(..., description="Credit category")):
    cursor.execute("SELECT T2.name FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T1.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the person with the highest number of credits
@app.get("/v1/law_episode/top_person_by_credits", operation_id="get_top_person_by_credits", summary="Retrieves the individual with the most credits and their proportion of total credits. This operation calculates the total number of credits for each person and identifies the person with the highest count. The result includes the person's ID and the percentage of total credits they hold.")
async def get_top_person_by_credits():
    cursor.execute("SELECT T2.person_id, CAST(COUNT(T2.person_id) AS REAL) * 100 / ( SELECT COUNT(T2.person_id) AS num FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id ) AS per FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id GROUP BY T2.person_id ORDER BY COUNT(T2.person_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"person_id": [], "percentage": []}
    return {"person_id": result[0], "percentage": result[1]}

# Endpoint to get distinct episode IDs based on award category
@app.get("/v1/law_episode/distinct_episode_ids_by_award_category", operation_id="get_distinct_episode_ids_by_award_category", summary="Retrieves a unique set of episode IDs associated with a given award category. The operation filters episodes based on the specified award category and returns a list of distinct episode IDs.")
async def get_distinct_episode_ids_by_award_category(award_category: str = Query(..., description="Award category")):
    cursor.execute("SELECT DISTINCT episode_id FROM Award WHERE award_category = ?", (award_category,))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

# Endpoint to get the count of awards based on result
@app.get("/v1/law_episode/award_count_by_result", operation_id="get_award_count_by_result", summary="Retrieves the total number of awards associated with a given result. The result parameter specifies the outcome of the award, such as Nominee or Winner.")
async def get_award_count_by_result(result: str = Query(..., description="Result of the award (e.g., Nominee, Winner)")):
    cursor.execute("SELECT COUNT(award_id) FROM Award WHERE Result = ?", (result,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct roles based on credited status
@app.get("/v1/law_episode/distinct_roles_by_credited_status", operation_id="get_distinct_roles_by_credited_status", summary="Retrieves a unique list of roles based on the provided credited status. This operation allows you to filter roles by whether they have been officially credited or not, providing a focused view of the roles in the system.")
async def get_distinct_roles_by_credited_status(credited: str = Query(..., description="Credited status (e.g., true, false)")):
    cursor.execute("SELECT DISTINCT role FROM Credit WHERE credited = ?", (credited,))
    result = cursor.fetchall()
    if not result:
        return {"roles": []}
    return {"roles": [row[0] for row in result]}

# Endpoint to get episode titles ordered by rating with a limit
@app.get("/v1/law_episode/episode_titles_by_rating", operation_id="get_episode_titles_by_rating", summary="Retrieves a specified number of episode titles, ordered by their respective ratings in descending order. The limit parameter determines the maximum number of episodes to return.")
async def get_episode_titles_by_rating(limit: int = Query(..., description="Limit of the number of episodes to return")):
    cursor.execute("SELECT title FROM Episode ORDER BY rating LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get birth place and region based on birth name
@app.get("/v1/law_episode/birth_place_region_by_birth_name", operation_id="get_birth_place_region_by_birth_name", summary="Retrieves the birth place and region associated with a given birth name. The operation uses the provided birth name to search for corresponding records in the database and returns the birth place and region for each match.")
async def get_birth_place_region_by_birth_name(birth_name: str = Query(..., description="Birth name of the person")):
    cursor.execute("SELECT birth_place, birth_region FROM Person WHERE birth_name = ?", (birth_name,))
    result = cursor.fetchone()
    if not result:
        return {"birth_place": [], "birth_region": []}
    return {"birth_place": result[0], "birth_region": result[1]}

# Endpoint to get names of persons based on birth country
@app.get("/v1/law_episode/person_names_by_birth_country", operation_id="get_person_names_by_birth_country", summary="Retrieves the names of individuals who were born in a specified country. The birth country is provided as an input parameter.")
async def get_person_names_by_birth_country(birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT name FROM Person WHERE birth_country = ?", (birth_country,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get episode titles based on star rating
@app.get("/v1/law_episode/episode_titles_by_star_rating", operation_id="get_episode_titles_by_star_rating", summary="Retrieves the titles of episodes that have been rated with a specific star rating. The operation filters episodes based on the provided star rating and returns their corresponding titles.")
async def get_episode_titles_by_star_rating(stars: int = Query(..., description="Star rating of the episode")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T2.stars = ?", (stars,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get names of persons based on award result and role
@app.get("/v1/law_episode/person_names_by_award_result_role", operation_id="get_person_names_by_award_result_role", summary="Retrieves the names of individuals who have achieved a specific award result in a given role. The operation filters the Person table based on the provided award result and role, returning a list of matching names.")
async def get_person_names_by_award_result_role(result: str = Query(..., description="Result of the award (e.g., Winner)"), role: str = Query(..., description="Role of the person (e.g., director)")):
    cursor.execute("SELECT T1.name FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T2.Result = ? AND T2.role = ?", (result, role))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get distinct years of awards for a specific episode title and award result
@app.get("/v1/law_episode/distinct_award_years_by_episode_title_and_result", operation_id="get_distinct_award_years_by_episode_title_and_result", summary="Retrieves a list of unique years in which a specific episode received an award, based on the provided episode title and award result. The response includes only the distinct years, excluding any duplicates.")
async def get_distinct_award_years_by_episode_title_and_result(title: str = Query(..., description="Title of the episode"), result: str = Query(..., description="Result of the award (e.g., Winner, Nominee)")):
    cursor.execute("SELECT DISTINCT T1.year FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T2.title = ? AND T1.result = ?", (title, result))
    result = cursor.fetchall()
    if not result:
        return {"years": []}
    return {"years": [row[0] for row in result]}

# Endpoint to get birth regions of persons based on their role
@app.get("/v1/law_episode/birth_regions_by_role", operation_id="get_birth_regions_by_role", summary="Retrieves the birth regions of individuals based on their specified role in the credits. The role parameter is used to filter the results.")
async def get_birth_regions_by_role(role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T2.birth_region FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T1.role = ?", (role,))
    result = cursor.fetchall()
    if not result:
        return {"birth_regions": []}
    return {"birth_regions": [row[0] for row in result]}

# Endpoint to get the count of persons in episodes based on person name
@app.get("/v1/law_episode/count_persons_in_episodes_by_name", operation_id="get_count_persons_in_episodes_by_name", summary="Retrieves the total number of episodes in which a specific person has appeared. The person is identified by their name.")
async def get_count_persons_in_episodes_by_name(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT COUNT(T3.person_id) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T3.name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get roles of persons based on award result and person name
@app.get("/v1/law_episode/roles_by_award_result_and_person_name", operation_id="get_roles_by_award_result_and_person_name", summary="Retrieves the roles of individuals based on the specified award result and person's name. The operation filters the data by matching the provided award result and person's name, then returns the corresponding roles.")
async def get_roles_by_award_result_and_person_name(result: str = Query(..., description="Result of the award (e.g., Winner, Nominee)"), name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T2.role FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T2.Result = ? AND T1.name = ?", (result, name))
    result = cursor.fetchall()
    if not result:
        return {"roles": []}
    return {"roles": [row[0] for row in result]}

# Endpoint to get the role of the tallest person in credits
@app.get("/v1/law_episode/role_of_tallest_person_in_credits", operation_id="get_role_of_tallest_person_in_credits", summary="Retrieves the role of the tallest person who has received an award in the credits. The role is determined by comparing the heights of all persons who have received an award and selecting the tallest one.")
async def get_role_of_tallest_person_in_credits():
    cursor.execute("SELECT T2.role FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.person_id = T2.person_id INNER JOIN Award AS T3 ON T2.episode_id = T3.episode_id ORDER BY T1.height_meters DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"role": []}
    return {"role": result[0]}

# Endpoint to get the episode title with the most nominations
@app.get("/v1/law_episode/episode_title_with_most_nominations", operation_id="get_episode_title_with_most_nominations", summary="Retrieves the title of the episode that has received the most nominations for a specific award result. The result parameter is used to filter the nominations by the award result, such as Winner or Nominee.")
async def get_episode_title_with_most_nominations(result: str = Query(..., description="Result of the award (e.g., Winner, Nominee)")):
    cursor.execute("SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.result = ? GROUP BY T2.episode_id ORDER BY COUNT(T1.result) DESC LIMIT 1", (result,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the rating of episodes based on person name
@app.get("/v1/law_episode/episode_ratings_by_person_name", operation_id="get_episode_ratings_by_person_name", summary="Retrieves the ratings of episodes associated with a specified person. The operation filters episodes based on the provided person's name and returns their corresponding ratings.")
async def get_episode_ratings_by_person_name(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.rating FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T3.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"ratings": []}
    return {"ratings": [row[0] for row in result]}

# Endpoint to get person names based on episode and season
@app.get("/v1/law_episode/person_names_by_episode_and_season", operation_id="get_person_names_by_episode_and_season", summary="Retrieves the names of individuals associated with a specific episode and season. The episode and season numbers are used to filter the results.")
async def get_person_names_by_episode_and_season(episode: int = Query(..., description="Episode number"), season: int = Query(..., description="Season number")):
    cursor.execute("SELECT T3.name FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T1.episode = ? AND T1.season = ?", (episode, season))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the average star rating for a person's episodes
@app.get("/v1/law_episode/average_star_rating", operation_id="get_average_star_rating", summary="Retrieves the average star rating for episodes associated with a specific person. The operation calculates the average by summing the total stars received for the episodes and dividing by the total number of episodes. The input parameters allow for filtering the results based on the number of stars and the name of the person.")
async def get_average_star_rating(stars: int = Query(..., description="Number of stars"), name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT CAST(SUM(T3.stars) AS REAL) / COUNT(T2.episode_id) FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.person_id = T2.person_id INNER JOIN Vote AS T3 ON T2.episode_id = T3.episode_id WHERE T3.stars = ? AND T1.name = ?", (stars, name))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the percentage of episodes with a specific role
@app.get("/v1/law_episode/percentage_episodes_with_role", operation_id="get_percentage_episodes_with_role", summary="Retrieves the percentage of episodes with a specified role for a given episode title. This operation calculates the proportion of episodes that have the provided role, based on the total number of episodes with the given title. The role and title are input parameters that determine the scope of the calculation.")
async def get_percentage_episodes_with_role(role: str = Query(..., description="Role of the credit"), title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.role = ? THEN 1 ELSE 0 END) AS REAL ) * 100 / COUNT(T1.episode_id) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ?", (role, title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get episode details based on star rating and number of votes
@app.get("/v1/law_episode/episode_details_by_stars_votes", operation_id="get_episode_details_by_stars_votes", summary="Retrieves details of episodes that have been rated with a specific number of stars and received a certain number of votes. The response includes the episode title, summary, and air date.")
async def get_episode_details_by_stars_votes(stars: int = Query(..., description="Number of stars"), votes: int = Query(..., description="Number of votes")):
    cursor.execute("SELECT T2.title, T2.summary, T2.air_date FROM Vote AS T1 INNER JOIN Episode AS T2 ON T2.episode_id = T1.episode_id WHERE T1.stars = ? AND T1.votes = ?", (stars, votes))
    result = cursor.fetchall()
    if not result:
        return {"episode_details": []}
    return {"episode_details": result}

# Endpoint to get episode air date and rating based on star rating and episode number
@app.get("/v1/law_episode/episode_air_date_rating", operation_id="get_episode_air_date_rating", summary="Retrieves the air date and rating of a specific episode based on the provided star rating and episode number. This operation fetches the relevant episode details from the database using the given star rating and episode number.")
async def get_episode_air_date_rating(stars: int = Query(..., description="Number of stars"), episode: int = Query(..., description="Episode number")):
    cursor.execute("SELECT T2.air_date, T2.rating FROM Vote AS T1 INNER JOIN Episode AS T2 ON T2.episode_id = T1.episode_id WHERE T1.stars = ? AND T2.episode = ?", (stars, episode))
    result = cursor.fetchall()
    if not result:
        return {"episode_details": []}
    return {"episode_details": result}

# Endpoint to get person names and roles based on award details
@app.get("/v1/law_episode/person_roles_by_award", operation_id="get_person_roles_by_award", summary="Retrieves the names and roles of individuals who have received a specific award in a given year and category. The response includes a list of names and their corresponding roles.")
async def get_person_roles_by_award(year: int = Query(..., description="Year of the award"), award_category: str = Query(..., description="Award category"), award: str = Query(..., description="Award")):
    cursor.execute("SELECT T1.name, T2.role FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T2.year = ? AND T2.award_category = ? AND T2.award = ?", (year, award_category, award))
    result = cursor.fetchall()
    if not result:
        return {"person_roles": []}
    return {"person_roles": result}

# Endpoint to get award details for a specific person
@app.get("/v1/law_episode/award_details_by_person", operation_id="get_award_details_by_person", summary="Retrieves detailed information about the awards won by a specific individual, based on their name and the result of the award. The response includes the organization that granted the award, the year it was awarded, the name of the award, and the category of the award.")
async def get_award_details_by_person(name: str = Query(..., description="Name of the person"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T2.organization, T2.year, T2.award, T2.award_category FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T1.name = ? AND T2.result = ?", (name, result))
    result = cursor.fetchall()
    if not result:
        return {"award_details": []}
    return {"award_details": result}

# Endpoint to get years and episode IDs based on award details and person name
@app.get("/v1/law_episode/years_episode_ids_by_award", operation_id="get_years_episode_ids_by_award", summary="Retrieves the years and corresponding episode IDs for a specific award, award category, and person. The operation filters results based on the award's outcome and the organization, returning only those with a minimum count of years as specified. The data is grouped by episode ID to ensure unique entries.")
async def get_years_episode_ids_by_award(award: str = Query(..., description="Award"), award_category: str = Query(..., description="Award category"), name: str = Query(..., description="Name of the person"), result: str = Query(..., description="Result of the award"), organization: str = Query(..., description="Organization"), count: int = Query(..., description="Minimum count of years") ):
    cursor.execute("SELECT t3.years, t3.episode_id FROM ( SELECT DISTINCT T2.year AS years, T2.episode_id, row_number() OVER (PARTITION BY T2.episode_id ORDER BY T2.year) AS rm FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T2.award = ? AND T2.award_category = ? AND T1.name = ? AND T2.result = ? AND T2.organization = ? ) AS T3 GROUP BY t3.episode_id HAVING COUNT(t3.years - t3.rm) >= ?", (award, award_category, name, result, organization, count))
    result = cursor.fetchall()
    if not result:
        return {"years_episode_ids": []}
    return {"years_episode_ids": result}

# Endpoint to get the count of awards for a specific episode
@app.get("/v1/law_episode/award_count_by_episode", operation_id="get_award_count_by_episode", summary="Retrieves the total number of awards for a specific episode, filtered by year, result, episode number, organization, and series. This operation provides a quantitative measure of the episode's recognition, considering various factors such as the year, outcome, and affiliated organization.")
async def get_award_count_by_episode(year: int = Query(..., description="Year of the award"), result: str = Query(..., description="Result of the award"), episode: int = Query(..., description="Episode number"), organization: str = Query(..., description="Organization"), series: str = Query(..., description="Series")):
    cursor.execute("SELECT COUNT(T2.award_id) FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T2.year = ? AND T2.result = ? AND T1.episode = ? AND T2.organization = ? AND T1.series = ?", (year, result, episode, organization, series))
    result = cursor.fetchone()
    if not result:
        return {"award_count": []}
    return {"award_count": result[0]}

# Endpoint to get episode IDs and roles based on award details and person name
@app.get("/v1/law_episode/episode_ids_roles_by_award", operation_id="get_episode_ids_roles_by_award", summary="Retrieves episode IDs and associated roles for a specific person, based on the provided award details. The operation filters results by the year, award type, organization, person's name, and award outcome.")
async def get_episode_ids_roles_by_award(year: int = Query(..., description="Year of the award"), award: str = Query(..., description="Award"), organization: str = Query(..., description="Organization"), name: str = Query(..., description="Name of the person"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T3.episode_id, T2.role FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id INNER JOIN Episode AS T3 ON T2.episode_id = T3.episode_id WHERE T2.year = ? AND T2.award = ? AND T2.organization = ? AND T1.name = ? AND T2.result = ?", (year, award, organization, name, result))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids_roles": []}
    return {"episode_ids_roles": result}

# Endpoint to get episode titles and air dates produced by a specific person
@app.get("/v1/law_episode/episodes_produced_by", operation_id="get_episodes_produced_by", summary="Retrieves the titles and air dates of episodes produced by a specific individual, based on their role and credit category. The operation filters episodes by the provided category and role, and then identifies those produced by the specified person.")
async def get_episodes_produced_by(category: str = Query(..., description="Category of the credit"), role: str = Query(..., description="Role of the person"), name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.title, T1.air_date FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T2.category = ? AND T2.role = ? AND T3.name = ?", (category, role, name))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": result}

# Endpoint to get the count of uncredited cast members from a specific country in a given episode
@app.get("/v1/law_episode/count_uncredited_cast", operation_id="get_count_uncredited_cast", summary="Retrieves the count of cast members who were not credited for their appearance in a specific episode, belonging to a certain category and hailing from a particular country.")
async def get_count_uncredited_cast(episode_id: str = Query(..., description="Episode ID"), category: str = Query(..., description="Category of the credit"), credited: str = Query(..., description="Credited status"), birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT COUNT(T1.person_id) FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T1.episode_id = ? AND T1.category = ? AND T1.credited = ? AND T2.birth_country = ?", (episode_id, category, credited, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the roles of a specific person in a given episode
@app.get("/v1/law_episode/roles_in_episode", operation_id="get_roles_in_episode", summary="Retrieves the roles played by a specific individual in a given episode. The operation requires the episode number and the person's name as input parameters. It returns a list of roles associated with the provided person in the specified episode.")
async def get_roles_in_episode(episode: int = Query(..., description="Episode number"), name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.role FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id INNER JOIN Episode AS T3 ON T1.episode_id = T3.episode_id WHERE T3.episode = ? AND T2.name = ?", (episode, name))
    result = cursor.fetchall()
    if not result:
        return {"roles": []}
    return {"roles": result}

# Endpoint to get the names of persons with a specific role in a given episode
@app.get("/v1/law_episode/names_by_role_in_episode", operation_id="get_names_by_role_in_episode", summary="Retrieve the names of individuals who have a specified role in a particular episode. The operation filters the data based on the provided episode number and role, and returns the corresponding names.")
async def get_names_by_role_in_episode(episode: int = Query(..., description="Episode number"), role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T2.name FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id INNER JOIN Episode AS T3 ON T1.episode_id = T3.episode_id WHERE T3.episode = ? AND T1.role = ?", (episode, role))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the top episodes based on star ratings
@app.get("/v1/law_episode/top_episodes_by_stars", operation_id="get_top_episodes_by_stars", summary="Retrieves the top episodes, ranked by their average star ratings within the specified range. The number of episodes returned is determined by the provided limit. The episodes are ordered from highest to lowest average star rating.")
async def get_top_episodes_by_stars(min_stars: int = Query(..., description="Minimum star rating"), max_stars: int = Query(..., description="Maximum star rating"), limit: int = Query(..., description="Number of top episodes to return")):
    cursor.execute("SELECT T2.title FROM Vote AS T1 INNER JOIN Episode AS T2 ON T2.episode_id = T1.episode_id WHERE T1.stars BETWEEN ? AND ? GROUP BY T2.title ORDER BY CAST(SUM(T1.stars * T1.percent) AS REAL) / 100 DESC LIMIT ?", (min_stars, max_stars, limit))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": result}

# Endpoint to get the percentage of cast members in a specific role from a given country in a given episode
@app.get("/v1/law_episode/cast_percentage_by_role", operation_id="get_cast_percentage_by_role", summary="Retrieves the percentage of cast members in a specific role from a given country who have been credited in a particular episode. The calculation is based on the total number of cast members in the episode and the specified role and country.")
async def get_cast_percentage_by_role(category: str = Query(..., description="Category of the credit"), episode: int = Query(..., description="Episode number"), birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.category = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.category), T1.role FROM Award AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Episode AS T3 ON T2.episode_id = T3.episode_id INNER JOIN Person AS T4 ON T2.person_id = T4.person_id WHERE T3.episode = ? AND T4.birth_country = ?", (category, episode, birth_country))
    result = cursor.fetchall()
    if not result:
        return {"percentages": []}
    return {"percentages": result}

# Endpoint to get the count of persons from a specific country who have received awards
@app.get("/v1/law_episode/count_awarded_persons_by_country", operation_id="get_count_awarded_persons_by_country", summary="Retrieves the total number of individuals from a specified country who have been awarded. The birth country of the individuals is used to filter the results.")
async def get_count_awarded_persons_by_country(birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT COUNT(T1.person_id) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id WHERE T1.birth_country = ?", (birth_country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of credits for a specific person
@app.get("/v1/law_episode/count_credits_by_person", operation_id="get_count_credits_by_person", summary="Retrieves the total number of credits associated with a specific individual. The operation requires the individual's name as input and returns the count of credits linked to that person.")
async def get_count_credits_by_person(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM Credit AS T1 INNER JOIN Person AS T2 ON T2.person_id = T1.person_id WHERE T2.name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of persons in a given episode
@app.get("/v1/law_episode/names_in_episode", operation_id="get_names_in_episode", summary="Retrieves the names of all individuals who appear in a specified episode. The episode is identified by its unique number.")
async def get_names_in_episode(episode: int = Query(..., description="Episode number")):
    cursor.execute("SELECT T3.name FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T3.person_id = T2.person_id WHERE T1.episode = ?", (episode,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the awards for a given episode with specific results
@app.get("/v1/law_episode/awards_by_episode", operation_id="get_awards_by_episode", summary="Retrieves the awards associated with a specific episode, filtered by the provided result types. The operation returns a list of awards that match the given episode number and result types.")
async def get_awards_by_episode(episode: int = Query(..., description="Episode number"), result1: str = Query(..., description="First result type"), result2: str = Query(..., description="Second result type")):
    cursor.execute("SELECT T2.award FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T1.episode = ? AND T2.result IN (?, ?)", (episode, result1, result2))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": result}

# Endpoint to get names of persons who have received more than a specified number of awards
@app.get("/v1/law_episode/persons_with_multiple_awards", operation_id="get_persons_with_multiple_awards", summary="Retrieves the names of individuals who have been honored with more than a specified number of awards. The operation considers the number of awards received in each role and returns the names of individuals who have surpassed the provided minimum threshold.")
async def get_persons_with_multiple_awards(min_awards: int = Query(..., description="Minimum number of awards received")):
    cursor.execute("SELECT T1.name FROM Person AS T1 INNER JOIN Award AS T2 ON T1.person_id = T2.person_id GROUP BY T2.role HAVING COUNT(T2.award_id) > ?", (min_awards,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get names and episode IDs of directors who won awards
@app.get("/v1/law_episode/directors_who_won_awards", operation_id="get_directors_who_won_awards", summary="Retrieves the names of directors and their corresponding episode IDs who have won awards for a specified role and result. This operation filters the data based on the provided role and result parameters, ensuring that only relevant records are returned.")
async def get_directors_who_won_awards(role: str = Query(..., description="Role of the person"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T3.name, T1.episode_id FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Person AS T3 ON T2.person_id = T3.person_id WHERE T2.role = ? AND T2.result = ?", (role, result))
    result = cursor.fetchall()
    if not result:
        return {"directors": []}
    return {"directors": [{"name": row[0], "episode_id": row[1]} for row in result]}

# Endpoint to get episodes that were nominated for awards
@app.get("/v1/law_episode/episodes_nominated_for_awards", operation_id="get_episodes_nominated_for_awards", summary="Retrieves a list of episodes that have been nominated for awards, filtered by the specified award result. The result parameter determines which nominated episodes are returned.")
async def get_episodes_nominated_for_awards(result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T1.episode FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T2.result = ?", (result,))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": [row[0] for row in result]}

# Endpoint to get the average rating of episodes in a specific season
@app.get("/v1/law_episode/average_rating_by_season", operation_id="get_average_rating_by_season", summary="Retrieves the average rating of all episodes in a specified season. The season number is provided as an input parameter, and the operation calculates the average rating by summing up the ratings of all episodes in the season and dividing by the total number of episodes in the season.")
async def get_average_rating_by_season(season: int = Query(..., description="Season number")):
    cursor.execute("SELECT SUM(rating) / COUNT(episode_id) FROM Episode WHERE season = ?", (season,))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the difference in votes between two episodes for votes with a specific star rating
@app.get("/v1/law_episode/vote_difference_between_episodes", operation_id="get_vote_difference_between_episodes", summary="Retrieves the difference in votes between two specified episodes for votes with a particular star rating. This operation compares the total votes for the first and second episodes, considering only votes with the provided star rating. The result is the difference between the total votes for the first episode and the total votes for the second episode.")
async def get_vote_difference_between_episodes(episode1: int = Query(..., description="First episode number"), episode2: int = Query(..., description="Second episode number"), stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT SUM(CASE WHEN T2.episode = ? THEN T1.votes ELSE 0 END) - SUM(CASE WHEN T2.episode = ? THEN T1.votes ELSE 0 END) FROM Vote AS T1 INNER JOIN Episode AS T2 ON T2.episode_id = T1.episode_id WHERE T1.stars = ?", (episode1, episode2, stars))
    result = cursor.fetchone()
    if not result:
        return {"vote_difference": []}
    return {"vote_difference": result[0]}

# Endpoint to get the rating of the episode with the most awards
@app.get("/v1/law_episode/rating_of_episode_with_most_awards", operation_id="get_rating_of_episode_with_most_awards", summary="Retrieves the rating of the episode that has received the most awards, specifically those with the provided award result.")
async def get_rating_of_episode_with_most_awards(result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T1.rating FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id WHERE T2.result = ? GROUP BY T1.episode_id ORDER BY COUNT(T2.award_id) DESC LIMIT 1", (result,))
    result = cursor.fetchone()
    if not result:
        return {"rating": []}
    return {"rating": result[0]}

# Endpoint to get the count of credited persons in episodes within a specific range
@app.get("/v1/law_episode/count_credited_persons_in_episode_range", operation_id="get_count_credited_persons_in_episode_range", summary="Retrieves the total number of individuals credited in episodes falling within a specified range. The operation considers the credited status and the episode range provided as input parameters.")
async def get_count_credited_persons_in_episode_range(credited: str = Query(..., description="Credited status"), start_episode: int = Query(..., description="Start episode number"), end_episode: int = Query(..., description="End episode number")):
    cursor.execute("SELECT COUNT(T1.person_id) FROM Credit AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.credited = ? AND T2.episode BETWEEN ? AND ?", (credited, start_episode, end_episode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episodes associated with a specific keyword
@app.get("/v1/law_episode/episodes_by_keyword", operation_id="get_episodes_by_keyword", summary="Retrieves all episodes linked to a specified keyword. The keyword is used to filter the episodes, returning only those that match the provided keyword.")
async def get_episodes_by_keyword(keyword: str = Query(..., description="Keyword")):
    cursor.execute("SELECT T1.episode FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T2.Keyword = ?", (keyword,))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": [row[0] for row in result]}

# Endpoint to get the person ID with the highest vote percentage
@app.get("/v1/law_episode/person_with_highest_vote_percentage", operation_id="get_person_with_highest_vote_percentage", summary="Retrieves the ID of the person who received the highest vote percentage in an episode. This operation identifies the person with the highest voting percentage by comparing votes across episodes and awards. The result is the ID of the person with the highest percentage.")
async def get_person_with_highest_vote_percentage():
    cursor.execute("SELECT T2.person_id FROM Vote AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id ORDER BY T1.percent DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"person_id": []}
    return {"person_id": result[0]}

# Endpoint to get the average rating of episodes that have received awards
@app.get("/v1/law_episode/average_rating_of_awarded_episodes", operation_id="get_average_rating_of_awarded_episodes", summary="Retrieves the average rating of episodes that have been recognized with awards. This operation calculates the mean rating of episodes that have received at least one award, providing a statistical overview of the quality of these episodes.")
async def get_average_rating_of_awarded_episodes():
    cursor.execute("SELECT SUM(T1.rating) / COUNT(T1.episode) FROM Episode AS T1 INNER JOIN Award AS T2 ON T1.episode_id = T2.episode_id")
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the count of awards based on result, award, and organization
@app.get("/v1/law_episode/award_count_by_result_award_organization", operation_id="get_award_count", summary="Retrieves the total number of awards that match the specified result, award name, and awarding organization. This operation is useful for understanding the distribution of awards based on these criteria.")
async def get_award_count(result: str = Query(..., description="Result of the award"), award: str = Query(..., description="Name of the award"), organization: str = Query(..., description="Organization that awarded") ):
    cursor.execute("SELECT COUNT(award_id) FROM Award WHERE result = ? AND award = ? AND organization = ?", (result, award, organization))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the birth country of a person by name
@app.get("/v1/law_episode/person_birth_country_by_name", operation_id="get_birth_country", summary="Retrieves the birth country of a specific person, identified by their name.")
async def get_birth_country(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT birth_country FROM Person WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"birth_country": []}
    return {"birth_country": result[0]}

# Endpoint to get the names of persons who were nominees for a specific role in a specific series
@app.get("/v1/law_episode/nominee_names_by_role_series", operation_id="get_nominee_names", summary="Retrieve the names of individuals who were nominated for a particular role within a specific series. The operation filters nominees based on the award result, the role they were nominated for, and the series they were part of.")
async def get_nominee_names(result: str = Query(..., description="Result of the award"), role: str = Query(..., description="Role in the series"), series: str = Query(..., description="Name of the series")):
    cursor.execute("SELECT T2.name FROM Award AS T1 INNER JOIN Person AS T2 ON T1.person_id = T2.person_id WHERE T1.result = ? AND T1.role = ? AND T1.series = ?", (result, role, series))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the names of persons who won a specific award
@app.get("/v1/law_episode/winner_names_by_award", operation_id="get_winner_names", summary="Retrieves the names of individuals who have won a specific award, based on the provided award name and result.")
async def get_winner_names(result: str = Query(..., description="Result of the award"), award: str = Query(..., description="Name of the award")):
    cursor.execute("SELECT T2.name FROM Award AS T1 INNER JOIN Person AS T2 ON T1.person_id = T2.person_id WHERE T1.result = ? AND T1.award = ?", (result, award))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the birthdates of persons who won an award for a specific role
@app.get("/v1/law_episode/birthdates_by_role", operation_id="get_birthdates_by_role", summary="Retrieve the birthdates of individuals who have been awarded for a specific role. The operation filters the award recipients based on the provided role and returns their birthdates.")
async def get_birthdates_by_role(role: str = Query(..., description="Role in the award")):
    cursor.execute("SELECT T2.birthdate FROM Award AS T1 INNER JOIN Person AS T2 ON T1.person_id = T2.person_id WHERE T1.role = ?", (role,))
    result = cursor.fetchall()
    if not result:
        return {"birthdates": []}
    return {"birthdates": [row[0] for row in result]}

# Endpoint to get the titles of episodes that won a specific award
@app.get("/v1/law_episode/episode_titles_by_award", operation_id="get_episode_titles", summary="Retrieves the titles of episodes that have been awarded a specific honor. The operation filters episodes based on the provided award name and returns their respective titles.")
async def get_episode_titles(award: str = Query(..., description="Name of the award")):
    cursor.execute("SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.award = ?", (award,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the title of the episode with the highest number of votes
@app.get("/v1/law_episode/top_voted_episode", operation_id="get_top_voted_episode", summary="Retrieves the title of the episode that has received the most votes. This operation identifies the episode with the highest cumulative vote count and returns its title. The result is determined by aggregating votes across all episodes and ranking them in descending order.")
async def get_top_voted_episode():
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id GROUP BY T1.title ORDER BY SUM(T1.votes) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the names of persons credited for a specific role
@app.get("/v1/law_episode/credited_person_names_by_role", operation_id="get_credited_person_names", summary="Retrieves the names of individuals who have been credited for a specific role, based on the provided role and credited status.")
async def get_credited_person_names(role: str = Query(..., description="Role in the credit"), credited: str = Query(..., description="Credited status (true or false)")):
    cursor.execute("SELECT T1.name FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.person_id = T2.person_id WHERE T2.role = ? AND T2.credited = ?", (role, credited))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the distinct birth places of a person based on their role and credited status
@app.get("/v1/law_episode/distinct_birth_places_by_person_role_credited", operation_id="get_distinct_birth_places", summary="Retrieves a unique list of birth places for a specific individual, based on their role and credited status in the credits.")
async def get_distinct_birth_places(person_id: str = Query(..., description="Person ID"), role: str = Query(..., description="Role in the credit"), credited: str = Query(..., description="Credited status (true or false)")):
    cursor.execute("SELECT DISTINCT T1.birth_place FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.person_id = T2.person_id WHERE T1.person_id = ? AND T2.role = ? AND T2.credited = ?", (person_id, role, credited))
    result = cursor.fetchall()
    if not result:
        return {"birth_places": []}
    return {"birth_places": [row[0] for row in result]}

# Endpoint to get the names of persons who won an award and are taller than a specified height
@app.get("/v1/law_episode/winner_names_by_height", operation_id="get_winner_names_by_height", summary="Retrieve the names of individuals who have won an award with a specified result and have a height greater than a given value in meters.")
async def get_winner_names_by_height(result: str = Query(..., description="Result of the award"), height_meters: float = Query(..., description="Height in meters")):
    cursor.execute("SELECT T2.name FROM Award AS T1 INNER JOIN Person AS T2 ON T1.person_id = T2.person_id WHERE T1.result = ? AND T2.height_meters > ?", (result, height_meters))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get top-rated award-winning episodes
@app.get("/v1/law_episode/top_rated_award_winning_episodes", operation_id="get_top_rated_award_winning_episodes", summary="Retrieves the top-rated episodes that have won a specific award, ordered by their rating. The award and its result are used to filter the episodes, and the number of episodes returned can be limited.")
async def get_top_rated_award_winning_episodes(award: str = Query(..., description="Name of the award"), result: str = Query(..., description="Result of the award (e.g., Winner)"), limit: int = Query(..., description="Number of top-rated episodes to return")):
    cursor.execute("SELECT T2.episode_id FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.award = ? AND T1.result = ? ORDER BY T2.rating DESC LIMIT ?", (award, result, limit))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

api_calls = [
    "/v1/law_episode/keywords_by_episode_title?title=Refuge:%20Part%201",
    "/v1/law_episode/keyword_count_by_season_episode?season=9&episode=23",
    "/v1/law_episode/episode_titles_by_keyword?keyword=laundering%20money",
    "/v1/law_episode/keywords_by_rating?rating=8",
    "/v1/law_episode/votes_by_title_and_stars?title=Cherished&stars=10",
    "/v1/law_episode/total_votes_by_title?title=Cherished",
    "/v1/law_episode/top_episode_by_stars?stars=10",
    "/v1/law_episode/roles_by_title_name_credited?title=Cherished&name=Park%20Dietz&credited=true",
    "/v1/law_episode/episode_count_by_title_credited?title=Cherished&credited=false",
    "/v1/law_episode/person_names_by_title_role_credited?title=Cherished&credited=true&role=technical%20advisor",
    "/v1/law_episode/count_persons_credited_in_episodes?credited=true&name=Park%20Dietz",
    "/v1/law_episode/episode_titles_by_credited_person?credited=true&name=Park%20Dietz",
    "/v1/law_episode/credited_status_by_person_episode?name=Anthony%20Azzara&episode_id=tt0629204",
    "/v1/law_episode/count_keywords_by_episode_title?title=Disciple",
    "/v1/law_episode/top_episode_by_star_rating?stars=1",
    "/v1/law_episode/count_awards_by_series_season_episode?series=Law%20and%20Order&season=9&episode=20",
    "/v1/law_episode/count_roles_by_series_season_episode_credited?series=Law%20and%20Order&season=9&episode=17&credited=true",
    "/v1/law_episode/episode_summary_by_award_id?award_id=296",
    "/v1/law_episode/roles_by_person_name?name=Joseph%20Blair",
    "/v1/law_episode/count_awards_by_person_name?name=Rene%20Balcer",
    "/v1/law_episode/air_date_highest_votes?limit=1",
    "/v1/law_episode/person_names_by_award?award_id=313",
    "/v1/law_episode/role_count_by_person?name=J.K.%20Simmons",
    "/v1/law_episode/votes_by_episode_title_and_stars?stars=9&title=Sideshow",
    "/v1/law_episode/episode_title_ratio?title1=Refuge:%20Part%201&title2=Shield",
    "/v1/law_episode/credited_cast_ratio?category=Cast&credited=true&start_episode=185&end_episode=193",
    "/v1/law_episode/uncredited_persons_by_episode?credited=false&episode_id=tt0629391",
    "/v1/law_episode/person_count_by_min_awards?result=Winner&min_awards=3",
    "/v1/law_episode/person_names_by_role_and_episode?episode_id=tt0629204&role=script%20supervisor",
    "/v1/law_episode/award_count_by_person_and_result?name=Julia%20Roberts&result=Nominee",
    "/v1/law_episode/tallest_person_by_role?role=camera%20operator",
    "/v1/law_episode/count_persons_by_country_and_year?year=1999&birth_country=Canada",
    "/v1/law_episode/count_episodes_by_star_rating?stars=10",
    "/v1/law_episode/percentage_votes_by_star_rating_and_episode?stars=1&title=True%20North&episode_id=tt0629477",
    "/v1/law_episode/episode_with_most_keywords",
    "/v1/law_episode/count_episodes_by_year_organization_result?year=1998&organization=International%20Monitor%20Awards&result=Winner",
    "/v1/law_episode/count_awards_by_episode_title_and_result?title=Agony&result=Winner",
    "/v1/law_episode/count_and_percentage_episodes_by_season_series_category?season=9&category=Cast&series=Law%20and%20Order",
    "/v1/law_episode/keyword_of_second_highest_votes",
    "/v1/law_episode/count_awards_by_title_result?title=Agony&result=Winner",
    "/v1/law_episode/get_person_names_by_title_role?title=Flight&role=Narrator",
    "/v1/law_episode/top_organization_by_person_name_result?name=Constantine%20Makris&result=Winner",
    "/v1/law_episode/get_person_names_by_episode_role?episode=3&role=stunt%20coordinator",
    "/v1/law_episode/count_persons_by_title_credited?title=Admissions&credited=false",
    "/v1/law_episode/top_episode_by_credit_category?category=Art%20Department",
    "/v1/law_episode/top_episodes_by_votes_stars?votes=30&stars=10",
    "/v1/law_episode/oldest_person_by_role?role=Clerk",
    "/v1/law_episode/count_episodes_by_episode_stars?episode=24&stars=1",
    "/v1/law_episode/person_names_by_credit_category?category=Cast",
    "/v1/law_episode/top_person_by_credits",
    "/v1/law_episode/distinct_episode_ids_by_award_category?award_category=Primetime%20Emmy",
    "/v1/law_episode/award_count_by_result?result=Nominee",
    "/v1/law_episode/distinct_roles_by_credited_status?credited=false",
    "/v1/law_episode/episode_titles_by_rating?limit=3",
    "/v1/law_episode/birth_place_region_by_birth_name?birth_name=Rene%20Chenevert%20Balcer",
    "/v1/law_episode/person_names_by_birth_country?birth_country=USA",
    "/v1/law_episode/episode_titles_by_star_rating?stars=1",
    "/v1/law_episode/person_names_by_award_result_role?result=Winner&role=director",
    "/v1/law_episode/distinct_award_years_by_episode_title_and_result?title=DWB&result=Winner",
    "/v1/law_episode/birth_regions_by_role?role=president%20of%20NBC%20West%20Coast",
    "/v1/law_episode/count_persons_in_episodes_by_name?name=Donna%20Villella",
    "/v1/law_episode/roles_by_award_result_and_person_name?result=Nominee&name=Julia%20Roberts",
    "/v1/law_episode/role_of_tallest_person_in_credits",
    "/v1/law_episode/episode_title_with_most_nominations?result=Nominee",
    "/v1/law_episode/episode_ratings_by_person_name?name=Jace%20Alexander",
    "/v1/law_episode/person_names_by_episode_and_season?episode=19&season=9",
    "/v1/law_episode/average_star_rating?stars=1&name=Jim%20Bracchitta",
    "/v1/law_episode/percentage_episodes_with_role?role=Additional%20Crew&title=True%20North",
    "/v1/law_episode/episode_details_by_stars_votes?stars=10&votes=72",
    "/v1/law_episode/episode_air_date_rating?stars=6&episode=12",
    "/v1/law_episode/person_roles_by_award?year=2000&award_category=Edgar&award=Best%20Television%20Episode",
    "/v1/law_episode/award_details_by_person?name=Rene%20Balcer&result=Winner",
    "/v1/law_episode/years_episode_ids_by_award?award=Television&award_category=Silver%20Gavel%20Award&name=Constantine%20Makris&result=Winner&organization=American%20Bar%20Association%20Silver%20Gavel%20Awards%20for%20Media%20and%20the%20Arts&count=2",
    "/v1/law_episode/award_count_by_episode?year=1999&result=Nominee&episode=20&organization=Primetime%20Emmy%20Awards&series=Law%20and%20Order",
    "/v1/law_episode/episode_ids_roles_by_award?year=1999&award=Outstanding%20Guest%20Actress%20in%20a%20Drama%20Series&organization=Primetime%20Emmy%20Awards&name=Julia%20Roberts&result=Nominee",
    "/v1/law_episode/episodes_produced_by?category=Produced%20by&role=producer&name=Billy%20Fox",
    "/v1/law_episode/count_uncredited_cast?episode_id=tt0629228&category=Cast&credited=false&birth_country=USA",
    "/v1/law_episode/roles_in_episode?episode=9&name=Jason%20Kuschner",
    "/v1/law_episode/names_by_role_in_episode?episode=1&role=president%20of%20NBC%20West%20Coast",
    "/v1/law_episode/top_episodes_by_stars?min_stars=1&max_stars=10&limit=3",
    "/v1/law_episode/cast_percentage_by_role?category=Cast&episode=2&birth_country=USA",
    "/v1/law_episode/count_awarded_persons_by_country?birth_country=Canada",
    "/v1/law_episode/count_credits_by_person?name=Jerry%20Orbach",
    "/v1/law_episode/names_in_episode?episode=9",
    "/v1/law_episode/awards_by_episode?episode=20&result1=Winner&result2=Nominee",
    "/v1/law_episode/persons_with_multiple_awards?min_awards=1",
    "/v1/law_episode/directors_who_won_awards?role=director&result=Winner",
    "/v1/law_episode/episodes_nominated_for_awards?result=Nominee",
    "/v1/law_episode/average_rating_by_season?season=9",
    "/v1/law_episode/vote_difference_between_episodes?episode1=24&episode2=1&stars=10",
    "/v1/law_episode/rating_of_episode_with_most_awards?result=Winner",
    "/v1/law_episode/count_credited_persons_in_episode_range?credited=true&start_episode=1&end_episode=10",
    "/v1/law_episode/episodes_by_keyword?keyword=mafia",
    "/v1/law_episode/person_with_highest_vote_percentage",
    "/v1/law_episode/average_rating_of_awarded_episodes",
    "/v1/law_episode/award_count_by_result_award_organization?result=Winner&award=Television&organization=American%20Bar%20Association%20Silver%20Gavel%20Awards%20for%20Media%20and%20the%20Arts",
    "/v1/law_episode/person_birth_country_by_name?name=Michael%20Preston",
    "/v1/law_episode/nominee_names_by_role_series?result=Nominee&role=Katrina%20Ludlow&series=Law%20and%20Order",
    "/v1/law_episode/winner_names_by_award?result=Winner&award=Best%20Television%20Episode",
    "/v1/law_episode/birthdates_by_role?role=writer",
    "/v1/law_episode/episode_titles_by_award?award=Outstanding%20Costume%20Design%20for%20a%20Series",
    "/v1/law_episode/top_voted_episode",
    "/v1/law_episode/credited_person_names_by_role?role=Alex%20Brown&credited=true",
    "/v1/law_episode/distinct_birth_places_by_person_role_credited?person_id=nm0007064&role=Narrator&credited=false",
    "/v1/law_episode/winner_names_by_height?result=Winner&height_meters=1.80",
    "/v1/law_episode/top_rated_award_winning_episodes?award=Best%20Television%20Episode&result=Winner&limit=2"
]
