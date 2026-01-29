from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/simpson_episodes/simpson_episodes.sqlite')
cursor = conn.cursor()

# Endpoint to get the name of the person with the earliest birthdate
@app.get("/v1/simpson_episodes/earliest_birthdate_person", operation_id="get_earliest_birthdate_person", summary="Retrieves the name of the person with the earliest birthdate from the database. This operation returns the name of the person who was born first, based on the available birthdate data.")
async def get_earliest_birthdate_person():
    cursor.execute("SELECT name FROM Person WHERE birthdate IS NOT NULL ORDER BY birthdate ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the nickname of a person by their name
@app.get("/v1/simpson_episodes/nickname_by_name", operation_id="get_nickname_by_name", summary="Retrieves the nickname associated with a specific person's name. The operation requires the person's name as input and returns the corresponding nickname from the database.")
async def get_nickname_by_name(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT nickname FROM Person WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"nickname": []}
    return {"nickname": result[0]}

# Endpoint to get the count of people born in a specific region after a certain year
@app.get("/v1/simpson_episodes/count_people_by_region_and_year", operation_id="get_count_people_by_region_and_year", summary="Retrieves the total number of individuals born in a specified region after a given year. The birth region and the year are provided as input parameters.")
async def get_count_people_by_region_and_year(birth_region: str = Query(..., description="Birth region of the person"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(name) FROM Person WHERE birth_region = ? AND SUBSTR(birthdate, 1, 4) > ?", (birth_region, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the birth country of award winners for a specific award, year, and result
@app.get("/v1/simpson_episodes/birth_country_of_award_winners", operation_id="get_birth_country_of_award_winners", summary="Retrieves the birth country of individuals who have won a specific award in a given year. The operation requires the name of the award, the year it was awarded, and the result of the award (e.g., 'Winner').")
async def get_birth_country_of_award_winners(award: str = Query(..., description="Name of the award"), year: int = Query(..., description="Year of the award"), result: str = Query(..., description="Result of the award (e.g., 'Winner')")):
    cursor.execute("SELECT T1.birth_country FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.award = ? AND T2.year = ? AND T2.result = ?", (award, year, result))
    result = cursor.fetchone()
    if not result:
        return {"birth_country": []}
    return {"birth_country": result[0]}

# Endpoint to get the awards won by a person with a specific nickname
@app.get("/v1/simpson_episodes/awards_by_nickname", operation_id="get_awards_by_nickname", summary="Retrieves the awards won by a person with a specific nickname. The operation requires the nickname of the person and the result of the award (e.g., 'Winner') as input parameters. It returns a list of awards that match the provided nickname and result.")
async def get_awards_by_nickname(nickname: str = Query(..., description="Nickname of the person"), result: str = Query(..., description="Result of the award (e.g., 'Winner')")):
    cursor.execute("SELECT T2.award FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T1.nickname = ? AND T2.result = ?", (nickname, result))
    result = cursor.fetchone()
    if not result:
        return {"award": []}
    return {"award": result[0]}

# Endpoint to get the count of award nominees from a specific country for a specific award and year
@app.get("/v1/simpson_episodes/count_award_nominees", operation_id="get_count_award_nominees", summary="Retrieves the total number of individuals from a specified country who have been nominated for a particular award in a given year.")
async def get_count_award_nominees(birth_country: str = Query(..., description="Birth country of the person"), result: str = Query(..., description="Result of the award (e.g., 'Nominee')"), award: str = Query(..., description="Name of the award"), year: int = Query(..., description="Year of the award")):
    cursor.execute("SELECT COUNT(*) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T1.birth_country = ? AND T2.result = ? AND T2.award = ? AND T2.year = ?", (birth_country, result, award, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct characters who won a specific award in a specific year
@app.get("/v1/simpson_episodes/distinct_characters_by_award", operation_id="get_distinct_characters_by_award", summary="Retrieves a unique list of characters who have been awarded a specific honor in a given year. The operation filters the results based on the award name, year, and outcome (e.g., winner).")
async def get_distinct_characters_by_award(award: str = Query(..., description="Name of the award"), year: int = Query(..., description="Year of the award"), result: str = Query(..., description="Result of the award (e.g., 'Winner')")):
    cursor.execute("SELECT DISTINCT T1.character FROM Character_Award AS T1 INNER JOIN Award AS T2 ON T1.award_id = T2.award_id WHERE T2.award = ? AND T2.year = ? AND T2.result = ?", (award, year, result))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get keywords for an episode by its title
@app.get("/v1/simpson_episodes/keywords_by_episode_title", operation_id="get_keywords_by_episode_title", summary="Retrieves a list of keywords associated with a specific episode of The Simpsons, based on the provided episode title.")
async def get_keywords_by_episode_title(title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT T2.keyword FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get the count of keywords for an episode by its air date
@app.get("/v1/simpson_episodes/count_keywords_by_air_date", operation_id="get_count_keywords_by_air_date", summary="Retrieves the total number of keywords associated with an episode, based on the provided air date. The air date should be formatted as 'YYYY-MM-DD'.")
async def get_count_keywords_by_air_date(air_date: str = Query(..., description="Air date of the episode in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T2.keyword) FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.air_date = ?", (air_date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the rating of an episode associated with a specific award, year, and person
@app.get("/v1/simpson_episodes/episode_rating_by_award", operation_id="get_episode_rating_by_award", summary="Retrieves the rating of a Simpsons episode that has been awarded a specific accolade in a given year to a particular individual. The operation requires the name of the award, the year it was granted (in YYYY format), and the recipient's name to locate the corresponding episode and return its rating.")
async def get_episode_rating_by_award(award: str = Query(..., description="Name of the award"), year: str = Query(..., description="Year of the award in 'YYYY' format"), person: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T2.rating FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.award = ? AND SUBSTR(T1.year, 1, 4) = ? AND T1.person = ?", (award, year, person))
    result = cursor.fetchone()
    if not result:
        return {"rating": []}
    return {"rating": result[0]}

# Endpoint to get the count of votes for a specific episode title and star rating
@app.get("/v1/simpson_episodes/count_votes_by_title_and_stars", operation_id="get_count_votes_by_title_and_stars", summary="Retrieves the total number of votes for a specific Simpsons episode, based on its title and the given star rating. This operation provides a quantitative measure of user preferences for a particular episode and its associated star rating.")
async def get_count_votes_by_title_and_stars(title: str = Query(..., description="Title of the episode"), stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT COUNT(*) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ? AND T2.stars = ?", (title, stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest star rating for a specific episode title
@app.get("/v1/simpson_episodes/highest_star_rating_by_title", operation_id="get_highest_star_rating_by_title", summary="Retrieves the highest star rating for a given episode title. The rating is determined by the highest number of votes received for that specific episode title. The episode title is provided as an input parameter.")
async def get_highest_star_rating_by_title(title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT T2.stars FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ? ORDER BY T2.votes DESC LIMIT 1", (title,))
    result = cursor.fetchone()
    if not result:
        return {"stars": []}
    return {"stars": result[0]}

# Endpoint to get episode titles with votes greater than a specified number and a specific star rating
@app.get("/v1/simpson_episodes/episode_titles_by_votes_and_stars", operation_id="get_episode_titles_by_votes_and_stars", summary="Retrieves the titles of episodes that have received a specified number of votes and a particular star rating. The operation filters episodes based on the provided vote count and star rating, returning only those that meet the criteria.")
async def get_episode_titles_by_votes_and_stars(votes: int = Query(..., description="Number of votes"), stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.votes > ? AND T2.stars = ?", (votes, stars))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of episodes aired in a specific year with a specific star rating and percentage
@app.get("/v1/simpson_episodes/count_episodes_by_year_stars_percent", operation_id="get_count_episodes_by_year_stars_percent", summary="Retrieves the total number of episodes that aired in a specified year, with a given star rating and a minimum percentage. The year should be provided in 'YYYY' format, the star rating is a numerical value, and the percentage is a decimal representing the minimum acceptable value.")
async def get_count_episodes_by_year_stars_percent(year: str = Query(..., description="Year in 'YYYY' format"), stars: int = Query(..., description="Star rating"), percent: float = Query(..., description="Percentage")):
    cursor.execute("SELECT COUNT(*) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE SUBSTR(T1.air_date, 1, 4) = ? AND T2.stars = ? AND T2.percent > ?", (year, stars, percent))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the episode title with the highest votes for a specific star rating
@app.get("/v1/simpson_episodes/highest_votes_episode_by_stars", operation_id="get_highest_votes_episode_by_stars", summary="Retrieves the title of the episode that has received the most votes for a given star rating. The star rating is a measure of the episode's quality, as rated by users.")
async def get_highest_votes_episode_by_stars(stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ? ORDER BY T2.votes DESC LIMIT 1", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the weighted average star rating for a specific episode title
@app.get("/v1/simpson_episodes/weighted_average_stars_by_title", operation_id="get_weighted_average_stars_by_title", summary="Retrieves the weighted average star rating for a specific Simpsons episode, calculated by summing the product of each vote and its corresponding star rating, then dividing by the total number of votes for that episode.")
async def get_weighted_average_stars_by_title(title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT CAST(SUM(T2.votes * T2.stars) AS REAL) / SUM(T2.votes) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"average_stars": []}
    return {"average_stars": result[0]}

# Endpoint to get the percentage of episodes with a specific award category and result for episodes with a rating greater than a specified value
@app.get("/v1/simpson_episodes/percentage_award_category_by_rating_and_result", operation_id="get_percentage_award_category_by_rating_and_result", summary="Retrieves the percentage of episodes that have won a specific award category and result, out of all episodes with a rating higher than the provided value.")
async def get_percentage_award_category_by_rating_and_result(award_category: str = Query(..., description="Award category"), rating: float = Query(..., description="Rating"), result: str = Query(..., description="Result")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.award_category = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T2.rating > ? AND T1.result = ?", (award_category, rating, result))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get episode titles for a specific person and role
@app.get("/v1/simpson_episodes/episode_titles_by_person_and_role", operation_id="get_episode_titles_by_person_and_role", summary="Retrieves a list of episode titles featuring a specific individual in a given role. The operation filters episodes based on the provided person's name and their role, returning only the titles that meet the criteria.")
async def get_episode_titles_by_person_and_role(person: str = Query(..., description="Name of the person"), role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.person = ? AND T2.role = ?", (person, role))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get role, episode, and number in series for a specific person and episode title
@app.get("/v1/simpson_episodes/role_episode_number_by_person_and_title", operation_id="get_role_episode_number_by_person_and_title", summary="Retrieves the role, episode, and its number in the series for a given person and episode title. The person's name and the episode's title are required as input parameters.")
async def get_role_episode_number_by_person_and_title(person: str = Query(..., description="Name of the person"), title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT T2.role, T1.episode, T1.number_in_series FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.person = ? AND T1.title = ?", (person, title))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": [{"role": row[0], "episode": row[1], "number_in_series": row[2]} for row in result]}

# Endpoint to get episode titles and summaries for a specific keyword
@app.get("/v1/simpson_episodes/episode_titles_summaries_by_keyword", operation_id="get_episode_titles_summaries_by_keyword", summary="Retrieve a list of episode titles and their corresponding summaries that contain a specific keyword. The keyword is used to filter the episodes and return only those that match the provided keyword.")
async def get_episode_titles_summaries_by_keyword(keyword: str = Query(..., description="Keyword")):
    cursor.execute("SELECT T1.title, T1.summary FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T2.keyword = ?", (keyword,))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": [{"title": row[0], "summary": row[1]} for row in result]}

# Endpoint to get the average stars for an episode by title
@app.get("/v1/simpson_episodes/average_stars_by_title", operation_id="get_average_stars_by_title", summary="Retrieves the average star rating for a specific Simpsons episode, based on its title. The title of the episode is provided as an input parameter.")
async def get_average_stars_by_title(title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT AVG(T2.stars) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"average_stars": []}
    return {"average_stars": result[0]}

# Endpoint to get episode titles and keywords by air date
@app.get("/v1/simpson_episodes/episode_keywords_by_air_date", operation_id="get_episode_keywords_by_air_date", summary="Retrieves the titles and associated keywords of episodes that aired on a specific date. The date should be provided in 'YYYY-MM-DD' format.")
async def get_episode_keywords_by_air_date(air_date: str = Query(..., description="Air date of the episode in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.title, T2.keyword FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.air_date = ?", (air_date,))
    result = cursor.fetchall()
    if not result:
        return {"episode_keywords": []}
    return {"episode_keywords": result}

# Endpoint to get distinct birth names of persons by role
@app.get("/v1/simpson_episodes/birth_names_by_role", operation_id="get_birth_names_by_role", summary="Retrieve a unique list of birth names for individuals who have been credited with a specific role in the Simpsons episodes. The role is provided as an input parameter.")
async def get_birth_names_by_role(role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT DISTINCT T1.birth_name FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T2.role = ?", (role,))
    result = cursor.fetchall()
    if not result:
        return {"birth_names": []}
    return {"birth_names": result}

# Endpoint to get the percentage of uncredited roles in award-winning episodes
@app.get("/v1/simpson_episodes/uncredited_roles_percentage", operation_id="get_uncredited_roles_percentage", summary="Retrieves the percentage of uncredited roles in award-winning episodes from a specified year, award category, award, and result. The response includes the percentage, episode title, and person involved.")
async def get_uncredited_roles_percentage(year: str = Query(..., description="Year in 'YYYY' format"), award_category: str = Query(..., description="Award category"), award: str = Query(..., description="Award"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.credited = 'false' THEN 1 END) AS REAL) * 100 / SUM(CASE WHEN T2.credited = 'true' THEN 1 END), T3.title, T2.person FROM Award AS T1 INNER JOIN Credit AS T2 ON T2.episode_id = T1.episode_id INNER JOIN Episode AS T3 ON T1.episode_id = T3.episode_id WHERE SUBSTR(T1.year, 1, 4) = ? AND T1.award_category = ? AND T1.award = ? AND T1.result = ?", (year, award_category, award, result))
    result = cursor.fetchall()
    if not result:
        return {"uncredited_roles_percentage": []}
    return {"uncredited_roles_percentage": result}

# Endpoint to get the count of episodes with votes greater than a specified number
@app.get("/v1/simpson_episodes/episode_count_by_votes", operation_id="get_episode_count_by_votes", summary="Retrieves the total number of episodes that have received more than the specified minimum number of votes.")
async def get_episode_count_by_votes(min_votes: int = Query(..., description="Minimum number of votes")):
    cursor.execute("SELECT COUNT(episode_id) FROM Episode WHERE votes > ?", (min_votes,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons based on birth place and country
@app.get("/v1/simpson_episodes/person_count_by_birth_place_country", operation_id="get_person_count_by_birth_place_country", summary="Retrieves the total count of individuals whose birth place and country match the provided input parameters. This operation is useful for obtaining a quantitative overview of the population distribution based on birth location.")
async def get_person_count_by_birth_place_country(birth_place: str = Query(..., description="Birth place of the person"), birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT COUNT(name) FROM Person WHERE birth_place = ? AND birth_country = ?", (birth_place, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get award details by person
@app.get("/v1/simpson_episodes/awards_by_person", operation_id="get_awards_by_person", summary="Retrieves the award details, including the award ID and category, for a specified person.")
async def get_awards_by_person(person: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT award_id, award_category FROM Award WHERE person = ?", (person,))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": result}

# Endpoint to get the count of persons with a nickname
@app.get("/v1/simpson_episodes/person_count_with_nickname", operation_id="get_person_count_with_nickname", summary="Retrieves the total number of individuals with a registered nickname.")
async def get_person_count_with_nickname():
    cursor.execute("SELECT COUNT(name) FROM Person WHERE nickname IS NOT NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average height of persons
@app.get("/v1/simpson_episodes/average_height", operation_id="get_average_height", summary="Retrieves the average height of all persons in the database, expressed in meters.")
async def get_average_height():
    cursor.execute("SELECT AVG(height_meters) FROM Person")
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the difference between maximum and minimum votes
@app.get("/v1/simpson_episodes/vote_difference", operation_id="get_vote_difference", summary="Retrieves the range of votes by calculating the difference between the highest and lowest vote counts.")
async def get_vote_difference():
    cursor.execute("SELECT MAX(votes) - MIN(votes) FROM Vote")
    result = cursor.fetchone()
    if not result:
        return {"vote_difference": []}
    return {"vote_difference": result[0]}

# Endpoint to get characters associated with a specific award in a given year
@app.get("/v1/simpson_episodes/characters_by_award_year", operation_id="get_characters_by_award_year", summary="Retrieves the characters who have received a specific award in a given year. The operation requires the year and name of the award as input parameters.")
async def get_characters_by_award_year(year: int = Query(..., description="Year of the award"), award: str = Query(..., description="Name of the award")):
    cursor.execute("SELECT T2.character FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T1.year = ? AND T1.award = ?", (year, award))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get the count of people from a specific region who won a specific award in a given year
@app.get("/v1/simpson_episodes/count_people_by_award_year_region", operation_id="get_count_people_by_award_year_region", summary="Retrieves the number of individuals from a specified region who received a particular award in a given year. The operation requires the year of the award, the name of the award, and the birth region of the individuals as input parameters.")
async def get_count_people_by_award_year_region(year: int = Query(..., description="Year of the award"), award: str = Query(..., description="Name of the award"), birth_region: str = Query(..., description="Birth region of the person")):
    cursor.execute("SELECT COUNT(*) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.year = ? AND T2.award = ? AND T1.birth_region = ?", (year, award, birth_region))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episode titles that won a specific award in a given year
@app.get("/v1/simpson_episodes/episode_titles_by_award_year_result", operation_id="get_episode_titles_by_award_year_result", summary="Retrieves the titles of episodes that won a specific award in a given year. The operation filters episodes based on the year, award name, and award result, and returns the corresponding episode titles.")
async def get_episode_titles_by_award_year_result(year: str = Query(..., description="Year of the award (YYYY)"), award: str = Query(..., description="Name of the award"), result: str = Query(..., description="Result of the award (e.g., 'Winner')")):
    cursor.execute("SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE SUBSTR(T1.year, 1, 4) = ? AND T1.award = ? AND T1.result = ?", (year, award, result))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get people credited in a specific episode
@app.get("/v1/simpson_episodes/people_credited_in_episode", operation_id="get_people_credited_in_episode", summary="Retrieves the list of individuals credited in a specific episode based on the provided episode title and credited status. The operation filters the episode by its title and the credited status of the individuals involved.")
async def get_people_credited_in_episode(title: str = Query(..., description="Title of the episode"), credited: str = Query(..., description="Credited status (e.g., 'false')")):
    cursor.execute("SELECT T2.person FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T2.credited = ?", (title, credited))
    result = cursor.fetchall()
    if not result:
        return {"people": []}
    return {"people": [row[0] for row in result]}

# Endpoint to get distinct episode titles based on keywords
@app.get("/v1/simpson_episodes/episode_titles_by_keywords", operation_id="get_episode_titles_by_keywords", summary="Retrieves a list of unique episode titles that contain the provided keywords. The endpoint accepts two keywords as input parameters, which are used to filter the episodes and return only those that match the specified keywords.")
async def get_episode_titles_by_keywords(keyword1: str = Query(..., description="First keyword"), keyword2: str = Query(..., description="Second keyword")):
    cursor.execute("SELECT DISTINCT T1.title FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T2.keyword IN (?, ?)", (keyword1, keyword2))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the top-voted episode with a specific star rating
@app.get("/v1/simpson_episodes/top_voted_episode_by_stars", operation_id="get_top_voted_episode_by_stars", summary="Retrieves the title of the episode with the highest number of votes for a given star rating. The star rating is a user-defined value that represents the quality of the episode.")
async def get_top_voted_episode_by_stars(stars: int = Query(..., description="Star rating of the episode")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ? ORDER BY T1.votes DESC LIMIT 1", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get distinct people credited in episodes aired in specific months
@app.get("/v1/simpson_episodes/people_credited_by_air_months", operation_id="get_people_credited_by_air_months", summary="Retrieves a list of unique individuals who have been credited in episodes that aired between the specified start and end months. The response includes only distinct names, excluding any duplicates.")
async def get_people_credited_by_air_months(start_month: str = Query(..., description="Start month (MM)"), end_month: str = Query(..., description="End month (MM)")):
    cursor.execute("SELECT DISTINCT T2.person FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE SUBSTR(T1.air_date, 6, 2) BETWEEN ? AND ?", (start_month, end_month))
    result = cursor.fetchall()
    if not result:
        return {"people": []}
    return {"people": [row[0] for row in result]}

# Endpoint to get people credited with a specific role in a specific episode
@app.get("/v1/simpson_episodes/people_by_role_in_episode", operation_id="get_people_by_role_in_episode", summary="Retrieves the names of individuals who have been credited with a specific role in a given episode of The Simpsons. The operation requires the title of the episode and the role of the person as input parameters.")
async def get_people_by_role_in_episode(title: str = Query(..., description="Title of the episode"), role: str = Query(..., description="Role of the person (e.g., 'director')")):
    cursor.execute("SELECT T2.person FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T2.role = ?", (title, role))
    result = cursor.fetchall()
    if not result:
        return {"people": []}
    return {"people": [row[0] for row in result]}

# Endpoint to get the count of distinct roles in a specific episode
@app.get("/v1/simpson_episodes/count_distinct_roles_in_episode", operation_id="get_count_distinct_roles_in_episode", summary="Retrieves the total number of unique roles that appear in a specific episode of the Simpsons. The episode is identified by its number.")
async def get_count_distinct_roles_in_episode(episode: int = Query(..., description="Episode number")):
    cursor.execute("SELECT COUNT(DISTINCT T2.role) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.episode = ?", (episode,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of episodes with a specific award and votes greater than a specified number in a given year
@app.get("/v1/simpson_episodes/percentage_episodes_with_award", operation_id="get_percentage_episodes_with_award", summary="Retrieves the percentage of episodes that have received a specified award and have more than a certain number of votes in a given year. The calculation is based on the total number of episodes in the specified year.")
async def get_percentage_episodes_with_award(award: str = Query(..., description="Award name"), min_votes: int = Query(..., description="Minimum number of votes"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.award = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.episode_id) FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T2.votes > ? AND T1.year = ?", (award, min_votes, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of votes between two episodes with a specific number of stars
@app.get("/v1/simpson_episodes/vote_ratio_between_episodes", operation_id="get_vote_ratio_between_episodes", summary="Retrieves the ratio of votes between two specified episodes, filtered by a given number of stars. The operation compares the votes of the first episode, 'No Loan Again, Naturally', with the votes of the second episode provided as input. The result is a real number representing the ratio of votes between the two episodes.")
async def get_vote_ratio_between_episodes(title1: str = Query(..., description="Title of the first episode"), title2: str = Query(..., description="Title of the second episode"), stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.title = ? THEN T1.votes ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.title = ? THEN T1.votes ELSE 0 END) AS ratio FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ?", (title1, title2, stars))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of awards for a specific person from a specific organization with a specific result
@app.get("/v1/simpson_episodes/count_awards_person_organization_result", operation_id="get_count_awards_person_organization_result", summary="Retrieves the total number of awards received by a specified individual from the Writers Guild of America, USA, for a given award result. The result can be a win, a nomination, or any other outcome.")
async def get_count_awards_person_organization_result(person: str = Query(..., description="Person's name"), organization: str = Query(..., description="Organization name"), result: str = Query(..., description="Result (e.g., Nominee)")):
    cursor.execute("SELECT COUNT(award_id) FROM Award WHERE person = ? AND organization = ? AND result = ?", (person, organization, result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the person with the most awards of a specific result
@app.get("/v1/simpson_episodes/top_person_by_award_result", operation_id="get_top_person_by_award_result", summary="Retrieves the individual who has received the highest number of awards for a specified result. The result parameter is used to filter the awards and determine the top awardee.")
async def get_top_person_by_award_result(result: str = Query(..., description="Result (e.g., Nominee)")):
    cursor.execute("SELECT person FROM Award WHERE result = ? GROUP BY person ORDER BY COUNT(person) DESC LIMIT 1", (result,))
    result = cursor.fetchone()
    if not result:
        return {"person": []}
    return {"person": result[0]}

# Endpoint to get the episode with the highest rating
@app.get("/v1/simpson_episodes/top_rated_episode", operation_id="get_top_rated_episode", summary="Retrieves the title of the top-rated episode from the database. The episode with the highest rating is determined and returned.")
async def get_top_rated_episode():
    cursor.execute("SELECT title FROM Episode ORDER BY rating LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the most recent year of a specific award with a specific result
@app.get("/v1/simpson_episodes/most_recent_award_year", operation_id="get_most_recent_award_year", summary="Retrieves the most recent year in which a specific award was achieved with a particular result. The operation requires the result and the name of the award as input parameters.")
async def get_most_recent_award_year(result: str = Query(..., description="Result (e.g., Winner)"), award: str = Query(..., description="Award name")):
    cursor.execute("SELECT year FROM Award WHERE result = ? AND award = ? ORDER BY year DESC LIMIT 1", (result, award))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the count of credits for a specific person with a specific credited status
@app.get("/v1/simpson_episodes/count_credits_person_credited", operation_id="get_count_credits_person_credited", summary="Retrieves the total number of credits associated with a given person, filtered by a specified credited status. The person is identified by their name, and the credited status indicates whether the person was officially credited for their role in the production.")
async def get_count_credits_person_credited(person: str = Query(..., description="Person's name"), credited: str = Query(..., description="Credited status (e.g., 'false')")):
    cursor.execute("SELECT COUNT(*) FROM Credit WHERE person = ? AND credited = ?", (person, credited))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the age of award winners for a specific award, organization, and result
@app.get("/v1/simpson_episodes/age_of_award_winners", operation_id="get_age_of_award_winners", summary="Retrieve the age of individuals who have won a specific award from a given organization. The age is calculated based on the year of the award and the birth year of the award winner. The result parameter allows for filtering based on the outcome of the award, such as 'Winner'.")
async def get_age_of_award_winners(award: str = Query(..., description="Award name"), organization: str = Query(..., description="Organization name"), result: str = Query(..., description="Result (e.g., Winner)")):
    cursor.execute("SELECT T2.year - CAST(SUBSTR(T1.birthdate, 1, 4) AS int) AS age FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.award = ? AND T2.organization = ? AND T2.result = ?", (award, organization, result))
    result = cursor.fetchall()
    if not result:
        return {"ages": []}
    return {"ages": [row[0] for row in result]}

# Endpoint to get distinct characters for a specific person, award, organization, and year
@app.get("/v1/simpson_episodes/distinct_characters_award", operation_id="get_distinct_characters_award", summary="Retrieves a list of distinct characters associated with a specific individual, award, organization, and year. The operation filters the data based on the provided person's name, award name, organization name, and year, and returns the unique characters who have received the specified award from the given organization in the particular year.")
async def get_distinct_characters_award(person: str = Query(..., description="Person's name"), award: str = Query(..., description="Award name"), organization: str = Query(..., description="Organization name"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT DISTINCT T2.character FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T1.person = ? AND T1.award = ? AND T1.organization = ? AND T1.year = ?", (person, award, organization, year))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get the count of episodes with awards in a specific year and air date pattern
@app.get("/v1/simpson_episodes/count_episodes_with_awards", operation_id="get_count_episodes_with_awards", summary="Retrieves the total number of episodes that have received awards in a specified year and air date pattern. The year is represented by the first four digits, and the air date pattern is a string that follows the format 'YYYY-MM%'.")
async def get_count_episodes_with_awards(year: str = Query(..., description="Year (e.g., '2009')"), air_date_pattern: str = Query(..., description="Air date pattern (e.g., '2009-04%')")):
    cursor.execute("SELECT COUNT(T1.episode_id) FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE SUBSTR(T1.year, 1, 4) = ? AND T2.air_date LIKE ?", (year, air_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the birth place of a person who won a specific award from a specific organization in a specific year
@app.get("/v1/simpson_episodes/birth_place_by_award_organization_year", operation_id="get_birth_place_by_award_organization_year", summary="Retrieves the birth place of the award recipient from a specific organization in a given year. The operation requires the name of the award, the organization, and the year as input parameters.")
async def get_birth_place_by_award_organization_year(award: str = Query(..., description="Award name"), organization: str = Query(..., description="Organization name"), year: int = Query(..., description="Year of the award")):
    cursor.execute("SELECT T1.birth_place FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.award = ? AND T2.organization = ? AND T2.year = ?", (award, organization, year))
    result = cursor.fetchall()
    if not result:
        return {"birth_places": []}
    return {"birth_places": [row[0] for row in result]}

# Endpoint to get the sum of votes for episodes with a specific star rating, ordered by rating and limited to a specific number of results
@app.get("/v1/simpson_episodes/sum_votes_by_stars_limit", operation_id="get_sum_votes_by_stars_limit", summary="Retrieves the total number of votes for episodes with a specified star rating, sorted by their rating in descending order. The results are limited to a specified number. This operation provides a summary of the most popular episodes based on the given star rating and the number of votes they have received.")
async def get_sum_votes_by_stars_limit(stars: int = Query(..., description="Star rating"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT SUM(T1.votes) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ? ORDER BY T1.rating DESC LIMIT ?", (stars, limit))
    result = cursor.fetchone()
    if not result:
        return {"sum_votes": []}
    return {"sum_votes": result[0]}

# Endpoint to get the count of episodes with a specific title and votes less than a specific number
@app.get("/v1/simpson_episodes/count_episodes_by_title_votes", operation_id="get_count_episodes_by_title_votes", summary="Retrieve the number of episodes with a specified title and fewer votes than a given threshold. This operation allows you to filter episodes based on their title and the number of votes they have received, providing a count of the episodes that meet the criteria.")
async def get_count_episodes_by_title_votes(title: str = Query(..., description="Episode title"), votes: int = Query(..., description="Maximum number of votes")):
    cursor.execute("SELECT COUNT(*) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ? AND T2.votes < ?", (title, votes))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons from a specific birth country who won an award
@app.get("/v1/simpson_episodes/count_persons_by_birth_country_award_result", operation_id="get_count_persons_by_birth_country_award_result", summary="Retrieves the total number of individuals from a specified country of birth who have achieved a particular award outcome. The operation requires the country of birth and the award result as input parameters.")
async def get_count_persons_by_birth_country_award_result(birth_country: str = Query(..., description="Birth country"), result: str = Query(..., description="Award result")):
    cursor.execute("SELECT COUNT(*) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T1.birth_country = ? AND T2.result = ?", (birth_country, result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons with a specific nickname who were credited in a specific episode
@app.get("/v1/simpson_episodes/count_persons_by_nickname_credited_episode", operation_id="get_count_persons_by_nickname_credited_episode", summary="Retrieve the number of individuals with a given nickname who were credited in a specific episode. The operation requires the nickname, credited status, and episode ID as input parameters.")
async def get_count_persons_by_nickname_credited_episode(nickname: str = Query(..., description="Nickname"), credited: str = Query(..., description="Credited status"), episode_id: str = Query(..., description="Episode ID")):
    cursor.execute("SELECT COUNT(*) FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T1.nickname = ? AND T2.credited = ? AND T2.episode_id = ?", (nickname, credited, episode_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the award and name of a person who won an award, ordered by year and limited to a specific number of results
@app.get("/v1/simpson_episodes/award_name_by_result_limit", operation_id="get_award_name_by_result_limit", summary="Retrieves the award and name of a person who won an award, ordered by year and limited to a specific number of results. The award result and the maximum number of results to return can be specified as input parameters.")
async def get_award_name_by_result_limit(result: str = Query(..., description="Award result"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT T2.award, T1.name FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.result = ? ORDER BY T2.year LIMIT ?", (result, limit))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": [{"award": row[0], "name": row[1]} for row in result]}

# Endpoint to get the vote percent for an episode with a specific title and star rating
@app.get("/v1/simpson_episodes/vote_percent_by_title_stars", operation_id="get_vote_percent_by_title_stars", summary="Retrieves the percentage of votes for a specific Simpsons episode based on its title and star rating. The operation returns the vote percentage for the episode that matches the provided title and star rating.")
async def get_vote_percent_by_title_stars(title: str = Query(..., description="Episode title"), stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T2.percent FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ? AND T2.stars = ?", (title, stars))
    result = cursor.fetchall()
    if not result:
        return {"vote_percents": []}
    return {"vote_percents": [row[0] for row in result]}

# Endpoint to get the award and person who won an award in a specific year
@app.get("/v1/simpson_episodes/award_person_by_result_year", operation_id="get_award_person_by_result_year", summary="Retrieves the award name and the person who won it in a specified year. The operation requires the award result and the year (in 'YYYY' format) as input parameters. The result parameter determines the specific award, while the year parameter filters the results to a particular year.")
async def get_award_person_by_result_year(result: str = Query(..., description="Award result"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT award, person FROM Award WHERE result = ? AND SUBSTR(year, 1, 4) = ?", (result, year))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": [{"award": row[0], "person": row[1]} for row in result]}

# Endpoint to get the name and birthdate of persons from a specific birth place and minimum height
@app.get("/v1/simpson_episodes/person_by_birth_place_height", operation_id="get_person_by_birth_place_height", summary="Retrieves the names and birthdates of individuals born in a specified location who meet or exceed a given minimum height. The birth place and minimum height are provided as input parameters.")
async def get_person_by_birth_place_height(birth_place: str = Query(..., description="Birth place"), height_meters: float = Query(..., description="Minimum height in meters")):
    cursor.execute("SELECT name, birthdate FROM Person WHERE birth_place = ? AND height_meters >= ?", (birth_place, height_meters))
    result = cursor.fetchall()
    if not result:
        return {"persons": []}
    return {"persons": [{"name": row[0], "birthdate": row[1]} for row in result]}

# Endpoint to get the episode, title, and rating of episodes aired in a specific month and year
@app.get("/v1/simpson_episodes/episode_by_air_date", operation_id="get_episode_by_air_date", summary="Retrieves the episode number, title, and rating of episodes aired in a specific month and year. The air date should be provided in 'YYYY-MM' format.")
async def get_episode_by_air_date(air_date: str = Query(..., description="Air date in 'YYYY-MM' format")):
    cursor.execute("SELECT episode, title, rating FROM Episode WHERE SUBSTR(air_date, 1, 7) LIKE ?", (air_date + '%',))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": [{"episode": row[0], "title": row[1], "rating": row[2]} for row in result]}

# Endpoint to get award details for a specific character and result
@app.get("/v1/simpson_episodes/award_details_character_result", operation_id="get_award_details_character_result", summary="Retrieves detailed information about awards won by a specific character, filtered by a given award result. The response includes the award ID, award name, and the person who received the award.")
async def get_award_details_character_result(character: str = Query(..., description="Character name"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T1.award_id, T1.award, T1.person FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T2.character = ? AND T1.result = ?", (character, result))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": result}

# Endpoint to get distinct award details for a specific person
@app.get("/v1/simpson_episodes/distinct_award_details_person", operation_id="get_distinct_award_details_person", summary="Retrieve unique award details, including the award name, result, category, and credited status, for a specific individual. The individual's name is provided as an input parameter.")
async def get_distinct_award_details_person(person: str = Query(..., description="Person name")):
    cursor.execute("SELECT DISTINCT T1.award, T1.result, T2.category, T2.credited FROM Award AS T1 INNER JOIN Credit AS T2 ON T2.episode_id = T1.episode_id WHERE T2.person = ?", (person,))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": result}

# Endpoint to get award details for a specific episode and role
@app.get("/v1/simpson_episodes/award_details_episode_role", operation_id="get_award_details_episode_role", summary="Retrieves detailed award information for a specific role in a given episode. The response includes the person's name, award name, awarding organization, award result, and role credit status.")
async def get_award_details_episode_role(episode_id: str = Query(..., description="Episode ID"), role: str = Query(..., description="Role in the episode")):
    cursor.execute("SELECT T1.person, T1.award, T1.organization, T1.result, T2.credited FROM Award AS T1 INNER JOIN Credit AS T2 ON T2.episode_id = T1.episode_id WHERE T2.episode_id = ? AND T2.role = ?", (episode_id, role))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": result}

# Endpoint to get person details based on category and credited status
@app.get("/v1/simpson_episodes/person_details_category_credited", operation_id="get_person_details_category_credited", summary="Retrieves the birth country, height in meters, and name of a person who has been credited in a specific category. The category and credited status are provided as input parameters.")
async def get_person_details_category_credited(category: str = Query(..., description="Category of the credit"), credited: str = Query(..., description="Credited status (true or false)")):
    cursor.execute("SELECT T1.birth_country, T1.height_meters, T1.name FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T2.category = ? AND T2.credited = ?", (category, credited))
    result = cursor.fetchall()
    if not result:
        return {"persons": []}
    return {"persons": result}

# Endpoint to get person and keyword details for a specific episode title and award result
@app.get("/v1/simpson_episodes/person_keyword_details_episode_title_award_result", operation_id="get_person_keyword_details_episode_title_award_result", summary="Retrieves details about the persons and keywords associated with a specific episode, given its title and the result of an award. The response includes the person's name and the related keyword for the episode.")
async def get_person_keyword_details_episode_title_award_result(title: str = Query(..., description="Episode title"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T3.person, T1.keyword, T1.episode_id FROM Keyword AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Award AS T3 ON T2.episode_id = T3.episode_id WHERE T2.title = ? AND T3.result = ?", (title, result))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get vote details for a specific keyword and star rating
@app.get("/v1/simpson_episodes/vote_details_keyword_stars", operation_id="get_vote_details_keyword_stars", summary="Retrieves the vote details, including the total number of votes and the percentage of votes, for a specific keyword and star rating. The keyword and star rating are used to filter the results.")
async def get_vote_details_keyword_stars(keyword: str = Query(..., description="Keyword"), stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T2.votes, T2.percent FROM Keyword AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T1.keyword = ? AND T2.stars = ?", (keyword, stars))
    result = cursor.fetchall()
    if not result:
        return {"votes": []}
    return {"votes": result}

# Endpoint to get award details for a specific organization and result
@app.get("/v1/simpson_episodes/award_details_organization_result", operation_id="get_award_details_organization_result", summary="Retrieves detailed information about awards received by a specific organization for a given result. The data includes the award name, the air date of the episode in which the award was received, and the rating of that episode. The organization and result are provided as input parameters.")
async def get_award_details_organization_result(organization: str = Query(..., description="Organization name"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T1.award, T2.air_date, T2.rating FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.organization = ? AND T1.result = ?", (organization, result))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": result}

# Endpoint to get distinct episode details with a specific star rating, ordered by votes
@app.get("/v1/simpson_episodes/distinct_episode_details_stars", operation_id="get_distinct_episode_details_stars", summary="Retrieves unique episode details, including title and keyword, for episodes with a specified star rating. The results are ordered by the number of votes in descending order and limited to the top three. The star rating is a required input parameter.")
async def get_distinct_episode_details_stars(stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT DISTINCT T3.episode_id, T2.title, T1.keyword FROM Keyword AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Vote AS T3 ON T2.episode_id = T3.episode_id WHERE T3.stars = ? ORDER BY T3.votes DESC LIMIT 3", (stars,))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": result}

# Endpoint to get episode and award details for a specific rating range, air date year, and result
@app.get("/v1/simpson_episodes/episode_award_details_rating_air_date_result", operation_id="get_episode_award_details_rating_air_date_result", summary="Retrieves episode details, including title and episode image, along with associated award information for episodes that fall within a specified rating range and air date year, and have a particular award result. The rating range is defined by a minimum and maximum rating, while the air date year is provided in 'YYYY' format. The award result indicates the outcome of the award for the episode.")
async def get_episode_award_details_rating_air_date_result(min_rating: float = Query(..., description="Minimum rating"), max_rating: float = Query(..., description="Maximum rating"), air_date_year: str = Query(..., description="Air date year in 'YYYY' format"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T2.title, T2.episode_image, T1.award, T1.person FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T2.rating BETWEEN ? AND ? AND SUBSTR(T2.air_date, 1, 4) = ? AND T1.result = ?", (min_rating, max_rating, air_date_year, result))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get distinct award details for a specific award ID
@app.get("/v1/simpson_episodes/distinct_award_details_award_id", operation_id="get_distinct_award_details_award_id", summary="Retrieves unique award details, including the award name, recipient, and associated character, for a given award ID.")
async def get_distinct_award_details_award_id(award_id: int = Query(..., description="Award ID")):
    cursor.execute("SELECT DISTINCT T1.award, T1.person, T2.character FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T2.award_id = ?", (award_id,))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": result}

# Endpoint to get the oldest person's details
@app.get("/v1/simpson_episodes/oldest_person_details", operation_id="get_oldest_person_details", summary="Retrieve the name, birth place, role, and calculated age of the oldest person(s) with a known birthdate, up to the specified limit. The results are ordered by birthdate in ascending order.")
async def get_oldest_person_details(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T1.name, T1.birth_place, T2.role, 2022 - CAST(SUBSTR(T1.birthdate, 1, 4) AS int) AS age FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T1.birthdate IS NOT NULL ORDER BY T1.birthdate LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get distinct credits for persons from a specific birth country
@app.get("/v1/simpson_episodes/credits_by_birth_country", operation_id="get_credits_by_birth_country", summary="Retrieve unique credits for individuals born in a specified country. The operation returns a list of distinct credits, including the credited person, their role, and category, along with their birthplace. The birth country is used as a filter to narrow down the results.")
async def get_credits_by_birth_country(birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT DISTINCT T2.credited, T2.category, T2.role, T1.birth_place FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T1.birth_country = ?", (birth_country,))
    result = cursor.fetchall()
    if not result:
        return {"credits": []}
    return {"credits": result}

# Endpoint to get award rates and details for winners in a specific year
@app.get("/v1/simpson_episodes/award_rates_and_details", operation_id="get_award_rates_and_details", summary="Get award rates and details for winners in a specific year")
async def get_award_rates_and_details(year: int = Query(..., description="Year of the award (YYYY)")):
    cursor.execute("SELECT T3.rate, T4.person, T4.award, T5.title, T4.role FROM ( SELECT CAST(SUM(CASE WHEN T1.result = 'Winner' THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.result IN ('Winner', 'Nominee') THEN 1 ELSE 0 END) AS rate , T1.person, T1.award, T2.title, T1.role FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE SUBSTR(T1.year, 1, 4) = ? ) AS T3 INNER JOIN Award AS T4 INNER JOIN Episode AS T5 ON T4.episode_id = T5.episode_id WHERE T4.year = ? AND T4.result = 'Winner'", (year, year))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the highest rated episode
@app.get("/v1/simpson_episodes/highest_rated_episode", operation_id="get_highest_rated_episode", summary="Retrieves the highest rated episode(s) based on a combination of star ratings and vote counts. The number of episodes returned can be limited by specifying the desired limit.")
async def get_highest_rated_episode(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id ORDER BY T2.stars DESC, T2.votes DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": result}

# Endpoint to get the count of awards based on result
@app.get("/v1/simpson_episodes/count_awards_by_result", operation_id="get_count_awards_by_result", summary="Retrieves the total number of awards that match the specified result status (Winner or Nominee).")
async def get_count_awards_by_result(result: str = Query(..., description="Result of the award (Winner or Nominee)")):
    cursor.execute("SELECT COUNT(award_id) FROM Award WHERE result = ?", (result,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episode titles based on award criteria
@app.get("/v1/simpson_episodes/episode_titles_by_award_criteria", operation_id="get_episode_titles_by_award_criteria", summary="Retrieves the titles of episodes that have been awarded a specific number of times by a particular organization. The award name and result (winner or nominee) are also considered in the search criteria.")
async def get_episode_titles_by_award_criteria(organization: str = Query(..., description="Organization of the award"), award: str = Query(..., description="Name of the award"), result: str = Query(..., description="Result of the award (Winner or Nominee)"), count: int = Query(..., description="Number of times the episode was nominated or won")):
    cursor.execute("SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.organization = ? AND T1.award = ? AND T1.result = ? GROUP BY T1.episode_id HAVING COUNT(T1.episode_id) = ?", (organization, award, result, count))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get the average stars and summary for a specific episode
@app.get("/v1/simpson_episodes/average_stars_and_summary", operation_id="get_average_stars_and_summary", summary="Retrieves the average star rating and summary for a specific episode. The episode is identified by its unique ID, which is provided as an input parameter. The operation calculates the average star rating from all votes cast for the episode and returns this value along with the episode's summary.")
async def get_average_stars_and_summary(episode_id: str = Query(..., description="Episode ID")):
    cursor.execute("SELECT AVG(T2.stars), T1.summary FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.episode_id = ?", (episode_id,))
    result = cursor.fetchone()
    if not result:
        return {"average_stars": [], "summary": []}
    return {"average_stars": result[0], "summary": result[1]}

# Endpoint to get persons who won a specific award for a specific character
@app.get("/v1/simpson_episodes/persons_by_character_award", operation_id="get_persons_by_character_award", summary="Retrieves a list of individuals who have been recognized with a specific award for a given character by a particular organization. The result of the award (whether the individual was a winner or nominee) is also considered.")
async def get_persons_by_character_award(character: str = Query(..., description="Character name"), organization: str = Query(..., description="Organization of the award"), award: str = Query(..., description="Name of the award"), result: str = Query(..., description="Result of the award (Winner or Nominee)")):
    cursor.execute("SELECT T1.person FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T2.character = ? AND T1.organization = ? AND T1.award = ? AND T1.result = ?", (character, organization, award, result))
    result = cursor.fetchall()
    if not result:
        return {"persons": []}
    return {"persons": result}

# Endpoint to get distinct birth names and roles for a specific person
@app.get("/v1/simpson_episodes/birth_names_and_roles_by_person", operation_id="get_birth_names_and_roles_by_person", summary="Retrieve a unique list of birth names and associated roles for a specified person. The operation filters the data based on the provided person's name.")
async def get_birth_names_and_roles_by_person(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT DISTINCT T1.birth_name, T2.role FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T1.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the count of awards for a specific person in a specific year
@app.get("/v1/simpson_episodes/count_awards_by_person_year", operation_id="get_count_awards_by_person_year", summary="Retrieves the total number of awards won or received by a nominated person in a specific year. The result can be filtered to show either the count of awards won or the count of nominations received.")
async def get_count_awards_by_person_year(person: str = Query(..., description="Name of the person"), year: str = Query(..., description="Year of the award (YYYY)"), result: str = Query(..., description="Result of the award (Winner or Nominee)")):
    cursor.execute("SELECT COUNT(award_id) FROM Award WHERE person = ? AND SUBSTR(year, 1, 4) = ? AND result = ?", (person, year, result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latest award and category for a specific result
@app.get("/v1/simpson_episodes/latest_award_category", operation_id="get_latest_award_category", summary="Retrieves the most recent award and its corresponding category for a given result. The result parameter specifies the outcome of the award, such as 'Winner'.")
async def get_latest_award_category(result: str = Query(..., description="Result of the award (e.g., 'Winner')")):
    cursor.execute("SELECT award, award_category FROM Award WHERE result = ? ORDER BY year DESC LIMIT 1;", (result,))
    result = cursor.fetchone()
    if not result:
        return {"award": [], "award_category": []}
    return {"award": result[0], "award_category": result[1]}

# Endpoint to get the win rate of awards
@app.get("/v1/simpson_episodes/award_win_rate", operation_id="get_award_win_rate", summary="Retrieves the percentage of awards won by a specific category. The category is determined by the provided result parameter, which represents the outcome of the award (e.g., 'Winner'). The calculation is based on the total number of awards and the count of awards with the specified result.")
async def get_award_win_rate(result: str = Query(..., description="Result of the award (e.g., 'Winner')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN result = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(award) AS rate FROM Award;", (result,))
    result = cursor.fetchone()
    if not result:
        return {"rate": []}
    return {"rate": result[0]}

# Endpoint to get the count of episodes aired in specific months of a year
@app.get("/v1/simpson_episodes/episode_count_by_month", operation_id="get_episode_count_by_month", summary="Retrieves the total number of episodes aired during two specified months of a given year. The operation filters episodes by their air date and returns the count of episodes that match the criteria.")
async def get_episode_count_by_month(year: str = Query(..., description="Year in 'YYYY' format"), month1: str = Query(..., description="First month in 'MM' format"), month2: str = Query(..., description="Second month in 'MM' format")):
    cursor.execute("SELECT COUNT(episode_id) FROM Episode WHERE air_date LIKE ? OR air_date LIKE ?;", (f'{year}-{month1}%', f'{year}-{month2}%'))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest-rated episode aired in a specific year
@app.get("/v1/simpson_episodes/highest_rated_episode_by_year", operation_id="get_highest_rated_episode_by_year", summary="Retrieves the episode with the highest rating that aired in a specific year. The year should be provided in 'YYYY' format.")
async def get_highest_rated_episode_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT episode_id FROM Episode WHERE air_date LIKE ? ORDER BY rating LIMIT 1;", (f'{year}%',))
    result = cursor.fetchone()
    if not result:
        return {"episode_id": []}
    return {"episode_id": result[0]}

# Endpoint to get distinct categories and roles for a specific person
@app.get("/v1/simpson_episodes/person_categories_roles", operation_id="get_person_categories_roles", summary="Retrieves the unique categories and roles associated with a specific person. The provided person's name is used to filter the results.")
async def get_person_categories_roles(person: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT DISTINCT category, role FROM Credit WHERE person = ?;", (person,))
    result = cursor.fetchall()
    if not result:
        return {"categories_roles": []}
    return {"categories_roles": result}

# Endpoint to get the name of a person based on birthdate, birth place, and birth region
@app.get("/v1/simpson_episodes/person_by_birth_details", operation_id="get_person_by_birth_details", summary="Retrieves the name of a person whose birthdate, birth place, and birth region match the provided details. The birthdate should be in 'YYYY-MM-DD' format.")
async def get_person_by_birth_details(birthdate: str = Query(..., description="Birthdate in 'YYYY-MM-DD' format"), birth_place: str = Query(..., description="Birth place"), birth_region: str = Query(..., description="Birth region")):
    cursor.execute("SELECT name FROM Person WHERE birthdate = ? AND birth_place = ? AND birth_region = ?;", (birthdate, birth_place, birth_region))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get distinct persons based on a specific role
@app.get("/v1/simpson_episodes/persons_by_role", operation_id="get_persons_by_role", summary="Retrieves a unique list of individuals who have been credited with a specific role in the Simpsons episodes. The role is specified as an input parameter.")
async def get_persons_by_role(role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT DISTINCT person FROM Credit WHERE role = ?;", (role,))
    result = cursor.fetchall()
    if not result:
        return {"persons": []}
    return {"persons": [item[0] for item in result]}

# Endpoint to get the age in 2009 and name of award nominees based on specific criteria
@app.get("/v1/simpson_episodes/award_nominees_age_name", operation_id="get_award_nominees_age_name", summary="Retrieves the age in 2009 and name of individuals who were nominated for a specific award, based on their role, the awarding organization, the award name, the nomination result, and the year of the award.")
async def get_award_nominees_age_name(role: str = Query(..., description="Role of the nominee"), organization: str = Query(..., description="Organization of the award"), award: str = Query(..., description="Award name"), result: str = Query(..., description="Result of the award (e.g., 'Nominee')"), year: int = Query(..., description="Year of the award")):
    cursor.execute("SELECT T1.year - T2.birthdate AS ageIn2009, T2.name FROM Award AS T1 INNER JOIN Person AS T2 ON T1.person = T2.name WHERE T1.role = ? AND T1.organization = ? AND T1.award = ? AND T1.result = ? AND T1.year = ?;", (role, organization, award, result, year))
    result = cursor.fetchall()
    if not result:
        return {"age_name": []}
    return {"age_name": result}

# Endpoint to get the title of the episode with the most awards
@app.get("/v1/simpson_episodes/most_awarded_episode", operation_id="get_most_awarded_episode", summary="Retrieves the title of the episode that has received the highest number of awards. This operation identifies the episode with the most awards by joining the Award and Episode tables, grouping by episode_id, and ordering the results in descending order based on the count of awards. The title of the top-ranked episode is then returned.")
async def get_most_awarded_episode():
    cursor.execute("SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id GROUP BY T1.episode_id ORDER BY COUNT(*) DESC LIMIT 1;")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the birth place of a specific person
@app.get("/v1/simpson_episodes/person_birth_place", operation_id="get_person_birth_place", summary="Retrieves the birth place of a specified person. The operation requires the person's name as input and returns the corresponding birth place from the database.")
async def get_person_birth_place(name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT birth_place FROM Person WHERE name = ?;", (name,))
    result = cursor.fetchone()
    if not result:
        return {"birth_place": []}
    return {"birth_place": result[0]}

# Endpoint to get the count of persons based on birth country
@app.get("/v1/simpson_episodes/count_persons_by_birth_country", operation_id="get_count_persons_by_birth_country", summary="Retrieves the total number of individuals born in a specified country. The operation filters the Person table by the provided birth country and returns the count of matching records.")
async def get_count_persons_by_birth_country(birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT COUNT(name) FROM Person WHERE birth_country = ?", (birth_country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names of persons born before a specific year
@app.get("/v1/simpson_episodes/persons_born_before_year", operation_id="get_persons_born_before_year", summary="Retrieve the names of individuals who were born before the specified year. The year should be provided in the 'YYYY' format.")
async def get_persons_born_before_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT name FROM Person WHERE SUBSTR(birthdate, 1, 4) < ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get names of persons based on birth year, birth place, and birth country
@app.get("/v1/simpson_episodes/persons_by_birth_year_place_country", operation_id="get_persons_by_birth_year_place_country", summary="Retrieve the names of individuals from the database, filtered by their birth year, birth place, and birth country. The birth year should be provided in 'YYYY' format. This operation allows for a targeted search of individuals based on their birth details.")
async def get_persons_by_birth_year_place_country(birth_year: str = Query(..., description="Birth year in 'YYYY' format"), birth_place: str = Query(..., description="Birth place of the person"), birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT name FROM Person WHERE SUBSTR(birthdate, 1, 4) = ? AND birth_place = ? AND birth_country = ?", (birth_year, birth_place, birth_country))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get episode IDs based on stars and votes
@app.get("/v1/simpson_episodes/episode_ids_by_stars_votes", operation_id="get_episode_ids_by_stars_votes", summary="Retrieves the IDs of episodes that have been rated with a specific number of stars and have received more than a certain number of votes. The operation filters episodes based on the provided star rating and minimum vote count, returning a list of episode IDs that meet the criteria.")
async def get_episode_ids_by_stars_votes(stars: int = Query(..., description="Number of stars"), votes: int = Query(..., description="Number of votes")):
    cursor.execute("SELECT episode_id FROM Vote WHERE stars = ? AND votes > ?", (stars, votes))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

# Endpoint to get distinct episodes based on episode range and votes
@app.get("/v1/simpson_episodes/distinct_episodes_by_range_votes", operation_id="get_distinct_episodes_by_range_votes", summary="Retrieve a list of unique episodes that fall within a specified range of episode numbers and have received more than a certain number of votes. The range is defined by the start and end episode numbers, and the minimum number of votes is also provided as input.")
async def get_distinct_episodes_by_range_votes(start_episode: int = Query(..., description="Start episode number"), end_episode: int = Query(..., description="End episode number"), votes: int = Query(..., description="Number of votes")):
    cursor.execute("SELECT DISTINCT T1.episode FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.episode BETWEEN ? AND ? AND T2.votes > ?", (start_episode, end_episode, votes))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": [row[0] for row in result]}

# Endpoint to get keywords for episodes based on award category
@app.get("/v1/simpson_episodes/keywords_by_award_category", operation_id="get_keywords_by_award_category", summary="Retrieves a list of keywords associated with episodes that have received awards in a specific category. The award category is provided as an input parameter.")
async def get_keywords_by_award_category(award_category: str = Query(..., description="Award category")):
    cursor.execute("SELECT T2.keyword FROM Award AS T1 INNER JOIN Keyword AS T2 ON T2.episode_id = T1.episode_id WHERE T1.award_category = ?", (award_category,))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get the top person based on award category and votes
@app.get("/v1/simpson_episodes/top_person_by_award_category", operation_id="get_top_person_by_award_category", summary="Retrieves the individual who has received the most votes in a specified award category. The award category is provided as an input parameter.")
async def get_top_person_by_award_category(award_category: str = Query(..., description="Award category")):
    cursor.execute("SELECT T1.person FROM Award AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.award_category = ? ORDER BY T2.votes DESC LIMIT 1", (award_category,))
    result = cursor.fetchone()
    if not result:
        return {"person": []}
    return {"person": result[0]}

# Endpoint to get distinct episode IDs based on award and stars
@app.get("/v1/simpson_episodes/distinct_episode_ids_by_award_stars", operation_id="get_distinct_episode_ids_by_award_stars", summary="Retrieves a unique set of episode IDs that have been awarded a specific award and received a certain number of stars. The award and star count are provided as input parameters.")
async def get_distinct_episode_ids_by_award_stars(award: str = Query(..., description="Award name"), stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT DISTINCT T1.episode_id FROM Award AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.award = ? AND T2.stars = ?", (award, stars))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

# Endpoint to get persons based on role, award, and episode title
@app.get("/v1/simpson_episodes/persons_by_role_award_title", operation_id="get_persons_by_role_award_title", summary="Retrieves a list of persons who have a specific role and award in the episode with the given title. The role and award are provided as input parameters, while the episode title is predefined.")
async def get_persons_by_role_award_title(role: str = Query(..., description="Role of the person"), award: str = Query(..., description="Award name"), title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT T1.person FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.role = ? AND T1.award = ? AND T2.title = ?", (role, award, title))
    result = cursor.fetchall()
    if not result:
        return {"persons": []}
    return {"persons": [row[0] for row in result]}

# Endpoint to get names of people born in a specific year and region
@app.get("/v1/simpson_episodes/person_names_by_birthyear_region", operation_id="get_person_names", summary="Retrieves the names of individuals born in a specific year and region. The year of birth should be provided in 'YYYY' format, and the region of birth must be specified. This operation returns a list of names that match the given birth year and region.")
async def get_person_names(birth_year: str = Query(..., description="Year of birth in 'YYYY' format"), birth_region: str = Query(..., description="Region of birth")):
    cursor.execute("SELECT name FROM Person WHERE SUBSTR(birthdate, 1, 4) = ? AND birth_region = ?", (birth_year, birth_region))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of people based on height and birth country
@app.get("/v1/simpson_episodes/person_count_by_height_country", operation_id="get_person_count", summary="Retrieves the number of individuals who are taller than the specified height and hail from the given country. The height is provided in meters, and the country is specified by its name.")
async def get_person_count(height_meters: float = Query(..., description="Height in meters"), birth_country: str = Query(..., description="Country of birth")):
    cursor.execute("SELECT COUNT(name) FROM Person WHERE height_meters > ? AND birth_country = ?", (height_meters, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of awards based on award type and result
@app.get("/v1/simpson_episodes/award_count_by_type_result", operation_id="get_award_count", summary="Retrieves the total count of a specific award type with a given result. This operation allows you to determine the number of times an award has been received or nominated, providing insights into the frequency and distribution of awards.")
async def get_award_count(award: str = Query(..., description="Type of award"), result: str = Query(..., description="Result of the award (e.g., Nominee, Winner)")):
    cursor.execute("SELECT COUNT(*) FROM Award WHERE award = ? AND result = ?", (award, result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episode titles based on rating range and limit
@app.get("/v1/simpson_episodes/episode_titles_by_rating_range", operation_id="get_episode_titles", summary="Retrieve a specified number of episode titles with ratings that fall within a given range. The range is defined by a minimum and maximum rating, and the number of results returned is limited by the provided limit parameter.")
async def get_episode_titles(min_rating: int = Query(..., description="Minimum rating"), max_rating: int = Query(..., description="Maximum rating"), limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT title FROM Episode WHERE rating BETWEEN ? AND ? LIMIT ?", (min_rating, max_rating, limit))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get episode IDs ordered by votes in descending order
@app.get("/v1/simpson_episodes/top_episodes_by_votes", operation_id="get_top_episodes", summary="Retrieves a specified number of top-rated episodes from The Simpsons series, sorted by the highest number of votes in descending order.")
async def get_top_episodes(limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT episode_id FROM Episode ORDER BY votes DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

# Endpoint to get episode titles with the minimum stars and ordered by votes
@app.get("/v1/simpson_episodes/episode_titles_min_stars_by_votes", operation_id="get_episode_titles_min_stars", summary="Retrieves the titles of episodes that have the lowest star rating, sorted by the number of votes in descending order. The number of results returned is limited to the specified value.")
async def get_episode_titles_min_stars(limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ( SELECT MIN(stars) FROM Vote ) ORDER BY T2.votes DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get distinct characters based on award type and person
@app.get("/v1/simpson_episodes/distinct_characters_by_award_person", operation_id="get_distinct_characters", summary="Get distinct characters based on award type and person")
async def get_distinct_characters(award_type: str = Query(..., description="Type of award (use % for wildcard)"), person: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT DISTINCT T2.character FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T1.award LIKE ? AND T1.person = ?", (award_type, person))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get episode IDs based on air date year and ordered by votes
@app.get("/v1/simpson_episodes/episode_ids_by_air_date_year", operation_id="get_episode_ids_by_air_date", summary="Retrieves a specified number of episode IDs that aired in a given year, sorted by the number of votes in descending order. The year is specified in 'YYYY' format.")
async def get_episode_ids_by_air_date(air_date_year: str = Query(..., description="Year of air date in 'YYYY' format"), limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT T1.episode_id FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE SUBSTR(T1.air_date, 1, 4) = ? ORDER BY T2.votes DESC LIMIT ?", (air_date_year, limit))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

# Endpoint to get episode titles based on keyword and limit
@app.get("/v1/simpson_episodes/episode_titles_by_keyword", operation_id="get_episode_titles_by_keyword", summary="Retrieve a specified number of episode titles that contain a given keyword. The keyword is used to filter the episodes, and the limit parameter determines the maximum number of results returned.")
async def get_episode_titles_by_keyword(keyword: str = Query(..., description="Keyword to search for"), limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T2.keyword = ? LIMIT ?", (keyword, limit))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get episode titles based on award year and person
@app.get("/v1/simpson_episodes/episode_titles_by_award_year_person", operation_id="get_episode_titles_by_award_year_person", summary="Retrieves the titles of episodes that were awarded in a specific year to a particular person. The operation filters episodes based on the provided four-digit award year and the name of the person associated with the award.")
async def get_episode_titles_by_award_year_person(award_year: str = Query(..., description="Year of the award in 'YYYY' format"), person: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE SUBSTR(T1.year, 1, 4) = ? AND T1.person = ?", (award_year, person))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the sum of votes for a specific episode title and star rating
@app.get("/v1/simpson_episodes/sum_votes_by_title_and_stars", operation_id="get_sum_votes_by_title_and_stars", summary="Retrieves the total number of votes for a specific Simpsons episode, based on its title and star rating. This operation aggregates votes from the database, providing a summative view of user preferences for the given episode and star rating.")
async def get_sum_votes_by_title_and_stars(title: str = Query(..., description="Title of the episode"), stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT SUM(T2.votes) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ? AND T2.stars = ?", (title, stars))
    result = cursor.fetchone()
    if not result:
        return {"sum_votes": []}
    return {"sum_votes": result[0]}

# Endpoint to get votes for a specific keyword and star rating
@app.get("/v1/simpson_episodes/votes_by_keyword_and_stars", operation_id="get_votes_by_keyword_and_stars", summary="Retrieves the total number of votes for a specific keyword and star rating combination. The operation filters episodes based on the provided keyword and star rating, then aggregates the votes for those episodes.")
async def get_votes_by_keyword_and_stars(stars: int = Query(..., description="Star rating"), keyword: str = Query(..., description="Keyword")):
    cursor.execute("SELECT T2.votes FROM Keyword AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T2.stars = ? AND T1.keyword = ?", (stars, keyword))
    result = cursor.fetchall()
    if not result:
        return {"votes": []}
    return {"votes": [row[0] for row in result]}

# Endpoint to get the difference in votes between two star ratings for a specific episode title
@app.get("/v1/simpson_episodes/vote_difference_by_title_and_stars", operation_id="get_vote_difference_by_title_and_stars", summary="Retrieve the difference in votes between two specified star ratings for a given episode title. This operation compares the total votes for the higher and lower star ratings, providing a numerical comparison of user preferences for the episode.")
async def get_vote_difference_by_title_and_stars(title: str = Query(..., description="Title of the episode"), stars_high: int = Query(..., description="Higher star rating"), stars_low: int = Query(..., description="Lower star rating")):
    cursor.execute("SELECT SUM(CASE WHEN T2.stars = ? THEN T2.votes ELSE 0 END) - SUM(CASE WHEN T2.stars = ? THEN T2.votes ELSE 0 END) AS Difference FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ?", (stars_high, stars_low, title))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the keyword of the episode with the highest votes
@app.get("/v1/simpson_episodes/keyword_of_highest_voted_episode", operation_id="get_keyword_of_highest_voted_episode", summary="Retrieves the keyword associated with the episode that has received the most votes. This operation identifies the highest-rated episode based on user votes and returns the corresponding keyword. The keyword is a descriptive term or phrase that characterizes the content of the episode.")
async def get_keyword_of_highest_voted_episode():
    cursor.execute("SELECT T2.keyword FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id ORDER BY T1.votes LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"keyword": []}
    return {"keyword": result[0]}

# Endpoint to get distinct episode IDs based on star rating, votes, and rating range
@app.get("/v1/simpson_episodes/distinct_episode_ids_by_stars_votes_rating", operation_id="get_distinct_episode_ids_by_stars_votes_rating", summary="Retrieve a unique set of episode IDs that meet the specified star rating, minimum votes, and rating range criteria. The operation filters episodes based on the provided star rating, votes, and rating range, ensuring that only episodes with the given star rating, more than the specified minimum votes, and a rating within the defined range are included in the result set.")
async def get_distinct_episode_ids_by_stars_votes_rating(stars: int = Query(..., description="Star rating"), votes: int = Query(..., description="Minimum votes"), min_rating: float = Query(..., description="Minimum rating"), max_rating: float = Query(..., description="Maximum rating")):
    cursor.execute("SELECT DISTINCT T1.episode_id FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ? AND T2.votes > ? AND T1.rating > ? AND T1.rating <= ?", (stars, votes, min_rating, max_rating))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

# Endpoint to get the episode with the maximum votes
@app.get("/v1/simpson_episodes/episode_with_max_votes", operation_id="get_episode_with_max_votes", summary="Retrieves the episode that has received the highest number of votes. This operation identifies the episode with the maximum votes from the Episode table and returns its details.")
async def get_episode_with_max_votes():
    cursor.execute("SELECT episode FROM Episode WHERE votes = ( SELECT MAX(votes) FROM Episode )")
    result = cursor.fetchone()
    if not result:
        return {"episode": []}
    return {"episode": result[0]}

# Endpoint to get the name of the oldest person
@app.get("/v1/simpson_episodes/oldest_person", operation_id="get_oldest_person", summary="Retrieves the name of the oldest person in the database, sorted by birthdate in ascending order.")
async def get_oldest_person():
    cursor.execute("SELECT name FROM Person ORDER BY birthdate ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get episode IDs for a specific person and credited status
@app.get("/v1/simpson_episodes/episode_ids_by_person_and_credited", operation_id="get_episode_ids_by_person_and_credited", summary="Retrieves a list of episode IDs associated with a given person and their credited status. The person's name and credited status are used to filter the results.")
async def get_episode_ids_by_person_and_credited(person: str = Query(..., description="Name of the person"), credited: str = Query(..., description="Credited status (true or false)")):
    cursor.execute("SELECT episode_id FROM Credit WHERE person = ? AND credited = ?", (person, credited))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

# Endpoint to get roles of persons not born in a specific country
@app.get("/v1/simpson_episodes/roles_by_non_birth_country", operation_id="get_roles_by_non_birth_country", summary="Retrieves the roles of individuals who were not born in the specified country. The operation filters the list of persons based on their birth country and returns the corresponding roles from the credit records.")
async def get_roles_by_non_birth_country(birth_country: str = Query(..., description="Birth country")):
    cursor.execute("SELECT T2.role FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T1.birth_country != ?", (birth_country,))
    result = cursor.fetchall()
    if not result:
        return {"roles": []}
    return {"roles": [row[0] for row in result]}

# Endpoint to get the count of episodes with a specific star rating and the highest rating
@app.get("/v1/simpson_episodes/count_episodes_by_stars_and_highest_rating", operation_id="get_count_episodes_by_stars_and_highest_rating", summary="Retrieves the number of episodes that have a specified star rating and the highest rating. The star rating is provided as an input parameter.")
async def get_count_episodes_by_stars_and_highest_rating(stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT COUNT(*) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ? ORDER BY T1.rating LIMIT 1", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episode summaries based on a person's credit
@app.get("/v1/simpson_episodes/episode_summaries_by_person", operation_id="get_episode_summaries", summary="Retrieves summaries of episodes in which a specified person has a credit. The person's name is used to filter the episodes.")
async def get_episode_summaries(person: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.summary FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.person = ?", (person,))
    result = cursor.fetchall()
    if not result:
        return {"summaries": []}
    return {"summaries": [row[0] for row in result]}

# Endpoint to get roles based on a person's nickname
@app.get("/v1/simpson_episodes/roles_by_nickname", operation_id="get_roles_by_nickname", summary="Retrieves the roles associated with a person based on their specified nickname. The operation returns a list of roles that the person has been credited with in the database.")
async def get_roles_by_nickname(nickname: str = Query(..., description="Nickname of the person")):
    cursor.execute("SELECT T2.role FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T1.nickname = ?", (nickname,))
    result = cursor.fetchall()
    if not result:
        return {"roles": []}
    return {"roles": [row[0] for row in result]}

# Endpoint to get the top episode based on star rating
@app.get("/v1/simpson_episodes/top_episode_by_stars", operation_id="get_top_episode_by_stars", summary="Retrieves the top-rated episode based on the specified number of stars. The episode is determined by the highest percentage of votes with the given star rating. The result is a single episode ID.")
async def get_top_episode_by_stars(stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT T1.episode_id FROM Award AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ? ORDER BY T2.percent DESC LIMIT 1", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"episode_id": []}
    return {"episode_id": result[0]}

# Endpoint to get distinct persons based on episode title, category, and credited status
@app.get("/v1/simpson_episodes/distinct_persons_by_episode_title_category_credited", operation_id="get_distinct_persons", summary="Retrieve a unique list of individuals who have been credited in a specific episode, based on the provided episode title, credit category, and credited status.")
async def get_distinct_persons(title: str = Query(..., description="Title of the episode"), category: str = Query(..., description="Category of the credit"), credited: str = Query(..., description="Credited status")):
    cursor.execute("SELECT DISTINCT T2.person FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T2.category = ? AND T2.credited = ?", (title, category, credited))
    result = cursor.fetchall()
    if not result:
        return {"persons": []}
    return {"persons": [row[0] for row in result]}

# Endpoint to get birthdates of award winners
@app.get("/v1/simpson_episodes/birthdates_of_award_winners", operation_id="get_birthdates_of_winners", summary="Retrieves the birthdates of individuals who have won awards with the specified result. The operation filters the list of award winners based on the provided award result and returns their birthdates.")
async def get_birthdates_of_winners(result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T1.birthdate FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.result = ?", (result,))
    result = cursor.fetchall()
    if not result:
        return {"birthdates": []}
    return {"birthdates": [row[0] for row in result]}

# Endpoint to get the top person based on role and star rating
@app.get("/v1/simpson_episodes/top_person_by_role_stars", operation_id="get_top_person_by_role_stars", summary="Retrieves the person with the highest occurrence in a specific role who has been rated with a certain number of stars. The role and star rating are provided as input parameters.")
async def get_top_person_by_role_stars(role: str = Query(..., description="Role of the person"), stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT T1.person FROM Credit AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T1.role = ? AND T2.stars = ? GROUP BY T1.person ORDER BY COUNT(*) DESC LIMIT 1", (role, stars))
    result = cursor.fetchone()
    if not result:
        return {"person": []}
    return {"person": result[0]}

# Endpoint to get awards based on result and air date
@app.get("/v1/simpson_episodes/awards_by_result_air_date", operation_id="get_awards_by_result_air_date", summary="Retrieves a list of awards that match the specified result and air date. The result parameter filters awards based on their outcome, while the air date parameter narrows down the search to a specific broadcast date. The response includes the names of the awards that meet the provided criteria.")
async def get_awards_by_result_air_date(result: str = Query(..., description="Result of the award"), air_date: str = Query(..., description="Air date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.award FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.result = ? AND T2.air_date = ?", (result, air_date))
    result = cursor.fetchall()
    if not result:
        return {"awards": []}
    return {"awards": [row[0] for row in result]}

# Endpoint to get distinct persons based on credit category
@app.get("/v1/simpson_episodes/distinct_persons_by_credit_category", operation_id="get_distinct_persons_by_category", summary="Retrieves a list of unique individuals, along with their respective details, based on a specified credit category. The response includes each person's name, birthdate, birth name, birth place, birth region, birth country, height in meters, and nickname.")
async def get_distinct_persons_by_category(category: str = Query(..., description="Category of the credit")):
    cursor.execute("SELECT DISTINCT person, name, birthdate, birth_name, birth_place, birth_region, birth_country, height_meters, nickname FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T2.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"persons": []}
    return {"persons": [{"person": row[0], "name": row[1], "birthdate": row[2], "birth_name": row[3], "birth_place": row[4], "birth_region": row[5], "birth_country": row[6], "height_meters": row[7], "nickname": row[8]} for row in result]}

# Endpoint to get keywords based on episode number in series
@app.get("/v1/simpson_episodes/keywords_by_episode_number", operation_id="get_keywords_by_episode_number", summary="Retrieves a list of keywords associated with a specific episode in the series, based on the episode's number in the series.")
async def get_keywords_by_episode_number(number_in_series: int = Query(..., description="Number in series of the episode")):
    cursor.execute("SELECT T2.keyword FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.number_in_series = ?", (number_in_series,))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get distinct episode IDs based on star rating and vote threshold
@app.get("/v1/simpson_episodes/distinct_episode_ids_by_stars_votes", operation_id="get_distinct_episode_ids", summary="Retrieve a unique list of episode IDs that meet a specified star rating and surpass a given vote percentage threshold. The threshold is calculated as a percentage of the total votes for episodes with the same star rating.")
async def get_distinct_episode_ids(stars: int = Query(..., description="Star rating of the episode"), threshold: float = Query(..., description="Vote threshold as a percentage")):
    cursor.execute("SELECT DISTINCT T1.episode_id FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ? AND T2.votes > 0.7 * ( SELECT CAST(COUNT(votes) AS REAL) / COUNT(CASE WHEN stars = ? THEN 1 ELSE 0 END) FROM Vote );", (stars, stars))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

# Endpoint to get the percentage of votes for nominees
@app.get("/v1/simpson_episodes/percentage_votes_for_nominees", operation_id="get_percentage_votes_for_nominees", summary="Retrieves the percentage of total votes received by episodes that were nominees for an award. The result type parameter specifies the nominee category for which the percentage is calculated.")
async def get_percentage_votes_for_nominees(result: str = Query(..., description="Result type (e.g., 'Nominee')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.result = ? THEN T2.votes ELSE 0 END) AS REAL) * 100 / SUM(T2.votes) FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id;", (result,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get names of persons who have a nickname
@app.get("/v1/simpson_episodes/names_with_nicknames", operation_id="get_names_with_nicknames", summary="Retrieves a list of names belonging to individuals who have a registered nickname. This operation does not require any input parameters and returns a simple list of names.")
async def get_names_with_nicknames():
    cursor.execute("SELECT name FROM Person WHERE nickname IS NOT NULL;")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the birth country of the tallest person
@app.get("/v1/simpson_episodes/birth_country_tallest_person", operation_id="get_birth_country_tallest_person", summary="Retrieves the birth country of the tallest person in the database, sorted by height in descending order and limited to the top result.")
async def get_birth_country_tallest_person():
    cursor.execute("SELECT birth_country FROM Person ORDER BY height_meters DESC LIMIT 1;")
    result = cursor.fetchone()
    if not result:
        return {"birth_country": []}
    return {"birth_country": result[0]}

# Endpoint to get the average height of persons from a specific birth country
@app.get("/v1/simpson_episodes/average_height_by_birth_country", operation_id="get_average_height_by_birth_country", summary="Retrieves the average height of persons born in a specified country. The calculation is based on the height data of individuals from the requested birth country.")
async def get_average_height_by_birth_country(birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT AVG(height_meters) FROM Person WHERE birth_country = ?;", (birth_country,))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the percentage of persons born in a specific region after a certain year
@app.get("/v1/simpson_episodes/percentage_born_in_region_after_year", operation_id="get_percentage_born_in_region_after_year", summary="Retrieves the percentage of individuals born in a specified region after a given year. The calculation is based on the total number of persons born in the specified region after the provided year, divided by the total number of birthdates in the database. The birth region and year are provided as input parameters.")
async def get_percentage_born_in_region_after_year(birth_region: str = Query(..., description="Birth region of the person"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN birth_region = ? AND SUBSTR(birthdate, 1, 4) > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(birthdate) FROM Person;", (birth_region, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of persons not born in a specific region and country
@app.get("/v1/simpson_episodes/count_persons_not_born_in_region_country", operation_id="get_count_persons_not_born_in_region_country", summary="Retrieves the total count of individuals who were not born in a specified region and country. The response is based on the provided birth region and country parameters, which are used to filter the data.")
async def get_count_persons_not_born_in_region_country(birth_region: str = Query(..., description="Birth region of the person"), birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT COUNT(name) FROM Person WHERE birth_region != ? AND birth_country != ?;", (birth_region, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episode titles by episode IDs
@app.get("/v1/simpson_episodes/episode_titles_by_ids", operation_id="get_episode_titles_by_ids", summary="Retrieves the titles of up to three episodes specified by their unique IDs. This operation allows you to look up episode titles using the provided episode IDs, enabling you to access and display episode-specific information.")
async def get_episode_titles_by_ids(episode_id1: str = Query(..., description="First episode ID"), episode_id2: str = Query(..., description="Second episode ID"), episode_id3: str = Query(..., description="Third episode ID")):
    cursor.execute("SELECT title FROM Episode WHERE episode_id IN (?, ?, ?);", (episode_id1, episode_id2, episode_id3))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the episode image by episode number
@app.get("/v1/simpson_episodes/episode_image_by_number", operation_id="get_episode_image_by_number", summary="Retrieves the image associated with a specific episode of The Simpsons, identified by its unique episode number.")
async def get_episode_image_by_number(episode: int = Query(..., description="Episode number")):
    cursor.execute("SELECT episode_image FROM Episode WHERE episode = ?;", (episode,))
    result = cursor.fetchone()
    if not result:
        return {"episode_image": []}
    return {"episode_image": result[0]}

# Endpoint to get votes for episodes with a rating above a certain value
@app.get("/v1/simpson_episodes/votes_by_rating", operation_id="get_votes_by_rating", summary="Retrieve the total number of votes for episodes that have a rating higher than the specified minimum value. This operation allows you to filter episodes based on their rating and provides the aggregated votes for the qualifying episodes.")
async def get_votes_by_rating(rating: float = Query(..., description="Minimum rating of the episode")):
    cursor.execute("SELECT votes FROM Episode WHERE rating > ?;", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"votes": []}
    return {"votes": [row[0] for row in result]}

# Endpoint to get the count of episodes based on credit category
@app.get("/v1/simpson_episodes/count_episodes_by_credit_category", operation_id="get_count_episodes_by_credit_category", summary="Retrieves the total number of episodes associated with a specific credit category. The category is provided as an input parameter.")
async def get_count_episodes_by_credit_category(category: str = Query(..., description="Credit category")):
    cursor.execute("SELECT COUNT(*) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons based on role and birth country
@app.get("/v1/simpson_episodes/count_persons_by_role_and_birth_country", operation_id="get_count_persons_by_role_and_birth_country", summary="Retrieves the total number of persons categorized by their role and birth country. The operation requires specifying the role and birth country as input parameters to filter the count accordingly.")
async def get_count_persons_by_role_and_birth_country(role: str = Query(..., description="Role of the person"), birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT COUNT(*) FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T2.role = ? AND T1.birth_country = ?", (role, birth_country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons based on birthdate and role
@app.get("/v1/simpson_episodes/count_persons_by_birthdate_and_role", operation_id="get_count_persons_by_birthdate_and_role", summary="Retrieves the total count of persons who were born after a specified date and have a particular role. The birthdate should be provided in 'YYYY-MM-DD' format. The role parameter determines the type of role for which the count is calculated.")
async def get_count_persons_by_birthdate_and_role(birthdate: str = Query(..., description="Birthdate of the person in 'YYYY-MM-DD' format"), role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT COUNT(*) FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE STRFTIME(T1.birthdate) > ? AND T2.role = ?", (birthdate, role))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episode summaries based on credited status
@app.get("/v1/simpson_episodes/episode_summaries_by_credited_status", operation_id="get_episode_summaries_by_credited_status", summary="Retrieves summaries of episodes that are either credited or not, based on the provided credited status. This operation returns a list of episode summaries that match the specified credited status, as determined by the relationship between the Episode and Credit tables in the database.")
async def get_episode_summaries_by_credited_status(credited: str = Query(..., description="Credited status (true or false)")):
    cursor.execute("SELECT T1.summary FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.credited = ?", (credited,))
    result = cursor.fetchall()
    if not result:
        return {"summaries": []}
    return {"summaries": [row[0] for row in result]}

# Endpoint to get episode ratings based on person
@app.get("/v1/simpson_episodes/episode_ratings_by_person", operation_id="get_episode_ratings_by_person", summary="Retrieves the ratings of episodes in which a specified person is credited. The person's name is used to filter the episodes and return their corresponding ratings.")
async def get_episode_ratings_by_person(person: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.rating FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.person = ?", (person,))
    result = cursor.fetchall()
    if not result:
        return {"ratings": []}
    return {"ratings": [row[0] for row in result]}

# Endpoint to get the average height of persons based on credit category
@app.get("/v1/simpson_episodes/average_height_by_credit_category", operation_id="get_average_height_by_credit_category", summary="Retrieves the average height of persons who have been credited in a specific category. The category is provided as an input parameter.")
async def get_average_height_by_credit_category(category: str = Query(..., description="Credit category")):
    cursor.execute("SELECT AVG(T1.height_meters) FROM Person AS T1 INNER JOIN Credit AS T2 ON T1.name = T2.person WHERE T2.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get distinct characters based on award category, year, and result
@app.get("/v1/simpson_episodes/distinct_characters_by_award_category_year_result", operation_id="get_distinct_characters_by_award_category_year_result", summary="Retrieves a unique list of Simpsons characters who have received awards in a specific category, during a certain year, and with a particular result. The response includes characters who have been awarded in the provided category, year, and result.")
async def get_distinct_characters_by_award_category_year_result(award_category: str = Query(..., description="Award category"), year: int = Query(..., description="Year of the award"), result: str = Query(..., description="Result of the award (e.g., Winner)")):
    cursor.execute("SELECT DISTINCT T2.character FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T1.award_category = ? AND T1.year = ? AND T1.result = ?", (award_category, year, result))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get characters based on award category, year range, and result
@app.get("/v1/simpson_episodes/characters_by_award_category_year_range_result", operation_id="get_characters_by_award_category_year_range_result", summary="Retrieves a list of characters who have received awards within a specified category and year range, excluding those with a specific result. The category, year range, and result to exclude are provided as input parameters.")
async def get_characters_by_award_category_year_range_result(award_category: str = Query(..., description="Award category"), start_year: int = Query(..., description="Start year of the award range"), end_year: int = Query(..., description="End year of the award range"), result: str = Query(..., description="Result of the award (e.g., Winner)")):
    cursor.execute("SELECT T2.character FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T1.award_category = ? AND T1.year BETWEEN ? AND ? AND T1.result != ?", (award_category, start_year, end_year, result))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get the sum of votes for episodes based on person
@app.get("/v1/simpson_episodes/sum_votes_by_person", operation_id="get_sum_votes_by_person", summary="Retrieves the total number of votes for all episodes in which a specified person has been involved. The person's name is provided as an input parameter.")
async def get_sum_votes_by_person(person: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT SUM(T1.votes) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.person = ?", (person,))
    result = cursor.fetchone()
    if not result:
        return {"sum_votes": []}
    return {"sum_votes": result[0]}

# Endpoint to get keywords and persons based on episode ID
@app.get("/v1/simpson_episodes/keywords_persons_by_episode_id", operation_id="get_keywords_persons_by_episode_id", summary="Retrieves a list of keywords and associated persons that appear in a specific episode, identified by the provided episode ID.")
async def get_keywords_persons_by_episode_id(episode_id: str = Query(..., description="Episode ID")):
    cursor.execute("SELECT T1.keyword, T2.person FROM Keyword AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.episode_id = ?", (episode_id,))
    result = cursor.fetchall()
    if not result:
        return {"keywords_persons": []}
    return {"keywords_persons": [{"keyword": row[0], "person": row[1]} for row in result]}

# Endpoint to get keywords from episodes with specific star ratings and percentages
@app.get("/v1/simpson_episodes/keywords_by_stars_percent", operation_id="get_keywords_by_stars_percent", summary="Retrieves keywords from episodes that meet a specified star rating and exceed a given percentage threshold. The operation filters episodes based on their star rating and the percentage of votes that meet the threshold, then extracts the associated keywords.")
async def get_keywords_by_stars_percent(stars: int = Query(..., description="Star rating of the episode"), percent: float = Query(..., description="Percentage threshold for the episode")):
    cursor.execute("SELECT T1.keyword FROM Keyword AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T2.stars = ? AND T2.percent > ?", (stars, percent))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get the percentage of votes with a specific star rating for a given episode title
@app.get("/v1/simpson_episodes/percentage_votes_by_stars_title", operation_id="get_percentage_votes_by_stars_title", summary="Retrieves the percentage of votes with a specific star rating for a given episode title. The operation calculates the proportion of votes with the specified star rating out of the total votes for the episode. The input parameters include the star rating and the episode title.")
async def get_percentage_votes_by_stars_title(stars: int = Query(..., description="Star rating of the votes"), title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.stars = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ?", (stars, title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to check if a specific keyword is associated with a given episode title
@app.get("/v1/simpson_episodes/keyword_check_by_title", operation_id="get_keyword_check_by_title", summary="Check if a specific keyword is associated with a given episode title")
async def get_keyword_check_by_title(keyword: str = Query(..., description="Keyword to check"), title: str = Query(..., description="Title of the episode")):
    cursor.execute("SELECT CASE WHEN T2.Keyword = ? THEN 'Yes' ELSE 'No' END AS result FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ?", (keyword, title))
    result = cursor.fetchone()
    if not result:
        return {"result": []}
    return {"result": result[0]}

# Endpoint to get episode titles based on award and year
@app.get("/v1/simpson_episodes/episode_titles_by_award_year", operation_id="get_episode_titles_by_award_year", summary="Retrieves the titles of episodes that have received a specific award in a given year. The operation filters episodes based on the provided award name and year, and returns a list of corresponding episode titles.")
async def get_episode_titles_by_award_year(award: str = Query(..., description="Award name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.award = ? AND SUBSTR(T1.year, 1, 4) = ?", (award, year))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get birth places of persons based on award and role
@app.get("/v1/simpson_episodes/birth_places_by_award_role", operation_id="get_birth_places_by_award_role", summary="Retrieves the birth places of individuals who have received a specific award for a particular role. The operation requires the name of the award and the role as input parameters.")
async def get_birth_places_by_award_role(award: str = Query(..., description="Award name"), role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT T1.birth_place FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.award = ? AND T2.role = ?", (award, role))
    result = cursor.fetchall()
    if not result:
        return {"birth_places": []}
    return {"birth_places": [row[0] for row in result]}

# Endpoint to get the sum of stars grouped by star ratings for a specific award category
@app.get("/v1/simpson_episodes/sum_stars_by_award_category_range", operation_id="get_sum_stars_by_award_category_range", summary="Retrieves the total number of stars, grouped by star ratings, for a specified award category within a given range. The range is defined by a minimum and maximum star rating.")
async def get_sum_stars_by_award_category_range(award_category: str = Query(..., description="Award category"), min_stars: int = Query(..., description="Minimum star rating"), max_stars: int = Query(..., description="Maximum star rating")):
    cursor.execute("SELECT T2.stars, SUM(T2.stars) FROM Award AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.award_category = ? AND T2.stars BETWEEN ? AND ? GROUP BY T2.stars", (award_category, min_stars, max_stars))
    result = cursor.fetchall()
    if not result:
        return {"sum_stars": []}
    return {"sum_stars": [{"stars": row[0], "sum": row[1]} for row in result]}

# Endpoint to get the sum of ratings for episodes with specific award categories
@app.get("/v1/simpson_episodes/sum_ratings_by_award_categories", operation_id="get_sum_ratings_by_award_categories", summary="Get the sum of ratings for episodes with specific award categories")
async def get_sum_ratings_by_award_categories(award_category1: str = Query(..., description="First award category"), award_category2: str = Query(..., description="Second award category")):
    cursor.execute("SELECT SUM(T2.rating) FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.award_category IN (?, ?)", (award_category1, award_category2))
    result = cursor.fetchone()
    if not result:
        return {"sum_ratings": []}
    return {"sum_ratings": result[0]}

# Endpoint to get the sum of percentages for votes based on episode title, rating, and star range
@app.get("/v1/simpson_episodes/sum_percentages_by_title_rating_stars", operation_id="get_sum_percentages_by_title_rating_stars", summary="Retrieves the total percentage of votes for a specific Simpsons episode, filtered by its rating and a range of star ratings. The episode is identified by its title, and the rating and star range are provided as input parameters.")
async def get_sum_percentages_by_title_rating_stars(title: str = Query(..., description="Title of the episode"), rating: float = Query(..., description="Rating of the episode"), min_stars: int = Query(..., description="Minimum star rating"), max_stars: int = Query(..., description="Maximum star rating")):
    cursor.execute("SELECT SUM(T2.percent) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ? AND T1.rating = ? AND T2.stars BETWEEN ? AND ?", (title, rating, min_stars, max_stars))
    result = cursor.fetchone()
    if not result:
        return {"sum_percentages": []}
    return {"sum_percentages": result[0]}

# Endpoint to get the count of distinct episode IDs with star ratings above a threshold
@app.get("/v1/simpson_episodes/count_distinct_episodes_by_stars", operation_id="get_count_distinct_episodes_by_stars", summary="Retrieves the total number of unique episodes that have received a star rating higher than the specified minimum threshold.")
async def get_count_distinct_episodes_by_stars(min_stars: int = Query(..., description="Minimum star rating")):
    cursor.execute("SELECT COUNT(DISTINCT episode_id) FROM Vote WHERE stars > ?", (min_stars,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the episode with the highest votes
@app.get("/v1/simpson_episodes/top_voted_episode", operation_id="get_top_voted_episode", summary="Retrieves the top-voted episode(s) from the database, based on the specified limit. The operation returns the episode ID(s) of the episode(s) with the highest number of votes, up to the provided limit.")
async def get_top_voted_episode(limit: int = Query(1, description="Limit the number of top voted episodes to return")):
    cursor.execute("SELECT episode_id FROM Vote ORDER BY votes DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"episode_id": []}
    return {"episode_id": result[0]}

# Endpoint to get the count of episodes with a specific keyword
@app.get("/v1/simpson_episodes/count_episodes_by_keyword", operation_id="get_count_episodes_by_keyword", summary="Retrieves the total number of episodes linked to a given keyword. The keyword is used to filter the episodes and determine the count.")
async def get_count_episodes_by_keyword(keyword: str = Query(..., description="Keyword to filter episodes")):
    cursor.execute("SELECT COUNT(episode_id) FROM Keyword WHERE keyword = ?", (keyword,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the organization of a specific award
@app.get("/v1/simpson_episodes/award_organization", operation_id="get_award_organization", summary="Retrieve the organization associated with a particular award. This operation requires the unique identifier of the award as input. The output will be the name of the organization that granted the specified award.")
async def get_award_organization(award_id: int = Query(..., description="Award ID to filter the organization")):
    cursor.execute("SELECT organization FROM Award WHERE award_id = ?", (award_id,))
    result = cursor.fetchone()
    if not result:
        return {"organization": []}
    return {"organization": result[0]}

# Endpoint to get the count of awards in a specific year
@app.get("/v1/simpson_episodes/count_awards_by_year", operation_id="get_count_awards_by_year", summary="Retrieves the total number of awards given in a specified year. The year should be provided in the format YYYY.")
async def get_count_awards_by_year(year: str = Query(..., description="Year to filter the awards (format: YYYY)")):
    cursor.execute("SELECT COUNT(award_id) FROM Award WHERE SUBSTR(year, 1, 4) = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of awards in a specific category
@app.get("/v1/simpson_episodes/count_awards_by_category", operation_id="get_count_awards_by_category", summary="Retrieves the total number of awards in a specified category. The category is provided as an input parameter, allowing the user to filter the awards and obtain a precise count.")
async def get_count_awards_by_category(award_category: str = Query(..., description="Award category to filter the awards")):
    cursor.execute("SELECT COUNT(award_id) FROM Award WHERE award_category = ?", (award_category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the birth names of persons with a specific role and height
@app.get("/v1/simpson_episodes/birth_names_by_role_and_height", operation_id="get_birth_names_by_role_and_height", summary="Retrieve the birth names of individuals who have a specified role and are taller than a given height. The role and minimum height are used as filtering criteria to narrow down the results.")
async def get_birth_names_by_role_and_height(role: str = Query(..., description="Role to filter the persons"), height_meters: float = Query(..., description="Minimum height in meters to filter the persons")):
    cursor.execute("SELECT T1.birth_name FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.role = ? AND T1.height_meters > ?", (role, height_meters))
    result = cursor.fetchall()
    if not result:
        return {"birth_names": []}
    return {"birth_names": [row[0] for row in result]}

# Endpoint to get the percentage of award nominees from a specific country
@app.get("/v1/simpson_episodes/percentage_nominees_by_country", operation_id="get_percentage_nominees_by_country", summary="Retrieves the percentage of award nominees from a specified country. This operation filters the nominees based on their birth country and result status, then calculates the percentage of nominees from the specified country relative to the total number of nominees.")
async def get_percentage_nominees_by_country(birth_country: str = Query(..., description="Birth country to filter the nominees"), result: str = Query(..., description="Result status to filter the nominees (e.g., Nominee)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.birth_country = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.result = ?", (birth_country, result))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of award winners taller than a specific height
@app.get("/v1/simpson_episodes/percentage_winners_by_height", operation_id="get_percentage_winners_by_height", summary="Retrieves the percentage of award winners who have a height greater than the specified minimum height. The result status of the awards is also considered for filtering the winners.")
async def get_percentage_winners_by_height(height_meters: float = Query(..., description="Minimum height in meters to filter the winners"), result: str = Query(..., description="Result status to filter the winners (e.g., Winner)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.height_meters > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.result = ?", (height_meters, result))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the titles of episodes with a specific star rating
@app.get("/v1/simpson_episodes/episode_titles_by_stars", operation_id="get_episode_titles_by_stars", summary="Retrieve the titles of episodes that have been rated with a specific number of stars. The operation filters episodes based on the provided star rating and returns their corresponding titles.")
async def get_episode_titles_by_stars(stars: int = Query(..., description="Star rating to filter the episodes")):
    cursor.execute("SELECT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars = ?", (stars,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the star ratings of a specific episode
@app.get("/v1/simpson_episodes/episode_stars_by_title", operation_id="get_episode_stars_by_title", summary="Retrieve the star ratings associated with a specific episode, filtered by the episode's title. The provided title is used to identify the relevant episode and its corresponding star ratings.")
async def get_episode_stars_by_title(title: str = Query(..., description="Title of the episode to filter the star ratings")):
    cursor.execute("SELECT T2.stars FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"stars": []}
    return {"stars": [row[0] for row in result]}

# Endpoint to get the count of distinct episodes based on air year and minimum stars
@app.get("/v1/simpson_episodes/count_episodes_by_year_stars", operation_id="get_count_episodes_by_year_stars", summary="Retrieves the number of unique episodes that aired in a given year and received more than a specified number of stars. The year is provided in 'YYYY' format, and the minimum star rating is a numerical value.")
async def get_count_episodes_by_year_stars(air_year: str = Query(..., description="Year of airing in 'YYYY' format"), min_stars: int = Query(..., description="Minimum number of stars")):
    cursor.execute("SELECT COUNT(DISTINCT T2.episode_id) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE strftime('%Y', T1.air_date) = ? AND T2.stars > ?", (air_year, min_stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct years a character received awards
@app.get("/v1/simpson_episodes/award_years_by_character", operation_id="get_award_years_by_character", summary="Retrieves the unique years in which a specified character was awarded. The operation filters the awards database based on the provided character name and returns the distinct years associated with the character's awards.")
async def get_award_years_by_character(character: str = Query(..., description="Character name")):
    cursor.execute("SELECT DISTINCT T1.year FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T2.character = ?", (character,))
    result = cursor.fetchall()
    if not result:
        return {"years": []}
    return {"years": [row[0] for row in result]}

# Endpoint to get distinct award categories for a specific character
@app.get("/v1/simpson_episodes/award_categories_by_character", operation_id="get_award_categories_by_character", summary="Retrieves a unique set of award categories associated with a specified character. The character's name is required as an input parameter.")
async def get_award_categories_by_character(character: str = Query(..., description="Character name")):
    cursor.execute("SELECT DISTINCT T1.award_category FROM Award AS T1 INNER JOIN Character_Award AS T2 ON T1.award_id = T2.award_id WHERE T2.character = ?", (character,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get keywords for episodes aired in a specific year
@app.get("/v1/simpson_episodes/keywords_by_air_year", operation_id="get_keywords_by_air_year", summary="Retrieves a list of keywords associated with episodes that aired in a specific year. The year is provided in 'YYYY' format as an input parameter.")
async def get_keywords_by_air_year(air_year: str = Query(..., description="Year of airing in 'YYYY' format")):
    cursor.execute("SELECT T2.keyword FROM Episode AS T1 INNER JOIN Keyword AS T2 ON T1.episode_id = T2.episode_id WHERE SUBSTR(T1.air_date, 1, 4) = ?", (air_year,))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get the count of distinct episodes based on air year and maximum stars
@app.get("/v1/simpson_episodes/count_episodes_by_year_max_stars", operation_id="get_count_episodes_by_year_max_stars", summary="Retrieves the number of unique episodes that aired in a given year and received less than a specified number of stars. The year is provided in 'YYYY' format, and the maximum number of stars is a numerical value.")
async def get_count_episodes_by_year_max_stars(air_year: str = Query(..., description="Year of airing in 'YYYY' format"), max_stars: int = Query(..., description="Maximum number of stars")):
    cursor.execute("SELECT COUNT(DISTINCT T2.episode_id) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE strftime('%Y', T1.air_date) = ? AND T2.stars < ?", (air_year, max_stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get birth names of persons based on role and birth country
@app.get("/v1/simpson_episodes/birth_names_by_role_country", operation_id="get_birth_names_by_role_country", summary="Retrieves the birth names of individuals who have a specific role and were born in a particular country. The role and birth country are used as filters to determine the relevant individuals.")
async def get_birth_names_by_role_country(role: str = Query(..., description="Role of the person"), birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT T1.birth_name FROM Person AS T1 INNER JOIN Award AS T2 ON T1.name = T2.person WHERE T2.role = ? AND T1.birth_country = ?", (role, birth_country))
    result = cursor.fetchall()
    if not result:
        return {"birth_names": []}
    return {"birth_names": [row[0] for row in result]}

# Endpoint to get the count of awards based on year and result
@app.get("/v1/simpson_episodes/count_awards_by_year_result", operation_id="get_count_awards_by_year_result", summary="Retrieves the total number of awards that match a specified year and result. The year should be provided in 'YYYY' format, and the result should indicate the outcome of the award, such as 'Winner'.")
async def get_count_awards_by_year_result(year: str = Query(..., description="Year in 'YYYY' format"), result: str = Query(..., description="Result of the award (e.g., 'Winner')")):
    cursor.execute("SELECT COUNT(award_id) FROM Award WHERE SUBSTR(year, 1, 4) = ? AND result = ?", (year, result))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of episodes with a rating below a specified value
@app.get("/v1/simpson_episodes/count_episodes_below_rating", operation_id="get_count_episodes_below_rating", summary="Retrieves the total number of episodes that have a rating lower than the provided threshold. This operation is useful for understanding the distribution of episode ratings and identifying episodes that fall below a certain quality standard.")
async def get_count_episodes_below_rating(rating: int = Query(..., description="Rating threshold")):
    cursor.execute("SELECT COUNT(episode_id) FROM Episode WHERE rating < ?", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names of people based on birth region and country
@app.get("/v1/simpson_episodes/get_names_by_birth_region_country", operation_id="get_names_by_birth_region_country", summary="Retrieves the names of individuals who were born in a specific region and country. The operation filters the data based on the provided birth region and birth country parameters, returning a list of matching names.")
async def get_names_by_birth_region_country(birth_region: str = Query(..., description="Birth region of the person"), birth_country: str = Query(..., description="Birth country of the person")):
    cursor.execute("SELECT name FROM Person WHERE birth_region = ? AND birth_country = ?", (birth_region, birth_country))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of credited roles in specific episodes
@app.get("/v1/simpson_episodes/count_credited_roles_in_episodes", operation_id="get_count_credited_roles_in_episodes", summary="Retrieve the total number of credited roles in a set of specified episodes, filtered by a specific credited status and role.")
async def get_count_credited_roles_in_episodes(episode_id_1: str = Query(..., description="Episode ID 1"), episode_id_2: str = Query(..., description="Episode ID 2"), episode_id_3: str = Query(..., description="Episode ID 3"), episode_id_4: str = Query(..., description="Episode ID 4"), episode_id_5: str = Query(..., description="Episode ID 5"), episode_id_6: str = Query(..., description="Episode ID 6"), credited: str = Query(..., description="Credited status"), role: str = Query(..., description="Role")):
    cursor.execute("SELECT COUNT(credited) FROM Credit WHERE episode_id IN (?, ?, ?, ?, ?, ?) AND credited = ? AND role = ?", (episode_id_1, episode_id_2, episode_id_3, episode_id_4, episode_id_5, episode_id_6, credited, role))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get episode titles based on award organization, year, and result
@app.get("/v1/simpson_episodes/get_episode_titles_by_award", operation_id="get_episode_titles_by_award", summary="Retrieves the titles of episodes that have received a specific award from a given organization in a particular year. The award result (e.g., won, nominated) is also considered in the search.")
async def get_episode_titles_by_award(organization: str = Query(..., description="Award organization"), year: int = Query(..., description="Year of the award"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.organization = ? AND T1.year = ? AND T1.result = ?", (organization, year, result))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get distinct episode IDs based on air date year and stars
@app.get("/v1/simpson_episodes/get_distinct_episode_ids_by_air_date_stars", operation_id="get_distinct_episode_ids_by_air_date_stars", summary="Retrieves a list of unique episode IDs that aired in a specific year and received fewer stars than the provided threshold. The year is specified in the 'YYYY' format, and the star threshold is a numerical value.")
async def get_distinct_episode_ids_by_air_date_stars(air_date_year: str = Query(..., description="Year part of the air date in 'YYYY' format"), stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT DISTINCT T1.episode_id FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE SUBSTR(T1.air_date, 1, 4) = ? AND T2.stars < ?", (air_date_year, stars))
    result = cursor.fetchall()
    if not result:
        return {"episode_ids": []}
    return {"episode_ids": [row[0] for row in result]}

# Endpoint to get the top category based on person and credited status
@app.get("/v1/simpson_episodes/get_top_category_by_person_credited", operation_id="get_top_category_by_person_credited", summary="Retrieves the category with the highest number of votes for episodes in which a specified person has been credited or not credited, depending on the provided credited status.")
async def get_top_category_by_person_credited(person: str = Query(..., description="Name of the person"), credited: str = Query(..., description="Credited status")):
    cursor.execute("SELECT T2.category FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.person = ? AND T2.credited = ? ORDER BY T1.votes DESC LIMIT 1", (person, credited))
    result = cursor.fetchone()
    if not result:
        return {"category": []}
    return {"category": result[0]}

# Endpoint to get vote percentages based on air date year and vote range
@app.get("/v1/simpson_episodes/get_vote_percentages_by_air_date_votes", operation_id="get_vote_percentages_by_air_date_votes", summary="Retrieves the percentage of votes for episodes that aired in a specific year and received a certain range of votes. The year is provided in 'YYYY' format, and the range is defined by a minimum and maximum number of votes.")
async def get_vote_percentages_by_air_date_votes(air_date_year: str = Query(..., description="Year part of the air date in 'YYYY' format"), min_votes: int = Query(..., description="Minimum number of votes"), max_votes: int = Query(..., description="Maximum number of votes")):
    cursor.execute("SELECT T2.percent FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE SUBSTR(T1.air_date, 1, 4) = ? AND T1.votes BETWEEN ? AND ?", (air_date_year, min_votes, max_votes))
    result = cursor.fetchall()
    if not result:
        return {"percentages": []}
    return {"percentages": [row[0] for row in result]}

# Endpoint to get episode IDs and titles based on credited status, person, and role
@app.get("/v1/simpson_episodes/get_episode_ids_titles_by_credited_person_role", operation_id="get_episode_ids_titles_by_credited_person_role", summary="Retrieves a list of episode IDs and their corresponding titles for episodes where a specific person has been credited in a particular role. The operation filters episodes based on the provided credited status, person's name, and role.")
async def get_episode_ids_titles_by_credited_person_role(credited: str = Query(..., description="Credited status"), person: str = Query(..., description="Name of the person"), role: str = Query(..., description="Role")):
    cursor.execute("SELECT T1.episode_id, T1.title FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.credited = ? AND T2.person = ? AND T2.role = ?", (credited, person, role))
    result = cursor.fetchall()
    if not result:
        return {"episodes": []}
    return {"episodes": [{"episode_id": row[0], "title": row[1]} for row in result]}

# Endpoint to get the count of awards based on organization, result, and vote percentage
@app.get("/v1/simpson_episodes/count_awards_by_organization_result_percent", operation_id="count_awards_by_organization_result_percent", summary="Retrieves the total count of awards from a specific organization, with a particular result, and a vote percentage greater than the provided value. This operation is useful for analyzing the distribution of awards based on the organization, result, and vote percentage.")
async def count_awards_by_organization_result_percent(organization: str = Query(..., description="Award organization"), result: str = Query(..., description="Result of the award"), percent: float = Query(..., description="Vote percentage")):
    cursor.execute("SELECT COUNT(*) FROM Award AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T1.organization = ? AND T1.result = ? AND T2.percent > ?", (organization, result, percent))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct episode titles based on votes, award category, and result
@app.get("/v1/simpson_episodes/distinct_episode_titles", operation_id="get_distinct_episode_titles", summary="Retrieve a list of unique episode titles that meet the specified criteria. The criteria include a minimum number of votes, a specific award category, and a particular award result. This operation returns episodes that have been recognized for their excellence in the specified award category and have achieved the specified result.")
async def get_distinct_episode_titles(min_votes: int = Query(..., description="Minimum number of votes"), award_category: str = Query(..., description="Award category"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT DISTINCT T2.title FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T2.votes > ? AND T1.award_category = ? AND T1.result = ?", (min_votes, award_category, result))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get persons based on episode title, role, credited status, and category
@app.get("/v1/simpson_episodes/persons_by_episode_title_role_credited_category", operation_id="get_persons_by_episode_title_role_credited_category", summary="Retrieve a list of persons who have been credited in a specific role for a given episode, based on the provided title, role, credited status, and category.")
async def get_persons_by_episode_title_role_credited_category(title: str = Query(..., description="Title of the episode"), role: str = Query(..., description="Role of the person"), credited: str = Query(..., description="Credited status"), category: str = Query(..., description="Category of the credit")):
    cursor.execute("SELECT T2.person FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = ? AND T2.role = ? AND T2.credited = ? AND T2.category = ?", (title, role, credited, category))
    result = cursor.fetchall()
    if not result:
        return {"persons": []}
    return {"persons": [row[0] for row in result]}

# Endpoint to get stars based on air date month and year
@app.get("/v1/simpson_episodes/stars_by_air_date_month_year", operation_id="get_stars_by_air_date_month_year", summary="Retrieves the stars (votes) for episodes that aired within a specified month and year. The input parameter 'air_date_month_year' in 'YYYY-MM' format is used to filter the episodes.")
async def get_stars_by_air_date_month_year(air_date_month_year: str = Query(..., description="Air date month and year in 'YYYY-MM' format")):
    cursor.execute("SELECT T2.stars FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE SUBSTR(T1.air_date, 1, 7) = ?", (air_date_month_year,))
    result = cursor.fetchall()
    if not result:
        return {"stars": []}
    return {"stars": [row[0] for row in result]}

# Endpoint to get episode titles based on air date, award category, stars, and result
@app.get("/v1/simpson_episodes/episode_titles_by_air_date_award_category_stars_result", operation_id="get_episode_titles_by_air_date_award_category_stars_result", summary="Retrieves the titles of episodes that aired on a specific date, belong to a certain award category, received a given number of stars, and achieved a particular result. The air date, award category, number of stars, and result are provided as input parameters.")
async def get_episode_titles_by_air_date_award_category_stars_result(air_date: str = Query(..., description="Air date in 'YYYY-MM-DD' format"), award_category: str = Query(..., description="Award category"), stars: int = Query(..., description="Number of stars"), result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T3.title FROM Award AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id INNER JOIN Episode AS T3 ON T1.episode_id = T3.episode_id WHERE T3.air_date = ? AND T1.award_category = ? AND T2.stars = ? AND T1.result = ?", (air_date, award_category, stars, result))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the top award category based on result and votes
@app.get("/v1/simpson_episodes/top_award_category_by_result_votes", operation_id="get_top_award_category_by_result_votes", summary="Retrieves the top award category based on the provided award result and the number of votes received by the associated episode. The award category with the highest number of votes is returned.")
async def get_top_award_category_by_result_votes(result: str = Query(..., description="Result of the award")):
    cursor.execute("SELECT T1.award_category FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T1.result = ? ORDER BY T2.votes DESC LIMIT 1", (result,))
    result = cursor.fetchone()
    if not result:
        return {"award_category": []}
    return {"award_category": result[0]}

# Endpoint to get the count of episodes based on credited status, person, air date year, and role
@app.get("/v1/simpson_episodes/count_episodes_by_credited_person_air_date_year_role", operation_id="get_count_episodes_by_credited_person_air_date_year_role", summary="Retrieve the total number of episodes in which a specific person was credited, based on a given role and air date year. This operation provides a count of episodes that meet the specified criteria, offering insights into a person's involvement in the series.")
async def get_count_episodes_by_credited_person_air_date_year_role(credited: str = Query(..., description="Credited status"), person: str = Query(..., description="Person's name"), air_date_year: str = Query(..., description="Air date year in 'YYYY' format"), role: str = Query(..., description="Role of the person")):
    cursor.execute("SELECT COUNT(*) FROM Episode AS T1 INNER JOIN Credit AS T2 ON T1.episode_id = T2.episode_id WHERE T2.credited = ? AND T2.person = ? AND SUBSTR(T1.air_date, 1, 4) = ? AND T2.role = ?", (credited, person, air_date_year, role))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct episode titles based on stars above a certain threshold
@app.get("/v1/simpson_episodes/distinct_episode_titles_above_star_threshold", operation_id="get_distinct_episode_titles_above_star_threshold", summary="Retrieves a list of unique episode titles that have received a rating above a certain threshold. The threshold is calculated as 70% of the average rating across all episodes. This operation does not require any input parameters.")
async def get_distinct_episode_titles_above_star_threshold():
    cursor.execute("SELECT DISTINCT T1.title FROM Episode AS T1 INNER JOIN Vote AS T2 ON T2.episode_id = T1.episode_id WHERE T2.stars > 0.7 * ( SELECT AVG(stars) FROM Vote )")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the percentage difference between nominees and winners for a specific episode title and year
@app.get("/v1/simpson_episodes/percentage_difference_nominees_winners", operation_id="get_percentage_difference_nominees_winners", summary="Retrieves the percentage difference between the number of nominations and wins for a specific episode in a given year. The calculation is based on the total count of nominations and wins for the episode in the specified year.")
async def get_percentage_difference_nominees_winners(title: str = Query(..., description="Title of the episode"), year: int = Query(..., description="Year of the award")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T1.result = 'Nominee' THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.result = 'Winner' THEN 1 ELSE 0 END)) AS REAL) * 100 / COUNT(T1.result) FROM Award AS T1 INNER JOIN Episode AS T2 ON T1.episode_id = T2.episode_id WHERE T2.title = ? AND T1.year = ?", (title, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

api_calls = [
    "/v1/simpson_episodes/earliest_birthdate_person",
    "/v1/simpson_episodes/nickname_by_name?name=Dan%20Castellaneta",
    "/v1/simpson_episodes/count_people_by_region_and_year?birth_region=New%20York&year=1970",
    "/v1/simpson_episodes/birth_country_of_award_winners?award=Outstanding%20Voice-Over%20Performance&year=2009&result=Winner",
    "/v1/simpson_episodes/awards_by_nickname?nickname=Doofus&result=Winner",
    "/v1/simpson_episodes/count_award_nominees?birth_country=USA&result=Nominee&award=Outstanding%20Animated%20Program%20(For%20Programming%20Less%20Than%20One%20Hour)&year=2009",
    "/v1/simpson_episodes/distinct_characters_by_award?award=Outstanding%20Voice-Over%20Performance&year=2009&result=Winner",
    "/v1/simpson_episodes/keywords_by_episode_title?title=Lost%20Verizon",
    "/v1/simpson_episodes/count_keywords_by_air_date?air_date=2008-10-19",
    "/v1/simpson_episodes/episode_rating_by_award?award=Outstanding%20Voice-Over%20Performance&year=2009&person=Dan%20Castellaneta",
    "/v1/simpson_episodes/count_votes_by_title_and_stars?title=Lost%20Verizon&stars=7",
    "/v1/simpson_episodes/highest_star_rating_by_title?title=Lost%20Verizon",
    "/v1/simpson_episodes/episode_titles_by_votes_and_stars?votes=200&stars=10",
    "/v1/simpson_episodes/count_episodes_by_year_stars_percent?year=2009&stars=10&percent=15",
    "/v1/simpson_episodes/highest_votes_episode_by_stars?stars=7",
    "/v1/simpson_episodes/weighted_average_stars_by_title?title=Lost%20Verizon",
    "/v1/simpson_episodes/percentage_award_category_by_rating_and_result?award_category=Primetime%20Emmy&rating=7&result=Nominee",
    "/v1/simpson_episodes/episode_titles_by_person_and_role?person=Pamela%20Hayden&role=Ruthie",
    "/v1/simpson_episodes/role_episode_number_by_person_and_title?person=Matt%20Groening&title=In%20the%20Name%20of%20the%20Grandfather",
    "/v1/simpson_episodes/episode_titles_summaries_by_keyword?keyword=eviction",
    "/v1/simpson_episodes/average_stars_by_title?title=Wedding%20for%20Disaster",
    "/v1/simpson_episodes/episode_keywords_by_air_date?air_date=2009-03-22",
    "/v1/simpson_episodes/birth_names_by_role?role=Helen%20Lovejoy",
    "/v1/simpson_episodes/uncredited_roles_percentage?year=2017&award_category=Jupiter%20Award&award=Best%20International%20TV%20Series&result=Winner",
    "/v1/simpson_episodes/episode_count_by_votes?min_votes=1000",
    "/v1/simpson_episodes/person_count_by_birth_place_country?birth_place=New%20York%20City&birth_country=USA",
    "/v1/simpson_episodes/awards_by_person?person=Marc%20Wilmore",
    "/v1/simpson_episodes/person_count_with_nickname",
    "/v1/simpson_episodes/average_height",
    "/v1/simpson_episodes/vote_difference",
    "/v1/simpson_episodes/characters_by_award_year?year=2009&award=Outstanding%20Voice-Over%20Performance",
    "/v1/simpson_episodes/count_people_by_award_year_region?year=2009&award=Comedy%20Series&birth_region=California",
    "/v1/simpson_episodes/episode_titles_by_award_year_result?year=2017&award=Best%20International%20TV%20Series&result=Winner",
    "/v1/simpson_episodes/people_credited_in_episode?title=How%20the%20Test%20Was%20Won&credited=false",
    "/v1/simpson_episodes/episode_titles_by_keywords?keyword1=riot&keyword2=cake",
    "/v1/simpson_episodes/top_voted_episode_by_stars?stars=10",
    "/v1/simpson_episodes/people_credited_by_air_months?start_month=10&end_month=11",
    "/v1/simpson_episodes/people_by_role_in_episode?title=Treehouse%20of%20Horror%20XIX&role=director",
    "/v1/simpson_episodes/count_distinct_roles_in_episode?episode=5",
    "/v1/simpson_episodes/percentage_episodes_with_award?award=Outstanding%20Voice-Over%20Performance&min_votes=950&year=2009",
    "/v1/simpson_episodes/vote_ratio_between_episodes?title1=No%20Loan%20Again,%20Naturally&title2=Coming%20to%20Homerica&stars=5",
    "/v1/simpson_episodes/count_awards_person_organization_result?person=Ian%20Maxtone-Graham&organization=Writers%20Guild%20of%20America,%20USA&result=Nominee",
    "/v1/simpson_episodes/top_person_by_award_result?result=Nominee",
    "/v1/simpson_episodes/top_rated_episode",
    "/v1/simpson_episodes/most_recent_award_year?result=Winner&award=Favorite%20Animated%20Comedy",
    "/v1/simpson_episodes/count_credits_person_credited?person=Dell%20Hake&credited=false",
    "/v1/simpson_episodes/age_of_award_winners?award=Outstanding%20Voice-Over%20Performance&organization=Primetime%20Emmy%20Awards&result=Winner",
    "/v1/simpson_episodes/distinct_characters_award?person=Dan%20Castellaneta&award=Outstanding%20Voice-Over%20Performance&organization=Primetime%20Emmy%20Awards&year=2009",
    "/v1/simpson_episodes/count_episodes_with_awards?year=2009&air_date_pattern=2009-04%",
    "/v1/simpson_episodes/birth_place_by_award_organization_year?award=Best%20Voice-Over%20Performance&organization=Online%20Film%20%26%20Television%20Association&year=2009",
    "/v1/simpson_episodes/sum_votes_by_stars_limit?stars=10&limit=4",
    "/v1/simpson_episodes/count_episodes_by_title_votes?title=No%20Loan%20Again%2C%20Naturally&votes=50",
    "/v1/simpson_episodes/count_persons_by_birth_country_award_result?birth_country=USA&result=Winner",
    "/v1/simpson_episodes/count_persons_by_nickname_credited_episode?nickname=Doofus&credited=true&episode_id=S20-E11",
    "/v1/simpson_episodes/award_name_by_result_limit?result=Winner&limit=1",
    "/v1/simpson_episodes/vote_percent_by_title_stars?title=Sex%2C%20Pies%20and%20Idiot%20Scrapes&stars=9",
    "/v1/simpson_episodes/award_person_by_result_year?result=Winner&year=2009",
    "/v1/simpson_episodes/person_by_birth_place_height?birth_place=Los%20Angeles&height_meters=1.8",
    "/v1/simpson_episodes/episode_by_air_date?air_date=2008-10",
    "/v1/simpson_episodes/award_details_character_result?character=Homer%20Simpson&result=Winner",
    "/v1/simpson_episodes/distinct_award_details_person?person=Billy%20Kimball",
    "/v1/simpson_episodes/award_details_episode_role?episode_id=S20-E13&role=assistant%20director",
    "/v1/simpson_episodes/person_details_category_credited?category=Cast&credited=false",
    "/v1/simpson_episodes/person_keyword_details_episode_title_award_result?title=The%20Good%2C%20the%20Sad%20and%20the%20Drugly&result=Nominee",
    "/v1/simpson_episodes/vote_details_keyword_stars?keyword=arab%20stereotype&stars=10",
    "/v1/simpson_episodes/award_details_organization_result?organization=Jupiter%20Award&result=Winner",
    "/v1/simpson_episodes/distinct_episode_details_stars?stars=1",
    "/v1/simpson_episodes/episode_award_details_rating_air_date_result?min_rating=7&max_rating=10&air_date_year=2008&result=Nominee",
    "/v1/simpson_episodes/distinct_award_details_award_id?award_id=326",
    "/v1/simpson_episodes/oldest_person_details?limit=1",
    "/v1/simpson_episodes/credits_by_birth_country?birth_country=North%20Korea",
    "/v1/simpson_episodes/award_rates_and_details?year=2010",
    "/v1/simpson_episodes/highest_rated_episode?limit=1",
    "/v1/simpson_episodes/count_awards_by_result?result=Winner",
    "/v1/simpson_episodes/episode_titles_by_award_criteria?organization=Primetime%20Emmy%20Awards&award=Outstanding%20Animated%20Program%20(For%20Programming%20Less%20Than%20One%20Hour)&result=Nominee&count=21",
    "/v1/simpson_episodes/average_stars_and_summary?episode_id=S20-E12",
    "/v1/simpson_episodes/persons_by_character_award?character=Homer%20simpson%2020&organization=Primetime%20Emmy%20Awards&award=Outstanding%20Voice-Over%20Performance&result=Winner",
    "/v1/simpson_episodes/birth_names_and_roles_by_person?name=Al%20Jean",
    "/v1/simpson_episodes/count_awards_by_person_year?person=Billy%20Kimball&year=2010&result=Nominee",
    "/v1/simpson_episodes/latest_award_category?result=Winner",
    "/v1/simpson_episodes/award_win_rate?result=Winner",
    "/v1/simpson_episodes/episode_count_by_month?year=2008&month1=10&month2=11",
    "/v1/simpson_episodes/highest_rated_episode_by_year?year=2009",
    "/v1/simpson_episodes/person_categories_roles?person=Bonita%20Pietila",
    "/v1/simpson_episodes/person_by_birth_details?birthdate=1957-10-29&birth_place=Chicago&birth_region=Illinois",
    "/v1/simpson_episodes/persons_by_role?role=producer",
    "/v1/simpson_episodes/award_nominees_age_name?role=composer&organization=Primetime%20Emmy%20Awards&award=Outstanding%20Music%20Composition%20for%20a%20Series%20(Original%20Dramatic%20Score)&result=Nominee&year=2009",
    "/v1/simpson_episodes/most_awarded_episode",
    "/v1/simpson_episodes/person_birth_place?name=Dan%20Castellaneta",
    "/v1/simpson_episodes/count_persons_by_birth_country?birth_country=USA",
    "/v1/simpson_episodes/persons_born_before_year?year=1970",
    "/v1/simpson_episodes/persons_by_birth_year_place_country?birth_year=1958&birth_place=California&birth_country=USA",
    "/v1/simpson_episodes/episode_ids_by_stars_votes?stars=5&votes=100",
    "/v1/simpson_episodes/distinct_episodes_by_range_votes?start_episode=10&end_episode=20&votes=200",
    "/v1/simpson_episodes/keywords_by_award_category?award_category=Primetime%20Emmy",
    "/v1/simpson_episodes/top_person_by_award_category?award_category=Primetime%20Emmy",
    "/v1/simpson_episodes/distinct_episode_ids_by_award_stars?award=Outstanding%20Animated%20Program%20(For%20Programming%20Less%20Than%20One%20Hour)&stars=10",
    "/v1/simpson_episodes/persons_by_role_award_title?role=director&award=Outstanding%20Animated%20Program%20(For%20Programming%20Less%20Than%20One%20Hour)&title=No%20Loan%20Again,%20Naturally",
    "/v1/simpson_episodes/person_names_by_birthyear_region?birth_year=1962&birth_region=California",
    "/v1/simpson_episodes/person_count_by_height_country?height_meters=1.70&birth_country=Canada",
    "/v1/simpson_episodes/award_count_by_type_result?award=Animation&result=Nominee",
    "/v1/simpson_episodes/episode_titles_by_rating_range?min_rating=7&max_rating=10&limit=3",
    "/v1/simpson_episodes/top_episodes_by_votes?limit=5",
    "/v1/simpson_episodes/episode_titles_min_stars_by_votes?limit=3",
    "/v1/simpson_episodes/distinct_characters_by_award_person?award_type=%Voice-Over%&person=Dan%20Castellaneta",
    "/v1/simpson_episodes/episode_ids_by_air_date_year?air_date_year=2008&limit=1",
    "/v1/simpson_episodes/episode_titles_by_keyword?keyword=1930s%20to%202020s&limit=2",
    "/v1/simpson_episodes/episode_titles_by_award_year_person?award_year=2010&person=Joel%20H.%20Cohen",
    "/v1/simpson_episodes/sum_votes_by_title_and_stars?title=Lisa%20the%20Drama%20Queen&stars=5",
    "/v1/simpson_episodes/votes_by_keyword_and_stars?stars=10&keyword=reference%20to%20the%20fantastic%20four",
    "/v1/simpson_episodes/vote_difference_by_title_and_stars?title=The%20Burns%20and%20the%20Bees&stars_high=10&stars_low=1",
    "/v1/simpson_episodes/keyword_of_highest_voted_episode",
    "/v1/simpson_episodes/distinct_episode_ids_by_stars_votes_rating?stars=2&votes=20&min_rating=5.0&max_rating=7.0",
    "/v1/simpson_episodes/episode_with_max_votes",
    "/v1/simpson_episodes/oldest_person",
    "/v1/simpson_episodes/episode_ids_by_person_and_credited?person=Oscar%20Cervantes&credited=true",
    "/v1/simpson_episodes/roles_by_non_birth_country?birth_country=USA",
    "/v1/simpson_episodes/count_episodes_by_stars_and_highest_rating?stars=1",
    "/v1/simpson_episodes/episode_summaries_by_person?person=Emily%20Blunt",
    "/v1/simpson_episodes/roles_by_nickname?nickname=The%20Tiny%20Canadian",
    "/v1/simpson_episodes/top_episode_by_stars?stars=5",
    "/v1/simpson_episodes/distinct_persons_by_episode_title_category_credited?title=In%20the%20Name%20of%20the%20Grandfather&category=Cast&credited=true",
    "/v1/simpson_episodes/birthdates_of_award_winners?result=Winner",
    "/v1/simpson_episodes/top_person_by_role_stars?role=Writer&stars=10",
    "/v1/simpson_episodes/awards_by_result_air_date?result=Winner&air_date=2008-11-30",
    "/v1/simpson_episodes/distinct_persons_by_credit_category?category=Music%20Department",
    "/v1/simpson_episodes/keywords_by_episode_number?number_in_series=426",
    "/v1/simpson_episodes/distinct_episode_ids_by_stars_votes?stars=7&threshold=0.7",
    "/v1/simpson_episodes/percentage_votes_for_nominees?result=Nominee",
    "/v1/simpson_episodes/names_with_nicknames",
    "/v1/simpson_episodes/birth_country_tallest_person",
    "/v1/simpson_episodes/average_height_by_birth_country?birth_country=USA",
    "/v1/simpson_episodes/percentage_born_in_region_after_year?birth_region=California&year=1970",
    "/v1/simpson_episodes/count_persons_not_born_in_region_country?birth_region=Connecticut&birth_country=USA",
    "/v1/simpson_episodes/episode_titles_by_ids?episode_id1=S20-E1&episode_id2=S20-E2&episode_id3=S20-E3",
    "/v1/simpson_episodes/episode_image_by_number?episode=5",
    "/v1/simpson_episodes/votes_by_rating?rating=7",
    "/v1/simpson_episodes/count_episodes_by_credit_category?category=Casting%20Department",
    "/v1/simpson_episodes/count_persons_by_role_and_birth_country?role=additional%20timer&birth_country=USA",
    "/v1/simpson_episodes/count_persons_by_birthdate_and_role?birthdate=1970-01-01&role=animation%20executive%20producer",
    "/v1/simpson_episodes/episode_summaries_by_credited_status?credited=false",
    "/v1/simpson_episodes/episode_ratings_by_person?person=Jason%20Bikowski",
    "/v1/simpson_episodes/average_height_by_credit_category?category=Animation%20Department",
    "/v1/simpson_episodes/distinct_characters_by_award_category_year_result?award_category=Primetime%20Emmy&year=2009&result=Winner",
    "/v1/simpson_episodes/characters_by_award_category_year_range_result?award_category=Primetime%20Emmy&start_year=2009&end_year=2010&result=Winner",
    "/v1/simpson_episodes/sum_votes_by_person?person=Adam%20Kuhlman",
    "/v1/simpson_episodes/keywords_persons_by_episode_id?episode_id=S20-E1",
    "/v1/simpson_episodes/keywords_by_stars_percent?stars=10&percent=29",
    "/v1/simpson_episodes/percentage_votes_by_stars_title?stars=5&title=Sex,%20Pies%20and%20Idiot%20Scrapes",
    "/v1/simpson_episodes/keyword_check_by_title?keyword=limbo%20dancing&title=Dangerous%20Curves",
    "/v1/simpson_episodes/episode_titles_by_award_year?award=Best%20International%20TV%20Series&year=2017",
    "/v1/simpson_episodes/birth_places_by_award_role?award=Outstanding%20Animated%20Program%20(For%20Programming%20Less%20Than%20One%20Hour)&role=co-executive%20producer",
    "/v1/simpson_episodes/sum_stars_by_award_category_range?award_category=Blimp%20Award&min_stars=1&max_stars=5",
    "/v1/simpson_episodes/sum_ratings_by_award_categories?award_category1=Jupiter%20Award&award_category2=WGA%20Award%20(TV)",
    "/v1/simpson_episodes/sum_percentages_by_title_rating_stars?title=No%20Loan%20Again,%20Naturally&rating=6.8&min_stars=5&max_stars=10",
    "/v1/simpson_episodes/count_distinct_episodes_by_stars?min_stars=8",
    "/v1/simpson_episodes/top_voted_episode?limit=1",
    "/v1/simpson_episodes/count_episodes_by_keyword?keyword=2d%20animation",
    "/v1/simpson_episodes/award_organization?award_id=328",
    "/v1/simpson_episodes/count_awards_by_year?year=2009",
    "/v1/simpson_episodes/count_awards_by_category?award_category=Primetime%20Emmy",
    "/v1/simpson_episodes/birth_names_by_role_and_height?role=co-executive%20producer&height_meters=1.60",
    "/v1/simpson_episodes/percentage_nominees_by_country?birth_country=USA&result=Nominee",
    "/v1/simpson_episodes/percentage_winners_by_height?height_meters=1.75&result=Winner",
    "/v1/simpson_episodes/episode_titles_by_stars?stars=2",
    "/v1/simpson_episodes/episode_stars_by_title?title=How%20the%20Test%20Was%20Won",
    "/v1/simpson_episodes/count_episodes_by_year_stars?air_year=2008&min_stars=5",
    "/v1/simpson_episodes/award_years_by_character?character=Mr.%20Burns",
    "/v1/simpson_episodes/award_categories_by_character?character=Lenny",
    "/v1/simpson_episodes/keywords_by_air_year?air_year=2008",
    "/v1/simpson_episodes/count_episodes_by_year_max_stars?air_year=2009&max_stars=8",
    "/v1/simpson_episodes/birth_names_by_role_country?role=director&birth_country=South%20Korea",
    "/v1/simpson_episodes/count_awards_by_year_result?year=2009&result=Winner",
    "/v1/simpson_episodes/count_episodes_below_rating?rating=7",
    "/v1/simpson_episodes/get_names_by_birth_region_country?birth_region=California&birth_country=USA",
    "/v1/simpson_episodes/count_credited_roles_in_episodes?episode_id_1=S20-E5&episode_id_2=S20-E6&episode_id_3=S20-E7&episode_id_4=S20-E8&episode_id_5=S20-E9&episode_id_6=S20-E10&credited=true&role=casting",
    "/v1/simpson_episodes/get_episode_titles_by_award?organization=Primetime%20Emmy%20Awards&year=2009&result=Winner",
    "/v1/simpson_episodes/get_distinct_episode_ids_by_air_date_stars?air_date_year=2008&stars=5",
    "/v1/simpson_episodes/get_top_category_by_person_credited?person=Carlton%20Batten&credited=true",
    "/v1/simpson_episodes/get_vote_percentages_by_air_date_votes?air_date_year=2008&min_votes=950&max_votes=960",
    "/v1/simpson_episodes/get_episode_ids_titles_by_credited_person_role?credited=true&person=Bonita%20Pietila&role=casting",
    "/v1/simpson_episodes/count_awards_by_organization_result_percent?organization=Annie%20Awards&result=Nominee&percent=6",
    "/v1/simpson_episodes/distinct_episode_titles?min_votes=1000&award_category=WGA%20Award%20(TV)&result=Nominee",
    "/v1/simpson_episodes/persons_by_episode_title_role_credited_category?title=How%20the%20Test%20Was%20Won&role=additional%20timer&credited=true&category=Animation%20Department",
    "/v1/simpson_episodes/stars_by_air_date_month_year?air_date_month_year=2008-11",
    "/v1/simpson_episodes/episode_titles_by_air_date_award_category_stars_result?air_date=2009-04-19&award_category=Prism%20Award&stars=5&result=Nominee",
    "/v1/simpson_episodes/top_award_category_by_result_votes?result=Nominee",
    "/v1/simpson_episodes/count_episodes_by_credited_person_air_date_year_role?credited=true&person=Sam%20Im&air_date_year=2009&role=additional%20timer",
    "/v1/simpson_episodes/distinct_episode_titles_above_star_threshold",
    "/v1/simpson_episodes/percentage_difference_nominees_winners?title=Gone%20Maggie%20Gone&year=2009"
]
