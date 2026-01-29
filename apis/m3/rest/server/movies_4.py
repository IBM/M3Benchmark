from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/movies_4/movies_4.sqlite')
cursor = conn.cursor()

# Endpoint to get the company names of production companies for a given movie title
@app.get("/v1/movies_4/production_companies_by_movie_title", operation_id="get_production_companies_by_movie_title", summary="Retrieves the names of production companies associated with a specific movie. The operation requires the title of the movie as input and returns a list of company names that have contributed to the production of the movie.")
async def get_production_companies_by_movie_title(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T1.company_name FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T3.title = ?", (movie_title,))
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": [row[0] for row in result]}

# Endpoint to get the count of production companies for a given movie title
@app.get("/v1/movies_4/count_production_companies_by_movie_title", operation_id="get_count_production_companies_by_movie_title", summary="Retrieves the total number of production companies associated with a specific movie, based on the provided movie title.")
async def get_count_production_companies_by_movie_title(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT COUNT(CNAME) FROM ( SELECT T1.company_name AS CNAME FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T3.title = ? )", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the movie titles produced by a given production company
@app.get("/v1/movies_4/movies_by_production_company", operation_id="get_movies_by_production_company", summary="Retrieve a list of movie titles produced by the specified production company. The operation fetches the movie titles from the database by joining the production_company, movie_company, and movie tables using the provided company name.")
async def get_movies_by_production_company(company_name: str = Query(..., description="Name of the production company")):
    cursor.execute("SELECT T3.title FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T1.company_name = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the latest movie title produced by a given production company
@app.get("/v1/movies_4/latest_movie_by_production_company", operation_id="get_latest_movie_by_production_company", summary="Retrieves the title of the most recently released movie produced by the specified production company.")
async def get_latest_movie_by_production_company(company_name: str = Query(..., description="Name of the production company")):
    cursor.execute("SELECT T3.title FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T1.company_name = ? ORDER BY T3.release_date DESC LIMIT 1", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"latest_movie": []}
    return {"latest_movie": result[0]}

# Endpoint to get the names of people with a specific job in a given movie
@app.get("/v1/movies_4/crew_members_by_movie_and_job", operation_id="get_crew_members_by_movie_and_job", summary="Retrieve the names of crew members who hold a specific job in a given movie. The operation filters crew members by their job title and the movie's title, providing a targeted list of names.")
async def get_crew_members_by_movie_and_job(movie_title: str = Query(..., description="Title of the movie"), job: str = Query(..., description="Job title")):
    cursor.execute("SELECT T3.person_name FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T1.title LIKE ? AND T2.job = ?", (movie_title, job))
    result = cursor.fetchall()
    if not result:
        return {"person_names": []}
    return {"person_names": [row[0] for row in result]}

# Endpoint to get the jobs of a specific person in a given movie
@app.get("/v1/movies_4/jobs_by_person_and_movie", operation_id="get_jobs_by_person_and_movie", summary="Retrieve the job roles of a specific individual in a given movie. This operation requires the movie title and the person's name as input parameters. The job roles are fetched by matching the provided movie title and person's name with the corresponding entries in the movie and person tables, respectively. The result is a list of job roles that the specified person has performed in the mentioned movie.")
async def get_jobs_by_person_and_movie(movie_title: str = Query(..., description="Title of the movie"), person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T2.job FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T1.title LIKE ? AND T3.person_name = ?", (movie_title, person_name))
    result = cursor.fetchall()
    if not result:
        return {"jobs": []}
    return {"jobs": [row[0] for row in result]}

# Endpoint to get the names of all crew members in a given movie
@app.get("/v1/movies_4/crew_members_by_movie", operation_id="get_crew_members_by_movie", summary="Retrieve the names of all crew members associated with a specific movie. The operation filters crew members based on the provided movie title.")
async def get_crew_members_by_movie(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T3.person_name FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T1.title LIKE ?", (movie_title,))
    result = cursor.fetchall()
    if not result:
        return {"person_names": []}
    return {"person_names": [row[0] for row in result]}

# Endpoint to get the count of people with a specific job in a given movie
@app.get("/v1/movies_4/count_crew_members_by_movie_and_job", operation_id="get_count_crew_members_by_movie_and_job", summary="Retrieve the number of individuals with a specific job role in a given movie. This operation requires the title of the movie and the job title as input parameters. The result is a count of individuals who hold the specified job in the provided movie.")
async def get_count_crew_members_by_movie_and_job(movie_title: str = Query(..., description="Title of the movie"), job: str = Query(..., description="Job title")):
    cursor.execute("SELECT COUNT(T3.person_id) FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T1.title LIKE ? AND T2.job = ?", (movie_title, job))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies a specific person has worked on with a specific job
@app.get("/v1/movies_4/count_movies_by_person_and_job", operation_id="get_count_movies_by_person_and_job", summary="Retrieves the total number of movies a particular individual has contributed to in a specific role. The operation requires the individual's name and their job title as input parameters.")
async def get_count_movies_by_person_and_job(person_name: str = Query(..., description="Name of the person"), job: str = Query(..., description="Job title")):
    cursor.execute("SELECT COUNT(T2.movie_id) FROM person AS T1 INNER JOIN movie_crew AS T2 ON T1.person_id = T2.person_id WHERE T1.person_name = ? AND T2.job = ?", (person_name, job))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest-rated movie title directed by a specific person
@app.get("/v1/movies_4/highest_rated_movie_by_director", operation_id="get_highest_rated_movie_by_director", summary="Retrieves the title of the highest-rated movie directed by a specified individual, based on their name and job title. The operation filters movies by the provided job title and name, then sorts them by their average rating to determine the top-rated film.")
async def get_highest_rated_movie_by_director(person_name: str = Query(..., description="Name of the person"), job: str = Query(..., description="Job title of the person")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ? AND T2.job = ? ORDER BY T1.vote_average DESC LIMIT 1", (person_name, job))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the latest movie release date by a specific person
@app.get("/v1/movies_4/latest_movie_release_date_by_person", operation_id="get_latest_movie_release_date_by_person", summary="Retrieves the most recent release date of a movie associated with a specific person. The operation filters movies based on the provided person's name and returns the latest release date.")
async def get_latest_movie_release_date_by_person(person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.release_date FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ? ORDER BY T1.release_date DESC LIMIT 1", (person_name,))
    result = cursor.fetchone()
    if not result:
        return {"release_date": []}
    return {"release_date": result[0]}

# Endpoint to get the percentage of movies with a vote average above a certain threshold directed by a specific person
@app.get("/v1/movies_4/percentage_movies_above_vote_average", operation_id="get_percentage_movies_above_vote_average", summary="Retrieve the percentage of movies directed by a specific person, in a certain role, that have a vote average surpassing a given threshold.")
async def get_percentage_movies_above_vote_average(vote_average: float = Query(..., description="Vote average threshold"), person_name: str = Query(..., description="Name of the person"), job: str = Query(..., description="Job title of the person")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.vote_average > ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.vote_average) FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ? AND T2.job = ?", (vote_average, person_name, job))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average revenue of movies directed by a specific person
@app.get("/v1/movies_4/average_revenue_by_director", operation_id="get_average_revenue_by_director", summary="Retrieves the average revenue of movies directed by a specific individual. The operation requires the name of the person and their job title to filter the results. The average revenue is calculated by summing the total revenue of all movies directed by the specified person and then dividing it by the total number of movies they have directed.")
async def get_average_revenue_by_director(person_name: str = Query(..., description="Name of the person"), job: str = Query(..., description="Job title of the person")):
    cursor.execute("SELECT CAST(SUM(T1.revenue) AS REAL) / COUNT(T1.movie_id) FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ? AND T2.job = ?", (person_name, job))
    result = cursor.fetchone()
    if not result:
        return {"average_revenue": []}
    return {"average_revenue": result[0]}

# Endpoint to get the movie title based on revenue
@app.get("/v1/movies_4/movie_title_by_revenue", operation_id="get_movie_title_by_revenue", summary="Retrieves the title of a movie that matches the specified revenue. The revenue parameter is used to filter the movie records and return the title of the corresponding movie.")
async def get_movie_title_by_revenue(revenue: int = Query(..., description="Revenue of the movie")):
    cursor.execute("SELECT title FROM movie WHERE revenue = ?", (revenue,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the job title of a person in a specific movie
@app.get("/v1/movies_4/job_title_by_person_and_movie", operation_id="get_job_title_by_person_and_movie", summary="Retrieves the job title of a specific person in a given movie. The operation requires the person's name and the movie title as input parameters to accurately identify the job title.")
async def get_job_title_by_person_and_movie(person_name: str = Query(..., description="Name of the person"), title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.job FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ? AND T1.title = ?", (person_name, title))
    result = cursor.fetchone()
    if not result:
        return {"job": []}
    return {"job": result[0]}

# Endpoint to get the count of keywords for a specific movie
@app.get("/v1/movies_4/count_keywords_by_movie", operation_id="get_count_keywords_by_movie", summary="Retrieves the total number of keywords associated with a specific movie. The movie is identified by its title.")
async def get_count_keywords_by_movie(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT COUNT(T2.keyword_id) FROM movie AS T1 INNER JOIN movie_keywords AS T2 ON T1.movie_id = T2.movie_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the director's name for a specific movie
@app.get("/v1/movies_4/director_name_by_movie", operation_id="get_director_name_by_movie", summary="Retrieves the name of the director associated with a specific movie. The movie is identified by its title, and the job title of the person is used to ensure the correct individual is selected. This operation returns the name of the person who holds the specified job title for the provided movie.")
async def get_director_name_by_movie(title: str = Query(..., description="Title of the movie"), job: str = Query(..., description="Job title of the person")):
    cursor.execute("SELECT T3.person_name FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T1.title = ? AND T2.job = ?", (title, job))
    result = cursor.fetchone()
    if not result:
        return {"person_name": []}
    return {"person_name": result[0]}

# Endpoint to get the count of movies produced by a specific company
@app.get("/v1/movies_4/count_movies_by_company", operation_id="get_count_movies_by_company", summary="Retrieves the total number of movies produced by a specified production company.")
async def get_count_movies_by_company(company_name: str = Query(..., description="Name of the production company")):
    cursor.execute("SELECT COUNT(T2.movie_id) FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id WHERE T1.company_name = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cast members based on movie title and gender
@app.get("/v1/movies_4/count_cast_members_by_title_and_gender", operation_id="get_count_cast_members", summary="Retrieves the total number of cast members in a specific movie, filtered by a given gender. The operation requires the movie title and the desired gender as input parameters.")
async def get_count_cast_members(title: str = Query(..., description="Title of the movie"), gender: str = Query(..., description="Gender of the cast member")):
    cursor.execute("SELECT COUNT(*) FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN gender AS T3 ON T2.gender_id = T3.gender_id WHERE T1.title = ? AND T3.gender = ?", (title, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common keyword
@app.get("/v1/movies_4/most_common_keyword", operation_id="get_most_common_keyword", summary="Retrieves the keyword that is most frequently used in movies. This operation returns the keyword with the highest occurrence in the movie database, providing insights into the most prevalent themes or topics in the movies.")
async def get_most_common_keyword():
    cursor.execute("SELECT T1.keyword_name FROM keyword AS T1 INNER JOIN movie_keywords AS T2 ON T1.keyword_id = T2.keyword_id GROUP BY T1.keyword_name ORDER BY COUNT(T1.keyword_name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"keyword": []}
    return {"keyword": result[0]}

# Endpoint to get the count of crew members based on movie title and job
@app.get("/v1/movies_4/count_crew_members_by_title_and_job", operation_id="get_count_crew_members", summary="Retrieves the total number of crew members in a specific movie, filtered by a given job role. The operation requires the title of the movie and the job role as input parameters.")
async def get_count_crew_members(title: str = Query(..., description="Title of the movie"), job: str = Query(..., description="Job of the crew member")):
    cursor.execute("SELECT COUNT(T2.person_id) FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id WHERE T1.title = ? AND T2.job = ?", (title, job))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of movies by keyword
@app.get("/v1/movies_4/count_movies_by_keyword", operation_id="get_count_movies_by_keyword", summary="Retrieves the total number of movies linked to a given keyword. The keyword is specified as an input parameter.")
async def get_count_movies_by_keyword(keyword_name: str = Query(..., description="Name of the keyword")):
    cursor.execute("SELECT COUNT(T2.movie_id) FROM keyword AS T1 INNER JOIN movie_keywords AS T2 ON T1.keyword_id = T2.keyword_id WHERE keyword_name = ?", (keyword_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most popular movie by production company
@app.get("/v1/movies_4/most_popular_movie_by_company", operation_id="get_most_popular_movie_by_company", summary="Retrieves the title of the most popular movie produced by the specified production company. The popularity of a movie is determined by its ranking in the descending order of popularity scores. The production company is identified by its name.")
async def get_most_popular_movie_by_company(company_name: str = Query(..., description="Name of the production company")):
    cursor.execute("SELECT T3.title FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T1.company_name = ? ORDER BY T3.popularity DESC LIMIT 1", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the department name based on person name and movie title
@app.get("/v1/movies_4/department_name_by_person_and_title", operation_id="get_department_name", summary="Retrieves the department name of a specific person involved in a given movie. The operation requires the person's name and the movie title as input parameters to accurately identify the department.")
async def get_department_name(person_name: str = Query(..., description="Name of the person"), title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T4.department_name FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id INNER JOIN department AS T4 ON T2.department_id = T4.department_id WHERE T3.person_name = ? AND T1.title = ?", (person_name, title))
    result = cursor.fetchone()
    if not result:
        return {"department": []}
    return {"department": result[0]}

# Endpoint to get the average budget of movies directed by a specific person
@app.get("/v1/movies_4/average_budget_by_director", operation_id="get_average_budget", summary="Retrieves the average budget of movies directed by a specific person. This operation calculates the total budget of all movies directed by the specified person and divides it by the total number of movies they have directed. The input parameters include the name of the director and their job title to ensure accurate results.")
async def get_average_budget(person_name: str = Query(..., description="Name of the director"), job: str = Query(..., description="Job of the person (Director)")):
    cursor.execute("SELECT CAST(SUM(T1.budget) AS REAL) / COUNT(T1.movie_id) FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ? AND T2.job = ?", (person_name, job))
    result = cursor.fetchone()
    if not result:
        return {"average_budget": []}
    return {"average_budget": result[0]}

# Endpoint to get the percentage of male cast members in a movie
@app.get("/v1/movies_4/percentage_male_cast_by_title", operation_id="get_percentage_male_cast", summary="Retrieves the percentage of male cast members in a movie with a specified title. The calculation is based on the total number of cast members in the movie. The title of the movie is provided as an input parameter.")
async def get_percentage_male_cast(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.gender = 'Male' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T3.gender) FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN gender AS T3 ON T2.gender_id = T3.gender_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the movie with the highest budget
@app.get("/v1/movies_4/highest_budget_movie", operation_id="get_highest_budget_movie", summary="Retrieves the title of the movie with the highest budget from the database.")
async def get_highest_budget_movie():
    cursor.execute("SELECT title FROM movie ORDER BY budget DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of movies with revenue greater than a specified amount
@app.get("/v1/movies_4/count_movies_by_revenue", operation_id="get_count_movies_by_revenue", summary="Retrieves the total number of movies that have a revenue greater than the specified minimum amount.")
async def get_count_movies_by_revenue(min_revenue: int = Query(..., description="Minimum revenue amount")):
    cursor.execute("SELECT COUNT(movie_id) FROM movie WHERE revenue > ?", (min_revenue,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the earliest release date of movies with a specific status
@app.get("/v1/movies_4/earliest_release_date_by_status", operation_id="get_earliest_release_date_by_status", summary="Retrieves the earliest release date among movies that share a specified status. The status is provided as an input parameter.")
async def get_earliest_release_date_by_status(movie_status: str = Query(..., description="Status of the movie")):
    cursor.execute("SELECT MIN(release_date) FROM movie WHERE movie_status = ?", (movie_status,))
    result = cursor.fetchone()
    if not result:
        return {"release_date": []}
    return {"release_date": result[0]}

# Endpoint to get the count of persons with a specific name
@app.get("/v1/movies_4/count_persons_by_name", operation_id="get_count_persons_by_name", summary="Retrieves the total count of individuals with a specified name from the database.")
async def get_count_persons_by_name(person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT COUNT(person_id) FROM person WHERE person_name = ?", (person_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most popular movie title
@app.get("/v1/movies_4/most_popular_movie", operation_id="get_most_popular_movie", summary="Retrieves the title of the most popular movie, as determined by the highest popularity score.")
async def get_most_popular_movie():
    cursor.execute("SELECT title FROM movie ORDER BY popularity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the name of a person by their ID
@app.get("/v1/movies_4/person_name_by_id", operation_id="get_person_name_by_id", summary="Retrieves the name of a specific person identified by their unique ID. The ID is used to locate the person's record in the database and return the corresponding name.")
async def get_person_name_by_id(person_id: int = Query(..., description="ID of the person")):
    cursor.execute("SELECT person_name FROM person WHERE person_id = ?", (person_id,))
    result = cursor.fetchone()
    if not result:
        return {"person_name": []}
    return {"person_name": result[0]}

# Endpoint to get the production company with the most movies
@app.get("/v1/movies_4/top_production_company", operation_id="get_top_production_company", summary="Retrieves the name of the production company that has produced the highest number of movies. The operation calculates the count of movies produced by each company and returns the top one.")
async def get_top_production_company():
    cursor.execute("SELECT T1.company_name FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id GROUP BY T1.company_id ORDER BY COUNT(T2.movie_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get distinct person names who played a specific character in movies with titles starting with a specific phrase
@app.get("/v1/movies_4/person_names_by_character_and_title", operation_id="get_person_names_by_character_and_title", summary="Retrieve a list of unique person names who portrayed a specific character in movies with titles beginning with a given phrase. This operation filters the movie database based on the provided character name and the starting phrase of the movie title.")
async def get_person_names_by_character_and_title(character_name: str = Query(..., description="Name of the character"), title_start: str = Query(..., description="Starting phrase of the movie title")):
    cursor.execute("SELECT DISTINCT T3.person_name FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T2.character_name = ? AND T1.title LIKE ?", (character_name, title_start + '%'))
    result = cursor.fetchall()
    if not result:
        return {"person_names": []}
    return {"person_names": [row[0] for row in result]}

# Endpoint to get the production company with the highest total revenue
@app.get("/v1/movies_4/top_revenue_company", operation_id="get_top_revenue_company", summary="Retrieves the name of the production company that has generated the highest total revenue from its movies. This operation calculates the total revenue for each production company by summing the revenue of all movies produced by each company. The company with the highest total revenue is then returned.")
async def get_top_revenue_company():
    cursor.execute("SELECT T1.company_name FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id GROUP BY T1.company_id ORDER BY SUM(T3.revenue) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the count of a specific gender in a movie
@app.get("/v1/movies_4/count_gender_in_movie", operation_id="get_count_gender_in_movie", summary="Retrieves the number of individuals of a specified gender appearing in a movie, based on the movie's title and the desired gender.")
async def get_count_gender_in_movie(title: str = Query(..., description="Title of the movie"), gender: str = Query(..., description="Gender to count")):
    cursor.execute("SELECT COUNT(T3.gender) FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN gender AS T3 ON T2.gender_id = T3.gender_id WHERE T1.title = ? AND T3.gender = ?", (title, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get production companies with more than a specified number of movies
@app.get("/v1/movies_4/production_companies_with_more_than_movies", operation_id="get_production_companies_with_more_than_movies", summary="Retrieve a list of production companies that have produced more than the specified number of movies. The operation filters the production companies based on the minimum count of movies they have produced, providing a targeted list of companies that meet the criteria.")
async def get_production_companies_with_more_than_movies(movie_count: int = Query(..., description="Minimum number of movies produced by the company")):
    cursor.execute("SELECT T1.company_name FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id GROUP BY T1.company_id HAVING COUNT(T2.movie_id) > ?", (movie_count,))
    result = cursor.fetchall()
    if not result:
        return {"companies": []}
    return {"companies": [row[0] for row in result]}

# Endpoint to get the count of movies a person has acted in
@app.get("/v1/movies_4/count_movies_person_acted_in", operation_id="get_count_movies_person_acted_in", summary="Retrieves the total number of movies in which a specified person has acted. The count is determined by matching the provided person's name with the records in the database.")
async def get_count_movies_person_acted_in(person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT COUNT(T2.movie_id) FROM person AS T1 INNER JOIN movie_cast AS T2 ON T1.person_id = T2.person_id WHERE T1.person_name = ?", (person_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latest movie title a person has acted in
@app.get("/v1/movies_4/latest_movie_person_acted_in", operation_id="get_latest_movie_person_acted_in", summary="Retrieves the title of the most recent movie in which the specified person has acted. The person is identified by their name, and the movie's release date is used to determine recency.")
async def get_latest_movie_person_acted_in(person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ? ORDER BY T1.release_date DESC LIMIT 1", (person_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of movies a person has acted in a specific year
@app.get("/v1/movies_4/count_movies_person_acted_in_year", operation_id="get_count_movies_person_acted_in_year", summary="Retrieves the total number of movies a specified person has acted in during a given year. The count is determined by matching the person's name and the year of movie release.")
async def get_count_movies_person_acted_in_year(person_name: str = Query(..., description="Name of the person"), year: int = Query(..., description="Year of the movie release")):
    cursor.execute("SELECT COUNT(T1.movie_id) FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ? AND CAST(STRFTIME('%Y', T1.release_date) AS INT) = ?", (person_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the earliest movie title in a specific genre
@app.get("/v1/movies_4/earliest_movie_in_genre", operation_id="get_earliest_movie_in_genre", summary="Retrieves the title of the earliest released movie in a specified genre. The genre is determined by the provided genre name.")
async def get_earliest_movie_in_genre(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id WHERE T3.genre_name = ? ORDER BY T1.release_date LIMIT 1", (genre_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of movies in a specific genre
@app.get("/v1/movies_4/count_movies_in_genre", operation_id="get_count_movies_in_genre", summary="Retrieves the total number of movies that belong to a specified genre. The genre is identified by its name.")
async def get_count_movies_in_genre(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT COUNT(T1.movie_id) FROM movie_genres AS T1 INNER JOIN genre AS T2 ON T1.genre_id = T2.genre_id WHERE T2.genre_name = ?", (genre_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the person IDs for a specific job in a movie
@app.get("/v1/movies_4/person_ids_for_job_in_movie", operation_id="get_person_ids_for_job_in_movie", summary="Retrieves the unique identifiers of individuals who held a specific job in a movie. The operation requires the movie's ID and the job title as input parameters. The result is a list of person IDs who were involved in the specified job for the given movie.")
async def get_person_ids_for_job_in_movie(movie_id: int = Query(..., description="ID of the movie"), job: str = Query(..., description="Job title")):
    cursor.execute("SELECT person_id FROM movie_crew WHERE movie_id = ? AND job = ?", (movie_id, job))
    result = cursor.fetchall()
    if not result:
        return {"person_ids": []}
    return {"person_ids": [row[0] for row in result]}

# Endpoint to get the count of a specific job in a movie
@app.get("/v1/movies_4/count_job_in_movie", operation_id="get_count_job_in_movie", summary="Retrieves the total number of occurrences of a specific job title within a movie, based on the provided movie ID and job title.")
async def get_count_job_in_movie(movie_id: int = Query(..., description="ID of the movie"), job: str = Query(..., description="Job title")):
    cursor.execute("SELECT COUNT(movie_id) FROM movie_crew WHERE movie_id = ? AND job = ?", (movie_id, job))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct jobs in a specific department for a movie
@app.get("/v1/movies_4/count_distinct_jobs_in_department", operation_id="get_count_distinct_jobs_in_department", summary="Retrieves the number of unique job roles within a specific department for a given movie. This operation requires the movie's ID and the department's ID as input parameters.")
async def get_count_distinct_jobs_in_department(movie_id: int = Query(..., description="ID of the movie"), department_id: int = Query(..., description="ID of the department")):
    cursor.execute("SELECT COUNT(DISTINCT job) FROM movie_crew WHERE movie_id = ? AND department_id = ?", (movie_id, department_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get cast members of a movie within a specific cast order range
@app.get("/v1/movies_4/cast_members_by_movie_and_order", operation_id="get_cast_members", summary="Retrieves the cast members of a specific movie, filtered by a range of cast order values. The cast order represents the order in which the cast members appear in the movie credits. The response includes the ID of each cast member and the name of the character they portray.")
async def get_cast_members(movie_id: int = Query(..., description="ID of the movie"), min_cast_order: int = Query(..., description="Minimum cast order"), max_cast_order: int = Query(..., description="Maximum cast order")):
    cursor.execute("SELECT person_id, character_name FROM movie_cast WHERE movie_id = ? AND cast_order BETWEEN ? AND ?", (movie_id, min_cast_order, max_cast_order))
    result = cursor.fetchall()
    if not result:
        return {"cast_members": []}
    return {"cast_members": result}

# Endpoint to get movies and characters of a person
@app.get("/v1/movies_4/movies_and_characters_by_person", operation_id="get_movies_and_characters_by_person", summary="Retrieves a list of movies and corresponding character roles played by a specified person. The person's name is used to filter the results.")
async def get_movies_and_characters_by_person(person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T2.movie_id, T2.character_name FROM person AS T1 INNER JOIN movie_cast AS T2 ON T1.person_id = T2.person_id WHERE T1.person_name = ?", (person_name,))
    result = cursor.fetchall()
    if not result:
        return {"movies_and_characters": []}
    return {"movies_and_characters": result}

# Endpoint to get cast members of a movie by gender
@app.get("/v1/movies_4/cast_members_by_movie_and_gender", operation_id="get_cast_members_by_gender", summary="Retrieves the names of cast members from a specific movie based on their gender. The operation requires the movie's unique identifier and the desired gender as input parameters. The response includes a list of cast members who meet the specified criteria.")
async def get_cast_members_by_gender(movie_id: int = Query(..., description="ID of the movie"), gender: str = Query(..., description="Gender of the cast members")):
    cursor.execute("SELECT T2.person_name FROM movie_cast AS T1 INNER JOIN person AS T2 ON T1.person_id = T2.person_id INNER JOIN gender AS T3 ON T1.gender_id = T3.gender_id WHERE T1.movie_id = ? AND T3.gender = ?", (movie_id, gender))
    result = cursor.fetchall()
    if not result:
        return {"cast_members": []}
    return {"cast_members": result}

# Endpoint to get movie titles a person has acted in
@app.get("/v1/movies_4/movie_titles_by_person", operation_id="get_movie_titles_by_person", summary="Retrieves a list of movie titles in which a specified person has acted. The operation filters the movies based on the provided person's name and returns the corresponding titles.")
async def get_movie_titles_by_person(person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ?", (person_name,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": result}

# Endpoint to get cast members of movies directed by a specific job role within a date range
@app.get("/v1/movies_4/cast_members_by_director_and_date_range", operation_id="get_cast_members_by_director_and_date_range", summary="Retrieve the names of cast members who have worked in movies directed by a specific job role within a specified date range. The operation filters movies based on the provided job role of the director and the release date range. The results include the names of cast members who have acted in these filtered movies.")
async def get_cast_members_by_director_and_date_range(job: str = Query(..., description="Job role of the director"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.person_name FROM movie_cast AS T1 INNER JOIN person AS T2 ON T1.person_id = T2.person_id INNER JOIN movie AS T3 ON T1.movie_id = T3.movie_id INNER JOIN movie_crew AS T4 ON T1.movie_id = T4.movie_id WHERE T4.job = ? AND T3.release_date BETWEEN ? AND ?", (job, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"cast_members": []}
    return {"cast_members": result}

# Endpoint to get the count of movies a person has acted in within a date range
@app.get("/v1/movies_4/movie_count_by_person_and_date_range", operation_id="get_movie_count_by_person_and_date_range", summary="Retrieve the total number of movies a specified person has acted in between a given start and end date.")
async def get_movie_count_by_person_and_date_range(person_name: str = Query(..., description="Name of the person"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.movie_id) FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T3.person_name = ? AND T1.release_date BETWEEN ? AND ?", (person_name, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get movie titles of a specific genre with a minimum vote average
@app.get("/v1/movies_4/movie_titles_by_genre_and_vote_average", operation_id="get_movie_titles_by_genre_and_vote_average", summary="Retrieves a list of movie titles that belong to a specified genre and have a vote average greater than a provided minimum threshold. The genre is identified by its name, and the minimum vote average is a numerical value.")
async def get_movie_titles_by_genre_and_vote_average(genre_name: str = Query(..., description="Name of the genre"), min_vote_average: float = Query(..., description="Minimum vote average")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id WHERE T3.genre_name = ? AND vote_average > ?", (genre_name, min_vote_average))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": result}

# Endpoint to get genre names and movie popularity within a revenue and date range
@app.get("/v1/movies_4/genre_popularity_by_revenue_and_date_range", operation_id="get_genre_popularity_by_revenue_and_date_range", summary="Retrieves the popularity of movies and their respective genres based on a specified revenue range and date interval. The operation filters movies by their revenue, which must exceed a provided minimum value, and their release dates, which must fall within a given start and end date. The result is a list of genre names and their corresponding movie popularity scores.")
async def get_genre_popularity_by_revenue_and_date_range(min_revenue: int = Query(..., description="Minimum revenue"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T3.genre_name, T1.popularity FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id WHERE T1.revenue > ? AND T1.release_date BETWEEN ? AND ?", (min_revenue, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"genre_popularity": []}
    return {"genre_popularity": result}

# Endpoint to get the count of movies produced in a country within a revenue, popularity, and date range
@app.get("/v1/movies_4/movie_count_by_revenue_popularity_and_date_range", operation_id="get_movie_count_by_revenue_popularity_and_date_range", summary="Retrieve the number of movies produced in a specific country that meet the provided revenue, popularity, and release date criteria. The minimum revenue and popularity thresholds, as well as the start and end dates of the release period, are required as input parameters.")
async def get_movie_count_by_revenue_popularity_and_date_range(min_revenue: int = Query(..., description="Minimum revenue"), min_popularity: float = Query(..., description="Minimum popularity"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T2.movie_id) FROM movie AS T1 INNER JOIN production_country AS T2 ON T1.movie_id = T2.movie_id WHERE T1.revenue > ? AND T1.popularity >= ? AND T1.release_date BETWEEN ? AND ?", (min_revenue, min_popularity, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get movies in a specific language released within a date range
@app.get("/v1/movies_4/movies_by_language_and_date_range", operation_id="get_movies_by_language_and_date_range", summary="Retrieve a list of movies released within a specified date range, filtered by a selected language. The language is identified by its name, and the date range is defined by a start and end date in 'YYYY-MM-DD' format.")
async def get_movies_by_language_and_date_range(language_name: str = Query(..., description="Language name"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id WHERE T3.language_name = ? AND T1.release_date BETWEEN ? AND ?", (language_name, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": [row[0] for row in result]}

# Endpoint to get the average revenue of movies from a specific country in a specific year
@app.get("/v1/movies_4/average_revenue_by_country_and_year", operation_id="get_average_revenue_by_country_and_year", summary="Retrieves the average revenue of movies from a specified country in a given year. The operation calculates the average revenue by joining the movie, production_country, and country tables. The result is filtered by the provided country name and year.")
async def get_average_revenue_by_country_and_year(country_name: str = Query(..., description="Country name"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT AVG(T1.revenue) FROM movie AS T1 INNER JOIN production_COUNTry AS T2 ON T1.movie_id = T2.movie_id INNER JOIN COUNTry AS T3 ON T2.COUNTry_id = T3.COUNTry_id WHERE T3.COUNTry_name = ? AND CAST(STRFTIME('%Y', T1.release_date) AS INT) = ?", (country_name, year))
    result = cursor.fetchone()
    if not result:
        return {"average_revenue": []}
    return {"average_revenue": result[0]}

# Endpoint to get the difference in average revenue between two countries in a specific year
@app.get("/v1/movies_4/revenue_difference_by_countries_and_year", operation_id="get_revenue_difference_by_countries_and_year", summary="Retrieve the difference in average revenue between two specified countries for a given year. This operation calculates the average revenue for each of the two countries in the provided year and returns the difference between these averages.")
async def get_revenue_difference_by_countries_and_year(country_name_1: str = Query(..., description="First country name"), country_name_2: str = Query(..., description="Second country name"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT AVG(CASE WHEN T3.COUNTry_name = ? THEN T1.revenue END) - AVG(CASE WHEN T3.COUNTry_name = ? THEN T1.revenue END) AS CALCULATE FROM movie AS T1 INNER JOIN production_COUNTry AS T2 ON T1.movie_id = T2.movie_id INNER JOIN COUNTry AS T3 ON T2.COUNTry_id = T3.COUNTry_id WHERE CAST(STRFTIME('%Y', T1.release_date) AS INT) = ?", (country_name_1, country_name_2, year))
    result = cursor.fetchone()
    if not result:
        return {"revenue_difference": []}
    return {"revenue_difference": result[0]}

# Endpoint to get the percentage of movies of a specific genre in a specific country within a date range
@app.get("/v1/movies_4/genre_percentage_by_country_and_date_range", operation_id="get_genre_percentage_by_country_and_date_range", summary="Retrieve the percentage of movies belonging to a specified genre in a given country, within a defined date range. The calculation is based on the total count of movies in the specified genre and the total count of movies in the country during the selected period.")
async def get_genre_percentage_by_country_and_date_range(genre_name: str = Query(..., description="Genre name"), country_name: str = Query(..., description="Country name"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T4.genre_name = ? THEN T1.movie_id ELSE NULL END) AS REAL) * 100 / COUNT(T1.movie_id) FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN production_COUNTry AS T3 ON T1.movie_id = T3.movie_id INNER JOIN genre AS T4 ON T2.genre_id = T4.genre_id INNER JOIN COUNTry AS T5 ON T3.COUNTry_id = T5.COUNTry_id WHERE T5.COUNTry_name = ? AND T1.release_date BETWEEN ? AND ?", (genre_name, country_name, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct person names who played a specific character
@app.get("/v1/movies_4/person_names_by_character", operation_id="get_person_names_by_character", summary="Retrieve a list of unique person names who have portrayed a specified character in movies. The character name is provided as an input parameter.")
async def get_person_names_by_character(character_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT DISTINCT T1.person_name FROM person AS T1 INNER JOIN movie_cast AS T2 ON T1.person_id = T2.person_id WHERE T2.character_name = ?", (character_name,))
    result = cursor.fetchall()
    if not result:
        return {"person_names": []}
    return {"person_names": [row[0] for row in result]}

# Endpoint to get the gender of a specific character
@app.get("/v1/movies_4/gender_by_character", operation_id="get_gender_by_character", summary="Retrieves the gender of a specified character from the movie cast database. The operation requires the character's name as input and returns the corresponding gender.")
async def get_gender_by_character(character_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT T2.gender FROM movie_cast AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.gender_id WHERE T1.character_name = ?", (character_name,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the genres of a specific movie
@app.get("/v1/movies_4/genres_by_movie", operation_id="get_genres_by_movie", summary="Retrieve the genres associated with a specific movie. The operation requires the movie title as input and returns a list of genre names that the movie belongs to.")
async def get_genres_by_movie(movie_title: str = Query(..., description="Movie title")):
    cursor.execute("SELECT T3.genre_name FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id WHERE T1.title = ?", (movie_title,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get keywords for a movie by title
@app.get("/v1/movies_4/keywords_by_movie_title", operation_id="get_keywords_by_movie_title", summary="Retrieve the keywords linked to a specific movie by providing its title. This operation fetches the relevant keywords from the database by matching the given movie title with the corresponding entries in the movie table.")
async def get_keywords_by_movie_title(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T3.keyword_name FROM movie AS T1 INNER JOIN movie_keywords AS T2 ON T1.movie_id = T2.movie_id INNER JOIN keyword AS T3 ON T2.keyword_id = T3.keyword_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get production countries for a movie by title
@app.get("/v1/movies_4/production_countries_by_movie_title", operation_id="get_production_countries_by_movie_title", summary="Retrieve the names of the production countries associated with a specific movie, identified by its title.")
async def get_production_countries_by_movie_title(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T3.COUNTry_name FROM movie AS T1 INNER JOIN production_COUNTry AS T2 ON T1.movie_id = T2.movie_id INNER JOIN COUNTry AS T3 ON T2.COUNTry_id = T3.COUNTry_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get movie titles by keyword
@app.get("/v1/movies_4/movie_titles_by_keyword", operation_id="get_movie_titles_by_keyword", summary="Retrieve the titles of movies that are associated with a given keyword. The keyword is used to filter the movies and return only those that match the specified keyword.")
async def get_movie_titles_by_keyword(keyword_name: str = Query(..., description="Name of the keyword")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_keywords AS T2 ON T1.movie_id = T2.movie_id INNER JOIN keyword AS T3 ON T2.keyword_id = T3.keyword_id WHERE T3.keyword_name = ?", (keyword_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get movie titles by production country with a limit
@app.get("/v1/movies_4/movie_titles_by_production_country", operation_id="get_movie_titles_by_production_country", summary="Retrieve a specified number of movie titles produced in a given country. The operation fetches the titles from the movie database, filtering them based on the provided country name and limiting the results to the specified number.")
async def get_movie_titles_by_production_country(country_name: str = Query(..., description="Name of the production country"), limit: int = Query(..., description="Limit of the number of titles to return")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN production_COUNTry AS T2 ON T1.movie_id = T2.movie_id INNER JOIN COUNTry AS T3 ON T2.COUNTry_id = T3.COUNTry_id WHERE T3.COUNTry_name = ? LIMIT ?", (country_name, limit))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get crew members by movie title with a limit
@app.get("/v1/movies_4/crew_members_by_movie_title", operation_id="get_crew_members_by_movie_title", summary="Retrieve a specified number of crew members' names for a given movie. The operation filters crew members by the provided movie title and returns the requested number of names.")
async def get_crew_members_by_movie_title(title: str = Query(..., description="Title of the movie"), limit: int = Query(..., description="Limit of the number of crew members to return")):
    cursor.execute("SELECT T3.person_name FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T1.title = ? LIMIT ?", (title, limit))
    result = cursor.fetchall()
    if not result:
        return {"crew_members": []}
    return {"crew_members": [row[0] for row in result]}

# Endpoint to get the percentage of movies of a specific genre in a specific country
@app.get("/v1/movies_4/genre_percentage_by_country", operation_id="get_genre_percentage_by_country", summary="Retrieve the percentage of movies from a specific genre that were produced in a given country. This operation calculates the ratio of movies belonging to the input genre and produced in the input country, compared to the total number of movies produced in that country.")
async def get_genre_percentage_by_country(genre_name: str = Query(..., description="Name of the genre"), country_name: str = Query(..., description="Name of the production country")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T4.genre_name = ? THEN T1.movie_id ELSE NULL END) AS REAL) * 100 / COUNT(T1.movie_id) FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN production_COUNTry AS T3 ON T1.movie_id = T3.movie_id INNER JOIN genre AS T4 ON T2.genre_id = T4.genre_id INNER JOIN COUNTry AS T5 ON T3.COUNTry_id = T5.COUNTry_id WHERE T5.COUNTry_name = ?", (genre_name, country_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get gender ratio and count of unspecified genders for a movie by title
@app.get("/v1/movies_4/gender_ratio_and_unspecified_count", operation_id="get_gender_ratio_and_unspecified_count", summary="Retrieves the ratio of two specified genders and the count of an unspecified gender in the cast of a given movie. The operation calculates the ratio by dividing the count of the first gender by the count of the second gender. It also returns the count of the unspecified gender in the cast. This operation is useful for analyzing the gender distribution in the cast of a specific movie.")
async def get_gender_ratio_and_unspecified_count(title: str = Query(..., description="Title of the movie"), gender_1: str = Query(..., description="First gender for ratio calculation"), gender_2: str = Query(..., description="Second gender for ratio calculation"), unspecified_gender: str = Query(..., description="Gender to count as unspecified")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.gender = ? THEN 1 ELSE NULL END) AS REAL) / COUNT(CASE WHEN T3.gender = ? THEN 1 ELSE NULL END) AS RATIO , COUNT(CASE WHEN T3.gender = ? THEN 1 ELSE NULL END) AS UNGENDERS FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN gender AS T3 ON T2.gender_id = T3.gender_id WHERE T1.title = ?", (gender_1, gender_2, unspecified_gender, title))
    result = cursor.fetchone()
    if not result:
        return {"ratio": [], "unspecified_count": []}
    return {"ratio": result[0], "unspecified_count": result[1]}

# Endpoint to get movie titles released before a specific year with a limit
@app.get("/v1/movies_4/movie_titles_before_year", operation_id="get_movie_titles_before_year", summary="Retrieves a list of movie titles released before the specified year, with a limit on the number of titles returned. This operation is useful for obtaining a subset of older movie titles for further analysis or display.")
async def get_movie_titles_before_year(year: int = Query(..., description="Year to filter movies released before"), limit: int = Query(..., description="Limit of the number of titles to return")):
    cursor.execute("SELECT title FROM movie WHERE CAST(STRFTIME('%Y', release_date) AS INT) < ? LIMIT ?", (year, limit))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get keyword IDs for a movie by title
@app.get("/v1/movies_4/keyword_ids_by_movie_title", operation_id="get_keyword_ids_by_movie_title", summary="Retrieves the unique identifiers of keywords linked to a specific movie, based on the movie's title. The input parameter is the title of the movie.")
async def get_keyword_ids_by_movie_title(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.keyword_id FROM movie AS T1 INNER JOIN movie_keywords AS T2 ON T1.movie_id = T2.movie_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"keyword_ids": []}
    return {"keyword_ids": [row[0] for row in result]}

# Endpoint to get the most popular keyword
@app.get("/v1/movies_4/most_popular_keyword", operation_id="get_most_popular_keyword", summary="Retrieves the most frequently occurring keyword in the most popular movies. The keyword is determined by analyzing the popularity of movies associated with each keyword. The popularity of a movie is based on its rating and the number of votes it has received.")
async def get_most_popular_keyword():
    cursor.execute("SELECT T3.keyword_name FROM movie AS T1 INNER JOIN movie_keywords AS T2 ON T1.movie_id = T2.movie_id INNER JOIN keyword AS T3 ON T2.keyword_id = T3.keyword_id ORDER BY T1.popularity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"keyword": []}
    return {"keyword": result[0]}

# Endpoint to get genre IDs for a specific movie title
@app.get("/v1/movies_4/genre_ids_by_movie_title", operation_id="get_genre_ids_by_movie_title", summary="Retrieves the genre IDs associated with a specific movie title. The operation filters the movie_genres table based on the provided movie title, returning the corresponding genre IDs.")
async def get_genre_ids_by_movie_title(title: str = Query(..., description="Title of the movie to filter genre IDs")):
    cursor.execute("SELECT T2.genre_id FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"genre_ids": []}
    return {"genre_ids": [row[0] for row in result]}

# Endpoint to get movie titles based on a specific genre
@app.get("/v1/movies_4/movie_titles_by_genre", operation_id="get_movie_titles_by_genre", summary="Retrieves a list of movie titles that belong to a specified genre. The genre is used as a filter to select the relevant movie titles from the database.")
async def get_movie_titles_by_genre(genre_name: str = Query(..., description="Genre name to filter movie titles")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id WHERE T3.genre_name = ?", (genre_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get movie titles based on status and genre
@app.get("/v1/movies_4/movie_titles_by_status_and_genre", operation_id="get_movie_titles_by_status_and_genre", summary="Retrieves a list of up to five movie titles that match a specified status and genre. The status and genre are provided as input parameters, allowing for targeted search results.")
async def get_movie_titles_by_status_and_genre(movie_status: str = Query(..., description="Status of the movie"), genre_name: str = Query(..., description="Genre name to filter movie titles")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id WHERE T1.movie_status = ? AND T3.genre_name = ? LIMIT 5", (movie_status, genre_name))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the genre with the highest revenue
@app.get("/v1/movies_4/highest_revenue_genre", operation_id="get_highest_revenue_genre", summary="Retrieves the genre that has generated the highest total revenue from all movies. The genre is determined by aggregating the revenue of all movies associated with each genre and selecting the one with the highest sum.")
async def get_highest_revenue_genre():
    cursor.execute("SELECT T3.genre_name FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id ORDER BY T1.revenue LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"genre": []}
    return {"genre": result[0]}

# Endpoint to get genres based on movie runtime
@app.get("/v1/movies_4/genres_by_runtime", operation_id="get_genres_by_runtime", summary="Retrieves the genres of movies that have a specified runtime. The runtime is provided in minutes.")
async def get_genres_by_runtime(runtime: int = Query(..., description="Runtime of the movie in minutes")):
    cursor.execute("SELECT T3.genre_name FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id WHERE T1.runtime = ?", (runtime,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get the genre with the highest vote average and revenue
@app.get("/v1/movies_4/highest_vote_average_revenue_genre", operation_id="get_highest_vote_average_revenue_genre", summary="Retrieves the genre with the highest combined vote average and revenue from the movie database. The genre is determined by considering both the average vote and the total revenue of the movies it is associated with.")
async def get_highest_vote_average_revenue_genre():
    cursor.execute("SELECT T3.genre_name FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id ORDER BY T1.vote_average DESC, T1.revenue LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"genre": []}
    return {"genre": result[0]}

# Endpoint to get genres based on movie tagline
@app.get("/v1/movies_4/genres_by_tagline", operation_id="get_genres_by_tagline", summary="Retrieves the genres of movies that have a specific tagline. The tagline is provided as an input parameter, allowing the user to filter the results to a particular tagline. The operation returns a list of genre names associated with the movies that match the provided tagline.")
async def get_genres_by_tagline(tagline: str = Query(..., description="Tagline of the movie")):
    cursor.execute("SELECT T3.genre_name FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id WHERE T1.tagline = ?", (tagline,))
    result = cursor.fetchall()
    if not result:
        return {"genres": []}
    return {"genres": [row[0] for row in result]}

# Endpoint to get country IDs based on movie title pattern
@app.get("/v1/movies_4/country_ids_by_title_pattern", operation_id="get_country_ids_by_title_pattern", summary="Retrieves the country IDs of movies whose titles match the provided pattern. The pattern can include wildcards to broaden the search. This operation does not return movie details, only the associated country IDs.")
async def get_country_ids_by_title_pattern(title_pattern: str = Query(..., description="Pattern to match movie titles (use %% for wildcard)")):
    cursor.execute("SELECT T2.COUNTry_id FROM movie AS T1 INNER JOIN production_COUNTry AS T2 ON T1.movie_id = T2.movie_id WHERE T1.title LIKE ?", (title_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"country_ids": []}
    return {"country_ids": [row[0] for row in result]}

# Endpoint to get movie titles produced in a specific country
@app.get("/v1/movies_4/movie_titles_by_country", operation_id="get_movie_titles_by_country", summary="Retrieves a list of movie titles produced in the specified country. The operation filters movies based on the provided country name and returns their titles.")
async def get_movie_titles_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN production_country AS T2 ON T1.movie_id = T2.movie_id INNER JOIN country AS T3 ON T2.country_id = T3.country_id WHERE T3.country_name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the total budget of movies with a specific keyword
@app.get("/v1/movies_4/total_budget_by_keyword", operation_id="get_total_budget_by_keyword", summary="Retrieves the cumulative budget of all movies associated with a specified keyword. The keyword is provided as an input parameter.")
async def get_total_budget_by_keyword(keyword_name: str = Query(..., description="Name of the keyword")):
    cursor.execute("SELECT SUM(T1.budget) FROM movie AS T1 INNER JOIN movie_keywords AS T2 ON T1.movie_id = T2.movie_id INNER JOIN keyword AS T3 ON T2.keyword_id = T3.keyword_id WHERE T3.keyword_name = ?", (keyword_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_budget": []}
    return {"total_budget": result[0]}

# Endpoint to get the average revenue of movies with a specific keyword
@app.get("/v1/movies_4/average_revenue_by_keyword", operation_id="get_average_revenue_by_keyword", summary="Retrieves the average revenue of movies associated with a specified keyword. The keyword is provided as an input parameter.")
async def get_average_revenue_by_keyword(keyword_name: str = Query(..., description="Name of the keyword")):
    cursor.execute("SELECT AVG(T1.revenue) FROM movie AS T1 INNER JOIN movie_keywords AS T2 ON T1.movie_id = T2.movie_id INNER JOIN keyword AS T3 ON T2.keyword_id = T3.keyword_id WHERE T3.keyword_name = ?", (keyword_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_revenue": []}
    return {"average_revenue": result[0]}

# Endpoint to get the popularity of the movie with the highest vote count
@app.get("/v1/movies_4/most_popular_movie_by_vote_count", operation_id="get_most_popular_movie_by_vote_count", summary="Retrieves the popularity score of the movie that has received the most votes. This operation returns a single popularity value, representing the highest-voted movie's popularity.")
async def get_most_popular_movie_by_vote_count():
    cursor.execute("SELECT popularity FROM movie ORDER BY vote_count DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"popularity": []}
    return {"popularity": result[0]}

# Endpoint to get the title of the movie with the highest revenue for a given budget
@app.get("/v1/movies_4/highest_revenue_movie_by_budget", operation_id="get_highest_revenue_movie_by_budget", summary="Retrieves the title of the movie that generated the highest revenue for a specified budget. The operation filters movies by the provided budget and returns the title of the movie with the highest revenue.")
async def get_highest_revenue_movie_by_budget(budget: int = Query(..., description="Budget of the movie")):
    cursor.execute("SELECT title FROM movie WHERE budget = ? ORDER BY revenue DESC LIMIT 1", (budget,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the company IDs associated with a specific movie title
@app.get("/v1/movies_4/company_ids_by_movie_title", operation_id="get_company_ids_by_movie_title", summary="Retrieves the unique identifiers of companies involved in the production of a movie, based on the provided movie title.")
async def get_company_ids_by_movie_title(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.company_id FROM movie AS T1 INNER JOIN movie_company AS T2 ON T1.movie_id = T2.movie_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"company_ids": []}
    return {"company_ids": [row[0] for row in result]}

# Endpoint to get the company IDs associated with movies released in a specific year
@app.get("/v1/movies_4/company_ids_by_release_year", operation_id="get_company_ids_by_release_year", summary="Retrieve the unique identifiers of companies that have produced movies released in a specified year. The operation filters movies based on the provided release year and identifies the associated production companies.")
async def get_company_ids_by_release_year(release_year: int = Query(..., description="Year of release (YYYY format)")):
    cursor.execute("SELECT T2.company_id FROM movie AS T1 INNER JOIN movie_company AS T2 ON T1.movie_id = T2.movie_id WHERE CAST(STRFTIME('%Y', T1.release_date) AS INT) = ?", (release_year,))
    result = cursor.fetchall()
    if not result:
        return {"company_ids": []}
    return {"company_ids": [row[0] for row in result]}

# Endpoint to get the highest revenue movie title produced by a specific production company
@app.get("/v1/movies_4/highest_revenue_movie_by_production_company", operation_id="get_highest_revenue_movie_by_production_company", summary="Retrieves the title of the movie that generated the highest revenue produced by the specified production company. The production company is identified by its name.")
async def get_highest_revenue_movie_by_production_company(company_name: str = Query(..., description="Name of the production company")):
    cursor.execute("SELECT T3.title FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T1.company_name = ? ORDER BY T3.revenue DESC LIMIT 1", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the release dates of movies produced by a specific production company
@app.get("/v1/movies_4/movie_release_dates_by_production_company", operation_id="get_movie_release_dates_by_production_company", summary="Retrieves the release dates of all movies produced by the specified production company. The operation requires the name of the production company as an input parameter.")
async def get_movie_release_dates_by_production_company(company_name: str = Query(..., description="Name of the production company")):
    cursor.execute("SELECT T3.release_date FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T1.company_name = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"release_dates": []}
    return {"release_dates": [row[0] for row in result]}

# Endpoint to get the language ID of a movie by its title
@app.get("/v1/movies_4/language_id_by_title", operation_id="get_language_id_by_title", summary="Retrieves the language identifier associated with a specific movie, based on the provided movie title. This operation returns the language ID of the movie, which can be used to identify the language in which the movie is available.")
async def get_language_id_by_title(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.language_id FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"language_ids": []}
    return {"language_ids": [row[0] for row in result]}

# Endpoint to get the language ID of the most popular movie
@app.get("/v1/movies_4/language_id_most_popular_movie", operation_id="get_language_id_most_popular_movie", summary="Retrieves the language identifier of the movie with the highest popularity rating. The popularity rating is determined by the number of user votes and the movie's release date. The language identifier corresponds to the language in which the movie is primarily presented.")
async def get_language_id_most_popular_movie():
    cursor.execute("SELECT T2.language_id FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.popularity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"language_id": []}
    return {"language_id": result[0]}

# Endpoint to get the language names of movies with titles starting with a specific pattern
@app.get("/v1/movies_4/language_names_by_title_pattern", operation_id="get_language_names_by_title_pattern", summary="Get the language names of movies with titles starting with a specific pattern")
async def get_language_names_by_title_pattern(title_pattern: str = Query(..., description="Pattern to match the movie title (use % as a wildcard)")):
    cursor.execute("SELECT T3.language_name FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id WHERE T1.title LIKE ?", (title_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"language_names": []}
    return {"language_names": [row[0] for row in result]}

# Endpoint to check if movies in a specific language are in a specific status
@app.get("/v1/movies_4/check_movie_status_by_language", operation_id="check_movie_status_by_language", summary="Determines whether movies in a specified language have a particular status. The operation returns a distinct 'YES' or 'NO' response based on the provided movie status and language name.")
async def check_movie_status_by_language(movie_status: str = Query(..., description="Status of the movie"), language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT DISTINCT CASE WHEN T1.movie_status = ? THEN 'YES' ELSE 'NO' END AS YORN FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id WHERE T3.language_name = ?", (movie_status, language_name))
    result = cursor.fetchall()
    if not result:
        return {"status_check": []}
    return {"status_check": [row[0] for row in result]}

# Endpoint to get distinct taglines of movies in a specific language
@app.get("/v1/movies_4/taglines_by_language", operation_id="get_taglines_by_language", summary="Retrieve unique taglines of movies available in the specified language. The operation filters movies based on the provided language and returns a list of distinct taglines.")
async def get_taglines_by_language(language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT DISTINCT T1.tagline FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id WHERE T3.language_name = ?", (language_name,))
    result = cursor.fetchall()
    if not result:
        return {"taglines": []}
    return {"taglines": [row[0] for row in result]}

# Endpoint to get distinct homepages of movies in a specific language
@app.get("/v1/movies_4/homepages_by_language", operation_id="get_homepages_by_language", summary="Retrieve a unique set of homepages for movies associated with a specified language. The language is identified by its name.")
async def get_homepages_by_language(language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT DISTINCT T1.homepage FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id WHERE T3.language_name = ?", (language_name,))
    result = cursor.fetchall()
    if not result:
        return {"homepages": []}
    return {"homepages": [row[0] for row in result]}

# Endpoint to get the revenue difference between movies in two different languages
@app.get("/v1/movies_4/revenue_difference_by_languages", operation_id="get_revenue_difference_by_languages", summary="Retrieve the difference in total revenue between movies produced in two specified languages. The operation calculates the sum of revenues for movies in each language and returns the difference between these sums.")
async def get_revenue_difference_by_languages(language_name_1: str = Query(..., description="First language name"), language_name_2: str = Query(..., description="Second language name")):
    cursor.execute("SELECT SUM(CASE WHEN T3.language_name = ? THEN T1.revenue ELSE 0 END) - SUM(CASE WHEN T3.language_name = ? THEN T1.revenue ELSE 0 END) AS DIFFERENCE FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id", (language_name_1, language_name_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the total revenue of movies produced by specific companies
@app.get("/v1/movies_4/total_revenue_by_companies", operation_id="get_total_revenue_by_companies", summary="Retrieves the total revenue generated by movies produced by the specified companies. The operation accepts two company names as input parameters and returns the sum of the revenue for movies produced by these companies.")
async def get_total_revenue_by_companies(company_name_1: str = Query(..., description="First company name"), company_name_2: str = Query(..., description="Second company name")):
    cursor.execute("SELECT SUM(T3.revenue) FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T1.company_name IN (?, ?)", (company_name_1, company_name_2))
    result = cursor.fetchone()
    if not result:
        return {"total_revenue": []}
    return {"total_revenue": result[0]}

# Endpoint to get the average revenue of movies in a specific language
@app.get("/v1/movies_4/average_revenue_by_language", operation_id="get_average_revenue_by_language", summary="Retrieves the average revenue of movies produced in a specified language. The operation calculates the mean revenue from a dataset of movies, filtering by the provided language name.")
async def get_average_revenue_by_language(language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT AVG(T1.revenue) FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id WHERE T3.language_name = ?", (language_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_revenue": []}
    return {"average_revenue": result[0]}

# Endpoint to get the most common person name
@app.get("/v1/movies_4/most_common_person_name", operation_id="get_most_common_person_name", summary="Retrieves the most frequently occurring person name from the person table, sorted in descending order by the count of occurrences.")
async def get_most_common_person_name():
    cursor.execute("SELECT person_name FROM person GROUP BY person_name ORDER BY COUNT(person_name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"person_name": []}
    return {"person_name": result[0]}

# Endpoint to get the average number of crew members per movie
@app.get("/v1/movies_4/average_crew_per_movie", operation_id="get_average_crew_per_movie", summary="Retrieves the average number of crew members involved in each movie. This is calculated by summing the total number of crew members across all movies and dividing it by the total number of movies.")
async def get_average_crew_per_movie():
    cursor.execute("SELECT CAST(SUM(CD) AS REAL) / COUNT(movie_id) FROM ( SELECT movie_id, COUNT(person_id) AS CD FROM movie_crew GROUP BY movie_id )")
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get keywords containing a specific substring
@app.get("/v1/movies_4/keywords_containing_substring", operation_id="get_keywords_containing_substring", summary="Retrieves all keyword names that contain a specified substring. The substring is used to filter the keyword names, returning only those that match the provided substring.")
async def get_keywords_containing_substring(substring: str = Query(..., description="Substring to search for in keyword names")):
    cursor.execute("SELECT keyword_name FROM keyword WHERE keyword_name LIKE ?", ('%' + substring + '%',))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get the maximum runtime of all movies
@app.get("/v1/movies_4/max_runtime", operation_id="get_max_runtime", summary="Retrieves the longest runtime duration among all movies in the database.")
async def get_max_runtime():
    cursor.execute("SELECT MAX(runtime) FROM movie")
    result = cursor.fetchone()
    if not result:
        return {"max_runtime": []}
    return {"max_runtime": result[0]}

# Endpoint to get the ISO code of a country by its name
@app.get("/v1/movies_4/country_iso_code", operation_id="get_country_iso_code", summary="Retrieves the ISO code of a specified country. The operation requires the name of the country as input and returns the corresponding ISO code. This endpoint is useful for obtaining the ISO code of a country when only its name is known.")
async def get_country_iso_code(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT COUNTry_iso_code FROM COUNTry WHERE COUNTry_name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"iso_code": []}
    return {"iso_code": result[0]}

# Endpoint to get the overview of a movie by its title
@app.get("/v1/movies_4/movie_overview", operation_id="get_movie_overview", summary="Retrieves the synopsis of a specific movie by its title. The provided title is used to locate the movie and return its overview.")
async def get_movie_overview(title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT overview FROM movie WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"overview": []}
    return {"overview": result[0]}

# Endpoint to get the count of movies produced by a specific production company
@app.get("/v1/movies_4/movie_count_by_production_company", operation_id="get_movie_count_by_production_company", summary="Retrieves the total number of movies produced by a specified production company. The operation requires the name of the production company as an input parameter.")
async def get_movie_count_by_production_company(company_name: str = Query(..., description="Name of the production company")):
    cursor.execute("SELECT COUNT(T1.movie_id) FROM movie_company AS T1 INNER JOIN production_company AS T2 ON T1.company_id = T2.company_id WHERE T2.company_name = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct person names who played characters with a specific substring in their name
@app.get("/v1/movies_4/distinct_person_names_by_character_substring", operation_id="get_distinct_person_names_by_character_substring", summary="Retrieve a list of unique names of individuals who have portrayed characters with a specified substring in their names. The input parameter allows you to define the substring to search for in character names.")
async def get_distinct_person_names_by_character_substring(character_substring: str = Query(..., description="Substring to search for in character names")):
    cursor.execute("SELECT DISTINCT T1.person_name FROM person AS T1 INNER JOIN movie_cast AS T2 ON T1.person_id = T2.person_id WHERE T2.character_name LIKE ?", ('%' + character_substring + '%',))
    result = cursor.fetchall()
    if not result:
        return {"person_names": []}
    return {"person_names": [row[0] for row in result]}

# Endpoint to get the most common keyword for movies released in a specific year
@app.get("/v1/movies_4/most_common_keyword_by_year", operation_id="get_most_common_keyword_by_year", summary="Retrieve the keyword that appears most frequently in movies released in a specific year. The year of release is provided in 'YYYY' format.")
async def get_most_common_keyword_by_year(release_year: str = Query(..., description="Year of release in 'YYYY' format")):
    cursor.execute("SELECT T3.keyword_name FROM movie AS T1 INNER JOIN movie_keywords AS T2 ON T1.movie_id = T2.movie_id INNER JOIN keyword AS T3 ON T2.keyword_id = T3.keyword_id WHERE T1.release_date LIKE ? GROUP BY T3.keyword_name ORDER BY COUNT(T3.keyword_name) DESC LIMIT 1", (release_year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"keyword": []}
    return {"keyword": result[0]}

# Endpoint to get the count of movies in a specific language
@app.get("/v1/movies_4/movie_count_by_language", operation_id="get_movie_count_by_language", summary="Retrieves the total number of movies available in a specified language. The language is identified using its unique code (e.g., 'vi' for Vietnamese).")
async def get_movie_count_by_language(language_code: str = Query(..., description="Language code (e.g., 'vi' for Vietnamese)")):
    cursor.execute("SELECT COUNT(T1.movie_id) FROM movie_languages AS T1 INNER JOIN language AS T2 ON T1.language_id = T2.language_id WHERE T2.language_code = ?", (language_code,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the third most common genre by movie count
@app.get("/v1/movies_4/third_most_common_genre", operation_id="get_third_most_common_genre", summary="Retrieves the genre that ranks third in terms of movie count. This operation identifies the genre with the third highest number of associated movies by joining the movie_genres and genre tables, grouping by genre_id, and ordering by the count of movie_ids. The result is limited to the third genre in the ordered list.")
async def get_third_most_common_genre():
    cursor.execute("SELECT T2.genre_name FROM movie_genres AS T1 INNER JOIN genre AS T2 ON T1.genre_id = T2.genre_id GROUP BY T2.genre_id ORDER BY COUNT(T1.movie_id) LIMIT 2, 1")
    result = cursor.fetchone()
    if not result:
        return {"genre": []}
    return {"genre": result[0]}

# Endpoint to get the language names of movies with a specific tagline and language role
@app.get("/v1/movies_4/language_names_by_tagline_and_role", operation_id="get_language_names", summary="Retrieve the names of languages associated with movies that have a specified tagline and language role. The operation filters movies based on the provided tagline and language role, then returns the corresponding language names.")
async def get_language_names(language_role: str = Query(..., description="Language role of the movie"), tagline: str = Query(..., description="Tagline of the movie")):
    cursor.execute("SELECT T3.language_name FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id INNER JOIN language_role AS T4 ON T2.language_role_id = T4.role_id WHERE T4.language_role = ? AND T1.tagline LIKE ?", (language_role, tagline))
    result = cursor.fetchall()
    if not result:
        return {"language_names": []}
    return {"language_names": [row[0] for row in result]}

# Endpoint to get the average revenue of movies produced in a specific country
@app.get("/v1/movies_4/average_revenue_by_country", operation_id="get_average_revenue", summary="Retrieves the average revenue of movies produced in a specified country. The operation calculates the average revenue from a dataset of movies, filtering by the provided country name.")
async def get_average_revenue(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT AVG(T1.revenue) FROM movie AS T1 INNER JOIN production_COUNTry AS T2 ON T1.movie_id = T2.movie_id INNER JOIN COUNTry AS T3 ON T2.COUNTry_id = T3.COUNTry_id WHERE T3.COUNTry_name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_revenue": []}
    return {"average_revenue": result[0]}

# Endpoint to get the character names of movie cast members with a specific gender
@app.get("/v1/movies_4/character_names_by_gender", operation_id="get_character_names", summary="Retrieve the names of movie cast members who identify as a specific gender. The operation filters the cast members based on the provided gender and returns their character names.")
async def get_character_names(gender: str = Query(..., description="Gender of the cast member")):
    cursor.execute("SELECT T1.character_name FROM movie_cast AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.gender_id WHERE T2.gender = ?", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"character_names": []}
    return {"character_names": [row[0] for row in result]}

# Endpoint to get the names of people with a specific job in movie crews, ordered by movie popularity
@app.get("/v1/movies_4/person_names_by_job", operation_id="get_person_names", summary="Retrieve a list of names of individuals who hold a specific role in movie crews, sorted by the popularity of the movies they have worked on. The number of results can be limited by specifying a maximum count.")
async def get_person_names(job: str = Query(..., description="Job of the person in the movie crew"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.person_name FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T2.job = ? ORDER BY T1.popularity DESC LIMIT ?", (job, limit))
    result = cursor.fetchall()
    if not result:
        return {"person_names": []}
    return {"person_names": [row[0] for row in result]}

# Endpoint to get the maximum budget for each genre
@app.get("/v1/movies_4/max_budget_by_genre", operation_id="get_max_budget_by_genre", summary="Retrieves the highest budget allocated to a movie for each genre. The operation groups movies by genre and identifies the maximum budget within each group.")
async def get_max_budget_by_genre():
    cursor.execute("SELECT T3.genre_name, MAX(T1.budget) FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id GROUP BY T3.genre_name")
    result = cursor.fetchall()
    if not result:
        return {"max_budget_by_genre": []}
    return {"max_budget_by_genre": [{"genre_name": row[0], "max_budget": row[1]} for row in result]}

# Endpoint to get the movie title with the most keywords
@app.get("/v1/movies_4/most_keywords_movie_title", operation_id="get_most_keywords_movie_title", summary="Retrieves the title of the movie with the highest number of associated keywords, up to the specified limit.")
async def get_most_keywords_movie_title(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_keywords AS T2 ON T1.movie_id = T2.movie_id GROUP BY T1.title ORDER BY COUNT(T2.keyword_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the department with the most crew members
@app.get("/v1/movies_4/most_crew_department", operation_id="get_most_crew_department", summary="Retrieves the department with the highest number of crew members, up to a specified limit.")
async def get_most_crew_department(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.department_name FROM department AS T1 INNER JOIN movie_crew AS T2 ON T1.department_id = T2.department_id GROUP BY T1.department_id ORDER BY COUNT(T2.department_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"department_names": []}
    return {"department_names": [row[0] for row in result]}

# Endpoint to get the percentage of movies produced in a specific country
@app.get("/v1/movies_4/percentage_movies_by_country", operation_id="get_percentage_movies_by_country", summary="Retrieves the percentage of movies produced in a specific country, as identified by its ISO code. The calculation is based on the total count of movies produced in that country compared to the overall count of movies in the database.")
async def get_percentage_movies_by_country(country_iso_code: str = Query(..., description="ISO code of the country")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.COUNTry_iso_code = ? THEN T1.movie_id ELSE NULL END) AS REAL) * 100 / COUNT(T1.movie_id) FROM movie AS T1 INNER JOIN production_COUNTry AS T2 ON T1.movie_id = T2.movie_id INNER JOIN COUNTry AS T3 ON T2.COUNTry_id = T3.COUNTry_id", (country_iso_code,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of female to male cast members
@app.get("/v1/movies_4/female_to_male_ratio", operation_id="get_female_to_male_ratio", summary="Retrieves the ratio of female to male cast members based on the provided gender parameters. This operation calculates the ratio by counting the number of cast members of each gender and dividing the female count by the male count.")
async def get_female_to_male_ratio(female_gender: str = Query(..., description="Gender of the female cast members"), male_gender: str = Query(..., description="Gender of the male cast members")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.gender = ? THEN T1.person_id ELSE NULL END) AS REAL) / COUNT(CASE WHEN T2.gender = ? THEN T1.person_id ELSE NULL END) FROM movie_cast AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.gender_id", (female_gender, male_gender))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get movie titles released in a specific year
@app.get("/v1/movies_4/movie_titles_by_year", operation_id="get_movie_titles_by_year", summary="Retrieves a list of movie titles that were released in a specified year. The year should be provided in 'YYYY' format.")
async def get_movie_titles_by_year(year: int = Query(..., description="Year of the movie release in 'YYYY' format")):
    cursor.execute("SELECT title FROM movie WHERE CAST(STRFTIME('%Y', release_date) AS INT) = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get country details based on country name
@app.get("/v1/movies_4/country_details", operation_id="get_country_details", summary="Retrieves the unique identifier and ISO code of a country, based on the provided country name. This operation allows you to look up a country's details using its name as a reference.")
async def get_country_details(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT COUNTry_id, COUNTry_iso_code FROM COUNTry WHERE COUNTry_name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get character names based on person name
@app.get("/v1/movies_4/character_names_by_person", operation_id="get_character_names_by_person", summary="Retrieves the names of all characters played by a specified person. The operation filters the movie cast based on the provided person's name and returns the corresponding character names.")
async def get_character_names_by_person(person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T2.character_name FROM person AS T1 INNER JOIN movie_cast AS T2 ON T1.person_id = T2.person_id WHERE T1.person_name = ?", (person_name,))
    result = cursor.fetchall()
    if not result:
        return {"character_names": []}
    return {"character_names": [row[0] for row in result]}

# Endpoint to get movie titles based on language name
@app.get("/v1/movies_4/movie_titles_by_language", operation_id="get_movie_titles_by_language", summary="Retrieves a list of movie titles that are available in the specified language. The language is identified by its name.")
async def get_movie_titles_by_language(language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT T1.title FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id WHERE T3.language_name = ?", (language_name,))
    result = cursor.fetchall()
    if not result:
        return {"movie_titles": []}
    return {"movie_titles": [row[0] for row in result]}

# Endpoint to get the most popular movie's release date and language
@app.get("/v1/movies_4/most_popular_movie_details", operation_id="get_most_popular_movie_details", summary="Get the release date and language of the most popular movie")
async def get_most_popular_movie_details():
    cursor.execute("SELECT T1.release_date, T3.language_name FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id ORDER BY T1.popularity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get language names based on language role and movie title
@app.get("/v1/movies_4/language_names_by_role_and_title", operation_id="get_language_names_by_role_and_title", summary="Retrieve the names of languages used in a specific role for a given movie. The operation requires the language role and the movie title as input parameters to filter the results.")
async def get_language_names_by_role_and_title(language_role: str = Query(..., description="Role of the language"), movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T3.language_name FROM movie AS T1 INNER JOIN movie_languages AS T2 ON T1.movie_id = T2.movie_id INNER JOIN language AS T3 ON T2.language_id = T3.language_id INNER JOIN language_role AS T4 ON T2.language_role_id = T4.role_id WHERE T4.language_role = ? AND T1.title = ?", (language_role, movie_title))
    result = cursor.fetchall()
    if not result:
        return {"language_names": []}
    return {"language_names": [row[0] for row in result]}

# Endpoint to get character names based on movie title
@app.get("/v1/movies_4/character_names_by_movie", operation_id="get_character_names_by_movie", summary="Retrieves the names of all characters appearing in a specified movie. The movie is identified by its title.")
async def get_character_names_by_movie(movie_title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT T2.character_name FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id WHERE T1.title = ?", (movie_title,))
    result = cursor.fetchall()
    if not result:
        return {"character_names": []}
    return {"character_names": [row[0] for row in result]}

# Endpoint to get the first cast member based on movie title
@app.get("/v1/movies_4/first_cast_member_by_movie", operation_id="get_first_cast_member_by_movie", summary="Retrieves the name of the cast member with the earliest cast order in a specified movie. The movie is identified by its title, which can include wildcard characters for partial matches. The result is the first cast member's name, sorted by their order in the movie's cast list.")
async def get_first_cast_member_by_movie(movie_title: str = Query(..., description="Title of the movie (use %% for wildcard)")):
    cursor.execute("SELECT T3.person_name FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T1.title LIKE ? ORDER BY T2.cast_order LIMIT 1", (movie_title,))
    result = cursor.fetchone()
    if not result:
        return {"cast_member": []}
    return {"cast_member": result[0]}

# Endpoint to get distinct job titles based on person name
@app.get("/v1/movies_4/distinct_jobs_by_person", operation_id="get_distinct_jobs_by_person", summary="Retrieve a unique set of job titles associated with a specific individual in the movie industry. The operation filters the data based on the provided person's name.")
async def get_distinct_jobs_by_person(person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT DISTINCT T2.job FROM person AS T1 INNER JOIN movie_crew AS T2 ON T1.person_id = T2.person_id WHERE T1.person_name = ?", (person_name,))
    result = cursor.fetchall()
    if not result:
        return {"job_titles": []}
    return {"job_titles": [row[0] for row in result]}

# Endpoint to get person names and department names based on movie title and job title
@app.get("/v1/movies_4/person_department_by_movie_job", operation_id="get_person_department_by_movie_job", summary="Retrieve the names of individuals and their respective departments based on a specific job title in a given movie. The movie title can be partially matched using wildcard characters. The job title must be an exact match.")
async def get_person_department_by_movie_job(movie_title: str = Query(..., description="Title of the movie (use %% for wildcard)"), job_title: str = Query(..., description="Job title")):
    cursor.execute("SELECT T3.person_name, T4.department_name FROM movie AS T1 INNER JOIN movie_crew AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id INNER JOIN department AS T4 ON T2.department_id = T4.department_id WHERE T1.title LIKE ? AND T2.job = ?", (movie_title, job_title))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": [{"person_name": row[0], "department_name": row[1]} for row in result]}

# Endpoint to get movie titles and revenues by production company name
@app.get("/v1/movies_4/movie_titles_revenues_by_company", operation_id="get_movie_titles_revenues_by_company", summary="Retrieves the titles and revenues of movies produced by a specified production company. The operation filters movies based on the provided production company name and returns the corresponding movie titles and revenues.")
async def get_movie_titles_revenues_by_company(company_name: str = Query(..., description="Name of the production company")):
    cursor.execute("SELECT T1.title, T1.revenue FROM movie AS T1 INNER JOIN movie_company AS T2 ON T1.movie_id = T2.movie_id INNER JOIN production_company AS T3 ON T2.company_id = T3.company_id WHERE T3.company_name = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": result}

# Endpoint to get the count of movies produced in a specific country
@app.get("/v1/movies_4/count_movies_by_country", operation_id="get_count_movies_by_country", summary="Retrieves the total number of movies produced in a specified country. The operation uses the provided country name to filter the production data and calculate the count.")
async def get_count_movies_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT COUNT(T2.movie_id) FROM COUNTry AS T1 INNER JOIN production_COUNTry AS T2 ON T1.COUNTry_id = T2.COUNTry_id WHERE T1.COUNTry_name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get production companies with movies having runtime above a certain threshold in a specific year
@app.get("/v1/movies_4/companies_runtime_above_threshold", operation_id="get_companies_runtime_above_threshold", summary="Retrieves the names of production companies that have released movies with a runtime significantly above the average runtime for the specified year. The runtime threshold is set at 35% above the average runtime for that year.")
async def get_companies_runtime_above_threshold(release_year: str = Query(..., description="Release year in 'YYYY%' format")):
    cursor.execute("SELECT T.company_name FROM ( SELECT DISTINCT T3.company_name, T1.runtime FROM movie AS T1 INNER JOIN movie_company AS T2 ON T1.movie_id = T2.movie_id INNER JOIN production_company AS T3 ON T3.company_id = T2.company_id WHERE T1.release_date LIKE ? ) T WHERE T.runtime * 100 > (0.35 * ( SELECT AVG(T1.runtime) FROM movie AS T1 INNER JOIN movie_company AS T2 ON T1.movie_id = T2.movie_id INNER JOIN production_company AS T3 ON T3.company_id = T2.company_id WHERE T1.release_date LIKE ? ) + ( SELECT AVG(T1.runtime) FROM movie AS T1 INNER JOIN movie_company AS T2 ON T1.movie_id = T2.movie_id INNER JOIN production_company AS T3 ON T3.company_id = T2.company_id WHERE T1.release_date LIKE ? )) * 100", (release_year, release_year, release_year))
    result = cursor.fetchall()
    if not result:
        return {"companies": []}
    return {"companies": result}

# Endpoint to get the percentage difference between two keywords in movies
@app.get("/v1/movies_4/keyword_percentage_difference", operation_id="get_keyword_percentage_difference", summary="Get the percentage difference between two keywords in movies")
async def get_keyword_percentage_difference(keyword1: str = Query(..., description="First keyword")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T1.keyword_name = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.keyword_name = ? THEN 1 ELSE 0 END)) AS REAL) * 100 / SUM(CASE WHEN T1.keyword_name = ? THEN 1 ELSE 0 END) FROM keyword AS T1 INNER JOIN movie_keywords AS T2 ON T1.keyword_id = T2.keyword_id", (keyword1, keyword1, keyword1))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the count of movies in a specific genre and release year
@app.get("/v1/movies_4/count_movies_by_genre_and_year", operation_id="get_count_movies_by_genre_and_year", summary="Retrieves the total number of movies belonging to a specified genre and released in a particular year. The genre is identified by its name, and the year is provided in 'YYYY' format.")
async def get_count_movies_by_genre_and_year(genre_name: str = Query(..., description="Name of the genre"), release_year: int = Query(..., description="Release year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.movie_id) FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id WHERE T3.genre_name = ? AND CAST(STRFTIME('%Y', T1.release_date) AS INT) = ?", (genre_name, release_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get movie titles produced by a specific company in a specific year
@app.get("/v1/movies_4/movie_titles_by_company_and_year", operation_id="get_movie_titles_by_company_and_year", summary="Retrieves the titles of movies produced by a specified production company in a given year. The operation filters movies based on the provided company name and release year, and returns a list of corresponding movie titles.")
async def get_movie_titles_by_company_and_year(company_name: str = Query(..., description="Name of the production company"), release_year: int = Query(..., description="Release year in 'YYYY' format")):
    cursor.execute("SELECT T3.title FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id INNER JOIN movie AS T3 ON T2.movie_id = T3.movie_id WHERE T1.company_name = ? AND CAST(STRFTIME('%Y', T3.release_date) AS INT) = ?", (company_name, release_year))
    result = cursor.fetchall()
    if not result:
        return {"movies": []}
    return {"movies": result}

# Endpoint to get the count of production companies with more than a specified number of movies
@app.get("/v1/movies_4/count_production_companies", operation_id="get_count_production_companies", summary="Retrieves the count of production companies that have produced more than a specified minimum number of movies. This operation provides a quantitative measure of the productivity of production companies in the database.")
async def get_count_production_companies(min_movies: int = Query(..., description="Minimum number of movies produced by the company")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T1.company_name AS CNAME FROM production_company AS T1 INNER JOIN movie_company AS T2 ON T1.company_id = T2.company_id GROUP BY T1.company_id HAVING COUNT(T1.company_name) > ? )", (min_movies,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the jobs of a specific person in movie crews
@app.get("/v1/movies_4/person_jobs", operation_id="get_person_jobs", summary="Retrieves the job roles held by a specific individual within various movie crews. The operation requires the person's name as input and returns a list of their job titles.")
async def get_person_jobs(person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T2.job FROM person AS T1 INNER JOIN movie_crew AS T2 ON T1.person_id = T2.person_id WHERE T1.person_name = ?", (person_name,))
    result = cursor.fetchall()
    if not result:
        return {"jobs": []}
    return {"jobs": [row[0] for row in result]}

# Endpoint to get the count of cast members of a specific gender in a specific movie
@app.get("/v1/movies_4/count_cast_members_by_gender", operation_id="get_count_cast_members_by_gender", summary="Get the count of cast members of a specific gender in a specific movie")
async def get_count_cast_members_by_gender(gender1: str = Query(..., description="First gender"), gender2: str = Query(..., description="Second gender"), title: str = Query(..., description="Title of the movie")):
    cursor.execute("SELECT COUNT(T2.cast_order) FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN gender AS T3 ON T3.gender_id = T2.gender_id WHERE (T3.gender = ? OR T3.gender = ?) AND T1.title = ? AND T2.cast_order = ( SELECT MIN(T2.cast_order) FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN gender AS T3 ON T3.gender_id = T2.gender_id WHERE (T3.gender = ? OR T3.gender = ?) AND T1.title = ? )", (gender1, gender2, title, gender1, gender2, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the character names played by a specific person in a specific movie
@app.get("/v1/movies_4/character_names_by_person_and_movie", operation_id="get_character_names_by_person_and_movie", summary="Retrieves the names of characters played by a given person in a specific movie. The operation requires the movie title and the person's name as input parameters to filter the results.")
async def get_character_names_by_person_and_movie(title: str = Query(..., description="Title of the movie"), person_name: str = Query(..., description="Name of the person")):
    cursor.execute("SELECT T2.character_name FROM movie AS T1 INNER JOIN movie_cast AS T2 ON T1.movie_id = T2.movie_id INNER JOIN person AS T3 ON T2.person_id = T3.person_id WHERE T1.title = ? AND T3.person_name = ?", (title, person_name))
    result = cursor.fetchall()
    if not result:
        return {"character_names": []}
    return {"character_names": [row[0] for row in result]}

# Endpoint to get the proportion of movies in a specific genre
@app.get("/v1/movies_4/proportion_of_movies_by_genre", operation_id="get_proportion_of_movies_by_genre", summary="Retrieves the proportion of movies belonging to a specific genre. This operation calculates the ratio of movies in the given genre to the total number of movies in the database. The genre is specified as an input parameter.")
async def get_proportion_of_movies_by_genre(genre_name: str = Query(..., description="Name of the genre")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.genre_name = ? THEN T1.movie_id ELSE NULL END) AS REAL) / COUNT(T1.movie_id) FROM movie AS T1 INNER JOIN movie_genres AS T2 ON T1.movie_id = T2.movie_id INNER JOIN genre AS T3 ON T2.genre_id = T3.genre_id", (genre_name,))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

api_calls = [
    "/v1/movies_4/production_companies_by_movie_title?movie_title=Four%20Rooms",
    "/v1/movies_4/count_production_companies_by_movie_title?movie_title=Four%20Rooms",
    "/v1/movies_4/movies_by_production_company?company_name=Universal%20Pictures",
    "/v1/movies_4/latest_movie_by_production_company?company_name=Universal%20Pictures",
    "/v1/movies_4/crew_members_by_movie_and_job?movie_title=Pirates%20of%20the%20Caribbean:%20At%20World%25s%20End&job=Director%20of%20Photography",
    "/v1/movies_4/jobs_by_person_and_movie?movie_title=Pirates%20of%20the%20Caribbean:%20At%20World%25s%20End&person_name=Dariusz%20Wolski",
    "/v1/movies_4/crew_members_by_movie?movie_title=Pirates%20of%20the%20Caribbean:%20At%20World%25s%20End",
    "/v1/movies_4/count_crew_members_by_movie_and_job?movie_title=Pirates%20of%20the%20Caribbean:%20At%20World%25s%20End&job=Producer",
    "/v1/movies_4/count_movies_by_person_and_job?person_name=Dariusz%20Wolski&job=Director%20of%20Photography",
    "/v1/movies_4/highest_rated_movie_by_director?person_name=Dariusz%20Wolski&job=Director%20of%20Photography",
    "/v1/movies_4/latest_movie_release_date_by_person?person_name=Dariusz%20Wolski",
    "/v1/movies_4/percentage_movies_above_vote_average?vote_average=5&person_name=Dariusz%20Wolski&job=Director%20of%20Photography",
    "/v1/movies_4/average_revenue_by_director?person_name=Dariusz%20Wolski&job=Director%20of%20Photography",
    "/v1/movies_4/movie_title_by_revenue?revenue=559852396",
    "/v1/movies_4/job_title_by_person_and_movie?person_name=David%20Rubin&title=Days%20of%20Thunder",
    "/v1/movies_4/count_keywords_by_movie?title=I%20Hope%20They%20Serve%20Beer%20in%20Hell",
    "/v1/movies_4/director_name_by_movie?title=Land%20of%20the%20Dead&job=Director",
    "/v1/movies_4/count_movies_by_company?company_name=Paramount%20Animation",
    "/v1/movies_4/count_cast_members_by_title_and_gender?title=Spider-Man%203&gender=Female",
    "/v1/movies_4/most_common_keyword",
    "/v1/movies_4/count_crew_members_by_title_and_job?title=The%20Amityville%20Horror&job=Producer",
    "/v1/movies_4/count_movies_by_keyword?keyword_name=saving%20the%20world",
    "/v1/movies_4/most_popular_movie_by_company?company_name=Cruel%20and%20Unusual%20Films",
    "/v1/movies_4/department_name_by_person_and_title?person_name=Marcia%20Ross&title=Reign%20of%20Fire",
    "/v1/movies_4/average_budget_by_director?person_name=Jaume%20Collet-Serra&job=Director",
    "/v1/movies_4/percentage_male_cast_by_title?title=Bride%20Wars",
    "/v1/movies_4/highest_budget_movie",
    "/v1/movies_4/count_movies_by_revenue?min_revenue=1000000000",
    "/v1/movies_4/earliest_release_date_by_status?movie_status=Released",
    "/v1/movies_4/count_persons_by_name?person_name=John%20Young",
    "/v1/movies_4/most_popular_movie",
    "/v1/movies_4/person_name_by_id?person_id=1325273",
    "/v1/movies_4/top_production_company",
    "/v1/movies_4/person_names_by_character_and_title?character_name=Captain%20Jack%20Sparrow&title_start=Pirates%20of%20the%20Caribbean",
    "/v1/movies_4/top_revenue_company",
    "/v1/movies_4/count_gender_in_movie?title=Mr.%20Smith%20Goes%20to%20Washington&gender=Female",
    "/v1/movies_4/production_companies_with_more_than_movies?movie_count=200",
    "/v1/movies_4/count_movies_person_acted_in?person_name=Harrison%20Ford",
    "/v1/movies_4/latest_movie_person_acted_in?person_name=Jamie%20Foxx",
    "/v1/movies_4/count_movies_person_acted_in_year?person_name=Quentin%20Tarantino&year=1995",
    "/v1/movies_4/earliest_movie_in_genre?genre_name=Crime",
    "/v1/movies_4/count_movies_in_genre?genre_name=Horror",
    "/v1/movies_4/person_ids_for_job_in_movie?movie_id=12&job=Second%20Film%20Editor",
    "/v1/movies_4/count_job_in_movie?movie_id=129&job=Animation",
    "/v1/movies_4/count_distinct_jobs_in_department?movie_id=19&department_id=7",
    "/v1/movies_4/cast_members_by_movie_and_order?movie_id=285&min_cast_order=1&max_cast_order=10",
    "/v1/movies_4/movies_and_characters_by_person?person_name=Jim%20Carrey",
    "/v1/movies_4/cast_members_by_movie_and_gender?movie_id=1865&gender=Female",
    "/v1/movies_4/movie_titles_by_person?person_name=Jim%20Carrey",
    "/v1/movies_4/cast_members_by_director_and_date_range?job=Director&start_date=1916-01-01&end_date=1925-12-31",
    "/v1/movies_4/movie_count_by_person_and_date_range?person_name=Uma%20Thurman&start_date=1990-01-01&end_date=2000-12-31",
    "/v1/movies_4/movie_titles_by_genre_and_vote_average?genre_name=Horror&min_vote_average=7",
    "/v1/movies_4/genre_popularity_by_revenue_and_date_range?min_revenue=120000000&start_date=2012-01-01&end_date=2015-12-31",
    "/v1/movies_4/movie_count_by_revenue_popularity_and_date_range?min_revenue=75000000&min_popularity=20&start_date=1990-01-01&end_date=2003-12-31",
    "/v1/movies_4/movies_by_language_and_date_range?language_name=Latin&start_date=1990-01-01&end_date=1995-12-31",
    "/v1/movies_4/average_revenue_by_country_and_year?country_name=United%20States%20of%20America&year=2006",
    "/v1/movies_4/revenue_difference_by_countries_and_year?country_name_1=United%20States%20of%20America&country_name_2=India&year=2016",
    "/v1/movies_4/genre_percentage_by_country_and_date_range?genre_name=Romance&country_name=India&start_date=2015-01-01&end_date=2015-12-31",
    "/v1/movies_4/person_names_by_character?character_name=Optimus%20Prime%20(voice)",
    "/v1/movies_4/gender_by_character?character_name=USAF%20Master%20Sgt.%20Epps",
    "/v1/movies_4/genres_by_movie?movie_title=Sky%20Captain%20and%20the%20World%20of%20Tomorrow",
    "/v1/movies_4/keywords_by_movie_title?title=Sky%20Captain%20and%20the%20World%20of%20Tomorrow",
    "/v1/movies_4/production_countries_by_movie_title?title=Gojira%20ni-sen%20mireniamu",
    "/v1/movies_4/movie_titles_by_keyword?keyword_name=extremis",
    "/v1/movies_4/movie_titles_by_production_country?country_name=France&limit=10",
    "/v1/movies_4/crew_members_by_movie_title?title=Mad%20Max%3A%20Fury%20Road&limit=10",
    "/v1/movies_4/genre_percentage_by_country?genre_name=Animation&country_name=Japan",
    "/v1/movies_4/gender_ratio_and_unspecified_count?title=Iron%20Man&gender_1=Male&gender_2=Female&unspecified_gender=Unspecified",
    "/v1/movies_4/movie_titles_before_year?year=2000&limit=5",
    "/v1/movies_4/keyword_ids_by_movie_title?title=Sin%20City",
    "/v1/movies_4/most_popular_keyword",
    "/v1/movies_4/genre_ids_by_movie_title?title=The%20Dark%20Knight",
    "/v1/movies_4/movie_titles_by_genre?genre_name=Thriller",
    "/v1/movies_4/movie_titles_by_status_and_genre?movie_status=Rumored&genre_name=Drama",
    "/v1/movies_4/highest_revenue_genre",
    "/v1/movies_4/genres_by_runtime?runtime=14",
    "/v1/movies_4/highest_vote_average_revenue_genre",
    "/v1/movies_4/genres_by_tagline?tagline=A%20long%20time%20ago%20in%20a%20galaxy%20far%2C%20far%20away...",
    "/v1/movies_4/country_ids_by_title_pattern?title_pattern=Pirates%20of%20the%20Caribbean%3A%20Dead%20Man%25s%20Chest",
    "/v1/movies_4/movie_titles_by_country?country_name=Canada",
    "/v1/movies_4/total_budget_by_keyword?keyword_name=video%20game",
    "/v1/movies_4/average_revenue_by_keyword?keyword_name=civil%20war",
    "/v1/movies_4/most_popular_movie_by_vote_count",
    "/v1/movies_4/highest_revenue_movie_by_budget?budget=0",
    "/v1/movies_4/company_ids_by_movie_title?title=Gladiator",
    "/v1/movies_4/company_ids_by_release_year?release_year=1916",
    "/v1/movies_4/highest_revenue_movie_by_production_company?company_name=Warner%20Bros.%20Pictures",
    "/v1/movies_4/movie_release_dates_by_production_company?company_name=Twentieth%20Century%20Fox%20Film%20Corporation",
    "/v1/movies_4/language_id_by_title?title=Walk%20the%20Line",
    "/v1/movies_4/language_id_most_popular_movie",
    "/v1/movies_4/language_names_by_title_pattern?title_pattern=C%era%20una%20volta%20il%20West",
    "/v1/movies_4/check_movie_status_by_language?movie_status=Post%20Production&language_name=Nederlands",
    "/v1/movies_4/taglines_by_language?language_name=Polski",
    "/v1/movies_4/homepages_by_language?language_name=Bahasa%20indonesia",
    "/v1/movies_4/revenue_difference_by_languages?language_name_1=English&language_name_2=Latin",
    "/v1/movies_4/total_revenue_by_companies?company_name_1=Fantasy%20Films&company_name_2=Live%20Entertainment",
    "/v1/movies_4/average_revenue_by_language?language_name=Latin",
    "/v1/movies_4/most_common_person_name",
    "/v1/movies_4/average_crew_per_movie",
    "/v1/movies_4/keywords_containing_substring?substring=christmas",
    "/v1/movies_4/max_runtime",
    "/v1/movies_4/country_iso_code?country_name=Kyrgyz%20Republic",
    "/v1/movies_4/movie_overview?title=The%20Pacifier",
    "/v1/movies_4/movie_count_by_production_company?company_name=Eddie%20Murphy%20Productions",
    "/v1/movies_4/distinct_person_names_by_character_substring?character_substring=captain",
    "/v1/movies_4/most_common_keyword_by_year?release_year=2006",
    "/v1/movies_4/movie_count_by_language?language_code=vi",
    "/v1/movies_4/third_most_common_genre",
    "/v1/movies_4/language_names_by_tagline_and_role?language_role=Original&tagline=An%20offer%20you%20can%25t%20refuse.",
    "/v1/movies_4/average_revenue_by_country?country_name=France",
    "/v1/movies_4/character_names_by_gender?gender=Unspecified",
    "/v1/movies_4/person_names_by_job?job=Director&limit=5",
    "/v1/movies_4/max_budget_by_genre",
    "/v1/movies_4/most_keywords_movie_title?limit=1",
    "/v1/movies_4/most_crew_department?limit=1",
    "/v1/movies_4/percentage_movies_by_country?country_iso_code=US",
    "/v1/movies_4/female_to_male_ratio?female_gender=Female&male_gender=Male",
    "/v1/movies_4/movie_titles_by_year?year=1945",
    "/v1/movies_4/country_details?country_name=Belgium",
    "/v1/movies_4/character_names_by_person?person_name=Catherine%20Deneuve",
    "/v1/movies_4/movie_titles_by_language?language_name=Somali",
    "/v1/movies_4/most_popular_movie_details",
    "/v1/movies_4/language_names_by_role_and_title?language_role=Original&movie_title=Four%20Rooms",
    "/v1/movies_4/character_names_by_movie?movie_title=Open%20Water",
    "/v1/movies_4/first_cast_member_by_movie?movie_title=Pirates%20of%20the%20Caribbean:%20At%20World%25s%20End",
    "/v1/movies_4/distinct_jobs_by_person?person_name=Sally%20Menke",
    "/v1/movies_4/person_department_by_movie_job?movie_title=Pirates%20of%20the%20Caribbean:%20At%20World%25s%20End&job_title=Music%20Editor",
    "/v1/movies_4/movie_titles_revenues_by_company?company_name=DreamWorks",
    "/v1/movies_4/count_movies_by_country?country_name=Canada",
    "/v1/movies_4/companies_runtime_above_threshold?release_year=2016%25",
    "/v1/movies_4/keyword_percentage_difference?keyword1=woman%20director&keyword2=independent%20film",
    "/v1/movies_4/count_movies_by_genre_and_year?genre_name=Adventure&release_year=2000",
    "/v1/movies_4/movie_titles_by_company_and_year?company_name=Paramount%20Pictures&release_year=2000",
    "/v1/movies_4/count_production_companies?min_movies=150",
    "/v1/movies_4/person_jobs?person_name=Mark%20Hammel",
    "/v1/movies_4/count_cast_members_by_gender?gender1=Male&gender2=Female&title=Pirates%20of%20the%20Caribbean:%20At%20World's%20End",
    "/v1/movies_4/character_names_by_person_and_movie?title=Pirates%20of%20the%20Caribbean:%20The%20Curse%20of%20the%20Black%20Pearl&person_name=Orlando%20Bloom",
    "/v1/movies_4/proportion_of_movies_by_genre?genre_name=Horror"
]
