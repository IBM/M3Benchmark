from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/university/university.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of university years based on the number of students and year
@app.get("/v1/university/count_university_years", operation_id="get_count_university_years", summary="Retrieves the count of university years that have more students than the specified number and correspond to the given year.")
async def get_count_university_years(num_students: int = Query(..., description="Number of students"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT COUNT(*) FROM university_year WHERE num_students > ? AND year = ?", (num_students, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ranking system ID based on criteria name
@app.get("/v1/university/ranking_system_id_by_criteria", operation_id="get_ranking_system_id_by_criteria", summary="Retrieves the unique identifier of the ranking system associated with a specific criteria. The criteria is identified by its name, which is provided as an input parameter.")
async def get_ranking_system_id_by_criteria(criteria_name: str = Query(..., description="Criteria name")):
    cursor.execute("SELECT ranking_system_id FROM ranking_criteria WHERE criteria_name = ?", (criteria_name,))
    result = cursor.fetchall()
    if not result:
        return {"ranking_system_ids": []}
    return {"ranking_system_ids": [row[0] for row in result]}

# Endpoint to get the count of universities based on a partial name match
@app.get("/v1/university/count_universities_by_name", operation_id="get_count_universities_by_name", summary="Retrieves the number of universities whose names contain a specified substring. The input parameter is used to define the substring for the search.")
async def get_count_universities_by_name(university_name: str = Query(..., description="Partial university name")):
    cursor.execute("SELECT COUNT(*) FROM university WHERE university_name LIKE ?", ('%' + university_name + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum student-staff ratio
@app.get("/v1/university/max_student_staff_ratio", operation_id="get_max_student_staff_ratio", summary="Retrieves the highest student-staff ratio from the university year data. This operation provides a single value representing the maximum ratio of students to staff in a given year.")
async def get_max_student_staff_ratio():
    cursor.execute("SELECT MAX(student_staff_ratio) FROM university_year WHERE student_staff_ratio = ( SELECT MAX(student_staff_ratio) FROM university_year )")
    result = cursor.fetchone()
    if not result:
        return {"max_student_staff_ratio": []}
    return {"max_student_staff_ratio": result[0]}

# Endpoint to get the count of ranking criteria based on ranking system ID
@app.get("/v1/university/count_ranking_criteria", operation_id="get_count_ranking_criteria", summary="Retrieves the total number of ranking criteria associated with a specific ranking system. The ranking system is identified by its unique ID.")
async def get_count_ranking_criteria(ranking_system_id: int = Query(..., description="Ranking system ID")):
    cursor.execute("SELECT COUNT(id) FROM ranking_criteria WHERE ranking_system_id = ?", (ranking_system_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get university IDs based on the percentage of international students and year range
@app.get("/v1/university/university_ids_by_international_students", operation_id="get_university_ids_by_international_students", summary="Retrieves the IDs of universities where the proportion of international students matches the provided percentage and the year falls within the specified range.")
async def get_university_ids_by_international_students(pct_international_students: float = Query(..., description="Percentage of international students"), start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT university_id FROM university_year WHERE pct_international_students = ? AND year BETWEEN ? AND ?", (pct_international_students, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"university_ids": []}
    return {"university_ids": [row[0] for row in result]}

# Endpoint to get the country with the most universities
@app.get("/v1/university/country_with_most_universities", operation_id="get_country_with_most_universities", summary="Retrieves the country with the most universities. This operation identifies the country with the highest number of universities by counting the number of universities in each country and returning the country with the highest count.")
async def get_country_with_most_universities():
    cursor.execute("SELECT T2.country_name FROM university AS T1 INNER JOIN country AS T2 ON T1.country_id = T2.id GROUP BY T2.country_name ORDER BY COUNT(T1.university_name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the university with the highest percentage of international students
@app.get("/v1/university/university_with_highest_international_students", operation_id="get_university_with_highest_international_students", summary="Retrieves the name of the university with the highest percentage of international students. This operation fetches the university with the most significant international student representation, based on the data from the university_year table. The result is determined by ordering the universities in descending order of the percentage of international students and selecting the top entry.")
async def get_university_with_highest_international_students():
    cursor.execute("SELECT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id ORDER BY T1.pct_international_students DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the top university based on a specific ranking criteria and year
@app.get("/v1/university/top_university_by_ranking_criteria", operation_id="get_top_university_by_ranking_criteria", summary="Retrieves the top-ranked university for a given ranking criteria and year. The criteria and year are used to filter the results, and the university with the highest score is returned. The ranking criteria ID is also required to ensure the correct criteria is used for ranking.")
async def get_top_university_by_ranking_criteria(criteria_name: str = Query(..., description="Criteria name"), year: int = Query(..., description="Year"), ranking_criteria_id: int = Query(..., description="Ranking criteria ID")):
    cursor.execute("SELECT T3.university_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T1.criteria_name = ? AND T2.year = ? AND T1.id = ? ORDER BY T2.score DESC LIMIT 1", (criteria_name, year, ranking_criteria_id))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the university with the highest number of students
@app.get("/v1/university/university_with_most_students", operation_id="get_university_with_most_students", summary="Retrieves the name of the university with the highest number of students in a given year. This operation returns the university with the most students based on the data from the university_year table, which is joined with the university table to obtain the university name.")
async def get_university_with_most_students():
    cursor.execute("SELECT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id ORDER BY T1.num_students LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the count of universities in a specific country
@app.get("/v1/university/count_universities_by_country", operation_id="get_count_universities_by_country", summary="Retrieves the total number of universities located in a specified country. The operation requires the name of the country as an input parameter.")
async def get_count_universities_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT COUNT(*) FROM university AS T1 INNER JOIN country AS T2 ON T1.country_id = T2.id WHERE T2.country_name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top university based on specific ranking criteria, year, criteria ID, and country
@app.get("/v1/university/top_university_by_criteria", operation_id="get_top_university_by_criteria", summary="Retrieve the top-ranked university based on a specified ranking criterion, year, criteria ID, and country. The operation returns the name of the university with the highest score for the given criteria, year, and country. The criteria name, year, criteria ID, and country name are required input parameters.")
async def get_top_university_by_criteria(criteria_name: str = Query(..., description="Name of the ranking criteria"), year: int = Query(..., description="Year of the ranking"), criteria_id: int = Query(..., description="ID of the ranking criteria"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T3.university_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id INNER JOIN country AS T4 ON T4.id = T3.country_id WHERE T1.criteria_name = ? AND T2.year = ? AND T1.id = ? AND T4.country_name = ? ORDER BY T2.score DESC LIMIT 1", (criteria_name, year, criteria_id, country_name))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the count of universities based on ranking criteria, year range, and score
@app.get("/v1/university/count_universities_by_criteria_year_score", operation_id="get_count_universities_by_criteria_year_score", summary="Retrieves the total number of universities that meet the specified ranking criteria, within a given year range, and with a particular score. The criteria_name parameter determines the ranking category, while the start_year and end_year parameters define the time period. The score parameter specifies the exact ranking score to consider.")
async def get_count_universities_by_criteria_year_score(criteria_name: str = Query(..., description="Name of the ranking criteria"), start_year: int = Query(..., description="Start year of the ranking"), end_year: int = Query(..., description="End year of the ranking"), score: int = Query(..., description="Score of the ranking")):
    cursor.execute("SELECT COUNT(*) FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id WHERE T1.criteria_name = ? AND T2.year BETWEEN ? AND ? AND T2.score = ?", (criteria_name, start_year, end_year, score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country name of a specific university
@app.get("/v1/university/country_of_university", operation_id="get_country_of_university", summary="Retrieves the country name associated with a specific university. The operation requires the university's name as input and returns the corresponding country name.")
async def get_country_of_university(university_name: str = Query(..., description="Name of the university")):
    cursor.execute("SELECT T2.country_name FROM university AS T1 INNER JOIN country AS T2 ON T1.country_id = T2.id WHERE university_name = ?", (university_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the count of universities based on university name, minimum score, and ranking criteria
@app.get("/v1/university/count_universities_by_name_score_criteria", operation_id="get_count_universities_by_name_score_criteria", summary="Retrieves the count of universities that meet the specified name, minimum score, and ranking criteria. The operation filters universities by name and score, and further refines the results based on the selected ranking criteria.")
async def get_count_universities_by_name_score_criteria(university_name: str = Query(..., description="Name of the university"), min_score: int = Query(..., description="Minimum score"), criteria_name: str = Query(..., description="Name of the ranking criteria")):
    cursor.execute("SELECT COUNT(*) FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T3.university_name = ? AND T2.score >= ? AND T1.criteria_name = ?", (university_name, min_score, criteria_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ranking criteria names for a specific ranking system
@app.get("/v1/university/criteria_by_ranking_system", operation_id="get_criteria_by_ranking_system", summary="Retrieves the names of all criteria associated with a specified ranking system. The system_name parameter is used to identify the ranking system of interest.")
async def get_criteria_by_ranking_system(system_name: str = Query(..., description="Name of the ranking system")):
    cursor.execute("SELECT T2.criteria_name FROM ranking_system AS T1 INNER JOIN ranking_criteria AS T2 ON T1.id = T2.ranking_system_id WHERE T1.system_name = ?", (system_name,))
    result = cursor.fetchall()
    if not result:
        return {"criteria_names": []}
    return {"criteria_names": [row[0] for row in result]}

# Endpoint to get the names of universities with more than a specified number of students in a given year
@app.get("/v1/university/universities_by_student_count_year", operation_id="get_universities_by_student_count_year", summary="Retrieves the names of universities that have more students than the specified minimum in a given year. The operation requires the minimum number of students and the year as input parameters.")
async def get_universities_by_student_count_year(min_students: int = Query(..., description="Minimum number of students"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.num_students > ? AND T1.year = ?", (min_students, year))
    result = cursor.fetchall()
    if not result:
        return {"university_names": []}
    return {"university_names": [row[0] for row in result]}

# Endpoint to get the distinct country names of universities with more than a specified percentage of international students within a given year range
@app.get("/v1/university/countries_by_international_students_year_range", operation_id="get_countries_by_international_students_year_range", summary="Retrieve the unique country names of universities that have more than a specified percentage of international students within a given year range. The operation requires the minimum percentage of international students, the start year, and the end year as input parameters.")
async def get_countries_by_international_students_year_range(min_pct_international_students: float = Query(..., description="Minimum percentage of international students"), start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT DISTINCT T3.country_name FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T2.pct_international_students > ? AND T2.year BETWEEN ? AND ?", (min_pct_international_students, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get the count of universities and the number of universities in the USA with more than a specified number of female students in a given year
@app.get("/v1/university/count_universities_female_students_year", operation_id="get_count_universities_female_students_year", summary="Retrieves the total count of universities and the number of universities in the USA that have more than a specified number of female students in a given year. The operation considers the total number of students and the percentage of female students in each university for the specified year.")
async def get_count_universities_female_students_year(country_name: str = Query(..., description="Name of the country"), year: int = Query(..., description="Year"), min_female_students: int = Query(..., description="Minimum number of female students")):
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN T3.country_name = ? THEN 1 ELSE 0 END) AS nums_in_usa FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T2.year = ? AND T2.num_students * T2.pct_female_students / 100 > ?", (country_name, year, min_female_students))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "nums_in_usa": []}
    return {"count": result[0], "nums_in_usa": result[1]}

# Endpoint to get the top universities by the percentage of international students
@app.get("/v1/university/top_universities_by_international_students", operation_id="get_top_universities_by_international_students", summary="Retrieves a list of top universities ranked by the number of international students. The list is sorted in descending order based on the calculated number of international students, which is derived from the total number of students and the percentage of international students. The operation allows specifying the number of universities to return in the list.")
async def get_top_universities_by_international_students(limit: int = Query(..., description="Number of top universities to return")):
    cursor.execute("SELECT DISTINCT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id ORDER BY (CAST(T1.num_students * T1.pct_international_students AS REAL) / 100) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"university_names": []}
    return {"university_names": [row[0] for row in result]}

# Endpoint to get the university with the highest student-staff ratio
@app.get("/v1/university/highest_student_staff_ratio", operation_id="get_highest_student_staff_ratio", summary="Retrieves the unique identifier of the university with the highest student-to-staff ratio for the most recent academic year.")
async def get_highest_student_staff_ratio():
    cursor.execute("SELECT university_id FROM university_year ORDER BY student_staff_ratio DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"university_id": []}
    return {"university_id": result[0]}

# Endpoint to get the year with the lowest number of students
@app.get("/v1/university/lowest_num_students_year", operation_id="get_lowest_num_students_year", summary="Retrieves the year with the smallest student population from the university database. This operation returns the year with the least number of students enrolled, based on the data available in the university_year table.")
async def get_lowest_num_students_year():
    cursor.execute("SELECT year FROM university_year ORDER BY num_students ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the average percentage of female students
@app.get("/v1/university/avg_pct_female_students", operation_id="get_avg_pct_female_students", summary="Retrieves the average percentage of female students across all years in the university database.")
async def get_avg_pct_female_students():
    cursor.execute("SELECT AVG(pct_female_students) FROM university_year")
    result = cursor.fetchone()
    if not result:
        return {"avg_pct_female_students": []}
    return {"avg_pct_female_students": result[0]}

# Endpoint to get the number of international students and total number of students for a specific year and university
@app.get("/v1/university/international_students_count", operation_id="get_international_students_count", summary="Retrieves the total number of students and the count of international students for a specific university and year. The year should be provided in YYYY format, and the university is identified by its unique ID.")
async def get_international_students_count(year: int = Query(..., description="Year in YYYY format"), university_id: int = Query(..., description="University ID")):
    cursor.execute("SELECT pct_international_students * num_students, num_students FROM university_year WHERE year = ? AND university_id = ?", (year, university_id))
    result = cursor.fetchone()
    if not result:
        return {"international_students": [], "total_students": []}
    return {"international_students": result[0], "total_students": result[1]}

# Endpoint to get the university ID based on the university name
@app.get("/v1/university/id_by_name", operation_id="get_university_id_by_name", summary="Retrieves the unique identifier of a university by its name. The operation searches for a university with the provided name and returns its corresponding ID.")
async def get_university_id_by_name(university_name: str = Query(..., description="Name of the university")):
    cursor.execute("SELECT id FROM university WHERE university_name = ?", (university_name,))
    result = cursor.fetchone()
    if not result:
        return {"university_id": []}
    return {"university_id": result[0]}

# Endpoint to get the university ID based on ranking score and year
@app.get("/v1/university/id_by_ranking_score_year", operation_id="get_university_id_by_ranking_score_year", summary="Get the university ID based on ranking score and year")
async def get_university_id_by_ranking_score_year(score: int = Query(..., description="Ranking score"), year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT university_id FROM university_ranking_year WHERE score = ? AND year = ?", (score, year))
    result = cursor.fetchone()
    if not result:
        return {"university_id": []}
    return {"university_id": result[0]}

# Endpoint to get the ranking system name based on criteria name
@app.get("/v1/university/ranking_system_by_criteria", operation_id="get_ranking_system_by_criteria", summary="Retrieves the name of the ranking system that corresponds to the specified criteria name. This operation fetches the system name from the ranking_system table, which is linked to the ranking_criteria table via the ranking_system_id. The criteria_name parameter is used to filter the results.")
async def get_ranking_system_by_criteria(criteria_name: str = Query(..., description="Criteria name")):
    cursor.execute("SELECT T1.system_name FROM ranking_system AS T1 INNER JOIN ranking_criteria AS T2 ON T1.id = T2.ranking_system_id WHERE T2.criteria_name = ?", (criteria_name,))
    result = cursor.fetchone()
    if not result:
        return {"system_name": []}
    return {"system_name": result[0]}

# Endpoint to get the student-staff ratio for a specific university and year
@app.get("/v1/university/student_staff_ratio", operation_id="get_student_staff_ratio", summary="Retrieves the student-staff ratio for a specific university in a given year. The operation requires the university's name and the year as input parameters. The university's name is used to identify the institution, while the year is used to determine the academic period for which the student-staff ratio is calculated.")
async def get_student_staff_ratio(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT T1.student_staff_ratio FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T2.university_name = ? AND T1.year = ?", (university_name, year))
    result = cursor.fetchone()
    if not result:
        return {"student_staff_ratio": []}
    return {"student_staff_ratio": result[0]}

# Endpoint to get the country name based on university ID
@app.get("/v1/university/country_by_university_id", operation_id="get_country_by_university_id", summary="Retrieves the name of the country associated with the specified university ID. This operation fetches the country name from the country table by joining it with the university table using the country ID. The university ID is used to filter the results.")
async def get_country_by_university_id(university_id: int = Query(..., description="University ID")):
    cursor.execute("SELECT T2.country_name FROM university AS T1 INNER JOIN country AS T2 ON T1.country_id = T2.id WHERE T1.id = ?", (university_id,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the total number of students in universities of a specific country
@app.get("/v1/university/total_students_by_country", operation_id="get_total_students_by_country", summary="Retrieves the total number of students enrolled in universities located in a specified country. The operation calculates the sum of students across all universities in the given country.")
async def get_total_students_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(T2.num_students) FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T3.country_name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_students": []}
    return {"total_students": result[0]}

# Endpoint to get ranking criteria IDs for a specific university and year
@app.get("/v1/university/ranking_criteria_ids", operation_id="get_ranking_criteria_ids", summary="Retrieves the unique identifiers of the ranking criteria associated with a specific university for a given year. The operation requires the university's name and the year as input parameters to filter the results.")
async def get_ranking_criteria_ids(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.ranking_criteria_id FROM university_ranking_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T2.university_name = ? AND T1.year = ?", (university_name, year))
    result = cursor.fetchall()
    if not result:
        return {"ranking_criteria_ids": []}
    return {"ranking_criteria_ids": [row[0] for row in result]}

# Endpoint to get university names in a specific country
@app.get("/v1/university/university_names_by_country", operation_id="get_university_names_by_country", summary="Retrieves the names of universities located in a specified country. The operation filters universities based on the provided country name and returns a list of their names.")
async def get_university_names_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.university_name FROM university AS T1 INNER JOIN country AS T2 ON T1.country_id = T2.id WHERE T2.country_name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"university_names": []}
    return {"university_names": [row[0] for row in result]}

# Endpoint to get ranking criteria names for a specific university and year
@app.get("/v1/university/ranking_criteria_names", operation_id="get_ranking_criteria_names", summary="Retrieves the names of the ranking criteria used for a specific university in a given year. The criteria are determined by the university's ranking data for the provided year.")
async def get_ranking_criteria_names(university_id: int = Query(..., description="ID of the university"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.criteria_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id WHERE T2.university_id = ? AND T2.year = ?", (university_id, year))
    result = cursor.fetchall()
    if not result:
        return {"ranking_criteria_names": []}
    return {"ranking_criteria_names": [row[0] for row in result]}

# Endpoint to get the average score of universities in a specific country
@app.get("/v1/university/average_score_by_country", operation_id="get_average_score_by_country", summary="Retrieves the average score of universities located in a specified country. The score is calculated based on the university's ranking across various years.")
async def get_average_score_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT AVG(T2.score) FROM university AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T3.country_name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the country ID of the university with the highest number of students in a specific year
@app.get("/v1/university/country_id_highest_students", operation_id="get_country_id_highest_students", summary="Retrieves the country ID of the university with the highest number of students in a specified year. The operation filters universities by the given year and ranks them by student count, returning the country ID of the top-ranked university.")
async def get_country_id_highest_students(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T2.country_id FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? ORDER BY T1.num_students DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"country_id": []}
    return {"country_id": result[0]}

# Endpoint to get the number of international students and score for a specific university and year
@app.get("/v1/university/international_students_score", operation_id="get_international_students_score", summary="Retrieves the number of international students and their corresponding score for a specific university in a given year. The score is calculated by multiplying the total number of students by the percentage of international students, then dividing by 100. The university's score for the specified year is also provided.")
async def get_international_students_score(year: int = Query(..., description="Year"), university_id: int = Query(..., description="ID of the university")):
    cursor.execute("SELECT CAST(T1.num_students * T1.pct_international_students AS REAL) / 100, T2.score FROM university_year AS T1 INNER JOIN university_ranking_year AS T2 ON T1.university_id = T2.university_id WHERE T2.year = ? AND T1.university_id = ?", (year, university_id))
    result = cursor.fetchall()
    if not result:
        return {"international_students_score": []}
    return {"international_students_score": [{"international_students": row[0], "score": row[1]} for row in result]}

# Endpoint to get the total number of students for universities with a specific score and year
@app.get("/v1/university/total_students_by_score_year", operation_id="get_total_students_by_score_year", summary="Retrieves the total number of students enrolled in universities that have a specified score and year. The score and year are used to filter the universities and calculate the aggregated student count.")
async def get_total_students_by_score_year(score: int = Query(..., description="Score"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT SUM(T1.num_students) FROM university_year AS T1 INNER JOIN university_ranking_year AS T2 ON T1.university_id = T2.university_id WHERE T2.score = ? AND T1.year = ?", (score, year))
    result = cursor.fetchone()
    if not result:
        return {"total_students": []}
    return {"total_students": result[0]}

# Endpoint to get distinct country names where universities have more students than a percentage of the average number of students in a specific year
@app.get("/v1/university/countries_above_avg_students", operation_id="get_countries_above_avg_students", summary="Retrieves a list of unique country names where universities have a student population exceeding a specified percentage of the average student population for a given year. The year and the percentage are provided as input parameters.")
async def get_countries_above_avg_students(year: int = Query(..., description="Year"), percentage: int = Query(..., description="Percentage of the average number of students")):
    cursor.execute("SELECT DISTINCT T3.country_name FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T2.year = ? AND T2.num_students * 100 > ( SELECT AVG(num_students) FROM university_year ) * ?", (year, percentage))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get the average percentage of international students for universities with a score below a certain threshold in a specific year
@app.get("/v1/university/avg_pct_international_students", operation_id="get_avg_pct_international_students", summary="Retrieves the average percentage of international students for universities that scored below a specified threshold in a given year. The calculation is based on the total number of international students and the total number of universities that meet the criteria.")
async def get_avg_pct_international_students(score: int = Query(..., description="Score threshold"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT SUM(CAST(T1.num_students * T1.pct_international_students AS REAL) / 100) / COUNT(*) * 100 FROM university_year AS T1 INNER JOIN university_ranking_year AS T2 ON T1.university_id = T2.university_id WHERE T2.score < ? AND T1.year = ?", (score, year))
    result = cursor.fetchone()
    if not result:
        return {"avg_pct_international_students": []}
    return {"avg_pct_international_students": result[0]}

# Endpoint to get the total number of students for a specific year
@app.get("/v1/university/total_students_by_year", operation_id="get_total_students_by_year", summary="Retrieves the total count of students enrolled in the university for a specified year.")
async def get_total_students_by_year(year: int = Query(..., description="Year")):
    cursor.execute("SELECT SUM(num_students) FROM university_year WHERE year = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"total_students": []}
    return {"total_students": result[0]}

# Endpoint to get the sum of female students in a given year
@app.get("/v1/university/sum_female_students_by_year", operation_id="get_sum_female_students", summary="Retrieves the total number of female students enrolled in the university for a specific year. The calculation is based on the total number of students and the percentage of female students in that year.")
async def get_sum_female_students(year: int = Query(..., description="Year to filter the data")):
    cursor.execute("SELECT SUM(CAST(num_students * pct_female_students AS REAL) / 100) FROM university_year WHERE year = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the university ID with the highest number of students in a given year
@app.get("/v1/university/university_id_max_students_by_year", operation_id="get_university_id_max_students", summary="Retrieves the ID of the university with the maximum number of students in the specified year.")
async def get_university_id_max_students(year: int = Query(..., description="Year to filter the data")):
    cursor.execute("SELECT university_id FROM university_year WHERE year = ? ORDER BY num_students DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"university_id": []}
    return {"university_id": result[0]}

# Endpoint to get the count of universities with more than a specified number of students and percentage of international students in a given year
@app.get("/v1/university/count_universities_by_criteria", operation_id="get_count_universities", summary="Retrieve the count of universities that meet specific criteria in a given year. The criteria include having more than a specified number of students and a certain percentage of international students. The year, minimum number of students, and minimum percentage of international students are provided as input parameters.")
async def get_count_universities(year: int = Query(..., description="Year to filter the data"), min_students: int = Query(..., description="Minimum number of students"), min_pct_international: float = Query(..., description="Minimum percentage of international students")):
    cursor.execute("SELECT COUNT(*) FROM university_year WHERE year = ? AND num_students > ? AND pct_international_students > ?", (year, min_students, min_pct_international))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the university ID with the highest percentage of female students in a given year
@app.get("/v1/university/university_id_max_pct_female_students_by_year", operation_id="get_university_id_max_pct_female_students", summary="Retrieves the university ID with the highest percentage of female students in a specified year. The operation filters the data based on the provided year and returns the university ID with the highest percentage of female students.")
async def get_university_id_max_pct_female_students(year: int = Query(..., description="Year to filter the data")):
    cursor.execute("SELECT university_id FROM university_year WHERE year = ? ORDER BY pct_female_students DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"university_id": []}
    return {"university_id": result[0]}

# Endpoint to get the university name with the highest score in a given year
@app.get("/v1/university/university_name_max_score_by_year", operation_id="get_university_name_max_score", summary="Retrieves the name of the university with the highest score in a specified year. The operation filters the university ranking data by the provided year and returns the university name with the highest score for that year.")
async def get_university_name_max_score(year: int = Query(..., description="Year to filter the data")):
    cursor.execute("SELECT T2.university_name FROM university_ranking_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? ORDER BY T1.score DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the university name with the highest number of students in a given year
@app.get("/v1/university/university_name_max_students_by_year", operation_id="get_university_name_max_students", summary="Retrieves the name of the university with the highest student enrollment for a specified year. The year parameter is used to filter the data.")
async def get_university_name_max_students(year: int = Query(..., description="Year to filter the data")):
    cursor.execute("SELECT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? ORDER BY T1.num_students DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the university name with the highest score in a specific ranking criteria
@app.get("/v1/university/university_name_max_score_by_criteria", operation_id="get_university_name_max_score_by_criteria", summary="Retrieves the name of the university with the highest score in a specified ranking criteria. The criteria is used to filter the ranking data and identify the university with the top score.")
async def get_university_name_max_score_by_criteria(criteria_name: str = Query(..., description="Ranking criteria name")):
    cursor.execute("SELECT T1.university_name FROM university AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.university_id INNER JOIN ranking_criteria AS T3 ON T3.id = T2.ranking_criteria_id WHERE T3.criteria_name = ? ORDER BY T2.score DESC LIMIT 1", (criteria_name,))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the percentage of international students for a specific university in a given year
@app.get("/v1/university/pct_international_students_by_university_year", operation_id="get_pct_international_students", summary="Retrieves the percentage of international students enrolled in a specific university during a given year. The operation requires the year and the name of the university as input parameters to filter the data and return the corresponding percentage.")
async def get_pct_international_students(year: int = Query(..., description="Year to filter the data"), university_name: str = Query(..., description="University name")):
    cursor.execute("SELECT T1.pct_international_students FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? AND T2.university_name = ?", (year, university_name))
    result = cursor.fetchone()
    if not result:
        return {"pct_international_students": []}
    return {"pct_international_students": result[0]}

# Endpoint to get the number of female students for a specific university in a given year
@app.get("/v1/university/num_female_students_by_university_year", operation_id="get_num_female_students", summary="Retrieves the estimated number of female students enrolled in a specific university during a given year. The calculation is based on the total number of students and the percentage of female students in that year for the specified university.")
async def get_num_female_students(year: int = Query(..., description="Year to filter the data"), university_name: str = Query(..., description="University name")):
    cursor.execute("SELECT CAST(T1.num_students * T1.pct_female_students AS REAL) / 100 FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? AND T2.university_name = ?", (year, university_name))
    result = cursor.fetchone()
    if not result:
        return {"num_female_students": []}
    return {"num_female_students": result[0]}

# Endpoint to get the country name of a specific university
@app.get("/v1/university/country_name_by_university", operation_id="get_country_name_by_university", summary="Retrieves the country name associated with a specific university, identified by its name. The operation returns the country name from the country table that matches the university name provided in the university table.")
async def get_country_name_by_university(university_name: str = Query(..., description="Name of the university")):
    cursor.execute("SELECT T2.country_name FROM university AS T1 INNER JOIN country AS T2 ON T1.country_id = T2.id WHERE T1.university_name = ?", (university_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the top university in a specific country by ranking score
@app.get("/v1/university/top_university_by_country", operation_id="get_top_university_by_country", summary="Retrieves the top-ranked university in a specified country, based on aggregated ranking scores from multiple years. The operation filters universities by country name and calculates the sum of their scores. The university with the highest total score is returned.")
async def get_top_university_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.university_name FROM university AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T3.country_name = ? GROUP BY T1.university_name ORDER BY SUM(T2.score) DESC LIMIT 1", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get universities with more than a specific number of students in a given year
@app.get("/v1/university/universities_by_year_and_students", operation_id="get_universities_by_year_and_students", summary="Retrieves the names of universities that had more than a specified number of students in a given year. The year is provided in YYYY format, and the minimum number of students is also specified as an input parameter.")
async def get_universities_by_year_and_students(year: int = Query(..., description="Year in YYYY format"), num_students: int = Query(..., description="Number of students")):
    cursor.execute("SELECT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? AND T1.num_students > ?", (year, num_students))
    result = cursor.fetchall()
    if not result:
        return {"university_names": []}
    return {"university_names": [row[0] for row in result]}

# Endpoint to get the count of ranking criteria for a specific ranking system
@app.get("/v1/university/count_criteria_by_ranking_system", operation_id="get_count_criteria_by_ranking_system", summary="Retrieves the total number of criteria associated with a specified ranking system. The system_name parameter is used to identify the ranking system of interest.")
async def get_count_criteria_by_ranking_system(system_name: str = Query(..., description="Name of the ranking system")):
    cursor.execute("SELECT COUNT(T2.criteria_name) FROM ranking_system AS T1 INNER JOIN ranking_criteria AS T2 ON T1.id = T2.ranking_system_id WHERE T1.system_name = ?", (system_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of universities with a specific score in a given year
@app.get("/v1/university/count_universities_by_score_and_year", operation_id="get_count_universities_by_score_and_year", summary="Retrieves the total number of universities that have achieved a specified score during a particular year. The score and year are provided as input parameters.")
async def get_count_universities_by_score_and_year(score: int = Query(..., description="Score of the university"), year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT COUNT(*) FROM university_year AS T1 INNER JOIN university_ranking_year AS T2 ON T1.university_id = T2.university_id WHERE T2.score = ? AND T1.year = ?", (score, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the number of domestic students for a specific university over a range of years
@app.get("/v1/university/domestic_students_by_university_and_year_range", operation_id="get_domestic_students_by_university_and_year_range", summary="Retrieve the total number of domestic students enrolled in a specific university between a given start and end year. The calculation is based on the total number of students and the percentage of international students for each year within the specified range.")
async def get_domestic_students_by_university_and_year_range(university_name: str = Query(..., description="Name of the university"), start_year: int = Query(..., description="Start year in YYYY format"), end_year: int = Query(..., description="End year in YYYY format")):
    cursor.execute("SELECT SUM(T1.num_students) - SUM(CAST(T1.num_students * T1.pct_international_students AS REAL) / 100) FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T2.university_name = ? AND T1.year BETWEEN ? AND ?", (university_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"domestic_students": []}
    return {"domestic_students": result[0]}

# Endpoint to get the count of universities with more than a specific number of students in a given year
@app.get("/v1/university/count_universities_by_year_and_students", operation_id="get_count_universities_by_year_and_students", summary="Retrieves the total count of universities that had more than the specified number of students in a given year. The year should be provided in YYYY format, and the number of students is a numerical value.")
async def get_count_universities_by_year_and_students(year: int = Query(..., description="Year in YYYY format"), num_students: int = Query(..., description="Number of students")):
    cursor.execute("SELECT COUNT(*) FROM university_year WHERE year = ? AND num_students > ?", (year, num_students))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country ID of a university by its name
@app.get("/v1/university/country_id_by_university_name", operation_id="get_country_id_by_university_name", summary="Retrieves the country ID associated with a specific university, identified by its name. The operation returns the country ID of the university that matches the provided name.")
async def get_country_id_by_university_name(university_name: str = Query(..., description="Name of the university")):
    cursor.execute("SELECT country_id FROM university WHERE university_name = ?", (university_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_id": []}
    return {"country_id": result[0]}

# Endpoint to get the ID of a ranking system by its name
@app.get("/v1/university/ranking_system_id_by_name", operation_id="get_ranking_system_id_by_name", summary="Retrieves the unique identifier of a specific ranking system, based on the provided system name.")
async def get_ranking_system_id_by_name(system_name: str = Query(..., description="Name of the ranking system")):
    cursor.execute("SELECT id FROM ranking_system WHERE system_name = ?", (system_name,))
    result = cursor.fetchone()
    if not result:
        return {"id": []}
    return {"id": result[0]}

# Endpoint to get the ID of a ranking criteria by its name
@app.get("/v1/university/ranking_criteria_id_by_name", operation_id="get_ranking_criteria_id_by_name", summary="Retrieves the unique identifier of a specific university ranking criteria based on its given name. The criteria name is used to locate the corresponding criteria record and return its associated ID.")
async def get_ranking_criteria_id_by_name(criteria_name: str = Query(..., description="Name of the ranking criteria")):
    cursor.execute("SELECT id FROM ranking_criteria WHERE criteria_name = ?", (criteria_name,))
    result = cursor.fetchone()
    if not result:
        return {"id": []}
    return {"id": result[0]}

# Endpoint to get the count of universities with a percentage of international students above a certain threshold in a specific year
@app.get("/v1/university/count_universities_by_international_students_and_year", operation_id="get_count_universities_by_international_students_and_year", summary="Retrieves the number of universities that have a percentage of international students exceeding a specified threshold in a given year.")
async def get_count_universities_by_international_students_and_year(pct_international_students: float = Query(..., description="Percentage of international students"), year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT COUNT(*) FROM university_year WHERE pct_international_students > ? AND year = ?", (pct_international_students, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of university rankings based on criteria ID, year, and score threshold
@app.get("/v1/university/count_university_rankings_by_criteria_year_score", operation_id="get_count_university_rankings_by_criteria_year_score", summary="Retrieves the total count of university rankings that meet the specified ranking criteria, year, and score threshold. This operation is useful for understanding the distribution of rankings based on these factors.")
async def get_count_university_rankings_by_criteria_year_score(ranking_criteria_id: int = Query(..., description="ID of the ranking criteria"), year: int = Query(..., description="Year in YYYY format"), score: float = Query(..., description="Score threshold")):
    cursor.execute("SELECT COUNT(*) FROM university_ranking_year WHERE ranking_criteria_id = ? AND year = ? AND score < ?", (ranking_criteria_id, year, score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the number of students in a specific university in a specific year
@app.get("/v1/university/num_students_by_university_year", operation_id="get_num_students_by_university_year", summary="Retrieves the total number of students enrolled in a specific university during a particular year. The operation requires the university's name and the year (in YYYY format) as input parameters to filter the data and provide accurate results.")
async def get_num_students_by_university_year(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT T1.num_students FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T2.university_name = ? AND T1.year = ?", (university_name, year))
    result = cursor.fetchone()
    if not result:
        return {"num_students": []}
    return {"num_students": result[0]}

# Endpoint to get the ratio of total students to student-staff ratio for a specific university in a specific year
@app.get("/v1/university/student_ratio_by_university_year", operation_id="get_student_ratio_by_university_year", summary="Retrieves the ratio of total students to student-staff ratio for a specific university in a specific year. This operation calculates the ratio based on the provided university name and year, providing insights into the student-staff balance at the given institution during the specified academic year.")
async def get_student_ratio_by_university_year(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT CAST(SUM(T1.num_students) AS REAL) / SUM(T1.student_staff_ratio) FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T2.university_name = ? AND T1.year = ?", (university_name, year))
    result = cursor.fetchone()
    if not result:
        return {"student_ratio": []}
    return {"student_ratio": result[0]}

# Endpoint to get the number of international students in a specific university in a specific year
@app.get("/v1/university/international_students_by_university_year", operation_id="get_international_students_by_university_year", summary="Retrieves the precise number of international students enrolled in a specific university during a particular year. The calculation is based on the total number of students and the percentage of international students in that year.")
async def get_international_students_by_university_year(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT CAST(T2.num_students * T2.pct_international_students AS REAL) / 100 FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id WHERE T1.university_name = ? AND T2.year = ?", (university_name, year))
    result = cursor.fetchone()
    if not result:
        return {"international_students": []}
    return {"international_students": result[0]}

# Endpoint to get the number of female students in a specific university for a given year
@app.get("/v1/university/female_students_count", operation_id="get_female_students_count", summary="Retrieves the estimated number of female students in a specific university for a given year. The calculation is based on the total number of students and the percentage of female students in that year.")
async def get_female_students_count(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT CAST(T2.num_students * T2.pct_female_students AS REAL) / 100 FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id WHERE T1.university_name = ? AND T2.year = ?", (university_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top university by total ranking score
@app.get("/v1/university/top_university_by_score", operation_id="get_top_university_by_score", summary="Retrieves the name of the university with the highest total ranking score, based on aggregated scores from multiple years.")
async def get_top_university_by_score():
    cursor.execute("SELECT T1.university_name FROM university AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.university_id GROUP BY T1.university_name ORDER BY SUM(T2.score) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the university with the lowest score for a specific ranking criteria in a given year
@app.get("/v1/university/lowest_score_university", operation_id="get_lowest_score_university", summary="Retrieves the name of the university with the lowest score for a specific ranking criteria in a given year. The operation considers the provided criteria name and year to determine the university with the lowest score, based on the available rankings data.")
async def get_lowest_score_university(criteria_name: str = Query(..., description="Name of the ranking criteria"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T3.university_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T1.criteria_name = ? AND T2.year = ? ORDER BY T2.score ASC LIMIT 1", (criteria_name, year))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the percentage of universities meeting a specific score threshold for a given ranking criteria in a specific year
@app.get("/v1/university/percentage_universities_meeting_score", operation_id="get_percentage_universities_meeting_score", summary="Retrieves the percentage of universities that meet or exceed a specified score threshold for a given ranking criteria in a particular year. The operation also returns the name of the university with the highest score that meets the threshold. The ranking criteria and year are used to filter the results.")
async def get_percentage_universities_meeting_score(criteria_name: str = Query(..., description="Name of the ranking criteria"), year: int = Query(..., description="Year"), score_threshold: int = Query(..., description="Score threshold")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.score > ? THEN 1 ELSE 0 END) AS REAL) / COUNT(*), ( SELECT T3.university_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T1.criteria_name = ? AND T2.year = ? AND T2.score > ? ORDER BY T2.score DESC LIMIT 1 ) AS max FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T1.criteria_name = ? AND T2.year = ?", (score_threshold, criteria_name, year, score_threshold, criteria_name, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": [], "max_university": []}
    return {"percentage": result[0], "max_university": result[1]}

# Endpoint to get the ranking criteria and scores for a specific university in a given year
@app.get("/v1/university/ranking_criteria_scores", operation_id="get_ranking_criteria_scores", summary="Retrieves the ranking criteria and corresponding scores for a specific university in a given year. The university is identified by its name, and the year parameter determines the year of the ranking data. This operation provides a comprehensive view of the university's performance across various ranking criteria.")
async def get_ranking_criteria_scores(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.criteria_name, T2.score FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T3.university_name = ? AND T2.year = ?", (university_name, year))
    result = cursor.fetchall()
    if not result:
        return {"criteria_scores": []}
    return {"criteria_scores": [{"criteria_name": row[0], "score": row[1]} for row in result]}

# Endpoint to get the average score for a specific ranking criteria in a given year
@app.get("/v1/university/average_score_criteria", operation_id="get_average_score_criteria", summary="Retrieves the average score for a specific ranking criteria in a given year. The operation calculates the average score based on the provided ranking criteria name and year, providing a comprehensive overview of the performance in that particular area during the specified year.")
async def get_average_score_criteria(criteria_name: str = Query(..., description="Name of the ranking criteria"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT AVG(T2.score) FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id WHERE T1.criteria_name = ? AND T2.year = ?", (criteria_name, year))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the university with the highest number of students in a specific year
@app.get("/v1/university/highest_students_university", operation_id="get_highest_students_university", summary="Retrieves the university with the highest number of students in a given year, along with its country. The year is specified as an input parameter.")
async def get_highest_students_university(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.university_name, T3.country_name FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T2.year = ? ORDER BY T2.num_students DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"university_name": [], "country_name": []}
    return {"university_name": result[0], "country_name": result[1]}

# Endpoint to get the number of students for a specific university in a given year
@app.get("/v1/university/num_students", operation_id="get_num_students", summary="Retrieves the total number of students enrolled in a specific university during a given year. The operation requires the year and the unique identifier of the university as input parameters.")
async def get_num_students(year: int = Query(..., description="Year"), university_id: int = Query(..., description="University ID")):
    cursor.execute("SELECT num_students FROM university_year WHERE year = ? AND university_id = ?", (year, university_id))
    result = cursor.fetchone()
    if not result:
        return {"num_students": []}
    return {"num_students": result[0]}

# Endpoint to get the university IDs with a student-staff ratio greater than a specified value in a given year
@app.get("/v1/university/university_ids_by_student_staff_ratio", operation_id="get_university_ids_by_student_staff_ratio", summary="Retrieve the identifiers of universities that have a student-to-staff ratio surpassing a specified threshold in a particular year. The year and the minimum student-to-staff ratio are provided as input parameters.")
async def get_university_ids_by_student_staff_ratio(year: int = Query(..., description="Year"), student_staff_ratio: float = Query(..., description="Student-staff ratio")):
    cursor.execute("SELECT university_id FROM university_year WHERE year = ? AND student_staff_ratio > ?", (year, student_staff_ratio))
    result = cursor.fetchall()
    if not result:
        return {"university_ids": []}
    return {"university_ids": [row[0] for row in result]}

# Endpoint to get university IDs based on year and ordered by percentage of female students
@app.get("/v1/university/university_ids_by_year_ordered_by_female_students", operation_id="get_university_ids", summary="Retrieves a list of university IDs for a given year, sorted by the percentage of female students in descending order. The number of results returned is limited by the provided limit parameter.")
async def get_university_ids(year: int = Query(..., description="Year"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT university_id FROM university_year WHERE year = ? ORDER BY pct_female_students DESC LIMIT ?", (year, limit))
    result = cursor.fetchall()
    if not result:
        return {"university_ids": []}
    return {"university_ids": [row[0] for row in result]}

# Endpoint to get the year with the highest number of students for a specific university
@app.get("/v1/university/year_with_highest_students_by_university_id", operation_id="get_year_with_highest_students", summary="Retrieves the year with the highest number of students for a given university. The operation returns the year with the most students, based on the provided university ID. The 'limit' parameter can be used to restrict the number of results returned.")
async def get_year_with_highest_students(university_id: int = Query(..., description="University ID"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT year FROM university_year WHERE university_id = ? ORDER BY num_students DESC LIMIT ?", (university_id, limit))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the university name with the highest percentage of international students for a specific year
@app.get("/v1/university/university_name_by_year_ordered_by_international_students", operation_id="get_university_name", summary="Retrieves the name of the university with the highest percentage of international students for a given year, with the option to limit the number of results returned.")
async def get_university_name(year: int = Query(..., description="Year"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? ORDER BY T1.pct_international_students DESC LIMIT ?", (year, limit))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the university name with the highest score for a specific ranking criteria and year
@app.get("/v1/university/university_name_by_ranking_criteria_year_ordered_by_score", operation_id="get_university_name_by_ranking", summary="Retrieves the name of the university with the highest score for a given ranking criteria and year. The operation returns a single university name or a specified number of top-scoring universities, depending on the provided limit parameter. The ranking criteria and year are used to filter the results.")
async def get_university_name_by_ranking(criteria_name: str = Query(..., description="Ranking criteria name"), year: int = Query(..., description="Year"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT T3.university_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T1.criteria_name = ? AND T2.year = ? ORDER BY T2.score DESC LIMIT ?", (criteria_name, year, limit))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get university names based on ranking criteria, year, and score
@app.get("/v1/university/university_names_by_ranking_criteria_year_score", operation_id="get_university_names_by_ranking", summary="Retrieves the names of universities that meet a specific ranking criteria, year, and score threshold. The operation filters universities based on the provided criteria name, year, and score, returning only those that have a score greater than the specified value.")
async def get_university_names_by_ranking(criteria_name: str = Query(..., description="Ranking criteria name"), year: int = Query(..., description="Year"), score: float = Query(..., description="Score")):
    cursor.execute("SELECT T3.university_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T1.criteria_name = ? AND T2.year = ? AND T2.score > ?", (criteria_name, year, score))
    result = cursor.fetchall()
    if not result:
        return {"university_names": []}
    return {"university_names": [row[0] for row in result]}

# Endpoint to get the count of universities based on ranking criteria, year, and score
@app.get("/v1/university/count_universities_by_ranking_criteria_year_score", operation_id="get_count_universities_by_ranking", summary="Retrieves the number of universities that meet a specific ranking criteria, year, and score threshold. The criteria_name parameter filters universities by a particular ranking category, while the year parameter specifies the year of the ranking. The score parameter sets a minimum score threshold for the universities to be included in the count.")
async def get_count_universities_by_ranking(criteria_name: str = Query(..., description="Ranking criteria name"), year: int = Query(..., description="Year"), score: float = Query(..., description="Score")):
    cursor.execute("SELECT COUNT(*) FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T1.criteria_name = ? AND T2.year = ? AND T2.score > ?", (criteria_name, year, score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get criteria names for a specific university and year
@app.get("/v1/university/criteria_names_by_university_year", operation_id="get_criteria_names", summary="Retrieves the names of the criteria used to rank a specific university in a given year. The criteria are determined by the ranking system and may vary between universities and years.")
async def get_criteria_names(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.criteria_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T3.university_name = ? AND T2.year = ?", (university_name, year))
    result = cursor.fetchall()
    if not result:
        return {"criteria_names": []}
    return {"criteria_names": [row[0] for row in result]}

# Endpoint to get university names based on criteria name, year, and score
@app.get("/v1/university/university_names_by_criteria_year_score", operation_id="get_university_names", summary="Retrieves the names of universities that meet the specified criteria, year, and score. The criteria name, year, and score are used to filter the results, providing a targeted list of university names that match the given conditions.")
async def get_university_names(criteria_name: str = Query(..., description="Name of the criteria"), year: int = Query(..., description="Year"), score: int = Query(..., description="Score")):
    cursor.execute("SELECT T3.university_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T1.criteria_name = ? AND T2.year = ? AND T2.score = ?", (criteria_name, year, score))
    result = cursor.fetchall()
    if not result:
        return {"university_names": []}
    return {"university_names": [row[0] for row in result]}

# Endpoint to get the percentage of universities in a specific country based on criteria name, year, and score
@app.get("/v1/university/percentage_universities_by_country_criteria_year_score", operation_id="get_percentage_universities", summary="Retrieves the percentage of universities in a specified country that meet the given criteria, year, and minimum score. The calculation is based on the total number of universities in the database.")
async def get_percentage_universities(country_name: str = Query(..., description="Name of the country"), criteria_name: str = Query(..., description="Name of the criteria"), year: int = Query(..., description="Year"), min_score: int = Query(..., description="Minimum score")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T4.country_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) AS per FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id INNER JOIN country AS T4 ON T4.id = T3.country_id WHERE T1.criteria_name = ? AND T2.year = ? AND T2.score > ?", (country_name, criteria_name, year, min_score))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the country name by ID
@app.get("/v1/university/country_name_by_id", operation_id="get_country_name", summary="Retrieves the name of a specific country based on its unique identifier. The operation requires the country's ID as input and returns the corresponding country name.")
async def get_country_name(country_id: int = Query(..., description="ID of the country")):
    cursor.execute("SELECT country_name FROM country WHERE id = ?", (country_id,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the count of university records for a specific university and year
@app.get("/v1/university/count_by_university_and_year", operation_id="get_count_by_university_and_year", summary="Retrieves the total number of records for a specific university in a given year. The operation requires the university's name and the year as input parameters.")
async def get_count_by_university_and_year(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT COUNT(*) FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id WHERE T1.university_name = ? AND T2.year = ?", (university_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the score of a specific university for a specific ranking criteria and year
@app.get("/v1/university/score_by_university_criteria_year", operation_id="get_score_by_university_criteria_year", summary="Retrieves the score of a specific university for a given ranking criteria and year. The score is determined by comparing the university's performance against the specified criteria for the selected year. The input parameters include the university name, ranking criteria, and the year for which the score is requested.")
async def get_score_by_university_criteria_year(university_name: str = Query(..., description="Name of the university"), criteria_name: str = Query(..., description="Name of the ranking criteria"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T2.score FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T3.university_name = ? AND T1.criteria_name = ? AND T2.year = ?", (university_name, criteria_name, year))
    result = cursor.fetchone()
    if not result:
        return {"score": []}
    return {"score": result[0]}

# Endpoint to get the top ranking criteria for a specific university and year
@app.get("/v1/university/top_ranking_criteria_by_university_year", operation_id="get_top_ranking_criteria_by_university_year", summary="Retrieves the top-ranked criteria for a given university in a specific year. The criteria are determined by the highest score in the university's ranking for that year. The university's name and the year are required as input parameters.")
async def get_top_ranking_criteria_by_university_year(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.criteria_name FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id INNER JOIN university AS T3 ON T3.id = T2.university_id WHERE T3.university_name = ? AND T2.year = ? ORDER BY T2.score DESC LIMIT 1", (university_name, year))
    result = cursor.fetchone()
    if not result:
        return {"criteria_name": []}
    return {"criteria_name": result[0]}

# Endpoint to get the number of international students for a specific university and year
@app.get("/v1/university/num_international_students_by_university_year", operation_id="get_num_international_students_by_university_year", summary="Retrieves the estimated number of international students enrolled in a specific university during a given year. The calculation is based on the total number of students and the percentage of international students in that university for the specified year.")
async def get_num_international_students_by_university_year(year: int = Query(..., description="Year"), university_name: str = Query(..., description="Name of the university")):
    cursor.execute("SELECT CAST(T1.num_students * T1.pct_international_students AS REAL) / 100 FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? AND T2.university_name = ?", (year, university_name))
    result = cursor.fetchone()
    if not result:
        return {"num_international_students": []}
    return {"num_international_students": result[0]}

# Endpoint to get the university name with the least number of students in a given year
@app.get("/v1/university/least_students_university", operation_id="get_least_students_university", summary="Retrieves the name of the university with the smallest student population in a specified year. The year parameter is used to filter the data.")
async def get_least_students_university(year: int = Query(..., description="Year to filter the data")):
    cursor.execute("SELECT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? ORDER BY T1.num_students ASC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get the ratio of the number of students between two universities in a given year
@app.get("/v1/university/student_ratio_between_universities", operation_id="get_student_ratio_between_universities", summary="Retrieves the ratio of the number of students between two specified universities in a given year. The operation calculates the ratio by summing the number of students for each university in the specified year and then dividing the sum of the first university by the sum of the second university.")
async def get_student_ratio_between_universities(university_name_1: str = Query(..., description="Name of the first university"), university_name_2: str = Query(..., description="Name of the second university"), year: int = Query(..., description="Year to filter the data")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.university_name = ? THEN T1.num_students ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.university_name = ? THEN T1.num_students ELSE 0 END) FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ?", (university_name_1, university_name_2, year))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the average count of specific ranking systems
@app.get("/v1/university/average_ranking_systems_count", operation_id="get_average_ranking_systems_count", summary="Retrieves the average count of three specified ranking systems. The operation calculates the sum of instances for each of the three ranking systems and divides it by three to provide the average count. This endpoint is useful for obtaining a general understanding of the prevalence of these ranking systems in the university context.")
async def get_average_ranking_systems_count(system_name_1: str = Query(..., description="Name of the first ranking system"), system_name_2: str = Query(..., description="Name of the second ranking system"), system_name_3: str = Query(..., description="Name of the third ranking system")):
    cursor.execute("SELECT (SUM(CASE WHEN T1.system_name = ? THEN 1 ELSE 0 END) + SUM(CASE WHEN T1.system_name = ? THEN 1 ELSE 0 END) + SUM(CASE WHEN T1.system_name = ? THEN 1 ELSE 0 END)) / 3 FROM ranking_system AS T1 INNER JOIN ranking_criteria AS T2 ON T1.id = T2.ranking_system_id", (system_name_1, system_name_2, system_name_3))
    result = cursor.fetchone()
    if not result:
        return {"average_count": []}
    return {"average_count": result[0]}

# Endpoint to get the average number of students in a given year
@app.get("/v1/university/average_students_per_year", operation_id="get_average_students_per_year", summary="Retrieves the average number of students enrolled in the university for a specific year. The year parameter is used to filter the data and calculate the average.")
async def get_average_students_per_year(year: int = Query(..., description="Year to filter the data")):
    cursor.execute("SELECT AVG(num_students) FROM university_year WHERE year = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"average_students": []}
    return {"average_students": result[0]}

# Endpoint to get the score of a university in a given year
@app.get("/v1/university/university_score_by_year", operation_id="get_university_score_by_year", summary="Get the score of a university in a given year")
async def get_university_score_by_year(year: int = Query(..., description="Year to filter the data"), university_id: int = Query(..., description="ID of the university")):
    cursor.execute("SELECT score FROM university_ranking_year WHERE year = ? AND university_id = ?", (year, university_id))
    result = cursor.fetchone()
    if not result:
        return {"score": []}
    return {"score": result[0]}

# Endpoint to get the country ID by country name
@app.get("/v1/university/country_id_by_name", operation_id="get_country_id_by_name", summary="Retrieves the unique identifier of a country based on its name. The operation requires the country's name as input and returns the corresponding country ID.")
async def get_country_id_by_name(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT id FROM country WHERE country_name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_id": []}
    return {"country_id": result[0]}

# Endpoint to get the university ID with the highest percentage of international students
@app.get("/v1/university/highest_international_students", operation_id="get_highest_international_students", summary="Retrieves the unique identifier of the university with the highest percentage of international students. The operation ranks universities based on the proportion of international students and returns the top-ranked university's identifier.")
async def get_highest_international_students():
    cursor.execute("SELECT university_id FROM university_year ORDER BY pct_international_students DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"university_id": []}
    return {"university_id": result[0]}

# Endpoint to get the ranking criteria name by ID
@app.get("/v1/university/ranking_criteria_by_id", operation_id="get_ranking_criteria_by_id", summary="Retrieves the name of a specific university ranking criteria based on its unique identifier. The criteria_id input parameter is used to specify the desired ranking criteria.")
async def get_ranking_criteria_by_id(criteria_id: int = Query(..., description="ID of the ranking criteria")):
    cursor.execute("SELECT criteria_name FROM ranking_criteria WHERE id = ?", (criteria_id,))
    result = cursor.fetchone()
    if not result:
        return {"criteria_name": []}
    return {"criteria_name": result[0]}

# Endpoint to get the average score of universities in a given year
@app.get("/v1/university/average_score_by_year", operation_id="get_average_score_by_year", summary="Retrieves the average score of universities for a specified year. The year parameter is used to filter the data and calculate the average score.")
async def get_average_score_by_year(year: int = Query(..., description="Year to filter the data")):
    cursor.execute("SELECT AVG(score) FROM university_ranking_year WHERE year = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the total number of female students in a given year range for a specific university
@app.get("/v1/university/total_female_students_by_year_range", operation_id="get_total_female_students_by_year_range", summary="Retrieves the total count of female students enrolled in a specific university within a specified year range. The calculation is based on the total number of students and the percentage of female students in each year. The year range and university ID are provided as input parameters.")
async def get_total_female_students_by_year_range(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), university_id: int = Query(..., description="ID of the university")):
    cursor.execute("SELECT SUM(CAST(num_students * pct_female_students AS REAL) / 100) FROM university_year WHERE year BETWEEN ? AND ? AND university_id = ?", (start_year, end_year, university_id))
    result = cursor.fetchone()
    if not result:
        return {"total_female_students": []}
    return {"total_female_students": result[0]}

# Endpoint to get the average score of a university within a specific year range
@app.get("/v1/university/average_score_by_year_range", operation_id="get_average_score_by_year_range", summary="Retrieves the average score of a specific university for a given year range. The operation calculates the mean score from the university's ranking data within the specified start and end years.")
async def get_average_score_by_year_range(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), university_id: int = Query(..., description="University ID")):
    cursor.execute("SELECT AVG(score) FROM university_ranking_year WHERE year BETWEEN ? AND ? AND university_id = ?", (start_year, end_year, university_id))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the score of the university with the highest number of students in a specific year
@app.get("/v1/university/highest_students_score_by_year", operation_id="get_highest_students_score_by_year", summary="Retrieves the score of the university with the highest number of students in a given year. The score is determined by joining the university_year and university_ranking_year tables on the university_id field, filtering for the specified year, and ordering the results by the number of students in descending order. The score of the top university is then returned.")
async def get_highest_students_score_by_year(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T2.score FROM university_year AS T1 INNER JOIN university_ranking_year AS T2 ON T1.university_id = T2.university_id WHERE T1.year = ? ORDER BY T1.num_students DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"score": []}
    return {"score": result[0]}

# Endpoint to get distinct ranking criteria names for a specific university and score
@app.get("/v1/university/ranking_criteria_by_university_and_score", operation_id="get_ranking_criteria_by_university_and_score", summary="Retrieves unique ranking criteria names associated with a specific university and score. The university is identified by its name, and the score is used to filter the criteria. This operation does not return any ranking criteria that are not distinct.")
async def get_ranking_criteria_by_university_and_score(university_name: str = Query(..., description="University name"), score: int = Query(..., description="Score")):
    cursor.execute("SELECT DISTINCT T3.criteria_name FROM university AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.university_id INNER JOIN ranking_criteria AS T3 ON T3.id = T2.ranking_criteria_id WHERE T1.university_name = ? AND T2.score = ?", (university_name, score))
    result = cursor.fetchall()
    if not result:
        return {"criteria_names": []}
    return {"criteria_names": [row[0] for row in result]}

# Endpoint to get universities in a specific country
@app.get("/v1/university/universities_by_country", operation_id="get_universities_by_country", summary="Retrieves a list of universities located in the specified country. The operation returns the name and unique identifier of each university.")
async def get_universities_by_country(country_name: str = Query(..., description="Country name")):
    cursor.execute("SELECT T1.university_name, T1.id FROM university AS T1 INNER JOIN country AS T2 ON T1.country_id = T2.id WHERE T2.country_name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"universities": []}
    return {"universities": [{"university_name": row[0], "university_id": row[1]} for row in result]}

# Endpoint to get the count of ranking criteria for a specific ranking system
@app.get("/v1/university/ranking_criteria_count_by_system", operation_id="get_ranking_criteria_count_by_system", summary="Retrieves the total number of ranking criteria associated with a specific ranking system. The system is identified by its unique name.")
async def get_ranking_criteria_count_by_system(system_name: str = Query(..., description="Ranking system name")):
    cursor.execute("SELECT COUNT(*) FROM ranking_system AS T1 INNER JOIN ranking_criteria AS T2 ON T1.id = T2.ranking_system_id WHERE T1.system_name = ?", (system_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the university name and score for a specific university ID
@app.get("/v1/university/university_name_and_score_by_id", operation_id="get_university_name_and_score_by_id", summary="Get the university name and score for a specific university ID")
async def get_university_name_and_score_by_id(university_id: int = Query(..., description="University ID")):
    cursor.execute("SELECT T2.university_name, T1.score FROM university_ranking_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T2.id = ?", (university_id,))
    result = cursor.fetchone()
    if not result:
        return {"university_name": [], "score": []}
    return {"university_name": result[0], "score": result[1]}

# Endpoint to get distinct university names with scores below a specific value
@app.get("/v1/university/universities_with_score_below", operation_id="get_universities_with_score_below", summary="Retrieve a list of unique university names that have a score below the provided threshold. This operation filters the university ranking data for a specific year and returns the distinct university names that meet the score criteria.")
async def get_universities_with_score_below(score: int = Query(..., description="Score threshold")):
    cursor.execute("SELECT DISTINCT T2.university_name FROM university_ranking_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.score < ?", (score,))
    result = cursor.fetchall()
    if not result:
        return {"university_names": []}
    return {"university_names": [row[0] for row in result]}

# Endpoint to get the university with the highest difference in female student percentage and total students
@app.get("/v1/university/highest_female_student_difference", operation_id="get_highest_female_student_difference", summary="Retrieves the university with the highest difference between the number of female students and the total number of students. The calculation is based on the percentage of female students and the total number of students in each university for a given year.")
async def get_highest_female_student_difference():
    cursor.execute("SELECT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id ORDER BY T1.num_students * T1.pct_female_students / 100 - T1.num_students DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"university_name": []}
    return {"university_name": result[0]}

# Endpoint to get distinct country names based on university ranking score and year
@app.get("/v1/university/distinct_countries_by_score_year", operation_id="get_distinct_countries_by_score_year", summary="Retrieves a list of unique country names where the university ranking score is below a specified threshold and the year matches the provided value. This operation is useful for identifying countries with universities that have not met a certain score in a given year.")
async def get_distinct_countries_by_score_year(score: int = Query(..., description="Score threshold"), year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT DISTINCT T3.country_name FROM university AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T2.score < ? AND T2.year = ?", (score, year))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get the number of male students for a specific university and year
@app.get("/v1/university/male_student_count", operation_id="get_male_student_count", summary="Retrieves the count of male students enrolled in a specific university for a given year. The calculation is based on the total number of students and the percentage of female students. The university is identified by its name, and the year is specified in the YYYY format.")
async def get_male_student_count(university_name: str = Query(..., description="Name of the university"), year: int = Query(..., description="Year in YYYY format")):
    cursor.execute("SELECT CAST((T1.num_students - (T1.num_students * T1.pct_female_students)) AS REAL) / 100 FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T2.university_name = ? AND T1.year = ?", (university_name, year))
    result = cursor.fetchone()
    if not result:
        return {"male_student_count": []}
    return {"male_student_count": result[0]}

# Endpoint to get distinct university names based on the number of students range
@app.get("/v1/university/distinct_universities_by_student_range", operation_id="get_distinct_universities_by_student_range", summary="Retrieves a list of unique university names that have a student population within the specified range. The range is defined by the minimum and maximum number of students provided as input parameters.")
async def get_distinct_universities_by_student_range(min_students: int = Query(..., description="Minimum number of students"), max_students: int = Query(..., description="Maximum number of students")):
    cursor.execute("SELECT DISTINCT T2.university_name FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.num_students BETWEEN ? AND ?", (min_students, max_students))
    result = cursor.fetchall()
    if not result:
        return {"university_names": []}
    return {"university_names": [row[0] for row in result]}

# Endpoint to get the year with the highest ranking score for a specific university
@app.get("/v1/university/highest_ranking_year", operation_id="get_highest_ranking_year", summary="Retrieves the year in which a specific university achieved its highest ranking score. The ranking is determined by comparing scores across all years for the given university.")
async def get_highest_ranking_year(university_name: str = Query(..., description="Name of the university")):
    cursor.execute("SELECT T1.year FROM university_ranking_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T2.university_name = ? ORDER BY T1.score DESC LIMIT 1", (university_name,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the average ranking score for a specific university over a range of years
@app.get("/v1/university/average_ranking_score", operation_id="get_average_ranking_score", summary="Retrieves the average ranking score for a specific university over a specified range of years. The operation calculates the average score from the university's annual rankings, which are filtered by the provided university name and the inclusive range of years.")
async def get_average_ranking_score(university_name: str = Query(..., description="Name of the university"), start_year: int = Query(..., description="Start year in YYYY format"), end_year: int = Query(..., description="End year in YYYY format")):
    cursor.execute("SELECT AVG(T1.score) FROM university_ranking_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T2.university_name = ? AND T1.year BETWEEN ? AND ?", (university_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the country name and number of female students for a specific university and year
@app.get("/v1/university/country_female_students", operation_id="get_country_female_students", summary="Retrieves the country name and the estimated number of female students for a specific university in a given year. The calculation is based on the total number of students and the percentage of female students in that year.")
async def get_country_female_students(year: int = Query(..., description="Year in YYYY format"), university_id: int = Query(..., description="University ID")):
    cursor.execute("SELECT T3.country_name, CAST(T2.num_students * T2.pct_female_students AS REAL) / 100 FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T2.year = ? AND T1.id = ?", (year, university_id))
    result = cursor.fetchone()
    if not result:
        return {"country_name": [], "female_students": []}
    return {"country_name": result[0], "female_students": result[1]}

# Endpoint to get the count of universities based on ranking criteria and score
@app.get("/v1/university/count_by_ranking_criteria_score", operation_id="get_count_by_ranking_criteria_score", summary="Retrieves the total number of universities that meet the specified ranking criteria and score. The operation considers the provided score and criteria name to filter the universities and calculate the count.")
async def get_count_by_ranking_criteria_score(score: int = Query(..., description="Score"), criteria_name: str = Query(..., description="Criteria name")):
    cursor.execute("SELECT COUNT(*) FROM ranking_criteria AS T1 INNER JOIN university_ranking_year AS T2 ON T1.id = T2.ranking_criteria_id WHERE T2.score = ? AND T1.criteria_name = ?", (score, criteria_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of female students in a specific country for a given year
@app.get("/v1/university/percentage_female_students", operation_id="get_percentage_female_students", summary="Retrieves the percentage of female students in a given country for a specified year. This operation calculates the total number of female students and the total number of students in the specified year, then returns the percentage of female students based on these figures.")
async def get_percentage_female_students(country_name: str = Query(..., description="Name of the country"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT SUM(CAST(T2.pct_female_students * T2.num_students AS REAL) / 100) * 100 / SUM(T2.num_students) FROM university AS T1 INNER JOIN university_year AS T2 ON T1.id = T2.university_id INNER JOIN country AS T3 ON T3.id = T1.country_id WHERE T3.country_name = ? AND T2.year = ?", (country_name, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the number of domestic students in a specific university over a range of years
@app.get("/v1/university/domestic_students_count", operation_id="get_domestic_students_count", summary="Retrieves the total count of domestic students in a specified university for a given range of years. The calculation is based on the total number of students and the percentage of international students, excluding the latter from the count. The operation requires the university name and the start and end years of the range.")
async def get_domestic_students_count(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), university_name: str = Query(..., description="Name of the university")):
    cursor.execute("SELECT SUM(T1.num_students) - SUM(CAST(T1.num_students * T1.pct_international_students AS REAL) / 100) FROM university_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year BETWEEN ? AND ? AND T2.university_name = ?", (start_year, end_year, university_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get university names with scores below a certain threshold for a given year
@app.get("/v1/university/universities_below_score_threshold", operation_id="get_universities_below_score_threshold", summary="Retrieves the names of universities that scored below the average score for a given year. The average score is calculated as twice the mean score of all universities in the specified year. The score is represented as a percentage.")
async def get_universities_below_score_threshold(year: int = Query(..., description="Year")):
    cursor.execute("SELECT T2.university_name FROM university_ranking_year AS T1 INNER JOIN university AS T2 ON T1.university_id = T2.id WHERE T1.year = ? AND T1.score * 100 < ( SELECT AVG(score) * 28 FROM university_ranking_year WHERE year = ? )", (year, year))
    result = cursor.fetchall()
    if not result:
        return {"universities": []}
    return {"universities": [row[0] for row in result]}

api_calls = [
    "/v1/university/count_university_years?num_students=80000&year=2011",
    "/v1/university/ranking_system_id_by_criteria?criteria_name=Award",
    "/v1/university/count_universities_by_name?university_name=State",
    "/v1/university/max_student_staff_ratio",
    "/v1/university/count_ranking_criteria?ranking_system_id=3",
    "/v1/university/university_ids_by_international_students?pct_international_students=1&start_year=2011&end_year=2015",
    "/v1/university/country_with_most_universities",
    "/v1/university/university_with_highest_international_students",
    "/v1/university/top_university_by_ranking_criteria?criteria_name=Publications%20Rank&year=2014&ranking_criteria_id=17",
    "/v1/university/university_with_most_students",
    "/v1/university/count_universities_by_country?country_name=United%20States%20of%20America",
    "/v1/university/top_university_by_criteria?criteria_name=Citations&year=2016&criteria_id=4&country_name=Australia",
    "/v1/university/count_universities_by_criteria_year_score?criteria_name=Award&start_year=2005&end_year=2015&score=0",
    "/v1/university/country_of_university?university_name=University%20of%20Oxford",
    "/v1/university/count_universities_by_name_score_criteria?university_name=Yale%20University&min_score=10&criteria_name=Quality%20of%20Education%20Rank",
    "/v1/university/criteria_by_ranking_system?system_name=Center%20for%20World%20University%20Rankings",
    "/v1/university/universities_by_student_count_year?min_students=50000&year=2012",
    "/v1/university/countries_by_international_students_year_range?min_pct_international_students=50&start_year=2011&end_year=2016",
    "/v1/university/count_universities_female_students_year?country_name=United%20States%20of%20America&year=2016&min_female_students=20000",
    "/v1/university/top_universities_by_international_students?limit=5",
    "/v1/university/highest_student_staff_ratio",
    "/v1/university/lowest_num_students_year",
    "/v1/university/avg_pct_female_students",
    "/v1/university/international_students_count?year=2013&university_id=20",
    "/v1/university/id_by_name?university_name=Harvard%20University",
    "/v1/university/id_by_ranking_score_year?score=100&year=2011",
    "/v1/university/ranking_system_by_criteria?criteria_name=Quality%20of%20Education%20Rank",
    "/v1/university/student_staff_ratio?university_name=Harvard%20University&year=2012",
    "/v1/university/country_by_university_id?university_id=112",
    "/v1/university/total_students_by_country?country_name=Sweden",
    "/v1/university/ranking_criteria_ids?university_name=Brown%20University&year=2014",
    "/v1/university/university_names_by_country?country_name=Spain",
    "/v1/university/ranking_criteria_names?university_id=32&year=2015",
    "/v1/university/average_score_by_country?country_name=Brazil",
    "/v1/university/country_id_highest_students?year=2014",
    "/v1/university/international_students_score?year=2015&university_id=100",
    "/v1/university/total_students_by_score_year?score=98&year=2013",
    "/v1/university/countries_above_avg_students?year=2013&percentage=98",
    "/v1/university/avg_pct_international_students?score=80&year=2015",
    "/v1/university/total_students_by_year?year=2011",
    "/v1/university/sum_female_students_by_year?year=2011",
    "/v1/university/university_id_max_students_by_year?year=2011",
    "/v1/university/count_universities_by_criteria?year=2011&min_students=50000&min_pct_international=10",
    "/v1/university/university_id_max_pct_female_students_by_year?year=2012",
    "/v1/university/university_name_max_score_by_year?year=2012",
    "/v1/university/university_name_max_students_by_year?year=2011",
    "/v1/university/university_name_max_score_by_criteria?criteria_name=Teaching",
    "/v1/university/pct_international_students_by_university_year?year=2011&university_name=Harvard%20University",
    "/v1/university/num_female_students_by_university_year?year=2011&university_name=Stanford%20University",
    "/v1/university/country_name_by_university?university_name=Harvard%20University",
    "/v1/university/top_university_by_country?country_name=Argentina",
    "/v1/university/universities_by_year_and_students?year=2011&num_students=100000",
    "/v1/university/count_criteria_by_ranking_system?system_name=Center%20for%20World%20University%20Rankings",
    "/v1/university/count_universities_by_score_and_year?score=90&year=2011",
    "/v1/university/domestic_students_by_university_and_year_range?university_name=Harvard%20University&start_year=2011&end_year=2012",
    "/v1/university/count_universities_by_year_and_students?year=2011&num_students=30000",
    "/v1/university/country_id_by_university_name?university_name=University%20of%20Tokyo",
    "/v1/university/ranking_system_id_by_name?system_name=Center%20for%20World%20University%20Rankings",
    "/v1/university/ranking_criteria_id_by_name?criteria_name=Publications%20Rank",
    "/v1/university/count_universities_by_international_students_and_year?pct_international_students=30&year=2013",
    "/v1/university/count_university_rankings_by_criteria_year_score?ranking_criteria_id=6&year=2011&score=50",
    "/v1/university/num_students_by_university_year?university_name=Yale%20University&year=2016",
    "/v1/university/student_ratio_by_university_year?university_name=University%20of%20Auckland&year=2015",
    "/v1/university/international_students_by_university_year?university_name=Harvard%20University&year=2012",
    "/v1/university/female_students_count?university_name=Arizona%20State%20University&year=2014",
    "/v1/university/top_university_by_score",
    "/v1/university/lowest_score_university?criteria_name=Teaching&year=2011",
    "/v1/university/percentage_universities_meeting_score?criteria_name=International&year=2016&score_threshold=80",
    "/v1/university/ranking_criteria_scores?university_name=Harvard%20University&year=2005",
    "/v1/university/average_score_criteria?criteria_name=Alumni&year=2008",
    "/v1/university/highest_students_university?year=2015",
    "/v1/university/num_students?year=2011&university_id=1",
    "/v1/university/university_ids_by_student_staff_ratio?year=2011&student_staff_ratio=15",
    "/v1/university/university_ids_by_year_ordered_by_female_students?year=2011&limit=3",
    "/v1/university/year_with_highest_students_by_university_id?university_id=1&limit=1",
    "/v1/university/university_name_by_year_ordered_by_international_students?year=2011&limit=1",
    "/v1/university/university_name_by_ranking_criteria_year_ordered_by_score?criteria_name=Teaching&year=2011&limit=1",
    "/v1/university/university_names_by_ranking_criteria_year_score?criteria_name=Teaching&year=2011&score=90",
    "/v1/university/count_universities_by_ranking_criteria_year_score?criteria_name=Teaching&year=2011&score=90",
    "/v1/university/criteria_names_by_university_year?university_name=Harvard%20University&year=2011",
    "/v1/university/university_names_by_criteria_year_score?criteria_name=Teaching&year=2011&score=98",
    "/v1/university/percentage_universities_by_country_criteria_year_score?country_name=United%20States%20of%20America&criteria_name=Teaching&year=2011&min_score=90",
    "/v1/university/country_name_by_id?country_id=66",
    "/v1/university/count_by_university_and_year?university_name=University%20of%20Michigan&year=2011",
    "/v1/university/score_by_university_criteria_year?university_name=Chosun%20University&criteria_name=Influence%20Rank&year=2015",
    "/v1/university/top_ranking_criteria_by_university_year?university_name=University%20of%20Southampton&year=2015",
    "/v1/university/num_international_students_by_university_year?year=2013&university_name=University%20of%20Wisconsin-Madison",
    "/v1/university/least_students_university?year=2015",
    "/v1/university/student_ratio_between_universities?university_name_1=University%20of%20Ottawa&university_name_2=Joseph%20Fourier%20University&year=2013",
    "/v1/university/average_ranking_systems_count?system_name_1=Center%20for%20World%20University%20Rankings&system_name_2=Shanghai%20Ranking&system_name_3=Times%20Higher%20Education%20World%20University%20Ranking",
    "/v1/university/average_students_per_year?year=2012",
    "/v1/university/university_score_by_year?year=2015&university_id=68",
    "/v1/university/country_id_by_name?country_name=Cyprus",
    "/v1/university/highest_international_students",
    "/v1/university/ranking_criteria_by_id?criteria_id=13",
    "/v1/university/average_score_by_year?year=2012",
    "/v1/university/total_female_students_by_year_range?start_year=2011&end_year=2013&university_id=40",
    "/v1/university/average_score_by_year_range?start_year=2013&end_year=2015&university_id=79",
    "/v1/university/highest_students_score_by_year?year=2011",
    "/v1/university/ranking_criteria_by_university_and_score?university_name=Harvard%20University&score=100",
    "/v1/university/universities_by_country?country_name=Turkey",
    "/v1/university/ranking_criteria_count_by_system?system_name=Shanghai%20Ranking",
    "/v1/university/university_name_and_score_by_id?university_id=124",
    "/v1/university/universities_with_score_below?score=50",
    "/v1/university/highest_female_student_difference",
    "/v1/university/distinct_countries_by_score_year?score=70&year=2016",
    "/v1/university/male_student_count?university_name=Emory%20University&year=2011",
    "/v1/university/distinct_universities_by_student_range?min_students=400&max_students=1000",
    "/v1/university/highest_ranking_year?university_name=Brown%20University",
    "/v1/university/average_ranking_score?university_name=Emory%20University&start_year=2011&end_year=2016",
    "/v1/university/country_female_students?year=2011&university_id=23",
    "/v1/university/count_by_ranking_criteria_score?score=40&criteria_name=Teaching",
    "/v1/university/percentage_female_students?country_name=United%20States%20of%20America&year=2016",
    "/v1/university/domestic_students_count?start_year=2011&end_year=2014&university_name=University%20of%20Tokyo",
    "/v1/university/universities_below_score_threshold?year=2015"
]
