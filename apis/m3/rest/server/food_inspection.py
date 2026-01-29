from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/food_inspection/food_inspection.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of businesses based on owner state
@app.get("/v1/food_inspection/business_count_by_owner_state", operation_id="get_business_count_by_owner_state", summary="Retrieves the total number of businesses associated with a given owner state. The owner state is a crucial attribute that identifies the location of the business owner.")
async def get_business_count_by_owner_state(owner_state: str = Query(..., description="Owner state of the business")):
    cursor.execute("SELECT COUNT(owner_state) FROM businesses WHERE owner_state = ?", (owner_state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of inspections based on score
@app.get("/v1/food_inspection/inspection_count_by_score", operation_id="get_inspection_count_by_score", summary="Retrieves the total number of inspections that have a specified score. The score parameter is used to filter the inspections and calculate the count.")
async def get_inspection_count_by_score(score: int = Query(..., description="Score of the inspection")):
    cursor.execute("SELECT COUNT(score) FROM inspections WHERE score = ?", (score,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of inspections based on year and type
@app.get("/v1/food_inspection/inspection_count_by_year_and_type", operation_id="get_inspection_count_by_year_and_type", summary="Retrieves the total number of inspections conducted in a given year and of a specific type. The year should be provided in 'YYYY' format, and the inspection type should be specified to filter the results accordingly.")
async def get_inspection_count_by_year_and_type(year: str = Query(..., description="Year of the inspection in 'YYYY' format"), inspection_type: str = Query(..., description="Type of the inspection")):
    cursor.execute("SELECT COUNT(`date`) FROM inspections WHERE STRFTIME('%Y', `date`) = ? AND type = ?", (year, inspection_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct business names based on inspection score
@app.get("/v1/food_inspection/distinct_business_names_by_score", operation_id="get_distinct_business_names_by_score", summary="Retrieves a list of unique business names that have received a specific inspection score. The score parameter is used to filter the results.")
async def get_distinct_business_names_by_score(score: int = Query(..., description="Score of the inspection")):
    cursor.execute("SELECT DISTINCT T2.name FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score = ?", (score,))
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": [row[0] for row in result]}

# Endpoint to get the count of distinct business IDs based on year and city
@app.get("/v1/food_inspection/distinct_business_ids_by_year_and_city", operation_id="get_distinct_business_ids_by_year_and_city", summary="Retrieve the count of unique businesses that have undergone inspections in a specified year and up to four cities. This operation allows you to analyze the distribution of inspected businesses across different locations and time periods.")
async def get_distinct_business_ids_by_year_and_city(year: str = Query(..., description="Year of the inspection in 'YYYY' format"), city1: str = Query(..., description="City name"), city2: str = Query(..., description="City name"), city3: str = Query(..., description="City name"), city4: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.business_id) FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE STRFTIME('%Y', T1.`date`) = ? AND T2.city IN (?, ?, ?, ?)", (year, city1, city2, city3, city4))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get inspection types based on date and business name
@app.get("/v1/food_inspection/inspection_types_by_date_and_business_name", operation_id="get_inspection_types_by_date_and_business_name", summary="Retrieves the types of inspections conducted on a specific date for a given business. The operation requires the date of the inspection in 'YYYY-MM-DD' format and the name of the business as input parameters. The result is a list of inspection types associated with the provided date and business name.")
async def get_inspection_types_by_date_and_business_name(date: str = Query(..., description="Date of the inspection in 'YYYY-MM-DD' format"), business_name: str = Query(..., description="Name of the business")):
    cursor.execute("SELECT T1.type FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.`date` = ? AND T2.name = ?", (date, business_name))
    result = cursor.fetchall()
    if not result:
        return {"inspection_types": []}
    return {"inspection_types": [row[0] for row in result]}

# Endpoint to get the count of violations based on date, business name, and risk category
@app.get("/v1/food_inspection/violation_count_by_date_business_name_risk_category", operation_id="get_violation_count_by_date_business_name_risk_category", summary="Retrieves the total number of violations for a specific business on a given date, categorized by risk level. The business is identified by its name, and the date is provided in 'YYYY-MM-DD' format.")
async def get_violation_count_by_date_business_name_risk_category(date: str = Query(..., description="Date of the violation in 'YYYY-MM-DD' format"), business_name: str = Query(..., description="Name of the business"), risk_category: str = Query(..., description="Risk category of the violation")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.`date` = ? AND T2.name = ? AND T1.risk_category = ?", (date, business_name, risk_category))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct business names based on year and risk category
@app.get("/v1/food_inspection/distinct_business_names_by_year_and_risk_category", operation_id="get_distinct_business_names_by_year_and_risk_category", summary="Retrieves a list of unique business names that have had violations in a specified year and risk category. The year is provided in 'YYYY' format, and the risk category is a defined category for the type of violation.")
async def get_distinct_business_names_by_year_and_risk_category(year: str = Query(..., description="Year of the violation in 'YYYY' format"), risk_category: str = Query(..., description="Risk category of the violation")):
    cursor.execute("SELECT DISTINCT T2.name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE STRFTIME('%Y', T1.`date`) = ? AND T1.risk_category = ?", (year, risk_category))
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": [row[0] for row in result]}

# Endpoint to get violation descriptions based on date, business name, and risk category
@app.get("/v1/food_inspection/violation_descriptions_by_date_business_name_risk_category", operation_id="get_violation_descriptions_by_date_business_name_risk_category", summary="Retrieve a list of violation descriptions for a specific date, business, and risk category. The endpoint filters the violations based on the provided date, business name, and risk category, and returns the corresponding descriptions.")
async def get_violation_descriptions_by_date_business_name_risk_category(date: str = Query(..., description="Date of the violation in 'YYYY-MM-DD' format"), business_name: str = Query(..., description="Name of the business"), risk_category: str = Query(..., description="Risk category of the violation")):
    cursor.execute("SELECT T1.description FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.`date` = ? AND T2.name = ? AND T1.risk_category = ?", (date, business_name, risk_category))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct violation descriptions based on risk category and business name
@app.get("/v1/food_inspection/distinct_violation_descriptions_by_risk_category_business_name", operation_id="get_distinct_violation_descriptions_by_risk_category_business_name", summary="Retrieves unique descriptions of violations associated with a given risk category and business name. This operation filters the violations based on the specified risk category and business name, ensuring that only distinct violation descriptions are returned.")
async def get_distinct_violation_descriptions_by_risk_category_business_name(risk_category: str = Query(..., description="Risk category of the violation"), business_name: str = Query(..., description="Name of the business")):
    cursor.execute("SELECT DISTINCT T1.description FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.risk_category = ? AND T2.name = ?", (risk_category, business_name))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the count of business inspections based on inspection type and business name
@app.get("/v1/food_inspection/count_inspections_by_type_and_name", operation_id="get_count_inspections_by_type_and_name", summary="Retrieves the total number of inspections for a specific business, filtered by the type of inspection. The operation requires the inspection type and the name of the business as input parameters.")
async def get_count_inspections_by_type_and_name(inspection_type: str = Query(..., description="Type of inspection"), business_name: str = Query(..., description="Name of the business")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.type = ? AND T2.name = ?", (inspection_type, business_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of business inspections based on business name, inspection type, and minimum score
@app.get("/v1/food_inspection/count_inspections_by_name_type_and_score", operation_id="get_count_inspections_by_name_type_and_score", summary="Retrieves the total number of inspections for a specific business, filtered by inspection type and minimum score. The endpoint requires the business name, inspection type, and minimum score as input parameters to accurately calculate the count.")
async def get_count_inspections_by_name_type_and_score(business_name: str = Query(..., description="Name of the business"), inspection_type: str = Query(..., description="Type of inspection"), min_score: int = Query(..., description="Minimum score")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.name = ? AND T1.type = ? AND T1.score > ?", (business_name, inspection_type, min_score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to compare the number of violations between two businesses in a specific risk category
@app.get("/v1/food_inspection/compare_violations_by_business_and_risk", operation_id="compare_violations_by_business_and_risk", summary="This operation compares the total number of violations between two specified businesses within a given risk category. The result indicates which business has more violations.")
async def compare_violations_by_business_and_risk(business_name1: str = Query(..., description="Name of the first business"), business_name2: str = Query(..., description="Name of the second business"), risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT CASE WHEN SUM(CASE WHEN T2.name = ? THEN 1 ELSE 0 END) > SUM(CASE WHEN T2.name = ? THEN 1 ELSE 0 END) THEN ? ELSE ? END AS result FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.risk_category = ?", (business_name1, business_name2, business_name1, business_name2, risk_category))
    result = cursor.fetchone()
    if not result:
        return {"result": []}
    return {"result": result[0]}

# Endpoint to get the count of businesses with violations in specific cities and risk category
@app.get("/v1/food_inspection/count_violations_by_city_and_risk", operation_id="get_count_violations_by_city_and_risk", summary="Retrieves the total number of businesses with violations in up to four specified cities, filtered by a given risk category. This operation provides a quantitative overview of businesses with violations in the selected cities and risk category.")
async def get_count_violations_by_city_and_risk(city1: str = Query(..., description="First city name"), city2: str = Query(..., description="Second city name"), city3: str = Query(..., description="Third city name"), city4: str = Query(..., description="Fourth city name"), risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.city IN (?, ?, ?, ?) AND T1.risk_category = ?", (city1, city2, city3, city4, risk_category))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the business name with the most violations in a specific risk category
@app.get("/v1/food_inspection/top_business_by_violations_and_risk", operation_id="get_top_business_by_violations_and_risk", summary="Retrieves the name of the business with the highest number of violations in a specified risk category. The risk category is provided as an input parameter.")
async def get_top_business_by_violations_and_risk(risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT T2.name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.risk_category = ? GROUP BY T2.name ORDER BY COUNT(T2.name) DESC LIMIT 1", (risk_category,))
    result = cursor.fetchone()
    if not result:
        return {"business_name": []}
    return {"business_name": result[0]}

# Endpoint to get the average inspection score for a specific business
@app.get("/v1/food_inspection/average_inspection_score_by_business", operation_id="get_average_inspection_score_by_business", summary="Retrieves the average inspection score for a specified business. The operation calculates the average score from all inspections associated with the business, providing a comprehensive view of its inspection performance.")
async def get_average_inspection_score_by_business(business_name: str = Query(..., description="Name of the business")):
    cursor.execute("SELECT AVG(T1.score) FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.name = ?", (business_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the business ID with the most inspections
@app.get("/v1/food_inspection/top_business_by_inspections", operation_id="get_top_business_by_inspections", summary="Retrieves the unique identifier of the business that has undergone the highest number of inspections. The operation ranks businesses based on the count of their inspections and returns the top-ranked business.")
async def get_top_business_by_inspections():
    cursor.execute("SELECT business_id FROM inspections GROUP BY business_id ORDER BY COUNT(business_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"business_id": []}
    return {"business_id": result[0]}

# Endpoint to get the business ID with the most violations
@app.get("/v1/food_inspection/top_business_by_violations", operation_id="get_top_business_by_violations", summary="Retrieves the unique identifier of the business with the highest number of recorded violations.")
async def get_top_business_by_violations():
    cursor.execute("SELECT business_id FROM violations GROUP BY business_id ORDER BY COUNT(business_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"business_id": []}
    return {"business_id": result[0]}

# Endpoint to get the business name based on inspection score, date, and type
@app.get("/v1/food_inspection/business_name_by_score_date_and_type", operation_id="get_business_name_by_score_date_and_type", summary="Get the business name based on inspection score, date, and type")
async def get_business_name_by_score_date_and_type(score: int = Query(..., description="Inspection score"), date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), inspection_type: str = Query(..., description="Type of inspection")):
    cursor.execute("SELECT T2.name FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score = ? AND T1.date = ? AND T1.type = ?", (score, date, inspection_type))
    result = cursor.fetchone()
    if not result:
        return {"business_name": []}
    return {"business_name": result[0]}

# Endpoint to get the count of distinct violation types for a specific business on a specific date
@app.get("/v1/food_inspection/count_distinct_violation_types_by_business_and_date", operation_id="get_count_distinct_violation_types_by_business_and_date", summary="Retrieves the number of unique violation types associated with a specific business on a given date. The business is identified by its name, and the date is provided in 'YYYY-MM-DD' format.")
async def get_count_distinct_violation_types_by_business_and_date(business_name: str = Query(..., description="Name of the business"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.violation_type_id) FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.name = ? AND T1.date = ?", (business_name, date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get violation descriptions for a specific business on a specific date with a specific risk category
@app.get("/v1/food_inspection/violation_descriptions", operation_id="get_violation_descriptions", summary="Retrieves the violation descriptions for a specific business, on a given date, and within a particular risk category. The operation filters the violations based on the provided business name, date, and risk category, and returns the corresponding descriptions.")
async def get_violation_descriptions(business_name: str = Query(..., description="Name of the business"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT T1.description FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.name = ? AND T1.`date` = ? AND T1.risk_category = ?", (business_name, date, risk_category))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get business names with the minimum inspection score on a specific date and inspection type
@app.get("/v1/food_inspection/min_score_businesses", operation_id="get_min_score_businesses", summary="Retrieves the names of businesses that received the lowest inspection score on a specific date and for a particular inspection type. The date and inspection type are provided as input parameters.")
async def get_min_score_businesses(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), inspection_type: str = Query(..., description="Type of inspection")):
    cursor.execute("SELECT T2.name FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE score = ( SELECT MIN(score) FROM inspections WHERE `date` = ? AND type = ? ) AND T1.`date` = ? AND T1.type = ?", (date, inspection_type, date, inspection_type))
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": [row[0] for row in result]}

# Endpoint to get the business name with the most inspections of a specific type
@app.get("/v1/food_inspection/most_inspected_business", operation_id="get_most_inspected_business", summary="Retrieves the name of the business with the highest number of inspections of a specified type. The operation filters inspections by type and counts the number of inspections per business. The business with the most inspections is then identified and its name is returned.")
async def get_most_inspected_business(inspection_type: str = Query(..., description="Type of inspection")):
    cursor.execute("SELECT T2.name FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.type = ? GROUP BY T2.name ORDER BY COUNT(T1.business_id) DESC LIMIT 1", (inspection_type,))
    result = cursor.fetchone()
    if not result:
        return {"business_name": []}
    return {"business_name": result[0]}

# Endpoint to get the count of inspections for a specific business and inspection type
@app.get("/v1/food_inspection/inspection_count", operation_id="get_inspection_count", summary="Retrieves the total number of inspections conducted for a specific business and inspection type. The operation requires the business name and the type of inspection as input parameters to filter the results accordingly.")
async def get_inspection_count(business_name: str = Query(..., description="Name of the business"), inspection_type: str = Query(..., description="Type of inspection")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.name = ? AND T1.type = ?", (business_name, inspection_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the address with the most violations of a specific risk category
@app.get("/v1/food_inspection/most_violations_address", operation_id="get_most_violations_address", summary="Retrieves the address of the business with the highest number of violations in a specified risk category. The risk category is provided as an input parameter.")
async def get_most_violations_address(risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT T2.address FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.risk_category = ? GROUP BY T2.address ORDER BY COUNT(T1.business_id) DESC LIMIT 1", (risk_category,))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get business names with the earliest violation date for a specific risk category and description
@app.get("/v1/food_inspection/earliest_violation_businesses", operation_id="get_earliest_violation_businesses", summary="Retrieves the names of businesses that have the earliest recorded violation date for a specified risk category and violation description. The risk category and description are used to filter the results and ensure the returned businesses match the provided criteria.")
async def get_earliest_violation_businesses(risk_category: str = Query(..., description="Risk category"), description: str = Query(..., description="Description of the violation")):
    cursor.execute("SELECT T2.name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.`date` = ( SELECT MIN(`date`) FROM violations WHERE risk_category = ? AND description = ? ) AND T1.risk_category = ? AND T1.description = ?", (risk_category, description, risk_category, description))
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": [row[0] for row in result]}

# Endpoint to get the business with the most violations
@app.get("/v1/food_inspection/most_violations_business", operation_id="get_most_violations_business", summary="Retrieves the business with the highest number of violations, based on the aggregated data from the inspections and violations records.")
async def get_most_violations_business():
    cursor.execute("SELECT COUNT(T2.business_id) FROM violations AS T1 INNER JOIN inspections AS T2 ON T1.business_id = T2.business_id GROUP BY T1.business_id ORDER BY COUNT(T1.business_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of violations for a specific business certificate on a specific date
@app.get("/v1/food_inspection/violation_count_by_certificate", operation_id="get_violation_count_by_certificate", summary="Retrieves the total number of violations associated with a specific business certificate on a given date. The certificate and date are provided as input parameters.")
async def get_violation_count_by_certificate(business_certificate: str = Query(..., description="Business certificate"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.business_certificate = ? AND T1.`date` = ?", (business_certificate, date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average inspection score for a specific business name and inspection type
@app.get("/v1/food_inspection/average_inspection_score", operation_id="get_average_inspection_score", summary="Retrieves the average inspection score for a specific business, filtered by the type of inspection. The operation calculates the sum of scores for the specified business and divides it by the count of inspections of the given type, providing a precise average score.")
async def get_average_inspection_score(business_name: str = Query(..., description="Name of the business"), inspection_type: str = Query(..., description="Type of inspection")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.name = ? THEN T1.score ELSE 0 END) AS REAL) / COUNT(CASE WHEN T1.type = ? THEN T1.score ELSE 0 END) FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id", (business_name, inspection_type))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the percentage of violations of a specific risk category for a specific business name
@app.get("/v1/food_inspection/violation_percentage", operation_id="get_violation_percentage", summary="Retrieves the percentage of violations in a specified risk category for a given business. The calculation is based on the total number of violations in the specified risk category and the total number of violations for the business.")
async def get_violation_percentage(business_name: str = Query(..., description="Name of the business"), risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.risk_category = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.business_id) FROM businesses AS T1 INNER JOIN violations AS T2 ON T1.business_id = T2.business_id WHERE T1.name = ?", (risk_category, business_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of businesses in a specific city
@app.get("/v1/food_inspection/business_count_by_city", operation_id="get_business_count_by_city", summary="Retrieves the total number of businesses located in a specified city. The city is provided as an input parameter.")
async def get_business_count_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(business_id) FROM businesses WHERE city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct businesses with inspection scores below a certain threshold
@app.get("/v1/food_inspection/distinct_business_count_by_score", operation_id="get_distinct_business_count_by_score", summary="Retrieves the number of unique businesses that have received an inspection score lower than the provided threshold.")
async def get_distinct_business_count_by_score(score: int = Query(..., description="Inspection score threshold")):
    cursor.execute("SELECT COUNT(DISTINCT business_id) FROM inspections WHERE score < ?", (score,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses that applied in a specific year
@app.get("/v1/food_inspection/business_count_by_application_year", operation_id="get_business_count_by_application_year", summary="Retrieves the total number of businesses that applied in a given year. The year must be provided in the 'YYYY' format.")
async def get_business_count_by_application_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(business_id) FROM businesses WHERE STRFTIME('%Y', application_date) = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses inspected in a specific year and type
@app.get("/v1/food_inspection/business_count_by_inspection_year_and_type", operation_id="get_business_count_by_inspection_year_and_type", summary="Retrieves the total number of businesses that underwent inspections in a specified year and type. The year should be provided in 'YYYY' format, and the inspection type must be indicated.")
async def get_business_count_by_inspection_year_and_type(year: str = Query(..., description="Year in 'YYYY' format"), inspection_type: str = Query(..., description="Inspection type")):
    cursor.execute("SELECT COUNT(business_id) FROM inspections WHERE STRFTIME('%Y', `date`) = ? AND type = ?", (year, inspection_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of business owners with more than a certain number of businesses
@app.get("/v1/food_inspection/owner_count_by_business_count", operation_id="get_owner_count_by_business_count", summary="Retrieves the count of unique business owners who own more than a specified number of businesses. The input parameter determines the minimum number of businesses an owner must have to be included in the count.")
async def get_owner_count_by_business_count(business_count: int = Query(..., description="Number of businesses owned")):
    cursor.execute("SELECT COUNT(T1.owner_name) FROM ( SELECT owner_name FROM businesses GROUP BY owner_name HAVING COUNT(owner_name) > ? ) T1", (business_count,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct business names inspected in a specific year with a specific score
@app.get("/v1/food_inspection/distinct_business_names_by_year_and_score", operation_id="get_distinct_business_names_by_year_and_score", summary="Retrieves a list of unique business names that were inspected in a specified year and received a particular inspection score. The year should be provided in 'YYYY' format, and the score represents the inspection result.")
async def get_distinct_business_names_by_year_and_score(year: str = Query(..., description="Year in 'YYYY' format"), score: int = Query(..., description="Inspection score")):
    cursor.execute("SELECT DISTINCT T2.name FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE STRFTIME('%Y', T1.`date`) = ? AND T1.score = ?", (year, score))
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": [row[0] for row in result]}

# Endpoint to get the city with the highest number of high-risk violations in a specific year
@app.get("/v1/food_inspection/city_with_most_high_risk_violations_by_year", operation_id="get_city_with_most_high_risk_violations_by_year", summary="Retrieves the city with the most high-risk violations in a given year. The operation filters violations by the specified year and risk category, groups them by city, and returns the city with the highest count of violations.")
async def get_city_with_most_high_risk_violations_by_year(year: str = Query(..., description="Year in 'YYYY' format"), risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT T2.city FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE STRFTIME('%Y', T1.`date`) = ? AND T1.risk_category = ? GROUP BY T2.city ORDER BY COUNT(T2.city) DESC LIMIT 1", (year, risk_category))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the business names with the lowest inspection score
@app.get("/v1/food_inspection/business_names_with_lowest_inspection_score", operation_id="get_business_names_with_lowest_inspection_score", summary="Retrieves the names of businesses that have received the lowest inspection score. This operation identifies the minimum score from all inspections and returns the names of businesses that have received this score.")
async def get_business_names_with_lowest_inspection_score():
    cursor.execute("SELECT T2.name FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score = ( SELECT MIN(score) FROM inspections )")
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": [row[0] for row in result]}

# Endpoint to get the count of violations for a specific business name and risk category
@app.get("/v1/food_inspection/violation_count_by_business_name_and_risk_category", operation_id="get_violation_count_by_business_name_and_risk_category", summary="Retrieves the total number of violations associated with a specific business, filtered by a given risk category. The business is identified by its name, and the risk category is used to narrow down the violations considered in the count.")
async def get_violation_count_by_business_name_and_risk_category(business_name: str = Query(..., description="Business name"), risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.name = ? AND T1.risk_category = ?", (business_name, risk_category))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with more than a certain number of complaints and a specific tax code
@app.get("/v1/food_inspection/business_count_by_tax_code_and_complaint_count", operation_id="get_business_count_by_tax_code_and_complaint_count", summary="Retrieve the count of businesses that have a specified tax code and exceed a certain number of complaints of a particular type. This operation filters businesses based on the provided tax code and complaint type, then groups them by business ID. It only includes businesses with a complaint count surpassing the given threshold.")
async def get_business_count_by_tax_code_and_complaint_count(tax_code: str = Query(..., description="Tax code"), complaint_type: str = Query(..., description="Complaint type"), complaint_count: int = Query(..., description="Number of complaints")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T1.business_id FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.tax_code = ? AND T1.type = ? GROUP BY T1.business_id HAVING COUNT(T1.business_id) > ? ) T3", (tax_code, complaint_type, complaint_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business names with specific violation description in a given year
@app.get("/v1/food_inspection/business_names_by_violation_year", operation_id="get_business_names_by_violation_year", summary="Retrieves the names of businesses that have been issued a violation with a specific description in a given year. The year and the description of the violation are required as input parameters.")
async def get_business_names_by_violation_year(year: str = Query(..., description="Year in 'YYYY' format"), description: str = Query(..., description="Description of the violation")):
    cursor.execute("SELECT T2.name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE STRFTIME('%Y', T1.`date`) = ? AND T1.description = ?", (year, description))
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": [row[0] for row in result]}

# Endpoint to get the count of distinct businesses with violations in a given year, postal code, and inspection score
@app.get("/v1/food_inspection/count_businesses_by_year_postal_score", operation_id="get_count_businesses_by_year_postal_score", summary="Retrieve the number of unique businesses that have reported violations in a specific year, postal code, and with an inspection score higher than the provided value. This operation considers only businesses with violations and filters them based on the year of violation, postal code, and inspection score.")
async def get_count_businesses_by_year_postal_score(year: str = Query(..., description="Year in 'YYYY' format"), postal_code: str = Query(..., description="Postal code"), score: int = Query(..., description="Inspection score")):
    cursor.execute("SELECT COUNT(DISTINCT T2.business_id) FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id INNER JOIN inspections AS T3 ON T2.business_id = T3.business_id WHERE STRFTIME('%Y', T1.`date`) = ? AND T2.postal_code = ? AND T3.score > ?", (year, postal_code, score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct business names with a perfect inspection score over a specified number of consecutive years
@app.get("/v1/food_inspection/business_names_perfect_score_consecutive_years", operation_id="get_business_names_perfect_score_consecutive_years", summary="Retrieves unique business names that have maintained a perfect inspection score for a specified number of consecutive years. The operation filters inspections by a given perfect score and identifies businesses that have consistently achieved this score over the requested period.")
async def get_business_names_perfect_score_consecutive_years(score: int = Query(..., description="Perfect inspection score"), consecutive_years: int = Query(..., description="Number of consecutive years")):
    cursor.execute("SELECT DISTINCT T4.name FROM ( SELECT T3.name, T3.years, row_number() OVER (PARTITION BY T3.name ORDER BY T3.years) AS rowNumber FROM ( SELECT DISTINCT name, STRFTIME('%Y', `date`) AS years FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score = ? ) AS T3 ) AS T4 GROUP BY T4.name, date(T4.years || '-01-01', '-' || (T4.rowNumber - 1) || ' years') HAVING COUNT(T4.years) = ?", (score, consecutive_years))
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": [row[0] for row in result]}

# Endpoint to get the average inspection score for a specific owner in a given time range and location
@app.get("/v1/food_inspection/avg_inspection_score_owner_time_location", operation_id="get_avg_inspection_score_owner_time_location", summary="Retrieves the average inspection score for a specific owner's establishments within a given time range and location. The operation considers inspections conducted between the provided start and end years, and filters results based on the owner's name, address, and city.")
async def get_avg_inspection_score_owner_time_location(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), owner_name: str = Query(..., description="Owner name"), address: str = Query(..., description="Address"), city: str = Query(..., description="City")):
    cursor.execute("SELECT AVG(T1.score) FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE STRFTIME('%Y', T1.`date`) BETWEEN ? AND ? AND T2.owner_name = ? AND T2.address = ? AND T2.city = ?", (start_year, end_year, owner_name, address, city))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the average inspection score for the owner with the most businesses
@app.get("/v1/food_inspection/avg_inspection_score_top_owner", operation_id="get_avg_inspection_score_top_owner", summary="Retrieves the average inspection score for the owner who operates the most businesses. This operation calculates the average score from all inspections conducted for each business owned by the respective owner, and then identifies the owner with the highest number of businesses. The result is the average inspection score for this top owner.")
async def get_avg_inspection_score_top_owner():
    cursor.execute("SELECT AVG(T1.score) FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id GROUP BY T2.owner_name ORDER BY COUNT(T2.business_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the business name with the most violations in a given year and risk category
@app.get("/v1/food_inspection/top_business_by_violations_year_risk", operation_id="get_top_business_by_violations_year_risk", summary="Retrieves the name of the business with the highest number of violations in a specified year and risk category. The year should be provided in 'YYYY' format, and the risk category should be specified. The data is grouped by business name and ordered in descending order based on the count of business IDs, with the top result being returned.")
async def get_top_business_by_violations_year_risk(year: str = Query(..., description="Year in 'YYYY' format"), risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT T2.name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE STRFTIME('%Y', T1.`date`) = ? AND T1.risk_category = ? GROUP BY T2.name ORDER BY COUNT(T2.business_id) DESC LIMIT 1", (year, risk_category))
    result = cursor.fetchone()
    if not result:
        return {"business_name": []}
    return {"business_name": result[0]}

# Endpoint to get the owner name with the most high-risk violations among the top 5 owners with the most violations
@app.get("/v1/food_inspection/top_owner_high_risk_violations", operation_id="get_top_owner_high_risk_violations", summary="Retrieves the name of the owner with the highest number of high-risk violations among the top five owners with the most violations. The risk category is specified as an input parameter.")
async def get_top_owner_high_risk_violations(risk_category: str = Query(..., description="Risk category")):
    cursor.execute("SELECT T4.owner_name FROM violations AS T3 INNER JOIN businesses AS T4 ON T3.business_id = T4.business_id INNER JOIN ( SELECT T2.owner_name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id GROUP BY T2.owner_name ORDER BY COUNT(T1.business_id) DESC LIMIT 5 ) AS T5 ON T4.owner_name = T5.owner_name WHERE T3.risk_category = ? GROUP BY T4.owner_name ORDER BY COUNT(T3.risk_category) DESC LIMIT 1", (risk_category,))
    result = cursor.fetchone()
    if not result:
        return {"owner_name": []}
    return {"owner_name": result[0]}

# Endpoint to get the business name and average inspection score for the business with the most inspections
@app.get("/v1/food_inspection/top_business_avg_inspection_score", operation_id="get_top_business_avg_inspection_score", summary="Retrieves the name and average inspection score of the business that has undergone the most inspections. The data is sourced from the inspections and businesses tables, with the results grouped by business name and ordered by the count of inspections in descending order. Only the top result is returned.")
async def get_top_business_avg_inspection_score():
    cursor.execute("SELECT T2.name, AVG(T1.score) FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id GROUP BY T2.name ORDER BY COUNT(T2.business_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"business_name": [], "average_score": []}
    return {"business_name": result[0], "average_score": result[1]}

# Endpoint to get the count of distinct businesses with the highest inspection score in a given year
@app.get("/v1/food_inspection/count_businesses_highest_score_year", operation_id="get_count_businesses_highest_score_year", summary="Retrieves the number of unique businesses that achieved the highest inspection score during a specified year. The year must be provided in the 'YYYY' format.")
async def get_count_businesses_highest_score_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(DISTINCT business_id) FROM inspections WHERE STRFTIME('%Y', `date`) = ? AND score = ( SELECT MAX(score) FROM inspections WHERE STRFTIME('%Y', `date`) = ? )", (year, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs for a specific inspection type and date pattern
@app.get("/v1/food_inspection/business_ids_by_inspection_type_date", operation_id="get_business_ids_by_inspection_type_date", summary="Get business IDs for a specific inspection type and date pattern")
async def get_business_ids_by_inspection_type_date(inspection_type: str = Query(..., description="Inspection type"), date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM%' format")):
    cursor.execute("SELECT business_id FROM inspections WHERE type = ? AND `date` LIKE ?", (inspection_type, date_pattern))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct business IDs based on risk category and description
@app.get("/v1/food_inspection/count_distinct_business_ids", operation_id="get_count_distinct_business_ids", summary="Retrieves the total number of unique businesses that have a specific risk category and violation description in their records. This operation allows you to understand the distribution of businesses based on their risk level and the type of violations they have committed.")
async def get_count_distinct_business_ids(risk_category: str = Query(..., description="Risk category of the violation"), description: str = Query(..., description="Description of the violation")):
    cursor.execute("SELECT COUNT(DISTINCT business_id) FROM violations WHERE risk_category = ? AND description = ?", (risk_category, description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs, risk categories, and descriptions based on violation type ID
@app.get("/v1/food_inspection/get_violations_by_type_id", operation_id="get_violations_by_type_id", summary="Retrieves a list of businesses with their associated risk categories and violation descriptions based on the provided violation type ID.")
async def get_violations_by_type_id(violation_type_id: str = Query(..., description="Violation type ID")):
    cursor.execute("SELECT business_id, risk_category, description FROM violations WHERE violation_type_id = ?", (violation_type_id,))
    result = cursor.fetchall()
    if not result:
        return {"violations": []}
    return {"violations": result}

# Endpoint to get the latest inspection date for a given city based on the highest score
@app.get("/v1/food_inspection/latest_inspection_date_by_city", operation_id="get_latest_inspection_date_by_city", summary="Retrieves the most recent inspection date for a specified city, based on the inspection with the highest score. The city is identified by its name.")
async def get_latest_inspection_date_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.`date` FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.city = ? ORDER BY T1.score DESC LIMIT 1", (city,))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get distinct inspection types and violation descriptions based on business name and risk category
@app.get("/v1/food_inspection/get_inspection_types_and_violations", operation_id="get_inspection_types_and_violations", summary="Retrieves unique inspection types and corresponding violation descriptions for a specified business and risk category. This operation filters data based on the provided business name and risk category, ensuring that only relevant and distinct results are returned.")
async def get_inspection_types_and_violations(business_name: str = Query(..., description="Name of the business"), risk_category: str = Query(..., description="Risk category of the violation")):
    cursor.execute("SELECT DISTINCT T2.type, T1.description FROM violations AS T1 INNER JOIN inspections AS T2 ON T1.business_id = T2.business_id INNER JOIN businesses AS T3 ON T2.business_id = T3.business_id WHERE T3.name = ? AND T1.risk_category = ?", (business_name, risk_category))
    result = cursor.fetchall()
    if not result:
        return {"inspections": []}
    return {"inspections": result}

# Endpoint to get distinct violation type IDs and descriptions based on business name and risk category
@app.get("/v1/food_inspection/get_violation_types_and_descriptions", operation_id="get_violation_types_and_descriptions", summary="Retrieves unique violation type identifiers and their corresponding descriptions for a specific business and risk category. This operation filters the violations based on the provided business name and risk category, ensuring that only distinct violation types are returned.")
async def get_violation_types_and_descriptions(business_name: str = Query(..., description="Name of the business"), risk_category: str = Query(..., description="Risk category of the violation")):
    cursor.execute("SELECT DISTINCT T1.violation_type_id, T1.description FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.name = ? AND T1.risk_category = ?", (business_name, risk_category))
    result = cursor.fetchall()
    if not result:
        return {"violations": []}
    return {"violations": result}

# Endpoint to get inspection dates, scores, and types based on tax code
@app.get("/v1/food_inspection/get_inspections_by_tax_code", operation_id="get_inspections_by_tax_code", summary="Retrieves the dates, scores, and types of inspections for a business identified by its tax code. The operation returns a list of inspection records, each containing the inspection date, score, and type.")
async def get_inspections_by_tax_code(tax_code: str = Query(..., description="Tax code of the business")):
    cursor.execute("SELECT T1.`date`, T1.score, T1.type FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.tax_code = ?", (tax_code,))
    result = cursor.fetchall()
    if not result:
        return {"inspections": []}
    return {"inspections": result}

# Endpoint to get distinct business IDs, names, and addresses based on inspection date
@app.get("/v1/food_inspection/get_businesses_by_inspection_date", operation_id="get_businesses_by_inspection_date", summary="Retrieves unique businesses, identified by their IDs, names, and addresses, that were inspected on a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_businesses_by_inspection_date(date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.business_id, T2.name, T2.address FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.date = ?", (date,))
    result = cursor.fetchall()
    if not result:
        return {"businesses": []}
    return {"businesses": result}

# Endpoint to get violation dates, risk categories, descriptions, and business names based on owner name
@app.get("/v1/food_inspection/get_violations_by_owner_name", operation_id="get_violations_by_owner_name", summary="Retrieves the dates, risk categories, descriptions, and business names of violations associated with a specific owner name. The owner name is used to filter the results.")
async def get_violations_by_owner_name(owner_name: str = Query(..., description="Owner name of the business")):
    cursor.execute("SELECT T1.`date`, T1.risk_category, T1.description, T2.name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.owner_name = ?", (owner_name,))
    result = cursor.fetchall()
    if not result:
        return {"violations": []}
    return {"violations": result}

# Endpoint to get business names, risk categories, and descriptions based on violation type ID
@app.get("/v1/food_inspection/get_businesses_by_violation_type_id", operation_id="get_businesses_by_violation_type_id", summary="Retrieves the names, risk categories, and descriptions of businesses that have a specific violation type. The violation type is identified by its unique ID.")
async def get_businesses_by_violation_type_id(violation_type_id: str = Query(..., description="Violation type ID")):
    cursor.execute("SELECT T2.name, T1.risk_category, T1.description FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.violation_type_id = ?", (violation_type_id,))
    result = cursor.fetchall()
    if not result:
        return {"businesses": []}
    return {"businesses": result}

# Endpoint to get distinct business names, cities, and tax codes based on risk category and date
@app.get("/v1/food_inspection/get_businesses_by_risk_category_and_date", operation_id="get_businesses_by_risk_category_and_date", summary="Retrieves unique business details, including name, city, and tax code, based on a specified risk category and date. The operation returns a maximum of 5 records that match the provided risk category and date.")
async def get_businesses_by_risk_category_and_date(risk_category: str = Query(..., description="Risk category of the violation"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.name, T2.city, T2.tax_code FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.risk_category = ? AND T1.`date` = ? LIMIT 5", (risk_category, date))
    result = cursor.fetchall()
    if not result:
        return {"businesses": []}
    return {"businesses": result}

# Endpoint to get the type of the highest-scoring inspection for a given business name
@app.get("/v1/food_inspection/highest_scoring_inspection_type", operation_id="get_highest_scoring_inspection_type", summary="Retrieves the type of the inspection with the highest score for a specified business. The business is identified by its name.")
async def get_highest_scoring_inspection_type(business_name: str = Query(..., description="Name of the business")):
    cursor.execute("SELECT T1.type FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.name = ? ORDER BY T1.score DESC LIMIT 1", (business_name,))
    result = cursor.fetchone()
    if not result:
        return {"type": []}
    return {"type": result[0]}

# Endpoint to get the owner names of businesses with specific risk category and violation description
@app.get("/v1/food_inspection/owner_names_by_risk_category_and_description", operation_id="get_owner_names_by_risk_category_and_description", summary="Retrieve the names of business owners who have been cited for a specific type of violation, as defined by the risk category and description. This operation filters businesses based on the provided risk category and violation description, then returns the corresponding owner names.")
async def get_owner_names_by_risk_category_and_description(risk_category: str = Query(..., description="Risk category of the violation"), description: str = Query(..., description="Description of the violation")):
    cursor.execute("SELECT T2.owner_name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.risk_category = ? AND T1.description = ?", (risk_category, description))
    result = cursor.fetchall()
    if not result:
        return {"owner_names": []}
    return {"owner_names": [row[0] for row in result]}

# Endpoint to get business names and addresses based on inspection date and type
@app.get("/v1/food_inspection/business_info_by_inspection_date_and_type", operation_id="get_business_info_by_inspection_date_and_type", summary="Retrieves the names and addresses of businesses that underwent a specific type of inspection on a given date. The operation filters businesses based on the provided inspection date and type, and returns the matching records.")
async def get_business_info_by_inspection_date_and_type(date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), inspection_type: str = Query(..., description="Type of inspection")):
    cursor.execute("SELECT T2.name, T2.address FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.date = ? AND T1.type = ?", (date, inspection_type))
    result = cursor.fetchall()
    if not result:
        return {"business_info": []}
    return {"business_info": [{"name": row[0], "address": row[1]} for row in result]}

# Endpoint to get business names and IDs with inspection scores below a certain threshold
@app.get("/v1/food_inspection/business_info_by_score_threshold", operation_id="get_business_info_by_score_threshold", summary="Retrieves the names and unique identifiers of businesses that have received inspection scores below the specified threshold. This operation is useful for identifying businesses with potentially subpar food safety practices.")
async def get_business_info_by_score_threshold(score_threshold: int = Query(..., description="Threshold score for inspections")):
    cursor.execute("SELECT T2.name, T2.business_id FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score < ?", (score_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"business_info": []}
    return {"business_info": [{"name": row[0], "business_id": row[1]} for row in result]}

# Endpoint to get the count of businesses based on address and city
@app.get("/v1/food_inspection/business_count_by_address_and_city", operation_id="get_business_count_by_address_and_city", summary="Retrieves the total number of businesses located at a specific address within a given city. The address and city are provided as input parameters.")
async def get_business_count_by_address_and_city(address: str = Query(..., description="Address of the business"), city: str = Query(..., description="City of the business")):
    cursor.execute("SELECT COUNT(business_id) FROM businesses WHERE address = ? AND city = ?", (address, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct owner names based on owner zip code
@app.get("/v1/food_inspection/distinct_owner_names_by_zip", operation_id="get_distinct_owner_names_by_zip", summary="Retrieves a list of unique owner names associated with a specific zip code. This operation allows you to identify the distinct owners operating in a particular area, providing valuable insights into business ownership distribution.")
async def get_distinct_owner_names_by_zip(owner_zip: str = Query(..., description="Zip code of the owner")):
    cursor.execute("SELECT DISTINCT owner_name FROM businesses WHERE owner_zip = ?", (owner_zip,))
    result = cursor.fetchall()
    if not result:
        return {"owner_names": []}
    return {"owner_names": [row[0] for row in result]}

# Endpoint to get the count of businesses based on tax code
@app.get("/v1/food_inspection/business_count_by_tax_code", operation_id="get_business_count_by_tax_code", summary="Retrieves the total number of businesses associated with a specific tax code. The tax code is provided as an input parameter, allowing the user to filter the count based on the desired tax code.")
async def get_business_count_by_tax_code(tax_code: str = Query(..., description="Tax code of the business")):
    cursor.execute("SELECT COUNT(tax_code) FROM businesses WHERE tax_code = ?", (tax_code,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of violations based on year and risk category
@app.get("/v1/food_inspection/violation_count_by_year_and_risk_category", operation_id="get_violation_count_by_year_and_risk_category", summary="Retrieves the total number of violations that occurred in a specific year and risk category. The year is provided in 'YYYY' format, and the risk category is a classification of the violation's severity or impact.")
async def get_violation_count_by_year_and_risk_category(year: str = Query(..., description="Year of the violation in 'YYYY' format"), risk_category: str = Query(..., description="Risk category of the violation")):
    cursor.execute("SELECT COUNT(risk_category) FROM violations WHERE STRFTIME('%Y', date) = ? AND risk_category = ?", (year, risk_category))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct business IDs and risk categories based on owner name
@app.get("/v1/food_inspection/business_ids_and_risk_categories_by_owner_name", operation_id="get_business_ids_and_risk_categories_by_owner_name", summary="Retrieves unique business identifiers and their associated risk categories for a specified owner name. This operation filters businesses based on the provided owner name and returns distinct business IDs along with their corresponding risk categories.")
async def get_business_ids_and_risk_categories_by_owner_name(owner_name: str = Query(..., description="Name of the owner")):
    cursor.execute("SELECT DISTINCT T2.business_id, T1.risk_category FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.owner_name = ?", (owner_name,))
    result = cursor.fetchall()
    if not result:
        return {"business_info": []}
    return {"business_info": [{"business_id": row[0], "risk_category": row[1]} for row in result]}

# Endpoint to get distinct owner names based on inspection score
@app.get("/v1/food_inspection/distinct_owner_names_by_inspection_score", operation_id="get_distinct_owner_names_by_inspection_score", summary="Retrieves a list of unique owner names associated with businesses that have received a specific inspection score. The score is provided as an input parameter.")
async def get_distinct_owner_names_by_inspection_score(score: int = Query(..., description="Inspection score")):
    cursor.execute("SELECT DISTINCT T2.owner_name FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score = ?", (score,))
    result = cursor.fetchall()
    if not result:
        return {"owner_names": []}
    return {"owner_names": [row[0] for row in result]}

# Endpoint to get the count of distinct businesses based on postal code and risk category
@app.get("/v1/food_inspection/count_businesses_by_postal_code_risk_category", operation_id="get_count_businesses_by_postal_code_risk_category", summary="Retrieves the number of unique businesses in a given postal code that have a specific risk category associated with their violations.")
async def get_count_businesses_by_postal_code_risk_category(postal_code: int = Query(..., description="Postal code of the business"), risk_category: str = Query(..., description="Risk category of the violation")):
    cursor.execute("SELECT COUNT(DISTINCT T2.business_id) FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.postal_code = ? AND T1.risk_category = ?", (postal_code, risk_category))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct violation types and risk categories based on inspection score range
@app.get("/v1/food_inspection/violation_types_risk_categories_by_score_range", operation_id="get_violation_types_risk_categories_by_score_range", summary="Retrieves unique combinations of violation types and risk categories associated with businesses that have inspection scores within the specified range. The range is defined by the minimum and maximum scores provided as input parameters.")
async def get_violation_types_risk_categories_by_score_range(min_score: int = Query(..., description="Minimum inspection score"), max_score: int = Query(..., description="Maximum inspection score")):
    cursor.execute("SELECT DISTINCT T1.violation_type_id, T1.risk_category FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id INNER JOIN inspections AS T3 ON T2.business_id = T3.business_id WHERE T3.score BETWEEN ? AND ?", (min_score, max_score))
    result = cursor.fetchall()
    if not result:
        return {"violations": []}
    return {"violations": result}

# Endpoint to get distinct tax codes and inspection types based on business name
@app.get("/v1/food_inspection/tax_codes_inspection_types_by_business_name", operation_id="get_tax_codes_inspection_types_by_business_name", summary="Retrieves unique tax codes and inspection types associated with a specific business. The operation filters results based on the provided business name.")
async def get_tax_codes_inspection_types_by_business_name(business_name: str = Query(..., description="Name of the business")):
    cursor.execute("SELECT DISTINCT T3.tax_code, T2.type FROM violations AS T1 INNER JOIN inspections AS T2 ON T1.business_id = T2.business_id INNER JOIN businesses AS T3 ON T2.business_id = T3.business_id WHERE T3.name = ?", (business_name,))
    result = cursor.fetchall()
    if not result:
        return {"tax_codes_inspection_types": []}
    return {"tax_codes_inspection_types": result}

# Endpoint to get distinct business names based on violation date, violation type, and inspection type
@app.get("/v1/food_inspection/business_names_by_violation_date_type_inspection_type", operation_id="get_business_names_by_violation_date_type_inspection_type", summary="Retrieves a list of unique business names that have been inspected and found to have a specific violation on a given date. The violation is identified by its type, and the inspection type is also specified.")
async def get_business_names_by_violation_date_type_inspection_type(violation_date: str = Query(..., description="Violation date in 'YYYY-MM-DD' format"), violation_type_id: int = Query(..., description="Violation type ID"), inspection_type: str = Query(..., description="Inspection type")):
    cursor.execute("SELECT DISTINCT T3.name FROM violations AS T1 INNER JOIN inspections AS T2 ON T1.business_id = T2.business_id INNER JOIN businesses AS T3 ON T2.business_id = T3.business_id WHERE T1.`date` = ? AND T1.violation_type_id = ? AND T2.type = ?", (violation_date, violation_type_id, inspection_type))
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": result}

# Endpoint to get distinct owner names based on risk category, violation type, and description
@app.get("/v1/food_inspection/owner_names_by_risk_category_violation_type_description", operation_id="get_owner_names_by_risk_category_violation_type_description", summary="Retrieves a list of unique owner names associated with businesses that have violations matching the specified risk category, violation type, and description. This operation filters the violations based on the provided parameters and returns the distinct owner names of the businesses linked to these violations.")
async def get_owner_names_by_risk_category_violation_type_description(risk_category: str = Query(..., description="Risk category of the violation"), violation_type_id: int = Query(..., description="Violation type ID"), description: str = Query(..., description="Description of the violation")):
    cursor.execute("SELECT DISTINCT T2.owner_name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.risk_category = ? AND T1.violation_type_id = ? AND T1.description = ?", (risk_category, violation_type_id, description))
    result = cursor.fetchall()
    if not result:
        return {"owner_names": []}
    return {"owner_names": result}

# Endpoint to get distinct business names based on owner city and inspection score
@app.get("/v1/food_inspection/business_names_by_owner_city_inspection_score", operation_id="get_business_names_by_owner_city_inspection_score", summary="Retrieves a list of unique business names that are located in a specified city and have received a particular inspection score. The city and score are provided as input parameters.")
async def get_business_names_by_owner_city_inspection_score(owner_city: str = Query(..., description="Owner city of the business"), inspection_score: int = Query(..., description="Inspection score")):
    cursor.execute("SELECT DISTINCT T2.name FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.owner_city = ? AND T1.score = ?", (owner_city, inspection_score))
    result = cursor.fetchall()
    if not result:
        return {"business_names": []}
    return {"business_names": result}

# Endpoint to get distinct violation types based on business ID range, address, and city
@app.get("/v1/food_inspection/violation_types_by_business_id_range_address_city", operation_id="get_violation_types_by_business_id_range_address_city", summary="Retrieves a list of unique violation types associated with businesses within a specified range of business IDs, located at a given address and city. This operation filters the businesses based on the provided address and city, and returns the distinct violation types linked to these businesses.")
async def get_violation_types_by_business_id_range_address_city(min_business_id: int = Query(..., description="Minimum business ID"), max_business_id: int = Query(..., description="Maximum business ID"), address: str = Query(..., description="Address of the business"), city: str = Query(..., description="City of the business")):
    cursor.execute("SELECT DISTINCT T1.violation_type_id FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.business_id BETWEEN ? AND ? AND T2.address = ? AND T2.city = ?", (min_business_id, max_business_id, address, city))
    result = cursor.fetchall()
    if not result:
        return {"violation_types": []}
    return {"violation_types": result}

# Endpoint to get distinct owner names based on violation type and date
@app.get("/v1/food_inspection/owner_names_by_violation_type_date", operation_id="get_owner_names_by_violation_type_date", summary="Retrieve a list of unique owner names associated with a specific violation type and date. The operation filters businesses by the provided violation type ID and date, then returns the distinct owner names linked to these businesses.")
async def get_owner_names_by_violation_type_date(violation_type_id: int = Query(..., description="Violation type ID"), violation_date: str = Query(..., description="Violation date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.owner_name FROM violations AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.violation_type_id = ? AND T1.`date` = ?", (violation_type_id, violation_date))
    result = cursor.fetchall()
    if not result:
        return {"owner_names": []}
    return {"owner_names": result}

# Endpoint to get the inspection type with the highest score based on owner address and city
@app.get("/v1/food_inspection/highest_score_inspection_type_by_owner_address_city", operation_id="get_highest_score_inspection_type_by_owner_address_city", summary="Retrieves the type of inspection with the highest score for a business located at a specific address in a given city. The business address and city are provided as input parameters.")
async def get_highest_score_inspection_type_by_owner_address_city(owner_address: str = Query(..., description="Owner address of the business"), owner_city: str = Query(..., description="Owner city of the business")):
    cursor.execute("SELECT T1.type FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T2.owner_address = ? AND T2.owner_city = ? ORDER BY T1.score DESC LIMIT 1", (owner_address, owner_city))
    result = cursor.fetchone()
    if not result:
        return {"inspection_type": []}
    return {"inspection_type": result[0]}

# Endpoint to get the count of businesses based on violation year and inspection type
@app.get("/v1/food_inspection/count_businesses_by_violation_year_inspection_type", operation_id="get_count_businesses_by_violation_year_inspection_type", summary="Retrieves the total number of businesses that have been inspected and found to have violations in a specific year, categorized by the type of inspection. The year is provided in 'YYYY' format and the inspection type is specified.")
async def get_count_businesses_by_violation_year_inspection_type(violation_year: str = Query(..., description="Violation year in 'YYYY' format"), inspection_type: str = Query(..., description="Inspection type")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM violations AS T1 INNER JOIN inspections AS T2 ON T1.business_id = T2.business_id WHERE STRFTIME('%Y', T1.`date`) = ? AND T2.type = ?", (violation_year, inspection_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct business names and their risk categories based on inspection scores
@app.get("/v1/food_inspection/business_risk_categories", operation_id="get_business_risk_categories", summary="Retrieve a list of unique businesses along with their associated risk categories, filtered by inspections with scores exceeding a specified percentage of the average score.")
async def get_business_risk_categories(score_percentage: float = Query(..., description="Percentage of the average inspection score")):
    cursor.execute("SELECT DISTINCT T1.name, T3.risk_category FROM businesses AS T1 INNER JOIN inspections AS T2 ON T1.business_id = T2.business_id INNER JOIN violations AS T3 ON T1.business_id = T3.business_id WHERE T2.score > ? * ( SELECT AVG(score) FROM inspections )", (score_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"businesses": []}
    return {"businesses": result}

# Endpoint to get the percentage of low-risk violations based on inspection scores and postal code
@app.get("/v1/food_inspection/low_risk_violation_percentage", operation_id="get_low_risk_violation_percentage", summary="Retrieves the percentage of low-risk violations for businesses in a specific postal code that have an inspection score below a certain value. The risk category of the violations is also considered.")
async def get_low_risk_violation_percentage(risk_category: str = Query(..., description="Risk category of the violation"), max_score: int = Query(..., description="Maximum inspection score"), postal_code: int = Query(..., description="Postal code of the business")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.risk_category = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.risk_category) FROM violations AS T1 INNER JOIN inspections AS T2 ON T1.business_id = T2.business_id INNER JOIN businesses AS T3 ON T2.business_id = T3.business_id WHERE T2.score < ? AND T3.postal_code = ?", (risk_category, max_score, postal_code))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/food_inspection/business_count_by_owner_state?owner_state=CA",
    "/v1/food_inspection/inspection_count_by_score?score=100",
    "/v1/food_inspection/inspection_count_by_year_and_type?year=2016&inspection_type=Routine%20-%20Unscheduled",
    "/v1/food_inspection/distinct_business_names_by_score?score=100",
    "/v1/food_inspection/distinct_business_ids_by_year_and_city?year=2016&city1=San%20Francisco&city2=SAN%20FRANCISCO&city3=SF&city4=S.F.",
    "/v1/food_inspection/inspection_types_by_date_and_business_name?date=2014-01-14&business_name=Tiramisu%20Kitchen",
    "/v1/food_inspection/violation_count_by_date_business_name_risk_category?date=2014-01-14&business_name=Tiramisu%20Kitchen&risk_category=Low%20Risk",
    "/v1/food_inspection/distinct_business_names_by_year_and_risk_category?year=2014&risk_category=Low%20Risk",
    "/v1/food_inspection/violation_descriptions_by_date_business_name_risk_category?date=2014-01-14&business_name=Tiramisu%20Kitchen&risk_category=Low%20Risk",
    "/v1/food_inspection/distinct_violation_descriptions_by_risk_category_business_name?risk_category=High%20Risk&business_name=Tiramisu%20Kitchen",
    "/v1/food_inspection/count_inspections_by_type_and_name?inspection_type=Routine%20-%20Unscheduled&business_name=Tiramisu%20Kitchen",
    "/v1/food_inspection/count_inspections_by_name_type_and_score?business_name=Tiramisu%20Kitchen&inspection_type=Routine%20-%20Unscheduled&min_score=70",
    "/v1/food_inspection/compare_violations_by_business_and_risk?business_name1=OMNI%20S.F.%20Hotel%20-%202nd%20Floor%20Pantry&business_name2=Tiramisu%20Kitchen&risk_category=Low%20Risk",
    "/v1/food_inspection/count_violations_by_city_and_risk?city1=San%20Francisco&city2=SF&city3=S.F.&city4=SAN%20FRANCISCO&risk_category=High%20Risk",
    "/v1/food_inspection/top_business_by_violations_and_risk?risk_category=High%20Risk",
    "/v1/food_inspection/average_inspection_score_by_business?business_name=Tiramisu%20Kitchen",
    "/v1/food_inspection/top_business_by_inspections",
    "/v1/food_inspection/top_business_by_violations",
    "/v1/food_inspection/business_name_by_score_date_and_type?score=100&date=2016-09-28&inspection_type=Routine%20-%20Unscheduled",
    "/v1/food_inspection/count_distinct_violation_types_by_business_and_date?business_name=Stacks%20Restaurant&date=2016-10-04",
    "/v1/food_inspection/violation_descriptions?business_name=Chez%20Fayala%2C%20Inc.&date=2016-07-01&risk_category=Moderate%20Risk",
    "/v1/food_inspection/min_score_businesses?date=2016-09-26&inspection_type=Routine%20-%20Unscheduled",
    "/v1/food_inspection/most_inspected_business?inspection_type=Complaint",
    "/v1/food_inspection/inspection_count?business_name=Soma%20Restaurant%20And%20Bar&inspection_type=Routine%20-%20Unscheduled",
    "/v1/food_inspection/most_violations_address?risk_category=Low%20Risk",
    "/v1/food_inspection/earliest_violation_businesses?risk_category=Low%20Risk&description=Permit%20license%20or%20inspection%20report%20not%20posted",
    "/v1/food_inspection/most_violations_business",
    "/v1/food_inspection/violation_count_by_certificate?business_certificate=304977&date=2013-10-07",
    "/v1/food_inspection/average_inspection_score?business_name=Chairman%20Bao&inspection_type=Routine%20-%20Unscheduled",
    "/v1/food_inspection/violation_percentage?business_name=Melody%20Lounge&risk_category=Moderate%20Risk",
    "/v1/food_inspection/business_count_by_city?city=HAYWARD",
    "/v1/food_inspection/distinct_business_count_by_score?score=50",
    "/v1/food_inspection/business_count_by_application_year?year=2012",
    "/v1/food_inspection/business_count_by_inspection_year_and_type?year=2014&inspection_type=Foodborne%20Illness%20Investigation",
    "/v1/food_inspection/owner_count_by_business_count?business_count=5",
    "/v1/food_inspection/distinct_business_names_by_year_and_score?year=2013&score=100",
    "/v1/food_inspection/city_with_most_high_risk_violations_by_year?year=2016&risk_category=High%20Risk",
    "/v1/food_inspection/business_names_with_lowest_inspection_score",
    "/v1/food_inspection/violation_count_by_business_name_and_risk_category?business_name=Tiramisu%20Kitchen&risk_category=High%20Risk",
    "/v1/food_inspection/business_count_by_tax_code_and_complaint_count?tax_code=H24&complaint_type=Complaint&complaint_count=5",
    "/v1/food_inspection/business_names_by_violation_year?year=2013&description=Contaminated%20or%20adulterated%20food",
    "/v1/food_inspection/count_businesses_by_year_postal_score?year=2015&postal_code=94102&score=90",
    "/v1/food_inspection/business_names_perfect_score_consecutive_years?score=100&consecutive_years=4",
    "/v1/food_inspection/avg_inspection_score_owner_time_location?start_year=2014&end_year=2016&owner_name=Yiu%20Tim%20Chan&address=808%20Pacific%20Ave&city=San%20Francisco",
    "/v1/food_inspection/avg_inspection_score_top_owner",
    "/v1/food_inspection/top_business_by_violations_year_risk?year=2014&risk_category=Low%20Risk",
    "/v1/food_inspection/top_owner_high_risk_violations?risk_category=High%20Risk",
    "/v1/food_inspection/top_business_avg_inspection_score",
    "/v1/food_inspection/count_businesses_highest_score_year?year=2013",
    "/v1/food_inspection/business_ids_by_inspection_type_date?inspection_type=Structural%20Inspection&date_pattern=2016-02%",
    "/v1/food_inspection/count_distinct_business_ids?risk_category=Low%20Risk&description=Unpermitted%20food%20facility",
    "/v1/food_inspection/get_violations_by_type_id?violation_type_id=103101",
    "/v1/food_inspection/latest_inspection_date_by_city?city=SAN%20BRUNO",
    "/v1/food_inspection/get_inspection_types_and_violations?business_name=ART%27S%20CAF\u00c9&risk_category=Moderate%20Risk",
    "/v1/food_inspection/get_violation_types_and_descriptions?business_name=STARBUCKS&risk_category=High%20Risk",
    "/v1/food_inspection/get_inspections_by_tax_code?tax_code=AA",
    "/v1/food_inspection/get_businesses_by_inspection_date?date=2016-07-30",
    "/v1/food_inspection/get_violations_by_owner_name?owner_name=Jade%20Chocolates%20LLC",
    "/v1/food_inspection/get_businesses_by_violation_type_id?violation_type_id=103111",
    "/v1/food_inspection/get_businesses_by_risk_category_and_date?risk_category=High%20Risk&date=2014-06-03",
    "/v1/food_inspection/highest_scoring_inspection_type?business_name=El%20Aji%20Peruvian%20Restaurant",
    "/v1/food_inspection/owner_names_by_risk_category_and_description?risk_category=High%20Risk&description=Improper%20cooking%20time%20or%20temperatures",
    "/v1/food_inspection/business_info_by_inspection_date_and_type?date=2015-02-02&inspection_type=Reinspection/Followup",
    "/v1/food_inspection/business_info_by_score_threshold?score_threshold=50",
    "/v1/food_inspection/business_count_by_address_and_city?address=1825%20POST%20St%20%23223&city=SAN%20FRANCISCO",
    "/v1/food_inspection/distinct_owner_names_by_zip?owner_zip=94104",
    "/v1/food_inspection/business_count_by_tax_code?tax_code=H25",
    "/v1/food_inspection/violation_count_by_year_and_risk_category?year=2014&risk_category=Low%20Risk",
    "/v1/food_inspection/business_ids_and_risk_categories_by_owner_name?owner_name=San%20Francisco%20Madeleine%2C%20Inc.",
    "/v1/food_inspection/distinct_owner_names_by_inspection_score?score=100",
    "/v1/food_inspection/count_businesses_by_postal_code_risk_category?postal_code=94117&risk_category=High%20Risk",
    "/v1/food_inspection/violation_types_risk_categories_by_score_range?min_score=70&max_score=80",
    "/v1/food_inspection/tax_codes_inspection_types_by_business_name?business_name=Rue%20Lepic",
    "/v1/food_inspection/business_names_by_violation_date_type_inspection_type?violation_date=2016-05-27&violation_type_id=103157&inspection_type=Routine%20-%20Unscheduled",
    "/v1/food_inspection/owner_names_by_risk_category_violation_type_description?risk_category=High%20Risk&violation_type_id=103109&description=Unclean%20or%20unsanitary%20food%20contact%20surfaces",
    "/v1/food_inspection/business_names_by_owner_city_inspection_score?owner_city=Cameron%20Park&inspection_score=100",
    "/v1/food_inspection/violation_types_by_business_id_range_address_city?min_business_id=30&max_business_id=50&address=747%20IRVING%20St&city=San%20Francisco",
    "/v1/food_inspection/owner_names_by_violation_type_date?violation_type_id=103156&violation_date=2014-06-12",
    "/v1/food_inspection/highest_score_inspection_type_by_owner_address_city?owner_address=500%20California%20St,%202nd%20Floor&owner_city=SAN%20FRANCISCO",
    "/v1/food_inspection/count_businesses_by_violation_year_inspection_type?violation_year=2016&inspection_type=Routine%20-%20Unscheduled",
    "/v1/food_inspection/business_risk_categories?score_percentage=0.8",
    "/v1/food_inspection/low_risk_violation_percentage?risk_category=Low%20Risk&max_score=95&postal_code=94110"
]
