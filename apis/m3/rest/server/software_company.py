from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/software_company/software_company.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of customers based on marital status
@app.get("/v1/software_company/customer_count_by_marital_status", operation_id="get_customer_count_by_marital_status", summary="Retrieves the total number of customers with a specified marital status from the database.")
async def get_customer_count_by_marital_status(marital_status: str = Query(..., description="Marital status of the customer")):
    cursor.execute("SELECT COUNT(ID) FROM Customers WHERE MARITAL_STATUS = ?", (marital_status,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers within a specific age range
@app.get("/v1/software_company/customer_count_by_age_range", operation_id="get_customer_count_by_age_range", summary="Retrieves the total number of customers within a specified age range. The age range is defined by the minimum and maximum age values provided as input parameters. This operation does not return individual customer data, but rather a single count value representing the number of customers that fall within the specified age range.")
async def get_customer_count_by_age_range(min_age: int = Query(..., description="Minimum age of the customer"), max_age: int = Query(..., description="Maximum age of the customer")):
    cursor.execute("SELECT COUNT(ID) FROM Customers WHERE age >= ? AND age <= ?", (min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct occupations based on education number
@app.get("/v1/software_company/distinct_occupations_by_education_number", operation_id="get_distinct_occupations_by_education_number", summary="Retrieves a unique list of occupations held by customers who have a specific education number. The education number is provided as an input parameter.")
async def get_distinct_occupations_by_education_number(education_number: int = Query(..., description="Education number of the customer")):
    cursor.execute("SELECT DISTINCT OCCUPATION FROM Customers WHERE EDUCATIONNUM = ?", (education_number,))
    result = cursor.fetchall()
    if not result:
        return {"occupations": []}
    return {"occupations": [row[0] for row in result]}

# Endpoint to get the count of mailings based on response
@app.get("/v1/software_company/mailing_count_by_response", operation_id="get_mailing_count_by_response", summary="Retrieves the total number of mailings that have a specified response status. The response status is used to filter the mailings and calculate the count.")
async def get_mailing_count_by_response(response: str = Query(..., description="Response status of the mailing")):
    cursor.execute("SELECT COUNT(REFID) custmoer_number FROM Mailings1_2 WHERE RESPONSE = ?", (response,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on occupation and age
@app.get("/v1/software_company/customer_count_by_occupation_and_age", operation_id="get_customer_count_by_occupation_and_age", summary="Retrieves the number of customers with a specified occupation and an age greater than a given value. This operation provides a breakdown of customer demographics based on occupation and age, enabling targeted marketing and customer segmentation.")
async def get_customer_count_by_occupation_and_age(occupation: str = Query(..., description="Occupation of the customer"), age: int = Query(..., description="Age of the customer")):
    cursor.execute("SELECT COUNT(ID) FROM Customers WHERE OCCUPATION = ? AND age > ?", (occupation, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on education number and sex
@app.get("/v1/software_company/customer_count_by_education_and_sex", operation_id="get_customer_count_by_education_and_sex", summary="Retrieves the total number of customers who have an education level higher than the specified value and belong to a particular gender. The education level is determined by the provided education number, and the gender is specified using the sex parameter.")
async def get_customer_count_by_education_and_sex(education_number: int = Query(..., description="Education number of the customer"), sex: str = Query(..., description="Sex of the customer")):
    cursor.execute("SELECT COUNT(ID) FROM Customers WHERE EDUCATIONNUM > ? AND SEX = ?", (education_number, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on sex and mailing response
@app.get("/v1/software_company/customer_count_by_sex_and_mailing_response", operation_id="get_customer_count_by_sex_and_mailing_response", summary="Retrieves the total number of customers who have a specific gender and a particular response status to a mailing campaign. This operation is useful for analyzing the demographic distribution of customer engagement with mailing campaigns.")
async def get_customer_count_by_sex_and_mailing_response(sex: str = Query(..., description="Sex of the customer"), response: str = Query(..., description="Response status of the mailing")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID WHERE T1.SEX = ? AND T2.RESPONSE = ?", (sex, response))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct occupations based on age and mailing response
@app.get("/v1/software_company/distinct_occupations_by_age_and_mailing_response", operation_id="get_distinct_occupations_by_age_and_mailing_response", summary="Retrieve a unique list of occupations for customers who are older than the specified age and have a particular mailing response status. This operation filters the customer database based on age and mailing response, then returns the distinct occupations found.")
async def get_distinct_occupations_by_age_and_mailing_response(age: int = Query(..., description="Age of the customer"), response: str = Query(..., description="Response status of the mailing")):
    cursor.execute("SELECT DISTINCT T1.OCCUPATION FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID WHERE T1.age > ? AND T2.RESPONSE = ?", (age, response))
    result = cursor.fetchall()
    if not result:
        return {"occupations": []}
    return {"occupations": [row[0] for row in result]}

# Endpoint to get the count of geographic IDs based on sex and number of inhabitants
@app.get("/v1/software_company/geoid_count_by_sex_and_inhabitants", operation_id="get_geoid_count_by_sex_and_inhabitants", summary="Retrieves the count of geographic IDs for customers who meet the specified sex and inhabitant criteria. The operation filters customers based on their sex and the number of inhabitants in their geographic area, which must be greater than the provided value in thousands.")
async def get_geoid_count_by_sex_and_inhabitants(sex: str = Query(..., description="Sex of the customer"), inhabitants_k: int = Query(..., description="Number of inhabitants in thousands")):
    cursor.execute("SELECT COUNT(T1.GEOID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.SEX = ? AND T2.INHABITANTS_K > ?", (sex, inhabitants_k))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on the highest income
@app.get("/v1/software_company/customer_count_by_highest_income", operation_id="get_customer_count_by_highest_income", summary="Retrieves the total number of customers residing in the geographic area with the highest average income. This operation utilizes demographic data to determine the area with the highest income and then counts the corresponding customers.")
async def get_customer_count_by_highest_income():
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID ORDER BY T2.INCOME_K DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on occupation and inhabitants range
@app.get("/v1/software_company/customer_count_occupation_inhabitants", operation_id="get_customer_count_occupation_inhabitants", summary="Retrieves the total number of customers with a specified occupation and inhabitants range. The operation filters customers based on their occupation and the inhabitants range of their geographical location. The inhabitants range is defined by a minimum and maximum number of inhabitants (in thousands).")
async def get_customer_count_occupation_inhabitants(occupation: str = Query(..., description="Occupation of the customer"), min_inhabitants: int = Query(..., description="Minimum number of inhabitants (in thousands)"), max_inhabitants: int = Query(..., description="Maximum number of inhabitants (in thousands)")):
    cursor.execute("SELECT COUNT(T1.GEOID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.OCCUPATION = ? AND T2.INHABITANTS_K > ? AND T2.INHABITANTS_K < ?", (occupation, min_inhabitants, max_inhabitants))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer ID based on specific IDs and order by inhabitants
@app.get("/v1/software_company/customer_id_by_ids_inhabitants", operation_id="get_customer_id_by_ids_inhabitants", summary="Retrieves the customer ID with the highest number of inhabitants from the provided list of customer IDs. The operation filters the customers based on the given IDs and sorts them in descending order by the number of inhabitants. The customer ID with the highest number of inhabitants is then returned.")
async def get_customer_id_by_ids_inhabitants(id1: int = Query(..., description="First customer ID"), id2: int = Query(..., description="Second customer ID")):
    cursor.execute("SELECT T1.ID FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.ID = ? OR T1.ID = ? ORDER BY INHABITANTS_K DESC LIMIT 1", (id1, id2))
    result = cursor.fetchone()
    if not result:
        return {"id": []}
    return {"id": result[0]}

# Endpoint to get the count of customers based on inhabitants and response
@app.get("/v1/software_company/customer_count_inhabitants_response", operation_id="get_customer_count_inhabitants_response", summary="Retrieves the count of customers who reside in areas with more than the specified minimum number of inhabitants and have the given response status. This operation considers the customer's geographical location and their response to mailings.")
async def get_customer_count_inhabitants_response(min_inhabitants: int = Query(..., description="Minimum number of inhabitants (in thousands)"), response: str = Query(..., description="Response status")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID INNER JOIN Demog AS T3 ON T1.GEOID = T3.GEOID WHERE T3.INHABITANTS_K > ? AND T2.RESPONSE = ?", (min_inhabitants, response))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on sex, marital status, and response
@app.get("/v1/software_company/customer_count_sex_marital_response", operation_id="get_customer_count_sex_marital_response", summary="Retrieves the total number of customers who meet the specified criteria for sex, marital status, and response. This operation is useful for obtaining a precise count of customers based on these demographic and response attributes.")
async def get_customer_count_sex_marital_response(sex: str = Query(..., description="Sex of the customer"), marital_status: str = Query(..., description="Marital status of the customer"), response: str = Query(..., description="Response status")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID WHERE T1.SEX = ? AND T1.MARITAL_STATUS = ? AND T2.RESPONSE = ?", (sex, marital_status, response))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on response and order by income
@app.get("/v1/software_company/customer_count_response_income", operation_id="get_customer_count_response_income", summary="Retrieves the count of customers who have a specific response status, sorted by their income in descending order. The response status is provided as an input parameter.")
async def get_customer_count_response_income(response: str = Query(..., description="Response status")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID INNER JOIN Demog AS T3 ON T1.GEOID = T3.GEOID WHERE T2.RESPONSE = ? ORDER BY T3.INCOME_K DESC LIMIT 1", (response,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct inhabitants count ordered by inhabitants
@app.get("/v1/software_company/distinct_inhabitants", operation_id="get_distinct_inhabitants", summary="Retrieves a distinct count of inhabitants, sorted in descending order. This operation fetches unique inhabitants data from the database, providing a comprehensive overview of the population distribution.")
async def get_distinct_inhabitants():
    cursor.execute("SELECT DISTINCT T2.INHABITANTS_K FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID ORDER BY T2.INHABITANTS_K DESC")
    result = cursor.fetchall()
    if not result:
        return {"inhabitants": []}
    return {"inhabitants": [row[0] for row in result]}

# Endpoint to get the count of customers based on inhabitants and sex
@app.get("/v1/software_company/customer_count_inhabitants_sex", operation_id="get_customer_count_inhabitants_sex", summary="Retrieves the total count of customers who reside in a location with a specific number of inhabitants (in thousands) and belong to a particular sex category. This operation provides a quantitative measure of customer distribution based on population density and gender.")
async def get_customer_count_inhabitants_sex(inhabitants: float = Query(..., description="Number of inhabitants (in thousands)"), sex: str = Query(..., description="Sex of the customer")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T2.INHABITANTS_K = ? AND T1.SEX = ?", (inhabitants, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on age range and response
@app.get("/v1/software_company/customer_count_age_response", operation_id="get_customer_count_age_response", summary="Retrieves the count of customers who fall within a specified age range and have a particular response status. The age range is defined by a minimum and maximum age, while the response status indicates whether the customer has responded to a mailing campaign.")
async def get_customer_count_age_response(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), response: str = Query(..., description="Response status")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID WHERE T1.age >= ? AND T1.age <= ? AND T2.RESPONSE = ?", (min_age, max_age, response))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average education number ordered by income
@app.get("/v1/software_company/average_education_income", operation_id="get_average_education_income", summary="Retrieves the average education number from the software company's customer database, ordered by income in descending order. The result represents the average education level of the highest-earning customers.")
async def get_average_education_income():
    cursor.execute("SELECT AVG(T1.EDUCATIONNUM) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID ORDER BY T2.INCOME_K DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"average_education": []}
    return {"average_education": result[0]}

# Endpoint to get the average age of customers based on response
@app.get("/v1/software_company/average_age_response", operation_id="get_average_age_response", summary="Retrieves the average age of customers who have a specific response status. The response status is used to filter the customers and calculate the average age.")
async def get_average_age_response(response: str = Query(..., description="Response status")):
    cursor.execute("SELECT AVG(T1.age) FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID WHERE T2.RESPONSE = ?", (response,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the count of customers based on gender
@app.get("/v1/software_company/customer_count_by_gender", operation_id="get_customer_count_by_gender", summary="Retrieves the total number of customers categorized by gender. The operation requires the specification of a gender to filter the count accordingly.")
async def get_customer_count_by_gender(sex: str = Query(..., description="Gender of the customer (e.g., 'Male', 'Female')")):
    cursor.execute("SELECT COUNT(ID) FROM Customers WHERE SEX = ?", (sex,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get GEOIDs of customers based on occupation
@app.get("/v1/software_company/customer_geoids_by_occupation", operation_id="get_customer_geoids_by_occupation", summary="Retrieves the GEOIDs of customers who have the specified occupation. The operation filters the customer database based on the provided occupation and returns the corresponding GEOIDs.")
async def get_customer_geoids_by_occupation(occupation: str = Query(..., description="Occupation of the customer (e.g., 'Handlers-cleaners')")):
    cursor.execute("SELECT GEOID FROM Customers WHERE OCCUPATION = ?", (occupation,))
    result = cursor.fetchall()
    if not result:
        return {"geoids": []}
    return {"geoids": [row[0] for row in result]}

# Endpoint to get the count of customers under a certain age
@app.get("/v1/software_company/customer_count_under_age", operation_id="get_customer_count_under_age", summary="Retrieves the total number of customers who are younger than the specified age. The age parameter is used to filter the customers.")
async def get_customer_count_under_age(age: int = Query(..., description="Maximum age of the customer")):
    cursor.execute("SELECT COUNT(ID) FROM Customers WHERE age < ?", (age,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get GEOIDs based on income range
@app.get("/v1/software_company/geoids_by_income_range", operation_id="get_geoids_by_income_range", summary="Retrieves the GEOIDs of areas with an income range between the specified minimum and maximum values (in thousands).")
async def get_geoids_by_income_range(min_income: int = Query(..., description="Minimum income in thousands"), max_income: int = Query(..., description="Maximum income in thousands")):
    cursor.execute("SELECT GEOID FROM Demog WHERE INCOME_K >= ? AND INCOME_K <= ?", (min_income, max_income))
    result = cursor.fetchall()
    if not result:
        return {"geoids": []}
    return {"geoids": [row[0] for row in result]}

# Endpoint to get the count of GEOIDs based on inhabitants and GEOID range
@app.get("/v1/software_company/geoid_count_by_inhabitants_and_range", operation_id="get_geoid_count_by_inhabitants_and_range", summary="Retrieves the total count of geographical areas (GEOIDs) that have a population less than the specified number of inhabitants (in thousands) and fall within the provided GEOID range.")
async def get_geoid_count_by_inhabitants_and_range(inhabitants: int = Query(..., description="Maximum number of inhabitants in thousands"), min_geoid: int = Query(..., description="Minimum GEOID"), max_geoid: int = Query(..., description="Maximum GEOID")):
    cursor.execute("SELECT COUNT(GEOID) FROM Demog WHERE INHABITANTS_K < ? AND GEOID >= ? AND GEOID <= ?", (inhabitants, min_geoid, max_geoid))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the number of inhabitants based on GEOID
@app.get("/v1/software_company/inhabitants_by_geoid", operation_id="get_inhabitants_by_geoid", summary="Retrieves the number of inhabitants for a specific area identified by its GEOID. The GEOID is a unique identifier for a geographic area, which is used to look up the corresponding inhabitants count in the Demog table.")
async def get_inhabitants_by_geoid(geoid: int = Query(..., description="GEOID of the area")):
    cursor.execute("SELECT INHABITANTS_K FROM Demog WHERE GEOID = ?", (geoid,))
    result = cursor.fetchone()
    if not result:
        return {"inhabitants": []}
    return {"inhabitants": result[0]}

# Endpoint to get education and occupation of customers based on income and age range
@app.get("/v1/software_company/customer_education_occupation_by_income_age", operation_id="get_customer_education_occupation_by_income_age", summary="Retrieves the education level and occupation of customers who have an income less than the specified maximum and fall within the defined age range.")
async def get_customer_education_occupation_by_income_age(income: int = Query(..., description="Maximum income in thousands"), min_age: int = Query(..., description="Minimum age of the customer"), max_age: int = Query(..., description="Maximum age of the customer")):
    cursor.execute("SELECT T1.EDUCATIONNUM, T1.OCCUPATION FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T2.INCOME_K < ? AND T1.age >= ? AND T1.age <= ?", (income, min_age, max_age))
    result = cursor.fetchall()
    if not result:
        return {"education_occupation": []}
    return {"education_occupation": [{"education": row[0], "occupation": row[1]} for row in result]}

# Endpoint to get the count of customers based on marital status and age
@app.get("/v1/software_company/customer_count_by_marital_status_age", operation_id="get_customer_count_by_marital_status_age", summary="Retrieves the total number of customers who are below a specified age and have a particular marital status. This operation provides a demographic breakdown of the customer base, enabling targeted marketing strategies.")
async def get_customer_count_by_marital_status_age(marital_status: str = Query(..., description="Marital status of the customer (e.g., 'Divorced')"), age: int = Query(..., description="Maximum age of the customer")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.MARITAL_STATUS = ? AND T1.age < ?", (marital_status, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the GEOID and income of the oldest customer
@app.get("/v1/software_company/oldest_customer_geoid_income", operation_id="get_oldest_customer_geoid_income", summary="Retrieves the geographical identifier (GEOID) and income of the oldest customer from the database. The operation joins the Customers and Demog tables on the GEOID field and returns the GEOID and income of the customer with the highest age.")
async def get_oldest_customer_geoid_income():
    cursor.execute("SELECT T1.GEOID, T2.INCOME_K FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID ORDER BY T1.age DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"geoid": [], "income": []}
    return {"geoid": result[0], "income": result[1]}

# Endpoint to get incomes based on education and gender
@app.get("/v1/software_company/incomes_by_education_gender", operation_id="get_incomes_by_education_gender", summary="Retrieves the income data for customers with an education level below the specified maximum and of the given gender. The data is filtered based on the provided education number and gender.")
async def get_incomes_by_education_gender(education_num: int = Query(..., description="Maximum education number"), sex: str = Query(..., description="Gender of the customer (e.g., 'Male', 'Female')")):
    cursor.execute("SELECT INCOME_K FROM Demog WHERE GEOID IN ( SELECT GEOID FROM Customers WHERE EDUCATIONNUM < ? AND SEX = ? )", (education_num, sex))
    result = cursor.fetchall()
    if not result:
        return {"incomes": []}
    return {"incomes": [row[0] for row in result]}

# Endpoint to get occupation and income based on education level and gender
@app.get("/v1/software_company/occupation_income_by_education_gender", operation_id="get_occupation_income", summary="Retrieves occupation and income data for individuals within a specified range of education levels and a particular gender. The data is filtered based on the minimum and maximum education levels provided, as well as the gender. The response includes the occupation and corresponding income.")
async def get_occupation_income(min_education: int = Query(..., description="Minimum education level"), max_education: int = Query(..., description="Maximum education level"), sex: str = Query(..., description="Gender")):
    cursor.execute("SELECT T1.OCCUPATION, T2.INCOME_K FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.EDUCATIONNUM >= ? AND T1.EDUCATIONNUM <= ? AND T1.SEX = ?", (min_education, max_education, sex))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of customers based on age, marital status, gender, and income range
@app.get("/v1/software_company/customer_count_by_age_marital_gender_income", operation_id="get_customer_count", summary="Retrieve the number of customers within a specified age range, marital status, gender, and income bracket. This operation provides a detailed breakdown of customer demographics, enabling targeted marketing and segmentation strategies.")
async def get_customer_count(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), marital_status: str = Query(..., description="Marital status"), sex: str = Query(..., description="Gender"), min_income: int = Query(..., description="Minimum income in thousands"), max_income: int = Query(..., description="Maximum income in thousands")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.age >= ? AND T1.age <= ? AND T1.MARITAL_STATUS = ? AND T1.SEX = ? AND T2.INCOME_K >= ? AND T2.INCOME_K <= ?", (min_age, max_age, marital_status, sex, min_income, max_income))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct occupations based on the number of inhabitants
@app.get("/v1/software_company/distinct_occupations_by_inhabitants", operation_id="get_distinct_occupations", summary="Retrieves a list of unique occupations for areas with a population ranging between the specified minimum and maximum number of inhabitants (in thousands).")
async def get_distinct_occupations(min_inhabitants: int = Query(..., description="Minimum number of inhabitants in thousands"), max_inhabitants: int = Query(..., description="Maximum number of inhabitants in thousands")):
    cursor.execute("SELECT DISTINCT T1.OCCUPATION FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T2.INHABITANTS_K >= ? AND T2.INHABITANTS_K <= ?", (min_inhabitants, max_inhabitants))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get income based on education level, gender, and marital status
@app.get("/v1/software_company/income_by_education_gender_marital", operation_id="get_income", summary="Retrieves the average income of customers with an education level less than the provided value, filtered by gender and marital status.")
async def get_income(education_num: int = Query(..., description="Education level"), sex: str = Query(..., description="Gender"), marital_status: str = Query(..., description="Marital status")):
    cursor.execute("SELECT INCOME_K FROM Demog WHERE GEOID IN ( SELECT GEOID FROM Customers WHERE EDUCATIONNUM < ? AND SEX = ? AND MARITAL_STATUS = ? )", (education_num, sex, marital_status))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the marital status of the highest income customer within a specific age range
@app.get("/v1/software_company/highest_income_marital_status_by_age", operation_id="get_highest_income_marital_status", summary="Retrieves the marital status of the customer with the highest income within the specified age range. The age range is defined by the minimum and maximum age parameters. The operation returns the marital status of the customer who has the highest income among those within the provided age range.")
async def get_highest_income_marital_status(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age")):
    cursor.execute("SELECT T1.MARITAL_STATUS FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.age >= ? AND T1.age <= ? ORDER BY T2.INCOME_K DESC LIMIT 1", (min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result[0]}

# Endpoint to get the number of inhabitants based on occupation, gender, and age range
@app.get("/v1/software_company/inhabitants_by_occupation_gender_age", operation_id="get_inhabitants", summary="Retrieves the count of inhabitants based on their occupation, gender, and age range. The operation filters inhabitants by a specific occupation, gender, and age range, and returns the corresponding count.")
async def get_inhabitants(occupation: str = Query(..., description="Occupation"), sex: str = Query(..., description="Gender"), min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age")):
    cursor.execute("SELECT T2.INHABITANTS_K FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.OCCUPATION = ? AND T1.SEX = ? AND T1.age >= ? AND T1.age <= ?", (occupation, sex, min_age, max_age))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of customers based on gender, age range, and number of inhabitants
@app.get("/v1/software_company/customer_count_by_gender_age_inhabitants", operation_id="get_customer_count_by_gender_age_inhabitants", summary="Retrieves the count of customers based on specified gender, age range, and population size. The age range is defined by a minimum and maximum age, while the population size is determined by a range of thousands of inhabitants. This operation provides a demographic breakdown of the customer base.")
async def get_customer_count_by_gender_age_inhabitants(sex: str = Query(..., description="Gender"), min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), min_inhabitants: int = Query(..., description="Minimum number of inhabitants in thousands"), max_inhabitants: int = Query(..., description="Maximum number of inhabitants in thousands")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.SEX = ? AND T1.age >= ? AND T1.age <= ? AND T2.INHABITANTS_K >= ? AND T2.INHABITANTS_K <= ?", (sex, min_age, max_age, min_inhabitants, max_inhabitants))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get income and inhabitants based on age criteria
@app.get("/v1/software_company/income_inhabitants_by_age_criteria", operation_id="get_income_inhabitants", summary="Retrieves the income and population data for areas where the average age of customers is at least 80% of the overall average age. The data is aggregated by income and population categories.")
async def get_income_inhabitants():
    cursor.execute("SELECT T2.INCOME_K, T2.INHABITANTS_K FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID GROUP BY T2.INCOME_K, T2.INHABITANTS_K HAVING T1.age > 0.8 * AVG(T1.age)")
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of customers with income above a certain threshold based on marital status
@app.get("/v1/software_company/income_percentage_by_marital_status", operation_id="get_income_percentage", summary="Retrieves the percentage of customers with an income above a specified threshold, categorized by their marital status. This operation calculates the proportion of customers who meet the income criterion within each marital status category.")
async def get_income_percentage(income_threshold: int = Query(..., description="Income threshold in thousands"), marital_status: str = Query(..., description="Marital status")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.INCOME_K > ? THEN 1.0 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.MARITAL_STATUS = ?", (income_threshold, marital_status))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get customer IDs and GEOIDs based on education level and age
@app.get("/v1/software_company/customer_ids_geoids_by_education_age", operation_id="get_customer_ids_geoids", summary="Retrieves the unique identifiers and geographical codes of customers who have an education level below the specified threshold and are older than the provided age.")
async def get_customer_ids_geoids(education_num: int = Query(..., description="Education level"), age: int = Query(..., description="Age")):
    cursor.execute("SELECT ID, GEOID FROM Customers WHERE EDUCATIONNUM < ? AND age > ?", (education_num, age))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the average income
@app.get("/v1/software_company/average_income", operation_id="get_average_income", summary="Retrieves the average income of individuals in the demographic dataset. This endpoint calculates the mean value of the income field, providing a statistical summary of the financial status of the population.")
async def get_average_income():
    cursor.execute("SELECT AVG(INCOME_K) FROM Demog")
    result = cursor.fetchone()
    if not result:
        return {"average_income": []}
    return {"average_income": result[0]}

# Endpoint to get the count of teenagers based on occupation and age range
@app.get("/v1/software_company/teenager_count_occupation_age_range", operation_id="get_teenager_count", summary="Retrieves the number of teenage customers categorized by a specific occupation and within a defined age range.")
async def get_teenager_count(occupation: str = Query(..., description="Occupation"), min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age")):
    cursor.execute("SELECT COUNT(ID) teenager_number FROM Customers WHERE OCCUPATION = ? AND age >= ? AND age <= ?", (occupation, min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of GEOIDs based on occupation and number of inhabitants
@app.get("/v1/software_company/geoid_count_occupation_inhabitants", operation_id="get_geoid_count", summary="Retrieves the count of geographical areas (GEOIDs) where the specified occupation is prevalent and the number of inhabitants exceeds the provided minimum threshold.")
async def get_geoid_count(occupation: str = Query(..., description="Occupation"), min_inhabitants: int = Query(..., description="Minimum number of inhabitants")):
    cursor.execute("SELECT COUNT(T2.GEOID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.OCCUPATION = ? AND T2.INHABITANTS_K > ?", (occupation, min_inhabitants))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of customers with a specific marital status in a given GEOID
@app.get("/v1/software_company/marital_status_percentage_geoid", operation_id="get_marital_status_percentage", summary="Retrieves the percentage of customers with a specified marital status within a given GEOID. This operation calculates the proportion of customers with the provided marital status in a specific geographical area, as defined by the GEOID.")
async def get_marital_status_percentage(marital_status: str = Query(..., description="Marital status"), geoid: int = Query(..., description="GEOID")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.MARITAL_STATUS = ? THEN 1.0 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.GEOID = ?", (marital_status, geoid))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of customers in a specific age range with income above a certain threshold
@app.get("/v1/software_company/age_range_percentage_income", operation_id="get_age_range_percentage", summary="Retrieves the percentage of customers within a specified age range who have an income above a given threshold. This operation calculates the proportion of customers between the provided minimum and maximum ages whose income exceeds the defined minimum income level.")
async def get_age_range_percentage(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), min_income: int = Query(..., description="Minimum income")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.age BETWEEN ? AND ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T2.INCOME_K > ?", (min_age, max_age, min_income))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of mailings based on response
@app.get("/v1/software_company/mailing_count_response", operation_id="get_mailing_count", summary="Retrieves the total number of mailings that match a specific response value. This operation allows you to determine the count of mailings based on the provided response parameter, providing insights into the volume of mailings associated with a particular response.")
async def get_mailing_count(response: str = Query(..., description="Response value")):
    cursor.execute("SELECT COUNT(REFID) FROM Mailings1_2 WHERE RESPONSE = ?", (response,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the list of REFIDs based on response
@app.get("/v1/software_company/mailing_refids_response", operation_id="get_mailing_refids", summary="Retrieves a list of unique REFIDs from the mailings database, filtered by a specific response value. This operation is useful for identifying mailings that have received a particular response.")
async def get_mailing_refids(response: str = Query(..., description="Response value")):
    cursor.execute("SELECT REFID FROM Mailings1_2 WHERE RESPONSE = ?", (response,))
    result = cursor.fetchall()
    if not result:
        return {"refids": []}
    return {"refids": [row[0] for row in result]}

# Endpoint to get GEOIDs based on the number of inhabitants
@app.get("/v1/software_company/geoids_by_inhabitants", operation_id="get_geoids_by_inhabitants", summary="Retrieves a list of GEOIDs for areas with a population less than the specified value. The input parameter represents the maximum population threshold.")
async def get_geoids_by_inhabitants(inhabitants_k: int = Query(..., description="Number of inhabitants (must be an integer)")):
    cursor.execute("SELECT GEOID FROM Demog WHERE INHABITANTS_K < ?", (inhabitants_k,))
    result = cursor.fetchall()
    if not result:
        return {"geoids": []}
    return {"geoids": [row[0] for row in result]}

# Endpoint to get the count of GEOIDs based on income and GEOID range
@app.get("/v1/software_company/count_geoids_by_income_and_range", operation_id="get_count_geoids_by_income_and_range", summary="Retrieves the count of geographical areas (GEOIDs) where the average income is less than the provided value and the GEOID falls within the specified range. The income and GEOID range values must be integers.")
async def get_count_geoids_by_income_and_range(income_k: int = Query(..., description="Income (must be an integer)"), geoid_min: int = Query(..., description="Minimum GEOID (must be an integer)"), geoid_max: int = Query(..., description="Maximum GEOID (must be an integer)")):
    cursor.execute("SELECT COUNT(GEOID) FROM Demog WHERE INCOME_K < ? AND GEOID >= ? AND GEOID <= ?", (income_k, geoid_min, geoid_max))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct marital statuses based on education number and age
@app.get("/v1/software_company/distinct_marital_statuses", operation_id="get_distinct_marital_statuses", summary="Retrieves the unique marital statuses of customers who have a specific education level and age. The operation filters customers based on the provided education number and age, then returns a list of distinct marital statuses.")
async def get_distinct_marital_statuses(education_num: int = Query(..., description="Education number (must be an integer)"), age: int = Query(..., description="Age (must be an integer)")):
    cursor.execute("SELECT DISTINCT MARITAL_STATUS FROM Customers WHERE EDUCATIONNUM = ? AND age = ?", (education_num, age))
    result = cursor.fetchall()
    if not result:
        return {"marital_statuses": []}
    return {"marital_statuses": [row[0] for row in result]}

# Endpoint to get the count of customers based on marital status and response
@app.get("/v1/software_company/count_customers_by_marital_status_and_response", operation_id="get_count_customers_by_marital_status_and_response", summary="Retrieves the total number of customers who have a specified marital status and response to a mailing campaign. The response parameter should be either 'true' or 'false'.")
async def get_count_customers_by_marital_status_and_response(marital_status: str = Query(..., description="Marital status"), response: str = Query(..., description="Response (must be 'true' or 'false')")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID INNER JOIN Demog AS T3 ON T1.GEOID = T3.GEOID WHERE T1.MARITAL_STATUS = ? AND T2.RESPONSE = ?", (marital_status, response))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the response and number of inhabitants for the oldest customer of a specific sex
@app.get("/v1/software_company/response_and_inhabitants_by_sex", operation_id="get_response_and_inhabitants_by_sex", summary="Retrieves the response and the number of inhabitants for the oldest customer of a specific sex. The sex is provided as an input parameter. The response is obtained from the Mailings1_2 table, while the number of inhabitants is derived from the Demog table. The data is filtered by the sex of the customer and ordered by age in descending order. Only the oldest customer's data is returned.")
async def get_response_and_inhabitants_by_sex(sex: str = Query(..., description="Sex (must be 'Male' or 'Female')")):
    cursor.execute("SELECT T2.RESPONSE, T3.INHABITANTS_K FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID INNER JOIN Demog AS T3 ON T1.GEOID = T3.GEOID WHERE T1.SEX = ? ORDER BY T1.age DESC LIMIT 1", (sex,))
    result = cursor.fetchone()
    if not result:
        return {"response": [], "inhabitants_k": []}
    return {"response": result[0], "inhabitants_k": result[1]}

# Endpoint to get education number and income based on age range and response
@app.get("/v1/software_company/education_and_income_by_age_and_response", operation_id="get_education_and_income_by_age_and_response", summary="Retrieves the education level and income of customers who fall within a specified age range and have a particular response status. The age range is defined by a minimum and maximum age, both of which must be integers. The response status must be either 'true' or 'false'.")
async def get_education_and_income_by_age_and_response(age_min: int = Query(..., description="Minimum age (must be an integer)"), age_max: int = Query(..., description="Maximum age (must be an integer)"), response: str = Query(..., description="Response (must be 'true' or 'false')")):
    cursor.execute("SELECT T1.EDUCATIONNUM, T3.INCOME_K FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID INNER JOIN Demog AS T3 ON T1.GEOID = T3.GEOID WHERE T1.age >= ? AND T1.age <= ? AND T2.RESPONSE = ?", (age_min, age_max, response))
    result = cursor.fetchall()
    if not result:
        return {"education_num": [], "income_k": []}
    return {"education_num": [row[0] for row in result], "income_k": [row[1] for row in result]}

# Endpoint to get the count of customers based on sex, age range, and income range
@app.get("/v1/software_company/count_customers_by_sex_age_and_income", operation_id="get_count_customers_by_sex_age_and_income", summary="Retrieve the number of customers of a specific gender within a defined age bracket and income bracket. The gender must be either 'Male' or 'Female'. The age and income ranges are inclusive and must be provided as integers.")
async def get_count_customers_by_sex_age_and_income(sex: str = Query(..., description="Sex (must be 'Male' or 'Female')"), age_min: int = Query(..., description="Minimum age (must be an integer)"), age_max: int = Query(..., description="Maximum age (must be an integer)"), income_min: int = Query(..., description="Minimum income (must be an integer)"), income_max: int = Query(..., description="Maximum income (must be an integer)")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.SEX = ? AND T1.age >= ? AND T1.age <= ? AND T2.INCOME_K >= ? AND T2.INCOME_K <= ?", (sex, age_min, age_max, income_min, income_max))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get education number and response for the customer with the highest number of inhabitants in a specified age range
@app.get("/v1/software_company/education_and_response_by_age_and_inhabitants", operation_id="get_education_and_response_by_age_and_inhabitants", summary="Retrieves the education number and response for the customer with the highest number of inhabitants within the specified age range. The age range is defined by the minimum and maximum age parameters, both of which must be integers.")
async def get_education_and_response_by_age_and_inhabitants(age_min: int = Query(..., description="Minimum age (must be an integer)"), age_max: int = Query(..., description="Maximum age (must be an integer)")):
    cursor.execute("SELECT T1.EDUCATIONNUM, T2.RESPONSE FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID INNER JOIN Demog AS T3 ON T1.GEOID = T3.GEOID WHERE T1.age >= ? AND T1.age <= ? ORDER BY T3.INHABITANTS_K DESC LIMIT 1", (age_min, age_max))
    result = cursor.fetchone()
    if not result:
        return {"education_num": [], "response": []}
    return {"education_num": result[0], "response": result[1]}

# Endpoint to get income based on sex, age range, and occupation
@app.get("/v1/software_company/income_by_sex_age_and_occupation", operation_id="get_income_by_sex_age_and_occupation", summary="Retrieves the average income of customers within a specific sex, age range, and occupation. The operation requires the sex, minimum age, maximum age, and occupation as input parameters. The sex parameter must be either 'Male' or 'Female'. The age parameters should be integers representing the lower and upper bounds of the age range. The occupation parameter specifies the job or profession of the customers.")
async def get_income_by_sex_age_and_occupation(sex: str = Query(..., description="Sex (must be 'Male' or 'Female')"), age_min: int = Query(..., description="Minimum age (must be an integer)"), age_max: int = Query(..., description="Maximum age (must be an integer)"), occupation: str = Query(..., description="Occupation")):
    cursor.execute("SELECT T2.INCOME_K FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.SEX = ? AND T1.age >= ? AND T1.age <= ? AND T1.OCCUPATION = ?", (sex, age_min, age_max, occupation))
    result = cursor.fetchall()
    if not result:
        return {"income_k": []}
    return {"income_k": [row[0] for row in result]}

# Endpoint to get distinct marital status and response based on education number and sex
@app.get("/v1/software_company/marital_status_response", operation_id="get_marital_status_response", summary="Retrieves unique combinations of marital status and response from customers who have an education number greater than the provided value and are of the specified sex.")
async def get_marital_status_response(education_num: int = Query(..., description="Education number"), sex: str = Query(..., description="Sex of the customer")):
    cursor.execute("SELECT DISTINCT T1.MARITAL_STATUS, T2.RESPONSE FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID WHERE T1.EDUCATIONNUM > ? AND T1.SEX = ?", (education_num, sex))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get ages of customers based on inhabitants count and sex
@app.get("/v1/software_company/customer_ages", operation_id="get_customer_ages", summary="Retrieves the ages of customers from areas with a population less than the specified inhabitants count and of the specified sex.")
async def get_customer_ages(inhabitants_k: int = Query(..., description="Inhabitants count"), sex: str = Query(..., description="Sex of the customer")):
    cursor.execute("SELECT age FROM Customers WHERE GEOID IN ( SELECT GEOID FROM Demog WHERE INHABITANTS_K < ? ) AND SEX = ?", (inhabitants_k, sex))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct income and response based on education number, sex, and marital status
@app.get("/v1/software_company/income_response", operation_id="get_income_response", summary="Retrieves unique income and response data for customers who meet specific education, sex, and marital status criteria. The data is filtered based on the provided education number, sex, and marital status, ensuring that only relevant customer information is returned.")
async def get_income_response(education_num: int = Query(..., description="Education number"), sex: str = Query(..., description="Sex of the customer"), marital_status: str = Query(..., description="Marital status of the customer")):
    cursor.execute("SELECT DISTINCT T3.INCOME_K, T2.RESPONSE FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID INNER JOIN Demog AS T3 ON T1.GEOID = T3.GEOID WHERE T1.EDUCATIONNUM > ? AND T1.SEX = ? AND T1.MARITAL_STATUS = ?", (education_num, sex, marital_status))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct occupation and response based on sex and inhabitants count range
@app.get("/v1/software_company/occupation_response", operation_id="get_occupation_response", summary="Retrieves unique occupation and response combinations from customers, filtered by sex and inhabitants count range. The inhabitants count range is defined by a minimum and maximum value.")
async def get_occupation_response(sex: str = Query(..., description="Sex of the customer"), min_inhabitants_k: int = Query(..., description="Minimum inhabitants count"), max_inhabitants_k: int = Query(..., description="Maximum inhabitants count")):
    cursor.execute("SELECT DISTINCT T1.OCCUPATION, T2.RESPONSE FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID INNER JOIN Demog AS T3 ON T1.GEOID = T3.GEOID WHERE T1.SEX = ? AND T3.INHABITANTS_K >= ? AND T3.INHABITANTS_K <= ?", (sex, min_inhabitants_k, max_inhabitants_k))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of positive responses based on occupation and sex
@app.get("/v1/software_company/positive_response_percentage", operation_id="get_positive_response_percentage", summary="Retrieves the percentage of a specific customer response, filtered by occupation and sex. This operation calculates the proportion of a particular response (e.g., positive, negative, or neutral) among customers with a given occupation and sex. The result is expressed as a percentage.")
async def get_positive_response_percentage(response: str = Query(..., description="Response value"), occupation: str = Query(..., description="Occupation of the customer"), sex: str = Query(..., description="Sex of the customer")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.RESPONSE = ? THEN 1.0 ELSE 0 END) AS REAL) * 100 / COUNT(T2.REFID) FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID WHERE T1.OCCUPATION = ? AND T1.SEX = ?", (response, occupation, sex))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of male to female customers based on age range and education number
@app.get("/v1/software_company/male_female_ratio", operation_id="get_male_female_ratio", summary="Retrieves the ratio of male to female customers within a specified age range and education level. The operation calculates the ratio based on the provided sex values for male and female, age range, and minimum education number. The result reflects the proportion of male to female customers that meet the given criteria.")
async def get_male_female_ratio(sex_male: str = Query(..., description="Sex value for male"), sex_female: str = Query(..., description="Sex value for female"), min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), education_num: int = Query(..., description="Education number")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN SEX = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN SEX = ? THEN 1 ELSE 0 END) FROM Customers WHERE age BETWEEN ? AND ? AND EDUCATIONNUM > ?", (sex_male, sex_female, min_age, max_age, education_num))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get GEOID and annual income based on income threshold
@app.get("/v1/software_company/annual_income", operation_id="get_annual_income", summary="Retrieves the annual income and GEOID for all areas where the average income per thousand inhabitants exceeds the provided income threshold.")
async def get_annual_income(income_k: int = Query(..., description="Income threshold")):
    cursor.execute("SELECT GEOID, INHABITANTS_K * INCOME_K * 12 FROM Demog WHERE INCOME_K > ?", (income_k,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the most common response
@app.get("/v1/software_company/most_common_response", operation_id="get_most_common_response", summary="Retrieves the most frequently occurring response from the mailings data. This operation returns the response that has been recorded the most times in the mailings dataset, providing valuable insights into the most common customer feedback or reaction to the mailings.")
async def get_most_common_response():
    cursor.execute("SELECT RESPONSE FROM Mailings1_2 GROUP BY RESPONSE ORDER BY COUNT(RESPONSE) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"response": []}
    return {"response": result[0]}

# Endpoint to get annual income based on sex and occupation
@app.get("/v1/software_company/annual_income_by_sex_occupation", operation_id="get_annual_income_by_sex_occupation", summary="Retrieves the estimated annual income of customers based on their sex and occupation. The calculation is derived from the total number of inhabitants and average income in the customer's geographical area, multiplied by twelve to represent a yearly figure.")
async def get_annual_income_by_sex_occupation(sex: str = Query(..., description="Sex of the customer"), occupation: str = Query(..., description="Occupation of the customer")):
    cursor.execute("SELECT T2.INHABITANTS_K * T2.INCOME_K * 12 FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.SEX = ? AND T1.OCCUPATION = ?", (sex, occupation))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get customer details based on demographic and personal attributes
@app.get("/v1/software_company/customer_details_by_attributes", operation_id="get_customer_details", summary="Retrieves customer education level, occupation, and age based on demographic and personal attributes. The operation filters customers by the number of inhabitants in their area, sex, and marital status.")
async def get_customer_details(inhabitants_k: float = Query(..., description="Number of inhabitants in thousands"), sex: str = Query(..., description="Sex of the customer"), marital_status: str = Query(..., description="Marital status of the customer")):
    cursor.execute("SELECT T1.EDUCATIONNUM, T1.OCCUPATION, T1.age FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T2.INHABITANTS_K = ? AND T1.SEX = ? AND T1.MARITAL_STATUS = ?", (inhabitants_k, sex, marital_status))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get mailing responses based on customer geographic identifier
@app.get("/v1/software_company/mailing_responses_by_geoid", operation_id="get_mailing_responses", summary="Retrieves the mailing responses associated with a specific customer based on their geographic identifier. The operation returns the response data for the customer with the provided geographic identifier.")
async def get_mailing_responses(geoid: int = Query(..., description="Geographic identifier of the customer")):
    cursor.execute("SELECT T2.RESPONSE FROM Customers AS T1 INNER JOIN mailings3 AS T2 ON T1.ID = T2.REFID WHERE T1.GEOID = ?", (geoid,))
    result = cursor.fetchall()
    if not result:
        return {"responses": []}
    return {"responses": result}

# Endpoint to get income and demographic details based on customer IDs
@app.get("/v1/software_company/income_demographic_by_ids", operation_id="get_income_demographic", summary="Retrieves the annual income and population demographic details for the specified customer IDs. The operation calculates the annual income by multiplying the monthly income by 12 and the population demographic by multiplying the number of inhabitants by the income per capita. The results are returned for each provided customer ID.")
async def get_income_demographic(id1: int = Query(..., description="First customer ID"), id2: int = Query(..., description="Second customer ID")):
    cursor.execute("SELECT T2.INCOME_K, T2.INHABITANTS_K * T2.INCOME_K * 12 FROM Customers AS T1 INNER JOIN Demog AS T2 ON T1.GEOID = T2.GEOID WHERE T1.ID = ? OR T1.ID = ?", (id1, id2))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get education numbers based on mailing response criteria
@app.get("/v1/software_company/education_numbers_by_response", operation_id="get_education_numbers", summary="Retrieves the education numbers of customers who have a reference ID less than the provided limit and a specific mailing response. The response value determines the type of mailing response required for a customer to be included in the results.")
async def get_education_numbers(refid_limit: int = Query(..., description="Upper limit for reference ID"), response: str = Query(..., description="Response value")):
    cursor.execute("SELECT T1.EDUCATIONNUM FROM Customers AS T1 INNER JOIN Mailings1_2 AS T2 ON T1.ID = T2.REFID WHERE T2.REFID < ? AND T2.RESPONSE = ?", (refid_limit, response))
    result = cursor.fetchall()
    if not result:
        return {"education_numbers": []}
    return {"education_numbers": result}

api_calls = [
    "/v1/software_company/customer_count_by_marital_status?marital_status=Never-married",
    "/v1/software_company/customer_count_by_age_range?min_age=13&max_age=19",
    "/v1/software_company/distinct_occupations_by_education_number?education_number=11",
    "/v1/software_company/mailing_count_by_response?response=true",
    "/v1/software_company/customer_count_by_occupation_and_age?occupation=Machine-op-inspct&age=30",
    "/v1/software_company/customer_count_by_education_and_sex?education_number=11&sex=Female",
    "/v1/software_company/customer_count_by_sex_and_mailing_response?sex=Female&response=true",
    "/v1/software_company/distinct_occupations_by_age_and_mailing_response?age=40&response=true",
    "/v1/software_company/geoid_count_by_sex_and_inhabitants?sex=Male&inhabitants_k=30",
    "/v1/software_company/customer_count_by_highest_income",
    "/v1/software_company/customer_count_occupation_inhabitants?occupation=Machine-op-inspct&min_inhabitants=20&max_inhabitants=30",
    "/v1/software_company/customer_id_by_ids_inhabitants?id1=0&id2=1",
    "/v1/software_company/customer_count_inhabitants_response?min_inhabitants=30&response=true",
    "/v1/software_company/customer_count_sex_marital_response?sex=Male&marital_status=Divorced&response=true",
    "/v1/software_company/customer_count_response_income?response=true",
    "/v1/software_company/distinct_inhabitants",
    "/v1/software_company/customer_count_inhabitants_sex?inhabitants=25.746&sex=Male",
    "/v1/software_company/customer_count_age_response?min_age=13&max_age=19&response=true",
    "/v1/software_company/average_education_income",
    "/v1/software_company/average_age_response?response=true",
    "/v1/software_company/customer_count_by_gender?sex=Male",
    "/v1/software_company/customer_geoids_by_occupation?occupation=Handlers-cleaners",
    "/v1/software_company/customer_count_under_age?age=30",
    "/v1/software_company/geoids_by_income_range?min_income=2100&max_income=2500",
    "/v1/software_company/geoid_count_by_inhabitants_and_range?inhabitants=20&min_geoid=20&max_geoid=50",
    "/v1/software_company/inhabitants_by_geoid?geoid=239",
    "/v1/software_company/customer_education_occupation_by_income_age?income=2000&min_age=20&max_age=35",
    "/v1/software_company/customer_count_by_marital_status_age?marital_status=Divorced&age=50",
    "/v1/software_company/oldest_customer_geoid_income",
    "/v1/software_company/incomes_by_education_gender?education_num=4&sex=Male",
    "/v1/software_company/occupation_income_by_education_gender?min_education=4&max_education=6&sex=Male",
    "/v1/software_company/customer_count_by_age_marital_gender_income?min_age=40&max_age=60&marital_status=Widowed&sex=Male&min_income=2000&max_income=3000",
    "/v1/software_company/distinct_occupations_by_inhabitants?min_inhabitants=30&max_inhabitants=40",
    "/v1/software_company/income_by_education_gender_marital?education_num=5&sex=Female&marital_status=Widowed",
    "/v1/software_company/highest_income_marital_status_by_age?min_age=40&max_age=60",
    "/v1/software_company/inhabitants_by_occupation_gender_age?occupation=Farming-fishing&sex=Male&min_age=20&max_age=30",
    "/v1/software_company/customer_count_by_gender_age_inhabitants?sex=Female&min_age=50&max_age=60&min_inhabitants=19&max_inhabitants=24",
    "/v1/software_company/income_inhabitants_by_age_criteria",
    "/v1/software_company/income_percentage_by_marital_status?income_threshold=2500&marital_status=Never-married",
    "/v1/software_company/customer_ids_geoids_by_education_age?education_num=3&age=65",
    "/v1/software_company/average_income",
    "/v1/software_company/teenager_count_occupation_age_range?occupation=Machine-op-inspct&min_age=13&max_age=19",
    "/v1/software_company/geoid_count_occupation_inhabitants?occupation=Other-service&min_inhabitants=20",
    "/v1/software_company/marital_status_percentage_geoid?marital_status=never%20married&geoid=24",
    "/v1/software_company/age_range_percentage_income?min_age=80&max_age=89&min_income=3000",
    "/v1/software_company/mailing_count_response?response=true",
    "/v1/software_company/mailing_refids_response?response=true",
    "/v1/software_company/geoids_by_inhabitants?inhabitants_k=30",
    "/v1/software_company/count_geoids_by_income_and_range?income_k=2000&geoid_min=10&geoid_max=30",
    "/v1/software_company/distinct_marital_statuses?education_num=7&age=62",
    "/v1/software_company/count_customers_by_marital_status_and_response?marital_status=Widowed&response=true",
    "/v1/software_company/response_and_inhabitants_by_sex?sex=Female",
    "/v1/software_company/education_and_income_by_age_and_response?age_min=30&age_max=55&response=true",
    "/v1/software_company/count_customers_by_sex_age_and_income?sex=Male&age_min=30&age_max=50&income_min=2000&income_max=2300",
    "/v1/software_company/education_and_response_by_age_and_inhabitants?age_min=20&age_max=30",
    "/v1/software_company/income_by_sex_age_and_occupation?sex=Female&age_min=30&age_max=55&occupation=Machine-op-inspct",
    "/v1/software_company/marital_status_response?education_num=8&sex=Female",
    "/v1/software_company/customer_ages?inhabitants_k=30&sex=Female",
    "/v1/software_company/income_response?education_num=6&sex=Male&marital_status=Divorced",
    "/v1/software_company/occupation_response?sex=Female&min_inhabitants_k=20&max_inhabitants_k=25",
    "/v1/software_company/positive_response_percentage?response=true&occupation=Handlers-cleaners&sex=Male",
    "/v1/software_company/male_female_ratio?sex_male=Male&sex_female=Female&min_age=13&max_age=19&education_num=10",
    "/v1/software_company/annual_income?income_k=3300",
    "/v1/software_company/most_common_response",
    "/v1/software_company/annual_income_by_sex_occupation?sex=Female&occupation=Sales",
    "/v1/software_company/customer_details_by_attributes?inhabitants_k=33.658&sex=Female&marital_status=Widowed",
    "/v1/software_company/mailing_responses_by_geoid?geoid=134",
    "/v1/software_company/income_demographic_by_ids?id1=209556&id2=290135",
    "/v1/software_company/education_numbers_by_response?refid_limit=10&response=true"
]
