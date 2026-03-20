from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

try:
    conn = sqlite3.connect('db/food_inspection_2/food_inspection_2.sqlite')
    cursor = conn.cursor()
except Exception as e:
    print(f"Warning: could not connect to database: {e}")
    conn = None
    cursor = None

# Endpoint to get employee details based on title and supervisor's name
@app.get("/v1/food_inspection_2/employee_details_by_title_supervisor", operation_id="get_employee_details", summary="Retrieves the first and last names of employees who hold a specific job title and report to a designated supervisor. The operation requires the employee's title and the supervisor's first and last names as input parameters.")
async def get_employee_details(title: str = Query(..., description="Title of the employee"), supervisor_first_name: str = Query(..., description="First name of the supervisor"), supervisor_last_name: str = Query(..., description="Last name of the supervisor")):
    cursor.execute("SELECT first_name, last_name FROM employee WHERE title = ? AND supervisor = ( SELECT employee_id FROM employee WHERE first_name = ? AND last_name = ? )", (title, supervisor_first_name, supervisor_last_name))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get distinct employee names based on inspection date and title
@app.get("/v1/food_inspection_2/employee_names_by_inspection_date_title", operation_id="get_employee_names", summary="Retrieve the unique first and last names of employees who performed inspections during a specific month and year and hold a particular job title. The input parameters include the inspection date in 'YYYY-MM' format and the employee's title.")
async def get_employee_names(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM' format"), title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT DISTINCT T1.first_name, T1.last_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE strftime('%Y-%m', T2.inspection_date) = ? AND T1.title = ?", (inspection_date, title))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the count of inspections by employee name and year
@app.get("/v1/food_inspection_2/inspection_count_by_employee_year", operation_id="get_inspection_count", summary="Retrieves the total number of inspections performed by a specific employee in a given year. The operation requires the employee's first and last names, as well as the year of interest in 'YYYY' format.")
async def get_inspection_count(year: str = Query(..., description="Year in 'YYYY' format"), first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT COUNT(T2.inspection_id) FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE strftime('%Y', T2.inspection_date) = ? AND T1.first_name = ? AND T1.last_name = ?", (year, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct establishment names inspected by a specific employee
@app.get("/v1/food_inspection_2/establishment_names_by_employee", operation_id="get_establishment_names", summary="Retrieve a unique list of establishment names that have been inspected by a specific employee. The employee is identified by their first and last name. This operation returns the distinct business names of establishments that the employee has inspected.")
async def get_establishment_names(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT DISTINCT T3.dba_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id INNER JOIN establishment AS T3 ON T2.license_no = T3.license_no WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"establishments": []}
    return {"establishments": result}

# Endpoint to get the count of distinct establishments inspected in a specific year and facility type
@app.get("/v1/food_inspection_2/establishment_count_by_year_facility_type", operation_id="get_establishment_count", summary="Retrieves the number of unique establishments inspected in a given year and facility type. The year is specified in 'YYYY' format, and the facility type is selected from available options. This operation provides a snapshot of inspection activity based on the chosen criteria.")
async def get_establishment_count(year: str = Query(..., description="Year in 'YYYY' format"), facility_type: str = Query(..., description="Type of the facility")):
    cursor.execute("SELECT COUNT(DISTINCT T1.license_no) FROM inspection AS T1 INNER JOIN establishment AS T2 ON T1.license_no = T2.license_no WHERE strftime('%Y', T1.inspection_date) = ? AND T2.facility_type = ?", (year, facility_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct latitude and longitude of establishments inspected on a specific date
@app.get("/v1/food_inspection_2/establishment_coordinates_by_inspection_date", operation_id="get_establishment_coordinates", summary="Retrieve unique geographical coordinates of establishments that underwent inspection on a specified date. The input parameter is the inspection date in 'YYYY-MM-DD' format.")
async def get_establishment_coordinates(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.latitude, T2.longitude FROM inspection AS T1 INNER JOIN establishment AS T2 ON T1.license_no = T2.license_no WHERE T1.inspection_date = ?", (inspection_date,))
    result = cursor.fetchall()
    if not result:
        return {"coordinates": []}
    return {"coordinates": result}

# Endpoint to get the count of distinct establishments inspected in a specific year and ward
@app.get("/v1/food_inspection_2/establishment_count_by_year_ward", operation_id="get_establishment_count_by_ward", summary="Retrieve the number of unique establishments that underwent inspection in a specified year and ward. The year should be provided in 'YYYY' format, and the ward number is also required.")
async def get_establishment_count_by_ward(year: str = Query(..., description="Year in 'YYYY' format"), ward: int = Query(..., description="Ward number")):
    cursor.execute("SELECT COUNT(DISTINCT T1.license_no) FROM inspection AS T1 INNER JOIN establishment AS T2 ON T1.license_no = T2.license_no WHERE strftime('%Y', T1.inspection_date) = ? AND T2.ward = ?", (year, ward))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct employee names based on establishment name and title
@app.get("/v1/food_inspection_2/employee_names_by_establishment_title", operation_id="get_employee_names_by_establishment", summary="Retrieves unique first and last names of employees with a specific title who have inspected a given establishment. The operation filters employees by their title and the establishment they inspected, based on the provided establishment name.")
async def get_employee_names_by_establishment(dba_name: str = Query(..., description="Name of the establishment"), title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT DISTINCT T1.first_name, T1.last_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id INNER JOIN establishment AS T3 ON T2.license_no = T3.license_no WHERE T3.dba_name = ? AND T1.title = ?", (dba_name, title))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get distinct establishment names based on inspection results and year
@app.get("/v1/food_inspection_2/establishment_names_by_inspection_results_year", operation_id="get_establishment_names_by_results", summary="Retrieves a unique list of establishment names that have undergone inspections with specific results in a given year. The operation filters establishments based on the provided inspection results and year, ensuring that only those meeting the criteria are included in the returned list.")
async def get_establishment_names_by_results(results: str = Query(..., description="Inspection results"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T2.dba_name FROM inspection AS T1 INNER JOIN establishment AS T2 ON T1.license_no = T2.license_no WHERE T1.results = ? AND strftime('%Y', T1.inspection_date) = ?", (results, year))
    result = cursor.fetchall()
    if not result:
        return {"establishments": []}
    return {"establishments": result}

# Endpoint to get employee names based on inspection date, establishment name, and title
@app.get("/v1/food_inspection_2/employee_names_by_inspection_date_establishment_title", operation_id="get_employee_names_by_inspection", summary="Retrieve the first and last names of employees who conducted inspections at a specified establishment on a given date and hold a particular job title. The operation requires the inspection date, establishment name, and employee title as input parameters.")
async def get_employee_names_by_inspection(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), dba_name: str = Query(..., description="Name of the establishment"), title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id INNER JOIN establishment AS T3 ON T2.license_no = T3.license_no WHERE T2.inspection_date = ? AND T3.dba_name = ? AND T1.title = ?", (inspection_date, dba_name, title))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the count of inspections conducted by an employee with a specific name and result
@app.get("/v1/food_inspection_2/count_inspections_by_employee", operation_id="get_count_inspections_by_employee", summary="Retrieves the total number of inspections performed by an employee, filtered by the inspection result and the employee's first and last names.")
async def get_count_inspections_by_employee(results: str = Query(..., description="Result of the inspection"), first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT COUNT(T2.inspection_id) FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.results = ? AND T1.first_name = ? AND T1.last_name = ?", (results, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the follow-up inspection ID for a specific establishment and inspection result on a given date
@app.get("/v1/food_inspection_2/followup_inspection_id", operation_id="get_followup_inspection_id", summary="Retrieve the follow-up inspection ID for a specific establishment based on the provided inspection result and date. The endpoint requires the establishment's DBA name, the inspection result, and the inspection date in 'YYYY-MM-DD' format to accurately identify the follow-up inspection.")
async def get_followup_inspection_id(dba_name: str = Query(..., description="DBA name of the establishment"), results: str = Query(..., description="Result of the inspection"), inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.followup_to FROM inspection AS T1 INNER JOIN establishment AS T2 ON T1.license_no = T2.license_no WHERE T2.dba_name = ? AND T1.results = ? AND T1.inspection_date = ?", (dba_name, results, inspection_date))
    result = cursor.fetchone()
    if not result:
        return {"followup_to": []}
    return {"followup_to": result[0]}

# Endpoint to get the count of distinct establishments inspected in a specific year with a given risk level
@app.get("/v1/food_inspection_2/count_distinct_establishments", operation_id="get_count_distinct_establishments", summary="Retrieves the total number of unique establishments that underwent inspection in a specified year, filtered by a given risk level.")
async def get_count_distinct_establishments(year: str = Query(..., description="Year of the inspection in 'YYYY' format"), risk_level: int = Query(..., description="Risk level of the establishment")):
    cursor.execute("SELECT COUNT(DISTINCT T2.license_no) FROM inspection AS T1 INNER JOIN establishment AS T2 ON T1.license_no = T2.license_no WHERE strftime('%Y', T1.inspection_date) = ? AND T2.risk_level = ?", (year, risk_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of inspections in a specific year to employees with a salary above a certain amount
@app.get("/v1/food_inspection_2/inspection_to_salary_ratio", operation_id="get_inspection_to_salary_ratio", summary="Retrieves the ratio of inspections conducted in a specific year to the number of employees earning above a certain salary threshold. This operation calculates the ratio by summing the inspections performed in the given year and dividing it by the count of employees with salaries exceeding the provided threshold. The result is a numerical value representing the inspection-to-salary ratio.")
async def get_inspection_to_salary_ratio(year: str = Query(..., description="Year of the inspection in 'YYYY' format"), salary: int = Query(..., description="Salary threshold")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.inspection_date LIKE ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.salary > ? THEN 1 ELSE 0 END) FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id", (year + '%', salary))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the point level for a specific inspection point description
@app.get("/v1/food_inspection_2/point_level_by_description", operation_id="get_point_level_by_description", summary="Get the point level for a specific inspection point description")
async def get_point_level_by_description(description: str = Query(..., description="Description of the inspection point")):
    cursor.execute("SELECT point_level FROM inspection_point WHERE Description = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"point_level": []}
    return {"point_level": result[0]}

# Endpoint to get the employee details for a specific inspection ID
@app.get("/v1/food_inspection_2/employee_details_by_inspection_id", operation_id="get_employee_details_by_inspection_id", summary="Retrieves the first and last names of the employee associated with the provided inspection ID. This operation fetches the employee details from the database by matching the inspection ID with the corresponding employee ID.")
async def get_employee_details_by_inspection_id(inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM inspection AS T1 INNER JOIN employee AS T2 ON T1.employee_id = T2.employee_id WHERE T1.inspection_id = ?", (inspection_id,))
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of inspections for a specific establishment type and DBA name
@app.get("/v1/food_inspection_2/count_inspections_by_establishment", operation_id="get_count_inspections_by_establishment", summary="Retrieves the total number of inspections conducted for a specific type of establishment with a given DBA name. The operation filters the establishments based on the provided facility type and DBA name, and then counts the associated inspections.")
async def get_count_inspections_by_establishment(facility_type: str = Query(..., description="Type of the facility"), dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT COUNT(T2.inspection_id) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.facility_type = ? AND T1.dba_name = ?", (facility_type, dba_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the earliest inspection date for a specific establishment AKA name
@app.get("/v1/food_inspection_2/earliest_inspection_date", operation_id="get_earliest_inspection_date", summary="Retrieves the date of the earliest inspection conducted for a specific establishment. The establishment is identified by its AKA name, which is a unique identifier used to distinguish it from other establishments.")
async def get_earliest_inspection_date(aka_name: str = Query(..., description="AKA name of the establishment")):
    cursor.execute("SELECT MIN(T2.inspection_date) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.aka_name = ?", (aka_name,))
    result = cursor.fetchone()
    if not result:
        return {"earliest_inspection_date": []}
    return {"earliest_inspection_date": result[0]}

# Endpoint to get the count of establishments inspected on a specific date and facility type
@app.get("/v1/food_inspection_2/count_establishments_by_inspection_date", operation_id="get_count_establishments_by_inspection_date", summary="Retrieves the total number of establishments inspected on a specific date and facility type. The operation filters establishments by the provided inspection date and facility type, then returns the count of establishments that meet these criteria.")
async def get_count_establishments_by_inspection_date(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), facility_type: str = Query(..., description="Type of the facility")):
    cursor.execute("SELECT COUNT(T2.license_no) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T2.inspection_date = ? AND T1.facility_type = ?", (inspection_date, facility_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of violation points for a specific inspection ID and category
@app.get("/v1/food_inspection_2/count_violation_points", operation_id="get_count_violation_points", summary="Retrieves the total count of violation points associated with a specific inspection and category. The inspection is identified by its unique ID, and the category refers to the type of violation point. This operation provides a quantitative measure of the violations found during the inspection, categorized by the specified category.")
async def get_count_violation_points(inspection_id: str = Query(..., description="Inspection ID"), category: str = Query(..., description="Category of the violation point")):
    cursor.execute("SELECT COUNT(T2.point_id) FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T2.inspection_id = ? AND T1.category = ?", (inspection_id, category))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of violations for a specific establishment on a specific date
@app.get("/v1/food_inspection_2/violation_count_by_date_and_establishment", operation_id="get_violation_count", summary="Retrieves the total number of violations for a specific establishment on a given date. The operation requires the inspection date in 'YYYY-MM-DD' format and the DBA name of the establishment as input parameters. The result is a count of all violations associated with the provided establishment and date.")
async def get_violation_count(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT COUNT(T3.point_id) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE T2.inspection_date = ? AND T1.dba_name = ?", (inspection_date, dba_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the employee with the most inspections
@app.get("/v1/food_inspection_2/employee_with_most_inspections", operation_id="get_employee_with_most_inspections", summary="Retrieves the first name and last name of the employee who has conducted the highest number of inspections. This operation identifies the employee with the most inspections by counting the number of inspections associated with each employee and then selecting the employee with the highest count.")
async def get_employee_with_most_inspections():
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T2.employee_id, T2.first_name, T2.last_name, COUNT(T1.inspection_id) FROM inspection AS T1 INNER JOIN employee AS T2 ON T1.employee_id = T2.employee_id GROUP BY T2.employee_id, T2.first_name, T2.last_name ORDER BY COUNT(T1.inspection_id) DESC LIMIT 1 ) AS T")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of inspections for a specific employee with specific results
@app.get("/v1/food_inspection_2/inspection_count_by_employee_and_results", operation_id="get_inspection_count_by_employee_and_results", summary="Retrieves the total number of inspections conducted by a specific employee that yielded particular results. The operation requires the first and last name of the employee, as well as the desired inspection results as input parameters.")
async def get_inspection_count_by_employee_and_results(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), results: str = Query(..., description="Results of the inspection")):
    cursor.execute("SELECT COUNT(T1.inspection_id) FROM inspection AS T1 INNER JOIN employee AS T2 ON T1.employee_id = T2.employee_id WHERE T2.first_name = ? AND T2.last_name = ? AND T1.results = ?", (first_name, last_name, results))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of inspections for a specific employee address, title, and inspection month
@app.get("/v1/food_inspection_2/inspection_count_by_employee_address_title_and_month", operation_id="get_inspection_count_by_employee_address_title_and_month", summary="Retrieve the total number of inspections conducted by an employee with a specific address and title during a given month. The input parameters include the employee's address, job title, and the desired month in 'YYYY-MM' format.")
async def get_inspection_count_by_employee_address_title_and_month(address: str = Query(..., description="Address of the employee"), title: str = Query(..., description="Title of the employee"), inspection_month: str = Query(..., description="Inspection month in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(T1.inspection_id) FROM inspection AS T1 INNER JOIN employee AS T2 ON T1.employee_id = T2.employee_id WHERE T2.address = ? AND T2.title = ? AND strftime('%Y-%m', T1.inspection_date) = ?", (address, title, inspection_month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone number of an employee for a specific inspection
@app.get("/v1/food_inspection_2/employee_phone_by_inspection_id_and_title", operation_id="get_employee_phone", summary="Retrieves the phone number of an employee associated with a specific inspection, based on the provided inspection ID and employee title.")
async def get_employee_phone(inspection_id: int = Query(..., description="Inspection ID"), title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT T2.phone FROM inspection AS T1 INNER JOIN employee AS T2 ON T1.employee_id = T2.employee_id WHERE T1.inspection_id = ? AND T2.title = ?", (inspection_id, title))
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the salary of the employee with the most inspections
@app.get("/v1/food_inspection_2/salary_of_employee_with_most_inspections", operation_id="get_salary_of_employee_with_most_inspections", summary="Retrieves the salary of the employee who has conducted the highest number of inspections. This operation identifies the employee with the most inspections and returns their salary, providing insight into the compensation of the most active inspector.")
async def get_salary_of_employee_with_most_inspections():
    cursor.execute("SELECT T1.salary FROM employee AS T1 INNER JOIN ( SELECT T.employee_id, COUNT(T.inspection_id) FROM inspection AS T GROUP BY T.employee_id ORDER BY COUNT(T.inspection_id) DESC LIMIT 1 ) AS T2 ON T1.employee_id = T2.employee_id")
    result = cursor.fetchone()
    if not result:
        return {"salary": []}
    return {"salary": result[0]}

# Endpoint to get the average number of inspections per establishment for a specific risk level and facility type
@app.get("/v1/food_inspection_2/average_inspections_per_establishment", operation_id="get_average_inspections_per_establishment", summary="Retrieves the average number of inspections per establishment for a given risk level and facility type. This operation calculates the ratio of total inspections to the number of unique establishments, considering only those establishments that match the specified risk level and facility type.")
async def get_average_inspections_per_establishment(risk_level: int = Query(..., description="Risk level of the establishment"), facility_type: str = Query(..., description="Facility type of the establishment")):
    cursor.execute("SELECT CAST(COUNT(T2.inspection_id) AS REAL) / COUNT(DISTINCT T1.license_no) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.risk_level = ? AND T1.facility_type = ?", (risk_level, facility_type))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of passed inspections for a specific establishment and facility type
@app.get("/v1/food_inspection_2/percentage_passed_inspections", operation_id="get_percentage_passed_inspections", summary="Retrieves the percentage of successful inspections for a specific establishment, based on its DBA name and facility type. The calculation considers all inspections conducted for the establishment and returns the proportion of inspections that resulted in a 'Pass' outcome.")
async def get_percentage_passed_inspections(dba_name: str = Query(..., description="DBA name of the establishment"), facility_type: str = Query(..., description="Facility type of the establishment")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.results = 'Pass' THEN T2.inspection_id ELSE NULL END) AS REAL) * 100 / COUNT(T2.inspection_id) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.dba_name = ? AND T1.facility_type = ?", (dba_name, facility_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of employees in a specific zip code
@app.get("/v1/food_inspection_2/employee_count_by_zip", operation_id="get_employee_count_by_zip", summary="Retrieves the total number of employees associated with a given zip code. The input parameter specifies the zip code for which the employee count is calculated.")
async def get_employee_count_by_zip(zip: str = Query(..., description="Zip code of the employee")):
    cursor.execute("SELECT COUNT(employee_id) FROM employee WHERE zip = ?", (zip,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct dba_names based on address
@app.get("/v1/food_inspection_2/distinct_dba_names_by_address", operation_id="get_distinct_dba_names", summary="Get distinct dba_names from establishments based on a specific address")
async def get_distinct_dba_names(address: str = Query(..., description="Address of the establishment")):
    cursor.execute("SELECT DISTINCT dba_name FROM establishment WHERE address = ?", (address,))
    result = cursor.fetchall()
    if not result:
        return {"dba_names": []}
    return {"dba_names": [row[0] for row in result]}

# Endpoint to get the employee with the lowest salary
@app.get("/v1/food_inspection_2/employee_with_lowest_salary", operation_id="get_employee_with_lowest_salary", summary="Retrieves the full name of the employee with the lowest salary from the employee database.")
async def get_employee_with_lowest_salary():
    cursor.execute("SELECT first_name, last_name FROM employee ORDER BY salary ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": {}}
    return {"employee": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of establishments based on risk level and dba_name
@app.get("/v1/food_inspection_2/count_establishments_by_risk_level_and_dba_name", operation_id="get_count_establishments", summary="Retrieves the number of establishments that match a given risk level and DBA name. The risk level indicates the potential health risk posed by the establishment, while the DBA name refers to the business name under which the establishment operates.")
async def get_count_establishments(risk_level: int = Query(..., description="Risk level of the establishment"), dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT COUNT(license_no) FROM establishment WHERE risk_level = ? AND dba_name = ?", (risk_level, dba_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of violations based on point_id and fine
@app.get("/v1/food_inspection_2/count_violations_by_point_id_and_fine", operation_id="get_count_violations", summary="Retrieves the total number of violations associated with a specific location (identified by point_id) and a particular fine amount. This operation is useful for understanding the frequency of specific violations at a given location.")
async def get_count_violations(point_id: int = Query(..., description="Point ID of the violation"), fine: int = Query(..., description="Fine amount of the violation")):
    cursor.execute("SELECT COUNT(inspection_id) FROM violation WHERE point_id = ? AND fine = ?", (point_id, fine))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of employees under a specific supervisor
@app.get("/v1/food_inspection_2/count_employees_under_supervisor", operation_id="get_count_employees_under_supervisor", summary="Retrieves the total number of employees who are directly supervised by a specific individual. The supervisor is identified by their first and last names. This operation does not return any employee details, only the count of employees under the specified supervisor.")
async def get_count_employees_under_supervisor(first_name: str = Query(..., description="First name of the supervisor"), last_name: str = Query(..., description="Last name of the supervisor")):
    cursor.execute("SELECT COUNT(T1.employee_id) FROM employee AS T1 WHERE T1.supervisor = ( SELECT employee_id FROM employee WHERE first_name = ? AND last_name = ? )", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct dba_names based on inspection date and results
@app.get("/v1/food_inspection_2/distinct_dba_names_by_inspection_date_and_results", operation_id="get_distinct_dba_names_by_inspection", summary="Retrieve a unique list of business names (DBA) that underwent inspections in a specific month and yielded certain results. The month is provided in 'YYYY-MM' format, and the results indicate the outcome of the inspections.")
async def get_distinct_dba_names_by_inspection(month: str = Query(..., description="Month in 'YYYY-MM' format"), results: str = Query(..., description="Results of the inspection")):
    cursor.execute("SELECT DISTINCT T2.dba_name FROM inspection AS T1 INNER JOIN establishment AS T2 ON T1.license_no = T2.license_no WHERE strftime('%Y-%m', T1.inspection_date) = ? AND T1.results = ?", (month, results))
    result = cursor.fetchall()
    if not result:
        return {"dba_names": []}
    return {"dba_names": [row[0] for row in result]}

# Endpoint to get the count of distinct license numbers based on employee details and inspection type and results
@app.get("/v1/food_inspection_2/count_distinct_license_numbers_by_employee_and_inspection", operation_id="get_count_distinct_license_numbers", summary="Retrieve the count of unique license numbers associated with a specific employee, based on their first and last name, employee ID, the type of inspection, and the inspection results.")
async def get_count_distinct_license_numbers(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), employee_id: int = Query(..., description="Employee ID"), inspection_type: str = Query(..., description="Type of inspection"), results: str = Query(..., description="Results of the inspection")):
    cursor.execute("SELECT COUNT(DISTINCT T2.license_no) FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T1.first_name = ? AND T1.last_name = ? AND T1.employee_id = ? AND T2.inspection_type = ? AND T2.results = ?", (first_name, last_name, employee_id, inspection_type, results))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct license numbers based on inspection date range, ward, and inspection results
@app.get("/v1/food_inspection_2/count_distinct_license_numbers_by_date_range_ward_and_results", operation_id="get_count_distinct_license_numbers_by_date_range", summary="Retrieve the count of unique establishments that have undergone inspections within a specified date range, ward, and inspection results. The count is based on distinct license numbers, ensuring each establishment is counted only once. The results parameter filters establishments based on their inspection results, and the minimum count parameter ensures that only establishments with a certain number of inspections meeting the specified results are included in the count.")
async def get_count_distinct_license_numbers_by_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), ward: int = Query(..., description="Ward number"), results: str = Query(..., description="Results of the inspection"), min_count: int = Query(..., description="Minimum count of results")):
    cursor.execute("SELECT COUNT(DISTINCT T1.license_no) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T2.inspection_date BETWEEN ? AND ? AND T1.ward = ? AND T1.license_no IN ( SELECT license_no FROM ( SELECT license_no FROM inspection WHERE results = ? GROUP BY license_no HAVING COUNT(results) >= ? ) )", (start_date, end_date, ward, results, min_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the salary of the employee with the highest number of inspections
@app.get("/v1/food_inspection_2/employee_salary_highest_inspections", operation_id="get_employee_salary_highest_inspections", summary="Retrieves the salary of the employee who has performed the most inspections. This operation identifies the employee with the highest inspection count and returns their salary.")
async def get_employee_salary_highest_inspections():
    cursor.execute("SELECT T1.salary FROM employee AS T1 INNER JOIN ( SELECT employee_id, COUNT(inspection_id) FROM inspection GROUP BY employee_id ORDER BY COUNT(inspection_id) DESC LIMIT 1 ) AS T2 ON T1.employee_id = T2.employee_id")
    result = cursor.fetchone()
    if not result:
        return {"salary": []}
    return {"salary": result[0]}

# Endpoint to get the establishment with the highest total fines in a given year
@app.get("/v1/food_inspection_2/establishment_highest_fines_year", operation_id="get_establishment_highest_fines_year", summary="Retrieves the establishment with the highest total fines in a specified year. The year is provided in 'YYYY' format. The operation calculates the total fines for each establishment based on inspections and violations data, then returns the establishment with the highest sum of fines for the given year.")
async def get_establishment_highest_fines_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T.dba_name FROM ( SELECT T1.dba_name, SUM(T3.fine) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE strftime('%Y', T2.inspection_date) = ? GROUP BY T1.dba_name ORDER BY SUM(T3.fine) DESC LIMIT 1 ) AS T", (year,))
    result = cursor.fetchone()
    if not result:
        return {"dba_name": []}
    return {"dba_name": result[0]}

# Endpoint to get the latitude and longitude of the establishment with the most failed inspections
@app.get("/v1/food_inspection_2/establishment_most_failed_inspections", operation_id="get_establishment_most_failed_inspections", summary="Retrieve the geographical coordinates of the establishment that has the highest number of failed inspections. The operation filters inspections based on the provided result and identifies the establishment with the most occurrences of that result. The latitude and longitude of the establishment are then returned.")
async def get_establishment_most_failed_inspections(results: str = Query(..., description="Inspection result (e.g., 'Fail')")):
    cursor.execute("SELECT T1.latitude, T1.longitude FROM establishment AS T1 INNER JOIN ( SELECT license_no FROM inspection WHERE results = ? GROUP BY license_no ORDER BY COUNT(results) DESC LIMIT 1 ) AS T2 ON T1.license_no = T2.license_no", (results,))
    result = cursor.fetchone()
    if not result:
        return {"latitude": [], "longitude": []}
    return {"latitude": result[0], "longitude": result[1]}

# Endpoint to get inspector comments for a specific establishment on a specific date
@app.get("/v1/food_inspection_2/inspector_comments_establishment_date", operation_id="get_inspector_comments_establishment_date", summary="Retrieves all comments made by inspectors for a specific establishment on a given date. The operation requires the inspection date in 'YYYY-MM-DD' format and the DBA name of the establishment to accurately filter and return the relevant inspector comments.")
async def get_inspector_comments_establishment_date(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT T3.inspector_comment FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE T2.inspection_date = ? AND T1.dba_name = ?", (inspection_date, dba_name))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": [row[0] for row in result]}

# Endpoint to get the total fines for a specific establishment in a specific month and year
@app.get("/v1/food_inspection_2/total_fines_establishment_month_year", operation_id="get_total_fines_establishment_month_year", summary="Retrieves the total amount of fines incurred by a specific food establishment during a particular month and year. The operation requires the DBA name of the establishment and the month-year in 'YYYY-MM' format to accurately calculate the sum of fines.")
async def get_total_fines_establishment_month_year(month_year: str = Query(..., description="Month and year in 'YYYY-MM' format"), dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT SUM(T3.fine) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE strftime('%Y-%m', T2.inspection_date) = ? AND T1.dba_name = ?", (month_year, dba_name))
    result = cursor.fetchone()
    if not result:
        return {"total_fines": []}
    return {"total_fines": result[0]}

# Endpoint to get distinct employee names for a specific establishment
@app.get("/v1/food_inspection_2/employee_names_establishment", operation_id="get_employee_names_establishment", summary="Retrieves unique employee names associated with a specific establishment. The operation filters the employee records based on the provided DBA name of the establishment.")
async def get_employee_names_establishment(dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT DISTINCT T3.first_name, T3.last_name FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN employee AS T3 ON T2.employee_id = T3.employee_id WHERE T1.dba_name = ?", (dba_name,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the employee with the highest total fines
@app.get("/v1/food_inspection_2/employee_highest_total_fines", operation_id="get_employee_highest_total_fines", summary="Retrieves the employee with the highest cumulative fines, calculated based on the sum of fines associated with their inspections.")
async def get_employee_highest_total_fines():
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T1.first_name, T1.last_name, SUM(T3.fine) FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id GROUP BY T1.first_name, T1.last_name ORDER BY SUM(T3.fine) DESC LIMIT 1 ) t")
    result = cursor.fetchone()
    if not result:
        return {"first_name": [], "last_name": []}
    return {"first_name": result[0], "last_name": result[1]}

# Endpoint to get the top 5 employees by salary with a specific title and their average number of inspections
@app.get("/v1/food_inspection_2/top_employees_by_salary_title", operation_id="get_top_employees_by_salary_title", summary="Retrieves the top 5 employees with the highest salaries for a given job title, along with their average number of inspections. The results are sorted in descending order based on salary.")
async def get_top_employees_by_salary_title(title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT CAST(COUNT(DISTINCT T2.inspection_id) AS REAL) / 5, T1.first_name, T1.last_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T1.title = ? ORDER BY T1.salary DESC LIMIT 5", (title,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"average_inspections": row[0], "first_name": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get the establishment with the highest number of inspections and their pass/fail percentages
@app.get("/v1/food_inspection_2/establishment_highest_inspections_pass_fail", operation_id="get_establishment_highest_inspections_pass_fail", summary="Retrieves the establishment with the most inspections and displays the percentage of those inspections that passed and failed.")
async def get_establishment_highest_inspections_pass_fail():
    cursor.execute("SELECT T2.dba_name , CAST(SUM(CASE WHEN T1.results = 'Pass' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.inspection_id) AS percentagePassed , CAST(SUM(CASE WHEN T1.results = 'Fail' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.inspection_id) FROM inspection AS T1 INNER JOIN establishment AS T2 ON T1.license_no = T2.license_no GROUP BY T2.dba_name ORDER BY COUNT(T1.license_no) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"dba_name": [], "percentagePassed": [], "percentageFailed": []}
    return {"dba_name": result[0], "percentagePassed": result[1], "percentageFailed": result[2]}

# Endpoint to get the last names of employees based on address, city, and state
@app.get("/v1/food_inspection_2/employee_last_names_address", operation_id="get_employee_last_names_address", summary="Retrieves the last names of employees who reside at a specific address, within a given city and state. The endpoint requires the address, city, and state as input parameters to filter the employee records and return the corresponding last names.")
async def get_employee_last_names_address(address: str = Query(..., description="Address of the employee"), city: str = Query(..., description="City of the employee"), state: str = Query(..., description="State of the employee")):
    cursor.execute("SELECT last_name FROM employee WHERE address = ? AND city = ? AND state = ?", (address, city, state))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get establishment and employee details based on inspection date and ID
@app.get("/v1/food_inspection_2/establishment_employee_details", operation_id="get_establishment_employee_details", summary="Retrieves the establishment name and employee details associated with a specific inspection. The operation requires the inspection date and ID to accurately identify the inspection and return the corresponding establishment and employee information.")
async def get_establishment_employee_details(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT T1.dba_name, T3.first_name, T3.last_name FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN employee AS T3 ON T2.employee_id = T3.employee_id WHERE T2.inspection_date = ? AND T2.inspection_id = ?", (inspection_date, inspection_id))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get distinct addresses of establishments based on inspection month, results, and facility type
@app.get("/v1/food_inspection_2/distinct_addresses", operation_id="get_distinct_addresses", summary="Retrieve a unique list of addresses for establishments that underwent inspections in a specific month, with a particular inspection result and facility type. The response includes addresses that meet the specified criteria.")
async def get_distinct_addresses(inspection_month: str = Query(..., description="Inspection month in 'YYYY-MM' format"), results: str = Query(..., description="Inspection results"), facility_type: str = Query(..., description="Facility type")):
    cursor.execute("SELECT DISTINCT T1.address FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE strftime('%Y-%m', T2.inspection_date) = ? AND T2.results = ? AND T1.facility_type = ?", (inspection_month, results, facility_type))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get distinct employee names based on inspection date and type
@app.get("/v1/food_inspection_2/distinct_employee_names", operation_id="get_distinct_employee_names", summary="Retrieve a unique list of employee names who conducted inspections on a specific date and of a particular type. The response includes the first and last names of the employees.")
async def get_distinct_employee_names(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), inspection_type: str = Query(..., description="Inspection type")):
    cursor.execute("SELECT DISTINCT T1.first_name, T1.last_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.inspection_date = ? AND T2.inspection_type = ?", (inspection_date, inspection_type))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get distinct inspection IDs based on establishment DBA name
@app.get("/v1/food_inspection_2/distinct_inspection_ids", operation_id="get_distinct_inspection_ids", summary="Retrieves a unique set of inspection IDs associated with a specific establishment, identified by its DBA name.")
async def get_distinct_inspection_ids(dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT DISTINCT T2.inspection_id FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.dba_name = ?", (dba_name,))
    result = cursor.fetchall()
    if not result:
        return {"inspection_ids": []}
    return {"inspection_ids": result}

# Endpoint to get the count of distinct establishment licenses based on risk level, inspection results, and facility type
@app.get("/v1/food_inspection_2/count_distinct_licenses", operation_id="get_count_distinct_licenses", summary="Retrieves the total number of unique establishment licenses that meet the specified risk level, inspection results, and facility type criteria.")
async def get_count_distinct_licenses(risk_level: int = Query(..., description="Risk level"), results: str = Query(..., description="Inspection results"), facility_type: str = Query(..., description="Facility type")):
    cursor.execute("SELECT COUNT(DISTINCT T1.license_no) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.risk_level = ? AND T2.results = ? AND T1.facility_type = ?", (risk_level, results, facility_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct employee names based on inspection point category
@app.get("/v1/food_inspection_2/distinct_employee_names_by_category", operation_id="get_distinct_employee_names_by_category", summary="Retrieves a list of unique employee names associated with a specific inspection point category. This operation filters employees based on their involvement in inspections related to the provided category, ensuring that only distinct names are returned.")
async def get_distinct_employee_names_by_category(category: str = Query(..., description="Inspection point category")):
    cursor.execute("SELECT DISTINCT T1.first_name, T1.last_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id INNER JOIN inspection_point AS T4 ON T3.point_id = T4.point_id WHERE T4.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get employee titles based on inspection ID
@app.get("/v1/food_inspection_2/employee_titles", operation_id="get_employee_titles", summary="Retrieves the job titles of employees who conducted a specific food inspection. The operation requires the unique identifier of the inspection as input and returns a list of corresponding employee titles.")
async def get_employee_titles(inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT T1.title FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.inspection_id = ?", (inspection_id,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get the count of distinct establishment licenses based on risk level, inspection type, facility type, and results
@app.get("/v1/food_inspection_2/count_distinct_licenses_by_criteria", operation_id="get_count_distinct_licenses_by_criteria", summary="Retrieve the number of unique establishment licenses that meet the specified criteria for risk level, inspection type, facility type, and inspection results.")
async def get_count_distinct_licenses_by_criteria(risk_level: str = Query(..., description="Risk level"), inspection_type: str = Query(..., description="Inspection type"), facility_type: str = Query(..., description="Facility type"), results: str = Query(..., description="Inspection results")):
    cursor.execute("SELECT COUNT(DISTINCT T1.license_no) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.risk_level = ? AND T2.inspection_type = ? AND T1.facility_type = ? AND T2.results = ?", (risk_level, inspection_type, facility_type, results))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct fines and establishment details based on inspection ID
@app.get("/v1/food_inspection_2/distinct_fines_establishment_details", operation_id="get_distinct_fines_establishment_details", summary="Retrieve unique fine amounts along with the corresponding establishment's state, city, and address based on a specific inspection ID. This operation fetches data from the establishment, inspection, and violation tables, ensuring that only distinct fine records are returned.")
async def get_distinct_fines_establishment_details(inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT DISTINCT T3.fine, T1.state, T1.city, T1.address FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE T2.inspection_id = ?", (inspection_id,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get inspection IDs based on fine, point level, and inspector comment
@app.get("/v1/food_inspection_2/inspection_ids_by_criteria", operation_id="get_inspection_ids_by_criteria", summary="Retrieves a list of inspection IDs that match the specified fine amount, point level, and inspector comment. The endpoint filters the inspections based on the provided criteria and returns the corresponding IDs.")
async def get_inspection_ids_by_criteria(fine: int = Query(..., description="Fine amount"), point_level: str = Query(..., description="Point level"), inspector_comment: str = Query(..., description="Inspector comment")):
    cursor.execute("SELECT T2.inspection_id FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T2.fine = ? AND T1.point_level = ? AND T2.inspector_comment = ?", (fine, point_level, inspector_comment))
    result = cursor.fetchall()
    if not result:
        return {"inspection_ids": []}
    return {"inspection_ids": result}

# Endpoint to get inspection point descriptions and inspector comments based on inspection ID
@app.get("/v1/food_inspection_2/inspection_point_description_comment", operation_id="get_inspection_point_description_comment", summary="Retrieves the descriptions of inspection points and corresponding inspector comments for a specific inspection, identified by its unique ID.")
async def get_inspection_point_description_comment(inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT T1.Description, T2.inspector_comment FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T2.inspection_id = ?", (inspection_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get inspector comments and codes based on inspection ID and point ID
@app.get("/v1/food_inspection_2/inspector_comment_code", operation_id="get_inspector_comment_code", summary="Retrieves the inspector comments and corresponding codes associated with a specific inspection and point. The operation requires the unique identifiers for the inspection and point as input parameters to accurately locate and return the requested data.")
async def get_inspector_comment_code(inspection_id: int = Query(..., description="Inspection ID"), point_id: int = Query(..., description="Point ID")):
    cursor.execute("SELECT T2.inspector_comment, T1.code FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T2.inspection_id = ? AND T2.point_id = ?", (inspection_id, point_id))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of establishments with a specific risk level that failed inspections
@app.get("/v1/food_inspection_2/percentage_risk_level_fail", operation_id="get_percentage_risk_level_fail", summary="Retrieves the percentage of establishments with a specified risk level that have failed inspections. The risk level and inspection results are used to filter the establishments and calculate the percentage.")
async def get_percentage_risk_level_fail(risk_level: int = Query(..., description="Risk level"), results: str = Query(..., description="Inspection results")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.risk_level = ? THEN T1.license_no END) AS REAL) * 100 / COUNT(T1.risk_level) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T2.results = ?", (risk_level, results))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in counts of specific fines for employees within a salary range
@app.get("/v1/food_inspection_2/fine_difference_salary_range", operation_id="get_fine_difference_salary_range", summary="Retrieve the difference in the number of occurrences of two specific fines for employees whose salaries fall within a specified range. This operation calculates the total count of the first fine and subtracts the total count of the second fine for employees with salaries between the provided minimum and maximum values.")
async def get_fine_difference_salary_range(fine1: int = Query(..., description="First fine amount"), fine2: int = Query(..., description="Second fine amount"), min_salary: int = Query(..., description="Minimum salary"), max_salary: int = Query(..., description="Maximum salary")):
    cursor.execute("SELECT SUM(CASE WHEN T3.fine = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T3.fine = ? THEN 1 ELSE 0 END) FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE T1.salary BETWEEN ? AND ?", (fine1, fine2, min_salary, max_salary))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of inspections for a specific month and year
@app.get("/v1/food_inspection_2/inspection_count_month_year", operation_id="get_inspection_count_month_year", summary="Retrieves the total number of food inspections conducted in a specified month and year. The input parameter determines the time period for which the count is calculated.")
async def get_inspection_count_month_year(month_year: str = Query(..., description="Month and year in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(inspection_id) FROM inspection WHERE strftime('%Y-%m', inspection_date) = ?", (month_year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of inspections for a specific year and result
@app.get("/v1/food_inspection_2/inspection_count_year_result", operation_id="get_inspection_count_year_result", summary="Retrieves the total number of inspections conducted in a specified year that resulted in a particular outcome. The year should be provided in 'YYYY' format, and the inspection result can be any valid outcome.")
async def get_inspection_count_year_result(year: str = Query(..., description="Year in 'YYYY' format"), results: str = Query(..., description="Inspection results")):
    cursor.execute("SELECT COUNT(inspection_id) FROM inspection WHERE strftime('%Y', inspection_date) = ? AND results = ?", (year, results))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of inspections with a specific fine amount
@app.get("/v1/food_inspection_2/percentage_inspections_fine", operation_id="get_percentage_inspections_fine", summary="Retrieves the percentage of inspections that resulted in a specific fine amount. The fine amount is provided as an input parameter. The calculation is based on the total number of inspections and the count of inspections with the specified fine.")
async def get_percentage_inspections_fine(fine: int = Query(..., description="Fine amount")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN fine = ? THEN inspection_id END) AS REAL) * 100 / COUNT(inspection_id) FROM violation", (fine,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get point IDs and fines for inspections on a specific date
@app.get("/v1/food_inspection_2/point_id_fine_inspection_date", operation_id="get_point_id_fine_inspection_date", summary="Retrieves the point IDs and associated fines for food inspections conducted on a specific date. The input parameter specifies the date of the inspections in 'YYYY-MM-DD' format.")
async def get_point_id_fine_inspection_date(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.point_id, T2.fine FROM inspection AS T1 INNER JOIN violation AS T2 ON T1.inspection_id = T2.inspection_id WHERE T1.inspection_date = ?", (inspection_date,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of inspections for a specific category
@app.get("/v1/food_inspection_2/inspection_count_category", operation_id="get_inspection_count_category", summary="Retrieves the total number of inspections conducted for a specific category. The category is provided as an input parameter.")
async def get_inspection_count_category(category: str = Query(..., description="Category")):
    cursor.execute("SELECT COUNT(T1.inspection_id) FROM violation AS T1 INNER JOIN inspection_point AS T2 ON T1.point_id = T2.point_id WHERE T2.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct establishment names and inspection results for a specific city
@app.get("/v1/food_inspection_2/establishment_results_city", operation_id="get_establishment_results_city", summary="Retrieves unique establishment names along with their respective inspection results for a specified city. The city parameter is used to filter the establishments.")
async def get_establishment_results_city(city: str = Query(..., description="City")):
    cursor.execute("SELECT DISTINCT T1.dba_name, T2.results FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of specific violation categories
@app.get("/v1/food_inspection_2/count_violation_categories", operation_id="get_count_violation_categories", summary="Retrieves the count of inspections with violations in two specified categories. The categories are provided as input parameters, and the operation returns the total number of inspections with violations in each category.")
async def get_count_violation_categories(category1: str = Query(..., description="First category of violation"), category2: str = Query(..., description="Second category of violation")):
    cursor.execute("SELECT COUNT(CASE WHEN T2.category = ? THEN T1.inspection_id END) AS Tox_nums , COUNT(CASE WHEN T2.category = ? THEN T1.inspection_id END) AS NosmoNums FROM violation AS T1 INNER JOIN inspection_point AS T2 ON T1.point_id = T2.point_id", (category1, category2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"Tox_nums": result[0], "NosmoNums": result[1]}

# Endpoint to get distinct establishment names based on inspection date and employee name
@app.get("/v1/food_inspection_2/distinct_establishment_names", operation_id="get_distinct_establishment_names", summary="Retrieves a list of unique establishment names associated with a specific inspection date and employee. The operation filters establishments based on the provided inspection date and employee's first and last names, returning only the distinct names of establishments that meet the criteria.")
async def get_distinct_establishment_names(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT DISTINCT T1.dba_name FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN employee AS T3 ON T2.employee_id = T3.employee_id WHERE T2.inspection_date = ? AND T3.first_name = ? AND T3.last_name = ?", (inspection_date, first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"dba_names": []}
    return {"dba_names": [row[0] for row in result]}

# Endpoint to get distinct violation categories and fines based on employee name and inspection month
@app.get("/v1/food_inspection_2/distinct_violation_categories_fines", operation_id="get_distinct_violation_categories_fines", summary="Retrieves unique violation categories and their associated fines for a specific employee's inspections in a given month. The employee is identified by their first and last names, and the month is specified in 'YYYY-MM' format.")
async def get_distinct_violation_categories_fines(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), inspection_month: str = Query(..., description="Inspection month in 'YYYY-MM' format")):
    cursor.execute("SELECT DISTINCT T4.category, T3.fine FROM inspection AS T1 INNER JOIN employee AS T2 ON T1.employee_id = T2.employee_id INNER JOIN violation AS T3 ON T1.inspection_id = T3.inspection_id INNER JOIN inspection_point AS T4 ON T3.point_id = T4.point_id WHERE T2.first_name = ? AND T2.last_name = ? AND strftime('%Y-%m', T1.inspection_date) = ?", (first_name, last_name, inspection_month))
    result = cursor.fetchall()
    if not result:
        return {"categories_fines": []}
    return {"categories_fines": [{"category": row[0], "fine": row[1]} for row in result]}

# Endpoint to get the count of inspections based on violation category
@app.get("/v1/food_inspection_2/count_inspections_by_category", operation_id="get_count_inspections_by_category", summary="Retrieves the total number of inspections associated with a specific violation category. The category is provided as an input parameter, allowing for a targeted count of inspections related to that category.")
async def get_count_inspections_by_category(category: str = Query(..., description="Category of the violation")):
    cursor.execute("SELECT COUNT(T2.inspection_id) FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T1.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get inspection types and results based on facility type
@app.get("/v1/food_inspection_2/inspection_types_results_by_facility_type", operation_id="get_inspection_types_results_by_facility_type", summary="Retrieves the types of inspections and their respective results for a specific facility type. The facility type is used to filter the data, providing a targeted view of inspection outcomes.")
async def get_inspection_types_results_by_facility_type(facility_type: str = Query(..., description="Type of the facility")):
    cursor.execute("SELECT T2.inspection_type, T2.results FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.facility_type = ?", (facility_type,))
    result = cursor.fetchall()
    if not result:
        return {"inspection_types_results": []}
    return {"inspection_types_results": [{"inspection_type": row[0], "results": row[1]} for row in result]}

# Endpoint to get employee names and inspection results based on establishment name
@app.get("/v1/food_inspection_2/employee_names_inspection_results_by_establishment", operation_id="get_employee_names_inspection_results_by_establishment", summary="Retrieves the first and last names of employees along with their respective inspection results for a specified establishment. The establishment is identified by its 'dba_name'.")
async def get_employee_names_inspection_results_by_establishment(dba_name: str = Query(..., description="Name of the establishment")):
    cursor.execute("SELECT T3.first_name, T3.last_name, T2.results FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN employee AS T3 ON T2.employee_id = T3.employee_id WHERE T1.dba_name = ?", (dba_name,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names_results": []}
    return {"employee_names_results": [{"first_name": row[0], "last_name": row[1], "results": row[2]} for row in result]}

# Endpoint to get the total fine amount based on establishment name, ward, and inspection results
@app.get("/v1/food_inspection_2/total_fine_amount", operation_id="get_total_fine_amount", summary="Retrieves the total fine amount for a specific establishment, based on its name, ward, and inspection results. The endpoint calculates the sum of all fines associated with the establishment, considering the provided ward and inspection results.")
async def get_total_fine_amount(dba_name: str = Query(..., description="Name of the establishment"), ward: int = Query(..., description="Ward number"), results: str = Query(..., description="Inspection results")):
    cursor.execute("SELECT SUM(T3.fine) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE T1.dba_name = ? AND T1.ward = ? AND T2.results = ?", (dba_name, ward, results))
    result = cursor.fetchone()
    if not result:
        return {"total_fine": []}
    return {"total_fine": result[0]}

# Endpoint to get the total fine amount based on violation category
@app.get("/v1/food_inspection_2/total_fine_by_category", operation_id="get_total_fine_by_category", summary="Retrieves the total fine amount for a specific violation category. This operation calculates the sum of all fines associated with the specified category, providing a comprehensive overview of the financial impact of violations in that category.")
async def get_total_fine_by_category(category: str = Query(..., description="Category of the violation")):
    cursor.execute("SELECT SUM(T2.fine) FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T1.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"total_fine": []}
    return {"total_fine": result[0]}

# Endpoint to get establishment details based on inspection date and results
@app.get("/v1/food_inspection_2/establishment_details_by_inspection", operation_id="get_establishment_details_by_inspection", summary="Retrieves the name, longitude, and latitude of establishments that underwent an inspection on a specific date and yielded certain results. The input parameters include the inspection date in 'YYYY-MM-DD' format and the inspection results.")
async def get_establishment_details_by_inspection(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), results: str = Query(..., description="Inspection results")):
    cursor.execute("SELECT T2.dba_name, T2.longitude, T2.latitude FROM inspection AS T1 INNER JOIN establishment AS T2 ON T1.license_no = T2.license_no WHERE T1.inspection_date = ? AND T1.results = ?", (inspection_date, results))
    result = cursor.fetchall()
    if not result:
        return {"establishment_details": []}
    return {"establishment_details": [{"dba_name": row[0], "longitude": row[1], "latitude": row[2]} for row in result]}

# Endpoint to get the percentage of inspections passing and the count of distinct establishments based on city
@app.get("/v1/food_inspection_2/inspection_pass_percentage_and_establishment_count", operation_id="get_inspection_pass_percentage_and_establishment_count", summary="Retrieves the percentage of successful inspections and the total number of unique establishments in a specified city. The inspection results are evaluated against a provided pattern to determine success.")
async def get_inspection_pass_percentage_and_establishment_count(results_pattern: str = Query(..., description="Pattern to match in inspection results (e.g., '%Pass%')"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.results LIKE ? THEN T2.inspection_id END) AS REAL) * 100 / COUNT(T2.inspection_id), COUNT(DISTINCT T2.license_no) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.city = ?", (results_pattern, city))
    result = cursor.fetchone()
    if not result:
        return {"pass_percentage": [], "establishment_count": []}
    return {"pass_percentage": result[0], "establishment_count": result[1]}

# Endpoint to get the average number of inspections per year for a specific employee
@app.get("/v1/food_inspection_2/average_inspections_per_year", operation_id="get_average_inspections_per_year", summary="Retrieves the average annual number of inspections conducted by a specific employee between the provided start and end years. The calculation is based on the total count of inspections performed by the employee, divided by the number of years in the specified range.")
async def get_average_inspections_per_year(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.first_name = ? AND T1.last_name = ? THEN T2.inspection_id ELSE 0 END) AS REAL) / 8 FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE strftime('%Y', T2.inspection_date) BETWEEN ? AND ?", (first_name, last_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the first name of an employee based on inspection ID
@app.get("/v1/food_inspection_2/employee_first_name_by_inspection_id", operation_id="get_employee_first_name_by_inspection_id", summary="Retrieves the first name of the employee associated with the provided inspection ID. This operation fetches the employee's first name from the database by joining the employee and inspection tables using the employee_id field and filtering the results based on the given inspection_id.")
async def get_employee_first_name_by_inspection_id(inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT T1.first_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.inspection_id = ?", (inspection_id,))
    result = cursor.fetchone()
    if not result:
        return {"first_name": []}
    return {"first_name": result[0]}

# Endpoint to get the address of an employee based on inspection ID
@app.get("/v1/food_inspection_2/employee_address_by_inspection_id", operation_id="get_employee_address_by_inspection_id", summary="Retrieves the address of the employee associated with the provided inspection ID. This operation fetches the address from the employee table by joining it with the inspection table using the employee_id field and filtering the results based on the given inspection_id.")
async def get_employee_address_by_inspection_id(inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT T1.address FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.inspection_id = ?", (inspection_id,))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get the last name of an employee based on inspection ID
@app.get("/v1/food_inspection_2/employee_last_name_by_inspection_id", operation_id="get_employee_last_name_by_inspection_id", summary="Retrieves the last name of the employee associated with the provided inspection ID. This operation fetches the employee's last name from the database by joining the employee and inspection tables using the employee_id field.")
async def get_employee_last_name_by_inspection_id(inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT T1.last_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.inspection_id = ?", (inspection_id,))
    result = cursor.fetchone()
    if not result:
        return {"last_name": []}
    return {"last_name": result[0]}

# Endpoint to get distinct inspection results for a specific employee
@app.get("/v1/food_inspection_2/distinct_inspection_results_by_employee", operation_id="get_distinct_inspection_results_by_employee", summary="Retrieve a unique set of inspection results associated with a specific employee, identified by their first and last names.")
async def get_distinct_inspection_results_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT DISTINCT T2.results FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"results": []}
    return {"results": [row[0] for row in result]}

# Endpoint to get distinct addresses of employees based on inspection date
@app.get("/v1/food_inspection_2/distinct_employee_addresses_by_inspection_date", operation_id="get_distinct_employee_addresses_by_inspection_date", summary="Retrieves a list of unique addresses associated with employees who conducted inspections on a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_distinct_employee_addresses_by_inspection_date(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T1.address FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.inspection_date = ?", (inspection_date,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [row[0] for row in result]}

# Endpoint to get distinct phone numbers of employees based on inspection type
@app.get("/v1/food_inspection_2/distinct_employee_phones_by_inspection_type", operation_id="get_distinct_employee_phones_by_inspection_type", summary="Retrieves a unique set of employee phone numbers associated with a specific inspection type.")
async def get_distinct_employee_phones_by_inspection_type(inspection_type: str = Query(..., description="Inspection type")):
    cursor.execute("SELECT DISTINCT T1.phone FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.inspection_type = ?", (inspection_type,))
    result = cursor.fetchall()
    if not result:
        return {"phones": []}
    return {"phones": [row[0] for row in result]}

# Endpoint to get distinct inspection results for a specific establishment
@app.get("/v1/food_inspection_2/distinct_inspection_results_by_establishment", operation_id="get_distinct_inspection_results_by_establishment", summary="Retrieve the unique inspection results for a specific establishment, identified by its DBA name.")
async def get_distinct_inspection_results_by_establishment(dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT DISTINCT T2.results FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.dba_name = ?", (dba_name,))
    result = cursor.fetchall()
    if not result:
        return {"results": []}
    return {"results": [row[0] for row in result]}

# Endpoint to get distinct inspection types for a specific establishment
@app.get("/v1/food_inspection_2/distinct_inspection_types_by_establishment", operation_id="get_distinct_inspection_types_by_establishment", summary="Retrieves a unique set of inspection types associated with a specific establishment, identified by its DBA name.")
async def get_distinct_inspection_types_by_establishment(dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT DISTINCT T2.inspection_type FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.dba_name = ?", (dba_name,))
    result = cursor.fetchall()
    if not result:
        return {"inspection_types": []}
    return {"inspection_types": [row[0] for row in result]}

# Endpoint to get the DBA names of establishments based on inspection type
@app.get("/v1/food_inspection_2/dba_names_by_inspection_type", operation_id="get_dba_names_by_inspection_type", summary="Retrieves the names of establishments (DBA names) that have undergone a specific type of inspection. The inspection type is provided as an input parameter, allowing the user to filter the results accordingly.")
async def get_dba_names_by_inspection_type(inspection_type: str = Query(..., description="Type of inspection")):
    cursor.execute("SELECT T1.dba_name FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T2.inspection_type = ?", (inspection_type,))
    result = cursor.fetchall()
    if not result:
        return {"dba_names": []}
    return {"dba_names": [row[0] for row in result]}

# Endpoint to get the count of inspections based on year and risk level
@app.get("/v1/food_inspection_2/inspection_count_by_year_risk_level", operation_id="get_inspection_count_by_year_risk_level", summary="Retrieves the total number of inspections conducted in a given year, filtered by a specified risk level. The year should be provided in 'YYYY' format, and the risk level is a categorical value.")
async def get_inspection_count_by_year_risk_level(year: str = Query(..., description="Year in 'YYYY' format"), risk_level: int = Query(..., description="Risk level")):
    cursor.execute("SELECT COUNT(T2.inspection_id) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE strftime('%Y', T2.inspection_date) = ? AND T1.risk_level = ?", (year, risk_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct DBA names based on inspection results
@app.get("/v1/food_inspection_2/dba_names_by_inspection_results", operation_id="get_dba_names_by_inspection_results", summary="Retrieve a unique list of business names (DBA) that have undergone inspections with results containing a specified substring. This operation filters establishments based on the provided substring in their inspection results, ensuring only relevant businesses are returned.")
async def get_dba_names_by_inspection_results(results: str = Query(..., description="Substring to match in inspection results")):
    cursor.execute("SELECT DISTINCT T1.dba_name FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T2.results LIKE ?", ('%' + results + '%',))
    result = cursor.fetchall()
    if not result:
        return {"dba_names": []}
    return {"dba_names": [row[0] for row in result]}

# Endpoint to get the total salary of employees based on inspection ID range
@app.get("/v1/food_inspection_2/total_salary_by_inspection_id_range", operation_id="get_total_salary_by_inspection_id_range", summary="Retrieves the total salary of employees who have conducted inspections within a specified range of inspection IDs. The range is defined by the minimum and maximum inspection IDs provided as input parameters.")
async def get_total_salary_by_inspection_id_range(min_inspection_id: int = Query(..., description="Minimum inspection ID"), max_inspection_id: int = Query(..., description="Maximum inspection ID")):
    cursor.execute("SELECT SUM(T2.salary) FROM inspection AS T1 INNER JOIN employee AS T2 ON T1.employee_id = T2.employee_id WHERE T1.inspection_id BETWEEN ? AND ?", (min_inspection_id, max_inspection_id))
    result = cursor.fetchone()
    if not result:
        return {"total_salary": []}
    return {"total_salary": result[0]}

# Endpoint to get the average salary of employees based on inspection type
@app.get("/v1/food_inspection_2/average_salary_by_inspection_type", operation_id="get_average_salary_by_inspection_type", summary="Retrieves the average salary of employees who have conducted inspections of a specified type. The inspection type is provided as an input parameter.")
async def get_average_salary_by_inspection_type(inspection_type: str = Query(..., description="Type of inspection")):
    cursor.execute("SELECT AVG(T2.salary) FROM inspection AS T1 INNER JOIN employee AS T2 ON T1.employee_id = T2.employee_id WHERE T1.inspection_type = ?", (inspection_type,))
    result = cursor.fetchone()
    if not result:
        return {"average_salary": []}
    return {"average_salary": result[0]}

# Endpoint to get distinct inspection results and ZIP codes based on license number
@app.get("/v1/food_inspection_2/inspection_results_zip_by_license_no", operation_id="get_inspection_results_zip_by_license_no", summary="Retrieve unique combinations of inspection results and associated ZIP codes for a given license number.")
async def get_inspection_results_zip_by_license_no(license_no: int = Query(..., description="License number")):
    cursor.execute("SELECT DISTINCT T2.results, T1.zip FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.license_no = ?", (license_no,))
    result = cursor.fetchall()
    if not result:
        return {"inspection_results_zip": []}
    return {"inspection_results_zip": [{"results": row[0], "zip": row[1]} for row in result]}

# Endpoint to get the earliest inspection date based on DBA name and year
@app.get("/v1/food_inspection_2/earliest_inspection_date_by_dba_name_year", operation_id="get_earliest_inspection_date_by_dba_name_year", summary="Retrieves the earliest inspection date for a given DBA name within a specified year. The operation filters establishments by the provided DBA name and identifies the minimum inspection date from the corresponding inspections in the given year.")
async def get_earliest_inspection_date_by_dba_name_year(dba_name: str = Query(..., description="DBA name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT MIN(T2.inspection_date) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.dba_name = ? AND strftime('%Y', T2.inspection_date) = ?", (dba_name, year))
    result = cursor.fetchone()
    if not result:
        return {"earliest_inspection_date": []}
    return {"earliest_inspection_date": result[0]}

# Endpoint to get distinct employee names based on license number
@app.get("/v1/food_inspection_2/employee_names_by_license_no", operation_id="get_employee_names_by_license_no", summary="Retrieve a unique list of employee names associated with a given license number. This operation fetches the distinct first and last names of employees who have conducted inspections under the specified license number.")
async def get_employee_names_by_license_no(license_no: int = Query(..., description="License number")):
    cursor.execute("SELECT DISTINCT T1.first_name, T1.last_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.license_no = ?", (license_no,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of distinct employees based on inspection results and salary
@app.get("/v1/food_inspection_2/employee_count_by_inspection_results_salary", operation_id="get_employee_count_by_inspection_results_salary", summary="Retrieves the number of unique employees who have conducted inspections with specific results and earn a salary above a certain threshold.")
async def get_employee_count_by_inspection_results_salary(results: str = Query(..., description="Inspection results"), salary: int = Query(..., description="Minimum salary")):
    cursor.execute("SELECT COUNT(DISTINCT T1.employee_id) FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.results = ? AND T1.salary > ?", (results, salary))
    result = cursor.fetchone()
    if not result:
        return {"employee_count": []}
    return {"employee_count": result[0]}

# Endpoint to get distinct DBA names based on year, inspection results, and facility type
@app.get("/v1/food_inspection_2/dba_names_by_year_results_facility_type", operation_id="get_dba_names_by_year_results_facility_type", summary="Retrieves a list of unique business names (DBA) that match the specified year, inspection results, and facility type. The operation filters establishments based on the provided year, inspection results, and facility type, and returns the distinct DBA names associated with these establishments.")
async def get_dba_names_by_year_results_facility_type(year: str = Query(..., description="Year in 'YYYY' format"), results: str = Query(..., description="Inspection results"), facility_type: str = Query(..., description="Facility type")):
    cursor.execute("SELECT DISTINCT T1.dba_name FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE strftime('%Y', T2.inspection_date) = ? AND T2.results = ? AND T1.facility_type = ?", (year, results, facility_type))
    result = cursor.fetchall()
    if not result:
        return {"dba_names": []}
    return {"dba_names": [row[0] for row in result]}

# Endpoint to get distinct risk levels for establishments inspected by a specific employee
@app.get("/v1/food_inspection_2/distinct_risk_levels_by_employee", operation_id="get_distinct_risk_levels_by_employee", summary="Retrieve the unique risk levels associated with establishments inspected by a specific employee. This operation requires the first and last names of the employee as input parameters to filter the results accordingly.")
async def get_distinct_risk_levels_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT DISTINCT T3.risk_level FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id INNER JOIN establishment AS T3 ON T2.license_no = T3.license_no WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"risk_levels": []}
    return {"risk_levels": [row[0] for row in result]}

# Endpoint to get distinct establishment names and inspection results for a specific employee and year
@app.get("/v1/food_inspection_2/distinct_establishment_names_results_by_employee_year", operation_id="get_distinct_establishment_names_results_by_employee_year", summary="Retrieve unique establishment names and their corresponding inspection results for a specific employee in a given year. The operation filters data based on the employee's first and last name, and the year of the inspection.")
async def get_distinct_establishment_names_results_by_employee_year(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T3.dba_name, T2.results FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id INNER JOIN establishment AS T3 ON T2.license_no = T3.license_no WHERE T1.first_name = ? AND T1.last_name = ? AND strftime('%Y', T2.inspection_date) = ?", (first_name, last_name, year))
    result = cursor.fetchall()
    if not result:
        return {"establishments": []}
    return {"establishments": [{"dba_name": row[0], "results": row[1]} for row in result]}

# Endpoint to get distinct titles of employees who inspected a specific license number
@app.get("/v1/food_inspection_2/distinct_employee_titles_by_license_number", operation_id="get_distinct_employee_titles_by_license_number", summary="Retrieve a unique set of job titles held by employees who have inspected a given license number. The license number is provided as an input parameter.")
async def get_distinct_employee_titles_by_license_number(license_no: int = Query(..., description="License number")):
    cursor.execute("SELECT DISTINCT T1.title FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.license_no = ?", (license_no,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of distinct inspection points with a specific level and fine amount
@app.get("/v1/food_inspection_2/count_distinct_inspection_points", operation_id="get_count_distinct_inspection_points", summary="Get the count of distinct inspection points with a specific level and fine amount")
async def get_count_distinct_inspection_points(point_level: str = Query(..., description="Inspection point level"), fine: int = Query(..., description="Fine amount")):
    cursor.execute("SELECT COUNT(DISTINCT T2.point_id) FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T1.point_level = ? AND T2.fine = ?", (point_level, fine))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of establishments with a specific fine amount and facility type
@app.get("/v1/food_inspection_2/percentage_establishments_with_fine", operation_id="get_percentage_establishments_with_fine", summary="Retrieves the percentage of establishments with a specified fine amount and facility type. This operation calculates the ratio of establishments with the given fine amount to the total number of establishments of the specified facility type.")
async def get_percentage_establishments_with_fine(fine: int = Query(..., description="Fine amount"), facility_type: str = Query(..., description="Facility type")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.fine = ? THEN T1.license_no END) AS REAL) * 100 / COUNT(T1.license_no) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE T1.facility_type = ?", (fine, facility_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of establishments with a specific risk level and inspection result
@app.get("/v1/food_inspection_2/percentage_establishments_with_risk_level", operation_id="get_percentage_establishments_with_risk_level", summary="Retrieves the percentage of establishments with a specified risk level and inspection result. This operation calculates the ratio of establishments with the given risk level and inspection result to the total number of establishments. The risk level and inspection result are provided as input parameters.")
async def get_percentage_establishments_with_risk_level(risk_level: int = Query(..., description="Risk level"), results: str = Query(..., description="Inspection results")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.risk_level = ? THEN T1.license_no END) AS REAL) * 100 / COUNT(T1.license_no) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T2.results = ?", (risk_level, results))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the address, city, and state of an employee by their first and last name
@app.get("/v1/food_inspection_2/employee_address", operation_id="get_employee_address", summary="Retrieves the residential address, city, and state of a specific employee based on their first and last name. The endpoint requires both names to be provided for accurate identification.")
async def get_employee_address(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT address, city, state FROM employee WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0], "city": result[1], "state": result[2]}

# Endpoint to get the facility type of an establishment by its DBA name
@app.get("/v1/food_inspection_2/facility_type_by_dba_name", operation_id="get_facility_type_by_dba_name", summary="Retrieves the type of facility associated with a specific business name. The operation uses the provided DBA name to search for the corresponding establishment and returns its facility type.")
async def get_facility_type_by_dba_name(dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT facility_type FROM establishment WHERE dba_name = ?", (dba_name,))
    result = cursor.fetchone()
    if not result:
        return {"facility_type": []}
    return {"facility_type": result[0]}

# Endpoint to get the salary of an employee by their first and last name
@app.get("/v1/food_inspection_2/employee_salary", operation_id="get_employee_salary", summary="Retrieves the salary of a specific employee identified by their first and last names.")
async def get_employee_salary(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT salary FROM employee WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"salary": []}
    return {"salary": result[0]}

# Endpoint to get the DBA name of an establishment based on latitude, longitude, and facility type
@app.get("/v1/food_inspection_2/dba_name_by_location_and_type", operation_id="get_dba_name_by_location_and_type", summary="Retrieves the DBA name of a specific establishment based on its geographical coordinates and facility type. The endpoint uses the provided latitude and longitude to pinpoint the establishment's location and the facility type to filter the results. This operation is useful for obtaining the DBA name of an establishment when its exact location and type are known.")
async def get_dba_name_by_location_and_type(latitude: float = Query(..., description="Latitude of the establishment"), longitude: float = Query(..., description="Longitude of the establishment"), facility_type: str = Query(..., description="Type of the facility")):
    cursor.execute("SELECT dba_name FROM establishment WHERE latitude = ? AND longitude = ? AND facility_type = ?", (latitude, longitude, facility_type))
    result = cursor.fetchone()
    if not result:
        return {"dba_name": []}
    return {"dba_name": result[0]}

# Endpoint to get the count of employees based on their title
@app.get("/v1/food_inspection_2/employee_count_by_title", operation_id="get_employee_count_by_title", summary="Retrieves the total number of employees with a specific job title. The title is provided as an input parameter.")
async def get_employee_count_by_title(title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT COUNT(employee_id) FROM employee WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the address of an establishment based on city, DBA name, and facility type
@app.get("/v1/food_inspection_2/address_by_city_dba_and_type", operation_id="get_address_by_city_dba_and_type", summary="Retrieves the address of a specific establishment based on the provided city, DBA name, and facility type. This operation allows you to locate an establishment's address by specifying these three identifying attributes.")
async def get_address_by_city_dba_and_type(city: str = Query(..., description="City of the establishment"), dba_name: str = Query(..., description="DBA name of the establishment"), facility_type: str = Query(..., description="Type of the facility")):
    cursor.execute("SELECT address FROM establishment WHERE city = ? AND dba_name = ? AND facility_type = ?", (city, dba_name, facility_type))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get the count of employees based on state and city
@app.get("/v1/food_inspection_2/employee_count_by_state_and_city", operation_id="get_employee_count_by_state_and_city", summary="Retrieves the total number of employees located in a specific state and city. The operation requires the state and city as input parameters to filter the employee count accordingly.")
async def get_employee_count_by_state_and_city(state: str = Query(..., description="State of the employee"), city: str = Query(..., description="City of the employee")):
    cursor.execute("SELECT COUNT(employee_id) FROM employee WHERE state = ? AND city = ?", (state, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of establishments based on risk level
@app.get("/v1/food_inspection_2/establishment_count_by_risk_level", operation_id="get_establishment_count_by_risk_level", summary="Retrieves the total number of establishments that match the specified risk level. The risk level is a categorical value that indicates the potential health risk posed by the establishment.")
async def get_establishment_count_by_risk_level(risk_level: int = Query(..., description="Risk level of the establishment")):
    cursor.execute("SELECT COUNT(license_no) FROM establishment WHERE risk_level = ?", (risk_level,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last names of employees based on salary
@app.get("/v1/food_inspection_2/employee_names_by_salary", operation_id="get_employee_names_by_salary", summary="Retrieves the first and last names of employees who earn a specified salary. The salary is provided as an input parameter.")
async def get_employee_names_by_salary(salary: int = Query(..., description="Salary of the employee")):
    cursor.execute("SELECT first_name, last_name FROM employee WHERE salary = ?", (salary,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get distinct last names of employees based on inspection ID
@app.get("/v1/food_inspection_2/distinct_employee_last_names_by_inspection_id", operation_id="get_distinct_employee_last_names_by_inspection_id", summary="Retrieves a list of unique last names of employees who have conducted inspections with the provided inspection ID.")
async def get_distinct_employee_last_names_by_inspection_id(inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT DISTINCT T1.last_name FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.inspection_id = ?", (inspection_id,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get distinct DBA names of establishments based on facility type and fine amount
@app.get("/v1/food_inspection_2/distinct_dba_names_by_facility_type_and_fine", operation_id="get_distinct_dba_names_by_facility_type_and_fine", summary="Retrieve a unique list of business names for establishments that match a specific facility type and fine amount. This operation filters establishments based on their facility type and the fine amount associated with their inspections, providing a distinct set of business names that meet the specified criteria.")
async def get_distinct_dba_names_by_facility_type_and_fine(facility_type: str = Query(..., description="Type of the facility"), fine: int = Query(..., description="Fine amount")):
    cursor.execute("SELECT DISTINCT T1.dba_name FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE T1.facility_type = ? AND T3.fine = ?", (facility_type, fine))
    result = cursor.fetchall()
    if not result:
        return {"dba_names": []}
    return {"dba_names": [row[0] for row in result]}

# Endpoint to get distinct point levels based on fine amount
@app.get("/v1/food_inspection_2/distinct_point_levels_by_fine", operation_id="get_distinct_point_levels_by_fine", summary="Retrieves unique point levels associated with a specific fine amount. The operation filters point levels based on the provided fine amount, returning only those that match the given fine.")
async def get_distinct_point_levels_by_fine(fine: int = Query(..., description="Fine amount")):
    cursor.execute("SELECT DISTINCT T1.point_level FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T2.fine = ?", (fine,))
    result = cursor.fetchall()
    if not result:
        return {"point_levels": []}
    return {"point_levels": [row[0] for row in result]}

# Endpoint to get distinct facility types and license numbers based on risk level and inspection results
@app.get("/v1/food_inspection_2/distinct_facility_types_and_license_nos_by_risk_level_and_results", operation_id="get_facility_types_and_license_nos_by_risk_and_result", summary="Retrieves unique facility types and their corresponding license numbers, filtered by a specified risk level and inspection result. This operation provides a concise overview of establishments that meet the given criteria, aiding in targeted analysis and decision-making.")
async def get_distinct_facility_types_and_license_nos_by_risk_level_and_results(risk_level: int = Query(..., description="Risk level of the establishment"), results: str = Query(..., description="Inspection results")):
    cursor.execute("SELECT DISTINCT T1.facility_type, T1.license_no FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T1.risk_level = ? AND T2.results = ?", (risk_level, results))
    result = cursor.fetchall()
    if not result:
        return {"facility_types_and_license_nos": []}
    return {"facility_types_and_license_nos": [{"facility_type": row[0], "license_no": row[1]} for row in result]}

# Endpoint to get distinct inspection results for a specific employee on a specific date
@app.get("/v1/food_inspection_2/distinct_inspection_results", operation_id="get_distinct_inspection_results", summary="Retrieve the unique inspection results for a specific employee on a given date. The operation requires the employee's first and last name, as well as the inspection date in 'YYYY-MM-DD' format. The results returned are distinct, meaning no duplicates are included.")
async def get_distinct_inspection_results(inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format"), first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT DISTINCT T2.results FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.inspection_date = ? AND T1.first_name = ? AND T1.last_name = ?", (inspection_date, first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"results": []}
    return {"results": [row[0] for row in result]}

# Endpoint to get distinct inspection types for a specific employee on a specific date
@app.get("/v1/food_inspection_2/distinct_inspection_types", operation_id="get_distinct_inspection_types", summary="Retrieve the unique types of inspections conducted by a specific employee on a given date. The operation requires the employee's first and last names, as well as the inspection date in 'YYYY-MM-DD' format.")
async def get_distinct_inspection_types(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.inspection_type FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T1.first_name = ? AND T1.last_name = ? AND T2.inspection_date = ?", (first_name, last_name, inspection_date))
    result = cursor.fetchall()
    if not result:
        return {"inspection_types": []}
    return {"inspection_types": [row[0] for row in result]}

# Endpoint to get inspection IDs based on inspector comment and code
@app.get("/v1/food_inspection_2/inspection_ids_by_comment_and_code", operation_id="get_inspection_ids_by_comment_and_code", summary="Retrieves the inspection IDs associated with a specific inspector comment and a predefined code. The operation filters inspection points based on the provided inspector comment and a fixed code, then returns the corresponding inspection IDs.")
async def get_inspection_ids_by_comment_and_code(inspector_comment: str = Query(..., description="Inspector comment"), code: str = Query(..., description="Code")):
    cursor.execute("SELECT T2.inspection_id FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T2.inspector_comment = ? AND T1.code = ?", (inspector_comment, code))
    result = cursor.fetchall()
    if not result:
        return {"inspection_ids": []}
    return {"inspection_ids": [row[0] for row in result]}

# Endpoint to get inspection IDs for a specific employee and DBA name
@app.get("/v1/food_inspection_2/inspection_ids_by_employee_and_dba", operation_id="get_inspection_ids_by_employee_and_dba", summary="Retrieves a list of inspection IDs associated with a specific employee and establishment. The employee is identified by their first and last names, while the establishment is identified by its DBA name. The operation returns the inspection IDs where the employee and establishment match the provided criteria.")
async def get_inspection_ids_by_employee_and_dba(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), dba_name: str = Query(..., description="DBA name of the establishment")):
    cursor.execute("SELECT T2.inspection_id FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN employee AS T3 ON T2.employee_id = T3.employee_id WHERE T3.first_name = ? AND T3.last_name = ? AND T1.dba_name = ?", (first_name, last_name, dba_name))
    result = cursor.fetchall()
    if not result:
        return {"inspection_ids": []}
    return {"inspection_ids": [row[0] for row in result]}

# Endpoint to get salaries of employees involved in a specific inspection
@app.get("/v1/food_inspection_2/employee_salaries_by_inspection", operation_id="get_employee_salaries_by_inspection", summary="Retrieves the highest and lowest salaries of employees who participated in a specific inspection. The inspection is identified by its unique ID.")
async def get_employee_salaries_by_inspection(inspection_id: int = Query(..., description="Inspection ID")):
    cursor.execute("SELECT T1.salary, T3.salary FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id INNER JOIN employee AS T3 WHERE T2.inspection_id = ? ORDER BY T1.salary, T3.salary DESC LIMIT 1", (inspection_id,))
    result = cursor.fetchone()
    if not result:
        return {"salaries": []}
    return {"salaries": [result[0], result[1]]}

# Endpoint to get inspection IDs based on category and inspector comment
@app.get("/v1/food_inspection_2/inspection_ids_by_category_and_comment", operation_id="get_inspection_ids_by_category_and_comment", summary="Retrieves a list of inspection IDs that match a specific category and inspector comment. The category and comment are used to filter the inspection points and associated violations, returning only the relevant inspection IDs.")
async def get_inspection_ids_by_category_and_comment(category: str = Query(..., description="Category"), inspector_comment: str = Query(..., description="Inspector comment")):
    cursor.execute("SELECT T2.inspection_id FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T1.category = ? AND T2.inspector_comment = ?", (category, inspector_comment))
    result = cursor.fetchall()
    if not result:
        return {"inspection_ids": []}
    return {"inspection_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct license numbers for a specific facility type and fine amount
@app.get("/v1/food_inspection_2/count_distinct_license_numbers_by_facility_type_and_fine", operation_id="get_count_distinct_license_numbers_by_facility_type_and_fine", summary="Retrieve the number of unique establishments, categorized by a specific facility type and fine amount, that have been inspected and found to have violations.")
async def get_count_distinct_license_numbers_by_facility_type_and_fine(facility_type: str = Query(..., description="Type of the facility"), fine: int = Query(..., description="Fine amount")):
    cursor.execute("SELECT COUNT(DISTINCT T1.license_no) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE T1.facility_type = ? AND T3.fine = ?", (facility_type, fine))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct categories for a given establishment name
@app.get("/v1/food_inspection_2/distinct_categories_by_establishment_name", operation_id="get_distinct_categories", summary="Retrieve a unique set of categories associated with a specific establishment. The operation filters the categories based on the provided establishment name.")
async def get_distinct_categories(dba_name: str = Query(..., description="Name of the establishment")):
    cursor.execute("SELECT DISTINCT T4.category FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id INNER JOIN inspection_point AS T4 ON T3.point_id = T4.point_id WHERE T1.dba_name = ?", (dba_name,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get the count of distinct inspection IDs based on point level and fine amount
@app.get("/v1/food_inspection_2/count_distinct_inspection_ids_by_point_level_fine", operation_id="get_count_distinct_inspection_ids", summary="Get the count of distinct inspection IDs based on point level and fine amount")
async def get_count_distinct_inspection_ids(point_level: str = Query(..., description="Point level"), fine: int = Query(..., description="Fine amount")):
    cursor.execute("SELECT COUNT(DISTINCT T2.inspection_id) FROM inspection_point AS T1 INNER JOIN violation AS T2 ON T1.point_id = T2.point_id WHERE T1.point_level = ? AND T2.fine = ?", (point_level, fine))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct establishment names based on violation category
@app.get("/v1/food_inspection_2/distinct_establishment_names_by_violation_category", operation_id="get_distinct_establishment_names_by_category", summary="Retrieves a list of unique establishment names that have been cited for a specific violation category. The operation filters establishments based on the provided violation category and returns their distinct names.")
async def get_distinct_establishment_names_by_category(category: str = Query(..., description="Violation category")):
    cursor.execute("SELECT DISTINCT T1.dba_name FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id INNER JOIN inspection_point AS T4 ON T3.point_id = T4.point_id WHERE T4.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"establishment_names": []}
    return {"establishment_names": [row[0] for row in result]}

# Endpoint to get the difference between pass and fail counts based on inspection type and facility type
@app.get("/v1/food_inspection_2/diff_pass_fail_counts_by_inspection_type_facility_type", operation_id="get_diff_pass_fail_counts", summary="Retrieve the difference between the number of passed and failed inspections for a specific inspection type and facility type. This operation calculates the count of passed inspections and subtracts the count of failed inspections based on the provided inspection type and facility type.")
async def get_diff_pass_fail_counts(inspection_type: str = Query(..., description="Inspection type"), facility_type: str = Query(..., description="Facility type")):
    cursor.execute("SELECT COUNT(CASE WHEN T2.results = 'Pass' THEN T1.license_no END) - COUNT(CASE WHEN T2.results = 'Fail' THEN T1.license_no END) AS diff FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no WHERE T2.inspection_type = ? AND T1.facility_type = ?", (inspection_type, facility_type))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get distinct employee IDs based on inspection result, date, and salary threshold
@app.get("/v1/food_inspection_2/distinct_employee_ids_by_inspection_result_date_salary", operation_id="get_distinct_employee_ids", summary="Retrieves a unique set of employee IDs who have a specific inspection result, conducted within a given month, and earn a salary above a certain percentage of the average salary.")
async def get_distinct_employee_ids(results: str = Query(..., description="Inspection result"), inspection_date: str = Query(..., description="Inspection date in 'YYYY-MM' format"), salary_threshold: float = Query(..., description="Salary threshold as a percentage of the average salary")):
    cursor.execute("SELECT DISTINCT T1.employee_id FROM employee AS T1 INNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id WHERE T2.results = ? AND strftime('%Y-%m', T2.inspection_date) = ? AND T1.salary > ? * ( SELECT AVG(salary) FROM employee )", (results, inspection_date, salary_threshold))
    result = cursor.fetchall()
    if not result:
        return {"employee_ids": []}
    return {"employee_ids": [row[0] for row in result]}

# Endpoint to get the percentage of restaurants based on fine amount
@app.get("/v1/food_inspection_2/percentage_restaurants_by_fine_amount", operation_id="get_percentage_restaurants", summary="Retrieves the percentage of restaurants that have been fined a specific amount. This operation calculates the ratio of restaurants with a fine equal to the provided amount to the total number of facilities, using data from the establishment, inspection, and violation tables.")
async def get_percentage_restaurants(fine: int = Query(..., description="Fine amount")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.facility_type = 'Restaurant' THEN T1.license_no END) AS REAL) * 100 / COUNT(T1.facility_type) FROM establishment AS T1 INNER JOIN inspection AS T2 ON T1.license_no = T2.license_no INNER JOIN violation AS T3 ON T2.inspection_id = T3.inspection_id WHERE T3.fine = ?", (fine,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/food_inspection_2/employee_details_by_title_supervisor?title=Sanitarian&supervisor_first_name=Darlisha&supervisor_last_name=Jacobs",
    "/v1/food_inspection_2/employee_names_by_inspection_date_title?inspection_date=2010-05&title=Sanitarian",
    "/v1/food_inspection_2/inspection_count_by_employee_year?year=2010&first_name=Joshua&last_name=Rosa",
    "/v1/food_inspection_2/establishment_names_by_employee?first_name=Joshua&last_name=Rosa",
    "/v1/food_inspection_2/establishment_count_by_year_facility_type?year=2010&facility_type=Restaurant",
    "/v1/food_inspection_2/establishment_coordinates_by_inspection_date?inspection_date=2010-05-11",
    "/v1/food_inspection_2/establishment_count_by_year_ward?year=2010&ward=42",
    "/v1/food_inspection_2/employee_names_by_establishment_title?dba_name=Burbank&title=Sanitarian",
    "/v1/food_inspection_2/establishment_names_by_inspection_results_year?results=Fail&year=2010",
    "/v1/food_inspection_2/employee_names_by_inspection_date_establishment_title?inspection_date=2010-05-11&dba_name=AMUNDSEN%20HIGH%20SCHOOL&title=Sanitarian",
    "/v1/food_inspection_2/count_inspections_by_employee?results=Pass&first_name=Joshua&last_name=Rosa",
    "/v1/food_inspection_2/followup_inspection_id?dba_name=Azha%20Restaurant%20Inc.&results=Pass&inspection_date=2010-01-21",
    "/v1/food_inspection_2/count_distinct_establishments?year=2010&risk_level=3",
    "/v1/food_inspection_2/inspection_to_salary_ratio?year=2010&salary=70000",
    "/v1/food_inspection_2/point_level_by_description?description=Refrigeration%20and%20metal%20stem%20thermometers%20provided%20and%20conspicuous",
    "/v1/food_inspection_2/employee_details_by_inspection_id?inspection_id=48224",
    "/v1/food_inspection_2/count_inspections_by_establishment?facility_type=Restaurant&dba_name=All%20Style%20Buffet",
    "/v1/food_inspection_2/earliest_inspection_date?aka_name=WING%20HUNG%20CHOP%20SUEY%20RESTAURANT",
    "/v1/food_inspection_2/count_establishments_by_inspection_date?inspection_date=2015-05-08&facility_type=Restaurant",
    "/v1/food_inspection_2/count_violation_points?inspection_id=1454071&category=Food%20Maintenance",
    "/v1/food_inspection_2/violation_count_by_date_and_establishment?inspection_date=2015-05-08&dba_name=ROYAL%20THAI%20CUISINE",
    "/v1/food_inspection_2/employee_with_most_inspections",
    "/v1/food_inspection_2/inspection_count_by_employee_and_results?first_name=Lisa&last_name=Tillman&results=Out%20of%20Business",
    "/v1/food_inspection_2/inspection_count_by_employee_address_title_and_month?address=5000%20N%20Wolcott%20Ave&title=Sanitarian&inspection_month=2011-05",
    "/v1/food_inspection_2/employee_phone_by_inspection_id_and_title?inspection_id=634597&title=Sanitarian",
    "/v1/food_inspection_2/salary_of_employee_with_most_inspections",
    "/v1/food_inspection_2/average_inspections_per_establishment?risk_level=3&facility_type=TAVERN",
    "/v1/food_inspection_2/percentage_passed_inspections?dba_name=POCKETS&facility_type=Restaurant",
    "/v1/food_inspection_2/employee_count_by_zip?zip=60617",
    "/v1/food_inspection_2/distinct_dba_names_by_address?address=2903%20W%20IRVING%20PARK%20RD",
    "/v1/food_inspection_2/employee_with_lowest_salary",
    "/v1/food_inspection_2/count_establishments_by_risk_level_and_dba_name?risk_level=2&dba_name=HOMEMADE%20PIZZA",
    "/v1/food_inspection_2/count_violations_by_point_id_and_fine?point_id=3&fine=500",
    "/v1/food_inspection_2/count_employees_under_supervisor?first_name=Gregory&last_name=Cardenas",
    "/v1/food_inspection_2/distinct_dba_names_by_inspection_date_and_results?month=2012-05&results=Pass%20w/%20Conditions",
    "/v1/food_inspection_2/count_distinct_license_numbers_by_employee_and_inspection?first_name=David&last_name=Hodges&employee_id=153225&inspection_type=Short%20Form%20Complaint&results=Pass",
    "/v1/food_inspection_2/count_distinct_license_numbers_by_date_range_ward_and_results?start_date=2010-01-01&end_date=2015-12-31&ward=42&results=Fail&min_count=5",
    "/v1/food_inspection_2/employee_salary_highest_inspections",
    "/v1/food_inspection_2/establishment_highest_fines_year?year=2014",
    "/v1/food_inspection_2/establishment_most_failed_inspections?results=Fail",
    "/v1/food_inspection_2/inspector_comments_establishment_date?inspection_date=2010-01-25&dba_name=TAQUERIA%20LA%20FIESTA",
    "/v1/food_inspection_2/total_fines_establishment_month_year?month_year=2014-02&dba_name=RON%20OF%20JAPAN%20INC",
    "/v1/food_inspection_2/employee_names_establishment?dba_name=TAQUERIA%20LA%20PAZ",
    "/v1/food_inspection_2/employee_highest_total_fines",
    "/v1/food_inspection_2/top_employees_by_salary_title?title=Sanitarian",
    "/v1/food_inspection_2/establishment_highest_inspections_pass_fail",
    "/v1/food_inspection_2/employee_last_names_address?address=7211%20S%20Hermitage%20Ave&city=Chicago&state=IL",
    "/v1/food_inspection_2/establishment_employee_details?inspection_date=2010-05-05&inspection_id=44256",
    "/v1/food_inspection_2/distinct_addresses?inspection_month=2010-03&results=Pass&facility_type=School",
    "/v1/food_inspection_2/distinct_employee_names?inspection_date=2010-03-09&inspection_type=Canvass",
    "/v1/food_inspection_2/distinct_inspection_ids?dba_name=PIZZA%20RUSTICA,%20INC",
    "/v1/food_inspection_2/count_distinct_licenses?risk_level=3&results=Pass&facility_type=Restaurant",
    "/v1/food_inspection_2/distinct_employee_names_by_category?category=Display%20of%20Inspection%20Report%20Summary",
    "/v1/food_inspection_2/employee_titles?inspection_id=60332",
    "/v1/food_inspection_2/count_distinct_licenses_by_criteria?risk_level=1&inspection_type=Complaint&facility_type=Restaurant&results=Fail",
    "/v1/food_inspection_2/distinct_fines_establishment_details?inspection_id=48216",
    "/v1/food_inspection_2/inspection_ids_by_criteria?fine=500&point_level=Critical&inspector_comment=CDI%20ON%205-17-10",
    "/v1/food_inspection_2/inspection_point_description_comment?inspection_id=44247",
    "/v1/food_inspection_2/inspector_comment_code?inspection_id=54216&point_id=34",
    "/v1/food_inspection_2/percentage_risk_level_fail?risk_level=3&results=Fail",
    "/v1/food_inspection_2/fine_difference_salary_range?fine1=100&fine2=500&min_salary=75000&max_salary=80000",
    "/v1/food_inspection_2/inspection_count_month_year?month_year=2011-01",
    "/v1/food_inspection_2/inspection_count_year_result?year=2014&results=Fail",
    "/v1/food_inspection_2/percentage_inspections_fine?fine=100",
    "/v1/food_inspection_2/point_id_fine_inspection_date?inspection_date=2010-08-07",
    "/v1/food_inspection_2/inspection_count_category?category=Personnel",
    "/v1/food_inspection_2/establishment_results_city?city=BURNHAM",
    "/v1/food_inspection_2/count_violation_categories?category1=Toxic%20Items&category2=No%20Smoking%20Regulations",
    "/v1/food_inspection_2/distinct_establishment_names?inspection_date=2012-11-20&first_name=Sarah&last_name=Lindsey",
    "/v1/food_inspection_2/distinct_violation_categories_fines?first_name=Lisa&last_name=Tillman&inspection_month=2014-01",
    "/v1/food_inspection_2/count_inspections_by_category?category=Display%20of%20Inspection%20Report%20Summary",
    "/v1/food_inspection_2/inspection_types_results_by_facility_type?facility_type=RIVERWALK%20CAFE",
    "/v1/food_inspection_2/employee_names_inspection_results_by_establishment?dba_name=JEAN%20SAMOCKI",
    "/v1/food_inspection_2/total_fine_amount?dba_name=HACIENDA%20LOS%20TORRES&ward=36&results=Fail",
    "/v1/food_inspection_2/total_fine_by_category?category=Food%20Equipment%20and%20Utensil",
    "/v1/food_inspection_2/establishment_details_by_inspection?inspection_date=2013-07-29&results=Fail",
    "/v1/food_inspection_2/inspection_pass_percentage_and_establishment_count?results_pattern=%25Pass%25&city=CHICAGO",
    "/v1/food_inspection_2/average_inspections_per_year?first_name=Jessica&last_name=Anthony&start_year=2010&end_year=2017",
    "/v1/food_inspection_2/employee_first_name_by_inspection_id?inspection_id=48225",
    "/v1/food_inspection_2/employee_address_by_inspection_id?inspection_id=52238",
    "/v1/food_inspection_2/employee_last_name_by_inspection_id?inspection_id=52238",
    "/v1/food_inspection_2/distinct_inspection_results_by_employee?first_name=Thomas&last_name=Langley",
    "/v1/food_inspection_2/distinct_employee_addresses_by_inspection_date?inspection_date=2010-11-05",
    "/v1/food_inspection_2/distinct_employee_phones_by_inspection_type?inspection_type=Canvass",
    "/v1/food_inspection_2/distinct_inspection_results_by_establishment?dba_name=XANDO%20COFFEE%20%26%20BAR%20%2F%20COSI%20SANDWICH%20BAR",
    "/v1/food_inspection_2/distinct_inspection_types_by_establishment?dba_name=JOHN%20SCHALLER",
    "/v1/food_inspection_2/dba_names_by_inspection_type?inspection_type=License",
    "/v1/food_inspection_2/inspection_count_by_year_risk_level?year=2010&risk_level=3",
    "/v1/food_inspection_2/dba_names_by_inspection_results?results=Pass",
    "/v1/food_inspection_2/total_salary_by_inspection_id_range?min_inspection_id=52270&max_inspection_id=52272",
    "/v1/food_inspection_2/average_salary_by_inspection_type?inspection_type=License%20Re-Inspection",
    "/v1/food_inspection_2/inspection_results_zip_by_license_no?license_no=1222441",
    "/v1/food_inspection_2/earliest_inspection_date_by_dba_name_year?dba_name=JOHN%20SCHALLER&year=2010",
    "/v1/food_inspection_2/employee_names_by_license_no?license_no=1334073",
    "/v1/food_inspection_2/employee_count_by_inspection_results_salary?results=Fail&salary=70000",
    "/v1/food_inspection_2/dba_names_by_year_results_facility_type?year=2010&results=Pass&facility_type=Liquor",
    "/v1/food_inspection_2/distinct_risk_levels_by_employee?first_name=Bob&last_name=Benson",
    "/v1/food_inspection_2/distinct_establishment_names_results_by_employee_year?first_name=Bob&last_name=Benson&year=2010",
    "/v1/food_inspection_2/distinct_employee_titles_by_license_number?license_no=1576687",
    "/v1/food_inspection_2/count_distinct_inspection_points?point_level=Serious&fine=0",
    "/v1/food_inspection_2/percentage_establishments_with_fine?fine=250&facility_type=Restaurant",
    "/v1/food_inspection_2/percentage_establishments_with_risk_level?risk_level=1&results=Pass",
    "/v1/food_inspection_2/employee_address?first_name=Standard&last_name=Murray",
    "/v1/food_inspection_2/facility_type_by_dba_name?dba_name=Kinetic%20Playground",
    "/v1/food_inspection_2/employee_salary?first_name=Jessica&last_name=Anthony",
    "/v1/food_inspection_2/dba_name_by_location_and_type?latitude=41.9532864854&longitude=-87.7673790701422&facility_type=Restaurant",
    "/v1/food_inspection_2/employee_count_by_title?title=Supervisor",
    "/v1/food_inspection_2/address_by_city_dba_and_type?city=CHICAGO&dba_name=OLD%20TIMERS%20REST%20%26%20LOUNGE&facility_type=Restaurant",
    "/v1/food_inspection_2/employee_count_by_state_and_city?state=IL&city=Hoffman%20Estates",
    "/v1/food_inspection_2/establishment_count_by_risk_level?risk_level=3",
    "/v1/food_inspection_2/employee_names_by_salary?salary=82700",
    "/v1/food_inspection_2/distinct_employee_last_names_by_inspection_id?inspection_id=52256",
    "/v1/food_inspection_2/distinct_dba_names_by_facility_type_and_fine?facility_type=Tavern&fine=100",
    "/v1/food_inspection_2/distinct_point_levels_by_fine?fine=0",
    "/v1/food_inspection_2/distinct_facility_types_and_license_nos_by_risk_level_and_results?risk_level=1&results=Fail",
    "/v1/food_inspection_2/distinct_inspection_results?inspection_date=2010-02-24&first_name=Arnold&last_name=Holder",
    "/v1/food_inspection_2/distinct_inspection_types?first_name=Lisa&last_name=Tillman&inspection_date=2010-07-07",
    "/v1/food_inspection_2/inspection_ids_by_comment_and_code?inspector_comment=MUST%20CLEAN%20AND%20BETTER%20ORGANIZE%20HALLWAY%20AREA&code=7-38-030,%20015,%20010%20(A),%20005%20(A)",
    "/v1/food_inspection_2/inspection_ids_by_employee_and_dba?first_name=David&last_name=Hodges&dba_name=KAMAYAN%20EXPRESS",
    "/v1/food_inspection_2/employee_salaries_by_inspection?inspection_id=58424",
    "/v1/food_inspection_2/inspection_ids_by_category_and_comment?category=Personnel&inspector_comment=A%20certified%20food%20service%20manager%20must%20be%20present%20in%20all%20establishments%20at%20which%20potentially%20hazardous%20food%20is%20prepared%20or%20served.FOUND%20NO%20CITY%20OF%20CHICAGO%20SANITATION%20CERTIFICATE%20POSTED%20OR%20VALID%20DOCUMENTATION%20DURING%20THIS%20INSPECTION.",
    "/v1/food_inspection_2/count_distinct_license_numbers_by_facility_type_and_fine?facility_type=Grocery%20Store&fine=250",
    "/v1/food_inspection_2/distinct_categories_by_establishment_name?dba_name=J%20%26%20J%20FOOD",
    "/v1/food_inspection_2/count_distinct_inspection_ids_by_point_level_fine?point_level=Serious&fine=0",
    "/v1/food_inspection_2/distinct_establishment_names_by_violation_category?category=No%20Smoking%20Regulations",
    "/v1/food_inspection_2/diff_pass_fail_counts_by_inspection_type_facility_type?inspection_type=Canvass&facility_type=Restaurant",
    "/v1/food_inspection_2/distinct_employee_ids_by_inspection_result_date_salary?results=Fail&inspection_date=2010-02&salary_threshold=0.7",
    "/v1/food_inspection_2/percentage_restaurants_by_fine_amount?fine=500"
]
