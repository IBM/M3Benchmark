from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/human_resources/human_resources.sqlite')
cursor = conn.cursor()

# Endpoint to get the top employee by salary
@app.get("/v1/human_resources/top_employee_by_salary", operation_id="get_top_employee_by_salary", summary="Retrieves the full name of the employee with the highest salary in the human resources department.")
async def get_top_employee_by_salary():
    cursor.execute("SELECT firstname, lastname FROM employee ORDER BY salary DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"firstname": result[0], "lastname": result[1]}}

# Endpoint to get the count of employees based on performance
@app.get("/v1/human_resources/employee_count_by_performance", operation_id="get_employee_count_by_performance", summary="Retrieves the total number of employees who have a specified performance rating. The performance rating is provided as an input parameter.")
async def get_employee_count_by_performance(performance: str = Query(..., description="Performance rating of the employee")):
    cursor.execute("SELECT COUNT(*) FROM employee WHERE performance = ?", (performance,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get SSNs of employees based on gender and salary
@app.get("/v1/human_resources/employee_ssns_by_gender_and_salary", operation_id="get_employee_ssns_by_gender_and_salary", summary="Retrieves the Social Security Numbers (SSNs) of employees who meet the specified gender and minimum salary criteria. The gender parameter filters employees by their gender, while the minimum salary parameter ensures that only employees with a salary greater than the provided amount are included in the results.")
async def get_employee_ssns_by_gender_and_salary(gender: str = Query(..., description="Gender of the employee"), min_salary: float = Query(..., description="Minimum salary of the employee")):
    cursor.execute("SELECT ssn FROM employee WHERE gender = ? AND CAST(REPLACE(SUBSTR(salary, 4), ',', '') AS REAL) > ?", (gender, min_salary))
    result = cursor.fetchall()
    if not result:
        return {"ssns": []}
    return {"ssns": [row[0] for row in result]}

# Endpoint to get the education required for a specific position
@app.get("/v1/human_resources/education_required_by_position", operation_id="get_education_required_by_position", summary="Retrieves the minimum education level required for a specific job position. The position title is used to identify the corresponding education requirement.")
async def get_education_required_by_position(position_title: str = Query(..., description="Title of the position")):
    cursor.execute("SELECT educationrequired FROM position WHERE positiontitle = ?", (position_title,))
    result = cursor.fetchone()
    if not result:
        return {"education_required": []}
    return {"education_required": result[0]}

# Endpoint to get the position title with the minimum salary among specified titles
@app.get("/v1/human_resources/min_salary_position_title", operation_id="get_min_salary_position_title", summary="Retrieves the position title with the lowest minimum salary from the provided list of position titles. This operation compares the minimum salaries of the specified positions and returns the title of the position with the lowest salary.")
async def get_min_salary_position_title(position_title1: str = Query(..., description="First position title"), position_title2: str = Query(..., description="Second position title")):
    cursor.execute("SELECT positiontitle FROM position WHERE positiontitle = ? OR positiontitle = ? ORDER BY minsalary ASC LIMIT 1", (position_title1, position_title2))
    result = cursor.fetchone()
    if not result:
        return {"position_title": []}
    return {"position_title": result[0]}

# Endpoint to get the city location of an employee by name
@app.get("/v1/human_resources/employee_city_by_name", operation_id="get_employee_city_by_name", summary="Retrieve the city location of a specific employee using their first and last names. This operation searches for an employee with the provided names and returns the corresponding city location.")
async def get_employee_city_by_name(lastname: str = Query(..., description="Last name of the employee"), firstname: str = Query(..., description="First name of the employee")):
    cursor.execute("SELECT T2.locationcity FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T1.lastname = ? AND T1.firstname = ?", (lastname, firstname))
    result = cursor.fetchone()
    if not result:
        return {"location_city": []}
    return {"location_city": result[0]}

# Endpoint to get the count of employees in a specific state with a specific performance rating
@app.get("/v1/human_resources/employee_count_by_state_and_performance", operation_id="get_employee_count_by_state_and_performance", summary="Retrieves the total number of employees in a given state who have a specific performance rating. The state and performance rating are provided as input parameters.")
async def get_employee_count_by_state_and_performance(state: str = Query(..., description="State of the employee's location"), performance: str = Query(..., description="Performance rating of the employee")):
    cursor.execute("SELECT COUNT(*) FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T2.state = ? AND T1.performance = ?", (state, performance))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the office phone number of an employee by name
@app.get("/v1/human_resources/employee_office_phone_by_name", operation_id="get_employee_office_phone_by_name", summary="Retrieves the office phone number of a specific employee using their first and last name. The operation searches for the employee's record in the employee table and joins it with the location table to obtain the office phone number. The input parameters are used to filter the employee records based on the provided first and last names.")
async def get_employee_office_phone_by_name(lastname: str = Query(..., description="Last name of the employee"), firstname: str = Query(..., description="First name of the employee")):
    cursor.execute("SELECT T2.officephone FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T1.lastname = ? AND T1.firstname = ?", (lastname, firstname))
    result = cursor.fetchone()
    if not result:
        return {"office_phone": []}
    return {"office_phone": result[0]}

# Endpoint to get the count of employees at a specific address with a specific gender
@app.get("/v1/human_resources/employee_count_by_address_and_gender", operation_id="get_employee_count_by_address_and_gender", summary="Retrieves the total number of employees at a given address who identify with a specific gender. The response is based on the provided address and gender parameters.")
async def get_employee_count_by_address_and_gender(address: str = Query(..., description="Address of the employee's location"), gender: str = Query(..., description="Gender of the employee")):
    cursor.execute("SELECT COUNT(*) FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T2.address = ? AND T1.gender = ?", (address, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of employees with a specific position title
@app.get("/v1/human_resources/employee_count_by_position_title", operation_id="get_employee_count_by_position_title", summary="Retrieves the total number of employees holding a specific position title. The position title is provided as an input parameter.")
async def get_employee_count_by_position_title(position_title: str = Query(..., description="Title of the position")):
    cursor.execute("SELECT COUNT(*) FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T2.positiontitle = ?", (position_title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the salary difference for a specific employee
@app.get("/v1/human_resources/salary_difference", operation_id="get_salary_difference", summary="Retrieves the difference between an employee's salary and the minimum salary for their position, based on the provided first and last names.")
async def get_salary_difference(lastname: str = Query(..., description="Last name of the employee"), firstname: str = Query(..., description="First name of the employee")):
    cursor.execute("SELECT CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL) - CAST(REPLACE(SUBSTR(T2.minsalary, 4), ',', '') AS REAL) AS diff FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.lastname = ? AND T1.firstname = ?", (lastname, firstname))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get the count of employees based on position title and state
@app.get("/v1/human_resources/employee_count_position_state", operation_id="get_employee_count_position_state", summary="Retrieves the total number of employees who hold a specific position title and are located in a particular state. The position title and state are provided as input parameters.")
async def get_employee_count_position_state(positiontitle: str = Query(..., description="Position title of the employee"), state: str = Query(..., description="State of the employee's location")):
    cursor.execute("SELECT COUNT(*) FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID INNER JOIN position AS T3 ON T3.positionID = T1.positionID WHERE T3.positiontitle = ? AND T2.state = ?", (positiontitle, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of employees based on position title
@app.get("/v1/human_resources/employee_names_by_position", operation_id="get_employee_names_by_position", summary="Retrieves the first and last names of employees who hold a specific position, as identified by the provided position title.")
async def get_employee_names_by_position(positiontitle: str = Query(..., description="Position title of the employee")):
    cursor.execute("SELECT T1.firstname, T1.lastname FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T2.positiontitle = ?", (positiontitle,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the names of employees based on specific name pairs
@app.get("/v1/human_resources/employee_names_by_name_pairs", operation_id="get_employee_names_by_name_pairs", summary="Retrieve the first and last names of employees who match the provided name pairs, sorted by the education level required for their position in descending order.")
async def get_employee_names_by_name_pairs(lastname1: str = Query(..., description="Last name of the first employee"), firstname1: str = Query(..., description="First name of the first employee"), lastname2: str = Query(..., description="Last name of the second employee"), firstname2: str = Query(..., description="First name of the second employee")):
    cursor.execute("SELECT T1.firstname, T1.lastname FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE (T1.lastname = ? AND T1.firstname = ?) OR (T1.lastname = ? AND T1.firstname = ?) ORDER BY T2.educationrequired DESC LIMIT 1", (lastname1, firstname1, lastname2, firstname2))
    result = cursor.fetchone()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the zipcodes of employees based on gender and performance
@app.get("/v1/human_resources/employee_zipcodes_by_gender_performance", operation_id="get_employee_zipcodes_by_gender_performance", summary="Retrieves the zipcodes of employees who match the specified gender and performance rating. The data is sourced from the employee and location tables, which are linked by the locationID field. This operation is useful for analyzing the geographical distribution of employees based on their gender and performance.")
async def get_employee_zipcodes_by_gender_performance(gender: str = Query(..., description="Gender of the employee"), performance: str = Query(..., description="Performance rating of the employee")):
    cursor.execute("SELECT T2.zipcode FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T1.gender = ? AND T1.performance = ?", (gender, performance))
    result = cursor.fetchall()
    if not result:
        return {"zipcodes": []}
    return {"zipcodes": result}

# Endpoint to get the SSNs of employees based on state
@app.get("/v1/human_resources/employee_ssns_by_state", operation_id="get_employee_ssns_by_state", summary="Retrieves the Social Security Numbers (SSNs) of employees who are located in the specified state. The state is provided as an input parameter.")
async def get_employee_ssns_by_state(state: str = Query(..., description="State of the employee's location")):
    cursor.execute("SELECT T1.ssn FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T2.state = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"ssns": []}
    return {"ssns": result}

# Endpoint to get the count of employees based on salary and position title
@app.get("/v1/human_resources/employee_count_salary_position", operation_id="get_employee_count_salary_position", summary="Retrieves the number of employees with a salary greater than the specified minimum and holding a specific position title.")
async def get_employee_count_salary_position(min_salary: int = Query(..., description="Minimum salary of the employee"), positiontitle: str = Query(..., description="Position title of the employee")):
    cursor.execute("SELECT COUNT(*) FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL) > ? AND T2.positiontitle = ?", (min_salary, positiontitle))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average salary of employees based on position title
@app.get("/v1/human_resources/average_salary_by_position", operation_id="get_average_salary_by_position", summary="Retrieves the average salary of employees holding a specific position. The position is identified by its title, which is provided as an input parameter. The result is calculated by averaging the salaries of all employees in the specified position.")
async def get_average_salary_by_position(positiontitle: str = Query(..., description="Position title of the employee")):
    cursor.execute("SELECT AVG(CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL)) AS avg FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T2.positiontitle = ?", (positiontitle,))
    result = cursor.fetchone()
    if not result:
        return {"avg": []}
    return {"avg": result[0]}

# Endpoint to get the percentage difference of average salary from minimum salary based on position title
@app.get("/v1/human_resources/percentage_salary_difference", operation_id="get_percentage_salary_difference", summary="Retrieves the percentage difference between the average salary of employees in a specific position and the minimum salary for that position. The position is identified by its title.")
async def get_percentage_salary_difference(positiontitle: str = Query(..., description="Position title of the employee")):
    cursor.execute("SELECT 100 * (AVG(CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL)) - CAST(REPLACE(SUBSTR(T2.minsalary, 4), ',', '') AS REAL)) / CAST(REPLACE(SUBSTR(T2.minsalary, 4), ',', '') AS REAL) AS per FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T2.positiontitle = ?", (positiontitle,))
    result = cursor.fetchone()
    if not result:
        return {"per": []}
    return {"per": result[0]}

# Endpoint to get the count of employees based on gender
@app.get("/v1/human_resources/employee_count_by_gender", operation_id="get_employee_count_by_gender", summary="Retrieves the total number of employees of a specific gender. The gender is provided as an input parameter.")
async def get_employee_count_by_gender(gender: str = Query(..., description="Gender of the employee")):
    cursor.execute("SELECT COUNT(*) FROM employee WHERE gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city location of an employee based on their first and last name
@app.get("/v1/human_resources/employee_location_city", operation_id="get_employee_location_city", summary="Retrieves the city location of a specific employee by matching their first and last names. The operation requires the employee's first and last names as input parameters to accurately identify the employee and return the corresponding city location.")
async def get_employee_location_city(firstname: str = Query(..., description="First name of the employee"), lastname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.locationcity FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T1.firstname = ? AND T1.lastname = ?", (firstname, lastname))
    result = cursor.fetchone()
    if not result:
        return {"location_city": []}
    return {"location_city": result[0]}

# Endpoint to get the state location of an employee based on their first and last name
@app.get("/v1/human_resources/employee_location_state", operation_id="get_employee_location_state", summary="Retrieves the state location of a specific employee by matching their first and last names. The operation uses the provided names to search for the corresponding employee record and returns the associated state location.")
async def get_employee_location_state(firstname: str = Query(..., description="First name of the employee"), lastname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.state FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T1.firstname = ? AND T1.lastname = ?", (firstname, lastname))
    result = cursor.fetchone()
    if not result:
        return {"state": []}
    return {"state": result[0]}

# Endpoint to get the required education for an employee's position based on their first name, last name, and gender
@app.get("/v1/human_resources/employee_education_required", operation_id="get_employee_education_required", summary="Retrieves the required education for a specific employee's position. The employee is identified by their first name, last name, and gender. The operation returns the education level required for the employee's position, as defined in the position table.")
async def get_employee_education_required(firstname: str = Query(..., description="First name of the employee"), lastname: str = Query(..., description="Last name of the employee"), gender: str = Query(..., description="Gender of the employee")):
    cursor.execute("SELECT T2.educationrequired FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.firstname = ? AND T1.lastname = ? AND T1.gender = ?", (firstname, lastname, gender))
    result = cursor.fetchone()
    if not result:
        return {"education_required": []}
    return {"education_required": result[0]}

# Endpoint to get the count of employees in a specific city
@app.get("/v1/human_resources/employee_count_by_city", operation_id="get_employee_count_by_city", summary="Retrieves the total number of employees working in a specified city. The city is identified by the provided locationcity parameter.")
async def get_employee_count_by_city(locationcity: str = Query(..., description="City location")):
    cursor.execute("SELECT COUNT(*) FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T2.locationcity = ?", (locationcity,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest paid employee in a specific city
@app.get("/v1/human_resources/highest_paid_employee_by_city", operation_id="get_highest_paid_employee_by_city", summary="Retrieves the first name and last name of the employee with the highest salary in the specified city. The city is provided as an input parameter.")
async def get_highest_paid_employee_by_city(locationcity: str = Query(..., description="City location")):
    cursor.execute("SELECT T1.firstname, T1.lastname FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T2.locationcity = ? ORDER BY T1.salary DESC LIMIT 1", (locationcity,))
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"firstname": result[0], "lastname": result[1]}}

# Endpoint to get the details of employees in a specific city with a specific performance rating
@app.get("/v1/human_resources/employee_details_by_city_and_performance", operation_id="get_employee_details_by_city_and_performance", summary="Retrieves the first name, last name, and social security number of employees located in a specified city who have a particular performance rating. The city and performance rating are provided as input parameters.")
async def get_employee_details_by_city_and_performance(locationcity: str = Query(..., description="City location"), performance: str = Query(..., description="Performance rating of the employee")):
    cursor.execute("SELECT T1.firstname, T1.lastname, T1.ssn FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T2.locationcity = ? AND T1.performance = ?", (locationcity, performance))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"firstname": row[0], "lastname": row[1], "ssn": row[2]} for row in result]}

# Endpoint to get the count of employees with a specific position title, city location, and performance rating
@app.get("/v1/human_resources/employee_count_by_position_city_performance", operation_id="get_employee_count_by_position_city_performance", summary="Retrieves the total number of employees who hold a specific job title, work in a particular city, and have a certain performance rating. This operation is useful for understanding the distribution of employees based on their roles, locations, and performance levels.")
async def get_employee_count_by_position_city_performance(positiontitle: str = Query(..., description="Position title"), locationcity: str = Query(..., description="City location"), performance: str = Query(..., description="Performance rating of the employee")):
    cursor.execute("SELECT COUNT(*) FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID INNER JOIN position AS T3 ON T3.positionID = T1.positionID WHERE T3.positiontitle = ? AND T2.locationcity = ? AND T1.performance = ?", (positiontitle, locationcity, performance))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the position title of an employee based on their first and last name
@app.get("/v1/human_resources/employee_position_title", operation_id="get_employee_position_title", summary="Retrieves the position title of a specific employee by matching their first and last names in the database. The operation requires the employee's first and last names as input parameters to accurately locate the corresponding position title.")
async def get_employee_position_title(firstname: str = Query(..., description="First name of the employee"), lastname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.positiontitle FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.firstname = ? AND T1.lastname = ?", (firstname, lastname))
    result = cursor.fetchone()
    if not result:
        return {"position_title": []}
    return {"position_title": result[0]}

# Endpoint to get the address of the highest paid employee with a specific position title
@app.get("/v1/human_resources/highest_paid_employee_address_by_position", operation_id="get_highest_paid_employee_address_by_position", summary="Retrieves the address of the employee with the highest salary for a given position title. The position title is used to filter the employee data, and the result is ordered by salary in descending order to ensure the highest paid employee is selected.")
async def get_highest_paid_employee_address_by_position(positiontitle: str = Query(..., description="Position title")):
    cursor.execute("SELECT T2.address FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID INNER JOIN position AS T3 ON T3.positionID = T1.positionID WHERE T3.positiontitle = ? ORDER BY T1.salary DESC LIMIT 1", (positiontitle,))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get the maximum salary for an employee's position based on their first and last name
@app.get("/v1/human_resources/employee_max_salary", operation_id="get_employee_max_salary", summary="Retrieves the maximum salary for a specific employee's position, based on their first and last name. This operation requires the employee's first and last name as input parameters to identify the position and determine the corresponding maximum salary.")
async def get_employee_max_salary(firstname: str = Query(..., description="First name of the employee"), lastname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.maxsalary FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.firstname = ? AND T1.lastname = ?", (firstname, lastname))
    result = cursor.fetchone()
    if not result:
        return {"max_salary": []}
    return {"max_salary": result[0]}

# Endpoint to get the percentage salary difference for a specific employee
@app.get("/v1/human_resources/salary_difference_percentage", operation_id="get_salary_difference_percentage", summary="Retrieves the percentage difference between an employee's salary and the maximum salary for their position, based on the provided first and last name.")
async def get_salary_difference_percentage(firstname: str = Query(..., description="First name of the employee"), lastname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT 100 * (CAST(REPLACE(SUBSTR(T2.maxsalary, 4), ',', '') AS REAL) - CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL)) / CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL) AS per FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.firstname = ? AND T1.lastname = ?", (firstname, lastname))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of employees with a specific performance rating and salary above a certain amount
@app.get("/v1/human_resources/employee_count_performance_salary", operation_id="get_employee_count_performance_salary", summary="Retrieves the number of employees who have a specified performance rating and earn a salary greater than a given amount. The performance rating is a measure of an employee's work quality and productivity, while the salary amount is the minimum threshold for consideration.")
async def get_employee_count_performance_salary(performance: str = Query(..., description="Performance rating of the employee"), salary: int = Query(..., description="Minimum salary amount")):
    cursor.execute("SELECT COUNT(*) FROM employee WHERE performance = ? AND CAST(REPLACE(SUBSTR(salary, 4), ',', '') AS REAL) > ?", (performance, salary))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the employee with the highest salary
@app.get("/v1/human_resources/highest_salary_employee", operation_id="get_highest_salary_employee", summary="Retrieves the employee with the highest salary from the human resources database. The operation returns the first and last name of the employee with the highest salary, calculated after removing any commas and converting the salary to a real number.")
async def get_highest_salary_employee():
    cursor.execute("SELECT firstname, lastname FROM employee WHERE CAST(REPLACE(SUBSTR(salary, 4), ',', '') AS REAL) = ( SELECT MAX(CAST(REPLACE(SUBSTR(salary, 4), ',', '') AS REAL)) FROM employee )")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"firstname": result[0], "lastname": result[1]}}

# Endpoint to get the count of positions with a maximum salary below a certain amount
@app.get("/v1/human_resources/position_count_max_salary", operation_id="get_position_count_max_salary", summary="Retrieves the total number of positions with a maximum salary less than the specified amount. The input parameter determines the maximum salary threshold for the count.")
async def get_position_count_max_salary(max_salary: int = Query(..., description="Maximum salary amount")):
    cursor.execute("SELECT COUNT(*) FROM position WHERE CAST(REPLACE(SUBSTR(maxsalary, 4), ',', '') AS REAL) < ?", (max_salary,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the salary of the earliest hired employee
@app.get("/v1/human_resources/earliest_hired_employee_salary", operation_id="get_earliest_hired_employee_salary", summary="Retrieves the salary of the employee who was hired first. This operation returns the salary of the employee with the earliest hire date in the human resources database.")
async def get_earliest_hired_employee_salary():
    cursor.execute("SELECT salary FROM employee ORDER BY hiredate ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"salary": []}
    return {"salary": result[0]}

# Endpoint to get the minimum salary of the position with the highest education requirement
@app.get("/v1/human_resources/min_salary_highest_education", operation_id="get_min_salary_highest_education", summary="Retrieves the lowest salary associated with the position that demands the highest level of education. The response provides a single value representing the minimum salary for the most education-intensive position.")
async def get_min_salary_highest_education():
    cursor.execute("SELECT minsalary FROM position ORDER BY educationrequired DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"minsalary": []}
    return {"minsalary": result[0]}

# Endpoint to get the most common employee location
@app.get("/v1/human_resources/most_common_employee_location", operation_id="get_most_common_employee_location", summary="Retrieves the most frequently occurring location of employees, based on the address, city, state, and zip code. The data is obtained by aggregating employee records and their associated location details, then sorting the results in descending order by the count of occurrences. The top result is returned.")
async def get_most_common_employee_location():
    cursor.execute("SELECT T2.address, T2.locationcity, T2.state, T2.zipcode FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID GROUP BY T2.address, T2.locationcity, T2.state, T2.zipcode ORDER BY COUNT(*) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": {"address": result[0], "city": result[1], "state": result[2], "zipcode": result[3]}}

# Endpoint to get the average salary for employees with a specific education requirement
@app.get("/v1/human_resources/average_salary_education", operation_id="get_average_salary_education", summary="Retrieves the average salary of employees who hold a specific education requirement. The education requirement is provided as an input parameter.")
async def get_average_salary_education(education_required: str = Query(..., description="Education requirement")):
    cursor.execute("SELECT AVG(CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL)) FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T2.educationrequired = ?", (education_required,))
    result = cursor.fetchone()
    if not result:
        return {"average_salary": []}
    return {"average_salary": result[0]}

# Endpoint to get the count of employees with a specific position title and gender
@app.get("/v1/human_resources/employee_count_position_gender", operation_id="get_employee_count_position_gender", summary="Retrieves the total number of employees with a specified position title and gender. The response is based on the provided position title and gender, which are used to filter the employee records.")
async def get_employee_count_position_gender(position_title: str = Query(..., description="Position title"), gender: str = Query(..., description="Gender of the employee")):
    cursor.execute("SELECT COUNT(*) FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T2.positiontitle = ? AND T1.gender = ?", (position_title, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common position title for employees with a specific performance rating
@app.get("/v1/human_resources/most_common_position_performance", operation_id="get_most_common_position_performance", summary="Retrieves the position title that is most frequently held by employees with a specified performance rating. The performance rating is used to filter the employees and determine the most common position title among them.")
async def get_most_common_position_performance(performance: str = Query(..., description="Performance rating of the employee")):
    cursor.execute("SELECT T2.positiontitle FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.performance = ? GROUP BY T2.positiontitle ORDER BY COUNT(T2.positiontitle) DESC LIMIT 1", (performance,))
    result = cursor.fetchone()
    if not result:
        return {"position_title": []}
    return {"position_title": result[0]}

# Endpoint to get the most common position title for employees with a specific education requirement and gender
@app.get("/v1/human_resources/most_common_position_title", operation_id="get_most_common_position_title", summary="Retrieves the most frequently occurring position title for employees who meet a specified education requirement and gender. The education requirement and gender are provided as input parameters.")
async def get_most_common_position_title(education_required: str = Query(..., description="Education requirement for the position"), gender: str = Query(..., description="Gender of the employee")):
    cursor.execute("SELECT T2.positiontitle FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T2.educationrequired = ? AND T1.gender = ? GROUP BY T2.positiontitle ORDER BY COUNT(T2.positiontitle) DESC LIMIT 1", (education_required, gender))
    result = cursor.fetchone()
    if not result:
        return {"position_title": []}
    return {"position_title": result[0]}

# Endpoint to get the count of employees with a specific position title, performance rating, and state
@app.get("/v1/human_resources/employee_count_by_position_performance_state", operation_id="get_employee_count", summary="Retrieves the total number of employees who hold a specific job title, have a certain performance rating, and are located in a particular state. The response is based on the provided position title, performance rating, and state.")
async def get_employee_count(position_title: str = Query(..., description="Position title of the employee"), performance: str = Query(..., description="Performance rating of the employee"), state: str = Query(..., description="State of the employee's location")):
    cursor.execute("SELECT COUNT(*) FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID INNER JOIN position AS T3 ON T3.positionID = T1.positionID WHERE T3.positiontitle = ? AND T1.performance = ? AND T2.state = ?", (position_title, performance, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average salary of employees with a specific performance rating and position title
@app.get("/v1/human_resources/average_salary_by_performance_position", operation_id="get_average_salary", summary="Retrieves the average salary of employees who have a specified performance rating and hold a particular position. The performance rating and position title are used to filter the employees and calculate the average salary.")
async def get_average_salary(performance: str = Query(..., description="Performance rating of the employee"), position_title: str = Query(..., description="Position title of the employee")):
    cursor.execute("SELECT AVG(CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL)) FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.performance = ? AND T2.positiontitle = ?", (performance, position_title))
    result = cursor.fetchone()
    if not result:
        return {"average_salary": []}
    return {"average_salary": result[0]}

# Endpoint to get the state with the most employees for a specific position title and performance rating
@app.get("/v1/human_resources/most_common_state_by_position_performance", operation_id="get_most_common_state", summary="Retrieves the state with the highest number of employees in a specific role and performance level. The operation considers the position title and performance rating as input parameters to filter the employee data and determine the most common state.")
async def get_most_common_state(position_title: str = Query(..., description="Position title of the employee"), performance: str = Query(..., description="Performance rating of the employee")):
    cursor.execute("SELECT T2.state FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID INNER JOIN position AS T3 ON T3.positionID = T1.positionID WHERE T3.positiontitle = ? AND T1.performance = ? GROUP BY T2.state ORDER BY COUNT(T2.state) DESC LIMIT 1", (position_title, performance))
    result = cursor.fetchone()
    if not result:
        return {"state": []}
    return {"state": result[0]}

# Endpoint to get the employee with the lowest salary
@app.get("/v1/human_resources/lowest_salary_employee", operation_id="get_lowest_salary_employee", summary="Retrieves the employee with the lowest salary, along with their performance rating. The response includes the first name, last name, and performance score of the employee with the lowest salary in the database.")
async def get_lowest_salary_employee():
    cursor.execute("SELECT firstname, lastname, performance FROM employee ORDER BY salary ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"firstname": result[0], "lastname": result[1], "performance": result[2]}}

# Endpoint to get cities in specific states
@app.get("/v1/human_resources/cities_in_states", operation_id="get_cities_in_states", summary="Retrieves a list of cities located in the provided states. The operation accepts up to three state names as input parameters and returns the corresponding city names from the database.")
async def get_cities_in_states(state1: str = Query(..., description="First state"), state2: str = Query(..., description="Second state"), state3: str = Query(..., description="Third state")):
    cursor.execute("SELECT locationcity FROM location WHERE state IN (?, ?, ?)", (state1, state2, state3))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get cities and addresses with zip codes greater than a specified value
@app.get("/v1/human_resources/cities_addresses_by_zipcode", operation_id="get_cities_addresses_by_zipcode", summary="Retrieves a list of cities and their corresponding addresses where the zip code is greater than the provided minimum value. This operation allows you to filter locations based on a specific zip code threshold.")
async def get_cities_addresses_by_zipcode(min_zipcode: int = Query(..., description="Minimum zip code value")):
    cursor.execute("SELECT locationcity, address FROM location WHERE zipcode > ?", (min_zipcode,))
    result = cursor.fetchall()
    if not result:
        return {"cities_addresses": []}
    return {"cities_addresses": [{"city": row[0], "address": row[1]} for row in result]}

# Endpoint to get position titles with a specific education requirement
@app.get("/v1/human_resources/position_titles_by_education", operation_id="get_position_titles_by_education", summary="Retrieves a list of position titles that require a specific level of education. The education requirement is provided as an input parameter, allowing the user to filter the results accordingly.")
async def get_position_titles_by_education(education_required: str = Query(..., description="Education requirement for the position")):
    cursor.execute("SELECT positiontitle FROM position WHERE educationrequired = ?", (education_required,))
    result = cursor.fetchall()
    if not result:
        return {"position_titles": []}
    return {"position_titles": [row[0] for row in result]}

# Endpoint to get the maximum salary for a specific position title
@app.get("/v1/human_resources/max_salary_by_position", operation_id="get_max_salary_by_position", summary="Retrieves the highest salary associated with a given position title in the human resources department.")
async def get_max_salary_by_position(position_title: str = Query(..., description="Position title of the employee")):
    cursor.execute("SELECT maxsalary FROM position WHERE positiontitle = ?", (position_title,))
    result = cursor.fetchone()
    if not result:
        return {"max_salary": []}
    return {"max_salary": result[0]}

# Endpoint to get employee details with a specific performance rating
@app.get("/v1/human_resources/employee_details_by_performance", operation_id="get_employee_details_by_performance", summary="Retrieves the first name, last name, and social security number of employees who have a specified performance rating. The performance rating is used to filter the results.")
async def get_employee_details_by_performance(performance: str = Query(..., description="Performance rating of the employee")):
    cursor.execute("SELECT T1.firstname, T1.lastname, T1.ssn FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.performance = ?", (performance,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"firstname": row[0], "lastname": row[1], "ssn": row[2]} for row in result]}

# Endpoint to get employee details based on first name and last name
@app.get("/v1/human_resources/employee_details_by_name", operation_id="get_employee_details_by_name", summary="Retrieves the hire date, position title, and salary of an employee using their first and last names. The operation returns the specified details for the employee whose first and last names match the provided input parameters.")
async def get_employee_details_by_name(firstname: str = Query(..., description="First name of the employee"), lastname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T1.hiredate, T2.positiontitle, T1.salary FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.firstname = ? AND T1.lastname = ?", (firstname, lastname))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get position details based on employee first name and last name
@app.get("/v1/human_resources/position_details_by_employee_name", operation_id="get_position_details_by_employee_name", summary="Retrieves the maximum and minimum salary, as well as the position title, for an employee identified by their first and last names. This operation provides a comprehensive view of the employee's position and salary range within the organization.")
async def get_position_details_by_employee_name(firstname: str = Query(..., description="First name of the employee"), lastname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.maxsalary, T2.minsalary, T2.positiontitle FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.firstname = ? AND T1.lastname = ?", (firstname, lastname))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get employee details based on location city
@app.get("/v1/human_resources/employee_details_by_location_city", operation_id="get_employee_details_by_location_city", summary="Retrieves the first name, last name, gender, and position title of employees who work in the specified city. The city is provided as an input parameter.")
async def get_employee_details_by_location_city(locationcity: str = Query(..., description="Location city of the employee")):
    cursor.execute("SELECT T1.firstname, T1.lastname, T1.gender, T3.positiontitle FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID INNER JOIN position AS T3 ON T3.positionID = T1.positionID WHERE T2.locationcity = ?", (locationcity,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get employee details based on state
@app.get("/v1/human_resources/employee_details_by_state", operation_id="get_employee_details_by_state", summary="Retrieves the first name, last name, hire date, and performance of employees located in the specified state. The state parameter is used to filter the results.")
async def get_employee_details_by_state(state: str = Query(..., description="State of the employee")):
    cursor.execute("SELECT T1.firstname, T1.lastname, T1.hiredate, T1.performance FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T2.state = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get employee details based on position title and performance
@app.get("/v1/human_resources/employee_details_by_position_performance", operation_id="get_employee_details_by_position_performance", summary="Retrieves the personal and location details of employees who hold a specific position and have a certain performance level. The response includes the first name, last name, city, address, and zip code of the employees.")
async def get_employee_details_by_position_performance(positiontitle: str = Query(..., description="Position title of the employee"), performance: str = Query(..., description="Performance of the employee")):
    cursor.execute("SELECT T1.firstname, T1.lastname, T2.locationcity, T2.address, T2.zipcode FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID INNER JOIN position AS T3 ON T3.positionID = T1.positionID WHERE T3.positiontitle = ? AND T1.performance = ?", (positiontitle, performance))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get employee details based on performance and position title
@app.get("/v1/human_resources/employee_details_by_performance_position", operation_id="get_employee_details_by_performance_position", summary="Retrieves the education requirement, first name, last name, and salary of employees who match the specified performance level and position title. The performance level and position title are used as filters to narrow down the results.")
async def get_employee_details_by_performance_position(performance: str = Query(..., description="Performance of the employee"), positiontitle: str = Query(..., description="Position title of the employee")):
    cursor.execute("SELECT T2.educationrequired, T1.firstname, T1.lastname, T1.salary FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID WHERE T1.performance = ? AND T2.positiontitle = ?", (performance, positiontitle))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get employee details based on SSN
@app.get("/v1/human_resources/employee_details_by_ssn", operation_id="get_employee_details_by_ssn", summary="Retrieves the first name, last name, state, and city of an employee's location based on the provided Social Security Number (SSN). The SSN is used to identify the employee and their associated location details.")
async def get_employee_details_by_ssn(ssn: str = Query(..., description="SSN of the employee")):
    cursor.execute("SELECT T1.firstname, T1.lastname, T2.state, T2.locationcity FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID WHERE T1.ssn = ?", (ssn,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get employee details based on state
@app.get("/v1/human_resources/employee_details_by_state_with_office_phone", operation_id="get_employee_details_by_state_with_office_phone", summary="Retrieves the first name, last name, position title, location city, and office phone of employees who are based in the specified state. This operation provides a comprehensive view of employee details, including their job title and contact information, enabling efficient management and communication.")
async def get_employee_details_by_state_with_office_phone(state: str = Query(..., description="State of the employee")):
    cursor.execute("SELECT T1.firstname, T1.lastname, T3.positiontitle, T2.locationcity, T2.officephone FROM employee AS T1 INNER JOIN location AS T2 ON T1.locationID = T2.locationID INNER JOIN position AS T3 ON T3.positionID = T1.positionID WHERE T2.state = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the highest average monthly salary details
@app.get("/v1/human_resources/highest_avg_monthly_salary_details", operation_id="get_highest_avg_monthly_salary_details", summary="Retrieves the details of the employee with the highest average monthly salary. The response includes the employee's first name, last name, position title, and location city.")
async def get_highest_avg_monthly_salary_details():
    cursor.execute("SELECT SUM(CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL)) / 12 AS avg, T1.firstname, T1.lastname, T2.positiontitle, T3.locationcity FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID INNER JOIN location AS T3 ON T1.locationID = T3.locationID WHERE CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL) = (SELECT MAX(CAST(REPLACE(SUBSTR(T1.salary, 4), ',', '') AS REAL)) FROM employee AS T1 INNER JOIN position AS T2 ON T1.positionID = T2.positionID INNER JOIN location AS T3 ON T1.locationID = T3.locationID)")
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

api_calls = [
    "/v1/human_resources/top_employee_by_salary",
    "/v1/human_resources/employee_count_by_performance?performance=Good",
    "/v1/human_resources/employee_ssns_by_gender_and_salary?gender=M&min_salary=70000",
    "/v1/human_resources/education_required_by_position?position_title=Regional%20Manager",
    "/v1/human_resources/min_salary_position_title?position_title1=Account%20Representative&position_title2=Trainee",
    "/v1/human_resources/employee_city_by_name?lastname=Adams&firstname=Sandy",
    "/v1/human_resources/employee_count_by_state_and_performance?state=NY&performance=Good",
    "/v1/human_resources/employee_office_phone_by_name?lastname=Adams&firstname=Sandy",
    "/v1/human_resources/employee_count_by_address_and_gender?address=450%20Peachtree%20Rd&gender=M",
    "/v1/human_resources/employee_count_by_position_title?position_title=Account%20Representative",
    "/v1/human_resources/salary_difference?lastname=Johnson&firstname=James",
    "/v1/human_resources/employee_count_position_state?positiontitle=Trainee&state=NY",
    "/v1/human_resources/employee_names_by_position?positiontitle=Trainee",
    "/v1/human_resources/employee_names_by_name_pairs?lastname1=Adams&firstname1=Sandy&lastname2=Rodriguez&firstname2=Jose",
    "/v1/human_resources/employee_zipcodes_by_gender_performance?gender=M&performance=Good",
    "/v1/human_resources/employee_ssns_by_state?state=CA",
    "/v1/human_resources/employee_count_salary_position?min_salary=20000&positiontitle=Trainee",
    "/v1/human_resources/average_salary_by_position?positiontitle=Trainee",
    "/v1/human_resources/percentage_salary_difference?positiontitle=Trainee",
    "/v1/human_resources/employee_count_by_gender?gender=F",
    "/v1/human_resources/employee_location_city?firstname=Jose&lastname=Rodriguez",
    "/v1/human_resources/employee_location_state?firstname=Emily&lastname=Wood",
    "/v1/human_resources/employee_education_required?firstname=David&lastname=Whitehead&gender=M",
    "/v1/human_resources/employee_count_by_city?locationcity=Miami",
    "/v1/human_resources/highest_paid_employee_by_city?locationcity=Boston",
    "/v1/human_resources/employee_details_by_city_and_performance?locationcity=New%20York%20City&performance=Good",
    "/v1/human_resources/employee_count_by_position_city_performance?positiontitle=Account%20Representative&locationcity=Chicago&performance=Good",
    "/v1/human_resources/employee_position_title?firstname=Kenneth&lastname=Charles",
    "/v1/human_resources/highest_paid_employee_address_by_position?positiontitle=Manager",
    "/v1/human_resources/employee_max_salary?firstname=Tracy&lastname=Coulter",
    "/v1/human_resources/salary_difference_percentage?firstname=Jose&lastname=Rodriguez",
    "/v1/human_resources/employee_count_performance_salary?performance=Poor&salary=50000",
    "/v1/human_resources/highest_salary_employee",
    "/v1/human_resources/position_count_max_salary?max_salary=100000",
    "/v1/human_resources/earliest_hired_employee_salary",
    "/v1/human_resources/min_salary_highest_education",
    "/v1/human_resources/most_common_employee_location",
    "/v1/human_resources/average_salary_education?education_required=2%20year%20degree",
    "/v1/human_resources/employee_count_position_gender?position_title=Regional%20Manager&gender=M",
    "/v1/human_resources/most_common_position_performance?performance=Poor",
    "/v1/human_resources/most_common_position_title?education_required=2%20year%20degree&gender=F",
    "/v1/human_resources/employee_count_by_position_performance_state?position_title=Account%20Representative&performance=Good&state=IL",
    "/v1/human_resources/average_salary_by_performance_position?performance=Poor&position_title=Manager",
    "/v1/human_resources/most_common_state_by_position_performance?position_title=Account%20Representative&performance=Good",
    "/v1/human_resources/lowest_salary_employee",
    "/v1/human_resources/cities_in_states?state1=CO&state2=UT&state3=CA",
    "/v1/human_resources/cities_addresses_by_zipcode?min_zipcode=90000",
    "/v1/human_resources/position_titles_by_education?education_required=4%20year%20degree",
    "/v1/human_resources/max_salary_by_position?position_title=Trainee",
    "/v1/human_resources/employee_details_by_performance?performance=Average",
    "/v1/human_resources/employee_details_by_name?firstname=Emily&lastname=Wood",
    "/v1/human_resources/position_details_by_employee_name?firstname=Bill&lastname=Marlin",
    "/v1/human_resources/employee_details_by_location_city?locationcity=New%20York%20City",
    "/v1/human_resources/employee_details_by_state?state=UT",
    "/v1/human_resources/employee_details_by_position_performance?positiontitle=Manager&performance=Poor",
    "/v1/human_resources/employee_details_by_performance_position?performance=Poor&positiontitle=Account%20Representative",
    "/v1/human_resources/employee_details_by_ssn?ssn=767-74-7373",
    "/v1/human_resources/employee_details_by_state_with_office_phone?state=CO",
    "/v1/human_resources/highest_avg_monthly_salary_details"
]
