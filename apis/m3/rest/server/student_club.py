from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/student_club/student_club.sqlite')
cursor = conn.cursor()

# Endpoint to get the major name of a member based on their first and last name
@app.get("/v1/student_club/major_name_by_member_name", operation_id="get_major_name_by_member_name", summary="Retrieves the major name associated with a specific member using their first and last name. This operation requires the first and last name of the member as input parameters to identify the corresponding major name from the database.")
async def get_major_name_by_member_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.major_name FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"major_names": []}
    return {"major_names": [row[0] for row in result]}

# Endpoint to get the count of members in a specific college
@app.get("/v1/student_club/member_count_by_college", operation_id="get_member_count_by_college", summary="Retrieves the total number of members belonging to a specific college. The operation calculates the count based on the college name provided as an input parameter.")
async def get_member_count_by_college(college: str = Query(..., description="Name of the college")):
    cursor.execute("SELECT COUNT(T1.member_id) FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T2.college = ?", (college,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of members in a specific department
@app.get("/v1/student_club/member_names_by_department", operation_id="get_member_names_by_department", summary="Retrieves the first and last names of members who are majoring in a specified department. The department name is provided as an input parameter.")
async def get_member_names_by_department(department: str = Query(..., description="Name of the department")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T2.department = ?", (department,))
    result = cursor.fetchall()
    if not result:
        return {"member_names": []}
    return {"member_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of events with a specific name
@app.get("/v1/student_club/event_count_by_name", operation_id="get_event_count_by_name", summary="Retrieves the total number of occurrences of a specific event, based on the provided event name. This operation considers the event's attendance records to determine the count.")
async def get_event_count_by_name(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT COUNT(T1.event_id) FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_name = ?", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone numbers of members who attended a specific event
@app.get("/v1/student_club/member_phones_by_event_name", operation_id="get_member_phones_by_event_name", summary="Retrieves the phone numbers of members who attended a particular event. The event is identified by its unique name, which is provided as an input parameter.")
async def get_member_phones_by_event_name(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT T3.phone FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event INNER JOIN member AS T3 ON T2.link_to_member = T3.member_id WHERE T1.event_name = ?", (event_name,))
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": [row[0] for row in result]}

# Endpoint to get the count of events attended by members with a specific t-shirt size at a specific event
@app.get("/v1/student_club/event_count_by_name_and_tshirt_size", operation_id="get_event_count_by_name_and_tshirt_size", summary="Retrieves the total number of members who attended a specific event and have a particular t-shirt size. The response is based on the event name and the t-shirt size provided as input parameters.")
async def get_event_count_by_name_and_tshirt_size(event_name: str = Query(..., description="Name of the event"), t_shirt_size: str = Query(..., description="T-shirt size of the member")):
    cursor.execute("SELECT COUNT(T1.event_id) FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event INNER JOIN member AS T3 ON T2.link_to_member = T3.member_id WHERE T1.event_name = ? AND T3.t_shirt_size = ?", (event_name, t_shirt_size))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most attended event
@app.get("/v1/student_club/most_attended_event", operation_id="get_most_attended_event", summary="Retrieves the event with the highest attendance count from the student club's event records. The operation returns the name of the most attended event, which is determined by aggregating and ordering attendance records.")
async def get_most_attended_event():
    cursor.execute("SELECT T1.event_name FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event GROUP BY T1.event_name ORDER BY COUNT(T2.link_to_event) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"event_name": []}
    return {"event_name": result[0]}

# Endpoint to get the colleges of members with a specific position
@app.get("/v1/student_club/colleges_by_position", operation_id="get_colleges_by_position", summary="Retrieve the colleges of members who hold a specific position. The position can be specified using a wildcard character for partial matches. The response includes a list of colleges associated with the members who meet the position criteria.")
async def get_colleges_by_position(position: str = Query(..., description="Position of the member (use %% for wildcard)")):
    cursor.execute("SELECT T2.college FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T1.position LIKE ?", (position,))
    result = cursor.fetchall()
    if not result:
        return {"colleges": []}
    return {"colleges": [row[0] for row in result]}

# Endpoint to get the event names attended by a specific member
@app.get("/v1/student_club/event_names_by_member_name", operation_id="get_event_names_by_member_name", summary="Retrieves the names of events attended by a specific member, identified by their first and last names. The operation returns a list of event names that the member has attended.")
async def get_event_names_by_member_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T1.event_name FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event INNER JOIN member AS T3 ON T2.link_to_member = T3.member_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"event_names": []}
    return {"event_names": [row[0] for row in result]}

# Endpoint to get the count of events attended by a specific member in a specific year
@app.get("/v1/student_club/event_count_by_member_name_and_year", operation_id="get_event_count_by_member_name_and_year", summary="Retrieves the total number of events attended by a specific member in a given year. The operation requires the first and last name of the member, as well as the year in 'YYYY' format. The result is a count of events that the specified member attended during the provided year.")
async def get_event_count_by_member_name_and_year(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.event_id) FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event INNER JOIN member AS T3 ON T2.link_to_member = T3.member_id WHERE T3.first_name = ? AND T3.last_name = ? AND SUBSTR(T1.event_date, 1, 4) = ?", (first_name, last_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get event names with attendance count greater than a specified number and excluding events of a specific type
@app.get("/v1/student_club/event_names_attendance_count_exclude_type", operation_id="get_event_names_attendance_count_exclude_type", summary="Retrieves the names of events with an attendance count exceeding the specified minimum, excluding events of a particular type. The response includes event names that meet the criteria, providing a filtered list of events based on attendance and type.")
async def get_event_names_attendance_count_exclude_type(attendance_count: int = Query(..., description="Minimum attendance count"), event_type: str = Query(..., description="Event type to exclude")):
    cursor.execute("SELECT T1.event_name FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event GROUP BY T1.event_id HAVING COUNT(T2.link_to_event) > ? EXCEPT SELECT T1.event_name FROM event AS T1 WHERE T1.type = ?", (attendance_count, event_type))
    result = cursor.fetchall()
    if not result:
        return {"event_names": []}
    return {"event_names": [row[0] for row in result]}

# Endpoint to get the ratio of total attendance to unique events for a specific year and event type
@app.get("/v1/student_club/attendance_ratio_year_type", operation_id="get_attendance_ratio_year_type", summary="Retrieves the proportion of total attendances to unique events for a given year and event type. This operation calculates the ratio by dividing the total count of attendances by the count of distinct events that occurred in the specified year and match the provided event type.")
async def get_attendance_ratio_year_type(year: str = Query(..., description="Year in 'YYYY' format"), event_type: str = Query(..., description="Event type")):
    cursor.execute("SELECT CAST(COUNT(T2.link_to_event) AS REAL) / COUNT(DISTINCT T2.link_to_event) FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event WHERE SUBSTR(T1.event_date, 1, 4) = ? AND T1.type = ?", (year, event_type))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the most expensive expense description
@app.get("/v1/student_club/most_expensive_expense", operation_id="get_most_expensive_expense", summary="Retrieves the description of the most expensive expense(s) in the expense table, up to the specified limit. The operation returns the description of the top expense(s) with the highest cost, providing insight into the most significant expenses.")
async def get_most_expensive_expense(limit: int = Query(..., description="Number of top expenses to return")):
    cursor.execute("SELECT expense_description FROM expense ORDER BY cost DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"expense_descriptions": []}
    return {"expense_descriptions": [row[0] for row in result]}

# Endpoint to get the count of members in a specific major
@app.get("/v1/student_club/member_count_by_major", operation_id="get_member_count_by_major", summary="Retrieves the total number of members belonging to a specific major. The major is identified by its name, which is provided as an input parameter.")
async def get_member_count_by_major(major_name: str = Query(..., description="Name of the major")):
    cursor.execute("SELECT COUNT(T1.member_id) FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T2.major_name = ?", (major_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of members who attended a specific event
@app.get("/v1/student_club/member_names_by_event", operation_id="get_member_names_by_event", summary="Retrieves the first and last names of members who attended a specific event. The operation requires the name of the event as an input parameter to filter the members accordingly.")
async def get_member_names_by_event(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN attendance AS T2 ON T1.member_id = T2.link_to_member INNER JOIN event AS T3 ON T2.link_to_event = T3.event_id WHERE T3.event_name = ?", (event_name,))
    result = cursor.fetchall()
    if not result:
        return {"member_names": []}
    return {"member_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the last names of members in a specific major
@app.get("/v1/student_club/member_last_names_by_major", operation_id="get_member_last_names_by_major", summary="Retrieves the last names of student club members who are majoring in a specific discipline. The major is specified as an input parameter.")
async def get_member_last_names_by_major(major_name: str = Query(..., description="Name of the major")):
    cursor.execute("SELECT T1.last_name FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T2.major_name = ?", (major_name,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get the county of a member by their first and last name
@app.get("/v1/student_club/member_county_by_name", operation_id="get_member_county_by_name", summary="Retrieves the county of a specific member based on their first and last name. The operation uses the provided first and last names to search for a matching member and returns the county associated with their zip code.")
async def get_member_county_by_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.county FROM member AS T1 INNER JOIN zip_code AS T2 ON T1.zip = T2.zip_code WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get the college of a member by their first and last name
@app.get("/v1/student_club/member_college_by_name", operation_id="get_member_college_by_name", summary="Retrieves the college of a specific member based on their first and last name. The operation uses the provided first and last name to search for a match in the member database and returns the associated college.")
async def get_member_college_by_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.college FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"college": []}
    return {"college": result[0]}

# Endpoint to get the income amount of members by their position
@app.get("/v1/student_club/member_income_by_position", operation_id="get_member_income_by_position", summary="Retrieves the total income amount for members holding a specific position. The position is provided as an input parameter.")
async def get_member_income_by_position(position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT T2.amount FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T1.position = ?", (position,))
    result = cursor.fetchall()
    if not result:
        return {"income_amounts": []}
    return {"income_amounts": [row[0] for row in result]}

# Endpoint to get the amount spent on a specific category for a specific event in a specific month
@app.get("/v1/student_club/event_spent_category_month", operation_id="get_event_spent_category_month", summary="Retrieves the total amount spent on a specified budget category for a particular event that occurred in a given month. The event is identified by its name, the category by its label, and the month by its two-digit numerical representation.")
async def get_event_spent_category_month(event_name: str = Query(..., description="Name of the event"), category: str = Query(..., description="Category of the budget"), month: str = Query(..., description="Month of the event in MM format")):
    cursor.execute("SELECT T2.spent FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_name = ? AND T2.category = ? AND SUBSTR(T1.event_date, 6, 2) = ?", (event_name, category, month))
    result = cursor.fetchone()
    if not result:
        return {"spent": []}
    return {"spent": result[0]}

# Endpoint to get the city and state of members based on their position
@app.get("/v1/student_club/member_location_by_position", operation_id="get_member_location_by_position", summary="Retrieves the city and state of members who hold a specific position. The position is provided as an input parameter.")
async def get_member_location_by_position(position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT T2.city, T2.state FROM member AS T1 INNER JOIN zip_code AS T2 ON T1.zip = T2.zip_code WHERE T1.position = ?", (position,))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": result}

# Endpoint to get the names of members based on their state
@app.get("/v1/student_club/member_names_by_state", operation_id="get_member_names_by_state", summary="Retrieves the first and last names of members who reside in the specified state. The state is determined by the member's zip code.")
async def get_member_names_by_state(state: str = Query(..., description="State of the member")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN zip_code AS T2 ON T1.zip = T2.zip_code WHERE T2.state = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the departments of members based on their last names
@app.get("/v1/student_club/member_departments_by_last_name", operation_id="get_member_departments_by_last_name", summary="Retrieves the departments of members whose last names match either of the two provided last names. This operation allows for querying the departments of members based on their last names, providing a convenient way to filter and view departmental affiliations.")
async def get_member_departments_by_last_name(last_name1: str = Query(..., description="First last name of the member"), last_name2: str = Query(..., description="Second last name of the member")):
    cursor.execute("SELECT T2.department FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T1.last_name = ? OR T1.last_name = ?", (last_name1, last_name2))
    result = cursor.fetchall()
    if not result:
        return {"departments": []}
    return {"departments": result}

# Endpoint to get the total amount spent for a specific event
@app.get("/v1/student_club/total_spent_event", operation_id="get_total_spent_event", summary="Retrieves the total amount of money spent on a specific event. The calculation is based on the sum of all budget entries linked to the event. The event is identified by its name.")
async def get_total_spent_event(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT SUM(T2.amount) FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_name = ?", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_spent": []}
    return {"total_spent": result[0]}

# Endpoint to get the approval status of expenses for a specific event on a specific date
@app.get("/v1/student_club/expense_approval_status", operation_id="get_expense_approval_status", summary="Retrieves the approval status of expenses for a specific event on a given date. The event is identified by its name and date, and the status indicates whether the expenses associated with the event's budget have been approved.")
async def get_expense_approval_status(event_name: str = Query(..., description="Name of the event"), event_date: str = Query(..., description="Date of the event in YYYY-MM-DD format")):
    cursor.execute("SELECT T3.approved FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget WHERE T1.event_name = ? AND T1.event_date LIKE ?", (event_name, event_date + '%'))
    result = cursor.fetchall()
    if not result:
        return {"approval_status": []}
    return {"approval_status": result}

# Endpoint to get the average cost of expenses for a specific member in specific months
@app.get("/v1/student_club/average_expense_cost_member_months", operation_id="get_average_expense_cost_member_months", summary="Retrieves the average cost of expenses incurred by a specific member during two specified months. The member is identified by their first and last names. The months are provided in MM format.")
async def get_average_expense_cost_member_months(last_name: str = Query(..., description="Last name of the member"), first_name: str = Query(..., description="First name of the member"), month1: str = Query(..., description="First month in MM format"), month2: str = Query(..., description="Second month in MM format")):
    cursor.execute("SELECT AVG(T2.cost) FROM member AS T1 INNER JOIN expense AS T2 ON T1.member_id = T2.link_to_member WHERE T1.last_name = ? AND T1.first_name = ? AND (SUBSTR(T2.expense_date, 6, 2) = ? OR SUBSTR(T2.expense_date, 6, 2) = ?)", (last_name, first_name, month1, month2))
    result = cursor.fetchone()
    if not result:
        return {"average_cost": []}
    return {"average_cost": result[0]}

# Endpoint to get the difference in total spending between two years
@app.get("/v1/student_club/spending_difference_years", operation_id="get_spending_difference_years", summary="Retrieves the difference in total spending between two specified years for student club events. The operation calculates the sum of spending for each year, then subtracts the total for the second year from the first year. The result is returned as a single numeric value.")
async def get_spending_difference_years(year1: str = Query(..., description="First year in YYYY format"), year2: str = Query(..., description="Second year in YYYY format")):
    cursor.execute("SELECT SUM(CASE WHEN SUBSTR(T1.event_date, 1, 4) = ? THEN T2.spent ELSE 0 END) - SUM(CASE WHEN SUBSTR(T1.event_date, 1, 4) = ? THEN T2.spent ELSE 0 END) AS num FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event", (year1, year2))
    result = cursor.fetchone()
    if not result:
        return {"spending_difference": []}
    return {"spending_difference": result[0]}

# Endpoint to get the location of a specific event
@app.get("/v1/student_club/event_location", operation_id="get_event_location", summary="Retrieves the location of a specific event by its name. The event name is provided as an input parameter, and the location of the corresponding event is returned.")
async def get_event_location(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT location FROM event WHERE event_name = ?", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": result[0]}

# Endpoint to get the cost of an expense based on description and date
@app.get("/v1/student_club/expense_cost", operation_id="get_expense_cost", summary="Retrieves the cost of a specific expense based on its description and date. The expense description and date are used to identify the expense and return its associated cost. The date should be provided in 'YYYY-MM-DD' format.")
async def get_expense_cost(expense_description: str = Query(..., description="Description of the expense"), expense_date: str = Query(..., description="Date of the expense in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT cost FROM expense WHERE expense_description = ? AND expense_date = ?", (expense_description, expense_date))
    result = cursor.fetchone()
    if not result:
        return {"cost": []}
    return {"cost": result[0]}

# Endpoint to get the remaining budget for a category with the maximum amount
@app.get("/v1/student_club/remaining_budget_max_amount", operation_id="get_remaining_budget_max_amount", summary="Retrieves the remaining budget for the category with the highest budget amount. The category is specified as an input parameter.")
async def get_remaining_budget_max_amount(category: str = Query(..., description="Category of the budget")):
    cursor.execute("SELECT remaining FROM budget WHERE category = ? AND amount = ( SELECT MAX(amount) FROM budget WHERE category = ? )", (category, category))
    result = cursor.fetchone()
    if not result:
        return {"remaining": []}
    return {"remaining": result[0]}

# Endpoint to get notes from income based on source and date received
@app.get("/v1/student_club/income_notes", operation_id="get_income_notes", summary="Retrieves notes associated with a specific income source and date received. The source and date are used to filter the income records and return the corresponding notes.")
async def get_income_notes(source: str = Query(..., description="Source of the income"), date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT notes FROM income WHERE source = ? AND date_received = ?", (source, date_received))
    result = cursor.fetchone()
    if not result:
        return {"notes": []}
    return {"notes": result[0]}

# Endpoint to get the count of majors in a specific college
@app.get("/v1/student_club/major_count", operation_id="get_major_count", summary="Retrieves the total number of unique majors offered in a specified college. The operation requires the college name as input and returns the count of distinct majors associated with the college.")
async def get_major_count(college: str = Query(..., description="Name of the college")):
    cursor.execute("SELECT COUNT(major_name) FROM major WHERE college = ?", (college,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone number of a member based on first and last name
@app.get("/v1/student_club/member_phone", operation_id="get_member_phone", summary="Retrieves the phone number of a specific member in the student club, based on their first and last name.")
async def get_member_phone(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT phone FROM member WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the count of events with a remaining budget less than a specified amount
@app.get("/v1/student_club/event_count_remaining_budget", operation_id="get_event_count_remaining_budget", summary="Retrieves the number of events that have a remaining budget less than the specified amount. The operation filters events by name and checks the remaining budget against the provided amount.")
async def get_event_count_remaining_budget(event_name: str = Query(..., description="Name of the event"), remaining_budget: int = Query(..., description="Remaining budget amount")):
    cursor.execute("SELECT COUNT(T2.event_id) FROM budget AS T1 INNER JOIN event AS T2 ON T1.link_to_event = T2.event_id WHERE T2.event_name = ? AND T1.remaining < ?", (event_name, remaining_budget))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total budget amount for a specific event
@app.get("/v1/student_club/total_budget_event", operation_id="get_total_budget_event", summary="Retrieves the total budget amount allocated for a specific event. The operation calculates the sum of all budget amounts associated with the provided event name.")
async def get_total_budget_event(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT SUM(T1.amount) FROM budget AS T1 INNER JOIN event AS T2 ON T1.link_to_event = T2.event_id WHERE T2.event_name = ?", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the event status based on expense description and date
@app.get("/v1/student_club/event_status", operation_id="get_event_status", summary="Retrieves the status of an event based on a specific expense description and date. The expense description and date are used to identify the relevant event and return its current status. This operation is useful for tracking the progress of events and their associated expenses.")
async def get_event_status(expense_description: str = Query(..., description="Description of the expense"), expense_date: str = Query(..., description="Date of the expense in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.event_status FROM budget AS T1 INNER JOIN expense AS T2 ON T1.budget_id = T2.link_to_budget WHERE T2.expense_description = ? AND T2.expense_date = ?", (expense_description, expense_date))
    result = cursor.fetchone()
    if not result:
        return {"event_status": []}
    return {"event_status": result[0]}

# Endpoint to get the count of members based on major name and t-shirt size
@app.get("/v1/student_club/count_members_by_major_tshirt_size", operation_id="get_count_members_by_major_tshirt_size", summary="Retrieves the total number of student club members who share a specific major and t-shirt size. The major and t-shirt size are provided as input parameters, allowing for a targeted count of members that meet these criteria.")
async def get_count_members_by_major_tshirt_size(major_name: str = Query(..., description="Major name of the member"), t_shirt_size: str = Query(..., description="T-shirt size of the member")):
    cursor.execute("SELECT COUNT(T1.member_id) FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T2.major_name = ? AND T1.t_shirt_size = ?", (major_name, t_shirt_size))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the type of zip code based on member's first and last name
@app.get("/v1/student_club/get_zip_code_type_by_name", operation_id="get_zip_code_type_by_name", summary="Retrieves the type of zip code associated with a member, based on their first and last name. This operation uses the provided first and last name to search for a matching member in the database and returns the type of their zip code.")
async def get_zip_code_type_by_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.type FROM member AS T1 INNER JOIN zip_code AS T2 ON T1.zip = T2.zip_code WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"type": []}
    return {"type": result[0]}

# Endpoint to get the major name based on member's position
@app.get("/v1/student_club/get_major_name_by_position", operation_id="get_major_name_by_position", summary="Retrieves the name of the major associated with a member based on their position. The position is used to identify the member and subsequently determine their major. The major name is then returned as the result.")
async def get_major_name_by_position(position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT T2.major_name FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T1.position = ?", (position,))
    result = cursor.fetchone()
    if not result:
        return {"major_name": []}
    return {"major_name": result[0]}

# Endpoint to get the state based on member's first and last name
@app.get("/v1/student_club/get_state_by_name", operation_id="get_state_by_name", summary="Retrieves the state associated with a member, based on their first and last name. This operation uses the provided first and last name to search for a matching member in the database. Once a match is found, the corresponding state is returned.")
async def get_state_by_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.state FROM member AS T1 INNER JOIN zip_code AS T2 ON T1.zip = T2.zip_code WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"state": []}
    return {"state": result[0]}

# Endpoint to get the department based on member's position
@app.get("/v1/student_club/get_department_by_position", operation_id="get_department_by_position", summary="Retrieves the department associated with a specific member position. This operation uses the provided member position to look up the corresponding department in the database. The result is a single department record.")
async def get_department_by_position(position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT T2.department FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T1.position = ?", (position,))
    result = cursor.fetchone()
    if not result:
        return {"department": []}
    return {"department": result[0]}

# Endpoint to get the date received based on member's first and last name and income source
@app.get("/v1/student_club/get_date_received_by_name_source", operation_id="get_date_received_by_name_source", summary="Get the date received based on member's first and last name and income source")
async def get_date_received_by_name_source(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member"), source: str = Query(..., description="Source of the income")):
    cursor.execute("SELECT T2.date_received FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T1.first_name = ? AND T1.last_name = ? AND T2.source = ?", (first_name, last_name, source))
    result = cursor.fetchone()
    if not result:
        return {"date_received": []}
    return {"date_received": result[0]}

# Endpoint to get the first and last name of the member based on income source
@app.get("/v1/student_club/get_member_name_by_source", operation_id="get_member_name_by_source", summary="Retrieves the first and last name of the member who most recently received income from the specified source.")
async def get_member_name_by_source(source: str = Query(..., description="Source of the income")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T2.source = ? ORDER BY T2.date_received LIMIT 1", (source,))
    result = cursor.fetchone()
    if not result:
        return {"first_name": [], "last_name": []}
    return {"first_name": result[0], "last_name": result[1]}

# Endpoint to get the ratio of budget amounts for two events based on category and event type
@app.get("/v1/student_club/get_budget_ratio_by_events", operation_id="get_budget_ratio_by_events", summary="Retrieves the ratio of budget amounts for two specified events, based on a given category and event type. The calculation considers the total budget amounts for each event and returns the ratio of the first event's budget to the second event's budget.")
async def get_budget_ratio_by_events(event_name_1: str = Query(..., description="First event name"), event_name_2: str = Query(..., description="Second event name"), category: str = Query(..., description="Category of the budget"), event_type: str = Query(..., description="Type of the event")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.event_name = ? THEN T1.amount ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.event_name = ? THEN T1.amount ELSE 0 END) FROM budget AS T1 INNER JOIN event AS T2 ON T1.link_to_event = T2.event_id WHERE T1.category = ? AND T2.type = ?", (event_name_1, event_name_2, category, event_type))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the percentage of budget amount for a specific category based on event name
@app.get("/v1/student_club/get_budget_percentage_by_category", operation_id="get_budget_percentage_by_category", summary="Retrieves the percentage of the total budget allocated to a specific category for a given event. The calculation is based on the sum of budget amounts for the specified category and the total sum of budget amounts for the event.")
async def get_budget_percentage_by_category(category: str = Query(..., description="Category of the budget"), event_name: str = Query(..., description="Event name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.category = ? THEN T1.amount ELSE 0 END) AS REAL) * 100 / SUM(T1.amount) FROM budget AS T1 INNER JOIN event AS T2 ON T1.link_to_event = T2.event_id WHERE T2.event_name = ?", (category, event_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total cost based on expense description
@app.get("/v1/student_club/get_total_cost_by_description", operation_id="get_total_cost_by_description", summary="Retrieves the total cost of expenses that match the provided description. The description is used to filter the expenses and calculate the sum of their costs.")
async def get_total_cost_by_description(expense_description: str = Query(..., description="Description of the expense")):
    cursor.execute("SELECT SUM(cost) FROM expense WHERE expense_description = ?", (expense_description,))
    result = cursor.fetchone()
    if not result:
        return {"total_cost": []}
    return {"total_cost": result[0]}

# Endpoint to get the count of cities in a specific county and state
@app.get("/v1/student_club/count_cities_in_county_state", operation_id="get_count_cities", summary="Retrieves the total number of unique cities located within a specified county and state. The operation requires the county and state names as input parameters to filter the data and calculate the count.")
async def get_count_cities(county: str = Query(..., description="County name"), state: str = Query(..., description="State name")):
    cursor.execute("SELECT COUNT(city) FROM zip_code WHERE county = ? AND state = ?", (county, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get departments from a specific college
@app.get("/v1/student_club/departments_by_college", operation_id="get_departments", summary="Retrieves a list of departments associated with a specific college. The operation requires the college name as input and returns the corresponding departments.")
async def get_departments(college: str = Query(..., description="College name")):
    cursor.execute("SELECT department FROM major WHERE college = ?", (college,))
    result = cursor.fetchall()
    if not result:
        return {"departments": []}
    return {"departments": [row[0] for row in result]}

# Endpoint to get city, county, and state of a member by first and last name
@app.get("/v1/student_club/member_location", operation_id="get_member_location", summary="Retrieves the city, county, and state associated with a specific member, identified by their first and last name. The member's location is determined by matching their zip code with a database of zip codes.")
async def get_member_location(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.city, T2.county, T2.state FROM member AS T1 INNER JOIN zip_code AS T2 ON T1.zip = T2.zip_code WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": {"city": result[0], "county": result[1], "state": result[2]}}

# Endpoint to get the first expense description based on remaining budget
@app.get("/v1/student_club/first_expense_by_remaining_budget", operation_id="get_first_expense", summary="Retrieves the description of the first expense that was made from a budget, sorted by the remaining budget in descending order.")
async def get_first_expense():
    cursor.execute("SELECT T2.expense_description FROM budget AS T1 INNER JOIN expense AS T2 ON T1.budget_id = T2.link_to_budget ORDER BY T1.remaining LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"expense_description": []}
    return {"expense_description": result[0]}

# Endpoint to get distinct member IDs who attended a specific event
@app.get("/v1/student_club/attendees_by_event", operation_id="get_attendees", summary="Retrieve a unique list of member IDs who attended a specific event. The operation filters the attendance records based on the provided event name and returns the distinct member IDs associated with the event.")
async def get_attendees(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT DISTINCT T3.member_id FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event INNER JOIN member AS T3 ON T2.link_to_member = T3.member_id WHERE T1.event_name = ?", (event_name,))
    result = cursor.fetchall()
    if not result:
        return {"member_ids": []}
    return {"member_ids": [row[0] for row in result]}

# Endpoint to get the college with the most members in a specific major
@app.get("/v1/student_club/most_popular_college_by_major", operation_id="get_most_popular_college", summary="Retrieves the college with the highest number of student members in a given major. The operation groups students by their major and counts the number of students in each college within that major. The college with the highest count is returned.")
async def get_most_popular_college():
    cursor.execute("SELECT T2.college FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id GROUP BY T2.major_id ORDER BY COUNT(T2.college) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"college": []}
    return {"college": result[0]}

# Endpoint to get the major name of a member by phone number
@app.get("/v1/student_club/major_by_phone", operation_id="get_major_by_phone", summary="Retrieves the major name associated with a member, identified by their phone number. The major name is obtained by joining the member and major tables using the member's link to the major table.")
async def get_major_by_phone(phone: str = Query(..., description="Phone number of the member")):
    cursor.execute("SELECT T2.major_name FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T1.phone = ?", (phone,))
    result = cursor.fetchone()
    if not result:
        return {"major_name": []}
    return {"major_name": result[0]}

# Endpoint to get the event with the highest budget amount
@app.get("/v1/student_club/highest_budget_event", operation_id="get_highest_budget_event", summary="Retrieves the name of the event with the highest budget allocation. This operation fetches the event with the maximum budget amount from the database, providing a quick overview of the most financially significant event.")
async def get_highest_budget_event():
    cursor.execute("SELECT T2.event_name FROM budget AS T1 INNER JOIN event AS T2 ON T1.link_to_event = T2.event_id ORDER BY T1.amount DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"event_name": []}
    return {"event_name": result[0]}

# Endpoint to get expenses for a member by their position
@app.get("/v1/student_club/expenses_by_position", operation_id="get_expenses_by_position", summary="Retrieves a list of expenses associated with a specific member position within the student club. The expenses are identified by their unique IDs and descriptions.")
async def get_expenses_by_position(position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT T2.expense_id, T2.expense_description FROM member AS T1 INNER JOIN expense AS T2 ON T1.member_id = T2.link_to_member WHERE T1.position = ?", (position,))
    result = cursor.fetchall()
    if not result:
        return {"expenses": []}
    return {"expenses": [{"expense_id": row[0], "expense_description": row[1]} for row in result]}

# Endpoint to get the count of attendees for a specific event
@app.get("/v1/student_club/attendee_count_by_event", operation_id="get_attendee_count", summary="Retrieves the total number of attendees for a specified event. The operation calculates the count based on the provided event name, which is used to identify the relevant event and its associated attendance records.")
async def get_attendee_count(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT COUNT(T2.link_to_member) FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_name = ?", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the date received for a member based on first and last name
@app.get("/v1/student_club/date_received_by_name", operation_id="get_date_received_by_name", summary="Retrieves the date a member received income, based on their first and last name. The operation uses the provided names to identify the member and returns the corresponding date from the income record.")
async def get_date_received_by_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.date_received FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"date_received": []}
    return {"date_received": [row[0] for row in result]}

# Endpoint to get the count of members in a specific state
@app.get("/v1/student_club/member_count_by_state", operation_id="get_member_count_by_state", summary="Retrieves the total number of members residing in a specified state. The state is identified by its name.")
async def get_member_count_by_state(state: str = Query(..., description="State name")):
    cursor.execute("SELECT COUNT(T2.member_id) FROM zip_code AS T1 INNER JOIN member AS T2 ON T1.zip_code = T2.zip WHERE T1.state = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of events attended by a member based on phone number
@app.get("/v1/student_club/event_count_by_phone", operation_id="get_event_count_by_phone", summary="Retrieves the total number of events attended by a specific member, identified by their phone number. The count is determined by matching the provided phone number with the member's records and then tallying the corresponding event attendance records.")
async def get_event_count_by_phone(phone: str = Query(..., description="Phone number of the member")):
    cursor.execute("SELECT COUNT(T2.link_to_event) FROM member AS T1 INNER JOIN attendance AS T2 ON T1.member_id = T2.link_to_member WHERE T1.phone = ?", (phone,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the event name with the highest budget utilization ratio for a given status
@app.get("/v1/student_club/event_with_highest_budget_utilization", operation_id="get_event_with_highest_budget_utilization", summary="Retrieves the name of the event that has the highest budget utilization ratio among events with the specified status. The budget utilization ratio is calculated as the amount spent divided by the total budget amount.")
async def get_event_with_highest_budget_utilization(status: str = Query(..., description="Status of the event")):
    cursor.execute("SELECT T2.event_name FROM budget AS T1 INNER JOIN event AS T2 ON T1.link_to_event = T2.event_id WHERE T2.status = ? ORDER BY T1.spent / T1.amount DESC LIMIT 1", (status,))
    result = cursor.fetchone()
    if not result:
        return {"event_name": []}
    return {"event_name": result[0]}

# Endpoint to get the count of members based on their position
@app.get("/v1/student_club/member_count_by_position", operation_id="get_member_count_by_position", summary="Retrieves the total number of members holding a specific position within the student club. The position is provided as an input parameter.")
async def get_member_count_by_position(position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT COUNT(member_id) FROM member WHERE position = ?", (position,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum amount spent from the budget
@app.get("/v1/student_club/max_spent_budget", operation_id="get_max_spent_budget", summary="Retrieves the highest amount spent from the budget. This operation provides a single value representing the maximum budget expenditure.")
async def get_max_spent_budget():
    cursor.execute("SELECT MAX(spent) FROM budget")
    result = cursor.fetchone()
    if not result:
        return {"max_spent": []}
    return {"max_spent": result[0]}

# Endpoint to get the count of events of a specific type in a specific year
@app.get("/v1/student_club/event_count_by_type_and_year", operation_id="get_event_count_by_type_and_year", summary="Retrieves the total number of events of a specified type that occurred in a given year. The event type and year are provided as input parameters.")
async def get_event_count_by_type_and_year(event_type: str = Query(..., description="Type of the event"), year: str = Query(..., description="Year of the event in 'YYYY' format")):
    cursor.execute("SELECT COUNT(event_id) FROM event WHERE type = ? AND SUBSTR(event_date, 1, 4) = ?", (event_type, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total amount spent in a specific category
@app.get("/v1/student_club/total_spent_by_category", operation_id="get_total_spent_by_category", summary="Retrieves the total amount spent in a specified budget category. The category is provided as an input parameter, allowing the user to select the category for which the total spent amount is calculated.")
async def get_total_spent_by_category(category: str = Query(..., description="Category of the budget")):
    cursor.execute("SELECT SUM(spent) FROM budget WHERE category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"total_spent": []}
    return {"total_spent": result[0]}

# Endpoint to get members who have attended more than a specified number of events
@app.get("/v1/student_club/members_by_event_attendance", operation_id="get_members_by_event_attendance", summary="Retrieves the first and last names of members who have attended more than the specified minimum number of events. The input parameter determines the minimum attendance threshold.")
async def get_members_by_event_attendance(min_events: int = Query(..., description="Minimum number of events attended")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN attendance AS T2 ON T1.member_id = T2.link_to_member GROUP BY T2.link_to_member HAVING COUNT(T2.link_to_event) > ?", (min_events,))
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the names of members who attended a specific event and belong to a specific major
@app.get("/v1/student_club/member_names_by_event_and_major", operation_id="get_member_names_by_event_and_major", summary="Retrieves the first and last names of members who attended a particular event and belong to a specific major. The operation requires the name of the event and the name of the major as input parameters.")
async def get_member_names_by_event_and_major(event_name: str = Query(..., description="Name of the event"), major_name: str = Query(..., description="Name of the major")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM major AS T1 INNER JOIN member AS T2 ON T1.major_id = T2.link_to_major INNER JOIN attendance AS T3 ON T2.member_id = T3.link_to_member INNER JOIN event AS T4 ON T3.link_to_event = T4.event_id WHERE T4.event_name = ? AND T1.major_name = ?", (event_name, major_name))
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": result}

# Endpoint to get the names of members based on city and state
@app.get("/v1/student_club/member_names_by_city_and_state", operation_id="get_member_names_by_city_and_state", summary="Retrieves the first and last names of members who reside in a specific city and state. The operation filters members based on the provided city and state names, returning a list of corresponding member names.")
async def get_member_names_by_city_and_state(city: str = Query(..., description="City name"), state: str = Query(..., description="State name")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN zip_code AS T2 ON T1.zip = T2.zip_code WHERE T2.city = ? AND T2.state = ?", (city, state))
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": result}

# Endpoint to get the income amount of a specific member
@app.get("/v1/student_club/income_amount_by_member_name", operation_id="get_income_amount_by_member_name", summary="Retrieves the total income amount associated with a specific member, identified by their first and last name. The endpoint fetches this information from the member and income tables, joining them based on the member's unique identifier. The response includes the total income amount linked to the member.")
async def get_income_amount_by_member_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.amount FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"income": []}
    return {"income": result}

# Endpoint to get the names of members with income greater than a specified amount
@app.get("/v1/student_club/member_names_by_income", operation_id="get_member_names_by_income", summary="Retrieves the first and last names of club members who have an income greater than the specified amount. The income amount is provided as an input parameter.")
async def get_member_names_by_income(amount: int = Query(..., description="Income amount")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T2.amount > ?", (amount,))
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": result}

# Endpoint to get the total cost of expenses for a specific event
@app.get("/v1/student_club/total_expense_by_event", operation_id="get_total_expense_by_event", summary="Retrieves the total cost of expenses incurred for a specific event. The operation calculates the sum of all expenses associated with the given event, providing a comprehensive view of the event's financial impact.")
async def get_total_expense_by_event(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT SUM(T3.cost) FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget WHERE T1.event_name = ?", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_cost": []}
    return {"total_cost": result[0]}

# Endpoint to get the names of members associated with expenses for a specific event
@app.get("/v1/student_club/member_names_by_event_expenses", operation_id="get_member_names_by_event_expenses", summary="Retrieves the first and last names of members who have incurred expenses for a specific event. The operation filters members based on the provided event name.")
async def get_member_names_by_event_expenses(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT T4.first_name, T4.last_name FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget INNER JOIN member AS T4 ON T3.link_to_member = T4.member_id WHERE T1.event_name = ?", (event_name,))
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": result}

# Endpoint to get the member with the highest total income and their income source
@app.get("/v1/student_club/top_income_member", operation_id="get_top_income_member", summary="Retrieves the full name and income source of the member with the highest total income from all income sources. The data is obtained by aggregating income amounts from all sources and ranking the members accordingly.")
async def get_top_income_member():
    cursor.execute("SELECT T1.first_name, T1.last_name, T2.source FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member GROUP BY T1.first_name, T1.last_name, T2.source ORDER BY SUM(T2.amount) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"top_member": []}
    return {"top_member": result}

# Endpoint to get the event with the highest expense cost
@app.get("/v1/student_club/highest_expense_event", operation_id="get_highest_expense_event", summary="Retrieves the name of the event that has the highest associated expense cost. This operation fetches the event name from the 'event' table, which is linked to the 'budget' and 'expense' tables via their respective IDs. The expense cost is used to determine the event with the highest expense.")
async def get_highest_expense_event():
    cursor.execute("SELECT T1.event_name FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget ORDER BY T3.cost LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"event": []}
    return {"event": result[0]}

# Endpoint to get the percentage of total expenses for a specific event
@app.get("/v1/student_club/expense_percentage_by_event", operation_id="get_expense_percentage_by_event", summary="Retrieves the proportion of total expenses attributed to a specific event. This operation calculates the percentage by summing the costs of expenses linked to the specified event and dividing it by the total expenses across all events. The event name is required as an input parameter.")
async def get_expense_percentage_by_event(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.event_name = ? THEN T3.cost ELSE 0 END) AS REAL) * 100 / SUM(T3.cost) FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of members in one major to another
@app.get("/v1/student_club/major_ratio", operation_id="get_major_ratio", summary="Retrieves the ratio of members in one major to another. The operation compares the number of members in two specified majors and returns the ratio. The input parameters define the majors to be compared.")
async def get_major_ratio(major_name_1: str = Query(..., description="Name of the first major"), major_name_2: str = Query(..., description="Name of the second major")):
    cursor.execute("SELECT SUM(CASE WHEN major_name = ? THEN 1 ELSE 0 END) / SUM(CASE WHEN major_name = ? THEN 1 ELSE 0 END) AS ratio FROM major", (major_name_1, major_name_2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the source of income received between two dates
@app.get("/v1/student_club/income_source_by_date", operation_id="get_income_source_by_date", summary="Retrieves the primary source of income received between the specified start and end dates, sorted in descending order. The start and end dates must be provided in 'YYYY-MM-DD' format.")
async def get_income_source_by_date(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT source FROM income WHERE date_received BETWEEN ? AND ? ORDER BY source DESC LIMIT 1", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"source": []}
    return {"source": result[0]}

# Endpoint to get member details based on position
@app.get("/v1/student_club/member_details_by_position", operation_id="get_member_details_by_position", summary="Retrieves the first name, last name, and email of a member based on their position within the club. The position is a required input parameter.")
async def get_member_details_by_position(position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT first_name, last_name, email FROM member WHERE position = ?", (position,))
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": result}

# Endpoint to get the count of attendees for a specific event in a given year
@app.get("/v1/student_club/attendee_count_by_event_year", operation_id="get_attendee_count_by_event_year", summary="Retrieves the total number of attendees for a specific event held in a given year. The operation requires the event's name and the year in 'YYYY' format as input parameters to accurately calculate the attendee count.")
async def get_attendee_count_by_event_year(event_name: str = Query(..., description="Name of the event"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.link_to_member) FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_name = ? AND SUBSTR(T1.event_date, 1, 4) = ?", (event_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of events attended by a member and their major
@app.get("/v1/student_club/event_count_by_member_major", operation_id="get_event_count_by_member_major", summary="Retrieves the total number of events attended by a member, along with their major. The operation requires the first and last name of the member to identify the individual and their major, and subsequently, the events they have attended.")
async def get_event_count_by_member_major(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT COUNT(T3.link_to_event), T1.major_name FROM major AS T1 INNER JOIN member AS T2 ON T1.major_id = T2.link_to_major INNER JOIN attendance AS T3 ON T2.member_id = T3.link_to_member WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"event_count_major": []}
    return {"event_count_major": result}

# Endpoint to get the average spent amount for a specific category and event status
@app.get("/v1/student_club/average_spent_by_category_status", operation_id="get_average_spent_by_category_status", summary="Retrieves the average amount spent for a given budget category and event status. This operation calculates the total spent amount for the specified category and event status, then divides it by the count of entries to provide the average. The input parameters determine the category and status for which the average is calculated.")
async def get_average_spent_by_category_status(category: str = Query(..., description="Category of the budget"), event_status: str = Query(..., description="Status of the event")):
    cursor.execute("SELECT SUM(spent) / COUNT(spent) FROM budget WHERE category = ? AND event_status = ?", (category, event_status))
    result = cursor.fetchone()
    if not result:
        return {"average_spent": []}
    return {"average_spent": result[0]}

# Endpoint to get the event name with the highest spent amount in a specific category
@app.get("/v1/student_club/highest_spent_event_by_category", operation_id="get_highest_spent_event_by_category", summary="Retrieves the name of the event that has the highest spent amount within a specified budget category. The category is provided as an input parameter.")
async def get_highest_spent_event_by_category(category: str = Query(..., description="Category of the budget")):
    cursor.execute("SELECT T2.event_name FROM budget AS T1 INNER JOIN event AS T2 ON T1.link_to_event = T2.event_id WHERE T1.category = ? ORDER BY T1.spent DESC LIMIT 1", (category,))
    result = cursor.fetchone()
    if not result:
        return {"event_name": []}
    return {"event_name": result[0]}

# Endpoint to check if a member attended a specific event
@app.get("/v1/student_club/check_member_attendance", operation_id="check_member_attendance", summary="Check if a member attended a specific event")
async def check_member_attendance(event_name: str = Query(..., description="Name of the event"), first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT CASE WHEN T3.event_name = ? THEN 'YES' END AS result FROM member AS T1 INNER JOIN attendance AS T2 ON T1.member_id = T2.link_to_member INNER JOIN event AS T3 ON T2.link_to_event = T3.event_id WHERE T1.first_name = ? AND T1.last_name = ?", (event_name, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"attendance": []}
    return {"attendance": result[0]}

# Endpoint to get the percentage of events of a specific type in a given year
@app.get("/v1/student_club/percentage_events_by_type_year", operation_id="get_percentage_events_by_type_year", summary="Retrieves the percentage of events of a specified type that occurred in a given year. The calculation is based on the total count of events of the specified type and the overall count of events in the provided year.")
async def get_percentage_events_by_type_year(event_type: str = Query(..., description="Type of the event"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN type = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(type) FROM event WHERE SUBSTR(event_date, 1, 4) = ?", (event_type, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the cost of a specific expense for a given event
@app.get("/v1/student_club/expense_cost_by_event", operation_id="get_expense_cost_by_event", summary="Retrieves the cost of a specific expense associated with a given event. The operation requires the name of the event and a description of the expense to accurately identify and return the cost.")
async def get_expense_cost_by_event(event_name: str = Query(..., description="Name of the event"), expense_description: str = Query(..., description="Description of the expense")):
    cursor.execute("SELECT T3.cost FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget WHERE T1.event_name = ? AND T3.expense_description = ?", (event_name, expense_description))
    result = cursor.fetchone()
    if not result:
        return {"cost": []}
    return {"cost": result[0]}

# Endpoint to get the most common t-shirt size among members
@app.get("/v1/student_club/most_common_tshirt_size", operation_id="get_most_common_tshirt_size", summary="Retrieves the most frequently occurring t-shirt size among the members of the student club. The operation returns the single most common size, providing a valuable insight into the club's membership demographics.")
async def get_most_common_tshirt_size():
    cursor.execute("SELECT t_shirt_size FROM member GROUP BY t_shirt_size ORDER BY COUNT(t_shirt_size) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"t_shirt_size": []}
    return {"t_shirt_size": result[0]}

# Endpoint to get the event name with the lowest remaining budget for a given event status
@app.get("/v1/student_club/event_with_lowest_remaining_budget", operation_id="get_event_with_lowest_remaining_budget", summary="Retrieves the name of the event with the lowest remaining budget that is below a specified threshold for a given event status. The operation returns the event name that has the least budget remaining, considering the provided event status and remaining budget threshold.")
async def get_event_with_lowest_remaining_budget(event_status: str = Query(..., description="Event status"), remaining_threshold: int = Query(..., description="Remaining budget threshold")):
    cursor.execute("SELECT T2.event_name FROM budget AS T1 INNER JOIN event AS T2 ON T2.event_id = T1.link_to_event WHERE T1.event_status = ? AND T1.remaining < ? ORDER BY T1.remaining LIMIT 1", (event_status, remaining_threshold))
    result = cursor.fetchone()
    if not result:
        return {"event_name": []}
    return {"event_name": result[0]}

# Endpoint to get the total cost of expenses for a given event name
@app.get("/v1/student_club/total_expense_by_event_name", operation_id="get_total_expense_by_event_name", summary="Get the total cost of expenses for a given event name")
async def get_total_expense_by_event_name(event_name: str = Query(..., description="Event name")):
    cursor.execute("SELECT T1.type, SUM(T3.cost) FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget WHERE T1.event_name = ?", (event_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_cost": []}
    return {"total_cost": result[1]}

# Endpoint to get the total budget amount by category for a given event name
@app.get("/v1/student_club/budget_by_category_event_name", operation_id="get_budget_by_category_event_name", summary="Retrieves the total budget amount allocated to each category for a specific event. The budget amounts are summed up and presented in ascending order. The event is identified by its name.")
async def get_budget_by_category_event_name(event_name: str = Query(..., description="Event name")):
    cursor.execute("SELECT T2.category, SUM(T2.amount) FROM event AS T1 JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_name = ? GROUP BY T2.category ORDER BY SUM(T2.amount) ASC", (event_name,))
    result = cursor.fetchall()
    if not result:
        return {"budget_by_category": []}
    return {"budget_by_category": [{"category": row[0], "total_amount": row[1]} for row in result]}

# Endpoint to get the budget ID with the maximum amount for a given category
@app.get("/v1/student_club/max_budget_by_category", operation_id="get_max_budget_by_category", summary="Retrieves the budget ID with the highest amount for a specified category. The category is provided as an input parameter.")
async def get_max_budget_by_category(category: str = Query(..., description="Budget category")):
    cursor.execute("SELECT budget_id FROM budget WHERE category = ? AND amount = ( SELECT MAX(amount) FROM budget )", (category,))
    result = cursor.fetchone()
    if not result:
        return {"budget_id": []}
    return {"budget_id": result[0]}

# Endpoint to get the top 3 budget IDs by amount for a given category
@app.get("/v1/student_club/top_3_budgets_by_category", operation_id="get_top_3_budgets_by_category", summary="Retrieves the identifiers of the top three budgets with the highest amounts for a specified category. The category is provided as an input parameter.")
async def get_top_3_budgets_by_category(category: str = Query(..., description="Budget category")):
    cursor.execute("SELECT budget_id FROM budget WHERE category = ? ORDER BY amount DESC LIMIT 3", (category,))
    result = cursor.fetchall()
    if not result:
        return {"budget_ids": []}
    return {"budget_ids": [row[0] for row in result]}

# Endpoint to get the total cost of expenses for a given date
@app.get("/v1/student_club/total_expense_by_date", operation_id="get_total_expense_by_date", summary="Retrieves the total cost of expenses incurred on a specific date. The date must be provided in the 'YYYY-MM-DD' format. This operation calculates the sum of all expenses recorded for the given date.")
async def get_total_expense_by_date(expense_date: str = Query(..., description="Expense date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(cost) FROM expense WHERE expense_date = ?", (expense_date,))
    result = cursor.fetchone()
    if not result:
        return {"total_cost": []}
    return {"total_cost": result[0]}

# Endpoint to get the total cost of expenses for a given member ID
@app.get("/v1/student_club/total_expense_by_member_id", operation_id="get_total_expense_by_member_id", summary="Retrieves the total cost of expenses incurred by a specific member. The member is identified by the provided member_id. The response includes the member's first and last names along with the total cost of their expenses.")
async def get_total_expense_by_member_id(member_id: str = Query(..., description="Member ID")):
    cursor.execute("SELECT T1.first_name, T1.last_name, SUM(T2.cost) FROM member AS T1 INNER JOIN expense AS T2 ON T1.member_id = T2.link_to_member WHERE T1.member_id = ?", (member_id,))
    result = cursor.fetchone()
    if not result:
        return {"total_cost": []}
    return {"first_name": result[0], "last_name": result[1], "total_cost": result[2]}

# Endpoint to get the expense descriptions for a given member's first and last name
@app.get("/v1/student_club/expense_descriptions_by_member_name", operation_id="get_expense_descriptions_by_member_name", summary="Retrieves the descriptions of expenses associated with a specific member, identified by their first and last name. The operation returns a list of expense descriptions linked to the member's ID in the database.")
async def get_expense_descriptions_by_member_name(first_name: str = Query(..., description="Member's first name"), last_name: str = Query(..., description="Member's last name")):
    cursor.execute("SELECT T2.expense_description FROM member AS T1 INNER JOIN expense AS T2 ON T1.member_id = T2.link_to_member WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"expense_descriptions": []}
    return {"expense_descriptions": [row[0] for row in result]}

# Endpoint to get expense descriptions for members with a specific t-shirt size
@app.get("/v1/student_club/expense_descriptions_by_t_shirt_size", operation_id="get_expense_descriptions", summary="Retrieves the expense descriptions associated with members who wear a specific t-shirt size. The t-shirt size is provided as an input parameter.")
async def get_expense_descriptions(t_shirt_size: str = Query(..., description="T-shirt size of the member")):
    cursor.execute("SELECT T2.expense_description FROM member AS T1 INNER JOIN expense AS T2 ON T1.member_id = T2.link_to_member WHERE T1.t_shirt_size = ?", (t_shirt_size,))
    result = cursor.fetchall()
    if not result:
        return {"expense_descriptions": []}
    return {"expense_descriptions": [row[0] for row in result]}

# Endpoint to get zip codes of members with expenses below a certain cost
@app.get("/v1/student_club/zip_codes_by_expense_cost", operation_id="get_zip_codes", summary="Retrieves the zip codes of members who have expenses below a specified cost. This operation allows you to filter members based on their expense cost, providing a list of zip codes that meet the specified criteria.")
async def get_zip_codes(cost: int = Query(..., description="Maximum cost of the expense")):
    cursor.execute("SELECT T1.zip FROM member AS T1 INNER JOIN expense AS T2 ON T1.member_id = T2.link_to_member WHERE T2.cost < ?", (cost,))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes": []}
    return {"zip_codes": [row[0] for row in result]}

# Endpoint to get major names of members with a specific first and last name
@app.get("/v1/student_club/major_names_by_member_name", operation_id="get_major_names", summary="Retrieves the major names associated with members who share a specified first and last name. The operation filters members by their first and last names and returns the corresponding major names.")
async def get_major_names(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T1.major_name FROM major AS T1 INNER JOIN member AS T2 ON T1.major_id = T2.link_to_major WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"major_names": []}
    return {"major_names": [row[0] for row in result]}

# Endpoint to get positions of members in a specific major
@app.get("/v1/student_club/positions_by_major_name", operation_id="get_positions", summary="Retrieves the positions held by members of a specific major. The major is identified by its name, which is provided as an input parameter.")
async def get_positions(major_name: str = Query(..., description="Name of the major")):
    cursor.execute("SELECT T2.position FROM major AS T1 INNER JOIN member AS T2 ON T1.major_id = T2.link_to_major WHERE T1.major_name = ?", (major_name,))
    result = cursor.fetchall()
    if not result:
        return {"positions": []}
    return {"positions": [row[0] for row in result]}

# Endpoint to get the count of members in a specific major with a specific t-shirt size
@app.get("/v1/student_club/count_members_by_major_and_t_shirt_size", operation_id="get_member_count", summary="Retrieves the total number of members in a specific major who have a particular t-shirt size. The major is identified by its name, and the t-shirt size is specified as a parameter. This operation is useful for understanding the distribution of t-shirt sizes among members of a specific major.")
async def get_member_count(major_name: str = Query(..., description="Name of the major"), t_shirt_size: str = Query(..., description="T-shirt size of the member")):
    cursor.execute("SELECT COUNT(T2.member_id) FROM major AS T1 INNER JOIN member AS T2 ON T1.major_id = T2.link_to_major WHERE T1.major_name = ? AND T2.t_shirt_size = ?", (major_name, t_shirt_size))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get event types with a budget remaining above a certain amount
@app.get("/v1/student_club/event_types_by_remaining_budget", operation_id="get_event_types", summary="Retrieves event types that have a remaining budget greater than the specified minimum amount. This operation returns a list of event types that meet the budget criteria, providing a useful overview of events with sufficient funds for planning and execution.")
async def get_event_types(remaining: int = Query(..., description="Minimum remaining budget")):
    cursor.execute("SELECT T1.type FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T2.remaining > ?", (remaining,))
    result = cursor.fetchall()
    if not result:
        return {"event_types": []}
    return {"event_types": [row[0] for row in result]}

# Endpoint to get budget categories for events at a specific location
@app.get("/v1/student_club/budget_categories_by_event_location", operation_id="get_budget_categories", summary="Retrieves the budget categories associated with events held at a specified location. The location parameter is used to filter the events and return the corresponding budget categories.")
async def get_budget_categories(location: str = Query(..., description="Location of the event")):
    cursor.execute("SELECT T2.category FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T1.location = ?", (location,))
    result = cursor.fetchall()
    if not result:
        return {"budget_categories": []}
    return {"budget_categories": [row[0] for row in result]}

# Endpoint to get budget categories for events on a specific date
@app.get("/v1/student_club/budget_categories_by_event_date", operation_id="get_budget_categories_by_date", summary="Retrieves the budget categories associated with events occurring on a specified date. The date should be provided in the 'YYYY-MM-DDTHH:MM:SS' format.")
async def get_budget_categories_by_date(event_date: str = Query(..., description="Event date in 'YYYY-MM-DDTHH:MM:SS' format")):
    cursor.execute("SELECT T2.category FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_date = ?", (event_date,))
    result = cursor.fetchall()
    if not result:
        return {"budget_categories": []}
    return {"budget_categories": [row[0] for row in result]}

# Endpoint to get major names of members with a specific position
@app.get("/v1/student_club/major_names_by_position", operation_id="get_major_names_by_position", summary="Retrieves the major names of members who hold a specific position. The position is provided as an input parameter, and the operation returns a list of major names that correspond to members with the specified position.")
async def get_major_names_by_position(position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT T1.major_name FROM major AS T1 INNER JOIN member AS T2 ON T1.major_id = T2.link_to_major WHERE T2.position = ?", (position,))
    result = cursor.fetchall()
    if not result:
        return {"major_names": []}
    return {"major_names": [row[0] for row in result]}

# Endpoint to get the percentage of members in a specific position who are in a specific major
@app.get("/v1/student_club/percentage_members_by_position_and_major", operation_id="get_percentage_members", summary="Retrieves the percentage of members in a specific position who belong to a specific major. The operation calculates this percentage by counting the number of members in the given position who are in the specified major, and then dividing that by the total number of members in the given position.")
async def get_percentage_members(position: str = Query(..., description="Position of the member"), major_name: str = Query(..., description="Name of the major")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.major_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.member_id) FROM member AS T1 INNER JOIN major AS T2 ON T2.major_id = T1.link_to_major WHERE T1.position = ?", (major_name, position))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct categories of budgets for events at a specific location
@app.get("/v1/student_club/distinct_budget_categories_by_location", operation_id="get_distinct_budget_categories", summary="Retrieve a unique set of budget categories for events held at a specified location. This operation provides a comprehensive view of the different budget categories associated with events in a particular area, enabling users to understand the range of budget allocations for events at that location.")
async def get_distinct_budget_categories(location: str = Query(..., description="Location of the event")):
    cursor.execute("SELECT DISTINCT T2.category FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T1.location = ?", (location,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get the count of income records with a specific amount
@app.get("/v1/student_club/count_income_by_amount", operation_id="get_count_income_by_amount", summary="Retrieves the total count of income records that match a specified amount. This operation provides a quantitative overview of income records based on the given amount.")
async def get_count_income_by_amount(amount: int = Query(..., description="Amount of income")):
    cursor.execute("SELECT COUNT(income_id) FROM income WHERE amount = ?", (amount,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of members with a specific position and t-shirt size
@app.get("/v1/student_club/count_members_by_position_and_tshirt_size", operation_id="get_count_members_by_position_and_tshirt_size", summary="Retrieves the total number of members who hold a certain position and wear a specific t-shirt size.")
async def get_count_members_by_position_and_tshirt_size(position: str = Query(..., description="Position of the member"), t_shirt_size: str = Query(..., description="T-shirt size of the member")):
    cursor.execute("SELECT COUNT(member_id) FROM member WHERE position = ? AND t_shirt_size = ?", (position, t_shirt_size))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of majors in a specific department and college
@app.get("/v1/student_club/count_majors_by_department_and_college", operation_id="get_count_majors_by_department_and_college", summary="Retrieves the total number of majors in a specified department and college. The department and college are provided as input parameters, allowing for a targeted count of majors within a specific academic division.")
async def get_count_majors_by_department_and_college(department: str = Query(..., description="Department of the major"), college: str = Query(..., description="College of the major")):
    cursor.execute("SELECT COUNT(major_id) FROM major WHERE department = ? AND college = ?", (department, college))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get member details based on position and major name
@app.get("/v1/student_club/member_details_by_position_and_major", operation_id="get_member_details_by_position_and_major", summary="Retrieves the last name, department, and college of a student club member based on their position and the name of their major. The position and major name are provided as input parameters.")
async def get_member_details_by_position_and_major(position: str = Query(..., description="Position of the member"), major_name: str = Query(..., description="Name of the major")):
    cursor.execute("SELECT T2.last_name, T1.department, T1.college FROM major AS T1 INNER JOIN member AS T2 ON T1.major_id = T2.link_to_major WHERE T2.position = ? AND T1.major_name = ?", (position, major_name))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": [{"last_name": row[0], "department": row[1], "college": row[2]} for row in result]}

# Endpoint to get distinct categories and event types for events at a specific location with a specific spent amount and event type
@app.get("/v1/student_club/distinct_categories_and_types_by_location_spent_and_type", operation_id="get_distinct_categories_and_types", summary="Retrieves unique categories and event types for events held at a specified location, with a specified spent amount and event type. This operation provides a comprehensive view of the distinct categories and event types associated with events that meet the given criteria.")
async def get_distinct_categories_and_types(location: str = Query(..., description="Location of the event"), spent: int = Query(..., description="Spent amount"), event_type: str = Query(..., description="Type of the event")):
    cursor.execute("SELECT DISTINCT T2.category, T1.type FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T1.location = ? AND T2.spent = ? AND T1.type = ?", (location, spent, event_type))
    result = cursor.fetchall()
    if not result:
        return {"categories_and_types": []}
    return {"categories_and_types": [{"category": row[0], "type": row[1]} for row in result]}

# Endpoint to get city and state of members based on department and position
@app.get("/v1/student_club/city_state_by_department_and_position", operation_id="get_city_state_by_department_and_position", summary="Retrieves the city and state of members who belong to a specific department and hold a certain position. The department and position are used to filter the members, and the city and state are returned as the result.")
async def get_city_state_by_department_and_position(department: str = Query(..., description="Department of the major"), position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT city, state FROM member AS T1 INNER JOIN major AS T2 ON T2.major_id = T1.link_to_major INNER JOIN zip_code AS T3 ON T3.zip_code = T1.zip WHERE department = ? AND position = ?", (department, position))
    result = cursor.fetchall()
    if not result:
        return {"city_state": []}
    return {"city_state": [{"city": row[0], "state": row[1]} for row in result]}

# Endpoint to get event names attended by members with a specific position at a specific location and event type
@app.get("/v1/student_club/event_names_by_position_location_and_type", operation_id="get_event_names_by_position_location_and_type", summary="Retrieves the names of events that were attended by members holding a specific position at a designated location and event type. The position, location, and event type are provided as input parameters.")
async def get_event_names_by_position_location_and_type(position: str = Query(..., description="Position of the member"), location: str = Query(..., description="Location of the event"), event_type: str = Query(..., description="Type of the event")):
    cursor.execute("SELECT T2.event_name FROM attendance AS T1 INNER JOIN event AS T2 ON T2.event_id = T1.link_to_event INNER JOIN member AS T3 ON T1.link_to_member = T3.member_id WHERE T3.position = ? AND T2.location = ? AND T2.type = ?", (position, location, event_type))
    result = cursor.fetchall()
    if not result:
        return {"event_names": []}
    return {"event_names": [row[0] for row in result]}

# Endpoint to get member details based on expense date and description
@app.get("/v1/student_club/member_details_by_expense_date_and_description", operation_id="get_member_details_by_expense_date_and_description", summary="Retrieves the last name and position of club members who incurred an expense on a specific date with a given description. The date should be provided in 'YYYY-MM-DD' format.")
async def get_member_details_by_expense_date_and_description(expense_date: str = Query(..., description="Expense date in 'YYYY-MM-DD' format"), expense_description: str = Query(..., description="Description of the expense")):
    cursor.execute("SELECT T1.last_name, T1.position FROM member AS T1 INNER JOIN expense AS T2 ON T1.member_id = T2.link_to_member WHERE T2.expense_date = ? AND T2.expense_description = ?", (expense_date, expense_description))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": [{"last_name": row[0], "position": row[1]} for row in result]}

# Endpoint to get last names of members who attended a specific event with a specific position
@app.get("/v1/student_club/member_last_names_by_event_and_position", operation_id="get_member_last_names_by_event_and_position", summary="Retrieves the last names of members who attended a particular event and held a specific position. The operation requires the name of the event and the position as input parameters.")
async def get_member_last_names_by_event_and_position(event_name: str = Query(..., description="Name of the event"), position: str = Query(..., description="Position of the member")):
    cursor.execute("SELECT T3.last_name FROM attendance AS T1 INNER JOIN event AS T2 ON T2.event_id = T1.link_to_event INNER JOIN member AS T3 ON T1.link_to_member = T3.member_id WHERE T2.event_name = ? AND T3.position = ?", (event_name, position))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get the percentage of members with a specific position and t-shirt size who have a certain income amount
@app.get("/v1/student_club/income_percentage_by_position_and_tshirt_size", operation_id="get_income_percentage", summary="Retrieves the percentage of members with a specific role and t-shirt size who have a certain income level. The operation calculates this percentage by comparing the count of members with the specified income, position, and t-shirt size to the total number of members with the same position and t-shirt size.")
async def get_income_percentage(amount: int = Query(..., description="Income amount"), position: str = Query(..., description="Position of the member"), t_shirt_size: str = Query(..., description="T-shirt size of the member")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.amount = ? THEN 1.0 ELSE 0 END) AS REAL) * 100 / COUNT(T2.income_id) FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T1.position = ? AND T1.t_shirt_size = ?", (amount, position, t_shirt_size))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct counties based on zip code type
@app.get("/v1/student_club/distinct_counties_by_zip_code_type", operation_id="get_distinct_counties", summary="Retrieves a list of unique counties that have the specified zip code type. The type parameter is used to filter the results.")
async def get_distinct_counties(type: str = Query(..., description="Type of zip code")):
    cursor.execute("SELECT DISTINCT county FROM zip_code WHERE type = ? AND county IS NOT NULL", (type,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get distinct event names based on type, date range, and status
@app.get("/v1/student_club/distinct_event_names_by_type_date_status", operation_id="get_distinct_event_names", summary="Retrieve a list of unique event names that match the specified event type, date range, and status. The event type, start date, end date, and status are provided as input parameters to filter the results.")
async def get_distinct_event_names(type: str = Query(..., description="Type of event"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), status: str = Query(..., description="Status of the event")):
    cursor.execute("SELECT DISTINCT event_name FROM event WHERE type = ? AND date(SUBSTR(event_date, 1, 10)) BETWEEN ? AND ? AND status = ?", (type, start_date, end_date, status))
    result = cursor.fetchall()
    if not result:
        return {"event_names": []}
    return {"event_names": [row[0] for row in result]}

# Endpoint to get distinct event links based on expense cost
@app.get("/v1/student_club/distinct_event_links_by_expense_cost", operation_id="get_distinct_event_links", summary="Retrieves unique event links associated with members who have incurred expenses exceeding the specified cost. This operation provides a list of distinct event links, enabling users to identify events where members have spent more than the given cost.")
async def get_distinct_event_links(cost: int = Query(..., description="Expense cost")):
    cursor.execute("SELECT DISTINCT T3.link_to_event FROM expense AS T1 INNER JOIN member AS T2 ON T1.link_to_member = T2.member_id INNER JOIN attendance AS T3 ON T2.member_id = T3.link_to_member WHERE T1.cost > ?", (cost,))
    result = cursor.fetchall()
    if not result:
        return {"event_links": []}
    return {"event_links": [row[0] for row in result]}

# Endpoint to get distinct member and event links based on expense date range and approval status
@app.get("/v1/student_club/distinct_member_event_links_by_expense_date_approval", operation_id="get_distinct_member_event_links", summary="Retrieve unique member-event associations within a specified date range and approval status. This operation returns a list of distinct member-event links based on the provided start and end dates, as well as the approval status of the associated expenses. The result set includes member and event identifiers, offering insights into the distinct member-event relationships that meet the given criteria.")
async def get_distinct_member_event_links(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), approved: str = Query(..., description="Approval status")):
    cursor.execute("SELECT DISTINCT T1.link_to_member, T3.link_to_event FROM expense AS T1 INNER JOIN member AS T2 ON T1.link_to_member = T2.member_id INNER JOIN attendance AS T3 ON T2.member_id = T3.link_to_member WHERE date(SUBSTR(T1.expense_date, 1, 10)) BETWEEN ? AND ? AND T1.approved = ?", (start_date, end_date, approved))
    result = cursor.fetchall()
    if not result:
        return {"member_event_links": []}
    return {"member_event_links": [{"member_link": row[0], "event_link": row[1]} for row in result]}

# Endpoint to get college based on major ID and first name
@app.get("/v1/student_club/college_by_major_id_first_name", operation_id="get_college", summary="Retrieves the college associated with a specific major and a member's first name. The major is identified by its unique ID, and the member is identified by their first name. This operation returns the name of the college that corresponds to the provided major and member.")
async def get_college(major_id: str = Query(..., description="Major ID"), first_name: str = Query(..., description="First name of the member")):
    cursor.execute("SELECT T2.college FROM member AS T1 INNER JOIN major AS T2 ON T2.major_id = T1.link_to_major WHERE T1.link_to_major = ? AND T1.first_name = ?", (major_id, first_name))
    result = cursor.fetchone()
    if not result:
        return {"college": []}
    return {"college": result[0]}

# Endpoint to get member phone numbers based on major name and college
@app.get("/v1/student_club/member_phones_by_major_college", operation_id="get_member_phones", summary="Retrieves the phone numbers of members who are enrolled in a specific major within a given college. The major and college are provided as input parameters.")
async def get_member_phones(major_name: str = Query(..., description="Major name"), college: str = Query(..., description="College")):
    cursor.execute("SELECT T1.phone FROM member AS T1 INNER JOIN major AS T2 ON T2.major_id = T1.link_to_major WHERE T2.major_name = ? AND T2.college = ?", (major_name, college))
    result = cursor.fetchall()
    if not result:
        return {"phones": []}
    return {"phones": [row[0] for row in result]}

# Endpoint to get distinct member emails based on expense date range and cost
@app.get("/v1/student_club/distinct_member_emails_by_expense_date_cost", operation_id="get_distinct_member_emails", summary="Retrieves a list of unique member emails that have incurred expenses within a specified date range and cost threshold. The date range is defined by the start and end dates, and the cost threshold is determined by the provided expense cost. This operation is useful for identifying members who have made significant expenses within a certain period.")
async def get_distinct_member_emails(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), cost: int = Query(..., description="Expense cost")):
    cursor.execute("SELECT DISTINCT T1.email FROM member AS T1 INNER JOIN expense AS T2 ON T1.member_id = T2.link_to_member WHERE date(SUBSTR(T2.expense_date, 1, 10)) BETWEEN ? AND ? AND T2.cost > ?", (start_date, end_date, cost))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get the percentage of budgets with remaining less than 0
@app.get("/v1/student_club/percentage_budgets_remaining_less_than_zero", operation_id="get_percentage_budgets_remaining_less_than_zero", summary="Retrieves the percentage of budgets that have a remaining balance less than zero. This operation calculates the proportion of budgets with a negative remaining balance out of the total number of budgets.")
async def get_percentage_budgets_remaining_less_than_zero():
    cursor.execute("SELECT CAST(SUM(CASE WHEN remaining < 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(budget_id) FROM budget")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get event details within a date range
@app.get("/v1/student_club/event_details_by_date_range", operation_id="get_event_details_by_date_range", summary="Retrieves event details, including event ID, location, and status, for events that fall within the specified date range. The date range is defined by the provided start and end dates, both in 'YYYY-MM-DD' format.")
async def get_event_details_by_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT event_id, location, status FROM event WHERE date(SUBSTR(event_date, 1, 10)) BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"events": []}
    return {"events": result}

# Endpoint to get expense descriptions with average cost greater than a specified value
@app.get("/v1/student_club/expense_descriptions_by_avg_cost", operation_id="get_expense_descriptions_by_avg_cost", summary="Retrieves expense descriptions for which the average cost exceeds a specified threshold. This operation groups expenses by their descriptions and filters out those with an average cost below the provided threshold.")
async def get_expense_descriptions_by_avg_cost(avg_cost: float = Query(..., description="Average cost threshold")):
    cursor.execute("SELECT expense_description FROM expense GROUP BY expense_description HAVING AVG(cost) > ?", (avg_cost,))
    result = cursor.fetchall()
    if not result:
        return {"expense_descriptions": []}
    return {"expense_descriptions": result}

# Endpoint to get member details based on t-shirt size
@app.get("/v1/student_club/member_details_by_t_shirt_size", operation_id="get_member_details_by_t_shirt_size", summary="Retrieves the first and last names of members who wear a specific t-shirt size. The t-shirt size is provided as an input parameter.")
async def get_member_details_by_t_shirt_size(t_shirt_size: str = Query(..., description="T-shirt size")):
    cursor.execute("SELECT first_name, last_name FROM member WHERE t_shirt_size = ?", (t_shirt_size,))
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": result}

# Endpoint to get the percentage of zip codes of a specific type
@app.get("/v1/student_club/percentage_zip_codes_by_type", operation_id="get_percentage_zip_codes_by_type", summary="Retrieves the percentage of zip codes of a specified type from the database. The operation calculates the proportion of zip codes that match the provided type out of the total number of zip codes.")
async def get_percentage_zip_codes_by_type(zip_type: str = Query(..., description="Type of zip code")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN type = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(zip_code) FROM zip_code", (zip_type,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct event names and locations with remaining budget greater than a specified value
@app.get("/v1/student_club/event_details_by_remaining_budget", operation_id="get_event_details_by_remaining_budget", summary="Retrieves unique event names and their respective locations where the remaining budget surpasses a specified threshold. This operation is useful for identifying events with substantial budgets that may require further attention or management.")
async def get_event_details_by_remaining_budget(remaining_budget: float = Query(..., description="Remaining budget threshold")):
    cursor.execute("SELECT DISTINCT T1.event_name, T1.location FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T2.remaining > ?", (remaining_budget,))
    result = cursor.fetchall()
    if not result:
        return {"events": []}
    return {"events": result}

# Endpoint to get event details based on expense description and cost range
@app.get("/v1/student_club/event_details_by_expense_description_and_cost_range", operation_id="get_event_details_by_expense_description_and_cost_range", summary="Retrieves event details for events that have expenses matching the provided description and falling within the specified cost range. The response includes the event name and date.")
async def get_event_details_by_expense_description_and_cost_range(expense_description: str = Query(..., description="Expense description"), min_cost: float = Query(..., description="Minimum cost"), max_cost: float = Query(..., description="Maximum cost")):
    cursor.execute("SELECT T1.event_name, T1.event_date FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget WHERE T3.expense_description = ? AND T3.cost > ? AND T3.cost < ?", (expense_description, min_cost, max_cost))
    result = cursor.fetchall()
    if not result:
        return {"events": []}
    return {"events": result}

# Endpoint to get member details with expenses greater than a specified value
@app.get("/v1/student_club/member_details_by_expense_cost", operation_id="get_member_details_by_expense_cost", summary="Retrieves unique member details, including their first name, last name, and major, for members who have incurred expenses exceeding the specified minimum cost threshold.")
async def get_member_details_by_expense_cost(min_cost: float = Query(..., description="Minimum cost threshold")):
    cursor.execute("SELECT DISTINCT T1.first_name, T1.last_name, T2.major_name FROM member AS T1 INNER JOIN major AS T2 ON T2.major_id = T1.link_to_major INNER JOIN expense AS T3 ON T1.member_id = T3.link_to_member WHERE T3.cost > ?", (min_cost,))
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": result}

# Endpoint to get distinct cities and counties based on income amount
@app.get("/v1/student_club/cities_counties_by_income_amount", operation_id="get_cities_counties_by_income_amount", summary="Retrieves a list of unique cities and counties associated with members who have an income greater than the specified minimum amount.")
async def get_cities_counties_by_income_amount(min_amount: float = Query(..., description="Minimum income amount")):
    cursor.execute("SELECT DISTINCT T3.city, T3.county FROM income AS T1 INNER JOIN member AS T2 ON T1.link_to_member = T2.member_id INNER JOIN zip_code AS T3 ON T3.zip_code = T2.zip WHERE T1.amount > ?", (min_amount,))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": result}

# Endpoint to get the member ID with the highest total expense cost across multiple events
@app.get("/v1/student_club/member_id_highest_total_expense", operation_id="get_member_id_highest_total_expense", summary="Retrieves the member ID with the highest total expense cost across multiple events, provided that the member has participated in at least a specified minimum number of distinct events.")
async def get_member_id_highest_total_expense(min_events: int = Query(..., description="Minimum number of distinct events")):
    cursor.execute("SELECT T2.member_id FROM expense AS T1 INNER JOIN member AS T2 ON T1.link_to_member = T2.member_id INNER JOIN budget AS T3 ON T1.link_to_budget = T3.budget_id INNER JOIN event AS T4 ON T3.link_to_event = T4.event_id GROUP BY T2.member_id HAVING COUNT(DISTINCT T4.event_id) > ? ORDER BY SUM(T1.cost) DESC LIMIT 1", (min_events,))
    result = cursor.fetchone()
    if not result:
        return {"member_id": []}
    return {"member_id": result[0]}

# Endpoint to get the average cost of expenses for members not in a specific position
@app.get("/v1/student_club/average_expense_excluding_position", operation_id="get_average_expense_excluding_position", summary="Retrieves the average cost of expenses incurred by members who do not hold a specific position. The position to exclude is provided as an input parameter.")
async def get_average_expense_excluding_position(position: str = Query(..., description="Position of the member to exclude")):
    cursor.execute("SELECT AVG(T1.cost) FROM expense AS T1 INNER JOIN member as T2 ON T1.link_to_member = T2.member_id WHERE T2.position != ?", (position,))
    result = cursor.fetchone()
    if not result:
        return {"average_cost": []}
    return {"average_cost": result[0]}

# Endpoint to get event names with expenses below the average cost for a specific budget category
@app.get("/v1/student_club/event_names_below_average_cost", operation_id="get_event_names_below_average_cost", summary="Retrieves the names of events that have expenses below the average cost for a specified budget category. The operation compares the cost of each event to the average cost across all expenses, and returns the names of events that meet this criteria for the given budget category.")
async def get_event_names_below_average_cost(category: str = Query(..., description="Budget category")):
    cursor.execute("SELECT T1.event_name FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget WHERE T2.category = ? AND T3.cost < (SELECT AVG(cost) FROM expense)", (category,))
    result = cursor.fetchall()
    if not result:
        return {"event_names": []}
    return {"event_names": [row[0] for row in result]}

# Endpoint to get the percentage of total expenses for a specific event type
@app.get("/v1/student_club/percentage_expenses_event_type", operation_id="get_percentage_expenses_event_type", summary="Retrieves the proportion of total expenses allocated to a specific event type. This operation calculates the percentage by summing up the costs associated with the specified event type and dividing it by the total expenses across all events. The event type is provided as an input parameter.")
async def get_percentage_expenses_event_type(event_type: str = Query(..., description="Type of the event")):
    cursor.execute("SELECT SUM(CASE WHEN T1.type = ? THEN T3.cost ELSE 0 END) * 100 / SUM(T3.cost) FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event INNER JOIN expense AS T3 ON T2.budget_id = T3.link_to_budget", (event_type,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the budget ID with the highest expense for a specific description
@app.get("/v1/student_club/highest_expense_budget_id", operation_id="get_highest_expense_budget_id", summary="Retrieves the budget ID associated with the highest expense for a given description. The operation filters expenses based on the provided description and identifies the budget with the maximum cost. This endpoint is useful for tracking the most expensive budget item for a specific expense type.")
async def get_highest_expense_budget_id(expense_description: str = Query(..., description="Description of the expense")):
    cursor.execute("SELECT T2.budget_id FROM expense AS T1 INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id WHERE T1.expense_description = ? ORDER BY T1.cost DESC LIMIT 1", (expense_description,))
    result = cursor.fetchone()
    if not result:
        return {"budget_id": []}
    return {"budget_id": result[0]}

# Endpoint to get the top 5 members by spent budget
@app.get("/v1/student_club/top_members_by_spent_budget", operation_id="get_top_members_by_spent_budget", summary="Retrieves the top five members who have spent the most from their respective budgets. The members are identified by their first and last names. The budget spent is calculated based on the expenses linked to each member and their associated budgets.")
async def get_top_members_by_spent_budget():
    cursor.execute("SELECT T3.first_name, T3.last_name FROM expense AS T1 INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id INNER JOIN member AS T3 ON T1.link_to_member = T3.member_id ORDER BY T2.spent DESC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get members with expenses above the average cost
@app.get("/v1/student_club/members_above_average_expense", operation_id="get_members_above_average_expense", summary="Retrieves a list of unique student club members who have incurred expenses exceeding the average cost. The response includes each member's first name, last name, and phone number.")
async def get_members_above_average_expense():
    cursor.execute("SELECT DISTINCT T3.first_name, T3.last_name, T3.phone FROM expense AS T1 INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id INNER JOIN member AS T3 ON T3.member_id = T1.link_to_member WHERE T1.cost > ( SELECT AVG(T1.cost) FROM expense AS T1 INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id INNER JOIN member AS T3 ON T3.member_id = T1.link_to_member )")
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": [{"first_name": row[0], "last_name": row[1], "phone": row[2]} for row in result]}

# Endpoint to get the percentage difference in members between two states
@app.get("/v1/student_club/percentage_difference_members_states", operation_id="get_percentage_difference_members_states", summary="Retrieves the percentage difference in the number of student club members between two specified states. The calculation is based on the total count of members and their respective states, which are provided as input parameters.")
async def get_percentage_difference_members_states(state1: str = Query(..., description="First state"), state2: str = Query(..., description="Second state")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T2.state = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.state = ? THEN 1 ELSE 0 END)) AS REAL) * 100 / COUNT(T1.member_id) AS diff FROM member AS T1 INNER JOIN zip_code AS T2 ON T2.zip_code = T1.zip", (state1, state2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the major and department of a member by their name
@app.get("/v1/student_club/major_department_by_name", operation_id="get_major_department_by_name", summary="Retrieves the major and department associated with a specific member using their first and last name. The operation searches for a member with the provided first and last name and returns the corresponding major and department information.")
async def get_major_department_by_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.major_name, T2.department FROM member AS T1 INNER JOIN major AS T2 ON T2.major_id = T1.link_to_major WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"major_department": []}
    return {"major_name": result[0], "department": result[1]}

# Endpoint to get members and their expenses for a specific expense description
@app.get("/v1/student_club/members_expenses_by_description", operation_id="get_members_expenses_by_description", summary="Retrieves the first name, last name, and cost associated with members who have incurred a specific expense. The expense description is used to filter the results.")
async def get_members_expenses_by_description(expense_description: str = Query(..., description="Description of the expense")):
    cursor.execute("SELECT T2.first_name, T2.last_name, T1.cost FROM expense AS T1 INNER JOIN member AS T2 ON T1.link_to_member = T2.member_id WHERE T1.expense_description = ?", (expense_description,))
    result = cursor.fetchall()
    if not result:
        return {"members_expenses": []}
    return {"members_expenses": [{"first_name": row[0], "last_name": row[1], "cost": row[2]} for row in result]}

# Endpoint to get members' last names and phone numbers by major name
@app.get("/v1/student_club/members_by_major", operation_id="get_members_by_major", summary="Retrieves the last names and phone numbers of student club members who are associated with a specific major. The major is identified by its name.")
async def get_members_by_major(major_name: str = Query(..., description="Name of the major")):
    cursor.execute("SELECT T1.last_name, T1.phone FROM member AS T1 INNER JOIN major AS T2 ON T2.major_id = T1.link_to_major WHERE T2.major_name = ?", (major_name,))
    result = cursor.fetchall()
    if not result:
        return {"members": []}
    return {"members": [{"last_name": row[0], "phone": row[1]} for row in result]}

# Endpoint to get budget categories and amounts for a specific event
@app.get("/v1/student_club/budget_categories_amounts_by_event", operation_id="get_budget_categories_amounts", summary="Retrieves the budget categories and their corresponding amounts for a specific event. The event is identified by its name, which is provided as an input parameter. The operation returns a list of categories and their respective budget allocations.")
async def get_budget_categories_amounts(event_name: str = Query(..., description="Name of the event")):
    cursor.execute("SELECT T2.category, T2.amount FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_name = ?", (event_name,))
    result = cursor.fetchall()
    if not result:
        return {"categories_amounts": []}
    return {"categories_amounts": result}

# Endpoint to get event names based on budget category
@app.get("/v1/student_club/event_names_by_budget_category", operation_id="get_event_names_by_category", summary="Retrieves the names of events that fall under a specified budget category. The category is used to filter the events and return only those that match the provided category.")
async def get_event_names_by_category(category: str = Query(..., description="Budget category")):
    cursor.execute("SELECT T1.event_name FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event WHERE T2.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"event_names": []}
    return {"event_names": result}

# Endpoint to get member names and income amounts based on date received
@app.get("/v1/student_club/member_income_by_date", operation_id="get_member_income_by_date", summary="Retrieves the unique first and last names of members along with their respective income amounts for a specific date. The date is provided in 'YYYY-MM-DD' format.")
async def get_member_income_by_date(date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T3.first_name, T3.last_name, T4.amount FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event INNER JOIN member AS T3 ON T3.member_id = T2.link_to_member INNER JOIN income AS T4 ON T4.link_to_member = T3.member_id WHERE T4.date_received = ?", (date_received,))
    result = cursor.fetchall()
    if not result:
        return {"member_income": []}
    return {"member_income": result}

# Endpoint to get budget categories based on expense description
@app.get("/v1/student_club/budget_categories_by_expense_description", operation_id="get_budget_categories_by_expense_description", summary="Retrieves the distinct budget categories associated with a given expense description. The operation filters the expense records based on the provided expense description and identifies the unique budget categories linked to these expenses.")
async def get_budget_categories_by_expense_description(expense_description: str = Query(..., description="Expense description")):
    cursor.execute("SELECT DISTINCT T2.category FROM expense AS T1 INNER JOIN budget AS T2 ON T1.link_to_budget = T2.budget_id WHERE T1.expense_description = ?", (expense_description,))
    result = cursor.fetchall()
    if not result:
        return {"budget_categories": []}
    return {"budget_categories": result}

# Endpoint to get total spent amount by event name based on budget category
@app.get("/v1/student_club/total_spent_by_event_category", operation_id="get_total_spent_by_event_category", summary="Retrieves the total amount spent on events, grouped by event name, for a specific budget category. The category parameter is used to filter the results.")
async def get_total_spent_by_event_category(category: str = Query(..., description="Budget category")):
    cursor.execute("SELECT SUM(T1.spent), T2.event_name FROM budget AS T1 INNER JOIN event AS T2 ON T1.link_to_event = T2.event_id WHERE T1.category = ? GROUP BY T2.event_name", (category,))
    result = cursor.fetchall()
    if not result:
        return {"total_spent": []}
    return {"total_spent": result}

# Endpoint to get city based on member's first and last name
@app.get("/v1/student_club/city_by_member_name", operation_id="get_city_by_member_name", summary="Retrieves the city associated with a specific member, identified by their first and last name. The operation uses the provided names to search for a match in the member database and returns the corresponding city from the zip code database.")
async def get_city_by_member_name(first_name: str = Query(..., description="First name of the member"), last_name: str = Query(..., description="Last name of the member")):
    cursor.execute("SELECT T2.city FROM member AS T1 INNER JOIN zip_code AS T2 ON T2.zip_code = T1.zip WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"city": []}
    return {"city": result}

# Endpoint to get member details based on city, state, and zip code
@app.get("/v1/student_club/member_details_by_location", operation_id="get_member_details_by_location", summary="Retrieves the first name, last name, and position of members from a specific city, state, and zip code. The operation filters members based on the provided location parameters, enabling targeted member information retrieval.")
async def get_member_details_by_location(city: str = Query(..., description="City"), state: str = Query(..., description="State"), zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT T1.first_name, T1.last_name, T1.position FROM member AS T1 INNER JOIN zip_code AS T2 ON T2.zip_code = T1.zip WHERE T2.city = ? AND T2.state = ? AND T2.zip_code = ?", (city, state, zip_code))
    result = cursor.fetchall()
    if not result:
        return {"member_details": []}
    return {"member_details": result}

api_calls = [
    "/v1/student_club/major_name_by_member_name?first_name=Angela&last_name=Sanders",
    "/v1/student_club/member_count_by_college?college=College%20of%20Engineering",
    "/v1/student_club/member_names_by_department?department=Art%20and%20Design%20Department",
    "/v1/student_club/event_count_by_name?event_name=Women%27s%20Soccer",
    "/v1/student_club/member_phones_by_event_name?event_name=Women%27s%20Soccer",
    "/v1/student_club/event_count_by_name_and_tshirt_size?event_name=Women%27s%20Soccer&t_shirt_size=Medium",
    "/v1/student_club/most_attended_event",
    "/v1/student_club/colleges_by_position?position=vice%20president",
    "/v1/student_club/event_names_by_member_name?first_name=Maya&last_name=Mclean",
    "/v1/student_club/event_count_by_member_name_and_year?first_name=Sacha&last_name=Harrison&year=2019",
    "/v1/student_club/event_names_attendance_count_exclude_type?attendance_count=10&event_type=Meeting",
    "/v1/student_club/attendance_ratio_year_type?year=2020&event_type=Meeting",
    "/v1/student_club/most_expensive_expense?limit=1",
    "/v1/student_club/member_count_by_major?major_name=Environmental%20Engineering",
    "/v1/student_club/member_names_by_event?event_name=Laugh%20Out%20Loud",
    "/v1/student_club/member_last_names_by_major?major_name=Law%20and%20Constitutional%20Studies",
    "/v1/student_club/member_county_by_name?first_name=Sherri&last_name=Ramsey",
    "/v1/student_club/member_college_by_name?first_name=Tyler&last_name=Hewitt",
    "/v1/student_club/member_income_by_position?position=Vice%20President",
    "/v1/student_club/event_spent_category_month?event_name=September%20Meeting&category=Food&month=09",
    "/v1/student_club/member_location_by_position?position=President",
    "/v1/student_club/member_names_by_state?state=Illinois",
    "/v1/student_club/member_departments_by_last_name?last_name1=Pierce&last_name2=Guidi",
    "/v1/student_club/total_spent_event?event_name=October%20Speaker",
    "/v1/student_club/expense_approval_status?event_name=October%20Meeting&event_date=2019-10-08",
    "/v1/student_club/average_expense_cost_member_months?last_name=Allen&first_name=Elijah&month1=09&month2=10",
    "/v1/student_club/spending_difference_years?year1=2019&year2=2020",
    "/v1/student_club/event_location?event_name=Spring%20Budget%20Review",
    "/v1/student_club/expense_cost?expense_description=Posters&expense_date=2019-09-04",
    "/v1/student_club/remaining_budget_max_amount?category=Food",
    "/v1/student_club/income_notes?source=Fundraising&date_received=2019-09-14",
    "/v1/student_club/major_count?college=College%20of%20Humanities%20and%20Social%20Sciences",
    "/v1/student_club/member_phone?first_name=Carlo&last_name=Jacobs",
    "/v1/student_club/event_count_remaining_budget?event_name=November%20Meeting&remaining_budget=0",
    "/v1/student_club/total_budget_event?event_name=September%20Speaker",
    "/v1/student_club/event_status?expense_description=Post%20Cards%2C%20Posters&expense_date=2019-08-20",
    "/v1/student_club/count_members_by_major_tshirt_size?major_name=Business&t_shirt_size=Medium",
    "/v1/student_club/get_zip_code_type_by_name?first_name=Christof&last_name=Nielson",
    "/v1/student_club/get_major_name_by_position?position=Vice%20President",
    "/v1/student_club/get_state_by_name?first_name=Sacha&last_name=Harrison",
    "/v1/student_club/get_department_by_position?position=President",
    "/v1/student_club/get_date_received_by_name_source?first_name=Connor&last_name=Hilton&source=Dues",
    "/v1/student_club/get_member_name_by_source?source=Dues",
    "/v1/student_club/get_budget_ratio_by_events?event_name_1=Yearly%20Kickoff&event_name_2=October%20Meeting&category=Advertisement&event_type=Meeting",
    "/v1/student_club/get_budget_percentage_by_category?category=Parking&event_name=November%20Speaker",
    "/v1/student_club/get_total_cost_by_description?expense_description=Pizza",
    "/v1/student_club/count_cities_in_county_state?county=Orange%20County&state=Virginia",
    "/v1/student_club/departments_by_college?college=College%20of%20Humanities%20and%20Social%20Sciences",
    "/v1/student_club/member_location?first_name=Amy&last_name=Firth",
    "/v1/student_club/first_expense_by_remaining_budget",
    "/v1/student_club/attendees_by_event?event_name=October%20Meeting",
    "/v1/student_club/most_popular_college_by_major",
    "/v1/student_club/major_by_phone?phone=809-555-3360",
    "/v1/student_club/highest_budget_event",
    "/v1/student_club/expenses_by_position?position=Vice%20President",
    "/v1/student_club/attendee_count_by_event?event_name=Women%27s%20Soccer",
    "/v1/student_club/date_received_by_name?first_name=Casey&last_name=Mason",
    "/v1/student_club/member_count_by_state?state=Maryland",
    "/v1/student_club/event_count_by_phone?phone=954-555-6240",
    "/v1/student_club/event_with_highest_budget_utilization?status=Closed",
    "/v1/student_club/member_count_by_position?position=President",
    "/v1/student_club/max_spent_budget",
    "/v1/student_club/event_count_by_type_and_year?event_type=Meeting&year=2020",
    "/v1/student_club/total_spent_by_category?category=Food",
    "/v1/student_club/members_by_event_attendance?min_events=7",
    "/v1/student_club/member_names_by_event_and_major?event_name=Community%20Theater&major_name=Interior%20Design",
    "/v1/student_club/member_names_by_city_and_state?city=Georgetown&state=South%20Carolina",
    "/v1/student_club/income_amount_by_member_name?first_name=Grant&last_name=Gilmour",
    "/v1/student_club/member_names_by_income?amount=40",
    "/v1/student_club/total_expense_by_event?event_name=Yearly%20Kickoff",
    "/v1/student_club/member_names_by_event_expenses?event_name=Yearly%20Kickoff",
    "/v1/student_club/top_income_member",
    "/v1/student_club/highest_expense_event",
    "/v1/student_club/expense_percentage_by_event?event_name=Yearly%20Kickoff",
    "/v1/student_club/major_ratio?major_name_1=Finance&major_name_2=Physics",
    "/v1/student_club/income_source_by_date?start_date=2019-09-01&end_date=2019-09-30",
    "/v1/student_club/member_details_by_position?position=Secretary",
    "/v1/student_club/attendee_count_by_event_year?event_name=Community%20Theater&year=2019",
    "/v1/student_club/event_count_by_member_major?first_name=Luisa&last_name=Guidi",
    "/v1/student_club/average_spent_by_category_status?category=Food&event_status=Closed",
    "/v1/student_club/highest_spent_event_by_category?category=Advertisement",
    "/v1/student_club/check_member_attendance?event_name=Women%27s%20Soccer&first_name=Maya&last_name=Mclean",
    "/v1/student_club/percentage_events_by_type_year?event_type=Community%20Service&year=2019",
    "/v1/student_club/expense_cost_by_event?event_name=September%20Speaker&expense_description=Posters",
    "/v1/student_club/most_common_tshirt_size",
    "/v1/student_club/event_with_lowest_remaining_budget?event_status=Closed&remaining_threshold=0",
    "/v1/student_club/total_expense_by_event_name?event_name=October%20Meeting",
    "/v1/student_club/budget_by_category_event_name?event_name=April%20Speaker",
    "/v1/student_club/max_budget_by_category?category=Food",
    "/v1/student_club/top_3_budgets_by_category?category=Advertisement",
    "/v1/student_club/total_expense_by_date?expense_date=2019-08-20",
    "/v1/student_club/total_expense_by_member_id?member_id=rec4BLdZHS2Blfp4v",
    "/v1/student_club/expense_descriptions_by_member_name?first_name=Sacha&last_name=Harrison",
    "/v1/student_club/expense_descriptions_by_t_shirt_size?t_shirt_size=X-Large",
    "/v1/student_club/zip_codes_by_expense_cost?cost=50",
    "/v1/student_club/major_names_by_member_name?first_name=Phillip&last_name=Cullen",
    "/v1/student_club/positions_by_major_name?major_name=Business",
    "/v1/student_club/count_members_by_major_and_t_shirt_size?major_name=Business&t_shirt_size=Medium",
    "/v1/student_club/event_types_by_remaining_budget?remaining=30",
    "/v1/student_club/budget_categories_by_event_location?location=MU%20215",
    "/v1/student_club/budget_categories_by_event_date?event_date=2020-03-24T12:00:00",
    "/v1/student_club/major_names_by_position?position=Vice%20President",
    "/v1/student_club/percentage_members_by_position_and_major?position=Member&major_name=Business",
    "/v1/student_club/distinct_budget_categories_by_location?location=MU%20215",
    "/v1/student_club/count_income_by_amount?amount=50",
    "/v1/student_club/count_members_by_position_and_tshirt_size?position=Member&t_shirt_size=X-Large",
    "/v1/student_club/count_majors_by_department_and_college?department=School%20of%20Applied%20Sciences%2C%20Technology%20and%20Education&college=College%20of%20Agriculture%20and%20Applied%20Sciences",
    "/v1/student_club/member_details_by_position_and_major?position=Member&major_name=Environmental%20Engineering",
    "/v1/student_club/distinct_categories_and_types_by_location_spent_and_type?location=MU%20215&spent=0&event_type=Guest%20Speaker",
    "/v1/student_club/city_state_by_department_and_position?department=Electrical%20and%20Computer%20Engineering%20Department&position=Member",
    "/v1/student_club/event_names_by_position_location_and_type?position=Vice%20President&location=900%20E.%20Washington%20St.&event_type=Social",
    "/v1/student_club/member_details_by_expense_date_and_description?expense_date=2019-09-10&expense_description=Pizza",
    "/v1/student_club/member_last_names_by_event_and_position?event_name=Women%27s%20Soccer&position=Member",
    "/v1/student_club/income_percentage_by_position_and_tshirt_size?amount=50&position=Member&t_shirt_size=Medium",
    "/v1/student_club/distinct_counties_by_zip_code_type?type=PO%20Box",
    "/v1/student_club/distinct_event_names_by_type_date_status?type=Game&start_date=2019-03-15&end_date=2020-03-20&status=Closed",
    "/v1/student_club/distinct_event_links_by_expense_cost?cost=50",
    "/v1/student_club/distinct_member_event_links_by_expense_date_approval?start_date=2019-01-10&end_date=2019-11-19&approved=true",
    "/v1/student_club/college_by_major_id_first_name?major_id=rec1N0upiVLy5esTO&first_name=Katy",
    "/v1/student_club/member_phones_by_major_college?major_name=Business&college=College%20of%20Agriculture%20and%20Applied%20Sciences",
    "/v1/student_club/distinct_member_emails_by_expense_date_cost?start_date=2019-09-10&end_date=2019-11-19&cost=20",
    "/v1/student_club/percentage_budgets_remaining_less_than_zero",
    "/v1/student_club/event_details_by_date_range?start_date=2019-11-01&end_date=2020-03-31",
    "/v1/student_club/expense_descriptions_by_avg_cost?avg_cost=50",
    "/v1/student_club/member_details_by_t_shirt_size?t_shirt_size=X-Large",
    "/v1/student_club/percentage_zip_codes_by_type?zip_type=PO%20Box",
    "/v1/student_club/event_details_by_remaining_budget?remaining_budget=0",
    "/v1/student_club/event_details_by_expense_description_and_cost_range?expense_description=Pizza&min_cost=50&max_cost=100",
    "/v1/student_club/member_details_by_expense_cost?min_cost=100",
    "/v1/student_club/cities_counties_by_income_amount?min_amount=40",
    "/v1/student_club/member_id_highest_total_expense?min_events=1",
    "/v1/student_club/average_expense_excluding_position?position=Member",
    "/v1/student_club/event_names_below_average_cost?category=Parking",
    "/v1/student_club/percentage_expenses_event_type?event_type=Meeting",
    "/v1/student_club/highest_expense_budget_id?expense_description=Water,%20chips,%20cookies",
    "/v1/student_club/top_members_by_spent_budget",
    "/v1/student_club/members_above_average_expense",
    "/v1/student_club/percentage_difference_members_states?state1=New%20Jersey&state2=Vermont",
    "/v1/student_club/major_department_by_name?first_name=Garrett&last_name=Gerke",
    "/v1/student_club/members_expenses_by_description?expense_description=Water,%20Veggie%20tray,%20supplies",
    "/v1/student_club/members_by_major?major_name=Elementary%20Education",
    "/v1/student_club/budget_categories_amounts_by_event?event_name=January%20Speaker",
    "/v1/student_club/event_names_by_budget_category?category=Food",
    "/v1/student_club/member_income_by_date?date_received=2019-09-09",
    "/v1/student_club/budget_categories_by_expense_description?expense_description=Posters",
    "/v1/student_club/total_spent_by_event_category?category=Speaker%20Gifts",
    "/v1/student_club/city_by_member_name?first_name=Garrett&last_name=Gerke",
    "/v1/student_club/member_details_by_location?city=Lincolnton&state=North%20Carolina&zip_code=28092"
]
