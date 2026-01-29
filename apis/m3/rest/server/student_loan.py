from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/student_loan/student_loan.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of names from longest_absense_from_school for a specific month
@app.get("/v1/student_loan/count_names_by_month", operation_id="get_count_names_by_month", summary="Retrieves the total count of unique student names who have been absent from school for the longest duration during a specific month. The month is provided as an input parameter.")
async def get_count_names_by_month(month: int = Query(..., description="Month value")):
    cursor.execute("SELECT COUNT(name) FROM longest_absense_from_school WHERE `month` = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum month value from longest_absense_from_school
@app.get("/v1/student_loan/max_month", operation_id="get_max_month", summary="Retrieves the highest month value from the longest absence from school records.")
async def get_max_month():
    cursor.execute("SELECT MAX(month) FROM longest_absense_from_school")
    result = cursor.fetchone()
    if not result:
        return {"max_month": []}
    return {"max_month": result[0]}

# Endpoint to get the count of names from enlist for a specific organ
@app.get("/v1/student_loan/count_names_by_organ", operation_id="get_count_names_by_organ", summary="Retrieves the total number of unique names associated with a specific organ in the enlistment data.")
async def get_count_names_by_organ(organ: str = Query(..., description="Organ value")):
    cursor.execute("SELECT COUNT(name) FROM enlist WHERE organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of names from longest_absense_from_school and male for a specific month threshold
@app.get("/v1/student_loan/count_names_by_month_threshold", operation_id="get_count_names_by_month_threshold", summary="Retrieves the count of names that have been absent from school for a month equal to or greater than the specified threshold. The data is filtered based on the longest absence from school and male records.")
async def get_count_names_by_month_threshold(month_threshold: int = Query(..., description="Month threshold value")):
    cursor.execute("SELECT COUNT(T1.name) FROM longest_absense_from_school AS T1 INNER JOIN male AS T2 ON T1.`name` = T2.`name` WHERE T1.`month` >= ?", (month_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names from enlist and male for a specific organ
@app.get("/v1/student_loan/names_by_organ", operation_id="get_names_by_organ", summary="Retrieves the names of students who are enlisted and male, associated with a specific organ. The organ is provided as an input parameter.")
async def get_names_by_organ(organ: str = Query(..., description="Organ value")):
    cursor.execute("SELECT T1.name FROM enlist AS T1 INNER JOIN male AS T2 ON T1.`name` = T2.`name` WHERE T1.organ = ?", (organ,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of names from filed_for_bankrupcy and longest_absense_from_school for a specific month threshold
@app.get("/v1/student_loan/count_names_by_month_threshold_bankrupcy", operation_id="get_count_names_by_month_threshold_bankrupcy", summary="Retrieves the total count of individuals who have filed for bankruptcy and have been absent from school for more than a specified number of months. The count is based on the provided month threshold.")
async def get_count_names_by_month_threshold_bankrupcy(month_threshold: int = Query(..., description="Month threshold value")):
    cursor.execute("SELECT COUNT(T1.name) FROM filed_for_bankrupcy AS T1 INNER JOIN longest_absense_from_school AS T2 ON T1.`name` = T2.`name` WHERE T2.`month` > ?", (month_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of names from enlist and no_payment_due for a specific organ and boolean value
@app.get("/v1/student_loan/count_names_by_organ_and_bool", operation_id="get_count_names_by_organ_and_bool", summary="Retrieves the count of student names that are enlisted and have no payment due for a specific organ and boolean value. The organ and boolean value are provided as input parameters.")
async def get_count_names_by_organ_and_bool(organ: str = Query(..., description="Organ value"), bool_value: str = Query(..., description="Boolean value (pos or neg)")):
    cursor.execute("SELECT COUNT(T1.name) FROM enlist AS T1 INNER JOIN no_payment_due AS T2 ON T1.`name` = T2.`name` WHERE T1.organ = ? AND T2.bool = ?", (organ, bool_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names from disabled, male, and filed_for_bankrupcy
@app.get("/v1/student_loan/names_from_disabled_male_bankrupcy", operation_id="get_names_from_disabled_male_bankrupcy", summary="Retrieves the names of individuals who are disabled, male, and have filed for bankruptcy. The operation returns a list of names that meet all three criteria by performing a join operation on the relevant tables.")
async def get_names_from_disabled_male_bankrupcy():
    cursor.execute("SELECT T1.name, T2.name, T3.name FROM disabled AS T1 INNER JOIN male AS T2 ON T1.`name` = T2.`name` INNER JOIN filed_for_bankrupcy AS T3 ON T1.`name` = T3.`name`")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"disabled": row[0], "male": row[1], "bankrupcy": row[2]} for row in result]}

# Endpoint to get the count of names from disabled that are not in male
@app.get("/v1/student_loan/count_names_not_in_male", operation_id="get_count_names_not_in_male", summary="Retrieves the total count of unique names from the disabled list that do not appear in the male list.")
async def get_count_names_not_in_male():
    cursor.execute("SELECT COUNT(name) FROM disabled WHERE name NOT IN ( SELECT name FROM male )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of names from unemployed and no_payment_due
@app.get("/v1/student_loan/count_names_unemployed_no_payment_due", operation_id="get_count_names_unemployed_no_payment_due", summary="Retrieves the total count of individuals who are both unemployed and have no payment due on their student loans. This operation combines data from the 'unemployed' and 'no_payment_due' tables, using the 'name' field as a common identifier to ensure accurate results.")
async def get_count_names_unemployed_no_payment_due():
    cursor.execute("SELECT COUNT(T1.name) FROM unemployed AS T1 INNER JOIN no_payment_due AS T2 ON T1.`name` = T2.`name`")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the month of the longest absence from school for a given organization
@app.get("/v1/student_loan/longest_absence_month_by_organ", operation_id="get_longest_absence_month", summary="Retrieves the month of the longest absence from school for a specific organization. The operation identifies the student with the longest absence from school within the given organization and returns the corresponding month. The input parameter specifies the organization name.")
async def get_longest_absence_month(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT T1.month FROM longest_absense_from_school AS T1 INNER JOIN enlist AS T2 ON T1.`name` = T2.`name` WHERE T2.organ = ? ORDER BY T1.`month` DESC LIMIT 1", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"month": []}
    return {"month": result[0]}

# Endpoint to get the count of names from longest absence from school for a given month
@app.get("/v1/student_loan/count_names_longest_absence_by_month", operation_id="get_count_names_longest_absence", summary="Retrieves the count of students who have been absent the longest from school for a given month. The month is specified as an input parameter.")
async def get_count_names_longest_absence(month: int = Query(..., description="Month of absence")):
    cursor.execute("SELECT COUNT(T1.name) FROM longest_absense_from_school AS T1 INNER JOIN disabled AS T2 ON T1.`name` = T2.`name` WHERE T1.`month` = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the organization of students with a given month of longest absence
@app.get("/v1/student_loan/organ_by_longest_absence_month", operation_id="get_organ_by_longest_absence_month", summary="Retrieves the organization of students who have been absent for the longest duration in a specified month. The month of absence is provided as an input parameter.")
async def get_organ_by_longest_absence_month(month: int = Query(..., description="Month of absence")):
    cursor.execute("SELECT T2.organ FROM longest_absense_from_school AS T1 INNER JOIN enlist AS T2 ON T1.`name` = T2.`name` WHERE T1.`month` = ?", (month,))
    result = cursor.fetchall()
    if not result:
        return {"organ": []}
    return {"organ": [row[0] for row in result]}

# Endpoint to get the organization with the highest count of disabled students
@app.get("/v1/student_loan/top_organ_by_disabled_count", operation_id="get_top_organ_by_disabled_count", summary="Retrieves the organization with the highest number of disabled students. This operation fetches data from the enlist table, filters for students with a corresponding entry in the disabled table, and groups the results by organization. The organization with the most disabled students is then returned.")
async def get_top_organ_by_disabled_count():
    cursor.execute("SELECT T2.organ, COUNT(T1.name) FROM disabled AS T1 INNER JOIN enlist AS T2 ON T1.`name` = T2.`name` GROUP BY T2.organ ORDER BY COUNT(T1.name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"organ": [], "count": []}
    return {"organ": result[0], "count": result[1]}

# Endpoint to get names of individuals who filed for bankruptcy but are not male
@app.get("/v1/student_loan/bankruptcy_non_male", operation_id="get_bankruptcy_non_male", summary="Retrieves the names of individuals who have filed for bankruptcy but are not identified as male. This operation does not return any data for individuals who are male or have not filed for bankruptcy.")
async def get_bankruptcy_non_male():
    cursor.execute("SELECT name FROM filed_for_bankrupcy WHERE name NOT IN ( SELECT name FROM male )")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the percentage of males in a given organization
@app.get("/v1/student_loan/percentage_males_by_organ", operation_id="get_percentage_males_by_organ", summary="Retrieves the percentage of males in a specified organization. The calculation is based on the total number of individuals in the organization and the count of males within that group.")
async def get_percentage_males_by_organ(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT CAST(COUNT(T2.name) AS REAL) * 100 / COUNT(T1.name) FROM enlist AS T1 LEFT JOIN male AS T2 ON T1.`name` = T2.`name` WHERE T1.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average month of longest absence from school for disabled students
@app.get("/v1/student_loan/avg_longest_absence_month_disabled", operation_id="get_avg_longest_absence_month_disabled", summary="Retrieves the average duration of the longest absence from school for students with disabilities.")
async def get_avg_longest_absence_month_disabled():
    cursor.execute("SELECT AVG(T1.month) FROM longest_absense_from_school AS T1 INNER JOIN disabled AS T2 ON T1.`name` = T2.`name`")
    result = cursor.fetchone()
    if not result:
        return {"average_month": []}
    return {"average_month": result[0]}

# Endpoint to get the payment due status for a given name
@app.get("/v1/student_loan/payment_due_status", operation_id="get_payment_due_status", summary="Retrieves the payment due status for a specific individual. The operation checks if the individual has any outstanding payments and returns a boolean value indicating whether a payment is due or not. The individual's name is used as the input parameter to determine the payment due status.")
async def get_payment_due_status(name: str = Query(..., description="Name of the individual")):
    cursor.execute("SELECT bool FROM no_payment_due WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"payment_due": []}
    return {"payment_due": result[0]}

# Endpoint to get the school of a given enrolled student
@app.get("/v1/student_loan/school_by_name", operation_id="get_school_by_name", summary="Get the school of a given enrolled student")
async def get_school_by_name(name: str = Query(..., description="Name of the student")):
    cursor.execute("SELECT school FROM enrolled WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"school": []}
    return {"school": result[0]}

# Endpoint to get the names of students with the longest absence in a given month
@app.get("/v1/student_loan/longest_absence_names", operation_id="get_longest_absence_names", summary="Retrieves the names of students who have been absent for the longest duration during a specified month. The month parameter is used to filter the results.")
async def get_longest_absence_names(month: int = Query(..., description="Month of the absence")):
    cursor.execute("SELECT name FROM longest_absense_from_school WHERE `month` = ?", (month,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the organization of a given student name
@app.get("/v1/student_loan/student_organization", operation_id="get_student_organization", summary="Retrieves the organization associated with a specific student. The operation requires the student's name as input and returns the corresponding organization from the enlistment records.")
async def get_student_organization(name: str = Query(..., description="Name of the student")):
    cursor.execute("SELECT organ FROM enlist WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"organ": []}
    return {"organ": result[0]}

# Endpoint to get the school and name of a disabled student
@app.get("/v1/student_loan/disabled_student_info", operation_id="get_disabled_student_info", summary="Retrieves the name and school of a disabled student, based on the provided student name.")
async def get_disabled_student_info(name: str = Query(..., description="Name of the student")):
    cursor.execute("SELECT T2.name, T1.school FROM enrolled AS T1 INNER JOIN disabled AS T2 ON T1.`name` = T2.`name` WHERE T1.name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"name": [], "school": []}
    return {"name": result[0], "school": result[1]}

# Endpoint to get the names and schools of students enlisted in a specific organization
@app.get("/v1/student_loan/enlisted_students_info", operation_id="get_enlisted_students_info", summary="Retrieves the names and schools of students who are enrolled and enlisted in a specified organization.")
async def get_enlisted_students_info(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT T1.name, T1.school FROM enrolled AS T1 INNER JOIN enlist AS T2 ON T1.`name` = T2.`name` WHERE T2.organ = ?", (organ,))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": [{"name": row[0], "school": row[1]} for row in result]}

# Endpoint to get the names of non-male students with a specific payment status
@app.get("/v1/student_loan/non_male_payment_status", operation_id="get_non_male_payment_status", summary="Retrieves the names of students who identify as non-male and have a specified payment status. The payment status is determined by the provided boolean value, where true indicates a positive payment status and false indicates a negative payment status.")
async def get_non_male_payment_status(bool: str = Query(..., description="Payment status (neg or pos)")):
    cursor.execute("SELECT T1.name FROM no_payment_due AS T1 INNER JOIN person AS T2 ON T1.`name` = T2.`name` WHERE T2.`name` NOT IN ( SELECT name FROM male ) AND T1.bool = ?", (bool,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the organization and names of students with a specific payment status, limited to 5 results
@app.get("/v1/student_loan/payment_status_organ_names", operation_id="get_payment_status_organ_names", summary="Retrieves the names of up to 5 students and their respective organizations who have a specified payment status. The payment status is determined by a boolean value, where a true value indicates no payment is due, and a false value indicates a payment is due.")
async def get_payment_status_organ_names(bool: str = Query(..., description="Payment status (neg or pos)")):
    cursor.execute("SELECT T2.organ, T1.name FROM no_payment_due AS T1 INNER JOIN enlist AS T2 ON T1.`name` = T2.`name` WHERE T1.bool = ? LIMIT 5", (bool,))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": [{"organ": row[0], "name": row[1]} for row in result]}

# Endpoint to get the names of disabled students enrolled in a specific school
@app.get("/v1/student_loan/disabled_students_by_school", operation_id="get_disabled_students_by_school", summary="Retrieves the names of disabled students who are enrolled in the specified school. The operation filters the list of enrolled students by the provided school name and returns the corresponding names from the disabled students' records.")
async def get_disabled_students_by_school(school: str = Query(..., description="Name of the school")):
    cursor.execute("SELECT T2.name FROM enrolled AS T1 INNER JOIN disabled AS T2 ON T1.`name` = T2.`name` WHERE T1.school = ?", (school,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the names of students who filed for bankruptcy and are enlisted in a specific organization
@app.get("/v1/student_loan/bankruptcy_enlisted_students", operation_id="get_bankruptcy_enlisted_students", summary="Retrieves the names of students who have filed for bankruptcy and are currently enrolled in a specified organization. The operation requires the organization name as an input parameter to filter the results.")
async def get_bankruptcy_enlisted_students(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT T2.name FROM enlist AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T1.`name` = T2.`name` WHERE T1.organ = ?", (organ,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of male students with a specific payment status
@app.get("/v1/student_loan/male_payment_status_count", operation_id="get_male_payment_status_count", summary="Retrieves the total count of male students who have a specified payment status. The payment status is determined by the provided boolean value, which indicates whether the student has a payment due or not.")
async def get_male_payment_status_count(bool: str = Query(..., description="Payment status (neg or pos)")):
    cursor.execute("SELECT COUNT(T1.name) FROM no_payment_due AS T1 INNER JOIN male AS T2 ON T1.name = T2.name WHERE T1.bool = ?", (bool,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names from longest_absense_from_school and enlist based on organ and month
@app.get("/v1/student_loan/names_by_organ_and_month", operation_id="get_names_by_organ_and_month", summary="Retrieves the names of students with the longest absence from school, filtered by a specific organization and month. The names are obtained from the longest_absense_from_school table and cross-referenced with the enlist table to ensure they belong to the provided organization. The month parameter is used to further narrow down the results.")
async def get_names_by_organ_and_month(organ: str = Query(..., description="Organization name"), month: int = Query(..., description="Month")):
    cursor.execute("SELECT T1.name FROM longest_absense_from_school AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ? AND T1.month = ?", (organ, month))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get names from enrolled and unemployed based on school
@app.get("/v1/student_loan/names_by_school", operation_id="get_names_by_school", summary="Retrieves the names of students who are both enrolled and unemployed at the specified school.")
async def get_names_by_school(school: str = Query(..., description="School name")):
    cursor.execute("SELECT T2.name FROM enrolled AS T1 INNER JOIN unemployed AS T2 ON T1.name = T2.name WHERE T1.school = ?", (school,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get names from longest_absense_from_school and unemployed based on month
@app.get("/v1/student_loan/names_by_month", operation_id="get_names_by_month", summary="Retrieves the names of individuals who have the longest absence from school and are unemployed for a specific month. The month is provided as an input parameter.")
async def get_names_by_month(month: int = Query(..., description="Month")):
    cursor.execute("SELECT T1.name FROM longest_absense_from_school AS T1 INNER JOIN unemployed AS T2 ON T1.name = T2.name WHERE T1.month = ?", (month,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of names from enlist and enrolled based on school and organ
@app.get("/v1/student_loan/count_names_by_school_and_organ", operation_id="get_count_names_by_school_and_organ", summary="Retrieves the total number of students who are both enlisted and enrolled in a specific school and organization. The count is based on matching student names in the enlist and enrolled records. The operation requires the name of the school and the organization as input parameters.")
async def get_count_names_by_school_and_organ(school: str = Query(..., description="School name"), organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM enlist AS T1 INNER JOIN enrolled AS T2 ON T1.name = T2.name WHERE T2.school = ? AND T1.organ = ?", (school, organ))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get month and organ from enlist and longest_absense_from_school based on name
@app.get("/v1/student_loan/month_and_organ_by_name", operation_id="get_month_and_organ_by_name", summary="Retrieves the month and organ associated with a given name from the enlist and longest_absense_from_school tables. The name parameter is used to filter the results.")
async def get_month_and_organ_by_name(name: str = Query(..., description="Name")):
    cursor.execute("SELECT T2.month, T1.organ FROM enlist AS T1 INNER JOIN longest_absense_from_school AS T2 ON T1.name = T2.name WHERE T1.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": [{"month": row[0], "organ": row[1]} for row in result]}

# Endpoint to get the percentage of positive to negative bool values from no_payment_due and enlist based on organ
@app.get("/v1/student_loan/percentage_positive_to_negative_by_organ", operation_id="get_percentage_positive_to_negative_by_organ", summary="Retrieves the percentage ratio of positive to negative values from the no_payment_due table, filtered by a specific organization from the enlist table. The input parameter is the name of the organization.")
async def get_percentage_positive_to_negative_by_organ(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.bool = 'pos', 1, 0)) AS REAL) * 100 / SUM(IIF(T1.bool = 'neg', 1, 0)) FROM no_payment_due AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of students from a specific school in enrolled and enlist based on organ
@app.get("/v1/student_loan/percentage_students_by_school_and_organ", operation_id="get_percentage_students_by_school_and_organ", summary="Retrieves the percentage of students from a specified school who are enrolled and enlisted in a particular organization. The calculation is based on the total number of students enrolled and enlisted in the given organization.")
async def get_percentage_students_by_school_and_organ(school: str = Query(..., description="School name"), organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.school = ?, 1.0, 0)) AS REAL) * 100 / COUNT(T1.name) FROM enrolled AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ?", (school, organ))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get names from longest_absense_from_school for the maximum month
@app.get("/v1/student_loan/names_for_max_month", operation_id="get_names_for_max_month", summary="Retrieves the names of students with the longest absence from school during the month with the maximum recorded absences.")
async def get_names_for_max_month():
    cursor.execute("SELECT name FROM longest_absense_from_school WHERE month = ( SELECT MAX(month) FROM longest_absense_from_school )")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of names from person
@app.get("/v1/student_loan/count_names", operation_id="get_count_names", summary="Retrieves the total count of unique names from the person database.")
async def get_count_names():
    cursor.execute("SELECT COUNT(name) FROM person")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names and month from longest_absense_from_school for the maximum month
@app.get("/v1/student_loan/names_and_month_for_max_month", operation_id="get_names_and_month_for_max_month", summary="Retrieves the names of students and the corresponding month with the longest absence from school, specifically the month with the maximum number of absences.")
async def get_names_and_month_for_max_month():
    cursor.execute("SELECT name, month FROM longest_absense_from_school WHERE month = ( SELECT MAX(month) FROM longest_absense_from_school )")
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": [{"name": row[0], "month": row[1]} for row in result]}

# Endpoint to get the count of unemployed individuals who filed for bankruptcy
@app.get("/v1/student_loan/count_unemployed_bankruptcy", operation_id="get_count_unemployed_bankruptcy", summary="Retrieves the total number of individuals who are both unemployed and have filed for bankruptcy.")
async def get_count_unemployed_bankruptcy():
    cursor.execute("SELECT COUNT(T1.name) FROM unemployed AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T1.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct organizations from disabled and enlisted individuals
@app.get("/v1/student_loan/distinct_organizations", operation_id="get_distinct_organizations", summary="Retrieves a list of unique organizations associated with individuals who are both disabled and enlisted. The data is obtained by cross-referencing the 'disabled' and 'enlist' tables using the individual's name as the common identifier.")
async def get_distinct_organizations():
    cursor.execute("SELECT DISTINCT T2.organ FROM disabled AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name")
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": [row[0] for row in result]}

# Endpoint to get the count of unemployed individuals who filed for bankruptcy and have no payment due based on a boolean value
@app.get("/v1/student_loan/count_unemployed_bankruptcy_no_payment", operation_id="get_count_unemployed_bankruptcy_no_payment", summary="Retrieves the count of unemployed individuals who have filed for bankruptcy and have no payment due, based on the provided boolean value.")
async def get_count_unemployed_bankruptcy_no_payment(bool_value: str = Query(..., description="Boolean value as a string ('pos' or 'neg')")):
    cursor.execute("SELECT COUNT(T1.name) FROM unemployed AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T1.name = T2.name INNER JOIN no_payment_due AS T3 ON T2.name = T3.name WHERE T3.bool = ?", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to determine gender based on the presence in the disabled table
@app.get("/v1/student_loan/determine_gender", operation_id="determine_gender", summary="This operation determines the gender of a student based on their presence in a specific table. It returns 'female' if the individual's name is not found in the table, and 'male' otherwise. The operation requires the name of the individual as input.")
async def determine_gender(name: str = Query(..., description="Name of the individual")):
    cursor.execute("SELECT IIF(T2.name IS NULL, 'female', 'male') FROM male AS T1 LEFT JOIN disabled AS T2 ON T1.name = T2.name WHERE T1.name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get counts of individuals in different categories based on names
@app.get("/v1/student_loan/counts_by_names", operation_id="get_counts_by_names", summary="Retrieves the counts of individuals with specific names in four distinct categories: disabled, unemployed, male, and those with no payment due. The operation accepts two input parameters, each representing a name to be checked against the respective categories.")
async def get_counts_by_names(name1: str = Query(..., description="First name to check"), name2: str = Query(..., description="Second name to check")):
    cursor.execute("SELECT ( SELECT COUNT(name) FROM disabled WHERE name IN (?, ?) ), ( SELECT COUNT(name) FROM unemployed WHERE name IN (?, ?) ), ( SELECT COUNT(name) FROM male WHERE name IN (?, ?) ), ( SELECT COUNT(name) FROM no_payment_due WHERE name IN (?, ?))", (name1, name2, name1, name2, name1, name2, name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"counts": []}
    return {"counts": result}

# Endpoint to get the count of enlisted individuals in specific organizations not in the male table
@app.get("/v1/student_loan/count_enlisted_not_male", operation_id="get_count_enlisted_not_male", summary="Retrieves the total number of individuals enlisted in the specified organizations who are not listed in the male table. The response includes a count of individuals who meet these criteria.")
async def get_count_enlisted_not_male(organ1: str = Query(..., description="First organization"), organ2: str = Query(..., description="Second organization")):
    cursor.execute("SELECT COUNT(name) FROM enlist WHERE organ IN (?, ?) AND name NOT IN ( SELECT name FROM male )", (organ1, organ2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get school and organization for enrolled and enlisted individuals based on names
@app.get("/v1/student_loan/school_organ_by_names", operation_id="get_school_organ_by_names", summary="Get school and organization for enrolled and enlisted individuals based on names")
async def get_school_organ_by_names(name1: str = Query(..., description="First name to check"), name2: str = Query(..., description="Second name to check"), name3: str = Query(..., description="Third name to check")):
    cursor.execute("SELECT T1.school, T2.organ FROM enrolled AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T1.name IN (?, ?, ?)", (name1, name2, name3))
    result = cursor.fetchall()
    if not result:
        return {"school_organ": []}
    return {"school_organ": [{"school": row[0], "organ": row[1]} for row in result]}

# Endpoint to get the percentage of disabled individuals not in the male table
@app.get("/v1/student_loan/percentage_disabled_not_male", operation_id="get_percentage_disabled_not_male", summary="Retrieves the percentage of disabled individuals who are not listed in the male table. This operation calculates the ratio of disabled individuals without a corresponding entry in the male table to the total number of disabled individuals.")
async def get_percentage_disabled_not_male():
    cursor.execute("SELECT CAST(SUM(IIF(T2.name IS NULL, 1, 0)) AS REAL) * 100 / COUNT(T2.name) FROM disabled AS T1 LEFT JOIN male AS T2 ON T1.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of individuals not in the male or enrolled tables
@app.get("/v1/student_loan/count_not_male_enrolled", operation_id="get_count_not_male_enrolled", summary="Retrieves the total count of individuals who are neither male nor enrolled in the system.")
async def get_count_not_male_enrolled():
    cursor.execute("SELECT COUNT(name) FROM person WHERE name NOT IN ( SELECT name FROM male ) AND name NOT IN ( SELECT name FROM enrolled )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names of individuals who filed for bankruptcy and have a specific payment status
@app.get("/v1/student_loan/bankruptcy_payment_status", operation_id="get_bankruptcy_payment_status", summary="Retrieves the names of individuals who have filed for bankruptcy and have a specified payment status. The payment status is determined by the provided boolean value, which indicates whether a payment is due or not.")
async def get_bankruptcy_payment_status(bool_status: str = Query(..., description="Payment status (e.g., 'neg' for negative)")):
    cursor.execute("SELECT T1.name FROM filed_for_bankrupcy AS T1 INNER JOIN no_payment_due AS T2 ON T1.name = T2.name WHERE T2.bool = ?", (bool_status,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the average month of longest absence from school
@app.get("/v1/student_loan/average_absence_month", operation_id="get_average_absence_month", summary="Retrieves the average month of the longest absence from school, calculated based on the longest continuous period of absence for each student.")
async def get_average_absence_month():
    cursor.execute("SELECT AVG(month) FROM longest_absense_from_school")
    result = cursor.fetchone()
    if not result:
        return {"average_month": []}
    return {"average_month": result[0]}

# Endpoint to get the average month of longest absence from school for individuals not unemployed
@app.get("/v1/student_loan/average_absence_month_not_unemployed", operation_id="get_average_absence_month_not_unemployed", summary="Retrieves the average month of the longest absence from school for individuals who are not currently unemployed.")
async def get_average_absence_month_not_unemployed():
    cursor.execute("SELECT AVG(month) FROM longest_absense_from_school WHERE name NOT IN ( SELECT name FROM unemployed )")
    result = cursor.fetchone()
    if not result:
        return {"average_month": []}
    return {"average_month": result[0]}

# Endpoint to get the average month of longest absence from school for disabled individuals
@app.get("/v1/student_loan/average_absence_month_disabled", operation_id="get_average_absence_month_disabled", summary="Retrieves the average month of the longest absence from school for individuals with disabilities. This operation calculates the average month from the longest absence data and the disabled individuals data.")
async def get_average_absence_month_disabled():
    cursor.execute("SELECT AVG(T1.month) FROM longest_absense_from_school AS T1 INNER JOIN disabled AS T2 ON T1.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"average_month": []}
    return {"average_month": result[0]}

# Endpoint to get the count of individuals with a specific month of longest absence from school
@app.get("/v1/student_loan/count_absence_by_month", operation_id="get_count_absence_by_month", summary="Retrieves the total number of individuals who had their longest absence from school during a specified month. The month is provided as an input parameter.")
async def get_count_absence_by_month(month: int = Query(..., description="Month of longest absence from school")):
    cursor.execute("SELECT COUNT(name) FROM longest_absense_from_school WHERE month = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of individuals with a longest absence from school greater than a specific month
@app.get("/v1/student_loan/count_absence_greater_than_month", operation_id="get_count_absence_greater_than_month", summary="Retrieves the total number of individuals who have been absent from school for a period longer than the specified month. The input parameter determines the minimum duration of absence to be considered.")
async def get_count_absence_greater_than_month(month: int = Query(..., description="Month of longest absence from school")):
    cursor.execute("SELECT COUNT(name) FROM longest_absense_from_school WHERE month > ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of individuals with a specific payment status
@app.get("/v1/student_loan/count_payment_status", operation_id="get_count_payment_status", summary="Retrieves the total number of individuals who have a specified payment status in their student loan records. The payment status is indicated by a boolean value, where a true value corresponds to a positive payment status and a false value corresponds to a negative payment status.")
async def get_count_payment_status(bool_status: str = Query(..., description="Payment status (e.g., 'neg' for negative)")):
    cursor.execute("SELECT COUNT(name) FROM no_payment_due WHERE bool = ?", (bool_status,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of disabled individuals with a specific month of longest absence from school
@app.get("/v1/student_loan/disabled_longest_absence_names", operation_id="get_disabled_longest_absence_names", summary="Retrieves the names of disabled individuals who were absent from school for the longest duration during a specified month. The month of longest absence is provided as an input parameter.")
async def get_disabled_longest_absence_names(month: int = Query(..., description="Month of longest absence from school")):
    cursor.execute("SELECT T1.name FROM disabled AS T1 INNER JOIN longest_absense_from_school AS T2 ON T1.name = T2.name WHERE T2.month = ?", (month,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of unemployed individuals enlisted in a specific organization
@app.get("/v1/student_loan/unemployed_enlist_count", operation_id="get_unemployed_enlist_count", summary="Retrieves the total number of unemployed individuals who are enlisted in a specified organization. The operation requires the name of the organization as an input parameter.")
async def get_unemployed_enlist_count(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM unemployed AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of male individuals enlisted in a specific organization
@app.get("/v1/student_loan/male_enlist_count", operation_id="get_male_enlist_count", summary="Retrieves the total number of male individuals enlisted in a specified organization.")
async def get_male_enlist_count(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM male AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of individuals enlisted in a specific organization who are not male
@app.get("/v1/student_loan/non_male_enlist_count", operation_id="get_non_male_enlist_count", summary="Retrieves the total count of individuals enlisted in a specified organization, excluding those identified as male.")
async def get_non_male_enlist_count(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(name) FROM enlist WHERE organ = ? AND name NOT IN ( SELECT name FROM male )", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of the disabled individual with the longest absence from school
@app.get("/v1/student_loan/disabled_longest_absence_name", operation_id="get_disabled_longest_absence_name", summary="Retrieves the name of the disabled individual who has been absent from school for the longest duration. The operation uses a join query to correlate data from the 'disabled' and 'longest_absence_from_school' tables, ordering the results by the length of absence in descending order and returning the top result.")
async def get_disabled_longest_absence_name():
    cursor.execute("SELECT T1.name FROM disabled AS T1 INNER JOIN longest_absense_from_school AS T2 ON T1.name = T2.name ORDER BY T2.month DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the names of unemployed individuals enlisted in a specific organization
@app.get("/v1/student_loan/unemployed_enlist_names", operation_id="get_unemployed_enlist_names", summary="Retrieves the names of unemployed individuals who are enlisted in a specified organization. The operation requires the organization's name as an input parameter to filter the results.")
async def get_unemployed_enlist_names(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT T1.name FROM unemployed AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ?", (organ,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the percentage of unemployed individuals with an absence from school longer than a specific number of months
@app.get("/v1/student_loan/unemployed_absence_percentage", operation_id="get_unemployed_absence_percentage", summary="Retrieves the percentage of unemployed individuals who have been absent from school for more than the specified number of months. This operation calculates the ratio of individuals with an absence longer than the given duration to the total number of unemployed individuals.")
async def get_unemployed_absence_percentage(month: int = Query(..., description="Number of months")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.month > ?, 1, 0)) AS REAL) * 100 / COUNT(T1.month) FROM longest_absense_from_school AS T1 INNER JOIN unemployed AS T2 ON T1.name = T2.name", (month,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of unemployed and disabled individuals with a specific month of longest absence from school
@app.get("/v1/student_loan/unemployed_disabled_absence_count", operation_id="get_unemployed_disabled_absence_count", summary="Retrieves the count of individuals who are both unemployed and disabled, and have a specific month as their longest absence from school.")
async def get_unemployed_disabled_absence_count(month: int = Query(..., description="Month of longest absence from school")):
    cursor.execute("SELECT COUNT(T1.name) FROM longest_absense_from_school AS T1 INNER JOIN unemployed AS T2 ON T1.name = T2.name INNER JOIN disabled AS T3 ON T2.name = T3.name WHERE T1.month = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of the unemployed individual with the longest absence from school
@app.get("/v1/student_loan/unemployed_longest_absence_name", operation_id="get_unemployed_longest_absence_name", summary="Retrieves the name of the unemployed individual who has been absent from school for the longest duration. The operation identifies the individual with the longest absence by comparing the duration of absence for all unemployed individuals and returns the name of the individual with the maximum duration.")
async def get_unemployed_longest_absence_name():
    cursor.execute("SELECT T1.name FROM longest_absense_from_school AS T1 INNER JOIN unemployed AS T2 ON T1.name = T2.name ORDER BY T1.month DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of students absent from school for a given month
@app.get("/v1/student_loan/count_absent_students_by_month", operation_id="get_count_absent_students_by_month", summary="Retrieves the total number of students who were absent from school during a specified month. The count is based on the longest absence period and includes students who are disabled.")
async def get_count_absent_students_by_month(month: int = Query(..., description="Month of absence")):
    cursor.execute("SELECT COUNT(T1.name) FROM longest_absense_from_school AS T1 INNER JOIN disabled AS T2 ON T1.name = T2.name WHERE T1.month = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of students absent from school for a given month who filed for bankruptcy
@app.get("/v1/student_loan/absent_students_filed_bankruptcy_by_month", operation_id="get_absent_students_filed_bankruptcy_by_month", summary="Retrieves the names of students who were absent from school for a specified month and have filed for bankruptcy. The month of absence is a required input parameter.")
async def get_absent_students_filed_bankruptcy_by_month(month: int = Query(..., description="Month of absence")):
    cursor.execute("SELECT T1.name FROM longest_absense_from_school AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T1.name = T2.name WHERE T1.month = ?", (month,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the name of the student with the longest absence who filed for bankruptcy
@app.get("/v1/student_loan/longest_absent_student_filed_bankruptcy", operation_id="get_longest_absent_student_filed_bankruptcy", summary="Retrieves the name of the student who has been absent the longest and has filed for bankruptcy. This operation identifies the student with the longest absence duration from school records and checks if they have filed for bankruptcy. The result is the name of the student who meets both criteria.")
async def get_longest_absent_student_filed_bankruptcy():
    cursor.execute("SELECT T1.name FROM longest_absense_from_school AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T1.name = T2.name ORDER BY T1.month DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to determine the gender of a student based on their name
@app.get("/v1/student_loan/determine_gender_by_name", operation_id="determine_gender_by_name", summary="This operation determines the gender of a student by analyzing their name. It returns either 'female' or 'male' based on the name provided as input. The input parameter is the student's name.")
async def determine_gender_by_name(name: str = Query(..., description="Name of the student")):
    cursor.execute("SELECT IIF(T.result = 0, 'female', 'male') AS re FROM ( SELECT COUNT(name) AS result FROM male WHERE name = ? ) T", (name,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the count of disabled students
@app.get("/v1/student_loan/count_disabled_students", operation_id="get_count_disabled_students", summary="Retrieves the total number of disabled students. This operation provides a count of students with disabilities, offering a quantitative overview of the student population.")
async def get_count_disabled_students():
    cursor.execute("SELECT COUNT(name) FROM disabled")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of students enlisted in specific organizations
@app.get("/v1/student_loan/count_enlisted_students_by_organs", operation_id="get_count_enlisted_students_by_organs", summary="Retrieves the total number of students enrolled in up to three specified organizations. The response provides a consolidated count of students across the selected organizations.")
async def get_count_enlisted_students_by_organs(organ1: str = Query(..., description="First organization name"), organ2: str = Query(..., description="Second organization name"), organ3: str = Query(..., description="Third organization name")):
    cursor.execute("SELECT COUNT(name) FROM enlist WHERE organ IN (?, ?, ?)", (organ1, organ2, organ3))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of disabled students enlisted in a specific organization
@app.get("/v1/student_loan/count_disabled_enlisted_students_by_organ", operation_id="get_count_disabled_enlisted_students_by_organ", summary="Retrieves the total number of students with disabilities who are enlisted in a specified organization. The operation requires the organization name as input to filter the count accordingly.")
async def get_count_disabled_enlisted_students_by_organ(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM enlist AS T1 INNER JOIN disabled AS T2 ON T1.name = T2.name WHERE T1.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of unemployed individuals with a specific boolean value in no_payment_due
@app.get("/v1/student_loan/unemployed_count_by_bool", operation_id="get_unemployed_count_by_bool", summary="Retrieves the count of unemployed individuals who have a specific boolean value in their no_payment_due status. The boolean value is provided as an input parameter.")
async def get_unemployed_count_by_bool(bool_value: str = Query(..., description="Boolean value in no_payment_due")):
    cursor.execute("SELECT COUNT(T1.name) FROM unemployed AS T1 INNER JOIN no_payment_due AS T2 ON T1.name = T2.name WHERE T2.bool = ?", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of individuals with the longest absence from school in a specific month
@app.get("/v1/student_loan/longest_absence_names_by_month", operation_id="get_longest_absence_names_by_month", summary="Retrieves the names of individuals who have had the longest absence from school in a given month. The month is specified as an input parameter.")
async def get_longest_absence_names_by_month(month: int = Query(..., description="Month of the longest absence from school")):
    cursor.execute("SELECT T2.name FROM male AS T1 INNER JOIN longest_absense_from_school AS T2 ON T1.name <> T2.name WHERE T2.month = ?", (month,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the school with the highest number of disabled enrolled students
@app.get("/v1/student_loan/school_with_most_disabled_enrolled", operation_id="get_school_with_most_disabled_enrolled", summary="Retrieves the school with the highest number of enrolled students who have disabilities. The operation calculates the count of disabled students enrolled in each school and returns the school with the maximum count.")
async def get_school_with_most_disabled_enrolled():
    cursor.execute("SELECT T.school FROM ( SELECT T2.school, COUNT(T2.name) AS num FROM disabled AS T1 INNER JOIN enrolled AS T2 ON T1.name = T2.name GROUP BY T2.school ) T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"school": []}
    return {"school": result[0]}

# Endpoint to get the organizations of individuals who filed for bankruptcy and are enlisted
@app.get("/v1/student_loan/organizations_of_bankrupt_enlisted", operation_id="get_organizations_of_bankrupt_enlisted", summary="Retrieves a list of organizations associated with individuals who have filed for bankruptcy and are currently enlisted. The data is obtained by cross-referencing the names of bankrupt individuals with those who are enlisted.")
async def get_organizations_of_bankrupt_enlisted():
    cursor.execute("SELECT T2.organ FROM filed_for_bankrupcy AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name")
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": [row[0] for row in result]}

# Endpoint to get the count of males enlisted in more than a specified number of organizations
@app.get("/v1/student_loan/count_males_enlisted_in_multiple_organizations", operation_id="get_count_males_enlisted_in_multiple_organizations", summary="Retrieves the count of male students who are enlisted in more than the specified number of organizations. This operation provides a quantitative measure of male students' involvement in multiple organizations.")
async def get_count_males_enlisted_in_multiple_organizations(num_organizations: int = Query(..., description="Number of organizations")):
    cursor.execute("SELECT COUNT(T.a) FROM ( SELECT COUNT(DISTINCT T1.name) AS a, COUNT(T2.organ) AS num FROM male AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name GROUP BY T1.name ) T WHERE T.num > ?", (num_organizations,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of disabled individuals enlisted in a specific organization
@app.get("/v1/student_loan/disabled_enlisted_in_organization", operation_id="get_disabled_enlisted_in_organization", summary="Retrieves the names of disabled individuals who are enlisted in a specified organization. The operation requires the organization's name as an input parameter to filter the results.")
async def get_disabled_enlisted_in_organization(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT T1.name FROM disabled AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ?", (organ,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of enrolled students with the longest absence from a specific school in a specific month
@app.get("/v1/student_loan/enrolled_longest_absence_count_by_school_month", operation_id="get_enrolled_longest_absence_count_by_school_month", summary="Retrieves the count of students who have been enrolled in a specific school and have had the longest absence from that school in a particular month.")
async def get_enrolled_longest_absence_count_by_school_month(school: str = Query(..., description="School name"), month: int = Query(..., description="Month of the longest absence from school")):
    cursor.execute("SELECT COUNT(T1.name) FROM enrolled AS T1 INNER JOIN longest_absense_from_school AS T2 ON T1.name = T2.name WHERE T1.school = ? AND T2.month = ?", (school, month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of disabled individuals who are not male
@app.get("/v1/student_loan/disabled_not_male", operation_id="get_disabled_not_male", summary="Retrieves the names of individuals who are identified as disabled but not male. This operation returns a list of names that meet these criteria, excluding those who are also identified as male.")
async def get_disabled_not_male():
    cursor.execute("SELECT T1.name FROM disabled AS T1 INNER JOIN male AS T2 ON T1.name <> T2.name")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the ratio of unemployed to disabled individuals
@app.get("/v1/student_loan/unemployed_to_disabled_ratio", operation_id="get_unemployed_to_disabled_ratio", summary="Retrieves the ratio of unemployed individuals to disabled individuals. This operation calculates the ratio by dividing the total count of unemployed individuals by the total count of disabled individuals.")
async def get_unemployed_to_disabled_ratio():
    cursor.execute("SELECT CAST(( SELECT COUNT(name) FROM unemployed ) AS REAL ) / ( SELECT COUNT(name) FROM disabled )")
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the percentage of males enlisted in a specific organization
@app.get("/v1/student_loan/percentage_males_enlisted_in_organization", operation_id="get_percentage_males_enlisted_in_organization", summary="Retrieves the percentage of males enrolled in a specific organization. The calculation is based on the total number of enlistees in the organization and the count of males within that group. The organization is identified by its name.")
async def get_percentage_males_enlisted_in_organization(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT CAST(COUNT(T2.name) AS REAL) * 100 / COUNT(T1.name) FROM enlist AS T1 LEFT JOIN male AS T2 ON T1.name = T2.name WHERE T1.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of names based on the school
@app.get("/v1/student_loan/count_names_by_school", operation_id="get_count_names_by_school", summary="Retrieves the total number of unique student names enrolled in a specific school.")
async def get_count_names_by_school(school: str = Query(..., description="Name of the school")):
    cursor.execute("SELECT COUNT(name) FROM enrolled WHERE school = ?", (school,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names ordered by month in descending order with a limit
@app.get("/v1/student_loan/names_ordered_by_month", operation_id="get_names_ordered_by_month", summary="Retrieves a specified number of student names from the database, sorted in descending order by the month of their longest absence from school.")
async def get_names_ordered_by_month(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT name FROM longest_absense_from_school ORDER BY month DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of names based on the boolean value
@app.get("/v1/student_loan/count_names_by_bool", operation_id="get_count_names_by_bool", summary="Retrieves the count of names that match a specified boolean value from the male table, which is joined with the no_payment_due table. The boolean value is provided as a string input ('neg' or 'pos').")
async def get_count_names_by_bool(bool_value: str = Query(..., description="Boolean value as a string ('neg' or 'pos')")):
    cursor.execute("SELECT COUNT(T1.name) FROM male AS T1 INNER JOIN no_payment_due AS T2 ON T1.name = T2.name WHERE T2.bool = ?", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of names based on organization and school
@app.get("/v1/student_loan/count_names_by_organ_and_school", operation_id="get_count_names_by_organ_and_school", summary="Retrieves the total number of students enlisted in a specific organization and enrolled in a particular school. The operation joins data from the enlist and enrolled tables based on matching student names and filters results by the provided organization and school names.")
async def get_count_names_by_organ_and_school(organ: str = Query(..., description="Organization name"), school: str = Query(..., description="School name")):
    cursor.execute("SELECT COUNT(T1.name) FROM enlist AS T1 INNER JOIN enrolled AS T2 ON T1.name = T2.name WHERE T1.organ = ? AND T2.school = ?", (organ, school))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of names based on the month
@app.get("/v1/student_loan/count_names_by_month_unemployed", operation_id="get_count_names_by_month_unemployed", summary="Retrieves the total count of individuals who have been unemployed for a specified number of months. This operation joins data from the longest_absence_from_school and unemployed tables, filtering by the provided month value.")
async def get_count_names_by_month_unemployed(month: int = Query(..., description="Month as an integer")):
    cursor.execute("SELECT COUNT(T2.name) FROM longest_absense_from_school AS T1 INNER JOIN unemployed AS T2 ON T2.name = T1.name WHERE T1.month = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names based on boolean value and organization with a limit
@app.get("/v1/student_loan/names_by_bool_and_organ", operation_id="get_names_by_bool_and_organ", summary="Retrieves names from the no_payment_due table that match a specified boolean value and organization. The results are limited to a specified number. The data is joined with the enlist table to ensure accurate and relevant information.")
async def get_names_by_bool_and_organ(bool_value: str = Query(..., description="Boolean value as a string ('neg' or 'pos')"), organ: str = Query(..., description="Organization name"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.name FROM no_payment_due AS T1 INNER JOIN enlist AS T2 ON T2.name = T1.name WHERE T1.bool = ? AND T2.organ = ? LIMIT ?", (bool_value, organ, limit))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of names in enlist who are unemployed based on organ
@app.get("/v1/student_loan/count_names_enlist_unemployed", operation_id="get_count_names_enlist_unemployed", summary="Retrieves the total count of individuals who are both enlisted and unemployed, belonging to a specific organization. The operation requires the organization name as input to filter the results.")
async def get_count_names_enlist_unemployed(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM enlist AS T1 INNER JOIN unemployed AS T2 ON T2.name = T1.name WHERE T1.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of names in no_payment_due who are unemployed based on bool
@app.get("/v1/student_loan/count_names_no_payment_due_unemployed", operation_id="get_count_names_no_payment_due_unemployed", summary="Retrieves the count of individuals who have no loan payments due and are currently unemployed, based on the provided boolean value.")
async def get_count_names_no_payment_due_unemployed(bool_value: str = Query(..., description="Boolean value (pos or neg)")):
    cursor.execute("SELECT COUNT(T1.name) FROM no_payment_due AS T1 INNER JOIN unemployed AS T2 ON T2.name = T1.name WHERE T1.bool = ?", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names from no_payment_due who are enlisted based on organ and bool
@app.get("/v1/student_loan/names_no_payment_due_enlisted", operation_id="get_names_no_payment_due_enlisted", summary="Retrieves the names of up to five individuals who have no payment due and are enlisted in a specific organization, based on a provided boolean value. The boolean value determines which subset of individuals is returned.")
async def get_names_no_payment_due_enlisted(organ: str = Query(..., description="Organization name"), bool_value: str = Query(..., description="Boolean value (pos or neg)")):
    cursor.execute("SELECT T1.name FROM no_payment_due AS T1 INNER JOIN enlist AS T2 ON T2.name = T1.name WHERE T2.organ = ? AND T1.bool = ? LIMIT 5", (organ, bool_value))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of names in disabled who have no payment due based on bool
@app.get("/v1/student_loan/count_names_disabled_no_payment_due", operation_id="get_count_names_disabled_no_payment_due", summary="Retrieves the total count of individuals in the disabled category who have no payment due, based on the provided boolean value.")
async def get_count_names_disabled_no_payment_due(bool_value: str = Query(..., description="Boolean value (pos or neg)")):
    cursor.execute("SELECT COUNT(T1.name) FROM disabled AS T1 INNER JOIN no_payment_due AS T2 ON T2.name = T1.name WHERE T2.bool = ?", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of disabled enlisted individuals not in male based on organ
@app.get("/v1/student_loan/ratio_disabled_enlisted_not_in_male", operation_id="get_ratio_disabled_enlisted_not_in_male", summary="Retrieves the proportion of disabled individuals enlisted in a specific organization who are not identified as male. The calculation is based on the total count of disabled enlisted individuals and the count of those who are not identified as male.")
async def get_ratio_disabled_enlisted_not_in_male(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.name IS NULL, 1, 0)) AS REAL) / COUNT(T1.name) FROM disabled AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name LEFT JOIN male AS T3 ON T2.name = T3.name WHERE T2.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the percentage of enlisted individuals in and not in male based on organ
@app.get("/v1/student_loan/percentage_enlisted_in_and_not_in_male", operation_id="get_percentage_enlisted_in_and_not_in_male", summary="Retrieves the percentage of enlisted individuals who are and are not part of the male population within a specified organization. The calculation is based on the total count of enlisted individuals and the count of those who are identified as male within the given organization.")
async def get_percentage_enlisted_in_and_not_in_male(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.name IS NULL, 1, 0)) AS REAL) * 100 / COUNT(T1.name), CAST(SUM(IIF(T2.name IS NOT NULL, 1, 0)) AS REAL) * 100 / COUNT(T1.name) FROM enlist AS T1 LEFT JOIN male AS T2 ON T2.name = T1.name WHERE T1.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_not_in_male": [], "percentage_in_male": []}
    return {"percentage_not_in_male": result[0], "percentage_in_male": result[1]}

# Endpoint to get the percentage of individuals absent from school for a specific month
@app.get("/v1/student_loan/percentage_absent_from_school", operation_id="get_percentage_absent_from_school", summary="Retrieves the percentage of individuals who were absent from school during a specific month. The calculation is based on the total count of individuals absent in the specified month compared to the overall count of individuals in the database.")
async def get_percentage_absent_from_school(month: int = Query(..., description="Month value")):
    cursor.execute("SELECT CAST(SUM(IIF(month = ?, 1, 0)) AS REAL) * 100 / COUNT(name) FROM longest_absense_from_school", (month,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of positive to negative boolean values in no_payment_due
@app.get("/v1/student_loan/ratio_pos_neg_no_payment_due", operation_id="get_ratio_pos_neg_no_payment_due", summary="Retrieves the ratio of a specified positive boolean value to a specified negative boolean value in the no_payment_due dataset. This operation calculates the proportion of the positive boolean value relative to the negative boolean value, providing insights into their distribution within the dataset.")
async def get_ratio_pos_neg_no_payment_due(pos_value: str = Query(..., description="Positive boolean value (pos)"), neg_value: str = Query(..., description="Negative boolean value (neg)")):
    cursor.execute("SELECT CAST(SUM(IIF(`bool` = ?, 1, 0)) AS REAL) / SUM(IIF(`bool` = ?, 1, 0)) FROM no_payment_due", (pos_value, neg_value))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get names and schools of individuals enrolled in a specific month
@app.get("/v1/student_loan/names_schools_enrolled", operation_id="get_names_schools_enrolled", summary="Retrieves the names and schools of individuals who were enrolled in a specific month. The month value is provided as an input parameter.")
async def get_names_schools_enrolled(month: int = Query(..., description="Month value")):
    cursor.execute("SELECT name, school FROM enrolled WHERE month = ?", (month,))
    result = cursor.fetchall()
    if not result:
        return {"enrolled": []}
    return {"enrolled": [{"name": row[0], "school": row[1]} for row in result]}

# Endpoint to get the percentage of distinct organs in the enlist table
@app.get("/v1/student_loan/percentage_distinct_organs", operation_id="get_percentage_distinct_organs", summary="Retrieves the percentage of distinct organs in the enlist table, calculated as the total count of names divided by the count of distinct organs, multiplied by 100.")
async def get_percentage_distinct_organs():
    cursor.execute("SELECT CAST(COUNT(NAME) AS REAL) * 100 / COUNT(DISTINCT organ) FROM enlist")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the school and month for a given student name
@app.get("/v1/student_loan/school_month_by_name", operation_id="get_school_month_by_name", summary="Retrieves the school and month associated with a specific student. The operation requires the student's name as input and returns the corresponding school and month details from the enrolled records.")
async def get_school_month_by_name(name: str = Query(..., description="Name of the student")):
    cursor.execute("SELECT school, month FROM enrolled WHERE name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"school_month": []}
    return {"school_month": result}

# Endpoint to get the percentage of disabled persons
@app.get("/v1/student_loan/percentage_disabled_persons", operation_id="get_percentage_disabled_persons", summary="Retrieves the percentage of persons with disabilities from the total number of persons in the database. This operation calculates the ratio of disabled persons to the total population, providing a statistical overview of the prevalence of disabilities.")
async def get_percentage_disabled_persons():
    cursor.execute("SELECT CAST(COUNT(T2.name) AS REAL) * 100 / COUNT(T1.name) FROM person AS T1 LEFT JOIN disabled AS T2 ON T2.name = T1.name")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of disabled persons with a specific month of longest absence from school
@app.get("/v1/student_loan/count_disabled_by_absence_month", operation_id="get_count_disabled_by_absence_month", summary="Retrieves the total number of disabled individuals who have been absent from school for a specified month. The month of longest absence is used to filter the count.")
async def get_count_disabled_by_absence_month(month: int = Query(..., description="Month of longest absence from school")):
    cursor.execute("SELECT COUNT(T1.name) FROM disabled AS T1 LEFT JOIN longest_absense_from_school AS T2 ON T2.name = T1.name WHERE T2.month = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of persons who have not filed for bankruptcy
@app.get("/v1/student_loan/persons_not_filed_bankruptcy", operation_id="get_persons_not_filed_bankruptcy", summary="Retrieves the names of individuals who have not filed for bankruptcy. This operation filters the list of persons and excludes those who have a record of filing for bankruptcy.")
async def get_persons_not_filed_bankruptcy():
    cursor.execute("SELECT name FROM person WHERE name NOT IN ( SELECT name FROM filed_for_bankrupcy )")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the names of females in a specific organ from the enlist table
@app.get("/v1/student_loan/females_in_organ", operation_id="get_females_in_organ", summary="Retrieves the names of female individuals associated with a specific organ in the enlist table. The operation returns a maximum of five names that meet the specified organ criteria. The organ parameter is used to filter the results.")
async def get_females_in_organ(organ: str = Query(..., description="Organ of the enlist table")):
    cursor.execute("SELECT T1.name FROM enlist AS T1 LEFT JOIN male AS T2 ON T2.name = T1.name WHERE T2.name IS NULL AND T1.organ = ? LIMIT 5", (organ,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of non-disabled persons
@app.get("/v1/student_loan/count_non_disabled_persons", operation_id="get_count_non_disabled_persons", summary="Retrieves the total count of persons who are not identified as disabled in the system. This operation does not require any input parameters and returns a single integer value representing the count.")
async def get_count_non_disabled_persons():
    cursor.execute("SELECT COUNT(CASE  WHEN T2.name IS NULL THEN T1.name END) AS \"number\" FROM person AS T1 LEFT JOIN disabled AS T2 ON T2.name = T1.name")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the gender and school of a specific student
@app.get("/v1/student_loan/gender_school_by_name", operation_id="get_gender_school_by_name", summary="Retrieves the gender and school of a specific student. The operation uses the provided student's name to look up their enrollment record and determine their gender based on the presence or absence of a corresponding entry in the 'male' table. The student's school is also returned.")
async def get_gender_school_by_name(name: str = Query(..., description="Name of the student")):
    cursor.execute("SELECT IIF(T2.name IS NULL, 'female', 'male') AS gen , T1.school FROM enrolled AS T1 LEFT JOIN male AS T2 ON T2.name = T1.name WHERE T1.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"gender_school": []}
    return {"gender_school": result}

# Endpoint to get the names of disabled and unemployed persons
@app.get("/v1/student_loan/disabled_unemployed_persons", operation_id="get_disabled_unemployed_persons", summary="Retrieves the names of individuals who are both disabled and unemployed. The operation returns a maximum of five records.")
async def get_disabled_unemployed_persons():
    cursor.execute("SELECT T1.name FROM disabled AS T1 INNER JOIN unemployed AS T2 ON T2.name = T1.name LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of unemployed persons who have filed for bankruptcy
@app.get("/v1/student_loan/count_unemployed_filed_bankruptcy", operation_id="get_count_unemployed_filed_bankruptcy", summary="Retrieves the total number of individuals who are both unemployed and have filed for bankruptcy. This operation provides a statistical overview of the financial situation of unemployed individuals.")
async def get_count_unemployed_filed_bankruptcy():
    cursor.execute("SELECT COUNT(T1.name) FROM unemployed AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T2.name = T1.name")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names and organizations of students with the longest absence from school in a specific month
@app.get("/v1/student_loan/longest_absence_students", operation_id="get_longest_absence_students", summary="Retrieves the names and organizations of students who have been absent from school for the longest duration in a specified month. The number of results returned can be limited by the user.")
async def get_longest_absence_students(month: int = Query(..., description="Month of absence"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.name, T2.organ FROM longest_absense_from_school AS T1 INNER JOIN enlist AS T2 ON T2.name = T1.name WHERE T1.month = ? LIMIT ?", (month, limit))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get organization and payment status of a specific student
@app.get("/v1/student_loan/student_payment_status", operation_id="get_student_payment_status", summary="Get organization and payment status of a specific student")
async def get_student_payment_status(name: str = Query(..., description="Name of the student")):
    cursor.execute("SELECT T1.organ, T2.bool FROM enlist AS T1 INNER JOIN no_payment_due AS T2 ON T2.name = T1.name WHERE T1.name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"payment_status": []}
    return {"payment_status": result}

# Endpoint to get names of male students in a specific organization
@app.get("/v1/student_loan/male_students_in_organization", operation_id="get_male_students_in_organization", summary="Retrieves a specified number of names of male students enrolled in a given organization.")
async def get_male_students_in_organization(organ: str = Query(..., description="Organization name"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.name FROM enlist AS T1 INNER JOIN male AS T2 ON T2.name = T1.name WHERE T1.organ = ? LIMIT ?", (organ, limit))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get the percentage of non-male persons
@app.get("/v1/student_loan/percentage_non_male", operation_id="get_percentage_non_male", summary="Retrieves the percentage of persons who are not identified as male in the database. This operation calculates the ratio of persons without a corresponding entry in the 'male' table to the total number of persons in the 'person' table.")
async def get_percentage_non_male():
    cursor.execute("SELECT CAST(SUM(IIF(T2.name IS NULL, 1, 0)) AS REAL) * 100 / COUNT(T1.name) FROM person AS T1 LEFT JOIN male AS T2 ON T2.name = T1.name")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of students who filed for bankruptcy
@app.get("/v1/student_loan/bankruptcy_count", operation_id="get_bankruptcy_count", summary="Retrieves the total number of students who have filed for bankruptcy.")
async def get_bankruptcy_count():
    cursor.execute("SELECT COUNT(name) FROM filed_for_bankrupcy")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of students enrolled in a specific school and month
@app.get("/v1/student_loan/enrolled_count", operation_id="get_enrolled_count", summary="Retrieves the total number of students enrolled in a specified school during a particular month. The response is based on the provided school name and month of enrollment.")
async def get_enrolled_count(school: str = Query(..., description="School name"), month: int = Query(..., description="Month of enrollment")):
    cursor.execute("SELECT COUNT(name) FROM enrolled WHERE school = ? AND month = ?", (school, month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of disabled male students who are enlisted
@app.get("/v1/student_loan/disabled_male_enlisted_count", operation_id="get_disabled_male_enlisted_count", summary="Retrieves the total number of male students who are both disabled and enlisted. This operation provides a count of students who meet these specific criteria, offering insights into the intersection of these demographic factors.")
async def get_disabled_male_enlisted_count():
    cursor.execute("SELECT COUNT(T1.name) FROM disabled AS T1 LEFT JOIN male AS T2 ON T2.name = T1.name INNER JOIN enlist AS T3 ON T3.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of disabled students enrolled in each school
@app.get("/v1/student_loan/disabled_enrolled_count_by_school", operation_id="get_disabled_enrolled_count_by_school", summary="Retrieves the number of students with disabilities currently enrolled in each school. The data is aggregated by school, providing a count of enrolled students who are also registered as disabled.")
async def get_disabled_enrolled_count_by_school():
    cursor.execute("SELECT COUNT(T1.name) FROM enrolled AS T1 INNER JOIN disabled AS T2 ON T2.name = T1.name GROUP BY T1.school")
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": result}

# Endpoint to get the gender distribution based on organization
@app.get("/v1/student_loan/gender_distribution_by_organ", operation_id="get_gender_distribution", summary="Retrieves the gender distribution across different organizations. The data is categorized into two groups: 'female' and 'male'. The distribution is determined based on the enlistment records, with the gender being inferred from the presence or absence of a corresponding entry in the 'male' table.")
async def get_gender_distribution():
    cursor.execute("SELECT IIF(T2.name IS NULL, 'female', 'male') AS gender FROM enlist AS T1 LEFT JOIN male AS T2 ON T2.name = T1.name GROUP BY T1.organ")
    result = cursor.fetchall()
    if not result:
        return {"gender_distribution": []}
    return {"gender_distribution": result}

# Endpoint to get names of individuals enrolled in a specific number of organizations
@app.get("/v1/student_loan/names_by_organ_count", operation_id="get_names_by_organ_count", summary="Retrieves the names of individuals who are enrolled in a specified number of organizations. The operation filters the list of enrolled individuals based on the count of organizations they are associated with, as provided in the input parameter.")
async def get_names_by_organ_count(num: int = Query(..., description="Number of organizations")):
    cursor.execute("SELECT T.name FROM ( SELECT T1.name, COUNT(T1.organ) AS num FROM enlist AS T1 INNER JOIN enrolled AS T2 ON T1.name = T2.name GROUP BY T1.name ) T WHERE T.num = ?", (num,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to calculate the percentage difference in absence between two months
@app.get("/v1/student_loan/absence_percentage_difference", operation_id="get_absence_percentage_difference", summary="Retrieves the percentage difference in student absences between two specified months. This operation compares the total number of absences in the first month to the total in the second month, and returns the difference as a percentage of the first month's total.")
async def get_absence_percentage_difference(month1: int = Query(..., description="First month"), month2: int = Query(..., description="Second month")):
    cursor.execute("SELECT CAST(((SUM(IIF(month = ?, 1, 0)) - SUM(IIF(month = ?, 1, 0)))) AS REAL) * 100 / SUM(IIF(month = ?, 1, 0)) FROM longest_absense_from_school", (month1, month2, month1))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the school with the highest number of individuals who filed for bankruptcy
@app.get("/v1/student_loan/school_with_most_bankruptcies", operation_id="get_school_with_most_bankruptcies", summary="Retrieves the school with the highest count of students who have filed for bankruptcy. The operation calculates the number of bankruptcy filings for each school based on the enrollment records and returns the school with the maximum count.")
async def get_school_with_most_bankruptcies():
    cursor.execute("SELECT T.school, num FROM ( SELECT T1.school, COUNT(T2.name) AS num FROM enrolled AS T1 LEFT JOIN filed_for_bankrupcy AS T2 ON T2.name = T1.name GROUP BY T1.school ) T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"school": []}
    return {"school": result[0], "count": result[1]}

# Endpoint to get the count of disabled and unemployed individuals
@app.get("/v1/student_loan/count_disabled_unemployed", operation_id="get_count_disabled_unemployed", summary="Retrieves the total count of individuals who are both disabled and unemployed.")
async def get_count_disabled_unemployed():
    cursor.execute("SELECT COUNT(T1.name) FROM disabled AS T1 INNER JOIN unemployed AS T2 ON T2.name = T1.name")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names of individuals enlisted in a specific organization and enrolled in a specific school
@app.get("/v1/student_loan/names_by_school_organ", operation_id="get_names_by_school_organ", summary="Retrieves the names of individuals who are both enrolled in a specified school and enlisted in a given organization.")
async def get_names_by_school_organ(school: str = Query(..., description="School name"), organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT T1.name FROM enlist AS T1 INNER JOIN enrolled AS T2 ON T2.name = T1.name WHERE T2.school = ? AND T1.organ = ?", (school, organ))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get names of unemployed and disabled individuals with a specific month of longest absence from school
@app.get("/v1/student_loan/names_by_absence_month", operation_id="get_names_by_absence_month", summary="Retrieves the names of individuals who are both unemployed and disabled, and have a specific month as their longest absence from school.")
async def get_names_by_absence_month(month: int = Query(..., description="Month of longest absence from school")):
    cursor.execute("SELECT T1.name FROM unemployed AS T1 INNER JOIN disabled AS T2 ON T2.name = T1.name INNER JOIN longest_absense_from_school AS T3 ON T3.name = T2.name WHERE T3.month = ?", (month,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get names of individuals enrolled in specific schools and not in the male table
@app.get("/v1/student_loan/names_by_schools_not_male", operation_id="get_names_by_schools_not_male", summary="Retrieves the names of individuals who are enrolled in the specified schools and are not listed in the male table. The operation accepts two school names as input parameters to filter the results.")
async def get_names_by_schools_not_male(school1: str = Query(..., description="First school name"), school2: str = Query(..., description="Second school name")):
    cursor.execute("SELECT name FROM enrolled WHERE school IN (?, ?) AND name NOT IN ( SELECT name FROM male )", (school1, school2))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the school and organization of a specific individual
@app.get("/v1/student_loan/school_organ_by_name", operation_id="get_school_organ_by_name", summary="Retrieves the school and organization associated with a specific individual. The operation uses the provided individual's name to search for corresponding records in the enlist and enrolled tables, returning the school and organization details.")
async def get_school_organ_by_name(name: str = Query(..., description="Name of the individual")):
    cursor.execute("SELECT T2.school, T1.organ FROM enlist AS T1 INNER JOIN enrolled AS T2 ON T2.name = T1.name WHERE T1.name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"school": [], "organ": []}
    return {"school": result[0], "organ": result[1]}

# Endpoint to get the count of individuals who filed for bankruptcy and are not in the male table
@app.get("/v1/student_loan/count_bankrupt_not_male", operation_id="get_count_bankrupt_not_male", summary="Retrieves the count of individuals who have filed for bankruptcy and are not identified as male in the database. This operation calculates the total number of bankruptcy filers and subtracts the count of those who are also listed in the male table, providing a precise count of non-male bankruptcy filers.")
async def get_count_bankrupt_not_male():
    cursor.execute("SELECT COUNT(T2.name) - SUM(IIF(T2.name IS NULL, 1, 0)) AS num FROM filed_for_bankrupcy AS T1 LEFT JOIN male AS T2 ON T2.name = T1.name")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average month of absence for unemployed males
@app.get("/v1/student_loan/avg_month_absence_unemployed_males", operation_id="get_avg_month_absence_unemployed_males", summary="Retrieves the average duration of the longest absence from school for unemployed males.")
async def get_avg_month_absence_unemployed_males():
    cursor.execute("SELECT AVG(T2.month) AS avg FROM unemployed AS T1 INNER JOIN longest_absense_from_school AS T2 ON T2.name = T1.name INNER JOIN male AS T3 ON T3.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"avg": []}
    return {"avg": result[0]}

# Endpoint to get the percentage and indication of high absence for a specific month
@app.get("/v1/student_loan/percentage_high_absence", operation_id="get_percentage_high_absence", summary="Retrieves the percentage of students with high absence from school for a specific month. The calculation is based on the total number of students with a record of longest absence from school and those who are disabled. The response also includes an indication of whether the calculated percentage is above zero, represented as 'YES' or 'NO'.")
async def get_percentage_high_absence(month: int = Query(..., description="Month to check for high absence")):
    cursor.execute("SELECT CAST((SUM(IIF(T2.name IS NOT NULL AND T1.month = ?, 1, 0)) - SUM(IIF(T2.name IS NULL AND T1.month = ?, 1, 0))) AS REAL) * 100 / COUNT(T1.name), IIF(SUM(IIF(T2.name IS NOT NULL AND T1.month = ?, 1, 0)) - SUM(IIF(T2.name IS NULL AND T1.month = ?, 1, 0)) > 0, 'YES', 'NO') AS isHigh FROM longest_absense_from_school AS T1 LEFT JOIN disabled AS T2 ON T2.name = T1.name", (month, month, month, month))
    result = cursor.fetchone()
    if not result:
        return {"percentage": [], "isHigh": []}
    return {"percentage": result[0], "isHigh": result[1]}

# Endpoint to get the average month of absence for disabled males
@app.get("/v1/student_loan/avg_month_absence_disabled_males", operation_id="get_avg_month_absence_disabled_males", summary="Retrieves the average duration of the longest absence from school for male students with disabilities.")
async def get_avg_month_absence_disabled_males():
    cursor.execute("SELECT AVG(T1.month) FROM longest_absense_from_school AS T1 INNER JOIN disabled AS T2 ON T2.name = T1.name INNER JOIN male AS T3 ON T3.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"avg": []}
    return {"avg": result[0]}

# Endpoint to get the percentage of unemployed individuals with a specific month of absence
@app.get("/v1/student_loan/percentage_unemployed_specific_month", operation_id="get_percentage_unemployed_specific_month", summary="Retrieves the percentage of unemployed individuals who have been absent from school for a specific month. The calculation is based on the total count of unemployed individuals and the count of those absent for the specified month.")
async def get_percentage_unemployed_specific_month(month: int = Query(..., description="Month of absence to check")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.month = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.name) FROM unemployed AS T1 INNER JOIN longest_absense_from_school AS T2 ON T2.name = T1.name", (month,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of males with no payment due based on a specific boolean value
@app.get("/v1/student_loan/count_no_payment_due_males", operation_id="get_count_no_payment_due_males", summary="Retrieves the total count of male students who have no payment due, filtered by a specified boolean value.")
async def get_count_no_payment_due_males(bool_value: str = Query(..., description="Boolean value to filter no payment due males")):
    cursor.execute("SELECT COUNT(T1.name) FROM no_payment_due AS T1 INNER JOIN male AS T2 ON T2.name = T1.name WHERE T1.bool = ?", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of individuals who filed for bankruptcy with no payment due based on a specific boolean value
@app.get("/v1/student_loan/count_bankruptcy_no_payment_due", operation_id="get_count_bankruptcy_no_payment_due", summary="Retrieves the count of individuals who have filed for bankruptcy and have no payment due, based on a specified boolean value. This operation filters the results using the provided boolean value to determine which individuals meet the criteria.")
async def get_count_bankruptcy_no_payment_due(bool_value: str = Query(..., description="Boolean value to filter no payment due individuals who filed for bankruptcy")):
    cursor.execute("SELECT COUNT(T1.name) FROM filed_for_bankrupcy AS T1 INNER JOIN no_payment_due AS T2 ON T2.name = T1.name WHERE T2.bool = ?", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the boolean value for a specific individual with no payment due
@app.get("/v1/student_loan/bool_value_no_payment_due", operation_id="get_bool_value_no_payment_due", summary="Retrieves a boolean value indicating whether a specific individual has no payment due for their student loan.")
async def get_bool_value_no_payment_due(name: str = Query(..., description="Name of the individual to check")):
    cursor.execute("SELECT `bool` FROM no_payment_due WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"bool_value": []}
    return {"bool_value": result[0]}

# Endpoint to get the count of disabled individuals with no payment due based on a specific boolean value
@app.get("/v1/student_loan/count_no_payment_due_disabled", operation_id="get_count_no_payment_due_disabled", summary="Retrieves the count of disabled individuals who have no payment due, filtered by a specified boolean value.")
async def get_count_no_payment_due_disabled(bool_value: str = Query(..., description="Boolean value to filter no payment due disabled individuals")):
    cursor.execute("SELECT COUNT(T1.name) FROM no_payment_due AS T1 INNER JOIN disabled AS T2 ON T1.name = T2.name WHERE T1.bool = ?", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of the individual with the longest absence from school and no payment due based on a specific boolean value
@app.get("/v1/student_loan/longest_absence_no_payment_due", operation_id="get_longest_absence_no_payment_due", summary="Retrieves the name of the individual who has been absent from school the longest and has no payment due, based on the provided boolean value. The operation filters individuals with no payment due based on the boolean value and orders the results by the length of absence in descending order, returning the name of the individual with the longest absence.")
async def get_longest_absence_no_payment_due(bool_value: str = Query(..., description="Boolean value to filter no payment due individuals with the longest absence")):
    cursor.execute("SELECT T1.name FROM longest_absense_from_school AS T1 INNER JOIN no_payment_due AS T2 ON T1.name = T2.name WHERE T2.bool = ? ORDER BY T1.month DESC LIMIT 1", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the names of individuals with a specific month of absence and no payment due based on a specific boolean value
@app.get("/v1/student_loan/names_specific_month_absence_no_payment_due", operation_id="get_names_specific_month_absence_no_payment_due", summary="Retrieves the names of individuals who were absent for a specific month and have no payment due, based on a provided boolean value. The operation filters the data using the specified month and boolean value, and returns the names that match the criteria.")
async def get_names_specific_month_absence_no_payment_due(month: int = Query(..., description="Month of absence to check"), bool_value: str = Query(..., description="Boolean value to filter no payment due individuals")):
    cursor.execute("SELECT T1.name FROM longest_absense_from_school AS T1 INNER JOIN no_payment_due AS T2 ON T1.name = T2.name WHERE T1.month = ? AND T2.bool = ?", (month, bool_value))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of disabled individuals enlisted in a specific organization
@app.get("/v1/student_loan/count_disabled_enlisted", operation_id="get_count_disabled_enlisted", summary="Retrieves the total number of disabled individuals who are enlisted in a specified organization.")
async def get_count_disabled_enlisted(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM disabled AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of individuals with the longest absence from school enlisted in a specific organization
@app.get("/v1/student_loan/count_longest_absence_enlisted", operation_id="get_count_longest_absence_enlisted", summary="Retrieves the number of individuals who have had the longest absence from school and are enlisted in a specified organization. The count is determined by matching the names of individuals in the longest_absense_from_school table with those in the enlist table for the given organization. The result is ordered by the length of absence in descending order and limited to the top record.")
async def get_count_longest_absence_enlisted(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.NAME) FROM longest_absense_from_school AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ? ORDER BY T1.month DESC LIMIT 1", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of individuals enlisted in a specific organization who filed for bankruptcy
@app.get("/v1/student_loan/count_enlisted_bankruptcy", operation_id="get_count_enlisted_bankruptcy", summary="Retrieves the total number of individuals enlisted in a specified organization who have filed for bankruptcy. The operation requires the name of the organization as an input parameter.")
async def get_count_enlisted_bankruptcy(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM enlist AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T1.name = T2.name WHERE T1.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of individuals with no payment due enlisted in a specific organization
@app.get("/v1/student_loan/count_no_payment_due_enlisted", operation_id="get_count_no_payment_due_enlisted", summary="Retrieves the count of individuals who are enlisted in a specific organization and have no payment due. The operation requires a boolean value to filter individuals based on their payment status and an organization name to specify the enlistment.")
async def get_count_no_payment_due_enlisted(bool: str = Query(..., description="Boolean value indicating payment status"), organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM no_payment_due AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T1.bool = ? AND T2.organ = ?", (bool, organ))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of disabled individuals with no absence from school
@app.get("/v1/student_loan/percentage_disabled_no_absence", operation_id="get_percentage_disabled_no_absence", summary="Retrieves the percentage of disabled individuals who have not been absent from school. This operation calculates the total number of disabled individuals who have not missed any school days and returns this value as a percentage.")
async def get_percentage_disabled_no_absence():
    cursor.execute("SELECT 100 * SUM(IIF(T2.month = 0, 1, 0)) AS num FROM disabled AS T1 INNER JOIN longest_absense_from_school AS T2 ON T1.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of individuals with absence from school less than a specified number of months
@app.get("/v1/student_loan/count_absence_less_than_months", operation_id="get_count_absence_less_than_months", summary="Retrieves the count of students who have been absent from school for less than a specified number of months. The input parameter determines the maximum number of months of absence to consider.")
async def get_count_absence_less_than_months(month: int = Query(..., description="Number of months")):
    cursor.execute("SELECT COUNT(name) FROM longest_absense_from_school WHERE month < ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of non-male individuals enlisted in a specific organization
@app.get("/v1/student_loan/count_non_male_enlisted", operation_id="get_count_non_male_enlisted", summary="Retrieves the count of individuals who are not identified as male and are enlisted in a specified organization. The input parameter is the name of the organization.")
async def get_count_non_male_enlisted(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(*) FROM person AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name LEFT JOIN male AS T3 ON T1.name = T3.name WHERE T2.organ = ? AND T3.name IS NULL", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average absence from school for non-male individuals
@app.get("/v1/student_loan/avg_absence_non_male", operation_id="get_avg_absence_non_male", summary="Retrieves the average duration of the longest absence from school for individuals who are not identified as male in the database.")
async def get_avg_absence_non_male():
    cursor.execute("SELECT AVG(T2.month) FROM person AS T1 INNER JOIN longest_absense_from_school AS T2 ON T1.name = T2.name LEFT JOIN male AS T3 ON T1.name = T3.name WHERE T3.name IS NULL")
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the names of non-male individuals who filed for bankruptcy
@app.get("/v1/student_loan/names_non_male_bankruptcy", operation_id="get_names_non_male_bankruptcy", summary="Retrieves the names of individuals who have filed for bankruptcy and are not identified as male in the database. This operation returns a list of names that meet these criteria, providing insights into non-male bankruptcy filings.")
async def get_names_non_male_bankruptcy():
    cursor.execute("SELECT T1.name FROM person AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T1.name = T2.name LEFT JOIN male AS T3 ON T1.name = T3.name WHERE T3.name IS NULL")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get names from enlist where organ is a specified value
@app.get("/v1/student_loan/get_names_enlist_organ", operation_id="get_names_enlist_organ", summary="Retrieves the names of students enlisted in a specific organization. The operation filters the enlistment records based on the provided organization value.")
async def get_names_enlist_organ(organ: str = Query(..., description="Organ value to filter records")):
    cursor.execute("SELECT name FROM enlist WHERE organ = ?", (organ,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of names from filed_for_bankrupcy and enrolled where school is a specified value
@app.get("/v1/student_loan/count_names_filed_for_bankrupcy_enrolled_school", operation_id="get_count_names_filed_for_bankrupcy_enrolled_school", summary="Retrieves the count of individuals who have filed for bankruptcy and are enrolled in a specified school. The operation filters records based on the provided school value.")
async def get_count_names_filed_for_bankrupcy_enrolled_school(school: str = Query(..., description="School value to filter records")):
    cursor.execute("SELECT COUNT(T1.name) FROM filed_for_bankrupcy AS T1 INNER JOIN enrolled AS T2 ON T1.name = T2.name WHERE T2.school = ?", (school,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of names from male and disabled
@app.get("/v1/student_loan/count_names_male_disabled", operation_id="get_count_names_male_disabled", summary="Retrieves the total count of names that are common between the male and disabled groups. This operation provides a quantitative measure of the intersection between these two demographics.")
async def get_count_names_male_disabled():
    cursor.execute("SELECT COUNT(T1.name) FROM male AS T1 INNER JOIN disabled AS T2 ON T1.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get names from enlist and disabled where organ is a specified value
@app.get("/v1/student_loan/get_names_enlist_disabled_organ", operation_id="get_names_enlist_disabled_organ", summary="Retrieves the names of students who are both enlisted and disabled, filtered by a specified organ value. The organ parameter is used to narrow down the results to a specific organ value.")
async def get_names_enlist_disabled_organ(organ: str = Query(..., description="Organ value to filter records")):
    cursor.execute("SELECT T1.name FROM enlist AS T1 INNER JOIN disabled AS T2 ON T1.name = T2.name WHERE T1.organ = ?", (organ,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get names from male and no_payment_due where bool is a specified value
@app.get("/v1/student_loan/get_names_male_no_payment_due_bool", operation_id="get_names_male_no_payment_due_bool", summary="Retrieves the names of male students who have no payment due, filtered by a specified boolean value.")
async def get_names_male_no_payment_due_bool(bool_value: str = Query(..., description="Bool value to filter records")):
    cursor.execute("SELECT T1.name FROM male AS T1 INNER JOIN no_payment_due AS T2 ON T1.name = T2.name WHERE T2.bool = ?", (bool_value,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get bool values from no_payment_due and unemployed where name is a specified value
@app.get("/v1/student_loan/get_bool_no_payment_due_unemployed_name", operation_id="get_bool_no_payment_due_unemployed_name", summary="Retrieves a boolean value indicating whether a payment is due and if the individual is unemployed, based on a specified name. This operation filters records in the no_payment_due and unemployed tables using the provided name parameter.")
async def get_bool_no_payment_due_unemployed_name(name: str = Query(..., description="Name value to filter records")):
    cursor.execute("SELECT T1.bool FROM no_payment_due AS T1 INNER JOIN unemployed AS T2 ON T1.name = T2.name WHERE T1.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"bool_values": []}
    return {"bool_values": [row[0] for row in result]}

# Endpoint to get the percentage of males in the person table
@app.get("/v1/student_loan/get_percentage_males", operation_id="get_percentage_males", summary="Retrieves the percentage of males in the person table by comparing the count of names in the person table with the count of names in the male table.")
async def get_percentage_males():
    cursor.execute("SELECT CAST(COUNT(T2.name) AS REAL) * 100 / COUNT(T1.name) FROM person AS T1 LEFT JOIN male AS T2 ON T1.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get school and gender from enrolled, person, and male tables where name is a specified value
@app.get("/v1/student_loan/get_school_gender_enrolled_person_male_name", operation_id="get_school_gender_enrolled_person_male_name", summary="Retrieves the school and gender of a student with a specified name. The operation filters records from the enrolled and person tables based on the provided name. If the student's name is found in the male table, the gender is set to 'male'; otherwise, it is set to 'female'.")
async def get_school_gender_enrolled_person_male_name(name: str = Query(..., description="Name value to filter records")):
    cursor.execute("SELECT T1.school, IIF(T3.name IS NULL, 'female', 'male') AS gender FROM enrolled AS T1 INNER JOIN person AS T2 ON T1.name = T2.name LEFT JOIN male AS T3 ON T2.name = T3.name WHERE T2.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"school_gender": []}
    return {"school_gender": [{"school": row[0], "gender": row[1]} for row in result]}

# Endpoint to get the latest month of absence from school for a specific organization
@app.get("/v1/student_loan/latest_absence_month_by_organ", operation_id="get_latest_absence_month", summary="Retrieves the most recent month in which a student from the specified organization was absent from school. The operation identifies the longest absence period for each student in the organization and returns the latest month of absence across all students.")
async def get_latest_absence_month(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT T1.month FROM longest_absense_from_school AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ? ORDER BY T1.month DESC LIMIT 1", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"month": []}
    return {"month": result[0]}

# Endpoint to get the count of names with a specific month and organization
@app.get("/v1/student_loan/count_names_by_month_organ", operation_id="get_count_names_by_month_organ", summary="Retrieves the total count of students with a specific month of longest absence from school and a specific organization. The count is determined by matching the student names in the longest_absense_from_school table with those in the enlist table, based on the provided month and organization.")
async def get_count_names_by_month_organ(month: int = Query(..., description="Month"), organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM longest_absense_from_school AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T1.month = ? AND T2.organ = ?", (month, organ))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of unemployed males who are also disabled
@app.get("/v1/student_loan/unemployed_male_disabled_names", operation_id="get_unemployed_male_disabled_names", summary="Retrieves the names of individuals who are unemployed, male, and disabled. This operation combines data from three distinct sources to provide a comprehensive list of names that meet all three criteria.")
async def get_unemployed_male_disabled_names():
    cursor.execute("SELECT T2.NAME FROM unemployed AS T1 INNER JOIN male AS T2 ON T1.name = T2.name INNER JOIN disabled AS T3 ON T3.name = T2.name")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of male students enrolled in a specific school
@app.get("/v1/student_loan/count_male_students_by_school", operation_id="get_count_male_students_by_school", summary="Retrieves the total number of male students currently enrolled in a specified school. The operation requires the name of the school as an input parameter.")
async def get_count_male_students_by_school(school: str = Query(..., description="School name")):
    cursor.execute("SELECT COUNT(T1.name) FROM enrolled AS T1 INNER JOIN male AS T2 ON T1.name = T2.name WHERE T1.school = ?", (school,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of males in a specific school and organization
@app.get("/v1/student_loan/percentage_males_by_school_organ", operation_id="get_percentage_males_by_school_organ", summary="Retrieves the percentage of males enrolled in a specific school and organization. The calculation is based on the total number of enrolled students and the count of males in the given school and organization.")
async def get_percentage_males_by_school_organ(school: str = Query(..., description="School name"), organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT CAST(COUNT(T4.name) AS REAL) * 100 / COUNT(T2.name) FROM enlist AS T1 INNER JOIN person AS T2 ON T1.name = T2.name INNER JOIN enrolled AS T3 ON T3.name = T2.name LEFT JOIN male AS T4 ON T2.name = T4.name WHERE T3.school = ? AND T1.organ = ?", (school, organ))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of non-male students in a specific school
@app.get("/v1/student_loan/percentage_non_male_students_by_school", operation_id="get_percentage_non_male_students_by_school", summary="Retrieves the proportion of students who do not identify as male in a specified school. This operation calculates the ratio of students without a male identifier to the total number of students enrolled in the school. The school is identified by its name.")
async def get_percentage_non_male_students_by_school(school: str = Query(..., description="School name")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.school = ? AND T4.name IS NULL, 1, 0)) AS REAL) / COUNT(T1.name) FROM enrolled AS T1 INNER JOIN disabled AS T2 ON T1.name = T2.name INNER JOIN person AS T3 ON T1.name = T3.name LEFT JOIN male AS T4 ON T3.name = T4.name", (school,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the organization with the highest number of enlistments
@app.get("/v1/student_loan/top_organ_by_enlistments", operation_id="get_top_organ_by_enlistments", summary="Retrieves the organization with the most enlistments. This operation returns the name of the organization that has the highest number of enlistments, as determined by the count of enlistments associated with each organization.")
async def get_top_organ_by_enlistments():
    cursor.execute("SELECT organ FROM ( SELECT organ, COUNT(organ) AS num FROM enlist GROUP BY organ ) T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"organ": []}
    return {"organ": result[0]}

# Endpoint to get the count of unemployed individuals with no payment due based on a boolean value
@app.get("/v1/student_loan/count_unemployed_no_payment_due", operation_id="get_count_unemployed_no_payment_due", summary="Retrieves the count of unemployed individuals who have no payment due, based on the provided boolean value. This operation filters the data based on the boolean value and returns the count of individuals who meet the specified criteria.")
async def get_count_unemployed_no_payment_due(bool_value: str = Query(..., description="Boolean value (pos or neg)")):
    cursor.execute("SELECT COUNT(T1.name) FROM no_payment_due AS T1 INNER JOIN unemployed AS T2 ON T1.name = T2.name WHERE T1.bool = ?", (bool_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of enlisted persons not in the male table for a specific organization
@app.get("/v1/student_loan/enlist_not_male_count", operation_id="get_enlist_not_male_count", summary="Retrieves the total count of enlisted individuals who are not identified as male for a specific organization. The count is determined by comparing the enlist and person tables, and then excluding those who are also present in the male table. The organization name is used to filter the results.")
async def get_enlist_not_male_count(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT SUM(IIF(T3.name IS NULL, 1, 0)) AS 'result' FROM enlist AS T1 INNER JOIN person AS T2 ON T1.name = T2.name LEFT JOIN male AS T3 ON T2.name = T3.name WHERE T1.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the organization with the highest number of enlisted males
@app.get("/v1/student_loan/top_organ_enlisted_males", operation_id="get_top_organ_enlisted_males", summary="Retrieves the organization with the highest count of enlisted males. The operation calculates the number of enlisted males for each organization and returns the one with the most enlisted males.")
async def get_top_organ_enlisted_males():
    cursor.execute("SELECT T.organ FROM ( SELECT T2.organ, COUNT(T1.name) AS num FROM male AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name GROUP BY T2.organ ) T ORDER BY T.num LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"organ": []}
    return {"organ": result[0]}

# Endpoint to get the count of disabled persons with a specific month of longest absence from school
@app.get("/v1/student_loan/disabled_longest_absence_count", operation_id="get_disabled_longest_absence_count", summary="Retrieves the count of disabled individuals who have been absent from school for the longest duration during a specified month. The month of longest absence is provided as an input parameter.")
async def get_disabled_longest_absence_count(month: int = Query(..., description="Month of longest absence from school")):
    cursor.execute("SELECT COUNT(T1.name) FROM disabled AS T1 INNER JOIN longest_absense_from_school AS T2 ON T1.name = T2.name WHERE T2.month = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of unemployed disabled persons
@app.get("/v1/student_loan/unemployed_disabled_count", operation_id="get_unemployed_disabled_count", summary="Retrieves the total count of individuals who are both unemployed and disabled. This operation does not require any input parameters and returns a single integer value representing the count.")
async def get_unemployed_disabled_count():
    cursor.execute("SELECT COUNT(T1.name) FROM unemployed AS T1 INNER JOIN disabled AS T2 ON T1.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons who filed for bankruptcy and are enlisted in a specific organization
@app.get("/v1/student_loan/bankruptcy_enlist_count", operation_id="get_bankruptcy_enlist_count", summary="Retrieves the number of individuals who have filed for bankruptcy and are enrolled in a specified organization. The operation requires the name of the organization as an input parameter.")
async def get_bankruptcy_enlist_count(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM filed_for_bankrupcy AS T1 INNER JOIN enlist AS T2 ON T1.name = T2.name WHERE T2.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of disabled persons who filed for bankruptcy
@app.get("/v1/student_loan/disabled_bankruptcy_count", operation_id="get_disabled_bankruptcy_count", summary="Retrieves the total count of disabled individuals who have filed for bankruptcy. This operation provides a statistical overview of the financial status of disabled persons.")
async def get_disabled_bankruptcy_count():
    cursor.execute("SELECT COUNT(T1.name) FROM disabled AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T1.name = T2.name")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons with the longest absence from school who filed for bankruptcy and are enlisted in a specific organization
@app.get("/v1/student_loan/longest_absence_bankruptcy_enlist_count", operation_id="get_longest_absence_bankruptcy_enlist_count", summary="Retrieves the count of individuals who have been absent from school for the longest duration, have filed for bankruptcy, and are enlisted in a specified organization.")
async def get_longest_absence_bankruptcy_enlist_count(organ: str = Query(..., description="Organization name")):
    cursor.execute("SELECT COUNT(T1.name) FROM longest_absense_from_school AS T1 INNER JOIN filed_for_bankrupcy AS T2 ON T1.name = T2.name INNER JOIN enlist AS T3 ON T3.name = T2.name WHERE T3.organ = ?", (organ,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of unemployed disabled persons with a specific month of longest absence from school
@app.get("/v1/student_loan/unemployed_disabled_longest_absence_count", operation_id="get_unemployed_disabled_longest_absence_count", summary="Retrieves the number of individuals who are both unemployed and disabled, and have a specific month as their longest absence from school. The month is provided as an input parameter.")
async def get_unemployed_disabled_longest_absence_count(month: int = Query(..., description="Month of longest absence from school")):
    cursor.execute("SELECT COUNT(T1.name) FROM longest_absense_from_school AS T1 INNER JOIN disabled AS T2 ON T1.name = T2.name INNER JOIN unemployed AS T3 ON T3.name = T2.name WHERE T1.month = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the organization with the highest percentage of males
@app.get("/v1/student_loan/organization_highest_male_percentage", operation_id="get_organization_highest_male_percentage", summary="Retrieves the organization with the highest percentage of males, based on the number of enrolled males. The response is limited to the specified number of results.")
async def get_organization_highest_male_percentage(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T.organ, T.per FROM ( SELECT T1.organ, CAST(COUNT(T3.name) AS REAL) / COUNT(T2.name) AS per , COUNT(T3.name) AS num FROM enlist AS T1 INNER JOIN person AS T2 ON T1.name = T2.name LEFT JOIN male AS T3 ON T2.name = T3.name GROUP BY T1.organ ) T ORDER BY T.num DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": [{"organ": row[0], "percentage": row[1]} for row in result]}

api_calls = [
    "/v1/student_loan/count_names_by_month?month=0",
    "/v1/student_loan/max_month",
    "/v1/student_loan/count_names_by_organ?organ=navy",
    "/v1/student_loan/count_names_by_month_threshold?month_threshold=5",
    "/v1/student_loan/names_by_organ?organ=navy",
    "/v1/student_loan/count_names_by_month_threshold_bankrupcy?month_threshold=5",
    "/v1/student_loan/count_names_by_organ_and_bool?organ=navy&bool_value=pos",
    "/v1/student_loan/names_from_disabled_male_bankrupcy",
    "/v1/student_loan/count_names_not_in_male",
    "/v1/student_loan/count_names_unemployed_no_payment_due",
    "/v1/student_loan/longest_absence_month_by_organ?organ=navy",
    "/v1/student_loan/count_names_longest_absence_by_month?month=0",
    "/v1/student_loan/organ_by_longest_absence_month?month=9",
    "/v1/student_loan/top_organ_by_disabled_count",
    "/v1/student_loan/bankruptcy_non_male",
    "/v1/student_loan/percentage_males_by_organ?organ=navy",
    "/v1/student_loan/avg_longest_absence_month_disabled",
    "/v1/student_loan/payment_due_status?name=student348",
    "/v1/student_loan/school_by_name?name=student829",
    "/v1/student_loan/longest_absence_names?month=6",
    "/v1/student_loan/student_organization?name=student285",
    "/v1/student_loan/disabled_student_info?name=student281",
    "/v1/student_loan/enlisted_students_info?organ=air_force",
    "/v1/student_loan/non_male_payment_status?bool=neg",
    "/v1/student_loan/payment_status_organ_names?bool=pos",
    "/v1/student_loan/disabled_students_by_school?school=smc",
    "/v1/student_loan/bankruptcy_enlisted_students?organ=foreign_legion",
    "/v1/student_loan/male_payment_status_count?bool=neg",
    "/v1/student_loan/names_by_organ_and_month?organ=marines&month=6",
    "/v1/student_loan/names_by_school?school=smc",
    "/v1/student_loan/names_by_month?month=6",
    "/v1/student_loan/count_names_by_school_and_organ?school=ucsd&organ=peace_corps",
    "/v1/student_loan/month_and_organ_by_name?name=student21",
    "/v1/student_loan/percentage_positive_to_negative_by_organ?organ=foreign_legion",
    "/v1/student_loan/percentage_students_by_school_and_organ?school=occ&organ=navy",
    "/v1/student_loan/names_for_max_month",
    "/v1/student_loan/count_names",
    "/v1/student_loan/names_and_month_for_max_month",
    "/v1/student_loan/count_unemployed_bankruptcy",
    "/v1/student_loan/distinct_organizations",
    "/v1/student_loan/count_unemployed_bankruptcy_no_payment?bool_value=pos",
    "/v1/student_loan/determine_gender?name=student124",
    "/v1/student_loan/counts_by_names?name1=student180&name2=student117",
    "/v1/student_loan/count_enlisted_not_male?organ1=marines&organ2=air_force",
    "/v1/student_loan/school_organ_by_names?name1=student27&name2=student17&name3=student101",
    "/v1/student_loan/percentage_disabled_not_male",
    "/v1/student_loan/count_not_male_enrolled",
    "/v1/student_loan/bankruptcy_payment_status?bool_status=neg",
    "/v1/student_loan/average_absence_month",
    "/v1/student_loan/average_absence_month_not_unemployed",
    "/v1/student_loan/average_absence_month_disabled",
    "/v1/student_loan/count_absence_by_month?month=0",
    "/v1/student_loan/count_absence_greater_than_month?month=2",
    "/v1/student_loan/count_payment_status?bool_status=neg",
    "/v1/student_loan/disabled_longest_absence_names?month=0",
    "/v1/student_loan/unemployed_enlist_count?organ=navy",
    "/v1/student_loan/male_enlist_count?organ=foreign_legion",
    "/v1/student_loan/non_male_enlist_count?organ=air_force",
    "/v1/student_loan/disabled_longest_absence_name",
    "/v1/student_loan/unemployed_enlist_names?organ=marines",
    "/v1/student_loan/unemployed_absence_percentage?month=5",
    "/v1/student_loan/unemployed_disabled_absence_count?month=8",
    "/v1/student_loan/unemployed_longest_absence_name",
    "/v1/student_loan/count_absent_students_by_month?month=3",
    "/v1/student_loan/absent_students_filed_bankruptcy_by_month?month=0",
    "/v1/student_loan/longest_absent_student_filed_bankruptcy",
    "/v1/student_loan/determine_gender_by_name?name=studenT1000",
    "/v1/student_loan/count_disabled_students",
    "/v1/student_loan/count_enlisted_students_by_organs?organ1=army&organ2=peace_corps&organ3=foreign_legion",
    "/v1/student_loan/count_disabled_enlisted_students_by_organ?organ=marines",
    "/v1/student_loan/unemployed_count_by_bool?bool_value=pos",
    "/v1/student_loan/longest_absence_names_by_month?month=0",
    "/v1/student_loan/school_with_most_disabled_enrolled",
    "/v1/student_loan/organizations_of_bankrupt_enlisted",
    "/v1/student_loan/count_males_enlisted_in_multiple_organizations?num_organizations=1",
    "/v1/student_loan/disabled_enlisted_in_organization?organ=navy",
    "/v1/student_loan/enrolled_longest_absence_count_by_school_month?school=smc&month=7",
    "/v1/student_loan/disabled_not_male",
    "/v1/student_loan/unemployed_to_disabled_ratio",
    "/v1/student_loan/percentage_males_enlisted_in_organization?organ=fire_department",
    "/v1/student_loan/count_names_by_school?school=ucla",
    "/v1/student_loan/names_ordered_by_month?limit=5",
    "/v1/student_loan/count_names_by_bool?bool_value=neg",
    "/v1/student_loan/count_names_by_organ_and_school?organ=peace_corps&school=ucsd",
    "/v1/student_loan/count_names_by_month_unemployed?month=0",
    "/v1/student_loan/names_by_bool_and_organ?bool_value=neg&organ=fire_department&limit=10",
    "/v1/student_loan/count_names_enlist_unemployed?organ=army",
    "/v1/student_loan/count_names_no_payment_due_unemployed?bool_value=pos",
    "/v1/student_loan/names_no_payment_due_enlisted?organ=peace_corps&bool_value=pos",
    "/v1/student_loan/count_names_disabled_no_payment_due?bool_value=pos",
    "/v1/student_loan/ratio_disabled_enlisted_not_in_male?organ=foreign_legion",
    "/v1/student_loan/percentage_enlisted_in_and_not_in_male?organ=fire_department",
    "/v1/student_loan/percentage_absent_from_school?month=0",
    "/v1/student_loan/ratio_pos_neg_no_payment_due?pos_value=pos&neg_value=neg",
    "/v1/student_loan/names_schools_enrolled?month=15",
    "/v1/student_loan/percentage_distinct_organs",
    "/v1/student_loan/school_month_by_name?name=student214",
    "/v1/student_loan/percentage_disabled_persons",
    "/v1/student_loan/count_disabled_by_absence_month?month=9",
    "/v1/student_loan/persons_not_filed_bankruptcy",
    "/v1/student_loan/females_in_organ?organ=air_force",
    "/v1/student_loan/count_non_disabled_persons",
    "/v1/student_loan/gender_school_by_name?name=student995",
    "/v1/student_loan/disabled_unemployed_persons",
    "/v1/student_loan/count_unemployed_filed_bankruptcy",
    "/v1/student_loan/longest_absence_students?month=4&limit=5",
    "/v1/student_loan/student_payment_status?name=student160",
    "/v1/student_loan/male_students_in_organization?organ=foreign_legion&limit=10",
    "/v1/student_loan/percentage_non_male",
    "/v1/student_loan/bankruptcy_count",
    "/v1/student_loan/enrolled_count?school=smc&month=1",
    "/v1/student_loan/disabled_male_enlisted_count",
    "/v1/student_loan/disabled_enrolled_count_by_school",
    "/v1/student_loan/gender_distribution_by_organ",
    "/v1/student_loan/names_by_organ_count?num=2",
    "/v1/student_loan/absence_percentage_difference?month1=0&month2=9",
    "/v1/student_loan/school_with_most_bankruptcies",
    "/v1/student_loan/count_disabled_unemployed",
    "/v1/student_loan/names_by_school_organ?school=occ&organ=fire_department",
    "/v1/student_loan/names_by_absence_month?month=5",
    "/v1/student_loan/names_by_schools_not_male?school1=occ&school2=ulca",
    "/v1/student_loan/school_organ_by_name?name=student211",
    "/v1/student_loan/count_bankrupt_not_male",
    "/v1/student_loan/avg_month_absence_unemployed_males",
    "/v1/student_loan/percentage_high_absence?month=0",
    "/v1/student_loan/avg_month_absence_disabled_males",
    "/v1/student_loan/percentage_unemployed_specific_month?month=0",
    "/v1/student_loan/count_no_payment_due_males?bool_value=pos",
    "/v1/student_loan/count_bankruptcy_no_payment_due?bool_value=pos",
    "/v1/student_loan/bool_value_no_payment_due?name=student124",
    "/v1/student_loan/count_no_payment_due_disabled?bool_value=neg",
    "/v1/student_loan/longest_absence_no_payment_due?bool_value=neg",
    "/v1/student_loan/names_specific_month_absence_no_payment_due?month=5&bool_value=neg",
    "/v1/student_loan/count_disabled_enlisted?organ=marines",
    "/v1/student_loan/count_longest_absence_enlisted?organ=peace_corps",
    "/v1/student_loan/count_enlisted_bankruptcy?organ=navy",
    "/v1/student_loan/count_no_payment_due_enlisted?bool=pos&organ=marines",
    "/v1/student_loan/percentage_disabled_no_absence",
    "/v1/student_loan/count_absence_less_than_months?month=4",
    "/v1/student_loan/count_non_male_enlisted?organ=marines",
    "/v1/student_loan/avg_absence_non_male",
    "/v1/student_loan/names_non_male_bankruptcy",
    "/v1/student_loan/get_names_enlist_organ?organ=fire_department",
    "/v1/student_loan/count_names_filed_for_bankrupcy_enrolled_school?school=occ",
    "/v1/student_loan/count_names_male_disabled",
    "/v1/student_loan/get_names_enlist_disabled_organ?organ=navy",
    "/v1/student_loan/get_names_male_no_payment_due_bool?bool_value=pos",
    "/v1/student_loan/get_bool_no_payment_due_unemployed_name?name=student110",
    "/v1/student_loan/get_percentage_males",
    "/v1/student_loan/get_school_gender_enrolled_person_male_name?name=student34",
    "/v1/student_loan/latest_absence_month_by_organ?organ=fire_department",
    "/v1/student_loan/count_names_by_month_organ?month=1&organ=air_force",
    "/v1/student_loan/unemployed_male_disabled_names",
    "/v1/student_loan/count_male_students_by_school?school=occ",
    "/v1/student_loan/percentage_males_by_school_organ?school=ucla&organ=air_force",
    "/v1/student_loan/percentage_non_male_students_by_school?school=uci",
    "/v1/student_loan/top_organ_by_enlistments",
    "/v1/student_loan/count_unemployed_no_payment_due?bool_value=pos",
    "/v1/student_loan/enlist_not_male_count?organ=army",
    "/v1/student_loan/top_organ_enlisted_males",
    "/v1/student_loan/disabled_longest_absence_count?month=0",
    "/v1/student_loan/unemployed_disabled_count",
    "/v1/student_loan/bankruptcy_enlist_count?organ=marines",
    "/v1/student_loan/disabled_bankruptcy_count",
    "/v1/student_loan/longest_absence_bankruptcy_enlist_count?organ=fire_department",
    "/v1/student_loan/unemployed_disabled_longest_absence_count?month=0",
    "/v1/student_loan/organization_highest_male_percentage?limit=1"
]
