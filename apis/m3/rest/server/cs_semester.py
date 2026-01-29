from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/cs_semester/cs_semester.sqlite')
cursor = conn.cursor()

# Endpoint to get the name of a course based on specific names and order by difficulty
@app.get("/v1/cs_semester/course_name_by_difficulty", operation_id="get_course_name_by_difficulty", summary="Retrieves the name of a course that matches either of the provided course names, sorted by descending difficulty. The result is limited to one course.")
async def get_course_name_by_difficulty(name1: str = Query(..., description="First course name"), name2: str = Query(..., description="Second course name")):
    cursor.execute("SELECT name FROM course WHERE name = ? OR name = ? ORDER BY diff DESC LIMIT 1", (name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the names of courses with credits less than a specific course
@app.get("/v1/cs_semester/course_names_by_credit", operation_id="get_course_names_by_credit", summary="Retrieves the names of courses that have fewer credits than a specified course. The comparison is based on the input parameter, which represents the name of the course used as the reference for credit comparison.")
async def get_course_names_by_credit(course_name: str = Query(..., description="Name of the course to compare credits with")):
    cursor.execute("SELECT name FROM course WHERE credit < ( SELECT credit FROM course WHERE name = ? )", (course_name,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of professors with popularity greater than a specific professor
@app.get("/v1/cs_semester/prof_count_by_popularity", operation_id="get_prof_count_by_popularity", summary="Retrieve the number of professors who have a higher popularity rating than a specified professor. The comparison is based on the first and last names of the professor.")
async def get_prof_count_by_popularity(first_name: str = Query(..., description="First name of the professor to compare popularity with"), last_name: str = Query(..., description="Last name of the professor to compare popularity with")):
    cursor.execute("SELECT COUNT(prof_id) FROM prof WHERE popularity > ( SELECT popularity FROM prof WHERE first_name = ? AND last_name = ? )", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone number of a student based on their first and last name
@app.get("/v1/cs_semester/student_phone_number", operation_id="get_student_phone_number", summary="Retrieves the phone number of a student by matching their first and last names in the student database. The operation requires the student's last and first names as input parameters.")
async def get_student_phone_number(l_name: str = Query(..., description="Last name of the student"), f_name: str = Query(..., description="First name of the student")):
    cursor.execute("SELECT phone_number FROM student WHERE l_name = ? AND f_name = ?", (l_name, f_name))
    result = cursor.fetchone()
    if not result:
        return {"phone_number": []}
    return {"phone_number": result[0]}

# Endpoint to get the names of professors associated with a specific student
@app.get("/v1/cs_semester/prof_names_by_student", operation_id="get_prof_names_by_student", summary="Retrieves the first and last names of professors who are associated with a student identified by their first and last names. The operation uses the provided student's first and last names to filter the results.")
async def get_prof_names_by_student(f_name: str = Query(..., description="First name of the student"), l_name: str = Query(..., description="Last name of the student")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T2.student_id = T3.student_id WHERE T3.f_name = ? AND T3.l_name = ?", (f_name, l_name))
    result = cursor.fetchall()
    if not result:
        return {"professors": []}
    return {"professors": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of students associated with a specific professor
@app.get("/v1/cs_semester/student_count_by_prof", operation_id="get_student_count_by_prof", summary="Retrieves the total number of students associated with a specific professor, identified by their first and last names.")
async def get_student_count_by_prof(first_name: str = Query(..., description="First name of the professor"), last_name: str = Query(..., description="Last name of the professor")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of students with a specific capability
@app.get("/v1/cs_semester/student_names_by_capability", operation_id="get_student_names_by_capability", summary="Retrieves the first and last names of students who possess a specified capability level. The capability level is provided as an input parameter.")
async def get_student_names_by_capability(capability: int = Query(..., description="Capability level of the student")):
    cursor.execute("SELECT T1.f_name, T1.l_name FROM student AS T1 INNER JOIN RA AS T2 ON T1.student_id = T2.student_id WHERE T2.capability = ?", (capability,))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of professors with a specific salary and name
@app.get("/v1/cs_semester/prof_count_by_salary_and_name", operation_id="get_prof_count_by_salary_and_name", summary="Retrieves the number of professors with a specified first name, last name, and salary level. This operation is useful for obtaining a count of professors who meet specific criteria.")
async def get_prof_count_by_salary_and_name(first_name: str = Query(..., description="First name of the professor"), salary: str = Query(..., description="Salary level of the professor"), last_name: str = Query(..., description="Last name of the professor")):
    cursor.execute("SELECT COUNT(T1.prof_id) FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id WHERE T2.first_name = ? AND T1.salary = ? AND T2.last_name = ?", (first_name, salary, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of students registered in a specific course
@app.get("/v1/cs_semester/student_names_by_course", operation_id="get_student_names_by_course", summary="Retrieve the first and last names of students enrolled in a specified course. The course is identified by its name, which is provided as an input parameter.")
async def get_student_names_by_course(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT T1.f_name, T1.l_name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ?", (course_name,))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of students with a specific grade in a specific course and GPA
@app.get("/v1/cs_semester/student_count_by_grade_gpa_course", operation_id="get_student_count_by_grade_gpa_course", summary="Retrieves the number of students who have a certain grade in a specific course and a GPA above a given threshold. The grade, GPA, and course name are provided as input parameters.")
async def get_student_count_by_grade_gpa_course(grade: str = Query(..., description="Grade of the student"), gpa: float = Query(..., description="GPA of the student"), course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT COUNT(student_id) FROM registration WHERE grade = ? AND student_id IN ( SELECT student_id FROM student WHERE gpa > ? AND course_id IN ( SELECT course_id FROM course WHERE name = ? ) )", (grade, gpa, course_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get course names for a specific student
@app.get("/v1/cs_semester/course_names_by_student", operation_id="get_course_names_by_student", summary="Retrieves the names of courses taken by a student identified by their first and last names. The operation uses the provided first and last names to search for the student's registration records and returns the corresponding course names.")
async def get_course_names_by_student(f_name: str = Query(..., description="First name of the student"), l_name: str = Query(..., description="Last name of the student")):
    cursor.execute("SELECT T3.name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T1.f_name = ? AND T1.l_name = ?", (f_name, l_name))
    result = cursor.fetchall()
    if not result:
        return {"course_names": []}
    return {"course_names": [row[0] for row in result]}

# Endpoint to get students with no grade in a specific course
@app.get("/v1/cs_semester/students_no_grade_in_course", operation_id="get_students_no_grade_in_course", summary="Retrieves the first and last names of students who are enrolled in a specific course but have not received a grade. The course is identified by its name.")
async def get_students_no_grade_in_course(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT T1.f_name, T1.l_name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T2.grade IS NULL AND T3.name = ?", (course_name,))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": [{"f_name": row[0], "l_name": row[1]} for row in result]}

# Endpoint to get the top student by SAT score in a specific course
@app.get("/v1/cs_semester/top_student_by_sat_in_course", operation_id="get_top_student_by_sat_in_course", summary="Retrieve the full name of the student with the highest SAT score in a specified course. The operation accepts two possible first names and two possible last names to identify the student, along with the course name. The student's full name is returned as the result.")
async def get_top_student_by_sat_in_course(f_name1: str = Query(..., description="First name of the student"), f_name2: str = Query(..., description="First name of the student"), l_name1: str = Query(..., description="Last name of the student"), l_name2: str = Query(..., description="Last name of the student"), course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT T1.f_name, T1.l_name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE (T1.f_name = ? OR T1.f_name = ?) AND (T1.l_name = ? OR T1.l_name = ?) AND T3.name = ? ORDER BY T2.sat DESC LIMIT 1", (f_name1, f_name2, l_name1, l_name2, course_name))
    result = cursor.fetchone()
    if not result:
        return {"student": {}}
    return {"student": {"f_name": result[0], "l_name": result[1]}}

# Endpoint to get the count of professors with more than a certain number of RAs of a specific gender
@app.get("/v1/cs_semester/prof_count_by_ra_gender", operation_id="get_prof_count_by_ra_gender", summary="Retrieves the count of professors who have more than a specified number of research assistants (RAs) of a certain gender. The gender and minimum RA count are provided as input parameters.")
async def get_prof_count_by_ra_gender(gender: str = Query(..., description="Gender of the RAs"), min_ra_count: int = Query(..., description="Minimum number of RAs")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T2.prof_id FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id WHERE T2.gender = ? GROUP BY T1.prof_id HAVING COUNT(T1.student_id) > ? )", (gender, min_ra_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of students of a specific type in a specific course
@app.get("/v1/cs_semester/student_count_by_type_in_course", operation_id="get_student_count_by_type_in_course", summary="Retrieves the total number of students of a certain type enrolled in a specified course. The operation requires the course name and student type as input parameters to filter the count.")
async def get_student_count_by_type_in_course(course_name: str = Query(..., description="Name of the course"), student_type: str = Query(..., description="Type of the student")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ? AND T1.type = ?", (course_name, student_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average GPA of students associated with a specific professor
@app.get("/v1/cs_semester/average_gpa_by_professor", operation_id="get_average_gpa_by_professor", summary="Retrieves the average GPA of students who have been taught by a specific professor. The professor is identified by their first and last names. The operation calculates the average GPA by summing the GPAs of all students associated with the professor and dividing by the total number of students.")
async def get_average_gpa_by_professor(first_name: str = Query(..., description="First name of the professor"), last_name: str = Query(..., description="Last name of the professor")):
    cursor.execute("SELECT SUM(T3.gpa) / COUNT(T1.student_id) FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T1.student_id = T3.student_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"average_gpa": []}
    return {"average_gpa": result[0]}

# Endpoint to get the average SAT score for a specific course
@app.get("/v1/cs_semester/average_sat_by_course", operation_id="get_average_sat_by_course", summary="Retrieves the average SAT score of students enrolled in a specified course. The calculation is based on the sum of SAT scores divided by the number of students enrolled in the course.")
async def get_average_sat_by_course(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT CAST(SUM(T1.sat) AS REAL) / COUNT(T1.student_id) FROM registration AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T2.name = ?", (course_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_sat": []}
    return {"average_sat": result[0]}

# Endpoint to get the count of students of a specific type
@app.get("/v1/cs_semester/student_count_by_type", operation_id="get_student_count_by_type", summary="Retrieves the total number of students of a specified type. The operation filters students based on the provided type and returns the count.")
async def get_student_count_by_type(student_type: str = Query(..., description="Type of the student")):
    cursor.execute("SELECT COUNT(student_id) FROM student WHERE type = ?", (student_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the students with the highest GPA
@app.get("/v1/cs_semester/students_with_highest_gpa", operation_id="get_students_with_highest_gpa", summary="Retrieves the first and last names of students who have achieved the highest GPA in the student database.")
async def get_students_with_highest_gpa():
    cursor.execute("SELECT f_name, l_name FROM student WHERE gpa = ( SELECT MAX(gpa) FROM student )")
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": [{"f_name": row[0], "l_name": row[1]} for row in result]}

# Endpoint to get the count of students with a specific grade, course credit, and difficulty
@app.get("/v1/cs_semester/count_students_grade_credit_difficulty", operation_id="get_count_students_grade_credit_difficulty", summary="Retrieves the number of students who have achieved a certain grade in a course with a specific credit and difficulty level.")
async def get_count_students_grade_credit_difficulty(grade: str = Query(..., description="Grade of the student"), credit: str = Query(..., description="Credit of the course"), diff: int = Query(..., description="Difficulty of the course")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM registration AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T1.grade = ? AND T2.credit = ? AND T2.diff = ?", (grade, credit, diff))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of students based on course difficulty
@app.get("/v1/cs_semester/count_students_difficulty", operation_id="get_count_students_difficulty", summary="Retrieves the total number of students enrolled in courses of a specified difficulty level.")
async def get_count_students_difficulty(diff: int = Query(..., description="Difficulty of the course")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM registration AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T2.diff = ?", (diff,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get professor popularity based on student's first and last name
@app.get("/v1/cs_semester/professor_popularity_student_name", operation_id="get_professor_popularity_student_name", summary="Retrieves the popularity rating of a professor based on the first and last name of a student. This operation uses the student's name to identify the professor they have rated and returns the corresponding popularity score. The input parameters are the student's first and last names.")
async def get_professor_popularity_student_name(f_name: str = Query(..., description="First name of the student"), l_name: str = Query(..., description="Last name of the student")):
    cursor.execute("SELECT T1.popularity FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T2.student_id = T3.student_id WHERE T3.f_name = ? AND T3.l_name = ?", (f_name, l_name))
    result = cursor.fetchone()
    if not result:
        return {"popularity": []}
    return {"popularity": result[0]}

# Endpoint to get the count of students based on professor's teaching ability and gender
@app.get("/v1/cs_semester/count_students_teachingability_gender", operation_id="get_count_students_teachingability_gender", summary="Retrieves the total number of students taught by professors with a specific teaching ability and gender. The teaching ability and gender are provided as input parameters.")
async def get_count_students_teachingability_gender(teachingability: str = Query(..., description="Teaching ability of the professor"), gender: str = Query(..., description="Gender of the professor")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id WHERE T2.teachingability = ? AND T2.gender = ?", (teachingability, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the professor with the highest teaching ability who has more than a specified number of students
@app.get("/v1/cs_semester/top_professor_teachingability", operation_id="get_top_professor_teachingability", summary="Retrieves the professor with the highest teaching ability who is teaching more than the specified minimum number of students.")
async def get_top_professor_teachingability(min_students: int = Query(..., description="Minimum number of students")):
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T2.first_name, T2.last_name, T2.teachingability FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id GROUP BY T1.prof_id HAVING COUNT(student_id) > ? ) T ORDER BY T.teachingability DESC LIMIT 1", (min_students,))
    result = cursor.fetchone()
    if not result:
        return {"professor": []}
    return {"professor": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the grade result for a student in a specific course
@app.get("/v1/cs_semester/grade_result_student_course", operation_id="get_grade_result_student_course", summary="Retrieves the numerical grade result for a student in a specific course. The result is calculated based on the student's grade in the course, with 'A' being the highest and 'D' or below being the lowest. The operation requires the student's first and last name, as well as the name of the course, to identify the correct record.")
async def get_grade_result_student_course(f_name: str = Query(..., description="First name of the student"), l_name: str = Query(..., description="Last name of the student"), course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT CASE grade WHEN 'A' THEN 4 WHEN 'B' THEN 3 WHEN 'C' THEN 2 ELSE 1 END AS result FROM registration WHERE student_id IN ( SELECT student_id FROM student WHERE f_name = ? AND l_name = ? AND course_id IN ( SELECT course_id FROM course WHERE name = ? ) )", (f_name, l_name, course_name))
    result = cursor.fetchall()
    if not result:
        return {"results": []}
    return {"results": [r[0] for r in result]}

# Endpoint to get the count of courses for a specific student
@app.get("/v1/cs_semester/count_courses_student", operation_id="get_count_courses_student", summary="Retrieves the total number of courses a student is enrolled in. The operation uses the student's first and last names as input parameters to identify the student and calculate the course count.")
async def get_count_courses_student(f_name: str = Query(..., description="First name of the student"), l_name: str = Query(..., description="Last name of the student")):
    cursor.execute("SELECT COUNT(T1.course_id) FROM registration AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T2.f_name = ? AND T2.l_name = ?", (f_name, l_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get student details based on professor's name and student type
@app.get("/v1/cs_semester/student_details_professor_name_type", operation_id="get_student_details_professor_name_type", summary="Retrieves the first and last names of students who are associated with a specific professor and belong to a certain student type. The professor is identified by their first and last names, and the student type is specified as an input parameter.")
async def get_student_details_professor_name_type(prof_first_name: str = Query(..., description="First name of the professor"), student_type: str = Query(..., description="Type of the student"), prof_last_name: str = Query(..., description="Last name of the professor")):
    cursor.execute("SELECT T3.f_name, T3.l_name FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T2.student_id = T3.student_id WHERE T1.first_name = ? AND T3.type = ? AND T1.last_name = ?", (prof_first_name, student_type, prof_last_name))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get the count of students in a specific course
@app.get("/v1/cs_semester/count_students_course", operation_id="get_count_students_course", summary="Retrieves the total number of students enrolled in a specific course. The course is identified by its name, which is provided as an input parameter.")
async def get_count_students_course(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT COUNT(T2.student_id) FROM course AS T1 INNER JOIN registration AS T2 ON T1.course_id = T2.course_id WHERE T1.name = ?", (course_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last names of students registered for a specific course with no grade
@app.get("/v1/cs_semester/students_no_grade_course", operation_id="get_students_no_grade_course", summary="Get the first and last names of students registered for a specific course with no grade")
async def get_students_no_grade_course(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT T1.f_name, T1.l_name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ? AND T2.grade IS NULL", (course_name,))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get the phone numbers of students who got a specific grade in a specific course
@app.get("/v1/cs_semester/students_phone_numbers_course_grade", operation_id="get_students_phone_numbers_course_grade", summary="Retrieve the phone numbers of students who achieved a specified grade in a particular course. This operation requires the course name and the grade as input parameters to filter the students accordingly.")
async def get_students_phone_numbers_course_grade(course_name: str = Query(..., description="Name of the course"), grade: str = Query(..., description="Grade obtained by the student")):
    cursor.execute("SELECT T1.phone_number FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ? AND T2.grade = ?", (course_name, grade))
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": result}

# Endpoint to get the percentage of students of a specific type under a specific professor
@app.get("/v1/cs_semester/percentage_students_type_professor", operation_id="get_percentage_students_type_professor", summary="Retrieves the percentage of students of a specified type who are associated with a professor identified by their first and last names. The operation calculates this percentage by comparing the count of students of the specified type to the total number of students under the professor.")
async def get_percentage_students_type_professor(student_type: str = Query(..., description="Type of the student"), first_name: str = Query(..., description="First name of the professor"), last_name: str = Query(..., description="Last name of the professor")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.type = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.student_id) FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T1.student_id = T3.student_id WHERE T2.first_name = ? AND T2.last_name = ?", (student_type, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of students who got a specific grade in a specific course
@app.get("/v1/cs_semester/percentage_students_grade_course", operation_id="get_percentage_students_grade_course", summary="Retrieves the percentage of students who achieved a specific grade in a given course. The operation calculates this percentage by comparing the number of students who received the specified grade to the total number of students enrolled in the course. The grade and course name are provided as input parameters.")
async def get_percentage_students_grade_course(grade: str = Query(..., description="Grade obtained by the student"), course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.grade = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.student_id) FROM registration AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T2.name = ?", (grade, course_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of courses with a specific difficulty level
@app.get("/v1/cs_semester/count_courses_difficulty", operation_id="get_count_courses_difficulty", summary="Retrieves the total number of courses that match a specified difficulty level. The difficulty level is provided as an input parameter.")
async def get_count_courses_difficulty(difficulty: int = Query(..., description="Difficulty level of the course")):
    cursor.execute("SELECT COUNT(course_id) FROM course WHERE diff = ?", (difficulty,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of professors who graduated from specific universities
@app.get("/v1/cs_semester/professors_graduated_from_universities", operation_id="get_professors_graduated_from_universities", summary="Retrieve the first and last names of professors who graduated from up to eight specified universities. The endpoint returns a list of professors who meet the provided university criteria.")
async def get_professors_graduated_from_universities(university1: str = Query(..., description="University name"), university2: str = Query(..., description="University name"), university3: str = Query(..., description="University name"), university4: str = Query(..., description="University name"), university5: str = Query(..., description="University name"), university6: str = Query(..., description="University name"), university7: str = Query(..., description="University name"), university8: str = Query(..., description="University name")):
    cursor.execute("SELECT first_name, last_name FROM prof WHERE graduate_from IN (?, ?, ?, ?, ?, ?, ?, ?)", (university1, university2, university3, university4, university5, university6, university7, university8))
    result = cursor.fetchall()
    if not result:
        return {"professors": []}
    return {"professors": result}

# Endpoint to get the names of courses with the maximum credit and difficulty
@app.get("/v1/cs_semester/courses_max_credit_difficulty", operation_id="get_courses_max_credit_difficulty", summary="Retrieves the names of courses that have the highest credit and difficulty values in the course database.")
async def get_courses_max_credit_difficulty():
    cursor.execute("SELECT name FROM course WHERE credit = (SELECT MAX(credit) FROM course) AND diff = (SELECT MAX(diff) FROM course)")
    result = cursor.fetchall()
    if not result:
        return {"courses": []}
    return {"courses": result}

# Endpoint to get the count of students of a specific type with the maximum intelligence
@app.get("/v1/cs_semester/count_students_type_max_intelligence", operation_id="get_count_students_type_max_intelligence", summary="Retrieves the total number of students of a specified type who possess the highest intelligence level in the system.")
async def get_count_students_type_max_intelligence(student_type: str = Query(..., description="Type of the student")):
    cursor.execute("SELECT COUNT(student_id) FROM student WHERE type = ? AND intelligence = (SELECT MAX(intelligence) FROM student)", (student_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of professors of a specific gender with the maximum popularity
@app.get("/v1/cs_semester/count_professors_gender_max_popularity", operation_id="get_count_professors_gender_max_popularity", summary="Retrieves the total number of professors of a specified gender who have the highest popularity rating. The gender is provided as an input parameter.")
async def get_count_professors_gender_max_popularity(gender: str = Query(..., description="Gender of the professor")):
    cursor.execute("SELECT COUNT(prof_id) FROM prof WHERE gender = ? AND popularity = (SELECT MAX(popularity) FROM prof)", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of students who got a specific grade in a specific course
@app.get("/v1/cs_semester/count_students_grade_course", operation_id="get_count_students_grade_course", summary="Get the count of students who got a specific grade in a specific course")
async def get_count_students_grade_course(grade: str = Query(..., description="Grade obtained by the student"), course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT COUNT(T2.student_id) FROM course AS T1 INNER JOIN registration AS T2 ON T1.course_id = T2.course_id WHERE T2.grade = ? AND T1.name = ?", (grade, course_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the GPA of students who are RAs with a specific salary
@app.get("/v1/cs_semester/ra_student_gpa", operation_id="get_ra_student_gpa", summary="Retrieves the GPA of students who hold a Research Assistant (RA) position with a specified salary. The operation returns the GPA of students who meet the salary criteria, providing insights into the academic performance of RAs at a particular salary level.")
async def get_ra_student_gpa(salary: str = Query(..., description="Salary of the RA")):
    cursor.execute("SELECT T2.gpa FROM RA AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T1.salary = ?", (salary,))
    result = cursor.fetchall()
    if not result:
        return {"gpa": []}
    return {"gpa": [row[0] for row in result]}

# Endpoint to get the name of the course with the highest number of students who got a specific grade and difficulty
@app.get("/v1/cs_semester/top_course_by_grade_diff", operation_id="get_top_course_by_grade_diff", summary="Retrieves the name of the course with the highest enrollment for students who received a specific grade and a certain level of difficulty. The grade and difficulty are provided as input parameters.")
async def get_top_course_by_grade_diff(grade: str = Query(..., description="Grade of the students"), diff: int = Query(..., description="Difficulty of the course")):
    cursor.execute("SELECT T2.name FROM registration AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T1.grade = ? AND T2.diff = ? GROUP BY T2.name ORDER BY COUNT(T1.student_id) DESC LIMIT 1", (grade, diff))
    result = cursor.fetchone()
    if not result:
        return {"course_name": []}
    return {"course_name": result[0]}

# Endpoint to get the count of courses taken by students with the highest GPA
@app.get("/v1/cs_semester/course_count_top_gpa", operation_id="get_course_count_top_gpa", summary="Retrieves the total number of courses taken by students who have achieved the highest GPA in the system. This operation provides a quantitative measure of course participation among top-performing students.")
async def get_course_count_top_gpa():
    cursor.execute("SELECT COUNT(course_id) FROM registration WHERE student_id IN ( SELECT student_id FROM student WHERE gpa = ( SELECT MAX(gpa) FROM student ) )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct course names based on SAT score
@app.get("/v1/cs_semester/distinct_courses_by_sat", operation_id="get_distinct_courses_by_sat", summary="Retrieves a unique list of course names that have students with the specified SAT score. The operation filters the registration data based on the provided SAT score and returns the distinct course names from the filtered data.")
async def get_distinct_courses_by_sat(sat: int = Query(..., description="SAT score")):
    cursor.execute("SELECT DISTINCT T2.name FROM registration AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T1.sat = ?", (sat,))
    result = cursor.fetchall()
    if not result:
        return {"course_names": []}
    return {"course_names": [row[0] for row in result]}

# Endpoint to get course names based on SAT score and intelligence level
@app.get("/v1/cs_semester/courses_by_sat_intelligence", operation_id="get_courses_by_sat_intelligence", summary="Retrieves the names of courses that are associated with students who have a specific SAT score and intelligence level. The operation filters the courses based on the provided SAT score and intelligence level, returning only the names of the relevant courses.")
async def get_courses_by_sat_intelligence(sat: int = Query(..., description="SAT score"), intelligence: int = Query(..., description="Intelligence level")):
    cursor.execute("SELECT T3.name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T2.sat = ? AND T1.intelligence = ?", (sat, intelligence))
    result = cursor.fetchall()
    if not result:
        return {"course_names": []}
    return {"course_names": [row[0] for row in result]}

# Endpoint to get the top course name based on grade and specific course names
@app.get("/v1/cs_semester/top_course_by_grade_names", operation_id="get_top_course_by_grade_names", summary="Retrieves the most popular course name among students with a specific grade who are enrolled in a set of given courses. The grade and course names are provided as input parameters. The result is determined by counting the number of students enrolled in each course and selecting the course with the highest count.")
async def get_top_course_by_grade_names(grade: str = Query(..., description="Grade of the students"), course_name1: str = Query(..., description="First course name"), course_name2: str = Query(..., description="Second course name")):
    cursor.execute("SELECT T2.name FROM registration AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T1.grade = ? AND T2.name IN (?, ?) GROUP BY T2.name ORDER BY COUNT(T1.student_id) DESC LIMIT 1", (grade, course_name1, course_name2))
    result = cursor.fetchone()
    if not result:
        return {"course_name": []}
    return {"course_name": result[0]}

# Endpoint to get the popularity of the professor with the highest number of RAs and highest capability
@app.get("/v1/cs_semester/top_prof_popularity", operation_id="get_top_prof_popularity", summary="Retrieves the popularity rating of the professor who has the most research assistants (RAs) and the highest capability. The popularity is determined by the number of RAs and their respective capabilities.")
async def get_top_prof_popularity():
    cursor.execute("SELECT T2.popularity FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id GROUP BY T1.prof_id, T1.capability ORDER BY COUNT(T1.student_id) DESC, T1.capability DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"popularity": []}
    return {"popularity": result[0]}

# Endpoint to get the average number of students per course based on course difficulty
@app.get("/v1/cs_semester/avg_students_per_course_diff", operation_id="get_avg_students_per_course_diff", summary="Retrieves the average number of students enrolled per course, filtered by the specified course difficulty level.")
async def get_avg_students_per_course_diff(diff: int = Query(..., description="Difficulty of the course")):
    cursor.execute("SELECT CAST(COUNT(T1.student_id) AS REAL) / COUNT(DISTINCT T2.course_id) FROM registration AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T2.diff = ?", (diff,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the count of students with no grade and GPA within a specific range
@app.get("/v1/cs_semester/student_count_no_grade_gpa_range", operation_id="get_student_count_no_grade_gpa_range", summary="Retrieves the number of students who have not received a grade and whose GPA falls within a specified range. The range is defined by the minimum and maximum GPA values provided as input parameters.")
async def get_student_count_no_grade_gpa_range(min_gpa: float = Query(..., description="Minimum GPA"), max_gpa: float = Query(..., description="Maximum GPA")):
    cursor.execute("SELECT COUNT(T2.student_id) FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id WHERE T2.grade IS NULL AND T1.gpa BETWEEN ? AND ?", (min_gpa, max_gpa))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of students with a specific grade and type
@app.get("/v1/cs_semester/student_count_by_grade_and_type", operation_id="get_student_count", summary="Retrieves the total number of students with a specified grade and type. The grade and type are used to filter the student population, providing a count of students who meet the specified criteria.")
async def get_student_count(grade: str = Query(..., description="Grade of the student"), student_type: str = Query(..., description="Type of the student (e.g., UG for undergraduate)")):
    cursor.execute("SELECT COUNT(T2.student_id) FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id WHERE T2.grade = ? AND T1.type = ?", (grade, student_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average GPA of students with a specific salary and capability
@app.get("/v1/cs_semester/avg_gpa_by_salary_and_capability", operation_id="get_avg_gpa", summary="Retrieves the average GPA of students who share a common salary level and capability. The response includes the average GPA along with the first and last names of each student.")
async def get_avg_gpa(salary: str = Query(..., description="Salary level of the student (e.g., high)"), capability: int = Query(..., description="Capability level of the student")):
    cursor.execute("SELECT AVG(T2.gpa), T2.f_name, T2.l_name FROM RA AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T1.salary = ? AND T1.capability = ? GROUP BY T2.student_id", (salary, capability))
    result = cursor.fetchall()
    if not result:
        return {"avg_gpa": []}
    return {"avg_gpa": result}

# Endpoint to get the professor and student IDs of RAs with the minimum capability
@app.get("/v1/cs_semester/ra_with_min_capability", operation_id="get_ra_with_min_capability", summary="Retrieves the unique identifiers of professors and students who are research assistants (RAs) with the lowest capability rating. This operation provides a list of RAs who have the minimum capability score, which can be useful for identifying those who may need additional support or resources.")
async def get_ra_with_min_capability():
    cursor.execute("SELECT prof_id, student_id FROM RA WHERE capability = ( SELECT MIN(capability) FROM RA )")
    result = cursor.fetchall()
    if not result:
        return {"ra_ids": []}
    return {"ra_ids": result}

# Endpoint to get the names of professors who graduated from a specific university
@app.get("/v1/cs_semester/prof_by_graduate_university", operation_id="get_prof_by_university", summary="Retrieves the first and last names of professors who have graduated from a specified university. The university is provided as an input parameter.")
async def get_prof_by_university(graduate_from: str = Query(..., description="University from which the professor graduated")):
    cursor.execute("SELECT first_name, last_name FROM prof WHERE graduate_from = ?", (graduate_from,))
    result = cursor.fetchall()
    if not result:
        return {"professors": []}
    return {"professors": result}

# Endpoint to get the course and student IDs for registrations with no grade
@app.get("/v1/cs_semester/registrations_with_no_grade", operation_id="get_registrations_with_no_grade", summary="Retrieves a list of course and student IDs for registrations that do not have a grade assigned. This endpoint is useful for identifying students who have not yet received a grade for a course they are registered in.")
async def get_registrations_with_no_grade():
    cursor.execute("SELECT course_id, student_id FROM registration WHERE grade IS NULL OR grade = ''")
    result = cursor.fetchall()
    if not result:
        return {"registrations": []}
    return {"registrations": result}

# Endpoint to get the ratio of male to female professors
@app.get("/v1/cs_semester/male_to_female_prof_ratio", operation_id="get_male_to_female_ratio", summary="Retrieves the ratio of male to female professors in the database. This operation calculates the proportion of male professors to female professors, providing a statistical comparison of gender distribution among faculty members.")
async def get_male_to_female_ratio():
    cursor.execute("SELECT CAST(SUM(CASE WHEN gender = 'Male' THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN gender = 'Female' THEN 1 ELSE 0 END) FROM prof")
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the course names and credits with the minimum difficulty
@app.get("/v1/cs_semester/courses_with_min_difficulty", operation_id="get_courses_with_min_difficulty", summary="Retrieves the names and credits of courses with the lowest difficulty level.")
async def get_courses_with_min_difficulty():
    cursor.execute("SELECT name, credit FROM course WHERE diff = ( SELECT MIN(diff) FROM course )")
    result = cursor.fetchall()
    if not result:
        return {"courses": []}
    return {"courses": result}

# Endpoint to get the most popular professor's RA details
@app.get("/v1/cs_semester/most_popular_prof_ra_details", operation_id="get_most_popular_prof_ra_details", summary="Retrieves the first name, last name, and GPA of the student who is the research assistant (RA) of the most popular professor. The popularity of a professor is determined by the number of students who have taken their courses. The student with the highest GPA is returned if there are multiple RAs for the most popular professor.")
async def get_most_popular_prof_ra_details():
    cursor.execute("SELECT T3.f_name, T3.l_name, T3.gpa FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T2.student_id = T3.student_id ORDER BY T1.popularity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ra_details": []}
    return {"ra_details": result}

# Endpoint to get the details of RAs with a specific salary
@app.get("/v1/cs_semester/ra_details_by_salary", operation_id="get_ra_details_by_salary", summary="Retrieves the first name, last name, and email of students who are RAs and receive a specified salary level. The salary level is provided as an input parameter.")
async def get_ra_details_by_salary(salary: str = Query(..., description="Salary level of the RA (e.g., free)")):
    cursor.execute("SELECT T2.f_name, T2.l_name, T2.email FROM RA AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T1.salary = ?", (salary,))
    result = cursor.fetchall()
    if not result:
        return {"ra_details": []}
    return {"ra_details": result}

# Endpoint to get the details of RAs under a specific professor
@app.get("/v1/cs_semester/ra_details_by_professor", operation_id="get_ra_details_by_professor", summary="Retrieves the first name, last name, capability, and GPA of research assistants (RAs) associated with a specific professor. The professor is identified by their first and last names.")
async def get_ra_details_by_professor(first_name: str = Query(..., description="First name of the professor"), last_name: str = Query(..., description="Last name of the professor")):
    cursor.execute("SELECT T3.f_name, T3.l_name, T2.capability, T3.gpa FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T2.student_id = T3.student_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"ra_details": []}
    return {"ra_details": result}

# Endpoint to get student details and their grades for a specific course
@app.get("/v1/cs_semester/student_grades_by_course", operation_id="get_student_grades_by_course", summary="Retrieves the first name, last name, and grade of students enrolled in a specified course. The course is identified by its name, which is provided as an input parameter.")
async def get_student_grades_by_course(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT T1.f_name, T1.l_name, T2.grade FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ?", (course_name,))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get the top student by course difficulty for a specific grade
@app.get("/v1/cs_semester/top_student_by_grade", operation_id="get_top_student_by_grade", summary="Retrieves the full name of the student who has the highest course difficulty grade in a specific grade level. The grade level is provided as an input parameter.")
async def get_top_student_by_grade(grade: str = Query(..., description="Grade of the student")):
    cursor.execute("SELECT T1.f_name, T1.l_name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T2.grade = ? ORDER BY T3.diff DESC LIMIT 1", (grade,))
    result = cursor.fetchone()
    if not result:
        return {"student": []}
    return {"student": result}

# Endpoint to get professor details based on student's first and last name
@app.get("/v1/cs_semester/professor_details_by_student", operation_id="get_professor_details_by_student", summary="Retrieves the first name, last name, and graduation details of professors who have advised a student with the specified first and last names. The operation uses the student's first and last names as input parameters to filter the results.")
async def get_professor_details_by_student(student_f_name: str = Query(..., description="First name of the student"), student_l_name: str = Query(..., description="Last name of the student")):
    cursor.execute("SELECT T1.first_name, T1.last_name, T1.graduate_from FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T2.student_id = T3.student_id WHERE T3.f_name = ? AND T3.l_name = ?", (student_f_name, student_l_name))
    result = cursor.fetchall()
    if not result:
        return {"professors": []}
    return {"professors": result}

# Endpoint to get the top student by SAT score for a specific course
@app.get("/v1/cs_semester/top_student_by_course", operation_id="get_top_student_by_course", summary="Retrieves the full name of the student with the highest SAT score in a specified course. The course is identified by its name, which is provided as an input parameter.")
async def get_top_student_by_course(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT T1.f_name, T1.l_name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ? ORDER BY T2.sat DESC LIMIT 1", (course_name,))
    result = cursor.fetchone()
    if not result:
        return {"student": []}
    return {"student": result}

# Endpoint to get the average GPA of a student
@app.get("/v1/cs_semester/student_average_gpa", operation_id="get_student_average_gpa", summary="Retrieves the average GPA of a student by calculating the sum of the product of course credits and corresponding grade points, divided by the total number of credits. The calculation is based on the student's first and last name, which are used to filter the relevant records.")
async def get_student_average_gpa(student_f_name: str = Query(..., description="First name of the student"), student_l_name: str = Query(..., description="Last name of the student")):
    cursor.execute("SELECT CAST(SUM(T3.credit * CASE T1.grade WHEN 'A' THEN 4 WHEN 'B' THEN 3 WHEN 'C' THEN 2 WHEN 'D' THEN 1 ELSE 1 END) AS REAL) / COUNT(T3.credit) FROM registration AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T1.course_id = T3.course_id WHERE T2.f_name = ? AND T2.l_name = ?", (student_f_name, student_l_name))
    result = cursor.fetchone()
    if not result:
        return {"average_gpa": []}
    return {"average_gpa": result[0]}

# Endpoint to get distinct student first names based on type and GPA
@app.get("/v1/cs_semester/distinct_student_names", operation_id="get_distinct_student_names", summary="Retrieves a list of unique first names of students who meet the specified criteria. The operation filters students based on their type and GPA, returning only those with a GPA higher than the provided minimum value.")
async def get_distinct_student_names(student_type: str = Query(..., description="Type of the student (e.g., UG for undergraduate)"), min_gpa: float = Query(..., description="Minimum GPA of the student")):
    cursor.execute("SELECT DISTINCT T1.f_name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T1.type = ? AND T1.gpa > ?", (student_type, min_gpa))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get student details and their capabilities based on professor's graduation
@app.get("/v1/cs_semester/student_capabilities_by_professor", operation_id="get_student_capabilities_by_professor", summary="Retrieves the first name, last name, and capability of students who are associated with professors that graduated from a specified institution. This operation provides a comprehensive view of the capabilities of students linked to professors from a particular educational background.")
async def get_student_capabilities_by_professor(graduate_from: str = Query(..., description="Institution from which the professor graduated")):
    cursor.execute("SELECT T3.f_name, T3.l_name, T2.capability FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T2.student_id = T3.student_id WHERE T1.graduate_from = ?", (graduate_from,))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get student details based on RA salary and maximum capability
@app.get("/v1/cs_semester/student_details_by_ra", operation_id="get_student_details_by_ra", summary="Retrieves the first name, last name, email, and intelligence of students who are Resident Assistants (RAs) with a specified salary and the highest capability. The salary parameter is used to filter the RAs by their salary.")
async def get_student_details_by_ra(salary: str = Query(..., description="Salary of the RA")):
    cursor.execute("SELECT f_name, l_name, email, intelligence FROM student WHERE student_id IN ( SELECT student_id FROM RA WHERE salary = ? AND capability = ( SELECT MAX(capability) FROM RA ) )", (salary,))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get the top course by teaching ability for a specific professor gender
@app.get("/v1/cs_semester/top_course_by_professor_gender", operation_id="get_top_course_by_professor_gender", summary="Retrieves the name and credit of the top-rated course, based on the teaching ability of professors of a specified gender. The course data is determined by considering the professor's gender and teaching ability, as well as the student's registration information.")
async def get_top_course_by_professor_gender(gender: str = Query(..., description="Gender of the professor")):
    cursor.execute("SELECT T5.name, T5.credit FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T1.student_id = T3.student_id INNER JOIN registration AS T4 ON T3.student_id = T4.student_id INNER JOIN course AS T5 ON T4.course_id = T5.course_id WHERE T2.gender = ? ORDER BY T2.teachingability DESC LIMIT 1", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"course": []}
    return {"course": result}

# Endpoint to get the count of professors based on gender
@app.get("/v1/cs_semester/professor_count_by_gender", operation_id="get_professor_count_by_gender", summary="Retrieves the total number of professors categorized by gender. The operation requires the gender as an input parameter to filter the count accordingly.")
async def get_professor_count_by_gender(gender: str = Query(..., description="Gender of the professor")):
    cursor.execute("SELECT COUNT(prof_id) FROM prof WHERE gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of the course with the highest difficulty
@app.get("/v1/cs_semester/course_with_highest_difficulty", operation_id="get_course_with_highest_difficulty", summary="Get the name of the course with the highest difficulty")
async def get_course_with_highest_difficulty():
    cursor.execute("SELECT name FROM course WHERE diff = ( SELECT MAX(diff) FROM course )")
    result = cursor.fetchone()
    if not result:
        return {"course_name": []}
    return {"course_name": result[0]}

# Endpoint to get the count of students with a GPA in a specific range and type
@app.get("/v1/cs_semester/student_count_gpa_range_type", operation_id="get_student_count_gpa_range_type", summary="Retrieves the number of students within a specified GPA range and type. The GPA range is defined by a minimum and maximum value, while the student type is indicated by a specific code (e.g., UG for undergraduate).")
async def get_student_count_gpa_range_type(min_gpa: float = Query(..., description="Minimum GPA"), max_gpa: float = Query(..., description="Maximum GPA"), student_type: str = Query(..., description="Type of student (e.g., UG for undergraduate)")):
    cursor.execute("SELECT COUNT(student_id) FROM student WHERE gpa BETWEEN ? AND ? AND type = ?", (min_gpa, max_gpa, student_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the credit of a course by its name
@app.get("/v1/cs_semester/course_credit_by_name", operation_id="get_course_credit_by_name", summary="Retrieves the credit value associated with a specific course, identified by its unique name. The course name is provided as an input parameter.")
async def get_course_credit_by_name(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT credit FROM course WHERE name = ?", (course_name,))
    result = cursor.fetchone()
    if not result:
        return {"credit": []}
    return {"credit": result[0]}

# Endpoint to get student IDs registered in a specific course with a specific GPA
@app.get("/v1/cs_semester/student_ids_course_gpa", operation_id="get_student_ids_course_gpa", summary="Retrieves the IDs of students who are enrolled in a specific course and have a particular GPA. The course is identified by its name, and the GPA is a numerical value representing the student's academic performance.")
async def get_student_ids_course_gpa(course_name: str = Query(..., description="Name of the course"), gpa: float = Query(..., description="GPA of the student")):
    cursor.execute("SELECT T2.student_id FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ? AND T1.gpa = ?", (course_name, gpa))
    result = cursor.fetchall()
    if not result:
        return {"student_ids": []}
    return {"student_ids": [row[0] for row in result]}

# Endpoint to get the last name of the student with the highest SAT score in a specific course
@app.get("/v1/cs_semester/top_sat_student_last_name", operation_id="get_top_sat_student_last_name", summary="Retrieves the last name of the student who achieved the highest SAT score in a specified course. The course is identified by its name, which is provided as an input parameter.")
async def get_top_sat_student_last_name(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT T1.l_name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ? ORDER BY T2.sat DESC LIMIT 1", (course_name,))
    result = cursor.fetchone()
    if not result:
        return {"last_name": []}
    return {"last_name": result[0]}

# Endpoint to get the count of RAs with a specific salary and GPA above a certain value
@app.get("/v1/cs_semester/ra_count_salary_gpa", operation_id="get_ra_count_salary_gpa", summary="Retrieves the number of research assistants (RAs) with a specified salary and a GPA exceeding a given minimum value. This operation considers both the salary of the RA and the GPA of the corresponding student.")
async def get_ra_count_salary_gpa(salary: str = Query(..., description="Salary of the RA"), min_gpa: float = Query(..., description="Minimum GPA")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM RA AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T1.salary = ? AND T2.gpa > ?", (salary, min_gpa))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of the course with the highest SAT score for a specific student type
@app.get("/v1/cs_semester/top_sat_course_name", operation_id="get_top_sat_course_name", summary="Retrieves the name of the course with the highest SAT score for a specified student type. The operation filters students by their type and identifies the course with the highest SAT score associated with that student type.")
async def get_top_sat_course_name(student_type: str = Query(..., description="Type of student (e.g., UG for undergraduate)")):
    cursor.execute("SELECT T3.name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T1.type = ? ORDER BY T2.sat DESC LIMIT 1", (student_type,))
    result = cursor.fetchone()
    if not result:
        return {"course_name": []}
    return {"course_name": result[0]}

# Endpoint to get the capabilities of RAs with a specific type and intelligence level
@app.get("/v1/cs_semester/ra_capabilities_type_intelligence", operation_id="get_ra_capabilities_type_intelligence", summary="Retrieves the capabilities of Research Assistants (RAs) who are of a specified student type and possess an intelligence level equal to or greater than a given threshold. The operation filters RAs based on their type and intelligence, then returns their associated capabilities.")
async def get_ra_capabilities_type_intelligence(student_type: str = Query(..., description="Type of student (e.g., RPG)"), min_intelligence: int = Query(..., description="Minimum intelligence level")):
    cursor.execute("SELECT T1.capability FROM RA AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T2.type = ? AND T2.intelligence >= ?", (student_type, min_intelligence))
    result = cursor.fetchall()
    if not result:
        return {"capabilities": []}
    return {"capabilities": [row[0] for row in result]}

# Endpoint to get the count of students with a specific grade and intelligence level
@app.get("/v1/cs_semester/student_count_grade_intelligence", operation_id="get_student_count_grade_intelligence", summary="Retrieves the total number of students who have a specified grade and intelligence level. The grade parameter filters students based on their academic performance, while the intelligence parameter narrows down the results based on the students' cognitive abilities. This operation provides a quantitative measure of students who meet the specified criteria.")
async def get_student_count_grade_intelligence(grade: str = Query(..., description="Grade of the student (e.g., B)"), intelligence: int = Query(..., description="Intelligence level of the student")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM registration AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T1.grade = ? AND T2.intelligence = ?", (grade, intelligence))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difficulty of courses for students with a specific grade and intelligence level
@app.get("/v1/cs_semester/course_difficulty_grade_intelligence", operation_id="get_course_difficulty_grade_intelligence", summary="Retrieve the difficulty level of courses that students with a specific grade and intelligence level have taken. The grade and intelligence level are used to filter the courses and determine their difficulty.")
async def get_course_difficulty_grade_intelligence(grade: str = Query(..., description="Grade of the student (e.g., A)"), intelligence: int = Query(..., description="Intelligence level of the student")):
    cursor.execute("SELECT T3.diff FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T2.grade = ? AND T1.intelligence = ?", (grade, intelligence))
    result = cursor.fetchall()
    if not result:
        return {"difficulties": []}
    return {"difficulties": [row[0] for row in result]}

# Endpoint to get the count of students with a specific capability, ordered by professor popularity
@app.get("/v1/cs_semester/count_students_by_capability", operation_id="get_count_students_by_capability", summary="Retrieves the number of students who possess a specified capability, sorted by the popularity of their professors. The result is limited to the most popular professor.")
async def get_count_students_by_capability(capability: int = Query(..., description="Capability of the student")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id WHERE T1.capability = ? ORDER BY T2.popularity DESC LIMIT 1", (capability,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get course names where students received a specific grade
@app.get("/v1/cs_semester/course_names_by_grade", operation_id="get_course_names_by_grade", summary="Retrieve the names of courses in which students have earned a specified grade. The operation filters courses based on the provided grade and returns a list of corresponding course names.")
async def get_course_names_by_grade(grade: str = Query(..., description="Grade received by the student")):
    cursor.execute("SELECT T1.name FROM course AS T1 INNER JOIN registration AS T2 ON T1.course_id = T2.course_id WHERE T2.grade = ?", (grade,))
    result = cursor.fetchall()
    if not result:
        return {"course_names": []}
    return {"course_names": [row[0] for row in result]}

# Endpoint to get the capability of a student based on their first and last name
@app.get("/v1/cs_semester/student_capability_by_name", operation_id="get_student_capability_by_name", summary="Retrieves the capability of a student by matching their first and last names in the database. The capability is determined based on the student's ID and their corresponding record in the RA table.")
async def get_student_capability_by_name(f_name: str = Query(..., description="First name of the student"), l_name: str = Query(..., description="Last name of the student")):
    cursor.execute("SELECT T2.capability FROM student AS T1 INNER JOIN RA AS T2 ON T1.student_id = T2.student_id WHERE T1.f_name = ? AND T1.l_name = ?", (f_name, l_name))
    result = cursor.fetchone()
    if not result:
        return {"capability": []}
    return {"capability": result[0]}

# Endpoint to get the count of students enrolled in courses with specific credits and GPA
@app.get("/v1/cs_semester/count_students_by_credit_gpa", operation_id="get_count_students_by_credit_gpa", summary="Retrieves the number of students who are enrolled in courses with a specified credit value and have a certain GPA. This operation provides a count of students based on the provided credit and GPA parameters.")
async def get_count_students_by_credit_gpa(credit: int = Query(..., description="Credit of the course"), gpa: float = Query(..., description="GPA of the student")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.credit = ? AND T1.gpa = ?", (credit, gpa))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of RA students with specific GPA and salary
@app.get("/v1/cs_semester/count_ra_students_by_gpa_salary", operation_id="get_count_ra_students_by_gpa_salary", summary="Retrieves the number of research assistant (RA) students who have a specific GPA and salary. The GPA and salary values are used to filter the RA students, providing a count of those who meet the specified criteria.")
async def get_count_ra_students_by_gpa_salary(gpa: float = Query(..., description="GPA of the student"), salary: str = Query(..., description="Salary of the RA")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM RA AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T2.gpa = ? AND T1.salary = ?", (gpa, salary))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get emails of students with a specific grade in courses with difficulty above a certain threshold
@app.get("/v1/cs_semester/student_emails_by_grade_difficulty", operation_id="get_student_emails_by_grade_difficulty", summary="Retrieve the email addresses of students who have achieved a specified grade in courses that are more challenging than the average course. The operation filters students based on their grade and identifies courses with a difficulty level exceeding 80% of the average difficulty. The result is a list of email addresses for students who meet these criteria.")
async def get_student_emails_by_grade_difficulty(grade: str = Query(..., description="Grade received by the student")):
    cursor.execute("SELECT T2.email FROM registration AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T1.course_id = T3.course_id WHERE T1.grade = ? GROUP BY T3.diff HAVING T3.diff > AVG(T3.diff) * 0.8", (grade,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get the percentage of RAs with a specific salary under professors with a certain teaching ability
@app.get("/v1/cs_semester/percentage_ra_salary_by_teachingability", operation_id="get_percentage_ra_salary_by_teachingability", summary="Retrieves the percentage of research assistants (RAs) earning a specified salary who are supervised by professors with a teaching ability below a given threshold. This operation calculates the proportion of RAs with the specified salary relative to the total number of RAs under professors with lower teaching ability. The input parameters include the salary of the RA and the teaching ability threshold for professors.")
async def get_percentage_ra_salary_by_teachingability(salary: str = Query(..., description="Salary of the RA"), teachingability: int = Query(..., description="Teaching ability of the professor")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.salary = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.salary) FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id WHERE T2.teachingability < ?", (salary, teachingability))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average teaching ability of the most popular professor
@app.get("/v1/cs_semester/average_teachingability_most_popular_prof", operation_id="get_average_teachingability_most_popular_prof", summary="Retrieves the average teaching ability score of the professor with the highest popularity rating. The teaching ability score is calculated by summing the teachingability values and dividing by the total number of records for the most popular professor.")
async def get_average_teachingability_most_popular_prof():
    cursor.execute("SELECT CAST(SUM(teachingability) AS REAL) / COUNT(prof_id) FROM prof WHERE popularity = ( SELECT MAX(popularity) FROM prof )")
    result = cursor.fetchone()
    if not result:
        return {"average_teachingability": []}
    return {"average_teachingability": result[0]}

# Endpoint to get the average SAT score of students with a specific grade
@app.get("/v1/cs_semester/average_sat_by_grade", operation_id="get_average_sat_by_grade", summary="Retrieves the average SAT score of students who received a specific grade. The grade is provided as an input parameter, and the operation calculates the average SAT score based on the corresponding student records.")
async def get_average_sat_by_grade(grade: str = Query(..., description="Grade received by the student")):
    cursor.execute("SELECT CAST(SUM(sat) AS REAL) / COUNT(course_id) FROM registration WHERE grade = ?", (grade,))
    result = cursor.fetchone()
    if not result:
        return {"average_sat": []}
    return {"average_sat": result[0]}

# Endpoint to get student details based on GPA and intelligence
@app.get("/v1/cs_semester/student_details_by_gpa_intelligence", operation_id="get_student_details_by_gpa_intelligence", summary="Retrieves the first and last names, as well as the phone numbers, of students who have a GPA greater than the provided value and an intelligence level less than the provided value.")
async def get_student_details_by_gpa_intelligence(gpa: float = Query(..., description="GPA of the student"), intelligence: int = Query(..., description="Intelligence level of the student")):
    cursor.execute("SELECT f_name, l_name, phone_number FROM student WHERE gpa > ? AND intelligence < ?", (gpa, intelligence))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": [{"f_name": row[0], "l_name": row[1], "phone_number": row[2]} for row in result]}

# Endpoint to get the first and last names of students who have a capability greater than the average capability of all RAs
@app.get("/v1/cs_semester/students_above_avg_capability", operation_id="get_students_above_avg_capability", summary="Retrieves the first and last names of students who are Research Assistants (RAs) and have a capability score higher than the average capability score of all RAs.")
async def get_students_above_avg_capability():
    cursor.execute("SELECT T1.f_name, T1.l_name FROM student AS T1 INNER JOIN RA AS T2 ON T1.student_id = T2.student_id WHERE T2.capability > ( SELECT AVG(capability) FROM RA )")
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get the first and last names of students and the course name for students with a specific intelligence and GPA less than a specified value
@app.get("/v1/cs_semester/students_course_by_intelligence_gpa", operation_id="get_students_course_by_intelligence_gpa", summary="Retrieves the first and last names of students along with their respective course names, based on a specified intelligence level and a maximum GPA threshold. This operation filters students by their intelligence level and GPA, returning only those who meet the specified criteria.")
async def get_students_course_by_intelligence_gpa(intelligence: int = Query(..., description="Intelligence level of the student"), gpa: float = Query(..., description="GPA of the student")):
    cursor.execute("SELECT T1.f_name, T1.l_name, T3.name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T1.intelligence = ? AND T1.gpa < ?", (intelligence, gpa))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get the average capability of RAs for students with a GPA less than a specified value
@app.get("/v1/cs_semester/avg_capability_by_gpa", operation_id="get_avg_capability_by_gpa", summary="Retrieves the average capability score of research assistants (RAs) for students with a GPA below the specified value. This operation calculates the mean capability score by summing the individual scores and dividing by the total number of students with a GPA less than the provided value.")
async def get_avg_capability_by_gpa(gpa: float = Query(..., description="GPA of the student")):
    cursor.execute("SELECT CAST(SUM(T1.capability) AS REAL) / COUNT(T1.student_id) FROM RA AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T2.gpa < ?", (gpa,))
    result = cursor.fetchone()
    if not result:
        return {"average_capability": []}
    return {"average_capability": result[0]}

# Endpoint to get the first and last names of professors associated with students of a specific intelligence level
@app.get("/v1/cs_semester/profs_by_student_intelligence", operation_id="get_profs_by_student_intelligence", summary="Retrieves the first and last names of professors who have students with a specified intelligence level. The intelligence level is provided as an input parameter.")
async def get_profs_by_student_intelligence(intelligence: int = Query(..., description="Intelligence level of the student")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T2.student_id = T3.student_id WHERE T3.intelligence = ?", (intelligence,))
    result = cursor.fetchall()
    if not result:
        return {"professors": []}
    return {"professors": result}

# Endpoint to get the average GPA of students for courses with specific difficulty levels
@app.get("/v1/cs_semester/avg_gpa_by_course_difficulty", operation_id="get_avg_gpa_by_course_difficulty", summary="Retrieves the average GPA of students who have taken courses with the specified difficulty levels. The difficulty levels are provided as input parameters, and the operation returns the average GPA for each difficulty level separately.")
async def get_avg_gpa_by_course_difficulty(diff1: int = Query(..., description="First difficulty level of the course"), diff2: int = Query(..., description="Second difficulty level of the course")):
    cursor.execute("SELECT AVG(T1.gpa) FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.diff IN (?, ?) GROUP BY T3.diff", (diff1, diff2))
    result = cursor.fetchall()
    if not result:
        return {"average_gpa": []}
    return {"average_gpa": result}

# Endpoint to get the first and last names of students and their capability for students with no grade or an empty grade
@app.get("/v1/cs_semester/students_capability_no_grade", operation_id="get_students_capability_no_grade", summary="Retrieves the first and last names of students along with their respective capabilities. This operation focuses on students who have not received a grade or have an empty grade. The data is sourced from the 'RA', 'student', and 'registration' tables, with the 'student_id' serving as the linking attribute.")
async def get_students_capability_no_grade():
    cursor.execute("SELECT T2.f_name, T2.l_name, T1.capability FROM RA AS T1 INNER JOIN student AS T2 ON T2.student_id = T1.student_id INNER JOIN registration AS T3 ON T2.student_id = T3.student_id WHERE T3.grade IS NULL OR T3.grade = ''")
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get the count of students with a specific salary and enrolled in a specific course
@app.get("/v1/cs_semester/count_students_by_salary_course", operation_id="get_count_students_by_salary_course", summary="Retrieves the number of students who have a specified salary and are enrolled in a particular course. The operation considers the student's salary and the course name to calculate the count.")
async def get_count_students_by_salary_course(salary: str = Query(..., description="Salary level of the student"), course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM RA AS T1 INNER JOIN registration AS T2 ON T2.student_id = T1.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T1.salary = ? AND T3.name = ?", (salary, course_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most popular professor based on the number of students
@app.get("/v1/cs_semester/most_popular_prof", operation_id="get_most_popular_prof", summary="Retrieves the professor with the highest student count, indicating the most popular professor. This operation joins the prof and RA tables to calculate popularity based on the number of associated students. The result includes the first name, last name, and popularity score of the most popular professor.")
async def get_most_popular_prof():
    cursor.execute("SELECT T1.first_name, T1.last_name, T1.popularity FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id GROUP BY T1.prof_id ORDER BY COUNT(T2.student_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"professor": []}
    return {"professor": result}

# Endpoint to get the course name and the first and last names of the student with the highest number of students getting a specific grade
@app.get("/v1/cs_semester/top_course_by_grade", operation_id="get_top_course_by_grade", summary="Retrieves the name of the course and the first and last names of the student who has the highest number of students achieving a specified grade. The grade is provided as an input parameter.")
async def get_top_course_by_grade(grade: str = Query(..., description="Grade of the student")):
    cursor.execute("SELECT T3.name, T2.f_name, T2.l_name FROM registration AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T1.course_id = T3.course_id WHERE T1.grade = ? GROUP BY T3.name ORDER BY COUNT(T1.student_id) DESC LIMIT 1", (grade,))
    result = cursor.fetchone()
    if not result:
        return {"course": []}
    return {"course": result}

# Endpoint to get the difference in average SAT scores between students with a specific salary and those with another salary
@app.get("/v1/cs_semester/sat_diff_by_salary", operation_id="get_sat_diff_by_salary", summary="Retrieves the difference in average SAT scores between two groups of students, categorized by their respective salary levels.")
async def get_sat_diff_by_salary(salary1: str = Query(..., description="First salary level of the student"), salary2: str = Query(..., description="Second salary level of the student")):
    cursor.execute("SELECT AVG(T2.sat) - ( SELECT AVG(T2.sat) FROM RA AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id WHERE T1.salary = ? ) AS diff FROM RA AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id WHERE T1.salary = ?", (salary1, salary2))
    result = cursor.fetchone()
    if not result:
        return {"sat_diff": []}
    return {"sat_diff": result[0]}

# Endpoint to get the graduate institution of the professor with the most students of a given type
@app.get("/v1/cs_semester/professor_graduate_institution", operation_id="get_professor_graduate_institution", summary="Retrieves the graduate institution of the professor who has the highest number of students of a specified type. The student type is provided as an input parameter.")
async def get_professor_graduate_institution(student_type: str = Query(..., description="Type of student (e.g., 'UG' for undergraduate)")):
    cursor.execute("SELECT T1.graduate_from FROM prof AS T1 INNER JOIN RA AS T2 ON T1.prof_id = T2.prof_id INNER JOIN student AS T3 ON T2.student_id = T3.student_id WHERE T3.type = ? GROUP BY T1.prof_id ORDER BY COUNT(T2.student_id) DESC LIMIT 1", (student_type,))
    result = cursor.fetchone()
    if not result:
        return {"graduate_from": []}
    return {"graduate_from": result[0]}

# Endpoint to get professors with above-average teaching ability and at least a specified number of students
@app.get("/v1/cs_semester/professors_above_avg_teaching", operation_id="get_professors_above_avg_teaching", summary="Retrieves a list of professors who have a teaching ability score higher than the average and are associated with at least a specified minimum number of students. The response includes the first name, last name, and email of each professor.")
async def get_professors_above_avg_teaching(min_students: int = Query(..., description="Minimum number of students")):
    cursor.execute("SELECT T2.first_name, T2.last_name, T2.email FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id WHERE T2.teachingability > ( SELECT AVG(teachingability) FROM prof ) GROUP BY T2.prof_id HAVING COUNT(T1.student_id) >= ?", (min_students,))
    result = cursor.fetchall()
    if not result:
        return {"professors": []}
    return {"professors": result}

# Endpoint to get the percentage of students in a course with the highest SAT score
@app.get("/v1/cs_semester/percentage_students_highest_sat", operation_id="get_percentage_students_highest_sat", summary="Retrieves the percentage of students in a specified course who have achieved the highest SAT score. The calculation is based on the total number of students registered in the course.")
async def get_percentage_students_highest_sat(course_name: str = Query(..., description="Name of the course")):
    cursor.execute("SELECT CAST(( SELECT COUNT(*) FROM course WHERE name = ? AND course_id IN ( SELECT course_id FROM registration WHERE sat = ( SELECT MAX(sat) FROM registration ) ) ) AS REAL) * 100  / COUNT(T1.student_id) FROM registration AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T2.name = ?", (course_name, course_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of students with a specific salary and the highest teaching ability professor
@app.get("/v1/cs_semester/student_count_highest_teaching_ability", operation_id="get_student_count_highest_teaching_ability", summary="Retrieves the number of students who have a specific salary level and are taught by the professor with the highest teaching ability. The salary level is provided as an input parameter.")
async def get_student_count_highest_teaching_ability(salary: str = Query(..., description="Salary level (e.g., 'high')")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM RA AS T1 INNER JOIN prof AS T2 ON T1.prof_id = T2.prof_id WHERE T1.salary = ? ORDER BY T2.teachingability DESC LIMIT 1", (salary,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the salary of a student based on their email
@app.get("/v1/cs_semester/student_salary_by_email", operation_id="get_student_salary_by_email", summary="Get the salary of a student based on their email")
async def get_student_salary_by_email(email: str = Query(..., description="Email of the student")):
    cursor.execute("SELECT T1.salary FROM RA AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T2.email = ?", (email,))
    result = cursor.fetchone()
    if not result:
        return {"salary": []}
    return {"salary": result[0]}

# Endpoint to get the count of students in a specific course with specific SAT and GPA
@app.get("/v1/cs_semester/student_count_course_sat_gpa", operation_id="get_student_count_course_sat_gpa", summary="Retrieves the number of students enrolled in a specific course who have a particular SAT score and GPA. The course is identified by its name, and the SAT score and GPA are provided as input parameters.")
async def get_student_count_course_sat_gpa(course_name: str = Query(..., description="Name of the course"), sat: int = Query(..., description="SAT score"), gpa: float = Query(..., description="GPA")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ? AND T2.sat = ? AND T1.gpa = ?", (course_name, sat, gpa))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of students in courses with specific difficulty and intelligence level
@app.get("/v1/cs_semester/student_count_difficulty_intelligence", operation_id="get_student_count_difficulty_intelligence", summary="Retrieves the number of students enrolled in courses of a specified difficulty level who have a particular intelligence level. This operation provides insights into the distribution of students based on their intelligence and the difficulty of the courses they are enrolled in.")
async def get_student_count_difficulty_intelligence(difficulty: int = Query(..., description="Difficulty level of the course"), intelligence: int = Query(..., description="Intelligence level of the student")):
    cursor.execute("SELECT COUNT(T1.student_id) FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.diff = ? AND T1.intelligence = ?", (difficulty, intelligence))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of students in a specific course with a specific grade
@app.get("/v1/cs_semester/student_names_course_grade", operation_id="get_student_names_course_grade", summary="Get the names of students in a specific course with a specific grade")
async def get_student_names_course_grade(course_name: str = Query(..., description="Name of the course"), grade: str = Query(..., description="Grade (e.g., 'C')")):
    cursor.execute("SELECT T1.f_name, T1.l_name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.name = ? AND T2.grade = ?", (course_name, grade))
    result = cursor.fetchall()
    if not result:
        return {"students": []}
    return {"students": result}

# Endpoint to get the names of courses taken by students with a specific SAT score and type
@app.get("/v1/cs_semester/course_names_sat_student_type", operation_id="get_course_names_sat_student_type", summary="Retrieves the names of courses that students of a specific type, with a given SAT score, have enrolled in. The operation filters courses based on the provided SAT score and student type, and returns the corresponding course names.")
async def get_course_names_sat_student_type(sat: int = Query(..., description="SAT score"), student_type: str = Query(..., description="Type of student (e.g., 'RPG')")):
    cursor.execute("SELECT T3.name FROM student AS T1 INNER JOIN registration AS T2 ON T1.student_id = T2.student_id INNER JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T2.sat = ? AND T1.type = ?", (sat, student_type))
    result = cursor.fetchall()
    if not result:
        return {"courses": []}
    return {"courses": result}

# Endpoint to get the difference in the count of UG and RPG students with a capability below a specified value
@app.get("/v1/cs_semester/difference_ug_rpg_capability", operation_id="get_difference_ug_rpg_capability", summary="Retrieve the difference in the number of undergraduate (UG) and research postgraduate (RPG) students who have a capability score below a specified threshold. The capability threshold is provided as an input parameter.")
async def get_difference_ug_rpg_capability(capability: int = Query(..., description="Capability threshold")):
    cursor.execute("SELECT SUM(CASE WHEN T2.type = 'UG' THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.type = 'RPG' THEN 1 ELSE 0 END) FROM RA AS T1 INNER JOIN student AS T2 ON T1.student_id = T2.student_id WHERE T1.capability < ?", (capability,))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

api_calls = [
    "/v1/cs_semester/course_name_by_difficulty?name1=Intro%20to%20BlockChain&name2=Computer%20Network",
    "/v1/cs_semester/course_names_by_credit?course_name=Machine%20Learning%20Theory",
    "/v1/cs_semester/prof_count_by_popularity?first_name=Zhihua&last_name=Zhou",
    "/v1/cs_semester/student_phone_number?l_name=Pryor&f_name=Kerry",
    "/v1/cs_semester/prof_names_by_student?f_name=Faina&l_name=Mallinar",
    "/v1/cs_semester/student_count_by_prof?first_name=Sauveur&last_name=Skyme",
    "/v1/cs_semester/student_names_by_capability?capability=5",
    "/v1/cs_semester/prof_count_by_salary_and_name?first_name=Ogdon&salary=med&last_name=Zywicki",
    "/v1/cs_semester/student_names_by_course?course_name=Machine%20Learning%20Theory",
    "/v1/cs_semester/student_count_by_grade_gpa_course?grade=B&gpa=3&course_name=Machine%20Learning%20Theory",
    "/v1/cs_semester/course_names_by_student?f_name=Laughton&l_name=Antonio",
    "/v1/cs_semester/students_no_grade_in_course?course_name=Intro%20to%20Database%202",
    "/v1/cs_semester/top_student_by_sat_in_course?f_name1=Laughton&f_name2=Willie&l_name1=Antonio&l_name2=Rechert&course_name=Machine%20Learning%20Theory",
    "/v1/cs_semester/prof_count_by_ra_gender?gender=Male&min_ra_count=3",
    "/v1/cs_semester/student_count_by_type_in_course?course_name=Machine%20Learning%20Theory&student_type=UG",
    "/v1/cs_semester/average_gpa_by_professor?first_name=Ogdon&last_name=Zywicki",
    "/v1/cs_semester/average_sat_by_course?course_name=Machine%20Learning%20Theory",
    "/v1/cs_semester/student_count_by_type?student_type=RPG",
    "/v1/cs_semester/students_with_highest_gpa",
    "/v1/cs_semester/count_students_grade_credit_difficulty?grade=A&credit=3&diff=1",
    "/v1/cs_semester/count_students_difficulty?diff=5",
    "/v1/cs_semester/professor_popularity_student_name?f_name=Harrietta&l_name=Lydford",
    "/v1/cs_semester/count_students_teachingability_gender?teachingability=1&gender=Female",
    "/v1/cs_semester/top_professor_teachingability?min_students=2",
    "/v1/cs_semester/grade_result_student_course?f_name=Rik&l_name=Unsworth&course_name=Computer%20Network",
    "/v1/cs_semester/count_courses_student?f_name=Alvera&l_name=McQuillin",
    "/v1/cs_semester/student_details_professor_name_type?prof_first_name=Zhihua&student_type=RPG&prof_last_name=Zhou",
    "/v1/cs_semester/count_students_course?course_name=Statistical%20learning",
    "/v1/cs_semester/students_no_grade_course?course_name=Applied%20Deep%20Learning",
    "/v1/cs_semester/students_phone_numbers_course_grade?course_name=Intro%20to%20BlockChain&grade=A",
    "/v1/cs_semester/percentage_students_type_professor?student_type=TPG&first_name=Ogdon&last_name=Zywicki",
    "/v1/cs_semester/percentage_students_grade_course?grade=B&course_name=Computer%20Network",
    "/v1/cs_semester/count_courses_difficulty?difficulty=5",
    "/v1/cs_semester/professors_graduated_from_universities?university1=Brown%20University&university2=Columbia%20University&university3=Cornell%20University&university4=Dartmouth%20College&university5=Harvard%20University&university6=Princeton%20University&university7=University%20of%20Pennsylvania&university8=Yale%20University",
    "/v1/cs_semester/courses_max_credit_difficulty",
    "/v1/cs_semester/count_students_type_max_intelligence?student_type=UG",
    "/v1/cs_semester/count_professors_gender_max_popularity?gender=Female",
    "/v1/cs_semester/count_students_grade_course?grade=A&course_name=Applied%20Deep%20Learning",
    "/v1/cs_semester/ra_student_gpa?salary=free",
    "/v1/cs_semester/top_course_by_grade_diff?grade=A&diff=1",
    "/v1/cs_semester/course_count_top_gpa",
    "/v1/cs_semester/distinct_courses_by_sat?sat=5",
    "/v1/cs_semester/courses_by_sat_intelligence?sat=1&intelligence=1",
    "/v1/cs_semester/top_course_by_grade_names?grade=A&course_name1=Advanced%20Operating%20System&course_name2=Intro%20to%20BlockChain",
    "/v1/cs_semester/top_prof_popularity",
    "/v1/cs_semester/avg_students_per_course_diff?diff=4",
    "/v1/cs_semester/student_count_no_grade_gpa_range?min_gpa=3&max_gpa=4",
    "/v1/cs_semester/student_count_by_grade_and_type?grade=A&student_type=UG",
    "/v1/cs_semester/avg_gpa_by_salary_and_capability?salary=high&capability=5",
    "/v1/cs_semester/ra_with_min_capability",
    "/v1/cs_semester/prof_by_graduate_university?graduate_from=University%20of%20Boston",
    "/v1/cs_semester/registrations_with_no_grade",
    "/v1/cs_semester/male_to_female_prof_ratio",
    "/v1/cs_semester/courses_with_min_difficulty",
    "/v1/cs_semester/most_popular_prof_ra_details",
    "/v1/cs_semester/ra_details_by_salary?salary=free",
    "/v1/cs_semester/ra_details_by_professor?first_name=Merwyn&last_name=Conkay",
    "/v1/cs_semester/student_grades_by_course?course_name=Intro%20to%20BlockChain",
    "/v1/cs_semester/top_student_by_grade?grade=A",
    "/v1/cs_semester/professor_details_by_student?student_f_name=Olia&student_l_name=Rabier",
    "/v1/cs_semester/top_student_by_course?course_name=Advanced%20Database%20Systems",
    "/v1/cs_semester/student_average_gpa?student_f_name=Laughton&student_l_name=Antonio",
    "/v1/cs_semester/distinct_student_names?student_type=UG&min_gpa=3.7",
    "/v1/cs_semester/student_capabilities_by_professor?graduate_from=University%20of%20Washington",
    "/v1/cs_semester/student_details_by_ra?salary=high",
    "/v1/cs_semester/top_course_by_professor_gender?gender=Female",
    "/v1/cs_semester/professor_count_by_gender?gender=Female",
    "/v1/cs_semester/course_with_highest_difficulty",
    "/v1/cs_semester/student_count_gpa_range_type?min_gpa=3.1&max_gpa=3.7&student_type=UG",
    "/v1/cs_semester/course_credit_by_name?course_name=Computer%20Vision",
    "/v1/cs_semester/student_ids_course_gpa?course_name=C%20for%20Programmers&gpa=2.5",
    "/v1/cs_semester/top_sat_student_last_name?course_name=Intro%20to%20Database%202",
    "/v1/cs_semester/ra_count_salary_gpa?salary=high&min_gpa=3",
    "/v1/cs_semester/top_sat_course_name?student_type=UG",
    "/v1/cs_semester/ra_capabilities_type_intelligence?student_type=RPG&min_intelligence=4",
    "/v1/cs_semester/student_count_grade_intelligence?grade=B&intelligence=3",
    "/v1/cs_semester/course_difficulty_grade_intelligence?grade=A&intelligence=5",
    "/v1/cs_semester/count_students_by_capability?capability=5",
    "/v1/cs_semester/course_names_by_grade?grade=D",
    "/v1/cs_semester/student_capability_by_name?f_name=Alvera&l_name=McQuillin",
    "/v1/cs_semester/count_students_by_credit_gpa?credit=3&gpa=3.2",
    "/v1/cs_semester/count_ra_students_by_gpa_salary?gpa=3.5&salary=low",
    "/v1/cs_semester/student_emails_by_grade_difficulty?grade=B",
    "/v1/cs_semester/percentage_ra_salary_by_teachingability?salary=low&teachingability=3",
    "/v1/cs_semester/average_teachingability_most_popular_prof",
    "/v1/cs_semester/average_sat_by_grade?grade=B",
    "/v1/cs_semester/student_details_by_gpa_intelligence?gpa=3&intelligence=4",
    "/v1/cs_semester/students_above_avg_capability",
    "/v1/cs_semester/students_course_by_intelligence_gpa?intelligence=5&gpa=3",
    "/v1/cs_semester/avg_capability_by_gpa?gpa=2.5",
    "/v1/cs_semester/profs_by_student_intelligence?intelligence=1",
    "/v1/cs_semester/avg_gpa_by_course_difficulty?diff1=2&diff2=1",
    "/v1/cs_semester/students_capability_no_grade",
    "/v1/cs_semester/count_students_by_salary_course?salary=high&course_name=Computer%20Vision",
    "/v1/cs_semester/most_popular_prof",
    "/v1/cs_semester/top_course_by_grade?grade=A",
    "/v1/cs_semester/sat_diff_by_salary?salary1=free&salary2=high",
    "/v1/cs_semester/professor_graduate_institution?student_type=UG",
    "/v1/cs_semester/professors_above_avg_teaching?min_students=2",
    "/v1/cs_semester/percentage_students_highest_sat?course_name=Intro%20to%20Database%202",
    "/v1/cs_semester/student_count_highest_teaching_ability?salary=high",
    "/v1/cs_semester/student_salary_by_email?email=grosellg@hku.hk",
    "/v1/cs_semester/student_count_course_sat_gpa?course_name=Statistical%20learning&sat=4&gpa=3.8",
    "/v1/cs_semester/student_count_difficulty_intelligence?difficulty=3&intelligence=2",
    "/v1/cs_semester/student_names_course_grade?course_name=Applied%20Deep%20Learning&grade=C",
    "/v1/cs_semester/course_names_sat_student_type?sat=1&student_type=RPG",
    "/v1/cs_semester/difference_ug_rpg_capability?capability=3"
]
