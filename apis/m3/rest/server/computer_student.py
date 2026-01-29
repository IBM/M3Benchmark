from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/computer_student/computer_student.sqlite')
cursor = conn.cursor()

# Endpoint to get the average number of advisors per student
@app.get("/v1/computer_student/average_advisors_per_student", operation_id="get_average_advisors_per_student", summary="Retrieves the average number of advisors assigned to each student in the computer science department. This operation calculates the ratio of total advisor assignments to the number of unique students, providing a statistical overview of advisor distribution.")
async def get_average_advisors_per_student():
    cursor.execute("SELECT CAST(COUNT(p_id) AS REAL) / COUNT(DISTINCT p_id_dummy) AS avgnum FROM advisedBy GROUP BY p_id_dummy")
    result = cursor.fetchall()
    if not result:
        return {"average": []}
    return {"average": [row[0] for row in result]}

# Endpoint to get the count of distinct professors teaching a specific course
@app.get("/v1/computer_student/count_professors_by_course", operation_id="get_count_professors_by_course", summary="Retrieves the number of unique professors teaching a specific course. The course is identified by its unique course_id.")
async def get_count_professors_by_course(course_id: int = Query(..., description="ID of the course")):
    cursor.execute("SELECT COUNT(DISTINCT p_id) FROM taughtBy WHERE course_id = ?", (course_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get course IDs based on course level
@app.get("/v1/computer_student/course_ids_by_level", operation_id="get_course_ids_by_level", summary="Retrieves a list of course IDs that correspond to the specified course level. The course level is a parameter that determines the difficulty or progression of the course.")
async def get_course_ids_by_level(course_level: str = Query(..., description="Level of the course")):
    cursor.execute("SELECT course_id FROM course WHERE courseLevel = ?", (course_level,))
    result = cursor.fetchall()
    if not result:
        return {"course_ids": []}
    return {"course_ids": [row[0] for row in result]}

# Endpoint to get the count of courses based on course level
@app.get("/v1/computer_student/count_courses_by_level", operation_id="get_count_courses_by_level", summary="Retrieves the total number of courses available at a specified level. The level is provided as an input parameter, allowing the user to filter the count based on the desired course level.")
async def get_count_courses_by_level(course_level: str = Query(..., description="Level of the course")):
    cursor.execute("SELECT COUNT(course_id) FROM course WHERE courseLevel = ?", (course_level,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get professor IDs and course IDs based on professor status and position
@app.get("/v1/computer_student/professor_course_ids", operation_id="get_professor_course_ids", summary="Retrieves the IDs of professors and their corresponding course IDs based on their status as a professor and whether they hold a position. The professor status is a binary value indicating whether the person is a professor or not, while the position status indicates if the professor holds a position or not.")
async def get_professor_course_ids(professor: int = Query(..., description="Professor status (1 for professor, 0 for not)"), has_position: int = Query(..., description="Position status (0 for no position, non-zero for having a position)")):
    cursor.execute("SELECT T2.p_id, T2.course_id FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id WHERE T1.professor = ? AND T1.hasPosition <> ?", (professor, has_position))
    result = cursor.fetchall()
    if not result:
        return {"professor_course_ids": []}
    return {"professor_course_ids": [{"p_id": row[0], "course_id": row[1]} for row in result]}

# Endpoint to get professor IDs based on course level
@app.get("/v1/computer_student/professor_ids_by_course_level", operation_id="get_professor_ids_by_course_level", summary="Retrieves the IDs of professors who teach courses at a specified level. The level of the course is provided as an input parameter.")
async def get_professor_ids_by_course_level(course_level: str = Query(..., description="Level of the course")):
    cursor.execute("SELECT T2.p_id FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T1.courseLevel = ?", (course_level,))
    result = cursor.fetchall()
    if not result:
        return {"professor_ids": []}
    return {"professor_ids": [row[0] for row in result]}

# Endpoint to get course IDs based on advisor ID
@app.get("/v1/computer_student/course_ids_by_advisor", operation_id="get_course_ids_by_advisor", summary="Retrieves a list of course IDs associated with a specific advisor. The operation uses the provided advisor ID to look up the corresponding advisor and then identifies the courses they are involved in teaching. The result is a collection of course IDs that can be used for further queries or operations.")
async def get_course_ids_by_advisor(advisor_id: int = Query(..., description="ID of the advisor")):
    cursor.execute("SELECT T3.course_id FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id INNER JOIN taughtBy AS T3 ON T2.p_id = T3.p_id WHERE T1.p_id = ?", (advisor_id,))
    result = cursor.fetchall()
    if not result:
        return {"course_ids": []}
    return {"course_ids": [row[0] for row in result]}

# Endpoint to get student IDs based on years in program
@app.get("/v1/computer_student/student_ids_by_years_in_program", operation_id="get_student_ids_by_years_in_program", summary="Retrieves the IDs of students who have been in the program for the specified number of years. The input parameter 'years_in_program' is used to filter the results. The endpoint returns a list of student IDs that match the given criteria.")
async def get_student_ids_by_years_in_program(years_in_program: str = Query(..., description="Years in program (e.g., 'Year_3')")):
    cursor.execute("SELECT T1.p_id FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.yearsInProgram = ?", (years_in_program,))
    result = cursor.fetchall()
    if not result:
        return {"student_ids": []}
    return {"student_ids": [row[0] for row in result]}

# Endpoint to get course levels based on professor ID
@app.get("/v1/computer_student/course_levels_by_professor", operation_id="get_course_levels_by_professor", summary="Retrieves the levels of courses taught by a specific professor. The operation uses the provided professor ID to filter the courses and returns the corresponding course levels.")
async def get_course_levels_by_professor(professor_id: int = Query(..., description="ID of the professor")):
    cursor.execute("SELECT T1.courseLevel FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T2.p_id = ?", (professor_id,))
    result = cursor.fetchall()
    if not result:
        return {"course_levels": []}
    return {"course_levels": [row[0] for row in result]}

# Endpoint to get course levels and professor IDs based on course ID
@app.get("/v1/computer_student/course_levels_professor_ids_by_course", operation_id="get_course_levels_professor_ids_by_course", summary="Retrieves the course level and the IDs of professors teaching the course, based on the provided course ID. This operation returns a list of course levels and corresponding professor IDs, enabling users to identify the professors associated with a specific course and their respective course levels.")
async def get_course_levels_professor_ids_by_course(course_id: int = Query(..., description="ID of the course")):
    cursor.execute("SELECT T1.courseLevel, T2.p_id FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T2.course_id = ?", (course_id,))
    result = cursor.fetchall()
    if not result:
        return {"course_levels_professor_ids": []}
    return {"course_levels_professor_ids": [{"course_level": row[0], "p_id": row[1]} for row in result]}

# Endpoint to get the p_id and yearsInProgram of advised students based on p_id_dummy
@app.get("/v1/computer_student/advised_students_info", operation_id="get_advised_students_info", summary="Retrieves the unique identifier (p_id) and the number of years in the program (yearsInProgram) for students advised by a specific individual, identified by the provided p_id_dummy.")
async def get_advised_students_info(p_id_dummy: int = Query(..., description="p_id_dummy of the advised student")):
    cursor.execute("SELECT T1.p_id, T2.yearsInProgram FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T1.p_id_dummy = ?", (p_id_dummy,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get course details based on the position of the instructor
@app.get("/v1/computer_student/course_details_by_position", operation_id="get_course_details_by_position", summary="Retrieves course details, including course ID and level, for courses taught by an instructor with a specific position. The position of the instructor is provided as an input parameter.")
async def get_course_details_by_position(hasPosition: str = Query(..., description="Position of the instructor")):
    cursor.execute("SELECT T3.course_id, T3.courseLevel FROM taughtBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id INNER JOIN course AS T3 ON T3.course_id = T1.course_id WHERE T2.hasPosition = ?", (hasPosition,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the p_id_dummy and courseLevel based on p_id
@app.get("/v1/computer_student/advised_course_level", operation_id="get_advised_course_level", summary="Retrieves the dummy identifier and course level for a given student based on their advisor's course. This operation joins the advisedBy, course, and taughtBy tables to find the course level associated with the student's advisor. The student is identified by their unique p_id.")
async def get_advised_course_level(p_id: int = Query(..., description="p_id of the advised student")):
    cursor.execute("SELECT T1.p_id_dummy, T2.courseLevel FROM advisedBy AS T1 INNER JOIN course AS T2 ON T1.p_id = T2.course_id INNER JOIN taughtBy AS T3 ON T2.course_id = T3.course_id WHERE T1.p_id = ?", (p_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct p_id of instructors based on course levels
@app.get("/v1/computer_student/instructors_by_course_level", operation_id="get_instructors_by_course_level", summary="Retrieves the unique identifiers of instructors who teach courses at the specified course levels. The operation accepts two course levels as input parameters and returns a list of distinct instructor identifiers who teach courses at either of the provided levels.")
async def get_instructors_by_course_level(courseLevel1: str = Query(..., description="First course level"), courseLevel2: str = Query(..., description="Second course level")):
    cursor.execute("SELECT DISTINCT T2.p_id FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T1.courseLevel = ? OR T1.courseLevel = ?", (courseLevel1, courseLevel2))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get p_id_dummy of advised students based on years in program
@app.get("/v1/computer_student/advised_students_by_years_in_program", operation_id="get_advised_students_by_years_in_program", summary="Retrieves a list of students advised by a specific advisor, filtered by the number of years they have been in the program. The input parameter 'yearsInProgram' is used to determine the duration of a student's enrollment in the program.")
async def get_advised_students_by_years_in_program(yearsInProgram: str = Query(..., description="Years in program")):
    cursor.execute("SELECT T1.p_id_dummy FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.yearsInProgram = ?", (yearsInProgram,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the most frequently taught course
@app.get("/v1/computer_student/most_frequently_taught_course", operation_id="get_most_frequently_taught_course", summary="Retrieves the course that is most frequently taught, including its level. This operation returns the course with the highest count of occurrences in the taughtBy table, which is determined by joining the course and taughtBy tables on the course_id field. The result is ordered in descending order based on the count of course_id occurrences and limited to the top record.")
async def get_most_frequently_taught_course():
    cursor.execute("SELECT T1.course_id, T1.courseLevel FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_id, T1.courseLevel ORDER BY COUNT(T1.course_id) DESC LIMIT 1")
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of courses based on course level
@app.get("/v1/computer_student/course_count_by_level", operation_id="get_course_count_by_level", summary="Retrieves the total number of courses available at a specified level. The level is provided as an input parameter, allowing the user to filter the count based on their desired course level.")
async def get_course_count_by_level(courseLevel: str = Query(..., description="Course level")):
    cursor.execute("SELECT COUNT(*) FROM course WHERE courseLevel = ?", (courseLevel,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of instructors based on course id
@app.get("/v1/computer_student/instructor_count_by_course", operation_id="get_instructor_count_by_course", summary="Retrieves the total number of instructors associated with a specific course. The course is identified by its unique course_id.")
async def get_instructor_count_by_course(course_id: int = Query(..., description="Course id")):
    cursor.execute("SELECT COUNT(*) FROM taughtBy WHERE course_id = ?", (course_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most frequently taught course between two course ids
@app.get("/v1/computer_student/most_frequently_taught_course_between_two", operation_id="get_most_frequently_taught_course_between_two", summary="Retrieves the course that is most frequently taught between the two provided course IDs. The operation compares the frequency of occurrence of the two courses in the database and returns the one that is taught more often. The input parameters are used to specify the two courses to be compared.")
async def get_most_frequently_taught_course_between_two(course_id1: int = Query(..., description="First course id"), course_id2: int = Query(..., description="Second course id")):
    cursor.execute("SELECT course_id FROM taughtBy WHERE course_id = ? OR course_id = ? GROUP BY course_id ORDER BY COUNT(course_id) DESC LIMIT 1", (course_id1, course_id2))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of persons based on position
@app.get("/v1/computer_student/person_count_by_position", operation_id="get_person_count_by_position", summary="Retrieves the total count of persons who hold a specific position. The position is provided as an input parameter.")
async def get_person_count_by_position(hasPosition: str = Query(..., description="Position of the person")):
    cursor.execute("SELECT COUNT(*) FROM person WHERE hasPosition = ?", (hasPosition,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get p_id_dummy values from advisedBy table where the count of p_id_dummy exceeds a specified threshold
@app.get("/v1/computer_student/advisedBy_p_id_dummy_count", operation_id="get_advisedBy_p_id_dummy_count", summary="Retrieves unique p_id_dummy values from the advisedBy table where the count of each p_id_dummy surpasses a specified threshold. The threshold is provided as an input parameter.")
async def get_advisedBy_p_id_dummy_count(count_threshold: int = Query(..., description="Threshold count for p_id_dummy")):
    cursor.execute("SELECT p_id_dummy FROM advisedBy GROUP BY p_id_dummy HAVING COUNT(p_id_dummy) > ?", (count_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"p_id_dummy": []}
    return {"p_id_dummy": [row[0] for row in result]}

# Endpoint to get the count of courses taught by professors at a specific course level
@app.get("/v1/computer_student/course_count_by_level_and_professor", operation_id="get_course_count_by_level_and_professor", summary="Retrieves the total number of courses taught by professors at a specified course level. The operation requires the course level and a boolean value indicating whether the person is a professor (1 for professor).")
async def get_course_count_by_level_and_professor(course_level: str = Query(..., description="Course level"), professor: int = Query(..., description="Professor status (1 for professor)")):
    cursor.execute("SELECT COUNT(*) FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id INNER JOIN person AS T3 ON T3.p_id = T2.p_id WHERE T1.courseLevel = ? AND T3.professor = ?", (course_level, professor))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get p_id of persons teaching a specific course level and holding a specific position
@app.get("/v1/computer_student/p_id_by_course_level_and_position", operation_id="get_p_id_by_course_level_and_position", summary="Retrieves the unique identifier (p_id) of persons who are teaching a course at a specified level and hold a specific position. The course level and position are provided as input parameters.")
async def get_p_id_by_course_level_and_position(course_level: str = Query(..., description="Course level"), has_position: str = Query(..., description="Position held by the person")):
    cursor.execute("SELECT T2.p_id FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id INNER JOIN person AS T3 ON T3.p_id = T2.p_id WHERE T1.courseLevel = ? AND T3.hasPosition = ?", (course_level, has_position))
    result = cursor.fetchall()
    if not result:
        return {"p_id": []}
    return {"p_id": [row[0] for row in result]}

# Endpoint to get the positions of persons teaching a specific course
@app.get("/v1/computer_student/positions_by_course_id", operation_id="get_positions_by_course_id", summary="Retrieves the positions held by individuals who are teaching a specific course, identified by the provided course ID.")
async def get_positions_by_course_id(course_id: int = Query(..., description="Course ID")):
    cursor.execute("SELECT T2.hasPosition FROM taughtBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T1.course_id = ?", (course_id,))
    result = cursor.fetchall()
    if not result:
        return {"hasPosition": []}
    return {"hasPosition": [row[0] for row in result]}

# Endpoint to get the count of distinct persons advised by professors teaching a specific course level
@app.get("/v1/computer_student/distinct_advised_persons_count", operation_id="get_distinct_advised_persons_count", summary="Retrieve the unique count of students advised by professors teaching a specific course level. This operation requires the professor status and the course level as input parameters.")
async def get_distinct_advised_persons_count(professor: int = Query(..., description="Professor status (1 for professor)"), course_level: str = Query(..., description="Course level")):
    cursor.execute("SELECT COUNT(DISTINCT T4.p_id) FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id INNER JOIN course AS T3 ON T3.course_id = T2.course_id INNER JOIN advisedBy AS T4 ON T4.p_id = T1.p_id WHERE T1.professor = ? AND T3.courseLevel = ?", (professor, course_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of courses taught by professors at a specific course level
@app.get("/v1/computer_student/course_count_by_professor_and_level", operation_id="get_course_count_by_professor_and_level", summary="Retrieves the total number of courses taught by professors at a specified course level. The operation requires the professor status and the desired course level as input parameters.")
async def get_course_count_by_professor_and_level(professor: int = Query(..., description="Professor status (1 for professor)"), course_level: str = Query(..., description="Course level")):
    cursor.execute("SELECT COUNT(*) FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id INNER JOIN person AS T3 ON T2.p_id = T3.p_id WHERE T3.professor = ? AND T1.courseLevel = ?", (professor, course_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get course IDs taught by a specific person who is also an advisor
@app.get("/v1/computer_student/course_ids_by_advisor_p_id", operation_id="get_course_ids_by_advisor_p_id", summary="Retrieves the course IDs of courses taught by a specific person who also serves as an advisor. The person is identified by their unique ID. The response includes a list of course IDs associated with the specified advisor.")
async def get_course_ids_by_advisor_p_id(p_id: int = Query(..., description="Person ID")):
    cursor.execute("SELECT T2.course_id FROM taughtBy AS T1 INNER JOIN course AS T2 ON T1.course_id = T2.course_id INNER JOIN advisedBy AS T3 ON T3.p_id = T1.p_id WHERE T1.p_id = ?", (p_id,))
    result = cursor.fetchall()
    if not result:
        return {"course_id": []}
    return {"course_id": [row[0] for row in result]}

# Endpoint to get the course level with the highest number of instructors
@app.get("/v1/computer_student/course_level_with_most_instructors", operation_id="get_course_level_with_most_instructors", summary="Retrieves the course level that has the most instructors associated with it. This operation identifies the course level with the highest number of instructors by joining the course and taughtBy tables and grouping by course_id. The result is ordered by the count of instructor IDs in descending order and limited to the top record.")
async def get_course_level_with_most_instructors():
    cursor.execute("SELECT T1.courseLevel FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id GROUP BY T2.course_id ORDER BY COUNT(T2.p_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"courseLevel": []}
    return {"courseLevel": result[0]}

# Endpoint to get professors who have taught more than a specified number of distinct courses
@app.get("/v1/computer_student/professors_taught_more_than_n_courses", operation_id="get_professors_taught_more_than_n_courses", summary="Retrieves a list of professors who have taught more than a specified number of distinct courses. The operation filters professors based on a provided indicator and returns those who have taught a minimum number of unique courses, as defined by the input parameters.")
async def get_professors_taught_more_than_n_courses(professor: int = Query(..., description="Professor indicator (1 for professor)"), min_courses: int = Query(..., description="Minimum number of distinct courses taught")):
    cursor.execute("SELECT T1.p_id FROM taughtBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.professor = ? GROUP BY T1.p_id HAVING COUNT(DISTINCT T1.course_id) > ?", (professor, min_courses))
    result = cursor.fetchall()
    if not result:
        return {"p_ids": []}
    return {"p_ids": [row[0] for row in result]}

# Endpoint to get top N professors by the number of courses taught
@app.get("/v1/computer_student/top_n_professors_by_courses_taught", operation_id="get_top_n_professors_by_courses_taught", summary="Retrieves the top N professors, ranked by the number of courses they teach. The operation filters professors based on the provided indicator and returns the specified number of top professors.")
async def get_top_n_professors_by_courses_taught(professor: int = Query(..., description="Professor indicator (1 for professor)"), limit: int = Query(..., description="Number of top professors to return")):
    cursor.execute("SELECT T1.p_id FROM taughtBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.professor = ? GROUP BY T1.p_id ORDER BY COUNT(*) DESC LIMIT ?", (professor, limit))
    result = cursor.fetchall()
    if not result:
        return {"p_ids": []}
    return {"p_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct advisors for a specific year in the program
@app.get("/v1/computer_student/count_distinct_advisors_by_year", operation_id="get_count_distinct_advisors_by_year", summary="Retrieves the number of unique advisors associated with students in a specific year of the program. The year is provided as an input parameter.")
async def get_count_distinct_advisors_by_year(years_in_program: str = Query(..., description="Year in the program (e.g., 'Year_3')")):
    cursor.execute("SELECT COUNT(DISTINCT T1.p_id_dummy) FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.yearsInProgram = ?", (years_in_program,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average number of courses taught by professors
@app.get("/v1/computer_student/average_courses_taught_by_professors", operation_id="get_average_courses_taught_by_professors", summary="Retrieves the average number of courses taught by professors, based on the provided professor indicator. This operation calculates the total number of courses taught by professors and divides it by the distinct count of professor IDs, providing a clear overview of the average course load per professor.")
async def get_average_courses_taught_by_professors(professor: int = Query(..., description="Professor indicator (1 for professor)")):
    cursor.execute("SELECT CAST(COUNT(T1.course_id) AS REAL) / COUNT(DISTINCT T2.p_id) AS num FROM taughtBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.professor = ?", (professor,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of professors relative to students
@app.get("/v1/computer_student/percentage_professors_to_students", operation_id="get_percentage_professors_to_students", summary="Retrieves the percentage of professors relative to students from the person table. The operation uses the provided indicators to identify professors and students, and calculates the percentage based on their respective counts.")
async def get_percentage_professors_to_students(professor: int = Query(..., description="Professor indicator (1 for professor)"), student: int = Query(..., description="Student indicator (1 for student)")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN professor = ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE  WHEN student = ? THEN 1 ELSE 0 END) AS per FROM person", (professor, student))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of courses at a specific level
@app.get("/v1/computer_student/percentage_courses_at_level", operation_id="get_percentage_courses_at_level", summary="Retrieves the percentage of courses at a specified level. The operation calculates the proportion of courses at the given level relative to the total number of courses in the database. The level is provided as an input parameter.")
async def get_percentage_courses_at_level(course_level: str = Query(..., description="Course level (e.g., 'Level_400')")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN courseLevel = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) AS per FROM course", (course_level,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get professors who taught a specific course
@app.get("/v1/computer_student/professors_taught_course", operation_id="get_professors_taught_course", summary="Retrieves the identifiers of professors who have taught a specific course. The course is identified by its unique course_id.")
async def get_professors_taught_course(course_id: int = Query(..., description="Course ID")):
    cursor.execute("SELECT p_id FROM taughtBy WHERE course_id = ?", (course_id,))
    result = cursor.fetchall()
    if not result:
        return {"p_ids": []}
    return {"p_ids": [row[0] for row in result]}

# Endpoint to get the positions of advisors for a specific person
@app.get("/v1/computer_student/advisor_positions_for_person", operation_id="get_advisor_positions_for_person", summary="Retrieves the advisor positions held by a specific person, identified by their unique person ID. The response includes the position titles and the corresponding person IDs.")
async def get_advisor_positions_for_person(p_id: int = Query(..., description="Person ID")):
    cursor.execute("SELECT T2.hasPosition, T1.p_id_dummy FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id_dummy = T2.p_id WHERE T1.p_id = ?", (p_id,))
    result = cursor.fetchall()
    if not result:
        return {"positions": []}
    return {"positions": [{"hasPosition": row[0], "p_id_dummy": row[1]} for row in result]}

# Endpoint to get professors and the levels of courses they taught based on their position
@app.get("/v1/computer_student/professors_and_course_levels", operation_id="get_professors_and_course_levels", summary="Retrieves a list of professors and the corresponding levels of courses they have taught, based on a specified position. The position is used to filter the professors included in the results.")
async def get_professors_and_course_levels(has_position: str = Query(..., description="Position of the professor (e.g., 'Faculty_aff')")):
    cursor.execute("SELECT T1.p_id, T3.courseLevel FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id INNER JOIN course AS T3 ON T3.course_id = T2.course_id WHERE T1.hasPosition = ?", (has_position,))
    result = cursor.fetchall()
    if not result:
        return {"professors_and_levels": []}
    return {"professors_and_levels": [{"p_id": row[0], "courseLevel": row[1]} for row in result]}

# Endpoint to get the top N persons by the number of advisors and their program details
@app.get("/v1/computer_student/top_n_persons_by_advisors", operation_id="get_top_n_persons_by_advisors", summary="Retrieves the top N persons, ranked by the number of advisors they have, along with their respective years in the program and current phase. The 'limit' parameter determines the number of top persons to return.")
async def get_top_n_persons_by_advisors(limit: int = Query(..., description="Number of top persons to return")):
    cursor.execute("SELECT T2.yearsInProgram, T2.inPhase FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id GROUP BY T1.p_id ORDER BY COUNT(*) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"program_details": []}
    return {"program_details": [{"yearsInProgram": row[0], "inPhase": row[1]} for row in result]}

# Endpoint to get the p_id of advisors and advisees based on a specific position
@app.get("/v1/computer_student/advisors_advisees_by_position", operation_id="get_advisors_advisees_by_position", summary="Retrieves the unique identifiers (p_id) of advisors and advisees associated with a specified position. The position is provided as an input parameter.")
async def get_advisors_advisees_by_position(has_position: str = Query(..., description="Position of the advisor")):
    cursor.execute("SELECT T1.p_id, T2.p_id FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id_dummy = T2.p_id WHERE hasPosition = ?", (has_position,))
    result = cursor.fetchall()
    if not result:
        return {"advisors_advisees": []}
    return {"advisors_advisees": result}

# Endpoint to get course details based on a range of p_id
@app.get("/v1/computer_student/course_details_by_pid_range", operation_id="get_course_details_by_pid_range", summary="Retrieves course details for courses taught by professors with p_id values within a specified range. The range is defined by the minimum and maximum p_id values provided as input parameters. The returned data includes the course_id and courseLevel for each course that meets the criteria.")
async def get_course_details_by_pid_range(min_pid: int = Query(..., description="Minimum p_id"), max_pid: int = Query(..., description="Maximum p_id")):
    cursor.execute("SELECT T1.course_id, T1.courseLevel FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T2.p_id BETWEEN ? AND ?", (min_pid, max_pid))
    result = cursor.fetchall()
    if not result:
        return {"course_details": []}
    return {"course_details": result}

# Endpoint to get course details based on a specific p_id
@app.get("/v1/computer_student/course_details_by_pid", operation_id="get_course_details_by_pid", summary="Retrieves the course level and course ID for courses taught by a specific instructor, identified by their unique p_id.")
async def get_course_details_by_pid(p_id: int = Query(..., description="p_id of the instructor")):
    cursor.execute("SELECT T1.courseLevel, T1.course_id FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T2.p_id = ?", (p_id,))
    result = cursor.fetchall()
    if not result:
        return {"course_details": []}
    return {"course_details": result}

# Endpoint to get person details based on course_id and hasPosition
@app.get("/v1/computer_student/person_details_by_course_id_and_position", operation_id="get_person_details_by_course_id_and_position", summary="Retrieves the details of a person associated with a specific course and position. The details include the person's ID and the level of the course. The course is identified by its unique ID, and the person's position is used to filter the results.")
async def get_person_details_by_course_id_and_position(course_id: int = Query(..., description="Course ID"), has_position: int = Query(..., description="Position of the person")):
    cursor.execute("SELECT T1.p_id, T3.courseLevel FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id INNER JOIN course AS T3 ON T3.course_id = T2.course_id WHERE T3.course_id = ? AND T1.hasPosition <> ?", (course_id, has_position))
    result = cursor.fetchall()
    if not result:
        return {"person_details": []}
    return {"person_details": result}

# Endpoint to get person details based on course level and maximum course_id
@app.get("/v1/computer_student/person_details_by_course_level_and_max_course_id", operation_id="get_person_details_by_course_level_and_max_course_id", summary="Retrieves details of individuals who have taught courses at a specified level and with a course ID less than a given maximum value. The response includes the person's ID and their position.")
async def get_person_details_by_course_level_and_max_course_id(course_level: str = Query(..., description="Course level"), max_course_id: int = Query(..., description="Maximum course ID")):
    cursor.execute("SELECT T1.p_id, T1.hasPosition FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id INNER JOIN course AS T3 ON T3.course_id = T2.course_id WHERE T3.courseLevel = ? AND T2.course_id < ?", (course_level, max_course_id))
    result = cursor.fetchall()
    if not result:
        return {"person_details": []}
    return {"person_details": result}

# Endpoint to get p_id of instructors based on course level and course_id range
@app.get("/v1/computer_student/instructors_by_course_level_and_course_id_range", operation_id="get_instructors_by_course_level_and_course_id_range", summary="Retrieves the unique identifiers of instructors who teach courses within a specified level and course ID range. The level and range are defined by the provided parameters, allowing for a targeted search of instructors based on their teaching assignments.")
async def get_instructors_by_course_level_and_course_id_range(course_level: str = Query(..., description="Course level"), min_course_id: int = Query(..., description="Minimum course ID"), max_course_id: int = Query(..., description="Maximum course ID")):
    cursor.execute("SELECT T2.p_id FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T1.courseLevel = ? AND T1.course_id > ? AND T1.course_id < ?", (course_level, min_course_id, max_course_id))
    result = cursor.fetchall()
    if not result:
        return {"instructors": []}
    return {"instructors": result}

# Endpoint to get advisedBy details based on years in program
@app.get("/v1/computer_student/advisedBy_details_by_years_in_program", operation_id="get_advisedBy_details_by_years_in_program", summary="Retrieves details of advisors based on the number of years they have been in the program. The input parameter specifies the number of years in the program.")
async def get_advisedBy_details_by_years_in_program(years_in_program: str = Query(..., description="Years in program")):
    cursor.execute("SELECT T1.p_id_dummy, T2.hasPosition FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.yearsInProgram = ?", (years_in_program,))
    result = cursor.fetchall()
    if not result:
        return {"advisedBy_details": []}
    return {"advisedBy_details": result}

# Endpoint to get course and instructor details based on course level with a limit
@app.get("/v1/computer_student/course_instructor_details_by_course_level_with_limit", operation_id="get_course_instructor_details_by_course_level_with_limit", summary="Retrieves a limited number of course and instructor details based on a specified course level. The course level parameter filters the results, and the limit parameter restricts the number of records returned.")
async def get_course_instructor_details_by_course_level_with_limit(course_level: str = Query(..., description="Course level"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT T1.course_id, T2.p_id FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T1.courseLevel = ? LIMIT ?", (course_level, limit))
    result = cursor.fetchall()
    if not result:
        return {"course_instructor_details": []}
    return {"course_instructor_details": result}

# Endpoint to get the count of advisedBy entries based on p_id_dummy
@app.get("/v1/computer_student/count_advisedBy_by_pid_dummy", operation_id="get_count_advisedBy_by_pid_dummy", summary="Retrieves the total number of student advisor relationships associated with a specific advisor, identified by the provided p_id_dummy.")
async def get_count_advisedBy_by_pid_dummy(p_id_dummy: int = Query(..., description="p_id_dummy of the advisor")):
    cursor.execute("SELECT COUNT(*) FROM advisedBy WHERE p_id_dummy = ?", (p_id_dummy,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons based on position and phase
@app.get("/v1/computer_student/person_count_by_position_phase", operation_id="get_person_count", summary="Retrieves the total number of individuals who hold a certain position and are currently in a specific phase. The position and phase statuses are represented as binary values (0 or 1).")
async def get_person_count(has_position: int = Query(..., description="Position status of the person (0 or 1)"), in_phase: int = Query(..., description="Phase status of the person (0 or 1)")):
    cursor.execute("SELECT COUNT(*) FROM person WHERE hasPosition = ? AND inPhase = ?", (has_position, in_phase))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the p_id of the person who taught the least number of courses
@app.get("/v1/computer_student/least_taught_courses", operation_id="get_least_taught_courses", summary="Retrieves the unique identifier of the person who has taught the fewest courses. The endpoint calculates the count of courses taught by each person and returns the identifier of the person with the lowest count.")
async def get_least_taught_courses():
    cursor.execute("SELECT p_id FROM taughtBy GROUP BY p_id ORDER BY COUNT(course_id) ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"p_id": []}
    return {"p_id": result[0]}

# Endpoint to get the count of advised students based on dummy p_id, student status, and years in program
@app.get("/v1/computer_student/advised_students_count", operation_id="get_advised_students_count", summary="Retrieves the total number of students advised by a specific individual, filtered by student status and years in the program. The response is based on a dummy p_id, student status (0 or 1), and the student's years in the program (e.g., 'Year_5').")
async def get_advised_students_count(p_id_dummy: int = Query(..., description="Dummy p_id"), student: int = Query(..., description="Student status (0 or 1)"), years_in_program: str = Query(..., description="Years in program (e.g., 'Year_5')")):
    cursor.execute("SELECT COUNT(*) FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T1.p_id_dummy = ? AND T2.student = ? AND T2.yearsInProgram = ?", (p_id_dummy, student, years_in_program))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the p_id of the person who taught the most courses at a specific course level
@app.get("/v1/computer_student/most_taught_courses_by_level", operation_id="get_most_taught_courses_by_level", summary="Retrieves the ID of the person who has taught the most courses at a specified course level. The course level is provided as an input parameter, and the result is determined by counting the number of courses taught by each person at that level.")
async def get_most_taught_courses_by_level(course_level: str = Query(..., description="Course level (e.g., 'Level_500')")):
    cursor.execute("SELECT T2.p_id FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T1.courseLevel = ? GROUP BY T2.p_id ORDER BY COUNT(T2.course_id) DESC LIMIT 1", (course_level,))
    result = cursor.fetchone()
    if not result:
        return {"p_id": []}
    return {"p_id": result[0]}

# Endpoint to get the count of professors with a specific position teaching a specific course level
@app.get("/v1/computer_student/professor_count_by_position_course_level", operation_id="get_professor_count", summary="Retrieve the number of professors with a specified position who are teaching courses at a particular level. The operation requires the position of the professor, their status as a professor, and the level of the course they are teaching.")
async def get_professor_count(has_position: str = Query(..., description="Position of the professor (e.g., 'Faculty_aff')"), professor: int = Query(..., description="Professor status (0 or 1)"), course_level: str = Query(..., description="Course level (e.g., 'Level_500')")):
    cursor.execute("SELECT COUNT(*) FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id INNER JOIN course AS T3 ON T3.course_id = T2.course_id WHERE T1.hasPosition = ? AND T1.professor = ? AND T3.courseLevel = ?", (has_position, professor, course_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top 5 p_ids of persons who taught the most courses at a specific course level
@app.get("/v1/computer_student/top_taught_courses_by_level", operation_id="get_top_taught_courses_by_level", summary="Retrieves the top 5 unique identifiers of individuals who have taught the highest number of courses at a specified course level. The course level is provided as an input parameter.")
async def get_top_taught_courses_by_level(course_level: str = Query(..., description="Course level (e.g., 'Level_500')")):
    cursor.execute("SELECT T2.p_id FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T1.courseLevel = ? GROUP BY T2.p_id ORDER BY COUNT(T2.p_id) DESC LIMIT 5", (course_level,))
    result = cursor.fetchall()
    if not result:
        return {"p_ids": []}
    return {"p_ids": [row[0] for row in result]}

# Endpoint to get the count of advised students based on years in program and student status
@app.get("/v1/computer_student/advised_students_count_by_years_in_program", operation_id="get_advised_students_count_by_years_in_program", summary="Retrieves the total number of students advised by a faculty member, filtered by the number of years they have been in the program and their student status. The response is based on the provided years in program and student status.")
async def get_advised_students_count_by_years_in_program(years_in_program: str = Query(..., description="Years in program (e.g., 'Year_1')"), student: int = Query(..., description="Student status (0 or 1)")):
    cursor.execute("SELECT COUNT(T1.p_id_dummy) FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.yearsInProgram = ? AND T2.student = ?", (years_in_program, student))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons who taught a limited number of distinct courses at a specific course level
@app.get("/v1/computer_student/limited_taught_courses_count", operation_id="get_limited_taught_courses_count", summary="Retrieves the count of individuals who have taught a limited number of unique courses at a specified course level. The operation considers the course level and the maximum number of distinct courses taught as input parameters to filter the results.")
async def get_limited_taught_courses_count(course_level: str = Query(..., description="Course level (e.g., 'Level_400')"), max_distinct_courses: int = Query(..., description="Maximum number of distinct courses taught")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT COUNT(T2.p_id) FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T1.courseLevel = ? GROUP BY T2.p_id HAVING COUNT(DISTINCT T1.course_id) <= ? )", (course_level, max_distinct_courses))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the position of the professor who taught the most courses
@app.get("/v1/computer_student/professor_position_most_taught_courses", operation_id="get_professor_position_most_taught_courses", summary="Retrieves the position of the professor who has taught the most courses. The professor's status must be provided to filter the results. The data is grouped by professor ID and ordered by the count of courses taught in descending order, with the top result being returned.")
async def get_professor_position_most_taught_courses(professor: int = Query(..., description="Professor status (0 or 1)")):
    cursor.execute("SELECT T1.hasPosition FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id WHERE T1.professor = ? GROUP BY T1.p_id ORDER BY COUNT(T2.course_id) DESC LIMIT 1", (professor,))
    result = cursor.fetchone()
    if not result:
        return {"hasPosition": []}
    return {"hasPosition": result[0]}

# Endpoint to get years in program for students advised by more than a specified number of advisors
@app.get("/v1/computer_student/years_in_program_advised_by_count", operation_id="get_years_in_program_advised_by_count", summary="Retrieves the years in program for students who are advised by more than a specified number of advisors. The operation filters students based on their status and returns the years in program for those who meet the minimum advisor count criteria.")
async def get_years_in_program_advised_by_count(student: int = Query(..., description="Student status (1 for student)"), count: int = Query(..., description="Minimum number of advisors")):
    cursor.execute("SELECT T2.yearsInProgram FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.student = ? GROUP BY T2.p_id HAVING COUNT(T2.p_id) > ?", (student, count))
    result = cursor.fetchall()
    if not result:
        return {"years_in_program": []}
    return {"years_in_program": [row[0] for row in result]}

# Endpoint to get the most common years in program for students
@app.get("/v1/computer_student/most_common_years_in_program", operation_id="get_most_common_years_in_program", summary="Retrieves the most frequently occurring years in program for students. The operation filters students based on their status and returns the year in program that appears most frequently. The result is sorted in descending order based on the count of students with the same year in program.")
async def get_most_common_years_in_program(student: int = Query(..., description="Student status (1 for student)")):
    cursor.execute("SELECT T2.yearsInProgram FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.student = ? GROUP BY T2.yearsInProgram ORDER BY COUNT(T1.p_id_dummy) DESC LIMIT 1", (student,))
    result = cursor.fetchone()
    if not result:
        return {"years_in_program": []}
    return {"years_in_program": result[0]}

# Endpoint to get the count of students in a specific phase
@app.get("/v1/computer_student/student_count_in_phase", operation_id="get_student_count_in_phase", summary="Retrieves the total number of students who are currently in a specified phase. The phase is identified by the 'in_phase' parameter, and the student status is confirmed by the 'student' parameter. This operation is useful for tracking the progression of students through different phases.")
async def get_student_count_in_phase(in_phase: str = Query(..., description="Phase of the student (e.g., 'Pre_Quals')"), student: int = Query(..., description="Student status (1 for student)")):
    cursor.execute("SELECT COUNT(T1.p_id_dummy) FROM advisedBy AS T1 INNER JOIN person AS T2 ON T1.p_id = T2.p_id WHERE T2.inPhase = ? AND T2.student = ?", (in_phase, student))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average number of courses per person at a specific level
@app.get("/v1/computer_student/average_courses_per_person", operation_id="get_average_courses_per_person", summary="Retrieves the average number of courses per person for a given course level. This operation calculates the ratio of the total number of courses to the distinct number of people teaching those courses at the specified level.")
async def get_average_courses_per_person(course_level: str = Query(..., description="Course level (e.g., 'Level_500')")):
    cursor.execute("SELECT CAST(COUNT(T1.course_id) AS REAL) / COUNT(DISTINCT T2.p_id) FROM course AS T1 INNER JOIN taughtBy AS T2 ON T1.course_id = T2.course_id WHERE T1.courseLevel = ?", (course_level,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the count of courses taught by more than a specified number of instructors
@app.get("/v1/computer_student/course_count_taught_by_more_than", operation_id="get_course_count_taught_by_more_than", summary="Retrieves the total number of courses that are taught by more than the specified number of instructors. The count parameter determines the minimum number of instructors required for a course to be included in the result.")
async def get_course_count_taught_by_more_than(count: int = Query(..., description="Minimum number of instructors")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT COUNT(course_id) FROM taughtBy GROUP BY course_id HAVING COUNT(course_id) > ? )", (count,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of persons in specific years of the program
@app.get("/v1/computer_student/person_count_by_years_in_program", operation_id="get_person_count_by_years_in_program", summary="Retrieves the total count of persons who have been in the program for either the first or second specified year. The input parameters define the specific years in the program to consider for the count.")
async def get_person_count_by_years_in_program(year1: str = Query(..., description="First year in program (e.g., 'Year_1')"), year2: str = Query(..., description="Second year in program (e.g., 'Year_2')")):
    cursor.execute("SELECT COUNT(*) FROM person WHERE yearsInProgram = ? OR yearsInProgram = ?", (year1, year2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of professors with a position
@app.get("/v1/computer_student/professor_count_with_position", operation_id="get_professor_count_with_position", summary="Retrieves the total number of professors who have a position. The operation filters professors based on their status and position availability, returning a count of those who meet the specified criteria.")
async def get_professor_count_with_position(professor: int = Query(..., description="Professor status (1 for professor)"), has_position: int = Query(..., description="Position status (0 for no position)")):
    cursor.execute("SELECT COUNT(*) FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id WHERE T1.professor = ? AND T1.hasPosition <> ?", (professor, has_position))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the professor with the most courses taught
@app.get("/v1/computer_student/professor_with_most_courses", operation_id="get_professor_with_most_courses", summary="Retrieves the professor who has taught the highest number of courses. The response includes the professor's unique identifier and their position.")
async def get_professor_with_most_courses():
    cursor.execute("SELECT T1.p_id, T1.hasPosition FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id GROUP BY T1.p_id ORDER BY COUNT(T2.course_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"professor": []}
    return {"professor": {"p_id": result[0], "hasPosition": result[1]}}

# Endpoint to get distinct course IDs taught by professors without a position
@app.get("/v1/computer_student/distinct_course_ids_taught_by_professors", operation_id="get_distinct_course_ids", summary="Retrieves a unique set of course IDs taught by professors who do not hold a position. The operation filters professors based on the provided parameters and returns the distinct course IDs associated with them.")
async def get_distinct_course_ids(professor: int = Query(..., description="Indicator if the person is a professor (1 for yes, 0 for no)"), has_position: int = Query(..., description="Indicator if the person holds a position (1 for yes, 0 for no)")):
    cursor.execute("SELECT DISTINCT T2.course_id FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id WHERE T1.professor = ? AND T1.hasPosition = ?", (professor, has_position))
    result = cursor.fetchall()
    if not result:
        return {"course_ids": []}
    return {"course_ids": [row[0] for row in result]}

# Endpoint to get the person ID and course level of the person who has taught the most courses
@app.get("/v1/computer_student/top_teacher_by_course_count", operation_id="get_top_teacher_by_course_count", summary="Retrieves the unique identifier and course level of the individual who has instructed the highest number of courses. This operation considers all individuals and their respective courses, grouping them by the individual's identifier and ordering the results by the count of courses in descending order. The top result is returned.")
async def get_top_teacher_by_course_count():
    cursor.execute("SELECT T1.p_id, T3.courseLevel FROM person AS T1 INNER JOIN taughtBy AS T2 ON T1.p_id = T2.p_id INNER JOIN course AS T3 ON T3.course_id = T2.course_id GROUP BY T1.p_id ORDER BY COUNT(T2.course_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"p_id": None, "course_level": None}
    return {"p_id": result[0], "course_level": result[1]}

api_calls = [
    "/v1/computer_student/average_advisors_per_student",
    "/v1/computer_student/count_professors_by_course?course_id=18",
    "/v1/computer_student/course_ids_by_level?course_level=Level_500",
    "/v1/computer_student/count_courses_by_level?course_level=Level_300",
    "/v1/computer_student/professor_course_ids?professor=1&has_position=0",
    "/v1/computer_student/professor_ids_by_course_level?course_level=Level_400",
    "/v1/computer_student/course_ids_by_advisor?advisor_id=141",
    "/v1/computer_student/student_ids_by_years_in_program?years_in_program=Year_3",
    "/v1/computer_student/course_levels_by_professor?professor_id=297",
    "/v1/computer_student/course_levels_professor_ids_by_course?course_id=165",
    "/v1/computer_student/advised_students_info?p_id_dummy=5",
    "/v1/computer_student/course_details_by_position?hasPosition=Faculty_eme",
    "/v1/computer_student/advised_course_level?p_id=80",
    "/v1/computer_student/instructors_by_course_level?courseLevel1=Level_400&courseLevel2=Level_500",
    "/v1/computer_student/advised_students_by_years_in_program?yearsInProgram=Year_12",
    "/v1/computer_student/most_frequently_taught_course",
    "/v1/computer_student/course_count_by_level?courseLevel=Level_300",
    "/v1/computer_student/instructor_count_by_course?course_id=11",
    "/v1/computer_student/most_frequently_taught_course_between_two?course_id1=11&course_id2=18",
    "/v1/computer_student/person_count_by_position?hasPosition=Faculty_eme",
    "/v1/computer_student/advisedBy_p_id_dummy_count?count_threshold=4",
    "/v1/computer_student/course_count_by_level_and_professor?course_level=Level_300&professor=1",
    "/v1/computer_student/p_id_by_course_level_and_position?course_level=Level_300&has_position=Faculty_eme",
    "/v1/computer_student/positions_by_course_id?course_id=9",
    "/v1/computer_student/distinct_advised_persons_count?professor=1&course_level=Level_300",
    "/v1/computer_student/course_count_by_professor_and_level?professor=1&course_level=Level_300",
    "/v1/computer_student/course_ids_by_advisor_p_id?p_id=9",
    "/v1/computer_student/course_level_with_most_instructors",
    "/v1/computer_student/professors_taught_more_than_n_courses?professor=1&min_courses=3",
    "/v1/computer_student/top_n_professors_by_courses_taught?professor=1&limit=3",
    "/v1/computer_student/count_distinct_advisors_by_year?years_in_program=Year_3",
    "/v1/computer_student/average_courses_taught_by_professors?professor=1",
    "/v1/computer_student/percentage_professors_to_students?professor=1&student=1",
    "/v1/computer_student/percentage_courses_at_level?course_level=Level_400",
    "/v1/computer_student/professors_taught_course?course_id=18",
    "/v1/computer_student/advisor_positions_for_person?p_id=303",
    "/v1/computer_student/professors_and_course_levels?has_position=Faculty_aff",
    "/v1/computer_student/top_n_persons_by_advisors?limit=1",
    "/v1/computer_student/advisors_advisees_by_position?has_position=Faculty_eme",
    "/v1/computer_student/course_details_by_pid_range?min_pid=40&max_pid=50",
    "/v1/computer_student/course_details_by_pid?p_id=141",
    "/v1/computer_student/person_details_by_course_id_and_position?course_id=104&has_position=0",
    "/v1/computer_student/person_details_by_course_level_and_max_course_id?course_level=Level_400&max_course_id=10",
    "/v1/computer_student/instructors_by_course_level_and_course_id_range?course_level=Level_300&min_course_id=121&max_course_id=130",
    "/v1/computer_student/advisedBy_details_by_years_in_program?years_in_program=Year_8",
    "/v1/computer_student/course_instructor_details_by_course_level_with_limit?course_level=Level_500&limit=5",
    "/v1/computer_student/count_advisedBy_by_pid_dummy?p_id_dummy=415",
    "/v1/computer_student/person_count_by_position_phase?has_position=0&in_phase=0",
    "/v1/computer_student/least_taught_courses",
    "/v1/computer_student/advised_students_count?p_id_dummy=5&student=1&years_in_program=Year_5",
    "/v1/computer_student/most_taught_courses_by_level?course_level=Level_500",
    "/v1/computer_student/professor_count_by_position_course_level?has_position=Faculty_aff&professor=1&course_level=Level_500",
    "/v1/computer_student/top_taught_courses_by_level?course_level=Level_500",
    "/v1/computer_student/advised_students_count_by_years_in_program?years_in_program=Year_1&student=1",
    "/v1/computer_student/limited_taught_courses_count?course_level=Level_400&max_distinct_courses=2",
    "/v1/computer_student/professor_position_most_taught_courses?professor=1",
    "/v1/computer_student/years_in_program_advised_by_count?student=1&count=2",
    "/v1/computer_student/most_common_years_in_program?student=1",
    "/v1/computer_student/student_count_in_phase?in_phase=Pre_Quals&student=1",
    "/v1/computer_student/average_courses_per_person?course_level=Level_500",
    "/v1/computer_student/course_count_taught_by_more_than?count=4",
    "/v1/computer_student/person_count_by_years_in_program?year1=Year_1&year2=Year_2",
    "/v1/computer_student/professor_count_with_position?professor=1&has_position=0",
    "/v1/computer_student/professor_with_most_courses",
    "/v1/computer_student/distinct_course_ids_taught_by_professors?professor=1&has_position=0",
    "/v1/computer_student/top_teacher_by_course_count"
]
