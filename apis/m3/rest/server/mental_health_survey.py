from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/mental_health_survey/mental_health_survey.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of answers for a specific question ID and answer text
@app.get("/v1/mental_health_survey/answer_count_by_question_and_text", operation_id="get_answer_count", summary="Retrieves the total number of responses for a given question and answer text. The question is identified by its unique ID, and the answer text is matched using a case-insensitive partial search. This operation is useful for analyzing the distribution of responses to a specific question in a mental health survey.")
async def get_answer_count(question_id: int = Query(..., description="ID of the question"), answer_text: str = Query(..., description="Text of the answer")):
    cursor.execute("SELECT COUNT(QuestionID) FROM Answer WHERE QuestionID = ? AND AnswerText LIKE ?", (question_id, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users who answered a specific question in a range of survey years
@app.get("/v1/mental_health_survey/user_count_by_question_and_survey_years", operation_id="get_user_count_by_survey_years", summary="Retrieves the number of users who responded to a specific question within a specified range of survey years. The response includes a count of users for each survey year in the range.")
async def get_user_count_by_survey_years(question_id: int = Query(..., description="ID of the question"), start_year: int = Query(..., description="Start year of the survey range"), end_year: int = Query(..., description="End year of the survey range")):
    cursor.execute("SELECT SurveyID, COUNT(UserID) FROM Answer WHERE QuestionID = ? AND SurveyID BETWEEN ? AND ? GROUP BY SurveyID", (question_id, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": result}

# Endpoint to get user IDs based on survey year, question ID, and answer text range
@app.get("/v1/mental_health_survey/user_ids_by_survey_question_and_answer_range", operation_id="get_user_ids_by_survey_question_and_answer_range", summary="Retrieves a list of user IDs who have answered a specific survey question within a given range of answer text. The operation filters users based on the survey year, question ID, and the range of answer text provided.")
async def get_user_ids_by_survey_question_and_answer_range(survey_id: int = Query(..., description="Survey year"), question_id: int = Query(..., description="ID of the question"), min_answer_text: str = Query(..., description="Minimum answer text"), max_answer_text: str = Query(..., description="Maximum answer text")):
    cursor.execute("SELECT T1.UserID FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T1.SurveyID = ? AND T2.questionid = ? AND T1.AnswerText BETWEEN ? AND ?", (survey_id, question_id, min_answer_text, max_answer_text))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the count of users based on survey year, question IDs, and answer text
@app.get("/v1/mental_health_survey/user_count_by_survey_questions_and_answer_text", operation_id="get_user_count_by_survey_questions_and_answer_text", summary="Retrieves the number of users who have provided specific answers to designated questions in a given survey year. The operation accepts two sets of conditions, each comprising a survey year, a question ID, and an answer text. It returns the count of users who meet both sets of conditions.")
async def get_user_count_by_survey_questions_and_answer_text(survey_id_1: int = Query(..., description="Survey year for the first condition"), question_id_1: int = Query(..., description="ID of the first question"), answer_text_1: str = Query(..., description="Answer text for the first condition"), survey_id_2: int = Query(..., description="Survey year for the second condition"), question_id_2: int = Query(..., description="ID of the second question"), answer_text_2: str = Query(..., description="Answer text for the second condition")):
    cursor.execute("SELECT COUNT(T1.UserID) FROM Answer AS T1 INNER JOIN ( SELECT T2.questionid FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T1.SurveyID = ? AND T2.questionid = ? AND T1.AnswerText LIKE ? ) AS T2 ON T1.QuestionID = T2.questionid WHERE T1.SurveyID = ? AND T2.questionid = ? AND T1.AnswerText LIKE ?", (survey_id_1, question_id_1, answer_text_1, survey_id_2, question_id_2, answer_text_2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the survey ID with the highest user count based on survey year range, question ID, and answer text
@app.get("/v1/mental_health_survey/top_survey_by_year_range_question_and_answer", operation_id="get_top_survey_by_year_range_question_and_answer", summary="Retrieves the ID of the survey with the highest number of user responses within a specified year range, for a given question and answer. The survey range is defined by the start and end years, while the question and answer are identified by their respective ID and text.")
async def get_top_survey_by_year_range_question_and_answer(start_year: int = Query(..., description="Start year of the survey range"), end_year: int = Query(..., description="End year of the survey range"), question_id: int = Query(..., description="ID of the question"), answer_text: str = Query(..., description="Answer text")):
    cursor.execute("SELECT T1.SurveyID FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T1.SurveyID BETWEEN ? AND ? AND T2.questionid = ? AND T1.AnswerText LIKE ? GROUP BY T1.SurveyID ORDER BY COUNT(T1.UserID) DESC LIMIT 1", (start_year, end_year, question_id, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"survey_id": []}
    return {"survey_id": result[0]}

# Endpoint to get the count of users based on survey description, question IDs, and answer text
@app.get("/v1/mental_health_survey/user_count_by_survey_description_questions_and_answer_text", operation_id="get_user_count_by_survey_description_questions_and_answer_text", summary="Retrieve the number of users who have provided specific answers to two distinct questions within a survey that matches a given description. The count is determined by the intersection of users who meet both conditions.")
async def get_user_count_by_survey_description_questions_and_answer_text(survey_description: str = Query(..., description="Description of the survey"), question_id_1: int = Query(..., description="ID of the first question"), answer_text_1: str = Query(..., description="Answer text for the first condition"), question_id_2: int = Query(..., description="ID of the second question"), answer_text_2: str = Query(..., description="Answer text for the second condition")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T2.UserID FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID INNER JOIN Survey AS T3 ON T2.SurveyID = T3.SurveyID WHERE T3.Description = ? AND T1.questionid = ? AND T2.AnswerText = ? UNION SELECT T2.UserID FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID INNER JOIN Survey AS T3 ON T2.SurveyID = T3.SurveyID WHERE T1.questionid = ? AND T2.AnswerText = ? AND T3.Description = ? )", (survey_description, question_id_1, answer_text_1, question_id_2, answer_text_2, survey_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users based on question ID and answer text
@app.get("/v1/mental_health_survey/user_count_by_question_and_answer_text", operation_id="get_user_count_by_question_and_answer_text", summary="Retrieves the number of users who provided a specific answer to a given question in a mental health survey. The response is based on the provided question ID and a partial or full match of the answer text.")
async def get_user_count_by_question_and_answer_text(question_id: int = Query(..., description="ID of the question"), answer_text: str = Query(..., description="Answer text")):
    cursor.execute("SELECT COUNT(T1.UserID) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questionid = ? AND T1.AnswerText LIKE ?", (question_id, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum and minimum answer text and the youngest user ID based on question ID
@app.get("/v1/mental_health_survey/answer_text_stats_and_youngest_user", operation_id="get_answer_text_stats_and_youngest_user", summary="Retrieves the maximum and minimum answer text and the ID of the youngest user who answered a specific question. The youngest user is determined by the earliest answer text in alphabetical order.")
async def get_answer_text_stats_and_youngest_user(question_id: int = Query(..., description="ID of the question")):
    cursor.execute("SELECT MAX(T1.AnswerText), MIN(T1.AnswerText), ( SELECT T1.UserID FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questionid = ? ORDER BY T1.AnswerText LIMIT 1 ) AS 'youngest id' FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questionid = ?", (question_id, question_id))
    result = cursor.fetchone()
    if not result:
        return {"stats": []}
    return {"max_answer_text": result[0], "min_answer_text": result[1], "youngest_user_id": result[2]}

# Endpoint to get the most common answer text based on question ID
@app.get("/v1/mental_health_survey/most_common_answer_text", operation_id="get_most_common_answer_text", summary="Retrieves the most frequently provided answer text for a specific question in a mental health survey. The response is determined by grouping answers by text and ordering them by the number of users who provided that answer, in descending order. The top result is returned.")
async def get_most_common_answer_text(question_id: int = Query(..., description="ID of the question")):
    cursor.execute("SELECT T1.AnswerText FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questionid = ? GROUP BY T1.AnswerText ORDER BY COUNT(T1.UserID) DESC LIMIT 1", (question_id,))
    result = cursor.fetchone()
    if not result:
        return {"answer_text": []}
    return {"answer_text": result[0]}

# Endpoint to get the average answer text based on question IDs and answer text
@app.get("/v1/mental_health_survey/average_answer_text_by_questions_and_answer", operation_id="get_average_answer_text_by_questions_and_answer", summary="Retrieves the average answer text for a specific question and answer combination. The operation calculates the average by summing all instances of the provided answer text for the given question and dividing by the count of users who provided that answer. The input parameters include the IDs of the first and second questions, and the answer text to be averaged.")
async def get_average_answer_text_by_questions_and_answer(question_id_1: int = Query(..., description="ID of the first question"), answer_text: str = Query(..., description="Answer text"), question_id_2: int = Query(..., description="ID of the second question")):
    cursor.execute("SELECT CAST(SUM(T1.AnswerText) AS REAL) / COUNT(T1.UserID) FROM Answer AS T1 INNER JOIN ( SELECT T1.UserID FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questionid = ? AND T1.AnswerText = ? ) AS T2 ON T1.UserID = T2.UserID INNER JOIN Question AS T3 ON T1.QuestionID = T3.questionid WHERE T3.questionid = ?", (question_id_1, answer_text, question_id_2))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of 'Yes' answers for a specific question
@app.get("/v1/mental_health_survey/percentage_yes_answers", operation_id="get_percentage_yes_answers", summary="Retrieves the percentage of responses that match a specified answer for a given question in a mental health survey. The operation filters the answers by a provided text and calculates the percentage of 'Yes' answers for a specific question identified by its unique ID.")
async def get_percentage_yes_answers(answer_text: str = Query(..., description="Answer text to filter by"), question_id: int = Query(..., description="ID of the question")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN T1.AnswerText LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.UserID) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questionid = ?", (answer_text, question_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of answers for a specific survey ID and threshold
@app.get("/v1/mental_health_survey/count_answers_survey", operation_id="get_count_answers_survey", summary="Retrieves the count of answers for a specific survey, filtered by a given threshold. The count is determined by the number of unique question IDs associated with the survey. The results are ordered by whether the count exceeds the provided threshold, with the highest count returned first.")
async def get_count_answers_survey(survey_id: int = Query(..., description="Survey ID to filter by"), threshold: int = Query(..., description="Threshold count to filter by")):
    cursor.execute("SELECT COUNT(QuestionID) FROM Answer WHERE SurveyID LIKE ? GROUP BY QuestionID ORDER BY COUNT(QuestionID) > ? LIMIT 1", (survey_id, threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latest answer text for a specific question and survey ID
@app.get("/v1/mental_health_survey/latest_answer_text", operation_id="get_latest_answer_text", summary="Retrieves the most recent answer text associated with a specific question and survey. The operation filters the answers based on the provided question text and survey ID, then returns the latest answer text.")
async def get_latest_answer_text(question_text: str = Query(..., description="Question text to filter by"), survey_id: int = Query(..., description="Survey ID to filter by")):
    cursor.execute("SELECT T2.AnswerText FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T1.questiontext = ? AND T2.SurveyID = ? ORDER BY T2.AnswerText DESC LIMIT 1", (question_text, survey_id))
    result = cursor.fetchone()
    if not result:
        return {"answer_text": []}
    return {"answer_text": result[0]}

# Endpoint to get the count of users who answered a specific question in a specific way
@app.get("/v1/mental_health_survey/count_users_specific_answer", operation_id="get_count_users_specific_answer", summary="Retrieves the number of users who provided a specific response to a given question in a particular survey. The response is identified by a partial or full match with the provided answer text.")
async def get_count_users_specific_answer(question_text: str = Query(..., description="Question text to filter by"), survey_id: int = Query(..., description="Survey ID to filter by"), answer_text: str = Query(..., description="Answer text to filter by")):
    cursor.execute("SELECT COUNT(T2.UserID) FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T1.questiontext = ? AND T2.SurveyID = ? AND T2.AnswerText LIKE ?", (question_text, survey_id, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user IDs who answered a specific question in a specific way
@app.get("/v1/mental_health_survey/user_ids_specific_answer", operation_id="get_user_ids_specific_answer", summary="Retrieves the user IDs of individuals who provided a specific response to a given question within a particular survey. The operation filters the users based on the exact question text, the answer text pattern, and the survey ID.")
async def get_user_ids_specific_answer(question_text: str = Query(..., description="Question text to filter by"), answer_text: str = Query(..., description="Answer text to filter by"), survey_id: int = Query(..., description="Survey ID to filter by")):
    cursor.execute("SELECT T2.UserID FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T1.questiontext = ? AND T2.AnswerText LIKE ? AND T2.SurveyID = ?", (question_text, answer_text, survey_id))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct users for a specific survey description
@app.get("/v1/mental_health_survey/count_distinct_users", operation_id="get_count_distinct_users", summary="Retrieves the total number of unique users who have responded to a specific mental health survey. The survey is identified by its description, which is provided as an input parameter.")
async def get_count_distinct_users(description: str = Query(..., description="Survey description to filter by")):
    cursor.execute("SELECT COUNT(DISTINCT T1.UserID) FROM Answer AS T1 INNER JOIN Survey AS T2 ON T1.SurveyID = T2.SurveyID WHERE T2.Description = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get answer texts for a specific question and survey ID, excluding a specific answer
@app.get("/v1/mental_health_survey/answer_texts_excluding", operation_id="get_answer_texts_excluding", summary="Retrieves the answer texts for a specific question and survey, excluding a particular answer. The operation filters the answers based on the provided question text and survey ID, and then excludes the specified answer text from the results.")
async def get_answer_texts_excluding(question_text: str = Query(..., description="Question text to filter by"), survey_id: int = Query(..., description="Survey ID to filter by"), exclude_answer: str = Query(..., description="Answer text to exclude")):
    cursor.execute("SELECT T2.AnswerText FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T1.questiontext = ? AND T2.SurveyID = ? AND T2.AnswerText <> ?", (question_text, survey_id, exclude_answer))
    result = cursor.fetchall()
    if not result:
        return {"answer_texts": []}
    return {"answer_texts": [row[0] for row in result]}

# Endpoint to get question texts for specific survey IDs
@app.get("/v1/mental_health_survey/question_texts_survey_ids", operation_id="get_question_texts_survey_ids", summary="Retrieves unique question texts associated with the provided survey IDs. This operation filters questions based on the specified survey IDs and returns a distinct list of question texts.")
async def get_question_texts_survey_ids(survey_id_1: int = Query(..., description="First survey ID to filter by"), survey_id_2: int = Query(..., description="Second survey ID to filter by")):
    cursor.execute("SELECT T1.questiontext FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T2.SurveyID IN (?, ?) GROUP BY T1.questiontext", (survey_id_1, survey_id_2))
    result = cursor.fetchall()
    if not result:
        return {"question_texts": []}
    return {"question_texts": [row[0] for row in result]}

# Endpoint to get the count of users who answered a specific question in a specific way for a specific survey ID
@app.get("/v1/mental_health_survey/count_users_specific_answer_survey", operation_id="get_count_users_specific_answer_survey", summary="Retrieves the number of users who provided a specific response to a particular question in a given survey. The count is determined based on the provided survey ID, question text, and the corresponding answer text.")
async def get_count_users_specific_answer_survey(survey_id: int = Query(..., description="Survey ID to filter by"), question_text: str = Query(..., description="Question text to filter by"), answer_text: str = Query(..., description="Answer text to filter by")):
    cursor.execute("SELECT COUNT(T2.UserID) FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T2.SurveyID = ? AND T1.questiontext = ? AND T2.AnswerText = ?", (survey_id, question_text, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get question texts for a specific survey description
@app.get("/v1/mental_health_survey/question_texts_survey_description", operation_id="get_question_texts_survey_description", summary="Retrieves the question texts associated with a specific survey description. The operation filters the questions based on the provided survey description, ensuring that only relevant questions are returned. This endpoint is useful for obtaining a list of questions that pertain to a particular survey theme or topic.")
async def get_question_texts_survey_description(description: str = Query(..., description="Survey description to filter by")):
    cursor.execute("SELECT T2.questiontext FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid INNER JOIN Survey AS T3 ON T1.SurveyID = T3.SurveyID WHERE T3.Description LIKE ? GROUP BY T2.questiontext", (description,))
    result = cursor.fetchall()
    if not result:
        return {"question_texts": []}
    return {"question_texts": [row[0] for row in result]}

# Endpoint to get the count of users who answered a specific question in a specific survey with a specific answer
@app.get("/v1/mental_health_survey/count_users_specific_question_answer", operation_id="get_count_users_specific_question_answer", summary="Retrieves the number of users who provided a specific answer to a given question within a designated survey. The input parameters include the survey ID, the question text, and the answer text.")
async def get_count_users_specific_question_answer(survey_id: int = Query(..., description="Survey ID"), question_text: str = Query(..., description="Question text"), answer_text: str = Query(..., description="Answer text")):
    cursor.execute("SELECT COUNT(T2.UserID) FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T2.SurveyID = ? AND T1.questiontext LIKE ? AND T2.AnswerText = ?", (survey_id, question_text, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users who answered a specific question in multiple surveys with a specific answer
@app.get("/v1/mental_health_survey/count_users_multiple_surveys_question_answer", operation_id="get_count_users_multiple_surveys_question_answer", summary="Retrieves the number of users who provided a specific answer to a given question across multiple surveys. The operation requires the IDs of the surveys, the exact question text, and the answer text.")
async def get_count_users_multiple_surveys_question_answer(survey_id_1: int = Query(..., description="First Survey ID"), survey_id_2: int = Query(..., description="Second Survey ID"), survey_id_3: int = Query(..., description="Third Survey ID"), question_text: str = Query(..., description="Question text"), answer_text: str = Query(..., description="Answer text")):
    cursor.execute("SELECT COUNT(T2.UserID) FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T2.SurveyID IN (?, ?, ?) AND T1.questiontext LIKE ? AND T2.AnswerText = ?", (survey_id_1, survey_id_2, survey_id_3, question_text, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average answer text for a specific question in a specific survey
@app.get("/v1/mental_health_survey/average_answer_text_specific_question", operation_id="get_average_answer_text_specific_question", summary="Retrieves the average answer text for a specific question within a given mental health survey. The operation calculates the average by summing all answer texts for the specified question and dividing by the count of unique users who answered. The input parameters include the survey ID and the question text, which are used to filter the results.")
async def get_average_answer_text_specific_question(survey_id: int = Query(..., description="Survey ID"), question_text: str = Query(..., description="Question text")):
    cursor.execute("SELECT CAST(SUM(T2.AnswerText) AS REAL) / COUNT(T2.UserID) AS 'avg' FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T2.SurveyID = ? AND T1.questiontext LIKE ?", (survey_id, question_text))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage change in the count of users who answered a specific question with a specific answer between two surveys
@app.get("/v1/mental_health_survey/percentage_change_users_specific_question_answer", operation_id="get_percentage_change_users_specific_question_answer", summary="Retrieves the percentage change in the number of users who provided a specific answer to a specific question between two surveys. This operation compares the count of users who answered a particular question with a specific answer in the first survey to the count in the second survey. The result is expressed as a percentage of the count in the first survey.")
async def get_percentage_change_users_specific_question_answer(survey_id_1: int = Query(..., description="First Survey ID"), survey_id_2: int = Query(..., description="Second Survey ID"), question_text: str = Query(..., description="Question text"), answer_text: str = Query(..., description="Answer text")):
    cursor.execute("SELECT CAST(( SELECT COUNT(T2.UserID) FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T2.SurveyID = ? AND T1.questiontext LIKE ? AND T2.AnswerText = ? ) - ( SELECT COUNT(T2.UserID) FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T2.SurveyID = ? AND T1.questiontext LIKE ? AND T2.AnswerText = ? ) AS REAL) * 100 / ( SELECT COUNT(T2.UserID) FROM Question AS T1 INNER JOIN Answer AS T2 ON T1.questionid = T2.QuestionID WHERE T2.SurveyID = ? AND T1.questiontext LIKE ? AND T2.AnswerText = ? )", (survey_id_1, question_text, answer_text, survey_id_2, question_text, answer_text, survey_id_2, question_text, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result[0]}

# Endpoint to get the question IDs based on the question text
@app.get("/v1/mental_health_survey/question_ids_by_text", operation_id="get_question_ids_by_text", summary="Retrieves the unique identifiers of mental health survey questions that match the provided question text. The search is case-insensitive and supports partial matches.")
async def get_question_ids_by_text(question_text: str = Query(..., description="Question text")):
    cursor.execute("SELECT questionid FROM Question WHERE questiontext LIKE ?", (question_text,))
    result = cursor.fetchall()
    if not result:
        return {"question_ids": []}
    return {"question_ids": [row[0] for row in result]}

# Endpoint to get the range of UserIDs for a specific question
@app.get("/v1/mental_health_survey/user_id_range_by_question", operation_id="get_user_id_range_by_question", summary="Retrieves the total number of unique users who have answered a specific question in the mental health survey. The calculation is based on the range between the maximum and minimum UserID associated with the given QuestionID.")
async def get_user_id_range_by_question(question_id: int = Query(..., description="Question ID")):
    cursor.execute("SELECT MAX(UserID) - MIN(UserID) + 1 FROM Answer WHERE QuestionID = ?", (question_id,))
    result = cursor.fetchone()
    if not result:
        return {"user_id_range": []}
    return {"user_id_range": result[0]}

# Endpoint to get the count of questions answered by a specific user
@app.get("/v1/mental_health_survey/count_questions_by_user", operation_id="get_count_questions_by_user", summary="Retrieves the total number of questions answered by a specific user. The operation requires the user's unique identifier as input to accurately determine the count.")
async def get_count_questions_by_user(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT COUNT(QuestionID) FROM Answer WHERE UserID = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct users who answered a specific survey
@app.get("/v1/mental_health_survey/count_distinct_users_by_survey", operation_id="get_count_distinct_users_by_survey", summary="Retrieves the total number of unique users who have responded to a specific survey. The survey is identified by its unique ID, which is provided as an input parameter.")
async def get_count_distinct_users_by_survey(survey_id: int = Query(..., description="Survey ID")):
    cursor.execute("SELECT COUNT(DISTINCT UserID) FROM Answer WHERE SurveyID LIKE ?", (survey_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of questions answered in a survey with a specific description
@app.get("/v1/mental_health_survey/count_questions_by_survey_description", operation_id="get_count_questions_by_survey_description", summary="Retrieves the total number of questions that have been answered in a survey with a specified description. The survey description is used to identify the relevant survey and count the associated questions.")
async def get_count_questions_by_survey_description(description: str = Query(..., description="Survey description")):
    cursor.execute("SELECT COUNT(T1.QuestionID) FROM Answer AS T1 INNER JOIN Survey AS T2 ON T1.SurveyID = T2.SurveyID WHERE T2.Description = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct questions with a specific text across all surveys
@app.get("/v1/mental_health_survey/count_distinct_questions_by_text", operation_id="get_count_distinct_questions_by_text", summary="Retrieve the total number of unique questions that match a specific text across all mental health surveys. The input parameter specifies the question text to be matched.")
async def get_count_distinct_questions_by_text(question_text: str = Query(..., description="Question text")):
    cursor.execute("SELECT COUNT(DISTINCT T1.QuestionID) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid INNER JOIN Survey AS T3 ON T1.SurveyID = T3.SurveyID WHERE T2.questiontext = ?", (question_text,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the answer text for a specific question and user
@app.get("/v1/mental_health_survey/answer_text_by_question_and_user", operation_id="get_answer_text_by_question_and_user", summary="Retrieves the textual response provided by a specific user for a given question in a mental health survey. The operation requires the exact question text and the user's unique identifier to locate the corresponding answer.")
async def get_answer_text_by_question_and_user(question_text: str = Query(..., description="Text of the question"), user_id: int = Query(..., description="ID of the user")):
    cursor.execute("SELECT T1.AnswerText FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questiontext = ? AND T1.UserID = ?", (question_text, user_id))
    result = cursor.fetchone()
    if not result:
        return {"answer_text": []}
    return {"answer_text": result[0]}

# Endpoint to get the most common answer text for a specific question
@app.get("/v1/mental_health_survey/most_common_answer_text_by_question", operation_id="get_most_common_answer_text_by_question", summary="Retrieves the most frequently provided answer for a specific question in the mental health survey. The input parameter is used to identify the question for which the most common answer is sought. The response will contain the answer text that has been most frequently submitted for the specified question.")
async def get_most_common_answer_text_by_question(question_text: str = Query(..., description="Text of the question")):
    cursor.execute("SELECT T1.AnswerText FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questiontext = ? GROUP BY T1.AnswerText ORDER BY COUNT(T1.AnswerText) DESC LIMIT 1", (question_text,))
    result = cursor.fetchone()
    if not result:
        return {"answer_text": []}
    return {"answer_text": result[0]}

# Endpoint to get the count of distinct answer texts for a specific question
@app.get("/v1/mental_health_survey/count_distinct_answer_texts_by_question", operation_id="get_count_distinct_answer_texts_by_question", summary="Retrieves the number of unique responses to a specific question. The question is identified by its text. This operation is useful for understanding the diversity of responses to a particular question, which can provide insights into the range of experiences or perspectives shared by respondents.")
async def get_count_distinct_answer_texts_by_question(question_text: str = Query(..., description="Text of the question")):
    cursor.execute("SELECT COUNT(DISTINCT T1.AnswerText) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questiontext LIKE ?", (question_text,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users for a specific question and answer text
@app.get("/v1/mental_health_survey/count_users_by_question_and_answer", operation_id="get_count_users_by_question_and_answer", summary="Retrieves the number of users who provided a specific answer to a given question in a mental health survey. The operation requires the exact text of the question and the corresponding answer as input parameters.")
async def get_count_users_by_question_and_answer(question_text: str = Query(..., description="Text of the question"), answer_text: str = Query(..., description="Text of the answer")):
    cursor.execute("SELECT COUNT(T1.UserID) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questiontext LIKE ? AND T1.AnswerText = ?", (question_text, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users for a specific question with non-null answer text
@app.get("/v1/mental_health_survey/count_users_by_question_with_non_null_answer", operation_id="get_count_users_by_question_with_non_null_answer", summary="Retrieves the number of users who have provided a non-null answer for a specific question. The question is identified by its text.")
async def get_count_users_by_question_with_non_null_answer(question_text: str = Query(..., description="Text of the question")):
    cursor.execute("SELECT COUNT(T1.UserID) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questiontext LIKE ? AND T1.AnswerText IS NOT NULL", (question_text,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users for a specific question
@app.get("/v1/mental_health_survey/count_users_by_question", operation_id="get_count_users_by_question", summary="Retrieves the total number of users who have responded to a specific question regarding the importance of physical health in their workplace.")
async def get_count_users_by_question(question_text: str = Query(..., description="Text of the question")):
    cursor.execute("SELECT COUNT(T1.UserID) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questiontext LIKE ?", (question_text,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the answer text for a specific question ID and answer text
@app.get("/v1/mental_health_survey/answer_text_by_question_id_and_answer", operation_id="get_answer_text_by_question_id_and_answer", summary="Retrieves the specific answer text associated with a given question ID and answer text. This operation is useful for obtaining detailed information about a particular response to a mental health survey question. The input parameters include the question ID and the answer text, which are used to filter the results and return the exact answer text that matches the provided criteria.")
async def get_answer_text_by_question_id_and_answer(question_id: int = Query(..., description="ID of the question"), answer_text: str = Query(..., description="Text of the answer")):
    cursor.execute("SELECT T1.AnswerText FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T2.questionid = ? AND T1.AnswerText = ?", (question_id, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"answer_text": []}
    return {"answer_text": result[0]}

# Endpoint to get the ratio of users who participated in a survey in a given year compared to another year
@app.get("/v1/mental_health_survey/user_participation_ratio", operation_id="get_user_participation_ratio", summary="Retrieves the proportion of users who participated in a specific survey in a given year compared to another year. The operation calculates the ratio by dividing the number of users who participated in the survey in the first year by the number of users who participated in the survey in the second year.")
async def get_user_participation_ratio(description_year1: str = Query(..., description="Description of the survey for the first year"), description_year2: str = Query(..., description="Description of the survey for the second year")):
    cursor.execute("SELECT CAST(COUNT(T1.UserID) AS REAL) / ( SELECT COUNT(T1.UserID) FROM Answer AS T1 INNER JOIN Survey AS T2 ON T1.SurveyID = T2.SurveyID WHERE T2.Description = ? ) FROM Answer AS T1 INNER JOIN Survey AS T2 ON T1.SurveyID = T2.SurveyID WHERE T2.Description = ?", (description_year1, description_year2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the percentage of 'Yes' answers for a specific question and survey
@app.get("/v1/mental_health_survey/percentage_yes_answers_question_survey", operation_id="get_percentage_yes_answers_question_survey", summary="Retrieves the percentage of 'Yes' responses for a specific question within a given survey. The calculation is based on the total number of responses to the question in the survey. The operation requires the unique identifiers for the question and the survey as input parameters.")
async def get_percentage_yes_answers_question_survey(question_id: int = Query(..., description="ID of the question"), survey_id: int = Query(..., description="ID of the survey")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN T1.AnswerText LIKE 'Yes' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.SurveyID) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T1.QuestionID = ? AND T1.SurveyID = ?", (question_id, survey_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of distinct users who answered a specific question in a survey with an answer less than or equal to a given value
@app.get("/v1/mental_health_survey/count_distinct_users_question_survey", operation_id="get_count_distinct_users_question_survey", summary="Retrieve the number of unique users who provided an answer to a specific question in a survey, with the answer value being less than or equal to a specified maximum. This operation considers both the question and survey IDs to ensure accurate results.")
async def get_count_distinct_users_question_survey(question_id: int = Query(..., description="ID of the question"), survey_id: int = Query(..., description="ID of the survey"), max_answer_text: int = Query(..., description="Maximum value of the answer text")):
    cursor.execute("SELECT COUNT(DISTINCT T1.UserID) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T1.QuestionID = ? AND T1.SurveyID = ? AND T1.AnswerText <= ?", (question_id, survey_id, max_answer_text))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average number of surveys taken per year within a given range
@app.get("/v1/mental_health_survey/average_surveys_per_year", operation_id="get_average_surveys_per_year", summary="Retrieves the average number of mental health surveys taken per year within a specified range. The range is defined by the start and end years provided as input parameters. The calculation is based on the total count of surveys taken, divided by the number of years in the range.")
async def get_average_surveys_per_year(start_year: int = Query(..., description="Start year of the survey range"), end_year: int = Query(..., description="End year of the survey range")):
    cursor.execute("SELECT CAST(COUNT(SurveyID) AS REAL) / 5 FROM Answer WHERE SurveyID BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the count of specific answers to a question in a survey
@app.get("/v1/mental_health_survey/count_specific_answers", operation_id="get_count_specific_answers", summary="Retrieves the count of a specific answer provided for a given question in a particular survey. The input parameters determine the question, survey, and answer text for which the count is calculated.")
async def get_count_specific_answers(question_id: int = Query(..., description="ID of the question"), survey_id: int = Query(..., description="ID of the survey"), answer_text: str = Query(..., description="The specific answer text")):
    cursor.execute("SELECT COUNT(T1.AnswerText) FROM Answer AS T1 INNER JOIN Question AS T2 ON T1.QuestionID = T2.questionid WHERE T1.QuestionID = ? AND T1.SurveyID = ? AND T1.AnswerText = ?", (question_id, survey_id, answer_text))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total count of questions
@app.get("/v1/mental_health_survey/total_questions", operation_id="get_total_questions", summary="Retrieves the total number of questions available in the mental health survey.")
async def get_total_questions():
    cursor.execute("SELECT COUNT(questiontext) FROM Question")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of answers matching a specific text
@app.get("/v1/mental_health_survey/count_answers_matching_text", operation_id="get_count_answers_matching_text", summary="Retrieves the total count of answers that contain a specified text. The input parameter determines the text to be matched in the answers.")
async def get_count_answers_matching_text(answer_text: str = Query(..., description="The text to match in the answers")):
    cursor.execute("SELECT COUNT(AnswerText) FROM Answer WHERE AnswerText LIKE ?", (answer_text,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top 3 most common answers for a specific question in a survey
@app.get("/v1/mental_health_survey/top_answers", operation_id="get_top_answers", summary="Retrieves the three most frequently provided responses to a specific question within a given survey. The operation requires the unique identifiers for the question and the survey as input parameters.")
async def get_top_answers(question_id: int = Query(..., description="ID of the question"), survey_id: int = Query(..., description="ID of the survey")):
    cursor.execute("SELECT AnswerText FROM Answer WHERE QuestionID = ? AND SurveyID = ? GROUP BY AnswerText ORDER BY COUNT(AnswerText) DESC LIMIT 3", (question_id, survey_id))
    result = cursor.fetchall()
    if not result:
        return {"answers": []}
    return {"answers": [row[0] for row in result]}

api_calls = [
    "/v1/mental_health_survey/answer_count_by_question_and_text?question_id=19&answer_text=No",
    "/v1/mental_health_survey/user_count_by_question_and_survey_years?question_id=13&start_year=2016&end_year=2019",
    "/v1/mental_health_survey/user_ids_by_survey_question_and_answer_range?survey_id=2018&question_id=1&min_answer_text=27&max_answer_text=35",
    "/v1/mental_health_survey/user_count_by_survey_questions_and_answer_text?survey_id_1=2019&question_id_1=6&answer_text_1=Yes&survey_id_2=2019&question_id_2=3&answer_text_2=United%20States",
    "/v1/mental_health_survey/top_survey_by_year_range_question_and_answer?start_year=2016&end_year=2019&question_id=34&answer_text=Yes",
    "/v1/mental_health_survey/user_count_by_survey_description_questions_and_answer_text?survey_description=mental%20health%20survey%20for%202017&question_id_1=2&answer_text_1=Female&question_id_2=4&answer_text_2=Nebraska",
    "/v1/mental_health_survey/user_count_by_question_and_answer_text?question_id=54&answer_text=Yes",
    "/v1/mental_health_survey/answer_text_stats_and_youngest_user?question_id=1",
    "/v1/mental_health_survey/most_common_answer_text?question_id=3",
    "/v1/mental_health_survey/average_answer_text_by_questions_and_answer?question_id_1=3&answer_text=United%20States&question_id_2=1",
    "/v1/mental_health_survey/percentage_yes_answers?answer_text=Yes&question_id=12",
    "/v1/mental_health_survey/count_answers_survey?survey_id=2014&threshold=200",
    "/v1/mental_health_survey/latest_answer_text?question_text=What%20is%20your%20age%3F&survey_id=2014",
    "/v1/mental_health_survey/count_users_specific_answer?question_text=Would%20you%20bring%20up%20a%20mental%20health%20issue%20with%20a%20potential%20employer%20in%20an%20interview%3F&survey_id=2014&answer_text=NO",
    "/v1/mental_health_survey/user_ids_specific_answer?question_text=Do%20you%20think%20that%20discussing%20a%20physical%20health%20issue%20with%20your%20employer%20would%20have%20negative%20consequences%3F&answer_text=Yes&survey_id=2014",
    "/v1/mental_health_survey/count_distinct_users?description=mental%20health%20survey%20for%202014",
    "/v1/mental_health_survey/answer_texts_excluding?question_text=Any%20additional%20notes%20or%20comments&survey_id=2014&exclude_answer=-1",
    "/v1/mental_health_survey/question_texts_survey_ids?survey_id_1=2014&survey_id_2=2016",
    "/v1/mental_health_survey/count_users_specific_answer_survey?survey_id=2018&question_text=What%20country%20do%20you%20live%20in%3F&answer_text=Canada",
    "/v1/mental_health_survey/question_texts_survey_description?description=mental%20health%20survey%20for%202014",
    "/v1/mental_health_survey/count_users_specific_question_answer?survey_id=2016&question_text=Have%20you%20had%20a%20mental%20health%20disorder%20in%20the%20past%3F&answer_text=Yes",
    "/v1/mental_health_survey/count_users_multiple_surveys_question_answer?survey_id_1=2016&survey_id_2=2017&survey_id_3=2018&question_text=Have%20you%20had%20a%20mental%20health%20disorder%20in%20the%20past%3F&answer_text=Yes",
    "/v1/mental_health_survey/average_answer_text_specific_question?survey_id=2014&question_text=What%20is%20your%20age%3F",
    "/v1/mental_health_survey/percentage_change_users_specific_question_answer?survey_id_1=2019&survey_id_2=2016&question_text=Do%20you%20currently%20have%20a%20mental%20health%20disorder%3F&answer_text=Yes",
    "/v1/mental_health_survey/question_ids_by_text?question_text=Would%20you%20bring%20up%20a%20physical%20health%20issue%20with%20a%20potential%20employer%20in%20an%20interview%3F",
    "/v1/mental_health_survey/user_id_range_by_question?question_id=20",
    "/v1/mental_health_survey/count_questions_by_user?user_id=5",
    "/v1/mental_health_survey/count_distinct_users_by_survey?survey_id=2016",
    "/v1/mental_health_survey/count_questions_by_survey_description?description=mental%20health%20survey%20for%202018",
    "/v1/mental_health_survey/count_distinct_questions_by_text?question_text=What%20country%20do%20you%20work%20in%3F",
    "/v1/mental_health_survey/answer_text_by_question_and_user?question_text=Do%20you%20currently%20have%20a%20mental%20health%20disorder%3F&user_id=2681",
    "/v1/mental_health_survey/most_common_answer_text_by_question?question_text=What%20country%20do%20you%20work%20in%3F",
    "/v1/mental_health_survey/count_distinct_answer_texts_by_question?question_text=Describe%20the%20conversation%20you%20had%20with%20your%20previous%20employer%20about%20your%20mental%20health%2C%20including%20their%20reactions%20and%20actions%20taken%20to%20address%20your%20mental%20health%20issue%2Fquestions.",
    "/v1/mental_health_survey/count_users_by_question_and_answer?question_text=What%20US%20state%20or%20territory%20do%20you%20work%20in%3F&answer_text=Kansas",
    "/v1/mental_health_survey/count_users_by_question_with_non_null_answer?question_text=Any%20additional%20notes%20or%20comments",
    "/v1/mental_health_survey/count_users_by_question?question_text=Overall%2C%20how%20much%20importance%20does%20your%20employer%20place%20on%20physical%20health%3F",
    "/v1/mental_health_survey/answer_text_by_question_id_and_answer?question_id=2183&answer_text=Mood%20Disorder%20(Depression%2C%20Bipolar%20Disorder%2C%20etc)",
    "/v1/mental_health_survey/user_participation_ratio?description_year1=mental%20health%20survey%20for%202018&description_year2=mental%20health%20survey%20for%202017",
    "/v1/mental_health_survey/percentage_yes_answers_question_survey?question_id=32&survey_id=2016",
    "/v1/mental_health_survey/count_distinct_users_question_survey?question_id=1&survey_id=2016&max_answer_text=25",
    "/v1/mental_health_survey/average_surveys_per_year?start_year=2014&end_year=2019",
    "/v1/mental_health_survey/count_specific_answers?question_id=93&survey_id=2014&answer_text=Yes",
    "/v1/mental_health_survey/total_questions",
    "/v1/mental_health_survey/count_answers_matching_text?answer_text=Substance%20Use%20Disorder",
    "/v1/mental_health_survey/top_answers?question_id=85&survey_id=2017"
]
