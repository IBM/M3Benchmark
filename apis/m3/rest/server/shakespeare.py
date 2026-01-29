from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/shakespeare/shakespeare.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of works before a specific date
@app.get("/v1/shakespeare/count_works_before_date", operation_id="get_count_works_before_date", summary="Retrieves the total number of works created prior to a specified year. The year is provided in the YYYY format.")
async def get_count_works_before_date(date: int = Query(..., description="Date in YYYY format")):
    cursor.execute("SELECT COUNT(id) FROM works WHERE Date < ?", (date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of works with a specific title and act number
@app.get("/v1/shakespeare/count_works_by_title_and_act", operation_id="get_count_works_by_title_and_act", summary="Retrieve the total number of works that match a given title and act number. The operation filters works based on the provided act number and title, then returns the count of matching works.")
async def get_count_works_by_title_and_act(act: int = Query(..., description="Act number"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT COUNT(T1.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.Act = ? AND T1.Title = ?", (act, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the description of a chapter in a specific work, act, and scene
@app.get("/v1/shakespeare/get_chapter_description", operation_id="get_chapter_description", summary="Retrieves the description of a specific chapter in a given work, act, and scene. The operation uses the provided title, act, and scene parameters to identify the chapter and return its description.")
async def get_chapter_description(title: str = Query(..., description="Title of the work"), act: int = Query(..., description="Act number"), scene: int = Query(..., description="Scene number")):
    cursor.execute("SELECT T2.Description FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.Title = ? AND T2.Act = ? AND T2.Scene = ?", (title, act, scene))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the difference in the number of acts between two specified acts in a work
@app.get("/v1/shakespeare/get_act_difference", operation_id="get_act_difference", summary="Retrieve the difference in the number of acts between two specified acts in a given work. The operation compares the total number of acts in the first specified act with the total number of acts in the second specified act. The work is identified by its title.")
async def get_act_difference(act1: int = Query(..., description="First act number"), act2: int = Query(..., description="Second act number"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT SUM(IIF(T2.Act = ?, 1, 0)) - SUM(IIF(T2.Act = ?, 1, 0)) AS more FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.Title = ?", (act1, act2, title))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get distinct titles of works featuring a specific character
@app.get("/v1/shakespeare/get_titles_by_character", operation_id="get_titles_by_character", summary="Retrieve a unique list of titles from works that feature a specified character. The character's name is provided as an input parameter.")
async def get_titles_by_character(char_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT DISTINCT T1.Title FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T4.CharName = ?", (char_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get distinct character names in a specific work
@app.get("/v1/shakespeare/get_characters_by_title", operation_id="get_characters_by_title", summary="Retrieves a list of unique character names that appear in a specified work. The operation filters the characters based on the provided work title.")
async def get_characters_by_title(title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT DISTINCT T4.CharName FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T1.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get the sum of paragraph numbers in a specific work, act, and scene
@app.get("/v1/shakespeare/get_sum_paragraph_numbers", operation_id="get_sum_paragraph_numbers", summary="Retrieves the total number of paragraphs in a specific act and scene of a given work by William Shakespeare. The input parameters specify the act, scene, and title of the work, allowing the operation to calculate and return the sum of paragraph numbers in the requested section.")
async def get_sum_paragraph_numbers(act: int = Query(..., description="Act number"), scene: int = Query(..., description="Scene number"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT SUM(T3.ParagraphNum) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id WHERE T2.Act = ? AND T2.Scene = ? AND T1.Title = ?", (act, scene, title))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the plain text of paragraphs in a specific work, act, and scene
@app.get("/v1/shakespeare/get_paragraph_text", operation_id="get_paragraph_text", summary="Retrieve the plain text of paragraphs from a specific act and scene within a given work. The operation requires the act and scene numbers, as well as the title of the work, to accurately locate and return the requested text.")
async def get_paragraph_text(act: int = Query(..., description="Act number"), scene: int = Query(..., description="Scene number"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT T3.PlainText FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id WHERE T2.Act = ? AND T2.Scene = ? AND T1.Title = ?", (act, scene, title))
    result = cursor.fetchall()
    if not result:
        return {"text": []}
    return {"text": [row[0] for row in result]}

# Endpoint to get the sum of paragraph numbers for a specific character
@app.get("/v1/shakespeare/get_sum_paragraph_numbers_by_character", operation_id="get_sum_paragraph_numbers_by_character", summary="Retrieves the total number of paragraphs spoken by a specific character in the Shakespearean corpus. The character is identified by the provided name.")
async def get_sum_paragraph_numbers_by_character(char_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT SUM(T1.ParagraphNum) FROM paragraphs AS T1 INNER JOIN characters AS T2 ON T1.character_id = T2.id WHERE T2.CharName = ?", (char_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get paragraph IDs for a specific character description
@app.get("/v1/shakespeare/get_paragraph_ids_by_character_description", operation_id="get_paragraph_ids_by_character_description", summary="Retrieves the IDs of paragraphs associated with a character that matches the provided description. The character description is used to filter the results.")
async def get_paragraph_ids_by_character_description(description: str = Query(..., description="Character description")):
    cursor.execute("SELECT T1.id FROM paragraphs AS T1 INNER JOIN characters AS T2 ON T1.character_id = T2.id WHERE T2.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get the count of chapters based on description and work title
@app.get("/v1/shakespeare/count_chapters_by_description_and_title", operation_id="get_count_chapters", summary="Retrieve the total number of chapters in a specific work that match a given description. The operation requires the title of the work and the description of the chapter as input parameters.")
async def get_count_chapters(description: str = Query(..., description="Description of the chapter"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT COUNT(T2.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.Description = ? AND T1.Title = ?", (description, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct characters in a specific work
@app.get("/v1/shakespeare/count_distinct_characters_in_work", operation_id="get_count_distinct_characters", summary="Retrieves the total number of unique characters that appear in a specified work. The operation considers all chapters and paragraphs within the work to ensure an accurate count.")
async def get_count_distinct_characters(title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT COUNT(DISTINCT T4.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the title of the work with the highest number of character appearances
@app.get("/v1/shakespeare/work_with_most_characters", operation_id="get_work_with_most_characters", summary="Retrieves the title of the work with the most character appearances. This operation calculates the number of appearances for each character across all works, chapters, and paragraphs, then returns the title of the work with the highest count.")
async def get_work_with_most_characters():
    cursor.execute("SELECT T.Title FROM ( SELECT T1.Title, COUNT(T3.character_id) AS num FROM works T1 INNER JOIN chapters T2 ON T1.id = T2.work_id INNER JOIN paragraphs T3 ON T2.id = T3.chapter_id INNER JOIN characters T4 ON T3.character_id = T4.id GROUP BY T3.character_id, T1.Title ) T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the average number of distinct characters per work
@app.get("/v1/shakespeare/average_distinct_characters_per_work", operation_id="get_average_distinct_characters", summary="Retrieves the average count of unique characters that appear across all works. This calculation is based on the total number of distinct characters divided by the total number of works.")
async def get_average_distinct_characters():
    cursor.execute("SELECT SUM(DISTINCT T4.id) / COUNT(T1.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id")
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the average number of scenes per act in a specific work
@app.get("/v1/shakespeare/average_scenes_per_act", operation_id="get_average_scenes_per_act", summary="Retrieves the average number of scenes per act in a specified work. The calculation is based on the total number of scenes divided by the total number of acts in the work. The work is identified by its title.")
async def get_average_scenes_per_act(title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT SUM(T2.Scene) / COUNT(T2.Act) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the count of works based on genre type
@app.get("/v1/shakespeare/count_works_by_genre", operation_id="get_count_works_by_genre", summary="Retrieves the total number of works belonging to a specified genre. The genre is provided as an input parameter.")
async def get_count_works_by_genre(genre_type: str = Query(..., description="Genre type of the work")):
    cursor.execute("SELECT COUNT(id) FROM works WHERE GenreType = ?", (genre_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the earliest date of a work based on genre type
@app.get("/v1/shakespeare/earliest_work_date_by_genre", operation_id="get_earliest_work_date_by_genre", summary="Retrieve the earliest date of publication for a work belonging to a specified genre. The genre type is provided as an input parameter.")
async def get_earliest_work_date_by_genre(genre_type: str = Query(..., description="Genre type of the work")):
    cursor.execute("SELECT MIN(Date) FROM works WHERE GenreType = ?", (genre_type,))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the distinct abbreviations of a character based on character name
@app.get("/v1/shakespeare/distinct_abbreviations_by_charname", operation_id="get_distinct_abbreviations", summary="Retrieve a unique set of abbreviations associated with a specific character in the Shakespearean universe. The character is identified by its name, which is provided as an input parameter.")
async def get_distinct_abbreviations(char_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT DISTINCT Abbrev FROM characters WHERE CharName = ?", (char_name,))
    result = cursor.fetchall()
    if not result:
        return {"abbreviations": []}
    return {"abbreviations": [row[0] for row in result]}

# Endpoint to get the description of the chapter with the highest paragraph number
@app.get("/v1/shakespeare/chapter_description_with_highest_paragraph", operation_id="get_chapter_description_with_highest_paragraph", summary="Retrieves the description of the chapter that contains the paragraph with the highest number. This operation identifies the chapter with the highest paragraph number by joining the chapters and paragraphs tables and ordering the results by the paragraph number in descending order. The description of the chapter with the highest paragraph number is then returned.")
async def get_chapter_description_with_highest_paragraph():
    cursor.execute("SELECT T1.Description FROM chapters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.chapter_id ORDER BY T2.ParagraphNum DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the character name based on specific plain text in paragraphs
@app.get("/v1/shakespeare/character_name_by_plaintext", operation_id="get_character_name_by_plaintext", summary="Retrieves the name of a character from Shakespeare's works based on a specific plain text provided in a paragraph. The input plain text is used to search for a matching paragraph, and the character associated with that paragraph is returned.")
async def get_character_name_by_plaintext(plain_text: str = Query(..., description="Plain text in the paragraph")):
    cursor.execute("SELECT T1.CharName FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T2.PlainText = ?", (plain_text,))
    result = cursor.fetchall()
    if not result:
        return {"char_names": []}
    return {"char_names": [row[0] for row in result]}

# Endpoint to get distinct acts from chapters of a specific work
@app.get("/v1/shakespeare/distinct_acts_in_work", operation_id="get_distinct_acts", summary="Retrieve a unique list of acts from a specific work, identified by its long title. This operation filters chapters associated with the work and extracts distinct acts, providing a comprehensive overview of the work's structure.")
async def get_distinct_acts(long_title: str = Query(..., description="Long title of the work")):
    cursor.execute("SELECT DISTINCT T1.Act FROM chapters AS T1 INNER JOIN works AS T2 ON T1.id = T1.work_id WHERE T2.LongTitle = ?", (long_title,))
    result = cursor.fetchall()
    if not result:
        return {"acts": []}
    return {"acts": [row[0] for row in result]}

# Endpoint to get the description of a character from a specific paragraph
@app.get("/v1/shakespeare/character_description_by_paragraph", operation_id="get_character_description", summary="Retrieve the description of a character associated with a given paragraph. The paragraph is identified by its unique ID, which is used to locate the corresponding character and return their description.")
async def get_character_description(paragraph_id: str = Query(..., description="ID of the paragraph")):
    cursor.execute("SELECT T1.Description FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T2.id = ?", (paragraph_id,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the count of scenes in a specific work
@app.get("/v1/shakespeare/count_scenes_in_work", operation_id="get_count_scenes", summary="Retrieves the total number of scenes in a specific work, identified by its title. The operation calculates this count by joining the 'works' and 'chapters' tables on the 'id' and 'work_id' fields, respectively, and filtering for the specified work title.")
async def get_count_scenes(title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT COUNT(T2.Scene) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct chapters featuring a specific character
@app.get("/v1/shakespeare/count_distinct_chapters_by_character", operation_id="get_count_distinct_chapters", summary="Retrieve the number of unique chapters in which a specified character appears. The character is identified by their name.")
async def get_count_distinct_chapters(char_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT COUNT(DISTINCT T2.chapter_id) FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T1.CharName = ?", (char_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the title of a work based on chapter ID and description
@app.get("/v1/shakespeare/work_title_by_chapter_id_and_description", operation_id="get_work_title", summary="Retrieves the title of a work associated with a specific chapter, identified by its unique ID and description. The operation filters the works based on the provided chapter ID and description, and returns the title of the corresponding work.")
async def get_work_title(chapter_id: str = Query(..., description="ID of the chapter"), description: str = Query(..., description="Description of the chapter")):
    cursor.execute("SELECT T1.Title FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T2.id = ? AND T2.Description = ?", (chapter_id, description))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the long title of the work with the most scenes in a specific genre
@app.get("/v1/shakespeare/most_scenes_in_genre", operation_id="get_most_scenes_in_genre", summary="Retrieves the long title of the work with the highest number of scenes in a specified genre. The genre type is provided as an input parameter.")
async def get_most_scenes_in_genre(genre_type: str = Query(..., description="Genre type of the work")):
    cursor.execute("SELECT T.LongTitle FROM ( SELECT T1.LongTitle, COUNT(T2.Scene) AS num FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.GenreType = ? GROUP BY T1.LongTitle, T2.Scene ) AS T ORDER BY T.num DESC LIMIT 1", (genre_type,))
    result = cursor.fetchone()
    if not result:
        return {"long_title": []}
    return {"long_title": result[0]}

# Endpoint to get the percentage of scenes in a specific genre type for a given date
@app.get("/v1/shakespeare/percentage_scenes_genre_date", operation_id="get_percentage_scenes_genre_date", summary="Retrieves the percentage of scenes in a specified genre type for a given date. The genre type and date are used to filter the scenes and calculate the percentage. The result provides insights into the distribution of scenes across genres for a particular date.")
async def get_percentage_scenes_genre_date(genre_type: str = Query(..., description="Genre type of the work"), date: str = Query(..., description="Date of the work in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.GenreType = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.Scene) FROM chapters AS T1 INNER JOIN works AS T2 ON T1.work_id = T2.id WHERE T2.Date = ?", (genre_type, date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of chapters to distinct works for a given date
@app.get("/v1/shakespeare/ratio_chapters_to_works", operation_id="get_ratio_chapters_to_works", summary="Retrieve the ratio of chapters to distinct works that were published on a specific date. The calculation is based on the count of chapters and the count of distinct works available in the database for the provided date.")
async def get_ratio_chapters_to_works(date: str = Query(..., description="Date of the work in 'YYYY' format")):
    cursor.execute("SELECT CAST(COUNT(T1.id) AS REAL) / COUNT(DISTINCT T2.id) FROM chapters AS T1 INNER JOIN works AS T2 ON T1.work_id = T2.id WHERE T2.Date = ?", (date,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of characters based on description
@app.get("/v1/shakespeare/count_characters_by_description", operation_id="get_count_characters_by_description", summary="Retrieves the total count of characters in the database that match the provided description. The description parameter is used to filter the characters and return the count of those that meet the specified criteria.")
async def get_count_characters_by_description(description: str = Query(..., description="Description of the character")):
    cursor.execute("SELECT COUNT(id) FROM characters WHERE Description = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the title of the earliest work
@app.get("/v1/shakespeare/earliest_work_title", operation_id="get_earliest_work_title", summary="Retrieves the title of the earliest work by Shakespeare. The operation identifies the work with the earliest date in the database and returns its title.")
async def get_earliest_work_title():
    cursor.execute("SELECT Title FROM works WHERE Date = ( SELECT MIN(Date) FROM works )")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of chapters based on work ID and act
@app.get("/v1/shakespeare/count_chapters_by_work_act", operation_id="get_count_chapters_by_work_act", summary="Retrieves the total number of chapters associated with a specific work and act. The work is identified by its unique ID, and the act is specified by its number. This operation provides a quantitative overview of the chapter distribution within a work's act.")
async def get_count_chapters_by_work_act(work_id: int = Query(..., description="ID of the work"), act: int = Query(..., description="Act number")):
    cursor.execute("SELECT COUNT(id) FROM chapters WHERE work_id = ? AND Act = ?", (work_id, act))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of works based on genre type and date range
@app.get("/v1/shakespeare/count_works_by_genre_date_range", operation_id="get_count_works_by_genre_date_range", summary="Retrieves the total number of works that fall within a specified genre and date range. The genre type and the start and end dates of the range are provided as input parameters. The result is a count of works that match the genre and date criteria.")
async def get_count_works_by_genre_date_range(genre_type: str = Query(..., description="Genre type of the work"), start_date: int = Query(..., description="Start date in 'YYYY' format"), end_date: int = Query(..., description="End date in 'YYYY' format")):
    cursor.execute("SELECT COUNT(id) FROM works WHERE GenreType = ? AND Date BETWEEN ? AND ?", (genre_type, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the character name based on description
@app.get("/v1/shakespeare/character_name_by_description", operation_id="get_character_name_by_description", summary="Retrieves the name of a character from the Shakespearean universe based on a provided description. The description is used to search for a matching character in the database and return the corresponding name.")
async def get_character_name_by_description(description: str = Query(..., description="Description of the character")):
    cursor.execute("SELECT CharName FROM characters WHERE Description = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"character_name": []}
    return {"character_name": result[0]}

# Endpoint to get the paragraph number based on plain text
@app.get("/v1/shakespeare/paragraph_number_by_plain_text", operation_id="get_paragraph_number_by_plain_text", summary="Retrieves the paragraph number associated with the provided plain text from the Shakespeare dataset. The input plain text is used to search for a matching paragraph in the dataset. The paragraph number of the first match is returned.")
async def get_paragraph_number_by_plain_text(plain_text: str = Query(..., description="Plain text of the paragraph")):
    cursor.execute("SELECT ParagraphNum FROM paragraphs WHERE PlainText = ?", (plain_text,))
    result = cursor.fetchone()
    if not result:
        return {"paragraph_number": []}
    return {"paragraph_number": result[0]}

# Endpoint to get the long title of the work with the highest scene number in a specific act
@app.get("/v1/shakespeare/long_title_by_act_highest_scene", operation_id="get_long_title_by_act_highest_scene", summary="Retrieves the full title of the work that has the highest scene number within a specified act. The act is identified by the provided act number.")
async def get_long_title_by_act_highest_scene(act: int = Query(..., description="Act number")):
    cursor.execute("SELECT T2.LongTitle FROM chapters AS T1 INNER JOIN works AS T2 ON T1.work_id = T2.id WHERE T1.Act = ? ORDER BY T1.Scene DESC LIMIT 1", (act,))
    result = cursor.fetchone()
    if not result:
        return {"long_title": []}
    return {"long_title": result[0]}

# Endpoint to get the description of the last paragraph in a chapter
@app.get("/v1/shakespeare/last_paragraph_description", operation_id="get_last_paragraph_description", summary="Retrieves the description of the last paragraph in a chapter. The operation identifies the chapter with the highest paragraph number and returns its corresponding description.")
async def get_last_paragraph_description():
    cursor.execute("SELECT T2.Description FROM paragraphs AS T1 INNER JOIN chapters AS T2 ON T1.chapter_id = T2.id ORDER BY T1.ParagraphNum DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get distinct chapter descriptions for paragraphs with a paragraph number less than a specified value
@app.get("/v1/shakespeare/distinct_chapter_descriptions", operation_id="get_distinct_chapter_descriptions", summary="Retrieve unique chapter descriptions for paragraphs with a paragraph number less than the provided threshold. This operation allows you to filter and obtain distinct chapter descriptions based on a specified paragraph number limit.")
async def get_distinct_chapter_descriptions(paragraph_num: int = Query(..., description="Paragraph number threshold")):
    cursor.execute("SELECT DISTINCT T2.Description FROM paragraphs AS T1 INNER JOIN chapters AS T2 ON T1.chapter_id = T2.id WHERE T1.ParagraphNum < ?", (paragraph_num,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the paragraph number of the last scene in a specified act
@app.get("/v1/shakespeare/last_paragraph_number_in_act", operation_id="get_last_paragraph_number_in_act", summary="Retrieves the paragraph number of the final scene in a given act. The act number is required as input to specify the act for which the last paragraph number is sought.")
async def get_last_paragraph_number_in_act(act: int = Query(..., description="Act number")):
    cursor.execute("SELECT T1.ParagraphNum FROM paragraphs AS T1 INNER JOIN chapters AS T2 ON T1.chapter_id = T2.id WHERE T2.Act = ? ORDER BY T2.Scene DESC LIMIT 1", (act,))
    result = cursor.fetchone()
    if not result:
        return {"paragraph_num": []}
    return {"paragraph_num": result[0]}

# Endpoint to get character names based on chapter description, character ID, and count of appearances
@app.get("/v1/shakespeare/character_names_by_description_id_count", operation_id="get_character_names", summary="Retrieves the names of characters that appear a specified number of times in a chapter with a given description and ID. The characters with stage directions are excluded from the results.")
async def get_character_names(description: str = Query(..., description="Chapter description"), chapter_id: int = Query(..., description="Chapter ID"), count: int = Query(..., description="Count of appearances")):
    cursor.execute("SELECT T.CharName FROM ( SELECT T3.CharName, COUNT(T3.id) AS num FROM paragraphs AS T1 INNER JOIN chapters AS T2 ON T1.chapter_id = T2.id INNER JOIN characters AS T3 ON T1.character_id = T3.id WHERE T2.Description = ? AND T3.CharName != '(stage directions)' AND T1.chapter_id = ? GROUP BY T3.id, T3.CharName ) AS T WHERE T.num = ?", (description, chapter_id, count))
    result = cursor.fetchall()
    if not result:
        return {"char_names": []}
    return {"char_names": [row[0] for row in result]}

# Endpoint to get the character ID based on plain text and chapter description
@app.get("/v1/shakespeare/character_id_by_plain_text_description", operation_id="get_character_id", summary="Retrieves the unique identifier of a character in the Shakespearean corpus based on a provided plain text and chapter description. The plain text and chapter description are used to locate the character's paragraph and chapter, respectively, and subsequently, the character's ID is returned.")
async def get_character_id(plain_text: str = Query(..., description="Plain text"), description: str = Query(..., description="Chapter description")):
    cursor.execute("SELECT T1.character_id FROM paragraphs AS T1 INNER JOIN chapters AS T2 ON T1.chapter_id = T2.id WHERE T1.PlainText = ? AND T2.Description = ?", (plain_text, description))
    result = cursor.fetchone()
    if not result:
        return {"character_id": []}
    return {"character_id": result[0]}

# Endpoint to get the sum of scenes in a specified act of a work
@app.get("/v1/shakespeare/sum_scenes_by_act_work", operation_id="get_sum_scenes", summary="Retrieves the total number of scenes in a specific act of a given work. The act is identified by its number, and the work is specified by its long title.")
async def get_sum_scenes(act: int = Query(..., description="Act number"), long_title: str = Query(..., description="Long title of the work")):
    cursor.execute("SELECT SUM(T2.Scene) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.Act = ? AND T1.LongTitle = ?", (act, long_title))
    result = cursor.fetchone()
    if not result:
        return {"sum_scenes": []}
    return {"sum_scenes": result[0]}

# Endpoint to get the count of distinct work IDs based on act, scene, and genre type
@app.get("/v1/shakespeare/count_distinct_work_ids", operation_id="get_count_distinct_work_ids", summary="Retrieves the total number of unique works that meet the specified act, scene, and genre criteria. The count is based on distinct work IDs from the 'chapters' table, filtered by the provided act and scene numbers, and the genre type from the 'works' table.")
async def get_count_distinct_work_ids(act: int = Query(..., description="Act number"), scene: int = Query(..., description="Scene number"), genre_type: str = Query(..., description="Genre type")):
    cursor.execute("SELECT COUNT(DISTINCT T2.work_id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.Act = ? AND T2.Scene < ? AND T1.GenreType = ?", (act, scene, genre_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of distinct acts based on work title
@app.get("/v1/shakespeare/sum_distinct_acts", operation_id="get_sum_distinct_acts", summary="Retrieves the total number of unique acts from a specified work. The work is identified by its title.")
async def get_sum_distinct_acts(title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT SUM(DISTINCT T2.Act) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct titles and genre types based on act
@app.get("/v1/shakespeare/distinct_titles_genre_types", operation_id="get_distinct_titles_genre_types", summary="Retrieve a unique list of play titles and their respective genre types, filtered by a specific act number.")
async def get_distinct_titles_genre_types(act: int = Query(..., description="Act number")):
    cursor.execute("SELECT DISTINCT T1.Title, T1.GenreType FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.Act = ?", (act,))
    result = cursor.fetchall()
    if not result:
        return {"titles_genre_types": []}
    return {"titles_genre_types": result}

# Endpoint to get the maximum paragraph number based on character name
@app.get("/v1/shakespeare/max_paragraph_number", operation_id="get_max_paragraph_number", summary="Retrieves the highest paragraph number associated with a specific character in the Shakespeare dataset. The character is identified by the provided name.")
async def get_max_paragraph_number(char_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT MAX(T2.ParagraphNum) FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T1.CharName = ?", (char_name,))
    result = cursor.fetchone()
    if not result:
        return {"max_paragraph_number": []}
    return {"max_paragraph_number": result[0]}

# Endpoint to get the description of a chapter based on act, scene, and work title
@app.get("/v1/shakespeare/chapter_description_by_act_scene_title", operation_id="get_chapter_description_by_act_scene_title", summary="Retrieves the description of a specific chapter from a work in the Shakespeare collection. The chapter is identified by its act and scene numbers, and the work is specified by its title. This operation provides detailed information about the selected chapter.")
async def get_chapter_description_by_act_scene_title(act: int = Query(..., description="Act number"), scene: int = Query(..., description="Scene number"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT T2.Description FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.Act = ? AND T2.Scene = ? AND T1.Title = ?", (act, scene, title))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the date and ID of works based on scene number
@app.get("/v1/shakespeare/work_date_id_by_scene", operation_id="get_work_date_id_by_scene", summary="Retrieves the date and unique identifier of works that correspond to a specified scene number. The operation filters works based on the provided scene number and returns the date and ID of the matching works.")
async def get_work_date_id_by_scene(scene: int = Query(..., description="Scene number")):
    cursor.execute("SELECT T1.Date, T1.id FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.Scene = ?", (scene,))
    result = cursor.fetchall()
    if not result:
        return {"date_id": []}
    return {"date_id": result}

# Endpoint to get the average scene number based on genre type
@app.get("/v1/shakespeare/average_scene_number", operation_id="get_average_scene_number", summary="Retrieves the average scene number for a given genre type. This operation calculates the sum of all scene numbers within the specified genre and divides it by the total number of works in that genre. The result provides a statistical overview of the average scene distribution across works of the selected genre.")
async def get_average_scene_number(genre_type: str = Query(..., description="Genre type")):
    cursor.execute("SELECT CAST(SUM(T2.Scene) AS REAL) / COUNT(T1.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.GenreType = ?", (genre_type,))
    result = cursor.fetchone()
    if not result:
        return {"average_scene_number": []}
    return {"average_scene_number": result[0]}

# Endpoint to get the count and percentage of specific characters in works
@app.get("/v1/shakespeare/character_count_percentage", operation_id="get_character_count_percentage", summary="Retrieves the total count and percentage of occurrences for two specified characters in all works. The operation calculates the sum of appearances for each character and the overall percentage of their combined occurrences relative to the total number of characters in all works.")
async def get_character_count_percentage(char_name_1: str = Query(..., description="First character name"), char_name_2: str = Query(..., description="Second character name")):
    cursor.execute("SELECT SUM(IIF(T4.CharName = ?, 1, 0)), SUM(IIF(T4.CharName = ?, 1, 0)), CAST(SUM(IIF(T4.CharName = ?, 1, 0)) + SUM(IIF(T4.CharName = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id", (char_name_1, char_name_2, char_name_1, char_name_2))
    result = cursor.fetchone()
    if not result:
        return {"character_count_percentage": []}
    return {"character_count_percentage": result}

# Endpoint to get the count of paragraphs in a specific chapter
@app.get("/v1/shakespeare/paragraph_count_by_chapter", operation_id="get_paragraph_count", summary="Retrieves the total number of paragraphs in a specified chapter. The chapter is identified by its unique ID.")
async def get_paragraph_count(chapter_id: int = Query(..., description="ID of the chapter")):
    cursor.execute("SELECT COUNT(ParagraphNum) FROM paragraphs WHERE chapter_id = ?", (chapter_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get titles of works based on genre type
@app.get("/v1/shakespeare/work_titles_by_genre", operation_id="get_work_titles", summary="Retrieves the titles of up to five works that belong to the specified genre type.")
async def get_work_titles(genre_type: str = Query(..., description="Genre type of the work")):
    cursor.execute("SELECT Title FROM works WHERE GenreType = ? LIMIT 5", (genre_type,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of scenes in a specific act of a work
@app.get("/v1/shakespeare/scene_count_by_work_and_act", operation_id="get_scene_count", summary="Retrieves the total number of scenes in a specific act of a given work. The operation requires the work's unique identifier and the act number as input parameters.")
async def get_scene_count(work_id: int = Query(..., description="ID of the work"), act: int = Query(..., description="Act number")):
    cursor.execute("SELECT COUNT(Scene) FROM chapters WHERE work_id = ? AND Act = ?", (work_id, act))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct character names and descriptions in a specific chapter
@app.get("/v1/shakespeare/characters_by_chapter", operation_id="get_characters_by_chapter", summary="Retrieve a unique list of character names and their descriptions that appear in a specified chapter. The chapter is identified by its unique ID.")
async def get_characters_by_chapter(chapter_id: int = Query(..., description="ID of the chapter")):
    cursor.execute("SELECT DISTINCT T1.CharName, T1.Description FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T2.Chapter_id = ?", (chapter_id,))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [{"name": row[0], "description": row[1]} for row in result]}

# Endpoint to get the count of chapters in a specific work
@app.get("/v1/shakespeare/chapter_count_by_work", operation_id="get_chapter_count_by_work", summary="Retrieves the total number of chapters in a specified work. The work is identified by its title, which is provided as an input parameter.")
async def get_chapter_count_by_work(title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT COUNT(T2.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of paragraphs in a specific act and scene of a work
@app.get("/v1/shakespeare/paragraph_count_by_act_scene_work", operation_id="get_paragraph_count_by_act_scene_work", summary="Retrieve the total number of paragraphs in a specific act and scene of a given work. This operation requires the act number, scene number, and the title of the work as input parameters. The result is a count of paragraphs that meet the specified criteria.")
async def get_paragraph_count_by_act_scene_work(act: int = Query(..., description="Act number"), scene: int = Query(..., description="Scene number"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT COUNT(T3.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id WHERE T2.Act = ? AND T2.Scene = ? AND T1.Title = ?", (act, scene, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct character names and descriptions in a specific work
@app.get("/v1/shakespeare/characters_by_work", operation_id="get_characters_by_work", summary="Retrieves a list of unique character names and their respective descriptions from a specified work. The operation filters the data based on the provided work title.")
async def get_characters_by_work(title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT DISTINCT T4.CharName, T2.Description FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T1.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [{"name": row[0], "description": row[1]} for row in result]}

# Endpoint to get distinct titles of works featuring a specific character
@app.get("/v1/shakespeare/works_by_character", operation_id="get_works_by_character", summary="Retrieve a unique list of titles for works in which a specified character appears. The character's name is used to filter the results.")
async def get_works_by_character(char_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT DISTINCT T1.title FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T4.CharName = ?", (char_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of chapters featuring a specific character
@app.get("/v1/shakespeare/chapter_count_by_character", operation_id="get_chapter_count_by_character", summary="Retrieve the total number of chapters in which a specified character appears. The character's name is used to filter the results.")
async def get_chapter_count_by_character(char_name: str = Query(..., description="Name of the character")):
    cursor.execute("SELECT COUNT(T2.chapter_id) FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T1.CharName = ?", (char_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get scenes and descriptions in a specific act of a work
@app.get("/v1/shakespeare/scenes_by_work_and_act", operation_id="get_scenes_by_work_and_act", summary="Retrieves scenes and their descriptions from a specific act of a given work. The work is identified by its long title, and the act is specified by its number. This operation provides a detailed view of the selected act, enabling users to explore its scenes and understand their context within the work.")
async def get_scenes_by_work_and_act(long_title: str = Query(..., description="Long title of the work"), act: int = Query(..., description="Act number")):
    cursor.execute("SELECT T2.Scene, T2.Description FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.LongTitle = ? AND T2.Act = ?", (long_title, act))
    result = cursor.fetchall()
    if not result:
        return {"scenes": []}
    return {"scenes": [{"scene": row[0], "description": row[1]} for row in result]}

# Endpoint to get distinct long titles of works based on character description
@app.get("/v1/shakespeare/distinct_long_titles_by_character_description", operation_id="get_distinct_long_titles", summary="Retrieve a unique list of long titles of works in which a character with the specified description appears. The operation filters works based on the provided character description and returns the distinct long titles of the matching works.")
async def get_distinct_long_titles(description: str = Query(..., description="Description of the character")):
    cursor.execute("SELECT DISTINCT T1.LongTitle FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T4.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"long_titles": []}
    return {"long_titles": [row[0] for row in result]}

# Endpoint to get act, scene, and title of works based on chapter description
@app.get("/v1/shakespeare/act_scene_title_by_chapter_description", operation_id="get_act_scene_title", summary="Retrieves the act, scene, and title of Shakespearean works that correspond to a given chapter description. The chapter description is used to filter the results, providing a targeted list of works and their respective act, scene, and title.")
async def get_act_scene_title(description: str = Query(..., description="Description of the chapter")):
    cursor.execute("SELECT T2.Act, T2.Scene, T1.Title FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"act_scene_title": []}
    return {"act_scene_title": [{"act": row[0], "scene": row[1], "title": row[2]} for row in result]}

# Endpoint to get character name, paragraph number, and plain text based on character description
@app.get("/v1/shakespeare/character_paragraph_by_description", operation_id="get_character_paragraph", summary="Retrieves the name of a character, the corresponding paragraph number, and the plain text of the paragraph based on the provided description of the character. This operation allows you to find specific character information by matching the input description with the character's description in the database.")
async def get_character_paragraph(description: str = Query(..., description="Description of the character")):
    cursor.execute("SELECT T1.CharName, T2.ParagraphNum, T2.PlainText FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T1.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"character_paragraph": []}
    return {"character_paragraph": [{"char_name": row[0], "paragraph_num": row[1], "plain_text": row[2]} for row in result]}

# Endpoint to get the percentage of paragraphs in a specific work title
@app.get("/v1/shakespeare/percentage_paragraphs_by_title", operation_id="get_percentage_paragraphs", summary="Retrieves the percentage of paragraphs that belong to a specific work title. This operation calculates the proportion of paragraphs in a given work title relative to the total number of paragraphs across all works. The work title is provided as an input parameter.")
async def get_percentage_paragraphs(title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.Title = ?, 1, 0)) AS REAL) * 100 / COUNT(T3.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id", (title,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_paragraphs": []}
    return {"percentage_paragraphs": result[0]}

# Endpoint to get the count of characters based on abbreviation
@app.get("/v1/shakespeare/count_characters_by_abbrev", operation_id="get_count_characters", summary="Retrieves the total number of characters in the Shakespeare database that match the provided abbreviation.")
async def get_count_characters(abbrev: str = Query(..., description="Abbreviation of the character")):
    cursor.execute("SELECT COUNT(id) FROM characters WHERE Abbrev = ?", (abbrev,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get titles of works based on genre type with a limit
@app.get("/v1/shakespeare/titles_by_genre_type", operation_id="get_titles_by_genre_type", summary="Retrieves a specified number of titles of works that belong to a particular genre. The genre type and the maximum number of results to return can be provided as input parameters.")
async def get_titles_by_genre_type(genre_type: str = Query(..., description="Genre type of the work"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Title FROM works WHERE GenreType = ? LIMIT ?", (genre_type, limit))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the difference in count between two genres after a specific date
@app.get("/v1/shakespeare/genre_count_difference", operation_id="get_genre_count_difference", summary="Retrieve the difference in the number of works between two specified genres that were published after a given year. The operation compares the total count of works for each genre and returns the difference.")
async def get_genre_count_difference(genre_type_1: str = Query(..., description="First genre type"), genre_type_2: str = Query(..., description="Second genre type"), date: int = Query(..., description="Date in YYYY format")):
    cursor.execute("SELECT SUM(IIF(GenreType = ?, 1, 0)) - SUM(IIF(GenreType = ?, 1, 0)) FROM works WHERE Date > ?", (genre_type_1, genre_type_2, date))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the most recent long title of a work based on genre type
@app.get("/v1/shakespeare/most_recent_long_title_by_genre", operation_id="get_most_recent_long_title", summary="Retrieves the most recent work's long title from the specified genre. The genre type is used to filter the results, and the work with the latest date is selected.")
async def get_most_recent_long_title(genre_type: str = Query(..., description="Genre type of the work")):
    cursor.execute("SELECT LongTitle FROM works WHERE GenreType = ? ORDER BY Date DESC LIMIT 1", (genre_type,))
    result = cursor.fetchone()
    if not result:
        return {"long_title": []}
    return {"long_title": result[0]}

# Endpoint to get work IDs based on a title pattern
@app.get("/v1/shakespeare/work_ids_by_title_pattern", operation_id="get_work_ids_by_title_pattern", summary="Retrieves the unique identifiers of works that match a specified title pattern. The title pattern is case-insensitive and supports the use of wildcard characters (%%) to broaden the search. The endpoint returns a list of work IDs that satisfy the provided title pattern.")
async def get_work_ids_by_title_pattern(title_pattern: str = Query(..., description="Pattern to match in the title (use %% for wildcard)")):
    cursor.execute("SELECT id FROM works WHERE Title LIKE ?", (title_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"work_ids": []}
    return {"work_ids": [row[0] for row in result]}

# Endpoint to get character names based on paragraph number and chapter ID
@app.get("/v1/shakespeare/characters/by_paragraph_chapter", operation_id="get_characters_by_paragraph_chapter", summary="Retrieve the names of characters who appear in a given paragraph within a specified chapter. The operation requires the paragraph number and chapter ID as input parameters to identify the relevant characters.")
async def get_characters_by_paragraph_chapter(paragraph_num: int = Query(..., description="Paragraph number"), chapter_id: int = Query(..., description="Chapter ID")):
    cursor.execute("SELECT T1.CharName FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T2.ParagraphNum = ? AND T2.chapter_id = ?", (paragraph_num, chapter_id))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get distinct descriptions based on character name and chapter ID
@app.get("/v1/shakespeare/descriptions/by_character_chapter", operation_id="get_descriptions_by_character_chapter", summary="Retrieves unique descriptions associated with a given character and chapter. The operation filters descriptions based on the provided character name and chapter ID, ensuring that only distinct descriptions are returned.")
async def get_descriptions_by_character_chapter(char_name: str = Query(..., description="Character name"), chapter_id: int = Query(..., description="Chapter ID")):
    cursor.execute("SELECT DISTINCT T3.Description FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id INNER JOIN chapters AS T3 ON T2.chapter_id = T3.id WHERE T1.CharName = ? AND T3.ID = ?", (char_name, chapter_id))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the count of scenes based on work title
@app.get("/v1/shakespeare/scenes/count_by_work_title", operation_id="get_scene_count_by_work_title", summary="Retrieves the total number of scenes associated with a specific work, identified by its long title. The operation returns a count of scenes that are part of the work, providing a quantitative measure of its content.")
async def get_scene_count_by_work_title(long_title: str = Query(..., description="Long title of the work")):
    cursor.execute("SELECT COUNT(T2.Scene) AS cnt FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.LongTitle = ?", (long_title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get character descriptions based on paragraph number
@app.get("/v1/shakespeare/characters/descriptions_by_paragraph", operation_id="get_descriptions_by_paragraph", summary="Retrieves the descriptions of characters that appear in a specified paragraph. The paragraph is identified by its unique number.")
async def get_descriptions_by_paragraph(paragraph_num: int = Query(..., description="Paragraph number")):
    cursor.execute("SELECT T1.Description FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T2.ParagraphNum = ?", (paragraph_num,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the count of distinct chapters based on character name and description
@app.get("/v1/shakespeare/chapters/count_by_character_description", operation_id="get_chapter_count_by_character_description", summary="Retrieves the number of unique chapters in which a character with a specific name and description appears. The character's name and description are used to filter the results.")
async def get_chapter_count_by_character_description(char_name: str = Query(..., description="Character name"), description: str = Query(..., description="Description of the character")):
    cursor.execute("SELECT COUNT(DISTINCT T2.chapter_id) FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T1.CharName = ? AND T1.Description = ?", (char_name, description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get chapter descriptions based on chapter ID and work title
@app.get("/v1/shakespeare/chapters/descriptions_by_chapter_work", operation_id="get_descriptions_by_chapter_work", summary="Retrieves the description of a specific chapter from a work in the Shakespeare collection. The chapter is identified by its unique ID, and the work is specified by its title. This operation returns the full description of the chapter.")
async def get_descriptions_by_chapter_work(chapter_id: int = Query(..., description="Chapter ID"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT T2.Description FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.id = ? AND T1.Title = ?", (chapter_id, title))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct character names based on paragraph number
@app.get("/v1/shakespeare/characters/distinct_by_paragraph", operation_id="get_distinct_characters_by_paragraph", summary="Retrieve a list of unique character names that appear in a specified paragraph. The operation filters the character list based on the provided paragraph number.")
async def get_distinct_characters_by_paragraph(paragraph_num: int = Query(..., description="Paragraph number")):
    cursor.execute("SELECT DISTINCT T1.CharName FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T2.ParagraphNum = ?", (paragraph_num,))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get paragraph numbers based on character name
@app.get("/v1/shakespeare/paragraphs/by_character", operation_id="get_paragraphs_by_character", summary="Retrieves the paragraph numbers associated with a given character from the Shakespearean corpus. The character's name is provided as an input parameter, and the operation returns a list of paragraph numbers in which the character appears.")
async def get_paragraphs_by_character(char_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT T2.ParagraphNum FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T1.CharName = ?", (char_name,))
    result = cursor.fetchall()
    if not result:
        return {"paragraphs": []}
    return {"paragraphs": [row[0] for row in result]}

# Endpoint to get character names based on chapter ID with a limit
@app.get("/v1/shakespeare/characters/by_chapter_with_limit", operation_id="get_characters_by_chapter_with_limit", summary="Retrieves a specified number of character names that appear in a given chapter. The operation filters characters based on their presence in a particular chapter and returns a limited set of results.")
async def get_characters_by_chapter_with_limit(chapter_id: int = Query(..., description="Chapter ID"), limit: int = Query(..., description="Limit on the number of results")):
    cursor.execute("SELECT T1.CharName FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T2.chapter_id = ? LIMIT ?", (chapter_id, limit))
    result = cursor.fetchall()
    if not result:
        return {"characters": []}
    return {"characters": [row[0] for row in result]}

# Endpoint to get the count of acts in chapters of a specific genre and title
@app.get("/v1/shakespeare/count_acts_by_genre_and_title", operation_id="get_count_acts_by_genre_and_title", summary="Retrieve the total number of acts in chapters of a specific work, filtered by genre and title. The genre and title are provided as input parameters to narrow down the results.")
async def get_count_acts_by_genre_and_title(genre_type: str = Query(..., description="Genre type of the work"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT COUNT(T1.ACT) FROM chapters AS T1 LEFT JOIN works AS T2 ON T1.work_id = T2.id WHERE T2.GenreType = ? AND T2.Title = ?", (genre_type, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of works with a specific genre and date range having a minimum number of scenes
@app.get("/v1/shakespeare/percentage_works_with_min_scenes", operation_id="get_percentage_works_with_min_scenes", summary="Retrieves the percentage of works within a specified genre and date range that contain at least a certain number of scenes. The genre type, start and end dates, and minimum scene count are provided as input parameters.")
async def get_percentage_works_with_min_scenes(genre_type: str = Query(..., description="Genre type of the work"), start_date: int = Query(..., description="Start date in YYYY format"), end_date: int = Query(..., description="End date in YYYY format"), min_scenes: int = Query(..., description="Minimum number of scenes")):
    cursor.execute("SELECT CAST(( SELECT COUNT(T1.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.GenreType = ? AND T1.DATE BETWEEN ? AND ? GROUP BY T1.id HAVING COUNT(T2.Scene) >= ? ) AS REAL) * 100 / COUNT(id) FROM works WHERE GenreType = ? AND DATE BETWEEN ? AND ?", (genre_type, start_date, end_date, min_scenes, genre_type, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of acts in a specific work
@app.get("/v1/shakespeare/percentage_acts_in_work", operation_id="get_percentage_acts_in_work", summary="Retrieves the percentage of a specific act within a given work. The calculation is based on the total number of chapters in the work. The act number and the title of the work are required as input parameters.")
async def get_percentage_acts_in_work(act_number: int = Query(..., description="Act number"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.act = ?, 1, 0)) AS REAL) * 100 / COUNT(T2.act) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.Title = ?", (act_number, title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of works with titles containing a specific substring
@app.get("/v1/shakespeare/count_works_by_title_substring", operation_id="get_count_works_by_title_substring", summary="Retrieves the total number of works whose titles contain a specified substring. The substring is case-insensitive and can appear anywhere within the title.")
async def get_count_works_by_title_substring(title_substring: str = Query(..., description="Substring to search in the title")):
    cursor.execute("SELECT COUNT(id) FROM works WHERE Title LIKE ?", ('%' + title_substring + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the character ID based on plain text
@app.get("/v1/shakespeare/get_character_id_by_plain_text", operation_id="get_character_id_by_plain_text", summary="Retrieves the unique identifier of the character associated with the provided plain text. The operation searches for the exact match of the input plain text in the paragraphs database and returns the corresponding character ID.")
async def get_character_id_by_plain_text(plain_text: str = Query(..., description="Plain text to search for")):
    cursor.execute("SELECT character_id FROM paragraphs WHERE PlainText = ?", (plain_text,))
    result = cursor.fetchone()
    if not result:
        return {"character_id": []}
    return {"character_id": result[0]}

# Endpoint to get paragraph details by character name
@app.get("/v1/shakespeare/get_paragraph_details_by_character_name", operation_id="get_paragraph_details_by_character_name", summary="Retrieves the paragraph details associated with a specific character in the Shakespeare dataset. The operation returns the paragraph number and its unique identifier, filtered by the provided character name.")
async def get_paragraph_details_by_character_name(char_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT T2.ParagraphNum, T2.id FROM characters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.character_id WHERE T1.CharName = ?", (char_name,))
    result = cursor.fetchall()
    if not result:
        return {"paragraph_details": []}
    return {"paragraph_details": result}

# Endpoint to get the latest work title and character name
@app.get("/v1/shakespeare/get_latest_work_and_character", operation_id="get_latest_work_and_character", summary="Retrieves the title of the most recent work and the name of the associated character. This operation fetches data from the 'works', 'chapters', 'paragraphs', and 'characters' tables, joining them based on their respective relationships. The result is ordered by the work's date in descending order, and only the top record is returned.")
async def get_latest_work_and_character():
    cursor.execute("SELECT T1.Title, T4.CharName FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id ORDER BY T1.Date DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"work_title": [], "character_name": []}
    return {"work_title": result[0], "character_name": result[1]}

# Endpoint to get the long title of works based on chapter description
@app.get("/v1/shakespeare/get_long_title_by_chapter_description", operation_id="get_long_title_by_chapter_description", summary="Retrieves the full title of works that contain a chapter with the specified description. The description is used to identify the relevant chapter, and the corresponding work's long title is returned.")
async def get_long_title_by_chapter_description(description: str = Query(..., description="Chapter description")):
    cursor.execute("SELECT T1.LongTitle FROM works AS T1 RIGHT JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T2.Description = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"long_title": []}
    return {"long_title": result[0]}

# Endpoint to get the count of distinct works based on date range and character name
@app.get("/v1/shakespeare/count_distinct_works_by_date_and_character", operation_id="get_count_distinct_works_by_date_and_character", summary="Retrieve the number of unique works associated with a specific character, filtered by a specified date range. The date range is defined by a start and end year, and the character is identified by name. This operation provides a quantitative measure of a character's presence across distinct works within the given time frame.")
async def get_count_distinct_works_by_date_and_character(start_date: int = Query(..., description="Start date (YYYY)"), end_date: int = Query(..., description="End date (YYYY)"), char_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.work_id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T1.DATE BETWEEN ? AND ? AND T4.CharName = ?", (start_date, end_date, char_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the description of the last scene of a work by title
@app.get("/v1/shakespeare/last_scene_description_by_title", operation_id="get_last_scene_description_by_title", summary="Retrieves the description of the final scene from a work, identified by its title. The operation searches for the work by title and returns the description of the scene with the highest scene number.")
async def get_last_scene_description_by_title(title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT T2.Description FROM works AS T1 RIGHT JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.Title = ? ORDER BY T2.Scene DESC LIMIT 1", (title,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the count of character appearances in a specific act, scene, and work
@app.get("/v1/shakespeare/count_character_appearances", operation_id="get_count_character_appearances", summary="Retrieves the total number of appearances made by a specific character in a given act, scene, and work. The character is identified by both its unique ID and name, while the work is specified by its title. This operation is useful for analyzing character prominence and involvement in the narrative.")
async def get_count_character_appearances(act: int = Query(..., description="Act number"), scene: int = Query(..., description="Scene number"), char_id: int = Query(..., description="Character ID"), char_name: str = Query(..., description="Character name"), title: str = Query(..., description="Title of the work")):
    cursor.execute("SELECT COUNT(T4.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T2.Act = ? AND T2.Scene = ? AND T4.id = ? AND T4.CharName = ? AND T1.Title = ?", (act, scene, char_id, char_name, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get character and work IDs based on specific text in paragraphs
@app.get("/v1/shakespeare/character_work_ids_by_text", operation_id="get_character_work_ids_by_text", summary="Retrieves the character and work IDs associated with a specific text in paragraphs. The provided text is used to search for matching paragraphs, and the corresponding character and work IDs are returned.")
async def get_character_work_ids_by_text(plain_text: str = Query(..., description="Text in paragraphs")):
    cursor.execute("SELECT T2.character_id, T1.work_id FROM chapters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.chapter_id WHERE T2.PlainText = ?", (plain_text,))
    result = cursor.fetchall()
    if not result:
        return {"character_work_ids": []}
    return {"character_work_ids": [{"character_id": row[0], "work_id": row[1]} for row in result]}

# Endpoint to get chapter IDs and descriptions based on specific text in paragraphs
@app.get("/v1/shakespeare/chapter_ids_descriptions_by_text", operation_id="get_chapter_ids_descriptions_by_text", summary="Retrieves chapter IDs and their corresponding descriptions that contain a specific text in their paragraphs. The provided text is used to filter the results.")
async def get_chapter_ids_descriptions_by_text(plain_text: str = Query(..., description="Text in paragraphs")):
    cursor.execute("SELECT T1.id, T1.Description FROM chapters AS T1 INNER JOIN paragraphs AS T2 ON T1.id = T2.chapter_id WHERE T2.PlainText = ?", (plain_text,))
    result = cursor.fetchall()
    if not result:
        return {"chapter_ids_descriptions": []}
    return {"chapter_ids_descriptions": [{"id": row[0], "description": row[1]} for row in result]}

# Endpoint to get distinct scenes based on work title and character name
@app.get("/v1/shakespeare/distinct_scenes_by_title_and_character", operation_id="get_distinct_scenes_by_title_and_character", summary="Retrieves a list of distinct scenes from a specific work, filtered by a given character's name. This operation allows you to identify unique scenes in which a particular character appears within a chosen work.")
async def get_distinct_scenes_by_title_and_character(title: str = Query(..., description="Title of the work"), char_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT DISTINCT T2.Scene FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T1.Title = ? AND T4.CharName = ?", (title, char_name))
    result = cursor.fetchall()
    if not result:
        return {"scenes": []}
    return {"scenes": [row[0] for row in result]}

# Endpoint to get distinct work titles based on date, genre type, and character name
@app.get("/v1/shakespeare/distinct_titles_by_date_genre_character", operation_id="get_distinct_titles_by_date_genre_character", summary="Retrieve a list of unique work titles that were published before a specified date, belong to a certain genre, and feature a particular character. The date, genre type, and character name are provided as input parameters.")
async def get_distinct_titles_by_date_genre_character(date: int = Query(..., description="Date (YYYY)"), genre_type: str = Query(..., description="Genre type"), char_name: str = Query(..., description="Character name")):
    cursor.execute("SELECT DISTINCT T1.title FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T1.DATE < ? AND T1.GenreType = ? AND T4.CharName = ?", (date, genre_type, char_name))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get chapter IDs based on a percentage of the average date of works
@app.get("/v1/shakespeare/chapter_ids_by_date_percentage", operation_id="get_chapter_ids_by_date_percentage", summary="Retrieves chapter IDs from works whose date is greater than a specified percentage of the average date of all works. The percentage is provided as an input parameter.")
async def get_chapter_ids_by_date_percentage(percentage: float = Query(..., description="Percentage of the average date")):
    cursor.execute("SELECT T2.id FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id WHERE T1.DATE > ( SELECT AVG(DATE) FROM works ) * ?", (percentage,))
    result = cursor.fetchall()
    if not result:
        return {"chapter_ids": []}
    return {"chapter_ids": [row[0] for row in result]}

# Endpoint to get the percentage of a character's appearances in a specific genre
@app.get("/v1/shakespeare/percentage_character_appearances_by_genre", operation_id="get_percentage_character_appearances_by_genre", summary="Retrieve the percentage of appearances made by a specific character in a given genre. This operation calculates the proportion of appearances by comparing the total count of the character's appearances in the specified genre to the overall count of character appearances in that genre.")
async def get_percentage_character_appearances_by_genre(char_name: str = Query(..., description="Character name"), genre_type: str = Query(..., description="Genre type")):
    cursor.execute("SELECT CAST(SUM(IIF(T4.CharName = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.id) FROM works AS T1 INNER JOIN chapters AS T2 ON T1.id = T2.work_id INNER JOIN paragraphs AS T3 ON T2.id = T3.chapter_id INNER JOIN characters AS T4 ON T3.character_id = T4.id WHERE T1.GenreType = ?", (char_name, genre_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/shakespeare/count_works_before_date?date=1602",
    "/v1/shakespeare/count_works_by_title_and_act?act=1&title=Twelfth%20Night",
    "/v1/shakespeare/get_chapter_description?title=Twelfth%20Night&act=1&scene=2",
    "/v1/shakespeare/get_act_difference?act1=1&act2=5&title=Twelfth%20Night",
    "/v1/shakespeare/get_titles_by_character?char_name=Lord%20Abergavenny",
    "/v1/shakespeare/get_characters_by_title?title=Twelfth%20Night",
    "/v1/shakespeare/get_sum_paragraph_numbers?act=1&scene=1&title=Twelfth%20Night",
    "/v1/shakespeare/get_paragraph_text?act=1&scene=1&title=Twelfth%20Night",
    "/v1/shakespeare/get_sum_paragraph_numbers_by_character?char_name=Lord%20Abergavenny",
    "/v1/shakespeare/get_paragraph_ids_by_character_description?description=son%20to%20Tamora",
    "/v1/shakespeare/count_chapters_by_description_and_title?description=OLIVIA\u2019S%20house.&title=Twelfth%20Night",
    "/v1/shakespeare/count_distinct_characters_in_work?title=Twelfth%20Night",
    "/v1/shakespeare/work_with_most_characters",
    "/v1/shakespeare/average_distinct_characters_per_work",
    "/v1/shakespeare/average_scenes_per_act?title=Twelfth%20Night",
    "/v1/shakespeare/count_works_by_genre?genre_type=Comedy",
    "/v1/shakespeare/earliest_work_date_by_genre?genre_type=Poem",
    "/v1/shakespeare/distinct_abbreviations_by_charname?char_name=Earl%20of%20Westmoreland",
    "/v1/shakespeare/chapter_description_with_highest_paragraph",
    "/v1/shakespeare/character_name_by_plaintext?plain_text=Would%20he%20do%20so,%20I'ld%20beg%20your%20precious%20mistress,Which%20he%20counts%20but%20a%20trifle.",
    "/v1/shakespeare/distinct_acts_in_work?long_title=Two%20Gentlemen%20of%20Verona",
    "/v1/shakespeare/character_description_by_paragraph?paragraph_id=640171",
    "/v1/shakespeare/count_scenes_in_work?title=King%20John",
    "/v1/shakespeare/count_distinct_chapters_by_character?char_name=Demetrius",
    "/v1/shakespeare/work_title_by_chapter_id_and_description?chapter_id=324&description=friend%20to%20Caesar",
    "/v1/shakespeare/most_scenes_in_genre?genre_type=Tragedy",
    "/v1/shakespeare/percentage_scenes_genre_date?genre_type=Tragedy&date=1594",
    "/v1/shakespeare/ratio_chapters_to_works?date=1599",
    "/v1/shakespeare/count_characters_by_description?description=servant%20to%20Timon",
    "/v1/shakespeare/earliest_work_title",
    "/v1/shakespeare/count_chapters_by_work_act?work_id=7&act=1",
    "/v1/shakespeare/count_works_by_genre_date_range?genre_type=Tragedy&start_date=1500&end_date=1599",
    "/v1/shakespeare/character_name_by_description?description=Daughter%20to%20Capulet",
    "/v1/shakespeare/paragraph_number_by_plain_text?plain_text=Ay%2C%20surely%2C%20mere%20the%20truth%3A%20I%20know%20his%20lady.",
    "/v1/shakespeare/long_title_by_act_highest_scene?act=1",
    "/v1/shakespeare/last_paragraph_description",
    "/v1/shakespeare/distinct_chapter_descriptions?paragraph_num=150",
    "/v1/shakespeare/last_paragraph_number_in_act?act=1",
    "/v1/shakespeare/character_names_by_description_id_count?description=The%20sea-coast.&chapter_id=18709&count=5",
    "/v1/shakespeare/character_id_by_plain_text_description?plain_text=His%20name,%20I%20pray%20you.&description=Florence.%20Without%20the%20walls.%20A%20tucket%20afar%20off.",
    "/v1/shakespeare/sum_scenes_by_act_work?act=5&long_title=History%20of%20Henry%20VIII",
    "/v1/shakespeare/count_distinct_work_ids?act=1&scene=2&genre_type=History",
    "/v1/shakespeare/sum_distinct_acts?title=Sonnets",
    "/v1/shakespeare/distinct_titles_genre_types?act=1",
    "/v1/shakespeare/max_paragraph_number?char_name=Sir%20Richard%20Ratcliff",
    "/v1/shakespeare/chapter_description_by_act_scene_title?act=1&scene=1&title=A%20Lover's%20Complaint",
    "/v1/shakespeare/work_date_id_by_scene?scene=154",
    "/v1/shakespeare/average_scene_number?genre_type=Comedy",
    "/v1/shakespeare/character_count_percentage?char_name_1=Romeo&char_name_2=Juliet",
    "/v1/shakespeare/paragraph_count_by_chapter?chapter_id=18881",
    "/v1/shakespeare/work_titles_by_genre?genre_type=History",
    "/v1/shakespeare/scene_count_by_work_and_act?work_id=9&act=5",
    "/v1/shakespeare/characters_by_chapter?chapter_id=18710",
    "/v1/shakespeare/chapter_count_by_work?title=Midsummer%20Night's%20Dream",
    "/v1/shakespeare/paragraph_count_by_act_scene_work?act=5&scene=1&title=Comedy%20of%20Errors",
    "/v1/shakespeare/characters_by_work?title=Venus%20and%20Adonis",
    "/v1/shakespeare/works_by_character?char_name=Froth",
    "/v1/shakespeare/chapter_count_by_character?char_name=First%20Witch",
    "/v1/shakespeare/scenes_by_work_and_act?long_title=Pericles,%20Prince%20of%20Tyre&act=1",
    "/v1/shakespeare/distinct_long_titles_by_character_description?description=Servant%20to%20Montague",
    "/v1/shakespeare/act_scene_title_by_chapter_description?description=The%20house%20of%20ANTIPHOLUS%20of%20Ephesus.",
    "/v1/shakespeare/character_paragraph_by_description?description=cousin%20to%20the%20king",
    "/v1/shakespeare/percentage_paragraphs_by_title?title=All's%20Well%20That%20Ends%20Well",
    "/v1/shakespeare/count_characters_by_abbrev?abbrev=All",
    "/v1/shakespeare/titles_by_genre_type?genre_type=comedy&limit=3",
    "/v1/shakespeare/genre_count_difference?genre_type_1=Comedy&genre_type_2=History&date=1593",
    "/v1/shakespeare/most_recent_long_title_by_genre?genre_type=History",
    "/v1/shakespeare/work_ids_by_title_pattern?title_pattern=%25Henry%25",
    "/v1/shakespeare/characters/by_paragraph_chapter?paragraph_num=8&chapter_id=18820",
    "/v1/shakespeare/descriptions/by_character_chapter?char_name=Orsino&chapter_id=18704",
    "/v1/shakespeare/scenes/count_by_work_title?long_title=Cymbeline%2C%20King%20of%20Britain",
    "/v1/shakespeare/characters/descriptions_by_paragraph?paragraph_num=20",
    "/v1/shakespeare/chapters/count_by_character_description?char_name=Gratiano&description=friend%20to%20Antonio%20and%20Bassiano",
    "/v1/shakespeare/chapters/descriptions_by_chapter_work?chapter_id=18706&title=All%27s%20Well%20That%20Ends%20Well",
    "/v1/shakespeare/characters/distinct_by_paragraph?paragraph_num=3",
    "/v1/shakespeare/paragraphs/by_character?char_name=Aedile",
    "/v1/shakespeare/characters/by_chapter_with_limit?chapter_id=18708&limit=2",
    "/v1/shakespeare/count_acts_by_genre_and_title?genre_type=Comedy&title=Two%20Gentlemen%20of%20Verona",
    "/v1/shakespeare/percentage_works_with_min_scenes?genre_type=History&start_date=1500&end_date=1599&min_scenes=5",
    "/v1/shakespeare/percentage_acts_in_work?act_number=5&title=Titus%20Andronicus",
    "/v1/shakespeare/count_works_by_title_substring?title_substring=Henry",
    "/v1/shakespeare/get_character_id_by_plain_text?plain_text=O%20my%20poor%20brother!%20and%20so%20perchance%20may%20he%20be.",
    "/v1/shakespeare/get_paragraph_details_by_character_name?char_name=Sir%20Andrew%20Aguecheek",
    "/v1/shakespeare/get_latest_work_and_character",
    "/v1/shakespeare/get_long_title_by_chapter_description?description=Mytilene.%20A%20street%20before%20the%20brothel.",
    "/v1/shakespeare/count_distinct_works_by_date_and_character?start_date=1600&end_date=1610&char_name=Third%20Servingman",
    "/v1/shakespeare/last_scene_description_by_title?title=Venus%20and%20Adonis",
    "/v1/shakespeare/count_character_appearances?act=1&scene=2&char_id=1238&char_name=Viola&title=Twelfth%20Night",
    "/v1/shakespeare/character_work_ids_by_text?plain_text=Fear%20not%20thou,%20man,%20thou%20shalt%20lose%20nothing%20here.",
    "/v1/shakespeare/chapter_ids_descriptions_by_text?plain_text=What,%20wilt%20thou%20hear%20some%20music,%20my%20sweet%20love?",
    "/v1/shakespeare/distinct_scenes_by_title_and_character?title=Twelfth%20Night&char_name=Sir%20Toby%20Belch",
    "/v1/shakespeare/distinct_titles_by_date_genre_character?date=1600&genre_type=Tragedy&char_name=Tybalt",
    "/v1/shakespeare/chapter_ids_by_date_percentage?percentage=0.89",
    "/v1/shakespeare/percentage_character_appearances_by_genre?char_name=antonio&genre_type=Comedy"
]
