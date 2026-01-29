from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/citeseer/citeseer.sqlite')
cursor = conn.cursor()

# Endpoint to get the percentage of papers with a specific class label
@app.get("/v1/citeseer/percentage_papers_by_class_label", operation_id="get_percentage_papers_by_class_label", summary="Retrieves the percentage of papers that belong to a specified class label. The calculation is based on the total count of papers with the given class label divided by the total count of all papers in the database.")
async def get_percentage_papers_by_class_label(class_label: str = Query(..., description="Class label of the paper")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN class_label = ? THEN paper_id ELSE NULL END) AS REAL) * 100 / COUNT(paper_id) FROM paper", (class_label,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the most frequently cited word
@app.get("/v1/citeseer/most_frequently_cited_word", operation_id="get_most_frequently_cited_word", summary="Retrieves the most frequently cited word and its count from the content database, sorted in descending order. The number of results returned can be limited by specifying the desired limit.")
async def get_most_frequently_cited_word(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT word_cited_id, COUNT(paper_id) FROM content GROUP BY word_cited_id ORDER BY COUNT(word_cited_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"word_cited_id": [], "count": []}
    return {"word_cited_id": result[0], "count": result[1]}

# Endpoint to get the count of distinct cited words for a specific class label
@app.get("/v1/citeseer/count_distinct_cited_words_by_class_label", operation_id="get_count_distinct_cited_words_by_class_label", summary="Retrieves the count of unique words cited in papers belonging to a specific class label. The class label is used to filter the papers and count the distinct cited words.")
async def get_count_distinct_cited_words_by_class_label(class_label: str = Query(..., description="Class label of the paper")):
    cursor.execute("SELECT COUNT(DISTINCT T2.word_cited_id) FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id WHERE T1.class_label = ?", (class_label,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get class label and cited word for a specific paper ID
@app.get("/v1/citeseer/class_label_and_cited_word_by_paper_id", operation_id="get_class_label_and_cited_word_by_paper_id", summary="Retrieves the class label and the ID of the cited word associated with a specific paper. The paper is identified by its unique paper_id.")
async def get_class_label_and_cited_word_by_paper_id(paper_id: int = Query(..., description="Paper ID")):
    cursor.execute("SELECT T1.class_label, T2.word_cited_id FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id WHERE T1.paper_id = ?", (paper_id,))
    result = cursor.fetchall()
    if not result:
        return {"class_label": [], "word_cited_id": []}
    return {"class_label": [row[0] for row in result], "word_cited_id": [row[1] for row in result]}

# Endpoint to get the most frequently cited word for a specific class label
@app.get("/v1/citeseer/most_frequently_cited_word_by_class_label", operation_id="get_most_frequently_cited_word_by_class_label", summary="Retrieves the most frequently cited word in papers belonging to a specific class label. The class label is used to filter the papers, and the results are ordered by the frequency of the cited word. The number of results returned can be limited by specifying the desired limit.")
async def get_most_frequently_cited_word_by_class_label(class_label: str = Query(..., description="Class label of the paper"), limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T2.word_cited_id FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id WHERE T1.class_label = ? GROUP BY T2.word_cited_id ORDER BY COUNT(T2.word_cited_id) DESC LIMIT ?", (class_label, limit))
    result = cursor.fetchone()
    if not result:
        return {"word_cited_id": []}
    return {"word_cited_id": result[0]}

# Endpoint to get the percentage of distinct cited words for a specific class label
@app.get("/v1/citeseer/percentage_distinct_cited_words_by_class_label", operation_id="get_percentage_distinct_cited_words_by_class_label", summary="Retrieves the percentage of unique cited words that belong to a specific class label. This operation calculates the ratio of distinct cited words associated with the given class label to the total distinct cited words across all papers. The class label is a categorical attribute of the paper.")
async def get_percentage_distinct_cited_words_by_class_label(class_label: str = Query(..., description="Class label of the paper")):
    cursor.execute("SELECT CAST(COUNT(DISTINCT CASE WHEN T1.class_label = ? THEN T2.word_cited_id ELSE NULL END) AS REAL) * 100 / COUNT(DISTINCT T2.word_cited_id) FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id", (class_label,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the paper ID and class label with the highest number of cited words
@app.get("/v1/citeseer/paper_with_highest_cited_words", operation_id="get_paper_with_highest_cited_words", summary="Retrieves the paper ID and its corresponding class label that have the highest number of cited words. The operation returns a limited number of results based on the provided limit parameter. The results are ordered in descending order based on the count of cited words.")
async def get_paper_with_highest_cited_words(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T1.paper_id, T1.class_label FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id GROUP BY T1.paper_id, T1.class_label ORDER BY COUNT(T2.word_cited_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"paper_id": [], "class_label": []}
    return {"paper_id": result[0], "class_label": result[1]}

# Endpoint to get the paper ID and class label for a specific cited word
@app.get("/v1/citeseer/paper_and_class_label_by_cited_word", operation_id="get_paper_and_class_label_by_cited_word", summary="Retrieves the ID and class label of a paper that contains a specific cited word. The operation uses the provided cited word ID to search for the paper and its associated class label.")
async def get_paper_and_class_label_by_cited_word(word_cited_id: str = Query(..., description="Cited word ID")):
    cursor.execute("SELECT T1.paper_id, T1.class_label FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id WHERE T2.word_cited_id = ?", (word_cited_id,))
    result = cursor.fetchall()
    if not result:
        return {"paper_id": [], "class_label": []}
    return {"paper_id": [row[0] for row in result], "class_label": [row[1] for row in result]}

# Endpoint to get distinct cited words for a specific class label
@app.get("/v1/citeseer/distinct_cited_words_by_class_label", operation_id="get_distinct_cited_words_by_class_label", summary="Retrieves a unique set of words cited in papers belonging to a specific class label. The class label is used to filter the papers and extract the distinct words cited within them.")
async def get_distinct_cited_words_by_class_label(class_label: str = Query(..., description="Class label of the paper")):
    cursor.execute("SELECT DISTINCT T2.word_cited_id FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id WHERE T1.class_label = ?", (class_label,))
    result = cursor.fetchall()
    if not result:
        return {"word_cited_id": []}
    return {"word_cited_id": [row[0] for row in result]}

# Endpoint to get the count of cited words for each class label for a specific paper ID
@app.get("/v1/citeseer/count_cited_words_by_class_label_and_paper_id", operation_id="get_count_cited_words_by_class_label_and_paper_id", summary="Retrieves the total count of distinct words cited in a paper, categorized by their respective class labels. The operation requires a specific paper ID to filter the results.")
async def get_count_cited_words_by_class_label_and_paper_id(paper_id: str = Query(..., description="Paper ID")):
    cursor.execute("SELECT DISTINCT T1.class_label, COUNT(T2.word_cited_id) FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id WHERE T1.paper_id = ? GROUP BY T1.class_label", (paper_id,))
    result = cursor.fetchall()
    if not result:
        return {"class_label": [], "count": []}
    return {"class_label": [row[0] for row in result], "count": [row[1] for row in result]}

# Endpoint to get distinct paper IDs and class labels based on the count of word cited IDs
@app.get("/v1/citeseer/distinct_paper_class_label_by_word_cited_count", operation_id="get_distinct_paper_class_label", summary="Retrieve unique paper identifiers and their corresponding class labels for papers that have a word cited count surpassing a specified threshold. This operation groups papers by their identifiers and class labels, and only includes those with a word cited count exceeding the provided threshold.")
async def get_distinct_paper_class_label(word_cited_count: int = Query(..., description="Threshold count of word cited IDs")):
    cursor.execute("SELECT DISTINCT T1.paper_id, T1.class_label FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id GROUP BY T2.paper_id, T1.class_label HAVING COUNT(T2.word_cited_id) > ?", (word_cited_count,))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": result}

# Endpoint to get distinct word cited IDs based on class labels
@app.get("/v1/citeseer/distinct_word_cited_ids_by_class_labels", operation_id="get_distinct_word_cited_ids", summary="Retrieves a unique set of word cited IDs for papers that belong to either of the two specified class labels. This operation is useful for identifying distinct references across papers with particular classifications.")
async def get_distinct_word_cited_ids(class_label_1: str = Query(..., description="First class label"), class_label_2: str = Query(..., description="Second class label")):
    cursor.execute("SELECT DISTINCT T2.word_cited_id FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id WHERE T1.class_label = ? OR T1.class_label = ?", (class_label_1, class_label_2))
    result = cursor.fetchall()
    if not result:
        return {"word_cited_ids": []}
    return {"word_cited_ids": result}

# Endpoint to get the most cited paper and its citation count
@app.get("/v1/citeseer/most_cited_paper", operation_id="get_most_cited_paper", summary="Retrieves the paper with the highest citation count and the total number of citations it has received. This operation also provides the paper with the least citations and its citation count for comparison.")
async def get_most_cited_paper():
    cursor.execute("SELECT cited_paper_id, COUNT(cited_paper_id), ( SELECT cited_paper_id FROM cites GROUP BY cited_paper_id ORDER BY COUNT(cited_paper_id) ASC LIMIT 1 ), ( SELECT COUNT(cited_paper_id) FROM cites GROUP BY cited_paper_id ORDER BY COUNT(cited_paper_id) ASC LIMIT 1 ) FROM cites GROUP BY cited_paper_id ORDER BY COUNT(cited_paper_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"most_cited_paper": []}
    return {"most_cited_paper": result}

# Endpoint to get the proportion of papers with a specific class label
@app.get("/v1/citeseer/proportion_of_papers_by_class_label", operation_id="get_proportion_of_papers", summary="Retrieves the proportion of papers that belong to a specific class label. The class label is provided as an input parameter, and the operation calculates the ratio of papers with this label to the total number of papers in the database.")
async def get_proportion_of_papers(class_label: str = Query(..., description="Class label of the papers")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN class_label = ? THEN paper_id ELSE NULL END) AS REAL) / COUNT(paper_id) FROM paper", (class_label,))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

# Endpoint to get distinct word cited IDs based on citing paper ID
@app.get("/v1/citeseer/distinct_word_cited_ids_by_citing_paper", operation_id="get_distinct_word_cited_ids_by_citing_paper", summary="Retrieves a unique set of word cited IDs associated with a specific citing paper. This operation filters the 'cites' table based on the provided citing paper ID and joins it with the 'content' table to extract the distinct word cited IDs.")
async def get_distinct_word_cited_ids_by_citing_paper(citing_paper_id: str = Query(..., description="Citing paper ID")):
    cursor.execute("SELECT DISTINCT T2.word_cited_id FROM cites AS T1 INNER JOIN content AS T2 ON T1.cited_paper_id = T2.paper_id WHERE T1.citing_paper_id = ?", (citing_paper_id,))
    result = cursor.fetchall()
    if not result:
        return {"word_cited_ids": []}
    return {"word_cited_ids": result}

# Endpoint to get the count of papers based on citing paper ID and word cited ID
@app.get("/v1/citeseer/count_papers_by_citing_paper_and_word_cited", operation_id="get_count_papers_by_citing_paper_and_word_cited", summary="Retrieves the total number of papers that cite a specific paper and contain a particular word. The operation requires the ID of the citing paper and the ID of the word cited.")
async def get_count_papers_by_citing_paper_and_word_cited(citing_paper_id: str = Query(..., description="Citing paper ID"), word_cited_id: str = Query(..., description="Word cited ID")):
    cursor.execute("SELECT COUNT(T2.paper_id) FROM cites AS T1 INNER JOIN content AS T2 ON T1.cited_paper_id = T2.paper_id WHERE T1.citing_paper_id = ? AND T2.word_cited_id = ?", (citing_paper_id, word_cited_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct class labels based on word cited ID
@app.get("/v1/citeseer/distinct_class_labels_by_word_cited", operation_id="get_distinct_class_labels_by_word_cited", summary="Retrieves the unique class labels associated with papers that cite a specific word. The operation filters papers based on the provided word cited ID and returns the distinct class labels linked to those papers.")
async def get_distinct_class_labels_by_word_cited(word_cited_id: str = Query(..., description="Word cited ID")):
    cursor.execute("SELECT DISTINCT T1.class_label FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id WHERE T2.word_cited_id = ?", (word_cited_id,))
    result = cursor.fetchall()
    if not result:
        return {"class_labels": []}
    return {"class_labels": result}

# Endpoint to get the paper ID with the highest count of word cited IDs for a specific class label
@app.get("/v1/citeseer/top_paper_by_class_label", operation_id="get_top_paper_by_class_label", summary="Retrieves the paper with the highest frequency of cited words within a specified class label. The class label is used to filter the papers, and the paper with the most cited words is returned.")
async def get_top_paper_by_class_label(class_label: str = Query(..., description="Class label of the papers")):
    cursor.execute("SELECT T1.paper_id FROM paper AS T1 INNER JOIN content AS T2 ON T1.paper_id = T2.paper_id WHERE T1.class_label = ? GROUP BY T1.paper_id ORDER BY COUNT(T2.word_cited_id) DESC LIMIT 1", (class_label,))
    result = cursor.fetchone()
    if not result:
        return {"paper_id": []}
    return {"paper_id": result[0]}

# Endpoint to get the count of papers based on class label and cited paper ID
@app.get("/v1/citeseer/count_papers_by_class_label_and_cited_paper", operation_id="get_count_papers_by_class_label_and_cited_paper", summary="Retrieves the total number of papers that belong to a specific class label and cite a particular paper. The operation requires the class label and the ID of the cited paper as input parameters.")
async def get_count_papers_by_class_label_and_cited_paper(class_label: str = Query(..., description="Class label of the papers"), cited_paper_id: str = Query(..., description="Cited paper ID")):
    cursor.execute("SELECT COUNT(T1.paper_id) FROM paper AS T1 INNER JOIN cites AS T2 ON T1.paper_id = T2.citing_paper_id WHERE T1.class_label = ? AND T2.cited_paper_id = ?", (class_label, cited_paper_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

api_calls = [
    "/v1/citeseer/percentage_papers_by_class_label?class_label=Agents",
    "/v1/citeseer/most_frequently_cited_word?limit=1",
    "/v1/citeseer/count_distinct_cited_words_by_class_label?class_label=AI",
    "/v1/citeseer/class_label_and_cited_word_by_paper_id?paper_id=315017",
    "/v1/citeseer/most_frequently_cited_word_by_class_label?class_label=DB&limit=1",
    "/v1/citeseer/percentage_distinct_cited_words_by_class_label?class_label=Agents",
    "/v1/citeseer/paper_with_highest_cited_words?limit=1",
    "/v1/citeseer/paper_and_class_label_by_cited_word?word_cited_id=word1002",
    "/v1/citeseer/distinct_cited_words_by_class_label?class_label=AI",
    "/v1/citeseer/count_cited_words_by_class_label_and_paper_id?paper_id=chakrabarti01integrating",
    "/v1/citeseer/distinct_paper_class_label_by_word_cited_count?word_cited_count=20",
    "/v1/citeseer/distinct_word_cited_ids_by_class_labels?class_label_1=AI&class_label_2=IR",
    "/v1/citeseer/most_cited_paper",
    "/v1/citeseer/proportion_of_papers_by_class_label?class_label=ML",
    "/v1/citeseer/distinct_word_cited_ids_by_citing_paper?citing_paper_id=sima01computational",
    "/v1/citeseer/count_papers_by_citing_paper_and_word_cited?citing_paper_id=schmidt99advanced&word_cited_id=word3555",
    "/v1/citeseer/distinct_class_labels_by_word_cited?word_cited_id=word1163",
    "/v1/citeseer/top_paper_by_class_label?class_label=DB",
    "/v1/citeseer/count_papers_by_class_label_and_cited_paper?class_label=ML&cited_paper_id=butz01algorithmic"
]
