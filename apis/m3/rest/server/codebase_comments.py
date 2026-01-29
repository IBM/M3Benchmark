from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/codebase_comments/codebase_comments.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of methods based on language and comment format
@app.get("/v1/codebase_comments/method_count_lang_comment_is_xml", operation_id="get_method_count", summary="Retrieves the total count of methods that match a specified programming language and comment format. The comment format is indicated by a boolean value, where 1 signifies XML and 0 indicates non-XML.")
async def get_method_count(lang: str = Query(..., description="Language of the method"), comment_is_xml: int = Query(..., description="Whether the comment is in XML format (1 for true, 0 for false)")):
    cursor.execute("SELECT COUNT(Lang) FROM Method WHERE Lang = ? AND CommentIsXml = ?", (lang, comment_is_xml))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the task from method name based on tokenized name
@app.get("/v1/codebase_comments/method_task_from_name", operation_id="get_method_task", summary="Retrieves the task associated with a method, based on its tokenized name. The method's name is parsed to extract the task, which is then returned.")
async def get_method_task(name_tokenized: str = Query(..., description="Tokenized name of the method")):
    cursor.execute("SELECT SUBSTR(SUBSTR(Name, INSTR(Name, '.') + 1), 1, INSTR(SUBSTR(Name, INSTR(Name, '.') + 1), '.') - 1) task FROM Method WHERE NameTokenized = ?", (name_tokenized,))
    result = cursor.fetchall()
    if not result:
        return {"tasks": []}
    return {"tasks": [row[0] for row in result]}

# Endpoint to get distinct sampled times and solution IDs for the latest sampled time
@app.get("/v1/codebase_comments/latest_sampled_methods", operation_id="get_latest_sampled_methods", summary="Retrieves the distinct timestamps and corresponding solution IDs for the most recent sampling of methods. This operation provides a snapshot of the latest sampled methods, offering insights into the most recent method sampling activities.")
async def get_latest_sampled_methods():
    cursor.execute("SELECT DISTINCT SampledAt, SolutionId FROM Method WHERE SampledAt = ( SELECT MAX(SampledAt) FROM Method )")
    result = cursor.fetchall()
    if not result:
        return {"methods": []}
    return {"methods": [{"SampledAt": row[0], "SolutionId": row[1]} for row in result]}

# Endpoint to get the repository with the maximum number of forks
@app.get("/v1/codebase_comments/max_forks_repo", operation_id="get_max_forks_repo", summary="Retrieves the repository with the highest number of forks. The response includes the number of forks and the URL of the repository.")
async def get_max_forks_repo():
    cursor.execute("SELECT Forks, Url FROM Repo WHERE Forks = ( SELECT MAX(Forks) FROM Repo )")
    result = cursor.fetchall()
    if not result:
        return {"repos": []}
    return {"repos": [{"Forks": row[0], "Url": row[1]} for row in result]}

# Endpoint to get the repository ID with the most solutions
@app.get("/v1/codebase_comments/repo_with_most_solutions", operation_id="get_repo_with_most_solutions", summary="Retrieves the unique identifier of the repository that contains the highest number of solutions. The operation ranks repositories based on the count of distinct solution paths and returns the top-ranked repository.")
async def get_repo_with_most_solutions():
    cursor.execute("SELECT RepoId FROM solution GROUP BY RepoId ORDER BY COUNT(Path) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"RepoId": []}
    return {"RepoId": result[0]}

# Endpoint to get the repository with the maximum number of stars
@app.get("/v1/codebase_comments/max_stars_repo", operation_id="get_max_stars_repo", summary="Retrieves the URL and star count of the repository with the highest number of stars. This operation identifies the repository with the maximum star count and returns its URL and star count.")
async def get_max_stars_repo():
    cursor.execute("SELECT Url, Stars FROM Repo WHERE Stars = ( SELECT MAX(Stars) FROM Repo )")
    result = cursor.fetchall()
    if not result:
        return {"repos": []}
    return {"repos": [{"Url": row[0], "Stars": row[1]} for row in result]}

# Endpoint to get the solution paths for the latest processed time
@app.get("/v1/codebase_comments/latest_processed_solutions", operation_id="get_latest_processed_solutions", summary="Retrieves the paths of the most recently processed solutions. This operation returns the paths of the solutions that were processed at the latest time available in the database.")
async def get_latest_processed_solutions():
    cursor.execute("SELECT Path FROM Solution WHERE ProcessedTime = ( SELECT MAX(ProcessedTime) FROM Solution )")
    result = cursor.fetchall()
    if not result:
        return {"paths": []}
    return {"paths": [row[0] for row in result]}

# Endpoint to get the processed time for the repository with the maximum number of watchers
@app.get("/v1/codebase_comments/max_watchers_repo_processed_time", operation_id="get_max_watchers_repo_processed_time", summary="Retrieves the most recent processing time for the repository with the highest number of watchers. This operation provides insight into the activity level of the most popular repository.")
async def get_max_watchers_repo_processed_time():
    cursor.execute("SELECT ProcessedTime FROM Repo WHERE Watchers = ( SELECT MAX(Watchers) FROM Repo )")
    result = cursor.fetchall()
    if not result:
        return {"processed_times": []}
    return {"processed_times": [row[0] for row in result]}

# Endpoint to get the repository URL based on solution path
@app.get("/v1/codebase_comments/repo_url_by_solution_path", operation_id="get_repo_url_by_solution_path", summary="Retrieves the URL of the repository associated with the specified solution path. The solution path is used to identify the corresponding repository, enabling the retrieval of its URL.")
async def get_repo_url_by_solution_path(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT Url FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE Path = ?", (path,))
    result = cursor.fetchall()
    if not result:
        return {"urls": []}
    return {"urls": [row[0] for row in result]}

# Endpoint to get distinct repository IDs and compilation status for the latest processed time
@app.get("/v1/codebase_comments/latest_processed_repos_compilation_status", operation_id="get_latest_processed_repos_compilation_status", summary="Retrieves a list of unique repository IDs along with their respective compilation statuses, based on the most recent processing time. This operation provides an overview of the latest processed repositories and their compilation outcomes.")
async def get_latest_processed_repos_compilation_status():
    cursor.execute("SELECT DISTINCT T1.id, T2.WasCompiled FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.ProcessedTime = ( SELECT MAX(ProcessedTime) FROM Repo )")
    result = cursor.fetchall()
    if not result:
        return {"repos": []}
    return {"repos": [{"id": row[0], "WasCompiled": row[1]} for row in result]}

# Endpoint to get distinct method names from a specific solution path
@app.get("/v1/codebase_comments/distinct_method_names", operation_id="get_distinct_method_names", summary="Retrieves a unique list of method names associated with a specified solution path. The solution path is used to filter the methods returned.")
async def get_distinct_method_names(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT DISTINCT T2.NameTokenized FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.Path = ?", (path,))
    result = cursor.fetchall()
    if not result:
        return {"method_names": []}
    return {"method_names": [row[0] for row in result]}

# Endpoint to get the top repository by solution count within a star range
@app.get("/v1/codebase_comments/top_repo_by_solution_count", operation_id="get_top_repo_by_solution_count", summary="Retrieves the repository with the highest number of solutions within a specified star range and compilation status. The star range is defined by the minimum and maximum number of stars, and the compilation status indicates whether the solutions were compiled or not.")
async def get_top_repo_by_solution_count(min_stars: int = Query(..., description="Minimum number of stars"), max_stars: int = Query(..., description="Maximum number of stars"), was_compiled: int = Query(..., description="Compilation status (0 or 1)")):
    cursor.execute("SELECT T2.RepoId, COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Stars BETWEEN ? AND ? AND T2.WasCompiled = ? GROUP BY T2.RepoId ORDER BY COUNT(T2.RepoId) DESC LIMIT 1", (min_stars, max_stars, was_compiled))
    result = cursor.fetchone()
    if not result:
        return {"repo_id": [], "solution_count": []}
    return {"repo_id": result[0], "solution_count": result[1]}

# Endpoint to get API calls from a specific repository URL
@app.get("/v1/codebase_comments/api_calls_from_repo", operation_id="get_api_calls_from_repo", summary="Retrieves the API calls made within a specific repository. The operation requires the URL of the repository as input and returns the corresponding API calls.")
async def get_api_calls_from_repo(url: str = Query(..., description="URL of the repository")):
    cursor.execute("SELECT T3.ApiCalls FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId INNER JOIN Method AS T3 ON T2.Id = T3.SolutionId WHERE T1.Url = ?", (url,))
    result = cursor.fetchall()
    if not result:
        return {"api_calls": []}
    return {"api_calls": [row[0] for row in result]}

# Endpoint to get the count of distinct solution paths from the repository with the second highest number of watchers
@app.get("/v1/codebase_comments/count_distinct_solution_paths", operation_id="get_count_distinct_solution_paths", summary="Retrieves the number of unique solution paths from the repository with the second highest number of watchers. This operation provides a count of distinct solution paths, offering insights into the diversity of solutions in the repository.")
async def get_count_distinct_solution_paths():
    cursor.execute("SELECT COUNT(DISTINCT T2.Path) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Watchers = ( SELECT Watchers FROM Repo ORDER BY Watchers DESC LIMIT 1, 1 )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average processed time for solutions in a specific repository
@app.get("/v1/codebase_comments/average_processed_time", operation_id="get_average_processed_time", summary="Retrieves the average time taken to process solutions in a specified repository. The calculation is based on the sum of processed times for all solutions in the repository, divided by the total number of solutions in the repository. The repository is identified by its URL.")
async def get_average_processed_time(url: str = Query(..., description="URL of the repository")):
    cursor.execute("SELECT CAST(SUM(T2.ProcessedTime) AS REAL) / COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Url = ?", (url,))
    result = cursor.fetchone()
    if not result:
        return {"average_processed_time": []}
    return {"average_processed_time": result[0]}

# Endpoint to get full comments from a specific solution path and method name
@app.get("/v1/codebase_comments/full_comments", operation_id="get_full_comments", summary="Retrieves comprehensive comments associated with a specific method within a solution. The method is identified by its tokenized name, and the solution is located using its path. This operation returns the full comment text for the specified method.")
async def get_full_comments(path: str = Query(..., description="Path of the solution"), name_tokenized: str = Query(..., description="Tokenized name of the method")):
    cursor.execute("SELECT T2.FullComment FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.Path = ? AND T2.NameTokenized = ?", (path, name_tokenized))
    result = cursor.fetchall()
    if not result:
        return {"full_comments": []}
    return {"full_comments": [row[0] for row in result]}

# Endpoint to get API calls from a specific solution path
@app.get("/v1/codebase_comments/api_calls_from_solution", operation_id="get_api_calls_from_solution", summary="Retrieves the API calls made by a specific solution identified by its path. The path parameter is used to locate the solution and its associated API calls.")
async def get_api_calls_from_solution(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT T2.ApiCalls FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.Path = ?", (path,))
    result = cursor.fetchall()
    if not result:
        return {"api_calls": []}
    return {"api_calls": [row[0] for row in result]}

# Endpoint to get the count of solution paths from a specific repository URL and compilation status
@app.get("/v1/codebase_comments/count_solution_paths", operation_id="get_count_solution_paths", summary="Retrieves the total number of unique solution paths associated with a specific repository, filtered by a given compilation status. The repository is identified by its URL, and the compilation status indicates whether the solutions were successfully compiled (1) or not (0).")
async def get_count_solution_paths(url: str = Query(..., description="URL of the repository"), was_compiled: int = Query(..., description="Compilation status (0 or 1)")):
    cursor.execute("SELECT COUNT(T2.Path) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Url = ? AND T2.WasCompiled = ?", (url, was_compiled))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct processed times and languages from a specific method name
@app.get("/v1/codebase_comments/distinct_processed_times_languages", operation_id="get_distinct_processed_times_languages", summary="Retrieve unique processing times and associated programming languages for a given method name. The method name is provided in a tokenized format.")
async def get_distinct_processed_times_languages(name_tokenized: str = Query(..., description="Tokenized name of the method")):
    cursor.execute("SELECT DISTINCT T1.ProcessedTime, T2.Lang FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.NameTokenized = ?", (name_tokenized,))
    result = cursor.fetchall()
    if not result:
        return {"processed_times_languages": []}
    return {"processed_times_languages": [{"processed_time": row[0], "language": row[1]} for row in result]}

# Endpoint to get sampled times from a specific solution path and method name
@app.get("/v1/codebase_comments/sampled_times", operation_id="get_sampled_times", summary="Retrieves a list of timestamps when a specific method within a solution was sampled. The solution path and method name are used to filter the results.")
async def get_sampled_times(path: str = Query(..., description="Path of the solution"), name: str = Query(..., description="Name of the method")):
    cursor.execute("SELECT T2.SampledAt FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.Path = ? AND T2.Name = ?", (path, name))
    result = cursor.fetchall()
    if not result:
        return {"sampled_times": []}
    return {"sampled_times": [row[0] for row in result]}

# Endpoint to get the language of a solution based on its path
@app.get("/v1/codebase_comments/solution_language_by_path", operation_id="get_solution_language", summary="Retrieves the programming language used in a solution, identified by its unique path. The solution's language is determined by examining the associated method's language metadata.")
async def get_solution_language(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT T2.Lang FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.Path = ?", (path,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to check if a method's comment is in XML format based on its name
@app.get("/v1/codebase_comments/method_comment_is_xml", operation_id="get_method_comment_is_xml", summary="This operation verifies whether the comment associated with a specified method is formatted in XML. It returns a 'Yes' or 'No' response based on the comment's format. The method's name is required as an input parameter.")
async def get_method_comment_is_xml(name: str = Query(..., description="Name of the method")):
    cursor.execute("SELECT CASE WHEN CommentIsXml = 0 THEN 'No' WHEN CommentIsXml = 1 THEN 'Yes' END isXMLFormat FROM Method WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"isXMLFormat": []}
    return {"isXMLFormat": result[0]}

# Endpoint to get the name of a method based on its summary
@app.get("/v1/codebase_comments/method_name_by_summary", operation_id="get_method_name", summary="Retrieves the name of a method that matches the provided summary. The summary is used to identify the method and return its corresponding name.")
async def get_method_name(summary: str = Query(..., description="Summary of the method")):
    cursor.execute("SELECT Name FROM Method WHERE Summary = ?", (summary,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of repositories with the maximum stars
@app.get("/v1/codebase_comments/count_repos_with_max_stars", operation_id="get_count_repos_with_max_stars", summary="Retrieves the total number of repositories that have the highest star count. This operation does not require any input parameters and returns a single integer value representing the count.")
async def get_count_repos_with_max_stars():
    cursor.execute("SELECT COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Stars = ( SELECT MAX(Stars) FROM Repo )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct paths of repositories with the maximum stars
@app.get("/v1/codebase_comments/distinct_paths_repos_with_max_stars", operation_id="get_distinct_paths_repos_with_max_stars", summary="Retrieves a list of unique paths for repositories that have the highest number of stars. This operation does not require any input parameters and returns the distinct paths of the repositories with the maximum stars.")
async def get_distinct_paths_repos_with_max_stars():
    cursor.execute("SELECT DISTINCT T2.Path FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Stars = ( SELECT MAX(Stars) FROM Repo )")
    result = cursor.fetchall()
    if not result:
        return {"paths": []}
    return {"paths": [row[0] for row in result]}

# Endpoint to get the URL of a repository based on the solution ID
@app.get("/v1/codebase_comments/repo_url_by_solution_id", operation_id="get_repo_url", summary="Retrieves the URL of a repository associated with a specific solution. The solution is identified by its unique ID, which is provided as an input parameter. The operation returns the URL of the repository linked to the solution.")
async def get_repo_url(solution_id: int = Query(..., description="ID of the solution")):
    cursor.execute("SELECT T1.Url FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.Id = ?", (solution_id,))
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get the count of repositories with more than a specified number of forks and compiled solutions
@app.get("/v1/codebase_comments/count_repos_forks_compiled", operation_id="get_count_repos_forks_compiled", summary="Retrieves the count of repositories that have surpassed a specified number of forks and contain compiled solutions. The operation considers a solution as compiled if the 'was_compiled' parameter is set to 1.")
async def get_count_repos_forks_compiled(forks: int = Query(..., description="Number of forks"), was_compiled: int = Query(..., description="Whether the solution was compiled (1 for yes, 0 for no)")):
    cursor.execute("SELECT COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Forks > ? AND T2.WasCompiled = ?", (forks, was_compiled))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to determine which solution ID is more popular based on watchers
@app.get("/v1/codebase_comments/compare_solution_popularity", operation_id="compare_solution_popularity", summary="This operation compares the popularity of two solutions based on the number of watchers. It returns the ID of the solution with more watchers. The comparison is performed by aggregating the watchers for each solution across all repositories where they are present.")
async def compare_solution_popularity(solution_id_18: int = Query(..., description="ID of the first solution"), solution_id_19: int = Query(..., description="ID of the second solution")):
    cursor.execute("SELECT CASE WHEN SUM(CASE WHEN T2.Id = ? THEN T1.Watchers ELSE 0 END) > SUM(CASE WHEN T2.Id = ? THEN T1.Watchers ELSE 0 END) THEN 'SolutionID18' WHEN SUM(CASE WHEN T2.Id = ? THEN T1.Watchers ELSE 0 END) < SUM(CASE WHEN T2.Id = ? THEN T1.Watchers ELSE 0 END) THEN 'SolutionID19' END isMorePopular FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId", (solution_id_18, solution_id_19, solution_id_18, solution_id_19))
    result = cursor.fetchone()
    if not result:
        return {"isMorePopular": []}
    return {"isMorePopular": result[0]}

# Endpoint to get the count of repositories with the latest processed time and compiled solutions
@app.get("/v1/codebase_comments/count_repos_latest_processed_time_compiled", operation_id="get_count_repos_latest_processed_time_compiled", summary="Retrieves the total number of repositories that have been processed most recently and contain compiled solutions. The operation considers whether the solutions were compiled or not based on the provided input parameter.")
async def get_count_repos_latest_processed_time_compiled(was_compiled: int = Query(..., description="Whether the solution was compiled (1 for yes, 0 for no)")):
    cursor.execute("SELECT COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.ProcessedTime = ( SELECT MAX(ProcessedTime) FROM Repo ) AND T2.WasCompiled = ?", (was_compiled,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct processed times for a given solution path
@app.get("/v1/codebase_comments/distinct_processed_times", operation_id="get_distinct_processed_times", summary="Retrieves a list of unique time stamps when solutions under a specific path were processed. This operation is useful for tracking the distinct processing times of solutions in a given path. The path of the solution is required as an input parameter.")
async def get_distinct_processed_times(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT DISTINCT T2.ProcessedTime FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.Path = ?", (path,))
    result = cursor.fetchall()
    if not result:
        return {"processed_times": []}
    return {"processed_times": [row[0] for row in result]}

# Endpoint to get solution paths for a given repository URL
@app.get("/v1/codebase_comments/solution_paths_by_repo_url", operation_id="get_solution_paths_by_repo_url", summary="Retrieves the paths of solutions associated with a specific repository. The operation uses the provided repository URL to identify the relevant repository and returns the paths of all solutions linked to it.")
async def get_solution_paths_by_repo_url(repo_url: str = Query(..., description="URL of the repository")):
    cursor.execute("SELECT T2.Path FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Url = ?", (repo_url,))
    result = cursor.fetchall()
    if not result:
        return {"paths": []}
    return {"paths": [row[0] for row in result]}

# Endpoint to get the count of repositories with solutions processed before a given time and having more than a specified number of stars
@app.get("/v1/codebase_comments/repo_count_by_processed_time_and_stars", operation_id="get_repo_count_by_processed_time_and_stars", summary="Retrieve the number of repositories that have been processed before a specified time and have a star count exceeding a given threshold. The processed time is measured in ticks, and the star count threshold is provided as an input parameter.")
async def get_repo_count_by_processed_time_and_stars(processed_time: int = Query(..., description="Processed time in ticks"), stars: int = Query(..., description="Minimum number of stars")):
    cursor.execute("SELECT COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.ProcessedTime < ? AND T1.Stars > ?", (processed_time, stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get solution IDs ordered by the number of forks in descending order with a limit
@app.get("/v1/codebase_comments/solution_ids_by_forks", operation_id="get_solution_ids_by_forks", summary="Retrieves a list of solution IDs, ordered by the number of forks in descending order. The operation limits the number of results based on the provided limit parameter. This endpoint is useful for obtaining the most forked solutions, which can indicate popular or highly-adopted solutions.")
async def get_solution_ids_by_forks(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.Id FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId ORDER BY T1.Forks DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"solution_ids": []}
    return {"solution_ids": [row[0] for row in result]}

# Endpoint to get the percentage difference in forks between two solution IDs
@app.get("/v1/codebase_comments/forks_percentage_difference", operation_id="get_forks_percentage_difference", summary="Retrieves the percentage difference in forks between two specified solutions. This operation calculates the difference in the total number of forks between the two solutions and expresses it as a percentage of the total forks for the first solution. The input parameters represent the unique identifiers of the two solutions being compared.")
async def get_forks_percentage_difference(solution_id_1: int = Query(..., description="First solution ID"), solution_id_2: int = Query(..., description="Second solution ID")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T2.Id = ? THEN T1.Forks ELSE 0 END) - SUM(CASE WHEN T2.Id = ? THEN T1.Forks ELSE 0 END)) AS REAL) * 100 / SUM(CASE WHEN T2.Id = ? THEN T1.Forks ELSE 0 END) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId", (solution_id_1, solution_id_2, solution_id_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the language of a method by its name
@app.get("/v1/codebase_comments/method_language", operation_id="get_method_language", summary="Retrieves the programming language used in a method, identified by its unique name. The method name is provided as an input parameter.")
async def get_method_language(method_name: str = Query(..., description="Name of the method")):
    cursor.execute("SELECT Lang FROM Method WHERE Name = ?", (method_name,))
    result = cursor.fetchone()
    if not result:
        return {"language": []}
    return {"language": result[0]}

# Endpoint to get the full comment of a method by its name
@app.get("/v1/codebase_comments/method_full_comment", operation_id="get_method_full_comment", summary="Retrieves the complete comment associated with a specific method, identified by its unique name. The comment provides detailed information about the method's functionality and usage.")
async def get_method_full_comment(method_name: str = Query(..., description="Name of the method")):
    cursor.execute("SELECT FullComment FROM Method WHERE Name = ?", (method_name,))
    result = cursor.fetchone()
    if not result:
        return {"full_comment": []}
    return {"full_comment": result[0]}

# Endpoint to get distinct summaries of a method by its name
@app.get("/v1/codebase_comments/method_summaries", operation_id="get_method_summaries", summary="Retrieves unique summaries for a specific method, identified by its name. This operation provides a concise overview of the method's purpose and functionality.")
async def get_method_summaries(method_name: str = Query(..., description="Name of the method")):
    cursor.execute("SELECT DISTINCT Summary FROM Method WHERE Name = ?", (method_name,))
    result = cursor.fetchall()
    if not result:
        return {"summaries": []}
    return {"summaries": [row[0] for row in result]}

# Endpoint to get the tokenized name of a method based on its name
@app.get("/v1/codebase_comments/method_name_tokenized", operation_id="get_method_name_tokenized", summary="Retrieves the tokenized representation of a method's name, given the original name as input. This operation is useful for searching or processing method names in a standardized format.")
async def get_method_name_tokenized(name: str = Query(..., description="Name of the method")):
    cursor.execute("SELECT NameTokenized FROM Method WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"name_tokenized": []}
    return {"name_tokenized": result[0]}

# Endpoint to get the stars of a repository based on solution ID
@app.get("/v1/codebase_comments/repo_stars_by_solution_id", operation_id="get_repo_stars_by_solution_id", summary="Retrieves the star count of a repository associated with the specified solution ID. The solution ID is used to identify the relevant repository and fetch its star count.")
async def get_repo_stars_by_solution_id(solution_id: int = Query(..., description="ID of the solution")):
    cursor.execute("SELECT T1.Stars FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.Id = ?", (solution_id,))
    result = cursor.fetchone()
    if not result:
        return {"stars": []}
    return {"stars": result[0]}

# Endpoint to get the count of solutions in repositories with a specific number of stars
@app.get("/v1/codebase_comments/count_solutions_by_repo_stars", operation_id="get_count_solutions_by_repo_stars", summary="Retrieves the total number of solutions found in repositories with a specified number of stars. The input parameter determines the number of stars that the repositories should have.")
async def get_count_solutions_by_repo_stars(stars: int = Query(..., description="Number of stars of the repository")):
    cursor.execute("SELECT COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Stars = ?", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the path of a solution based on the method name
@app.get("/v1/codebase_comments/solution_path_by_method_name", operation_id="get_solution_path_by_method_name", summary="Retrieves the file path of a solution associated with a specified method name. The method name is used to identify the solution and return its corresponding file path.")
async def get_solution_path_by_method_name(method_name: str = Query(..., description="Name of the method")):
    cursor.execute("SELECT T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Name = ?", (method_name,))
    result = cursor.fetchone()
    if not result:
        return {"path": []}
    return {"path": result[0]}

# Endpoint to get the processed time of a solution based on the tokenized method name
@app.get("/v1/codebase_comments/solution_processed_time_by_method_name_tokenized", operation_id="get_solution_processed_time_by_method_name_tokenized", summary="Get the processed time of a solution based on the tokenized method name")
async def get_solution_processed_time_by_method_name_tokenized(name_tokenized: str = Query(..., description="Tokenized name of the method")):
    cursor.execute("SELECT T1.ProcessedTime FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.NameTokenized = ?", (name_tokenized,))
    result = cursor.fetchone()
    if not result:
        return {"processed_time": []}
    return {"processed_time": result[0]}

# Endpoint to get the repository ID of a solution based on the method name
@app.get("/v1/codebase_comments/repo_id_by_method_name", operation_id="get_repo_id_by_method_name", summary="Retrieves the unique identifier of the repository associated with a specific method. The method name is used to locate the corresponding repository.")
async def get_repo_id_by_method_name(method_name: str = Query(..., description="Name of the method")):
    cursor.execute("SELECT T1.RepoId FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Name = ?", (method_name,))
    result = cursor.fetchone()
    if not result:
        return {"repo_id": []}
    return {"repo_id": result[0]}

# Endpoint to get the count of solutions in repositories with a specific number of watchers
@app.get("/v1/codebase_comments/count_solutions_by_repo_watchers", operation_id="get_count_solutions_by_repo_watchers", summary="Retrieves the total number of solutions found in repositories that have a specified number of watchers. The input parameter determines the number of watchers that the repositories should have.")
async def get_count_solutions_by_repo_watchers(watchers: int = Query(..., description="Number of watchers of the repository")):
    cursor.execute("SELECT COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Watchers = ?", (watchers,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the URL of the repository with the highest number of solutions
@app.get("/v1/codebase_comments/repo_url_with_most_solutions", operation_id="get_repo_url_with_most_solutions", summary="Retrieves the URL of the repository that has the most solutions. This operation identifies the repository with the highest number of solutions by joining the Repo and Solution tables, grouping by the RepoId, and ordering the results in descending order based on the count of RepoId. The URL of the top repository is then returned.")
async def get_repo_url_with_most_solutions():
    cursor.execute("SELECT T1.Url FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId GROUP BY T2.RepoId ORDER BY COUNT(T2.RepoId) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get the count of solutions in repositories with a specific number of forks
@app.get("/v1/codebase_comments/count_solutions_by_repo_forks", operation_id="get_count_solutions_by_repo_forks", summary="Retrieves the total number of solutions found in repositories with a specified number of forks. The input parameter determines the number of forks to consider when counting the solutions.")
async def get_count_solutions_by_repo_forks(forks: int = Query(..., description="Number of forks of the repository")):
    cursor.execute("SELECT COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Forks = ?", (forks,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common solution path for a given language
@app.get("/v1/codebase_comments/most_common_solution_path_by_lang", operation_id="get_most_common_solution_path", summary="Retrieves the most frequently used solution path for a specified programming language. The language is identified by its code (e.g., 'zh-cn'). The result is determined by counting the occurrences of each solution path associated with the given language and selecting the one with the highest count.")
async def get_most_common_solution_path(lang: str = Query(..., description="Language code (e.g., 'zh-cn')")):
    cursor.execute("SELECT T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Lang = ? GROUP BY T1.Path ORDER BY COUNT(T1.Path) DESC LIMIT 1", (lang,))
    result = cursor.fetchone()
    if not result:
        return {"path": []}
    return {"path": result[0]}

# Endpoint to get the number of watchers for a repository by solution ID
@app.get("/v1/codebase_comments/repo_watchers_by_solution_id", operation_id="get_repo_watchers", summary="Retrieves the total number of watchers for a specific repository associated with the provided solution ID. The solution ID is used to identify the relevant repository and fetch the corresponding watcher count.")
async def get_repo_watchers(solution_id: int = Query(..., description="Solution ID")):
    cursor.execute("SELECT T1.Watchers FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.Id = ?", (solution_id,))
    result = cursor.fetchone()
    if not result:
        return {"watchers": []}
    return {"watchers": result[0]}

# Endpoint to get the count of repository IDs based on stars and compilation status
@app.get("/v1/codebase_comments/repo_count_by_stars_and_compilation", operation_id="get_repo_count", summary="Retrieves the total count of repositories that have a specific number of stars and a particular compilation status. The stars parameter filters repositories based on their star count, while the was_compiled parameter determines whether the count includes only repositories that have been compiled or not.")
async def get_repo_count(stars: int = Query(..., description="Number of stars"), was_compiled: int = Query(..., description="Compilation status (0 or 1)")):
    cursor.execute("SELECT COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Stars = ? AND T2.WasCompiled = ?", (stars, was_compiled))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct repository IDs by method name
@app.get("/v1/codebase_comments/distinct_repo_ids_by_method_name", operation_id="get_distinct_repo_ids", summary="Retrieves a unique set of repository IDs associated with a specific method name. The method name is used to filter the results, ensuring that only relevant repositories are included in the response.")
async def get_distinct_repo_ids(method_name: str = Query(..., description="Method name")):
    cursor.execute("SELECT DISTINCT T1.RepoId FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Name = ?", (method_name,))
    result = cursor.fetchall()
    if not result:
        return {"repo_ids": []}
    return {"repo_ids": [item[0] for item in result]}

# Endpoint to get distinct solution paths by method summary
@app.get("/v1/codebase_comments/distinct_solution_paths_by_summary", operation_id="get_distinct_solution_paths", summary="Retrieves a list of distinct solution paths where the associated method summary matches the provided input. The method summary describes the functionality of the method, such as refetching an entity from persistent storage.")
async def get_distinct_solution_paths(summary: str = Query(..., description="Method summary")):
    cursor.execute("SELECT DISTINCT T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Summary = ?", (summary,))
    result = cursor.fetchall()
    if not result:
        return {"paths": []}
    return {"paths": [item[0] for item in result]}

# Endpoint to get distinct solution paths by language
@app.get("/v1/codebase_comments/distinct_solution_paths_by_lang", operation_id="get_distinct_solution_paths_by_lang", summary="Retrieves a list of distinct solution paths for a given programming language. The operation filters solutions based on the specified language code and returns unique paths from the filtered results.")
async def get_distinct_solution_paths_by_lang(lang: str = Query(..., description="Language code (e.g., 'sw')")):
    cursor.execute("SELECT DISTINCT T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Lang = ?", (lang,))
    result = cursor.fetchall()
    if not result:
        return {"paths": []}
    return {"paths": [item[0] for item in result]}

# Endpoint to get the percentage difference in watchers between two solution IDs
@app.get("/v1/codebase_comments/watcher_percentage_difference", operation_id="get_watcher_percentage_difference", summary="Retrieves the percentage difference in the number of watchers between two specified solutions. This operation compares the total watchers of the first solution with the total watchers of the second solution, and returns the difference as a percentage of the first solution's total watchers.")
async def get_watcher_percentage_difference(solution_id_1: int = Query(..., description="First solution ID"), solution_id_2: int = Query(..., description="Second solution ID")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Id = ? THEN T1.Watchers ELSE 0 END) - SUM(CASE WHEN T2.Id = ? THEN T1.Watchers ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN T2.Id = ? THEN T1.Watchers ELSE 0 END) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId", (solution_id_1, solution_id_2, solution_id_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the percentage difference in stars between two solutions
@app.get("/v1/codebase_comments/percentage_difference_stars", operation_id="get_percentage_difference_stars", summary="Retrieve the percentage difference in stars awarded to two solutions, identified by their unique IDs. This operation compares the total stars received by each solution and calculates the percentage difference, providing insights into their relative popularity or quality.")
async def get_percentage_difference_stars(id1: int = Query(..., description="ID of the first solution"), id2: int = Query(..., description="ID of the second solution")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Id = ? THEN T1.Stars ELSE 0 END) - SUM(CASE WHEN T2.Id = ? THEN T1.Stars ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN T2.Id = ? THEN T1.Stars ELSE 0 END) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId", (id1, id2, id2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the percentage difference in forks between two solutions
@app.get("/v1/codebase_comments/percentage_difference_forks", operation_id="get_percentage_difference_forks", summary="Retrieves the percentage difference in the number of forks between two specified solutions. The calculation is based on the total number of forks for each solution, which is determined by summing the forks across all repositories associated with each solution. The result is expressed as a percentage relative to the total forks of the first solution.")
async def get_percentage_difference_forks(id1: int = Query(..., description="ID of the first solution"), id2: int = Query(..., description="ID of the second solution")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Id = ? THEN T1.Forks ELSE 0 END) - SUM(CASE WHEN T2.Id = ? THEN T1.Forks ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN T2.Id = ? THEN T1.Forks ELSE 0 END) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId", (id1, id2, id2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get distinct method names based on processed time
@app.get("/v1/codebase_comments/distinct_method_names_by_processed_time", operation_id="get_distinct_method_names_by_processed_time", summary="Retrieve a unique set of method names associated with solutions processed at a specific time. The operation filters solutions based on the provided processed time and returns the distinct method names linked to these solutions.")
async def get_distinct_method_names_by_processed_time(processed_time: int = Query(..., description="Processed time in ticks")):
    cursor.execute("SELECT DISTINCT T2.Name FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.ProcessedTime = ?", (processed_time,))
    result = cursor.fetchall()
    if not result:
        return {"method_names": []}
    return {"method_names": [row[0] for row in result]}

# Endpoint to get the count of repository IDs based on repository URL
@app.get("/v1/codebase_comments/count_repo_ids_by_url", operation_id="get_count_repo_ids_by_url", summary="Retrieves the total number of unique repositories associated with a given repository URL. The count is determined by matching the provided URL with the URLs of repositories in the database and then counting the corresponding repository IDs.")
async def get_count_repo_ids_by_url(url: str = Query(..., description="Repository URL")):
    cursor.execute("SELECT COUNT(T2.RepoId) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Url = ?", (url,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total processed time for repositories with the maximum number of forks
@app.get("/v1/codebase_comments/total_processed_time_max_forks", operation_id="get_total_processed_time_max_forks", summary="Retrieves the cumulative time spent processing repositories that have the highest number of forks. This operation calculates the total processing time by aggregating the individual processing times of each solution associated with the repositories that have the maximum number of forks.")
async def get_total_processed_time_max_forks():
    cursor.execute("SELECT SUM(T2.ProcessedTime) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Forks = ( SELECT MAX(Forks) FROM Repo )")
    result = cursor.fetchone()
    if not result:
        return {"total_processed_time": []}
    return {"total_processed_time": result[0]}

# Endpoint to get distinct method IDs based on repository ID and method language
@app.get("/v1/codebase_comments/distinct_method_ids_by_repo_id_lang", operation_id="get_distinct_method_ids_by_repo_id_lang", summary="Retrieves a unique set of method identifiers associated with a specific repository and programming language. The operation filters methods based on the provided repository ID and language, ensuring that only distinct method IDs are returned.")
async def get_distinct_method_ids_by_repo_id_lang(repo_id: int = Query(..., description="Repository ID"), lang: str = Query(..., description="Method language")):
    cursor.execute("SELECT DISTINCT T2.id FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.RepoId = ? AND T2.Lang = ?", (repo_id, lang))
    result = cursor.fetchall()
    if not result:
        return {"method_ids": []}
    return {"method_ids": [row[0] for row in result]}

# Endpoint to get distinct solution paths based on repository URL
@app.get("/v1/codebase_comments/distinct_solution_paths_by_url", operation_id="get_distinct_solution_paths_by_url", summary="Retrieves a list of unique solution paths associated with a specific repository URL. The operation filters the solutions based on the provided repository URL and returns only the distinct paths.")
async def get_distinct_solution_paths_by_url(url: str = Query(..., description="Repository URL")):
    cursor.execute("SELECT DISTINCT T2.Path FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Url = ?", (url,))
    result = cursor.fetchall()
    if not result:
        return {"solution_paths": []}
    return {"solution_paths": [row[0] for row in result]}

# Endpoint to get distinct repository IDs based on method language
@app.get("/v1/codebase_comments/distinct_repo_ids_by_lang", operation_id="get_distinct_repo_ids_by_lang", summary="Retrieves a unique set of repository IDs associated with a specified programming language. This operation filters the repositories based on the language used in their methods, providing a distinct list of repository IDs that match the given language.")
async def get_distinct_repo_ids_by_lang(lang: str = Query(..., description="Method language")):
    cursor.execute("SELECT DISTINCT T1.RepoId FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Lang = ?", (lang,))
    result = cursor.fetchall()
    if not result:
        return {"repo_ids": []}
    return {"repo_ids": [row[0] for row in result]}

# Endpoint to get repository IDs based on tokenized method name
@app.get("/v1/codebase_comments/repo_ids_by_method_name", operation_id="get_repo_ids_by_method_name", summary="Retrieves the unique identifiers of repositories that contain a method with the specified tokenized name. The tokenized name is a processed version of the method name used for efficient searching and matching.")
async def get_repo_ids_by_method_name(name_tokenized: str = Query(..., description="Tokenized name of the method")):
    cursor.execute("SELECT T1.RepoId FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.NameTokenized = ?", (name_tokenized,))
    result = cursor.fetchall()
    if not result:
        return {"repo_ids": []}
    return {"repo_ids": [row[0] for row in result]}

# Endpoint to get the count of solution IDs based on solution path
@app.get("/v1/codebase_comments/count_solution_ids_by_path", operation_id="get_count_solution_ids_by_path", summary="Retrieves the total number of unique solution IDs associated with a specific solution path. The solution path is used to filter the results, providing a count of solutions that meet the specified path criteria.")
async def get_count_solution_ids_by_path(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT COUNT(T2.SolutionId) FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.Path = ?", (path,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get solution IDs based on repository processed time
@app.get("/v1/codebase_comments/solution_ids_by_processed_time", operation_id="get_solution_ids_by_processed_time", summary="Retrieves the IDs of solutions associated with a repository that was processed at a specific time. The operation uses the provided repository processed time to identify and return the corresponding solution IDs.")
async def get_solution_ids_by_processed_time(processed_time: int = Query(..., description="Processed time of the repository")):
    cursor.execute("SELECT T2.Id FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.ProcessedTime = ?", (processed_time,))
    result = cursor.fetchall()
    if not result:
        return {"solution_ids": []}
    return {"solution_ids": [row[0] for row in result]}

# Endpoint to get repository URLs based on the maximum processed time of solutions
@app.get("/v1/codebase_comments/repo_urls_by_max_processed_time", operation_id="get_repo_urls_by_max_processed_time", summary="Retrieves a list of repository URLs associated with the solution that has the latest processing time. This operation does not require any input parameters and returns the URLs of repositories that contain the most recently processed solutions.")
async def get_repo_urls_by_max_processed_time():
    cursor.execute("SELECT T1.Url FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.ProcessedTime = ( SELECT MAX(ProcessedTime) FROM Solution )")
    result = cursor.fetchall()
    if not result:
        return {"repo_urls": []}
    return {"repo_urls": [row[0] for row in result]}

# Endpoint to get distinct solution IDs based on repository forks and watchers
@app.get("/v1/codebase_comments/distinct_solution_ids_by_forks_watchers", operation_id="get_distinct_solution_ids_by_forks_watchers", summary="Retrieves a list of unique solution identifiers that meet the criteria of having more forks than half the number of watchers. This operation fetches data from the Repo and Solution tables, joining them based on their shared IDs. The results are filtered to only include solutions where the number of forks surpasses half the count of watchers.")
async def get_distinct_solution_ids_by_forks_watchers():
    cursor.execute("SELECT DISTINCT T2.Id FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Forks > T1.Watchers / 2")
    result = cursor.fetchall()
    if not result:
        return {"solution_ids": []}
    return {"solution_ids": [row[0] for row in result]}

# Endpoint to get the percentage of forks relative to stars for a specific solution ID
@app.get("/v1/codebase_comments/forks_percentage_by_solution_id", operation_id="get_forks_percentage_by_solution_id", summary="Retrieves the percentage of forks relative to stars for a specific solution. The solution is identified by its unique ID. This operation provides insights into the popularity and engagement of the solution within the repository.")
async def get_forks_percentage_by_solution_id(solution_id: int = Query(..., description="ID of the solution")):
    cursor.execute("SELECT CAST(T1.Forks AS REAL) * 100 / T1.Stars FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.Id = ?", (solution_id,))
    result = cursor.fetchone()
    if not result:
        return {"forks_percentage": []}
    return {"forks_percentage": result[0]}

# Endpoint to get repository IDs with the maximum number of stars
@app.get("/v1/codebase_comments/repo_ids_by_max_stars", operation_id="get_repo_ids_by_max_stars", summary="Retrieves the unique identifiers of repositories that have the highest number of stars. This operation returns a list of repository IDs that have the maximum star count in the database.")
async def get_repo_ids_by_max_stars():
    cursor.execute("SELECT Id FROM Repo WHERE Stars = ( SELECT MAX(Stars) FROM Repo )")
    result = cursor.fetchall()
    if not result:
        return {"repo_ids": []}
    return {"repo_ids": [row[0] for row in result]}

# Endpoint to get the number of forks for a specific solution ID
@app.get("/v1/codebase_comments/forks_by_solution_id", operation_id="get_forks_by_solution_id", summary="Retrieves the total number of forks associated with a specific solution. The solution is identified by its unique ID. This operation provides insights into the replication and distribution of the solution within the repository.")
async def get_forks_by_solution_id(solution_id: int = Query(..., description="ID of the solution")):
    cursor.execute("SELECT T1.Forks FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.Id = ?", (solution_id,))
    result = cursor.fetchone()
    if not result:
        return {"forks": []}
    return {"forks": result[0]}

# Endpoint to get solution IDs for repositories with the maximum number of watchers
@app.get("/v1/codebase_comments/solution_ids_by_max_watchers", operation_id="get_solution_ids_by_max_watchers", summary="Retrieves the unique identifiers of solutions associated with repositories that have the highest number of watchers. This operation returns a list of solution IDs for repositories with the maximum number of watchers, providing insights into the most popular repositories and their corresponding solutions.")
async def get_solution_ids_by_max_watchers():
    cursor.execute("SELECT T2.Id FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Watchers = ( SELECT MAX(Watchers) FROM Repo )")
    result = cursor.fetchall()
    if not result:
        return {"solution_ids": []}
    return {"solution_ids": [row[0] for row in result]}

# Endpoint to get the count of methods and compilation status based on solution ID
@app.get("/v1/codebase_comments/method_count_compilation_status", operation_id="get_method_count_compilation_status", summary="Retrieves the count of methods associated with a specific solution and their compilation status. The solution is identified by its unique ID. The response indicates whether the methods need to be compiled or not.")
async def get_method_count_compilation_status(solution_id: int = Query(..., description="ID of the solution")):
    cursor.execute("SELECT COUNT(T2.SolutionId), CASE WHEN T1.WasCompiled = 0 THEN 'Needs' ELSE 'NoNeeds' END needToCompile FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.SolutionId = ?", (solution_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "needToCompile": []}
    return {"count": result[0], "needToCompile": result[1]}

# Endpoint to get the percentage of solutions that need compilation based on XML comments
@app.get("/v1/codebase_comments/percentage_needs_compilation", operation_id="get_percentage_needs_compilation", summary="Retrieves the percentage of solutions that require compilation, based on whether their comments are formatted in XML. The input parameter specifies whether to consider solutions with XML comments.")
async def get_percentage_needs_compilation(comment_is_xml: int = Query(..., description="Whether the comment is in XML format (1 for yes, 0 for no)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.WasCompiled = 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.SolutionId) FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.CommentIsXml = ?", (comment_is_xml,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get method names based on solution path
@app.get("/v1/codebase_comments/method_names_by_solution_path", operation_id="get_method_names_by_solution_path", summary="Retrieves the names of methods associated with a specific solution path. The solution path is used to identify the relevant solution, and the method names are then fetched from the corresponding solution.")
async def get_method_names_by_solution_path(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT T2.Name FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.Path = ?", (path,))
    result = cursor.fetchall()
    if not result:
        return {"method_names": []}
    return {"method_names": [row[0] for row in result]}

# Endpoint to get the percentage of repositories with stars greater than a specified number
@app.get("/v1/codebase_comments/percentage_repos_with_stars", operation_id="get_percentage_repos_with_stars", summary="Retrieves the percentage of repositories that have a star count exceeding the specified minimum. This operation calculates the proportion of repositories with a star count greater than the provided minimum value, providing insights into the popularity of repositories based on their star count.")
async def get_percentage_repos_with_stars(min_stars: int = Query(..., description="Minimum number of stars")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Stars > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(Stars) FROM Repo", (min_stars,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ID of the repository with the maximum forks for a given number of stars
@app.get("/v1/codebase_comments/repo_id_max_forks_by_stars", operation_id="get_repo_id_max_forks_by_stars", summary="Retrieves the unique identifier of the repository that has the highest number of forks for a specified number of stars. This operation is useful for identifying the most popular repositories based on their star count and fork count.")
async def get_repo_id_max_forks_by_stars(stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT Id FROM Repo WHERE Stars = ? AND Forks = (SELECT MAX(Forks) FROM Repo WHERE Stars = ?)", (stars, stars))
    result = cursor.fetchone()
    if not result:
        return {"id": []}
    return {"id": result[0]}

# Endpoint to get the percentage of methods in a specific language with XML comments
@app.get("/v1/codebase_comments/percentage_methods_by_language_xml_comments", operation_id="get_percentage_methods_by_language_xml_comments", summary="Retrieves the percentage of methods written in a specified programming language that have XML comments. The calculation is based on the total count of methods in the database.")
async def get_percentage_methods_by_language_xml_comments(lang: str = Query(..., description="Language of the method"), comment_is_xml: int = Query(..., description="Whether the comment is in XML format (1 for yes, 0 for no)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Lang = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(Name) FROM Method WHERE CommentIsXml = ?", (lang, comment_is_xml))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the path of a solution based on tokenized method name
@app.get("/v1/codebase_comments/solution_path_by_tokenized_method_name", operation_id="get_solution_path_by_tokenized_method_name", summary="Retrieves the file path of a solution that corresponds to a given tokenized method name. The method name is provided as a parameter, and the operation returns the path of the solution associated with that method.")
async def get_solution_path_by_tokenized_method_name(name_tokenized: str = Query(..., description="Tokenized name of the method")):
    cursor.execute("SELECT T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.NameTokenized = ?", (name_tokenized,))
    result = cursor.fetchone()
    if not result:
        return {"path": []}
    return {"path": result[0]}

# Endpoint to get the ID of solutions based on the number of forks of the repository
@app.get("/v1/codebase_comments/solution_ids_by_repo_forks", operation_id="get_solution_ids_by_repo_forks", summary="Retrieves the unique identifiers of solutions associated with a repository that has a specific number of forks. The operation filters solutions based on the provided number of forks of the repository.")
async def get_solution_ids_by_repo_forks(forks: int = Query(..., description="Number of forks of the repository")):
    cursor.execute("SELECT T2.Id FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Forks = ?", (forks,))
    result = cursor.fetchall()
    if not result:
        return {"solution_ids": []}
    return {"solution_ids": [row[0] for row in result]}

# Endpoint to get tokenized names from solutions based on language and compilation status
@app.get("/v1/codebase_comments/tokenized_names_by_lang_compiled", operation_id="get_tokenized_names", summary="Retrieves a list of tokenized names from solutions that match the specified programming language and compilation status. The response includes tokenized names from solutions that were written in the provided language and have the corresponding compilation status.")
async def get_tokenized_names(lang: str = Query(..., description="Language of the solution"), was_compiled: int = Query(..., description="Compilation status (0 for not compiled, 1 for compiled)")):
    cursor.execute("SELECT NameTokenized FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE Lang = ? AND WasCompiled = ?", (lang, was_compiled))
    result = cursor.fetchall()
    if not result:
        return {"tokenized_names": []}
    return {"tokenized_names": [row[0] for row in result]}

# Endpoint to get solution paths based on the full comment of the method
@app.get("/v1/codebase_comments/solution_paths_by_full_comment", operation_id="get_solution_paths", summary="Retrieves the solution paths associated with a method based on its full comment. The method's full comment is used to identify the corresponding solution paths in the database.")
async def get_solution_paths(full_comment: str = Query(..., description="Full comment of the method")):
    cursor.execute("SELECT T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.FullComment = ?", (full_comment,))
    result = cursor.fetchall()
    if not result:
        return {"paths": []}
    return {"paths": [row[0] for row in result]}

# Endpoint to get the count of solution IDs based on repository ID and XML comment status
@app.get("/v1/codebase_comments/solution_count_by_repo_xml_comment", operation_id="get_solution_count", summary="Retrieves the total number of unique solutions associated with a specific repository, filtered by whether the solution's comments are in XML format. The response is based on the provided repository ID and XML comment status.")
async def get_solution_count(repo_id: int = Query(..., description="Repository ID"), comment_is_xml: int = Query(..., description="XML comment status (0 for not XML, 1 for XML)")):
    cursor.execute("SELECT COUNT(T2.SolutionId) FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.RepoId = ? AND T2.CommentIsXml = ?", (repo_id, comment_is_xml))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of solutions that were not compiled based on language
@app.get("/v1/codebase_comments/percentage_not_compiled_by_lang", operation_id="get_percentage_not_compiled", summary="Retrieves the percentage of solutions in a specific programming language that were not successfully compiled. This operation calculates the ratio of uncompiled solutions to the total number of solutions in the given language.")
async def get_percentage_not_compiled(lang: str = Query(..., description="Language of the solution")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.WasCompiled = 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(Lang) FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Lang = ?", (lang,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of distinct repositories based on the ratio of stars to forks
@app.get("/v1/codebase_comments/distinct_repo_count_by_stars_forks_ratio", operation_id="get_distinct_repo_count", summary="Retrieve the count of unique repositories where the number of stars surpasses a specified fraction of the number of forks. The input parameter determines the fraction used for comparison.")
async def get_distinct_repo_count(stars_forks_ratio: float = Query(..., description="Ratio of stars to forks (e.g., 3 for 1/3)")):
    cursor.execute("SELECT COUNT(DISTINCT T1.Id) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Stars > CAST(T1.Forks AS REAL) / ?", (stars_forks_ratio,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the task name from a method based on its ID
@app.get("/v1/codebase_comments/task_name_by_method_id", operation_id="get_task_name", summary="Retrieves the task name associated with a specific method, identified by its unique ID. The task name is extracted from the method's full name, which is stored in a structured format containing multiple dot-separated components. The endpoint returns the task name as a standalone string.")
async def get_task_name(method_id: int = Query(..., description="Method ID")):
    cursor.execute("SELECT SUBSTR(SUBSTR(Name, INSTR(Name, '.') + 1), 1, INSTR(SUBSTR(Name, INSTR(Name, '.') + 1), '.') - 1) task FROM Method WHERE Id = ?", (method_id,))
    result = cursor.fetchone()
    if not result:
        return {"task_name": []}
    return {"task_name": result[0]}

# Endpoint to get the comment format of a method based on its ID
@app.get("/v1/codebase_comments/comment_format_by_method_id", operation_id="get_comment_format", summary="Retrieves the comment format of a specific method, identified by its unique ID. The format can be either 'isNotXMLFormat' or 'isXMLFormat'.")
async def get_comment_format(method_id: int = Query(..., description="Method ID")):
    cursor.execute("SELECT CASE WHEN CommentIsXml = 0 THEN 'isNotXMLFormat' WHEN CommentIsXml = 1 THEN 'isXMLFormat' END format FROM Method WHERE Id = ?", (method_id,))
    result = cursor.fetchone()
    if not result:
        return {"comment_format": []}
    return {"comment_format": result[0]}

# Endpoint to get the URL of the repository with the maximum number of watchers
@app.get("/v1/codebase_comments/repo_url_max_watchers", operation_id="get_repo_url_max_watchers", summary="Retrieves the URL of the repository with the highest number of watchers. This operation identifies the repository with the maximum number of watchers and returns its URL.")
async def get_repo_url_max_watchers():
    cursor.execute("SELECT Url FROM Repo WHERE Watchers = ( SELECT MAX(Watchers) FROM Repo )")
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get distinct tasks based on language
@app.get("/v1/codebase_comments/distinct_tasks_by_lang", operation_id="get_distinct_tasks_by_lang", summary="Retrieves a list of unique tasks associated with the specified programming language. This operation filters methods based on the provided language and extracts the distinct task names from their fully qualified names.")
async def get_distinct_tasks_by_lang(lang: str = Query(..., description="Language of the method")):
    cursor.execute("SELECT DISTINCT SUBSTR(SUBSTR(Name, INSTR(Name, '.') + 1), 1, INSTR(SUBSTR(Name, INSTR(Name, '.') + 1), '.') - 1) task FROM Method WHERE Lang = ?", (lang,))
    result = cursor.fetchall()
    if not result:
        return {"tasks": []}
    return {"tasks": [row[0] for row in result]}

# Endpoint to get solution path based on method ID
@app.get("/v1/codebase_comments/solution_path_by_method_id", operation_id="get_solution_path_by_method_id", summary="Retrieves the file path of the solution associated with the provided method ID. This operation fetches the solution's path by querying the Solution and Method tables using the given method ID.")
async def get_solution_path_by_method_id(method_id: int = Query(..., description="ID of the method")):
    cursor.execute("SELECT T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Id = ?", (method_id,))
    result = cursor.fetchone()
    if not result:
        return {"path": []}
    return {"path": result[0]}

# Endpoint to get method language based on method ID and repository ID
@app.get("/v1/codebase_comments/method_lang_by_method_id_repo_id", operation_id="get_method_lang_by_method_id_repo_id", summary="Retrieves the programming language used in the method identified by the provided method ID and repository ID. This operation fetches the language information from the Method table, which is linked to the Solution table via the SolutionId field. The method and repository are identified using their respective IDs.")
async def get_method_lang_by_method_id_repo_id(method_id: int = Query(..., description="ID of the method"), repo_id: int = Query(..., description="ID of the repository")):
    cursor.execute("SELECT T2.Lang FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Id = ? AND T1.RepoId = ?", (method_id, repo_id))
    result = cursor.fetchone()
    if not result:
        return {"lang": []}
    return {"lang": result[0]}

# Endpoint to get processed time and count of solution IDs based on solution ID
@app.get("/v1/codebase_comments/processed_time_count_by_solution_id", operation_id="get_processed_time_count_by_solution_id", summary="Retrieves the total count and the most recent processed time for a specific solution, based on the provided solution ID.")
async def get_processed_time_count_by_solution_id(solution_id: int = Query(..., description="ID of the solution")):
    cursor.execute("SELECT T1.ProcessedTime, COUNT(T2.SolutionId) FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.SolutionId = ?", (solution_id,))
    result = cursor.fetchone()
    if not result:
        return {"processed_time": [], "count": []}
    return {"processed_time": result[0], "count": result[1]}

# Endpoint to get count of solution IDs based on tokenized name pattern
@app.get("/v1/codebase_comments/count_solution_ids_by_name_tokenized", operation_id="get_count_solution_ids_by_name_tokenized", summary="Retrieves the total number of unique solution IDs that match the provided tokenized name pattern. The input parameter allows for a wildcard search using the '%' symbol, enabling a flexible search for solutions based on their tokenized names.")
async def get_count_solution_ids_by_name_tokenized(name_tokenized: str = Query(..., description="Tokenized name pattern (use % for wildcard)")):
    cursor.execute("SELECT COUNT(T2.SolutionId) FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.NameTokenized LIKE ?", (name_tokenized,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get API calls and solution path based on method ID
@app.get("/v1/codebase_comments/api_calls_path_by_method_id", operation_id="get_api_calls_path_by_method_id", summary="Retrieves the API calls and corresponding solution path associated with the specified method ID. This operation allows you to obtain detailed information about a particular method's API calls and the solution path it follows.")
async def get_api_calls_path_by_method_id(method_id: int = Query(..., description="ID of the method")):
    cursor.execute("SELECT T2.ApiCalls, T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Id = ?", (method_id,))
    result = cursor.fetchone()
    if not result:
        return {"api_calls": [], "path": []}
    return {"api_calls": result[0], "path": result[1]}

# Endpoint to get count of solution IDs based on repository ID and null comments
@app.get("/v1/codebase_comments/count_solution_ids_by_repo_id_null_comments", operation_id="get_count_solution_ids_by_repo_id_null_comments", summary="Retrieves the total number of solutions in a specified repository that have no comments. The count is based on the absence of both full comments and summaries for the solutions.")
async def get_count_solution_ids_by_repo_id_null_comments(repo_id: int = Query(..., description="ID of the repository")):
    cursor.execute("SELECT COUNT(T2.SolutionId) FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.RepoId = ? AND T2.FullComment IS NULL AND T2.Summary IS NULL", (repo_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get comment format based on method ID and solution path
@app.get("/v1/codebase_comments/comment_format_by_method_id_path", operation_id="get_comment_format_by_method_id_path", summary="Retrieves the comment format (XML or non-XML) associated with a specific method identified by its unique ID and a solution path. The method's comment format is determined by evaluating the CommentIsXml field in the Method table, which is linked to the Solution table via the SolutionId field.")
async def get_comment_format_by_method_id_path(method_id: int = Query(..., description="ID of the method"), path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT CASE WHEN T2.CommentIsXml = 0 THEN 'isNotXMLFormat' WHEN T2.CommentIsXml = 1 THEN 'isXMLFormat' END format FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.Id = ? AND T1.Path = ?", (method_id, path))
    result = cursor.fetchone()
    if not result:
        return {"format": []}
    return {"format": result[0]}

# Endpoint to get distinct tasks based on tokenized name
@app.get("/v1/codebase_comments/distinct_tasks_by_name_tokenized", operation_id="get_distinct_tasks_by_name_tokenized", summary="Retrieves a list of unique task names from methods that match the provided tokenized method name. The tokenized name is used to filter the methods, and the task name is extracted from the method name using a specific pattern.")
async def get_distinct_tasks_by_name_tokenized(name_tokenized: str = Query(..., description="Tokenized name of the method")):
    cursor.execute("SELECT DISTINCT SUBSTR(SUBSTR(Name, INSTR(Name, '.') + 1), 1, INSTR(SUBSTR(Name, INSTR(Name, '.') + 1), '.') - 1) task FROM Method WHERE NameTokenized = ?", (name_tokenized,))
    result = cursor.fetchall()
    if not result:
        return {"tasks": []}
    return {"tasks": [row[0] for row in result]}

# Endpoint to get the count of distinct solution paths for a given repository URL
@app.get("/v1/codebase_comments/distinct_solution_paths_count", operation_id="get_distinct_solution_paths_count", summary="Retrieves the total number of unique solution paths associated with a specific repository. The repository is identified by its URL, which is provided as an input parameter.")
async def get_distinct_solution_paths_count(url: str = Query(..., description="URL of the repository")):
    cursor.execute("SELECT COUNT(DISTINCT T2.Path) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T1.Url = ?", (url,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the number of watchers for a repository based on solution path
@app.get("/v1/codebase_comments/repo_watchers_by_solution_path", operation_id="get_repo_watchers_by_solution_path", summary="Retrieves the total number of watchers for a repository associated with the specified solution path. The solution path is used to identify the relevant repository and its corresponding watcher count.")
async def get_repo_watchers_by_solution_path(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT T1.Watchers FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.Path = ?", (path,))
    result = cursor.fetchone()
    if not result:
        return {"watchers": []}
    return {"watchers": result[0]}

# Endpoint to get repository URLs where the solution was compiled
@app.get("/v1/codebase_comments/repo_urls_compiled_solutions", operation_id="get_repo_urls_compiled_solutions", summary="Retrieves a list of repository URLs where the solution was compiled. The operation allows filtering results based on whether the solution was compiled and limits the number of results returned. This endpoint is useful for identifying repositories where a specific solution has been compiled.")
async def get_repo_urls_compiled_solutions(was_compiled: int = Query(..., description="Whether the solution was compiled (1 for true, 0 for false)"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Url FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.WasCompiled = ? LIMIT ?", (was_compiled, limit))
    result = cursor.fetchall()
    if not result:
        return {"urls": []}
    return {"urls": [row[0] for row in result]}

# Endpoint to get distinct solution paths based on tokenized method name
@app.get("/v1/codebase_comments/distinct_solution_paths_by_method_name", operation_id="get_distinct_solution_paths_by_method_name", summary="Retrieves a list of distinct solution paths that contain a method with the specified tokenized name. The input parameter is used to filter the results based on the tokenized name of the method.")
async def get_distinct_solution_paths_by_method_name(name_tokenized: str = Query(..., description="Tokenized name of the method")):
    cursor.execute("SELECT DISTINCT T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.NameTokenized = ?", (name_tokenized,))
    result = cursor.fetchall()
    if not result:
        return {"paths": []}
    return {"paths": [row[0] for row in result]}

# Endpoint to get the count of XML comments for a given repository URL
@app.get("/v1/codebase_comments/xml_comments_count", operation_id="get_xml_comments_count", summary="Retrieves the count of XML comments associated with a specific repository. The operation filters comments based on the provided repository URL and whether they are XML comments. The result is a numerical count of XML comments that meet the specified criteria.")
async def get_xml_comments_count(url: str = Query(..., description="URL of the repository"), comment_is_xml: int = Query(..., description="Whether the comment is XML (1 for true, 0 for false)")):
    cursor.execute("SELECT COUNT(T3.CommentIsXml) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId INNER JOIN Method AS T3 ON T2.Id = T3.SolutionId WHERE T1.Url = ? AND T3.CommentIsXml = ?", (url, comment_is_xml))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct solution paths based on sampling time
@app.get("/v1/codebase_comments/distinct_solution_paths_by_sampling_time", operation_id="get_distinct_solution_paths_by_sampling_time", summary="Retrieves a list of distinct solution paths that were sampled at a specific time, with the option to limit the number of results returned. This operation is useful for analyzing unique solution paths at a given point in time.")
async def get_distinct_solution_paths_by_sampling_time(sampled_at: int = Query(..., description="Sampling time in ticks"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T1.Path FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T2.SampledAt = ? LIMIT ?", (sampled_at, limit))
    result = cursor.fetchall()
    if not result:
        return {"paths": []}
    return {"paths": [row[0] for row in result]}

# Endpoint to get repository URLs based on solution path
@app.get("/v1/codebase_comments/repo_urls_by_solution_path", operation_id="get_repo_urls_by_solution_path", summary="Retrieves the URLs of repositories associated with a specific solution path. The solution path is used to identify the relevant repositories and return their URLs.")
async def get_repo_urls_by_solution_path(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT T1.Url FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.Path = ?", (path,))
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get distinct method names based on solution path
@app.get("/v1/codebase_comments/distinct_method_names_by_solution_path", operation_id="get_distinct_method_names_by_solution_path", summary="Retrieves a list of unique method names associated with the specified solution path. This operation filters methods based on the provided solution path and returns only the distinct method names.")
async def get_distinct_method_names_by_solution_path(path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT DISTINCT T2.Name FROM Solution AS T1 INNER JOIN Method AS T2 ON T1.Id = T2.SolutionId WHERE T1.Path = ?", (path,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get repository URLs based on method summary
@app.get("/v1/codebase_comments/repo_urls_by_method_summary", operation_id="get_repo_urls_by_method_summary", summary="Retrieves repository URLs associated with a specific method summary. The method summary is used to filter the results, returning only the URLs of repositories that contain solutions with methods matching the provided summary.")
async def get_repo_urls_by_method_summary(summary: str = Query(..., description="Summary of the method")):
    cursor.execute("SELECT T1.Url FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId INNER JOIN Method AS T3 ON T2.Id = T3.SolutionId WHERE T3.Summary = ?", (summary,))
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get distinct star counts for repositories with a specific solution path
@app.get("/v1/codebase_comments/distinct_stars_by_solution_path", operation_id="get_distinct_stars", summary="Retrieves the unique star counts for repositories associated with a specific solution path. The solution path is used to filter the repositories and return only the distinct star counts.")
async def get_distinct_stars(solution_path: str = Query(..., description="Path of the solution")):
    cursor.execute("SELECT DISTINCT T1.Stars FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId WHERE T2.Path = ?", (solution_path,))
    result = cursor.fetchall()
    if not result:
        return {"stars": []}
    return {"stars": [row[0] for row in result]}

# Endpoint to get the count of distinct languages used in methods for a repository with a specific URL
@app.get("/v1/codebase_comments/count_distinct_languages_by_repo_url", operation_id="get_count_distinct_languages", summary="Retrieves the number of unique programming languages used in the methods of a specific repository. The repository is identified by its URL.")
async def get_count_distinct_languages(repo_url: str = Query(..., description="URL of the repository")):
    cursor.execute("SELECT COUNT(DISTINCT T3.Lang) FROM Repo AS T1 INNER JOIN Solution AS T2 ON T1.Id = T2.RepoId INNER JOIN Method AS T3 ON T2.Id = T3.SolutionId WHERE T1.Url = ?", (repo_url,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

api_calls = [
    "/v1/codebase_comments/method_count_lang_comment_is_xml?lang=en&comment_is_xml=1",
    "/v1/codebase_comments/method_task_from_name?name_tokenized=online%20median%20filter%20test%20median%20window%20filling",
    "/v1/codebase_comments/latest_sampled_methods",
    "/v1/codebase_comments/max_forks_repo",
    "/v1/codebase_comments/repo_with_most_solutions",
    "/v1/codebase_comments/max_stars_repo",
    "/v1/codebase_comments/latest_processed_solutions",
    "/v1/codebase_comments/max_watchers_repo_processed_time",
    "/v1/codebase_comments/repo_url_by_solution_path?path=nofear_MaraMara.sln",
    "/v1/codebase_comments/latest_processed_repos_compilation_status",
    "/v1/codebase_comments/distinct_method_names?path=maravillas_linq-to-delicious",
    "/v1/codebase_comments/top_repo_by_solution_count?min_stars=6000&max_stars=9000&was_compiled=0",
    "/v1/codebase_comments/api_calls_from_repo?url=https://github.com/wallerdev/htmlsharp.git",
    "/v1/codebase_comments/count_distinct_solution_paths",
    "/v1/codebase_comments/average_processed_time?url=https://github.com/zphingphong/DiscardCustomerApp.git",
    "/v1/codebase_comments/full_comments?path=bmatzelle_niniSourceNini.sln&name_tokenized=alias%20text%20add%20alias",
    "/v1/codebase_comments/api_calls_from_solution?path=mauriciodeamorim_tdd.encontro2Tdd.Encontro2.sln",
    "/v1/codebase_comments/count_solution_paths?url=https://github.com/jeffdik/tachy.git&was_compiled=0",
    "/v1/codebase_comments/distinct_processed_times_languages?name_tokenized=about%20box1%20dispose",
    "/v1/codebase_comments/sampled_times?path=maxild_playgroundPlayground.sln&name=GitHubRepo.Cli.GitHubClientWrapper.GetReleases",
    "/v1/codebase_comments/solution_language_by_path?path=opendns_diagnosticappwindowsOpenDnsDiagnostic.sln",
    "/v1/codebase_comments/method_comment_is_xml?name=HtmlSharp.HtmlParser.Feed",
    "/v1/codebase_comments/method_name_by_summary?summary=Write%20a%20command%20to%20the%20log",
    "/v1/codebase_comments/count_repos_with_max_stars",
    "/v1/codebase_comments/distinct_paths_repos_with_max_stars",
    "/v1/codebase_comments/repo_url_by_solution_id?solution_id=12",
    "/v1/codebase_comments/count_repos_forks_compiled?forks=1000&was_compiled=1",
    "/v1/codebase_comments/compare_solution_popularity?solution_id_18=18&solution_id_19=19",
    "/v1/codebase_comments/count_repos_latest_processed_time_compiled?was_compiled=1",
    "/v1/codebase_comments/distinct_processed_times?path=jeffdik_tachysrcTachy.sln",
    "/v1/codebase_comments/solution_paths_by_repo_url?repo_url=https://github.com/maxild/playground.git",
    "/v1/codebase_comments/repo_count_by_processed_time_and_stars?processed_time=636439500080712000&stars=200",
    "/v1/codebase_comments/solution_ids_by_forks?limit=3",
    "/v1/codebase_comments/forks_percentage_difference?solution_id_1=18&solution_id_2=19",
    "/v1/codebase_comments/method_language?method_name=PixieTests.SqlConnectionLayerTests.TestSqlCreateGuidColumn",
    "/v1/codebase_comments/method_full_comment?method_name=DE2_UE_Fahrradkurier.de2_uebung_fahrradkurierDataSet1TableAdapters.TableAdapterManager.UpdateInsertedRows",
    "/v1/codebase_comments/method_summaries?method_name=Castle.MonoRail.Framework.Test.StubViewComponentContext.RenderSection",
    "/v1/codebase_comments/method_name_tokenized?name=Supay.Irc.Messages.KnockMessage.GetTokens",
    "/v1/codebase_comments/repo_stars_by_solution_id?solution_id=45997",
    "/v1/codebase_comments/count_solutions_by_repo_stars?stars=8094",
    "/v1/codebase_comments/solution_path_by_method_name?method_name=IQ.Data.DbQueryProvider.CanBeEvaluatedLocally",
    "/v1/codebase_comments/solution_processed_time_by_method_name_tokenized?name_tokenized=interp%20parser%20expr",
    "/v1/codebase_comments/repo_id_by_method_name?method_name=SCore.Poisson.ngtIndex",
    "/v1/codebase_comments/count_solutions_by_repo_watchers?watchers=8094",
    "/v1/codebase_comments/repo_url_with_most_solutions",
    "/v1/codebase_comments/count_solutions_by_repo_forks?forks=1445",
    "/v1/codebase_comments/most_common_solution_path_by_lang?lang=zh-cn",
    "/v1/codebase_comments/repo_watchers_by_solution_id?solution_id=338082",
    "/v1/codebase_comments/repo_count_by_stars_and_compilation?stars=189&was_compiled=0",
    "/v1/codebase_comments/distinct_repo_ids_by_method_name?method_name=Kalibrasi.Data.EntityClasses.THistoryJadwalEntity.GetSingleTjadwal",
    "/v1/codebase_comments/distinct_solution_paths_by_summary?summary=Refetches%20the%20Entity%20FROM%20the%20persistent%20storage.%20Refetch%20is%20used%20to%20re-load%20an%20Entity%20which%20is%20marked%20%22Out-of-sync%22,%20due%20to%20a%20save%20action.%20Refetching%20an%20empty%20Entity%20has%20no%20effect.",
    "/v1/codebase_comments/distinct_solution_paths_by_lang?lang=sw",
    "/v1/codebase_comments/watcher_percentage_difference?solution_id_1=83855&solution_id_2=1502",
    "/v1/codebase_comments/percentage_difference_stars?id1=51424&id2=167053",
    "/v1/codebase_comments/percentage_difference_forks?id1=53546&id2=1502",
    "/v1/codebase_comments/distinct_method_names_by_processed_time?processed_time=636449700980488000",
    "/v1/codebase_comments/count_repo_ids_by_url?url=https://github.com/derickbailey/presentations-and-training.git",
    "/v1/codebase_comments/total_processed_time_max_forks",
    "/v1/codebase_comments/distinct_method_ids_by_repo_id_lang?repo_id=1093&lang=en",
    "/v1/codebase_comments/distinct_solution_paths_by_url?url=https://github.com/ecoffey/Bebop.git",
    "/v1/codebase_comments/distinct_repo_ids_by_lang?lang=ro",
    "/v1/codebase_comments/repo_ids_by_method_name?name_tokenized=crc%20parameters%20get%20hash%20code",
    "/v1/codebase_comments/count_solution_ids_by_path?path=maravillas_linq-to-delicious%5Ctasty.sln",
    "/v1/codebase_comments/solution_ids_by_processed_time?processed_time=636430969128176000",
    "/v1/codebase_comments/repo_urls_by_max_processed_time",
    "/v1/codebase_comments/distinct_solution_ids_by_forks_watchers",
    "/v1/codebase_comments/forks_percentage_by_solution_id?solution_id=104086",
    "/v1/codebase_comments/repo_ids_by_max_stars",
    "/v1/codebase_comments/forks_by_solution_id?solution_id=35",
    "/v1/codebase_comments/solution_ids_by_max_watchers",
    "/v1/codebase_comments/method_count_compilation_status?solution_id=1",
    "/v1/codebase_comments/percentage_needs_compilation?comment_is_xml=1",
    "/v1/codebase_comments/method_names_by_solution_path?path=wallerdev_htmlsharpHtmlSharp.sln",
    "/v1/codebase_comments/percentage_repos_with_stars?min_stars=2000",
    "/v1/codebase_comments/repo_id_max_forks_by_stars?stars=21",
    "/v1/codebase_comments/percentage_methods_by_language_xml_comments?lang=en&comment_is_xml=1",
    "/v1/codebase_comments/solution_path_by_tokenized_method_name?name_tokenized=html%20parser%20feed",
    "/v1/codebase_comments/solution_ids_by_repo_forks?forks=238",
    "/v1/codebase_comments/tokenized_names_by_lang_compiled?lang=en&was_compiled=0",
    "/v1/codebase_comments/solution_paths_by_full_comment?full_comment=Feeds%20data%20into%20the%20parser",
    "/v1/codebase_comments/solution_count_by_repo_xml_comment?repo_id=3&comment_is_xml=1",
    "/v1/codebase_comments/percentage_not_compiled_by_lang?lang=en",
    "/v1/codebase_comments/distinct_repo_count_by_stars_forks_ratio?stars_forks_ratio=3",
    "/v1/codebase_comments/task_name_by_method_id?method_id=2",
    "/v1/codebase_comments/comment_format_by_method_id?method_id=8",
    "/v1/codebase_comments/repo_url_max_watchers",
    "/v1/codebase_comments/distinct_tasks_by_lang?lang=cs",
    "/v1/codebase_comments/solution_path_by_method_id?method_id=3",
    "/v1/codebase_comments/method_lang_by_method_id_repo_id?method_id=28&repo_id=3",
    "/v1/codebase_comments/processed_time_count_by_solution_id?solution_id=1",
    "/v1/codebase_comments/count_solution_ids_by_name_tokenized?name_tokenized=query%20language%25",
    "/v1/codebase_comments/api_calls_path_by_method_id?method_id=10",
    "/v1/codebase_comments/count_solution_ids_by_repo_id_null_comments?repo_id=150",
    "/v1/codebase_comments/comment_format_by_method_id_path?method_id=50&path=managedfusion_managedfusionManagedFusion.sln",
    "/v1/codebase_comments/distinct_tasks_by_name_tokenized?name_tokenized=string%20extensions%20to%20pascal%20case",
    "/v1/codebase_comments/distinct_solution_paths_count?url=https://github.com/jeffdik/tachy.git",
    "/v1/codebase_comments/repo_watchers_by_solution_path?path=maff_se3ue7US7.sln",
    "/v1/codebase_comments/repo_urls_compiled_solutions?was_compiled=1&limit=5",
    "/v1/codebase_comments/distinct_solution_paths_by_method_name?name_tokenized=matrix%20multiply",
    "/v1/codebase_comments/xml_comments_count?url=https://github.com/dogeth/vss2git.git&comment_is_xml=1",
    "/v1/codebase_comments/distinct_solution_paths_by_sampling_time?sampled_at=636431758961741000&limit=5",
    "/v1/codebase_comments/repo_urls_by_solution_path?path=joeyrobert_bloomfilterDataTypes.BloomFilter.sln",
    "/v1/codebase_comments/distinct_method_names_by_solution_path?path=graffen_NLog.Targets.SyslogsrcNLog.Targets.Syslog.sln",
    "/v1/codebase_comments/repo_urls_by_method_summary?summary=A%20test%20for%20Decompose",
    "/v1/codebase_comments/distinct_stars_by_solution_path?solution_path=ninject_NinjectNinject.sln",
    "/v1/codebase_comments/count_distinct_languages_by_repo_url?repo_url=https://github.com/managedfusion/managedfusion.git"
]
