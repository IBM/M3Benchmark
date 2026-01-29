from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/authors/authors.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of authors based on affiliation
@app.get("/v1/authors/author_count_given_affiliation", operation_id="get_author_count", summary="Retrieves the total number of authors associated with a specified affiliation.")
async def get_author_count(affiliation: str = Query(..., description="Affiliation of the author")):
    cursor.execute("SELECT COUNT(Id) FROM Author WHERE Affiliation = ?", (affiliation,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get keywords from a paper based on its title
@app.get("/v1/authors/paper_keywords_by_title", operation_id="get_paper_keywords", summary="Retrieve the keywords associated with a specific paper by providing its title. The operation searches for the paper using the supplied title and returns the corresponding keywords.")
async def get_paper_keywords(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT Keyword FROM Paper WHERE Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get paper titles from a journal based on its full name and year
@app.get("/v1/authors/paper_titles_by_journal_and_year", operation_id="get_paper_titles", summary="Retrieve the titles of papers published in a specific journal during a given year. The operation requires the full name of the journal and the year of publication as input parameters.")
async def get_paper_titles(full_name: str = Query(..., description="Full name of the journal"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT T2.Title FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.FullName = ? AND T2.Year = ?", (full_name, year))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of papers in a journal based on its full name
@app.get("/v1/authors/paper_count_by_journal", operation_id="get_paper_count", summary="Retrieves the total number of papers published in a specific journal, identified by its full name.")
async def get_paper_count(full_name: str = Query(..., description="Full name of the journal")):
    cursor.execute("SELECT COUNT(T2.Id) FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.FullName = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the homepage of a journal based on a paper title
@app.get("/v1/authors/journal_homepage_by_paper_title", operation_id="get_journal_homepage", summary="Retrieves the homepage URL of the journal that published a paper with the specified title. The operation searches for the paper by its title and returns the homepage of the corresponding journal.")
async def get_journal_homepage(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T1.HomePage FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T2.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"homepage": []}
    return {"homepage": result[0]}

# Endpoint to get the count of papers in a journal based on year and full name
@app.get("/v1/authors/paper_count_by_year_and_journal", operation_id="get_paper_count_by_year", summary="Retrieves the total number of papers published in a specific journal during a given year. The operation requires the year of publication and the full name of the journal as input parameters.")
async def get_paper_count_by_year(year: int = Query(..., description="Year of the paper"), full_name: str = Query(..., description="Full name of the journal")):
    cursor.execute("SELECT COUNT(T2.Id) FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T2.Year = ? AND T1.FullName = ?", (year, full_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of papers in a journal based on year and short name
@app.get("/v1/authors/paper_count_by_year_and_short_name", operation_id="get_paper_count_by_year_and_short_name", summary="Retrieves the total number of papers published in a specific journal during a given year. The operation requires the year of publication and the short name of the journal as input parameters.")
async def get_paper_count_by_year_and_short_name(year: int = Query(..., description="Year of the paper"), short_name: str = Query(..., description="Short name of the journal")):
    cursor.execute("SELECT COUNT(T2.Id) FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T2.Year = ? AND T1.ShortName = ?", (year, short_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of authors for a paper based on its title
@app.get("/v1/authors/author_count_by_paper_title", operation_id="get_author_count_by_paper_title", summary="Retrieves the total number of authors associated with a specific paper, identified by its title.")
async def get_author_count_by_paper_title(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT COUNT(T1.AuthorId) FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T2.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of authors for a paper based on its title
@app.get("/v1/authors/author_names_by_paper_title", operation_id="get_author_names_by_paper_title", summary="Retrieves the names of authors associated with a specific paper, based on the provided paper title.")
async def get_author_names_by_paper_title(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T1.Name FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T2.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of authors for a paper based on affiliation and title
@app.get("/v1/authors/author_count_by_affiliation_and_title", operation_id="get_author_count_by_affiliation_and_title", summary="Retrieves the total number of authors associated with a specific paper, based on the provided affiliation and paper title. This operation is useful for understanding the authorship distribution across different affiliations and papers.")
async def get_author_count_by_affiliation_and_title(affiliation: str = Query(..., description="Affiliation of the author"), title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT COUNT(T1.AuthorId) FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T1.Affiliation = ? AND T2.Title = ?", (affiliation, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get paper titles by author name
@app.get("/v1/authors/paper_titles_by_author", operation_id="get_paper_titles_by_author", summary="Retrieves a list of paper titles associated with the specified author. The author's name is used to filter the results.")
async def get_paper_titles_by_author(name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T2.Title FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T1.Name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get author names by paper title and affiliation
@app.get("/v1/authors/author_names_by_paper_title_and_affiliation", operation_id="get_author_names_by_paper_title_and_affiliation", summary="Retrieves the names of authors who have written a paper with the specified title and are affiliated with the provided organization. The response includes a list of author names that meet the given criteria.")
async def get_author_names_by_paper_title_and_affiliation(title: str = Query(..., description="Title of the paper"), affiliation: str = Query(..., description="Affiliation of the author")):
    cursor.execute("SELECT T1.Name FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T2.Title = ? AND T1.Affiliation = ?", (title, affiliation))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get paper titles by author name and year
@app.get("/v1/authors/paper_titles_by_author_and_year", operation_id="get_paper_titles_by_author_and_year", summary="Retrieves a list of paper titles authored by a specific individual in a given year. The operation requires the author's name and the year of publication as input parameters.")
async def get_paper_titles_by_author_and_year(name: str = Query(..., description="Name of the author"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT T2.Title FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T1.Name = ? AND T2.Year = ?", (name, year))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of paper titles by author name and year
@app.get("/v1/authors/count_paper_titles_by_author_and_year", operation_id="get_count_paper_titles_by_author_and_year", summary="Retrieves the total number of distinct paper titles authored by a specific individual in a given year. The operation requires the author's name and the year of publication as input parameters.")
async def get_count_paper_titles_by_author_and_year(name: str = Query(..., description="Name of the author"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT COUNT(T2.Title) FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T1.Name = ? AND T2.Year = ?", (name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average number of papers per year for a journal within a year range
@app.get("/v1/authors/average_papers_per_year_for_journal", operation_id="get_average_papers_per_year_for_journal", summary="Retrieves the average number of papers published per year for a specific journal within a given year range. The calculation is based on the count of distinct papers published in the journal during the specified range, divided by the number of years in that range.")
async def get_average_papers_per_year_for_journal(full_name: str = Query(..., description="Full name of the journal"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT CAST(COUNT(T2.Id) AS REAL) / COUNT(DISTINCT T2.Year) FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.FullName = ? AND T2.Year BETWEEN ? AND ?", (full_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the difference in the number of papers between two journals
@app.get("/v1/authors/difference_in_paper_count_between_journals", operation_id="get_difference_in_paper_count_between_journals", summary="Retrieves the difference in the number of papers published between two specified journals. The operation compares the total paper count of the first journal with that of the second journal and returns the difference.")
async def get_difference_in_paper_count_between_journals(journal1: str = Query(..., description="Full name of the first journal"), journal2: str = Query(..., description="Full name of the second journal")):
    cursor.execute("SELECT SUM(CASE WHEN T1.FullName = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.FullName = ? THEN 1 ELSE 0 END) AS DIFF FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId", (journal1, journal2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of journals with a specific name pattern
@app.get("/v1/authors/count_journals_by_name_pattern", operation_id="get_count_journals_by_name_pattern", summary="Retrieves the total number of journals whose names contain a specified pattern. The pattern can include wildcard characters to broaden the search scope. This operation is useful for determining the prevalence of journals with certain naming conventions or themes.")
async def get_count_journals_by_name_pattern(name_pattern: str = Query(..., description="Pattern to match in the journal name (use %% for wildcard)")):
    cursor.execute("SELECT COUNT(Id) FROM Journal WHERE FullName LIKE ?", (name_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get paper and author IDs by affiliation pattern
@app.get("/v1/authors/paper_author_ids_by_affiliation", operation_id="get_paper_author_ids_by_affiliation", summary="Retrieves paper and author IDs for authors affiliated with institutions that match the specified pattern. The affiliation pattern supports wildcard matching using double percent signs (%%).")
async def get_paper_author_ids_by_affiliation(affiliation_pattern: str = Query(..., description="Pattern to match in the affiliation (use %% for wildcard)")):
    cursor.execute("SELECT PaperId, AuthorId FROM PaperAuthor WHERE Affiliation LIKE ?", (affiliation_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [{"paper_id": row[0], "author_id": row[1]} for row in result]}

# Endpoint to get conference names by homepage URL
@app.get("/v1/authors/conference_names_by_homepage", operation_id="get_conference_names_by_homepage", summary="Retrieves the short and full names of a conference based on its homepage URL. The provided homepage URL is used to identify the conference and return its corresponding names.")
async def get_conference_names_by_homepage(homepage: str = Query(..., description="Homepage URL of the conference")):
    cursor.execute("SELECT ShortName, FullName FROM Conference WHERE HomePage = ?", (homepage,))
    result = cursor.fetchall()
    if not result:
        return {"conferences": []}
    return {"conferences": [{"short_name": row[0], "full_name": row[1]} for row in result]}

# Endpoint to get author IDs by name
@app.get("/v1/authors/author_ids_by_name", operation_id="get_author_ids_by_name", summary="Retrieves the unique identifiers of authors who match the provided name. The operation searches for authors with the specified name and returns their corresponding identifiers.")
async def get_author_ids_by_name(name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT AuthorId FROM PaperAuthor WHERE Name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"author_ids": []}
    return {"author_ids": [row[0] for row in result]}

# Endpoint to get distinct paper titles and conference short names based on a range of conference IDs
@app.get("/v1/authors/paper_conference_titles", operation_id="get_paper_conference_titles", summary="Retrieves unique paper titles and corresponding conference names for conferences within a specified range of IDs. The range is defined by the minimum and maximum conference IDs provided as input parameters.")
async def get_paper_conference_titles(min_conference_id: int = Query(..., description="Minimum conference ID"), max_conference_id: int = Query(..., description="Maximum conference ID")):
    cursor.execute("SELECT DISTINCT T1.Title, T2.ShortName FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.ConferenceId BETWEEN ? AND ?", (min_conference_id, max_conference_id))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get the latest paper title and journal homepage
@app.get("/v1/authors/latest_paper_journal", operation_id="get_latest_paper_journal", summary="Retrieves the title of the most recent paper and the homepage of the corresponding journal.")
async def get_latest_paper_journal():
    cursor.execute("SELECT T1.Title, T2.HomePage FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id ORDER BY T1.Year DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"paper": []}
    return {"paper": result}

# Endpoint to get paper titles and conference short names based on year and author name
@app.get("/v1/authors/paper_conference_by_year_author", operation_id="get_paper_conference_by_year_author", summary="Retrieves the titles of papers and the corresponding conference short names for a given year and author. The author's name is used to filter papers where the author's name starts with the provided string. The year parameter is used to further narrow down the results to papers published in the specified year.")
async def get_paper_conference_by_year_author(year: int = Query(..., description="Year of the paper"), author_name: str = Query(..., description="Author name starting with the given string")):
    cursor.execute("SELECT T1.Title, T3.ShortName FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Conference AS T3 ON T1.ConferenceId = T3.Id WHERE T1.Year = ? AND T2.Name LIKE ?", (year, author_name + '%'))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": result}

# Endpoint to get the count of papers and conference homepage based on year range and conference ID
@app.get("/v1/authors/paper_count_conference_homepage", operation_id="get_paper_count_conference_homepage", summary="Retrieves the total number of papers and the conference homepage for a given year range and conference. The operation requires a minimum year, maximum year, and conference ID as input parameters. The output includes the count of papers and the homepage URL of the specified conference.")
async def get_paper_count_conference_homepage(min_year: int = Query(..., description="Minimum year"), max_year: int = Query(..., description="Maximum year"), conference_id: int = Query(..., description="Conference ID")):
    cursor.execute("SELECT COUNT(T2.ConferenceId), T1.HomePage FROM Conference AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.ConferenceId WHERE T2.Year BETWEEN ? AND ? AND T2.ConferenceId = ?", (min_year, max_year, conference_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0], "homepage": result[1]}

# Endpoint to get distinct paper titles based on affiliation and year
@app.get("/v1/authors/paper_titles_by_affiliation_year", operation_id="get_paper_titles_by_affiliation_year", summary="Retrieves a list of unique paper titles associated with a specific affiliation and year. The affiliation and year are used to filter the results, providing a targeted set of paper titles.")
async def get_paper_titles_by_affiliation_year(affiliation: str = Query(..., description="Affiliation of the author"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT DISTINCT T2.Title FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T1.Affiliation = ? AND T2.Year = ?", (affiliation, year))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get author names and journal short names based on journal ID range and paper title
@app.get("/v1/authors/author_journal_by_journal_id_title", operation_id="get_author_journal_by_journal_id_title", summary="Retrieves the names of authors and corresponding journal short names for papers that fall within a specified journal ID range and contain a given keyword in their title.")
async def get_author_journal_by_journal_id_title(min_journal_id: int = Query(..., description="Minimum journal ID"), max_journal_id: int = Query(..., description="Maximum journal ID"), title_keyword: str = Query(..., description="Keyword in the paper title")):
    cursor.execute("SELECT T2.Name, T3.ShortName FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Journal AS T3 ON T1.JournalId = T3.Id WHERE T1.JournalId BETWEEN ? AND ? AND T1.Title LIKE ?", (min_journal_id, max_journal_id, '%' + title_keyword + '%'))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get distinct author names based on conference ID and journal ID
@app.get("/v1/authors/author_names_by_conference_journal", operation_id="get_author_names_by_conference_journal", summary="Retrieves a list of unique author names who have published papers in a specific conference, excluding those with a journal ID equal to or greater than the provided maximum journal ID.")
async def get_author_names_by_conference_journal(conference_id: int = Query(..., description="Conference ID"), max_journal_id: int = Query(..., description="Maximum journal ID")):
    cursor.execute("SELECT DISTINCT T2.Name FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.ConferenceId = ? AND T1.JournalId < ?", (conference_id, max_journal_id))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get distinct paper titles and author IDs based on year and conference ID
@app.get("/v1/authors/paper_titles_author_ids_by_year_conference", operation_id="get_paper_titles_author_ids_by_year_conference", summary="Retrieves unique paper titles and their corresponding author IDs published in a specific year and conference. The year and maximum conference ID are used to filter the results.")
async def get_paper_titles_author_ids_by_year_conference(year: int = Query(..., description="Year of the paper"), max_conference_id: int = Query(..., description="Maximum conference ID")):
    cursor.execute("SELECT DISTINCT T1.Title, T2.AuthorId FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Year = ? AND T1.ConferenceId < ?", (year, max_conference_id))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": result}

# Endpoint to get paper IDs based on conference homepage
@app.get("/v1/authors/paper_ids_by_conference_homepage", operation_id="get_paper_ids_by_conference_homepage", summary="Retrieves a list of paper IDs associated with a conference whose homepage contains the specified keyword. The keyword is case-insensitive and can appear anywhere within the homepage URL.")
async def get_paper_ids_by_conference_homepage(homepage_keyword: str = Query(..., description="Keyword in the conference homepage")):
    cursor.execute("SELECT T1.Id FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T2.HomePage LIKE ?", ('%' + homepage_keyword + '%',))
    result = cursor.fetchall()
    if not result:
        return {"paper_ids": []}
    return {"paper_ids": result}

# Endpoint to get journal homepages and author IDs based on year range and paper title
@app.get("/v1/authors/journal_homepages_author_ids_by_year_title", operation_id="get_journal_homepages_author_ids_by_year_title", summary="Retrieves the homepages of journals and the corresponding author IDs for papers published within a specified year range and containing a given keyword in their titles.")
async def get_journal_homepages_author_ids_by_year_title(min_year: int = Query(..., description="Minimum year"), max_year: int = Query(..., description="Maximum year"), title_keyword: str = Query(..., description="Keyword in the paper title")):
    cursor.execute("SELECT T3.HomePage, T2.AuthorId FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Journal AS T3 ON T1.JournalId = T3.Id WHERE T1.Year BETWEEN ? AND ? AND T1.Title LIKE ?", (min_year, max_year, '%' + title_keyword + '%'))
    result = cursor.fetchall()
    if not result:
        return {"journals": []}
    return {"journals": result}

# Endpoint to get distinct authors and their affiliations from papers in a specific journal and year
@app.get("/v1/authors/distinct_authors_by_journal_year", operation_id="get_distinct_authors", summary="Retrieves a list of unique authors and their respective affiliations from papers published in a specific journal during a given year. The operation filters out authors without affiliations.")
async def get_distinct_authors(journal_id: int = Query(..., description="Journal ID"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT DISTINCT T2.AuthorId, T2.Affiliation FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.JournalId = ? AND T1.Year = ? AND T2.Affiliation IS NOT NULL", (journal_id, year))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get the percentage of papers in a specific journal range with a specific conference ID and short name pattern
@app.get("/v1/authors/percentage_papers_by_conference_journal_range", operation_id="get_percentage_papers", summary="Retrieves the percentage of papers associated with a specific conference within a defined range of journals that match a given short name pattern. The calculation is based on the total number of papers in the specified journal range.")
async def get_percentage_papers(conference_id: int = Query(..., description="Conference ID"), min_journal_id: int = Query(..., description="Minimum Journal ID"), max_journal_id: int = Query(..., description="Maximum Journal ID"), short_name_pattern: str = Query(..., description="Short name pattern (e.g., 'A%')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.ConferenceId = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.ConferenceId) FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id WHERE T1.JournalId BETWEEN ? AND ? AND T2.ShortName LIKE ?", (conference_id, min_journal_id, max_journal_id, short_name_pattern))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of papers published in a specific year range
@app.get("/v1/authors/percentage_papers_by_year_range", operation_id="get_percentage_papers_by_year", summary="Get the percentage of papers published in a specific year range")
async def get_percentage_papers_by_year(year: int = Query(..., description="Specific year"), min_year: int = Query(..., description="Minimum year"), max_year: int = Query(..., description="Maximum year")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Year = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(Id) FROM Paper WHERE Year BETWEEN ? AND ?", (year, min_year, max_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get conference details by short name
@app.get("/v1/authors/conference_details_by_short_name", operation_id="get_conference_details", summary="Retrieves the full name and homepage of a conference based on its short name. The short name is a required input parameter.")
async def get_conference_details(short_name: str = Query(..., description="Short name of the conference")):
    cursor.execute("SELECT FullName, Homepage FROM Conference WHERE ShortName = ?", (short_name,))
    result = cursor.fetchone()
    if not result:
        return {"conference": []}
    return {"conference": result}

# Endpoint to get the top affiliation by count of authors
@app.get("/v1/authors/top_affiliation_by_author_count", operation_id="get_top_affiliation", summary="Retrieves the affiliation with the highest number of associated authors, considering only the provided affiliations. The affiliations are specified as input parameters.")
async def get_top_affiliation(affiliation1: str = Query(..., description="First affiliation"), affiliation2: str = Query(..., description="Second affiliation")):
    cursor.execute("SELECT Affiliation FROM Author WHERE Affiliation IN (?, ?) GROUP BY Affiliation ORDER BY COUNT(Id) DESC LIMIT 1", (affiliation1, affiliation2))
    result = cursor.fetchone()
    if not result:
        return {"affiliation": []}
    return {"affiliation": result[0]}

# Endpoint to get the percentage of authors without an affiliation
@app.get("/v1/authors/percentage_authors_without_affiliation", operation_id="get_percentage_authors_without_affiliation", summary="Retrieves the proportion of authors in the database who do not have an affiliation associated with their profile. This endpoint calculates the ratio by dividing the count of authors without affiliations by the total number of authors.")
async def get_percentage_authors_without_affiliation():
    cursor.execute("SELECT CAST(SUM(CASE WHEN Affiliation IS NULL THEN 1 ELSE 0 END) AS REAL) / COUNT(*) FROM Author")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get author details by affiliation
@app.get("/v1/authors/author_details_by_affiliation", operation_id="get_author_details", summary="Retrieves the name and unique identifier of an author based on their affiliation. The affiliation is used to filter the author records and return the requested details.")
async def get_author_details(affiliation: str = Query(..., description="Affiliation of the author")):
    cursor.execute("SELECT Name, id FROM Author WHERE Affiliation = ?", (affiliation,))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get the top paper by author count
@app.get("/v1/authors/top_paper_by_author_count", operation_id="get_top_paper", summary="Retrieves the paper with the highest number of authors. The paper's title, publication year, and the full name of the journal it was published in are returned.")
async def get_top_paper():
    cursor.execute("SELECT T1.Id, T1.Title, T1.Year, T3.FullName FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Journal AS T3 ON T1.JournalId = T3.Id GROUP BY T2.AuthorId ORDER BY COUNT(T2.AuthorId) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"paper": []}
    return {"paper": result}

# Endpoint to get paper details by paper ID
@app.get("/v1/authors/paper_details_by_id", operation_id="get_paper_details", summary="Retrieves the distinct title, year, short name of the conference, and author name for a specific paper identified by its unique ID. The operation requires the paper ID as input and returns the requested details if the paper exists in the database.")
async def get_paper_details(paper_id: int = Query(..., description="Paper ID")):
    cursor.execute("SELECT DISTINCT T1.Title, T1.Year, T3.ShortName, T2.Name FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Conference AS T3 ON T1.ConferenceId = T3.Id WHERE T1.Id = ?", (paper_id,))
    result = cursor.fetchall()
    if not result:
        return {"paper_details": []}
    return {"paper_details": result}

# Endpoint to get author names and paper IDs based on paper title and affiliation
@app.get("/v1/authors/author_names_by_paper_title_affiliation", operation_id="get_author_names", summary="Retrieves the names of authors and their corresponding paper IDs for a given paper title and affiliation. The affiliation is set to 'Microsoft Research, USA'.")
async def get_author_names(title: str = Query(..., description="Title of the paper"), affiliation: str = Query(..., description="Affiliation of the author")):
    cursor.execute("SELECT T2.Name, T1.Id FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Title = ? AND T2.Affiliation = ?", (title, affiliation))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get author details and journal information based on paper title
@app.get("/v1/authors/author_journal_info_by_paper_title", operation_id="get_author_journal_info", summary="Retrieves the name, affiliation, and journal details of the author(s) associated with a specific paper. The paper is identified by its title.")
async def get_author_journal_info(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T2.Name, T2.Affiliation, T3.ShortName, T3.FullName FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Journal AS T3 ON T1.JournalId = T3.Id WHERE T1.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"info": []}
    return {"info": result}

# Endpoint to get paper details based on author ID and affiliation
@app.get("/v1/authors/paper_details_by_author_id_affiliation", operation_id="get_paper_details_by_author", summary="Retrieves the title, publication year, and keywords of papers authored by a specific researcher from the 'Scientific Computing and Imaging Institute, University of Utah, UT 84112, USA'. The author is identified by their unique ID, and the affiliation is used to ensure the correct author is selected. This operation is useful for obtaining detailed information about an author's publications.")
async def get_paper_details_by_author(author_id: int = Query(..., description="Author ID"), affiliation: str = Query(..., description="Affiliation of the author")):
    cursor.execute("SELECT T2.Title, T2.Year, T2.Keyword FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T1.AuthorId = ? AND T1.Affiliation = ?", (author_id, affiliation))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": result}

# Endpoint to get the difference in the number of papers published in two different years for a specific journal
@app.get("/v1/authors/paper_count_diff_by_years_journal", operation_id="get_paper_count_diff", summary="Retrieve the difference in the total number of papers published in two specified years for a given journal. This operation compares the paper count from the first year to the second year, providing a numerical difference as the result.")
async def get_paper_count_diff(year1: int = Query(..., description="First year"), year2: int = Query(..., description="Second year"), short_name: str = Query(..., description="Short name of the journal")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Year = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.Year = ? THEN 1 ELSE 0 END) AS DIFF FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.ShortName = ?", (year1, year2, short_name))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get distinct journal details based on year and journal IDs
@app.get("/v1/authors/journal_details_by_year_journal_ids", operation_id="get_journal_details", summary="Retrieve unique journal details, excluding specific journals, from a given year. The operation returns a limited set of journal IDs, short names, and full names. The year, excluded journal IDs, and result limit are provided as input parameters.")
async def get_journal_details(year: int = Query(..., description="Year of the paper"), journal_id1: int = Query(..., description="First journal ID to exclude"), journal_id2: int = Query(..., description="Second journal ID to exclude"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T2.JournalId, T1.ShortName, T1.FullName FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T2.Year = ? AND T2.JournalId != ? AND T2.JournalId != ? LIMIT ?", (year, journal_id1, journal_id2, limit))
    result = cursor.fetchall()
    if not result:
        return {"journals": []}
    return {"journals": result}

# Endpoint to get paper titles, author names, and conference full names based on conference short name and year range
@app.get("/v1/authors/paper_author_conference_by_shortname_year_range", operation_id="get_paper_author_conference", summary="Retrieves a list of paper titles, author names, and corresponding conference full names for a specific conference, based on its short name and a defined year range. The start and end years of the range are provided as input parameters.")
async def get_paper_author_conference(short_name: str = Query(..., description="Short name of the conference"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT T2.title, T3.name, T1.FullName FROM Conference AS T1 INNER JOIN Paper AS T2 ON T1.id = T2.ConferenceId INNER JOIN PaperAuthor AS T3 ON T1.id = T3.PaperId WHERE T1.ShortName = ? AND T2.Year BETWEEN ? AND ?", (short_name, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get conference full names based on paper title
@app.get("/v1/authors/conference_fullname_by_paper_title", operation_id="get_conference_fullname", summary="Retrieves the full names of conferences where a paper with the specified title was presented. The operation uses the paper's title to search for matching conference records and returns the full names of the corresponding conferences.")
async def get_conference_fullname(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T2.FullName FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"conferences": []}
    return {"conferences": result}

# Endpoint to get the homepage of a conference based on the paper title
@app.get("/v1/authors/conference_homepage_by_paper_title", operation_id="get_conference_homepage", summary="Retrieves the homepage URL of the conference associated with a given paper title. The operation searches for the paper by its title and returns the homepage of the corresponding conference.")
async def get_conference_homepage(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T2.HomePage FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"homepage": []}
    return {"homepage": result[0]}

# Endpoint to get paper IDs and conference short names for papers published in a specific year
@app.get("/v1/authors/paper_conference_by_year", operation_id="get_paper_conference", summary="Retrieves a list of up to 10 paper IDs and their corresponding conference short names for papers published in a specific year. The year of publication is a required input parameter.")
async def get_paper_conference(year: int = Query(..., description="Year of publication")):
    cursor.execute("SELECT T2.PaperId, T4.ShortName FROM Author AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.AuthorId INNER JOIN Paper AS T3 ON T2.PaperId = T3.Id INNER JOIN Conference AS T4 ON T3.ConferenceId = T4.Id WHERE T3.Year = ? LIMIT 10", (year,))
    result = cursor.fetchall()
    if not result:
        return {"paper_conference": []}
    return {"paper_conference": [{"paper_id": row[0], "conference_short_name": row[1]} for row in result]}

# Endpoint to get author names for papers published in a specific year with a specific keyword
@app.get("/v1/authors/author_names_by_year_keyword", operation_id="get_author_names_by_year_keyword", summary="Retrieves the names of authors who have published papers in a specified year that contain a particular keyword. The keyword is 'KEY WORDS: LOAD IDE SNP haplotype association studies'. The year of publication is also provided as input.")
async def get_author_names_by_year_keyword(year: int = Query(..., description="Year of publication"), keyword: str = Query(..., description="Keyword in the paper")):
    cursor.execute("SELECT T2.Name FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Year = ? AND T1.Keyword = ?", (year, keyword))
    result = cursor.fetchall()
    if not result:
        return {"author_names": []}
    return {"author_names": [row[0] for row in result]}

# Endpoint to get the count of distinct authors for a specific paper title
@app.get("/v1/authors/distinct_author_count_by_paper_title", operation_id="get_distinct_author_count", summary="Retrieves the number of unique authors associated with a specific paper, based on the provided paper title.")
async def get_distinct_author_count(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT COUNT(DISTINCT T2.Name) FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get author IDs and paper titles for a specific author name
@app.get("/v1/authors/author_paper_titles", operation_id="get_author_paper_titles", summary="Retrieves the unique author IDs and corresponding paper titles associated with a given author name. The author name is used to filter the results.")
async def get_author_paper_titles(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T2.AuthorId, T1.Title FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T2.Name = ?", (author_name,))
    result = cursor.fetchall()
    if not result:
        return {"author_paper_titles": []}
    return {"author_paper_titles": [{"author_id": row[0], "title": row[1]} for row in result]}

# Endpoint to get conference IDs, journal IDs, author names, and paper titles for a specific author name
@app.get("/v1/authors/author_conference_journal_paper_titles", operation_id="get_author_conference_journal_paper_titles", summary="Retrieves the conference IDs, journal IDs, author names, and paper titles associated with a specific author. The author's name is used to filter the results.")
async def get_author_conference_journal_paper_titles(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T1.ConferenceId, T1.JournalId, T2.Name, T1.Title FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Conference AS T3 ON T1.ConferenceId = T3.Id INNER JOIN Journal AS T4 ON T1.JournalId = T4.Id WHERE T2.Name = ?", (author_name,))
    result = cursor.fetchall()
    if not result:
        return {"author_conference_journal_paper_titles": []}
    return {"author_conference_journal_paper_titles": [{"conference_id": row[0], "journal_id": row[1], "author_name": row[2], "title": row[3]} for row in result]}

# Endpoint to get conference short names based on full name pattern
@app.get("/v1/authors/conference_short_names_by_full_name", operation_id="get_conference_short_names", summary="Retrieves a list of short names for conferences that match the provided full name pattern. The full name pattern can include wildcard characters to broaden the search scope.")
async def get_conference_short_names(full_name_pattern: str = Query(..., description="Pattern to match the full name of the conference (use % for wildcard)")):
    cursor.execute("SELECT ShortName FROM Conference WHERE FullName LIKE ?", (full_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"short_names": []}
    return {"short_names": [row[0] for row in result]}

# Endpoint to get the count of journals with a specific homepage
@app.get("/v1/authors/journal_count_by_homepage", operation_id="get_journal_count", summary="Retrieves the total number of journals that have a specified homepage. The homepage parameter is used to filter the journals and calculate the count.")
async def get_journal_count(homepage: str = Query(..., description="Homepage of the journal")):
    cursor.execute("SELECT COUNT(HomePage) FROM Journal WHERE HomePage = ?", (homepage,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get author names from papers published in a specific year
@app.get("/v1/authors/author_names_by_paper_year", operation_id="get_author_names_by_year", summary="Retrieve the names of authors who have published papers in a specified year. The operation filters papers by the provided year and returns the corresponding author names.")
async def get_author_names_by_year(year: int = Query(..., description="Year the paper was published")):
    cursor.execute("SELECT T2.Name FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get author affiliations based on paper keyword
@app.get("/v1/authors/author_affiliations_by_keyword", operation_id="get_author_affiliations_by_keyword", summary="Retrieves the affiliations of authors who have published papers containing a specified keyword. The keyword is used to filter the papers, and the affiliations of the authors associated with those papers are returned.")
async def get_author_affiliations_by_keyword(keyword: str = Query(..., description="Keyword in the paper")):
    cursor.execute("SELECT T2.Affiliation FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Keyword = ?", (keyword,))
    result = cursor.fetchall()
    if not result:
        return {"affiliations": []}
    return {"affiliations": [row[0] for row in result]}

# Endpoint to get conference full names based on specific criteria
@app.get("/v1/authors/conference_full_names_by_criteria", operation_id="get_conference_full_names", summary="Retrieves the full names of conferences that meet specific criteria, excluding a specified conference, pertaining to a particular journal, and occurring in a year other than the provided one.")
async def get_conference_full_names(conference_id: int = Query(..., description="Conference ID (must not be this value)"), journal_id: int = Query(..., description="Journal ID (must be this value)"), year: int = Query(..., description="Year (must not be this value)")):
    cursor.execute("SELECT T2.FullName FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.ConferenceId != ? AND T1.JournalId = ? AND T1.Year != ?", (conference_id, journal_id, year))
    result = cursor.fetchall()
    if not result:
        return {"full_names": []}
    return {"full_names": [row[0] for row in result]}

# Endpoint to get paper titles from conferences with specific homepage and title criteria
@app.get("/v1/authors/paper_titles_by_conference_criteria", operation_id="get_paper_titles_by_conference_criteria", summary="Retrieve the titles of papers presented at conferences with a specified homepage and excluding papers with a certain title. The homepage of the conference and the title to be excluded are provided as input parameters.")
async def get_paper_titles_by_conference_criteria(homepage: str = Query(..., description="Homepage of the conference"), title: str = Query(..., description="Title of the paper (must not be this value)")):
    cursor.execute("SELECT T2.Title FROM Conference AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.ConferenceId WHERE T1.HomePage = ? AND T2.Title <> ?", (homepage, title))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get distinct years and full names of journals based on short name
@app.get("/v1/authors/journal_years_and_full_names_by_short_name", operation_id="get_journal_years_and_full_names", summary="Retrieves a list of unique years and full names of journals that match the provided short name. This operation is useful for obtaining a comprehensive overview of the publication years and full names of a specific journal.")
async def get_journal_years_and_full_names(short_name: str = Query(..., description="Short name of the journal")):
    cursor.execute("SELECT DISTINCT T2.Year, FullName FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.ShortName = ?", (short_name,))
    result = cursor.fetchall()
    if not result:
        return {"years_and_full_names": []}
    return {"years_and_full_names": [{"year": row[0], "full_name": row[1]} for row in result]}

# Endpoint to get author affiliations based on paper title
@app.get("/v1/authors/author_affiliations_by_paper_title", operation_id="get_author_affiliations_by_title", summary="Retrieves the affiliations of authors associated with a paper, based on the provided paper title. This operation returns a list of affiliations linked to the authors of the specified paper. The input parameter is the title of the paper, which is used to identify the relevant authors and their affiliations.")
async def get_author_affiliations_by_title(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T1.Affiliation FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T2.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"affiliations": []}
    return {"affiliations": [row[0] for row in result]}

# Endpoint to get journal full names based on specific criteria
@app.get("/v1/authors/journal_full_names_by_criteria", operation_id="get_journal_full_names_by_criteria", summary="Retrieves the full names of journals that published papers in a specific year, after a certain conference, and with a journal ID greater than a specified value. This operation filters papers based on the provided year, conference ID, and journal ID, and returns the full names of the corresponding journals.")
async def get_journal_full_names_by_criteria(year: int = Query(..., description="Year the paper was published"), conference_id: int = Query(..., description="Conference ID (must be greater than this value)"), journal_id: int = Query(..., description="Journal ID (must be greater than this value)")):
    cursor.execute("SELECT T2.FullName FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id WHERE T1.Year = ? AND T1.ConferenceId > ? AND T1.JournalId > ?", (year, conference_id, journal_id))
    result = cursor.fetchall()
    if not result:
        return {"full_names": []}
    return {"full_names": [row[0] for row in result]}

# Endpoint to get the count of papers and distinct years for a specific conference
@app.get("/v1/authors/paper_count_years_by_conference", operation_id="get_paper_count_years_by_conference", summary="Retrieves the total number of papers and the distinct years they were published for a specific conference. The conference is identified by its full name.")
async def get_paper_count_years_by_conference(conference_name: str = Query(..., description="Full name of the conference")):
    cursor.execute("SELECT COUNT(T1.Id) AS PAPER, COUNT(DISTINCT T1.Year) AS YEARS FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE year != 0 AND T2.FullName = ?", (conference_name,))
    result = cursor.fetchone()
    if not result:
        return {"paper_count": 0, "years_count": 0}
    return {"paper_count": result[0], "years_count": result[1]}

# Endpoint to get the full names of journals for a specific keyword
@app.get("/v1/authors/journal_names_by_keyword", operation_id="get_journal_names_by_keyword", summary="Retrieve the full names of journals that have published papers containing a specific keyword. The keyword is provided as an input parameter, and the operation returns a list of journal names that match the search criteria.")
async def get_journal_names_by_keyword(keyword: str = Query(..., description="Keyword to search for in papers")):
    cursor.execute("SELECT T2.FullName FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id WHERE T1.Keyword = ?", (keyword,))
    result = cursor.fetchall()
    if not result:
        return {"journals": []}
    return {"journals": [row[0] for row in result]}

# Endpoint to get the names of authors for papers presented at conferences with a specific name pattern
@app.get("/v1/authors/author_names_by_conference_pattern", operation_id="get_author_names_by_conference_pattern", summary="Retrieve the names of authors who have presented papers at conferences with a specified name pattern. The input parameter allows for a pattern match in conference full names, using %% as a wildcard.")
async def get_author_names_by_conference_pattern(conference_pattern: str = Query(..., description="Pattern to match in conference full names (use %% for wildcard)")):
    cursor.execute("SELECT T2.Name FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Conference AS T3 ON T1.ConferenceId = T3.Id WHERE T3.FullName LIKE ?", (conference_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": [row[0] for row in result]}

# Endpoint to get the percentage of papers with a specific affiliation pattern in the title
@app.get("/v1/authors/percentage_papers_by_affiliation_pattern", operation_id="get_percentage_papers_by_affiliation_pattern", summary="Retrieves the percentage of papers with a specified affiliation pattern in their title. The affiliation pattern is matched against the author affiliations, while the title pattern is matched against the paper titles. The result is calculated by summing the number of papers with matching affiliations and dividing it by the total number of papers.")
async def get_percentage_papers_by_affiliation_pattern(affiliation_pattern: str = Query(..., description="Pattern to match in affiliations (use %% for wildcard)"), title_pattern: str = Query(..., description="Pattern to match in paper titles (use %% for wildcard)")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T1.Affiliation LIKE ? THEN 1 ELSE 0 END)) AS REAL) * 100 / COUNT(T2.Id) FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T2.Title LIKE ?", (affiliation_pattern, title_pattern))
    result = cursor.fetchone()
    if not result:
        return {"percentage": 0}
    return {"percentage": result[0]}

# Endpoint to get the percentage of journals with a specific short name pattern in a given year
@app.get("/v1/authors/percentage_journals_by_short_name_pattern", operation_id="get_percentage_journals_by_short_name_pattern", summary="Retrieves the percentage of journals that match a specified short name pattern for a given year. The calculation is based on the total number of journals in the specified year. The short name pattern can include a wildcard for broader matches.")
async def get_percentage_journals_by_short_name_pattern(short_name_pattern: str = Query(..., description="Pattern to match in journal short names (use %% for wildcard)"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T1.ShortName LIKE ? THEN 1 ELSE 0 END)) AS REAL) * 100 / COUNT(T1.ShortName) FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T2.Year = ?", (short_name_pattern, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": 0}
    return {"percentage": result[0]}

# Endpoint to get the count of papers by authors with a specific affiliation pattern
@app.get("/v1/authors/paper_count_by_affiliation_pattern", operation_id="get_paper_count_by_affiliation_pattern", summary="Retrieves the total number of papers authored by individuals whose affiliations match the provided pattern. The affiliation pattern can include wildcards to broaden the search scope.")
async def get_paper_count_by_affiliation_pattern(affiliation_pattern: str = Query(..., description="Pattern to match in affiliations (use %% for wildcard)")):
    cursor.execute("SELECT COUNT(PaperId) FROM PaperAuthor WHERE Affiliation LIKE ?", (affiliation_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": 0}
    return {"count": result[0]}

# Endpoint to get the keywords and year for a specific paper title
@app.get("/v1/authors/keywords_year_by_title", operation_id="get_keywords_year_by_title", summary="Retrieves the keywords and publication year associated with a specific paper title. The provided paper title is used to search for the corresponding record in the database.")
async def get_keywords_year_by_title(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT Keyword, Year FROM Paper WHERE Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"keywords_year": []}
    return {"keywords_year": [{"keyword": row[0], "year": row[1]} for row in result]}

# Endpoint to get the percentage of papers published after a specific year
@app.get("/v1/authors/percentage_papers_after_year", operation_id="get_percentage_papers_after_year", summary="Retrieves the percentage of papers published after a specified year. This operation calculates the proportion of papers published after the input year by comparing it to the total number of papers in the database. The result is a numerical value representing the percentage of papers published after the given year.")
async def get_percentage_papers_after_year(year: int = Query(..., description="Year to compare against")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Year > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(Id) FROM Paper", (year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": 0}
    return {"percentage": result[0]}

# Endpoint to get the names of authors for a specific paper title
@app.get("/v1/authors/paper_authors_by_title", operation_id="get_paper_authors_by_title", summary="Retrieves the names of authors associated with a specific paper title. The operation uses the provided paper title to search for corresponding author names in the database.")
async def get_paper_authors_by_title(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T2.Name FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": [row[0] for row in result]}

# Endpoint to get the full name of the journal for a given paper title
@app.get("/v1/authors/journal_full_name_by_paper_title", operation_id="get_journal_full_name_by_paper_title", summary="Retrieves the full name of the journal associated with a specific paper title. The operation uses the provided paper title to search for the corresponding journal's full name in the database.")
async def get_journal_full_name_by_paper_title(paper_title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T2.FullName FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id WHERE T1.Title = ?", (paper_title,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the full name of the conference for a given paper ID
@app.get("/v1/authors/conference_full_name_by_paper_id", operation_id="get_conference_full_name_by_paper_id", summary="Retrieves the full name of the conference associated with a specific paper. The paper is identified by its unique ID, which is provided as an input parameter.")
async def get_conference_full_name_by_paper_id(paper_id: int = Query(..., description="ID of the paper")):
    cursor.execute("SELECT T2.FullName FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.Id = ?", (paper_id,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get paper titles by author name pattern
@app.get("/v1/authors/paper_titles_by_author_name_pattern", operation_id="get_paper_titles_by_author_name_pattern", summary="Retrieves a list of up to two paper titles associated with authors whose names match the provided pattern. The author name pattern should include wildcard characters (%%) to indicate the desired match criteria.")
async def get_paper_titles_by_author_name_pattern(author_name_pattern: str = Query(..., description="Pattern of the author name (use %% for wildcard)")):
    cursor.execute("SELECT T1.Title FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T2.Name LIKE ? LIMIT 2", (author_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get paper titles by author name and affiliation
@app.get("/v1/authors/paper_titles_by_author_name_and_affiliation", operation_id="get_paper_titles_by_author_name_and_affiliation", summary="Retrieves a list of paper titles associated with a specific author and their affiliation. The author's name and affiliation are used to filter the results.")
async def get_paper_titles_by_author_name_and_affiliation(author_name: str = Query(..., description="Name of the author"), affiliation: str = Query(..., description="Affiliation of the author")):
    cursor.execute("SELECT T1.Title FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T2.Name = ? AND T2.Affiliation = ?", (author_name, affiliation))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the full name of the conference for a given author name
@app.get("/v1/authors/conference_full_name_by_author_name", operation_id="get_conference_full_name_by_author_name", summary="Retrieves the full name of the conference where the specified author has published a paper. The author's name is used to search for a single conference, ensuring that the conference name returned is unique to the author.")
async def get_conference_full_name_by_author_name(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT DISTINCT T3.FullName FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Conference AS T3 ON T1.ConferenceId = T3.Id WHERE T2.Name = ? LIMIT 1", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the full name of the journal for a given author name
@app.get("/v1/authors/journal_full_name_by_author_name", operation_id="get_journal_full_name_by_author_name", summary="Retrieves the full name of the journal associated with a specific author. The author's name is used to search for corresponding papers and subsequently identify the journal's full name.")
async def get_journal_full_name_by_author_name(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T3.FullName FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Journal AS T3 ON T1.JournalId = T3.Id WHERE T2.Name = ?", (author_name,))
    result = cursor.fetchall()
    if not result:
        return {"full_name": []}
    return {"full_name": [row[0] for row in result]}

# Endpoint to get distinct affiliations for a given paper title
@app.get("/v1/authors/distinct_affiliations_by_paper_title", operation_id="get_distinct_affiliations_by_paper_title", summary="Retrieves a list of unique affiliations associated with a specific paper, based on the provided paper title.")
async def get_distinct_affiliations_by_paper_title(paper_title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT DISTINCT T2.Affiliation FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Title = ?", (paper_title,))
    result = cursor.fetchall()
    if not result:
        return {"affiliations": []}
    return {"affiliations": [row[0] for row in result]}

# Endpoint to get paper titles and journal full names for papers published before a given year
@app.get("/v1/authors/paper_titles_journal_full_names_before_year", operation_id="get_paper_titles_journal_full_names_before_year", summary="Retrieves a list of paper titles and their corresponding journal full names for papers published before the specified year. The list is limited to the first two results.")
async def get_paper_titles_journal_full_names_before_year(year: int = Query(..., description="Year before which the papers were published")):
    cursor.execute("SELECT T1.Title, T2.FullName FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id WHERE T1.Year < ? LIMIT 2", (year,))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": [{"title": row[0], "journal_full_name": row[1]} for row in result]}

# Endpoint to get the titles of papers and author names from a specific journal and year
@app.get("/v1/authors/paper_titles_authors_by_journal_year", operation_id="get_paper_titles_authors", summary="Retrieves the titles of papers and corresponding author names from a specific journal and year. The operation requires the full name of the journal and the year of the papers as input parameters. The output includes a list of paper titles and their respective authors.")
async def get_paper_titles_authors(journal_full_name: str = Query(..., description="Full name of the journal"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT T1.Title, T2.Name FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Journal AS T3 ON T1.JournalId = T3.Id WHERE T3.FullName = ? AND T1.Year = ?", (journal_full_name, year))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": [{"title": row[0], "author": row[1]} for row in result]}

# Endpoint to get the titles of papers and author names from a specific affiliation and year
@app.get("/v1/authors/paper_titles_authors_by_affiliation_year", operation_id="get_paper_titles_authors_affiliation", summary="Retrieves the titles of papers and corresponding author names associated with a specified affiliation and year. This operation allows you to filter papers by the author's affiliation and the year of publication, providing a targeted list of papers and their authors.")
async def get_paper_titles_authors_affiliation(affiliation: str = Query(..., description="Affiliation of the author"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT T2.Title, T1.Name FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T1.Affiliation = ? AND T2.Year = ?", (affiliation, year))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": [{"title": row[0], "author": row[1]} for row in result]}

# Endpoint to get the titles of papers and author names based on keyword, year range, and non-empty title
@app.get("/v1/authors/paper_titles_authors_by_keyword_year_range", operation_id="get_paper_titles_authors_keyword", summary="Retrieves the titles of papers and corresponding author names that match a specified keyword and fall within a given year range. The keyword search supports wildcard matching, and the year range is inclusive. Only papers with non-empty titles are considered.")
async def get_paper_titles_authors_keyword(keyword: str = Query(..., description="Keyword to match in the paper (use %% for wildcard)"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT T1.Title, T2.Name FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Keyword LIKE ? AND T1.Year BETWEEN ? AND ? AND T1.Title <> ?", (keyword, start_year, end_year, ''))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": [{"title": row[0], "author": row[1]} for row in result]}

# Endpoint to get the proportion of papers in a specific journal by year
@app.get("/v1/authors/proportion_papers_by_journal_year", operation_id="get_proportion_papers_by_journal", summary="Retrieve the proportion of papers published in a specific journal each year. The calculation is based on the total number of papers published in the journal divided by the total number of papers published in that year. The full name of the journal is required as an input parameter.")
async def get_proportion_papers_by_journal(journal_full_name: str = Query(..., description="Full name of the journal")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.FullName = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(T1.Id) AS Div1, T1.Year FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id GROUP BY T1.YEAR HAVING Div1 != 0", (journal_full_name,))
    result = cursor.fetchall()
    if not result:
        return {"proportions": []}
    return {"proportions": [{"proportion": row[0], "year": row[1]} for row in result]}

# Endpoint to get distinct author names and paper titles for a specific year, conference, and journal
@app.get("/v1/authors/distinct_authors_papers_by_year_conference_journal", operation_id="get_distinct_authors_papers", summary="Retrieve a unique list of author names and their corresponding paper titles, filtered by a specific year, conference, and journal. This operation ensures that only papers with titles are included in the results.")
async def get_distinct_authors_papers(conference_id: int = Query(..., description="Conference ID"), journal_id: int = Query(..., description="Journal ID"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT DISTINCT T2.Name, T1.Title FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.ConferenceId = ? AND T1.JournalId = ? AND T1.Year = ? AND T1.Title <> ?", (conference_id, journal_id, year, ''))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": [{"author": row[0], "title": row[1]} for row in result]}

# Endpoint to get the count of authors from a specific affiliation
@app.get("/v1/authors/author_count_by_affiliation", operation_id="get_author_count_by_affiliation", summary="Retrieves the total number of authors associated with a given affiliation. The affiliation is provided as an input parameter.")
async def get_author_count_by_affiliation(affiliation: str = Query(..., description="Affiliation of the author")):
    cursor.execute("SELECT COUNT(Name) FROM Author WHERE Affiliation = ?", (affiliation,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of papers for a given conference full name
@app.get("/v1/authors/count_papers_by_conference_full_name", operation_id="get_count_papers_by_conference", summary="Retrieves the total number of papers associated with a specific conference, identified by its full name. This operation provides a quantitative measure of the conference's publication output.")
async def get_count_papers_by_conference(full_name: str = Query(..., description="Full name of the conference")):
    cursor.execute("SELECT COUNT(T1.Id) FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T2.FullName = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of papers for a given year and conference short name
@app.get("/v1/authors/count_papers_by_year_and_conference_short_name", operation_id="get_count_papers_by_year_and_conference", summary="Retrieves the total number of papers published in a specific year and conference. The operation requires the year of publication and the short name of the conference as input parameters. The result is a count of papers that meet the specified criteria.")
async def get_count_papers_by_year_and_conference(year: int = Query(..., description="Year of the paper"), short_name: str = Query(..., description="Short name of the conference")):
    cursor.execute("SELECT COUNT(T1.Id) FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.Year = ? AND T2.ShortName = ?", (year, short_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct paper titles for a given journal full name and year
@app.get("/v1/authors/distinct_paper_titles_by_journal_and_year", operation_id="get_distinct_paper_titles", summary="Retrieves a list of unique paper titles published in a specific journal during a given year. The full name of the journal and the year of publication are required as input parameters.")
async def get_distinct_paper_titles(full_name: str = Query(..., description="Full name of the journal"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT DISTINCT T1.Title FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id WHERE T2.FullName = ? AND T1.Year = ? AND T1.Title <> ?", (full_name, year, ''))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the names of authors for a given affiliation and paper title
@app.get("/v1/authors/author_names_by_affiliation_and_paper_title", operation_id="get_author_names_by_affiliation", summary="Retrieves the names of authors associated with a specific affiliation and paper title. The affiliation and paper title are provided as input parameters, allowing for targeted results. This operation is useful for identifying authors who have contributed to a particular paper from a certain institution.")
async def get_author_names_by_affiliation(affiliation: str = Query(..., description="Affiliation of the author"), title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T2.Name FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T2.Affiliation = ? AND T1.Title = ?", (affiliation, title))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the average number of papers per year for a specific conference within a given year range
@app.get("/v1/authors/average_papers_per_year_given_conference_year_range", operation_id="get_average_papers_per_year", summary="Retrieves the average number of papers published per year for a specific conference within a given year range. The calculation is based on the count of distinct papers and their respective years of publication. The conference is identified by its full name, and the year range is defined by a start and end year.")
async def get_average_papers_per_year(conference_name: str = Query(..., description="Full name of the conference"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT CAST(COUNT(T1.Id) AS REAL) / COUNT(DISTINCT T1.Year) FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T2.FullName = ? AND T1.Year >= ? AND T1.Year <= ?", (conference_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the difference in the number of papers between two conferences within a given year range
@app.get("/v1/authors/paper_count_difference_between_conferences_year_range", operation_id="get_paper_count_difference", summary="Retrieve the difference in the number of papers published between two specified conferences within a given year range. This operation compares the total papers from the first conference to the second conference, considering only the papers published between the start and end years.")
async def get_paper_count_difference(conference_name_1: str = Query(..., description="Full name of the first conference"), conference_name_2: str = Query(..., description="Full name of the second conference"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT SUM(CASE WHEN T2.FullName = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.FullName = ? THEN 1 ELSE 0 END) AS DIFF FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.Year > ? AND T1.Year < ?", (conference_name_1, conference_name_2, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the short name of a journal given its full name
@app.get("/v1/authors/journal_short_name_given_full_name", operation_id="get_journal_short_name", summary="Retrieves the abbreviated name of a journal based on its complete name. The operation requires the full name of the journal as input and returns the corresponding short name.")
async def get_journal_short_name(full_name: str = Query(..., description="Full name of the journal")):
    cursor.execute("SELECT ShortName FROM Journal WHERE FullName = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"short_name": []}
    return {"short_name": result[0]}

# Endpoint to get the full name of a journal given a paper title
@app.get("/v1/authors/journal_full_name_given_paper_title", operation_id="get_journal_full_name", summary="Retrieves the full name of the journal associated with a specific paper. The paper is identified by its title. This operation returns the full name of the journal where the paper was published.")
async def get_journal_full_name(paper_title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T1.FullName FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T2.Title = ?", (paper_title,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the count of papers in a specific journal for a given year
@app.get("/v1/authors/paper_count_given_journal_year", operation_id="get_paper_count_year", summary="Retrieves the total number of papers published in a specific journal during a given year. The operation requires the full name of the journal and the year as input parameters.")
async def get_paper_count_year(journal_name: str = Query(..., description="Full name of the journal"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT COUNT(T2.Id) FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.FullName = ? AND T2.Year = ?", (journal_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the titles of papers in a specific journal for a given year
@app.get("/v1/authors/paper_titles_given_journal_year", operation_id="get_paper_titles_year", summary="Retrieve the titles of papers published in a specific journal during a given year. The operation filters out papers with empty titles and returns a list of relevant paper titles.")
async def get_paper_titles_year(journal_name: str = Query(..., description="Full name of the journal"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT T2.Title FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.FullName = ? AND T2.Year = ? AND T2.Title <> ?", (journal_name, year, ''))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the keywords of papers in a specific journal for a given year
@app.get("/v1/authors/paper_keywords_given_journal_year", operation_id="get_paper_keywords_year", summary="Retrieves the keywords of papers published in a specific journal during a given year. The operation requires the full name of the journal and the year of the paper as input parameters.")
async def get_paper_keywords_year(journal_name: str = Query(..., description="Full name of the journal"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT T2.Keyword FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.FullName = ? AND T2.Year = ?", (journal_name, year))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to check if a paper has a year of 0 for a specific author with no conference or journal
@app.get("/v1/authors/check_paper_year_zero_given_author", operation_id="check_paper_year_zero", summary="Determines whether a paper associated with a specific author, who has no conference or journal affiliation, has a year value of 0. The operation returns a boolean value indicating the result.")
async def check_paper_year_zero(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT CASE WHEN T1.Year = 0 THEN 'TRUE' ELSE 'FALSE' END FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T2.Name = ? AND T1.ConferenceId = 0 AND T1.JournalId = 0", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"year_zero": []}
    return {"year_zero": result[0]}

# Endpoint to get the distinct home pages of conferences where a specific paper was presented
@app.get("/v1/authors/conference_home_pages_by_paper_title", operation_id="get_conference_home_pages", summary="Retrieve the unique home pages of conferences where a paper with a given title was presented. The operation requires the title of the paper as input and returns a list of distinct conference home pages.")
async def get_conference_home_pages(paper_title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT DISTINCT T2.HomePage FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.Title = ?", (paper_title,))
    result = cursor.fetchall()
    if not result:
        return {"home_pages": []}
    return {"home_pages": [row[0] for row in result]}

# Endpoint to get the short name of the conference with the most papers in a specific year
@app.get("/v1/authors/conference_short_name_most_papers_by_year", operation_id="get_conference_short_name_most_papers", summary="Retrieves the short name of the conference that published the highest number of papers in a given year. The year is specified in 'YYYY' format.")
async def get_conference_short_name_most_papers(year: int = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.ShortName FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.Year = ? GROUP BY T1.ConferenceId ORDER BY COUNT(T1.Id) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"short_name": []}
    return {"short_name": result[0]}

# Endpoint to get the count of papers presented at a specific conference in a specific year
@app.get("/v1/authors/paper_count_by_conference_short_name_and_year", operation_id="get_paper_count_by_conference_and_year", summary="Retrieves the total number of papers presented at a particular conference during a specified year. The operation requires the short name of the conference and the year in 'YYYY' format as input parameters.")
async def get_paper_count_by_conference_and_year(conference_short_name: str = Query(..., description="Short name of the conference"), year: int = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.Id) FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T2.ShortName = ? AND T1.Year = ?", (conference_short_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the keywords of papers presented at a specific conference in a specific year
@app.get("/v1/authors/paper_keywords_by_conference_full_name_and_year", operation_id="get_paper_keywords_by_conference_and_year", summary="Retrieves the keywords of papers presented at a specified conference during a particular year. The operation requires the full name of the conference and the year of the conference in 'YYYY' format as input parameters.")
async def get_paper_keywords_by_conference_and_year(conference_full_name: str = Query(..., description="Full name of the conference"), year: int = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T1.Keyword FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T2.FullName = ? AND T1.Year = ?", (conference_full_name, year))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get the ratio of papers presented at two specific conferences
@app.get("/v1/authors/paper_ratio_by_conference_full_names", operation_id="get_paper_ratio_by_conferences", summary="Retrieves the ratio of papers presented at a specified conference compared to the 'International Conference on Wireless Networks, Communications and Mobile Computing'. The operation calculates this ratio by summing the number of papers presented at the specified conference and dividing it by the total number of papers presented at the reference conference.")
async def get_paper_ratio_by_conferences(conference_full_name_1: str = Query(..., description="Full name of the first conference"), conference_full_name_2: str = Query(..., description="Full name of the second conference")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.FullName = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.FullName = ? THEN 1 ELSE 0 END) FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id", (conference_full_name_1, conference_full_name_2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the ratio of papers not associated with any conference or journal for a specific author
@app.get("/v1/authors/paper_ratio_not_associated_by_author_name", operation_id="get_paper_ratio_not_associated_by_author", summary="Retrieves the proportion of papers that are not linked to any conference or journal for a given author. The author is identified by their name.")
async def get_paper_ratio_not_associated_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.ConferenceId = 0 AND T1.JournalId = 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(T1.Id) FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T2.Name = ?", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the title of the earliest paper published after a specific year
@app.get("/v1/authors/earliest_paper_title_after_year", operation_id="get_earliest_paper_title", summary="Retrieves the title of the earliest paper published after the specified year. The operation filters papers by year, sorts them in ascending order, and returns the title of the first paper that meets the criteria.")
async def get_earliest_paper_title(year: int = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT Title FROM Paper WHERE Year > ? ORDER BY Year ASC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the conference with the longest name
@app.get("/v1/authors/longest_conference_name", operation_id="get_longest_conference_name", summary="Retrieves the conference with the longest name from the available records. The operation does not require any input parameters and returns the full name of the conference as the response.")
async def get_longest_conference_name():
    cursor.execute("SELECT FullName FROM Conference ORDER BY LENGTH(FullName) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"FullName": []}
    return {"FullName": result[0]}

# Endpoint to get the paper with the most authors
@app.get("/v1/authors/paper_with_most_authors", operation_id="get_paper_with_most_authors", summary="Retrieves the title of the paper that has the highest number of authors. This operation identifies the paper with the most collaborators by joining the PaperAuthor and Paper tables, grouping by paper ID, and ordering the results in descending order based on the count of authors per paper. The top result is then returned.")
async def get_paper_with_most_authors():
    cursor.execute("SELECT T2.Title FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id GROUP BY T1.PaperId ORDER BY COUNT(T1.PaperId) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"Title": []}
    return {"Title": result[0]}

# Endpoint to get the earliest paper from a journal with a specific short name
@app.get("/v1/authors/earliest_paper_given_journal_short_name", operation_id="get_earliest_paper", summary="Retrieves the title of the earliest paper published in a journal identified by its short name. The operation returns the paper with the earliest publication year from the specified journal.")
async def get_earliest_paper(short_name: str = Query(..., description="Short name of the journal")):
    cursor.execute("SELECT T2.Title FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.ShortName = ? ORDER BY T2.Year ASC LIMIT 1", (short_name,))
    result = cursor.fetchone()
    if not result:
        return {"Title": []}
    return {"Title": result[0]}

# Endpoint to get the count of papers in a specific conference and year
@app.get("/v1/authors/paper_count_given_conference_year", operation_id="get_paper_count_conference_year", summary="Retrieves the total number of papers published in a specific conference during a given year. The operation requires the full name of the conference and the year of the papers as input parameters.")
async def get_paper_count_conference_year(full_name: str = Query(..., description="Full name of the conference"), year: int = Query(..., description="Year of the paper")):
    cursor.execute("SELECT COUNT(T2.Id) FROM Conference AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.ConferenceId WHERE T1.FullName = ? AND T2.Year = ?", (full_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the affiliations of a specific author
@app.get("/v1/authors/affiliations_by_author", operation_id="get_affiliations_by_author", summary="Retrieves the affiliations of a specific author by querying the author's name. The response includes a list of affiliations associated with the author.")
async def get_affiliations_by_author(name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T1.Affiliation FROM PaperAuthor AS T1 INNER JOIN Author AS T2 ON T1.AuthorId = T2.Id WHERE T2.Name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"Affiliations": []}
    return {"Affiliations": [row[0] for row in result]}

# Endpoint to get paper titles and journal IDs within a given year range
@app.get("/v1/authors/papers_by_year_range", operation_id="get_papers_by_year_range", summary="Retrieves the titles and corresponding journal IDs of papers published within a specified year range. The operation filters papers based on the provided start and end years, returning only those that fall within this range.")
async def get_papers_by_year_range(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT T1.Title, T1.JournalId FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id WHERE T1.Year >= ? AND T1.Year <= ?", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": [{"title": row[0], "journal_id": row[1]} for row in result]}

# Endpoint to get the year with the most papers for a given conference
@app.get("/v1/authors/most_papers_year_by_conference", operation_id="get_most_papers_year_by_conference", summary="Retrieves the year in which the specified conference had the highest number of published papers. The conference is identified by its full name.")
async def get_most_papers_year_by_conference(conference_name: str = Query(..., description="Full name of the conference")):
    cursor.execute("SELECT T2.Year FROM Conference AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.ConferenceId WHERE T1.FullName = ? GROUP BY T2.Year ORDER BY COUNT(T2.Id) DESC LIMIT 1", (conference_name,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get distinct co-authors of a given author
@app.get("/v1/authors/co_authors_by_author_name", operation_id="get_co_authors_by_author_name", summary="Retrieve a list of unique co-authors associated with a specific author. The author's name is used to filter the results, ensuring that the author themselves are not included in the list of co-authors.")
async def get_co_authors_by_author_name(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT DISTINCT T1.Name FROM PaperAuthor AS T1 INNER JOIN Author AS T2 ON T1.AuthorId = T2.Id WHERE T2.Name = ? AND T1.Name != ?", (author_name, author_name))
    result = cursor.fetchall()
    if not result:
        return {"co_authors": []}
    return {"co_authors": [row[0] for row in result]}

# Endpoint to get distinct conference names for a given year
@app.get("/v1/authors/conferences_by_year", operation_id="get_conferences_by_year", summary="Retrieve a unique list of conference names that occurred in a specified year. This operation filters papers by the provided year and extracts the corresponding conference names, ensuring no duplicates.")
async def get_conferences_by_year(year: int = Query(..., description="Year of the conference")):
    cursor.execute("SELECT DISTINCT T2.FullName FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T1.Year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"conferences": []}
    return {"conferences": [row[0] for row in result]}

# Endpoint to get the homepage of the journal with the most papers
@app.get("/v1/authors/journal_homepage_most_papers", operation_id="get_journal_homepage_most_papers", summary="Retrieves the homepage URL of the journal that has published the highest number of papers. This operation identifies the journal with the most papers by counting the number of papers associated with each journal and then orders the results in descending order. The homepage URL of the top-ranked journal is then returned.")
async def get_journal_homepage_most_papers():
    cursor.execute("SELECT T2.HomePage FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id GROUP BY T1.JournalId ORDER BY COUNT(T1.JournalId) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"homepage": []}
    return {"homepage": result[0]}

# Endpoint to get the proportion of papers with a specific keyword and author details
@app.get("/v1/authors/keyword_proportion_and_author_details", operation_id="get_keyword_proportion_and_author_details", summary="Retrieves the proportion of papers containing a specified keyword and provides details about the authors of those papers. The response includes the proportion of papers with the keyword, as well as the names and affiliations of the authors associated with those papers.")
async def get_keyword_proportion_and_author_details(keyword: str = Query(..., description="Keyword to search in papers")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Keyword = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(T1.Id), T2.Name, T2.Affiliation FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId", (keyword,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": [{"proportion": row[0], "name": row[1], "affiliation": row[2]} for row in result]}

# Endpoint to get the name of an author by ID
@app.get("/v1/authors/author_name_by_id", operation_id="get_author_name_by_id", summary="Retrieves the name of a specific author using their unique identifier. The author's ID is required as an input parameter to identify the correct author.")
async def get_author_name_by_id(author_id: int = Query(..., description="ID of the author")):
    cursor.execute("SELECT Name FROM Author WHERE Id = ?", (author_id,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of papers based on year or conference and journal IDs
@app.get("/v1/authors/paper_count_by_year_or_conference_journal", operation_id="get_paper_count_by_year_or_conference_journal", summary="Retrieves the total number of papers published in a specific year or associated with a particular conference and journal. The response is based on the provided year, conference ID, and journal ID.")
async def get_paper_count_by_year_or_conference_journal(year: int = Query(..., description="Year of the paper"), conference_id: int = Query(..., description="Conference ID of the paper"), journal_id: int = Query(..., description="Journal ID of the paper")):
    cursor.execute("SELECT COUNT(Id) FROM Paper WHERE Year = ? OR (ConferenceId = ? AND JournalId = ?)", (year, conference_id, journal_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of papers published in a specific year and the difference in the number of papers between two years
@app.get("/v1/authors/paper_count_and_diff", operation_id="get_paper_count_and_diff", summary="Get the sum of papers published in a specific year and the difference in the number of papers between two years")
async def get_paper_count_and_diff(year1: int = Query(..., description="First year to compare"), year2: int = Query(..., description="Second year to compare")):
    cursor.execute("SELECT SUM(CASE WHEN Year = ? THEN 1 ELSE 0 END) , SUM(CASE WHEN year = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN year = ? THEN 1 ELSE 0 END) AS diff FROM Paper", (year1, year2, year2))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "diff": []}
    return {"count": result[0], "diff": result[1]}

# Endpoint to get paper titles from a specific journal
@app.get("/v1/authors/paper_titles_by_journal", operation_id="get_paper_titles_by_journal", summary="Retrieves a list of paper titles published in the specified journal. The operation requires the full name of the journal as an input parameter.")
async def get_paper_titles_by_journal(journal_name: str = Query(..., description="Full name of the journal")):
    cursor.execute("SELECT T2.Title FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.FullName = ?", (journal_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get author names and affiliations for a specific paper title
@app.get("/v1/authors/author_info_by_paper_title", operation_id="get_author_info_by_paper_title", summary="Retrieves the names and affiliations of authors associated with a specific paper. The paper is identified by its title, which is provided as an input parameter.")
async def get_author_info_by_paper_title(paper_title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T2.Name, T2.Affiliation FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Title = ?", (paper_title,))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": [{"name": row[0], "affiliation": row[1]} for row in result]}

# Endpoint to get the count of authors for papers in a specific journal
@app.get("/v1/authors/author_count_by_journal", operation_id="get_author_count_by_journal", summary="Retrieves the total number of authors who have published papers in a specific journal. The operation requires the full name of the journal as an input parameter.")
async def get_author_count_by_journal(journal_name: str = Query(..., description="Full name of the journal")):
    cursor.execute("SELECT COUNT(T2.Name) FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId INNER JOIN Journal AS T3 ON T1.JournalId = T3.Id WHERE T3.FullName = ?", (journal_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of papers in a specific conference within a year range
@app.get("/v1/authors/paper_count_by_conference_year_range", operation_id="get_paper_count_by_conference_year_range", summary="Retrieves the total number of papers published in a specific conference within a specified year range. The operation requires the full name of the conference and the start and end years of the range as input parameters.")
async def get_paper_count_by_conference_year_range(conference_name: str = Query(..., description="Full name of the conference"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT COUNT(T1.Id) FROM Paper AS T1 INNER JOIN Conference AS T2 ON T1.ConferenceId = T2.Id WHERE T2.FullName = ? AND T1.Year BETWEEN ? AND ?", (conference_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of distinct authors to distinct paper titles within a year range
@app.get("/v1/authors/author_paper_ratio_by_year_range", operation_id="get_author_paper_ratio_by_year_range", summary="Retrieves the ratio of unique authors to unique paper titles published within a specified year range. The operation calculates this ratio by counting the distinct author IDs and paper titles that fall within the provided start and end years.")
async def get_author_paper_ratio_by_year_range(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT CAST(COUNT(DISTINCT T2.AuthorId) AS REAL) / COUNT(DISTINCT T1.Title) FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Year BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the year and journal name for a specific paper title
@app.get("/v1/authors/paper_year_journal_by_title", operation_id="get_paper_year_journal_by_title", summary="Retrieves the publication year and journal name for a specific paper, based on the provided paper title.")
async def get_paper_year_journal_by_title(paper_title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T1.Year, T2.FullName FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id WHERE T1.Title = ?", (paper_title,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": [{"year": row[0], "journal": row[1]} for row in result]}

# Endpoint to get the count of papers in a specific journal within a given year range
@app.get("/v1/authors/paper_count_journal_year_range", operation_id="get_paper_count_journal_year_range", summary="Retrieves the total number of papers published in a specific journal within a defined year range. The operation requires the full name of the journal, the start year, and the end year of the range as input parameters.")
async def get_paper_count_journal_year_range(journal_name: str = Query(..., description="Full name of the journal"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT COUNT(T2.JournalId) FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.FullName = ? AND T2.Year BETWEEN ? AND ?", (journal_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latest paper title by a specific author
@app.get("/v1/authors/latest_paper_by_author", operation_id="get_latest_paper_by_author", summary="Retrieves the title of the most recent paper authored by the specified author. The operation filters papers by author name and sorts them in descending order by publication year, returning the title of the top result.")
async def get_latest_paper_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T2.Title FROM PaperAuthor AS T1 INNER JOIN Paper AS T2 ON T1.PaperId = T2.Id WHERE T1.Name = ? ORDER BY T2.Year DESC LIMIT 1", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the average number of papers per year in a specific journal
@app.get("/v1/authors/average_papers_per_year_journal", operation_id="get_average_papers_per_year_journal", summary="Retrieves the average number of papers published per year in a specific journal. The calculation is based on the count of papers associated with the journal, divided by the number of distinct years in which those papers were published. The journal is identified by its full name.")
async def get_average_papers_per_year_journal(journal_name: str = Query(..., description="Full name of the journal")):
    cursor.execute("SELECT CAST(COUNT(T2.JournalId) AS REAL) / COUNT(DISTINCT T2.Year) FROM Journal AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.JournalId WHERE T1.FullName = ?", (journal_name,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the count of papers by a specific author
@app.get("/v1/authors/paper_count_by_author", operation_id="get_paper_count_by_author", summary="Retrieves the total number of papers authored by a specific individual. The operation requires the author's name as input and returns the corresponding count.")
async def get_paper_count_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT COUNT(PaperId) FROM PaperAuthor WHERE Name = ?", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of papers with a specific keyword to papers in a specific journal
@app.get("/v1/authors/keyword_journal_ratio", operation_id="get_keyword_journal_ratio", summary="Retrieves the proportion of papers containing a specified keyword within a particular journal. The operation calculates this ratio by dividing the total count of papers with the given keyword by the total count of papers in the specified journal.")
async def get_keyword_journal_ratio(keyword: str = Query(..., description="Keyword to search for"), journal_name: str = Query(..., description="Full name of the journal")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Keyword = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.FullName = ? THEN 1 ELSE 0 END) FROM Paper AS T1 INNER JOIN Journal AS T2 ON T1.JournalId = T2.Id", (keyword, journal_name))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of papers by authors from a specific affiliation
@app.get("/v1/authors/paper_count_by_affiliation", operation_id="get_paper_count_by_affiliation", summary="Retrieves the total number of papers authored by individuals from a specified institution. The affiliation parameter is used to filter the authors by their institutional affiliation.")
async def get_paper_count_by_affiliation(affiliation: str = Query(..., description="Affiliation of the author")):
    cursor.execute("SELECT COUNT(T2.PaperId) FROM Author AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.AuthorId WHERE T1.Affiliation = ?", (affiliation,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the year and title of papers by a specific author
@app.get("/v1/authors/papers_by_author", operation_id="get_papers_by_author", summary="Retrieves the year and title of papers authored by a specific individual. The operation filters papers based on the provided author's name.")
async def get_papers_by_author(name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T1.Year, T1.Title FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T2.Name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"papers": []}
    return {"papers": result}

# Endpoint to get the names of authors and the year of a specific paper
@app.get("/v1/authors/authors_by_paper", operation_id="get_authors_by_paper", summary="Retrieves the names of authors associated with a specific paper and the year the paper was published. The paper is identified by its unique ID.")
async def get_authors_by_paper(paper_id: int = Query(..., description="ID of the paper")):
    cursor.execute("SELECT T1.Name, T3.Year FROM Author AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.AuthorId INNER JOIN Paper AS T3 ON T2.PaperId = T3.Id WHERE T2.PaperId = ?", (paper_id,))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get the names of authors and the keywords of a specific paper
@app.get("/v1/authors/authors_keywords_by_paper", operation_id="get_authors_keywords_by_paper", summary="Retrieves the names of authors and their associated keywords for a specific paper, identified by its unique ID.")
async def get_authors_keywords_by_paper(paper_id: int = Query(..., description="ID of the paper")):
    cursor.execute("SELECT T1.Name, T3.Keyword FROM Author AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.AuthorId INNER JOIN Paper AS T3 ON T2.PaperId = T3.Id WHERE T2.PaperId = ?", (paper_id,))
    result = cursor.fetchall()
    if not result:
        return {"authors_keywords": []}
    return {"authors_keywords": result}

# Endpoint to get the full name of conferences where a specific paper was presented
@app.get("/v1/authors/conferences_by_paper_title", operation_id="get_conferences_by_paper_title", summary="Retrieves the full names of conferences where a paper with the specified title was presented. The operation requires the title of the paper as an input parameter.")
async def get_conferences_by_paper_title(title: str = Query(..., description="Title of the paper")):
    cursor.execute("SELECT T1.FullName FROM Conference AS T1 INNER JOIN Paper AS T2 ON T1.Id = T2.ConferenceId WHERE T2.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"conferences": []}
    return {"conferences": result}

api_calls = [
    "/v1/authors/author_count_given_affiliation?affiliation=University%20of%20California%20Berkeley",
    "/v1/authors/paper_keywords_by_title?title=Stitching%20videos%20streamed%20by%20mobile%20phones%20in%20real-time",
    "/v1/authors/paper_titles_by_journal_and_year?full_name=Concepts%20in%20Magnetic%20Resonance%20Part%20A&year=2008",
    "/v1/authors/paper_count_by_journal?full_name=Concepts%20in%20Magnetic%20Resonance%20Part%20A",
    "/v1/authors/journal_homepage_by_paper_title?title=Area%20Effects%20in%20Cepaea",
    "/v1/authors/paper_count_by_year_and_journal?year=2011&full_name=Molecular%20Brain",
    "/v1/authors/paper_count_by_year_and_short_name?year=2011&short_name=Mol%20Brain",
    "/v1/authors/author_count_by_paper_title?title=Equation%20Solving%20in%20Geometrical%20Theories",
    "/v1/authors/author_names_by_paper_title?title=Area%20Effects%20in%20Cepaea",
    "/v1/authors/author_count_by_affiliation_and_title?affiliation=University%20of%20Tokyo&title=FIBER:%20A%20Generalized%20Framework%20for%20Auto-tuning%20Software",
    "/v1/authors/paper_titles_by_author?name=Klaus%20Zimmermann",
    "/v1/authors/author_names_by_paper_title_and_affiliation?title=Incremental%20Extraction%20of%20Keyterms%20for%20Classifying%20Multilingual%20Documents%20in%20the%20Web&affiliation=National%20Taiwan%20University%20Department%20of%20Computer%20Science%20and%20Information%20Engineering%20Taiwan",
    "/v1/authors/paper_titles_by_author_and_year?name=Thomas%20Wiegand&year=1995",
    "/v1/authors/count_paper_titles_by_author_and_year?name=Thomas%20Wiegand&year=1995",
    "/v1/authors/average_papers_per_year_for_journal?full_name=Molecular%20Brain&start_year=2008&end_year=2011",
    "/v1/authors/difference_in_paper_count_between_journals?journal1=Cases%20Journal&journal2=Molecular%20Brain",
    "/v1/authors/count_journals_by_name_pattern?name_pattern=%25computing%25",
    "/v1/authors/paper_author_ids_by_affiliation?affiliation_pattern=Cairo%20Microsoft%20Innovation%20Lab%25",
    "/v1/authors/conference_names_by_homepage?homepage=http%3A%2F%2Fwww.informatik.uni-trier.de%2F~ley%2Fdb%2Fconf%2Fices%2Findex.html",
    "/v1/authors/author_ids_by_name?name=Peter",
    "/v1/authors/paper_conference_titles?min_conference_id=160&max_conference_id=170",
    "/v1/authors/latest_paper_journal",
    "/v1/authors/paper_conference_by_year_author?year=1999&author_name=Philip",
    "/v1/authors/paper_count_conference_homepage?min_year=1990&max_year=2000&conference_id=187",
    "/v1/authors/paper_titles_by_affiliation_year?affiliation=Department%20of%20Network%20Science%2C%20Graduate%20School%20of%20Information%20Systems%2C%20The%20University%20of%20Electro-Communications&year=2003",
    "/v1/authors/author_journal_by_journal_id_title?min_journal_id=245&max_journal_id=250&title_keyword=chemiluminescence",
    "/v1/authors/author_names_by_conference_journal?conference_id=0&max_journal_id=100",
    "/v1/authors/paper_titles_author_ids_by_year_conference?year=2006&max_conference_id=100",
    "/v1/authors/paper_ids_by_conference_homepage?homepage_keyword=http://www.informatik.uni-trier.de/~ley/db/conf/%",
    "/v1/authors/journal_homepages_author_ids_by_year_title?min_year=2000&max_year=2005&title_keyword=SOCIAL",
    "/v1/authors/distinct_authors_by_journal_year?journal_id=0&year=2009",
    "/v1/authors/percentage_papers_by_conference_journal_range?conference_id=0&min_journal_id=200&max_journal_id=300&short_name_pattern=A%25",
    "/v1/authors/percentage_papers_by_year_range?year=2001&min_year=2001&max_year=2011",
    "/v1/authors/conference_details_by_short_name?short_name=ICWE",
    "/v1/authors/top_affiliation_by_author_count?affiliation1=Stanford%20University&affiliation2=Massachusetts%20Institute%20of%20Technology",
    "/v1/authors/percentage_authors_without_affiliation",
    "/v1/authors/author_details_by_affiliation?affiliation=University%20of%20Oulu",
    "/v1/authors/top_paper_by_author_count",
    "/v1/authors/paper_details_by_id?paper_id=15",
    "/v1/authors/author_names_by_paper_title_affiliation?title=Inspection%20resistant%20memory%3A%20Architectural%20support%20for%20security%20FROM%20physical%20examination&affiliation=Microsoft%20Research%2C%20USA",
    "/v1/authors/author_journal_info_by_paper_title?title=Decreased%20Saliva%20Secretion%20and%20Down-Regulation%20of%20AQP5%20in%20Submandibular%20Gland%20in%20Irradiated%20Rats",
    "/v1/authors/paper_details_by_author_id_affiliation?author_id=661002&affiliation=Scientific%20Computing%20and%20Imaging%20Institute%2C%20University%20of%20Utah%2C%20UT%2084112%2C%20USA",
    "/v1/authors/paper_count_diff_by_years_journal?year1=2000&year2=2010&short_name=IWC",
    "/v1/authors/journal_details_by_year_journal_ids?year=2013&journal_id1=0&journal_id2=-1&limit=4",
    "/v1/authors/paper_author_conference_by_shortname_year_range?short_name=MICRO&start_year=1971&end_year=1980",
    "/v1/authors/conference_fullname_by_paper_title?title=The%20Dissimilarity%20Representation%20as%20a%20Tool%20for%20Three-Way%20Data%20Classification%3A%20A%202D%20Measure",
    "/v1/authors/conference_homepage_by_paper_title?title=Energy-efficiency%20bounds%20for%20noise-tolerant%20dynamic%20circuits",
    "/v1/authors/paper_conference_by_year?year=2009",
    "/v1/authors/author_names_by_year_keyword?year=2005&keyword=KEY%20WORDS:%20LOAD%20IDE%20SNP%20haplotype%20asso-%20cation%20studies",
    "/v1/authors/distinct_author_count_by_paper_title?title=145%20GROWTH%20HORMONE%20RECEPTORS%20AND%20THE%20ONSET%20OF%20HYPERINSULINEMIA%20IN%20THE%20OBESE%20ZUCKER%20RAT:%20",
    "/v1/authors/author_paper_titles?author_name=Jei%20Keon%20Chae",
    "/v1/authors/author_conference_journal_paper_titles?author_name=Shueh-Lin%20Yau",
    "/v1/authors/conference_short_names_by_full_name?full_name_pattern=International%20Symposium%25",
    "/v1/authors/journal_count_by_homepage?homepage=",
    "/v1/authors/author_names_by_paper_year?year=0",
    "/v1/authors/author_affiliations_by_keyword?keyword=Quantum%20Physics",
    "/v1/authors/conference_full_names_by_criteria?conference_id=0&journal_id=0&year=0",
    "/v1/authors/paper_titles_by_conference_criteria?homepage=&title=",
    "/v1/authors/journal_years_and_full_names_by_short_name?short_name=",
    "/v1/authors/author_affiliations_by_paper_title?title=A%20combined%20search%20for%20the%20standard%20model%20Higgs%20boson%20at%20s%20%3D%201.96%20%C3%85%20TeV",
    "/v1/authors/journal_full_names_by_criteria?year=2001&conference_id=0&journal_id=0",
    "/v1/authors/paper_count_years_by_conference?conference_name=International%20Conference%20on%20Database%20Theory",
    "/v1/authors/journal_names_by_keyword?keyword=Sustainability",
    "/v1/authors/author_names_by_conference_pattern?conference_pattern=%25Workshop%25",
    "/v1/authors/percentage_papers_by_affiliation_pattern?affiliation_pattern=%25INFN%25&title_pattern=%25Charged%20particle%20multiplicity%25",
    "/v1/authors/percentage_journals_by_short_name_pattern?short_name_pattern=ANN%25&year=1989",
    "/v1/authors/paper_count_by_affiliation_pattern?affiliation_pattern=%25Microsoft%20Research%25",
    "/v1/authors/keywords_year_by_title?title=A%20Formal%20Approach%20to%20Service%20Component%20Architecture",
    "/v1/authors/percentage_papers_after_year?year=2000",
    "/v1/authors/paper_authors_by_title?title=Hypermethylation%20of%20the%20%3CI%3ETPEF%2FHPP1%3C%2FI%3E%20Gene%20in%20Primary%20and%20Metastatic%20Colorectal%20Cancers",
    "/v1/authors/journal_full_name_by_paper_title?paper_title=Multiple%20paternity%20in%20a%20natural%20population%20of%20a%20salamander%20with%20long-term%20sperm%20storage",
    "/v1/authors/conference_full_name_by_paper_id?paper_id=5",
    "/v1/authors/paper_titles_by_author_name_pattern?author_name_pattern=Jun%20du%25",
    "/v1/authors/paper_titles_by_author_name_and_affiliation?author_name=Cheng%20Huang&affiliation=Microsoft",
    "/v1/authors/conference_full_name_by_author_name?author_name=Jean-luc%20Hainaut",
    "/v1/authors/journal_full_name_by_author_name?author_name=Andrew%20Cain",
    "/v1/authors/distinct_affiliations_by_paper_title?paper_title=FIBER:%20A%20Generalized%20Framework%20for%20Auto-tuning%20Software",
    "/v1/authors/paper_titles_journal_full_names_before_year?year=1",
    "/v1/authors/paper_titles_authors_by_journal_year?journal_full_name=Neoplasia&year=2007",
    "/v1/authors/paper_titles_authors_by_affiliation_year?affiliation=Soongsil%20University&year=2000",
    "/v1/authors/paper_titles_authors_by_keyword_year_range?keyword=%25optical%20properties%25&start_year=2000&end_year=2005",
    "/v1/authors/proportion_papers_by_journal_year?journal_full_name=International%20Congress%20Series",
    "/v1/authors/distinct_authors_papers_by_year_conference_journal?conference_id=0&journal_id=0&year=1997",
    "/v1/authors/author_count_by_affiliation?affiliation=Otterbein%20University",
    "/v1/authors/count_papers_by_conference_full_name?full_name=Mathematics%20of%20Program%20Construction",
    "/v1/authors/count_papers_by_year_and_conference_short_name?year=2000&short_name=SSPR",
    "/v1/authors/distinct_paper_titles_by_journal_and_year?full_name=Theoretical%20Computer%20Science&year=2003",
    "/v1/authors/author_names_by_affiliation_and_paper_title?affiliation=Asan%20Medical%20Center%2C%20University%20of%20Ulsan%20College%20of%20Medicine%2C%20Seoul%2C%20Korea&title=A%20Randomized%20Comparison%20of%20Sirolimus-%20Versus%20Paclitaxel-Eluting%20Stent%20Implantation%20in%20Patients%20With%20Diabetes%20Mellitus",
    "/v1/authors/average_papers_per_year_given_conference_year_range?conference_name=Information%20and%20Knowledge%20Engineering&start_year=2002&end_year=2010",
    "/v1/authors/paper_count_difference_between_conferences_year_range?conference_name_1=Informatik%20%26%20Schule&conference_name_2=International%20Conference%20on%20Supercomputing&start_year=1990&end_year=2001",
    "/v1/authors/journal_short_name_given_full_name?full_name=Software%20-%20Concepts%20and%20Tools%20/%20Structured%20Programming",
    "/v1/authors/journal_full_name_given_paper_title?paper_title=Education%2C%20democracy%20and%20growth",
    "/v1/authors/paper_count_given_journal_year?journal_name=IEEE%20Transactions%20on%20Nuclear%20Science&year=1999",
    "/v1/authors/paper_titles_given_journal_year?journal_name=IEEE%20Transactions%20on%20Pattern%20Analysis%20and%20Machine%20Intelligence&year=2011",
    "/v1/authors/paper_keywords_given_journal_year?journal_name=Modeling%20Identification%20and%20Control&year=1994",
    "/v1/authors/check_paper_year_zero_given_author?author_name=Zvezdan%20Proti\u0107",
    "/v1/authors/conference_home_pages_by_paper_title?paper_title=Increasing%20the%20Concurrency%20in%20Estelle",
    "/v1/authors/conference_short_name_most_papers_by_year?year=2012",
    "/v1/authors/paper_count_by_conference_short_name_and_year?conference_short_name=ECSQARU&year=2003",
    "/v1/authors/paper_keywords_by_conference_full_name_and_year?conference_full_name=International%20Radar%20Symposium&year=2012",
    "/v1/authors/paper_ratio_by_conference_full_names?conference_full_name_1=International%20Conference%20on%20Thermoelectrics&conference_full_name_2=International%20Conference%20on%20Wireless%20Networks%2C%20Communications%20and%20Mobile%20Computing",
    "/v1/authors/paper_ratio_not_associated_by_author_name?author_name=John%20Van%20Reenen",
    "/v1/authors/earliest_paper_title_after_year?year=0",
    "/v1/authors/longest_conference_name",
    "/v1/authors/paper_with_most_authors",
    "/v1/authors/earliest_paper_given_journal_short_name?short_name=TUBERCLE%20LUNG%20DIS",
    "/v1/authors/paper_count_given_conference_year?full_name=Virtual%20Reality%2C%20IEEE%20Annual%20International%20Symposium&year=2012",
    "/v1/authors/affiliations_by_author?name=Mark%20A.%20Musen",
    "/v1/authors/papers_by_year_range?start_year=1960&end_year=1970",
    "/v1/authors/most_papers_year_by_conference?conference_name=Internet%2C%20Multimedia%20Systems%20and%20Applications",
    "/v1/authors/co_authors_by_author_name?author_name=Randall%20Davis",
    "/v1/authors/conferences_by_year?year=2008",
    "/v1/authors/journal_homepage_most_papers",
    "/v1/authors/keyword_proportion_and_author_details?keyword=cancer",
    "/v1/authors/author_name_by_id?author_id=1722",
    "/v1/authors/paper_count_by_year_or_conference_journal?year=0&conference_id=0&journal_id=0",
    "/v1/authors/paper_count_and_diff?year1=2005&year2=2004",
    "/v1/authors/paper_titles_by_journal?journal_name=Ibm%20Journal%20of%20Research%20and%20Development",
    "/v1/authors/author_info_by_paper_title?paper_title=Education%2C%20democracy%20and%20growth",
    "/v1/authors/author_count_by_journal?journal_name=IEEE%20Computer",
    "/v1/authors/paper_count_by_conference_year_range?conference_name=International%20Workshop%20on%20Inductive%20Logic%20Programming&start_year=2001&end_year=2009",
    "/v1/authors/author_paper_ratio_by_year_range?start_year=1990&end_year=2000",
    "/v1/authors/paper_year_journal_by_title?paper_title=Area%20Effects%20in%20Cepaea",
    "/v1/authors/paper_count_journal_year_range?journal_name=Academic%20Medicine&start_year=2005&end_year=2010",
    "/v1/authors/latest_paper_by_author?author_name=Zuliang%20Du",
    "/v1/authors/average_papers_per_year_journal?journal_name=Information%20Sciences",
    "/v1/authors/paper_count_by_author?author_name=Howard%20F.%20Lipson",
    "/v1/authors/keyword_journal_ratio?keyword=Turbulent%20Fluids&journal_name=Physics%20of%20Fluids",
    "/v1/authors/paper_count_by_affiliation?affiliation=University%20of%20Hong%20Kong",
    "/v1/authors/papers_by_author?name=Barrasa",
    "/v1/authors/authors_by_paper?paper_id=2",
    "/v1/authors/authors_keywords_by_paper?paper_id=5",
    "/v1/authors/conferences_by_paper_title?title=2004%20YD5"
]
