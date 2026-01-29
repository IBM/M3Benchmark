from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/book_publishing_company/book_publishing_company.sqlite')
cursor = conn.cursor()

# Endpoint to get the date with the highest total quantity sold
@app.get("/v1/book_publishing_company/highest_sales_date", operation_id="get_highest_sales_date", summary="Retrieves the date on which the highest total quantity of books was sold. This operation calculates the total quantity sold for each date and returns the date with the maximum total sales.")
async def get_highest_sales_date():
    cursor.execute("SELECT ord_date, SUM(qty) FROM sales GROUP BY ord_date ORDER BY SUM(qty) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"date": [], "total_quantity": []}
    return {"date": result[0], "total_quantity": result[1]}

# Endpoint to get the title with the highest quantity sold in a specific year
@app.get("/v1/book_publishing_company/top_title_by_year", operation_id="get_top_title_by_year", summary="Retrieves the title of the book that sold the most copies in a given year. The year is specified in the 'YYYY' format. The operation returns the title of the top-selling book for the provided year.")
async def get_top_title_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.title FROM sales AS T1 INNER JOIN titles AS T2 ON T1.title_id = T2.title_id WHERE STRFTIME('%Y', T1.ord_date) = ? ORDER BY T1.qty DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get titles with specific payment terms
@app.get("/v1/book_publishing_company/titles_by_payment_terms", operation_id="get_titles_by_payment_terms", summary="Retrieves a list of book titles along with their respective prices and publication dates based on the specified payment terms.")
async def get_titles_by_payment_terms(payterms: str = Query(..., description="Payment terms")):
    cursor.execute("SELECT T2.title, T2.price, T2.pubdate FROM sales AS T1 INNER JOIN titles AS T2 ON T1.title_id = T2.title_id WHERE T1.payterms = ?", (payterms,))
    results = cursor.fetchall()
    if not results:
        return {"titles": []}
    return {"titles": [{"title": row[0], "price": row[1], "pubdate": row[2]} for row in results]}

# Endpoint to get titles with specific royalty range and minimum royalty
@app.get("/v1/book_publishing_company/titles_by_royalty_range", operation_id="get_titles_by_royalty_range", summary="Retrieves a list of book titles that fall within a specified royalty range and meet a minimum royalty threshold. The operation filters titles based on the lower range of the royalty and the minimum royalty value provided as input parameters. The result is a set of titles that satisfy these criteria.")
async def get_titles_by_royalty_range(lorange: int = Query(..., description="Lower range of royalty"), royalty: int = Query(..., description="Minimum royalty")):
    cursor.execute("SELECT T1.title FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id WHERE T2.lorange = ? AND T2.royalty >= ?", (lorange, royalty))
    results = cursor.fetchall()
    if not results:
        return {"titles": []}
    return {"titles": [row[0] for row in results]}

# Endpoint to get titles with specific royalty range and title ID
@app.get("/v1/book_publishing_company/titles_by_royalty_range_and_id", operation_id="get_titles_by_royalty_range_and_id", summary="Retrieves the titles that fall within a specified royalty range and match a given title ID. The operation filters the titles based on the lower and upper bounds of the royalty range, and the provided title ID, returning the title and its associated royalty rate.")
async def get_titles_by_royalty_range_and_id(lorange: int = Query(..., description="Lower range of royalty"), hirange: int = Query(..., description="Higher range of royalty"), title_id: str = Query(..., description="Title ID")):
    cursor.execute("SELECT T1.title, T2.royalty FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id WHERE T2.lorange > ? AND T2.hirange < ? AND T1.title_ID = ?", (lorange, hirange, title_id))
    results = cursor.fetchall()
    if not results:
        return {"titles": []}
    return {"titles": [{"title": row[0], "royalty": row[1]} for row in results]}

# Endpoint to get the title with the highest royalty
@app.get("/v1/book_publishing_company/top_royalty_title", operation_id="get_top_royalty_title", summary="Retrieves the title of the book that has the highest royalty rate. The royalty rate is determined by joining the titles table with the roysched table using the title_id field. The result is ordered by the royalty rate in descending order and limited to the top entry.")
async def get_top_royalty_title():
    cursor.execute("SELECT T1.title, T2.lorange FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id ORDER BY T2.royalty DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": [], "lorange": []}
    return {"title": result[0], "lorange": result[1]}

# Endpoint to get titles published in a specific country
@app.get("/v1/book_publishing_company/titles_by_country", operation_id="get_titles_by_country", summary="Retrieves a list of book titles published by companies based in a specified country. The response includes the title and the name of the publishing company.")
async def get_titles_by_country(country: str = Query(..., description="Country")):
    cursor.execute("SELECT T1.title, T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.country = ?", (country,))
    results = cursor.fetchall()
    if not results:
        return {"titles": []}
    return {"titles": [{"title": row[0], "pub_name": row[1]} for row in results]}

# Endpoint to get the maximum year-to-date sales for titles within a specific royalty range
@app.get("/v1/book_publishing_company/max_ytd_sales_by_royalty_range", operation_id="get_max_ytd_sales_by_royalty_range", summary="Retrieves the highest year-to-date sales figure for titles that have a royalty rate within the specified range. The range is defined by a lower and upper limit, which are provided as input parameters.")
async def get_max_ytd_sales_by_royalty_range(lorange: int = Query(..., description="Lower range of royalty"), hirange: int = Query(..., description="Higher range of royalty")):
    cursor.execute("SELECT MAX(T1.ytd_sales) FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id WHERE T2.lorange > ? AND T2.hirange < ?", (lorange, hirange))
    result = cursor.fetchone()
    if not result:
        return {"max_ytd_sales": []}
    return {"max_ytd_sales": result[0]}

# Endpoint to get titles published in a specific year with their notes and publisher name
@app.get("/v1/book_publishing_company/titles_by_year", operation_id="get_titles_by_year", summary="Retrieves a list of book titles published in a specific year, along with their respective notes and the name of the publishing company. The year is provided in 'YYYY' format.")
async def get_titles_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T1.title, T1.notes, T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE STRFTIME('%Y', T1.pubdate) = ?", (year,))
    results = cursor.fetchall()
    if not results:
        return {"titles": []}
    return {"titles": [{"title": row[0], "notes": row[1], "pub_name": row[2]} for row in results]}

# Endpoint to get titles sold in a specific state with quantity greater than a specified amount
@app.get("/v1/book_publishing_company/titles_by_state_and_quantity", operation_id="get_titles_by_state_and_quantity", summary="Retrieves the titles of books sold in a specified state with a quantity greater than a given minimum. This operation allows you to filter the titles based on the quantity sold and the state where the sales occurred. The result includes the title of the book and the corresponding quantity sold.")
async def get_titles_by_state_and_quantity(qty: int = Query(..., description="Minimum quantity"), state: str = Query(..., description="State")):
    cursor.execute("SELECT T1.title, T2.qty FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id INNER JOIN stores AS T3 ON T2.stor_id = T3.stor_id WHERE T2.qty > ? AND T3.state = ?", (qty, state))
    results = cursor.fetchall()
    if not results:
        return {"titles": []}
    return {"titles": [{"title": row[0], "qty": row[1]} for row in results]}

# Endpoint to get the store ID and title of the least sold book in the store with the highest total sales
@app.get("/v1/book_publishing_company/top_store_least_sold_book", operation_id="get_top_store_least_sold_book", summary="Retrieves the store ID and title of the book with the lowest sales in the store that has the highest total sales. This operation identifies the top-performing store based on total sales and then determines the least popular book within that store.")
async def get_top_store_least_sold_book():
    cursor.execute("SELECT T3.stor_id, T2.title FROM sales AS T1 INNER JOIN titles AS T2 ON T1.title_id = T2.title_id INNER JOIN stores AS T3 ON T3.stor_id = T1.stor_id WHERE T3.stor_id = ( SELECT stor_id FROM sales GROUP BY stor_id ORDER BY SUM(qty) DESC LIMIT 1 ) GROUP BY T3.stor_id, T2.title ORDER BY SUM(T1.qty) ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"store_id": [], "title": []}
    return {"store_id": result[0], "title": result[1]}

# Endpoint to get the title, publisher name, royalty range, and royalty for a specific title ID
@app.get("/v1/book_publishing_company/title_details", operation_id="get_title_details", summary="Get the title, publisher name, royalty range, and royalty for a specific title ID")
async def get_title_details(title_id: str = Query(..., description="Title ID of the book")):
    cursor.execute("SELECT T1.title, T3.pub_name, T2.lorange, T2.hirange, T2.royalty FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id INNER JOIN publishers AS T3 ON T1.pub_id = T3.pub_id WHERE T1.title_id = ?", (title_id,))
    result = cursor.fetchone()
    if not result:
        return {"title": [], "pub_name": [], "lorange": [], "hirange": [], "royalty": []}
    return {"title": result[0], "pub_name": result[1], "lorange": result[2], "hirange": result[3], "royalty": result[4]}

# Endpoint to get the percentage of sales with 'Net 30' payment terms for a specific store ID
@app.get("/v1/book_publishing_company/net_30_sales_percentage", operation_id="get_net_30_sales_percentage", summary="Retrieves the percentage of sales with a specific payment term for a given store. The operation calculates the proportion of sales with 'Net 30' payment terms relative to the total sales for the provided store ID.")
async def get_net_30_sales_percentage(stor_id: str = Query(..., description="Store ID")):
    cursor.execute("SELECT T2.stor_name , CAST(SUM(CASE WHEN payterms = 'Net 30' THEN qty ELSE 0 END) AS REAL) * 100 / SUM(qty) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id WHERE T1.stor_id = ? GROUP BY T2.stor_name", (stor_id,))
    result = cursor.fetchone()
    if not result:
        return {"store_name": [], "percentage": []}
    return {"store_name": result[0], "percentage": result[1]}

# Endpoint to get the average year-to-date sales for a specific publisher ID
@app.get("/v1/book_publishing_company/avg_ytd_sales_by_publisher", operation_id="get_avg_ytd_sales_by_publisher", summary="Retrieves the average year-to-date sales for a specific book publisher. The operation requires a publisher ID as input and returns the average sales figure along with the publisher's name.")
async def get_avg_ytd_sales_by_publisher(pub_id: str = Query(..., description="Publisher ID")):
    cursor.execute("SELECT T2.pub_name, AVG(T1.ytd_sales) FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.pub_id = ? GROUP BY T2.pub_name", (pub_id,))
    result = cursor.fetchone()
    if not result:
        return {"pub_name": [], "avg_ytd_sales": []}
    return {"pub_name": result[0], "avg_ytd_sales": result[1]}

# Endpoint to get the first and last names of employees hired before a specific year
@app.get("/v1/book_publishing_company/employees_hired_before_year", operation_id="get_employees_hired_before_year", summary="Retrieves the first and last names of employees who were hired before the specified year. The year is provided in 'YYYY' format.")
async def get_employees_hired_before_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT fname, lname FROM employee WHERE STRFTIME('%Y', hire_date) < ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"fname": row[0], "lname": row[1]} for row in result]}

# Endpoint to get the first and last names and hire date of the employee with the lowest job level
@app.get("/v1/book_publishing_company/lowest_job_level_employee", operation_id="get_lowest_job_level_employee", summary="Retrieves the full name and hire date of the employee with the lowest job level in the book publishing company.")
async def get_lowest_job_level_employee():
    cursor.execute("SELECT fname, lname, hire_date FROM employee ORDER BY job_lvl LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"fname": [], "lname": [], "hire_date": []}
    return {"fname": result[0], "lname": result[1], "hire_date": result[2]}

# Endpoint to get the year with the highest number of employees hired
@app.get("/v1/book_publishing_company/year_with_most_hires", operation_id="get_year_with_most_hires", summary="Retrieves the year in which the book publishing company hired the most employees. The operation returns the year with the highest number of new hires, based on the hire_date field in the employee records.")
async def get_year_with_most_hires():
    cursor.execute("SELECT STRFTIME('%Y', hire_date) FROM employee GROUP BY STRFTIME('%Y', hire_date) ORDER BY COUNT(emp_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the first and last names of employees at the maximum job level
@app.get("/v1/book_publishing_company/employees_at_max_job_level", operation_id="get_employees_at_max_job_level", summary="Retrieves the first and last names of employees who hold the highest job level within the company. This operation returns a list of employees who have reached the maximum job level, providing insight into the most senior staff members.")
async def get_employees_at_max_job_level():
    cursor.execute("SELECT T1.fname, T1.lname FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T1.job_lvl = T2.max_lvl")
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"fname": row[0], "lname": row[1]} for row in result]}

# Endpoint to get the first and last names and hire date of employees with a specific job description
@app.get("/v1/book_publishing_company/employees_by_job_description", operation_id="get_employees_by_job_description", summary="Retrieve the first and last names and hire date of employees who hold a specific job role. The job role is determined by the provided job description.")
async def get_employees_by_job_description(job_desc: str = Query(..., description="Job description")):
    cursor.execute("SELECT T1.fname, T1.lname, T1.hire_date FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T2.job_desc = ?", (job_desc,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"fname": row[0], "lname": row[1], "hire_date": row[2]} for row in result]}

# Endpoint to get the first and last names and publisher name of employees from publishers not in a specific country
@app.get("/v1/book_publishing_company/employees_by_publisher_country", operation_id="get_employees_by_publisher_country", summary="Retrieve the first and last names of employees along with their respective publisher names, excluding those from the specified country.")
async def get_employees_by_publisher_country(country: str = Query(..., description="Country")):
    cursor.execute("SELECT T1.fname, T1.lname, T2.pub_name FROM employee AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.country != ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"fname": row[0], "lname": row[1], "pub_name": row[2]} for row in result]}

# Endpoint to get employee details based on publisher name
@app.get("/v1/book_publishing_company/employee_details_by_publisher", operation_id="get_employee_details_by_publisher", summary="Retrieves the first name, last name, and job description of employees associated with a specific publisher. The publisher is identified by its name.")
async def get_employee_details_by_publisher(pub_name: str = Query(..., description="Publisher name")):
    cursor.execute("SELECT T1.fname, T1.lname, T3.job_desc FROM employee AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id INNER JOIN jobs AS T3 ON T1.job_id = T3.job_id WHERE T2.pub_name = ?", (pub_name,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get distinct publisher names and title types
@app.get("/v1/book_publishing_company/distinct_publisher_title_types", operation_id="get_distinct_publisher_title_types", summary="Retrieves a list of unique publisher names along with their corresponding title types. The data is sorted by publisher name for easy reference.")
async def get_distinct_publisher_title_types():
    cursor.execute("SELECT DISTINCT T2.pub_name, T1.type FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id ORDER BY T2.pub_name")
    result = cursor.fetchall()
    if not result:
        return {"publishers": []}
    return {"publishers": result}

# Endpoint to get the publisher with the most titles in a given year
@app.get("/v1/book_publishing_company/top_publisher_by_year", operation_id="get_top_publisher_by_year", summary="Retrieves the publishing company with the highest number of published titles in a specified year. The year must be provided in the 'YYYY' format.")
async def get_top_publisher_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE STRFTIME('%Y', T1.pubdate) = ? GROUP BY T1.pub_id, T2.pub_name ORDER BY COUNT(T1.title_id) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"publisher": []}
    return {"publisher": result[0]}

# Endpoint to get the most expensive title from a given publisher
@app.get("/v1/book_publishing_company/most_expensive_title_by_publisher", operation_id="get_most_expensive_title_by_publisher", summary="Retrieves the title of the most expensive book published by the specified publisher. The operation filters books by publisher name and sorts them in descending order of price to identify the most expensive title.")
async def get_most_expensive_title_by_publisher(pub_name: str = Query(..., description="Publisher name")):
    cursor.execute("SELECT T1.title FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.pub_name = ? ORDER BY T1.price DESC LIMIT 1", (pub_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get employee details based on job level
@app.get("/v1/book_publishing_company/employee_details_by_job_level", operation_id="get_employee_details_by_job_level", summary="Retrieves the first and last names, along with job descriptions, of employees whose job level exceeds the provided level. This operation facilitates the extraction of specific employee details based on their job level, enabling targeted data retrieval.")
async def get_employee_details_by_job_level(job_lvl: int = Query(..., description="Job level")):
    cursor.execute("SELECT T1.fname, T1.lname, T2.job_desc FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T1.job_lvl > ?", (job_lvl,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get author details based on title type
@app.get("/v1/book_publishing_company/author_details_by_title_type", operation_id="get_author_details_by_title_type", summary="Retrieves the first and last names of authors who have written books of a specific type. The type of book is provided as an input parameter.")
async def get_author_details_by_title_type(title_type: str = Query(..., description="Title type")):
    cursor.execute("SELECT T3.au_fname, T3.au_lname FROM titles AS T1 INNER JOIN titleauthor AS T2 ON T1.title_id = T2.title_id INNER JOIN authors AS T3 ON T2.au_id = T3.au_id WHERE T1.type = ?", (title_type,))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get title IDs and year-to-date sales based on author contract status
@app.get("/v1/book_publishing_company/title_sales_by_contract_status", operation_id="get_title_sales_by_contract_status", summary="Retrieves the unique identifiers and year-to-date sales of titles associated with authors based on their contract status. The contract status is a binary value indicating whether the author is under contract (1) or not (0).")
async def get_title_sales_by_contract_status(contract: int = Query(..., description="Contract status (0 or 1)")):
    cursor.execute("SELECT T1.title_id, T1.ytd_sales FROM titles AS T1 INNER JOIN titleauthor AS T2 ON T1.title_id = T2.title_id INNER JOIN authors AS T3 ON T2.au_id = T3.au_id WHERE T3.contract = ?", (contract,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get the top-selling title by authors with a specific contract status and state
@app.get("/v1/book_publishing_company/top_selling_title_by_contract_state", operation_id="get_top_selling_title_by_contract_state", summary="Retrieves the title of the book that has sold the most copies in the current year, authored by individuals with a specific contract status and residing in a particular state. The contract status can be either active (1) or inactive (0).")
async def get_top_selling_title_by_contract_state(contract: int = Query(..., description="Contract status (0 or 1)"), state: str = Query(..., description="State")):
    cursor.execute("SELECT T1.title FROM titles AS T1 INNER JOIN titleauthor AS T2 ON T1.title_id = T2.title_id INNER JOIN authors AS T3 ON T2.au_id = T3.au_id WHERE T3.contract = ? AND T3.state = ? ORDER BY T1.ytd_sales DESC LIMIT 1", (contract, state))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get author details based on title
@app.get("/v1/book_publishing_company/author_details_by_title", operation_id="get_author_details_by_title", summary="Retrieves the first and last names of the author(s) associated with a specific book title. The title of the book is provided as an input parameter.")
async def get_author_details_by_title(title: str = Query(..., description="Title")):
    cursor.execute("SELECT T3.au_fname, T3.au_lname FROM titles AS T1 INNER JOIN titleauthor AS T2 ON T1.title_id = T2.title_id INNER JOIN authors AS T3 ON T2.au_id = T3.au_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get the percentage of employees with specific job descriptions
@app.get("/v1/book_publishing_company/percentage_employees_by_job_desc", operation_id="get_percentage_employees_by_job_desc", summary="Retrieves the percentage of employees who hold specific job descriptions. The operation calculates this percentage by comparing the count of employees with the provided job descriptions to the total number of employees. The input parameters represent the job descriptions to be considered in the calculation.")
async def get_percentage_employees_by_job_desc(job_desc1: str = Query(..., description="First job description"), job_desc2: str = Query(..., description="Second job description")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.job_desc IN (?, ?) THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.job_id) FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id", (job_desc1, job_desc2))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get titles with above average year-to-date sales
@app.get("/v1/book_publishing_company/titles_above_avg_ytd_sales", operation_id="get_titles_above_avg_ytd_sales", summary="Retrieves a list of book titles that have exceeded the average year-to-date sales across all titles. This operation involves comparing each title's year-to-date sales to the overall average, and only returns those titles with sales figures above this average. The data is sourced from the titles and publishers tables, ensuring accurate and up-to-date sales information.")
async def get_titles_above_avg_ytd_sales():
    cursor.execute("SELECT T1.title FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.ytd_sales > ( SELECT AVG(ytd_sales) FROM titles )")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of publishers in a specific country
@app.get("/v1/book_publishing_company/publisher_count_by_country", operation_id="get_publisher_count_by_country", summary="Retrieves the total number of publishers based in a specified country. The operation filters publishers by the provided country and returns the count.")
async def get_publisher_count_by_country(country: str = Query(..., description="Country of the publisher")):
    cursor.execute("SELECT COUNT(pub_id) FROM publishers WHERE country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get PR information for a specific publisher
@app.get("/v1/book_publishing_company/pr_info_by_publisher", operation_id="get_pr_info_by_publisher", summary="Retrieves detailed PR information for a specific publisher. The operation requires the publisher's name as input and returns the corresponding PR data from the database.")
async def get_pr_info_by_publisher(pub_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT T1.pr_info FROM pub_info AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.pub_name = ?", (pub_name,))
    result = cursor.fetchall()
    if not result:
        return {"pr_info": []}
    return {"pr_info": [row[0] for row in result]}

# Endpoint to get first names of employees with a specific job description
@app.get("/v1/book_publishing_company/employee_names_by_job_desc", operation_id="get_employee_names_by_job_desc", summary="Retrieves the first names of employees who have a specified job description. The job description is provided as an input parameter.")
async def get_employee_names_by_job_desc(job_desc: str = Query(..., description="Job description")):
    cursor.execute("SELECT T1.fname FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T2.job_desc = ?", (job_desc,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [row[0] for row in result]}

# Endpoint to get the maximum level of the most recently hired employee
@app.get("/v1/book_publishing_company/max_level_recent_hire", operation_id="get_max_level_recent_hire", summary="Retrieves the highest job level of the most recently hired employee in the book publishing company. The response is based on the latest hire date and the corresponding job level from the jobs table.")
async def get_max_level_recent_hire():
    cursor.execute("SELECT T2.max_lvl FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id ORDER BY T1.hire_date LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"max_level": []}
    return {"max_level": result[0]}

# Endpoint to get the city with the highest total sales quantity
@app.get("/v1/book_publishing_company/top_sales_city", operation_id="get_top_sales_city", summary="Retrieves the city with the highest total sales quantity by aggregating sales data from all stores and ranking cities based on their total sales volume.")
async def get_top_sales_city():
    cursor.execute("SELECT T2.city FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id GROUP BY T2.city ORDER BY SUM(T1.qty) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the price of the most sold title
@app.get("/v1/book_publishing_company/price_most_sold_title", operation_id="get_price_most_sold_title", summary="Retrieves the price of the book title that has been sold the most. This operation returns the price of the top-selling title, providing valuable insights into the most popular book in terms of sales volume.")
async def get_price_most_sold_title():
    cursor.execute("SELECT T2.price FROM sales AS T1 INNER JOIN titles AS T2 ON T1.title_id = T2.title_id ORDER BY T1.qty DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get store names selling a specific title
@app.get("/v1/book_publishing_company/store_names_by_title", operation_id="get_store_names_by_title", summary="Retrieves the names of stores that sell a specific book title. The operation filters the sales data based on the provided book title and returns the corresponding store names.")
async def get_store_names_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT T2.stor_name FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id INNER JOIN titles AS T3 ON T1.title_id = T3.title_id WHERE T3.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"store_names": []}
    return {"store_names": [row[0] for row in result]}

# Endpoint to get the count of stores selling titles in a specific state
@app.get("/v1/book_publishing_company/store_count_by_state", operation_id="get_store_count_by_state", summary="Retrieve the total number of stores in a specified state that sell titles. The state is provided as an input parameter.")
async def get_store_count_by_state(state: str = Query(..., description="State where the stores are located")):
    cursor.execute("SELECT COUNT(T1.stor_id) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id INNER JOIN titles AS T3 ON T1.title_id = T3.title_id WHERE T2.state = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country of the publisher for a specific title
@app.get("/v1/book_publishing_company/publisher_country_by_title", operation_id="get_publisher_country_by_title", summary="Retrieves the country of the publisher associated with a specific book title. The operation uses the provided book title to look up the publisher's country in the database.")
async def get_publisher_country_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT T2.country FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the publisher name of the most expensive book
@app.get("/v1/book_publishing_company/most_expensive_book_publisher", operation_id="get_most_expensive_book_publisher", summary="Retrieves the name of the publishing company that has published the most expensive book. The operation returns the publisher's name based on the highest price of a book in the database.")
async def get_most_expensive_book_publisher():
    cursor.execute("SELECT T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id ORDER BY T1.price DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"publisher": []}
    return {"publisher": result[0]}

# Endpoint to get the count of distinct publishers in a specific country with books priced above a certain amount
@app.get("/v1/book_publishing_company/count_publishers_by_country_and_price", operation_id="get_count_publishers_by_country_and_price", summary="Retrieve the number of unique publishers from a specified country who have published books priced higher than a given amount.")
async def get_count_publishers_by_country_and_price(country: str = Query(..., description="Country of the publisher"), min_price: float = Query(..., description="Minimum price of the book")):
    cursor.execute("SELECT COUNT(DISTINCT T1.pub_id) FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.country = ? AND T1.price > ?", (country, min_price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the notes of the top 3 best-selling books
@app.get("/v1/book_publishing_company/top_3_best_selling_books_notes", operation_id="get_top_3_best_selling_books_notes", summary="Retrieves the notes of the top three best-selling books, as determined by sales quantity. The notes provide additional information about each book, such as its genre, author background, or critical reception.")
async def get_top_3_best_selling_books_notes():
    cursor.execute("SELECT T1.notes FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id ORDER BY T2.qty DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"notes": []}
    return {"notes": [row[0] for row in result]}

# Endpoint to get the total quantity sold for a specific book type in a specific state
@app.get("/v1/book_publishing_company/total_quantity_sold_by_state_and_type", operation_id="get_total_quantity_sold_by_state_and_type", summary="Retrieves the total quantity of books sold for a specific type in a given state. This operation calculates the sum of quantities sold from sales data, considering the specified state and book type. The data is derived from sales records, filtered by the state of the store and the type of the book.")
async def get_total_quantity_sold_by_state_and_type(state: str = Query(..., description="State where the store is located"), book_type: str = Query(..., description="Type of the book")):
    cursor.execute("SELECT SUM(T1.qty) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id INNER JOIN titles AS T3 ON T1.title_id = T3.title_id WHERE T2.state = ? AND T3.type = ?", (state, book_type))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the average quantity sold for a specific book title
@app.get("/v1/book_publishing_company/average_quantity_sold_by_title", operation_id="get_average_quantity_sold_by_title", summary="Retrieves the average quantity sold for a specific book title. This operation calculates the total quantity sold for the specified book title and divides it by the number of times the book was sold, providing a precise average quantity sold.")
async def get_average_quantity_sold_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT CAST(SUM(T2.qty) AS REAL) / COUNT(T1.title_id) FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"average_quantity": []}
    return {"average_quantity": result[0]}

# Endpoint to get the average job level and the difference from the maximum level for a specific job description
@app.get("/v1/book_publishing_company/average_job_level_and_difference", operation_id="get_average_job_level_and_difference", summary="Retrieves the average job level and the difference from the maximum level for a specific job description. The job description is used to filter the data and calculate the average job level. The difference is then determined by subtracting the average job level from the maximum level for the specified job description.")
async def get_average_job_level_and_difference(job_desc: str = Query(..., description="Job description")):
    cursor.execute("SELECT AVG(T2.job_lvl), T1.max_lvl - AVG(T2.job_lvl) FROM jobs AS T1 INNER JOIN employee AS T2 ON T1.job_id = T2.job_id WHERE T1.job_desc = ? GROUP BY T2.job_id, T1.max_lvl", (job_desc,))
    result = cursor.fetchall()
    if not result:
        return {"job_levels": []}
    return {"job_levels": [{"average_job_level": row[0], "difference": row[1]} for row in result]}

# Endpoint to get the title of the most expensive book of a specific type
@app.get("/v1/book_publishing_company/most_expensive_book_by_type", operation_id="get_most_expensive_book_by_type", summary="Retrieves the title of the highest-priced book of a specified type. The operation considers all books of the given type and returns the title of the most expensive one.")
async def get_most_expensive_book_by_type(book_type: str = Query(..., description="Type of the book")):
    cursor.execute("SELECT title FROM titles WHERE type = ? ORDER BY price LIMIT 1", (book_type,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the type of the book with the highest advance
@app.get("/v1/book_publishing_company/book_type_with_highest_advance", operation_id="get_book_type_with_highest_advance", summary="Retrieves the type of the book that has the highest advance payment. The operation returns the book type with the highest advance payment from the available titles.")
async def get_book_type_with_highest_advance():
    cursor.execute("SELECT type FROM titles ORDER BY advance DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"type": []}
    return {"type": result[0]}

# Endpoint to get the royalty of the book with the highest year-to-date sales
@app.get("/v1/book_publishing_company/royalty_of_highest_ytd_sales", operation_id="get_royalty_of_highest_ytd_sales", summary="Retrieves the royalty rate for the book that has achieved the highest year-to-date sales. This operation provides a single royalty rate value, offering insight into the top-performing title in terms of sales.")
async def get_royalty_of_highest_ytd_sales():
    cursor.execute("SELECT royalty FROM titles ORDER BY ytd_sales DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"royalty": []}
    return {"royalty": result[0]}

# Endpoint to get the job level of an employee by last name
@app.get("/v1/book_publishing_company/employee_job_level_by_last_name", operation_id="get_employee_job_level_by_last_name", summary="Retrieves the job level of an employee based on the provided last name. The input parameter specifies the last name of the employee to search for in the database.")
async def get_employee_job_level_by_last_name(lname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT job_lvl FROM employee WHERE lname = ?", (lname,))
    result = cursor.fetchone()
    if not result:
        return {"job_level": []}
    return {"job_level": result[0]}

# Endpoint to get the employee ID with a specific middle initial, ordered by job level in descending order and limited to one result
@app.get("/v1/book_publishing_company/employee_id_by_minit", operation_id="get_employee_id_by_minit", summary="Retrieves the employee ID of the highest-ranking employee with a specific middle initial. The search is limited to one result.")
async def get_employee_id_by_minit(minit: str = Query(..., description="Middle initial of the employee")):
    cursor.execute("SELECT emp_id FROM employee WHERE minit = ? ORDER BY job_lvl DESC LIMIT 1", (minit,))
    result = cursor.fetchone()
    if not result:
        return {"emp_id": []}
    return {"emp_id": result[0]}

# Endpoint to get the contract of authors based on a specific book title
@app.get("/v1/book_publishing_company/author_contract_by_title", operation_id="get_author_contract_by_title", summary="Retrieves the contract details of authors associated with a specified book title. The operation filters authors based on the provided book title and returns their respective contract information.")
async def get_author_contract_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT T1.contract FROM authors AS T1 INNER JOIN titleauthor AS T2 ON T1.au_id = T2.au_id INNER JOIN titles AS T3 ON T2.title_id = T3.title_id WHERE T3.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"contracts": []}
    return {"contracts": [row[0] for row in result]}

# Endpoint to get the job description of an employee based on their full name
@app.get("/v1/book_publishing_company/job_description_by_full_name", operation_id="get_job_description_by_full_name", summary="Retrieves the job description of an employee in the book publishing company by matching their full name (first, middle initial, and last) in the employee database. The job description is fetched from the associated job record.")
async def get_job_description_by_full_name(fname: str = Query(..., description="First name of the employee"), minit: str = Query(..., description="Middle initial of the employee"), lname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.job_desc FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T1.fname = ? AND T1.minit = ? AND T1.lname = ?", (fname, minit, lname))
    result = cursor.fetchone()
    if not result:
        return {"job_desc": []}
    return {"job_desc": result[0]}

# Endpoint to get the difference between the maximum job level and the employee's job level based on their full name
@app.get("/v1/book_publishing_company/job_level_difference_by_full_name", operation_id="get_job_level_difference_by_full_name", summary="Retrieve the difference between the highest job level in the company and the job level of a specific employee, identified by their full name. This operation provides a measure of the employee's position relative to the top job level in the company.")
async def get_job_level_difference_by_full_name(fname: str = Query(..., description="First name of the employee"), minit: str = Query(..., description="Middle initial of the employee"), lname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.max_lvl - T1.job_lvl FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T1.fname = ? AND T1.minit = ? AND T1.lname = ?", (fname, minit, lname))
    result = cursor.fetchone()
    if not result:
        return {"level_difference": []}
    return {"level_difference": result[0]}

# Endpoint to get the notes of titles based on a specific order date
@app.get("/v1/book_publishing_company/title_notes_by_order_date", operation_id="get_title_notes_by_order_date", summary="Retrieves the notes associated with titles that were ordered on a specific date. The operation filters the titles based on the provided order date and returns the corresponding notes.")
async def get_title_notes_by_order_date(order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.notes FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id WHERE STRFTIME('%Y-%m-%d', T2.ord_date) = ?", (order_date,))
    result = cursor.fetchall()
    if not result:
        return {"notes": []}
    return {"notes": [row[0] for row in result]}

# Endpoint to get distinct types of titles based on a specific order date
@app.get("/v1/book_publishing_company/distinct_title_types_by_order_date", operation_id="get_distinct_title_types_by_order_date", summary="Retrieve the unique types of titles that were ordered on a specific date. The operation filters the titles based on the provided order date and returns the distinct title types.")
async def get_distinct_title_types_by_order_date(order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T1.type FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id WHERE STRFTIME('%Y-%m-%d', T2.ord_date) = ?", (order_date,))
    result = cursor.fetchall()
    if not result:
        return {"types": []}
    return {"types": [row[0] for row in result]}

# Endpoint to get the PR information of publishers based on a specific country
@app.get("/v1/book_publishing_company/pr_info_by_country", operation_id="get_pr_info_by_country", summary="Retrieves detailed PR information for publishers based in a specified country. The operation filters publishers by the provided country and returns their corresponding PR information.")
async def get_pr_info_by_country(country: str = Query(..., description="Country of the publisher")):
    cursor.execute("SELECT T1.pr_info FROM pub_info AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"pr_info": []}
    return {"pr_info": [row[0] for row in result]}

# Endpoint to get the publisher name based on a specific book title
@app.get("/v1/book_publishing_company/publisher_name_by_title", operation_id="get_publisher_name_by_title", summary="Retrieves the name of the publishing company associated with a given book title. The operation searches for the specified book title in the titles table and returns the corresponding publisher name from the publishers table.")
async def get_publisher_name_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"pub_name": []}
    return {"pub_name": result[0]}

# Endpoint to get the city of the publisher based on an employee's full name
@app.get("/v1/book_publishing_company/publisher_city_by_employee_full_name", operation_id="get_publisher_city_by_employee_full_name", summary="Retrieves the city of the publishing company associated with a specific employee, identified by their full name (first, middle initial, and last).")
async def get_publisher_city_by_employee_full_name(fname: str = Query(..., description="First name of the employee"), minit: str = Query(..., description="Middle initial of the employee"), lname: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.city FROM employee AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.fname = ? AND T1.minit = ? AND T1.lname = ?", (fname, minit, lname))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the count of orders from a specific city
@app.get("/v1/book_publishing_company/order_count_by_city", operation_id="get_order_count_by_city", summary="Retrieves the total number of orders placed from a specified city. The operation calculates the count based on the provided city name, which is used to filter the sales data.")
async def get_order_count_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(T1.ord_num) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id WHERE T2.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage difference in quantity sold between two cities in a specific year
@app.get("/v1/book_publishing_company/quantity_difference_by_cities_year", operation_id="get_quantity_difference_by_cities_year", summary="Retrieve the percentage difference in the total quantity of sales between two specified cities in a given year. The calculation is based on the sum of quantities sold in each city, with the result expressed as a percentage of the total quantity sold in the first city.")
async def get_quantity_difference_by_cities_year(city1: str = Query(..., description="First city name"), city2: str = Query(..., description="Second city name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.city = ? THEN qty END) - SUM(CASE WHEN T2.city = ? THEN qty END) AS REAL) * 100 / SUM(CASE WHEN T2.city = ? THEN qty END) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id WHERE STRFTIME('%Y', T1.ord_date) = ?", (city1, city2, city1, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the percentage difference in the number of employees between two job descriptions
@app.get("/v1/book_publishing_company/employee_count_difference_by_job_desc", operation_id="get_employee_count_difference_by_job_desc", summary="Retrieves the percentage difference in the number of employees between two specified job descriptions. This operation compares the total count of employees in the first job description with that of the second job description, and returns the percentage difference relative to the total number of employees across all job descriptions.")
async def get_employee_count_difference_by_job_desc(job_desc1: str = Query(..., description="First job description"), job_desc2: str = Query(..., description="Second job description")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.job_desc = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.job_desc = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.job_id) FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id", (job_desc1, job_desc2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get employee details hired between specific years
@app.get("/v1/book_publishing_company/employee_details_by_hire_years", operation_id="get_employee_details_by_hire_years", summary="Retrieves the full names of employees hired between the specified start and end years, sorted by their job level in descending order.")
async def get_employee_details_by_hire_years(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT fname, minit, lname FROM employee WHERE STRFTIME('%Y', hire_date) BETWEEN ? AND ? ORDER BY job_lvl DESC", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get titles with royalty above the average
@app.get("/v1/book_publishing_company/titles_above_average_royalty", operation_id="get_titles_above_average_royalty", summary="Retrieves a list of unique book titles, along with their respective types and prices, that have a royalty rate higher than the average royalty rate. This is achieved by joining the 'titles' and 'roysched' tables on the 'title_id' field and filtering for titles with a royalty rate above the average.")
async def get_titles_above_average_royalty():
    cursor.execute("SELECT DISTINCT T1.title, T1.type, T1.price FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id WHERE T2.royalty > ( SELECT CAST(SUM(royalty) AS REAL) / COUNT(title_id) FROM roysched )")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get titles with sales quantity below the average in a specific year
@app.get("/v1/book_publishing_company/titles_below_average_sales_quantity", operation_id="get_titles_below_average_sales_quantity", summary="Retrieves a list of unique book titles, their types, and prices that have sold below the average sales quantity for a specified year. The year is provided in 'YYYY' format.")
async def get_titles_below_average_sales_quantity(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T1.title, T1.type, T1.price FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id WHERE T2.ord_date LIKE ? AND T2.Qty < ( SELECT CAST(SUM(T4.qty) AS REAL) / COUNT(T3.title_id) FROM titles AS T3 INNER JOIN sales AS T4 ON T3.title_id = T4.title_id )", (year + '%',))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get titles from a specific publisher ordered by price
@app.get("/v1/book_publishing_company/titles_by_publisher_ordered_by_price", operation_id="get_titles_by_publisher_ordered_by_price", summary="Retrieves a list of titles from a specific publisher, sorted by price in ascending order. The publisher is identified by its name, which is provided as an input parameter.")
async def get_titles_by_publisher_ordered_by_price(pub_name: str = Query(..., description="Publisher name")):
    cursor.execute("SELECT T1.title, T1.type, T1.price FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.pub_name = ? ORDER BY T1.price", (pub_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get titles from publishers in a specific country ordered by royalty
@app.get("/v1/book_publishing_company/titles_by_country_ordered_by_royalty", operation_id="get_titles_by_country_ordered_by_royalty", summary="Retrieves a list of book titles from publishers based in a specified country, sorted in descending order by the royalty rate. The royalty rate is determined by the royalties schedule for each title.")
async def get_titles_by_country_ordered_by_royalty(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT T1.title FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id INNER JOIN roysched AS T3 ON T1.title_id = T3.title_id WHERE T2.country = ? ORDER BY T1.royalty DESC", (country,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get the difference in average royalty between titles from a specific country and others
@app.get("/v1/book_publishing_company/royalty_difference_by_country", operation_id="get_royalty_difference_by_country", summary="Retrieve the difference in average royalty between titles from a specified country and those from other countries. The calculation is based on the sum of royalties for titles from the specified country and the sum of royalties for titles from all other countries. The result is the difference between these two averages.")
async def get_royalty_difference_by_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT (CAST(SUM(CASE WHEN T2.country = ? THEN T1.royalty ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.country = ? THEN 1 ELSE 0 END)) - (CAST(SUM(CASE WHEN T2.country != ? THEN T1.royalty ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.country != ? THEN 1 ELSE 0 END)) FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id INNER JOIN roysched AS T3 ON T1.title_id = T3.title_id", (country, country, country, country))
    result = cursor.fetchone()
    if not result:
        return {"royalty_difference": []}
    return {"royalty_difference": result[0]}

# Endpoint to get the difference in average job level between employees in a specific country and others for a specific job description
@app.get("/v1/book_publishing_company/job_level_difference_by_country_job_desc", operation_id="get_job_level_difference_by_country_job_desc", summary="Retrieve the difference in average job level between employees in a specific country and those in other countries for a given job description. This operation calculates the average job level for employees in the specified country and job description, and then subtracts the average job level for employees in other countries with the same job description. The result provides insights into the relative job level distribution for the specified job description across different countries.")
async def get_job_level_difference_by_country_job_desc(country: str = Query(..., description="Country name"), job_desc: str = Query(..., description="Job description")):
    cursor.execute("SELECT (CAST(SUM(CASE WHEN T1.country = ? THEN job_lvl ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.country = ? THEN 1 ELSE 0 END)) - (CAST(SUM(CASE WHEN T1.country != ? THEN job_lvl ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.country != ? THEN 1 ELSE 0 END)) FROM publishers AS T1 INNER JOIN employee AS T2 ON T1.pub_id = T2.pub_id INNER JOIN jobs AS T3 ON T2.job_id = T3.job_id WHERE T3.job_desc = ?", (country, country, country, country, job_desc))
    result = cursor.fetchone()
    if not result:
        return {"job_level_difference": []}
    return {"job_level_difference": result[0]}

# Endpoint to get titles, publisher names, and prices based on specific notes
@app.get("/v1/book_publishing_company/titles_publisher_price_by_notes", operation_id="get_titles_publisher_price_by_notes", summary="Retrieves a list of book titles, their respective publishers, and prices that match the provided notes. The notes parameter is used to filter the results.")
async def get_titles_publisher_price_by_notes(notes: str = Query(..., description="Notes associated with the title")):
    cursor.execute("SELECT T1.title, T2.pub_name, T1.price FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.notes = ?", (notes,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get titles, publisher names, and year-to-date sales based on specific notes
@app.get("/v1/book_publishing_company/titles_publisher_ytd_sales_by_notes", operation_id="get_titles_publisher_ytd_sales_by_notes", summary="Retrieves a list of book titles, their respective publishers, and year-to-date sales figures for titles that match the provided notes. This operation is useful for obtaining sales data for specific titles based on their associated notes.")
async def get_titles_publisher_ytd_sales_by_notes(notes: str = Query(..., description="Notes associated with the title")):
    cursor.execute("SELECT T1.title, T2.pub_name, T1.ytd_sales FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.notes = ?", (notes,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get top 5 titles with sales quantity above the average in a specific state
@app.get("/v1/book_publishing_company/top_titles_above_avg_sales_by_state", operation_id="get_top_titles_above_avg_sales_by_state", summary="Retrieves the top five titles that have sold more than the average quantity in a specified state. The titles are ranked by their sales quantity in descending order.")
async def get_top_titles_above_avg_sales_by_state(state: str = Query(..., description="State of the publisher")):
    cursor.execute("SELECT T1.title FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id INNER JOIN publishers AS T3 ON T1.pub_id = T3.pub_id WHERE T2.qty > ( SELECT CAST(SUM(qty) AS REAL) / COUNT(title_id) FROM sales ) AND T3.state = ? ORDER BY T2.qty DESC LIMIT 5", (state,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

api_calls = [
    "/v1/book_publishing_company/highest_sales_date",
    "/v1/book_publishing_company/top_title_by_year?year=1992",
    "/v1/book_publishing_company/titles_by_payment_terms?payterms=ON%20invoice",
    "/v1/book_publishing_company/titles_by_royalty_range?lorange=0&royalty=10",
    "/v1/book_publishing_company/titles_by_royalty_range_and_id?lorange=10000&hirange=50000&title_id=BU2075",
    "/v1/book_publishing_company/top_royalty_title",
    "/v1/book_publishing_company/titles_by_country?country=USA",
    "/v1/book_publishing_company/max_ytd_sales_by_royalty_range?lorange=20000&hirange=20000",
    "/v1/book_publishing_company/titles_by_year?year=1991",
    "/v1/book_publishing_company/titles_by_state_and_quantity?qty=20&state=CA",
    "/v1/book_publishing_company/top_store_least_sold_book",
    "/v1/book_publishing_company/title_details?title_id=BU2075",
    "/v1/book_publishing_company/net_30_sales_percentage?stor_id=7066",
    "/v1/book_publishing_company/avg_ytd_sales_by_publisher?pub_id=0877",
    "/v1/book_publishing_company/employees_hired_before_year?year=1990",
    "/v1/book_publishing_company/lowest_job_level_employee",
    "/v1/book_publishing_company/year_with_most_hires",
    "/v1/book_publishing_company/employees_at_max_job_level",
    "/v1/book_publishing_company/employees_by_job_description?job_desc=Chief%20Financial%20Officier",
    "/v1/book_publishing_company/employees_by_publisher_country?country=USA",
    "/v1/book_publishing_company/employee_details_by_publisher?pub_name=GGG%26G",
    "/v1/book_publishing_company/distinct_publisher_title_types",
    "/v1/book_publishing_company/top_publisher_by_year?year=1991",
    "/v1/book_publishing_company/most_expensive_title_by_publisher?pub_name=Binnet%20%26%20Hardley",
    "/v1/book_publishing_company/employee_details_by_job_level?job_lvl=200",
    "/v1/book_publishing_company/author_details_by_title_type?title_type=business",
    "/v1/book_publishing_company/title_sales_by_contract_status?contract=0",
    "/v1/book_publishing_company/top_selling_title_by_contract_state?contract=0&state=CA",
    "/v1/book_publishing_company/author_details_by_title?title=Sushi%2C%20Anyone%3F",
    "/v1/book_publishing_company/percentage_employees_by_job_desc?job_desc1=Editor&job_desc2=Designer",
    "/v1/book_publishing_company/titles_above_avg_ytd_sales",
    "/v1/book_publishing_company/publisher_count_by_country?country=USA",
    "/v1/book_publishing_company/pr_info_by_publisher?pub_name=New%20Moon%20Books",
    "/v1/book_publishing_company/employee_names_by_job_desc?job_desc=Managing%20Editor",
    "/v1/book_publishing_company/max_level_recent_hire",
    "/v1/book_publishing_company/top_sales_city",
    "/v1/book_publishing_company/price_most_sold_title",
    "/v1/book_publishing_company/store_names_by_title?title=Life%20Without%20Fear",
    "/v1/book_publishing_company/store_count_by_state?state=Massachusetts",
    "/v1/book_publishing_company/publisher_country_by_title?title=Life%20Without%20Fear",
    "/v1/book_publishing_company/most_expensive_book_publisher",
    "/v1/book_publishing_company/count_publishers_by_country_and_price?country=USA&min_price=15",
    "/v1/book_publishing_company/top_3_best_selling_books_notes",
    "/v1/book_publishing_company/total_quantity_sold_by_state_and_type?state=Massachusetts&book_type=business",
    "/v1/book_publishing_company/average_quantity_sold_by_title?title=Life%20Without%20Fear",
    "/v1/book_publishing_company/average_job_level_and_difference?job_desc=Managing%20Editor",
    "/v1/book_publishing_company/most_expensive_book_by_type?book_type=business",
    "/v1/book_publishing_company/book_type_with_highest_advance",
    "/v1/book_publishing_company/royalty_of_highest_ytd_sales",
    "/v1/book_publishing_company/employee_job_level_by_last_name?lname=O%27Rourke",
    "/v1/book_publishing_company/employee_id_by_minit?minit=",
    "/v1/book_publishing_company/author_contract_by_title?title=Sushi%2C%20Anyone%3F",
    "/v1/book_publishing_company/job_description_by_full_name?fname=Pedro&minit=S&lname=Afonso",
    "/v1/book_publishing_company/job_level_difference_by_full_name?fname=Diego&minit=W&lname=Roel",
    "/v1/book_publishing_company/title_notes_by_order_date?order_date=1994-09-14",
    "/v1/book_publishing_company/distinct_title_types_by_order_date?order_date=1993-05-29",
    "/v1/book_publishing_company/pr_info_by_country?country=France",
    "/v1/book_publishing_company/publisher_name_by_title?title=Silicon%20Valley%20Gastronomic%20Treats",
    "/v1/book_publishing_company/publisher_city_by_employee_full_name?fname=Victoria&minit=P&lname=Ashworth",
    "/v1/book_publishing_company/order_count_by_city?city=Remulade",
    "/v1/book_publishing_company/quantity_difference_by_cities_year?city1=Fremont&city2=Portland&year=1993",
    "/v1/book_publishing_company/employee_count_difference_by_job_desc?job_desc1=publisher&job_desc2=designer",
    "/v1/book_publishing_company/employee_details_by_hire_years?start_year=1990&end_year=1995",
    "/v1/book_publishing_company/titles_above_average_royalty",
    "/v1/book_publishing_company/titles_below_average_sales_quantity?year=1994",
    "/v1/book_publishing_company/titles_by_publisher_ordered_by_price?pub_name=New%20Moon%20Books",
    "/v1/book_publishing_company/titles_by_country_ordered_by_royalty?country=USA",
    "/v1/book_publishing_company/royalty_difference_by_country?country=USA",
    "/v1/book_publishing_company/job_level_difference_by_country_job_desc?country=USA&job_desc=Managing%20Editor",
    "/v1/book_publishing_company/titles_publisher_price_by_notes?notes=Helpful%20hints%20on%20how%20to%20use%20your%20electronic%20resources%20to%20the%20best%20advantage.",
    "/v1/book_publishing_company/titles_publisher_ytd_sales_by_notes?notes=Carefully%20researched%20study%20of%20the%20effects%20of%20strong%20emotions%20on%20the%20body.%20Metabolic%20charts%20included.",
    "/v1/book_publishing_company/top_titles_above_avg_sales_by_state?state=CA"
]
