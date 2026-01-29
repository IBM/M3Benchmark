from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/retail_complains/retail_complains.sqlite')
cursor = conn.cursor()

# Endpoint to get the date received for the latest service time
@app.get("/v1/retail_complains/latest_date_received", operation_id="get_latest_date_received", summary="Retrieves the date received for the most recent service time entry in the call center logs. This operation provides the latest date on which a service was received, offering insights into the most recent service activity.")
async def get_latest_date_received():
    cursor.execute("SELECT `Date received` FROM callcenterlogs WHERE ser_time = ( SELECT MAX(ser_time) FROM callcenterlogs )")
    result = cursor.fetchone()
    if not result:
        return {"date_received": []}
    return {"date_received": result[0]}

# Endpoint to get the minimum service time for a given date received
@app.get("/v1/retail_complains/min_service_time_by_date", operation_id="get_min_service_time_by_date", summary="Retrieves the shortest service time for a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_min_service_time_by_date(date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT MIN(ser_time) FROM callcenterlogs WHERE `Date received` = ?", (date_received,))
    result = cursor.fetchone()
    if not result:
        return {"min_service_time": []}
    return {"min_service_time": result[0]}

# Endpoint to compare priorities of two complaint IDs
@app.get("/v1/retail_complains/compare_complaint_priorities", operation_id="compare_complaint_priorities", summary="This operation compares the priority levels of two specified complaint IDs and returns the one with the higher priority. The input parameters represent the complaint IDs to be compared.")
async def compare_complaint_priorities(complaint_id1: str = Query(..., description="First complaint ID"), complaint_id2: str = Query(..., description="Second complaint ID")):
    cursor.execute("SELECT CASE WHEN SUM(CASE WHEN `Complaint ID` = ? THEN priority END) > SUM(CASE WHEN `Complaint ID` = ? THEN priority END) THEN ? ELSE ? END FROM callcenterlogs", (complaint_id1, complaint_id2, complaint_id1, complaint_id2))
    result = cursor.fetchone()
    if not result:
        return {"higher_priority_complaint": []}
    return {"higher_priority_complaint": result[0]}

# Endpoint to get client names for a given year
@app.get("/v1/retail_complains/client_names_by_year", operation_id="get_client_names_by_year", summary="Retrieves the first, middle, and last names of clients who have a year greater than the provided year.")
async def get_client_names_by_year(year: int = Query(..., description="Year")):
    cursor.execute("SELECT first, middle, last FROM client WHERE year > ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"client_names": []}
    return {"client_names": result}

# Endpoint to get the count of client IDs based on first and last name
@app.get("/v1/retail_complains/count_client_ids_by_name", operation_id="get_count_client_ids_by_name", summary="Retrieves the total number of unique client IDs associated with a specific first and last name. This operation is useful for determining the number of clients with a given name.")
async def get_count_client_ids_by_name(first: str = Query(..., description="First name of the client"), last: str = Query(..., description="Last name of the client")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.last = ?", (first, last))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sub-product for a client based on name and date received
@app.get("/v1/retail_complains/sub_product_by_name_and_date", operation_id="get_sub_product_by_name_and_date", summary="Retrieves the sub-product associated with a specific client, identified by their first and last name and the date their complaint was received. The sub-product is determined based on the client's ID and the date of the corresponding event.")
async def get_sub_product_by_name_and_date(first: str = Query(..., description="First name of the client"), last: str = Query(..., description="Last name of the client"), date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.`Sub-product` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.last = ? AND T2.`Date received` = ?", (first, last, date_received))
    result = cursor.fetchall()
    if not result:
        return {"sub_products": []}
    return {"sub_products": result}

# Endpoint to check consumer consent provided for a client based on name and date received
@app.get("/v1/retail_complains/consumer_consent_by_name_and_date", operation_id="get_consumer_consent_by_name_and_date", summary="This operation verifies if a client has provided consumer consent based on their first and last name and the date the event was received. The client's first and last names, along with the date received in 'YYYY-MM-DD' format, are used to determine the consent status. The operation returns 'Yes' if the client has provided consent and 'No' otherwise.")
async def get_consumer_consent_by_name_and_date(first: str = Query(..., description="First name of the client"), last: str = Query(..., description="Last name of the client"), date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CASE WHEN T2.`Consumer consent provided?` IN (NULL, 'N/A', 'Empty') THEN 'No' ELSE 'Yes' END FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.last = ? AND T2.`Date received` = ?", (first, last, date_received))
    result = cursor.fetchone()
    if not result:
        return {"consumer_consent": []}
    return {"consumer_consent": result[0]}

# Endpoint to calculate the number of days between date sent to company and date received for a client based on name and date received
@app.get("/v1/retail_complains/days_between_dates_by_name_and_date", operation_id="get_days_between_dates_by_name_and_date", summary="Retrieves the number of days between the date a complaint was sent to the company and the date it was received for a specific client. The client is identified by their first and last name, and the date received is provided in the 'YYYY-MM-DD' format.")
async def get_days_between_dates_by_name_and_date(first: str = Query(..., description="First name of the client"), last: str = Query(..., description="Last name of the client"), date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT 365 * (strftime('%Y', T2.`Date sent to company`) - strftime('%Y', T2.`Date received`)) + 30 * (strftime('%M', T2.`Date sent to company`) - strftime('%M', T2.`Date received`)) + (strftime('%d', T2.`Date sent to company`) - strftime('%d', T2.`Date received`)) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.last = ? AND T2.`Date received` = ?", (first, last, date_received))
    result = cursor.fetchone()
    if not result:
        return {"days_between": []}
    return {"days_between": result[0]}

# Endpoint to get client names based on date received and server
@app.get("/v1/retail_complains/client_names_by_date_and_server", operation_id="get_client_names_by_date_and_server", summary="Retrieves the full names of clients who had a call on a specific date and server. The date should be provided in 'YYYY-MM-DD' format.")
async def get_client_names_by_date_and_server(date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format"), server: str = Query(..., description="Server name")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T2.`Date received` = ? AND T2.server = ?", (date_received, server))
    result = cursor.fetchall()
    if not result:
        return {"client_names": []}
    return {"client_names": result}

# Endpoint to get service time for a client based on name and date received
@app.get("/v1/retail_complains/service_time_by_name_and_date", operation_id="get_service_time_by_name_and_date", summary="Retrieves the service time for a specific client based on their first and last name and the date their call was received. The service time is obtained by joining the client and call center logs tables using the client's ID. The provided first name, last name, and date received in 'YYYY-MM-DD' format are used to filter the results.")
async def get_service_time_by_name_and_date(first: str = Query(..., description="First name of the client"), last: str = Query(..., description="Last name of the client"), date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.ser_time FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T1.first = ? AND T1.last = ? AND T2.`Date received` = ?", (first, last, date_received))
    result = cursor.fetchall()
    if not result:
        return {"service_times": []}
    return {"service_times": result}

# Endpoint to get the count of issues based on issue type and city
@app.get("/v1/retail_complains/count_issues_by_type_and_city", operation_id="get_count_issues_by_type_and_city", summary="Retrieves the total count of a specific issue type in a given city. The operation requires the issue type and city as input parameters to filter the data and provide an accurate count.")
async def get_count_issues_by_type_and_city(issue: str = Query(..., description="Issue type"), city: str = Query(..., description="City")):
    cursor.execute("SELECT COUNT(T2.Issue) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.Issue = ? AND T1.city = ?", (issue, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client names based on company response to consumer
@app.get("/v1/retail_complains/client_names_by_company_response", operation_id="get_client_names_by_company_response", summary="Retrieves the first, middle, and last names of clients who have received a specific company response to a consumer complaint. The response is determined by the provided 'company_response' parameter.")
async def get_client_names_by_company_response(company_response: str = Query(..., description="Company response to consumer")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Company response to consumer` = ?", (company_response,))
    result = cursor.fetchall()
    if not result:
        return {"client_names": []}
    return {"client_names": result}

# Endpoint to get the count of clients based on timely response and city
@app.get("/v1/retail_complains/count_clients_by_timely_response_and_city", operation_id="get_count_clients_by_timely_response_and_city", summary="Retrieves the count of clients who have a timely response in a specific city. The operation filters clients based on their city and whether their response was timely, then returns the total count of clients that meet these criteria.")
async def get_count_clients_by_timely_response_and_city(timely_response: str = Query(..., description="Timely response"), city: str = Query(..., description="City")):
    cursor.execute("SELECT COUNT(T1.city) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Timely response?` = ? AND T1.city = ?", (timely_response, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of clients based on year, sex, and product
@app.get("/v1/retail_complains/count_clients_by_year_sex_product", operation_id="get_count_clients_by_year_sex_product", summary="Retrieves the total number of clients categorized by the specified year, sex, and product. The year must be provided in 'YYYY' format. The sex and product parameters are used to further filter the count of clients.")
async def get_count_clients_by_year_sex_product(year: str = Query(..., description="Year in 'YYYY' format"), sex: str = Query(..., description="Sex"), product: str = Query(..., description="Product")):
    cursor.execute("SELECT COUNT(T1.sex) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE strftime('%Y', T2.`Date received`) = ? AND T1.sex = ? AND T2.Product = ?", (year, sex, product))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the division based on client's first and last name
@app.get("/v1/retail_complains/division_by_client_name", operation_id="get_division_by_client_name", summary="Retrieves the division associated with a specific client, identified by their first and last names. This operation returns the division to which the client belongs, based on the client's district information.")
async def get_division_by_client_name(first: str = Query(..., description="Client's first name"), last: str = Query(..., description="Client's last name")):
    cursor.execute("SELECT T2.division FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.first = ? AND T1.last = ?", (first, last))
    result = cursor.fetchall()
    if not result:
        return {"division": []}
    return {"division": result}

# Endpoint to get client names based on division and sex
@app.get("/v1/retail_complains/client_names_by_division_and_sex", operation_id="get_client_names_by_division_and_sex", summary="Retrieves the first, middle, and last names of clients who belong to a specific division and have a particular sex. The division and sex are provided as input parameters.")
async def get_client_names_by_division_and_sex(division: str = Query(..., description="Division"), sex: str = Query(..., description="Sex")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.division = ? AND T1.sex = ?", (division, sex))
    result = cursor.fetchall()
    if not result:
        return {"client_names": []}
    return {"client_names": result}

# Endpoint to get the average number of complaints based on year range, city, and product
@app.get("/v1/retail_complains/average_complaints_by_year_range_city_product", operation_id="get_average_complaints_by_year_range_city_product", summary="Retrieves the average number of complaints for a specific product in a given city, calculated over a specified year range. The year range is inclusive and must be provided in 'YYYY' format. The result is the total number of complaints divided by 3.")
async def get_average_complaints_by_year_range_city_product(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), city: str = Query(..., description="City"), product: str = Query(..., description="Product")):
    cursor.execute("SELECT CAST(COUNT(T2.`Complaint ID`) AS REAL) / 3 AS average FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE strftime('%Y', T2.`Date received`) BETWEEN ? AND ? AND T1.city = ? AND T2.Product = ?", (start_year, end_year, city, product))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage change in complaints between two years for a specific city
@app.get("/v1/retail_complains/percentage_change_complaints_by_years_city", operation_id="get_percentage_change_complaints_by_years_city", summary="Retrieves the percentage change in the number of complaints between two specified years for a given city. The calculation is based on the total complaints received in the first year and the second year, with the result expressed as a percentage of the total in the first year.")
async def get_percentage_change_complaints_by_years_city(year1: str = Query(..., description="First year in 'YYYY' format"), year2: str = Query(..., description="Second year in 'YYYY' format"), city: str = Query(..., description="City")):
    cursor.execute("SELECT 100.0 * (SUM(CASE WHEN strftime('%Y', T2.`Date received`) = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN strftime('%Y', T2.`Date received`) = ? THEN 1 ELSE 0 END)) / SUM(CASE WHEN strftime('%Y', T2.`Date received`) = ? THEN 1 ELSE 0 END) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.city = ?", (year1, year2, year2, city))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result[0]}

# Endpoint to get the service time based on client ID and date received
@app.get("/v1/retail_complains/service_time_by_client_id_date", operation_id="get_service_time_by_client_id_date", summary="Retrieves the service time for a specific client on a given date. The operation uses the provided client ID and date received to filter the call center logs and events, returning the service time associated with the matching complaint.")
async def get_service_time_by_client_id_date(client_id: str = Query(..., description="Client ID"), date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.ser_time FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T2.Client_ID = ? AND T1.`Date received` = ?", (client_id, date_received))
    result = cursor.fetchall()
    if not result:
        return {"service_time": []}
    return {"service_time": result}

# Endpoint to get the state based on client email
@app.get("/v1/retail_complains/state_by_client_email", operation_id="get_state_by_client_email", summary="Retrieves the state associated with a specific client based on the provided email address. This operation returns the state information by querying the client and district tables using the client's email as a filter.")
async def get_state_by_client_email(email: str = Query(..., description="Client email")):
    cursor.execute("SELECT T1.state FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.email = ?", (email,))
    result = cursor.fetchall()
    if not result:
        return {"state": []}
    return {"state": result}

# Endpoint to get distinct sub-products for a client based on their name and sex
@app.get("/v1/retail_complains/distinct_sub_products", operation_id="get_distinct_sub_products", summary="Retrieves a unique list of sub-products associated with a client, filtered by their first name, middle name, last name, and sex. This operation is useful for identifying the specific sub-products a client has interacted with, based on their personal details.")
async def get_distinct_sub_products(first: str = Query(..., description="First name of the client"), middle: str = Query(..., description="Middle name of the client"), last: str = Query(..., description="Last name of the client"), sex: str = Query(..., description="Sex of the client")):
    cursor.execute("SELECT DISTINCT T2.`Sub-product` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.middle = ? AND T1.last = ? AND T1.sex = ?", (first, middle, last, sex))
    result = cursor.fetchall()
    if not result:
        return {"sub_products": []}
    return {"sub_products": [row[0] for row in result]}

# Endpoint to get sub-issues for a client based on their name and sex
@app.get("/v1/retail_complains/sub_issues", operation_id="get_sub_issues", summary="Retrieves a list of sub-issues associated with a specific client, identified by their first name, middle name, last name, and sex. The operation filters the client data based on the provided input parameters and returns the corresponding sub-issues.")
async def get_sub_issues(first: str = Query(..., description="First name of the client"), middle: str = Query(..., description="Middle name of the client"), last: str = Query(..., description="Last name of the client"), sex: str = Query(..., description="Sex of the client")):
    cursor.execute("SELECT T2.`Sub-issue` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.middle = ? AND T1.last = ? AND T1.sex = ?", (first, middle, last, sex))
    result = cursor.fetchall()
    if not result:
        return {"sub_issues": []}
    return {"sub_issues": [row[0] for row in result]}

# Endpoint to check consumer consent provided for a client based on their name, sex, and date received
@app.get("/v1/retail_complains/consumer_consent", operation_id="get_consumer_consent", summary="This operation verifies if a client has provided consumer consent based on their full name, sex, and the date their information was received. The client's first, middle, and last names, as well as their sex and the date their information was received, are used to determine the status of their consumer consent.")
async def get_consumer_consent(first: str = Query(..., description="First name of the client"), middle: str = Query(..., description="Middle name of the client"), last: str = Query(..., description="Last name of the client"), sex: str = Query(..., description="Sex of the client"), date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CASE WHEN T2.`Consumer consent provided?` IN (NULL, 'N/A', '') THEN 'No' ELSE 'Yes' END FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.middle = ? AND T1.last = ? AND T1.sex = ? AND T2.`Date received` = ?", (first, middle, last, sex, date_received))
    result = cursor.fetchall()
    if not result:
        return {"consumer_consent": []}
    return {"consumer_consent": [row[0] for row in result]}

# Endpoint to calculate the number of days between date sent to company and date received for a client based on their name, sex, and date received
@app.get("/v1/retail_complains/days_between_dates", operation_id="get_days_between_dates", summary="Retrieves the number of days between the date a client's complaint was sent to the company and the date it was received. The calculation is based on the client's first name, middle name, last name, sex, and the date received. The date received should be provided in 'YYYY-MM-DD' format.")
async def get_days_between_dates(date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format"), sex: str = Query(..., description="Sex of the client"), first: str = Query(..., description="First name of the client"), middle: str = Query(..., description="Middle name of the client"), last: str = Query(..., description="Last name of the client")):
    cursor.execute("SELECT 365 * (strftime('%Y', T2.`Date sent to company`) - strftime('%Y', T2.`Date received`)) + 30 * (strftime('%M', T2.`Date sent to company`) - strftime('%M', T2.`Date received`)) + (strftime('%d', T2.`Date sent to company`) - strftime('%d', T2.`Date received`)) AS days FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Date received` = ? AND T1.sex = ? AND T1.first = ? AND T1.middle = ? AND T1.last = ?", (date_received, sex, first, middle, last))
    result = cursor.fetchall()
    if not result:
        return {"days": []}
    return {"days": [row[0] for row in result]}

# Endpoint to get district ID and city for reviews on a specific date
@app.get("/v1/retail_complains/district_info_by_date", operation_id="get_district_info_by_date", summary="Retrieves the district ID and corresponding city for all retail reviews submitted on a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_district_info_by_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.district_id, T2.city FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.Date = ?", (date,))
    result = cursor.fetchall()
    if not result:
        return {"district_info": []}
    return {"district_info": [{"district_id": row[0], "city": row[1]} for row in result]}

# Endpoint to get reviews for a specific city and date
@app.get("/v1/retail_complains/reviews_by_city_date", operation_id="get_reviews_by_city_date", summary="Retrieves the reviews for a specific city and date. The operation uses the provided city name and date in 'YYYY-MM-DD' format to filter the reviews from the database. The result is a collection of reviews associated with the specified city and date.")
async def get_reviews_by_city_date(city: str = Query(..., description="City name"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.Reviews FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.city = ? AND T1.Date = ?", (city, date))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [row[0] for row in result]}

# Endpoint to get products for a specific city and date
@app.get("/v1/retail_complains/products_by_city_date", operation_id="get_products_by_city_date", summary="Retrieves a list of products that have been reviewed in a specific city on a given date. The operation filters the reviews based on the provided city and date, and returns the corresponding product names.")
async def get_products_by_city_date(city: str = Query(..., description="City name"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.Product FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.city = ? AND T1.Date = ?", (city, date))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get the count of stars for a specific product, city, and date
@app.get("/v1/retail_complains/star_count_by_product_city_date", operation_id="get_star_count_by_product_city_date", summary="Retrieves the total number of star ratings for a specific product in a given city on a particular date. The operation requires the product name, city name, and date in 'YYYY-MM-DD' format as input parameters.")
async def get_star_count_by_product_city_date(product: str = Query(..., description="Product name"), city: str = Query(..., description="City name"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.Stars) FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.Product = ? AND T2.city = ? AND T1.Date = ?", (product, city, date))
    result = cursor.fetchone()
    if not result:
        return {"star_count": []}
    return {"star_count": result[0]}

# Endpoint to get month and day for a specific complaint ID
@app.get("/v1/retail_complains/month_day_by_complaint_id", operation_id="get_month_day_by_complaint_id", summary="Retrieves the month and day associated with a specific complaint ID. This operation fetches the relevant data from the client and events tables, based on the provided complaint ID.")
async def get_month_day_by_complaint_id(complaint_id: str = Query(..., description="Complaint ID")):
    cursor.execute("SELECT T1.month, T1.day FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Complaint ID` = ?", (complaint_id,))
    result = cursor.fetchall()
    if not result:
        return {"month_day": []}
    return {"month_day": [{"month": row[0], "day": row[1]} for row in result]}

# Endpoint to get phone number for a specific complaint ID
@app.get("/v1/retail_complains/phone_by_complaint_id", operation_id="get_phone_by_complaint_id", summary="Retrieves the phone number associated with a specific complaint ID. This operation fetches the client's phone number linked to the provided complaint ID by joining the 'client' and 'events' tables on the 'client_id' field. The complaint ID is used to filter the results.")
async def get_phone_by_complaint_id(complaint_id: str = Query(..., description="Complaint ID")):
    cursor.execute("SELECT T1.phone FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Complaint ID` = ?", (complaint_id,))
    result = cursor.fetchall()
    if not result:
        return {"phone": []}
    return {"phone": [row[0] for row in result]}

# Endpoint to get the percentage of female clients based on a specific date
@app.get("/v1/retail_complains/percentage_female_clients", operation_id="get_percentage_female_clients", summary="Retrieves the percentage of female clients who had an event on a specific date. The date should be provided in 'YYYY-MM-DD' format.")
async def get_percentage_female_clients(date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.sex = 'Female' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.sex) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Date received` = ?", (date_received,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of clients who provided consent based on specific client details
@app.get("/v1/retail_complains/percentage_consent_provided", operation_id="get_percentage_consent_provided", summary="Retrieves the percentage of clients who have provided consent, based on the specified client details such as their sex and full name. This operation calculates the ratio of clients who have provided consent to the total number of clients matching the provided details.")
async def get_percentage_consent_provided(sex: str = Query(..., description="Sex of the client"), first: str = Query(..., description="First name of the client"), middle: str = Query(..., description="Middle name of the client"), last: str = Query(..., description="Last name of the client")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.`Consumer consent provided?` = 'Consent provided' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.`Consumer consent provided?`) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.sex = ? AND T1.first = ? AND T1.middle = ? AND T1.last = ?", (sex, first, middle, last))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of complaints with the highest priority for a specific month
@app.get("/v1/retail_complains/count_highest_priority_complaints", operation_id="get_count_highest_priority_complaints", summary="Retrieves the total number of complaints with the highest priority received during a specific month. The month is provided in the 'YYYY-MM%' format.")
async def get_count_highest_priority_complaints(date_received: str = Query(..., description="Date received in 'YYYY-MM%' format")):
    cursor.execute("SELECT COUNT(`Complaint ID`) FROM callcenterlogs WHERE `Date received` LIKE ? AND priority = ( SELECT MAX(priority) FROM callcenterlogs )", (date_received,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client details for clients older than a specific age
@app.get("/v1/retail_complains/client_details_by_age", operation_id="get_client_details_by_age", summary="Retrieves personal details of clients who are older than the specified minimum age, sorted in descending order by age. The response includes the client's first, middle, and last names, birth year, month, and day, and email address.")
async def get_client_details_by_age(min_age: int = Query(..., description="Minimum age of the client")):
    cursor.execute("SELECT first, middle, last, year, month , day, email FROM client WHERE age > ? ORDER BY age DESC", (min_age,))
    result = cursor.fetchall()
    if not result:
        return {"clients": []}
    return {"clients": result}

# Endpoint to get the product with the maximum number of 5-star reviews
@app.get("/v1/retail_complains/max_five_star_reviews", operation_id="get_max_five_star_reviews", summary="Retrieves the product that has received the highest number of 5-star reviews. The input parameter specifies the star rating to consider for the count.")
async def get_max_five_star_reviews(stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT T.Product, MAX(T.num) FROM ( SELECT Product, COUNT(Stars) AS num FROM reviews WHERE Stars = ? GROUP BY Product ) T", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": result[0], "max_reviews": result[1]}

# Endpoint to get states in a specific region
@app.get("/v1/retail_complains/states_by_region", operation_id="get_states_by_region", summary="Retrieves a list of states that belong to the specified region. The region is identified by its name.")
async def get_states_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT state FROM state WHERE Region = ?", (region,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": result}

# Endpoint to get client emails based on call center outcome
@app.get("/v1/retail_complains/client_emails_by_outcome", operation_id="get_client_emails_by_outcome", summary="Retrieves the email addresses of clients who have had a specific outcome in their interactions with the call center. The outcome is specified as an input parameter.")
async def get_client_emails_by_outcome(outcome: str = Query(..., description="Call center outcome")):
    cursor.execute("SELECT T1.email FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T2.outcome = ?", (outcome,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": result}

# Endpoint to get the average age of clients in a specific region
@app.get("/v1/retail_complains/average_age_by_region", operation_id="get_average_age_by_region", summary="Retrieves the average age of clients residing in a specified region. This operation calculates the average age by summing up the ages of all clients in the given region and dividing by the total number of clients in that region. The region is identified by its name.")
async def get_average_age_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT CAST(SUM(T1.age) AS REAL) / COUNT(T3.Region) AS average FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN state AS T3 ON T2.state_abbrev = T3.StateCode WHERE T3.Region = ?", (region,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get client details based on submission method
@app.get("/v1/retail_complains/client_details_by_submission", operation_id="get_client_details_by_submission", summary="Retrieves the first, middle, and last names, as well as the phone number of a client who submitted a complaint using the specified method. The submission method is provided as an input parameter.")
async def get_client_details_by_submission(submitted_via: str = Query(..., description="Submission method")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last, T1.phone FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Submitted via` = ?", (submitted_via,))
    result = cursor.fetchall()
    if not result:
        return {"clients": []}
    return {"clients": result}

# Endpoint to get divisions with above-average reviews for a specific product
@app.get("/v1/retail_complains/divisions_above_average_reviews", operation_id="get_divisions_above_average_reviews", summary="Retrieves the divisions where the average review rating for a specified product surpasses the overall average review rating. The product name is required as an input parameter.")
async def get_divisions_above_average_reviews(product: str = Query(..., description="Product name")):
    cursor.execute("SELECT T2.division FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.Product = ? AND T1.Stars > ( SELECT AVG(Stars) FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id )", (product,))
    result = cursor.fetchall()
    if not result:
        return {"divisions": []}
    return {"divisions": result}

# Endpoint to get the count of clients in a specific age range and division
@app.get("/v1/retail_complains/client_count_age_division", operation_id="get_client_count_age_division", summary="Retrieves the total number of clients within a specified age range and division. The age range is defined by a minimum and maximum age, while the division refers to a specific geographical area. This operation provides a count of clients who meet these criteria, offering insights into the client demographics.")
async def get_client_count_age_division(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), division: str = Query(..., description="Division")):
    cursor.execute("SELECT COUNT(T1.age) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.age BETWEEN ? AND ? AND T2.division = ?", (min_age, max_age, division))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of clients based on sex and product
@app.get("/v1/retail_complains/client_count_sex_product", operation_id="get_client_count_sex_product", summary="Retrieves the total number of clients categorized by sex and product. The operation requires the sex of the client and the product as input parameters to filter the count.")
async def get_client_count_sex_product(sex: str = Query(..., description="Sex of the client"), product: str = Query(..., description="Product")):
    cursor.execute("SELECT COUNT(T1.sex) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.sex = ? AND T2.Product = ?", (sex, product))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client names based on year range, sex, and submission method
@app.get("/v1/retail_complains/client_names_year_sex_submission", operation_id="get_client_names_year_sex_submission", summary="Retrieves the first, middle, and last names of clients who meet the specified criteria. The criteria include a range of years, the sex of the client, and the method of submission. The response is based on a join operation between the 'client' and 'events' tables, ensuring that only clients who meet all the specified conditions are included in the results.")
async def get_client_names_year_sex_submission(min_year: int = Query(..., description="Minimum year"), max_year: int = Query(..., description="Maximum year"), sex: str = Query(..., description="Sex of the client"), submission_method: str = Query(..., description="Submission method")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.year BETWEEN ? AND ? AND T1.sex = ? AND T2.`Submitted via` = ?", (min_year, max_year, sex, submission_method))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the most common submission method for complaints in a specific state
@app.get("/v1/retail_complains/most_common_submission_method_state", operation_id="get_most_common_submission_method_state", summary="Retrieves the most frequently used submission method for complaints in a specified state. The operation considers all complaints logged in the call center and associated with clients in the given state. It then identifies the submission method with the highest occurrence across these complaints.")
async def get_most_common_submission_method_state(state: str = Query(..., description="State")):
    cursor.execute("SELECT T3.`Submitted via` FROM callcenterlogs AS T1 INNER JOIN client AS T2 ON T1.`rand client` = T2.client_id INNER JOIN events AS T3 ON T1.`Complaint ID` = T3.`Complaint ID` WHERE T2.state = ? GROUP BY T1.`Complaint ID` ORDER BY COUNT(T1.`Complaint ID`) DESC LIMIT 1", (state,))
    result = cursor.fetchone()
    if not result:
        return {"submission_method": []}
    return {"submission_method": result[0]}

# Endpoint to get the average company response rate by year for a specific city
@app.get("/v1/retail_complains/average_response_rate_city", operation_id="get_average_response_rate_city", summary="Retrieves the average rate of a specific company response type in a given city, calculated yearly. The response rate is determined by comparing the number of instances of the specified response type to the total number of complaints received in the city each year.")
async def get_average_response_rate_city(response_type: str = Query(..., description="Company response type"), city: str = Query(..., description="City")):
    cursor.execute("SELECT STRFTIME('%Y', T3.`Date received`) , CAST(SUM(CASE WHEN T3.`Company response to consumer` = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(T3.`Complaint ID`) AS average FROM callcenterlogs AS T1 INNER JOIN client AS T2 ON T1.`rand client` = T2.client_id INNER JOIN events AS T3 ON T1.`Complaint ID` = T3.`Complaint ID` WHERE T2.city = ? GROUP BY strftime('%Y', T3.`Date received`)", (response_type, city))
    result = cursor.fetchall()
    if not result:
        return {"average_response_rate": []}
    return {"average_response_rate": result}

# Endpoint to get the percentage of disputed consumers in a specific city
@app.get("/v1/retail_complains/percentage_disputed_consumers_city", operation_id="get_percentage_disputed_consumers_city", summary="Retrieves the percentage of consumers who have disputed their transactions in a specific city. The operation calculates this percentage by comparing the number of disputed consumers in the given city to the total number of consumers in that city.")
async def get_percentage_disputed_consumers_city(disputed: str = Query(..., description="Consumer disputed status"), city: str = Query(..., description="City")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.`Consumer disputed?` = ? AND T1.city = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID", (disputed, city))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of clients with a specific tag in a specific city
@app.get("/v1/retail_complains/client_count_tag_city", operation_id="get_client_count_tag_city", summary="Retrieves the total number of clients associated with a particular tag in a given city. The operation requires the tag and city as input parameters to filter the client count.")
async def get_client_count_tag_city(tag: str = Query(..., description="Tag"), city: str = Query(..., description="City")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.Tags = ? AND T1.city = ?", (tag, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of high priority complaints in a specific state
@app.get("/v1/retail_complains/percentage_high_priority_complaints_state", operation_id="get_percentage_high_priority_complaints_state", summary="Retrieves the percentage of high priority complaints in a specific state. The operation calculates this percentage by comparing the count of high priority complaints to the total number of complaints in the given state. The priority level and state are provided as input parameters.")
async def get_percentage_high_priority_complaints_state(priority: int = Query(..., description="Priority level"), state: str = Query(..., description="State")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.priority = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.priority) FROM callcenterlogs AS T1 INNER JOIN client AS T2 ON T1.`rand client` = T2.client_id INNER JOIN district AS T3 ON T2.district_id = T3.district_id INNER JOIN state AS T4 ON T3.state_abbrev = T4.StateCode WHERE T4.State = ?", (priority, state))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the age difference between two age groups in a specific region
@app.get("/v1/retail_complains/age_difference_region", operation_id="get_age_difference_region", summary="Retrieves the average age difference between two specified age groups in a given region. The first age group is defined by the minimum and maximum ages provided, while the second group includes all ages above the specified minimum. The calculation is based on client data, filtered by the selected region.")
async def get_age_difference_region(min_age_1: int = Query(..., description="Minimum age for the first group"), max_age_1: int = Query(..., description="Maximum age for the first group"), min_age_2: int = Query(..., description="Minimum age for the second group"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT (CAST(SUM(CASE WHEN T1.age BETWEEN ? AND ? THEN T1.age ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.age BETWEEN ? AND ? THEN 1 ELSE 0 END)) - (CAST(SUM(CASE WHEN T1.age > ? THEN T1.age ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.age > ? THEN 1 ELSE 0 END)) AS difference FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN state AS T3 ON T2.state_abbrev = T3.StateCode WHERE T3.Region = ?", (min_age_1, max_age_1, min_age_1, max_age_1, min_age_2, min_age_2, region))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the latest complaint IDs
@app.get("/v1/retail_complains/latest_complaint_ids", operation_id="get_latest_complaint_ids", summary="Retrieves the most recent complaint IDs from the call center logs. The number of IDs returned can be specified using the provided limit parameter.")
async def get_latest_complaint_ids(limit: int = Query(..., description="Number of latest complaint IDs to retrieve")):
    cursor.execute("SELECT `Complaint ID` FROM callcenterlogs ORDER BY ser_time DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"complaint_ids": []}
    return {"complaint_ids": result}

# Endpoint to get the count of emails not matching a specific domain
@app.get("/v1/retail_complains/count_emails_not_matching_domain", operation_id="get_count_emails_not_matching_domain", summary="Retrieves the total count of client emails that do not belong to a specified domain. This operation is useful for identifying email addresses that do not conform to a particular domain standard. The input parameter allows you to specify the domain to exclude from the count.")
async def get_count_emails_not_matching_domain(domain: str = Query(..., description="Domain to exclude from email addresses")):
    cursor.execute("SELECT COUNT(email) FROM client WHERE email NOT LIKE ?", (f'%@{domain}',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client IDs based on consumer consent status
@app.get("/v1/retail_complains/client_ids_by_consumer_consent", operation_id="get_client_ids_by_consumer_consent", summary="Get client IDs where consumer consent is either 'N/A', NULL, or an empty string")
async def get_client_ids_by_consumer_consent(consent_status: str = Query(..., description="Consumer consent status, e.g., 'N/A'"), empty_status: str = Query(..., description="Empty status, e.g., an empty string")):
    cursor.execute("SELECT Client_ID FROM events WHERE `Consumer consent provided?` = ? OR `Consumer consent provided?` IS NULL OR `Consumer consent provided?` = ?", (consent_status, empty_status))
    result = cursor.fetchall()
    if not result:
        return {"client_ids": []}
    return {"client_ids": [row[0] for row in result]}

# Endpoint to get complaint IDs with the maximum difference in dates
@app.get("/v1/retail_complains/complaint_ids_max_date_difference", operation_id="get_complaint_ids_max_date_difference", summary="Retrieves the IDs of complaints that have the maximum time difference between the date they were sent to the company and the date they were received, for a specific date sent to the company.")
async def get_complaint_ids_max_date_difference(date_sent: str = Query(..., description="Date sent to company in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT `Complaint ID` FROM events WHERE strftime('%J', `Date sent to company`) - strftime('%J', `Date received`) = ( SELECT MAX(strftime('%J', `Date sent to company`) - strftime('%J', `Date received`)) FROM events WHERE `Date sent to company` = ? ) AND `Date sent to company` = ?", (date_sent, date_sent))
    result = cursor.fetchall()
    if not result:
        return {"complaint_ids": []}
    return {"complaint_ids": [row[0] for row in result]}

# Endpoint to get distinct complaint IDs based on priority
@app.get("/v1/retail_complains/distinct_complaint_ids_by_priority", operation_id="get_distinct_complaint_ids_by_priority", summary="Retrieves a list of unique complaint identifiers that have a specified priority level, sorted by the date they were received in descending order.")
async def get_distinct_complaint_ids_by_priority(priority: int = Query(..., description="Priority level")):
    cursor.execute("SELECT DISTINCT `Complaint ID` FROM callcenterlogs WHERE priority = ? ORDER BY `Date received` DESC", (priority,))
    result = cursor.fetchall()
    if not result:
        return {"complaint_ids": []}
    return {"complaint_ids": [row[0] for row in result]}

# Endpoint to get the count of outcomes not matching a specific value
@app.get("/v1/retail_complains/count_outcomes_not_matching", operation_id="get_count_outcomes_not_matching", summary="Retrieves the total number of call center logs with outcomes that do not match the provided value. This operation allows you to filter out a specific outcome and count the remaining instances in the call center logs.")
async def get_count_outcomes_not_matching(outcome: str = Query(..., description="Outcome value to exclude")):
    cursor.execute("SELECT COUNT(outcome) FROM callcenterlogs WHERE outcome != ?", (outcome,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of complaint IDs based on product and server
@app.get("/v1/retail_complains/count_complaint_ids_by_product_server", operation_id="get_count_complaint_ids_by_product_server", summary="Retrieves the total number of unique complaint IDs associated with a specific product and server. This operation requires the product name and server name as input parameters to filter the results accordingly.")
async def get_count_complaint_ids_by_product_server(product: str = Query(..., description="Product name"), server: str = Query(..., description="Server name")):
    cursor.execute("SELECT COUNT(T1.`Complaint ID`) FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T2.Product = ? AND T1.server = ?", (product, server))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the region with the most reviews for a specific star rating
@app.get("/v1/retail_complains/top_region_by_star_rating", operation_id="get_top_region_by_star_rating", summary="Retrieves the region with the highest number of reviews for a given star rating. The operation filters reviews by the specified star rating, groups them by region, and orders the results by the count of reviews in descending order. The region with the most reviews is then returned.")
async def get_top_region_by_star_rating(stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T3.Region FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN state AS T3 ON T2.state_abbrev = T3.StateCode WHERE T1.Stars = ? GROUP BY T3.Region ORDER BY COUNT(T3.Region) DESC LIMIT 1", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"region": []}
    return {"region": result[0]}

# Endpoint to get the year with the most clients for a specific sub-product
@app.get("/v1/retail_complains/top_year_by_sub_product", operation_id="get_top_year_by_sub_product", summary="Retrieves the year with the highest number of clients for a specified sub-product. The sub-product name is used to filter the data and determine the year with the most clients.")
async def get_top_year_by_sub_product(sub_product: str = Query(..., description="Sub-product name")):
    cursor.execute("SELECT T1.year FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Sub-product` = ? GROUP BY T1.year ORDER BY COUNT(T1.year) DESC LIMIT 1", (sub_product,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the count of issues based on issue type and division
@app.get("/v1/retail_complains/count_issues_by_issue_division", operation_id="get_count_issues_by_issue_division", summary="Retrieves the total count of a specific issue type within a given division. The operation requires the issue type and division name as input parameters to filter the data and provide an accurate count.")
async def get_count_issues_by_issue_division(issue: str = Query(..., description="Issue type"), division: str = Query(..., description="Division name")):
    cursor.execute("SELECT COUNT(T1.Issue) FROM events AS T1 INNER JOIN client AS T2 ON T1.Client_ID = T2.client_id INNER JOIN district AS T3 ON T2.district_id = T3.district_id WHERE T1.Issue = ? AND T3.division = ?", (issue, division))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of clients based on state and sex
@app.get("/v1/retail_complains/count_clients_by_state_sex", operation_id="get_count_clients_by_state_sex", summary="Retrieves the total number of clients in a specific state, filtered by sex. The operation uses the state name and sex as input parameters to calculate the count.")
async def get_count_clients_by_state_sex(state: str = Query(..., description="State name"), sex: str = Query(..., description="Sex, e.g., 'Male'")):
    cursor.execute("SELECT COUNT(T3.sex) FROM state AS T1 INNER JOIN district AS T2 ON T1.StateCode = T2.state_abbrev INNER JOIN client AS T3 ON T2.district_id = T3.district_id WHERE T1.state = ? AND T3.sex = ?", (state, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the last names of clients based on call center log type and server
@app.get("/v1/retail_complains/client_last_names_by_log_type_server", operation_id="get_client_last_names", summary="Retrieves the last names of clients associated with a specific call center log type and server. The operation filters clients based on the provided log type and server, and returns their last names.")
async def get_client_last_names(log_type: str = Query(..., description="Type of call center log"), server: str = Query(..., description="Server name")):
    cursor.execute("SELECT t1.last FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T2.type = ? AND T2.server = ?", (log_type, server))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get the count of clients under a certain age who reviewed a specific product with a certain star rating
@app.get("/v1/retail_complains/count_clients_by_product_stars_age", operation_id="get_count_clients_by_product_stars_age", summary="Retrieves the number of clients below a specified age who have reviewed a particular product with a given star rating. This operation provides insights into the demographic distribution of product reviews based on age and star rating.")
async def get_count_clients_by_product_stars_age(product: str = Query(..., description="Product name"), stars: int = Query(..., description="Star rating"), max_age: int = Query(..., description="Maximum age of the client")):
    cursor.execute("SELECT COUNT(T2.age) FROM reviews AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T1.Product = ? AND T1.Stars = ? AND T2.age < ?", (product, stars, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of clients of a specific sex, priority, and year
@app.get("/v1/retail_complains/count_clients_by_sex_priority_year", operation_id="get_count_clients_by_sex_priority_year", summary="Retrieves the total number of clients of a specified gender, priority level, and year. This operation provides a count of clients based on the provided gender, priority level, and year, offering insights into client demographics and call center activity.")
async def get_count_clients_by_sex_priority_year(sex: str = Query(..., description="Sex of the client"), priority: int = Query(..., description="Priority level"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT COUNT(T1.sex) FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T1.sex = ? AND T2.priority = ? AND T1.year = ?", (sex, priority, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client names based on event tags and consumer consent status
@app.get("/v1/retail_complains/client_names_by_event_tags_consent", operation_id="get_client_names_by_event_tags_consent", summary="Get client names based on event tags and consumer consent status")
async def get_client_names_by_event_tags_consent(tags: str = Query(..., description="Event tags"), consent_not_equal: str = Query(..., description="Consumer consent status not equal to"), consent_not_null: str = Query(..., description="Consumer consent status not null"), consent_not_empty: str = Query(..., description="Consumer consent status not empty")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.Tags = ? AND T2.`Consumer consent provided?` != ? AND T2.`Consumer consent provided?` IS NOT NULL AND T2.`Consumer consent provided?` != ?", (tags, consent_not_equal, consent_not_null, consent_not_empty))
    result = cursor.fetchall()
    if not result:
        return {"client_names": []}
    return {"client_names": [{"first": row[0], "middle": row[1], "last": row[2]} for row in result]}

# Endpoint to get the state with the highest number of call center logs of a specific priority
@app.get("/v1/retail_complains/top_state_by_priority", operation_id="get_top_state_by_priority", summary="Retrieves the state with the most call center logs of a given priority. The operation considers the client and district associated with each log, and returns the state with the highest count of logs at the specified priority level.")
async def get_top_state_by_priority(priority: int = Query(..., description="Priority level")):
    cursor.execute("SELECT T2.state FROM callcenterlogs AS T1 INNER JOIN client AS T2 ON T1.`rand client` = T2.client_id INNER JOIN district AS T3 ON T2.district_id = T3.district_id INNER JOIN state AS T4 ON T3.state_abbrev = T4.StateCode WHERE T1.priority = ? GROUP BY T2.state ORDER BY COUNT(T2.state) DESC LIMIT 1", (priority,))
    result = cursor.fetchone()
    if not result:
        return {"state": []}
    return {"state": result[0]}

# Endpoint to get the count of complaints based on client sex, service start time range, and timely response
@app.get("/v1/retail_complains/count_complaints_by_sex_time_response", operation_id="get_count_complaints_by_sex_time_response", summary="Retrieves the total number of complaints lodged by clients of a specific gender, within a defined service start time range, and based on whether the response was timely. The response status indicates whether the service was provided within the expected timeframe.")
async def get_count_complaints_by_sex_time_response(sex: str = Query(..., description="Sex of the client"), start_time_start: str = Query(..., description="Start time start in 'HH:MM:SS' format"), start_time_end: str = Query(..., description="Start time end in 'HH:MM:SS' format"), timely_response: str = Query(..., description="Timely response status")):
    cursor.execute("SELECT COUNT(T1.`Complaint ID`) FROM callcenterlogs AS T1 INNER JOIN client AS T2 ON T1.`rand client` = T2.client_id INNER JOIN events AS T3 ON T1.`Complaint ID` = T3.`Complaint ID` WHERE T2.sex = ? AND T1.ser_start BETWEEN ? AND ? AND T3.`Timely response?` = ?", (sex, start_time_start, start_time_end, timely_response))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of complaints based on service time, server, submission method, and company response
@app.get("/v1/retail_complains/count_complaints_by_time_server_submission_response", operation_id="get_count_complaints_by_time_server_submission_response", summary="Retrieves the total number of complaints that meet specific criteria: service time before a given time, specific server, submission method, and company response. This operation is useful for tracking and analyzing complaint trends based on these factors.")
async def get_count_complaints_by_time_server_submission_response(service_time: str = Query(..., description="Service time in 'HH:MM:SS' format"), server: str = Query(..., description="Server name"), submitted_via: str = Query(..., description="Submission method"), company_response: str = Query(..., description="Company response to consumer")):
    cursor.execute("SELECT COUNT(T1.`Complaint ID`) FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T1.ser_time < ? AND T1.server = ? AND T2.`Submitted via` = ? AND T2.`Company response to consumer` = ?", (service_time, server, submitted_via, company_response))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of clients based on last name and state
@app.get("/v1/retail_complains/count_clients_by_last_name_state", operation_id="get_count_clients_by_last_name_state", summary="Retrieves the total number of clients with a specific last name in a given state. The operation uses the provided last name and state abbreviation to filter the client list and calculate the count.")
async def get_count_clients_by_last_name_state(last_name: str = Query(..., description="Last name of the client"), state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT COUNT(T2.client_id) FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id INNER JOIN state AS T3 ON T1.state_abbrev = T3.StateCode WHERE T2.last = ? AND T2.state = ?", (last_name, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of reviews based on client age range and review content
@app.get("/v1/retail_complains/count_reviews_by_age_content", operation_id="get_count_reviews_by_age_content", summary="Retrieves the total number of reviews that meet the specified age range and review content criteria. The age range is defined by the minimum and maximum age of the clients, and the review content must contain a specific text. This operation is useful for analyzing review trends based on client age and review content.")
async def get_count_reviews_by_age_content(min_age: int = Query(..., description="Minimum age of the client"), max_age: int = Query(..., description="Maximum age of the client"), review_content: str = Query(..., description="Review content containing specific text")):
    cursor.execute("SELECT COUNT(T1.Reviews) FROM reviews AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T2.age BETWEEN ? AND ? AND T1.Reviews LIKE ?", (min_age, max_age, f'%{review_content}%'))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client addresses based on timely response and consumer dispute status
@app.get("/v1/retail_complains/client_addresses_by_response_dispute", operation_id="get_client_addresses_by_response_dispute", summary="Retrieves the addresses of clients who have a specific timely response status and consumer dispute status. The response includes the first and second lines of the client's address.")
async def get_client_addresses_by_response_dispute(timely_response: str = Query(..., description="Timely response status"), consumer_disputed: str = Query(..., description="Consumer dispute status")):
    cursor.execute("SELECT T1.address_1, T1.address_2 FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Timely response?` = ? AND T2.`Consumer disputed?` = ?", (timely_response, consumer_disputed))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [{"address_1": row[0], "address_2": row[1]} for row in result]}

# Endpoint to get the count of submissions via different methods for clients based on sex and year
@app.get("/v1/retail_complains/count_submissions_by_sex_year", operation_id="get_count_submissions_by_sex_year", summary="Retrieves the total number of submissions, excluding a specified method, for clients of a given sex and year. This operation provides a breakdown of submission counts based on the client's sex and year, offering insights into submission patterns and trends.")
async def get_count_submissions_by_sex_year(sex: str = Query(..., description="Sex of the client"), year: int = Query(..., description="Year of the client"), submitted_via: str = Query(..., description="Submission method to exclude")):
    cursor.execute("SELECT COUNT(T2.`Submitted via`) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.sex = ? AND T1.year = ? AND T2.`Submitted via` != ?", (sex, year, submitted_via))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get consumer complaint narratives for a specific client
@app.get("/v1/retail_complains/consumer_complaint_narrative", operation_id="get_consumer_complaint_narrative", summary="Retrieves the consumer complaint narratives associated with a specific client, identified by their first and last name. This operation returns a list of narratives related to the client's complaints, providing detailed information about the nature and context of each complaint.")
async def get_consumer_complaint_narrative(first: str = Query(..., description="First name of the client"), last: str = Query(..., description="Last name of the client")):
    cursor.execute("SELECT T2.`Consumer complaint narrative` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.last = ?", (first, last))
    result = cursor.fetchall()
    if not result:
        return {"narratives": []}
    return {"narratives": [row[0] for row in result]}

# Endpoint to get the count of emails for clients based on date received and email domain
@app.get("/v1/retail_complains/count_emails_by_date_domain", operation_id="get_count_emails_by_date_domain", summary="Retrieves the total number of emails received from a specific domain within a given date range. The date range is defined by two parameters, both in 'YYYY-MM%' format, and the domain is specified by the 'email_domain' parameter.")
async def get_count_emails_by_date_domain(date_received_1: str = Query(..., description="Date received in 'YYYY-MM%' format"), date_received_2: str = Query(..., description="Date received in 'YYYY-MM%' format"), email_domain: str = Query(..., description="Email domain to filter")):
    cursor.execute("SELECT COUNT(T1.email) FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE (T2.`Date received` LIKE ? OR T2.`Date received` LIKE ?) AND T1.email LIKE ?", (date_received_1, date_received_2, email_domain))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average star rating for a specific state
@app.get("/v1/retail_complains/average_star_rating_by_state", operation_id="get_average_star_rating_by_state", summary="Retrieves the average star rating for a specified state. This operation calculates the average star rating by aggregating the total star ratings and dividing by the count of ratings for the given state. The state is identified by its name.")
async def get_average_star_rating_by_state(state: str = Query(..., description="State name")):
    cursor.execute("SELECT CAST(SUM(T3.Stars) AS REAL) / COUNT(T3.Stars) AS average FROM state AS T1 INNER JOIN district AS T2 ON T1.StateCode = T2.state_abbrev INNER JOIN reviews AS T3 ON T2.district_id = T3.district_id WHERE T1.State = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of clients above a certain age who submitted via a specific method
@app.get("/v1/retail_complains/percentage_clients_above_age", operation_id="get_percentage_clients_above_age", summary="Retrieves the percentage of clients who are older than a specified age and submitted their complaints via a specific method. The calculation is based on the total number of clients who submitted complaints via the same method.")
async def get_percentage_clients_above_age(age: int = Query(..., description="Age threshold"), submitted_via: str = Query(..., description="Submission method")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.age > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.`Submitted via`) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Submitted via` = ?", (age, submitted_via))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average age of clients in a specific city
@app.get("/v1/retail_complains/average_age_by_city", operation_id="get_average_age_by_city", summary="Retrieves the average age of clients residing in a specified city. This operation calculates the average age by summing up the ages of all clients in the given city and dividing by the total count of clients in that city.")
async def get_average_age_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT CAST(SUM(T1.age) AS REAL) / COUNT(T1.age) AS average FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the count of reviews with a specific star rating in a specific city
@app.get("/v1/retail_complains/count_reviews_by_city_stars", operation_id="get_count_reviews_by_city_stars", summary="Retrieves the total number of reviews with a specified star rating in a given city. The operation requires the city name and the desired star rating as input parameters.")
async def get_count_reviews_by_city_stars(city: str = Query(..., description="City name"), stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT COUNT(T1.Stars) FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.city = ? AND T1.Stars = ?", (city, stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the state abbreviation with the highest number of reviews with a specific star rating
@app.get("/v1/retail_complains/top_state_by_stars", operation_id="get_top_state_by_stars", summary="Retrieves the state abbreviation with the highest count of reviews for a given star rating. This operation aggregates reviews based on their star rating and corresponding state, then returns the state with the most reviews for the specified star rating.")
async def get_top_state_by_stars(stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T2.state_abbrev FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.Stars = ? GROUP BY T2.state_abbrev ORDER BY COUNT(T2.state_abbrev) DESC LIMIT 1", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"state_abbrev": []}
    return {"state_abbrev": result[0]}

# Endpoint to get distinct submission methods for a specific client based on first and last name
@app.get("/v1/retail_complains/distinct_submission_methods", operation_id="get_distinct_submission_methods", summary="Retrieves a unique set of submission methods associated with a specific client, identified by their first and last name. This operation returns the various ways in which the client has submitted their data, providing insights into their preferred communication channels.")
async def get_distinct_submission_methods(first: str = Query(..., description="First name of the client"), last: str = Query(..., description="Last name of the client")):
    cursor.execute("SELECT DISTINCT T2.`Submitted via` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.last = ?", (first, last))
    result = cursor.fetchall()
    if not result:
        return {"submission_methods": []}
    return {"submission_methods": [row[0] for row in result]}

# Endpoint to get distinct products for clients after a specific year
@app.get("/v1/retail_complains/distinct_products_after_year", operation_id="get_distinct_products", summary="Retrieves a list of unique products associated with clients who have records after the specified year. This operation filters clients based on the provided year and returns the distinct products linked to them.")
async def get_distinct_products(year: int = Query(..., description="Year after which to filter clients")):
    cursor.execute("SELECT DISTINCT T2.Product FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.year > ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get service times for a specific client with a specific product
@app.get("/v1/retail_complains/service_times_for_client_product", operation_id="get_service_times", summary="Retrieves the service times for a specific client with a specific product. This operation requires the first name, last name, sex of the client, and the product associated with the complaint. The service times are obtained by joining the events, client, and callcenterlogs tables using the client ID and complaint ID.")
async def get_service_times(first: str = Query(..., description="First name of the client"), last: str = Query(..., description="Last name of the client"), sex: str = Query(..., description="Sex of the client"), product: str = Query(..., description="Product associated with the complaint")):
    cursor.execute("SELECT T3.ser_time FROM events AS T1 INNER JOIN client AS T2 ON T1.Client_ID = T2.client_id INNER JOIN callcenterlogs AS T3 ON T1.`Complaint ID` = T3.`Complaint ID` WHERE T2.first = ? AND T2.last = ? AND T2.sex = ? AND T1.Product = ?", (first, last, sex, product))
    result = cursor.fetchall()
    if not result:
        return {"service_times": []}
    return {"service_times": [row[0] for row in result]}

# Endpoint to get the issue of the latest service time
@app.get("/v1/retail_complains/latest_service_time_issue", operation_id="get_latest_service_time_issue", summary="Retrieves the issue associated with the most recent service time. This operation identifies the latest service time from the call center logs and returns the corresponding issue from the events table. The result provides insight into the most recent service-related concern.")
async def get_latest_service_time_issue():
    cursor.execute("SELECT T2.Issue FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T1.ser_time = ( SELECT MAX(ser_time) FROM callcenterlogs )")
    result = cursor.fetchall()
    if not result:
        return {"issues": []}
    return {"issues": [row[0] for row in result]}

# Endpoint to get the count of clients in a specific city who submitted via a specific method
@app.get("/v1/retail_complains/client_count_city_submission", operation_id="get_client_count", summary="Retrieves the total number of clients from a particular city who used a specific submission method. The city and submission method are provided as input parameters.")
async def get_client_count(city: str = Query(..., description="City of the client"), submitted_via: str = Query(..., description="Method of submission")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.city = ? AND T2.`Submitted via` = ?", (city, submitted_via))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of male clients for a specific product
@app.get("/v1/retail_complains/percentage_male_clients_product", operation_id="get_percentage_male_clients", summary="Retrieves the percentage of male clients who have interacted with a specific product. The product is identified by the provided input parameter.")
async def get_percentage_male_clients(product: str = Query(..., description="Product associated with the complaint")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.sex = 'Male' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.sex) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.Product = ?", (product,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get client names with specific tags and no consumer consent
@app.get("/v1/retail_complains/client_names_tags_no_consent", operation_id="get_client_names", summary="Retrieves the first, middle, and last names of clients associated with a specific complaint tag, excluding those who have provided consumer consent. The operation returns a maximum of two records.")
async def get_client_names(tags: str = Query(..., description="Tags associated with the complaint")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.Tags = ? AND T2.`Consumer consent provided?` IN (NULL, 'N/A', '') LIMIT 2", (tags,))
    result = cursor.fetchall()
    if not result:
        return {"clients": []}
    return {"clients": [{"first": row[0], "middle": row[1], "last": row[2]} for row in result]}

# Endpoint to get the latest client date
@app.get("/v1/retail_complains/latest_client_date", operation_id="get_latest_client_date", summary="Retrieves the most recent date associated with a client. The date is determined by the latest combination of year, month, and day available in the client data.")
async def get_latest_client_date():
    cursor.execute("SELECT day, month, year FROM client ORDER BY year DESC, month DESC, day DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": {"day": result[0], "month": result[1], "year": result[2]}}

# Endpoint to get the count of timely responses with specific consumer dispute status
@app.get("/v1/retail_complains/count_timely_responses_dispute_status", operation_id="get_count_timely_responses", summary="Retrieves the count of records where the timely response status matches the provided value and the consumer disputed status also matches the provided value. This operation is useful for tracking the number of timely responses with a specific consumer dispute status.")
async def get_count_timely_responses(timely_response: str = Query(..., description="Timely response status"), consumer_disputed: str = Query(..., description="Consumer disputed status")):
    cursor.execute("SELECT COUNT(`Timely response?`) FROM events WHERE `Timely response?` = ? AND `Consumer disputed?` = ?", (timely_response, consumer_disputed))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of service times after a specific minute
@app.get("/v1/retail_complains/count_service_times_after_minute", operation_id="get_count_service_times", summary="Retrieves the total number of service times that occurred after a specified minute. The input parameter determines the minute threshold for counting service times. This operation is useful for analyzing service time distribution and identifying trends in call center activity.")
async def get_count_service_times(minute: str = Query(..., description="Minute after which to count service times, in 'MM' format")):
    cursor.execute("SELECT COUNT(ser_time) FROM callcenterlogs WHERE strftime('%M', ser_time) > ?", (minute,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common issue for a specific priority
@app.get("/v1/retail_complains/most_common_issue_priority", operation_id="get_most_common_issue", summary="Retrieves the most frequently reported issue for a given priority level. The operation filters events based on the specified priority level and identifies the issue with the highest occurrence. The result is a single issue that is most commonly reported for the provided priority.")
async def get_most_common_issue(priority: int = Query(..., description="Priority level")):
    cursor.execute("SELECT T1.Issue FROM events AS T1 INNER JOIN callcenterlogs AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T2.priority = ? GROUP BY T1.Issue ORDER BY COUNT(T1.Issue) DESC LIMIT 1", (priority,))
    result = cursor.fetchone()
    if not result:
        return {"issue": []}
    return {"issue": result[0]}

# Endpoint to get client names based on division
@app.get("/v1/retail_complains/client_names_by_division", operation_id="get_client_names_by_division", summary="Retrieves the first, middle, and last names of clients associated with a specified division. The division is determined by the input parameter, which represents the division of the district.")
async def get_client_names_by_division(division: str = Query(..., description="Division of the district")):
    cursor.execute("SELECT T2.first, T2.middle, T2.last FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T1.division = ?", (division,))
    result = cursor.fetchall()
    if not result:
        return {"client_names": []}
    return {"client_names": result}

# Endpoint to get the social information of the client with the most events
@app.get("/v1/retail_complains/top_client_social", operation_id="get_top_client_social", summary="Retrieves the social information of the client who has the highest number of associated events. The data is obtained by joining the client and events tables, grouping by client_id, and ordering by the count of client_id in descending order. The result is limited to the top record.")
async def get_top_client_social():
    cursor.execute("SELECT T1.social FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID GROUP BY T1.client_id ORDER BY COUNT(T1.client_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"social": []}
    return {"social": result[0]}

# Endpoint to get the city with the most reviews of a given star rating
@app.get("/v1/retail_complains/top_city_by_star_rating", operation_id="get_top_city_by_star_rating", summary="Retrieves the city with the highest number of reviews for a specified star rating. The operation groups reviews by city and orders them in descending order based on the count of reviews. The city with the most reviews of the given star rating is returned.")
async def get_top_city_by_star_rating(stars: int = Query(..., description="Star rating of the reviews")):
    cursor.execute("SELECT T2.city FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.Stars = ? GROUP BY T2.city ORDER BY COUNT(T2.city) DESC LIMIT 1", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get client addresses based on event date received and submission method
@app.get("/v1/retail_complains/client_addresses_by_event", operation_id="get_client_addresses_by_event", summary="Retrieves the addresses of clients who have submitted an event on a specific date using a particular submission method. The response includes the first and second lines of the client's address.")
async def get_client_addresses_by_event(date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format"), submitted_via: str = Query(..., description="Submission method")):
    cursor.execute("SELECT T1.address_1, T1.address_2 FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Date received` = ? AND T2.`Submitted via` = ?", (date_received, submitted_via))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get the count of clients based on sex and call center priority
@app.get("/v1/retail_complains/client_count_by_sex_priority", operation_id="get_client_count_by_sex_priority", summary="Retrieves the total number of clients categorized by sex and call center priority. The operation filters clients based on the provided sex and priority level, then calculates the count for each category.")
async def get_client_count_by_sex_priority(sex: str = Query(..., description="Sex of the client"), priority: int = Query(..., description="Priority level")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T1.sex = ? AND T2.priority = ?", (sex, priority))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct servers based on submission method and timely response
@app.get("/v1/retail_complains/distinct_servers_by_submission_timely", operation_id="get_distinct_servers_by_submission_timely", summary="Retrieves a list of unique servers where complaints were submitted via a specific method and received a timely response. The submission method and timely response status are provided as input parameters.")
async def get_distinct_servers_by_submission_timely(submitted_via: str = Query(..., description="Submission method"), timely_response: str = Query(..., description="Timely response (Yes/No)")):
    cursor.execute("SELECT DISTINCT T2.server FROM events AS T1 INNER JOIN callcenterlogs AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T1.`Submitted via` = ? AND T1.`Timely response?` = ?", (submitted_via, timely_response))
    result = cursor.fetchall()
    if not result:
        return {"servers": []}
    return {"servers": result}

# Endpoint to get distinct issues based on client name
@app.get("/v1/retail_complains/distinct_issues_by_client_name", operation_id="get_distinct_issues_by_client_name", summary="Retrieves a list of unique issues associated with a specific client, identified by their full name (first, middle, and last). This operation is useful for understanding the range of distinct issues a client has encountered.")
async def get_distinct_issues_by_client_name(first: str = Query(..., description="First name of the client"), middle: str = Query(..., description="Middle name of the client"), last: str = Query(..., description="Last name of the client")):
    cursor.execute("SELECT DISTINCT T2.Issue FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.middle = ? AND T1.last = ?", (first, middle, last))
    result = cursor.fetchall()
    if not result:
        return {"issues": []}
    return {"issues": result}

# Endpoint to get the division with the most clients
@app.get("/v1/retail_complains/top_division_by_clients", operation_id="get_top_division_by_clients", summary="Get the division with the most clients")
async def get_top_division_by_clients():
    cursor.execute("SELECT T2.division FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id GROUP BY T2.division ORDER BY COUNT(T2.division) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"division": []}
    return {"division": result[0]}

# Endpoint to get client names based on server and date received
@app.get("/v1/retail_complains/client_names_by_server_date", operation_id="get_client_names_by_server_date", summary="Retrieves the first, middle, and last names of clients who have interacted with a specific server on a given date. The server and date are provided as input parameters.")
async def get_client_names_by_server_date(server: str = Query(..., description="Server name"), date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T2.server = ? AND T2.`Date received` = ?", (server, date_received))
    result = cursor.fetchall()
    if not result:
        return {"client_names": []}
    return {"client_names": result}

# Endpoint to get average service time for different age groups
@app.get("/v1/retail_complains/average_service_time_age_groups", operation_id="get_average_service_time", summary="Retrieves the average service time in minutes for three age groups: teenagers (14-19), adults (20-65), and elders (66+). The calculation is based on the sum of service times for each group divided by the number of clients in that group.")
async def get_average_service_time():
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.age > 13 AND T1.age <= 19 THEN 60 * strftime('%H', ser_time) + strftime('%M', ser_time) + strftime('%S', ser_time) / 60 ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.age > 13 AND T1.age <= 19 THEN 1 ELSE 0 END) AS teenagerAverageMins, CAST(SUM(CASE WHEN T1.age > 19 AND T1.age <= 65 THEN 60 * strftime('%H', ser_time) + strftime('%M', ser_time) + strftime('%S', ser_time) / 60 ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.age > 19 AND T1.age <= 65 THEN 1 ELSE 0 END) AS adultAverageMins , CAST(SUM(CASE WHEN T1.age > 65 THEN 60 * strftime('%H', ser_time) + strftime('%M', ser_time) + strftime('%S', ser_time) / 60 ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.age > 65 THEN 1 ELSE 0 END) AS elderAverageMins FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client`")
    result = cursor.fetchone()
    if not result:
        return {"teenagerAverageMins": [], "adultAverageMins": [], "elderAverageMins": []}
    return {"teenagerAverageMins": result[0], "adultAverageMins": result[1], "elderAverageMins": result[2]}

# Endpoint to get the percentage of clients over 65 years old
@app.get("/v1/retail_complains/percentage_clients_over_65", operation_id="get_percentage_clients_over_65", summary="Retrieves the percentage of clients who are over 65 years old. This operation calculates the proportion of clients in the database that meet the specified age criteria, providing a statistical overview of the client demographic.")
async def get_percentage_clients_over_65():
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.age > 65 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.age) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of clients of a specific gender in a given city
@app.get("/v1/retail_complains/percentage_clients_gender_city", operation_id="get_percentage_clients_gender_city", summary="Retrieves the percentage of clients of a specific gender residing in a given city. The operation calculates this percentage by summing the number of clients of the specified gender in the provided city and dividing it by the total number of clients in that city. The gender and city are provided as input parameters.")
async def get_percentage_clients_gender_city(gender: str = Query(..., description="Gender of the client (e.g., 'Male')"), city: str = Query(..., description="City of the client (e.g., 'Indianapolis')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN sex = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(client_id) FROM client WHERE city = ?", (gender, city))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to determine the dominant email provider among clients in a specific age range
@app.get("/v1/retail_complains/dominant_email_provider_age_range", operation_id="get_dominant_email_provider_age_range", summary="This operation identifies the most prevalent email provider among clients within a specified age range. It compares the number of Google and Microsoft accounts, based on the provided domains, and returns the dominant provider. The age range is determined by the minimum and maximum age values.")
async def get_dominant_email_provider_age_range(gmail_domain: str = Query(..., description="Domain for Google accounts (e.g., '%@gmail.com')"), outlook_domain: str = Query(..., description="Domain for Microsoft accounts (e.g., '%@outlook.com')"), min_age: int = Query(..., description="Minimum age of the client"), max_age: int = Query(..., description="Maximum age of the client")):
    cursor.execute("SELECT CASE WHEN SUM(CASE WHEN email LIKE ? THEN 1 ELSE 0 END) > SUM(CASE WHEN email LIKE ? THEN 1 ELSE 0 END) THEN 'Google account' ELSE 'Microsoft account' END FROM client WHERE age BETWEEN ? AND ?", (gmail_domain, outlook_domain, min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"dominant_email_provider": []}
    return {"dominant_email_provider": result[0]}

# Endpoint to get client details by email
@app.get("/v1/retail_complains/client_details_by_email", operation_id="get_client_details_by_email", summary="Retrieves the full name of a client based on the provided email address. The operation returns the first, middle, and last names of the client associated with the input email.")
async def get_client_details_by_email(email: str = Query(..., description="Email of the client (e.g., 'emily.garcia43@outlook.com')")):
    cursor.execute("SELECT first, middle, last FROM client WHERE email = ?", (email,))
    result = cursor.fetchone()
    if not result:
        return {"first": [], "middle": [], "last": []}
    return {"first": result[0], "middle": result[1], "last": result[2]}

# Endpoint to get the first name of clients with the highest priority call center logs
@app.get("/v1/retail_complains/client_first_name_highest_priority", operation_id="get_client_first_name_highest_priority", summary="Retrieves the first names of clients who have call center logs with the highest priority. This operation identifies the maximum priority level in the call center logs and returns the first names of clients associated with those logs.")
async def get_client_first_name_highest_priority():
    cursor.execute("SELECT T1.first FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T2.priority = ( SELECT MAX(priority) FROM callcenterlogs )")
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get client emails by call center log type
@app.get("/v1/retail_complains/client_emails_by_log_type", operation_id="get_client_emails_by_log_type", summary="Retrieves the email addresses of clients associated with a specific type of call center log. The log type is provided as an input parameter, allowing for targeted retrieval of client emails based on the type of call center log.")
async def get_client_emails_by_log_type(log_type: str = Query(..., description="Type of call center log (e.g., 'PS')")):
    cursor.execute("SELECT T1.email FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T2.type = ?", (log_type,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get the last names of clients over a certain age with a specific server in call center logs
@app.get("/v1/retail_complains/client_last_names_age_server", operation_id="get_client_last_names_age_server", summary="Retrieves the last names of clients who are older than a specified age and have interacted with a particular server in the call center logs. The operation filters clients based on their age and the server they contacted, providing a targeted list of last names.")
async def get_client_last_names_age_server(min_age: int = Query(..., description="Minimum age of the client"), server: str = Query(..., description="Server in call center logs (e.g., 'YIFAT')")):
    cursor.execute("SELECT T1.last FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T1.age > ? AND T2.server = ?", (min_age, server))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get the count of clients in a specific city with a specific call center log outcome
@app.get("/v1/retail_complains/client_count_city_outcome", operation_id="get_client_count_city_outcome", summary="Retrieves the total number of clients from a specific city who have a particular call center log outcome. The operation filters clients based on their city and the outcome of their call center logs, then returns the count of clients who meet these criteria.")
async def get_client_count_city_outcome(city: str = Query(..., description="City of the client (e.g., 'New York City')"), outcome: str = Query(..., description="Outcome of the call center log (e.g., 'AGENT')")):
    cursor.execute("SELECT COUNT(T2.`rand client`) FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T1.city = ? AND T2.outcome = ?", (city, outcome))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client details based on consumer dispute status
@app.get("/v1/retail_complains/client_details_consumer_dispute", operation_id="get_client_details_consumer_dispute", summary="Retrieves the full names of clients who have a specific consumer dispute status. The status is provided as an input parameter.")
async def get_client_details_consumer_dispute(dispute_status: str = Query(..., description="Consumer dispute status (e.g., 'Yes')")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Consumer disputed?` = ?", (dispute_status,))
    result = cursor.fetchall()
    if not result:
        return {"client_details": []}
    return {"client_details": [{"first": row[0], "middle": row[1], "last": row[2]} for row in result]}

# Endpoint to get complaint IDs for clients from a specific year
@app.get("/v1/retail_complains/complaint_ids_by_year", operation_id="get_complaint_ids_by_year", summary="Retrieves a list of complaint IDs for clients who joined in a specific year. The year is provided as an input parameter.")
async def get_complaint_ids_by_year(year: int = Query(..., description="Year of the client")):
    cursor.execute("SELECT T2.`Complaint ID` FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T1.year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"complaint_ids": []}
    return {"complaint_ids": [row[0] for row in result]}

# Endpoint to get the percentage of clients with a specific email domain from a specific server
@app.get("/v1/retail_complains/percentage_email_domain_by_server", operation_id="get_percentage_email_domain_by_server", summary="Retrieves the percentage of clients associated with a given email domain from a specific server. This operation calculates the proportion of clients with the specified email domain out of the total clients connected to the provided server.")
async def get_percentage_email_domain_by_server(email_domain: str = Query(..., description="Email domain to filter by, e.g., '%@gmail.com'"), server: str = Query(..., description="Server name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.email LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.email) FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T2.server = ?", (email_domain, server))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get client names where the service time minute is greater than a specific value
@app.get("/v1/retail_complains/client_names_by_service_time", operation_id="get_client_names_by_service_time", summary="Retrieves the first, middle, and last names of clients who have a service time greater than the specified minute value. The service time is compared in 'MM' format.")
async def get_client_names_by_service_time(service_time_minute: str = Query(..., description="Service time minute in 'MM' format")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE strftime('%M', T2.ser_time) > ?", (service_time_minute,))
    result = cursor.fetchall()
    if not result:
        return {"client_names": []}
    return {"client_names": [{"first": row[0], "middle": row[1], "last": row[2]} for row in result]}

# Endpoint to get the last names of clients with a specific call priority
@app.get("/v1/retail_complains/client_last_names_by_priority", operation_id="get_client_last_names_by_priority", summary="Retrieves the last names of up to five clients who have a specific call priority. The call priority is used to filter the clients, and the last names are fetched from the client table.")
async def get_client_last_names_by_priority(priority: int = Query(..., description="Call priority")):
    cursor.execute("SELECT T1.last FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T2.priority = ? LIMIT 5", (priority,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get call IDs for clients whose first name starts with a specific letter
@app.get("/v1/retail_complains/call_ids_by_first_name", operation_id="get_call_ids_by_first_name", summary="Retrieves a list of call IDs associated with clients whose first name begins with a specified letter. This operation filters the client database based on the provided first name starting letter and returns the corresponding call IDs from the call center logs.")
async def get_call_ids_by_first_name(first_name_start: str = Query(..., description="First name starting letter, e.g., 'B%'")):
    cursor.execute("SELECT T2.call_id FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T1.first LIKE ?", (first_name_start,))
    result = cursor.fetchall()
    if not result:
        return {"call_ids": []}
    return {"call_ids": [row[0] for row in result]}

# Endpoint to get distinct products for a client with a specific full name
@app.get("/v1/retail_complains/distinct_products_by_full_name", operation_id="get_distinct_products_by_full_name", summary="Retrieves a list of distinct products associated with a client, identified by their full name. The operation filters clients based on their first, middle, and last names, and returns the unique products linked to the matching client.")
async def get_distinct_products_by_full_name(first: str = Query(..., description="First name of the client"), middle: str = Query(..., description="Middle name of the client"), last: str = Query(..., description="Last name of the client")):
    cursor.execute("SELECT DISTINCT T2.Product FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.first = ? AND T1.middle = ? AND T1.last = ?", (first, middle, last))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get first names of clients with a specific timely response and sex
@app.get("/v1/retail_complains/client_first_names_by_response_and_sex", operation_id="get_client_first_names_by_response_and_sex", summary="Retrieves the first names of clients who have a specific timely response to an event and belong to a certain sex category. The response and sex are provided as input parameters.")
async def get_client_first_names_by_response_and_sex(timely_response: str = Query(..., description="Timely response value, e.g., 'No'"), sex: str = Query(..., description="Sex of the client, e.g., 'Male'")):
    cursor.execute("SELECT T1.first FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Timely response?` = ? AND T1.sex = ?", (timely_response, sex))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the product of the oldest client above a specific age
@app.get("/v1/retail_complains/product_of_oldest_client_above_age", operation_id="get_product_of_oldest_client_above_age", summary="Retrieves the product associated with the oldest client whose age exceeds the provided age threshold. The operation uses the age threshold to filter clients and then identifies the oldest client based on their client ID. The product associated with this client is then returned.")
async def get_product_of_oldest_client_above_age(age: int = Query(..., description="Age threshold")):
    cursor.execute("SELECT T2.Product FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.age > ? ORDER BY T1.client_id DESC LIMIT 1", (age,))
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": result[0]}

# Endpoint to get the age distribution of clients for a specific product
@app.get("/v1/retail_complains/age_distribution_by_product", operation_id="get_age_distribution_by_product", summary="Retrieves the age distribution of clients who have interacted with a specific product. The response includes the count of clients aged between 14 and 19, between 20 and 65, and above 65. To use this endpoint, provide the name of the product as a parameter.")
async def get_age_distribution_by_product(product: str = Query(..., description="Product name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.age > 13 AND T1.age <= 19 THEN 1 ELSE 0 END), SUM(CASE WHEN T1.age > 19 AND T1.age <= 65 THEN 1 ELSE 0 END) AS adult , SUM(CASE WHEN T1.age > 65 THEN 1 ELSE 0 END) AS elder FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.Product = ?", (product,))
    result = cursor.fetchone()
    if not result:
        return {"age_distribution": []}
    return {"age_distribution": {"teen": result[0], "adult": result[1], "elder": result[2]}}

# Endpoint to get issues for complaints where the service time minute is less than a specific value
@app.get("/v1/retail_complains/issues_by_service_time", operation_id="get_issues_by_service_time", summary="Retrieves the issues associated with complaints that have a service time of less than the specified minute value. The service time is compared in the 'MM' format.")
async def get_issues_by_service_time(service_time_minute: str = Query(..., description="Service time minute in 'MM' format")):
    cursor.execute("SELECT T2.Issue FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE strftime('%M', T1.ser_time) < ?", (service_time_minute,))
    result = cursor.fetchall()
    if not result:
        return {"issues": []}
    return {"issues": [row[0] for row in result]}

# Endpoint to get the date received for complaints submitted via a specific method
@app.get("/v1/retail_complains/date_received_by_submission_method", operation_id="get_date_received_by_submission_method", summary="Retrieves the date received for complaints submitted via a specific method. The method of submission is provided as an input parameter, allowing the user to filter the results accordingly.")
async def get_date_received_by_submission_method(submitted_via: str = Query(..., description="Method of submission (e.g., 'Fax')")):
    cursor.execute("SELECT T1.`Date received` FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T2.`Submitted via` = ?", (submitted_via,))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": [row[0] for row in result]}

# Endpoint to get client names based on a specific issue
@app.get("/v1/retail_complains/client_names_by_issue", operation_id="get_client_names_by_issue", summary="Retrieves the first, middle, and last names of clients who have reported a specific issue. The issue type is provided as an input parameter, allowing for targeted retrieval of client names associated with that issue.")
async def get_client_names_by_issue(issue: str = Query(..., description="Issue type (e.g., 'Balance transfer')")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.Issue = ?", (issue,))
    result = cursor.fetchall()
    if not result:
        return {"clients": []}
    return {"clients": [{"first": row[0], "middle": row[1], "last": row[2]} for row in result]}

# Endpoint to get client emails based on submission method
@app.get("/v1/retail_complains/client_emails_by_submission_method", operation_id="get_client_emails_by_submission_method", summary="Retrieves the email addresses of clients who submitted their complaints using the specified method. The method of submission is provided as an input parameter.")
async def get_client_emails_by_submission_method(submitted_via: str = Query(..., description="Method of submission (e.g., 'Postal mail')")):
    cursor.execute("SELECT T1.email FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Submitted via` = ?", (submitted_via,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get the average age of clients based on company response
@app.get("/v1/retail_complains/average_age_by_company_response", operation_id="get_average_age_by_company_response", summary="Retrieves the average age of clients who have received a specific company response. The response is determined by the provided company_response parameter, which should be a valid company response value (e.g., 'Closed with relief').")
async def get_average_age_by_company_response(company_response: str = Query(..., description="Company response to consumer (e.g., 'Closed with relief')")):
    cursor.execute("SELECT AVG(T1.age) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Company response to consumer` = ?", (company_response,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the average age of clients based on call type
@app.get("/v1/retail_complains/average_age_by_call_type", operation_id="get_average_age_by_call_type", summary="Retrieves the average age of clients who have made a specific type of call. The call type is provided as an input parameter, allowing the calculation of the average age for clients who have made calls of that particular type.")
async def get_average_age_by_call_type(call_type: str = Query(..., description="Type of call (e.g., 'TT')")):
    cursor.execute("SELECT AVG(T1.age) FROM client AS T1 INNER JOIN callcenterlogs AS T2 ON T1.client_id = T2.`rand client` WHERE T2.type = ?", (call_type,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get call center logs based on year received and server
@app.get("/v1/retail_complains/call_center_logs_by_year_and_server", operation_id="get_call_center_logs_by_year_and_server", summary="Retrieves call center logs filtered by the specified year of receipt and server. The year should be provided in 'YYYY' format, and the server name should be the exact name (e.g., 'AVIDAN'). The response includes the complaint ID, call ID, and final phone number associated with each log entry.")
async def get_call_center_logs_by_year_and_server(year: str = Query(..., description="Year received in 'YYYY' format"), server: str = Query(..., description="Server name (e.g., 'AVIDAN')")):
    cursor.execute("SELECT `Complaint ID`, call_id, phonefinal FROM callcenterlogs WHERE strftime('%Y', `Date received`) = ? AND server = ?", (year, server))
    result = cursor.fetchall()
    if not result:
        return {"logs": []}
    return {"logs": [{"complaint_id": row[0], "call_id": row[1], "phonefinal": row[2]} for row in result]}

# Endpoint to get the average service time between two dates
@app.get("/v1/retail_complains/average_service_time_between_dates", operation_id="get_average_service_time_between_dates", summary="Retrieves the average service time for customer complaints received between the specified start and end dates. The service time is calculated based on the 'ser_time' field in the 'callcenterlogs' table, which is a string representation of the time in 'HH:MM' format. The average is computed by converting the 'MM' part of the time to a real number and taking the mean. The result is a single value representing the average service time in minutes.")
async def get_average_service_time_between_dates(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT AVG(CAST(SUBSTR(ser_time, 4, 2) AS REAL)) FROM callcenterlogs WHERE `Date received` BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_service_time": []}
    return {"average_service_time": result[0]}

# Endpoint to get the count of events by year for a specific product and issue
@app.get("/v1/retail_complains/event_count_by_year_product_issue", operation_id="get_event_count_by_year_product_issue", summary="Retrieves the annual count of events for a specified product and issue. This operation groups events by year and returns the total count for each year. The product and issue parameters are used to filter the events.")
async def get_event_count_by_year_product_issue(product: str = Query(..., description="Product type (e.g., 'Credit card')"), issue: str = Query(..., description="Issue type (e.g., 'Overlimit fee')")):
    cursor.execute("SELECT strftime('%Y', `Date received`), COUNT(`Date received`) FROM events WHERE product = ? AND issue = ? GROUP BY strftime('%Y', `Date received`) HAVING COUNT(`Date received`)", (product, issue))
    result = cursor.fetchall()
    if not result:
        return {"event_counts": []}
    return {"event_counts": [{"year": row[0], "count": row[1]} for row in result]}

# Endpoint to get the count of clients based on division, sex, and age
@app.get("/v1/retail_complains/client_count_by_division_sex_age", operation_id="get_client_count_by_division_sex_age", summary="Retrieves the number of clients within a specified division, categorized by sex and age. The division is identified by its name, and the sex is specified as either 'Male' or 'Female'. The age range is determined by the provided maximum age.")
async def get_client_count_by_division_sex_age(division: str = Query(..., description="Division name (e.g., 'Middle Atlantic')"), sex: str = Query(..., description="Sex (e.g., 'Female')"), max_age: int = Query(..., description="Maximum age")):
    cursor.execute("SELECT COUNT(T1.sex) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.division = ? AND T1.sex = ? AND T1.age < ?", (division, sex, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latest client details based on city
@app.get("/v1/retail_complains/latest_client_details_by_city", operation_id="get_latest_client_details_by_city", summary="Retrieves the most recent client details for a specified city. The operation returns the client's email and phone number, along with the date of the latest interaction. The city is identified by its name, and the results are sorted by year, month, and day in descending order.")
async def get_latest_client_details_by_city(city: str = Query(..., description="City name (e.g., 'Indianapolis')")):
    cursor.execute("SELECT T1.year, T1.month, T1.day, T1.email, T1.phone FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.city = ? ORDER BY T1.year DESC, T1.month DESC, T1.day DESC LIMIT 1", (city,))
    result = cursor.fetchone()
    if not result:
        return {"client_details": []}
    return {"client_details": {"year": result[0], "month": result[1], "day": result[2], "email": result[3], "phone": result[4]}}

# Endpoint to get cities from reviews with a specific star rating and date pattern
@app.get("/v1/retail_complains/cities_from_reviews", operation_id="get_cities_from_reviews", summary="Retrieves a list of cities with reviews that match a specific star rating and date pattern. The reviews are filtered by the provided star rating and date pattern, and the results are ordered by date in descending order. The number of results returned can be limited by specifying a value for the limit parameter.")
async def get_cities_from_reviews(stars: int = Query(..., description="Star rating of the review"), date_pattern: str = Query(..., description="Date pattern in 'YYYY%' format"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.city FROM reviews AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.Stars = ? AND T1.Date LIKE ? ORDER BY T1.Date DESC LIMIT ?", (stars, date_pattern, limit))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the maximum service time for a specific issue
@app.get("/v1/retail_complains/max_service_time_for_issue", operation_id="get_max_service_time_for_issue", summary="Retrieves the longest service time for a given issue type. This operation identifies the maximum service time from the call center logs for a specific issue, as recorded in the events table. The issue type is provided as an input parameter.")
async def get_max_service_time_for_issue(issue: str = Query(..., description="Issue type")):
    cursor.execute("SELECT MAX(T1.ser_time) FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T2.issue = ?", (issue,))
    result = cursor.fetchone()
    if not result:
        return {"max_service_time": []}
    return {"max_service_time": result[0]}

# Endpoint to get social and state information for a client with a specific phone number
@app.get("/v1/retail_complains/client_social_state_by_phone", operation_id="get_client_social_state_by_phone", summary="Retrieves the social and state information for a client based on the provided phone number. This operation fetches data from the client table, joining it with the district and state tables to provide comprehensive details about the client's social status and state of residence.")
async def get_client_social_state_by_phone(phone: str = Query(..., description="Phone number of the client")):
    cursor.execute("SELECT T1.social, T1.state FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN state AS T3 ON T2.state_abbrev = T3.StateCode WHERE T1.phone = ?", (phone,))
    result = cursor.fetchall()
    if not result:
        return {"client_info": []}
    return {"client_info": [{"social": row[0], "state": row[1]} for row in result]}

# Endpoint to get client details based on division
@app.get("/v1/retail_complains/client_details_by_division", operation_id="get_client_details_by_division", summary="Retrieves the first, middle, and last names, as well as the phone number of clients belonging to a specific division. The division is determined by the input parameter.")
async def get_client_details_by_division(division: str = Query(..., description="Division of the district")):
    cursor.execute("SELECT T1.first, T1.middle, T1.last, T1.phone FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.division = ?", (division,))
    result = cursor.fetchall()
    if not result:
        return {"client_details": []}
    return {"client_details": [{"first": row[0], "middle": row[1], "last": row[2], "phone": row[3]} for row in result]}

# Endpoint to get the longest response time for a client's complaint
@app.get("/v1/retail_complains/longest_response_time", operation_id="get_longest_response_time", summary="Retrieves the longest response time for a client's complaint, along with the associated social media platform and company response. The response time is calculated based on the difference between the date the complaint was sent to the company and the date it was received. The results can be limited by specifying the maximum number of records to return.")
async def get_longest_response_time(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.social, 365 * (strftime('%Y', T2.`Date sent to company`) - strftime('%Y', T2.`Date received`)) + 30 * (strftime('%M', T2.`Date sent to company`) - strftime('%M', T2.`Date received`)) + (strftime('%d', T2.`Date sent to company`) - strftime('%d', T2.`Date received`)), T2.`Company response to consumer` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID ORDER BY 365 * (strftime('%Y', T2.`Date sent to company`) - strftime('%Y', T2.`Date received`)) + 30 * (strftime('%M', T2.`Date sent to company`) - strftime('%M', T2.`Date received`)) + (strftime('%d', T2.`Date sent to company`) - strftime('%d', T2.`Date received`)) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"response_time": []}
    return {"response_time": [{"social": row[0], "days": row[1], "response": row[2]} for row in result]}

# Endpoint to get the count of clients based on sex and age
@app.get("/v1/retail_complains/client_count_by_sex_and_age", operation_id="get_client_count_by_sex_and_age", summary="Retrieves the number of clients who meet the specified sex and minimum age criteria. This operation provides a count of clients based on the given sex and age parameters, offering insights into the client demographics.")
async def get_client_count_by_sex_and_age(sex: str = Query(..., description="Sex of the client"), age: int = Query(..., description="Minimum age of the client")):
    cursor.execute("SELECT COUNT(sex) FROM client WHERE sex = ? AND age > ?", (sex, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client names based on city
@app.get("/v1/retail_complains/client_names_by_city", operation_id="get_client_names_by_city", summary="Retrieves the first and last names of clients residing in a specified city. The city is provided as an input parameter.")
async def get_client_names_by_city(city: str = Query(..., description="City of the client")):
    cursor.execute("SELECT first, last FROM client WHERE city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"client_names": []}
    return {"client_names": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get the maximum age of clients based on sex
@app.get("/v1/retail_complains/max_age_by_sex", operation_id="get_max_age_by_sex", summary="Retrieves the maximum age of clients of a specific sex. The operation returns the highest age value found in the client database for the provided sex.")
async def get_max_age_by_sex(sex: str = Query(..., description="Sex of the client")):
    cursor.execute("SELECT MAX(age) FROM client WHERE sex = ?", (sex,))
    result = cursor.fetchone()
    if not result:
        return {"max_age": []}
    return {"max_age": result[0]}

# Endpoint to get the count of clients grouped by division
@app.get("/v1/retail_complains/client_count_by_division", operation_id="get_client_count_by_division", summary="Retrieves the total number of clients categorized by their respective divisions. This operation aggregates client data from the district level, grouping them based on their division affiliation.")
async def get_client_count_by_division():
    cursor.execute("SELECT T2.division, COUNT(T2.division) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id GROUP BY T2.division")
    result = cursor.fetchall()
    if not result:
        return {"division_counts": []}
    return {"division_counts": [{"division": row[0], "count": row[1]} for row in result]}

# Endpoint to get the percentage of female clients in a specific division
@app.get("/v1/retail_complains/percentage_female_clients_by_division", operation_id="get_percentage_female_clients_by_division", summary="Retrieves the percentage of female clients in a specified division. The division is identified by the provided input parameter. The calculation is based on the total number of clients in the division.")
async def get_percentage_female_clients_by_division(division: str = Query(..., description="Division of the district")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.sex = 'Female' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.sex) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.division = ?", (division,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average age of clients in a specific division
@app.get("/v1/retail_complains/average_age_by_division", operation_id="get_average_age_by_division", summary="Retrieves the average age of clients from a specific division. The division is determined by the provided input parameter, which corresponds to the district's division.")
async def get_average_age_by_division(division: str = Query(..., description="Division of the district")):
    cursor.execute("SELECT AVG(T1.age) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.division = ?", (division,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the city with the most clients in a specific region
@app.get("/v1/retail_complains/most_clients_city_by_region", operation_id="get_most_clients_city_by_region", summary="Retrieves the city with the highest number of clients in a specified region. The operation groups clients by city and orders the results by the count of clients in each city, returning the top city. The region is determined by the provided input parameter.")
async def get_most_clients_city_by_region(region: str = Query(..., description="Region of the state")):
    cursor.execute("SELECT T2.city FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN state AS T3 ON T2.state_abbrev = T3.StateCode WHERE T3.Region = ? GROUP BY T2.city ORDER BY COUNT(T2.city) LIMIT 1", (region,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the count of clients with a specific email domain in a specific region
@app.get("/v1/retail_complains/count_clients_email_domain_region", operation_id="get_count_clients_email_domain_region", summary="Retrieves the total number of clients in a specified region whose email addresses belong to a particular domain. The region is determined by the state's region, and the email domain is identified by the suffix following the '@' symbol in the email address.")
async def get_count_clients_email_domain_region(region: str = Query(..., description="Region of the state"), email_domain: str = Query(..., description="Email domain to search for, e.g., '%@outlook.com'")):
    cursor.execute("SELECT COUNT(T1.email) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN state AS T3 ON T2.state_abbrev = T3.StateCode WHERE T3.Region = ? AND T1.email LIKE ?", (region, email_domain))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city with the most clients over a certain age in a specific division
@app.get("/v1/retail_complains/most_clients_city_age_division", operation_id="get_most_clients_city_age_division", summary="Retrieves the city with the highest number of clients above a specified age in a given division. The division and minimum age are provided as input parameters.")
async def get_most_clients_city_age_division(division: str = Query(..., description="Division of the district"), min_age: int = Query(..., description="Minimum age of the clients")):
    cursor.execute("SELECT T2.city FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.division = ? AND T1.age > ? GROUP BY T2.city ORDER BY COUNT(T2.city) DESC LIMIT 1", (division, min_age))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the percentage of complaints with a specific priority in a given year
@app.get("/v1/retail_complains/percentage_complaints_priority_year", operation_id="get_percentage_complaints_priority_year", summary="Retrieves the percentage of complaints with a specified priority level that were received in a given year. The calculation is based on the total number of complaints logged in the call center and the count of complaints with the specified priority level. The year is determined by the date the complaint was received.")
async def get_percentage_complaints_priority_year(priority: int = Query(..., description="Priority level of the complaint"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.priority = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.`Complaint ID`) FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE strftime('%Y', T1.`Date received`) = ?", (priority, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the state abbreviation with the most cities
@app.get("/v1/retail_complains/state_most_cities", operation_id="get_state_most_cities", summary="Retrieves the abbreviation of the state that has the highest number of cities. The data is aggregated and ordered by the count of cities in each state, with the state having the most cities being returned.")
async def get_state_most_cities():
    cursor.execute("SELECT state_abbrev FROM district GROUP BY state_abbrev ORDER BY COUNT(city) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"state_abbrev": []}
    return {"state_abbrev": result[0]}

# Endpoint to get the first name and phone number of a client by complaint ID
@app.get("/v1/retail_complains/client_info_by_complaint_id", operation_id="get_client_info_by_complaint_id", summary="Retrieves the first name and phone number of a client associated with the provided complaint ID. This operation fetches the client's details from the client table and matches it with the corresponding complaint ID from the events table.")
async def get_client_info_by_complaint_id(complaint_id: str = Query(..., description="Complaint ID")):
    cursor.execute("SELECT T1.first, T1.phone FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Complaint ID` = ?", (complaint_id,))
    result = cursor.fetchone()
    if not result:
        return {"first": [], "phone": []}
    return {"first": result[0], "phone": result[1]}

# Endpoint to get the email of clients by the date a complaint was received
@app.get("/v1/retail_complains/client_emails_by_date_received", operation_id="get_client_emails_by_date_received", summary="Retrieves the email addresses of clients who have submitted a complaint on a specific date. The date should be provided in 'YYYY-MM-DD' format.")
async def get_client_emails_by_date_received(date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.email FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Date received` = ?", (date_received,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get the count of clients in a specific city with a specific product in a given year
@app.get("/v1/retail_complains/count_clients_city_product_year", operation_id="get_count_clients_city_product_year", summary="Retrieves the total number of clients from a specific city who have a particular product in a given year. The city, year, and product type are used to filter the results.")
async def get_count_clients_city_product_year(city: str = Query(..., description="City of the client"), year: str = Query(..., description="Year in 'YYYY' format"), product: str = Query(..., description="Product type")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.city = ? AND strftime('%Y', T2.`Date received`) = ? AND T2.Product = ?", (city, year, product))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of clients based on submission method, sex, and date range
@app.get("/v1/retail_complains/count_clients_submission_sex_date_range", operation_id="get_count_clients_submission_sex_date_range", summary="Retrieves the total number of clients who submitted complaints within a specified date range, filtered by submission method and client sex. The date range is defined by the start and end years, while the submission method and client sex are specified as input parameters.")
async def get_count_clients_submission_sex_date_range(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), submission_method: str = Query(..., description="Submission method"), sex: str = Query(..., description="Sex of the client")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE strftime('%Y', T2.`Date received`) BETWEEN ? AND ? AND T2.`Submitted via` = ? AND T1.sex = ?", (start_year, end_year, submission_method, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get phone numbers and complaint IDs based on company response to consumer
@app.get("/v1/retail_complains/phone_complaint_id_by_company_response", operation_id="get_phone_complaint_id_by_company_response", summary="Retrieves phone numbers and their associated complaint IDs for cases where the company's response matches the provided input. This operation is useful for tracking consumer complaints and the company's corresponding responses.")
async def get_phone_complaint_id_by_company_response(company_response: str = Query(..., description="Company response to consumer")):
    cursor.execute("SELECT T1.phone, T2.`Complaint ID` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.`Company response to consumer` = ?", (company_response,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of clients based on city, date received, and issue
@app.get("/v1/retail_complains/client_count_by_city_date_issue", operation_id="get_client_count_by_city_date_issue", summary="Retrieves the total number of clients from a specific city who have reported a particular issue within a given year. The count is based on the date the issue was received.")
async def get_client_count_by_city_date_issue(city: str = Query(..., description="City of the client"), date_received: str = Query(..., description="Date received in 'YYYY%' format"), issue: str = Query(..., description="Issue type")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.city = ? AND T2.`Date received` LIKE ? AND T2.Issue = ?", (city, date_received, issue))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of complaints resolved within a certain time frame
@app.get("/v1/retail_complains/percentage_complaints_resolved_time_frame", operation_id="get_percentage_complaints_resolved_time_frame", summary="Retrieves the percentage of complaints resolved within a specified time frame for a given city and year. The calculation considers the time elapsed between the date a complaint was received and the date it was sent to the company. The result is expressed as a percentage of the total number of complaints for the specified city and year.")
async def get_percentage_complaints_resolved_time_frame(city: str = Query(..., description="City of the client"), year_received: str = Query(..., description="Year received in 'YYYY' format")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN strftime('%J', T2.`Date sent to company`) - strftime('%J', T2.`Date received`) > 5 THEN 1 ELSE 0 END)) AS REAL) * 100 / COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.city = ? AND strftime('%Y', T2.`Date received`) = ?", (city, year_received))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of submissions via a specific method in a given year
@app.get("/v1/retail_complains/submission_count_by_method_year", operation_id="get_submission_count_by_method_year", summary="Retrieves the total number of retail complaints submitted through a specific method during a particular year. The year is specified in 'YYYY' format, and the method of submission is provided as input. This operation does not modify any data.")
async def get_submission_count_by_method_year(year_received: str = Query(..., description="Year received in 'YYYY' format"), submitted_via: str = Query(..., description="Method of submission")):
    cursor.execute("SELECT COUNT(`Submitted via`) FROM events WHERE strftime('%Y', `Date received`) = ? AND `Submitted via` = ?", (year_received, submitted_via))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client IDs based on submission method and date received
@app.get("/v1/retail_complains/client_ids_by_submission_method_date", operation_id="get_client_ids_by_submission_method_date", summary="Retrieves a list of client IDs who submitted complaints via a specific method on a given date. The method of submission and the date received are used to filter the results.")
async def get_client_ids_by_submission_method_date(submitted_via: str = Query(..., description="Method of submission"), date_received: str = Query(..., description="Date received in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.Client_ID FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T2.`Submitted via` = ? AND T1.`Date received` = ?", (submitted_via, date_received))
    result = cursor.fetchall()
    if not result:
        return {"client_ids": []}
    return {"client_ids": result}

# Endpoint to get review dates based on product, city, and state
@app.get("/v1/retail_complains/review_dates_by_product_city_state", operation_id="get_review_dates_by_product_city_state", summary="Retrieves the review dates for a specific product in a given city and state. The operation filters the review dates based on the provided product name, city, and state abbreviation. This endpoint is useful for analyzing review patterns for a product in a particular location.")
async def get_review_dates_by_product_city_state(product: str = Query(..., description="Product name"), city: str = Query(..., description="City"), state_abbrev: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT T2.Date FROM district AS T1 INNER JOIN reviews AS T2 ON T1.district_id = T2.district_id WHERE T2.Product = ? AND T1.city = ? AND T1.state_abbrev = ?", (product, city, state_abbrev))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": result}

# Endpoint to get the count of complaints based on year received, timely response, and company response
@app.get("/v1/retail_complains/complaint_count_by_year_timely_response_company_response", operation_id="get_complaint_count_by_year_timely_response_company_response", summary="Retrieves the total number of complaints that match the specified year of receipt, timely response status, and company response to the consumer. The year of receipt is provided in 'YYYY' format, while the timely response status and company response are represented by their respective values.")
async def get_complaint_count_by_year_timely_response_company_response(year_received: str = Query(..., description="Year received in 'YYYY' format"), timely_response: str = Query(..., description="Timely response status"), company_response: str = Query(..., description="Company response to consumer")):
    cursor.execute("SELECT COUNT(T1.`Complaint ID`) FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE strftime('%Y', T1.`Date received`) = ? AND T2.`Timely response?` = ? AND T2.`Company response to consumer` = ?", (year_received, timely_response, company_response))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct dates received based on age range and sex
@app.get("/v1/retail_complains/distinct_dates_received_by_age_sex", operation_id="get_distinct_dates_received_by_age_sex", summary="Retrieves the unique dates when complaints were received, filtered by a specified age range and the sex of the client. The age range is defined by the minimum and maximum age parameters, while the sex parameter determines the gender of the client.")
async def get_distinct_dates_received_by_age_sex(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), sex: str = Query(..., description="Sex of the client")):
    cursor.execute("SELECT DISTINCT T3.`Date received` FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID INNER JOIN callcenterlogs AS T3 ON T2.`Complaint ID` = T3.`Complaint ID` WHERE T1.age BETWEEN ? AND ? AND T1.sex = ?", (min_age, max_age, sex))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": result}

# Endpoint to get products based on city, state, date, and stars
@app.get("/v1/retail_complains/products_by_city_state_date_stars", operation_id="get_products_by_city_state_date_stars", summary="Retrieves a list of products associated with a specific city, state, date, and star rating. The city and state abbreviation are used to filter the results, while the date and star rating further narrow down the selection. The endpoint returns products that match the provided criteria.")
async def get_products_by_city_state_date_stars(city: str = Query(..., description="City"), state_abbrev: str = Query(..., description="State abbreviation"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT T2.Product FROM district AS T1 INNER JOIN reviews AS T2 ON T1.district_id = T2.district_id WHERE T1.city = ? AND T1.state_abbrev = ? AND T2.Date = ? AND T2.Stars = ?", (city, state_abbrev, date, stars))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the count of stars based on city, state, product, and stars
@app.get("/v1/retail_complains/star_count_by_city_state_product_stars", operation_id="get_star_count_by_city_state_product_stars", summary="Retrieves the total count of a specific star rating for a given product in a particular city and state. This operation allows you to understand the distribution of star ratings for a product within a specific geographical area.")
async def get_star_count_by_city_state_product_stars(city: str = Query(..., description="City"), state_abbrev: str = Query(..., description="State abbreviation"), product: str = Query(..., description="Product name"), stars: int = Query(..., description="Number of stars")):
    cursor.execute("SELECT COUNT(T2.Stars) FROM district AS T1 INNER JOIN reviews AS T2 ON T1.district_id = T2.district_id WHERE T1.city = ? AND T1.state_abbrev = ? AND T2.Product = ? AND T2.Stars = ?", (city, state_abbrev, product, stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct issues for clients based on sex and age
@app.get("/v1/retail_complains/distinct_issues_by_sex_age", operation_id="get_distinct_issues", summary="Retrieves a list of unique issues reported by clients, filtered by their sex and age. The operation considers clients whose age is less than the provided maximum age and returns distinct issues for the specified sex.")
async def get_distinct_issues(sex: str = Query(..., description="Sex of the client"), age: int = Query(..., description="Maximum age of the client")):
    cursor.execute("SELECT DISTINCT T2.Issue FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.sex = ? AND T1.age < ?", (sex, age))
    result = cursor.fetchall()
    if not result:
        return {"issues": []}
    return {"issues": [row[0] for row in result]}

# Endpoint to get distinct products based on server and date received
@app.get("/v1/retail_complains/distinct_products_by_server_date", operation_id="get_distinct_products_by_server_date", summary="Retrieves a list of distinct products associated with a specific server and date. The server is identified by its name, and the date is provided in 'YYYY-MM' format. This operation filters the data based on the server and date, ensuring that only unique products are returned.")
async def get_distinct_products_by_server_date(server: str = Query(..., description="Server name"), date_received: str = Query(..., description="Date received in 'YYYY-MM' format")):
    cursor.execute("SELECT DISTINCT T2.Product FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T1.server = ? AND T2.`Date received` LIKE ?", (server, date_received + '%'))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get divisions based on star rating, date, and product
@app.get("/v1/retail_complains/divisions_by_stars_date_product", operation_id="get_divisions", summary="Retrieves the divisions that match the specified star rating, date, and product. The operation filters the data based on the provided star rating, date, and product name, and returns the corresponding divisions.")
async def get_divisions(stars: int = Query(..., description="Star rating of the product"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), product: str = Query(..., description="Product name")):
    cursor.execute("SELECT T1.division FROM district AS T1 INNER JOIN reviews AS T2 ON T1.district_id = T2.district_id WHERE T2.Stars = ? AND T2.Date = ? AND T2.Product = ?", (stars, date, product))
    result = cursor.fetchall()
    if not result:
        return {"divisions": []}
    return {"divisions": [row[0] for row in result]}

# Endpoint to get the phone number of the oldest client for a specific product
@app.get("/v1/retail_complains/oldest_client_phone_by_product", operation_id="get_oldest_client_phone", summary="Retrieves the phone number of the oldest client who has purchased a specific product. The product name is required as an input parameter.")
async def get_oldest_client_phone(product: str = Query(..., description="Product name")):
    cursor.execute("SELECT T1.phone FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T2.Product = ? ORDER BY T1.age DESC LIMIT 1", (product,))
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the count of complaints submitted via a specific method in a given year
@app.get("/v1/retail_complains/complaint_count_by_submission_method_year", operation_id="get_complaint_count", summary="Retrieves the total number of complaints submitted through a specified method during a particular year. The submission method and year are provided as input parameters.")
async def get_complaint_count(submission_method: str = Query(..., description="Submission method"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.`Complaint ID`) FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE T2.`Submitted via` = ? AND strftime('%Y', T1.`Date received`) = ?", (submission_method, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct products and issues for clients older than a certain age threshold
@app.get("/v1/retail_complains/distinct_products_issues_by_age_threshold", operation_id="get_distinct_products_issues", summary="Retrieves a list of unique product and issue combinations for clients whose age is above the average client age multiplied by 0.6. This operation filters the client data based on age and joins it with the events data to provide a distinct set of products and issues.")
async def get_distinct_products_issues():
    cursor.execute("SELECT DISTINCT T2.Product, T2.Issue FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.age * 100 > ( SELECT AVG(age) * 60 FROM client )")
    result = cursor.fetchall()
    if not result:
        return {"products_issues": []}
    return {"products_issues": [{"product": row[0], "issue": row[1]} for row in result]}

# Endpoint to get the percentage of a specific division in reviews with a given star rating
@app.get("/v1/retail_complains/division_percentage_by_stars", operation_id="get_division_percentage", summary="Retrieves the percentage of reviews with a specific star rating that belong to a given division. This operation calculates the proportion of reviews from a particular division with a specified star rating, out of all reviews in the database. The division is identified by its name, and the star rating is a numerical value.")
async def get_division_percentage(division: str = Query(..., description="Division name"), stars: int = Query(..., description="Star rating of the product")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.division = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.division) FROM district AS T1 INNER JOIN reviews AS T2 ON T1.district_id = T2.district_id WHERE T2.Stars = ?", (division, stars))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/retail_complains/latest_date_received",
    "/v1/retail_complains/min_service_time_by_date?date_received=2017-03-22",
    "/v1/retail_complains/compare_complaint_priorities?complaint_id1=CR2400594&complaint_id2=CR2405641",
    "/v1/retail_complains/client_names_by_year?year=1990",
    "/v1/retail_complains/count_client_ids_by_name?first=Diesel&last=Galloway",
    "/v1/retail_complains/sub_product_by_name_and_date?first=Diesel&last=Galloway&date_received=2014-07-03",
    "/v1/retail_complains/consumer_consent_by_name_and_date?first=Matthew&last=Pierce&date_received=2016-10-28",
    "/v1/retail_complains/days_between_dates_by_name_and_date?first=Matthew&last=Pierce&date_received=2016-10-28",
    "/v1/retail_complains/client_names_by_date_and_server?date_received=2017-03-27&server=MICHAL",
    "/v1/retail_complains/service_time_by_name_and_date?first=Rachel&last=Hicks&date_received=2017-03-27",
    "/v1/retail_complains/count_issues_by_type_and_city?issue=Deposits%20and%20withdrawals&city=New%20York%20City",
    "/v1/retail_complains/client_names_by_company_response?company_response=In%20progress",
    "/v1/retail_complains/count_clients_by_timely_response_and_city?timely_response=No&city=New%20York%20City",
    "/v1/retail_complains/count_clients_by_year_sex_product?year=2016&sex=Male&product=Credit%20card",
    "/v1/retail_complains/division_by_client_name?first=Diesel&last=Galloway",
    "/v1/retail_complains/client_names_by_division_and_sex?division=Pacific&sex=Male",
    "/v1/retail_complains/average_complaints_by_year_range_city_product?start_year=2015&end_year=2017&city=New%20York%20City&product=Credit%20card",
    "/v1/retail_complains/percentage_change_complaints_by_years_city?year1=2017&year2=2016&city=New%20York%20City",
    "/v1/retail_complains/service_time_by_client_id_date?client_id=C00007127&date_received=2017-02-22",
    "/v1/retail_complains/state_by_client_email?email=wyatt.collins@gmail.com",
    "/v1/retail_complains/distinct_sub_products?first=Lennox&middle=Oliver&last=Drake&sex=Male",
    "/v1/retail_complains/sub_issues?first=Gunner&middle=Omer&last=Fuller&sex=Male",
    "/v1/retail_complains/consumer_consent?first=Lyric&middle=Emely&last=Taylor&sex=Female&date_received=2016-05-20",
    "/v1/retail_complains/days_between_dates?date_received=2012-05-18&sex=Male&first=Brantley&middle=Julian&last=Stanley",
    "/v1/retail_complains/district_info_by_date?date=2018-09-11",
    "/v1/retail_complains/reviews_by_city_date?city=Jacksonville&date=2017-07-22",
    "/v1/retail_complains/products_by_city_date?city=Indianapolis&date=2016-10-07",
    "/v1/retail_complains/star_count_by_product_city_date?product=Eagle%20Capital&city=Little%20Rock&date=2013-04-04",
    "/v1/retail_complains/month_day_by_complaint_id?complaint_id=CR0217298",
    "/v1/retail_complains/phone_by_complaint_id?complaint_id=CR0100432",
    "/v1/retail_complains/percentage_female_clients?date_received=2017-03-27",
    "/v1/retail_complains/percentage_consent_provided?sex=Male&first=Mason&middle=Javen&last=Lopez",
    "/v1/retail_complains/count_highest_priority_complaints?date_received=2017-01%",
    "/v1/retail_complains/client_details_by_age?min_age=65",
    "/v1/retail_complains/max_five_star_reviews?stars=5",
    "/v1/retail_complains/states_by_region?region=South",
    "/v1/retail_complains/client_emails_by_outcome?outcome=HANG",
    "/v1/retail_complains/average_age_by_region?region=Midwest",
    "/v1/retail_complains/client_details_by_submission?submitted_via=Fax",
    "/v1/retail_complains/divisions_above_average_reviews?product=Eagle%20Capital",
    "/v1/retail_complains/client_count_age_division?min_age=12&max_age=20&division=Mountain",
    "/v1/retail_complains/client_count_sex_product?sex=Female&product=Credit%20card",
    "/v1/retail_complains/client_names_year_sex_submission?min_year=1980&max_year=2000&sex=Male&submission_method=Referral",
    "/v1/retail_complains/most_common_submission_method_state?state=FL",
    "/v1/retail_complains/average_response_rate_city?response_type=Closed%20with%20explanation&city=New%20Bedford",
    "/v1/retail_complains/percentage_disputed_consumers_city?disputed=Yes&city=Houston",
    "/v1/retail_complains/client_count_tag_city?tag=Servicemember&city=Syracuse",
    "/v1/retail_complains/percentage_high_priority_complaints_state?priority=1&state=California",
    "/v1/retail_complains/age_difference_region?min_age_1=35&max_age_1=55&min_age_2=65&region=Northeast",
    "/v1/retail_complains/latest_complaint_ids?limit=3",
    "/v1/retail_complains/count_emails_not_matching_domain?domain=gmail.com",
    "/v1/retail_complains/client_ids_by_consumer_consent?consent_status=N/A&empty_status=",
    "/v1/retail_complains/complaint_ids_max_date_difference?date_sent=2014-09-25",
    "/v1/retail_complains/distinct_complaint_ids_by_priority?priority=2",
    "/v1/retail_complains/count_outcomes_not_matching?outcome=AGENT",
    "/v1/retail_complains/count_complaint_ids_by_product_server?product=Credit%20card&server=SHARON",
    "/v1/retail_complains/top_region_by_star_rating?stars=1",
    "/v1/retail_complains/top_year_by_sub_product?sub_product=(CD)%20Certificate%20of%20deposit",
    "/v1/retail_complains/count_issues_by_issue_division?issue=Billing%20disputes&division=Mountain",
    "/v1/retail_complains/count_clients_by_state_sex?state=Massachusetts&sex=Male",
    "/v1/retail_complains/client_last_names_by_log_type_server?log_type=PS&server=TOVA",
    "/v1/retail_complains/count_clients_by_product_stars_age?product=Eagle%20National%20Mortgage&stars=1&max_age=35",
    "/v1/retail_complains/count_clients_by_sex_priority_year?sex=Male&priority=0&year=1997",
    "/v1/retail_complains/client_names_by_event_tags_consent?tags=Older%20American&consent_not_equal=N%2FA&consent_not_null=&consent_not_empty=",
    "/v1/retail_complains/top_state_by_priority?priority=0",
    "/v1/retail_complains/count_complaints_by_sex_time_response?sex=Female&start_time_start=15:00:01&start_time_end=23:59:59&timely_response=Yes",
    "/v1/retail_complains/count_complaints_by_time_server_submission_response?service_time=00:05:00&server=DORIT&submitted_via=Phone&company_response=Closed%20with%20explanation",
    "/v1/retail_complains/count_clients_by_last_name_state?last_name=Alvarado&state=MD",
    "/v1/retail_complains/count_reviews_by_age_content?min_age=30&max_age=50&review_content=great",
    "/v1/retail_complains/client_addresses_by_response_dispute?timely_response=Yes&consumer_disputed=Yes",
    "/v1/retail_complains/count_submissions_by_sex_year?sex=Female&year=2000&submitted_via=Web",
    "/v1/retail_complains/consumer_complaint_narrative?first=Brenda&last=Mayer",
    "/v1/retail_complains/count_emails_by_date_domain?date_received_1=2017-02%25&date_received_2=2017-01%25&email_domain=%25@gmail.com",
    "/v1/retail_complains/average_star_rating_by_state?state=Oregon",
    "/v1/retail_complains/percentage_clients_above_age?age=50&submitted_via=Postal%20mail",
    "/v1/retail_complains/average_age_by_city?city=Norwalk",
    "/v1/retail_complains/count_reviews_by_city_stars?city=Kansas%20City&stars=1",
    "/v1/retail_complains/top_state_by_stars?stars=5",
    "/v1/retail_complains/distinct_submission_methods?first=Kyran&last=Muller",
    "/v1/retail_complains/distinct_products_after_year?year=2005",
    "/v1/retail_complains/service_times_for_client_product?first=Kendall&last=Allen&sex=Female&product=Credit%20card",
    "/v1/retail_complains/latest_service_time_issue",
    "/v1/retail_complains/client_count_city_submission?city=New%20York%20City&submitted_via=Fax",
    "/v1/retail_complains/percentage_male_clients_product?product=Credit%20card",
    "/v1/retail_complains/client_names_tags_no_consent?tags=Older%20American",
    "/v1/retail_complains/latest_client_date",
    "/v1/retail_complains/count_timely_responses_dispute_status?timely_response=No&consumer_disputed=No",
    "/v1/retail_complains/count_service_times_after_minute?minute=15",
    "/v1/retail_complains/most_common_issue_priority?priority=2",
    "/v1/retail_complains/client_names_by_division?division=Pacific",
    "/v1/retail_complains/top_client_social",
    "/v1/retail_complains/top_city_by_star_rating?stars=1",
    "/v1/retail_complains/client_addresses_by_event?date_received=2012-03-14&submitted_via=Postal%20mail",
    "/v1/retail_complains/client_count_by_sex_priority?sex=Female&priority=1",
    "/v1/retail_complains/distinct_servers_by_submission_timely?submitted_via=Phone&timely_response=No",
    "/v1/retail_complains/distinct_issues_by_client_name?first=Kaitlyn&middle=Eliza&last=Elliott",
    "/v1/retail_complains/top_division_by_clients",
    "/v1/retail_complains/client_names_by_server_date?server=MORIAH&date_received=2013-09-11",
    "/v1/retail_complains/average_service_time_age_groups",
    "/v1/retail_complains/percentage_clients_over_65",
    "/v1/retail_complains/percentage_clients_gender_city?gender=Male&city=Indianapolis",
    "/v1/retail_complains/dominant_email_provider_age_range?gmail_domain=%25@gmail.com&outlook_domain=%25@outlook.com&min_age=13&max_age=19",
    "/v1/retail_complains/client_details_by_email?email=emily.garcia43@outlook.com",
    "/v1/retail_complains/client_first_name_highest_priority",
    "/v1/retail_complains/client_emails_by_log_type?log_type=PS",
    "/v1/retail_complains/client_last_names_age_server?min_age=65&server=YIFAT",
    "/v1/retail_complains/client_count_city_outcome?city=New%20York%20City&outcome=AGENT",
    "/v1/retail_complains/client_details_consumer_dispute?dispute_status=Yes",
    "/v1/retail_complains/complaint_ids_by_year?year=1931",
    "/v1/retail_complains/percentage_email_domain_by_server?email_domain=%25%40gmail.com&server=ZOHARI",
    "/v1/retail_complains/client_names_by_service_time?service_time_minute=20",
    "/v1/retail_complains/client_last_names_by_priority?priority=0",
    "/v1/retail_complains/call_ids_by_first_name?first_name_start=B%25",
    "/v1/retail_complains/distinct_products_by_full_name?first=Alexander&middle=Bronx&last=Lewis",
    "/v1/retail_complains/client_first_names_by_response_and_sex?timely_response=No&sex=Male",
    "/v1/retail_complains/product_of_oldest_client_above_age?age=65",
    "/v1/retail_complains/age_distribution_by_product?product=Credit%20card",
    "/v1/retail_complains/issues_by_service_time?service_time_minute=10",
    "/v1/retail_complains/date_received_by_submission_method?submitted_via=Fax",
    "/v1/retail_complains/client_names_by_issue?issue=Balance%20transfer",
    "/v1/retail_complains/client_emails_by_submission_method?submitted_via=Postal%20mail",
    "/v1/retail_complains/average_age_by_company_response?company_response=Closed%20with%20relief",
    "/v1/retail_complains/average_age_by_call_type?call_type=TT",
    "/v1/retail_complains/call_center_logs_by_year_and_server?year=2014&server=AVIDAN",
    "/v1/retail_complains/average_service_time_between_dates?start_date=2017-01-01&end_date=2017-04-01",
    "/v1/retail_complains/event_count_by_year_product_issue?product=Credit%20card&issue=Overlimit%20fee",
    "/v1/retail_complains/client_count_by_division_sex_age?division=Middle%20Atlantic&sex=Female&max_age=18",
    "/v1/retail_complains/latest_client_details_by_city?city=Indianapolis",
    "/v1/retail_complains/cities_from_reviews?stars=5&date_pattern=2016%25&limit=5",
    "/v1/retail_complains/max_service_time_for_issue?issue=Arbitration",
    "/v1/retail_complains/client_social_state_by_phone?phone=100-121-8371",
    "/v1/retail_complains/client_details_by_division?division=Pacific",
    "/v1/retail_complains/longest_response_time?limit=1",
    "/v1/retail_complains/client_count_by_sex_and_age?sex=Female&age=30",
    "/v1/retail_complains/client_names_by_city?city=New%20York%20City",
    "/v1/retail_complains/max_age_by_sex?sex=Male",
    "/v1/retail_complains/client_count_by_division",
    "/v1/retail_complains/percentage_female_clients_by_division?division=South%20Atlantic",
    "/v1/retail_complains/average_age_by_division?division=South%20Atlantic",
    "/v1/retail_complains/most_clients_city_by_region?region=Midwest",
    "/v1/retail_complains/count_clients_email_domain_region?region=Northeast&email_domain=%25%40outlook.com",
    "/v1/retail_complains/most_clients_city_age_division?division=West%20North%20Central&min_age=60",
    "/v1/retail_complains/percentage_complaints_priority_year?priority=2&year=2017",
    "/v1/retail_complains/state_most_cities",
    "/v1/retail_complains/client_info_by_complaint_id?complaint_id=CR0922485",
    "/v1/retail_complains/client_emails_by_date_received?date_received=2014-07-03",
    "/v1/retail_complains/count_clients_city_product_year?city=Omaha&year=2012&product=Credit%20card",
    "/v1/retail_complains/count_clients_submission_sex_date_range?start_year=2012&end_year=2015&submission_method=Email&sex=Male",
    "/v1/retail_complains/phone_complaint_id_by_company_response?company_response=In%20progress",
    "/v1/retail_complains/client_count_by_city_date_issue?city=Portland&date_received=2015%25&issue=Billing%20disputes",
    "/v1/retail_complains/percentage_complaints_resolved_time_frame?city=Houston&year_received=2014",
    "/v1/retail_complains/submission_count_by_method_year?year_received=2012&submitted_via=Email",
    "/v1/retail_complains/client_ids_by_submission_method_date?submitted_via=Fax&date_received=2014-04-16",
    "/v1/retail_complains/review_dates_by_product_city_state?product=Eagle%20Capital&city=Indianapolis&state_abbrev=IN",
    "/v1/retail_complains/complaint_count_by_year_timely_response_company_response?year_received=2015&timely_response=Yes&company_response=Closed%20with%20explanation",
    "/v1/retail_complains/distinct_dates_received_by_age_sex?min_age=20&max_age=40&sex=Female",
    "/v1/retail_complains/products_by_city_state_date_stars?city=Newton&state_abbrev=MA&date=2016-03-14&stars=1",
    "/v1/retail_complains/star_count_by_city_state_product_stars?city=Nashville&state_abbrev=TN&product=Eagle%20National%20Mortgage&stars=5",
    "/v1/retail_complains/distinct_issues_by_sex_age?sex=Male&age=25",
    "/v1/retail_complains/distinct_products_by_server_date?server=TOVA&date_received=2017-03",
    "/v1/retail_complains/divisions_by_stars_date_product?stars=5&date=2017-12-17&product=Eagle%20National%20Mortgage",
    "/v1/retail_complains/oldest_client_phone_by_product?product=Credit%20card",
    "/v1/retail_complains/complaint_count_by_submission_method_year?submission_method=Phone&year=2014",
    "/v1/retail_complains/distinct_products_issues_by_age_threshold",
    "/v1/retail_complains/division_percentage_by_stars?division=East%20North%20Central&stars=5"
]
