from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/financial/financial.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of account IDs based on district and account frequency
@app.get("/v1/financial/count_account_ids_by_district_and_frequency", operation_id="get_count_account_ids_by_district_and_frequency", summary="Retrieves the total number of account IDs associated with a specific district and account frequency. The district and frequency parameters are used to filter the results.")
async def get_count_account_ids_by_district_and_frequency(district: str = Query(..., description="District name"), frequency: str = Query(..., description="Account frequency")):
    cursor.execute("SELECT COUNT(T2.account_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE T1.A3 = ? AND T2.frequency = ?", (district, frequency))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of account IDs based on district
@app.get("/v1/financial/count_account_ids_by_district", operation_id="get_count_account_ids_by_district", summary="Retrieves the total number of account IDs associated with a specific district. The district is identified by its name, which is provided as an input parameter.")
async def get_count_account_ids_by_district(district: str = Query(..., description="District name")):
    cursor.execute("SELECT COUNT(T1.account_id) FROM account AS T1 INNER JOIN loan AS T2 ON T1.account_id = T2.account_id INNER JOIN district AS T3 ON T1.district_id = T3.district_id WHERE T3.A3 = ?", (district,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the year based on average comparison
@app.get("/v1/financial/get_year_based_on_average_comparison", operation_id="get_year_based_on_average_comparison", summary="Retrieves a specific year based on the comparison of average values from two distinct data sets. The operation calculates the average values of A13 and A12, and returns the year '1996' if the average of A13 is greater than that of A12. Otherwise, it returns the year '1995'. This endpoint is useful for identifying trends or patterns based on the comparison of these averages.")
async def get_year_based_on_average_comparison():
    cursor.execute("SELECT DISTINCT IIF(AVG(A13) > AVG(A12), '1996', '1995') FROM district")
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the count of distinct district IDs based on gender and A11 range
@app.get("/v1/financial/count_distinct_district_ids_by_gender_and_a11_range", operation_id="get_count_distinct_district_ids_by_gender_and_a11_range", summary="Retrieves the count of unique districts based on the specified gender and A11 range. The gender and A11 range are used to filter the clients and districts, respectively. The result is the number of distinct districts that meet the specified criteria.")
async def get_count_distinct_district_ids_by_gender_and_a11_range(gender: str = Query(..., description="Gender of the client"), min_a11: int = Query(..., description="Minimum value of A11"), max_a11: int = Query(..., description="Maximum value of A11")):
    cursor.execute("SELECT COUNT(DISTINCT T2.district_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = ? AND T2.A11 BETWEEN ? AND ?", (gender, min_a11, max_a11))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of client IDs based on gender, district, and A11 value
@app.get("/v1/financial/count_client_ids_by_gender_district_and_a11", operation_id="get_count_client_ids_by_gender_district_and_a11", summary="Retrieves the total number of unique client IDs that match the specified gender, district, and minimum A11 value. This operation considers the client's gender and district, as well as the A11 value of the district, to provide an accurate count.")
async def get_count_client_ids_by_gender_district_and_a11(gender: str = Query(..., description="Gender of the client"), district: str = Query(..., description="District name"), min_a11: int = Query(..., description="Minimum value of A11")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = ? AND T2.A3 = ? AND T2.A11 > ?", (gender, district, min_a11))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get account ID and A11 range based on the earliest born female client
@app.get("/v1/financial/get_account_id_and_a11_range_by_earliest_born_female", operation_id="get_account_id_and_a11_range_by_earliest_born_female", summary="Retrieves the account ID and the range of A11 values for the district with the earliest born female client. The gender of the client is used to identify the district. The account ID and A11 range are determined based on the district with the earliest born female client and the maximum and minimum A11 values within that district.")
async def get_account_id_and_a11_range_by_earliest_born_female(gender: str = Query(..., description="Gender of the client")):
    cursor.execute("SELECT T1.account_id, (SELECT MAX(A11) - MIN(A11) FROM district) FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T3 ON T1.account_id = T3.account_id INNER JOIN client AS T4 ON T3.client_id = T4.client_id WHERE T2.district_id = (SELECT district_id FROM client WHERE gender = ? ORDER BY birth_date ASC LIMIT 1) ORDER BY T2.A11 DESC LIMIT 1", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"account_id": [], "a11_range": []}
    return {"account_id": result[0], "a11_range": result[1]}

# Endpoint to get account IDs based on the latest born client
@app.get("/v1/financial/get_account_ids_by_latest_born_client", operation_id="get_account_ids_by_latest_born_client", summary="Retrieves the account IDs associated with the client who was born most recently. The operation joins data from the account, disp, client, and district tables, filtering for the client with the latest birth date. The results are grouped by district and account ID.")
async def get_account_ids_by_latest_born_client():
    cursor.execute("SELECT T1.account_id FROM account AS T1 INNER JOIN disp AS T2 ON T1.account_id = T2.account_id INNER JOIN client AS T3 ON T2.client_id = T3.client_id INNER JOIN district AS T4 ON T4.district_id = T1.district_id WHERE T2.client_id = (SELECT client_id FROM client ORDER BY birth_date DESC LIMIT 1) GROUP BY T4.A11, T1.account_id")
    result = cursor.fetchall()
    if not result:
        return {"account_ids": []}
    return {"account_ids": [row[0] for row in result]}

# Endpoint to get the count of account IDs based on disp type and account frequency
@app.get("/v1/financial/count_account_ids_by_disp_type_and_frequency", operation_id="get_count_account_ids_by_disp_type_and_frequency", summary="Retrieves the total number of account IDs that match a specific disp type and account frequency. The disp type and account frequency are provided as input parameters.")
async def get_count_account_ids_by_disp_type_and_frequency(disp_type: str = Query(..., description="Disp type"), frequency: str = Query(..., description="Account frequency")):
    cursor.execute("SELECT COUNT(T1.account_id) FROM account AS T1 INNER JOIN disp AS T2 ON T1.account_id = T2.account_id WHERE T2.type = ? AND T1.frequency = ?", (disp_type, frequency))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client IDs based on account frequency and disp type
@app.get("/v1/financial/get_client_ids_by_frequency_and_disp_type", operation_id="get_client_ids_by_frequency_and_disp_type", summary="Retrieves the unique identifiers of clients who have accounts with a specified frequency and disp type. The operation filters accounts based on the provided frequency and disp type, and returns the corresponding client IDs.")
async def get_client_ids_by_frequency_and_disp_type(frequency: str = Query(..., description="Account frequency"), disp_type: str = Query(..., description="Disp type")):
    cursor.execute("SELECT T2.client_id FROM account AS T1 INNER JOIN disp AS T2 ON T1.account_id = T2.account_id WHERE T1.frequency = ? AND T2.type = ?", (frequency, disp_type))
    result = cursor.fetchall()
    if not result:
        return {"client_ids": []}
    return {"client_ids": [row[0] for row in result]}

# Endpoint to get account ID based on loan date year and account frequency
@app.get("/v1/financial/get_account_id_by_loan_date_year_and_frequency", operation_id="get_account_id_by_loan_date_year_and_frequency", summary="Retrieves the account ID associated with the loan that occurred in a specific year and has a particular frequency. The operation filters loans by the provided year and account frequency, then returns the account ID of the loan with the highest amount. The year should be in 'YYYY' format.")
async def get_account_id_by_loan_date_year_and_frequency(year: str = Query(..., description="Year in 'YYYY' format"), frequency: str = Query(..., description="Account frequency")):
    cursor.execute("SELECT T2.account_id FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE STRFTIME('%Y', T1.date) = ? AND T2.frequency = ? ORDER BY T1.amount LIMIT 1", (year, frequency))
    result = cursor.fetchone()
    if not result:
        return {"account_id": []}
    return {"account_id": result[0]}

# Endpoint to get the account ID of the loan with the highest amount for a given year and duration
@app.get("/v1/financial/loan_account_id_by_year_duration", operation_id="get_loan_account_id_by_year_duration", summary="Retrieves the account ID of the loan with the highest amount for a specified year and duration. The year is provided in 'YYYY' format, and the duration is given in months. The operation filters loans based on the provided year and duration, then sorts them by amount in descending order to identify the loan with the highest amount.")
async def get_loan_account_id_by_year_duration(year: str = Query(..., description="Year in 'YYYY' format"), duration: int = Query(..., description="Loan duration in months")):
    cursor.execute("SELECT T1.account_id FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE STRFTIME('%Y', T2.date) = ? AND T1.duration > ? ORDER BY T1.amount DESC LIMIT 1", (year, duration))
    result = cursor.fetchone()
    if not result:
        return {"account_id": []}
    return {"account_id": result[0]}

# Endpoint to get the count of female clients born before a certain year in a specific district
@app.get("/v1/financial/count_female_clients_by_year_district", operation_id="get_count_female_clients_by_year_district", summary="Retrieves the number of female clients born before a specified year in a given district. The operation filters clients based on their gender, birth year, and district, and returns the count of clients who meet these criteria.")
async def get_count_female_clients_by_year_district(gender: str = Query(..., description="Gender of the client"), year: str = Query(..., description="Year in 'YYYY' format"), district: str = Query(..., description="District name")):
    cursor.execute("SELECT COUNT(T2.client_id) FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T2.gender = ? AND STRFTIME('%Y', T2.birth_date) < ? AND T1.A2 = ?", (gender, year, district))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the account ID of the earliest transaction in a given year
@app.get("/v1/financial/earliest_transaction_account_id_by_year", operation_id="get_earliest_transaction_account_id_by_year", summary="Retrieves the account ID associated with the earliest transaction that occurred in the specified year. The year should be provided in 'YYYY' format.")
async def get_earliest_transaction_account_id_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT account_id FROM trans WHERE STRFTIME('%Y', date) = ? ORDER BY date ASC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"account_id": []}
    return {"account_id": result[0]}

# Endpoint to get distinct account IDs with transactions above a certain amount before a given year
@app.get("/v1/financial/distinct_account_ids_by_year_amount", operation_id="get_distinct_account_ids_by_year_amount", summary="Retrieves unique account identifiers associated with transactions exceeding a specified amount, occurring before a given year. This operation filters accounts based on the provided year and transaction amount, returning only those that meet the criteria.")
async def get_distinct_account_ids_by_year_amount(year: str = Query(..., description="Year in 'YYYY' format"), amount: int = Query(..., description="Transaction amount")):
    cursor.execute("SELECT DISTINCT T2.account_id FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE STRFTIME('%Y', T2.date) < ? AND T1.amount > ?", (year, amount))
    result = cursor.fetchall()
    if not result:
        return {"account_ids": []}
    return {"account_ids": [row[0] for row in result]}

# Endpoint to get client IDs based on the issue date of their card
@app.get("/v1/financial/client_ids_by_card_issue_date", operation_id="get_client_ids_by_card_issue_date", summary="Retrieves a list of client IDs whose cards were issued on the specified date. The input parameter is the issue date in 'YYYY-MM-DD' format.")
async def get_client_ids_by_card_issue_date(issue_date: str = Query(..., description="Issue date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.client_id FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN card AS T3 ON T2.disp_id = T3.disp_id WHERE T3.issued = ?", (issue_date,))
    result = cursor.fetchall()
    if not result:
        return {"client_ids": []}
    return {"client_ids": [row[0] for row in result]}

# Endpoint to get the date of accounts with a specific transaction amount and date
@app.get("/v1/financial/account_date_by_amount_and_date", operation_id="get_account_date_by_amount_and_date", summary="Retrieves the date of accounts that have a transaction with a specified amount and date. The transaction amount and date are provided as input parameters to filter the results.")
async def get_account_date_by_amount_and_date(amount: int = Query(..., description="Transaction amount"), date: str = Query(..., description="Transaction date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.date FROM account AS T1 INNER JOIN trans AS T2 ON T1.account_id = T2.account_id WHERE T2.amount = ? AND T2.date = ?", (amount, date))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": [row[0] for row in result]}

# Endpoint to get district IDs based on the loan date
@app.get("/v1/financial/district_ids_by_loan_date", operation_id="get_district_ids_by_loan_date", summary="Retrieves a list of district IDs associated with loans issued on a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_district_ids_by_loan_date(date: str = Query(..., description="Loan date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.district_id FROM account AS T1 INNER JOIN loan AS T2 ON T1.account_id = T2.account_id WHERE T2.date = ?", (date,))
    result = cursor.fetchall()
    if not result:
        return {"district_ids": []}
    return {"district_ids": [row[0] for row in result]}

# Endpoint to get the highest transaction amount for a card issued on a specific date
@app.get("/v1/financial/highest_transaction_amount_by_card_issue_date", operation_id="get_highest_transaction_amount_by_card_issue_date", summary="Retrieves the highest transaction amount associated with a card issued on a specific date. The input parameter specifies the issue date in 'YYYY-MM-DD' format. The operation returns the amount of the highest transaction linked to the card issued on the provided date.")
async def get_highest_transaction_amount_by_card_issue_date(issue_date: str = Query(..., description="Issue date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T4.amount FROM card AS T1 JOIN disp AS T2 ON T1.disp_id = T2.disp_id JOIN account AS T3 on T2.account_id = T3.account_id JOIN trans AS T4 on T3.account_id = T4.account_id WHERE T1.issued = ? ORDER BY T4.amount DESC LIMIT 1", (issue_date,))
    result = cursor.fetchone()
    if not result:
        return {"amount": []}
    return {"amount": result[0]}

# Endpoint to get the gender of the client with the earliest birth date in a specific district
@app.get("/v1/financial/client_gender_by_district", operation_id="get_client_gender_by_district", summary="Retrieves the gender of the oldest client in a given district. The client is determined by the earliest birth date.")
async def get_client_gender_by_district():
    cursor.execute("SELECT T2.gender FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id ORDER BY T1.A11 DESC, T2.birth_date ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the transaction amount for the loan with the highest amount and earliest transaction date
@app.get("/v1/financial/transaction_amount_by_highest_loan", operation_id="get_transaction_amount_by_highest_loan", summary="Retrieves the transaction amount associated with the loan that has the highest principal and the earliest transaction date. This operation provides a single value representing the transaction amount for the loan that meets the specified criteria.")
async def get_transaction_amount_by_highest_loan():
    cursor.execute("SELECT T3.amount FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN trans AS T3 ON T2.account_id = T3.account_id ORDER BY T1.amount DESC, T3.date ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"amount": []}
    return {"amount": result[0]}

# Endpoint to get the count of clients based on gender and district
@app.get("/v1/financial/client_count_by_gender_district", operation_id="get_client_count_by_gender_district", summary="Retrieves the total number of clients categorized by gender and district. The operation requires the gender and district as input parameters to filter the client count accordingly.")
async def get_client_count_by_gender_district(gender: str = Query(..., description="Gender of the client"), district: str = Query(..., description="District name")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = ? AND T2.A2 = ?", (gender, district))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get disp_id based on transaction date and amount
@app.get("/v1/financial/disp_id_by_transaction_date_amount", operation_id="get_disp_id_by_transaction_date_amount", summary="Retrieves the unique identifier (disp_id) of a dispensation based on a specific transaction date and amount. This operation requires the transaction date in 'YYYY-MM-DD' format and the exact transaction amount. It returns the disp_id associated with the provided transaction details, enabling users to access or manage the corresponding dispensation.")
async def get_disp_id_by_transaction_date_amount(date: str = Query(..., description="Transaction date in 'YYYY-MM-DD' format"), amount: int = Query(..., description="Transaction amount")):
    cursor.execute("SELECT T1.disp_id FROM disp AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN trans AS T3 ON T2.account_id = T3.account_id WHERE T3.date = ? AND T3.amount = ?", (date, amount))
    result = cursor.fetchall()
    if not result:
        return {"disp_ids": []}
    return {"disp_ids": [row[0] for row in result]}

# Endpoint to get the count of accounts based on year and district
@app.get("/v1/financial/account_count_by_year_district", operation_id="get_account_count_by_year_district", summary="Retrieves the total number of accounts in a specific district for a given year. The operation filters accounts based on the provided year and district name, and returns the count of matching accounts.")
async def get_account_count_by_year_district(year: str = Query(..., description="Year in 'YYYY' format"), district: str = Query(..., description="District name")):
    cursor.execute("SELECT COUNT(T2.account_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE STRFTIME('%Y', T2.date) = ? AND T1.A2 = ?", (year, district))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get district names based on client birth date and gender
@app.get("/v1/financial/district_names_by_birth_date_gender", operation_id="get_district_names_by_birth_date_gender", summary="Retrieves a list of district names for clients who share a specific birth date and gender. The birth date must be provided in 'YYYY-MM-DD' format, and the gender should be specified as either 'male' or 'female'.")
async def get_district_names_by_birth_date_gender(birth_date: str = Query(..., description="Client birth date in 'YYYY-MM-DD' format"), gender: str = Query(..., description="Gender of the client")):
    cursor.execute("SELECT T1.A2 FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T2.birth_date = ? AND T2.gender = ?", (birth_date, gender))
    result = cursor.fetchall()
    if not result:
        return {"district_names": []}
    return {"district_names": [row[0] for row in result]}

# Endpoint to get client birth dates based on loan date and amount
@app.get("/v1/financial/client_birth_dates_by_loan_date_amount", operation_id="get_client_birth_dates_by_loan_date_amount", summary="Retrieves the birth dates of clients who have taken a loan on a specific date and for a specific amount. The operation filters the loan data based on the provided loan date and amount, and returns the birth dates of the corresponding clients.")
async def get_client_birth_dates_by_loan_date_amount(loan_date: str = Query(..., description="Loan date in 'YYYY-MM-DD' format"), amount: int = Query(..., description="Loan amount")):
    cursor.execute("SELECT T4.birth_date FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN disp AS T3 ON T2.account_id = T3.account_id INNER JOIN client AS T4 ON T3.client_id = T4.client_id WHERE T1.date = ? AND T1.amount = ?", (loan_date, amount))
    result = cursor.fetchall()
    if not result:
        return {"birth_dates": []}
    return {"birth_dates": [row[0] for row in result]}

# Endpoint to get the earliest account ID based on district
@app.get("/v1/financial/earliest_account_id_by_district", operation_id="get_earliest_account_id_by_district", summary="Retrieves the account ID with the earliest date from the specified district. The district is identified by its name.")
async def get_earliest_account_id_by_district(district: str = Query(..., description="District name")):
    cursor.execute("SELECT T1.account_id FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.A3 = ? ORDER BY T1.date ASC LIMIT 1", (district,))
    result = cursor.fetchone()
    if not result:
        return {"account_id": []}
    return {"account_id": result[0]}

# Endpoint to get the percentage of male clients in a district
@app.get("/v1/financial/percentage_male_clients_by_district", operation_id="get_percentage_male_clients_by_district", summary="Retrieves the percentage of male clients in a specified district. The district is identified by its name, and the result is calculated by summing the number of male clients and dividing it by the total number of clients in the district. The percentage is then multiplied by 100 for a more intuitive representation. The result is sorted in descending order by district name and limited to the top entry.")
async def get_percentage_male_clients_by_district(district: str = Query(..., description="District name")):
    cursor.execute("SELECT CAST(SUM(T1.gender = 'M') AS REAL) * 100 / COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.A3 = ? GROUP BY T2.A4 ORDER BY T2.A4 DESC LIMIT 1", (district,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage change in balance between two dates for loans on a specific date
@app.get("/v1/financial/percentage_change_balance_by_dates", operation_id="get_percentage_change_balance_by_dates", summary="Retrieves the percentage change in balance between two specified dates for loans issued on a particular date. The calculation considers the total balance of all accounts associated with the loans on each date.")
async def get_percentage_change_balance_by_dates(date1: str = Query(..., description="First date in 'YYYY-MM-DD' format"), date2: str = Query(..., description="Second date in 'YYYY-MM-DD' format"), loan_date: str = Query(..., description="Loan date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST((SUM(IIF(T3.date = ?, T3.balance, 0)) - SUM(IIF(T3.date = ?, T3.balance, 0))) AS REAL) * 100 / SUM(IIF(T3.date = ?, T3.balance, 0)) FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN trans AS T3 ON T3.account_id = T2.account_id WHERE T1.date = ?", (date1, date2, date2, loan_date))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result[0]}

# Endpoint to get the percentage of loan amount with status 'A'
@app.get("/v1/financial/percentage_loan_amount_status_a", operation_id="get_percentage_loan_amount_status_a", summary="Retrieves the percentage of total loan amount that has a status of 'A'. This operation calculates the sum of loan amounts with status 'A' and divides it by the total sum of all loan amounts. The result is then multiplied by 100 to obtain the percentage.")
async def get_percentage_loan_amount_status_a():
    cursor.execute("SELECT (CAST(SUM(CASE WHEN status = 'A' THEN amount ELSE 0 END) AS REAL) * 100) / SUM(amount) FROM loan")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of loans with status 'C' and amount less than a specified value
@app.get("/v1/financial/percentage_loans_status_c_below_amount", operation_id="get_percentage_loans_status_c_below_amount", summary="Retrieves the percentage of loans with a status of 'C' and an amount less than the specified maximum value. This operation calculates the proportion of loans that meet the given criteria, providing a useful metric for financial analysis.")
async def get_percentage_loans_status_c_below_amount(amount: int = Query(..., description="Maximum loan amount")):
    cursor.execute("SELECT CAST(SUM(status = 'C') AS REAL) * 100 / COUNT(account_id) FROM loan WHERE amount < ?", (amount,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get account details based on frequency and year
@app.get("/v1/financial/account_details_by_frequency_year", operation_id="get_account_details_by_frequency_year", summary="Retrieves account details, including account ID, district code, and district name, based on the specified frequency and year. The frequency parameter determines the periodicity of the account, while the year parameter filters the results to a specific year.")
async def get_account_details_by_frequency_year(frequency: str = Query(..., description="Frequency of the account"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T1.account_id, T2.A2, T2.A3 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.frequency = ? AND STRFTIME('%Y', T1.date) = ?", (frequency, year))
    result = cursor.fetchall()
    if not result:
        return {"accounts": []}
    return {"accounts": result}

# Endpoint to get account details based on district and year range
@app.get("/v1/financial/account_details_by_district_year_range", operation_id="get_account_details_by_district_year_range", summary="Retrieves account details, including account ID and frequency, for a specific district within a given year range. The district is identified by its name, and the year range is defined by the start and end years in 'YYYY' format.")
async def get_account_details_by_district_year_range(district: str = Query(..., description="District name"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT T1.account_id, T1.frequency FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.A3 = ? AND STRFTIME('%Y', T1.date) BETWEEN ? AND ?", (district, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"accounts": []}
    return {"accounts": result}

# Endpoint to get account details based on district name
@app.get("/v1/financial/account_details_by_district_name", operation_id="get_account_details_by_district_name", summary="Retrieves account details, including account ID and date, for a specific district. The district is identified by its name, which is provided as an input parameter.")
async def get_account_details_by_district_name(district_name: str = Query(..., description="District name")):
    cursor.execute("SELECT T1.account_id, T1.date FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T2.A2 = ?", (district_name,))
    result = cursor.fetchall()
    if not result:
        return {"accounts": []}
    return {"accounts": result}

# Endpoint to get district details based on loan ID
@app.get("/v1/financial/district_details_by_loan_id", operation_id="get_district_details_by_loan_id", summary="Retrieves district details associated with a specific loan ID. This operation fetches the district's name and code from the database by joining the account, district, and loan tables using the provided loan ID. The result is a set of district details linked to the given loan.")
async def get_district_details_by_loan_id(loan_id: int = Query(..., description="Loan ID")):
    cursor.execute("SELECT T2.A2, T2.A3 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN loan AS T3 ON T1.account_id = T3.account_id WHERE T3.loan_id = ?", (loan_id,))
    result = cursor.fetchall()
    if not result:
        return {"districts": []}
    return {"districts": result}

# Endpoint to get account details based on loan amount
@app.get("/v1/financial/account_details_by_loan_amount", operation_id="get_account_details_by_loan_amount", summary="Retrieves account details for accounts with a loan amount greater than the specified value. The response includes the account ID, district code, and district name.")
async def get_account_details_by_loan_amount(amount: int = Query(..., description="Loan amount")):
    cursor.execute("SELECT T1.account_id, T2.A2, T2.A3 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN loan AS T3 ON T1.account_id = T3.account_id WHERE T3.amount > ?", (amount,))
    result = cursor.fetchall()
    if not result:
        return {"accounts": []}
    return {"accounts": result}

# Endpoint to get loan details based on duration
@app.get("/v1/financial/loan_details_by_duration", operation_id="get_loan_details_by_duration", summary="Retrieves detailed information about loans based on their duration. The operation fetches the loan ID, district code, and district name for loans with the specified duration. The duration is provided as an input parameter.")
async def get_loan_details_by_duration(duration: int = Query(..., description="Loan duration")):
    cursor.execute("SELECT T3.loan_id, T2.A2, T2.A11 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN loan AS T3 ON T1.account_id = T3.account_id WHERE T3.duration = ?", (duration,))
    result = cursor.fetchall()
    if not result:
        return {"loans": []}
    return {"loans": result}

# Endpoint to get loan status percentage based on status
@app.get("/v1/financial/loan_status_percentage", operation_id="get_loan_status_percentage", summary="Retrieves the percentage of loans with the specified status in each district. The calculation is based on the difference between the total number of loans in a district and the number of loans with the given status, divided by the total number of loans in the district. The result is then multiplied by 100 to obtain a percentage.")
async def get_loan_status_percentage(status: str = Query(..., description="Loan status")):
    cursor.execute("SELECT CAST((T3.A13 - T3.A12) AS REAL) * 100 / T3.A12 FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN district AS T3 ON T2.district_id = T3.district_id WHERE T1.status = ?", (status,))
    result = cursor.fetchall()
    if not result:
        return {"percentages": []}
    return {"percentages": result}

# Endpoint to get percentage of accounts in a specific district for a given year
@app.get("/v1/financial/account_percentage_by_district_year", operation_id="get_account_percentage_by_district_year", summary="Retrieves the percentage of accounts in a specific district for a given year. The operation calculates this percentage by comparing the number of accounts in the specified district and year to the total number of accounts in the district. The district is identified by its name, and the year is provided in the 'YYYY' format.")
async def get_account_percentage_by_district_year(district: str = Query(..., description="District name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(T1.A2 = ?) AS REAL) * 100 / COUNT(account_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE STRFTIME('%Y', T2.date) = ?", (district, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get account IDs based on frequency
@app.get("/v1/financial/account_ids_by_frequency", operation_id="get_account_ids_by_frequency", summary="Retrieves the unique identifiers of accounts that match the specified frequency. The frequency parameter is used to filter the accounts and return only those that meet the given criteria.")
async def get_account_ids_by_frequency(frequency: str = Query(..., description="Frequency of the account")):
    cursor.execute("SELECT account_id FROM account WHERE frequency = ?", (frequency,))
    result = cursor.fetchall()
    if not result:
        return {"account_ids": []}
    return {"account_ids": result}

# Endpoint to get the count of clients by district and gender
@app.get("/v1/financial/client_count_by_district_gender", operation_id="get_client_count_by_district_gender", summary="Retrieves the number of clients in each district, categorized by gender. The response is sorted in descending order based on the client count. The number of results can be limited using the 'limit' parameter.")
async def get_client_count_by_district_gender(gender: str = Query(..., description="Gender of the client"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.A2, COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = ? GROUP BY T2.district_id, T2.A2 ORDER BY COUNT(T1.client_id) DESC LIMIT ?", (gender, limit))
    result = cursor.fetchall()
    if not result:
        return {"client_counts": []}
    return {"client_counts": result}

# Endpoint to get distinct A2 values from district based on transaction type and date pattern
@app.get("/v1/financial/distinct_a2_values", operation_id="get_distinct_a2_values", summary="Retrieves the first 10 distinct A2 values from the district, filtered by a specific transaction type and date pattern. The transaction type and date pattern are used to narrow down the results, ensuring that only relevant A2 values are returned. The A2 values are ordered in ascending order.")
async def get_distinct_a2_values(transaction_type: str = Query(..., description="Type of transaction"), date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM%' format")):
    cursor.execute("SELECT DISTINCT T1.A2 FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN trans AS T3 ON T2.account_id = T3.account_id WHERE T3.type = ? AND T3.date LIKE ? ORDER BY A2 ASC LIMIT 10", (transaction_type, date_pattern))
    result = cursor.fetchall()
    if not result:
        return {"distinct_a2_values": []}
    return {"distinct_a2_values": [row[0] for row in result]}

# Endpoint to get the count of account IDs based on district A3 and disposition type
@app.get("/v1/financial/count_account_ids_by_district_a3_and_type", operation_id="get_count_account_ids_by_district_a3_and_type", summary="Retrieves the total number of account IDs associated with a specific district (A3) and excluding a certain disposition type. This operation is useful for understanding the distribution of accounts across districts, excluding a particular disposition type.")
async def get_count_account_ids_by_district_a3_and_type(district_a3: str = Query(..., description="District A3 value"), disposition_type: str = Query(..., description="Disposition type to exclude")):
    cursor.execute("SELECT COUNT(T3.account_id) FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T3 ON T2.client_id = T3.client_id WHERE T1.A3 = ? AND T3.type != ?", (district_a3, disposition_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the district A3 with the highest total loan amount based on loan status
@app.get("/v1/financial/top_district_by_loan_amount", operation_id="get_top_district_by_loan_amount", summary="Retrieves the district with the highest total loan amount, considering loans with the specified statuses. The operation filters loans based on the provided statuses, aggregates the loan amounts by district, and returns the district with the highest sum.")
async def get_top_district_by_loan_amount(status1: str = Query(..., description="First loan status"), status2: str = Query(..., description="Second loan status")):
    cursor.execute("SELECT T2.A3 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN loan AS T3 ON T1.account_id = T3.account_id WHERE T3.status IN (?, ?) GROUP BY T2.A3 ORDER BY SUM(T3.amount) DESC LIMIT 1", (status1, status2))
    result = cursor.fetchone()
    if not result:
        return {"top_district": []}
    return {"top_district": result[0]}

# Endpoint to get the average loan amount based on client gender
@app.get("/v1/financial/average_loan_amount_by_gender", operation_id="get_average_loan_amount_by_gender", summary="Retrieves the average loan amount for clients of a specific gender. The calculation is based on the loan amounts associated with the client's accounts. The gender parameter is used to filter the results.")
async def get_average_loan_amount_by_gender(gender: str = Query(..., description="Gender of the client")):
    cursor.execute("SELECT AVG(T4.amount) FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN account AS T3 ON T2.account_id = T3.account_id INNER JOIN loan AS T4 ON T3.account_id = T4.account_id WHERE T1.gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"average_loan_amount": []}
    return {"average_loan_amount": result[0]}

# Endpoint to get the top district based on A13 value
@app.get("/v1/financial/top_district_by_a13", operation_id="get_top_district_by_a13", summary="Retrieves the district with the highest A13 value. The response includes the district's ID and A2 value.")
async def get_top_district_by_a13():
    cursor.execute("SELECT district_id, A2 FROM district ORDER BY A13 DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"top_district": []}
    return {"top_district": {"district_id": result[0], "A2": result[1]}}

# Endpoint to get the count of account IDs grouped by district A16
@app.get("/v1/financial/count_account_ids_by_district_a16", operation_id="get_count_account_ids_by_district_a16", summary="Retrieves the total number of unique account IDs categorized by district A16. The data is sorted in descending order, with the district A16 value of the last record being the highest. Only the topmost district A16 is returned.")
async def get_count_account_ids_by_district_a16():
    cursor.execute("SELECT COUNT(T2.account_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id GROUP BY T1.A16 ORDER BY T1.A16 DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of account IDs based on transaction balance, operation, and account frequency
@app.get("/v1/financial/count_account_ids_by_transaction_criteria", operation_id="get_count_account_ids_by_transaction_criteria", summary="Retrieves the count of unique account IDs that meet the specified transaction balance, operation type, and account frequency criteria. The operation filters transactions based on the provided balance and operation type, and further narrows down the results by considering the frequency of the associated accounts.")
async def get_count_account_ids_by_transaction_criteria(balance: float = Query(..., description="Transaction balance"), operation: str = Query(..., description="Operation type"), frequency: str = Query(..., description="Account frequency")):
    cursor.execute("SELECT COUNT(T1.account_id) FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE T1.balance < ? AND T1.operation = ? AND T2.frequency = ?", (balance, operation, frequency))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of account IDs based on loan date range, frequency, and amount
@app.get("/v1/financial/count_account_ids_by_loan_criteria", operation_id="get_count_account_ids_by_loan_criteria", summary="Retrieves the total number of account IDs that meet the specified loan criteria, including date range, account frequency, and minimum loan amount.")
async def get_count_account_ids_by_loan_criteria(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), frequency: str = Query(..., description="Account frequency"), amount: float = Query(..., description="Loan amount")):
    cursor.execute("SELECT COUNT(T1.account_id) FROM account AS T1 INNER JOIN loan AS T2 ON T1.account_id = T2.account_id WHERE T2.date BETWEEN ? AND ? AND T1.frequency = ? AND T2.amount >= ?", (start_date, end_date, frequency, amount))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of account IDs based on district ID and loan status
@app.get("/v1/financial/count_account_ids_by_district_and_loan_status", operation_id="get_count_account_ids_by_district_and_loan_status", summary="Retrieves the total number of account IDs associated with a specific district and having one of two specified loan statuses. This operation is useful for understanding the distribution of accounts based on district and loan status.")
async def get_count_account_ids_by_district_and_loan_status(district_id: int = Query(..., description="District ID"), status1: str = Query(..., description="First loan status"), status2: str = Query(..., description="Second loan status")):
    cursor.execute("SELECT COUNT(T1.account_id) FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN loan AS T3 ON T1.account_id = T3.account_id WHERE T1.district_id = ? AND (T3.status = ? OR T3.status = ?)", (district_id, status1, status2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of client IDs based on gender and district A15
@app.get("/v1/financial/count_client_ids_by_gender_and_district_a15", operation_id="get_count_client_ids_by_gender_and_district_a15", summary="Retrieves the total number of unique client IDs that match the specified gender and belong to the second most recent district A15. The district is determined by the descending order of A15 values.")
async def get_count_client_ids_by_gender_and_district_a15(gender: str = Query(..., description="Gender of the client")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = ? AND T2.A15 = (SELECT T3.A15 FROM district AS T3 ORDER BY T3.A15 DESC LIMIT 1, 1)", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of card IDs based on card type and disp type
@app.get("/v1/financial/count_card_ids_by_type", operation_id="get_count_card_ids_by_type", summary="Retrieves the total count of card IDs that match the specified card type and disp type. This operation is useful for understanding the distribution of card IDs across different card and disp types.")
async def get_count_card_ids_by_type(card_type: str = Query(..., description="Type of the card"), disp_type: str = Query(..., description="Type of the disp")):
    cursor.execute("SELECT COUNT(T1.card_id) FROM card AS T1 INNER JOIN disp AS T2 ON T1.disp_id = T2.disp_id WHERE T1.type = ? AND T2.type = ?", (card_type, disp_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get district IDs based on year and minimum total transaction amount
@app.get("/v1/financial/district_ids_by_year_and_amount", operation_id="get_district_ids_by_year_and_amount", summary="Retrieves the IDs of districts that have a total transaction amount exceeding a specified minimum, within a given year. The year is provided in 'YYYY' format, and the minimum total transaction amount is a numerical value.")
async def get_district_ids_by_year_and_amount(year: str = Query(..., description="Year in 'YYYY' format"), min_amount: int = Query(..., description="Minimum total transaction amount")):
    cursor.execute("SELECT T1.district_id FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN trans AS T3 ON T1.account_id = T3.account_id WHERE STRFTIME('%Y', T3.date) = ? GROUP BY T1.district_id HAVING SUM(T3.amount) > ?", (year, min_amount))
    result = cursor.fetchall()
    if not result:
        return {"district_ids": []}
    return {"district_ids": [row[0] for row in result]}

# Endpoint to get distinct account IDs based on transaction symbol and district A2
@app.get("/v1/financial/distinct_account_ids_by_symbol_and_district", operation_id="get_distinct_account_ids_by_symbol_and_district", summary="Retrieves a unique set of account IDs associated with a specific transaction symbol and district. The operation filters transactions by the provided symbol and district, and returns the distinct account IDs linked to these transactions.")
async def get_distinct_account_ids_by_symbol_and_district(k_symbol: str = Query(..., description="Transaction symbol"), district_a2: str = Query(..., description="District A2")):
    cursor.execute("SELECT DISTINCT T2.account_id FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN district AS T3 ON T2.district_id = T3.district_id WHERE T1.k_symbol = ? AND T3.A2 = ?", (k_symbol, district_a2))
    result = cursor.fetchall()
    if not result:
        return {"account_ids": []}
    return {"account_ids": [row[0] for row in result]}

# Endpoint to get account IDs based on card type
@app.get("/v1/financial/account_ids_by_card_type", operation_id="get_account_ids_by_card_type", summary="Retrieves a list of account IDs associated with a specific card type. The card type is used to filter the results, providing a targeted set of account IDs that match the specified card type.")
async def get_account_ids_by_card_type(card_type: str = Query(..., description="Type of the card")):
    cursor.execute("SELECT T2.account_id FROM disp AS T2 INNER JOIN card AS T1 ON T1.disp_id = T2.disp_id WHERE T1.type = ?", (card_type,))
    result = cursor.fetchall()
    if not result:
        return {"account_ids": []}
    return {"account_ids": [row[0] for row in result]}

# Endpoint to get the average transaction amount based on year and operation type
@app.get("/v1/financial/avg_transaction_amount_by_year_and_operation", operation_id="get_avg_transaction_amount_by_year_and_operation", summary="Retrieves the average transaction amount for a specific year and operation type. The operation type can be any valid transaction operation. The year should be provided in 'YYYY' format. The result is calculated by aggregating transaction amounts from the associated accounts and dispensers.")
async def get_avg_transaction_amount_by_year_and_operation(year: str = Query(..., description="Year in 'YYYY' format"), operation: str = Query(..., description="Operation type")):
    cursor.execute("SELECT AVG(T4.amount) FROM card AS T1 INNER JOIN disp AS T2 ON T1.disp_id = T2.disp_id INNER JOIN account AS T3 ON T2.account_id = T3.account_id INNER JOIN trans AS T4 ON T3.account_id = T4.account_id WHERE STRFTIME('%Y', T4.date) = ? AND T4.operation = ?", (year, operation))
    result = cursor.fetchone()
    if not result:
        return {"average_amount": []}
    return {"average_amount": result[0]}

# Endpoint to get account IDs based on year, operation type, and amount less than the average
@app.get("/v1/financial/account_ids_by_year_operation_and_amount", operation_id="get_account_ids_by_year_operation_and_amount", summary="Retrieves account IDs that match the specified year, operation type, and have an amount less than the average for that year. The operation type is used to filter transactions based on their type (e.g., deposit, withdrawal).")
async def get_account_ids_by_year_operation_and_amount(year: str = Query(..., description="Year in 'YYYY' format"), operation: str = Query(..., description="Operation type")):
    cursor.execute("SELECT T1.account_id FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE STRFTIME('%Y', T1.date) = ? AND T1.operation = ? AND T1.amount < (SELECT AVG(amount) FROM trans WHERE STRFTIME('%Y', date) = ?)", (year, operation, year))
    result = cursor.fetchall()
    if not result:
        return {"account_ids": []}
    return {"account_ids": [row[0] for row in result]}

# Endpoint to get client IDs based on gender
@app.get("/v1/financial/client_ids_by_gender", operation_id="get_client_ids_by_gender", summary="Retrieves a list of client IDs who have a specific gender. The operation considers clients who have an account and a loan, and who have a card associated with their account. The gender parameter is used to filter the results.")
async def get_client_ids_by_gender(gender: str = Query(..., description="Gender of the client")):
    cursor.execute("SELECT T1.client_id FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN account AS T5 ON T2.account_id = T5.account_id INNER JOIN loan AS T3 ON T5.account_id = T3.account_id INNER JOIN card AS T4 ON T2.disp_id = T4.disp_id WHERE T1.gender = ?", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"client_ids": []}
    return {"client_ids": [row[0] for row in result]}

# Endpoint to get the count of client IDs based on gender and district A3
@app.get("/v1/financial/count_client_ids_by_gender_and_district", operation_id="get_count_client_ids_by_gender_and_district", summary="Retrieves the total number of client IDs categorized by gender and district A3. The operation requires the gender and district A3 as input parameters to filter the data.")
async def get_count_client_ids_by_gender_and_district(gender: str = Query(..., description="Gender of the client"), district_a3: str = Query(..., description="District A3")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE T1.gender = ? AND T2.A3 = ?", (gender, district_a3))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get account IDs based on disp type and district A2
@app.get("/v1/financial/account_ids_by_disp_type_and_district", operation_id="get_account_ids_by_disp_type_and_district", summary="Retrieves a list of account IDs that match the specified disp type and district A2. This operation filters accounts based on the provided disp type and district A2, and returns the corresponding account IDs.")
async def get_account_ids_by_disp_type_and_district(disp_type: str = Query(..., description="Type of the disp"), district_a2: str = Query(..., description="District A2")):
    cursor.execute("SELECT T2.account_id FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T3 ON T2.account_id = T3.account_id WHERE T3.type = ? AND T1.A2 = ?", (disp_type, district_a2))
    result = cursor.fetchall()
    if not result:
        return {"account_ids": []}
    return {"account_ids": [row[0] for row in result]}

# Endpoint to get the type of dispensations based on district A11 range and excluding a specific type
@app.get("/v1/financial/disp_types_by_district_range", operation_id="get_disp_types_by_district_range", summary="Retrieves the types of dispensations, excluding a specified type, for accounts in districts with A11 values within a given range. The range is defined by a minimum and maximum A11 value.")
async def get_disp_types_by_district_range(excluded_type: str = Query(..., description="Type to exclude"), min_a11: int = Query(..., description="Minimum A11 value"), max_a11: int = Query(..., description="Maximum A11 value")):
    cursor.execute("SELECT T3.type FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T3 ON T2.account_id = T3.account_id WHERE T3.type != ? AND T1.A11 BETWEEN ? AND ?", (excluded_type, min_a11, max_a11))
    result = cursor.fetchall()
    if not result:
        return {"types": []}
    return {"types": [row[0] for row in result]}

# Endpoint to get the count of accounts based on bank and district A3
@app.get("/v1/financial/account_count_by_bank_and_district", operation_id="get_account_count_by_bank_and_district", summary="Retrieves the total number of accounts associated with a specific bank and district, based on the provided bank name and district A3 value. The operation performs a search across the account and transaction data, filtering results by the given bank and district A3 value.")
async def get_account_count_by_bank_and_district(bank: str = Query(..., description="Bank name"), a3: str = Query(..., description="District A3 value")):
    cursor.execute("SELECT COUNT(T2.account_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN trans AS T3 ON T2.account_id = T3.account_id WHERE T3.bank = ? AND T1.A3 = ?", (bank, a3))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct district A2 values based on transaction type
@app.get("/v1/financial/distinct_district_a2_by_trans_type", operation_id="get_distinct_district_a2_by_trans_type", summary="Retrieves unique district identifiers (A2) associated with a specified transaction type. This operation filters the district table based on a given transaction type, joining it with the account and trans tables to ensure accurate results.")
async def get_distinct_district_a2_by_trans_type(trans_type: str = Query(..., description="Transaction type")):
    cursor.execute("SELECT DISTINCT T1.A2 FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN trans AS T3 ON T2.account_id = T3.account_id WHERE T3.type = ?", (trans_type,))
    result = cursor.fetchall()
    if not result:
        return {"a2_values": []}
    return {"a2_values": [row[0] for row in result]}

# Endpoint to get the average of district A15 values based on year and minimum A15 value
@app.get("/v1/financial/avg_district_a15_by_year_and_min_value", operation_id="get_avg_district_a15_by_year_and_min_value", summary="Retrieves the average value of district A15 for a given year and minimum A15 value. The operation filters accounts based on the specified year and minimum A15 value, then calculates the average A15 value for the corresponding districts.")
async def get_avg_district_a15_by_year_and_min_value(year: str = Query(..., description="Year in 'YYYY' format"), min_a15: int = Query(..., description="Minimum A15 value")):
    cursor.execute("SELECT AVG(T1.A15) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE STRFTIME('%Y', T2.date) >= ? AND T1.A15 > ?", (year, min_a15))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of cards of a specific type issued before a certain year
@app.get("/v1/financial/percentage_cards_by_type_and_year", operation_id="get_percentage_cards_by_type_and_year", summary="Retrieves the percentage of cards of a specified type issued before a given year. The calculation is based on the total count of cards in the database. The card type and year are provided as input parameters.")
async def get_percentage_cards_by_type_and_year(card_type: str = Query(..., description="Card type"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(type = ? AND STRFTIME('%Y', issued) < ?) AS REAL) * 100 / COUNT(card_id) FROM card", (card_type, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the client ID with the highest loan amount for a specific dispensation type
@app.get("/v1/financial/client_id_highest_loan_by_disp_type", operation_id="get_client_id_highest_loan_by_disp_type", summary="Retrieves the client ID associated with the highest loan amount for a given dispensation type. This operation joins data from the disp, account, and loan tables to identify the client with the largest loan amount matching the specified dispensation type.")
async def get_client_id_highest_loan_by_disp_type(disp_type: str = Query(..., description="Dispensation type")):
    cursor.execute("SELECT T1.client_id FROM disp AS T1 INNER JOIN account AS T3 ON T1.account_id = T3.account_id INNER JOIN loan AS T2 ON T3.account_id = T2.account_id WHERE T1.type = ? ORDER BY T2.amount DESC LIMIT 1", (disp_type,))
    result = cursor.fetchone()
    if not result:
        return {"client_id": []}
    return {"client_id": result[0]}

# Endpoint to get the district A15 value based on account ID
@app.get("/v1/financial/district_a15_by_account_id", operation_id="get_district_a15_by_account_id", summary="Retrieves the A15 value for a specific district associated with the provided account ID. This operation fetches the A15 value from the district table by joining it with the account table using the district_id. The account_id is used to filter the results.")
async def get_district_a15_by_account_id(account_id: int = Query(..., description="Account ID")):
    cursor.execute("SELECT T1.A15 FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE T2.account_id = ?", (account_id,))
    result = cursor.fetchone()
    if not result:
        return {"a15_value": []}
    return {"a15_value": result[0]}

# Endpoint to get the district ID based on order ID
@app.get("/v1/financial/district_id_by_order_id", operation_id="get_district_id_by_order_id", summary="Retrieves the district ID associated with a given order ID. This operation fetches the district ID by joining the 'order', 'account', and 'district' tables using the provided order ID. The result is the district ID linked to the specified order.")
async def get_district_id_by_order_id(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T3.district_id FROM `order` AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN district AS T3 ON T2.district_id = T3.district_id WHERE T1.order_id = ?", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"district_id": []}
    return {"district_id": result[0]}

# Endpoint to get transaction IDs for a specific client and operation
@app.get("/v1/financial/trans_ids_by_client_and_operation", operation_id="get_trans_ids_by_client_and_operation", summary="Retrieves a list of transaction IDs associated with a specific client and operation type. This endpoint fetches the transaction IDs by joining the client, disp, account, and trans tables using the provided client ID and operation type.")
async def get_trans_ids_by_client_and_operation(client_id: int = Query(..., description="Client ID"), operation: str = Query(..., description="Operation type")):
    cursor.execute("SELECT T4.trans_id FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN account AS T3 ON T2.account_id = T3.account_id INNER JOIN trans AS T4 ON T3.account_id = T4.account_id WHERE T1.client_id = ? AND T4.operation = ?", (client_id, operation))
    result = cursor.fetchall()
    if not result:
        return {"trans_ids": []}
    return {"trans_ids": [row[0] for row in result]}

# Endpoint to get the count of loan accounts with a specific frequency and amount less than a specified value
@app.get("/v1/financial/loan_account_count_by_frequency_and_amount", operation_id="get_loan_account_count", summary="Retrieves the total number of loan accounts that have a specified frequency and an amount less than a given maximum value. The frequency type and maximum amount are provided as input parameters.")
async def get_loan_account_count(frequency: str = Query(..., description="Frequency type"), max_amount: int = Query(..., description="Maximum amount")):
    cursor.execute("SELECT COUNT(T1.account_id) FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE T2.frequency = ? AND T1.amount < ?", (frequency, max_amount))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get card types for a specific client
@app.get("/v1/financial/card_types_by_client", operation_id="get_card_types_by_client", summary="Retrieves the types of cards associated with a specific client. The client is identified by the provided client_id.")
async def get_card_types_by_client(client_id: int = Query(..., description="Client ID")):
    cursor.execute("SELECT T3.type FROM disp AS T1 INNER JOIN client AS T2 ON T1.client_id = T2.client_id INNER JOIN card AS T3 ON T1.disp_id = T3.disp_id WHERE T2.client_id = ?", (client_id,))
    result = cursor.fetchall()
    if not result:
        return {"card_types": []}
    return {"card_types": [row[0] for row in result]}

# Endpoint to get district A3 values for a specific client
@app.get("/v1/financial/district_a3_by_client", operation_id="get_district_a3_by_client", summary="Retrieves the district A3 values associated with a specific client. The client is identified by its unique ID, which is used to look up corresponding district A3 values from the database.")
async def get_district_a3_by_client(client_id: int = Query(..., description="Client ID")):
    cursor.execute("SELECT T1.A3 FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T2.client_id = ?", (client_id,))
    result = cursor.fetchall()
    if not result:
        return {"district_a3": []}
    return {"district_a3": [row[0] for row in result]}

# Endpoint to get the district A2 value with the highest number of loan accounts for a specific status
@app.get("/v1/financial/top_district_a2_by_loan_status", operation_id="get_top_district_a2_by_loan_status", summary="Retrieves the district with the highest number of loan accounts for a given loan status. The district is identified by its A2 value. The loan status is specified as an input parameter.")
async def get_top_district_a2_by_loan_status(status: str = Query(..., description="Loan status")):
    cursor.execute("SELECT T1.A2 FROM District AS T1 INNER JOIN Account AS T2 ON T1.District_id = T2.District_id INNER JOIN Loan AS T3 ON T2.Account_id = T3.Account_id WHERE T3.status = ? GROUP BY T1.District_id ORDER BY COUNT(T2.Account_id) DESC LIMIT 1", (status,))
    result = cursor.fetchone()
    if not result:
        return {"district_a2": []}
    return {"district_a2": result[0]}

# Endpoint to get client IDs for a specific order
@app.get("/v1/financial/client_ids_by_order", operation_id="get_client_ids_by_order", summary="Retrieves the client IDs associated with a specific order. The operation identifies the client IDs by joining the order, account, disp, and client tables using the provided order ID. The result is a list of client IDs linked to the given order.")
async def get_client_ids_by_order(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T3.client_id FROM `order` AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN disp AS T4 ON T4.account_id = T2.account_id INNER JOIN client AS T3 ON T4.client_id = T3.client_id WHERE T1.order_id = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"client_ids": []}
    return {"client_ids": [row[0] for row in result]}

# Endpoint to get transaction IDs for a specific district
@app.get("/v1/financial/trans_ids_by_district", operation_id="get_trans_ids_by_district", summary="Retrieves a list of transaction IDs associated with a specific district. The operation uses the provided district ID to look up corresponding accounts and their respective transactions.")
async def get_trans_ids_by_district(district_id: int = Query(..., description="District ID")):
    cursor.execute("SELECT T3.trans_id FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id INNER JOIN trans AS T3 ON T2.account_id = T3.account_id WHERE T1.district_id = ?", (district_id,))
    result = cursor.fetchall()
    if not result:
        return {"trans_ids": []}
    return {"trans_ids": [row[0] for row in result]}

# Endpoint to get the count of accounts in a specific district
@app.get("/v1/financial/account_count_by_district", operation_id="get_account_count_by_district", summary="Retrieves the total number of accounts associated with a specific district, identified by its A2 value.")
async def get_account_count_by_district(a2: str = Query(..., description="District A2 value")):
    cursor.execute("SELECT COUNT(T2.account_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE T1.A2 = ?", (a2,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get client IDs for cards of a specific type issued on or after a given date
@app.get("/v1/financial/client_ids_by_card_type_and_issue_date", operation_id="get_client_ids_by_card_type_and_issue_date", summary="Retrieve the client IDs associated with cards of a specified type issued on or after a given date. The input parameters include the card type and the issue date in 'YYYY-MM-DD' format. This operation returns a list of client IDs that meet the specified criteria.")
async def get_client_ids_by_card_type_and_issue_date(card_type: str = Query(..., description="Card type"), issued_date: str = Query(..., description="Issued date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.client_id FROM card AS T1 INNER JOIN disp AS T2 ON T1.disp_id = T2.disp_id WHERE T1.type = ? AND T1.issued >= ?", (card_type, issued_date))
    result = cursor.fetchall()
    if not result:
        return {"client_ids": []}
    return {"client_ids": [row[0] for row in result]}

# Endpoint to get the percentage of female clients in districts with a specific A11 value greater than a given threshold
@app.get("/v1/financial/female_client_percentage_by_district_a11", operation_id="get_female_client_percentage_by_district_a11", summary="Retrieve the proportion of female clients in districts where the A11 value surpasses a specified threshold. The calculation is based on the total number of clients in these districts.")
async def get_female_client_percentage_by_district_a11(a11_threshold: int = Query(..., description="A11 threshold value")):
    cursor.execute("SELECT CAST(SUM(T2.gender = 'F') AS REAL) * 100 / COUNT(T2.client_id) FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T1.A11 > ?", (a11_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage change in loan amounts between two years for a specific gender and type
@app.get("/v1/financial/loan_amount_change_percentage", operation_id="get_loan_amount_change_percentage", summary="Retrieves the percentage change in total loan amounts between two specified years for a particular gender and disp type. The calculation is based on the sum of loan amounts for the specified years and gender, with the percentage change computed against the first year provided.")
async def get_loan_amount_change_percentage(year1: str = Query(..., description="First year in 'YYYY' format"), year2: str = Query(..., description="Second year in 'YYYY' format"), year3: str = Query(..., description="Third year in 'YYYY' format"), gender: str = Query(..., description="Gender of the client"), type: str = Query(..., description="Type of the disp")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN STRFTIME('%Y', T1.date) = ? THEN T1.amount ELSE 0 END) - SUM(CASE WHEN STRFTIME('%Y', T1.date) = ? THEN T1.amount ELSE 0 END)) AS REAL) * 100 / SUM(CASE WHEN STRFTIME('%Y', T1.date) = ? THEN T1.amount ELSE 0 END) FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN disp AS T3 ON T3.account_id = T2.account_id INNER JOIN client AS T4 ON T4.client_id = T3.client_id WHERE T4.gender = ? AND T3.type = ?", (year1, year2, year3, gender, type))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result[0]}

# Endpoint to get the count of transactions after a specific year and with a specific operation
@app.get("/v1/financial/transaction_count_after_year", operation_id="get_transaction_count_after_year", summary="Retrieves the total number of transactions that occurred after a specified year and for a particular operation type. The year is provided in 'YYYY' format, and the operation type is a specific transaction category.")
async def get_transaction_count_after_year(year: str = Query(..., description="Year in 'YYYY' format"), operation: str = Query(..., description="Operation type")):
    cursor.execute("SELECT COUNT(account_id) FROM trans WHERE STRFTIME('%Y', date) > ? AND operation = ?", (year, operation))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in sums of a specific column based on two regions
@app.get("/v1/financial/sum_difference_by_region", operation_id="get_sum_difference_by_region", summary="Retrieves the difference in the sum of a specific financial value between two regions. The operation compares the total of the specified value for the first region with that of the second region, and returns the difference.")
async def get_sum_difference_by_region(region1: str = Query(..., description="First region"), region2: str = Query(..., description="Second region")):
    cursor.execute("SELECT SUM(IIF(A3 = ?, A16, 0)) - SUM(IIF(A3 = ?, A16, 0)) FROM district", (region1, region2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the sum of specific types within a range of account IDs
@app.get("/v1/financial/sum_types_by_account_range", operation_id="get_sum_types_by_account_range", summary="Get the sum of specific types within a range of account IDs")
async def get_sum_types_by_account_range(type1: str = Query(..., description="First type"), type2: str = Query(..., description="Second type"), min_account_id: int = Query(..., description="Minimum account ID"), max_account_id: int = Query(..., description="Maximum account ID")):
    cursor.execute("SELECT SUM(type = ?) , SUM(type = ?) FROM disp WHERE account_id BETWEEN ? AND ?", (type1, type2, min_account_id, max_account_id))
    result = cursor.fetchone()
    if not result:
        return {"sums": []}
    return {"sums": result}

# Endpoint to get the frequency and k_symbol for a specific account ID and total amount
@app.get("/v1/financial/frequency_k_symbol_by_account_total", operation_id="get_frequency_k_symbol_by_account_total", summary="Retrieves the frequency and associated symbol for a given account, based on a specified total amount. This operation joins data from the account and order tables, filtering results by the provided account ID and total amount.")
async def get_frequency_k_symbol_by_account_total(account_id: int = Query(..., description="Account ID"), total_amount: float = Query(..., description="Total amount")):
    cursor.execute("SELECT T1.frequency, T2.k_symbol FROM account AS T1 INNER JOIN (SELECT account_id, k_symbol, SUM(amount) AS total_amount FROM `order` GROUP BY account_id, k_symbol) AS T2 ON T1.account_id = T2.account_id WHERE T1.account_id = ? AND T2.total_amount = ?", (account_id, total_amount))
    result = cursor.fetchone()
    if not result:
        return {"frequency": [], "k_symbol": []}
    return {"frequency": result[0], "k_symbol": result[1]}

# Endpoint to get the birth year of clients associated with a specific account ID
@app.get("/v1/financial/client_birth_year_by_account", operation_id="get_client_birth_year_by_account", summary="Retrieves the birth year of clients linked to a specific account. The account is identified by its unique account ID. This operation returns the birth year of each client associated with the provided account ID.")
async def get_client_birth_year_by_account(account_id: int = Query(..., description="Account ID")):
    cursor.execute("SELECT STRFTIME('%Y', T1.birth_date) FROM client AS T1 INNER JOIN disp AS T3 ON T1.client_id = T3.client_id INNER JOIN account AS T2 ON T3.account_id = T2.account_id WHERE T2.account_id = ?", (account_id,))
    result = cursor.fetchall()
    if not result:
        return {"birth_years": []}
    return {"birth_years": [row[0] for row in result]}

# Endpoint to get the loan amount and status for a specific client ID
@app.get("/v1/financial/loan_amount_status_by_client", operation_id="get_loan_amount_status_by_client", summary="Retrieves the total loan amount and its current status for a specific client. The client is identified by a unique client_id. The operation fetches the loan details from the associated account and displays the amount and status.")
async def get_loan_amount_status_by_client(client_id: int = Query(..., description="Client ID")):
    cursor.execute("SELECT T4.amount, T4.status FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN account AS T3 on T2.account_id = T3.account_id INNER JOIN loan AS T4 ON T3.account_id = T4.account_id WHERE T1.client_id = ?", (client_id,))
    result = cursor.fetchall()
    if not result:
        return {"loans": []}
    return {"loans": [{"amount": row[0], "status": row[1]} for row in result]}

# Endpoint to get the balance and gender for a specific client ID and transaction ID
@app.get("/v1/financial/balance_gender_by_client_transaction", operation_id="get_balance_gender_by_client_transaction", summary="Retrieves the balance and gender associated with a specific client and transaction. This operation requires the client's unique identifier and the transaction's unique identifier as input. The balance and gender information is obtained by joining data from the client, disp, account, and trans tables.")
async def get_balance_gender_by_client_transaction(client_id: int = Query(..., description="Client ID"), trans_id: int = Query(..., description="Transaction ID")):
    cursor.execute("SELECT T4.balance, T1.gender FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN account AS T3 ON T2.account_id =T3.account_id INNER JOIN trans AS T4 ON T3.account_id = T4.account_id WHERE T1.client_id = ? AND T4.trans_id = ?", (client_id, trans_id))
    result = cursor.fetchone()
    if not result:
        return {"balance": [], "gender": []}
    return {"balance": result[0], "gender": result[1]}

# Endpoint to get the card type for a specific client ID
@app.get("/v1/financial/card_type_by_client", operation_id="get_card_type_by_client", summary="Retrieves the type of card associated with a specific client. The client is identified by a unique client_id. The operation returns the card type after performing a series of joins on the client, disp, and card tables using the provided client_id.")
async def get_card_type_by_client(client_id: int = Query(..., description="Client ID")):
    cursor.execute("SELECT T3.type FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN card AS T3 ON T2.disp_id = T3.disp_id WHERE T1.client_id = ?", (client_id,))
    result = cursor.fetchall()
    if not result:
        return {"card_types": []}
    return {"card_types": [row[0] for row in result]}

# Endpoint to get the sum of transaction amounts for a specific client in a given year
@app.get("/v1/financial/sum_transaction_amounts_by_client_year", operation_id="get_sum_transaction_amounts", summary="Retrieves the total sum of transaction amounts for a specific client in a given year. The operation requires the client's unique identifier and the year for which the sum is to be calculated. The year should be provided in the 'YYYY' format.")
async def get_sum_transaction_amounts(year: str = Query(..., description="Year in 'YYYY' format"), client_id: int = Query(..., description="Client ID")):
    cursor.execute("SELECT SUM(T3.amount) FROM client AS T1 INNER JOIN disp AS T4 ON T1.client_id = T4.client_id INNER JOIN account AS T2 ON T4.account_id = T2.account_id INNER JOIN trans AS T3 ON T2.account_id = T3.account_id WHERE STRFTIME('%Y', T3.date) = ? AND T1.client_id = ?", (year, client_id))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get client and account IDs based on district and birth year range
@app.get("/v1/financial/client_account_ids_by_district_birth_year", operation_id="get_client_account_ids", summary="Retrieves the client and account IDs for clients born within a specified year range in a given district. The district is identified by its name, and the birth year range is defined by the start and end years in 'YYYY' format. This operation returns a list of client and account ID pairs that meet the specified criteria.")
async def get_client_account_ids(district: str = Query(..., description="District name"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT T1.client_id, T3.account_id FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T4 ON T1.client_id = T4.client_id INNER JOIN account AS T3 ON T2.district_id = T3.district_id AND T4.account_id = T3.account_id WHERE T2.A3 = ? AND STRFTIME('%Y', T1.birth_date) BETWEEN ? AND ?", (district, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"client_account_ids": []}
    return {"client_account_ids": result}

# Endpoint to get top 3 female clients based on loan amount
@app.get("/v1/financial/top_female_clients_by_loan_amount", operation_id="get_top_female_clients", summary="Retrieves the top three female clients with the highest loan amounts. The operation filters clients based on the provided gender and sorts them by loan amount in descending order.")
async def get_top_female_clients(gender: str = Query(..., description="Gender of the client")):
    cursor.execute("SELECT T1.client_id FROM client AS T1 INNER JOIN disp AS T4 ON T1.client_id = T4.client_id INNER JOIN account AS T2 ON T4.account_id = T2.account_id INNER JOIN loan AS T3 ON T2.account_id = T3.account_id AND T4.account_id = T3.account_id WHERE T1.gender = ? ORDER BY T3.amount DESC LIMIT 3", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"client_ids": []}
    return {"client_ids": result}

# Endpoint to get the count of accounts based on birth year range, gender, transaction amount, and transaction type
@app.get("/v1/financial/count_accounts_by_birth_year_gender_amount_type", operation_id="get_count_accounts", summary="Retrieves the total number of accounts that meet the specified criteria: birth year range, gender, minimum transaction amount, and transaction type. The birth year range is defined by the start and end years. The gender, minimum transaction amount, and transaction type are also considered in the count.")
async def get_count_accounts(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), gender: str = Query(..., description="Gender of the client"), min_amount: int = Query(..., description="Minimum transaction amount"), k_symbol: str = Query(..., description="Transaction type")):
    cursor.execute("SELECT COUNT(T1.account_id) FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN disp AS T4 ON T2.account_id = T4.account_id INNER JOIN client AS T3 ON T4.client_id = T3.client_id WHERE STRFTIME('%Y', T3.birth_date) BETWEEN ? AND ? AND T3.gender = ? AND T1.amount > ? AND T1.k_symbol = ?", (start_year, end_year, gender, min_amount, k_symbol))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of accounts in a specific district after a given year
@app.get("/v1/financial/count_accounts_by_district_year", operation_id="get_count_accounts_by_district_year", summary="Retrieves the total number of accounts in a specified district that were created after a given year. The year is provided in 'YYYY' format, and the district is identified by its name.")
async def get_count_accounts_by_district_year(year: str = Query(..., description="Year in 'YYYY' format"), district: str = Query(..., description="District name")):
    cursor.execute("SELECT COUNT(account_id) FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE STRFTIME('%Y', T1.date) > ? AND T2.A2 = ?", (year, district))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of clients based on gender and card type
@app.get("/v1/financial/count_clients_by_gender_card_type", operation_id="get_count_clients_by_gender_card_type", summary="Retrieves the total number of clients categorized by gender and card type. The operation uses the provided gender and card type as filters to calculate the count.")
async def get_count_clients_by_gender_card_type(gender: str = Query(..., description="Gender of the client"), card_type: str = Query(..., description="Card type")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN card AS T3 ON T2.disp_id = T3.disp_id WHERE T1.gender = ? AND T3.type = ?", (gender, card_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of clients of a specific gender in a given district
@app.get("/v1/financial/percentage_clients_by_gender_district", operation_id="get_percentage_clients_by_gender_district", summary="Retrieves the percentage of clients of a specific gender in a given district. The operation calculates this percentage by comparing the count of clients of the specified gender in the district to the total number of clients in the district. The gender and district are provided as input parameters.")
async def get_percentage_clients_by_gender_district(gender: str = Query(..., description="Gender of the client"), district: str = Query(..., description="District name")):
    cursor.execute("SELECT CAST(SUM(T2.gender = ?) AS REAL) / COUNT(T2.client_id) * 100 FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T1.A3 = ?", (gender, district))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of clients of a specific gender with a given account frequency
@app.get("/v1/financial/percentage_clients_by_gender_account_frequency", operation_id="get_percentage_clients_by_gender_account_frequency", summary="Retrieves the percentage of clients of a specific gender who have a given account frequency. This operation calculates the percentage by comparing the count of clients with the specified gender and account frequency to the total number of clients in the same district. The gender and account frequency are provided as input parameters.")
async def get_percentage_clients_by_gender_account_frequency(gender: str = Query(..., description="Gender of the client"), frequency: str = Query(..., description="Account frequency")):
    cursor.execute("SELECT CAST(SUM(T1.gender = ?) AS REAL) * 100 / COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T3 ON T1.district_id = T3.district_id INNER JOIN account AS T2 ON T2.district_id = T3.district_id INNER JOIN disp AS T4 ON T1.client_id = T4.client_id AND T2.account_id = T4.account_id WHERE T2.frequency = ?", (gender, frequency))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of accounts based on account frequency and dispensation type
@app.get("/v1/financial/count_accounts_by_frequency_type", operation_id="get_count_accounts_by_frequency_type", summary="Retrieves the total number of accounts that match the specified frequency and dispensation type. The frequency parameter determines the account frequency, while the disp_type parameter filters the accounts based on their dispensation type.")
async def get_count_accounts_by_frequency_type(frequency: str = Query(..., description="Account frequency"), disp_type: str = Query(..., description="Dispensation type")):
    cursor.execute("SELECT COUNT(T2.account_id) FROM account AS T1 INNER JOIN disp AS T2 ON T2.account_id = T1.account_id WHERE T1.frequency = ? AND T2.type = ?", (frequency, disp_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the account ID with the smallest loan amount for loans with a duration greater than a specified value and created before a given year
@app.get("/v1/financial/smallest_loan_account_by_duration_year", operation_id="get_smallest_loan_account", summary="Retrieves the account identifier associated with the loan account that has the smallest loan amount. This operation considers loans with a duration exceeding the provided duration in months and accounts created before the specified year. The result is sorted in ascending order by loan amount and limited to the top record.")
async def get_smallest_loan_account(duration: int = Query(..., description="Loan duration in months"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T1.account_id FROM loan AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id WHERE T1.duration > ? AND STRFTIME('%Y', T2.date) < ? ORDER BY T1.amount ASC LIMIT 1", (duration, year))
    result = cursor.fetchone()
    if not result:
        return {"account_id": []}
    return {"account_id": result[0]}

# Endpoint to get the account ID of the first female client based on birth date and A11
@app.get("/v1/financial/first_female_client_account_id", operation_id="get_first_female_client_account_id", summary="Retrieves the account ID of the first female client, sorted by birth date and district A11 value. The client's gender is used as a filter.")
async def get_first_female_client_account_id(gender: str = Query(..., description="Gender of the client (e.g., 'F' for female)")):
    cursor.execute("SELECT T3.account_id FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN account AS T3 ON T2.district_id = T3.district_id INNER JOIN disp AS T4 ON T1.client_id = T4.client_id AND T4.account_id = T3.account_id WHERE T1.gender = ? ORDER BY T1.birth_date ASC, T2.A11 ASC LIMIT 1", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"account_id": []}
    return {"account_id": result[0]}

# Endpoint to get the count of clients born in a specific year and living in a specific district
@app.get("/v1/financial/client_count_by_birth_year_and_district", operation_id="get_client_count_by_birth_year_and_district", summary="Retrieves the total number of clients born in a specific year and residing in a given district. The response is based on the provided birth year and district name.")
async def get_client_count_by_birth_year_and_district(birth_year: str = Query(..., description="Birth year in 'YYYY' format"), district: str = Query(..., description="District name")):
    cursor.execute("SELECT COUNT(T1.client_id) FROM client AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id WHERE STRFTIME('%Y', T1.birth_date) = ? AND T2.A3 = ?", (birth_year, district))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of accounts with a specific loan duration and payment frequency
@app.get("/v1/financial/account_count_by_loan_duration_and_frequency", operation_id="get_account_count_by_loan_duration_and_frequency", summary="Retrieves the total number of accounts that have a loan with a specified duration and payment frequency. The duration is measured in months, and the frequency can be any valid payment frequency, such as 'POPLATEK TYDNE'.")
async def get_account_count_by_loan_duration_and_frequency(duration: int = Query(..., description="Loan duration in months"), frequency: str = Query(..., description="Payment frequency (e.g., 'POPLATEK TYDNE')")):
    cursor.execute("SELECT COUNT(T2.account_id) FROM account AS T1 INNER JOIN loan AS T2 ON T1.account_id = T2.account_id WHERE T2.duration = ? AND T1.frequency = ?", (duration, frequency))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average loan amount for accounts with specific statuses and payment frequency
@app.get("/v1/financial/average_loan_amount_by_status_and_frequency", operation_id="get_average_loan_amount_by_status_and_frequency", summary="Retrieves the average loan amount for accounts with the specified statuses and payment frequency. The operation calculates the average loan amount from the loan table, filtering by the provided statuses and payment frequency. The result is a single average value representing the average loan amount for accounts that meet the specified criteria.")
async def get_average_loan_amount_by_status_and_frequency(status1: str = Query(..., description="First loan status (e.g., 'C')"), status2: str = Query(..., description="Second loan status (e.g., 'D')"), frequency: str = Query(..., description="Payment frequency (e.g., 'POPLATEK PO OBRATU')")):
    cursor.execute("SELECT AVG(T2.amount) FROM account AS T1 INNER JOIN loan AS T2 ON T1.account_id = T2.account_id WHERE T2.status IN (?, ?) AND T1.frequency = ?", (status1, status2, frequency))
    result = cursor.fetchone()
    if not result:
        return {"average_amount": []}
    return {"average_amount": result[0]}

# Endpoint to get client details based on dispensation type
@app.get("/v1/financial/client_details_by_dispensation_type", operation_id="get_client_details_by_dispensation_type", summary="Retrieves client details, including client ID, district ID, and a specific attribute (A2), based on the provided dispensation type. This operation fetches data from the 'account', 'district', and 'disp' tables, joining them on the 'account_id' and 'district_id' fields. The dispensation type is used to filter the results.")
async def get_client_details_by_dispensation_type(dispensation_type: str = Query(..., description="Dispensation type (e.g., 'OWNER')")):
    cursor.execute("SELECT T3.client_id, T2.district_id, T2.A2 FROM account AS T1 INNER JOIN district AS T2 ON T1.district_id = T2.district_id INNER JOIN disp AS T3 ON T1.account_id = T3.account_id WHERE T3.type = ?", (dispensation_type,))
    result = cursor.fetchall()
    if not result:
        return {"client_details": []}
    return {"client_details": result}

# Endpoint to get client IDs and their ages based on card and dispensation types
@app.get("/v1/financial/client_ids_and_ages_by_card_and_dispensation_types", operation_id="get_client_ids_and_ages_by_card_and_dispensation_types", summary="Retrieves the IDs and ages of clients who possess a specific card type and are associated with a certain dispensation type. The card type and dispensation type are provided as input parameters.")
async def get_client_ids_and_ages_by_card_and_dispensation_types(card_type: str = Query(..., description="Card type (e.g., 'gold')"), dispensation_type: str = Query(..., description="Dispensation type (e.g., 'OWNER')")):
    cursor.execute("SELECT T1.client_id, STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T3.birth_date) FROM disp AS T1 INNER JOIN card AS T2 ON T2.disp_id = T1.disp_id INNER JOIN client AS T3 ON T1.client_id = T3.client_id WHERE T2.type = ? AND T1.type = ?", (card_type, dispensation_type))
    result = cursor.fetchall()
    if not result:
        return {"client_details": []}
    return {"client_details": result}

api_calls = [
    "/v1/financial/count_account_ids_by_district_and_frequency?district=east%20Bohemia&frequency=POPLATEK%20PO%20OBRATU",
    "/v1/financial/count_account_ids_by_district?district=Prague",
    "/v1/financial/get_year_based_on_average_comparison",
    "/v1/financial/count_distinct_district_ids_by_gender_and_a11_range?gender=F&min_a11=6000&max_a11=10000",
    "/v1/financial/count_client_ids_by_gender_district_and_a11?gender=M&district=north%20Bohemia&min_a11=8000",
    "/v1/financial/get_account_id_and_a11_range_by_earliest_born_female?gender=F",
    "/v1/financial/get_account_ids_by_latest_born_client",
    "/v1/financial/count_account_ids_by_disp_type_and_frequency?disp_type=OWNER&frequency=POPLATEK%20TYDNE",
    "/v1/financial/get_client_ids_by_frequency_and_disp_type?frequency=POPLATEK%20PO%20OBRATU&disp_type=DISPONENT",
    "/v1/financial/get_account_id_by_loan_date_year_and_frequency?year=1997&frequency=POPLATEK%20TYDNE",
    "/v1/financial/loan_account_id_by_year_duration?year=1993&duration=12",
    "/v1/financial/count_female_clients_by_year_district?gender=F&year=1950&district=Sokolov",
    "/v1/financial/earliest_transaction_account_id_by_year?year=1995",
    "/v1/financial/distinct_account_ids_by_year_amount?year=1997&amount=3000",
    "/v1/financial/client_ids_by_card_issue_date?issue_date=1994-03-03",
    "/v1/financial/account_date_by_amount_and_date?amount=840&date=1998-10-14",
    "/v1/financial/district_ids_by_loan_date?date=1994-08-25",
    "/v1/financial/highest_transaction_amount_by_card_issue_date?issue_date=1996-10-21",
    "/v1/financial/client_gender_by_district",
    "/v1/financial/transaction_amount_by_highest_loan",
    "/v1/financial/client_count_by_gender_district?gender=F&district=Jesenik",
    "/v1/financial/disp_id_by_transaction_date_amount?date=1997-08-20&amount=5100",
    "/v1/financial/account_count_by_year_district?year=1996&district=Litomerice",
    "/v1/financial/district_names_by_birth_date_gender?birth_date=1976-01-29&gender=F",
    "/v1/financial/client_birth_dates_by_loan_date_amount?loan_date=1996-01-03&amount=98832",
    "/v1/financial/earliest_account_id_by_district?district=Prague",
    "/v1/financial/percentage_male_clients_by_district?district=south%20Bohemia",
    "/v1/financial/percentage_change_balance_by_dates?date1=1998-12-27&date2=1993-03-22&loan_date=1993-07-05",
    "/v1/financial/percentage_loan_amount_status_a",
    "/v1/financial/percentage_loans_status_c_below_amount?amount=100000",
    "/v1/financial/account_details_by_frequency_year?frequency=POPLATEK%20PO%20OBRATU&year=1993",
    "/v1/financial/account_details_by_district_year_range?district=east%20Bohemia&start_year=1995&end_year=2000",
    "/v1/financial/account_details_by_district_name?district_name=Prachatice",
    "/v1/financial/district_details_by_loan_id?loan_id=4990",
    "/v1/financial/account_details_by_loan_amount?amount=300000",
    "/v1/financial/loan_details_by_duration?duration=60",
    "/v1/financial/loan_status_percentage?status=D",
    "/v1/financial/account_percentage_by_district_year?district=Decin&year=1993",
    "/v1/financial/account_ids_by_frequency?frequency=POPLATEK%20MESICNE",
    "/v1/financial/client_count_by_district_gender?gender=F&limit=9",
    "/v1/financial/distinct_a2_values?transaction_type=VYDAJ&date_pattern=1996-01%",
    "/v1/financial/count_account_ids_by_district_a3_and_type?district_a3=south%20Bohemia&disposition_type=OWNER",
    "/v1/financial/top_district_by_loan_amount?status1=C&status2=D",
    "/v1/financial/average_loan_amount_by_gender?gender=M",
    "/v1/financial/top_district_by_a13",
    "/v1/financial/count_account_ids_by_district_a16",
    "/v1/financial/count_account_ids_by_transaction_criteria?balance=0&operation=VYBER%20KARTOU&frequency=POPLATEK%20MESICNE",
    "/v1/financial/count_account_ids_by_loan_criteria?start_date=1995-01-01&end_date=1997-12-31&frequency=POPLATEK%20MESICNE&amount=250000",
    "/v1/financial/count_account_ids_by_district_and_loan_status?district_id=1&status1=C&status2=D",
    "/v1/financial/count_client_ids_by_gender_and_district_a15?gender=M",
    "/v1/financial/count_card_ids_by_type?card_type=gold&disp_type=OWNER",
    "/v1/financial/district_ids_by_year_and_amount?year=1997&min_amount=10000",
    "/v1/financial/distinct_account_ids_by_symbol_and_district?k_symbol=SIPO&district_a2=Pisek",
    "/v1/financial/account_ids_by_card_type?card_type=gold",
    "/v1/financial/avg_transaction_amount_by_year_and_operation?year=1998&operation=VYBER%20KARTOU",
    "/v1/financial/account_ids_by_year_operation_and_amount?year=1998&operation=VYBER%20KARTOU",
    "/v1/financial/client_ids_by_gender?gender=F",
    "/v1/financial/count_client_ids_by_gender_and_district?gender=F&district_a3=south%20Bohemia",
    "/v1/financial/account_ids_by_disp_type_and_district?disp_type=OWNER&district_a2=Tabor",
    "/v1/financial/disp_types_by_district_range?excluded_type=OWNER&min_a11=8000&max_a11=9000",
    "/v1/financial/account_count_by_bank_and_district?bank=AB&a3=north%20Bohemia",
    "/v1/financial/distinct_district_a2_by_trans_type?trans_type=VYDAJ",
    "/v1/financial/avg_district_a15_by_year_and_min_value?year=1997&min_a15=4000",
    "/v1/financial/percentage_cards_by_type_and_year?card_type=gold&year=1998",
    "/v1/financial/client_id_highest_loan_by_disp_type?disp_type=OWNER",
    "/v1/financial/district_a15_by_account_id?account_id=532",
    "/v1/financial/district_id_by_order_id?order_id=33333",
    "/v1/financial/trans_ids_by_client_and_operation?client_id=3356&operation=VYBER",
    "/v1/financial/loan_account_count_by_frequency_and_amount?frequency=POPLATEK%20TYDNE&max_amount=200000",
    "/v1/financial/card_types_by_client?client_id=13539",
    "/v1/financial/district_a3_by_client?client_id=3541",
    "/v1/financial/top_district_a2_by_loan_status?status=A",
    "/v1/financial/client_ids_by_order?order_id=32423",
    "/v1/financial/trans_ids_by_district?district_id=5",
    "/v1/financial/account_count_by_district?a2=Jesenik",
    "/v1/financial/client_ids_by_card_type_and_issue_date?card_type=junior&issued_date=1997-01-01",
    "/v1/financial/female_client_percentage_by_district_a11?a11_threshold=10000",
    "/v1/financial/loan_amount_change_percentage?year1=1997&year2=1996&year3=1996&gender=M&type=OWNER",
    "/v1/financial/transaction_count_after_year?year=1995&operation=VYBER%20KARTOU",
    "/v1/financial/sum_difference_by_region?region1=east%20Bohemia&region2=north%20Bohemia",
    "/v1/financial/sum_types_by_account_range?type1=OWNER&type2=DISPONENT&min_account_id=1&max_account_id=10",
    "/v1/financial/frequency_k_symbol_by_account_total?account_id=3&total_amount=3539",
    "/v1/financial/client_birth_year_by_account?account_id=130",
    "/v1/financial/loan_amount_status_by_client?client_id=992",
    "/v1/financial/balance_gender_by_client_transaction?client_id=4&trans_id=851",
    "/v1/financial/card_type_by_client?client_id=9",
    "/v1/financial/sum_transaction_amounts_by_client_year?year=1998&client_id=617",
    "/v1/financial/client_account_ids_by_district_birth_year?district=east%20Bohemia&start_year=1983&end_year=1987",
    "/v1/financial/top_female_clients_by_loan_amount?gender=F",
    "/v1/financial/count_accounts_by_birth_year_gender_amount_type?start_year=1974&end_year=1976&gender=M&min_amount=4000&k_symbol=SIPO",
    "/v1/financial/count_accounts_by_district_year?year=1996&district=Beroun",
    "/v1/financial/count_clients_by_gender_card_type?gender=F&card_type=junior",
    "/v1/financial/percentage_clients_by_gender_district?gender=F&district=Prague",
    "/v1/financial/percentage_clients_by_gender_account_frequency?gender=M&frequency=POPLATEK%20TYDNE",
    "/v1/financial/count_accounts_by_frequency_type?frequency=POPLATEK%20TYDNE&disp_type=OWNER",
    "/v1/financial/smallest_loan_account_by_duration_year?duration=24&year=1997",
    "/v1/financial/first_female_client_account_id?gender=F",
    "/v1/financial/client_count_by_birth_year_and_district?birth_year=1920&district=east%20Bohemia",
    "/v1/financial/account_count_by_loan_duration_and_frequency?duration=24&frequency=POPLATEK%20TYDNE",
    "/v1/financial/average_loan_amount_by_status_and_frequency?status1=C&status2=D&frequency=POPLATEK%20PO%20OBRATU",
    "/v1/financial/client_details_by_dispensation_type?dispensation_type=OWNER",
    "/v1/financial/client_ids_and_ages_by_card_and_dispensation_types?card_type=gold&dispensation_type=OWNER"
]
