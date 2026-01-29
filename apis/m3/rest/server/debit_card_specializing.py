from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/debit_card_specializing/debit_card_specializing.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of gas stations based on country and segment
@app.get("/v1/debit_card_specializing/gas_station_count", operation_id="get_gas_station_count", summary="Retrieves the total number of gas stations in a specified country and segment. The operation filters gas stations based on the provided country and segment, then returns the count of matching records.")
async def get_gas_station_count(country: str = Query(..., description="Country of the gas station"), segment: str = Query(..., description="Segment of the gas station")):
    cursor.execute("SELECT COUNT(GasStationID) FROM gasstations WHERE Country = ? AND Segment = ?", (country, segment))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of customers based on currency
@app.get("/v1/debit_card_specializing/customer_currency_ratio", operation_id="get_customer_currency_ratio", summary="Retrieves the proportion of customers who use a specific currency compared to another currency. The operation calculates the ratio by dividing the count of customers using the first currency by the count of customers using the second currency.")
async def get_customer_currency_ratio(currency1: str = Query(..., description="First currency for comparison"), currency2: str = Query(..., description="Second currency for comparison")):
    cursor.execute("SELECT CAST(SUM(IIF(Currency = ?, 1, 0)) AS FLOAT) / SUM(IIF(Currency = ?, 1, 0)) AS ratio FROM customers", (currency1, currency2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the customer ID with the lowest consumption in a specific year and segment
@app.get("/v1/debit_card_specializing/lowest_consumption_customer", operation_id="get_lowest_consumption_customer", summary="Retrieves the unique identifier of the customer who has the lowest total consumption in a given year and segment. The customer's segment and the year are required as input parameters to filter the results.")
async def get_lowest_consumption_customer(segment: str = Query(..., description="Segment of the customer"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = ? AND SUBSTR(T2.Date, 1, 4) = ? GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) ASC LIMIT 1", (segment, year))
    result = cursor.fetchone()
    if not result:
        return {"customer_id": []}
    return {"customer_id": result[0]}

# Endpoint to get the average monthly consumption for a specific year and segment
@app.get("/v1/debit_card_specializing/average_monthly_consumption", operation_id="get_average_monthly_consumption", summary="Retrieves the average monthly consumption for a given year and customer segment. This operation calculates the average annual consumption for the specified year and segment, then divides it by 12 to obtain the average monthly consumption. The result provides insights into spending patterns for a particular customer segment within a year.")
async def get_average_monthly_consumption(year: str = Query(..., description="Year in 'YYYY' format"), segment: str = Query(..., description="Segment of the customer")):
    cursor.execute("SELECT AVG(T2.Consumption) / 12 FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE SUBSTR(T2.Date, 1, 4) = ? AND T1.Segment = ?", (year, segment))
    result = cursor.fetchone()
    if not result:
        return {"average_consumption": []}
    return {"average_consumption": result[0]}

# Endpoint to get the customer ID with the highest consumption in a specific date range and currency
@app.get("/v1/debit_card_specializing/highest_consumption_customer", operation_id="get_highest_consumption_customer", summary="Retrieve the ID of the customer who has the highest total consumption within a specified date range and currency. The operation filters customers based on the provided currency and date range, then calculates the total consumption for each customer. The customer with the highest total consumption is identified and returned.")
async def get_highest_consumption_customer(currency: str = Query(..., description="Currency of the customer"), start_date: int = Query(..., description="Start date in 'YYYYMM' format"), end_date: int = Query(..., description="End date in 'YYYYMM' format")):
    cursor.execute("SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Currency = ? AND T2.Date BETWEEN ? AND ? GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) DESC LIMIT 1", (currency, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"customer_id": []}
    return {"customer_id": result[0]}

# Endpoint to get the count of customers with consumption below a threshold in a specific year and segment
@app.get("/v1/debit_card_specializing/customer_count_below_threshold", operation_id="get_customer_count_below_threshold", summary="Retrieves the number of customers whose total consumption in a given year falls below a specified threshold for a particular segment. The segment, year, and consumption threshold are provided as input parameters.")
async def get_customer_count_below_threshold(segment: str = Query(..., description="Segment of the customer"), year: str = Query(..., description="Year in 'YYYY' format"), threshold: int = Query(..., description="Consumption threshold")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T2.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = ? AND SUBSTRING(T2.Date, 1, 4) = ? GROUP BY T2.CustomerID HAVING SUM(T2.Consumption) < ? ) AS t1", (segment, year, threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in consumption between two currencies in a specific year
@app.get("/v1/debit_card_specializing/consumption_difference", operation_id="get_consumption_difference", summary="Retrieves the difference in total consumption between two specified currencies for a given year. The calculation is based on aggregated consumption data from customers who use the respective currencies.")
async def get_consumption_difference(currency1: str = Query(..., description="First currency for comparison"), currency2: str = Query(..., description="Second currency for comparison"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(IIF(T1.Currency = ?, T2.Consumption, 0)) - SUM(IIF(T1.Currency = ?, T2.Consumption, 0)) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE SUBSTR(T2.Date, 1, 4) = ?", (currency1, currency2, year))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the year with the highest consumption for a specific currency
@app.get("/v1/debit_card_specializing/highest_consumption_year", operation_id="get_highest_consumption_year", summary="Retrieves the year with the highest total consumption for a specific currency. This operation calculates the total consumption for each year based on the provided currency, then returns the year with the highest total consumption. The input parameter specifies the currency for which the highest consumption year is determined.")
async def get_highest_consumption_year(currency: str = Query(..., description="Currency of the customer")):
    cursor.execute("SELECT SUBSTRING(T2.Date, 1, 4) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Currency = ? GROUP BY SUBSTRING(T2.Date, 1, 4) ORDER BY SUM(T2.Consumption) DESC LIMIT 1", (currency,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the segment with the lowest consumption
@app.get("/v1/debit_card_specializing/lowest_consumption_segment", operation_id="get_lowest_consumption_segment", summary="Retrieves the customer segment with the lowest total consumption. This operation calculates the total consumption for each segment by aggregating consumption data from the yearmonth table and joining it with the customers table. The segment with the lowest total consumption is then returned.")
async def get_lowest_consumption_segment():
    cursor.execute("SELECT T1.Segment FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID GROUP BY T1.Segment ORDER BY SUM(T2.Consumption) ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"segment": []}
    return {"segment": result[0]}

# Endpoint to get the year with the highest consumption for a specific currency
@app.get("/v1/debit_card_specializing/highest_consumption_year_currency", operation_id="get_highest_consumption_year_currency", summary="Retrieves the year with the highest total consumption for a given currency. This operation fetches the year with the maximum aggregate consumption for a specific currency, based on the customer's transactions. The currency is provided as an input parameter.")
async def get_highest_consumption_year_currency(currency: str = Query(..., description="Currency of the customer")):
    cursor.execute("SELECT SUBSTR(T2.Date, 1, 4) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Currency = ? GROUP BY SUBSTR(T2.Date, 1, 4) ORDER BY SUM(T2.Consumption) DESC LIMIT 1", (currency,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the month with the highest consumption for a specific year and segment
@app.get("/v1/debit_card_specializing/highest_consumption_month", operation_id="get_highest_consumption_month", summary="Retrieves the month with the highest total consumption for a given year and customer segment. The operation calculates the total consumption for each month in the specified year and segment, then returns the month with the highest sum.")
async def get_highest_consumption_month(year: str = Query(..., description="Year in 'YYYY' format"), segment: str = Query(..., description="Customer segment")):
    cursor.execute("SELECT SUBSTR(T2.Date, 5, 2) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE SUBSTR(T2.Date, 1, 4) = ? AND T1.Segment = ? GROUP BY SUBSTR(T2.Date, 5, 2) ORDER BY SUM(T2.Consumption) DESC LIMIT 1", (year, segment))
    result = cursor.fetchone()
    if not result:
        return {"month": []}
    return {"month": result[0]}

# Endpoint to get the consumption difference between segments for a specific currency and date range
@app.get("/v1/debit_card_specializing/consumption_difference_segments", operation_id="get_consumption_difference_segments", summary="Retrieves the average consumption difference between three distinct customer segments for a specified currency and date range. The segments are compared pairwise, and the results are presented as the difference in their average consumption values. The date range must be provided in 'YYYYMM' format, and the currency is identified by its code.")
async def get_consumption_difference_segments(segment1: str = Query(..., description="First customer segment"), segment2: str = Query(..., description="Second customer segment"), segment3: str = Query(..., description="Third customer segment"), currency: str = Query(..., description="Currency code"), start_date: int = Query(..., description="Start date in 'YYYYMM' format"), end_date: int = Query(..., description="End date in 'YYYYMM' format")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.Segment = ?, T2.Consumption, 0)) AS REAL) / COUNT(T1.CustomerID) - CAST(SUM(IIF(T1.Segment = ?, T2.Consumption, 0)) AS REAL) / COUNT(T1.CustomerID) , CAST(SUM(IIF(T1.Segment = ?, T2.Consumption, 0)) AS REAL) / COUNT(T1.CustomerID) - CAST(SUM(IIF(T1.Segment = ?, T2.Consumption, 0)) AS REAL) / COUNT(T1.CustomerID) , CAST(SUM(IIF(T1.Segment = ?, T2.Consumption, 0)) AS REAL) / COUNT(T1.CustomerID) - CAST(SUM(IIF(T1.Segment = ?, T2.Consumption, 0)) AS REAL) / COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Currency = ? AND T2.Consumption = ( SELECT MIN(Consumption) FROM yearmonth ) AND T2.Date BETWEEN ? AND ?", (segment1, segment2, segment3, segment1, segment2, segment3, currency, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result}

# Endpoint to get the percentage change in consumption between two years for different segments
@app.get("/v1/debit_card_specializing/consumption_change_percentage", operation_id="get_consumption_change_percentage", summary="Retrieves the percentage change in consumption for three distinct customer segments between three different years. The calculation is based on the total consumption of each segment in the specified years.")
async def get_consumption_change_percentage(segment1: str = Query(..., description="First customer segment"), year1: str = Query(..., description="First year in 'YYYY' format"), segment2: str = Query(..., description="Second customer segment"), year2: str = Query(..., description="Second year in 'YYYY' format"), segment3: str = Query(..., description="Third customer segment"), year3: str = Query(..., description="Third year in 'YYYY' format")):
    cursor.execute("SELECT CAST((SUM(IIF(T1.Segment = ? AND T2.Date LIKE ?, T2.Consumption, 0)) - SUM(IIF(T1.Segment = ? AND T2.Date LIKE ?, T2.Consumption, 0))) AS FLOAT) * 100 / SUM(IIF(T1.Segment = ? AND T2.Date LIKE ?, T2.Consumption, 0)), CAST(SUM(IIF(T1.Segment = ? AND T2.Date LIKE ?, T2.Consumption, 0)) - SUM(IIF(T1.Segment = ? AND T2.Date LIKE ?, T2.Consumption, 0)) AS FLOAT) * 100 / SUM(IIF(T1.Segment = ? AND T2.Date LIKE ?, T2.Consumption, 0)) , CAST(SUM(IIF(T1.Segment = ? AND T2.Date LIKE ?, T2.Consumption, 0)) - SUM(IIF(T1.Segment = ? AND T2.Date LIKE ?, T2.Consumption, 0)) AS FLOAT) * 100 / SUM(IIF(T1.Segment = ? AND T2.Date LIKE ?, T2.Consumption, 0)) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID", (segment1, year1+'%', segment1, year2+'%', segment1, year2+'%', segment2, year1+'%', segment2, year2+'%', segment2, year2+'%', segment3, year1+'%', segment3, year2+'%', segment3, year2+'%'))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result}

# Endpoint to get the total consumption for a specific customer and date range
@app.get("/v1/debit_card_specializing/total_consumption_customer_date_range", operation_id="get_total_consumption_customer_date_range", summary="Retrieves the total consumption for a given customer within a specified date range. The calculation is based on the sum of consumption values from the yearmonth table, filtered by the provided customer ID and date range. The date range is inclusive and should be provided in 'YYYYMM' format.")
async def get_total_consumption_customer_date_range(customer_id: int = Query(..., description="Customer ID"), start_date: str = Query(..., description="Start date in 'YYYYMM' format"), end_date: str = Query(..., description="End date in 'YYYYMM' format")):
    cursor.execute("SELECT SUM(Consumption) FROM yearmonth WHERE CustomerID = ? AND Date BETWEEN ? AND ?", (customer_id, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"total_consumption": []}
    return {"total_consumption": result[0]}

# Endpoint to get the difference in the number of gas stations between two countries for a specific segment
@app.get("/v1/debit_card_specializing/gas_station_count_difference", operation_id="get_gas_station_count_difference", summary="Retrieves the difference in the number of gas stations between two specified countries for a given customer segment. This operation compares the total count of gas stations in the first country with the total count in the second country, considering only those gas stations that belong to the provided customer segment.")
async def get_gas_station_count_difference(country1: str = Query(..., description="First country code"), country2: str = Query(..., description="Second country code"), segment: str = Query(..., description="Customer segment")):
    cursor.execute("SELECT SUM(IIF(Country = ?, 1, 0)) - SUM(IIF(Country = ?, 1, 0)) FROM gasstations WHERE Segment = ?", (country1, country2, segment))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the consumption difference between two customers for a specific date
@app.get("/v1/debit_card_specializing/consumption_difference_customers", operation_id="get_consumption_difference_customers", summary="Retrieves the difference in consumption between two specified customers for a given date. The operation compares the total consumption of the first customer with that of the second customer, both on the provided date. The date should be in the 'YYYYMM' format.")
async def get_consumption_difference_customers(customer_id1: int = Query(..., description="First customer ID"), customer_id2: int = Query(..., description="Second customer ID"), date: str = Query(..., description="Date in 'YYYYMM' format")):
    cursor.execute("SELECT SUM(IIF(CustomerID = ?, Consumption, 0)) - SUM(IIF(CustomerID = ?, Consumption, 0)) FROM yearmonth WHERE Date = ?", (customer_id1, customer_id2, date))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the difference in the number of customers between two currencies for a specific segment
@app.get("/v1/debit_card_specializing/customer_count_difference_currencies", operation_id="get_customer_count_difference_currencies", summary="Retrieves the net difference in the number of customers between two specified currencies for a given customer segment. This operation compares the total count of customers using the first currency with those using the second currency, providing a single numerical value as the result.")
async def get_customer_count_difference_currencies(currency1: str = Query(..., description="First currency code"), currency2: str = Query(..., description="Second currency code"), segment: str = Query(..., description="Customer segment")):
    cursor.execute("SELECT SUM(Currency = ?) - SUM(Currency = ?) FROM customers WHERE Segment = ?", (currency1, currency2, segment))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the customer with the highest total consumption for a specific segment
@app.get("/v1/debit_card_specializing/highest_total_consumption_customer", operation_id="get_highest_total_consumption_customer", summary="Retrieves the customer with the highest total consumption for a specified segment. This operation calculates the total consumption for each customer within the given segment and returns the customer with the highest sum. The segment parameter is used to filter the customers.")
async def get_highest_total_consumption_customer(segment: str = Query(..., description="Customer segment")):
    cursor.execute("SELECT T2.CustomerID, SUM(T2.Consumption) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = ? GROUP BY T2.CustomerID ORDER BY SUM(T2.Consumption) DESC LIMIT 1", (segment,))
    result = cursor.fetchone()
    if not result:
        return {"customer_id": [], "total_consumption": []}
    return {"customer_id": result[0], "total_consumption": result[1]}

# Endpoint to get the total consumption for a specific date and segment
@app.get("/v1/debit_card_specializing/total_consumption_date_segment", operation_id="get_total_consumption_date_segment", summary="Retrieves the total consumption for a specific customer segment on a given date. The date should be provided in 'YYYYMM' format. The customer segment is a categorization of customers based on certain criteria.")
async def get_total_consumption_date_segment(date: str = Query(..., description="Date in 'YYYYMM' format"), segment: str = Query(..., description="Customer segment")):
    cursor.execute("SELECT SUM(T2.Consumption) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.Date = ? AND T1.Segment = ?", (date, segment))
    result = cursor.fetchone()
    if not result:
        return {"total_consumption": []}
    return {"total_consumption": result[0]}

# Endpoint to get the percentage of customers with consumption above a certain threshold in a specific segment
@app.get("/v1/debit_card_specializing/consumption_percentage_above_threshold", operation_id="get_consumption_percentage_above_threshold", summary="Retrieves the percentage of customers in a specified segment whose consumption exceeds a given threshold. This operation calculates the proportion of customers with consumption above the provided threshold, based on the joined data from the 'customers' and 'yearmonth' tables.")
async def get_consumption_percentage_above_threshold(consumption_threshold: float = Query(..., description="Consumption threshold"), segment: str = Query(..., description="Customer segment")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.Consumption > ?, 1, 0)) AS FLOAT) * 100 / COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = ?", (consumption_threshold, segment))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the country with the highest number of gas stations in a specific segment
@app.get("/v1/debit_card_specializing/top_country_by_gas_stations", operation_id="get_top_country_by_gas_stations", summary="Get the country with the highest number of gas stations in a specific segment")
async def get_top_country_by_gas_stations(segment: str = Query(..., description="Gas station segment")):
    cursor.execute("SELECT Country , ( SELECT COUNT(GasStationID) FROM gasstations WHERE Segment = ? ) FROM gasstations WHERE Segment = ? GROUP BY Country ORDER BY COUNT(GasStationID) DESC LIMIT 1", (segment, segment))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the percentage of customers using a specific currency in a specific segment
@app.get("/v1/debit_card_specializing/currency_percentage_in_segment", operation_id="get_currency_percentage_in_segment", summary="Retrieves the percentage of customers in a specific segment who use a given currency. The calculation is based on the total number of customers in the specified segment.")
async def get_currency_percentage_in_segment(currency: str = Query(..., description="Currency code"), segment: str = Query(..., description="Customer segment")):
    cursor.execute("SELECT CAST(SUM(Currency = ?) AS FLOAT) * 100 / COUNT(CustomerID) FROM customers WHERE Segment = ?", (currency, segment))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of customers with consumption above a certain threshold on a specific date
@app.get("/v1/debit_card_specializing/consumption_percentage_above_threshold_by_date", operation_id="get_consumption_percentage_above_threshold_by_date", summary="Retrieves the percentage of customers who have exceeded a specified consumption threshold on a given date. The calculation is based on the total number of customers and those who have surpassed the provided consumption threshold on the specified date.")
async def get_consumption_percentage_above_threshold_by_date(consumption_threshold: float = Query(..., description="Consumption threshold"), date: str = Query(..., description="Date in YYYYMM format")):
    cursor.execute("SELECT CAST(SUM(IIF(Consumption > ?, 1, 0)) AS FLOAT) * 100 / COUNT(CustomerID) FROM yearmonth WHERE Date = ?", (consumption_threshold, date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of gas stations in a specific segment in a given country
@app.get("/v1/debit_card_specializing/gas_station_percentage_in_segment_by_country", operation_id="get_gas_station_percentage_in_segment_by_country", summary="Retrieves the percentage of gas stations in a specific segment within a given country. The segment is a categorization of gas stations, and the country is identified by its code. The calculation is based on the total count of gas stations in the specified country.")
async def get_gas_station_percentage_in_segment_by_country(segment: str = Query(..., description="Gas station segment"), country: str = Query(..., description="Country code")):
    cursor.execute("SELECT CAST(SUM(IIF(Segment = ?, 1, 0)) AS FLOAT) * 100 / COUNT(GasStationID) FROM gasstations WHERE Country = ?", (segment, country))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the customer with the highest consumption on a specific date
@app.get("/v1/debit_card_specializing/top_customer_by_consumption", operation_id="get_top_customer_by_consumption", summary="Retrieves the customer who has the highest total consumption on a specific date. The date is provided in the YYYYMM format. The operation returns the unique identifier of the top consumer.")
async def get_top_customer_by_consumption(date: str = Query(..., description="Date in YYYYMM format")):
    cursor.execute("SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.Date = ? GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) DESC LIMIT 1", (date,))
    result = cursor.fetchone()
    if not result:
        return {"customer_id": []}
    return {"customer_id": result[0]}

# Endpoint to get the customer with the lowest consumption in a specific segment on a specific date
@app.get("/v1/debit_card_specializing/lowest_consumption_customer_in_segment", operation_id="get_lowest_consumption_customer_in_segment", summary="Retrieves the customer with the lowest total consumption for a given date and segment. The customer is identified by their unique ID. The date is specified in YYYYMM format, and the segment refers to the customer's segment.")
async def get_lowest_consumption_customer_in_segment(date: str = Query(..., description="Date in YYYYMM format"), segment: str = Query(..., description="Customer segment")):
    cursor.execute("SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.Date = ? AND T1.Segment = ? GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) ASC LIMIT 1", (date, segment))
    result = cursor.fetchone()
    if not result:
        return {"customer_id": []}
    return {"customer_id": result[0]}

# Endpoint to get the highest monthly average consumption for customers using a specific currency
@app.get("/v1/debit_card_specializing/highest_monthly_average_consumption", operation_id="get_highest_monthly_average_consumption", summary="Retrieves the highest monthly average consumption for customers who use a specific currency. The calculation is based on the total annual consumption divided by 12, grouped by customer ID and ordered in descending order. The result is limited to the top record.")
async def get_highest_monthly_average_consumption(currency: str = Query(..., description="Currency code")):
    cursor.execute("SELECT SUM(T2.Consumption) / 12 AS MonthlyConsumption FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Currency = ? GROUP BY T1.CustomerID ORDER BY MonthlyConsumption DESC LIMIT 1", (currency,))
    result = cursor.fetchone()
    if not result:
        return {"monthly_consumption": []}
    return {"monthly_consumption": result[0]}

# Endpoint to get product descriptions based on a specific date
@app.get("/v1/debit_card_specializing/product_descriptions_by_date", operation_id="get_product_descriptions_by_date", summary="Retrieves the descriptions of products associated with transactions that occurred on a specified date. The date should be provided in the 'YYYYMM' format.")
async def get_product_descriptions_by_date(date: str = Query(..., description="Date in 'YYYYMM' format")):
    cursor.execute("SELECT T3.Description FROM transactions_1k AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN products AS T3 ON T1.ProductID = T3.ProductID WHERE T2.Date = ?", (date,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct countries based on a specific date
@app.get("/v1/debit_card_specializing/distinct_countries_by_date", operation_id="get_distinct_countries_by_date", summary="Retrieves a list of unique countries where transactions were made on a specific date. The date is provided in the 'YYYYMM' format.")
async def get_distinct_countries_by_date(date: str = Query(..., description="Date in 'YYYYMM' format")):
    cursor.execute("SELECT DISTINCT T2.Country FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID INNER JOIN yearmonth AS T3 ON T1.CustomerID = T3.CustomerID WHERE T3.Date = ?", (date,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get distinct chain IDs based on a specific currency
@app.get("/v1/debit_card_specializing/distinct_chain_ids_by_currency", operation_id="get_distinct_chain_ids_by_currency", summary="Retrieves a unique set of chain identifiers associated with a given currency. This operation filters transactions based on the specified currency and identifies the distinct chains where these transactions occurred.")
async def get_distinct_chain_ids_by_currency(currency: str = Query(..., description="Currency code")):
    cursor.execute("SELECT DISTINCT T3.ChainID FROM transactions_1k AS T1 INNER JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN gasstations AS T3 ON T1.GasStationID = T3.GasStationID WHERE T2.Currency = ?", (currency,))
    result = cursor.fetchall()
    if not result:
        return {"chain_ids": []}
    return {"chain_ids": [row[0] for row in result]}

# Endpoint to get distinct product IDs and descriptions based on a specific currency
@app.get("/v1/debit_card_specializing/distinct_product_ids_descriptions_by_currency", operation_id="get_distinct_product_ids_descriptions_by_currency", summary="Retrieves unique product identifiers and their corresponding descriptions for a given currency. This operation filters transactions based on the specified currency and returns distinct product details from the associated customers and products.")
async def get_distinct_product_ids_descriptions_by_currency(currency: str = Query(..., description="Currency code")):
    cursor.execute("SELECT DISTINCT T1.ProductID, T3.Description FROM transactions_1k AS T1 INNER JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN products AS T3 ON T1.ProductID = T3.ProductID WHERE T2.Currency = ?", (currency,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [{"product_id": row[0], "description": row[1]} for row in result]}

# Endpoint to get the average transaction amount for a specific month and year
@app.get("/v1/debit_card_specializing/average_transaction_amount_by_month_year", operation_id="get_average_transaction_amount_by_month_year", summary="Retrieves the average transaction amount for a given month and year. The input parameter specifies the month and year using a date pattern in 'YYYY-MM%' format. The operation calculates the average transaction amount from the transactions_1k table, considering only the transactions that match the provided date pattern.")
async def get_average_transaction_amount_by_month_year(date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM%' format")):
    cursor.execute("SELECT AVG(Amount) FROM transactions_1k WHERE Date LIKE ?", (date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"average_amount": []}
    return {"average_amount": result[0]}

# Endpoint to get the count of customers based on currency and consumption
@app.get("/v1/debit_card_specializing/customer_count_by_currency_consumption", operation_id="get_customer_count_by_currency_consumption", summary="Retrieves the total number of customers who have used a specific currency and have a consumption amount greater than a given value.")
async def get_customer_count_by_currency_consumption(currency: str = Query(..., description="Currency code"), consumption: float = Query(..., description="Consumption amount")):
    cursor.execute("SELECT COUNT(*) FROM yearmonth AS T1 INNER JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.Currency = ? AND T1.Consumption > ?", (currency, consumption))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct product descriptions based on a specific country
@app.get("/v1/debit_card_specializing/distinct_product_descriptions_by_country", operation_id="get_distinct_product_descriptions_by_country", summary="Retrieves a unique set of product descriptions associated with a specified country. The operation filters transactions based on the provided country code and returns the distinct product descriptions from the corresponding gas stations.")
async def get_distinct_product_descriptions_by_country(country: str = Query(..., description="Country code")):
    cursor.execute("SELECT DISTINCT T3.Description FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID INNER JOIN products AS T3 ON T1.ProductID = T3.ProductID WHERE T2.Country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct transaction times based on a specific chain ID
@app.get("/v1/debit_card_specializing/distinct_transaction_times_by_chain_id", operation_id="get_distinct_transaction_times_by_chain_id", summary="Retrieve a unique set of transaction times associated with a particular chain of gas stations. The operation filters transactions based on the provided chain ID, ensuring that only times from the specified chain are returned.")
async def get_distinct_transaction_times_by_chain_id(chain_id: int = Query(..., description="Chain ID")):
    cursor.execute("SELECT DISTINCT T1.Time FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T2.ChainID = ?", (chain_id,))
    result = cursor.fetchall()
    if not result:
        return {"times": []}
    return {"times": [row[0] for row in result]}

# Endpoint to get the count of transactions based on country and price
@app.get("/v1/debit_card_specializing/transaction_count_by_country_price", operation_id="get_transaction_count_by_country_price", summary="Retrieves the total number of transactions that occurred in a specific country and exceeded a given price. The operation filters transactions based on the provided country code and price threshold.")
async def get_transaction_count_by_country_price(country: str = Query(..., description="Country code"), price: float = Query(..., description="Price amount")):
    cursor.execute("SELECT COUNT(T1.TransactionID) FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T2.Country = ? AND T1.Price > ?", (country, price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of transactions based on country and year
@app.get("/v1/debit_card_specializing/transaction_count_by_country_year", operation_id="get_transaction_count_by_country_year", summary="Retrieves the total number of transactions that occurred in a specific country from a given year onwards. The operation filters transactions based on the provided country code and year, and returns the count of matching transactions.")
async def get_transaction_count_by_country_year(country: str = Query(..., description="Country code"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.TransactionID) FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T2.Country = ? AND STRFTIME('%Y', T1.Date) >= ?", (country, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average price of transactions in a specific country
@app.get("/v1/debit_card_specializing/average_price_by_country", operation_id="get_average_price_by_country", summary="Retrieves the average price of transactions conducted in a specified country. The calculation is based on a dataset of transactions and their associated gas station information. The country is identified using its code (e.g., 'CZE').")
async def get_average_price_by_country(country: str = Query(..., description="Country code (e.g., 'CZE')")):
    cursor.execute("SELECT AVG(T1.Price) FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T2.Country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the average price of transactions in a specific currency
@app.get("/v1/debit_card_specializing/average_price_by_currency", operation_id="get_average_price_by_currency", summary="Retrieves the average price of transactions conducted in a specified currency. This operation calculates the mean value of transaction prices from a dataset of transactions, filtered by the provided currency code. The result represents the typical transaction price in the given currency.")
async def get_average_price_by_currency(currency: str = Query(..., description="Currency code (e.g., 'EUR')")):
    cursor.execute("SELECT AVG(T1.Price) FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID INNER JOIN customers AS T3 ON T1.CustomerID = T3.CustomerID WHERE T3.Currency = ?", (currency,))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the customer ID with the highest total price on a specific date
@app.get("/v1/debit_card_specializing/top_customer_by_date", operation_id="get_top_customer_by_date", summary="Retrieves the ID of the customer who made the highest total purchase on a specified date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_top_customer_by_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CustomerID FROM transactions_1k WHERE Date = ? GROUP BY CustomerID ORDER BY SUM(Price) DESC LIMIT 1", (date,))
    result = cursor.fetchone()
    if not result:
        return {"customer_id": []}
    return {"customer_id": result[0]}

# Endpoint to get the country of the latest transaction on a specific date
@app.get("/v1/debit_card_specializing/latest_transaction_country_by_date", operation_id="get_latest_transaction_country_by_date", summary="Retrieves the country of the most recent transaction made on a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_latest_transaction_country_by_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.Country FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T1.Date = ? ORDER BY T1.Time DESC LIMIT 1", (date,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get distinct currencies used in transactions on a specific date and time
@app.get("/v1/debit_card_specializing/distinct_currencies_by_date_time", operation_id="get_distinct_currencies_by_date_time", summary="Retrieves the unique currencies used in transactions that occurred on a specific date and time. The operation filters transactions based on the provided date and time, and returns the distinct currencies associated with those transactions.")
async def get_distinct_currencies_by_date_time(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), time: str = Query(..., description="Time in 'HH:MM:SS' format")):
    cursor.execute("SELECT DISTINCT T3.Currency FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID INNER JOIN customers AS T3 ON T1.CustomerID = T3.CustomerID WHERE T1.Date = ? AND T1.Time = ?", (date, time))
    result = cursor.fetchall()
    if not result:
        return {"currencies": []}
    return {"currencies": [row[0] for row in result]}

# Endpoint to get the customer segment for transactions on a specific date and time
@app.get("/v1/debit_card_specializing/customer_segment_by_date_time", operation_id="get_customer_segment_by_date_time", summary="Retrieves the customer segment associated with transactions that occurred on a specific date and time. The date and time parameters are required to filter the transactions and determine the corresponding customer segments.")
async def get_customer_segment_by_date_time(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), time: str = Query(..., description="Time in 'HH:MM:SS' format")):
    cursor.execute("SELECT T2.Segment FROM transactions_1k AS T1 INNER JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.date = ? AND T1.time = ?", (date, time))
    result = cursor.fetchall()
    if not result:
        return {"segments": []}
    return {"segments": [row[0] for row in result]}

# Endpoint to get the count of transactions before a specific time on a specific date for a specific currency
@app.get("/v1/debit_card_specializing/transaction_count_before_time_by_date_currency", operation_id="get_transaction_count_before_time_by_date_currency", summary="Retrieves the total number of transactions that occurred before a specific time on a given date for a particular currency. The date and time are provided in 'YYYY-MM-DD' and 'HH:MM:SS' formats, respectively. The currency is specified using its code (e.g., 'CZK').")
async def get_transaction_count_before_time_by_date_currency(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), time: str = Query(..., description="Time in 'HH:MM:SS' format"), currency: str = Query(..., description="Currency code (e.g., 'CZK')")):
    cursor.execute("SELECT COUNT(T1.TransactionID) FROM transactions_1k AS T1 INNER JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Date = ? AND T1.Time < ? AND T2.Currency = ?", (date, time, currency))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer segment of the earliest transaction
@app.get("/v1/debit_card_specializing/earliest_transaction_segment", operation_id="get_earliest_transaction_segment", summary="Retrieves the customer segment associated with the earliest transaction. This operation identifies the first transaction based on the date and returns the corresponding customer segment. The customer segment is determined by joining the transactions and customers tables using the CustomerID.")
async def get_earliest_transaction_segment():
    cursor.execute("SELECT T2.Segment FROM transactions_1k AS T1 INNER JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID ORDER BY Date ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"segment": []}
    return {"segment": result[0]}

# Endpoint to get the country of transactions on a specific date and time
@app.get("/v1/debit_card_specializing/transaction_country_by_date_time", operation_id="get_transaction_country_by_date_time", summary="Retrieves the country of transactions that occurred on a specific date and time. The operation filters transactions by the provided date and time, and returns the corresponding country information from the associated gas station data.")
async def get_transaction_country_by_date_time(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), time: str = Query(..., description="Time in 'HH:MM:SS' format")):
    cursor.execute("SELECT T2.Country FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T1.Date = ? AND T1.Time = ?", (date, time))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the product IDs of transactions on a specific date and time
@app.get("/v1/debit_card_specializing/product_ids_by_date_time", operation_id="get_product_ids_by_date_time", summary="Retrieve the product IDs associated with transactions that occurred on a specific date and time. This operation filters transactions based on the provided date and time, and returns the corresponding product IDs. The date should be in 'YYYY-MM-DD' format, and the time should be in 'HH:MM:SS' format.")
async def get_product_ids_by_date_time(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), time: str = Query(..., description="Time in 'HH:MM:SS' format")):
    cursor.execute("SELECT T1.ProductID FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T1.Date = ? AND T1.Time = ?", (date, time))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get customer transactions based on date, price, and year-month
@app.get("/v1/debit_card_specializing/customer_transactions", operation_id="get_customer_transactions", summary="Retrieves customer transactions based on a specific date, price, and year-month. The operation filters transactions by matching the provided date, price, and year-month with the corresponding transaction details. The result includes the customer ID, transaction date, and consumption amount.")
async def get_customer_transactions(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), price: float = Query(..., description="Price"), year_month: str = Query(..., description="Year-month in 'YYYYMM' format")):
    cursor.execute("SELECT T1.CustomerID, T2.Date, T2.Consumption FROM transactions_1k AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Date = ? AND T1.Price = ? AND T2.Date = ?", (date, price, year_month))
    result = cursor.fetchall()
    if not result:
        return {"transactions": []}
    return {"transactions": result}

# Endpoint to get the count of transactions based on date, time range, and country
@app.get("/v1/debit_card_specializing/transaction_count", operation_id="get_transaction_count", summary="Retrieves the total number of transactions that occurred within a specific date, time range, and country. The date must be provided in 'YYYY-MM-DD' format, while the start and end times should be given in 'HH:MM:SS' format. The country is identified using its code.")
async def get_transaction_count(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), start_time: str = Query(..., description="Start time in 'HH:MM:SS' format"), end_time: str = Query(..., description="End time in 'HH:MM:SS' format"), country: str = Query(..., description="Country code")):
    cursor.execute("SELECT COUNT(T1.TransactionID) FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T1.Date = ? AND T1.Time BETWEEN ? AND ? AND T2.Country = ?", (date, start_time, end_time, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer currency based on date and consumption
@app.get("/v1/debit_card_specializing/customer_currency", operation_id="get_customer_currency", summary="Retrieves the currency associated with a customer based on a specific date and consumption level. The operation uses the provided date in 'YYYYMM' format and consumption level to identify the customer and return their currency.")
async def get_customer_currency(date: str = Query(..., description="Date in 'YYYYMM' format"), consumption: float = Query(..., description="Consumption")):
    cursor.execute("SELECT T2.Currency FROM yearmonth AS T1 INNER JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Date = ? AND T1.Consumption = ?", (date, consumption))
    result = cursor.fetchall()
    if not result:
        return {"currency": []}
    return {"currency": result}

# Endpoint to get the country of a gas station based on card ID
@app.get("/v1/debit_card_specializing/gas_station_country", operation_id="get_gas_station_country", summary="Retrieves the country where a gas station is located, based on the provided card ID. This operation uses the card ID to identify the relevant gas station and returns the country associated with it.")
async def get_gas_station_country(card_id: str = Query(..., description="Card ID")):
    cursor.execute("SELECT T2.Country FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T1.CardID = ?", (card_id,))
    result = cursor.fetchall()
    if not result:
        return {"country": []}
    return {"country": result}

# Endpoint to get the country of a gas station based on date and price
@app.get("/v1/debit_card_specializing/gas_station_country_by_date_price", operation_id="get_gas_station_country_by_date_price", summary="Retrieves the country of a gas station based on a specific date and price. The operation filters transactions by the provided date and price, then identifies the corresponding gas station and returns its country.")
async def get_gas_station_country_by_date_price(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), price: float = Query(..., description="Price")):
    cursor.execute("SELECT T2.Country FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T1.Date = ? AND T1.Price = ?", (date, price))
    result = cursor.fetchall()
    if not result:
        return {"country": []}
    return {"country": result}

# Endpoint to get the percentage of customers using a specific currency on a given date
@app.get("/v1/debit_card_specializing/currency_usage_percentage", operation_id="get_currency_usage_percentage", summary="Retrieves the percentage of customers who used a specific currency on a given date. This operation calculates the proportion of customers who transacted in the provided currency on the specified date, based on the transactions and customer data.")
async def get_currency_usage_percentage(currency: str = Query(..., description="Currency code"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.Currency = ?, 1, 0)) AS FLOAT) * 100 / COUNT(T1.CustomerID) FROM transactions_1k AS T1 INNER JOIN customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Date = ?", (currency, date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the gas station with the highest total price
@app.get("/v1/debit_card_specializing/top_gas_station", operation_id="get_top_gas_station", summary="Retrieves the ID of the gas station that has the highest total transaction price. The operation calculates the sum of all transaction prices for each gas station and returns the ID of the gas station with the highest total.")
async def get_top_gas_station():
    cursor.execute("SELECT GasStationID FROM transactions_1k GROUP BY GasStationID ORDER BY SUM(Price) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"gas_station_id": []}
    return {"gas_station_id": result[0]}

# Endpoint to get the percentage of gas stations in a specific country and segment
@app.get("/v1/debit_card_specializing/gas_station_percentage", operation_id="get_gas_station_percentage", summary="Retrieves the percentage of gas stations in a given country and segment. The calculation is based on the total number of gas stations in the specified country. The country and segment are provided as input parameters.")
async def get_gas_station_percentage(country: str = Query(..., description="Country code"), segment: str = Query(..., description="Segment")):
    cursor.execute("SELECT CAST(SUM(IIF(Country = ? AND Segment = ?, 1, 0)) AS FLOAT) * 100 / SUM(IIF(Country = ?, 1, 0)) FROM gasstations", (country, segment, country))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total price and price for a specific date for a given customer
@app.get("/v1/debit_card_specializing/customer_total_price", operation_id="get_customer_total_price", summary="Retrieves the total price and the price for a specific date for a given customer. The operation calculates the total price by summing up the prices of all transactions made by the customer at various gas stations. It also calculates the price for a specific date by summing up the prices of transactions made by the customer on that date. The date should be provided in 'YYYYMM' format, and the customer ID is required to identify the customer.")
async def get_customer_total_price(date: str = Query(..., description="Date in 'YYYYMM' format"), customer_id: str = Query(..., description="Customer ID")):
    cursor.execute("SELECT SUM(T1.Price) , SUM(IIF(T3.Date = ?, T1.Price, 0)) FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID INNER JOIN yearmonth AS T3 ON T1.CustomerID = T3.CustomerID WHERE T1.CustomerID = ?", (date, customer_id))
    result = cursor.fetchone()
    if not result:
        return {"total_price": [], "price_for_date": []}
    return {"total_price": result[0], "price_for_date": result[1]}

# Endpoint to get product descriptions for top transactions by amount
@app.get("/v1/debit_card_specializing/top_product_descriptions", operation_id="get_top_product_descriptions", summary="Retrieves the descriptions of the products associated with the top transactions, sorted by the transaction amount in descending order. The number of top transactions to retrieve can be specified.")
async def get_top_product_descriptions(limit: int = Query(..., description="Number of top transactions to retrieve")):
    cursor.execute("SELECT T2.Description FROM transactions_1k AS T1 INNER JOIN products AS T2 ON T1.ProductID = T2.ProductID ORDER BY T1.Amount DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get customer details with highest consumption
@app.get("/v1/debit_card_specializing/highest_consumption_customer_details", operation_id="get_highest_consumption_customer_details", summary="Retrieves the details of the customer who has the highest total consumption, including their unique identifier, the total amount spent in their primary currency, and the currency itself. This is determined by summing the price-to-amount ratio of all transactions associated with the customer and grouping the results by customer ID and currency.")
async def get_highest_consumption_customer_details():
    cursor.execute("SELECT T2.CustomerID, SUM(T2.Price / T2.Amount), T1.Currency FROM customers AS T1 INNER JOIN transactions_1k AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.CustomerID = ( SELECT CustomerID FROM yearmonth ORDER BY Consumption DESC LIMIT 1 ) GROUP BY T2.CustomerID, T1.Currency")
    result = cursor.fetchall()
    if not result:
        return {"customer_details": []}
    return {"customer_details": [{"customer_id": row[0], "total_price_per_amount": row[1], "currency": row[2]} for row in result]}

# Endpoint to get the country of the most expensive gas station for a specific product
@app.get("/v1/debit_card_specializing/most_expensive_gas_station_country", operation_id="get_most_expensive_gas_station_country", summary="Retrieves the country of the gas station with the highest price for a specific product. The product is identified by its unique ID. The operation returns the country name of the gas station with the highest recorded price for the specified product.")
async def get_most_expensive_gas_station_country(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T2.Country FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T1.ProductID = ? ORDER BY T1.Price DESC LIMIT 1", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get consumption details based on price per amount, product ID, and date
@app.get("/v1/debit_card_specializing/consumption_details", operation_id="get_consumption_details", summary="Retrieves consumption details for a specific product, filtered by a price per amount threshold and a given date. The operation returns the total consumption for the product, considering only transactions that meet the specified price per amount threshold and occur within the provided date.")
async def get_consumption_details(price_per_amount: float = Query(..., description="Price per amount threshold"), product_id: int = Query(..., description="Product ID"), date: str = Query(..., description="Date in 'YYYYMM' format")):
    cursor.execute("SELECT T2.Consumption FROM transactions_1k AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Price / T1.Amount > ? AND T1.ProductID = ? AND T2.Date = ?", (price_per_amount, product_id, date))
    result = cursor.fetchall()
    if not result:
        return {"consumption": []}
    return {"consumption": [row[0] for row in result]}

api_calls = [
    "/v1/debit_card_specializing/gas_station_count?country=CZE&segment=Premium",
    "/v1/debit_card_specializing/customer_currency_ratio?currency1=EUR&currency2=CZK",
    "/v1/debit_card_specializing/lowest_consumption_customer?segment=LAM&year=2012",
    "/v1/debit_card_specializing/average_monthly_consumption?year=2013&segment=SME",
    "/v1/debit_card_specializing/highest_consumption_customer?currency=CZK&start_date=201101&end_date=201112",
    "/v1/debit_card_specializing/customer_count_below_threshold?segment=KAM&year=2012&threshold=30000",
    "/v1/debit_card_specializing/consumption_difference?currency1=CZK&currency2=EUR&year=2012",
    "/v1/debit_card_specializing/highest_consumption_year?currency=EUR",
    "/v1/debit_card_specializing/lowest_consumption_segment",
    "/v1/debit_card_specializing/highest_consumption_year_currency?currency=CZK",
    "/v1/debit_card_specializing/highest_consumption_month?year=2013&segment=SME",
    "/v1/debit_card_specializing/consumption_difference_segments?segment1=SME&segment2=LAM&segment3=KAM&currency=CZK&start_date=201301&end_date=201312",
    "/v1/debit_card_specializing/consumption_change_percentage?segment1=SME&year1=2013&segment2=LAM&year2=2012&segment3=KAM&year3=2012",
    "/v1/debit_card_specializing/total_consumption_customer_date_range?customer_id=6&start_date=201308&end_date=201311",
    "/v1/debit_card_specializing/gas_station_count_difference?country1=CZE&country2=SVK&segment=Discount",
    "/v1/debit_card_specializing/consumption_difference_customers?customer_id1=7&customer_id2=5&date=201304",
    "/v1/debit_card_specializing/customer_count_difference_currencies?currency1=CZK&currency2=EUR&segment=SME",
    "/v1/debit_card_specializing/highest_total_consumption_customer?segment=KAM",
    "/v1/debit_card_specializing/total_consumption_date_segment?date=201305&segment=KAM",
    "/v1/debit_card_specializing/consumption_percentage_above_threshold?consumption_threshold=46.73&segment=LAM",
    "/v1/debit_card_specializing/top_country_by_gas_stations?segment=Value%20for%20money",
    "/v1/debit_card_specializing/currency_percentage_in_segment?currency=EUR&segment=KAM",
    "/v1/debit_card_specializing/consumption_percentage_above_threshold_by_date?consumption_threshold=528.3&date=201202",
    "/v1/debit_card_specializing/gas_station_percentage_in_segment_by_country?segment=Premium&country=SVK",
    "/v1/debit_card_specializing/top_customer_by_consumption?date=201309",
    "/v1/debit_card_specializing/lowest_consumption_customer_in_segment?date=201206&segment=SME",
    "/v1/debit_card_specializing/highest_monthly_average_consumption?currency=EUR",
    "/v1/debit_card_specializing/product_descriptions_by_date?date=201309",
    "/v1/debit_card_specializing/distinct_countries_by_date?date=201306",
    "/v1/debit_card_specializing/distinct_chain_ids_by_currency?currency=EUR",
    "/v1/debit_card_specializing/distinct_product_ids_descriptions_by_currency?currency=EUR",
    "/v1/debit_card_specializing/average_transaction_amount_by_month_year?date_pattern=2012-01%",
    "/v1/debit_card_specializing/customer_count_by_currency_consumption?currency=EUR&consumption=1000.00",
    "/v1/debit_card_specializing/distinct_product_descriptions_by_country?country=CZE",
    "/v1/debit_card_specializing/distinct_transaction_times_by_chain_id?chain_id=11",
    "/v1/debit_card_specializing/transaction_count_by_country_price?country=CZE&price=1000",
    "/v1/debit_card_specializing/transaction_count_by_country_year?country=CZE&year=2012",
    "/v1/debit_card_specializing/average_price_by_country?country=CZE",
    "/v1/debit_card_specializing/average_price_by_currency?currency=EUR",
    "/v1/debit_card_specializing/top_customer_by_date?date=2012-08-25",
    "/v1/debit_card_specializing/latest_transaction_country_by_date?date=2012-08-25",
    "/v1/debit_card_specializing/distinct_currencies_by_date_time?date=2012-08-24&time=16:25:00",
    "/v1/debit_card_specializing/customer_segment_by_date_time?date=2012-08-23&time=21:20:00",
    "/v1/debit_card_specializing/transaction_count_before_time_by_date_currency?date=2012-08-26&time=13:00:00&currency=CZK",
    "/v1/debit_card_specializing/earliest_transaction_segment",
    "/v1/debit_card_specializing/transaction_country_by_date_time?date=2012-08-24&time=12:42:00",
    "/v1/debit_card_specializing/product_ids_by_date_time?date=2012-08-23&time=21:20:00",
    "/v1/debit_card_specializing/customer_transactions?date=2012-08-24&price=124.05&year_month=201201",
    "/v1/debit_card_specializing/transaction_count?date=2012-08-26&start_time=08:00:00&end_time=09:00:00&country=CZE",
    "/v1/debit_card_specializing/customer_currency?date=201306&consumption=214582.17",
    "/v1/debit_card_specializing/gas_station_country?card_id=667467",
    "/v1/debit_card_specializing/gas_station_country_by_date_price?date=2012-08-24&price=548.4",
    "/v1/debit_card_specializing/currency_usage_percentage?currency=EUR&date=2012-08-25",
    "/v1/debit_card_specializing/top_gas_station",
    "/v1/debit_card_specializing/gas_station_percentage?country=SVK&segment=Premium",
    "/v1/debit_card_specializing/customer_total_price?date=201201&customer_id=38508",
    "/v1/debit_card_specializing/top_product_descriptions?limit=5",
    "/v1/debit_card_specializing/highest_consumption_customer_details",
    "/v1/debit_card_specializing/most_expensive_gas_station_country?product_id=2",
    "/v1/debit_card_specializing/consumption_details?price_per_amount=29.00&product_id=5&date=201208"
]
