from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/sales_in_weather/sales_in_weather.sqlite')
cursor = conn.cursor()

# Endpoint to get units sold on a specific date for a specific store and item
@app.get("/v1/sales_in_weather/units_by_date_store_item", operation_id="get_units_by_date_store_item", summary="Retrieves the number of units sold for a specific product in a particular store on a given date. The operation requires the date, store number, and item number as input parameters to filter the sales data.")
async def get_units_by_date_store_item(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), store_nbr: int = Query(..., description="Store number"), item_nbr: int = Query(..., description="Item number")):
    cursor.execute("SELECT units FROM sales_in_weather WHERE `date` = ? AND store_nbr = ? AND item_nbr = ?", (date, store_nbr, item_nbr))
    result = cursor.fetchone()
    if not result:
        return {"units": []}
    return {"units": result[0]}

# Endpoint to get the total units sold in a specific month and year for a specific store and item
@app.get("/v1/sales_in_weather/total_units_by_month_year_store_item", operation_id="get_total_units_by_month_year_store_item", summary="Retrieves the total number of units sold for a specific item in a given store during a particular month and year. The response is based on the aggregated sales data for the specified parameters.")
async def get_total_units_by_month_year_store_item(month: str = Query(..., description="Month in 'MM' format"), year: str = Query(..., description="Year in 'YYYY' format"), item_nbr: int = Query(..., description="Item number"), store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT SUM(units) FROM sales_in_weather WHERE SUBSTR(`date`, 6, 2) = ? AND SUBSTR(`date`, 1, 4) = ? AND item_nbr = ? AND store_nbr = ?", (month, year, item_nbr, store_nbr))
    result = cursor.fetchone()
    if not result:
        return {"total_units": []}
    return {"total_units": result[0]}

# Endpoint to get the item number with the highest units sold on a specific date for a specific store
@app.get("/v1/sales_in_weather/top_item_by_date_store", operation_id="get_top_item_by_date_store", summary="Retrieves the item number with the highest sales volume for a given date and store. The date should be provided in 'YYYY-MM-DD' format, and the store number is required to identify the specific store. The operation returns the item number with the most units sold on the specified date for the selected store.")
async def get_top_item_by_date_store(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT item_nbr FROM sales_in_weather WHERE `date` = ? AND store_nbr = ? ORDER BY units DESC LIMIT 1", (date, store_nbr))
    result = cursor.fetchone()
    if not result:
        return {"item_nbr": []}
    return {"item_nbr": result[0]}

# Endpoint to get the temperature range for a specific date and station
@app.get("/v1/sales_in_weather/weather/temperature_range_by_date_station", operation_id="get_temperature_range_by_date_station", summary="Retrieves the temperature range (difference between maximum and minimum temperatures) for a specific date and station. The station is identified by its unique number, and the date is provided in 'YYYY-MM-DD' format.")
async def get_temperature_range_by_date_station(station_nbr: int = Query(..., description="Station number"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT tmax - tmin AS temrange FROM weather WHERE station_nbr = ? AND `date` = ?", (station_nbr, date))
    result = cursor.fetchone()
    if not result:
        return {"temrange": []}
    return {"temrange": result[0]}

# Endpoint to get dates with departure greater than a specific value for a specific station
@app.get("/v1/sales_in_weather/weather/dates_with_departure_greater_than", operation_id="get_dates_with_departure_greater_than", summary="Retrieves all dates for a specific weather station where the departure from the long-term average temperature is greater than a specified value.")
async def get_dates_with_departure_greater_than(station_nbr: int = Query(..., description="Station number"), depart: float = Query(..., description="Departure value")):
    cursor.execute("SELECT `date` FROM weather WHERE station_nbr = ? AND depart > ?", (station_nbr, depart))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": [row[0] for row in result]}

# Endpoint to compare average speeds on two dates for a specific station
@app.get("/v1/sales_in_weather/weather/compare_avgspeed_by_dates_station", operation_id="compare_avgspeed_by_dates_station", summary="This operation compares the average speeds at a specific station on two provided dates and returns the date with the higher average speed. The input parameters include two dates in 'YYYY-MM-DD' format and a station number.")
async def compare_avgspeed_by_dates_station(date1: str = Query(..., description="First date in 'YYYY-MM-DD' format"), date2: str = Query(..., description="Second date in 'YYYY-MM-DD' format"), station_nbr: int = Query(..., description="Station number")):
    cursor.execute("SELECT CASE WHEN (SUM(CASE WHEN `date` = ? THEN avgspeed ELSE 0 END) - SUM(CASE WHEN `date` = ? THEN avgspeed ELSE 0 END)) > 0 THEN ? ELSE ? END FROM weather WHERE station_nbr = ?", (date1, date2, date1, date2, station_nbr))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the sum of units sold for a specific store, year, and item when departure is less than a specific value
@app.get("/v1/sales_in_weather/sum_units_by_store_year_item_depart", operation_id="get_sum_units_by_store_year_item_depart", summary="Retrieves the total units sold for a specific store, year, and item, considering only those instances where the departure value is less than the provided threshold. This operation aggregates sales data from the sales_in_weather table, joining it with the relation and weather tables based on store and station numbers. The result is a sum of units sold that meet the specified criteria.")
async def get_sum_units_by_store_year_item_depart(depart: float = Query(..., description="Departure value"), store_nbr: int = Query(..., description="Store number"), year: str = Query(..., description="Year in 'YYYY' format"), item_nbr: int = Query(..., description="Item number")):
    cursor.execute("SELECT SUM(CASE WHEN T3.depart < ? THEN units ELSE 0 END) AS sum FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND SUBSTR(T1.`date`, 1, 4) = ? AND T1.item_nbr = ?", (depart, store_nbr, year, item_nbr))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the units sold for a specific store, year, and item with the highest maximum temperature
@app.get("/v1/sales_in_weather/units_by_store_year_item_max_temp", operation_id="get_units_by_store_year_item_max_temp", summary="Retrieves the total units sold for a specific store, year, and item during the day with the highest maximum temperature. The store is identified by its unique number, the year is specified in 'YYYY' format, and the item is designated by its number. The operation returns the sales data for the hottest day within the provided parameters.")
async def get_units_by_store_year_item_max_temp(store_nbr: int = Query(..., description="Store number"), year: str = Query(..., description="Year in 'YYYY' format"), item_nbr: int = Query(..., description="Item number")):
    cursor.execute("SELECT T1.units FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND SUBSTR(T1.`date`, 1, 4) = ? AND T1.item_nbr = ? ORDER BY tmax DESC LIMIT 1", (store_nbr, year, item_nbr))
    result = cursor.fetchone()
    if not result:
        return {"units": []}
    return {"units": result[0]}

# Endpoint to get the dewpoint for a specific store, year, and item with the highest units sold
@app.get("/v1/sales_in_weather/dewpoint_by_store_year_item_max_units", operation_id="get_dewpoint_by_store_year_item_max_units", summary="Retrieve the dewpoint associated with the highest sales volume for a specific store, year, and item. The operation identifies the maximum units sold for the given store, year, and item, and returns the corresponding dewpoint value from the weather data. This endpoint requires the store number, year (in 'YYYY' format), and item number as input parameters.")
async def get_dewpoint_by_store_year_item_max_units(store_nbr: int = Query(..., description="Store number"), year: str = Query(..., description="Year in 'YYYY' format"), item_nbr: int = Query(..., description="Item number")):
    cursor.execute("SELECT dewpoint FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND SUBSTR(T1.`date`, 1, 4) = ? AND T1.item_nbr = ? ORDER BY units DESC LIMIT 1", (store_nbr, year, item_nbr))
    result = cursor.fetchone()
    if not result:
        return {"dewpoint": []}
    return {"dewpoint": result[0]}

# Endpoint to get the count of days with units sold above a specific value for a specific store, year, item, and maximum temperature
@app.get("/v1/sales_in_weather/count_days_units_above_value", operation_id="get_count_days_units_above_value", summary="Retrieves the total number of days in a given year where a specific item's sales in a particular store surpassed a certain threshold, provided the maximum temperature also exceeded a specified value.")
async def get_count_days_units_above_value(units: int = Query(..., description="Units value"), store_nbr: int = Query(..., description="Store number"), year: str = Query(..., description="Year in 'YYYY' format"), item_nbr: int = Query(..., description="Item number"), tmax: float = Query(..., description="Maximum temperature")):
    cursor.execute("SELECT SUM(CASE WHEN units > ? THEN 1 ELSE 0 END) AS count FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND SUBSTR(T1.`date`, 1, 4) = ? AND T1.item_nbr = ? AND tmax > ?", (units, store_nbr, year, item_nbr, tmax))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the units of a specific item from a specific store based on the maximum temperature difference
@app.get("/v1/sales_in_weather/units_by_max_temp_diff", operation_id="get_units_by_max_temp_diff", summary="Retrieves the quantity of a specific item from a particular store, based on the day with the highest temperature difference. The operation considers the store's location and the weather data to determine the day with the maximum temperature difference. The item quantity for that day is then returned.")
async def get_units_by_max_temp_diff(store_nbr: int = Query(..., description="Store number"), item_nbr: int = Query(..., description="Item number")):
    cursor.execute("SELECT t2.units FROM relation AS T1 INNER JOIN sales_in_weather AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T1.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND T2.item_nbr = ? ORDER BY t3.tmax - t3.tmin DESC LIMIT 1", (store_nbr, item_nbr))
    result = cursor.fetchone()
    if not result:
        return {"units": []}
    return {"units": result[0]}

# Endpoint to get the date of a specific item from a specific store with units greater than a specified value based on the maximum temperature difference
@app.get("/v1/sales_in_weather/date_by_max_temp_diff", operation_id="get_date_by_max_temp_diff", summary="Retrieve the date when a specific item at a particular store had the highest difference in maximum and minimum temperatures, given that the item's sales units surpassed a certain threshold. The input parameters include the store number, item number, and minimum sales units.")
async def get_date_by_max_temp_diff(store_nbr: int = Query(..., description="Store number"), item_nbr: int = Query(..., description="Item number"), units: int = Query(..., description="Units") ):
    cursor.execute("SELECT T2.`date` FROM relation AS T1 INNER JOIN sales_in_weather AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T1.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND T2.item_nbr = ? AND T2.units > ? ORDER BY tmax - tmin DESC LIMIT 1", (store_nbr, item_nbr, units))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the sum of units for a specific item from a specific store where precipitation total is greater than a specified value
@app.get("/v1/sales_in_weather/sum_units_by_precip", operation_id="get_sum_units_by_precip", summary="Retrieves the total number of units sold for a specific item in a particular store when the precipitation total surpasses a given threshold. The calculation considers only the sales data where the precipitation total is higher than the provided value.")
async def get_sum_units_by_precip(store_nbr: int = Query(..., description="Store number"), item_nbr: int = Query(..., description="Item number"), preciptotal: float = Query(..., description="Precipitation total") ):
    cursor.execute("SELECT SUM(CASE WHEN T3.preciptotal > ? THEN units ELSE 0 END) AS sum FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND T1.item_nbr = ?", (preciptotal, store_nbr, item_nbr))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the date of a specific item from a specific store with units and average speed greater than specified values
@app.get("/v1/sales_in_weather/date_by_units_avgspeed", operation_id="get_date_by_units_avgspeed", summary="Get the date of a specific item from a specific store with units and average speed greater than specified values")
async def get_date_by_units_avgspeed(store_nbr: int = Query(..., description="Store number"), item_nbr: int = Query(..., description="Item number"), units: int = Query(..., description="Units"), avgspeed: float = Query(..., description="Average speed") ):
    cursor.execute("SELECT T1.`date` FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND T1.item_nbr = ? AND T1.units > ? AND T3.avgspeed > ?", (store_nbr, item_nbr, units, avgspeed))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the sum of units for a specific store and date pattern based on the maximum temperature
@app.get("/v1/sales_in_weather/sum_units_by_date_pattern", operation_id="get_sum_units_by_date_pattern", summary="Retrieves the total units sold for a specific store and date pattern, filtered by the maximum temperature. The store is identified by its unique number, and the date pattern is specified in 'YYYY' format. The result is sorted by the maximum temperature in descending order and limited to the top record.")
async def get_sum_units_by_date_pattern(store_nbr: int = Query(..., description="Store number"), date_pattern: str = Query(..., description="Date pattern in 'YYYY' format") ):
    cursor.execute("SELECT SUM(units) FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND T1.`date` LIKE ? GROUP BY T3.tmax ORDER BY T3.tmax DESC LIMIT 1", (store_nbr, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the difference in sum of units between two stores for a specific item and date pattern based on the maximum temperature
@app.get("/v1/sales_in_weather/diff_sum_units_by_stores", operation_id="get_diff_sum_units_by_stores", summary="Get the difference in sum of units between two stores for a specific item and date pattern based on the maximum temperature")
async def get_diff_sum_units_by_stores(item_nbr: int = Query(..., description="Item number"), date_pattern: str = Query(..., description="Date pattern in 'YYYY' format"), store_nbr_1: int = Query(..., description="First store number"), store_nbr_2: int = Query(..., description="Second store number") ):
    cursor.execute("( SELECT SUM(units) FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T1.item_nbr = ? AND T1.`date` LIKE ? AND T1.store_nbr = ? GROUP BY tmax ORDER BY T3.tmax DESC LIMIT 1 ) - ( SELECT SUM(units) FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T1.item_nbr = ? AND T1.`date` LIKE ? AND T1.store_nbr = ? GROUP BY tmax ORDER BY T3.tmax DESC LIMIT 1 )", (item_nbr, date_pattern, store_nbr_1, item_nbr, date_pattern, store_nbr_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the item number with the highest sum of units for a specific store and date pattern based on the maximum temperature
@app.get("/v1/sales_in_weather/item_nbr_by_max_temp", operation_id="get_item_nbr_by_max_temp", summary="Retrieves the item number with the highest total units sold for a specific store and date pattern, based on the maximum temperature recorded on that date.")
async def get_item_nbr_by_max_temp(store_nbr: int = Query(..., description="Store number"), date_pattern: str = Query(..., description="Date pattern in 'YYYY' format") ):
    cursor.execute("SELECT T1.item_nbr FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T1.store_nbr = ? AND T1.`date` LIKE ? AND tmax = ( SELECT MAX(tmax) FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T1.store_nbr = ? AND T1.`date` LIKE ? ) GROUP BY T1.item_nbr ORDER BY SUM(units) DESC LIMIT 1", (store_nbr, date_pattern, store_nbr, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"item_nbr": []}
    return {"item_nbr": result[0]}

# Endpoint to get the count of distinct items with zero units for a specific store and year based on the maximum temperature
@app.get("/v1/sales_in_weather/count_distinct_items_zero_units", operation_id="get_count_distinct_items_zero_units", summary="Retrieves the count of unique items that have not been sold (zero units) in a specific store during a given year, based on the highest recorded temperature. The result is filtered by the provided store number, year, and units.")
async def get_count_distinct_items_zero_units(store_nbr: int = Query(..., description="Store number"), year: str = Query(..., description="Year in 'YYYY' format"), units: int = Query(..., description="Units") ):
    cursor.execute("SELECT COUNT(DISTINCT T1.item_nbr) FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr AND T1.store_nbr = ? AND SUBSTR(T1.`date`, 1, 4) = ? AND T1.units = ? GROUP BY T3.tmax ORDER BY T3.tmax DESC LIMIT 1", (store_nbr, year, units))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average units per date for a specific item and store where the maximum temperature is greater than a specified value
@app.get("/v1/sales_in_weather/avg_units_per_date", operation_id="get_avg_units_per_date", summary="Retrieves the average number of units sold per date for a specific item in a given store, considering only dates where the maximum temperature exceeded a specified value.")
async def get_avg_units_per_date(store_nbr: int = Query(..., description="Store number"), item_nbr: int = Query(..., description="Item number"), tmax: float = Query(..., description="Maximum temperature") ):
    cursor.execute("SELECT CAST(SUM(T1.units) AS REAL) / COUNT(T1.`date`) FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T1.store_nbr = ? AND T1.item_nbr = ? AND T3.tmax > ?", (store_nbr, item_nbr, tmax))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of units for a specific item and store based on the maximum temperature for a given date pattern
@app.get("/v1/sales_in_weather/percentage_units_by_max_temp", operation_id="get_percentage_units_by_max_temp", summary="Retrieves the percentage of units sold for a specific item in a given store, based on the maximum temperature recorded on days that match a specified date pattern. The calculation considers all sales data and weather information for the store.")
async def get_percentage_units_by_max_temp(item_nbr: int = Query(..., description="Item number"), store_nbr: int = Query(..., description="Store number"), date_pattern: str = Query(..., description="Date pattern in 'YYYY' format") ):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.item_nbr = ? THEN units * 1 ELSE 0 END) AS REAL) * 100 / SUM(units) FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T1.store_nbr = ? AND T1.`date` LIKE ? AND T3.tmax = ( SELECT MAX(T3.tmax) FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T1.store_nbr = ? AND T1.`date` LIKE ? )", (item_nbr, store_nbr, date_pattern, store_nbr, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average temperature on a specific date for a specific weather station
@app.get("/v1/sales_in_weather/average_temperature", operation_id="get_average_temperature", summary="Retrieves the average temperature recorded on a specific date for a given weather station. The operation requires the date in 'YYYY-MM-DD' format and the station number as input parameters.")
async def get_average_temperature(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), station_nbr: int = Query(..., description="Station number")):
    cursor.execute("SELECT tavg FROM weather WHERE `date` = ? AND station_nbr = ?", (date, station_nbr))
    result = cursor.fetchone()
    if not result:
        return {"tavg": []}
    return {"tavg": result[0]}

# Endpoint to get the resultant wind speed on a specific date for a specific weather station
@app.get("/v1/sales_in_weather/resultant_wind_speed", operation_id="get_resultant_wind_speed", summary="Retrieves the resultant wind speed recorded on a specific date for a given weather station. The operation requires the date in 'YYYY-MM-DD' format and the station number as input parameters.")
async def get_resultant_wind_speed(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), station_nbr: int = Query(..., description="Station number")):
    cursor.execute("SELECT resultspeed FROM weather WHERE `date` = ? AND station_nbr = ?", (date, station_nbr))
    result = cursor.fetchone()
    if not result:
        return {"resultspeed": []}
    return {"resultspeed": result[0]}

# Endpoint to get the station number with the most relations
@app.get("/v1/sales_in_weather/most_related_station", operation_id="get_most_related_station", summary="Retrieves the station number that has the highest number of relations. This operation returns the station with the most connections, providing insights into the most interconnected station in the dataset.")
async def get_most_related_station():
    cursor.execute("SELECT station_nbr FROM relation GROUP BY station_nbr ORDER BY COUNT(station_nbr) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"station_nbr": []}
    return {"station_nbr": result[0]}

# Endpoint to get the station numbers related to a specific store
@app.get("/v1/sales_in_weather/related_stations", operation_id="get_related_stations", summary="Retrieves the station numbers associated with a given store. The operation uses the provided store number to look up related station numbers in the database.")
async def get_related_stations(store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT station_nbr FROM relation WHERE store_nbr = ?", (store_nbr,))
    result = cursor.fetchall()
    if not result:
        return {"station_nbr": []}
    return {"station_nbr": [row[0] for row in result]}

# Endpoint to get the temperature range for a specific store on a specific date
@app.get("/v1/sales_in_weather/temperature_range", operation_id="get_temperature_range", summary="Retrieves the temperature range for a specific store on a given date. The temperature range is calculated as the difference between the maximum and minimum temperatures recorded on that date. The store is identified by its unique store number, and the date is provided in the 'YYYY-MM-DD' format.")
async def get_temperature_range(store_nbr: int = Query(..., description="Store number"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.tmax - T1.tmin AS temprange FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T2.store_nbr = ? AND T1.`date` = ?", (store_nbr, date))
    result = cursor.fetchone()
    if not result:
        return {"temprange": []}
    return {"temprange": result[0]}

# Endpoint to get the store numbers related to the station with the highest departure
@app.get("/v1/sales_in_weather/stores_highest_departure", operation_id="get_stores_highest_departure", summary="Retrieves the store numbers associated with the weather station that has the highest recorded temperature departure from the average. This operation identifies the station with the most extreme temperature deviation and returns the corresponding store numbers.")
async def get_stores_highest_departure():
    cursor.execute("SELECT store_nbr FROM relation WHERE station_nbr = ( SELECT station_nbr FROM weather ORDER BY depart DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"store_nbr": []}
    return {"store_nbr": [row[0] for row in result]}

# Endpoint to get the dew point for a specific store on a specific date
@app.get("/v1/sales_in_weather/dew_point", operation_id="get_dew_point", summary="Retrieves the dew point for a specific store on a given date. The operation requires the store number and the date in 'YYYY-MM-DD' format as input parameters. The dew point data is fetched from the weather table, which is joined with the relation table using the station number to ensure accurate results.")
async def get_dew_point(store_nbr: int = Query(..., description="Store number"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.dewpoint FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T2.store_nbr = ? AND T1.`date` = ?", (store_nbr, date))
    result = cursor.fetchone()
    if not result:
        return {"dewpoint": []}
    return {"dewpoint": result[0]}

# Endpoint to get the wet bulb temperature for a specific store on a specific date
@app.get("/v1/sales_in_weather/wet_bulb_temperature", operation_id="get_wet_bulb_temperature", summary="Retrieves the wet bulb temperature for a specific store on a given date. The operation requires the store number and the date in 'YYYY-MM-DD' format as input parameters. The wet bulb temperature is determined by cross-referencing weather data with store-weather station relations.")
async def get_wet_bulb_temperature(store_nbr: int = Query(..., description="Store number"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.wetbulb FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T2.store_nbr = ? AND T1.`date` = ?", (store_nbr, date))
    result = cursor.fetchone()
    if not result:
        return {"wetbulb": []}
    return {"wetbulb": result[0]}

# Endpoint to get the count of distinct stores related to the station with the highest average wind speed
@app.get("/v1/sales_in_weather/count_stores_highest_avgspeed", operation_id="get_count_stores_highest_avgspeed", summary="Retrieves the total number of unique stores associated with the weather station that has the highest average wind speed. This operation does not require any input parameters and provides a single numerical value as output.")
async def get_count_stores_highest_avgspeed():
    cursor.execute("SELECT COUNT(T.store_nbr) FROM ( SELECT DISTINCT store_nbr FROM relation WHERE station_nbr = ( SELECT station_nbr FROM weather ORDER BY avgspeed DESC LIMIT 1 ) ) T")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum temperature for a specific store on a specific date
@app.get("/v1/sales_in_weather/tmax_by_store_date", operation_id="get_tmax_by_store_date", summary="Retrieves the highest recorded temperature for a given store on a specific date. The operation uses the provided store number and date to filter the data and return the maximum temperature value.")
async def get_tmax_by_store_date(store_nbr: int = Query(..., description="Store number"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT tmax FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T2.store_nbr = ? AND T1.`date` = ?", (store_nbr, date))
    result = cursor.fetchone()
    if not result:
        return {"tmax": []}
    return {"tmax": result[0]}

# Endpoint to get the sunrise time for a specific store on a specific date
@app.get("/v1/sales_in_weather/sunrise_by_store_date", operation_id="get_sunrise_by_store_date", summary="Retrieves the sunrise time for a given store on a specific date. This operation requires a date in 'YYYY-MM-DD' format and a store number as input parameters. The date and store number are used to filter the weather data and the store-weather station relationship data, respectively. The result is the sunrise time for the specified store on the provided date.")
async def get_sunrise_by_store_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT T1.sunrise FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T1.`date` = ? AND store_nbr = ?", (date, store_nbr))
    result = cursor.fetchone()
    if not result:
        return {"sunrise": []}
    return {"sunrise": result[0]}

# Endpoint to get the store number with the highest snowfall
@app.get("/v1/sales_in_weather/store_with_highest_snowfall", operation_id="get_store_with_highest_snowfall", summary="Retrieves the store number that has experienced the highest amount of snowfall. This operation utilizes weather data and a relation table to determine the store with the most snowfall. The result is a single store number.")
async def get_store_with_highest_snowfall():
    cursor.execute("SELECT T2.store_nbr FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr ORDER BY snowfall DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"store_nbr": []}
    return {"store_nbr": result[0]}

# Endpoint to get the codesum for a specific store on a specific date
@app.get("/v1/sales_in_weather/codesum_by_store_date", operation_id="get_codesum_by_store_date", summary="Retrieves the weather codesum for a given store on a specific date. The operation requires a date in 'YYYY-MM-DD' format and a store number as input parameters. The result is a single codesum value that represents the weather conditions for the provided store and date.")
async def get_codesum_by_store_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT T1.codesum FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T1.`date` = ? AND T2.store_nbr = ?", (date, store_nbr))
    result = cursor.fetchone()
    if not result:
        return {"codesum": []}
    return {"codesum": result[0]}

# Endpoint to get the sea level for a specific store on a specific date
@app.get("/v1/sales_in_weather/sealevel_by_store_date", operation_id="get_sealevel_by_store_date", summary="Retrieves the sea level data for a specific store on a given date. The operation requires a date in 'YYYY-MM-DD' format and a store number as input parameters. These parameters are used to filter the data and return the sea level information for the specified store and date.")
async def get_sealevel_by_store_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT T1.sealevel FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T1.`date` = ? AND T2.store_nbr = ?", (date, store_nbr))
    result = cursor.fetchone()
    if not result:
        return {"sealevel": []}
    return {"sealevel": result[0]}

# Endpoint to get the total precipitation for a specific store on a specific date
@app.get("/v1/sales_in_weather/preciptotal_by_store_date", operation_id="get_preciptotal_by_store_date", summary="Retrieves the total precipitation for a given store on a specific date. The operation requires a date in 'YYYY-MM-DD' format and a store number as input parameters. The data is fetched from a weather database, which is linked to a store database using a station number.")
async def get_preciptotal_by_store_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT T1.preciptotal FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T1.`date` = ? AND T2.store_nbr = ?", (date, store_nbr))
    result = cursor.fetchone()
    if not result:
        return {"preciptotal": []}
    return {"preciptotal": result[0]}

# Endpoint to get the station pressure for a specific store on a specific date
@app.get("/v1/sales_in_weather/stnpressure_by_store_date", operation_id="get_stnpressure_by_store_date", summary="Retrieves the station pressure for a given store on a specific date. This operation requires a date in 'YYYY-MM-DD' format and a store number as input parameters. The provided inputs are used to filter the weather data and return the corresponding station pressure.")
async def get_stnpressure_by_store_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT T1.stnpressure FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T1.`date` = ? AND T2.store_nbr = ?", (date, store_nbr))
    result = cursor.fetchone()
    if not result:
        return {"stnpressure": []}
    return {"stnpressure": result[0]}

# Endpoint to get the percentage of units sold by a specific store on a specific date
@app.get("/v1/sales_in_weather/percentage_units_sold_by_store_date", operation_id="get_percentage_units_sold_by_store_date", summary="Retrieves the percentage of total units sold for a specific store on a given date. The calculation is based on the total units sold across all stores on the same date. The input parameters include the store number and the date in 'YYYY-MM-DD' format.")
async def get_percentage_units_sold_by_store_date(store_nbr: int = Query(..., description="Store number"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.store_nbr = ? THEN units * 1 ELSE 0 END) AS REAL) * 100 / SUM(units) FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr WHERE T1.`date` = ?", (store_nbr, date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage change in average temperature between two dates for a specific store
@app.get("/v1/sales_in_weather/percentage_change_tavg_by_store_dates", operation_id="get_percentage_change_tavg_by_store_dates", summary="Retrieves the percentage change in average temperature for a specific store between two provided dates. The calculation is based on the daily average temperature data from the weather station nearest to the store.")
async def get_percentage_change_tavg_by_store_dates(date1: str = Query(..., description="First date in 'YYYY-MM-DD' format"), date2: str = Query(..., description="Second date in 'YYYY-MM-DD' format"), store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T1.`date` = ? THEN T1.tavg * 1 ELSE 0 END) - SUM(CASE WHEN T1.`date` = ? THEN T1.tavg * 1 ELSE 0 END)) AS REAL) * 100 / SUM(CASE WHEN T1.`date` = ? THEN T1.tavg * 1 ELSE 0 END) FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T2.store_nbr = ?", (date1, date2, date2, store_nbr))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result[0]}

# Endpoint to get the sum of store numbers for a specific station
@app.get("/v1/sales_in_weather/sum_store_nbr_by_station", operation_id="get_sum_store_nbr_by_station", summary="Retrieves the total number of stores associated with a given station. The station is identified by its unique station number.")
async def get_sum_store_nbr_by_station(station_nbr: int = Query(..., description="Station number")):
    cursor.execute("SELECT SUM(store_nbr) FROM relation WHERE station_nbr = ?", (station_nbr,))
    result = cursor.fetchone()
    if not result:
        return {"sum_store_nbr": []}
    return {"sum_store_nbr": result[0]}

# Endpoint to get the count of items based on store number, units, and date
@app.get("/v1/sales_in_weather/count_items_by_store_units_date", operation_id="get_count_items", summary="Retrieves the total number of items sold in a particular store, with specific units, on a given date. The input parameters include the store number, units, and date in 'YYYY-MM-DD' format.")
async def get_count_items(store_nbr: int = Query(..., description="Store number"), units: int = Query(..., description="Units"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(item_nbr) FROM sales_in_weather WHERE store_nbr = ? AND units = ? AND `date` = ?", (store_nbr, units, date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the date with the highest maximum temperature within a specific year range for a given station
@app.get("/v1/sales_in_weather/weather/date_with_highest_tmax_in_year_range", operation_id="get_date_with_highest_tmax", summary="Retrieves the date with the highest recorded maximum temperature within a specified year range for a particular weather station. The operation requires the station number, start year, and end year as input parameters.")
async def get_date_with_highest_tmax(station_nbr: int = Query(..., description="Station number"), start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT `date` FROM weather WHERE station_nbr = ? AND CAST(SUBSTR(`date`, 1, 4) AS int) BETWEEN ? AND ? ORDER BY tmax DESC LIMIT 1", (station_nbr, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the station number with the highest number of associated stores
@app.get("/v1/sales_in_weather/relation/station_with_most_stores", operation_id="get_station_with_most_stores", summary="Retrieves the station number that is associated with the highest number of stores. This operation returns the station number that has the most stores linked to it, based on the data in the 'relation' table.")
async def get_station_with_most_stores():
    cursor.execute("SELECT station_nbr FROM relation GROUP BY station_nbr ORDER BY COUNT(store_nbr) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"station_nbr": []}
    return {"station_nbr": result[0]}

# Endpoint to get the station number with the most days of negative departure in a specific month and year
@app.get("/v1/sales_in_weather/weather/station_with_most_negative_departure_days", operation_id="get_station_with_most_negative_departure_days", summary="Get the station number with the most days of negative departure in a specific month and year")
async def get_station_with_most_negative_departure_days(year: str = Query(..., description="Year in 'YYYY' format"), month: str = Query(..., description="Month in 'MM' format")):
    cursor.execute("SELECT station_nbr FROM weather WHERE SUBSTR(`date`, 1, 4) = ? AND SUBSTR(`date`, 6, 2) = ? AND depart < 0 GROUP BY station_nbr HAVING COUNT(DISTINCT `date`) = ( SELECT COUNT(DISTINCT `date`) FROM weather WHERE SUBSTR(`date`, 1, 4) = ? AND SUBSTR(`date`, 6, 2) = ? AND depart < 0 GROUP BY station_nbr ORDER BY COUNT(`date`) DESC LIMIT 1 )", (year, month, year, month))
    result = cursor.fetchone()
    if not result:
        return {"station_nbr": []}
    return {"station_nbr": result[0]}

# Endpoint to get the station number with the highest total units sold for a specific item
@app.get("/v1/sales_in_weather/station_with_highest_units_sold", operation_id="get_station_with_highest_units_sold", summary="Retrieves the station number that has recorded the highest total sales for a specific item. This operation considers the sales data from all stores associated with each station, as determined by the relation table. The item number is required to identify the relevant sales data.")
async def get_station_with_highest_units_sold(item_nbr: int = Query(..., description="Item number")):
    cursor.execute("SELECT station_nbr FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr WHERE T1.item_nbr = ? GROUP BY T2.station_nbr ORDER BY SUM(T1.units) DESC LIMIT 1", (item_nbr,))
    result = cursor.fetchone()
    if not result:
        return {"station_nbr": []}
    return {"station_nbr": result[0]}

# Endpoint to get the count of stores associated with the station having the highest average wind speed
@app.get("/v1/sales_in_weather/relation/count_stores_with_highest_avgspeed", operation_id="get_count_stores_with_highest_avgspeed", summary="Retrieves the total number of stores linked to the weather station with the highest average wind speed. This operation provides a quantitative measure of the stores' distribution in relation to the most windy station.")
async def get_count_stores_with_highest_avgspeed():
    cursor.execute("SELECT COUNT(store_nbr) FROM relation WHERE station_nbr = ( SELECT station_nbr FROM weather ORDER BY avgspeed DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get store numbers based on station number, date pattern, item number, and minimum units sold
@app.get("/v1/sales_in_weather/store_numbers_by_criteria", operation_id="get_store_numbers_by_criteria", summary="Retrieves the store numbers that meet the specified criteria, including a particular station number, date pattern, item number, and minimum units sold. The data returned is based on sales information and weather conditions, ensuring that the store numbers provided have met the specified sales threshold for the given item during the specified date range and station location.")
async def get_store_numbers_by_criteria(station_nbr: int = Query(..., description="Station number"), date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM' format"), item_nbr: int = Query(..., description="Item number"), min_units: int = Query(..., description="Minimum units sold")):
    cursor.execute("SELECT T1.store_nbr FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr WHERE T2.station_nbr = ? AND T1.`date` LIKE ? AND T1.item_nbr = ? AND units >= ?", (station_nbr, date_pattern, item_nbr, min_units))
    result = cursor.fetchall()
    if not result:
        return {"store_nbr": []}
    return {"store_nbr": [row[0] for row in result]}

# Endpoint to get the item number with the highest units sold for a specific station, date pattern, and weather code
@app.get("/v1/sales_in_weather/item_with_highest_units_sold", operation_id="get_item_with_highest_units_sold", summary="Retrieves the item number that has sold the most units for a given station, date pattern, and weather code. The item number is determined by comparing sales data from the specified station and date pattern, filtered by the provided weather code. The result is the item number with the highest sales volume.")
async def get_item_with_highest_units_sold(station_nbr: int = Query(..., description="Station number"), date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM' format"), codesum: str = Query(..., description="Weather code")):
    cursor.execute("SELECT T1.item_nbr FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN weather AS T3 ON T2.station_nbr = T3.station_nbr WHERE T3.station_nbr = ? AND T1.`date` LIKE ? AND codesum = ? ORDER BY T1.units DESC LIMIT 1", (station_nbr, date_pattern, codesum))
    result = cursor.fetchone()
    if not result:
        return {"item_nbr": []}
    return {"item_nbr": result[0]}

# Endpoint to get the station number with the highest total units sold for a specific item
@app.get("/v1/sales_in_weather/station_with_highest_units_sold_for_item", operation_id="get_station_with_highest_units_sold_for_item", summary="Retrieves the station number that has recorded the highest total sales for a specific item. The operation considers sales data from all stores associated with each station and sums up the units sold for the specified item. The station with the maximum total units sold is returned.")
async def get_station_with_highest_units_sold_for_item(item_nbr: int = Query(..., description="Item number")):
    cursor.execute("SELECT T2.station_nbr FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr WHERE T1.item_nbr = ? GROUP BY T2.station_nbr ORDER BY SUM(T1.units) DESC LIMIT 1", (item_nbr,))
    result = cursor.fetchone()
    if not result:
        return {"station_nbr": []}
    return {"station_nbr": result[0]}

# Endpoint to get the station number with the earliest sunrise for a specific date pattern and having exactly one associated store
@app.get("/v1/sales_in_weather/relation/station_with_earliest_sunrise", operation_id="get_station_with_earliest_sunrise", summary="Retrieves the station number with the earliest sunrise for a given date pattern, considering only stations associated with exactly one store. The date pattern should be provided in 'YYYY-MM' format.")
async def get_station_with_earliest_sunrise(date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM' format")):
    cursor.execute("SELECT T1.station_nbr FROM relation AS T1 INNER JOIN weather AS T2 ON T1.station_nbr = T2.station_nbr WHERE sunrise IS NOT NULL AND T2.`date` LIKE ? AND T1.station_nbr IN ( SELECT station_nbr FROM relation GROUP BY station_nbr HAVING COUNT(store_nbr) = 1 ) ORDER BY sunrise LIMIT 1", (date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"station_nbr": []}
    return {"station_nbr": result[0]}

# Endpoint to get the store number with the highest units sold for a specific item, station, and date pattern
@app.get("/v1/sales_in_weather/top_store_by_item_station_date", operation_id="get_top_store_by_item_station_date", summary="Retrieves the store with the highest sales for a specific item and station during a given month. The store is identified by its unique store number. The input parameters include the item number, station number, and a date pattern in 'YYYY-MM' format.")
async def get_top_store_by_item_station_date(item_nbr: int = Query(..., description="Item number"), station_nbr: int = Query(..., description="Station number"), date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM' format")):
    cursor.execute("SELECT T1.store_nbr FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr WHERE T1.item_nbr = ? AND T2.station_nbr = ? AND T1.`date` LIKE ? GROUP BY T1.store_nbr ORDER BY SUM(T1.units) DESC LIMIT 1", (item_nbr, station_nbr, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"store_nbr": []}
    return {"store_nbr": result[0]}

# Endpoint to get item numbers from sales data associated with the station having the highest snowfall
@app.get("/v1/sales_in_weather/items_by_highest_snowfall_station", operation_id="get_items_by_highest_snowfall_station", summary="Retrieves the item numbers from sales data linked to the store located at the station with the highest recorded snowfall. This operation identifies the station with the maximum snowfall and returns the corresponding sales data.")
async def get_items_by_highest_snowfall_station():
    cursor.execute("SELECT T1.item_nbr FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr INNER JOIN ( SELECT station_nbr, `date` FROM weather ORDER BY snowfall DESC LIMIT 1 ) AS T3 ON T2.station_nbr = T3.station_nbr")
    result = cursor.fetchall()
    if not result:
        return {"item_nbrs": []}
    return {"item_nbrs": [row[0] for row in result]}

# Endpoint to get the top 3 stations by units sold
@app.get("/v1/sales_in_weather/top_stations_by_units", operation_id="get_top_stations_by_units", summary="Retrieves the top three stations with the highest sales volume. The stations are identified by their unique station numbers and are ranked based on the total units sold. The data is sourced from the sales_in_weather table and cross-referenced with the relation table to ensure accurate station identification.")
async def get_top_stations_by_units():
    cursor.execute("SELECT T2.station_nbr FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr ORDER BY T1.units DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"station_nbrs": []}
    return {"station_nbrs": [row[0] for row in result]}

# Endpoint to get the count of stores associated with the station having the highest heat
@app.get("/v1/sales_in_weather/store_count_by_highest_heat_station", operation_id="get_store_count_by_highest_heat_station", summary="Retrieves the total number of stores linked to the weather station with the highest recorded heat. This operation does not require any input parameters and returns a single integer value.")
async def get_store_count_by_highest_heat_station():
    cursor.execute("SELECT COUNT(T2.store_nbr) FROM ( SELECT station_nbr FROM weather ORDER BY heat DESC LIMIT 1 ) AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the minimum temperature for a specific store and date
@app.get("/v1/sales_in_weather/min_temperature_by_store_date", operation_id="get_min_temperature_by_store_date", summary="Retrieves the minimum temperature recorded for a specific store on a given date. The operation uses the provided store number and date in 'YYYY-MM-DD' format to filter the data and return the minimum temperature value.")
async def get_min_temperature_by_store_date(store_nbr: int = Query(..., description="Store number"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT tmin FROM relation AS T1 INNER JOIN weather AS T2 ON T1.station_nbr = T2.station_nbr WHERE T1.store_nbr = ? AND T2.`date` = ?", (store_nbr, date))
    result = cursor.fetchone()
    if not result:
        return {"tmin": []}
    return {"tmin": result[0]}

# Endpoint to get the count of stations with specific pressure and store count on a given date
@app.get("/v1/sales_in_weather/station_count_by_pressure_store_count_date", operation_id="get_station_count_by_pressure_store_count_date", summary="Retrieves the number of weather stations that have a specific pressure and a certain number of associated stores, on a given date. The date, station pressure, and store count are provided as input parameters.")
async def get_station_count_by_pressure_store_count_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), stnpressure: int = Query(..., description="Station pressure"), store_count: int = Query(..., description="Number of stores")):
    cursor.execute("SELECT COUNT(station_nbr) FROM weather WHERE `date` = ? AND stnpressure < ? AND station_nbr IN ( SELECT station_nbr FROM relation GROUP BY station_nbr HAVING COUNT(store_nbr) = ? )", (date, stnpressure, store_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average maximum temperature for the station with the most stores in a specific month
@app.get("/v1/sales_in_weather/avg_max_temp_by_most_stores_month", operation_id="get_avg_max_temp_by_most_stores_month", summary="Retrieves the average maximum temperature for the month specified in the 'YYYY-MM' format. The data is sourced from the station with the highest number of stores.")
async def get_avg_max_temp_by_most_stores_month(month: str = Query(..., description="Month in 'YYYY-MM' format")):
    cursor.execute("SELECT CAST(SUM(T2.tmax) AS REAL) / 29 FROM ( SELECT station_nbr FROM relation GROUP BY station_nbr ORDER BY COUNT(store_nbr) DESC LIMIT 1 ) AS T1 INNER JOIN weather AS T2 ON T1.station_nbr = T2.station_nbr WHERE SUBSTR(T2.`date`, 1, 7) = ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"avg_max_temp": []}
    return {"avg_max_temp": result[0]}

# Endpoint to get the percentage of units sold by a specific store for a given station, item, and date pattern
@app.get("/v1/sales_in_weather/percentage_units_sold_by_store", operation_id="get_percentage_units_sold_by_store", summary="Retrieves the percentage of units sold by a specific store for a given station, item, and date pattern. The calculation is based on the total units sold for the specified station and item, filtered by the provided date pattern. The store's sales are compared to the overall sales to determine the percentage.")
async def get_percentage_units_sold_by_store(store_nbr: int = Query(..., description="Store number"), station_nbr: int = Query(..., description="Station number"), item_nbr: int = Query(..., description="Item number"), date_pattern: str = Query(..., description="Date pattern in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.store_nbr = ? THEN units * 1 ELSE 0 END) AS REAL) * 100 / SUM(units) FROM relation AS T1 INNER JOIN sales_in_weather AS T2 ON T1.store_nbr = T2.store_nbr WHERE station_nbr = ? AND item_nbr = ? AND T2.`date` LIKE ?", (store_nbr, station_nbr, item_nbr, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the maximum average speed from weather data
@app.get("/v1/sales_in_weather/max_avg_speed", operation_id="get_max_avg_speed", summary="Retrieves the highest average speed value from the weather data.")
async def get_max_avg_speed():
    cursor.execute("SELECT MAX(avgspeed) FROM weather")
    result = cursor.fetchone()
    if not result:
        return {"max_avg_speed": []}
    return {"max_avg_speed": result[0]}

# Endpoint to get the count of distinct dates with snowfall greater than a specified value
@app.get("/v1/sales_in_weather/count_dates_by_snowfall", operation_id="get_count_dates_by_snowfall", summary="Retrieve the number of unique dates where the snowfall surpasses a given threshold. This operation provides a count of days with snowfall greater than the specified value, offering insights into weather patterns and their potential impact on sales.")
async def get_count_dates_by_snowfall(snowfall: float = Query(..., description="Snowfall value")):
    cursor.execute("SELECT COUNT(DISTINCT `date`) FROM weather WHERE snowfall > ?", (snowfall,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct dates where sunrise is before a given time
@app.get("/v1/sales_in_weather/count_days_before_sunrise", operation_id="get_count_days_before_sunrise", summary="Retrieves the number of unique days where the sunrise occurred before the specified time. The time should be provided in 'HH:MM:SS' format.")
async def get_count_days_before_sunrise(sunrise_time: str = Query(..., description="Time in 'HH:MM:SS' format")):
    cursor.execute("SELECT COUNT(DISTINCT `date`) AS days FROM weather WHERE sunrise < time(?)", (sunrise_time,))
    result = cursor.fetchone()
    if not result:
        return {"days": []}
    return {"days": result[0]}

# Endpoint to get the minimum dewpoint from the weather data
@app.get("/v1/sales_in_weather/min_dewpoint", operation_id="get_min_dewpoint", summary="Retrieves the lowest recorded dewpoint from the weather data.")
async def get_min_dewpoint():
    cursor.execute("SELECT MIN(dewpoint) FROM weather")
    result = cursor.fetchone()
    if not result:
        return {"min_dewpoint": []}
    return {"min_dewpoint": result[0]}

# Endpoint to get the maximum and minimum temperatures for a specific station and date
@app.get("/v1/sales_in_weather/temperature_by_station_date", operation_id="get_temperature_by_station_date", summary="Retrieves the maximum and minimum temperatures recorded for a specific weather station on a given date. The station is identified by its unique number, and the date is provided in 'YYYY-MM-DD' format. This operation returns the highest and lowest temperatures observed on that day at the specified station.")
async def get_temperature_by_station_date(station_nbr: int = Query(..., description="Station number"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT tmax, tmin FROM weather WHERE station_nbr = ? AND `date` = ?", (station_nbr, date))
    result = cursor.fetchone()
    if not result:
        return {"tmax": [], "tmin": []}
    return {"tmax": result[0], "tmin": result[1]}

# Endpoint to get the count of distinct station numbers for a specific month and item number
@app.get("/v1/sales_in_weather/count_stations_by_month_item", operation_id="get_count_stations_by_month_item", summary="Retrieves the total number of unique sales stations for a specific item during a given month. The month should be provided in 'YYYY-MM' format, and the item number should be specified.")
async def get_count_stations_by_month_item(month: str = Query(..., description="Month in 'YYYY-MM' format"), item_nbr: int = Query(..., description="Item number")):
    cursor.execute("SELECT COUNT(DISTINCT T2.station_nbr) AS number FROM sales_in_weather AS T1 INNER JOIN relation AS T2 ON T1.store_nbr = T2.store_nbr WHERE SUBSTR(`date`, 1, 7) = ? AND item_nbr = ?", (month, item_nbr))
    result = cursor.fetchone()
    if not result:
        return {"number": []}
    return {"number": result[0]}

# Endpoint to get the total units for a specific store, item, and snowfall condition
@app.get("/v1/sales_in_weather/total_units_by_store_item_snowfall", operation_id="get_total_units_by_store_item_snowfall", summary="Retrieves the total number of units sold for a specific store and item, considering only the days with snowfall less than the provided maximum value.")
async def get_total_units_by_store_item_snowfall(store_nbr: int = Query(..., description="Store number"), item_nbr: int = Query(..., description="Item number"), snowfall: float = Query(..., description="Maximum snowfall")):
    cursor.execute("SELECT SUM(units) FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr INNER JOIN sales_in_weather AS T3 ON T2.store_nbr = T3.store_nbr WHERE T2.store_nbr = ? AND T3.item_nbr = ? AND T1.snowfall < ?", (store_nbr, item_nbr, snowfall))
    result = cursor.fetchone()
    if not result:
        return {"total_units": []}
    return {"total_units": result[0]}

# Endpoint to get the count of distinct items for a specific store with non-zero snowfall
@app.get("/v1/sales_in_weather/count_items_by_store_snowfall", operation_id="get_count_items_by_store_snowfall", summary="Retrieves the total number of unique items sold at a specific store during periods of non-zero snowfall. The store is identified by its unique store number.")
async def get_count_items_by_store_snowfall(store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT COUNT(DISTINCT item_nbr) FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr INNER JOIN sales_in_weather AS T3 ON T2.store_nbr = T3.store_nbr WHERE T3.store_nbr = ? AND T1.snowfall <> 0 AND T1.snowfall IS NOT NULL", (store_nbr,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get station numbers and item numbers for a specific store
@app.get("/v1/sales_in_weather/station_item_by_store", operation_id="get_station_item_by_store", summary="Retrieves a list of unique station numbers and corresponding item numbers associated with a specific store. The store is identified by its unique store number.")
async def get_station_item_by_store(store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT T1.station_nbr, T2.item_nbr FROM relation AS T1 INNER JOIN sales_in_weather AS T2 ON T1.store_nbr = T2.store_nbr WHERE T1.store_nbr = ? GROUP BY T1.station_nbr, T2.item_nbr", (store_nbr,))
    result = cursor.fetchall()
    if not result:
        return {"stations_items": []}
    return {"stations_items": result}

# Endpoint to get dates for a specific store and codesum pattern
@app.get("/v1/sales_in_weather/dates_by_store_codesum", operation_id="get_dates_by_store_codesum", summary="Get dates for a specific store and codesum pattern")
async def get_dates_by_store_codesum(store_nbr: int = Query(..., description="Store number"), codesum_pattern: str = Query(..., description="Codesum pattern")):
    cursor.execute("SELECT T1.`date` FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T2.store_nbr = ? AND T1.codesum LIKE ?", (store_nbr, codesum_pattern))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": result}

# Endpoint to get sealevel and average speed for specific stores
@app.get("/v1/sales_in_weather/sealevel_avgspeed_by_stores", operation_id="get_sealevel_avgspeed_by_stores", summary="Retrieves the sealevel and average speed data for two specified stores. The data is obtained by joining the weather and relation tables using the station number, and filtering the results based on the provided store numbers.")
async def get_sealevel_avgspeed_by_stores(store_nbr_1: int = Query(..., description="First store number"), store_nbr_2: int = Query(..., description="Second store number")):
    cursor.execute("SELECT T1.sealevel, T1.avgspeed FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T2.store_nbr = ? OR T2.store_nbr = ?", (store_nbr_1, store_nbr_2))
    result = cursor.fetchall()
    if not result:
        return {"sealevel_avgspeed": []}
    return {"sealevel_avgspeed": result}

# Endpoint to get the item number with the highest units sold for a specific store and weather condition
@app.get("/v1/sales_in_weather/top_item_by_store_and_weather", operation_id="get_top_item_by_store_and_weather", summary="Retrieves the item number with the highest sales for a given store and weather condition. The weather condition is specified using a pattern, with the first, second, and third parts of the pattern provided as separate input parameters. The operation returns the item number that has the most units sold under the specified store and weather condition.")
async def get_top_item_by_store_and_weather(store_nbr: int = Query(..., description="Store number"), codesum_like_1: str = Query(..., description="First part of the codesum pattern"), codesum_like_2: str = Query(..., description="Second part of the codesum pattern"), codesum_like_3: str = Query(..., description="Third part of the codesum pattern")):
    cursor.execute("SELECT T2.item_nbr FROM weather AS T1 INNER JOIN sales_in_weather AS T2 ON T1.`date` = T2.`date` INNER JOIN relation AS T3 ON T2.store_nbr = T3.store_nbr AND T1.station_nbr = T3.station_nbr WHERE T2.store_nbr = ? AND T1.codesum LIKE ? OR ? OR ? GROUP BY T2.item_nbr ORDER BY T2.units DESC LIMIT 1", (store_nbr, codesum_like_1, codesum_like_2, codesum_like_3))
    result = cursor.fetchone()
    if not result:
        return {"item_nbr": []}
    return {"item_nbr": result[0]}

# Endpoint to get the temperature range ratio for a specific store
@app.get("/v1/sales_in_weather/temperature_range_ratio", operation_id="get_temperature_range_ratio", summary="Retrieves the ratio of the temperature range to the minimum temperature for a specific store. This operation calculates the ratio by subtracting the minimum temperature from the maximum temperature, then dividing by the minimum temperature. The store is identified by its unique store number.")
async def get_temperature_range_ratio(store_nbr: int = Query(..., description="Store number")):
    cursor.execute("SELECT CAST((MAX(T1.tmax) - MIN(T1.tmin)) AS REAL) / MIN(T1.tmin) FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T2.store_nbr = ?", (store_nbr,))
    result = cursor.fetchone()
    if not result:
        return {"temperature_range_ratio": []}
    return {"temperature_range_ratio": result[0]}

# Endpoint to get the difference in units sold between two stations for a specific year
@app.get("/v1/sales_in_weather/units_difference_by_stations_and_year", operation_id="get_units_difference_by_stations_and_year", summary="Retrieve the difference in total units sold between two specified stations for a given year. The operation compares sales data from two stations and calculates the difference in units sold for the provided year.")
async def get_units_difference_by_stations_and_year(station_nbr_1: int = Query(..., description="First station number"), station_nbr_2: int = Query(..., description="Second station number"), date_pattern: str = Query(..., description="Date pattern in 'YYYY' format")):
    cursor.execute("SELECT SUM(CASE WHEN T1.station_nbr = ? THEN units ELSE 0 END) - SUM(CASE WHEN T1.station_nbr = ? THEN units ELSE 0 END) FROM relation AS T1 INNER JOIN sales_in_weather AS T2 ON T1.store_nbr = T2.store_nbr WHERE T2.`date` LIKE ?", (station_nbr_1, station_nbr_2, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"units_difference": []}
    return {"units_difference": result[0]}

# Endpoint to get the difference in average temperature between two stores for a specific date
@app.get("/v1/sales_in_weather/temperature_difference_by_stores_and_date", operation_id="get_temperature_difference_by_stores_and_date", summary="Retrieve the difference in average daily temperature between two specified stores on a given date. The operation calculates the sum of average temperatures for each store and returns the difference between them.")
async def get_temperature_difference_by_stores_and_date(store_nbr_1: int = Query(..., description="First store number"), store_nbr_2: int = Query(..., description="Second store number"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(CASE WHEN T1.store_nbr = ? THEN T2.tavg ELSE 0 END) - SUM(CASE WHEN T1.store_nbr = ? THEN T2.tavg ELSE 0 END) FROM relation AS T1 INNER JOIN weather AS T2 ON T1.station_nbr = T2.station_nbr WHERE T2.`date` = ?", (store_nbr_1, store_nbr_2, date))
    result = cursor.fetchone()
    if not result:
        return {"temperature_difference": []}
    return {"temperature_difference": result[0]}

# Endpoint to get the total units sold for a specific item and average temperature
@app.get("/v1/sales_in_weather/total_units_by_item_and_temperature", operation_id="get_total_units_by_item_and_temperature", summary="Retrieves the total units sold for a specific item under a given average temperature. This operation combines data from the sales_in_weather and weather tables, using the date as a common link. The store_nbr from sales_in_weather is also used to establish a relationship with the relation table. The item_nbr and average temperature are used to filter the results.")
async def get_total_units_by_item_and_temperature(item_nbr: int = Query(..., description="Item number"), tavg: int = Query(..., description="Average temperature")):
    cursor.execute("SELECT SUM(units) FROM weather AS T1 INNER JOIN sales_in_weather AS T2 ON T1.`date` = T2.`date` INNER JOIN relation AS T3 ON T2.store_nbr = T3.store_nbr WHERE T2.item_nbr = ? AND T1.tavg = ?", (item_nbr, tavg))
    result = cursor.fetchone()
    if not result:
        return {"total_units": []}
    return {"total_units": result[0]}

# Endpoint to get the difference in units sold for a specific item between the earliest and latest sunset times
@app.get("/v1/sales_in_weather/units_difference_by_item_and_sunset", operation_id="get_units_difference_by_item_and_sunset", summary="Get the difference in units sold for a specific item between the earliest and latest sunset times")
async def get_units_difference_by_item_and_sunset(item_nbr: int = Query(..., description="Item number")):
    cursor.execute("( SELECT SUM(T2.units) AS sumunit FROM weather AS T1 INNER JOIN sales_in_weather AS T2 ON T1.`date` = T2.`date` INNER JOIN relation AS T3 ON T2.store_nbr = T3.store_nbr WHERE T2.item_nbr = ? AND sunset IS NOT NULL GROUP BY T1.sunset ORDER BY T1.sunset LIMIT 1 ) - ( SELECT SUM(T2.units) AS sumunit FROM weather AS T1 INNER JOIN sales_in_weather AS T2 ON T1.`date` = T2.`date` INNER JOIN relation AS T3 ON T2.store_nbr = T3.store_nbr WHERE T2.item_nbr = ? AND sunset IS NOT NULL GROUP BY T1.sunset ORDER BY T1.sunset DESC LIMIT 1 )", (item_nbr, item_nbr))
    result = cursor.fetchone()
    if not result:
        return {"units_difference": []}
    return {"units_difference": result[0]}

# Endpoint to get the total units sold for a specific item when the average temperature is below the average for that item
@app.get("/v1/sales_in_weather/total_units_below_avg_temperature", operation_id="get_total_units_below_avg_temperature", summary="Retrieves the total number of units sold for a specific item when the average temperature is below the overall average for that item. The item is identified by its unique number.")
async def get_total_units_below_avg_temperature(item_nbr: int = Query(..., description="Item number")):
    cursor.execute("SELECT SUM(T5.units) FROM weather AS T4 INNER JOIN sales_in_weather AS T5 ON T4.`date` = T5.`date` INNER JOIN relation AS T6 ON T5.store_nbr = T6.store_nbr WHERE T5.item_nbr = ? AND T4.tavg < ( SELECT AVG(T1.tavg) FROM weather AS T1 INNER JOIN sales_in_weather AS T2 ON T1.`date` = T2.`date` INNER JOIN relation AS T3 ON T2.store_nbr = T3.store_nbr WHERE T2.item_nbr = ? )", (item_nbr, item_nbr))
    result = cursor.fetchone()
    if not result:
        return {"total_units": []}
    return {"total_units": result[0]}

# Endpoint to get the difference in average temperature between two stores for a specific month and year
@app.get("/v1/sales_in_weather/avg_temperature_difference_by_stores_and_month", operation_id="get_avg_temperature_difference_by_stores_and_month", summary="Get the difference in average temperature between two stores for a specific month and year")
async def get_avg_temperature_difference_by_stores_and_month(date_pattern_1: str = Query(..., description="Date pattern in 'YYYY-MM' format for the first store"), store_nbr_1: int = Query(..., description="First store number"), date_pattern_2: str = Query(..., description="Date pattern in 'YYYY-MM' format for the second store"), store_nbr_2: int = Query(..., description="Second store number")):
    cursor.execute("( SELECT CAST(SUM(tavg) AS REAL) / COUNT(`date`) FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr AND T1.`date` LIKE ? AND T2.store_nbr = ? ) - ( SELECT CAST(SUM(tavg) AS REAL) / COUNT(`date`) FROM weather AS T1 INNER JOIN relation AS T2 ON T1.station_nbr = T2.station_nbr WHERE T1.`date` LIKE ? AND T2.store_nbr = ? )", (date_pattern_1, store_nbr_1, date_pattern_2, store_nbr_2))
    result = cursor.fetchone()
    if not result:
        return {"avg_temperature_difference": []}
    return {"avg_temperature_difference": result[0]}

api_calls = [
    "/v1/sales_in_weather/units_by_date_store_item?date=2012-01-01&store_nbr=1&item_nbr=9",
    "/v1/sales_in_weather/total_units_by_month_year_store_item?month=01&year=2012&item_nbr=9&store_nbr=1",
    "/v1/sales_in_weather/top_item_by_date_store?date=2012-01-01&store_nbr=1",
    "/v1/sales_in_weather/weather/temperature_range_by_date_station?station_nbr=1&date=2012-01-01",
    "/v1/sales_in_weather/weather/dates_with_departure_greater_than?station_nbr=2&depart=0",
    "/v1/sales_in_weather/weather/compare_avgspeed_by_dates_station?date1=2012-01-01&date2=2012-01-02&station_nbr=1",
    "/v1/sales_in_weather/sum_units_by_store_year_item_depart?depart=0&store_nbr=3&year=2012&item_nbr=5",
    "/v1/sales_in_weather/units_by_store_year_item_max_temp?store_nbr=3&year=2012&item_nbr=5",
    "/v1/sales_in_weather/dewpoint_by_store_year_item_max_units?store_nbr=3&year=2012&item_nbr=5",
    "/v1/sales_in_weather/count_days_units_above_value?units=100&store_nbr=3&year=2012&item_nbr=5&tmax=90",
    "/v1/sales_in_weather/units_by_max_temp_diff?store_nbr=3&item_nbr=5",
    "/v1/sales_in_weather/date_by_max_temp_diff?store_nbr=3&item_nbr=5&units=100",
    "/v1/sales_in_weather/sum_units_by_precip?store_nbr=3&item_nbr=5&preciptotal=0.05",
    "/v1/sales_in_weather/date_by_units_avgspeed?store_nbr=3&item_nbr=5&units=100&avgspeed=10",
    "/v1/sales_in_weather/sum_units_by_date_pattern?store_nbr=3&date_pattern=%252012%25",
    "/v1/sales_in_weather/diff_sum_units_by_stores?item_nbr=16&date_pattern=%252012%25&store_nbr_1=5&store_nbr_2=6",
    "/v1/sales_in_weather/item_nbr_by_max_temp?store_nbr=3&date_pattern=%252012%25",
    "/v1/sales_in_weather/count_distinct_items_zero_units?store_nbr=3&year=2012&units=0",
    "/v1/sales_in_weather/avg_units_per_date?store_nbr=3&item_nbr=5&tmax=90",
    "/v1/sales_in_weather/percentage_units_by_max_temp?item_nbr=5&store_nbr=3&date_pattern=%252012%25",
    "/v1/sales_in_weather/average_temperature?date=2014-10-17&station_nbr=20",
    "/v1/sales_in_weather/resultant_wind_speed?date=2014-01-15&station_nbr=9",
    "/v1/sales_in_weather/most_related_station",
    "/v1/sales_in_weather/related_stations?store_nbr=20",
    "/v1/sales_in_weather/temperature_range?store_nbr=7&date=2014-04-28",
    "/v1/sales_in_weather/stores_highest_departure",
    "/v1/sales_in_weather/dew_point?store_nbr=15&date=2012-02-18",
    "/v1/sales_in_weather/wet_bulb_temperature?store_nbr=14&date=2012-02-15",
    "/v1/sales_in_weather/count_stores_highest_avgspeed",
    "/v1/sales_in_weather/tmax_by_store_date?store_nbr=21&date=2012-11-09",
    "/v1/sales_in_weather/sunrise_by_store_date?date=2014-02-21&store_nbr=30",
    "/v1/sales_in_weather/store_with_highest_snowfall",
    "/v1/sales_in_weather/codesum_by_store_date?date=2013-02-12&store_nbr=2",
    "/v1/sales_in_weather/sealevel_by_store_date?date=2013-02-24&store_nbr=19",
    "/v1/sales_in_weather/preciptotal_by_store_date?date=2012-12-25&store_nbr=2",
    "/v1/sales_in_weather/stnpressure_by_store_date?date=2012-05-15&store_nbr=12",
    "/v1/sales_in_weather/percentage_units_sold_by_store_date?store_nbr=10&date=2014-10-31",
    "/v1/sales_in_weather/percentage_change_tavg_by_store_dates?date1=2012-02-03&date2=2012-02-02&store_nbr=9",
    "/v1/sales_in_weather/sum_store_nbr_by_station?station_nbr=12",
    "/v1/sales_in_weather/count_items_by_store_units_date?store_nbr=2&units=0&date=2012-01-01",
    "/v1/sales_in_weather/weather/date_with_highest_tmax_in_year_range?station_nbr=1&start_year=2012&end_year=2014",
    "/v1/sales_in_weather/relation/station_with_most_stores",
    "/v1/sales_in_weather/weather/station_with_most_negative_departure_days?year=2014&month=03",
    "/v1/sales_in_weather/station_with_highest_units_sold?item_nbr=9",
    "/v1/sales_in_weather/relation/count_stores_with_highest_avgspeed",
    "/v1/sales_in_weather/store_numbers_by_criteria?station_nbr=14&date_pattern=%252014-02%25&item_nbr=44&min_units=300",
    "/v1/sales_in_weather/item_with_highest_units_sold?station_nbr=9&date_pattern=%252013-06%25&codesum=RA",
    "/v1/sales_in_weather/station_with_highest_units_sold_for_item?item_nbr=5",
    "/v1/sales_in_weather/relation/station_with_earliest_sunrise?date_pattern=%252012-02%25",
    "/v1/sales_in_weather/top_store_by_item_station_date?item_nbr=45&station_nbr=17&date_pattern=%252012-10%25",
    "/v1/sales_in_weather/items_by_highest_snowfall_station",
    "/v1/sales_in_weather/top_stations_by_units",
    "/v1/sales_in_weather/store_count_by_highest_heat_station",
    "/v1/sales_in_weather/min_temperature_by_store_date?store_nbr=29&date=2014-02-08",
    "/v1/sales_in_weather/station_count_by_pressure_store_count_date?date=2014-02-18&stnpressure=30&store_count=3",
    "/v1/sales_in_weather/avg_max_temp_by_most_stores_month?month=2012-02",
    "/v1/sales_in_weather/percentage_units_sold_by_store?store_nbr=10&station_nbr=12&item_nbr=5&date_pattern=%252014%25",
    "/v1/sales_in_weather/max_avg_speed",
    "/v1/sales_in_weather/count_dates_by_snowfall?snowfall=5",
    "/v1/sales_in_weather/count_days_before_sunrise?sunrise_time=05:00:00",
    "/v1/sales_in_weather/min_dewpoint",
    "/v1/sales_in_weather/temperature_by_station_date?station_nbr=1&date=2012-01-15",
    "/v1/sales_in_weather/count_stations_by_month_item?month=2014-01&item_nbr=5",
    "/v1/sales_in_weather/total_units_by_store_item_snowfall?store_nbr=7&item_nbr=7&snowfall=5",
    "/v1/sales_in_weather/count_items_by_store_snowfall?store_nbr=9",
    "/v1/sales_in_weather/station_item_by_store?store_nbr=17",
    "/v1/sales_in_weather/dates_by_store_codesum?store_nbr=35&codesum_pattern=%HZ%",
    "/v1/sales_in_weather/sealevel_avgspeed_by_stores?store_nbr_1=3&store_nbr_2=4",
    "/v1/sales_in_weather/top_item_by_store_and_weather?store_nbr=1&codesum_like_1=%25&codesum_like_2=RA&codesum_like_3=%25",
    "/v1/sales_in_weather/temperature_range_ratio?store_nbr=11",
    "/v1/sales_in_weather/units_difference_by_stations_and_year?station_nbr_1=1&station_nbr_2=2&date_pattern=%252012%25",
    "/v1/sales_in_weather/temperature_difference_by_stores_and_date?store_nbr_1=18&store_nbr_2=19&date=2012-09-16",
    "/v1/sales_in_weather/total_units_by_item_and_temperature?item_nbr=1&tavg=83",
    "/v1/sales_in_weather/units_difference_by_item_and_sunset?item_nbr=5",
    "/v1/sales_in_weather/total_units_below_avg_temperature?item_nbr=10",
    "/v1/sales_in_weather/avg_temperature_difference_by_stores_and_month?date_pattern_1=%252012-05%25&store_nbr_1=6&date_pattern_2=%252012-05%25&store_nbr_2=7"
]
