from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/airline/airline.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of flights on a specific date
@app.get("/v1/airline/flight_count_by_date", operation_id="get_flight_count_by_date", summary="Retrieves the total number of flights scheduled for a specific date. The date must be provided in the 'YYYY/MM/DD' format.")
async def get_flight_count_by_date(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT COUNT(*) FROM Airlines WHERE FL_DATE = ?", (fl_date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of flights on a specific date from a specific origin
@app.get("/v1/airline/flight_count_by_date_and_origin", operation_id="get_flight_count_by_date_and_origin", summary="Retrieves the total number of flights departing from a specific airport on a given date. The date should be provided in 'YYYY/MM/DD' format, and the origin airport is identified by its unique code.")
async def get_flight_count_by_date_and_origin(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), origin: str = Query(..., description="Origin airport code")):
    cursor.execute("SELECT COUNT(*) FROM Airlines WHERE FL_DATE = ? AND ORIGIN = ?", (fl_date, origin))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the destinations of cancelled flights on a specific date
@app.get("/v1/airline/cancelled_flights_destinations_by_date", operation_id="get_cancelled_flights_destinations_by_date", summary="Retrieve a list of destinations for flights that were cancelled on a specific date. The operation filters flights based on the provided date and cancellation status, and groups the results by destination.")
async def get_cancelled_flights_destinations_by_date(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), cancelled: int = Query(..., description="Cancellation status (1 for cancelled)")):
    cursor.execute("SELECT DEST FROM Airlines WHERE FL_DATE = ? AND CANCELLED = ? GROUP BY DEST", (fl_date, cancelled))
    result = cursor.fetchall()
    if not result:
        return {"destinations": []}
    return {"destinations": [row[0] for row in result]}

# Endpoint to get the flight dates with a specific cancellation code
@app.get("/v1/airline/flight_dates_by_cancellation_code", operation_id="get_flight_dates_by_cancellation_code", summary="Retrieves a list of unique flight dates associated with a specific cancellation code. The cancellation code is used to filter the results.")
async def get_flight_dates_by_cancellation_code(cancellation_code: str = Query(..., description="Cancellation code")):
    cursor.execute("SELECT FL_DATE FROM Airlines WHERE CANCELLATION_CODE = ? GROUP BY FL_DATE", (cancellation_code,))
    result = cursor.fetchall()
    if not result:
        return {"flight_dates": []}
    return {"flight_dates": [row[0] for row in result]}

# Endpoint to get the descriptions of airports with delayed departures on a specific date
@app.get("/v1/airline/airport_descriptions_with_delayed_departures", operation_id="get_airport_descriptions_with_delayed_departures", summary="Retrieve the descriptions of airports that have experienced delayed departures on a specified date. The operation filters airports based on a given flight date and a minimum departure delay threshold. The results are grouped by airport descriptions.")
async def get_airport_descriptions_with_delayed_departures(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), dep_delay: int = Query(..., description="Departure delay in minutes")):
    cursor.execute("SELECT T1.Description FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T2.FL_DATE = ? AND T2.DEP_DELAY > ? GROUP BY T1.Description", (fl_date, dep_delay))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the count of flights from a specific airport on a specific date
@app.get("/v1/airline/flight_count_from_airport_by_date", operation_id="get_flight_count_from_airport_by_date", summary="Retrieves the total number of flights departing from a specified airport on a given date. The input parameters include the flight date in 'YYYY/MM/DD' format and the description of the airport. The operation returns a count of flights that match the provided criteria.")
async def get_flight_count_from_airport_by_date(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), description: str = Query(..., description="Airport description")):
    cursor.execute("SELECT COUNT(T1.Code) FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T2.FL_DATE = ? AND T1.Description = ?", (fl_date, description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the description of the airport with the highest departure delay on a specific date
@app.get("/v1/airline/airport_with_highest_departure_delay", operation_id="get_airport_with_highest_departure_delay", summary="Retrieves the description of the airport with the highest average departure delay for flights on a specific date. The input parameter 'fl_date' in 'YYYY/MM/DD' format is used to filter the results.")
async def get_airport_with_highest_departure_delay(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT T1.Description FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.DEST WHERE T2.FL_DATE = ? ORDER BY T2.DEP_DELAY DESC LIMIT 1", (fl_date,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the count of flights with arrival delay less than a specific value from a specific airport
@app.get("/v1/airline/flight_count_with_arrival_delay_less_than", operation_id="get_flight_count_with_arrival_delay_less_than", summary="Retrieves the total number of flights from a specific airport that have an arrival delay less than the provided value. The airport is identified by its description, and the arrival delay is specified in minutes.")
async def get_flight_count_with_arrival_delay_less_than(arr_delay: int = Query(..., description="Arrival delay in minutes"), description: str = Query(..., description="Airport description")):
    cursor.execute("SELECT SUM(CASE WHEN T2.ARR_DELAY < ? THEN 1 ELSE 0 END) AS count FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T1.Description = ?", (arr_delay, description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the earliest departure time from a specific airport on a specific date
@app.get("/v1/airline/earliest_departure_time_from_airport", operation_id="get_earliest_departure_time_from_airport", summary="Retrieves the earliest departure time from a specified airport on a given date. The operation filters flights based on the provided date and airport description, then returns the earliest departure time from the filtered results. The date should be provided in 'YYYY/MM/DD' format.")
async def get_earliest_departure_time_from_airport(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), description: str = Query(..., description="Airport description")):
    cursor.execute("SELECT T2.DEP_TIME FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T2.FL_DATE = ? AND T1.Description = ? AND T2.DEP_TIME IS NOT NULL ORDER BY T2.DEP_TIME ASC LIMIT 1", (fl_date, description))
    result = cursor.fetchone()
    if not result:
        return {"departure_time": []}
    return {"departure_time": result[0]}

# Endpoint to get the count of flights from a specific air carrier on a specific date
@app.get("/v1/airline/flight_count_by_air_carrier_and_date", operation_id="get_flight_count_by_air_carrier_and_date", summary="Retrieve the total number of flights operated by a specific air carrier on a given date. The operation requires the flight date and the description of the air carrier as input parameters. The result is a single integer value representing the total count of flights.")
async def get_flight_count_by_air_carrier_and_date(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), description: str = Query(..., description="Air carrier description")):
    cursor.execute("SELECT COUNT(*) FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN INNER JOIN `Air Carriers` AS T3 ON T2.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T2.FL_DATE = ? AND T3.Description = ?", (fl_date, description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get flight numbers for a specific airline, airport, and date
@app.get("/v1/airline/flight_numbers_by_airline_airport_date", operation_id="get_flight_numbers", summary="Retrieves the flight numbers for a specific airline and date at the New York, NY: John F. Kennedy International airport. The operation requires the description of the airline, the flight date in 'YYYY/MM/DD' format, and the description of the airport, which is predefined as 'New York, NY: John F. Kennedy International'.")
async def get_flight_numbers(airline_description: str = Query(..., description="Description of the airline"), airport_description: str = Query(..., description="Description of the airport"), flight_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT T2.OP_CARRIER_FL_NUM FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN INNER JOIN `Air Carriers` AS T3 ON T2.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T3.Description = ? AND T1.Description = ? AND T2.FL_DATE = ?", (airline_description, airport_description, flight_date))
    result = cursor.fetchall()
    if not result:
        return {"flight_numbers": []}
    return {"flight_numbers": [row[0] for row in result]}

# Endpoint to get the count of flights with actual elapsed time less than scheduled elapsed time for a specific airline and date
@app.get("/v1/airline/count_flights_actual_elapsed_time_less_than_scheduled", operation_id="get_count_flights_actual_elapsed_time_less_than_scheduled", summary="Retrieve the total number of flights for a specific airline and date where the actual elapsed time was less than the scheduled elapsed time. The input parameters include the flight date in 'YYYY/MM/DD' format and the description of the airline.")
async def get_count_flights_actual_elapsed_time_less_than_scheduled(flight_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT SUM(CASE WHEN T2.ACTUAL_ELAPSED_TIME < CRS_ELAPSED_TIME THEN 1 ELSE 0 END) AS count FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN INNER JOIN `Air Carriers` AS T3 ON T2.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T2.FL_DATE = ? AND T3.Description = ?", (flight_date, airline_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latest flight number for a specific airline
@app.get("/v1/airline/latest_flight_number_by_airline", operation_id="get_latest_flight_number", summary="Retrieves the most recent flight number for a specific airline, based on departure time. The airline is identified by its description.")
async def get_latest_flight_number(airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT T1.OP_CARRIER_FL_NUM FROM Airlines AS T1 INNER JOIN Airports AS T2 ON T2.Code = T1.ORIGIN INNER JOIN `Air Carriers` AS T3 ON T1.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T3.Description = ? ORDER BY T1.DEP_TIME DESC LIMIT 1", (airline_description,))
    result = cursor.fetchone()
    if not result:
        return {"flight_number": []}
    return {"flight_number": result[0]}

# Endpoint to get the count of flights to a specific destination for a specific airline
@app.get("/v1/airline/count_flights_to_destination_by_airline", operation_id="get_count_flights_to_destination", summary="Retrieves the total number of flights operated by a specific airline to a given destination. The operation requires the destination airport code and the description of the airline as input parameters.")
async def get_count_flights_to_destination(destination: str = Query(..., description="Destination airport code"), airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT SUM(CASE WHEN T2.DEST = ? THEN 1 ELSE 0 END) AS count FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.DEST INNER JOIN `Air Carriers` AS T3 ON T2.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T3.Description = ?", (destination, airline_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cancelled flights for a specific airline and date
@app.get("/v1/airline/count_cancelled_flights_by_airline_date", operation_id="get_count_cancelled_flights", summary="Retrieve the total number of cancelled flights for a specific airline on a given date. The operation calculates the sum of cancelled flights based on the provided flight date and airline description.")
async def get_count_cancelled_flights(flight_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT SUM(CASE WHEN T2.CANCELLED = 1 THEN 1 ELSE 0 END) AS count FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN INNER JOIN `Air Carriers` AS T3 ON T2.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T2.FL_DATE = ? AND T3.Description = ?", (flight_date, airline_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to compare the number of flights between two airlines on a specific date
@app.get("/v1/airline/compare_flights_between_airlines", operation_id="compare_flights_between_airlines", summary="This operation compares the number of flights between two airlines on a specific date. It returns the airline with the higher number of flights on the given date. The input parameters include the descriptions of the two airlines and the flight date in 'YYYY/MM/DD' format.")
async def compare_flights_between_airlines(airline_description_1: str = Query(..., description="Description of the first airline"), airline_description_2: str = Query(..., description="Description of the second airline"), flight_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT CASE WHEN COUNT(CASE WHEN T3.Description = ? THEN 1 ELSE NULL END) > COUNT(CASE WHEN T3.Description = ? THEN 1 ELSE NULL END) THEN ? ELSE ? END AS RESULT FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN INNER JOIN `Air Carriers` AS T3 ON T2.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T2.FL_DATE = ?", (airline_description_1, airline_description_2, airline_description_1, airline_description_2, flight_date))
    result = cursor.fetchone()
    if not result:
        return {"result": []}
    return {"result": result[0]}

# Endpoint to get the average departure delay for a specific airline
@app.get("/v1/airline/average_departure_delay_by_airline", operation_id="get_average_departure_delay", summary="Retrieves the average departure delay for a specific airline, calculated based on the airline's origin airport and its operational carrier. The input parameter specifies the airline by its description.")
async def get_average_departure_delay(airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT AVG(T1.DEP_DELAY) FROM Airlines AS T1 INNER JOIN Airports AS T2 ON T2.Code = T1.ORIGIN INNER JOIN `Air Carriers` AS T3 ON T1.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T3.Description = ?", (airline_description,))
    result = cursor.fetchone()
    if not result:
        return {"average_delay": []}
    return {"average_delay": result[0]}

# Endpoint to get the average number of flights per day for a specific airline in a given month
@app.get("/v1/airline/average_flights_per_day_by_airline_month", operation_id="get_average_flights_per_day", summary="Retrieves the average daily flight count for a specific airline during a given month. The calculation is based on the total number of flights operated by the airline in the specified month, divided by the number of days in that month. The input parameters include the flight date pattern in 'YYYY/MM%' format and the description of the airline.")
async def get_average_flights_per_day(flight_date_pattern: str = Query(..., description="Flight date pattern in 'YYYY/MM%' format"), airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT CAST( SUM(CASE WHEN T2.FL_DATE LIKE ? THEN 1 ELSE 0 END) AS REAL) / 31 FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN INNER JOIN `Air Carriers` AS T3 ON T2.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T3.Description = ?", (flight_date_pattern, airline_description))
    result = cursor.fetchone()
    if not result:
        return {"average_flights_per_day": []}
    return {"average_flights_per_day": result[0]}

# Endpoint to get the count of air carriers
@app.get("/v1/airline/count_air_carriers", operation_id="get_count_air_carriers", summary="Retrieves the total number of air carriers currently in the system.")
async def get_count_air_carriers():
    cursor.execute("SELECT COUNT(Code) FROM `Air Carriers`")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of flights from a specific airport on a specific date
@app.get("/v1/airline/count_flights_from_airport_date", operation_id="get_count_flights_from_airport", summary="Retrieves the total number of flights departing from a specified airport on a given date. The input parameters include the date of the flight and the description of the airport. The operation returns a count of flights that match the provided criteria.")
async def get_count_flights_from_airport(flight_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), airport_description: str = Query(..., description="Description of the airport")):
    cursor.execute("SELECT SUM(CASE WHEN T2.FL_DATE = ? THEN 1 ELSE 0 END) AS count FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T1.Description = ?", (flight_date, airport_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of airports with a description containing a specific substring on a given date
@app.get("/v1/airline/count_airports_description_substring", operation_id="get_count_airports_description_substring", summary="Retrieves the total number of airports whose descriptions include a specified substring and are destinations for flights on a given date.")
async def get_count_airports_description_substring(description_substring: str = Query(..., description="Substring to search in the airport description"), fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Description LIKE ? THEN 1 ELSE 0 END) AS count FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.DEST WHERE T2.FL_DATE = ?", ('%' + description_substring + '%', fl_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of flights with a departure delay for a specific airline on a given date
@app.get("/v1/airline/count_flights_departure_delay", operation_id="get_count_flights_departure_delay", summary="Retrieves the total number of flights that experienced a departure delay for a particular airline on a specified date. The operation requires the flight date in 'YYYY/MM/DD' format and the description of the airline as input parameters.")
async def get_count_flights_departure_delay(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT COUNT(*) FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T1.OP_CARRIER_AIRLINE_ID = T2.Code WHERE T1.FL_DATE = ? AND T2.Description = ? AND T1.DEP_DELAY > 0", (fl_date, airline_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of flights with an arrival delay less than zero for a specific destination on a given date
@app.get("/v1/airline/count_flights_arrival_delay_less_than_zero", operation_id="get_count_flights_arrival_delay_less_than_zero", summary="Retrieve the number of flights that arrived ahead of schedule for a specific destination on a given date. The operation requires the flight date and the destination airport code as input parameters.")
async def get_count_flights_arrival_delay_less_than_zero(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), destination: str = Query(..., description="Destination airport code")):
    cursor.execute("SELECT COUNT(*) FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.DEST WHERE T2.FL_DATE = ? AND T2.DEST = ? AND T2.ARR_DELAY < 0", (fl_date, destination))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of flights with actual elapsed time less than scheduled elapsed time for a specific airline on a given date
@app.get("/v1/airline/count_flights_elapsed_time_less_than_scheduled", operation_id="get_count_flights_elapsed_time_less_than_scheduled", summary="Retrieve the total number of flights for a specific airline on a given date where the actual elapsed time was less than the scheduled elapsed time. The input parameters include the flight date and the description of the airline.")
async def get_count_flights_elapsed_time_less_than_scheduled(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT SUM(CASE WHEN T1.ACTUAL_ELAPSED_TIME < CRS_ELAPSED_TIME THEN 1 ELSE 0 END) AS count FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T1.OP_CARRIER_AIRLINE_ID = T2.Code WHERE T1.FL_DATE = ? AND T2.Description = ?", (fl_date, airline_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the descriptions of air carriers for a specific tail number
@app.get("/v1/airline/air_carrier_descriptions_by_tail_num", operation_id="get_air_carrier_descriptions_by_tail_num", summary="Retrieves the descriptions of air carriers associated with a specific aircraft tail number. The descriptions provide detailed information about the air carriers operating the aircraft. The input parameter is the tail number of the aircraft.")
async def get_air_carrier_descriptions_by_tail_num(tail_num: str = Query(..., description="Tail number of the aircraft")):
    cursor.execute("SELECT T2.Description FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T1.OP_CARRIER_AIRLINE_ID = T2.Code WHERE T1.TAIL_NUM = ? GROUP BY T2.Description", (tail_num,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the description of the airport with the latest destination on a given date
@app.get("/v1/airline/latest_destination_airport_description", operation_id="get_latest_destination_airport_description", summary="Retrieves the description of the airport that was the most recent destination on a specified date. The date must be provided in 'YYYY/MM/DD' format.")
async def get_latest_destination_airport_description(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT T1.Description FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.DEST WHERE T2.FL_DATE = ? ORDER BY T2.DEST DESC LIMIT 1", (fl_date,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the descriptions of air carriers for a specific flight on a given date, origin, destination, and departure time
@app.get("/v1/airline/air_carrier_descriptions_by_flight", operation_id="get_air_carrier_descriptions_by_flight", summary="Retrieve the descriptions of air carriers associated with a specific flight, based on the provided flight date, origin and destination airports, and scheduled departure time. The operation returns a list of unique air carrier descriptions that match the given criteria.")
async def get_air_carrier_descriptions_by_flight(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), origin: str = Query(..., description="Origin airport code"), destination: str = Query(..., description="Destination airport code"), crs_dep_time: str = Query(..., description="Scheduled departure time in 'HHMM' format")):
    cursor.execute("SELECT T2.Description FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T1.OP_CARRIER_AIRLINE_ID = T2.Code WHERE T1.FL_DATE = ? AND T1.ORIGIN = ? AND T1.DEST = ? AND T1.CRS_DEP_TIME = ? GROUP BY T2.Description", (fl_date, origin, destination, crs_dep_time))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the count of airports with a specific description on a given date
@app.get("/v1/airline/count_airports_by_description", operation_id="get_count_airports_by_description", summary="Retrieves the number of airports matching a specified description on a given flight date. The count is determined by cross-referencing the flight schedule for the provided date with the airport database.")
async def get_count_airports_by_description(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format"), airport_description: str = Query(..., description="Description of the airport")):
    cursor.execute("SELECT COUNT(T1.Code) FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.DEST WHERE T2.FL_DATE = ? AND T1.Description = ?", (fl_date, airport_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of flights between two airports in a specific month
@app.get("/v1/airline/flight_count_between_airports", operation_id="get_flight_count_between_airports", summary="Retrieve the total number of flights between two specified airports during a given month. The origin and destination airports are identified by their descriptions, and the month is determined by a date pattern. The result provides a quantitative measure of air traffic between the two airports in the selected month.")
async def get_flight_count_between_airports(fl_date_pattern: str = Query(..., description="Flight date pattern in 'YYYY/M%' format"), origin_description: str = Query(..., description="Description of the origin airport"), dest_description: str = Query(..., description="Description of the destination airport")):
    cursor.execute("SELECT COUNT(FL_DATE) FROM Airlines WHERE FL_DATE LIKE ? AND ORIGIN = ( SELECT T2.ORIGIN FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T1.Description = ? ) AND DEST = ( SELECT T4.DEST FROM Airports AS T3 INNER JOIN Airlines AS T4 ON T3.Code = T4.DEST WHERE T3.Description = ? )", (fl_date_pattern, origin_description, dest_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of flights cancelled due to carrier reasons on a specific date from a specific airport
@app.get("/v1/airline/cancellation_percentage_carrier", operation_id="get_cancellation_percentage_carrier", summary="Retrieve the percentage of flights cancelled due to carrier reasons on a specific date from a particular airport. The operation calculates this percentage by considering all flights from the specified airport on the given date that have a cancellation code. The input parameters include the flight date and the description of the airport.")
async def get_cancellation_percentage_carrier(fl_date: str = Query(..., description="Flight date in 'YYYY/M/D' format"), airport_description: str = Query(..., description="Description of the airport")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.CANCELLATION_CODE = 'C' THEN 1.0 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T2.FL_DATE = ? AND T2.CANCELLATION_CODE IS NOT NULL AND T1.Description = ?", (fl_date, airport_description))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of flights that arrived early to a specific destination
@app.get("/v1/airline/early_arrival_percentage", operation_id="get_early_arrival_percentage", summary="Retrieves the percentage of flights that arrived earlier than their scheduled time at a specified destination. The destination is identified by its description. The calculation considers only flights with both a scheduled and an actual elapsed time.")
async def get_early_arrival_percentage(destination_description: str = Query(..., description="Description of the destination airport")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.ACTUAL_ELAPSED_TIME < T1.CRS_ELAPSED_TIME THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Airlines AS T1 INNER JOIN Airports AS T2 ON T2.Code = T1.DEST WHERE T2.Description LIKE ? AND T1.CRS_ELAPSED_TIME IS NOT NULL AND T1.ACTUAL_ELAPSED_TIME IS NOT NULL", (destination_description,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the description of an air carrier by code
@app.get("/v1/airline/air_carrier_description", operation_id="get_air_carrier_description", summary="Retrieves the description of a specific air carrier based on its unique code. The code is used to identify the air carrier and fetch its corresponding description from the database.")
async def get_air_carrier_description(code: int = Query(..., description="Code of the air carrier")):
    cursor.execute("SELECT Description FROM `Air Carriers` WHERE Code = ?", (code,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the count of flights with a departure delay less than or equal to a specified value on a specific date
@app.get("/v1/airline/flight_count_dep_delay", operation_id="get_flight_count_dep_delay", summary="Retrieves the total number of flights that departed on a given date and had a departure delay of up to a specified number of minutes.")
async def get_flight_count_dep_delay(fl_date: str = Query(..., description="Flight date in 'YYYY/M/D' format"), dep_delay: int = Query(..., description="Departure delay in minutes")):
    cursor.execute("SELECT COUNT(*) FROM Airlines WHERE FL_DATE = ? AND DEP_DELAY <= ?", (fl_date, dep_delay))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the code of airports with a description matching a specific pattern
@app.get("/v1/airline/airport_code_by_description", operation_id="get_airport_code_by_description", summary="Retrieves the unique code of airports whose descriptions contain a specified pattern. The input pattern is used to filter the airports and return only those that match the given description pattern.")
async def get_airport_code_by_description(description_pattern: str = Query(..., description="Description pattern of the airport")):
    cursor.execute("SELECT Code FROM Airports WHERE Description LIKE ?", (description_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"codes": []}
    return {"codes": [row[0] for row in result]}

# Endpoint to get the maximum weather delay for flights from a specific airport
@app.get("/v1/airline/max_weather_delay", operation_id="get_max_weather_delay", summary="Retrieves the maximum weather delay experienced by flights departing from a specified airport. The origin airport is identified by its unique ID.")
async def get_max_weather_delay(origin_airport_id: int = Query(..., description="ID of the origin airport")):
    cursor.execute("SELECT WEATHER_DELAY FROM Airlines WHERE ORIGIN_AIRPORT_ID = ? ORDER BY WEATHER_DELAY DESC LIMIT 1", (origin_airport_id,))
    result = cursor.fetchone()
    if not result:
        return {"weather_delay": []}
    return {"weather_delay": result[0]}

# Endpoint to get the code of an airport by its description
@app.get("/v1/airline/airport_code_by_exact_description", operation_id="get_airport_code_by_exact_description", summary="Retrieves the unique code of an airport based on its exact description. The input description must match the airport's description in the database to return the corresponding code.")
async def get_airport_code_by_exact_description(description: str = Query(..., description="Exact description of the airport")):
    cursor.execute("SELECT Code FROM Airports WHERE Description = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"code": []}
    return {"code": result[0]}

# Endpoint to get the origin airport ID with the maximum late aircraft delay
@app.get("/v1/airline/max_late_aircraft_delay_origin", operation_id="get_max_late_aircraft_delay_origin", summary="Retrieves the ID of the origin airport that has the longest late aircraft delay. The operation returns the airport ID with the highest late aircraft delay value, providing insight into the airport with the most significant delays.")
async def get_max_late_aircraft_delay_origin():
    cursor.execute("SELECT ORIGIN_AIRPORT_ID FROM Airlines ORDER BY LATE_AIRCRAFT_DELAY DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"origin_airport_id": []}
    return {"origin_airport_id": result[0]}

# Endpoint to get the count of flights between two airports
@app.get("/v1/airline/flight_count_between_airports_general", operation_id="get_flight_count_between_airports_general", summary="Retrieves the total number of flights between the specified origin and destination airports. The origin and destination airports are identified by their descriptions, which should correspond to the exact descriptions in the database. The count includes all flights that have occurred between these two airports.")
async def get_flight_count_between_airports_general(origin_description: str = Query(..., description="Description of the origin airport"), dest_description: str = Query(..., description="Description of the destination airport")):
    cursor.execute("SELECT COUNT(FL_DATE) FROM Airlines WHERE ORIGIN = ( SELECT T2.ORIGIN FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T1.Description = ? ) AND DEST = ( SELECT T4.DEST FROM Airports AS T3 INNER JOIN Airlines AS T4 ON T3.Code = T4.DEST WHERE T3.Description = ? )", (origin_description, dest_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of tail numbers for a specific airline description
@app.get("/v1/airline/count_tail_numbers_by_airline", operation_id="get_count_tail_numbers", summary="Retrieves the total count of unique tail numbers associated with a specific airline. The airline is identified by its description. This operation does not return individual tail numbers, but rather the total number of distinct tail numbers linked to the specified airline.")
async def get_count_tail_numbers(airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT COUNT(T3.TAIL_NUM) FROM ( SELECT T1.TAIL_NUM FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T1.OP_CARRIER_AIRLINE_ID = T2.Code WHERE T2.Description = ? GROUP BY T1.TAIL_NUM ) T3", (airline_description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most frequent flight date for a specific airport and cancellation criteria
@app.get("/v1/airline/most_frequent_flight_date", operation_id="get_most_frequent_flight_date", summary="Retrieves the date with the highest number of flights from a specific airport, considering a given flight date pattern, cancellation status, and cancellation code. The input parameters allow filtering by a partial flight date, airport description, origin code, cancellation status, and cancellation code. The result is the date with the most flights that match the provided criteria.")
async def get_most_frequent_flight_date(fl_date: str = Query(..., description="Flight date in 'YYYY/M%' format"), airport_description: str = Query(..., description="Description of the airport"), origin: str = Query(..., description="Origin code of the airport"), cancelled: int = Query(..., description="Cancellation status (1 for cancelled)"), cancellation_code: str = Query(..., description="Cancellation code")):
    cursor.execute("SELECT T2.FL_DATE FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T2.FL_DATE LIKE ? AND T1.Description = ? AND T2.ORIGIN = ? AND T2.CANCELLED = ? AND T2.CANCELLATION_CODE = ? GROUP BY T2.FL_DATE ORDER BY COUNT(T2.FL_DATE) DESC LIMIT 1", (fl_date, airport_description, origin, cancelled, cancellation_code))
    result = cursor.fetchone()
    if not result:
        return {"fl_date": []}
    return {"fl_date": result[0]}

# Endpoint to get tail numbers for flights to a specific destination with arrival delay criteria
@app.get("/v1/airline/tail_numbers_by_destination_arrival_delay", operation_id="get_tail_numbers_by_destination_arrival_delay", summary="Retrieve the tail numbers of flights that meet specific arrival delay criteria for a given destination. The operation filters flights based on the provided flight date, destination airport, and maximum arrival delay. The result is a list of unique tail numbers that match the specified conditions.")
async def get_tail_numbers_by_destination_arrival_delay(fl_date: str = Query(..., description="Flight date in 'YYYY/M%' format"), airport_description: str = Query(..., description="Description of the airport"), dest: str = Query(..., description="Destination code of the airport"), arr_delay: int = Query(..., description="Maximum arrival delay")):
    cursor.execute("SELECT T2.TAIL_NUM FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.DEST WHERE T2.FL_DATE LIKE ? AND T1.Description = ? AND T2.DEST = ? AND T2.ARR_DELAY <= ? GROUP BY T2.TAIL_NUM", (fl_date, airport_description, dest, arr_delay))
    result = cursor.fetchall()
    if not result:
        return {"tail_numbers": []}
    return {"tail_numbers": [row[0] for row in result]}

# Endpoint to get the airline with the highest security delay for a specific destination
@app.get("/v1/airline/highest_security_delay_airline", operation_id="get_highest_security_delay_airline", summary="Retrieves the airline with the longest security delay for a specific destination. The destination is identified by its description and destination code. The operation returns the unique identifier of the airline with the highest security delay.")
async def get_highest_security_delay_airline(airport_description: str = Query(..., description="Description of the airport"), dest: str = Query(..., description="Destination code of the airport")):
    cursor.execute("SELECT T2.OP_CARRIER_AIRLINE_ID FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.DEST WHERE T1.Description = ? AND T2.DEST = ? ORDER BY T2.SECURITY_DELAY DESC LIMIT 1", (airport_description, dest))
    result = cursor.fetchone()
    if not result:
        return {"airline_id": []}
    return {"airline_id": result[0]}

# Endpoint to get the top 5 airline descriptions by tail number
@app.get("/v1/airline/top_airline_descriptions", operation_id="get_top_airline_descriptions", summary="Retrieves the top five airline descriptions based on the highest tail number count. The descriptions are obtained by joining the Airlines and Air Carriers tables using the OP_CARRIER_AIRLINE_ID and Code fields, respectively. The results are grouped by description and ordered in descending order based on the tail number count.")
async def get_top_airline_descriptions():
    cursor.execute("SELECT T2.Description FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T1.OP_CARRIER_AIRLINE_ID = T2.Code GROUP BY T2.Description ORDER BY T1.TAIL_NUM DESC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the airline with the most cancellations
@app.get("/v1/airline/most_cancellations", operation_id="get_most_cancellations", summary="Retrieves the airline with the highest number of cancellations. The operation filters airlines based on the provided cancellation status and returns the airline with the most occurrences of that status. The cancellation status is a binary value, with 0 indicating not cancelled.")
async def get_most_cancellations(cancelled: int = Query(..., description="Cancellation status (0 for not cancelled)")):
    cursor.execute("SELECT T2.Description FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T1.OP_CARRIER_AIRLINE_ID = T2.Code WHERE T1.CANCELLED = ? GROUP BY T2.Description ORDER BY COUNT(T1.CANCELLED) DESC LIMIT 1", (cancelled,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the most frequent airline description for a specific destination
@app.get("/v1/airline/most_frequent_airline_description", operation_id="get_most_frequent_airline_description", summary="Retrieves the most frequently occurring airline description for a specific destination. The operation requires the description of the airport and the destination code as input parameters. The result is determined by grouping and ordering the airline descriptions based on their frequency, with the most frequent one being returned.")
async def get_most_frequent_airline_description(airport_description: str = Query(..., description="Description of the airport"), dest: str = Query(..., description="Destination code of the airport")):
    cursor.execute("SELECT T3.Description FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.DEST INNER JOIN `Air Carriers` AS T3 ON T2.OP_CARRIER_AIRLINE_ID = T3.Code WHERE T1.Description = ? AND T2.DEST = ? GROUP BY T3.Description ORDER BY COUNT(T3.Description) DESC LIMIT 1", (airport_description, dest))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the most frequent airline ID for a specific route
@app.get("/v1/airline/most_frequent_airline_id", operation_id="get_most_frequent_airline_id", summary="Retrieve the airline ID that operates the most flights between the specified origin and destination airports, based on the provided airline description.")
async def get_most_frequent_airline_id(airline_description: str = Query(..., description="Description of the airline"), origin: str = Query(..., description="Origin code of the airport"), dest: str = Query(..., description="Destination code of the airport")):
    cursor.execute("SELECT T2.OP_CARRIER_AIRLINE_ID FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description = ? AND T2.ORIGIN = ? AND T2.DEST = ? GROUP BY T2.OP_CARRIER_AIRLINE_ID ORDER BY COUNT(T2.OP_CARRIER_AIRLINE_ID) DESC LIMIT 1", (airline_description, origin, dest))
    result = cursor.fetchone()
    if not result:
        return {"airline_id": []}
    return {"airline_id": result[0]}

# Endpoint to get the most frequent destination for a specific airline
@app.get("/v1/airline/most_frequent_destination", operation_id="get_most_frequent_destination", summary="Retrieves the most frequently visited destination for a specific airline. The operation requires the description of the airline as input and returns the destination with the highest frequency of flights operated by the specified airline.")
async def get_most_frequent_destination(airline_description: str = Query(..., description="Description of the airline")):
    cursor.execute("SELECT T2.DEST FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description = ? GROUP BY T2.DEST ORDER BY COUNT(T2.DEST) DESC LIMIT 1", (airline_description,))
    result = cursor.fetchone()
    if not result:
        return {"destination": []}
    return {"destination": result[0]}

# Endpoint to get the airline ID with the smallest difference between actual and scheduled elapsed time
@app.get("/v1/airline/min_elapsed_time_difference", operation_id="get_min_elapsed_time_difference", summary="Retrieve the airline ID with the smallest difference between actual and scheduled elapsed time. The operation returns the airline ID that has the least discrepancy between the actual and scheduled elapsed time for flights. The input parameter 'limit' can be used to restrict the number of results returned.")
async def get_min_elapsed_time_difference(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.OP_CARRIER_AIRLINE_ID FROM Airlines AS T1 INNER JOIN Airports AS T2 ON T1.ORIGIN = T2.Code WHERE T1.ACTUAL_ELAPSED_TIME IS NOT NULL AND T1.CRS_ELAPSED_TIME IS NOT NULL ORDER BY T1.ACTUAL_ELAPSED_TIME - T1.CRS_ELAPSED_TIME ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"airline_id": []}
    return {"airline_id": result[0]}

# Endpoint to get the tail number with the highest delay for a specific airline and date
@app.get("/v1/airline/max_delay_tail_number", operation_id="get_max_delay_tail_number", summary="Get the tail number with the highest delay for a specific airline and date")
async def get_max_delay_tail_number(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/%' format"), description: str = Query(..., description="Airline description"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.TAIL_NUM, SUM(CAST(T1.LATE_AIRCRAFT_DELAY AS REAL) / 60) AS delay FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T2.Code = T1.OP_CARRIER_AIRLINE_ID WHERE T1.FL_DATE LIKE ? AND T2.Description = ? ORDER BY delay DESC LIMIT ?", (fl_date, description, limit))
    result = cursor.fetchone()
    if not result:
        return {"tail_num": [], "delay": []}
    return {"tail_num": result[0], "delay": result[1]}

# Endpoint to get a list of airports with their codes and descriptions
@app.get("/v1/airline/airports_list", operation_id="get_airports_list", summary="Retrieves a limited list of airports, each with a unique code and descriptive text. The number of results returned can be controlled using the provided limit parameter.")
async def get_airports_list(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Code, Description FROM Airports LIMIT ?", (limit,))
    results = cursor.fetchall()
    if not results:
        return {"airports": []}
    return {"airports": [{"code": row[0], "description": row[1]} for row in results]}

# Endpoint to get air carrier codes based on description
@app.get("/v1/airline/air_carrier_codes", operation_id="get_air_carrier_codes", summary="Retrieves the unique code of an air carrier based on a provided description. The description is used to search for a matching air carrier in the database, and the corresponding code is returned.")
async def get_air_carrier_codes(description: str = Query(..., description="Description of the air carrier")):
    cursor.execute("SELECT Code FROM `Air Carriers` WHERE Description LIKE ?", (description,))
    results = cursor.fetchall()
    if not results:
        return {"codes": []}
    return {"codes": [row[0] for row in results]}

# Endpoint to get scheduled and actual departure times for a specific flight
@app.get("/v1/airline/departure_times", operation_id="get_departure_times", summary="Retrieves the scheduled and actual departure times for a specific flight based on the origin and destination airports, the aircraft's tail number, and the flight date. The response includes the scheduled departure time (CRS_DEP_TIME) and the actual departure time (DEP_TIME) for the given flight.")
async def get_departure_times(origin: str = Query(..., description="Origin airport code"), dest: str = Query(..., description="Destination airport code"), tail_num: str = Query(..., description="Tail number of the aircraft"), fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT CRS_DEP_TIME, DEP_TIME FROM Airlines WHERE ORIGIN = ? AND DEST = ? AND TAIL_NUM = ? AND FL_DATE = ?", (origin, dest, tail_num, fl_date))
    result = cursor.fetchone()
    if not result:
        return {"crs_dep_time": [], "dep_time": []}
    return {"crs_dep_time": result[0], "dep_time": result[1]}

# Endpoint to get all destinations from a specific origin
@app.get("/v1/airline/destinations_from_origin", operation_id="get_destinations_from_origin", summary="Retrieves a list of unique destinations that can be reached from the specified origin airport. The origin is identified by its airport code.")
async def get_destinations_from_origin(origin: str = Query(..., description="Origin airport code")):
    cursor.execute("SELECT DEST FROM Airlines WHERE ORIGIN = ? GROUP BY DEST", (origin,))
    results = cursor.fetchall()
    if not results:
        return {"destinations": []}
    return {"destinations": [row[0] for row in results]}

# Endpoint to get the count of flights with specific origin, destination, and departure delay
@app.get("/v1/airline/flight_count_origin_dest_delay", operation_id="get_flight_count_origin_dest_delay", summary="Retrieves the total number of flights that departed from a specific origin airport, arrived at a specific destination airport, and experienced a certain amount of departure delay. The origin and destination airports are identified by their respective codes, and the departure delay is measured in minutes.")
async def get_flight_count_origin_dest_delay(dest: str = Query(..., description="Destination airport code"), origin: str = Query(..., description="Origin airport code"), dep_delay: int = Query(..., description="Departure delay in minutes")):
    cursor.execute("SELECT COUNT(*) FROM Airlines WHERE DEST = ? AND ORIGIN = ? AND DEP_DELAY = ?", (dest, origin, dep_delay))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of flights with specific origin, destination, airport description, and cancellation code
@app.get("/v1/airline/flight_count_origin_dest_description_cancellation", operation_id="get_flight_count_origin_dest_description_cancellation", summary="Retrieves the total number of flights from a specific origin to a specific destination, with the origin airport being 'Charlotte, NC: Charlotte Douglas International', and a particular cancellation code. The origin and destination are identified by their respective airport codes.")
async def get_flight_count_origin_dest_description_cancellation(origin: str = Query(..., description="Origin airport code"), dest: str = Query(..., description="Destination airport code"), description: str = Query(..., description="Airport description"), cancellation_code: str = Query(..., description="Cancellation code")):
    cursor.execute("SELECT COUNT(*) FROM Airlines AS T1 INNER JOIN Airports AS T2 ON T1.ORIGIN = T2.Code WHERE T1.ORIGIN = ? AND T1.DEST = ? AND T2.Description = ? AND T1.CANCELLATION_CODE = ?", (origin, dest, description, cancellation_code))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the air carrier description with the highest number of cancelled flights
@app.get("/v1/airline/max_cancelled_flights_carrier", operation_id="get_max_cancelled_flights_carrier", summary="Retrieves the description of the air carrier with the highest number of cancelled flights, up to the specified limit.")
async def get_max_cancelled_flights_carrier(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Description FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID ORDER BY T2.CANCELLED DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get airport description based on airport code
@app.get("/v1/airline/airport_description", operation_id="get_airport_description", summary="Retrieves the description of a specific airport based on its unique code. The operation returns a detailed description of the airport, providing information about its location, facilities, and services.")
async def get_airport_description(code: str = Query(..., description="Airport code")):
    cursor.execute("SELECT Description FROM Airports WHERE Code = ?", (code,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the count of flights based on origin
@app.get("/v1/airline/flight_count_by_origin", operation_id="get_flight_count_by_origin", summary="Retrieves the total number of flights originating from the specified airport. The origin parameter is used to filter the results.")
async def get_flight_count_by_origin(origin: str = Query(..., description="Origin airport code")):
    cursor.execute("SELECT COUNT(*) AS num FROM Airlines WHERE Origin = ?", (origin,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of airports based on code pattern
@app.get("/v1/airline/airport_count_by_code_pattern", operation_id="get_airport_count_by_code_pattern", summary="Retrieves the total number of airports whose codes match the specified pattern. The pattern can include a wildcard character to broaden the search. This operation is useful for determining the prevalence of certain code patterns in the airport database.")
async def get_airport_count_by_code_pattern(code_pattern: str = Query(..., description="Code pattern to match (use % for wildcard)")):
    cursor.execute("SELECT COUNT(*) FROM Airports WHERE Code LIKE ?", (code_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get destinations based on operating carrier flight number
@app.get("/v1/airline/destinations_by_flight_number", operation_id="get_destinations_by_flight_number", summary="Retrieves the destination(s) associated with a specific operating carrier flight number. The input parameter specifies the operating carrier flight number for which the destination(s) are to be returned.")
async def get_destinations_by_flight_number(op_carrier_fl_num: int = Query(..., description="Operating carrier flight number")):
    cursor.execute("SELECT DEST FROM Airlines WHERE OP_CARRIER_FL_NUM = ?", (op_carrier_fl_num,))
    result = cursor.fetchall()
    if not result:
        return {"destinations": []}
    return {"destinations": [item[0] for item in result]}

# Endpoint to get airport descriptions based on code pattern
@app.get("/v1/airline/airport_descriptions_by_code_pattern", operation_id="get_airport_descriptions_by_code_pattern", summary="Retrieves a list of airport descriptions that match the provided code pattern. The code pattern can include a wildcard character to broaden the search. The endpoint returns the description of each airport that matches the specified code pattern.")
async def get_airport_descriptions_by_code_pattern(code_pattern: str = Query(..., description="Code pattern to match (use % for wildcard)")):
    cursor.execute("SELECT Description FROM Airports WHERE Code LIKE ?", (code_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [item[0] for item in result]}

# Endpoint to get the count of cancelled flights
@app.get("/v1/airline/cancelled_flight_count", operation_id="get_cancelled_flight_count", summary="Retrieves the total count of flights that have been cancelled. The operation filters flights based on the provided cancellation status, where 1 indicates cancelled flights and 0 indicates non-cancelled flights.")
async def get_cancelled_flight_count(cancelled: int = Query(..., description="Cancelled status (1 for cancelled, 0 for not cancelled)")):
    cursor.execute("SELECT COUNT(*) FROM Airlines WHERE CANCELLED = ?", (cancelled,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get tail numbers based on flight date
@app.get("/v1/airline/tail_numbers_by_flight_date", operation_id="get_tail_numbers_by_flight_date", summary="Retrieves a list of unique tail numbers for flights that occurred on a specific date. The date is provided in the 'YYYY/MM/DD' format.")
async def get_tail_numbers_by_flight_date(fl_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT TAIL_NUM FROM Airlines WHERE FL_DATE = ? GROUP BY TAIL_NUM", (fl_date,))
    result = cursor.fetchall()
    if not result:
        return {"tail_numbers": []}
    return {"tail_numbers": [item[0] for item in result]}

# Endpoint to get the origin of the flight with the shortest actual elapsed time
@app.get("/v1/airline/origin_shortest_elapsed_time", operation_id="get_origin_shortest_elapsed_time", summary="Retrieves the origin of the flight with the shortest actual elapsed time, allowing the user to limit the number of results returned.")
async def get_origin_shortest_elapsed_time(limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT ORIGIN FROM Airlines ORDER BY ACTUAL_ELAPSED_TIME ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"origin": []}
    return {"origin": result[0]}

# Endpoint to get flight date and tail number for a specific air carrier description
@app.get("/v1/airline/flight_date_tail_num_by_description", operation_id="get_flight_date_tail_num_by_description", summary="Retrieves the flight date and tail number of flights operated by a specific air carrier. The operation requires the description of the air carrier as input and returns the corresponding flight date and tail number.")
async def get_flight_date_tail_num_by_description(description: str = Query(..., description="Description of the air carrier")):
    cursor.execute("SELECT T1.FL_DATE, T1.TAIL_NUM FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T1.OP_CARRIER_AIRLINE_ID = T2.Code WHERE T2.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"flights": []}
    return {"flights": result}

# Endpoint to get the description and code of the air carrier with the earliest arrival time
@app.get("/v1/airline/air_carrier_earliest_arrival", operation_id="get_air_carrier_earliest_arrival", summary="Get the description and code of the air carrier with the earliest arrival time")
async def get_air_carrier_earliest_arrival(limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT T1.Description, T1.Code FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID ORDER BY T2.ARR_TIME ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"air_carrier": []}
    return {"air_carrier": result}

# Endpoint to get the count of flights for a specific air carrier description with no arrival delay
@app.get("/v1/airline/count_flights_no_arrival_delay", operation_id="get_count_flights_no_arrival_delay", summary="Retrieves the total number of flights for a specific airline that did not experience any arrival delay. The airline is identified by its description, which can include wildcard characters for a broader search. The arrival delay is determined by a specified value, with zero indicating no delay.")
async def get_count_flights_no_arrival_delay(description: str = Query(..., description="Description of the air carrier (use %% for wildcard)"), arr_delay_new: int = Query(0, description="Arrival delay (0 for no delay)")):
    cursor.execute("SELECT COUNT(*) FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description LIKE ? AND T2.ARR_DELAY_NEW = ?", (description, arr_delay_new))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the description of air carriers flying to a specific destination
@app.get("/v1/airline/air_carrier_description_by_destination", operation_id="get_air_carrier_description_by_destination", summary="Retrieves the description of air carriers that operate flights to the specified destination airport. The destination is identified by its unique airport code.")
async def get_air_carrier_description_by_destination(destination: str = Query(..., description="Destination airport code")):
    cursor.execute("SELECT T1.Description FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T2.DEST = ?", (destination,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the description of air carriers with cancelled flights
@app.get("/v1/airline/air_carrier_description_cancelled_flights", operation_id="get_air_carrier_description_cancelled_flights", summary="Retrieves a list of air carrier descriptions for flights that have been cancelled. The operation filters the results based on the provided cancelled status, allowing users to view specific cancellation categories.")
async def get_air_carrier_description_cancelled_flights(cancelled: int = Query(1, description="Cancelled status (1 for cancelled)")):
    cursor.execute("SELECT T1.Description FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T2.CANCELLED = ? GROUP BY T1.Description", (cancelled,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the actual elapsed time for flights operated by a specific air carrier
@app.get("/v1/airline/actual_elapsed_time_by_description", operation_id="get_actual_elapsed_time_by_description", summary="Retrieves the actual elapsed time for flights operated by a specific air carrier, as identified by its description. The operation returns the elapsed time data for the specified air carrier, enabling users to analyze flight durations.")
async def get_actual_elapsed_time_by_description(description: str = Query(..., description="Description of the air carrier")):
    cursor.execute("SELECT T2.ACTUAL_ELAPSED_TIME FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"elapsed_times": []}
    return {"elapsed_times": [row[0] for row in result]}

# Endpoint to get the tail number of the flight with the highest departure delay for a specific air carrier
@app.get("/v1/airline/tail_num_highest_departure_delay", operation_id="get_tail_num_highest_departure_delay", summary="Retrieve the tail number of the flight with the highest departure delay for a specific air carrier. The operation filters flights based on the provided air carrier description and returns the tail number of the flight with the longest departure delay. The number of results can be limited by specifying a value for the limit parameter.")
async def get_tail_num_highest_departure_delay(description: str = Query(..., description="Description of the air carrier"), limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT T2.TAIL_NUM FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description = ? ORDER BY T2.DEP_DELAY DESC LIMIT ?", (description, limit))
    result = cursor.fetchone()
    if not result:
        return {"tail_num": []}
    return {"tail_num": result[0]}

# Endpoint to get the description of air carriers with flights having a specific departure delay
@app.get("/v1/airline/air_carrier_description_by_departure_delay", operation_id="get_air_carrier_description_by_departure_delay", summary="Retrieve the descriptions of air carriers whose flights have experienced a specified departure delay. The delay is provided in minutes. The descriptions are grouped to avoid duplicates.")
async def get_air_carrier_description_by_departure_delay(dep_delay: int = Query(..., description="Departure delay in minutes")):
    cursor.execute("SELECT T1.Description FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T2.DEP_DELAY = ? GROUP BY T1.Description", (dep_delay,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the description of the air carrier with the longest actual elapsed time
@app.get("/v1/airline/air_carrier_longest_elapsed_time", operation_id="get_air_carrier_longest_elapsed_time", summary="Retrieves the description of the air carrier with the longest actual elapsed time, as determined by the data in the Airlines table. The operation allows you to limit the number of results returned.")
async def get_air_carrier_longest_elapsed_time(limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT T1.Description FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID ORDER BY T2.ACTUAL_ELAPSED_TIME DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the destination of airlines based on air carrier description
@app.get("/v1/airline/get_destinations_by_description", operation_id="get_destinations_by_description", summary="Retrieves the destinations of airlines that match the provided description of the air carrier. The description parameter is used to filter the results.")
async def get_destinations_by_description(description: str = Query(..., description="Description of the air carrier")):
    cursor.execute("SELECT T2.DEST FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"destinations": []}
    return {"destinations": [row[0] for row in result]}

# Endpoint to get the count of flights based on air carrier description, cancellation status, and flight date range
@app.get("/v1/airline/get_flight_count_by_description_cancelled_date_range", operation_id="get_flight_count_by_description_cancelled_date_range", summary="Retrieve the total number of flights that match a specific air carrier description, cancellation status, and date range. The description parameter filters flights by the air carrier's description, while the cancelled parameter determines whether to include cancelled flights. The start_date and end_date parameters define the date range for the flights to be considered.")
async def get_flight_count_by_description_cancelled_date_range(description: str = Query(..., description="Description of the air carrier"), cancelled: int = Query(..., description="Cancellation status (0 for not cancelled, 1 for cancelled)"), start_date: str = Query(..., description="Start date in 'YYYY/MM/DD' format"), end_date: str = Query(..., description="End date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT COUNT(*) FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description = ? AND T2.CANCELLED = ? AND T2.FL_DATE BETWEEN ? AND ?", (description, cancelled, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of flights based on air carrier description pattern and flight date
@app.get("/v1/airline/get_flight_count_by_description_pattern_date", operation_id="get_flight_count_by_description_pattern_date", summary="Retrieves the total number of flights that match a specified description pattern for the air carrier and a given flight date. The description pattern supports wildcard characters for flexible matching. The flight date must be provided in 'YYYY/MM/DD' format.")
async def get_flight_count_by_description_pattern_date(description_pattern: str = Query(..., description="Description pattern of the air carrier (use %% for wildcard)"), flight_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT COUNT(*) FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description LIKE ? AND T2.FL_DATE = ?", (description_pattern, flight_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the tail numbers of flights based on arrival time and air carrier description
@app.get("/v1/airline/get_tail_numbers_by_arrival_time_description", operation_id="get_tail_numbers_by_arrival_time_description", summary="Retrieve the tail numbers of flights that arrived before the specified time and are operated by the air carrier with the given description.")
async def get_tail_numbers_by_arrival_time_description(arrival_time: int = Query(..., description="Arrival time in 24-hour format (e.g., 1000 for 10:00 AM)"), description: str = Query(..., description="Description of the air carrier")):
    cursor.execute("SELECT T2.TAIL_NUM FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T2.ARR_TIME <= ? AND T1.Description = ?", (arrival_time, description))
    result = cursor.fetchall()
    if not result:
        return {"tail_numbers": []}
    return {"tail_numbers": [row[0] for row in result]}

# Endpoint to get the flight dates based on actual elapsed time and air carrier description
@app.get("/v1/airline/get_flight_dates_by_elapsed_time_description", operation_id="get_flight_dates_by_elapsed_time_description", summary="Retrieve the flight dates for flights with an actual elapsed time less than the provided duration and operated by the specified air carrier. The air carrier is identified by its description.")
async def get_flight_dates_by_elapsed_time_description(elapsed_time: int = Query(..., description="Actual elapsed time in minutes"), description: str = Query(..., description="Description of the air carrier")):
    cursor.execute("SELECT T2.FL_DATE FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T2.ACTUAL_ELAPSED_TIME < ? AND T1.Description = ?", (elapsed_time, description))
    result = cursor.fetchall()
    if not result:
        return {"flight_dates": []}
    return {"flight_dates": [row[0] for row in result]}

# Endpoint to get the count of flights based on air carrier description pattern and departure delay
@app.get("/v1/airline/get_flight_count_by_description_pattern_dep_delay", operation_id="get_flight_count_by_description_pattern_dep_delay", summary="Retrieves the total number of flights that match a specified description pattern for the air carrier and have a departure delay greater than a given value. The description pattern supports wildcard characters (%%) for flexible matching. The departure delay is measured in minutes.")
async def get_flight_count_by_description_pattern_dep_delay(description_pattern: str = Query(..., description="Description pattern of the air carrier (use %% for wildcard)"), dep_delay: int = Query(..., description="Departure delay in minutes")):
    cursor.execute("SELECT COUNT(*) FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description LIKE ? AND T2.DEP_DELAY > ?", (description_pattern, dep_delay))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the descriptions of air carriers based on flight date
@app.get("/v1/airline/get_descriptions_by_flight_date", operation_id="get_descriptions_by_flight_date", summary="Retrieves the descriptions of air carriers that have flights scheduled on the specified date. The descriptions are grouped to eliminate duplicates.")
async def get_descriptions_by_flight_date(flight_date: str = Query(..., description="Flight date in 'YYYY/MM/DD' format")):
    cursor.execute("SELECT T1.Description FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T2.FL_DATE = ? GROUP BY T1.Description", (flight_date,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the descriptions of air carriers based on tail number and origin
@app.get("/v1/airline/get_descriptions_by_tail_num_origin", operation_id="get_descriptions_by_tail_num_origin", summary="Retrieve the descriptions of air carriers that match the specified tail number and origin airport code. The operation returns a list of unique descriptions, grouped by the air carrier's code.")
async def get_descriptions_by_tail_num_origin(tail_num: str = Query(..., description="Tail number of the aircraft"), origin: str = Query(..., description="Origin airport code")):
    cursor.execute("SELECT T2.Description FROM Airlines AS T1 INNER JOIN `Air Carriers` AS T2 ON T2.Code = T1.OP_CARRIER_AIRLINE_ID WHERE T1.TAIL_NUM = ? AND T1.ORIGIN = ? GROUP BY T2.Description", (tail_num, origin))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the descriptions of air carriers based on arrival and departure delays
@app.get("/v1/airline/get_descriptions_by_arrival_departure_delays", operation_id="get_descriptions_by_arrival_departure_delays", summary="Retrieve the descriptions of air carriers that have both arrival and departure delays less than the specified values. The results are grouped by the description of the air carrier.")
async def get_descriptions_by_arrival_departure_delays(arr_delay: int = Query(..., description="Arrival delay in minutes"), dep_delay: int = Query(..., description="Departure delay in minutes")):
    cursor.execute("SELECT T1.Description FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T2.ARR_DELAY < ? AND T2.DEP_DELAY < ? GROUP BY T1.Description", (arr_delay, dep_delay))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the tail numbers of flights based on air carrier description and elapsed time below 80% of the average
@app.get("/v1/airline/get_tail_numbers_by_description_elapsed_time", operation_id="get_tail_numbers_by_description_elapsed_time", summary="Retrieves the tail numbers of flights operated by the air carrier with the specified description and an elapsed time below 80% of the average elapsed time.")
async def get_tail_numbers_by_description_elapsed_time(description: str = Query(..., description="Description of the air carrier")):
    cursor.execute("SELECT T2.TAIL_NUM FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description = ? AND T2.ACTUAL_ELAPSED_TIME < ( SELECT AVG(ACTUAL_ELAPSED_TIME) * 0.8 FROM Airlines )", (description,))
    result = cursor.fetchall()
    if not result:
        return {"tail_numbers": []}
    return {"tail_numbers": [row[0] for row in result]}

# Endpoint to get air carrier descriptions for flights to a specific destination with arrival times less than a percentage of the average arrival time
@app.get("/v1/airline/carrier_descriptions_by_destination_arrival_time", operation_id="get_carrier_descriptions", summary="Retrieves descriptions of air carriers that operate flights to a specified destination, with arrival times less than a given percentage of the average arrival time. The results are grouped by air carrier description.")
async def get_carrier_descriptions(destination: str = Query(..., description="Destination airport code (e.g., 'PHX')"), percentage: float = Query(..., description="Percentage of the average arrival time")):
    cursor.execute("SELECT T1.Description FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T2.DEST = ? AND T2.ARR_TIME < ( SELECT AVG(ARR_TIME) * ? FROM Airlines ) GROUP BY T1.Description", (destination, percentage))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the percentage of flights with departure delays for a specific air carrier description
@app.get("/v1/airline/percentage_departure_delays_by_description", operation_id="get_percentage_departure_delays", summary="Retrieves the percentage of flights with departure delays for a specific air carrier. The air carrier is identified by its description, which is used to filter the results. The calculation is based on the total number of flights for the specified air carrier.")
async def get_percentage_departure_delays(description: str = Query(..., description="Air carrier description (e.g., '%American Airlines%')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.DEP_DELAY < 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM `Air Carriers` AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.OP_CARRIER_AIRLINE_ID WHERE T1.Description LIKE ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/airline/flight_count_by_date?fl_date=2018/8/1",
    "/v1/airline/flight_count_by_date_and_origin?fl_date=2018/8/1&origin=JFK",
    "/v1/airline/cancelled_flights_destinations_by_date?fl_date=2018/8/1&cancelled=1",
    "/v1/airline/flight_dates_by_cancellation_code?cancellation_code=A",
    "/v1/airline/airport_descriptions_with_delayed_departures?fl_date=2018/8/1&dep_delay=0",
    "/v1/airline/flight_count_from_airport_by_date?fl_date=2018/8/1&description=New%20York%2C%20NY%3A%20John%20F.%20Kennedy%20International",
    "/v1/airline/airport_with_highest_departure_delay?fl_date=2018/8/1",
    "/v1/airline/flight_count_with_arrival_delay_less_than?arr_delay=0&description=New%20York%2C%20NY%3A%20John%20F.%20Kennedy%20International",
    "/v1/airline/earliest_departure_time_from_airport?fl_date=2018/8/1&description=New%20York%2C%20NY%3A%20John%20F.%20Kennedy%20International",
    "/v1/airline/flight_count_by_air_carrier_and_date?fl_date=2018/8/1&description=American%20Airlines%20Inc.%3A%20AA",
    "/v1/airline/flight_numbers_by_airline_airport_date?airline_description=American%20Airlines%20Inc.%3A%20AA&airport_description=New%20York%2C%20NY%3A%20John%20F.%20Kennedy%20International&flight_date=2018/8/1",
    "/v1/airline/count_flights_actual_elapsed_time_less_than_scheduled?flight_date=2018/8/1&airline_description=American%20Airlines%20Inc.%3A%20AA",
    "/v1/airline/latest_flight_number_by_airline?airline_description=American%20Airlines%20Inc.%3A%20AA",
    "/v1/airline/count_flights_to_destination_by_airline?destination=JFK&airline_description=American%20Airlines%20Inc.%3A%20AA",
    "/v1/airline/count_cancelled_flights_by_airline_date?flight_date=2018/8/1&airline_description=American%20Airlines%20Inc.%3A%20AA",
    "/v1/airline/compare_flights_between_airlines?airline_description_1=American%20Airlines%20Inc.%3A%20AA&airline_description_2=Endeavor%20Air%20Inc.%3A%209E&flight_date=2018/8/1",
    "/v1/airline/average_departure_delay_by_airline?airline_description=American%20Airlines%20Inc.%3A%20AA",
    "/v1/airline/average_flights_per_day_by_airline_month?flight_date_pattern=2018/8%25&airline_description=American%20Airlines%20Inc.%3A%20AA",
    "/v1/airline/count_air_carriers",
    "/v1/airline/count_flights_from_airport_date?flight_date=2018/8/27&airport_description=Los%20Angeles%2C%20CA%3A%20Los%20Angeles%20International",
    "/v1/airline/count_airports_description_substring?description_substring=Oakland&fl_date=2018/8/7",
    "/v1/airline/count_flights_departure_delay?fl_date=2018/8/2&airline_description=Alaska%20Airlines%20Inc.:%20AS",
    "/v1/airline/count_flights_arrival_delay_less_than_zero?fl_date=2018/8/12&destination=MIA",
    "/v1/airline/count_flights_elapsed_time_less_than_scheduled?fl_date=2018/8/31&airline_description=Endeavor%20Air%20Inc.:%209E",
    "/v1/airline/air_carrier_descriptions_by_tail_num?tail_num=N702SK",
    "/v1/airline/latest_destination_airport_description?fl_date=2018/8/15",
    "/v1/airline/air_carrier_descriptions_by_flight?fl_date=2018/8/1&origin=ATL&destination=PHL&crs_dep_time=2040",
    "/v1/airline/count_airports_by_description?fl_date=2018/8/15&airport_description=Lake%20Charles,%20LA:%20Lake%20Charles%20Regional",
    "/v1/airline/flight_count_between_airports?fl_date_pattern=2018/8%25&origin_description=San%20Diego,%20CA:%20San%20Diego%20International&dest_description=Los%20Angeles,%20CA:%20Los%20Angeles%20International",
    "/v1/airline/cancellation_percentage_carrier?fl_date=2018/8/15&airport_description=Los%20Angeles,%20CA:%20Los%20Angeles%20International",
    "/v1/airline/early_arrival_percentage?destination_description=%25Pittsburgh%25",
    "/v1/airline/air_carrier_description?code=19049",
    "/v1/airline/flight_count_dep_delay?fl_date=2018/8/1&dep_delay=0",
    "/v1/airline/airport_code_by_description?description_pattern=%25Ankara,%20Turkey%25",
    "/v1/airline/max_weather_delay?origin_airport_id=12264",
    "/v1/airline/airport_code_by_exact_description?description=Anita%20Bay,%20AK:%20Anita%20Bay%20Airport",
    "/v1/airline/max_late_aircraft_delay_origin",
    "/v1/airline/flight_count_between_airports_general?origin_description=Chicago,%20IL:%20Chicago%20O'Hare%20International&dest_description=Atlanta,%20GA:%20Hartsfield-Jackson%20Atlanta%20International",
    "/v1/airline/count_tail_numbers_by_airline?airline_description=Southwest%20Airlines%20Co.:%20WN",
    "/v1/airline/most_frequent_flight_date?fl_date=2018/8%25&airport_description=Dallas/Fort%20Worth,%20TX:%20Dallas/Fort%20Worth%20International&origin=DFW&cancelled=1&cancellation_code=A",
    "/v1/airline/tail_numbers_by_destination_arrival_delay?fl_date=2018/8%25&airport_description=Bakersfield,%20CA:%20Meadows%20Field&dest=BFL&arr_delay=0",
    "/v1/airline/highest_security_delay_airline?airport_description=Boston,%20MA:%20Logan%20International&dest=BOS",
    "/v1/airline/top_airline_descriptions",
    "/v1/airline/most_cancellations?cancelled=0",
    "/v1/airline/most_frequent_airline_description?airport_description=Chicago,%20IL:%20Chicago%20Midway%20International&dest=MDW",
    "/v1/airline/most_frequent_airline_id?airline_description=Compass%20Airlines:%20CP&origin=LAX&dest=ABQ",
    "/v1/airline/most_frequent_destination?airline_description=Republic%20Airline:%20YX",
    "/v1/airline/min_elapsed_time_difference?limit=1",
    "/v1/airline/max_delay_tail_number?fl_date=2018/8/%&description=Delta%20Air%20Lines%20Inc.:%20DL&limit=1",
    "/v1/airline/airports_list?limit=3",
    "/v1/airline/air_carrier_codes?description=Mississippi%20Valley%20Airlines%25",
    "/v1/airline/departure_times?origin=PHL&dest=MDT&tail_num=N627AE&fl_date=2018/8/13",
    "/v1/airline/destinations_from_origin?origin=ABY",
    "/v1/airline/flight_count_origin_dest_delay?dest=SNA&origin=DFW&dep_delay=0",
    "/v1/airline/flight_count_origin_dest_description_cancellation?origin=CLT&dest=AUS&description=Charlotte,%20NC:%20Charlotte%20Douglas%20International&cancellation_code=A",
    "/v1/airline/max_cancelled_flights_carrier?limit=1",
    "/v1/airline/airport_description?code=A11",
    "/v1/airline/flight_count_by_origin?origin=OKC",
    "/v1/airline/airport_count_by_code_pattern?code_pattern=C%",
    "/v1/airline/destinations_by_flight_number?op_carrier_fl_num=1596",
    "/v1/airline/airport_descriptions_by_code_pattern?code_pattern=%3",
    "/v1/airline/cancelled_flight_count?cancelled=1",
    "/v1/airline/tail_numbers_by_flight_date?fl_date=2018/8/17",
    "/v1/airline/origin_shortest_elapsed_time?limit=1",
    "/v1/airline/flight_date_tail_num_by_description?description=Ross%20Aviation%20Inc.:%20GWE",
    "/v1/airline/air_carrier_earliest_arrival?limit=1",
    "/v1/airline/count_flights_no_arrival_delay?description=%25JetBlue%20Airways:%20B6%25&arr_delay_new=0",
    "/v1/airline/air_carrier_description_by_destination?destination=MIA",
    "/v1/airline/air_carrier_description_cancelled_flights?cancelled=1",
    "/v1/airline/actual_elapsed_time_by_description?description=Semo%20Aviation%20Inc.:%20SEM",
    "/v1/airline/tail_num_highest_departure_delay?description=Asap%20Air%20Inc.:%20ASP&limit=1",
    "/v1/airline/air_carrier_description_by_departure_delay?dep_delay=0",
    "/v1/airline/air_carrier_longest_elapsed_time?limit=1",
    "/v1/airline/get_destinations_by_description?description=Southeast%20Alaska%20Airlines:%20WEB",
    "/v1/airline/get_flight_count_by_description_cancelled_date_range?description=Spirit%20Air%20Lines:%20NK&cancelled=0&start_date=2018/8/10&end_date=2018/8/20",
    "/v1/airline/get_flight_count_by_description_pattern_date?description_pattern=%25Horizon%20Air%25&flight_date=2018/8/2",
    "/v1/airline/get_tail_numbers_by_arrival_time_description?arrival_time=1000&description=Iscargo%20Hf:%20ICQ",
    "/v1/airline/get_flight_dates_by_elapsed_time_description?elapsed_time=100&description=Profit%20Airlines%20Inc.:%20XBH",
    "/v1/airline/get_flight_count_by_description_pattern_dep_delay?description_pattern=%25Republic%20Airline%25&dep_delay=30",
    "/v1/airline/get_descriptions_by_flight_date?flight_date=2018/8/25",
    "/v1/airline/get_descriptions_by_tail_num_origin?tail_num=N922US&origin=PHX",
    "/v1/airline/get_descriptions_by_arrival_departure_delays?arr_delay=0&dep_delay=0",
    "/v1/airline/get_tail_numbers_by_description_elapsed_time?description=Southwest%20Airlines%20Co.:%20WN",
    "/v1/airline/carrier_descriptions_by_destination_arrival_time?destination=PHX&percentage=0.4",
    "/v1/airline/percentage_departure_delays_by_description?description=%25American%20Airlines%25"
]
