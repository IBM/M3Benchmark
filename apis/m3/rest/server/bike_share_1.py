from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/bike_share_1/bike_share_1.sqlite')
cursor = conn.cursor()

# Endpoint to get the start and end station names for the trip with the maximum duration
@app.get("/v1/bike_share_1/trip_with_max_duration", operation_id="get_trip_with_max_duration", summary="Retrieves the names of the start and end stations for the trip with the longest duration.")
async def get_trip_with_max_duration():
    cursor.execute("SELECT start_station_name, end_station_name FROM trip WHERE duration = ( SELECT MAX(duration) FROM trip )")
    result = cursor.fetchone()
    if not result:
        return {"stations": []}
    return {"start_station_name": result[0], "end_station_name": result[1]}

# Endpoint to get the percentage of trips by a specific subscription type
@app.get("/v1/bike_share_1/subscription_type_percentage", operation_id="get_subscription_type_percentage", summary="Retrieves the percentage of trips associated with a particular subscription type. The subscription type is specified as an input parameter, and the result is calculated by comparing the count of trips with the specified subscription type to the total number of trips.")
async def get_subscription_type_percentage(subscription_type: str = Query(..., description="Subscription type (e.g., 'Subscriber')")):
    cursor.execute("SELECT CAST(COUNT(subscription_type) AS REAL) * 100 / ( SELECT COUNT(subscription_type) FROM trip ) FROM trip WHERE subscription_type = ?", (subscription_type,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the end station ID and city for the most recent trip of a specific bike
@app.get("/v1/bike_share_1/recent_trip_by_bike_id", operation_id="get_recent_trip_by_bike_id", summary="Retrieves the end station ID and city of the most recent trip for a specific bike. The operation uses the provided bike ID to search for the corresponding trip data, which is then joined with the station data to provide the end station ID and city. The results are sorted by end date in descending order and limited to the most recent trip.")
async def get_recent_trip_by_bike_id(bike_id: int = Query(..., description="Bike ID")):
    cursor.execute("SELECT T2.end_station_id, T1.city FROM station AS T1 INNER JOIN trip AS T2 ON T1.name = T2.end_station_name WHERE T2.bike_id = ? ORDER BY T2.end_date DESC LIMIT 1", (bike_id,))
    result = cursor.fetchone()
    if not result:
        return {"end_station_id": [], "city": []}
    return {"end_station_id": result[0], "city": result[1]}

# Endpoint to get distinct cities where trips have different start and end months
@app.get("/v1/bike_share_1/cities_with_different_start_end_months", operation_id="get_cities_with_different_start_end_months", summary="Retrieve a list of unique cities where bike trips have different start and end months. This operation identifies cities with trips that span across months, providing insights into cycling patterns and seasonal variations.")
async def get_cities_with_different_start_end_months():
    cursor.execute("SELECT DISTINCT T1.city FROM station AS T1 INNER JOIN trip AS T2 ON T2.start_station_name = T1.name WHERE SUBSTR(CAST(T2.start_date AS TEXT), INSTR(T2.start_date, '/') + 1) - SUBSTR(CAST(T2.start_date AS TEXT), INSTR(T2.start_date, ' ') - 5) <> SUBSTR(CAST(T2.end_date AS TEXT), INSTR(T2.end_date, '/') + 1) - SUBSTR(CAST(T2.end_date AS TEXT), INSTR(T2.end_date, ' ') - 5)")
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get station names and longitudes with no bikes available at a specific time
@app.get("/v1/bike_share_1/stations_no_bikes_available", operation_id="get_stations_no_bikes_available", summary="Retrieves the names and longitudes of bike share stations that have no bikes available at a specific time. The time and the number of bikes available are provided as input parameters.")
async def get_stations_no_bikes_available(time: str = Query(..., description="Time in 'YYYY/MM/DD HH:MM:SS' format"), bikes_available: int = Query(..., description="Number of bikes available")):
    cursor.execute("SELECT T1.name, T1.long FROM station AS T1 INNER JOIN status AS T2 ON T2.station_id = T1.id WHERE T2.time = ? AND T2.bikes_available = ?", (time, bikes_available))
    result = cursor.fetchall()
    if not result:
        return {"stations": []}
    return {"stations": [{"name": row[0], "long": row[1]} for row in result]}

# Endpoint to get the most popular start station and its city
@app.get("/v1/bike_share_1/most_popular_start_station", operation_id="get_most_popular_start_station", summary="Retrieves the most frequently used starting station for bike trips and its corresponding city. The station is determined by counting the number of trips that started from each station and selecting the one with the highest count.")
async def get_most_popular_start_station():
    cursor.execute("SELECT T2.start_station_name, T1.city FROM station AS T1 INNER JOIN trip AS T2 ON T2.start_station_name = T1.name GROUP BY T2.start_station_name ORDER BY COUNT(T2.start_station_name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"start_station_name": [], "city": []}
    return {"start_station_name": result[0], "city": result[1]}

# Endpoint to get the maximum temperature for a specific trip ID
@app.get("/v1/bike_share_1/max_temperature_for_trip", operation_id="get_max_temperature_for_trip", summary="Retrieves the highest temperature recorded during a specific trip, based on the trip's start date and zip code. The trip ID is used to identify the relevant trip data.")
async def get_max_temperature_for_trip(trip_id: int = Query(..., description="Trip ID")):
    cursor.execute("SELECT MAX(T2.max_temperature_f) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code AND T2.date = SUBSTR(CAST(T1.start_date AS TEXT), 1, INSTR(T1.start_date, ' ') - 1) WHERE T1.id = ?", (trip_id,))
    result = cursor.fetchone()
    if not result:
        return {"max_temperature": []}
    return {"max_temperature": result[0]}

# Endpoint to get the time when the maximum number of bikes were available at a specific station
@app.get("/v1/bike_share_1/max_bikes_available_time", operation_id="get_max_bikes_available_time", summary="Get the time when the maximum number of bikes were available at a specific station")
async def get_max_bikes_available_time(station_name: str = Query(..., description="Station name")):
    cursor.execute("SELECT T2.time FROM station AS T1 INNER JOIN status AS T2 ON T2.station_id = T1.id WHERE T1.name = ? AND T2.bikes_available = ( SELECT MAX(T2.bikes_available) FROM station AS T1 INNER JOIN status AS T2 ON T2.station_id = T1.id WHERE T1.name = ? )", (station_name, station_name))
    result = cursor.fetchone()
    if not result:
        return {"time": []}
    return {"time": result[0]}

# Endpoint to get trip IDs and durations for trips that occurred during rain
@app.get("/v1/bike_share_1/trips_during_rain", operation_id="get_trips_during_rain", summary="Retrieves the IDs and durations of trips that took place during rainy weather. The operation filters trips based on the specified weather events, which are matched against the weather data associated with the trips' start dates and zip codes.")
async def get_trips_during_rain(event1: str = Query(..., description="Weather event (e.g., '%Rain%')"), event2: str = Query(..., description="Weather event (e.g., '%rain%')")):
    cursor.execute("SELECT T1.id, T1.duration FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code AND T2.date = SUBSTR(CAST(T1.start_date AS TEXT), 1, INSTR(T1.start_date, ' ') - 1) WHERE T2.events LIKE ? OR T2.events LIKE ?", (event1, event2))
    result = cursor.fetchall()
    if not result:
        return {"trips": []}
    return {"trips": [{"id": row[0], "duration": row[1]} for row in result]}

# Endpoint to get end station names and start dates for trips ending at a specific location
@app.get("/v1/bike_share_1/trips_ending_at_location", operation_id="get_trips_ending_at_location", summary="Retrieves the end station names and start dates for trips that ended at a specific location, identified by its latitude and longitude.")
async def get_trips_ending_at_location(lat: float = Query(..., description="Latitude of the station"), long: float = Query(..., description="Longitude of the station")):
    cursor.execute("SELECT T2.end_station_name, T2.start_date FROM station AS T1 INNER JOIN trip AS T2 ON T2.end_station_name = T1.name WHERE T1.lat = ? AND T1.long = ?", (lat, long))
    result = cursor.fetchall()
    if not result:
        return {"trips": []}
    return {"trips": [{"end_station_name": row[0], "start_date": row[1]} for row in result]}

# Endpoint to get the count of trips starting in a specific month and year for a given city
@app.get("/v1/bike_share_1/count_trips_start_date_city", operation_id="get_count_trips_start_date_city", summary="Get the count of trips starting in a specific month and year for a given city")
async def get_count_trips_start_date_city(start_date_pattern: str = Query(..., description="Start date pattern in 'MM/%/YYYY%' format"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(T2.start_date) FROM station AS T1 INNER JOIN trip AS T2 ON T2.start_station_name = T1.name WHERE T2.start_date LIKE ? AND T1.city = ?", (start_date_pattern, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get trip details with mean temperature for trips under a specific duration
@app.get("/v1/bike_share_1/trip_details_mean_temperature", operation_id="get_trip_details_mean_temperature", summary="Get trip details with mean temperature for trips under a specific duration")
async def get_trip_details_mean_temperature(max_duration: int = Query(..., description="Maximum trip duration in seconds")):
    cursor.execute("SELECT T1.start_station_name, T1.end_station_name, T2.mean_temperature_f FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T1.duration < ?", (max_duration,))
    result = cursor.fetchall()
    if not result:
        return {"trips": []}
    return {"trips": result}

# Endpoint to get the date with the highest number of trips and its average minimum temperature
@app.get("/v1/bike_share_1/date_highest_trips_avg_min_temp", operation_id="get_date_highest_trips_avg_min_temp", summary="Retrieves the date with the highest recorded number of trips and its corresponding average minimum temperature. This operation combines data from the 'trip' and 'weather' tables, grouping by date and calculating the average minimum temperature for each date. The result is ordered by the count of trips in descending order, with the top record being returned.")
async def get_date_highest_trips_avg_min_temp():
    cursor.execute("SELECT T2.date, AVG(T2.min_temperature_f) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code GROUP BY T2.date ORDER BY COUNT(T1.start_date) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"date": [], "avg_min_temp": []}
    return {"date": result[0], "avg_min_temp": result[1]}

# Endpoint to get average trip duration and wind direction for specific months and year
@app.get("/v1/bike_share_1/avg_trip_duration_wind_direction", operation_id="get_avg_trip_duration_wind_direction", summary="Retrieves the average trip duration and wind direction for the specified months and year. The operation calculates these averages by joining the trip and weather tables on the zip code, and filtering the results based on the provided months and year. The output provides insights into the relationship between trip duration and wind direction for the given time period.")
async def get_avg_trip_duration_wind_direction(month1: str = Query(..., description="First month (e.g., '7')"), month2: str = Query(..., description="Second month (e.g., '8')"), month3: str = Query(..., description="Third month (e.g., '9')"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT AVG(T1.duration), AVG(T2.wind_dir_degrees) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE SUBSTR(CAST(T2.date AS TEXT), 1, INSTR(T2.date, '/') - 1) IN (?, ?, ?) AND SUBSTR(CAST(T2.date AS TEXT), -4) = ?", (month1, month2, month3, year))
    result = cursor.fetchone()
    if not result:
        return {"avg_duration": [], "avg_wind_dir": []}
    return {"avg_duration": result[0], "avg_wind_dir": result[1]}

# Endpoint to get the count of stations installed in a specific year and city, along with their names
@app.get("/v1/bike_share_1/count_stations_installed_year_city", operation_id="get_count_stations_installed_year_city", summary="Get the count of stations installed in a specific year and city, along with their names")
async def get_count_stations_installed_year_city(city: str = Query(..., description="City name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(CASE WHEN city = ? AND SUBSTR(installation_date, -4) = ? THEN 1 ELSE 0 END) FROM station UNION SELECT name FROM station WHERE city = ? AND SUBSTR(installation_date, -4) = ?", (city, year, city, year))
    result = cursor.fetchall()
    if not result:
        return {"count": [], "stations": []}
    return {"count": result[0][0], "stations": [row[0] for row in result[1:]]}

# Endpoint to get the maximum trip duration for a specific date
@app.get("/v1/bike_share_1/max_trip_duration_date", operation_id="get_max_trip_duration_date", summary="Retrieves the longest trip duration for a given date range. The date range is specified using start and end date patterns in 'MM/DD/YYYY%' format. The operation returns the maximum duration of any trip that started and ended within the specified date range.")
async def get_max_trip_duration_date(start_date_pattern: str = Query(..., description="Start date pattern in 'MM/DD/YYYY%' format"), end_date_pattern: str = Query(..., description="End date pattern in 'MM/DD/YYYY%' format")):
    cursor.execute("SELECT MAX(duration) FROM trip WHERE start_date LIKE ? AND end_date LIKE ?", (start_date_pattern, end_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"max_duration": []}
    return {"max_duration": result[0]}

# Endpoint to get the trip duration in minutes for a specific bike and date
@app.get("/v1/bike_share_1/trip_duration_minutes", operation_id="get_trip_duration_minutes", summary="Get the trip duration in minutes for a specific bike and date")
async def get_trip_duration_minutes(bike_id: int = Query(..., description="Bike ID"), end_station_name: str = Query(..., description="End station name"), start_station_name: str = Query(..., description="Start station name"), start_date_pattern: str = Query(..., description="Start date pattern in 'MM/DD/YYYY%' format"), end_date_pattern: str = Query(..., description="End date pattern in 'MM/DD/YYYY%' format")):
    cursor.execute("SELECT CAST(duration AS REAL) / 60 FROM trip WHERE bike_id = ? AND end_station_name = ? AND start_station_name = ? AND start_date LIKE ? AND end_date LIKE ?", (bike_id, end_station_name, start_station_name, start_date_pattern, end_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"duration_minutes": []}
    return {"duration_minutes": result[0]}

# Endpoint to get the bike ID for the longest trip between specific stations on a specific date
@app.get("/v1/bike_share_1/longest_trip_bike_id", operation_id="get_longest_trip_bike_id", summary="Get the bike ID for the longest trip between specific stations on a specific date")
async def get_longest_trip_bike_id(start_date_pattern: str = Query(..., description="Start date pattern in 'MM/DD/YYYY%' format"), end_date_pattern: str = Query(..., description="End date pattern in 'MM/DD/YYYY%' format"), end_station_name: str = Query(..., description="End station name"), start_station_name: str = Query(..., description="Start station name")):
    cursor.execute("SELECT bike_id FROM trip WHERE start_date LIKE ? AND end_date LIKE ? AND end_station_name = ? AND start_station_name = ? AND duration = ( SELECT MAX(duration) FROM trip WHERE start_date LIKE ? AND end_date LIKE ? AND end_station_name = ? AND start_station_name = ? )", (start_date_pattern, end_date_pattern, end_station_name, start_station_name, start_date_pattern, end_date_pattern, end_station_name, start_station_name))
    result = cursor.fetchone()
    if not result:
        return {"bike_id": []}
    return {"bike_id": result[0]}

# Endpoint to get the count of stations with more than a specific number of docks in a given city
@app.get("/v1/bike_share_1/count_stations_dock_count", operation_id="get_count_stations_dock_count", summary="Retrieve the total number of stations in a specified city that have more than a certain number of docks available. This operation allows you to assess the availability of bike docks in a given city based on a minimum dock count threshold.")
async def get_count_stations_dock_count(city: str = Query(..., description="City name"), min_dock_count: int = Query(..., description="Minimum dock count")):
    cursor.execute("SELECT SUM(CASE WHEN city = ? AND dock_count > ? THEN 1 ELSE 0 END) FROM station", (city, min_dock_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the date and maximum temperature recorded
@app.get("/v1/bike_share_1/max_temperature_date", operation_id="get_max_temperature_date", summary="Get the date and maximum temperature recorded")
async def get_max_temperature_date():
    cursor.execute("SELECT max_temperature_f, date FROM weather WHERE max_temperature_f = ( SELECT MAX(max_temperature_f) FROM weather WHERE max_temperature_f IS NOT NULL AND max_temperature_f IS NOT '' )")
    result = cursor.fetchone()
    if not result:
        return {"max_temperature": [], "date": []}
    return {"max_temperature": result[0], "date": result[1]}

# Endpoint to get the max dew point for a specific date and zip code
@app.get("/v1/bike_share_1/max_dew_point_by_date_zip", operation_id="get_max_dew_point", summary="Get the max dew point for a specific date and zip code")
async def get_max_dew_point(date: str = Query(..., description="Date in 'MM/DD/YYYY' format"), zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT DISTINCT CASE WHEN date = ? AND zip_code = ? THEN max_dew_point_f END FROM weather", (date, zip_code))
    result = cursor.fetchone()
    if not result:
        return {"max_dew_point": []}
    return {"max_dew_point": result[0]}

# Endpoint to get the year with the most rain events
@app.get("/v1/bike_share_1/year_with_most_rain_events", operation_id="get_year_with_most_rain_events", summary="Retrieves the year with the highest combined count of the specified rain events. The operation filters weather data based on two user-defined event types and returns the year with the most occurrences of these events.")
async def get_year_with_most_rain_events(event1: str = Query(..., description="First event type (e.g., '%Rain%')"), event2: str = Query(..., description="Second event type (e.g., '%rain%')")):
    cursor.execute("SELECT SUBSTR(CAST(date AS TEXT), -4) FROM weather GROUP BY SUBSTR(CAST(date AS TEXT), -4) ORDER BY SUM(CASE WHEN events LIKE ? OR events LIKE ? THEN 1 ELSE 0 END) DESC LIMIT 1", (event1, event2))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the minimum trip duration and bike ID for trips starting on a specific date and location
@app.get("/v1/bike_share_1/min_trip_duration_by_date_location", operation_id="get_min_trip_duration", summary="Retrieve the shortest trip duration and corresponding bike ID for trips that began on a specific date and location. The date is provided in 'MM/DD/YYYY%' format, and the location is defined by latitude and longitude coordinates.")
async def get_min_trip_duration(start_date: str = Query(..., description="Start date in 'MM/DD/YYYY%' format"), lat: float = Query(..., description="Latitude"), long: float = Query(..., description="Longitude")):
    cursor.execute("SELECT MIN(T2.duration), T2.bike_id FROM station AS T1 INNER JOIN trip AS T2 ON T2.start_station_name = T1.name WHERE T2.start_date LIKE ? AND T1.lat = ? AND T1.long = ?", (start_date, lat, long))
    result = cursor.fetchone()
    if not result:
        return {"min_duration": [], "bike_id": []}
    return {"min_duration": result[0], "bike_id": result[1]}

# Endpoint to get the minimum trip duration, end station name, and count of trips for a specific start date, start station, and subscription type
@app.get("/v1/bike_share_1/min_trip_duration_by_start_date_station_subscription", operation_id="get_min_trip_duration_by_start_date_station_subscription", summary="Retrieve the shortest trip duration, the name of the end station, and the total number of trips for a given start date, start station, and subscription type. This operation provides insights into the quickest trips, their destinations, and the frequency of such trips under specific conditions.")
async def get_min_trip_duration_by_start_date_station_subscription(start_date: str = Query(..., description="Start date in 'MM/DD/YYYY%' format"), start_station_name: str = Query(..., description="Start station name"), subscription_type: str = Query(..., description="Subscription type")):
    cursor.execute("SELECT MIN(T2.duration), T2.end_station_name, COUNT(T2.start_station_name) FROM station AS T1 INNER JOIN trip AS T2 ON T2.start_station_name = T1.name WHERE T2.start_date LIKE ? AND T2.start_station_name = ? AND T2.subscription_type = ?", (start_date, start_station_name, subscription_type))
    result = cursor.fetchone()
    if not result:
        return {"min_duration": [], "end_station_name": [], "count": []}
    return {"min_duration": result[0], "end_station_name": result[1], "count": result[2]}

# Endpoint to get the maximum humidity for a specific trip start date, bike ID, and start station name
@app.get("/v1/bike_share_1/max_humidity_by_trip_details", operation_id="get_max_humidity_by_trip_details", summary="Get the maximum humidity for a specific trip start date, bike ID, and start station name")
async def get_max_humidity_by_trip_details(start_date: str = Query(..., description="Start date in 'MM/DD/YYYY%' format"), bike_id: int = Query(..., description="Bike ID"), start_station_name: str = Query(..., description="Start station name")):
    cursor.execute("SELECT T2.max_humidity FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T1.start_date LIKE ? AND T1.bike_id = ? AND T1.start_station_name = ?", (start_date, bike_id, start_station_name))
    result = cursor.fetchone()
    if not result:
        return {"max_humidity": []}
    return {"max_humidity": result[0]}

# Endpoint to get the date and count of trips for a specific date pattern, zip code, event, and subscription type
@app.get("/v1/bike_share_1/trip_count_by_date_zip_event_subscription", operation_id="get_trip_count_by_date_zip_event_subscription", summary="Retrieve the count of bike trips that occurred on specific dates, in a given zip code, for a certain event, and with a particular subscription type. The response includes the date and the total number of trips.")
async def get_trip_count_by_date_zip_event_subscription(date_pattern: str = Query(..., description="Date pattern in 'MM/%/YYYY%' format"), zip_code: int = Query(..., description="Zip code"), events: str = Query(..., description="Event type"), subscription_type: str = Query(..., description="Subscription type")):
    cursor.execute("SELECT T2.date, COUNT(T1.start_station_name) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.date LIKE ? AND T2.zip_code = ? AND T2.events = ? AND T1.subscription_type = ?", (date_pattern, zip_code, events, subscription_type))
    result = cursor.fetchone()
    if not result:
        return {"date": [], "count": []}
    return {"date": result[0], "count": result[1]}

# Endpoint to get the start station name and installation date for the most popular start station for a specific subscription type
@app.get("/v1/bike_share_1/most_popular_start_station_by_subscription", operation_id="get_most_popular_start_station_by_subscription", summary="Retrieves the name and installation date of the most frequently used start station for a given subscription type. The data is filtered by the specified subscription type and ordered by the frequency of usage, with the most popular station returned.")
async def get_most_popular_start_station_by_subscription(subscription_type: str = Query(..., description="Subscription type")):
    cursor.execute("SELECT T1.start_station_name, T2.installation_date FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.subscription_type = ? GROUP BY T1.start_station_name ORDER BY COUNT(T1.subscription_type) LIMIT 1", (subscription_type,))
    result = cursor.fetchone()
    if not result:
        return {"start_station_name": [], "installation_date": []}
    return {"start_station_name": result[0], "installation_date": result[1]}

# Endpoint to get distinct station names with no bikes available at a specific time
@app.get("/v1/bike_share_1/stations_with_no_bikes_available", operation_id="get_stations_with_no_bikes_available", summary="Retrieve a unique list of station names that have no bikes available at a specific time. The operation filters the station data based on the provided number of bikes available and a time pattern in 'YYYY/MM/DD%' format.")
async def get_stations_with_no_bikes_available(bikes_available: int = Query(..., description="Number of bikes available"), time_pattern: str = Query(..., description="Time pattern in 'YYYY/MM/DD%' format")):
    cursor.execute("SELECT DISTINCT T1.name FROM station AS T1 INNER JOIN status AS T2 ON T2.station_id = T1.id WHERE T2.bikes_available = ? AND T2.time LIKE ?", (bikes_available, time_pattern))
    result = cursor.fetchall()
    if not result:
        return {"station_names": []}
    return {"station_names": [row[0] for row in result]}

# Endpoint to get the average trip duration for trips starting in a specific city
@app.get("/v1/bike_share_1/avg_trip_duration_by_city", operation_id="get_avg_trip_duration_by_city", summary="Retrieves the average duration of trips that started in the specified city. The calculation is based on the duration of each trip, which is determined by the difference between the start and end times of the trip. The city is identified by its name.")
async def get_avg_trip_duration_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT AVG(T1.duration) FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"avg_duration": []}
    return {"avg_duration": result[0]}

# Endpoint to get station details based on installation date criteria
@app.get("/v1/bike_share_1/station_details_by_installation_date", operation_id="get_station_details", summary="Retrieves station details for those installed in a specific year and month(s). The operation filters stations based on the provided month, day, and year, or a list of months and a year. The result includes the station name, installation date, and city.")
async def get_station_details(month: str = Query(..., description="Month part of the installation date"), day: str = Query(..., description="Day part of the installation date"), year: str = Query(..., description="Year part of the installation date"), months_list: str = Query(..., description="Comma-separated list of months"), year_filter: str = Query(..., description="Year to filter by")):
    months_list = months_list.split(',')
    cursor.execute("SELECT name, installation_date, city FROM station WHERE (SUBSTR(CAST(installation_date AS TEXT), 1, INSTR(installation_date, '/') - 1) = ? AND SUBSTR(CAST(installation_date AS TEXT), INSTR(installation_date, '/') + 1, -6) >= ? AND SUBSTR(CAST(installation_date AS TEXT), -4) = ?) OR (SUBSTR(CAST(installation_date AS TEXT), 1, INSTR(installation_date, '/') - 1) IN (?, ?, ?, ?, ?, ?, ?) AND SUBSTR(CAST(installation_date AS TEXT), -4) = ?)", (month, day, year, *months_list, year_filter))
    result = cursor.fetchall()
    if not result:
        return {"stations": []}
    return {"stations": result}

# Endpoint to get the average trip duration between two stations
@app.get("/v1/bike_share_1/average_trip_duration", operation_id="get_average_trip_duration", summary="Retrieves the average duration of trips that start at the specified start station and end at the specified end station.")
async def get_average_trip_duration(start_station_name: str = Query(..., description="Start station name"), end_station_name: str = Query(..., description="End station name")):
    cursor.execute("SELECT AVG(duration) FROM trip WHERE start_station_name = ? AND end_station_name = ?", (start_station_name, end_station_name))
    result = cursor.fetchone()
    if not result:
        return {"average_duration": []}
    return {"average_duration": result[0]}

# Endpoint to get station status details where no bikes are available
@app.get("/v1/bike_share_1/station_status_no_bikes", operation_id="get_station_status_no_bikes", summary="Retrieves the status of bike stations where no bikes are currently available. The response includes the time of the status, the name of the station, and its geographical coordinates (latitude and longitude).")
async def get_station_status_no_bikes(bikes_available: int = Query(..., description="Number of bikes available")):
    cursor.execute("SELECT T2.time, T1.name, T1.lat, T1.long FROM station AS T1 INNER JOIN status AS T2 ON T2.station_id = T1.id WHERE T2.bikes_available = ?", (bikes_available,))
    result = cursor.fetchall()
    if not result:
        return {"station_status": []}
    return {"station_status": result}

# Endpoint to get trip details where start and end stations are the same
@app.get("/v1/bike_share_1/trip_details_same_start_end", operation_id="get_trip_details_same_start_end", summary="Retrieves trip details for journeys that start and end at the same station. The response includes the trip ID and the latitude and longitude of the station.")
async def get_trip_details_same_start_end():
    cursor.execute("SELECT T1.id, T2.lat, T2.long FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.start_station_name = T1.end_station_name")
    result = cursor.fetchall()
    if not result:
        return {"trip_details": []}
    return {"trip_details": result}

# Endpoint to get the sum of available docks at a specific station and time
@app.get("/v1/bike_share_1/sum_available_docks", operation_id="get_sum_available_docks", summary="Retrieves the total number of available docks at a specific station at a given time. The calculation is based on the total dock count at the station and the number of bikes currently available at that time.")
async def get_sum_available_docks(station_name: str = Query(..., description="Name of the station"), time: str = Query(..., description="Time in 'YYYY/MM/DD HH:MM:SS' format")):
    cursor.execute("SELECT SUM(T1.dock_count - T2.bikes_available) FROM station AS T1 INNER JOIN status AS T2 ON T1.id = T2.station_id WHERE T1.name = ? AND T2.time = ?", (station_name, time))
    result = cursor.fetchone()
    if not result:
        return {"sum_available_docks": []}
    return {"sum_available_docks": result[0]}

# Endpoint to get trip IDs based on bike ID, mean temperature, and subscription type
@app.get("/v1/bike_share_1/trip_ids_by_bike_temp_subscription", operation_id="get_trip_ids_by_bike_temp_subscription", summary="Retrieves the IDs of trips that meet the specified criteria: bike ID, minimum mean temperature, and subscription type. The trips are filtered based on the weather conditions and subscription type associated with the bike ID.")
async def get_trip_ids_by_bike_temp_subscription(bike_id: int = Query(..., description="Bike ID"), mean_temperature_f: int = Query(..., description="Mean temperature in Fahrenheit"), subscription_type: str = Query(..., description="Subscription type")):
    cursor.execute("SELECT T1.id FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T1.bike_id = ? AND T2.mean_temperature_f > ? AND T1.subscription_type = ?", (bike_id, mean_temperature_f, subscription_type))
    result = cursor.fetchall()
    if not result:
        return {"trip_ids": []}
    return {"trip_ids": result}

# Endpoint to get weather details for a specific trip based on various criteria
@app.get("/v1/bike_share_1/weather_details_by_trip_criteria", operation_id="get_weather_details_by_trip_criteria", summary="Retrieve weather details for a specific bike trip based on provided criteria. The criteria include bike ID, minimum mean temperature, subscription type, start and end station names, and trip duration. The response includes maximum gust speed and cloud cover data.")
async def get_weather_details_by_trip_criteria(bike_id: int = Query(..., description="Bike ID"), mean_temperature_f: int = Query(..., description="Mean temperature in Fahrenheit"), subscription_type: str = Query(..., description="Subscription type"), start_station_name: str = Query(..., description="Start station name"), end_station_name: str = Query(..., description="End station name"), duration: int = Query(..., description="Trip duration")):
    cursor.execute("SELECT T2.max_gust_speed_mph, T2.cloud_cover FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code AND T2.date = SUBSTR(CAST(T1.start_date AS TEXT), 1, INSTR(T1.start_date, ' ') - 1) WHERE T1.bike_id = ? AND T2.mean_temperature_f > ? AND T1.subscription_type = ? AND T1.start_station_name = ? AND T1.end_station_name = ? AND T1.duration = ?", (bike_id, mean_temperature_f, subscription_type, start_station_name, end_station_name, duration))
    result = cursor.fetchall()
    if not result:
        return {"weather_details": []}
    return {"weather_details": result}

# Endpoint to get the count of trips based on various criteria
@app.get("/v1/bike_share_1/count_trips_by_criteria", operation_id="get_count_trips_by_criteria", summary="Retrieves the total number of trips that meet the specified criteria, including subscription type, minimum visibility, maximum duration, and start and end station names.")
async def get_count_trips_by_criteria(subscription_type: str = Query(..., description="Subscription type"), min_visibility_miles: int = Query(..., description="Minimum visibility in miles"), duration: int = Query(..., description="Maximum trip duration"), start_station_name: str = Query(..., description="Start station name"), end_station_name: str = Query(..., description="End station name")):
    cursor.execute("SELECT COUNT(T1.id) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T1.subscription_type = ? AND T2.min_visibility_miles = ? AND T1.duration < ? AND T1.start_station_name = ? AND T1.end_station_name = ?", (subscription_type, min_visibility_miles, duration, start_station_name, end_station_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of available docks for trips starting from a specific zip code
@app.get("/v1/bike_share_1/sum_available_docks_by_zip", operation_id="get_sum_available_docks_by_zip", summary="Retrieves the total number of available docks for trips originating from a specified zip code. The calculation is based on the sum of available docks at each station within the given zip code.")
async def get_sum_available_docks_by_zip(zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT SUM(T2.docks_available) FROM trip AS T1 INNER JOIN status AS T2 ON T2.station_id = T1.start_station_id WHERE T1.zip_code = ?", (zip_code,))
    result = cursor.fetchone()
    if not result:
        return {"sum_available_docks": []}
    return {"sum_available_docks": result[0]}

# Endpoint to get trip IDs based on minimum temperature
@app.get("/v1/bike_share_1/trip_ids_by_min_temperature", operation_id="get_trip_ids_by_min_temperature", summary="Retrieves the IDs of all trips that occurred in locations with a minimum temperature below the specified value. The temperature is provided in Fahrenheit.")
async def get_trip_ids_by_min_temperature(min_temperature_f: int = Query(..., description="Minimum temperature in Fahrenheit")):
    cursor.execute("SELECT T1.id FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.min_temperature_f < ?", (min_temperature_f,))
    result = cursor.fetchall()
    if not result:
        return {"trip_ids": []}
    return {"trip_ids": result}

# Endpoint to get the minimum trip duration, difference between minimum and average duration, and minimum temperature for a specific date range, start and end station, and subscription type
@app.get("/v1/bike_share_1/trip_duration_temperature", operation_id="get_trip_duration_temperature", summary="Get the minimum trip duration, difference between minimum and average duration, and minimum temperature for a specific date range, start and end station, and subscription type")
async def get_trip_duration_temperature(start_date: str = Query(..., description="Start date in 'MM/DD/YYYY HH:MM' format"), end_date: str = Query(..., description="End date in 'MM/DD/YYYY HH:MM' format"), start_station_name: str = Query(..., description="Start station name"), end_station_name: str = Query(..., description="End station name"), subscription_type: str = Query(..., description="Subscription type")):
    cursor.execute("SELECT MIN(T1.duration) , MIN(T1.duration) - AVG(T1.duration), T2.min_temperature_f FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T1.start_date = ? AND T1.end_date = ? AND T1.start_station_name = ? AND T1.end_station_name = ? AND T1.subscription_type = ?", (start_date, end_date, start_station_name, end_station_name, subscription_type))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the average maximum and minimum temperature for a specific month and humidity range
@app.get("/v1/bike_share_1/average_temperature_humidity", operation_id="get_average_temperature_humidity", summary="Retrieves the average maximum and minimum temperature for a specific month, filtered by a given humidity range. The month is specified using a date in 'M/%/YYYY' format, and the humidity range is defined by minimum and maximum values.")
async def get_average_temperature_humidity(date: str = Query(..., description="Date in 'M/%/YYYY' format"), min_humidity: int = Query(..., description="Minimum humidity"), max_humidity: int = Query(..., description="Maximum humidity")):
    cursor.execute("SELECT AVG(max_temperature_f), AVG(min_temperature_f) FROM weather WHERE date LIKE ? AND mean_humidity BETWEEN ? AND ?", (date, min_humidity, max_humidity))
    result = cursor.fetchone()
    if not result:
        return {"temperatures": []}
    return {"temperatures": result}

# Endpoint to get the difference in the number of subscribers and customers for a specific month
@app.get("/v1/bike_share_1/subscription_difference", operation_id="get_subscription_difference", summary="Retrieve the net difference between the total number of subscribers and customers for a given month. The operation considers the specified subscription types for both subscribers and customers. The start date, provided in 'M/%/YYYY%' format, determines the month for which the difference is calculated.")
async def get_subscription_difference(subscriber_type: str = Query(..., description="Subscription type for subscribers"), customer_type: str = Query(..., description="Subscription type for customers"), start_date: str = Query(..., description="Start date in 'M/%/YYYY%' format")):
    cursor.execute("SELECT SUM(IIF(subscription_type = ?, 1, 0)) - SUM(IIF(subscription_type = ?, 1, 0)) FROM trip WHERE start_date LIKE ?", (subscriber_type, customer_type, start_date))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the date and bike ID for trips in a specific year and weather event
@app.get("/v1/bike_share_1/trip_date_bike_id", operation_id="get_trip_date_bike_id", summary="Retrieve the date and bike ID for trips that occurred in a specific year and during certain weather events. The year is provided in 'YYYY' format, and the weather events are specified as input parameters.")
async def get_trip_date_bike_id(year: str = Query(..., description="Year in 'YYYY' format"), events: str = Query(..., description="Weather events")):
    cursor.execute("SELECT T2.date, T1.bike_id FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE SUBSTR(CAST(T2.date AS TEXT), -4) = ? AND T2.events = ?", (year, events))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get weather and trip details for the longest trip during a specific weather event
@app.get("/v1/bike_share_1/weather_trip_details", operation_id="get_weather_trip_details", summary="Retrieves the weather and trip details for the longest trip that occurred during a specific weather event. The weather details include mean visibility in miles, mean wind speed in mph, and the weather event itself. The trip details include the start and end station names, as well as the latitude and longitude of the start station.")
async def get_weather_trip_details(events: str = Query(..., description="Weather events")):
    cursor.execute("SELECT T3.mean_visibility_miles, T3.mean_wind_speed_mph, T3.events, T1.lat, T1.long , T2.start_station_name, T2.end_station_name FROM station AS T1 INNER JOIN trip AS T2 ON T2.start_station_name = T1.name INNER JOIN weather AS T3 ON T3.zip_code = T2.zip_code WHERE T3.events = ? ORDER BY T2.duration DESC LIMIT 1", (events,))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the start date of trips with the minimum temperature for a specific month, start and end station
@app.get("/v1/bike_share_1/trip_start_date_min_temperature", operation_id="get_trip_start_date_min_temperature", summary="Retrieve the earliest date of trips that started and ended at specific stations during a given month, with the lowest recorded temperature for that month.")
async def get_trip_start_date_min_temperature(date: str = Query(..., description="Date in 'M/%/YYYY' format"), start_station_name: str = Query(..., description="Start station name"), end_station_name: str = Query(..., description="End station name")):
    cursor.execute("SELECT T1.start_date FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code AND T2.date = SUBSTR(CAST(T1.start_date AS TEXT), 1, INSTR(T1.start_date, ' ') - 1) WHERE T2.date LIKE ? AND T1.start_station_name = ? AND T1.end_station_name = ? AND T2.min_temperature_f = ( SELECT MIN(T2.min_temperature_f) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code AND T2.date = SUBSTR(CAST(T1.start_date AS TEXT), 1, INSTR(T1.start_date, ' ') - 1) WHERE T2.date LIKE ? AND T1.start_station_name = ? AND T1.end_station_name = ? )", (date, start_station_name, end_station_name, date, start_station_name, end_station_name))
    result = cursor.fetchall()
    if not result:
        return {"start_dates": []}
    return {"start_dates": result}

# Endpoint to get the longest trip details during specific weather events
@app.get("/v1/bike_share_1/longest_trip_weather_events", operation_id="get_longest_trip_weather_events", summary="Retrieve the details of the longest bike trip that occurred during either of the specified weather events. The response includes the start and end station names, as well as the duration of the trip. The weather events are determined by the zip code of the trip's start station.")
async def get_longest_trip_weather_events(event1: str = Query(..., description="Weather event 1"), event2: str = Query(..., description="Weather event 2")):
    cursor.execute("SELECT T1.start_station_name, T1.end_station_name, T1.duration FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.events = ? OR T2.events = ? ORDER BY T1.duration DESC LIMIT 1", (event1, event2))
    result = cursor.fetchone()
    if not result:
        return {"trip_details": []}
    return {"trip_details": result}

# Endpoint to get the average trip duration during specific weather events and precipitation levels
@app.get("/v1/bike_share_1/average_trip_duration_weather_precipitation", operation_id="get_average_trip_duration_weather_precipitation", summary="Retrieve the average duration of trips that occurred during two specific weather events and precipitation levels. The operation calculates the average trip duration for each event and precipitation level combination, providing insights into how weather conditions impact trip durations.")
async def get_average_trip_duration_weather_precipitation(event1: str = Query(..., description="Weather event 1"), precipitation1: float = Query(..., description="Precipitation level for event 1"), event2: str = Query(..., description="Weather event 2"), precipitation2: float = Query(..., description="Precipitation level for event 2")):
    cursor.execute("SELECT AVG(T1.duration) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE (T2.events = ? AND T2.precipitation_inches > ?) OR (T2.events = ? AND T2.precipitation_inches > ?)", (event1, precipitation1, event2, precipitation2))
    result = cursor.fetchone()
    if not result:
        return {"average_duration": []}
    return {"average_duration": result[0]}

# Endpoint to get distinct start station names and cities for trips longer than the average trip duration
@app.get("/v1/bike_share_1/long_trip_stations_cities", operation_id="get_long_trip_stations_cities", summary="Retrieves unique combinations of start station names and their respective cities for trips that exceed the average trip duration. This operation provides insights into the most popular starting points for longer trips.")
async def get_long_trip_stations_cities():
    cursor.execute("SELECT DISTINCT T1.start_station_name, T2.city FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.duration > ( SELECT AVG(T1.duration) FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name )")
    result = cursor.fetchall()
    if not result:
        return {"stations_cities": []}
    return {"stations_cities": result}

# Endpoint to get the sum of stations installed in a specific city and year
@app.get("/v1/bike_share_1/sum_stations_city_year", operation_id="get_sum_stations_city_year", summary="Retrieves the total number of bike share stations installed in a specified city during a given year. The city and year are provided as input parameters.")
async def get_sum_stations_city_year(city: str = Query(..., description="City name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(CASE WHEN city = ? AND SUBSTR(installation_date, -4) = ? THEN 1 ELSE 0 END) FROM station", (city, year))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the count of trips ending in a specific city and year
@app.get("/v1/bike_share_1/count_trips_end_city_year", operation_id="get_count_trips_end_city_year", summary="Retrieves the total number of bike trips that ended in a specified city during a given year. The city is identified by its name, and the year is determined by the start date of the trips. This operation provides a quantitative measure of bike usage in the specified city for the selected year.")
async def get_count_trips_end_city_year(city: str = Query(..., description="City name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.city) FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.end_station_name WHERE T2.city = ? AND T1.start_date LIKE ?", (city, '%' + year + '%'))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the trip ID with the maximum duration starting from a specific city
@app.get("/v1/bike_share_1/max_duration_trip_id_city", operation_id="get_max_duration_trip_id_city", summary="Retrieves the unique identifier of the trip with the longest duration that originated from the specified city.")
async def get_max_duration_trip_id_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.id FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.city = ? AND T1.duration = ( SELECT MAX(T1.duration) FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.city = ? )", (city, city))
    result = cursor.fetchone()
    if not result:
        return {"trip_id": []}
    return {"trip_id": result[0]}

# Endpoint to get distinct bike IDs from trips starting at stations installed in a specific year
@app.get("/v1/bike_share_1/distinct_bike_ids_installation_year", operation_id="get_distinct_bike_ids_installation_year", summary="Retrieve a unique list of bike IDs for trips that began at stations installed in a specified year. The year should be provided in the 'YYYY' format.")
async def get_distinct_bike_ids_installation_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T1.bike_id FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.installation_date LIKE ?", ('%' + year,))
    result = cursor.fetchall()
    if not result:
        return {"bike_ids": []}
    return {"bike_ids": [row[0] for row in result]}

# Endpoint to get the count of trips ending in a specific city with a specific subscription type
@app.get("/v1/bike_share_1/count_trips_subscription_type_city", operation_id="get_count_trips_subscription_type_city", summary="Retrieves the total number of trips that ended in a specified city and were made using a particular subscription type. The subscription type and city are provided as input parameters.")
async def get_count_trips_subscription_type_city(subscription_type: str = Query(..., description="Subscription type"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(T1.subscription_type) FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.end_station_name WHERE T1.subscription_type = ? AND T2.city = ?", (subscription_type, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the trip ID with the minimum duration starting from stations with a specific dock count
@app.get("/v1/bike_share_1/min_duration_trip_id_dock_count", operation_id="get_min_duration_trip_id_dock_count", summary="Get the trip ID with the minimum duration starting from stations with a specific dock count")
async def get_min_duration_trip_id_dock_count(dock_count: int = Query(..., description="Dock count")):
    cursor.execute("SELECT T1.id FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.dock_count = ? AND T1.duration = ( SELECT MIN(T1.duration) FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.dock_count = ? )", (dock_count, dock_count))
    result = cursor.fetchone()
    if not result:
        return {"trip_id": []}
    return {"trip_id": result[0]}

# Endpoint to get the most popular start station in a specific city based on trip count
@app.get("/v1/bike_share_1/most_popular_start_station_city", operation_id="get_most_popular_start_station_city", summary="Retrieves the most frequently used start station in a given city, based on the total number of trips originating from that station. The city is specified as an input parameter.")
async def get_most_popular_start_station_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT SUBSTR(CAST(T1.start_date AS TEXT), INSTR(T1.start_date, ' '), -4) FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.city = ? GROUP BY T1.start_station_name ORDER BY COUNT(T1.id) DESC LIMIT 1", (city,))
    result = cursor.fetchone()
    if not result:
        return {"start_station": []}
    return {"start_station": result[0]}

# Endpoint to get the percentage of trips longer than a specific duration in a specific city
@app.get("/v1/bike_share_1/percentage_long_trips_city", operation_id="get_percentage_long_trips_city", summary="Retrieves the percentage of trips in a specific city that exceed a given duration. The duration is provided in seconds, and the city is identified by its name. The calculation is based on the total number of trips in the specified city.")
async def get_percentage_long_trips_city(duration: int = Query(..., description="Trip duration in seconds"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.duration > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.city = ?", (duration, city))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of trips longer than a specific duration in a specific year
@app.get("/v1/bike_share_1/count_long_trips_year", operation_id="get_count_long_trips_year", summary="Retrieves the total number of bike trips that exceed a specified duration in a given year. The year is provided in 'YYYY' format, and the duration is expressed in seconds.")
async def get_count_long_trips_year(year: str = Query(..., description="Year in 'YYYY' format"), duration: int = Query(..., description="Trip duration in seconds")):
    cursor.execute("SELECT COUNT(duration) FROM trip WHERE start_date LIKE ? AND duration > ?", ('%/%/' + year + '%', duration))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average trip duration starting from a specific station in a specific year
@app.get("/v1/bike_share_1/avg_trip_duration_station_year", operation_id="get_avg_trip_duration_station_year", summary="Retrieves the average duration of trips that started from a specified station in a given year. The year should be provided in 'YYYY' format, and the start station name must be accurate to ensure correct data retrieval.")
async def get_avg_trip_duration_station_year(year: str = Query(..., description="Year in 'YYYY' format"), start_station_name: str = Query(..., description="Start station name")):
    cursor.execute("SELECT AVG(duration) FROM trip WHERE start_date LIKE ? AND start_station_name = ?", ('%' + year + '%', start_station_name))
    result = cursor.fetchone()
    if not result:
        return {"average_duration": []}
    return {"average_duration": result[0]}

# Endpoint to get the sum of trips where start and end stations are the same
@app.get("/v1/bike_share_1/sum_round_trips", operation_id="get_sum_round_trips", summary="Retrieves the total count of round trips, where the start and end stations are identical, from the trip records.")
async def get_sum_round_trips():
    cursor.execute("SELECT SUM(IIF(start_station_id = end_station_id, 1, 0)) FROM trip")
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the trip ID with the highest maximum temperature
@app.get("/v1/bike_share_1/trip_with_highest_max_temperature", operation_id="get_trip_with_highest_max_temperature", summary="Retrieves the ID of the trip that occurred in the location with the highest recorded maximum temperature. The operation compares the maximum temperature data from the weather table with the zip code of each trip in the trip table. The trip ID with the highest maximum temperature is then returned.")
async def get_trip_with_highest_max_temperature():
    cursor.execute("SELECT T1.id FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code ORDER BY T2.max_temperature_f DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"trip_id": []}
    return {"trip_id": result[0]}

# Endpoint to get trip IDs based on weather event and start station name
@app.get("/v1/bike_share_1/trip_ids_by_weather_event_and_start_station", operation_id="get_trip_ids_by_weather_event_and_start_station", summary="Get trip IDs based on a specific weather event and start station name")
async def get_trip_ids_by_weather_event_and_start_station(weather_event: str = Query(..., description="Weather event (e.g., 'Rain')"), start_station_name: str = Query(..., description="Start station name")):
    cursor.execute("SELECT T1.id FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.events = ? AND T1.start_station_name = ?", (weather_event, start_station_name))
    result = cursor.fetchall()
    if not result:
        return {"trip_ids": []}
    return {"trip_ids": [row[0] for row in result]}

# Endpoint to get the average trip duration based on weather event
@app.get("/v1/bike_share_1/average_trip_duration_by_weather_event", operation_id="get_average_trip_duration_by_weather_event", summary="Retrieves the average duration of bike trips that occurred during a specified weather event. The weather event is identified by the input parameter, which can be any valid weather condition such as 'Fog'.")
async def get_average_trip_duration_by_weather_event(weather_event: str = Query(..., description="Weather event (e.g., 'Fog')")):
    cursor.execute("SELECT AVG(T1.duration) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.events = ?", (weather_event,))
    result = cursor.fetchone()
    if not result:
        return {"average_duration": []}
    return {"average_duration": result[0]}

# Endpoint to get the longest trip duration based on maximum wind speed
@app.get("/v1/bike_share_1/longest_trip_duration_by_max_wind_speed", operation_id="get_longest_trip_duration_by_max_wind_speed", summary="Retrieves the duration of the longest trip that occurred during a specific maximum wind speed. The operation filters trips based on the provided maximum wind speed and returns the duration of the longest trip that matches the filter criteria.")
async def get_longest_trip_duration_by_max_wind_speed(max_wind_speed: int = Query(..., description="Maximum wind speed in mph")):
    cursor.execute("SELECT T1.duration FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.max_wind_Speed_mph = ? ORDER BY T1.duration DESC LIMIT 1", (max_wind_speed,))
    result = cursor.fetchone()
    if not result:
        return {"longest_duration": []}
    return {"longest_duration": result[0]}

# Endpoint to get the average mean temperature based on year and start station name
@app.get("/v1/bike_share_1/average_mean_temperature_by_year_and_start_station", operation_id="get_average_mean_temperature_by_year_and_start_station", summary="Retrieves the average mean temperature for a given year and start station. The operation calculates the average temperature based on the weather data associated with the start station's zip code. The year and start station name are required input parameters.")
async def get_average_mean_temperature_by_year_and_start_station(year: str = Query(..., description="Year in 'YYYY' format"), start_station_name: str = Query(..., description="Start station name")):
    cursor.execute("SELECT AVG(T2.mean_temperature_f) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE SUBSTR(CAST(T2.date AS TEXT), -4) = ? AND T1.start_station_name = ?", (year, start_station_name))
    result = cursor.fetchone()
    if not result:
        return {"average_mean_temperature": []}
    return {"average_mean_temperature": result[0]}

# Endpoint to get the mean humidity based on trip ID
@app.get("/v1/bike_share_1/mean_humidity_by_trip_id", operation_id="get_mean_humidity_by_trip_id", summary="Get the mean humidity based on a specific trip ID")
async def get_mean_humidity_by_trip_id(trip_id: int = Query(..., description="Trip ID")):
    cursor.execute("SELECT T2.mean_humidity FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T1.id = ?", (trip_id,))
    result = cursor.fetchone()
    if not result:
        return {"mean_humidity": []}
    return {"mean_humidity": result[0]}

# Endpoint to get the percentage of trips with a specific weather event based on year and subscription type
@app.get("/v1/bike_share_1/percentage_trips_by_weather_event_year_subscription", operation_id="get_percentage_trips_by_weather_event_year_subscription", summary="Retrieves the percentage of trips that occurred under a specific weather condition in a given year, filtered by subscription type. The weather condition and year are provided as input parameters, along with the subscription type. The result is calculated by summing the trips that match the specified weather condition and dividing by the total number of trips for the given year and subscription type.")
async def get_percentage_trips_by_weather_event_year_subscription(weather_event: str = Query(..., description="Weather event (e.g., 'Rain')"), year: str = Query(..., description="Year in 'YYYY' format"), subscription_type: str = Query(..., description="Subscription type (e.g., 'Customer')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.events = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE SUBSTR(CAST(T2.date AS TEXT), -4) = ? AND T1.subscription_type = ?", (weather_event, year, subscription_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of station installations based on city and installation date criteria
@app.get("/v1/bike_share_1/count_station_installations_by_city_and_date", operation_id="get_count_station_installations_by_city_and_date", summary="Get the count of station installations based on a specific city and installation date criteria")
async def get_count_station_installations_by_city_and_date(city: str = Query(..., description="City name"), month1: str = Query(..., description="Month in 'MM' format"), month2: str = Query(..., description="Month in 'MM' format"), month3: str = Query(..., description="Month in 'MM' format"), month4: str = Query(..., description="Month in 'MM' format"), month5: str = Query(..., description="Month in 'MM' format"), year1: str = Query(..., description="Year in 'YYYY' format"), year2: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(installation_date) FROM station WHERE city = ? AND (SUBSTR(CAST(installation_date AS TEXT), 1, INSTR(installation_date, '/') - 1) IN (?, ?, ?, ?, ?) AND SUBSTR(CAST(installation_date AS TEXT), -4) = ?) OR SUBSTR(CAST(installation_date AS TEXT), -4) > ?", (city, month1, month2, month3, month4, month5, year1, year2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of mean temperatures based on zip code and date
@app.get("/v1/bike_share_1/sum_mean_temperatures_by_zip_code_and_date", operation_id="get_sum_mean_temperatures_by_zip_code_and_date", summary="Retrieves the total sum of mean temperatures for a given zip code and date. The input parameters specify the zip code and date in 'MM/DD/YYYY' format, which are used to filter the data and calculate the sum of mean temperatures.")
async def get_sum_mean_temperatures_by_zip_code_and_date(zip_code: int = Query(..., description="Zip code"), date: str = Query(..., description="Date in 'MM/DD/YYYY' format")):
    cursor.execute("SELECT SUM(IIF(zip_code = ? AND date = ?, mean_temperature_f, 0)) FROM weather", (zip_code, date))
    result = cursor.fetchone()
    if not result:
        return {"sum_mean_temperature": []}
    return {"sum_mean_temperature": result[0]}

# Endpoint to get the sum of temperature differences for a specific zip code and date
@app.get("/v1/bike_share_1/sum_temperature_difference", operation_id="get_sum_temperature_difference", summary="Retrieves the total temperature variation for a given zip code and date. The temperature variation is calculated as the difference between the maximum and minimum temperatures recorded on the specified date for the provided zip code.")
async def get_sum_temperature_difference(zip_code: int = Query(..., description="Zip code"), date: str = Query(..., description="Date in 'MM/DD/YYYY' format")):
    cursor.execute("SELECT SUM(IIF(zip_code = ? AND date = ?, max_temperature_f - min_temperature_f, 0)) FROM weather", (zip_code, date))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the number of bikes available at a specific station and time
@app.get("/v1/bike_share_1/bikes_available_at_station", operation_id="get_bikes_available", summary="Retrieves the current number of bikes available at a specific station at a given time. The station is identified by its name, and the time is provided in 'YYYY/MM/DD HH:MM:SS' format. This operation returns a single value representing the number of bikes available at the specified station and time.")
async def get_bikes_available(station_name: str = Query(..., description="Name of the station"), time: str = Query(..., description="Time in 'YYYY/MM/DD HH:MM:SS' format")):
    cursor.execute("SELECT T2.bikes_available FROM station AS T1 INNER JOIN status AS T2 ON T1.id = T2.station_id WHERE T1.name = ? AND T2.time = ?", (station_name, time))
    result = cursor.fetchone()
    if not result:
        return {"bikes_available": []}
    return {"bikes_available": result[0]}

# Endpoint to get the city of a trip's start station by trip ID
@app.get("/v1/bike_share_1/trip_start_station_city", operation_id="get_trip_start_station_city", summary="Retrieves the city where a specific trip started. The trip is identified by its unique ID. The city is determined based on the start station of the trip.")
async def get_trip_start_station_city(trip_id: int = Query(..., description="Trip ID")):
    cursor.execute("SELECT T2.city FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.id = ?", (trip_id,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the count of trips in a specific month, year, zip code, and temperature range
@app.get("/v1/bike_share_1/count_trips_by_date_zip_temp", operation_id="get_count_trips_by_date_zip_temp", summary="Retrieves the total number of trips that occurred within a specific month, year, zip code, and temperature range. The date pattern should be provided in 'M/%/YYYY' format, and the minimum temperature in Fahrenheit. The zip code is also required to filter the results.")
async def get_count_trips_by_date_zip_temp(date_pattern: str = Query(..., description="Date pattern in 'M/%/YYYY' format"), zip_code: int = Query(..., description="Zip code"), min_temp: int = Query(..., description="Minimum temperature in Fahrenheit")):
    cursor.execute("SELECT COUNT(T1.id) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.date LIKE ? AND T2.zip_code = ? AND T2.max_temperature_f > ?", (date_pattern, zip_code, min_temp))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct start station names for trips in a specific year, zip code, and temperature range
@app.get("/v1/bike_share_1/distinct_start_stations_by_year_zip_temp", operation_id="get_distinct_start_stations_by_year_zip_temp", summary="Retrieve a list of unique start station names for bike trips that occurred in a specific year, zip code, and temperature range. The year is provided in 'YYYY' format, the zip code is a five-digit number, and the minimum temperature is in Fahrenheit. This operation returns only the distinct start station names that meet the specified criteria.")
async def get_distinct_start_stations_by_year_zip_temp(year: str = Query(..., description="Year in 'YYYY' format"), zip_code: int = Query(..., description="Zip code"), min_temp: int = Query(..., description="Minimum temperature in Fahrenheit")):
    cursor.execute("SELECT DISTINCT T1.start_station_name FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE SUBSTR(CAST(T2.date AS TEXT), -4) = ? AND T2.zip_code = ? AND T2.max_temperature_f > ?", (year, zip_code, min_temp))
    result = cursor.fetchall()
    if not result:
        return {"start_stations": []}
    return {"start_stations": [row[0] for row in result]}

# Endpoint to get the count of trips from stations with a specific subscription type, start date pattern, and dock count
@app.get("/v1/bike_share_1/count_trips_by_subscription_date_dock", operation_id="get_count_trips_by_subscription_date_dock", summary="Retrieve the total number of trips originating from stations that meet specific criteria. These criteria include a particular subscription type, a start date matching a given pattern, and a minimum dock count. This operation provides insights into trip frequency based on subscription type, date, and station capacity.")
async def get_count_trips_by_subscription_date_dock(subscription_type: str = Query(..., description="Subscription type"), start_date_pattern: str = Query(..., description="Start date pattern in 'M/%/YYYY%' format"), min_dock_count: int = Query(..., description="Minimum dock count")):
    cursor.execute("SELECT COUNT(T2.id) FROM station AS T1 INNER JOIN trip AS T2 ON T1.id = T2.start_station_id WHERE T2.subscription_type = ? AND T2.start_date LIKE ? AND T1.dock_count > ?", (subscription_type, start_date_pattern, min_dock_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latitude and longitude of the start station with the longest trip duration
@app.get("/v1/bike_share_1/longest_trip_start_station_location", operation_id="get_longest_trip_start_station_location", summary="Retrieves the geographical coordinates of the station where the longest trip originated. The duration of the trip is calculated based on the time difference between the start and end times of the trip. The station's latitude and longitude are returned.")
async def get_longest_trip_start_station_location():
    cursor.execute("SELECT T2.lat, T2.long FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.duration = ( SELECT MAX(T1.duration) FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name )")
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": {"lat": result[0], "long": result[1]}}

# Endpoint to get the sum of available docks at the end station of a specific trip
@app.get("/v1/bike_share_1/sum_docks_available_end_station", operation_id="get_sum_docks_available_end_station", summary="Retrieves the total number of available docks at the end station of a specific trip. The trip is identified by its unique ID, which is used to determine the end station and calculate the sum of available docks.")
async def get_sum_docks_available_end_station(trip_id: int = Query(..., description="Trip ID")):
    cursor.execute("SELECT SUM(T2.docks_available) FROM trip AS T1 INNER JOIN status AS T2 ON T2.station_id = T1.end_station_id WHERE T1.ID = ?", (trip_id,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the count of trips starting from a specific station with specific weather events and date pattern
@app.get("/v1/bike_share_1/count_trips_by_weather_events", operation_id="get_count_trips_by_weather_events", summary="Get the count of trips starting from a specific station with specific weather events and date pattern")
async def get_count_trips_by_weather_events(date_pattern: str = Query(..., description="Date pattern in '%YYYY%' format"), events: str = Query(..., description="Weather events"), start_station_name: str = Query(..., description="Start station name"), zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT COUNT(T1.start_station_name) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.date LIKE ? AND T2.events = ? AND T1.start_station_name = ? AND T2.zip_code = ?", (date_pattern, events, start_station_name, zip_code))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum trip duration with specific weather events and date pattern
@app.get("/v1/bike_share_1/max_trip_duration_by_weather_events", operation_id="get_max_trip_duration_by_weather_events", summary="Retrieves the longest trip duration associated with specific weather events and date pattern within a given zip code area. The date pattern should be provided in the 'YYYY' format, and the weather events and zip code must be specified.")
async def get_max_trip_duration_by_weather_events(date_pattern: str = Query(..., description="Date pattern in '%YYYY%' format"), events: str = Query(..., description="Weather events"), zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT MAX(T1.duration) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.date LIKE ? AND T2.events = ? AND T2.zip_code = ?", (date_pattern, events, zip_code))
    result = cursor.fetchone()
    if not result:
        return {"max_duration": []}
    return {"max_duration": result[0]}

# Endpoint to get the installation date of a station based on trip ID
@app.get("/v1/bike_share_1/installation_date_by_trip_id", operation_id="get_installation_date_by_trip_id", summary="Retrieves the installation date of the station where a specific trip started. The trip is identified by its unique ID.")
async def get_installation_date_by_trip_id(trip_id: int = Query(..., description="ID of the trip")):
    cursor.execute("SELECT T2.installation_date FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.id = ?", (trip_id,))
    result = cursor.fetchone()
    if not result:
        return {"installation_date": []}
    return {"installation_date": result[0]}

# Endpoint to get the count of trips based on city and subscription type
@app.get("/v1/bike_share_1/count_trips_by_city_subscription", operation_id="get_count_trips_by_city_subscription", summary="Retrieves the total number of trips that started in a specific city and were made using a particular subscription type. The city and subscription type are provided as input parameters.")
async def get_count_trips_by_city_subscription(city: str = Query(..., description="City name"), subscription_type: str = Query(..., description="Subscription type")):
    cursor.execute("SELECT COUNT(T1.id) FROM trip AS T1 INNER JOIN station AS T2 ON T2.ID = T1.start_station_id WHERE T2.city = ? AND T1.subscription_type = ?", (city, subscription_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of trips based on date pattern, zip code, and start station name
@app.get("/v1/bike_share_1/count_trips_by_date_zip_station", operation_id="get_count_trips_by_date_zip_station", summary="Retrieve the total number of trips that started on a specific date, in a particular zip code, and at a given station. The result is filtered by the maximum temperature recorded on that date in the specified zip code. The date pattern, zip code, and start station name are required as input parameters.")
async def get_count_trips_by_date_zip_station(date_pattern: str = Query(..., description="Date pattern (e.g., '%2014%')"), zip_code: int = Query(..., description="Zip code"), start_station_name: str = Query(..., description="Start station name")):
    cursor.execute("SELECT COUNT(T1.start_station_name) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.date LIKE ? AND T2.zip_code = ? AND T1.start_station_name = ? ORDER BY T2.max_temperature_f DESC LIMIT 1", (date_pattern, zip_code, start_station_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average trip duration based on date pattern, start station name, and maximum temperature
@app.get("/v1/bike_share_1/avg_trip_duration_by_date_station_temp", operation_id="get_avg_trip_duration_by_date_station_temp", summary="Retrieves the average duration of trips that started on a specific date and at a particular station, considering the day with the highest recorded temperature.")
async def get_avg_trip_duration_by_date_station_temp(date_pattern: str = Query(..., description="Date pattern (e.g., '%2014%')"), start_station_name: str = Query(..., description="Start station name")):
    cursor.execute("SELECT AVG(T1.duration) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T2.date LIKE ? AND T1.start_station_name = ? AND T2.max_temperature_f = ( SELECT max_temperature_f FROM weather ORDER BY max_temperature_f DESC LIMIT 1 )", (date_pattern, start_station_name))
    result = cursor.fetchone()
    if not result:
        return {"average_duration": []}
    return {"average_duration": result[0]}

# Endpoint to get distinct end station names based on start station name
@app.get("/v1/bike_share_1/distinct_end_stations_by_start_station", operation_id="get_distinct_end_stations_by_start_station", summary="Retrieves a list of unique end station names that have trips originating from the specified start station.")
async def get_distinct_end_stations_by_start_station(start_station_name: str = Query(..., description="Start station name")):
    cursor.execute("SELECT DISTINCT end_station_name FROM trip WHERE start_station_name = ?", (start_station_name,))
    result = cursor.fetchall()
    if not result:
        return {"end_stations": []}
    return {"end_stations": [row[0] for row in result]}

# Endpoint to get the sum of weather events based on zip code and event type
@app.get("/v1/bike_share_1/sum_weather_events_by_zip_event", operation_id="get_sum_weather_events_by_zip_event", summary="Retrieves the total count of a specific weather event type for a given zip code. The operation calculates the sum of instances where the provided zip code and event type match the records in the weather data.")
async def get_sum_weather_events_by_zip_event(zip_code: int = Query(..., description="Zip code"), event_type: str = Query(..., description="Event type")):
    cursor.execute("SELECT SUM(IIF(zip_code = ? AND events = ?, 1, 0)) FROM weather", (zip_code, event_type))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of dock counts based on city and installation date
@app.get("/v1/bike_share_1/sum_dock_counts_by_city_installation_date", operation_id="get_sum_dock_counts_by_city_installation_date", summary="Retrieves the total number of docks installed in a specific city before a given year. The operation calculates the sum of dock counts for the specified city and installation year, providing a comprehensive view of the city's bike share infrastructure development over time.")
async def get_sum_dock_counts_by_city_installation_date(city: str = Query(..., description="City name"), installation_year: str = Query(..., description="Installation year (e.g., '2014')")):
    cursor.execute("SELECT SUM(CASE WHEN city = ? AND SUBSTR(installation_date, -4) < ? THEN dock_count ELSE 0 END) NUM FROM station", (city, installation_year))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the maximum trip duration and its equivalent in days
@app.get("/v1/bike_share_1/max_trip_duration", operation_id="get_max_trip_duration", summary="Retrieves the longest trip duration in seconds and converts it to days. This operation provides insight into the maximum time a bike was rented out in the bike share system.")
async def get_max_trip_duration():
    cursor.execute("SELECT MAX(duration), CAST(MAX(duration) AS REAL) / 86400 FROM trip")
    result = cursor.fetchone()
    if not result:
        return {"max_duration": [], "max_duration_days": []}
    return {"max_duration": result[0], "max_duration_days": result[1]}

# Endpoint to get temperatures in Celsius based on month, year, and zip code
@app.get("/v1/bike_share_1/temperatures_celsius_by_month_year_zip", operation_id="get_temperatures_celsius_by_month_year_zip", summary="Get the temperatures in Celsius based on the month, year, and zip code")
async def get_temperatures_celsius_by_month_year_zip(month: str = Query(..., description="Month (e.g., '8')"), year: str = Query(..., description="Year (e.g., '2013')"), zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT (max_temperature_f - 32) / 1.8000 , (mean_temperature_f - 32) / 1.8000 , (min_temperature_f - 32) / 1.8000 FROM weather WHERE SUBSTR(CAST(date AS TEXT), 1, INSTR(date, '/') - 1) = ? AND SUBSTR(CAST(date AS TEXT), -4) = ? AND zip_code = ?", (month, year, zip_code))
    result = cursor.fetchone()
    if not result:
        return {"max_temperature_c": [], "mean_temperature_c": [], "min_temperature_c": []}
    return {"max_temperature_c": result[0], "mean_temperature_c": result[1], "min_temperature_c": result[2]}

# Endpoint to get the ratio of subscriber trips to customer trips based on start and end station names
@app.get("/v1/bike_share_1/subscriber_customer_ratio_by_stations", operation_id="get_subscriber_customer_ratio_by_stations", summary="Retrieve the proportion of trips made by subscribers relative to customers, based on the specified start and end station names. This operation calculates the ratio by comparing the total number of trips made by each subscription type.")
async def get_subscriber_customer_ratio_by_stations(subscription_type_1: str = Query(..., description="First subscription type (e.g., 'Subscriber')"), subscription_type_2: str = Query(..., description="Second subscription type (e.g., 'Customer')"), start_station_name: str = Query(..., description="Start station name"), end_station_name: str = Query(..., description="End station name")):
    cursor.execute("SELECT CAST(SUM(IIF(subscription_type = ?, 1, 0)) AS REAL) / SUM(IIF(subscription_type = ?, 1, 0)) FROM trip WHERE start_station_name = ? AND end_station_name = ?", (subscription_type_1, subscription_type_2, start_station_name, end_station_name))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get distinct cities based on zip code
@app.get("/v1/bike_share_1/distinct_cities_by_zip_code", operation_id="get_distinct_cities_by_zip_code", summary="Retrieve a list of unique cities that have a bike share station with the specified zip code. The operation filters the cities based on the provided zip code.")
async def get_distinct_cities_by_zip_code(zip_code: int = Query(..., description="Zip code to filter the cities")):
    cursor.execute("SELECT DISTINCT T2.city FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.zip_code = ?", (zip_code,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get distinct start and end station names based on date and mean temperature
@app.get("/v1/bike_share_1/distinct_stations_by_date_temperature", operation_id="get_distinct_stations_by_date_temperature", summary="Retrieve the unique start and end station names for bike trips that occurred on a specific date and under a certain mean temperature. The date is provided in 'YYYY' format, and the temperature is in Fahrenheit. This operation considers weather data and trip information to provide accurate results.")
async def get_distinct_stations_by_date_temperature(date: str = Query(..., description="Date in 'YYYY' format to filter the stations"), mean_temperature_f: float = Query(..., description="Mean temperature in Fahrenheit to filter the stations")):
    cursor.execute("SELECT DISTINCT T2.start_station_name, T2.end_station_name FROM weather AS T1 INNER JOIN trip AS T2 ON T1.zip_code = T2.zip_code WHERE T1.date LIKE ? AND T1.mean_temperature_f = ?", (date, mean_temperature_f))
    result = cursor.fetchall()
    if not result:
        return {"stations": []}
    return {"stations": [{"start_station_name": row[0], "end_station_name": row[1]} for row in result]}

# Endpoint to get the count of trips based on city, start date, and station names
@app.get("/v1/bike_share_1/count_trips_by_city_date_stations", operation_id="get_count_trips_by_city_date_stations", summary="Retrieves the total number of trips that started and ended on specific dates in a given city, filtered by the start and end station names. The start date must be provided in the 'M/%/YYYY%' format.")
async def get_count_trips_by_city_date_stations(city: str = Query(..., description="City to filter the trips"), start_date: str = Query(..., description="Start date in 'M/%/YYYY%' format to filter the trips"), start_station_name: str = Query(..., description="Start station name to filter the trips"), end_station_name: str = Query(..., description="End station name to filter the trips")):
    cursor.execute("SELECT COUNT(T2.id) FROM station AS T1 INNER JOIN trip AS T2 ON T2.start_station_name = T1.name WHERE T1.city = ? AND T2.start_date LIKE ? AND T2.start_station_name LIKE ? AND T2.end_station_name LIKE ?", (city, start_date, start_station_name, end_station_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get start and end station names based on start date and different start and end stations
@app.get("/v1/bike_share_1/stations_by_start_date_different_stations", operation_id="get_stations_by_start_date_different_stations", summary="Retrieve the names of the start and end stations for trips that began on a specific date and had different start and end stations. The start date must be provided in the format 'MM/DD/YYYY' to filter the results.")
async def get_stations_by_start_date_different_stations(start_date: str = Query(..., description="Start date in '%/%/YYYY%' format to filter the stations")):
    cursor.execute("SELECT T1.start_station_name, T1.end_station_name FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.start_date LIKE ? AND T1.start_station_name != T1.end_station_name", (start_date,))
    result = cursor.fetchall()
    if not result:
        return {"stations": []}
    return {"stations": [{"start_station_name": row[0], "end_station_name": row[1]} for row in result]}

# Endpoint to get end station name, city, and total duration in hours based on bike ID and different start and end stations
@app.get("/v1/bike_share_1/end_station_city_duration_by_bike_id", operation_id="get_end_station_city_duration_by_bike_id", summary="Retrieves the end station name, city, and total duration in hours for trips made with a specific bike ID, excluding trips that start and end at the same station.")
async def get_end_station_city_duration_by_bike_id(bike_id: int = Query(..., description="Bike ID to filter the trips")):
    cursor.execute("SELECT T1.end_station_name, T2.city, CAST(SUM(T1.duration) AS REAL) / 3600 FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.bike_id = ? AND T1.start_station_name != T1.end_station_name", (bike_id,))
    result = cursor.fetchall()
    if not result:
        return {"trips": []}
    return {"trips": [{"end_station_name": row[0], "city": row[1], "total_duration_hours": row[2]} for row in result]}

# Endpoint to get the percentage of customer trips compared to subscriber trips based on city
@app.get("/v1/bike_share_1/customer_subscriber_percentage_by_city", operation_id="get_customer_subscriber_percentage_by_city", summary="Retrieves the proportion of trips made by customers relative to subscribers in a specified city. The calculation is based on the total number of trips originating from the given city.")
async def get_customer_subscriber_percentage_by_city(city: str = Query(..., description="City to filter the trips")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.subscription_type = 'Customer' THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN T1.subscription_type = 'Subscriber' THEN 1 ELSE 0 END) FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total duration in hours of trips based on city
@app.get("/v1/bike_share_1/total_duration_by_city", operation_id="get_total_duration_by_city", summary="Retrieves the total duration of all trips originating from the specified city, expressed in hours. The city is used as a filter to calculate the sum of trip durations.")
async def get_total_duration_by_city(city: str = Query(..., description="City to filter the trips")):
    cursor.execute("SELECT CAST(SUM(T1.duration) AS REAL) / 3600 FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"total_duration_hours": []}
    return {"total_duration_hours": result[0]}

# Endpoint to get the bike with the most trips and its details
@app.get("/v1/bike_share_1/most_trips_bike_details", operation_id="get_most_trips_bike_details", summary="Retrieves the bike that has been used for the most trips, along with its start and end station names, the city, and the average duration of the trips in hours. The data is grouped by bike ID and ordered by the number of trips in descending order, with the top result being returned.")
async def get_most_trips_bike_details():
    cursor.execute("SELECT T2.bike_id, T2.start_station_name, T2.end_station_name, T1.city, CAST(T2.duration AS REAL) / 3600 FROM station AS T1 INNER JOIN trip AS T2 ON T1.name = T2.start_station_name GROUP BY T2.bike_id ORDER BY COUNT(T2.id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"bike_details": []}
    return {"bike_details": {"bike_id": result[0], "start_station_name": result[1], "end_station_name": result[2], "city": result[3], "total_duration_hours": result[4]}}

# Endpoint to get the count of subscriber trips based on start station name
@app.get("/v1/bike_share_1/count_subscriber_trips_by_start_station", operation_id="get_count_subscriber_trips_by_start_station", summary="Retrieves the total number of trips taken by subscribers from a specific start station. The start station is determined by the provided station name.")
async def get_count_subscriber_trips_by_start_station(start_station_name: str = Query(..., description="Start station name to filter the trips")):
    cursor.execute("SELECT COUNT(CASE WHEN subscription_type = 'Subscriber' AND start_station_name = ? THEN id END) FROM trip", (start_station_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get station names based on installation date and city
@app.get("/v1/bike_share_1/station_names_by_installation_date_city", operation_id="get_station_names_by_installation_date_city", summary="Retrieves the names of all bike share stations in a specified city that were installed on a given date. The date should be provided in 'MM/DD/YYYY' format.")
async def get_station_names_by_installation_date_city(installation_date: str = Query(..., description="Installation date in 'MM/DD/YYYY' format to filter the stations"), city: str = Query(..., description="City to filter the stations")):
    cursor.execute("SELECT name FROM station WHERE installation_date = ? AND city = ?", (installation_date, city))
    result = cursor.fetchall()
    if not result:
        return {"station_names": []}
    return {"station_names": [row[0] for row in result]}

# Endpoint to get the sum of dock counts for a specific station name
@app.get("/v1/bike_share_1/sum_dock_count_by_station_name", operation_id="get_sum_dock_count", summary="Retrieves the total number of docks for a specified station, aggregated by city. The station is identified by its name.")
async def get_sum_dock_count(station_name: str = Query(..., description="Name of the station")):
    cursor.execute("SELECT city, SUM(dock_count) FROM station WHERE name = ?", (station_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum_dock_count": []}
    return {"city": result[0], "sum_dock_count": result[1]}

# Endpoint to get the sum of dock counts and count of subscription types for a specific station name and subscription type
@app.get("/v1/bike_share_1/sum_dock_count_subscription_type_count", operation_id="get_sum_dock_count_subscription_type_count", summary="Retrieves the total number of docks and the count of a specific subscription type for a given station. The station is identified by its name, and the subscription type is specified as an input parameter.")
async def get_sum_dock_count_subscription_type_count(station_name: str = Query(..., description="Name of the station"), subscription_type: str = Query(..., description="Subscription type")):
    cursor.execute("SELECT SUM(T2.dock_count), COUNT(T1.subscription_type) FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T2.name = ? AND T1.start_station_name = T2.name AND T1.subscription_type = ?", (station_name, subscription_type))
    result = cursor.fetchone()
    if not result:
        return {"sum_dock_count": [], "subscription_type_count": []}
    return {"sum_dock_count": result[0], "subscription_type_count": result[1]}

# Endpoint to get the count of zip codes and maximum temperature for a specific zip code and subscription type
@app.get("/v1/bike_share_1/count_zip_codes_max_temperature", operation_id="get_count_zip_codes_max_temperature", summary="Retrieves the count of unique zip codes and the maximum recorded temperature for a given zip code and subscription type. The data is filtered based on the provided zip code and subscription type, and the maximum temperature is determined from the weather data associated with the filtered trips.")
async def get_count_zip_codes_max_temperature(zip_code: int = Query(..., description="Zip code"), subscription_type: str = Query(..., description="Subscription type")):
    cursor.execute("SELECT COUNT(T3.zip_code), T3.max_temperature_f FROM trip AS T2 INNER JOIN weather AS T3 ON T3.zip_code = T2.zip_code WHERE T3.zip_code = ? AND T2.subscription_type = ? ORDER BY T3.max_temperature_f DESC LIMIT 1", (zip_code, subscription_type))
    result = cursor.fetchone()
    if not result:
        return {"zip_code_count": [], "max_temperature": []}
    return {"zip_code_count": result[0], "max_temperature": result[1]}

# Endpoint to get the average trip duration and coordinates for a specific start station name
@app.get("/v1/bike_share_1/avg_trip_duration_coordinates", operation_id="get_avg_trip_duration_coordinates", summary="Retrieves the average duration of trips that started at the specified station, along with the latitude and longitude coordinates of that station.")
async def get_avg_trip_duration_coordinates(start_station_name: str = Query(..., description="Start station name")):
    cursor.execute("SELECT AVG(T1.duration), T2.lat, T2.long FROM trip AS T1 LEFT JOIN station AS T2 ON T2.name = T1.start_station_name LEFT JOIN station AS T3 ON T3.name = T1.end_station_name WHERE T1.start_station_name = ?", (start_station_name,))
    result = cursor.fetchone()
    if not result:
        return {"avg_duration": [], "lat": [], "long": []}
    return {"avg_duration": result[0], "lat": result[1], "long": result[2]}

# Endpoint to get the minimum trip duration and maximum wind speed for a specific start station name and date
@app.get("/v1/bike_share_1/min_trip_duration_max_wind_speed", operation_id="get_min_trip_duration_max_wind_speed", summary="Retrieves the shortest trip duration and the highest wind speed recorded for a given start station and date. The start station is identified by its name, and the date is provided in 'MM/DD/YYYY' format. This operation combines data from the 'trip' and 'weather' tables, using the zip code as a common link between them.")
async def get_min_trip_duration_max_wind_speed(start_station_name: str = Query(..., description="Start station name"), date: str = Query(..., description="Date in 'MM/DD/YYYY' format")):
    cursor.execute("SELECT MIN(T1.duration), MAX(T2.max_wind_Speed_mph) FROM trip AS T1 INNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code WHERE T1.start_station_name = ? AND T2.date = ?", (start_station_name, date))
    result = cursor.fetchone()
    if not result:
        return {"min_duration": [], "max_wind_speed": []}
    return {"min_duration": result[0], "max_wind_speed": result[1]}

# Endpoint to get the sum of available bikes and coordinates for a specific station name and time
@app.get("/v1/bike_share_1/sum_bikes_available_coordinates", operation_id="get_sum_bikes_available_coordinates", summary="Retrieves the total number of available bikes and the corresponding geographical coordinates for a specific bike station at a given time. The station is identified by its name, and the time is provided in 'YYYY/MM/DD HH:MM:SS' format.")
async def get_sum_bikes_available_coordinates(time: str = Query(..., description="Time in 'YYYY/MM/DD HH:MM:SS' format"), station_name: str = Query(..., description="Station name")):
    cursor.execute("SELECT SUM(T2.bikes_available), T1.long, T1.lat FROM station AS T1 INNER JOIN status AS T2 ON T2.station_id = T1.id WHERE T2.time = ? AND T1.name = ?", (time, station_name))
    result = cursor.fetchone()
    if not result:
        return {"sum_bikes_available": [], "long": [], "lat": []}
    return {"sum_bikes_available": result[0], "long": result[1], "lat": result[2]}

# Endpoint to get the city and installation date for a specific trip ID
@app.get("/v1/bike_share_1/city_installation_date_by_trip_id", operation_id="get_city_installation_date_by_trip_id", summary="Retrieves the city and installation date of the station where a specific trip started. The trip is identified by its unique ID.")
async def get_city_installation_date_by_trip_id(trip_id: int = Query(..., description="Trip ID")):
    cursor.execute("SELECT T2.city, T2.installation_date FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.id = ?", (trip_id,))
    result = cursor.fetchone()
    if not result:
        return {"city": [], "installation_date": []}
    return {"city": result[0], "installation_date": result[1]}

# Endpoint to get the count of station names and sum of customer subscription types for a specific installation date and subscription type
@app.get("/v1/bike_share_1/count_station_names_sum_customer_subscription", operation_id="get_count_station_names_sum_customer_subscription", summary="Retrieves the total number of unique station names and the sum of a specific subscription type for trips starting on a given installation date. The subscription type is used to filter the trips considered in the sum.")
async def get_count_station_names_sum_customer_subscription(installation_date: str = Query(..., description="Installation date in 'MM/DD/YYYY' format"), subscription_type: str = Query(..., description="Subscription type")):
    cursor.execute("SELECT COUNT(T1.name), SUM(CASE WHEN T2.subscription_type = 'Customer' THEN 1 ELSE 0 END) FROM station AS T1 INNER JOIN trip AS T2 ON T2.start_station_name = T1.name WHERE T1.installation_date = ? AND T2.subscription_type = ?", (installation_date, subscription_type))
    result = cursor.fetchone()
    if not result:
        return {"station_name_count": [], "customer_subscription_sum": []}
    return {"station_name_count": result[0], "customer_subscription_sum": result[1]}

# Endpoint to get the station name and coordinates for a specific start station name and end date
@app.get("/v1/bike_share_1/station_name_coordinates_by_start_station_end_date", operation_id="get_station_name_coordinates_by_start_station_end_date", summary="Get the station name and coordinates for a specific start station name and end date")
async def get_station_name_coordinates_by_start_station_end_date(start_station_name: str = Query(..., description="Start station name"), end_date: str = Query(..., description="End date in 'MM/DD/YYYY HH:MM' format")):
    cursor.execute("SELECT T1.name, T1.lat, T1.long FROM station AS T1 INNER JOIN trip AS T2 ON T2.end_station_name = T1.name WHERE T2.start_station_name = ? AND T2.end_date = ?", (start_station_name, end_date))
    result = cursor.fetchone()
    if not result:
        return {"station_name": [], "lat": [], "long": []}
    return {"station_name": result[0], "lat": result[1], "long": result[2]}

# Endpoint to get the count of trips and dock count based on end station name, subscription type, and dock count
@app.get("/v1/bike_share_1/trip_count_dock_count", operation_id="get_trip_count_dock_count", summary="Retrieves the total number of trips and the corresponding dock count for a specific end station, filtered by subscription type and dock count.")
async def get_trip_count_dock_count(end_station_name: str = Query(..., description="End station name"), subscription_type: str = Query(..., description="Subscription type"), dock_count: int = Query(..., description="Dock count")):
    cursor.execute("SELECT COUNT(T1.id), T2.dock_count FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.end_station_name = ? AND T1.subscription_type = ? AND T2.dock_count = ?", (end_station_name, subscription_type, dock_count))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "dock_count": []}
    return {"count": result[0], "dock_count": result[1]}

# Endpoint to get the average minimum temperature, longitude, and latitude based on zip code
@app.get("/v1/bike_share_1/avg_min_temp_long_lat", operation_id="get_avg_min_temp_long_lat", summary="Retrieves the average minimum temperature, along with the corresponding longitude and latitude, for the specified zip code. This operation calculates the average minimum temperature based on weather data associated with the given zip code and the start station of bike trips. The longitude and latitude are derived from the start station's location.")
async def get_avg_min_temp_long_lat(zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT AVG(T3.min_temperature_f), T1.long, T1.lat FROM station AS T1 INNER JOIN trip AS T2 ON T2.start_station_name = T1.name INNER JOIN weather AS T3 ON T3.zip_code = T2.zip_code WHERE T3.zip_code = ?", (zip_code,))
    result = cursor.fetchone()
    if not result:
        return {"avg_min_temperature_f": [], "long": [], "lat": []}
    return {"avg_min_temperature_f": result[0], "long": result[1], "lat": result[2]}

# Endpoint to get the average trip duration and installation date based on start station name, subscription type, and end station name
@app.get("/v1/bike_share_1/avg_trip_duration_installation_date", operation_id="get_avg_trip_duration_installation_date", summary="Retrieves the average duration of trips and the corresponding installation date for a specific route, defined by the start and end station names, and the type of subscription used.")
async def get_avg_trip_duration_installation_date(start_station_name: str = Query(..., description="Start station name"), subscription_type: str = Query(..., description="Subscription type"), end_station_name: str = Query(..., description="End station name")):
    cursor.execute("SELECT AVG(T1.duration), T2.installation_date FROM trip AS T1 INNER JOIN station AS T2 ON T2.name = T1.start_station_name WHERE T1.start_station_name = ? AND T1.subscription_type = ? AND T1.end_station_name = ?", (start_station_name, subscription_type, end_station_name))
    result = cursor.fetchone()
    if not result:
        return {"avg_duration": [], "installation_date": []}
    return {"avg_duration": result[0], "installation_date": result[1]}

api_calls = [
    "/v1/bike_share_1/trip_with_max_duration",
    "/v1/bike_share_1/subscription_type_percentage?subscription_type=Subscriber",
    "/v1/bike_share_1/recent_trip_by_bike_id?bike_id=13",
    "/v1/bike_share_1/cities_with_different_start_end_months",
    "/v1/bike_share_1/stations_no_bikes_available?time=2013/11/03%2002:01:01&bikes_available=0",
    "/v1/bike_share_1/most_popular_start_station",
    "/v1/bike_share_1/max_temperature_for_trip?trip_id=4080",
    "/v1/bike_share_1/max_bikes_available_time?station_name=San%20Jose%20Diridon%20Caltrain%20Station",
    "/v1/bike_share_1/trips_during_rain?event1=%25Rain%25&event2=%25rain%25",
    "/v1/bike_share_1/trips_ending_at_location?lat=37.331415&long=-121.8932",
    "/v1/bike_share_1/count_trips_start_date_city?start_date_pattern=8/%/2013%&city=Redwood%20City",
    "/v1/bike_share_1/trip_details_mean_temperature?max_duration=300",
    "/v1/bike_share_1/date_highest_trips_avg_min_temp",
    "/v1/bike_share_1/avg_trip_duration_wind_direction?month1=7&month2=8&month3=9&year=2013",
    "/v1/bike_share_1/count_stations_installed_year_city?city=San%20Jose&year=2014",
    "/v1/bike_share_1/max_trip_duration_date?start_date_pattern=8/29/2013%&end_date_pattern=8/29/2013%",
    "/v1/bike_share_1/trip_duration_minutes?bike_id=426&end_station_name=2nd%20at%20South%20Park&start_station_name=Market%20at%204th&start_date_pattern=8/29/2013%&end_date_pattern=8/29/2013%",
    "/v1/bike_share_1/longest_trip_bike_id?start_date_pattern=8/29/2013%&end_date_pattern=8/29/2013%&end_station_name=California%20Ave%20Caltrain%20Station&start_station_name=University%20and%20Emerson",
    "/v1/bike_share_1/count_stations_dock_count?city=San%20Francisco&min_dock_count=20",
    "/v1/bike_share_1/max_temperature_date",
    "/v1/bike_share_1/max_dew_point_by_date_zip?date=7%2F15%2F2014&zip_code=94301",
    "/v1/bike_share_1/year_with_most_rain_events?event1=%25Rain%25&event2=%25rain%25",
    "/v1/bike_share_1/min_trip_duration_by_date_location?start_date=10%2F20%2F2014%25&lat=37.789625&long=-122.400811",
    "/v1/bike_share_1/min_trip_duration_by_start_date_station_subscription?start_date=12%2F1%2F2013%25&start_station_name=South%20Van%20Ness%20at%20Market&subscription_type=Subscriber",
    "/v1/bike_share_1/max_humidity_by_trip_details?start_date=8%2F29%2F2013%25&bike_id=496&start_station_name=Powell%20Street%20BART",
    "/v1/bike_share_1/trip_count_by_date_zip_event_subscription?date_pattern=11%2F%25%2F2014%25&zip_code=94301&events=Fog&subscription_type=Subscriber",
    "/v1/bike_share_1/most_popular_start_station_by_subscription?subscription_type=Customer",
    "/v1/bike_share_1/stations_with_no_bikes_available?bikes_available=0&time_pattern=2013%2F11%2F03%25",
    "/v1/bike_share_1/avg_trip_duration_by_city?city=Palo%20Alto",
    "/v1/bike_share_1/station_details_by_installation_date?month=5&day=8&year=2013&months_list=6,7,8,9,10,11,12&year_filter=2013",
    "/v1/bike_share_1/average_trip_duration?start_station_name=Adobe%20on%20Almaden&end_station_name=Ryland%20Park",
    "/v1/bike_share_1/station_status_no_bikes?bikes_available=0",
    "/v1/bike_share_1/trip_details_same_start_end",
    "/v1/bike_share_1/sum_available_docks?station_name=San%20Jose%20Diridon%20Caltrain%20Station&time=2013/08/29%2006:14:01",
    "/v1/bike_share_1/trip_ids_by_bike_temp_subscription?bike_id=10&mean_temperature_f=62&subscription_type=Subscriber",
    "/v1/bike_share_1/weather_details_by_trip_criteria?bike_id=10&mean_temperature_f=62&subscription_type=Customer&start_station_name=MLK%20Library&end_station_name=San%20Salvador%20at%201st&duration=386",
    "/v1/bike_share_1/count_trips_by_criteria?subscription_type=Subscriber&min_visibility_miles=4&duration=490&start_station_name=2nd%20at%20Folsom&end_station_name=Civic%20Center%20BART%20(7th%20at%20Market)",
    "/v1/bike_share_1/sum_available_docks_by_zip?zip_code=912900",
    "/v1/bike_share_1/trip_ids_by_min_temperature?min_temperature_f=45",
    "/v1/bike_share_1/trip_duration_temperature?start_date=1/1/2014%200:00&end_date=12/31/2014%2011:59&start_station_name=2nd%20at%20Folsom&end_station_name=5th%20at%20Howard&subscription_type=Subscriber",
    "/v1/bike_share_1/average_temperature_humidity?date=5/%25/2015&min_humidity=65&max_humidity=75",
    "/v1/bike_share_1/subscription_difference?subscriber_type=Subscriber&customer_type=Customer&start_date=6/%25/2013%25",
    "/v1/bike_share_1/trip_date_bike_id?year=2013&events=Fog-Rain",
    "/v1/bike_share_1/weather_trip_details?events=Fog",
    "/v1/bike_share_1/trip_start_date_min_temperature?date=8/%25/2013&start_station_name=Market%20at%2010th&end_station_name=South%20Van%20Ness%20at%20Market",
    "/v1/bike_share_1/longest_trip_weather_events?event1=Rain&event2=rain",
    "/v1/bike_share_1/average_trip_duration_weather_precipitation?event1=Rain&precipitation1=0.8&event2=rain&precipitation2=0.8",
    "/v1/bike_share_1/long_trip_stations_cities",
    "/v1/bike_share_1/sum_stations_city_year?city=San%20Francisco&year=2014",
    "/v1/bike_share_1/count_trips_end_city_year?city=Mountain%20View&year=2006",
    "/v1/bike_share_1/max_duration_trip_id_city?city=Redwood%20City",
    "/v1/bike_share_1/distinct_bike_ids_installation_year?year=2013",
    "/v1/bike_share_1/count_trips_subscription_type_city?subscription_type=Subscriber&city=San%20Jose",
    "/v1/bike_share_1/min_duration_trip_id_dock_count?dock_count=15",
    "/v1/bike_share_1/most_popular_start_station_city?city=San%20Francisco",
    "/v1/bike_share_1/percentage_long_trips_city?duration=800&city=San%20Jose",
    "/v1/bike_share_1/count_long_trips_year?year=2013&duration=1000",
    "/v1/bike_share_1/avg_trip_duration_station_year?year=2015&start_station_name=South%20Van%20Ness%20at%20Market",
    "/v1/bike_share_1/sum_round_trips",
    "/v1/bike_share_1/trip_with_highest_max_temperature",
    "/v1/bike_share_1/trip_ids_by_weather_event_and_start_station?weather_event=Rain&start_station_name=Mountain%20View%20City%20Hall",
    "/v1/bike_share_1/average_trip_duration_by_weather_event?weather_event=Fog",
    "/v1/bike_share_1/longest_trip_duration_by_max_wind_speed?max_wind_speed=30",
    "/v1/bike_share_1/average_mean_temperature_by_year_and_start_station?year=2013&start_station_name=Market%20at%204th",
    "/v1/bike_share_1/mean_humidity_by_trip_id?trip_id=4275",
    "/v1/bike_share_1/percentage_trips_by_weather_event_year_subscription?weather_event=Rain&year=2015&subscription_type=Customer",
    "/v1/bike_share_1/count_station_installations_by_city_and_date?city=San%20Jose&month1=08&month2=09&month3=10&month4=11&month5=12&year1=2013&year2=2013",
    "/v1/bike_share_1/sum_mean_temperatures_by_zip_code_and_date?zip_code=94107&date=8/29/2013",
    "/v1/bike_share_1/sum_temperature_difference?zip_code=94107&date=8/29/2013",
    "/v1/bike_share_1/bikes_available_at_station?station_name=San%20Jose%20Diridon%20Caltrain%20Station&time=2013/08/29%2012:06:01",
    "/v1/bike_share_1/trip_start_station_city?trip_id=4069",
    "/v1/bike_share_1/count_trips_by_date_zip_temp?date_pattern=9/%/2013&zip_code=94107&min_temp=70",
    "/v1/bike_share_1/distinct_start_stations_by_year_zip_temp?year=2013&zip_code=94107&min_temp=80",
    "/v1/bike_share_1/count_trips_by_subscription_date_dock?subscription_type=Subscriber&start_date_pattern=8/%/2013%&min_dock_count=20",
    "/v1/bike_share_1/longest_trip_start_station_location",
    "/v1/bike_share_1/sum_docks_available_end_station?trip_id=4069",
    "/v1/bike_share_1/count_trips_by_weather_events?date_pattern=%2013%&events=Fog&start_station_name=2nd%20at%20Townsend&zip_code=94107",
    "/v1/bike_share_1/max_trip_duration_by_weather_events?date_pattern=%2013%&events=Fog&zip_code=94107",
    "/v1/bike_share_1/installation_date_by_trip_id?trip_id=4069",
    "/v1/bike_share_1/count_trips_by_city_subscription?city=San%20Francisco&subscription_type=Subscriber",
    "/v1/bike_share_1/count_trips_by_date_zip_station?date_pattern=%252014%25&zip_code=94107&start_station_name=2nd%20at%20Folsom",
    "/v1/bike_share_1/avg_trip_duration_by_date_station_temp?date_pattern=%252014%25&start_station_name=2nd%20at%20Folsom",
    "/v1/bike_share_1/distinct_end_stations_by_start_station?start_station_name=2nd%20at%20South%20Park",
    "/v1/bike_share_1/sum_weather_events_by_zip_event?zip_code=94041&event_type=Rain",
    "/v1/bike_share_1/sum_dock_counts_by_city_installation_date?city=Redwood%20City&installation_year=2014",
    "/v1/bike_share_1/max_trip_duration",
    "/v1/bike_share_1/temperatures_celsius_by_month_year_zip?month=8&year=2013&zip_code=94107",
    "/v1/bike_share_1/subscriber_customer_ratio_by_stations?subscription_type_1=Subscriber&subscription_type_2=Customer&start_station_name=2nd%20at%20South%20Park&end_station_name=2nd%20at%20South%20Park",
    "/v1/bike_share_1/distinct_cities_by_zip_code?zip_code=94107",
    "/v1/bike_share_1/distinct_stations_by_date_temperature?date=%252014&mean_temperature_f=68.0",
    "/v1/bike_share_1/count_trips_by_city_date_stations?city=San%20Jose&start_date=8%25%2F2013%25&start_station_name=San%20Jose%25&end_station_name=San%20Jose%25",
    "/v1/bike_share_1/stations_by_start_date_different_stations?start_date=%25%2F%25%2F2014%25",
    "/v1/bike_share_1/end_station_city_duration_by_bike_id?bike_id=16",
    "/v1/bike_share_1/customer_subscriber_percentage_by_city?city=Mountain%20View",
    "/v1/bike_share_1/total_duration_by_city?city=Palo%20Alto",
    "/v1/bike_share_1/most_trips_bike_details",
    "/v1/bike_share_1/count_subscriber_trips_by_start_station?start_station_name=Market%20at%204th",
    "/v1/bike_share_1/station_names_by_installation_date_city?installation_date=12%2F31%2F2013&city=Mountain%20View",
    "/v1/bike_share_1/sum_dock_count_by_station_name?station_name=Townsend%20at%207th",
    "/v1/bike_share_1/sum_dock_count_subscription_type_count?station_name=Evelyn%20Park%20and%20Ride&subscription_type=Subscriber",
    "/v1/bike_share_1/count_zip_codes_max_temperature?zip_code=94301&subscription_type=Subscriber",
    "/v1/bike_share_1/avg_trip_duration_coordinates?start_station_name=Santa%20Clara%20at%20Almaden",
    "/v1/bike_share_1/min_trip_duration_max_wind_speed?start_station_name=Franklin%20at%20Maple&date=9/4/2013",
    "/v1/bike_share_1/sum_bikes_available_coordinates?time=2013/10/20%208:11:01&station_name=San%20Jose%20Diridon%20Caltrain%20Station",
    "/v1/bike_share_1/city_installation_date_by_trip_id?trip_id=585842",
    "/v1/bike_share_1/count_station_names_sum_customer_subscription?installation_date=8/16/2013&subscription_type=Customer",
    "/v1/bike_share_1/station_name_coordinates_by_start_station_end_date?start_station_name=Market%20at%204th&end_date=8/29/2013%2012:45",
    "/v1/bike_share_1/trip_count_dock_count?end_station_name=MLK%20Library&subscription_type=Subscriber&dock_count=19",
    "/v1/bike_share_1/avg_min_temp_long_lat?zip_code=94301",
    "/v1/bike_share_1/avg_trip_duration_installation_date?start_station_name=Mountain%20View%20City%20Hall&subscription_type=Subscriber&end_station_name=Mountain%20View%20City%20Hall"
]
