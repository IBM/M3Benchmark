from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/formula_1/formula_1.sqlite')
cursor = conn.cursor()

# Endpoint to get driver references for a specific race ID ordered by q1 time in descending order
@app.get("/v1/formula_1/driver_references_by_race_id", operation_id="get_driver_references", summary="Retrieves the top five driver references for a given race, sorted by their fastest qualifying lap time in descending order. The race is identified by its unique ID.")
async def get_driver_references(race_id: int = Query(..., description="Race ID")):
    cursor.execute("SELECT T2.driverRef FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = ? ORDER BY T1.q1 DESC LIMIT 5", (race_id,))
    result = cursor.fetchall()
    if not result:
        return {"driver_references": []}
    return {"driver_references": [row[0] for row in result]}

# Endpoint to get the surname of the driver with the fastest q2 time for a specific race ID
@app.get("/v1/formula_1/fastest_q2_surname_by_race_id", operation_id="get_fastest_q2_surname", summary="Retrieves the surname of the driver who achieved the fastest qualifying 2 (Q2) time for a specific race. The race is identified by its unique race ID.")
async def get_fastest_q2_surname(race_id: int = Query(..., description="Race ID")):
    cursor.execute("SELECT T2.surname FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = ? ORDER BY T1.q2 ASC LIMIT 1", (race_id,))
    result = cursor.fetchone()
    if not result:
        return {"surname": []}
    return {"surname": result[0]}

# Endpoint to get the years of races held at a specific location
@app.get("/v1/formula_1/race_years_by_location", operation_id="get_race_years", summary="Retrieves the years in which Formula 1 races were held at the specified location. The location parameter is used to filter the results.")
async def get_race_years(location: str = Query(..., description="Location of the circuit")):
    cursor.execute("SELECT T2.year FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.location = ?", (location,))
    result = cursor.fetchall()
    if not result:
        return {"years": []}
    return {"years": [row[0] for row in result]}

# Endpoint to get the distinct URLs of circuits with a specific name
@app.get("/v1/formula_1/circuit_urls_by_name", operation_id="get_circuit_urls", summary="Retrieve a unique set of URLs for circuits that share a specified name. This operation filters circuits based on their name and returns the distinct URLs associated with them.")
async def get_circuit_urls(name: str = Query(..., description="Name of the circuit")):
    cursor.execute("SELECT DISTINCT T1.url FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"urls": []}
    return {"urls": [row[0] for row in result]}

# Endpoint to get the distinct names of races held in a specific country
@app.get("/v1/formula_1/race_names_by_country", operation_id="get_race_names", summary="Retrieve a unique list of race names that have been held in a specified country. The operation filters the races based on the provided country and returns only the distinct race names.")
async def get_race_names(country: str = Query(..., description="Country where the circuit is located")):
    cursor.execute("SELECT DISTINCT T2.name FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the distinct positions of a specific constructor in the standings
@app.get("/v1/formula_1/constructor_positions_by_name", operation_id="get_constructor_positions", summary="Retrieve the unique standings positions held by a specific constructor in the Formula 1 rankings. The constructor's name is used to filter the results.")
async def get_constructor_positions(name: str = Query(..., description="Name of the constructor")):
    cursor.execute("SELECT DISTINCT T1.position FROM constructorStandings AS T1 INNER JOIN constructors AS T2 ON T2.constructorId = T1.constructorId WHERE T2.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"positions": []}
    return {"positions": [row[0] for row in result]}

# Endpoint to get the count of races held in countries not in a specified list for a specific year
@app.get("/v1/formula_1/race_count_by_excluded_countries_and_year", operation_id="get_race_count", summary="Retrieves the total number of Formula 1 races held in countries not included in the provided list of excluded countries for a specific year.")
async def get_race_count(excluded_countries: str = Query(..., description="Comma-separated list of countries to exclude"), year: int = Query(..., description="Year of the races")):
    excluded_countries_list = excluded_countries.split(',')
    cursor.execute("SELECT COUNT(T3.raceId) FROM circuits AS T1 INNER JOIN races AS T3 ON T3.circuitID = T1.circuitId WHERE T1.country NOT IN (" + ",".join(["?"] * len(excluded_countries_list)) + ") AND T3.year = ?", tuple(excluded_countries_list + [year]))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the distinct latitude and longitude of circuits where a specific race was held
@app.get("/v1/formula_1/circuit_coordinates_by_race_name", operation_id="get_circuit_coordinates", summary="Retrieve the unique geographical coordinates of circuits where a particular race was conducted. The operation requires the name of the race as input.")
async def get_circuit_coordinates(race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT DISTINCT T1.lat, T1.lng FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T2.name = ?", (race_name,))
    result = cursor.fetchall()
    if not result:
        return {"coordinates": []}
    return {"coordinates": [{"lat": row[0], "lng": row[1]} for row in result]}

# Endpoint to get distinct race times for a specific circuit
@app.get("/v1/formula_1/race_times_by_circuit", operation_id="get_race_times_by_circuit", summary="Retrieve unique race times for a specified circuit. This operation fetches and returns the distinct race times associated with the provided circuit name. The input parameter is used to identify the circuit of interest.")
async def get_race_times_by_circuit(circuit_name: str = Query(..., description="Name of the circuit")):
    cursor.execute("SELECT DISTINCT T2.time FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.name = ?", (circuit_name,))
    result = cursor.fetchall()
    if not result:
        return {"times": []}
    return {"times": [row[0] for row in result]}

# Endpoint to get nationality of constructors based on race ID and points
@app.get("/v1/formula_1/constructor_nationality_by_race_points", operation_id="get_constructor_nationality_by_race_points", summary="Retrieves the nationality of constructors who participated in a specific race and earned a certain number of points. The race is identified by its unique ID, and the points are the total points earned by the constructor in that race.")
async def get_constructor_nationality_by_race_points(race_id: int = Query(..., description="Race ID"), points: int = Query(..., description="Points")):
    cursor.execute("SELECT T2.nationality FROM constructorResults AS T1 INNER JOIN constructors AS T2 ON T2.constructorId = T1.constructorId WHERE T1.raceId = ? AND T1.points = ?", (race_id, points))
    result = cursor.fetchall()
    if not result:
        return {"nationalities": []}
    return {"nationalities": [row[0] for row in result]}

# Endpoint to get qualifying time (q1) for a specific driver in a specific race
@app.get("/v1/formula_1/qualifying_time_q1_by_driver_race", operation_id="get_qualifying_time_q1_by_driver_race", summary="Retrieves the Q1 qualifying time of a specific driver in a given race. The operation requires the race ID, driver's forename, and surname as input parameters to accurately identify the driver and the race. The result is the Q1 qualifying time for the specified driver in the specified race.")
async def get_qualifying_time_q1_by_driver_race(race_id: int = Query(..., description="Race ID"), forename: str = Query(..., description="Driver's forename"), surname: str = Query(..., description="Driver's surname")):
    cursor.execute("SELECT T1.q1 FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = ? AND T2.forename = ? AND T2.surname = ?", (race_id, forename, surname))
    result = cursor.fetchone()
    if not result:
        return {"q1": []}
    return {"q1": result[0]}

# Endpoint to get distinct nationalities of drivers based on qualifying time (q2) in a specific race
@app.get("/v1/formula_1/driver_nationalities_by_qualifying_time_q2", operation_id="get_driver_nationalities_by_qualifying_time_q2", summary="Retrieves the unique nationalities of drivers who have achieved a specific qualifying time (q2) in a given race. The input parameters include the race ID and the qualifying time (q2) in the format '1:40%'.")
async def get_driver_nationalities_by_qualifying_time_q2(race_id: int = Query(..., description="Race ID"), q2_time: str = Query(..., description="Qualifying time (q2) in the format '1:40%'")):
    cursor.execute("SELECT DISTINCT T2.nationality FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = ? AND T1.q2 LIKE ?", (race_id, q2_time))
    result = cursor.fetchall()
    if not result:
        return {"nationalities": []}
    return {"nationalities": [row[0] for row in result]}

# Endpoint to get driver numbers based on qualifying time (q3) in a specific race
@app.get("/v1/formula_1/driver_numbers_by_qualifying_time_q3", operation_id="get_driver_numbers_by_qualifying_time_q3", summary="Retrieves the driver numbers who participated in a specific race, based on their qualifying time (q3). The qualifying time is provided in a specific format and is used to filter the results.")
async def get_driver_numbers_by_qualifying_time_q3(race_id: int = Query(..., description="Race ID"), q3_time: str = Query(..., description="Qualifying time (q3) in the format '1:54%'")):
    cursor.execute("SELECT T2.number FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = ? AND T1.q3 LIKE ?", (race_id, q3_time))
    result = cursor.fetchall()
    if not result:
        return {"numbers": []}
    return {"numbers": [row[0] for row in result]}

# Endpoint to get the count of drivers with no time recorded in a specific race and year
@app.get("/v1/formula_1/count_drivers_no_time_recorded", operation_id="get_count_drivers_no_time_recorded", summary="Retrieves the number of drivers who did not record a time in a specific race and year. The operation filters the results based on the provided year and race name.")
async def get_count_drivers_no_time_recorded(year: int = Query(..., description="Year of the race"), race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT COUNT(T3.driverId) FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T1.year = ? AND T1.name = ? AND T2.time IS NULL", (year, race_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the URL of a season based on race ID
@app.get("/v1/formula_1/season_url_by_race_id", operation_id="get_season_url_by_race_id", summary="Retrieves the URL of the season associated with the specified race ID. This operation identifies the year of the race using the provided race ID and then fetches the URL of the corresponding season from the seasons table.")
async def get_season_url_by_race_id(race_id: int = Query(..., description="Race ID")):
    cursor.execute("SELECT T2.url FROM races AS T1 INNER JOIN seasons AS T2 ON T2.year = T1.year WHERE T1.raceId = ?", (race_id,))
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get the count of drivers with recorded time on a specific race date
@app.get("/v1/formula_1/count_drivers_with_time_by_date", operation_id="get_count_drivers_with_time_by_date", summary="Retrieves the total number of Formula 1 drivers who have recorded a time on a specific race date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_count_drivers_with_time_by_date(date: str = Query(..., description="Race date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T2.driverId) FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId WHERE T1.date = ? AND T2.time IS NOT NULL", (date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the youngest driver with recorded time in a specific race
@app.get("/v1/formula_1/youngest_driver_with_time_by_race", operation_id="get_youngest_driver_with_time_by_race", summary="Retrieves the name of the youngest driver who has a recorded time in a specific race. The operation filters out drivers with no time or birthdate recorded and returns the youngest driver based on their date of birth.")
async def get_youngest_driver_with_time_by_race(race_id: int = Query(..., description="Race ID")):
    cursor.execute("SELECT T1.forename, T1.surname FROM drivers AS T1 INNER JOIN results AS T2 ON T2.driverId = T1.driverId WHERE T2.raceId = ? AND T2.time IS NOT NULL AND T1.dob IS NOT NULL ORDER BY T1.dob ASC LIMIT 1", (race_id,))
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"forename": result[0], "surname": result[1]}}

# Endpoint to get distinct driver details based on race ID and lap time pattern
@app.get("/v1/formula_1/driver_details_by_race_and_time", operation_id="get_driver_details", summary="Retrieves unique driver details, including their forename, surname, and URL, based on a specific race ID and a lap time pattern. The lap time pattern is used to filter the results, ensuring that only drivers with lap times matching the provided pattern are returned.")
async def get_driver_details(race_id: int = Query(..., description="Race ID"), time_pattern: str = Query(..., description="Lap time pattern (e.g., '1:27%')")):
    cursor.execute("SELECT DISTINCT T2.forename, T2.surname, T2.url FROM lapTimes AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = ? AND T1.time LIKE ?", (race_id, time_pattern))
    result = cursor.fetchall()
    if not result:
        return {"driver_details": []}
    return {"driver_details": result}

# Endpoint to get the nationality of the driver with the fastest lap speed in a given race
@app.get("/v1/formula_1/fastest_lap_nationality", operation_id="get_fastest_lap_nationality", summary="Retrieves the nationality of the driver who achieved the fastest lap speed in a specific race. The race is identified by its unique ID. The operation considers only laps with a recorded time and ranks them by speed to determine the fastest lap.")
async def get_fastest_lap_nationality(race_id: int = Query(..., description="Race ID")):
    cursor.execute("SELECT T1.nationality FROM drivers AS T1 INNER JOIN results AS T2 ON T2.driverId = T1.driverId WHERE T2.raceId = ? AND T2.fastestLapTime IS NOT NULL ORDER BY T2.fastestLapSpeed DESC LIMIT 1", (race_id,))
    result = cursor.fetchone()
    if not result:
        return {"nationality": []}
    return {"nationality": result[0]}

# Endpoint to get the constructor URL with the highest points in a given race
@app.get("/v1/formula_1/top_constructor_url_by_race", operation_id="get_top_constructor_url", summary="Retrieves the URL of the constructor with the highest points in a specific race. The operation uses the provided race ID to identify the race and returns the URL of the constructor with the most points in that race.")
async def get_top_constructor_url(race_id: int = Query(..., description="Race ID")):
    cursor.execute("SELECT T2.url FROM constructorResults AS T1 INNER JOIN constructors AS T2 ON T2.constructorId = T1.constructorId WHERE T1.raceId = ? ORDER BY T1.points DESC LIMIT 1", (race_id,))
    result = cursor.fetchone()
    if not result:
        return {"constructor_url": []}
    return {"constructor_url": result[0]}

# Endpoint to get driver codes based on race ID and Q3 time pattern
@app.get("/v1/formula_1/driver_codes_by_race_and_q3", operation_id="get_driver_codes", summary="Retrieves the unique codes of drivers who participated in a specific race and whose Q3 time matches a given pattern. The race is identified by its unique ID, and the Q3 time pattern is a string that follows the format '1:33%'.")
async def get_driver_codes(race_id: int = Query(..., description="Race ID"), q3_pattern: str = Query(..., description="Q3 time pattern (e.g., '1:33%')")):
    cursor.execute("SELECT T2.code FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = ? AND T1.q3 LIKE ?", (race_id, q3_pattern))
    result = cursor.fetchall()
    if not result:
        return {"driver_codes": []}
    return {"driver_codes": result}

# Endpoint to get the race time for a specific driver in a given race
@app.get("/v1/formula_1/race_time_by_driver_and_race", operation_id="get_race_time", summary="Retrieves the race time of a specific driver in a given race. The operation requires the race ID, driver's forename, and surname as input parameters. It returns the time taken by the driver to complete the race.")
async def get_race_time(race_id: int = Query(..., description="Race ID"), forename: str = Query(..., description="Driver's forename"), surname: str = Query(..., description="Driver's surname")):
    cursor.execute("SELECT T2.time FROM drivers AS T1 INNER JOIN results AS T2 ON T2.driverId = T1.driverId WHERE T2.raceId = ? AND T1.forename = ? AND T1.surname = ?", (race_id, forename, surname))
    result = cursor.fetchone()
    if not result:
        return {"race_time": []}
    return {"race_time": result[0]}

# Endpoint to get driver details based on race year, name, and position
@app.get("/v1/formula_1/driver_details_by_race_year_name_position", operation_id="get_driver_details_by_race", summary="Retrieves the first and last names of the driver who finished in a specific position during a race in a given year. The race is identified by its name. The position is the driver's finishing rank in the race.")
async def get_driver_details_by_race(year: int = Query(..., description="Race year"), race_name: str = Query(..., description="Race name"), position: int = Query(..., description="Driver's position")):
    cursor.execute("SELECT T3.forename, T3.surname FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T1.year = ? AND T1.name = ? AND T2.position = ?", (year, race_name, position))
    result = cursor.fetchall()
    if not result:
        return {"driver_details": []}
    return {"driver_details": result}

# Endpoint to get the count of drivers with no race time on a specific date
@app.get("/v1/formula_1/count_drivers_no_time_by_date", operation_id="get_count_drivers_no_time", summary="Retrieves the total number of Formula 1 drivers who did not record a race time on a specified date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_count_drivers_no_time(date: str = Query(..., description="Race date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T2.driverId) FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId WHERE T1.date = ? AND T2.time IS NULL", (date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the forename and surname of the youngest driver with a recorded time in a specific race
@app.get("/v1/formula_1/youngest_driver_with_time_in_race", operation_id="get_youngest_driver_with_time", summary="Retrieves the full name of the youngest driver who has a recorded time in a specific race. The race is identified by its unique ID. The driver's age is determined by their date of birth (dob). Only drivers with a recorded time in the specified race are considered. The result is sorted in descending order by the driver's age, and the youngest driver is returned.")
async def get_youngest_driver_with_time(race_id: int = Query(..., description="ID of the race")):
    cursor.execute("SELECT T1.forename, T1.surname FROM drivers AS T1 INNER JOIN results AS T2 ON T2.driverId = T1.driverId WHERE T2.raceId = ? AND T2.time IS NOT NULL ORDER BY T1.dob DESC LIMIT 1", (race_id,))
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"forename": result[0], "surname": result[1]}}

# Endpoint to get the forename and surname of the driver with the fastest lap time in a specific race
@app.get("/v1/formula_1/fastest_lap_driver_in_race", operation_id="get_fastest_lap_driver", summary="Retrieves the full name of the driver who achieved the fastest lap time in a given race. The race is identified by its unique ID.")
async def get_fastest_lap_driver(race_id: int = Query(..., description="ID of the race")):
    cursor.execute("SELECT T2.forename, T2.surname FROM lapTimes AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = ? ORDER BY T1.time ASC LIMIT 1", (race_id,))
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"forename": result[0], "surname": result[1]}}

# Endpoint to get the nationality of the driver with the fastest lap speed
@app.get("/v1/formula_1/fastest_lap_speed_nationality", operation_id="get_fastest_lap_speed_nationality", summary="Retrieves the nationality of the driver who achieved the fastest lap speed in a race. This operation fetches the driver's nationality by joining the drivers and results tables, ordering the results by fastest lap speed in descending order, and returning the top result.")
async def get_fastest_lap_speed_nationality():
    cursor.execute("SELECT T1.nationality FROM drivers AS T1 INNER JOIN results AS T2 ON T2.driverId = T1.driverId ORDER BY T2.fastestLapSpeed DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"nationality": []}
    return {"nationality": result[0]}

# Endpoint to get the percentage difference in fastest lap speeds between two races for a specific driver
@app.get("/v1/formula_1/fastest_lap_speed_percentage_difference", operation_id="get_fastest_lap_speed_percentage_difference", summary="Retrieve the percentage difference in fastest lap speeds between two specified races for a given driver. The operation calculates the difference in the driver's fastest lap speeds for the provided races and returns the result as a percentage of the fastest lap speed in the first race.")
async def get_fastest_lap_speed_percentage_difference(race_id_1: int = Query(..., description="ID of the first race"), race_id_2: int = Query(..., description="ID of the second race"), forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT (SUM(IIF(T2.raceId = ?, T2.fastestLapSpeed, 0)) - SUM(IIF(T2.raceId = ?, T2.fastestLapSpeed, 0))) * 100 / SUM(IIF(T2.raceId = ?, T2.fastestLapSpeed, 0)) FROM drivers AS T1 INNER JOIN results AS T2 ON T2.driverId = T1.driverId WHERE T1.forename = ? AND T1.surname = ?", (race_id_1, race_id_2, race_id_1, forename, surname))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the percentage of drivers with recorded times in a specific race
@app.get("/v1/formula_1/percentage_drivers_with_times", operation_id="get_percentage_drivers_with_times", summary="Retrieves the percentage of drivers who have recorded times in a specific race. The calculation is based on the provided race date. The result is a numerical value representing the percentage of drivers with recorded times out of the total number of drivers participating in the race.")
async def get_percentage_drivers_with_times(date: str = Query(..., description="Date of the race in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.time IS NOT NULL THEN T2.driverId END) AS REAL) * 100 / COUNT(T2.driverId) FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId WHERE T1.date = ?", (date,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the first year a specific race was held
@app.get("/v1/formula_1/first_year_of_race", operation_id="get_first_year_of_race", summary="Retrieves the inaugural year of a specified race. The operation filters the races by the provided name and returns the earliest year in which the race was held.")
async def get_first_year_of_race(name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT year FROM races WHERE name = ? ORDER BY year ASC LIMIT 1", (name,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the names of races held in a specific year
@app.get("/v1/formula_1/race_names_in_year", operation_id="get_race_names_in_year", summary="Retrieves the names of all Formula 1 races held in a specified year, sorted in descending alphabetical order.")
async def get_race_names_in_year(year: int = Query(..., description="Year of the races")):
    cursor.execute("SELECT name FROM races WHERE year = ? ORDER BY name DESC", (year,))
    result = cursor.fetchall()
    if not result:
        return {"races": []}
    return {"races": [row[0] for row in result]}

# Endpoint to get the name of the first race held in the database
@app.get("/v1/formula_1/first_race_name", operation_id="get_first_race_name", summary="Get the name of the first race held in the database")
async def get_first_race_name():
    cursor.execute("SELECT name FROM races WHERE STRFTIME('%Y', date) = ( SELECT STRFTIME('%Y', date) FROM races ORDER BY date ASC LIMIT 1 ) AND STRFTIME('%m', date) = ( SELECT STRFTIME('%m', date) FROM races ORDER BY date ASC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"race_name": []}
    return {"race_name": result[0]}

# Endpoint to get the name and date of the last race in a specific year
@app.get("/v1/formula_1/last_race_in_year", operation_id="get_last_race_in_year", summary="Retrieves the name and date of the final race held in a specified year. The operation filters the races based on the provided year and returns the details of the race with the highest round number, indicating the last race of the year.")
async def get_last_race_in_year(year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT name, date FROM races WHERE year = ? ORDER BY round DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"race": []}
    return {"race": {"name": result[0], "date": result[1]}}

# Endpoint to get the year with the most races
@app.get("/v1/formula_1/year_with_most_races", operation_id="get_year_with_most_races", summary="Retrieves the year in which the most Formula 1 races were held. The operation calculates the total number of races per year and returns the year with the highest count.")
async def get_year_with_most_races():
    cursor.execute("SELECT year FROM races GROUP BY year ORDER BY COUNT(round) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get race names from a specific year that were not present in another year
@app.get("/v1/formula_1/race_names_excluding_year", operation_id="get_race_names_excluding_year", summary="Retrieves the names of Formula 1 races held in a specific year, excluding those that were also held in another specified year. This operation is useful for identifying unique races in a given year.")
async def get_race_names_excluding_year(year: int = Query(..., description="Year to get race names from"), exclude_year: int = Query(..., description="Year to exclude race names from")):
    cursor.execute("SELECT name FROM races WHERE year = ? AND name NOT IN ( SELECT name FROM races WHERE year = ? )", (year, exclude_year))
    result = cursor.fetchall()
    if not result:
        return {"race_names": []}
    return {"race_names": [row[0] for row in result]}

# Endpoint to get the country and location of the first race with a specific name
@app.get("/v1/formula_1/first_race_country_location", operation_id="get_first_race_country_location", summary="Retrieves the country and location of the earliest race with the specified name. The operation filters races by name and sorts them chronologically to identify the first race. It then extracts the country and location details from the corresponding circuit record.")
async def get_first_race_country_location(race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT T1.country, T1.location FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T2.name = ? ORDER BY T2.year ASC LIMIT 1", (race_name,))
    result = cursor.fetchone()
    if not result:
        return {"country": [], "location": []}
    return {"country": result[0], "location": result[1]}

# Endpoint to get the date of the most recent race at a specific circuit with a specific name
@app.get("/v1/formula_1/most_recent_race_date", operation_id="get_most_recent_race_date", summary="Retrieves the date of the most recent race that took place at a specified circuit and was named as provided. The operation uses the circuit and race names to identify the relevant data and returns the date of the most recent race that matches the criteria.")
async def get_most_recent_race_date(circuit_name: str = Query(..., description="Name of the circuit"), race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT T2.date FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.name = ? AND T2.name = ? ORDER BY T2.year DESC LIMIT 1", (circuit_name, race_name))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the count of races at a specific circuit with a specific name
@app.get("/v1/formula_1/race_count_at_circuit", operation_id="get_race_count_at_circuit", summary="Retrieves the total number of occurrences of a specific race at a given circuit. The operation requires the names of both the circuit and the race as input parameters.")
async def get_race_count_at_circuit(circuit_name: str = Query(..., description="Name of the circuit"), race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT COUNT(T2.circuitid) FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.name = ? AND T2.name = ?", (circuit_name, race_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of drivers who participated in a specific race in a specific year, ordered by their position
@app.get("/v1/formula_1/driver_names_by_race_year", operation_id="get_driver_names_by_race_year", summary="Retrieve a list of driver names who competed in a specified race during a given year, sorted by their finishing positions. The endpoint requires the race name and year as input parameters.")
async def get_driver_names_by_race_year(race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT T3.forename, T3.surname FROM races AS T1 INNER JOIN driverStandings AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T1.name = ? AND T1.year = ? ORDER BY T2.position ASC", (race_name, year))
    result = cursor.fetchall()
    if not result:
        return {"drivers": []}
    return {"drivers": [{"forename": row[0], "surname": row[1]} for row in result]}

# Endpoint to get the driver with the highest points in any race
@app.get("/v1/formula_1/top_driver_by_points", operation_id="get_top_driver_by_points", summary="Retrieves the driver with the highest accumulated points across all races. This operation returns the full name of the top-performing driver and their total points, providing a snapshot of the current leader in the Formula 1 standings.")
async def get_top_driver_by_points():
    cursor.execute("SELECT T3.forename, T3.surname, T2.points FROM races AS T1 INNER JOIN driverStandings AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId ORDER BY T2.points DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"forename": result[0], "surname": result[1], "points": result[2]}}

# Endpoint to get the top 3 drivers by points in a specific race in a specific year
@app.get("/v1/formula_1/top_drivers_by_race_year", operation_id="get_top_drivers_by_race_year", summary="Retrieves the top three drivers with the highest points in a specific race during a given year. The operation requires the name of the race and the year as input parameters to filter the results.")
async def get_top_drivers_by_race_year(race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT T3.forename, T3.surname, T2.points FROM races AS T1 INNER JOIN driverStandings AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T1.name = ? AND T1.year = ? ORDER BY T2.points DESC LIMIT 3", (race_name, year))
    result = cursor.fetchall()
    if not result:
        return {"drivers": []}
    return {"drivers": [{"forename": row[0], "surname": row[1], "points": row[2]} for row in result]}

# Endpoint to get the fastest lap time and the corresponding driver and race name
@app.get("/v1/formula_1/fastest_lap_time", operation_id="get_fastest_lap_time", summary="Retrieves the fastest lap time recorded in a race, along with the name of the driver who achieved it and the race's name. The data is sourced from the drivers, lapTimes, and races tables, with the lap time being the primary sorting criterion.")
async def get_fastest_lap_time():
    cursor.execute("SELECT T2.milliseconds, T1.forename, T1.surname, T3.name FROM drivers AS T1 INNER JOIN lapTimes AS T2 ON T1.driverId = T2.driverId INNER JOIN races AS T3 ON T2.raceId = T3.raceId ORDER BY T2.milliseconds ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"lap_time": []}
    return {"lap_time": {"milliseconds": result[0], "forename": result[1], "surname": result[2], "race_name": result[3]}}

# Endpoint to get the average lap time for a specific driver in a specific race in a specific year
@app.get("/v1/formula_1/average_lap_time_driver_race_year", operation_id="get_average_lap_time_driver_race_year", summary="Retrieves the average lap time for a particular driver in a specific race during a given year. The operation requires the driver's first and last names, the year of the race, and the race's name to accurately calculate and return the average lap time.")
async def get_average_lap_time_driver_race_year(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver"), year: int = Query(..., description="Year of the race"), race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT AVG(T2.milliseconds) FROM races AS T1 INNER JOIN lapTimes AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T3.forename = ? AND T3.surname = ? AND T1.year = ? AND T1.name = ?", (forename, surname, year, race_name))
    result = cursor.fetchone()
    if not result:
        return {"average_lap_time": []}
    return {"average_lap_time": result[0]}

# Endpoint to get the percentage of races where a specific driver did not finish first from a specific year onwards
@app.get("/v1/formula_1/percentage_non_first_finishes", operation_id="get_percentage_non_first_finishes", summary="Retrieve the percentage of races in which a specific driver, identified by their surname, did not finish in the first position, starting from a given year. The calculation is based on the total number of races the driver participated in from the specified year onwards.")
async def get_percentage_non_first_finishes(surname: str = Query(..., description="Surname of the driver"), year: int = Query(..., description="Starting year")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.position <> 1 THEN T2.position END) AS REAL) * 100 / COUNT(T2.driverStandingsId) FROM races AS T1 INNER JOIN driverStandings AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T3.surname = ? AND T1.year >= ?", (surname, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get driver details with the highest points and at least a specified number of wins
@app.get("/v1/formula_1/driver_details_highest_points", operation_id="get_driver_details_highest_points", summary="Retrieves the driver details of the individual(s) with the highest points, who have won at least a specified number of races. The results can be limited to a certain number of top performers.")
async def get_driver_details_highest_points(min_wins: int = Query(..., description="Minimum number of wins"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.forename, T1.surname, T1.nationality, MAX(T2.points) FROM drivers AS T1 INNER JOIN driverStandings AS T2 ON T2.driverId = T1.driverId WHERE T2.wins >= ? GROUP BY T1.forename, T1.surname, T1.nationality ORDER BY COUNT(T2.wins) DESC LIMIT ?", (min_wins, limit))
    result = cursor.fetchall()
    if not result:
        return {"driver_details": []}
    return {"driver_details": result}

# Endpoint to get the age, forename, and surname of the youngest driver from a specified nationality
@app.get("/v1/formula_1/youngest_driver_by_nationality", operation_id="get_youngest_driver_by_nationality", summary="Retrieve the age, first name, and last name of the youngest driver from a specified nationality, with the option to limit the number of results. The age is calculated based on the current year and the driver's year of birth.")
async def get_youngest_driver_by_nationality(nationality: str = Query(..., description="Nationality of the driver"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', dob), forename , surname FROM drivers WHERE nationality = ? ORDER BY dob DESC LIMIT ?", (nationality, limit))
    result = cursor.fetchall()
    if not result:
        return {"driver_details": []}
    return {"driver_details": result}

# Endpoint to get distinct circuit names with a specified number of races within a given year range
@app.get("/v1/formula_1/distinct_circuit_names_by_year_range", operation_id="get_distinct_circuit_names_by_year_range", summary="Retrieve a list of unique circuit names that have hosted a specific number of races within a given year range. The operation filters circuits based on the start and end year, and the number of races held at each circuit. The result is a distinct set of circuit names that meet the specified criteria.")
async def get_distinct_circuit_names_by_year_range(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), race_count: int = Query(..., description="Number of races")):
    cursor.execute("SELECT DISTINCT T1.name FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE STRFTIME('%Y', T2.date) BETWEEN ? AND ? GROUP BY T1.name HAVING COUNT(T2.raceId) = ?", (start_year, end_year, race_count))
    result = cursor.fetchall()
    if not result:
        return {"circuit_names": []}
    return {"circuit_names": result}

# Endpoint to get circuit details and race names for a specified country and year
@app.get("/v1/formula_1/circuit_details_by_country_year", operation_id="get_circuit_details_by_country_year", summary="Retrieves the name, location, and associated race names of circuits located in a specified country and year. This operation is useful for obtaining detailed information about Formula 1 circuits and their corresponding races based on the provided country and year.")
async def get_circuit_details_by_country_year(country: str = Query(..., description="Country of the circuit"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT T1.name, T1.location, T2.name FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.country = ? AND T2.year = ?", (country, year))
    result = cursor.fetchall()
    if not result:
        return {"circuit_details": []}
    return {"circuit_details": result}

# Endpoint to get distinct race names, circuit names, and locations for a specified year and month
@app.get("/v1/formula_1/race_circuit_details_by_year_month", operation_id="get_race_circuit_details_by_year_month", summary="Retrieve unique race names, circuit names, and locations for a specific year and month. The operation filters races based on the provided year and month, ensuring that only distinct combinations of race, circuit, and location are returned.")
async def get_race_circuit_details_by_year_month(year: int = Query(..., description="Year of the race"), month: str = Query(..., description="Month of the race in 'MM' format")):
    cursor.execute("SELECT DISTINCT T2.name, T1.name, T1.location FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T2.year = ? AND STRFTIME('%m', T2.date) = ?", (year, month))
    result = cursor.fetchall()
    if not result:
        return {"race_circuit_details": []}
    return {"race_circuit_details": result}

# Endpoint to get race names where a specified driver finished in a position less than a given value
@app.get("/v1/formula_1/race_names_by_driver_position", operation_id="get_race_names_by_driver_position", summary="Retrieve the names of races in which a specific driver finished in a position below the provided maximum value. The operation requires the driver's first and last names, as well as the maximum position to filter the results.")
async def get_race_names_by_driver_position(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver"), max_position: int = Query(..., description="Maximum position of the driver")):
    cursor.execute("SELECT T1.name FROM races AS T1 INNER JOIN driverStandings AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T3.forename = ? AND T3.surname = ? AND T2.position < ?", (forename, surname, max_position))
    result = cursor.fetchall()
    if not result:
        return {"race_names": []}
    return {"race_names": result}

# Endpoint to get the total number of wins for a specified driver at a specified circuit
@app.get("/v1/formula_1/total_wins_by_driver_circuit", operation_id="get_total_wins_by_driver_circuit", summary="Retrieves the total number of wins for a specific driver at a particular circuit. The operation requires the driver's first and last name, as well as the name of the circuit, to accurately calculate the total wins.")
async def get_total_wins_by_driver_circuit(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver"), circuit_name: str = Query(..., description="Name of the circuit")):
    cursor.execute("SELECT SUM(T2.wins) FROM drivers AS T1 INNER JOIN driverStandings AS T2 ON T2.driverId = T1.driverId INNER JOIN races AS T3 ON T3.raceId = T2.raceId INNER JOIN circuits AS T4 ON T4.circuitId = T3.circuitId WHERE T1.forename = ? AND T1.surname = ? AND T4.name = ?", (forename, surname, circuit_name))
    result = cursor.fetchone()
    if not result:
        return {"total_wins": []}
    return {"total_wins": result[0]}

# Endpoint to get the race name and year of the fastest lap for a specified driver
@app.get("/v1/formula_1/fastest_lap_by_driver", operation_id="get_fastest_lap_by_driver", summary="Retrieves the race name and year of the fastest lap(s) achieved by a specific driver, sorted in ascending order of lap time. The number of results can be limited by specifying a value for the limit parameter.")
async def get_fastest_lap_by_driver(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.name, T1.year FROM races AS T1 INNER JOIN lapTimes AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T3.forename = ? AND T3.surname = ? ORDER BY T2.milliseconds ASC LIMIT ?", (forename, surname, limit))
    result = cursor.fetchall()
    if not result:
        return {"race_details": []}
    return {"race_details": result}

# Endpoint to get the average points for a specified driver in a specified year
@app.get("/v1/formula_1/average_points_by_driver_year", operation_id="get_average_points_by_driver_year", summary="Retrieves the average points scored by a specific driver in a given year. This operation calculates the average based on the driver's standings in each race of the specified year. The driver is identified by their forename and surname.")
async def get_average_points_by_driver_year(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT AVG(T2.points) FROM drivers AS T1 INNER JOIN driverStandings AS T2 ON T2.driverId = T1.driverId INNER JOIN races AS T3 ON T3.raceId = T2.raceId WHERE T1.forename = ? AND T1.surname = ? AND T3.year = ?", (forename, surname, year))
    result = cursor.fetchone()
    if not result:
        return {"average_points": []}
    return {"average_points": result[0]}

# Endpoint to get the race name and points for the earliest race of a specified driver
@app.get("/v1/formula_1/earliest_race_by_driver", operation_id="get_earliest_race_by_driver", summary="Retrieves the name and points of the earliest race in which a specified driver participated, based on the provided forename and surname. The number of results can be limited by the 'limit' parameter.")
async def get_earliest_race_by_driver(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.name, T2.points FROM races AS T1 INNER JOIN driverStandings AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T3.forename = ? AND T3.surname = ? ORDER BY T1.year ASC LIMIT ?", (forename, surname, limit))
    result = cursor.fetchall()
    if not result:
        return {"race_details": []}
    return {"race_details": result}

# Endpoint to get distinct circuit names and countries for a given year
@app.get("/v1/formula_1/distinct_circuits_by_year", operation_id="get_distinct_circuits_by_year", summary="Retrieves a list of unique circuit names and their respective countries, sorted by date, for a specified year. The year is provided as an input parameter.")
async def get_distinct_circuits_by_year(year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT DISTINCT T2.name, T1.country FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T2.year = ? ORDER BY T2.date ASC", (year,))
    result = cursor.fetchall()
    if not result:
        return {"circuits": []}
    return {"circuits": result}

# Endpoint to get the latest lap details with circuit location
@app.get("/v1/formula_1/latest_lap_details", operation_id="get_latest_lap_details", summary="Retrieves the most recent lap details, including the circuit location, from the latest race. The response includes the lap number, race year, and circuit location.")
async def get_latest_lap_details():
    cursor.execute("SELECT T3.lap, T2.name, T2.year, T1.location FROM circuits AS T1 INNER JOIN races AS T2 ON T1.circuitId = T2.circuitId INNER JOIN lapTimes AS T3 ON T3.raceId = T2.raceId ORDER BY T3.lap DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"lap_details": []}
    return {"lap_details": result}

# Endpoint to get the percentage of races held in a specific country for a specific race name
@app.get("/v1/formula_1/race_percentage_by_country", operation_id="get_race_percentage_by_country", summary="Retrieves the percentage of races held in a specified country for a given race name. This operation calculates the proportion of races held in the provided country for the specified race, by comparing the count of races in that country to the total count of races with the same name.")
async def get_race_percentage_by_country(country: str = Query(..., description="Country where the race was held"), race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.country = ? THEN T2.circuitID END) AS REAL) * 100 / COUNT(T2.circuitId) FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T2.name = ?", (country, race_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the circuit with the highest latitude among a list of circuit names
@app.get("/v1/formula_1/highest_latitude_circuit", operation_id="get_highest_latitude_circuit", summary="Retrieves the circuit with the highest latitude from a provided list of up to three circuit names. The circuit names are used to filter the results, and the circuit with the highest latitude is returned.")
async def get_highest_latitude_circuit(name1: str = Query(..., description="First circuit name"), name2: str = Query(..., description="Second circuit name"), name3: str = Query(..., description="Third circuit name")):
    cursor.execute("SELECT name FROM circuits WHERE name IN (?, ?, ?) ORDER BY lat DESC LIMIT 1", (name1, name2, name3))
    result = cursor.fetchone()
    if not result:
        return {"circuit": []}
    return {"circuit": result[0]}

# Endpoint to get the circuit reference for a specific circuit name
@app.get("/v1/formula_1/circuit_reference", operation_id="get_circuit_reference", summary="Retrieves the unique circuit reference associated with a specific circuit name. The circuit name is provided as an input parameter.")
async def get_circuit_reference(name: str = Query(..., description="Name of the circuit")):
    cursor.execute("SELECT circuitRef FROM circuits WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"circuit_ref": []}
    return {"circuit_ref": result[0]}

# Endpoint to get the country of the circuit with the highest altitude
@app.get("/v1/formula_1/highest_altitude_circuit_country", operation_id="get_highest_altitude_circuit_country", summary="Retrieves the country of the Formula 1 circuit with the highest altitude. The operation returns the country name of the circuit located at the highest altitude, based on the data available in the circuits table.")
async def get_highest_altitude_circuit_country():
    cursor.execute("SELECT country FROM circuits ORDER BY alt DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of drivers without a code
@app.get("/v1/formula_1/driver_count_without_code", operation_id="get_driver_count_without_code", summary="Retrieves the total number of drivers who do not have a code assigned to them. This operation calculates the difference between the total count of drivers and the count of drivers with a non-null code.")
async def get_driver_count_without_code():
    cursor.execute("SELECT COUNT(driverId) - COUNT(CASE WHEN code IS NOT NULL THEN code END) FROM drivers")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the nationality of the youngest driver
@app.get("/v1/formula_1/youngest_driver_nationality", operation_id="get_youngest_driver_nationality", summary="Retrieves the nationality of the youngest driver in the database. The driver's date of birth is used to determine their age, and the result is limited to a single record.")
async def get_youngest_driver_nationality():
    cursor.execute("SELECT nationality FROM drivers WHERE dob IS NOT NULL ORDER BY dob ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"nationality": []}
    return {"nationality": result[0]}

# Endpoint to get the surnames of drivers from a specific nationality
@app.get("/v1/formula_1/driver_surnames_by_nationality", operation_id="get_driver_surnames_by_nationality", summary="Retrieves the surnames of Formula 1 drivers who share a specified nationality. The nationality is provided as an input parameter.")
async def get_driver_surnames_by_nationality(nationality: str = Query(..., description="Nationality of the drivers")):
    cursor.execute("SELECT surname FROM drivers WHERE nationality = ?", (nationality,))
    result = cursor.fetchall()
    if not result:
        return {"surnames": []}
    return {"surnames": [row[0] for row in result]}

# Endpoint to get the URL of a driver based on forename and surname
@app.get("/v1/formula_1/driver_url", operation_id="get_driver_url", summary="Retrieves the URL associated with a specific Formula 1 driver using their forename and surname as identifiers.")
async def get_driver_url(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT url FROM drivers WHERE forename = ? AND surname = ?", (forename, surname))
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get the driver reference based on forename and surname
@app.get("/v1/formula_1/driver_reference", operation_id="get_driver_reference", summary="Retrieves the unique reference identifier for a Formula 1 driver using their first and last names. This operation requires both the driver's forename and surname as input parameters to accurately locate the driver in the database and return their reference identifier.")
async def get_driver_reference(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT driverRef FROM drivers WHERE forename = ? AND surname = ?", (forename, surname))
    result = cursor.fetchone()
    if not result:
        return {"driverRef": []}
    return {"driverRef": result[0]}

# Endpoint to get the circuit name based on race year and name
@app.get("/v1/formula_1/circuit_name_by_race", operation_id="get_circuit_name_by_race", summary="Retrieves the name of the circuit where a specific race was held, based on the provided race year and name. The operation uses the race year and name to identify the circuit and returns its name.")
async def get_circuit_name_by_race(year: int = Query(..., description="Year of the race"), name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT T1.name FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T2.year = ? AND T2.name = ?", (year, name))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get distinct years of races held at a specific circuit
@app.get("/v1/formula_1/race_years_by_circuit", operation_id="get_race_years_by_circuit", summary="Retrieves a list of unique years in which races were held at a specified circuit. The circuit is identified by its name.")
async def get_race_years_by_circuit(name: str = Query(..., description="Name of the circuit")):
    cursor.execute("SELECT DISTINCT T2.year FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"years": []}
    return {"years": [row[0] for row in result]}

# Endpoint to get the date and time of a race based on year and name
@app.get("/v1/formula_1/race_date_time", operation_id="get_race_date_time", summary="Retrieves the date and time of a specific Formula 1 race based on the provided year and race name. The operation fetches this information from the races table, which is linked to the circuits table via the circuitID field. The year and race name parameters are used to filter the results.")
async def get_race_date_time(year: int = Query(..., description="Year of the race"), name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT T2.date, T2.time FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T2.year = ? AND T2.name = ?", (year, name))
    result = cursor.fetchone()
    if not result:
        return {"date": [], "time": []}
    return {"date": result[0], "time": result[1]}

# Endpoint to get the count of races held in a specific country
@app.get("/v1/formula_1/race_count_by_country", operation_id="get_race_count_by_country", summary="Retrieves the total number of Formula 1 races held in a specified country. The operation filters the races based on the provided country and returns the count.")
async def get_race_count_by_country(country: str = Query(..., description="Country where the races were held")):
    cursor.execute("SELECT COUNT(T2.circuitId) FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the dates of races held at a specific circuit
@app.get("/v1/formula_1/race_dates_by_circuit", operation_id="get_race_dates_by_circuit", summary="Retrieves the dates of all Formula 1 races held at a specified circuit. The circuit is identified by its name, which is provided as an input parameter.")
async def get_race_dates_by_circuit(name: str = Query(..., description="Name of the circuit")):
    cursor.execute("SELECT T2.date FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": [row[0] for row in result]}

# Endpoint to get the URL of a circuit based on race year and name
@app.get("/v1/formula_1/circuit_url_by_race", operation_id="get_circuit_url_by_race", summary="Retrieves the URL of a specific circuit based on the provided race year and name. This operation fetches the circuit URL by joining the circuits and races tables using the circuitID field, and filtering the results based on the input race year and name.")
async def get_circuit_url_by_race(year: int = Query(..., description="Year of the race"), name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT T1.url FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T2.year = ? AND T2.name = ?", (year, name))
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get the race names where a specific driver won
@app.get("/v1/formula_1/race_names_by_driver_win", operation_id="get_race_names_by_win", summary="Retrieves the names of races where a driver, identified by their forename and surname, achieved a specific rank. The rank parameter determines the position at which the driver finished the race.")
async def get_race_names_by_win(rank: int = Query(..., description="Rank of the driver"), forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT name FROM races WHERE raceId IN ( SELECT raceId FROM results WHERE rank = ? AND driverId = ( SELECT driverId FROM drivers WHERE forename = ? AND surname = ? ) )", (rank, forename, surname))
    results = cursor.fetchall()
    if not results:
        return {"races": []}
    return {"races": [result[0] for result in results]}

# Endpoint to get the fastest lap speed for a specific race and year
@app.get("/v1/formula_1/fastest_lap_speed_by_race_year", operation_id="get_fastest_lap_speed", summary="Retrieves the fastest lap speed recorded in a specific race during a given year. The operation filters results by the provided race name and year, and returns the highest speed that is not null.")
async def get_fastest_lap_speed(race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT T2.fastestLapSpeed FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId WHERE T1.name = ? AND T1.year = ? AND T2.fastestLapSpeed IS NOT NULL ORDER BY T2.fastestLapSpeed DESC LIMIT 1", (race_name, year))
    result = cursor.fetchone()
    if not result:
        return {"fastest_lap_speed": []}
    return {"fastest_lap_speed": result[0]}

# Endpoint to get the distinct years a specific driver participated in
@app.get("/v1/formula_1/driver_participation_years", operation_id="get_driver_participation_years", summary="Retrieve the unique years in which a specific driver has participated in races. This operation requires the driver's first and last names as input parameters to filter the results accordingly.")
async def get_driver_participation_years(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT DISTINCT T1.year FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T3.forename = ? AND T3.surname = ?", (forename, surname))
    results = cursor.fetchall()
    if not results:
        return {"years": []}
    return {"years": [result[0] for result in results]}

# Endpoint to get the position order of a specific driver in a specific race and year
@app.get("/v1/formula_1/driver_position_order", operation_id="get_driver_position_order", summary="Retrieves the position order of a specific driver in a given race and year. The operation requires the driver's first and last names, the race name, and the year of the race as input parameters. It returns the position order of the driver in the specified race and year.")
async def get_driver_position_order(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver"), race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT T2.positionOrder FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T3.forename = ? AND T3.surname = ? AND T1.name = ? AND T1.year = ?", (forename, surname, race_name, year))
    result = cursor.fetchone()
    if not result:
        return {"position_order": []}
    return {"position_order": result[0]}

# Endpoint to get the driver details for a specific grid position in a specific race and year
@app.get("/v1/formula_1/driver_details_by_grid_position", operation_id="get_driver_details_by_grid", summary="Retrieve the forename and surname of the driver who started at a specific grid position in a given race and year. The operation requires the grid position, race name, and year as input parameters.")
async def get_driver_details_by_grid(grid: int = Query(..., description="Grid position"), race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT T3.forename, T3.surname FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T2.grid = ? AND T1.name = ? AND T1.year = ?", (grid, race_name, year))
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"forename": result[0], "surname": result[1]}}

# Endpoint to get the count of drivers with recorded time in a specific race and year
@app.get("/v1/formula_1/driver_count_with_time", operation_id="get_driver_count_with_time", summary="Retrieves the number of drivers who have recorded a time in a specific race and year. The operation filters the results based on the provided race name and year, ensuring that only drivers with a recorded time are counted.")
async def get_driver_count_with_time(race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT COUNT(T2.driverId) FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId WHERE T1.name = ? AND T1.year = ? AND T2.time IS NOT NULL", (race_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the time of a driver who finished in a specific rank in a specific race and year
@app.get("/v1/formula_1/driver_time_by_rank_race_year", operation_id="get_driver_time_by_rank_race_year", summary="Retrieve the time taken by a driver to complete a race, given their finishing rank, the race's name, and the year it was held. This operation fetches the time from the results table, which is joined with the races table using the raceId. The data returned corresponds to the driver who finished at the specified rank in the specified race and year.")
async def get_driver_time_by_rank_race_year(rank: int = Query(..., description="Rank of the driver"), race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT T1.time FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId WHERE T1.rank = ? AND T2.name = ? AND T2.year = ?", (rank, race_name, year))
    result = cursor.fetchall()
    if not result:
        return {"times": []}
    return {"times": [row[0] for row in result]}

# Endpoint to get driver details who participated in a specific race and year with a specific time format
@app.get("/v1/formula_1/driver_details_by_race_time_year", operation_id="get_driver_details_by_race_time_year", summary="Get driver details who participated in a specific race and year with a specific time format")
async def get_driver_details_by_race_time_year(race_name: str = Query(..., description="Name of the race"), time_format: str = Query(..., description="Time format (e.g., '_:%:__.___')"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT T1.forename, T1.surname, T1.url FROM drivers AS T1 INNER JOIN results AS T2 ON T1.driverId = T2.driverId INNER JOIN races AS T3 ON T3.raceId = T2.raceId WHERE T3.name = ? AND T2.time LIKE ? AND T3.year = ?", (race_name, time_format, year))
    result = cursor.fetchall()
    if not result:
        return {"drivers": []}
    return {"drivers": [{"forename": row[0], "surname": row[1], "url": row[2]} for row in result]}

# Endpoint to get the count of drivers of a specific nationality who participated in a specific race and year
@app.get("/v1/formula_1/driver_count_by_nationality_race_year", operation_id="get_driver_count_by_nationality_race_year", summary="Retrieve the number of drivers from a specified nationality who competed in a particular race during a given year.")
async def get_driver_count_by_nationality_race_year(race_name: str = Query(..., description="Name of the race"), nationality: str = Query(..., description="Nationality of the driver"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT COUNT(*) FROM drivers AS T1 INNER JOIN results AS T2 ON T1.driverId = T2.driverId INNER JOIN races AS T3 ON T3.raceId = T2.raceId WHERE T3.name = ? AND T1.nationality = ? AND T3.year = ?", (race_name, nationality, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of drivers who have completed at least one race in a specific race and year
@app.get("/v1/formula_1/driver_count_completed_race_year", operation_id="get_driver_count_completed_race_year", summary="Retrieves the number of drivers who have successfully finished at least one race in a specified race and year. The operation requires the name of the race and the year as input parameters to filter the results.")
async def get_driver_count_completed_race_year(race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T1.driverId FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId WHERE T2.name = ? AND T2.year = ? AND T1.time IS NOT NULL GROUP BY T1.driverId HAVING COUNT(T2.raceId) > 0 )", (race_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total points of a driver with a specific forename and surname
@app.get("/v1/formula_1/driver_total_points", operation_id="get_driver_total_points", summary="Retrieves the cumulative points earned by a Formula 1 driver, identified by their forename and surname. The operation calculates the total points by summing up the points from all the races in which the driver has participated.")
async def get_driver_total_points(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT SUM(T2.points) FROM drivers AS T1 INNER JOIN results AS T2 ON T1.driverId = T2.driverId WHERE T1.forename = ? AND T1.surname = ?", (forename, surname))
    result = cursor.fetchone()
    if not result:
        return {"total_points": []}
    return {"total_points": result[0]}

# Endpoint to get the average fastest lap time of a driver with a specific forename and surname
@app.get("/v1/formula_1/driver_avg_fastest_lap_time", operation_id="get_driver_avg_fastest_lap_time", summary="Retrieves the average fastest lap time for a Formula 1 driver identified by their forename and surname. The calculation considers all races in which the driver has participated.")
async def get_driver_avg_fastest_lap_time(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT AVG(CAST(SUBSTR(T2.fastestLapTime, 1, INSTR(T2.fastestLapTime, ':') - 1) AS INTEGER) * 60 + CAST(SUBSTR(T2.fastestLapTime, INSTR(T2.fastestLapTime, ':') + 1) AS REAL)) FROM drivers AS T1 INNER JOIN results AS T2 ON T1.driverId = T2.driverId WHERE T1.surname = ? AND T1.forename = ?", (surname, forename))
    result = cursor.fetchone()
    if not result:
        return {"avg_fastest_lap_time": []}
    return {"avg_fastest_lap_time": result[0]}

# Endpoint to get the percentage of drivers who completed a specific race in a specific year
@app.get("/v1/formula_1/completion_percentage_race_year", operation_id="get_completion_percentage_race_year", summary="Retrieves the completion percentage of a specific race in a given year. This operation calculates the ratio of drivers who finished the race to the total number of participants. The race is identified by its name and year.")
async def get_completion_percentage_race_year(race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.time IS NOT NULL, 1, 0)) AS REAL) * 100 / COUNT(T1.resultId) FROM results AS T1 INNER JOIN races AS T2 ON T1.raceId = T2.raceId WHERE T2.name = ? AND T2.year = ?", (race_name, year))
    result = cursor.fetchone()
    if not result:
        return {"completion_percentage": []}
    return {"completion_percentage": result[0]}

# Endpoint to get the percentage time difference between the champion and the last driver in a specific race and year
@app.get("/v1/formula_1/percentage_time_difference_champion_last_driver", operation_id="get_percentage_time_difference_champion_last_driver", summary="Retrieves the percentage difference in time between the champion and the last driver in a specific race and year. The calculation is based on the time in seconds for each driver, with the champion's time being the first position and the last driver's time being the last position in the race results. The input parameters are the name of the race and the year of the race.")
async def get_percentage_time_difference_champion_last_driver(race_name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race")):
    cursor.execute("WITH time_in_seconds AS ( SELECT T1.positionOrder, CASE WHEN T1.positionOrder = 1 THEN (CAST(SUBSTR(T1.time, 1, 1) AS REAL) * 3600) + (CAST(SUBSTR(T1.time, 3, 2) AS REAL) * 60) + CAST(SUBSTR(T1.time, 6) AS REAL) ELSE CAST(SUBSTR(T1.time, 2) AS REAL) END AS time_seconds FROM results AS T1 INNER JOIN races AS T2 ON T1.raceId = T2.raceId WHERE T2.name = ? AND T1.time IS NOT NULL AND T2.year = ? ), champion_time AS ( SELECT time_seconds FROM time_in_seconds WHERE positionOrder = 1), last_driver_incremental AS ( SELECT time_seconds FROM time_in_seconds WHERE positionOrder = (SELECT MAX(positionOrder) FROM time_in_seconds) ) SELECT (CAST((SELECT time_seconds FROM last_driver_incremental) AS REAL) * 100) / (SELECT time_seconds + (SELECT time_seconds FROM last_driver_incremental) FROM champion_time)", (race_name, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage_time_difference": []}
    return {"percentage_time_difference": result[0]}

# Endpoint to get the count of circuits in a specific location and country
@app.get("/v1/formula_1/circuit_count_by_location_country", operation_id="get_circuit_count_by_location_country", summary="Retrieves the total number of circuits located in a specified location within a given country. The response is based on the provided location and country parameters.")
async def get_circuit_count_by_location_country(location: str = Query(..., description="Location of the circuit"), country: str = Query(..., description="Country of the circuit")):
    cursor.execute("SELECT COUNT(circuitId) FROM circuits WHERE location = ? AND country = ?", (location, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latitude and longitude of circuits in a specific country
@app.get("/v1/formula_1/circuit_coordinates_by_country", operation_id="get_circuit_coordinates_by_country", summary="Retrieve the geographical coordinates of all circuits located in a specified country. The response includes the latitude and longitude of each circuit.")
async def get_circuit_coordinates_by_country(country: str = Query(..., description="Country of the circuit")):
    cursor.execute("SELECT lat, lng FROM circuits WHERE country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"coordinates": []}
    return {"coordinates": [{"lat": row[0], "lng": row[1]} for row in result]}

# Endpoint to get the count of drivers based on nationality and birth year
@app.get("/v1/formula_1/driver_count_by_nationality_and_birth_year", operation_id="get_driver_count", summary="Retrieves the number of Formula 1 drivers from a specified nationality who were born after a given year. The response is based on the provided nationality and birth year.")
async def get_driver_count(nationality: str = Query(..., description="Nationality of the driver"), birth_year: str = Query(..., description="Birth year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(driverId) FROM drivers WHERE nationality = ? AND STRFTIME('%Y', dob) > ?", (nationality, birth_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum points of constructors based on nationality
@app.get("/v1/formula_1/max_constructor_points_by_nationality", operation_id="get_max_constructor_points", summary="Retrieves the highest points total achieved by a constructor from a specified nationality. The nationality of the constructor is provided as an input parameter.")
async def get_max_constructor_points(nationality: str = Query(..., description="Nationality of the constructor")):
    cursor.execute("SELECT MAX(T1.points) FROM constructorStandings AS T1 INNER JOIN constructors AS T2 on T1.constructorId = T2.constructorId WHERE T2.nationality = ?", (nationality,))
    result = cursor.fetchone()
    if not result:
        return {"max_points": []}
    return {"max_points": result[0]}

# Endpoint to get the constructor with the highest points
@app.get("/v1/formula_1/top_constructor_by_points", operation_id="get_top_constructor", summary="Retrieves the name of the constructor with the highest total points in the Formula 1 standings.")
async def get_top_constructor():
    cursor.execute("SELECT T2.name FROM constructorStandings AS T1 INNER JOIN constructors AS T2 on T1.constructorId = T2.constructorId ORDER BY T1.points DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"constructor_name": []}
    return {"constructor_name": result[0]}

# Endpoint to get the constructor names based on points and race ID
@app.get("/v1/formula_1/constructor_names_by_points_and_race_id", operation_id="get_constructor_names", summary="Retrieves the names of constructors who have earned a specific number of points in a particular race. The points and race ID are provided as input parameters.")
async def get_constructor_names(points: int = Query(..., description="Points of the constructor"), race_id: int = Query(..., description="Race ID")):
    cursor.execute("SELECT T2.name FROM constructorStandings AS T1 INNER JOIN constructors AS T2 on T1.constructorId = T2.constructorId WHERE T1.points = ? AND T1.raceId = ?", (points, race_id))
    result = cursor.fetchall()
    if not result:
        return {"constructor_names": []}
    return {"constructor_names": [row[0] for row in result]}

# Endpoint to get distinct constructor names based on rank
@app.get("/v1/formula_1/distinct_constructor_names_by_rank", operation_id="get_distinct_constructor_names", summary="Retrieves a list of unique constructor names that have achieved a specific rank in the results. The rank is provided as an input parameter.")
async def get_distinct_constructor_names(rank: int = Query(..., description="Rank of the constructor")):
    cursor.execute("SELECT DISTINCT T2.name FROM results AS T1 INNER JOIN constructors AS T2 on T1.constructorId = T2.constructorId WHERE T1.rank = ?", (rank,))
    result = cursor.fetchall()
    if not result:
        return {"constructor_names": []}
    return {"constructor_names": [row[0] for row in result]}

# Endpoint to get the count of distinct constructors based on laps and nationality
@app.get("/v1/formula_1/distinct_constructor_count_by_laps_and_nationality", operation_id="get_distinct_constructor_count", summary="Retrieve the number of unique constructors that have completed more laps than the specified amount and belong to the given nationality.")
async def get_distinct_constructor_count(laps: int = Query(..., description="Number of laps"), nationality: str = Query(..., description="Nationality of the constructor")):
    cursor.execute("SELECT COUNT(DISTINCT T2.constructorId) FROM results AS T1 INNER JOIN constructors AS T2 on T1.constructorId = T2.constructorId WHERE T1.laps > ? AND T2.nationality = ?", (laps, nationality))
    result = cursor.fetchone()
    if not result:
        return {"constructor_count": []}
    return {"constructor_count": result[0]}

# Endpoint to get the percentage of races with valid times based on driver nationality and year range
@app.get("/v1/formula_1/percentage_valid_times_by_nationality_and_year_range", operation_id="get_percentage_valid_times", summary="Retrieve the percentage of Formula 1 races with valid times for a specific driver nationality within a given year range. The calculation is based on the total number of races in which the drivers of the specified nationality participated during the selected period.")
async def get_percentage_valid_times(nationality: str = Query(..., description="Nationality of the driver"), start_year: int = Query(..., description="Start year in 'YYYY' format"), end_year: int = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.time IS NOT NULL, 1, 0)) AS REAL) * 100 / COUNT(T1.raceId) FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId INNER JOIN drivers AS T3 on T1.driverId = T3.driverId WHERE T3.nationality = ? AND T2.year BETWEEN ? AND ?", (nationality, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average champion time in seconds for races before a specific year
@app.get("/v1/formula_1/average_champion_time_before_year", operation_id="get_average_champion_time", summary="Retrieves the average time in seconds taken by the champion to complete races held before a specified year. The year is provided in 'YYYY' format.")
async def get_average_champion_time(year: int = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("WITH time_in_seconds AS ( SELECT T2.year, T2.raceId, T1.positionOrder, CASE WHEN T1.positionOrder = 1 THEN (CAST(SUBSTR(T1.time, 1, 1) AS REAL) * 3600) + (CAST(SUBSTR(T1.time, 3, 2) AS REAL) * 60) + CAST(SUBSTR(T1.time, 6,2) AS REAL )   + CAST(SUBSTR(T1.time, 9) AS REAL)/1000 ELSE 0 END AS time_seconds FROM results AS T1 INNER JOIN races AS T2 ON T1.raceId = T2.raceId WHERE T1.time IS NOT NULL ), champion_time AS ( SELECT year, raceId, time_seconds FROM time_in_seconds WHERE positionOrder = 1 ) SELECT year, AVG(time_seconds) FROM champion_time WHERE year < ? GROUP BY year HAVING AVG(time_seconds) IS NOT NULL", (year,))
    result = cursor.fetchall()
    if not result:
        return {"average_times": []}
    return {"average_times": [{"year": row[0], "average_time": row[1]} for row in result]}

# Endpoint to get driver names based on birth year and rank
@app.get("/v1/formula_1/driver_names_by_birth_year_and_rank", operation_id="get_driver_names", summary="Retrieves the forenames and surnames of Formula 1 drivers born after the specified year and ranked at the given position. The birth year is provided in 'YYYY' format, and the rank indicates the driver's position in the results.")
async def get_driver_names(birth_year: str = Query(..., description="Birth year in 'YYYY' format"), rank: int = Query(..., description="Rank of the driver")):
    cursor.execute("SELECT T2.forename, T2.surname FROM results AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE STRFTIME('%Y', T2.dob) > ? AND T1.rank = ?", (birth_year, rank))
    result = cursor.fetchall()
    if not result:
        return {"driver_names": []}
    return {"driver_names": [{"forename": row[0], "surname": row[1]} for row in result]}

# Endpoint to get the count of drivers with a specific nationality who did not finish the race
@app.get("/v1/formula_1/count_drivers_nationality_not_finished", operation_id="get_count_drivers_nationality_not_finished", summary="Retrieves the number of drivers from a specific country who did not complete the race. The nationality of the drivers is used to filter the results.")
async def get_count_drivers_nationality_not_finished(nationality: str = Query(..., description="Nationality of the driver")):
    cursor.execute("SELECT COUNT(T1.driverId) FROM results AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE T2.nationality = ? AND T1.time IS NULL", (nationality,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the driver with the fastest lap time
@app.get("/v1/formula_1/fastest_lap_time_driver", operation_id="get_fastest_lap_time_driver", summary="Retrieves the driver with the fastest lap time from the results database. The operation returns the forename, surname, and fastest lap time of the driver who has achieved the quickest lap time in a race. The data is sorted in ascending order based on the lap time, with the fastest lap time appearing first.")
async def get_fastest_lap_time_driver():
    cursor.execute("SELECT T2.forename, T2.surname, T1.fastestLapTime FROM results AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE T1.fastestLapTime IS NOT NULL ORDER BY T1.fastestLapTime ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"forename": result[0], "surname": result[1], "fastestLapTime": result[2]}}

# Endpoint to get the fastest lap in a specific year with a specific time format
@app.get("/v1/formula_1/fastest_lap_year_time_format", operation_id="get_fastest_lap_year_time_format", summary="Get the fastest lap in a specific year with a specific time format")
async def get_fastest_lap_year_time_format(year: int = Query(..., description="Year of the race"), time_format: str = Query(..., description="Time format (e.g., '_:%:__.___')")):
    cursor.execute("SELECT T1.fastestLap FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId WHERE T2.year = ? AND T1.time LIKE ?", (year, time_format))
    result = cursor.fetchone()
    if not result:
        return {"fastestLap": []}
    return {"fastestLap": result[0]}

# Endpoint to get the average fastest lap speed in a specific year and race name
@app.get("/v1/formula_1/avg_fastest_lap_speed_year_race", operation_id="get_avg_fastest_lap_speed_year_race", summary="Retrieves the average fastest lap speed for a specific race in a given year. The operation calculates this average by considering the fastest lap speeds from all participants in the specified race and year. The input parameters include the year of the race and the name of the race.")
async def get_avg_fastest_lap_speed_year_race(year: int = Query(..., description="Year of the race"), race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT AVG(T1.fastestLapSpeed) FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId WHERE T2.year = ? AND T2.name = ?", (year, race_name))
    result = cursor.fetchone()
    if not result:
        return {"avg_speed": []}
    return {"avg_speed": result[0]}

# Endpoint to get the race with the fastest time
@app.get("/v1/formula_1/fastest_race", operation_id="get_fastest_race", summary="Retrieves the race with the fastest recorded time, including the race's name and year. The operation fetches this information by joining the races and results tables and ordering the results by the shortest time duration.")
async def get_fastest_race():
    cursor.execute("SELECT T1.name, T1.year FROM races AS T1 INNER JOIN results AS T2 on T1.raceId = T2.raceId WHERE T2.milliseconds IS NOT NULL ORDER BY T2.milliseconds LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"race": []}
    return {"race": {"name": result[0], "year": result[1]}}

# Endpoint to get the percentage of drivers born before a specific year who completed more than a specific number of laps in races between specific years
@app.get("/v1/formula_1/percentage_drivers_laps_years", operation_id="get_percentage_drivers_laps_years", summary="Retrieve the percentage of drivers who were born before a specific year and completed more than a certain number of laps in races held between specific years. The calculation considers all drivers who participated in races during the specified period.")
async def get_percentage_drivers_laps_years(birth_year: str = Query(..., description="Year of birth (format: 'YYYY')"), min_laps: int = Query(..., description="Minimum number of laps completed"), start_year: int = Query(..., description="Start year of the race period"), end_year: int = Query(..., description="End year of the race period")):
    cursor.execute("SELECT CAST(SUM(IIF(STRFTIME('%Y', T3.dob) < ? AND T1.laps > ?, 1, 0)) AS REAL) * 100 / COUNT(*) FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId INNER JOIN drivers AS T3 on T1.driverId = T3.driverId WHERE T2.year BETWEEN ? AND ?", (birth_year, min_laps, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of drivers with a specific nationality who completed a lap in less than a specific time
@app.get("/v1/formula_1/count_drivers_nationality_lap_time", operation_id="get_count_drivers_nationality_lap_time", summary="Retrieves the number of drivers from a specific nationality who have completed a lap in less than the specified time. The time is provided in seconds and is compared to the total lap time, which is calculated from the lap time string in the format 'MM:SS.MS'.")
async def get_count_drivers_nationality_lap_time(nationality: str = Query(..., description="Nationality of the driver"), max_time: float = Query(..., description="Maximum lap time in seconds")):
    cursor.execute("SELECT COUNT(T1.driverId) FROM drivers AS T1 INNER JOIN lapTimes AS T2 on T1.driverId = T2.driverId WHERE T1.nationality = ? AND (CAST(SUBSTR(T2.time, 1, 2) AS INTEGER) * 60 + CAST(SUBSTR(T2.time, 4, 2) AS INTEGER) + CAST(SUBSTR(T2.time, 7, 2) AS REAL) / 1000) < ?", (nationality, max_time))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the driver codes for a specific nationality
@app.get("/v1/formula_1/driver_codes_nationality", operation_id="get_driver_codes_nationality", summary="Retrieves the unique codes of all drivers who share a specified nationality. The nationality is provided as an input parameter.")
async def get_driver_codes_nationality(nationality: str = Query(..., description="Nationality of the driver")):
    cursor.execute("SELECT code FROM drivers WHERE Nationality = ?", (nationality,))
    result = cursor.fetchall()
    if not result:
        return {"codes": []}
    return {"codes": [row[0] for row in result]}

# Endpoint to get the race IDs for a specific year
@app.get("/v1/formula_1/race_ids_year", operation_id="get_race_ids_year", summary="Retrieves the unique identifiers of all Formula 1 races that took place in a specified year.")
async def get_race_ids_year(year: int = Query(..., description="Year of the race")):
    cursor.execute("SELECT raceId FROM races WHERE year = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"raceIds": []}
    return {"raceIds": [row[0] for row in result]}

# Endpoint to get the count of drivers in the standings for a specific race
@app.get("/v1/formula_1/count_drivers_standings_race", operation_id="get_count_drivers_standings_race", summary="Retrieves the total number of drivers who participated in a specific race, as indicated by the provided race ID.")
async def get_count_drivers_standings_race(race_id: int = Query(..., description="Race ID")):
    cursor.execute("SELECT COUNT(driverId) FROM driverStandings WHERE raceId = ?", (race_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of drivers with a specific nationality among the top 3 youngest drivers
@app.get("/v1/formula_1/count_youngest_drivers_by_nationality", operation_id="get_count_youngest_drivers_by_nationality", summary="Retrieves the number of drivers from a specific nationality who are among the top 3 youngest drivers in the database.")
async def get_count_youngest_drivers_by_nationality(nationality: str = Query(..., description="Nationality of the driver")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T1.nationality FROM drivers AS T1 ORDER BY JULIANDAY(T1.dob) DESC LIMIT 3) AS T3 WHERE T3.nationality = ?", (nationality,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of drivers based on nationality and year of birth
@app.get("/v1/formula_1/count_drivers_by_nationality_and_year", operation_id="get_count_drivers_by_nationality_and_year", summary="Retrieves the total number of Formula 1 drivers from a specific nationality who were born in a given year. The nationality and year of birth are provided as input parameters.")
async def get_count_drivers_by_nationality_and_year(nationality: str = Query(..., description="Nationality of the driver"), year: str = Query(..., description="Year of birth in 'YYYY' format")):
    cursor.execute("SELECT COUNT(driverId) FROM drivers WHERE nationality = ? AND STRFTIME('%Y', dob) = ?", (nationality, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the driver IDs of German drivers born between specific years with the fastest pit stops
@app.get("/v1/formula_1/fastest_pit_stops_german_drivers", operation_id="get_fastest_pit_stops_german_drivers", summary="Retrieve the top three German drivers with the fastest pit stops, born between the specified start and end years.")
async def get_fastest_pit_stops_german_drivers(nationality: str = Query(..., description="Nationality of the driver"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT T2.driverId FROM pitStops AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE T2.nationality = ? AND STRFTIME('%Y', T2.dob) BETWEEN ? AND ? ORDER BY T1.time LIMIT 3", (nationality, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"driverIds": []}
    return {"driverIds": [row[0] for row in result]}

# Endpoint to get the driver reference of the oldest driver from a specific nationality
@app.get("/v1/formula_1/oldest_driver_by_nationality", operation_id="get_oldest_driver_by_nationality", summary="Retrieves the reference of the oldest driver from the specified nationality. The operation sorts the drivers by their date of birth in ascending order and returns the reference of the oldest driver.")
async def get_oldest_driver_by_nationality(nationality: str = Query(..., description="Nationality of the driver")):
    cursor.execute("SELECT driverRef FROM drivers WHERE nationality = ? ORDER BY JULIANDAY(dob) ASC LIMIT 1", (nationality,))
    result = cursor.fetchone()
    if not result:
        return {"driverRef": []}
    return {"driverRef": result[0]}

# Endpoint to get the driver IDs and codes of drivers born in a specific year with a fastest lap time
@app.get("/v1/formula_1/drivers_with_fastest_lap_by_year", operation_id="get_drivers_with_fastest_lap_by_year", summary="Retrieves the unique identifiers and codes of Formula 1 drivers born in a specific year who have recorded a fastest lap time. The year of birth must be provided in the 'YYYY' format.")
async def get_drivers_with_fastest_lap_by_year(year: str = Query(..., description="Year of birth in 'YYYY' format")):
    cursor.execute("SELECT T2.driverId, T2.code FROM results AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE STRFTIME('%Y', T2.dob) = ? AND T1.fastestLapTime IS NOT NULL", (year,))
    result = cursor.fetchall()
    if not result:
        return {"drivers": []}
    return {"drivers": [{"driverId": row[0], "code": row[1]} for row in result]}

# Endpoint to get the driver IDs of Spanish drivers born before a specific year with the longest pit stops
@app.get("/v1/formula_1/longest_pit_stops_spanish_drivers", operation_id="get_longest_pit_stops_spanish_drivers", summary="Retrieves the top 10 Spanish drivers with the longest pit stop durations who were born before a specified year. The nationality and year of birth are used to filter the results.")
async def get_longest_pit_stops_spanish_drivers(nationality: str = Query(..., description="Nationality of the driver"), year: str = Query(..., description="Year of birth in 'YYYY' format")):
    cursor.execute("SELECT T2.driverId FROM pitStops AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE T2.nationality = ? AND STRFTIME('%Y', T2.dob) < ? ORDER BY T1.time DESC LIMIT 10", (nationality, year))
    result = cursor.fetchall()
    if not result:
        return {"driverIds": []}
    return {"driverIds": [row[0] for row in result]}

# Endpoint to get the years of races where the fastest lap time was recorded
@app.get("/v1/formula_1/race_years_with_fastest_lap", operation_id="get_race_years_with_fastest_lap", summary="Retrieves the years of Formula 1 races in which the fastest lap time was recorded. This operation returns a list of years, each representing a season where at least one race had a recorded fastest lap time. The data is sourced from the results and races tables, with a focus on races where the fastest lap time is not null.")
async def get_race_years_with_fastest_lap():
    cursor.execute("SELECT T2.year FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId WHERE T1.fastestLapTime IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"years": []}
    return {"years": [row[0] for row in result]}

# Endpoint to get the year of the race with the longest lap time
@app.get("/v1/formula_1/race_year_longest_lap_time", operation_id="get_race_year_longest_lap_time", summary="Retrieves the year of the race with the longest lap time from the database. The operation identifies the race with the longest lap time by comparing the lap times of all races and then returns the year of that race.")
async def get_race_year_longest_lap_time():
    cursor.execute("SELECT T2.year FROM lapTimes AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId ORDER BY T1.time DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the driver IDs with the fastest lap times for a specific lap
@app.get("/v1/formula_1/fastest_lap_times_by_lap", operation_id="get_fastest_lap_times_by_lap", summary="Retrieves the top five driver IDs with the fastest lap times for a specified lap number. The lap times are ordered in ascending order, with the fastest time listed first.")
async def get_fastest_lap_times_by_lap(lap: int = Query(..., description="Lap number")):
    cursor.execute("SELECT driverId FROM lapTimes WHERE lap = ? ORDER BY time LIMIT 5", (lap,))
    result = cursor.fetchall()
    if not result:
        return {"driverIds": []}
    return {"driverIds": [row[0] for row in result]}

# Endpoint to get the sum of non-null times for a specific status ID and race ID range
@app.get("/v1/formula_1/sum_non_null_times", operation_id="get_sum_non_null_times", summary="Retrieves the total count of non-null times for a specific status, within a defined range of race IDs. The operation filters results based on the provided status ID and race ID range, and calculates the sum of non-null times.")
async def get_sum_non_null_times(status_id: int = Query(..., description="Status ID"), max_race_id: int = Query(..., description="Maximum race ID"), min_race_id: int = Query(..., description="Minimum race ID")):
    cursor.execute("SELECT SUM(IIF(time IS NOT NULL, 1, 0)) FROM results WHERE statusId = ? AND raceID < ? AND raceId > ?", (status_id, max_race_id, min_race_id))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct circuit locations for a specific country
@app.get("/v1/formula_1/distinct_circuit_locations", operation_id="get_distinct_circuit_locations", summary="Retrieve a unique set of circuit locations, along with their latitude and longitude coordinates, for a specified country.")
async def get_distinct_circuit_locations(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT DISTINCT location, lat, lng FROM circuits WHERE country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": result}

# Endpoint to get the race ID with the highest count of non-null times
@app.get("/v1/formula_1/race_with_most_non_null_times", operation_id="get_race_with_most_non_null_times", summary="Retrieves the ID of the race with the most recorded times. This operation identifies the race with the highest number of non-null times, indicating the race with the most complete time data.")
async def get_race_with_most_non_null_times():
    cursor.execute("SELECT raceId FROM results GROUP BY raceId ORDER BY COUNT(time IS NOT NULL) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"race_id": []}
    return {"race_id": result[0]}

# Endpoint to get driver details for a specific race ID where q2 is not null
@app.get("/v1/formula_1/driver_details_qualifying", operation_id="get_driver_details_qualifying", summary="Retrieves the details of drivers who have qualified for Q2 in a specific race. The details include the driver's reference, nationality, and date of birth. The race is identified by its unique race ID.")
async def get_driver_details_qualifying(race_id: int = Query(..., description="Race ID")):
    cursor.execute("SELECT T2.driverRef, T2.nationality, T2.dob FROM qualifying AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE T1.raceId = ? AND T1.q2 IS NOT NULL", (race_id,))
    result = cursor.fetchall()
    if not result:
        return {"driver_details": []}
    return {"driver_details": result}

# Endpoint to get the earliest race details for the youngest driver
@app.get("/v1/formula_1/earliest_race_youngest_driver", operation_id="get_earliest_race_youngest_driver", summary="Retrieves the details of the earliest race in which the youngest driver in the database participated. The response includes the year, name, date, and time of the race.")
async def get_earliest_race_youngest_driver():
    cursor.execute("SELECT T3.year, T3.name, T3.date, T3.time FROM qualifying AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId INNER JOIN races AS T3 on T1.raceId = T3.raceId WHERE T1.driverId = ( SELECT driverId FROM drivers ORDER BY dob DESC LIMIT 1 ) ORDER BY T3.date ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"race_details": []}
    return {"race_details": result}

# Endpoint to get the count of drivers with a specific status and nationality
@app.get("/v1/formula_1/count_drivers_status_nationality", operation_id="get_count_drivers_status_nationality", summary="Retrieves the total number of Formula 1 drivers who have a specified status and nationality. The status and nationality are provided as input parameters, allowing for a targeted count of drivers that meet the given criteria.")
async def get_count_drivers_status_nationality(status: str = Query(..., description="Status of the driver"), nationality: str = Query(..., description="Nationality of the driver")):
    cursor.execute("SELECT COUNT(T1.driverId) FROM drivers AS T1 INNER JOIN results AS T2 on T1.driverId = T2.driverId INNER JOIN status AS T3 on T2.statusId = T3.statusId WHERE T3.status = ? AND T1.nationality = ?", (status, nationality))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the constructor URL with the most wins
@app.get("/v1/formula_1/top_constructor_by_wins", operation_id="get_top_constructor_by_wins", summary="Retrieves the URL of the constructor with the highest number of wins in the Formula 1 constructor standings.")
async def get_top_constructor_by_wins():
    cursor.execute("SELECT T1.url FROM constructors AS T1 INNER JOIN constructorStandings AS T2 on T1.constructorId = T2.constructorId ORDER BY T2.wins DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"url": []}
    return {"url": result[0]}

# Endpoint to get the driver ID with the slowest lap time for a specific race and lap
@app.get("/v1/formula_1/slowest_lap_driver", operation_id="get_slowest_lap_driver", summary="Retrieves the ID of the driver who recorded the slowest lap time for a specified race and lap number. The operation filters lap times for the given race and lap, then identifies the driver with the highest lap time.")
async def get_slowest_lap_driver(race_name: str = Query(..., description="Name of the race"), lap: int = Query(..., description="Lap number")):
    cursor.execute("SELECT T1.driverId FROM lapTimes AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId WHERE T2.name = ? AND T1.lap = ? ORDER BY T1.time DESC LIMIT 1", (race_name, lap))
    result = cursor.fetchone()
    if not result:
        return {"driver_id": []}
    return {"driver_id": result[0]}

# Endpoint to get the average fastest lap time for races with specific criteria
@app.get("/v1/formula_1/avg_fastest_lap_time", operation_id="get_avg_fastest_lap_time", summary="Retrieves the average fastest lap time from races that meet the specified criteria. These criteria include a rank below a certain threshold, a specific year, and a particular race name.")
async def get_avg_fastest_lap_time(rank: int = Query(..., description="Rank threshold"), year: int = Query(..., description="Year of the race"), name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT AVG(T1.fastestLapTime) FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId WHERE T1.rank < ? AND T2.year = ? AND T2.name = ?", (rank, year, name))
    result = cursor.fetchone()
    if not result:
        return {"average_fastest_lap_time": []}
    return {"average_fastest_lap_time": result[0]}

# Endpoint to get race times based on race name, year, and time format
@app.get("/v1/formula_1/race_times_by_name_year_and_time_format", operation_id="get_race_times", summary="Get the race times based on the race name, year, and a specific time format.")
async def get_race_times(name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race"), time_format: str = Query(..., description="Time format to match, e.g., '_:%:__.___'")):
    cursor.execute("SELECT T1.time FROM results AS T1 INNER JOIN races AS T2 ON T1.raceId = T2.raceId WHERE T2.name = ? AND T2.year = ? AND T1.time LIKE ?", (name, year, time_format))
    result = cursor.fetchall()
    if not result:
        return {"times": []}
    return {"times": result}

# Endpoint to get constructor references and URLs based on race name, year, and time format
@app.get("/v1/formula_1/constructor_refs_and_urls_by_race_name_year_and_time_format", operation_id="get_constructor_refs_and_urls", summary="Get the constructor references and URLs based on the race name, year, and a specific time format.")
async def get_constructor_refs_and_urls(name: str = Query(..., description="Name of the race"), year: int = Query(..., description="Year of the race"), time_format: str = Query(..., description="Time format to match, e.g., '_:%:__.___'")):
    cursor.execute("SELECT T3.constructorRef, T3.url FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId INNER JOIN constructors AS T3 on T1.constructorId = T3.constructorId WHERE T2.name = ? AND T2.year = ? AND T1.time LIKE ?", (name, year, time_format))
    result = cursor.fetchall()
    if not result:
        return {"constructor_refs_and_urls": []}
    return {"constructor_refs_and_urls": result}

# Endpoint to get driver details based on nationality and birth year range, ordered by date of birth
@app.get("/v1/formula_1/driver_details_by_nationality_and_birth_year_ordered", operation_id="get_driver_details_ordered", summary="Retrieve the forename, surname, URL, and date of birth of drivers who share a specified nationality and were born within a given year range. The results are sorted in descending order by date of birth.")
async def get_driver_details_ordered(nationality: str = Query(..., description="Nationality of the driver"), start_year: str = Query(..., description="Start year of the birth year range in 'YYYY' format"), end_year: str = Query(..., description="End year of the birth year range in 'YYYY' format")):
    cursor.execute("SELECT forename, surname, url, dob FROM drivers WHERE nationality = ? AND STRFTIME('%Y', dob) BETWEEN ? AND ? ORDER BY dob DESC", (nationality, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"drivers": []}
    return {"drivers": result}

# Endpoint to get circuit details based on circuit name
@app.get("/v1/formula_1/circuit_details_by_name", operation_id="get_circuit_details", summary="Retrieves the geographical details of a specific Formula 1 circuit, including the country, latitude, and longitude, based on the provided circuit name.")
async def get_circuit_details(name: str = Query(..., description="Name of the circuit")):
    cursor.execute("SELECT country, lat, lng FROM circuits WHERE name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"circuits": []}
    return {"circuits": result}

# Endpoint to get constructor points based on race name and year range
@app.get("/v1/formula_1/constructor_points_by_race_name_and_year_range", operation_id="get_constructor_points", summary="Retrieve the total points earned by constructors in a specific race, within a given year range. The results are ordered by total points in descending order and limited to a specified number of results. The input parameters include the race name, start and end years of the range, and the maximum number of results to return.")
async def get_constructor_points(name: str = Query(..., description="Name of the race"), start_year: int = Query(..., description="Start year of the year range"), end_year: int = Query(..., description="End year of the year range"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT SUM(T1.points), T2.name, T2.nationality FROM constructorResults AS T1 INNER JOIN constructors AS T2 ON T1.constructorId = T2.constructorId INNER JOIN races AS T3 ON T3.raceid = T1.raceid WHERE T3.name = ? AND T3.year BETWEEN ? AND ? GROUP BY T2.name ORDER BY SUM(T1.points) DESC LIMIT ?", (name, start_year, end_year, limit))
    result = cursor.fetchall()
    if not result:
        return {"constructors": []}
    return {"constructors": result}

# Endpoint to get the average points of a driver in a specific race
@app.get("/v1/formula_1/avg_driver_points_by_race", operation_id="get_avg_driver_points", summary="Retrieves the average points earned by a specific driver in a given race. The driver is identified by their forename and surname, while the race is specified by its name. This operation calculates the average points based on the driver's performance across all instances of the specified race.")
async def get_avg_driver_points(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver"), name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT AVG(T2.points) FROM drivers AS T1 INNER JOIN driverStandings AS T2 ON T1.driverId = T2.driverId INNER JOIN races AS T3 ON T3.raceId = T2.raceId WHERE T1.forename = ? AND T1.surname = ? AND T3.name = ?", (forename, surname, name))
    result = cursor.fetchone()
    if not result:
        return {"average_points": []}
    return {"average_points": result[0]}

# Endpoint to get the average number of races per year within a date range
@app.get("/v1/formula_1/avg_races_per_year", operation_id="get_avg_races_per_year", summary="Retrieves the average number of races per year within a specified date range. The calculation considers the total number of years in the range and the number of races that occurred within that period. The date range can be defined using either the start and end years or the start and end dates in 'YYYY-MM-DD' format.")
async def get_avg_races_per_year(start_year: int = Query(..., description="Start year of the date range"), end_year: int = Query(..., description="End year of the date range"), total_years: int = Query(..., description="Total number of years in the range"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN year BETWEEN ? AND ? THEN 1 ELSE 0 END) AS REAL) / ? FROM races WHERE date BETWEEN ? AND ?", (start_year, end_year, total_years, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_races_per_year": []}
    return {"average_races_per_year": result[0]}

# Endpoint to get the most common nationality among drivers
@app.get("/v1/formula_1/most_common_nationality", operation_id="get_most_common_nationality", summary="Retrieves the most frequently occurring nationality among Formula 1 drivers. The operation allows you to limit the number of results returned, providing a concise overview of the most common nationalities.")
async def get_most_common_nationality(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT nationality FROM drivers GROUP BY nationality ORDER BY COUNT(driverId) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"nationality": []}
    return {"nationality": result[0]}

# Endpoint to get the sum of wins for drivers with a specific points value
@app.get("/v1/formula_1/sum_wins_by_points", operation_id="get_sum_wins_by_points", summary="Retrieves the total number of wins accumulated by drivers who have earned a specific number of points. The points value is used to filter the drivers for which wins are summed.")
async def get_sum_wins_by_points(points: int = Query(..., description="Points value to filter drivers")):
    cursor.execute("SELECT SUM(CASE WHEN points = ? THEN wins ELSE 0 END) FROM driverStandings", (points,))
    result = cursor.fetchone()
    if not result:
        return {"sum_wins": []}
    return {"sum_wins": result[0]}

# Endpoint to get the race with the fastest lap time
@app.get("/v1/formula_1/fastest_lap_race", operation_id="get_fastest_lap_race", summary="Retrieve the race with the fastest lap time, with the option to limit the number of results returned. The operation returns the name of the race with the fastest lap time, based on the results from the 'results' table. The 'limit' parameter can be used to specify the maximum number of results to return.")
async def get_fastest_lap_race(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T1.name FROM races AS T1 INNER JOIN results AS T2 ON T1.raceId = T2.raceId WHERE T2.fastestLapTime IS NOT NULL ORDER BY T2.fastestLapTime ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"race_name": []}
    return {"race_name": result[0]}

# Endpoint to get the location of the most recent race
@app.get("/v1/formula_1/most_recent_race_location", operation_id="get_most_recent_race_location", summary="Retrieves the location of the most recent race, with the option to limit the number of results returned. This operation provides the latest race location data, which can be used to keep track of the current Formula 1 race schedule.")
async def get_most_recent_race_location(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T1.location FROM circuits AS T1 INNER JOIN races AS T2 ON T1.circuitId = T2.circuitId ORDER BY T2.date DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": result[0]}

# Endpoint to get the fastest qualifying driver in a specific year and circuit
@app.get("/v1/formula_1/fastest_qualifying_driver", operation_id="get_fastest_qualifying_driver", summary="Retrieve the driver with the fastest qualifying time in a specific year and circuit. The operation returns the forename and surname of the driver. The year and circuit name are used to filter the results, and the limit parameter can be used to restrict the number of results returned.")
async def get_fastest_qualifying_driver(year: int = Query(..., description="Year of the race"), circuit_name: str = Query(..., description="Name of the circuit"), limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T2.forename, T2.surname FROM qualifying AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId INNER JOIN races AS T3 ON T1.raceid = T3.raceid WHERE q3 IS NOT NULL AND T3.year = ? AND T3.circuitId IN ( SELECT circuitId FROM circuits WHERE name = ? ) ORDER BY CAST(SUBSTR(q3, 1, INSTR(q3, ':') - 1) AS INTEGER) * 60 + CAST(SUBSTR(q3, INSTR(q3, ':') + 1, INSTR(q3, '.') - INSTR(q3, ':') - 1) AS REAL) + CAST(SUBSTR(q3, INSTR(q3, '.') + 1) AS REAL) / 1000 ASC LIMIT ?", (year, circuit_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"forename": result[0], "surname": result[1]}

# Endpoint to get the youngest driver in the standings
@app.get("/v1/formula_1/youngest_driver_in_standings", operation_id="get_youngest_driver_in_standings", summary="Retrieve the youngest driver(s) in the current standings, based on their date of birth. The number of results returned can be limited by specifying the 'limit' parameter.")
async def get_youngest_driver_in_standings(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T1.forename, T1.surname, T1.nationality, T3.name FROM drivers AS T1 INNER JOIN driverStandings AS T2 on T1.driverId = T2.driverId INNER JOIN races AS T3 on T2.raceId = T3.raceId ORDER BY JULIANDAY(T1.dob) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"forename": result[0], "surname": result[1], "nationality": result[2], "race_name": result[3]}

# Endpoint to get the driver with the most occurrences of a specific status in a specific race
@app.get("/v1/formula_1/most_occurrences_of_status_in_race", operation_id="get_most_occurrences_of_status_in_race", summary="Retrieve the driver who has achieved a specific status the most times in a given race. The operation allows you to specify the status, the race, and the maximum number of results to return.")
async def get_most_occurrences_of_status_in_race(status_id: int = Query(..., description="Status ID to filter drivers"), race_name: str = Query(..., description="Name of the race"), limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT COUNT(T1.driverId) FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId INNER JOIN status AS T3 on T1.statusId = T3.statusId WHERE T3.statusId = ? AND T2.name = ? GROUP BY T1.driverId ORDER BY COUNT(T1.driverId) DESC LIMIT ?", (status_id, race_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the oldest driver with the most wins
@app.get("/v1/formula_1/oldest_driver_with_most_wins", operation_id="get_oldest_driver_with_most_wins", summary="Retrieves the oldest driver with the most wins, limited to the specified number of results. The operation calculates the total wins for each driver and sorts them by their date of birth in ascending order. The oldest driver with the highest total wins is returned.")
async def get_oldest_driver_with_most_wins(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT SUM(T1.wins),T2.forename, T2.surname FROM driverStandings AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId ORDER BY T2.dob ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"sum_wins": result[0], "forename": result[1], "surname": result[2]}

# Endpoint to get the longest pit stop duration
@app.get("/v1/formula_1/longest_pit_stop_duration", operation_id="get_longest_pit_stop_duration", summary="Retrieves the duration of the longest pit stop(s) in descending order, up to the specified limit. This operation allows you to obtain the most time-consuming pit stop durations, providing valuable insights into race performance and strategy.")
async def get_longest_pit_stop_duration(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT duration FROM pitStops ORDER BY duration DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"duration": []}
    return {"duration": result[0]}

# Endpoint to get the lap number of a pit stop for a specific driver in a specific race
@app.get("/v1/formula_1/pit_stop_lap_number", operation_id="get_pit_stop_lap_number", summary="Get the lap number of a pit stop for a driver given their forename, surname, race year, and race name")
async def get_pit_stop_lap_number(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver"), year: int = Query(..., description="Year of the race"), race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT T1.lap FROM pitStops AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId INNER JOIN races AS T3 on T1.raceId = T3.raceId WHERE T2.forename = ? AND T2.surname = ? AND T3.year = ? AND T3.name = ?", (forename, surname, year, race_name))
    result = cursor.fetchone()
    if not result:
        return {"lap": []}
    return {"lap": result[0]}

# Endpoint to get the pit stop durations for a specific race
@app.get("/v1/formula_1/pit_stop_durations_by_race", operation_id="get_pit_stop_durations_by_race", summary="Retrieves the pit stop durations for a specific race in a given year. The operation requires the year and name of the race as input parameters. It returns a list of pit stop durations for the specified race.")
async def get_pit_stop_durations_by_race(year: int = Query(..., description="Year of the race"), race_name: str = Query(..., description="Name of the race")):
    cursor.execute("SELECT T1.duration FROM pitStops AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId WHERE T2.year = ? AND T2.name = ?", (year, race_name))
    results = cursor.fetchall()
    if not results:
        return {"durations": []}
    return {"durations": [result[0] for result in results]}

# Endpoint to get the lap times for a specific driver
@app.get("/v1/formula_1/lap_times_by_driver", operation_id="get_lap_times_by_driver", summary="Retrieves the lap times for a specific Formula 1 driver, identified by their forename and surname. The operation returns a list of lap times associated with the driver, obtained by querying the lapTimes table and joining it with the drivers table based on the driverId.")
async def get_lap_times_by_driver(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT T1.time FROM lapTimes AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE T2.forename = ? AND T2.surname = ?", (forename, surname))
    results = cursor.fetchall()
    if not results:
        return {"times": []}
    return {"times": [result[0] for result in results]}

# Endpoint to get the top 20 drivers with the fastest lap times
@app.get("/v1/formula_1/top_20_fastest_lap_times", operation_id="get_top_20_fastest_lap_times", summary="Retrieve the top 20 Formula 1 drivers with the fastest lap times. The operation calculates the lap times in seconds and ranks the drivers based on their minimum lap time. The results are ordered from the fastest to the slowest, with a limit of 20 drivers.")
async def get_top_20_fastest_lap_times():
    cursor.execute("WITH lap_times_in_seconds AS (SELECT driverId, (CASE WHEN SUBSTR(time, 1, INSTR(time, ':') - 1) <> '' THEN CAST(SUBSTR(time, 1, INSTR(time, ':') - 1) AS REAL) * 60 ELSE 0 END + CASE WHEN SUBSTR(time, INSTR(time, ':') + 1, INSTR(time, '.') - INSTR(time, ':') - 1) <> '' THEN CAST(SUBSTR(time, INSTR(time, ':') + 1, INSTR(time, '.') - INSTR(time, ':') - 1) AS REAL) ELSE 0 END + CASE WHEN SUBSTR(time, INSTR(time, '.') + 1) <> '' THEN CAST(SUBSTR(time, INSTR(time, '.') + 1) AS REAL) / 1000 ELSE 0 END) AS time_in_seconds FROM lapTimes) SELECT T2.forename, T2.surname, T1.driverId FROM (SELECT driverId, MIN(time_in_seconds) AS min_time_in_seconds FROM lap_times_in_seconds GROUP BY driverId) AS T1 INNER JOIN drivers AS T2 ON T1.driverId = T2.driverId ORDER BY T1.min_time_in_seconds ASC LIMIT 20")
    results = cursor.fetchall()
    if not results:
        return {"drivers": []}
    return {"drivers": [{"forename": result[0], "surname": result[1], "driverId": result[2]} for result in results]}

# Endpoint to get the best position of a specific driver
@app.get("/v1/formula_1/best_position_by_driver", operation_id="get_best_position_by_driver", summary="Retrieves the best position achieved by a Formula 1 driver, based on their forename and surname. The operation identifies the driver using the provided names, then scans their lap times to determine the best position they have ever achieved. The position is determined by the fastest lap time, with lower times indicating better positions.")
async def get_best_position_by_driver(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT T1.position FROM lapTimes AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE T2.forename = ? AND T2.surname = ? ORDER BY T1.time ASC LIMIT 1", (forename, surname))
    result = cursor.fetchone()
    if not result:
        return {"position": []}
    return {"position": result[0]}

# Endpoint to get the fastest lap time for a specific race
@app.get("/v1/formula_1/fastest_lap_time_by_race", operation_id="get_fastest_lap_time_by_race", summary="Retrieves the fastest lap time recorded for a specific race. The operation identifies the minimum lap time from the results where a fastest lap time is available, and filters the data based on the provided race name. The response includes the fastest lap time for the specified race.")
async def get_fastest_lap_time_by_race(race_name: str = Query(..., description="Name of the race")):
    cursor.execute("WITH fastest_lap_times AS ( SELECT T1.raceId, T1.fastestLapTime FROM results AS T1 WHERE T1.FastestLapTime IS NOT NULL) SELECT MIN(fastest_lap_times.fastestLapTime) as lap_record FROM fastest_lap_times INNER JOIN races AS T2 on fastest_lap_times.raceId = T2.raceId INNER JOIN circuits AS T3 on T2.circuitId = T3.circuitId WHERE T2.name = ?", (race_name,))
    result = cursor.fetchone()
    if not result:
        return {"lap_record": []}
    return {"lap_record": result[0]}

# Endpoint to get the fastest lap time in a specific country
@app.get("/v1/formula_1/fastest_lap_time_by_country", operation_id="get_fastest_lap_time_by_country", summary="Retrieves the fastest lap time recorded in a race held in the specified country. The operation calculates the lap time in seconds and returns the fastest lap time from the results table, which is joined with the races and circuits tables to determine the country of the race.")
async def get_fastest_lap_time_by_country(country: str = Query(..., description="Name of the country")):
    cursor.execute("WITH fastest_lap_times AS (SELECT T1.raceId, T1.FastestLapTime, (CAST(SUBSTR(T1.FastestLapTime, 1, INSTR(T1.FastestLapTime, ':') - 1) AS REAL) * 60) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, ':') + 1, INSTR(T1.FastestLapTime, '.') - INSTR(T1.FastestLapTime, ':') - 1) AS REAL)) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, '.') + 1) AS REAL) / 1000) as time_in_seconds FROM results AS T1 WHERE T1.FastestLapTime IS NOT NULL ) SELECT T1.FastestLapTime as lap_record FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId INNER JOIN circuits AS T3 on T2.circuitId = T3.circuitId INNER JOIN (SELECT MIN(fastest_lap_times.time_in_seconds) as min_time_in_seconds FROM fastest_lap_times INNER JOIN races AS T2 on fastest_lap_times.raceId = T2.raceId INNER JOIN circuits AS T3 on T2.circuitId = T3.circuitId WHERE T3.country = ? ) AS T4 ON (CAST(SUBSTR(T1.FastestLapTime, 1, INSTR(T1.FastestLapTime, ':') - 1) AS REAL) * 60) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, ':') + 1, INSTR(T1.FastestLapTime, '.') - INSTR(T1.FastestLapTime, ':') - 1) AS REAL)) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, '.') + 1) AS REAL) / 1000) = T4.min_time_in_seconds LIMIT 1", (country,))
    result = cursor.fetchone()
    if not result:
        return {"lap_record": []}
    return {"lap_record": result[0]}

# Endpoint to get the name of the race with the fastest lap time for a specific race
@app.get("/v1/formula_1/fastest_lap_time_race_name", operation_id="get_fastest_lap_time_race_name", summary="Retrieves the name of the race with the fastest lap time, based on the provided race name. The operation calculates the lap time in seconds and compares it with the minimum lap time across all races. The result is the name of the race with the fastest lap time.")
async def get_fastest_lap_time_race_name(race_name: str = Query(..., description="Name of the race")):
    cursor.execute("WITH fastest_lap_times AS ( SELECT T1.raceId, T1.FastestLapTime, (CAST(SUBSTR(T1.FastestLapTime, 1, INSTR(T1.FastestLapTime, ':') - 1) AS REAL) * 60) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, ':') + 1, INSTR(T1.FastestLapTime, '.') - INSTR(T1.FastestLapTime, ':') - 1) AS REAL)) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, '.') + 1) AS REAL) / 1000) as time_in_seconds FROM results AS T1 WHERE T1.FastestLapTime IS NOT NULL ) SELECT T2.name FROM races AS T2 INNER JOIN circuits AS T3 on T2.circuitId = T3.circuitId INNER JOIN results AS T1 on T2.raceId = T1.raceId INNER JOIN ( SELECT MIN(fastest_lap_times.time_in_seconds) as min_time_in_seconds FROM fastest_lap_times INNER JOIN races AS T2 on fastest_lap_times.raceId = T2.raceId INNER JOIN circuits AS T3 on T2.circuitId = T3.circuitId WHERE T2.name = ?) AS T4 ON (CAST(SUBSTR(T1.FastestLapTime, 1, INSTR(T1.FastestLapTime, ':') - 1) AS REAL) * 60) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, ':') + 1, INSTR(T1.FastestLapTime, '.') - INSTR(T1.FastestLapTime, ':') - 1) AS REAL)) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, '.') + 1) AS REAL) / 1000) = T4.min_time_in_seconds WHERE T2.name = ?", (race_name, race_name))
    result = cursor.fetchone()
    if not result:
        return {"race_name": []}
    return {"race_name": result[0]}

# Endpoint to get the pit stop duration for the driver with the fastest lap time in a specific race
@app.get("/v1/formula_1/pit_stop_duration_for_fastest_lap", operation_id="get_pit_stop_duration_for_fastest_lap", summary="Retrieves the pit stop duration for the driver who achieved the fastest lap time in a specific race. The race is identified by its name.")
async def get_pit_stop_duration_for_fastest_lap(race_name: str = Query(..., description="Name of the race")):
    cursor.execute("WITH fastest_lap_times AS ( SELECT T1.raceId, T1.driverId, T1.FastestLapTime, (CAST(SUBSTR(T1.FastestLapTime, 1, INSTR(T1.FastestLapTime, ':') - 1) AS REAL) * 60) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, ':') + 1, INSTR(T1.FastestLapTime, '.') - INSTR(T1.FastestLapTime, ':') - 1) AS REAL)) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, '.') + 1) AS REAL) / 1000) as time_in_seconds FROM results AS T1 WHERE T1.FastestLapTime IS NOT NULL), lap_record_race AS ( SELECT T1.raceId, T1.driverId FROM results AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId INNER JOIN circuits AS T3 on T2.circuitId = T3.circuitId INNER JOIN ( SELECT MIN(fastest_lap_times.time_in_seconds) as min_time_in_seconds FROM fastest_lap_times INNER JOIN races AS T2 on fastest_lap_times.raceId = T2.raceId INNER JOIN circuits AS T3 on T2.circuitId = T3.circuitId WHERE T2.name = ?) AS T4 ON (CAST(SUBSTR(T1.FastestLapTime, 1, INSTR(T1.FastestLapTime, ':') - 1) AS REAL) * 60) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, ':') + 1, INSTR(T1.FastestLapTime, '.') - INSTR(T1.FastestLapTime, ':') - 1) AS REAL)) + (CAST(SUBSTR(T1.FastestLapTime, INSTR(T1.FastestLapTime, '.') + 1) AS REAL) / 1000) = T4.min_time_in_seconds WHERE T2.name = ?) SELECT T4.duration FROM lap_record_race INNER JOIN pitStops AS T4 on lap_record_race.raceId = T4.raceId AND lap_record_race.driverId = T4.driverId", (race_name, race_name))
    result = cursor.fetchone()
    if not result:
        return {"duration": []}
    return {"duration": result[0]}

# Endpoint to get the latitude and longitude of a circuit based on lap time
@app.get("/v1/formula_1/circuit_location_by_lap_time", operation_id="get_circuit_location_by_lap_time", summary="Retrieves the geographical coordinates (latitude and longitude) of the circuit where a given lap time was recorded. The lap time should be provided in the format 'MM:SS.sss'.")
async def get_circuit_location_by_lap_time(lap_time: str = Query(..., description="Lap time in the format 'MM:SS.sss'")):
    cursor.execute("SELECT T3.lat, T3.lng FROM lapTimes AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId INNER JOIN circuits AS T3 on T2.circuitId = T3.circuitId WHERE T1.time = ?", (lap_time,))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": result}

# Endpoint to get the average pit stop time for a specific driver
@app.get("/v1/formula_1/average_pit_stop_time_by_driver", operation_id="get_average_pit_stop_time_by_driver", summary="Retrieves the average pit stop time for a Formula 1 driver identified by their forename and surname. The endpoint calculates this average by aggregating the pit stop durations for the specified driver from the pitStops table, which is joined with the drivers table to match the driver's forename and surname.")
async def get_average_pit_stop_time_by_driver(forename: str = Query(..., description="Forename of the driver"), surname: str = Query(..., description="Surname of the driver")):
    cursor.execute("SELECT AVG(milliseconds) FROM pitStops AS T1 INNER JOIN drivers AS T2 on T1.driverId = T2.driverId WHERE T2.forename = ? AND T2.surname = ?", (forename, surname))
    result = cursor.fetchone()
    if not result:
        return {"average_pit_stop_time": []}
    return {"average_pit_stop_time": result[0]}

# Endpoint to get the average lap time in milliseconds for races in a specific country
@app.get("/v1/formula_1/average_lap_time_by_country", operation_id="get_average_lap_time_by_country", summary="Retrieves the average lap time, in milliseconds, for all races held in a specified country. This operation calculates the total lap time in milliseconds for all laps in races held in the given country, and then divides this sum by the total number of laps to obtain the average lap time.")
async def get_average_lap_time_by_country(country: str = Query(..., description="Country where the race was held")):
    cursor.execute("SELECT CAST(SUM(T1.milliseconds) AS REAL) / COUNT(T1.lap) FROM lapTimes AS T1 INNER JOIN races AS T2 on T1.raceId = T2.raceId INNER JOIN circuits AS T3 on T2.circuitId = T3.circuitId WHERE T3.country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"average_lap_time": []}
    return {"average_lap_time": result[0]}

api_calls = [
    "/v1/formula_1/driver_references_by_race_id?race_id=20",
    "/v1/formula_1/fastest_q2_surname_by_race_id?race_id=19",
    "/v1/formula_1/race_years_by_location?location=Shanghai",
    "/v1/formula_1/circuit_urls_by_name?name=Circuit%20de%20Barcelona-Catalunya",
    "/v1/formula_1/race_names_by_country?country=Germany",
    "/v1/formula_1/constructor_positions_by_name?name=Renault",
    "/v1/formula_1/race_count_by_excluded_countries_and_year?excluded_countries=Bahrain,China,Singapore,Japan,Korea,Turkey,UAE,Malaysia,Spain,Monaco,Azerbaijan,Austria,Belgium,France,Germany,Hungary,Italy,UK&year=2010",
    "/v1/formula_1/circuit_coordinates_by_race_name?race_name=Australian%20Grand%20Prix",
    "/v1/formula_1/race_times_by_circuit?circuit_name=Sepang%20International%20Circuit",
    "/v1/formula_1/constructor_nationality_by_race_points?race_id=24&points=1",
    "/v1/formula_1/qualifying_time_q1_by_driver_race?race_id=354&forename=Bruno&surname=Senna",
    "/v1/formula_1/driver_nationalities_by_qualifying_time_q2?race_id=355&q2_time=1:40%25",
    "/v1/formula_1/driver_numbers_by_qualifying_time_q3?race_id=903&q3_time=1:54%25",
    "/v1/formula_1/count_drivers_no_time_recorded?year=2007&race_name=Bahrain%20Grand%20Prix",
    "/v1/formula_1/season_url_by_race_id?race_id=901",
    "/v1/formula_1/count_drivers_with_time_by_date?date=2015-11-29",
    "/v1/formula_1/youngest_driver_with_time_by_race?race_id=592",
    "/v1/formula_1/driver_details_by_race_and_time?race_id=161&time_pattern=1:27%",
    "/v1/formula_1/fastest_lap_nationality?race_id=933",
    "/v1/formula_1/top_constructor_url_by_race?race_id=9",
    "/v1/formula_1/driver_codes_by_race_and_q3?race_id=45&q3_pattern=1:33%",
    "/v1/formula_1/race_time_by_driver_and_race?race_id=743&forename=Bruce&surname=McLaren",
    "/v1/formula_1/driver_details_by_race_year_name_position?year=2006&race_name=San%20Marino%20Grand%20Prix&position=2",
    "/v1/formula_1/count_drivers_no_time_by_date?date=2015-11-29",
    "/v1/formula_1/youngest_driver_with_time_in_race?race_id=872",
    "/v1/formula_1/fastest_lap_driver_in_race?race_id=348",
    "/v1/formula_1/fastest_lap_speed_nationality",
    "/v1/formula_1/fastest_lap_speed_percentage_difference?race_id_1=853&race_id_2=854&forename=Paul&surname=di%20Resta",
    "/v1/formula_1/percentage_drivers_with_times?date=1983-07-16",
    "/v1/formula_1/first_year_of_race?name=Singapore%20Grand%20Prix",
    "/v1/formula_1/race_names_in_year?year=2005",
    "/v1/formula_1/first_race_name",
    "/v1/formula_1/last_race_in_year?year=1999",
    "/v1/formula_1/year_with_most_races",
    "/v1/formula_1/race_names_excluding_year?year=2017&exclude_year=2000",
    "/v1/formula_1/first_race_country_location?race_name=European%20Grand%20Prix",
    "/v1/formula_1/most_recent_race_date?circuit_name=Brands%20Hatch&race_name=British%20Grand%20Prix",
    "/v1/formula_1/race_count_at_circuit?circuit_name=Silverstone%20Circuit&race_name=British%20Grand%20Prix",
    "/v1/formula_1/driver_names_by_race_year?race_name=Singapore%20Grand%20Prix&year=2010",
    "/v1/formula_1/top_driver_by_points",
    "/v1/formula_1/top_drivers_by_race_year?race_name=Chinese%20Grand%20Prix&year=2017",
    "/v1/formula_1/fastest_lap_time",
    "/v1/formula_1/average_lap_time_driver_race_year?forename=Lewis&surname=Hamilton&year=2009&race_name=Malaysian%20Grand%20Prix",
    "/v1/formula_1/percentage_non_first_finishes?surname=Hamilton&year=2010",
    "/v1/formula_1/driver_details_highest_points?min_wins=1&limit=1",
    "/v1/formula_1/youngest_driver_by_nationality?nationality=Japanese&limit=1",
    "/v1/formula_1/distinct_circuit_names_by_year_range?start_year=1990&end_year=2000&race_count=4",
    "/v1/formula_1/circuit_details_by_country_year?country=USA&year=2006",
    "/v1/formula_1/race_circuit_details_by_year_month?year=2005&month=09",
    "/v1/formula_1/race_names_by_driver_position?forename=Alex&surname=Yoong&max_position=20",
    "/v1/formula_1/total_wins_by_driver_circuit?forename=Michael&surname=Schumacher&circuit_name=Sepang%20International%20Circuit",
    "/v1/formula_1/fastest_lap_by_driver?forename=Michael&surname=Schumacher&limit=1",
    "/v1/formula_1/average_points_by_driver_year?forename=Eddie&surname=Irvine&year=2000",
    "/v1/formula_1/earliest_race_by_driver?forename=Lewis&surname=Hamilton&limit=1",
    "/v1/formula_1/distinct_circuits_by_year?year=2017",
    "/v1/formula_1/latest_lap_details",
    "/v1/formula_1/race_percentage_by_country?country=Germany&race_name=European%20Grand%20Prix",
    "/v1/formula_1/highest_latitude_circuit?name1=Silverstone%20Circuit&name2=Hockenheimring&name3=Hungaroring",
    "/v1/formula_1/circuit_reference?name=Marina%20Bay%20Street%20Circuit",
    "/v1/formula_1/highest_altitude_circuit_country",
    "/v1/formula_1/driver_count_without_code",
    "/v1/formula_1/youngest_driver_nationality",
    "/v1/formula_1/driver_surnames_by_nationality?nationality=Italian",
    "/v1/formula_1/driver_url?forename=Anthony&surname=Davidson",
    "/v1/formula_1/driver_reference?forename=Lewis&surname=Hamilton",
    "/v1/formula_1/circuit_name_by_race?year=2009&name=Spanish%20Grand%20Prix",
    "/v1/formula_1/race_years_by_circuit?name=Silverstone%20Circuit",
    "/v1/formula_1/race_date_time?year=2010&name=Abu%20Dhabi%20Grand%20Prix",
    "/v1/formula_1/race_count_by_country?country=Italy",
    "/v1/formula_1/race_dates_by_circuit?name=Circuit%20de%20Barcelona-Catalunya",
    "/v1/formula_1/circuit_url_by_race?year=2009&name=Spanish%20Grand%20Prix",
    "/v1/formula_1/race_names_by_driver_win?rank=1&forename=Lewis&surname=Hamilton",
    "/v1/formula_1/fastest_lap_speed_by_race_year?race_name=Spanish%20Grand%20Prix&year=2009",
    "/v1/formula_1/driver_participation_years?forename=Lewis&surname=Hamilton",
    "/v1/formula_1/driver_position_order?forename=Lewis&surname=Hamilton&race_name=Chinese%20Grand%20Prix&year=2008",
    "/v1/formula_1/driver_details_by_grid_position?grid=4&race_name=Australian%20Grand%20Prix&year=1989",
    "/v1/formula_1/driver_count_with_time?race_name=Australian%20Grand%20Prix&year=2008",
    "/v1/formula_1/driver_time_by_rank_race_year?rank=2&race_name=Chinese%20Grand%20Prix&year=2008",
    "/v1/formula_1/driver_details_by_race_time_year?race_name=Australian%20Grand%20Prix&time_format=_:%:__.___&year=2008",
    "/v1/formula_1/driver_count_by_nationality_race_year?race_name=Australian%20Grand%20Prix&nationality=British&year=2008",
    "/v1/formula_1/driver_count_completed_race_year?race_name=Chinese%20Grand%20Prix&year=2008",
    "/v1/formula_1/driver_total_points?forename=Lewis&surname=Hamilton",
    "/v1/formula_1/driver_avg_fastest_lap_time?forename=Lewis&surname=Hamilton",
    "/v1/formula_1/completion_percentage_race_year?race_name=Australian%20Grand%20Prix&year=2008",
    "/v1/formula_1/percentage_time_difference_champion_last_driver?race_name=Australian%20Grand%20Prix&year=2008",
    "/v1/formula_1/circuit_count_by_location_country?location=Adelaide&country=Australia",
    "/v1/formula_1/circuit_coordinates_by_country?country=USA",
    "/v1/formula_1/driver_count_by_nationality_and_birth_year?nationality=British&birth_year=1980",
    "/v1/formula_1/max_constructor_points_by_nationality?nationality=British",
    "/v1/formula_1/top_constructor_by_points",
    "/v1/formula_1/constructor_names_by_points_and_race_id?points=0&race_id=291",
    "/v1/formula_1/distinct_constructor_names_by_rank?rank=1",
    "/v1/formula_1/distinct_constructor_count_by_laps_and_nationality?laps=50&nationality=French",
    "/v1/formula_1/percentage_valid_times_by_nationality_and_year_range?nationality=Japanese&start_year=2007&end_year=2009",
    "/v1/formula_1/average_champion_time_before_year?year=1975",
    "/v1/formula_1/driver_names_by_birth_year_and_rank?birth_year=1975&rank=2",
    "/v1/formula_1/count_drivers_nationality_not_finished?nationality=Italian",
    "/v1/formula_1/fastest_lap_time_driver",
    "/v1/formula_1/fastest_lap_year_time_format?year=2009&time_format=_:%:__.___",
    "/v1/formula_1/avg_fastest_lap_speed_year_race?year=2009&race_name=Spanish%20Grand%20Prix",
    "/v1/formula_1/fastest_race",
    "/v1/formula_1/percentage_drivers_laps_years?birth_year=1985&min_laps=50&start_year=2000&end_year=2005",
    "/v1/formula_1/count_drivers_nationality_lap_time?nationality=French&max_time=120",
    "/v1/formula_1/driver_codes_nationality?nationality=American",
    "/v1/formula_1/race_ids_year?year=2009",
    "/v1/formula_1/count_drivers_standings_race?race_id=18",
    "/v1/formula_1/count_youngest_drivers_by_nationality?nationality=Dutch",
    "/v1/formula_1/count_drivers_by_nationality_and_year?nationality=British&year=1980",
    "/v1/formula_1/fastest_pit_stops_german_drivers?nationality=German&start_year=1980&end_year=1990",
    "/v1/formula_1/oldest_driver_by_nationality?nationality=German",
    "/v1/formula_1/drivers_with_fastest_lap_by_year?year=1971",
    "/v1/formula_1/longest_pit_stops_spanish_drivers?nationality=Spanish&year=1982",
    "/v1/formula_1/race_years_with_fastest_lap",
    "/v1/formula_1/race_year_longest_lap_time",
    "/v1/formula_1/fastest_lap_times_by_lap?lap=1",
    "/v1/formula_1/sum_non_null_times?status_id=2&max_race_id=100&min_race_id=50",
    "/v1/formula_1/distinct_circuit_locations?country=Austria",
    "/v1/formula_1/race_with_most_non_null_times",
    "/v1/formula_1/driver_details_qualifying?race_id=23",
    "/v1/formula_1/earliest_race_youngest_driver",
    "/v1/formula_1/count_drivers_status_nationality?status=Puncture&nationality=American",
    "/v1/formula_1/top_constructor_by_wins",
    "/v1/formula_1/slowest_lap_driver?race_name=French%20Grand%20Prix&lap=3",
    "/v1/formula_1/avg_fastest_lap_time?rank=11&year=2006&name=United%20States%20Grand%20Prix",
    "/v1/formula_1/race_times_by_name_year_and_time_format?name=Canadian%20Grand%20Prix&year=2008&time_format=_:%:__.___",
    "/v1/formula_1/constructor_refs_and_urls_by_race_name_year_and_time_format?name=Singapore%20Grand%20Prix&year=2009&time_format=_:%:__.___",
    "/v1/formula_1/driver_details_by_nationality_and_birth_year_ordered?nationality=German&start_year=1971&end_year=1985",
    "/v1/formula_1/circuit_details_by_name?name=Hungaroring",
    "/v1/formula_1/constructor_points_by_race_name_and_year_range?name=Monaco%20Grand%20Prix&start_year=1980&end_year=2010&limit=1",
    "/v1/formula_1/avg_driver_points_by_race?forename=Lewis&surname=Hamilton&name=Turkish%20Grand%20Prix",
    "/v1/formula_1/avg_races_per_year?start_year=2000&end_year=2010&total_years=10&start_date=2000-01-01&end_date=2010-12-31",
    "/v1/formula_1/most_common_nationality?limit=1",
    "/v1/formula_1/sum_wins_by_points?points=91",
    "/v1/formula_1/fastest_lap_race?limit=1",
    "/v1/formula_1/most_recent_race_location?limit=1",
    "/v1/formula_1/fastest_qualifying_driver?year=2008&circuit_name=Marina%20Bay%20Street%20Circuit&limit=1",
    "/v1/formula_1/youngest_driver_in_standings?limit=1",
    "/v1/formula_1/most_occurrences_of_status_in_race?status_id=3&race_name=Canadian%20Grand%20Prix&limit=1",
    "/v1/formula_1/oldest_driver_with_most_wins?limit=1",
    "/v1/formula_1/longest_pit_stop_duration?limit=1",
    "/v1/formula_1/pit_stop_lap_number?forename=Lewis&surname=Hamilton&year=2011&race_name=Australian%20Grand%20Prix",
    "/v1/formula_1/pit_stop_durations_by_race?year=2011&race_name=Australian%20Grand%20Prix",
    "/v1/formula_1/lap_times_by_driver?forename=Lewis&surname=Hamilton",
    "/v1/formula_1/top_20_fastest_lap_times",
    "/v1/formula_1/best_position_by_driver?forename=Lewis&surname=Hamilton",
    "/v1/formula_1/fastest_lap_time_by_race?race_name=Austrian%20Grand%20Prix",
    "/v1/formula_1/fastest_lap_time_by_country?country=Italy",
    "/v1/formula_1/fastest_lap_time_race_name?race_name=Austrian%20Grand%20Prix",
    "/v1/formula_1/pit_stop_duration_for_fastest_lap?race_name=Austrian%20Grand%20Prix",
    "/v1/formula_1/circuit_location_by_lap_time?lap_time=1:29.488",
    "/v1/formula_1/average_pit_stop_time_by_driver?forename=Lewis&surname=Hamilton",
    "/v1/formula_1/average_lap_time_by_country?country=Italy"
]
