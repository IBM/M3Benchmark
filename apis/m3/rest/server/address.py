from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/address/address.sqlite')
cursor = conn.cursor()

# Endpoint to get the sum of households in a given county
@app.get("/v1/address/sum_households_by_county", operation_id="get_sum_households", summary="Retrieves the total number of households in the specified county by aggregating data from the zip_data table based on the provided county name.")
async def get_sum_households(county: str = Query(..., description="County name")):
    cursor.execute("SELECT SUM(T1.households) FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T2.county = ?", (county,))
    result = cursor.fetchone()
    if not result:
        return {"sum_households": []}
    return {"sum_households": result[0]}

# Endpoint to get the zip code with the highest average house value in a given county
@app.get("/v1/address/zip_code_highest_avg_house_value", operation_id="get_zip_code_highest_avg_house_value", summary="Retrieves the zip code with the highest average house value within a specified county. The operation filters data based on the provided county name and returns the zip code with the highest average house value.")
async def get_zip_code_highest_avg_house_value(county: str = Query(..., description="County name")):
    cursor.execute("SELECT T1.zip_code FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T2.county = ? ORDER BY T1.avg_house_value DESC LIMIT 1", (county,))
    result = cursor.fetchone()
    if not result:
        return {"zip_code": []}
    return {"zip_code": result[0]}

# Endpoint to get the sum of male population in a given county
@app.get("/v1/address/sum_male_population_by_county", operation_id="get_sum_male_population", summary="Retrieves the total male population for a specified county. The operation calculates the sum of the male population from the zip_data table, which is joined with the country table based on the zip_code. The county name is used to filter the results.")
async def get_sum_male_population(county: str = Query(..., description="County name")):
    cursor.execute("SELECT SUM(T1.male_population) FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T2.county = ?", (county,))
    result = cursor.fetchone()
    if not result:
        return {"sum_male_population": []}
    return {"sum_male_population": result[0]}

# Endpoint to get the count of zip codes in a given county with a specific daylight savings setting
@app.get("/v1/address/count_zip_codes_by_county_daylight_savings", operation_id="get_count_zip_codes", summary="Retrieves the total number of unique zip codes within a specified county that adhere to a given daylight savings setting. The response is based on a comparison between the provided county name and the daylight savings setting against corresponding data in the system.")
async def get_count_zip_codes(county: str = Query(..., description="County name"), daylight_savings: str = Query(..., description="Daylight savings setting (e.g., 'Yes' or 'No')")):
    cursor.execute("SELECT COUNT(T1.zip_code) FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T2.county = ? AND T1.daylight_savings = ?", (county, daylight_savings))
    result = cursor.fetchone()
    if not result:
        return {"count_zip_codes": []}
    return {"count_zip_codes": result[0]}

# Endpoint to get the zip code with the highest white population in a given county
@app.get("/v1/address/zip_code_highest_white_population", operation_id="get_zip_code_highest_white_population", summary="Retrieves the zip code with the highest white population within a specified county. The operation filters data based on the provided county name and returns the zip code with the highest white population.")
async def get_zip_code_highest_white_population(county: str = Query(..., description="County name")):
    cursor.execute("SELECT T1.zip_code FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T2.county = ? ORDER BY T1.white_population DESC LIMIT 1", (county,))
    result = cursor.fetchone()
    if not result:
        return {"zip_code": []}
    return {"zip_code": result[0]}

# Endpoint to get the county with the highest average income per household
@app.get("/v1/address/county_highest_avg_income", operation_id="get_county_highest_avg_income", summary="Retrieves the county with the highest average income per household for the specified county. The operation filters data by the provided county name and calculates the average income per household. It then orders the results in descending order and returns the county with the highest average income.")
async def get_county_highest_avg_income(county: str = Query(..., description="County name")):
    cursor.execute("SELECT T2.county FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T2.county = ? GROUP BY T2.county ORDER BY T1.avg_income_per_household DESC LIMIT 1", (county,))
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get distinct counties with a specific daylight savings setting
@app.get("/v1/address/distinct_counties_by_daylight_savings", operation_id="get_distinct_counties", summary="Retrieves a list of unique counties that adhere to a specified daylight savings setting. The operation filters the data based on the provided daylight savings setting and returns the distinct counties that match the criteria.")
async def get_distinct_counties(daylight_savings: str = Query(..., description="Daylight savings setting (e.g., 'Yes' or 'No')")):
    cursor.execute("SELECT DISTINCT T2.county FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T1.daylight_savings = ?", (daylight_savings,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get distinct zip codes in a given county with more than a specified number of employees
@app.get("/v1/address/distinct_zip_codes_by_county_employees", operation_id="get_distinct_zip_codes", summary="Retrieves a list of unique zip codes within a specified county where the number of employees exceeds a given threshold. This operation is useful for identifying areas with a high concentration of employees in a particular county.")
async def get_distinct_zip_codes(county: str = Query(..., description="County name"), employees: int = Query(..., description="Number of employees")):
    cursor.execute("SELECT DISTINCT T1.zip_code FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T2.county = ? AND T1.employees > ?", (county, employees))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes": []}
    return {"zip_codes": [row[0] for row in result]}

# Endpoint to get the sum of Asian population for a given bad alias
@app.get("/v1/address/sum_asian_population_by_bad_alias", operation_id="get_sum_asian_population", summary="Retrieves the total Asian population associated with a specific bad alias. This operation aggregates population data from the zip_data table, filtering results based on the provided bad alias from the avoid table. The bad alias is used to identify relevant zip codes and their corresponding Asian population data.")
async def get_sum_asian_population(bad_alias: str = Query(..., description="Bad alias")):
    cursor.execute("SELECT SUM(T1.asian_population) FROM zip_data AS T1 INNER JOIN avoid AS T2 ON T1.zip_code = T2.zip_code WHERE T2.bad_alias = ?", (bad_alias,))
    result = cursor.fetchone()
    if not result:
        return {"sum_asian_population": []}
    return {"sum_asian_population": result[0]}

# Endpoint to get the count of zip codes for a given bad alias and time zone
@app.get("/v1/address/count_zip_codes_by_bad_alias_time_zone", operation_id="get_count_zip_codes_by_bad_alias_time_zone", summary="Retrieves the total number of zip codes associated with a specific bad alias and time zone. The bad alias and time zone are provided as input parameters, which are used to filter the zip code data and calculate the count.")
async def get_count_zip_codes_by_bad_alias_time_zone(bad_alias: str = Query(..., description="Bad alias"), time_zone: str = Query(..., description="Time zone")):
    cursor.execute("SELECT COUNT(T1.zip_code) FROM zip_data AS T1 INNER JOIN avoid AS T2 ON T1.zip_code = T2.zip_code WHERE T2.bad_alias = ? AND T1.time_zone = ?", (bad_alias, time_zone))
    result = cursor.fetchone()
    if not result:
        return {"count_zip_codes": []}
    return {"count_zip_codes": result[0]}

# Endpoint to get the bad alias for the zip code with the highest average house value
@app.get("/v1/address/bad_alias_highest_avg_house_value", operation_id="get_bad_alias_highest_avg_house_value", summary="Retrieves the bad alias associated with the zip code that has the highest average house value. This operation identifies the zip code with the maximum average house value and returns the corresponding bad alias from the 'avoid' table.")
async def get_bad_alias_highest_avg_house_value():
    cursor.execute("SELECT T2.bad_alias FROM zip_data AS T1 INNER JOIN avoid AS T2 ON T1.zip_code = T2.zip_code WHERE T1.avg_house_value = ( SELECT MAX(avg_house_value) FROM zip_data ) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"bad_alias": []}
    return {"bad_alias": result[0]}

# Endpoint to get distinct bad aliases for zip codes where the female median age is greater than a specified value
@app.get("/v1/address/distinct_bad_aliases_female_median_age", operation_id="get_distinct_bad_aliases_female_median_age", summary="Retrieves unique bad aliases associated with zip codes where the female median age surpasses a given threshold.")
async def get_distinct_bad_aliases_female_median_age(female_median_age: int = Query(..., description="Female median age")):
    cursor.execute("SELECT DISTINCT T2.bad_alias FROM zip_data AS T1 INNER JOIN avoid AS T2 ON T1.zip_code = T2.zip_code WHERE T1.female_median_age > ?", (female_median_age,))
    result = cursor.fetchall()
    if not result:
        return {"bad_aliases": []}
    return {"bad_aliases": [row[0] for row in result]}

# Endpoint to get the highest male to female population ratio for a specified county
@app.get("/v1/address/highest_male_to_female_ratio", operation_id="get_highest_male_to_female_ratio", summary="Retrieves the highest male to female population ratio for a specified county. The operation calculates this ratio by dividing the male population by the female population for each zip code within the specified county. The zip code with the highest ratio is then returned. This operation requires the county name as an input parameter.")
async def get_highest_male_to_female_ratio(county: str = Query(..., description="County name")):
    cursor.execute("SELECT CAST(T1.male_population AS REAL) / T1.female_population FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T2.county = ? AND T1.female_population <> 0 ORDER BY 1 DESC LIMIT 1", (county,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the average female median age for a specified county
@app.get("/v1/address/average_female_median_age", operation_id="get_average_female_median_age", summary="Retrieves the average female median age for a specified county. This operation calculates the average by summing the female median ages of all zip codes within the specified county and then dividing by the total number of zip codes in that county.")
async def get_average_female_median_age(county: str = Query(..., description="County name")):
    cursor.execute("SELECT SUM(T1.female_median_age) / COUNT(T1.zip_code) FROM zip_data AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T2.county = ?", (county,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get area codes for zip codes where the female median age is greater than a specified value
@app.get("/v1/address/area_codes_female_median_age", operation_id="get_area_codes_female_median_age", summary="Retrieves area codes associated with zip codes where the female median age surpasses a specified threshold. The data is grouped by area code.")
async def get_area_codes_female_median_age(female_median_age: int = Query(..., description="Female median age")):
    cursor.execute("SELECT T1.area_code FROM area_code AS T1 INNER JOIN ZIP_Data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.female_median_age > ? GROUP BY T1.area_code", (female_median_age,))
    result = cursor.fetchall()
    if not result:
        return {"area_codes": []}
    return {"area_codes": [row[0] for row in result]}

# Endpoint to get distinct aliases for a specified city
@app.get("/v1/address/distinct_aliases_city", operation_id="get_distinct_aliases_city", summary="Retrieves a unique set of aliases associated with a specific city. The operation filters the data based on the provided city name and returns the distinct aliases linked to the corresponding zip codes.")
async def get_distinct_aliases_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT DISTINCT T2.alias FROM zip_data AS T1 INNER JOIN alias AS T2 ON T1.zip_code = T2.zip_code WHERE T1.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"aliases": []}
    return {"aliases": [row[0] for row in result]}

# Endpoint to get the count of counties for a specified state
@app.get("/v1/address/count_counties_state", operation_id="get_count_counties_state", summary="Retrieves the total number of counties associated with a specified state. The operation requires the state name as input and returns the count of counties linked to that state.")
async def get_count_counties_state(state_name: str = Query(..., description="State name")):
    cursor.execute("SELECT COUNT(T2.county) FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state WHERE T1.name = ?", (state_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct zip codes for a specified state and zip code type
@app.get("/v1/address/count_distinct_zip_codes", operation_id="get_count_distinct_zip_codes", summary="Retrieves the total number of unique zip codes for a given state and zip code type. The state is identified by its abbreviation, and the zip code type can be either 'standard' or 'military'. This operation is useful for understanding the distribution of zip codes within a state and their respective types.")
async def get_count_distinct_zip_codes(state_abbreviation: str = Query(..., description="State abbreviation"), zip_code_type: str = Query(..., description="Zip code type")):
    cursor.execute("SELECT COUNT(DISTINCT T2.zip_code) FROM state AS T1 INNER JOIN zip_data AS T2 ON T1.abbreviation = T2.state WHERE T1.abbreviation = ? AND T2.type = ?", (state_abbreviation, zip_code_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get latitude and longitude for a specified area code
@app.get("/v1/address/latitude_longitude_area_code", operation_id="get_latitude_longitude_area_code", summary="Retrieves the latitude and longitude coordinates for the specified area code. The operation returns a list of unique latitude and longitude pairs associated with the given area code, which is determined by matching the area code with its corresponding zip codes and then retrieving the latitude and longitude data from the zip_data table.")
async def get_latitude_longitude_area_code(area_code: str = Query(..., description="Area code")):
    cursor.execute("SELECT T2.latitude, T2.longitude FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.area_code = ? GROUP BY T2.latitude, T2.longitude", (area_code,))
    result = cursor.fetchall()
    if not result:
        return {"coordinates": []}
    return {"coordinates": [{"latitude": row[0], "longitude": row[1]} for row in result]}

# Endpoint to get the count of zip codes for a specified state, zip code type, state name, and state abbreviation
@app.get("/v1/address/count_zip_codes_state", operation_id="get_count_zip_codes_state", summary="Retrieves the total number of zip codes that match the specified state abbreviation, zip code type, state name, and state abbreviation. The count is derived from a database query that filters results based on the provided parameters.")
async def get_count_zip_codes_state(state_abbreviation: str = Query(..., description="State abbreviation"), zip_code_type: str = Query(..., description="Zip code type"), state_name: str = Query(..., description="State name"), state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT COUNT(*) FROM state AS T1 INNER JOIN zip_data AS T2 ON T1.abbreviation = T2.state WHERE T1.abbreviation = ? AND T2.type LIKE ? AND T1.name = ? AND T2.state = ?", (state_abbreviation, zip_code_type, state_name, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the county with the highest female population
@app.get("/v1/address/county_highest_female_population", operation_id="get_county_highest_female_population", summary="Retrieves the county with the highest female population based on aggregated data from the zip_data table. The data is joined with the country table using the zip_code field, and the result is ordered in descending order by female_population before selecting the top county.")
async def get_county_highest_female_population():
    cursor.execute("SELECT T4.county FROM zip_data AS T3 INNER JOIN country AS T4 ON T3.zip_code = T4.zip_code GROUP BY T4.county ORDER BY T3.female_population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get distinct state names where division is null
@app.get("/v1/address/distinct_state_names_null_division", operation_id="get_distinct_state_names_null_division", summary="Retrieves a unique list of state names that do not have a corresponding division in the zip code data.")
async def get_distinct_state_names_null_division():
    cursor.execute("SELECT DISTINCT T2.name FROM zip_data AS T1 INNER JOIN state AS T2 ON T1.state = T2.abbreviation WHERE T1.division IS NULL")
    result = cursor.fetchall()
    if not result:
        return {"state_names": []}
    return {"state_names": [row[0] for row in result]}

# Endpoint to get the population difference between 2020 and 2010 for a specific CBSA name
@app.get("/v1/address/population_difference_cbsa", operation_id="get_population_difference_cbsa", summary="Retrieves the difference in population between 2020 and 2010 for the specified Core Based Statistical Area (CBSA). The result is based on the zip code data and is ordered by the 2020 population in descending order, with a limit of one record.")
async def get_population_difference_cbsa(cbsa_name: str = Query(..., description="CBSA name")):
    cursor.execute("SELECT T1.population_2020 - T1.population_2010 AS result_data FROM zip_data AS T1 INNER JOIN CBSA AS T2 ON T1.CBSA = T2.CBSA WHERE T2.CBSA_name = ? ORDER BY T1.population_2020 DESC LIMIT 1", (cbsa_name,))
    result = cursor.fetchone()
    if not result:
        return {"population_difference": []}
    return {"population_difference": result[0]}

# Endpoint to get distinct zip codes for a specific county and state name
@app.get("/v1/address/distinct_zip_codes_county_state", operation_id="get_distinct_zip_codes_county_state", summary="Retrieves a list of unique zip codes associated with a specific county and state. The operation filters the data based on the provided county and state names, ensuring that only distinct zip codes are returned.")
async def get_distinct_zip_codes_county_state(county: str = Query(..., description="County name"), state_name: str = Query(..., description="State name")):
    cursor.execute("SELECT DISTINCT T2.zip_code FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state WHERE T2.county = ? AND T1.name = ?", (county, state_name))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes": []}
    return {"zip_codes": [row[0] for row in result]}

# Endpoint to get the count of congress representatives for the state with the highest monthly benefits for retired workers
@app.get("/v1/address/congress_rep_count_highest_benefits", operation_id="get_congress_rep_count_highest_benefits", summary="Retrieves the total number of congressional representatives for the state with the highest monthly benefits for retired workers. The data is derived from a combination of zip code, state, and congressional data, with the state's monthly benefits for retired workers serving as the primary sorting criterion.")
async def get_congress_rep_count_highest_benefits():
    cursor.execute("SELECT COUNT(T3.cognress_rep_id) FROM zip_data AS T1 INNER JOIN state AS T2 ON T1.state = T2.abbreviation INNER JOIN congress AS T3 ON T2.abbreviation = T3.abbreviation ORDER BY T1.monthly_benefits_retired_workers DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"congress_rep_count": []}
    return {"congress_rep_count": result[0]}

# Endpoint to get the count of cities for a specific congress member with a given number of employees
@app.get("/v1/address/city_count_congress_member_employees", operation_id="get_city_count_congress_member_employees", summary="Retrieves the total number of cities where a congress member's constituents reside, based on a specified number of employees. This operation requires the first and last name of the congress member, as well as the number of employees to consider.")
async def get_city_count_congress_member_employees(first_name: str = Query(..., description="First name of the congress member"), last_name: str = Query(..., description="Last name of the congress member"), employees: int = Query(..., description="Number of employees")):
    cursor.execute("SELECT COUNT(T3.city) FROM congress AS T1 INNER JOIN state AS T2 ON T1.abbreviation = T2.abbreviation INNER JOIN zip_data AS T3 ON T2.abbreviation = T3.state WHERE T1.first_name = ? AND T1.last_name = ? AND T3.employees = ?", (first_name, last_name, employees))
    result = cursor.fetchone()
    if not result:
        return {"city_count": []}
    return {"city_count": result[0]}

# Endpoint to get the top 3 states with the highest Asian population and their congress representatives
@app.get("/v1/address/top_states_asian_population", operation_id="get_top_states_asian_population", summary="Retrieves the top three states with the highest Asian population and their respective congressional representatives. The data is grouped by state and ordered in descending order based on the total Asian population. Only the top three states are returned.")
async def get_top_states_asian_population():
    cursor.execute("SELECT t.state, T1.first_name, T1.last_name FROM zip_data AS T INNER JOIN congress AS T1 ON t.state = T1.abbreviation GROUP BY t.state ORDER BY SUM(t.asian_population) DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [{"state": row[0], "first_name": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get distinct state names for a specific county
@app.get("/v1/address/distinct_state_names_county", operation_id="get_distinct_state_names_county", summary="Retrieves a list of unique state names associated with the specified county. The operation filters the data based on the provided county name and returns the distinct state names that match the filter.")
async def get_distinct_state_names_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT DISTINCT T2.name FROM country AS T1 INNER JOIN state AS T2 ON T1.state = T2.abbreviation WHERE T1.county = ?", (county,))
    result = cursor.fetchall()
    if not result:
        return {"state_names": []}
    return {"state_names": [row[0] for row in result]}

# Endpoint to get the political parties of congress members for a specific zip code
@app.get("/v1/address/congress_parties_zip_code", operation_id="get_congress_parties_zip_code", summary="Retrieves the political parties of congress members associated with a specific zip code. The operation returns a list of unique political parties, aggregated from the congress members who represent the state containing the provided zip code.")
async def get_congress_parties_zip_code(zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT T1.party FROM congress AS T1 INNER JOIN state AS T2 ON T1.abbreviation = T2.abbreviation INNER JOIN zip_data AS T3 ON T2.abbreviation = T3.state WHERE T3.zip_code = ? GROUP BY T1.party", (zip_code,))
    result = cursor.fetchall()
    if not result:
        return {"parties": []}
    return {"parties": [row[0] for row in result]}

# Endpoint to get alias based on latitude and longitude
@app.get("/v1/address/get_alias_by_lat_long", operation_id="get_alias_by_lat_long", summary="Retrieves the alias associated with the provided latitude and longitude coordinates. This operation fetches the alias from the database by first identifying the zip code corresponding to the given latitude and longitude, and then using that zip code to look up the alias in the alias table.")
async def get_alias_by_lat_long(latitude: float = Query(..., description="Latitude of the location"), longitude: float = Query(..., description="Longitude of the location")):
    cursor.execute("SELECT T2.alias FROM zip_data AS T1 INNER JOIN alias AS T2 ON T1.zip_code = T2.zip_code WHERE T1.latitude = ? AND T1.longitude = ?", (latitude, longitude))
    result = cursor.fetchall()
    if not result:
        return {"alias": []}
    return {"alias": [row[0] for row in result]}

# Endpoint to get the top area code by over 65 population
@app.get("/v1/address/top_area_code_by_over_65", operation_id="get_top_area_code_by_over_65", summary="Retrieves the area code with the highest population of individuals aged 65 and over. This operation joins data from the zip_data and area_code tables, groups the results by area code, and orders them in descending order based on the over_65 population count. The top area code is then returned.")
async def get_top_area_code_by_over_65():
    cursor.execute("SELECT T2.area_code FROM zip_data AS T1 INNER JOIN area_code AS T2 ON T1.zip_code = T2.zip_code GROUP BY T2.area_code ORDER BY T1.over_65 DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"area_code": []}
    return {"area_code": result[0]}

# Endpoint to get the count of distinct bad aliases based on congress representative's name
@app.get("/v1/address/count_bad_aliases_by_congress_rep", operation_id="get_count_bad_aliases_by_congress_rep", summary="Retrieves the count of unique bad aliases associated with a specific congress representative, based on their first and last names. The count is determined by joining data from the zip_congress, avoid, and congress tables using the provided first and last names of the congress representative.")
async def get_count_bad_aliases_by_congress_rep(first_name: str = Query(..., description="First name of the congress representative"), last_name: str = Query(..., description="Last name of the congress representative")):
    cursor.execute("SELECT COUNT(DISTINCT T2.bad_alias) FROM zip_congress AS T1 INNER JOIN avoid AS T2 ON T1.zip_code = T2.zip_code INNER JOIN congress AS T3 ON T1.district = T3.cognress_rep_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get latitude and longitude based on area code
@app.get("/v1/address/get_lat_long_by_area_code", operation_id="get_lat_long_by_area_code", summary="Retrieves the latitude and longitude coordinates for a specific area code. The operation uses the provided area code to look up the corresponding zip code, then fetches the latitude and longitude from the zip_data table.")
async def get_lat_long_by_area_code(area_code: int = Query(..., description="Area code")):
    cursor.execute("SELECT T2.latitude, T2.longitude FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.area_code = ?", (area_code,))
    result = cursor.fetchall()
    if not result:
        return {"lat_long": []}
    return {"lat_long": [{"latitude": row[0], "longitude": row[1]} for row in result]}

# Endpoint to get zip codes based on congress representative's name
@app.get("/v1/address/get_zip_codes_by_congress_rep", operation_id="get_zip_codes_by_congress_rep", summary="Retrieves a list of zip codes associated with the specified congress representative. The operation requires the first and last name of the representative to accurately identify the corresponding zip codes.")
async def get_zip_codes_by_congress_rep(first_name: str = Query(..., description="First name of the congress representative"), last_name: str = Query(..., description="Last name of the congress representative")):
    cursor.execute("SELECT T2.zip_code FROM congress AS T1 INNER JOIN zip_congress AS T2 ON T1.cognress_rep_id = T2.district WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes": []}
    return {"zip_codes": [row[0] for row in result]}

# Endpoint to get state based on area code
@app.get("/v1/address/get_state_by_area_code", operation_id="get_state_by_area_code", summary="Retrieves the state associated with the provided area code. This operation fetches the state information from the zip_data table by joining it with the area_code table using the zip_code as the common identifier. The area code is used to filter the results and return the corresponding state.")
async def get_state_by_area_code(area_code: int = Query(..., description="Area code")):
    cursor.execute("SELECT T2.state FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.area_code = ?", (area_code,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get CBSA name, latitude, and longitude based on zip code
@app.get("/v1/address/get_cbsa_lat_long_by_zip", operation_id="get_cbsa_lat_long_by_zip", summary="Retrieves the Core Based Statistical Area (CBSA) name, latitude, and longitude associated with the provided zip code. The data is obtained by querying the CBSA and zip_data tables, filtering results based on the input zip code, and grouping the results by CBSA name, latitude, and longitude.")
async def get_cbsa_lat_long_by_zip(zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT T1.CBSA_name, T2.latitude, T2.longitude FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T2.zip_code = ? GROUP BY T1.CBSA_name, T2.latitude, T2.longitude", (zip_code,))
    result = cursor.fetchall()
    if not result:
        return {"cbsa_info": []}
    return {"cbsa_info": [{"CBSA_name": row[0], "latitude": row[1], "longitude": row[2]} for row in result]}

# Endpoint to get counties based on congress representative's name
@app.get("/v1/address/get_counties_by_congress_rep", operation_id="get_counties_by_congress_rep", summary="Retrieves a list of counties associated with a specific congress representative, identified by their first and last names. The operation filters the data based on the provided names and returns a distinct set of counties.")
async def get_counties_by_congress_rep(first_name: str = Query(..., description="First name of the congress representative"), last_name: str = Query(..., description="Last name of the congress representative")):
    cursor.execute("SELECT T1.county FROM country AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code INNER JOIN congress AS T3 ON T2.district = T3.cognress_rep_id WHERE T3.first_name = ? AND T3.last_name = ? GROUP BY T1.county", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get the average male median age based on county
@app.get("/v1/address/get_avg_male_median_age_by_county", operation_id="get_avg_male_median_age_by_county", summary="Retrieves the average male median age for a specified county. This operation calculates the sum of male median ages and divides it by the total count of median ages for the given county. The county is identified by its name.")
async def get_avg_male_median_age_by_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT SUM(T2.male_median_age) / COUNT(T2.median_age) FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.county = ?", (county,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the area code for a given city and state
@app.get("/v1/address/area_code_by_city_state", operation_id="get_area_code_by_city_state", summary="Retrieves the area code associated with a specific city and state. The operation uses the provided city and state parameters to search for corresponding zip codes and returns the area code linked to those zip codes.")
async def get_area_code_by_city_state(city: str = Query(..., description="City name"), state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT T1.area_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = ? AND T2.state = ?", (city, state))
    result = cursor.fetchall()
    if not result:
        return {"area_codes": []}
    return {"area_codes": [row[0] for row in result]}

# Endpoint to get the bad alias for a given city and state
@app.get("/v1/address/bad_alias_by_city_state", operation_id="get_bad_alias_by_city_state", summary="Retrieves the bad alias associated with a specific city and state. The operation uses the provided city and state parameters to search for a matching record in the database. The bad alias is then returned as the result.")
async def get_bad_alias_by_city_state(city: str = Query(..., description="City name"), state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT T1.bad_alias FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = ? AND T2.state = ?", (city, state))
    result = cursor.fetchall()
    if not result:
        return {"bad_aliases": []}
    return {"bad_aliases": [row[0] for row in result]}

# Endpoint to get the city and state for a given bad alias
@app.get("/v1/address/city_state_by_bad_alias", operation_id="get_city_state_by_bad_alias", summary="Retrieves the city and state associated with a given bad alias. This operation uses the provided bad alias to search for a corresponding zip code in the 'avoid' table. Once found, it joins the 'zip_data' table on the matching zip code to extract the city and state information. The results are then grouped by city and state.")
async def get_city_state_by_bad_alias(bad_alias: str = Query(..., description="Bad alias")):
    cursor.execute("SELECT T2.city, T2.state FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.bad_alias = ? GROUP BY T2.city, T2.state", (bad_alias,))
    result = cursor.fetchall()
    if not result:
        return {"city_state": []}
    return {"city_state": [{"city": row[0], "state": row[1]} for row in result]}

# Endpoint to get the alias and bad alias for a given zip code
@app.get("/v1/address/alias_bad_alias_by_zip_code", operation_id="get_alias_bad_alias_by_zip_code", summary="Retrieves the alias and associated bad alias for a specified zip code. The operation returns the alias and its corresponding bad alias from the database, based on the provided zip code.")
async def get_alias_bad_alias_by_zip_code(zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT T1.alias, T2.bad_alias FROM alias AS T1 INNER JOIN avoid AS T2 ON T1.zip_code = T2.zip_code WHERE T1.zip_code = ?", (zip_code,))
    result = cursor.fetchall()
    if not result:
        return {"alias_bad_alias": []}
    return {"alias_bad_alias": [{"alias": row[0], "bad_alias": row[1]} for row in result]}

# Endpoint to get the CBSA name and type for a given city and state
@app.get("/v1/address/cbsa_by_city_state", operation_id="get_cbsa_by_city_state", summary="Retrieves the name and type of the Core Based Statistical Area (CBSA) associated with a given city and state. The operation uses the provided city and state to search for matching records in the zip_data table, then returns the corresponding CBSA name and type from the CBSA table.")
async def get_cbsa_by_city_state(city: str = Query(..., description="City name"), state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT T1.CBSA_name, T1.CBSA_type FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T2.city = ? AND T2.state = ?", (city, state))
    result = cursor.fetchall()
    if not result:
        return {"cbsa": []}
    return {"cbsa": [{"name": row[0], "type": row[1]} for row in result]}

# Endpoint to get the city, zip code, and area code for a given minimum median age
@app.get("/v1/address/city_zip_area_code_by_median_age", operation_id="get_city_zip_area_code_by_median_age", summary="Retrieves the top 10 cities, their corresponding zip codes, and area codes where the median age is equal to or greater than the provided minimum median age.")
async def get_city_zip_area_code_by_median_age(min_median_age: int = Query(..., description="Minimum median age")):
    cursor.execute("SELECT T2.city, T2.zip_code, T1.area_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.median_age >= ? LIMIT 10", (min_median_age,))
    result = cursor.fetchall()
    if not result:
        return {"city_zip_area_code": []}
    return {"city_zip_area_code": [{"city": row[0], "zip_code": row[1], "area_code": row[2]} for row in result]}

# Endpoint to get the county for a given bad alias
@app.get("/v1/address/county_by_bad_alias", operation_id="get_county_by_bad_alias", summary="Retrieves the county associated with a given bad alias. This operation uses the provided bad alias to look up the corresponding zip code in the 'avoid' table, then joins this information with the 'country' table to find the county. The result is the county name.")
async def get_county_by_bad_alias(bad_alias: str = Query(..., description="Bad alias")):
    cursor.execute("SELECT T2.county FROM avoid AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T1.bad_alias = ?", (bad_alias,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get the distinct area codes for a given county and state name
@app.get("/v1/address/area_codes_by_county_state", operation_id="get_area_codes_by_county_state", summary="Retrieve the unique area codes associated with a specific county and state. This operation filters area codes based on the provided county and state names, returning only the distinct area codes that match the criteria.")
async def get_area_codes_by_county_state(county: str = Query(..., description="County name"), state_name: str = Query(..., description="State name")):
    cursor.execute("SELECT DISTINCT T1.area_code FROM area_code AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code INNER JOIN state AS T3 ON T2.state = T3.abbreviation WHERE T2.county = ? AND T3.name = ?", (county, state_name))
    result = cursor.fetchall()
    if not result:
        return {"area_codes": []}
    return {"area_codes": [row[0] for row in result]}

# Endpoint to get the zip code, first name, and last name of a congress representative from a specific state with the largest land area
@app.get("/v1/address/congress_rep_by_state", operation_id="get_congress_rep_by_state", summary="Retrieves the zip code, first name, and last name of the congress representative with the largest land area in the specified state. The state parameter is used to filter the results.")
async def get_congress_rep_by_state(state: str = Query(..., description="State of the congress representative")):
    cursor.execute("SELECT T2.zip_code, T1.first_name, T1.last_name FROM congress AS T1 INNER JOIN zip_congress AS T2 ON T1.cognress_rep_id = T2.district WHERE T1.state = ? ORDER BY T1.land_area DESC LIMIT 1", (state,))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of zip codes in a specific time zone
@app.get("/v1/address/zip_count_by_time_zone", operation_id="get_zip_count_by_time_zone", summary="Retrieves the total number of zip codes associated with a specific time zone. The time zone is provided as an input parameter, and the count is calculated by aggregating the matching records from the zip_data table, which is joined with the state table using the state abbreviation.")
async def get_zip_count_by_time_zone(time_zone: str = Query(..., description="Time zone of the zip codes")):
    cursor.execute("SELECT SUM(CASE WHEN T1.time_zone = ? THEN 1 ELSE 0 END) AS count FROM zip_data AS T1 INNER JOIN state AS T2 ON T2.abbreviation = T1.state WHERE T1.time_zone = ?", (time_zone, time_zone))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct cities and states in a specific CBSA area
@app.get("/v1/address/cities_states_by_cbsa", operation_id="get_cities_states_by_cbsa", summary="Retrieves a list of unique city and state combinations within a specified Core Based Statistical Area (CBSA). The operation limits the results to the top 10 records.")
async def get_cities_states_by_cbsa(cbsa_name: str = Query(..., description="Name of the CBSA area")):
    cursor.execute("SELECT DISTINCT T2.city, T2.state FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T1.CBSA_name = ? LIMIT 10", (cbsa_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the ratio of Democrat to Republican congress representatives
@app.get("/v1/address/democrat_republican_ratio", operation_id="get_democrat_republican_ratio", summary="Retrieves the ratio of congress representatives from the specified Democrat and Republican parties. This operation calculates the ratio based on the count of representatives from each party, using the provided party names as input parameters.")
async def get_democrat_republican_ratio(democrat_party: str = Query(..., description="Party name for Democrats"), republican_party: str = Query(..., description="Party name for Republicans")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.party = ? THEN 1 ELSE NULL END) AS REAL) / COUNT(CASE WHEN T2.party = ? THEN 1 ELSE NULL END) FROM zip_congress AS T1 INNER JOIN congress AS T2 ON T2.cognress_rep_id = T1.district", (democrat_party, republican_party))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the ratio of congress representatives between two states
@app.get("/v1/address/congress_rep_ratio_by_states", operation_id="get_congress_rep_ratio_by_states", summary="Retrieves the ratio of congressional representatives between two specified states. The operation compares the number of representatives from the first state to the number from the second state, and returns the calculated ratio.")
async def get_congress_rep_ratio_by_states(state1: str = Query(..., description="First state for comparison"), state2: str = Query(..., description="Second state for comparison")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN state = ? THEN cognress_rep_id ELSE NULL END) AS REAL) / COUNT(CASE WHEN state = ? THEN cognress_rep_id ELSE NULL END) FROM congress", (state1, state2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the average population per zip code
@app.get("/v1/address/average_population_per_zip", operation_id="get_average_population_per_zip", summary="Retrieves the average population across all zip codes. This operation calculates the total population and divides it by the total number of zip codes to provide a statistical overview of population distribution.")
async def get_average_population_per_zip():
    cursor.execute("SELECT CAST(SUM(population_2020) AS REAL) / COUNT(zip_code) FROM zip_data")
    result = cursor.fetchone()
    if not result:
        return {"average_population": []}
    return {"average_population": result[0]}

# Endpoint to get the male population in a specific CBSA area
@app.get("/v1/address/male_population_by_cbsa", operation_id="get_male_population_by_cbsa", summary="Retrieves the total male population residing in a specified CBSA area. The CBSA area is identified by its name, which is provided as an input parameter. The data is aggregated to provide a comprehensive count of the male population in the given area.")
async def get_male_population_by_cbsa(cbsa_name: str = Query(..., description="Name of the CBSA area")):
    cursor.execute("SELECT T2.male_population FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T1.CBSA_name = ? GROUP BY T2.male_population", (cbsa_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get CBSA names with more than a specified number of zip codes
@app.get("/v1/address/cbsa_names_by_zip_count", operation_id="get_cbsa_names_by_zip_count", summary="Retrieve the names of Core Based Statistical Areas (CBSA) that have more than a specified number of associated zip codes. The input parameter determines the minimum number of zip codes required for a CBSA to be included in the results.")
async def get_cbsa_names_by_zip_count(zip_count: int = Query(..., description="Minimum number of zip codes")):
    cursor.execute("SELECT T1.CBSA_name FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA GROUP BY T1.CBSA HAVING COUNT(T2.zip_code) > ?", (zip_count,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get bad aliases for a specific state
@app.get("/v1/address/bad_aliases_by_state", operation_id="get_bad_aliases_by_state", summary="Retrieves a list of bad aliases associated with a specific state. The state is identified by the input parameter, which filters the data to return only the relevant bad aliases.")
async def get_bad_aliases_by_state(state: str = Query(..., description="State for which to get bad aliases")):
    cursor.execute("SELECT T1.bad_alias FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.state = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the latitude and longitude of zip codes for a specific congress representative
@app.get("/v1/address/zip_coordinates_by_congress_rep", operation_id="get_zip_coordinates_by_congress_rep", summary="Retrieve the geographical coordinates (latitude and longitude) of zip codes associated with a specific congress representative. The operation requires the first and last names of the representative as input parameters to filter the data.")
async def get_zip_coordinates_by_congress_rep(first_name: str = Query(..., description="First name of the congress representative"), last_name: str = Query(..., description="Last name of the congress representative")):
    cursor.execute("SELECT T1.latitude, T1.longitude FROM zip_data AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code INNER JOIN congress AS T3 ON T2.district = T3.cognress_rep_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct states based on area code
@app.get("/v1/address/distinct_states_by_area_code", operation_id="get_distinct_states_by_area_code", summary="Retrieves a list of unique states associated with the specified area code. This operation fetches data from the area_code and zip_data tables, filtering results based on the provided area code and returning only distinct state names.")
async def get_distinct_states_by_area_code(area_code: int = Query(..., description="Area code")):
    cursor.execute("SELECT DISTINCT T2.state FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.area_code = ?", (area_code,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get congress representatives based on population
@app.get("/v1/address/congress_reps_by_population", operation_id="get_congress_reps_by_population", summary="Retrieves the names of congressional representatives from districts with a population greater than the specified value in 2020. The response is grouped by the first and last names of the representatives.")
async def get_congress_reps_by_population(population_2020: int = Query(..., description="Population in 2020")):
    cursor.execute("SELECT T3.first_name, T3.last_name FROM zip_data AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code INNER JOIN congress AS T3 ON T2.district = T3.cognress_rep_id WHERE T1.population_2020 > ? GROUP BY T3.first_name, T3.last_name", (population_2020,))
    result = cursor.fetchall()
    if not result:
        return {"representatives": []}
    return {"representatives": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get zip codes with multiple area codes in a given state
@app.get("/v1/address/zip_codes_with_multiple_area_codes", operation_id="get_zip_codes_with_multiple_area_codes", summary="Retrieves a list of zip codes within a specified state that have more than a given minimum number of associated area codes. This operation is useful for identifying areas with a high density of phone numbers from different area codes.")
async def get_zip_codes_with_multiple_area_codes(state: str = Query(..., description="State abbreviation"), min_area_codes: int = Query(..., description="Minimum number of area codes")):
    cursor.execute("SELECT T1.zip_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.state = ? GROUP BY T1.zip_code HAVING COUNT(T1.area_code) > ?", (state, min_area_codes))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes": []}
    return {"zip_codes": [row[0] for row in result]}

# Endpoint to get distinct counties based on city
@app.get("/v1/address/distinct_counties_by_city", operation_id="get_distinct_counties_by_city", summary="Retrieves a list of unique counties associated with the specified city. The operation filters data from a country table based on the provided city name and returns the distinct county names.")
async def get_distinct_counties_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT DISTINCT T1.county FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get the count of zip codes in a given CBSA
@app.get("/v1/address/count_zip_codes_in_cbsa", operation_id="get_count_zip_codes_in_cbsa", summary="Retrieves the total number of unique zip codes associated with a given Core Based Statistical Area (CBSA). The CBSA name is used to filter the data and determine the count.")
async def get_count_zip_codes_in_cbsa(cbsa_name: str = Query(..., description="CBSA name")):
    cursor.execute("SELECT COUNT(T2.zip_code) FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T1.CBSA_name = ?", (cbsa_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the zip code with the largest land area in a given county
@app.get("/v1/address/zip_code_largest_land_area_in_county", operation_id="get_zip_code_largest_land_area_in_county", summary="Retrieves the zip code with the largest land area within a specified county. The operation requires the county name as input and returns the zip code with the highest land area based on the provided county.")
async def get_zip_code_largest_land_area_in_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT T1.zip_code FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.county = ? ORDER BY T2.land_area DESC LIMIT 1", (county,))
    result = cursor.fetchone()
    if not result:
        return {"zip_code": []}
    return {"zip_code": result[0]}

# Endpoint to get the population change between 2020 and 2010 for a given congress representative
@app.get("/v1/address/population_change_by_congress_rep", operation_id="get_population_change_by_congress_rep", summary="Get the population change between 2020 and 2010 for a given congress representative")
async def get_population_change_by_congress_rep(first_name: str = Query(..., description="First name of the congress representative"), last_name: str = Query(..., description="Last name of the congress representative")):
    cursor.execute("SELECT T1.population_2020 - T1.population_2010 FROM zip_data AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code INNER JOIN congress AS T3 ON T2.district = T3.cognress_rep_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"population_change": []}
    return {"population_change": result[0]}

# Endpoint to get the percentage of Asian population in a given CBSA
@app.get("/v1/address/asian_population_percentage_in_cbsa", operation_id="get_asian_population_percentage_in_cbsa", summary="Get the percentage of Asian population in a given CBSA")
async def get_asian_population_percentage_in_cbsa(cbsa_name: str = Query(..., description="CBSA name")):
    cursor.execute("SELECT CAST(T2.asian_population AS REAL) * 100 / T2.population_2010 FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T1.CBSA_name = ?", (cbsa_name,))
    result = cursor.fetchone()
    if not result:
        return {"asian_population_percentage": []}
    return {"asian_population_percentage": result[0]}

# Endpoint to get the city with the highest Asian population in a given area code
@app.get("/v1/address/city_highest_asian_population_by_area_code", operation_id="get_city_highest_asian_population_by_area_code", summary="Retrieves the city with the highest Asian population within the specified area code. The area code is used to filter the data, and the result is determined by ranking the cities based on their Asian population in descending order.")
async def get_city_highest_asian_population_by_area_code(area_code: int = Query(..., description="Area code")):
    cursor.execute("SELECT T2.city FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.area_code = ? ORDER BY T2.asian_population DESC LIMIT 1", (area_code,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get distinct state names and abbreviations based on elevation
@app.get("/v1/address/state_names_by_elevation", operation_id="get_state_names_by_elevation", summary="Retrieve a list of unique state names and their corresponding abbreviations that match a specified elevation. This operation filters data from the state, country, and zip_data tables based on the provided elevation value.")
async def get_state_names_by_elevation(elevation: int = Query(..., description="Elevation")):
    cursor.execute("SELECT DISTINCT T1.name, T2.state FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state INNER JOIN zip_data AS T3 ON T2.zip_code = T3.zip_code WHERE T3.elevation = ?", (elevation,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [{"name": row[0], "abbreviation": row[1]} for row in result]}

# Endpoint to get alias and elevation based on zip code
@app.get("/v1/address/alias_elevation_by_zip_code", operation_id="get_alias_elevation", summary="Retrieves the alias and elevation associated with a specific zip code. The operation uses the provided zip code to look up corresponding data from the alias and zip_data tables, returning the alias and elevation values.")
async def get_alias_elevation(zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT T1.alias, T2.elevation FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.zip_code = ?", (zip_code,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get area code for the zip code with the maximum land area
@app.get("/v1/address/area_code_max_land_area", operation_id="get_area_code_max_land_area", summary="Retrieves the area code associated with the zip code that has the largest land area. This operation identifies the zip code with the maximum land area and returns the corresponding area code.")
async def get_area_code_max_land_area():
    cursor.execute("SELECT T1.area_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.land_area = ( SELECT MAX(land_area) FROM zip_data )")
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get area codes based on white population range
@app.get("/v1/address/area_code_by_white_population_range", operation_id="get_area_code_by_white_population", summary="Retrieves area codes for zip codes where the white population falls within a specified range. The range is defined by the minimum and maximum white population values provided as input parameters. This operation returns a list of area codes that meet the specified criteria.")
async def get_area_code_by_white_population(min_population: int = Query(..., description="Minimum white population"), max_population: int = Query(..., description="Maximum white population")):
    cursor.execute("SELECT T1.area_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.white_population BETWEEN ? AND ?", (min_population, max_population))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get area code and county based on city
@app.get("/v1/address/area_code_county_by_city", operation_id="get_area_code_county", summary="Retrieves the area code and county associated with a specific city. The operation uses the provided city name to search for corresponding zip codes, and then returns the area code and county linked to those zip codes.")
async def get_area_code_county(city: str = Query(..., description="City")):
    cursor.execute("SELECT T1.area_code, T2.county FROM area_code AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code INNER JOIN zip_data AS T3 ON T1.zip_code = T3.zip_code WHERE T3.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct aliases based on population in 2010
@app.get("/v1/address/distinct_aliases_by_population_2010", operation_id="get_distinct_aliases", summary="Retrieve unique aliases associated with zip codes that had a specific population in 2010. The population value is provided as an input parameter.")
async def get_distinct_aliases(population_2010: int = Query(..., description="Population in 2010")):
    cursor.execute("SELECT DISTINCT T1.alias FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.population_2010 = ?", (population_2010,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of cities based on area code and daylight savings
@app.get("/v1/address/count_cities_by_area_code_daylight_savings", operation_id="get_count_cities", summary="Retrieves the total number of cities associated with a specific area code and daylight savings status. The operation uses the provided area code and daylight savings status to filter the data and calculate the count.")
async def get_count_cities(area_code: int = Query(..., description="Area code"), daylight_savings: str = Query(..., description="Daylight savings status (Yes/No)")):
    cursor.execute("SELECT COUNT(T2.city) FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.area_code = ? AND T2.daylight_savings = ?", (area_code, daylight_savings))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average elevation based on alias
@app.get("/v1/address/avg_elevation_by_alias", operation_id="get_avg_elevation", summary="Retrieves the average elevation associated with the specified alias. This operation calculates the mean elevation from the zip_data table, filtered by the alias provided as input. The alias is used to look up the corresponding zip code in the alias table, which is then matched with the zip_data table to obtain the elevation data.")
async def get_avg_elevation(alias: str = Query(..., description="Alias")):
    cursor.execute("SELECT AVG(T2.elevation) FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.alias = ?", (alias,))
    result = cursor.fetchone()
    if not result:
        return {"avg": []}
    return {"avg": result[0]}

# Endpoint to get counties based on city
@app.get("/v1/address/counties_by_city", operation_id="get_counties_by_city", summary="Retrieves a list of counties associated with the specified city. The operation filters data from the state, country, and zip_data tables based on the provided city name, and returns a distinct list of counties.")
async def get_counties_by_city(city: str = Query(..., description="City")):
    cursor.execute("SELECT T2.county FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state INNER JOIN zip_data AS T3 ON T2.zip_code = T3.zip_code WHERE T3.city = ? GROUP BY T2.county", (city,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get aliases based on type with a limit
@app.get("/v1/address/aliases_by_type_with_limit", operation_id="get_aliases_by_type", summary="Retrieves a specified number of aliases associated with a given type. The operation filters aliases based on the provided type and returns a limited set of results.")
async def get_aliases_by_type(type: str = Query(..., description="Type"), limit: int = Query(..., description="Limit")):
    cursor.execute("SELECT T1.alias FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.type = ? LIMIT ?", (type, limit))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the difference in counts of 'P.O. Box Only' and 'Post Office' types for a given area code
@app.get("/v1/address/difference_po_box_post_office", operation_id="get_difference_po_box_post_office", summary="Retrieves the difference in the number of 'P.O. Box Only' and 'Post Office' locations within a specified area code.")
async def get_difference_po_box_post_office(area_code: int = Query(..., description="Area code")):
    cursor.execute("SELECT COUNT(CASE WHEN T2.type = 'P.O. Box Only' THEN 1 ELSE NULL END) - COUNT(CASE WHEN T2.type = 'Post Office' THEN 1 ELSE NULL END) AS DIFFERENCE FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.area_code = ?", (area_code,))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the percentage change in population from 2010 to 2020 for a given city
@app.get("/v1/address/population_change_percentage", operation_id="get_population_change_percentage", summary="Retrieves the percentage change in population from 2010 to 2020 for a specified city. The calculation is based on aggregated data from all zip codes associated with the city.")
async def get_population_change_percentage(city: str = Query(..., description="City name")):
    cursor.execute("SELECT CAST((SUM(T2.population_2020) - SUM(T2.population_2010)) AS REAL) * 100 / SUM(T2.population_2010) FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result[0]}

# Endpoint to get zip codes and cities in a given state with total beneficiaries greater than a specified number
@app.get("/v1/address/zip_codes_cities_by_state_beneficiaries", operation_id="get_zip_codes_cities_by_state_beneficiaries", summary="Retrieves a list of zip codes and their corresponding cities within a specified state where the total number of beneficiaries exceeds a given threshold. The state is identified by its name, and the minimum beneficiary count is provided as input.")
async def get_zip_codes_cities_by_state_beneficiaries(state_name: str = Query(..., description="State name"), min_beneficiaries: int = Query(..., description="Minimum number of beneficiaries")):
    cursor.execute("SELECT T2.zip_code, T2.city FROM state AS T1 INNER JOIN zip_data AS T2 ON T1.abbreviation = T2.state WHERE T1.name = ? AND T2.total_beneficiaries > ?", (state_name, min_beneficiaries))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes_cities": []}
    return {"zip_codes_cities": result}

# Endpoint to get the count of zip codes in a given county with black population greater than a specified number
@app.get("/v1/address/zip_code_count_by_county_black_population", operation_id="get_zip_code_count_by_county_black_population", summary="Retrieves the number of zip codes within a specified county where the black population exceeds a given threshold. This operation requires the county name and the minimum black population as input parameters.")
async def get_zip_code_count_by_county_black_population(county: str = Query(..., description="County name"), min_black_population: int = Query(..., description="Minimum black population")):
    cursor.execute("SELECT COUNT(T1.zip_code) FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.county = ? AND T2.black_population > ?", (county, min_black_population))
    result = cursor.fetchone()
    if not result:
        return {"zip_code_count": []}
    return {"zip_code_count": result[0]}

# Endpoint to get the city and alias for a given zip code
@app.get("/v1/address/city_alias_by_zip_code", operation_id="get_city_alias_by_zip_code", summary="Retrieves the city and its alias associated with the provided zip code. The operation fetches the alias from the alias table and matches it with the corresponding city in the zip_data table using the given zip code.")
async def get_city_alias_by_zip_code(zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT T2.city, T1.alias FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.zip_code = ?", (zip_code,))
    result = cursor.fetchall()
    if not result:
        return {"city_alias": []}
    return {"city_alias": result}

# Endpoint to get bad aliases for a given city
@app.get("/v1/address/bad_aliases_by_city", operation_id="get_bad_aliases_by_city", summary="Retrieves a list of bad aliases associated with a specific city. The operation filters the data based on the provided city name and returns the corresponding bad aliases.")
async def get_bad_aliases_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.bad_alias FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"bad_aliases": []}
    return {"bad_aliases": result}

# Endpoint to get the zip code, city, and congress representative details for the most populous district
@app.get("/v1/address/most_populous_district_congress_rep", operation_id="get_most_populous_district_congress_rep", summary="Retrieves the zip code, city, and congress representative details for the district with the highest population in the 2020 census. The data is sourced from the zip_data and congress tables, joined by the zip_congress table, and ordered by population in descending order. Only the top result is returned.")
async def get_most_populous_district_congress_rep():
    cursor.execute("SELECT T1.zip_code, T1.city, T3.first_name, T3.last_name FROM zip_data AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code INNER JOIN congress AS T3 ON T2.district = T3.cognress_rep_id GROUP BY T2.district ORDER BY T1.population_2020 DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"congress_rep_details": []}
    return {"congress_rep_details": result}

# Endpoint to get the count of zip codes in a given state with specific daylight savings and region
@app.get("/v1/address/zip_code_count_by_state_daylight_region", operation_id="get_zip_code_count_by_state_daylight_region", summary="Retrieves the total number of zip codes in a specified state, filtered by daylight savings and region. The state is identified by its name, and the daylight savings and region are provided as input parameters.")
async def get_zip_code_count_by_state_daylight_region(state_name: str = Query(..., description="State name"), daylight_savings: str = Query(..., description="Daylight savings (Yes/No)"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT COUNT(T2.zip_code) FROM state AS T1 INNER JOIN zip_data AS T2 ON T1.abbreviation = T2.state WHERE T1.name = ? AND T2.daylight_savings = ? AND T2.region = ?", (state_name, daylight_savings, region))
    result = cursor.fetchone()
    if not result:
        return {"zip_code_count": []}
    return {"zip_code_count": result[0]}

# Endpoint to get the county and zip code for a given state
@app.get("/v1/address/county_zip_code_by_state", operation_id="get_county_zip_code_by_state", summary="Retrieves the county and zip code information for a specified state. The operation uses the provided state name to look up the corresponding county and zip code data.")
async def get_county_zip_code_by_state(state_name: str = Query(..., description="State name")):
    cursor.execute("SELECT T2.county, T2.zip_code FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state WHERE T1.name = ?", (state_name,))
    result = cursor.fetchall()
    if not result:
        return {"county_zip_code": []}
    return {"county_zip_code": result}

# Endpoint to get the zip code and alias for a given city
@app.get("/v1/address/zip_code_alias_by_city", operation_id="get_zip_code_alias_by_city", summary="Retrieves the zip code and its associated alias for a specified city. The operation uses the provided city name to search for corresponding entries in the zip_data table, then joins the results with the alias table to obtain the alias information.")
async def get_zip_code_alias_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T2.zip_code, T1.alias FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"zip_code_alias": []}
    return {"zip_code_alias": result}

# Endpoint to compare the count of zip codes for two specific names
@app.get("/v1/address/congress/compare_zip_code_counts", operation_id="compare_zip_code_counts", summary="This operation compares the number of unique zip codes associated with two different individuals in the congressional database. The comparison is based on the provided first and last names. The result indicates whether the first individual has more or fewer unique zip codes than the second individual.")
async def compare_zip_code_counts(first_name_1: str = Query(..., description="First name 1"), last_name_1: str = Query(..., description="Last name 1"), first_name_2: str = Query(..., description="First name 2"), last_name_2: str = Query(..., description="Last name 2"), comparison_result_1: str = Query(..., description="Comparison result 1"), comparison_result_2: str = Query(..., description="Comparison result 2")):
    cursor.execute("SELECT CASE WHEN COUNT(CASE WHEN T1.first_name = ? AND T1.last_name = ? THEN T2.zip_code ELSE NULL END) > COUNT(CASE WHEN T1.first_name = ? AND T1.last_name = ? THEN T2.zip_code ELSE NULL END) THEN ? ELSE ? END AS COMPARE FROM congress AS T1 INNER JOIN zip_congress AS T2 ON T1.cognress_rep_id = T2.district", (first_name_1, last_name_1, first_name_2, last_name_2, comparison_result_1, comparison_result_2))
    result = cursor.fetchone()
    if not result:
        return {"compare": []}
    return {"compare": result[0]}

# Endpoint to get zip codes and CBSA names for a specific city
@app.get("/v1/address/cbsa/zip_codes_cbsa_names", operation_id="get_zip_codes_cbsa_names", summary="Retrieves a list of zip codes and their corresponding Core Based Statistical Area (CBSA) names for a specified city. The city name is used to filter the results.")
async def get_zip_codes_cbsa_names(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T2.zip_code, T1.CBSA_name FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T2.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes_cbsa_names": []}
    return {"zip_codes_cbsa_names": result}

# Endpoint to get zip codes and organizations for a specific CBSA name
@app.get("/v1/address/cbsa/zip_codes_organizations", operation_id="get_zip_codes_organizations", summary="Retrieves a list of zip codes and their associated organizations within a specific Core Based Statistical Area (CBSA). The CBSA is identified by its name, which is provided as an input parameter. The response includes the zip code and the name of the organization for each record.")
async def get_zip_codes_organizations(cbsa_name: str = Query(..., description="CBSA name")):
    cursor.execute("SELECT T2.zip_code, T2.organization FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T1.CBSA_name = ?", (cbsa_name,))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes_organizations": []}
    return {"zip_codes_organizations": result}

# Endpoint to get zip codes, first names, and last names for a specific organization
@app.get("/v1/address/zip_data/zip_codes_names", operation_id="get_zip_codes_names", summary="Retrieves a list of zip codes along with the first and last names of representatives associated with a specific organization. The organization is identified by its name.")
async def get_zip_codes_names(organization: str = Query(..., description="Organization name")):
    cursor.execute("SELECT T1.zip_code, T3.first_name, T3.last_name FROM zip_data AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code INNER JOIN congress AS T3 ON T2.district = T3.cognress_rep_id WHERE T1.organization = ?", (organization,))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes_names": []}
    return {"zip_codes_names": result}

# Endpoint to get the percentage of zip codes of a specific type in a given state
@app.get("/v1/address/state/zip_code_type_percentage", operation_id="get_zip_code_type_percentage", summary="Retrieves the percentage of a specific type of zip code within a given state. The operation calculates this percentage by comparing the count of the specified zip code type to the total count of all zip codes in the state.")
async def get_zip_code_type_percentage(zip_code_type: str = Query(..., description="Type of zip code"), state_name: str = Query(..., description="State name")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.type = ? THEN T2.zip_code ELSE NULL END) AS REAL) * 100 / COUNT(T2.zip_code) FROM state AS T1 INNER JOIN zip_data AS T2 ON T1.abbreviation = T2.state WHERE T1.name = ?", (zip_code_type, state_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get zip codes for a specific house
@app.get("/v1/address/congress/zip_codes_by_house", operation_id="get_zip_codes_by_house", summary="Retrieves a list of unique zip codes associated with a specific house in the congressional district. The house name is used to filter the results.")
async def get_zip_codes_by_house(house: str = Query(..., description="House name")):
    cursor.execute("SELECT T2.zip_code FROM congress AS T1 INNER JOIN zip_congress AS T2 ON T1.cognress_rep_id = T2.district WHERE T1.House = ? GROUP BY T2.zip_code", (house,))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes": []}
    return {"zip_codes": result}

# Endpoint to get the city with the highest count of bad aliases
@app.get("/v1/address/avoid/city_with_most_bad_aliases", operation_id="get_city_with_most_bad_aliases", summary="Retrieves the city with the most occurrences of bad aliases. This operation identifies the city with the highest count of bad aliases by joining the 'avoid' and 'zip_data' tables on the 'zip_code' field. The result is ordered by the count of bad aliases in descending order and limited to the top entry.")
async def get_city_with_most_bad_aliases():
    cursor.execute("SELECT T2.city FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code GROUP BY T1.bad_alias ORDER BY COUNT(T1.zip_code) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get counties for a specific state
@app.get("/v1/address/state/counties_by_state", operation_id="get_counties_by_state", summary="Retrieves a list of counties within a specified state. The operation filters the counties based on the provided state name and returns them grouped by county.")
async def get_counties_by_state(state_name: str = Query(..., description="State name")):
    cursor.execute("SELECT T2.county FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state WHERE T1.name = ? GROUP BY T2.county", (state_name,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": result}

# Endpoint to get the CBSA name with the highest average house value
@app.get("/v1/address/cbsa/highest_avg_house_value", operation_id="get_highest_avg_house_value", summary="Retrieves the name of the Core Based Statistical Area (CBSA) with the highest average house value. This operation identifies the CBSA with the maximum average house value from the available data and returns its name. The result provides valuable insights into the real estate market trends and property valuation within the CBSA.")
async def get_highest_avg_house_value():
    cursor.execute("SELECT DISTINCT T1.CBSA_name FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T2.avg_house_value = ( SELECT MAX(avg_house_value) FROM zip_data ) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"cbsa_name": []}
    return {"cbsa_name": result[0]}

# Endpoint to get the total population of a county
@app.get("/v1/address/total_population_by_county", operation_id="get_total_population_by_county", summary="Retrieves the total population of a specified county by aggregating the population data from the associated zip codes.")
async def get_total_population_by_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT SUM(T2.population_2010) FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.county = ?", (county,))
    result = cursor.fetchone()
    if not result:
        return {"total_population": []}
    return {"total_population": result[0]}

# Endpoint to get the zip code with the highest Asian population
@app.get("/v1/address/zip_code_highest_asian_population", operation_id="get_zip_code_highest_asian_population", summary="Retrieves the zip code with the highest Asian population by aggregating and sorting data from the area_code and zip_data tables.")
async def get_zip_code_highest_asian_population():
    cursor.execute("SELECT T1.zip_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code GROUP BY T2.asian_population ORDER BY T2.asian_population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"zip_code": []}
    return {"zip_code": result[0]}

# Endpoint to get cities based on CBSA type
@app.get("/v1/address/cities_by_cbsa_type", operation_id="get_cities_by_cbsa_type", summary="Retrieves a list of cities that belong to the specified Core Based Statistical Area (CBSA) type. The CBSA type is used to filter the cities, which are then returned in the response.")
async def get_cities_by_cbsa_type(cbsa_type: str = Query(..., description="CBSA type")):
    cursor.execute("SELECT T2.city FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T1.CBSA_type = ?", (cbsa_type,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the state with the most counties
@app.get("/v1/address/state_with_most_counties", operation_id="get_state_with_most_counties", summary="Retrieves the name of the state with the highest number of counties. This operation returns the state with the most counties based on the data in the backend database.")
async def get_state_with_most_counties():
    cursor.execute("SELECT T1.name FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state GROUP BY T2.state ORDER BY COUNT(T2.county) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"state": []}
    return {"state": result[0]}

# Endpoint to get the count of zip codes in a specific congressional district
@app.get("/v1/address/zip_code_count_by_district", operation_id="get_zip_code_count_by_district", summary="Retrieves the total number of unique zip codes associated with a specified congressional district. This operation provides a quantitative measure of the district's geographical coverage, which can be useful for various analytical purposes.")
async def get_zip_code_count_by_district(district: str = Query(..., description="Congressional district")):
    cursor.execute("SELECT SUM(CASE WHEN T2.district = ? THEN 1 ELSE 0 END) FROM zip_data AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code", (district,))
    result = cursor.fetchone()
    if not result:
        return {"zip_code_count": []}
    return {"zip_code_count": result[0]}

# Endpoint to get the average income per household based on bad alias
@app.get("/v1/address/avg_income_by_bad_alias", operation_id="get_avg_income_by_bad_alias", summary="Retrieves the average income per household for a given bad alias. This operation calculates the average income based on the provided bad alias, which is used to filter the data from the zip_data table. The result is a single value representing the average income per household for the specified bad alias.")
async def get_avg_income_by_bad_alias(bad_alias: str = Query(..., description="Bad alias")):
    cursor.execute("SELECT T2.avg_income_per_household FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.bad_alias = ?", (bad_alias,))
    result = cursor.fetchone()
    if not result:
        return {"avg_income": []}
    return {"avg_income": result[0]}

# Endpoint to get states with above average female population
@app.get("/v1/address/states_above_avg_female_population", operation_id="get_states_above_avg_female_population", summary="Retrieves a list of states where the female population is above the average female population across all zip codes. This operation uses data from the state and zip_data tables, filtering for states where the female population surpasses the overall average.")
async def get_states_above_avg_female_population():
    cursor.execute("SELECT DISTINCT T2.state FROM state AS T1 INNER JOIN zip_data AS T2 ON T1.abbreviation = T2.state WHERE T2.female_population > ( SELECT AVG(female_population) FROM zip_data )")
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get the percentage of households in a specific county
@app.get("/v1/address/percentage_households_by_county", operation_id="get_percentage_households_by_county", summary="Retrieves the percentage of households in a specified county by comparing the total number of households in the given county to the total number of households across all counties. The calculation is based on data from the country and zip_data tables, which are joined on the zip_code field.")
async def get_percentage_households_by_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.county = ? THEN T2.households ELSE 0 END) AS REAL) * 100 / SUM(T2.households) FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code", (county,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_households": []}
    return {"percentage_households": result[0]}

# Endpoint to get CBSA name and type based on city
@app.get("/v1/address/cbsa_info_by_city", operation_id="get_cbsa_info_by_city", summary="Retrieves the name and type of the Core Based Statistical Area (CBSA) associated with the specified city. The operation returns a list of unique CBSA names and types that correspond to the provided city, based on the data from the CBSA and zip_data tables.")
async def get_cbsa_info_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.CBSA_name, T1.CBSA_type FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T2.city = ? GROUP BY T1.CBSA_name, T1.CBSA_type", (city,))
    result = cursor.fetchall()
    if not result:
        return {"cbsa_info": []}
    return {"cbsa_info": [{"CBSA_name": row[0], "CBSA_type": row[1]} for row in result]}

# Endpoint to get the county based on city
@app.get("/v1/address/county_by_city", operation_id="get_county_by_city", summary="Retrieves the county associated with the specified city. This operation uses the provided city name to search for matching records in the zip_data table, then returns the corresponding county from the country table.")
async def get_county_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.county FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get the count of distinct cities based on congress representative's first and last name
@app.get("/v1/address/count_distinct_cities_by_congress_rep", operation_id="get_count_distinct_cities_by_congress_rep", summary="Retrieves the number of unique cities represented by a specific congress representative, identified by their first and last name. The count is determined by matching the representative's district with the corresponding zip codes and their associated cities.")
async def get_count_distinct_cities_by_congress_rep(first_name: str = Query(..., description="First name of the congress representative"), last_name: str = Query(..., description="Last name of the congress representative")):
    cursor.execute("SELECT COUNT(DISTINCT T1.city) FROM zip_data AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code INNER JOIN congress AS T3 ON T2.district = T3.cognress_rep_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get congress representatives based on city
@app.get("/v1/address/congress_reps_by_city", operation_id="get_congress_reps_by_city", summary="Retrieves the first and last names of congressional representatives associated with the specified city. The operation uses the provided city name to search for corresponding zip codes and congressional districts, then returns the names of the representatives for those districts.")
async def get_congress_reps_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T3.first_name, T3.last_name FROM zip_data AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code INNER JOIN congress AS T3 ON T2.district = T3.cognress_rep_id WHERE T1.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"congress_reps": []}
    return {"congress_reps": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the state with the highest number of bad aliases
@app.get("/v1/address/state_with_most_bad_aliases", operation_id="get_state_with_most_bad_aliases", summary="Retrieves the state with the highest count of bad aliases. This operation identifies the state with the most bad aliases by joining data from the 'avoid' and 'zip_data' tables using the 'zip_code' field. The result is determined by grouping the data by state and ordering it in descending order based on the count of bad aliases. Only the top state is returned.")
async def get_state_with_most_bad_aliases():
    cursor.execute("SELECT T2.state FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code GROUP BY T2.state ORDER BY COUNT(T1.bad_alias) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"state": []}
    return {"state": result[0]}

# Endpoint to get the difference in the count of bad aliases between two cities
@app.get("/v1/address/bad_alias_difference_between_cities", operation_id="get_bad_alias_difference_between_cities", summary="Retrieves the difference in the number of bad aliases between two specified cities. The data is obtained by comparing the count of bad aliases in the first city with that of the second city. The result provides insights into the disparity in the occurrence of bad aliases between the two cities.")
async def get_bad_alias_difference_between_cities(city1: str = Query(..., description="First city name"), city2: str = Query(..., description="Second city name")):
    cursor.execute("SELECT COUNT(CASE WHEN T2.city = ? THEN T1.bad_alias ELSE NULL END) - COUNT(CASE WHEN T2.city = ? THEN T1.bad_alias ELSE NULL END) AS DIFFERENCE FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code", (city1, city2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get states with a specific CBSA type and count greater than a specified number
@app.get("/v1/address/states_by_cbsa_type_and_count", operation_id="get_states_by_cbsa_type_and_count", summary="Retrieves states that have a specified Core Based Statistical Area (CBSA) type and a count of that type greater than a given threshold. The operation filters the data based on the provided CBSA type and count, returning only the states that meet the criteria.")
async def get_states_by_cbsa_type_and_count(cbsa_type: str = Query(..., description="CBSA type"), count: int = Query(..., description="Count threshold")):
    cursor.execute("SELECT T2.state FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T1.CBSA_type = ? GROUP BY T2.state HAVING COUNT(T1.CBSA_type) > ?", (cbsa_type, count))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get the total population in 2020 for a given county
@app.get("/v1/address/total_population_2020_by_county", operation_id="get_total_population_2020_by_county", summary="Retrieves the total population in 2020 for a specified county by aggregating population data from associated zip codes.")
async def get_total_population_2020_by_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT SUM(T2.population_2020) FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.county = ?", (county,))
    result = cursor.fetchone()
    if not result:
        return {"total_population": []}
    return {"total_population": result[0]}

# Endpoint to get distinct counties with population in 2010 greater than a specified number
@app.get("/v1/address/distinct_counties_by_population_2010", operation_id="get_distinct_counties_by_population_2010", summary="Retrieves a list of unique counties where the population in 2010 surpassed a specified threshold. This operation filters counties based on their population data from 2010, which is compared to a user-defined threshold. The result is a set of distinct counties that meet the specified population criteria.")
async def get_distinct_counties_by_population_2010(population_2010: int = Query(..., description="Population threshold in 2010")):
    cursor.execute("SELECT DISTINCT T1.county FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.population_2010 > ?", (population_2010,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get the county with the highest number of households
@app.get("/v1/address/county_with_most_households", operation_id="get_county_with_most_households", summary="Retrieves the county with the highest number of households. This operation aggregates data from the country and zip_data tables, grouping by county and ordering by the number of households in descending order. The county with the most households is then returned.")
async def get_county_with_most_households():
    cursor.execute("SELECT T1.county FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code GROUP BY T1.county ORDER BY T2.households DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get the percentage of zip codes with households greater than a specified number
@app.get("/v1/address/percentage_zip_codes_households", operation_id="get_percentage_zip_codes_households", summary="Retrieves the percentage of zip codes that have a number of households exceeding a specified threshold. This operation calculates the ratio of zip codes with more than the given number of households to the total number of zip codes in the country. The result is returned as a real number multiplied by 100 to represent a percentage.")
async def get_percentage_zip_codes_households(households: int = Query(..., description="Number of households")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.households > ? THEN T1.zip_code ELSE NULL END) AS REAL) * 100 / COUNT(T1.zip_code) FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code", (households,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of zip codes with a specific type in a given county
@app.get("/v1/address/percentage_zip_codes_type_county", operation_id="get_percentage_zip_codes_type_county", summary="Retrieves the percentage of zip codes of a specific type within a given county. The operation calculates this percentage by comparing the count of zip codes of the specified type in the county to the total count of zip codes in the county.")
async def get_percentage_zip_codes_type_county(type: str = Query(..., description="Type of zip code"), county: str = Query(..., description="County")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.type = ? THEN T1.zip_code ELSE NULL END) AS REAL) * 100 / COUNT(T1.zip_code) FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.county = ?", (type, county))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct counties for a given area code and zip code type
@app.get("/v1/address/distinct_counties_area_code_type", operation_id="get_distinct_counties_area_code_type", summary="Retrieves a list of unique counties associated with a specific area code and zip code type. The operation filters the data based on the provided area code and zip code type, ensuring that only relevant counties are returned.")
async def get_distinct_counties_area_code_type(area_code: str = Query(..., description="Area code"), type: str = Query(..., description="Type of zip code")):
    cursor.execute("SELECT DISTINCT T2.county FROM area_code AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code INNER JOIN zip_data AS T3 ON T1.zip_code = T3.zip_code WHERE T1.area_code = ? AND T3.type = ?", (area_code, type))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get the elevation for a given alias
@app.get("/v1/address/elevation_by_alias", operation_id="get_elevation_by_alias", summary="Retrieves the elevation data associated with a specific alias. The alias is used to look up the corresponding zip code, which is then used to fetch the elevation from the zip_data table.")
async def get_elevation_by_alias(alias: str = Query(..., description="Alias")):
    cursor.execute("SELECT T2.elevation FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.alias = ?", (alias,))
    result = cursor.fetchone()
    if not result:
        return {"elevation": []}
    return {"elevation": result[0]}

# Endpoint to get the count of area codes with a specific daylight savings setting
@app.get("/v1/address/count_area_codes_daylight_savings", operation_id="get_count_area_codes_daylight_savings", summary="Retrieves the total number of area codes that adhere to a specified daylight savings setting. This operation considers the daylight savings setting of the zip codes associated with each area code.")
async def get_count_area_codes_daylight_savings(daylight_savings: str = Query(..., description="Daylight savings setting (Yes/No)")):
    cursor.execute("SELECT COUNT(T1.area_code) FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.daylight_savings = ?", (daylight_savings,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the county and area code for a given zip code
@app.get("/v1/address/county_area_code_by_zip", operation_id="get_county_area_code_by_zip", summary="Retrieves the county and area code associated with the provided zip code. This operation fetches the relevant data from the database using the specified zip code as a reference.")
async def get_county_area_code_by_zip(zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT T2.county, T1.area_code FROM area_code AS T1 INNER JOIN country AS T2 ON T1.zip_code = T2.zip_code WHERE T1.zip_code = ?", (zip_code,))
    result = cursor.fetchone()
    if not result:
        return {"county": [], "area_code": []}
    return {"county": result[0], "area_code": result[1]}

# Endpoint to get distinct types for a given alias
@app.get("/v1/address/distinct_types_by_alias", operation_id="get_distinct_types_by_alias", summary="Retrieves a list of unique address types associated with a specific alias. The alias is used to filter the results, ensuring that only relevant address types are returned.")
async def get_distinct_types_by_alias(alias: str = Query(..., description="Alias")):
    cursor.execute("SELECT DISTINCT T2.type FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.alias = ?", (alias,))
    result = cursor.fetchall()
    if not result:
        return {"types": []}
    return {"types": [row[0] for row in result]}

# Endpoint to get the city for a given state name and county
@app.get("/v1/address/city_by_state_county", operation_id="get_city_by_state_county", summary="Get the city for a given state name and county")
async def get_city_by_state_county(state_name: str = Query(..., description="State name"), county: str = Query(..., description="County")):
    cursor.execute("SELECT T3.city FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state INNER JOIN zip_data AS T3 ON T2.zip_code = T3.zip_code WHERE T1.name = ? AND T2.county = ?", (state_name, county))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the area code with the maximum water area
@app.get("/v1/address/area_code_max_water_area", operation_id="get_area_code_max_water_area", summary="Retrieves the area code associated with the zip code that has the largest water area. This operation identifies the zip code with the maximum water area and returns the corresponding area code. The data is sourced from the area_code and zip_data tables, which are joined based on their zip code.")
async def get_area_code_max_water_area():
    cursor.execute("SELECT T1.area_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.water_area = ( SELECT MAX(water_area) FROM zip_data )")
    result = cursor.fetchone()
    if not result:
        return {"area_code": []}
    return {"area_code": result[0]}

# Endpoint to get the alias with the maximum population in 2020
@app.get("/v1/address/alias_max_population_2020", operation_id="get_alias_max_population_2020", summary="Retrieves the alias associated with the highest population in 2020. This operation identifies the zip code with the maximum population in 2020 and returns the corresponding alias. The result provides insight into the most populated area based on the 2020 census data.")
async def get_alias_max_population_2020():
    cursor.execute("SELECT T1.alias FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.population_2020 = ( SELECT MAX(population_2020) FROM zip_data )")
    result = cursor.fetchone()
    if not result:
        return {"alias": []}
    return {"alias": result[0]}

# Endpoint to get elevation data based on state name and county
@app.get("/v1/address/elevation_by_state_and_county", operation_id="get_elevation_by_state_and_county", summary="Retrieves the elevation data for a specific state and county. The operation uses the provided state name and county name to filter the data and returns the elevation data grouped by elevation levels.")
async def get_elevation_by_state_and_county(state_name: str = Query(..., description="Name of the state"), county: str = Query(..., description="Name of the county")):
    cursor.execute("SELECT T3.elevation FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state INNER JOIN zip_data AS T3 ON T2.zip_code = T3.zip_code WHERE T1.name = ? AND T2.county = ? GROUP BY T3.elevation", (state_name, county))
    result = cursor.fetchall()
    if not result:
        return {"elevations": []}
    return {"elevations": [row[0] for row in result]}

# Endpoint to get area codes with the highest Hispanic population
@app.get("/v1/address/area_codes_with_highest_hispanic_population", operation_id="get_area_codes_with_highest_hispanic_population", summary="Retrieves the area codes that have the highest Hispanic population based on the zip code data. The data is determined by finding the maximum Hispanic population in the zip code data and then identifying the area codes associated with those zip codes.")
async def get_area_codes_with_highest_hispanic_population():
    cursor.execute("SELECT T1.area_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.hispanic_population = ( SELECT MAX(hispanic_population) FROM zip_data )")
    result = cursor.fetchall()
    if not result:
        return {"area_codes": []}
    return {"area_codes": [row[0] for row in result]}

# Endpoint to get aliases based on Asian population
@app.get("/v1/address/aliases_by_asian_population", operation_id="get_aliases_by_asian_population", summary="Retrieves aliases for zip codes with a specified Asian population. The operation filters zip codes based on the provided Asian population and returns their corresponding aliases. The input parameter determines the Asian population threshold for the zip codes.")
async def get_aliases_by_asian_population(asian_population: int = Query(..., description="Asian population in the zip code")):
    cursor.execute("SELECT T1.alias FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.asian_population = ?", (asian_population,))
    result = cursor.fetchall()
    if not result:
        return {"aliases": []}
    return {"aliases": [row[0] for row in result]}

# Endpoint to get the average white population for a specific area code
@app.get("/v1/address/avg_white_population_by_area_code", operation_id="get_avg_white_population_by_area_code", summary="Retrieves the average white population for a given area code. This operation calculates the average white population by aggregating data from the zip codes associated with the specified area code.")
async def get_avg_white_population_by_area_code(area_code: int = Query(..., description="Area code")):
    cursor.execute("SELECT AVG(T2.white_population) FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.area_code = ?", (area_code,))
    result = cursor.fetchone()
    if not result:
        return {"average_white_population": []}
    return {"average_white_population": result[0]}

# Endpoint to get the percentage of zip codes with county FIPS less than a specified value for a given alias
@app.get("/v1/address/percentage_zip_codes_by_county_fips_and_alias", operation_id="get_percentage_zip_codes_by_county_fips_and_alias", summary="Retrieves the percentage of zip codes associated with an alias that have a county FIPS value less than the provided value. This operation calculates the ratio of zip codes with a county FIPS less than the specified value to the total number of zip codes linked to the alias.")
async def get_percentage_zip_codes_by_county_fips_and_alias(county_fips: int = Query(..., description="County FIPS value"), alias: str = Query(..., description="Alias")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.county_fips < ? THEN T2.zip_code ELSE NULL END) AS REAL) * 100 / COUNT(T2.zip_code) FROM alias AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.alias = ?", (county_fips, alias))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get counties with population greater than a specified percentage of the average population
@app.get("/v1/address/counties_by_population_percentage", operation_id="get_counties_by_population_percentage", summary="Retrieves counties where the population exceeds a given percentage of the average population. The operation calculates the average population from the zip_data table and compares it to the population of each county. Counties with a population surpassing the specified percentage of the average are returned.")
async def get_counties_by_population_percentage(population_percentage: float = Query(..., description="Percentage of the average population")):
    cursor.execute("SELECT T1.county FROM country AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.population_2020 > ? * ( SELECT AVG(population_2020) FROM zip_data )", (population_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get the count of zip codes for a specific congress representative
@app.get("/v1/address/zip_code_count_by_congress_rep", operation_id="get_zip_code_count_by_congress_rep", summary="Retrieves the total number of zip codes associated with a specific congress representative. The operation requires the first and last name of the representative as input parameters to accurately determine the count.")
async def get_zip_code_count_by_congress_rep(first_name: str = Query(..., description="First name of the congress representative"), last_name: str = Query(..., description="Last name of the congress representative")):
    cursor.execute("SELECT COUNT(T2.zip_code) FROM congress AS T1 INNER JOIN zip_congress AS T2 ON T1.cognress_rep_id = T2.district WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"zip_code_count": []}
    return {"zip_code_count": result[0]}

# Endpoint to get zip codes, latitude, and longitude for a specific CBSA name
@app.get("/v1/address/zip_codes_by_cbsa_name", operation_id="get_zip_codes_by_cbsa_name", summary="Retrieves a list of zip codes along with their corresponding latitude and longitude for a specified Core Based Statistical Area (CBSA). The CBSA is identified by its name, which is provided as an input parameter.")
async def get_zip_codes_by_cbsa_name(cbsa_name: str = Query(..., description="CBSA name")):
    cursor.execute("SELECT T2.zip_code, T2.latitude, T2.longitude FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T1.CBSA_name = ?", (cbsa_name,))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes": []}
    return {"zip_codes": [{"zip_code": row[0], "latitude": row[1], "longitude": row[2]} for row in result]}

# Endpoint to get zip codes, city, latitude, and longitude for a specific bad alias
@app.get("/v1/address/zip_codes_by_bad_alias", operation_id="get_zip_codes_by_bad_alias", summary="Retrieves the zip code, city, latitude, and longitude associated with a specific bad alias. The bad alias is used to look up the corresponding zip code in the avoid table, which is then joined with the zip_data table to obtain the additional location information.")
async def get_zip_codes_by_bad_alias(bad_alias: str = Query(..., description="Bad alias")):
    cursor.execute("SELECT T1.zip_code, T2.city, T2.latitude, T2.longitude FROM avoid AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T1.bad_alias = ?", (bad_alias,))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes": []}
    return {"zip_codes": [{"zip_code": row[0], "city": row[1], "latitude": row[2], "longitude": row[3]} for row in result]}

# Endpoint to get the count of zip codes in a given state
@app.get("/v1/address/count_zip_codes_by_state", operation_id="get_count_zip_codes_by_state", summary="Retrieves the total number of unique zip codes associated with a specified state. The state is identified by its name, and the count is determined based on the number of distinct zip codes linked to the state's congressional districts.")
async def get_count_zip_codes_by_state(state: str = Query(..., description="State name")):
    cursor.execute("SELECT COUNT(T2.zip_code) FROM congress AS T1 INNER JOIN zip_congress AS T2 ON T1.cognress_rep_id = T2.district WHERE T1.state = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct zip codes and counties in a given state
@app.get("/v1/address/count_distinct_zip_codes_and_counties_by_state", operation_id="get_count_distinct_zip_codes_and_counties_by_state", summary="Retrieves the count of distinct zip codes and counties within a specified state. The operation requires the state name as input and returns the total number of unique zip codes and counties found in the corresponding state.")
async def get_count_distinct_zip_codes_and_counties_by_state(state_name: str = Query(..., description="State name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.zip_code), COUNT(DISTINCT T2.county) FROM state AS T1 INNER JOIN country AS T2 ON T1.abbreviation = T2.state WHERE T1.name = ?", (state_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result}

# Endpoint to get zip codes and area codes based on type and elevation
@app.get("/v1/address/zip_codes_area_codes_by_type_elevation", operation_id="get_zip_codes_area_codes_by_type_elevation", summary="Get zip codes and area codes based on type and elevation")
async def get_zip_codes_area_codes_by_type_elevation(type: str = Query(..., description="Type of zip code"), elevation: int = Query(..., description="Elevation")):
    cursor.execute("SELECT T1.zip_code, T1.area_code FROM area_code AS T1 INNER JOIN zip_data AS T2 ON T1.zip_code = T2.zip_code WHERE T2.type = ? AND T2.elevation > ?", (type, elevation))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the district of a given city
@app.get("/v1/address/district_by_city", operation_id="get_district_by_city", summary="Retrieves the district associated with a specified city. The operation uses the provided city name to search for corresponding records in the zip_data table, then joins these records with the zip_congress table to determine the district. The result is a single district value.")
async def get_district_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T2.district FROM zip_data AS T1 INNER JOIN zip_congress AS T2 ON T1.zip_code = T2.zip_code WHERE T1.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"districts": []}
    return {"districts": result}

# Endpoint to get the CBSA name with the maximum number of employees
@app.get("/v1/address/cbsa_name_with_max_employees", operation_id="get_cbsa_name_with_max_employees", summary="Retrieves the name of the Core Based Statistical Area (CBSA) with the highest number of employees. This operation identifies the CBSA with the maximum employee count by joining the CBSA and zip_data tables on the CBSA field and filtering for the maximum employee count.")
async def get_cbsa_name_with_max_employees():
    cursor.execute("SELECT T1.CBSA_name FROM CBSA AS T1 INNER JOIN zip_data AS T2 ON T1.CBSA = T2.CBSA WHERE T2.employees = ( SELECT MAX(employees) FROM zip_data )")
    result = cursor.fetchall()
    if not result:
        return {"cbsa_names": []}
    return {"cbsa_names": result}

# Endpoint to get the count of zip codes in a given state and type
@app.get("/v1/address/count_zip_codes_by_state_and_type", operation_id="get_count_zip_codes_by_state_and_type", summary="Retrieves the total number of zip codes of a specified type within a given state. The operation requires the state name and the type of zip code as input parameters.")
async def get_count_zip_codes_by_state_and_type(state_name: str = Query(..., description="State name"), zip_type: str = Query(..., description="Type of zip code")):
    cursor.execute("SELECT COUNT(T2.zip_code) FROM state AS T1 INNER JOIN zip_data AS T2 ON T1.abbreviation = T2.state WHERE T1.name = ? AND T2.type = ?", (state_name, zip_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average number of total beneficiaries per zip code in a given state
@app.get("/v1/address/average_beneficiaries_per_zip_code", operation_id="get_average_beneficiaries_per_zip_code", summary="Retrieves the average number of total beneficiaries per zip code within a specified state. The calculation is based on the sum of total beneficiaries across all zip codes in the state, divided by the total number of unique zip codes in the state.")
async def get_average_beneficiaries_per_zip_code(state_name: str = Query(..., description="State name")):
    cursor.execute("SELECT CAST(SUM(T2.total_beneficiaries) AS REAL) / COUNT(T2.zip_code) FROM state AS T1 INNER JOIN zip_data AS T2 ON T1.abbreviation = T2.state WHERE T1.name = ?", (state_name,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of congress members from a specific party and the count of congress members from a specific state
@app.get("/v1/address/congress_party_state_stats", operation_id="get_congress_party_state_stats", summary="Retrieves the percentage of congress members from a specified party and the count of congress members from a specified state. The operation calculates these statistics by querying the congress and zip_congress tables, considering the provided party and state names as input parameters.")
async def get_congress_party_state_stats(party: str = Query(..., description="Party name"), state: str = Query(..., description="State name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.party = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*), SUM(CASE WHEN T1.state = ? THEN 1 ELSE 0 END) FROM congress AS T1 INNER JOIN zip_congress AS T2 ON T1.cognress_rep_id = T2.district", (party, state))
    result = cursor.fetchone()
    if not result:
        return {"stats": []}
    return {"stats": result}

api_calls = [
    "/v1/address/sum_households_by_county?county=ARECIBO",
    "/v1/address/zip_code_highest_avg_house_value?county=ARECIBO",
    "/v1/address/sum_male_population_by_county?county=ARECIBO",
    "/v1/address/count_zip_codes_by_county_daylight_savings?county=DELAWARE&daylight_savings=Yes",
    "/v1/address/zip_code_highest_white_population?county=ARECIBO",
    "/v1/address/county_highest_avg_income?county=ARECIBO",
    "/v1/address/distinct_counties_by_daylight_savings?daylight_savings=Yes",
    "/v1/address/distinct_zip_codes_by_county_employees?county=HUNTINGDON&employees=30",
    "/v1/address/sum_asian_population_by_bad_alias?bad_alias=URB%20San%20Joaquin",
    "/v1/address/count_zip_codes_by_bad_alias_time_zone?bad_alias=Internal%20Revenue%20Service&time_zone=Eastern",
    "/v1/address/bad_alias_highest_avg_house_value",
    "/v1/address/distinct_bad_aliases_female_median_age?female_median_age=32",
    "/v1/address/highest_male_to_female_ratio?county=ARECIBO",
    "/v1/address/average_female_median_age?county=ARECIBO",
    "/v1/address/area_codes_female_median_age?female_median_age=32",
    "/v1/address/distinct_aliases_city?city=Hartford",
    "/v1/address/count_counties_state?state_name=Alabama",
    "/v1/address/count_distinct_zip_codes?state_abbreviation=NY&zip_code_type=Post%20Office",
    "/v1/address/latitude_longitude_area_code?area_code=787",
    "/v1/address/count_zip_codes_state?state_abbreviation=CA&zip_code_type=%25Community%20Post%20Office%25&state_name=California&state=CA",
    "/v1/address/county_highest_female_population",
    "/v1/address/distinct_state_names_null_division",
    "/v1/address/population_difference_cbsa?cbsa_name=Allentown-Bethlehem-Easton%2C%20PA-NJ",
    "/v1/address/distinct_zip_codes_county_state?county=NEW%20CASTLE&state_name=Delaware",
    "/v1/address/congress_rep_count_highest_benefits",
    "/v1/address/city_count_congress_member_employees?first_name=Murkowski&last_name=Lisa&employees=0",
    "/v1/address/top_states_asian_population",
    "/v1/address/distinct_state_names_county?county=OUTAGAMIE",
    "/v1/address/congress_parties_zip_code?zip_code=91701",
    "/v1/address/get_alias_by_lat_long?latitude=18.090875&longitude=-66.867756",
    "/v1/address/top_area_code_by_over_65",
    "/v1/address/count_bad_aliases_by_congress_rep?first_name=Thompson&last_name=Bennie%20G",
    "/v1/address/get_lat_long_by_area_code?area_code=636",
    "/v1/address/get_zip_codes_by_congress_rep?first_name=Buchanan&last_name=Vernon",
    "/v1/address/get_state_by_area_code?area_code=878",
    "/v1/address/get_cbsa_lat_long_by_zip?zip_code=45503",
    "/v1/address/get_counties_by_congress_rep?first_name=Hartzler&last_name=Vicky",
    "/v1/address/get_avg_male_median_age_by_county?county=WINDHAM",
    "/v1/address/area_code_by_city_state?city=Bishopville&state=SC",
    "/v1/address/bad_alias_by_city_state?city=Geneva&state=AL",
    "/v1/address/city_state_by_bad_alias?bad_alias=Lawrenceville",
    "/v1/address/alias_bad_alias_by_zip_code?zip_code=38015",
    "/v1/address/cbsa_by_city_state?city=York&state=ME",
    "/v1/address/city_zip_area_code_by_median_age?min_median_age=40",
    "/v1/address/county_by_bad_alias?bad_alias=Druid%20Hills",
    "/v1/address/area_codes_by_county_state?county=PHILLIPS&state_name=Montana",
    "/v1/address/congress_rep_by_state?state=Wisconsin",
    "/v1/address/zip_count_by_time_zone?time_zone=Central",
    "/v1/address/cities_states_by_cbsa?cbsa_name=Lexington-Fayette%2C%20KY",
    "/v1/address/democrat_republican_ratio?democrat_party=Democrat&republican_party=Republican",
    "/v1/address/congress_rep_ratio_by_states?state1=Alabama&state2=Illinois",
    "/v1/address/average_population_per_zip",
    "/v1/address/male_population_by_cbsa?cbsa_name=Berlin%2C%20NH",
    "/v1/address/cbsa_names_by_zip_count?zip_count=10",
    "/v1/address/bad_aliases_by_state?state=PR",
    "/v1/address/zip_coordinates_by_congress_rep?first_name=Grayson&last_name=Alan",
    "/v1/address/distinct_states_by_area_code?area_code=787",
    "/v1/address/congress_reps_by_population?population_2020=30000",
    "/v1/address/zip_codes_with_multiple_area_codes?state=MA&min_area_codes=1",
    "/v1/address/distinct_counties_by_city?city=Arecibo",
    "/v1/address/count_zip_codes_in_cbsa?cbsa_name=Barre%2C%20VT",
    "/v1/address/zip_code_largest_land_area_in_county?county=SAINT%20CROIX",
    "/v1/address/population_change_by_congress_rep?first_name=Griffin&last_name=Tim",
    "/v1/address/asian_population_percentage_in_cbsa?cbsa_name=Atmore%2C%20AL",
    "/v1/address/city_highest_asian_population_by_area_code?area_code=939",
    "/v1/address/state_names_by_elevation?elevation=1039",
    "/v1/address/alias_elevation_by_zip_code?zip_code=1028",
    "/v1/address/area_code_max_land_area",
    "/v1/address/area_code_by_white_population_range?min_population=1700&max_population=2000",
    "/v1/address/area_code_county_by_city?city=Savoy",
    "/v1/address/distinct_aliases_by_population_2010?population_2010=0",
    "/v1/address/count_cities_by_area_code_daylight_savings?area_code=608&daylight_savings=Yes",
    "/v1/address/avg_elevation_by_alias?alias=Amherst",
    "/v1/address/counties_by_city?city=Dalton",
    "/v1/address/aliases_by_type_with_limit?type=Post%20Office&limit=5",
    "/v1/address/difference_po_box_post_office?area_code=787",
    "/v1/address/population_change_percentage?city=Arroyo",
    "/v1/address/zip_codes_cities_by_state_beneficiaries?state_name=Texas&min_beneficiaries=10000",
    "/v1/address/zip_code_count_by_county_black_population?county=DISTRICT%20OF%20COLUMBIA&min_black_population=20000",
    "/v1/address/city_alias_by_zip_code?zip_code=19019",
    "/v1/address/bad_aliases_by_city?city=Camuy",
    "/v1/address/most_populous_district_congress_rep",
    "/v1/address/zip_code_count_by_state_daylight_region?state_name=Illinois&daylight_savings=Yes&region=Midwest",
    "/v1/address/county_zip_code_by_state?state_name=Virgin%20Islands",
    "/v1/address/zip_code_alias_by_city?city=Greeneville",
    "/v1/address/congress/compare_zip_code_counts?first_name_1=Smith&last_name_1=Adrian&first_name_2=Heck&last_name_2=Joe&comparison_result_1=Smith%20Adrian%3EHeck%20Joe&comparison_result_2=Smith%20Adrian%3C%3DHeck%20Joe",
    "/v1/address/cbsa/zip_codes_cbsa_names?city=Oxford",
    "/v1/address/cbsa/zip_codes_organizations?cbsa_name=Kingsport-Bristol%2C%20TN-VA",
    "/v1/address/zip_data/zip_codes_names?organization=Readers%20Digest",
    "/v1/address/state/zip_code_type_percentage?zip_code_type=Post%20Office&state_name=California",
    "/v1/address/congress/zip_codes_by_house?house=House%20of%20Repsentatives",
    "/v1/address/avoid/city_with_most_bad_aliases",
    "/v1/address/state/counties_by_state?state_name=Georgia",
    "/v1/address/cbsa/highest_avg_house_value",
    "/v1/address/total_population_by_county?county=WILCOX",
    "/v1/address/zip_code_highest_asian_population",
    "/v1/address/cities_by_cbsa_type?cbsa_type=Micro",
    "/v1/address/state_with_most_counties",
    "/v1/address/zip_code_count_by_district?district=FL-10",
    "/v1/address/avg_income_by_bad_alias?bad_alias=Danzig",
    "/v1/address/states_above_avg_female_population",
    "/v1/address/percentage_households_by_county?county=CORYELL",
    "/v1/address/cbsa_info_by_city?city=Cabo%20Rojo",
    "/v1/address/county_by_city?city=Las%20Marias",
    "/v1/address/count_distinct_cities_by_congress_rep?first_name=Pierluisi&last_name=Pedro",
    "/v1/address/congress_reps_by_city?city=Guanica",
    "/v1/address/state_with_most_bad_aliases",
    "/v1/address/bad_alias_difference_between_cities?city1=Aguada&city2=Aguadilla",
    "/v1/address/states_by_cbsa_type_and_count?cbsa_type=Metro&count=50",
    "/v1/address/total_population_2020_by_county?county=ARECIBO",
    "/v1/address/distinct_counties_by_population_2010?population_2010=10000",
    "/v1/address/county_with_most_households",
    "/v1/address/percentage_zip_codes_households?households=10000",
    "/v1/address/percentage_zip_codes_type_county?type=Post%20Office&county=SAINT%20CROIX",
    "/v1/address/distinct_counties_area_code_type?area_code=787&type=Unique%20Post%20Office",
    "/v1/address/elevation_by_alias?alias=East%20Longmeadow",
    "/v1/address/count_area_codes_daylight_savings?daylight_savings=No",
    "/v1/address/county_area_code_by_zip?zip_code=1116",
    "/v1/address/distinct_types_by_alias?alias=St%20Thomas",
    "/v1/address/city_by_state_county?state_name=Oklahoma&county=NOBLE",
    "/v1/address/area_code_max_water_area",
    "/v1/address/alias_max_population_2020",
    "/v1/address/elevation_by_state_and_county?state_name=Massachusetts&county=HAMPDEN",
    "/v1/address/area_codes_with_highest_hispanic_population",
    "/v1/address/aliases_by_asian_population?asian_population=7",
    "/v1/address/avg_white_population_by_area_code?area_code=920",
    "/v1/address/percentage_zip_codes_by_county_fips_and_alias?county_fips=20&alias=Ponce",
    "/v1/address/counties_by_population_percentage?population_percentage=0.97",
    "/v1/address/zip_code_count_by_congress_rep?first_name=Kirkpatrick&last_name=Ann",
    "/v1/address/zip_codes_by_cbsa_name?cbsa_name=Allentown-Bethlehem-Easton%2C%20PA-NJ",
    "/v1/address/zip_codes_by_bad_alias?bad_alias=Shared%20Reshipper",
    "/v1/address/count_zip_codes_by_state?state=Puerto%20Rico",
    "/v1/address/count_distinct_zip_codes_and_counties_by_state?state_name=West%20Virginia",
    "/v1/address/zip_codes_area_codes_by_type_elevation?type=Community%20Post%20Office&elevation=6000",
    "/v1/address/district_by_city?city=East%20Springfield",
    "/v1/address/cbsa_name_with_max_employees",
    "/v1/address/count_zip_codes_by_state_and_type?state_name=Ohio&zip_type=Unique%20Post%20Office",
    "/v1/address/average_beneficiaries_per_zip_code?state_name=Guam",
    "/v1/address/congress_party_state_stats?party=Democrat&state=Hawaii"
]
