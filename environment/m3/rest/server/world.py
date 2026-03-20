from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

try:
    conn = sqlite3.connect('db/world/world.sqlite')
    cursor = conn.cursor()
except Exception as e:
    print(f"Warning: could not connect to database: {e}")
    conn = None
    cursor = None

# Endpoint to get the name of the country with the highest life expectancy
@app.get("/v1/world/country_with_highest_life_expectancy", operation_id="get_country_with_highest_life_expectancy", summary="Retrieves the name of the country with the highest life expectancy, allowing the user to limit the number of results returned.")
async def get_country_with_highest_life_expectancy(limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT Name FROM Country ORDER BY LifeExpectancy LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the names of countries where a specific language is official
@app.get("/v1/world/countries_with_official_language", operation_id="get_countries_with_official_language", summary="Retrieve a list of country names where the specified language is officially recognized. The operation allows you to filter results based on whether the language is officially recognized and limit the number of results returned.")
async def get_countries_with_official_language(language: str = Query(..., description="Language to filter by"), is_official: str = Query(..., description="Whether the language is official ('T' or 'F')"), limit: int = Query(5, description="Limit the number of results")):
    cursor.execute("SELECT T1.Name FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = ? AND T2.IsOfficial = ? LIMIT ?", (language, is_official, limit))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the average population of cities in a specific district
@app.get("/v1/world/average_population_by_district", operation_id="get_average_population_by_district", summary="Get the average population of cities in a specific district")
async def get_average_population_by_district(district: str = Query(..., description="District to filter by")):
    cursor.execute("SELECT AVG(Population) FROM City WHERE District = ? GROUP BY ID", (district,))
    result = cursor.fetchone()
    if not result:
        return {"average_population": []}
    return {"average_population": result[0]}

# Endpoint to get the languages spoken in a specific country
@app.get("/v1/world/languages_by_country", operation_id="get_languages_by_country", summary="Retrieve the languages spoken in a specified country. The operation filters the available languages based on the provided country code.")
async def get_languages_by_country(country_code: str = Query(..., description="Country code to filter by")):
    cursor.execute("SELECT Language FROM CountryLanguage WHERE CountryCode = ?", (country_code,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the count of countries where a specific language is spoken
@app.get("/v1/world/count_countries_by_language", operation_id="get_count_countries_by_language", summary="Retrieves the total number of countries where the specified language is spoken. The operation filters the countries based on the provided language parameter.")
async def get_count_countries_by_language(language: str = Query(..., description="Language to filter by")):
    cursor.execute("SELECT SUM(CASE WHEN Language = ? THEN 1 ELSE 0 END) FROM CountryLanguage", (language,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cities in a specific district
@app.get("/v1/world/count_cities_by_district", operation_id="get_count_cities_by_district", summary="Retrieves the total number of cities in a specified district. The district is used as a filter to count the cities.")
async def get_count_cities_by_district(district: str = Query(..., description="District to filter by")):
    cursor.execute("SELECT COUNT(ID) FROM City WHERE District = ?", (district,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the IDs of cities in the country with the largest surface area
@app.get("/v1/world/cities_in_largest_country", operation_id="get_cities_in_largest_country", summary="Retrieves the unique identifiers of cities located in the country with the largest surface area. The operation identifies the country with the maximum surface area and returns the corresponding city IDs.")
async def get_cities_in_largest_country():
    cursor.execute("SELECT T2.ID FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T1.SurfaceArea = ( SELECT MAX(SurfaceArea) FROM Country )")
    result = cursor.fetchall()
    if not result:
        return {"city_ids": []}
    return {"city_ids": [row[0] for row in result]}

# Endpoint to get the capital and population of a specific country
@app.get("/v1/world/capital_and_population_by_country", operation_id="get_capital_and_population_by_country", summary="Get the capital and population of a specific country")
async def get_capital_and_population_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Capital, T2.Population FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"capital": [], "population": []}
    return {"capital": result[0], "population": result[1]}

# Endpoint to get the languages spoken in a specific country
@app.get("/v1/world/languages_by_country_name", operation_id="get_languages_by_country_name", summary="Retrieves the languages spoken in a specified country. The operation uses the provided country name to search for corresponding language data in the database.")
async def get_languages_by_country_name(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the country with the highest life expectancy where a specific language is official
@app.get("/v1/world/country_with_highest_life_expectancy_official_language", operation_id="get_country_with_highest_life_expectancy_official_language", summary="Retrieves the country with the highest life expectancy where the specified language is officially recognized. The response includes the country's name, capital, and the official language. The number of results can be limited by the user.")
async def get_country_with_highest_life_expectancy_official_language(is_official: str = Query(..., description="Whether the language is official ('T' or 'F')"), limit: int = Query(1, description="Limit the number of results")):
    cursor.execute("SELECT T1.Name, T1.Capital, T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode INNER JOIN City AS T3 ON T1.Code = T3.CountryCode WHERE T2.IsOfficial = ? ORDER BY T1.LifeExpectancy DESC LIMIT ?", (is_official, limit))
    result = cursor.fetchone()
    if not result:
        return {"name": [], "capital": [], "language": []}
    return {"name": result[0], "capital": result[1], "language": result[2]}

# Endpoint to get the name and language of countries in a specific continent where the language is official
@app.get("/v1/world/countries_by_continent_official_language", operation_id="get_countries_by_continent_official_language", summary="Retrieves the names of countries and their official languages from a specified continent. The operation filters the results based on the official status of the language, which is indicated by the input parameter.")
async def get_countries_by_continent_official_language(continent: str = Query(..., description="Continent of the country"), is_official: str = Query(..., description="Whether the language is official (T or F)")):
    cursor.execute("SELECT T1.Name, T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Continent = ? AND T2.IsOfficial = ?", (continent, is_official))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the languages spoken in a specific region with a percentage greater than a specified value
@app.get("/v1/world/languages_by_region_percentage", operation_id="get_languages_by_region_percentage", summary="Retrieve the languages spoken in a specified region where the percentage of speakers surpasses a given threshold. The operation filters languages based on the provided region and minimum percentage, returning only those that meet the criteria.")
async def get_languages_by_region_percentage(region: str = Query(..., description="Region of the country"), percentage: float = Query(..., description="Percentage of language spoken")):
    cursor.execute("SELECT T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Region = ? AND T2.Percentage > ?", (region, percentage))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": result}

# Endpoint to get the most populous city in each country with its life expectancy
@app.get("/v1/world/most_populous_city_life_expectancy", operation_id="get_most_populous_city_life_expectancy", summary="Retrieves the name of the most populous city in each country along with the corresponding life expectancy of that country.")
async def get_most_populous_city_life_expectancy():
    cursor.execute("SELECT T2.Name, T1.Code, T1.LifeExpectancy FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode ORDER BY T2.Population DESC LIMIT 1")
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": result}

# Endpoint to get the capital and language of the country with the highest life expectancy
@app.get("/v1/world/capital_language_highest_life_expectancy", operation_id="get_capital_language_highest_life_expectancy", summary="Retrieves the capital city and official language of the country with the highest life expectancy. The data is obtained by joining the Country, CountryLanguage, and City tables, and sorting the results by life expectancy in descending order. The top result is returned.")
async def get_capital_language_highest_life_expectancy():
    cursor.execute("SELECT T1.Capital, T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode INNER JOIN City AS T3 ON T1.Code = T3.CountryCode ORDER BY T1.LifeExpectancy LIMIT 1")
    result = cursor.fetchall()
    if not result:
        return {"capitals": []}
    return {"capitals": result}

# Endpoint to get the country with the smallest surface area where the language is official
@app.get("/v1/world/smallest_country_official_language", operation_id="get_smallest_country_official_language", summary="Retrieves the country with the smallest surface area where the specified language is officially recognized. The response includes the country's name, population, capital, and the official language.")
async def get_smallest_country_official_language(is_official: str = Query(..., description="Whether the language is official (T or F)")):
    cursor.execute("SELECT T1.Name, T1.Population, T1.Capital, T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode INNER JOIN City AS T3 ON T1.Code = T3.CountryCode WHERE T2.IsOfficial = ? ORDER BY T1.SurfaceArea LIMIT 1", (is_official,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the percentage of countries where a specific language is spoken
@app.get("/v1/world/percentage_countries_language", operation_id="get_percentage_countries_language", summary="Retrieves the percentage of countries where the specified language is spoken. This operation calculates the proportion of countries where the given language is used, based on the total number of countries in the database. The language parameter is used to identify the specific language of interest.")
async def get_percentage_countries_language(language: str = Query(..., description="Language spoken in the country")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.Language = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.Code) FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode", (language,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the district of the most populous city
@app.get("/v1/world/most_populous_city_district", operation_id="get_most_populous_city_district", summary="Retrieves the district of the city with the highest population. The operation returns the name of the district for the city with the largest population, as determined by the data in the backend database.")
async def get_most_populous_city_district():
    cursor.execute("SELECT District FROM City ORDER BY Population LIMIT 1")
    result = cursor.fetchall()
    if not result:
        return {"districts": []}
    return {"districts": result}

# Endpoint to get the continent of the country with the smallest surface area
@app.get("/v1/world/smallest_country_continent", operation_id="get_smallest_country_continent", summary="Retrieves the continent of the country with the smallest surface area. The operation returns the continent name of the country with the least surface area, as determined by the data in the backend database.")
async def get_smallest_country_continent():
    cursor.execute("SELECT Continent FROM Country ORDER BY SurfaceArea LIMIT 1")
    result = cursor.fetchall()
    if not result:
        return {"continents": []}
    return {"continents": result}

# Endpoint to get the head of state of the country with the most populous city
@app.get("/v1/world/head_of_state_most_populous_city", operation_id="get_head_of_state_most_populous_city", summary="Retrieves the head of state of the country that has the most populous city. The operation identifies the country with the highest population in its most populous city and returns the head of state of that country.")
async def get_head_of_state_most_populous_city():
    cursor.execute("SELECT T1.HeadOfState FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode ORDER BY T2.Population DESC LIMIT 1")
    result = cursor.fetchall()
    if not result:
        return {"heads_of_state": []}
    return {"heads_of_state": result}

# Endpoint to get the country code where a specific language is official and ordered by capital
@app.get("/v1/world/country_code_official_language_capital", operation_id="get_country_code_official_language_capital", summary="Retrieves the country code of the nation with the largest capital city where the specified language is officially recognized. The language and its official status are provided as input parameters.")
async def get_country_code_official_language_capital(language: str = Query(..., description="Language spoken in the country"), is_official: str = Query(..., description="Whether the language is official (T or F)")):
    cursor.execute("SELECT T1.Code FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = ? AND T2.IsOfficial = ? ORDER BY T1.Capital DESC LIMIT 1", (language, is_official))
    result = cursor.fetchall()
    if not result:
        return {"country_codes": []}
    return {"country_codes": result}

# Endpoint to get city names based on country life expectancy
@app.get("/v1/world/city_names_by_life_expectancy", operation_id="get_city_names_by_life_expectancy", summary="Retrieves the names of cities located in countries with a specified life expectancy. The operation filters the data based on the provided life expectancy value, returning a list of city names that meet the criteria.")
async def get_city_names_by_life_expectancy(life_expectancy: float = Query(..., description="Life expectancy of the country")):
    cursor.execute("SELECT T2.Name FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T1.LifeExpectancy = ?", (life_expectancy,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the head of state of a country where a specific language is spoken
@app.get("/v1/world/head_of_state_by_language", operation_id="get_head_of_state_by_language", summary="Retrieve the head of state of a country where the specified language is spoken, ordered by the percentage of speakers. The response is limited to one result.")
async def get_head_of_state_by_language(language: str = Query(..., description="Language spoken in the country")):
    cursor.execute("SELECT T1.HeadOfState FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = ? ORDER BY T2.Percentage LIMIT 1", (language,))
    result = cursor.fetchone()
    if not result:
        return {"head_of_state": []}
    return {"head_of_state": result[0]}

# Endpoint to get the surface area of a country based on a city name
@app.get("/v1/world/surface_area_by_city_name", operation_id="get_surface_area_by_city_name", summary="Retrieves the total surface area of the country where the specified city is located. The operation requires the name of the city as an input parameter.")
async def get_surface_area_by_city_name(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.SurfaceArea FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T2.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"surface_area": []}
    return {"surface_area": result[0]}

# Endpoint to get languages spoken in countries with a population less than a specified number
@app.get("/v1/world/languages_by_population", operation_id="get_languages_by_population", summary="Retrieve a list of languages spoken in countries with a population less than the specified number. The input parameter determines the population threshold for the query.")
async def get_languages_by_population(population: int = Query(..., description="Population of the country")):
    cursor.execute("SELECT T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Population < ?", (population,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get official languages of a specific country
@app.get("/v1/world/official_languages_by_country", operation_id="get_official_languages_by_country", summary="Retrieves the official languages of a specified country. The operation requires the country's name and a boolean value indicating whether the language is officially recognized. The response includes a list of official languages that meet the specified criteria.")
async def get_official_languages_by_country(country_name: str = Query(..., description="Name of the country"), is_official: str = Query(..., description="Whether the language is official ('T' for true)")):
    cursor.execute("SELECT T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Name = ? AND T2.IsOfficial = ?", (country_name, is_official))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get city names and districts based on the local name of a country
@app.get("/v1/world/city_info_by_local_name", operation_id="get_city_info_by_local_name", summary="Retrieves the names and districts of cities located within a country, based on the local name of the country provided as input. The operation returns a list of city names and their respective districts.")
async def get_city_info_by_local_name(local_name: str = Query(..., description="Local name of the country")):
    cursor.execute("SELECT T2.Name, T2.District FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T1.LocalName = ?", (local_name,))
    result = cursor.fetchall()
    if not result:
        return {"city_info": []}
    return {"city_info": [{"name": row[0], "district": row[1]} for row in result]}

# Endpoint to get the count of a specific country name in the database
@app.get("/v1/world/count_country_name", operation_id="get_count_country_name", summary="Retrieves the total count of a specified country from the database. This operation considers the country's name and its associated languages. The input parameter determines the country to be counted.")
async def get_count_country_name(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Name = ? THEN 1 ELSE 0 END) FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get districts based on the head of state of a country
@app.get("/v1/world/districts_by_head_of_state", operation_id="get_districts_by_head_of_state", summary="Retrieves a list of districts from a country whose head of state matches the provided input. The operation filters the districts based on the head of state's name and returns the corresponding district names.")
async def get_districts_by_head_of_state(head_of_state: str = Query(..., description="Head of state of the country")):
    cursor.execute("SELECT T2.District FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T1.HeadOfState = ?", (head_of_state,))
    result = cursor.fetchall()
    if not result:
        return {"districts": []}
    return {"districts": [row[0] for row in result]}

# Endpoint to get the head of state based on a city district
@app.get("/v1/world/head_of_state_by_district", operation_id="get_head_of_state_by_district", summary="Get the head of state based on a city district")
async def get_head_of_state_by_district(district: str = Query(..., description="District of the city")):
    cursor.execute("SELECT T1.HeadOfState FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T2.District = ?", (district,))
    result = cursor.fetchone()
    if not result:
        return {"head_of_state": []}
    return {"head_of_state": result[0]}

# Endpoint to get the percentage of countries speaking a specific language with GNP greater than a specified value
@app.get("/v1/world/language_percentage_by_gnp", operation_id="get_language_percentage_by_gnp", summary="Retrieves the percentage of countries with a GNP greater than the specified value that speak a given language. The calculation is based on the total number of countries with a GNP above the provided threshold.")
async def get_language_percentage_by_gnp(language: str = Query(..., description="Language spoken in the country"), gnp: int = Query(..., description="GNP value")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.Language = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.Code) FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.GNP > ?", (language, gnp))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in count between two government forms for countries speaking a specific language
@app.get("/v1/world/government_form_difference_by_language", operation_id="get_government_form_difference_by_language", summary="Retrieve the difference in the number of countries with two specified government forms, for countries where a particular language is spoken.")
async def get_government_form_difference_by_language(government_form1: str = Query(..., description="First government form"), government_form2: str = Query(..., description="Second government form"), language: str = Query(..., description="Language spoken in the country")):
    cursor.execute("SELECT COUNT(T1.GovernmentForm = ?) - COUNT(T1.GovernmentForm = ?) FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = ?", (government_form1, government_form2, language))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the names of countries that gained independence in a specific year
@app.get("/v1/world/countries_by_independence_year", operation_id="get_countries_by_independence_year", summary="Retrieve the names of countries that gained independence in a specified year. The year of independence is provided as an input parameter.")
async def get_countries_by_independence_year(indep_year: int = Query(..., description="Year of independence")):
    cursor.execute("SELECT Name FROM Country WHERE IndepYear = ?", (indep_year,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the names of countries in a specific continent
@app.get("/v1/world/countries_by_continent", operation_id="get_countries_by_continent", summary="Retrieves the names of all countries located within a specified continent. The continent is identified by its name.")
async def get_countries_by_continent(continent: str = Query(..., description="Continent name")):
    cursor.execute("SELECT Name FROM Country WHERE Continent = ?", (continent,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the name of the country with the highest GNP in a specific continent
@app.get("/v1/world/highest_gnp_country_by_continent", operation_id="get_highest_gnp_country_by_continent", summary="Retrieves the name of the country with the highest Gross National Product (GNP) in a specified continent. The operation requires the continent name as input and returns the name of the country with the highest GNP in that continent.")
async def get_highest_gnp_country_by_continent(continent: str = Query(..., description="Continent name")):
    cursor.execute("SELECT Name FROM Country WHERE Continent = ? ORDER BY GNP DESC LIMIT 1", (continent,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of cities with a specific name
@app.get("/v1/world/city_count_by_name", operation_id="get_city_count_by_name", summary="Retrieves the total count of cities that share a specified name. The operation requires the name of the city as an input parameter.")
async def get_city_count_by_name(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT COUNT(ID) FROM City WHERE Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the local name of a country by its name
@app.get("/v1/world/local_name_by_country_name", operation_id="get_local_name_by_country_name", summary="Retrieves the local name of a specified country. The operation requires the country's name as input and returns the corresponding local name, as recorded in the database.")
async def get_local_name_by_country_name(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT LocalName FROM Country WHERE Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"local_name": []}
    return {"local_name": result[0]}

# Endpoint to get the count of countries with a specific government form
@app.get("/v1/world/countries_by_government_form", operation_id="get_countries_by_government_form", summary="Retrieve the number of countries that adhere to a specific government form. The operation filters countries based on the provided government form and returns the count.")
async def get_countries_by_government_form(government_form: str = Query(..., description="Government form")):
    cursor.execute("SELECT COUNT(Code) FROM Country WHERE GovernmentForm = ?", (government_form,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of a specific language spoken in a specific country
@app.get("/v1/world/language_percentage_by_country", operation_id="get_language_percentage_by_country", summary="Retrieves the percentage of a specified language spoken in a given country. The operation requires the country's name and the language as input parameters to filter the data and return the corresponding percentage.")
async def get_language_percentage_by_country(country_name: str = Query(..., description="Name of the country"), language: str = Query(..., description="Language spoken in the country")):
    cursor.execute("SELECT T2.Percentage FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Name = ? AND T2.Language = ?", (country_name, language))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the government form of a country based on the city name
@app.get("/v1/world/government_form_by_city", operation_id="get_government_form_by_city", summary="Retrieves the government form of a country associated with the specified city. The operation uses the provided city name to identify the corresponding country and its government form.")
async def get_government_form_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.GovernmentForm FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T2.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"government_form": []}
    return {"government_form": result[0]}

# Endpoint to get the capital of a country based on the country name
@app.get("/v1/world/capital_by_country", operation_id="get_capital_by_country", summary="Retrieves the capital city of a specified country. The operation uses the provided country name to search for a match in the database and returns the corresponding capital city.")
async def get_capital_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Capital FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"capital": []}
    return {"capital": result[0]}

# Endpoint to get the languages spoken in countries of a specific continent
@app.get("/v1/world/languages_by_continent", operation_id="get_languages_by_continent", summary="Retrieve a list of languages spoken in countries within a specified continent. The operation filters countries by continent and returns the corresponding languages.")
async def get_languages_by_continent(continent: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Continent = ?", (continent,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the head of state of a country based on the city name
@app.get("/v1/world/head_of_state_by_city", operation_id="get_head_of_state_by_city", summary="Retrieves the head of state of a country associated with the specified city. The operation uses the provided city name to search for a match in the database and returns the corresponding head of state.")
async def get_head_of_state_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.HeadOfState FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T2.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"head_of_state": []}
    return {"head_of_state": result[0]}

# Endpoint to get the count of unofficial languages spoken in a specific country
@app.get("/v1/world/unofficial_languages_count_by_country", operation_id="get_unofficial_languages_count_by_country", summary="Retrieve the total number of unofficial languages spoken in a specified country. The operation requires the name of the country as an input parameter and returns the count of unofficial languages spoken in that country.")
async def get_unofficial_languages_count_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(CASE WHEN T2.IsOfficial = 'F' THEN 1 ELSE 0 END) FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the least populated city in a specific country
@app.get("/v1/world/least_populated_city_by_country", operation_id="get_least_populated_city_by_country", summary="Retrieves the name of the city with the smallest population in a given country. The operation requires the name of the country as input and returns the name of the least populated city.")
async def get_least_populated_city_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Name FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T1.Name = ? ORDER BY T2.Population ASC LIMIT 1", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the city in the country with the highest life expectancy
@app.get("/v1/world/city_with_highest_life_expectancy", operation_id="get_city_with_highest_life_expectancy", summary="Retrieves the name of the city with the highest life expectancy in a country. The operation identifies the country with the highest life expectancy and returns the name of the city within that country. The result is determined by comparing life expectancy data across countries and cities.")
async def get_city_with_highest_life_expectancy():
    cursor.execute("SELECT T2.Name FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode ORDER BY T1.LifeExpectancy DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the languages and their official status for countries that gained independence in a specific year
@app.get("/v1/world/languages_by_independence_year", operation_id="get_languages_by_independence_year", summary="Retrieve the languages and their official statuses for countries that gained independence in a specified year. This operation provides a comprehensive overview of the linguistic landscape in countries that became independent in the given year.")
async def get_languages_by_independence_year(independence_year: int = Query(..., description="Year of independence")):
    cursor.execute("SELECT T2.Language, T2.IsOfficial FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.IndepYear = ? GROUP BY T2.Language, T2.IsOfficial", (independence_year,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [{"language": row[0], "is_official": row[1]} for row in result]}

# Endpoint to get the capital of the most populated country
@app.get("/v1/world/capital_of_most_populated_country", operation_id="get_capital_of_most_populated_country", summary="Retrieves the capital city of the country with the highest population. The operation uses a database query to determine the most populated country and returns its capital.")
async def get_capital_of_most_populated_country():
    cursor.execute("SELECT T1.Capital FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode ORDER BY T1.Population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"capital": []}
    return {"capital": result[0]}

# Endpoint to get the names of countries in a specific continent where the language is not official
@app.get("/v1/world/countries_with_unofficial_languages_by_continent", operation_id="get_countries_with_unofficial_languages_by_continent", summary="Retrieve the names of countries in a specified continent where the spoken languages are not officially recognized. The operation filters countries based on the provided continent name and returns a list of unique country names.")
async def get_countries_with_unofficial_languages_by_continent(continent: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT T1.Name FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.Continent = ? AND T2.IsOfficial = 'F' GROUP BY T1.Name", (continent,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the average GNP of countries where a specific language is spoken
@app.get("/v1/world/average_gnp_by_language", operation_id="get_average_gnp_by_language", summary="Retrieves the average Gross National Product (GNP) of countries where the specified language is spoken. The calculation is based on the GNP data of countries that have the given language listed as one of their spoken languages.")
async def get_average_gnp_by_language(language: str = Query(..., description="Language spoken in the country")):
    cursor.execute("SELECT AVG(T1.GNP) FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = ?", (language,))
    result = cursor.fetchone()
    if not result:
        return {"average_gnp": []}
    return {"average_gnp": result[0]}

# Endpoint to get the percentage of the world's surface area where a specific language is spoken
@app.get("/v1/world/percentage_surface_area_by_language", operation_id="get_percentage_surface_area_by_language", summary="Retrieves the percentage of the world's total surface area where the specified language is spoken. This operation calculates the sum of the surface areas of all countries where the given language is spoken, and then divides it by the total surface area of all countries to obtain the percentage.")
async def get_percentage_surface_area_by_language(language: str = Query(..., description="Language spoken in the country")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.Language = ?, T1.SurfaceArea, 0)) AS REAL) * 100 / SUM(T1.SurfaceArea) FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode", (language,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_surface_area": []}
    return {"percentage_surface_area": result[0]}

# Endpoint to get the country with the smallest surface area
@app.get("/v1/world/smallest_country_by_surface_area", operation_id="get_smallest_country_by_surface_area", summary="Retrieves the name of the country with the smallest surface area from the database.")
async def get_smallest_country_by_surface_area():
    cursor.execute("SELECT Name FROM Country ORDER BY SurfaceArea ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the country with the largest population
@app.get("/v1/world/largest_country_by_population", operation_id="get_largest_country_by_population", summary="Retrieves the name of the country with the highest population. The operation returns the top result from a sorted list of countries by population in descending order.")
async def get_largest_country_by_population():
    cursor.execute("SELECT Name FROM Country ORDER BY Population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the language spoken in the country with the smallest population
@app.get("/v1/world/language_smallest_population_country", operation_id="get_language_smallest_population_country", summary="Retrieves the language spoken in the country with the smallest population. This operation identifies the country with the least number of inhabitants and returns the primary language spoken there. The result provides insights into linguistic diversity and population distribution.")
async def get_language_smallest_population_country():
    cursor.execute("SELECT T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode ORDER BY T1.Population ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"language": []}
    return {"language": result[0]}

# Endpoint to get the names and official languages of countries that gained independence after a specific year
@app.get("/v1/world/countries_independent_after_year", operation_id="get_countries_independent_after_year", summary="Retrieve the names of countries and their official languages that gained independence after the specified year. The operation filters countries based on the provided year and language official status.")
async def get_countries_independent_after_year(indep_year: int = Query(..., description="Year after which the country gained independence"), is_official: str = Query(..., description="Whether the language is official ('T' for true, 'F' for false)")):
    cursor.execute("SELECT T1.Name, T2.Language FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T1.IndepYear > ? AND T2.IsOfficial = ?", (indep_year, is_official))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [{"name": row[0], "language": row[1]} for row in result]}

# Endpoint to get the country with the most populous city
@app.get("/v1/world/country_with_most_populous_city", operation_id="get_country_with_most_populous_city", summary="Retrieves the name of the country that has the city with the highest population. The operation uses a database query to determine the country with the most populous city.")
async def get_country_with_most_populous_city():
    cursor.execute("SELECT T1.Name FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode ORDER BY T2.Population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the life expectancy of the most populous city
@app.get("/v1/world/life_expectancy_most_populous_city", operation_id="get_life_expectancy_most_populous_city", summary="Retrieves the life expectancy of the most populous city, or cities, based on the specified limit. The operation returns the life expectancy data for the city, or cities, with the highest population. The limit parameter allows you to control the number of results returned.")
async def get_life_expectancy_most_populous_city(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.LifeExpectancy FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code ORDER BY T1.Population DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"life_expectancy": []}
    return {"life_expectancy": result[0]}

# Endpoint to get the GNP of the least populous city
@app.get("/v1/world/gnp_least_populous_city", operation_id="get_gnp_least_populous_city", summary="Retrieves the Gross National Product (GNP) of the least populous city, as determined by the provided limit. This operation returns the GNP of the city with the smallest population, up to the specified limit. The result provides insight into the economic output of the least populous city.")
async def get_gnp_least_populous_city(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.GNP FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code ORDER BY T1.Population ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"gnp": []}
    return {"gnp": result[0]}

# Endpoint to get the city with the most languages spoken
@app.get("/v1/world/city_most_languages", operation_id="get_city_most_languages", summary="Retrieves the city with the most spoken languages, based on the population. The response is limited to the specified number of results.")
async def get_city_most_languages(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Name FROM ( SELECT T1.Name, T2.Language FROM City AS T1 INNER JOIN CountryLanguage AS T2 ON T1.CountryCode = T2.CountryCode GROUP BY T1.Name, T1.Population, T2.Language ORDER BY T1.Population DESC ) AS T3 GROUP BY T3.Name ORDER BY COUNT(Language) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the country with the most populous city and largest surface area
@app.get("/v1/world/country_most_populous_city_largest_surface_area", operation_id="get_country_most_populous_city_largest_surface_area", summary="Retrieves the country with the most populous city and largest surface area, based on the provided limit. The limit parameter determines the maximum number of results to return.")
async def get_country_most_populous_city_largest_surface_area(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.Name FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code ORDER BY T1.Population DESC, T2.SurfaceArea DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get cities in a specific country
@app.get("/v1/world/cities_in_country", operation_id="get_cities_in_country", summary="Retrieves a list of cities located within a specified country. The operation filters the cities based on the provided country name.")
async def get_cities_in_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Name FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get cities in a country with a specific local name
@app.get("/v1/world/cities_in_country_local_name", operation_id="get_cities_in_country_local_name", summary="Retrieves a list of cities located within a country identified by its local name. The local name is used to pinpoint the country, and the operation returns the names of the cities within that country.")
async def get_cities_in_country_local_name(local_name: str = Query(..., description="Local name of the country")):
    cursor.execute("SELECT T1.Name FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.LocalName = ?", (local_name,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the average life expectancy of countries speaking a specific language
@app.get("/v1/world/avg_life_expectancy_by_language", operation_id="get_avg_life_expectancy_by_language", summary="Retrieves the average life expectancy of countries where the specified language is spoken. The calculation is based on the life expectancy data of countries that have the given language as one of their spoken languages.")
async def get_avg_life_expectancy_by_language(language: str = Query(..., description="Language spoken in the country")):
    cursor.execute("SELECT AVG(T1.LifeExpectancy) FROM Country AS T1 INNER JOIN CountryLanguage AS T2 ON T1.Code = T2.CountryCode WHERE T2.Language = ?", (language,))
    result = cursor.fetchone()
    if not result:
        return {"avg_life_expectancy": []}
    return {"avg_life_expectancy": result[0]}

# Endpoint to get the GNP growth rate of a country with a specific city
@app.get("/v1/world/gnp_growth_rate_by_city", operation_id="get_gnp_growth_rate_by_city", summary="Retrieves the Gross National Product (GNP) growth rate for a country that contains the specified city. The growth rate is calculated by comparing the current GNP to the previous GNP, providing a percentage change over time.")
async def get_gnp_growth_rate_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT CAST((T1.GNP - T1.GNPOld) AS REAL) / T1.GNPOld FROM Country AS T1 INNER JOIN City AS T2 ON T1.Code = T2.CountryCode WHERE T2.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"gnp_growth_rate": []}
    return {"gnp_growth_rate": result[0]}

# Endpoint to get the district of a specific city
@app.get("/v1/world/district_by_city_name", operation_id="get_district_by_city_name", summary="Retrieves the district of a specified city. The operation requires the name of the city as an input parameter and returns the corresponding district.")
async def get_district_by_city_name(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT District FROM City WHERE name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"district": []}
    return {"district": result[0]}

# Endpoint to get the most populous city
@app.get("/v1/world/most_populous_city", operation_id="get_most_populous_city", summary="Retrieves the name of the most populous city, or cities, based on the provided limit. The operation returns the city name(s) with the highest population, up to the specified limit.")
async def get_most_populous_city(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Name FROM City ORDER BY Population DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the district of a city based on population
@app.get("/v1/world/city_district_by_population", operation_id="get_city_district_by_population", summary="Retrieves the district of a city with the specified population. The population parameter is used to identify the city and return its corresponding district.")
async def get_city_district_by_population(population: int = Query(..., description="Population of the city")):
    cursor.execute("SELECT District FROM City WHERE population = ?", (population,))
    result = cursor.fetchone()
    if not result:
        return {"district": []}
    return {"district": result[0]}

# Endpoint to get the name of the country with the largest surface area
@app.get("/v1/world/largest_country_by_surface_area", operation_id="get_largest_country_by_surface_area", summary="Retrieves the name of the country with the largest surface area from the database. The operation returns the name of the country that has the highest surface area, based on the data available in the Country table.")
async def get_largest_country_by_surface_area():
    cursor.execute("SELECT Name FROM Country ORDER BY SurfaceArea DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of countries based on life expectancy
@app.get("/v1/world/count_countries_by_life_expectancy", operation_id="get_count_countries_by_life_expectancy", summary="Retrieve the number of countries with a specified life expectancy. The operation filters countries based on the provided life expectancy and returns the count.")
async def get_count_countries_by_life_expectancy(life_expectancy: float = Query(..., description="Life expectancy of the country")):
    cursor.execute("SELECT COUNT(*) FROM Country WHERE LifeExpectancy = ?", (life_expectancy,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the independence year of a country based on its name
@app.get("/v1/world/independence_year_by_country_name", operation_id="get_independence_year_by_country_name", summary="Retrieves the year of independence for a specified country. The operation requires the country's name as input and returns the corresponding independence year from the database.")
async def get_independence_year_by_country_name(name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT IndepYear FROM Country WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"independence_year": []}
    return {"independence_year": result[0]}

# Endpoint to get the count of countries based on GNP
@app.get("/v1/world/count_countries_by_gnp", operation_id="get_count_countries_by_gnp", summary="Retrieve the number of countries with a specified Gross National Product (GNP). The GNP value is provided as an input parameter.")
async def get_count_countries_by_gnp(gnp: float = Query(..., description="GNP of the country")):
    cursor.execute("SELECT COUNT(*) FROM Country WHERE GNP = ?", (gnp,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average surface area of all countries
@app.get("/v1/world/average_surface_area", operation_id="get_average_surface_area", summary="Retrieves the average surface area of all countries in the database. This operation calculates the mean value of the surface area field across all country records, providing a statistical summary of the global land distribution.")
async def get_average_surface_area():
    cursor.execute("SELECT AVG(SurfaceArea) FROM Country")
    result = cursor.fetchone()
    if not result:
        return {"average_surface_area": []}
    return {"average_surface_area": result[0]}

# Endpoint to get the count of distinct languages spoken in cities of a specific district
@app.get("/v1/world/count_distinct_languages_by_district", operation_id="get_count_distinct_languages_by_district", summary="Retrieves the number of unique languages spoken in cities within a specified district. The district is identified by its name.")
async def get_count_distinct_languages_by_district(district: str = Query(..., description="District name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.Language) FROM City AS T1 INNER JOIN CountryLanguage AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.District = ?", (district,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the districts of cities in the country with the largest surface area
@app.get("/v1/world/districts_in_largest_country", operation_id="get_districts_in_largest_country", summary="Retrieves the names of all districts in the cities of the country with the largest surface area. This operation returns a list of district names, providing insights into the geographical distribution of urban areas within the largest country.")
async def get_districts_in_largest_country():
    cursor.execute("SELECT T1.District FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.Name = ( SELECT Name FROM Country ORDER BY SurfaceArea DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"districts": []}
    return {"districts": [row[0] for row in result]}

# Endpoint to get the count of distinct city names in countries with a specific head of state
@app.get("/v1/world/count_distinct_city_names_by_head_of_state", operation_id="get_count_distinct_city_names_by_head_of_state", summary="Retrieve the number of unique city names in countries governed by a specified head of state. The input parameter determines the head of state for which the count is calculated.")
async def get_count_distinct_city_names_by_head_of_state(head_of_state: str = Query(..., description="Head of state")):
    cursor.execute("SELECT COUNT(DISTINCT T1.Name) FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.HeadOfState = ?", (head_of_state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the population of a country based on the city name
@app.get("/v1/world/city_population", operation_id="get_city_population", summary="Retrieves the population of a country associated with the specified city. The operation uses the provided city name to look up the corresponding country and its population.")
async def get_city_population(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T2.Population FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T1.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"population": []}
    return {"population": result[0]}

# Endpoint to get the official language of the least populated city
@app.get("/v1/world/official_language_least_populated_city", operation_id="get_official_language_least_populated_city", summary="Retrieves the official language of the least populated city in the world. The operation considers only the languages marked as official. The input parameter determines whether the language should be considered official or not.")
async def get_official_language_least_populated_city(is_official: str = Query(..., description="Whether the language is official ('T' for true)")):
    cursor.execute("SELECT T2.Language FROM City AS T1 INNER JOIN CountryLanguage AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.IsOfficial = ? ORDER BY T1.Population ASC LIMIT 1", (is_official,))
    result = cursor.fetchone()
    if not result:
        return {"language": []}
    return {"language": result[0]}

# Endpoint to get the surface area and GNP of a country based on the district name
@app.get("/v1/world/surface_area_gnp_by_district", operation_id="get_surface_area_gnp_by_district", summary="Retrieves the surface area and Gross National Product (GNP) of a country associated with the specified district. The district name is used to identify the relevant country and its corresponding data.")
async def get_surface_area_gnp_by_district(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT T2.SurfaceArea, T2.GNP FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T1.District = ?", (district_name,))
    result = cursor.fetchone()
    if not result:
        return {"surface_area": [], "gnp": []}
    return {"surface_area": result[0], "gnp": result[1]}

# Endpoint to get the district of the city in the country with the smallest surface area
@app.get("/v1/world/district_smallest_surface_area", operation_id="get_district_smallest_surface_area", summary="Retrieves the name of the district with the smallest surface area in a country. This operation identifies the country with the least surface area and then determines the district within its city that has the smallest area. The result provides insights into the geographical distribution of urban areas.")
async def get_district_smallest_surface_area():
    cursor.execute("SELECT T1.District FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code ORDER BY T2.SurfaceArea ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"district": []}
    return {"district": result[0]}

# Endpoint to get the name of countries with GNP less than a specified value and a specific official language
@app.get("/v1/world/countries_gnp_official_language", operation_id="get_countries_gnp_official_language", summary="Retrieve the names of countries with a Gross National Product (GNP) less than the provided value and an official language matching the specified name. The operation filters countries based on their GNP and the official status of the given language.")
async def get_countries_gnp_official_language(gnp: int = Query(..., description="GNP value"), is_official: str = Query(..., description="Whether the language is official ('T' for true)"), language: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT T2.Name FROM CountryLanguage AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.GNP < ? AND T1.IsOfficial = ? AND T1.Language = ?", (gnp, is_official, language))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the GNP of a country based on the district name
@app.get("/v1/world/gnp_by_district", operation_id="get_gnp_by_district", summary="Retrieves the Gross National Product (GNP) of a country associated with the specified district. The district name is used to identify the relevant country and its corresponding GNP. Only the first matching result is returned.")
async def get_gnp_by_district(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT T2.GNP FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T1.District = ? LIMIT 1", (district_name,))
    result = cursor.fetchone()
    if not result:
        return {"gnp": []}
    return {"gnp": result[0]}

# Endpoint to get the local name of a country based on the city name
@app.get("/v1/world/local_name_by_city", operation_id="get_local_name_by_city", summary="Retrieves the local name of a country associated with the specified city. The operation uses the provided city name to search for a match in the City table, then retrieves the corresponding local name from the Country table.")
async def get_local_name_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T2.LocalName FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T1.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"local_name": []}
    return {"local_name": result[0]}

# Endpoint to get the names of countries and cities with a surface area greater than a specified value
@app.get("/v1/world/countries_cities_surface_area", operation_id="get_countries_cities_surface_area", summary="Retrieve the names of countries and their respective cities that have a surface area larger than the provided value. This operation allows you to filter countries and cities based on a specified surface area threshold.")
async def get_countries_cities_surface_area(surface_area: int = Query(..., description="Surface area value")):
    cursor.execute("SELECT T2.Name, T1.Name FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.SurfaceArea > ?", (surface_area,))
    result = cursor.fetchall()
    if not result:
        return {"countries_cities": []}
    return {"countries_cities": [{"country": row[0], "city": row[1]} for row in result]}

# Endpoint to get the count of cities in countries with a specific surface area
@app.get("/v1/world/city_count_by_country_surface_area", operation_id="get_city_count_by_country_surface_area", summary="Retrieves the number of cities in each country with a specified surface area. The surface area is provided as an input parameter.")
async def get_city_count_by_country_surface_area(surface_area: int = Query(..., description="Surface area of the country")):
    cursor.execute("SELECT T2.Name, COUNT(T1.Name) FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.SurfaceArea = ? GROUP BY T2.Name", (surface_area,))
    result = cursor.fetchall()
    if not result:
        return {"count": []}
    return {"count": result}

# Endpoint to get the languages spoken in countries that gained independence within a specific year range
@app.get("/v1/world/languages_by_independence_year_range", operation_id="get_languages_by_independence_year_range", summary="Retrieve the languages spoken in countries that gained independence between the specified start and end years. This operation returns a list of languages and their respective countries, filtered by the provided year range.")
async def get_languages_by_independence_year_range(start_year: int = Query(..., description="Start year of the independence range"), end_year: int = Query(..., description="End year of the independence range")):
    cursor.execute("SELECT T2.Name, T1.Language FROM CountryLanguage AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.IndepYear BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": result}

# Endpoint to get the life expectancy of a country based on a specific city name
@app.get("/v1/world/life_expectancy_by_city_name", operation_id="get_life_expectancy_by_city_name", summary="Retrieves the life expectancy of a country associated with a given city name. The operation uses the provided city name to look up the corresponding country's life expectancy in the database.")
async def get_life_expectancy_by_city_name(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T2.LifeExpectancy FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T1.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"life_expectancy": []}
    return {"life_expectancy": result[0]}

# Endpoint to get the languages spoken in countries with a specific head of state
@app.get("/v1/world/languages_by_head_of_state", operation_id="get_languages_by_head_of_state", summary="Retrieve the languages spoken in countries governed by a specific head of state. The operation requires the name of the head of state as an input parameter.")
async def get_languages_by_head_of_state(head_of_state: str = Query(..., description="Name of the head of state")):
    cursor.execute("SELECT T1.Language FROM CountryLanguage AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.HeadOfState = ?", (head_of_state,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": result}

# Endpoint to get the percentage of cities in a specific district within countries with a specific government form
@app.get("/v1/world/percentage_cities_in_district_by_government_form", operation_id="get_percentage_cities_in_district_by_government_form", summary="Retrieves the percentage of cities in a specified district that are located within countries sharing a particular form of government. The calculation is based on the total count of cities in the district and the count of cities in countries with the specified government form.")
async def get_percentage_cities_in_district_by_government_form(district: str = Query(..., description="Name of the district"), government_form: str = Query(..., description="Form of government")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.District = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T2.GovernmentForm = ?", (district, government_form))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the names of countries with cities in a specific population range and below-average life expectancy
@app.get("/v1/world/countries_with_cities_in_population_range_and_low_life_expectancy", operation_id="get_countries_with_cities_in_population_range_and_low_life", summary="Retrieve the names of countries that have cities with populations between the specified range and a life expectancy lower than 80% of the global average. The population range is defined by the minimum and maximum population parameters.")
async def get_countries_with_cities_in_population_range_and_low_life_expectancy(min_population: int = Query(..., description="Minimum population of the city"), max_population: int = Query(..., description="Maximum population of the city")):
    cursor.execute("SELECT T2.Name FROM City AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T1.Population BETWEEN ? AND ? GROUP BY T2.Name, LifeExpectancy HAVING LifeExpectancy < ( SELECT AVG(LifeExpectancy) FROM Country ) * 0.8", (min_population, max_population))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the percentage of countries with a specific government form that speak a specific language
@app.get("/v1/world/percentage_countries_by_government_form_and_language", operation_id="get_percentage_countries_by_government_form_and_language", summary="Retrieves the percentage of countries with a specified form of government that speak a particular language. This operation calculates the proportion of countries with the given government form and language, based on the total number of countries in the database.")
async def get_percentage_countries_by_government_form_and_language(government_form: str = Query(..., description="Form of government"), language: str = Query(..., description="Language spoken")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.GovernmentForm = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM CountryLanguage AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.Code WHERE T1.Language = ?", (government_form, language))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/world/country_with_highest_life_expectancy?limit=1",
    "/v1/world/countries_with_official_language?language=English&is_official=T&limit=5",
    "/v1/world/average_population_by_district?district=Karnataka",
    "/v1/world/languages_by_country?country_code=USA",
    "/v1/world/count_countries_by_language?language=Portuguese",
    "/v1/world/count_cities_by_district?district=England",
    "/v1/world/cities_in_largest_country",
    "/v1/world/capital_and_population_by_country?country_name=San%20Marino",
    "/v1/world/languages_by_country_name?country_name=Turkmenistan",
    "/v1/world/country_with_highest_life_expectancy_official_language?is_official=T&limit=1",
    "/v1/world/countries_by_continent_official_language?continent=Antarctica&is_official=T",
    "/v1/world/languages_by_region_percentage?region=Baltic%20Countries&percentage=80",
    "/v1/world/most_populous_city_life_expectancy",
    "/v1/world/capital_language_highest_life_expectancy",
    "/v1/world/smallest_country_official_language?is_official=T",
    "/v1/world/percentage_countries_language?language=English",
    "/v1/world/most_populous_city_district",
    "/v1/world/smallest_country_continent",
    "/v1/world/head_of_state_most_populous_city",
    "/v1/world/country_code_official_language_capital?language=English&is_official=T",
    "/v1/world/city_names_by_life_expectancy?life_expectancy=66.4",
    "/v1/world/head_of_state_by_language?language=English",
    "/v1/world/surface_area_by_city_name?city_name=Sutton%20Coldfield",
    "/v1/world/languages_by_population?population=8000",
    "/v1/world/official_languages_by_country?country_name=Belgium&is_official=T",
    "/v1/world/city_info_by_local_name?local_name=Hajastan",
    "/v1/world/count_country_name?country_name=Cyprus",
    "/v1/world/districts_by_head_of_state?head_of_state=Adolf%20Ogi",
    "/v1/world/head_of_state_by_district?district=Santa%20Catarina",
    "/v1/world/language_percentage_by_gnp?language=English&gnp=1500",
    "/v1/world/government_form_difference_by_language?government_form1=Republic&government_form2=ConstitutionalMonarchy&language=English",
    "/v1/world/countries_by_independence_year?indep_year=1994",
    "/v1/world/countries_by_continent?continent=Asia",
    "/v1/world/highest_gnp_country_by_continent?continent=Asia",
    "/v1/world/city_count_by_name?city_name=PHL",
    "/v1/world/local_name_by_country_name?country_name=Ukraine",
    "/v1/world/countries_by_government_form?government_form=Socialistic%20Republic",
    "/v1/world/language_percentage_by_country?country_name=China&language=Chinese",
    "/v1/world/government_form_by_city?city_name=Manila",
    "/v1/world/capital_by_country?country_name=Philipiines",
    "/v1/world/languages_by_continent?continent=Europe",
    "/v1/world/head_of_state_by_city?city_name=Pyongyang",
    "/v1/world/unofficial_languages_count_by_country?country_name=Italy",
    "/v1/world/least_populated_city_by_country?country_name=Russian%20Federation",
    "/v1/world/city_with_highest_life_expectancy",
    "/v1/world/languages_by_independence_year?independence_year=1830",
    "/v1/world/capital_of_most_populated_country",
    "/v1/world/countries_with_unofficial_languages_by_continent?continent=Asia",
    "/v1/world/average_gnp_by_language?language=Arabic",
    "/v1/world/percentage_surface_area_by_language?language=Chinese",
    "/v1/world/smallest_country_by_surface_area",
    "/v1/world/largest_country_by_population",
    "/v1/world/language_smallest_population_country",
    "/v1/world/countries_independent_after_year?indep_year=1990&is_official=T",
    "/v1/world/country_with_most_populous_city",
    "/v1/world/life_expectancy_most_populous_city?limit=1",
    "/v1/world/gnp_least_populous_city?limit=1",
    "/v1/world/city_most_languages?limit=1",
    "/v1/world/country_most_populous_city_largest_surface_area?limit=1",
    "/v1/world/cities_in_country?country_name=China",
    "/v1/world/cities_in_country_local_name?local_name=\u00c2\u00b4Uman",
    "/v1/world/avg_life_expectancy_by_language?language=Arabic",
    "/v1/world/gnp_growth_rate_by_city?city_name=Shanghai",
    "/v1/world/district_by_city_name?city_name=Zaanstad",
    "/v1/world/most_populous_city?limit=1",
    "/v1/world/city_district_by_population?population=201843",
    "/v1/world/largest_country_by_surface_area",
    "/v1/world/count_countries_by_life_expectancy?life_expectancy=75.1",
    "/v1/world/independence_year_by_country_name?name=Brunei",
    "/v1/world/count_countries_by_gnp?gnp=0",
    "/v1/world/average_surface_area",
    "/v1/world/count_distinct_languages_by_district?district=Tocantins",
    "/v1/world/districts_in_largest_country",
    "/v1/world/count_distinct_city_names_by_head_of_state?head_of_state=Kostis%20Stefanopoulos",
    "/v1/world/city_population?city_name=Queimados",
    "/v1/world/official_language_least_populated_city?is_official=T",
    "/v1/world/surface_area_gnp_by_district?district_name=Namibe",
    "/v1/world/district_smallest_surface_area",
    "/v1/world/countries_gnp_official_language?gnp=1000&is_official=T&language=Dutch",
    "/v1/world/gnp_by_district?district_name=Entre%20Rios",
    "/v1/world/local_name_by_city?city_name=The%20Valley",
    "/v1/world/countries_cities_surface_area?surface_area=7000000",
    "/v1/world/city_count_by_country_surface_area?surface_area=652090",
    "/v1/world/languages_by_independence_year_range?start_year=1980&end_year=1995",
    "/v1/world/life_expectancy_by_city_name?city_name=Calama",
    "/v1/world/languages_by_head_of_state?head_of_state=Pierre%20Buyoya",
    "/v1/world/percentage_cities_in_district_by_government_form?district=England&government_form=Constitutional%20Monarchy",
    "/v1/world/countries_with_cities_in_population_range_and_low_life_expectancy?min_population=140000&max_population=150000",
    "/v1/world/percentage_countries_by_government_form_and_language?government_form=Republic&language=Italian"
]
