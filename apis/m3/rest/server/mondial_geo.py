from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/mondial_geo/mondial_geo.sqlite')
cursor = conn.cursor()

# Endpoint to get the country name with the lowest percentage of a specific ethnic group
@app.get("/v1/mondial_geo/country_with_lowest_ethnic_percentage", operation_id="get_country_with_lowest_ethnic_percentage", summary="Retrieve the name of the country with the lowest percentage of a specified ethnic group. The operation filters the data based on the provided ethnic group name and returns the country name with the lowest percentage associated with the specified ethnic group.")
async def get_country_with_lowest_ethnic_percentage(ethnic_group: str = Query(..., description="Name of the ethnic group")):
    cursor.execute("SELECT T2.Name FROM ethnicGroup AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T1.Name = ? GROUP BY T2.Name, T1.Percentage ORDER BY T1.Percentage ASC LIMIT 1", (ethnic_group,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the country names with a specific ethnic group and percentage greater than a given value
@app.get("/v1/mondial_geo/countries_with_ethnic_percentage_above", operation_id="get_countries_with_ethnic_percentage_above", summary="Retrieve the names of countries where a specified ethnic group constitutes more than a given percentage of the population. The operation requires the name of the ethnic group and the minimum percentage threshold as input parameters.")
async def get_countries_with_ethnic_percentage_above(ethnic_group: str = Query(..., description="Name of the ethnic group"), percentage: float = Query(..., description="Percentage threshold")):
    cursor.execute("SELECT T2.Name FROM ethnicGroup AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T1.Name = ? AND T1.Percentage > ?", (ethnic_group, percentage))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the ethnic groups and their percentages in a specific country
@app.get("/v1/mondial_geo/ethnic_groups_in_country", operation_id="get_ethnic_groups_in_country", summary="Retrieve the names and percentages of ethnic groups present in a specified country. The operation returns a list of ethnic groups and their respective percentages, providing a comprehensive overview of the country's ethnic composition.")
async def get_ethnic_groups_in_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Name, T1.Percentage FROM ethnicGroup AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T2.Name = ? GROUP BY T1.Name, T1.Percentage", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"ethnic_groups": []}
    return {"ethnic_groups": [{"name": row[0], "percentage": row[1]} for row in result]}

# Endpoint to get the percentage of republics among countries that gained independence after a specific year
@app.get("/v1/mondial_geo/percentage_of_republics_after_year", operation_id="get_percentage_of_republics_after_year", summary="Retrieves the percentage of republics among countries that gained independence after the specified year. The calculation is based on the total number of countries that gained independence after the given year. The year should be provided in 'YYYY' format.")
async def get_percentage_of_republics_after_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Government = 'republic' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(Country) FROM politics WHERE STRFTIME('%Y', Independence) > ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the GDP and government type of a specific country
@app.get("/v1/mondial_geo/gdp_and_government_of_country", operation_id="get_gdp_and_government_of_country", summary="Retrieves the Gross Domestic Product (GDP) and the type of government of a specified country. The operation requires the name of the country as an input parameter and returns the corresponding GDP and government type.")
async def get_gdp_and_government_of_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.GDP, T2.Government FROM economy AS T1 INNER JOIN politics AS T2 ON T1.Country = T2.Country INNER JOIN country AS T3 ON T3.Code = T2.Country WHERE T3.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"gdp": [], "government": []}
    return {"gdp": result[0], "government": result[1]}

# Endpoint to get the countries with population growth above a threshold and infant mortality below a threshold
@app.get("/v1/mondial_geo/countries_with_population_growth_and_infant_mortality", operation_id="get_countries_with_population_growth_and_infant_mortality", summary="Retrieves a list of countries where the population growth rate surpasses a specified threshold and the infant mortality rate is below another specified threshold. The response includes the name and population of each qualifying country.")
async def get_countries_with_population_growth_and_infant_mortality(population_growth: float = Query(..., description="Population growth threshold"), infant_mortality: float = Query(..., description="Infant mortality threshold")):
    cursor.execute("SELECT T1.Name, T1.Population FROM country AS T1 INNER JOIN population AS T2 ON T1.Code = T2.Country WHERE T2.Population_Growth > ? AND T2.Infant_Mortality < ?", (population_growth, infant_mortality))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [{"name": row[0], "population": row[1]} for row in result]}

# Endpoint to get the top 2 ethnic groups by percentage in countries with a population above a threshold
@app.get("/v1/mondial_geo/top_ethnic_groups_by_population", operation_id="get_top_ethnic_groups_by_population", summary="Retrieves the names of the top 2 ethnic groups with the highest percentage in countries where the population exceeds the provided threshold. The population threshold is a customizable input parameter.")
async def get_top_ethnic_groups_by_population(population: int = Query(..., description="Population threshold")):
    cursor.execute("SELECT T2.Name FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T1.Population > ? GROUP BY T2.Name, T2.Percentage ORDER BY T2.Percentage DESC LIMIT 2", (population,))
    result = cursor.fetchall()
    if not result:
        return {"ethnic_groups": []}
    return {"ethnic_groups": [row[0] for row in result]}

# Endpoint to get the ethnic groups and their percentages in the country with the most ethnic groups
@app.get("/v1/mondial_geo/ethnic_groups_in_most_diverse_country", operation_id="get_ethnic_groups_in_most_diverse_country", summary="Retrieves the names and respective percentages of ethnic groups in the country with the highest number of distinct ethnic groups. The data is sourced from a relational database, where the country and ethnic group tables are joined based on a common code. The result is filtered to only include the country with the most diverse ethnic composition, and the output is grouped by country name, ethnic group name, and percentage.")
async def get_ethnic_groups_in_most_diverse_country():
    cursor.execute("SELECT T1.Name, T2.Name, T2.Percentage FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T1.Name = ( SELECT T1.Name FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country GROUP BY T1.Name ORDER BY COUNT(T2.Name) DESC LIMIT 1 ) GROUP BY T1.Name, T2.Name, T2.Percentage")
    result = cursor.fetchall()
    if not result:
        return {"ethnic_groups": []}
    return {"ethnic_groups": [{"country": row[0], "ethnic_group": row[1], "percentage": row[2]} for row in result]}

# Endpoint to get the country names with a specific ethnic group and percentage
@app.get("/v1/mondial_geo/countries_with_specific_ethnic_percentage", operation_id="get_countries_with_specific_ethnic_percentage", summary="Retrieve the names of countries where a specific ethnic group constitutes a given percentage of the population. The operation requires the name of the ethnic group and the desired percentage as input parameters.")
async def get_countries_with_specific_ethnic_percentage(percentage: float = Query(..., description="Percentage of the ethnic group"), ethnic_group: str = Query(..., description="Name of the ethnic group")):
    cursor.execute("SELECT T1.Name FROM ethnicGroup AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T1.Percentage = ? AND T1.Name = ?", (percentage, ethnic_group))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the infant mortality rate in the country with the lowest percentage of a specific ethnic group
@app.get("/v1/mondial_geo/infant_mortality_by_lowest_ethnic_percentage", operation_id="get_infant_mortality_by_lowest_ethnic_percentage", summary="Retrieve the infant mortality rate for the country with the lowest percentage of a specified ethnic group. The input parameter determines the ethnic group of interest.")
async def get_infant_mortality_by_lowest_ethnic_percentage(ethnic_group: str = Query(..., description="Name of the ethnic group")):
    cursor.execute("SELECT T1.Infant_Mortality FROM population AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Country = T2.Country WHERE T2.Name = ? ORDER BY T2.Percentage ASC LIMIT 1", (ethnic_group,))
    result = cursor.fetchone()
    if not result:
        return {"infant_mortality": []}
    return {"infant_mortality": result[0]}

# Endpoint to get agriculture data for countries with an area greater than a specified value
@app.get("/v1/mondial_geo/agriculture_by_area", operation_id="get_agriculture_by_area", summary="Retrieves agriculture data for countries with an area larger than the specified minimum value. The operation filters countries based on their area and returns the corresponding agriculture data from the economy table.")
async def get_agriculture_by_area(area: int = Query(..., description="Minimum area of the country")):
    cursor.execute("SELECT T2.Agriculture FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T1.Area > ? AND T2.Agriculture IS NOT NULL", (area,))
    result = cursor.fetchall()
    if not result:
        return {"agriculture": []}
    return {"agriculture": [row[0] for row in result]}

# Endpoint to get the country with the highest population growth for a specific government type
@app.get("/v1/mondial_geo/highest_population_growth_by_government", operation_id="get_highest_population_growth_by_government", summary="Retrieves the country with the highest population growth rate for a given government type. The operation filters countries based on the specified government type and sorts them by population growth in descending order. The country with the highest population growth is then returned.")
async def get_highest_population_growth_by_government(government: str = Query(..., description="Type of government (e.g., 'republic')")):
    cursor.execute("SELECT T2.Country FROM population AS T1 INNER JOIN politics AS T2 ON T1.Country = T2.Country WHERE T2.Government = ? ORDER BY T1.Population_Growth DESC LIMIT 1", (government,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the independence date of a specific country
@app.get("/v1/mondial_geo/independence_by_country", operation_id="get_independence_by_country", summary="Retrieves the independence date of a specified country. The operation uses the provided country name to look up the corresponding independence date in the database.")
async def get_independence_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Independence FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"independence": []}
    return {"independence": result[0]}

# Endpoint to get the population percentage of a specific ethnic group
@app.get("/v1/mondial_geo/ethnic_group_population_percentage", operation_id="get_ethnic_group_population_percentage", summary="Retrieves the percentage of a country's population that belongs to a specific ethnic group. The operation calculates this value by multiplying the total population of the country by the percentage of the population that belongs to the specified ethnic group.")
async def get_ethnic_group_population_percentage(ethnic_group: str = Query(..., description="Name of the ethnic group")):
    cursor.execute("SELECT T2.Percentage * T1.Population FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T2.Name = ?", (ethnic_group,))
    result = cursor.fetchall()
    if not result:
        return {"population_percentage": []}
    return {"population_percentage": [row[0] for row in result]}

# Endpoint to get the population percentage of a specific ethnic group in a specific country
@app.get("/v1/mondial_geo/ethnic_group_population_percentage_by_country", operation_id="get_ethnic_group_population_percentage_by_country", summary="Retrieves the estimated population of a specific ethnic group in a given country. The calculation is based on the percentage of the ethnic group's population in the country and the total population of the country.")
async def get_ethnic_group_population_percentage_by_country(ethnic_group: str = Query(..., description="Name of the ethnic group"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Percentage * T1.Population FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? AND T1.Name = ?", (ethnic_group, country_name))
    result = cursor.fetchone()
    if not result:
        return {"population_percentage": []}
    return {"population_percentage": result[0]}

# Endpoint to get the population growth of the country with the lowest infant mortality
@app.get("/v1/mondial_geo/lowest_infant_mortality_population_growth", operation_id="get_lowest_infant_mortality_population_growth", summary="Retrieves the population growth rate of the country with the lowest infant mortality rate. The calculation is based on the population and population growth data of the country.")
async def get_lowest_infant_mortality_population_growth():
    cursor.execute("SELECT T2.Population_Growth * T1.Population FROM country AS T1 INNER JOIN population AS T2 ON T1.Code = T2.Country WHERE T2.Infant_Mortality IS NOT NULL ORDER BY T2.Infant_Mortality ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"population_growth": []}
    return {"population_growth": result[0]}

# Endpoint to get the capital and population of countries with GDP greater than a specified value
@app.get("/v1/mondial_geo/capital_population_by_gdp", operation_id="get_capital_population_by_gdp", summary="Retrieves the capital and population of countries with a GDP surpassing the provided minimum value. This operation enables users to filter countries based on their economic output and view their respective capitals and populations.")
async def get_capital_population_by_gdp(gdp: int = Query(..., description="Minimum GDP of the country")):
    cursor.execute("SELECT T1.Capital, T1.Population FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T2.GDP > ?", (gdp,))
    result = cursor.fetchall()
    if not result:
        return {"capital_population": []}
    return {"capital_population": [{"capital": row[0], "population": row[1]} for row in result]}

# Endpoint to get the service sector contribution to GDP for a specific country
@app.get("/v1/mondial_geo/service_sector_gdp_by_country", operation_id="get_service_sector_gdp_by_country", summary="Retrieves the percentage of a country's Gross Domestic Product (GDP) that is attributed to the service sector. The operation requires the name of the country as an input parameter.")
async def get_service_sector_gdp_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Service * T2.GDP FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"service_sector_gdp": []}
    return {"service_sector_gdp": result[0]}

# Endpoint to get the country with the highest infant mortality and its population growth
@app.get("/v1/mondial_geo/highest_infant_mortality_population_growth", operation_id="get_highest_infant_mortality_population_growth", summary="Retrieves the name of the country with the highest infant mortality rate and its corresponding population growth rate. The data is fetched from the 'country' and 'population' tables, which are joined based on the country code. The result is sorted in descending order by the infant mortality rate and limited to the top entry.")
async def get_highest_infant_mortality_population_growth():
    cursor.execute("SELECT T1.Name, T2.Population_Growth FROM country AS T1 INNER JOIN population AS T2 ON T1.Code = T2.Country ORDER BY T2.Infant_Mortality DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country_population_growth": []}
    return {"country_population_growth": {"name": result[0], "population_growth": result[1]}}

# Endpoint to get countries with population growth less than a specified value
@app.get("/v1/mondial_geo/countries_by_negative_population_growth", operation_id="get_countries_by_negative_population_growth", summary="Retrieves a list of countries where the population growth rate is less than the specified value. The response includes the country name, current population, and population growth rate.")
async def get_countries_by_negative_population_growth(population_growth: float = Query(..., description="Maximum population growth rate")):
    cursor.execute("SELECT T1.Name, T1.Population, T2.Population_Growth FROM country AS T1 INNER JOIN population AS T2 ON T1.Code = T2.Country WHERE T2.Population_Growth < ?", (population_growth,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [{"name": row[0], "population": row[1], "population_growth": row[2]} for row in result]}

# Endpoint to get country names and infant mortality rates for countries with areas between specified values
@app.get("/v1/mondial_geo/country_infant_mortality_by_area", operation_id="get_country_infant_mortality_by_area", summary="Retrieves the names and infant mortality rates of countries with areas falling within the specified range. The range is defined by a minimum and maximum area value, which are provided as input parameters.")
async def get_country_infant_mortality_by_area(min_area: int = Query(..., description="Minimum area of the country"), max_area: int = Query(..., description="Maximum area of the country")):
    cursor.execute("SELECT T1.Name, T2.Infant_Mortality FROM country AS T1 INNER JOIN population AS T2 ON T1.Code = T2.Country WHERE T1.Area BETWEEN ? AND ?", (min_area, max_area))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get country names and GDP for countries with population growth greater than a specified value
@app.get("/v1/mondial_geo/country_gdp_by_population_growth", operation_id="get_country_gdp_by_population_growth", summary="Retrieves the names and GDP of countries where the population growth rate surpasses a specified threshold. This operation allows users to filter countries based on their population growth and provides their corresponding GDP data.")
async def get_country_gdp_by_population_growth(population_growth: float = Query(..., description="Population growth rate")):
    cursor.execute("SELECT T1.Name, T3.GDP FROM country AS T1 INNER JOIN population AS T2 ON T1.Code = T2.Country INNER JOIN economy AS T3 ON T3.Country = T2.Country WHERE T2.Population_Growth > ?", (population_growth,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get infant mortality rate for a specified country
@app.get("/v1/mondial_geo/infant_mortality_by_country", operation_id="get_infant_mortality_by_country", summary="Retrieves the infant mortality rate for a specified country. The operation uses the provided country name to look up the corresponding infant mortality rate from the population data, which is then returned.")
async def get_infant_mortality_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Infant_Mortality FROM country AS T1 INNER JOIN population AS T2 ON T1.Code = T2.Country WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"infant_mortality": []}
    return {"infant_mortality": result[0]}

# Endpoint to get the product of GDP and Industry for a specified country
@app.get("/v1/mondial_geo/gdp_industry_product_by_country", operation_id="get_gdp_industry_product_by_country", summary="Retrieves the product of a country's Gross Domestic Product (GDP) and its industrial output. The operation requires the name of the country as an input parameter and returns the calculated product value.")
async def get_gdp_industry_product_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.GDP * T2.Industry FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"gdp_industry_product": []}
    return {"gdp_industry_product": result[0]}

# Endpoint to get the product of GDP and Agriculture for the country with the smallest area
@app.get("/v1/mondial_geo/gdp_agriculture_product_smallest_area", operation_id="get_gdp_agriculture_product_smallest_area", summary="Retrieves the combined value of Gross Domestic Product (GDP) and Agriculture for the country with the smallest geographical area. The calculation is based on the economy data and the country's area, with the result being the product of GDP and Agriculture for the country with the least area.")
async def get_gdp_agriculture_product_smallest_area():
    cursor.execute("SELECT T2.GDP * T2.Agriculture FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country ORDER BY T1.Area ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"gdp_agriculture_product": []}
    return {"gdp_agriculture_product": result[0]}

# Endpoint to get the country with the highest percentage of a specified ethnic group
@app.get("/v1/mondial_geo/country_highest_ethnic_group_percentage", operation_id="get_country_highest_ethnic_group_percentage", summary="Retrieves the name of the country with the highest percentage of the specified ethnic group. The operation uses the provided ethnic group name to search for matching records in the country and ethnicGroup tables, then orders the results by the percentage of the ethnic group in descending order. The name of the country with the highest percentage is returned.")
async def get_country_highest_ethnic_group_percentage(ethnic_group: str = Query(..., description="Name of the ethnic group")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? ORDER BY T2.Percentage DESC LIMIT 1", (ethnic_group,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of countries with a specified ethnic group and area greater than a specified value
@app.get("/v1/mondial_geo/count_countries_ethnic_group_area", operation_id="get_count_countries_ethnic_group_area", summary="Retrieves the number of countries that have a specified ethnic group and an area larger than a given value. The input parameters define the ethnic group and the minimum area threshold.")
async def get_count_countries_ethnic_group_area(ethnic_group: str = Query(..., description="Name of the ethnic group"), min_area: int = Query(..., description="Minimum area of the country")):
    cursor.execute("SELECT COUNT(T1.Name) FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? AND T1.Area > ?", (ethnic_group, min_area))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get countries with more than a specified number of ethnic groups
@app.get("/v1/mondial_geo/countries_with_multiple_ethnic_groups", operation_id="get_countries_with_multiple_ethnic_groups", summary="Retrieve a list of countries that have more than the specified minimum number of ethnic groups. This operation filters countries based on the number of distinct ethnic groups present, providing a focused view of countries with diverse populations.")
async def get_countries_with_multiple_ethnic_groups(min_ethnic_groups: int = Query(..., description="Minimum number of ethnic groups")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country GROUP BY T1.Name HAVING COUNT(T1.Name) > ?", (min_ethnic_groups,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the country with the highest GDP
@app.get("/v1/mondial_geo/country_highest_gdp", operation_id="get_country_highest_gdp", summary="Retrieves the name of the country with the highest Gross Domestic Product (GDP) from the database. The operation uses an inner join between the country and economy tables to determine the country with the highest GDP.")
async def get_country_highest_gdp():
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country ORDER BY T2.GDP DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of countries with GDP and population greater than specified values
@app.get("/v1/mondial_geo/count_countries_gdp_population", operation_id="get_count_countries_gdp_population", summary="Retrieves the number of countries with a GDP and population exceeding the provided minimum values. This operation considers the economic and demographic data of each country to determine the count.")
async def get_count_countries_gdp_population(min_gdp: int = Query(..., description="Minimum GDP"), min_population: int = Query(..., description="Minimum population")):
    cursor.execute("SELECT COUNT(T1.Name) FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T2.GDP > ? AND T1.Population > ?", (min_gdp, min_population))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the capital of countries with inflation less than a specified value
@app.get("/v1/mondial_geo/capital_by_inflation", operation_id="get_capital_by_inflation", summary="Retrieve the capital cities of countries with an inflation rate below the specified threshold. This operation filters countries based on their inflation rate and returns the corresponding capital cities.")
async def get_capital_by_inflation(inflation: float = Query(..., description="Maximum inflation value")):
    cursor.execute("SELECT T1.Capital FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T2.Inflation < ?", (inflation,))
    result = cursor.fetchall()
    if not result:
        return {"capitals": []}
    return {"capitals": [row[0] for row in result]}

# Endpoint to get the country with the lowest inflation
@app.get("/v1/mondial_geo/country_with_lowest_inflation", operation_id="get_country_with_lowest_inflation", summary="Retrieves the name of the country with the lowest inflation rate. The operation filters out countries with no inflation data and returns the country with the lowest inflation rate.")
async def get_country_with_lowest_inflation():
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T2.Inflation IS NOT NULL ORDER BY T2.Inflation ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of countries with agriculture less than a specified value and area greater than a specified value
@app.get("/v1/mondial_geo/count_countries_by_agriculture_area", operation_id="get_count_countries_by_agriculture_area", summary="Retrieves the number of countries with an agricultural output below a specified maximum and a total area above a specified minimum. This operation considers the relationship between the agriculture and economy tables to determine the countries that meet the criteria.")
async def get_count_countries_by_agriculture_area(agriculture: float = Query(..., description="Maximum agriculture value"), area: float = Query(..., description="Minimum area value")):
    cursor.execute("SELECT COUNT(T1.Name) FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T2.Agriculture < ? AND T1.Area > ?", (agriculture, area))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cities located near a lake of a specified type
@app.get("/v1/mondial_geo/count_cities_by_lake_type", operation_id="get_count_cities_by_lake_type", summary="Retrieves the total number of cities situated near lakes of a specific type. The lake type is provided as an input parameter.")
async def get_count_cities_by_lake_type(lake_type: str = Query(..., description="Type of the lake")):
    cursor.execute("SELECT COUNT(T1.City) FROM located AS T1 INNER JOIN lake AS T2 ON T1.Lake = T2.Name WHERE T2.Type = ?", (lake_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the depth of lakes in a specified province
@app.get("/v1/mondial_geo/lake_depth_by_province", operation_id="get_lake_depth_by_province", summary="Retrieves the depth of all lakes located within a specified province. The operation returns a list of lake depths for the given province.")
async def get_lake_depth_by_province(province: str = Query(..., description="Province name")):
    cursor.execute("SELECT T2.Depth FROM located AS T1 INNER JOIN lake AS T2 ON T1.Lake = T2.Name WHERE T1.Province = ?", (province,))
    result = cursor.fetchall()
    if not result:
        return {"depths": []}
    return {"depths": [row[0] for row in result]}

# Endpoint to get the city with the highest altitude lake
@app.get("/v1/mondial_geo/city_with_highest_altitude_lake", operation_id="get_city_with_highest_altitude_lake", summary="Retrieves the name of the city that hosts the lake with the highest altitude. The operation returns the city name as a single string value.")
async def get_city_with_highest_altitude_lake():
    cursor.execute("SELECT T2.City FROM lake AS T1 LEFT JOIN located AS T2 ON T2.Lake = T1.Name ORDER BY T1.Altitude DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the count of lakes in a specified province with an area greater than a specified value
@app.get("/v1/mondial_geo/count_lakes_by_province_area", operation_id="get_count_lakes_by_province_area", summary="Retrieve the number of lakes in a specified province that have an area greater than a given value. This operation requires the name of the province and the minimum area value as input parameters.")
async def get_count_lakes_by_province_area(province: str = Query(..., description="Province name"), area: float = Query(..., description="Minimum area value")):
    cursor.execute("SELECT COUNT(T2.Name) FROM located AS T1 INNER JOIN lake AS T2 ON T1.Lake = T2.Name WHERE T1.Province = ? AND T2.Area > ?", (province, area))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country with the most languages
@app.get("/v1/mondial_geo/country_with_most_languages", operation_id="get_country_with_most_languages", summary="Retrieves the name of the country that has the highest number of languages spoken within its borders. This operation utilizes a complex query to aggregate and count the number of languages per country, then orders the results in descending order to identify the country with the most languages.")
async def get_country_with_most_languages():
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN language AS T2 ON T1.Code = T2.Country GROUP BY T1.Name ORDER BY COUNT(T2.Name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the capital of countries where a specified language is spoken by more than a specified percentage
@app.get("/v1/mondial_geo/capital_by_language_percentage", operation_id="get_capital_by_language_percentage", summary="Retrieve the capital cities of countries where the specified language is spoken by more than the given percentage of the population. The operation requires the language name and the minimum percentage value as input parameters.")
async def get_capital_by_language_percentage(language: str = Query(..., description="Language name"), percentage: float = Query(..., description="Minimum percentage value")):
    cursor.execute("SELECT T1.Capital FROM country AS T1 INNER JOIN language AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? AND T2.Percentage > ?", (language, percentage))
    result = cursor.fetchall()
    if not result:
        return {"capitals": []}
    return {"capitals": [row[0] for row in result]}

# Endpoint to get countries with a population less than a specified value and more than a specified number of languages
@app.get("/v1/mondial_geo/countries_by_population_languages", operation_id="get_countries_by_population_languages", summary="Retrieves a list of countries with a population less than the specified maximum value and a number of languages greater than the specified minimum. The data is filtered based on the provided population and language count parameters.")
async def get_countries_by_population_languages(population: int = Query(..., description="Maximum population value"), language_count: int = Query(..., description="Minimum number of languages")):
    cursor.execute("SELECT T2.Country FROM country AS T1 INNER JOIN language AS T2 ON T1.Code = T2.Country WHERE T1.Population < ? GROUP BY T2.Country HAVING COUNT(T1.Name) > ?", (population, language_count))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the count of organizations in countries with a population less than a specified value
@app.get("/v1/mondial_geo/count_organizations_by_population", operation_id="get_count_organizations_by_population", summary="Retrieve the total number of organizations located in countries with a population below the specified threshold.")
async def get_count_organizations_by_population(population: int = Query(..., description="Maximum population of the country")):
    cursor.execute("SELECT COUNT(T2.Name) FROM country AS T1 INNER JOIN organization AS T2 ON T1.Code = T2.Country WHERE T1.Population < ?", (population,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of organizations in countries with GDP less than a specified value and established before a specified year
@app.get("/v1/mondial_geo/count_organizations_by_gdp_and_year", operation_id="get_count_organizations_by_gdp_and_year", summary="Retrieves the number of organizations in countries with a GDP below a specified threshold and established before a certain year. The response includes a count of organizations per country that meet the criteria.")
async def get_count_organizations_by_gdp_and_year(gdp: int = Query(..., description="Maximum GDP of the country"), year: str = Query(..., description="Year established in 'YYYY' format")):
    cursor.execute("SELECT T1.Country, COUNT(T1.Country) FROM economy AS T1 INNER JOIN organization AS T2 ON T1.Country = T2.Country WHERE T1.GDP < ? AND STRFTIME('%Y', T2.Established) < ? GROUP BY T1.Country", (gdp, year))
    result = cursor.fetchall()
    if not result:
        return {"count": []}
    return {"count": result}

# Endpoint to get the count of countries with more than a specified number of organizations and inflation greater than a specified value
@app.get("/v1/mondial_geo/count_countries_by_organizations_and_inflation", operation_id="get_count_countries_by_organizations_and_inflation", summary="Retrieves the count of countries that have more than a specified number of organizations and an inflation rate exceeding a given value. The operation filters countries based on the provided minimum organization count and inflation rate, then returns the total number of countries that meet these criteria.")
async def get_count_countries_by_organizations_and_inflation(organization_count: int = Query(..., description="Minimum number of organizations in the country"), inflation: float = Query(..., description="Minimum inflation rate of the country")):
    cursor.execute("SELECT COUNT(T2.Country) FROM economy AS T1 INNER JOIN organization AS T2 ON T1.Country = T2.Country WHERE T2.Country IN ( SELECT Country FROM organization GROUP BY Country HAVING COUNT(Country) > ? ) AND T1.Inflation > ?", (organization_count, inflation))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country with the highest number of ethnic groups
@app.get("/v1/mondial_geo/country_with_most_ethnic_groups", operation_id="get_country_with_most_ethnic_groups", summary="Retrieves the country with the highest number of distinct ethnic groups. This operation calculates the count of unique ethnic groups per country by joining the 'country', 'organization', and 'ethnicGroup' tables. The result is ordered in descending order based on the count of ethnic groups, and the top country is returned.")
async def get_country_with_most_ethnic_groups():
    cursor.execute("SELECT COUNT(T2.Province) FROM country AS T1 INNER JOIN organization AS T2 ON T1.Code = T2.Country INNER JOIN ethnicGroup AS T3 ON T3.Country = T2.Country GROUP BY T1.Name ORDER BY COUNT(T3.Name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of organizations in countries where a specific language is spoken
@app.get("/v1/mondial_geo/organizations_by_language", operation_id="get_organizations_by_language", summary="Retrieve the names of organizations based in countries where the specified language is spoken. The operation requires the name of the language as an input parameter.")
async def get_organizations_by_language(language: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT T2.Name FROM language AS T1 INNER JOIN organization AS T2 ON T1.Country = T2.Country WHERE T1.Name = ?", (language,))
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": [row[0] for row in result]}

# Endpoint to get the count of organizations in countries where a specific language is spoken
@app.get("/v1/mondial_geo/count_organizations_by_language", operation_id="get_count_organizations_by_language", summary="Retrieves the total number of organizations located in countries where the specified language is spoken. The language is provided as an input parameter.")
async def get_count_organizations_by_language(language: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT COUNT(T2.Name) FROM language AS T1 INNER JOIN organization AS T2 ON T1.Country = T2.Country WHERE T1.Name = ?", (language,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum infant mortality rate in countries with inflation less than a specified value
@app.get("/v1/mondial_geo/max_infant_mortality_by_inflation", operation_id="get_max_infant_mortality_by_inflation", summary="Retrieves the highest infant mortality rate among countries with an inflation rate below the specified value.")
async def get_max_infant_mortality_by_inflation(inflation: float = Query(..., description="Maximum inflation rate of the country")):
    cursor.execute("SELECT MAX(T2.Infant_Mortality) FROM economy AS T1 INNER JOIN population AS T2 ON T1.Country = T2.Country WHERE T1.Inflation < ?", (inflation,))
    result = cursor.fetchone()
    if not result:
        return {"max_infant_mortality": []}
    return {"max_infant_mortality": result[0]}

# Endpoint to get the count of countries with GDP greater than a specified value and population growth greater than a specified value
@app.get("/v1/mondial_geo/count_countries_by_gdp_and_population_growth", operation_id="get_count_countries_by_gdp_and_population_growth", summary="Retrieves the number of countries with a GDP and population growth rate that exceed the provided thresholds. The GDP and population growth rate parameters are used to filter the countries.")
async def get_count_countries_by_gdp_and_population_growth(gdp: int = Query(..., description="Minimum GDP of the country"), population_growth: float = Query(..., description="Minimum population growth rate of the country")):
    cursor.execute("SELECT COUNT(T1.Country) FROM economy AS T1 INNER JOIN population AS T2 ON T1.Country = T2.Country WHERE T1.GDP > ? AND T2.Population_Growth > ?", (gdp, population_growth))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country with the highest GDP per capita
@app.get("/v1/mondial_geo/country_with_highest_gdp_per_capita", operation_id="get_country_with_highest_gdp_per_capita", summary="Retrieves the name of the country with the highest GDP per capita. This operation calculates the GDP per capita for each country by dividing the GDP by the population, then returns the country with the highest calculated value.")
async def get_country_with_highest_gdp_per_capita():
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country ORDER BY T2.GDP / T1.Population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the percentage of the area of the largest lake relative to the country it is located in
@app.get("/v1/mondial_geo/largest_lake_area_percentage", operation_id="get_largest_lake_area_percentage", summary="Retrieves the percentage of the area of the largest lake in a country relative to the total area of that country. The lake is selected based on its longitude, with the lake having the highest longitude value being considered the largest.")
async def get_largest_lake_area_percentage():
    cursor.execute("SELECT T2.Area * 100  / T3.Area FROM located AS T1 INNER JOIN lake AS T2 ON T1.Lake = T2.Name INNER JOIN country AS T3 ON T3.Code = T1.Country ORDER BY T2.Longitude DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average population growth for countries with more than a specified number of languages
@app.get("/v1/mondial_geo/average_population_growth_by_language_count", operation_id="get_average_population_growth", summary="Retrieves the average population growth rate for countries where the number of spoken languages exceeds a specified threshold. The calculation is based on the sum of population growth rates divided by the count of qualifying countries.")
async def get_average_population_growth(language_count: int = Query(..., description="Number of languages spoken in the country")):
    cursor.execute("SELECT SUM(T3.Population_Growth) / COUNT(T3.Country) FROM country AS T1 INNER JOIN language AS T2 ON T1.Code = T2.Country INNER JOIN population AS T3 ON T3.Country = T2.Country WHERE T2.Country IN ( SELECT Country FROM language GROUP BY Country HAVING COUNT(Country) > ? ) GROUP BY T3.Country", (language_count,))
    result = cursor.fetchall()
    if not result:
        return {"average_population_growth": []}
    return {"average_population_growth": [row[0] for row in result]}

# Endpoint to get countries with inflation higher than a specified multiple of the average inflation
@app.get("/v1/mondial_geo/countries_with_high_inflation", operation_id="get_countries_with_high_inflation", summary="Retrieves a list of countries with an inflation rate that exceeds the average inflation rate by a specified multiple. The multiplier is provided as an input parameter, allowing for customization of the inflation threshold.")
async def get_countries_with_high_inflation(inflation_multiplier: float = Query(..., description="Multiplier for the average inflation")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country GROUP BY T1.Name, T2.Inflation HAVING T2.Inflation > AVG(T2.Inflation) * ?", (inflation_multiplier,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get countries by province
@app.get("/v1/mondial_geo/countries_by_province", operation_id="get_countries_by_province", summary="Retrieves a list of country names that belong to the specified province. The operation filters the countries based on the provided province name.")
async def get_countries_by_province(province: str = Query(..., description="Province name")):
    cursor.execute("SELECT Name FROM country WHERE Province = ?", (province,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the most populous religion in a specified country
@app.get("/v1/mondial_geo/most_populous_religion_by_country", operation_id="get_most_populous_religion", summary="Retrieves the most widely practiced religion in the specified country, based on the country's population. The country is identified by its name.")
async def get_most_populous_religion(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Name FROM country AS T1 INNER JOIN religion AS T2 ON T1.Code = T2.Country WHERE T1.Name = ? ORDER BY T1.population DESC LIMIT 1", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"religion": []}
    return {"religion": result[0]}

# Endpoint to get countries with a specific religion and percentage
@app.get("/v1/mondial_geo/countries_by_religion_and_percentage", operation_id="get_countries_by_religion_and_percentage", summary="Retrieves a list of countries where a specified religion is practiced by a given percentage of the population. The operation filters countries based on the provided religion name and the corresponding percentage of adherents.")
async def get_countries_by_religion_and_percentage(religion_name: str = Query(..., description="Name of the religion"), percentage: int = Query(..., description="Percentage of the religion")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN religion AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? AND T2.Percentage = ?", (religion_name, percentage))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get countries located on a specific river
@app.get("/v1/mondial_geo/countries_on_river", operation_id="get_countries_on_river", summary="Retrieves the names of all countries that are located on a specified river. The operation requires the name of the river as an input parameter.")
async def get_countries_on_river(river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT T3.Name FROM located AS T1 INNER JOIN river AS T2 ON T1.River = T2.Name INNER JOIN country AS T3 ON T3.Code = T1.Country WHERE T2.Name = ?", (river_name,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the longest border between two countries
@app.get("/v1/mondial_geo/longest_border", operation_id="get_longest_border", summary="Retrieves the two countries that share the longest border. The operation identifies the countries with the longest shared border by comparing the lengths of all borders and returns the top result.")
async def get_longest_border():
    cursor.execute("SELECT T2.Country1, T2.Country2 FROM country AS T1 INNER JOIN borders AS T2 ON T1.Code = T2.Country1 ORDER BY T2.Length DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"border": []}
    return {"border": {"country1": result[0], "country2": result[1]}}

# Endpoint to get the country with the most borders
@app.get("/v1/mondial_geo/country_with_most_borders", operation_id="get_country_with_most_borders", summary="Retrieves the name of the country that shares the most borders with other countries. The operation uses an internal algorithm to determine the country with the highest number of shared borders.")
async def get_country_with_most_borders():
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN borders AS T2 ON T1.Code = T2.Country1 GROUP BY T1.Name ORDER BY COUNT(T1.Name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get countries with a specific mountain
@app.get("/v1/mondial_geo/countries_with_mountain", operation_id="get_countries_with_mountain", summary="Retrieves a list of distinct countries that have a mountain with the specified name. The input parameter is used to identify the mountain by its name.")
async def get_countries_with_mountain(mountain_name: str = Query(..., description="Name of the mountain")):
    cursor.execute("SELECT DISTINCT T1.Name FROM country AS T1 INNER JOIN geo_mountain AS T2 ON T1.Code = T2.Country WHERE T2.Mountain = ?", (mountain_name,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the count of distinct mountains in a specified country
@app.get("/v1/mondial_geo/count_mountains_in_country", operation_id="get_count_mountains_in_country", summary="Retrieves the total number of unique mountains located within a specified country. The operation requires the name of the country as an input parameter.")
async def get_count_mountains_in_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT COUNT(DISTINCT T2.Mountain) FROM country AS T1 INNER JOIN geo_mountain AS T2 ON T1.Code = T2.Country WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latitude of an island where a specific mountain is located
@app.get("/v1/mondial_geo/latitude_of_island_with_mountain", operation_id="get_latitude_of_island", summary="Retrieves the latitude of an island that hosts a specific mountain. The operation requires the name of the mountain as an input parameter to identify the island and return its latitude.")
async def get_latitude_of_island(mountain_name: str = Query(..., description="Name of the mountain")):
    cursor.execute("SELECT T1.Latitude FROM island AS T1 INNER JOIN mountainOnIsland AS T2 ON T1.Name = T2.Island WHERE T2.Mountain = ?", (mountain_name,))
    result = cursor.fetchone()
    if not result:
        return {"latitude": []}
    return {"latitude": result[0]}

# Endpoint to get the country code of the country with the second highest mountain
@app.get("/v1/mondial_geo/country_code_with_second_highest_mountain", operation_id="get_country_code_with_second_highest_mountain", summary="Retrieves the country code of the nation that hosts the second highest mountain in the world. This operation identifies the mountain with the second highest elevation and returns the country code of the nation where it is located. The result is determined by comparing the heights of all mountains in the database and selecting the second highest.")
async def get_country_code_with_second_highest_mountain():
    cursor.execute("SELECT T1.Code FROM country AS T1 INNER JOIN geo_mountain AS T2 ON T1.Code = T2.Country WHERE T2.Mountain = ( SELECT Name FROM mountain ORDER BY Height DESC LIMIT 1, 1 )")
    result = cursor.fetchone()
    if not result:
        return {"country_code": []}
    return {"country_code": result[0]}

# Endpoint to get the percentage of a country in a continent
@app.get("/v1/mondial_geo/percentage_of_country_in_continent", operation_id="get_percentage_of_country_in_continent", summary="Retrieves the percentage of a specified country that is located within a given continent. The operation requires the names of both the continent and the country as input parameters.")
async def get_percentage_of_country_in_continent(continent_name: str = Query(..., description="Name of the continent"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Percentage FROM country AS T1 INNER JOIN encompasses AS T2 ON T1.Code = T2.Country INNER JOIN continent AS T3 ON T3.Name = T2.Continent WHERE T3.Name = ? AND T1.Name = ?", (continent_name, country_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of a country's area relative to a continent's area
@app.get("/v1/mondial_geo/percentage_of_country_area_in_continent", operation_id="get_percentage_of_country_area_in_continent", summary="Retrieves the percentage of a country's area relative to the total area of a specified continent. The operation requires the names of both the continent and the country as input parameters.")
async def get_percentage_of_country_area_in_continent(continent_name: str = Query(..., description="Name of the continent"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Area * 100 / T3.Area FROM country AS T1 INNER JOIN encompasses AS T2 ON T1.Code = T2.Country INNER JOIN continent AS T3 ON T3.Name = T2.Continent WHERE T3.Name = ? AND T1.Name = ?", (continent_name, country_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the most populous city in a given country
@app.get("/v1/mondial_geo/most_populous_city_in_country", operation_id="get_most_populous_city", summary="Retrieves the name of the city with the highest population in a specified country. The operation uses the provided country name to search for the corresponding country record and then identifies the city with the maximum population within that country.")
async def get_most_populous_city(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Name FROM country AS T1 INNER JOIN city AS T2 ON T1.Code = T2.Country WHERE T1.Name = ? ORDER BY T2.Population DESC LIMIT 1", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the capital of the country where a specific city is located
@app.get("/v1/mondial_geo/capital_of_country_with_city", operation_id="get_capital_of_country_with_city", summary="Retrieves the capital of the country where the specified city is located. The operation uses the provided city name to identify the country and subsequently returns its capital.")
async def get_capital_of_country_with_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.Capital FROM country AS T1 INNER JOIN city AS T2 ON T1.Code = T2.Country WHERE T2.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"capital": []}
    return {"capital": result[0]}

# Endpoint to get the province of the highest volcano in a country
@app.get("/v1/mondial_geo/province_of_highest_volcano", operation_id="get_province_of_highest_volcano", summary="Retrieves the province with the highest volcano in a country. The operation filters mountains by a specified type and orders them by height in descending order. The province of the highest mountain of the specified type is then returned.")
async def get_province_of_highest_volcano(mountain_type: str = Query(..., description="Type of the mountain (e.g., 'volcano')")):
    cursor.execute("SELECT T1.Province FROM country AS T1 INNER JOIN geo_mountain AS T2 ON T1.Code = T2.Country INNER JOIN mountain AS T3 ON T3.Name = T2.Mountain WHERE T3.Type = ? ORDER BY T3.Height DESC LIMIT 1", (mountain_type,))
    result = cursor.fetchone()
    if not result:
        return {"province": []}
    return {"province": result[0]}

# Endpoint to get the government type of a given country
@app.get("/v1/mondial_geo/government_type_of_country", operation_id="get_government_type", summary="Retrieves the government type of a specified country. The operation uses the provided country name to search for a match in the database and returns the corresponding government type.")
async def get_government_type(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Government FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"government_type": []}
    return {"government_type": result[0]}

# Endpoint to get the name of the country located on an island with a specific name
@app.get("/v1/mondial_geo/country_name_on_island", operation_id="get_country_name_on_island", summary="Retrieves the name of the country located on an island, given the name of the country. This operation returns the name of the country that matches the provided country name and is located on an island.")
async def get_country_name_on_island(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T3.Name FROM locatedOn AS T1 INNER JOIN island AS T2 ON T1.Island = T2.Name INNER JOIN country AS T3 ON T3.Code = T1.Country WHERE T3.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the capital of the country with the highest percentage of a specific ethnic group
@app.get("/v1/mondial_geo/capital_by_ethnic_group", operation_id="get_capital_by_ethnic_group", summary="Retrieves the capital city of the country with the highest percentage of the specified ethnic group. The operation uses the provided ethnic group name to search for the corresponding country and returns the capital of the country with the highest percentage of that ethnic group.")
async def get_capital_by_ethnic_group(ethnic_group: str = Query(..., description="Name of the ethnic group")):
    cursor.execute("SELECT T1.Capital FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? ORDER BY T2.Percentage DESC LIMIT 1", (ethnic_group,))
    result = cursor.fetchone()
    if not result:
        return {"capital": []}
    return {"capital": result[0]}

# Endpoint to get the name of the ethnic group with the highest population percentage in the second largest country by area
@app.get("/v1/mondial_geo/ethnic_group_in_second_largest_country", operation_id="get_ethnic_group_in_second_largest_country", summary="Retrieves the name of the ethnic group with the highest population percentage in the second largest country by area. The operation calculates the population percentage of each ethnic group in the specified country and returns the group with the highest value.")
async def get_ethnic_group_in_second_largest_country():
    cursor.execute("SELECT T2.Name FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T1.Name = ( SELECT Name FROM country ORDER BY Area DESC LIMIT 2, 1 ) GROUP BY T2.Name ORDER BY T2.Percentage * T1.Population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ethnic_group": []}
    return {"ethnic_group": result[0]}

# Endpoint to get the name of the country with a city of a specific population
@app.get("/v1/mondial_geo/country_by_city_population", operation_id="get_country_by_city_population", summary="Retrieve the name of the country that contains a city with the specified population. The population value is used to filter the cities and identify the corresponding country.")
async def get_country_by_city_population(city_population: int = Query(..., description="Population of the city")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN city AS T2 ON T1.Code = T2.Country WHERE T2.Population = ?", (city_population,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of rivers flowing into a sea with a specific depth
@app.get("/v1/mondial_geo/river_count_by_sea_depth", operation_id="get_river_count_by_sea_depth", summary="Retrieves the total number of rivers that flow into seas with a specified depth. The depth of the sea is provided as an input parameter.")
async def get_river_count_by_sea_depth(sea_depth: int = Query(..., description="Depth of the sea")):
    cursor.execute("SELECT COUNT(*) FROM river WHERE Sea IN ( SELECT Name FROM sea WHERE Depth = ? )", (sea_depth,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the area of the country that gained independence on a specific date
@app.get("/v1/mondial_geo/country_area_by_independence_date", operation_id="get_country_area_by_independence_date", summary="Retrieves the area of the country that gained independence on the specified date. The input parameter is the independence date in 'YYYY-MM-DD' format. The operation returns the area of the country that matches the provided independence date.")
async def get_country_area_by_independence_date(independence_date: str = Query(..., description="Independence date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.Area FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country WHERE T2.Independence = ?", (independence_date,))
    result = cursor.fetchone()
    if not result:
        return {"area": []}
    return {"area": result[0]}

# Endpoint to get the population density of the country with a specific city
@app.get("/v1/mondial_geo/population_density_by_city", operation_id="get_population_density_by_city", summary="Retrieves the population density of the country that contains the specified city. The population density is calculated by dividing the country's total population by its area. The input parameter is the name of the city, which is used to identify the corresponding country.")
async def get_population_density_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT CAST(T1.Population AS REAL) / T1.Area FROM country AS T1 INNER JOIN city AS T2 ON T1.Code = T2.Country WHERE T2.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"population_density": []}
    return {"population_density": result[0]}

# Endpoint to get the population difference between two ethnic groups in a specific country
@app.get("/v1/mondial_geo/ethnic_group_population_difference", operation_id="get_ethnic_group_population_difference", summary="Retrieves the population difference between two specified ethnic groups within a given country. The calculation is based on the percentage of each ethnic group's population relative to the total population of the country.")
async def get_ethnic_group_population_difference(ethnic_group1: str = Query(..., description="Name of the first ethnic group"), ethnic_group2: str = Query(..., description="Name of the second ethnic group"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T3.Population * (T2.Percentage - T1.Percentage) FROM ethnicGroup AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Country = T2.Country INNER JOIN country AS T3 ON T1.Country = T3.Code WHERE T1.Name = ? AND T2.Name = ? AND T3.Name = ?", (ethnic_group1, ethnic_group2, country_name))
    result = cursor.fetchone()
    if not result:
        return {"population_difference": []}
    return {"population_difference": result[0]}

# Endpoint to get the name of the most populous city in the 12th most densely populated country
@app.get("/v1/mondial_geo/most_populous_city_in_12th_densest_country", operation_id="get_most_populous_city_in_12th_densest_country", summary="Retrieves the name of the city with the highest population in the 12th most densely populated country. The population density of a country is determined by dividing its total population by its area. The city with the highest population in the selected country is then identified.")
async def get_most_populous_city_in_12th_densest_country():
    cursor.execute("SELECT T2.Name FROM country AS T1 INNER JOIN city AS T2 ON T1.Code = T2.Country WHERE T1.Name = ( SELECT Name FROM country ORDER BY CAST(Population AS REAL) / Area LIMIT 11, 1 ) ORDER BY T2.Population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the ratio of the longest to the shortest border of a specific country
@app.get("/v1/mondial_geo/border_length_ratio", operation_id="get_border_length_ratio", summary="Retrieves the ratio of the longest to the shortest border length for a specified country. The operation calculates this ratio by comparing the maximum and minimum border lengths associated with the provided country name.")
async def get_border_length_ratio(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT MAX(T2.Length) / MIN(T2.Length) FROM country AS T1 INNER JOIN borders AS T2 ON T1.Code = T2.Country2 WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"border_length_ratio": []}
    return {"border_length_ratio": result[0]}

# Endpoint to get the capital of countries with more than a specified number of mountains
@app.get("/v1/mondial_geo/capitals_with_mountain_count", operation_id="get_capitals_with_mountain_count", summary="Retrieve the capitals of countries that have more than the specified minimum number of mountains. This operation returns a list of capitals from countries where the number of mountains exceeds the provided threshold. The input parameter determines the minimum mountain count required for a country to be included in the results.")
async def get_capitals_with_mountain_count(min_mountain_count: int = Query(..., description="Minimum number of mountains in the country")):
    cursor.execute("SELECT T1.Capital FROM country AS T1 INNER JOIN geo_mountain AS T2 ON T1.Code = T2.Country GROUP BY T1.Name, T1.Capital HAVING COUNT(T1.Name) > ?", (min_mountain_count,))
    result = cursor.fetchall()
    if not result:
        return {"capitals": []}
    return {"capitals": [row[0] for row in result]}

# Endpoint to get the count of mountains in the most populous country
@app.get("/v1/mondial_geo/mountain_count_most_populous_country", operation_id="get_mountain_count_most_populous_country", summary="Retrieves the total number of mountains located in the country with the highest population. The operation calculates this count by joining the country and geo_mountain tables, grouping the results by country name, and ordering the grouped data by population in descending order. The result is then limited to the top record, which corresponds to the most populous country.")
async def get_mountain_count_most_populous_country():
    cursor.execute("SELECT COUNT(T2.Mountain) FROM country AS T1 INNER JOIN geo_mountain AS T2 ON T1.Code = T2.Country GROUP BY T1.Name ORDER BY T1.Population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of countries with industry below a specified value and fewer than a specified number of mountains
@app.get("/v1/mondial_geo/count_countries_industry_mountains", operation_id="get_count_countries_industry_mountains", summary="Retrieves the count of countries where the industry level is less than the specified maximum value and the number of mountains is fewer than the provided maximum count.")
async def get_count_countries_industry_mountains(max_industry: int = Query(..., description="Maximum industry value"), max_mountain_count: int = Query(..., description="Maximum number of mountains in the country")):
    cursor.execute("SELECT COUNT(T3.Country) FROM ( SELECT T1.Country FROM economy AS T1 INNER JOIN geo_mountain AS T2 ON T1.Country = T2.Country WHERE T1.Industry < ? GROUP BY T1.Country HAVING COUNT(T1.Country) < ? ) AS T3", (max_industry, max_mountain_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the mountains in the country with the lowest inflation
@app.get("/v1/mondial_geo/mountains_lowest_inflation_country", operation_id="get_mountains_lowest_inflation_country", summary="Retrieves a list of mountains located in the country with the lowest inflation rate. The data is sourced from the geo_mountain table, filtered by the country with the lowest inflation rate, as determined by the economy table.")
async def get_mountains_lowest_inflation_country():
    cursor.execute("SELECT Mountain FROM geo_mountain WHERE Country = ( SELECT Country FROM economy ORDER BY Inflation ASC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"mountains": []}
    return {"mountains": [row[0] for row in result]}

# Endpoint to get the count of deserts in countries with a specific government type
@app.get("/v1/mondial_geo/count_deserts_government_type", operation_id="get_count_deserts_government_type", summary="Retrieves the total number of deserts found in countries that have a specified type of government. The government type is provided as an input parameter.")
async def get_count_deserts_government_type(government_type: str = Query(..., description="Type of government")):
    cursor.execute("SELECT COUNT(T3.Desert) FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country INNER JOIN geo_desert AS T3 ON T3.Country = T2.Country WHERE T2.Government = ?", (government_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the deserts in countries with area greater than a specified value and population less than a specified value
@app.get("/v1/mondial_geo/deserts_area_population", operation_id="get_deserts_area_population", summary="Retrieve a list of deserts located in countries with an area larger than the specified minimum and a population smaller than the specified maximum.")
async def get_deserts_area_population(min_area: int = Query(..., description="Minimum area of the country"), max_population: int = Query(..., description="Maximum population of the country")):
    cursor.execute("SELECT T2.Desert FROM country AS T1 INNER JOIN geo_desert AS T2 ON T1.Code = T2.Country WHERE T1.Area > ? AND T1.Population < ?", (min_area, max_population))
    result = cursor.fetchall()
    if not result:
        return {"deserts": []}
    return {"deserts": [row[0] for row in result]}

# Endpoint to get the count of deserts in countries with a specific language spoken by more than a specified percentage
@app.get("/v1/mondial_geo/count_deserts_language_percentage", operation_id="get_count_deserts_language_percentage", summary="Retrieve the number of deserts in countries where the specified language is spoken by more than the given percentage of the population. This operation provides a quantitative measure of the prevalence of a particular language in desert regions.")
async def get_count_deserts_language_percentage(language_name: str = Query(..., description="Name of the language"), min_percentage: float = Query(..., description="Minimum percentage of the language spoken")):
    cursor.execute("SELECT COUNT(T2.Desert) FROM country AS T1 INNER JOIN geo_desert AS T2 ON T1.Code = T2.Country INNER JOIN language AS T3 ON T1.Code = T2.Country WHERE T3.Name = ? AND T3.Percentage > ?", (language_name, min_percentage))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest mountain in an independent country
@app.get("/v1/mondial_geo/highest_mountain_independent_country", operation_id="get_highest_mountain_independent_country", summary="Retrieves the name of the highest mountain located in an independent country. The operation identifies the country, then determines the highest mountain within its borders by comparing the heights of all mountains in that country. The result is the name of the tallest mountain.")
async def get_highest_mountain_independent_country():
    cursor.execute("SELECT T4.Name FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country INNER JOIN geo_mountain AS T3 ON T3.Country = T2.Country INNER JOIN mountain AS T4 ON T4.Name = T3.Mountain WHERE T2.Independence IS NOT NULL ORDER BY T4.Height DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"mountain": []}
    return {"mountain": result[0]}

# Endpoint to get the count of distinct volcanic mountains in countries with a population less than or equal to a specified value
@app.get("/v1/mondial_geo/count_volcanic_mountains_population", operation_id="get_count_volcanic_mountains_population", summary="Retrieves the number of unique volcanic mountains located in countries with a population at or below a specified threshold. The operation considers the provided mountain type and population limit to filter the results.")
async def get_count_volcanic_mountains_population(mountain_type: str = Query(..., description="Type of the mountain"), max_population: int = Query(..., description="Maximum population of the country")):
    cursor.execute("SELECT COUNT(DISTINCT T3.Name) FROM country AS T1 INNER JOIN geo_mountain AS T2 ON T1.Code = T2.Country INNER JOIN mountain AS T3 ON T3.Name = T2.Mountain WHERE T3.Type = ? AND T1.Population <= ?", (mountain_type, max_population))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct countries with mountains higher than a specified height and GDP greater than a specified value
@app.get("/v1/mondial_geo/count_countries_mountain_height_gdp", operation_id="get_count_countries_mountain_height_gdp", summary="Retrieve the number of unique countries that have mountains exceeding a certain height and a GDP surpassing a specified value. This operation considers the country's GDP and the height of its mountains to determine the count.")
async def get_count_countries_mountain_height_gdp(min_gdp: int = Query(..., description="Minimum GDP of the country"), min_height: int = Query(..., description="Minimum height of the mountain")):
    cursor.execute("SELECT COUNT(DISTINCT T1.Name) FROM country AS T1 INNER JOIN geo_mountain AS T2 ON T1.Code = T2.Country INNER JOIN economy AS T3 ON T3.Country = T1.Code INNER JOIN mountain AS T4 ON T4.Name = T2.Mountain WHERE T3.GDP > ? AND T4.Height > ?", (min_gdp, min_height))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum border length of independent countries
@app.get("/v1/mondial_geo/max_border_length_independent_countries", operation_id="get_max_border_length", summary="Retrieves the maximum border length among all independent countries. This operation calculates the length of the longest border of any country that has achieved independence, excluding those that are not independent.")
async def get_max_border_length():
    cursor.execute("SELECT MAX(T3.Length) FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country INNER JOIN borders AS T3 ON T3.Country1 = T2.Country WHERE T2.Independence IS NOT NULL")
    result = cursor.fetchone()
    if not result:
        return {"max_length": []}
    return {"max_length": result[0]}

# Endpoint to get the count of distinct countries with a specific government type and border length greater than a specified value
@app.get("/v1/mondial_geo/count_distinct_countries_government_border_length", operation_id="get_count_distinct_countries", summary="Retrieve the number of unique countries that have a specified government type and a border length exceeding a given value. This operation considers the government type and border length data to provide an accurate count.")
async def get_count_distinct_countries(government_type: str = Query(..., description="Government type of the country"), border_length: int = Query(..., description="Minimum border length")):
    cursor.execute("SELECT COUNT(DISTINCT T1.Name) FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country INNER JOIN borders AS T3 ON T3.Country1 = T2.Country WHERE T2.Government = ? AND T3.Length > ?", (government_type, border_length))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country with the shortest border length
@app.get("/v1/mondial_geo/country_shortest_border_length", operation_id="get_country_shortest_border_length", summary="Retrieves the name of the country with the shortest border length. This operation calculates the border length by joining the country and borders tables, then orders the results by length in ascending order. The country with the shortest border is returned.")
async def get_country_shortest_border_length():
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN borders AS T2 ON T1.Code = T2.Country1 ORDER BY T2.Length ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the total GDP of countries in a specific continent
@app.get("/v1/mondial_geo/total_gdp_continent", operation_id="get_total_gdp_continent", summary="Retrieves the total GDP of all countries within a specified continent. The operation calculates the sum of GDP values from the economy table, based on the provided continent name.")
async def get_total_gdp_continent(continent_name: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT SUM(T4.GDP) FROM country AS T1 INNER JOIN encompasses AS T2 ON T1.Code = T2.Country INNER JOIN continent AS T3 ON T3.Name = T2.Continent INNER JOIN economy AS T4 ON T4.Country = T1.Code WHERE T3.Name = ?", (continent_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_gdp": []}
    return {"total_gdp": result[0]}

# Endpoint to get the count of mountains in a specific continent
@app.get("/v1/mondial_geo/count_mountains_continent", operation_id="get_count_mountains_continent", summary="Retrieves the total number of mountains located in a specified continent. The operation calculates this count by considering the countries and provinces within the continent.")
async def get_count_mountains_continent(continent_name: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT COUNT(T3.Name) FROM country AS T1 INNER JOIN encompasses AS T2 ON T1.Code = T2.Country INNER JOIN continent AS T3 ON T3.Name = T2.Continent INNER JOIN province AS T4 ON T4.Country = T1.Code INNER JOIN geo_mountain AS T5 ON T5.Province = T4.Name WHERE T3.Name = ?", (continent_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the largest desert in a specific continent
@app.get("/v1/mondial_geo/largest_desert_continent", operation_id="get_largest_desert_continent", summary="Retrieves the name of the largest desert located within a specified continent. The operation identifies the continent by its name and determines the desert with the largest area within that continent.")
async def get_largest_desert_continent(continent_name: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT T5.Name FROM country AS T1 INNER JOIN encompasses AS T2 ON T1.Code = T2.Country INNER JOIN continent AS T3 ON T3.Name = T2.Continent INNER JOIN geo_desert AS T4 ON T4.Country = T1.Code INNER JOIN desert AS T5 ON T5.Name = T4.Desert WHERE T3.Name = ? ORDER BY T5.Area DESC LIMIT 1", (continent_name,))
    result = cursor.fetchone()
    if not result:
        return {"desert": []}
    return {"desert": result[0]}

# Endpoint to get countries in a specific continent with population growth greater than a specified value
@app.get("/v1/mondial_geo/countries_population_growth_continent", operation_id="get_countries_population_growth_continent", summary="Retrieves a list of countries within a specified continent where the population growth rate exceeds a given threshold. The continent is identified by its name, and the population growth rate is expressed as a percentage.")
async def get_countries_population_growth_continent(continent_name: str = Query(..., description="Name of the continent"), population_growth: float = Query(..., description="Minimum population growth rate")):
    cursor.execute("SELECT T2.Country FROM country AS T1 INNER JOIN encompasses AS T2 ON T1.Code = T2.Country INNER JOIN continent AS T3 ON T3.Name = T2.Continent INNER JOIN population AS T4 ON T4.Country = T1.Code WHERE T3.Name = ? AND T4.Population_Growth > ?", (continent_name, population_growth))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the count of countries in a specific continent with infant mortality less than a specified value
@app.get("/v1/mondial_geo/count_countries_infant_mortality_continent", operation_id="get_count_countries_infant_mortality_continent", summary="Retrieve the number of countries in a specified continent where the infant mortality rate is below a given threshold.")
async def get_count_countries_infant_mortality_continent(continent_name: str = Query(..., description="Name of the continent"), infant_mortality: int = Query(..., description="Maximum infant mortality rate")):
    cursor.execute("SELECT COUNT(T1.Name) FROM country AS T1 INNER JOIN encompasses AS T2 ON T1.Code = T2.Country INNER JOIN continent AS T3 ON T3.Name = T2.Continent INNER JOIN population AS T4 ON T4.Country = T1.Code WHERE T3.Name = ? AND T4.Infant_Mortality < ?", (continent_name, infant_mortality))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct countries with a specific language and population growth less than a specified value
@app.get("/v1/mondial_geo/count_distinct_countries_language_population_growth", operation_id="get_count_distinct_countries_language_population_growth", summary="Retrieve the number of unique countries where the specified language is spoken and the population growth rate is below the provided threshold.")
async def get_count_distinct_countries_language_population_growth(language_name: str = Query(..., description="Name of the language"), population_growth: float = Query(..., description="Maximum population growth rate")):
    cursor.execute("SELECT COUNT(DISTINCT T1.Name) FROM country AS T1 INNER JOIN language AS T2 ON T1.Code = T2.Country INNER JOIN population AS T3 ON T3.Country = T2.Country WHERE T2.Name = ? AND T3.Population_Growth < ?", (language_name, population_growth))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average agriculture percentage of countries in a specific continent
@app.get("/v1/mondial_geo/average_agriculture_continent", operation_id="get_average_agriculture_continent", summary="Retrieves the average percentage of agriculture in countries located within a specified continent. The calculation is based on the economy data of each country in the continent.")
async def get_average_agriculture_continent(continent_name: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT AVG(T4.Agriculture) FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T3.Code = T2.Country INNER JOIN economy AS T4 ON T4.Country = T3.Code WHERE T1.Name = ?", (continent_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_agriculture": []}
    return {"average_agriculture": result[0]}

# Endpoint to get the count of distinct countries with a GDP greater than a specified value
@app.get("/v1/mondial_geo/count_distinct_countries_gdp", operation_id="get_count_distinct_countries_gdp", summary="Retrieve the number of unique countries with a Gross Domestic Product (GDP) surpassing a specified threshold. This operation filters countries based on their GDP and returns the count of distinct countries that meet the criteria. The input parameter allows you to set the minimum GDP value for the filter.")
async def get_count_distinct_countries_gdp(gdp: int = Query(..., description="GDP value to filter countries")):
    cursor.execute("SELECT COUNT(DISTINCT T1.Name) FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country INNER JOIN economy AS T3 ON T3.Country = T2.Country WHERE T2.Independence IS NOT NULL AND T3.GDP > ?", (gdp,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average inflation rate of the largest continent by area
@app.get("/v1/mondial_geo/avg_inflation_largest_continent", operation_id="get_avg_inflation_largest_continent", summary="Retrieves the average inflation rate of the continent with the largest area. This operation calculates the average inflation rate by aggregating data from the economy table, which is linked to the country table via the country code. The country table is further linked to the continent table through the encompasses table, which specifies the continent-country relationship. The calculation is performed for the continent with the largest area.")
async def get_avg_inflation_largest_continent():
    cursor.execute("SELECT AVG(T4.Inflation) FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T3.Code = T2.Country INNER JOIN economy AS T4 ON T4.Country = T3.Code WHERE T1.Name = ( SELECT Name FROM continent ORDER BY Area DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"average_inflation": []}
    return {"average_inflation": result[0]}

# Endpoint to get the name and area of the island where a specific city is located
@app.get("/v1/mondial_geo/island_info_by_city", operation_id="get_island_info_by_city", summary="Retrieves the name and area of the island where the specified city is located. The operation requires the name of the city as input and returns the corresponding island's name and area.")
async def get_island_info_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T3.Name, T3.Area FROM city AS T1 INNER JOIN locatedOn AS T2 ON T1.Name = T2.City INNER JOIN island AS T3 ON T3.Name = T2.Island WHERE T1.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"island_info": []}
    return {"island_info": {"name": result[0], "area": result[1]}}

# Endpoint to get the name and population of cities located on a specific island
@app.get("/v1/mondial_geo/cities_info_by_island", operation_id="get_cities_info_by_island", summary="Retrieves the names and populations of cities situated on a specified island. The operation requires the name of the island as an input parameter.")
async def get_cities_info_by_island(island_name: str = Query(..., description="Name of the island")):
    cursor.execute("SELECT T1.Name, T1.Population FROM city AS T1 INNER JOIN locatedOn AS T2 ON T1.Name = T2.City INNER JOIN island AS T3 ON T3.Name = T2.Island WHERE T3.Name = ?", (island_name,))
    result = cursor.fetchall()
    if not result:
        return {"cities_info": []}
    return {"cities_info": [{"name": row[0], "population": row[1]} for row in result]}

# Endpoint to get the distinct longitude and latitude of cities in a specific province
@app.get("/v1/mondial_geo/city_coordinates_by_province", operation_id="get_city_coordinates_by_province", summary="Retrieve the unique geographical coordinates of cities located within a specified province. The operation returns a list of distinct longitude and latitude pairs for cities in the given province.")
async def get_city_coordinates_by_province(province_name: str = Query(..., description="Name of the province")):
    cursor.execute("SELECT DISTINCT T3.Longitude, T3.Latitude FROM city AS T1 INNER JOIN locatedOn AS T2 ON T1.Name = T2.City INNER JOIN island AS T3 ON T3.Name = T2.Island WHERE T1.Province = ?", (province_name,))
    result = cursor.fetchall()
    if not result:
        return {"coordinates": []}
    return {"coordinates": [{"longitude": row[0], "latitude": row[1]} for row in result]}

# Endpoint to get the names of islands with an area greater than the area of the island in a specific province
@app.get("/v1/mondial_geo/islands_larger_than_province_island", operation_id="get_islands_larger_than_province_island", summary="Retrieves the names of islands that have a larger area than the area of the island in the specified province. The operation requires the name of the province as an input parameter.")
async def get_islands_larger_than_province_island(province_name: str = Query(..., description="Name of the province")):
    cursor.execute("SELECT DISTINCT Name FROM island WHERE Area > ( SELECT DISTINCT T3.Area FROM city AS T1 INNER JOIN locatedOn AS T2 ON T1.Name = T2.City INNER JOIN island AS T3 ON T3.Name = T2.Island WHERE T1.Province = ? )", (province_name,))
    result = cursor.fetchall()
    if not result:
        return {"island_names": []}
    return {"island_names": [row[0] for row in result]}

# Endpoint to get the names of islands and cities located on islands with an area less than a specified value
@app.get("/v1/mondial_geo/islands_cities_by_area", operation_id="get_islands_cities_by_area", summary="Retrieve the names of islands and their respective cities where the island area is less than the provided area value. This operation allows for filtering islands based on their size, enabling the identification of smaller islands and their associated cities.")
async def get_islands_cities_by_area(area: int = Query(..., description="Area value to filter islands")):
    cursor.execute("SELECT DISTINCT T3.Name, T1.Name FROM city AS T1 INNER JOIN locatedOn AS T2 ON T1.Name = T2.City INNER JOIN island AS T3 ON T3.Name = T2.Island WHERE T3.Area < ?", (area,))
    result = cursor.fetchall()
    if not result:
        return {"island_cities": []}
    return {"island_cities": [{"island_name": row[0], "city_name": row[1]} for row in result]}

# Endpoint to get the province and capital of a specific city
@app.get("/v1/mondial_geo/province_capital_by_city", operation_id="get_province_capital_by_city", summary="Retrieves the province and capital of a specified city. The operation uses the provided city name to look up the corresponding province and capital information in the database.")
async def get_province_capital_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T2.Province, T1.Capital FROM province AS T1 INNER JOIN city AS T2 ON T1.Name = T2.Province AND T1.Country = T2.Country WHERE T2.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"province_capital": []}
    return {"province_capital": {"province": result[0], "capital": result[1]}}

# Endpoint to get the name and population of cities in provinces with a population greater than a specified value
@app.get("/v1/mondial_geo/cities_by_province_population", operation_id="get_cities_by_province_population", summary="Retrieve the names and populations of cities located in provinces with a population exceeding the specified value. This operation allows you to filter cities based on the population of their respective provinces, providing a targeted list of urban centers and their corresponding populations.")
async def get_cities_by_province_population(province_population: int = Query(..., description="Population value to filter provinces")):
    cursor.execute("SELECT T1.Name, T1.Population FROM city AS T1 INNER JOIN province AS T2 ON T2.Name = T1.Province WHERE T2.Population > ?", (province_population,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [{"name": row[0], "population": row[1]} for row in result]}

# Endpoint to get the cities and provinces located on islands of a specific type
@app.get("/v1/mondial_geo/cities_provinces_by_island_type", operation_id="get_cities_provinces_by_island_type", summary="Retrieves the names of cities and provinces situated on islands of a specified type. The operation filters islands based on the provided type and returns the corresponding cities and provinces.")
async def get_cities_provinces_by_island_type(island_type: str = Query(..., description="Type of the island")):
    cursor.execute("SELECT City, Province FROM locatedOn WHERE Island IN ( SELECT Name FROM island WHERE Type = ? )", (island_type,))
    result = cursor.fetchall()
    if not result:
        return {"cities_provinces": []}
    return {"cities_provinces": [{"city": row[0], "province": row[1]} for row in result]}

# Endpoint to get the average population of cities located near a specific sea
@app.get("/v1/mondial_geo/average_population_cities_near_sea", operation_id="get_average_population_cities_near_sea", summary="Retrieves the average population of cities situated near a specified sea. The operation calculates the mean population of cities located in proximity to the sea identified by the provided sea name.")
async def get_average_population_cities_near_sea(sea_name: str = Query(..., description="Name of the sea")):
    cursor.execute("SELECT AVG(T1.Population) FROM city AS T1 INNER JOIN located AS T2 ON T1.Name = T2.City INNER JOIN sea AS T3 ON T3.Name = T2.Sea WHERE T3.Name = ?", (sea_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_population": []}
    return {"average_population": result[0]}

# Endpoint to get the percentage of a city's population relative to its province
@app.get("/v1/mondial_geo/city_population_percentage_of_province", operation_id="get_city_population_percentage_of_province", summary="Retrieves the percentage of a city's population relative to its province. The calculation is based on the population of the specified city and the total population of its corresponding province. The city is identified by its name.")
async def get_city_population_percentage_of_province(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT CAST(T1.Population AS REAL) * 100 / T2.Population FROM city AS T1 INNER JOIN province AS T2 ON T1.Province = T2.Name WHERE T1.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"population_percentage": []}
    return {"population_percentage": result[0]}

# Endpoint to get the names of rivers flowing into a specific sea
@app.get("/v1/mondial_geo/rivers_flowing_into_sea", operation_id="get_rivers_flowing_into_sea", summary="Retrieve the names of rivers that flow into a specified sea. The operation requires the name of the sea as an input parameter.")
async def get_rivers_flowing_into_sea(sea_name: str = Query(..., description="Name of the sea")):
    cursor.execute("SELECT Name FROM river WHERE Sea = ?", (sea_name,))
    result = cursor.fetchall()
    if not result:
        return {"rivers": []}
    return {"rivers": [row[0] for row in result]}

# Endpoint to get the lakes and cities in a specific province
@app.get("/v1/mondial_geo/lakes_and_cities_in_province", operation_id="get_lakes_and_cities_in_province", summary="Retrieves a list of lakes and cities located within a specified province. The operation requires the name of the province as an input parameter and returns a list of lakes and their respective cities.")
async def get_lakes_and_cities_in_province(province_name: str = Query(..., description="Name of the province")):
    cursor.execute("SELECT Lake, City FROM located WHERE Province = ? AND Lake IS NOT NULL", (province_name,))
    result = cursor.fetchall()
    if not result:
        return {"lakes_and_cities": []}
    return {"lakes_and_cities": [{"lake": row[0], "city": row[1]} for row in result]}

# Endpoint to get the highest mountain in a specific mountain range
@app.get("/v1/mondial_geo/highest_mountain_in_range", operation_id="get_highest_mountain_in_range", summary="Retrieves the highest mountain in a specified mountain range. The operation returns the name and height of the highest mountain, sorted by height in descending order. The input parameter determines the mountain range for which the highest mountain is sought.")
async def get_highest_mountain_in_range(mountain_range: str = Query(..., description="Name of the mountain range")):
    cursor.execute("SELECT Name, Height FROM mountain WHERE Mountains = ? ORDER BY Height DESC LIMIT 1", (mountain_range,))
    result = cursor.fetchone()
    if not result:
        return {"highest_mountain": []}
    return {"highest_mountain": {"name": result[0], "height": result[1]}}

# Endpoint to get the names, latitudes, and longitudes of mountains of a specific type
@app.get("/v1/mondial_geo/mountains_by_type", operation_id="get_mountains_by_type", summary="Retrieve the names, latitudes, and longitudes of mountains that match the specified type. The type of mountain is provided as an input parameter.")
async def get_mountains_by_type(mountain_type: str = Query(..., description="Type of the mountain")):
    cursor.execute("SELECT Name, Latitude, Longitude FROM mountain WHERE Type = ?", (mountain_type,))
    result = cursor.fetchall()
    if not result:
        return {"mountains": []}
    return {"mountains": [{"name": row[0], "latitude": row[1], "longitude": row[2]} for row in result]}

# Endpoint to get the names of mountains of a specific type within a height range
@app.get("/v1/mondial_geo/mountains_by_type_and_height_range", operation_id="get_mountains_by_type_and_height_range", summary="Retrieve the names of mountains that match a specified type and fall within a given height range. The operation filters mountains based on their type and height, returning only those that meet the provided criteria.")
async def get_mountains_by_type_and_height_range(mountain_type: str = Query(..., description="Type of the mountain"), min_height: int = Query(..., description="Minimum height of the mountain"), max_height: int = Query(..., description="Maximum height of the mountain")):
    cursor.execute("SELECT Name FROM mountain WHERE Type = ? AND Height BETWEEN ? AND ?", (mountain_type, min_height, max_height))
    result = cursor.fetchall()
    if not result:
        return {"mountains": []}
    return {"mountains": [row[0] for row in result]}

# Endpoint to get the longest river flowing into a specific sea
@app.get("/v1/mondial_geo/longest_river_into_sea", operation_id="get_longest_river_into_sea", summary="Retrieves the name of the longest river that flows into the specified sea. The operation sorts rivers by length in descending order and returns the name of the longest river that flows into the sea provided as input.")
async def get_longest_river_into_sea(sea_name: str = Query(..., description="Name of the sea")):
    cursor.execute("SELECT Name FROM river WHERE Sea = ? ORDER BY Length DESC LIMIT 1", (sea_name,))
    result = cursor.fetchone()
    if not result:
        return {"longest_river": []}
    return {"longest_river": result[0]}

# Endpoint to get the percentage of mountains in a range that are not of a specific type
@app.get("/v1/mondial_geo/percentage_non_specific_type_mountains", operation_id="get_percentage_non_specific_type_mountains", summary="Retrieves the percentage of mountains in a specified range that do not match a given type. The operation calculates the proportion of mountains in the range that are not of the provided type and returns this value as a percentage.")
async def get_percentage_non_specific_type_mountains(mountain_type: str = Query(..., description="Type of the mountain"), mountain_range: str = Query(..., description="Name of the mountain range")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN type != ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM mountain WHERE Mountains = ?", (mountain_type, mountain_range))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the names and provinces of cities located near rivers flowing into a specific sea
@app.get("/v1/mondial_geo/cities_near_rivers_flowing_into_sea", operation_id="get_cities_near_rivers_flowing_into_sea", summary="Retrieves the names and provinces of cities situated near rivers that flow into a specified sea. The operation requires the name of the sea as an input parameter.")
async def get_cities_near_rivers_flowing_into_sea(sea_name: str = Query(..., description="Name of the sea")):
    cursor.execute("SELECT T1.Name, T1.Province FROM city AS T1 INNER JOIN located AS T2 ON T1.Name = T2.City INNER JOIN river AS T3 ON T3.Name = T2.River WHERE T3.Sea = ?", (sea_name,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [{"name": row[0], "province": row[1]} for row in result]}

# Endpoint to get the name and length of a river associated with a specific city
@app.get("/v1/mondial_geo/river_info_by_city", operation_id="get_river_info_by_city", summary="Retrieves the name and length of the river that flows through a specified city. The operation requires the name of the city as input and returns the corresponding river's name and length.")
async def get_river_info_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T3.Name, T3.Length FROM city AS T1 INNER JOIN located AS T2 ON T1.Name = T2.City INNER JOIN river AS T3 ON T3.Name = T2.River WHERE T1.Name = ?", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"river_info": []}
    return {"river_info": result}

# Endpoint to get the height, latitude, and longitude of mountains associated with a specific river
@app.get("/v1/mondial_geo/mountain_info_by_river", operation_id="get_mountain_info_by_river", summary="Retrieves the height, latitude, and longitude of mountains located in provinces that are associated with the specified river. The input parameter is the name of the river.")
async def get_mountain_info_by_river(river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT T1.Height, T1.Latitude, T1.Longitude FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain INNER JOIN province AS T3 ON T3.Name = T2.Province INNER JOIN located AS T4 ON T4.Province = T3.Name WHERE T4.River = ?", (river_name,))
    result = cursor.fetchall()
    if not result:
        return {"mountain_info": []}
    return {"mountain_info": result}

# Endpoint to get the source longitude, latitude, and altitude of rivers in a specific province
@app.get("/v1/mondial_geo/river_source_info_by_province", operation_id="get_river_source_info_by_province", summary="Retrieve the longitude, latitude, and altitude of the sources of rivers located within a specified province. The operation requires the name of the province as an input parameter.")
async def get_river_source_info_by_province(province_name: str = Query(..., description="Name of the province")):
    cursor.execute("SELECT T1.SourceLongitude, T1.SourceLatitude, T1.SourceAltitude FROM river AS T1 INNER JOIN geo_river AS T2 ON T2.River = T1.Name WHERE T2.Province = ?", (province_name,))
    result = cursor.fetchall()
    if not result:
        return {"river_source_info": []}
    return {"river_source_info": result}

# Endpoint to get the name and height of mountains associated with a specific river
@app.get("/v1/mondial_geo/mountain_info_by_river_source", operation_id="get_mountain_info_by_river_source", summary="Retrieves the name and height of mountains that are the source of a specified river. The operation uses the river's name as input to identify the relevant mountains and their corresponding heights.")
async def get_mountain_info_by_river_source(river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT T1.Name, T1.Height FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain INNER JOIN province AS T3 ON T3.Name = T2.Province INNER JOIN geo_source AS T4 ON T4.Province = T3.Name WHERE T4.River = ?", (river_name,))
    result = cursor.fetchall()
    if not result:
        return {"mountain_info": []}
    return {"mountain_info": result}

# Endpoint to get the length of a river associated with a specific city
@app.get("/v1/mondial_geo/river_length_by_city", operation_id="get_river_length_by_city", summary="Retrieves the length of the river that flows through a specified city. The operation requires the name of the city as an input parameter and returns the length of the associated river.")
async def get_river_length_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T3.Length FROM city AS T1 INNER JOIN located AS T2 ON T1.Name = T2.City INNER JOIN river AS T3 ON T3.Name = T2.River WHERE T1.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"river_length": []}
    return {"river_length": result[0]}

# Endpoint to get the province and name of rivers longer than a specified length
@app.get("/v1/mondial_geo/rivers_by_length", operation_id="get_rivers_by_length", summary="Retrieve the names and provinces of rivers that exceed a specified minimum length. This operation allows you to filter rivers based on their length, providing a list of rivers that meet the specified length criteria along with their respective provinces.")
async def get_rivers_by_length(min_length: int = Query(..., description="Minimum length of the river")):
    cursor.execute("SELECT T1.Province, T2.Name FROM geo_river AS T1 INNER JOIN river AS T2 ON T1.River = T2.Name WHERE T2.Length > ?", (min_length,))
    result = cursor.fetchall()
    if not result:
        return {"rivers": []}
    return {"rivers": result}

# Endpoint to get the province, country, and height of a specific mountain
@app.get("/v1/mondial_geo/mountain_info_by_name", operation_id="get_mountain_info_by_name", summary="Retrieves the province, country, and height of a specific mountain. The operation requires the name of the mountain as an input parameter, which is used to filter the results and return the requested information.")
async def get_mountain_info_by_name(mountain_name: str = Query(..., description="Name of the mountain")):
    cursor.execute("SELECT T2.Province, T2.Country, T1.Height FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain WHERE T1.Name = ?", (mountain_name,))
    result = cursor.fetchone()
    if not result:
        return {"mountain_info": []}
    return {"mountain_info": result}

# Endpoint to get the distinct names and lengths of rivers in a specific country
@app.get("/v1/mondial_geo/rivers_by_country", operation_id="get_rivers_by_country", summary="Retrieve the unique names and lengths of rivers that flow through a specified country. The operation returns a list of rivers, each with its name and length, providing a comprehensive overview of the country's river system.")
async def get_rivers_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT DISTINCT T3.Name, T3.Length FROM city AS T1 INNER JOIN located AS T2 ON T1.Name = T2.City INNER JOIN river AS T3 ON T3.Name = T2.River WHERE T2.Country = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"rivers": []}
    return {"rivers": result}

# Endpoint to get the average height of mountains in a specific province
@app.get("/v1/mondial_geo/average_mountain_height_by_province", operation_id="get_average_mountain_height_by_province", summary="Retrieves the average height of mountains located within a specified province. The operation calculates the mean height of mountains in the given province, providing a single numerical value as the result.")
async def get_average_mountain_height_by_province(province_name: str = Query(..., description="Name of the province")):
    cursor.execute("SELECT AVG(T1.Height) FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain WHERE T2.Province = ?", (province_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the population range of cities associated with a specific river
@app.get("/v1/mondial_geo/city_population_range_by_river", operation_id="get_city_population_range_by_river", summary="Retrieves the population range of cities located along a specified river. The population range is calculated as the difference between the maximum and minimum populations of these cities. The input parameter is the name of the river.")
async def get_city_population_range_by_river(river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT MAX(T1.Population) - MIN(T1.Population) FROM city AS T1 INNER JOIN located AS T2 ON T1.Name = T2.City INNER JOIN river AS T3 ON T3.Name = T2.River WHERE T3.Name = ?", (river_name,))
    result = cursor.fetchone()
    if not result:
        return {"population_range": []}
    return {"population_range": result[0]}

# Endpoint to get river details based on city name
@app.get("/v1/mondial_geo/river_details_by_city", operation_id="get_river_details_by_city", summary="Retrieve the name and length of rivers that flow through a specified city. The operation requires the name of the city as input and returns the corresponding river details.")
async def get_river_details_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.Name, T1.Length FROM river AS T1 INNER JOIN located AS T2 ON T1.Name = T2.River INNER JOIN city AS T3 ON T3.Name = T2.City WHERE T3.Name = ?", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"rivers": []}
    return {"rivers": result}

# Endpoint to get countries where a specific language is spoken by a specific percentage
@app.get("/v1/mondial_geo/countries_by_language_percentage", operation_id="get_countries_by_language_percentage", summary="Retrieves a list of countries where the specified language is spoken by the given percentage of the population. The language is identified by its name, and the percentage is expressed as a decimal value.")
async def get_countries_by_language_percentage(language_name: str = Query(..., description="Name of the language"), percentage: int = Query(..., description="Percentage of the language spoken")):
    cursor.execute("SELECT Country FROM language WHERE Name = ? AND Percentage = ?", (language_name, percentage))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get countries with a specific type of government
@app.get("/v1/mondial_geo/countries_by_government_type", operation_id="get_countries_by_government_type", summary="Retrieve a list of countries that have a specified type of government. The operation filters the countries based on the provided government type.")
async def get_countries_by_government_type(government_type: str = Query(..., description="Type of government")):
    cursor.execute("SELECT Country FROM politics WHERE Government = ?", (government_type,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get distinct rivers in a specific country
@app.get("/v1/mondial_geo/distinct_rivers_by_country", operation_id="get_distinct_rivers_by_country", summary="Retrieve a unique list of rivers flowing within a specified country. The operation filters the rivers based on the provided country name and excludes any null or empty river names.")
async def get_distinct_rivers_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT DISTINCT T1.River FROM located AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T2.Name = ? AND T1.River IS NOT NULL", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"rivers": []}
    return {"rivers": result}

# Endpoint to get the country with the highest inflation rate
@app.get("/v1/mondial_geo/country_with_highest_inflation", operation_id="get_country_with_highest_inflation", summary="Retrieves the name of the country with the highest inflation rate from the economy and country tables. The inflation rate is determined by comparing the rates across all countries and selecting the highest one.")
async def get_country_with_highest_inflation():
    cursor.execute("SELECT T2.Name FROM economy AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code ORDER BY T1.Inflation DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the top provinces by population in a specific country
@app.get("/v1/mondial_geo/top_provinces_by_population", operation_id="get_top_provinces_by_population", summary="Get the top provinces by population in a specific country")
async def get_top_provinces_by_population(country_name: str = Query(..., description="Name of the country"), limit: int = Query(..., description="Number of top provinces to return"), offset: int = Query(..., description="Offset for pagination")):
    cursor.execute("SELECT T1.Province, T1.Population FROM city AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T2.Name = ? ORDER BY T1.Population DESC LIMIT ? OFFSET ?", (country_name, limit, offset))
    result = cursor.fetchall()
    if not result:
        return {"provinces": []}
    return {"provinces": result}

# Endpoint to get the population of a specific ethnic group in a specific country
@app.get("/v1/mondial_geo/ethnic_group_population", operation_id="get_ethnic_group_population", summary="Retrieves the population of a specific ethnic group within a given country. The calculation is based on the percentage of the ethnic group in the country and the total population of the country.")
async def get_ethnic_group_population(country_name: str = Query(..., description="Name of the country"), ethnic_group: str = Query(..., description="Name of the ethnic group")):
    cursor.execute("SELECT T2.Percentage * T1.Population FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T1.Name = ? AND T2.Name = ?", (country_name, ethnic_group))
    result = cursor.fetchone()
    if not result:
        return {"population": []}
    return {"population": result[0]}

# Endpoint to get the average area of countries in a specific continent
@app.get("/v1/mondial_geo/average_area_by_continent", operation_id="get_average_area_by_continent", summary="Retrieves the average area of countries located within a specified continent. The continent is identified by its name, which is provided as an input parameter.")
async def get_average_area_by_continent(continent: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT AVG(Area) FROM country AS T1 INNER JOIN encompasses AS T2 ON T1.Code = T2.Country WHERE T2.Continent = ?", (continent,))
    result = cursor.fetchone()
    if not result:
        return {"average_area": []}
    return {"average_area": result[0]}

# Endpoint to get the country, latitude, and longitude of the smallest desert
@app.get("/v1/mondial_geo/smallest_desert_details", operation_id="get_smallest_desert_details", summary="Get the country, latitude, and longitude of the smallest desert")
async def get_smallest_desert_details():
    cursor.execute("SELECT T2.Country, T1.Latitude, T1.Longitude FROM desert AS T1 INNER JOIN geo_desert AS T2 ON T1.Name = T2.Desert WHERE T1.Name = ( SELECT Name FROM desert ORDER BY Area ASC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"desert_details": []}
    return {"desert_details": result}

# Endpoint to get the population of speakers of a specific language in a specific country
@app.get("/v1/mondial_geo/language_speakers_population", operation_id="get_language_speakers_population", summary="Retrieves the estimated number of speakers of a specified language in a given country. This is calculated by multiplying the percentage of speakers of the language in the country by the country's total population.")
async def get_language_speakers_population(language_name: str = Query(..., description="Name of the language"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Percentage * T2.Population FROM language AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T1.Name = ? AND T2.Name = ?", (language_name, country_name))
    result = cursor.fetchone()
    if not result:
        return {"population": []}
    return {"population": result[0]}

# Endpoint to get the count of mountains in the country with the largest area
@app.get("/v1/mondial_geo/count_mountains_largest_country", operation_id="get_count_mountains_largest_country", summary="Retrieves the total number of mountains located in the country with the largest area. The operation does not require any input parameters and returns a single integer value.")
async def get_count_mountains_largest_country():
    cursor.execute("SELECT COUNT(Mountain) FROM geo_mountain WHERE Country = ( SELECT Code FROM country ORDER BY Area DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of countries located by the shallowest sea
@app.get("/v1/mondial_geo/countries_shallowest_sea", operation_id="get_countries_shallowest_sea", summary="Retrieves a list of unique country names that are located by the shallowest sea, as determined by the minimum depth of all seas.")
async def get_countries_shallowest_sea():
    cursor.execute("SELECT DISTINCT T2.Name FROM located AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE Sea = ( SELECT Name FROM sea ORDER BY Depth ASC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the country with the lowest GDP for a given government type
@app.get("/v1/mondial_geo/country_lowest_gdp_government", operation_id="get_country_lowest_gdp_government", summary="Retrieves the country with the lowest GDP associated with the specified government type. The operation filters countries based on the provided government type and sorts them by GDP in ascending order, returning the country with the lowest GDP.")
async def get_country_lowest_gdp_government(government_type: str = Query(..., description="Type of government (e.g., 'Communist state')")):
    cursor.execute("SELECT T2.Country FROM politics AS T1 INNER JOIN economy AS T2 ON T1.Country = T2.Country WHERE T1.Government = ? ORDER BY T2.GDP ASC LIMIT 1", (government_type,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the government type of the country with the highest inflation
@app.get("/v1/mondial_geo/government_highest_inflation", operation_id="get_government_highest_inflation", summary="Retrieves the government type of the country with the highest inflation rate. This operation compares inflation rates across all countries and identifies the country with the highest rate. The government type of this country is then returned.")
async def get_government_highest_inflation():
    cursor.execute("SELECT T1.Government FROM politics AS T1 INNER JOIN economy AS T2 ON T1.Country = T2.Country ORDER BY T2.Inflation DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"government": []}
    return {"government": result[0]}

# Endpoint to get the country with the highest infant mortality for a given year of independence
@app.get("/v1/mondial_geo/country_highest_infant_mortality_independence_year", operation_id="get_country_highest_infant_mortality_independence_year", summary="Retrieves the country with the highest infant mortality rate in the year of its independence. The operation accepts the year of independence as a parameter, formatted as 'YYYY', and returns the name of the country with the highest infant mortality rate in that year.")
async def get_country_highest_infant_mortality_independence_year(independence_year: str = Query(..., description="Year of independence in 'YYYY' format")):
    cursor.execute("SELECT T1.Country FROM politics AS T1 INNER JOIN population AS T2 ON T1.Country = T2.Country WHERE STRFTIME('%Y', T1.Independence) = ? ORDER BY T2.Infant_Mortality DESC LIMIT 1", (independence_year,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the government types of countries with the shortest border
@app.get("/v1/mondial_geo/government_types_shortest_border", operation_id="get_government_types_shortest_border", summary="Retrieves the government types of the two countries sharing the shortest border. The operation identifies the countries with the shortest border length and returns their respective government types.")
async def get_government_types_shortest_border():
    cursor.execute("SELECT T1.Government, T3.Government FROM politics AS T1 INNER JOIN borders AS T2 ON T1.Country = T2.Country1 INNER JOIN politics AS T3 ON T3.Country = T2.Country2 ORDER BY T2.Length ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"government_types": []}
    return {"government_types": [result[0], result[1]]}

# Endpoint to get the country with the smallest population for a given language and percentage
@app.get("/v1/mondial_geo/country_smallest_population_language_percentage", operation_id="get_country_smallest_population_language_percentage", summary="Retrieves the country with the smallest population where the specified language is spoken by the given percentage of the population. The operation uses the provided language name and percentage to filter the results and order them by population in ascending order. The country with the smallest population is then returned.")
async def get_country_smallest_population_language_percentage(language: str = Query(..., description="Name of the language"), percentage: int = Query(..., description="Percentage of the language spoken")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN language AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? AND T2.Percentage = ? ORDER BY T1.Population ASC LIMIT 1", (language, percentage))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the provinces in the largest desert
@app.get("/v1/mondial_geo/provinces_largest_desert", operation_id="get_provinces_largest_desert", summary="Retrieves a list of provinces located within the largest desert, as determined by area. This operation provides valuable geographical insights into the provinces within the most expansive desert.")
async def get_provinces_largest_desert():
    cursor.execute("SELECT Province FROM geo_desert WHERE Desert = ( SELECT Name FROM desert ORDER BY Area DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"provinces": []}
    return {"provinces": [row[0] for row in result]}

# Endpoint to get the count of lakes in the least populous republic in Africa
@app.get("/v1/mondial_geo/count_lakes_least_populous_republic_africa", operation_id="get_count_lakes_least_populous_republic_africa", summary="Retrieves the total number of lakes in the least populous republic within a specified continent, given a certain percentage of the country's encompassed area and a specific type of government.")
async def get_count_lakes_least_populous_republic_africa(continent: str = Query(..., description="Continent name"), percentage: int = Query(..., description="Percentage of the country encompassed"), government: str = Query(..., description="Type of government (e.g., 'republic')")):
    cursor.execute("SELECT COUNT(*) FROM geo_lake WHERE Country = ( SELECT T4.Code FROM ( SELECT T2.Code, T2.Population FROM encompasses AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code INNER JOIN politics AS T3 ON T1.Country = T3.Country WHERE T1.Continent = ? AND T1.Percentage = ? AND T3.Government = ? ORDER BY Population DESC LIMIT 4 ) AS T4 ORDER BY population ASC LIMIT 1 )", (continent, percentage, government))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most prevalent religion in a given continent
@app.get("/v1/mondial_geo/most_prevalent_religion_continent", operation_id="get_most_prevalent_religion_continent", summary="Retrieves the most prevalent religion in the specified continent. This operation identifies the religion with the highest percentage of adherents across all countries within the given continent. The continent is determined by the provided input parameter.")
async def get_most_prevalent_religion_continent(continent: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT T4.Name FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T3.Code = T2.Country INNER JOIN religion AS T4 ON T4.Country = T3.Code WHERE T1.Name = ? GROUP BY T4.Name ORDER BY SUM(T4.Percentage) DESC LIMIT 1", (continent,))
    result = cursor.fetchone()
    if not result:
        return {"religion": []}
    return {"religion": result[0]}

# Endpoint to get details of the highest mountain
@app.get("/v1/mondial_geo/highest_mountain_details", operation_id="get_highest_mountain_details", summary="Retrieves comprehensive details of the highest mountain in the world, including its geographical location and the country it belongs to. The response includes information about the mountain's height, name, and location, as well as the associated province and country details.")
async def get_highest_mountain_details():
    cursor.execute("SELECT * FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain INNER JOIN province AS T3 ON T3.Country = T2.Country INNER JOIN country AS T4 ON T4.Code = T3.Country WHERE T1.Name = ( SELECT Name FROM mountain ORDER BY Height DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the name of the sea surrounding the largest island
@app.get("/v1/mondial_geo/sea_surrounding_largest_island", operation_id="get_sea_surrounding_largest_island", summary="Retrieves the name of the sea that surrounds the largest island in the database. The operation identifies the largest island by its area and then determines the sea that encompasses it.")
async def get_sea_surrounding_largest_island():
    cursor.execute("SELECT T2.Name FROM islandIn AS T1 INNER JOIN sea AS T2 ON T2.Name = T1.Sea WHERE T1.Island = ( SELECT Name FROM island ORDER BY Area DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"sea_name": []}
    return {"sea_name": result}

# Endpoint to get distinct countries with cities located on the longest river flowing into a specific sea
@app.get("/v1/mondial_geo/countries_with_cities_on_longest_river", operation_id="get_countries_with_cities_on_longest_river", summary="Retrieves a list of unique countries that have cities situated along the longest river flowing into a specified sea. The input parameter determines the sea of interest.")
async def get_countries_with_cities_on_longest_river(sea: str = Query(..., description="Name of the sea")):
    cursor.execute("SELECT DISTINCT T1.Country FROM city AS T1 INNER JOIN located AS T2 ON T1.Name = T2.City INNER JOIN river AS T3 ON T3.Name = T2.River WHERE T3.Name = ( SELECT Name FROM river WHERE Sea = ? ORDER BY Length DESC LIMIT 1 )", (sea,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the population of the capital city of a country that gained independence on a specific date
@app.get("/v1/mondial_geo/capital_population_by_independence_date", operation_id="get_capital_population_by_independence_date", summary="Retrieve the population of the capital city of a country that achieved independence on a specified date. The input parameter is the independence date in 'YYYY-MM-DD' format. This operation returns the population of the capital city of the country that gained independence on the provided date.")
async def get_capital_population_by_independence_date(independence_date: str = Query(..., description="Independence date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T3.Population FROM politics AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code INNER JOIN city AS T3 ON T3.Name = T2.Capital WHERE T1.Independence = ?", (independence_date,))
    result = cursor.fetchall()
    if not result:
        return {"population": []}
    return {"population": result}

# Endpoint to get cities and capitals located on a specific river
@app.get("/v1/mondial_geo/cities_and_capitals_on_river", operation_id="get_cities_and_capitals_on_river", summary="Retrieves a list of cities and capitals that are situated on a specified river. The operation requires the name of the river as an input parameter to filter the results.")
async def get_cities_and_capitals_on_river(river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT T2.City, T1.Capital FROM country AS T1 INNER JOIN located AS T2 ON T1.Code = T2.Country INNER JOIN river AS T3 ON T3.Name = T2.River WHERE T3.Name = ?", (river_name,))
    result = cursor.fetchall()
    if not result:
        return {"cities_and_capitals": []}
    return {"cities_and_capitals": result}

# Endpoint to get the population percentage of a specific language in countries dependent on a specific country
@app.get("/v1/mondial_geo/language_population_percentage", operation_id="get_language_population_percentage", summary="Retrieves the population percentage of a specific language in countries that are dependent on a given country. The calculation is based on the population of the dependent countries and the percentage of speakers of the specified language within those countries.")
async def get_language_population_percentage(dependent_country: str = Query(..., description="Name of the dependent country"), language: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT T2.Percentage * T1.Population FROM country AS T1 INNER JOIN language AS T2 ON T1.Code = T2.Country INNER JOIN politics AS T3 ON T3.Country = T2.Country WHERE T3.Dependent = ? AND T2.Name = ?", (dependent_country, language))
    result = cursor.fetchall()
    if not result:
        return {"population_percentage": []}
    return {"population_percentage": result}

# Endpoint to get the population density and service of the country with the most provinces in a specific continent
@app.get("/v1/mondial_geo/population_density_and_service", operation_id="get_population_density_and_service", summary="Retrieves the population density and service information for the country with the highest number of provinces in a specified continent. The population density is calculated as the total population divided by the total area of all provinces in the country. The service information is obtained from the economy table. The country with the most provinces is determined by counting the number of province names associated with each country in the encompasses table.")
async def get_population_density_and_service(continent: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT T1.Country, T2.Service , SUM(T1.Population) / SUM(T1.Area) FROM province AS T1 INNER JOIN economy AS T2 ON T1.Country = T2.Country WHERE T1.Country IN ( SELECT Country FROM encompasses WHERE Continent = ? ) GROUP BY T1.Country, T2.Service ORDER BY COUNT(T1.Name) DESC LIMIT 1", (continent,))
    result = cursor.fetchall()
    if not result:
        return {"population_density_and_service": []}
    return {"population_density_and_service": result}

# Endpoint to get the population percentage of the capital city of the second most populous country in a specific continent
@app.get("/v1/mondial_geo/capital_population_percentage", operation_id="get_capital_population_percentage", summary="Retrieves the percentage of the total population that resides in the capital city of the second most populous country in a given continent. The continent is specified as an input parameter.")
async def get_capital_population_percentage(continent: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT T4.Capital, CAST(T3.Population AS REAL) * 100 / T4.Population FROM city AS T3 INNER JOIN ( SELECT T1.Capital , T1.Population FROM country AS T1 INNER JOIN encompasses AS T2 ON T1.Code = T2.Country WHERE T2.Continent = ? ORDER BY T1.Population DESC LIMIT 2, 1 ) AS T4 ON T3.Name = T4.Capital", (continent,))
    result = cursor.fetchall()
    if not result:
        return {"capital_population_percentage": []}
    return {"capital_population_percentage": result}

# Endpoint to get the name of the second largest desert by area
@app.get("/v1/mondial_geo/second_largest_desert", operation_id="get_second_largest_desert", summary="Retrieves the name of the second largest desert in the world, sorted by area in descending order.")
async def get_second_largest_desert():
    cursor.execute("SELECT Name FROM desert ORDER BY Area DESC LIMIT 1, 1")
    result = cursor.fetchall()
    if not result:
        return {"desert_name": []}
    return {"desert_name": result}

# Endpoint to get the most spoken language in a given country
@app.get("/v1/mondial_geo/most_spoken_language", operation_id="get_most_spoken_language", summary="Retrieves the name of the most widely spoken language in a specified country. The language is determined by the highest percentage of speakers in the country. The country is identified using its code.")
async def get_most_spoken_language(country: str = Query(..., description="Country code (e.g., 'MNE')")):
    cursor.execute("SELECT Name FROM language WHERE Country = ? ORDER BY Percentage DESC LIMIT 1", (country,))
    result = cursor.fetchone()
    if not result:
        return {"language": []}
    return {"language": result[0]}

# Endpoint to get the percentage of a specific language spoken in a given country
@app.get("/v1/mondial_geo/language_percentage", operation_id="get_language_percentage", summary="Retrieves the percentage of a specified language spoken in a given country. The operation requires the country and language names as input parameters to filter the data and return the corresponding percentage.")
async def get_language_percentage(country_name: str = Query(..., description="Name of the country"), language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT T1.Percentage FROM language AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T2.Name = ? AND T1.Name = ?", (country_name, language_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the country with the lowest GDP among those with a population greater than a specified number
@app.get("/v1/mondial_geo/country_lowest_gdp", operation_id="get_country_lowest_gdp", summary="Retrieve the name of the country with the lowest GDP among those with a population greater than the specified minimum population.")
async def get_country_lowest_gdp(min_population: int = Query(..., description="Minimum population of the country")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T1.Population > ? ORDER BY T2.GDP ASC LIMIT 1", (min_population,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the capital of the country where a specific mountain is located
@app.get("/v1/mondial_geo/mountain_capital", operation_id="get_mountain_capital", summary="Retrieves the capital city of the country where the specified mountain is located. The operation uses the mountain's name as input to search for the corresponding country and its capital.")
async def get_mountain_capital(mountain_name: str = Query(..., description="Name of the mountain")):
    cursor.execute("SELECT T4.Capital FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain INNER JOIN province AS T3 ON T3.Name = T2.Province INNER JOIN country AS T4 ON T4.Province = T3.Name WHERE T1.Name = ?", (mountain_name,))
    result = cursor.fetchone()
    if not result:
        return {"capital": []}
    return {"capital": result[0]}

# Endpoint to get the count of seas around the island where a specific mountain is located
@app.get("/v1/mondial_geo/seas_around_mountain_island", operation_id="get_seas_around_mountain_island", summary="Retrieves the number of seas surrounding the island that hosts a specific mountain. The operation requires the name of the mountain as an input parameter.")
async def get_seas_around_mountain_island(mountain_name: str = Query(..., description="Name of the mountain")):
    cursor.execute("SELECT COUNT(T4.Sea) FROM mountain AS T1 INNER JOIN mountainOnIsland AS T2 ON T1.Name = T2.Mountain INNER JOIN island AS T3 ON T3.Name = T2.Island INNER JOIN islandIn AS T4 ON T4.Island = T3.Name WHERE T1.Name = ?", (mountain_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the distinct names of countries with cities located on a specific river
@app.get("/v1/mondial_geo/countries_with_cities_on_river", operation_id="get_countries_with_cities_on_river", summary="Retrieve a unique list of country names that have cities situated on a specified river. The operation filters the data based on the provided river name.")
async def get_countries_with_cities_on_river(river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT DISTINCT T4.Name FROM city AS T1 INNER JOIN located AS T2 ON T1.Name = T2.City INNER JOIN river AS T3 ON T3.Name = T2.River INNER JOIN country AS T4 ON T4.Code = T2.Country WHERE T3.Name = ?", (river_name,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the names of countries that gained independence on a specific date
@app.get("/v1/mondial_geo/countries_by_independence_date", operation_id="get_countries_by_independence_date", summary="Retrieve the names of countries that achieved independence on a specified date. The input parameter is the independence date in 'YYYY-MM-DD' format.")
async def get_countries_by_independence_date(independence_date: str = Query(..., description="Independence date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country WHERE T2.Independence = ?", (independence_date,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the count of cities in a specific country with a population greater than a specified number
@app.get("/v1/mondial_geo/city_count_by_population", operation_id="get_city_count_by_population", summary="Retrieve the number of cities in a given country that have a population exceeding a specified threshold. The operation requires the name of the country and the minimum population as input parameters.")
async def get_city_count_by_population(country_name: str = Query(..., description="Name of the country"), min_population: int = Query(..., description="Minimum population of the city")):
    cursor.execute("SELECT COUNT(T2.Name) FROM country AS T1 INNER JOIN city AS T2 ON T2.Country = T1.Code WHERE T1.Name = ? AND T2.Population > ?", (country_name, min_population))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the longest river flowing into a sea with a specific depth
@app.get("/v1/mondial_geo/longest_river_by_sea_depth", operation_id="get_longest_river_by_sea_depth", summary="Retrieves the name of the longest river that flows into a sea with the specified depth. The operation sorts rivers by length in descending order and returns the top result.")
async def get_longest_river_by_sea_depth(sea_depth: int = Query(..., description="Depth of the sea")):
    cursor.execute("SELECT T2.Name FROM sea AS T1 INNER JOIN river AS T2 ON T2.Sea = T1.Name WHERE T1.Depth = ? ORDER BY T2.Length DESC LIMIT 1", (sea_depth,))
    result = cursor.fetchone()
    if not result:
        return {"river": []}
    return {"river": result[0]}

# Endpoint to get the country of the highest mountain
@app.get("/v1/mondial_geo/highest_mountain_country", operation_id="get_highest_mountain_country", summary="Retrieve the country with the highest mountain, based on a specified limit and offset. The limit parameter determines the maximum number of results to return, while the offset parameter specifies the starting point within the result set.")
async def get_highest_mountain_country(limit: int = Query(..., description="Limit the number of results"), offset: int = Query(..., description="Offset the results")):
    cursor.execute("SELECT T3.Country FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain INNER JOIN province AS T3 ON T3.Name = T2.Province ORDER BY T1.Height DESC LIMIT ?, ?", (limit, offset))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the longitude of a mountain on an island
@app.get("/v1/mondial_geo/mountain_longitude_on_island", operation_id="get_mountain_longitude_on_island", summary="Retrieves the longitude of a specific mountain located on an island. The mountain is identified by its name. The operation returns the longitude value of the mountain's location on the island.")
async def get_mountain_longitude_on_island(mountain_name: str = Query(..., description="Name of the mountain")):
    cursor.execute("SELECT T3.Longitude FROM mountain AS T1 INNER JOIN mountainOnIsland AS T2 ON T1.Name = T2.Mountain INNER JOIN island AS T3 ON T3.Name = T2.Island WHERE T1.Name = ?", (mountain_name,))
    result = cursor.fetchone()
    if not result:
        return {"longitude": []}
    return {"longitude": result[0]}

# Endpoint to get the name of the country with the highest GDP and area less than a specified value
@app.get("/v1/mondial_geo/country_highest_gdp_area_less_than", operation_id="get_country_highest_gdp_area_less_than", summary="Retrieve the name of the country with the highest Gross Domestic Product (GDP) among those with an area less than the specified value. The operation allows you to limit the number of results returned.")
async def get_country_highest_gdp_area_less_than(area: int = Query(..., description="Area of the country"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T1.Area < ? ORDER BY T2.GDP DESC LIMIT ?", (area, limit))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of cities in a country
@app.get("/v1/mondial_geo/count_cities_in_country", operation_id="get_count_cities_in_country", summary="Retrieves the total number of cities in a specified country. The operation calculates this count by considering the cities that belong to the provinces of the given country. The country is identified by its name.")
async def get_count_cities_in_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT COUNT(T3.Name) FROM country AS T1 INNER JOIN province AS T2 ON T1.Code = T2.Country INNER JOIN city AS T3 ON T3.Province = T2.Name WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most populous city in a country excluding the capital
@app.get("/v1/mondial_geo/most_populous_city_excluding_capital", operation_id="get_most_populous_city_excluding_capital", summary="Retrieves the most populous city in a specified country, excluding the capital. The operation accepts the name of the country and a limit for the number of results. The response is ordered by population in descending order.")
async def get_most_populous_city_excluding_capital(country_name: str = Query(..., description="Name of the country"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.Name FROM country AS T1 INNER JOIN province AS T2 ON T1.Code = T2.Country INNER JOIN city AS T3 ON T3.Province = T2.Name WHERE T1.Name = ? AND T3.Name <> T1.Capital ORDER BY T3.Population DESC LIMIT ?", (country_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the most populous city excluding the capital
@app.get("/v1/mondial_geo/most_populous_city_excluding_capital_global", operation_id="get_most_populous_city_excluding_capital_global", summary="Retrieves the name of the most populous city in the world, excluding the capital, based on the provided limit. The operation sorts cities by population in descending order and returns the top results up to the specified limit.")
async def get_most_populous_city_excluding_capital_global(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.Name FROM country AS T1 INNER JOIN province AS T2 ON T1.Code = T2.Country INNER JOIN city AS T3 ON T3.Province = T2.Name WHERE T3.Name <> T1.Capital ORDER BY T3.Population DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the country of a specific city
@app.get("/v1/mondial_geo/country_of_city", operation_id="get_country_of_city", summary="Retrieves the name of the country where the specified city is located. The operation uses the provided city name to search across the database and returns the name of the corresponding country.")
async def get_country_of_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN province AS T2 ON T1.Code = T2.Country INNER JOIN city AS T3 ON T3.Province = T2.Name WHERE T3.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the most prevalent religion in a country
@app.get("/v1/mondial_geo/most_prevalent_religion", operation_id="get_most_prevalent_religion", summary="Retrieve the most common religion(s) in the specified country, with the option to limit the number of results returned.")
async def get_most_prevalent_religion(country_name: str = Query(..., description="Name of the country"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.Name FROM country AS T1 INNER JOIN religion AS T2 ON T1.Code = T2.Country WHERE T1.Name = ? ORDER BY T2.Percentage DESC LIMIT ?", (country_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"religion": []}
    return {"religion": result[0]}

# Endpoint to get countries with a specific border length
@app.get("/v1/mondial_geo/countries_with_border_length", operation_id="get_countries_with_border_length", summary="Retrieves a list of countries that share a border of a specified length with another country. The response includes the names of both countries involved in the border relationship.")
async def get_countries_with_border_length(border_length: int = Query(..., description="Length of the border")):
    cursor.execute("SELECT T1.Name, T3.Name FROM country AS T1 INNER JOIN borders AS T2 ON T1.Code = T2.Country1 INNER JOIN country AS T3 ON T3.Code = T2.Country2 WHERE T2.Length = ?", (border_length,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [{"country1": row[0], "country2": row[1]} for row in result]}

# Endpoint to get the percentage of a country in a continent
@app.get("/v1/mondial_geo/country_percentage_in_continent", operation_id="get_country_percentage_in_continent", summary="Retrieves the percentage of a specified country's area within a specified continent. The operation requires the country and continent names as input parameters to calculate and return the corresponding percentage.")
async def get_country_percentage_in_continent(country_name: str = Query(..., description="Name of the country"), continent_name: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT T2.Percentage FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T3.Code = T2.Country WHERE T3.Name = ? AND T1.Name = ?", (country_name, continent_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get country names that appear more than a specified number of times in a continent
@app.get("/v1/mondial_geo/country_names_by_count", operation_id="get_country_names_by_count", summary="Retrieve the names of countries that appear more than a specified number of times within a continent. The input parameter determines the minimum occurrence count for a country name to be included in the results.")
async def get_country_names_by_count(count: int = Query(..., description="Minimum number of times a country name appears in a continent")):
    cursor.execute("SELECT T3.Name FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T3.Code = T2.Country GROUP BY T3.Name HAVING COUNT(T3.Name) > ?", (count,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the population of a country where a specific city is located
@app.get("/v1/mondial_geo/country_population_by_city", operation_id="get_country_population_by_city", summary="Retrieves the total population of a country that contains a specified city. The operation uses the provided city name to identify the country and return its population.")
async def get_country_population_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.Population FROM country AS T1 INNER JOIN province AS T2 ON T1.Code = T2.Country INNER JOIN city AS T3 ON T3.Province = T2.Name WHERE T3.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"population": []}
    return {"population": result[0]}

# Endpoint to get the product of infant mortality, population, and population growth for a specific country
@app.get("/v1/mondial_geo/country_infant_mortality_product", operation_id="get_country_infant_mortality_product", summary="Retrieves the product of infant mortality rate, population, and population growth rate for a specified country. This operation provides a composite metric that reflects the country's demographic and health conditions.")
async def get_country_infant_mortality_product(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Infant_Mortality * T1.Population * T2.Population_Growth FROM country AS T1 INNER JOIN population AS T2 ON T1.Code = T2.Country WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": result[0]}

# Endpoint to get the count of mountains in a specific country
@app.get("/v1/mondial_geo/mountain_count_by_country", operation_id="get_mountain_count_by_country", summary="Retrieves the total number of mountains located within a specified country. The operation calculates this count by aggregating data from the mountain, geo_mountain, province, and country tables, based on the provided country name.")
async def get_mountain_count_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT COUNT(T1.Name) FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain INNER JOIN province AS T3 ON T3.Name = T2.Province INNER JOIN country AS T4 ON T4.Province = T3.Name WHERE T4.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the GDP per capita of a specific country
@app.get("/v1/mondial_geo/country_gdp_per_capita", operation_id="get_country_gdp_per_capita", summary="Retrieves the GDP per capita for a specified country. This operation calculates the GDP per capita by dividing the country's GDP by its population. The country is identified using the provided country name.")
async def get_country_gdp_per_capita(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.GDP / T1.Population FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"gdp_per_capita": []}
    return {"gdp_per_capita": result[0]}

# Endpoint to get the product of service and GDP for a specific city
@app.get("/v1/mondial_geo/city_service_gdp_product", operation_id="get_city_service_gdp_product", summary="Retrieves the product of the service and GDP for a specified city. The city's name is required as an input parameter to identify the correct city and its associated service and GDP data.")
async def get_city_service_gdp_product(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T4.Service * T4.GDP FROM country AS T1 INNER JOIN province AS T2 ON T1.Code = T2.Country INNER JOIN city AS T3 ON T3.Province = T2.Name INNER JOIN economy AS T4 ON T4.Country = T2.Country WHERE T3.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": result[0]}

# Endpoint to get the ratio of the longest to the shortest river in a specific country
@app.get("/v1/mondial_geo/river_length_ratio_by_country", operation_id="get_river_length_ratio_by_country", summary="Retrieve the ratio of the longest to the shortest river length in a given country. The operation calculates this ratio based on the provided country code, which identifies the country of interest.")
async def get_river_length_ratio_by_country(country_code: str = Query(..., description="Country code (e.g., 'TJ' for Tajikistan)")):
    cursor.execute("SELECT MAX(T2.Length) / MIN(T2.Length) FROM located AS T1 INNER JOIN river AS T2 ON T1.River = T2.Name WHERE T1.Country = ?", (country_code,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the population density of a country where a specific city is located
@app.get("/v1/mondial_geo/country_population_density_by_city", operation_id="get_country_population_density_by_city", summary="Retrieve the population density of a country that contains a specified city. The calculation is based on the total population and area of the country. The input parameter is the name of the city.")
async def get_country_population_density_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.Population / T1.Area FROM country AS T1 INNER JOIN province AS T2 ON T1.Code = T2.Country INNER JOIN city AS T3 ON T3.Province = T2.Name WHERE T3.Name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"population_density": []}
    return {"population_density": result[0]}

# Endpoint to get the ethnic group with the highest percentage below a specified value in each country
@app.get("/v1/mondial_geo/ethnic_group_by_max_percentage", operation_id="get_ethnic_group_by_max_percentage", summary="Retrieves the ethnic group with the highest percentage below a specified maximum value in each country. The input parameter sets the maximum percentage value for the search.")
async def get_ethnic_group_by_max_percentage(max_percentage: float = Query(..., description="Maximum percentage value")):
    cursor.execute("SELECT Country, Name FROM ethnicGroup AS T1 WHERE Percentage < ? AND Percentage = ( SELECT MAX(Percentage) FROM ethnicGroup AS T2 WHERE T1.Country = T2.Country )", (max_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"ethnic_groups": []}
    return {"ethnic_groups": [{"country": row[0], "name": row[1]} for row in result]}

# Endpoint to get deserts with more than a specified number of distinct countries
@app.get("/v1/mondial_geo/deserts_with_distinct_countries", operation_id="get_deserts_with_distinct_countries", summary="Retrieves a list of deserts that span across more than a specified number of distinct countries. The operation filters deserts based on the provided minimum number of distinct countries they cover.")
async def get_deserts_with_distinct_countries(min_countries: int = Query(..., description="Minimum number of distinct countries")):
    cursor.execute("SELECT Desert FROM geo_desert GROUP BY Desert HAVING COUNT(DISTINCT Country) > ?", (min_countries,))
    result = cursor.fetchall()
    if not result:
        return {"deserts": []}
    return {"deserts": [row[0] for row in result]}

# Endpoint to get rivers with more than a specified number of distinct countries and their provinces
@app.get("/v1/mondial_geo/rivers_with_distinct_countries", operation_id="get_rivers_with_distinct_countries", summary="Retrieve a list of rivers that flow through more than a specified number of distinct countries, along with their respective provinces. The response includes the river name and a concatenated string of provinces it passes through.")
async def get_rivers_with_distinct_countries(min_countries: int = Query(..., description="Minimum number of distinct countries")):
    cursor.execute("SELECT River, GROUP_CONCAT(Province) FROM geo_river GROUP BY River HAVING COUNT(DISTINCT Country) > ?", (min_countries,))
    result = cursor.fetchall()
    if not result:
        return {"rivers": []}
    return {"rivers": [{"river": row[0], "provinces": row[1]} for row in result]}

# Endpoint to get the percentage of border length with a specific country
@app.get("/v1/mondial_geo/border_length_percentage", operation_id="get_border_length_percentage", summary="Retrieves the percentage of total border length that a specified country shares with its neighboring countries. The calculation is based on the sum of border lengths involving the specified country, divided by the total border length of that country.")
async def get_border_length_percentage(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Name = ? THEN T1.Length ELSE 0 END) * 100 / SUM(T1.Length) FROM borders AS T1 LEFT JOIN country AS T2 ON T1.Country1 = T2.Code", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of islands with a specific area and type
@app.get("/v1/mondial_geo/island_percentage", operation_id="get_island_percentage", summary="Retrieves the percentage of islands within a specified island group that have an area equal to or less than a given maximum area and do not match a specified island type. If the island type is not provided, all islands are considered.")
async def get_island_percentage(max_area: int = Query(..., description="Maximum area of the island"), islands: str = Query(..., description="Name of the island group"), island_type: str = Query(..., description="Type of the island")):
    cursor.execute("SELECT SUM(CASE WHEN Area <= ? THEN 1 ELSE 0 END) * 100 / COUNT(*) FROM island WHERE Islands = ? AND (Type != ? OR Type IS NULL)", (max_area, islands, island_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get capitals of countries with population less than a specified percentage of the maximum population
@app.get("/v1/mondial_geo/capitals_by_population_percentage", operation_id="get_capitals_by_population_percentage", summary="Get capitals of countries with population less than a specified percentage of the maximum population")
async def get_capitals_by_population_percentage(population_percentage: float = Query(..., description="Percentage of the maximum population")):
    cursor.execute("SELECT Capital FROM country WHERE Population <= ( SELECT MAX(Population) - MAX(Population) * ? FROM country )", (population_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"capitals": []}
    return {"capitals": [row[0] for row in result]}

# Endpoint to get details of a river by name
@app.get("/v1/mondial_geo/river_details", operation_id="get_river_details", summary="Get details of a river by name")
async def get_river_details(river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT * FROM river WHERE Name = ?", (river_name,))
    result = cursor.fetchall()
    if not result:
        return {"rivers": []}
    return {"rivers": [dict(row) for row in result]}

# Endpoint to get the percentage of countries not targeting a specific group
@app.get("/v1/mondial_geo/non_target_percentage", operation_id="get_non_target_percentage", summary="Retrieves the percentage of countries that do not target a specific group. The calculation is based on the total number of countries and the count of countries targeting the specified group. The target group is provided as an input parameter.")
async def get_non_target_percentage(target_group: str = Query(..., description="Target group")):
    cursor.execute("SELECT 100 - (CAST(SUM(CASE WHEN Target = ? THEN 1 ELSE 0 END) AS REAL)) * 100 / COUNT(Country) FROM target", (target_group,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get countries that are members of a specific organization with cities in a specified population range and established within a specific date range
@app.get("/v1/mondial_geo/member_countries", operation_id="get_member_countries", summary="Retrieves a list of countries that are members of a specified organization and have cities within a given population range, established within a certain date range. The organization is identified by its abbreviation, and the population range is defined by minimum and maximum population values. The establishment date range is specified using start and end dates in 'YYYY-MM-DD' format.")
async def get_member_countries(organization_abbreviation: str = Query(..., description="Abbreviation of the organization"), min_population: int = Query(..., description="Minimum population of the city"), max_population: int = Query(..., description="Maximum population of the city"), start_date: str = Query(..., description="Start date of establishment in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of establishment in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.Country FROM country AS T1 INNER JOIN isMember AS T2 ON T1.Code = T2.Country INNER JOIN organization AS T3 ON T3.Country = T2.Country INNER JOIN city AS T4 ON T4.Country = T3.Country WHERE T3.Abbreviation = ? AND T4.Population BETWEEN ? AND ? AND T3.Established BETWEEN ? AND ?", (organization_abbreviation, min_population, max_population, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get rivers in a specific country with a specified length and name
@app.get("/v1/mondial_geo/rivers_in_country", operation_id="get_rivers_in_country", summary="Retrieves a list of rivers in a specified country that meet the provided length and name criteria. The operation filters rivers based on the given country name, minimum length, and river name, returning only those that match all conditions.")
async def get_rivers_in_country(country_name: str = Query(..., description="Name of the country"), min_length: int = Query(..., description="Minimum length of the river"), river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT T2.River FROM country AS T1 INNER JOIN geo_river AS T2 ON T1.Code = T2.Country WHERE T1.Name = ? AND T2.River IN ( SELECT NAME FROM river WHERE Length > ? AND River = ? )", (country_name, min_length, river_name))
    result = cursor.fetchall()
    if not result:
        return {"rivers": []}
    return {"rivers": [row[0] for row in result]}

# Endpoint to get cities based on sea name pattern and depth difference
@app.get("/v1/mondial_geo/cities_by_sea_name_pattern_and_depth_difference", operation_id="get_cities_by_sea_name_pattern_and_depth_difference", summary="Retrieves a list of cities that are located by a sea with a name matching the provided pattern and a depth difference that matches the provided value. The depth difference is calculated as the difference between the depth of the sea with the matching name and the depth of the sea where the city is located.")
async def get_cities_by_sea_name_pattern_and_depth_difference(sea_name_pattern: str = Query(..., description="Pattern to match sea name (use % for wildcard)"), depth_difference: int = Query(..., description="Depth difference")):
    cursor.execute("SELECT T2.City FROM sea AS T1 INNER JOIN located AS T2 ON T1.Name = T2.Sea INNER JOIN city AS T3 ON T3.Name = T2.City WHERE ( SELECT Depth FROM sea WHERE Name LIKE ? ) - T1.Depth = ?", (sea_name_pattern, depth_difference))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get cities based on lake longitude and latitude
@app.get("/v1/mondial_geo/cities_by_lake_longitude_and_latitude", operation_id="get_cities_by_lake_longitude_and_latitude", summary="Retrieves a list of cities located on a lake, based on the provided longitude and latitude of the lake. The operation uses the longitude and latitude to identify the lake, and then returns the names of the cities situated on that lake.")
async def get_cities_by_lake_longitude_and_latitude(longitude: float = Query(..., description="Longitude of the lake"), latitude: float = Query(..., description="Latitude of the lake")):
    cursor.execute("SELECT T2.City FROM lake AS T1 INNER JOIN located AS T2 ON T1.Name = T2.Lake INNER JOIN province AS T3 ON T3.Name = T2.Province INNER JOIN city AS T4 ON T4.Province = T3.Name WHERE T1.Longitude = ? AND T1.Latitude = ?", (longitude, latitude))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the continent with the highest inflation country
@app.get("/v1/mondial_geo/continent_with_highest_inflation_country", operation_id="get_continent_with_highest_inflation_country", summary="Retrieves the name of the continent that contains the country with the highest inflation rate. The operation calculates the inflation rate for each country and identifies the continent with the highest rate.")
async def get_continent_with_highest_inflation_country():
    cursor.execute("SELECT T1.Name FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T3.Code = T2.Country INNER JOIN economy AS T4 ON T4.Country = T3.Code ORDER BY T4.Inflation DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"continent": []}
    return {"continent": result[0]}

# Endpoint to get bordering countries based on continent and border length
@app.get("/v1/mondial_geo/bordering_countries_by_continent_and_border_length", operation_id="get_bordering_countries_by_continent_and_border_length", summary="Retrieve a list of country pairs that share a border of a specified length within a given continent. This operation filters countries based on the provided continent name and border length, returning the names of the countries that meet these criteria.")
async def get_bordering_countries_by_continent_and_border_length(continent_name: str = Query(..., description="Name of the continent"), border_length: int = Query(..., description="Length of the border")):
    cursor.execute("SELECT T4.Country1, T4.Country2 FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T3.Code = T2.Country INNER JOIN borders AS T4 ON T4.Country1 = T3.Code WHERE T1.Name = ? AND T4.Length = ?", (continent_name, border_length))
    result = cursor.fetchall()
    if not result:
        return {"bordering_countries": []}
    return {"bordering_countries": [{"country1": row[0], "country2": row[1]} for row in result]}

# Endpoint to get the deepest lake in a specified country
@app.get("/v1/mondial_geo/deepest_lake_in_country", operation_id="get_deepest_lake_in_country", summary="Retrieves the name of the deepest lake in a specified country. The operation uses the provided country name to search for the corresponding country in the database. It then identifies the deepest lake within that country by comparing the depths of all lakes located in the country's provinces. The name of the deepest lake is returned as the result.")
async def get_deepest_lake_in_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Name FROM lake AS T1 INNER JOIN geo_lake AS T2 ON T1.Name = T2.Lake INNER JOIN province AS T3 ON T3.Name = T2.Province INNER JOIN country AS T4 ON T4.Code = T3.Country WHERE T4.Name = ? ORDER BY T1.Depth DESC LIMIT 1", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"lake": []}
    return {"lake": result[0]}

# Endpoint to get lakes based on a river name
@app.get("/v1/mondial_geo/lakes_by_river_name", operation_id="get_lakes_by_river_name", summary="Retrieves a list of lakes that are connected to a specified river. The operation filters lakes based on the provided river name and returns their names.")
async def get_lakes_by_river_name(river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT NAME FROM lake WHERE river = ( SELECT river FROM river WHERE NAME = ? )", (river_name,))
    result = cursor.fetchall()
    if not result:
        return {"lakes": []}
    return {"lakes": [row[0] for row in result]}

# Endpoint to get islands based on a mountain name
@app.get("/v1/mondial_geo/islands_by_mountain_name", operation_id="get_islands_by_mountain_name", summary="Retrieves a list of islands that have a mountain with the specified name. The operation uses the provided mountain name to search for matching records in the mountain and mountainOnIsland tables, and then returns the corresponding island names from the island table.")
async def get_islands_by_mountain_name(mountain_name: str = Query(..., description="Name of the mountain")):
    cursor.execute("SELECT T1.Islands FROM island AS T1 INNER JOIN mountainOnIsland AS T2 ON T1.Name = T2.Island INNER JOIN mountain AS T3 ON T3.Name = T2.Mountain WHERE T3.Name = ?", (mountain_name,))
    result = cursor.fetchall()
    if not result:
        return {"islands": []}
    return {"islands": [row[0] for row in result]}

# Endpoint to get the sea that merges with the deepest sea
@app.get("/v1/mondial_geo/sea_merging_with_deepest_sea", operation_id="get_sea_merging_with_deepest_sea", summary="Get the sea that merges with the deepest sea")
async def get_sea_merging_with_deepest_sea():
    cursor.execute("SELECT T2.Sea2 FROM sea AS T1 INNER JOIN mergesWith AS T2 ON T1.Name = T2.Sea1 WHERE T1.Name = ( SELECT Name FROM sea ORDER BY Depth DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"sea": []}
    return {"sea": result[0]}

# Endpoint to get countries based on continent count and population density
@app.get("/v1/mondial_geo/countries_by_continent_count_and_population_density", operation_id="get_countries_by_continent_count_and_population_density", summary="Retrieves the names of countries that are part of more than a specified number of continents and have a population density not exceeding a given threshold. The continent count and maximum population density are provided as input parameters.")
async def get_countries_by_continent_count_and_population_density(continent_count: int = Query(..., description="Number of continents the country is part of"), max_population_density: float = Query(..., description="Maximum population density (population per area)")):
    cursor.execute("SELECT NAME FROM country WHERE CODE IN ( SELECT country FROM encompasses GROUP BY country HAVING COUNT(continent) > ? ) AND population / Area <= ?", (continent_count, max_population_density))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the country with the lowest population density for a specific religion
@app.get("/v1/mondial_geo/country_with_lowest_population_density_for_religion", operation_id="get_country_with_lowest_population_density_for_religion", summary="Retrieves the country with the lowest population density where the specified religion is practiced. The population density is calculated as the ratio of the country's population to its area. The input parameter is the name of the religion.")
async def get_country_with_lowest_population_density_for_religion(religion_name: str = Query(..., description="Name of the religion")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN religion AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? ORDER BY T1.Population / T1.Area ASC LIMIT 1", (religion_name,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the independence status of a country based on its GDP
@app.get("/v1/mondial_geo/independence_by_gdp", operation_id="get_independence_by_gdp", summary="Retrieves the independence status of a country with a specified GDP. The operation uses the provided GDP to identify the country and then fetches its independence status from the politics table.")
async def get_independence_by_gdp(gdp: int = Query(..., description="GDP of the country")):
    cursor.execute("SELECT Independence FROM politics WHERE country = ( SELECT country FROM economy WHERE GDP = ? )", (gdp,))
    result = cursor.fetchone()
    if not result:
        return {"independence": []}
    return {"independence": result[0]}

# Endpoint to get the population density of countries whose organizations were established in a specific year
@app.get("/v1/mondial_geo/population_density_by_established_year", operation_id="get_population_density_by_established_year", summary="Retrieve the population density of countries where organizations were established in a specific year. The population density is calculated by dividing the total population by the total area of the country. The year parameter is used to filter the organizations based on their establishment year.")
async def get_population_density_by_established_year(year: str = Query(..., description="Year the organization was established in 'YYYY' format")):
    cursor.execute("SELECT T1.Population / T1.Area FROM country AS T1 INNER JOIN organization AS T2 ON T1.Code = T2.Country WHERE STRFTIME('%Y', T2.Established) = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"population_density": []}
    return {"population_density": [row[0] for row in result]}

# Endpoint to get the capital and province of a country by its name
@app.get("/v1/mondial_geo/capital_province_by_country_name", operation_id="get_capital_province_by_country_name", summary="Retrieves the capital city and province of a specified country. The operation requires the country's name as input and returns the corresponding capital and province information.")
async def get_capital_province_by_country_name(name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT Capital, Province FROM country WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"capital": [], "province": []}
    return {"capital": result[0], "province": result[1]}

# Endpoint to get the country with the smallest population
@app.get("/v1/mondial_geo/smallest_population_country", operation_id="get_smallest_population_country", summary="Retrieves the name and capital of the country with the smallest population from the database.")
async def get_smallest_population_country():
    cursor.execute("SELECT Name, Capital FROM country ORDER BY Population ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": [], "capital": []}
    return {"name": result[0], "capital": result[1]}

# Endpoint to get the difference between the largest and smallest area of two continents
@app.get("/v1/mondial_geo/area_difference_between_continents", operation_id="get_area_difference_between_continents", summary="Retrieves the difference in area between the largest and smallest countries of two specified continents. The operation compares the maximum and minimum areas of the countries within the provided continents to determine the difference.")
async def get_area_difference_between_continents(name1: str = Query(..., description="Name of the first continent"), name2: str = Query(..., description="Name of the second continent")):
    cursor.execute("SELECT MAX(Area) - MIN(Area) FROM continent WHERE Name = ? OR Name = ?", (name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"area_difference": []}
    return {"area_difference": result[0]}

# Endpoint to get the longitude and latitude of a city by its name
@app.get("/v1/mondial_geo/city_coordinates_by_name", operation_id="get_city_coordinates_by_name", summary="Retrieves the geographical coordinates (longitude and latitude) of a specified city. The city is identified by its name, which is provided as an input parameter.")
async def get_city_coordinates_by_name(name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT Longitude, Latitude FROM city WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"longitude": [], "latitude": []}
    return {"longitude": result[0], "latitude": result[1]}

# Endpoint to get the difference between the largest and smallest population of two countries
@app.get("/v1/mondial_geo/population_difference_between_countries", operation_id="get_population_difference_between_countries", summary="Retrieves the population difference between the two countries specified in the input parameters. The operation calculates the difference between the maximum and minimum population of the two countries, providing a comparison of their population sizes.")
async def get_population_difference_between_countries(name1: str = Query(..., description="Name of the first country"), name2: str = Query(..., description="Name of the second country")):
    cursor.execute("SELECT MAX(Population) - MIN(Population) FROM country WHERE Name = ? OR Name = ?", (name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"population_difference": []}
    return {"population_difference": result[0]}

# Endpoint to get the city and province of an organization by its name
@app.get("/v1/mondial_geo/organization_location_by_name", operation_id="get_organization_location_by_name", summary="Retrieves the city and province associated with a specific organization, identified by its name.")
async def get_organization_location_by_name(name: str = Query(..., description="Name of the organization")):
    cursor.execute("SELECT City, Province FROM organization WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"city": [], "province": []}
    return {"city": result[0], "province": result[1]}

# Endpoint to get the name of the lake with the largest volume
@app.get("/v1/mondial_geo/largest_volume_lake", operation_id="get_largest_volume_lake", summary="Retrieves the name of the lake with the largest volume, calculated as the product of its area and depth.")
async def get_largest_volume_lake():
    cursor.execute("SELECT Name FROM lake ORDER BY Area * Depth DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the countries with the longest border
@app.get("/v1/mondial_geo/longest_border_countries", operation_id="get_longest_border_countries", summary="Retrieves the pair of countries that share the longest border, as determined by the length of their common boundary. The response includes the names of both countries.")
async def get_longest_border_countries():
    cursor.execute("SELECT Country1, Country2 FROM borders ORDER BY Length DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country1": [], "country2": []}
    return {"country1": result[0], "country2": result[1]}

# Endpoint to get the name of a country with a specific ethnic group, ordered by percentage ascending
@app.get("/v1/mondial_geo/country_by_ethnic_group", operation_id="get_country_by_ethnic_group", summary="Retrieves the name of a country with the highest percentage of a specific ethnic group. The country is determined by finding the lowest percentage of the specified ethnic group among all countries, as this indicates the highest concentration of that group within a single country.")
async def get_country_by_ethnic_group(ethnic_group: str = Query(..., description="Name of the ethnic group")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN ethnicGroup AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? ORDER BY T2.Percentage ASC LIMIT 1", (ethnic_group,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get countries with specific religions and language spoken 100%
@app.get("/v1/mondial_geo/countries_by_religion_and_language", operation_id="get_countries_by_religion_and_language", summary="Retrieves a list of countries where the specified three religions are practiced and the given language is spoken by 100% of the population. The operation filters countries based on the provided religion and language parameters, ensuring that all three religions are present and the language is universally spoken.")
async def get_countries_by_religion_and_language(religion1: str = Query(..., description="First religion name"), religion2: str = Query(..., description="Second religion name"), religion3: str = Query(..., description="Third religion name"), language: str = Query(..., description="Language name"), percentage: int = Query(..., description="Percentage of language spoken")):
    cursor.execute("SELECT T2.Country FROM country AS T1 INNER JOIN religion AS T2 ON T1.Code = T2.Country INNER JOIN language AS T3 ON T3.Country = T2.Country WHERE (T2.Name = ? OR T2.Name = ? OR T2.Name = ?) AND T3.Name = ? AND T3.Percentage = ? GROUP BY T1.Name HAVING COUNT(T1.Name) = 3", (religion1, religion2, religion3, language, percentage))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the top 3 countries with the highest inflation
@app.get("/v1/mondial_geo/top_countries_by_inflation", operation_id="get_top_countries_by_inflation", summary="Retrieves the names of the top three countries with the highest inflation rates. The data is sourced from the country and economy tables, which are joined based on the country code. The results are ordered in descending order by inflation rate.")
async def get_top_countries_by_inflation():
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country ORDER BY T2.Inflation DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get countries where a specific language is spoken 100%
@app.get("/v1/mondial_geo/countries_by_language", operation_id="get_countries_by_language", summary="Retrieves a list of countries where the specified language is the only language spoken. The language and the percentage of its usage are provided as input parameters.")
async def get_countries_by_language(language: str = Query(..., description="Language name"), percentage: int = Query(..., description="Percentage of language spoken")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN language AS T2 ON T1.Code = T2.Country WHERE T2.Name = ? AND T2.Percentage = ?", (language, percentage))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the count of organizations established after a specific year in countries without independence
@app.get("/v1/mondial_geo/count_organizations_established_after_year", operation_id="get_count_organizations_established_after_year", summary="Retrieves the total number of organizations established in countries without independence after a specified year. The year is provided as a four-digit input parameter.")
async def get_count_organizations_established_after_year(year: int = Query(..., description="Year established (YYYY)")):
    cursor.execute("SELECT COUNT(T3.Name) FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country INNER JOIN organization AS T3 ON T3.Country = T2.Country WHERE T2.Independence = NULL AND STRFTIME('%Y', T3.Established) > ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the province of a city located on a specific river
@app.get("/v1/mondial_geo/province_by_river", operation_id="get_province_by_river", summary="Retrieves the province of a city that is located on the specified river. The operation uses the river's name as input to identify the relevant city and its corresponding province.")
async def get_province_by_river(river_name: str = Query(..., description="Name of the river")):
    cursor.execute("SELECT T1.Province FROM city AS T1 INNER JOIN located AS T2 ON T1.Name = T2.City INNER JOIN river AS T3 ON T3.Name = T2.River WHERE T3.Name = ?", (river_name,))
    result = cursor.fetchall()
    if not result:
        return {"provinces": []}
    return {"provinces": [row[0] for row in result]}

# Endpoint to get the count of distinct provinces and sea depth for a specific country and sea
@app.get("/v1/mondial_geo/province_count_and_sea_depth", operation_id="get_province_count_and_sea_depth", summary="Retrieves the number of unique provinces and the corresponding sea depth for a given country and sea. The operation filters the data based on the provided country code and sea name.")
async def get_province_count_and_sea_depth(country_code: str = Query(..., description="Country code"), sea_name: str = Query(..., description="Name of the sea")):
    cursor.execute("SELECT COUNT(DISTINCT T2.province), T3.Depth FROM country AS T1 INNER JOIN located AS T2 ON T1.Code = T2.Country INNER JOIN sea AS T3 ON T3.Name = T2.Sea WHERE T1.Code = ? AND T3.Name = ? GROUP BY T3.Depth", (country_code, sea_name))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": [{"province_count": row[0], "sea_depth": row[1]} for row in result]}

# Endpoint to get the names of countries with a specific government type
@app.get("/v1/mondial_geo/countries_by_government", operation_id="get_countries_by_government", summary="Retrieves the names of countries that have a specified government type. The government type is provided as an input parameter.")
async def get_countries_by_government(government_type: str = Query(..., description="Type of government")):
    cursor.execute("SELECT name FROM country WHERE CODE IN ( SELECT country FROM politics WHERE government = ? )", (government_type,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the country with the highest GDP and lowest agriculture
@app.get("/v1/mondial_geo/top_country_by_gdp_and_agriculture", operation_id="get_top_country_by_gdp_and_agriculture", summary="Retrieves the name of the country with the highest GDP and lowest agricultural output. The data is sourced from the country and economy tables, which are joined based on the country code. The results are ordered by GDP in descending order and agricultural output in ascending order, with the top result being returned.")
async def get_top_country_by_gdp_and_agriculture():
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country ORDER BY T2.GDP DESC, T2.Agriculture ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the area and count of countries in a specific continent
@app.get("/v1/mondial_geo/continent_area_and_country_count", operation_id="get_continent_area_and_country_count", summary="Retrieves the total area and the number of countries in a specified continent. The continent is identified by its name, which is provided as an input parameter.")
async def get_continent_area_and_country_count(continent_name: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT T1.Area, COUNT(T3.Name) FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T3.Code = T2.Country WHERE T1.Name = ? GROUP BY T1.Name, T1.Area", (continent_name,))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"area": result[0], "country_count": result[1]}

# Endpoint to get the province with the highest number of organizations in a given country
@app.get("/v1/mondial_geo/province_with_most_organizations", operation_id="get_province_with_most_organizations", summary="Retrieves the province with the highest number of organizations in a specified country. The operation groups the data by province and orders it in descending order based on the count of organizations. The result is limited to the top province.")
async def get_province_with_most_organizations(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Province FROM country AS T1 INNER JOIN organization AS T2 ON T1.Code = T2.Country WHERE T1.Name = ? GROUP BY T1.Province ORDER BY COUNT(T1.Name) DESC LIMIT 1", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"province": []}
    return {"province": result[0]}

# Endpoint to get the top N countries by independence date
@app.get("/v1/mondial_geo/top_countries_by_independence", operation_id="get_top_countries_by_independence", summary="Retrieves a list of the top N countries, ranked by their independence date in descending order. The 'limit' parameter determines the number of countries to include in the list.")
async def get_top_countries_by_independence(limit: int = Query(..., description="Number of top countries to return")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country ORDER BY T2.Independence DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the top N countries with a specific government type and independence year
@app.get("/v1/mondial_geo/top_countries_by_government_and_independence", operation_id="get_top_countries_by_government_and_independence", summary="Retrieve the top N countries that gained independence in a specific year or later, with a specified type of government. The results are ordered by the independence year, with the most recent first. The 'limit' parameter determines the number of countries to return.")
async def get_top_countries_by_government_and_independence(government_type: str = Query(..., description="Type of government"), independence_year: str = Query(..., description="Independence year in 'YYYY' format"), limit: int = Query(..., description="Number of top countries to return")):
    cursor.execute("SELECT country FROM politics WHERE government = ? AND STRFTIME('%Y', independence) >= ? AND country IN ( SELECT country FROM country ) ORDER BY independence LIMIT ?", (government_type, independence_year, limit))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get organizations in dependent countries
@app.get("/v1/mondial_geo/organizations_in_dependent_countries", operation_id="get_organizations_in_dependent_countries", summary="Retrieves a list of organization names located in countries that are dependent on other countries. The endpoint does not require any input parameters and returns a JSON array of organization names.")
async def get_organizations_in_dependent_countries():
    cursor.execute("SELECT NAME FROM organization WHERE country IN ( SELECT country FROM politics WHERE dependent != '' )")
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": [row[0] for row in result]}

# Endpoint to get countries associated with a specific desert
@app.get("/v1/mondial_geo/countries_by_desert", operation_id="get_countries_by_desert", summary="Retrieves a list of countries that are associated with the specified desert. The operation uses the desert's name to identify and return the relevant country names.")
async def get_countries_by_desert(desert_name: str = Query(..., description="Name of the desert")):
    cursor.execute("SELECT T3.Name FROM desert AS T1 INNER JOIN geo_desert AS T2 ON T1.Name = T2.Desert INNER JOIN country AS T3 ON T3.Code = T2.Country WHERE T1.Name = ?", (desert_name,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the largest desert in a specific country
@app.get("/v1/mondial_geo/largest_desert_in_country", operation_id="get_largest_desert_in_country", summary="Retrieves the name of the largest desert located within the specified country. The operation uses the provided country name to search for corresponding deserts and returns the one with the largest area.")
async def get_largest_desert_in_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Name FROM desert AS T1 INNER JOIN geo_desert AS T2 ON T1.Name = T2.Desert INNER JOIN country AS T3 ON T3.Code = T2.Country WHERE T3.Name = ? ORDER BY T1.Area DESC LIMIT 1", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"desert": []}
    return {"desert": result[0]}

# Endpoint to get the sea that merges with a specific sea and its depth
@app.get("/v1/mondial_geo/sea_merges_with_and_depth", operation_id="get_sea_merges_with_and_depth", summary="Retrieves the name of the sea that merges with the specified sea and its corresponding depth. The operation requires the name of the sea as an input parameter.")
async def get_sea_merges_with_and_depth(sea_name: str = Query(..., description="Name of the sea")):
    cursor.execute("SELECT T2.Sea2, T1.Depth FROM sea AS T1 INNER JOIN mergesWith AS T2 ON T1.Name = T2.Sea1 WHERE T1.Name = ?", (sea_name,))
    result = cursor.fetchall()
    if not result:
        return {"seas": []}
    return {"seas": [{"sea": row[0], "depth": row[1]} for row in result]}

# Endpoint to get the country with the most organizations established after a specific year and with a specific government type
@app.get("/v1/mondial_geo/country_with_most_organizations", operation_id="get_country_with_most_organizations", summary="Retrieves the name of the country with the highest number of unique organizations established after the specified year and having the specified type of government. The year is provided in 'YYYY' format and the government type is a string.")
async def get_country_with_most_organizations(established_year: str = Query(..., description="Year established in 'YYYY' format"), government_type: str = Query(..., description="Type of government")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN organization AS T2 ON T1.Code = T2.Country INNER JOIN politics AS T3 ON T3.Country = T2.Country WHERE STRFTIME('%Y', T2.Established) > ? AND T3.Government = ? GROUP BY T1.Name ORDER BY COUNT(DISTINCT T2.Name) DESC LIMIT 1", (established_year, government_type))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the type, province, and country of a specific mountain
@app.get("/v1/mondial_geo/mountain_details", operation_id="get_mountain_details", summary="Retrieves the type, province, and country of a specific mountain. The operation requires the name of the mountain as input and returns the corresponding type, province, and country information.")
async def get_mountain_details(mountain_name: str = Query(..., description="Name of the mountain")):
    cursor.execute("SELECT T1.Type, T3.Name, T4.Name FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain INNER JOIN province AS T3 ON T3.Name = T2.Province INNER JOIN country AS T4 ON T3.Country = T4.Code WHERE T1.Name = ?", (mountain_name,))
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"type": result[0], "province": result[1], "country": result[2]}

# Endpoint to get mountains of a specific type in a specific province
@app.get("/v1/mondial_geo/mountains_by_type_and_province", operation_id="get_mountains_by_type_and_province", summary="Retrieves a list of mountains of a specified type located within a given province. The operation filters mountains based on the provided province name and mountain type, returning the names of the mountains that meet the specified criteria.")
async def get_mountains_by_type_and_province(province_name: str = Query(..., description="Name of the province"), mountain_type: str = Query(..., description="Type of the mountain")):
    cursor.execute("SELECT T1.Name FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain INNER JOIN province AS T3 ON T3.Name = T2.Province WHERE T3.Name = ? AND T1.Type = ?", (province_name, mountain_type))
    result = cursor.fetchall()
    if not result:
        return {"mountains": []}
    return {"mountains": [row[0] for row in result]}

# Endpoint to get the percentage of countries with a specific government type established after a certain year
@app.get("/v1/mondial_geo/government_type_percentage", operation_id="get_government_type_percentage", summary="Get the percentage of countries with a specific government type established after a certain year")
async def get_government_type_percentage(government_type: str = Query(..., description="Type of government"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(IIF(government = ?, 1, 0)) , CAST(SUM(IIF(government = ?, 1, 0)) AS REAL) * 100 / COUNT(*) FROM politics AS t1 WHERE STRFTIME('%Y', independence) >= ?", (government_type, government_type, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[1]}

# Endpoint to get the percentage of rivers longer than a specified length
@app.get("/v1/mondial_geo/river_length_percentage", operation_id="get_river_length_percentage", summary="Retrieves the percentage of rivers that exceed a specified length. This operation calculates the proportion of rivers with lengths greater than the provided value, based on the total number of rivers in the database. The result is expressed as a percentage.")
async def get_river_length_percentage(length: int = Query(..., description="Length of the river")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Length > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.Name) FROM river AS T1 INNER JOIN located AS T2 ON T1.Name = T2.River INNER JOIN country AS T3 ON T3.Code = T2.Country", (length,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the name and establishment year of an organization by its abbreviation
@app.get("/v1/mondial_geo/organization_by_abbreviation", operation_id="get_organization_by_abbreviation", summary="Retrieves the full name and establishment year of an organization based on its abbreviation. The input parameter specifies the organization's abbreviation.")
async def get_organization_by_abbreviation(abbreviation: str = Query(..., description="Abbreviation of the organization")):
    cursor.execute("SELECT Name, Established FROM organization WHERE Abbreviation = ?", (abbreviation,))
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": result}

# Endpoint to get the names of organizations established between two years
@app.get("/v1/mondial_geo/organizations_by_establishment_years", operation_id="get_organizations_by_establishment_years", summary="Retrieves the names of organizations that were established between the specified start and end years. The start and end years are inclusive and should be provided in 'YYYY' format.")
async def get_organizations_by_establishment_years(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT Name FROM organization WHERE STRFTIME('%Y', Established) BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": result}

# Endpoint to get the names of organizations in a specific city
@app.get("/v1/mondial_geo/organizations_by_city", operation_id="get_organizations_by_city", summary="Retrieve the names of organizations situated in a specified city. The operation filters organizations based on the provided city and returns their names.")
async def get_organizations_by_city(city: str = Query(..., description="City where the organization is located")):
    cursor.execute("SELECT Name FROM organization WHERE City = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": result}

# Endpoint to get the names and cities of organizations in a specific country
@app.get("/v1/mondial_geo/organizations_by_country", operation_id="get_organizations_by_country", summary="Retrieves the names and cities of organizations based in a specified country. The country is provided as an input parameter.")
async def get_organizations_by_country(country: str = Query(..., description="Country where the organization is located")):
    cursor.execute("SELECT Name, City FROM organization WHERE Country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": result}

# Endpoint to get the oldest organization in a specific city
@app.get("/v1/mondial_geo/oldest_organization_by_city", operation_id="get_oldest_organization_by_city", summary="Get the oldest organization in a specific city")
async def get_oldest_organization_by_city(city: str = Query(..., description="City where the organization is located")):
    cursor.execute("SELECT Abbreviation, Name, Established FROM organization WHERE City = ? ORDER BY Established ASC LIMIT 1", (city,))
    result = cursor.fetchone()
    if not result:
        return {"organization": []}
    return {"organization": result}

# Endpoint to get the names and cities of organizations with a specific name pattern
@app.get("/v1/mondial_geo/organizations_by_name_pattern", operation_id="get_organizations_by_name_pattern", summary="Retrieves the names and cities of organizations whose names contain a specified pattern. The pattern is used to filter the results, returning only those organizations that match the provided pattern.")
async def get_organizations_by_name_pattern(name_pattern: str = Query(..., description="Pattern to match in the organization name")):
    cursor.execute("SELECT Name, City FROM organization WHERE Name LIKE ?", (name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": result}

# Endpoint to get the names of countries bordering a specific country
@app.get("/v1/mondial_geo/bordering_countries", operation_id="get_bordering_countries", summary="Retrieve the names of countries that share a border with a specified country. The operation uses the provided country name to identify its corresponding code and then retrieves the names of countries that border it.")
async def get_bordering_countries(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T3.Name FROM country AS T1 INNER JOIN borders AS T2 ON T1.Code = T2.Country1 INNER JOIN country AS T3 ON T3.Code = T2.Country2 WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the names of countries with borders longer than a specified length
@app.get("/v1/mondial_geo/countries_with_long_borders", operation_id="get_countries_with_long_borders", summary="Retrieve the names of countries that have borders exceeding a specified length. The operation filters countries based on the provided border length and returns the names of those that meet the criteria.")
async def get_countries_with_long_borders(border_length: int = Query(..., description="Length of the border")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN borders AS T2 ON T1.Code = T2.Country1 WHERE T2.Length > ?", (border_length,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the most populous country in a specified organization
@app.get("/v1/mondial_geo/most_populous_country_in_organization", operation_id="get_most_populous_country_in_organization", summary="Retrieves the name of the country with the highest population that is a member of the specified organization.")
async def get_most_populous_country_in_organization(organization: str = Query(..., description="Name of the organization")):
    cursor.execute("SELECT T2.Name FROM isMember AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T1.Organization = ? ORDER BY T2.Population DESC LIMIT 1", (organization,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the countries and their types in a specified organization
@app.get("/v1/mondial_geo/countries_and_types_in_organization", operation_id="get_countries_and_types_in_organization", summary="Retrieves a list of countries and their respective types that belong to a specified organization. The operation requires the name of the organization as an input parameter to filter the results.")
async def get_countries_and_types_in_organization(organization_name: str = Query(..., description="Name of the organization")):
    cursor.execute("SELECT T2.Country, T2.Type FROM organization AS T1 INNER JOIN isMember AS T2 ON T1.Abbreviation = T2.Organization INNER JOIN country AS T3 ON T2.Country = T3.Code WHERE T1.Name = ?", (organization_name,))
    result = cursor.fetchall()
    if not result:
        return {"countries_and_types": []}
    return {"countries_and_types": [{"country": row[0], "type": row[1]} for row in result]}

# Endpoint to get the names and populations of countries in a specified organization
@app.get("/v1/mondial_geo/countries_in_organization", operation_id="get_countries_in_organization", summary="Retrieves the names and populations of countries that are members of a specified organization. The organization is identified by its name.")
async def get_countries_in_organization(organization_name: str = Query(..., description="Name of the organization")):
    cursor.execute("SELECT T2.Name, T2.Population FROM organization AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T1.Name = ?", (organization_name,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [{"name": row[0], "population": row[1]} for row in result]}

# Endpoint to get the names of organizations with a specified type of membership
@app.get("/v1/mondial_geo/organizations_with_membership_type", operation_id="get_organizations_with_membership_type", summary="Retrieves the names of organizations that have a specified type of membership. The membership type is used to filter the organizations. The response includes the names of the organizations that meet the specified membership criteria.")
async def get_organizations_with_membership_type(membership_type: str = Query(..., description="Type of membership")):
    cursor.execute("SELECT T1.Name FROM organization AS T1 INNER JOIN isMember AS T2 ON T2.Country = T1.Country INNER JOIN country AS T3 ON T2.Country = T3.Code WHERE T2.Type = ?", (membership_type,))
    result = cursor.fetchall()
    if not result:
        return {"organizations": []}
    return {"organizations": [row[0] for row in result]}

# Endpoint to get the names and capitals of countries that are members of a specified organization with a specified type
@app.get("/v1/mondial_geo/countries_in_organization_with_type", operation_id="get_countries_in_organization_with_type", summary="Retrieves the names and capitals of countries that are members of a specified organization with a specified membership type. The organization and membership type are provided as input parameters.")
async def get_countries_in_organization_with_type(organization: str = Query(..., description="Name of the organization"), membership_type: str = Query(..., description="Type of membership")):
    cursor.execute("SELECT Name, Capital FROM country WHERE Code IN ( SELECT Country FROM isMember WHERE type = ? AND Organization = ? )", (membership_type, organization))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [{"name": row[0], "capital": row[1]} for row in result]}

# Endpoint to get the organization with the most member countries
@app.get("/v1/mondial_geo/organization_with_most_member_countries", operation_id="get_organization_with_most_member_countries", summary="Retrieves the name of the organization that has the highest number of member countries. This operation performs a database query to aggregate and count the number of member countries for each organization, then returns the organization with the maximum count.")
async def get_organization_with_most_member_countries():
    cursor.execute("SELECT T1.Name FROM organization AS T1 INNER JOIN isMember AS T2 ON T2.Country = T1.Country INNER JOIN country AS T3 ON T2.Country = T3.Code GROUP BY T1.Name ORDER BY COUNT(T3.Name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"organization": []}
    return {"organization": result[0]}

# Endpoint to get the capital and organization name for a specified country
@app.get("/v1/mondial_geo/capital_and_organization_name", operation_id="get_capital_and_organization_name", summary="Retrieves the capital city and organization name for a specified country. The operation uses the provided country name to look up the corresponding capital city and organization name in the database. The result is a list of capital cities and organization names that match the input country name.")
async def get_capital_and_organization_name(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Capital, T1.Name FROM organization AS T1 INNER JOIN country AS T2 ON T1.City = T2.Capital WHERE T2.Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"capitals_and_organizations": []}
    return {"capitals_and_organizations": [{"capital": row[0], "organization": row[1]} for row in result]}

# Endpoint to get the percentage of organizations in a specified city for a specified country
@app.get("/v1/mondial_geo/percentage_organizations_in_city", operation_id="get_percentage_organizations_in_city", summary="Retrieves the percentage of organizations located in a specified city within a given country. The calculation is based on the total count of organizations in the specified country.")
async def get_percentage_organizations_in_city(city: str = Query(..., description="Name of the city"), country: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.City = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.City) FROM country AS T1 INNER JOIN organization AS T2 ON T1.Code = T2.Country WHERE T2.Country = ?", (city, country))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the length of the border between two countries
@app.get("/v1/mondial_geo/border_length", operation_id="get_border_length", summary="Retrieves the length of the border shared between the specified pair of countries.")
async def get_border_length(country1: str = Query(..., description="First country"), country2: str = Query(..., description="Second country")):
    cursor.execute("SELECT Length FROM borders WHERE Country1 = ? AND Country2 = ?", (country1, country2))
    result = cursor.fetchone()
    if not result:
        return {"length": []}
    return {"length": result[0]}

# Endpoint to get the most recently established organization for a given country
@app.get("/v1/mondial_geo/most_recent_organization", operation_id="get_most_recent_organization", summary="Retrieves the name of the most recently established organization in a specified country. The operation filters organizations based on the provided country name and returns the name of the organization that was established most recently.")
async def get_most_recent_organization(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T3.Name FROM country AS T1 INNER JOIN isMember AS T2 ON T1.Code = T2.Country INNER JOIN organization AS T3 ON T3.Country = T2.Country WHERE T1.Name = ? ORDER BY T3.Established DESC LIMIT 1", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"organization": []}
    return {"organization": result[0]}

# Endpoint to get the population of the city where a given organization is located
@app.get("/v1/mondial_geo/organization_city_population", operation_id="get_organization_city_population", summary="Retrieves the population of the city where the specified organization is located. The operation requires the name of the organization as an input parameter.")
async def get_organization_city_population(organization_name: str = Query(..., description="Name of the organization")):
    cursor.execute("SELECT T2.Population FROM organization AS T1 INNER JOIN city AS T2 ON T1.City = T2.Name WHERE T1.Name = ?", (organization_name,))
    result = cursor.fetchone()
    if not result:
        return {"population": []}
    return {"population": result[0]}

# Endpoint to get the height and province of a given mountain
@app.get("/v1/mondial_geo/mountain_height_province", operation_id="get_mountain_height_province", summary="Retrieves the height and province of a specific mountain. The operation requires the mountain's name as input and returns the corresponding height and province information.")
async def get_mountain_height_province(mountain_name: str = Query(..., description="Name of the mountain")):
    cursor.execute("SELECT T1.Height, T2.Province FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain WHERE T1.Name = ?", (mountain_name,))
    result = cursor.fetchone()
    if not result:
        return {"height": [], "province": []}
    return {"height": result[0], "province": result[1]}

# Endpoint to get the names and heights of mountains in a given province
@app.get("/v1/mondial_geo/mountains_in_province", operation_id="get_mountains_in_province", summary="Retrieves the names and heights of mountains located within a specified province. The operation requires the name of the province as an input parameter.")
async def get_mountains_in_province(province: str = Query(..., description="Name of the province")):
    cursor.execute("SELECT T1.Name, T1.Height FROM mountain AS T1 INNER JOIN geo_mountain AS T2 ON T1.Name = T2.Mountain WHERE T2.Province = ?", (province,))
    results = cursor.fetchall()
    if not results:
        return {"mountains": []}
    return {"mountains": [{"name": row[0], "height": row[1]} for row in results]}

# Endpoint to get the population of the country with the highest infant mortality rate
@app.get("/v1/mondial_geo/highest_infant_mortality_population", operation_id="get_highest_infant_mortality_population", summary="Retrieves the population of the country with the highest infant mortality rate. The operation ranks countries by their infant mortality rates in descending order and returns the population of the top-ranked country.")
async def get_highest_infant_mortality_population():
    cursor.execute("SELECT T1.Population FROM country AS T1 INNER JOIN population AS T2 ON T1.Code = T2.Country ORDER BY T2.Infant_Mortality DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"population": []}
    return {"population": result[0]}

# Endpoint to get the inflation rate of a given country
@app.get("/v1/mondial_geo/country_inflation", operation_id="get_country_inflation", summary="Retrieves the inflation rate for a specified country. The operation uses the provided country name to search for a match in the database and returns the corresponding inflation rate.")
async def get_country_inflation(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T2.Inflation FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T1.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"inflation": []}
    return {"inflation": result[0]}

# Endpoint to get the government type of the country with the highest agriculture contribution
@app.get("/v1/mondial_geo/highest_agriculture_government", operation_id="get_highest_agriculture_government", summary="Retrieves the type of government of the country that has the highest contribution to agriculture. This operation returns the government type of the country with the highest agriculture percentage, as determined by the economy and politics data.")
async def get_highest_agriculture_government():
    cursor.execute("SELECT T3.Government FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country INNER JOIN politics AS T3 ON T3.Country = T2.Country ORDER BY T2.Agriculture DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"government": []}
    return {"government": result[0]}

# Endpoint to get the capital of countries with a given government type
@app.get("/v1/mondial_geo/capital_by_government", operation_id="get_capital_by_government", summary="Retrieves the capital city of countries that have the specified government type. The operation filters countries based on their government type and returns the capital city of each matching country.")
async def get_capital_by_government(government_type: str = Query(..., description="Type of government")):
    cursor.execute("SELECT T1.Capital FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country WHERE T2.Government = ?", (government_type,))
    results = cursor.fetchall()
    if not results:
        return {"capitals": []}
    return {"capitals": [row[0] for row in results]}

# Endpoint to get the city with the highest population percentage relative to the country's population
@app.get("/v1/mondial_geo/city_highest_population_percentage", operation_id="get_city_highest_population_percentage", summary="Get the city with the highest population percentage relative to the country's population")
async def get_city_highest_population_percentage(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.Name, CAST(T3.Population AS REAL) * 100 / T1.Population FROM country AS T1 INNER JOIN province AS T2 ON T1.Code = T2.Country INNER JOIN city AS T3 ON T3.Country = T2.Country ORDER BY T3.Population DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result}

# Endpoint to get the independence date of a country
@app.get("/v1/mondial_geo/country_independence_date", operation_id="get_country_independence_date", summary="Retrieves the independence date of a specified country. The operation uses the provided country name to search for a match in the database and returns the corresponding independence date.")
async def get_country_independence_date(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T1.Independence FROM politics AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T2.Name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"independence_date": []}
    return {"independence_date": result[0]}

# Endpoint to get the highest mountain of a specific type
@app.get("/v1/mondial_geo/highest_mountain_by_type", operation_id="get_highest_mountain_by_type", summary="Retrieves the highest mountain of a specified type, along with its height, and allows limiting the number of results. The operation returns the mountain with the greatest height that matches the provided mountain type.")
async def get_highest_mountain_by_type(mountain_type: str = Query(..., description="Type of the mountain"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Height, Name FROM mountain WHERE Type = ? ORDER BY Height DESC LIMIT ?", (mountain_type, limit))
    result = cursor.fetchone()
    if not result:
        return {"mountain": []}
    return {"mountain": result}

# Endpoint to get the most recently established organization in a specified country
@app.get("/v1/mondial_geo/recent_organization_by_country", operation_id="get_recent_organization_by_country", summary="Retrieves the most recently established organization in a specified country, with the option to limit the number of results returned.")
async def get_recent_organization_by_country(country_name: str = Query(..., description="Name of the country"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Name FROM organization AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code WHERE T2.Name = ? ORDER BY T1.Established DESC LIMIT ?", (country_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"organization": []}
    return {"organization": result[0]}

# Endpoint to get countries with the highest infant mortality rates
@app.get("/v1/mondial_geo/countries_highest_infant_mortality", operation_id="get_countries_highest_infant_mortality", summary="Get countries with the highest infant mortality rates")
async def get_countries_highest_infant_mortality(limit: int = Query(..., description="Limit the number of results"), offset: int = Query(..., description="Offset for the results")):
    cursor.execute("SELECT T2.Name FROM population AS T1 INNER JOIN country AS T2 ON T1.Country = T2.Code ORDER BY T1.Infant_Mortality DESC LIMIT ?, ?", (offset, limit))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the country with the most diverse religions
@app.get("/v1/mondial_geo/country_most_diverse_religions", operation_id="get_country_most_diverse_religions", summary="Retrieves the country with the highest number of unique religions, up to the specified limit. This operation returns the name of the country with the most diverse religious landscape, based on the distinct number of religions present. The limit parameter allows you to restrict the number of results returned.")
async def get_country_most_diverse_religions(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Name FROM country AS T1 INNER JOIN religion AS T2 ON T1.Code = T2.Country GROUP BY T1.Name ORDER BY COUNT(DISTINCT T2.Name) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the longest river with pagination
@app.get("/v1/mondial_geo/longest_river", operation_id="get_longest_river", summary="Get the longest river with pagination")
async def get_longest_river(limit: int = Query(..., description="Limit the number of results"), offset: int = Query(..., description="Offset for the results")):
    cursor.execute("SELECT Name, Length FROM river ORDER BY Length DESC LIMIT ?, ?", (offset, limit))
    result = cursor.fetchone()
    if not result:
        return {"river": []}
    return {"river": result}

# Endpoint to get the independence date of a country by its capital
@app.get("/v1/mondial_geo/independence_date_by_capital", operation_id="get_independence_date_by_capital", summary="Retrieves the independence date of a country based on its capital. The operation uses the provided capital name to search for a matching country in the database. Once found, it returns the corresponding independence date from the politics table.")
async def get_independence_date_by_capital(capital: str = Query(..., description="Capital of the country")):
    cursor.execute("SELECT T2.Independence FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country WHERE T1.Capital = ?", (capital,))
    result = cursor.fetchone()
    if not result:
        return {"independence_date": []}
    return {"independence_date": result[0]}

# Endpoint to get the country with the smallest population and its GDP
@app.get("/v1/mondial_geo/smallest_population_country_gdp", operation_id="get_smallest_population_country_gdp", summary="Retrieves the country with the smallest population and its corresponding GDP, up to the specified limit. This operation provides a concise overview of the least populated countries and their economic output.")
async def get_smallest_population_country_gdp(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Name, T2.GDP FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country ORDER BY T1.Population ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result}

# Endpoint to get lakes located in a specified country
@app.get("/v1/mondial_geo/lakes_in_country", operation_id="get_lakes_in_country", summary="Retrieves a list of lakes located within a specified country, along with their respective depths and the names of the provinces they are situated in. The country is identified by its name.")
async def get_lakes_in_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT T3.Name, T1.Name, T1.Depth FROM lake AS T1 INNER JOIN located AS T2 ON T1.Name = T2.Lake INNER JOIN province AS T3 ON T3.Name = T2.Province INNER JOIN country AS T4 ON T4.Code = T3.Country WHERE T4.Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"lakes": []}
    return {"lakes": [{"province": row[0], "lake": row[1], "depth": row[2]} for row in result]}

# Endpoint to get the highest mountain on a given island
@app.get("/v1/mondial_geo/highest_mountain_on_island", operation_id="get_highest_mountain", summary="Retrieves the highest mountain on a specified island. The operation returns the height and type of the mountain. The island is identified by its name, which is provided as an input parameter.")
async def get_highest_mountain(island_name: str = Query(..., description="Name of the island")):
    cursor.execute("SELECT T3.Height, T3.Type FROM island AS T1 INNER JOIN mountainOnIsland AS T2 ON T1.Name = T2.Island INNER JOIN mountain AS T3 ON T3.Name = T2.Mountain WHERE T1.Name = ? ORDER BY T3.Height DESC LIMIT 1", (island_name,))
    result = cursor.fetchone()
    if not result:
        return {"mountain": []}
    return {"height": result[0], "type": result[1]}

# Endpoint to get the country with the highest GDP within a given population range
@app.get("/v1/mondial_geo/highest_gdp_country_in_population_range", operation_id="get_highest_gdp_country", summary="Retrieves the country with the highest Gross Domestic Product (GDP) from a specified population range. The population range is defined by the minimum and maximum population values provided as input parameters. The operation returns the name of the country and its corresponding GDP.")
async def get_highest_gdp_country(min_population: int = Query(..., description="Minimum population"), max_population: int = Query(..., description="Maximum population")):
    cursor.execute("SELECT T1.Name, T2.GDP FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T1.Population BETWEEN ? AND ? ORDER BY T2.GDP DESC LIMIT 1", (min_population, max_population))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"name": result[0], "gdp": result[1]}

# Endpoint to get the country with the highest agriculture in a given continent
@app.get("/v1/mondial_geo/highest_agriculture_country_in_continent", operation_id="get_highest_agriculture_country", summary="Retrieves the country with the highest agricultural output within the specified continent. The operation uses the provided continent name to identify the relevant continent and its encompassed countries. It then ranks these countries based on their agricultural output and returns the top-ranked country.")
async def get_highest_agriculture_country(continent_name: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT T2.Country FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T2.Country = T3.Code INNER JOIN economy AS T4 ON T4.Country = T3.Code WHERE T1.Name = ? ORDER BY T4.Agriculture DESC LIMIT 1", (continent_name,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the government type of the country with the lowest GDP
@app.get("/v1/mondial_geo/government_of_lowest_gdp_country", operation_id="get_government_of_lowest_gdp_country", summary="Retrieves the type of government of the country with the lowest Gross Domestic Product (GDP). The operation identifies the country with the lowest GDP and returns the corresponding government type.")
async def get_government_of_lowest_gdp_country():
    cursor.execute("SELECT T3.Government FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country INNER JOIN politics AS T3 ON T3.Country = T2.Country WHERE T2.GDP IS NOT NULL ORDER BY T2.GDP ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"government": []}
    return {"government": result[0]}

# Endpoint to get the year with the most organizations established in a given continent
@app.get("/v1/mondial_geo/most_organizations_established_year_in_continent", operation_id="get_most_organizations_established_year", summary="Retrieves the year with the highest number of organizations established in a specified continent. The continent is identified by its name.")
async def get_most_organizations_established_year(continent_name: str = Query(..., description="Name of the continent")):
    cursor.execute("SELECT STRFTIME('%Y', T4.Established) FROM continent AS T1 INNER JOIN encompasses AS T2 ON T1.Name = T2.Continent INNER JOIN country AS T3 ON T2.Country = T3.Code INNER JOIN organization AS T4 ON T4.Country = T3.Code WHERE T1.Name = ? GROUP BY STRFTIME('%Y', T4.Established) ORDER BY COUNT(T4.Name) DESC LIMIT 1", (continent_name,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the borders of the most populous country
@app.get("/v1/mondial_geo/borders_of_most_populous_country", operation_id="get_borders_of_most_populous_country", summary="Retrieves the border details of the country with the highest population. The response includes the name of the neighboring country and the length of the shared border.")
async def get_borders_of_most_populous_country():
    cursor.execute("SELECT T2.Country2, T2.Length FROM country AS T1 INNER JOIN borders AS T2 ON T1.Code = T2.Country1 INNER JOIN country AS T3 ON T3.Code = T2.Country2 WHERE T1.Name = ( SELECT Name FROM country ORDER BY Population DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"borders": []}
    return {"borders": [{"country": row[0], "length": row[1]} for row in result]}

# Endpoint to get the population density and industry of a given province
@app.get("/v1/mondial_geo/population_density_and_industry_of_province", operation_id="get_population_density_and_industry", summary="Get the population density and industry of a given province")
async def get_population_density_and_industry(province_name: str = Query(..., description="Name of the province")):
    cursor.execute("SELECT T1.Population / T1.Area, T2.Industry FROM country AS T1 INNER JOIN economy AS T2 ON T1.Code = T2.Country WHERE T1.Province = ?", (province_name,))
    result = cursor.fetchone()
    if not result:
        return {"province": []}
    return {"population_density": result[0], "industry": result[1]}

# Endpoint to get political information for countries with independence between given years and a specific government type
@app.get("/v1/mondial_geo/politics_by_independence_years_and_government", operation_id="get_politics_by_independence_years_and_government", summary="Get political information for countries with independence between given years and a specific government type")
async def get_politics_by_independence_years_and_government(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), government_type: str = Query(..., description="Type of government")):
    cursor.execute("SELECT * FROM politics WHERE STRFTIME('%Y', Independence) BETWEEN ? AND ? AND Government = ?", (start_year, end_year, government_type))
    result = cursor.fetchall()
    if not result:
        return {"politics": []}
    return {"politics": [dict(row) for row in result]}

# Endpoint to get the percentage of countries that gained independence in a specific year
@app.get("/v1/mondial_geo/percentage_independence_in_year", operation_id="get_percentage_independence_in_year", summary="Retrieves the percentage of countries that gained independence in a specific year. The calculation is based on the total number of countries and the count of countries that gained independence in the provided year. The year should be provided in 'YYYY' format.")
async def get_percentage_independence_in_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN STRFTIME('%Y', Independence) = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(Country) FROM politics", (year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the names of deserts outside a given latitude range
@app.get("/v1/mondial_geo/deserts_outside_latitude_range", operation_id="get_deserts_outside_latitude_range", summary="Retrieve the names of deserts that are located outside the specified latitude range. This operation requires the minimum and maximum latitude values to filter the deserts accordingly.")
async def get_deserts_outside_latitude_range(min_latitude: float = Query(..., description="Minimum latitude"), max_latitude: float = Query(..., description="Maximum latitude")):
    cursor.execute("SELECT Name FROM desert WHERE Latitude < ? OR Latitude > ?", (min_latitude, max_latitude))
    result = cursor.fetchall()
    if not result:
        return {"deserts": []}
    return {"deserts": [row[0] for row in result]}

# Endpoint to get the latitude and longitude of deserts with a name count greater than a specified value
@app.get("/v1/mondial_geo/desert_coordinates", operation_id="get_desert_coordinates", summary="Retrieve the geographical coordinates of deserts that appear more than a specified number of times in the database. This operation returns the latitude and longitude of deserts that meet the specified name count threshold.")
async def get_desert_coordinates(name_count: int = Query(..., description="Minimum count of desert names")):
    cursor.execute("SELECT T1.Latitude, T1.Longitude FROM desert AS T1 INNER JOIN geo_desert AS T2 ON T1.Name = T2.Desert GROUP BY T1.Name, T1.Latitude, T1.Longitude HAVING COUNT(T1.Name) > ?", (name_count,))
    result = cursor.fetchall()
    if not result:
        return {"coordinates": []}
    return {"coordinates": result}

# Endpoint to get the capital or province with the highest population density among those with a population less than a specified value
@app.get("/v1/mondial_geo/capital_province_highest_density", operation_id="get_capital_province_highest_density", summary="Retrieves the capital or province with the highest population density from those with a population less than the specified maximum. The input parameter determines the upper limit for the population of the provinces considered.")
async def get_capital_province_highest_density(max_population: int = Query(..., description="Maximum population of the province")):
    cursor.execute("SELECT CapProv FROM province WHERE Population < ? ORDER BY Population / Area DESC LIMIT 1", (max_population,))
    result = cursor.fetchone()
    if not result:
        return {"capital_province": []}
    return {"capital_province": result[0]}

api_calls = [
    "/v1/mondial_geo/country_with_lowest_ethnic_percentage?ethnic_group=Polish",
    "/v1/mondial_geo/countries_with_ethnic_percentage_above?ethnic_group=African&percentage=90",
    "/v1/mondial_geo/ethnic_groups_in_country?country_name=Singapore",
    "/v1/mondial_geo/percentage_of_republics_after_year?year=1970",
    "/v1/mondial_geo/gdp_and_government_of_country?country_name=Bosnia%20and%20Herzegovina",
    "/v1/mondial_geo/countries_with_population_growth_and_infant_mortality?population_growth=2&infant_mortality=5",
    "/v1/mondial_geo/top_ethnic_groups_by_population?population=10000000",
    "/v1/mondial_geo/ethnic_groups_in_most_diverse_country",
    "/v1/mondial_geo/countries_with_specific_ethnic_percentage?percentage=100&ethnic_group=African",
    "/v1/mondial_geo/infant_mortality_by_lowest_ethnic_percentage?ethnic_group=Amerindian",
    "/v1/mondial_geo/agriculture_by_area?area=600000",
    "/v1/mondial_geo/highest_population_growth_by_government?government=republic",
    "/v1/mondial_geo/independence_by_country?country_name=Bulgaria",
    "/v1/mondial_geo/ethnic_group_population_percentage?ethnic_group=Arab",
    "/v1/mondial_geo/ethnic_group_population_percentage_by_country?ethnic_group=African&country_name=Turks%20and%20Caicos%20Islands",
    "/v1/mondial_geo/lowest_infant_mortality_population_growth",
    "/v1/mondial_geo/capital_population_by_gdp?gdp=400000",
    "/v1/mondial_geo/service_sector_gdp_by_country?country_name=Brazil",
    "/v1/mondial_geo/highest_infant_mortality_population_growth",
    "/v1/mondial_geo/countries_by_negative_population_growth?population_growth=0",
    "/v1/mondial_geo/country_infant_mortality_by_area?min_area=500000&max_area=1000000",
    "/v1/mondial_geo/country_gdp_by_population_growth?population_growth=3",
    "/v1/mondial_geo/infant_mortality_by_country?country_name=Ethiopia",
    "/v1/mondial_geo/gdp_industry_product_by_country?country_name=Singapore",
    "/v1/mondial_geo/gdp_agriculture_product_smallest_area",
    "/v1/mondial_geo/country_highest_ethnic_group_percentage?ethnic_group=Albanian",
    "/v1/mondial_geo/count_countries_ethnic_group_area?ethnic_group=African&min_area=10000000",
    "/v1/mondial_geo/countries_with_multiple_ethnic_groups?min_ethnic_groups=5",
    "/v1/mondial_geo/country_highest_gdp",
    "/v1/mondial_geo/count_countries_gdp_population?min_gdp=500000&min_population=10000000",
    "/v1/mondial_geo/capital_by_inflation?inflation=2",
    "/v1/mondial_geo/country_with_lowest_inflation",
    "/v1/mondial_geo/count_countries_by_agriculture_area?agriculture=50&area=8000000",
    "/v1/mondial_geo/count_cities_by_lake_type?lake_type=salt",
    "/v1/mondial_geo/lake_depth_by_province?province=Albania",
    "/v1/mondial_geo/city_with_highest_altitude_lake",
    "/v1/mondial_geo/count_lakes_by_province_area?province=Canary%20Islands&area=1000000",
    "/v1/mondial_geo/country_with_most_languages",
    "/v1/mondial_geo/capital_by_language_percentage?language=Armenian&percentage=90",
    "/v1/mondial_geo/countries_by_population_languages?population=1000000&language_count=2",
    "/v1/mondial_geo/count_organizations_by_population?population=1000000",
    "/v1/mondial_geo/count_organizations_by_gdp_and_year?gdp=500000&year=1999",
    "/v1/mondial_geo/count_countries_by_organizations_and_inflation?organization_count=3&inflation=5",
    "/v1/mondial_geo/country_with_most_ethnic_groups",
    "/v1/mondial_geo/organizations_by_language?language=Dutch",
    "/v1/mondial_geo/count_organizations_by_language?language=Bosnian",
    "/v1/mondial_geo/max_infant_mortality_by_inflation?inflation=3",
    "/v1/mondial_geo/count_countries_by_gdp_and_population_growth?gdp=1000000&population_growth=3",
    "/v1/mondial_geo/country_with_highest_gdp_per_capita",
    "/v1/mondial_geo/largest_lake_area_percentage",
    "/v1/mondial_geo/average_population_growth_by_language_count?language_count=3",
    "/v1/mondial_geo/countries_with_high_inflation?inflation_multiplier=1.3",
    "/v1/mondial_geo/countries_by_province?province=Baghdad",
    "/v1/mondial_geo/most_populous_religion_by_country?country_name=Martinique",
    "/v1/mondial_geo/countries_by_religion_and_percentage?religion_name=Christian&percentage=41",
    "/v1/mondial_geo/countries_on_river?river_name=Detroit%20River",
    "/v1/mondial_geo/longest_border",
    "/v1/mondial_geo/country_with_most_borders",
    "/v1/mondial_geo/countries_with_mountain?mountain_name=Cerro%20Chirripo",
    "/v1/mondial_geo/count_mountains_in_country?country_name=Indonesia",
    "/v1/mondial_geo/latitude_of_island_with_mountain?mountain_name=Andringitra",
    "/v1/mondial_geo/country_code_with_second_highest_mountain",
    "/v1/mondial_geo/percentage_of_country_in_continent?continent_name=Asia&country_name=Egypt",
    "/v1/mondial_geo/percentage_of_country_area_in_continent?continent_name=Asia&country_name=Egypt",
    "/v1/mondial_geo/most_populous_city_in_country?country_name=Japan",
    "/v1/mondial_geo/capital_of_country_with_city?city_name=Olsztyn",
    "/v1/mondial_geo/province_of_highest_volcano?mountain_type=volcano",
    "/v1/mondial_geo/government_type_of_country?country_name=Iran",
    "/v1/mondial_geo/country_name_on_island?country_name=Bermuda",
    "/v1/mondial_geo/capital_by_ethnic_group?ethnic_group=Malay",
    "/v1/mondial_geo/ethnic_group_in_second_largest_country",
    "/v1/mondial_geo/country_by_city_population?city_population=114339",
    "/v1/mondial_geo/river_count_by_sea_depth?sea_depth=459",
    "/v1/mondial_geo/country_area_by_independence_date?independence_date=1921-03-13",
    "/v1/mondial_geo/population_density_by_city?city_name=Petropavl",
    "/v1/mondial_geo/ethnic_group_population_difference?ethnic_group1=Scottish&ethnic_group2=English&country_name=United%20Kingdom",
    "/v1/mondial_geo/most_populous_city_in_12th_densest_country",
    "/v1/mondial_geo/border_length_ratio?country_name=United%20States",
    "/v1/mondial_geo/capitals_with_mountain_count?min_mountain_count=4",
    "/v1/mondial_geo/mountain_count_most_populous_country",
    "/v1/mondial_geo/count_countries_industry_mountains?max_industry=40&max_mountain_count=2",
    "/v1/mondial_geo/mountains_lowest_inflation_country",
    "/v1/mondial_geo/count_deserts_government_type?government_type=republic",
    "/v1/mondial_geo/deserts_area_population?min_area=100000&max_population=500000",
    "/v1/mondial_geo/count_deserts_language_percentage?language_name=Armenian&min_percentage=90",
    "/v1/mondial_geo/highest_mountain_independent_country",
    "/v1/mondial_geo/count_volcanic_mountains_population?mountain_type=volcanic&max_population=5000000",
    "/v1/mondial_geo/count_countries_mountain_height_gdp?min_gdp=1000000&min_height=1000",
    "/v1/mondial_geo/max_border_length_independent_countries",
    "/v1/mondial_geo/count_distinct_countries_government_border_length?government_type=republic&border_length=200",
    "/v1/mondial_geo/country_shortest_border_length",
    "/v1/mondial_geo/total_gdp_continent?continent_name=Europe",
    "/v1/mondial_geo/count_mountains_continent?continent_name=European",
    "/v1/mondial_geo/largest_desert_continent?continent_name=America",
    "/v1/mondial_geo/countries_population_growth_continent?continent_name=Europe&population_growth=0.03",
    "/v1/mondial_geo/count_countries_infant_mortality_continent?continent_name=Europe&infant_mortality=100",
    "/v1/mondial_geo/count_distinct_countries_language_population_growth?language_name=Bosnian&population_growth=0",
    "/v1/mondial_geo/average_agriculture_continent?continent_name=Africa",
    "/v1/mondial_geo/count_distinct_countries_gdp?gdp=5000",
    "/v1/mondial_geo/avg_inflation_largest_continent",
    "/v1/mondial_geo/island_info_by_city?city_name=Balikpapan",
    "/v1/mondial_geo/cities_info_by_island?island_name=Sumatra",
    "/v1/mondial_geo/city_coordinates_by_province?province_name=South%20Yorkshire",
    "/v1/mondial_geo/islands_larger_than_province_island?province_name=Warwickshire",
    "/v1/mondial_geo/islands_cities_by_area?area=200",
    "/v1/mondial_geo/province_capital_by_city?city_name=Glenrothes",
    "/v1/mondial_geo/cities_by_province_population?province_population=1000000",
    "/v1/mondial_geo/cities_provinces_by_island_type?island_type=coral",
    "/v1/mondial_geo/average_population_cities_near_sea?sea_name=Baltic%20Sea",
    "/v1/mondial_geo/city_population_percentage_of_province?city_name=Edmonton",
    "/v1/mondial_geo/rivers_flowing_into_sea?sea_name=Black%20Sea",
    "/v1/mondial_geo/lakes_and_cities_in_province?province_name=Albania",
    "/v1/mondial_geo/highest_mountain_in_range?mountain_range=Himalaya",
    "/v1/mondial_geo/mountains_by_type?mountain_type=volcano",
    "/v1/mondial_geo/mountains_by_type_and_height_range?mountain_type=volcano&min_height=2000&max_height=4000",
    "/v1/mondial_geo/longest_river_into_sea?sea_name=Mediterranean%20Sea",
    "/v1/mondial_geo/percentage_non_specific_type_mountains?mountain_type=volcano&mountain_range=Andes",
    "/v1/mondial_geo/cities_near_rivers_flowing_into_sea?sea_name=Atlantic%20Ocean",
    "/v1/mondial_geo/river_info_by_city?city_name=Orleans",
    "/v1/mondial_geo/mountain_info_by_river?river_name=Lech",
    "/v1/mondial_geo/river_source_info_by_province?province_name=Lorraine",
    "/v1/mondial_geo/mountain_info_by_river_source?river_name=Blue%20Nile",
    "/v1/mondial_geo/river_length_by_city?city_name=Little%20Rock",
    "/v1/mondial_geo/rivers_by_length?min_length=1000",
    "/v1/mondial_geo/mountain_info_by_name?mountain_name=Moldoveanu",
    "/v1/mondial_geo/rivers_by_country?country_name=USA",
    "/v1/mondial_geo/average_mountain_height_by_province?province_name=Nepal",
    "/v1/mondial_geo/city_population_range_by_river?river_name=Seine",
    "/v1/mondial_geo/river_details_by_city?city_name=Belgrade",
    "/v1/mondial_geo/countries_by_language_percentage?language_name=Spanish&percentage=100",
    "/v1/mondial_geo/countries_by_government_type?government_type=British%20crown%20dependency",
    "/v1/mondial_geo/distinct_rivers_by_country?country_name=Canada",
    "/v1/mondial_geo/country_with_highest_inflation",
    "/v1/mondial_geo/top_provinces_by_population?country_name=United%20Kingdom&limit=3&offset=1",
    "/v1/mondial_geo/ethnic_group_population?country_name=Moldova&ethnic_group=Jewish",
    "/v1/mondial_geo/average_area_by_continent?continent=Asia",
    "/v1/mondial_geo/smallest_desert_details",
    "/v1/mondial_geo/language_speakers_population?language_name=Serbian&country_name=Montenegro",
    "/v1/mondial_geo/count_mountains_largest_country",
    "/v1/mondial_geo/countries_shallowest_sea",
    "/v1/mondial_geo/country_lowest_gdp_government?government_type=Communist%20state",
    "/v1/mondial_geo/government_highest_inflation",
    "/v1/mondial_geo/country_highest_infant_mortality_independence_year?independence_year=1960",
    "/v1/mondial_geo/government_types_shortest_border",
    "/v1/mondial_geo/country_smallest_population_language_percentage?language=Arabic&percentage=100",
    "/v1/mondial_geo/provinces_largest_desert",
    "/v1/mondial_geo/count_lakes_least_populous_republic_africa?continent=Africa&percentage=100&government=republic",
    "/v1/mondial_geo/most_prevalent_religion_continent?continent=Asia",
    "/v1/mondial_geo/highest_mountain_details",
    "/v1/mondial_geo/sea_surrounding_largest_island",
    "/v1/mondial_geo/countries_with_cities_on_longest_river?sea=Atlantic%20Ocean",
    "/v1/mondial_geo/capital_population_by_independence_date?independence_date=1947-08-15",
    "/v1/mondial_geo/cities_and_capitals_on_river?river_name=Euphrat",
    "/v1/mondial_geo/language_population_percentage?dependent_country=USA&language=English",
    "/v1/mondial_geo/population_density_and_service?continent=Europe",
    "/v1/mondial_geo/capital_population_percentage?continent=Asia",
    "/v1/mondial_geo/second_largest_desert",
    "/v1/mondial_geo/most_spoken_language?country=MNE",
    "/v1/mondial_geo/language_percentage?country_name=Cayman%20Islands&language_name=English",
    "/v1/mondial_geo/country_lowest_gdp?min_population=1000000000",
    "/v1/mondial_geo/mountain_capital?mountain_name=Licancabur",
    "/v1/mondial_geo/seas_around_mountain_island?mountain_name=Kerinci",
    "/v1/mondial_geo/countries_with_cities_on_river?river_name=Amazonas",
    "/v1/mondial_geo/countries_by_independence_date?independence_date=1492-01-01",
    "/v1/mondial_geo/city_count_by_population?country_name=France&min_population=100000",
    "/v1/mondial_geo/longest_river_by_sea_depth?sea_depth=540",
    "/v1/mondial_geo/highest_mountain_country?limit=1&offset=1",
    "/v1/mondial_geo/mountain_longitude_on_island?mountain_name=Olympos",
    "/v1/mondial_geo/country_highest_gdp_area_less_than?area=100&limit=1",
    "/v1/mondial_geo/count_cities_in_country?country_name=Japan",
    "/v1/mondial_geo/most_populous_city_excluding_capital?country_name=Bangladesh&limit=1",
    "/v1/mondial_geo/most_populous_city_excluding_capital_global?limit=1",
    "/v1/mondial_geo/country_of_city?city_name=Grozny",
    "/v1/mondial_geo/most_prevalent_religion?country_name=Japan&limit=1",
    "/v1/mondial_geo/countries_with_border_length?border_length=803",
    "/v1/mondial_geo/country_percentage_in_continent?country_name=Russia&continent_name=Europe",
    "/v1/mondial_geo/country_names_by_count?count=1",
    "/v1/mondial_geo/country_population_by_city?city_name=Fareham",
    "/v1/mondial_geo/country_infant_mortality_product?country_name=Switzerland",
    "/v1/mondial_geo/mountain_count_by_country?country_name=United%20States",
    "/v1/mondial_geo/country_gdp_per_capita?country_name=Switzerland",
    "/v1/mondial_geo/city_service_gdp_product?city_name=Fuenlabrada",
    "/v1/mondial_geo/river_length_ratio_by_country?country_code=TJ",
    "/v1/mondial_geo/country_population_density_by_city?city_name=Hanoi",
    "/v1/mondial_geo/ethnic_group_by_max_percentage?max_percentage=100",
    "/v1/mondial_geo/deserts_with_distinct_countries?min_countries=1",
    "/v1/mondial_geo/rivers_with_distinct_countries?min_countries=1",
    "/v1/mondial_geo/border_length_percentage?country_name=Angola",
    "/v1/mondial_geo/island_percentage?max_area=300&islands=Lesser%20Antilles&island_type=volcanic",
    "/v1/mondial_geo/capitals_by_population_percentage?population_percentage=0.0005",
    "/v1/mondial_geo/river_details?river_name=Donau",
    "/v1/mondial_geo/non_target_percentage?target_group=Christian",
    "/v1/mondial_geo/member_countries?organization_abbreviation=EBRD&min_population=50000&max_population=300000&start_date=1991-01-31&end_date=1991-04-30",
    "/v1/mondial_geo/rivers_in_country?country_name=Slovenia&min_length=500&river_name=Donau",
    "/v1/mondial_geo/cities_by_sea_name_pattern_and_depth_difference?sea_name_pattern=%25Bengal%25&depth_difference=4235",
    "/v1/mondial_geo/cities_by_lake_longitude_and_latitude?longitude=-85.35&latitude=11.6",
    "/v1/mondial_geo/continent_with_highest_inflation_country",
    "/v1/mondial_geo/bordering_countries_by_continent_and_border_length?continent_name=Asia&border_length=1782",
    "/v1/mondial_geo/deepest_lake_in_country?country_name=Bolivia",
    "/v1/mondial_geo/lakes_by_river_name?river_name=Manicouagan",
    "/v1/mondial_geo/islands_by_mountain_name?mountain_name=Rinjani",
    "/v1/mondial_geo/sea_merging_with_deepest_sea",
    "/v1/mondial_geo/countries_by_continent_count_and_population_density?continent_count=1&max_population_density=10",
    "/v1/mondial_geo/country_with_lowest_population_density_for_religion?religion_name=Hindu",
    "/v1/mondial_geo/independence_by_gdp?gdp=1100",
    "/v1/mondial_geo/population_density_by_established_year?year=1947",
    "/v1/mondial_geo/capital_province_by_country_name?name=Anguilla",
    "/v1/mondial_geo/smallest_population_country",
    "/v1/mondial_geo/area_difference_between_continents?name1=Asia&name2=Europe",
    "/v1/mondial_geo/city_coordinates_by_name?name=Aarhus",
    "/v1/mondial_geo/population_difference_between_countries?name1=United%20Kingdom&name2=Italy",
    "/v1/mondial_geo/organization_location_by_name?name=European%20Bank%20for%20Reconstruction%20and%20Development",
    "/v1/mondial_geo/largest_volume_lake",
    "/v1/mondial_geo/longest_border_countries",
    "/v1/mondial_geo/country_by_ethnic_group?ethnic_group=African",
    "/v1/mondial_geo/countries_by_religion_and_language?religion1=Anglican&religion2=Christian&religion3=Roman%20Catholic&language=English&percentage=100",
    "/v1/mondial_geo/top_countries_by_inflation",
    "/v1/mondial_geo/countries_by_language?language=English&percentage=100",
    "/v1/mondial_geo/count_organizations_established_after_year?year=1960",
    "/v1/mondial_geo/province_by_river?river_name=Klaraelv",
    "/v1/mondial_geo/province_count_and_sea_depth?country_code=I&sea_name=Mediterranean%20Sea",
    "/v1/mondial_geo/countries_by_government?government_type=British%20Overseas%20Territories",
    "/v1/mondial_geo/top_country_by_gdp_and_agriculture",
    "/v1/mondial_geo/continent_area_and_country_count?continent_name=Asia",
    "/v1/mondial_geo/province_with_most_organizations?country_name=United%20States",
    "/v1/mondial_geo/top_countries_by_independence?limit=3",
    "/v1/mondial_geo/top_countries_by_government_and_independence?government_type=republic&independence_year=1991&limit=3",
    "/v1/mondial_geo/organizations_in_dependent_countries",
    "/v1/mondial_geo/countries_by_desert?desert_name=Kalahari",
    "/v1/mondial_geo/largest_desert_in_country?country_name=Kazakstan",
    "/v1/mondial_geo/sea_merges_with_and_depth?sea_name=Baltic%20Sea",
    "/v1/mondial_geo/country_with_most_organizations?established_year=1907&government_type=constitutional%20monarchy",
    "/v1/mondial_geo/mountain_details?mountain_name=Ampato",
    "/v1/mondial_geo/mountains_by_type_and_province?province_name=Ecuador&mountain_type=volcano",
    "/v1/mondial_geo/government_type_percentage?government_type=parliamentary%20democracy&year=1993",
    "/v1/mondial_geo/river_length_percentage?length=3000",
    "/v1/mondial_geo/organization_by_abbreviation?abbreviation=ABEDA",
    "/v1/mondial_geo/organizations_by_establishment_years?start_year=1970&end_year=1980",
    "/v1/mondial_geo/organizations_by_city?city=London",
    "/v1/mondial_geo/organizations_by_country?country=USA",
    "/v1/mondial_geo/oldest_organization_by_city?city=Paris",
    "/v1/mondial_geo/organizations_by_name_pattern?name_pattern=%25United%20Nation%25",
    "/v1/mondial_geo/bordering_countries?country_name=Bulgaria",
    "/v1/mondial_geo/countries_with_long_borders?border_length=4000",
    "/v1/mondial_geo/most_populous_country_in_organization?organization=IOC",
    "/v1/mondial_geo/countries_and_types_in_organization?organization_name=Islamic%20Development%20Bank",
    "/v1/mondial_geo/countries_in_organization?organization_name=Asia%20Pacific%20Economic%20Cooperation",
    "/v1/mondial_geo/organizations_with_membership_type?membership_type=National%20Society",
    "/v1/mondial_geo/countries_in_organization_with_type?organization=IFAD&membership_type=Category%20III",
    "/v1/mondial_geo/organization_with_most_member_countries",
    "/v1/mondial_geo/capital_and_organization_name?country_name=Australia",
    "/v1/mondial_geo/percentage_organizations_in_city?city=Washington&country=USA",
    "/v1/mondial_geo/border_length?country1=MEX&country2=USA",
    "/v1/mondial_geo/most_recent_organization?country_name=Singapore",
    "/v1/mondial_geo/organization_city_population?organization_name=World%20Tourism%20Organization",
    "/v1/mondial_geo/mountain_height_province?mountain_name=Dhaulagiri",
    "/v1/mondial_geo/mountains_in_province?province=Alaska",
    "/v1/mondial_geo/highest_infant_mortality_population",
    "/v1/mondial_geo/country_inflation?country_name=Greece",
    "/v1/mondial_geo/highest_agriculture_government",
    "/v1/mondial_geo/capital_by_government?government_type=parliamentary%20democracy",
    "/v1/mondial_geo/city_highest_population_percentage?limit=1",
    "/v1/mondial_geo/country_independence_date?country_name=United%20States",
    "/v1/mondial_geo/highest_mountain_by_type?mountain_type=volcanic&limit=1",
    "/v1/mondial_geo/recent_organization_by_country?country_name=Saudi%20Arabia&limit=1",
    "/v1/mondial_geo/countries_highest_infant_mortality?limit=4&offset=1",
    "/v1/mondial_geo/country_most_diverse_religions?limit=1",
    "/v1/mondial_geo/longest_river?limit=16&offset=1",
    "/v1/mondial_geo/independence_date_by_capital?capital=Nouakchott",
    "/v1/mondial_geo/smallest_population_country_gdp?limit=1",
    "/v1/mondial_geo/lakes_in_country?country_name=Zaire",
    "/v1/mondial_geo/highest_mountain_on_island?island_name=Madagaskar",
    "/v1/mondial_geo/highest_gdp_country_in_population_range?min_population=60000000&max_population=90000000",
    "/v1/mondial_geo/highest_agriculture_country_in_continent?continent_name=Asia",
    "/v1/mondial_geo/government_of_lowest_gdp_country",
    "/v1/mondial_geo/most_organizations_established_year_in_continent?continent_name=Europe",
    "/v1/mondial_geo/borders_of_most_populous_country",
    "/v1/mondial_geo/population_density_and_industry_of_province?province_name=Distrito%20Federal",
    "/v1/mondial_geo/politics_by_independence_years_and_government?start_year=1950&end_year=1999&government_type=parliamentary%20democracy",
    "/v1/mondial_geo/percentage_independence_in_year?year=1960",
    "/v1/mondial_geo/deserts_outside_latitude_range?min_latitude=30&max_latitude=40",
    "/v1/mondial_geo/desert_coordinates?name_count=1",
    "/v1/mondial_geo/capital_province_highest_density?max_population=80000"
]
