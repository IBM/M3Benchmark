from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/world_development_indicators/world_development_indicators.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of countries based on lending category and other groups
@app.get("/v1/world_development_indicators/count_countries_lending_other_groups", operation_id="get_count_countries_lending_other_groups", summary="Retrieves the total number of countries that belong to a specific lending category and other groups. The lending category and other groups are used to filter the countries and determine the count.")
async def get_count_countries_lending_other_groups(lending_category: str = Query(..., description="Lending category of the country"), other_groups: str = Query(..., description="Other groups of the country")):
    cursor.execute("SELECT COUNT(CountryCode) FROM Country WHERE LendingCategory = ? AND OtherGroups = ?", (lending_category, other_groups))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the short name and external debt reporting status of countries based on lending category
@app.get("/v1/world_development_indicators/country_short_name_debt_status", operation_id="get_country_short_name_debt_status", summary="Retrieves the short name and external debt reporting status of countries that belong to a specified lending category. The lending category is used to filter the results.")
async def get_country_short_name_debt_status(lending_category: str = Query(..., description="Lending category of the country")):
    cursor.execute("SELECT ShortName, ExternalDebtReportingStatus FROM Country WHERE LendingCategory = ?", (lending_category,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the description of country notes based on short name and series code
@app.get("/v1/world_development_indicators/country_notes_description", operation_id="get_country_notes_description", summary="Retrieves the description of country notes for a specific country and series code. The operation uses the provided short name of the country and series code to fetch the corresponding description from the database.")
async def get_country_notes_description(short_name: str = Query(..., description="Short name of the country"), series_code: str = Query(..., description="Series code of the country notes")):
    cursor.execute("SELECT T2.Description FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = ? AND T2.Seriescode = ?", (short_name, series_code))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": result}

# Endpoint to get the short name and description of country notes based on region and series code
@app.get("/v1/world_development_indicators/country_short_name_notes_description", operation_id="get_country_short_name_notes_description", summary="Retrieves the short name and description of country notes for a specific region and series code. The operation filters the data based on the provided region and series code, returning the corresponding short name and description of the country notes.")
async def get_country_short_name_notes_description(region: str = Query(..., description="Region of the country"), series_code: str = Query(..., description="Series code of the country notes")):
    cursor.execute("SELECT T1.SHORTNAME, T2.Description FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.Region = ? AND T2.Seriescode = ?", (region, series_code))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the count of countries based on series code and income group
@app.get("/v1/world_development_indicators/count_countries_series_income", operation_id="get_count_countries_series_income", summary="Retrieves the total number of countries that match a specified series code and income group. The series code and income group are used to filter the countries, providing a count of those that meet the specified criteria.")
async def get_count_countries_series_income(series_code: str = Query(..., description="Series code of the country notes"), income_group: str = Query(..., description="Income group of the country")):
    cursor.execute("SELECT COUNT(T1.Countrycode) FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Seriescode = ? AND T1.IncomeGroup = ?", (series_code, income_group))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the description of country notes based on lending category and series code
@app.get("/v1/world_development_indicators/country_notes_description_lending", operation_id="get_country_notes_description_lending", summary="Retrieves the description of country notes based on the specified lending category and series code. This operation fetches data from the Country and CountryNotes tables, filtering results by the provided lending category and series code. The response includes the description of the country notes that match the given criteria.")
async def get_country_notes_description_lending(lending_category: str = Query(..., description="Lending category of the country"), series_code: str = Query(..., description="Series code of the country notes")):
    cursor.execute("SELECT T2.Description FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.LendingCategory = ? AND T2.Seriescode = ?", (lending_category, series_code))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": result}

# Endpoint to get the count of countries based on lending category, series code, and income group
@app.get("/v1/world_development_indicators/count_countries_lending_series_income", operation_id="get_count_countries_lending_series_income", summary="Retrieves the number of countries that match the specified lending category, series code, and income group. This operation considers the lending category of the country, the series code of the country notes, and the income group to determine the count.")
async def get_count_countries_lending_series_income(lending_category: str = Query(..., description="Lending category of the country"), series_code: str = Query(..., description="Series code of the country notes"), income_group: str = Query(..., description="Income group of the country")):
    cursor.execute("SELECT COUNT(T1.Countrycode) FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.LendingCategory = ? AND T2.Seriescode = ? AND IncomeGroup = ?", (lending_category, series_code, income_group))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of countries based on income group, currency unit, and series code
@app.get("/v1/world_development_indicators/count_countries_income_currency_series", operation_id="get_count_countries_income_currency_series", summary="Retrieve the number of countries that match the specified income group, currency unit, and series code. This operation provides a count of countries based on the provided criteria, offering insights into the distribution of countries across different income levels, currency units, and series codes.")
async def get_count_countries_income_currency_series(income_group: str = Query(..., description="Income group of the country"), currency_unit: str = Query(..., description="Currency unit of the country"), series_code: str = Query(..., description="Series code of the country notes")):
    cursor.execute("SELECT COUNT(T1.Countrycode) FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.IncomeGroup = ? AND T1.CurrencyUnit = ? AND T2.Seriescode = ?", (income_group, currency_unit, series_code))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the long name of countries based on description and series code
@app.get("/v1/world_development_indicators/country_long_name_description_series", operation_id="get_country_long_name_description_series", summary="Retrieves the full name of a country based on the provided description and series code of its notes. This operation utilizes the description and series code to identify the country and return its long name.")
async def get_country_long_name_description_series(description: str = Query(..., description="Description of the country notes"), series_code: str = Query(..., description="Series code of the country notes")):
    cursor.execute("SELECT T1.LongName FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ? AND T2.Seriescode = ?", (description, series_code))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the description of footnotes based on short name, series code, and year
@app.get("/v1/world_development_indicators/footnotes_description", operation_id="get_footnotes_description", summary="Retrieve the description of footnotes associated with a specific country, series code, and year. The endpoint fetches the description from the FootNotes table, joining it with the Country table based on the provided short name, series code, and year. The short name identifies the country, the series code specifies the footnotes, and the year is represented in the 'YRYYYY' format.")
async def get_footnotes_description(short_name: str = Query(..., description="Short name of the country"), series_code: str = Query(..., description="Series code of the footnotes"), year: str = Query(..., description="Year in 'YRYYYY' format")):
    cursor.execute("SELECT T2.Description FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = ? AND T2.Seriescode = ? AND T2.Year = ?", (short_name, series_code, year))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": result}

# Endpoint to get the years for a specific country and series code
@app.get("/v1/world_development_indicators/years_by_country_series", operation_id="get_years_by_country_series", summary="Retrieve the years associated with a specific country and series code. This operation requires the short name of the country and the series code as input parameters. The output will be a list of years that correspond to the provided country and series code.")
async def get_years_by_country_series(short_name: str = Query(..., description="Short name of the country"), series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT T2.Year FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = ? AND T2.Seriescode = ?", (short_name, series_code))
    result = cursor.fetchall()
    if not result:
        return {"years": []}
    return {"years": [row[0] for row in result]}

# Endpoint to get the short names of countries based on description, series code, and year pattern
@app.get("/v1/world_development_indicators/country_short_names", operation_id="get_country_short_names", summary="Retrieve the short names of countries that match the provided description, series code, and year pattern. This operation filters countries based on their associated footnotes, series codes, and years, returning only the short names of those that meet the specified criteria.")
async def get_country_short_names(description: str = Query(..., description="Description of the footnote"), series_code: str = Query(..., description="Series code"), year_pattern: str = Query(..., description="Year pattern (e.g., '%2002%')")):
    cursor.execute("SELECT T1.SHORTNAME FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ? AND T2.Seriescode = ? AND T2.Year LIKE ?", (description, series_code, year_pattern))
    result = cursor.fetchall()
    if not result:
        return {"short_names": []}
    return {"short_names": [row[0] for row in result]}

# Endpoint to get the count of series codes for a specific country and year
@app.get("/v1/world_development_indicators/count_series_codes", operation_id="get_count_series_codes", summary="Retrieves the total number of series codes associated with a specific country and year. The operation uses the provided short name of the country and the year to filter the data and calculate the count.")
async def get_count_series_codes(short_name: str = Query(..., description="Short name of the country"), year: str = Query(..., description="Year (e.g., 'YR2002')")):
    cursor.execute("SELECT COUNT(T2.SeriesCode) FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = ? AND T2.Year = ?", (short_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of years for a specific country and series code
@app.get("/v1/world_development_indicators/count_years_by_country_series", operation_id="get_count_years_by_country_series", summary="Retrieves the total number of years for which data is available for a specific country and series code. The operation requires the short name of the country and the series code as input parameters. The short name identifies the country, and the series code specifies the type of data to be considered.")
async def get_count_years_by_country_series(short_name: str = Query(..., description="Short name of the country"), series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT COUNT(T2.Year) FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.ShortName = ? AND T2.Seriescode = ?", (short_name, series_code))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average value for a specific country and indicator name pattern
@app.get("/v1/world_development_indicators/avg_value_by_country_indicator", operation_id="get_avg_value_by_country_indicator", summary="Retrieves the average value for a specific country and a given indicator name pattern. The operation calculates the average value from the Indicators table, filtering by the provided Alpha-2 code of the country and a pattern in the indicator name. This endpoint is useful for obtaining a summary statistic for a particular country and a set of related indicators.")
async def get_avg_value_by_country_indicator(alpha2_code: str = Query(..., description="Alpha-2 code of the country"), indicator_name_pattern: str = Query(..., description="Indicator name pattern (e.g., 'adolescent fertility rate%')")):
    cursor.execute("SELECT AVG(T2.Value) FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.Alpha2Code = ? AND T2.IndicatorName LIKE ?", (alpha2_code, indicator_name_pattern))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get distinct special notes for the country with the highest indicator value
@app.get("/v1/world_development_indicators/special_notes_highest_indicator", operation_id="get_special_notes_highest_indicator", summary="Retrieves unique special notes for the country that has the highest value for a specified indicator. The indicator name pattern is used to filter the results. For example, providing 'Adolescent fertility rate%' as the pattern will return the special notes for the country with the highest adolescent fertility rate.")
async def get_special_notes_highest_indicator(indicator_name_pattern: str = Query(..., description="Indicator name pattern (e.g., 'Adolescent fertility rate%')")):
    cursor.execute("SELECT DISTINCT T1.SpecialNotes FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Value = ( SELECT Value FROM Indicators WHERE IndicatorName LIKE ? ORDER BY Value DESC LIMIT 1 )", (indicator_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"special_notes": []}
    return {"special_notes": [row[0] for row in result]}

# Endpoint to get country codes and alpha-2 codes based on region and income group
@app.get("/v1/world_development_indicators/country_codes_by_region_income", operation_id="get_country_codes_by_region_income", summary="Retrieves the country codes and corresponding alpha-2 codes for countries that belong to a specific region and income group. The operation filters the countries based on the provided region and income group parameters.")
async def get_country_codes_by_region_income(region: str = Query(..., description="Region of the country"), income_group: str = Query(..., description="Income group of the country")):
    cursor.execute("SELECT CountryCode, Alpha2Code FROM Country WHERE Region = ? AND IncomeGroup = ?", (region, income_group))
    result = cursor.fetchall()
    if not result:
        return {"country_codes": []}
    return {"country_codes": [{"CountryCode": row[0], "Alpha2Code": row[1]} for row in result]}

# Endpoint to get long names and alpha-2 codes based on latest trade and water withdrawal data years
@app.get("/v1/world_development_indicators/country_names_by_data_years", operation_id="get_country_names_by_data_years", summary="Retrieves the long names and alpha-2 codes of countries based on the specified years of the latest trade and water withdrawal data. This operation allows you to obtain country information that aligns with your desired data years, providing a tailored dataset for further analysis or integration.")
async def get_country_names_by_data_years(latest_trade_data: int = Query(..., description="Year of the latest trade data"), latest_water_withdrawal_data: int = Query(..., description="Year of the latest water withdrawal data")):
    cursor.execute("SELECT LongName, Alpha2Code FROM Country WHERE LatestTradeData = ? AND LatestWaterWithdrawalData = ?", (latest_trade_data, latest_water_withdrawal_data))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [{"LongName": row[0], "Alpha2Code": row[1]} for row in result]}

# Endpoint to get the average value of an indicator for a specific country and year range
@app.get("/v1/world_development_indicators/avg_indicator_value", operation_id="get_avg_indicator_value", summary="Retrieves the average value of a specified indicator for a given country within a defined year range. The indicator used in this operation is the 'Adjusted net enrolment rate, primary, both sexes (%)'. The input parameters include the country name, minimum year, maximum year, and the name of the indicator.")
async def get_avg_indicator_value(country_name: str = Query(..., description="Name of the country"), min_year: int = Query(..., description="Minimum year"), max_year: int = Query(..., description="Maximum year"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT CAST(SUM(Value) AS REAL) / COUNT(CountryCode) FROM Indicators WHERE CountryName = ? AND Year > ? AND Year < ? AND IndicatorName = ?", (country_name, min_year, max_year, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get indicator names and aggregation methods based on topic
@app.get("/v1/world_development_indicators/indicator_aggregation_by_topic", operation_id="get_indicator_aggregation_by_topic", summary="Retrieves a list of indicator names and their respective aggregation methods associated with a specific topic. The topic is provided as an input parameter, allowing the user to filter the results based on their desired topic of interest.")
async def get_indicator_aggregation_by_topic(topic: str = Query(..., description="Topic of the series")):
    cursor.execute("SELECT IndicatorName, AggregationMethod FROM Series WHERE Topic = ?", (topic,))
    result = cursor.fetchall()
    if not result:
        return {"indicators": []}
    return {"indicators": [{"IndicatorName": row[0], "AggregationMethod": row[1]} for row in result]}

# Endpoint to get series codes based on topic and license type
@app.get("/v1/world_development_indicators/series_codes_by_topic_license", operation_id="get_series_codes", summary="Retrieves a list of series codes that match a specified topic and license type. The topic and license type are used to filter the series codes, providing a targeted set of results.")
async def get_series_codes(topic: str = Query(..., description="Topic of the series"), license_type: str = Query(..., description="License type of the series")):
    cursor.execute("SELECT SeriesCode FROM Series WHERE Topic = ? AND LicenseType = ?", (topic, license_type))
    result = cursor.fetchall()
    if not result:
        return {"series_codes": []}
    return {"series_codes": [row[0] for row in result]}

# Endpoint to get the count of countries based on region, indicator name, year, and value
@app.get("/v1/world_development_indicators/count_countries_by_region_indicator_year_value", operation_id="get_count_countries", summary="Retrieve the number of countries that meet the specified criteria: a particular region, indicator name, year, and value. The count is based on the intersection of countries that satisfy all conditions.")
async def get_count_countries(region: str = Query(..., description="Region of the country"), indicator_name: str = Query(..., description="Name of the indicator"), year: int = Query(..., description="Year of the indicator"), value: float = Query(..., description="Value of the indicator")):
    cursor.execute("SELECT COUNT(T2.CountryCode) FROM Indicators AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Region = ? AND T1.IndicatorName = ? AND T1.Year = ? AND T1.Value > ?", (region, indicator_name, year, value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct country codes, years, and values based on region, indicator name, and year range
@app.get("/v1/world_development_indicators/distinct_country_codes_years_values", operation_id="get_distinct_country_codes_years_values", summary="Retrieve the top three distinct combinations of country codes, years, and corresponding values for a specified region and indicator, within a given year range. The results are ordered by value in descending order.")
async def get_distinct_country_codes_years_values(region: str = Query(..., description="Region of the country"), indicator_name: str = Query(..., description="Name of the indicator"), min_year: int = Query(..., description="Minimum year of the indicator"), max_year: int = Query(..., description="Maximum year of the indicator")):
    cursor.execute("SELECT DISTINCT T1.CountryCode, T1.Year, T1.Value FROM Indicators AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Region = ? AND T1.IndicatorName = ? AND T1.Year > ? AND T1.Year < ? ORDER BY T1.Value DESC LIMIT 3", (region, indicator_name, min_year, max_year))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": [{"country_code": row[0], "year": row[1], "value": row[2]} for row in result]}

# Endpoint to get series descriptions based on series code, topic, and year
@app.get("/v1/world_development_indicators/series_descriptions", operation_id="get_series_descriptions", summary="Retrieve the descriptions of a specific series based on its code, topic, and year. The series code, topic, and year are used to filter the series and its corresponding notes, providing a focused and relevant description.")
async def get_series_descriptions(series_code: str = Query(..., description="Series code"), topic: str = Query(..., description="Topic of the series"), year: str = Query(..., description="Year of the series note")):
    cursor.execute("SELECT T2.Description FROM Series AS T1 INNER JOIN SeriesNotes AS T2 ON T1.SeriesCode = T2.Seriescode WHERE T1.SeriesCode = ? AND T1.Topic = ? AND T2.Year = ?", (series_code, topic, year))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct footnote descriptions based on year and country short name
@app.get("/v1/world_development_indicators/distinct_footnote_descriptions", operation_id="get_distinct_footnote_descriptions", summary="Retrieves unique footnote descriptions for a given year and country. The operation filters footnotes based on the specified year and country short name, ensuring that only distinct descriptions are returned.")
async def get_distinct_footnote_descriptions(year: str = Query(..., description="Year of the footnote"), short_name: str = Query(..., description="Short name of the country")):
    cursor.execute("SELECT DISTINCT T1.Description FROM FootNotes AS T1 INNER JOIN Country AS T2 ON T1.Countrycode = T2.CountryCode WHERE T1.Year = ? AND T2.ShortName = ?", (year, short_name))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct footnote descriptions based on year, country short name, and series code
@app.get("/v1/world_development_indicators/distinct_footnote_descriptions_by_year_country_series", operation_id="get_distinct_footnote_descriptions_by_year_country_series", summary="Retrieve unique footnote descriptions for a given year, country, and series. This operation filters footnotes based on the provided year, country short name, and series code, returning only distinct descriptions.")
async def get_distinct_footnote_descriptions_by_year_country_series(year: str = Query(..., description="Year of the footnote"), short_name: str = Query(..., description="Short name of the country"), series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT DISTINCT T1.Description FROM FootNotes AS T1 INNER JOIN Country AS T2 ON T1.Countrycode = T2.CountryCode WHERE T1.Year = ? AND T2.ShortName = ? AND T1.Seriescode = ?", (year, short_name, series_code))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct WB2 codes and lending categories based on description and lending category
@app.get("/v1/world_development_indicators/distinct_wb2_codes_lending_categories", operation_id="get_distinct_wb2_codes_lending_categories", summary="Retrieves a unique list of World Bank (WB2) codes and their corresponding lending categories for countries that match a specific description and have a defined lending category. The provided description filters the countries based on their associated notes, while the lending category ensures that only countries with a valid lending category are included in the results.")
async def get_distinct_wb2_codes_lending_categories(description: str = Query(..., description="Description of the country note"), lending_category: str = Query(..., description="Lending category of the country")):
    cursor.execute("SELECT DISTINCT T1.Wb2code, T1.LendingCategory FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ? AND T1.LendingCategory != ?", (description, lending_category))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": [{"wb2_code": row[0], "lending_category": row[1]} for row in result]}

# Endpoint to get series details based on year, country name, indicator name, and value
@app.get("/v1/world_development_indicators/series_details", operation_id="get_series_details", summary="Retrieves detailed information about a specific series based on a given year, country, indicator, and its corresponding value. The data returned includes the topic, series code, and license type associated with the series.")
async def get_series_details(year: int = Query(..., description="Year of the indicator"), country_name: str = Query(..., description="Name of the country"), indicator_name: str = Query(..., description="Name of the indicator"), value: float = Query(..., description="Value of the indicator")):
    cursor.execute("SELECT T2.Topic, T2.Seriescode, T2.LicenseType FROM Indicators AS T1 INNER JOIN Series AS T2 ON T1.IndicatorName = T2.IndicatorName WHERE T1.Year = ? AND T1.CountryName = ? AND T1.IndicatorName = ? AND T1.Value = ?", (year, country_name, indicator_name, value))
    result = cursor.fetchall()
    if not result:
        return {"series_details": []}
    return {"series_details": [{"topic": row[0], "series_code": row[1], "license_type": row[2]} for row in result]}

# Endpoint to get the count of countries based on indicator name, external debt reporting status, and value
@app.get("/v1/world_development_indicators/count_countries_by_indicator_status_value", operation_id="get_count_countries_by_indicator_status_value", summary="Retrieve the number of countries that meet the specified criteria: having a particular indicator name, external debt reporting status, and an indicator value greater than the provided threshold.")
async def get_count_countries_by_indicator_status_value(indicator_name: str = Query(..., description="Name of the indicator"), external_debt_reporting_status: str = Query(..., description="External debt reporting status of the country"), value: float = Query(..., description="Value of the indicator")):
    cursor.execute("SELECT COUNT(T1.CountryCode) FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.IndicatorName = ? AND T1.ExternalDebtReportingStatus = ? AND T2.Value > ?", (indicator_name, external_debt_reporting_status, value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get country names based on year, indicator name, and value range
@app.get("/v1/world_development_indicators/country_names_by_year_indicator_value_range", operation_id="get_country_names", summary="Retrieves the names of countries that meet the specified criteria for a given year and indicator. The criteria include a value range for the indicator, which allows for filtering countries based on their indicator values. The indicator in question is the fertility rate, expressed as the total number of births per woman.")
async def get_country_names(year: int = Query(..., description="Year of the indicator"), indicator_name: str = Query(..., description="Name of the indicator"), min_value: float = Query(..., description="Minimum value of the indicator"), max_value: float = Query(..., description="Maximum value of the indicator")):
    cursor.execute("SELECT CountryName FROM Indicators WHERE Year = ? AND IndicatorName = ? AND value >= ? AND Value <= ?", (year, indicator_name, min_value, max_value))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get table names and source of most recent income and expenditure data based on region and income group
@app.get("/v1/world_development_indicators/table_names_and_source_by_region_income_group", operation_id="get_table_names_and_source", summary="Retrieves the names of tables and the source of the most recent income and expenditure data for countries in a specified region and income group. The operation filters the data based on the provided region and income group parameters.")
async def get_table_names_and_source(region: str = Query(..., description="Region of the country"), income_group: str = Query(..., description="Income group of the country")):
    cursor.execute("SELECT TableName, SourceOfMostRecentIncomeAndExpenditureData FROM Country WHERE Region = ? AND IncomeGroup = ?", (region, income_group))
    result = cursor.fetchall()
    if not result:
        return {"table_names_and_source": []}
    return {"table_names_and_source": [{"table_name": row[0], "source": row[1]} for row in result]}

# Endpoint to get distinct sources based on region and indicator name
@app.get("/v1/world_development_indicators/distinct_sources_by_region_indicator", operation_id="get_distinct_sources", summary="Retrieve a unique list of data sources for a specific region and the indicator 'Children out of school, primary'. The operation filters data based on the provided region and returns distinct sources associated with the given indicator.")
async def get_distinct_sources(region: str = Query(..., description="Region of the country"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT DISTINCT T2.Source FROM Footnotes AS T1 INNER JOIN Series AS T2 ON T1.Seriescode = T2.SeriesCode INNER JOIN Country AS T3 ON T1.Countrycode = T3.CountryCode WHERE T3.Region = ? AND T2.IndicatorName = ?", (region, indicator_name))
    result = cursor.fetchall()
    if not result:
        return {"sources": []}
    return {"sources": [row[0] for row in result]}

# Endpoint to get sources based on year pattern and indicator name
@app.get("/v1/world_development_indicators/sources_by_year_pattern_indicator", operation_id="get_sources_by_year_pattern", summary="Retrieves the sources associated with a specific year pattern and indicator name. The year pattern is used to filter the data, and the indicator name is used to identify the relevant data series. The sources are obtained by joining multiple tables, including CountryNotes, Series, Country, and SeriesNotes, based on their respective codes. The resulting sources are then filtered based on the provided year pattern and indicator name.")
async def get_sources_by_year_pattern(year_pattern: str = Query(..., description="Year pattern (e.g., '%2002%')"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT T2.Source FROM CountryNotes AS T1 INNER JOIN Series AS T2 ON T1.Seriescode = T2.SeriesCode INNER JOIN Country AS T3 ON T1.Countrycode = T3.CountryCode INNER JOIN SeriesNotes AS T4 ON T2.SeriesCode = T4.Seriescode WHERE T4.Year LIKE ? AND T2.IndicatorName = ?", (year_pattern, indicator_name))
    result = cursor.fetchall()
    if not result:
        return {"sources": []}
    return {"sources": [row[0] for row in result]}

# Endpoint to get distinct descriptions based on region and indicator name
@app.get("/v1/world_development_indicators/distinct_descriptions_by_region_indicator", operation_id="get_distinct_descriptions", summary="Retrieve unique descriptions for a specific region and indicator. This operation returns a list of distinct descriptions associated with the given region and the indicator 'Out-of-school children of primary school age, both sexes (number)'. The region and indicator name are provided as input parameters.")
async def get_distinct_descriptions(region: str = Query(..., description="Region of the country"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT DISTINCT T3.Description FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode INNER JOIN CountryNotes AS T3 ON T2.CountryCode = T3.Countrycode WHERE T1.Region = ? AND T2.IndicatorName = ?", (region, indicator_name))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct country codes and values based on latest trade data, indicator name pattern, year, and minimum value
@app.get("/v1/world_development_indicators/country_codes_values_by_trade_data_indicator_year_min_value", operation_id="get_country_codes_values", summary="Retrieve unique country codes and their corresponding values for a specific trade data, indicator name pattern, year, and minimum value. The operation filters data based on the latest trade data year, a partial match for the indicator name, a specific year, and a minimum value threshold. The results are ordered in ascending order based on the indicator value.")
async def get_country_codes_values(latest_trade_data: int = Query(..., description="Latest trade data year"), indicator_name_pattern: str = Query(..., description="Indicator name pattern (e.g., 'GDP growth (annual %)')"), year: int = Query(..., description="Year of the indicator"), min_value: float = Query(..., description="Minimum value of the indicator")):
    cursor.execute("SELECT DISTINCT T1.CountryCode, T2.Value FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.LatestTradeData = ? AND T2.IndicatorName LIKE ? AND T2.year = ? AND T2.Value > ? ORDER BY T2.Value ASC", (latest_trade_data, indicator_name_pattern, year, min_value))
    result = cursor.fetchall()
    if not result:
        return {"country_codes_values": []}
    return {"country_codes_values": [{"country_code": row[0], "value": row[1]} for row in result]}

# Endpoint to get distinct country codes and descriptions based on indicator name, minimum value, year, and limit
@app.get("/v1/world_development_indicators/country_codes_descriptions_by_indicator_min_value_year_limit", operation_id="get_country_codes_descriptions", summary="Get distinct country codes and descriptions for a specific indicator name, minimum value, year, and limit")
async def get_country_codes_descriptions(indicator_name: str = Query(..., description="Name of the indicator"), min_value: float = Query(..., description="Minimum value of the indicator"), year: int = Query(..., description="Year of the indicator"), limit: int = Query(..., description="Limit of the number of results")):
    cursor.execute("SELECT DISTINCT T1.CountryCode, T3.Description FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode INNER JOIN CountryNotes AS T3 ON T1.CountryCode = T3.Countrycode WHERE T2.IndicatorName = ? AND T2.Value > ? AND T2.year = ? ORDER BY T2.Value DESC LIMIT ?", (indicator_name, min_value, year, limit))
    result = cursor.fetchall()
    if not result:
        return {"country_codes_descriptions": []}
    return {"country_codes_descriptions": [{"country_code": row[0], "description": row[1]} for row in result]}

# Endpoint to get short names of countries based on latest trade data year
@app.get("/v1/world_development_indicators/short_names_by_latest_trade_data", operation_id="get_short_names", summary="Retrieves the short names of countries that have trade data from a year later than the provided input. This operation allows users to filter countries based on their most recent trade data, providing a concise list of countries that meet the specified criteria.")
async def get_short_names(latest_trade_data: int = Query(..., description="Latest trade data year")):
    cursor.execute("SELECT ShortName FROM Country WHERE LatestTradeData > ?", (latest_trade_data,))
    result = cursor.fetchall()
    if not result:
        return {"short_names": []}
    return {"short_names": [row[0] for row in result]}

# Endpoint to get the percentage of countries in a specific region with a specific system of trade
@app.get("/v1/world_development_indicators/percentage_countries_by_region_system_of_trade", operation_id="get_percentage_countries", summary="Retrieves the percentage of countries in a specified region that employ a particular system of trade. The calculation is based on the total number of countries in the database.")
async def get_percentage_countries(region: str = Query(..., description="Region of the country"), system_of_trade: str = Query(..., description="System of trade")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Region = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(CountryCode) FROM Country WHERE SystemOfTrade = ?", (region, system_of_trade))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average value of an indicator for a specific region
@app.get("/v1/world_development_indicators/average_indicator_value_by_region", operation_id="get_average_indicator_value_by_region", summary="Retrieves the average value of a specified indicator for a given region. The operation calculates the sum of the indicator values for all countries within the region and divides it by the total number of countries in the region. The region and indicator are provided as input parameters.")
async def get_average_indicator_value_by_region(region: str = Query(..., description="Region name"), indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT CAST(SUM(T2.Value) AS REAL) / COUNT(T1.CountryCode) FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Region = ? AND T2.IndicatorName = ?", (region, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"average_value": []}
    return {"average_value": result[0]}

# Endpoint to get the country code with the lowest value for a specific indicator in a given income group
@app.get("/v1/world_development_indicators/country_with_lowest_indicator_value", operation_id="get_country_with_lowest_indicator_value", summary="Retrieve the country code with the lowest value for a specified indicator within a given income group. The income group and indicator name are required as input parameters to filter the data and determine the country with the lowest value for the specified indicator.")
async def get_country_with_lowest_indicator_value(income_group: str = Query(..., description="Income group"), indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT T1.CountryCode FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IncomeGroup = ? AND T2.IndicatorName = ? ORDER BY T2.Value ASC LIMIT 1", (income_group, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"country_code": []}
    return {"country_code": result[0]}

# Endpoint to get the minimum value of an indicator for a specific group
@app.get("/v1/world_development_indicators/min_indicator_value_by_group", operation_id="get_min_indicator_value_by_group", summary="Retrieves the minimum value of the specified indicator for a given group. The group is identified by the 'other_groups' parameter, and the indicator is specified using the 'indicator_name' parameter. This operation returns the lowest value of the selected indicator within the specified group.")
async def get_min_indicator_value_by_group(other_groups: str = Query(..., description="Other groups"), indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT MIN(T2.Value) FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.OtherGroups = ? AND T2.IndicatorName = ?", (other_groups, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"min_value": []}
    return {"min_value": result[0]}

# Endpoint to get distinct indicator names for a specific year and topic
@app.get("/v1/world_development_indicators/distinct_indicator_names", operation_id="get_distinct_indicator_names", summary="Retrieve a unique list of indicator names associated with a given year and topic. The operation filters the data based on the provided year and topic, ensuring that only relevant and distinct indicator names are returned.")
async def get_distinct_indicator_names(year: str = Query(..., description="Year in 'YRXXXX' format"), topic: str = Query(..., description="Topic name")):
    cursor.execute("SELECT DISTINCT T2.IndicatorName FROM Footnotes AS T1 INNER JOIN Series AS T2 ON T1.Seriescode = T2.SeriesCode WHERE T1.Year = ? AND T2.Topic = ?", (year, topic))
    result = cursor.fetchall()
    if not result:
        return {"indicator_names": []}
    return {"indicator_names": [row[0] for row in result]}

# Endpoint to get the count of distinct series codes for specific years, periodicity, and aggregation method
@app.get("/v1/world_development_indicators/count_distinct_series_codes", operation_id="get_count_distinct_series_codes", summary="Retrieve the number of unique series codes that meet the specified criteria. The criteria include a range of years, periodicity, and aggregation method. This operation is useful for understanding the distribution of series codes across different time periods and aggregation methods.")
async def get_count_distinct_series_codes(year1: str = Query(..., description="Year in 'YRXXXX' format"), year2: str = Query(..., description="Year in 'YRXXXX' format"), year3: str = Query(..., description="Year in 'YRXXXX' format"), periodicity: str = Query(..., description="Periodicity"), aggregation_method: str = Query(..., description="Aggregation method")):
    cursor.execute("SELECT COUNT(DISTINCT T2.SeriesCode) FROM Footnotes AS T1 INNER JOIN Series AS T2 ON T1.Seriescode = T2.SeriesCode WHERE T1.Year IN (?, ?, ?) AND T2.Periodicity = ? AND T2.AggregationMethod = ?", (year1, year2, year3, periodicity, aggregation_method))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get series codes and sources for a specific year and source pattern
@app.get("/v1/world_development_indicators/series_codes_and_sources", operation_id="get_series_codes_and_sources", summary="Retrieves a list of series codes and their corresponding sources that match the specified year and source patterns. The year pattern should be provided in the format '\%XXXX\%', where 'XXXX' is the year of interest. The source pattern should be provided in the format '\%Source\%', where 'Source' is the desired source name. The operation returns a set of series codes and their associated sources that meet the provided criteria.")
async def get_series_codes_and_sources(year_pattern: str = Query(..., description="Year pattern in '%XXXX%' format"), source_pattern: str = Query(..., description="Source pattern in '%Source%' format")):
    cursor.execute("SELECT T1.Seriescode, T2.Source FROM Footnotes AS T1 INNER JOIN Series AS T2 ON T1.Seriescode = T2.SeriesCode WHERE T1.Year LIKE ? AND T2.Source LIKE ?", (year_pattern, source_pattern))
    result = cursor.fetchall()
    if not result:
        return {"series_codes_and_sources": []}
    return {"series_codes_and_sources": [{"series_code": row[0], "source": row[1]} for row in result]}

# Endpoint to get the percentage of countries in a region with an indicator value above a threshold
@app.get("/v1/world_development_indicators/percentage_above_threshold", operation_id="get_percentage_above_threshold", summary="Retrieves the percentage of countries in a specified region where the female life expectancy at birth surpasses a given threshold. The calculation is based on the provided region and threshold, and the result is expressed as a percentage.")
async def get_percentage_above_threshold(region: str = Query(..., description="Region name"), indicator_name: str = Query(..., description="Indicator name"), threshold: int = Query(..., description="Threshold value")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.value > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.CountryCode) FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Region = ? AND T2.IndicatorName = ?", (threshold, region, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the country with the highest value for a specific indicator within a year range
@app.get("/v1/world_development_indicators/country_with_highest_indicator_value", operation_id="get_country_with_highest_indicator_value", summary="Retrieves the country with the highest death rate within a specified year range. The operation filters data by the provided start and end years, and the specific indicator 'Death rate, crude (per 1,000 people)'. The result is ordered in descending order based on the death rate value, and only the top country is returned.")
async def get_country_with_highest_indicator_value(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year"), indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT CountryName FROM Indicators WHERE Year BETWEEN ? AND ? AND IndicatorName = ? ORDER BY Value DESC LIMIT 1", (start_year, end_year, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get indicator names for a specific country, year, and value threshold
@app.get("/v1/world_development_indicators/indicator_names_by_country_year_value", operation_id="get_indicator_names_by_country_year_value", summary="Retrieves the names of indicators for a given country and year, where the indicator value surpasses a specified threshold. This operation is useful for identifying key development indicators that meet or exceed a certain value in a specific year for a particular country.")
async def get_indicator_names_by_country_year_value(country_name: str = Query(..., description="Country name"), year: int = Query(..., description="Year"), value_threshold: int = Query(..., description="Value threshold")):
    cursor.execute("SELECT IndicatorName FROM Indicators WHERE CountryName = ? AND Year = ? AND Value > ?", (country_name, year, value_threshold))
    result = cursor.fetchall()
    if not result:
        return {"indicator_names": []}
    return {"indicator_names": [row[0] for row in result]}

# Endpoint to get the country with the highest value for a specific indicator
@app.get("/v1/world_development_indicators/country_with_highest_indicator_value_single", operation_id="get_country_with_highest_indicator_value_single", summary="Retrieves the name of the country with the highest value for a specific development indicator. The operation requires the name of the indicator as input and returns the country name that has the highest value for that indicator.")
async def get_country_with_highest_indicator_value_single(indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT CountryName FROM Indicators WHERE IndicatorName = ? ORDER BY Value DESC LIMIT 1", (indicator_name,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get distinct indicator names based on year range, license type, and value
@app.get("/v1/world_development_indicators/distinct_indicator_names_year_range", operation_id="get_distinct_indicator_names_year_range", summary="Retrieve a unique list of indicator names that fall within a specified year range, adhering to a particular license type and having values less than a given amount. This operation is useful for identifying distinct indicators that meet specific criteria, providing a concise overview of relevant data.")
async def get_distinct_indicator_names_year_range(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), license_type: str = Query(..., description="License type of the series"), value: float = Query(..., description="Maximum value of the indicator")):
    cursor.execute("SELECT DISTINCT T1.IndicatorName FROM Indicators AS T1 INNER JOIN Series AS T2 ON T1.IndicatorName = T2.IndicatorName WHERE T1.Year >= ? AND T1.Year < ? AND T2.LicenseType = ? AND T1.Value < ?", (start_year, end_year, license_type, value))
    result = cursor.fetchall()
    if not result:
        return {"indicator_names": []}
    return {"indicator_names": [row[0] for row in result]}

# Endpoint to get the top country and indicator name based on a specific topic
@app.get("/v1/world_development_indicators/top_country_indicator", operation_id="get_top_country_indicator", summary="Retrieves the country and indicator with the highest value for a given topic. The topic is specified as an input parameter.")
async def get_top_country_indicator(topic: str = Query(..., description="Topic of the series")):
    cursor.execute("SELECT T1.CountryName, T1.IndicatorName FROM Indicators AS T1 INNER JOIN Series AS T2 ON T1.IndicatorName = T2.IndicatorName WHERE T2.Topic = ? ORDER BY T1.Value DESC LIMIT 1", (topic,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": [], "indicator_name": []}
    return {"country_name": result[0], "indicator_name": result[1]}

# Endpoint to get the minimum value of indicators based on aggregation method
@app.get("/v1/world_development_indicators/min_indicator_value", operation_id="get_min_indicator_value", summary="Retrieves the minimum value of indicators based on a specified aggregation method. The aggregation method is used to filter the series and determine the minimum value for each indicator.")
async def get_min_indicator_value(aggregation_method: str = Query(..., description="Aggregation method of the series")):
    cursor.execute("SELECT T1.IndicatorName, MIN(T1.Value) FROM Indicators AS T1 INNER JOIN Series AS T2 ON T1.IndicatorName = T2.IndicatorName WHERE T2.AggregationMethod = ?", (aggregation_method,))
    result = cursor.fetchone()
    if not result:
        return {"indicator_name": [], "min_value": []}
    return {"indicator_name": result[0], "min_value": result[1]}

# Endpoint to get indicator names based on country, year, and periodicity
@app.get("/v1/world_development_indicators/indicator_names_country_year", operation_id="get_indicator_names_country_year", summary="Retrieves the names of indicators for a specified country, year, and periodicity. The operation filters indicators based on the provided country name, year, and periodicity, and returns a list of matching indicator names.")
async def get_indicator_names_country_year(country_name: str = Query(..., description="Name of the country"), year: int = Query(..., description="Year of the indicator"), periodicity: str = Query(..., description="Periodicity of the series")):
    cursor.execute("SELECT T1.IndicatorName FROM Indicators AS T1 INNER JOIN Series AS T2 ON T1.IndicatorName = T2.IndicatorName WHERE T1.CountryName = ? AND T1.Year = ? AND T2.Periodicity = ?", (country_name, year, periodicity))
    result = cursor.fetchall()
    if not result:
        return {"indicator_names": []}
    return {"indicator_names": [row[0] for row in result]}

# Endpoint to get country names with the minimum indicator value within a year range and topic
@app.get("/v1/world_development_indicators/country_min_indicator_value", operation_id="get_country_min_indicator_value", summary="Retrieves the names of countries with the lowest indicator value within a specified year range and topic. The year range is defined by the start_year and end_year parameters, while the topic is determined by the topic parameter. The operation returns a list of country names that meet these criteria.")
async def get_country_min_indicator_value(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), topic: str = Query(..., description="Topic of the series")):
    cursor.execute("SELECT CountryName FROM Indicators WHERE Value = ( SELECT MIN(T1.Value) FROM Indicators AS T1 INNER JOIN Series AS T2 ON T1.IndicatorName = T2.IndicatorName WHERE T1.Year >= ? AND T1.Year < ? AND T2.Topic = ? )", (start_year, end_year, topic))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get the percentage of countries with a specific indicator value below a threshold within an income group
@app.get("/v1/world_development_indicators/percentage_countries_indicator_value", operation_id="get_percentage_countries_indicator_value", summary="Retrieves the proportion of countries within a specified income group that have a value for the given indicator below a certain threshold. The indicator in question is 'CO2 emissions from liquid fuel consumption (% of total)'. The income group can be one of the following: High income, Upper middle income, Lower middle income, or Low income.")
async def get_percentage_countries_indicator_value(indicator_name: str = Query(..., description="Name of the indicator"), value_threshold: float = Query(..., description="Threshold value of the indicator"), income_group: str = Query(..., description="Income group of the countries")):
    cursor.execute("SELECT SUM(CASE WHEN T2.IndicatorName = ? AND t2.Value < ? THEN 1 ELSE 0 END) * 1.0 / COUNT(T1.CountryCode) persent FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IncomeGroup = ?", (indicator_name, value_threshold, income_group))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct indicator codes based on indicator name
@app.get("/v1/world_development_indicators/distinct_indicator_codes", operation_id="get_distinct_indicator_codes", summary="Retrieves a unique set of codes associated with a specific indicator name. The input parameter specifies the indicator name, and the operation returns the corresponding distinct codes.")
async def get_distinct_indicator_codes(indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT DISTINCT IndicatorCode FROM Indicators WHERE IndicatorName = ?", (indicator_name,))
    result = cursor.fetchall()
    if not result:
        return {"indicator_codes": []}
    return {"indicator_codes": [row[0] for row in result]}

# Endpoint to get table names based on the system of national accounts
@app.get("/v1/world_development_indicators/table_names_national_accounts", operation_id="get_table_names_national_accounts", summary="Retrieves the names of tables that contain development indicators for countries using a specified system of national accounts. The system of national accounts is a parameter that filters the results.")
async def get_table_names_national_accounts(system_of_national_accounts: str = Query(..., description="System of national accounts used by the country")):
    cursor.execute("SELECT TableName FROM Country WHERE SystemOfNationalAccounts = ?", (system_of_national_accounts,))
    result = cursor.fetchall()
    if not result:
        return {"table_names": []}
    return {"table_names": [row[0] for row in result]}

# Endpoint to get distinct series codes based on currency unit
@app.get("/v1/world_development_indicators/distinct_series_codes_currency", operation_id="get_distinct_series_codes_currency", summary="Retrieves a unique list of series codes associated with countries that use the specified currency unit. This operation provides a concise overview of the distinct series codes linked to countries sharing a common currency unit.")
async def get_distinct_series_codes_currency(currency_unit: str = Query(..., description="Currency unit of the country")):
    cursor.execute("SELECT DISTINCT T2.SeriesCode FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.CurrencyUnit = ?", (currency_unit,))
    result = cursor.fetchall()
    if not result:
        return {"series_codes": []}
    return {"series_codes": [row[0] for row in result]}

# Endpoint to get the long name of countries based on series code
@app.get("/v1/world_development_indicators/country_long_name_by_series_code", operation_id="get_country_long_name_by_series_code", summary="Retrieves the full name of countries associated with a given series code. The series code is used to filter the countries from the database.")
async def get_country_long_name_by_series_code(series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT T1.LongName FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.SeriesCode = ?", (series_code,))
    result = cursor.fetchall()
    if not result:
        return {"long_names": []}
    return {"long_names": [row[0] for row in result]}

# Endpoint to get series codes based on currency unit
@app.get("/v1/world_development_indicators/series_codes_by_currency_unit", operation_id="get_series_codes_by_currency_unit", summary="Retrieves a list of series codes associated with a specific currency unit. The operation filters the series codes based on the provided currency unit, enabling users to obtain relevant series codes for further analysis or processing.")
async def get_series_codes_by_currency_unit(currency_unit: str = Query(..., description="Currency unit")):
    cursor.execute("SELECT T2.SeriesCode FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.CurrencyUnit = ?", (currency_unit,))
    result = cursor.fetchall()
    if not result:
        return {"series_codes": []}
    return {"series_codes": [row[0] for row in result]}

# Endpoint to get table names based on series code
@app.get("/v1/world_development_indicators/table_names_by_series_code", operation_id="get_table_names_by_series_code", summary="Retrieves the names of tables that contain data related to a specific series code. The series code is used to filter the tables and return only those that are relevant to the provided code.")
async def get_table_names_by_series_code(series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT T1.TableName FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Seriescode = ?", (series_code,))
    result = cursor.fetchall()
    if not result:
        return {"table_names": []}
    return {"table_names": [row[0] for row in result]}

# Endpoint to get distinct country names based on income group
@app.get("/v1/world_development_indicators/distinct_country_names_by_income_group", operation_id="get_distinct_country_names_by_income_group", summary="Retrieves a list of unique country names that belong to a specified income group. The income group is used to filter the countries and return only those that match the specified group.")
async def get_distinct_country_names_by_income_group(income_group: str = Query(..., description="Income group")):
    cursor.execute("SELECT DISTINCT T2.CountryName FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IncomeGroup = ?", (income_group,))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get series codes and country codes based on region and income group
@app.get("/v1/world_development_indicators/series_country_codes_by_region_income_group", operation_id="get_series_country_codes_by_region_income_group", summary="Retrieves a list of series codes and their corresponding country codes for a given region and income group. The region and income group are used to filter the results, providing a targeted set of data.")
async def get_series_country_codes_by_region_income_group(region: str = Query(..., description="Region"), income_group: str = Query(..., description="Income group")):
    cursor.execute("SELECT T2.SeriesCode, T2.CountryCode FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.Region = ? AND T1.IncomeGroup = ?", (region, income_group))
    result = cursor.fetchall()
    if not result:
        return {"series_country_codes": []}
    return {"series_country_codes": [{"series_code": row[0], "country_code": row[1]} for row in result]}

# Endpoint to get country codes and series codes based on currency unit and income group
@app.get("/v1/world_development_indicators/country_series_codes_by_currency_income_group", operation_id="get_country_series_codes_by_currency_income_group", summary="Retrieves a list of country codes and their corresponding series codes that match a specified currency unit and income group. The currency unit and income group are used as filters to determine the relevant country codes and series codes.")
async def get_country_series_codes_by_currency_income_group(currency_unit: str = Query(..., description="Currency unit"), income_group: str = Query(..., description="Income group")):
    cursor.execute("SELECT T1.CountryCode, T2.SeriesCode FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.CurrencyUnit = ? AND T1.IncomeGroup = ?", (currency_unit, income_group))
    result = cursor.fetchall()
    if not result:
        return {"country_series_codes": []}
    return {"country_series_codes": [{"country_code": row[0], "series_code": row[1]} for row in result]}

# Endpoint to get distinct country names based on income group and the earliest national accounts base year
@app.get("/v1/world_development_indicators/distinct_country_names_by_income_group_earliest_base_year", operation_id="get_distinct_country_names_by_income_group_earliest_base_year", summary="Retrieves a list of unique country names that belong to a specified income group and have the earliest national accounts base year. The income group is a required input parameter.")
async def get_distinct_country_names_by_income_group_earliest_base_year(income_group: str = Query(..., description="Income group")):
    cursor.execute("SELECT DISTINCT T1.CountryName FROM indicators AS T1 INNER JOIN country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.IncomeGroup = ? UNION SELECT longname FROM ( SELECT longname FROM country WHERE NationalAccountsBaseYear <> '' ORDER BY NationalAccountsBaseYear ASC LIMIT 1 )", (income_group,))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get distinct country codes and names based on currency unit and income groups
@app.get("/v1/world_development_indicators/distinct_country_codes_names_by_currency_income_groups", operation_id="get_distinct_country_codes_names_by_currency_income_groups", summary="Retrieve unique country codes and their corresponding names that share a specified currency unit and belong to either of the two provided income groups.")
async def get_distinct_country_codes_names_by_currency_income_groups(currency_unit: str = Query(..., description="Currency unit"), income_group_1: str = Query(..., description="First income group"), income_group_2: str = Query(..., description="Second income group")):
    cursor.execute("SELECT DISTINCT T1.CountryCode, T2.CountryName FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.CurrencyUnit = ? AND (T1.IncomeGroup = ? OR T1.IncomeGroup = ?)", (currency_unit, income_group_1, income_group_2))
    result = cursor.fetchall()
    if not result:
        return {"country_codes_names": []}
    return {"country_codes_names": [{"country_code": row[0], "country_name": row[1]} for row in result]}

# Endpoint to get table names and currency units based on series code
@app.get("/v1/world_development_indicators/table_names_currency_units_by_series_code", operation_id="get_table_names_currency_units_by_series_code", summary="Retrieves the names of tables and their respective currency units that are associated with a given series code. The series code is used to filter the results, ensuring that only relevant data is returned.")
async def get_table_names_currency_units_by_series_code(series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT T1.TableName, T1.CurrencyUnit FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.SeriesCode = ?", (series_code,))
    result = cursor.fetchall()
    if not result:
        return {"table_names_currency_units": []}
    return {"table_names_currency_units": [{"table_name": row[0], "currency_unit": row[1]} for row in result]}

# Endpoint to get distinct country codes and names based on income group
@app.get("/v1/world_development_indicators/distinct_country_codes_names_by_income_group", operation_id="get_distinct_country_codes_names_by_income_group", summary="Retrieves unique country codes and their corresponding names for a specified income group. The income group is used to filter the results, ensuring that only countries within the specified income group are returned.")
async def get_distinct_country_codes_names_by_income_group(income_group: str = Query(..., description="Income group")):
    cursor.execute("SELECT DISTINCT T1.CountryCode, T2.CountryName FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IncomeGroup = ?", (income_group,))
    result = cursor.fetchall()
    if not result:
        return {"country_codes_names": []}
    return {"country_codes_names": [{"country_code": row[0], "country_name": row[1]} for row in result]}

# Endpoint to get distinct country details based on currency unit and income group
@app.get("/v1/world_development_indicators/country_details_currency_income", operation_id="get_country_details", summary="Retrieves unique country details, including the country code, currency unit, and income group, based on a specified currency unit and a partial or full income group match. The income group parameter supports wildcard searches using the '%' symbol.")
async def get_country_details(currency_unit: str = Query(..., description="Currency unit of the country"), income_group: str = Query(..., description="Income group of the country (use % for wildcard)")):
    cursor.execute("SELECT DISTINCT T1.CountryCode, T1.CurrencyUnit, T1.IncomeGroup FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.CurrencyUnit = ? AND T1.IncomeGroup LIKE ?", (currency_unit, income_group))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get a limited number of country codes and short names
@app.get("/v1/world_development_indicators/limited_country_codes", operation_id="get_limited_country_codes", summary="Retrieves a limited number of country codes and their corresponding short names. The number of records returned is determined by the provided limit parameter. This operation is useful for obtaining a subset of country codes and short names, which can be used for further data retrieval or analysis.")
async def get_limited_country_codes(limit: int = Query(..., description="Number of records to return")):
    cursor.execute("SELECT CountryCode, ShortName FROM Country LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the count of countries with short names starting with a specific letter and their alpha-2 codes
@app.get("/v1/world_development_indicators/count_countries_alpha2_codes", operation_id="get_count_countries_alpha2_codes", summary="Retrieve the total number of countries with short names starting with a specified prefix and their corresponding alpha-2 codes. The prefix can include a wildcard character for broader searches.")
async def get_count_countries_alpha2_codes(short_name_prefix: str = Query(..., description="Prefix for the short name of the country (use % for wildcard)")):
    cursor.execute("SELECT COUNT(ShortName) FROM Country WHERE ShortName LIKE ? UNION SELECT alpha2code FROM country WHERE shortname LIKE ?", (short_name_prefix, short_name_prefix))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get table names based on currency unit
@app.get("/v1/world_development_indicators/table_names_currency_unit", operation_id="get_table_names_currency_unit", summary="Retrieves a list of table names that correspond to the specified currency unit. This operation allows you to filter tables based on the currency unit used in a country. The input parameter is the currency unit, which determines the tables returned.")
async def get_table_names_currency_unit(currency_unit: str = Query(..., description="Currency unit of the country")):
    cursor.execute("SELECT TableName FROM Country WHERE CurrencyUnit = ?", (currency_unit,))
    result = cursor.fetchall()
    if not result:
        return {"table_names": []}
    return {"table_names": result}

# Endpoint to get the count of countries with no special notes and their long names
@app.get("/v1/world_development_indicators/count_countries_no_special_notes", operation_id="get_count_countries_no_special_notes", summary="Retrieves the total number of countries that have no special notes and returns their long names.")
async def get_count_countries_no_special_notes():
    cursor.execute("SELECT COUNT(LongName) FROM Country WHERE SpecialNotes = '' UNION SELECT longname FROM country WHERE specialnotes = ''")
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get country codes and regions based on income group and region
@app.get("/v1/world_development_indicators/country_codes_regions_income_group", operation_id="get_country_codes_regions", summary="Retrieves the country codes and regions that match the specified income groups and region. The operation allows for filtering by two income groups and supports partial region matches using wildcard characters.")
async def get_country_codes_regions(income_group1: str = Query(..., description="First income group of the country"), income_group2: str = Query(..., description="Second income group of the country"), region: str = Query(..., description="Region of the country (use % for wildcard)")):
    cursor.execute("SELECT CountryCode, Region FROM Country WHERE (IncomeGroup = ? OR IncomeGroup = ?) AND Region LIKE ?", (income_group1, income_group2, region))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get long names of countries based on national accounts base year
@app.get("/v1/world_development_indicators/long_names_national_accounts_base_year", operation_id="get_long_names_national_accounts_base_year", summary="Retrieves the full names of countries with a national accounts base year earlier than the provided year. The input parameter specifies the national accounts base year in the 'YYYY' format. The operation excludes countries with an empty base year.")
async def get_long_names_national_accounts_base_year(base_year: str = Query(..., description="National accounts base year (format: 'YYYY')")):
    cursor.execute("SELECT LongName FROM Country WHERE NationalAccountsBaseYear < ? AND NationalAccountsBaseYear != ''", (base_year,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get country codes based on income group and series code
@app.get("/v1/world_development_indicators/country_codes_income_group_series_code", operation_id="get_country_codes_income_group_series_code", summary="Retrieves the country codes that match the specified income group and series code. The income group and series code are used to filter the results from the Country and CountryNotes tables, respectively. The operation returns a list of country codes that meet the specified criteria.")
async def get_country_codes_income_group_series_code(income_group: str = Query(..., description="Income group of the country"), series_code: str = Query(..., description="Series code of the country")):
    cursor.execute("SELECT T1.CountryCode FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.IncomeGroup = ? AND T2.Seriescode = ?", (income_group, series_code))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get distinct table names based on description
@app.get("/v1/world_development_indicators/distinct_table_names_description", operation_id="get_distinct_table_names_description", summary="Retrieves a list of unique table names that match a specified description of country notes. This operation filters the Country table based on the provided description and returns the distinct table names associated with the matching records.")
async def get_distinct_table_names_description(description: str = Query(..., description="Description of the country notes")):
    cursor.execute("SELECT DISTINCT T1.TableName FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"table_names": []}
    return {"table_names": result}

# Endpoint to get descriptions and series codes based on table name and year
@app.get("/v1/world_development_indicators/descriptions_series_codes_table_name_year", operation_id="get_descriptions_series_codes", summary="Retrieves descriptions and series codes for a specific country table and year. The operation filters data based on the provided table name and year, returning relevant descriptions and series codes.")
async def get_descriptions_series_codes(table_name: str = Query(..., description="Table name of the country"), year: str = Query(..., description="Year (format: 'YRYYYY')")):
    cursor.execute("SELECT T2.Description, T2.Seriescode FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T1.TableName = ? AND T2.Year = ?", (table_name, year))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": result}

# Endpoint to get distinct years and table names for a specific indicator
@app.get("/v1/world_development_indicators/distinct_years_table_names_by_indicator", operation_id="get_distinct_years_table_names", summary="Retrieves a list of unique years and corresponding table names associated with a specific indicator. The provided indicator name filters the results to display only the relevant data.")
async def get_distinct_years_table_names(indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT DISTINCT T2.Year, T1.TableName FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.IndicatorName = ?", (indicator_name,))
    result = cursor.fetchall()
    if not result:
        return {"years_table_names": []}
    return {"years_table_names": [{"year": row[0], "table_name": row[1]} for row in result]}

# Endpoint to get distinct long names of countries for a specific year with non-null indicator names
@app.get("/v1/world_development_indicators/distinct_long_names_by_year", operation_id="get_distinct_long_names", summary="Retrieves a list of unique country names for a given year, excluding those with missing indicator names. This operation returns the full names of countries that have at least one non-null indicator name in the specified year.")
async def get_distinct_long_names(year: int = Query(..., description="Year")):
    cursor.execute("SELECT DISTINCT T1.LongName FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Year = ? AND T2.IndicatorName IS NOT NULL", (year,))
    result = cursor.fetchall()
    if not result:
        return {"long_names": []}
    return {"long_names": [row[0] for row in result]}

# Endpoint to get currency units and indicator codes for a specific country and year
@app.get("/v1/world_development_indicators/currency_units_indicator_codes_by_country_year", operation_id="get_currency_units_indicator_codes", summary="Retrieves the currency unit and associated indicator code for a given country in a specified year. The operation requires the table name and the year as input parameters. The table name is used to identify the relevant country data, while the year determines the time frame for the indicator code.")
async def get_currency_units_indicator_codes(table_name: str = Query(..., description="Name of the table"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T1.currencyunit, T2.IndicatorCode FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.TableName = ? AND T2.Year = ?", (table_name, year))
    result = cursor.fetchall()
    if not result:
        return {"currency_units_indicator_codes": []}
    return {"currency_units_indicator_codes": [{"currency_unit": row[0], "indicator_code": row[1]} for row in result]}

# Endpoint to get top 5 distinct country codes and regions based on indicator values
@app.get("/v1/world_development_indicators/top_5_country_codes_regions", operation_id="get_top_5_country_codes_regions", summary="Retrieves the top 5 distinct country codes and their respective regions, ranked by the highest indicator values. The data is sourced from a join operation between the Country and Indicator tables, ensuring that only the most relevant records are returned.")
async def get_top_5_country_codes_regions():
    cursor.execute("SELECT DISTINCT T1.CountryCode, T1.Region FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode ORDER BY T2.Value DESC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"country_codes_regions": []}
    return {"country_codes_regions": [{"country_code": row[0], "region": row[1]} for row in result]}

# Endpoint to get the count of distinct country codes and currency units for a specific description
@app.get("/v1/world_development_indicators/count_distinct_country_codes_currency_units", operation_id="get_count_distinct_country_codes_currency_units", summary="Retrieves the count of unique country codes and distinct currency units associated with a specific description in the country notes.")
async def get_count_distinct_country_codes_currency_units(description: str = Query(..., description="Description of the country notes")):
    cursor.execute("SELECT COUNT(DISTINCT T1.Countrycode) FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ? UNION SELECT DISTINCT t1.CurrencyUnit FROM country AS t1 INNER JOIN countrynotes AS t2 ON t1.CountryCode = t2.Countrycode WHERE t2.Description = ?", (description, description))
    result = cursor.fetchall()
    if not result:
        return {"count_distinct_country_codes": [], "currency_units": []}
    return {"count_distinct_country_codes": result[0][0], "currency_units": [row[0] for row in result[1:]]}

# Endpoint to get the count of descriptions and distinct table names for a specific year
@app.get("/v1/world_development_indicators/count_descriptions_distinct_table_names_by_year", operation_id="get_count_descriptions_distinct_table_names", summary="Retrieve the total count of distinct descriptions and table names associated with a specific year. The operation returns two separate results: the total count of unique descriptions and the list of distinct table names, both filtered by the provided year.")
async def get_count_descriptions_distinct_table_names(year: str = Query(..., description="Year in 'YRXXXX' format")):
    cursor.execute("SELECT COUNT(T2.Description) FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Year = ? UNION ALL SELECT DISTINCT T1.TableName FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Year = ?", (year, year))
    result = cursor.fetchall()
    if not result:
        return {"count_descriptions": [], "table_names": []}
    return {"count_descriptions": result[0][0], "table_names": [row[0] for row in result[1:]]}

# Endpoint to get series codes and WB2 codes for a specific description
@app.get("/v1/world_development_indicators/series_codes_wb2_codes_by_description", operation_id="get_series_codes_wb2_codes", summary="Retrieves the series codes and WB2 codes associated with a specific country notes description. The provided description is used to filter the results, ensuring that only relevant data is returned.")
async def get_series_codes_wb2_codes(description: str = Query(..., description="Description of the country notes")):
    cursor.execute("SELECT T2.seriescode, T1.Wb2Code FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"series_codes_wb2_codes": []}
    return {"series_codes_wb2_codes": [{"series_code": row[0], "wb2_code": row[1]} for row in result]}

# Endpoint to get the country name with the highest indicator value for a specific income group and indicator name
@app.get("/v1/world_development_indicators/top_country_by_income_group_indicator", operation_id="get_top_country_by_income_group_indicator", summary="Retrieves the name of the country with the highest value for a specified indicator within a given income group. The income group and indicator name are provided as input parameters.")
async def get_top_country_by_income_group_indicator(income_group: str = Query(..., description="Income group of the country"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT T2.CountryName FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IncomeGroup = ? AND T2.IndicatorName = ? ORDER BY T2.Value LIMIT 1", (income_group, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the sum of indicator values for a specific income group, year, and indicator name
@app.get("/v1/world_development_indicators/sum_indicator_values_by_income_group_year", operation_id="get_sum_indicator_values", summary="Retrieves the total value of a specified indicator for a given income group and year. The income group can be filtered using a wildcard character. The year and indicator name must be provided exactly.")
async def get_sum_indicator_values(income_group: str = Query(..., description="Income group of the country (use '%' for wildcard)"), year: int = Query(..., description="Year"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT SUM(T2.Value) FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IncomeGroup LIKE ? AND T2.Year = ? AND T2.IndicatorName = ?", (income_group, year, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"sum_indicator_values": []}
    return {"sum_indicator_values": result[0]}

# Endpoint to get the country name and currency unit for a specific indicator and year
@app.get("/v1/world_development_indicators/country_currency_by_indicator_year", operation_id="get_country_currency_by_indicator_year", summary="Retrieves the name of the country and its currency unit for the specified indicator and year, sorted by the indicator value in descending order. The result is limited to the top entry.")
async def get_country_currency_by_indicator_year(indicator_name: str = Query(..., description="Indicator name"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T2.countryname, T1.CurrencyUnit FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.IndicatorName = ? AND T2.Year = ? AND T1.CurrencyUnit != '' ORDER BY T2.Value DESC LIMIT 1", (indicator_name, year))
    result = cursor.fetchone()
    if not result:
        return {"country_name": [], "currency_unit": []}
    return {"country_name": result[0], "currency_unit": result[1]}

# Endpoint to get the count of countries using a specific system of national accounts
@app.get("/v1/world_development_indicators/count_countries_by_national_accounts", operation_id="get_count_countries_by_national_accounts", summary="Retrieves the total number of countries that employ a specified system of national accounts. The system of national accounts is provided as an input parameter.")
async def get_count_countries_by_national_accounts(system_of_national_accounts: str = Query(..., description="System of national accounts")):
    cursor.execute("SELECT COUNT(CountryCode) FROM Country WHERE SystemOfNationalAccounts = ?", (system_of_national_accounts,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the short names of countries with complete vital registration
@app.get("/v1/world_development_indicators/countries_with_complete_vital_registration", operation_id="get_countries_with_complete_vital_registration", summary="Retrieves the short names of countries where the vital registration status is complete. The operation filters countries based on the provided vital registration complete status.")
async def get_countries_with_complete_vital_registration(vital_registration_complete: str = Query(..., description="Vital registration complete status")):
    cursor.execute("SELECT ShortName FROM Country WHERE VitalRegistrationComplete = ?", (vital_registration_complete,))
    result = cursor.fetchall()
    if not result:
        return {"short_names": []}
    return {"short_names": [row[0] for row in result]}

# Endpoint to get the short and long names of countries with a specific range of latest population census and complete vital registration
@app.get("/v1/world_development_indicators/countries_by_census_range_vital_registration", operation_id="get_countries_by_census_range_vital_registration", summary="Retrieves the short and long names of countries that have conducted a population census between the specified minimum and maximum years, inclusive, and have complete vital registration. The response includes countries where the latest census year falls within the provided range and the vital registration status is complete.")
async def get_countries_by_census_range_vital_registration(min_census_year: int = Query(..., description="Minimum census year"), max_census_year: int = Query(..., description="Maximum census year"), vital_registration_complete: str = Query(..., description="Vital registration complete status")):
    cursor.execute("SELECT ShortName, LongName FROM Country WHERE LatestPopulationCensus >= ? AND LatestPopulationCensus < ? AND VitalRegistrationComplete = ?", (min_census_year, max_census_year, vital_registration_complete))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [{"short_name": row[0], "long_name": row[1]} for row in result]}

# Endpoint to get the country with the highest value for a specific indicator and year
@app.get("/v1/world_development_indicators/top_country_by_indicator_year", operation_id="get_top_country_by_indicator_year", summary="Retrieves the country with the highest recorded value for a specified development indicator in a given year. The operation requires the year and the name of the indicator as input parameters.")
async def get_top_country_by_indicator_year(year: int = Query(..., description="Year"), indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT CountryName FROM Indicators WHERE Year = ? AND IndicatorName = ? ORDER BY Value DESC LIMIT 1", (year, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the ratio of max to min values and countries with max and min values for a specific indicator and year
@app.get("/v1/world_development_indicators/ratio_max_min_values_indicator_year", operation_id="get_ratio_max_min_values_indicator_year", summary="Get the ratio of max to min values and countries with max and min values for a specific indicator and year")
async def get_ratio_max_min_values_indicator_year(indicator_name: str = Query(..., description="Indicator name"), year: str = Query(..., description="Year")):
    cursor.execute("SELECT CAST(MAX(value) AS REAL) / MIN(value) FROM indicators WHERE indicatorname = ? AND year = ? UNION ALL SELECT countryname FROM ( SELECT countryname, MAX(value) FROM indicators WHERE indicatorname = ? AND year = ? ) UNION SELECT countryname FROM ( SELECT countryname, MIN(value) FROM indicators WHERE indicatorname = ? AND year = ? )", (indicator_name, year, indicator_name, year, indicator_name, year))
    result = cursor.fetchall()
    if not result:
        return {"ratio": [], "countries": []}
    return {"ratio": result[0][0], "countries": [row[0] for row in result[1:]]}

# Endpoint to get the short names of countries with a specific series code in country notes
@app.get("/v1/world_development_indicators/countries_by_series_code", operation_id="get_countries_by_series_code", summary="Retrieves a list of short names of countries that have a specific series code in their notes. The series code is provided as an input parameter.")
async def get_countries_by_series_code(series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT T1.ShortName FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.SeriesCode = ?", (series_code,))
    result = cursor.fetchall()
    if not result:
        return {"short_names": []}
    return {"short_names": [row[0] for row in result]}

# Endpoint to get distinct country codes with a specific description in footnotes
@app.get("/v1/world_development_indicators/distinct_country_codes_by_description", operation_id="get_distinct_country_codes_by_description", summary="Retrieves a unique set of country codes that have a specific description in their footnotes. The description is provided as an input parameter.")
async def get_distinct_country_codes_by_description(description: str = Query(..., description="Description")):
    cursor.execute("SELECT DISTINCT T1.CountryCode FROM Country AS T1 INNER JOIN FootNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"country_codes": []}
    return {"country_codes": [row[0] for row in result]}

# Endpoint to get distinct short names of countries with a specific description in country notes
@app.get("/v1/world_development_indicators/distinct_short_names_by_description", operation_id="get_distinct_short_names_by_description", summary="Retrieves a list of unique country names that match a given description in their notes. This operation allows you to filter countries based on a specific description in their notes, providing a focused and concise result set.")
async def get_distinct_short_names_by_description(description: str = Query(..., description="Description")):
    cursor.execute("SELECT DISTINCT T1.ShortName FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"short_names": []}
    return {"short_names": [row[0] for row in result]}

# Endpoint to get distinct country codes with specific year, region, value, and indicator name
@app.get("/v1/world_development_indicators/distinct_country_codes_by_year_region_value_indicator", operation_id="get_distinct_country_codes_by_year_region_value_indicator", summary="Retrieve a unique set of country codes that meet the specified criteria: year, region, minimum value, and indicator name. The endpoint filters the data based on these parameters and returns the distinct country codes that match.")
async def get_distinct_country_codes_by_year_region_value_indicator(year: int = Query(..., description="Year"), region: str = Query(..., description="Region"), min_value: int = Query(..., description="Minimum value"), indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT DISTINCT T1.CountryCode FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Year = ? AND T1.Region = ? AND T2.Value > ? AND T2.IndicatorName = ?", (year, region, min_value, indicator_name))
    result = cursor.fetchall()
    if not result:
        return {"country_codes": []}
    return {"country_codes": [row[0] for row in result]}

# Endpoint to get the maximum value of a specific indicator for a given income group and year
@app.get("/v1/world_development_indicators/max_indicator_value", operation_id="get_max_indicator_value", summary="Retrieves the highest value of a specified indicator, such as the total population, for a particular income group in a given year. The income group and year are used to filter the data, providing a focused result.")
async def get_max_indicator_value(income_group: str = Query(..., description="Income group of the country"), year: int = Query(..., description="Year of the indicator"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT MAX(T2.Value) FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IncomeGroup = ? AND T2.Year = ? AND T2.IndicatorName = ?", (income_group, year, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"max_value": []}
    return {"max_value": result[0]}

# Endpoint to get distinct short names of countries based on income group, region, and topic
@app.get("/v1/world_development_indicators/distinct_short_names", operation_id="get_distinct_short_names", summary="Retrieves a list of unique short names for countries that match the specified income group, region, and topic. The operation filters countries based on the provided income group and region, and further narrows the results by considering the topic of the associated series.")
async def get_distinct_short_names(income_group: str = Query(..., description="Income group of the country"), region: str = Query(..., description="Region of the country"), topic: str = Query(..., description="Topic of the series")):
    cursor.execute("SELECT DISTINCT T1.ShortName FROM Country AS T1 INNER JOIN footnotes AS T2 ON T1.CountryCode = T2.CountryCode INNER JOIN Series AS T3 ON T2.Seriescode = T3.SeriesCode WHERE T1.IncomeGroup = ? AND T1.Region = ? AND T3.Topic = ?", (income_group, region, topic))
    result = cursor.fetchall()
    if not result:
        return {"short_names": []}
    return {"short_names": [row[0] for row in result]}

# Endpoint to get distinct long names of countries based on a specific topic
@app.get("/v1/world_development_indicators/distinct_long_names_by_topic", operation_id="get_distinct_long_names_by_topic", summary="Retrieve a unique list of country names associated with a specific topic. The topic is used to filter the series data and identify the relevant countries.")
async def get_distinct_long_names_by_topic(topic: str = Query(..., description="Topic of the series")):
    cursor.execute("SELECT DISTINCT T1.LongName FROM Country AS T1 INNER JOIN footnotes AS T2 ON T1.CountryCode = T2.Countrycode INNER JOIN Series AS T3 ON T2.Seriescode = T3.SeriesCode WHERE T3.Topic = ?", (topic,))
    result = cursor.fetchall()
    if not result:
        return {"long_names": []}
    return {"long_names": [row[0] for row in result]}

# Endpoint to get distinct table names based on latest trade data and indicator code
@app.get("/v1/world_development_indicators/distinct_table_names", operation_id="get_distinct_table_names", summary="Retrieves a list of distinct table names that match the specified latest trade data year and indicator code. This operation filters the Country and Indicator tables based on the provided parameters, returning only the unique table names that meet the criteria.")
async def get_distinct_table_names(latest_trade_data: int = Query(..., description="Latest trade data year"), indicator_code: str = Query(..., description="Indicator code")):
    cursor.execute("SELECT DISTINCT T1.TableName FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.LatestTradeData = ? AND T2.IndicatorCode = ?", (latest_trade_data, indicator_code))
    result = cursor.fetchall()
    if not result:
        return {"table_names": []}
    return {"table_names": [row[0] for row in result]}

# Endpoint to get distinct long names of countries based on region and series code
@app.get("/v1/world_development_indicators/distinct_long_names_by_region_series", operation_id="get_distinct_long_names_by_region_series", summary="Retrieve a unique list of country names based on the specified region and series code. This operation fetches the distinct long names of countries from the database, filtering by the provided region and series code.")
async def get_distinct_long_names_by_region_series(region: str = Query(..., description="Region of the country"), series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT DISTINCT T3.LongName FROM SeriesNotes AS T1 INNER JOIN CountryNotes AS T2 ON T1.SeriesCode = T2.Seriescode INNER JOIN Country AS T3 ON T2.Countrycode = T3.CountryCode WHERE T3.Region = ? AND T1.SeriesCode = ?", (region, series_code))
    result = cursor.fetchall()
    if not result:
        return {"long_names": []}
    return {"long_names": [row[0] for row in result]}

# Endpoint to get short names of countries based on currency unit with a limit
@app.get("/v1/world_development_indicators/short_names_by_currency", operation_id="get_short_names_by_currency", summary="Retrieves a limited number of short names of countries that use a specified currency unit. The currency unit is used to filter the results, and the limit parameter restricts the number of results returned.")
async def get_short_names_by_currency(currency_unit: str = Query(..., description="Currency unit of the country"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT ShortName FROM country WHERE currencyunit = ? LIMIT ?", (currency_unit, limit))
    result = cursor.fetchall()
    if not result:
        return {"short_names": []}
    return {"short_names": [row[0] for row in result]}

# Endpoint to get long names of countries based on income group and region
@app.get("/v1/world_development_indicators/long_names_by_income_region", operation_id="get_long_names_by_income_region", summary="Retrieves the full names of countries that belong to a specific income group and region. The operation filters countries based on the provided income group and region parameters, returning a list of corresponding country names.")
async def get_long_names_by_income_region(income_group: str = Query(..., description="Income group of the country"), region: str = Query(..., description="Region of the country")):
    cursor.execute("SELECT LongName FROM Country WHERE IncomeGroup = ? AND Region = ?", (income_group, region))
    result = cursor.fetchall()
    if not result:
        return {"long_names": []}
    return {"long_names": [row[0] for row in result]}

# Endpoint to get long names of countries based on system of trade with a limit
@app.get("/v1/world_development_indicators/long_names_by_trade_system", operation_id="get_long_names_by_trade_system", summary="Retrieves a limited number of long names of countries that use a specified system of trade. The system of trade is used to filter the results, and the limit parameter restricts the number of results returned.")
async def get_long_names_by_trade_system(system_of_trade: str = Query(..., description="System of trade of the country"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT LongName FROM Country WHERE SystemOfTrade = ? LIMIT ?", (system_of_trade, limit))
    result = cursor.fetchall()
    if not result:
        return {"long_names": []}
    return {"long_names": [row[0] for row in result]}

# Endpoint to get country names based on indicator name pattern with a limit
@app.get("/v1/world_development_indicators/country_names_by_indicator", operation_id="get_country_names_by_indicator", summary="Retrieves a list of country names that match the specified indicator name pattern, sorted by their corresponding values in descending order. The number of results returned is limited by the provided limit parameter.")
async def get_country_names_by_indicator(indicator_name_pattern: str = Query(..., description="Pattern of the indicator name (use %% for wildcard)"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT CountryName FROM Indicators WHERE IndicatorName LIKE ? ORDER BY Value DESC LIMIT ?", (indicator_name_pattern, limit))
    result = cursor.fetchall()
    if not result:
        return {"country_names": []}
    return {"country_names": [row[0] for row in result]}

# Endpoint to get the country with the highest value for a specific indicator
@app.get("/v1/world_development_indicators/top_country_by_indicator", operation_id="get_top_country_by_indicator", summary="Get the country with the highest value for a specific indicator")
async def get_top_country_by_indicator(indicator_name: str = Query(..., description="Name of the indicator (e.g., 'Arable land (% of land area)')")):
    cursor.execute("SELECT CountryName FROM Indicators WHERE IndicatorName LIKE ? ORDER BY Value DESC LIMIT 1", (indicator_name,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get distinct topics based on license type
@app.get("/v1/world_development_indicators/distinct_topics_by_license_type", operation_id="get_distinct_topics_by_license_type", summary="Retrieves a unique list of topics associated with a specific license type from the World Development Indicators dataset. The license type is provided as an input parameter, allowing the user to filter the topics based on their desired license type.")
async def get_distinct_topics_by_license_type(license_type: str = Query(..., description="Type of license (e.g., 'Restricted')")):
    cursor.execute("SELECT DISTINCT Topic FROM Series WHERE LicenseType = ?", (license_type,))
    result = cursor.fetchall()
    if not result:
        return {"topics": []}
    return {"topics": [row[0] for row in result]}

# Endpoint to get country codes based on income group and external debt reporting status
@app.get("/v1/world_development_indicators/country_codes_by_income_group_and_debt_status", operation_id="get_country_codes_by_income_group_and_debt_status", summary="Retrieve the country codes of nations that belong to a specified income group and have a particular external debt reporting status. The income group and debt status are provided as input parameters.")
async def get_country_codes_by_income_group_and_debt_status(income_group: str = Query(..., description="Income group (e.g., 'Upper middle income')"), external_debt_reporting_status: str = Query(..., description="External debt reporting status (e.g., 'Preliminary')")):
    cursor.execute("SELECT CountryCode FROM Country WHERE IncomeGroup = ? AND ExternalDebtReportingStatus = ?", (income_group, external_debt_reporting_status))
    result = cursor.fetchall()
    if not result:
        return {"country_codes": []}
    return {"country_codes": [row[0] for row in result]}

# Endpoint to get the percentage of countries with a specific external debt reporting status in a given region
@app.get("/v1/world_development_indicators/percentage_countries_by_debt_status_and_region", operation_id="get_percentage_countries_by_debt_status_and_region", summary="Retrieves the percentage of countries in a specified region that have a particular external debt reporting status. The operation calculates this percentage by summing the number of countries with the given debt status and dividing it by the total number of countries in the region. The debt status can be 'Actual', 'Estimated', or 'Not applicable'.")
async def get_percentage_countries_by_debt_status_and_region(external_debt_reporting_status: str = Query(..., description="External debt reporting status (e.g., 'Actual')"), region: str = Query(..., description="Region (e.g., 'Middle East & North Africa')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN ExternalDebtReportingStatus = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(CountryCode) FROM Country WHERE region = ?", (external_debt_reporting_status, region))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of countries with specific government accounting concepts and external debt reporting status
@app.get("/v1/world_development_indicators/count_countries_by_accounting_concepts_and_debt_status", operation_id="get_count_countries_by_accounting_concepts_and_debt_status", summary="Retrieve the total number of countries that meet specific government accounting concepts and external debt reporting criteria. This operation provides a breakdown of the count based on two distinct accounting concepts, allowing users to analyze the distribution of countries based on their accounting practices and debt status.")
async def get_count_countries_by_accounting_concepts_and_debt_status(government_accounting_concept_1: str = Query(..., description="First government accounting concept (e.g., 'Budgetary central government')"), government_accounting_concept_2: str = Query(..., description="Second government accounting concept (e.g., 'Consolidated central government')"), external_debt_reporting_status: str = Query(..., description="External debt reporting status (e.g., 'Actual')")):
    cursor.execute("SELECT SUM(CASE WHEN GovernmentAccountingConcept = ? THEN 1 ELSE 0 END), SUM(CASE WHEN GovernmentAccountingConcept = ? THEN 1 ELSE 0 END) central_nums FROM country WHERE ExternalDebtReportingStatus = ?", (government_accounting_concept_1, government_accounting_concept_2, external_debt_reporting_status))
    result = cursor.fetchone()
    if not result:
        return {"counts": []}
    return {"counts": {"concept_1": result[0], "concept_2": result[1]}}

# Endpoint to get the count of countries in a specific region with a specific external debt reporting status
@app.get("/v1/world_development_indicators/count_countries_by_region_and_debt_status", operation_id="get_count_countries_by_region_and_debt_status", summary="Retrieve the number of countries in a specified region that share a common external debt reporting status. This operation allows you to filter countries based on their region and external debt reporting status, providing a count of countries that meet the specified criteria.")
async def get_count_countries_by_region_and_debt_status(region: str = Query(..., description="Region (e.g., 'East Asia & Pacific')"), external_debt_reporting_status: str = Query(..., description="External debt reporting status (e.g., 'Estimate')")):
    cursor.execute("SELECT COUNT(CountryCode) FROM Country WHERE Region = ? AND ExternalDebtReportingStatus = ?", (region, external_debt_reporting_status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of countries in a specific region with a specific income group
@app.get("/v1/world_development_indicators/percentage_countries_by_income_group_and_region", operation_id="get_percentage_countries_by_income_group_and_region", summary="Retrieves the percentage of countries in a specified region that belong to a given income group. The income group and region are provided as input parameters.")
async def get_percentage_countries_by_income_group_and_region(income_group: str = Query(..., description="Income group (e.g., 'Lower middle income')"), region: str = Query(..., description="Region (e.g., 'Sub-Saharan Africa')")):
    cursor.execute("SELECT SUM(CASE WHEN IncomeGroup = ? THEN 1 ELSE 0 END) * 100.0 / COUNT(CountryCode) persentage FROM Country WHERE Region = ?", (income_group, region))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the maximum value of an indicator for a specific country within a given year range
@app.get("/v1/world_development_indicators/max_indicator_value_by_country_and_year_range", operation_id="get_max_indicator_value_by_country_and_year_range", summary="Get the maximum value of an indicator for a specific country within a given year range")
async def get_max_indicator_value_by_country_and_year_range(start_year: int = Query(..., description="Start year (e.g., 1961)"), end_year: int = Query(..., description="End year (e.g., 1981)"), indicator_name: str = Query(..., description="Name of the indicator (e.g., 'Agricultural land (% of land area)')"), country_long_name: str = Query(..., description="Long name of the country (e.g., 'Republic of Benin')")):
    cursor.execute("SELECT MAX(T1.Value) FROM Indicators AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Year >= ? AND T1.Year < ? AND T1.IndicatorName LIKE ? AND T2.LongName = ?", (start_year, end_year, indicator_name, country_long_name))
    result = cursor.fetchone()
    if not result:
        return {"max_value": []}
    return {"max_value": result[0]}

# Endpoint to get distinct country long names based on a specific description in country notes
@app.get("/v1/world_development_indicators/distinct_country_long_names_by_description", operation_id="get_distinct_country_long_names_by_description", summary="Retrieves a list of up to three distinct long names of countries that match a given description in their notes. The description is a specific attribute or characteristic of the country, such as its source of energy statistics.")
async def get_distinct_country_long_names_by_description(description: str = Query(..., description="Description (e.g., 'Sources: UN Energy Statistics (2014)')")):
    cursor.execute("SELECT DISTINCT T2.LongName FROM CountryNotes AS T1 INNER JOIN Country AS T2 ON T1.Countrycode = T2.CountryCode WHERE T1.Description = ? LIMIT 3", (description,))
    result = cursor.fetchall()
    if not result:
        return {"country_long_names": []}
    return {"country_long_names": [row[0] for row in result]}

# Endpoint to get the value of an indicator for a specific country and year
@app.get("/v1/world_development_indicators/indicator_value_by_country_and_year", operation_id="get_indicator_value_by_country_and_year", summary="Get the value of an indicator for a specific country and year")
async def get_indicator_value_by_country_and_year(country_long_name: str = Query(..., description="Long name of the country (e.g., 'Commonwealth of Australia')"), indicator_name: str = Query(..., description="Name of the indicator (e.g., 'Deposit interest rate (%)')"), year: int = Query(..., description="Year (e.g., 1979)")):
    cursor.execute("SELECT T1.Value FROM Indicators AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.LongName = ? AND T1.IndicatorName = ? AND T1.Year = ?", (country_long_name, indicator_name, year))
    result = cursor.fetchone()
    if not result:
        return {"value": []}
    return {"value": result[0]}

# Endpoint to get series code and description for a given country short name
@app.get("/v1/world_development_indicators/series_description_by_country", operation_id="get_series_description_by_country", summary="Retrieves the series code and description for a specific country, identified by its short name. The operation returns a list of series codes and their corresponding descriptions, providing detailed information about the development indicators for the requested country.")
async def get_series_description_by_country(short_name: str = Query(..., description="Short name of the country")):
    cursor.execute("SELECT T1.Seriescode, T1.Description FROM CountryNotes AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.ShortName = ?", (short_name,))
    result = cursor.fetchall()
    if not result:
        return {"series": []}
    return {"series": result}

# Endpoint to get distinct topics for a given country short name
@app.get("/v1/world_development_indicators/topics_by_country", operation_id="get_topics_by_country", summary="Retrieves a unique set of topics associated with a specific country, identified by its short name. This operation returns the distinct topics that are linked to the provided country in the database, offering a concise overview of the development indicators for that country.")
async def get_topics_by_country(short_name: str = Query(..., description="Short name of the country")):
    cursor.execute("SELECT DISTINCT T3.Topic FROM CountryNotes AS T1 INNER JOIN Country AS T2 ON T1.Countrycode = T2.CountryCode INNER JOIN Series AS T3 ON T1.Seriescode = T3.SeriesCode WHERE T2.ShortName = ?", (short_name,))
    result = cursor.fetchall()
    if not result:
        return {"topics": []}
    return {"topics": result}

# Endpoint to get distinct topics and descriptions for a given series code
@app.get("/v1/world_development_indicators/topics_description_by_series", operation_id="get_topics_description_by_series", summary="Retrieves unique topic names and their corresponding descriptions for a specific series code. The series code is used to filter the data and ensure the returned topics and descriptions are relevant to the specified series.")
async def get_topics_description_by_series(series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT DISTINCT T1.Topic, T2.Description FROM Series AS T1 INNER JOIN SeriesNotes AS T2 ON T1.SeriesCode = T2.Seriescode WHERE T1.SeriesCode = ?", (series_code,))
    result = cursor.fetchall()
    if not result:
        return {"topics": []}
    return {"topics": result}

# Endpoint to get the count of distinct country codes and long names for given footnote descriptions
@app.get("/v1/world_development_indicators/count_countries_by_footnote", operation_id="get_count_countries_by_footnote", summary="Retrieve the count of unique countries, identified by their codes and long names, that are associated with the provided footnote descriptions. The response is limited to a maximum of four entries.")
async def get_count_countries_by_footnote(description1: str = Query(..., description="First footnote description"), description2: str = Query(..., description="Second footnote description")):
    cursor.execute("SELECT COUNT(DISTINCT T1.CountryCode) FROM Country AS T1 INNER JOIN Footnotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ? OR T2.Description = ? UNION SELECT T1.LongName FROM Country AS T1 INNER JOIN Footnotes AS T2 ON T1.CountryCode = T2.Countrycode WHERE T2.Description = ? OR T2.Description = ? LIMIT 4", (description1, description2, description1, description2))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get country codes for specific indicator name, value, year, and external debt reporting status
@app.get("/v1/world_development_indicators/country_codes_by_indicator", operation_id="get_country_codes_by_indicator", summary="Retrieve the country codes associated with a specific indicator, based on the provided indicator name, value, year, and external debt reporting status. The response includes a list of country codes that match the given criteria.")
async def get_country_codes_by_indicator(indicator_name: str = Query(..., description="Indicator name (use % for wildcard)"), value: int = Query(..., description="Value of the indicator"), year: int = Query(..., description="Year of the indicator"), external_debt_status: str = Query(..., description="External debt reporting status")):
    cursor.execute("SELECT T2.CountryCode FROM Indicators AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IndicatorName LIKE ? AND T1.Value = ? AND T1.Year = ? AND T2.ExternalDebtReportingStatus = ?", (indicator_name, value, year, external_debt_status))
    result = cursor.fetchall()
    if not result:
        return {"country_codes": []}
    return {"country_codes": result}

# Endpoint to get the percentage of countries with indicator value above a threshold for a specific year, region, and indicator name
@app.get("/v1/world_development_indicators/percentage_countries_above_threshold", operation_id="get_percentage_countries_above_threshold", summary="Get the percentage of countries with indicator value above a threshold for a specific year, region, and indicator name")
async def get_percentage_countries_above_threshold(value_threshold: int = Query(..., description="Value threshold"), year: int = Query(..., description="Year of the indicator"), region: str = Query(..., description="Region of the country"), indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Value > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.CountryCode) FROM Indicators AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Year = ? AND T2.Region = ? AND indicatorname = ?", (value_threshold, year, region, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct indicator codes for a given country name and indicator name
@app.get("/v1/world_development_indicators/indicator_codes_by_country_indicator", operation_id="get_indicator_codes_by_country_indicator", summary="Retrieves unique indicator codes associated with a specific country and indicator. The operation filters the Indicators table based on the provided country name and indicator name, returning only the distinct indicator codes that match the criteria.")
async def get_indicator_codes_by_country_indicator(country_name: str = Query(..., description="Name of the country"), indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT DISTINCT IndicatorCode FROM Indicators WHERE CountryName = ? AND IndicatorName = ?", (country_name, indicator_name))
    result = cursor.fetchall()
    if not result:
        return {"indicator_codes": []}
    return {"indicator_codes": result}

# Endpoint to get the count and long names of countries in a specific region with a specific currency unit
@app.get("/v1/world_development_indicators/count_countries_by_region_currency", operation_id="get_count_countries_by_region_currency", summary="Retrieves the total number of countries and their long names from a specific region that use a particular currency unit. This operation provides a comprehensive view of the countries in the specified region that share the same currency.")
async def get_count_countries_by_region_currency(region: str = Query(..., description="Region of the country"), currency_unit: str = Query(..., description="Currency unit of the country")):
    cursor.execute("SELECT COUNT(longname) FROM country WHERE region = ? AND currencyunit = ? UNION SELECT longname FROM country WHERE currencyunit = ? AND region = ?", (region, currency_unit, currency_unit, region))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the country name and maximum value for a specific indicator name
@app.get("/v1/world_development_indicators/max_value_by_indicator", operation_id="get_max_value_by_indicator", summary="Get the country name and maximum value for a specific indicator name")
async def get_max_value_by_indicator(indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT countryname, MAX(value) FROM indicators WHERE indicatorname = ?", (indicator_name,))
    result = cursor.fetchall()
    if not result:
        return {"max_value": []}
    return {"max_value": result}

# Endpoint to get the count and long names of countries with a specific latest population census year
@app.get("/v1/world_development_indicators/count_countries_by_census_year", operation_id="get_count_countries_by_census_year", summary="Retrieve the total number of countries and their long names that have a latest population census in the specified year.")
async def get_count_countries_by_census_year(census_year: str = Query(..., description="Latest population census year")):
    cursor.execute("SELECT COUNT(LongName) FROM country WHERE LatestPopulationCensus = ? UNION ALL SELECT LongName FROM country WHERE LatestPopulationCensus = ?", (census_year, census_year))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": result}

# Endpoint to get the value of a specific indicator for a given year and country
@app.get("/v1/world_development_indicators/indicator_value", operation_id="get_indicator_value", summary="Retrieves the value of a specific development indicator for a given year and country. The operation requires the name of the indicator, the year, and the country as input parameters. The returned value represents the development status of the specified indicator in the provided year and country.")
async def get_indicator_value(indicator_name: str = Query(..., description="Name of the indicator"), year: int = Query(..., description="Year of the indicator"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT Value FROM Indicators WHERE IndicatorName = ? AND Year = ? AND CountryName = ?", (indicator_name, year, country_name))
    result = cursor.fetchone()
    if not result:
        return {"value": []}
    return {"value": result[0]}

# Endpoint to get the maximum value and year of a specific indicator for a given region
@app.get("/v1/world_development_indicators/max_indicator_value_by_region", operation_id="get_max_indicator_value_by_region", summary="Retrieves the maximum value and corresponding year of a specified indicator for a given region. The operation filters data based on the provided region and indicator name, returning the highest value and its associated year.")
async def get_max_indicator_value_by_region(region: str = Query(..., description="Region of the country"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT MAX(T1.value), T1.year FROM indicators AS T1 INNER JOIN country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Region = ? AND T1.IndicatorName = ?", (region, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"max_value": [], "year": []}
    return {"max_value": result[0], "year": result[1]}

# Endpoint to get distinct series codes for a specific indicator, country, and year
@app.get("/v1/world_development_indicators/distinct_series_codes", operation_id="get_distinct_series_codes", summary="Retrieve a unique list of series codes associated with a specific indicator, country, and year. This operation filters data based on the provided indicator name, country long name, and year, and returns the distinct series codes that match these criteria.")
async def get_distinct_series_codes(indicator_name: str = Query(..., description="Name of the indicator"), long_name: str = Query(..., description="Long name of the country"), year: int = Query(..., description="Year of the indicator")):
    cursor.execute("SELECT DISTINCT T3.Seriescode FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode INNER JOIN CountryNotes AS T3 ON T2.CountryCode = T3.Countrycode WHERE T2.IndicatorName = ? AND T1.LongName = ? AND T2.Year = ?", (indicator_name, long_name, year))
    result = cursor.fetchall()
    if not result:
        return {"series_codes": []}
    return {"series_codes": [row[0] for row in result]}

# Endpoint to get the count of distinct countries and the top country by indicator value for a specific region and national accounts system
@app.get("/v1/world_development_indicators/count_and_top_country_by_indicator", operation_id="get_count_and_top_country_by_indicator", summary="Get the count of distinct countries and the top country by indicator value for a specific region and national accounts system")
async def get_count_and_top_country_by_indicator(region: str = Query(..., description="Region of the country"), system_of_national_accounts: str = Query(..., description="System of National Accounts methodology"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT COUNT(DISTINCT T1.CountryCode) FROM indicators AS T1 INNER JOIN country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Region = ? AND T2.SystemOfNationalAccounts = ? AND T1.IndicatorName = ? UNION SELECT * FROM ( SELECT T1.CountryName FROM indicators AS T1 INNER JOIN country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Region = ? AND T2.SystemOfNationalAccounts = ? AND T1.IndicatorName = ? GROUP BY T1.CountryName ORDER BY SUM(T1.value) DESC LIMIT 1 )", (region, system_of_national_accounts, indicator_name, region, system_of_national_accounts, indicator_name))
    result = cursor.fetchall()
    if not result:
        return {"count": [], "top_country": []}
    return {"count": result[0][0], "top_country": result[1][0]}

# Endpoint to get the lending category for a specific indicator, value, and year
@app.get("/v1/world_development_indicators/lending_category", operation_id="get_lending_category", summary="Retrieves the lending category associated with a specific development indicator, its value, and the corresponding year. This operation allows users to obtain the lending category for a given indicator, value, and year, providing insights into the nature of the development indicator and its associated lending category.")
async def get_lending_category(indicator_name: str = Query(..., description="Name of the indicator"), value: int = Query(..., description="Value of the indicator"), year: int = Query(..., description="Year of the indicator")):
    cursor.execute("SELECT T1.LendingCategory FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.IndicatorName = ? AND T2.Value = ? AND T2.Year = ?", (indicator_name, value, year))
    result = cursor.fetchone()
    if not result:
        return {"lending_category": []}
    return {"lending_category": result[0]}

# Endpoint to get the country code and region for a specific indicator and year range
@app.get("/v1/world_development_indicators/country_region_by_indicator_year_range", operation_id="get_country_region_by_indicator_year_range", summary="Retrieves the country code and region of the highest-ranked country based on a specific development indicator within a given year range. The indicator name, minimum year, and maximum year are required as input parameters to filter the results.")
async def get_country_region_by_indicator_year_range(indicator_name: str = Query(..., description="Name of the indicator"), min_year: int = Query(..., description="Minimum year of the indicator"), max_year: int = Query(..., description="Maximum year of the indicator")):
    cursor.execute("SELECT T2.CountryCode, T2.Region FROM Indicators AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IndicatorName = ? AND T1.Year >= ? AND T1.Year < ? ORDER BY T2.Region DESC LIMIT 1", (indicator_name, min_year, max_year))
    result = cursor.fetchone()
    if not result:
        return {"country_code": [], "region": []}
    return {"country_code": result[0], "region": result[1]}

# Endpoint to get the country with the smallest land area for a specific indicator and year range
@app.get("/v1/world_development_indicators/smallest_land_area_country", operation_id="get_smallest_land_area_country", summary="Retrieves the country with the smallest land area for a specific indicator within a given year range. The operation calculates the total area for each country based on the provided indicator and year range, then returns the country with the smallest total area along with its income group.")
async def get_smallest_land_area_country(indicator_name: str = Query(..., description="Name of the indicator"), min_year: int = Query(..., description="Minimum year of the indicator"), max_year: int = Query(..., description="Maximum year of the indicator")):
    cursor.execute("SELECT T1.CountryName, SUM(T1.Value) area, T2.IncomeGroup FROM Indicators AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IndicatorName = ? AND T1.Year >= ? AND T1.Year < ? GROUP BY T1.CountryCode ORDER BY SUM(T1.Value) ASC LIMIT 1", (indicator_name, min_year, max_year))
    result = cursor.fetchone()
    if not result:
        return {"country_name": [], "area": [], "income_group": []}
    return {"country_name": result[0], "area": result[1], "income_group": result[2]}

# Endpoint to get the average value of a specific indicator for a given year range and country
@app.get("/v1/world_development_indicators/average_indicator_value", operation_id="get_average_indicator_value", summary="Retrieve the average value of a specific air transport indicator for a given year range and country. The operation calculates the average value of the specified indicator, which represents the number of passengers carried by air transport. The year range and country are used to filter the data, ensuring that the average is calculated only for the relevant years and country.")
async def get_average_indicator_value(indicator_name: str = Query(..., description="Name of the indicator"), min_year: int = Query(..., description="Minimum year of the indicator"), max_year: int = Query(..., description="Maximum year of the indicator"), country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT AVG(T1.Value), T2.SystemOfTrade FROM Indicators AS T1 INNER JOIN Country AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.IndicatorName = ? AND T1.Year >= ? AND T1.Year < ? AND T1.CountryName = ?", (indicator_name, min_year, max_year, country_name))
    result = cursor.fetchone()
    if not result:
        return {"average_value": [], "system_of_trade": []}
    return {"average_value": result[0], "system_of_trade": result[1]}

# Endpoint to get the latest household survey and PPP survey year for a given country
@app.get("/v1/world_development_indicators/latest_household_ppp_survey", operation_id="get_latest_household_ppp_survey", summary="Retrieves the most recent household survey year and purchasing power parity (PPP) survey year for a specified country. The country is identified using its short name.")
async def get_latest_household_ppp_survey(short_name: str = Query(..., description="Short name of the country")):
    cursor.execute("SELECT LatestHouseholdSurvey, PppSurveyYear FROM Country WHERE ShortName = ?", (short_name,))
    result = cursor.fetchone()
    if not result:
        return {"survey_data": []}
    return {"survey_data": result}

# Endpoint to get the count of countries with complete vital registration in a given region
@app.get("/v1/world_development_indicators/count_vital_registration_complete", operation_id="get_count_vital_registration_complete", summary="Retrieves the number of countries in a specified region that have complete vital registration. The operation considers the vital registration complete status and the region name as input parameters to filter the data.")
async def get_count_vital_registration_complete(vital_registration_complete: str = Query(..., description="Vital registration complete status"), region: str = Query(..., description="Region name")):
    cursor.execute("SELECT COUNT(CountryCode) FROM Country WHERE VitalRegistrationComplete = ? AND Region = ?", (vital_registration_complete, region))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the year for a specific indicator and country code
@app.get("/v1/world_development_indicators/indicator_year", operation_id="get_indicator_year", summary="Get the year for a specific indicator and country code")
async def get_indicator_year(alpha2_code: str = Query(..., description="Alpha-2 code of the country"), indicator_name: str = Query(..., description="Name of the indicator")):
    cursor.execute("SELECT T2.Year FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Alpha2Code = ? AND T2.IndicatorName = ?", (alpha2_code, indicator_name))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the long name of countries based on indicator name, year, and value
@app.get("/v1/world_development_indicators/country_long_name", operation_id="get_country_long_name", summary="Get the long name of countries based on indicator name, year, and value")
async def get_country_long_name(indicator_name: str = Query(..., description="Name of the indicator"), year: int = Query(..., description="Year of the indicator"), value: int = Query(..., description="Value of the indicator")):
    cursor.execute("SELECT T1.LongName FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.IndicatorName = ? AND T2.Year = ? AND T2.Value = ?", (indicator_name, year, value))
    result = cursor.fetchone()
    if not result:
        return {"long_name": []}
    return {"long_name": result[0]}

# Endpoint to get the Alpha-2 code of countries based on indicator name and year
@app.get("/v1/world_development_indicators/country_alpha2_code", operation_id="get_country_alpha2_code", summary="Get the Alpha-2 code of countries based on indicator name and year")
async def get_country_alpha2_code(indicator_name: str = Query(..., description="Name of the indicator"), year: int = Query(..., description="Year of the indicator")):
    cursor.execute("SELECT T1.Alpha2Code FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.IndicatorName = ? AND T2.Year = ?", (indicator_name, year))
    result = cursor.fetchone()
    if not result:
        return {"alpha2_code": []}
    return {"alpha2_code": result[0]}

# Endpoint to get distinct country codes and special notes based on the highest indicator value for a specific year
@app.get("/v1/world_development_indicators/distinct_country_special_notes", operation_id="get_distinct_country_special_notes", summary="Retrieves unique country codes and their respective special notes associated with the highest value of a specified indicator for a given year. The indicator and year are provided as input parameters.")
async def get_distinct_country_special_notes(indicator_name: str = Query(..., description="Name of the indicator"), year: int = Query(..., description="Year of the indicator")):
    cursor.execute("SELECT DISTINCT T1.CountryCode, T1.SpecialNotes FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T2.Value = ( SELECT Value FROM Indicators WHERE IndicatorName = ? AND Year = ? ORDER BY Value DESC LIMIT 1 )", (indicator_name, year))
    result = cursor.fetchall()
    if not result:
        return {"country_data": []}
    return {"country_data": result}

# Endpoint to get the difference in indicator values between two years for a specific country and indicator
@app.get("/v1/world_development_indicators/indicator_value_difference", operation_id="get_indicator_value_difference", summary="Retrieve the difference in the specified indicator value between two given years for a particular country. The indicator value is fetched from the database based on the provided country code, indicator name, and years. This operation is useful for comparing the indicator value across different years.")
async def get_indicator_value_difference(alpha2_code: str = Query(..., description="Alpha-2 code of the country"), indicator_name: str = Query(..., description="Name of the indicator"), year1: int = Query(..., description="First year"), year2: int = Query(..., description="Second year")):
    cursor.execute("SELECT ( SELECT T2.Value FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Alpha2Code = ? AND T2.IndicatorName = ? AND T2.Year = ? ) - ( SELECT T2.Value FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Alpha2Code = ? AND T2.IndicatorName = ? AND T2.Year = ? ) DIFF", (alpha2_code, indicator_name, year1, alpha2_code, indicator_name, year2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the description of country notes based on short name and topic
@app.get("/v1/world_development_indicators/country_notes_description_by_topic", operation_id="get_country_notes_description_by_topic", summary="Get the description of country notes based on short name and topic")
async def get_country_notes_description_by_topic(short_name: str = Query(..., description="Short name of the country"), topic: str = Query(..., description="Topic of the series")):
    cursor.execute("SELECT T2.Description FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode INNER JOIN Series AS T3 ON T2.Seriescode = T3.SeriesCode WHERE T1.ShortName = ? AND T3.Topic = ?", (short_name, topic))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the short name of countries based on series code
@app.get("/v1/world_development_indicators/country_short_name_by_series_code", operation_id="get_country_short_name_by_series_code", summary="Retrieves the short name of countries associated with a specific series code. The series code is used to filter the countries from the database, providing a concise list of country names relevant to the given series.")
async def get_country_short_name_by_series_code(series_code: str = Query(..., description="Series code")):
    cursor.execute("SELECT T1.ShortName FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode INNER JOIN Series AS T3 ON T2.Seriescode = T3.SeriesCode WHERE T3.Seriescode = ?", (series_code,))
    result = cursor.fetchall()
    if not result:
        return {"short_names": []}
    return {"short_names": result}

# Endpoint to get the region of countries based on a specific indicator name
@app.get("/v1/world_development_indicators/region_by_indicator", operation_id="get_region_by_indicator", summary="Retrieves the region of countries that match a specific indicator. The indicator is provided as an input parameter, allowing the user to filter the results based on the desired indicator.")
async def get_region_by_indicator(indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT T1.Region FROM Country AS T1 INNER JOIN CountryNotes AS T2 ON T1.CountryCode = T2.Countrycode INNER JOIN Series AS T3 ON T2.Seriescode = T3.SeriesCode WHERE T3.IndicatorName = ?", (indicator_name,))
    result = cursor.fetchall()
    if not result:
        return {"regions": []}
    return {"regions": [row[0] for row in result]}

# Endpoint to get the count of countries based on a specific indicator name
@app.get("/v1/world_development_indicators/count_countries_by_indicator", operation_id="get_count_countries_by_indicator", summary="Retrieves the number of countries that have data for a specific development indicator. The indicator is specified using the 'indicator_name' parameter. This operation provides a quantitative measure of the global coverage for the given indicator.")
async def get_count_countries_by_indicator(indicator_name: str = Query(..., description="Indicator name")):
    cursor.execute("SELECT COUNT(T1.Countrycode) FROM CountryNotes AS T1 INNER JOIN Series AS T2 ON T1.Seriescode = T2.SeriesCode WHERE T2.IndicatorName = ?", (indicator_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the aggregation method of indicators based on country short name, value, and year
@app.get("/v1/world_development_indicators/aggregation_method_by_country_value_year", operation_id="get_aggregation_method_by_country_value_year", summary="Retrieve the aggregation method used for a specific indicator based on the short name of the country, the value of the indicator, and the year it was recorded. This operation allows you to understand the methodology employed in aggregating the indicator data for a given country, value, and year.")
async def get_aggregation_method_by_country_value_year(short_name: str = Query(..., description="Short name of the country"), value: int = Query(..., description="Value of the indicator"), year: int = Query(..., description="Year of the indicator")):
    cursor.execute("SELECT T2.AggregationMethod FROM Indicators AS T1 INNER JOIN Series AS T2 ON T1.IndicatorName = T2.IndicatorName INNER JOIN Country AS T3 ON T1.CountryCode = T3.CountryCode WHERE T3.ShortName = ? AND T1.Value = ? AND T1.Year = ?", (short_name, value, year))
    result = cursor.fetchall()
    if not result:
        return {"aggregation_methods": []}
    return {"aggregation_methods": [row[0] for row in result]}

# Endpoint to get the value of indicators based on long definition, country short name, and year
@app.get("/v1/world_development_indicators/indicator_value_by_definition_country_year", operation_id="get_indicator_value_by_definition_country_year", summary="Retrieves the value of a specific indicator based on its long definition, the short name of a country, and the year. The indicator's long definition is used to identify the relevant data, while the country's short name and the year further narrow down the search. This operation provides a precise value for the specified indicator, country, and year.")
async def get_indicator_value_by_definition_country_year(long_definition: str = Query(..., description="Long definition of the indicator"), short_name: str = Query(..., description="Short name of the country"), year: int = Query(..., description="Year of the indicator")):
    cursor.execute("SELECT T1.Value FROM Indicators AS T1 INNER JOIN Series AS T2 ON T1.IndicatorName = T2.IndicatorName INNER JOIN Country AS T3 ON T1.CountryCode = T3.CountryCode WHERE T2.LongDefinition = ? AND T3.ShortName = ? AND T1.Year = ?", (long_definition, short_name, year))
    result = cursor.fetchall()
    if not result:
        return {"values": []}
    return {"values": [row[0] for row in result]}

# Endpoint to calculate the percentage change in indicator value between two years for a specific country
@app.get("/v1/world_development_indicators/percentage_change_indicator_value", operation_id="get_percentage_change_indicator_value", summary="Retrieves the percentage change in a specified indicator value between two given years for a particular country. The calculation involves subtracting the value of the first year from the second year, dividing the result by the sum of the values for both years, and then multiplying by 100 to obtain the percentage change.")
async def get_percentage_change_indicator_value(alpha2_code: str = Query(..., description="Alpha-2 code of the country"), indicator_name: str = Query(..., description="Indicator name"), year1: int = Query(..., description="First year"), year2: int = Query(..., description="Second year")):
    cursor.execute("SELECT (( SELECT T2.Value FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Alpha2Code = ? AND T2.IndicatorName = ? AND T2.Year = ? ) - ( SELECT T2.Value FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Alpha2Code = ? AND T2.IndicatorName = ? AND T2.Year = ? )) * 1.0 / ( SELECT SUM(T2.Value) FROM Country AS T1 INNER JOIN Indicators AS T2 ON T1.CountryCode = T2.CountryCode WHERE T1.Alpha2Code = ? AND T2.IndicatorName = ? AND T2.Year = ? )", (alpha2_code, indicator_name, year2, alpha2_code, indicator_name, year1, alpha2_code, indicator_name, year1))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result[0]}

api_calls = [
    "/v1/world_development_indicators/count_countries_lending_other_groups?lending_category=IDA&other_groups=HIPC",
    "/v1/world_development_indicators/country_short_name_debt_status?lending_category=IDA",
    "/v1/world_development_indicators/country_notes_description?short_name=Aruba&series_code=SM.POP.TOTL",
    "/v1/world_development_indicators/country_short_name_notes_description?region=Latin%20America%20%26%20Caribbean&series_code=SM.POP.TOTL",
    "/v1/world_development_indicators/count_countries_series_income?series_code=SM.POP.TOTL&income_group=Low%20income",
    "/v1/world_development_indicators/country_notes_description_lending?lending_category=IDA&series_code=SM.POP.TOTL",
    "/v1/world_development_indicators/count_countries_lending_series_income?lending_category=IDA&series_code=SM.POP.TOTL&income_group=Low%20income",
    "/v1/world_development_indicators/count_countries_income_currency_series?income_group=High%20income%3A%20OECD&currency_unit=Euro&series_code=SP.DYN.AMRT.FE",
    "/v1/world_development_indicators/country_long_name_description_series?description=Estimates%20are%20derived%20FROM%20data%20on%20foreign-born%20population.&series_code=SM.POP.TOTL",
    "/v1/world_development_indicators/footnotes_description?short_name=Aruba&series_code=AG.LND.FRST.K2&year=YR1990",
    "/v1/world_development_indicators/years_by_country_series?short_name=Aruba&series_code=AG.LND.FRST.K2",
    "/v1/world_development_indicators/country_short_names?description=Data%20are%20classified%20as%20official%20aid.&series_code=DC.DAC.AUSL.CD&year_pattern=%252002%25",
    "/v1/world_development_indicators/count_series_codes?short_name=Aruba&year=YR2002",
    "/v1/world_development_indicators/count_years_by_country_series?short_name=Aruba&series_code=BX.KLT.DINV.CD.WD",
    "/v1/world_development_indicators/avg_value_by_country_indicator?alpha2_code=1A&indicator_name_pattern=adolescent%20fertility%20rate%25",
    "/v1/world_development_indicators/special_notes_highest_indicator?indicator_name_pattern=Adolescent%20fertility%20rate%25",
    "/v1/world_development_indicators/country_codes_by_region_income?region=East%20Asia%20%26%20Pacific&income_group=High%20income%3A%20nonOECD",
    "/v1/world_development_indicators/country_names_by_data_years?latest_trade_data=2013&latest_water_withdrawal_data=2013",
    "/v1/world_development_indicators/avg_indicator_value?country_name=Algeria&min_year=1974&max_year=1981&indicator_name=Adjusted%20net%20enrolment%20rate%2C%20primary%2C%20both%20sexes%20(%25)",
    "/v1/world_development_indicators/indicator_aggregation_by_topic?topic=Economic%20Policy%20%26%20Debt%3A%20Balance%20of%20payments%3A%20Capital%20%26%20financial%20account",
    "/v1/world_development_indicators/series_codes_by_topic_license?topic=Environment%3A%20Emissions&license_type=Restricted",
    "/v1/world_development_indicators/count_countries_by_region_indicator_year_value?region=Middle%20East%20%26%20North%20Africa&indicator_name=CO2%20emissions%20FROM%20gaseous%20fuel%20consumption%20(kt)&year=1970&value=600",
    "/v1/world_development_indicators/distinct_country_codes_years_values?region=Latin%20America%20%26%20Caribbean&indicator_name=CO2%20emissions%20(kt)&min_year=1965&max_year=1980",
    "/v1/world_development_indicators/series_descriptions?series_code=SP.DYN.TO65.MA.ZS&topic=Health%3A%20Mortality&year=YR1967",
    "/v1/world_development_indicators/distinct_footnote_descriptions?year=YR1981&short_name=Albania",
    "/v1/world_development_indicators/distinct_footnote_descriptions_by_year_country_series?year=YR1984&short_name=The%20Bahamas&series_code=SH.DTH.IMRT",
    "/v1/world_development_indicators/distinct_wb2_codes_lending_categories?description=Data%20source%20%3A%20Human%20Mortality%20Database%20by%20University%20of%20California%2C%20Berkeley%2C%20and%20Max%20Planck%20Institute%20for%20Demographic%20Research.&lending_category=",
    "/v1/world_development_indicators/series_details?year=1961&country_name=Haiti&indicator_name=Total%20reserves%20minus%20gold%20(current%20US%24)&value=3000000",
    "/v1/world_development_indicators/count_countries_by_indicator_status_value?indicator_name=Adjusted%20net%20national%20income%20per%20capita%20(constant%202005%20US%24)&external_debt_reporting_status=Preliminary&value=1000",
    "/v1/world_development_indicators/country_names_by_year_indicator_value_range?year=1979&indicator_name=Fertility%20rate%2C%20total%20(births%20per%20woman)&min_value=4&max_value=5",
    "/v1/world_development_indicators/table_names_and_source_by_region_income_group?region=South%20Asia&income_group=Low%20income",
    "/v1/world_development_indicators/distinct_sources_by_region_indicator?region=Latin%20America%20%26%20Caribbean&indicator_name=Children%20out%20of%20school%2C%20primary",
    "/v1/world_development_indicators/sources_by_year_pattern_indicator?year_pattern=%252002%25&indicator_name=Net%20migration",
    "/v1/world_development_indicators/distinct_descriptions_by_region_indicator?region=North%20America&indicator_name=Out-of-school%20children%20of%20primary%20school%20age%2C%20both%20sexes%20(number)",
    "/v1/world_development_indicators/country_codes_values_by_trade_data_indicator_year_min_value?latest_trade_data=2013&indicator_name_pattern=GDP%20growth%20(annual%20%25)&year=2014&min_value=0",
    "/v1/world_development_indicators/country_codes_descriptions_by_indicator_min_value_year_limit?indicator_name=Out-of-pocket%20health%20expenditure%20(%20of%20private%20expenditure%20on%20health)&min_value=0&year=2005&limit=10",
    "/v1/world_development_indicators/short_names_by_latest_trade_data?latest_trade_data=2010",
    "/v1/world_development_indicators/percentage_countries_by_region_system_of_trade?region=Sub-Saharan%20Africa&system_of_trade=Special%20trade%20system",
    "/v1/world_development_indicators/average_indicator_value_by_region?region=Europe%20%26%20Central%20Asia&indicator_name=Arms%20imports%20(SIPRI%20trend%20indicator%20values)",
    "/v1/world_development_indicators/country_with_lowest_indicator_value?income_group=Upper%20middle%20income&indicator_name=CO2%20emissions%20(kt)",
    "/v1/world_development_indicators/min_indicator_value_by_group?other_groups=HIPC&indicator_name=International%20migrant%20stock%2C%20total",
    "/v1/world_development_indicators/distinct_indicator_names?year=YR2000&topic=Education%3A%20Inputs",
    "/v1/world_development_indicators/count_distinct_series_codes?year1=YR2001&year2=YR2002&year3=YR2003&periodicity=Annual&aggregation_method=Sum",
    "/v1/world_development_indicators/series_codes_and_sources?year_pattern=%252005%25&source_pattern=International%20Monetary%20Fund%25",
    "/v1/world_development_indicators/percentage_above_threshold?region=South%20Asia&indicator_name=Life%20expectancy%20at%20birth%2C%20female%20(years)&threshold=50",
    "/v1/world_development_indicators/country_with_highest_indicator_value?start_year=1960&end_year=1965&indicator_name=Death%20rate%2C%20crude%20(per%201%2C000%20people)",
    "/v1/world_development_indicators/indicator_names_by_country_year_value?country_name=Arab%20World&year=1960&value_threshold=50",
    "/v1/world_development_indicators/country_with_highest_indicator_value_single?indicator_name=Merchandise%20imports%20by%20the%20reporting%20economy%20(current%20US%24)",
    "/v1/world_development_indicators/distinct_indicator_names_year_range?start_year=1968&end_year=1971&license_type=Open&value=100",
    "/v1/world_development_indicators/top_country_indicator?topic=Private%20Sector%20%26%20Trade:%20Exports",
    "/v1/world_development_indicators/min_indicator_value?aggregation_method=Weighted%20average",
    "/v1/world_development_indicators/indicator_names_country_year?country_name=Sudan&year=1961&periodicity=Annual",
    "/v1/world_development_indicators/country_min_indicator_value?start_year=1960&end_year=1966&topic=Health:%20Population:%20Structure",
    "/v1/world_development_indicators/percentage_countries_indicator_value?indicator_name=CO2%20emissions%20FROM%20liquid%20fuel%20consumption%20(%20of%20total)&value_threshold=80&income_group=Upper%20middle%20income",
    "/v1/world_development_indicators/distinct_indicator_codes?indicator_name=Rural%20population",
    "/v1/world_development_indicators/table_names_national_accounts?system_of_national_accounts=Country%20uses%20the%202008%20System%20of%20National%20Accounts%20methodology.",
    "/v1/world_development_indicators/distinct_series_codes_currency?currency_unit=Euro",
    "/v1/world_development_indicators/country_long_name_by_series_code?series_code=DT.DOD.DSTC.CD",
    "/v1/world_development_indicators/series_codes_by_currency_unit?currency_unit=Hong%20Kong%20dollar",
    "/v1/world_development_indicators/table_names_by_series_code?series_code=SP.DYN.TO65.MA.ZS",
    "/v1/world_development_indicators/distinct_country_names_by_income_group?income_group=Low%20income",
    "/v1/world_development_indicators/series_country_codes_by_region_income_group?region=Latin%20America%20%26%20Caribbean&income_group=Low%20income",
    "/v1/world_development_indicators/country_series_codes_by_currency_income_group?currency_unit=Australian%20dollar&income_group=Lower%20middle%20income",
    "/v1/world_development_indicators/distinct_country_names_by_income_group_earliest_base_year?income_group=Upper%20middle%20income",
    "/v1/world_development_indicators/distinct_country_codes_names_by_currency_income_groups?currency_unit=Euro&income_group_1=High%20income:%20OECD&income_group_2=High%20income:%20nonOECD",
    "/v1/world_development_indicators/table_names_currency_units_by_series_code?series_code=FP.CPI.TOTL",
    "/v1/world_development_indicators/distinct_country_codes_names_by_income_group?income_group=High%20income:%20nonOECD",
    "/v1/world_development_indicators/country_details_currency_income?currency_unit=Pound%20sterling&income_group=%25high%20income%25",
    "/v1/world_development_indicators/limited_country_codes?limit=10",
    "/v1/world_development_indicators/count_countries_alpha2_codes?short_name_prefix=A%25",
    "/v1/world_development_indicators/table_names_currency_unit?currency_unit=Euro",
    "/v1/world_development_indicators/count_countries_no_special_notes",
    "/v1/world_development_indicators/country_codes_regions_income_group?income_group1=High%20income:%20OECD&income_group2=High%20income:%20nonOECD&region=%25Asia%25",
    "/v1/world_development_indicators/long_names_national_accounts_base_year?base_year=1980",
    "/v1/world_development_indicators/country_codes_income_group_series_code?income_group=Low%20income&series_code=DT.DOD.DECT.CD",
    "/v1/world_development_indicators/distinct_table_names_description?description=Covers%20mainland%20Tanzania%20only.",
    "/v1/world_development_indicators/descriptions_series_codes_table_name_year?table_name=Benin&year=YR2005",
    "/v1/world_development_indicators/distinct_years_table_names_by_indicator?indicator_name=Air%20transport,%20passengers%20carried",
    "/v1/world_development_indicators/distinct_long_names_by_year?year=1980",
    "/v1/world_development_indicators/currency_units_indicator_codes_by_country_year?table_name=Malaysia&year=1970",
    "/v1/world_development_indicators/top_5_country_codes_regions",
    "/v1/world_development_indicators/count_distinct_country_codes_currency_units?description=Sources:%20UN%20Energy%20Statistics%20(2014)",
    "/v1/world_development_indicators/count_descriptions_distinct_table_names_by_year?year=YR1980",
    "/v1/world_development_indicators/series_codes_wb2_codes_by_description?description=Data%20sources%20:%20Eurostat",
    "/v1/world_development_indicators/top_country_by_income_group_indicator?income_group=Low%20income&indicator_name=Adolescent%20fertility%20rate%20(births%20per%201,000%20women%20ages%2015-19)",
    "/v1/world_development_indicators/sum_indicator_values_by_income_group_year?income_group=%25middle%20income&year=1960&indicator_name=Urban%20population",
    "/v1/world_development_indicators/country_currency_by_indicator_year?indicator_name=Adjusted%20net%20national%20income%20(annual%20%25%20growth)&year=1980",
    "/v1/world_development_indicators/count_countries_by_national_accounts?system_of_national_accounts=Country%20uses%20the%201993%20System%20of%20National%20Accounts%20methodology.",
    "/v1/world_development_indicators/countries_with_complete_vital_registration?vital_registration_complete=Yes",
    "/v1/world_development_indicators/countries_by_census_range_vital_registration?min_census_year=2010&max_census_year=2013&vital_registration_complete=Yes",
    "/v1/world_development_indicators/top_country_by_indicator_year?year=1960&indicator_name=CO2%20emissions%20(metric%20tons%20per%20capita)",
    "/v1/world_development_indicators/ratio_max_min_values_indicator_year?indicator_name=Number%20of%20infant%20deaths&year=1971",
    "/v1/world_development_indicators/countries_by_series_code?series_code=SP.DYN.CBRT.IN",
    "/v1/world_development_indicators/distinct_country_codes_by_description?description=Data%20are%20classified%20as%20official%20aid.",
    "/v1/world_development_indicators/distinct_short_names_by_description?description=Estimates%20are%20based%20on%20regression.",
    "/v1/world_development_indicators/distinct_country_codes_by_year_region_value_indicator?year=1970&region=East%20Asia%20%26%20Pacific&min_value=2000000&indicator_name=Urban%20population",
    "/v1/world_development_indicators/max_indicator_value?income_group=Upper%20middle%20income&year=1960&indicator_name=Population%2C%20total",
    "/v1/world_development_indicators/distinct_short_names?income_group=Upper%20middle%20income&region=East%20Asia%20%26%20Pacific&topic=Social%20Protection%20%26%20Labor%3A%20Migration",
    "/v1/world_development_indicators/distinct_long_names_by_topic?topic=Poverty%3A%20Shared%20prosperity",
    "/v1/world_development_indicators/distinct_table_names?latest_trade_data=2013&indicator_code=SP.DYN.CDRT.IN",
    "/v1/world_development_indicators/distinct_long_names_by_region_series?region=Sub-Saharan%20Africa&series_code=SP.DYN.AMRT.FE",
    "/v1/world_development_indicators/short_names_by_currency?currency_unit=U.S.%20dollar&limit=3",
    "/v1/world_development_indicators/long_names_by_income_region?income_group=Low%20income&region=South%20Asia",
    "/v1/world_development_indicators/long_names_by_trade_system?system_of_trade=Special%20trade%20system&limit=2",
    "/v1/world_development_indicators/country_names_by_indicator?indicator_name_pattern=CO2%20emissions%20FROM%20transport%25&limit=1",
    "/v1/world_development_indicators/top_country_by_indicator?indicator_name=Arable%20land%20(%20of%20land%20area)",
    "/v1/world_development_indicators/distinct_topics_by_license_type?license_type=Restricted",
    "/v1/world_development_indicators/country_codes_by_income_group_and_debt_status?income_group=Upper%20middle%20income&external_debt_reporting_status=Preliminary",
    "/v1/world_development_indicators/percentage_countries_by_debt_status_and_region?external_debt_reporting_status=Actual&region=Middle%20East%20%26%20North%20Africa",
    "/v1/world_development_indicators/count_countries_by_accounting_concepts_and_debt_status?government_accounting_concept_1=Budgetary%20central%20government&government_accounting_concept_2=Consolidated%20central%20government&external_debt_reporting_status=Actual",
    "/v1/world_development_indicators/count_countries_by_region_and_debt_status?region=East%20Asia%20%26%20Pacific&external_debt_reporting_status=Estimate",
    "/v1/world_development_indicators/percentage_countries_by_income_group_and_region?income_group=Lower%20middle%20income&region=Sub-Saharan%20Africa",
    "/v1/world_development_indicators/max_indicator_value_by_country_and_year_range?start_year=1961&end_year=1981&indicator_name=Agricultural%20land%20(%20of%20land%20area)&country_long_name=Republic%20of%20Benin",
    "/v1/world_development_indicators/distinct_country_long_names_by_description?description=Sources:%20UN%20Energy%20Statistics%20(2014)",
    "/v1/world_development_indicators/indicator_value_by_country_and_year?country_long_name=Commonwealth%20of%20Australia&indicator_name=Deposit%20interest%20rate%20(%)&year=1979",
    "/v1/world_development_indicators/series_description_by_country?short_name=Germany",
    "/v1/world_development_indicators/topics_by_country?short_name=Austria",
    "/v1/world_development_indicators/topics_description_by_series?series_code=SP.DYN.AMRT.MA",
    "/v1/world_development_indicators/count_countries_by_footnote?description1=Unspecified&description2=Not%20specified",
    "/v1/world_development_indicators/country_codes_by_indicator?indicator_name=Land%20under%20cereal%20production%25&value=3018500&year=1980&external_debt_status=Actual",
    "/v1/world_development_indicators/percentage_countries_above_threshold?value_threshold=50&year=1961&region=Latin%20America%20%26%20Caribbean&indicator_name=Agricultural%20land%20(%20of%20land%20area)",
    "/v1/world_development_indicators/indicator_codes_by_country_indicator?country_name=Brazil&indicator_name=Mobile%20cellular%20subscriptions",
    "/v1/world_development_indicators/count_countries_by_region_currency?region=Europe%20%26%20Central%20Asia&currency_unit=Danish%20krone",
    "/v1/world_development_indicators/max_value_by_indicator?indicator_name=Rural%20population%20(%20of%20total%20population)",
    "/v1/world_development_indicators/count_countries_by_census_year?census_year=2011",
    "/v1/world_development_indicators/indicator_value?indicator_name=Agricultural%20land%20(sq.%20km)&year=1968&country_name=Italy",
    "/v1/world_development_indicators/max_indicator_value_by_region?region=Sub-Saharan%20Africa&indicator_name=Out-of-school%20children%20of%20primary%20school%20age,%20female%20(number)",
    "/v1/world_development_indicators/distinct_series_codes?indicator_name=Number%20of%20infant%20deaths&long_name=Islamic%20State%20of%20Afghanistan&year=1965",
    "/v1/world_development_indicators/count_and_top_country_by_indicator?region=Middle%20East%20%26%20North%20Africa&system_of_national_accounts=Country%20uses%20the%201968%20System%20of%20National%20Accounts%20methodology.&indicator_name=CO2%20emissions%20FROM%20solid%20fuel%20consumption%20(kt)",
    "/v1/world_development_indicators/lending_category?indicator_name=Cereal%20production%20(metric%20tons)&value=6140000&year=1966",
    "/v1/world_development_indicators/country_region_by_indicator_year_range?indicator_name=Population%20in%20largest%20city&min_year=1960&max_year=1980",
    "/v1/world_development_indicators/smallest_land_area_country?indicator_name=Land%20area%20(sq.%20km)&min_year=1961&max_year=1980",
    "/v1/world_development_indicators/average_indicator_value?indicator_name=Air%20transport,%20passengers%20carried&min_year=1970&max_year=1981&country_name=Bulgaria",
    "/v1/world_development_indicators/latest_household_ppp_survey?short_name=Angola",
    "/v1/world_development_indicators/count_vital_registration_complete?vital_registration_complete=Yes&region=North%20America",
    "/v1/world_development_indicators/indicator_year?alpha2_code=1A&indicator_name=Adolescent%20fertility%20rate%20(births%20per%201%2C000%20women%20ages%2015-19)",
    "/v1/world_development_indicators/country_long_name?indicator_name=Arms%20exports%20(SIPRI%20trend%20indicator%20values)&year=1960&value=3000000",
    "/v1/world_development_indicators/country_alpha2_code?indicator_name=Rural%20population&year=1960",
    "/v1/world_development_indicators/distinct_country_special_notes?indicator_name=Adolescent%20fertility%20rate%20(births%20per%201%2C000%20women%20ages%2015-19)&year=1960",
    "/v1/world_development_indicators/indicator_value_difference?alpha2_code=1A&indicator_name=Adolescent%20fertility%20rate%20(births%20per%201%2C000%20women%20ages%2015-19)&year1=1961&year2=1960",
    "/v1/world_development_indicators/country_notes_description_by_topic?short_name=Aruba&topic=Environment:%20Energy%20production%20&%20use",
    "/v1/world_development_indicators/country_short_name_by_series_code?series_code=BX.KLT.DINV.CD.WD",
    "/v1/world_development_indicators/region_by_indicator?indicator_name=Inflation%2C%20consumer%20prices%20(annual%20%25)",
    "/v1/world_development_indicators/count_countries_by_indicator?indicator_name=Stocks%20traded%2C%20turnover%20ratio%20of%20domestic%20shares%20(%25)",
    "/v1/world_development_indicators/aggregation_method_by_country_value_year?short_name=Arab%20World&value=133&year=1960",
    "/v1/world_development_indicators/indicator_value_by_definition_country_year?long_definition=Adolescent%20fertility%20rate%20is%20the%20number%20of%20births%20per%201%2C000%20women%20ages%2015-19.&short_name=Arab%20World&year=1960",
    "/v1/world_development_indicators/percentage_change_indicator_value?alpha2_code=1A&indicator_name=Adolescent%20fertility%20rate%20(births%20per%201%2C000%20women%20ages%2015-19)&year1=1960&year2=1961"
]
