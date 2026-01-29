from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/chicago_crime/chicago_crime.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of community areas based on side
@app.get("/v1/chicago_crime/community_area_count_by_side", operation_id="get_community_area_count_by_side", summary="Retrieves the total number of community areas located on the specified side. The side parameter is used to filter the results.")
async def get_community_area_count_by_side(side: str = Query(..., description="Side of the community area")):
    cursor.execute("SELECT COUNT(*) FROM Community_Area WHERE side = ?", (side,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the side of a community area based on its name
@app.get("/v1/chicago_crime/community_area_side_by_name", operation_id="get_community_area_side_by_name", summary="Retrieves the side of a specific community area by its name. The community area name is provided as an input parameter.")
async def get_community_area_side_by_name(community_area_name: str = Query(..., description="Name of the community area")):
    cursor.execute("SELECT side FROM Community_Area WHERE community_area_name = ?", (community_area_name,))
    result = cursor.fetchone()
    if not result:
        return {"side": []}
    return {"side": result[0]}

# Endpoint to get the most common side of community areas
@app.get("/v1/chicago_crime/most_common_community_area_side", operation_id="get_most_common_community_area_side", summary="Retrieves the most frequently occurring side of community areas in Chicago. The side is determined by the highest count of occurrences in the Community_Area dataset.")
async def get_most_common_community_area_side():
    cursor.execute("SELECT side FROM Community_Area GROUP BY side ORDER BY COUNT(side) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"side": []}
    return {"side": result[0]}

# Endpoint to get the community area with the smallest population
@app.get("/v1/chicago_crime/smallest_population_community_area", operation_id="get_smallest_population_community_area", summary="Retrieves the name of the community area with the smallest population in Chicago. This endpoint provides a quick and efficient way to identify the least populated community area based on the latest data.")
async def get_smallest_population_community_area():
    cursor.execute("SELECT community_area_name FROM Community_Area ORDER BY population ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"community_area_name": []}
    return {"community_area_name": result[0]}

# Endpoint to get the commander of a district based on its name
@app.get("/v1/chicago_crime/district_commander_by_name", operation_id="get_district_commander_by_name", summary="Retrieves the commander of a specified district. The district is identified by its name, which is provided as an input parameter.")
async def get_district_commander_by_name(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT commander FROM District WHERE district_name = ?", (district_name,))
    result = cursor.fetchone()
    if not result:
        return {"commander": []}
    return {"commander": result[0]}

# Endpoint to get the email of a district based on its name
@app.get("/v1/chicago_crime/district_email_by_name", operation_id="get_district_email_by_name", summary="Retrieves the email address associated with the specified district. The district is identified by its unique name.")
async def get_district_email_by_name(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT email FROM District WHERE district_name = ?", (district_name,))
    result = cursor.fetchone()
    if not result:
        return {"email": []}
    return {"email": result[0]}

# Endpoint to get the community area name based on neighborhood name
@app.get("/v1/chicago_crime/community_area_name_by_neighborhood", operation_id="get_community_area_name_by_neighborhood", summary="Retrieves the name of the community area associated with the specified neighborhood. The operation uses the provided neighborhood name to look up the corresponding community area name in the database.")
async def get_community_area_name_by_neighborhood(neighborhood_name: str = Query(..., description="Name of the neighborhood")):
    cursor.execute("SELECT T2.community_area_name FROM Neighborhood AS T1 INNER JOIN Community_Area AS T2 ON T1.community_area_no = T2.community_area_no WHERE T1.neighborhood_name = ?", (neighborhood_name,))
    result = cursor.fetchone()
    if not result:
        return {"community_area_name": []}
    return {"community_area_name": result[0]}

# Endpoint to get the count of community area numbers based on community area name
@app.get("/v1/chicago_crime/count_community_area_numbers_by_name", operation_id="get_count_community_area_numbers_by_name", summary="Retrieves the total number of community areas that match the provided community area name. The operation filters community areas based on the given name and aggregates the count of unique community area numbers.")
async def get_count_community_area_numbers_by_name(community_area_name: str = Query(..., description="Name of the community area")):
    cursor.execute("SELECT COUNT(T3.community_area_no) FROM ( SELECT T1.community_area_no FROM Community_Area AS T1 INNER JOIN Neighborhood AS T2 ON T1.community_area_no = T2.community_area_no WHERE community_area_name = ? GROUP BY T1.community_area_no ) T3", (community_area_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the neighborhood with the highest population
@app.get("/v1/chicago_crime/highest_population_neighborhood", operation_id="get_highest_population_neighborhood", summary="Get the neighborhood with the highest population")
async def get_highest_population_neighborhood():
    cursor.execute("SELECT T1.neighborhood_name FROM Neighborhood AS T1 INNER JOIN Community_Area AS T2 ON T2.community_area_no = T1.community_area_no ORDER BY T2.population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"neighborhood_name": []}
    return {"neighborhood_name": result[0]}

# Endpoint to get neighborhood names based on the side of the community area
@app.get("/v1/chicago_crime/neighborhood_names_by_side", operation_id="get_neighborhood_names_by_side", summary="Retrieves the names of neighborhoods located on the specified side of a community area. The side is a parameter that determines the neighborhoods to be returned.")
async def get_neighborhood_names_by_side(side: str = Query(..., description="Side of the community area")):
    cursor.execute("SELECT T2.neighborhood_name FROM Community_Area AS T1 INNER JOIN Neighborhood AS T2 ON T1.community_area_no = T2.community_area_no WHERE T1.side = ?", (side,))
    result = cursor.fetchall()
    if not result:
        return {"neighborhood_names": []}
    return {"neighborhood_names": [row[0] for row in result]}

# Endpoint to get the latitude and longitude of crimes in a specific district
@app.get("/v1/chicago_crime/crime_locations_by_district", operation_id="get_crime_locations_by_district", summary="Retrieve the geographical coordinates of crimes committed in a specified district. The operation returns a list of latitude and longitude pairs, each representing the location of a crime. The district is identified by its name.")
async def get_crime_locations_by_district(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT T2.latitude, T2.longitude FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no WHERE T1.district_name = ?", (district_name,))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": result}

# Endpoint to get the count of crimes in a specific district
@app.get("/v1/chicago_crime/crime_count_by_district", operation_id="get_crime_count_by_district", summary="Retrieves the total number of crimes that occurred in a specified district. The district is identified by its name.")
async def get_crime_count_by_district(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT COUNT(*) FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no WHERE T2.district_name = ?", (district_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of domestic crimes in a specific district
@app.get("/v1/chicago_crime/domestic_crime_count_by_district", operation_id="get_domestic_crime_count_by_district", summary="Retrieves the total number of domestic crimes in a specified district. The operation requires the district's name and a boolean value indicating whether the crime is domestic or not.")
async def get_domestic_crime_count_by_district(district_name: str = Query(..., description="Name of the district"), domestic: str = Query(..., description="Domestic status (TRUE or FALSE)")):
    cursor.execute("SELECT COUNT(*) FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no WHERE T2.district_name = ? AND T1.domestic = ?", (district_name, domestic))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of crimes with no arrest in a specific district
@app.get("/v1/chicago_crime/no_arrest_crime_count_by_district", operation_id="get_no_arrest_crime_count_by_district", summary="Retrieves the total number of crimes with no arrest in a specified district. The operation requires the district's name and the arrest status, which should be set to FALSE to filter for crimes without arrests.")
async def get_no_arrest_crime_count_by_district(district_name: str = Query(..., description="Name of the district"), arrest: str = Query(..., description="Arrest status (TRUE or FALSE)")):
    cursor.execute("SELECT COUNT(*) FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no WHERE T2.district_name = ? AND T1.arrest = ?", (district_name, arrest))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of crime reports in the most populous community area
@app.get("/v1/chicago_crime/crime_report_count_most_populous_community", operation_id="get_crime_report_count_most_populous_community", summary="Retrieves the total number of crime reports in the community area with the highest population. The operation calculates the count of crime reports by joining the Community_Area and Crime tables on the community area number. The result is then ordered by population in descending order and limited to the top record.")
async def get_crime_report_count_most_populous_community():
    cursor.execute("SELECT COUNT(T2.report_no) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no GROUP BY T1.community_area_name ORDER BY T1.population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of crimes in a specific community area for a given date pattern
@app.get("/v1/chicago_crime/crime_count_by_community_area_date", operation_id="get_crime_count_by_community_area_date", summary="Retrieve the total number of crimes that occurred in a specific community area during a given month and year. The operation requires the name of the community area and a date pattern in 'MM/YYYY' format.")
async def get_crime_count_by_community_area_date(community_area_name: str = Query(..., description="Name of the community area"), date_pattern: str = Query(..., description="Date pattern in '%%MM/YYYY%%' format")):
    cursor.execute("SELECT SUM(CASE WHEN T1.community_area_name = ? THEN 1 ELSE 0 END) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no WHERE T2.date LIKE ?", (community_area_name, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the community area with the highest crime count for a given date pattern
@app.get("/v1/chicago_crime/highest_crime_community_area_by_date", operation_id="get_highest_crime_community_area_by_date", summary="Retrieves the community area with the highest crime count for a specified date pattern, from a list of provided community area names. The date pattern should be in the format 'MM/YYYY'.")
async def get_highest_crime_community_area_by_date(community_area_name1: str = Query(..., description="First community area name"), community_area_name2: str = Query(..., description="Second community area name"), date_pattern: str = Query(..., description="Date pattern in '%%MM/YYYY%%' format")):
    cursor.execute("SELECT T1.community_area_name FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no WHERE T1.community_area_name IN (?, ?) AND T2.date LIKE ? GROUP BY T1.community_area_name ORDER BY COUNT(T1.community_area_name) DESC LIMIT 1", (community_area_name1, community_area_name2, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"community_area": []}
    return {"community_area": result[0]}

# Endpoint to get the fax number of the district with the highest crime count for a given date pattern
@app.get("/v1/chicago_crime/highest_crime_district_fax_by_date", operation_id="get_highest_crime_district_fax_by_date", summary="Retrieves the fax number of the district with the highest crime count for a specified month and year. The operation filters crimes by the provided date pattern, groups them by district, and orders the results by the count of case numbers in descending order. The fax number of the district with the highest count is then returned.")
async def get_highest_crime_district_fax_by_date(date_pattern: str = Query(..., description="Date pattern in '%%MM/YYYY%%' format")):
    cursor.execute("SELECT T1.fax FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no WHERE T2.date LIKE ? GROUP BY T2.district_no ORDER BY COUNT(case_number) DESC LIMIT 1", (date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"fax": []}
    return {"fax": result[0]}

# Endpoint to get the ratio of crime reports to community areas on a specific side of the city
@app.get("/v1/chicago_crime/crime_report_ratio_by_side", operation_id="get_crime_report_ratio_by_side", summary="Retrieves the proportion of crime reports to community areas located on a specified side of the city. The side parameter is used to filter the results.")
async def get_crime_report_ratio_by_side(side: str = Query(..., description="Side of the city (e.g., Central)")):
    cursor.execute("SELECT CAST(COUNT(T1.report_no) AS REAL) / COUNT(T2.community_area_no) FROM Crime AS T1 INNER JOIN Community_Area AS T2 ON T1.community_area_no = T2.community_area_no WHERE T2.side = ?", (side,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the percentage of crimes in a specific district
@app.get("/v1/chicago_crime/crime_percentage_by_district", operation_id="get_crime_percentage_by_district", summary="Retrieves the percentage of total crimes that occurred in a specific district. The district is identified by its name, which is used to calculate the proportion of crimes in that district relative to the total number of crimes.")
async def get_crime_percentage_by_district(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.district_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.case_number) FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no", (district_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the district name based on case number
@app.get("/v1/chicago_crime/district_name_by_case_number", operation_id="get_district_name", summary="Retrieves the name of the district associated with the provided case number. This operation fetches the district name from the District table, which is linked to the case number in the Crime table via the district number.")
async def get_district_name(case_number: str = Query(..., description="Case number")):
    cursor.execute("SELECT T1.district_name FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no WHERE T2.case_number = ?", (case_number,))
    result = cursor.fetchone()
    if not result:
        return {"district_name": []}
    return {"district_name": result[0]}

# Endpoint to get the district name based on longitude and latitude
@app.get("/v1/chicago_crime/district_name_by_coordinates", operation_id="get_district_name_by_coordinates", summary="Retrieves the name of the district associated with the provided longitude and latitude coordinates. This operation identifies the district by cross-referencing the input coordinates with the longitude and latitude of recorded crimes, then retrieves the corresponding district name from the related district table.")
async def get_district_name_by_coordinates(longitude: str = Query(..., description="Longitude of the crime location"), latitude: str = Query(..., description="Latitude of the crime location")):
    cursor.execute("SELECT T2.district_name FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no WHERE T1.longitude = ? AND T1.latitude = ?", (longitude, latitude))
    result = cursor.fetchone()
    if not result:
        return {"district_name": []}
    return {"district_name": result[0]}

# Endpoint to get the commander of a district based on case number
@app.get("/v1/chicago_crime/district_commander_by_case_number", operation_id="get_district_commander_by_case_number", summary="Retrieves the district commander associated with the provided case number. This operation fetches the commander's details by matching the case number with the corresponding district's crime records.")
async def get_district_commander_by_case_number(case_number: str = Query(..., description="Case number")):
    cursor.execute("SELECT T1.commander FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no WHERE T2.case_number = ?", (case_number,))
    result = cursor.fetchone()
    if not result:
        return {"commander": []}
    return {"commander": result[0]}

# Endpoint to get the sum of specific secondary descriptions based on date and primary description
@app.get("/v1/chicago_crime/sum_secondary_description_by_date_primary_description", operation_id="get_sum_secondary_description", summary="Get the sum of specific secondary descriptions based on the specified date and primary description")
async def get_sum_secondary_description(secondary_description: str = Query(..., description="Secondary description"), date: str = Query(..., description="Date in 'MM/DD/YYYY' format"), primary_description: str = Query(..., description="Primary description")):
    cursor.execute("SELECT SUM(CASE WHEN T2.secondary_description = ? THEN 1 ELSE 0 END) FROM Crime AS T1 INNER JOIN IUCR AS T2 ON T1.iucr_no = T2.iucr_no WHERE T1.date LIKE ? AND T2.primary_description = ?", (secondary_description, '%' + date + '%', primary_description))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the district number with the highest count of a specific secondary description
@app.get("/v1/chicago_crime/top_district_by_secondary_description", operation_id="get_top_district_by_secondary_description", summary="Retrieves the district number that has the highest count of a specified secondary description. The secondary description is a categorization of the crime committed. This operation does not return the count itself, but rather the district number with the most occurrences of the specified secondary description.")
async def get_top_district_by_secondary_description(secondary_description: str = Query(..., description="Secondary description")):
    cursor.execute("SELECT T2.district_no FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T1.iucr_no = T2.iucr_no WHERE T1.secondary_description = ? GROUP BY T2.district_no ORDER BY COUNT(*) DESC LIMIT 1", (secondary_description,))
    result = cursor.fetchone()
    if not result:
        return {"district_no": []}
    return {"district_no": result[0]}

# Endpoint to get the IUCR number based on case number
@app.get("/v1/chicago_crime/iucr_no_by_case_number", operation_id="get_iucr_no_by_case_number", summary="Retrieves the IUCR number associated with the provided case number. This operation fetches the IUCR number from the 'IUCR' table by matching the case number from the 'Crime' table. The case number is a required input parameter.")
async def get_iucr_no_by_case_number(case_number: str = Query(..., description="Case number")):
    cursor.execute("SELECT T2.iucr_no FROM Crime AS T1 INNER JOIN IUCR AS T2 ON T1.iucr_no = T2.iucr_no WHERE T1.case_number = ?", (case_number,))
    result = cursor.fetchone()
    if not result:
        return {"iucr_no": []}
    return {"iucr_no": result[0]}

# Endpoint to get the community area name based on primary and secondary descriptions
@app.get("/v1/chicago_crime/community_area_by_descriptions", operation_id="get_community_area_by_descriptions", summary="Retrieve the name of the community area that most recently reported a crime matching the provided primary and secondary descriptions. The primary and secondary descriptions are used to filter the crime records and identify the relevant community area.")
async def get_community_area_by_descriptions(primary_description: str = Query(..., description="Primary description"), secondary_description: str = Query(..., description="Secondary description")):
    cursor.execute("SELECT T3.community_area_name FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T1.iucr_no = T2.iucr_no INNER JOIN Community_Area AS T3 ON T2.community_area_no = T3.community_area_no WHERE T1.primary_description = ? AND T1.secondary_description = ? GROUP BY T2.community_area_no ORDER BY T2.case_number DESC LIMIT 1", (primary_description, secondary_description))
    result = cursor.fetchone()
    if not result:
        return {"community_area_name": []}
    return {"community_area_name": result[0]}

# Endpoint to get the alderman's name based on case number
@app.get("/v1/chicago_crime/alderman_by_case_number", operation_id="get_alderman_by_case_number", summary="Retrieves the first and last name of the alderman associated with the specified case number. This operation uses the case number to look up the corresponding ward number, then returns the alderman's name for that ward.")
async def get_alderman_by_case_number(case_number: str = Query(..., description="Case number")):
    cursor.execute("SELECT T1.alderman_first_name, T1.alderman_last_name FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE T2.case_number = ?", (case_number,))
    result = cursor.fetchone()
    if not result:
        return {"alderman": []}
    return {"alderman": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get neighborhood names based on community area name
@app.get("/v1/chicago_crime/neighborhood_names_by_community_area", operation_id="get_neighborhood_names", summary="Retrieves the names of neighborhoods that belong to a specified community area. The community area is identified by its name, which is provided as an input parameter. This operation returns a list of neighborhood names that are associated with the given community area.")
async def get_neighborhood_names(community_area_name: str = Query(..., description="Name of the community area")):
    cursor.execute("SELECT T1.neighborhood_name FROM Neighborhood AS T1 INNER JOIN Community_Area AS T2 ON T1.community_area_no = T2.community_area_no WHERE T2.community_area_name = ?", (community_area_name,))
    result = cursor.fetchall()
    if not result:
        return {"neighborhood_names": []}
    return {"neighborhood_names": [row[0] for row in result]}

# Endpoint to get the count of neighborhoods in a specific community area
@app.get("/v1/chicago_crime/neighborhood_count_by_community_area", operation_id="get_neighborhood_count", summary="Retrieves the total number of neighborhoods within a specified community area. The community area is identified by its name, which is provided as an input parameter.")
async def get_neighborhood_count(community_area_name: str = Query(..., description="Name of the community area")):
    cursor.execute("SELECT SUM(CASE WHEN T1.community_area_name = ? THEN 1 ELSE 0 END) FROM Community_Area AS T1 INNER JOIN Neighborhood AS T2 ON T1.community_area_no = T2.community_area_no", (community_area_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get crime descriptions based on case number
@app.get("/v1/chicago_crime/crime_description_by_case_number", operation_id="get_crime_description", summary="Retrieves the description of a specific crime based on its case number. The case number is used to locate the crime in the database and the description is returned. The description includes details about the crime and its classification according to the FBI code.")
async def get_crime_description(case_number: str = Query(..., description="Case number of the crime")):
    cursor.execute("SELECT description FROM Crime AS T1 INNER JOIN FBI_Code AS T2 ON T1.fbi_code_no = T2.fbi_code_no WHERE T1.case_number = ?", (case_number,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the count of crimes in a specific community area with arrests
@app.get("/v1/chicago_crime/crime_count_by_community_area_with_arrests", operation_id="get_crime_count_with_arrests", summary="Retrieves the total number of crimes in a specified community area where an arrest was made. The operation requires the name of the community area and the arrest status as input parameters.")
async def get_crime_count_with_arrests(community_area_name: str = Query(..., description="Name of the community area"), arrest: str = Query(..., description="Arrest status (TRUE or FALSE)")):
    cursor.execute("SELECT SUM(CASE WHEN T1.community_area_name = ? THEN 1 ELSE 0 END) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no WHERE T2.arrest = ?", (community_area_name, arrest))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of crimes with a specific secondary description in a specific community area
@app.get("/v1/chicago_crime/percentage_crimes_by_secondary_description", operation_id="get_percentage_crimes_by_secondary_description", summary="Retrieve the proportion of crimes with a specified secondary description, given a primary description and community area. This operation calculates the percentage of crimes that match the provided secondary description, out of all crimes with the specified primary description in the selected community area.")
async def get_percentage_crimes_by_secondary_description(secondary_description: str = Query(..., description="Secondary description of the crime"), primary_description: str = Query(..., description="Primary description of the crime"), community_area_name: str = Query(..., description="Name of the community area")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.secondary_description = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.case_number) FROM Crime AS T1 INNER JOIN IUCR AS T2 ON T1.iucr_no = T2.iucr_no INNER JOIN Community_Area AS T3 ON T1.community_area_no = T3.community_area_no WHERE T2.primary_description = ? AND T3.community_area_name = ?", (secondary_description, primary_description, community_area_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of crimes with a specific FBI code title in a specific community area
@app.get("/v1/chicago_crime/percentage_crimes_by_fbi_code_title", operation_id="get_percentage_crimes_by_fbi_code_title", summary="Retrieve the proportion of crimes associated with a specific FBI code title within a designated community area. The calculation is based on the total number of crimes in the community area.")
async def get_percentage_crimes_by_fbi_code_title(fbi_code_title: str = Query(..., description="Title of the FBI code"), community_area_name: str = Query(..., description="Name of the community area")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.title = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.case_number) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no INNER JOIN FBI_Code AS T3 ON T2.fbi_code_no = T3.fbi_code_no WHERE T1.community_area_name = ?", (fbi_code_title, community_area_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of crimes on a specific block within a specific date pattern
@app.get("/v1/chicago_crime/crime_count_by_block_and_date", operation_id="get_crime_count_by_block_and_date", summary="Get the count of crimes on a specific block within a specific date pattern")
async def get_crime_count_by_block_and_date(date_pattern: str = Query(..., description="Date pattern in 'MM/DD/YYYY' format"), block: str = Query(..., description="Block address")):
    cursor.execute("SELECT SUM(CASE WHEN date LIKE ? THEN 1 ELSE 0 END) FROM Crime WHERE block = ?", (date_pattern, block))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the community area with the highest population
@app.get("/v1/chicago_crime/most_populous_community_area", operation_id="get_most_populous_community_area", summary="Retrieves the name of the community area with the highest population in Chicago. This endpoint provides a single record containing the name of the most populous community area, based on the latest available data.")
async def get_most_populous_community_area():
    cursor.execute("SELECT community_area_name FROM Community_Area ORDER BY population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"community_area_name": []}
    return {"community_area_name": result[0]}

# Endpoint to get the count of crimes with a specific location description, date pattern, and domestic status
@app.get("/v1/chicago_crime/crime_count_by_location_date_domestic", operation_id="get_crime_count_by_location_date_domestic", summary="Retrieves the total number of crimes that match a specified location description, date pattern, and domestic status. The location description parameter filters crimes based on their location. The date pattern parameter, in 'YYYY' format, narrows down the results to crimes that occurred within a specific year. The domestic status parameter further refines the search to include only crimes that are classified as domestic or non-domestic.")
async def get_crime_count_by_location_date_domestic(location_description: str = Query(..., description="Location description of the crime"), date_pattern: str = Query(..., description="Date pattern in 'YYYY' format"), domestic: str = Query(..., description="Domestic status (TRUE or FALSE)")):
    cursor.execute("SELECT SUM(CASE WHEN location_description = ? THEN 1 ELSE 0 END) FROM Crime WHERE date LIKE ? AND domestic = ?", (location_description, date_pattern, domestic))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the side with the highest total population
@app.get("/v1/chicago_crime/most_populous_side", operation_id="get_most_populous_side", summary="Retrieves the side of the city with the highest total population. The operation calculates the sum of the population for each side and returns the side with the highest total population.")
async def get_most_populous_side():
    cursor.execute("SELECT SUM(population) FROM Community_Area GROUP BY side ORDER BY SUM(population) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"population": []}
    return {"population": result[0]}

# Endpoint to get the sum of arrests in a specific year, location, and FBI code
@app.get("/v1/chicago_crime/sum_arrests_year_location_fbi_code", operation_id="get_sum_arrests", summary="Retrieves the total number of arrests that occurred in a given year, location, and FBI code category. The year should be provided in 'YYYY' format, while the location and FBI code number are specified as per their respective descriptions.")
async def get_sum_arrests(year: str = Query(..., description="Year in 'YYYY' format"), location_description: str = Query(..., description="Location description"), fbi_code_no: str = Query(..., description="FBI code number")):
    cursor.execute("SELECT SUM(CASE WHEN arrest = 'TRUE' THEN 1 ELSE 0 END) FROM Crime WHERE date LIKE ? AND location_description = ? AND fbi_code_no = ?", (f'%{year}%', location_description, fbi_code_no))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the commander of the district with the highest number of a specific crime
@app.get("/v1/chicago_crime/commander_highest_crime_count", operation_id="get_commander_highest_crime_count", summary="Retrieves the commander of the district with the highest count of a specific crime type, as identified by the provided crime title and FBI code number.")
async def get_commander_highest_crime_count(title: str = Query(..., description="Title of the crime"), fbi_code_no: int = Query(..., description="FBI code number")):
    cursor.execute("SELECT T1.commander FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no INNER JOIN FBI_Code AS T3 ON T2.fbi_code_no = T3.fbi_code_no WHERE T3.title = ? AND T2.fbi_code_no = ? GROUP BY T2.fbi_code_no ORDER BY COUNT(T1.district_no) DESC LIMIT 1", (title, fbi_code_no))
    result = cursor.fetchone()
    if not result:
        return {"commander": []}
    return {"commander": result[0]}

# Endpoint to get the title of the crime with the highest FBI code number
@app.get("/v1/chicago_crime/crime_title_highest_fbi_code", operation_id="get_crime_title_highest_fbi_code", summary="Retrieves the title of the crime with the highest FBI code number from the database. The operation returns the title of the crime that has the highest FBI code number, which is determined by comparing the FBI code numbers of all crimes in the database.")
async def get_crime_title_highest_fbi_code():
    cursor.execute("SELECT T2.title FROM Crime AS T1 INNER JOIN FBI_Code AS T2 ON T1.fbi_code_no = T2.fbi_code_no ORDER BY T2.fbi_code_no DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the sum of a specific crime title in a district with specific arrest status and location description
@app.get("/v1/chicago_crime/sum_crime_title_district_arrest_location", operation_id="get_sum_crime_title", summary="Retrieve the total count of a specific crime type in a given district, filtered by arrest status and location description.")
async def get_sum_crime_title(title: str = Query(..., description="Title of the crime"), district_name: str = Query(..., description="Name of the district"), arrest: str = Query(..., description="Arrest status"), location_description: str = Query(..., description="Location description")):
    cursor.execute("SELECT SUM(CASE WHEN T3.title = ? THEN 1 ELSE 0 END) FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no INNER JOIN FBI_Code AS T3 ON T2.fbi_code_no = T3.fbi_code_no WHERE T1.district_name = ? AND T2.arrest = ? AND T2.location_description = ?", (title, district_name, arrest, location_description))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the latitude and longitude of crimes in a specific community area with a specific FBI code and title
@app.get("/v1/chicago_crime/crime_lat_long_community_area_fbi_code_title", operation_id="get_crime_lat_long", summary="Retrieve the geographical coordinates of crimes that match a specific FBI code and title within a designated community area. The operation requires the community area name, the FBI code number, and the title of the crime as input parameters.")
async def get_crime_lat_long(community_area_name: str = Query(..., description="Name of the community area"), title: str = Query(..., description="Title of the crime"), fbi_code_no: int = Query(..., description="FBI code number")):
    cursor.execute("SELECT T2.latitude, T2.longitude FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no INNER JOIN FBI_Code AS T3 ON T2.fbi_code_no = T3.fbi_code_no WHERE T1.community_area_name = ? AND T3.title = ? AND T3.fbi_code_no = ?", (community_area_name, title, fbi_code_no))
    result = cursor.fetchall()
    if not result:
        return {"coordinates": []}
    return {"coordinates": [{"latitude": row[0], "longitude": row[1]} for row in result]}

# Endpoint to get the sum of arrests in a specific community area with specific IUCR descriptions
@app.get("/v1/chicago_crime/sum_arrests_community_area_iucr_descriptions", operation_id="get_sum_arrests_iucr", summary="Retrieve the total number of arrests in a specified community area, filtered by primary and secondary IUCR descriptions. The arrest status is also considered in the calculation.")
async def get_sum_arrests_iucr(community_area_name: str = Query(..., description="Name of the community area"), arrest: str = Query(..., description="Arrest status"), secondary_description: str = Query(..., description="Secondary description"), primary_description: str = Query(..., description="Primary description")):
    cursor.execute("SELECT SUM(CASE WHEN T2.arrest = ? THEN 1 ELSE 0 END) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no INNER JOIN IUCR AS T3 ON T2.iucr_no = T3.iucr_no WHERE T1.community_area_name = ? AND T3.secondary_description = ? AND T3.primary_description = ?", (arrest, community_area_name, secondary_description, primary_description))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the count of domestic crimes in the most populated ward with a specific location description
@app.get("/v1/chicago_crime/count_domestic_crimes_most_populated_ward", operation_id="get_count_domestic_crimes", summary="Retrieves the total number of domestic crimes that match a specified location description in the most populous ward. The operation filters crimes based on their domestic status and location description, then counts the number of matching records in the most populated ward.")
async def get_count_domestic_crimes(domestic: str = Query(..., description="Domestic status"), location_description: str = Query(..., description="Location description")):
    cursor.execute("SELECT COUNT(T2.report_no) FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE T2.domestic = ? AND T2.location_description = ? ORDER BY T1.Population DESC LIMIT 1", (domestic, location_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of arrests in a specific ward with specific alderman details and domestic status
@app.get("/v1/chicago_crime/sum_arrests_ward_alderman_domestic", operation_id="get_sum_arrests_ward_alderman", summary="Get the sum of arrests in a specific ward with specific alderman details and domestic status")
async def get_sum_arrests_ward_alderman(alderman_first_name: str = Query(..., description="First name of the alderman"), alderman_last_name: str = Query(..., description="Last name of the alderman"), alderman_name_suffix: str = Query(..., description="Suffix of the alderman's name"), domestic: str = Query(..., description="Domestic status")):
    cursor.execute("SELECT SUM(CASE WHEN T2.arrest = ? THEN 1 ELSE 0 END) FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE T1.alderman_first_name = ? AND T1.alderman_last_name = ? AND alderman_name_suffix = ? AND T2.domestic = ?", (domestic, alderman_first_name, alderman_last_name, alderman_name_suffix))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get FBI crime titles grouped by community area and ordered by population and FBI code
@app.get("/v1/chicago_crime/fbi_crime_titles_by_community_area", operation_id="get_fbi_crime_titles", summary="Retrieves a list of FBI crime titles, grouped by community area and ordered by population and FBI code. The number of results can be limited using the input parameter.")
async def get_fbi_crime_titles(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.title FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no INNER JOIN FBI_Code AS T3 ON T2.fbi_code_no = T3.fbi_code_no GROUP BY T3.title ORDER BY T1.population ASC, T3.fbi_code_no DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get ward office addresses based on crime block
@app.get("/v1/chicago_crime/ward_office_addresses_by_block", operation_id="get_ward_office_addresses", summary="Retrieves the ward office addresses associated with a specific crime block. The block address is used to identify the relevant ward numbers, which are then used to fetch the corresponding ward office addresses. The results are grouped by ward office address.")
async def get_ward_office_addresses(block: str = Query(..., description="Block address")):
    cursor.execute("SELECT T1.ward_office_address FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE T2.block = ? GROUP BY T1.ward_office_address", (block,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [row[0] for row in result]}

# Endpoint to get community area names based on IUCR primary description
@app.get("/v1/chicago_crime/community_area_names_by_iucr_description", operation_id="get_community_area_names", summary="Retrieves the names of community areas where crimes with a specified primary description have occurred. The results are ordered by case number in descending order and limited to a specified number of records.")
async def get_community_area_names(primary_description: str = Query(..., description="Primary description of the IUCR"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.community_area_name FROM Crime AS T1 INNER JOIN IUCR AS T2 ON T1.iucr_no = T2.iucr_no INNER JOIN Community_Area AS T3 ON T1.community_area_no = T3.community_area_no WHERE T2.primary_description = ? GROUP BY T1.iucr_no ORDER BY T1.case_number DESC LIMIT ?", (primary_description, limit))
    result = cursor.fetchall()
    if not result:
        return {"community_area_names": []}
    return {"community_area_names": [row[0] for row in result]}

# Endpoint to get the sum of crimes based on alderman's last name, FBI title, arrest status, and alderman's first name
@app.get("/v1/chicago_crime/sum_crimes_by_alderman_fbi_title_arrest", operation_id="get_sum_crimes", summary="Retrieves the total number of crimes committed in wards represented by a specific alderman, filtered by the FBI title of the crime and the arrest status. The alderman is identified by their first and last names.")
async def get_sum_crimes(alderman_last_name: str = Query(..., description="Alderman's last name"), fbi_title: str = Query(..., description="FBI title"), arrest: str = Query(..., description="Arrest status"), alderman_first_name: str = Query(..., description="Alderman's first name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.alderman_last_name = ? THEN 1 ELSE 0 END) FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no INNER JOIN FBI_Code AS T3 ON T2.fbi_code_no = T3.fbi_code_no WHERE T3.title = ? AND T2.arrest = ? AND T1.alderman_first_name = ?", (alderman_last_name, fbi_title, arrest, alderman_first_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the count of crimes based on date pattern and domestic status
@app.get("/v1/chicago_crime/count_crimes_by_date_domestic", operation_id="get_count_crimes", summary="Retrieves the total number of crimes that match a specified date pattern and domestic status. The date pattern should be provided in 'M/D/YYYY' format. The domestic status indicates whether the crime was domestic or not.")
async def get_count_crimes(date_pattern: str = Query(..., description="Date pattern in 'M/D/YYYY' format"), domestic: str = Query(..., description="Domestic status")):
    cursor.execute("SELECT COUNT(*) FROM Crime WHERE date LIKE ? AND domestic = ?", (date_pattern, domestic))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get index codes based on primary description
@app.get("/v1/chicago_crime/index_codes_by_primary_description", operation_id="get_index_codes", summary="Retrieves the index code associated with the provided primary description from the IUCR database. This operation allows users to look up the index code for a specific primary description, facilitating efficient data retrieval and organization.")
async def get_index_codes(primary_description: str = Query(..., description="Primary description of the IUCR")):
    cursor.execute("SELECT index_code FROM IUCR WHERE primary_description = ?", (primary_description,))
    result = cursor.fetchall()
    if not result:
        return {"index_codes": []}
    return {"index_codes": [row[0] for row in result]}

# Endpoint to get district commander and email based on district name
@app.get("/v1/chicago_crime/district_commander_email", operation_id="get_district_commander_email", summary="Retrieves the district commander's name and email address based on the provided district name.")
async def get_district_commander_email(district_name: str = Query(..., description="District name")):
    cursor.execute("SELECT commander, email FROM District WHERE district_name = ?", (district_name,))
    result = cursor.fetchall()
    if not result:
        return {"commander_email": []}
    return {"commander_email": [{"commander": row[0], "email": row[1]} for row in result]}

# Endpoint to get alderman details based on population
@app.get("/v1/chicago_crime/alderman_details_by_population", operation_id="get_alderman_details", summary="Retrieves details of aldermen, sorted by the population of their respective wards in descending order. The operation allows you to limit the number of results returned. The response includes the alderman's name suffix, first name, and last name.")
async def get_alderman_details(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT alderman_name_suffix, alderman_first_name, alderman_last_name FROM Ward ORDER BY population DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"alderman_details": []}
    return {"alderman_details": [{"suffix": row[0], "first_name": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get community area names based on side
@app.get("/v1/chicago_crime/community_area_names_by_side", operation_id="get_community_area_names_by_side", summary="Retrieves the names of community areas located on a specified side. The side parameter determines the area's location, allowing for targeted searches.")
async def get_community_area_names_by_side(side: str = Query(..., description="Side of the community area")):
    cursor.execute("SELECT community_area_name FROM Community_Area WHERE side = ?", (side,))
    result = cursor.fetchall()
    if not result:
        return {"community_area_names": []}
    return {"community_area_names": [row[0] for row in result]}

# Endpoint to get FBI crime titles and descriptions based on crime against
@app.get("/v1/chicago_crime/fbi_crime_titles_descriptions_by_crime_against", operation_id="get_fbi_crime_titles_descriptions", summary="Retrieves detailed information about specific types of crimes, including their titles and descriptions, based on the category of crime against. This operation allows users to filter the results by specifying the category of crime against, providing a focused and relevant set of data.")
async def get_fbi_crime_titles_descriptions(crime_against: str = Query(..., description="Crime against category")):
    cursor.execute("SELECT title, description FROM FBI_Code WHERE crime_against = ?", (crime_against,))
    result = cursor.fetchall()
    if not result:
        return {"crime_titles_descriptions": []}
    return {"crime_titles_descriptions": [{"title": row[0], "description": row[1]} for row in result]}

# Endpoint to get secondary description, latitude, and longitude for a specific IUCR number
@app.get("/v1/chicago_crime/iucr_details", operation_id="get_iucr_details", summary="Retrieves the secondary description, latitude, and longitude associated with a specific IUCR number. This operation fetches the requested details from the IUCR and Crime tables, using the provided IUCR number to filter the results.")
async def get_iucr_details(iucr_no: int = Query(..., description="IUCR number")):
    cursor.execute("SELECT T1.secondary_description, T2.latitude, T2.longitude FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T1.iucr_no = T2.iucr_no WHERE T2.iucr_no = ?", (iucr_no,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the sum of crimes with a specific alderman last name, first name, and date pattern
@app.get("/v1/chicago_crime/crime_sum_alderman", operation_id="get_crime_sum_alderman", summary="Retrieves the total number of crimes associated with a specific alderman, identified by their first and last names, within a given date range. The date range is defined using a pattern that matches the desired date format.")
async def get_crime_sum_alderman(alderman_last_name: str = Query(..., description="Alderman last name"), alderman_first_name: str = Query(..., description="Alderman first name"), date_pattern: str = Query(..., description="Date pattern in 'MM/DD/YYYY%' format")):
    cursor.execute("SELECT SUM(CASE WHEN T2.alderman_last_name = ? THEN 1 ELSE 0 END) FROM Crime AS T1 INNER JOIN Ward AS T2 ON T1.ward_no = T2.ward_no WHERE T2.alderman_name_suffix IS NULL AND T2.alderman_first_name = ? AND date LIKE ?", (alderman_last_name, alderman_first_name, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of arrests in a specific district and location description
@app.get("/v1/chicago_crime/arrest_sum_district_location", operation_id="get_arrest_sum_district_location", summary="Retrieves the total number of arrests that match a specified arrest status, district name, and location description. This operation calculates the sum of arrests based on the provided criteria, offering a comprehensive view of arrest statistics in a particular district and location.")
async def get_arrest_sum_district_location(arrest: str = Query(..., description="Arrest status"), district_name: str = Query(..., description="District name"), location_description: str = Query(..., description="Location description")):
    cursor.execute("SELECT SUM(CASE WHEN T1.arrest = ? THEN 1 ELSE 0 END) FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no WHERE T2.district_name = ? AND T1.location_description = ?", (arrest, district_name, location_description))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get case details for a specific block
@app.get("/v1/chicago_crime/case_details_block", operation_id="get_case_details_block", summary="Retrieves case details for a specific block, including the case number, alderman's first and last name, and district name. The data is aggregated from the District, Crime, and Ward tables based on the provided block address.")
async def get_case_details_block(block: str = Query(..., description="Block address")):
    cursor.execute("SELECT T2.case_number, T3.alderman_first_name, T3.alderman_last_name, T1.district_name FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no INNER JOIN Ward AS T3 ON T2.ward_no = T3.ward_no WHERE T2.block = ? GROUP BY T2.case_number, T3.alderman_first_name, T3.alderman_last_name, T1.district_name", (block,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the sum of crimes with a specific FBI code title
@app.get("/v1/chicago_crime/crime_sum_fbi_code", operation_id="get_crime_sum_fbi_code", summary="Retrieves the total count of crimes associated with a specific FBI code title. The operation calculates the sum of crimes that match the provided FBI code title, which is obtained by joining the FBI_Code and Crime tables on the FBI code number.")
async def get_crime_sum_fbi_code(title: str = Query(..., description="FBI code title")):
    cursor.execute("SELECT SUM(CASE WHEN T1.title = ? THEN 1 ELSE 0 END) FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T2.fbi_code_no = T1.fbi_code_no", (title,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get neighborhood names for a specific community area
@app.get("/v1/chicago_crime/neighborhood_names_community_area", operation_id="get_neighborhood_names_community_area", summary="Retrieves the names of neighborhoods that belong to a specified community area. The operation requires the name of the community area as input and returns a list of corresponding neighborhood names.")
async def get_neighborhood_names_community_area(community_area_name: str = Query(..., description="Community area name")):
    cursor.execute("SELECT T2.neighborhood_name FROM Community_Area AS T1 INNER JOIN Neighborhood AS T2 ON T2.community_area_no = T1.community_area_no WHERE T1.community_area_name = ?", (community_area_name,))
    result = cursor.fetchall()
    if not result:
        return {"neighborhoods": []}
    return {"neighborhoods": result}

# Endpoint to get the average monthly crime reports for the community area with the highest population
@app.get("/v1/chicago_crime/avg_monthly_crime_reports", operation_id="get_avg_monthly_crime_reports", summary="Retrieves the average monthly crime reports for the community area with the highest population. This operation calculates the average by dividing the total number of crime reports by 12, providing a monthly average. The community area is determined by the highest population count.")
async def get_avg_monthly_crime_reports():
    cursor.execute("SELECT CAST(COUNT(T2.report_no) AS REAL) / 12 FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no GROUP BY T1.community_area_no HAVING COUNT(T1.population) ORDER BY COUNT(T1.population) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of crimes with a specific FBI code title in a specific side of the city
@app.get("/v1/chicago_crime/crime_percentage_fbi_code_side", operation_id="get_crime_percentage_fbi_code_side", summary="Retrieve the proportion of crimes associated with a specific FBI code title that occurred on a particular side of the city. The calculation is based on the total number of crimes reported on that side.")
async def get_crime_percentage_fbi_code_side(title: str = Query(..., description="FBI code title"), side: str = Query(..., description="Side of the city")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.title = ? THEN T2.report_no END) AS REAL) * 100 / COUNT(T2.report_no) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no INNER JOIN FBI_Code AS T3 ON T3.fbi_code_no = T2.fbi_code_no WHERE T1.side = ?", (title, side))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get crime details based on primary and secondary descriptions
@app.get("/v1/chicago_crime/crime_details_descriptions", operation_id="get_crime_details_descriptions", summary="Retrieves detailed information about crimes in Chicago based on primary and secondary descriptions. This operation returns the location description, alderman's first and last name, and any name suffixes associated with the specified primary and secondary descriptions. The data is obtained by joining the Ward, Crime, and IUCR tables using the ward number and IUCR number.")
async def get_crime_details_descriptions(primary_description: str = Query(..., description="Primary description"), secondary_description: str = Query(..., description="Secondary description")):
    cursor.execute("SELECT T2.location_description, T1.alderman_first_name, T1.alderman_last_name, T1.alderman_name_suffix FROM Ward AS T1 INNER JOIN Crime AS T2 ON T2.ward_no = T1.ward_no INNER JOIN IUCR AS T3 ON T3.iucr_no = T2.iucr_no WHERE T3.primary_description = ? AND T3.secondary_description = ?", (primary_description, secondary_description))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get crime details based on primary and secondary descriptions
@app.get("/v1/chicago_crime/crime_details_descriptions_date", operation_id="get_crime_details_descriptions_date", summary="Retrieves the date and geographical coordinates of crimes that match the provided primary and secondary descriptions. The primary and secondary descriptions are used to filter the crimes and return only those that meet the specified criteria.")
async def get_crime_details_descriptions_date(primary_description: str = Query(..., description="Primary description"), secondary_description: str = Query(..., description="Secondary description")):
    cursor.execute("SELECT T2.date, T2.latitude, T2.longitude FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T2.iucr_no = T1.iucr_no WHERE T1.primary_description = ? AND T1.secondary_description = ?", (primary_description, secondary_description))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the count of crime reports based on FBI code title, district commander, and location description
@app.get("/v1/chicago_crime/count_crime_reports_by_fbi_code_title_commander_location", operation_id="get_count_crime_reports", summary="Retrieves the total number of crime reports associated with a specific FBI code title, district commander, and location description. This operation provides a comprehensive count of crime reports that meet the specified criteria, offering valuable insights into crime patterns and trends.")
async def get_count_crime_reports(fbi_code_title: str = Query(..., description="Title of the FBI code"), commander: str = Query(..., description="Name of the district commander"), location_description: str = Query(..., description="Description of the location")):
    cursor.execute("SELECT COUNT(T2.report_no) FROM District AS T1 INNER JOIN Crime AS T2 ON T2.district_no = T1.district_no INNER JOIN FBI_Code AS T3 ON T3.fbi_code_no = T2.fbi_code_no WHERE T3.title = ? AND T1.commander = ? AND T2.location_description = ?", (fbi_code_title, commander, location_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of arrests in a community area with domestic incidents
@app.get("/v1/chicago_crime/percentage_arrests_domestic_incidents", operation_id="get_percentage_arrests", summary="Retrieves the percentage of arrests in a specified community area for domestic incidents. The calculation is based on the count of incidents with an arrest and the total count of incidents in the community area. The community area name and whether the incident is domestic are required as input parameters.")
async def get_percentage_arrests(community_area_name: str = Query(..., description="Name of the community area"), domestic: str = Query(..., description="Whether the incident is domestic (TRUE or FALSE)")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.arrest = 'TRUE' THEN T2.report_no END) AS REAL) * 100 / COUNT(T2.report_no) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no WHERE T1.community_area_name = ? AND T2.domestic = ?", (community_area_name, domestic))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of domestic incidents and count of domestic incidents at banks in a ward
@app.get("/v1/chicago_crime/domestic_incidents_percentage_and_count", operation_id="get_domestic_incidents", summary="Retrieves the percentage of domestic incidents and the count of domestic incidents at banks in a ward, based on the specified alderman's first and last name.")
async def get_domestic_incidents(alderman_first_name: str = Query(..., description="First name of the alderman"), alderman_last_name: str = Query(..., description="Last name of the alderman")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.domestic = 'TRUE' THEN T1.report_no END) AS REAL) * 100 / COUNT(T1.report_no), COUNT(CASE WHEN T1.domestic = 'TRUE' AND T1.location_description = 'BANK' THEN T1.report_no END) AS \"number\" FROM Crime AS T1 INNER JOIN Ward AS T2 ON T2.ward_no = T1.ward_no WHERE T2.alderman_first_name = ? AND T2.alderman_last_name = ?", (alderman_first_name, alderman_last_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": [], "number": []}
    return {"percentage": result[0], "number": result[1]}

# Endpoint to get the count of wards based on the alderman's first name
@app.get("/v1/chicago_crime/count_wards_by_alderman_first_name", operation_id="get_count_wards", summary="Retrieves the total number of wards associated with a specific alderman, identified by their first name.")
async def get_count_wards(alderman_first_name: str = Query(..., description="First name of the alderman")):
    cursor.execute("SELECT COUNT(*) FROM Ward WHERE alderman_first_name = ?", (alderman_first_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average population of community areas on a specific side
@app.get("/v1/chicago_crime/average_population_by_side", operation_id="get_average_population", summary="Get the average population of community areas on a specific side")
async def get_average_population(side: str = Query(..., description="Side of the community area (e.g., 'West')")):
    cursor.execute("SELECT AVG(population) FROM Community_Area WHERE side = ?", (side,))
    result = cursor.fetchone()
    if not result:
        return {"average_population": []}
    return {"average_population": result[0]}

# Endpoint to get the report number of the most recent crime in a ward based on alderman's name
@app.get("/v1/chicago_crime/most_recent_crime_report_by_alderman", operation_id="get_most_recent_crime_report", summary="Retrieves the report number of the most recent crime in a ward, based on the alderman's name. The operation considers the beat and population of the community area to determine the most recent crime. Input parameters include the first and last names of the alderman.")
async def get_most_recent_crime_report(alderman_first_name: str = Query(..., description="First name of the alderman"), alderman_last_name: str = Query(..., description="Last name of the alderman")):
    cursor.execute("SELECT T2.report_no FROM Ward AS T1 INNER JOIN Crime AS T2 ON T2.ward_no = T1.ward_no INNER JOIN Community_Area AS T3 ON T3.community_area_no = T2.community_area_no WHERE T1.alderman_first_name = ? AND T1.alderman_last_name = ? ORDER BY T2.beat DESC, T3.population DESC LIMIT 1", (alderman_first_name, alderman_last_name))
    result = cursor.fetchone()
    if not result:
        return {"report_no": []}
    return {"report_no": result[0]}

# Endpoint to get the count of crimes on the street based on FBI code title
@app.get("/v1/chicago_crime/count_street_crimes_by_fbi_code_title", operation_id="get_count_street_crimes", summary="Retrieves the total number of street crimes associated with a specific FBI code title. The operation filters crimes based on the provided FBI code title and calculates the sum of incidents occurring on the street.")
async def get_count_street_crimes(fbi_code_title: str = Query(..., description="Title of the FBI code")):
    cursor.execute("SELECT SUM(CASE WHEN T2.location_description = 'STREET' THEN 1 ELSE 0 END) FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T2.fbi_code_no = T1.fbi_code_no WHERE T1.title = ?", (fbi_code_title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the alderman's name based on a crime report number
@app.get("/v1/chicago_crime/alderman_name_by_report_number", operation_id="get_alderman_name", summary="Retrieves the first and last name of the alderman associated with a specific crime report number. The report number is used to identify the corresponding ward, which is then used to look up the alderman's name.")
async def get_alderman_name(report_no: int = Query(..., description="Crime report number")):
    cursor.execute("SELECT T2.alderman_first_name, T2.alderman_last_name FROM Crime AS T1 INNER JOIN Ward AS T2 ON T2.ward_no = T1.ward_no WHERE T1.report_no = ?", (report_no,))
    result = cursor.fetchone()
    if not result:
        return {"alderman_first_name": [], "alderman_last_name": []}
    return {"alderman_first_name": result[0], "alderman_last_name": result[1]}

# Endpoint to get the case numbers of domestic crimes in a community area
@app.get("/v1/chicago_crime/case_numbers_domestic_crimes", operation_id="get_case_numbers", summary="Retrieve the case numbers of domestic crimes that occurred in a specified community area. The operation requires the name of the community area and a boolean value indicating whether the incident is domestic or not.")
async def get_case_numbers(community_area_name: str = Query(..., description="Name of the community area"), domestic: str = Query(..., description="Whether the incident is domestic (TRUE or FALSE)")):
    cursor.execute("SELECT T2.case_number FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no WHERE T1.community_area_name = ? AND T2.domestic = ?", (community_area_name, domestic))
    result = cursor.fetchall()
    if not result:
        return {"case_numbers": []}
    return {"case_numbers": [row[0] for row in result]}

# Endpoint to get the sum of crimes with beat less than a specified value in a given community area
@app.get("/v1/chicago_crime/sum_crimes_beat_less_than_community_area", operation_id="get_sum_crimes_beat_less_than", summary="Retrieves the total number of crimes with a beat value less than a specified threshold in a given community area. The operation calculates the sum based on the provided beat threshold and community area name.")
async def get_sum_crimes_beat_less_than(beat_threshold: int = Query(..., description="Beat threshold value"), community_area_name: str = Query(..., description="Community area name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.beat < ? THEN 1 ELSE 0 END) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no WHERE T1.community_area_name = ?", (beat_threshold, community_area_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of community areas with population greater than a specified value in a given side
@app.get("/v1/chicago_crime/sum_community_areas_population_greater_than_side", operation_id="get_sum_community_areas_population_greater_than", summary="Get the sum of community areas with population greater than a specified value in a given side")
async def get_sum_community_areas_population_greater_than(population_threshold: int = Query(..., description="Population threshold value"), side: str = Query(..., description="Side of the community area")):
    cursor.execute("SELECT SUM(CASE WHEN T1.population > ? THEN 1 ELSE 0 END) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no WHERE T1.side = ?", (population_threshold, side))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of crimes against a specified category in a given community area
@app.get("/v1/chicago_crime/sum_crimes_against_category_community_area", operation_id="get_sum_crimes_against_category", summary="Retrieves the total count of a specific crime category within a designated community area. The operation requires the crime category and community area name as input parameters to calculate the sum.")
async def get_sum_crimes_against_category(crime_against: str = Query(..., description="Category of the crime"), community_area_name: str = Query(..., description="Community area name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.crime_against = ? THEN 1 ELSE 0 END) FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T2.fbi_code_no = T1.fbi_code_no INNER JOIN Community_Area AS T3 ON T3.community_area_no = T2.community_area_no WHERE T3.community_area_name = ?", (crime_against, community_area_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of domestic crimes in a given ward email
@app.get("/v1/chicago_crime/sum_domestic_crimes_ward_email", operation_id="get_sum_domestic_crimes", summary="Retrieves the total count of domestic crimes in a specific ward, as identified by its email address. The domestic crime status is used to filter the crimes considered in the count.")
async def get_sum_domestic_crimes(domestic: str = Query(..., description="Domestic crime status"), ward_email: str = Query(..., description="Ward email")):
    cursor.execute("SELECT SUM(CASE WHEN T2.domestic = ? THEN 1 ELSE 0 END) FROM Ward AS T1 INNER JOIN Crime AS T2 ON T2.ward_no = T1.ward_no WHERE T1.ward_email = ?", (domestic, ward_email))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the address of a district based on a case number
@app.get("/v1/chicago_crime/district_address_by_case_number", operation_id="get_district_address", summary="Retrieves the address of a district associated with a specific case number. The case number is used to locate the corresponding crime record, which is then linked to the district record to obtain the address.")
async def get_district_address(case_number: str = Query(..., description="Case number")):
    cursor.execute("SELECT T1.address FROM District AS T1 INNER JOIN Crime AS T2 ON T2.district_no = T1.district_no WHERE T2.case_number = ?", (case_number,))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get the sum of beats in community areas with population greater than a specified value in a given side
@app.get("/v1/chicago_crime/sum_beats_population_greater_than_side", operation_id="get_sum_beats_population_greater_than", summary="Retrieves the total number of beats in community areas with a population exceeding a specified threshold, filtered by a given side. The side parameter determines the specific area to consider.")
async def get_sum_beats_population_greater_than(population_threshold: int = Query(..., description="Population threshold value"), side: str = Query(..., description="Side of the community area")):
    cursor.execute("SELECT 1.0 * SUM(CASE WHEN T1.population > ? THEN T2.beat ELSE 0 END) AS sum FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no WHERE T1.side = ?", (population_threshold, side))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get case numbers based on date pattern and crime category
@app.get("/v1/chicago_crime/case_numbers_by_date_pattern_crime_category", operation_id="get_case_numbers_by_date_pattern", summary="Retrieves the case numbers of crimes that match a specified date pattern and crime category. The date pattern should be provided in the 'M/%/YYYY%' format. The crime category refers to the type of crime committed, such as property crime or violent crime.")
async def get_case_numbers_by_date_pattern(date_pattern: str = Query(..., description="Date pattern in 'M/%/YYYY%' format"), crime_against: str = Query(..., description="Category of the crime")):
    cursor.execute("SELECT T2.case_number FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T2.fbi_code_no = T1.fbi_code_no WHERE T2.date LIKE ? AND T1.crime_against = ?", (date_pattern, crime_against))
    result = cursor.fetchall()
    if not result:
        return {"case_numbers": []}
    return {"case_numbers": [row[0] for row in result]}

# Endpoint to get the percentage of domestic crimes in the community area with the highest population
@app.get("/v1/chicago_crime/percentage_domestic_crimes_highest_population", operation_id="get_percentage_domestic_crimes", summary="Retrieves the percentage of domestic crimes in the community area with the highest population. The operation calculates this percentage by counting the number of domestic crimes and dividing it by the total number of crimes in the community area. The domestic crime status is used to determine whether a crime is considered domestic or not.")
async def get_percentage_domestic_crimes(domestic: str = Query(..., description="Domestic crime status")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.domestic = ? THEN T2.domestic END) AS REAL) * 100 / COUNT(T2.domestic) FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no GROUP BY T1.community_area_no HAVING COUNT(T1.population) ORDER BY COUNT(T1.population) DESC LIMIT 1", (domestic,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get community area names and sides based on side and population range
@app.get("/v1/chicago_crime/community_area_side_population", operation_id="get_community_area_side_population", summary="Get community area names and sides based on side and population range")
async def get_community_area_side_population(side: str = Query(..., description="Side of the community area"), min_population: int = Query(..., description="Minimum population"), max_population: int = Query(..., description="Maximum population")):
    cursor.execute("SELECT community_area_name, side FROM Community_Area WHERE side = ? AND population BETWEEN ? AND ?", (side, min_population, max_population))
    result = cursor.fetchall()
    if not result:
        return {"community_areas": []}
    return {"community_areas": result}

# Endpoint to get latitude and longitude of crimes based on location description and arrest status
@app.get("/v1/chicago_crime/crime_latitude_longitude", operation_id="get_crime_latitude_longitude", summary="Retrieves the geographical coordinates (latitude and longitude) of crimes that match a specified location description and arrest status. The results are grouped by unique coordinate pairs.")
async def get_crime_latitude_longitude(location_description: str = Query(..., description="Location description of the crime"), arrest: str = Query(..., description="Arrest status of the crime")):
    cursor.execute("SELECT latitude, longitude FROM Crime WHERE location_description = ? AND arrest = ? GROUP BY latitude, longitude", (location_description, arrest))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": result}

# Endpoint to get district commander, email, and phone based on district name
@app.get("/v1/chicago_crime/district_commander_info", operation_id="get_district_commander_info", summary="Retrieves the contact information of the district commander for a specified district. The information includes the commander's name, email, and phone number. To obtain the data, provide the name of the district as an input parameter.")
async def get_district_commander_info(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT commander, email, phone FROM District WHERE district_name = ?", (district_name,))
    result = cursor.fetchall()
    if not result:
        return {"district_info": []}
    return {"district_info": result}

# Endpoint to get FBI code number and description based on title
@app.get("/v1/chicago_crime/fbi_code_description", operation_id="get_fbi_code_description", summary="Retrieves the FBI code number and its corresponding description based on the provided title. The title parameter is used to filter the FBI code records and return the matching code number and description.")
async def get_fbi_code_description(title: str = Query(..., description="Title of the FBI code")):
    cursor.execute("SELECT fbi_code_no, description FROM FBI_Code WHERE title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"fbi_codes": []}
    return {"fbi_codes": result}

# Endpoint to get the percentage of IUCR codes with a specific index code
@app.get("/v1/chicago_crime/iucr_percentage_by_index_code", operation_id="get_iucr_percentage_by_index_code", summary="Retrieves the percentage of IUCR codes that match a given index code. The calculation is based on the total count of IUCR codes with the specified index code divided by the total count of all IUCR codes.")
async def get_iucr_percentage_by_index_code(index_code: str = Query(..., description="Index code to filter IUCR codes")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN index_code = ? THEN iucr_no ELSE NULL END) AS REAL) * 100 / COUNT(iucr_no) FROM IUCR", (index_code,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the most common location description of crimes in a specific district
@app.get("/v1/chicago_crime/most_common_crime_location_description", operation_id="get_most_common_crime_location_description", summary="Retrieves the location description that has the highest frequency of crimes in the specified district. The district is identified by its name.")
async def get_most_common_crime_location_description(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT T2.location_description FROM District AS T1 INNER JOIN Crime AS T2 ON T2.district_no = T1.district_no WHERE T1.district_name = ? GROUP BY T2.location_description ORDER BY COUNT(T2.case_number) DESC LIMIT 1", (district_name,))
    result = cursor.fetchone()
    if not result:
        return {"location_description": []}
    return {"location_description": result[0]}

# Endpoint to get the ratio of ward numbers to distinct community area sides
@app.get("/v1/chicago_crime/ward_to_community_area_side_ratio", operation_id="get_ward_to_community_area_side_ratio", summary="Retrieves the ratio of the total number of ward numbers to the number of distinct community area sides. This operation calculates the ratio by joining the Ward, Crime, and Community_Area tables, considering the ward numbers and distinct community area sides.")
async def get_ward_to_community_area_side_ratio():
    cursor.execute("SELECT CAST(COUNT(T1.ward_no) AS REAL) / COUNT(DISTINCT T3.side) FROM Ward AS T1 INNER JOIN Crime AS T2 ON T2.ward_no = T1.ward_no INNER JOIN Community_Area AS T3 ON T3.community_area_no = T2.community_area_no")
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the community area number with the most crimes of a specific location description
@app.get("/v1/chicago_crime/most_crimes_community_area", operation_id="get_most_crimes_community_area", summary="Retrieves the community area number with the highest count of crimes that match a specified location description. The operation filters crimes based on the provided location description and aggregates them by community area number. The community area with the most crimes is then returned.")
async def get_most_crimes_community_area(location_description: str = Query(..., description="Location description of the crime")):
    cursor.execute("SELECT T1.community_area_no FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no WHERE T2.location_description = ? GROUP BY T1.community_area_no ORDER BY COUNT(T2.location_description) DESC LIMIT 1", (location_description,))
    result = cursor.fetchone()
    if not result:
        return {"community_area_no": []}
    return {"community_area_no": result[0]}

# Endpoint to get the ratio of crime reports to distinct district names based on secondary description
@app.get("/v1/chicago_crime/crime_report_to_district_ratio", operation_id="get_crime_report_to_district_ratio", summary="Retrieves the ratio of crime reports to the number of distinct districts, filtered by a specific secondary description of the crime.")
async def get_crime_report_to_district_ratio(secondary_description: str = Query(..., description="Secondary description of the crime")):
    cursor.execute("SELECT CAST(COUNT(T2.report_no) AS REAL) / COUNT(DISTINCT T1.district_name) FROM District AS T1 INNER JOIN Crime AS T2 ON T2.district_no = T1.district_no INNER JOIN IUCR AS T3 ON T3.iucr_no = T2.iucr_no WHERE T3.secondary_description = ?", (secondary_description,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the ward office address and phone with the most crimes without arrest
@app.get("/v1/chicago_crime/ward_office_most_crimes_no_arrest", operation_id="get_ward_office_most_crimes_no_arrest", summary="Retrieves the contact information of the ward office with the highest number of unresolved crimes. The operation filters crimes based on the provided arrest status and returns the address and phone number of the ward office with the most occurrences of such crimes.")
async def get_ward_office_most_crimes_no_arrest(arrest: str = Query(..., description="Arrest status of the crime")):
    cursor.execute("SELECT T2.ward_office_address, T2.ward_office_phone FROM Crime AS T1 INNER JOIN Ward AS T2 ON T2.ward_no = T1.ward_no WHERE T1.arrest = ? GROUP BY T2.ward_office_address, T2.ward_office_phone ORDER BY COUNT(T1.arrest) DESC LIMIT 1", (arrest,))
    result = cursor.fetchone()
    if not result:
        return {"ward_office": []}
    return {"ward_office": result}

# Endpoint to get case details based on secondary description
@app.get("/v1/chicago_crime/case_details_by_secondary_description", operation_id="get_case_details", summary="Retrieves case details, including case number, latitude, and longitude, for crimes that match the provided secondary description. The secondary description is a specific categorization of the crime.")
async def get_case_details(secondary_description: str = Query(..., description="Secondary description of the crime")):
    cursor.execute("SELECT T1.case_number, T1.latitude, T1.longitude FROM Crime AS T1 INNER JOIN IUCR AS T2 ON T2.iucr_no = T1.iucr_no WHERE T2.secondary_description = ?", (secondary_description,))
    result = cursor.fetchall()
    if not result:
        return {"cases": []}
    return {"cases": result}

# Endpoint to get the most common secondary description of crimes in a specific side of the community area
@app.get("/v1/chicago_crime/most_common_secondary_description_by_side", operation_id="get_most_common_secondary_description", summary="Get the most common secondary description of crimes in a specific side of the community area")
async def get_most_common_secondary_description(side: str = Query(..., description="Side of the community area")):
    cursor.execute("SELECT T3.secondary_description FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no INNER JOIN IUCR AS T3 ON T3.iucr_no = T2.iucr_no WHERE T1.side = ? GROUP BY T3.secondary_description ORDER BY COUNT(*) DESC LIMIT 1", (side,))
    result = cursor.fetchone()
    if not result:
        return {"secondary_description": []}
    return {"secondary_description": result[0]}

# Endpoint to get the community area with the least domestic crimes
@app.get("/v1/chicago_crime/least_domestic_crimes_community_area", operation_id="get_least_domestic_crimes_community_area", summary="Retrieves the community area with the lowest count of domestic crimes. The operation filters crimes based on the provided domestic status and returns the community area with the least occurrences of such crimes.")
async def get_least_domestic_crimes_community_area(domestic: str = Query(..., description="Domestic crime status (TRUE or FALSE)")):
    cursor.execute("SELECT T2.community_area_no FROM Crime AS T1 INNER JOIN Community_Area AS T2 ON T2.community_area_no = T1.community_area_no WHERE T1.domestic = ? GROUP BY T2.community_area_no ORDER BY COUNT(T2.community_area_no) ASC LIMIT 1", (domestic,))
    result = cursor.fetchone()
    if not result:
        return {"community_area_no": []}
    return {"community_area_no": result[0]}

# Endpoint to get the percentage of crimes related to a specific description under a specific FBI code title
@app.get("/v1/chicago_crime/percentage_crimes_by_description_and_fbi_title", operation_id="get_percentage_crimes_by_description_and_fbi_title", summary="Retrieves the percentage of crimes that match a specified description under a given FBI code title. The calculation is based on the total count of crimes with the same FBI code title.")
async def get_percentage_crimes_by_description_and_fbi_title(description: str = Query(..., description="Description of the crime"), fbi_title: str = Query(..., description="FBI code title")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.secondary_description LIKE ? THEN T1.secondary_description END) AS REAL) * 100 / COUNT(T1.secondary_description) FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T2.iucr_no = T1.iucr_no INNER JOIN FBI_Code AS T3 ON T3.fbi_code_no = T2.fbi_code_no WHERE T3.title = ?", (description, fbi_title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average number of cases per day in a specific month and year for a specific index code
@app.get("/v1/chicago_crime/average_cases_per_day", operation_id="get_average_cases_per_day", summary="Retrieves the average daily number of cases for a specific index code in a given month and year. The calculation is based on the total case count for the specified month and year, divided by the number of days in that month. The input parameters include the number of days in the month, the date pattern in 'M/%/YYYY%' format, and the index code.")
async def get_average_cases_per_day(days_in_month: int = Query(..., description="Number of days in the month"), date_pattern: str = Query(..., description="Date pattern in 'M/%/YYYY%' format"), index_code: str = Query(..., description="Index code")):
    cursor.execute("SELECT CAST(COUNT(T2.case_number) AS REAL) / ? FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T2.iucr_no = T1.iucr_no WHERE T2.date LIKE ? AND T1.index_code = ?", (days_in_month, date_pattern, index_code))
    result = cursor.fetchone()
    if not result:
        return {"average_cases": []}
    return {"average_cases": result[0]}

# Endpoint to get community area details based on a specific secondary description
@app.get("/v1/chicago_crime/community_area_details_by_secondary_description", operation_id="get_community_area_details", summary="Retrieve the community area details, including the name and population, for areas where a specific crime type occurs more frequently than the average. The crime type is determined by the provided secondary description.")
async def get_community_area_details(secondary_description: str = Query(..., description="Secondary description of the crime")):
    cursor.execute("SELECT T2.community_area_name, T2.population FROM Crime AS T1 INNER JOIN Community_Area AS T2 ON T2.community_area_no = T1.community_area_no INNER JOIN IUCR AS T3 ON T3.iucr_no = T1.iucr_no WHERE T3.iucr_no = ( SELECT iucr_no FROM IUCR WHERE secondary_description = ? GROUP BY iucr_no HAVING COUNT(iucr_no) > ( SELECT SUM(CASE WHEN secondary_description = ? THEN 1.0 ELSE 0 END) / COUNT(iucr_no) AS average FROM IUCR ) )", (secondary_description, secondary_description))
    result = cursor.fetchall()
    if not result:
        return {"community_areas": []}
    return {"community_areas": result}

# Endpoint to get the percentage of reports with a specific FBI code title in a given district
@app.get("/v1/chicago_crime/percentage_reports_by_fbi_title_and_district", operation_id="get_percentage_reports_by_fbi_title_and_district", summary="Retrieves the percentage of crime reports with a specific FBI code title in a given district. The operation calculates this percentage by comparing the count of reports with the specified FBI code title to the total number of reports in the district.")
async def get_percentage_reports_by_fbi_title_and_district(fbi_title: str = Query(..., description="FBI code title"), district_name: str = Query(..., description="District name")):
    cursor.execute("SELECT COUNT(CASE WHEN T3.title = ? THEN T2.report_no END) * 100.0 / COUNT(T2.report_no) AS per FROM District AS T1 INNER JOIN Crime AS T2 ON T2.district_no = T1.district_no INNER JOIN FBI_Code AS T3 ON T3.fbi_code_no = T2.fbi_code_no WHERE T1.district_name = ?", (fbi_title, district_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in crime rates between two specific secondary descriptions
@app.get("/v1/chicago_crime/crime_rate_difference_by_secondary_descriptions", operation_id="get_crime_rate_difference", summary="Retrieves the difference in crime rates between two specific secondary descriptions, calculated by comparing the count of incidents for each description against the number of distinct districts where they occur.")
async def get_crime_rate_difference(secondary_description_1: str = Query(..., description="First secondary description of the crime"), secondary_description_2: str = Query(..., description="Second secondary description of the crime")):
    cursor.execute("SELECT ROUND(CAST(COUNT(CASE WHEN T1.secondary_description = ? THEN T1.iucr_no END) AS REAL) / CAST(COUNT(DISTINCT CASE WHEN T1.secondary_description = ? THEN T3.district_name END) AS REAL) - CAST(COUNT(CASE WHEN T1.secondary_description = ? THEN T1.iucr_no END) AS REAL) / CAST(COUNT(DISTINCT CASE WHEN T1.secondary_description = ? THEN T3.district_name END) AS REAL), 4) AS \"difference\" FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T2.iucr_no = T1.iucr_no INNER JOIN District AS T3 ON T3.district_no = T2.district_no", (secondary_description_1, secondary_description_1, secondary_description_2, secondary_description_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of crimes at a specific longitude
@app.get("/v1/chicago_crime/crime_count_by_longitude", operation_id="get_crime_count_by_longitude", summary="Retrieves the total number of crimes that occurred at the specified longitude in Chicago. The longitude parameter is used to filter the results.")
async def get_crime_count_by_longitude(longitude: float = Query(..., description="Longitude of the crime location")):
    cursor.execute("SELECT COUNT(*) FROM Crime WHERE longitude = ?", (longitude,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get secondary descriptions based on primary description
@app.get("/v1/chicago_crime/secondary_descriptions_by_primary_description", operation_id="get_secondary_descriptions", summary="Retrieves a list of secondary descriptions associated with the provided primary description of a crime. The secondary descriptions are grouped to eliminate duplicates.")
async def get_secondary_descriptions(primary_description: str = Query(..., description="Primary description of the crime")):
    cursor.execute("SELECT secondary_description FROM IUCR WHERE primary_description = ? GROUP BY secondary_description", (primary_description,))
    result = cursor.fetchall()
    if not result:
        return {"secondary_descriptions": []}
    return {"secondary_descriptions": result}

# Endpoint to get the alderman first names for wards with a population greater than a specified number
@app.get("/v1/chicago_crime/alderman_first_names_by_population", operation_id="get_alderman_first_names", summary="Retrieve the first names of aldermen from wards with a population exceeding the specified minimum. This operation allows you to filter wards based on their population size, providing a targeted list of aldermen first names.")
async def get_alderman_first_names(min_population: int = Query(..., description="Minimum population of the ward")):
    cursor.execute("SELECT alderman_first_name FROM Ward WHERE Population > ?", (min_population,))
    result = cursor.fetchall()
    if not result:
        return {"alderman_first_names": []}
    return {"alderman_first_names": [row[0] for row in result]}

# Endpoint to get report numbers for crimes with a specific FBI code title
@app.get("/v1/chicago_crime/report_numbers_by_fbi_code_title", operation_id="get_report_numbers", summary="Retrieves the report numbers of crimes that match a given FBI code title. The operation filters the crimes based on the provided FBI code title and returns the corresponding report numbers.")
async def get_report_numbers(fbi_code_title: str = Query(..., description="Title of the FBI code")):
    cursor.execute("SELECT T2.report_no FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T2.fbi_code_no = T1.fbi_code_no WHERE T1.title = ?", (fbi_code_title,))
    result = cursor.fetchall()
    if not result:
        return {"report_numbers": []}
    return {"report_numbers": [row[0] for row in result]}

# Endpoint to get the count of crimes in a specific district with a specific primary description
@app.get("/v1/chicago_crime/crime_count_by_district_and_description", operation_id="get_crime_count", summary="Retrieves the total number of crimes that match a specific primary description in a given district. The operation uses the district name and primary description as input parameters to filter the data and calculate the crime count.")
async def get_crime_count(district_name: str = Query(..., description="Name of the district"), primary_description: str = Query(..., description="Primary description of the crime")):
    cursor.execute("SELECT SUM(CASE WHEN T3.district_name = ? THEN 1 ELSE 0 END) FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T2.iucr_no = T1.iucr_no INNER JOIN District AS T3 ON T3.district_no = T2.district_no WHERE T1.primary_description = ?", (district_name, primary_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latitude and longitude of crimes in a specific community area
@app.get("/v1/chicago_crime/crime_locations_by_community_area", operation_id="get_crime_locations", summary="Retrieve the geographical coordinates of crimes committed within a specified community area. The operation returns a list of unique latitude and longitude pairs, providing a comprehensive overview of crime locations in the selected area.")
async def get_crime_locations(community_area_name: str = Query(..., description="Name of the community area")):
    cursor.execute("SELECT T2.latitude, T2.longitude FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no WHERE T1.community_area_name = ? GROUP BY T2.latitude, T2.longitude", (community_area_name,))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": [{"latitude": row[0], "longitude": row[1]} for row in result]}

# Endpoint to get the count of crimes in a specific neighborhood
@app.get("/v1/chicago_crime/crime_count_by_neighborhood", operation_id="get_crime_count_by_neighborhood", summary="Retrieves the total number of crimes that occurred in a specified neighborhood. The operation calculates the sum of crimes based on the provided neighborhood name, using data from the IUCR, Crime, Community Area, and Neighborhood tables.")
async def get_crime_count_by_neighborhood(neighborhood_name: str = Query(..., description="Name of the neighborhood")):
    cursor.execute("SELECT SUM(CASE WHEN T4.neighborhood_name = ? THEN 1 ELSE 0 END) FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T2.iucr_no = T1.iucr_no INNER JOIN Community_Area AS T3 ON T3.community_area_no = T2.community_area_no INNER JOIN Neighborhood AS T4 ON T4.community_area_no = T3.community_area_no", (neighborhood_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of domestic crimes under a specific commander
@app.get("/v1/chicago_crime/domestic_crime_count_by_commander", operation_id="get_domestic_crime_count", summary="Retrieves the total count of domestic crimes committed under a specific commander. The operation considers the domestic status of the crime to determine the count. The commander's name is used to filter the results.")
async def get_domestic_crime_count(domestic: str = Query(..., description="Domestic status of the crime"), commander: str = Query(..., description="Name of the commander")):
    cursor.execute("SELECT SUM(CASE WHEN T2.domestic = ? THEN 1 ELSE 0 END) FROM District AS T1 INNER JOIN Crime AS T2 ON T2.district_no = T1.district_no WHERE T1.commander = ?", (domestic, commander))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of crimes against a specific category in a specific district
@app.get("/v1/chicago_crime/crime_count_by_category_and_district", operation_id="get_crime_count_by_category", summary="Retrieves the total number of crimes that fall under a specific category within a designated district. The operation requires the category of crime and the name of the district as input parameters.")
async def get_crime_count_by_category(crime_against: str = Query(..., description="Category of the crime"), district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT SUM(CASE WHEN T1.crime_against = ? THEN 1 ELSE 0 END) FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T2.fbi_code_no = T1.fbi_code_no INNER JOIN District AS T3 ON T3.district_no = T2.district_no WHERE T3.district_name = ?", (crime_against, district_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone numbers of districts associated with a specific alderman
@app.get("/v1/chicago_crime/district_phones_by_alderman", operation_id="get_district_phones", summary="Retrieve the phone numbers of districts that are associated with a specific alderman, identified by their first and last names.")
async def get_district_phones(alderman_first_name: str = Query(..., description="First name of the alderman"), alderman_last_name: str = Query(..., description="Last name of the alderman")):
    cursor.execute("SELECT T3.phone FROM Ward AS T1 INNER JOIN Crime AS T2 ON T2.ward_no = T1.ward_no INNER JOIN District AS T3 ON T3.district_no = T2.district_no WHERE T1.alderman_first_name = ? AND T1.alderman_last_name = ?", (alderman_first_name, alderman_last_name))
    result = cursor.fetchall()
    if not result:
        return {"phones": []}
    return {"phones": [row[0] for row in result]}

# Endpoint to get the count of crimes in a specific community area with a specific FBI code description
@app.get("/v1/chicago_crime/crime_count_by_community_area_and_description", operation_id="get_crime_count_by_community_area", summary="Retrieve the total number of crimes that match a specific FBI code description in a given community area. The operation requires the name of the community area and the description of the FBI code as input parameters.")
async def get_crime_count_by_community_area(community_area_name: str = Query(..., description="Name of the community area"), description: str = Query(..., description="Description of the FBI code")):
    cursor.execute("SELECT SUM(CASE WHEN T3.community_area_name = ? THEN 1 ELSE 0 END) FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T2.fbi_code_no = T1.fbi_code_no INNER JOIN Community_Area AS T3 ON T3.community_area_no = T2.community_area_no WHERE T1.description = ?", (community_area_name, description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the district with the highest count of a specific type of crime
@app.get("/v1/chicago_crime/top_district_by_crime_type", operation_id="get_top_district_by_crime_type", summary="Retrieves the district with the highest count of a specific type of crime, as identified by the primary description of the crime.")
async def get_top_district_by_crime_type(primary_description: str = Query(..., description="Primary description of the crime")):
    cursor.execute("SELECT T3.district_name FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T2.iucr_no = T1.iucr_no INNER JOIN District AS T3 ON T3.district_no = T2.district_no WHERE T1.primary_description = ? GROUP BY T3.district_name ORDER BY COUNT(T1.primary_description) DESC LIMIT 1", (primary_description,))
    result = cursor.fetchone()
    if not result:
        return {"district_name": []}
    return {"district_name": result[0]}

# Endpoint to get the domestic status of crimes in a specific community area
@app.get("/v1/chicago_crime/domestic_status_by_community_area", operation_id="get_domestic_status_by_community_area", summary="Retrieves the most common domestic status of crimes in a specified community area. The operation filters crimes based on the provided community area name and domestic status, then returns the most frequently occurring domestic status. The domestic status indicates whether a crime was domestic or not.")
async def get_domestic_status_by_community_area(community_area_name: str = Query(..., description="Name of the community area"), domestic: str = Query(..., description="Domestic status (TRUE or FALSE)")):
    cursor.execute("SELECT T2.domestic FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T2.community_area_no = T1.community_area_no WHERE T1.community_area_name = ? AND T2.domestic = ? GROUP BY T2.domestic ORDER BY COUNT(T2.domestic) DESC LIMIT 1", (community_area_name, domestic))
    result = cursor.fetchone()
    if not result:
        return {"domestic_status": []}
    return {"domestic_status": result[0]}

# Endpoint to get the ward number with the highest count of specific crime descriptions and population criteria
@app.get("/v1/chicago_crime/ward_no_by_crime_description_and_population", operation_id="get_ward_no_by_crime_description_and_population", summary="Retrieves the ward number with the highest count of a specific crime type, considering both primary and secondary crime descriptions, and a population threshold. The ward number returned is the one with the most occurrences of the specified crime type and a population greater than the provided threshold.")
async def get_ward_no_by_crime_description_and_population(primary_description: str = Query(..., description="Primary description of the crime"), secondary_description: str = Query(..., description="Secondary description of the crime"), population: int = Query(..., description="Population threshold")):
    cursor.execute("SELECT T3.ward_no FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T2.iucr_no = T1.iucr_no INNER JOIN Ward AS T3 ON T3.ward_no = T2.ward_no WHERE T1.primary_description = ? AND T1.secondary_description = ? AND T3.Population > ? GROUP BY T3.ward_no ORDER BY COUNT(T3.ward_no) DESC LIMIT 1", (primary_description, secondary_description, population))
    result = cursor.fetchone()
    if not result:
        return {"ward_no": []}
    return {"ward_no": result[0]}

# Endpoint to get the commander with the highest count of a specific secondary crime description
@app.get("/v1/chicago_crime/commander_by_secondary_crime_description", operation_id="get_commander_by_secondary_crime_description", summary="Retrieves the commander with the highest count of a specific secondary crime description. The operation filters crimes based on the provided secondary description and identifies the commander with the most occurrences of that crime in their district. The result is a single commander's name.")
async def get_commander_by_secondary_crime_description(secondary_description: str = Query(..., description="Secondary description of the crime")):
    cursor.execute("SELECT T3.commander FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T2.iucr_no = T1.iucr_no INNER JOIN District AS T3 ON T3.district_no = T2.district_no WHERE T1.secondary_description = ? GROUP BY T3.commander ORDER BY COUNT(T1.secondary_description) DESC LIMIT 1", (secondary_description,))
    result = cursor.fetchone()
    if not result:
        return {"commander": []}
    return {"commander": result[0]}

# Endpoint to get the percentage of non-domestic cases in a specific district
@app.get("/v1/chicago_crime/percentage_non_domestic_cases_by_district", operation_id="get_percentage_non_domestic_cases_by_district", summary="Retrieves the percentage of non-domestic cases in a specified district. The operation calculates the ratio of non-domestic cases to the total number of cases in the district, providing a statistical overview of crime distribution.")
async def get_percentage_non_domestic_cases_by_district(district_name: str = Query(..., description="Name of the district"), domestic: str = Query(..., description="Domestic status (TRUE or FALSE)")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.domestic = ? THEN T2.case_number END) AS REAL) * 100 / COUNT(T2.case_number) FROM District AS T1 INNER JOIN Crime AS T2 ON T2.district_no = T1.district_no WHERE T1.district_name = ?", (domestic, district_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average population of wards with specific crime location description and arrest status
@app.get("/v1/chicago_crime/average_population_by_location_description_and_arrest", operation_id="get_average_population_by_location_description_and_arrest", summary="Retrieves the average population of wards where crimes with a specified location description and arrest status have occurred. The location description and arrest status are provided as input parameters.")
async def get_average_population_by_location_description_and_arrest(location_description: str = Query(..., description="Location description of the crime"), arrest: str = Query(..., description="Arrest status (TRUE or FALSE)")):
    cursor.execute("SELECT AVG(T2.Population) FROM Crime AS T1 INNER JOIN Ward AS T2 ON T2.ward_no = T1.ward_no WHERE T1.location_description = ? AND T1.arrest = ?", (location_description, arrest))
    result = cursor.fetchone()
    if not result:
        return {"average_population": []}
    return {"average_population": result[0]}

# Endpoint to get the top 5 wards by population
@app.get("/v1/chicago_crime/top_wards_by_population", operation_id="get_top_wards_by_population", summary="Retrieves the top five wards with the highest population, along with the first and last names of the respective aldermen.")
async def get_top_wards_by_population():
    cursor.execute("SELECT alderman_first_name, alderman_last_name FROM Ward ORDER BY Population DESC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"wards": []}
    return {"wards": result}

# Endpoint to get the count of FBI codes for a specific crime against
@app.get("/v1/chicago_crime/count_fbi_codes_by_crime_against", operation_id="get_count_fbi_codes_by_crime_against", summary="Retrieves the total number of FBI codes associated with a specific type of crime against. The operation requires the type of crime against as an input parameter.")
async def get_count_fbi_codes_by_crime_against(crime_against: str = Query(..., description="Type of crime against")):
    cursor.execute("SELECT COUNT(*) AS cnt FROM FBI_Code WHERE crime_against = ?", (crime_against,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of districts by zip code
@app.get("/v1/chicago_crime/count_districts_by_zip_code", operation_id="get_count_districts_by_zip_code", summary="Retrieves the total number of districts associated with a specific zip code.")
async def get_count_districts_by_zip_code(zip_code: int = Query(..., description="Zip code")):
    cursor.execute("SELECT COUNT(*) AS cnt FROM District WHERE zip_code = ?", (zip_code,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the crime against for a specific FBI code title
@app.get("/v1/chicago_crime/crime_against_by_fbi_code_title", operation_id="get_crime_against_by_fbi_code_title", summary="Retrieves the specific crime against category associated with the provided FBI code title. The input parameter is used to identify the FBI code title, which is then used to look up the corresponding crime against category in the FBI_Code table.")
async def get_crime_against_by_fbi_code_title(title: str = Query(..., description="Title of the FBI code")):
    cursor.execute("SELECT crime_against FROM FBI_Code WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"crime_against": []}
    return {"crime_against": result[0]}

# Endpoint to get the community area name with the highest community area number
@app.get("/v1/chicago_crime/community_area_name_by_highest_community_area_no", operation_id="get_community_area_name_by_highest_community_area_no", summary="Retrieves the name of the community area with the highest community area number. This operation fetches the data from the Community_Area table, which is joined with the Neighborhood table using the community area number. The result is ordered in descending order based on the community area number and the top record is returned.")
async def get_community_area_name_by_highest_community_area_no():
    cursor.execute("SELECT T1.community_area_name FROM Community_Area AS T1 INNER JOIN Neighborhood AS T2 ON T1.community_area_no = T2.community_area_no ORDER BY T2.community_area_no DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"community_area_name": []}
    return {"community_area_name": result[0]}

# Endpoint to get the sum of crimes at a specific longitude, index code, and latitude
@app.get("/v1/chicago_crime/sum_crimes_longitude_index_latitude", operation_id="get_sum_crimes_longitude_index_latitude", summary="Retrieves the total number of crimes that occurred at a specific geographical location, identified by longitude, index code, and latitude. The index code represents the type of crime committed.")
async def get_sum_crimes_longitude_index_latitude(longitude: str = Query(..., description="Longitude of the crime location"), index_code: str = Query(..., description="Index code of the crime"), latitude: str = Query(..., description="Latitude of the crime location")):
    cursor.execute("SELECT SUM(CASE WHEN T1.longitude = ? THEN 1 ELSE 0 END) FROM Crime AS T1 INNER JOIN IUCR AS T2 ON T1.report_no = T2.iucr_no WHERE T2.index_code = ? AND T1.latitude = ?", (longitude, index_code, latitude))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the top commander by number of crimes where arrest status is specified
@app.get("/v1/chicago_crime/top_commander_by_arrest_status", operation_id="get_top_commander_by_arrest_status", summary="Retrieves the commander with the highest number of reported crimes where the arrest status is specified. The arrest status is used to filter the crimes considered for this ranking. The commander's name is returned as the result.")
async def get_top_commander_by_arrest_status(arrest: str = Query(..., description="Arrest status of the crime (e.g., 'FALSE')")):
    cursor.execute("SELECT T2.commander FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no WHERE T1.arrest = ? GROUP BY T2.commander ORDER BY COUNT(T1.report_no) DESC LIMIT 1", (arrest,))
    result = cursor.fetchone()
    if not result:
        return {"commander": []}
    return {"commander": result[0]}

# Endpoint to get the primary and secondary descriptions of crimes at a specific location
@app.get("/v1/chicago_crime/crime_descriptions_by_location", operation_id="get_crime_descriptions_by_location", summary="Retrieve the most frequently occurring primary and secondary descriptions of crimes that have taken place at a specific location. The location is identified by its description, such as 'AIRCRAFT'.")
async def get_crime_descriptions_by_location(location_description: str = Query(..., description="Location description of the crime (e.g., 'AIRCRAFT')")):
    cursor.execute("SELECT T2.primary_description, T2.secondary_description FROM Crime AS T1 INNER JOIN IUCR AS T2 ON T1.iucr_no = T2.iucr_no WHERE T1.location_description = ? GROUP BY T1.iucr_no ORDER BY COUNT(T1.iucr_no) DESC LIMIT 1", (location_description,))
    result = cursor.fetchone()
    if not result:
        return {"descriptions": []}
    return {"primary_description": result[0], "secondary_description": result[1]}

# Endpoint to get the top district by number of crimes at a specific location within specified districts
@app.get("/v1/chicago_crime/top_district_by_location_and_districts", operation_id="get_top_district_by_location_and_districts", summary="Retrieves the district with the highest number of crimes at a specific location, considering only the crimes that occurred within the provided districts. The location is identified by its description, and the districts are specified by their names.")
async def get_top_district_by_location_and_districts(district_name1: str = Query(..., description="First district name"), district_name2: str = Query(..., description="Second district name"), location_description: str = Query(..., description="Location description of the crime (e.g., 'LIBRARY')")):
    cursor.execute("SELECT T1.district_name FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no WHERE T1.district_name IN (?, ?) AND T2.location_description = ? GROUP BY T1.district_name ORDER BY COUNT(T2.district_no) DESC LIMIT 1", (district_name1, district_name2, location_description))
    result = cursor.fetchone()
    if not result:
        return {"district_name": []}
    return {"district_name": result[0]}

# Endpoint to get the sum of arrests for a specific crime description and location
@app.get("/v1/chicago_crime/sum_arrests_by_crime_description_and_location", operation_id="get_sum_arrests_by_crime_description_and_location", summary="Retrieve the total number of arrests for a specific crime type and location. The crime type is defined by its primary and secondary descriptions, and the arrest status is also considered. The location description is used to filter the results.")
async def get_sum_arrests_by_crime_description_and_location(arrest: str = Query(..., description="Arrest status of the crime (e.g., 'TRUE')"), location_description: str = Query(..., description="Location description of the crime (e.g., 'DAY CARE CENTER')"), secondary_description: str = Query(..., description="Secondary description of the crime (e.g., 'FORCIBLE ENTRY')"), primary_description: str = Query(..., description="Primary description of the crime (e.g., 'BURGLARY')")):
    cursor.execute("SELECT SUM(CASE WHEN T2.arrest = ? THEN 1 ELSE 0 END) FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T1.iucr_no = T2.iucr_no WHERE T2.location_description = ? AND T1.secondary_description = ? AND T1.primary_description = ?", (arrest, location_description, secondary_description, primary_description))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the top district by number of domestic crimes
@app.get("/v1/chicago_crime/top_district_by_domestic_crimes", operation_id="get_top_district_by_domestic_crimes", summary="Retrieves the district with the highest number of domestic crimes. The operation filters crimes based on a specified domestic status and returns the district name with the most occurrences of such crimes.")
async def get_top_district_by_domestic_crimes(domestic: str = Query(..., description="Domestic status of the crime (e.g., 'TRUE')")):
    cursor.execute("SELECT T2.district_name FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no WHERE T1.domestic = ? GROUP BY T2.district_name ORDER BY COUNT(T1.district_no) DESC LIMIT 1", (domestic,))
    result = cursor.fetchone()
    if not result:
        return {"district_name": []}
    return {"district_name": result[0]}

# Endpoint to get the location descriptions of crimes in the community area with the minimum population
@app.get("/v1/chicago_crime/location_descriptions_min_population", operation_id="get_location_descriptions_min_population", summary="Retrieves the distinct descriptions of crime locations in the community area with the smallest population. The data is fetched from the Community_Area and Crime tables, which are joined on the community_area_no field. The community area with the minimum population is determined, and only the crime locations with non-null descriptions are considered.")
async def get_location_descriptions_min_population():
    cursor.execute("SELECT T2.location_description FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no WHERE T1.population = ( SELECT MIN(population) FROM Community_Area ) AND T2.location_description IS NOT NULL GROUP BY T2.location_description")
    result = cursor.fetchall()
    if not result:
        return {"location_descriptions": []}
    return {"location_descriptions": [row[0] for row in result]}

# Endpoint to get the sum of crimes with a specific description and arrest status
@app.get("/v1/chicago_crime/sum_crimes_by_description_and_arrest", operation_id="get_sum_crimes_by_description_and_arrest", summary="Retrieves the total count of crimes that match a specified description and arrest status. The operation filters crimes based on the provided description and arrest status, then calculates the sum of matching crimes.")
async def get_sum_crimes_by_description_and_arrest(description: str = Query(..., description="Description of the crime (e.g., '%The violation of laws%')"), arrest: str = Query(..., description="Arrest status of the crime (e.g., 'FALSE')")):
    cursor.execute("SELECT SUM(CASE WHEN T1.description LIKE ? THEN 1 ELSE 0 END) FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no WHERE T2.Arrest = ?", (description, arrest))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the latitude and longitude of the most recent crime with a specific FBI code title and community area
@app.get("/v1/chicago_crime/crime_location_by_fbi_code_and_community", operation_id="get_crime_location_by_fbi_code_and_community", summary="Retrieves the geographical coordinates of the most recent crime incident that matches a specific FBI code title and community area. The operation filters crimes based on the provided FBI code title, community area name, and community area number, and returns the latitude and longitude of the most recent crime that meets these criteria.")
async def get_crime_location_by_fbi_code_and_community(title: str = Query(..., description="Title of the FBI code (e.g., 'Simple Assault')"), community_area_name: str = Query(..., description="Name of the community area (e.g., 'Chatham')"), community_area_no: int = Query(..., description="Number of the community area (e.g., 44)")):
    cursor.execute("SELECT T2.latitude, T2.longitude FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no INNER JOIN Community_Area AS T3 ON T2.community_area_no = T3.community_area_no WHERE T1.title = ? AND T3.community_area_name = ? AND T3.community_area_no = ? ORDER BY T2.latitude DESC, T2.longitude DESC LIMIT 1", (title, community_area_name, community_area_no))
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"latitude": result[0], "longitude": result[1]}

# Endpoint to get the top community area by number of crimes with a specific FBI code description and side
@app.get("/v1/chicago_crime/top_community_area_by_fbi_code_and_side", operation_id="get_top_community_area_by_fbi_code_and_side", summary="Retrieve the community area with the highest number of crimes that match a specific FBI code description and side. The side parameter filters the results to a specific side of the community area, while the description parameter narrows the results to a specific type of crime. The operation returns the name of the community area that meets these criteria.")
async def get_top_community_area_by_fbi_code_and_side(side: str = Query(..., description="Side of the community area (e.g., 'South')"), description: str = Query(..., description="Description of the FBI code (e.g., 'The unlawful taking, carrying, leading, or riding away of property FROM the possession or constructive possession of another person.')")):
    cursor.execute("SELECT T3.community_area_name FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no INNER JOIN Community_Area AS T3 ON T2.community_area_no = T3.community_area_no WHERE T3.side = ? AND T1.description = ? GROUP BY T3.community_area_name ORDER BY COUNT(T1.fbi_code_no) DESC LIMIT 1", (side, description))
    result = cursor.fetchone()
    if not result:
        return {"community_area_name": []}
    return {"community_area_name": result[0]}

# Endpoint to get the sum of crimes in a specific community area against a specific crime category
@app.get("/v1/chicago_crime/sum_crimes_community_area_crime_against", operation_id="get_sum_crimes_community_area_crime_against", summary="Retrieve the total count of a specific crime category that occurred in a designated community area. This operation calculates the sum of crimes based on the provided community area name and crime category.")
async def get_sum_crimes_community_area_crime_against(community_area_name: str = Query(..., description="Name of the community area"), crime_against: str = Query(..., description="Crime category")):
    cursor.execute("SELECT SUM(CASE WHEN T3.community_area_name = ? THEN 1 ELSE 0 END) FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no INNER JOIN Community_Area AS T3 ON T2.community_area_no = T3.community_area_no WHERE T1.crime_against = ?", (community_area_name, crime_against))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the average count of FBI codes for crimes in a specific month and year
@app.get("/v1/chicago_crime/avg_fbi_codes_month_year", operation_id="get_avg_fbi_codes_month_year", summary="Retrieves the average count of unique crime types that occurred in a specific month and year. The calculation is based on the FBI codes associated with each crime, which are used to categorize different types of crimes. The month and year are determined by the first and last four characters of the crime's date, respectively.")
async def get_avg_fbi_codes_month_year(month: str = Query(..., description="Month (first character of the date)"), year: str = Query(..., description="Year (last four characters of the date)")):
    cursor.execute("SELECT CAST(COUNT(T1.fbi_code_no) AS REAL) / 4 FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no WHERE SUBSTR(T2.date, 1, 1) = ? AND SUBSTR(T2.date, 5, 4) = ?", (month, year))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get community area names with a population greater than a specified value
@app.get("/v1/chicago_crime/community_area_names_population", operation_id="get_community_area_names_population", summary="Retrieves the names of community areas with a population exceeding a specified threshold. The number of results returned can be limited to a specified maximum. This operation is useful for identifying larger community areas based on population size.")
async def get_community_area_names_population(population: int = Query(..., description="Population threshold"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT community_area_name FROM Community_Area WHERE population > ? LIMIT ?", (population, limit))
    result = cursor.fetchall()
    if not result:
        return {"community_area_names": []}
    return {"community_area_names": [row[0] for row in result]}

# Endpoint to get the difference in population between two sides of the city for areas with a population greater than a specified value
@app.get("/v1/chicago_crime/population_diff_sides", operation_id="get_population_diff_sides", summary="Get the difference in population between two sides of the city for areas with a population greater than a specified value")
async def get_population_diff_sides(side1: str = Query(..., description="First side of the city"), side2: str = Query(..., description="Second side of the city"), population: int = Query(..., description="Population threshold")):
    cursor.execute("SELECT SUM(CASE WHEN side = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN side = ? THEN 1 ELSE 0 END) AS DIFF FROM Community_Area WHERE population > ?", (side1, side2, population))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get contact information for a specific district
@app.get("/v1/chicago_crime/district_contact_info", operation_id="get_district_contact_info", summary="Retrieves the contact information, including phone, fax, TTY, and Twitter handle, for a specific district in Chicago. The district is identified by its name.")
async def get_district_contact_info(district_name: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT phone, fax, tty, twitter FROM District WHERE district_name = ?", (district_name,))
    result = cursor.fetchone()
    if not result:
        return {"contact_info": []}
    return {"contact_info": {"phone": result[0], "fax": result[1], "tty": result[2], "twitter": result[3]}}

# Endpoint to get primary and secondary descriptions for a specific IUCR code
@app.get("/v1/chicago_crime/iucr_descriptions", operation_id="get_iucr_descriptions", summary="Retrieves the primary and secondary descriptions associated with a specific IUCR code number. This operation allows users to obtain detailed information about a particular crime category, as identified by its IUCR code.")
async def get_iucr_descriptions(iucr_no: int = Query(..., description="IUCR code number")):
    cursor.execute("SELECT primary_description, secondary_description FROM IUCR WHERE iucr_no = ?", (iucr_no,))
    result = cursor.fetchone()
    if not result:
        return {"descriptions": []}
    return {"descriptions": {"primary": result[0], "secondary": result[1]}}

# Endpoint to get the percentage of a specific primary description in IUCR codes with a given index code
@app.get("/v1/chicago_crime/percentage_primary_description", operation_id="get_percentage_primary_description", summary="Retrieves the percentage of a specific primary description in IUCR codes associated with a given index code. The primary description and index code are provided as input parameters.")
async def get_percentage_primary_description(primary_description: str = Query(..., description="Primary description"), index_code: str = Query(..., description="Index code")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN primary_description = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM IUCR WHERE index_code = ?", (primary_description, index_code))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total population of specified neighborhoods
@app.get("/v1/chicago_crime/total_population_of_neighborhoods", operation_id="get_total_population", summary="Retrieve the total population count for up to six specified neighborhoods in Chicago. The operation aggregates population data from the Community Area table, which is linked to the Neighborhood table via the community area number. The input parameters correspond to the names of the neighborhoods for which population data is requested.")
async def get_total_population(neighborhood_name1: str = Query(..., description="Name of the first neighborhood"), neighborhood_name2: str = Query(..., description="Name of the second neighborhood"), neighborhood_name3: str = Query(..., description="Name of the third neighborhood"), neighborhood_name4: str = Query(..., description="Name of the fourth neighborhood"), neighborhood_name5: str = Query(..., description="Name of the fifth neighborhood"), neighborhood_name6: str = Query(..., description="Name of the sixth neighborhood")):
    cursor.execute("SELECT SUM(T2.population) AS sum FROM Neighborhood AS T1 INNER JOIN Community_Area AS T2 ON T1.community_area_no = T2.community_area_no WHERE T1.neighborhood_name = ? OR T1.neighborhood_name = ? OR T1.neighborhood_name = ? OR T1.neighborhood_name = ? OR T1.neighborhood_name = ? OR T1.neighborhood_name = ?", (neighborhood_name1, neighborhood_name2, neighborhood_name3, neighborhood_name4, neighborhood_name5, neighborhood_name6))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the count of crimes with a specific title
@app.get("/v1/chicago_crime/crime_count_by_title", operation_id="get_crime_count_by_title", summary="Retrieves the total count of a specific crime type, as identified by its title, from the Chicago crime database. The operation uses the provided crime title to filter and aggregate the data.")
async def get_crime_count_by_title(title: str = Query(..., description="Title of the crime")):
    cursor.execute("SELECT SUM(CASE WHEN T2.title = ? THEN 1 ELSE 0 END) FROM Crime AS T1 INNER JOIN FBI_Code AS T2 ON T1.fbi_code_no = T2.fbi_code_no", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get case numbers of crimes with specific criteria
@app.get("/v1/chicago_crime/case_numbers_by_criteria", operation_id="get_case_numbers_by_criteria", summary="Retrieves up to three case numbers of crimes that match the specified title, arrest status, and crime against category. The operation filters crimes based on the provided title, arrest status, and crime against category, and returns the case numbers of the matching crimes.")
async def get_case_numbers_by_criteria(title: str = Query(..., description="Title of the crime"), arrest: str = Query(..., description="Arrest status"), crime_against: str = Query(..., description="Crime against category")):
    cursor.execute("SELECT T2.case_number FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no WHERE T1.title = ? AND T2.arrest = ? AND T1.crime_against = ? LIMIT 3", (title, arrest, crime_against))
    result = cursor.fetchall()
    if not result:
        return {"case_numbers": []}
    return {"case_numbers": [row[0] for row in result]}

# Endpoint to get case numbers of crimes with specific criteria
@app.get("/v1/chicago_crime/case_numbers_by_criteria_v2", operation_id="get_case_numbers_by_criteria_v2", summary="Retrieves the case numbers of up to three crimes that match the specified title, crime against category, and arrest status. The operation filters crimes based on the provided title, crime against category, and arrest status, and returns the case numbers of the matching crimes.")
async def get_case_numbers_by_criteria_v2(title: str = Query(..., description="Title of the crime"), crime_against: str = Query(..., description="Crime against category"), arrest: str = Query(..., description="Arrest status")):
    cursor.execute("SELECT T1.case_number FROM Crime AS T1 INNER JOIN FBI_Code AS T2 ON T1.fbi_code_no = T2.fbi_code_no WHERE T2.title = ? AND T2.crime_against = ? AND T1.arrest = ? LIMIT 3", (title, crime_against, arrest))
    result = cursor.fetchall()
    if not result:
        return {"case_numbers": []}
    return {"case_numbers": [row[0] for row in result]}

# Endpoint to get the primary description of a crime by case number
@app.get("/v1/chicago_crime/primary_description_by_case_number", operation_id="get_primary_description_by_case_number", summary="Retrieves the primary description of a specific crime case, identified by its unique case number. The case number is used to locate the corresponding crime record and extract its primary description.")
async def get_primary_description_by_case_number(case_number: str = Query(..., description="Case number of the crime")):
    cursor.execute("SELECT T1.primary_description FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T1.iucr_no = T2.iucr_no WHERE T2.case_number = ?", (case_number,))
    result = cursor.fetchone()
    if not result:
        return {"primary_description": []}
    return {"primary_description": result[0]}

# Endpoint to get community area names for crimes with a specific title
@app.get("/v1/chicago_crime/community_area_names_by_crime_title", operation_id="get_community_area_names_by_crime_title", summary="Retrieve the names of up to three community areas where a specific crime, identified by its title, has been reported. This operation fetches the community area names by joining the FBI_Code, Crime, and Community_Area tables using the respective code numbers and area numbers. The result is a list of community area names where the specified crime has occurred.")
async def get_community_area_names_by_crime_title(title: str = Query(..., description="Title of the crime")):
    cursor.execute("SELECT T3.community_area_name FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no INNER JOIN Community_Area AS T3 ON T2.community_area_no = T3.community_area_no WHERE T1.title = ? LIMIT 3", (title,))
    result = cursor.fetchall()
    if not result:
        return {"community_area_names": []}
    return {"community_area_names": [row[0] for row in result]}

# Endpoint to get FBI codes and titles for crimes in a specific community area
@app.get("/v1/chicago_crime/fbi_codes_titles_by_community_area", operation_id="get_fbi_codes_titles_by_community_area", summary="Retrieve a list of FBI codes and their corresponding crime titles for a specified community area. The operation filters crimes based on the provided community area name and returns unique FBI codes along with their associated crime titles.")
async def get_fbi_codes_titles_by_community_area(community_area_name: str = Query(..., description="Name of the community area")):
    cursor.execute("SELECT T1.fbi_code_no, T1.title FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no INNER JOIN Community_Area AS T3 ON T2.community_area_no = T3.community_area_no WHERE T3.community_area_name = ? GROUP BY T1.fbi_code_no, T1.title", (community_area_name,))
    result = cursor.fetchall()
    if not result:
        return {"fbi_codes_titles": []}
    return {"fbi_codes_titles": [{"fbi_code_no": row[0], "title": row[1]} for row in result]}

# Endpoint to get district information by case number
@app.get("/v1/chicago_crime/district_info_by_case_number", operation_id="get_district_info_by_case_number", summary="Retrieves district information associated with a specific crime case number. The response includes the district number and name. The case number is used to filter the results.")
async def get_district_info_by_case_number(case_number: str = Query(..., description="Case number of the crime")):
    cursor.execute("SELECT T1.district_no, T1.district_name FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no WHERE T2.case_number = ? GROUP BY T1.district_no, T1.district_name", (case_number,))
    result = cursor.fetchall()
    if not result:
        return {"district_info": []}
    return {"district_info": [{"district_no": row[0], "district_name": row[1]} for row in result]}

# Endpoint to get the percentage of crimes with a specific title and location description
@app.get("/v1/chicago_crime/percentage_crimes_by_title_location", operation_id="get_percentage_crimes_by_title_location", summary="Retrieves the percentage of crimes with a specific title and location description from the Chicago crime database. The operation calculates the proportion of crimes with the given title and location description relative to the total number of crimes in the database. The input parameters specify the title of the crime and the location description to filter the results.")
async def get_percentage_crimes_by_title_location(title: str = Query(..., description="Title of the crime"), location_description: str = Query(..., description="Location description of the crime")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.title = ? AND T1.location_description = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.fbi_code_no) FROM Crime AS T1 INNER JOIN FBI_Code AS T2 ON T1.fbi_code_no = T2.fbi_code_no", (title, location_description))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ward with the highest population
@app.get("/v1/chicago_crime/ward_with_highest_population", operation_id="get_ward_with_highest_population", summary="Retrieves the ward number with the highest population in Chicago. This operation returns the ward with the largest resident count, providing a snapshot of the most populous area in the city.")
async def get_ward_with_highest_population():
    cursor.execute("SELECT ward_no FROM Ward ORDER BY Population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ward_no": []}
    return {"ward_no": result[0]}

# Endpoint to get the beat and location description for a specific case number
@app.get("/v1/chicago_crime/crime_details_by_case_number", operation_id="get_crime_details_by_case_number", summary="Retrieves the beat and location description associated with a specific crime case number. The case number is used to identify the crime and fetch the corresponding beat and location details.")
async def get_crime_details_by_case_number(case_number: str = Query(..., description="Case number of the crime")):
    cursor.execute("SELECT beat, location_description FROM Crime WHERE case_number = ?", (case_number,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the FBI code number based on description
@app.get("/v1/chicago_crime/fbi_code_by_description", operation_id="get_fbi_code_by_description", summary="Retrieves the FBI code number associated with the provided description. This operation searches for the specified description in the FBI_Code table and returns the corresponding FBI code number.")
async def get_fbi_code_by_description(description: str = Query(..., description="Description of the FBI code")):
    cursor.execute("SELECT fbi_code_no FROM FBI_Code WHERE description = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"fbi_code_no": []}
    return {"fbi_code_no": result[0]}

# Endpoint to get ward office addresses for a specific community area name
@app.get("/v1/chicago_crime/ward_office_addresses_by_community_area", operation_id="get_ward_office_addresses_by_community_area", summary="Retrieves the addresses of ward offices associated with a specified community area. The operation returns up to five unique addresses, aggregated from the crime data linked to the community area. The community area is identified by its name.")
async def get_ward_office_addresses_by_community_area(community_area_name: str = Query(..., description="Name of the community area")):
    cursor.execute("SELECT T3.ward_office_address FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no INNER JOIN Ward AS T3 ON T2.ward_no = T3.ward_no WHERE T1.community_area_name = ? GROUP BY T3.ward_office_address LIMIT 5", (community_area_name,))
    result = cursor.fetchall()
    if not result:
        return {"ward_office_addresses": []}
    return {"ward_office_addresses": result}

# Endpoint to get district address and commander based on location description and beat
@app.get("/v1/chicago_crime/district_info_by_location_and_beat", operation_id="get_district_info_by_location_and_beat", summary="Retrieves the address and commander of a district based on the location description and beat number of a crime. This operation uses the provided location description and beat number to identify the corresponding district, and then returns the address and commander associated with that district.")
async def get_district_info_by_location_and_beat(location_description: str = Query(..., description="Location description of the crime"), beat: int = Query(..., description="Beat number of the crime")):
    cursor.execute("SELECT T2.address, T2.commander FROM Crime AS T1 INNER JOIN District AS T2 ON T1.district_no = T2.district_no WHERE T1.location_description = ? AND T1.beat = ?", (location_description, beat))
    result = cursor.fetchall()
    if not result:
        return {"district_info": []}
    return {"district_info": result}

# Endpoint to get neighborhood names for a specific crime report number
@app.get("/v1/chicago_crime/neighborhood_names_by_report_number", operation_id="get_neighborhood_names_by_report_number", summary="Retrieve the names of neighborhoods associated with a specific crime report number. This operation fetches the community area number from the crime report and uses it to look up the corresponding neighborhood name in the Neighborhood table. The report number is required as an input parameter.")
async def get_neighborhood_names_by_report_number(report_no: int = Query(..., description="Report number of the crime")):
    cursor.execute("SELECT T3.neighborhood_name FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no INNER JOIN Neighborhood AS T3 ON T2.community_area_no = T3.community_area_no WHERE T2.report_no = ?", (report_no,))
    result = cursor.fetchall()
    if not result:
        return {"neighborhood_names": []}
    return {"neighborhood_names": result}

# Endpoint to get FBI code descriptions for a specific crime report number
@app.get("/v1/chicago_crime/fbi_code_descriptions_by_report_number", operation_id="get_fbi_code_descriptions_by_report_number", summary="Retrieves the FBI code descriptions associated with a specific crime report number. The report number is used to identify the relevant crime record and its corresponding FBI code. The descriptions provide detailed information about the nature of the crime.")
async def get_fbi_code_descriptions_by_report_number(report_no: int = Query(..., description="Report number of the crime")):
    cursor.execute("SELECT T1.description FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no WHERE T2.report_no = ?", (report_no,))
    result = cursor.fetchall()
    if not result:
        return {"fbi_code_descriptions": []}
    return {"fbi_code_descriptions": result}

# Endpoint to get the count of crimes under a specific commander
@app.get("/v1/chicago_crime/crime_count_by_commander", operation_id="get_crime_count_by_commander", summary="Retrieves the total number of crimes associated with a specific commander. The commander's name is used to filter the data and calculate the crime count.")
async def get_crime_count_by_commander(commander: str = Query(..., description="Name of the commander")):
    cursor.execute("SELECT SUM(CASE WHEN T1.commander = ? THEN 1 ELSE 0 END) FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no", (commander,))
    result = cursor.fetchone()
    if not result:
        return {"crime_count": []}
    return {"crime_count": result[0]}

# Endpoint to get the percentage of crimes in a specific location description within a neighborhood
@app.get("/v1/chicago_crime/crime_percentage_by_location_and_neighborhood", operation_id="get_crime_percentage_by_location_and_neighborhood", summary="Retrieves the percentage of crimes that match a specific location description within a given neighborhood. The operation calculates this percentage by comparing the count of crimes with the specified location description to the total number of crimes in the neighborhood.")
async def get_crime_percentage_by_location_and_neighborhood(location_description: str = Query(..., description="Location description of the crime"), neighborhood_name: str = Query(..., description="Name of the neighborhood")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.location_description = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.location_description) AS persent FROM Community_Area AS T1 INNER JOIN Crime AS T2 ON T1.community_area_no = T2.community_area_no INNER JOIN Neighborhood AS T3 ON T2.community_area_no = T3.community_area_no WHERE T3.neighborhood_name = ?", (location_description, neighborhood_name))
    result = cursor.fetchone()
    if not result:
        return {"crime_percentage": []}
    return {"crime_percentage": result[0]}

# Endpoint to get the count of IUCR records based on primary description
@app.get("/v1/chicago_crime/iucr_count_by_primary_description", operation_id="get_iucr_count", summary="Retrieves the total number of IUCR records that match the provided primary description. The primary description is a key attribute of each IUCR record.")
async def get_iucr_count(primary_description: str = Query(..., description="Primary description of the IUCR record")):
    cursor.execute("SELECT COUNT(*) FROM IUCR WHERE primary_description = ?", (primary_description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of IUCR records based on index code
@app.get("/v1/chicago_crime/iucr_count_by_index_code", operation_id="get_iucr_count_by_index_code", summary="Retrieves the total number of IUCR records associated with a specific index code. The index code is used to filter the records and calculate the count.")
async def get_iucr_count_by_index_code(index_code: str = Query(..., description="Index code of the IUCR record")):
    cursor.execute("SELECT COUNT(*) FROM IUCR WHERE index_code = ?", (index_code,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of crimes with specific alderman details and arrest status
@app.get("/v1/chicago_crime/crime_count_by_alderman_and_arrest", operation_id="get_crime_count_by_alderman_and_arrest", summary="Retrieves the total count of crimes committed in the wards represented by a specific alderman, filtered by arrest status. The alderman is identified by their first and last names.")
async def get_crime_count_by_alderman_and_arrest(alderman_last_name: str = Query(..., description="Last name of the alderman"), arrest: str = Query(..., description="Arrest status (TRUE or FALSE)"), alderman_first_name: str = Query(..., description="First name of the alderman")):
    cursor.execute("SELECT SUM(CASE WHEN T1.alderman_last_name = ? THEN 1 ELSE 0 END) FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE T2.arrest = ? AND T1.alderman_first_name = ?", (alderman_last_name, arrest, alderman_first_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get ward details with the highest crime count in a specific month and year
@app.get("/v1/chicago_crime/ward_with_highest_crime_count", operation_id="get_ward_with_highest_crime_count", summary="Retrieves the ward with the highest crime count for a specified month and year. The response includes the ward number, alderman's first name, last name, and name suffix. The data is filtered by the provided month and year.")
async def get_ward_with_highest_crime_count(month: str = Query(..., description="Month (single digit)"), year: str = Query(..., description="Year (4 digits)")):
    cursor.execute("SELECT T1.ward_no, T1.alderman_first_name, T1.alderman_last_name, T1.alderman_name_suffix FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE SUBSTR(T2.date, 1, 1) = ? AND SUBSTR(T2.date, 5, 4) = ? GROUP BY T1.ward_no ORDER BY COUNT(T1.ward_no) DESC LIMIT 1", (month, year))
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the count of wards with domestic crimes ordered by population
@app.get("/v1/chicago_crime/ward_count_domestic_crimes_by_population", operation_id="get_ward_count_domestic_crimes_by_population", summary="Retrieves the number of wards with domestic crimes, sorted by the population of the most populous ward. The domestic status parameter determines whether to include only domestic crimes or all crimes.")
async def get_ward_count_domestic_crimes_by_population(domestic: str = Query(..., description="Domestic status (TRUE or FALSE)")):
    cursor.execute("SELECT COUNT(T1.ward_no) AS num FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE T2.domestic = ? ORDER BY T1.Population = ( SELECT Population FROM Ward ORDER BY Population DESC LIMIT 1 )", (domestic,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get crime locations based on alderman details
@app.get("/v1/chicago_crime/crime_locations_by_alderman", operation_id="get_crime_locations_by_alderman", summary="Retrieves the geographical coordinates of crimes committed within the ward of a specific alderman in Chicago. The operation requires the first and last name of the alderman to identify the corresponding ward and filter the crime data accordingly.")
async def get_crime_locations_by_alderman(alderman_first_name: str = Query(..., description="First name of the alderman"), alderman_last_name: str = Query(..., description="Last name of the alderman")):
    cursor.execute("SELECT T2.latitude, T2.longitude FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE T1.alderman_first_name = ? AND T1.alderman_last_name = ? AND T2.latitude IS NOT NULL AND T2.longitude IS NOT NULL", (alderman_first_name, alderman_last_name))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": result}

# Endpoint to get alderman details and crime count based on specific criteria
@app.get("/v1/chicago_crime/alderman_crime_count_by_criteria", operation_id="get_alderman_crime_count_by_criteria", summary="Retrieves the number of crimes and details of the alderman based on specified criteria. The criteria include a specific month, year, and alderman's first and last name. If no month and year are provided, the operation will return the total number of crimes and alderman details based on the alderman's first and last name.")
async def get_alderman_crime_count_by_criteria(month: str = Query(..., description="Month (single digit)"), year: str = Query(..., description="Year (4 digits)"), alderman_first_name_1: str = Query(..., description="First name of the alderman (criteria 1)"), alderman_last_name_1: str = Query(..., description="Last name of the alderman (criteria 1)"), alderman_first_name_2: str = Query(..., description="First name of the alderman (criteria 2)"), alderman_last_name_2: str = Query(..., description="Last name of the alderman (criteria 2)")):
    cursor.execute("SELECT T1.alderman_first_name, T1.alderman_last_name, COUNT(T1.ward_no) AS num FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE (SUBSTR(T2.date, 1, 1) = ? AND SUBSTR(T2.date, 5, 4) = ? AND T1.alderman_first_name = ? AND T1.alderman_last_name = ?) OR (T1.alderman_first_name = ? AND T1.alderman_last_name = ?) GROUP BY T1.ward_no", (month, year, alderman_first_name_1, alderman_last_name_1, alderman_first_name_2, alderman_last_name_2))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get case numbers based on primary description
@app.get("/v1/chicago_crime/case_numbers_by_primary_description", operation_id="get_case_numbers_by_primary_description", summary="Retrieves the case numbers associated with the specified primary description of the IUCR record. This operation allows you to filter case numbers based on the primary description, providing a focused view of cases that match the given description.")
async def get_case_numbers_by_primary_description(primary_description: str = Query(..., description="Primary description of the IUCR record")):
    cursor.execute("SELECT T2.case_number FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T1.iucr_no = T2.iucr_no WHERE T1.primary_description = ?", (primary_description,))
    result = cursor.fetchall()
    if not result:
        return {"case_numbers": []}
    return {"case_numbers": result}

# Endpoint to get the count of crimes with specific primary description and arrest status
@app.get("/v1/chicago_crime/crime_count_by_primary_description_and_arrest", operation_id="get_crime_count_by_primary_description_and_arrest", summary="Retrieves the total count of crimes that match a specified primary description and arrest status. The primary description refers to the main category of the crime, while the arrest status indicates whether an arrest was made in relation to the crime. This operation provides a quantitative overview of crimes based on these two criteria.")
async def get_crime_count_by_primary_description_and_arrest(arrest: str = Query(..., description="Arrest status (TRUE or FALSE)"), primary_description: str = Query(..., description="Primary description of the IUCR record")):
    cursor.execute("SELECT SUM(CASE WHEN T2.arrest = ? THEN 1 ELSE 0 END) FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T1.iucr_no = T2.iucr_no WHERE T1.primary_description = ?", (arrest, primary_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get case numbers for a specific FBI code title
@app.get("/v1/chicago_crime/case_numbers_by_fbi_code_title", operation_id="get_case_numbers_by_fbi_code_title", summary="Retrieves the case numbers associated with a specific FBI code title. The operation filters the FBI_Code table based on the provided title and joins it with the Crime table to obtain the corresponding case numbers.")
async def get_case_numbers_by_fbi_code_title(title: str = Query(..., description="Title of the FBI code")):
    cursor.execute("SELECT T2.case_number FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"case_numbers": []}
    return {"case_numbers": [row[0] for row in result]}

# Endpoint to get the sum of crimes against property for a specific year and month
@app.get("/v1/chicago_crime/sum_crimes_against_property", operation_id="get_sum_crimes_against_property", summary="Retrieves the total count of a specific type of property crime that occurred in a given year and month. The operation filters the crime data based on the provided year, month, and type of property crime, and then calculates the sum of these crimes.")
async def get_sum_crimes_against_property(year: str = Query(..., description="Year in 'YYYY' format"), crime_against: str = Query(..., description="Type of crime against"), month: str = Query(..., description="Month in 'M' format")):
    cursor.execute("SELECT SUM(CASE WHEN SUBSTR(T2.date, 5, 4) = ? THEN 1 ELSE 0 END) FROM FBI_Code AS T1 INNER JOIN Crime AS T2 ON T1.fbi_code_no = T2.fbi_code_no WHERE T1.crime_against = ? AND SUBSTR(T2.date, 1, 1) = ?", (year, crime_against, month))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of crimes for a specific commander and month
@app.get("/v1/chicago_crime/sum_crimes_by_commander", operation_id="get_sum_crimes_by_commander", summary="Retrieves the total number of crimes committed in a specific year, month, and under a particular commander. The response is based on the aggregated data from the Crime table, filtered by the commander's name and the specified month and year.")
async def get_sum_crimes_by_commander(year: str = Query(..., description="Year in 'YYYY' format"), commander: str = Query(..., description="Name of the commander"), month: str = Query(..., description="Month in 'M' format")):
    cursor.execute("SELECT SUM(CASE WHEN SUBSTR(T2.date, 5, 4) = ? THEN 1 ELSE 0 END) FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no WHERE T1.commander = ? AND SUBSTR(T2.date, 1, 1) = ?", (year, commander, month))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get commanders for specific criteria
@app.get("/v1/chicago_crime/commanders_by_criteria", operation_id="get_commanders_by_criteria", summary="Retrieves the commanders who have been assigned to districts where crimes occurred in a specific month and year. The commanders are identified by their names, and the month and year are provided in 'M' and 'YYYY' formats, respectively.")
async def get_commanders_by_criteria(commander1: str = Query(..., description="Name of the first commander"), commander2: str = Query(..., description="Name of the second commander"), month: str = Query(..., description="Month in 'M' format"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T1.commander FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no WHERE T1.commander IN (?, ?) AND SUBSTR(T2.date, 1, 1) = ? AND SUBSTR(T2.date, 5, 4) = ? GROUP BY T1.commander", (commander1, commander2, month, year))
    result = cursor.fetchall()
    if not result:
        return {"commanders": []}
    return {"commanders": [row[0] for row in result]}

# Endpoint to get blocks for a specific commander
@app.get("/v1/chicago_crime/blocks_by_commander", operation_id="get_blocks_by_commander", summary="Retrieves a list of blocks where crimes have been reported under the supervision of a specific commander. The commander's name is used to filter the data.")
async def get_blocks_by_commander(commander: str = Query(..., description="Name of the commander")):
    cursor.execute("SELECT T2.block FROM District AS T1 INNER JOIN Crime AS T2 ON T1.district_no = T2.district_no WHERE T1.commander = ?", (commander,))
    result = cursor.fetchall()
    if not result:
        return {"blocks": []}
    return {"blocks": [row[0] for row in result]}

# Endpoint to get the average ward count for a specific year
@app.get("/v1/chicago_crime/average_ward_count", operation_id="get_average_ward_count", summary="Retrieves the average monthly count of crimes for the ward with the highest population in a specified year. The year should be provided in 'YYYY' format.")
async def get_average_ward_count(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.ward_no) / 12 AS average FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE T2.date LIKE ? AND T1.Population = ( SELECT MAX(T1.Population) FROM Ward AS T1 INNER JOIN Crime AS T2 ON T1.ward_no = T2.ward_no WHERE T2.date LIKE ? )", (f'%{year}%', f'%{year}%'))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of crimes with a specific primary description and arrest status
@app.get("/v1/chicago_crime/percentage_crimes_by_description_arrest", operation_id="get_percentage_crimes_by_description_arrest", summary="Retrieves the percentage of crimes that match a specified primary description and arrest status. This operation calculates the proportion of crimes with the given primary description and arrest status against the total number of crimes in the database.")
async def get_percentage_crimes_by_description_arrest(primary_description: str = Query(..., description="Primary description of the crime"), arrest: str = Query(..., description="Arrest status (TRUE or FALSE)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.primary_description = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM IUCR AS T1 INNER JOIN Crime AS T2 ON T1.iucr_no = T2.iucr_no WHERE T2.arrest = ?", (primary_description, arrest))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/chicago_crime/community_area_count_by_side?side=Central",
    "/v1/chicago_crime/community_area_side_by_name?community_area_name=Lincoln%20Square",
    "/v1/chicago_crime/most_common_community_area_side",
    "/v1/chicago_crime/smallest_population_community_area",
    "/v1/chicago_crime/district_commander_by_name?district_name=Central",
    "/v1/chicago_crime/district_email_by_name?district_name=Central",
    "/v1/chicago_crime/community_area_name_by_neighborhood?neighborhood_name=Albany%20Park",
    "/v1/chicago_crime/count_community_area_numbers_by_name?community_area_name=Lincoln%20Square",
    "/v1/chicago_crime/highest_population_neighborhood",
    "/v1/chicago_crime/neighborhood_names_by_side?side=Central",
    "/v1/chicago_crime/crime_locations_by_district?district_name=Central",
    "/v1/chicago_crime/crime_count_by_district?district_name=Central",
    "/v1/chicago_crime/domestic_crime_count_by_district?district_name=Central&domestic=TRUE",
    "/v1/chicago_crime/no_arrest_crime_count_by_district?district_name=Central&arrest=FALSE",
    "/v1/chicago_crime/crime_report_count_most_populous_community",
    "/v1/chicago_crime/crime_count_by_community_area_date?community_area_name=Woodlawn&date_pattern=%251%2F2018%25",
    "/v1/chicago_crime/highest_crime_community_area_by_date?community_area_name1=Woodlawn&community_area_name2=Lincoln%20Square&date_pattern=%251%2F2018%25",
    "/v1/chicago_crime/highest_crime_district_fax_by_date?date_pattern=%251%2F2018%25",
    "/v1/chicago_crime/crime_report_ratio_by_side?side=Central",
    "/v1/chicago_crime/crime_percentage_by_district?district_name=Central",
    "/v1/chicago_crime/district_name_by_case_number?case_number=JB100065",
    "/v1/chicago_crime/district_name_by_coordinates?longitude=-87.63470194&latitude=41.66236555",
    "/v1/chicago_crime/district_commander_by_case_number?case_number=JB524952",
    "/v1/chicago_crime/sum_secondary_description_by_date_primary_description?secondary_description=SIMPLE&date=09/08/2018&primary_description=ASSAULT",
    "/v1/chicago_crime/top_district_by_secondary_description?secondary_description=FIRST%20DEGREE%20MURDER",
    "/v1/chicago_crime/iucr_no_by_case_number?case_number=JB296775",
    "/v1/chicago_crime/community_area_by_descriptions?primary_description=THEFT&secondary_description=POCKET-PICKING",
    "/v1/chicago_crime/alderman_by_case_number?case_number=JB103470",
    "/v1/chicago_crime/neighborhood_names_by_community_area?community_area_name=West%20Englewood",
    "/v1/chicago_crime/neighborhood_count_by_community_area?community_area_name=Roseland",
    "/v1/chicago_crime/crime_description_by_case_number?case_number=JB134191",
    "/v1/chicago_crime/crime_count_by_community_area_with_arrests?community_area_name=North%20Lawndale&arrest=TRUE",
    "/v1/chicago_crime/percentage_crimes_by_secondary_description?secondary_description=$500%20AND%20UNDER&primary_description=THEFT&community_area_name=West%20Englewood",
    "/v1/chicago_crime/percentage_crimes_by_fbi_code_title?fbi_code_title=Larceny&community_area_name=Edgewater",
    "/v1/chicago_crime/crime_count_by_block_and_date?date_pattern=5/%/2018&block=018XX%20S%20KOMENSKY%20AVE",
    "/v1/chicago_crime/most_populous_community_area",
    "/v1/chicago_crime/crime_count_by_location_date_domestic?location_description=ABANDONED%20BUILDING&date_pattern=%252018%25&domestic=TRUE",
    "/v1/chicago_crime/most_populous_side",
    "/v1/chicago_crime/sum_arrests_year_location_fbi_code?year=2018&location_description=ANIMAL%20HOSPITAL&fbi_code_no=08B",
    "/v1/chicago_crime/commander_highest_crime_count?title=Disorderly%20Conduct&fbi_code_no=24",
    "/v1/chicago_crime/crime_title_highest_fbi_code",
    "/v1/chicago_crime/sum_crime_title_district_arrest_location?title=Criminal%20Sexual%20Abuse&district_name=Albany%20Park&arrest=TRUE&location_description=APARTMENT",
    "/v1/chicago_crime/crime_lat_long_community_area_fbi_code_title?community_area_name=Rogers%20Park&title=Robbery&fbi_code_no=3",
    "/v1/chicago_crime/sum_arrests_community_area_iucr_descriptions?community_area_name=West%20Garfield%20Park&arrest=TRUE&secondary_description=SOLICIT%20ON%20PUBLIC%20WAY&primary_description=PROSTITUTION",
    "/v1/chicago_crime/count_domestic_crimes_most_populated_ward?domestic=TRUE&location_description=BAR%20OR%20TAVERN",
    "/v1/chicago_crime/sum_arrests_ward_alderman_domestic?alderman_first_name=Walter&alderman_last_name=Burnett&alderman_name_suffix=Jr.&domestic=TRUE",
    "/v1/chicago_crime/fbi_crime_titles_by_community_area?limit=1",
    "/v1/chicago_crime/ward_office_addresses_by_block?block=010XX%20W%20LAKE%20ST",
    "/v1/chicago_crime/community_area_names_by_iucr_description?primary_description=PROSTITUTION&limit=1",
    "/v1/chicago_crime/sum_crimes_by_alderman_fbi_title_arrest?alderman_last_name=Burke&fbi_title=Vandalism&arrest=TRUE&alderman_first_name=Edward",
    "/v1/chicago_crime/count_crimes_by_date_domestic?date_pattern=5/%/2018%&domestic=TRUE",
    "/v1/chicago_crime/index_codes_by_primary_description?primary_description=HOMICIDE",
    "/v1/chicago_crime/district_commander_email?district_name=Chicago%20Lawn",
    "/v1/chicago_crime/alderman_details_by_population?limit=1",
    "/v1/chicago_crime/community_area_names_by_side?side=Northwest",
    "/v1/chicago_crime/fbi_crime_titles_descriptions_by_crime_against?crime_against=Persons",
    "/v1/chicago_crime/iucr_details?iucr_no=142",
    "/v1/chicago_crime/crime_sum_alderman?alderman_last_name=Reilly&alderman_first_name=Brendan&date_pattern=10/7/2018%",
    "/v1/chicago_crime/arrest_sum_district_location?arrest=TRUE&district_name=Englewood&location_description=RESTAURANT",
    "/v1/chicago_crime/case_details_block?block=0000X%20N%20FRANCISCO%20AVE",
    "/v1/chicago_crime/crime_sum_fbi_code?title=Misc%20Non-Index%20Offense",
    "/v1/chicago_crime/neighborhood_names_community_area?community_area_name=Douglas",
    "/v1/chicago_crime/avg_monthly_crime_reports",
    "/v1/chicago_crime/crime_percentage_fbi_code_side?title=Larceny&side=Central",
    "/v1/chicago_crime/crime_details_descriptions?primary_description=ARSON&secondary_description=BY%20EXPLOSIVE",
    "/v1/chicago_crime/crime_details_descriptions_date?primary_description=DECEPTIVE%20PRACTICE&secondary_description=UNLAWFUL%20USE%20OF%20RECORDED%20SOUND",
    "/v1/chicago_crime/count_crime_reports_by_fbi_code_title_commander_location?fbi_code_title=Criminal%20Sexual%20Assault&commander=Adnardo%20Gutierrez&location_description=RESIDENCE",
    "/v1/chicago_crime/percentage_arrests_domestic_incidents?community_area_name=West%20Pullman&domestic=TRUE",
    "/v1/chicago_crime/domestic_incidents_percentage_and_count?alderman_first_name=Christopher&alderman_last_name=Taliaferro",
    "/v1/chicago_crime/count_wards_by_alderman_first_name?alderman_first_name=James",
    "/v1/chicago_crime/average_population_by_side?side=West",
    "/v1/chicago_crime/most_recent_crime_report_by_alderman?alderman_first_name=Edward&alderman_last_name=Burke",
    "/v1/chicago_crime/count_street_crimes_by_fbi_code_title?fbi_code_title=Homicide%201st%20%26%202nd%20Degree",
    "/v1/chicago_crime/alderman_name_by_report_number?report_no=23769",
    "/v1/chicago_crime/case_numbers_domestic_crimes?community_area_name=Lincoln%20Square&domestic=TRUE",
    "/v1/chicago_crime/sum_crimes_beat_less_than_community_area?beat_threshold=1000&community_area_name=Bridgeport",
    "/v1/chicago_crime/sum_community_areas_population_greater_than_side?population_threshold=60000&side=Far%20North",
    "/v1/chicago_crime/sum_crimes_against_category_community_area?crime_against=Property&community_area_name=Riverdale",
    "/v1/chicago_crime/sum_domestic_crimes_ward_email?domestic=TRUE&ward_email=ward13@cityofchicago.org",
    "/v1/chicago_crime/district_address_by_case_number?case_number=JB107731",
    "/v1/chicago_crime/sum_beats_population_greater_than_side?population_threshold=50000&side=Central",
    "/v1/chicago_crime/case_numbers_by_date_pattern_crime_category?date_pattern=6/%/2018%&crime_against=Society",
    "/v1/chicago_crime/percentage_domestic_crimes_highest_population?domestic=TRUE",
    "/v1/chicago_crime/community_area_side_population?side=Far%20North&min_population=50000&max_population=70000",
    "/v1/chicago_crime/crime_latitude_longitude?location_description=ALLEY&arrest=TRUE",
    "/v1/chicago_crime/district_commander_info?district_name=Ogden",
    "/v1/chicago_crime/fbi_code_description?title=Gambling",
    "/v1/chicago_crime/iucr_percentage_by_index_code?index_code=I",
    "/v1/chicago_crime/most_common_crime_location_description?district_name=Austin",
    "/v1/chicago_crime/ward_to_community_area_side_ratio",
    "/v1/chicago_crime/most_crimes_community_area?location_description=STREET",
    "/v1/chicago_crime/crime_report_to_district_ratio?secondary_description=RECKLESS%20HOMICIDE",
    "/v1/chicago_crime/ward_office_most_crimes_no_arrest?arrest=FALSE",
    "/v1/chicago_crime/case_details_by_secondary_description?secondary_description=CHILD%20ABDUCTION",
    "/v1/chicago_crime/most_common_secondary_description_by_side?side=Northwest",
    "/v1/chicago_crime/least_domestic_crimes_community_area?domestic=TRUE",
    "/v1/chicago_crime/percentage_crimes_by_description_and_fbi_title?description=%25CANNABIS%25&fbi_title=Drug%20Abuse",
    "/v1/chicago_crime/average_cases_per_day?days_in_month=28&date_pattern=2/%/2018%&index_code=N",
    "/v1/chicago_crime/community_area_details_by_secondary_description?secondary_description=SOLICIT%20FOR%20PROSTITUTE",
    "/v1/chicago_crime/percentage_reports_by_fbi_title_and_district?fbi_title=Disorderly%20Conduct&district_name=Harrison",
    "/v1/chicago_crime/crime_rate_difference_by_secondary_descriptions?secondary_description_1=VEHICULAR%20HIJACKING&secondary_description_2=AGGRAVATED%20VEHICULAR%20HIJACKING",
    "/v1/chicago_crime/crime_count_by_longitude?longitude=-87.72658001",
    "/v1/chicago_crime/secondary_descriptions_by_primary_description?primary_description=NARCOTICS",
    "/v1/chicago_crime/alderman_first_names_by_population?min_population=50000",
    "/v1/chicago_crime/report_numbers_by_fbi_code_title?fbi_code_title=Drug%20Abuse",
    "/v1/chicago_crime/crime_count_by_district_and_description?district_name=Calumet&primary_description=WEAPONS%20VIOLATION",
    "/v1/chicago_crime/crime_locations_by_community_area?community_area_name=Belmont%20Cragin",
    "/v1/chicago_crime/crime_count_by_neighborhood?neighborhood_name=Hermosa",
    "/v1/chicago_crime/domestic_crime_count_by_commander?domestic=TRUE&commander=Ronald%20A.%20Pontecore%20Jr.",
    "/v1/chicago_crime/crime_count_by_category_and_district?crime_against=Society&district_name=Wentworth",
    "/v1/chicago_crime/district_phones_by_alderman?alderman_first_name=Emma&alderman_last_name=Mitts",
    "/v1/chicago_crime/crime_count_by_community_area_and_description?community_area_name=Lake%20View&description=The%20theft%20of%20a%20motor%20vehicle.",
    "/v1/chicago_crime/top_district_by_crime_type?primary_description=INTIMIDATION",
    "/v1/chicago_crime/domestic_status_by_community_area?community_area_name=North%20Lawndale&domestic=TRUE",
    "/v1/chicago_crime/ward_no_by_crime_description_and_population?primary_description=INTIMIDATION&secondary_description=EXTORTION&population=55000",
    "/v1/chicago_crime/commander_by_secondary_crime_description?secondary_description=CRIMINAL%20SEXUAL%20ABUSE",
    "/v1/chicago_crime/percentage_non_domestic_cases_by_district?district_name=Jefferson%20Park&domestic=FALSE",
    "/v1/chicago_crime/average_population_by_location_description_and_arrest?location_description=APARTMENT&arrest=FALSE",
    "/v1/chicago_crime/top_wards_by_population",
    "/v1/chicago_crime/count_fbi_codes_by_crime_against?crime_against=Property",
    "/v1/chicago_crime/count_districts_by_zip_code?zip_code=60608",
    "/v1/chicago_crime/crime_against_by_fbi_code_title?title=Criminal%20Sexual%20Abuse",
    "/v1/chicago_crime/community_area_name_by_highest_community_area_no",
    "/v1/chicago_crime/sum_crimes_longitude_index_latitude?longitude=-87.54430496&index_code=I&latitude=41.64820251",
    "/v1/chicago_crime/top_commander_by_arrest_status?arrest=FALSE",
    "/v1/chicago_crime/crime_descriptions_by_location?location_description=AIRCRAFT",
    "/v1/chicago_crime/top_district_by_location_and_districts?district_name1=Deering&district_name2=Near%20West&location_description=LIBRARY",
    "/v1/chicago_crime/sum_arrests_by_crime_description_and_location?arrest=TRUE&location_description=DAY%20CARE%20CENTER&secondary_description=FORCIBLE%20ENTRY&primary_description=BURGLARY",
    "/v1/chicago_crime/top_district_by_domestic_crimes?domestic=TRUE",
    "/v1/chicago_crime/location_descriptions_min_population",
    "/v1/chicago_crime/sum_crimes_by_description_and_arrest?description=%25The%20violation%20of%20laws%25&arrest=FALSE",
    "/v1/chicago_crime/crime_location_by_fbi_code_and_community?title=Simple%20Assault&community_area_name=Chatham&community_area_no=44",
    "/v1/chicago_crime/top_community_area_by_fbi_code_and_side?side=South&description=The%20unlawful%20taking%2C%20carrying%2C%20leading%2C%20or%20riding%20away%20of%20property%20FROM%20the%20possession%20or%20constructive%20possession%20of%20another%20person.",
    "/v1/chicago_crime/sum_crimes_community_area_crime_against?community_area_name=Englewood&crime_against=Society",
    "/v1/chicago_crime/avg_fbi_codes_month_year?month=1&year=2018",
    "/v1/chicago_crime/community_area_names_population?population=50000&limit=3",
    "/v1/chicago_crime/population_diff_sides?side1=South&side2=North&population=300000",
    "/v1/chicago_crime/district_contact_info?district_name=Near%20West",
    "/v1/chicago_crime/iucr_descriptions?iucr_no=275",
    "/v1/chicago_crime/percentage_primary_description?primary_description=CRIM%20SEXUAL%20ASSAULT&index_code=I",
    "/v1/chicago_crime/total_population_of_neighborhoods?neighborhood_name1=Avondale%20Gardens&neighborhood_name2=Irving%20Park&neighborhood_name3=Kilbourn%20Park&neighborhood_name4=Merchant%20Park&neighborhood_name5=Old%20Irving%20Park&neighborhood_name6=The%20Villa",
    "/v1/chicago_crime/crime_count_by_title?title=Weapons%20Violation",
    "/v1/chicago_crime/case_numbers_by_criteria?title=Criminal%20Sexual%20Assault&arrest=TRUE&crime_against=Persons",
    "/v1/chicago_crime/case_numbers_by_criteria_v2?title=Criminal%20Sexual%20Assault&crime_against=Persons&arrest=TRUE",
    "/v1/chicago_crime/primary_description_by_case_number?case_number=JB106010",
    "/v1/chicago_crime/community_area_names_by_crime_title?title=Criminal%20Sexual%20Assault",
    "/v1/chicago_crime/fbi_codes_titles_by_community_area?community_area_name=Rogers%20Park",
    "/v1/chicago_crime/district_info_by_case_number?case_number=JB120039",
    "/v1/chicago_crime/percentage_crimes_by_title_location?title=Drug%20Abuse&location_description=STREET",
    "/v1/chicago_crime/ward_with_highest_population",
    "/v1/chicago_crime/crime_details_by_case_number?case_number=JB112212",
    "/v1/chicago_crime/fbi_code_by_description?description=The%20killing%20of%20one%20human%20being%20by%20another.",
    "/v1/chicago_crime/ward_office_addresses_by_community_area?community_area_name=Montclare",
    "/v1/chicago_crime/district_info_by_location_and_beat?location_description=YARD&beat=532",
    "/v1/chicago_crime/neighborhood_names_by_report_number?report_no=23778",
    "/v1/chicago_crime/fbi_code_descriptions_by_report_number?report_no=23843",
    "/v1/chicago_crime/crime_count_by_commander?commander=Jill%20M.%20Stevens",
    "/v1/chicago_crime/crime_percentage_by_location_and_neighborhood?location_description=HOUSE&neighborhood_name=Avalon%20Park",
    "/v1/chicago_crime/iucr_count_by_primary_description?primary_description=ASSAULT",
    "/v1/chicago_crime/iucr_count_by_index_code?index_code=I",
    "/v1/chicago_crime/crime_count_by_alderman_and_arrest?alderman_last_name=Dowell&arrest=FALSE&alderman_first_name=Pat",
    "/v1/chicago_crime/ward_with_highest_crime_count?month=1&year=2018",
    "/v1/chicago_crime/ward_count_domestic_crimes_by_population?domestic=TRUE",
    "/v1/chicago_crime/crime_locations_by_alderman?alderman_first_name=Pat&alderman_last_name=Dowell",
    "/v1/chicago_crime/alderman_crime_count_by_criteria?month=1&year=2018&alderman_first_name_1=Pat&alderman_last_name_1=Dowell&alderman_first_name_2=Sophia&alderman_last_name_2=King",
    "/v1/chicago_crime/case_numbers_by_primary_description?primary_description=BATTERY",
    "/v1/chicago_crime/crime_count_by_primary_description_and_arrest?arrest=FALSE&primary_description=BATTERY",
    "/v1/chicago_crime/case_numbers_by_fbi_code_title?title=Homicide%201st%20%26%202nd%20Degree",
    "/v1/chicago_crime/sum_crimes_against_property?year=2018&crime_against=Property&month=1",
    "/v1/chicago_crime/sum_crimes_by_commander?year=2018&commander=Robert%20A.%20Rubio&month=1",
    "/v1/chicago_crime/commanders_by_criteria?commander1=Robert%20A.%20Rubio&commander2=Glenn%20White&month=1&year=2018",
    "/v1/chicago_crime/blocks_by_commander?commander=Robert%20A.%20Rubio",
    "/v1/chicago_crime/average_ward_count?year=2018",
    "/v1/chicago_crime/percentage_crimes_by_description_arrest?primary_description=BATTERY&arrest=FALSE"
]
