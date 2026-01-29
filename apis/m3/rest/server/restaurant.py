from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/restaurant/restaurant.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of restaurants with a review score less than a specified value
@app.get("/v1/restaurant/count_restaurants_by_review", operation_id="get_count_restaurants_by_review", summary="Retrieves the number of restaurants that have a review score below the provided threshold. This operation is useful for understanding the distribution of review scores and identifying restaurants with lower ratings.")
async def get_count_restaurants_by_review(review_score: int = Query(..., description="Review score threshold")):
    cursor.execute("SELECT COUNT(id_restaurant) FROM generalinfo WHERE review < ?", (review_score,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the food types of restaurants with the highest review score, limited to a specified number of results
@app.get("/v1/restaurant/top_food_types", operation_id="get_top_food_types", summary="Retrieves the food types of the top-rated restaurants, limited to a specified number of results. The operation identifies the restaurants with the highest review scores and returns their respective food types, up to the defined limit.")
async def get_top_food_types(limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT food_type FROM generalinfo WHERE review = ( SELECT MAX(review) FROM generalinfo ) LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"food_types": []}
    return {"food_types": [row[0] for row in result]}

# Endpoint to get the count of restaurants with a specific food type in a specific city
@app.get("/v1/restaurant/count_restaurants_by_food_type_and_city", operation_id="get_count_restaurants_by_food_type_and_city", summary="Retrieves the total number of restaurants serving a particular cuisine in a given city. The operation requires the food type and city as input parameters to filter the results.")
async def get_count_restaurants_by_food_type_and_city(food_type: str = Query(..., description="Food type of the restaurant"), city: str = Query(..., description="City where the restaurant is located")):
    cursor.execute("SELECT COUNT(id_restaurant) FROM generalinfo WHERE food_type = ? AND city = ?", (food_type, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the cities in a specific county
@app.get("/v1/restaurant/cities_by_county", operation_id="get_cities_by_county", summary="Retrieves a list of cities located within the specified county. The county is identified by its name.")
async def get_cities_by_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT city FROM geographic WHERE county = ?", (county,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the distinct counties in a region other than a specified region
@app.get("/v1/restaurant/counties_by_region", operation_id="get_counties_by_region", summary="Retrieves a list of unique counties that are not part of the specified region. The operation filters the geographic data to exclude the provided region, ensuring that only counties from other regions are returned.")
async def get_counties_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT DISTINCT county FROM geographic WHERE region != ?", (region,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get the cities in a specific region
@app.get("/v1/restaurant/cities_by_region", operation_id="get_cities_by_region", summary="Retrieves a list of cities located within a specified region. The region is identified by its name.")
async def get_cities_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT city FROM geographic WHERE region = ?", (region,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the restaurant IDs in a specific city and street
@app.get("/v1/restaurant/restaurant_ids_by_city_and_street", operation_id="get_restaurant_ids_by_city_and_street", summary="Retrieves the unique identifiers of all restaurants located in a specified city and street. The operation requires the city and street name as input parameters to filter the results.")
async def get_restaurant_ids_by_city_and_street(city: str = Query(..., description="City name"), street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT id_restaurant FROM location WHERE city = ? AND street_name = ?", (city, street_name))
    result = cursor.fetchall()
    if not result:
        return {"restaurant_ids": []}
    return {"restaurant_ids": [row[0] for row in result]}

# Endpoint to get the count of restaurants with a specific street number
@app.get("/v1/restaurant/count_restaurants_by_street_number", operation_id="get_count_restaurants_by_street_number", summary="Retrieves the total number of restaurants located at a specific street number. The operation requires the street number as an input parameter to filter the results.")
async def get_count_restaurants_by_street_number(street_num: int = Query(..., description="Street number")):
    cursor.execute("SELECT COUNT(id_restaurant) FROM location WHERE street_num = ?", (street_num,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the food types of restaurants located on a specific street in a specific city
@app.get("/v1/restaurant/food_types_by_street_and_city", operation_id="get_food_types_by_street_and_city", summary="Retrieves the types of food served by restaurants situated on a given street within a specified city. The operation requires the street and city names as input parameters to filter the results.")
async def get_food_types_by_street_and_city(street_name: str = Query(..., description="Street name"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.food_type FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.street_name = ? AND T2.city = ?", (street_name, city))
    result = cursor.fetchall()
    if not result:
        return {"food_types": []}
    return {"food_types": [row[0] for row in result]}

# Endpoint to get distinct regions based on food type
@app.get("/v1/restaurant/distinct_regions_by_food_type", operation_id="get_distinct_regions_by_food_type", summary="Retrieves a list of unique regions where the food type is not the specified type. The operation filters out the specified food type and returns the distinct regions from the geographic table that are associated with the remaining food types in the generalinfo table.")
async def get_distinct_regions_by_food_type(food_type: str = Query(..., description="Food type to exclude")):
    cursor.execute("SELECT DISTINCT T2.region FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.food_type != ?", (food_type,))
    result = cursor.fetchall()
    if not result:
        return {"regions": []}
    return {"regions": [row[0] for row in result]}

# Endpoint to get distinct counties based on restaurant label
@app.get("/v1/restaurant/distinct_counties_by_label", operation_id="get_distinct_counties_by_label", summary="Retrieves a list of unique counties where restaurants with a specified label are located. The label parameter is used to filter the results.")
async def get_distinct_counties_by_label(label: str = Query(..., description="Restaurant label to match")):
    cursor.execute("SELECT DISTINCT T2.county FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get street name and number based on restaurant label
@app.get("/v1/restaurant/street_info_by_label", operation_id="get_street_info_by_label", summary="Retrieves the street name and number of a restaurant that matches a specific label. The label is used to identify the restaurant and retrieve its location details.")
async def get_street_info_by_label(label: str = Query(..., description="Restaurant label to match")):
    cursor.execute("SELECT T1.street_name, T1.street_num FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"street_info": []}
    return {"street_info": [{"street_name": row[0], "street_num": row[1]} for row in result]}

# Endpoint to get food type based on county, street name, and street number
@app.get("/v1/restaurant/food_type_by_location", operation_id="get_food_type_by_location", summary="Retrieves the food type of restaurants located in a specific county, street name, and street number. This operation filters the restaurants based on the provided geographical details and returns the corresponding food type.")
async def get_food_type_by_location(county: str = Query(..., description="County to match"), street_name: str = Query(..., description="Street name to match"), street_num: int = Query(..., description="Street number to match")):
    cursor.execute("SELECT T2.food_type FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant INNER JOIN geographic AS T3 ON T2.city = T3.city WHERE T3.County = ? AND T1.street_name = ? AND T1.street_num = ?", (county, street_name, street_num))
    result = cursor.fetchall()
    if not result:
        return {"food_types": []}
    return {"food_types": [row[0] for row in result]}

# Endpoint to get street names based on city and food type
@app.get("/v1/restaurant/street_names_by_city_food_type", operation_id="get_street_names_by_city_food_type", summary="Retrieves the names of streets in a specific city where restaurants serving a particular food type are located. The operation filters out streets with no restaurants.")
async def get_street_names_by_city_food_type(city: str = Query(..., description="City to match"), food_type: str = Query(..., description="Food type to match")):
    cursor.execute("SELECT T1.street_name FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.city = ? AND T2.food_type = ? AND street_name IS NOT NULL", (city, food_type))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get distinct counties based on restaurant label exclusion
@app.get("/v1/restaurant/distinct_counties_excluding_label", operation_id="get_distinct_counties_excluding_label", summary="Retrieves a list of unique counties where the restaurant label is not equal to the provided label. This operation filters the general information of restaurants based on a specific label and then identifies the distinct counties from the geographic data that do not match the excluded label.")
async def get_distinct_counties_excluding_label(label: str = Query(..., description="Restaurant label to exclude")):
    cursor.execute("SELECT DISTINCT T2.county FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.label != ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"counties": []}
    return {"counties": [row[0] for row in result]}

# Endpoint to get the count of distinct counties based on street name
@app.get("/v1/restaurant/count_distinct_counties_by_street_name", operation_id="get_count_distinct_counties_by_street_name", summary="Retrieve the number of unique counties that contain a street with a specified name. This operation allows you to determine the geographical distribution of a particular street name across different counties.")
async def get_count_distinct_counties_by_street_name(street_name: str = Query(..., description="Street name to match")):
    cursor.execute("SELECT COUNT(DISTINCT T2.county) FROM location AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.street_name = ?", (street_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get reviews based on street name
@app.get("/v1/restaurant/reviews_by_street_name", operation_id="get_reviews_by_street_name", summary="Retrieves all reviews for restaurants located on a specified street. The operation filters the reviews based on the provided street name.")
async def get_reviews_by_street_name(street_name: str = Query(..., description="Street name to match")):
    cursor.execute("SELECT T1.review FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.street_name = ?", (street_name,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [row[0] for row in result]}

# Endpoint to get restaurant IDs based on county
@app.get("/v1/restaurant/restaurant_ids_by_county", operation_id="get_restaurant_ids_by_county", summary="Retrieves the unique identifiers of restaurants located in a specified county. The operation filters the data based on the provided county name and returns a list of corresponding restaurant IDs.")
async def get_restaurant_ids_by_county(county: str = Query(..., description="County to match")):
    cursor.execute("SELECT T1.id_restaurant FROM location AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T2.county = ?", (county,))
    result = cursor.fetchall()
    if not result:
        return {"restaurant_ids": []}
    return {"restaurant_ids": [row[0] for row in result]}

# Endpoint to get restaurant IDs and labels based on county
@app.get("/v1/restaurant/restaurant_info_by_county", operation_id="get_restaurant_info_by_county", summary="Retrieves the unique identifiers and labels of restaurants located in a specified county. The operation filters the data based on the provided county name and returns the corresponding restaurant details.")
async def get_restaurant_info_by_county(county: str = Query(..., description="County to match")):
    cursor.execute("SELECT T1.id_restaurant, T1.label FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T2.county = ?", (county,))
    result = cursor.fetchall()
    if not result:
        return {"restaurant_info": []}
    return {"restaurant_info": [{"id_restaurant": row[0], "label": row[1]} for row in result]}

# Endpoint to get restaurant labels based on street name, food type, and city
@app.get("/v1/restaurant/labels_by_street_food_city", operation_id="get_labels_by_street_food_city", summary="Retrieves the labels of restaurants located on a specific street, serving a certain type of food, and situated in a particular city. The operation filters the results based on the provided street name, food type, and city.")
async def get_labels_by_street_food_city(street_name: str = Query(..., description="Street name of the restaurant"), food_type: str = Query(..., description="Food type of the restaurant"), city: str = Query(..., description="City of the restaurant")):
    cursor.execute("SELECT T1.label FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.street_name = ? AND T1.food_type != ? AND T2.city = ?", (street_name, food_type, city))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get street names based on city and review score
@app.get("/v1/restaurant/street_names_by_city_review", operation_id="get_street_names_by_city_review", summary="Retrieves a list of street names for restaurants located in a specified city and having a particular review score. The city and review score are provided as input parameters.")
async def get_street_names_by_city_review(city: str = Query(..., description="City of the restaurant"), review: float = Query(..., description="Review score of the restaurant")):
    cursor.execute("SELECT T2.street_name FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.city = ? AND T1.review = ?", (city, review))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get the top restaurant label based on street name and city
@app.get("/v1/restaurant/top_label_by_street_city", operation_id="get_top_label_by_street_city", summary="Retrieves the most popular restaurant label in a specific city based on the highest-rated restaurant located on a given street. The label represents the type of cuisine or specialty of the restaurant. The city and street name are required as input parameters to accurately identify the restaurant and its corresponding label.")
async def get_top_label_by_street_city(street_name: str = Query(..., description="Street name of the restaurant"), city: str = Query(..., description="City of the restaurant")):
    cursor.execute("SELECT T2.label FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.street_name = ? AND T2.city = ? ORDER BY review LIMIT 1", (street_name, city))
    result = cursor.fetchone()
    if not result:
        return {"label": []}
    return {"label": result[0]}

# Endpoint to get street names based on restaurant label and county
@app.get("/v1/restaurant/street_names_by_label_county", operation_id="get_street_names_by_label_county", summary="Retrieves a list of street names where restaurants with the specified label are located within the given county. The label and county parameters are used to filter the results.")
async def get_street_names_by_label_county(label: str = Query(..., description="Label of the restaurant"), county: str = Query(..., description="County of the restaurant")):
    cursor.execute("SELECT T1.street_name FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant INNER JOIN geographic AS T3 ON T2.city = T3.city WHERE T2.label = ? AND T3.county = ?", (label, county))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get distinct street numbers based on restaurant label
@app.get("/v1/restaurant/distinct_street_nums_by_label", operation_id="get_distinct_street_nums_by_label", summary="Retrieves a unique set of street numbers associated with restaurants that match the provided label. This operation is useful for identifying the distinct street numbers where restaurants with a specific label are located.")
async def get_distinct_street_nums_by_label(label: str = Query(..., description="Label of the restaurant")):
    cursor.execute("SELECT DISTINCT T1.street_num FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"street_nums": []}
    return {"street_nums": [row[0] for row in result]}

# Endpoint to get restaurant IDs based on county
@app.get("/v1/restaurant/ids_by_county", operation_id="get_ids_by_county", summary="Retrieves the unique identifiers of restaurants located in the specified county. The operation filters the list of restaurants based on the provided county and returns their corresponding identifiers.")
async def get_ids_by_county(county: str = Query(..., description="County of the restaurant")):
    cursor.execute("SELECT T1.id_restaurant FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T2.county = ?", (county,))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get the average review score based on county
@app.get("/v1/restaurant/avg_review_by_county", operation_id="get_avg_review_by_county", summary="Retrieves the average review score for restaurants located in a specified county. The calculation is based on the review scores of all restaurants in the given county.")
async def get_avg_review_by_county(county: str = Query(..., description="County of the restaurant")):
    cursor.execute("SELECT AVG(T2.review) FROM geographic AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T1.county = ?", (county,))
    result = cursor.fetchone()
    if not result:
        return {"avg_review": []}
    return {"avg_review": result[0]}

# Endpoint to get the percentage of restaurants of a specific food type in a county
@app.get("/v1/restaurant/percentage_food_type_by_county", operation_id="get_percentage_food_type_by_county", summary="Retrieves the percentage of restaurants serving a specific cuisine within a given county. The operation calculates this percentage by comparing the count of restaurants of the specified food type to the total number of restaurants in the county.")
async def get_percentage_food_type_by_county(food_type: str = Query(..., description="Food type of the restaurant"), county: str = Query(..., description="County of the restaurant")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.food_type = ?, 1, 0)) AS REAL) * 100 / COUNT(T2.id_restaurant) FROM geographic AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T1.county = ?", (food_type, county))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of restaurants on a specific street in a county
@app.get("/v1/restaurant/percentage_street_by_county", operation_id="get_percentage_street_by_county", summary="Retrieves the percentage of restaurants located on a specific street within a given county. The operation calculates this percentage by comparing the number of restaurants on the specified street to the total number of restaurants in the county.")
async def get_percentage_street_by_county(street_name: str = Query(..., description="Street name of the restaurant"), county: str = Query(..., description="County of the restaurant")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.street_name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.id_restaurant) FROM location AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T2.County = ?", (street_name, county))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get restaurant labels based on food type
@app.get("/v1/restaurant/labels_by_food_type", operation_id="get_labels_by_food_type", summary="Retrieves the labels of restaurants that serve the specified food type. The operation filters the general information based on the provided food type and returns the corresponding labels.")
async def get_labels_by_food_type(food_type: str = Query(..., description="Type of food")):
    cursor.execute("SELECT label FROM generalinfo WHERE food_type = ?", (food_type,))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get cities based on county and region
@app.get("/v1/restaurant/cities_by_county_region", operation_id="get_cities_by_county_region", summary="Retrieves up to five cities located within the specified county and region. The operation requires the county and region names as input to filter the results.")
async def get_cities_by_county_region(county: str = Query(..., description="County name"), region: str = Query(..., description="Region name")):
    cursor.execute("SELECT city FROM geographic WHERE county = ? AND region = ? LIMIT 5", (county, region))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get county and region based on city
@app.get("/v1/restaurant/county_region_by_city", operation_id="get_county_region_by_city", summary="Retrieves the county and region associated with the specified city. The operation uses the provided city name to look up the corresponding geographic information.")
async def get_county_region_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT county, region FROM geographic WHERE city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"county_region": []}
    return {"county_region": [{"county": row[0], "region": row[1]} for row in result]}

# Endpoint to get street names based on city
@app.get("/v1/restaurant/street_names_by_city", operation_id="get_street_names_by_city", summary="Retrieves a list of street names within the specified city. The operation filters the available location data based on the provided city name and returns the corresponding street names.")
async def get_street_names_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT street_name FROM location WHERE city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get restaurant ID and label with the highest review in a specified city
@app.get("/v1/restaurant/top_reviewed_restaurant_by_city", operation_id="get_top_reviewed_restaurant_by_city", summary="Retrieves the restaurant with the highest review score in the specified city. The operation returns the unique identifier and label of the restaurant.")
async def get_top_reviewed_restaurant_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT id_restaurant, label FROM generalinfo WHERE city = ? AND review = ( SELECT MAX(review) FROM generalinfo WHERE city = ? )", (city, city))
    result = cursor.fetchall()
    if not result:
        return {"top_reviewed_restaurant": []}
    return {"top_reviewed_restaurant": [{"id_restaurant": row[0], "label": row[1]} for row in result]}

# Endpoint to get the count of restaurants with the lowest review in a specified city and food type
@app.get("/v1/restaurant/count_lowest_reviewed_restaurants", operation_id="get_count_lowest_reviewed_restaurants", summary="Retrieves the number of restaurants serving a specific cuisine in a given city that have the lowest review rating. This operation helps identify the least favorably reviewed establishments in a particular food category within a certain location.")
async def get_count_lowest_reviewed_restaurants(food_type: str = Query(..., description="Type of food"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(id_restaurant) FROM generalinfo WHERE food_type = ? AND city = ? AND review = ( SELECT MIN(review) FROM generalinfo WHERE food_type = ? AND city = ? )", (food_type, city, food_type, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of restaurants of a specific food type in a specified city
@app.get("/v1/restaurant/percentage_food_type_by_city", operation_id="get_percentage_food_type_by_city", summary="Retrieves the percentage of restaurants serving a specific food type in the given city. The operation calculates this percentage by comparing the count of restaurants of the specified food type to the total number of restaurants in the city.")
async def get_percentage_food_type_by_city(food_type: str = Query(..., description="Type of food"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT CAST(SUM(IIF(food_type = ?, 1, 0)) AS REAL) * 100 / COUNT(id_restaurant) FROM generalinfo WHERE city = ?", (food_type, city))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get street number, street name, and city based on restaurant label
@app.get("/v1/restaurant/location_by_label", operation_id="get_location_by_label", summary="Retrieves the street number, street name, and city associated with a specific restaurant, identified by its unique label. This operation returns the precise location details of the restaurant, enabling users to pinpoint its exact position.")
async def get_location_by_label(label: str = Query(..., description="Restaurant label")):
    cursor.execute("SELECT T2.street_num, T2.street_name, T1.city FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"location": []}
    return {"location": [{"street_num": row[0], "street_name": row[1], "city": row[2]} for row in result]}

# Endpoint to get restaurant IDs based on city and street name
@app.get("/v1/restaurant/restaurant_ids_by_city_street", operation_id="get_restaurant_ids_by_city_street", summary="Retrieves the unique identifiers of restaurants located in the specified city and street. The operation filters the available restaurants based on the provided city and street name, ensuring an accurate and relevant result set.")
async def get_restaurant_ids_by_city_street(city: str = Query(..., description="City name"), street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT T1.id_restaurant FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.city = ? AND T2.street_name = ?", (city, street_name))
    result = cursor.fetchall()
    if not result:
        return {"restaurant_ids": []}
    return {"restaurant_ids": [row[0] for row in result]}

# Endpoint to get food types based on street number and street name
@app.get("/v1/restaurant/food_type_by_street", operation_id="get_food_type_by_street", summary="Retrieves the types of food served by restaurants located on a specific street. The street is identified by its number and name, which are provided as input parameters. The operation returns a list of food types available at these restaurants.")
async def get_food_type_by_street(street_num: int = Query(..., description="Street number"), street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT T1.food_type FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.street_num = ? AND T2.street_name = ?", (street_num, street_name))
    result = cursor.fetchall()
    if not result:
        return {"food_types": []}
    return {"food_types": [row[0] for row in result]}

# Endpoint to get restaurant labels based on region
@app.get("/v1/restaurant/labels_by_region", operation_id="get_labels_by_region", summary="Retrieves up to three distinct labels of restaurants located in a specified region. The labels provide information about the restaurants' characteristics or features.")
async def get_labels_by_region(region: str = Query(..., description="Region")):
    cursor.execute("SELECT T2.label FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant INNER JOIN geographic AS T3 ON T2.city = T3.city WHERE T3.region = ? LIMIT 3", (region,))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get restaurant labels based on street number, city, and street name
@app.get("/v1/restaurant/labels_by_street_city", operation_id="get_labels_by_street_city", summary="Retrieves the labels of restaurants located at a specific street number, within a given city, and along a particular street. The operation filters the restaurants based on the provided street number, city, and street name, and returns their corresponding labels.")
async def get_labels_by_street_city(street_num: int = Query(..., description="Street number"), city: str = Query(..., description="City"), street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT T1.label FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.street_num = ? AND T1.city = ? AND T2.street_name = ?", (street_num, city, street_name))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get the count of restaurants based on food type, city, and street name
@app.get("/v1/restaurant/count_by_food_type_city_street", operation_id="get_count_by_food_type_city_street", summary="Retrieves the total number of restaurants that serve a specific cuisine, located in a given city and along a particular street. This operation is useful for understanding the distribution of restaurants based on food type, city, and street.")
async def get_count_by_food_type_city_street(food_type: str = Query(..., description="Food type"), city: str = Query(..., description="City"), street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = ? AND T1.city = ? AND T2.street_name = ?", (food_type, city, street_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get geographic information based on restaurant label
@app.get("/v1/restaurant/geographic_info_by_label", operation_id="get_geographic_info_by_label", summary="Retrieves geographic information, including county and region, for a specific restaurant identified by its label. The data is obtained by matching the restaurant's city with the corresponding city in the geographic data.")
async def get_geographic_info_by_label(label: str = Query(..., description="Restaurant label")):
    cursor.execute("SELECT T1.county, T1.region, T2.label FROM geographic AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T2.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"geographic_info": []}
    return {"geographic_info": [{"county": row[0], "region": row[1], "label": row[2]} for row in result]}

# Endpoint to get restaurant labels based on region and county
@app.get("/v1/restaurant/labels_by_region_county", operation_id="get_labels_by_region_county", summary="Retrieves the labels of restaurants located in a specific region and county. The operation filters the data based on the provided region and county parameters, returning the corresponding labels.")
async def get_labels_by_region_county(region: str = Query(..., description="Region"), county: str = Query(..., description="County")):
    cursor.execute("SELECT T2.label FROM geographic AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T1.region = ? AND T1.county = ?", (region, county))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get distinct counties and regions based on street name
@app.get("/v1/restaurant/distinct_counties_regions_by_street", operation_id="get_distinct_counties_regions_by_street", summary="Retrieves unique county and region combinations associated with a specific street name. This operation returns a list of distinct counties and their corresponding regions, providing geographic context for the given street name.")
async def get_distinct_counties_regions_by_street(street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT DISTINCT T2.county, T2.region FROM location AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.street_name = ?", (street_name,))
    result = cursor.fetchall()
    if not result:
        return {"counties_regions": []}
    return {"counties_regions": [{"county": row[0], "region": row[1]} for row in result]}

# Endpoint to get the top-rated restaurant based on city, street name, and food type
@app.get("/v1/restaurant/top_rated_by_city_street_food_type", operation_id="get_top_rated_by_city_street_food_type", summary="Retrieves the top-rated restaurant in a specific city, street, and food type category. The operation filters restaurants based on the provided city, street name, and food type, then ranks them by review score to return the highest-rated establishment.")
async def get_top_rated_by_city_street_food_type(city: str = Query(..., description="City"), street_name: str = Query(..., description="Street name"), food_type: str = Query(..., description="Food type")):
    cursor.execute("SELECT T1.id_restaurant FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.city = ? AND T2.street_name = ? AND T1.food_type = ? ORDER BY T1.review LIMIT 1", (city, street_name, food_type))
    result = cursor.fetchone()
    if not result:
        return {"id_restaurant": []}
    return {"id_restaurant": result[0]}

# Endpoint to get the percentage of highly rated restaurants in a specific region
@app.get("/v1/restaurant/highly_rated_percentage_by_region", operation_id="get_highly_rated_percentage_by_region", summary="Retrieves the percentage of restaurants in a specified region that have a rating greater than 4. The calculation is based on the total number of restaurants in the region.")
async def get_highly_rated_percentage_by_region(region: str = Query(..., description="Region")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.review > 4, 1, 0)) AS REAL) * 100 / COUNT(T2.id_restaurant) FROM geographic AS T1 RIGHT JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T1.region = ?", (region,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the most common street name from the location table
@app.get("/v1/restaurant/most_common_street_name", operation_id="get_most_common_street_name", summary="Retrieves the street name that appears most frequently in the location data. This operation returns the street name with the highest occurrence count, providing insights into the most common street name in the dataset.")
async def get_most_common_street_name():
    cursor.execute("SELECT street_name FROM location GROUP BY street_name ORDER BY COUNT(street_name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"street_name": []}
    return {"street_name": result[0]}

# Endpoint to get the top-rated restaurant label for a specific food type
@app.get("/v1/restaurant/top_rated_restaurant_by_food_type", operation_id="get_top_rated_restaurant_by_food_type", summary="Retrieves the label of the highest-rated restaurant that serves a particular food type, based on review scores.")
async def get_top_rated_restaurant_by_food_type(food_type: str = Query(..., description="Type of food")):
    cursor.execute("SELECT label FROM generalinfo WHERE food_type = ? ORDER BY review DESC LIMIT 1", (food_type,))
    result = cursor.fetchone()
    if not result:
        return {"label": []}
    return {"label": result[0]}

# Endpoint to get the county for a specific city
@app.get("/v1/restaurant/county_by_city", operation_id="get_county_by_city", summary="Retrieves the county associated with a given city. The operation requires the city name as input and returns the corresponding county from the geographic data.")
async def get_county_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT county FROM geographic WHERE city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get the count of restaurants on a specific street
@app.get("/v1/restaurant/count_restaurants_by_street", operation_id="get_count_restaurants_by_street", summary="Retrieves the total number of restaurants located on a specified street. The street is identified by its name, which is provided as an input parameter.")
async def get_count_restaurants_by_street(street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT COUNT(id_restaurant) FROM location WHERE street_name = ?", (street_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get restaurant labels in a specific county
@app.get("/v1/restaurant/labels_by_county", operation_id="get_labels_by_county", summary="Retrieves the labels of all restaurants located in a specified county. The county is identified by its name.")
async def get_labels_by_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT T1.label FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T2.county = ?", (county,))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get street names for a specific restaurant label
@app.get("/v1/restaurant/street_names_by_label", operation_id="get_street_names_by_label", summary="Retrieves the street names of restaurants that match a given label. The label is used to filter the restaurants, and the street names of the matching restaurants are returned.")
async def get_street_names_by_label(label: str = Query(..., description="Restaurant label")):
    cursor.execute("SELECT T2.street_name FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get street names in a specific county with more than a specified number of restaurants
@app.get("/v1/restaurant/street_names_by_county_and_restaurant_count", operation_id="get_street_names_by_county_and_restaurant_count", summary="Retrieves the names of streets in a given county that have more than a specified number of restaurants. The operation filters streets based on the provided county name and the minimum number of restaurants required. The response includes a list of street names that meet the specified criteria.")
async def get_street_names_by_county_and_restaurant_count(county: str = Query(..., description="County name"), restaurant_count: int = Query(..., description="Minimum number of restaurants")):
    cursor.execute("SELECT T2.street_name FROM geographic AS T1 INNER JOIN location AS T2 ON T1.city = T2.city WHERE T1.county = ? GROUP BY T2.street_name HAVING COUNT(T2.id_restaurant) > ?", (county, restaurant_count))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get distinct regions for a specific food type
@app.get("/v1/restaurant/regions_by_food_type", operation_id="get_regions_by_food_type", summary="Retrieves a list of unique regions where a specific type of food is available. The operation filters the data based on the provided food type and returns the distinct regions where that food is served.")
async def get_regions_by_food_type(food_type: str = Query(..., description="Type of food")):
    cursor.execute("SELECT DISTINCT T1.region FROM geographic AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T2.food_type = ?", (food_type,))
    result = cursor.fetchall()
    if not result:
        return {"regions": []}
    return {"regions": [row[0] for row in result]}

# Endpoint to get street names for a specific region
@app.get("/v1/restaurant/street_names_by_region", operation_id="get_street_names_by_region", summary="Retrieves a list of street names for a specified region. The operation uses the provided region name to filter the geographic and location data, returning only the street names that match the given region.")
async def get_street_names_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT T2.street_name FROM geographic AS T1 INNER JOIN location AS T2 ON T1.city = T2.city WHERE T1.region = ?", (region,))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get reviews for a restaurant at a specific street name and number
@app.get("/v1/restaurant/reviews_by_street_name_and_number", operation_id="get_reviews_by_street_name_and_number", summary="Retrieves reviews for a restaurant located at the specified street name and number. The operation fetches review data from the database by matching the provided street name and number with the corresponding restaurant's location details.")
async def get_reviews_by_street_name_and_number(street_name: str = Query(..., description="Street name of the restaurant"), street_num: int = Query(..., description="Street number of the restaurant")):
    cursor.execute("SELECT T2.review FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.street_name = ? AND T1.street_num = ?", (street_name, street_num))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [row[0] for row in result]}

# Endpoint to get the most common food type in a specific county
@app.get("/v1/restaurant/most_common_food_type_by_county", operation_id="get_most_common_food_type_by_county", summary="Retrieves the most frequently occurring food type in a specified county. The operation identifies the food type that appears most frequently in the general information records associated with the specified county. The county is determined by matching the provided county name with the geographic records.")
async def get_most_common_food_type_by_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT T2.food_type FROM geographic AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T1.county = ? GROUP BY T2.food_type ORDER BY COUNT(T2.food_type) DESC LIMIT 1", (county,))
    result = cursor.fetchone()
    if not result:
        return {"food_type": []}
    return {"food_type": result[0]}

# Endpoint to get the most common street name for a specific food type in a specific city
@app.get("/v1/restaurant/most_common_street_name_by_city_and_food_type", operation_id="get_most_common_street_name_by_city_and_food_type", summary="Retrieves the street name that appears most frequently for a given food type in a specific city. The operation considers the restaurants' location data and filters them by the provided city and food type. The result is the street name that hosts the highest number of restaurants serving the specified food type in the selected city.")
async def get_most_common_street_name_by_city_and_food_type(city: str = Query(..., description="City name"), food_type: str = Query(..., description="Food type")):
    cursor.execute("SELECT T2.street_name FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.city = ? AND T1.food_type = ? GROUP BY T2.street_name ORDER BY COUNT(T2.id_restaurant) DESC LIMIT 1", (city, food_type))
    result = cursor.fetchone()
    if not result:
        return {"street_name": []}
    return {"street_name": result[0]}

# Endpoint to get the region of a restaurant at a specific street number and name
@app.get("/v1/restaurant/region_by_street_number_and_name", operation_id="get_region_by_street_number_and_name", summary="Retrieves the geographic region of a restaurant based on its street number and name. This operation requires the street number and name as input parameters to accurately locate the restaurant and determine its corresponding region.")
async def get_region_by_street_number_and_name(street_num: int = Query(..., description="Street number of the restaurant"), street_name: str = Query(..., description="Street name of the restaurant")):
    cursor.execute("SELECT T2.region FROM location AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.street_num = ? AND T1.street_name = ?", (street_num, street_name))
    result = cursor.fetchone()
    if not result:
        return {"region": []}
    return {"region": result[0]}

# Endpoint to get the county of a restaurant by its label
@app.get("/v1/restaurant/county_by_label", operation_id="get_county_by_label", summary="Get the county of a restaurant by its label")
async def get_county_by_label(label: str = Query(..., description="Label of the restaurant")):
    cursor.execute("SELECT T2.county FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.label = ?", (label,))
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get the count of cities in a specific region
@app.get("/v1/restaurant/count_cities_by_region", operation_id="get_count_cities_by_region", summary="Retrieves the total number of cities in a specified region. The operation calculates this count by joining the geographic and location tables on the city field and filtering the results based on the provided region name.")
async def get_count_cities_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT COUNT(T1.city) FROM geographic AS T1 INNER JOIN location AS T2 ON T1.city = T2.city WHERE T1.region = ?", (region,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the labels of restaurants on a specific street
@app.get("/v1/restaurant/labels_by_street_name", operation_id="get_labels_by_street_name", summary="Retrieves the labels of all restaurants located on a specified street. The street is identified by its name, which is provided as an input parameter.")
async def get_labels_by_street_name(street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT T2.label FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.street_name = ?", (street_name,))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get the percentage of restaurants in a specific region
@app.get("/v1/restaurant/percentage_restaurants_by_region", operation_id="get_percentage_restaurants_by_region", summary="Retrieves the percentage of restaurants located in a specified region. This operation calculates the proportion of restaurants in the given region by comparing the count of restaurants in that region to the total number of restaurants. The region is identified by its name.")
async def get_percentage_restaurants_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.region = ?, 1, 0)) AS REAL) * 100 / COUNT(T2.id_restaurant) FROM geographic AS T1 INNER JOIN location AS T2 ON T1.city = T2.city", (region,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average review of restaurants by food type
@app.get("/v1/restaurant/average_review_by_food_type", operation_id="get_average_review_by_food_type", summary="Retrieves the average review score for restaurants serving a specific food type. The operation calculates the average review score for each restaurant based on the provided food type, and returns the results in descending order. The food type is specified as an input parameter.")
async def get_average_review_by_food_type(food_type: str = Query(..., description="Food type")):
    cursor.execute("SELECT AVG(T1.review) FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.food_type = ? GROUP BY T1.id_restaurant ORDER BY AVG(T1.review) DESC", (food_type,))
    result = cursor.fetchall()
    if not result:
        return {"average_reviews": []}
    return {"average_reviews": [row[0] for row in result]}

# Endpoint to get restaurant IDs based on city
@app.get("/v1/restaurant/get_restaurant_ids_by_city", operation_id="get_restaurant_ids_by_city", summary="Retrieves a list of unique restaurant identifiers located in the specified city. The city name is used to filter the results.")
async def get_restaurant_ids_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT id_restaurant FROM location WHERE city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"restaurant_ids": []}
    return {"restaurant_ids": [row[0] for row in result]}

# Endpoint to get the count of restaurant labels in a specific county
@app.get("/v1/restaurant/count_labels_by_county", operation_id="count_labels_by_county", summary="Retrieves the total number of restaurant labels in a specified county. The operation uses the provided county name to filter the data and calculate the count.")
async def count_labels_by_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT COUNT(T1.label) FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T2.county = ?", (county,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get street names based on food type
@app.get("/v1/restaurant/get_street_names_by_food_type", operation_id="get_street_names_by_food_type", summary="Retrieves a list of street names in a city where a specific type of food is served. The food type is provided as an input parameter.")
async def get_street_names_by_food_type(food_type: str = Query(..., description="Food type")):
    cursor.execute("SELECT T1.street_name FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T2.food_type = ?", (food_type,))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get restaurant reviews based on street name and number
@app.get("/v1/restaurant/get_reviews_by_street", operation_id="get_reviews_by_street", summary="Retrieves reviews for restaurants located on a specific street and number. The operation filters the reviews based on the provided street name and number, returning only those that match the given criteria.")
async def get_reviews_by_street(street_name: str = Query(..., description="Street name"), street_num: int = Query(..., description="Street number")):
    cursor.execute("SELECT T1.review FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.street_name = ? AND T2.street_num = ?", (street_name, street_num))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [row[0] for row in result]}

# Endpoint to get the top-rated restaurant location
@app.get("/v1/restaurant/get_top_rated_location", operation_id="get_top_rated_location", summary="Retrieves the location of the top-rated restaurant, as determined by customer reviews. The location is specified by street number and name.")
async def get_top_rated_location():
    cursor.execute("SELECT T2.street_num, T2.street_name FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant ORDER BY T1.review DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": {"street_num": result[0], "street_name": result[1]}}

# Endpoint to get the top county for a specific food type
@app.get("/v1/restaurant/get_top_county_by_food_type", operation_id="get_top_county_by_food_type", summary="Retrieves the county with the highest number of restaurants serving a specific food type. The food type is provided as an input parameter.")
async def get_top_county_by_food_type(food_type: str = Query(..., description="Food type")):
    cursor.execute("SELECT T2.county FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.food_type = ? GROUP BY T2.county ORDER BY COUNT(T1.id_restaurant) DESC LIMIT 1", (food_type,))
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get the percentage of restaurants in a specific region
@app.get("/v1/restaurant/get_region_percentage", operation_id="get_region_percentage", summary="Retrieves the percentage of restaurants located in a specified region. The calculation is based on the total count of restaurants in the region compared to the overall count of restaurants across all regions.")
async def get_region_percentage(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT CAST(SUM(IIF(region = ?, 1, 0)) AS REAL) * 100 / COUNT(region) FROM geographic", (region,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get cities based on region and county
@app.get("/v1/restaurant/cities_by_region_county", operation_id="get_cities_by_region_county", summary="Retrieves a list of cities located within a specified region and county. The operation requires the region and county names as input parameters to filter the results.")
async def get_cities_by_region_county(region: str = Query(..., description="Region name"), county: str = Query(..., description="County name")):
    cursor.execute("SELECT city FROM geographic WHERE region = ? AND county = ?", (region, county))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of cities based on street name and street number
@app.get("/v1/restaurant/count_cities_by_street", operation_id="get_count_cities_by_street", summary="Retrieves the number of unique cities that have a street with the provided name and a street number less than the specified value.")
async def get_count_cities_by_street(street_name: str = Query(..., description="Street name"), street_num: int = Query(..., description="Street number")):
    cursor.execute("SELECT COUNT(city) FROM location WHERE street_name = ? AND street_num < ?", (street_name, street_num))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get restaurant labels based on food type and city
@app.get("/v1/restaurant/labels_by_food_type_city", operation_id="get_labels_by_food_type_city", summary="Retrieves the labels of restaurants that serve a particular food type in a specified city. The operation requires the food type and city name as input parameters to filter the results.")
async def get_labels_by_food_type_city(food_type: str = Query(..., description="Food type"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT label FROM generalinfo WHERE food_type = ? AND city = ?", (food_type, city))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get reviews based on city and street name
@app.get("/v1/restaurant/reviews_by_city_street", operation_id="get_reviews_by_city_street", summary="Retrieves restaurant reviews for a specific city and street. The operation filters reviews based on the provided city and street name, allowing users to access relevant reviews for a particular location.")
async def get_reviews_by_city_street(city: str = Query(..., description="City name"), street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT T2.review FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.city = ? AND T1.street_name = ?", (city, street_name))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [row[0] for row in result]}

# Endpoint to get street numbers based on review, city, and food type
@app.get("/v1/restaurant/street_numbers_by_review_city_food_type", operation_id="get_street_numbers_by_review_city_food_type", summary="Retrieves a list of street numbers for restaurants that match a specific review score, city, and food type. The operation filters restaurants based on the provided review score, city name, and food type, and returns the corresponding street numbers.")
async def get_street_numbers_by_review_city_food_type(review: float = Query(..., description="Review score"), city: str = Query(..., description="City name"), food_type: str = Query(..., description="Food type")):
    cursor.execute("SELECT T2.street_num FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.review = ? AND T2.city = ? AND T1.food_type = ?", (review, city, food_type))
    result = cursor.fetchall()
    if not result:
        return {"street_numbers": []}
    return {"street_numbers": [row[0] for row in result]}

# Endpoint to get the count of restaurants based on food type, city, and street name
@app.get("/v1/restaurant/count_restaurants_by_food_type_city_street", operation_id="get_count_restaurants_by_food_type_city_street", summary="Retrieves the total number of restaurants that serve a specific cuisine, located in a particular city and street. The operation filters the restaurants based on the provided food type, city, and street name.")
async def get_count_restaurants_by_food_type_city_street(food_type: str = Query(..., description="Food type"), city: str = Query(..., description="City name"), street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT COUNT(T1.id_restaurant) FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.food_type = ? AND T2.city = ? AND T1.street_name = ?", (food_type, city, street_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get street numbers based on label and food type
@app.get("/v1/restaurant/street_numbers_by_label_food_type", operation_id="get_street_numbers_by_label_food_type", summary="Retrieves the street numbers of restaurants that match a given label and food type. The label and food type are used to filter the results, providing a targeted list of street numbers for restaurants that meet the specified criteria.")
async def get_street_numbers_by_label_food_type(label: str = Query(..., description="Label of the restaurant"), food_type: str = Query(..., description="Food type")):
    cursor.execute("SELECT T1.street_num FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.label = ? AND T2.food_type = ?", (label, food_type))
    result = cursor.fetchall()
    if not result:
        return {"street_numbers": []}
    return {"street_numbers": [row[0] for row in result]}

# Endpoint to get reviews and labels based on city and street name
@app.get("/v1/restaurant/reviews_labels_by_city_street", operation_id="get_reviews_labels_by_city_street", summary="Retrieves reviews and their associated labels for restaurants located in a specific city and street. The operation filters results based on the provided city and street name, allowing users to access relevant reviews and labels for a particular area.")
async def get_reviews_labels_by_city_street(city: str = Query(..., description="City name"), street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT T2.review, T2.label FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.city = ? AND T1.street_name = ?", (city, street_name))
    result = cursor.fetchall()
    if not result:
        return {"reviews_labels": []}
    return {"reviews_labels": [{"review": row[0], "label": row[1]} for row in result]}

# Endpoint to get the count of restaurants based on street name, city, food type, and label pattern
@app.get("/v1/restaurant/count_restaurants_by_street_city_food_type_label", operation_id="get_count_restaurants_by_street_city_food_type_label", summary="Retrieve the number of restaurants that match a specific street name, city, food type, and label pattern. The label pattern supports wildcard searches using the '%' symbol. This operation provides a count of restaurants that meet all the specified criteria.")
async def get_count_restaurants_by_street_city_food_type_label(street_name: str = Query(..., description="Street name"), city: str = Query(..., description="City name"), food_type: str = Query(..., description="Food type"), label_pattern: str = Query(..., description="Label pattern (use % for wildcard)")):
    cursor.execute("SELECT COUNT(T1.id_restaurant) FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.street_name = ? AND T1.city = ? AND T2.food_type = ? AND T2.label LIKE ?", (street_name, city, food_type, label_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of restaurants based on review and street number
@app.get("/v1/restaurant/count_restaurants_by_review_street_num", operation_id="get_count_restaurants_by_review_street_num", summary="Retrieve the number of restaurants that have a specified review score and a street number below a given threshold. This operation provides a quantitative measure of restaurants meeting the specified criteria, aiding in the evaluation of restaurant quality and location.")
async def get_count_restaurants_by_review_street_num(review: int = Query(..., description="Review score of the restaurant"), street_num: int = Query(..., description="Street number less than this value")):
    cursor.execute("SELECT COUNT(T1.id_restaurant) FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.review = ? AND T1.street_num < ?", (review, street_num))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of restaurants based on city, food type, street name, and restaurant ID
@app.get("/v1/restaurant/count_restaurants_by_city_food_type_street_name_id", operation_id="get_count_restaurants_by_city_food_type_street_name_id", summary="Retrieves the number of restaurants in a specified city that serve a particular food type, are located on a certain street, and have an ID greater than a given value.")
async def get_count_restaurants_by_city_food_type_street_name_id(city: str = Query(..., description="City of the restaurant"), food_type: str = Query(..., description="Food type of the restaurant"), street_name: str = Query(..., description="Street name of the restaurant"), id_restaurant: int = Query(..., description="Restaurant ID greater than this value")):
    cursor.execute("SELECT COUNT(T1.id_restaurant) AS num FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.city = ? AND T2.food_type = ? AND T1.street_name = ? AND T1.id_restaurant > ?", (city, food_type, street_name, id_restaurant))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get restaurant labels and IDs based on city and street name
@app.get("/v1/restaurant/get_labels_ids_by_city_street_name", operation_id="get_labels_ids_by_city_street_name", summary="Retrieves the labels and unique identifiers of restaurants located in a specified city and street. The operation filters the results based on the provided city and street name.")
async def get_labels_ids_by_city_street_name(city: str = Query(..., description="City of the restaurant"), street_name: str = Query(..., description="Street name of the restaurant")):
    cursor.execute("SELECT T2.label, T1.id_restaurant FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.city = ? AND T1.street_name = ?", (city, street_name))
    result = cursor.fetchall()
    if not result:
        return {"labels_ids": []}
    return {"labels_ids": result}

# Endpoint to get restaurant city, street number, and street name based on label
@app.get("/v1/restaurant/get_city_street_by_label", operation_id="get_city_street_by_label", summary="Retrieve the city, street number, and street name of restaurants that match a given label. The label is used to filter the results and return only the relevant location information.")
async def get_city_street_by_label(label: str = Query(..., description="Label of the restaurant")):
    cursor.execute("SELECT T2.city, T1.street_num, T1.street_name FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"city_street": []}
    return {"city_street": result}

# Endpoint to get food type based on street number, street name, and city
@app.get("/v1/restaurant/get_food_type_by_street_city", operation_id="get_food_type_by_street_city", summary="Retrieve the food type of restaurants located at a specific street address within a given city. This operation requires the street number, street name, and city as input parameters to accurately identify the restaurants and their respective food types.")
async def get_food_type_by_street_city(street_num: int = Query(..., description="Street number of the restaurant"), street_name: str = Query(..., description="Street name of the restaurant"), city: str = Query(..., description="City of the restaurant")):
    cursor.execute("SELECT T2.food_type FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.street_num = ? AND T1.street_name = ? AND T2.city = ?", (street_num, street_name, city))
    result = cursor.fetchall()
    if not result:
        return {"food_type": []}
    return {"food_type": result}

# Endpoint to get the count of a specific food type based on city and street name
@app.get("/v1/restaurant/count_food_type_by_city_street", operation_id="get_count_food_type_by_city_street", summary="Retrieve the number of restaurants serving a particular food type in a given city and street. This operation allows you to specify the food type, city, and street name to obtain an accurate count of restaurants that meet the provided criteria.")
async def get_count_food_type_by_city_street(food_type: str = Query(..., description="Food type of the restaurant"), city: str = Query(..., description="City of the restaurant"), street_name: str = Query(..., description="Street name of the restaurant")):
    cursor.execute("SELECT COUNT(T2.food_type = ?) FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.city = ? AND T1.street_name = ?", (food_type, city, street_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get restaurant IDs based on food type and street number, ordered by average review
@app.get("/v1/restaurant/get_restaurant_ids_by_food_type_street_num", operation_id="get_restaurant_ids_by_food_type_street_num", summary="Retrieves a list of restaurant IDs that serve a specified food type and are located on a street number greater than a given value. The results are ordered by the average review score, with a weight of 0.7, to prioritize restaurants with higher ratings.")
async def get_restaurant_ids_by_food_type_street_num(food_type: str = Query(..., description="Food type of the restaurant"), street_num: int = Query(..., description="Street number greater than this value")):
    cursor.execute("SELECT T1.id_restaurant FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T2.food_type = ? AND T1.street_num > ? GROUP BY T1.id_restaurant ORDER BY AVG(T2.review) * 0.7 DESC", (food_type, street_num))
    result = cursor.fetchall()
    if not result:
        return {"restaurant_ids": []}
    return {"restaurant_ids": result}

# Endpoint to get the percentage of a specific food type within a street number range
@app.get("/v1/restaurant/get_food_type_percentage_by_street_num_range", operation_id="get_food_type_percentage_by_street_num_range", summary="Retrieves the percentage of a specified food type within a given street number range. This operation calculates the proportion of restaurants serving the requested food type among all restaurants located within the provided street number range.")
async def get_food_type_percentage_by_street_num_range(food_type: str = Query(..., description="Food type of the restaurant"), min_street_num: int = Query(..., description="Minimum street number"), max_street_num: int = Query(..., description="Maximum street number")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.food_type = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.id_restaurant) FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE ? <= T1.street_num <= ?", (food_type, min_street_num, max_street_num))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the label of the highest reviewed restaurant based on food type and city
@app.get("/v1/restaurant/get_highest_reviewed_label_by_food_type_city", operation_id="get_highest_reviewed_label_by_food_type_city", summary="Retrieves the label of the restaurant with the highest review score for a given food type and city. This operation identifies the restaurant with the highest review score within the specified food type and city, and returns its label.")
async def get_highest_reviewed_label_by_food_type_city(food_type: str = Query(..., description="Food type of the restaurant"), city: str = Query(..., description="City of the restaurant")):
    cursor.execute("SELECT label FROM generalinfo WHERE food_type = ? AND city = ? AND review = (SELECT MAX(review) FROM generalinfo WHERE food_type = ? AND city = ?)", (food_type, city, food_type, city))
    result = cursor.fetchall()
    if not result:
        return {"label": []}
    return {"label": result}

# Endpoint to get the count of distinct cities in a specific region
@app.get("/v1/restaurant/count_distinct_cities_by_region", operation_id="get_count_distinct_cities_by_region", summary="Retrieves the number of unique cities located within a specified region.")
async def get_count_distinct_cities_by_region(region: str = Query(..., description="Region")):
    cursor.execute("SELECT COUNT(DISTINCT city) FROM geographic WHERE region = ?", (region,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of restaurants based on city, review score, and food type
@app.get("/v1/restaurant/count_restaurants_by_city_review_foodtype", operation_id="get_count_restaurants", summary="Retrieves the number of restaurants in a given city that have a review score exceeding a specified threshold and offer a particular type of food. The city, minimum review score, and food type are provided as input parameters.")
async def get_count_restaurants(city: str = Query(..., description="City name"), review: int = Query(..., description="Minimum review score"), food_type: str = Query(..., description="Food type")):
    cursor.execute("SELECT COUNT(id_restaurant) FROM generalinfo WHERE city = ? AND review > ? AND food_type = ?", (city, review, food_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the county with the highest number of cities in a specific region
@app.get("/v1/restaurant/top_county_by_region", operation_id="get_top_county", summary="Retrieves the county with the most cities in a given region. The operation groups the data by county and orders it in descending order based on the count of cities within each county. The region is specified as an input parameter.")
async def get_top_county(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT county FROM geographic WHERE region = ? GROUP BY county ORDER BY COUNT(city) DESC LIMIT 1", (region,))
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get the count of restaurants in a specific city
@app.get("/v1/restaurant/count_restaurants_by_city", operation_id="get_count_restaurants_by_city", summary="Retrieves the total number of restaurants located in a specified city. The operation requires the name of the city as an input parameter.")
async def get_count_restaurants_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(id_restaurant) FROM location WHERE city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top 4 regions based on restaurant reviews
@app.get("/v1/restaurant/top_regions_by_review", operation_id="get_top_regions_by_review", summary="Retrieves the top four regions with the highest-rated restaurants, based on review scores. The regions are determined by aggregating review scores from all restaurants within each city and then selecting the top four regions with the highest cumulative scores.")
async def get_top_regions_by_review():
    cursor.execute("SELECT T2.region FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city ORDER BY T1.review DESC LIMIT 4")
    result = cursor.fetchall()
    if not result:
        return {"regions": []}
    return {"regions": [row[0] for row in result]}

# Endpoint to get the count of restaurants based on city, food type, and street name
@app.get("/v1/restaurant/count_restaurants_by_city_foodtype_street", operation_id="get_count_restaurants_by_city_foodtype_street", summary="Retrieves the total number of restaurants in a given city that serve a specific food type and are located on a particular street. This operation requires the city name, food type, and street name as input parameters.")
async def get_count_restaurants_by_city_foodtype_street(city: str = Query(..., description="City name"), food_type: str = Query(..., description="Food type"), street_name: str = Query(..., description="Street name")):
    cursor.execute("SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.city = ? AND T1.food_type = ? AND T2.street_name = ?", (city, food_type, street_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cities based on food type and region
@app.get("/v1/restaurant/count_cities_by_foodtype_region", operation_id="get_count_cities_by_foodtype_region", summary="Retrieves the number of cities in a specified region that offer a particular type of cuisine. The operation requires the food type and region as input parameters.")
async def get_count_cities_by_foodtype_region(food_type: str = Query(..., description="Food type"), region: str = Query(..., description="Region name")):
    cursor.execute("SELECT COUNT(T1.city) FROM geographic AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T2.food_type = ? AND T1.region = ?", (food_type, region))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common food type in a specific region
@app.get("/v1/restaurant/most_common_foodtype_by_region", operation_id="get_most_common_foodtype_by_region", summary="Retrieves the most frequently occurring food type in a given region. The operation identifies the food type that appears most frequently in the city-based data for the specified region. The region is determined by the input parameter.")
async def get_most_common_foodtype_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT T2.food_type FROM geographic AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T1.region = ? GROUP BY T2.food_type ORDER BY COUNT(T2.food_type) DESC LIMIT 1", (region,))
    result = cursor.fetchone()
    if not result:
        return {"food_type": []}
    return {"food_type": result[0]}

# Endpoint to get the count of restaurants based on street name, review score, and city
@app.get("/v1/restaurant/count_restaurants_by_street_review_city", operation_id="get_count_restaurants_by_street_review_city", summary="Retrieve the number of restaurants on a given street with a review score below a certain threshold in a specific city. This operation requires the street name, the maximum review score, and the city name as input parameters.")
async def get_count_restaurants_by_street_review_city(street_name: str = Query(..., description="Street name"), review: int = Query(..., description="Maximum review score"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(T1.id_restaurant) FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T1.street_name = ? AND T2.review < ? AND T1.city = ?", (street_name, review, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the region with the highest number of cities for a specific restaurant label
@app.get("/v1/restaurant/top_region_by_label", operation_id="get_top_region_by_label", summary="Retrieves the region with the most cities that have a restaurant with the specified label. The operation returns the region with the highest count of cities containing a restaurant of the given label.")
async def get_top_region_by_label(label: str = Query(..., description="Restaurant label")):
    cursor.execute("SELECT T2.region AS num FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.label = ? GROUP BY T2.region ORDER BY COUNT(T1.city) DESC LIMIT 1", (label,))
    result = cursor.fetchone()
    if not result:
        return {"region": []}
    return {"region": result[0]}

# Endpoint to get the street names of restaurants based on food type and city
@app.get("/v1/restaurant/street_names_by_foodtype_city", operation_id="get_street_names_by_foodtype_city", summary="Retrieves the street names of restaurants in a given city that serve a specific type of food. The operation requires the food type and city name as input parameters to filter the results.")
async def get_street_names_by_foodtype_city(food_type: str = Query(..., description="Food type"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.street_name FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T2.food_type = ? AND T1.city = ?", (food_type, city))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get the count of food types in a specific region
@app.get("/v1/restaurant/count_food_types_by_region", operation_id="get_count_food_types_by_region", summary="Retrieves the total number of unique food types available in a specified region. The operation filters the data based on the provided region parameter, which determines the geographical area for the count.")
async def get_count_food_types_by_region(region: str = Query(..., description="Region to filter by")):
    cursor.execute("SELECT COUNT(T2.food_type) FROM geographic AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T1.region = ?", (region,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top-rated restaurant location
@app.get("/v1/restaurant/top_rated_restaurant_location", operation_id="get_top_rated_restaurant_location", summary="Retrieves the street name, street number, and city of the restaurant with the highest review score. The data is sourced from the general information and location tables, which are joined based on the restaurant's unique identifier. The results are ordered by review score in descending order, with only the top-rated restaurant's location details returned.")
async def get_top_rated_restaurant_location():
    cursor.execute("SELECT T2.street_name, T2.street_num, T2.city FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant ORDER BY T1.review DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": {"street_name": result[0], "street_num": result[1], "city": result[2]}}

# Endpoint to get the county with the most restaurants
@app.get("/v1/restaurant/top_county_by_restaurant_count", operation_id="get_top_county_by_restaurant_count", summary="Retrieves the county with the highest number of restaurants. The operation calculates the count of restaurants in each county and returns the county with the maximum count. The data is obtained by joining the general information and geographic tables on the city field and grouping the results by county.")
async def get_top_county_by_restaurant_count():
    cursor.execute("SELECT T2.county FROM generalinfo AS T1 INNER JOIN geographic AS T2 ON T1.city = T2.city GROUP BY T2.county ORDER BY COUNT(T1.label) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get the region with the most restaurants
@app.get("/v1/restaurant/top_region_by_restaurant_count", operation_id="get_top_region_by_restaurant_count", summary="Retrieves the region with the highest number of restaurants. This operation identifies the region with the most restaurants by joining the geographic and location tables on the city field, grouping the results by region, and ordering them in descending order based on the count of restaurant IDs. The top region is then returned.")
async def get_top_region_by_restaurant_count():
    cursor.execute("SELECT T1.region FROM geographic AS T1 INNER JOIN location AS T2 ON T1.city = T2.city GROUP BY T1.region ORDER BY COUNT(T2.id_restaurant) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"region": []}
    return {"region": result[0]}

# Endpoint to get restaurant locations with a minimum review score
@app.get("/v1/restaurant/locations_by_min_review", operation_id="get_locations_by_min_review", summary="Retrieves a list of restaurant locations in cities where the review score is equal to or greater than the specified minimum review score. The response includes the street number, street name, and city for each location.")
async def get_locations_by_min_review(min_review: int = Query(..., description="Minimum review score")):
    cursor.execute("SELECT T1.street_num, T1.street_name, T1.city FROM location AS T1 INNER JOIN generalinfo AS T2 ON T1.city = T2.city WHERE T2.review >= ?", (min_review,))
    results = cursor.fetchall()
    if not results:
        return {"locations": []}
    locations = [{"street_num": row[0], "street_name": row[1], "city": row[2]} for row in results]
    return {"locations": locations}

api_calls = [
    "/v1/restaurant/count_restaurants_by_review?review_score=3",
    "/v1/restaurant/top_food_types?limit=4",
    "/v1/restaurant/count_restaurants_by_food_type_and_city?food_type=mediterranean&city=richmond",
    "/v1/restaurant/cities_by_county?county=sonoma%20county",
    "/v1/restaurant/counties_by_region?region=bay%20area",
    "/v1/restaurant/cities_by_region?region=northern%20california",
    "/v1/restaurant/restaurant_ids_by_city_and_street?city=oakland&street_name=11th%20street",
    "/v1/restaurant/count_restaurants_by_street_number?street_num=871",
    "/v1/restaurant/food_types_by_street_and_city?street_name=adeline%20st&city=berkeley",
    "/v1/restaurant/distinct_regions_by_food_type?food_type=african",
    "/v1/restaurant/distinct_counties_by_label?label=a%20%26%20w%20root%20beer",
    "/v1/restaurant/street_info_by_label?label=adelitas%20taqueria",
    "/v1/restaurant/food_type_by_location?county=san%20mateo%20county&street_name=alpine%20rd&street_num=3140",
    "/v1/restaurant/street_names_by_city_food_type?city=san%20francisco&food_type=seafood",
    "/v1/restaurant/distinct_counties_excluding_label?label=bakers%20square%20restaurant%20%26%20pie%20shop",
    "/v1/restaurant/count_distinct_counties_by_street_name?street_name=appian%20way",
    "/v1/restaurant/reviews_by_street_name?street_name=atlantic%20ave",
    "/v1/restaurant/restaurant_ids_by_county?county=contra%20costa%20county",
    "/v1/restaurant/restaurant_info_by_county?county=yolo%20county",
    "/v1/restaurant/labels_by_street_food_city?street_name=drive&food_type=american&city=san%20rafael",
    "/v1/restaurant/street_names_by_city_review?city=san%20francisco&review=1.7",
    "/v1/restaurant/top_label_by_street_city?street_name=avenida%20de%20las%20pulgas&city=menlo%20park",
    "/v1/restaurant/street_names_by_label_county?label=good%20heavens&county=tuolumne%20county",
    "/v1/restaurant/distinct_street_nums_by_label?label=aux%20delices%20vietnamese%20restaurant",
    "/v1/restaurant/ids_by_county?county=marin%20county",
    "/v1/restaurant/avg_review_by_county?county=santa%20cruz%20county",
    "/v1/restaurant/percentage_food_type_by_county?food_type=mexican&county=monterey%20county",
    "/v1/restaurant/percentage_street_by_county?street_name=11th%20st&county=alameda%20county",
    "/v1/restaurant/labels_by_food_type?food_type=european",
    "/v1/restaurant/cities_by_county_region?county=unknown&region=unknown",
    "/v1/restaurant/county_region_by_city?city=Davis",
    "/v1/restaurant/street_names_by_city?city=Clayton",
    "/v1/restaurant/top_reviewed_restaurant_by_city?city=San%20Francisco",
    "/v1/restaurant/count_lowest_reviewed_restaurants?food_type=american&city=carmel",
    "/v1/restaurant/percentage_food_type_by_city?food_type=american%20food&city=dublin",
    "/v1/restaurant/location_by_label?label=Albert's%20Caf\u00e9",
    "/v1/restaurant/restaurant_ids_by_city_street?city=Oakland&street_name=19th%20St",
    "/v1/restaurant/food_type_by_street?street_num=106&street_name=e%2025th%20ave",
    "/v1/restaurant/labels_by_region?region=unknown",
    "/v1/restaurant/labels_by_street_city?street_num=104&city=campbell&street_name=san%20tomas%20aquino%20road",
    "/v1/restaurant/count_by_food_type_city_street?food_type=thai&city=albany&street_name=san%20pablo%20ave",
    "/v1/restaurant/geographic_info_by_label?label=plearn-thai%20cuisine",
    "/v1/restaurant/labels_by_region_county?region=lake%20tahoe&county=el%20dorado%20county",
    "/v1/restaurant/distinct_counties_regions_by_street?street_name=E.%20El%20Camino%20Real",
    "/v1/restaurant/top_rated_by_city_street_food_type?city=berkeley&street_name=shattuck%20ave&food_type=Indian%20restaurant",
    "/v1/restaurant/highly_rated_percentage_by_region?region=bay%20area",
    "/v1/restaurant/most_common_street_name",
    "/v1/restaurant/top_rated_restaurant_by_food_type?food_type=chicken",
    "/v1/restaurant/county_by_city?city=el%20cerrito",
    "/v1/restaurant/count_restaurants_by_street?street_name=irving",
    "/v1/restaurant/labels_by_county?county=marin%20county",
    "/v1/restaurant/street_names_by_label?label=peking%20duck%20restaurant",
    "/v1/restaurant/street_names_by_county_and_restaurant_count?county=alameda%20county&restaurant_count=10",
    "/v1/restaurant/regions_by_food_type?food_type=greek",
    "/v1/restaurant/street_names_by_region?region=unknown",
    "/v1/restaurant/reviews_by_street_name_and_number?street_name=murray%20ave&street_num=8440",
    "/v1/restaurant/most_common_food_type_by_county?county=Monterey",
    "/v1/restaurant/most_common_street_name_by_city_and_food_type?city=san%20francisco&food_type=burgers",
    "/v1/restaurant/region_by_street_number_and_name?street_num=1149&street_name=el%20camino%20real",
    "/v1/restaurant/county_by_label?label=sankee",
    "/v1/restaurant/count_cities_by_region?region=northern%20california",
    "/v1/restaurant/labels_by_street_name?street_name=park%20st",
    "/v1/restaurant/percentage_restaurants_by_region?region=bay%20area",
    "/v1/restaurant/average_review_by_food_type?food_type=chinese",
    "/v1/restaurant/get_restaurant_ids_by_city?city=Danville",
    "/v1/restaurant/count_labels_by_county?county=unknown",
    "/v1/restaurant/get_street_names_by_food_type?food_type=American",
    "/v1/restaurant/get_reviews_by_street?street_name=Broadway&street_num=430",
    "/v1/restaurant/get_top_rated_location",
    "/v1/restaurant/get_top_county_by_food_type?food_type=Italian",
    "/v1/restaurant/get_region_percentage?region=Napa%20Valley",
    "/v1/restaurant/cities_by_region_county?region=bay%20area&county=santa%20clara%20county",
    "/v1/restaurant/count_cities_by_street?street_name=railroad&street_num=1000",
    "/v1/restaurant/labels_by_food_type_city?food_type=24%20hour%20diner&city=san%20francisco",
    "/v1/restaurant/reviews_by_city_street?city=santa%20cruz&street_name=ocean%20st",
    "/v1/restaurant/street_numbers_by_review_city_food_type?review=2.7&city=oakland&food_type=bar",
    "/v1/restaurant/count_restaurants_by_food_type_city_street?food_type=bakery&city=palo%20alto&street_name=university%20ave.",
    "/v1/restaurant/street_numbers_by_label_food_type?label=Tulocay%20winery&food_type=winery",
    "/v1/restaurant/reviews_labels_by_city_street?city=hayward&street_name=mission%20blvd",
    "/v1/restaurant/count_restaurants_by_street_city_food_type_label?street_name=castro%20st&city=mountain%20view&food_type=indian&label_pattern=%25cookhouse%25",
    "/v1/restaurant/count_restaurants_by_review_street_num?review=2&street_num=500",
    "/v1/restaurant/count_restaurants_by_city_food_type_street_name_id?city=milpitas&food_type=asian&street_name=n%20milpitas%20blvd&id_restaurant=385",
    "/v1/restaurant/get_labels_ids_by_city_street_name?city=san%20francisco&street_name=ocean%20avenue",
    "/v1/restaurant/get_city_street_by_label?label=sanuki%20restaurant",
    "/v1/restaurant/get_food_type_by_street_city?street_num=22779&street_name=6th%20St&city=hayward",
    "/v1/restaurant/count_food_type_by_city_street?food_type=american&city=san%20francisco&street_name=front",
    "/v1/restaurant/get_restaurant_ids_by_food_type_street_num?food_type=american&street_num=2000",
    "/v1/restaurant/get_food_type_percentage_by_street_num_range?food_type=afghani&min_street_num=1000&max_street_num=2000",
    "/v1/restaurant/get_highest_reviewed_label_by_food_type_city?food_type=asian&city=san%20francisco",
    "/v1/restaurant/count_distinct_cities_by_region?region=monterey",
    "/v1/restaurant/count_restaurants_by_city_review_foodtype?city=belmont&review=2&food_type=deli",
    "/v1/restaurant/top_county_by_region?region=northern%20california",
    "/v1/restaurant/count_restaurants_by_city?city=concord",
    "/v1/restaurant/top_regions_by_review",
    "/v1/restaurant/count_restaurants_by_city_foodtype_street?city=livermore&food_type=chinese&street_name=1st%20st",
    "/v1/restaurant/count_cities_by_foodtype_region?food_type=indian&region=los%20angeles%20area",
    "/v1/restaurant/most_common_foodtype_by_region?region=bay%20area",
    "/v1/restaurant/count_restaurants_by_street_review_city?street_name=broadway&review=3&city=oakland",
    "/v1/restaurant/top_region_by_label?label=baskin%20robbins",
    "/v1/restaurant/street_names_by_foodtype_city?food_type=pizza&city=san%20jose",
    "/v1/restaurant/count_food_types_by_region?region=yosemite%20and%20mono%20lake%20area",
    "/v1/restaurant/top_rated_restaurant_location",
    "/v1/restaurant/top_county_by_restaurant_count",
    "/v1/restaurant/top_region_by_restaurant_count",
    "/v1/restaurant/locations_by_min_review?min_review=4"
]
