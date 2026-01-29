from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/public_review_platform/public_review_platform.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of businesses in a specific state with stars below a certain threshold
@app.get("/v1/public_review_platform/business_count_state_stars", operation_id="get_business_count_state_stars", summary="Retrieves the total number of businesses located in a specified state that have a star rating below a given threshold. The state parameter filters the businesses by their location, while the max_stars parameter sets the upper limit for the star rating.")
async def get_business_count_state_stars(state: str = Query(..., description="State of the business"), max_stars: int = Query(..., description="Maximum stars for the business")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE state LIKE ? AND stars < ?", (state, max_stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses in a specific state with a specific active status
@app.get("/v1/public_review_platform/business_count_state_active", operation_id="get_business_count_state_active", summary="Retrieves the total number of businesses in a specified state that have a particular active status. The state and active status are provided as input parameters.")
async def get_business_count_state_active(state: str = Query(..., description="State of the business"), active: str = Query(..., description="Active status of the business")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE state LIKE ? AND active LIKE ?", (state, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of reviews of a specific length by a specific user
@app.get("/v1/public_review_platform/review_count_user_length", operation_id="get_review_count_user_length", summary="Retrieves the total number of reviews of a specified length that were written by a particular user. The user is identified by their unique user ID, and the length of the reviews is determined by a specified value.")
async def get_review_count_user_length(user_id: int = Query(..., description="User ID"), review_length: str = Query(..., description="Length of the review")):
    cursor.execute("SELECT COUNT(review_length) FROM Reviews WHERE user_id = ? AND review_length LIKE ?", (user_id, review_length))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users who are fans of a specific entity
@app.get("/v1/public_review_platform/user_count_fans", operation_id="get_user_count_fans", summary="Retrieves the total number of users who are fans of a specified entity. The input parameter determines the entity for which the fan count is calculated.")
async def get_user_count_fans(user_fans: str = Query(..., description="Entity the users are fans of")):
    cursor.execute("SELECT COUNT(user_id) FROM Users WHERE user_fans LIKE ?", (user_fans,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with a specific attribute name and value
@app.get("/v1/public_review_platform/business_count_attribute", operation_id="get_business_count_attribute", summary="Retrieves the total number of businesses that possess a specified attribute with a given value. The attribute name and value are provided as input parameters.")
async def get_business_count_attribute(attribute_name: str = Query(..., description="Name of the attribute"), attribute_value: str = Query(..., description="Value of the attribute")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T1.attribute_name LIKE ? AND T2.attribute_value LIKE ?", (attribute_name, attribute_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the attribute value for a specific business and attribute name
@app.get("/v1/public_review_platform/business_attribute_value", operation_id="get_business_attribute_value", summary="Retrieves the value of a specific attribute for a given business. The attribute is identified by its name. The business is identified by its unique ID.")
async def get_business_attribute_value(business_id: int = Query(..., description="Business ID"), attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT T2.attribute_value FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T2.business_id = ? AND T1.attribute_name LIKE ?", (business_id, attribute_name))
    result = cursor.fetchone()
    if not result:
        return {"attribute_value": []}
    return {"attribute_value": result[0]}

# Endpoint to get the count of businesses in a specific category
@app.get("/v1/public_review_platform/business_count_category", operation_id="get_business_count_category", summary="Retrieves the total number of businesses associated with a specified category. The category is identified by its name, which is used to filter the results.")
async def get_business_count_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT COUNT(T1.category_id) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id WHERE T1.category_name LIKE ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in count between two categories
@app.get("/v1/public_review_platform/category_count_difference", operation_id="get_category_count_difference", summary="Retrieves the difference in the number of businesses between two specified categories. The operation compares the count of businesses in the first category with that of the second category and returns the difference.")
async def get_category_count_difference(category_name_1: str = Query(..., description="Name of the first category"), category_name_2: str = Query(..., description="Name of the second category")):
    cursor.execute("SELECT SUM(CASE WHEN T1.category_name LIKE ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.category_name LIKE ? THEN 1 ELSE 0 END) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id", (category_name_1, category_name_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the top category with the most 5-star reviews
@app.get("/v1/public_review_platform/top_category_5_star_reviews", operation_id="get_top_category_5_star_reviews", summary="Retrieves the category with the highest number of 5-star reviews. This operation considers all businesses and their respective reviews, filtering for those with the specified number of stars. The result is the category name with the most occurrences of the selected star rating.")
async def get_top_category_5_star_reviews(review_stars: int = Query(..., description="Number of stars in the review")):
    cursor.execute("SELECT T1.category_name FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Reviews AS T4 ON T3.business_id = T4.business_id WHERE T4.review_stars = ? GROUP BY T1.category_name ORDER BY COUNT(T1.category_name) DESC LIMIT 1", (review_stars,))
    result = cursor.fetchone()
    if not result:
        return {"category_name": []}
    return {"category_name": result[0]}

# Endpoint to get the year with the most 5-star reviews
@app.get("/v1/public_review_platform/top_year_5_star_reviews", operation_id="get_top_year_5_star_reviews", summary="Retrieves the year in which the highest number of 5-star reviews were recorded. The input parameter specifies the star rating of the reviews to consider. The operation returns the year with the most reviews of the specified star rating.")
async def get_top_year_5_star_reviews(review_stars: int = Query(..., description="Number of stars in the review")):
    cursor.execute("SELECT T2.user_yelping_since_year FROM Reviews AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id WHERE T1.review_stars = ? GROUP BY T2.user_yelping_since_year ORDER BY COUNT(T1.review_stars) DESC LIMIT 1", (review_stars,))
    result = cursor.fetchone()
    if not result:
        return {"user_yelping_since_year": []}
    return {"user_yelping_since_year": result[0]}

# Endpoint to get the average review stars for users with a specific review length
@app.get("/v1/public_review_platform/average_review_stars", operation_id="get_average_review_stars", summary="Retrieves the average review stars for users who have written reviews of a specified length. The calculation is based on the sum of review stars divided by the count of reviews for each user, with the results ordered by the count of reviews in descending order. Only the top user's average review stars are returned.")
async def get_average_review_stars(review_length: str = Query(..., description="Length of the review (e.g., 'Long')")):
    cursor.execute("SELECT CAST(SUM(T1.review_stars) AS REAL) / COUNT(T1.review_stars) FROM Reviews AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id WHERE T1.review_length LIKE ? GROUP BY T1.user_id ORDER BY COUNT(T1.review_length) DESC LIMIT 1", (review_length,))
    result = cursor.fetchone()
    if not result:
        return {"average_stars": []}
    return {"average_stars": result[0]}

# Endpoint to get the top category name for businesses with a specific review length
@app.get("/v1/public_review_platform/top_category_by_review_length", operation_id="get_top_category_by_review_length", summary="Retrieves the name of the category with the most businesses having reviews of a specified length. The input parameter determines the length of the reviews to consider.")
async def get_top_category_by_review_length(review_length: str = Query(..., description="Length of the review (e.g., 'Long')")):
    cursor.execute("SELECT T4.category_name FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Business_Categories AS T3 ON T2.business_id = T3.business_id INNER JOIN Categories AS T4 ON T3.category_id = T4.category_id WHERE T1.review_length LIKE ? GROUP BY T2.business_id ORDER BY COUNT(T1.review_length) DESC LIMIT 1", (review_length,))
    result = cursor.fetchone()
    if not result:
        return {"category_name": []}
    return {"category_name": result[0]}

# Endpoint to get distinct category names for businesses with a specific tip length
@app.get("/v1/public_review_platform/distinct_categories_by_tip_length", operation_id="get_distinct_categories_by_tip_length", summary="Retrieve a unique set of category names for businesses that have received tips of a specified length. The length of the tip can be defined using a keyword such as 'short'.")
async def get_distinct_categories_by_tip_length(tip_length: str = Query(..., description="Length of the tip (e.g., 'short')")):
    cursor.execute("SELECT DISTINCT T1.category_name FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Tips AS T4 ON T3.business_id = T4.business_id WHERE T4.tip_length LIKE ?", (tip_length,))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the top user yelping since year for users with a specific tip length
@app.get("/v1/public_review_platform/top_user_yelping_since_year", operation_id="get_top_user_yelping_since_year", summary="Retrieves the year when the most active user, based on the number of tips of a specific length, started yelping. The user's activity is determined by the count of their tips of the specified length.")
async def get_top_user_yelping_since_year(tip_length: str = Query(..., description="Length of the tip (e.g., 'short')")):
    cursor.execute("SELECT T2.user_yelping_since_year FROM Tips AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id WHERE T1.tip_length LIKE ? GROUP BY T2.user_yelping_since_year ORDER BY COUNT(T1.tip_length) DESC LIMIT 1", (tip_length,))
    result = cursor.fetchone()
    if not result:
        return {"user_yelping_since_year": []}
    return {"user_yelping_since_year": result[0]}

# Endpoint to get category names for tips by a specific user
@app.get("/v1/public_review_platform/category_names_by_user", operation_id="get_category_names_by_user", summary="Retrieves the category names associated with tips submitted by a specific user. This operation fetches the category names from the database by joining the relevant tables based on the provided user ID.")
async def get_category_names_by_user(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T4.category_name FROM Tips AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Business_Categories AS T3 ON T2.business_id = T3.business_id INNER JOIN Categories AS T4 ON T3.category_id = T4.category_id WHERE T1.user_id = ?", (user_id,))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get business stars for tips by a specific user
@app.get("/v1/public_review_platform/business_stars_by_user", operation_id="get_business_stars_by_user", summary="Retrieves the star ratings of businesses that a specific user has reviewed. The user is identified by their unique user_id.")
async def get_business_stars_by_user(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T2.stars FROM Tips AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T1.user_id = ?", (user_id,))
    result = cursor.fetchall()
    if not result:
        return {"business_stars": []}
    return {"business_stars": [row[0] for row in result]}

# Endpoint to get the percentage of businesses in a specific category
@app.get("/v1/public_review_platform/percentage_businesses_in_category", operation_id="get_percentage_businesses_in_category", summary="Retrieves the percentage of businesses that fall under a specified category. The category is identified by its name, which is used to filter the businesses and calculate the percentage. The result is based on the total count of businesses and those that match the given category.")
async def get_percentage_businesses_in_category(category_name: str = Query(..., description="Category name (e.g., 'Automotive')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.category_name LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.business_id) AS 'percentage' FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in percentage of businesses between two categories
@app.get("/v1/public_review_platform/percentage_difference_between_categories", operation_id="get_percentage_difference_between_categories", summary="Retrieves the percentage difference in the number of businesses between two specified categories. This operation compares the total count of businesses in the first category with that of the second category, and returns the difference as a percentage.")
async def get_percentage_difference_between_categories(category_name_1: str = Query(..., description="First category name (e.g., 'Women's Clothing')"), category_name_2: str = Query(..., description="Second category name (e.g., 'Men's Clothing')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.category_name LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.business_id) - CAST(SUM(CASE WHEN T2.category_name LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.business_id) AS 'more percentage' FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id", (category_name_1, category_name_2))
    result = cursor.fetchone()
    if not result:
        return {"more_percentage": []}
    return {"more_percentage": result[0]}

# Endpoint to get the count of users who started yelping in a specific year
@app.get("/v1/public_review_platform/user_count_by_yelping_year", operation_id="get_user_count_by_yelping_year", summary="Retrieves the total number of users who began using the platform in a specified year. The year is provided as an input parameter.")
async def get_user_count_by_yelping_year(user_yelping_since_year: int = Query(..., description="Year when the user started yelping (e.g., 2004)")):
    cursor.execute("SELECT COUNT(user_id) FROM Users WHERE user_yelping_since_year = ?", (user_yelping_since_year,))
    result = cursor.fetchone()
    if not result:
        return {"user_count": []}
    return {"user_count": result[0]}

# Endpoint to get the count of users who started yelping in a specific year and have a specific number of fans
@app.get("/v1/public_review_platform/user_count_by_yelping_year_and_fans", operation_id="get_user_count_by_yelping_year_and_fans", summary="Retrieves the total number of users who began yelping in a specified year and have a certain number of fans. The year is provided as a four-digit number, and the number of fans can be a specific value or 'None' for users without fans.")
async def get_user_count_by_yelping_year_and_fans(user_yelping_since_year: int = Query(..., description="Year when the user started yelping (e.g., 2005)"), user_fans: str = Query(..., description="Number of fans (e.g., 'None')")):
    cursor.execute("SELECT COUNT(user_id) FROM Users WHERE user_yelping_since_year = ? AND user_fans LIKE ?", (user_yelping_since_year, user_fans))
    result = cursor.fetchone()
    if not result:
        return {"user_count": []}
    return {"user_count": result[0]}

# Endpoint to get the count of businesses based on city and active status
@app.get("/v1/public_review_platform/business_count_by_city_and_status", operation_id="get_business_count", summary="Retrieves the total number of businesses in a specified city that have a particular active status. The active status can be either TRUE or FALSE.")
async def get_business_count(city: str = Query(..., description="City name"), active: str = Query(..., description="Active status (TRUE or FALSE)")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE city LIKE ? AND active LIKE ?", (city, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of reviews based on user ID
@app.get("/v1/public_review_platform/review_count_by_user", operation_id="get_review_count_by_user", summary="Retrieves the total number of reviews authored by a particular user. The user is identified by their unique user ID.")
async def get_review_count_by_user(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT COUNT(review_length) FROM Reviews WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of reviews based on business ID and review stars
@app.get("/v1/public_review_platform/review_count_by_business_and_stars", operation_id="get_review_count_by_business_and_stars", summary="Retrieves the total number of reviews for a particular business, filtered by a specific star rating. The business is identified by its unique ID, and the star rating is a numerical value representing the review score.")
async def get_review_count_by_business_and_stars(business_id: int = Query(..., description="Business ID"), review_stars: int = Query(..., description="Review star rating")):
    cursor.execute("SELECT COUNT(review_length) FROM Reviews WHERE business_id = ? AND review_stars = ?", (business_id, review_stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the business ID with the most reviews in a specific city and active status
@app.get("/v1/public_review_platform/top_business_by_reviews", operation_id="get_top_business_by_reviews", summary="Retrieves the ID of the business with the highest number of reviews in a specified city, considering only businesses with a certain active status. The active status can be either TRUE or FALSE.")
async def get_top_business_by_reviews(city: str = Query(..., description="City name"), active: str = Query(..., description="Active status (TRUE or FALSE)")):
    cursor.execute("SELECT T1.business_id FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city LIKE ? AND T1.active LIKE ? GROUP BY T1.business_id ORDER BY COUNT(T2.review_length) DESC LIMIT 1", (city, active))
    result = cursor.fetchone()
    if not result:
        return {"business_id": []}
    return {"business_id": result[0]}

# Endpoint to get the count of reviews based on city and review length
@app.get("/v1/public_review_platform/review_count_by_city_and_length", operation_id="get_review_count_by_city_and_length", summary="Retrieves the total number of reviews in a specified city that match a given review length. The city is identified by its name, and the review length can be 'Short', 'Medium', or 'Long'.")
async def get_review_count_by_city_and_length(city: str = Query(..., description="City name"), review_length: str = Query(..., description="Review length (e.g., 'Medium')")):
    cursor.execute("SELECT COUNT(T2.review_length) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city LIKE ? AND T2.review_length LIKE ?", (city, review_length))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to check if a business has a specific attribute
@app.get("/v1/public_review_platform/check_business_attribute", operation_id="check_business_attribute", summary="Check if a business has a specific attribute")
async def check_business_attribute(attribute_name: str = Query(..., description="Attribute name (e.g., 'Has TV')"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT DISTINCT CASE WHEN T1.attribute_name LIKE ? THEN 'yes' ELSE 'no' END FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T2.business_id = ?", (attribute_name, business_id))
    result = cursor.fetchone()
    if not result:
        return {"has_attribute": []}
    return {"has_attribute": result[0]}

# Endpoint to get the operating hours of a business on a specific day
@app.get("/v1/public_review_platform/business_operating_hours", operation_id="get_business_operating_hours", summary="Retrieves the total operating hours of a specific business on a given day of the week. The day of the week and the business ID are required as input parameters to determine the operating hours.")
async def get_business_operating_hours(day_of_week: str = Query(..., description="Day of the week (e.g., 'Saturday')"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT T1.closing_time - T1.opening_time AS 'hour' FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id WHERE T2.day_of_week LIKE ? AND T1.business_id = ?", (day_of_week, business_id))
    result = cursor.fetchone()
    if not result:
        return {"operating_hours": []}
    return {"operating_hours": result[0]}

# Endpoint to get the city of businesses based on tips likes and user ID
@app.get("/v1/public_review_platform/business_city_by_tips_likes_user_id", operation_id="get_business_city_by_tips_likes_user_id", summary="Retrieves the city of businesses where tips, given by a specific user, have a specified number of likes. This operation allows you to filter tips based on the number of likes they have received and the user who provided them, providing a targeted view of businesses in a particular city.")
async def get_business_city_by_tips_likes_user_id(likes: int = Query(..., description="Number of likes on the tip"), user_id: int = Query(..., description="User ID who gave the tip")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Tips AS T2 ON T1.business_id = T2.business_id WHERE T2.likes = ? AND T2.user_id = ?", (likes, user_id))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of attributes for a business based on attribute value and business ID
@app.get("/v1/public_review_platform/count_attributes_by_value_business_id", operation_id="get_count_attributes_by_value_business_id", summary="Retrieves the count of attributes associated with a specific business, where the attribute value matches a given pattern. The business is identified by its unique ID, and the attribute value pattern is used to filter the attributes.")
async def get_count_attributes_by_value_business_id(attribute_value: str = Query(..., description="Attribute value pattern"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT COUNT(T1.attribute_name) FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T2.attribute_value LIKE ? AND T2.business_id = ?", (attribute_value, business_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of compliments for a user based on compliment type and user ID
@app.get("/v1/public_review_platform/count_compliments_by_type_user_id", operation_id="get_count_compliments_by_type_user_id", summary="Retrieves the total count of compliments for a specific user, filtered by a given compliment type pattern. This operation requires the user ID and a pattern to match against the compliment type.")
async def get_count_compliments_by_type_user_id(compliment_type: str = Query(..., description="Compliment type pattern"), user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT COUNT(T1.compliment_type) FROM Compliments AS T1 INNER JOIN Users_Compliments AS T2 ON T1.compliment_id = T2.compliment_id WHERE T1.compliment_type LIKE ? AND T2.user_id = ?", (compliment_type, user_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user IDs based on compliment type
@app.get("/v1/public_review_platform/user_ids_by_compliment_type", operation_id="get_user_ids_by_compliment_type", summary="Retrieves the user IDs of individuals who have given compliments that match a specified type pattern. The compliment type pattern is used to filter the compliments and identify the corresponding users.")
async def get_user_ids_by_compliment_type(compliment_type: str = Query(..., description="Compliment type pattern")):
    cursor.execute("SELECT user_id FROM Users_Compliments WHERE compliment_id IN ( SELECT compliment_id FROM Compliments WHERE compliment_type LIKE ? )", (compliment_type,))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get business IDs based on city, attribute name, and attribute value
@app.get("/v1/public_review_platform/business_ids_by_city_attribute", operation_id="get_business_ids_by_city_attribute", summary="Retrieves business IDs from a specific city that have a certain attribute name and value, and a business ID less than a specified maximum value. The city name, attribute name, and attribute value are matched using pattern matching.")
async def get_business_ids_by_city_attribute(max_business_id: int = Query(..., description="Maximum business ID"), city: str = Query(..., description="City name"), attribute_name: str = Query(..., description="Attribute name pattern"), attribute_value: str = Query(..., description="Attribute value pattern")):
    cursor.execute("SELECT T2.business_id FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T3.business_id < ? AND T3.city LIKE ? AND T1.attribute_name LIKE ? AND T2.attribute_value LIKE ?", (max_business_id, city, attribute_name, attribute_value))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the average stars of businesses based on attribute name and value
@app.get("/v1/public_review_platform/avg_stars_by_attribute", operation_id="get_avg_stars_by_attribute", summary="Retrieves the average star rating of businesses that match a specified attribute name and value pattern. The operation calculates the average by summing the star ratings of all businesses with the specified attribute and then dividing by the total number of businesses with that attribute.")
async def get_avg_stars_by_attribute(attribute_name: str = Query(..., description="Attribute name pattern"), attribute_value: str = Query(..., description="Attribute value pattern")):
    cursor.execute("SELECT CAST(SUM(T3.stars) AS REAL) / COUNT(T2.business_id) AS \"avg\" FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.attribute_name LIKE ? AND T2.attribute_value LIKE ?", (attribute_name, attribute_value))
    result = cursor.fetchone()
    if not result:
        return {"avg": []}
    return {"avg": result[0]}

# Endpoint to get the percentage of businesses in a specific city based on attribute name and value
@app.get("/v1/public_review_platform/percentage_businesses_by_city_attribute", operation_id="get_percentage_businesses_by_city_attribute", summary="Retrieves the percentage of businesses in a given city that possess a specific attribute. The attribute is identified by its name and value pattern. The result is calculated by comparing the count of businesses with the specified attribute in the city to the total number of businesses in the city.")
async def get_percentage_businesses_by_city_attribute(city: str = Query(..., description="City name"), attribute_name: str = Query(..., description="Attribute name pattern"), attribute_value: str = Query(..., description="Attribute value pattern")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.city LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.business_id) AS \"percentage\" FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.attribute_name LIKE ? AND T2.attribute_value LIKE ?", (city, attribute_name, attribute_value))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get business IDs based on state and stars
@app.get("/v1/public_review_platform/business_ids_by_state_stars", operation_id="get_business_ids_by_state_stars", summary="Retrieves the unique identifiers of businesses located in a specified state that have a particular star rating. The state and star rating are provided as input parameters.")
async def get_business_ids_by_state_stars(state: str = Query(..., description="State name"), stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT business_id FROM Business WHERE state LIKE ? AND stars = ?", (state, stars))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of businesses based on review count and active status
@app.get("/v1/public_review_platform/count_businesses_by_review_count_active", operation_id="get_count_businesses_by_review_count_active", summary="Retrieves the number of businesses that match the specified review count pattern and active status. The review count pattern is used to filter businesses based on their review count, while the active status pattern is used to filter businesses based on their active status.")
async def get_count_businesses_by_review_count_active(review_count: str = Query(..., description="Review count pattern"), active: str = Query(..., description="Active status pattern")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE review_count LIKE ? AND active LIKE ?", (review_count, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users based on user ID range, user fans, and user review count
@app.get("/v1/public_review_platform/count_users_by_id_range_fans_review_count", operation_id="get_count_users_by_id_range_fans_review_count", summary="Retrieves the total number of users within a specified user ID range who have a certain number of fans and a certain review count. The user ID range is defined by the minimum and maximum user IDs. The user fans pattern and user review count pattern are used to filter users based on their number of fans and review count, respectively.")
async def get_count_users_by_id_range_fans_review_count(min_user_id: int = Query(..., description="Minimum user ID"), max_user_id: int = Query(..., description="Maximum user ID"), user_fans: str = Query(..., description="User fans pattern"), user_review_count: str = Query(..., description="User review count pattern")):
    cursor.execute("SELECT COUNT(user_id) FROM Users WHERE user_id BETWEEN ? AND ? AND user_fans LIKE ? AND user_review_count LIKE ?", (min_user_id, max_user_id, user_fans, user_review_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the opening time of businesses in a specific category
@app.get("/v1/public_review_platform/business_opening_time_by_category", operation_id="get_business_opening_time_by_category", summary="Retrieve the opening time of businesses that belong to a specified category. The category is identified by its name, which is provided as an input parameter. This operation returns the opening time of all businesses in the given category.")
async def get_business_opening_time_by_category(category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT T4.opening_time FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Business_Hours AS T4 ON T3.business_id = T4.business_id WHERE T1.category_name LIKE ?", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"opening_times": []}
    return {"opening_times": [item[0] for item in result]}

# Endpoint to get the count of businesses in a specific category that open before a certain time
@app.get("/v1/public_review_platform/business_count_by_opening_time_and_category", operation_id="get_business_count_by_opening_time_and_category", summary="Retrieves the number of businesses in a specified category that open before a given time. The operation accepts the opening time in 'HH:MM AM/PM' format and the category name as input parameters to filter the businesses accordingly.")
async def get_business_count_by_opening_time_and_category(opening_time: str = Query(..., description="Opening time in 'HH:MM AM/PM' format"), category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT COUNT(T3.business_id) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Business_Hours AS T4 ON T3.business_id = T4.business_id WHERE T4.opening_time < ? AND T1.category_name LIKE ?", (opening_time, category_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the business IDs in a specific category that close after a certain time
@app.get("/v1/public_review_platform/business_ids_by_closing_time_and_category", operation_id="get_business_ids_by_closing_time_and_category", summary="Retrieve the identifiers of businesses in a specified category that close after a certain time. The operation requires the closing time in 'HH:MM AM/PM' format and the category name of the business as input parameters. The result is a list of business identifiers that meet the provided criteria.")
async def get_business_ids_by_closing_time_and_category(closing_time: str = Query(..., description="Closing time in 'HH:MM AM/PM' format"), category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT T3.business_id FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Business_Hours AS T4 ON T3.business_id = T4.business_id WHERE T4.closing_time > ? AND T1.category_name LIKE ?", (closing_time, category_name))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [item[0] for item in result]}

# Endpoint to get the count of businesses open on specific days of the week
@app.get("/v1/public_review_platform/business_count_by_days_of_week", operation_id="get_business_count_by_days_of_week", summary="Retrieve the total number of businesses that are open on up to four specified days of the week. The operation considers businesses from various categories and their respective opening hours.")
async def get_business_count_by_days_of_week(day1: str = Query(..., description="First day of the week"), day2: str = Query(..., description="Second day of the week"), day3: str = Query(..., description="Third day of the week"), day4: str = Query(..., description="Fourth day of the week")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Business_Hours AS T4 ON T3.business_id = T4.business_id INNER JOIN Days AS T5 ON T4.day_id = T5.day_id WHERE T5.day_of_week LIKE ? OR T5.day_of_week LIKE ? OR T5.day_of_week LIKE ? OR T5.day_of_week LIKE ?", (day1, day2, day3, day4))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses in a specific category and city
@app.get("/v1/public_review_platform/business_count_by_category_and_city", operation_id="get_business_count_by_category_and_city", summary="Retrieves the total number of businesses in a specified category and city. The operation uses the provided category name and city to filter the businesses and returns the count.")
async def get_business_count_by_category_and_city(category_name: str = Query(..., description="Category name of the business"), city: str = Query(..., description="City of the business")):
    cursor.execute("SELECT COUNT(*) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.category_name = ? AND T3.city = ?", (category_name, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the business IDs in a specific category with a rating below a certain number of stars
@app.get("/v1/public_review_platform/business_ids_by_category_and_stars", operation_id="get_business_ids_by_category_and_stars", summary="Retrieve the IDs of businesses in a specified category that have a rating below a certain number of stars. The category name and maximum star rating are provided as input parameters to filter the results.")
async def get_business_ids_by_category_and_stars(category_name: str = Query(..., description="Category name of the business"), max_stars: int = Query(..., description="Maximum number of stars")):
    cursor.execute("SELECT T2.business_id FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.category_name LIKE ? AND T3.stars < ?", (category_name, max_stars))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [item[0] for item in result]}

# Endpoint to get distinct business IDs based on review usefulness and business activity status
@app.get("/v1/public_review_platform/distinct_business_ids_by_review_usefulness_and_activity", operation_id="get_distinct_business_ids_by_review_usefulness_and_activity", summary="Retrieve a unique list of business identifiers based on the usefulness of their reviews and their current activity status. The operation filters businesses by their activity status and the usefulness of their reviews, as indicated by the input parameters.")
async def get_distinct_business_ids_by_review_usefulness_and_activity(active_status: str = Query(..., description="Activity status of the business"), review_votes_useful: str = Query(..., description="Usefulness of the review votes")):
    cursor.execute("SELECT DISTINCT T1.business_id FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Business_Categories AS T3 ON T2.business_id = T3.business_id INNER JOIN Categories AS T4 ON T3.category_id = T4.category_id WHERE T2.active LIKE ? AND T1.review_votes_useful LIKE ?", (active_status, review_votes_useful))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [item[0] for item in result]}

# Endpoint to get category names based on review length and category ID range
@app.get("/v1/public_review_platform/category_names_by_review_length_and_category_id_range", operation_id="get_category_names_by_review_length_and_category_id_range", summary="Retrieves the names of categories that meet the specified review length and fall within the given category ID range. The review length is used to filter the reviews, and the category ID range is used to filter the businesses associated with those categories. The results are grouped by category name.")
async def get_category_names_by_review_length_and_category_id_range(review_length: str = Query(..., description="Length of the review"), min_category_id: int = Query(..., description="Minimum category ID"), max_category_id: int = Query(..., description="Maximum category ID")):
    cursor.execute("SELECT T4.category_name FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Business_Categories AS T3 ON T2.business_id = T3.business_id INNER JOIN Categories AS T4 ON T3.category_id = T4.category_id WHERE T1.review_length LIKE ? AND T3.category_id BETWEEN ? AND ? GROUP BY T4.category_name", (review_length, min_category_id, max_category_id))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [item[0] for item in result]}

# Endpoint to get business attribute values based on category name and city
@app.get("/v1/public_review_platform/business_attribute_values_by_category_and_city", operation_id="get_business_attribute_values_by_category_and_city", summary="Retrieves the attribute values of businesses that belong to a specified category and are located in a given city. The category is identified by its name, and the city is determined by its name as well. This operation is useful for obtaining specific business details based on their category and location.")
async def get_business_attribute_values_by_category_and_city(category_name: str = Query(..., description="Category name of the business"), city: str = Query(..., description="City of the business")):
    cursor.execute("SELECT T2.attribute_value FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Business_Categories AS T3 ON T1.business_id = T3.business_id INNER JOIN Categories AS T4 ON T3.category_id = T4.category_id WHERE T4.category_name LIKE ? AND T1.city LIKE ?", (category_name, city))
    result = cursor.fetchall()
    if not result:
        return {"attribute_values": []}
    return {"attribute_values": [item[0] for item in result]}

# Endpoint to get the count of compliments based on city and number of compliments
@app.get("/v1/public_review_platform/compliment_count_by_city_and_compliment_level", operation_id="get_compliment_count_by_city_and_compliment_level", summary="Retrieves the total count of compliments for a specific city, filtered by the level of compliments. The level of compliments is determined by the number of compliments received. This operation provides insights into the distribution of compliments across different cities and levels.")
async def get_compliment_count_by_city_and_compliment_level(city: str = Query(..., description="City of the business"), compliment_level: str = Query(..., description="Level of compliments")):
    cursor.execute("SELECT COUNT(T1.number_of_compliments) FROM Users_Compliments AS T1 INNER JOIN Reviews AS T2 ON T1.user_id = T2.user_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T3.city LIKE ? AND T1.number_of_compliments LIKE ?", (city, compliment_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get category names of businesses in a specific city with opening time before a certain time
@app.get("/v1/public_review_platform/category_names_by_city_opening_time", operation_id="get_category_names_by_city_opening_time", summary="Retrieves the names of business categories in a specified city that open before a certain time. The operation filters businesses based on the provided city name and opening time, and returns the corresponding category names.")
async def get_category_names_by_city_opening_time(city: str = Query(..., description="City name"), opening_time: str = Query(..., description="Opening time in 'HHAM' format")):
    cursor.execute("SELECT T1.category_name FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Business_Hours AS T4 ON T3.business_id = T4.business_id WHERE T3.city LIKE ? AND T4.opening_time < ?", (city, opening_time))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the count of category names of businesses in a specific city with specific opening and closing times
@app.get("/v1/public_review_platform/count_category_names_by_city_opening_closing_time", operation_id="get_count_category_names_by_city_opening_closing_time", summary="Retrieve the count of distinct business categories in a given city that operate within specific opening and closing hours. The operation filters businesses by their city, opening time, and closing time, and then calculates the count of unique categories associated with these businesses.")
async def get_count_category_names_by_city_opening_closing_time(city: str = Query(..., description="City name"), opening_time: str = Query(..., description="Opening time in 'HHAM' format"), closing_time: str = Query(..., description="Closing time in 'HHAM' format")):
    cursor.execute("SELECT COUNT(T1.category_name) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Business_Hours AS T4 ON T3.business_id = T4.business_id WHERE T3.city LIKE ? AND T4.opening_time LIKE ? AND T4.closing_time LIKE ?", (city, opening_time, closing_time))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count and percentage of businesses in a specific city
@app.get("/v1/public_review_platform/count_percentage_businesses_by_city", operation_id="get_count_percentage_businesses_by_city", summary="Retrieves the total count and percentage of businesses located in a specified city. This operation calculates the count and percentage based on the provided city name, considering all businesses across different categories.")
async def get_count_percentage_businesses_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT SUM(CASE WHEN T3.city LIKE ? THEN 1 ELSE 0 END) AS \"num\" , CAST(SUM(CASE WHEN T3.city LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.city) FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id", (city, city))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "percentage": []}
    return {"count": result[0], "percentage": result[1]}

# Endpoint to get the count and percentage of businesses in a specific category and city with a specific review count
@app.get("/v1/public_review_platform/count_percentage_businesses_by_category_city_review_count", operation_id="get_count_percentage_businesses_by_category_city_review_count", summary="Retrieves the count and percentage of businesses in a given category and city that have a specific number of reviews. This operation provides insights into the distribution of businesses based on their review count within a particular category and city.")
async def get_count_percentage_businesses_by_category_city_review_count(category_name: str = Query(..., description="Category name"), city: str = Query(..., description="City name"), review_count: str = Query(..., description="Review count")):
    cursor.execute("SELECT SUM(CASE WHEN T2.category_name LIKE ? THEN 1 ELSE 0 END) AS \"num\" , CAST(SUM(CASE WHEN T3.city LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / ( SELECT COUNT(T3.review_count) FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id WHERE T3.review_count LIKE ? ) FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id", (category_name, city, review_count))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "percentage": []}
    return {"count": result[0], "percentage": result[1]}

# Endpoint to get the count of businesses in a specific city with stars greater than a certain value
@app.get("/v1/public_review_platform/count_businesses_by_city_stars", operation_id="get_count_businesses_by_city_stars", summary="Retrieves the number of businesses in a specified city that have a star rating exceeding a given value. This operation is useful for obtaining a quantitative measure of businesses in a particular location that meet a certain quality standard.")
async def get_count_businesses_by_city_stars(city: str = Query(..., description="City name"), stars: int = Query(..., description="Minimum star rating")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE city LIKE ? AND stars > ?", (city, stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average star rating of businesses with a specific active status
@app.get("/v1/public_review_platform/average_stars_by_active_status", operation_id="get_average_stars_by_active_status", summary="Retrieves the average star rating of businesses that are currently active or inactive, depending on the provided active status. This operation calculates the average star rating by summing up the total stars received by businesses with the specified active status and then dividing it by the total number of businesses with that active status.")
async def get_average_stars_by_active_status(active: str = Query(..., description="Active status")):
    cursor.execute("SELECT CAST(SUM(stars) AS REAL) / COUNT(business_id) AS \"average\" FROM Business WHERE active LIKE ?", (active,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the count of businesses with a specific attribute value and state
@app.get("/v1/public_review_platform/count_businesses_by_attribute_state", operation_id="get_count_businesses_by_attribute_state", summary="Retrieves the total number of businesses that possess a certain attribute value and are located in a specific state. The attribute value and state are provided as input parameters.")
async def get_count_businesses_by_attribute_state(attribute_value: str = Query(..., description="Attribute value"), state: str = Query(..., description="State")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id WHERE T2.attribute_value LIKE ? AND T1.state LIKE ?", (attribute_value, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get cities of businesses with a specific attribute value
@app.get("/v1/public_review_platform/cities_by_attribute_value", operation_id="get_cities_by_attribute_value", summary="Retrieves a list of cities where businesses possess a specified attribute value. The attribute value is provided as an input parameter.")
async def get_cities_by_attribute_value(attribute_value: str = Query(..., description="Attribute value")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id WHERE T2.attribute_value LIKE ? GROUP BY T1.city", (attribute_value,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of businesses with a specific star rating and category name
@app.get("/v1/public_review_platform/count_businesses_by_stars_category", operation_id="get_count_businesses_by_stars_category", summary="Retrieves the number of businesses that have a specified star rating and belong to a given category. The star rating is a measure of customer satisfaction, and the category indicates the type of business. This operation is useful for understanding the distribution of businesses across different categories and star ratings.")
async def get_count_businesses_by_stars_category(stars: int = Query(..., description="Star rating"), category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T1.stars = ? AND T3.category_name LIKE ?", (stars, category_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get cities of businesses with a specific review count and category name
@app.get("/v1/public_review_platform/cities_by_review_count_category", operation_id="get_cities_by_review_count_category", summary="Retrieve a list of cities where businesses with a specified review count and category name are located. The review count and category name are used as filters to narrow down the results. This operation does not return any business-specific details, only the city names.")
async def get_cities_by_review_count_category(review_count: str = Query(..., description="Review count of the business"), category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T1.review_count LIKE ? AND T3.category_name LIKE ? GROUP BY T1.city", (review_count, category_name))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get business IDs in a specific city with review stars above a certain threshold
@app.get("/v1/public_review_platform/business_ids_by_city_review_stars", operation_id="get_business_ids_by_city_review_stars", summary="Retrieve the IDs of businesses located in a specified city that have received review stars above a certain threshold. The operation filters businesses based on the provided city and minimum review star rating, returning a list of unique business IDs that meet the criteria.")
async def get_business_ids_by_city_review_stars(city: str = Query(..., description="City of the business"), review_stars: int = Query(..., description="Minimum review stars")):
    cursor.execute("SELECT T1.business_id FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city LIKE ? AND T2.review_stars > ? GROUP BY T1.business_id", (city, review_stars))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get cities of businesses with a specific number of funny review votes
@app.get("/v1/public_review_platform/cities_by_funny_review_votes", operation_id="get_cities_by_funny_review_votes", summary="Retrieves a list of cities where businesses have received a specified number of funny review votes. The response is grouped by city.")
async def get_cities_by_funny_review_votes(review_votes_funny: str = Query(..., description="Number of funny review votes")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T2.review_votes_funny LIKE ? GROUP BY T1.city", (review_votes_funny,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the percentage of 5-star businesses in a specific city and category
@app.get("/v1/public_review_platform/percentage_5_star_businesses", operation_id="get_percentage_5_star_businesses", summary="Retrieves the percentage of businesses with a 5-star rating within a specific city and category. The calculation is based on the total number of businesses in the given city and category. The city and category are provided as input parameters.")
async def get_percentage_5_star_businesses(city: str = Query(..., description="City of the business"), category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.stars = 5 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.business_id) AS 'percentage' FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T1.city LIKE ? AND T3.category_name LIKE ?", (city, category_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of users based on yelping since year and funny votes
@app.get("/v1/public_review_platform/user_count_by_yelping_year_funny_votes", operation_id="get_user_count_by_yelping_year_funny_votes", summary="Retrieves the total number of users who started yelping in a specific year and have received a certain amount of funny votes. The response is based on the provided year of yelping initiation and the number of funny votes.")
async def get_user_count_by_yelping_year_funny_votes(user_yelping_since_year: int = Query(..., description="Year the user started yelping"), user_votes_funny: str = Query(..., description="Number of funny votes")):
    cursor.execute("SELECT COUNT(user_id) FROM Users WHERE user_yelping_since_year = ? AND user_votes_funny LIKE ?", (user_yelping_since_year, user_votes_funny))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the useful review votes for a specific user and business
@app.get("/v1/public_review_platform/useful_review_votes", operation_id="get_useful_review_votes", summary="Retrieves the number of useful votes for a specific review left by a user for a business. The operation requires the user's unique identifier and the business's unique identifier as input parameters.")
async def get_useful_review_votes(user_id: int = Query(..., description="User ID"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT review_votes_useful FROM Reviews WHERE user_id = ? AND business_id = ?", (user_id, business_id))
    result = cursor.fetchone()
    if not result:
        return {"review_votes_useful": []}
    return {"review_votes_useful": result[0]}

# Endpoint to get attribute IDs based on attribute name
@app.get("/v1/public_review_platform/attribute_ids_by_name", operation_id="get_attribute_ids_by_name", summary="Retrieves the unique identifiers of attributes that match the provided name. The name can be a partial match, allowing for flexible search functionality.")
async def get_attribute_ids_by_name(attribute_name: str = Query(..., description="Attribute name")):
    cursor.execute("SELECT attribute_id FROM Attributes WHERE attribute_name LIKE ?", (attribute_name,))
    result = cursor.fetchall()
    if not result:
        return {"attribute_ids": []}
    return {"attribute_ids": [row[0] for row in result]}

# Endpoint to get the review length for a specific user, review stars, and business
@app.get("/v1/public_review_platform/review_length", operation_id="get_review_length", summary="Retrieves the length of a review written by a specific user for a particular business, based on the review's star rating. The user, business, and review star rating are identified using their respective unique IDs.")
async def get_review_length(user_id: int = Query(..., description="User ID"), review_stars: int = Query(..., description="Review stars"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT review_length FROM Reviews WHERE user_id = ? AND review_stars = ? AND business_id = ?", (user_id, review_stars, business_id))
    result = cursor.fetchone()
    if not result:
        return {"review_length": []}
    return {"review_length": result[0]}

# Endpoint to get the count of businesses in a specific state and review count
@app.get("/v1/public_review_platform/business_count_by_state_review_count", operation_id="get_business_count_by_state_review_count", summary="Retrieves the total number of businesses located in a specified state that have a particular review count. The state and review count are provided as input parameters.")
async def get_business_count_by_state_review_count(state: str = Query(..., description="State of the business"), review_count: str = Query(..., description="Review count of the business")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE state LIKE ? AND review_count LIKE ?", (state, review_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct business IDs based on state and review stars
@app.get("/v1/public_review_platform/distinct_business_ids", operation_id="get_distinct_business_ids", summary="Retrieve a unique set of business identifiers based on the specified state and review star rating. The operation limits the number of results returned to the value provided. This endpoint is useful for obtaining a diverse selection of businesses in a particular state that have received a specific review rating.")
async def get_distinct_business_ids(state: str = Query(..., description="State of the business"), review_stars: int = Query(..., description="Review stars"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T2.business_id FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T2.state LIKE ? AND T1.review_stars = ? LIMIT ?", (state, review_stars, limit))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get attribute names based on attribute value
@app.get("/v1/public_review_platform/attribute_names", operation_id="get_attribute_names", summary="Retrieves a list of attribute names that match a specified attribute value. The number of results returned can be limited by providing a value for the limit parameter.")
async def get_attribute_names(attribute_value: str = Query(..., description="Attribute value"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.attribute_name FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T2.attribute_value LIKE ? LIMIT ?", (attribute_value, limit))
    result = cursor.fetchall()
    if not result:
        return {"attribute_names": []}
    return {"attribute_names": [row[0] for row in result]}

# Endpoint to get the count of compliment types based on user ID and compliment type
@app.get("/v1/public_review_platform/compliment_type_count", operation_id="get_compliment_type_count", summary="Retrieves the total count of a specific compliment type associated with a given user. The user is identified by their unique user ID, and the compliment type is specified using a search pattern. This operation is useful for tracking the frequency of certain compliments received by a user.")
async def get_compliment_type_count(user_id: int = Query(..., description="User ID"), compliment_type: str = Query(..., description="Compliment type")):
    cursor.execute("SELECT COUNT(T2.compliment_type) FROM Users_Compliments AS T1 INNER JOIN Compliments AS T2 ON T1.compliment_id = T2.compliment_id WHERE T1.user_id = ? AND T2.compliment_type LIKE ?", (user_id, compliment_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get opening hours based on day of the week and business ID
@app.get("/v1/public_review_platform/opening_hours", operation_id="get_opening_hours", summary="Retrieves the total opening hours for a specific business on a given day of the week. The operation calculates the difference between the closing and opening times, providing a single duration value that represents the total hours the business is open on the specified day.")
async def get_opening_hours(day_of_week: str = Query(..., description="Day of the week"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT T1.closing_time - T1.opening_time AS 'opening hours' FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id WHERE T2.day_of_week LIKE ? AND T1.business_id = ?", (day_of_week, business_id))
    result = cursor.fetchall()
    if not result:
        return {"opening_hours": []}
    return {"opening_hours": [row[0] for row in result]}

# Endpoint to get attribute names based on attribute value and business ID
@app.get("/v1/public_review_platform/attribute_names_by_value_and_business", operation_id="get_attribute_names_by_value_and_business", summary="Retrieves the names of attributes that match a specified value for a given business. The operation filters attributes based on the provided attribute value and business ID, returning the names of the matching attributes.")
async def get_attribute_names_by_value_and_business(attribute_value: str = Query(..., description="Attribute value"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT T1.attribute_name FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T2.attribute_value LIKE ? AND T2.business_id = ?", (attribute_value, business_id))
    result = cursor.fetchall()
    if not result:
        return {"attribute_names": []}
    return {"attribute_names": [row[0] for row in result]}

# Endpoint to get category names based on business ID
@app.get("/v1/public_review_platform/category_names_by_business", operation_id="get_category_names_by_business", summary="Retrieves the names of categories associated with a specific business. The operation uses the provided business ID to look up the corresponding categories and returns their names.")
async def get_category_names_by_business(business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT T2.category_name FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id WHERE T1.business_id = ?", (business_id,))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the count of businesses based on city and category name
@app.get("/v1/public_review_platform/business_count_by_city_and_category", operation_id="get_business_count_by_city_and_category", summary="Retrieves the total number of businesses in a specific city that belong to a certain category. The operation requires the city and category name as input parameters to filter the results.")
async def get_business_count_by_city_and_category(city: str = Query(..., description="City"), category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T3.city LIKE ? AND T1.category_name LIKE ?", (city, category_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user IDs based on number of compliments and compliment type
@app.get("/v1/public_review_platform/user_ids_by_compliments", operation_id="get_user_ids_by_compliments", summary="Retrieves a list of user IDs who have received a specified number of compliments of a certain type. The number of results can be limited by the user.")
async def get_user_ids_by_compliments(number_of_compliments: str = Query(..., description="Number of compliments"), compliment_type: str = Query(..., description="Compliment type"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.user_id FROM Users_Compliments AS T1 INNER JOIN Compliments AS T2 ON T1.compliment_id = T2.compliment_id WHERE T1.number_of_compliments LIKE ? AND T2.compliment_type LIKE ? LIMIT ?", (number_of_compliments, compliment_type, limit))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the count of businesses based on star rating and category name
@app.get("/v1/public_review_platform/business_count_by_stars_and_category", operation_id="get_business_count_by_stars_and_category", summary="Retrieves the total number of businesses that have a star rating higher than the provided value and belong to a specified category. The category is identified by its name.")
async def get_business_count_by_stars_and_category(stars: int = Query(..., description="Star rating"), category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T3.stars > ? AND T1.category_name LIKE ?", (stars, category_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get opening hours based on business ID, city, and day ID
@app.get("/v1/public_review_platform/opening_hours_by_business_city_day", operation_id="get_opening_hours_by_business_city_day", summary="Retrieves the total operating hours for a specific business in a given city on a particular day. The business is identified by its unique ID, and the city is matched by name. The day is specified using its unique ID. The result is the difference between the closing and opening times.")
async def get_opening_hours_by_business_city_day(business_id: int = Query(..., description="Business ID"), city: str = Query(..., description="City"), day_id: int = Query(..., description="Day ID")):
    cursor.execute("SELECT T2.closing_time - T2.opening_time AS 'hour' FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id WHERE T1.business_id = ? AND T1.city LIKE ? AND T2.day_id = ?", (business_id, city, day_id))
    result = cursor.fetchall()
    if not result:
        return {"opening_hours": []}
    return {"opening_hours": [row[0] for row in result]}

# Endpoint to get the count of businesses based on check-in time label, state, and day of the week
@app.get("/v1/public_review_platform/business_count_checkin_time_state_day", operation_id="get_business_count_checkin_time_state_day", summary="Retrieves the total number of businesses that have a specific check-in time label, are located in a certain state, and have check-ins on a particular day of the week.")
async def get_business_count_checkin_time_state_day(label_time_4: str = Query(..., description="Check-in time label"), state: str = Query(..., description="State"), day_of_week: str = Query(..., description="Day of the week")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Checkins AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T2.label_time_4 LIKE ? AND T1.state LIKE ? AND T3.day_of_week LIKE ?", (label_time_4, state, day_of_week))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses in a specific city
@app.get("/v1/public_review_platform/business_count_by_city", operation_id="get_business_count_by_city", summary="Retrieves the total number of businesses located in a specified city. The city is identified by its name.")
async def get_business_count_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE city LIKE ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city with the highest review count
@app.get("/v1/public_review_platform/top_reviewed_city", operation_id="get_top_reviewed_city", summary="Retrieves the city with the highest number of reviews. This operation returns the name of the city that has been reviewed the most times, based on the review count data in the Business table.")
async def get_top_reviewed_city():
    cursor.execute("SELECT city FROM Business ORDER BY review_count DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the count of businesses in a specific state with stars greater than a specified value
@app.get("/v1/public_review_platform/business_count_by_state_and_stars", operation_id="get_business_count_by_state_and_stars", summary="Retrieves the number of businesses in a given state that have a star rating above a specified threshold. The state is identified by a string pattern, and the minimum star rating is a numerical value.")
async def get_business_count_by_state_and_stars(state: str = Query(..., description="State"), stars: int = Query(..., description="Minimum star rating")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE state LIKE ? AND stars > ?", (state, stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses in a specific state
@app.get("/v1/public_review_platform/business_count_by_state", operation_id="get_business_count_by_state", summary="Retrieves the total number of businesses located in a specified state. The state is provided as an input parameter.")
async def get_business_count_by_state(state: str = Query(..., description="State")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE state LIKE ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get cities with businesses having a specific star rating
@app.get("/v1/public_review_platform/cities_by_star_rating", operation_id="get_cities_by_star_rating", summary="Retrieves a list of cities that have businesses with a specified star rating. The star rating is a measure of the overall quality of a business, as rated by customers. This operation groups the results by city, providing a unique list of cities that meet the specified star rating criteria.")
async def get_cities_by_star_rating(stars: int = Query(..., description="Star rating")):
    cursor.execute("SELECT city FROM Business WHERE stars = ? GROUP BY city", (stars,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of reviews by a specific user with a specific review length and useful votes
@app.get("/v1/public_review_platform/review_count_by_user_length_and_useful_votes", operation_id="get_review_count_by_user_length_and_useful_votes", summary="Retrieves the total number of reviews written by a specific user that match a certain length and have received a certain number of useful votes. The user is identified by their unique ID, the review length is specified as a pattern, and the number of useful votes is also provided as a pattern.")
async def get_review_count_by_user_length_and_useful_votes(user_id: int = Query(..., description="User ID"), review_length: str = Query(..., description="Review length"), review_votes_useful: str = Query(..., description="Useful votes")):
    cursor.execute("SELECT COUNT(review_length) FROM Reviews WHERE user_id = ? AND review_length LIKE ? AND review_votes_useful LIKE ?", (user_id, review_length, review_votes_useful))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user IDs based on the number of fans
@app.get("/v1/public_review_platform/user_ids_by_fans", operation_id="get_user_ids_by_fans", summary="Retrieves a list of unique user IDs from the Users table, grouped by the number of fans each user has. The user_fans parameter is used to filter the results based on the number of fans a user has.")
async def get_user_ids_by_fans(user_fans: str = Query(..., description="Number of fans the user has")):
    cursor.execute("SELECT user_id FROM Users WHERE user_fans LIKE ? GROUP BY user_id", (user_fans,))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the count of attributes based on attribute name and value
@app.get("/v1/public_review_platform/attribute_count_by_name_and_value", operation_id="get_attribute_count_by_name_and_value", summary="Retrieves the total count of a specific attribute based on its name and value. This operation allows you to determine the frequency of a particular attribute value within a given attribute category. The attribute name and value are used as filters to calculate the count.")
async def get_attribute_count_by_name_and_value(attribute_name: str = Query(..., description="Name of the attribute"), attribute_value: str = Query(..., description="Value of the attribute")):
    cursor.execute("SELECT COUNT(T1.attribute_id) FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T1.attribute_name LIKE ? AND T2.attribute_value LIKE ?", (attribute_name, attribute_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses based on attribute name, value, and state
@app.get("/v1/public_review_platform/business_count_by_attribute_and_state", operation_id="get_business_count_by_attribute_and_state", summary="Retrieves the total number of businesses that match the specified attribute name, attribute value, and state. This operation is useful for obtaining a quantitative overview of businesses based on their attributes and location.")
async def get_business_count_by_attribute_and_state(attribute_name: str = Query(..., description="Name of the attribute"), attribute_value: str = Query(..., description="Value of the attribute"), state: str = Query(..., description="State of the business")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.attribute_name LIKE ? AND T2.attribute_value LIKE ? AND T3.state LIKE ?", (attribute_name, attribute_value, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs based on attribute name and value
@app.get("/v1/public_review_platform/business_ids_by_attribute", operation_id="get_business_ids_by_attribute", summary="Retrieves a list of business IDs that match the specified attribute name and value. The attribute name and value are used to filter the results, returning only those businesses that have the specified attribute and value.")
async def get_business_ids_by_attribute(attribute_name: str = Query(..., description="Name of the attribute"), attribute_value: str = Query(..., description="Value of the attribute")):
    cursor.execute("SELECT T2.business_id FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T1.attribute_name LIKE ? AND T2.attribute_value LIKE ?", (attribute_name, attribute_value))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get category names based on business ID
@app.get("/v1/public_review_platform/category_names_by_business_id", operation_id="get_category_names_by_business_id", summary="Retrieves the names of all categories associated with a specific business. The operation uses the provided business ID to look up the corresponding categories and returns their names.")
async def get_category_names_by_business_id(business_id: int = Query(..., description="ID of the business")):
    cursor.execute("SELECT T1.category_name FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id WHERE T2.business_id = ?", (business_id,))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the count of active businesses based on category name
@app.get("/v1/public_review_platform/active_business_count_by_category", operation_id="get_active_business_count_by_category", summary="Retrieves the count of active businesses associated with a specified category. The category is identified by its name, and the active status of the businesses is considered in the count. This operation provides a quantitative overview of active businesses within a particular category.")
async def get_active_business_count_by_category(category_name: str = Query(..., description="Name of the category"), active: str = Query(..., description="Active status of the business")):
    cursor.execute("SELECT COUNT(T3.business_id) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Tips AS T4 ON T3.business_id = T4.business_id WHERE T1.category_name LIKE ? AND T3.active LIKE ?", (category_name, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top business ID based on category name and star rating
@app.get("/v1/public_review_platform/top_business_by_category", operation_id="get_top_business_by_category", summary="Retrieves the ID of the top-rated business in a specified category. The category is identified by its name, and the business is selected based on its star rating, with the highest-rated business returned.")
async def get_top_business_by_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT T2.business_id FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.category_name LIKE ? ORDER BY T3.stars DESC LIMIT 1", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"business_id": []}
    return {"business_id": result[0]}

# Endpoint to get the count of stars for businesses in a specific category with a specific attribute
@app.get("/v1/public_review_platform/count_stars_category_attribute", operation_id="get_count_stars_category_attribute", summary="Retrieves the total number of stars received by businesses in a specified category that possess a particular attribute. The category and attribute are identified by their respective names, while the attribute value is used to filter the results.")
async def get_count_stars_category_attribute(category_name: str = Query(..., description="Category name"), attribute_name: str = Query(..., description="Attribute name"), attribute_value: str = Query(..., description="Attribute value")):
    cursor.execute("SELECT COUNT(T3.stars) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Business_Attributes AS T4 ON T3.business_id = T4.business_id INNER JOIN Attributes AS T5 ON T4.attribute_id = T5.attribute_id WHERE T1.category_name LIKE ? AND T5.attribute_name LIKE ? AND T4.attribute_value LIKE ?", (category_name, attribute_name, attribute_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses reviewed by a specific user in a specific state
@app.get("/v1/public_review_platform/count_businesses_reviewed_by_user_state", operation_id="get_count_businesses_reviewed_by_user_state", summary="Retrieves the total number of businesses that a particular user has reviewed in a specified state. The operation requires the user's ID and the state's name as input parameters.")
async def get_count_businesses_reviewed_by_user_state(state: str = Query(..., description="State"), user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T2.state LIKE ? AND T1.user_id = ?", (state, user_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the category names of businesses in a specific state
@app.get("/v1/public_review_platform/category_names_by_state", operation_id="get_category_names_by_state", summary="Retrieve a list of unique category names for businesses located in a specified state. The state parameter is used to filter the results.")
async def get_category_names_by_state(state: str = Query(..., description="State")):
    cursor.execute("SELECT T1.category_name FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T3.state LIKE ? GROUP BY T1.category_name", (state,))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the opening time of a business on a specific day
@app.get("/v1/public_review_platform/opening_time_by_day", operation_id="get_opening_time_by_day", summary="Retrieves the opening time of a specific business on a given day of the week. The operation requires the business ID and the day of the week as input parameters.")
async def get_opening_time_by_day(day_of_week: str = Query(..., description="Day of the week"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT T1.opening_time FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id WHERE T2.day_of_week LIKE ? AND T1.business_id = ?", (day_of_week, business_id))
    result = cursor.fetchone()
    if not result:
        return {"opening_time": []}
    return {"opening_time": result[0]}

# Endpoint to get the count of businesses open after a specific time on a specific day
@app.get("/v1/public_review_platform/count_businesses_open_after_time", operation_id="get_count_businesses_open_after_time", summary="Retrieves the total number of businesses that remain open after a specified closing time on a particular day of the week. The day of the week and the closing time are provided as input parameters.")
async def get_count_businesses_open_after_time(day_of_week: str = Query(..., description="Day of the week"), closing_time: str = Query(..., description="Closing time in 'HH:MM AM/PM' format")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id WHERE T2.day_of_week LIKE ? AND T1.closing_time > ?", (day_of_week, closing_time))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the opening times of active businesses in a specific city on a specific day
@app.get("/v1/public_review_platform/opening_times_active_businesses", operation_id="get_opening_times_active_businesses", summary="Retrieve the opening times of businesses that are currently active in a specified city for a given day of the week. The operation filters businesses based on their active status and location, and returns a list of their opening times.")
async def get_opening_times_active_businesses(day_of_week: str = Query(..., description="Day of the week"), city: str = Query(..., description="City"), active: str = Query(..., description="Active status")):
    cursor.execute("SELECT T1.opening_time FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id WHERE T2.day_of_week LIKE ? AND T3.city LIKE ? AND T3.active LIKE ? GROUP BY T1.opening_time", (day_of_week, city, active))
    result = cursor.fetchall()
    if not result:
        return {"opening_times": []}
    return {"opening_times": [row[0] for row in result]}

# Endpoint to get the count of businesses in a specific state with a specific closing time on a specific day
@app.get("/v1/public_review_platform/count_businesses_state_closing_time", operation_id="get_count_businesses_state_closing_time", summary="Retrieves the total number of businesses in a specified state that close at a particular time on a given day of the week.")
async def get_count_businesses_state_closing_time(day_of_week: str = Query(..., description="Day of the week"), closing_time: str = Query(..., description="Closing time in 'HH:MM AM/PM' format"), state: str = Query(..., description="State")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id WHERE T2.day_of_week LIKE ? AND T1.closing_time LIKE ? AND T3.state LIKE ?", (day_of_week, closing_time, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the category names of businesses with a specific closing time on a specific day
@app.get("/v1/public_review_platform/category_names_closing_time_day", operation_id="get_category_names_closing_time_day", summary="Retrieves the names of business categories that close at a specific time on a given day of the week. The operation filters businesses based on their closing time and day, then groups the results by category name.")
async def get_category_names_closing_time_day(closing_time: str = Query(..., description="Closing time in 'HH:MM AM/PM' format"), day_of_week: str = Query(..., description="Day of the week")):
    cursor.execute("SELECT T4.category_name FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id INNER JOIN Business_Categories AS T3 ON T1.business_id = T3.business_id INNER JOIN Categories AS T4 ON T4.category_id = T4.category_id WHERE T1.closing_time = ? AND T2.day_of_week = ? GROUP BY T4.category_name", (closing_time, day_of_week))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the count of businesses with a specific attribute on specific days
@app.get("/v1/public_review_platform/count_businesses_attribute_days", operation_id="get_count_businesses_attribute_days", summary="Retrieves the total number of businesses that possess a specified attribute on selected days. The operation requires the attribute name, its corresponding value, and a list of day IDs to filter the count.")
async def get_count_businesses_attribute_days(day_ids: str = Query(..., description="Comma-separated list of day IDs"), attribute_name: str = Query(..., description="Attribute name"), attribute_value: str = Query(..., description="Attribute value")):
    day_ids_list = [int(day_id) for day_id in day_ids.split(',')]
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id INNER JOIN Business_Attributes AS T3 ON T1.business_id = T3.business_id INNER JOIN Attributes AS T4 ON T4.attribute_id = T4.attribute_id WHERE T2.day_id IN (?, ?, ?, ?, ?, ?, ?) AND T4.attribute_name = ? AND T3.attribute_value = ?", (*day_ids_list, attribute_name, attribute_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users who started yelping in a specific year
@app.get("/v1/public_review_platform/user_count_yelping_since_year", operation_id="get_user_count_yelping_since_year", summary="Retrieves the total number of users who have started yelping in a specific year. This operation calculates the count based on the user's yelping start year and their elite status, which is determined by matching user IDs in the Users and Elite tables.")
async def get_user_count_yelping_since_year():
    cursor.execute("SELECT COUNT(T1.user_id) FROM Users AS T1 INNER JOIN Elite AS T2 ON T1.user_id = T2.user_id WHERE T1.user_yelping_since_year = T2.year_id")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the business with the longest operating hours on a specific day and category
@app.get("/v1/public_review_platform/longest_operating_hours", operation_id="get_longest_operating_hours", summary="Retrieves the business with the longest operating hours for a given day of the week and category. The operation calculates the total operating hours for each business in the specified category and day, then returns the business with the highest total. The day of the week and category are provided as input parameters.")
async def get_longest_operating_hours(day_of_week: str = Query(..., description="Day of the week (e.g., 'Monday')"), category_name: str = Query(..., description="Category name (e.g., 'Shopping')")):
    cursor.execute("SELECT T1.closing_time + 12 - T1.opening_time AS 'hour' FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id INNER JOIN Business_Categories AS T4 ON T3.business_id = T4.business_id INNER JOIN Categories AS T5 ON T4.category_id = T5.category_id WHERE T2.day_of_week LIKE ? AND T5.category_name LIKE ? ORDER BY T1.closing_time + 12 - T1.opening_time DESC LIMIT 1", (day_of_week, category_name))
    result = cursor.fetchone()
    if not result:
        return {"hour": []}
    return {"hour": result[0]}

# Endpoint to get businesses open for more than a specified number of hours on a specific day
@app.get("/v1/public_review_platform/businesses_open_longer_than", operation_id="get_businesses_open_longer_than", summary="Retrieves businesses that are open for more than the specified number of hours on a particular day of the week. The operation filters businesses based on their opening and closing times, and groups the results by business ID.")
async def get_businesses_open_longer_than(min_hours: int = Query(..., description="Minimum number of hours the business is open"), day_of_week: str = Query(..., description="Day of the week (e.g., 'Sunday')")):
    cursor.execute("SELECT T1.business_id FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id WHERE T1.closing_time + 12 - T1.opening_time > ? AND T2.day_of_week LIKE ? GROUP BY T1.business_id", (min_hours, day_of_week))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of elite users who reviewed a specific business
@app.get("/v1/public_review_platform/elite_user_count_for_business", operation_id="get_elite_user_count_for_business", summary="Retrieves the total number of elite users who have reviewed a specific business. The business is identified by its unique ID.")
async def get_elite_user_count_for_business(business_id: int = Query(..., description="ID of the business")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM Users AS T1 INNER JOIN Elite AS T2 ON T1.user_id = T2.user_id INNER JOIN Reviews AS T3 ON T1.user_id = T3.user_id WHERE T3.business_id = ?", (business_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of elite users who have reviewed more than a specified number of times
@app.get("/v1/public_review_platform/elite_user_count_reviews_more_than", operation_id="get_elite_user_count_reviews_more_than", summary="Retrieves the count of elite users who have left more than a specified number of reviews. The input parameter determines the minimum number of reviews required for a user to be included in the count.")
async def get_elite_user_count_reviews_more_than(min_reviews: int = Query(..., description="Minimum number of reviews")):
    cursor.execute("SELECT COUNT(T4.user_id) FROM ( SELECT T1.user_id FROM Users AS T1 INNER JOIN Elite AS T2 ON T1.user_id = T2.user_id INNER JOIN Reviews AS T3 ON T1.user_id = T3.user_id WHERE T3.user_id IS NOT NULL GROUP BY T3.user_id HAVING COUNT(T3.user_id) > ? ) T4", (min_reviews,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the user with the most reviews in a specific state
@app.get("/v1/public_review_platform/top_reviewer_in_state", operation_id="get_top_reviewer_in_state", summary="Retrieves the user who has written the most reviews in a specified state. The state is provided as an input parameter. The operation returns the user ID of the top reviewer.")
async def get_top_reviewer_in_state(state: str = Query(..., description="State (e.g., 'AZ')")):
    cursor.execute("SELECT T1.user_id FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T2.state LIKE ? GROUP BY T1.user_id ORDER BY COUNT(T1.user_id) DESC LIMIT 1", (state,))
    result = cursor.fetchone()
    if not result:
        return {"user_id": []}
    return {"user_id": result[0]}

# Endpoint to get the average review stars for businesses in a specific city
@app.get("/v1/public_review_platform/average_review_stars_city", operation_id="get_average_review_stars_city", summary="Retrieves the average review star rating for businesses located in a specified city. The city name is provided as an input parameter to filter the results.")
async def get_average_review_stars_city(city: str = Query(..., description="City name (e.g., 'Anthem')")):
    cursor.execute("SELECT AVG(T2.review_stars) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city LIKE ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"average_stars": []}
    return {"average_stars": result[0]}

# Endpoint to get the average review stars for a specific user in a specific state
@app.get("/v1/public_review_platform/average_review_stars_user_state", operation_id="get_average_review_stars_user_state", summary="Retrieves the average review rating for a particular user in a specified state. The calculation is based on the user's reviews for businesses located in the given state.")
async def get_average_review_stars_user_state(state: str = Query(..., description="State (e.g., 'AZ')"), user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT AVG(T2.review_stars) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.state LIKE ? AND T2.user_id = ?", (state, user_id))
    result = cursor.fetchone()
    if not result:
        return {"average_stars": []}
    return {"average_stars": result[0]}

# Endpoint to get the average opening hours for a specific business on specific days
@app.get("/v1/public_review_platform/avg_opening_hours_business_days", operation_id="get_avg_opening_hours_business_days", summary="Retrieves the average daily opening hours for a specific business on two selected weekdays. The calculation considers the opening and closing times of the business on those days.")
async def get_avg_opening_hours_business_days(business_id: int = Query(..., description="Business ID"), day_of_week_1: str = Query(..., description="First day of the week (e.g., 'Sunday')"), day_of_week_2: str = Query(..., description="Second day of the week (e.g., 'Sunday')")):
    cursor.execute("SELECT T1.closing_time + 12 - T1.opening_time AS 'avg opening hours' FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id WHERE T1.business_id = ? AND (T2.day_of_week = ? OR T2.day_of_week = ?)", (business_id, day_of_week_1, day_of_week_2))
    result = cursor.fetchone()
    if not result:
        return {"avg_opening_hours": []}
    return {"avg_opening_hours": result[0]}

# Endpoint to get the average stars for businesses open on a specific day and closing at a specific time
@app.get("/v1/public_review_platform/average_stars_day_closing_time", operation_id="get_average_stars_day_closing_time", summary="Retrieves the average star rating for businesses that operate on a specific day of the week and close at a particular time. The calculation is based on the sum of all star ratings divided by the total number of businesses that meet the specified criteria.")
async def get_average_stars_day_closing_time(day_of_week: str = Query(..., description="Day of the week (e.g., 'Sunday')"), closing_time: str = Query(..., description="Closing time (e.g., '12PM')")):
    cursor.execute("SELECT CAST(SUM(T3.stars) AS REAL) / COUNT(T1.business_id) AS 'average stars' FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id WHERE T2.day_of_week LIKE ? AND T1.closing_time LIKE ?", (day_of_week, closing_time))
    result = cursor.fetchone()
    if not result:
        return {"average_stars": []}
    return {"average_stars": result[0]}

# Endpoint to get the count of businesses based on state, active status, and review count
@app.get("/v1/public_review_platform/business_count_by_state_active_review", operation_id="get_business_count_by_state_active_review", summary="Retrieves the total number of businesses in a specified state that are currently active and have a certain number of reviews. The state, active status, and review count are provided as input parameters.")
async def get_business_count_by_state_active_review(state: str = Query(..., description="State name"), active: str = Query(..., description="Active status"), review_count: str = Query(..., description="Review count")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE state LIKE ? AND active LIKE ? AND review_count LIKE ?", (state, active, review_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs based on city and star rating range
@app.get("/v1/public_review_platform/business_ids_by_city_stars", operation_id="get_business_ids_by_city_stars", summary="Retrieves a list of business IDs located in a specified city and within a defined star rating range. The city name and the minimum and maximum star ratings are provided as input parameters to filter the results.")
async def get_business_ids_by_city_stars(city: str = Query(..., description="City name"), min_stars: int = Query(..., description="Minimum star rating"), max_stars: int = Query(..., description="Maximum star rating")):
    cursor.execute("SELECT business_id FROM Business WHERE city LIKE ? AND stars BETWEEN ? AND ?", (city, min_stars, max_stars))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of users based on yelping since year range and user fans
@app.get("/v1/public_review_platform/user_count_by_year_fans", operation_id="get_user_count_by_year_fans", summary="Retrieves the total number of users who have been yelping since a specified year range and have a certain number of fans. The year range is defined by the minimum and maximum years, and the number of fans is a specific value.")
async def get_user_count_by_year_fans(min_year: int = Query(..., description="Minimum yelping since year"), max_year: int = Query(..., description="Maximum yelping since year"), user_fans: str = Query(..., description="User fans")):
    cursor.execute("SELECT COUNT(user_id) FROM Users WHERE user_yelping_since_year BETWEEN ? AND ? AND user_fans LIKE ?", (min_year, max_year, user_fans))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get review length based on user ID and business ID
@app.get("/v1/public_review_platform/review_length_by_user_business", operation_id="get_review_length_by_user_business", summary="Retrieves the length of reviews written by a specific user for a particular business. The operation requires the user's unique identifier and the business's unique identifier as input parameters.")
async def get_review_length_by_user_business(user_id: int = Query(..., description="User ID"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT review_length FROM Reviews WHERE user_id = ? AND business_id = ?", (user_id, business_id))
    result = cursor.fetchone()
    if not result:
        return {"review_length": []}
    return {"review_length": result[0]}

# Endpoint to get distinct attributes based on review count and city
@app.get("/v1/public_review_platform/distinct_attributes_by_review_count_city", operation_id="get_distinct_attributes_by_review_count_city", summary="Retrieves a unique set of attributes for businesses based on a specified review count and city. This operation filters businesses by the given review count and city, then identifies the distinct attributes associated with these businesses.")
async def get_distinct_attributes_by_review_count_city(review_count: str = Query(..., description="Review count"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT DISTINCT T3.attribute_id, T3.attribute_name FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.attribute_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id WHERE T1.review_count = ? AND T1.city = ?", (review_count, city))
    result = cursor.fetchall()
    if not result:
        return {"attributes": []}
    return {"attributes": [{"attribute_id": row[0], "attribute_name": row[1]} for row in result]}

# Endpoint to get the count of businesses based on star rating and category name
@app.get("/v1/public_review_platform/business_count_by_stars_category", operation_id="get_business_count_by_stars_category", summary="Retrieves the number of businesses that have a star rating less than the specified maximum and belong to the provided category. The category is identified by its name.")
async def get_business_count_by_stars_category(max_stars: int = Query(..., description="Maximum star rating"), category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T1.stars < ? AND T3.category_name LIKE ?", (max_stars, category_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs and star ratings based on active status and category name
@app.get("/v1/public_review_platform/business_ids_stars_by_active_category", operation_id="get_business_ids_stars_by_active_category", summary="Retrieves the IDs and star ratings of businesses that are currently active and belong to a specified category. The active status and category name are used as filters to determine the relevant businesses.")
async def get_business_ids_stars_by_active_category(active: str = Query(..., description="Active status"), category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T1.business_id, T1.stars FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T1.active LIKE ? AND T3.category_name LIKE ?", (active, category_name))
    result = cursor.fetchall()
    if not result:
        return {"businesses": []}
    return {"businesses": [{"business_id": row[0], "stars": row[1]} for row in result]}

# Endpoint to get the top category name based on star ratings
@app.get("/v1/public_review_platform/top_category_by_stars", operation_id="get_top_category_by_stars", summary="Retrieves the name of the category with the highest average star rating. This operation considers all businesses and their respective categories, calculating the average star rating for each category. The category with the highest average rating is then returned.")
async def get_top_category_by_stars():
    cursor.execute("SELECT T3.category_name FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id ORDER BY T1.stars DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"category_name": []}
    return {"category_name": result[0]}

# Endpoint to get the top category name based on review length and business ID range
@app.get("/v1/public_review_platform/top_category_by_review_length_business_id", operation_id="get_top_category_by_review_length_business_id", summary="Retrieves the category name with the longest reviews for businesses within a specified ID range. The category is determined by the highest average review star rating for the given review length. The review length, minimum business ID, and maximum business ID are required as input parameters.")
async def get_top_category_by_review_length_business_id(review_length: str = Query(..., description="Review length"), min_business_id: int = Query(..., description="Minimum business ID"), max_business_id: int = Query(..., description="Maximum business ID")):
    cursor.execute("SELECT T4.category_name FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Business_Categories AS T3 ON T2.business_id = T3.business_id INNER JOIN Categories AS T4 ON T3.category_id = T4.category_id WHERE T1.review_length LIKE ? AND T2.business_id BETWEEN ? AND ? ORDER BY T1.review_stars DESC LIMIT 1", (review_length, min_business_id, max_business_id))
    result = cursor.fetchone()
    if not result:
        return {"category_name": []}
    return {"category_name": result[0]}

# Endpoint to get the count of businesses based on attribute name, review count, and active status
@app.get("/v1/public_review_platform/business_count_by_attributes", operation_id="get_business_count_by_attributes", summary="Retrieves the total number of businesses that match the specified attribute name, review count, and active status. The attribute name, review count, and active status are used as filters to determine the businesses included in the count.")
async def get_business_count_by_attributes(attribute_name: str = Query(..., description="Attribute name"), review_count: str = Query(..., description="Review count"), active: str = Query(..., description="Active status")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id WHERE T3.attribute_name LIKE ? AND T1.review_count LIKE ? AND T1.active LIKE ?", (attribute_name, review_count, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the closing and opening times of the top-rated business in a specific city
@app.get("/v1/public_review_platform/top_rated_business_hours", operation_id="get_top_rated_business_hours", summary="Retrieves the opening and closing times of the highest-rated business in a given city. The city is specified as a parameter, and the operation returns the business hours of the top-rated business in that city.")
async def get_top_rated_business_hours(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T2.closing_time, T2.opening_time FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id WHERE T1.city LIKE ? ORDER BY T1.stars DESC LIMIT 1", (city,))
    result = cursor.fetchone()
    if not result:
        return {"closing_time": [], "opening_time": []}
    return {"closing_time": result[0], "opening_time": result[1]}

# Endpoint to get category and attribute names of businesses based on active status, state, city, and review count
@app.get("/v1/public_review_platform/business_category_attribute_names", operation_id="get_business_category_attribute_names", summary="Retrieve the category and attribute names of businesses that meet the specified active status, state, city, and review count criteria. This operation filters businesses based on the provided parameters and returns the corresponding category and attribute names.")
async def get_business_category_attribute_names(active: str = Query(..., description="Active status"), state: str = Query(..., description="State"), city: str = Query(..., description="City"), review_count: str = Query(..., description="Review count")):
    cursor.execute("SELECT T3.category_name, T5.attribute_name FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id INNER JOIN Business_Attributes AS T4 ON T1.business_id = T4.business_id INNER JOIN Attributes AS T5 ON T4.attribute_id = T5.attribute_id WHERE T1.active LIKE ? AND T1.state LIKE ? AND T1.city LIKE ? AND T1.review_count LIKE ?", (active, state, city, review_count))
    result = cursor.fetchall()
    if not result:
        return {"category_attribute_names": []}
    return {"category_attribute_names": result}

# Endpoint to get category names of businesses based on active status, state, and city
@app.get("/v1/public_review_platform/business_category_names", operation_id="get_business_category_names", summary="Retrieves the names of business categories that meet the specified active status, state, and city criteria. The operation filters businesses based on their active status, state, and city, and then returns the distinct category names associated with these businesses.")
async def get_business_category_names(active: str = Query(..., description="Active status"), state: str = Query(..., description="State"), city: str = Query(..., description="City")):
    cursor.execute("SELECT T3.category_name FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T1.active LIKE ? AND T1.state LIKE ? AND T1.city LIKE ? GROUP BY T3.category_name", (active, state, city))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get cities of businesses based on closing time, opening time, and day of the week
@app.get("/v1/public_review_platform/business_cities_by_hours", operation_id="get_business_cities_by_hours", summary="Retrieves a list of cities where businesses operate within specific opening and closing times on a given day of the week. The operation filters businesses based on the provided opening and closing times, and the day of the week, then groups the results by city.")
async def get_business_cities_by_hours(closing_time: str = Query(..., description="Closing time"), opening_time: str = Query(..., description="Opening time"), day_of_week: str = Query(..., description="Day of the week")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T2.closing_time LIKE ? AND T2.opening_time LIKE ? AND T3.day_of_week LIKE ? GROUP BY T1.city", (closing_time, opening_time, day_of_week))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get distinct opening times of businesses based on city, active status, and review count
@app.get("/v1/public_review_platform/distinct_opening_times", operation_id="get_distinct_opening_times", summary="Retrieve the unique opening times of businesses in a specified city, filtered by their active status and review count. This operation provides a list of distinct opening times, enabling users to understand the variety of business hours in a given location.")
async def get_distinct_opening_times(city: str = Query(..., description="City"), active: str = Query(..., description="Active status"), review_count: str = Query(..., description="Review count")):
    cursor.execute("SELECT DISTINCT T2.opening_time FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city LIKE ? AND T1.active LIKE ? AND T1.review_count LIKE ?", (city, active, review_count))
    result = cursor.fetchall()
    if not result:
        return {"opening_times": []}
    return {"opening_times": [row[0] for row in result]}

# Endpoint to get the percentage of businesses with stars less than 4 in a specific category
@app.get("/v1/public_review_platform/percentage_low_stars_by_category", operation_id="get_percentage_low_stars_by_category", summary="Retrieves the percentage of businesses in a specified category that have a star rating less than 4. This operation calculates the proportion of businesses with lower ratings by comparing the count of businesses with a star rating below 4 to the total count of businesses in the given category.")
async def get_percentage_low_stars_by_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.stars < 4 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.stars) AS 'percentage' FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T3.category_name LIKE ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get closing times and days of the week for businesses with stars above 70% of the average in a specific city
@app.get("/v1/public_review_platform/high_star_business_hours", operation_id="get_high_star_business_hours", summary="Retrieves the closing times and days of the week for businesses in a specific city that have a star rating above 70% of the average. The businesses must also be active. The city and active status are provided as input parameters.")
async def get_high_star_business_hours(city: str = Query(..., description="City name"), active: str = Query(..., description="Active status")):
    cursor.execute("SELECT T2.closing_time, T3.day_of_week FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city LIKE ? AND T1.active LIKE ? AND T1.stars > 0.7 * ( SELECT AVG(T1.stars) FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city LIKE ? AND T1.active LIKE ? )", (city, active, city, active))
    result = cursor.fetchall()
    if not result:
        return {"business_hours": []}
    return {"business_hours": result}

# Endpoint to get the count of businesses based on review count
@app.get("/v1/public_review_platform/business_count_by_review_count", operation_id="get_business_count_by_review_count", summary="Retrieves the total number of businesses that have a specified review count. The review count is a parameter that filters the businesses to be counted.")
async def get_business_count_by_review_count(review_count: str = Query(..., description="Review count of the business")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE review_count LIKE ?", (review_count,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of attributes for a specific business
@app.get("/v1/public_review_platform/attribute_count_by_business", operation_id="get_attribute_count_by_business", summary="Retrieves the total count of attributes associated with a particular business, identified by its unique business ID.")
async def get_attribute_count_by_business(business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT COUNT(attribute_id) FROM Business_Attributes WHERE business_id = ?", (business_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users with specific compliments
@app.get("/v1/public_review_platform/user_count_by_compliments", operation_id="get_user_count_by_compliments", summary="Retrieves the count of users who have received a specific number of compliments and a particular compliment. The number of compliments and the compliment ID are used to filter the results.")
async def get_user_count_by_compliments(number_of_compliments: str = Query(..., description="Number of compliments"), compliment_id: int = Query(..., description="Compliment ID")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM Users_Compliments AS T1 INNER JOIN Compliments AS T2 ON T1.compliment_id = T2.compliment_id WHERE T1.number_of_compliments LIKE ? AND T2.compliment_id = ?", (number_of_compliments, compliment_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with specific attributes in a specific city
@app.get("/v1/public_review_platform/business_count_by_city_attributes", operation_id="get_business_count_by_city_attributes", summary="Retrieves the number of businesses in a specific city that possess a certain attribute, identified by its name and ID. This operation requires the city name, attribute name, and attribute ID as input parameters.")
async def get_business_count_by_city_attributes(city: str = Query(..., description="City name"), attribute_name: str = Query(..., description="Attribute name"), attribute_id: int = Query(..., description="Attribute ID")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Business_attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id WHERE T1.city LIKE ? AND T3.attribute_name LIKE ? AND T2.attribute_id = ?", (city, attribute_name, attribute_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get attribute names for businesses open 24/7 on specific days
@app.get("/v1/public_review_platform/attribute_names_24_7_businesses", operation_id="get_attribute_names_24_7_businesses", summary="Retrieve the distinct attribute names of businesses that operate 24/7 on specific days. The operation filters businesses based on their opening and closing times, and groups the results by attribute name. The input parameters determine the days of the week for which the attribute names are retrieved.")
async def get_attribute_names_24_7_businesses(day_id_1: int = Query(..., description="Day ID 1"), day_id_2: int = Query(..., description="Day ID 2"), day_id_3: int = Query(..., description="Day ID 3"), day_id_4: int = Query(..., description="Day ID 4"), day_id_5: int = Query(..., description="Day ID 5"), day_id_6: int = Query(..., description="Day ID 6"), day_id_7: int = Query(..., description="Day ID 7")):
    cursor.execute("SELECT T5.attribute_name FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id INNER JOIN Business_Attributes AS T4 ON T3.business_id = T4.business_id INNER JOIN Attributes AS T5 ON T4.attribute_id = T5.attribute_id WHERE T2.day_id IN (?, ?, ?, ?, ?, ?, ?) AND T1.opening_time = T1.closing_time GROUP BY T5.attribute_name", (day_id_1, day_id_2, day_id_3, day_id_4, day_id_5, day_id_6, day_id_7))
    result = cursor.fetchall()
    if not result:
        return {"attribute_names": []}
    return {"attribute_names": [row[0] for row in result]}

# Endpoint to get distinct category names for businesses in a specific state with a minimum review score
@app.get("/v1/public_review_platform/category_names_by_state_review_stars", operation_id="get_category_names_by_state_review_stars", summary="Retrieve a unique list of business category names from a specific state that have a minimum review score. The state and minimum review score are provided as input parameters.")
async def get_category_names_by_state_review_stars(state: str = Query(..., description="State name"), min_review_stars: int = Query(..., description="Minimum review stars")):
    cursor.execute("SELECT DISTINCT T4.category_name FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Business_Categories AS T3 ON T2.business_id = T3.business_id INNER JOIN Categories AS T4 ON T3.category_id = T4.category_id WHERE T2.state LIKE ? AND T1.review_stars >= ?", (state, min_review_stars))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the proportion of users with specific average stars in a given year
@app.get("/v1/public_review_platform/user_proportion_by_average_stars", operation_id="get_user_proportion_by_average_stars", summary="Retrieves the proportion of users who have a specific average star rating in a given year. The calculation is based on the total number of users who have a certain average star rating and the total number of users in the Elite group for the specified year. The input parameter 'year_id' is used to filter the data for a specific year.")
async def get_user_proportion_by_average_stars(year_id: int = Query(..., description="Year ID")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.user_average_stars = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(T2.user_id) , SUM(CASE WHEN T1.user_average_stars = 5 THEN 1 ELSE 0 END) * 1.0 / COUNT(T2.user_id) FROM Users AS T1 INNER JOIN Elite AS T2 ON T1.user_id = T2.user_id WHERE T2.year_id = ?", (year_id,))
    result = cursor.fetchone()
    if not result:
        return {"proportion_1_star": [], "proportion_5_star": []}
    return {"proportion_1_star": result[0], "proportion_5_star": result[1]}

# Endpoint to get the percentage increment of elite users from a specific year
@app.get("/v1/public_review_platform/elite_user_increment", operation_id="get_elite_user_increment", summary="Retrieves the percentage increase in the number of elite users from a specified year compared to a previous year. The calculation is based on the count of elite users in the specified year and the count of elite users in the year prior to the specified year.")
async def get_elite_user_increment(year_id_before: int = Query(..., description="Year ID before"), year_id_after: int = Query(..., description="Year ID after")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN year_id < ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(CASE WHEN year_id = ? THEN 1.0 ELSE NULL END) AS increment FROM Elite", (year_id_before, year_id_after))
    result = cursor.fetchone()
    if not result:
        return {"increment": []}
    return {"increment": result[0]}

# Endpoint to get the count of business IDs and the number of years a user has been yelping
@app.get("/v1/public_review_platform/business_count_and_yelping_years", operation_id="get_business_count_and_yelping_years", summary="Retrieves the total number of businesses reviewed by a user and the duration (in years) that the user has been active on the platform. The user is identified by the provided user_id.")
async def get_business_count_and_yelping_years(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT COUNT(T1.business_id) , strftime('%Y', 'now') - T2.user_yelping_since_year FROM Reviews AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id WHERE T1.user_id = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "years_yelping": []}
    return {"count": result[0], "years_yelping": result[1]}

# Endpoint to get the average review stars per year for a user
@app.get("/v1/public_review_platform/average_review_stars_per_year", operation_id="get_average_review_stars_per_year", summary="Retrieves the average number of review stars per year for a specific user. This operation calculates the average by dividing the total count of review stars by the number of years the user has been active on the platform, starting from the year they joined. The user is identified by their unique user ID.")
async def get_average_review_stars_per_year(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT COUNT(review_stars) / (strftime('%Y', 'now') - T1.user_yelping_since_year) FROM Users AS T1 INNER JOIN Reviews AS T2 ON T1.user_id = T2.user_id WHERE T1.user_id = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"average_stars_per_year": []}
    return {"average_stars_per_year": result[0]}

# Endpoint to get the ratio of user reviews to distinct businesses
@app.get("/v1/public_review_platform/user_review_to_business_ratio", operation_id="get_user_review_to_business_ratio", summary="Retrieves the average number of reviews per business, considering only users who are part of the Elite group. This operation calculates the ratio by dividing the total count of reviews by the number of distinct businesses reviewed by Elite users.")
async def get_user_review_to_business_ratio():
    cursor.execute("SELECT CAST(COUNT(T1.user_id) AS REAL) / COUNT(DISTINCT T1.business_id) FROM Reviews AS T1 INNER JOIN Elite AS T2 ON T1.user_id = T2.user_id")
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get user average stars and tip likes for users with more than a specified number of tips
@app.get("/v1/public_review_platform/user_average_stars_and_tip_likes", operation_id="get_user_average_stars_and_tip_likes", summary="Retrieves the average star rating and total likes for users who have posted more than a specified number of tips. The operation calculates these metrics by aggregating data from the Elite, Users, and Tips tables, filtering out users with fewer tips than the specified minimum.")
async def get_user_average_stars_and_tip_likes(min_tips: int = Query(..., description="Minimum number of tips")):
    cursor.execute("SELECT T2.user_average_stars, COUNT(T3.likes) FROM Elite AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id INNER JOIN Tips AS T3 ON T3.user_id = T2.user_id GROUP BY T1.user_id HAVING COUNT(T1.user_id) > ?", (min_tips,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the top category ID based on the number of reviews in a specific category
@app.get("/v1/public_review_platform/top_category_by_reviews", operation_id="get_top_category_by_reviews", summary="Retrieve the category ID with the highest number of reviews for a given category name. This operation fetches data from the Business_Categories, Categories, and Reviews tables, filtering by the specified category name. The result is the category ID with the most reviews in the selected category.")
async def get_top_category_by_reviews(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T2.category_id FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Reviews AS T3 ON T3.business_id = T1.business_id WHERE T2.category_name = ? GROUP BY T2.category_id ORDER BY COUNT(T2.category_id) DESC LIMIT 1", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"category_id": []}
    return {"category_id": result[0]}

# Endpoint to get the count of active businesses
@app.get("/v1/public_review_platform/count_active_businesses", operation_id="get_count_active_businesses", summary="Retrieves the total number of businesses currently marked as active. The active status is used to filter the businesses. For example, setting the active status to 'True' will return the count of businesses that are currently active.")
async def get_count_active_businesses(active: str = Query(..., description="Active status (e.g., 'True')")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE active LIKE ?", (active,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs based on city and review count
@app.get("/v1/public_review_platform/business_ids_by_city_and_review_count", operation_id="get_business_ids_by_city_and_review_count", summary="Retrieves the unique identifiers of businesses located in a specified city and having a particular review count. The city is matched using a partial string comparison, and the review count is determined based on a predefined categorization (e.g., 'Low').")
async def get_business_ids_by_city_and_review_count(city: str = Query(..., description="City name"), review_count: str = Query(..., description="Review count (e.g., 'Low')")):
    cursor.execute("SELECT business_id FROM Business WHERE city LIKE ? AND review_count LIKE ?", (city, review_count))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of businesses based on state, review count, and active status
@app.get("/v1/public_review_platform/count_businesses_by_state_review_count_active", operation_id="get_count_businesses_by_state_review_count_active", summary="Retrieves the total number of businesses that match the specified state, review count category, and active status. The state parameter filters businesses by their location, the review count parameter categorizes businesses based on their review volume, and the active parameter determines whether businesses are currently active or not.")
async def get_count_businesses_by_state_review_count_active(state: str = Query(..., description="State name"), review_count: str = Query(..., description="Review count (e.g., 'High')"), active: str = Query(..., description="Active status (e.g., 'True')")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE state LIKE ? AND review_count LIKE ? AND active LIKE ?", (state, review_count, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top attribute name based on business stars
@app.get("/v1/public_review_platform/top_attribute_name_by_stars", operation_id="get_top_attribute_name", summary="Retrieves the name of the attribute that is most associated with businesses having the highest star ratings. The attribute is determined by analyzing the business attributes and their corresponding star ratings.")
async def get_top_attribute_name():
    cursor.execute("SELECT T3.attribute_name FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id ORDER BY T1.stars DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"attribute_name": []}
    return {"attribute_name": result[0]}

# Endpoint to get the count of businesses with specific attributes, active status, and review count
@app.get("/v1/public_review_platform/business_count_attributes_active_review_count", operation_id="get_business_count_attributes", summary="Retrieves the number of businesses that match the specified attribute, active status, and review count level. This operation considers businesses that are associated with the given attribute and meet the provided active status and review count criteria.")
async def get_business_count_attributes(attribute_name: str = Query(..., description="Attribute name"), active: str = Query(..., description="Active status"), review_count: str = Query(..., description="Review count level")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id WHERE T3.attribute_name LIKE ? AND T1.active LIKE ? AND T1.review_count LIKE ?", (attribute_name, active, review_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get category names for businesses with specific review count, city, active status, and state
@app.get("/v1/public_review_platform/category_names_review_count_city_active_state", operation_id="get_category_names", summary="Retrieve the names of categories associated with businesses that meet specific criteria, including a certain review count, a particular city, an active status, and a specific state. This operation provides a filtered list of category names based on the provided parameters.")
async def get_category_names(review_count: str = Query(..., description="Review count level"), city: str = Query(..., description="City name"), active: str = Query(..., description="Active status"), state: str = Query(..., description="State")):
    cursor.execute("SELECT T3.category_name FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T1.review_count = ? AND T1.city = ? AND T1.active = ? AND T1.state = ?", (review_count, city, active, state))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get category names of inactive businesses in a specific state
@app.get("/v1/public_review_platform/inactive_business_categories", operation_id="get_inactive_business_categories", summary="Retrieves the names of categories associated with businesses that are currently inactive in a specified state. The operation filters businesses based on their active status and state, and returns the corresponding category names.")
async def get_inactive_business_categories(active: str = Query(..., description="Active status of the business"), state: str = Query(..., description="State of the business")):
    cursor.execute("SELECT T3.category_name FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T1.active LIKE ? AND T1.state LIKE ?", (active, state))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get opening times of businesses with specific criteria
@app.get("/v1/public_review_platform/business_opening_times", operation_id="get_business_opening_times", summary="Retrieves the opening times of businesses in a specified city that are currently active and have a certain review count. The results are grouped by opening time.")
async def get_business_opening_times(city: str = Query(..., description="City of the business"), active: str = Query(..., description="Active status of the business"), review_count: str = Query(..., description="Review count of the business")):
    cursor.execute("SELECT T2.opening_time FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city LIKE ? AND T1.active LIKE ? AND T1.review_count LIKE ? GROUP BY T2.opening_time", (city, active, review_count))
    result = cursor.fetchall()
    if not result:
        return {"opening_times": []}
    return {"opening_times": [row[0] for row in result]}

# Endpoint to get the percentage of businesses with less than 3 stars in a specific category
@app.get("/v1/public_review_platform/percentage_low_rated_businesses", operation_id="get_percentage_low_rated_businesses", summary="Retrieves the percentage of businesses in a specified category that have received less than 3 stars. The calculation is based on the total number of businesses in the given category.")
async def get_percentage_low_rated_businesses(category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.stars < 3 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.stars) AS 'percentage' FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T3.category_name LIKE ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get closing times and days of week for highly rated businesses in a specific city
@app.get("/v1/public_review_platform/highly_rated_business_hours", operation_id="get_highly_rated_business_hours", summary="Retrieves the closing times and days of the week for businesses in a specific city that have a rating higher than 60% of the average rating for active businesses in that city. The city and active status of the businesses are provided as input parameters.")
async def get_highly_rated_business_hours(city: str = Query(..., description="City of the business"), active: str = Query(..., description="Active status of the business")):
    cursor.execute("SELECT T2.closing_time, T3.day_of_week FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city LIKE ? AND T1.active LIKE ? AND T1.stars > 0.6 * ( SELECT AVG(T1.stars) FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city LIKE ? AND T1.active LIKE ? )", (city, active, city, active))
    result = cursor.fetchall()
    if not result:
        return {"business_hours": []}
    return {"business_hours": [{"closing_time": row[0], "day_of_week": row[1]} for row in result]}

# Endpoint to get the count of users with a specific average star rating
@app.get("/v1/public_review_platform/user_count_by_average_stars", operation_id="get_user_count_by_average_stars", summary="Get the count of users with a specific average star rating")
async def get_user_count_by_average_stars(user_average_stars: float = Query(..., description="Average star rating of the user"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT COUNT(user_id) FROM Users WHERE user_average_stars = ? LIMIT ?", (user_average_stars, limit))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get cities with businesses open during specific hours on a specific day
@app.get("/v1/public_review_platform/cities_with_specific_business_hours_day", operation_id="get_cities_with_specific_business_hours_day", summary="Retrieves a list of cities where businesses are open during the specified hours on the given day of the week. The operation filters businesses based on their opening and closing times, and the day of the week, to determine the cities with matching businesses.")
async def get_cities_with_specific_business_hours_day(opening_time: str = Query(..., description="Opening time of the business"), closing_time: str = Query(..., description="Closing time of the business"), day_of_week: str = Query(..., description="Day of the week")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T2.opening_time LIKE ? AND T2.closing_time LIKE ? AND T3.day_of_week LIKE ?", (opening_time, closing_time, day_of_week))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of businesses with specific attribute values and names
@app.get("/v1/public_review_platform/count_businesses_by_attribute", operation_id="get_count_businesses_by_attribute", summary="Retrieves the total number of businesses that match the specified attribute value and name. The attribute value and name are used as filters to determine the count.")
async def get_count_businesses_by_attribute(attribute_value: str = Query(..., description="Attribute value to filter by"), attribute_name: str = Query(..., description="Attribute name to filter by")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T2.attribute_value LIKE ? AND T1.attribute_name LIKE ?", (attribute_value, attribute_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs based on city, active status, and day of the week
@app.get("/v1/public_review_platform/business_ids_by_city_active_day", operation_id="get_business_ids_by_city_active_day", summary="Retrieves a list of business IDs that are located in a specified city, are currently active, and operate on a particular day of the week. The city, active status, and day of the week are provided as input parameters.")
async def get_business_ids_by_city_active_day(city: str = Query(..., description="City to filter by"), active: str = Query(..., description="Active status to filter by"), day_of_week: str = Query(..., description="Day of the week to filter by")):
    cursor.execute("SELECT T1.business_id FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city LIKE ? AND T1.active LIKE ? AND T3.day_of_week LIKE ?", (city, active, day_of_week))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get category names based on business active status and state
@app.get("/v1/public_review_platform/category_names_by_active_state", operation_id="get_category_names_by_active_state", summary="Retrieves the names of categories associated with businesses that meet the specified active status and are not located in the provided state. The active status and state are used as filters to narrow down the results.")
async def get_category_names_by_active_state(active: str = Query(..., description="Active status to filter by"), state: str = Query(..., description="State to filter by")):
    cursor.execute("SELECT T3.category_name FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T1.active LIKE ? AND T1.state NOT LIKE ?", (active, state))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get category names based on business stars and review count
@app.get("/v1/public_review_platform/category_names_by_stars_review_count", operation_id="get_category_names_by_stars_review_count", summary="Retrieves the names of categories associated with businesses that have a specific number of stars and a review count that matches a given pattern. The operation filters businesses based on the provided star rating and review count pattern, then returns the corresponding category names.")
async def get_category_names_by_stars_review_count(stars: int = Query(..., description="Number of stars to filter by"), review_count: str = Query(..., description="Review count to filter by")):
    cursor.execute("SELECT T3.category_name FROM Business AS T1 INNER JOIN Business_Categories ON T1.business_id = Business_Categories.business_id INNER JOIN Categories AS T3 ON Business_Categories.category_id = T3.category_id WHERE T1.stars = ? AND T1.review_count LIKE ?", (stars, review_count))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the count of businesses with specific attribute values and names
@app.get("/v1/public_review_platform/count_businesses_by_attribute_value_name", operation_id="get_count_businesses_by_attribute_value_name", summary="Retrieves the total number of businesses that match the specified attribute value and name. This operation filters businesses based on the provided attribute value and name, returning the count of businesses that meet the criteria.")
async def get_count_businesses_by_attribute_value_name(attribute_value: str = Query(..., description="Attribute value to filter by"), attribute_name: str = Query(..., description="Attribute name to filter by")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T2.attribute_value = ? AND T1.attribute_name = ?", (attribute_value, attribute_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get cities based on business hours and day of the week
@app.get("/v1/public_review_platform/cities_by_business_hours_day", operation_id="get_cities_by_business_hours_day", summary="Retrieves a list of cities where businesses operate within the specified opening and closing times on a given day of the week. The operation filters businesses based on their opening and closing times, and the day of the week, to provide a targeted list of cities.")
async def get_cities_by_business_hours_day(closing_time: str = Query(..., description="Closing time to filter by"), opening_time: str = Query(..., description="Opening time to filter by"), day_of_week: str = Query(..., description="Day of the week to filter by")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T2.closing_time LIKE ? AND T2.opening_time LIKE ? AND T3.day_of_week LIKE ?", (closing_time, opening_time, day_of_week))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of user fans ordered by the number of likes on their tips, limited to a certain number of results
@app.get("/v1/public_review_platform/count_user_fans_by_tips_likes", operation_id="get_count_user_fans_by_tips_likes", summary="Retrieves the total count of fans for users, ordered by the popularity of their tips as determined by the number of likes. The results are limited to a specified number.")
async def get_count_user_fans_by_tips_likes(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT COUNT(T1.user_fans) FROM Users AS T1 INNER JOIN Tips AS T2 ON T1.user_id = T2.user_id ORDER BY COUNT(T2.likes) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get cities based on business hours and day of the week
@app.get("/v1/public_review_platform/cities_by_business_hours_day_exact", operation_id="get_cities_by_business_hours_day_exact", summary="Retrieves a list of cities where businesses operate with specific opening and closing times on a particular day of the week. The operation filters businesses based on the provided opening and closing times, and the day of the week, returning the cities where these businesses are located.")
async def get_cities_by_business_hours_day_exact(closing_time: str = Query(..., description="Closing time to filter by"), opening_time: str = Query(..., description="Opening time to filter by"), day_of_week: str = Query(..., description="Day of the week to filter by")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T2.closing_time = ? AND T2.opening_time = ? AND T3.day_of_week = ?", (closing_time, opening_time, day_of_week))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of businesses in a specific category with a specific review count
@app.get("/v1/public_review_platform/count_businesses_by_category_review_count", operation_id="get_count_businesses_by_category_review_count", summary="Retrieves the total number of businesses in a specified category that have a certain number of reviews. The category and review count are provided as input parameters.")
async def get_count_businesses_by_category_review_count(category_name: str = Query(..., description="Category name to filter by"), review_count: str = Query(..., description="Review count to filter by")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.category_name = ? AND T3.review_count = ?", (category_name, review_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with a specific attribute name and value
@app.get("/v1/public_review_platform/business_count_by_attribute", operation_id="get_business_count_by_attribute", summary="Retrieves the number of businesses that possess a specified attribute with a given value. The attribute is identified by its name, and the count reflects businesses that have the attribute set to the provided value.")
async def get_business_count_by_attribute(attribute_name: str = Query(..., description="Name of the attribute"), attribute_value: str = Query(..., description="Value of the attribute")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business_Attributes AS T1 INNER JOIN Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T2.attribute_name = ? AND T1.attribute_value = ?", (attribute_name, attribute_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the weighted average of stars for businesses in two states
@app.get("/v1/public_review_platform/weighted_average_stars", operation_id="get_weighted_average_stars", summary="Retrieves the weighted average of star ratings for businesses in two specified states. The calculation considers the total sum of star ratings from businesses in each state and divides it by the overall sum of star ratings across all businesses. This operation requires the input of two state names to determine the businesses for which the weighted average is calculated.")
async def get_weighted_average_stars(state1: str = Query(..., description="First state"), state2: str = Query(..., description="Second state")):
    cursor.execute("SELECT 1.0 * (( SELECT SUM(T1.stars) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.state = ? ) + ( SELECT SUM(T1.stars) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.state = ? )) / ( SELECT SUM(T1.stars) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id ) AS result", (state1, state2))
    result = cursor.fetchone()
    if not result:
        return {"average_stars": []}
    return {"average_stars": result[0]}

# Endpoint to get the difference in the number of businesses open on two specific days of the week with specific opening and closing times
@app.get("/v1/public_review_platform/business_open_diff", operation_id="get_business_open_diff", summary="Retrieve the difference in the number of businesses that are open on two specific days of the week, with the same opening and closing times. The operation compares the total count of businesses open on the first day of the week with the total count of businesses open on the second day of the week. The input parameters include the first and second days of the week, as well as the opening and closing times in 'HHAM/PM' format.")
async def get_business_open_diff(day1: str = Query(..., description="First day of the week"), day2: str = Query(..., description="Second day of the week"), opening_time: str = Query(..., description="Opening time in 'HHAM/PM' format"), closing_time: str = Query(..., description="Closing time in 'HHAM/PM' format")):
    cursor.execute("SELECT SUM(CASE WHEN T3.day_of_week = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T3.day_of_week = ? THEN 1 ELSE 0 END) AS DIFF FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T2.opening_time = ? AND T2.closing_time = ?", (day1, day2, opening_time, closing_time))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the attribute ID based on the attribute name
@app.get("/v1/public_review_platform/attribute_id_by_name", operation_id="get_attribute_id_by_name", summary="Retrieves the unique identifier of an attribute by specifying its name. The attribute name is used to search the Attributes table and return the corresponding attribute ID.")
async def get_attribute_id_by_name(attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT attribute_id FROM Attributes WHERE attribute_name = ?", (attribute_name,))
    result = cursor.fetchone()
    if not result:
        return {"attribute_id": []}
    return {"attribute_id": result[0]}

# Endpoint to get the count of active businesses in a specific city
@app.get("/v1/public_review_platform/active_business_count_by_city", operation_id="get_active_business_count_by_city", summary="Retrieves the total number of active businesses located in a specified city. The operation considers businesses that are currently active and matches the provided city name.")
async def get_active_business_count_by_city(active: str = Query(..., description="Active status of the business"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE active = ? AND city = ?", (active, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with a specific star rating in a specific city
@app.get("/v1/public_review_platform/business_count_by_stars_city", operation_id="get_business_count_by_stars_city", summary="Retrieves the total number of businesses in a given city that have a specified star rating. The operation considers both the star rating and the city to provide an accurate count.")
async def get_business_count_by_stars_city(stars: int = Query(..., description="Star rating of the business"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE stars = ? AND city = ?", (stars, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with a specific review count in a specific city
@app.get("/v1/public_review_platform/business_count_by_review_count_city", operation_id="get_business_count_by_review_count_city", summary="Retrieves the total number of businesses in a given city that have a specified number of reviews. This operation is useful for understanding the distribution of businesses based on their review count within a particular city.")
async def get_business_count_by_review_count_city(review_count: str = Query(..., description="Review count of the business"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE review_count = ? AND city = ?", (review_count, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of reviews with a specific length for businesses in a specific city with a specific star rating and review count
@app.get("/v1/public_review_platform/review_length_count", operation_id="get_review_length_count", summary="Retrieves the count of reviews of a specific length for businesses in a given city, filtered by a specific star rating and review count.")
async def get_review_length_count(city: str = Query(..., description="City name"), stars: float = Query(..., description="Star rating of the business"), review_count: str = Query(..., description="Review count of the business"), review_length: str = Query(..., description="Length of the review")):
    cursor.execute("SELECT COUNT(T2.review_length) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ? AND T1.stars = ? AND T1.review_count = ? AND T2.review_length = ?", (city, stars, review_count, review_length))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get attribute names for businesses in a specific city with a specific review count and attribute name
@app.get("/v1/public_review_platform/attribute_names_by_city_review_count", operation_id="get_attribute_names_by_city_review_count", summary="Retrieves the names of attributes associated with businesses in a specified city that have a particular review count and a specific attribute name. This operation allows you to filter businesses based on their location, review count, and attribute name, providing a focused list of attribute names that meet the given criteria.")
async def get_attribute_names_by_city_review_count(city: str = Query(..., description="City name"), review_count: str = Query(..., description="Review count of the business"), attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT T3.attribute_name FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id WHERE T1.city = ? AND T1.review_count = ? AND T3.attribute_name = ?", (city, review_count, attribute_name))
    result = cursor.fetchall()
    if not result:
        return {"attribute_names": []}
    return {"attribute_names": [row[0] for row in result]}

# Endpoint to get the business hours difference for a specific day and business
@app.get("/v1/public_review_platform/business_hours_difference", operation_id="get_business_hours_difference", summary="Retrieves the total business hours for a specific day and business. The operation calculates the difference between the closing and opening times, providing the total hours a business is open on a given day of the week. The input parameters specify the day of the week and the business ID.")
async def get_business_hours_difference(day_of_week: str = Query(..., description="Day of the week"), business_id: int = Query(..., description="ID of the business")):
    cursor.execute("SELECT SUBSTR(T1.closing_time, 1, 2) + 12 - SUBSTR(T1.opening_time, 1, 2) AS YYSJ FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id WHERE T2.day_of_week = ? AND T1.business_id = ?", (day_of_week, business_id))
    result = cursor.fetchone()
    if not result:
        return {"YYSJ": []}
    return {"YYSJ": result[0]}

# Endpoint to get business IDs based on city, review stars, and review votes funny
@app.get("/v1/public_review_platform/business_ids_by_city_review_stars_funny", operation_id="get_business_ids_by_city_review_stars_funny", summary="Retrieves the IDs of businesses located in a specified city that have a certain number of review stars and a specific count of funny review votes. The operation filters businesses based on the provided city, review star rating, and the number of funny votes received on their reviews.")
async def get_business_ids_by_city_review_stars_funny(city: str = Query(..., description="City name"), review_stars: int = Query(..., description="Number of review stars"), review_votes_funny: str = Query(..., description="Review votes funny")):
    cursor.execute("SELECT T1.business_id FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ? AND T2.review_stars = ? AND T2.review_votes_funny = ?", (city, review_stars, review_votes_funny))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get cities based on tip length and likes
@app.get("/v1/public_review_platform/cities_by_tip_length_likes", operation_id="get_cities_by_tip_length_likes", summary="Retrieves a list of cities where businesses have tips of a specified length and a specified number of likes. The tip length and number of likes are provided as input parameters.")
async def get_cities_by_tip_length_likes(tip_length: str = Query(..., description="Tip length"), likes: int = Query(..., description="Number of likes")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Tips AS T2 ON T1.business_id = T2.business_id WHERE T2.tip_length = ? AND T2.likes = ?", (tip_length, likes))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of user compliments based on user details and compliment type
@app.get("/v1/public_review_platform/count_user_compliments", operation_id="get_count_user_compliments", summary="Retrieves the total count of compliments given by a user, based on the user's yelping history, average stars rating, fan base, and the type of compliment. This operation provides insights into the user's engagement and influence on the platform.")
async def get_count_user_compliments(user_yelping_since_year: int = Query(..., description="Year the user started yelping"), user_average_stars: float = Query(..., description="Average stars given by the user"), user_fans: str = Query(..., description="User fans"), compliment_type: str = Query(..., description="Type of compliment")):
    cursor.execute("SELECT COUNT(T2.user_id) FROM Users AS T1 INNER JOIN Users_Compliments AS T2 ON T1.user_id = T2.user_id INNER JOIN Compliments AS T3 ON T2.compliment_id = T3.compliment_id WHERE T1.user_yelping_since_year = ? AND T1.user_average_stars = ? AND T1.user_fans = ? AND T3.compliment_type = ?", (user_yelping_since_year, user_average_stars, user_fans, compliment_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of compliments for a specific user and compliment type
@app.get("/v1/public_review_platform/count_compliments_user_type", operation_id="get_count_compliments_user_type", summary="Retrieves the total count of a specific type of compliment associated with a given user. The operation requires the type of compliment and the user's ID as input parameters to accurately calculate the count.")
async def get_count_compliments_user_type(compliment_type: str = Query(..., description="Type of compliment"), user_id: int = Query(..., description="ID of the user")):
    cursor.execute("SELECT COUNT(T2.number_of_compliments) FROM Compliments AS T1 INNER JOIN Users_Compliments AS T2 ON T1.compliment_id = T2.compliment_id WHERE T1.compliment_type = ? AND T2.user_id = ?", (compliment_type, user_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of businesses in a specific category
@app.get("/v1/public_review_platform/percentage_businesses_category", operation_id="get_percentage_businesses_category", summary="Retrieves the percentage of businesses that belong to a specified category. This operation calculates the proportion of businesses in the given category by comparing the total count of businesses in that category to the overall count of businesses across all categories.")
async def get_percentage_businesses_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.category_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.category_name) FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of businesses in two specific categories
@app.get("/v1/public_review_platform/ratio_businesses_categories", operation_id="get_ratio_businesses_categories", summary="Retrieve the ratio of businesses in two specified categories. This operation calculates the proportion of businesses in the first category relative to the second category. The input parameters define the categories to be compared.")
async def get_ratio_businesses_categories(category_name_1: str = Query(..., description="Name of the first category"), category_name_2: str = Query(..., description="Name of the second category")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.category_name = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.category_name = ? THEN 1 ELSE 0 END) AS TIMES FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id", (category_name_1, category_name_2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get business details based on state and active status
@app.get("/v1/public_review_platform/business_details_by_state_active", operation_id="get_business_details", summary="Retrieves the business ID, active status, and city for businesses located in a specified state and with a specified active status. The operation filters businesses based on the provided state and active status parameters.")
async def get_business_details(state: str = Query(..., description="State of the business"), active: str = Query(..., description="Active status of the business")):
    cursor.execute("SELECT business_id, active, city FROM Business WHERE state = ? AND active = ?", (state, active))
    result = cursor.fetchall()
    if not result:
        return {"businesses": []}
    return {"businesses": result}

# Endpoint to get the percentage of active businesses
@app.get("/v1/public_review_platform/percentage_active_businesses", operation_id="get_percentage_active_businesses", summary="Retrieves the percentage of businesses that are currently active. The active status of businesses is considered to determine the percentage. The calculation is based on the total number of businesses in the system.")
async def get_percentage_active_businesses(active: str = Query(..., description="Active status of the business")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN active = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(business_id) FROM Business", (active,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get attributes based on a name pattern
@app.get("/v1/public_review_platform/attributes_by_name_pattern", operation_id="get_attributes_by_name_pattern", summary="Retrieves a list of attributes whose names match the provided pattern. The pattern is case-insensitive and can include wildcard characters (*) to represent any number of characters or (?) to represent a single character.")
async def get_attributes_by_name_pattern(attribute_name: str = Query(..., description="Pattern to match attribute names")):
    cursor.execute("SELECT attribute_id, attribute_name FROM Attributes WHERE attribute_name LIKE ?", (attribute_name,))
    result = cursor.fetchall()
    if not result:
        return {"attributes": []}
    return {"attributes": result}

# Endpoint to get the year with the most elite users
@app.get("/v1/public_review_platform/year_with_most_elite_users", operation_id="get_year_with_most_elite_users", summary="Retrieves the year with the highest number of elite users from the provided set of two years. The operation considers the given years and returns the year with the most elite users, based on the count of unique user IDs.")
async def get_year_with_most_elite_users(year1: int = Query(..., description="First year to consider"), year2: int = Query(..., description="Second year to consider")):
    cursor.execute("SELECT year_id FROM Elite WHERE year_id IN (?, ?) GROUP BY year_id ORDER BY COUNT(user_id) DESC LIMIT 1", (year1, year2))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the percentage of users with a specific number of compliments
@app.get("/v1/public_review_platform/percentage_users_by_compliments", operation_id="get_percentage_users_by_compliments", summary="Retrieves the percentage of users who have received a specified number of compliments. This operation calculates the proportion of users with the given number of compliments relative to the total number of users in the system.")
async def get_percentage_users_by_compliments(number_of_compliments: str = Query(..., description="Number of compliments")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN number_of_compliments = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(user_id) FROM Users_compliments", (number_of_compliments,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get reviews based on cool votes
@app.get("/v1/public_review_platform/reviews_by_cool_votes", operation_id="get_reviews_by_cool_votes", summary="Retrieves a list of reviews that have received a specified number of 'cool' votes. The response includes the business and user associated with each review.")
async def get_reviews_by_cool_votes(review_votes_cool: str = Query(..., description="Cool votes for the review")):
    cursor.execute("SELECT business_id, user_id FROM Reviews WHERE review_votes_cool = ?", (review_votes_cool,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get user tips based on yelping since year and user fans
@app.get("/v1/public_review_platform/user_tips_by_yelping_year_fans", operation_id="get_user_tips_by_yelping_year_fans", summary="Retrieves user tips based on the year the user started yelping and their fans level. This operation returns the user ID, associated business ID, and length of the tip for users who meet the specified yelping year and fans level criteria.")
async def get_user_tips_by_yelping_year_fans(user_yelping_since_year: int = Query(..., description="Year the user started yelping"), user_fans: str = Query(..., description="User fans level")):
    cursor.execute("SELECT T1.user_id, T2.business_id, T2.tip_length FROM Users AS T1 INNER JOIN Tips AS T2 ON T1.user_id = T2.user_id WHERE T1.user_yelping_since_year = ? AND T1.user_fans = ?", (user_yelping_since_year, user_fans))
    result = cursor.fetchall()
    if not result:
        return {"tips": []}
    return {"tips": result}

# Endpoint to get business and user details based on review votes and length
@app.get("/v1/public_review_platform/business_user_details_by_review_votes_length", operation_id="get_business_user_details_by_review_votes_length", summary="Retrieves details of businesses and their respective users who have received a specific number of cool and funny votes for their reviews of a certain length. The response includes the business ID, its active status, the user ID, and the year the user started using the platform.")
async def get_business_user_details_by_review_votes_length(review_votes_cool: str = Query(..., description="Cool votes for the review"), review_votes_funny: str = Query(..., description="Funny votes for the review"), review_length: str = Query(..., description="Length of the review")):
    cursor.execute("SELECT T1.business_id, T1.active, T3.user_id, T3.user_yelping_since_year FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id INNER JOIN Users AS T3 ON T2.user_id = T3.user_id WHERE T2.review_votes_cool = ? AND T2.review_votes_funny = ? AND T2.review_length = ?", (review_votes_cool, review_votes_funny, review_length))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get business attributes based on attribute name and active status
@app.get("/v1/public_review_platform/business_attributes_by_name_active", operation_id="get_business_attributes_by_name_active", summary="Retrieves business details, including the attribute ID, business ID, and city, for businesses with a specified attribute name and active status. The attribute name and active status are provided as input parameters.")
async def get_business_attributes_by_name_active(attribute_name: str = Query(..., description="Name of the attribute"), active: str = Query(..., description="Active status of the business")):
    cursor.execute("SELECT T1.attribute_id, T2.business_id, T3.city FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.attribute_name = ? AND T3.active = ?", (attribute_name, active))
    result = cursor.fetchall()
    if not result:
        return {"attributes": []}
    return {"attributes": result}

# Endpoint to get the percentage of businesses with a specific attribute
@app.get("/v1/public_review_platform/percentage_businesses_with_attribute", operation_id="get_percentage_businesses_with_attribute", summary="Retrieves the percentage of businesses that possess a specified attribute. The attribute is identified by its name, which is provided as an input parameter. The calculation is based on the total count of businesses with the given attribute and the overall count of businesses.")
async def get_percentage_businesses_with_attribute(attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.attribute_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.attribute_name) FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id", (attribute_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get user IDs and review lengths for reviews in a specific city with a specific business status and review stars
@app.get("/v1/public_review_platform/reviews_by_city_status_stars", operation_id="get_reviews_by_city_status_stars", summary="Retrieves the user IDs and review lengths for reviews of businesses in a specified city that are currently active or inactive, and have a specific number of review stars. This operation is useful for analyzing review data based on city, business status, and review stars.")
async def get_reviews_by_city_status_stars(city: str = Query(..., description="City name"), active: str = Query(..., description="Business status (true or false)"), review_stars: int = Query(..., description="Review stars")):
    cursor.execute("SELECT T2.user_id, T2.review_length FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ? AND T1.active = ? AND T2.review_stars = ?", (city, active, review_stars))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get user average stars, year ID, compliment type, and number of compliments for a specific user and compliment type
@app.get("/v1/public_review_platform/user_compliments_by_type_and_user", operation_id="get_user_compliments_by_type_and_user", summary="Retrieves the average star rating, year ID, compliment type, and count of compliments for a specific user and compliment type, based on the provided number of compliments and user ID.")
async def get_user_compliments_by_type_and_user(number_of_compliments: str = Query(..., description="Number of compliments"), user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T2.user_average_stars, T1.year_id, T4.compliment_type, T3.number_of_compliments FROM Elite AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id INNER JOIN Users_Compliments AS T3 ON T2.user_id = T3.user_id INNER JOIN Compliments AS T4 ON T3.compliment_id = T4.compliment_id INNER JOIN Years AS T5 ON T1.year_id = T5.year_id WHERE T3.number_of_compliments = ? AND T3.user_id = ?", (number_of_compliments, user_id))
    result = cursor.fetchall()
    if not result:
        return {"compliments": []}
    return {"compliments": result}

# Endpoint to get business IDs, states, and cities for businesses in a specific category
@app.get("/v1/public_review_platform/businesses_by_category", operation_id="get_businesses_by_category", summary="Retrieves a limited set of details for businesses in a specified category. The information includes business IDs, states, and cities. The category is determined by the provided category name.")
async def get_businesses_by_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T2.business_id, T3.state, T3.city FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.category_name = ? LIMIT 5", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"businesses": []}
    return {"businesses": result}

# Endpoint to get category names where the number of businesses exceeds a certain percentage of the total
@app.get("/v1/public_review_platform/categories_exceeding_business_percentage", operation_id="get_categories_exceeding_business_percentage", summary="Retrieves the names of categories where the number of associated businesses surpasses the specified percentage of the total business count. This operation provides a means to identify categories with a significant presence in the platform, offering valuable insights for business analysis and decision-making.")
async def get_categories_exceeding_business_percentage(percentage: float = Query(..., description="Percentage of total businesses")):
    cursor.execute("SELECT T1.category_name FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id GROUP BY T2.category_id HAVING COUNT(T2.business_id) > ( SELECT COUNT(T3.business_id) FROM Business_Categories AS T3 ) * ?", (percentage,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": result}

# Endpoint to get user IDs and user fans for reviews in a specific city with a specific business rating
@app.get("/v1/public_review_platform/user_fans_by_city_and_stars", operation_id="get_user_fans_by_city_and_stars", summary="Retrieves a list of user IDs and their corresponding fan counts for reviews in a specific city with a given business rating. The city and rating are provided as input parameters.")
async def get_user_fans_by_city_and_stars(city: str = Query(..., description="City name"), stars: int = Query(..., description="Business rating")):
    cursor.execute("SELECT T3.user_id, T3.user_fans FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id INNER JOIN Users AS T3 ON T2.user_id = T3.user_id WHERE T1.city = ? AND T1.stars = ?", (city, stars))
    result = cursor.fetchall()
    if not result:
        return {"user_fans": []}
    return {"user_fans": result}

# Endpoint to get the difference in the number of businesses between two categories
@app.get("/v1/public_review_platform/business_count_difference_by_categories", operation_id="get_business_count_difference_by_categories", summary="Retrieve the difference in the number of businesses between two specified categories. This operation compares the total count of businesses in the first category with that of the second category, returning the difference as a single numerical value.")
async def get_business_count_difference_by_categories(category_name_1: str = Query(..., description="First category name"), category_name_2: str = Query(..., description="Second category name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.category_name = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.category_name = ? THEN 1 ELSE 0 END) AS diff FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id", (category_name_1, category_name_2))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get compliment types and user fans for users with a specific number of compliments and user IDs below a certain value
@app.get("/v1/public_review_platform/compliment_types_and_user_fans", operation_id="get_compliment_types_and_user_fans", summary="Retrieves the types of compliments and the number of fans for users who have received a specific number of compliments and have a user ID less than a certain value. This operation is useful for analyzing user engagement and popularity based on the compliments they have received.")
async def get_compliment_types_and_user_fans(number_of_compliments: str = Query(..., description="Number of compliments"), max_user_id: int = Query(..., description="Maximum user ID")):
    cursor.execute("SELECT T1.compliment_type, T3.user_fans FROM Compliments AS T1 INNER JOIN Users_Compliments AS T2 ON T1.compliment_id = T2.compliment_id INNER JOIN Users AS T3 ON T2.user_id = T3.user_id WHERE T2.number_of_compliments = ? AND T2.user_id < ?", (number_of_compliments, max_user_id))
    result = cursor.fetchall()
    if not result:
        return {"compliment_types_and_user_fans": []}
    return {"compliment_types_and_user_fans": result}

# Endpoint to get distinct business IDs based on closing time
@app.get("/v1/public_review_platform/business_ids_by_closing_time", operation_id="get_business_ids_by_closing_time", summary="Retrieves a unique set of business identifiers that close at a specific time. The closing time is provided as input, allowing the user to filter the results accordingly.")
async def get_business_ids_by_closing_time(closing_time: str = Query(..., description="Closing time (e.g., '8PM')")):
    cursor.execute("SELECT DISTINCT business_id FROM Business_Hours WHERE closing_time = ?", (closing_time,))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": result}

# Endpoint to get the count of businesses in a specific city, state, and with a specific rating
@app.get("/v1/public_review_platform/business_count_by_city_state_stars", operation_id="get_business_count_by_city_state_stars", summary="Retrieves the total number of businesses located in a specified city and state, filtered by a particular star rating. The city and state are identified by their respective names, while the star rating is represented as a numerical value.")
async def get_business_count_by_city_state_stars(city: str = Query(..., description="City name"), state: str = Query(..., description="State abbreviation"), stars: int = Query(..., description="Business rating")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE city = ? AND state = ? AND stars = ?", (city, state, stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses in a specific city with a rating above a certain value
@app.get("/v1/public_review_platform/business_count_by_city_and_min_stars", operation_id="get_business_count_by_city_and_min_stars", summary="Retrieves the number of businesses in a given city that have a rating higher than a specified value. The city and minimum rating are provided as input parameters.")
async def get_business_count_by_city_and_min_stars(city: str = Query(..., description="City name"), min_stars: int = Query(..., description="Minimum business rating")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE city = ? AND stars > ?", (city, min_stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user IDs based on yelping since year and average stars
@app.get("/v1/public_review_platform/user_ids_by_yelping_year_and_stars", operation_id="get_user_ids_by_yelping_year_and_stars", summary="Retrieves the IDs of users who began using the platform in a specified year and have an average star rating below a given value. This operation is useful for identifying users based on their yelping history and rating performance.")
async def get_user_ids_by_yelping_year_and_stars(yelping_since_year: int = Query(..., description="Year the user started yelping"), average_stars: float = Query(..., description="Average star rating of the user")):
    cursor.execute("SELECT user_id FROM Users WHERE user_yelping_since_year = ? AND user_average_stars < ?", (yelping_since_year, average_stars))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the percentage of businesses with a specific star rating
@app.get("/v1/public_review_platform/percentage_businesses_by_stars", operation_id="get_percentage_businesses_by_stars", summary="Retrieves the percentage of businesses with a specified star rating. The operation calculates this percentage by dividing the count of businesses with the given star rating by the total number of businesses, then multiplying the result by 100.")
async def get_percentage_businesses_by_stars(stars: int = Query(..., description="Star rating of the business")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN stars = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(stars) FROM Business", (stars,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference between the maximum and minimum number of reviews per business
@app.get("/v1/public_review_platform/review_count_difference", operation_id="get_review_count_difference", summary="Retrieves the difference between the maximum and minimum number of reviews across all businesses. This operation provides a high-level overview of the review distribution, helping to identify the most and least reviewed businesses.")
async def get_review_count_difference():
    cursor.execute("SELECT ( SELECT COUNT(business_id) FROM Reviews GROUP BY business_id ORDER BY COUNT(business_id) DESC LIMIT 1 ) - ( SELECT COUNT(business_id) FROM Reviews GROUP BY business_id ORDER BY COUNT(business_id) ASC LIMIT 1 ) AS DIFF")
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get business IDs in a specific category open all week
@app.get("/v1/public_review_platform/business_ids_by_category_open_all_week", operation_id="get_business_ids_by_category_open_all_week", summary="Retrieves unique business identifiers for establishments in a specified category that operate seven days a week. The category is determined by the provided category name.")
async def get_business_ids_by_category_open_all_week(category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT DISTINCT T2.business_id FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id INNER JOIN Business_Hours AS T4 ON T3.business_id = T4.business_id WHERE T1.category_name = ? GROUP BY T2.business_id HAVING COUNT(day_id) = 7", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get user IDs of elite users from a specific year
@app.get("/v1/public_review_platform/elite_user_ids_by_year", operation_id="get_elite_user_ids_by_year", summary="Retrieves a unique list of user IDs who achieved elite status in a specified year. The year is provided as an input parameter.")
async def get_elite_user_ids_by_year(actual_year: int = Query(..., description="Year the user was elite")):
    cursor.execute("SELECT DISTINCT T1.user_id FROM Elite AS T1 INNER JOIN Years AS T2 ON T1.year_id = T2.year_id WHERE T2.actual_year = ?", (actual_year,))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the count of business IDs based on day of the week and check-in time label
@app.get("/v1/public_review_platform/count_business_ids_by_day_and_time_label", operation_id="get_count_business_ids_by_day_and_time_label", summary="Retrieves the total number of unique businesses that have check-ins on a specific day of the week and during a particular time period. The day of the week and time period are determined by the provided input parameters.")
async def get_count_business_ids_by_day_and_time_label(day_of_week: str = Query(..., description="Day of the week"), label_time_10: str = Query(..., description="Check-in time label")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Days AS T1 INNER JOIN Checkins AS T2 ON T1.day_id = T2.day_id WHERE T1.day_of_week = ? AND T2.label_time_10 = ?", (day_of_week, label_time_10))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of business IDs based on city and user ID
@app.get("/v1/public_review_platform/count_business_ids_by_city_and_user", operation_id="get_count_business_ids_by_city_and_user", summary="Retrieves the total number of businesses in a specific city that have been reviewed by a particular user. The operation requires the city name and the user's unique identifier as input parameters.")
async def get_count_business_ids_by_city_and_user(city: str = Query(..., description="City of the business"), user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ? AND T2.user_id = ?", (city, user_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get cities based on business category
@app.get("/v1/public_review_platform/cities_by_business_category", operation_id="get_cities_by_business_category", summary="Retrieves a list of cities where businesses of the specified category are located. The category name is used to filter the results.")
async def get_cities_by_business_category(category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT T1.city FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T3.category_name = ?", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the count of user IDs based on compliment type and review votes cool
@app.get("/v1/public_review_platform/count_user_ids_by_compliment_type_and_review_votes", operation_id="get_count_user_ids_by_compliment_type_and_review_votes", summary="Retrieves the total count of unique user IDs who have received a specific type of compliment and have a certain number of 'cool' votes on their reviews. The compliment type and the number of 'cool' votes are provided as input parameters.")
async def get_count_user_ids_by_compliment_type_and_review_votes(compliment_type: str = Query(..., description="Compliment type"), review_votes_cool: str = Query(..., description="Review votes cool")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM Users AS T1 INNER JOIN Users_Compliments AS T2 ON T1.user_id = T2.user_id INNER JOIN Compliments AS T3 ON T2.compliment_id = T3.compliment_id INNER JOIN Reviews AS T4 ON T1.user_id = T4.user_id WHERE T3.compliment_type = ? AND T4.review_votes_cool = ?", (compliment_type, review_votes_cool))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with a specific active status and more than a certain number of distinct attributes
@app.get("/v1/public_review_platform/business_count_by_active_status_and_attributes", operation_id="get_business_count_by_active_status_and_attributes", summary="Retrieves the count of businesses that meet the specified active status and have more than a certain number of distinct attributes. The active status can be either 'true' or 'false', and the minimum number of distinct attributes is also provided as input.")
async def get_business_count_by_active_status_and_attributes(active: str = Query(..., description="Active status of the business (e.g., 'true' or 'false')"), min_attributes: int = Query(..., description="Minimum number of distinct attributes")):
    cursor.execute("SELECT COUNT(*) FROM Business WHERE business_id IN ( SELECT T1.business_id FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id WHERE T1.active = ? GROUP BY T1.business_id HAVING COUNT(DISTINCT T2.attribute_id) > ? )", (active, min_attributes))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs based on city and attribute name
@app.get("/v1/public_review_platform/business_ids_by_city_and_attribute", operation_id="get_business_ids_by_city_and_attribute", summary="Retrieves the unique identifiers of businesses located in a specified city that possess a particular attribute. The operation filters businesses based on the provided city and attribute name, returning a list of corresponding business IDs.")
async def get_business_ids_by_city_and_attribute(city: str = Query(..., description="City of the business"), attribute_name: str = Query(..., description="Attribute name")):
    cursor.execute("SELECT T1.business_id FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id WHERE T1.city = ? AND T3.attribute_name = ?", (city, attribute_name))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the percentage of funny review votes for businesses in a specific city
@app.get("/v1/public_review_platform/funny_review_votes_percentage_by_city", operation_id="get_funny_review_votes_percentage_by_city", summary="Retrieves the percentage of review votes marked as funny for businesses in a specified city. The funny level of review votes and the city are provided as input parameters.")
async def get_funny_review_votes_percentage_by_city(review_votes_funny: str = Query(..., description="Review votes funny level (e.g., 'Low')"), city: str = Query(..., description="City of the business")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.review_votes_funny = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ?", (review_votes_funny, city))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of businesses in one category to another
@app.get("/v1/public_review_platform/category_ratio", operation_id="get_category_ratio", summary="Retrieves the ratio of businesses in the first category to the second category. This operation compares the number of businesses in two specified categories and returns the ratio between them. The input parameters define the categories to be compared.")
async def get_category_ratio(category_name_1: str = Query(..., description="First category name"), category_name_2: str = Query(..., description="Second category name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.category_name = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.category_name = ? THEN 1 ELSE 0 END) AS radio FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id", (category_name_1, category_name_2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of businesses in a specific category
@app.get("/v1/public_review_platform/business_count_by_category", operation_id="get_business_count_by_category", summary="Retrieves the total number of businesses associated with a specified category. The category is identified by its name, which is provided as an input parameter.")
async def get_business_count_by_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id WHERE T1.category_name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the opening time of a business on a specific day
@app.get("/v1/public_review_platform/business_opening_time", operation_id="get_business_opening_time", summary="Retrieves the opening time of a specific business on a given day of the week. The operation requires the unique identifier of the business and the day of the week as input parameters. The output is the opening time of the business for the provided day.")
async def get_business_opening_time(business_id: int = Query(..., description="Business ID"), day_of_week: str = Query(..., description="Day of the week")):
    cursor.execute("SELECT T1.opening_time FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id WHERE T1.business_id = ? AND T2.day_of_week = ?", (business_id, day_of_week))
    result = cursor.fetchone()
    if not result:
        return {"opening_time": []}
    return {"opening_time": result[0]}

# Endpoint to get the top business by stars in a specific city with a specific active status and review count
@app.get("/v1/public_review_platform/top_business_by_stars", operation_id="get_top_business_by_stars", summary="Retrieves the business with the highest star rating in a specified city, given a certain active status and review count level. The active status and review count level are used as filters to narrow down the search results.")
async def get_top_business_by_stars(city: str = Query(..., description="City of the business"), active: str = Query(..., description="Active status of the business (e.g., 'true' or 'false')"), review_count: str = Query(..., description="Review count level (e.g., 'High')")):
    cursor.execute("SELECT business_id FROM Business WHERE city = ? AND active = ? AND review_count = ? ORDER BY stars DESC LIMIT 1", (city, active, review_count))
    result = cursor.fetchone()
    if not result:
        return {"business_id": []}
    return {"business_id": result[0]}

# Endpoint to get business IDs and category names for businesses in a specific city with a specific star rating
@app.get("/v1/public_review_platform/business_ids_and_categories_by_city_and_stars", operation_id="get_business_ids_and_categories_by_city_and_stars", summary="Retrieves the unique identifiers and category names of businesses located in a specified city with a particular star rating. The city and star rating are provided as input parameters.")
async def get_business_ids_and_categories_by_city_and_stars(city: str = Query(..., description="City of the business"), stars: int = Query(..., description="Star rating of the business")):
    cursor.execute("SELECT T1.business_id, T3.category_name FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T1.city = ? AND T1.stars = ?", (city, stars))
    result = cursor.fetchall()
    if not result:
        return {"business_ids_and_categories": []}
    return {"business_ids_and_categories": [{"business_id": row[0], "category_name": row[1]} for row in result]}

# Endpoint to get the percentage of businesses with more than a specified number of stars in a given city and activity status
@app.get("/v1/public_review_platform/business_star_percentage", operation_id="get_business_star_percentage", summary="Retrieves the percentage of businesses in a specified city with a star rating exceeding a given threshold, based on their activity status. This operation considers all businesses in the given city and calculates the proportion of those that have a star rating higher than the provided minimum. The activity status filter allows for the inclusion or exclusion of businesses based on their current operational status.")
async def get_business_star_percentage(min_stars: int = Query(..., description="Minimum number of stars"), city: str = Query(..., description="City name"), active: str = Query(..., description="Activity status")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN stars > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(stars) FROM Business WHERE city = ? AND active = ?", (min_stars, city, active))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct user IDs of elite users who started yelping in a specific year
@app.get("/v1/public_review_platform/elite_users_by_year", operation_id="get_elite_users_by_year", summary="Retrieve a unique list of elite user identifiers who began their activity in a specified year. The operation filters users based on the year they started yelping and returns only those who have achieved elite status.")
async def get_elite_users_by_year(user_yelping_since_year: int = Query(..., description="Year the user started yelping")):
    cursor.execute("SELECT DISTINCT T2.user_id FROM Users AS T1 INNER JOIN Elite AS T2 ON T1.user_id = T2.user_id WHERE T1.user_yelping_since_year = ?", (user_yelping_since_year,))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the percentage of reviews with a specific length and star rating
@app.get("/v1/public_review_platform/review_length_percentage", operation_id="get_review_length_percentage", summary="Retrieves the percentage of reviews that have a specified length and star rating. The length of the review and the star rating are provided as input parameters. The operation calculates the percentage by dividing the count of reviews with the specified length and star rating by the total number of reviews.")
async def get_review_length_percentage(review_length: str = Query(..., description="Length of the review"), review_stars: int = Query(..., description="Star rating of the review")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN review_length = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(review_length) FROM Reviews WHERE review_stars = ?", (review_length, review_stars))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of users with a specific number of fans and average star rating
@app.get("/v1/public_review_platform/user_fans_percentage", operation_id="get_user_fans_percentage", summary="Retrieves the percentage of users who have a specified number of fans and an average star rating equal to or greater than a given value. This operation calculates the proportion of users meeting these criteria from the total number of users in the system.")
async def get_user_fans_percentage(user_fans: str = Query(..., description="Number of fans"), user_average_stars: int = Query(..., description="Average star rating of the user")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN user_fans = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(user_id) FROM Users WHERE user_average_stars >= ?", (user_fans, user_average_stars))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of tips for a specific business with a specific tip length
@app.get("/v1/public_review_platform/tip_count_by_business", operation_id="get_tip_count_by_business", summary="Retrieves the total number of tips for a particular business, filtered by a specified tip length. The business is identified by its unique ID, and the tip length is defined as the number of characters in the tip.")
async def get_tip_count_by_business(business_id: int = Query(..., description="Business ID"), tip_length: str = Query(..., description="Length of the tip")):
    cursor.execute("SELECT COUNT(business_id) FROM Tips WHERE business_id = ? AND tip_length = ?", (business_id, tip_length))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the user ID of the user with the highest average stars who started yelping earliest
@app.get("/v1/public_review_platform/top_user_by_average_stars", operation_id="get_top_user_by_average_stars", summary="Retrieves the user ID of the user with the highest average star rating who started reviewing earliest. The user's average star rating is provided as an input parameter.")
async def get_top_user_by_average_stars(user_average_stars: int = Query(..., description="Average star rating of the user")):
    cursor.execute("SELECT user_id FROM Users WHERE user_average_stars = ? ORDER BY user_yelping_since_year ASC LIMIT 1", (user_average_stars,))
    result = cursor.fetchone()
    if not result:
        return {"user_id": []}
    return {"user_id": result[0]}

# Endpoint to get the opening and closing times of businesses in a specific city with above-average review counts
@app.get("/v1/public_review_platform/business_hours_by_city", operation_id="get_business_hours_by_city", summary="Retrieve the opening and closing times of businesses in a given city that have more reviews than the city's average. The operation filters businesses based on their review count and returns the corresponding business hours.")
async def get_business_hours_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T2.opening_time, T2.closing_time FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city = ? GROUP BY T2.business_id HAVING T1.review_count > AVG(T1.review_count)", (city,))
    result = cursor.fetchall()
    if not result:
        return {"business_hours": []}
    return {"business_hours": [{"opening_time": row[0], "closing_time": row[1]} for row in result]}

# Endpoint to get the percentage of users with a specific compliment type and number of compliments
@app.get("/v1/public_review_platform/compliment_percentage", operation_id="get_compliment_percentage", summary="Retrieves the percentage of users who have received a specific type of compliment a certain number of times. This operation calculates the proportion of users with the given compliment type and count, providing insights into the distribution of compliments among users.")
async def get_compliment_percentage(compliment_type: str = Query(..., description="Type of compliment"), number_of_compliments: str = Query(..., description="Number of compliments")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.compliment_type = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.user_id) FROM Compliments AS T1 INNER JOIN Users_Compliments AS T2 ON T1.compliment_id = T2.compliment_id WHERE T2.number_of_compliments = ?", (compliment_type, number_of_compliments))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of businesses with specific attribute values
@app.get("/v1/public_review_platform/business_attribute_count", operation_id="get_business_attribute_count", summary="Retrieves the total number of businesses that possess any of the provided attribute values. The attribute values are used to filter the businesses and determine the count.")
async def get_business_attribute_count(attribute_value1: str = Query(..., description="First attribute value"), attribute_value2: str = Query(..., description="Second attribute value"), attribute_value3: str = Query(..., description="Third attribute value")):
    cursor.execute("SELECT COUNT(business_id) FROM Business_Attributes WHERE attribute_value IN (?, ?, ?)", (attribute_value1, attribute_value2, attribute_value3))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the opening and closing times of a specific business on a specific day
@app.get("/v1/public_review_platform/business_hours_by_day", operation_id="get_business_hours_by_day", summary="Retrieves the opening and closing times of a particular business for a specific day. The operation requires the business's unique identifier and the day's identifier as input parameters.")
async def get_business_hours_by_day(business_id: int = Query(..., description="Business ID"), day_id: int = Query(..., description="Day ID")):
    cursor.execute("SELECT opening_time, closing_time FROM Business_Hours WHERE business_id = ? AND day_id = ?", (business_id, day_id))
    result = cursor.fetchone()
    if not result:
        return {"business_hours": []}
    return {"business_hours": {"opening_time": result[0], "closing_time": result[1]}}

# Endpoint to get distinct cities based on review length
@app.get("/v1/public_review_platform/distinct_cities_by_review_length", operation_id="get_distinct_cities_by_review_length", summary="Retrieve a list of unique cities where businesses have received reviews of a specified length. The length can be 'Short', 'Medium', or 'Long'.")
async def get_distinct_cities_by_review_length(review_length: str = Query(..., description="Length of the review (e.g., 'Medium')")):
    cursor.execute("SELECT DISTINCT T1.city FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T2.review_length = ?", (review_length,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get closing time for a specific day and business
@app.get("/v1/public_review_platform/closing_time_by_day_and_business", operation_id="get_closing_time_by_day_and_business", summary="Retrieves the closing time for a specific business on a given day of the week. The operation requires the day of the week and the business ID as input parameters. The day of the week should be provided as a string (e.g., 'Sunday'), and the business ID should be a unique identifier for the business.")
async def get_closing_time_by_day_and_business(day_of_week: str = Query(..., description="Day of the week (e.g., 'Sunday')"), business_id: int = Query(..., description="Business ID")):
    cursor.execute("SELECT T2.closing_time FROM Days AS T1 INNER JOIN Business_Hours AS T2 ON T1.day_id = T2.day_id WHERE T1.day_of_week = ? AND T2.business_id = ?", (day_of_week, business_id))
    result = cursor.fetchall()
    if not result:
        return {"closing_times": []}
    return {"closing_times": [row[0] for row in result]}

# Endpoint to get distinct business IDs based on city and review length
@app.get("/v1/public_review_platform/distinct_business_ids_by_city_and_review_length", operation_id="get_distinct_business_ids_by_city_and_review_length", summary="Retrieves a list of unique business identifiers located in a specified city that have reviews of a certain length. The city and review length are provided as input parameters.")
async def get_distinct_business_ids_by_city_and_review_length(city: str = Query(..., description="City name (e.g., 'Phoenix')"), review_length: str = Query(..., description="Length of the review (e.g., 'Short')")):
    cursor.execute("SELECT DISTINCT T1.business_id FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ? AND T2.review_length = ?", (city, review_length))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of users based on number of compliments and user fans
@app.get("/v1/public_review_platform/user_count_by_compliments_and_fans", operation_id="get_user_count_by_compliments_and_fans", summary="Retrieves the number of users who have a certain number of compliments and a specific number of fans. The user count is determined by matching the provided number of compliments and the number of fans to the corresponding user records in the database.")
async def get_user_count_by_compliments_and_fans(number_of_compliments: str = Query(..., description="Number of compliments (e.g., 'High')"), user_fans: str = Query(..., description="User fans (e.g., 'Medium')")):
    cursor.execute("SELECT COUNT(T1.user_id) FROM Users AS T1 INNER JOIN Users_Compliments AS T2 ON T1.user_id = T2.user_id WHERE T2.number_of_compliments = ? AND T1.user_fans = ?", (number_of_compliments, user_fans))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct user IDs based on yelping since year and number of compliments
@app.get("/v1/public_review_platform/distinct_user_ids_by_yelping_since_year_and_compliments", operation_id="get_distinct_user_ids_by_yelping_since_year_and_compliments", summary="Retrieves a unique list of user IDs who have been active since a specified year and have received a certain number of compliments. The year and the number of compliments are provided as input parameters.")
async def get_distinct_user_ids_by_yelping_since_year_and_compliments(user_yelping_since_year: int = Query(..., description="Year the user started yelping (e.g., 2012)"), number_of_compliments: str = Query(..., description="Number of compliments (e.g., 'Low')")):
    cursor.execute("SELECT DISTINCT T2.user_id FROM Users AS T1 INNER JOIN Users_Compliments AS T2 ON T1.user_id = T2.user_id WHERE T1.user_yelping_since_year = ? AND T2.number_of_compliments = ?", (user_yelping_since_year, number_of_compliments))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the count of businesses based on city and attribute values
@app.get("/v1/public_review_platform/business_count_by_city_and_attribute_values", operation_id="get_business_count_by_city_and_attribute_values", summary="Retrieves the total number of businesses in a specified city that match a set of provided attribute values. The operation filters businesses based on their location and the given attribute values, returning the count of businesses that meet the criteria.")
async def get_business_count_by_city_and_attribute_values(city: str = Query(..., description="City name (e.g., 'Gilbert')"), attribute_value1: str = Query(..., description="First attribute value (e.g., 'None')"), attribute_value2: str = Query(..., description="Second attribute value (e.g., 'no')"), attribute_value3: str = Query(..., description="Third attribute value (e.g., 'false')")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Business_Attributes AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T2.city = ? AND T1.attribute_value IN (?, ?, ?)", (city, attribute_value1, attribute_value2, attribute_value3))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses based on attribute value
@app.get("/v1/public_review_platform/business_count_by_attribute_value", operation_id="get_business_count_by_attribute_value", summary="Retrieves the total number of businesses that possess a specified attribute value. The attribute value is provided as an input parameter, allowing for a targeted count of businesses with the desired attribute.")
async def get_business_count_by_attribute_value(attribute_value: str = Query(..., description="Attribute value (e.g., 'full_bar')")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business_Attributes AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T1.attribute_value = ?", (attribute_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct states based on opening time
@app.get("/v1/public_review_platform/distinct_states_by_opening_time", operation_id="get_distinct_states_by_opening_time", summary="Retrieve a unique list of states where businesses commence operations at a specific time. The operation requires the desired opening time as input, which is used to filter the results.")
async def get_distinct_states_by_opening_time(opening_time: str = Query(..., description="Opening time (e.g., '1AM')")):
    cursor.execute("SELECT DISTINCT T1.state FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id WHERE T2.opening_time = ?", (opening_time,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get the percentage of tips with a specific length and the year users started yelping
@app.get("/v1/public_review_platform/tip_length_percentage_and_yelping_year", operation_id="get_tip_length_percentage_and_yelping_year", summary="Retrieves the percentage of tips of a specified length and the year users began yelping. This operation calculates the proportion of tips with the given length relative to all tips and identifies the year users started yelping. The input parameter specifies the length of the tips to consider in the calculation.")
async def get_tip_length_percentage_and_yelping_year(tip_length: str = Query(..., description="Length of the tip (e.g., 'Medium')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.tip_length = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.tip_length), T2.user_yelping_since_year FROM Tips AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id", (tip_length,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of businesses in a specific city and their attribute values
@app.get("/v1/public_review_platform/business_percentage_and_attribute_value", operation_id="get_business_percentage_and_attribute_value", summary="Retrieves the percentage of businesses located in a specified city and their corresponding attribute values. The operation calculates the proportion of businesses in the given city relative to the total number of businesses, and then fetches the attribute values associated with these businesses.")
async def get_business_percentage_and_attribute_value(city: str = Query(..., description="City name")):
    cursor.execute("SELECT CAST(COUNT(T1.city) AS REAL) * 100 / ( SELECT COUNT(business_id) FROM Business ), T2.attribute_value FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct states of businesses with a specific closing time
@app.get("/v1/public_review_platform/distinct_states_by_closing_time", operation_id="get_distinct_states_by_closing_time", summary="Retrieve a unique list of states where businesses close at a specified time. This operation identifies the distinct states of businesses that share a common closing time, providing a geographical perspective on business hours.")
async def get_distinct_states_by_closing_time(closing_time: str = Query(..., description="Closing time of the business (e.g., '12AM')")):
    cursor.execute("SELECT DISTINCT T1.state FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id WHERE T2.closing_time = ?", (closing_time,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of businesses with a specific attribute value in a specific city
@app.get("/v1/public_review_platform/business_count_by_city_and_attribute", operation_id="get_business_count_by_city_and_attribute", summary="Retrieves the number of businesses in a specified city that possess a particular attribute. The attribute value is used to filter the businesses, providing a count of those that match the given criteria.")
async def get_business_count_by_city_and_attribute(city: str = Query(..., description="City name"), attribute_value: str = Query(..., description="Attribute value of the business (e.g., 'beer_and_wine')")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business_Attributes AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T2.city = ? AND T1.attribute_value = ?", (city, attribute_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user IDs with a specific number of compliments and the earliest yelping year
@app.get("/v1/public_review_platform/user_ids_by_compliments_and_earliest_year", operation_id="get_user_ids_by_compliments_and_earliest_year", summary="Retrieves the user IDs of users who have received a specified number of compliments and started yelping in the earliest year. The input parameter determines the number of compliments required for a user to be included in the results.")
async def get_user_ids_by_compliments_and_earliest_year(number_of_compliments: str = Query(..., description="Number of compliments (e.g., 'High')")):
    cursor.execute("SELECT T2.user_id FROM Users AS T1 INNER JOIN Users_Compliments AS T2 ON T1.user_id = T2.user_id WHERE T2.number_of_compliments = ? AND T1.user_yelping_since_year = ( SELECT MIN(user_yelping_since_year) FROM Users )", (number_of_compliments,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the business ID with the most reviews
@app.get("/v1/public_review_platform/business_with_most_reviews", operation_id="get_business_with_most_reviews", summary="Retrieves the unique identifier of the business that has received the highest number of reviews from users. This operation considers all reviews in the system and returns the business ID that has the most user reviews associated with it.")
async def get_business_with_most_reviews():
    cursor.execute("SELECT business_id FROM Reviews GROUP BY business_id ORDER BY COUNT(user_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"business_id": []}
    return {"business_id": result[0]}

# Endpoint to get the count of businesses with specific review stars and funny votes
@app.get("/v1/public_review_platform/business_count_by_review_stars_and_funny_votes", operation_id="get_business_count_by_review_stars_and_funny_votes", summary="Retrieves the number of businesses that have received a specific number of review stars and funny votes. The review stars parameter indicates the rating given to a business, while the funny votes parameter represents the number of votes a review has received for being funny.")
async def get_business_count_by_review_stars_and_funny_votes(review_stars: int = Query(..., description="Review stars (e.g., 5)"), review_votes_funny: str = Query(..., description="Funny votes (e.g., 'Uber')")):
    cursor.execute("SELECT COUNT(business_id) FROM Reviews WHERE review_stars = ? AND review_votes_funny = ?", (review_stars, review_votes_funny))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct users with specific funny votes
@app.get("/v1/public_review_platform/distinct_user_count_by_funny_votes", operation_id="get_distinct_user_count_by_funny_votes", summary="Retrieves the number of unique users who have given a specific funny vote to a review. The funny vote is provided as an input parameter.")
async def get_distinct_user_count_by_funny_votes(review_votes_funny: str = Query(..., description="Funny votes (e.g., 'Uber')")):
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM Reviews WHERE review_votes_funny = ?", (review_votes_funny,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the business ID with the longest operating hours
@app.get("/v1/public_review_platform/business_with_longest_operating_hours", operation_id="get_business_with_longest_operating_hours", summary="Retrieves the unique identifier of the business with the longest operating hours. This operation calculates the difference between the closing and opening times for each business and returns the identifier of the business with the greatest duration.")
async def get_business_with_longest_operating_hours():
    cursor.execute("SELECT business_id FROM Business_Hours ORDER BY closing_time - opening_time LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"business_id": []}
    return {"business_id": result[0]}

# Endpoint to get tip lengths and their total likes for a specific category
@app.get("/v1/public_review_platform/categories/tip_lengths_and_likes", operation_id="get_tip_lengths_and_likes", summary="Retrieves the distribution of tip lengths and their corresponding total likes for a specified category. This operation aggregates data from the Tips table, filtered by the provided category name, to provide a comprehensive overview of tip length preferences and their associated popularity.")
async def get_tip_lengths_and_likes(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT T3.tip_length, SUM(T3.likes) AS likes FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Tips AS T3 ON T2.business_id = T3.business_id WHERE T1.category_name = ? GROUP BY T3.tip_length", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"tip_lengths_and_likes": []}
    return {"tip_lengths_and_likes": [{"tip_length": row[0], "likes": row[1]} for row in result]}

# Endpoint to get the count of user IDs and their average stars based on review votes
@app.get("/v1/public_review_platform/reviews/user_ids_and_average_stars", operation_id="get_user_ids_and_average_stars", summary="Retrieves the total number of unique user IDs and their corresponding average star ratings, based on the specified review votes. The review votes criteria include 'funny', 'useful', and 'cool' votes, which can be used to filter the results.")
async def get_user_ids_and_average_stars(review_votes_funny: str = Query(..., description="Review votes funny"), review_votes_useful: str = Query(..., description="Review votes useful"), review_votes_cool: str = Query(..., description="Review votes cool")):
    cursor.execute("SELECT COUNT(T2.user_id) AS USER_IDS, T2.user_average_stars FROM Reviews AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id WHERE T1.review_votes_funny = ? AND T1.review_votes_useful = ? AND T1.review_votes_cool = ?", (review_votes_funny, review_votes_useful, review_votes_cool))
    result = cursor.fetchall()
    if not result:
        return {"user_ids_and_average_stars": []}
    return {"user_ids_and_average_stars": [{"user_ids": row[0], "average_stars": row[1]} for row in result]}

# Endpoint to get the ratio of high-rated to low-rated businesses
@app.get("/v1/public_review_platform/business/high_low_rating_ratio", operation_id="get_high_low_rating_ratio", summary="Retrieves the ratio of businesses with high ratings to those with low ratings. The high and low rating ranges are defined by the provided input parameters. The calculation considers businesses that are open, as determined by their business hours.")
async def get_high_low_rating_ratio(min_high_rating: float = Query(..., description="Minimum high rating"), max_high_rating: float = Query(..., description="Maximum high rating"), min_low_rating: float = Query(..., description="Minimum low rating"), max_low_rating: float = Query(..., description="Maximum low rating")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.stars BETWEEN ? AND ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.stars BETWEEN ? AND ? THEN 1 ELSE 0 END) AS ratio FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id", (min_high_rating, max_high_rating, min_low_rating, max_low_rating))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get top business IDs and their categories based on the number of user reviews
@app.get("/v1/public_review_platform/reviews/top_businesses_by_reviews", operation_id="get_top_businesses_by_reviews", summary="Retrieve a list of top-rated businesses and their respective categories, ranked by the number of user reviews. The response is limited to the specified number of results.")
async def get_top_businesses_by_reviews(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.business_id, T3.category_name FROM Reviews AS T1 INNER JOIN Business_categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id GROUP BY T2.business_id ORDER BY COUNT(T1.user_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"top_businesses": []}
    return {"top_businesses": [{"business_id": row[0], "category_name": row[1]} for row in result]}

# Endpoint to get the count of businesses with average review stars below a threshold in a specific state
@app.get("/v1/public_review_platform/business/count_low_rated_businesses", operation_id="get_count_low_rated_businesses", summary="Retrieves the number of businesses in a specified state that have an average review rating below a given threshold. The count is determined by aggregating businesses with distinct IDs that meet the criteria.")
async def get_count_low_rated_businesses(state: str = Query(..., description="State abbreviation"), threshold: float = Query(..., description="Threshold for average review stars")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE business_id IN ( SELECT DISTINCT T1.business_id FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T1.state = ? GROUP BY T1.business_id HAVING SUM(T2.review_stars) / COUNT(T2.user_id) < ? )", (state, threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of non-elite users
@app.get("/v1/public_review_platform/users/percentage_non_elite", operation_id="get_percentage_non_elite", summary="Retrieves the percentage of users who are not considered elite. This operation calculates the ratio of non-elite users to the total number of users in the system. The result is expressed as a percentage.")
async def get_percentage_non_elite():
    cursor.execute("SELECT CAST((( SELECT COUNT(user_id) FROM Users ) - ( SELECT COUNT(DISTINCT user_id) FROM Elite )) AS REAL) * 100 / ( SELECT COUNT(user_id) FROM Users )")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct compliment types for users with a specific number of fans
@app.get("/v1/public_review_platform/users/distinct_compliment_types", operation_id="get_distinct_compliment_types", summary="Retrieves the unique types of compliments received by users who have a specified number of fans. This operation does not return the compliments themselves, but rather the distinct categories or types of compliments that these users have received.")
async def get_distinct_compliment_types(user_fans: str = Query(..., description="Number of user fans")):
    cursor.execute("SELECT DISTINCT T3.compliment_type FROM Users AS T1 INNER JOIN Users_Compliments AS T2 ON T1.user_id = T2.user_id INNER JOIN Compliments AS T3 ON T2.compliment_id = T3.compliment_id WHERE T1.user_fans = ?", (user_fans,))
    result = cursor.fetchall()
    if not result:
        return {"compliment_types": []}
    return {"compliment_types": [row[0] for row in result]}

# Endpoint to get the average yelping years for elite users with a specific number of fans
@app.get("/v1/public_review_platform/users/average_yelping_years_elite", operation_id="get_average_yelping_years_elite", summary="Retrieves the average number of years that elite users with a specific number of fans have been yelping. The calculation is based on the difference between the current year and the user's yelping start year, summed up and averaged across all eligible users.")
async def get_average_yelping_years_elite(user_fans: str = Query(..., description="Number of user fans")):
    cursor.execute("SELECT CAST(SUM(T2.year_id - T1.user_yelping_since_year) AS REAL) / COUNT(T1.user_id) FROM Users AS T1 INNER JOIN Elite AS T2 ON T1.user_id = T2.user_id WHERE T1.user_fans = ?", (user_fans,))
    result = cursor.fetchone()
    if not result:
        return {"average_yelping_years": []}
    return {"average_yelping_years": result[0]}

# Endpoint to get the average yelping years for all elite users
@app.get("/v1/public_review_platform/users/average_yelping_years_all_elite", operation_id="get_average_yelping_years_all_elite", summary="Retrieves the average number of years that all elite users have been active on the platform. This is calculated by summing the difference between the current year and the user's yelping start year for each elite user, then dividing by the total number of elite users.")
async def get_average_yelping_years_all_elite():
    cursor.execute("SELECT CAST(SUM(T2.year_id - T1.user_yelping_since_year) AS REAL) / COUNT(T1.user_id) FROM Users AS T1 INNER JOIN Elite AS T2 ON T1.user_id = T2.user_id")
    result = cursor.fetchone()
    if not result:
        return {"average_yelping_years": []}
    return {"average_yelping_years": result[0]}

# Endpoint to get the percentage of active businesses in a given city
@app.get("/v1/public_review_platform/active_business_percentage", operation_id="get_active_business_percentage", summary="Retrieves the percentage of active businesses in a specified city. The operation calculates this value by considering the total number of businesses and the number of active businesses in the given city. The city is identified by the provided city name.")
async def get_active_business_percentage(city: str = Query(..., description="City name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.active = 'true' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.business_id) AS ACT FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct category names for active businesses in a given city with a specific opening time
@app.get("/v1/public_review_platform/distinct_category_names", operation_id="get_distinct_category_names", summary="Retrieves a list of up to three distinct category names for active businesses in the specified city that open at or after the provided time. The category names are derived from the businesses' associated categories.")
async def get_distinct_category_names(city: str = Query(..., description="City name"), opening_time: str = Query(..., description="Opening time in 'HH:MM AM/PM' format")):
    cursor.execute("SELECT DISTINCT T4.category_name FROM Business_Hours AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Business_Categories AS T3 ON T2.business_id = T3.business_id INNER JOIN Categories AS T4 ON T3.category_id = T4.category_id WHERE T2.active = 'true' AND T2.city = ? AND T1.opening_time >= ? LIMIT 3", (city, opening_time))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get the user ID with the most reviews for a specific attribute
@app.get("/v1/public_review_platform/top_user_by_attribute", operation_id="get_top_user_by_attribute", summary="Retrieves the user ID who has submitted the highest number of reviews for a specific attribute. The attribute is identified by the provided attribute_name parameter.")
async def get_top_user_by_attribute(attribute_name: str = Query(..., description="Attribute name")):
    cursor.execute("SELECT T3.user_id FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id INNER JOIN Reviews AS T3 ON T2.business_id = T3.business_id WHERE T1.attribute_name = ? GROUP BY T3.user_id ORDER BY COUNT(T2.business_id) DESC LIMIT 1", (attribute_name,))
    result = cursor.fetchone()
    if not result:
        return {"user_id": []}
    return {"user_id": result[0]}

# Endpoint to get the average user ID for active businesses with a specific total operating time
@app.get("/v1/public_review_platform/avg_user_id_by_operating_time", operation_id="get_avg_user_id_by_operating_time", summary="Get the average user ID for active businesses with a specific total operating time")
async def get_avg_user_id_by_operating_time(total_operating_time: int = Query(..., description="Total operating time in minutes")):
    cursor.execute("SELECT AVG(T3.user_id) FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Reviews AS T3 ON T1.business_id = T3.business_id WHERE T1.active = 'true' GROUP BY T2.closing_time - T2.opening_time HAVING SUM(T2.closing_time - T2.opening_time) < ?", (total_operating_time,))
    result = cursor.fetchone()
    if not result:
        return {"avg_user_id": []}
    return {"avg_user_id": result[0]}

# Endpoint to get distinct business IDs based on opening and closing times
@app.get("/v1/public_review_platform/business_ids_by_hours", operation_id="get_business_ids_by_hours", summary="Retrieve a unique set of business identifiers that operate during the specified opening and closing hours. The operation filters businesses based on their exact opening and closing times, providing a list of distinct business IDs that match the given criteria.")
async def get_business_ids_by_hours(opening_time: str = Query(..., description="Opening time in 'HH:MM AM/PM' format"), closing_time: str = Query(..., description="Closing time in 'HH:MM AM/PM' format")):
    cursor.execute("SELECT DISTINCT business_id FROM Business_Hours WHERE opening_time = ? AND closing_time = ?", (opening_time, closing_time))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get distinct business IDs based on day ID and opening time
@app.get("/v1/public_review_platform/business_ids_by_day_and_time", operation_id="get_business_ids_by_day_and_time", summary="Retrieves a unique set of business identifiers that are open on a specific day and time. The day is identified by a day ID, and the opening time is provided in 'HH:MM AM/PM' format.")
async def get_business_ids_by_day_and_time(day_id: int = Query(..., description="Day ID"), opening_time: str = Query(..., description="Opening time in 'HH:MM AM/PM' format")):
    cursor.execute("SELECT DISTINCT business_id FROM Business_Hours WHERE day_id = ? AND opening_time = ?", (day_id, opening_time))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get distinct day IDs based on opening and closing times
@app.get("/v1/public_review_platform/day_ids_by_hours", operation_id="get_day_ids_by_hours", summary="Retrieves unique day IDs for businesses that open and close at the specified times. The opening and closing times must be provided in 'HH:MM AM/PM' format.")
async def get_day_ids_by_hours(opening_time: str = Query(..., description="Opening time in 'HH:MM AM/PM' format"), closing_time: str = Query(..., description="Closing time in 'HH:MM AM/PM' format")):
    cursor.execute("SELECT DISTINCT day_id FROM Business_Hours WHERE opening_time = ? AND closing_time = ?", (opening_time, closing_time))
    result = cursor.fetchall()
    if not result:
        return {"day_ids": []}
    return {"day_ids": [row[0] for row in result]}

# Endpoint to get the count of businesses with a star rating greater than a specified value
@app.get("/v1/public_review_platform/business_count_by_stars", operation_id="get_business_count_by_stars", summary="Retrieves the total number of businesses that have a star rating higher than the specified minimum value. This operation allows you to understand the distribution of businesses based on their star ratings.")
async def get_business_count_by_stars(min_stars: float = Query(..., description="Minimum star rating")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE stars > ?", (min_stars,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct category names for businesses open on a specific day of the week
@app.get("/v1/public_review_platform/category_names_by_day", operation_id="get_category_names_by_day", summary="Retrieve a unique set of category names for businesses that are open on a specific day of the week. The day of the week is provided as an input parameter.")
async def get_category_names_by_day(day_of_week: str = Query(..., description="Day of the week")):
    cursor.execute("SELECT DISTINCT T1.category_name FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business_Hours AS T3 ON T2.business_id = T3.business_id INNER JOIN Days AS T4 ON T3.day_id = T4.day_id WHERE T4.day_of_week = ? AND T3.opening_time <> ''", (day_of_week,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get distinct days of the week for a specific category
@app.get("/v1/public_review_platform/days_by_category", operation_id="get_days_by_category", summary="Retrieves the unique days of the week associated with a specific category. This operation returns a list of distinct days when businesses in the given category are open, providing insights into their operational days.")
async def get_days_by_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT DISTINCT T4.day_of_week FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business_Hours AS T3 ON T1.business_id = T3.business_id INNER JOIN Days AS T4 ON T3.day_id = T4.day_id WHERE T2.category_name = ?", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"days": []}
    return {"days": [row[0] for row in result]}

# Endpoint to get distinct opening times and day IDs for businesses in a specific category
@app.get("/v1/public_review_platform/business_opening_times_by_category", operation_id="get_business_opening_times_by_category", summary="Retrieve unique opening times and corresponding day identifiers for businesses categorized under a specific category name.")
async def get_business_opening_times_by_category(category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT DISTINCT T3.opening_time, T3.day_id FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business_Hours AS T3 ON T1.business_id = T3.business_id INNER JOIN Days AS T4 ON T3.day_id = T4.day_id WHERE T2.category_name = ?", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"opening_times": []}
    return {"opening_times": result}

# Endpoint to get the category with the most business hours
@app.get("/v1/public_review_platform/top_category_by_business_hours", operation_id="get_top_category_by_business_hours", summary="Retrieves the category with the longest cumulative business hours. This operation identifies the category with the highest number of business hours by aggregating the total hours across all businesses within that category. The result is the category name with the most extensive business hours.")
async def get_top_category_by_business_hours():
    cursor.execute("SELECT T2.category_name FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business_Hours AS T3 ON T1.business_id = T3.business_id INNER JOIN Days AS T4 ON T3.day_id = T4.day_id GROUP BY T2.category_name ORDER BY COUNT(T3.day_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"category_name": []}
    return {"category_name": result[0]}

# Endpoint to get business IDs and closing times for a specific category and day of the week
@app.get("/v1/public_review_platform/business_closing_times_by_category_and_day", operation_id="get_business_closing_times_by_category_and_day", summary="Retrieves the business IDs and their respective closing times for a specified business category and day of the week. The category and day of the week are provided as input parameters.")
async def get_business_closing_times_by_category_and_day(category_name: str = Query(..., description="Category name of the business"), day_of_week: str = Query(..., description="Day of the week")):
    cursor.execute("SELECT T1.business_id, T3.closing_time FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business_Hours AS T3 ON T1.business_id = T3.business_id INNER JOIN Days AS T4 ON T3.day_id = T4.day_id WHERE T2.category_name = ? AND T4.day_of_week = ?", (category_name, day_of_week))
    result = cursor.fetchall()
    if not result:
        return {"business_closing_times": []}
    return {"business_closing_times": result}

# Endpoint to get the count of businesses in a specific category with stars below a certain threshold
@app.get("/v1/public_review_platform/business_count_by_category_and_stars", operation_id="get_business_count_by_category_and_stars", summary="Retrieves the number of businesses in a specified category with a star rating below a given threshold. The category is identified by its name, and the maximum star rating is provided as an input parameter.")
async def get_business_count_by_category_and_stars(category_name: str = Query(..., description="Category name of the business"), max_stars: int = Query(..., description="Maximum star rating")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T3.category_name = ? AND T1.stars < ?", (category_name, max_stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct business IDs based on activity status, opening time, and closing time
@app.get("/v1/public_review_platform/business_ids_by_activity_and_hours", operation_id="get_business_ids_by_activity_and_hours", summary="Retrieve a unique list of business identifiers that are currently active and operating within the specified opening and closing hours. The selection is based on the provided activity status and the business's opening and closing times.")
async def get_business_ids_by_activity_and_hours(active: str = Query(..., description="Activity status of the business"), opening_time: str = Query(..., description="Opening time of the business"), closing_time: str = Query(..., description="Closing time of the business")):
    cursor.execute("SELECT DISTINCT T4.business_id FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business_Hours AS T3 ON T2.business_id = T3.business_id INNER JOIN Business AS T4 ON T3.business_id = T4.business_id WHERE T4.active = ? AND T3.opening_time = ? AND T3.closing_time = ?", (active, opening_time, closing_time))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": result}

# Endpoint to get the count of businesses in a specific category with the maximum star rating
@app.get("/v1/public_review_platform/business_count_by_category_and_max_stars", operation_id="get_business_count_by_category_and_max_stars", summary="Retrieves the total number of businesses in a specified category that have the highest star rating. The category is identified by its name.")
async def get_business_count_by_category_and_max_stars(category_name: str = Query(..., description="Category name of the business")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id WHERE T2.category_name = ? AND T3.stars = ( SELECT MAX(stars) FROM Business )", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct categories for businesses with a specific review count and star rating
@app.get("/v1/public_review_platform/category_count_by_review_count_and_stars", operation_id="get_category_count_by_review_count_and_stars", summary="Retrieve the number of unique categories associated with businesses that have a specified review count and a star rating above a certain threshold. This operation provides insights into the distribution of categories based on the review count and minimum star rating criteria.")
async def get_category_count_by_review_count_and_stars(review_count: str = Query(..., description="Review count of the business"), min_stars: int = Query(..., description="Minimum star rating")):
    cursor.execute("SELECT COUNT(DISTINCT T1.category_id) FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T3.review_count = ? AND T3.stars > ?", (review_count, min_stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs based on category, opening time criteria, and time format
@app.get("/v1/public_review_platform/business_ids_by_category_and_opening_time", operation_id="get_business_ids_by_category_and_opening_time", summary="Retrieves business IDs that belong to a specified category and open before a certain hour in the given time format. The category is identified by its name, and the opening time is compared to the maximum hour in the provided format.")
async def get_business_ids_by_category_and_opening_time(category_name: str = Query(..., description="Category name of the business"), max_hour: int = Query(..., description="Maximum hour for opening time"), time_format: str = Query(..., description="Time format for opening time (e.g., '%AM')")):
    cursor.execute("SELECT T1.business_id FROM Business_Hours AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T3.category_name = ? AND SUBSTR(T1.opening_time, -4, 2) * 1 < ? AND T1.opening_time LIKE ?", (category_name, max_hour, time_format))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": result}

# Endpoint to get the count of distinct businesses based on activity status, state, and opening time
@app.get("/v1/public_review_platform/business_count_by_activity_state_and_opening_time", operation_id="get_business_count_by_activity_state_and_opening_time", summary="Retrieves the number of unique businesses that are currently active, located in a specific state, and open after a certain time. The activity status, state, and opening time are provided as input parameters.")
async def get_business_count_by_activity_state_and_opening_time(active: str = Query(..., description="Activity status of the business"), state: str = Query(..., description="State of the business"), opening_time: str = Query(..., description="Opening time of the business")):
    cursor.execute("SELECT COUNT(DISTINCT T2.business_id) FROM Business_Hours AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Business_Categories AS T3 ON T2.business_id = T3.business_id INNER JOIN Categories AS T4 ON T3.category_id = T4.category_id WHERE T2.active = ? AND T2.state = ? AND T1.opening_time > ?", (active, state, opening_time))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get category names based on user ID
@app.get("/v1/public_review_platform/category_names_by_user_id", operation_id="get_category_names_by_user_id", summary="Retrieves the names of categories associated with a specific user, based on their user ID. This operation returns a list of category names that the user has interacted with, as determined by their activity in the 'Tips' table. The user ID is a required input parameter.")
async def get_category_names_by_user_id(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T1.category_name FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Tips AS T3 ON T2.business_id = T3.business_id WHERE T3.user_id = ?", (user_id,))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": result}

# Endpoint to get business IDs and the percentage difference in star ratings for a specific category
@app.get("/v1/public_review_platform/business_star_rating_difference", operation_id="get_business_star_rating_difference", summary="Retrieves a list of business IDs and the corresponding percentage difference in star ratings for a specified category. The percentage difference is calculated by subtracting the count of businesses with star ratings below the provided lower threshold from the count of businesses with star ratings above the provided higher threshold, then dividing by the total count of businesses in the category. The result is multiplied by 100 to obtain a percentage.")
async def get_business_star_rating_difference(low_stars: int = Query(..., description="Lower threshold for star ratings"), high_stars: int = Query(..., description="Higher threshold for star ratings"), category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T3.business_id, CAST((( SELECT COUNT(business_id) FROM Business WHERE stars < ? ) - ( SELECT COUNT(business_id) FROM Business WHERE stars > ? )) AS REAL) * 100 / ( SELECT COUNT(stars) FROM Business ) FROM Business_Categories AS T1 INNER JOIN Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T1.business_id = T3.business_id WHERE T2.category_name = ?", (low_stars, high_stars, category_name))
    result = cursor.fetchall()
    if not result:
        return {"business_star_rating_difference": []}
    return {"business_star_rating_difference": result}

# Endpoint to get the percentage of businesses in a specific category
@app.get("/v1/public_review_platform/business_category_percentage", operation_id="get_business_category_percentage", summary="Retrieves the percentage of businesses that belong to a specified category. This operation calculates the proportion of businesses in the given category by comparing the count of businesses in that category to the total number of businesses. The category is identified by its name.")
async def get_business_category_percentage(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.category_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.category_name) FROM Business_Categories AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T1.category_id = T3.category_id", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"business_category_percentage": []}
    return {"business_category_percentage": result[0]}

# Endpoint to get the count of businesses by review count categories in a specific city
@app.get("/v1/public_review_platform/business_review_count_by_city", operation_id="get_business_review_count_by_city", summary="Retrieves the count of businesses in a specific city, categorized by their review count. The categories include high, medium, and low review counts. The active status of the businesses is also considered.")
async def get_business_review_count_by_city(high_review_count: str = Query(..., description="High review count category"), medium_review_count: str = Query(..., description="Medium review count category"), low_review_count: str = Query(..., description="Low review count category"), city: str = Query(..., description="City name"), active: str = Query(..., description="Active status of the business")):
    cursor.execute("SELECT SUM(CASE WHEN review_count = ? THEN 1 ELSE 0 END) AS high , SUM(CASE WHEN review_count = ? THEN 1 ELSE 0 END) AS Medium , SUM(CASE WHEN review_count = ? THEN 1 ELSE 0 END) AS low FROM Business WHERE city = ? AND active = ?", (high_review_count, medium_review_count, low_review_count, city, active))
    result = cursor.fetchone()
    if not result:
        return {"business_review_count": []}
    return {"business_review_count": result}

# Endpoint to get the average user ID for users who started yelping within a specific year range
@app.get("/v1/public_review_platform/average_user_id_by_year_range", operation_id="get_average_user_id_by_year_range", summary="Retrieves the average user ID for users who began using the platform within a specified year range. The start and end years of the range are provided as input parameters, allowing for a customizable time frame.")
async def get_average_user_id_by_year_range(start_year: int = Query(..., description="Start year for yelping since"), end_year: int = Query(..., description="End year for yelping since")):
    cursor.execute("SELECT AVG(user_id) FROM Users WHERE user_yelping_since_year >= ? AND user_yelping_since_year <= ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"average_user_id": []}
    return {"average_user_id": result[0]}

# Endpoint to get the ratio of active to inactive businesses for a specific review count category
@app.get("/v1/public_review_platform/active_inactive_business_ratio", operation_id="get_active_inactive_business_ratio", summary="Retrieves the ratio of active to inactive businesses for a specified review count category. The active and inactive statuses of businesses are used to calculate the ratio, providing insights into the proportion of active and inactive businesses within the given review count category.")
async def get_active_inactive_business_ratio(active_status: str = Query(..., description="Active status of the business"), inactive_status: str = Query(..., description="Inactive status of the business"), review_count: str = Query(..., description="Review count category")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN active = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN active = ? THEN 1 ELSE 0 END) AS radio FROM Business WHERE review_count = ?", (active_status, inactive_status, review_count))
    result = cursor.fetchone()
    if not result:
        return {"active_inactive_ratio": []}
    return {"active_inactive_ratio": result[0]}

# Endpoint to get category IDs and names that match a specific pattern with a limit
@app.get("/v1/public_review_platform/category_ids_names_by_pattern", operation_id="get_category_ids_names_by_pattern", summary="Retrieves a limited number of category IDs and their corresponding names that match a specified pattern. The pattern is used to filter category names, and the limit parameter determines the maximum number of results returned.")
async def get_category_ids_names_by_pattern(pattern: str = Query(..., description="Pattern to match category names"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT category_id, category_name FROM Categories WHERE category_name LIKE ? LIMIT ?", (pattern, limit))
    result = cursor.fetchall()
    if not result:
        return {"category_ids_names": []}
    return {"category_ids_names": result}

# Endpoint to get user IDs and review stars for a specific business ID and review length
@app.get("/v1/public_review_platform/user_review_stars_by_business", operation_id="get_user_review_stars_by_business", summary="Retrieves a list of user IDs and their corresponding review star ratings for a specific business, filtered by a selected review length category. This operation allows you to analyze user feedback for a particular business based on the length of their reviews.")
async def get_user_review_stars_by_business(business_id: int = Query(..., description="Business ID"), review_length: str = Query(..., description="Review length category")):
    cursor.execute("SELECT user_id, review_stars FROM Reviews WHERE business_id = ? AND review_length = ?", (business_id, review_length))
    result = cursor.fetchall()
    if not result:
        return {"user_review_stars": []}
    return {"user_review_stars": result}

# Endpoint to get business IDs and active status for a specific category name
@app.get("/v1/public_review_platform/business_active_status_by_category", operation_id="get_business_active_status_by_category", summary="Retrieves a list of business IDs and their active statuses for a specified category. The category is identified by its name.")
async def get_business_active_status_by_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T2.business_id, T3.active FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T1.category_name = ?", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"business_active_status": []}
    return {"business_active_status": result}

# Endpoint to get user details based on business ID and review stars
@app.get("/v1/public_review_platform/user_details_by_business_review_stars", operation_id="get_user_details", summary="Retrieves details of users who have reviewed a specific business with a certain star rating. The response includes the user's ID and the year they started using the platform. To filter results, provide the business ID and the desired review star rating.")
async def get_user_details(business_id: int = Query(..., description="Business ID"), review_stars: int = Query(..., description="Review stars")):
    cursor.execute("SELECT T2.user_id, T2.user_yelping_since_year FROM Reviews AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id WHERE T1.business_id = ? AND T1.review_stars = ?", (business_id, review_stars))
    result = cursor.fetchall()
    if not result:
        return {"user_details": []}
    return {"user_details": result}

# Endpoint to get user details based on compliment type and number of compliments
@app.get("/v1/public_review_platform/user_details_by_compliment_type_and_number", operation_id="get_user_details_by_compliment", summary="Retrieves the user ID and the year they started using the platform for up to five users who have received a specific type of compliment a certain number of times. The compliment type and the number of compliments are provided as input parameters.")
async def get_user_details_by_compliment(compliment_type: str = Query(..., description="Compliment type"), number_of_compliments: str = Query(..., description="Number of compliments")):
    cursor.execute("SELECT T3.user_id, T3.user_yelping_since_year FROM Compliments AS T1 INNER JOIN Users_Compliments AS T2 ON T1.compliment_id = T2.compliment_id INNER JOIN Users AS T3 ON T2.user_id = T3.user_id WHERE T1.compliment_type = ? AND T2.number_of_compliments = ? LIMIT 5", (compliment_type, number_of_compliments))
    result = cursor.fetchall()
    if not result:
        return {"user_details": []}
    return {"user_details": result}

# Endpoint to get the top tip based on likes
@app.get("/v1/public_review_platform/top_tip_by_likes", operation_id="get_top_tip", summary="Retrieves the top-rated tip based on the highest number of likes. This operation fetches the user ID, business ID, and review length associated with the top tip. The tip is determined by the number of likes it has received, with the tip having the most likes being returned.")
async def get_top_tip():
    cursor.execute("SELECT T1.user_id, T1.business_id, T2.review_length FROM Tips AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id ORDER BY T1.likes DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"top_tip": []}
    return {"top_tip": result}

# Endpoint to get user compliments based on year range and compliment type
@app.get("/v1/public_review_platform/user_compliments_by_year_range_and_type", operation_id="get_user_compliments", summary="Retrieves the number of compliments received by users within a specified year range and compliment type. The operation filters users based on their membership in the 'Elite' group and returns the user ID along with the total count of compliments for each user that meet the specified criteria.")
async def get_user_compliments(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year"), compliment_type: str = Query(..., description="Compliment type")):
    cursor.execute("SELECT T2.user_id, T2.number_of_compliments FROM Compliments AS T1 INNER JOIN Users_Compliments AS T2 ON T1.compliment_id = T2.compliment_id INNER JOIN Elite AS T3 ON T2.user_id = T3.user_id WHERE T3.year_id BETWEEN ? AND ? AND T1.compliment_type = ?", (start_year, end_year, compliment_type))
    result = cursor.fetchall()
    if not result:
        return {"user_compliments": []}
    return {"user_compliments": result}

# Endpoint to get the percentage of businesses open during specific hours on a given day
@app.get("/v1/public_review_platform/percentage_businesses_open_by_hours_and_day", operation_id="get_percentage_businesses_open", summary="Retrieves the percentage of businesses that are open during the specified hours on a given day of the week. The operation calculates this percentage by summing the number of businesses open during the provided opening and closing times, then dividing by the total number of businesses. The day of the week is also considered in the calculation.")
async def get_percentage_businesses_open(opening_time: str = Query(..., description="Opening time in 'HHAM' format"), closing_time: str = Query(..., description="Closing time in 'HHPM' format"), day_of_week: str = Query(..., description="Day of the week")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.opening_time = ? AND T2.closing_time = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.day_id) FROM Days AS T1 INNER JOIN Business_Hours AS T2 ON T1.day_id = T2.day_id WHERE T1.day_of_week = ?", (opening_time, closing_time, day_of_week))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get business IDs and days of the week based on city and active status
@app.get("/v1/public_review_platform/business_days_by_city_and_active_status", operation_id="get_business_days", summary="Retrieves the business identifiers and corresponding days of the week for businesses in a specified city that match a given active status. The active status can be used to filter results based on whether a business is currently active or not.")
async def get_business_days(city: str = Query(..., description="City name"), active: str = Query(..., description="Active status")):
    cursor.execute("SELECT T2.business_id, T3.day_of_week FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city = ? AND T1.active = ?", (city, active))
    result = cursor.fetchall()
    if not result:
        return {"business_days": []}
    return {"business_days": result}

# Endpoint to get user IDs and review lengths based on yelping since year and average stars
@app.get("/v1/public_review_platform/user_review_lengths_by_year_and_stars", operation_id="get_user_review_lengths", summary="Retrieves a list of user IDs and their corresponding review lengths for users who started yelping in a specified year and have a given average star rating. This operation is useful for analyzing review patterns based on user activity and rating behavior.")
async def get_user_review_lengths(user_yelping_since_year: int = Query(..., description="Year the user started yelping"), user_average_stars: int = Query(..., description="Average stars given by the user")):
    cursor.execute("SELECT T2.user_id, T2.review_length FROM Users AS T1 INNER JOIN Reviews AS T2 ON T1.user_id = T2.user_id WHERE T1.user_yelping_since_year = ? AND T1.user_average_stars = ?", (user_yelping_since_year, user_average_stars))
    result = cursor.fetchall()
    if not result:
        return {"user_review_lengths": []}
    return {"user_review_lengths": result}

# Endpoint to get distinct business IDs and cities based on review stars and percentage of users
@app.get("/v1/public_review_platform/distinct_businesses_by_review_stars_and_user_percentage", operation_id="get_distinct_businesses", summary="Retrieves unique business identifiers and their respective cities where the businesses have received a minimum specified number of review stars and the percentage of users who have reviewed these businesses exceeds a given threshold. The threshold for the percentage of users is set to 65%.")
async def get_distinct_businesses(review_stars: int = Query(..., description="Minimum review stars"), user_percentage: float = Query(..., description="Percentage of users")):
    cursor.execute("SELECT DISTINCT T2.business_id, T2.city FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T1.review_stars >= ? AND ( SELECT CAST(( SELECT COUNT(DISTINCT T1.user_id) FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T1.review_stars >= ? ) AS REAL) * 100 / ( SELECT COUNT(user_id) FROM Users ) > ? )", (review_stars, review_stars, user_percentage))
    result = cursor.fetchall()
    if not result:
        return {"distinct_businesses": []}
    return {"distinct_businesses": result}

# Endpoint to get the difference in the number of active businesses between two cities
@app.get("/v1/public_review_platform/business_diff_by_cities", operation_id="get_business_diff", summary="Retrieve the difference in the count of active businesses between two specified cities. The operation compares the total number of active businesses in the first city with that of the second city, and returns the difference.")
async def get_business_diff(city1: str = Query(..., description="First city"), city2: str = Query(..., description="Second city"), active: str = Query(..., description="Active status")):
    cursor.execute("SELECT SUM(CASE WHEN city = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN city = ? THEN 1 ELSE 0 END) AS diff FROM Business WHERE active = ?", (city1, city2, active))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get the total number of likes for tips by users who started yelping in a specific year
@app.get("/v1/public_review_platform/total_likes_by_yelping_year", operation_id="get_total_likes", summary="Retrieves the cumulative number of likes for tips authored by users who commenced their Yelping activities in a specified year.")
async def get_total_likes(user_yelping_since_year: int = Query(..., description="Year the user started yelping")):
    cursor.execute("SELECT SUM(T2.likes) FROM Users AS T1 INNER JOIN Tips AS T2 ON T1.user_id = T2.user_id WHERE T1.user_yelping_since_year = ?", (user_yelping_since_year,))
    result = cursor.fetchone()
    if not result:
        return {"total_likes": []}
    return {"total_likes": result[0]}

# Endpoint to get the most common tip length for users with a specific average star rating
@app.get("/v1/public_review_platform/most_common_tip_length", operation_id="get_most_common_tip_length", summary="Retrieves the most frequently occurring tip length for users who have a specified average star rating. The tip length is determined by analyzing the tips provided by users with the given average star rating. The result is the tip length that appears most frequently among these users.")
async def get_most_common_tip_length(user_average_stars: int = Query(..., description="Average star rating of the user")):
    cursor.execute("SELECT T2.tip_length FROM Users AS T1 INNER JOIN Tips AS T2 ON T1.user_id = T2.user_id WHERE T1.user_average_stars = ? GROUP BY T2.tip_length ORDER BY COUNT(T2.tip_length) DESC LIMIT 1", (user_average_stars,))
    result = cursor.fetchone()
    if not result:
        return {"tip_length": []}
    return {"tip_length": result[0]}

# Endpoint to get the total number of likes for tips in a specific city
@app.get("/v1/public_review_platform/total_likes_by_city", operation_id="get_total_likes_by_city", summary="Retrieves the cumulative number of likes for all tips associated with businesses in a specified city. The city is identified by its name.")
async def get_total_likes_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT SUM(T2.likes) AS likes FROM Business AS T1 INNER JOIN Tips AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"likes": []}
    return {"likes": result[0]}

# Endpoint to get distinct states for businesses with tips of a specific length
@app.get("/v1/public_review_platform/distinct_states_by_tip_length", operation_id="get_distinct_states_by_tip_length", summary="Retrieve a unique list of states where businesses with tips of a specified length are located. The operation filters businesses based on the length of their tips and returns the distinct states in which these businesses operate.")
async def get_distinct_states_by_tip_length(tip_length: str = Query(..., description="Length of the tip")):
    cursor.execute("SELECT DISTINCT T1.state FROM Business AS T1 INNER JOIN Tips AS T2 ON T1.business_id = T2.business_id WHERE T2.tip_length = ?", (tip_length,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get the total operating hours for businesses in a specific city and state
@app.get("/v1/public_review_platform/total_operating_hours", operation_id="get_total_operating_hours", summary="Retrieves the cumulative operating hours for businesses in a specified city and state. The calculation is based on the difference between closing and opening times for each business, summed up across all businesses in the given location.")
async def get_total_operating_hours(city: str = Query(..., description="City name"), state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT SUM(T2.closing_time - T2.opening_time) FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ? AND T1.state = ?", (city, state))
    result = cursor.fetchone()
    if not result:
        return {"operating_hours": []}
    return {"operating_hours": result[0]}

# Endpoint to get the difference in day IDs for businesses in a specific state
@app.get("/v1/public_review_platform/day_id_difference", operation_id="get_day_id_difference", summary="Retrieves the difference in day IDs for businesses located in a specified state. This operation calculates the disparity between the day IDs of businesses and their corresponding business hours, providing insights into the operational timelines of businesses within the given state.")
async def get_day_id_difference(state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT T3.day_id - T2.day_id FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.state = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"day_id_difference": []}
    return {"day_id_difference": [row[0] for row in result]}

# Endpoint to get distinct category names for businesses with a specific star rating
@app.get("/v1/public_review_platform/distinct_categories_by_stars", operation_id="get_distinct_categories_by_stars", summary="Retrieves a unique list of category names for businesses that have a specified star rating. The star rating is provided as an input parameter.")
async def get_distinct_categories_by_stars(stars: int = Query(..., description="Star rating of the business")):
    cursor.execute("SELECT DISTINCT T3.category_name FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T1.stars = ?", (stars,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get distinct states for businesses with a specific attribute value
@app.get("/v1/public_review_platform/distinct_states_by_attribute_value", operation_id="get_distinct_states_by_attribute_value", summary="Retrieve a unique list of states where businesses with a specified attribute value are located. The attribute value is provided as an input parameter.")
async def get_distinct_states_by_attribute_value(attribute_value: str = Query(..., description="Attribute value")):
    cursor.execute("SELECT DISTINCT T2.state FROM Business_Attributes AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T1.attribute_value = ?", (attribute_value,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get the count of users with a specific compliment type and number of compliments
@app.get("/v1/public_review_platform/user_count_by_compliment", operation_id="get_user_count_by_compliment", summary="Retrieves the total number of users who have received a specific type of compliment a certain number of times. The operation filters users based on the provided compliment type and the count of compliments they have received.")
async def get_user_count_by_compliment(compliment_type: str = Query(..., description="Type of compliment"), number_of_compliments: str = Query(..., description="Number of compliments")):
    cursor.execute("SELECT COUNT(T2.user_id) FROM Compliments AS T1 INNER JOIN Users_Compliments AS T2 ON T1.compliment_id = T2.compliment_id WHERE T1.compliment_type = ? AND T2.number_of_compliments = ?", (compliment_type, number_of_compliments))
    result = cursor.fetchone()
    if not result:
        return {"user_count": []}
    return {"user_count": result[0]}

# Endpoint to get the count of active businesses in a specific city
@app.get("/v1/public_review_platform/count_active_businesses_in_city", operation_id="get_count_active_businesses_in_city", summary="Retrieves the total number of active businesses located in a specified city. The active status of the businesses is also considered in the count.")
async def get_count_active_businesses_in_city(city: str = Query(..., description="City name"), active: str = Query(..., description="Active status of the business")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE city = ? AND active = ?", (city, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with specific star ratings
@app.get("/v1/public_review_platform/count_businesses_with_star_ratings", operation_id="get_count_businesses_with_star_ratings", summary="Retrieves the total number of businesses that have star ratings within the specified range. The range is defined by the first and second star rating parameters, which represent the lower and upper bounds, respectively.")
async def get_count_businesses_with_star_ratings(star1: int = Query(..., description="First star rating"), star2: int = Query(..., description="Second star rating")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE stars IN (?, ?)", (star1, star2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs based on star rating, city, and review count
@app.get("/v1/public_review_platform/business_ids_by_star_city_review_count", operation_id="get_business_ids_by_star_city_review_count", summary="Retrieves business identifiers for establishments in a specific city that have a star rating above a certain threshold and a specified review count.")
async def get_business_ids_by_star_city_review_count(min_stars: int = Query(..., description="Minimum star rating"), city: str = Query(..., description="City name"), review_count: str = Query(..., description="Review count")):
    cursor.execute("SELECT business_id FROM Business WHERE stars > ? AND city = ? AND review_count = ?", (min_stars, city, review_count))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of business attributes with a value greater than a specified number
@app.get("/v1/public_review_platform/count_business_attributes_greater_than", operation_id="get_count_business_attributes_greater_than", summary="Retrieves the total number of businesses that have an attribute value exceeding the provided minimum value. This operation is useful for filtering businesses based on a specific attribute threshold.")
async def get_count_business_attributes_greater_than(min_value: int = Query(..., description="Minimum attribute value")):
    cursor.execute("SELECT COUNT(business_id) FROM Business_Attributes WHERE attribute_value > ?", (min_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with specific opening and closing times
@app.get("/v1/public_review_platform/count_businesses_by_hours", operation_id="get_count_businesses_by_hours", summary="Retrieves the total number of businesses that operate within the specified opening and closing times. The operation considers the exact opening and closing times provided as input parameters.")
async def get_count_businesses_by_hours(opening_time: str = Query(..., description="Opening time in 'HHAM/PM' format"), closing_time: str = Query(..., description="Closing time in 'HHAM/PM' format")):
    cursor.execute("SELECT COUNT(business_id) FROM Business_Hours WHERE opening_time = ? AND closing_time = ?", (opening_time, closing_time))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of active businesses with a specific number of funny review votes
@app.get("/v1/public_review_platform/count_active_businesses_funny_review_votes", operation_id="get_count_active_businesses_funny_review_votes", summary="Retrieves the count of currently active businesses that have received a specified number of funny review votes. This operation provides a quantitative measure of businesses with a certain level of humorous engagement in their reviews.")
async def get_count_active_businesses_funny_review_votes(review_votes_funny: str = Query(..., description="Number of funny review votes"), active: str = Query(..., description="Active status of the business")):
    cursor.execute("SELECT COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Reviews AS T2 ON T1.business_id = T2.business_id WHERE T2.review_votes_funny = ? AND T1.active = ?", (review_votes_funny, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users with a specific number of compliments and fans
@app.get("/v1/public_review_platform/count_users_compliments_fans", operation_id="get_count_users_compliments_fans", summary="Retrieves the count of users who have received a specified number of compliments and have a certain number of fans. This operation provides a quantitative measure of user engagement and popularity based on the given criteria.")
async def get_count_users_compliments_fans(number_of_compliments: str = Query(..., description="Number of compliments"), user_fans: str = Query(..., description="Number of fans")):
    cursor.execute("SELECT COUNT(T2.user_id) FROM Users_Compliments AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id WHERE T1.number_of_compliments = ? AND T2.user_fans = ?", (number_of_compliments, user_fans))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common compliment type
@app.get("/v1/public_review_platform/most_common_compliment_type", operation_id="get_most_common_compliment_type", summary="Retrieves the most frequently occurring compliment type from the user compliments data. This operation ranks compliment types based on their frequency and returns the top-ranked type. The result provides insights into the most commonly used compliment in the platform.")
async def get_most_common_compliment_type():
    cursor.execute("SELECT T2.compliment_type FROM Users_Compliments AS T1 INNER JOIN Compliments AS T2 ON T1.compliment_id = T2.compliment_id GROUP BY T2.compliment_type ORDER BY COUNT(T2.compliment_type) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"compliment_type": []}
    return {"compliment_type": result[0]}

# Endpoint to get the top 3 users by average stars based on tip likes
@app.get("/v1/public_review_platform/top_users_by_average_stars", operation_id="get_top_users_by_average_stars", summary="Retrieves the top three users with the highest average star ratings, based on the total number of likes received on their tips. The users are ranked in descending order, with the user having the most likes at the top.")
async def get_top_users_by_average_stars():
    cursor.execute("SELECT T2.user_average_stars FROM Tips AS T1 INNER JOIN Users AS T2 ON T1.user_id = T2.user_id GROUP BY T2.user_id ORDER BY SUM(T1.likes) DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"user_average_stars": []}
    return {"user_average_stars": [row[0] for row in result]}

# Endpoint to get category names based on city
@app.get("/v1/public_review_platform/category_names_by_city", operation_id="get_category_names_by_city", summary="Retrieves the names of categories for businesses located in a specified city. The city name is provided as an input parameter.")
async def get_category_names_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.category_name FROM Categories AS T1 INNER JOIN Business_Categories AS T2 ON T1.category_id = T2.category_id INNER JOIN Business AS T3 ON T2.business_id = T3.business_id WHERE T3.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get business IDs based on day ID
@app.get("/v1/public_review_platform/business_ids_by_day", operation_id="get_business_ids_by_day", summary="Retrieves the unique identifiers of businesses that are open on a specific day, as determined by their business hours. The day is identified using its unique day ID.")
async def get_business_ids_by_day(day_id: int = Query(..., description="Day ID")):
    cursor.execute("SELECT T1.business_id FROM Business_Hours AS T1 INNER JOIN Days AS T2 ON T1.day_id = T2.day_id WHERE T1.day_id = ?", (day_id,))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct businesses based on review length, active status, and city
@app.get("/v1/public_review_platform/count_distinct_businesses_by_review_length_active_city", operation_id="get_count_distinct_businesses_by_review_length_active_city", summary="Retrieves the number of unique businesses that have reviews of a specified length, are currently active, and are located in a specific city.")
async def get_count_distinct_businesses_by_review_length_active_city(review_length: str = Query(..., description="Review length"), active: str = Query(..., description="Active status of the business"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.business_id) FROM Reviews AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id WHERE T1.review_length = ? AND T2.active = ? AND T2.city = ?", (review_length, active, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct users based on user votes cool and review votes cool
@app.get("/v1/public_review_platform/count_distinct_users_by_votes_cool", operation_id="get_count_distinct_users_by_votes_cool", summary="Retrieves the number of unique users who have a specific number of 'cool' votes on their user profile and their reviews. The operation considers both the 'cool' votes received on the user's profile and the 'cool' votes received on their reviews.")
async def get_count_distinct_users_by_votes_cool(user_votes_cool: str = Query(..., description="User votes cool"), review_votes_cool: str = Query(..., description="Review votes cool")):
    cursor.execute("SELECT COUNT(DISTINCT T1.user_id) FROM Users AS T1 INNER JOIN Reviews AS T2 ON T1.user_id = T2.user_id WHERE T1.user_votes_cool = ? AND T2.review_votes_cool = ?", (user_votes_cool, review_votes_cool))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct users based on tip length, likes, and user fans
@app.get("/v1/public_review_platform/count_distinct_users_by_tip_length_likes_fans", operation_id="get_count_distinct_users_by_tip_length_likes_fans", summary="Retrieves the number of unique users who have tips of a specific length, a certain number of likes, and a particular number of fans. This operation is useful for understanding user engagement and content popularity based on tip length and likes.")
async def get_count_distinct_users_by_tip_length_likes_fans(tip_length: str = Query(..., description="Tip length"), likes: int = Query(..., description="Number of likes"), user_fans: str = Query(..., description="User fans")):
    cursor.execute("SELECT COUNT(DISTINCT T1.user_id) FROM Users AS T1 INNER JOIN Tips AS T2 ON T1.user_id = T2.user_id WHERE T2.tip_length = ? AND T2.likes = ? AND T1.user_fans = ?", (tip_length, likes, user_fans))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of businesses with specific attribute values
@app.get("/v1/public_review_platform/count_businesses_by_attribute_values", operation_id="get_count_businesses_by_attribute_values", summary="Retrieves the number of businesses that possess the specified attribute and its corresponding values. The attribute name and up to three attribute values are required to filter the businesses.")
async def get_count_businesses_by_attribute_values(attribute_name: str = Query(..., description="Attribute name"), attribute_value1: str = Query(..., description="Attribute value 1"), attribute_value2: str = Query(..., description="Attribute value 2"), attribute_value3: str = Query(..., description="Attribute value 3")):
    cursor.execute("SELECT COUNT(T2.business_id) FROM Attributes AS T1 INNER JOIN Business_Attributes AS T2 ON T1.attribute_id = T2.attribute_id WHERE T1.attribute_name = ? AND T2.attribute_value IN (?, ?, ?)", (attribute_name, attribute_value1, attribute_value2, attribute_value3))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct businesses based on city, day of the week, and opening time
@app.get("/v1/public_review_platform/count_distinct_businesses_by_city_day_opening_time", operation_id="get_count_distinct_businesses_by_city_day_opening_time", summary="Retrieves the number of unique businesses in a specific city that operate on a given day of the week and open at a certain time. The response is based on the provided city name, day of the week, and opening time.")
async def get_count_distinct_businesses_by_city_day_opening_time(city: str = Query(..., description="City name"), day_of_week: str = Query(..., description="Day of the week"), opening_time: str = Query(..., description="Opening time")):
    cursor.execute("SELECT COUNT(DISTINCT T2.business_id) FROM Business AS T1 INNER JOIN Business_hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.city = ? AND T3.day_of_week = ? AND T2.opening_time = ?", (city, day_of_week, opening_time))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average star rating for businesses in a specific category
@app.get("/v1/public_review_platform/average_star_rating_by_category", operation_id="get_average_star_rating_by_category", summary="Retrieves the average star rating of businesses in a specified category. The calculation is based on the sum of all star ratings divided by the total number of businesses in the category. The category is identified by its name.")
async def get_average_star_rating_by_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT CAST(SUM(T1.stars) AS REAL) / COUNT(T1.business_id) FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T3.category_name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_star_rating": []}
    return {"average_star_rating": result[0]}

# Endpoint to get the percentage of elite users with high fans in a specific year
@app.get("/v1/public_review_platform/percentage_elite_users_high_fans_by_year", operation_id="get_percentage_elite_users_high_fans_by_year", summary="Retrieves the percentage of elite users who have a high number of fans in a specified year. The calculation is based on the total count of users with the specified fan level divided by the total count of elite users in the given year.")
async def get_percentage_elite_users_high_fans_by_year(user_fans: str = Query(..., description="User fans level"), actual_year: int = Query(..., description="Actual year")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.user_fans = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.user_fans) FROM Years AS T1 INNER JOIN Elite AS T2 ON T1.year_id = T2.year_id INNER JOIN Users AS T3 ON T2.user_id = T3.user_id WHERE T1.actual_year = ?", (user_fans, actual_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get business IDs based on review count and city
@app.get("/v1/public_review_platform/business_ids_by_review_count_and_city", operation_id="get_business_ids_by_review_count_and_city", summary="Retrieves the unique identifiers of businesses that have a specific number of reviews and are located in a particular city. The review count and city name are provided as input parameters.")
async def get_business_ids_by_review_count_and_city(review_count: str = Query(..., description="Review count"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT business_id FROM Business WHERE review_count = ? AND city = ?", (review_count, city))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of businesses based on review count, state, and active status
@app.get("/v1/public_review_platform/business_count_by_review_count_state_active", operation_id="get_business_count_by_review_count_state_active", summary="Retrieves the total number of businesses that match the specified review count, state, and active status. This operation allows you to filter businesses based on their review count, geographical location, and whether they are currently active or not.")
async def get_business_count_by_review_count_state_active(review_count: str = Query(..., description="Review count"), state: str = Query(..., description="State"), active: str = Query(..., description="Active status")):
    cursor.execute("SELECT COUNT(business_id) FROM Business WHERE review_count = ? AND state = ? AND active = ?", (review_count, state, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get business IDs based on star rating range and city
@app.get("/v1/public_review_platform/business_ids_by_star_range_and_city", operation_id="get_business_ids_by_star_range_and_city", summary="Retrieves the IDs of businesses that have a star rating within the specified range and are located in the provided city.")
async def get_business_ids_by_star_range_and_city(min_stars: int = Query(..., description="Minimum star rating"), max_stars: int = Query(..., description="Maximum star rating"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT business_id FROM Business WHERE stars >= ? AND stars < ? AND city = ?", (min_stars, max_stars, city))
    result = cursor.fetchall()
    if not result:
        return {"business_ids": []}
    return {"business_ids": [row[0] for row in result]}

# Endpoint to get the count of users based on yelping since year range and fan count
@app.get("/v1/public_review_platform/user_count_by_yelping_since_year_and_fans", operation_id="get_user_count_by_yelping_since_year_and_fans", summary="Retrieves the total number of users who started yelping within a specified year range and have a specific fan count. The year range is defined by the minimum and maximum years, and the fan count is a single value.")
async def get_user_count_by_yelping_since_year_and_fans(min_year: int = Query(..., description="Minimum yelping since year"), max_year: int = Query(..., description="Maximum yelping since year"), user_fans: str = Query(..., description="User fan count")):
    cursor.execute("SELECT COUNT(user_id) FROM Users WHERE user_yelping_since_year >= ? AND user_yelping_since_year < ? AND user_fans = ?", (min_year, max_year, user_fans))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct attribute names based on city and review count
@app.get("/v1/public_review_platform/distinct_attribute_names_by_city_and_review_count", operation_id="get_distinct_attribute_names_by_city_and_review_count", summary="Retrieves a list of unique attribute names for businesses in a specified city with a given review count. This operation filters businesses by city and review count, then identifies the distinct attribute names associated with these businesses.")
async def get_distinct_attribute_names_by_city_and_review_count(city: str = Query(..., description="City name"), review_count: str = Query(..., description="Review count")):
    cursor.execute("SELECT DISTINCT T3.attribute_name FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id WHERE T1.city = ? AND T1.review_count = ?", (city, review_count))
    result = cursor.fetchall()
    if not result:
        return {"attribute_names": []}
    return {"attribute_names": [row[0] for row in result]}

# Endpoint to get the count of distinct businesses based on category name and star rating
@app.get("/v1/public_review_platform/count_distinct_businesses_by_category_and_stars", operation_id="get_count_distinct_businesses_by_category_and_stars", summary="Retrieves the count of unique businesses that belong to a specific category and have a star rating less than a given value. The category is identified by its name, and the maximum star rating is provided as a parameter.")
async def get_count_distinct_businesses_by_category_and_stars(category_name: str = Query(..., description="Category name"), max_stars: int = Query(..., description="Maximum star rating")):
    cursor.execute("SELECT COUNT(DISTINCT T1.business_id) FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T3.category_name = ? AND T1.stars < ?", (category_name, max_stars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct business IDs and star ratings based on category name and active status
@app.get("/v1/public_review_platform/distinct_business_ids_and_stars_by_category_and_active", operation_id="get_distinct_business_ids_and_stars_by_category_and_active", summary="Retrieves unique business identifiers and their corresponding star ratings, filtered by a specified category and active status. This operation provides a concise overview of businesses within a certain category, along with their star ratings, based on the provided active status.")
async def get_distinct_business_ids_and_stars_by_category_and_active(category_name: str = Query(..., description="Category name"), active: str = Query(..., description="Active status")):
    cursor.execute("SELECT DISTINCT T1.business_id, T1.stars FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T3.category_name = ? AND T1.active = ?", (category_name, active))
    result = cursor.fetchall()
    if not result:
        return {"business_ids_and_stars": []}
    return {"business_ids_and_stars": [{"business_id": row[0], "stars": row[1]} for row in result]}

# Endpoint to get distinct category names and attribute names for businesses with the maximum star rating
@app.get("/v1/public_review_platform/distinct_category_and_attribute_names_by_max_stars", operation_id="get_distinct_category_and_attribute_names_by_max_stars", summary="Retrieves unique category names and attribute names for businesses that have achieved the highest star rating. This operation does not require any input parameters and returns a list of distinct category and attribute names.")
async def get_distinct_category_and_attribute_names_by_max_stars():
    cursor.execute("SELECT DISTINCT T3.category_name, T5.attribute_name FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id INNER JOIN Business_Attributes AS T4 ON T2.business_id = T4.business_id INNER JOIN Attributes AS T5 ON T4.attribute_id = T5.attribute_id WHERE T1.stars = ( SELECT MAX(stars) FROM Business )")
    result = cursor.fetchall()
    if not result:
        return {"category_and_attribute_names": []}
    return {"category_and_attribute_names": [{"category_name": row[0], "attribute_name": row[1]} for row in result]}

# Endpoint to get the count of distinct business IDs based on attribute name, review count, and active status
@app.get("/v1/public_review_platform/count_distinct_business_ids", operation_id="get_count_distinct_business_ids", summary="Retrieves the total number of unique businesses that match a specific attribute, review count, and active status. This operation considers businesses with the provided attribute name, review count, and active status, ensuring that only distinct businesses are counted.")
async def get_count_distinct_business_ids(attribute_name: str = Query(..., description="Attribute name"), review_count: str = Query(..., description="Review count"), active: str = Query(..., description="Active status")):
    cursor.execute("SELECT COUNT(DISTINCT T1.business_id) FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id WHERE T3.attribute_name = ? AND T1.review_count = ? AND T1.active = ?", (attribute_name, review_count, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the opening and closing times of the highest-rated business in a specific city
@app.get("/v1/public_review_platform/business_hours_highest_rated", operation_id="get_business_hours_highest_rated", summary="Retrieves the opening and closing times of the highest-rated business in a specified city. The business is determined by its star rating, with the highest-rated business being selected. The city is identified by the provided city name.")
async def get_business_hours_highest_rated(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T2.opening_time, T2.closing_time FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id WHERE T1.city = ? ORDER BY T1.stars DESC LIMIT 1", (city,))
    result = cursor.fetchone()
    if not result:
        return {"opening_time": [], "closing_time": []}
    return {"opening_time": result[0], "closing_time": result[1]}

# Endpoint to get category names and attribute names based on review count, city, state, and active status
@app.get("/v1/public_review_platform/category_attribute_names", operation_id="get_category_attribute_names", summary="Retrieves the category names and associated attribute names for businesses that meet specific criteria. These criteria include a certain review count, a particular city and state, and an active status. The response provides a comprehensive list of categories and their corresponding attributes for businesses that match the given parameters.")
async def get_category_attribute_names(review_count: str = Query(..., description="Review count"), city: str = Query(..., description="City name"), state: str = Query(..., description="State"), active: str = Query(..., description="Active status")):
    cursor.execute("SELECT T3.category_name, T5.attribute_name FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id INNER JOIN Business_Attributes AS T4 ON T1.business_id = T4.business_id INNER JOIN Attributes AS T5 ON T4.attribute_id = T5.attribute_id WHERE T1.review_count = ? AND T1.city = ? AND T1.state = ? AND T1.active = ?", (review_count, city, state, active))
    result = cursor.fetchall()
    if not result:
        return {"category_attribute_names": []}
    return {"category_attribute_names": [{"category_name": row[0], "attribute_name": row[1]} for row in result]}

# Endpoint to get distinct category names based on active status, state, and city
@app.get("/v1/public_review_platform/distinct_category_names_by_status_state_city", operation_id="get_distinct_category_names_by_status_state_city", summary="Retrieves a list of unique category names for businesses that are currently active, located in a specific state, and found within a particular city. This operation helps to identify the various categories of businesses that meet the specified criteria.")
async def get_distinct_category_names_by_status_state_city(active: str = Query(..., description="Active status"), state: str = Query(..., description="State"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT DISTINCT T3.category_name FROM Business_Categories AS T1 INNER JOIN Business AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T1.category_id = T3.category_id WHERE T2.active = ? AND T2.state = ? AND T2.city = ?", (active, state, city))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get distinct cities based on opening time, closing time, and day of the week
@app.get("/v1/public_review_platform/distinct_cities_by_hours_day", operation_id="get_distinct_cities_by_hours_day", summary="Retrieve a list of unique cities where businesses operate with specific opening and closing times on a given day of the week.")
async def get_distinct_cities_by_hours_day(opening_time: str = Query(..., description="Opening time"), closing_time: str = Query(..., description="Closing time"), day_of_week: str = Query(..., description="Day of the week")):
    cursor.execute("SELECT DISTINCT T1.city FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T2.opening_time = ? AND T2.closing_time = ? AND T3.day_of_week = ?", (opening_time, closing_time, day_of_week))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get distinct attribute values based on state, city, active status, stars, and review count
@app.get("/v1/public_review_platform/distinct_attribute_values", operation_id="get_distinct_attribute_values", summary="Retrieve unique attribute values for businesses located in a specific state and city, filtered by active status, star rating, and review count. This operation provides a comprehensive view of the distinct attributes associated with businesses that meet the specified criteria.")
async def get_distinct_attribute_values(state: str = Query(..., description="State"), city: str = Query(..., description="City name"), active: str = Query(..., description="Active status"), stars: int = Query(..., description="Stars"), review_count: str = Query(..., description="Review count")):
    cursor.execute("SELECT DISTINCT T2.attribute_value FROM Business AS T1 INNER JOIN Business_Attributes AS T2 ON T1.business_id = T2.business_id INNER JOIN Attributes AS T3 ON T2.attribute_id = T3.attribute_id WHERE T1.state = ? AND T1.city = ? AND T1.active = ? AND T1.stars = ? AND T1.review_count = ?", (state, city, active, stars, review_count))
    result = cursor.fetchall()
    if not result:
        return {"attribute_values": []}
    return {"attribute_values": [row[0] for row in result]}

# Endpoint to get the percentage of businesses with stars greater than a specified value in a specific category
@app.get("/v1/public_review_platform/percentage_high_stars_by_category", operation_id="get_percentage_high_stars_by_category", summary="Retrieves the percentage of businesses in a specified category that have a star rating higher than a given value. This operation calculates the proportion of businesses with a star rating exceeding the provided minimum value, within the context of a specific category.")
async def get_percentage_high_stars_by_category(min_stars: int = Query(..., description="Minimum stars"), category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.stars > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.stars) FROM Business AS T1 INNER JOIN Business_Categories AS T2 ON T1.business_id = T2.business_id INNER JOIN Categories AS T3 ON T2.category_id = T3.category_id WHERE T3.category_name = ?", (min_stars, category_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct closing times and days of the week for businesses with stars above a certain threshold in a specific city
@app.get("/v1/public_review_platform/closing_times_days_above_star_threshold", operation_id="get_closing_times_days_above_star_threshold", summary="Retrieve unique closing times and days of the week for businesses in a specific city that have a star rating above 80% of the average rating for active businesses in that city. The active status of the businesses is also considered.")
async def get_closing_times_days_above_star_threshold(active: str = Query(..., description="Active status"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT DISTINCT T2.closing_time, T3.day_of_week FROM Business AS T1 INNER JOIN Business_Hours AS T2 ON T1.business_id = T2.business_id INNER JOIN Days AS T3 ON T2.day_id = T3.day_id WHERE T1.active = ? AND T1.city = ? AND T1.stars > ( SELECT AVG(stars) * 0.8 FROM Business WHERE active = ? AND city = ? )", (active, city, active, city))
    result = cursor.fetchall()
    if not result:
        return {"closing_times_days": []}
    return {"closing_times_days": [{"closing_time": row[0], "day_of_week": row[1]} for row in result]}

api_calls = [
    "/v1/public_review_platform/business_count_state_stars?state=AZ&max_stars=3",
    "/v1/public_review_platform/business_count_state_active?state=AZ&active=False",
    "/v1/public_review_platform/review_count_user_length?user_id=36139&review_length=long",
    "/v1/public_review_platform/user_count_fans?user_fans=Uber",
    "/v1/public_review_platform/business_count_attribute?attribute_name=Open%2024%20Hours&attribute_value=TRUE",
    "/v1/public_review_platform/business_attribute_value?business_id=10172&attribute_name=wi-fi",
    "/v1/public_review_platform/business_count_category?category_name=Bars",
    "/v1/public_review_platform/category_count_difference?category_name_1=Buffets&category_name_2=Gyms",
    "/v1/public_review_platform/top_category_5_star_reviews?review_stars=5",
    "/v1/public_review_platform/top_year_5_star_reviews?review_stars=5",
    "/v1/public_review_platform/average_review_stars?review_length=Long",
    "/v1/public_review_platform/top_category_by_review_length?review_length=Long",
    "/v1/public_review_platform/distinct_categories_by_tip_length?tip_length=short",
    "/v1/public_review_platform/top_user_yelping_since_year?tip_length=short",
    "/v1/public_review_platform/category_names_by_user?user_id=70271",
    "/v1/public_review_platform/business_stars_by_user?user_id=69722",
    "/v1/public_review_platform/percentage_businesses_in_category?category_name=Automotive",
    "/v1/public_review_platform/percentage_difference_between_categories?category_name_1=Women's%20Clothing&category_name_2=Men's%20Clothing",
    "/v1/public_review_platform/user_count_by_yelping_year?user_yelping_since_year=2004",
    "/v1/public_review_platform/user_count_by_yelping_year_and_fans?user_yelping_since_year=2005&user_fans=None",
    "/v1/public_review_platform/business_count_by_city_and_status?city=Tolleson&active=TRUE",
    "/v1/public_review_platform/review_count_by_user?user_id=21679",
    "/v1/public_review_platform/review_count_by_business_and_stars?business_id=10682&review_stars=5",
    "/v1/public_review_platform/top_business_by_reviews?city=Sun%20City&active=FALSE",
    "/v1/public_review_platform/review_count_by_city_and_length?city=Yuma&review_length=Medium",
    "/v1/public_review_platform/check_business_attribute?attribute_name=Has%20TV&business_id=4960",
    "/v1/public_review_platform/business_operating_hours?day_of_week=Saturday&business_id=5734",
    "/v1/public_review_platform/business_city_by_tips_likes_user_id?likes=1&user_id=63469",
    "/v1/public_review_platform/count_attributes_by_value_business_id?attribute_value=TRUE&business_id=1141",
    "/v1/public_review_platform/count_compliments_by_type_user_id?compliment_type=cute&user_id=57400",
    "/v1/public_review_platform/user_ids_by_compliment_type?compliment_type=funny",
    "/v1/public_review_platform/business_ids_by_city_attribute?max_business_id=1000&city=Scottsdale&attribute_name=Drive-Thru&attribute_value=TRUE",
    "/v1/public_review_platform/avg_stars_by_attribute?attribute_name=Open%2024%20Hours&attribute_value=TRUE",
    "/v1/public_review_platform/percentage_businesses_by_city_attribute?city=Phoenix&attribute_name=BYOB&attribute_value=TRUE",
    "/v1/public_review_platform/business_ids_by_state_stars?state=AZ&stars=5",
    "/v1/public_review_platform/count_businesses_by_review_count_active?review_count=Low&active=TRUE",
    "/v1/public_review_platform/count_users_by_id_range_fans_review_count?min_user_id=1&max_user_id=20&user_fans=None&user_review_count=Low",
    "/v1/public_review_platform/business_opening_time_by_category?category_name=Fashion",
    "/v1/public_review_platform/business_count_by_opening_time_and_category?opening_time=8AM&category_name=Shopping",
    "/v1/public_review_platform/business_ids_by_closing_time_and_category?closing_time=9PM&category_name=Pets",
    "/v1/public_review_platform/business_count_by_days_of_week?day1=Monday&day2=Tuesday&day3=Wednesday&day4=Thursday",
    "/v1/public_review_platform/business_count_by_category_and_city?category_name=Active%20Life&city=Phoenix",
    "/v1/public_review_platform/business_ids_by_category_and_stars?category_name=Men's%20Clothing&max_stars=5",
    "/v1/public_review_platform/distinct_business_ids_by_review_usefulness_and_activity?active_status=FALSE&review_votes_useful=Low",
    "/v1/public_review_platform/category_names_by_review_length_and_category_id_range?review_length=Long&min_category_id=1&max_category_id=20",
    "/v1/public_review_platform/business_attribute_values_by_category_and_city?category_name=Fashion&city=Scottsdale",
    "/v1/public_review_platform/compliment_count_by_city_and_compliment_level?city=Phoenix&compliment_level=Medium",
    "/v1/public_review_platform/category_names_by_city_opening_time?city=Tempe&opening_time=8AM",
    "/v1/public_review_platform/count_category_names_by_city_opening_closing_time?city=Glendale&opening_time=8AM&closing_time=6PM",
    "/v1/public_review_platform/count_percentage_businesses_by_city?city=Phoenix",
    "/v1/public_review_platform/count_percentage_businesses_by_category_city_review_count?category_name=Active%20Life&city=Phoenix&review_count=Low",
    "/v1/public_review_platform/count_businesses_by_city_stars?city=Scottsdale&stars=3",
    "/v1/public_review_platform/average_stars_by_active_status?active=FALSE",
    "/v1/public_review_platform/count_businesses_by_attribute_state?attribute_value=beer_and_wine&state=AZ",
    "/v1/public_review_platform/cities_by_attribute_value?attribute_value=full_bar",
    "/v1/public_review_platform/count_businesses_by_stars_category?stars=5&category_name=Fashion",
    "/v1/public_review_platform/cities_by_review_count_category?review_count=High&category_name=Food",
    "/v1/public_review_platform/business_ids_by_city_review_stars?city=Mesa&review_stars=3",
    "/v1/public_review_platform/cities_by_funny_review_votes?review_votes_funny=low",
    "/v1/public_review_platform/percentage_5_star_businesses?city=Chandler&category_name=Real%20Estate",
    "/v1/public_review_platform/user_count_by_yelping_year_funny_votes?user_yelping_since_year=2012&user_votes_funny=High",
    "/v1/public_review_platform/useful_review_votes?user_id=52592&business_id=2",
    "/v1/public_review_platform/attribute_ids_by_name?attribute_name=%25payment%25",
    "/v1/public_review_platform/review_length?user_id=612&review_stars=5&business_id=2",
    "/v1/public_review_platform/business_count_by_state_review_count?state=AZ&review_count=Low",
    "/v1/public_review_platform/distinct_business_ids?state=AZ&review_stars=5&limit=3",
    "/v1/public_review_platform/attribute_names?attribute_value=none&limit=1",
    "/v1/public_review_platform/compliment_type_count?user_id=33&compliment_type=cool",
    "/v1/public_review_platform/opening_hours?day_of_week=Friday&business_id=53",
    "/v1/public_review_platform/attribute_names_by_value_and_business?attribute_value=TRUE&business_id=56",
    "/v1/public_review_platform/category_names_by_business?business_id=15",
    "/v1/public_review_platform/business_count_by_city_and_category?city=Scottsdale&category_name=Beauty%20%26%20Spas",
    "/v1/public_review_platform/user_ids_by_compliments?number_of_compliments=Uber&compliment_type=cute&limit=2",
    "/v1/public_review_platform/business_count_by_stars_and_category?stars=3&category_name=Accessories",
    "/v1/public_review_platform/opening_hours_by_business_city_day?business_id=12&city=Scottsdale&day_id=3",
    "/v1/public_review_platform/business_count_checkin_time_state_day?label_time_4=None&state=AZ&day_of_week=Thursday",
    "/v1/public_review_platform/business_count_by_city?city=Scottsdale",
    "/v1/public_review_platform/top_reviewed_city",
    "/v1/public_review_platform/business_count_by_state_and_stars?state=AZ&stars=4",
    "/v1/public_review_platform/business_count_by_state?state=AZ",
    "/v1/public_review_platform/cities_by_star_rating?stars=5",
    "/v1/public_review_platform/review_count_by_user_length_and_useful_votes?user_id=3&review_length=Long&review_votes_useful=Medium",
    "/v1/public_review_platform/user_ids_by_fans?user_fans=High",
    "/v1/public_review_platform/attribute_count_by_name_and_value?attribute_name=Alcohol&attribute_value=none",
    "/v1/public_review_platform/business_count_by_attribute_and_state?attribute_name=Alcohol&attribute_value=none&state=AZ",
    "/v1/public_review_platform/business_ids_by_attribute?attribute_name=Good%20for%20Kids&attribute_value=TRUE",
    "/v1/public_review_platform/category_names_by_business_id?business_id=1",
    "/v1/public_review_platform/active_business_count_by_category?category_name=Food&active=TRUE",
    "/v1/public_review_platform/top_business_by_category?category_name=Food",
    "/v1/public_review_platform/count_stars_category_attribute?category_name=Food&attribute_name=Good%20for%20Kids&attribute_value=TRUE",
    "/v1/public_review_platform/count_businesses_reviewed_by_user_state?state=AZ&user_id=3",
    "/v1/public_review_platform/category_names_by_state?state=AZ",
    "/v1/public_review_platform/opening_time_by_day?day_of_week=Tuesday&business_id=1",
    "/v1/public_review_platform/count_businesses_open_after_time?day_of_week=Monday&closing_time=8PM",
    "/v1/public_review_platform/opening_times_active_businesses?day_of_week=Monday&city=Anthem&active=True",
    "/v1/public_review_platform/count_businesses_state_closing_time?day_of_week=Sunday&closing_time=12PM&state=AZ",
    "/v1/public_review_platform/category_names_closing_time_day?closing_time=12PM&day_of_week=Sunday",
    "/v1/public_review_platform/count_businesses_attribute_days?day_ids=1,2,3,4,5,6,7&attribute_name=Good%20for%20Kids&attribute_value=true",
    "/v1/public_review_platform/user_count_yelping_since_year",
    "/v1/public_review_platform/longest_operating_hours?day_of_week=Monday&category_name=Shopping",
    "/v1/public_review_platform/businesses_open_longer_than?min_hours=12&day_of_week=Sunday",
    "/v1/public_review_platform/elite_user_count_for_business?business_id=1",
    "/v1/public_review_platform/elite_user_count_reviews_more_than?min_reviews=10",
    "/v1/public_review_platform/top_reviewer_in_state?state=AZ",
    "/v1/public_review_platform/average_review_stars_city?city=Anthem",
    "/v1/public_review_platform/average_review_stars_user_state?state=AZ&user_id=3",
    "/v1/public_review_platform/avg_opening_hours_business_days?business_id=1&day_of_week_1=Sunday&day_of_week_2=Sunday",
    "/v1/public_review_platform/average_stars_day_closing_time?day_of_week=Sunday&closing_time=12PM",
    "/v1/public_review_platform/business_count_by_state_active_review?state=AZ&active=True&review_count=low",
    "/v1/public_review_platform/business_ids_by_city_stars?city=Mesa&min_stars=2&max_stars=3",
    "/v1/public_review_platform/user_count_by_year_fans?min_year=2011&max_year=2013&user_fans=High",
    "/v1/public_review_platform/review_length_by_user_business?user_id=35026&business_id=2",
    "/v1/public_review_platform/distinct_attributes_by_review_count_city?review_count=Low&city=Chandler",
    "/v1/public_review_platform/business_count_by_stars_category?max_stars=4&category_name=Mexican",
    "/v1/public_review_platform/business_ids_stars_by_active_category?active=TRUE&category_name=Fashion",
    "/v1/public_review_platform/top_category_by_stars",
    "/v1/public_review_platform/top_category_by_review_length_business_id?review_length=Medium&min_business_id=6&max_business_id=9",
    "/v1/public_review_platform/business_count_by_attributes?attribute_name=Caters&review_count=Low&active=TRUE",
    "/v1/public_review_platform/top_rated_business_hours?city=Tempe",
    "/v1/public_review_platform/business_category_attribute_names?active=TRUE&state=AZ&city=Chandler&review_count=Medium",
    "/v1/public_review_platform/business_category_names?active=TRUE&state=AZ&city=Surprise",
    "/v1/public_review_platform/business_cities_by_hours?closing_time=9PM&opening_time=8AM&day_of_week=Friday",
    "/v1/public_review_platform/distinct_opening_times?city=Chandler&active=TRUE&review_count=Medium",
    "/v1/public_review_platform/percentage_low_stars_by_category?category_name=Accessories",
    "/v1/public_review_platform/high_star_business_hours?city=Tempe&active=TRUE",
    "/v1/public_review_platform/business_count_by_review_count?review_count=High",
    "/v1/public_review_platform/attribute_count_by_business?business_id=2",
    "/v1/public_review_platform/user_count_by_compliments?number_of_compliments=High&compliment_id=1",
    "/v1/public_review_platform/business_count_by_city_attributes?city=Phoenix&attribute_name=waiter_service&attribute_id=2",
    "/v1/public_review_platform/attribute_names_24_7_businesses?day_id_1=1&day_id_2=2&day_id_3=3&day_id_4=4&day_id_5=5&day_id_6=6&day_id_7=7",
    "/v1/public_review_platform/category_names_by_state_review_stars?state=AZ&min_review_stars=3",
    "/v1/public_review_platform/user_proportion_by_average_stars?year_id=2013",
    "/v1/public_review_platform/elite_user_increment?year_id_before=2014&year_id_after=2005",
    "/v1/public_review_platform/business_count_and_yelping_years?user_id=3",
    "/v1/public_review_platform/average_review_stars_per_year?user_id=3",
    "/v1/public_review_platform/user_review_to_business_ratio",
    "/v1/public_review_platform/user_average_stars_and_tip_likes?min_tips=5",
    "/v1/public_review_platform/top_category_by_reviews?category_name=Hotels%20%26%20Travel",
    "/v1/public_review_platform/count_active_businesses?active=True",
    "/v1/public_review_platform/business_ids_by_city_and_review_count?city=Phoenix&review_count=Low",
    "/v1/public_review_platform/count_businesses_by_state_review_count_active?state=AZ&review_count=High&active=True",
    "/v1/public_review_platform/top_attribute_name_by_stars",
    "/v1/public_review_platform/business_count_attributes_active_review_count?attribute_name=Wi-Fi&active=TRUE&review_count=Medium",
    "/v1/public_review_platform/category_names_review_count_city_active_state?review_count=Low&city=Mesa&active=true&state=AZ",
    "/v1/public_review_platform/inactive_business_categories?active=FALSE&state=AZ",
    "/v1/public_review_platform/business_opening_times?city=Surprise&active=TRUE&review_count=Low",
    "/v1/public_review_platform/percentage_low_rated_businesses?category_name=Local%20Services",
    "/v1/public_review_platform/highly_rated_business_hours?city=Scottsdale&active=TRUE",
    "/v1/public_review_platform/user_count_by_average_stars?user_average_stars=4&limit=10",
    "/v1/public_review_platform/cities_with_specific_business_hours_day?opening_time=10AM&closing_time=12PM&day_of_week=Sunday",
    "/v1/public_review_platform/count_businesses_by_attribute?attribute_value=TRUE&attribute_name=Open%2024%20Hours",
    "/v1/public_review_platform/business_ids_by_city_active_day?city=Ahwatukee&active=TRUE&day_of_week=Sunday",
    "/v1/public_review_platform/category_names_by_active_state?active=TRUE&state=AZ",
    "/v1/public_review_platform/category_names_by_stars_review_count?stars=2&review_count=High",
    "/v1/public_review_platform/count_businesses_by_attribute_value_name?attribute_value=true&attribute_name=ambience_romantic",
    "/v1/public_review_platform/cities_by_business_hours_day?closing_time=6PM&opening_time=1PM&day_of_week=Saturday",
    "/v1/public_review_platform/count_user_fans_by_tips_likes?limit=1",
    "/v1/public_review_platform/cities_by_business_hours_day_exact?closing_time=1AM&opening_time=12AM&day_of_week=Saturday",
    "/v1/public_review_platform/count_businesses_by_category_review_count?category_name=Shopping%20Centers&review_count=High",
    "/v1/public_review_platform/business_count_by_attribute?attribute_name=Accepts%20Insurance&attribute_value=true",
    "/v1/public_review_platform/weighted_average_stars?state1=SC&state2=CA",
    "/v1/public_review_platform/business_open_diff?day1=Monday&day2=Tuesday&opening_time=10AM&closing_time=9PM",
    "/v1/public_review_platform/attribute_id_by_name?attribute_name=Accepts%20Insurance",
    "/v1/public_review_platform/active_business_count_by_city?active=true&city=Phoenix",
    "/v1/public_review_platform/business_count_by_stars_city?stars=4&city=Mesa",
    "/v1/public_review_platform/business_count_by_review_count_city?review_count=High&city=Gilbert",
    "/v1/public_review_platform/review_length_count?city=Tempe&stars=3.5&review_count=Uber&review_length=Long",
    "/v1/public_review_platform/attribute_names_by_city_review_count?city=Mesa&review_count=Uber&attribute_name=Noise%20Level",
    "/v1/public_review_platform/business_hours_difference?day_of_week=Monday&business_id=15098",
    "/v1/public_review_platform/business_ids_by_city_review_stars_funny?city=Phoenix&review_stars=5&review_votes_funny=Uber",
    "/v1/public_review_platform/cities_by_tip_length_likes?tip_length=Medium&likes=3",
    "/v1/public_review_platform/count_user_compliments?user_yelping_since_year=2010&user_average_stars=4.5&user_fans=Uber&compliment_type=funny",
    "/v1/public_review_platform/count_compliments_user_type?compliment_type=cool&user_id=41717",
    "/v1/public_review_platform/percentage_businesses_category?category_name=Pets",
    "/v1/public_review_platform/ratio_businesses_categories?category_name_1=Women's%20Clothing&category_name_2=Men's%20Clothing",
    "/v1/public_review_platform/business_details_by_state_active?state=CA&active=true",
    "/v1/public_review_platform/percentage_active_businesses?active=true",
    "/v1/public_review_platform/attributes_by_name_pattern?attribute_name=music%25",
    "/v1/public_review_platform/year_with_most_elite_users?year1=2006&year2=2007",
    "/v1/public_review_platform/percentage_users_by_compliments?number_of_compliments=Low",
    "/v1/public_review_platform/reviews_by_cool_votes?review_votes_cool=Uber",
    "/v1/public_review_platform/user_tips_by_yelping_year_fans?user_yelping_since_year=2004&user_fans=High",
    "/v1/public_review_platform/business_user_details_by_review_votes_length?review_votes_cool=Uber&review_votes_funny=Uber&review_length=Long",
    "/v1/public_review_platform/business_attributes_by_name_active?attribute_name=music_playlist&active=false",
    "/v1/public_review_platform/percentage_businesses_with_attribute?attribute_name=Accepts%20Credit%20Cards",
    "/v1/public_review_platform/reviews_by_city_status_stars?city=San%20Tan%20Valley&active=false&review_stars=5",
    "/v1/public_review_platform/user_compliments_by_type_and_user?number_of_compliments=Uber&user_id=6027",
    "/v1/public_review_platform/businesses_by_category?category_name=Coffee%20%26%20Tea",
    "/v1/public_review_platform/categories_exceeding_business_percentage?percentage=0.1",
    "/v1/public_review_platform/user_fans_by_city_and_stars?city=Sun%20Lakes&stars=5",
    "/v1/public_review_platform/business_count_difference_by_categories?category_name_1=Men%27s%20Clothing&category_name_2=Women%27s%20Clothing",
    "/v1/public_review_platform/compliment_types_and_user_fans?number_of_compliments=Uber&max_user_id=100",
    "/v1/public_review_platform/business_ids_by_closing_time?closing_time=8PM",
    "/v1/public_review_platform/business_count_by_city_state_stars?city=Phoenix&state=AZ&stars=2",
    "/v1/public_review_platform/business_count_by_city_and_min_stars?city=Phoenix&min_stars=3",
    "/v1/public_review_platform/user_ids_by_yelping_year_and_stars?yelping_since_year=2012&average_stars=3",
    "/v1/public_review_platform/percentage_businesses_by_stars?stars=5",
    "/v1/public_review_platform/review_count_difference",
    "/v1/public_review_platform/business_ids_by_category_open_all_week?category_name=Tires",
    "/v1/public_review_platform/elite_user_ids_by_year?actual_year=2012",
    "/v1/public_review_platform/count_business_ids_by_day_and_time_label?day_of_week=Sunday&label_time_10=Low",
    "/v1/public_review_platform/count_business_ids_by_city_and_user?city=Glendale&user_id=20241",
    "/v1/public_review_platform/cities_by_business_category?category_name=Pet%20Services",
    "/v1/public_review_platform/count_user_ids_by_compliment_type_and_review_votes?compliment_type=photos&review_votes_cool=High",
    "/v1/public_review_platform/business_count_by_active_status_and_attributes?active=false&min_attributes=10",
    "/v1/public_review_platform/business_ids_by_city_and_attribute?city=Mesa&attribute_name=Alcohol",
    "/v1/public_review_platform/funny_review_votes_percentage_by_city?review_votes_funny=Low&city=Phoenix",
    "/v1/public_review_platform/category_ratio?category_name_1=Shopping&category_name_2=Pets",
    "/v1/public_review_platform/business_count_by_category?category_name=Banks%20%26%20Credit%20Unions",
    "/v1/public_review_platform/business_opening_time?business_id=12&day_of_week=Monday",
    "/v1/public_review_platform/top_business_by_stars?city=Gilbert&active=true&review_count=High",
    "/v1/public_review_platform/business_ids_and_categories_by_city_and_stars?city=Ahwatukee&stars=5",
    "/v1/public_review_platform/business_star_percentage?min_stars=3&city=Avondale&active=false",
    "/v1/public_review_platform/elite_users_by_year?user_yelping_since_year=2004",
    "/v1/public_review_platform/review_length_percentage?review_length=Long&review_stars=5",
    "/v1/public_review_platform/user_fans_percentage?user_fans=None&user_average_stars=4",
    "/v1/public_review_platform/tip_count_by_business?business_id=2&tip_length=Short",
    "/v1/public_review_platform/top_user_by_average_stars?user_average_stars=5",
    "/v1/public_review_platform/business_hours_by_city?city=Black%20Canyon%20City",
    "/v1/public_review_platform/compliment_percentage?compliment_type=cute&number_of_compliments=High",
    "/v1/public_review_platform/business_attribute_count?attribute_value1=none&attribute_value2=no&attribute_value3=false",
    "/v1/public_review_platform/business_hours_by_day?business_id=1&day_id=2",
    "/v1/public_review_platform/distinct_cities_by_review_length?review_length=Medium",
    "/v1/public_review_platform/closing_time_by_day_and_business?day_of_week=Sunday&business_id=4",
    "/v1/public_review_platform/distinct_business_ids_by_city_and_review_length?city=Phoenix&review_length=Short",
    "/v1/public_review_platform/user_count_by_compliments_and_fans?number_of_compliments=High&user_fans=Medium",
    "/v1/public_review_platform/distinct_user_ids_by_yelping_since_year_and_compliments?user_yelping_since_year=2012&number_of_compliments=Low",
    "/v1/public_review_platform/business_count_by_city_and_attribute_values?city=Gilbert&attribute_value1=None&attribute_value2=no&attribute_value3=false",
    "/v1/public_review_platform/business_count_by_attribute_value?attribute_value=full_bar",
    "/v1/public_review_platform/distinct_states_by_opening_time?opening_time=1AM",
    "/v1/public_review_platform/tip_length_percentage_and_yelping_year?tip_length=Medium",
    "/v1/public_review_platform/business_percentage_and_attribute_value?city=Mesa",
    "/v1/public_review_platform/distinct_states_by_closing_time?closing_time=12AM",
    "/v1/public_review_platform/business_count_by_city_and_attribute?city=Peoria&attribute_value=beer_and_wine",
    "/v1/public_review_platform/user_ids_by_compliments_and_earliest_year?number_of_compliments=High",
    "/v1/public_review_platform/business_with_most_reviews",
    "/v1/public_review_platform/business_count_by_review_stars_and_funny_votes?review_stars=5&review_votes_funny=Uber",
    "/v1/public_review_platform/distinct_user_count_by_funny_votes?review_votes_funny=Uber",
    "/v1/public_review_platform/business_with_longest_operating_hours",
    "/v1/public_review_platform/categories/tip_lengths_and_likes?category_name=Hotels%20%26%20Travel",
    "/v1/public_review_platform/reviews/user_ids_and_average_stars?review_votes_funny=Uber&review_votes_useful=Uber&review_votes_cool=Uber",
    "/v1/public_review_platform/business/high_low_rating_ratio?min_high_rating=3.5&max_high_rating=5&min_low_rating=1&max_low_rating=2.5",
    "/v1/public_review_platform/reviews/top_businesses_by_reviews?limit=10",
    "/v1/public_review_platform/business/count_low_rated_businesses?state=AZ&threshold=3",
    "/v1/public_review_platform/users/percentage_non_elite",
    "/v1/public_review_platform/users/distinct_compliment_types?user_fans=Uber",
    "/v1/public_review_platform/users/average_yelping_years_elite?user_fans=Uber",
    "/v1/public_review_platform/users/average_yelping_years_all_elite",
    "/v1/public_review_platform/active_business_percentage?city=Mesa",
    "/v1/public_review_platform/distinct_category_names?city=Phoenix&opening_time=5PM",
    "/v1/public_review_platform/top_user_by_attribute?attribute_name=Delivery",
    "/v1/public_review_platform/avg_user_id_by_operating_time?total_operating_time=30",
    "/v1/public_review_platform/business_ids_by_hours?opening_time=8AM&closing_time=6PM",
    "/v1/public_review_platform/business_ids_by_day_and_time?day_id=6&opening_time=10AM",
    "/v1/public_review_platform/day_ids_by_hours?opening_time=8AM&closing_time=6PM",
    "/v1/public_review_platform/business_count_by_stars?min_stars=4",
    "/v1/public_review_platform/category_names_by_day?day_of_week=Sunday",
    "/v1/public_review_platform/days_by_category?category_name=Pets",
    "/v1/public_review_platform/business_opening_times_by_category?category_name=Doctors",
    "/v1/public_review_platform/top_category_by_business_hours",
    "/v1/public_review_platform/business_closing_times_by_category_and_day?category_name=Arts%20%26%20Entertainment&day_of_week=Sunday",
    "/v1/public_review_platform/business_count_by_category_and_stars?category_name=DJs&max_stars=5",
    "/v1/public_review_platform/business_ids_by_activity_and_hours?active=true&opening_time=7AM&closing_time=8PM",
    "/v1/public_review_platform/business_count_by_category_and_max_stars?category_name=Stadiums%20%26%20Arenas",
    "/v1/public_review_platform/category_count_by_review_count_and_stars?review_count=Low&min_stars=2",
    "/v1/public_review_platform/business_ids_by_category_and_opening_time?category_name=Accessories&max_hour=7&time_format=%25AM",
    "/v1/public_review_platform/business_count_by_activity_state_and_opening_time?active=true&state=AZ&opening_time=12PM",
    "/v1/public_review_platform/category_names_by_user_id?user_id=16328",
    "/v1/public_review_platform/business_star_rating_difference?low_stars=2&high_stars=2&category_name=Food",
    "/v1/public_review_platform/business_category_percentage?category_name=Food",
    "/v1/public_review_platform/business_review_count_by_city?high_review_count=High&medium_review_count=Medium&low_review_count=Low&city=Cave%20Creek&active=true",
    "/v1/public_review_platform/average_user_id_by_year_range?start_year=2005&end_year=2015",
    "/v1/public_review_platform/active_inactive_business_ratio?active_status=true&inactive_status=false&review_count=Low",
    "/v1/public_review_platform/category_ids_names_by_pattern?pattern=P%25&limit=5",
    "/v1/public_review_platform/user_review_stars_by_business?business_id=15&review_length=Medium",
    "/v1/public_review_platform/business_active_status_by_category?category_name=Diagnostic%20Imaging",
    "/v1/public_review_platform/user_details_by_business_review_stars?business_id=143&review_stars=5",
    "/v1/public_review_platform/user_details_by_compliment_type_and_number?compliment_type=profile&number_of_compliments=Uber",
    "/v1/public_review_platform/top_tip_by_likes",
    "/v1/public_review_platform/user_compliments_by_year_range_and_type?start_year=2005&end_year=2014&compliment_type=photos",
    "/v1/public_review_platform/percentage_businesses_open_by_hours_and_day?opening_time=9AM&closing_time=9PM&day_of_week=Sunday",
    "/v1/public_review_platform/business_days_by_city_and_active_status?city=Black%20Canyon%20City&active=true",
    "/v1/public_review_platform/user_review_lengths_by_year_and_stars?user_yelping_since_year=2004&user_average_stars=5",
    "/v1/public_review_platform/distinct_businesses_by_review_stars_and_user_percentage?review_stars=4&user_percentage=65",
    "/v1/public_review_platform/business_diff_by_cities?city1=Glendale&city2=Mesa&active=true",
    "/v1/public_review_platform/total_likes_by_yelping_year?user_yelping_since_year=2010",
    "/v1/public_review_platform/most_common_tip_length?user_average_stars=3",
    "/v1/public_review_platform/total_likes_by_city?city=Goodyear",
    "/v1/public_review_platform/distinct_states_by_tip_length?tip_length=Long",
    "/v1/public_review_platform/total_operating_hours?city=El%20Mirage&state=AZ",
    "/v1/public_review_platform/day_id_difference?state=SC",
    "/v1/public_review_platform/distinct_categories_by_stars?stars=5",
    "/v1/public_review_platform/distinct_states_by_attribute_value?attribute_value=beer_and_wine",
    "/v1/public_review_platform/user_count_by_compliment?compliment_type=photos&number_of_compliments=Medium",
    "/v1/public_review_platform/count_active_businesses_in_city?city=Mesa&active=true",
    "/v1/public_review_platform/count_businesses_with_star_ratings?star1=1&star2=2",
    "/v1/public_review_platform/business_ids_by_star_city_review_count?min_stars=3&city=Paradise%20Valley&review_count=Low",
    "/v1/public_review_platform/count_business_attributes_greater_than?min_value=1",
    "/v1/public_review_platform/count_businesses_by_hours?opening_time=8AM&closing_time=6PM",
    "/v1/public_review_platform/count_active_businesses_funny_review_votes?review_votes_funny=Uber&active=true",
    "/v1/public_review_platform/count_users_compliments_fans?number_of_compliments=High&user_fans=None",
    "/v1/public_review_platform/most_common_compliment_type",
    "/v1/public_review_platform/top_users_by_average_stars",
    "/v1/public_review_platform/category_names_by_city?city=Arcadia",
    "/v1/public_review_platform/business_ids_by_day?day_id=1",
    "/v1/public_review_platform/count_distinct_businesses_by_review_length_active_city?review_length=Long&active=true&city=Phoenix",
    "/v1/public_review_platform/count_distinct_users_by_votes_cool?user_votes_cool=Low&review_votes_cool=Low",
    "/v1/public_review_platform/count_distinct_users_by_tip_length_likes_fans?tip_length=Long&likes=2&user_fans=High",
    "/v1/public_review_platform/count_businesses_by_attribute_values?attribute_name=ambience_trendy&attribute_value1=none&attribute_value2=no&attribute_value3=false",
    "/v1/public_review_platform/count_distinct_businesses_by_city_day_opening_time?city=Scottsdale&day_of_week=Sunday&opening_time=12PM",
    "/v1/public_review_platform/average_star_rating_by_category?category_name=Obstetricians%20%26%20Gynecologists",
    "/v1/public_review_platform/percentage_elite_users_high_fans_by_year?user_fans=High&actual_year=2011",
    "/v1/public_review_platform/business_ids_by_review_count_and_city?review_count=High&city=Tempe",
    "/v1/public_review_platform/business_count_by_review_count_state_active?review_count=Medium&state=AZ&active=true",
    "/v1/public_review_platform/business_ids_by_star_range_and_city?min_stars=3&max_stars=6&city=Chandler",
    "/v1/public_review_platform/user_count_by_yelping_since_year_and_fans?min_year=2009&max_year=2012&user_fans=Low",
    "/v1/public_review_platform/distinct_attribute_names_by_city_and_review_count?city=Tempe&review_count=Medium",
    "/v1/public_review_platform/count_distinct_businesses_by_category_and_stars?category_name=Food&max_stars=3",
    "/v1/public_review_platform/distinct_business_ids_and_stars_by_category_and_active?category_name=Food&active=true",
    "/v1/public_review_platform/distinct_category_and_attribute_names_by_max_stars",
    "/v1/public_review_platform/count_distinct_business_ids?attribute_name=BYOB&review_count=High&active=true",
    "/v1/public_review_platform/business_hours_highest_rated?city=Glendale",
    "/v1/public_review_platform/category_attribute_names?review_count=High&city=Goodyear&state=AZ&active=true",
    "/v1/public_review_platform/distinct_category_names_by_status_state_city?active=true&state=AZ&city=Glendale",
    "/v1/public_review_platform/distinct_cities_by_hours_day?opening_time=7AM&closing_time=7PM&day_of_week=Wednesday",
    "/v1/public_review_platform/distinct_attribute_values?state=AZ&city=Goodyear&active=true&stars=3&review_count=Low",
    "/v1/public_review_platform/percentage_high_stars_by_category?min_stars=3&category_name=Food",
    "/v1/public_review_platform/closing_times_days_above_star_threshold?active=true&city=Goodyear"
]
