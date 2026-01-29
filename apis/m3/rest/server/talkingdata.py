from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/talkingdata/talkingdata.sqlite')
cursor = conn.cursor()

# Endpoint to get the device ID of the oldest individual
@app.get("/v1/talkingdata/oldest_individual_device_id", operation_id="get_oldest_individual_device_id", summary="Retrieves the device ID of the individual with the highest age in the gender_age table.")
async def get_oldest_individual_device_id():
    cursor.execute("SELECT device_id FROM gender_age WHERE age = ( SELECT MAX(age) FROM gender_age )")
    result = cursor.fetchone()
    if not result:
        return {"device_id": []}
    return {"device_id": result[0]}

# Endpoint to get the count of events at a specific latitude and longitude
@app.get("/v1/talkingdata/event_count_by_location", operation_id="get_event_count_by_location", summary="Retrieves the total number of events that occurred at the specified geographical coordinates. The endpoint accepts latitude and longitude as input parameters to pinpoint the exact location of the events.")
async def get_event_count_by_location(latitude: float = Query(..., description="Latitude of the event"), longitude: float = Query(..., description="Longitude of the event")):
    cursor.execute("SELECT COUNT(event_id) FROM `events` WHERE latitude = ? AND longitude = ?", (latitude, longitude))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of individuals of a specific gender and age group
@app.get("/v1/talkingdata/gender_age_group_count", operation_id="get_gender_age_group_count", summary="Retrieves the count of individuals belonging to a specific gender and age group. The operation uses the provided gender and age group parameters to filter the data and return the corresponding count.")
async def get_gender_age_group_count(gender: str = Query(..., description="Gender of the individual"), group: str = Query(..., description="Age group of the individual")):
    cursor.execute("SELECT COUNT(gender) FROM gender_age WHERE gender = ? AND `group` = ?", (gender, group))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of individuals of a specific gender and age greater than a specified value
@app.get("/v1/talkingdata/gender_age_count", operation_id="get_gender_age_count", summary="Retrieves the count of individuals who are older than the specified age and belong to the given gender category.")
async def get_gender_age_count(age: int = Query(..., description="Minimum age of the individual"), gender: str = Query(..., description="Gender of the individual")):
    cursor.execute("SELECT COUNT(gender) FROM gender_age WHERE age > ? AND gender = ?", (age, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of active events for a specific event ID
@app.get("/v1/talkingdata/active_event_count", operation_id="get_active_event_count", summary="Retrieves the count of active events for a specific event ID. The active status of the event is used to filter the count. The event ID is required to identify the event.")
async def get_active_event_count(event_id: int = Query(..., description="ID of the event"), is_active: int = Query(..., description="Active status of the event (1 for active, 0 for inactive)")):
    cursor.execute("SELECT COUNT(is_active) FROM app_events WHERE event_id = ? AND is_active = ?", (event_id, is_active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the gender of the youngest individual
@app.get("/v1/talkingdata/youngest_individual_gender", operation_id="get_youngest_individual_gender", summary="Retrieves the gender of the youngest individual in the dataset. This operation identifies the minimum age in the gender_age table and returns the corresponding gender.")
async def get_youngest_individual_gender():
    cursor.execute("SELECT gender FROM gender_age WHERE age = ( SELECT MIN(age) FROM gender_age )")
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the most common app category
@app.get("/v1/talkingdata/most_common_app_category", operation_id="get_most_common_app_category", summary="Retrieves the app category that appears most frequently across all apps. This operation calculates the number of apps associated with each category and returns the category with the highest count.")
async def get_most_common_app_category():
    cursor.execute("SELECT T.category FROM ( SELECT T2.category, COUNT(T1.app_id) AS num FROM app_labels AS T1 INNER JOIN label_categories AS T2 ON T2.label_id = T1.label_id GROUP BY T1.app_id, T2.category ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"category": []}
    return {"category": result[0]}

# Endpoint to get the device model of the oldest individual
@app.get("/v1/talkingdata/oldest_individual_device_model", operation_id="get_oldest_individual_device_model", summary="Retrieves the device model of the oldest individual in the database. This operation identifies the oldest individual based on their age and returns the device model associated with their device ID. The result provides insights into the device model used by the oldest individual in the dataset.")
async def get_oldest_individual_device_model():
    cursor.execute("SELECT T1.device_model FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T2.device_id = T1.device_id ORDER BY T2.age DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"device_model": []}
    return {"device_model": result[0]}

# Endpoint to get the count of apps in a specific category
@app.get("/v1/talkingdata/app_count_by_category", operation_id="get_app_count_by_category", summary="Retrieves the total number of applications that belong to a specified category. The category is provided as an input parameter.")
async def get_app_count_by_category(category: str = Query(..., description="Category of the app")):
    cursor.execute("SELECT COUNT(T1.app_id) FROM app_labels AS T1 INNER JOIN label_categories AS T2 ON T2.label_id = T1.label_id WHERE T2.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of active events for a specific gender on a specific date
@app.get("/v1/talkingdata/active_event_count_by_gender_date", operation_id="get_active_event_count_by_gender_date", summary="Retrieve the total number of active events associated with a specific gender on a given date. The operation filters events based on their active status, gender, and date, providing a precise count of qualifying events.")
async def get_active_event_count_by_gender_date(is_active: int = Query(..., description="Active status of the event (1 for active, 0 for inactive)"), gender: str = Query(..., description="Gender of the individual"), timestamp: str = Query(..., description="Date in 'YYYY-MM-DD%' format")):
    cursor.execute("SELECT COUNT(T3.gender) FROM app_events AS T1 INNER JOIN events_relevant AS T2 ON T2.event_id = T1.event_id INNER JOIN gender_age AS T3 ON T3.device_id = T2.device_id WHERE T1.is_active = ? AND T3.gender = ? AND T2.timestamp LIKE ?", (is_active, gender, timestamp))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of a specific gender using a specific device model
@app.get("/v1/talkingdata/count_gender_device_model", operation_id="get_count_gender_device_model", summary="Retrieves the count of users with a specific gender who use a particular device model. The operation filters users based on their gender and device model, and returns the total count of users that match the specified criteria.")
async def get_count_gender_device_model(gender: str = Query(..., description="Gender of the user"), device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT COUNT(T1.gender) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T2.device_id = T1.device_id WHERE T1.gender = ? AND T2.device_model = ?", (gender, device_model))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the age of the oldest user based on specific event criteria
@app.get("/v1/talkingdata/oldest_user_age_by_event_criteria", operation_id="get_oldest_user_age_by_event_criteria", summary="Retrieves the age of the oldest user who has an active event that matches the provided event criteria, including the event's longitude, latitude, and timestamp in 'YYYY-MM-DD' format.")
async def get_oldest_user_age_by_event_criteria(is_active: int = Query(..., description="Is the event active (1 for active, 0 for inactive)"), longitude: float = Query(..., description="Longitude of the event"), latitude: float = Query(..., description="Latitude of the event"), timestamp: str = Query(..., description="Timestamp of the event in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T3.age FROM app_events AS T1 INNER JOIN events_relevant AS T2 ON T1.event_id = T2.event_id INNER JOIN gender_age AS T3 ON T2.device_id = T3.device_id WHERE T1.is_active = ? AND T2.longitude = ? AND T2.latitude = ? AND SUBSTR(T2.timestamp, 1, 10) = ? ORDER BY T3.age DESC LIMIT 1", (is_active, longitude, latitude, timestamp))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the device model of the latest user in a specific group and gender
@app.get("/v1/talkingdata/latest_device_model_by_group_gender", operation_id="get_latest_device_model_by_group_gender", summary="Retrieves the device model of the most recent user within a specified group and gender. The group and gender are used to filter the user data, and the result is determined by the highest device ID.")
async def get_latest_device_model_by_group_gender(group: str = Query(..., description="Group of the user"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT T2.device_model FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.`group` = ? AND T1.gender = ? ORDER BY T2.device_id DESC LIMIT 1", (group, gender))
    result = cursor.fetchone()
    if not result:
        return {"device_model": []}
    return {"device_model": result[0]}

# Endpoint to get the categories of the first N events
@app.get("/v1/talkingdata/event_categories", operation_id="get_event_categories", summary="Retrieves the categories of the most recent events, up to the specified limit. The operation returns the categories associated with the first N events, sorted by their timestamp in descending order.")
async def get_event_categories(limit: int = Query(..., description="Number of events to retrieve")):
    cursor.execute("SELECT T4.category FROM events_relevant AS T1 INNER JOIN app_events_relevant AS T2 ON T1.event_id = T2.event_id INNER JOIN app_labels AS T3 ON T3.app_id = T2.app_id INNER JOIN label_categories AS T4 ON T3.label_id = T4.label_id ORDER BY T1.timestamp LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get the most common gender for a specific phone brand
@app.get("/v1/talkingdata/most_common_gender_by_phone_brand", operation_id="get_most_common_gender_by_phone_brand", summary="Retrieves the most frequently occurring gender associated with a given phone brand. The operation identifies the gender with the highest count for the specified phone brand, based on the device_id association in the phone_brand_device_model2 and gender_age tables.")
async def get_most_common_gender_by_phone_brand(phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT T.gender FROM ( SELECT T2.gender, COUNT(T2.gender) AS num FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T2.device_id = T1.device_id WHERE T1.phone_brand = ? GROUP BY T2.gender ) AS T ORDER BY T.num DESC LIMIT 1", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the count of apps in a specific category
@app.get("/v1/talkingdata/count_apps_by_category", operation_id="get_count_apps_by_category", summary="Retrieves the total number of applications that belong to a specified category. The category is provided as an input parameter.")
async def get_count_apps_by_category(category: str = Query(..., description="Category of the app")):
    cursor.execute("SELECT COUNT(T2.app_id) FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T2.label_id = T1.label_id WHERE T1.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of female and male users for a specific phone brand
@app.get("/v1/talkingdata/gender_percentage_by_phone_brand", operation_id="get_gender_percentage_by_phone_brand", summary="Retrieves the percentage of female and male users for a specified phone brand. The calculation is based on the total number of users for the given phone brand. The input parameter is the phone brand, which is used to filter the data.")
async def get_gender_percentage_by_phone_brand(phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT SUM(IIF(T2.gender = 'F', 1, 0)) * 100 / COUNT(T2.device_id) AS perFemale , SUM(IIF(T2.gender = 'M', 1, 0)) * 100 / COUNT(T2.device_id) AS perMale FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T2.device_id = T1.device_id WHERE T1.phone_brand = ?", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": {"female": result[0], "male": result[1]}}

# Endpoint to get the longitude and latitude of events on a specific date
@app.get("/v1/talkingdata/event_locations_by_date", operation_id="get_event_locations_by_date", summary="Retrieves the geographical coordinates (longitude and latitude) of events that occurred on a specific date. The date is provided in the 'YYYY-MM-DD' format.")
async def get_event_locations_by_date(timestamp: str = Query(..., description="Timestamp of the event in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT longitude, latitude FROM `events` WHERE SUBSTR(`timestamp`, 1, 10) = ?", (timestamp,))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": [{"longitude": row[0], "latitude": row[1]} for row in result]}

# Endpoint to get the installation status of an app for a specific event
@app.get("/v1/talkingdata/app_installation_status", operation_id="get_app_installation_status", summary="Retrieves the installation status of an app associated with a specific event. The status indicates whether the app was installed ('YES') or not ('NO'). The event is identified by its unique event ID.")
async def get_app_installation_status(event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT app_id , IIF(is_installed = 1, 'YES', 'NO') AS status FROM app_events WHERE event_id = ?", (event_id,))
    result = cursor.fetchall()
    if not result:
        return {"status": []}
    return {"status": [{"app_id": row[0], "status": row[1]} for row in result]}

# Endpoint to get the count of events on a specific date
@app.get("/v1/talkingdata/count_events_by_date", operation_id="get_count_events_by_date", summary="Retrieves the total number of events that occurred on a specific date. The date is provided in the 'YYYY-MM-DD' format.")
async def get_count_events_by_date(timestamp: str = Query(..., description="Timestamp of the event in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(event_id) FROM events WHERE SUBSTR(`timestamp`, 1, 10) = ?", (timestamp,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of devices based on device model and phone brand
@app.get("/v1/talkingdata/device_count_by_model_and_brand", operation_id="get_device_count", summary="Retrieves the total number of unique devices for a given device model and phone brand. The operation requires the device model and phone brand as input parameters to filter the count of devices accordingly.")
async def get_device_count(device_model: str = Query(..., description="Device model"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT COUNT(device_id) FROM phone_brand_device_model2 WHERE device_model = ? AND phone_brand = ?", (device_model, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of male to female counts in a specific age group
@app.get("/v1/talkingdata/gender_ratio_by_group", operation_id="get_gender_ratio", summary="Retrieves the ratio of counts for a specific gender and age group compared to another gender and age group. This operation calculates the proportion of the first group relative to the second group, providing insights into their relative representation.")
async def get_gender_ratio(gender_m: str = Query(..., description="Gender for the numerator (e.g., 'M')"), group_m: str = Query(..., description="Group for the numerator (e.g., 'M27-28')"), gender_f: str = Query(..., description="Gender for the denominator (e.g., 'F')"), group_f: str = Query(..., description="Group for the denominator (e.g., 'F27-28')")):
    cursor.execute("SELECT SUM(IIF(gender = ? AND `group` = ?, 1, 0)) / SUM(IIF(gender = ? AND `group` = ?, 1, 0)) AS r FROM gender_age", (gender_m, group_m, gender_f, group_f))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get label IDs based on categories
@app.get("/v1/talkingdata/label_ids_by_categories", operation_id="get_label_ids", summary="Retrieves the unique label identifiers associated with the specified categories. The operation accepts two categories as input parameters and returns the corresponding label IDs from the label_categories table.")
async def get_label_ids(category1: str = Query(..., description="First category"), category2: str = Query(..., description="Second category")):
    cursor.execute("SELECT label_id FROM label_categories WHERE category IN (?, ?)", (category1, category2))
    result = cursor.fetchall()
    if not result:
        return {"label_ids": []}
    return {"label_ids": [row[0] for row in result]}

# Endpoint to get distinct phone brands and device models based on timestamp, longitude, and latitude
@app.get("/v1/talkingdata/distinct_phone_brands_and_models", operation_id="get_distinct_phone_brands_and_models", summary="Retrieves a list of unique phone brands and their corresponding device models based on a specific date, longitude, and latitude. The data is filtered by the provided timestamp, longitude, and latitude, ensuring that only distinct combinations of phone brands and device models are returned.")
async def get_distinct_phone_brands_and_models(timestamp: str = Query(..., description="Timestamp in 'YYYY-MM-DD%' format"), longitude: float = Query(..., description="Longitude"), latitude: float = Query(..., description="Latitude")):
    cursor.execute("SELECT DISTINCT T2.phone_brand, T2.device_model FROM events AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T2.device_id = T1.device_id WHERE T1.timestamp LIKE ? AND T1.longitude = ? AND T1.latitude = ?", (timestamp, longitude, latitude))
    result = cursor.fetchall()
    if not result:
        return {"phone_brands_and_models": []}
    return {"phone_brands_and_models": [{"phone_brand": row[0], "device_model": row[1]} for row in result]}

# Endpoint to get app IDs and timestamps based on event ID
@app.get("/v1/talkingdata/app_ids_and_timestamps_by_event_id", operation_id="get_app_ids_and_timestamps", summary="Retrieves the application IDs and their corresponding timestamps associated with the specified event ID. This operation fetches data from the app_events and events tables, joining them based on the event_id field. The result is a list of app_id and timestamp pairs that match the provided event_id.")
async def get_app_ids_and_timestamps(event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT T1.app_id, T2.timestamp FROM app_events AS T1 INNER JOIN events AS T2 ON T2.event_id = T1.event_id WHERE T2.event_id = ?", (event_id,))
    result = cursor.fetchall()
    if not result:
        return {"app_ids_and_timestamps": []}
    return {"app_ids_and_timestamps": [{"app_id": row[0], "timestamp": row[1]} for row in result]}

# Endpoint to get gender and age based on event ID
@app.get("/v1/talkingdata/gender_and_age_by_event_id", operation_id="get_gender_and_age", summary="Retrieves the gender and age of users associated with a specific event, identified by its unique event ID. The data is obtained by joining the gender_age and events tables using the device_id as the common key.")
async def get_gender_and_age(event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT T1.gender, T1.age FROM gender_age AS T1 INNER JOIN events AS T2 ON T2.device_id = T1.device_id WHERE T2.event_id = ?", (event_id,))
    result = cursor.fetchall()
    if not result:
        return {"gender_and_age": []}
    return {"gender_and_age": [{"gender": row[0], "age": row[1]} for row in result]}

# Endpoint to get the count of events based on gender, date, and age
@app.get("/v1/talkingdata/event_count_by_gender_date_age", operation_id="get_event_count", summary="Retrieves the total count of events that occurred on a specific date, filtered by gender and age. The response provides a comprehensive view of event activity based on these demographic factors.")
async def get_event_count(gender: str = Query(..., description="Gender"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), age: int = Query(..., description="Age")):
    cursor.execute("SELECT COUNT(T2.event_id) FROM gender_age AS T1 INNER JOIN events AS T2 ON T2.device_id = T1.device_id WHERE T1.gender = ? AND SUBSTR(`timestamp`, 1, 10) = ? AND T1.age = ?", (gender, date, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct age, gender, and event count based on longitude and latitude
@app.get("/v1/talkingdata/distinct_age_gender_event_count", operation_id="get_distinct_age_gender_event_count", summary="Retrieves the distinct count of events, categorized by age and gender, for a specific geographical location identified by longitude and latitude.")
async def get_distinct_age_gender_event_count(longitude: float = Query(..., description="Longitude"), latitude: float = Query(..., description="Latitude")):
    cursor.execute("SELECT DISTINCT T1.age, T1.gender, COUNT(T2.event_id) FROM gender_age AS T1 INNER JOIN `events` AS T2 ON T2.device_id = T1.device_id WHERE T2.longitude = ? AND T2.latitude = ? GROUP BY T1.age, T1.gender, T2.longitude, T2.latitude", (longitude, latitude))
    result = cursor.fetchall()
    if not result:
        return {"age_gender_event_count": []}
    return {"age_gender_event_count": [{"age": row[0], "gender": row[1], "event_count": row[2]} for row in result]}

# Endpoint to get distinct phone brands and device models based on longitude and latitude
@app.get("/v1/talkingdata/distinct_phone_brands_and_models_by_location", operation_id="get_distinct_phone_brands_and_models_by_location", summary="Retrieves a list of unique phone brands and device models associated with a specific geographical location, identified by longitude and latitude.")
async def get_distinct_phone_brands_and_models_by_location(longitude: float = Query(..., description="Longitude"), latitude: float = Query(..., description="Latitude")):
    cursor.execute("SELECT DISTINCT T1.phone_brand, T1.device_model FROM phone_brand_device_model2 AS T1 INNER JOIN events AS T2 ON T2.device_id = T1.device_id WHERE T2.longitude = ? AND T2.latitude = ?", (longitude, latitude))
    result = cursor.fetchall()
    if not result:
        return {"phone_brands_and_models": []}
    return {"phone_brands_and_models": [{"phone_brand": row[0], "device_model": row[1]} for row in result]}

# Endpoint to get distinct categories based on event ID
@app.get("/v1/talkingdata/distinct_categories_by_event_id", operation_id="get_distinct_categories", summary="Retrieves a unique set of categories associated with a specific event ID. This operation identifies distinct categories by examining the relationships between event IDs, app IDs, and label IDs in the database. The provided event ID is used to filter the results.")
async def get_distinct_categories(event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT DISTINCT T1.category FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T2.label_id = T1.label_id INNER JOIN app_events AS T3 ON T3.app_id = T2.app_id WHERE T3.event_id = ?", (event_id,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get device IDs based on device model, gender, and phone brand
@app.get("/v1/talkingdata/device_ids_by_model_gender_brand", operation_id="get_device_ids", summary="Retrieves a limited set of device IDs that match the specified device model, gender, and phone brand. The endpoint filters the data based on the provided parameters and returns up to five device IDs that meet the criteria.")
async def get_device_ids(device_model: str = Query(..., description="Device model"), gender: str = Query(..., description="Gender"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT T2.device_id FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T2.device_id = T1.device_id WHERE T1.device_model = ? AND T2.gender = ? AND T1.phone_brand = ? LIMIT 5", (device_model, gender, phone_brand))
    result = cursor.fetchall()
    if not result:
        return {"device_ids": []}
    return {"device_ids": [row[0] for row in result]}

# Endpoint to get age and gender based on phone brand and device model
@app.get("/v1/talkingdata/age_gender_by_brand_model", operation_id="get_age_gender", summary="Retrieves the age and gender distribution of users based on the specified phone brand and device model. The operation filters the data by the provided phone brand and device model, and returns the corresponding age and gender information.")
async def get_age_gender(phone_brand: str = Query(..., description="Phone brand"), device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT T2.age, T2.gender FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T2.device_id = T1.device_id WHERE T1.phone_brand = ? AND T1.device_model = ?", (phone_brand, device_model))
    result = cursor.fetchall()
    if not result:
        return {"age_gender": []}
    return {"age_gender": [{"age": row[0], "gender": row[1]} for row in result]}

# Endpoint to get the percentage of apps in a specific category
@app.get("/v1/talkingdata/percentage_apps_by_category", operation_id="get_percentage_apps", summary="Retrieves the percentage of apps that belong to a specified category. This operation calculates the proportion of apps in the given category by comparing the count of apps in that category to the total number of apps. The category is provided as an input parameter.")
async def get_percentage_apps(category: str = Query(..., description="Category")):
    cursor.execute("SELECT SUM(IIF(T1.category = ?, 1, 0)) * 100 / COUNT(T2.app_id) AS per FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T2.label_id = T1.label_id", (category,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage and ratio of device models by gender and phone brand
@app.get("/v1/talkingdata/percentage_ratio_device_model_gender_brand", operation_id="get_percentage_ratio", summary="Retrieves the percentage and ratio of a specific device model used by each gender for a given phone brand. The calculation is based on the total number of devices and the count of devices used by each gender.")
async def get_percentage_ratio(device_model: str = Query(..., description="Device model"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT SUM(IIF(T1.device_model = ?, 1, 0)) * 100 / COUNT(T1.device_id) AS per, SUM(IIF(T1.device_model = ? AND T2.gender = 'M', 1, 0)) / SUM(IIF(T1.device_model = ? AND T2.gender = 'F', 1, 0)) AS r FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T2.device_id = T1.device_id WHERE T1.phone_brand = ?", (device_model, device_model, device_model, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"percentage": [], "ratio": []}
    return {"percentage": result[0], "ratio": result[1]}

# Endpoint to get the count of events based on event ID and active status
@app.get("/v1/talkingdata/count_events_by_id_active", operation_id="get_count_events", summary="Retrieves the total count of a specific event, filtered by its unique identifier and active status. The active status indicates whether the event is currently active (1) or inactive (0).")
async def get_count_events(event_id: int = Query(..., description="Event ID"), is_active: int = Query(..., description="Active status (0 or 1)")):
    cursor.execute("SELECT COUNT(event_id) FROM app_events WHERE event_id = ? AND is_active = ?", (event_id, is_active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of events based on year
@app.get("/v1/talkingdata/count_events_by_year", operation_id="get_count_events_by_year", summary="Retrieves the total count of events that occurred in a specified year. The year is provided in the 'YYYY' format.")
async def get_count_events_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(event_id) FROM `events` WHERE SUBSTR(`timestamp`, 1, 4) = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of events based on year and device ID
@app.get("/v1/talkingdata/count_events_by_year_device", operation_id="get_count_events_by_year_device", summary="Retrieves the total number of events that occurred in a specific year for a given device. The year is provided in 'YYYY' format, and the device is identified by its unique ID.")
async def get_count_events_by_year_device(year: str = Query(..., description="Year in 'YYYY' format"), device_id: int = Query(..., description="Device ID")):
    cursor.execute("SELECT COUNT(event_id) FROM `events` WHERE SUBSTR(`timestamp`, 1, 4) = ? AND device_id = ?", (year, device_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of devices based on gender
@app.get("/v1/talkingdata/count_devices_by_gender", operation_id="get_count_devices_by_gender", summary="Retrieves the total count of unique devices used by a specific gender. The gender is provided as an input parameter.")
async def get_count_devices_by_gender(gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT COUNT(device_id) FROM gender_age WHERE gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum age
@app.get("/v1/talkingdata/max_age", operation_id="get_max_age", summary="Retrieves the maximum age value from the gender_age table.")
async def get_max_age():
    cursor.execute("SELECT MAX(age) FROM gender_age")
    result = cursor.fetchone()
    if not result:
        return {"max_age": []}
    return {"max_age": result[0]}

# Endpoint to get the count of devices based on gender and device model
@app.get("/v1/talkingdata/device_count_gender_model", operation_id="get_device_count_gender_model", summary="Retrieves the total count of devices used by individuals of a specified gender, filtered by a particular device model. This operation provides insights into device usage patterns based on gender and device model.")
async def get_device_count_gender_model(gender: str = Query(..., description="Gender of the user"), device_model: str = Query(..., description="Model of the device")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T2.device_id = T1.device_id WHERE T2.gender = ? AND T1.device_model = ?", (gender, device_model))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ages of users based on device model
@app.get("/v1/talkingdata/user_ages_device_model", operation_id="get_user_ages_device_model", summary="Retrieves the age distribution of users who utilize a specified device model. The operation filters users based on the provided device model and returns their respective ages.")
async def get_user_ages_device_model(device_model: str = Query(..., description="Model of the device")):
    cursor.execute("SELECT T2.age FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T2.device_id = T1.device_id WHERE T1.device_model = ?", (device_model,))
    result = cursor.fetchall()
    if not result:
        return {"ages": []}
    return {"ages": [row[0] for row in result]}

# Endpoint to get the device models of the oldest users
@app.get("/v1/talkingdata/device_models_oldest_users", operation_id="get_device_models_oldest_users", summary="Retrieves the device models used by the oldest users. This operation identifies the maximum age from the user data and returns the device models associated with users of that age.")
async def get_device_models_oldest_users():
    cursor.execute("SELECT device_model FROM phone_brand_device_model2 WHERE device_id IN ( SELECT device_id FROM gender_age WHERE age = ( SELECT MAX(age) FROM gender_age ) )")
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get the most common group for a specific phone brand
@app.get("/v1/talkingdata/most_common_group_phone_brand", operation_id="get_most_common_group_phone_brand", summary="Retrieves the most prevalent user group associated with a specified phone brand. The user group is determined based on the count of device IDs linked to the given phone brand. The result is ordered in descending order by the count of device IDs, and the top group is returned.")
async def get_most_common_group_phone_brand(phone_brand: str = Query(..., description="Brand of the phone")):
    cursor.execute("SELECT T.`group` FROM ( SELECT T2.`group`, COUNT(`group`) AS num FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T2.device_id = T1.device_id WHERE T1.phone_brand = ? GROUP BY T2.`group` ) AS T ORDER BY T.num DESC LIMIT 1", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"group": []}
    return {"group": result[0]}

# Endpoint to get the categories of a specific app
@app.get("/v1/talkingdata/app_categories", operation_id="get_app_categories", summary="Retrieves the categories associated with a specific app. The app is identified by its unique ID, which is used to look up corresponding category information from the label_categories and app_labels tables.")
async def get_app_categories(app_id: int = Query(..., description="ID of the app")):
    cursor.execute("SELECT T1.category FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id WHERE T2.app_id = ?", (app_id,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get distinct categories of apps based on event ID and activity status
@app.get("/v1/talkingdata/distinct_app_categories_event_active", operation_id="get_distinct_app_categories", summary="Retrieves a unique set of app categories associated with a specific event and activity status. The operation filters apps based on the provided event ID and activity status, then identifies the distinct categories to which these apps belong.")
async def get_distinct_app_categories(event_id: int = Query(..., description="ID of the event"), is_active: int = Query(..., description="Activity status of the event (0 or 1)")):
    cursor.execute("SELECT DISTINCT T1.category FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id INNER JOIN app_events AS T3 ON T2.app_id = T3.app_id WHERE T3.event_id = ? AND T3.is_active = ?", (event_id, is_active))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get distinct locations of events based on event ID and activity status
@app.get("/v1/talkingdata/distinct_event_locations", operation_id="get_distinct_event_locations", summary="Retrieves the unique geographical coordinates of a specific event, based on its ID and activity status. The response includes longitude and latitude values, providing the precise location of the event.")
async def get_distinct_event_locations(event_id: int = Query(..., description="ID of the event"), is_active: int = Query(..., description="Activity status of the event (0 or 1)")):
    cursor.execute("SELECT DISTINCT T2.longitude, T2.latitude FROM app_events AS T1 INNER JOIN events AS T2 ON T2.event_id = T1.event_id WHERE T2.event_id = ? AND T1.is_active = ?", (event_id, is_active))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": [{"longitude": row[0], "latitude": row[1]} for row in result]}

# Endpoint to get the earliest timestamp of an event based on event ID and activity status
@app.get("/v1/talkingdata/earliest_event_timestamp", operation_id="get_earliest_event_timestamp", summary="Retrieves the earliest timestamp associated with a specific event, based on the provided event ID and activity status. The operation filters events by their activity status and returns the earliest timestamp of the matching event.")
async def get_earliest_event_timestamp(is_active: int = Query(..., description="Activity status of the event (0 or 1)"), event_id: int = Query(..., description="ID of the event")):
    cursor.execute("SELECT T2.timestamp FROM app_events AS T1 INNER JOIN events AS T2 ON T2.event_id = T1.event_id WHERE T1.is_active = ? AND T2.event_id = ? ORDER BY T2.timestamp LIMIT 1", (is_active, event_id))
    result = cursor.fetchone()
    if not result:
        return {"timestamp": []}
    return {"timestamp": result[0]}

# Endpoint to get event IDs based on phone brand
@app.get("/v1/talkingdata/event_ids_by_phone_brand", operation_id="get_event_ids_by_phone_brand", summary="Retrieves a list of event IDs associated with a given phone brand. The operation filters events based on the provided phone brand and returns the corresponding event IDs.")
async def get_event_ids_by_phone_brand(phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT T2.event_id FROM phone_brand_device_model2 AS T1 INNER JOIN events AS T2 ON T2.device_id = T1.device_id WHERE T1.phone_brand = ?", (phone_brand,))
    result = cursor.fetchall()
    if not result:
        return {"event_ids": []}
    return {"event_ids": [row[0] for row in result]}

# Endpoint to get the count of devices based on phone brand and event ID
@app.get("/v1/talkingdata/device_count_by_phone_brand_event_id", operation_id="get_device_count_by_phone_brand_event_id", summary="Retrieves the total number of devices associated with a specific phone brand and event ID. The operation requires the phone brand and event ID as input parameters to filter the data and provide an accurate count.")
async def get_device_count_by_phone_brand_event_id(phone_brand: str = Query(..., description="Phone brand"), event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM phone_brand_device_model2 AS T1 INNER JOIN events AS T2 ON T2.device_id = T1.device_id WHERE T1.phone_brand = ? AND T2.event_id = ?", (phone_brand, event_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get timestamps based on phone brand and event ID
@app.get("/v1/talkingdata/timestamps_by_phone_brand_event_id", operation_id="get_timestamps_by_phone_brand_event_id", summary="Retrieves a list of timestamps associated with a specific phone brand and event ID. The operation filters events based on the provided phone brand and event ID, and returns the corresponding timestamps.")
async def get_timestamps_by_phone_brand_event_id(phone_brand: str = Query(..., description="Phone brand"), event_id: str = Query(..., description="Event ID")):
    cursor.execute("SELECT T1.timestamp FROM events AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.event_id = T2.device_id WHERE T2.phone_brand = ? AND T1.event_id = ?", (phone_brand, event_id))
    result = cursor.fetchall()
    if not result:
        return {"timestamps": []}
    return {"timestamps": [row[0] for row in result]}

# Endpoint to get the count of events based on year and phone brand
@app.get("/v1/talkingdata/event_count_by_year_phone_brand", operation_id="get_event_count_by_year_phone_brand", summary="Retrieves the total number of events that occurred in a specified year and were associated with a particular phone brand. The year is provided in 'YYYY' format, and the phone brand is identified by its name.")
async def get_event_count_by_year_phone_brand(year: str = Query(..., description="Year in 'YYYY' format"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT COUNT(T1.event_id) FROM events AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.event_id = T2.device_id WHERE STRFTIME('%Y', T1.timestamp) = ? AND T2.phone_brand = ?", (year, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of devices based on gender, phone brand, and age
@app.get("/v1/talkingdata/device_count_by_gender_phone_brand_age", operation_id="get_device_count_by_gender_phone_brand_age", summary="Retrieves the total number of devices used by individuals of a specific gender, who own a particular phone brand, and are under a certain age. The response is based on the aggregated data from the gender-age and phone brand-device model datasets.")
async def get_device_count_by_gender_phone_brand_age(gender: str = Query(..., description="Gender"), phone_brand: str = Query(..., description="Phone brand"), age: int = Query(..., description="Age")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.gender = ? AND T2.phone_brand = ? AND T1.age < ?", (gender, phone_brand, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top category based on the number of apps
@app.get("/v1/talkingdata/top_category_by_app_count", operation_id="get_top_category_by_app_count", summary="Retrieves the category with the highest number of associated apps. The operation calculates the count of apps linked to each category and returns the category with the maximum count.")
async def get_top_category_by_app_count():
    cursor.execute("SELECT T.category FROM ( SELECT T1.category, COUNT(T2.app_id) AS num FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id GROUP BY T1.label_id ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"category": []}
    return {"category": result[0]}

# Endpoint to get phone brands based on the youngest device ID for a specific gender
@app.get("/v1/talkingdata/phone_brands_by_youngest_device_id_gender", operation_id="get_phone_brands_by_youngest_device_id_gender", summary="Retrieves the phone brands associated with the youngest individual of a specified gender. The data is filtered based on the youngest device ID linked to the given gender in the gender_age table.")
async def get_phone_brands_by_youngest_device_id_gender(gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT phone_brand FROM phone_brand_device_model2 WHERE device_id IN ( SELECT * FROM ( SELECT device_id FROM gender_age WHERE gender = ? ORDER BY age LIMIT 1 ) AS T )", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"phone_brands": []}
    return {"phone_brands": [row[0] for row in result]}

# Endpoint to get the count of devices based on group and phone brand
@app.get("/v1/talkingdata/device_count_by_group_phone_brand", operation_id="get_device_count_by_group_phone_brand", summary="Retrieves the total number of devices associated with a particular group and phone brand. The group and phone brand are specified as input parameters, allowing for a targeted count of devices that meet the given criteria.")
async def get_device_count_by_group_phone_brand(group: str = Query(..., description="Group"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT COUNT(T2.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.`group` = ? AND T2.phone_brand = ?", (group, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of devices in a specific group for a given phone brand
@app.get("/v1/talkingdata/percentage_devices_in_group_by_phone_brand", operation_id="get_percentage_devices_in_group_by_phone_brand", summary="Retrieves the proportion of devices belonging to a specific group, categorized by a given phone brand. The group parameter filters devices based on their assigned group, while the phone brand parameter specifies the brand of the devices to be considered.")
async def get_percentage_devices_in_group_by_phone_brand(group: str = Query(..., description="Group"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT SUM(IIF(T1.`group` = ?, 1.0, 0)) / COUNT(T1.device_id) AS per FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ?", (group, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of devices with a specific phone brand for a given event ID
@app.get("/v1/talkingdata/percentage_devices_by_phone_brand_event_id", operation_id="get_percentage_devices_by_phone_brand_event_id", summary="Retrieves the proportion of devices associated with a specific phone brand for a given event. The operation calculates this percentage by comparing the count of devices with the specified phone brand to the total number of devices for the event.")
async def get_percentage_devices_by_phone_brand_event_id(phone_brand: str = Query(..., description="Phone brand"), event_id: str = Query(..., description="Event ID")):
    cursor.execute("SELECT SUM(IIF(T2.phone_brand = ?, 1, 0)) / COUNT(T1.device_id) AS per FROM events AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.event_id = T2.device_id WHERE T1.event_id = ?", (phone_brand, event_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average age of users with a specific phone brand
@app.get("/v1/talkingdata/average_age_by_phone_brand", operation_id="get_average_age_by_phone_brand", summary="Retrieves the average age of users who own a specific phone brand. The phone brand is provided as an input parameter.")
async def get_average_age_by_phone_brand(phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT AVG(age) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ?", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the count of device IDs for a specific group and gender
@app.get("/v1/talkingdata/count_device_ids_by_group_and_gender", operation_id="get_count_device_ids_by_group_and_gender", summary="Retrieves the total number of unique device IDs associated with a particular group and gender. The group and gender are specified as input parameters, allowing for a targeted count of device IDs.")
async def get_count_device_ids_by_group_and_gender(group: str = Query(..., description="Group"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT COUNT(device_id) FROM gender_age WHERE `group` = ? AND gender = ?", (group, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum age of users with a specific gender
@app.get("/v1/talkingdata/max_age_by_gender", operation_id="get_max_age_by_gender", summary="Retrieves the maximum age of users who identify as the specified gender.")
async def get_max_age_by_gender(gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT MAX(age) FROM gender_age WHERE gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"max_age": []}
    return {"max_age": result[0]}

# Endpoint to get the count of app IDs based on installation and activity status
@app.get("/v1/talkingdata/count_app_ids_by_status", operation_id="get_count_app_ids_by_status", summary="Retrieves the total count of unique app IDs that match the specified installation and activity status. The installation status indicates whether the app is installed (1) or not (0), while the activity status denotes whether the app is currently active (1) or inactive (0).")
async def get_count_app_ids_by_status(is_installed: int = Query(..., description="Installation status (1 for installed, 0 for not installed)"), is_active: int = Query(..., description="Activity status (1 for active, 0 for inactive)")):
    cursor.execute("SELECT COUNT(app_id) FROM app_events WHERE is_installed = ? AND is_active = ?", (is_installed, is_active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the minimum age of users with a specific gender
@app.get("/v1/talkingdata/min_age_by_gender", operation_id="get_min_age_by_gender", summary="Retrieves the youngest age among users of a specified gender. The gender is provided as an input parameter.")
async def get_min_age_by_gender(gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT MIN(age) FROM gender_age WHERE gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"min_age": []}
    return {"min_age": result[0]}

# Endpoint to get the count of device IDs for a specific phone brand
@app.get("/v1/talkingdata/count_device_ids_by_phone_brand", operation_id="get_count_device_ids_by_phone_brand", summary="Retrieves the total number of unique device IDs associated with a specified phone brand. The response provides a count of devices that belong to the given brand.")
async def get_count_device_ids_by_phone_brand(phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT COUNT(device_id) FROM phone_brand_device_model2 WHERE phone_brand = ?", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get device models for a specific phone brand with a limit
@app.get("/v1/talkingdata/device_models_by_phone_brand", operation_id="get_device_models_by_phone_brand", summary="Retrieves a specified number of device models associated with a given phone brand. The operation returns a list of device models, up to the defined limit, that are linked to the provided phone brand.")
async def get_device_models_by_phone_brand(phone_brand: str = Query(..., description="Phone brand"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT device_model FROM phone_brand_device_model2 WHERE phone_brand = ? LIMIT ?", (phone_brand, limit))
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get device models for a specific group and gender with a limit
@app.get("/v1/talkingdata/device_models_by_group_and_gender", operation_id="get_device_models_by_group_and_gender", summary="Get device models for a specific group and gender with a limit")
async def get_device_models_by_group_and_gender(group: str = Query(..., description="Group"), gender: str = Query(..., description="Gender"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT T1.device_model FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T1.device_id = T2.device_id WHERE T2.`group` = ? AND T2.gender = ? LIMIT ?", (group, gender, limit))
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get device models based on event activity and installation status with a limit
@app.get("/v1/talkingdata/device_models_by_event_status", operation_id="get_device_models_by_event_status", summary="Retrieves a list of device models based on their event activity and installation status, with a specified limit. The activity status indicates whether the device is currently active or inactive, while the installation status specifies if the device has been installed or not. The limit parameter restricts the number of results returned.")
async def get_device_models_by_event_status(is_active: int = Query(..., description="Activity status (1 for active, 0 for inactive)"), is_installed: int = Query(..., description="Installation status (1 for installed, 0 for not installed)"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT T1.device_model FROM phone_brand_device_model2 AS T1 INNER JOIN events AS T2 ON T1.device_id = T2.event_id INNER JOIN app_events AS T3 ON T2.event_id = T3.event_id WHERE T3.is_active = ? AND T3.is_installed = ? LIMIT ?", (is_active, is_installed, limit))
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get the count of app IDs for a specific label category
@app.get("/v1/talkingdata/count_app_ids_by_label_category", operation_id="get_count_app_ids_by_label_category", summary="Retrieves the total number of unique app IDs associated with a given label category. The category parameter is used to filter the results.")
async def get_count_app_ids_by_label_category(category: str = Query(..., description="Label category")):
    cursor.execute("SELECT COUNT(T1.app_id) FROM app_labels AS T1 INNER JOIN label_categories AS T2 ON T1.label_id = T2.label_id WHERE T2.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of device IDs based on phone brand and gender
@app.get("/v1/talkingdata/device_count_by_brand_gender", operation_id="get_device_count_by_brand_gender", summary="Retrieves the total count of unique device IDs associated with a specified phone brand and gender. This operation combines data from the gender_age and phone_brand_device_model2 tables, filtering results based on the provided phone brand and gender parameters.")
async def get_device_count_by_brand_gender(phone_brand: str = Query(..., description="Phone brand"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ? AND T1.gender = ?", (phone_brand, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get device models for the youngest age
@app.get("/v1/talkingdata/device_models_youngest_age", operation_id="get_device_models_youngest_age", summary="Retrieves the device models associated with the youngest age group in the gender_age table. This operation identifies the minimum age from the gender_age table and returns the corresponding device models from the phone_brand_device_model2 table.")
async def get_device_models_youngest_age():
    cursor.execute("SELECT device_model FROM phone_brand_device_model2 WHERE device_id IN ( SELECT device_id FROM gender_age WHERE age = ( SELECT MIN(age) FROM gender_age ) )")
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get categories ordered by label ID with a limit
@app.get("/v1/talkingdata/categories_ordered_by_label_id", operation_id="get_categories_ordered_by_label_id", summary="Retrieves a specified number of categories, ordered by their associated label IDs. The operation returns the categories in ascending order of their corresponding label IDs, up to the defined limit.")
async def get_categories_ordered_by_label_id(limit: int = Query(..., description="Limit the number of categories returned")):
    cursor.execute("SELECT T1.category FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id ORDER BY T2.label_id LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get the count of device IDs based on device model and gender
@app.get("/v1/talkingdata/device_count_by_model_gender", operation_id="get_device_count_by_model_gender", summary="Retrieves the total count of unique device IDs associated with a specific device model and gender. This operation provides insights into the distribution of devices based on the given model and gender.")
async def get_device_count_by_model_gender(device_model: str = Query(..., description="Device model"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.device_model = ? AND T1.gender = ?", (device_model, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common group for a specific phone brand
@app.get("/v1/talkingdata/most_common_group_by_brand", operation_id="get_most_common_group_by_brand", summary="Retrieves the most frequently occurring group for a given phone brand. The group is determined based on the gender and age demographic data associated with the specified phone brand. The result is obtained by counting the number of occurrences of each group for the provided phone brand and returning the group with the highest count.")
async def get_most_common_group_by_brand(phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT T.`group` FROM ( SELECT T1.`group`, COUNT(T1.`group`) AS num FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ? GROUP BY T1.`group` ) AS T ORDER BY T.num DESC LIMIT 1", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"group": []}
    return {"group": result[0]}

# Endpoint to get the most common device models for a specific gender
@app.get("/v1/talkingdata/most_common_device_models_by_gender", operation_id="get_most_common_device_models_by_gender", summary="Retrieves the top device models used by a specific gender, sorted by popularity. The number of device models returned can be limited by the user.")
async def get_most_common_device_models_by_gender(gender: str = Query(..., description="Gender of the user"), limit: int = Query(..., description="Limit the number of device models returned")):
    cursor.execute("SELECT T.device_model FROM ( SELECT T2.device_model, COUNT(T2.device_model) AS num FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.gender = ? GROUP BY T2.device_model ) AS T ORDER BY T.num DESC LIMIT ?", (gender, limit))
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get the proportion of specific categories in app labels
@app.get("/v1/talkingdata/proportion_of_categories", operation_id="get_proportion_of_categories", summary="Retrieves the proportion of two specified categories in app labels. The calculation is based on the count of labels associated with each category, divided by the total count of labels. This operation provides insights into the distribution of categories within app labels.")
async def get_proportion_of_categories(category1: str = Query(..., description="First category to calculate proportion"), category2: str = Query(..., description="Second category to calculate proportion")):
    cursor.execute("SELECT SUM(IIF(T1.category = ?, 1, 0)) / COUNT(T1.label_id) AS J8 , SUM(IIF(T1.category = ?, 1, 0)) / COUNT(T1.label_id) AS J9 FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id", (category1, category2))
    result = cursor.fetchone()
    if not result:
        return {"proportions": []}
    return {"proportions": {"J8": result[0], "J9": result[1]}}

# Endpoint to get the proportion of a specific phone brand for a specific device ID and gender
@app.get("/v1/talkingdata/proportion_of_phone_brand", operation_id="get_proportion_of_phone_brand", summary="Retrieves the proportion of a specific phone brand used by users with a given device ID and gender. The calculation is based on the aggregated data from the phone_brand_device_model2 and gender_age tables, where the phone brand matches the provided input and the device ID is associated with the specified gender.")
async def get_proportion_of_phone_brand(phone_brand: str = Query(..., description="Phone brand"), device_id: str = Query(..., description="Device ID"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT SUM(IIF(T1.phone_brand = ?, 1, 0)) / SUM(IIF(T1.device_id = ?, 1, 0)) AS num FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T1.device_id = T2.device_id WHERE T2.gender = ?", (phone_brand, device_id, gender))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

# Endpoint to get the phone brand based on device model
@app.get("/v1/talkingdata/phone_brand_by_device_model", operation_id="get_phone_brand_by_device_model", summary="Retrieves the phone brand associated with a given device model. The device model is provided as an input parameter, and the operation returns the corresponding phone brand.")
async def get_phone_brand_by_device_model(device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT phone_brand FROM phone_brand_device_model2 WHERE device_model = ?", (device_model,))
    result = cursor.fetchall()
    if not result:
        return {"phone_brands": []}
    return {"phone_brands": [row[0] for row in result]}

# Endpoint to get the count of devices based on the device model
@app.get("/v1/talkingdata/device_count_by_model", operation_id="get_device_count_by_model", summary="Retrieves the total number of devices for a given device model. The device model is specified as an input parameter.")
async def get_device_count_by_model(device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT COUNT(device_id) FROM phone_brand_device_model2 WHERE device_model = ?", (device_model,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the group based on age
@app.get("/v1/talkingdata/group_by_age", operation_id="get_group_by_age", summary="Get the group for a specific age")
async def get_group_by_age(age: str = Query(..., description="Age")):
    cursor.execute("SELECT `group` FROM gender_age WHERE age = ?", (age,))
    result = cursor.fetchone()
    if not result:
        return {"group": []}
    return {"group": result[0]}

# Endpoint to get the timestamp based on event ID
@app.get("/v1/talkingdata/timestamp_by_event_id", operation_id="get_timestamp_by_event_id", summary="Retrieves the exact time (timestamp) associated with a specific event, identified by its unique event ID.")
async def get_timestamp_by_event_id(event_id: str = Query(..., description="Event ID")):
    cursor.execute("SELECT timestamp FROM events WHERE event_id = ?", (event_id,))
    result = cursor.fetchone()
    if not result:
        return {"timestamp": []}
    return {"timestamp": result[0]}

# Endpoint to get the device model based on longitude, latitude, and timestamp
@app.get("/v1/talkingdata/device_model_by_location_timestamp", operation_id="get_device_model_by_location_timestamp", summary="Retrieves the device model associated with a specific geographical location and timestamp. The endpoint uses the provided longitude, latitude, and timestamp to identify the device model from the database.")
async def get_device_model_by_location_timestamp(longitude: str = Query(..., description="Longitude"), latitude: str = Query(..., description="Latitude"), timestamp: str = Query(..., description="Timestamp in YYYY-MM-DD HH:MM:SS format")):
    cursor.execute("SELECT T1.device_model FROM phone_brand_device_model2 AS T1 INNER JOIN events AS T2 ON T1.device_id = T2.event_id WHERE T2.longitude = ? AND T2.latitude = ? AND T2.timestamp = ?", (longitude, latitude, timestamp))
    result = cursor.fetchone()
    if not result:
        return {"device_model": []}
    return {"device_model": result[0]}

# Endpoint to get the count of app IDs based on category
@app.get("/v1/talkingdata/app_id_count_by_category", operation_id="get_app_id_count_by_category", summary="Retrieves the total number of unique app IDs associated with a given category. The category is specified as an input parameter.")
async def get_app_id_count_by_category(category: str = Query(..., description="Category")):
    cursor.execute("SELECT COUNT(T2.app_id) FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id WHERE T1.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of device IDs based on group and device model
@app.get("/v1/talkingdata/device_id_count_by_group_model", operation_id="get_device_id_count_by_group_model", summary="Retrieves the total count of unique device IDs associated with a specific group and device model. This operation combines data from the gender_age and phone_brand_device_model2 tables, filtering results based on the provided group and device model parameters.")
async def get_device_id_count_by_group_model(group: str = Query(..., description="Group"), device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT COUNT(T2.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.`group` = ? AND T2.device_model = ?", (group, device_model))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of device IDs based on date, active status, and age
@app.get("/v1/talkingdata/device_id_count_by_date_active_age", operation_id="get_device_id_count_by_date_active_age", summary="Retrieves the total number of unique device IDs that were active on a specific date and belong to a certain age group. The date should be provided in the format YYYY-MM-DD, and the active status should be indicated as 1 for active devices. The age group is specified by the age parameter.")
async def get_device_id_count_by_date_active_age(date: str = Query(..., description="Date in YYYY-MM-DD format"), is_active: int = Query(..., description="Active status (1 for active)"), age: str = Query(..., description="Age")):
    cursor.execute("SELECT COUNT(T3.device_id) FROM app_events AS T1 INNER JOIN events AS T2 ON T1.event_id = T2.event_id INNER JOIN gender_age AS T3 ON T2.device_id = T3.device_id WHERE SUBSTR(`timestamp`, 1, 10) = ? AND T1.is_active = ? AND T3.age = ?", (date, is_active, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of event IDs based on date and installed status
@app.get("/v1/talkingdata/event_id_count_by_date_installed", operation_id="get_event_id_count_by_date_installed", summary="Retrieves the total count of unique event IDs that occurred on a specific date and have a specified installed status. The date should be provided in the YYYY-MM-DD format, and the installed status should be indicated as 1 for installed events.")
async def get_event_id_count_by_date_installed(date: str = Query(..., description="Date in YYYY-MM-DD format"), is_installed: str = Query(..., description="Installed status (1 for installed)")):
    cursor.execute("SELECT COUNT(T1.event_id) FROM app_events AS T1 INNER JOIN events AS T2 ON T1.event_id = T2.event_id WHERE SUBSTR(T2.`timestamp`, 1, 10) = ? AND T1.is_installed = ?", (date, is_installed))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of device IDs based on gender and phone brand
@app.get("/v1/talkingdata/device_id_count_by_gender_brand", operation_id="get_device_id_count_by_gender_brand", summary="Get the count of device IDs for a specific gender and phone brand")
async def get_device_id_count_by_gender_brand(gender: str = Query(..., description="Gender"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT COUNT(T2.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.gender = ? AND T2.phone_brand = ?", (gender, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of devices based on device model and gender
@app.get("/v1/talkingdata/count_devices_by_model_and_gender", operation_id="get_count_devices_by_model_and_gender", summary="Retrieves the total count of devices that match the specified device model and gender. The count is determined by querying the phone_brand_device_model2 and gender_age tables, which are joined based on the device_id. The device_model and gender parameters are used to filter the results.")
async def get_count_devices_by_model_and_gender(device_model: str = Query(..., description="Device model"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T1.device_id = T2.device_id WHERE T1.device_model = ? AND T2.gender = ?", (device_model, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of app events based on date, active status, and group
@app.get("/v1/talkingdata/count_app_events_by_date_status_group", operation_id="get_count_app_events_by_date_status_group", summary="Retrieves the total number of app events that occurred on a specific date, filtered by active status and group. The date should be provided in 'YYYY-MM-DD' format. The active status indicates whether the app was active during the event. The group refers to the gender and age category of the user associated with the event.")
async def get_count_app_events_by_date_status_group(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), is_active: str = Query(..., description="Active status"), group: str = Query(..., description="Group")):
    cursor.execute("SELECT COUNT(T1.app_id) FROM app_events AS T1 INNER JOIN events AS T2 ON T1.event_id = T2.event_id INNER JOIN gender_age AS T3 ON T2.event_id = T3.device_id WHERE SUBSTR(T2.`timestamp`, 1, 10) = ? AND T1.is_active = ? AND T3.`group` = ?", (date, is_active, group))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of app events based on timestamp, active status, longitude, and latitude
@app.get("/v1/talkingdata/count_app_events_by_timestamp_status_location", operation_id="get_count_app_events_by_timestamp_status_location", summary="Retrieves the total number of app events that occurred at a specific time, for a given active status, and within a certain geographical location. The time is provided in 'YYYY-MM-DD HH:MM:SS' format, the active status indicates whether the app was active or not during the event, and the location is defined by longitude and latitude coordinates.")
async def get_count_app_events_by_timestamp_status_location(timestamp: str = Query(..., description="Timestamp in 'YYYY-MM-DD HH:MM:SS' format"), is_active: str = Query(..., description="Active status"), longitude: str = Query(..., description="Longitude"), latitude: str = Query(..., description="Latitude")):
    cursor.execute("SELECT COUNT(T1.app_id) FROM app_events AS T1 INNER JOIN events AS T2 ON T1.event_id = T2.event_id WHERE T2.timestamp = ? AND T1.is_active = ? AND T2.longitude = ? AND T2.latitude = ?", (timestamp, is_active, longitude, latitude))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of events based on two timestamps and active status
@app.get("/v1/talkingdata/ratio_events_by_timestamps_status", operation_id="get_ratio_events_by_timestamps_status", summary="Retrieves the ratio of events that occurred at the specified timestamps, filtered by active status. The first timestamp is used as the numerator and the second timestamp as the denominator in the ratio calculation.")
async def get_ratio_events_by_timestamps_status(timestamp1: str = Query(..., description="First timestamp in 'YYYY-MM-DD HH:MM:SS' format"), timestamp2: str = Query(..., description="Second timestamp in 'YYYY-MM-DD HH:MM:SS' format"), is_active: str = Query(..., description="Active status")):
    cursor.execute("SELECT SUM(IIF(timestamp = ?, 1, 0)) / SUM(IIF(timestamp = ?, 1, 0)) AS num FROM events AS T1 INNER JOIN app_events AS T2 ON T1.event_id = T2.event_id WHERE T2.is_active = ?", (timestamp1, timestamp2, is_active))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the difference in count of devices between two phone brands
@app.get("/v1/talkingdata/difference_count_devices_by_phone_brands", operation_id="get_difference_count_devices_by_phone_brands", summary="Retrieves the difference in the total number of devices between two specified phone brands. The operation compares the count of devices for the first phone brand with the count for the second phone brand and returns the difference.")
async def get_difference_count_devices_by_phone_brands(phone_brand1: str = Query(..., description="First phone brand"), phone_brand2: str = Query(..., description="Second phone brand")):
    cursor.execute("SELECT SUM(IIF(phone_brand = ?, 1, 0)) - SUM(IIF(phone_brand = ?, 1, 0)) AS num FROM phone_brand_device_model2", (phone_brand1, phone_brand2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the device ID of the youngest user
@app.get("/v1/talkingdata/get_device_id_youngest_user", operation_id="get_device_id_youngest_user", summary="Retrieves the device identifier associated with the youngest user, as determined by the minimum age recorded in the gender_age table.")
async def get_device_id_youngest_user():
    cursor.execute("SELECT device_id FROM gender_age WHERE age = ( SELECT MIN(age) FROM gender_age )")
    result = cursor.fetchone()
    if not result:
        return {"device_id": []}
    return {"device_id": result[0]}

# Endpoint to get the count of devices based on age and gender
@app.get("/v1/talkingdata/count_devices_by_age_and_gender", operation_id="get_count_devices_by_age_and_gender", summary="Retrieves the total number of devices used by individuals of a specific gender who are older than a given age. The age and gender criteria are provided as input parameters.")
async def get_count_devices_by_age_and_gender(min_age: int = Query(..., description="Minimum age"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT COUNT(device_id) FROM gender_age WHERE age > ? AND gender = ?", (min_age, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the gender of the oldest user
@app.get("/v1/talkingdata/get_gender_oldest_user", operation_id="get_gender_oldest_user", summary="Retrieves the gender of the user with the highest age from the gender_age table.")
async def get_gender_oldest_user():
    cursor.execute("SELECT gender FROM gender_age WHERE age = ( SELECT MAX(age) FROM gender_age )")
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the youngest age of users with a specific phone brand
@app.get("/v1/talkingdata/get_youngest_age_by_phone_brand", operation_id="get_youngest_age_by_phone_brand", summary="Retrieves the youngest age of users associated with a specified phone brand. The operation filters users based on the provided phone brand and identifies the youngest age from the filtered set.")
async def get_youngest_age_by_phone_brand(phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT T1.age FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ? ORDER BY T1.age LIMIT 1", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get app IDs based on category
@app.get("/v1/talkingdata/app_ids_by_category", operation_id="get_app_ids_by_category", summary="Retrieves a list of application IDs that belong to a specified category. The category is used to filter the app IDs from the label_categories and app_labels tables, which are joined based on their shared label IDs.")
async def get_app_ids_by_category(category: str = Query(..., description="Category of the app")):
    cursor.execute("SELECT T2.app_id FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id WHERE T1.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"app_ids": []}
    return {"app_ids": [row[0] for row in result]}

# Endpoint to get gender based on timestamp
@app.get("/v1/talkingdata/gender_by_timestamp", operation_id="get_gender_by_timestamp", summary="Retrieves the gender distribution for a specific timestamp. The operation uses the provided timestamp to filter events and returns the corresponding gender data from the gender_age table.")
async def get_gender_by_timestamp(timestamp: str = Query(..., description="Timestamp in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT T1.gender FROM gender_age AS T1 INNER JOIN events AS T2 ON T1.device_id = T2.device_id WHERE T2.timestamp = ?", (timestamp,))
    result = cursor.fetchall()
    if not result:
        return {"genders": []}
    return {"genders": [row[0] for row in result]}

# Endpoint to get the count of device IDs based on year and group
@app.get("/v1/talkingdata/device_count_by_year_group", operation_id="get_device_count_by_year_group", summary="Retrieves the total count of unique devices that were active in a specified year and group. The year is provided in 'YYYY' format, and the group is identified by a unique group identifier. This operation does not return individual device details, but rather an aggregated count of devices that meet the specified criteria.")
async def get_device_count_by_year_group(year: str = Query(..., description="Year in 'YYYY' format"), group: str = Query(..., description="Group identifier")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN events AS T2 ON T1.device_id = T2.device_id WHERE STRFTIME('%Y', T2.timestamp) = ? AND T1.`group` = ?", (year, group))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most popular phone brand based on group
@app.get("/v1/talkingdata/most_popular_phone_brand_by_group", operation_id="get_most_popular_phone_brand_by_group", summary="Retrieves the most popular phone brand based on the provided group identifier. The operation calculates the count of device IDs associated with each phone brand within the specified group and returns the phone brand with the highest count.")
async def get_most_popular_phone_brand_by_group(group: str = Query(..., description="Group identifier")):
    cursor.execute("SELECT T.phone_brand FROM ( SELECT T2.phone_brand, COUNT(T1.device_id) AS num FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.`group` = ? GROUP BY T2.phone_brand ) AS T ORDER BY T.num DESC LIMIT 1", (group,))
    result = cursor.fetchone()
    if not result:
        return {"phone_brand": []}
    return {"phone_brand": result[0]}

# Endpoint to get longitude and latitude based on device model
@app.get("/v1/talkingdata/location_by_device_model", operation_id="get_location_by_device_model", summary="Retrieves the geographical coordinates (longitude and latitude) associated with a specific device model. This operation identifies the device model from the provided input and returns the corresponding location data from the events table.")
async def get_location_by_device_model(device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT T1.longitude, T1.latitude FROM events AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.device_model = ?", (device_model,))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": [{"longitude": row[0], "latitude": row[1]} for row in result]}

# Endpoint to get device models based on gender
@app.get("/v1/talkingdata/device_models_by_gender", operation_id="get_device_models_by_gender", summary="Retrieves a list of device models associated with a specific gender. The operation filters device models based on the provided gender parameter, returning only those that match the specified gender.")
async def get_device_models_by_gender(gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT T1.device_model FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T1.device_id = T2.device_id WHERE T2.gender = ?", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get the count of app IDs based on activity status, category, and event ID
@app.get("/v1/talkingdata/app_count_by_activity_category_event", operation_id="get_app_count_by_activity_category_event", summary="Retrieves the total number of applications that meet the specified activity status, category, and event ID criteria. The activity status indicates whether the application is active or inactive, the category refers to the type or classification of the application, and the event ID corresponds to a specific event associated with the application.")
async def get_app_count_by_activity_category_event(is_active: int = Query(..., description="Activity status (0 or 1)"), category: str = Query(..., description="Category"), event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT COUNT(T2.app_id) FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id INNER JOIN app_events AS T3 ON T2.app_id = T3.app_id WHERE T3.is_active = ? AND T1.category = ? AND T3.event_id = ?", (is_active, category, event_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct categories based on event ID and activity status
@app.get("/v1/talkingdata/distinct_category_count_by_event_activity", operation_id="get_distinct_category_count_by_event_activity", summary="Retrieves the total count of unique categories associated with a specific event and activity status. The operation groups categories by event ID and activity status, then returns the count of distinct categories within each group.")
async def get_distinct_category_count_by_event_activity(event_id: int = Query(..., description="Event ID"), is_active: int = Query(..., description="Activity status (0 or 1)")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT COUNT(DISTINCT T1.category) AS result FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id INNER JOIN app_events AS T3 ON T2.app_id = T3.app_id WHERE T3.event_id = ? AND T3.is_active = ? GROUP BY T1.category ) T", (event_id, is_active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most popular device model based on age and gender
@app.get("/v1/talkingdata/most_popular_device_model_by_age_gender", operation_id="get_most_popular_device_model_by_age_gender", summary="Retrieves the most frequently used device model among users of a specific age and gender. The age and gender are provided as input parameters to filter the data and determine the most popular device model.")
async def get_most_popular_device_model_by_age_gender(age: int = Query(..., description="Age"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT T.device_model FROM ( SELECT T2.device_model, COUNT(T2.device_model) AS num FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.age > ? AND T1.gender = ? GROUP BY T2.device_model ) AS T ORDER BY T.num DESC LIMIT 1", (age, gender))
    result = cursor.fetchone()
    if not result:
        return {"device_model": []}
    return {"device_model": result[0]}

# Endpoint to get device models based on longitude and latitude
@app.get("/v1/talkingdata/device_models_by_location", operation_id="get_device_models_by_location", summary="Retrieves the device models associated with a specific geographical location, identified by longitude and latitude. The operation returns a list of device models that have been used at the given location, based on event data.")
async def get_device_models_by_location(longitude: float = Query(..., description="Longitude of the location"), latitude: float = Query(..., description="Latitude of the location")):
    cursor.execute("SELECT T2.device_model FROM events AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.longitude = ? AND T1.latitude = ?", (longitude, latitude))
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get top categories based on app count
@app.get("/v1/talkingdata/top_categories_by_app_count", operation_id="get_top_categories_by_app_count", summary="Retrieves the top categories ranked by the number of associated apps. The operation returns a specified number of categories with the highest app count. The input parameter determines the number of top categories to include in the response.")
async def get_top_categories_by_app_count(limit: int = Query(..., description="Number of top categories to return")):
    cursor.execute("SELECT T.category FROM ( SELECT T2.category, COUNT(T1.app_id) AS num FROM app_labels AS T1 INNER JOIN label_categories AS T2 ON T1.label_id = T2.label_id GROUP BY T2.category ) AS T ORDER BY T.num DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get event IDs based on age
@app.get("/v1/talkingdata/event_ids_by_age", operation_id="get_event_ids_by_age", summary="Retrieves a specified number of event IDs, ordered by age in descending order. The operation returns event IDs from devices associated with users of varying ages, with the oldest users' devices prioritized.")
async def get_event_ids_by_age(limit: int = Query(..., description="Number of event IDs to return")):
    cursor.execute("SELECT T2.event_id FROM gender_age AS T1 INNER JOIN events AS T2 ON T1.device_id = T2.device_id ORDER BY T1.age DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"event_ids": []}
    return {"event_ids": [row[0] for row in result]}

# Endpoint to get event count based on gender and age
@app.get("/v1/talkingdata/event_count_by_gender_age", operation_id="get_event_count_by_gender_age", summary="Retrieves the count of specific events based on the gender and age of the users. The operation returns the number of events for each age group within the specified gender category, up to a defined limit.")
async def get_event_count_by_gender_age(gender: str = Query(..., description="Gender of the user"), limit: int = Query(..., description="Number of events to return")):
    cursor.execute("SELECT COUNT(T1.event_id) FROM events AS T1 INNER JOIN gender_age AS T2 ON T1.device_id = T2.device_id WHERE T2.gender = ? GROUP BY T1.event_id, T2.age ORDER BY T2.age LIMIT ?", (gender, limit))
    result = cursor.fetchall()
    if not result:
        return {"event_count": []}
    return {"event_count": [row[0] for row in result]}

# Endpoint to get device count based on timestamp and gender
@app.get("/v1/talkingdata/device_count_by_timestamp_gender", operation_id="get_device_count_by_timestamp_gender", summary="Retrieves the total number of unique devices that were active on a specific date and belong to a particular gender category. The date is provided in 'YYYY-MM-DD' format, and the gender is specified as either 'male' or 'female'.")
async def get_device_count_by_timestamp_gender(timestamp: str = Query(..., description="Timestamp in 'YYYY-MM-DD' format"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM events AS T1 INNER JOIN gender_age AS T2 ON T1.device_id = T2.device_id WHERE T1.timestamp = ? AND T2.gender = ?", (timestamp, gender))
    result = cursor.fetchone()
    if not result:
        return {"device_count": []}
    return {"device_count": result[0]}

# Endpoint to get category difference
@app.get("/v1/talkingdata/category_difference", operation_id="get_category_difference", summary="This operation compares two categories and returns the category with the higher count of associated labels. The comparison is based on the provided labels for each category. The result will be either 'Securities' or 'Finance'.")
async def get_category_difference(category1: str = Query(..., description="First category"), category2: str = Query(..., description="Second category"), label1: str = Query(..., description="Label for the first category"), label2: str = Query(..., description="Label for the second category")):
    cursor.execute("SELECT IIF(SUM(IIF(T2.category = ?, 1, 0)) - SUM(IIF(T2.category = ?, 1, 0)) > 0, ?, ?) AS diff FROM app_labels AS T1 INNER JOIN label_categories AS T2 ON T1.label_id = T2.label_id", (category1, category2, label1, label2))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get device models based on group
@app.get("/v1/talkingdata/device_models_by_group", operation_id="get_device_models_by_group", summary="Retrieves a list of device models associated with a specific user group. The group is determined by the provided input parameter, which represents the user group to filter the device models by.")
async def get_device_models_by_group(group: str = Query(..., description="Group of the user")):
    cursor.execute("SELECT T2.device_model FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.`group` = ?", (group,))
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get average age based on phone brand and gender
@app.get("/v1/talkingdata/average_age_by_phone_brand_gender", operation_id="get_average_age_by_phone_brand_gender", summary="Retrieves the average age of users categorized by a specific phone brand and gender. The operation calculates the mean age from a dataset that combines user gender and age information with phone brand details.")
async def get_average_age_by_phone_brand_gender(phone_brand: str = Query(..., description="Phone brand"), gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT AVG(T1.age) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ? AND T1.gender = ?", (phone_brand, gender))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get gender ratio based on phone brand
@app.get("/v1/talkingdata/gender_ratio_by_phone_brand", operation_id="get_gender_ratio_by_phone_brand", summary="Retrieves the ratio of the specified genders for a given phone brand. The operation calculates the proportion of the first gender to the second gender among users of the specified phone brand. This is achieved by aggregating the number of users for each gender and computing their ratio.")
async def get_gender_ratio_by_phone_brand(gender1: str = Query(..., description="First gender"), gender2: str = Query(..., description="Second gender"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT SUM(IIF(T1.gender = ?, 1, 0)) / SUM(IIF(T1.gender = ?, 1, 0)) AS per FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ?", (gender1, gender2, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"gender_ratio": []}
    return {"gender_ratio": result[0]}

# Endpoint to get category ratio
@app.get("/v1/talkingdata/category_ratio", operation_id="get_category_ratio", summary="Retrieves the ratio of the specified categories by calculating the proportion of the first category relative to the second category. This operation considers all labels associated with the categories and returns the calculated ratio.")
async def get_category_ratio(category1: str = Query(..., description="First category"), category2: str = Query(..., description="Second category")):
    cursor.execute("SELECT SUM(IIF(T2.category = ?, 1, 0)) / SUM(IIF(T2.category = ?, 1, 0)) AS per FROM app_labels AS T1 INNER JOIN label_categories AS T2 ON T1.label_id = T2.label_id", (category1, category2))
    result = cursor.fetchone()
    if not result:
        return {"category_ratio": []}
    return {"category_ratio": result[0]}

# Endpoint to get label IDs based on category
@app.get("/v1/talkingdata/label_ids_by_category", operation_id="get_label_ids_by_category", summary="Retrieves a list of label IDs associated with a specified category. The category is used to filter the results, providing a targeted set of label IDs relevant to the given category.")
async def get_label_ids_by_category(category: str = Query(..., description="Category of the label")):
    cursor.execute("SELECT label_id FROM label_categories WHERE category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"label_ids": []}
    return {"label_ids": [row[0] for row in result]}

# Endpoint to get the percentage of active events for a specific event ID
@app.get("/v1/talkingdata/active_event_percentage", operation_id="get_active_event_percentage", summary="Retrieves the ratio of active to inactive events for a specific event ID. The operation calculates the proportion of active events by comparing the count of active events to the count of inactive events for the given event ID.")
async def get_active_event_percentage(event_id: int = Query(..., description="ID of the event")):
    cursor.execute("SELECT SUM(IIF(is_active = 1, 1, 0)) / SUM(IIF(is_active = 0, 1, 0)) AS per FROM app_events WHERE event_id = ?", (event_id,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of events for a specific device ID
@app.get("/v1/talkingdata/event_count_by_device", operation_id="get_event_count_by_device", summary="Retrieves the total number of events associated with a particular device, identified by its unique device ID.")
async def get_event_count_by_device(device_id: int = Query(..., description="ID of the device")):
    cursor.execute("SELECT COUNT(event_id) FROM events WHERE device_id = ?", (device_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of devices for a specific gender
@app.get("/v1/talkingdata/gender_percentage", operation_id="get_gender_percentage", summary="Retrieves the proportion of devices associated with a specific gender. The gender is provided as an input parameter, and the operation calculates the percentage by comparing the count of devices for the given gender against the total number of devices.")
async def get_gender_percentage(gender: str = Query(..., description="Gender (e.g., 'M' for male)")):
    cursor.execute("SELECT SUM(IIF(gender = ?, 1, 0)) / COUNT(device_id) AS per FROM gender_age", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of events for a specific longitude
@app.get("/v1/talkingdata/event_count_by_longitude", operation_id="get_event_count_by_longitude", summary="Retrieves the total number of events that occurred at a specific longitude. The operation requires a longitude value as input to filter the events and calculate the count.")
async def get_event_count_by_longitude(longitude: float = Query(..., description="Longitude value")):
    cursor.execute("SELECT COUNT(event_id) FROM events WHERE longitude = ?", (longitude,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of apps for a specific label ID
@app.get("/v1/talkingdata/app_count_by_label", operation_id="get_app_count_by_label", summary="Retrieves the total number of applications associated with a particular label. The label is identified by its unique ID.")
async def get_app_count_by_label(label_id: int = Query(..., description="ID of the label")):
    cursor.execute("SELECT COUNT(app_id) FROM app_labels WHERE label_id = ?", (label_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of label IDs for categories matching a pattern
@app.get("/v1/talkingdata/label_count_by_category_pattern", operation_id="get_label_count_by_category_pattern", summary="Retrieves the total count of unique label IDs associated with categories that match the provided pattern. The pattern is used to filter the categories, allowing for a targeted count of label IDs. For example, specifying 'game%' as the pattern will return the count of label IDs for categories starting with 'game'.")
async def get_label_count_by_category_pattern(category_pattern: str = Query(..., description="Pattern to match the category (e.g., 'game%')")):
    cursor.execute("SELECT COUNT(label_id) FROM label_categories WHERE category LIKE ?", (category_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get label IDs and categories for a specific app ID
@app.get("/v1/talkingdata/labels_and_categories_by_app", operation_id="get_labels_and_categories_by_app", summary="Retrieves the label IDs and corresponding categories for a given app ID. This operation fetches data from the app_labels and label_categories tables, joining them on the label_id field. The app_id parameter is used to filter the results for a specific app.")
async def get_labels_and_categories_by_app(app_id: int = Query(..., description="ID of the app")):
    cursor.execute("SELECT T1.label_id, T2.category FROM app_labels AS T1 INNER JOIN label_categories AS T2 ON T1.label_id = T2.label_id WHERE T1.app_id = ?", (app_id,))
    result = cursor.fetchall()
    if not result:
        return {"labels_and_categories": []}
    return {"labels_and_categories": [{"label_id": row[0], "category": row[1]} for row in result]}

# Endpoint to get label IDs and app IDs based on category
@app.get("/v1/talkingdata/label_app_ids_by_category", operation_id="get_label_app_ids_by_category", summary="Retrieves the label IDs and corresponding app IDs associated with a specific category. The category parameter is used to filter the results.")
async def get_label_app_ids_by_category(category: str = Query(..., description="Category of the label")):
    cursor.execute("SELECT T1.label_id, T2.app_id FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id WHERE T1.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of app IDs grouped by longitude and latitude for a specific event ID
@app.get("/v1/talkingdata/app_count_by_location", operation_id="get_app_count_by_location", summary="Retrieves the count of unique app IDs associated with a specific event, grouped by their respective longitude and latitude coordinates. This operation provides a geographical distribution of app usage for the given event ID.")
async def get_app_count_by_location(event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT COUNT(T1.app_id), T2.longitude, T2.latitude FROM app_events AS T1 INNER JOIN events AS T2 ON T1.event_id = T2.event_id WHERE T1.event_id = ? GROUP BY T2.longitude, T2.latitude", (event_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get event details based on app ID
@app.get("/v1/talkingdata/event_details_by_app_id", operation_id="get_event_details_by_app_id", summary="Retrieves the geographical location (longitude and latitude) and timestamp of events associated with a specific application. The app_id parameter is used to filter the events.")
async def get_event_details_by_app_id(app_id: int = Query(..., description="App ID")):
    cursor.execute("SELECT T1.longitude, T1.latitude, T1.timestamp FROM events AS T1 INNER JOIN app_events AS T2 ON T1.event_id = T2.event_id WHERE T2.app_id = ?", (app_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of device IDs for a specific phone brand and date
@app.get("/v1/talkingdata/device_count_by_brand_and_date", operation_id="get_device_count_by_brand_and_date", summary="Retrieves the total number of unique devices for a given phone brand on a specific date. The input parameters include the phone brand and the date in 'YYYY-MM-DD' format.")
async def get_device_count_by_brand_and_date(phone_brand: str = Query(..., description="Phone brand"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM events AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ? AND STRFTIME('%Y-%m-%d', T1.`timestamp`) = ?", (phone_brand, date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of male to female users for a specific phone brand and model
@app.get("/v1/talkingdata/gender_ratio_by_brand_and_model", operation_id="get_gender_ratio_by_brand_and_model", summary="Retrieves the ratio of male to female users for a specified phone brand and model. This operation calculates the proportion of male users to female users based on the provided phone brand and device model. The result is a single value representing the ratio of male to female users.")
async def get_gender_ratio_by_brand_and_model(phone_brand: str = Query(..., description="Phone brand"), device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT SUM(IIF(T1.gender = 'M', 1, 0)) / SUM(IIF(T1.gender = 'F', 1, 0)) AS per FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ? AND T2.device_model = ?", (phone_brand, device_model))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of device IDs for a specific gender, device model, group, and phone brand
@app.get("/v1/talkingdata/device_count_by_gender_model_group_brand", operation_id="get_device_count_by_gender_model_group_brand", summary="Retrieve the total number of unique devices categorized by a specific gender, device model, group, and phone brand. This operation provides a comprehensive view of device distribution across these categories, enabling targeted analysis and decision-making.")
async def get_device_count_by_gender_model_group_brand(gender: str = Query(..., description="Gender"), device_model: str = Query(..., description="Device model"), group: str = Query(..., description="Group"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.gender = ? AND T2.device_model = ? AND T1.`group` = ? AND T2.phone_brand = ?", (gender, device_model, group, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get phone brands and device models for users under a specific age
@app.get("/v1/talkingdata/phone_models_by_age", operation_id="get_phone_models_by_age", summary="Retrieves the phone brands and device models associated with users under a specified age. The age parameter is used to filter the results, providing a targeted list of phone brands and models.")
async def get_phone_models_by_age(age: int = Query(..., description="Age")):
    cursor.execute("SELECT T2.phone_brand, T2.device_model FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.age < ?", (age,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of female users over a specific age for a specific phone brand
@app.get("/v1/talkingdata/female_percentage_by_age_and_brand", operation_id="get_female_percentage_by_age_and_brand", summary="Retrieves the proportion of female users who are older than the specified age for a given phone brand. The calculation is based on the aggregated data from the gender_age and phone_brand_device_model2 tables.")
async def get_female_percentage_by_age_and_brand(age: int = Query(..., description="Age"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT SUM(IIF(T1.gender = 'F' AND T1.age > ?, 1, 0)) / COUNT(T1.device_id) AS per FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ?", (age, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get phone brands and device models for a specific event ID
@app.get("/v1/talkingdata/phone_models_by_event_id", operation_id="get_phone_models_by_event_id", summary="Retrieves the phone brands and device models associated with a specific event ID. The event ID is used to look up corresponding device IDs, which are then matched with their respective phone brands and device models.")
async def get_phone_models_by_event_id(event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT T2.phone_brand, T2.device_model FROM events AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.event_id = ?", (event_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the age and gender of the most frequent device IDs within a specific range
@app.get("/v1/talkingdata/most_frequent_age_gender", operation_id="get_most_frequent_age_gender", summary="Retrieves the age and gender of the device IDs that appear most frequently within the specified range of device IDs. The range is defined by the minimum and maximum device IDs provided as input parameters.")
async def get_most_frequent_age_gender(min_device_id: int = Query(..., description="Minimum device ID"), max_device_id: int = Query(..., description="Maximum device ID")):
    cursor.execute("SELECT T.age, T.gender FROM ( SELECT T2.age, T2.gender, COUNT(T1.device_id) AS num FROM events AS T1 INNER JOIN gender_age AS T2 ON T1.device_id = T2.device_id WHERE T1.device_id BETWEEN ? AND ? GROUP BY T2.age, T2.gender ) AS T ORDER BY T.num DESC LIMIT 1", (min_device_id, max_device_id))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of active app events
@app.get("/v1/talkingdata/active_app_events_count", operation_id="get_active_app_events_count", summary="Retrieves the total count of active app events. The operation filters app events based on their active status, which is provided as an input parameter. The result is a single integer value representing the count of app events that are currently active.")
async def get_active_app_events_count(is_active: int = Query(..., description="Active status of the app event (1 for active, 0 for inactive)")):
    cursor.execute("SELECT COUNT(app_id) FROM app_events WHERE is_active = ?", (is_active,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the gender based on device ID
@app.get("/v1/talkingdata/gender_by_device_id", operation_id="get_gender_by_device_id", summary="Retrieves the gender associated with the specified device ID. The operation uses the provided device ID to look up the corresponding gender in the gender_age table.")
async def get_gender_by_device_id(device_id: int = Query(..., description="ID of the device")):
    cursor.execute("SELECT gender FROM gender_age WHERE device_id = ?", (device_id,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the count of label IDs based on category
@app.get("/v1/talkingdata/label_count_by_category", operation_id="get_label_count_by_category", summary="Retrieves the total count of unique label identifiers associated with a specific category. The category is provided as an input parameter, allowing the operation to filter and count the relevant label identifiers.")
async def get_label_count_by_category(category: str = Query(..., description="Category of the label")):
    cursor.execute("SELECT COUNT(label_id) FROM label_categories WHERE category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the group based on phone brand
@app.get("/v1/talkingdata/group_by_phone_brand", operation_id="get_group_by_phone_brand", summary="Get the group based on the phone brand")
async def get_group_by_phone_brand(phone_brand: str = Query(..., description="Brand of the phone")):
    cursor.execute("SELECT T1.`group` FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ?", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"group": []}
    return {"group": result[0]}

# Endpoint to get the count of devices based on device model and age
@app.get("/v1/talkingdata/device_count_by_model_and_age", operation_id="get_device_count_by_model_and_age", summary="Retrieves the total count of devices that match the specified model and belong to users under a certain age. The count is determined by cross-referencing user age data with device model information.")
async def get_device_count_by_model_and_age(device_model: str = Query(..., description="Model of the device"), age: int = Query(..., description="Age of the user")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.device_model = ? AND T1.age < ?", (device_model, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of users under a certain age based on phone brand
@app.get("/v1/talkingdata/percentage_users_under_age_by_phone_brand", operation_id="get_percentage_users_under_age_by_phone_brand", summary="Retrieves the proportion of users below a specified age, grouped by the brand of their phone. The calculation is based on the total number of users for each phone brand.")
async def get_percentage_users_under_age_by_phone_brand(age: int = Query(..., description="Age of the user"), phone_brand: str = Query(..., description="Brand of the phone")):
    cursor.execute("SELECT SUM(IIF(T1.age < ?, 1, 0)) / COUNT(T1.device_id) AS per FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ?", (age, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average age of users based on device model
@app.get("/v1/talkingdata/average_age_by_device_model", operation_id="get_average_age_by_device_model", summary="Retrieves the average age of users associated with a specific device model. This operation calculates the average age by summing up the ages of users who own the specified device model and then dividing by the total count of users with that device model. The device model is provided as an input parameter.")
async def get_average_age_by_device_model(device_model: str = Query(..., description="Model of the device")):
    cursor.execute("SELECT SUM(T1.age) / COUNT(T1.device_id) AS avg FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.device_model = ?", (device_model,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the count of devices based on gender and device model
@app.get("/v1/talkingdata/device_count_by_gender_and_model", operation_id="get_device_count_by_gender_and_model", summary="Retrieves the total number of devices that match the specified gender and device model. The operation filters devices based on the provided gender and device model, and returns the count of devices that meet these criteria.")
async def get_device_count_by_gender_and_model(gender: str = Query(..., description="Gender"), device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.gender = ? AND T2.device_model = ?", (gender, device_model))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of devices based on age range and device model
@app.get("/v1/talkingdata/device_count_by_age_range_and_model", operation_id="get_device_count_by_age_range_and_model", summary="Retrieves the total number of devices that fall within the specified age range and device model. The age range is defined by the minimum and maximum age values, while the device model is specified by its unique identifier. This operation provides a comprehensive count of devices that meet the given criteria.")
async def get_device_count_by_age_range_and_model(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.age BETWEEN ? AND ? AND T2.device_model = ?", (min_age, max_age, device_model))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of devices based on gender and phone brand
@app.get("/v1/talkingdata/device_count_by_gender_and_brand", operation_id="get_device_count_by_gender_and_brand", summary="Retrieves the total number of devices associated with a specific gender and phone brand. The gender and phone brand are provided as input parameters, allowing for a targeted count of devices that meet the specified criteria.")
async def get_device_count_by_gender_and_brand(gender: str = Query(..., description="Gender"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.gender = ? AND T2.phone_brand = ?", (gender, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of devices based on age range and phone brand
@app.get("/v1/talkingdata/device_count_by_age_range_and_brand", operation_id="get_device_count_by_age_range_and_brand", summary="Retrieves the total number of devices that belong to a specific phone brand and fall within a given age range. The age range is determined by the minimum and maximum age values provided. The phone brand is specified using its unique identifier.")
async def get_device_count_by_age_range_and_brand(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.age BETWEEN ? AND ? AND T2.phone_brand = ?", (min_age, max_age, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of devices with age greater than a specified value for a given phone brand
@app.get("/v1/talkingdata/percentage_devices_age_greater_than", operation_id="get_percentage_devices_age_greater_than", summary="Retrieves the proportion of devices belonging to a specific phone brand that are older than a given age. The calculation is based on the total count of devices for that brand.")
async def get_percentage_devices_age_greater_than(age: int = Query(..., description="Age"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT SUM(IIF(T1.age > ?, 1, 0)) / COUNT(T1.device_id) AS per FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ?", (age, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get longitude and latitude based on a specific date
@app.get("/v1/talkingdata/longitude_latitude_by_date", operation_id="get_longitude_latitude_by_date", summary="Retrieves the longitude and latitude coordinates for events that occurred on the specified date. The date should be provided in the 'YYYY-MM-DD' format.")
async def get_longitude_latitude_by_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT longitude, latitude FROM events WHERE date(timestamp) = ?", (date,))
    result = cursor.fetchall()
    if not result:
        return {"coordinates": []}
    return {"coordinates": [{"longitude": row[0], "latitude": row[1]} for row in result]}

# Endpoint to get the count of device models for a specific phone brand
@app.get("/v1/talkingdata/count_device_models_by_brand", operation_id="get_count_device_models_by_brand", summary="Retrieves the total number of distinct device models associated with a specified phone brand.")
async def get_count_device_models_by_brand(phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT COUNT(device_model) FROM phone_brand_device_model2 WHERE phone_brand = ?", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get app IDs based on activity and installation status
@app.get("/v1/talkingdata/get_app_ids_by_status", operation_id="get_app_ids_by_status", summary="Retrieves a list of application IDs that match the specified activity and installation status. The activity status can be set to active or inactive, while the installation status can be set to installed or not installed.")
async def get_app_ids_by_status(is_active: int = Query(..., description="Activity status (1 for active, 0 for inactive)"), is_installed: int = Query(..., description="Installation status (1 for installed, 0 for not installed)")):
    cursor.execute("SELECT app_id FROM app_events WHERE is_active = ? AND is_installed = ?", (is_active, is_installed))
    result = cursor.fetchall()
    if not result:
        return {"app_ids": []}
    return {"app_ids": [row[0] for row in result]}

# Endpoint to get device IDs based on age range and gender
@app.get("/v1/talkingdata/get_device_ids_by_age_gender", operation_id="get_device_ids_by_age_gender", summary="Retrieves a list of device IDs that belong to users within a specified age range and gender. The age range is defined by the minimum and maximum age parameters, while the gender parameter filters results by the user's gender. The endpoint returns a collection of device IDs that meet the specified criteria.")
async def get_device_ids_by_age_gender(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), gender: str = Query(..., description="Gender (M or F)")):
    cursor.execute("SELECT device_id FROM gender_age_train WHERE age BETWEEN ? AND ? AND gender = ?", (min_age, max_age, gender))
    result = cursor.fetchall()
    if not result:
        return {"device_ids": []}
    return {"device_ids": [row[0] for row in result]}

# Endpoint to get the timestamp with the highest number of events for a specific gender and age
@app.get("/v1/talkingdata/get_timestamp_max_events_by_gender_age", operation_id="get_timestamp_max_events_by_gender_age", summary="Retrieves the timestamp with the maximum number of events for a given gender and age. The operation filters events based on the specified gender and age, then identifies the timestamp with the highest event count.")
async def get_timestamp_max_events_by_gender_age(gender: str = Query(..., description="Gender (M or F)"), age: int = Query(..., description="Age")):
    cursor.execute("SELECT T.timestamp FROM ( SELECT T2.timestamp, COUNT(T2.event_id) AS num FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id WHERE T1.gender = ? AND T1.age = ? GROUP BY T2.timestamp ) AS T ORDER BY T.num DESC LIMIT 1", (gender, age))
    result = cursor.fetchone()
    if not result:
        return {"timestamp": []}
    return {"timestamp": result[0]}

# Endpoint to get the phone brand with the highest number of active events
@app.get("/v1/talkingdata/get_top_phone_brand_by_active_events", operation_id="get_top_phone_brand_by_active_events", summary="Retrieves the phone brand with the highest number of active events, based on the provided activity status. The operation filters events by their activity status and calculates the total count for each phone brand. The brand with the highest count is returned.")
async def get_top_phone_brand_by_active_events(is_active: int = Query(..., description="Activity status (1 for active, 0 for inactive)")):
    cursor.execute("SELECT T.phone_brand FROM ( SELECT T1.phone_brand, COUNT(T4.is_active) AS num FROM phone_brand_device_model2 AS T1 INNER JOIN gender_age AS T2 ON T1.device_id = T2.device_id INNER JOIN events_relevant AS T3 ON T2.device_id = T3.device_id INNER JOIN app_events_relevant AS T4 ON T3.event_id = T4.event_id WHERE T4.is_active = ? GROUP BY T1.phone_brand ) AS T ORDER BY T.num DESC LIMIT 1", (is_active,))
    result = cursor.fetchone()
    if not result:
        return {"phone_brand": []}
    return {"phone_brand": result[0]}

# Endpoint to get the count of device IDs based on gender, activity status, and age
@app.get("/v1/talkingdata/count_device_ids_by_gender_activity_age", operation_id="get_count_device_ids_by_gender_activity_age", summary="Retrieves the total count of unique device IDs that match the specified gender, activity status, and age criteria. The gender must be either 'M' or 'F', the activity status can be '1' for active or '0' for inactive, and the maximum age is used to filter devices based on the age of their users.")
async def get_count_device_ids_by_gender_activity_age(gender: str = Query(..., description="Gender (M or F)"), is_active: int = Query(..., description="Activity status (1 for active, 0 for inactive)"), max_age: int = Query(..., description="Maximum age")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id INNER JOIN app_events_relevant AS T3 ON T2.event_id = T3.event_id WHERE T1.gender = ? AND T3.is_active = ? AND T1.age < ?", (gender, is_active, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of device IDs based on age, gender, activity status, and installation status
@app.get("/v1/talkingdata/count_device_ids_by_age_gender_activity_installation", operation_id="get_count_device_ids_by_age_gender_activity_installation", summary="Retrieve the count of unique device IDs that meet the specified age, gender, activity status, and installation status criteria. The age must be less than the provided maximum age, and the gender, activity status, and installation status must match the given values. The result reflects the number of devices that satisfy all these conditions.")
async def get_count_device_ids_by_age_gender_activity_installation(max_age: int = Query(..., description="Maximum age"), gender: str = Query(..., description="Gender (M or F)"), is_active: int = Query(..., description="Activity status (1 for active, 0 for inactive)"), is_installed: int = Query(..., description="Installation status (1 for installed, 0 for not installed)")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id INNER JOIN app_events_relevant AS T3 ON T2.event_id = T3.event_id WHERE T1.age < ? AND T1.gender = ? AND T3.is_active = ? AND T3.is_installed = ?", (max_age, gender, is_active, is_installed))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the gender with the highest number of events within a date range
@app.get("/v1/talkingdata/get_top_gender_by_events_date_range", operation_id="get_top_gender_by_events_date_range", summary="Retrieves the gender with the most events within the specified date range. The date range is defined by the start_date and end_date parameters, both in 'YYYY-MM-DD' format. The operation calculates the number of events for each gender and returns the gender with the highest count.")
async def get_top_gender_by_events_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T.gender FROM ( SELECT T1.gender, COUNT(T1.device_id) AS num FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id WHERE date(T2.timestamp) BETWEEN ? AND ? GROUP BY T1.gender ) AS T ORDER BY T.num DESC LIMIT 1", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the group with the highest number of devices for a specific device model
@app.get("/v1/talkingdata/group_with_highest_device_count", operation_id="get_group_with_highest_device_count", summary="Retrieves the group with the highest number of devices for a given device model. The operation filters data based on the provided device model and calculates the number of devices per group. The group with the maximum count is then returned.")
async def get_group_with_highest_device_count(device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT T.`group` FROM ( SELECT T1.`group`, COUNT(T1.device_id) AS num FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.device_model = ? GROUP BY T1.`group` ) AS T ORDER BY T.num DESC LIMIT 1", (device_model,))
    result = cursor.fetchone()
    if not result:
        return {"group": []}
    return {"group": result[0]}

# Endpoint to get the count of devices for users older than a certain age, excluding a specific device model and phone brand
@app.get("/v1/talkingdata/device_count_age_exclude_model_brand", operation_id="get_device_count_age_exclude_model_brand", summary="Retrieves the total number of devices used by users who are older than the specified age, excluding a particular device model and phone brand. This operation filters the data based on age, device model, and phone brand, and returns the count of devices that meet the criteria.")
async def get_device_count_age_exclude_model_brand(age: int = Query(..., description="Age threshold"), device_model: str = Query(..., description="Device model to exclude"), phone_brand: str = Query(..., description="Phone brand to exclude")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.age > ? AND T2.device_model != ? AND T2.phone_brand != ?", (age, device_model, phone_brand))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get device IDs for users of a specific gender and age range
@app.get("/v1/talkingdata/device_ids_gender_age", operation_id="get_device_ids_gender_age", summary="Retrieves device IDs for users who meet the specified gender and age criteria. The operation filters users based on their gender and age, returning the device IDs of those who match the input parameters. The age parameter serves as a threshold, with the operation returning device IDs for users under the specified age.")
async def get_device_ids_gender_age(gender: str = Query(..., description="Gender of the user"), age: int = Query(..., description="Age threshold")):
    cursor.execute("SELECT T1.device_id FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id WHERE T1.gender = ? AND T1.age < ?", (gender, age))
    result = cursor.fetchall()
    if not result:
        return {"device_ids": []}
    return {"device_ids": [row[0] for row in result]}

# Endpoint to get the ratio of female to male users for events that are not installed
@app.get("/v1/talkingdata/female_to_male_ratio_not_installed", operation_id="get_female_to_male_ratio_not_installed", summary="Retrieves the ratio of female to male users for events that have not been installed. The input parameter specifies the installation status, with 0 indicating not installed.")
async def get_female_to_male_ratio_not_installed(is_installed: int = Query(..., description="Installation status (0 for not installed)")):
    cursor.execute("SELECT SUM(IIF(T1.gender = 'F', 1, 0)) / SUM(IIF(T1.gender = 'M', 1, 0)) AS per FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id INNER JOIN app_events_relevant AS T3 ON T2.event_id = T3.event_id WHERE T3.is_installed = ?", (is_installed,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the average age of users for events that are installed and not active
@app.get("/v1/talkingdata/average_age_installed_not_active", operation_id="get_average_age_installed_not_active", summary="Retrieves the average age of users who have installed the application but are currently inactive. The operation filters users based on their installation and activity status, and calculates the average age from the filtered set.")
async def get_average_age_installed_not_active(is_installed: int = Query(..., description="Installation status (1 for installed)"), is_active: int = Query(..., description="Active status (0 for not active)")):
    cursor.execute("SELECT AVG(T1.age) FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id INNER JOIN app_events_relevant AS T3 ON T2.event_id = T3.event_id WHERE T3.is_installed = ? AND T3.is_active = ?", (is_installed, is_active))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get event IDs for a specific timestamp pattern and latitude
@app.get("/v1/talkingdata/event_ids_timestamp_latitude", operation_id="get_event_ids_timestamp_latitude", summary="Retrieves up to three event IDs that match a given timestamp pattern and latitude. The timestamp pattern should be provided in 'YYYY-MM-DD%' format, and the latitude value should correspond to the desired location.")
async def get_event_ids_timestamp_latitude(timestamp: str = Query(..., description="Timestamp pattern in 'YYYY-MM-DD%' format"), latitude: float = Query(..., description="Latitude value")):
    cursor.execute("SELECT event_id FROM events WHERE timestamp LIKE ? AND latitude = ? LIMIT 3", (timestamp, latitude))
    result = cursor.fetchall()
    if not result:
        return {"event_ids": []}
    return {"event_ids": [row[0] for row in result]}

# Endpoint to get event IDs for a specific longitude and latitude
@app.get("/v1/talkingdata/event_ids_longitude_latitude", operation_id="get_event_ids_longitude_latitude", summary="Retrieves up to three event IDs associated with the specified longitude and latitude coordinates. This operation is useful for identifying events that occurred at a particular geographical location.")
async def get_event_ids_longitude_latitude(longitude: float = Query(..., description="Longitude value"), latitude: float = Query(..., description="Latitude value")):
    cursor.execute("SELECT event_id FROM events WHERE longitude = ? AND latitude = ? LIMIT 3", (longitude, latitude))
    result = cursor.fetchall()
    if not result:
        return {"event_ids": []}
    return {"event_ids": [row[0] for row in result]}

# Endpoint to get the difference in counts of events with valid and invalid coordinates for a specific device
@app.get("/v1/talkingdata/diff_valid_invalid_coordinates", operation_id="get_diff_valid_invalid_coordinates", summary="Retrieve the difference in the number of events with valid and invalid coordinates for a specific device. The device is identified by its unique ID. This operation returns a single value representing the difference between the count of events with valid coordinates and the count of events with invalid coordinates.")
async def get_diff_valid_invalid_coordinates(device_id: str = Query(..., description="Device ID")):
    cursor.execute("SELECT SUM(IIF(latitude != 0 AND longitude != 0, 1, 0)) - SUM(IIF(latitude = 0 AND longitude = 0, 1, 0)) AS diff FROM events WHERE device_id = ?", (device_id,))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get device IDs for users of a specific gender
@app.get("/v1/talkingdata/device_ids_gender", operation_id="get_device_ids_gender", summary="Retrieves a limited set of device IDs associated with users of a specified gender. The gender parameter is used to filter the results.")
async def get_device_ids_gender(gender: str = Query(..., description="Gender of the user")):
    cursor.execute("SELECT device_id FROM gender_age WHERE gender = ? LIMIT 3", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"device_ids": []}
    return {"device_ids": [row[0] for row in result]}

# Endpoint to get categories and label IDs for a specific category pattern
@app.get("/v1/talkingdata/categories_label_ids_pattern", operation_id="get_categories_label_ids_pattern", summary="Retrieves up to five categories and their corresponding label IDs that match the provided category pattern. The category pattern should be provided in the '%game%' format.")
async def get_categories_label_ids_pattern(category_pattern: str = Query(..., description="Category pattern in '%game%' format")):
    cursor.execute("SELECT category, label_id FROM label_categories WHERE category LIKE ? LIMIT 5", (category_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"categories_label_ids": []}
    return {"categories_label_ids": [{"category": row[0], "label_id": row[1]} for row in result]}

# Endpoint to get device models for a specific phone brand
@app.get("/v1/talkingdata/device_models_by_brand", operation_id="get_device_models_by_brand", summary="Retrieves a list of up to three device models associated with the specified phone brand.")
async def get_device_models_by_brand(phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT device_model FROM phone_brand_device_model2 WHERE phone_brand = ? LIMIT 3", (phone_brand,))
    result = cursor.fetchall()
    if not result:
        return {"device_models": []}
    return {"device_models": [row[0] for row in result]}

# Endpoint to get the percentage of installed but inactive apps for a specific event
@app.get("/v1/talkingdata/percentage_installed_inactive_apps", operation_id="get_percentage_installed_inactive_apps", summary="Retrieves the proportion of apps that are installed but not active for a given event. The calculation is based on the total number of apps associated with the event.")
async def get_percentage_installed_inactive_apps(event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT SUM(IIF(is_installed = 1 AND is_active = 0, 1, 0)) / COUNT(app_id) AS perrcent FROM app_events WHERE event_id = ?", (event_id,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the predominant gender for a specific device model
@app.get("/v1/talkingdata/predominant_gender_by_device_model", operation_id="get_predominant_gender_by_device_model", summary="Retrieves the predominant gender associated with a specific device model. The operation compares the count of male and female users for the given device model and returns the gender with the higher count. The input parameter specifies the device model for which the predominant gender is determined.")
async def get_predominant_gender_by_device_model(device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT IIF(SUM(IIF(T1.gender = 'M', 1, 0)) - SUM(IIF(T1.gender = 'F', 1, 0)) > 0, 'M', 'F') AS gender FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.device_model = ?", (device_model,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the count of active apps for a specific event grouped by timestamp
@app.get("/v1/talkingdata/count_active_apps_by_event", operation_id="get_count_active_apps_by_event", summary="Retrieves the number of active applications associated with a specific event, grouped by timestamp. The operation considers the provided event ID and active status to filter the results.")
async def get_count_active_apps_by_event(event_id: int = Query(..., description="Event ID"), is_active: int = Query(..., description="Is active (1 for active, 0 for inactive)")):
    cursor.execute("SELECT COUNT(T1.app_id) AS num FROM app_events AS T1 INNER JOIN events AS T2 ON T1.event_id = T2.event_id WHERE T1.event_id = ? AND T1.is_active = ? GROUP BY T2.timestamp", (event_id, is_active))
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": [row[0] for row in result]}

# Endpoint to get age and gender for a specific device and event
@app.get("/v1/talkingdata/age_gender_by_device_event", operation_id="get_age_gender_by_device_event", summary="Retrieves the age and gender associated with a specific device and event. The operation requires the device ID and event ID as input parameters to filter the relevant data from the gender_age and events_relevant tables.")
async def get_age_gender_by_device_event(device_id: int = Query(..., description="Device ID"), event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT T1.age, T1.gender FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id WHERE T1.device_id = ? AND T2.event_id = ?", (device_id, event_id))
    result = cursor.fetchall()
    if not result:
        return {"age_gender": []}
    return {"age_gender": [{"age": row[0], "gender": row[1]} for row in result]}

# Endpoint to get the count of devices for a specific longitude and gender
@app.get("/v1/talkingdata/count_devices_by_longitude_gender", operation_id="get_count_devices_by_longitude_gender", summary="Retrieves the total number of devices that match a specific longitude and gender. The longitude and gender are used to filter the data, providing a count of devices that meet the specified criteria.")
async def get_count_devices_by_longitude_gender(longitude: float = Query(..., description="Longitude"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT COUNT(T1.device_id) FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id WHERE T2.longitude = ? AND T1.gender = ?", (longitude, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get timestamps for a specific app and event
@app.get("/v1/talkingdata/timestamps_by_app_event", operation_id="get_timestamps_by_app_event", summary="Retrieves the timestamps associated with a specific app and event. The operation requires the unique identifiers for the app and event as input parameters. The app ID is used to filter the relevant app events, and the event ID is used to match the corresponding timestamps.")
async def get_timestamps_by_app_event(app_id: int = Query(..., description="App ID"), event_id: int = Query(..., description="Event ID")):
    cursor.execute("SELECT T1.timestamp FROM events_relevant AS T1 INNER JOIN app_events AS T2 ON T1.event_id = T2.event_id WHERE T2.app_id = ? AND T1.event_id = ?", (app_id, event_id))
    result = cursor.fetchall()
    if not result:
        return {"timestamps": []}
    return {"timestamps": [row[0] for row in result]}

# Endpoint to get categories for a specific app
@app.get("/v1/talkingdata/categories_by_app", operation_id="get_categories_by_app", summary="Retrieves the categories associated with a specific application. The operation uses the provided app_id to look up the corresponding application and returns the categories linked to it.")
async def get_categories_by_app(app_id: int = Query(..., description="App ID")):
    cursor.execute("SELECT T3.category FROM app_all AS T1 INNER JOIN app_labels AS T2 ON T1.app_id = T2.app_id INNER JOIN label_categories AS T3 ON T2.label_id = T3.label_id WHERE T1.app_id = ?", (app_id,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get groups for a specific phone brand and device model
@app.get("/v1/talkingdata/groups_by_phone_brand_device_model", operation_id="get_groups_by_phone_brand_device_model", summary="Retrieves the groups associated with a specific phone brand and device model. The operation filters the data based on the provided phone brand and device model, and returns the corresponding groups.")
async def get_groups_by_phone_brand_device_model(phone_brand: str = Query(..., description="Phone brand"), device_model: str = Query(..., description="Device model")):
    cursor.execute("SELECT T1.`group` FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ? AND T2.device_model = ?", (phone_brand, device_model))
    result = cursor.fetchall()
    if not result:
        return {"groups": []}
    return {"groups": [row[0] for row in result]}

# Endpoint to get gender based on device model and phone brand
@app.get("/v1/talkingdata/gender_by_device_model_and_brand", operation_id="get_gender_by_device_model_and_brand", summary="Retrieves the gender distribution associated with a specific device model and phone brand. The operation requires the device model and phone brand as input parameters to filter the data and return the corresponding gender information.")
async def get_gender_by_device_model_and_brand(device_model: str = Query(..., description="Device model"), phone_brand: str = Query(..., description="Phone brand")):
    cursor.execute("SELECT T1.gender FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.device_model = ? AND T2.phone_brand = ?", (device_model, phone_brand))
    result = cursor.fetchall()
    if not result:
        return {"gender": []}
    return {"gender": [row[0] for row in result]}

# Endpoint to get the percentage of apps in a specific category
@app.get("/v1/talkingdata/percentage_apps_in_category", operation_id="get_percentage_apps_in_category", summary="Retrieves the proportion of apps that belong to a specified category. This operation calculates the ratio of apps in the given category to the total number of apps. The category is provided as an input parameter.")
async def get_percentage_apps_in_category(category: str = Query(..., description="Category of the app")):
    cursor.execute("SELECT SUM(IIF(T1.category = ?, 1.0, 0)) / COUNT(T2.app_id) AS per FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id", (category,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of devices in a specific group on a given date
@app.get("/v1/talkingdata/percentage_devices_in_group_on_date", operation_id="get_percentage_devices_in_group_on_date", summary="Retrieves the proportion of devices belonging to a specified group that were active on a given date. The calculation is based on the total number of devices in the group and the number of devices from that group that generated events on the provided date.")
async def get_percentage_devices_in_group_on_date(group: str = Query(..., description="Group of devices"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(IIF(T1.`group` = ?, 1, 0)) / COUNT(T1.device_id) AS per FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id WHERE SUBSTR(T2.timestamp, 1, 10) = ?", (group, date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of each group in gender_age
@app.get("/v1/talkingdata/count_each_group", operation_id="get_count_each_group", summary="Retrieves the total count of records for each distinct group in the gender_age dataset. The response provides a breakdown of the data distribution across different groups.")
async def get_count_each_group():
    cursor.execute("SELECT T.num FROM ( SELECT `group`, COUNT(`group`) AS num FROM gender_age GROUP BY `group` ) T")
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": [row[0] for row in result]}

# Endpoint to get longitude and latitude based on group and gender
@app.get("/v1/talkingdata/longitude_latitude_by_group_gender", operation_id="get_longitude_latitude_by_group_gender", summary="Retrieves the longitude and latitude coordinates of devices belonging to a specific group and gender. The group and gender parameters are used to filter the devices and return their corresponding geographical locations.")
async def get_longitude_latitude_by_group_gender(group: str = Query(..., description="Group of devices"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT T2.longitude, T2.latitude FROM gender_age AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id WHERE T1.`group` = ? AND T1.gender = ?", (group, gender))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": [{"longitude": row[0], "latitude": row[1]} for row in result]}

# Endpoint to get phone brand and device model based on timestamp
@app.get("/v1/talkingdata/phone_brand_device_model_by_timestamp", operation_id="get_phone_brand_device_model_by_timestamp", summary="Retrieves the phone brand and device model associated with a specific timestamp. The operation uses the provided timestamp to filter relevant events and returns the corresponding phone brand and device model details.")
async def get_phone_brand_device_model_by_timestamp(timestamp: str = Query(..., description="Timestamp in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT T1.phone_brand, T1.device_model FROM phone_brand_device_model2 AS T1 INNER JOIN events_relevant AS T2 ON T1.device_id = T2.device_id WHERE T2.timestamp = ?", (timestamp,))
    result = cursor.fetchall()
    if not result:
        return {"devices": []}
    return {"devices": [{"phone_brand": row[0], "device_model": row[1]} for row in result]}

# Endpoint to get the most popular phone brand among users within a specified age range
@app.get("/v1/talkingdata/most_popular_phone_brand_by_age_range", operation_id="get_most_popular_phone_brand", summary="Retrieves the most popular phone brand among users within a specified age range. The operation calculates the count of each phone brand used by users within the given age range and returns the brand with the highest count. The age range is defined by the minimum and maximum age parameters.")
async def get_most_popular_phone_brand(min_age: int = Query(..., description="Minimum age of the user"), max_age: int = Query(..., description="Maximum age of the user")):
    cursor.execute("SELECT T.phone_brand FROM ( SELECT T2.phone_brand, COUNT(T2.phone_brand) AS num FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T1.age BETWEEN ? AND ? GROUP BY T2.phone_brand ) AS T ORDER BY T.num DESC LIMIT 1", (min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"phone_brand": []}
    return {"phone_brand": result[0]}

# Endpoint to get the percentage of users with missing gender, age, and group information for a specific phone brand
@app.get("/v1/talkingdata/percentage_missing_info_by_phone_brand", operation_id="get_percentage_missing_info", summary="Retrieves the proportion of users with missing gender, age, and group information for a specified phone brand. The calculation is based on the total number of users associated with the given phone brand.")
async def get_percentage_missing_info(phone_brand: str = Query(..., description="Phone brand to filter users")):
    cursor.execute("SELECT SUM(IIF(T1.gender IS NULL AND T1.age IS NULL AND T1.`group` IS NULL, 1, 0)) / COUNT(T1.device_id) AS per FROM gender_age AS T1 INNER JOIN phone_brand_device_model2 AS T2 ON T1.device_id = T2.device_id WHERE T2.phone_brand = ?", (phone_brand,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/talkingdata/oldest_individual_device_id",
    "/v1/talkingdata/event_count_by_location?latitude=40&longitude=97",
    "/v1/talkingdata/gender_age_group_count?gender=M&group=M32-38",
    "/v1/talkingdata/gender_age_count?age=50&gender=F",
    "/v1/talkingdata/active_event_count?event_id=2&is_active=1",
    "/v1/talkingdata/youngest_individual_gender",
    "/v1/talkingdata/most_common_app_category",
    "/v1/talkingdata/oldest_individual_device_model",
    "/v1/talkingdata/app_count_by_category?category=Home%20Decoration",
    "/v1/talkingdata/active_event_count_by_gender_date?is_active=1&gender=M&timestamp=2016-05-01%",
    "/v1/talkingdata/count_gender_device_model?gender=F&device_model=ZenFone%205",
    "/v1/talkingdata/oldest_user_age_by_event_criteria?is_active=1&longitude=121&latitude=31&timestamp=2016-05-06",
    "/v1/talkingdata/latest_device_model_by_group_gender?group=F27-28&gender=F",
    "/v1/talkingdata/event_categories?limit=2",
    "/v1/talkingdata/most_common_gender_by_phone_brand?phone_brand=vivo",
    "/v1/talkingdata/count_apps_by_category?category=MOBA",
    "/v1/talkingdata/gender_percentage_by_phone_brand?phone_brand=OPPO",
    "/v1/talkingdata/event_locations_by_date?timestamp=2016-05-08",
    "/v1/talkingdata/app_installation_status?event_id=844",
    "/v1/talkingdata/count_events_by_date?timestamp=2016-04-30",
    "/v1/talkingdata/device_count_by_model_and_brand?device_model=Xplay3S&phone_brand=vivo",
    "/v1/talkingdata/gender_ratio_by_group?gender_m=M&group_m=M27-28&gender_f=F&group_f=F27-28",
    "/v1/talkingdata/label_ids_by_categories?category1=online%20shopping&category2=online%20malls",
    "/v1/talkingdata/distinct_phone_brands_and_models?timestamp=2016-05-05%25&longitude=112&latitude=44",
    "/v1/talkingdata/app_ids_and_timestamps_by_event_id?event_id=82",
    "/v1/talkingdata/gender_and_age_by_event_id?event_id=15251",
    "/v1/talkingdata/event_count_by_gender_date_age?gender=M&date=2016-05-04&age=88",
    "/v1/talkingdata/distinct_age_gender_event_count?longitude=-102&latitude=38",
    "/v1/talkingdata/distinct_phone_brands_and_models_by_location?longitude=80&latitude=44",
    "/v1/talkingdata/distinct_categories_by_event_id?event_id=155",
    "/v1/talkingdata/device_ids_by_model_gender_brand?device_model=Butterfly&gender=F&phone_brand=HTC",
    "/v1/talkingdata/age_gender_by_brand_model?phone_brand=LG&device_model=L70",
    "/v1/talkingdata/percentage_apps_by_category?category=Industry%20tag",
    "/v1/talkingdata/percentage_ratio_device_model_gender_brand?device_model=Nexus%205&phone_brand=LG",
    "/v1/talkingdata/count_events_by_id_active?event_id=2&is_active=0",
    "/v1/talkingdata/count_events_by_year?year=2016",
    "/v1/talkingdata/count_events_by_year_device?year=2016&device_id=29182687948017100",
    "/v1/talkingdata/count_devices_by_gender?gender=M",
    "/v1/talkingdata/max_age",
    "/v1/talkingdata/device_count_gender_model?gender=F&device_model=Galaxy%20Note%202",
    "/v1/talkingdata/user_ages_device_model?device_model=Galaxy%20Note%202",
    "/v1/talkingdata/device_models_oldest_users",
    "/v1/talkingdata/most_common_group_phone_brand?phone_brand=vivo",
    "/v1/talkingdata/app_categories?app_id=1977658975649780000",
    "/v1/talkingdata/distinct_app_categories_event_active?event_id=2&is_active=0",
    "/v1/talkingdata/distinct_event_locations?event_id=2&is_active=0",
    "/v1/talkingdata/earliest_event_timestamp?is_active=0&event_id=2",
    "/v1/talkingdata/event_ids_by_phone_brand?phone_brand=vivo",
    "/v1/talkingdata/device_count_by_phone_brand_event_id?phone_brand=vivo&event_id=2",
    "/v1/talkingdata/timestamps_by_phone_brand_event_id?phone_brand=vivo&event_id=2",
    "/v1/talkingdata/event_count_by_year_phone_brand?year=2016&phone_brand=vivo",
    "/v1/talkingdata/device_count_by_gender_phone_brand_age?gender=F&phone_brand=vivo&age=30",
    "/v1/talkingdata/top_category_by_app_count",
    "/v1/talkingdata/phone_brands_by_youngest_device_id_gender?gender=F",
    "/v1/talkingdata/device_count_by_group_phone_brand?group=M23-26&phone_brand=vivo",
    "/v1/talkingdata/percentage_devices_in_group_by_phone_brand?group=M23-26&phone_brand=vivo",
    "/v1/talkingdata/percentage_devices_by_phone_brand_event_id?phone_brand=vivo&event_id=2",
    "/v1/talkingdata/average_age_by_phone_brand?phone_brand=vivo",
    "/v1/talkingdata/count_device_ids_by_group_and_gender?group=F27-28&gender=F",
    "/v1/talkingdata/max_age_by_gender?gender=M",
    "/v1/talkingdata/count_app_ids_by_status?is_installed=1&is_active=0",
    "/v1/talkingdata/min_age_by_gender?gender=F",
    "/v1/talkingdata/count_device_ids_by_phone_brand?phone_brand=vivo",
    "/v1/talkingdata/device_models_by_phone_brand?phone_brand=OPPO&limit=15",
    "/v1/talkingdata/device_models_by_group_and_gender?group=M39+&gender=M&limit=10",
    "/v1/talkingdata/device_models_by_event_status?is_active=1&is_installed=1&limit=5",
    "/v1/talkingdata/count_app_ids_by_label_category?category=Financial%20Information",
    "/v1/talkingdata/device_count_by_brand_gender?phone_brand=OPPO&gender=M",
    "/v1/talkingdata/device_models_youngest_age",
    "/v1/talkingdata/categories_ordered_by_label_id?limit=3",
    "/v1/talkingdata/device_count_by_model_gender?device_model=Galaxy%20Ace%20Plus&gender=M",
    "/v1/talkingdata/most_common_group_by_brand?phone_brand=OPPO",
    "/v1/talkingdata/most_common_device_models_by_gender?gender=F&limit=5",
    "/v1/talkingdata/proportion_of_categories?category1=80s%20Japanese%20comic&category2=90s%20Japanese%20comic",
    "/v1/talkingdata/proportion_of_phone_brand?phone_brand=OPPO&device_id=R815T&gender=F",
    "/v1/talkingdata/phone_brand_by_device_model?device_model=%E5%9D%9A%E6%9E%9C%E6%89%8B%E6%9C%BA",
    "/v1/talkingdata/device_count_by_model?device_model=%E4%B8%AD%E5%85%B4",
    "/v1/talkingdata/group_by_age?age=24",
    "/v1/talkingdata/timestamp_by_event_id?event_id=887711",
    "/v1/talkingdata/device_model_by_location_timestamp?longitude=113&latitude=28&timestamp=2016-05-07%2023:55:16",
    "/v1/talkingdata/app_id_count_by_category?category=game-Fishing",
    "/v1/talkingdata/device_id_count_by_group_model?group=F29-32&device_model=%E9%AD%85%E8%93%9DNote%202",
    "/v1/talkingdata/device_id_count_by_date_active_age?date=2016-05-02&is_active=1&age=30",
    "/v1/talkingdata/event_id_count_by_date_installed?date=2016-05-06&is_installed=1",
    "/v1/talkingdata/device_id_count_by_gender_brand?gender=F&phone_brand=%E9%A2%86",
    "/v1/talkingdata/count_devices_by_model_and_gender?device_model=Galaxy%20S5&gender=M",
    "/v1/talkingdata/count_app_events_by_date_status_group?date=2016-05-07&is_active=1&group=F29-32",
    "/v1/talkingdata/count_app_events_by_timestamp_status_location?timestamp=2016-05-06%2014:09:49&is_active=1&longitude=116&latitude=40",
    "/v1/talkingdata/ratio_events_by_timestamps_status?timestamp1=2016-05-02%2007:50:28&timestamp2=2016-05-02%2007:41:03&is_active=1",
    "/v1/talkingdata/difference_count_devices_by_phone_brands?phone_brand1=vivo&phone_brand2=LG",
    "/v1/talkingdata/get_device_id_youngest_user",
    "/v1/talkingdata/count_devices_by_age_and_gender?min_age=30&gender=F",
    "/v1/talkingdata/get_gender_oldest_user",
    "/v1/talkingdata/get_youngest_age_by_phone_brand?phone_brand=vivo",
    "/v1/talkingdata/app_ids_by_category?category=Securities",
    "/v1/talkingdata/gender_by_timestamp?timestamp=2016-05-01%2000:55:25",
    "/v1/talkingdata/device_count_by_year_group?year=2016&group=M23-26",
    "/v1/talkingdata/most_popular_phone_brand_by_group?group=M23-26",
    "/v1/talkingdata/location_by_device_model?device_model=Galaxy%20Note%202",
    "/v1/talkingdata/device_models_by_gender?gender=F",
    "/v1/talkingdata/app_count_by_activity_category_event?is_active=0&category=Property%20Industry%201.0&event_id=2",
    "/v1/talkingdata/distinct_category_count_by_event_activity?event_id=2&is_active=0",
    "/v1/talkingdata/most_popular_device_model_by_age_gender?age=30&gender=F",
    "/v1/talkingdata/device_models_by_location?longitude=121&latitude=31",
    "/v1/talkingdata/top_categories_by_app_count?limit=3",
    "/v1/talkingdata/event_ids_by_age?limit=1",
    "/v1/talkingdata/event_count_by_gender_age?gender=F&limit=1",
    "/v1/talkingdata/device_count_by_timestamp_gender?timestamp=2016-05-01&gender=M",
    "/v1/talkingdata/category_difference?category1=Securities&category2=Finance&label1=Securities&label2=Finance",
    "/v1/talkingdata/device_models_by_group?group=M23-26",
    "/v1/talkingdata/average_age_by_phone_brand_gender?phone_brand=vivo&gender=F",
    "/v1/talkingdata/gender_ratio_by_phone_brand?gender1=M&gender2=F&phone_brand=vivo",
    "/v1/talkingdata/category_ratio?category1=Securities&category2=Finance",
    "/v1/talkingdata/label_ids_by_category?category=Third-party%20card%20management",
    "/v1/talkingdata/active_event_percentage?event_id=58",
    "/v1/talkingdata/event_count_by_device?device_id=3915082290673130000",
    "/v1/talkingdata/gender_percentage?gender=M",
    "/v1/talkingdata/event_count_by_longitude?longitude=-156",
    "/v1/talkingdata/app_count_by_label?label_id=48",
    "/v1/talkingdata/label_count_by_category_pattern?category_pattern=game%25",
    "/v1/talkingdata/labels_and_categories_by_app?app_id=5758400314709850000",
    "/v1/talkingdata/label_app_ids_by_category?category=Chinese%20Classical%20Mythology",
    "/v1/talkingdata/app_count_by_location?event_id=79641",
    "/v1/talkingdata/event_details_by_app_id?app_id=8715964299802120000",
    "/v1/talkingdata/device_count_by_brand_and_date?phone_brand=OPPO&date=2016-05-01",
    "/v1/talkingdata/gender_ratio_by_brand_and_model?phone_brand=vivo&device_model=X5Pro",
    "/v1/talkingdata/device_count_by_gender_model_group_brand?gender=F&device_model=Z1&group=F23-&phone_brand=ZUK",
    "/v1/talkingdata/phone_models_by_age?age=10",
    "/v1/talkingdata/female_percentage_by_age_and_brand?age=80&phone_brand=HTC",
    "/v1/talkingdata/phone_models_by_event_id?event_id=6701",
    "/v1/talkingdata/most_frequent_age_gender?min_device_id=-9215352913819630000&max_device_id=-9222956879900150000",
    "/v1/talkingdata/active_app_events_count?is_active=1",
    "/v1/talkingdata/gender_by_device_id?device_id=-9222956879900150000",
    "/v1/talkingdata/label_count_by_category?category=game-card",
    "/v1/talkingdata/group_by_phone_brand?phone_brand=vivo",
    "/v1/talkingdata/device_count_by_model_and_age?device_model=Galaxy%20Note%202&age=30",
    "/v1/talkingdata/percentage_users_under_age_by_phone_brand?age=50&phone_brand=OPPO",
    "/v1/talkingdata/average_age_by_device_model?device_model=R7",
    "/v1/talkingdata/device_count_by_gender_and_model?gender=F&device_model=MI%203",
    "/v1/talkingdata/device_count_by_age_range_and_model?min_age=20&max_age=50&device_model=Galaxy%20Premier",
    "/v1/talkingdata/device_count_by_gender_and_brand?gender=M&phone_brand=HTC",
    "/v1/talkingdata/device_count_by_age_range_and_brand?min_age=20&max_age=60&phone_brand=TCL",
    "/v1/talkingdata/percentage_devices_age_greater_than?age=20&phone_brand=SUGAR",
    "/v1/talkingdata/longitude_latitude_by_date?date=2016-04-30",
    "/v1/talkingdata/count_device_models_by_brand?phone_brand=HTC",
    "/v1/talkingdata/get_app_ids_by_status?is_active=1&is_installed=1",
    "/v1/talkingdata/get_device_ids_by_age_gender?min_age=29&max_age=31&gender=F",
    "/v1/talkingdata/get_timestamp_max_events_by_gender_age?gender=M&age=40",
    "/v1/talkingdata/get_top_phone_brand_by_active_events?is_active=1",
    "/v1/talkingdata/count_device_ids_by_gender_activity_age?gender=M&is_active=0&max_age=23",
    "/v1/talkingdata/count_device_ids_by_age_gender_activity_installation?max_age=23&gender=F&is_active=0&is_installed=1",
    "/v1/talkingdata/get_top_gender_by_events_date_range?start_date=2016-05-01&end_date=2016-05-10",
    "/v1/talkingdata/group_with_highest_device_count?device_model=SM-T2558",
    "/v1/talkingdata/device_count_age_exclude_model_brand?age=50&device_model=One%20M8%20Eye&phone_brand=HTC",
    "/v1/talkingdata/device_ids_gender_age?gender=F&age=30",
    "/v1/talkingdata/female_to_male_ratio_not_installed?is_installed=0",
    "/v1/talkingdata/average_age_installed_not_active?is_installed=1&is_active=0",
    "/v1/talkingdata/event_ids_timestamp_latitude?timestamp=2016-05-01%25&latitude=31",
    "/v1/talkingdata/event_ids_longitude_latitude?longitude=0&latitude=0",
    "/v1/talkingdata/diff_valid_invalid_coordinates?device_id=-922956879900150000",
    "/v1/talkingdata/device_ids_gender?gender=F",
    "/v1/talkingdata/categories_label_ids_pattern?category_pattern=%25game%25",
    "/v1/talkingdata/device_models_by_brand?phone_brand=OPPO",
    "/v1/talkingdata/percentage_installed_inactive_apps?event_id=6",
    "/v1/talkingdata/predominant_gender_by_device_model?device_model=Galaxy%20Note%202",
    "/v1/talkingdata/count_active_apps_by_event?event_id=2&is_active=1",
    "/v1/talkingdata/age_gender_by_device_event?device_id=29182687948017100&event_id=1",
    "/v1/talkingdata/count_devices_by_longitude_gender?longitude=114&gender=M",
    "/v1/talkingdata/timestamps_by_app_event?app_id=-8022267440849930000&event_id=7",
    "/v1/talkingdata/categories_by_app?app_id=-9222198347540750000",
    "/v1/talkingdata/groups_by_phone_brand_device_model?phone_brand=LG&device_model=Nexus%204",
    "/v1/talkingdata/gender_by_device_model_and_brand?device_model=Desire%20826&phone_brand=HTC",
    "/v1/talkingdata/percentage_apps_in_category?category=Academic%20Information",
    "/v1/talkingdata/percentage_devices_in_group_on_date?group=F27-28&date=2016-05-03",
    "/v1/talkingdata/count_each_group",
    "/v1/talkingdata/longitude_latitude_by_group_gender?group=F24-26&gender=F",
    "/v1/talkingdata/phone_brand_device_model_by_timestamp?timestamp=2016-05-07%2006:03:22",
    "/v1/talkingdata/most_popular_phone_brand_by_age_range?min_age=20&max_age=30",
    "/v1/talkingdata/percentage_missing_info_by_phone_brand?phone_brand=vivo"
]
