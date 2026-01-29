from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/superhero/superhero.sqlite')
cursor = conn.cursor()

# Endpoint to get the power names of a specific superhero
@app.get("/v1/superhero/power_names_by_superhero", operation_id="get_power_names_by_superhero", summary="Retrieve the names of powers associated with a specific superhero. The operation requires the superhero's name as input and returns a list of corresponding power names.")
async def get_power_names_by_superhero(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T3.power_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchall()
    if not result:
        return {"power_names": []}
    return {"power_names": [row[0] for row in result]}

# Endpoint to get the count of heroes with a specific power
@app.get("/v1/superhero/count_heroes_by_power", operation_id="get_count_heroes_by_power", summary="Retrieves the total number of heroes who possess a specific superpower. The power is identified by its name, which is provided as an input parameter.")
async def get_count_heroes_by_power(power_name: str = Query(..., description="Name of the power")):
    cursor.execute("SELECT COUNT(T1.hero_id) FROM hero_power AS T1 INNER JOIN superpower AS T2 ON T1.power_id = T2.id WHERE T2.power_name = ?", (power_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of superheroes with a specific power and height greater than a specified value
@app.get("/v1/superhero/count_superheroes_by_power_and_height", operation_id="get_count_superheroes_by_power_and_height", summary="Retrieves the number of superheroes who possess a specific power and are taller than a given height. The power is identified by its name, and the height is provided in centimeters.")
async def get_count_superheroes_by_power_and_height(power_name: str = Query(..., description="Name of the power"), height_cm: int = Query(..., description="Height in centimeters")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T3.power_name = ? AND T1.height_cm > ?", (power_name, height_cm))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the full names of superheroes with more than a specified number of powers
@app.get("/v1/superhero/full_names_by_power_count", operation_id="get_full_names_by_power_count", summary="Retrieve the full names of superheroes who possess more than a specified number of powers. This operation returns a unique list of superhero names that meet the given power count threshold.")
async def get_full_names_by_power_count(power_count: int = Query(..., description="Number of powers")):
    cursor.execute("SELECT DISTINCT T1.full_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id GROUP BY T1.full_name HAVING COUNT(T2.power_id) > ?", (power_count,))
    result = cursor.fetchall()
    if not result:
        return {"full_names": []}
    return {"full_names": [row[0] for row in result]}

# Endpoint to get the count of superheroes with a specific eye colour
@app.get("/v1/superhero/count_superheroes_by_eye_colour", operation_id="get_count_superheroes_by_eye_colour", summary="Retrieves the total number of superheroes with a specified eye colour. The operation filters superheroes based on the provided eye colour and returns the count.")
async def get_count_superheroes_by_eye_colour(eye_colour: str = Query(..., description="Eye colour")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T2.colour = ?", (eye_colour,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the skin colour of a specific superhero
@app.get("/v1/superhero/skin_colour_by_superhero", operation_id="get_skin_colour_by_superhero", summary="Retrieves the skin colour of a specified superhero. The operation requires the superhero's name as input and returns the corresponding skin colour from the database.")
async def get_skin_colour_by_superhero(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T2.colour FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.skin_colour_id = T2.id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchone()
    if not result:
        return {"skin_colour": []}
    return {"skin_colour": result[0]}

# Endpoint to get the count of superheroes with a specific power and eye colour
@app.get("/v1/superhero/count_superheroes_by_power_and_eye_colour", operation_id="get_count_superheroes_by_power_and_eye_colour", summary="Retrieve the number of superheroes who possess a specific power and have a certain eye colour. The operation requires the name of the power and the eye colour as input parameters.")
async def get_count_superheroes_by_power_and_eye_colour(power_name: str = Query(..., description="Name of the power"), eye_colour: str = Query(..., description="Eye colour")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id INNER JOIN colour AS T4 ON T1.eye_colour_id = T4.id WHERE T3.power_name = ? AND T4.colour = ?", (power_name, eye_colour))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of superheroes with specific eye and hair colours
@app.get("/v1/superhero/superhero_names_by_eye_and_hair_colour", operation_id="get_superhero_names_by_eye_and_hair_colour", summary="Retrieves the names of superheroes who share the specified eye and hair colours. The operation filters superheroes based on their eye and hair colours, returning a list of matching names.")
async def get_superhero_names_by_eye_and_hair_colour(eye_colour: str = Query(..., description="Eye colour"), hair_colour: str = Query(..., description="Hair colour")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id INNER JOIN colour AS T3 ON T1.hair_colour_id = T3.id WHERE T2.colour = ? AND T3.colour = ?", (eye_colour, hair_colour))
    result = cursor.fetchall()
    if not result:
        return {"superhero_names": []}
    return {"superhero_names": [row[0] for row in result]}

# Endpoint to get the count of superheroes from a specific publisher
@app.get("/v1/superhero/count_superheroes_by_publisher", operation_id="get_count_superheroes_by_publisher", summary="Retrieves the total number of superheroes associated with a given publisher. The operation requires the publisher's name as an input parameter to filter the count.")
async def get_count_superheroes_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T2.publisher_name = ?", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the superheroes ranked by height from a specific publisher
@app.get("/v1/superhero/superheroes_ranked_by_height", operation_id="get_superheroes_ranked_by_height", summary="Retrieve a ranked list of superheroes from a specific publisher, sorted by their height in descending order.")
async def get_superheroes_ranked_by_height(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT superhero_name, height_cm, RANK() OVER (ORDER BY height_cm DESC) AS HeightRank FROM superhero INNER JOIN publisher ON superhero.publisher_id = publisher.id WHERE publisher.publisher_name = ?", (publisher_name,))
    result = cursor.fetchall()
    if not result:
        return {"superheroes": []}
    return {"superheroes": [{"superhero_name": row[0], "height_cm": row[1], "HeightRank": row[2]} for row in result]}

# Endpoint to get the publisher name of a specific superhero
@app.get("/v1/superhero/publisher_name_by_superhero", operation_id="get_publisher_name_by_superhero", summary="Retrieves the name of the publisher associated with a specific superhero, based on the superhero's name.")
async def get_publisher_name_by_superhero(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T2.publisher_name FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the eye color popularity rank for superheroes from a specific publisher
@app.get("/v1/superhero/eye_color_popularity_rank", operation_id="get_eye_color_popularity_rank", summary="Retrieve the popularity rank of eye colors among superheroes from a specific publisher. The operation calculates the count of superheroes with each eye color and ranks them based on their frequency. The publisher's name is used to filter the superheroes.")
async def get_eye_color_popularity_rank(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT colour.colour AS EyeColor, COUNT(superhero.id) AS Count, RANK() OVER (ORDER BY COUNT(superhero.id) DESC) AS PopularityRank FROM superhero INNER JOIN colour ON superhero.eye_colour_id = colour.id INNER JOIN publisher ON superhero.publisher_id = publisher.id WHERE publisher.publisher_name = ? GROUP BY colour.colour", (publisher_name,))
    result = cursor.fetchall()
    if not result:
        return {"eye_color_rank": []}
    return {"eye_color_rank": result}

# Endpoint to get the average height of superheroes from a specific publisher
@app.get("/v1/superhero/average_height_by_publisher", operation_id="get_average_height_by_publisher", summary="Retrieves the average height of superheroes associated with a specified publisher. The operation calculates the mean height in centimeters for all superheroes linked to the provided publisher.")
async def get_average_height_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT AVG(T1.height_cm) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T2.publisher_name = ?", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get superheroes with a specific power and publisher
@app.get("/v1/superhero/superheroes_by_power_and_publisher", operation_id="get_superheroes_by_power_and_publisher", summary="Retrieves a list of superheroes who possess a specific power and are published by a certain publisher. The operation filters superheroes based on the provided power name and publisher name.")
async def get_superheroes_by_power_and_publisher(power_name: str = Query(..., description="Name of the power"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT superhero_name FROM superhero AS T1 WHERE EXISTS (SELECT 1 FROM hero_power AS T2 INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T3.power_name = ? AND T1.id = T2.hero_id) AND EXISTS (SELECT 1 FROM publisher AS T4 WHERE T4.publisher_name = ? AND T1.publisher_id = T4.id)", (power_name, publisher_name))
    result = cursor.fetchall()
    if not result:
        return {"superheroes": []}
    return {"superheroes": [row[0] for row in result]}

# Endpoint to get the publisher name of the superhero with the highest attribute value for a specific attribute
@app.get("/v1/superhero/publisher_name_by_highest_attribute", operation_id="get_publisher_name_by_highest_attribute", summary="Retrieves the publisher name of the superhero who has the highest value for a specified attribute. The attribute name must be provided as an input parameter.")
async def get_publisher_name_by_highest_attribute(attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT T2.publisher_name FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN hero_attribute AS T3 ON T1.id = T3.hero_id INNER JOIN attribute AS T4 ON T3.attribute_id = T4.id WHERE T4.attribute_name = ? ORDER BY T3.attribute_value LIMIT 1", (attribute_name,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the count of superheroes from a specific publisher with a specific eye color
@app.get("/v1/superhero/count_superheroes_by_publisher_and_eye_color", operation_id="get_count_superheroes_by_publisher_and_eye_color", summary="Retrieves the total number of superheroes associated with a given publisher who possess a specific eye color. The response is based on the provided publisher name and eye color.")
async def get_count_superheroes_by_publisher_and_eye_color(publisher_name: str = Query(..., description="Name of the publisher"), eye_color: str = Query(..., description="Eye color of the superhero")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN colour AS T3 ON T1.eye_colour_id = T3.id WHERE T2.publisher_name = ? AND T3.colour = ?", (publisher_name, eye_color))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of superheroes with a specific hair color
@app.get("/v1/superhero/count_superheroes_by_hair_color", operation_id="get_count_superheroes_by_hair_color", summary="Retrieves the total number of superheroes with a specified hair color. The operation filters the superheroes based on the provided hair color and returns the count.")
async def get_count_superheroes_by_hair_color(hair_color: str = Query(..., description="Hair color of the superhero")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.hair_colour_id = T2.id WHERE T2.colour = ?", (hair_color,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the superhero with the highest attribute value for a specific attribute
@app.get("/v1/superhero/superhero_by_highest_attribute", operation_id="get_superhero_by_highest_attribute", summary="Retrieves the name of the superhero with the highest value for a specified attribute. The attribute is identified by its name, which is provided as an input parameter.")
async def get_superhero_by_highest_attribute(attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id INNER JOIN attribute AS T3 ON T2.attribute_id = T3.id WHERE T3.attribute_name = ? ORDER BY T2.attribute_value LIMIT 1", (attribute_name,))
    result = cursor.fetchone()
    if not result:
        return {"superhero_name": []}
    return {"superhero_name": result[0]}

# Endpoint to get the race of a superhero by name
@app.get("/v1/superhero/get_race_by_name", operation_id="get_race_by_name", summary="Retrieves the race of a specific superhero by their name. The operation fetches the race information from the superhero and race tables in the database, based on the provided superhero name.")
async def get_race_by_name(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T2.race FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchone()
    if not result:
        return {"race": []}
    return {"race": result[0]}

# Endpoint to get superheroes with a specific attribute value below a threshold
@app.get("/v1/superhero/get_superheroes_by_attribute_value", operation_id="get_superheroes_by_attribute_value", summary="Retrieve a list of superheroes whose attribute value, specified by the attribute name, is below the provided threshold.")
async def get_superheroes_by_attribute_value(attribute_name: str = Query(..., description="Name of the attribute"), attribute_value: int = Query(..., description="Attribute value threshold")):
    cursor.execute("SELECT superhero_name FROM superhero AS T1 WHERE EXISTS (SELECT 1 FROM hero_attribute AS T2 INNER JOIN attribute AS T3 ON T2.attribute_id = T3.id WHERE T3.attribute_name = ? AND T2.attribute_value < ? AND T1.id = T2.hero_id)", (attribute_name, attribute_value))
    result = cursor.fetchall()
    if not result:
        return {"superheroes": []}
    return {"superheroes": [row[0] for row in result]}

# Endpoint to get superheroes with a specific power
@app.get("/v1/superhero/get_superheroes_by_power", operation_id="get_superheroes_by_power", summary="Retrieves a list of superheroes who possess a specific superpower. The power is identified by its name, which is provided as an input parameter.")
async def get_superheroes_by_power(power_name: str = Query(..., description="Name of the power")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T3.power_name = ?", (power_name,))
    result = cursor.fetchall()
    if not result:
        return {"superheroes": []}
    return {"superheroes": [row[0] for row in result]}

# Endpoint to get the count of superheroes with specific attribute and gender
@app.get("/v1/superhero/count_superheroes_by_attribute_and_gender", operation_id="count_superheroes_by_attribute_and_gender", summary="Retrieve the number of superheroes who possess a specific attribute and belong to a certain gender. The attribute is identified by its name and value, while the gender is specified as a parameter. This operation provides a quantitative overview of superheroes with a particular attribute and gender.")
async def count_superheroes_by_attribute_and_gender(attribute_name: str = Query(..., description="Name of the attribute"), attribute_value: int = Query(..., description="Attribute value"), gender: str = Query(..., description="Gender of the superhero")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id INNER JOIN attribute AS T3 ON T2.attribute_id = T3.id INNER JOIN gender AS T4 ON T1.gender_id = T4.id WHERE T3.attribute_name = ? AND T2.attribute_value = ? AND T4.gender = ?", (attribute_name, attribute_value, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the superhero with the most powers
@app.get("/v1/superhero/get_superhero_with_most_powers", operation_id="get_superhero_with_most_powers", summary="Retrieves the name of the superhero who possesses the highest number of powers. This operation identifies the superhero with the most powers by joining the superhero and hero_power tables, grouping by superhero name, and ordering the results in descending order based on the count of powers. The top result is then returned.")
async def get_superhero_with_most_powers():
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id GROUP BY T1.superhero_name ORDER BY COUNT(T2.hero_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"superhero": []}
    return {"superhero": result[0]}

# Endpoint to get the count of superheroes by race
@app.get("/v1/superhero/count_superheroes_by_race", operation_id="count_superheroes_by_race", summary="Retrieves the total number of superheroes belonging to a specific race. The operation filters superheroes based on the provided race and returns the count.")
async def count_superheroes_by_race(race: str = Query(..., description="Race of the superhero")):
    cursor.execute("SELECT COUNT(T1.superhero_name) FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id WHERE T2.race = ?", (race,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of superheroes with a specific alignment and publisher
@app.get("/v1/superhero/percentage_superheroes_by_alignment_and_publisher", operation_id="percentage_superheroes_by_alignment_and_publisher", summary="Retrieves the percentage of superheroes with a specific alignment and publisher. The operation calculates the total percentage of superheroes and the percentage of superheroes associated with the specified publisher, based on the provided alignment.")
async def percentage_superheroes_by_alignment_and_publisher(publisher_name: str = Query(..., description="Name of the publisher"), alignment: str = Query(..., description="Alignment of the superhero")):
    cursor.execute("SELECT (CAST(COUNT(*) AS REAL) * 100 / (SELECT COUNT(*) FROM superhero)), CAST(SUM(CASE WHEN T2.publisher_name = ? THEN 1 ELSE 0 END) AS REAL) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN alignment AS T3 ON T3.id = T1.alignment_id WHERE T3.alignment = ?", (publisher_name, alignment))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0], "publisher_count": result[1]}

# Endpoint to get the difference in the number of superheroes between two publishers
@app.get("/v1/superhero/difference_superheroes_between_publishers", operation_id="difference_superheroes_between_publishers", summary="Retrieves the difference in the total number of superheroes associated with two specified publishers. The operation compares the count of superheroes linked to the first publisher with the count of superheroes linked to the second publisher, and returns the difference.")
async def difference_superheroes_between_publishers(publisher_name_1: str = Query(..., description="Name of the first publisher"), publisher_name_2: str = Query(..., description="Name of the second publisher")):
    cursor.execute("SELECT SUM(CASE WHEN T2.publisher_name = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.publisher_name = ? THEN 1 ELSE 0 END) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id", (publisher_name_1, publisher_name_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the publisher ID by name
@app.get("/v1/superhero/get_publisher_id_by_name", operation_id="get_publisher_id_by_name", summary="Retrieves the unique identifier of a publisher based on their name. The operation requires the name of the publisher as input and returns the corresponding publisher ID.")
async def get_publisher_id_by_name(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT id FROM publisher WHERE publisher_name = ?", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_id": []}
    return {"publisher_id": result[0]}

# Endpoint to get the average attribute value of all superheroes
@app.get("/v1/superhero/average_attribute_value", operation_id="average_attribute_value", summary="Retrieves the average value of a specific attribute across all superheroes in the database.")
async def average_attribute_value():
    cursor.execute("SELECT AVG(attribute_value) FROM hero_attribute")
    result = cursor.fetchone()
    if not result:
        return {"average_attribute_value": []}
    return {"average_attribute_value": result[0]}

# Endpoint to get the count of superheroes with no full name
@app.get("/v1/superhero/count_no_full_name", operation_id="get_count_no_full_name", summary="Retrieves the total count of superheroes in the database who do not have a full name specified.")
async def get_count_no_full_name():
    cursor.execute("SELECT COUNT(id) FROM superhero WHERE full_name IS NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the eye colour of a superhero by ID
@app.get("/v1/superhero/eye_colour_by_id", operation_id="get_eye_colour_by_id", summary="Retrieves the eye colour of a specific superhero, identified by their unique ID. The operation returns the colour name associated with the superhero's eye colour ID from the colour table.")
async def get_eye_colour_by_id(id: int = Query(..., description="ID of the superhero")):
    cursor.execute("SELECT T2.colour FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T1.id = ?", (id,))
    result = cursor.fetchone()
    if not result:
        return {"colour": []}
    return {"colour": result[0]}

# Endpoint to get the average weight of superheroes by gender
@app.get("/v1/superhero/avg_weight_by_gender", operation_id="get_avg_weight_by_gender", summary="Retrieves the average weight of superheroes based on the specified gender. The operation calculates the mean weight from the superhero dataset, filtering results by the provided gender parameter.")
async def get_avg_weight_by_gender(gender: str = Query(..., description="Gender of the superhero")):
    cursor.execute("SELECT AVG(T1.weight_kg) FROM superhero AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.id WHERE T2.gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"avg_weight": []}
    return {"avg_weight": result[0]}

# Endpoint to get the powers of male superheroes with a limit
@app.get("/v1/superhero/male_powers_with_limit", operation_id="get_male_powers_with_limit", summary="Retrieves a specified number of powers associated with male superheroes. The gender parameter filters the results to male superheroes, while the limit parameter determines the maximum number of powers returned.")
async def get_male_powers_with_limit(gender: str = Query(..., description="Gender of the superhero"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT T3.power_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T3.id = T2.power_id INNER JOIN gender AS T4 ON T4.id = T1.gender_id WHERE T4.gender = ? LIMIT ?", (gender, limit))
    result = cursor.fetchall()
    if not result:
        return {"powers": []}
    return {"powers": [row[0] for row in result]}

# Endpoint to get the names of superheroes by race
@app.get("/v1/superhero/names_by_race", operation_id="get_names_by_race", summary="Retrieves the names of superheroes belonging to a specific race. The operation filters superheroes based on the provided race and returns their names.")
async def get_names_by_race(race: str = Query(..., description="Race of the superhero")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id WHERE T2.race = ?", (race,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get distinct superhero names by height range and eye colour
@app.get("/v1/superhero/names_by_height_and_eye_colour", operation_id="get_names_by_height_and_eye_colour", summary="Retrieves a list of distinct superhero names that fall within a specified height range and have a particular eye colour. The height range is defined by a minimum and maximum height in centimeters, and the eye colour is specified as a string.")
async def get_names_by_height_and_eye_colour(min_height: int = Query(..., description="Minimum height in cm"), max_height: int = Query(..., description="Maximum height in cm"), eye_colour: str = Query(..., description="Eye colour of the superhero")):
    cursor.execute("SELECT DISTINCT T1.superhero_name FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T1.height_cm BETWEEN ? AND ? AND T2.colour = ?", (min_height, max_height, eye_colour))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the powers of a superhero by hero ID
@app.get("/v1/superhero/powers_by_hero_id", operation_id="get_powers_by_hero_id", summary="Retrieves the names of the powers associated with a specific superhero, identified by their unique hero ID. This operation provides a comprehensive list of abilities that the superhero possesses.")
async def get_powers_by_hero_id(hero_id: int = Query(..., description="ID of the hero")):
    cursor.execute("SELECT T2.power_name FROM hero_power AS T1 INNER JOIN superpower AS T2 ON T1.power_id = T2.id WHERE T1.hero_id = ?", (hero_id,))
    result = cursor.fetchall()
    if not result:
        return {"powers": []}
    return {"powers": [row[0] for row in result]}

# Endpoint to get the full names of superheroes by race
@app.get("/v1/superhero/full_names_by_race", operation_id="get_full_names_by_race", summary="Retrieve the full names of superheroes belonging to a specific race. The operation filters superheroes based on the provided race and returns their full names.")
async def get_full_names_by_race(race: str = Query(..., description="Race of the superhero")):
    cursor.execute("SELECT T1.full_name FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id WHERE T2.race = ?", (race,))
    result = cursor.fetchall()
    if not result:
        return {"full_names": []}
    return {"full_names": [row[0] for row in result]}

# Endpoint to get the count of superheroes by alignment
@app.get("/v1/superhero/count_by_alignment", operation_id="get_count_by_alignment", summary="Retrieves the total number of superheroes who share a specified alignment. The alignment is a characteristic that defines the moral compass of a superhero.")
async def get_count_by_alignment(alignment: str = Query(..., description="Alignment of the superhero")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN alignment AS T2 ON T1.alignment_id = T2.id WHERE T2.alignment = ?", (alignment,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the race of superheroes based on their weight in kilograms
@app.get("/v1/superhero/race_by_weight", operation_id="get_race_by_weight", summary="Retrieve the race of superheroes that match the specified weight in kilograms. This operation returns the race of superheroes whose weight exactly matches the provided input. The input parameter is used to filter the superheroes by their weight.")
async def get_race_by_weight(weight_kg: int = Query(..., description="Weight of the superhero in kilograms")):
    cursor.execute("SELECT T2.race FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id WHERE T1.weight_kg = ?", (weight_kg,))
    result = cursor.fetchall()
    if not result:
        return {"races": []}
    return {"races": [row[0] for row in result]}

# Endpoint to get distinct hair colours of superheroes based on their height and race
@app.get("/v1/superhero/hair_colour_by_height_race", operation_id="get_hair_colour_by_height_race", summary="Retrieve the unique hair colours of superheroes who match the specified height and race. This operation allows you to filter superheroes by their height in centimeters and race, and then returns the distinct hair colours found among them.")
async def get_hair_colour_by_height_race(height_cm: int = Query(..., description="Height of the superhero in centimeters"), race: str = Query(..., description="Race of the superhero")):
    cursor.execute("SELECT DISTINCT T3.colour FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id INNER JOIN colour AS T3 ON T1.hair_colour_id = T3.id WHERE T1.height_cm = ? AND T2.race = ?", (height_cm, race))
    result = cursor.fetchall()
    if not result:
        return {"colours": []}
    return {"colours": [row[0] for row in result]}

# Endpoint to get the eye colour of the heaviest superhero
@app.get("/v1/superhero/eye_colour_heaviest", operation_id="get_eye_colour_heaviest", summary="Retrieves the eye colour of the superhero with the highest weight. The operation sorts superheroes by their weight in descending order and returns the eye colour of the top-ranked superhero.")
async def get_eye_colour_heaviest():
    cursor.execute("SELECT T2.colour FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id ORDER BY T1.weight_kg DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"colour": []}
    return {"colour": result[0]}

# Endpoint to get the percentage of superheroes from a specific publisher within a height range
@app.get("/v1/superhero/publisher_percentage_by_height", operation_id="get_publisher_percentage_by_height", summary="Retrieve the proportion of superheroes from a specified publisher who fall within a defined height range, expressed in centimeters. This operation calculates the percentage by comparing the count of superheroes from the given publisher within the height range to the total count of superheroes.")
async def get_publisher_percentage_by_height(publisher_name: str = Query(..., description="Name of the publisher"), min_height_cm: int = Query(..., description="Minimum height in centimeters"), max_height_cm: int = Query(..., description="Maximum height in centimeters")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.publisher_name = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T1.height_cm BETWEEN ? AND ?", (publisher_name, min_height_cm, max_height_cm))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get superheroes of a specific gender whose weight exceeds a certain percentage of the average weight
@app.get("/v1/superhero/superheroes_by_gender_weight", operation_id="get_superheroes_by_gender_weight", summary="Retrieves the names of superheroes of a specified gender who weigh more than a certain percentage of the average superhero weight. The gender is provided as a parameter, and the weight percentage is calculated based on the average weight of all superheroes.")
async def get_superheroes_by_gender_weight(gender: str = Query(..., description="Gender of the superhero"), weight_percentage: int = Query(..., description="Percentage of the average weight")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.id WHERE T2.gender = ? AND T1.weight_kg * 100 > ( SELECT AVG(weight_kg) FROM superhero ) * ?", (gender, weight_percentage))
    result = cursor.fetchall()
    if not result:
        return {"superheroes": []}
    return {"superheroes": [row[0] for row in result]}

# Endpoint to get the most common superpower among superheroes
@app.get("/v1/superhero/most_common_superpower", operation_id="get_most_common_superpower", summary="Retrieves the most frequently occurring superpower among all superheroes. The operation calculates the frequency of each superpower and returns the one with the highest count.")
async def get_most_common_superpower():
    cursor.execute("SELECT T2.power_name FROM hero_power AS T1 INNER JOIN superpower AS T2 ON T1.power_id = T2.id GROUP BY T2.power_name ORDER BY COUNT(T1.hero_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"power_name": []}
    return {"power_name": result[0]}

# Endpoint to get the attribute values of a specific superhero
@app.get("/v1/superhero/attribute_values_by_name", operation_id="get_attribute_values_by_name", summary="Retrieve the attribute values associated with a specific superhero. The operation requires the superhero's name as input and returns the corresponding attribute values from the database.")
async def get_attribute_values_by_name(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T2.attribute_value FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchall()
    if not result:
        return {"attribute_values": []}
    return {"attribute_values": [row[0] for row in result]}

# Endpoint to get distinct superpowers of a specific superhero
@app.get("/v1/superhero/superpowers_by_hero_id", operation_id="get_superpowers_by_hero_id", summary="Retrieves a unique list of superpowers associated with a specific superhero, identified by their unique hero_id.")
async def get_superpowers_by_hero_id(hero_id: int = Query(..., description="ID of the superhero")):
    cursor.execute("SELECT DISTINCT T2.power_name FROM hero_power AS T1 INNER JOIN superpower AS T2 ON T1.power_id = T2.id WHERE T1.hero_id = ?", (hero_id,))
    result = cursor.fetchall()
    if not result:
        return {"superpowers": []}
    return {"superpowers": [row[0] for row in result]}

# Endpoint to get the superhero with the highest value for a specific attribute
@app.get("/v1/superhero/highest_attribute_value", operation_id="get_highest_attribute_value", summary="Retrieves the superhero with the highest value for a specified attribute. The attribute is identified by its name, which is provided as an input parameter. The operation returns the full name of the superhero with the highest value for the specified attribute.")
async def get_highest_attribute_value(attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT T1.full_name FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id INNER JOIN attribute AS T3 ON T2.attribute_id = T3.id WHERE T3.attribute_name = ? ORDER BY T2.attribute_value DESC LIMIT 1", (attribute_name,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the ratio of superheroes with a specific skin colour
@app.get("/v1/superhero/ratio_skin_colour", operation_id="get_ratio_skin_colour", summary="Retrieves the proportion of superheroes with a specific skin colour. The operation calculates the ratio by dividing the total number of superheroes by the count of superheroes with the provided skin colour ID. This endpoint requires the skin colour ID as an input parameter.")
async def get_ratio_skin_colour(skin_colour_id: int = Query(..., description="ID of the skin colour")):
    cursor.execute("SELECT CAST(COUNT(*) AS REAL) / SUM(CASE WHEN T2.id = ? THEN 1 ELSE 0 END) FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.skin_colour_id = T2.id", (skin_colour_id,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the superhero with the highest attribute value from a specific publisher
@app.get("/v1/superhero/top_attribute_superhero", operation_id="get_top_attribute_superhero", summary="Retrieves the superhero with the highest value for a specified attribute from a given publisher. The operation filters superheroes by publisher and attribute, then sorts them by attribute value in descending order to identify the top superhero.")
async def get_top_attribute_superhero(publisher_name: str = Query(..., description="Name of the publisher"), attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id INNER JOIN attribute AS T3 ON T3.id = T2.attribute_id INNER JOIN publisher AS T4 ON T4.id = T1.publisher_id WHERE T4.publisher_name = ? AND T3.attribute_name = ? ORDER BY T2.attribute_value DESC LIMIT 1", (publisher_name, attribute_name))
    result = cursor.fetchone()
    if not result:
        return {"superhero_name": []}
    return {"superhero_name": result[0]}

# Endpoint to get the eye colour of a superhero by full name
@app.get("/v1/superhero/eye_colour_by_full_name", operation_id="get_eye_colour_by_full_name", summary="Retrieves the eye colour of a specific superhero based on their full name. The operation requires the full name of the superhero as an input parameter and returns the corresponding eye colour from the superhero database.")
async def get_eye_colour_by_full_name(full_name: str = Query(..., description="Full name of the superhero")):
    cursor.execute("SELECT T2.colour FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T1.full_name = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"colour": []}
    return {"colour": result[0]}

# Endpoint to get colour IDs of superheroes from a specific publisher and gender
@app.get("/v1/superhero/colour_ids_by_publisher_gender", operation_id="get_colour_ids_by_publisher_gender", summary="Retrieves the unique identifiers for eye, hair, and skin colours of superheroes from a specific publisher and gender. The operation filters superheroes based on the provided publisher name and gender, and returns the corresponding colour IDs.")
async def get_colour_ids_by_publisher_gender(publisher_name: str = Query(..., description="Name of the publisher"), gender: str = Query(..., description="Gender of the superhero")):
    cursor.execute("SELECT T1.eye_colour_id, T1.hair_colour_id, T1.skin_colour_id FROM superhero AS T1 INNER JOIN publisher AS T2 ON T2.id = T1.publisher_id INNER JOIN gender AS T3 ON T3.id = T1.gender_id WHERE T2.publisher_name = ? AND T3.gender = ?", (publisher_name, gender))
    result = cursor.fetchall()
    if not result:
        return {"colour_ids": []}
    return {"colour_ids": [{"eye_colour_id": row[0], "hair_colour_id": row[1], "skin_colour_id": row[2]} for row in result]}

# Endpoint to get superheroes with matching eye, hair, and skin colour IDs
@app.get("/v1/superhero/matching_colour_ids", operation_id="get_matching_colour_ids", summary="Retrieves a list of superheroes who share the same colour ID for their eye, hair, and skin. The response includes the superhero's name and the name of their publisher.")
async def get_matching_colour_ids():
    cursor.execute("SELECT T1.superhero_name, T2.publisher_name FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T1.eye_colour_id = T1.hair_colour_id AND T1.eye_colour_id = T1.skin_colour_id")
    result = cursor.fetchall()
    if not result:
        return {"superheroes": []}
    return {"superheroes": [{"superhero_name": row[0], "publisher_name": row[1]} for row in result]}

# Endpoint to get the percentage of female superheroes with a specific skin colour
@app.get("/v1/superhero/percentage_female_skin_colour", operation_id="get_percentage_female_skin_colour", summary="Retrieves the percentage of female superheroes with a specified skin colour. This operation calculates the ratio of female superheroes with the given skin colour to the total number of female superheroes in the database.")
async def get_percentage_female_skin_colour(skin_colour: str = Query(..., description="Skin colour of the superhero"), gender: str = Query(..., description="Gender of the superhero")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.colour = ? THEN T1.id ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM superhero AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.id INNER JOIN colour AS T3 ON T1.skin_colour_id = T3.id WHERE T2.gender = ?", (skin_colour, gender))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the race of a superhero by full name
@app.get("/v1/superhero/race_by_full_name", operation_id="get_race_by_full_name", summary="Retrieves the race of a specific superhero based on their full name. The operation uses the provided full name to search for the corresponding superhero in the database and returns the superhero's name along with their associated race.")
async def get_race_by_full_name(full_name: str = Query(..., description="Full name of the superhero")):
    cursor.execute("SELECT T1.superhero_name, T2.race FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id WHERE T1.full_name = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"superhero_name": [], "race": []}
    return {"superhero_name": result[0], "race": result[1]}

# Endpoint to get the gender of a superhero by their name
@app.get("/v1/superhero/gender_by_name", operation_id="get_gender_by_name", summary="Retrieves the gender of a specific superhero by their name. The operation uses the provided superhero name to look up the corresponding gender in the superhero and gender tables.")
async def get_gender_by_name(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T2.gender FROM superhero AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get the count of powers for a specific superhero
@app.get("/v1/superhero/power_count_by_name", operation_id="get_power_count_by_name", summary="Retrieve the total number of powers associated with a specific superhero. The operation requires the superhero's name as input and returns the count of powers linked to that superhero.")
async def get_power_count_by_name(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT COUNT(T1.power_id) FROM hero_power AS T1 INNER JOIN superhero AS T2 ON T1.hero_id = T2.id WHERE T2.superhero_name = ?", (superhero_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the powers of a superhero by their full name
@app.get("/v1/superhero/powers_by_full_name", operation_id="get_powers_by_full_name", summary="Retrieves the superpowers associated with a specific superhero, identified by their full name. The operation returns a list of power names that the superhero possesses.")
async def get_powers_by_full_name(full_name: str = Query(..., description="Full name of the superhero")):
    cursor.execute("SELECT T3.power_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T1.full_name = ?", (full_name,))
    result = cursor.fetchall()
    if not result:
        return {"powers": []}
    return {"powers": [row[0] for row in result]}

# Endpoint to get the height of superheroes with a specific eye colour
@app.get("/v1/superhero/height_by_eye_colour", operation_id="get_height_by_eye_colour", summary="Retrieves the height of superheroes with a specified eye colour. The operation filters superheroes based on the provided eye colour and returns their respective heights in centimeters.")
async def get_height_by_eye_colour(eye_colour: str = Query(..., description="Eye colour of the superhero")):
    cursor.execute("SELECT T1.height_cm FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T2.colour = ?", (eye_colour,))
    result = cursor.fetchall()
    if not result:
        return {"heights": []}
    return {"heights": [row[0] for row in result]}

# Endpoint to get superheroes with a specific eye and hair colour
@app.get("/v1/superhero/superheroes_by_eye_and_hair_colour", operation_id="get_superheroes_by_eye_and_hair_colour", summary="Retrieve the names of superheroes who share a specific eye and hair colour. The operation filters superheroes based on a provided colour, which is used to match both their eye and hair colour. The result is a list of superhero names that meet the specified criteria.")
async def get_superheroes_by_eye_and_hair_colour(colour: str = Query(..., description="Colour of the eyes and hair of the superhero")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id AND T1.hair_colour_id = T2.id WHERE T2.colour = ?", (colour,))
    result = cursor.fetchall()
    if not result:
        return {"superheroes": []}
    return {"superheroes": [row[0] for row in result]}

# Endpoint to get the eye colour of superheroes with a specific skin colour
@app.get("/v1/superhero/eye_colour_by_skin_colour", operation_id="get_eye_colour_by_skin_colour", summary="Retrieve the eye colour of superheroes who have a specified skin colour. The operation filters superheroes based on their skin colour and returns the corresponding eye colour.")
async def get_eye_colour_by_skin_colour(skin_colour: str = Query(..., description="Skin colour of the superhero")):
    cursor.execute("SELECT T2.colour FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id INNER JOIN colour AS T3 ON T1.skin_colour_id = T3.id WHERE T3.colour = ?", (skin_colour,))
    result = cursor.fetchall()
    if not result:
        return {"eye_colours": []}
    return {"eye_colours": [row[0] for row in result]}

# Endpoint to get the names of superheroes with a specific alignment
@app.get("/v1/superhero/superheroes_by_alignment", operation_id="get_superheroes_by_alignment", summary="Retrieve the names of superheroes who share a specified alignment. The alignment is used to filter the superheroes and return only those who match the given alignment.")
async def get_superheroes_by_alignment(alignment: str = Query(..., description="Alignment of the superhero")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN alignment AS T2 ON T1.alignment_id = T2.id WHERE T2.alignment = ?", (alignment,))
    result = cursor.fetchall()
    if not result:
        return {"superheroes": []}
    return {"superheroes": [row[0] for row in result]}

# Endpoint to get the count of superheroes with the maximum value of a specific attribute
@app.get("/v1/superhero/count_max_attribute", operation_id="get_count_max_attribute", summary="Retrieves the number of superheroes who possess the maximum value for a specified attribute. The attribute is identified by its name.")
async def get_count_max_attribute(attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT COUNT(T1.hero_id) FROM hero_attribute AS T1 INNER JOIN attribute AS T2 ON T1.attribute_id = T2.id WHERE T2.attribute_name = ? AND T1.attribute_value = ( SELECT MAX(attribute_value) FROM hero_attribute )", (attribute_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the race and alignment of a superhero by name
@app.get("/v1/superhero/get_race_alignment_by_name", operation_id="get_race_alignment_by_name", summary="Retrieves the race and alignment of a specific superhero by their name. The input parameter is used to identify the superhero and the operation returns the corresponding race and alignment details.")
async def get_race_alignment_by_name(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T2.race, T3.alignment FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id INNER JOIN alignment AS T3 ON T1.alignment_id = T3.id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchone()
    if not result:
        return {"race": [], "alignment": []}
    return {"race": result[0], "alignment": result[1]}

# Endpoint to get the percentage of superheroes from a specific publisher who are of a specific gender
@app.get("/v1/superhero/get_publisher_gender_percentage", operation_id="get_publisher_gender_percentage", summary="Retrieves the percentage of superheroes from a specified publisher who identify as a particular gender. This operation calculates the ratio of superheroes from the given publisher and gender to the total number of superheroes in the database.")
async def get_publisher_gender_percentage(publisher_name: str = Query(..., description="Name of the publisher"), gender: str = Query(..., description="Gender of the superhero")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.publisher_name = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN gender AS T3 ON T1.gender_id = T3.id WHERE T3.gender = ?", (publisher_name, gender))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average weight of superheroes of a specific race
@app.get("/v1/superhero/get_average_weight_by_race", operation_id="get_average_weight_by_race", summary="Retrieves the average weight of superheroes belonging to a specific race. The operation calculates the average weight by summing the weights of all superheroes of the specified race and dividing by the total count of those superheroes.")
async def get_average_weight_by_race(race: str = Query(..., description="Race of the superhero")):
    cursor.execute("SELECT CAST(SUM(T1.weight_kg) AS REAL) / COUNT(T1.id) FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id WHERE T2.race = ?", (race,))
    result = cursor.fetchone()
    if not result:
        return {"average_weight": []}
    return {"average_weight": result[0]}

# Endpoint to calculate the weight difference between two superheroes
@app.get("/v1/superhero/get_weight_difference", operation_id="get_weight_difference", summary="This operation computes the weight difference between two specified superheroes. It takes the full names of the two superheroes as input and returns the difference in their weights in kilograms. The result is calculated by subtracting the weight of the second superhero from the weight of the first superhero.")
async def get_weight_difference(full_name1: str = Query(..., description="Full name of the first superhero"), full_name2: str = Query(..., description="Full name of the second superhero")):
    cursor.execute("SELECT ( SELECT weight_kg FROM superhero WHERE full_name LIKE ? ) - ( SELECT weight_kg FROM superhero WHERE full_name LIKE ? ) AS CALCULATE", (full_name1, full_name2))
    result = cursor.fetchone()
    if not result:
        return {"weight_difference": []}
    return {"weight_difference": result[0]}

# Endpoint to get the average height of all superheroes
@app.get("/v1/superhero/get_average_height", operation_id="get_average_height", summary="Retrieves the average height of all superheroes in the database, calculated by summing up the heights and dividing by the total number of superheroes.")
async def get_average_height():
    cursor.execute("SELECT CAST(SUM(height_cm) AS REAL) / COUNT(id) FROM superhero")
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the count of superheroes by race and gender
@app.get("/v1/superhero/get_count_by_race_gender", operation_id="get_count_by_race_gender", summary="Retrieves the total count of superheroes belonging to a specific race and gender. The operation requires the unique identifiers for the desired race and gender as input parameters. The response will provide a single numerical value representing the count of superheroes that match the specified race and gender.")
async def get_count_by_race_gender(race_id: int = Query(..., description="ID of the race"), gender_id: int = Query(..., description="ID of the gender")):
    cursor.execute("SELECT COUNT(*) FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id INNER JOIN gender AS T3 ON T3.id = T1.gender_id WHERE T1.race_id = ? AND T1.gender_id = ?", (race_id, gender_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the superhero with the highest value of a specific attribute
@app.get("/v1/superhero/get_top_superhero_by_attribute", operation_id="get_top_superhero_by_attribute", summary="Retrieves the name of the superhero with the highest value for a specified attribute. The attribute is provided as an input parameter.")
async def get_top_superhero_by_attribute(attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id INNER JOIN attribute AS T3 ON T2.attribute_id = T3.id WHERE T3.attribute_name = ? ORDER BY T2.attribute_value DESC LIMIT 1", (attribute_name,))
    result = cursor.fetchone()
    if not result:
        return {"superhero_name": []}
    return {"superhero_name": result[0]}

# Endpoint to get the attributes of a superhero by name
@app.get("/v1/superhero/get_attributes_by_name", operation_id="get_attributes_by_name", summary="Retrieves the specific attributes and their corresponding values for a given superhero, based on the provided superhero name.")
async def get_attributes_by_name(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T3.attribute_name, T2.attribute_value FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id INNER JOIN attribute AS T3 ON T2.attribute_id = T3.id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchall()
    if not result:
        return {"attributes": []}
    return {"attributes": [{"attribute_name": row[0], "attribute_value": row[1]} for row in result]}

# Endpoint to get publisher names based on superhero names
@app.get("/v1/superhero/publisher_names_by_superhero_names", operation_id="get_publisher_names_by_superhero_names", summary="Retrieve the names of publishers associated with a specified set of superhero names. This operation accepts up to three superhero names and returns the corresponding publisher names. The input parameters represent the names of the superheroes for which the publisher names are sought.")
async def get_publisher_names_by_superhero_names(superhero_name1: str = Query(..., description="First superhero name"), superhero_name2: str = Query(..., description="Second superhero name"), superhero_name3: str = Query(..., description="Third superhero name")):
    cursor.execute("SELECT T2.publisher_name FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T1.superhero_name IN (?, ?, ?)", (superhero_name1, superhero_name2, superhero_name3))
    result = cursor.fetchall()
    if not result:
        return {"publisher_names": []}
    return {"publisher_names": [row[0] for row in result]}

# Endpoint to get the count of superheroes by publisher ID
@app.get("/v1/superhero/count_superheroes_by_publisher_id", operation_id="get_count_superheroes_by_publisher_id", summary="Retrieves the total number of superheroes associated with a specific publisher. The publisher is identified by its unique ID. The response provides a count of all superheroes linked to the given publisher.")
async def get_count_superheroes_by_publisher_id(publisher_id: int = Query(..., description="Publisher ID")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T2.id = ?", (publisher_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of superheroes with a specific eye colour
@app.get("/v1/superhero/percentage_superheroes_by_eye_colour", operation_id="get_percentage_superheroes_by_eye_colour", summary="Retrieves the percentage of superheroes with a specific eye colour. The operation calculates this percentage by comparing the count of superheroes with the specified eye colour to the total number of superheroes in the database.")
async def get_percentage_superheroes_by_eye_colour(eye_colour: str = Query(..., description="Eye colour of the superhero")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.colour = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id", (eye_colour,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of male to female superheroes
@app.get("/v1/superhero/ratio_male_to_female_superheroes", operation_id="get_ratio_male_to_female_superheroes", summary="Retrieves the ratio of male to female superheroes based on the provided gender parameters.")
async def get_ratio_male_to_female_superheroes(male_gender: str = Query(..., description="Gender of the superhero (Male)"), female_gender: str = Query(..., description="Gender of the superhero (Female)")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.gender = ? THEN T1.id ELSE NULL END) AS REAL) / COUNT(CASE WHEN T2.gender = ? THEN T1.id ELSE NULL END) FROM superhero AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.id", (male_gender, female_gender))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the tallest superhero
@app.get("/v1/superhero/tallest_superhero", operation_id="get_tallest_superhero", summary="Retrieves the name of the superhero with the greatest height. The operation returns the name of the tallest superhero in the database, sorted by height in descending order and limited to the top result.")
async def get_tallest_superhero():
    cursor.execute("SELECT superhero_name FROM superhero ORDER BY height_cm DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"superhero_name": []}
    return {"superhero_name": result[0]}

# Endpoint to get the superpower ID by power name
@app.get("/v1/superhero/superpower_id_by_name", operation_id="get_superpower_id_by_name", summary="Retrieves the unique identifier of a superpower, given its name. The operation searches for a superpower with the specified name and returns its corresponding ID.")
async def get_superpower_id_by_name(power_name: str = Query(..., description="Name of the superpower")):
    cursor.execute("SELECT id FROM superpower WHERE power_name = ?", (power_name,))
    result = cursor.fetchone()
    if not result:
        return {"superpower_id": []}
    return {"superpower_id": result[0]}

# Endpoint to get the superhero name by ID
@app.get("/v1/superhero/superhero_name_by_id", operation_id="get_superhero_name_by_id", summary="Retrieves the name of a specific superhero by using the provided superhero ID. The superhero's name is fetched from the superhero database using the given ID.")
async def get_superhero_name_by_id(superhero_id: int = Query(..., description="ID of the superhero")):
    cursor.execute("SELECT superhero_name FROM superhero WHERE id = ?", (superhero_id,))
    result = cursor.fetchone()
    if not result:
        return {"superhero_name": []}
    return {"superhero_name": result[0]}

# Endpoint to get distinct full names of superheroes with specific weight criteria
@app.get("/v1/superhero/distinct_full_names_by_weight", operation_id="get_distinct_full_names_by_weight", summary="Retrieve a unique list of full names of superheroes who either have no weight specified or weigh a specific amount. This operation allows you to filter superheroes based on their weight, providing a distinct set of full names that meet the specified weight criteria.")
async def get_distinct_full_names_by_weight(weight_kg: int = Query(..., description="Weight of the superhero in kilograms")):
    cursor.execute("SELECT DISTINCT full_name FROM superhero WHERE full_name IS NOT NULL AND (weight_kg IS NULL OR weight_kg = ?)", (weight_kg,))
    result = cursor.fetchall()
    if not result:
        return {"full_names": []}
    return {"full_names": [row[0] for row in result]}

# Endpoint to get distinct races of superheroes by weight and height
@app.get("/v1/superhero/distinct_races_by_weight_height", operation_id="get_distinct_races_by_weight_height", summary="Retrieve the unique races of superheroes who share the same weight and height. The operation requires the weight and height of the superhero as input parameters, which are used to filter the superhero data and identify the distinct races.")
async def get_distinct_races_by_weight_height(weight_kg: int = Query(..., description="Weight of the superhero in kilograms"), height_cm: int = Query(..., description="Height of the superhero in centimeters")):
    cursor.execute("SELECT DISTINCT T2.race FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id WHERE T1.weight_kg = ? AND T1.height_cm = ?", (weight_kg, height_cm))
    result = cursor.fetchall()
    if not result:
        return {"races": []}
    return {"races": [row[0] for row in result]}

# Endpoint to get the publisher name of a superhero by their ID
@app.get("/v1/superhero/publisher_name_by_id", operation_id="get_publisher_name_by_id", summary="Retrieves the name of the publisher associated with a specific superhero, identified by their unique ID. The publisher name is obtained by joining the superhero and publisher tables using the publisher ID.")
async def get_publisher_name_by_id(superhero_id: int = Query(..., description="ID of the superhero")):
    cursor.execute("SELECT T2.publisher_name FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T1.id = ?", (superhero_id,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the race of the superhero with the highest attribute value
@app.get("/v1/superhero/race_by_highest_attribute_value", operation_id="get_race_by_highest_attribute_value", summary="Retrieves the race of the superhero who possesses the highest attribute value. This operation fetches the race of the superhero with the highest attribute value from the database by joining the superhero, hero_attribute, and race tables. The result is the race of the superhero with the highest attribute value.")
async def get_race_by_highest_attribute_value():
    cursor.execute("SELECT T3.race FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id INNER JOIN race AS T3 ON T1.race_id = T3.id ORDER BY T2.attribute_value DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"race": []}
    return {"race": result[0]}

# Endpoint to get the alignment and power names of a superhero by their name
@app.get("/v1/superhero/alignment_power_names_by_name", operation_id="get_alignment_power_names_by_name", summary="Retrieves the alignment and associated power names of a specific superhero, identified by their name.")
async def get_alignment_power_names_by_name(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T4.alignment, T3.power_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T3.id = T2.power_id INNER JOIN alignment AS T4 ON T1.alignment_id = T4.id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchall()
    if not result:
        return {"alignment_power_names": []}
    return {"alignment_power_names": [{"alignment": row[0], "power_name": row[1]} for row in result]}

# Endpoint to get the names of superheroes with a specific eye colour
@app.get("/v1/superhero/names_by_eye_colour", operation_id="get_names_by_eye_colour", summary="Retrieve a list of up to five superhero names that match a specified eye colour. The endpoint fetches the names from the superhero table, filtering the results based on the provided eye colour. The eye colour parameter is used to determine the filtering criteria.")
async def get_names_by_eye_colour(eye_colour: str = Query(..., description="Eye colour of the superhero")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T2.colour = ? LIMIT 5", (eye_colour,))
    result = cursor.fetchall()
    if not result:
        return {"superhero_names": []}
    return {"superhero_names": [row[0] for row in result]}

# Endpoint to get the average attribute value of superheroes with a specific alignment
@app.get("/v1/superhero/avg_attribute_value_by_alignment", operation_id="get_avg_attribute_value_by_alignment", summary="Retrieves the average attribute value of superheroes who share a specific alignment. The alignment is provided as an input parameter, allowing the calculation to be tailored to a particular group of superheroes.")
async def get_avg_attribute_value_by_alignment(alignment: str = Query(..., description="Alignment of the superhero")):
    cursor.execute("SELECT AVG(T1.attribute_value) FROM hero_attribute AS T1 INNER JOIN superhero AS T2 ON T1.hero_id = T2.id INNER JOIN alignment AS T3 ON T2.alignment_id = T3.id WHERE T3.alignment = ?", (alignment,))
    result = cursor.fetchone()
    if not result:
        return {"avg_attribute_value": []}
    return {"avg_attribute_value": result[0]}

# Endpoint to get distinct skin colours of superheroes with a specific attribute value
@app.get("/v1/superhero/distinct_skin_colours_by_attribute_value", operation_id="get_distinct_skin_colours_by_attribute_value", summary="Retrieve the unique skin colours of superheroes who possess a specific attribute value. The attribute value is provided as an input parameter.")
async def get_distinct_skin_colours_by_attribute_value(attribute_value: int = Query(..., description="Attribute value of the superhero")):
    cursor.execute("SELECT DISTINCT T2.colour FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.skin_colour_id = T2.id INNER JOIN hero_attribute AS T3 ON T1.id = T3.hero_id WHERE T3.attribute_value = ?", (attribute_value,))
    result = cursor.fetchall()
    if not result:
        return {"skin_colours": []}
    return {"skin_colours": [row[0] for row in result]}

# Endpoint to get the count of superheroes with a specific alignment and gender
@app.get("/v1/superhero/count_by_alignment_gender", operation_id="get_count_by_alignment_gender", summary="Retrieve the number of superheroes that match a specific alignment and gender. The alignment and gender parameters are used to filter the superheroes and calculate the count.")
async def get_count_by_alignment_gender(alignment: str = Query(..., description="Alignment of the superhero"), gender: str = Query(..., description="Gender of the superhero")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN alignment AS T2 ON T1.alignment_id = T2.id INNER JOIN gender AS T3 ON T1.gender_id = T3.id WHERE T2.alignment = ? AND T3.gender = ?", (alignment, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of superheroes with attribute values within a specific range
@app.get("/v1/superhero/names_by_attribute_value_range", operation_id="get_names_by_attribute_value_range", summary="Retrieve the names of superheroes whose attribute values fall within a specified range. The range is defined by a minimum and maximum value, which are used to filter the superheroes based on their attribute values.")
async def get_names_by_attribute_value_range(min_value: int = Query(..., description="Minimum attribute value"), max_value: int = Query(..., description="Maximum attribute value")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id WHERE T2.attribute_value BETWEEN ? AND ?", (min_value, max_value))
    result = cursor.fetchall()
    if not result:
        return {"superhero_names": []}
    return {"superhero_names": [row[0] for row in result]}

# Endpoint to get the race of superheroes based on hair colour and gender
@app.get("/v1/superhero/race_by_hair_colour_gender", operation_id="get_race_by_hair_colour_gender", summary="Retrieve the race of superheroes who have a specific hair colour and gender. This operation filters superheroes based on their hair colour and gender, and returns the race of those who match the criteria.")
async def get_race_by_hair_colour_gender(hair_colour: str = Query(..., description="Hair colour of the superhero"), gender: str = Query(..., description="Gender of the superhero")):
    cursor.execute("SELECT T3.race FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.hair_colour_id = T2.id INNER JOIN race AS T3 ON T1.race_id = T3.id INNER JOIN gender AS T4 ON T1.gender_id = T4.id WHERE T2.colour = ? AND T4.gender = ?", (hair_colour, gender))
    result = cursor.fetchall()
    if not result:
        return {"races": []}
    return {"races": [row[0] for row in result]}

# Endpoint to get the percentage of superheroes of a specific gender and alignment
@app.get("/v1/superhero/percentage_gender_alignment", operation_id="get_percentage_gender_alignment", summary="Retrieves the percentage of superheroes with a specific gender and alignment from the database. The calculation is based on the total count of superheroes with the given alignment and the count of those who also have the specified gender.")
async def get_percentage_gender_alignment(gender: str = Query(..., description="Gender of the superhero"), alignment: str = Query(..., description="Alignment of the superhero")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.gender = ? THEN T1.id ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM superhero AS T1 INNER JOIN alignment AS T2 ON T1.alignment_id = T2.id INNER JOIN gender AS T3 ON T1.gender_id = T3.id WHERE T2.alignment = ?", (gender, alignment))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in counts of superheroes with specific eye colours and weight
@app.get("/v1/superhero/eye_colour_weight_difference", operation_id="get_eye_colour_weight_difference", summary="Retrieve the difference in the number of superheroes with two specified eye colours and a given weight. The operation considers superheroes with the provided weight or those whose weight is not specified.")
async def get_eye_colour_weight_difference(eye_colour_id_1: int = Query(..., description="First eye colour ID"), eye_colour_id_2: int = Query(..., description="Second eye colour ID"), weight_kg: int = Query(..., description="Weight in kilograms")):
    cursor.execute("SELECT SUM(CASE WHEN T2.id = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.id = ? THEN 1 ELSE 0 END) FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T1.weight_kg = ? OR T1.weight_kg is NULL", (eye_colour_id_1, eye_colour_id_2, weight_kg))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the attribute value of a superhero based on superhero name and attribute name
@app.get("/v1/superhero/attribute_value_by_name", operation_id="get_attribute_value_by_name", summary="Retrieves the specific attribute value of a superhero, given the superhero's name and the attribute's name. This operation returns the value of the requested attribute for the specified superhero, obtained by joining the superhero, hero_attribute, and attribute tables.")
async def get_attribute_value_by_name(superhero_name: str = Query(..., description="Name of the superhero"), attribute_name: str = Query(..., description="Name of the attribute")):
    cursor.execute("SELECT T2.attribute_value FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id INNER JOIN attribute AS T3 ON T2.attribute_id = T3.id WHERE T1.superhero_name = ? AND T3.attribute_name = ?", (superhero_name, attribute_name))
    result = cursor.fetchone()
    if not result:
        return {"attribute_value": []}
    return {"attribute_value": result[0]}

# Endpoint to get the count of superheroes based on alignment and skin colour
@app.get("/v1/superhero/count_by_alignment_skin_colour", operation_id="get_count_by_alignment_skin_colour", summary="Retrieves the total count of superheroes sharing a specific alignment and skin colour. The alignment and skin colour are provided as input parameters, allowing for a targeted search of the superhero database.")
async def get_count_by_alignment_skin_colour(alignment: str = Query(..., description="Alignment of the superhero"), skin_colour: str = Query(..., description="Skin colour of the superhero")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN alignment AS T2 ON T1.alignment_id = T2.id INNER JOIN colour AS T3 ON T1.skin_colour_id = T3.id WHERE T2.alignment = ? AND T3.colour = ?", (alignment, skin_colour))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of superheroes based on publisher and gender
@app.get("/v1/superhero/count_by_publisher_gender", operation_id="get_count_by_publisher_gender", summary="Retrieves the total number of superheroes associated with a specific publisher and gender. The response is based on the provided publisher name and gender.")
async def get_count_by_publisher_gender(publisher_name: str = Query(..., description="Name of the publisher"), gender: str = Query(..., description="Gender of the superhero")):
    cursor.execute("SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN gender AS T3 ON T1.gender_id = T3.id WHERE T2.publisher_name = ? AND T3.gender = ?", (publisher_name, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of superheroes with a specific power
@app.get("/v1/superhero/names_by_power", operation_id="get_names_by_power", summary="Retrieve a sorted list of superhero names that possess a specific power. The power is identified by its name.")
async def get_names_by_power(power_name: str = Query(..., description="Name of the power")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T3.power_name = ? ORDER BY T1.superhero_name", (power_name,))
    result = cursor.fetchall()
    if not result:
        return {"superhero_names": []}
    return {"superhero_names": [row[0] for row in result]}

# Endpoint to get the gender of superheroes with a specific power
@app.get("/v1/superhero/gender_by_power", operation_id="get_gender_by_power", summary="Retrieves the gender distribution of superheroes possessing a specific power. The operation requires the name of the power as an input parameter to filter the results.")
async def get_gender_by_power(power_name: str = Query(..., description="Name of the power")):
    cursor.execute("SELECT T4.gender FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id INNER JOIN gender AS T4 ON T1.gender_id = T4.id WHERE T3.power_name = ?", (power_name,))
    result = cursor.fetchall()
    if not result:
        return {"genders": []}
    return {"genders": [row[0] for row in result]}

# Endpoint to get the heaviest superhero from a specific publisher
@app.get("/v1/superhero/heaviest_by_publisher", operation_id="get_heaviest_by_publisher", summary="Retrieves the name of the heaviest superhero associated with a given publisher. The operation filters superheroes by publisher name and sorts them by weight in descending order, returning the top result.")
async def get_heaviest_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id WHERE T2.publisher_name = ? ORDER BY T1.weight_kg DESC LIMIT 1", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"superhero_name": []}
    return {"superhero_name": result[0]}

# Endpoint to get the count of superheroes with a specific attribute and value
@app.get("/v1/superhero/count_superheroes_attribute_value", operation_id="get_count_superheroes", summary="Retrieves the total number of superheroes who possess a specific attribute with a given value. The attribute and its corresponding value are provided as input parameters.")
async def get_count_superheroes(attribute_name: str = Query(..., description="Name of the attribute"), attribute_value: int = Query(..., description="Value of the attribute")):
    cursor.execute("SELECT COUNT(T3.superhero_name) FROM hero_attribute AS T1 INNER JOIN attribute AS T2 ON T1.attribute_id = T2.id INNER JOIN superhero AS T3 ON T1.hero_id = T3.id WHERE T2.attribute_name = ? AND T1.attribute_value = ?", (attribute_name, attribute_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the attribute name of a superhero with the lowest attribute value
@app.get("/v1/superhero/attribute_name_lowest_value", operation_id="get_attribute_name_lowest_value", summary="Retrieves the name of the attribute with the lowest value for a specified superhero. The superhero is identified by their name, and the attribute with the lowest value is determined by comparing the attribute values associated with the superhero.")
async def get_attribute_name_lowest_value(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T3.attribute_name FROM superhero AS T1 INNER JOIN hero_attribute AS T2 ON T1.id = T2.hero_id INNER JOIN attribute AS T3 ON T2.attribute_id = T3.id WHERE T1.superhero_name = ? ORDER BY T2.attribute_value ASC LIMIT 1", (superhero_name,))
    result = cursor.fetchone()
    if not result:
        return {"attribute_name": []}
    return {"attribute_name": result[0]}

# Endpoint to get the eye colour of a specific superhero
@app.get("/v1/superhero/eye_colour", operation_id="get_eye_colour", summary="Retrieves the eye colour of a specific superhero by their name. The operation returns the colour value associated with the superhero's eye colour ID from the colour table.")
async def get_eye_colour(superhero_name: str = Query(..., description="Name of the superhero")):
    cursor.execute("SELECT T2.colour FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T1.superhero_name = ?", (superhero_name,))
    result = cursor.fetchone()
    if not result:
        return {"eye_colour": []}
    return {"eye_colour": result[0]}

# Endpoint to get the superhero name based on full name
@app.get("/v1/superhero/superhero_name_by_full_name", operation_id="get_superhero_name", summary="Retrieves the superhero name associated with the provided full name. The full name is used as a unique identifier to locate the corresponding superhero name in the database.")
async def get_superhero_name(full_name: str = Query(..., description="Full name of the superhero")):
    cursor.execute("SELECT superhero_name FROM superhero WHERE full_name = ?", (full_name,))
    result = cursor.fetchone()
    if not result:
        return {"superhero_name": []}
    return {"superhero_name": result[0]}

# Endpoint to get the percentage of superheroes of a specific gender from a specific publisher
@app.get("/v1/superhero/percentage_gender_publisher", operation_id="get_percentage_gender", summary="Retrieves the percentage of superheroes of a specific gender associated with a given publisher. The operation calculates this percentage by counting the number of superheroes of the specified gender from the provided publisher and dividing it by the total count of superheroes from that publisher.")
async def get_percentage_gender(gender: str = Query(..., description="Gender of the superhero"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.gender = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN gender AS T3 ON T1.gender_id = T3.id WHERE T2.publisher_name = ?", (gender, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of superheroes with a specific alignment from a specific publisher
@app.get("/v1/superhero/percentage_alignment_publisher", operation_id="get_percentage_alignment", summary="Retrieves the percentage of superheroes with a specified alignment that belong to a given publisher. The alignment and publisher name are provided as input parameters.")
async def get_percentage_alignment(alignment: str = Query(..., description="Alignment of the superhero"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.alignment = ? THEN T1.id ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM superhero AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.id INNER JOIN alignment AS T3 ON T1.alignment_id = T3.id WHERE T2.publisher_name = ?", (alignment, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of superheroes with a full name starting with a specific prefix
@app.get("/v1/superhero/count_superheroes_by_full_name_prefix", operation_id="get_count_superheroes_by_prefix", summary="Retrieves the total number of superheroes whose full names begin with a specified prefix. The prefix is used to filter the superheroes and count those that match.")
async def get_count_superheroes_by_prefix(full_name_prefix: str = Query(..., description="Prefix of the full name")):
    cursor.execute("SELECT COUNT(id) FROM superhero WHERE full_name LIKE ?", (full_name_prefix + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the hero ID with the minimum attribute value
@app.get("/v1/superhero/hero_id_min_attribute_value", operation_id="get_hero_id_min_attribute_value", summary="Get the hero ID with the minimum attribute value")
async def get_hero_id_min_attribute_value():
    cursor.execute("SELECT hero_id FROM hero_attribute WHERE attribute_value = ( SELECT MIN(attribute_value) FROM hero_attribute )")
    result = cursor.fetchone()
    if not result:
        return {"hero_id": []}
    return {"hero_id": result[0]}

# Endpoint to get the full name of a superhero based on their superhero name
@app.get("/v1/superhero/full_name_by_superhero_name", operation_id="get_full_name_by_superhero_name", summary="Retrieves the full name of a superhero based on their superhero name. The operation requires the superhero's name as input and returns the corresponding full name from the superhero database.")
async def get_full_name_by_superhero_name(superhero_name: str = Query(..., description="Superhero name")):
    cursor.execute("SELECT full_name FROM superhero WHERE superhero_name = ?", (superhero_name,))
    result = cursor.fetchone()
    if not result:
        return {"full_name": []}
    return {"full_name": result[0]}

# Endpoint to get the full names of superheroes based on weight and eye colour
@app.get("/v1/superhero/full_name_by_weight_and_eye_colour", operation_id="get_full_name_by_weight_and_eye_colour", summary="Retrieve the full names of superheroes who weigh less than the specified weight and have the given eye colour. The operation filters superheroes based on their weight and eye colour, returning a list of their full names.")
async def get_full_name_by_weight_and_eye_colour(weight_kg: int = Query(..., description="Weight in kilograms"), eye_colour: str = Query(..., description="Eye colour")):
    cursor.execute("SELECT T1.full_name FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T1.weight_kg < ? AND T2.colour = ?", (weight_kg, eye_colour))
    result = cursor.fetchall()
    if not result:
        return {"full_names": []}
    return {"full_names": [row[0] for row in result]}

# Endpoint to get the weight and race of a superhero based on their ID
@app.get("/v1/superhero/weight_and_race_by_id", operation_id="get_weight_and_race_by_id", summary="Retrieves the weight in kilograms and the race of a specific superhero, identified by their unique ID. The superhero's race is determined by referencing the race table using the superhero's race ID.")
async def get_weight_and_race_by_id(hero_id: int = Query(..., description="Superhero ID")):
    cursor.execute("SELECT T1.weight_kg, T2.race FROM superhero AS T1 INNER JOIN race AS T2 ON T1.race_id = T2.id WHERE T1.id = ?", (hero_id,))
    result = cursor.fetchone()
    if not result:
        return {"weight_kg": [], "race": []}
    return {"weight_kg": result[0], "race": result[1]}

# Endpoint to get the average height of superheroes based on their alignment
@app.get("/v1/superhero/average_height_by_alignment", operation_id="get_average_height_by_alignment", summary="Retrieves the average height of superheroes based on their alignment. The alignment parameter is used to filter the superheroes and calculate the average height in centimeters.")
async def get_average_height_by_alignment(alignment: str = Query(..., description="Alignment of the superhero")):
    cursor.execute("SELECT AVG(T1.height_cm) FROM superhero AS T1 INNER JOIN alignment AS T2 ON T1.alignment_id = T2.id WHERE T2.alignment = ?", (alignment,))
    result = cursor.fetchone()
    if not result:
        return {"average_height": []}
    return {"average_height": result[0]}

# Endpoint to get the hero IDs based on a specific superpower
@app.get("/v1/superhero/hero_ids_by_superpower", operation_id="get_hero_ids_by_superpower", summary="Retrieve the unique identifiers of superheroes who exhibit a particular superpower. The operation filters the superheroes based on the provided superpower name and returns their corresponding hero IDs.")
async def get_hero_ids_by_superpower(power_name: str = Query(..., description="Superpower name")):
    cursor.execute("SELECT T1.hero_id FROM hero_power AS T1 INNER JOIN superpower AS T2 ON T1.power_id = T2.id WHERE T2.power_name = ?", (power_name,))
    result = cursor.fetchall()
    if not result:
        return {"hero_ids": []}
    return {"hero_ids": [row[0] for row in result]}

# Endpoint to get the power names of superheroes whose height is above a certain percentage of the average height
@app.get("/v1/superhero/power_names_by_height_percentage", operation_id="get_power_names_by_height_percentage", summary="Retrieves the names of powers possessed by superheroes who are taller than 80% of the average height. This operation calculates the average height of all superheroes and identifies those who exceed this threshold by 20%. It then returns the names of powers associated with these superheroes.")
async def get_power_names_by_height_percentage():
    cursor.execute("SELECT T3.power_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T1.height_cm * 100 > ( SELECT AVG(height_cm) FROM superhero ) * 80")
    result = cursor.fetchall()
    if not result:
        return {"power_names": []}
    return {"power_names": [row[0] for row in result]}

api_calls = [
    "/v1/superhero/power_names_by_superhero?superhero_name=3-D%20Man",
    "/v1/superhero/count_heroes_by_power?power_name=Super%20Strength",
    "/v1/superhero/count_superheroes_by_power_and_height?power_name=Super%20Strength&height_cm=200",
    "/v1/superhero/full_names_by_power_count?power_count=15",
    "/v1/superhero/count_superheroes_by_eye_colour?eye_colour=Blue",
    "/v1/superhero/skin_colour_by_superhero?superhero_name=Apocalypse",
    "/v1/superhero/count_superheroes_by_power_and_eye_colour?power_name=Agility&eye_colour=Blue",
    "/v1/superhero/superhero_names_by_eye_and_hair_colour?eye_colour=Blue&hair_colour=Blond",
    "/v1/superhero/count_superheroes_by_publisher?publisher_name=Marvel%20Comics",
    "/v1/superhero/superheroes_ranked_by_height?publisher_name=Marvel%20Comics",
    "/v1/superhero/publisher_name_by_superhero?superhero_name=Sauron",
    "/v1/superhero/eye_color_popularity_rank?publisher_name=Marvel%20Comics",
    "/v1/superhero/average_height_by_publisher?publisher_name=Marvel%20Comics",
    "/v1/superhero/superheroes_by_power_and_publisher?power_name=Super%20Strength&publisher_name=Marvel%20Comics",
    "/v1/superhero/publisher_name_by_highest_attribute?attribute_name=Speed",
    "/v1/superhero/count_superheroes_by_publisher_and_eye_color?publisher_name=Marvel%20Comics&eye_color=Gold",
    "/v1/superhero/count_superheroes_by_hair_color?hair_color=Blond",
    "/v1/superhero/superhero_by_highest_attribute?attribute_name=Intelligence",
    "/v1/superhero/get_race_by_name?superhero_name=Copycat",
    "/v1/superhero/get_superheroes_by_attribute_value?attribute_name=Durability&attribute_value=50",
    "/v1/superhero/get_superheroes_by_power?power_name=Death%20Touch",
    "/v1/superhero/count_superheroes_by_attribute_and_gender?attribute_name=Strength&attribute_value=100&gender=Female",
    "/v1/superhero/get_superhero_with_most_powers",
    "/v1/superhero/count_superheroes_by_race?race=Vampire",
    "/v1/superhero/percentage_superheroes_by_alignment_and_publisher?publisher_name=Marvel%20Comics&alignment=Bad",
    "/v1/superhero/difference_superheroes_between_publishers?publisher_name_1=Marvel%20Comics&publisher_name_2=DC%20Comics",
    "/v1/superhero/get_publisher_id_by_name?publisher_name=Star%20Trek",
    "/v1/superhero/average_attribute_value",
    "/v1/superhero/count_no_full_name",
    "/v1/superhero/eye_colour_by_id?id=75",
    "/v1/superhero/avg_weight_by_gender?gender=Female",
    "/v1/superhero/male_powers_with_limit?gender=Male&limit=5",
    "/v1/superhero/names_by_race?race=Alien",
    "/v1/superhero/names_by_height_and_eye_colour?min_height=170&max_height=190&eye_colour=No%20Colour",
    "/v1/superhero/powers_by_hero_id?hero_id=56",
    "/v1/superhero/full_names_by_race?race=Demi-God",
    "/v1/superhero/count_by_alignment?alignment=Bad",
    "/v1/superhero/race_by_weight?weight_kg=169",
    "/v1/superhero/hair_colour_by_height_race?height_cm=185&race=Human",
    "/v1/superhero/eye_colour_heaviest",
    "/v1/superhero/publisher_percentage_by_height?publisher_name=Marvel%20Comics&min_height_cm=150&max_height_cm=180",
    "/v1/superhero/superheroes_by_gender_weight?gender=Male&weight_percentage=79",
    "/v1/superhero/most_common_superpower",
    "/v1/superhero/attribute_values_by_name?superhero_name=Abomination",
    "/v1/superhero/superpowers_by_hero_id?hero_id=1",
    "/v1/superhero/highest_attribute_value?attribute_name=Strength",
    "/v1/superhero/ratio_skin_colour?skin_colour_id=1",
    "/v1/superhero/top_attribute_superhero?publisher_name=Dark%20Horse%20Comics&attribute_name=Durability",
    "/v1/superhero/eye_colour_by_full_name?full_name=Abraham%20Sapien",
    "/v1/superhero/colour_ids_by_publisher_gender?publisher_name=Dark%20Horse%20Comics&gender=Female",
    "/v1/superhero/matching_colour_ids",
    "/v1/superhero/percentage_female_skin_colour?skin_colour=Blue&gender=Female",
    "/v1/superhero/race_by_full_name?full_name=Charles%20Chandler",
    "/v1/superhero/gender_by_name?superhero_name=Agent%2013",
    "/v1/superhero/power_count_by_name?superhero_name=Amazo",
    "/v1/superhero/powers_by_full_name?full_name=Hunter%20Zolomon",
    "/v1/superhero/height_by_eye_colour?eye_colour=Amber",
    "/v1/superhero/superheroes_by_eye_and_hair_colour?colour=Black",
    "/v1/superhero/eye_colour_by_skin_colour?skin_colour=Gold",
    "/v1/superhero/superheroes_by_alignment?alignment=Neutral",
    "/v1/superhero/count_max_attribute?attribute_name=Strength",
    "/v1/superhero/get_race_alignment_by_name?superhero_name=Cameron%20Hicks",
    "/v1/superhero/get_publisher_gender_percentage?publisher_name=Marvel%20Comics&gender=Female",
    "/v1/superhero/get_average_weight_by_race?race=Alien",
    "/v1/superhero/get_weight_difference?full_name1=Emil%20Blonsky&full_name2=Charles%20Chandler",
    "/v1/superhero/get_average_height",
    "/v1/superhero/get_count_by_race_gender?race_id=21&gender_id=1",
    "/v1/superhero/get_top_superhero_by_attribute?attribute_name=Speed",
    "/v1/superhero/get_attributes_by_name?superhero_name=3-D%20Man",
    "/v1/superhero/publisher_names_by_superhero_names?superhero_name1=Hawkman&superhero_name2=Karate%20Kid&superhero_name3=Speedy",
    "/v1/superhero/count_superheroes_by_publisher_id?publisher_id=1",
    "/v1/superhero/percentage_superheroes_by_eye_colour?eye_colour=Blue",
    "/v1/superhero/ratio_male_to_female_superheroes?male_gender=Male&female_gender=Female",
    "/v1/superhero/tallest_superhero",
    "/v1/superhero/superpower_id_by_name?power_name=Cryokinesis",
    "/v1/superhero/superhero_name_by_id?superhero_id=294",
    "/v1/superhero/distinct_full_names_by_weight?weight_kg=0",
    "/v1/superhero/distinct_races_by_weight_height?weight_kg=108&height_cm=188",
    "/v1/superhero/publisher_name_by_id?superhero_id=38",
    "/v1/superhero/race_by_highest_attribute_value",
    "/v1/superhero/alignment_power_names_by_name?superhero_name=Atom%20IV",
    "/v1/superhero/names_by_eye_colour?eye_colour=Blue",
    "/v1/superhero/avg_attribute_value_by_alignment?alignment=Neutral",
    "/v1/superhero/distinct_skin_colours_by_attribute_value?attribute_value=100",
    "/v1/superhero/count_by_alignment_gender?alignment=Good&gender=Female",
    "/v1/superhero/names_by_attribute_value_range?min_value=75&max_value=80",
    "/v1/superhero/race_by_hair_colour_gender?hair_colour=Blue&gender=Male",
    "/v1/superhero/percentage_gender_alignment?gender=Female&alignment=Bad",
    "/v1/superhero/eye_colour_weight_difference?eye_colour_id_1=7&eye_colour_id_2=1&weight_kg=0",
    "/v1/superhero/attribute_value_by_name?superhero_name=Hulk&attribute_name=Strength",
    "/v1/superhero/count_by_alignment_skin_colour?alignment=Bad&skin_colour=Green",
    "/v1/superhero/count_by_publisher_gender?publisher_name=Marvel%20Comics&gender=Female",
    "/v1/superhero/names_by_power?power_name=Wind%20Control",
    "/v1/superhero/gender_by_power?power_name=Phoenix%20Force",
    "/v1/superhero/heaviest_by_publisher?publisher_name=DC%20Comics",
    "/v1/superhero/count_superheroes_attribute_value?attribute_name=Speed&attribute_value=100",
    "/v1/superhero/attribute_name_lowest_value?superhero_name=Black%20Panther",
    "/v1/superhero/eye_colour?superhero_name=Abomination",
    "/v1/superhero/superhero_name_by_full_name?full_name=Charles%20Chandler",
    "/v1/superhero/percentage_gender_publisher?gender=Female&publisher_name=George%20Lucas",
    "/v1/superhero/percentage_alignment_publisher?alignment=Good&publisher_name=Marvel%20Comics",
    "/v1/superhero/count_superheroes_by_full_name_prefix?full_name_prefix=John",
    "/v1/superhero/hero_id_min_attribute_value",
    "/v1/superhero/full_name_by_superhero_name?superhero_name=Alien",
    "/v1/superhero/full_name_by_weight_and_eye_colour?weight_kg=100&eye_colour=Brown",
    "/v1/superhero/weight_and_race_by_id?hero_id=40",
    "/v1/superhero/average_height_by_alignment?alignment=Neutral",
    "/v1/superhero/hero_ids_by_superpower?power_name=Intelligence",
    "/v1/superhero/power_names_by_height_percentage"
]
