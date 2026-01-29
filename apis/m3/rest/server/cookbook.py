from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/cookbook/cookbook.sqlite')
cursor = conn.cursor()

# Endpoint to get the recipe title with the highest total fat
@app.get("/v1/cookbook/recipe_highest_total_fat", operation_id="get_recipe_highest_total_fat", summary="Retrieves the title of the recipe with the highest total fat content. The operation allows you to limit the number of results returned. This endpoint is useful for identifying the recipe with the most fat content, which can be helpful for dietary planning or nutritional analysis.")
async def get_recipe_highest_total_fat(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id ORDER BY T2.total_fat DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the difference between total fat and saturated fat for a given recipe title
@app.get("/v1/cookbook/fat_difference_by_title", operation_id="get_fat_difference_by_title", summary="Retrieves the difference between total fat and saturated fat for a specific recipe, based on the provided recipe title.")
async def get_fat_difference_by_title(title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT T2.total_fat - T2.sat_fat FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get recipe titles with sodium content less than a specified value
@app.get("/v1/cookbook/recipes_low_sodium", operation_id="get_recipes_low_sodium", summary="Retrieves the titles of recipes that have a sodium content lower than the specified maximum value. This operation allows users to filter recipes based on their sodium content, providing a healthier selection of options.")
async def get_recipes_low_sodium(max_sodium: int = Query(..., description="Maximum sodium content")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T2.sodium < ?", (max_sodium,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get recipe titles with iron content greater than a specified value
@app.get("/v1/cookbook/recipes_high_iron", operation_id="get_recipes_high_iron", summary="Retrieves the titles of recipes that contain iron in quantities exceeding the specified minimum value. This operation is useful for users seeking recipes with high iron content.")
async def get_recipes_high_iron(min_iron: int = Query(..., description="Minimum iron content")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T2.iron > ?", (min_iron,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to compare vitamin C content between two recipes
@app.get("/v1/cookbook/compare_vitamin_c", operation_id="compare_vitamin_c", summary="This operation compares the vitamin C content of two specified recipes and returns the recipe with the higher vitamin C content. The comparison is based on the nutrition data associated with each recipe.")
async def compare_vitamin_c(title1: str = Query(..., description="Title of the first recipe"), title2: str = Query(..., description="Title of the second recipe")):
    cursor.execute("SELECT DISTINCT CASE WHEN CASE WHEN T2.title = ? THEN T1.vitamin_c END > CASE WHEN T2.title = ? THEN T1.vitamin_c END THEN ? ELSE ? END AS 'vitamin_c is higher' FROM Nutrition T1 INNER JOIN Recipe T2 ON T2.recipe_id = T1.recipe_id", (title1, title2, title1, title2))
    result = cursor.fetchone()
    if not result:
        return {"comparison": []}
    return {"comparison": result[0]}

# Endpoint to get the recipe title with the highest calories for recipes with preparation time greater than a specified value
@app.get("/v1/cookbook/highest_calories_prep_time", operation_id="get_highest_calories_prep_time", summary="Retrieves the title of the recipe with the highest calorie count from among those recipes that require more than the specified minimum preparation time. The number of results returned can be limited by the user.")
async def get_highest_calories_prep_time(min_prep_min: int = Query(..., description="Minimum preparation time in minutes"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.prep_min > ? ORDER BY T2.calories DESC LIMIT ?", (min_prep_min, limit))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the calories for a given recipe title
@app.get("/v1/cookbook/calories_by_title", operation_id="get_calories_by_title", summary="Retrieves the total calorie count for a specific recipe, identified by its title. The operation returns the calorie information associated with the recipe, enabling users to understand its nutritional value.")
async def get_calories_by_title(title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT T2.calories FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"calories": []}
    return {"calories": result[0]}

# Endpoint to get the optional status of an ingredient in a given recipe
@app.get("/v1/cookbook/ingredient_optional_status", operation_id="get_ingredient_optional_status", summary="Retrieve the optional status of a specific ingredient within a given recipe. This operation requires the title of the recipe and the name of the ingredient as input parameters. The optional status indicates whether the ingredient is optional or required in the recipe.")
async def get_ingredient_optional_status(title: str = Query(..., description="Title of the recipe"), ingredient_name: str = Query(..., description="Name of the ingredient")):
    cursor.execute("SELECT T2.optional FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T1.title = ? AND T3.name = ?", (title, ingredient_name))
    result = cursor.fetchone()
    if not result:
        return {"optional": []}
    return {"optional": result[0]}

# Endpoint to get the count of ingredients with equal maximum and minimum quantities in a given recipe
@app.get("/v1/cookbook/count_equal_quantities", operation_id="get_count_equal_quantities", summary="Retrieves the count of ingredients in a specific recipe that have the same maximum and minimum quantities. The recipe is identified by its title.")
async def get_count_equal_quantities(title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT COUNT(*) FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.title = ? AND T2.max_qty = T2.min_qty", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of ingredients with no preparation in a given recipe
@app.get("/v1/cookbook/ingredients_no_preparation", operation_id="get_ingredients_no_preparation", summary="Retrieves the names of ingredients that do not require any preparation for a specified recipe. The operation filters ingredients based on the provided recipe title and returns a list of ingredient names that have no associated preparation steps.")
async def get_ingredients_no_preparation(title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT T3.name FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T1.title = ? AND T2.preparation IS NULL", (title,))
    result = cursor.fetchall()
    if not result:
        return {"ingredients": []}
    return {"ingredients": [row[0] for row in result]}

# Endpoint to get the count of ingredients based on the ingredient name
@app.get("/v1/cookbook/ingredient_count_by_name", operation_id="get_ingredient_count_by_name", summary="Retrieves the total count of a specific ingredient, as determined by its name, from the ingredient database. This operation considers the ingredient's presence in the quantity database, which may indicate its usage in recipes.")
async def get_ingredient_count_by_name(ingredient_name: str = Query(..., description="Name of the ingredient")):
    cursor.execute("SELECT COUNT(*) FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id WHERE T1.name = ?", (ingredient_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the minimum quantity of an ingredient in a specific recipe
@app.get("/v1/cookbook/min_quantity_ingredient_recipe", operation_id="get_min_quantity_ingredient_recipe", summary="Retrieves the minimum quantity of a specified ingredient used in a particular recipe. The operation requires the title of the recipe and the name of the ingredient as input parameters.")
async def get_min_quantity_ingredient_recipe(recipe_title: str = Query(..., description="Title of the recipe"), ingredient_name: str = Query(..., description="Name of the ingredient")):
    cursor.execute("SELECT T2.min_qty FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T1.title = ? AND T3.name = ?", (recipe_title, ingredient_name))
    result = cursor.fetchone()
    if not result:
        return {"min_qty": []}
    return {"min_qty": result[0]}

# Endpoint to get the calories from fat for a specific recipe
@app.get("/v1/cookbook/calories_from_fat_recipe", operation_id="get_calories_from_fat_recipe", summary="Retrieves the total calories from fat for a specific recipe. The calculation is based on the recipe's total calories and the percentage of calories derived from fat. The recipe is identified by its title.")
async def get_calories_from_fat_recipe(recipe_title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT T2.calories * T2.pcnt_cal_fat FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.title = ?", (recipe_title,))
    result = cursor.fetchone()
    if not result:
        return {"calories_from_fat": []}
    return {"calories_from_fat": result[0]}

# Endpoint to get the average calories of recipes from a specific source
@app.get("/v1/cookbook/average_calories_by_source", operation_id="get_average_calories_by_source", summary="Retrieves the average caloric content of recipes sourced from a specified provider. The source parameter is used to filter the recipes and calculate the average calories.")
async def get_average_calories_by_source(source: str = Query(..., description="Source of the recipe")):
    cursor.execute("SELECT AVG(T2.calories) FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.source = ?", (source,))
    result = cursor.fetchone()
    if not result:
        return {"average_calories": []}
    return {"average_calories": result[0]}

# Endpoint to get the count of ingredients based on name, unit, and recipe ID
@app.get("/v1/cookbook/ingredient_count_by_name_unit_recipe", operation_id="get_ingredient_count_by_name_unit_recipe", summary="Retrieves the total count of a specific ingredient, identified by its name and unit, used in a particular recipe. The ingredient name, unit, and recipe ID are required as input parameters.")
async def get_ingredient_count_by_name_unit_recipe(ingredient_name: str = Query(..., description="Name of the ingredient"), unit: str = Query(..., description="Unit of the ingredient"), recipe_id: int = Query(..., description="Recipe ID")):
    cursor.execute("SELECT COUNT(*) FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id WHERE T1.name = ? AND T2.unit = ? AND T2.recipe_id = ?", (ingredient_name, unit, recipe_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of optional ingredients in a specific recipe
@app.get("/v1/cookbook/optional_ingredients_by_recipe", operation_id="get_optional_ingredients_by_recipe", summary="Retrieves the names of optional ingredients associated with a specific recipe. The operation requires a recipe ID and an optional status (TRUE or FALSE) to filter the ingredients accordingly.")
async def get_optional_ingredients_by_recipe(recipe_id: int = Query(..., description="Recipe ID"), optional: str = Query(..., description="Optional status (TRUE or FALSE)")):
    cursor.execute("SELECT T1.name FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id WHERE T2.recipe_id = ? AND T2.optional = ?", (recipe_id, optional))
    result = cursor.fetchall()
    if not result:
        return {"ingredients": []}
    return {"ingredients": [row[0] for row in result]}

# Endpoint to get the titles of recipes with a specific ingredient and equal max and min quantities
@app.get("/v1/cookbook/recipe_titles_by_ingredient_equal_quantities", operation_id="get_recipe_titles_by_ingredient_equal_quantities", summary="Retrieves the titles of recipes that contain a specified ingredient with equal maximum and minimum quantities. The ingredient name is provided as an input parameter.")
async def get_recipe_titles_by_ingredient_equal_quantities(ingredient_name: str = Query(..., description="Name of the ingredient")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T3.name = ? AND T2.max_qty = T2.min_qty", (ingredient_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the most frequently used ingredient
@app.get("/v1/cookbook/most_frequent_ingredient", operation_id="get_most_frequent_ingredient", summary="Retrieves the ingredient that is most frequently used in recipes. This operation calculates the frequency of ingredient usage by joining the Ingredient and Quantity tables, grouping by ingredient name, and ordering the results in descending order based on the count of each ingredient. The top result is then returned.")
async def get_most_frequent_ingredient():
    cursor.execute("SELECT T1.name FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id GROUP BY T1.name ORDER BY COUNT(T1.name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ingredient": []}
    return {"ingredient": result[0]}

# Endpoint to get the preparation of an ingredient in a specific recipe
@app.get("/v1/cookbook/ingredient_preparation", operation_id="get_ingredient_preparation", summary="Retrieves the preparation steps for a specific ingredient used in a given recipe. The operation requires the title of the recipe and the name of the ingredient as input parameters. The returned data includes the preparation instructions for the specified ingredient in the context of the provided recipe.")
async def get_ingredient_preparation(recipe_title: str = Query(..., description="Title of the recipe"), ingredient_name: str = Query(..., description="Name of the ingredient")):
    cursor.execute("SELECT T2.preparation FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T1.title = ? AND T3.name = ?", (recipe_title, ingredient_name))
    result = cursor.fetchall()
    if not result:
        return {"preparation": []}
    return {"preparation": [row[0] for row in result]}

# Endpoint to get the count of recipes with specific ingredient and unit
@app.get("/v1/cookbook/recipe_count_ingredient_unit", operation_id="get_recipe_count_ingredient_unit", summary="Retrieves the total number of recipes that contain a specific ingredient in a given unit of measurement. The operation filters recipes by their title, ingredient name, and unit of measurement.")
async def get_recipe_count_ingredient_unit(recipe_title: str = Query(..., description="Title of the recipe"), ingredient_name: str = Query(..., description="Name of the ingredient"), unit: str = Query(..., description="Unit of measurement")):
    cursor.execute("SELECT COUNT(*) FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T1.title = ? AND T3.name = ? AND T2.unit = ?", (recipe_title, ingredient_name, unit))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the recipe with the highest vitamin C content
@app.get("/v1/cookbook/recipe_highest_vitamin_c", operation_id="get_recipe_highest_vitamin_c", summary="Retrieves the recipe with the highest vitamin C content from the database. The operation returns the title of the recipe that has the most vitamin C, based on the nutrition data associated with each recipe.")
async def get_recipe_highest_vitamin_c():
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id ORDER BY T2.vitamin_c DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the vitamin A content of a specific recipe
@app.get("/v1/cookbook/recipe_vitamin_a", operation_id="get_recipe_vitamin_a", summary="Retrieves the vitamin A content of a specific recipe by its title. The operation returns the vitamin A value associated with the provided recipe title.")
async def get_recipe_vitamin_a(recipe_title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT T2.vitamin_a FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.title = ?", (recipe_title,))
    result = cursor.fetchone()
    if not result:
        return {"vitamin_a": []}
    return {"vitamin_a": result[0]}

# Endpoint to get the most common recipe title
@app.get("/v1/cookbook/most_common_recipe", operation_id="get_most_common_recipe", summary="Retrieves the title of the recipe that appears most frequently in the database. This operation considers the number of times a recipe is used in various quantities, as indicated by the join with the Quantity table. The result is ordered by the count of recipe titles in descending order, and only the top result is returned.")
async def get_most_common_recipe():
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id GROUP BY T1.title ORDER BY COUNT(title) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the percentage of sodium content between two recipes
@app.get("/v1/cookbook/sodium_percentage", operation_id="get_sodium_percentage", summary="Retrieves the percentage of sodium content in the first recipe compared to the second recipe. The operation calculates the ratio of the total sodium content in the first recipe to the total sodium content in the second recipe, expressed as a percentage.")
async def get_sodium_percentage(recipe_title_1: str = Query(..., description="Title of the first recipe"), recipe_title_2: str = Query(..., description="Title of the second recipe")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.title = ? THEN T2.sodium ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN T1.title = ? THEN T2.sodium ELSE 0 END) FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id", (recipe_title_1, recipe_title_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average calories of recipes containing a specific ingredient
@app.get("/v1/cookbook/average_calories_ingredient", operation_id="get_average_calories_ingredient", summary="Retrieves the average calorie count of all recipes that include a specified ingredient. The ingredient is identified by its name, which is provided as an input parameter.")
async def get_average_calories_ingredient(ingredient_name: str = Query(..., description="Name of the ingredient")):
    cursor.execute("SELECT AVG(T3.calories) FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T2.ingredient_id = T1.ingredient_id INNER JOIN Nutrition AS T3 ON T3.recipe_id = T2.recipe_id WHERE T1.name = ?", (ingredient_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_calories": []}
    return {"average_calories": result[0]}

# Endpoint to get the count of recipes with a specific title
@app.get("/v1/cookbook/recipe_count", operation_id="get_recipe_count", summary="Retrieves the total number of recipes that match the provided title. This operation considers recipes that have associated quantity data. The title parameter is used to filter the recipes.")
async def get_recipe_count(recipe_title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT COUNT(*) FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.title = ?", (recipe_title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of ingredients with a specific preparation
@app.get("/v1/cookbook/ingredient_names_preparation", operation_id="get_ingredient_names_preparation", summary="Retrieves the names of ingredients that undergo a specified preparation process. The preparation parameter is used to filter the ingredients based on their preparation method.")
async def get_ingredient_names_preparation(preparation: str = Query(..., description="Preparation of the ingredient")):
    cursor.execute("SELECT T1.name FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id WHERE T2.preparation = ?", (preparation,))
    result = cursor.fetchall()
    if not result:
        return {"ingredient_names": []}
    return {"ingredient_names": [row[0] for row in result]}

# Endpoint to get the count of recipes with vitamin A content greater than a specified value
@app.get("/v1/cookbook/recipe_count_vitamin_a", operation_id="get_recipe_count_vitamin_a", summary="Retrieves the number of recipes that contain more than the specified amount of Vitamin A. The threshold for Vitamin A content is provided as an input parameter.")
async def get_recipe_count_vitamin_a(vitamin_a: int = Query(..., description="Vitamin A content threshold")):
    cursor.execute("SELECT COUNT(*) FROM Nutrition AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.vitamin_a > ?", (vitamin_a,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top recipes by vitamin C content
@app.get("/v1/cookbook/top_recipes_by_vitamin_c", operation_id="get_top_recipes_by_vitamin_c", summary="Retrieves a list of top recipes, ordered by their vitamin C content in descending order. The number of recipes returned can be specified using the input parameter.")
async def get_top_recipes_by_vitamin_c(limit: int = Query(..., description="Number of top recipes to return")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id ORDER BY T2.vitamin_c DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"recipes": []}
    return {"recipes": [row[0] for row in result]}

# Endpoint to get the least used ingredient
@app.get("/v1/cookbook/least_used_ingredient", operation_id="get_least_used_ingredient", summary="Retrieves the least frequently used ingredient(s) based on the count of their usage. The number of ingredients returned can be specified using the 'limit' input parameter.")
async def get_least_used_ingredient(limit: int = Query(..., description="Number of least used ingredients to return")):
    cursor.execute("SELECT T1.name FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id GROUP BY T2.ingredient_id ORDER BY COUNT(T2.ingredient_id) ASC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"ingredients": []}
    return {"ingredients": [row[0] for row in result]}

# Endpoint to get the count of recipes with specific ingredient category and title
@app.get("/v1/cookbook/count_recipes_by_category_and_title", operation_id="get_count_recipes_by_category_and_title", summary="Retrieves the total number of recipes that contain a specific ingredient category and have a particular title. The operation filters recipes based on the provided ingredient category and title, then returns the count of matching recipes.")
async def get_count_recipes_by_category_and_title(category: str = Query(..., description="Category of the ingredient"), title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT COUNT(*) FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T3.category = ? AND T1.title = ?", (category, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get ingredients for a specific recipe
@app.get("/v1/cookbook/ingredients_by_recipe_title", operation_id="get_ingredients_by_recipe_title", summary="Retrieves the names of all ingredients used in a specific recipe, as identified by its title. The input parameter is the title of the recipe.")
async def get_ingredients_by_recipe_title(title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT T3.name FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"ingredients": []}
    return {"ingredients": [row[0] for row in result]}

# Endpoint to get optional ingredients for a specific recipe
@app.get("/v1/cookbook/optional_ingredients_by_recipe_title", operation_id="get_optional_ingredients_by_recipe_title", summary="Retrieves the names of optional ingredients associated with a specific recipe, based on the provided recipe title and optional status.")
async def get_optional_ingredients_by_recipe_title(title: str = Query(..., description="Title of the recipe"), optional: str = Query(..., description="Optional status of the ingredient (TRUE or FALSE)")):
    cursor.execute("SELECT T3.name FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T1.title = ? AND T2.optional = ?", (title, optional))
    result = cursor.fetchall()
    if not result:
        return {"ingredients": []}
    return {"ingredients": [row[0] for row in result]}

# Endpoint to get the recipe with the highest preparation time and alcohol content above a threshold
@app.get("/v1/cookbook/recipe_by_alcohol_and_prep_time", operation_id="get_recipe_by_alcohol_and_prep_time", summary="Retrieves the recipe(s) with the longest preparation time and alcohol content exceeding a specified threshold. The number of recipes returned can be limited.")
async def get_recipe_by_alcohol_and_prep_time(alcohol_threshold: float = Query(..., description="Alcohol content threshold"), limit: int = Query(..., description="Number of recipes to return")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T2.alcohol > ? ORDER BY T1.prep_min DESC LIMIT ?", (alcohol_threshold, limit))
    result = cursor.fetchall()
    if not result:
        return {"recipes": []}
    return {"recipes": [row[0] for row in result]}

# Endpoint to get the count of recipes with the highest difference between total fat and saturated fat
@app.get("/v1/cookbook/count_recipes_by_fat_difference", operation_id="get_count_recipes_by_fat_difference", summary="Retrieves the count of recipes that have the highest difference between total fat and saturated fat, up to a specified limit.")
async def get_count_recipes_by_fat_difference(limit: int = Query(..., description="Number of recipes to return")):
    cursor.execute("SELECT COUNT(T1.title) FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id ORDER BY T2.total_fat - T2.sat_fat DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the recipe with the highest calories from a specific source
@app.get("/v1/cookbook/recipe_by_source_and_calories", operation_id="get_recipe_by_source_and_calories", summary="Retrieves the recipe(s) with the highest calorie count from a specified source. The source parameter identifies the origin of the recipe, while the limit parameter determines the number of recipes to return. The operation returns the title of the recipe(s).")
async def get_recipe_by_source_and_calories(source: str = Query(..., description="Source of the recipe"), limit: int = Query(..., description="Number of recipes to return")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.source = ? ORDER BY T2.calories DESC LIMIT ?", (source, limit))
    result = cursor.fetchall()
    if not result:
        return {"recipes": []}
    return {"recipes": [row[0] for row in result]}

# Endpoint to get the recipe with the most ingredients and its total preparation time
@app.get("/v1/cookbook/recipe_by_most_ingredients", operation_id="get_recipe_by_most_ingredients", summary="Retrieves the recipe(s) with the highest number of ingredients, along with their total preparation time. The number of recipes returned can be limited by specifying the desired quantity.")
async def get_recipe_by_most_ingredients(limit: int = Query(..., description="Number of recipes to return")):
    cursor.execute("SELECT T2.recipe_id, T1.prep_min + T1.cook_min + T1.stnd_min FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id GROUP BY T2.recipe_id ORDER BY COUNT(T2.ingredient_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"recipes": []}
    return {"recipes": [{"recipe_id": row[0], "total_time": row[1]} for row in result]}

# Endpoint to get the most used ingredient and its usage percentage
@app.get("/v1/cookbook/most_used_ingredient_percentage", operation_id="get_most_used_ingredient_percentage", summary="Retrieves the top ingredient(s) with the highest usage percentage in the cookbook, based on the specified limit. The response includes the ingredient name and its corresponding usage percentage, calculated as a proportion of the total ingredient count in the cookbook.")
async def get_most_used_ingredient_percentage(limit: int = Query(..., description="Number of ingredients to return")):
    cursor.execute("SELECT T1.name, CAST(COUNT(T2.ingredient_id) AS FLOAT) * 100 / ( SELECT COUNT(T2.ingredient_id) FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T2.ingredient_id = T1.ingredient_id ) AS 'percentage' FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T2.ingredient_id = T1.ingredient_id GROUP BY T2.ingredient_id ORDER BY COUNT(T2.ingredient_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"ingredients": []}
    return {"ingredients": [{"name": row[0], "percentage": row[1]} for row in result]}

# Endpoint to get recipes containing a specific ingredient
@app.get("/v1/cookbook/recipes_with_ingredient", operation_id="get_recipes_with_ingredient", summary="Retrieves a list of recipes that include a specified ingredient. The ingredient is identified by its name, which is provided as an input parameter.")
async def get_recipes_with_ingredient(ingredient_name: str = Query(..., description="Name of the ingredient")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T3.name = ?", (ingredient_name,))
    results = cursor.fetchall()
    if not results:
        return {"recipes": []}
    return {"recipes": [result[0] for result in results]}

# Endpoint to get the ingredient with the highest carbohydrate content where max and min quantities are equal
@app.get("/v1/cookbook/ingredient_highest_carbo_equal_qty", operation_id="get_ingredient_highest_carbo_equal_qty", summary="Retrieves the name of the ingredient that has the highest carbohydrate content in recipes where the maximum and minimum quantities are equal. The ingredient is determined by comparing the carbohydrate content of all ingredients that meet the specified quantity criteria and selecting the one with the highest value.")
async def get_ingredient_highest_carbo_equal_qty():
    cursor.execute("SELECT T1.name FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id INNER JOIN Nutrition AS T3 ON T3.recipe_id = T2.recipe_id WHERE T2.max_qty = T2.min_qty ORDER BY T3.carbo DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ingredient": []}
    return {"ingredient": result[0]}

# Endpoint to get the ingredient with the highest vitamin A content
@app.get("/v1/cookbook/ingredient_highest_vitamin_a", operation_id="get_ingredient_highest_vitamin_a", summary="Retrieves the name of the ingredient that has the highest vitamin A content in the database. This operation considers the vitamin A content of the ingredient across all recipes in which it is used. The ingredient with the highest vitamin A content is determined by sorting the ingredients in descending order based on their vitamin A content and selecting the top result.")
async def get_ingredient_highest_vitamin_a():
    cursor.execute("SELECT T1.name FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id INNER JOIN Nutrition AS T3 ON T3.recipe_id = T2.recipe_id ORDER BY T3.vitamin_a DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ingredient": []}
    return {"ingredient": result[0]}

# Endpoint to get ingredients and their maximum quantities for recipes with a specific number of servings
@app.get("/v1/cookbook/ingredients_max_qty_for_servings", operation_id="get_ingredients_max_qty_for_servings", summary="Retrieves the names of ingredients and their maximum quantities used in recipes that serve a specific number of people. The number of servings is provided as an input parameter.")
async def get_ingredients_max_qty_for_servings(servings: int = Query(..., description="Number of servings")):
    cursor.execute("SELECT T3.name, T2.max_qty FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T1.servings = ?", (servings,))
    results = cursor.fetchall()
    if not results:
        return {"ingredients": []}
    return {"ingredients": [{"name": result[0], "max_qty": result[1]} for result in results]}

# Endpoint to get the percentage of recipes with sodium content below a specified value from a specific source
@app.get("/v1/cookbook/percentage_recipes_sodium_below", operation_id="get_percentage_recipes_sodium_below", summary="Retrieves the percentage of recipes from a specified source that have a sodium content below a given threshold. This operation calculates the proportion of recipes with sodium levels less than the provided threshold, offering insights into the distribution of sodium content in recipes from the selected source.")
async def get_percentage_recipes_sodium_below(sodium_threshold: int = Query(..., description="Sodium threshold value"), source: str = Query(..., description="Source of the recipe")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.sodium < ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.source = ?", (sodium_threshold, source))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get ingredients measured in a specific unit
@app.get("/v1/cookbook/ingredients_by_unit", operation_id="get_ingredients_by_unit", summary="Retrieves the names of all ingredients that are measured using the specified unit. The unit parameter determines the unit of measurement for which ingredients are returned.")
async def get_ingredients_by_unit(unit: str = Query(..., description="Unit of measurement")):
    cursor.execute("SELECT T1.name FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id WHERE T2.unit = ?", (unit,))
    results = cursor.fetchall()
    if not results:
        return {"ingredients": []}
    return {"ingredients": [result[0] for result in results]}

# Endpoint to get the count of ingredients in a specific category
@app.get("/v1/cookbook/count_ingredients_by_category", operation_id="get_count_ingredients_by_category", summary="Retrieves the total count of ingredients that belong to a specified category. This operation provides a quantitative overview of ingredients within a particular category, aiding in inventory management and recipe planning.")
async def get_count_ingredients_by_category(category: str = Query(..., description="Category of the ingredient")):
    cursor.execute("SELECT COUNT(*) FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T1.ingredient_id = T2.ingredient_id WHERE T1.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get recipe titles and total preparation time for recipes containing a specific ingredient
@app.get("/v1/cookbook/recipe_prep_time_by_ingredient", operation_id="get_recipe_prep_time_by_ingredient", summary="Retrieves the titles and total preparation time of recipes that include a specified ingredient. The ingredient name is used to filter the recipes and calculate the combined preparation, cooking, and standing time.")
async def get_recipe_prep_time_by_ingredient(ingredient_name: str = Query(..., description="Name of the ingredient")):
    cursor.execute("SELECT T1.title, T1.prep_min + T1.cook_min + T1.stnd_min FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T3.name = ?", (ingredient_name,))
    result = cursor.fetchall()
    if not result:
        return {"recipes": []}
    return {"recipes": result}

# Endpoint to get the percentage of recipes with a specific ingredient that serve a minimum number of people
@app.get("/v1/cookbook/percentage_recipes_serving_minimum", operation_id="get_percentage_recipes_serving_minimum", summary="Retrieves the percentage of recipes containing a specified ingredient that can serve a minimum number of people. The calculation is based on the total count of recipes with the given ingredient.")
async def get_percentage_recipes_serving_minimum(min_servings: int = Query(..., description="Minimum number of servings"), ingredient_name: str = Query(..., description="Name of the ingredient")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.servings >= ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T3.name = ?", (min_servings, ingredient_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total fat content of a specific recipe
@app.get("/v1/cookbook/total_fat_by_recipe", operation_id="get_total_fat_by_recipe", summary="Retrieves the total fat content of a specific recipe by its title. The operation calculates the total fat content by joining the Recipe and Nutrition tables using the recipe_id field.")
async def get_total_fat_by_recipe(recipe_title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT T2.total_fat FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.title = ?", (recipe_title,))
    result = cursor.fetchone()
    if not result:
        return {"total_fat": []}
    return {"total_fat": result[0]}

# Endpoint to get the percentage of calories from protein for a specific recipe
@app.get("/v1/cookbook/protein_calories_percentage_by_recipe", operation_id="get_protein_calories_percentage_by_recipe", summary="Retrieves the proportion of calories derived from protein in a specific recipe. The recipe is identified by its title, which is provided as an input parameter.")
async def get_protein_calories_percentage_by_recipe(recipe_title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT pcnt_cal_prot FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T1.title = ?", (recipe_title,))
    result = cursor.fetchone()
    if not result:
        return {"pcnt_cal_prot": []}
    return {"pcnt_cal_prot": result[0]}

# Endpoint to get the count of ingredients in a specific recipe
@app.get("/v1/cookbook/ingredient_count_by_recipe", operation_id="get_ingredient_count_by_recipe", summary="Retrieves the total number of ingredients used in a specific recipe. The operation requires the title of the recipe as input to accurately determine the ingredient count.")
async def get_ingredient_count_by_recipe(recipe_title: str = Query(..., description="Title of the recipe")):
    cursor.execute("SELECT COUNT(*) FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T1.title = ?", (recipe_title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get recipe titles with a specific alcohol content
@app.get("/v1/cookbook/recipes_by_alcohol_content", operation_id="get_recipes_by_alcohol_content", summary="Retrieves the titles of recipes that contain a specified alcohol content. The alcohol content is provided as an input parameter, with 0 indicating non-alcoholic recipes. The endpoint returns a list of recipe titles that match the given alcohol content.")
async def get_recipes_by_alcohol_content(alcohol_content: int = Query(..., description="Alcohol content (0 for non-alcoholic)")):
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id WHERE T2.alcohol = ?", (alcohol_content,))
    result = cursor.fetchall()
    if not result:
        return {"recipes": []}
    return {"recipes": result}

# Endpoint to get the average vitamin C content of recipes with titles containing a specific keyword
@app.get("/v1/cookbook/average_vitamin_c_by_keyword", operation_id="get_average_vitamin_c_by_keyword", summary="Retrieves the average vitamin C content of recipes whose titles include a specified keyword. The keyword is used to filter the recipes, and the average vitamin C content is calculated based on the filtered results.")
async def get_average_vitamin_c_by_keyword(keyword: str = Query(..., description="Keyword to search in recipe titles")):
    cursor.execute("SELECT AVG(T1.vitamin_c) FROM Nutrition AS T1 INNER JOIN Recipe AS T2 ON T2.recipe_id = T1.recipe_id WHERE T2.title LIKE ?", ('%' + keyword + '%',))
    result = cursor.fetchone()
    if not result:
        return {"average_vitamin_c": []}
    return {"average_vitamin_c": result[0]}

# Endpoint to get the count of recipes with a specific ingredient category and minimum servings
@app.get("/v1/cookbook/recipe_count_by_category_and_servings", operation_id="get_recipe_count_by_category_and_servings", summary="Retrieves the number of recipes that contain a specific ingredient category and meet a minimum serving requirement. The category parameter filters recipes based on the ingredient category, while the min_servings parameter ensures that only recipes with a serving count greater than the specified value are considered.")
async def get_recipe_count_by_category_and_servings(category: str = Query(..., description="Category of the ingredient"), min_servings: int = Query(..., description="Minimum number of servings")):
    cursor.execute("SELECT COUNT(*) FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id WHERE T3.category = ? AND T1.servings > ?", (category, min_servings))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the recipe with the highest calorie content
@app.get("/v1/cookbook/highest_calorie_recipe", operation_id="get_highest_calorie_recipe", summary="Retrieves the recipe with the highest calorie content from the database. The operation returns the title of the recipe that has the most calories, based on the nutrition information.")
async def get_highest_calorie_recipe():
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id ORDER BY T2.calories DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"recipe": []}
    return {"recipe": result[0]}

# Endpoint to get the count of recipes excluding a specific ingredient category
@app.get("/v1/cookbook/recipe_count_excluding_category", operation_id="get_recipe_count_excluding_category", summary="Retrieves the total number of recipes that do not include any ingredients from a specified category. The category to be excluded is provided as an input parameter.")
async def get_recipe_count_excluding_category(excluded_category: str = Query(..., description="Category of the ingredient to exclude")):
    cursor.execute("SELECT COUNT(T2.recipe_id) FROM Ingredient AS T1 INNER JOIN Quantity AS T2 ON T2.ingredient_id = T1.ingredient_id INNER JOIN Nutrition AS T3 ON T3.recipe_id = T2.recipe_id WHERE T1.category NOT LIKE ?", ('%' + excluded_category + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get recipe titles where the maximum quantity is not equal to the minimum quantity
@app.get("/v1/cookbook/recipes_with_varying_quantities", operation_id="get_recipes_with_varying_quantities", summary="Retrieves the titles of recipes that have varying quantities. This operation identifies recipes where the maximum quantity differs from the minimum quantity, indicating a variation in ingredient amounts.")
async def get_recipes_with_varying_quantities():
    cursor.execute("SELECT T1.title FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id WHERE T2.max_qty <> T2.min_qty")
    result = cursor.fetchall()
    if not result:
        return {"recipes": []}
    return {"recipes": result}

# Endpoint to get the ingredient name of the recipe with the longest cooking time
@app.get("/v1/cookbook/ingredient_of_longest_cooking_recipe", operation_id="get_ingredient_of_longest_cooking_recipe", summary="Retrieves the name of the ingredient that is used in the recipe with the longest cooking time. The ingredient is determined by considering the cooking time of all recipes and selecting the one with the highest value. The result is the name of the ingredient used in that recipe.")
async def get_ingredient_of_longest_cooking_recipe():
    cursor.execute("SELECT T3.name FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id ORDER BY T1.cook_min DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ingredient": []}
    return {"ingredient": result[0]}

# Endpoint to get the percentage of recipes with cooking time less than a specified value and zero cholesterol
@app.get("/v1/cookbook/percentage_recipes_cook_time_cholesterol", operation_id="get_percentage_recipes_cook_time_cholesterol", summary="Retrieves the percentage of recipes that have a cooking time less than the specified duration and contain no cholesterol. The calculation is based on the total number of recipes in the database.")
async def get_percentage_recipes_cook_time_cholesterol(cook_min: int = Query(..., description="Cooking time in minutes"), cholestrl: int = Query(..., description="Cholesterol value, must be 0")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.cook_min < ? AND T2.cholestrl = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Recipe AS T1 INNER JOIN Nutrition AS T2 ON T1.recipe_id = T2.recipe_id", (cook_min, cholestrl))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of recipes with calories greater than a specified value for a given ingredient category
@app.get("/v1/cookbook/percentage_recipes_calories_ingredient_category", operation_id="get_percentage_recipes_calories_ingredient_category", summary="Retrieves the percentage of recipes that have a calorie count exceeding a specified value for a given ingredient category. This operation calculates the proportion of recipes with calories greater than the provided threshold within the specified ingredient category.")
async def get_percentage_recipes_calories_ingredient_category(calories: int = Query(..., description="Calorie value"), category: str = Query(..., description="Ingredient category")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T4.calories > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Recipe AS T1 INNER JOIN Quantity AS T2 ON T1.recipe_id = T2.recipe_id INNER JOIN Ingredient AS T3 ON T3.ingredient_id = T2.ingredient_id INNER JOIN Nutrition AS T4 ON T4.recipe_id = T1.recipe_id WHERE T3.category = ?", (calories, category))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/cookbook/recipe_highest_total_fat?limit=1",
    "/v1/cookbook/fat_difference_by_title?title=Raspberry%20Chiffon%20Pie",
    "/v1/cookbook/recipes_low_sodium?max_sodium=5",
    "/v1/cookbook/recipes_high_iron?min_iron=20",
    "/v1/cookbook/compare_vitamin_c?title1=Raspberry%20Chiffon%20Pie&title2=Fresh%20Apricot%20Bavarian",
    "/v1/cookbook/highest_calories_prep_time?min_prep_min=10&limit=1",
    "/v1/cookbook/calories_by_title?title=Raspberry%20Chiffon%20Pie",
    "/v1/cookbook/ingredient_optional_status?title=Raspberry%20Chiffon%20Pie&ingredient_name=graham%20cracker%20crumbs",
    "/v1/cookbook/count_equal_quantities?title=Raspberry%20Chiffon%20Pie",
    "/v1/cookbook/ingredients_no_preparation?title=Raspberry%20Chiffon%20Pie",
    "/v1/cookbook/ingredient_count_by_name?ingredient_name=graham%20cracker%20crumbs",
    "/v1/cookbook/min_quantity_ingredient_recipe?recipe_title=Raspberry%20Chiffon%20Pie&ingredient_name=graham%20cracker%20crumbs",
    "/v1/cookbook/calories_from_fat_recipe?recipe_title=Raspberry%20Chiffon%20Pie",
    "/v1/cookbook/average_calories_by_source?source=Produce%20for%20Better%20Health%20Foundation%20and%205%20a%20Day",
    "/v1/cookbook/ingredient_count_by_name_unit_recipe?ingredient_name=1%25%20lowfat%20milk&unit=cup(s)&recipe_id=1436",
    "/v1/cookbook/optional_ingredients_by_recipe?recipe_id=1397&optional=TRUE",
    "/v1/cookbook/recipe_titles_by_ingredient_equal_quantities?ingredient_name=frozen%20raspberries%20in%20light%20syrup",
    "/v1/cookbook/most_frequent_ingredient",
    "/v1/cookbook/ingredient_preparation?recipe_title=Raspberry-Pear%20Couscous%20Cake&ingredient_name=apple%20juice",
    "/v1/cookbook/recipe_count_ingredient_unit?recipe_title=Chicken%20Pocket%20Sandwich&ingredient_name=almonds&unit=cup(s)",
    "/v1/cookbook/recipe_highest_vitamin_c",
    "/v1/cookbook/recipe_vitamin_a?recipe_title=Sherried%20Beef",
    "/v1/cookbook/most_common_recipe",
    "/v1/cookbook/sodium_percentage?recipe_title_1=Lasagne-Spinach%20Spirals&recipe_title_2=Beef%20and%20Spinach%20Pita%20Pockets",
    "/v1/cookbook/average_calories_ingredient?ingredient_name=coarsely%20ground%20black%20pepper",
    "/v1/cookbook/recipe_count?recipe_title=Apricot%20Yogurt%20Parfaits",
    "/v1/cookbook/ingredient_names_preparation?preparation=cooked%20in%20beef%20broth",
    "/v1/cookbook/recipe_count_vitamin_a?vitamin_a=0",
    "/v1/cookbook/top_recipes_by_vitamin_c?limit=5",
    "/v1/cookbook/least_used_ingredient?limit=1",
    "/v1/cookbook/count_recipes_by_category_and_title?category=baking%20products&title=No-Bake%20Chocolate%20Cheesecake",
    "/v1/cookbook/ingredients_by_recipe_title?title=Strawberry%20Sorbet",
    "/v1/cookbook/optional_ingredients_by_recipe_title?title=Warm%20Chinese%20Chicken%20Salad&optional=TRUE",
    "/v1/cookbook/recipe_by_alcohol_and_prep_time?alcohol_threshold=10&limit=1",
    "/v1/cookbook/count_recipes_by_fat_difference?limit=1",
    "/v1/cookbook/recipe_by_source_and_calories?source=National%20Potato%20Board&limit=1",
    "/v1/cookbook/recipe_by_most_ingredients?limit=1",
    "/v1/cookbook/most_used_ingredient_percentage?limit=1",
    "/v1/cookbook/recipes_with_ingredient?ingredient_name=almond%20extract",
    "/v1/cookbook/ingredient_highest_carbo_equal_qty",
    "/v1/cookbook/ingredient_highest_vitamin_a",
    "/v1/cookbook/ingredients_max_qty_for_servings?servings=7",
    "/v1/cookbook/percentage_recipes_sodium_below?sodium_threshold=5&source=The%20California%20Tree%20Fruit%20Agreement",
    "/v1/cookbook/ingredients_by_unit?unit=slice(s)",
    "/v1/cookbook/count_ingredients_by_category?category=canned%20dairy",
    "/v1/cookbook/recipe_prep_time_by_ingredient?ingredient_name=lima%20beans",
    "/v1/cookbook/percentage_recipes_serving_minimum?min_servings=10&ingredient_name=sea%20bass%20steak",
    "/v1/cookbook/total_fat_by_recipe?recipe_title=Raspberry%20Chiffon%20Pie",
    "/v1/cookbook/protein_calories_percentage_by_recipe?recipe_title=Raspberry%20Chiffon%20Pie",
    "/v1/cookbook/ingredient_count_by_recipe?recipe_title=Raspberry%20Chiffon%20Pie",
    "/v1/cookbook/recipes_by_alcohol_content?alcohol_content=0",
    "/v1/cookbook/average_vitamin_c_by_keyword?keyword=cake",
    "/v1/cookbook/recipe_count_by_category_and_servings?category=dairy&min_servings=10",
    "/v1/cookbook/highest_calorie_recipe",
    "/v1/cookbook/recipe_count_excluding_category?excluded_category=dairy",
    "/v1/cookbook/recipes_with_varying_quantities",
    "/v1/cookbook/ingredient_of_longest_cooking_recipe",
    "/v1/cookbook/percentage_recipes_cook_time_cholesterol?cook_min=20&cholestrl=0",
    "/v1/cookbook/percentage_recipes_calories_ingredient_category?calories=200&category=cheese"
]
