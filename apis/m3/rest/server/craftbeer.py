from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/craftbeer/craftbeer.sqlite')
cursor = conn.cursor()

# Endpoint to get distinct states and IBUs of beers with the minimum IBU
@app.get("/v1/craftbeer/distinct_states_min_ibu", operation_id="get_distinct_states_min_ibu", summary="Retrieves a list of unique states and the corresponding minimum International Bitterness Unit (IBU) value for beers. This operation identifies the state of each brewery and the IBU of the beer with the lowest bitterness level across all beers in the database.")
async def get_distinct_states_min_ibu():
    cursor.execute("SELECT DISTINCT T2.state, T1.ibu FROM beers AS T1 INNER JOIN breweries AS T2 ON T1.brewery_id = T2.id WHERE T1.ibu IS NOT NULL AND T1.ibu = ( SELECT MIN(ibu) FROM beers )")
    result = cursor.fetchall()
    if not result:
        return {"states_ibu": []}
    return {"states_ibu": result}

# Endpoint to get the brewery with the highest IBU beer in a given state
@app.get("/v1/craftbeer/highest_ibu_brewery_by_state", operation_id="get_highest_ibu_brewery_by_state", summary="Retrieves the name and city of the brewery that produces the beer with the highest International Bitterness Units (IBU) in a specified state. The IBU is a measure of the bitterness of beer, which is provided by the hops used during brewing. The state is a required input parameter.")
async def get_highest_ibu_brewery_by_state(state: str = Query(..., description="State of the brewery")):
    cursor.execute("SELECT T2.name, T2.city FROM beers AS T1 INNER JOIN breweries AS T2 ON T1.brewery_id = T2.id WHERE T2.state = ? ORDER BY T1.ibu DESC LIMIT 1", (state,))
    result = cursor.fetchone()
    if not result:
        return {"brewery": []}
    return {"brewery": result}

# Endpoint to get the average ABV of beers from a specific brewery with a given ounces
@app.get("/v1/craftbeer/avg_abv_by_brewery_and_ounces", operation_id="get_avg_abv_by_brewery_and_ounces", summary="Retrieves the average alcohol by volume (ABV) of beers from a specified brewery, filtered by a given ounce volume. This operation provides a statistical overview of the alcohol content in beers produced by the selected brewery, based on the provided ounce volume.")
async def get_avg_abv_by_brewery_and_ounces(brewery_name: str = Query(..., description="Name of the brewery"), ounces: int = Query(..., description="Ounces of the beer")):
    cursor.execute("SELECT AVG(T1.abv) FROM beers AS T1 INNER JOIN breweries AS T2 ON T1.brewery_id = T2.id WHERE T2.name = ? AND T1.ounces = ?", (brewery_name, ounces))
    result = cursor.fetchone()
    if not result:
        return {"avg_abv": []}
    return {"avg_abv": result[0]}

# Endpoint to get the percentage of beers of a specific style from a given brewery
@app.get("/v1/craftbeer/percentage_beer_style_by_brewery", operation_id="get_percentage_beer_style_by_brewery", summary="Retrieves the proportion of beers of a specified style produced by a given brewery. The calculation is based on the total count of beers from the brewery. The style and brewery name are provided as input parameters.")
async def get_percentage_beer_style_by_brewery(beer_style: str = Query(..., description="Style of the beer"), brewery_name: str = Query(..., description="Name of the brewery")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.style = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.brewery_id) FROM beers AS T1 INNER JOIN breweries AS T2 ON T1.brewery_id = T2.id WHERE T2.name = ?", (beer_style, brewery_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get breweries with beers having both maximum and minimum IBUs
@app.get("/v1/craftbeer/breweries_with_max_min_ibu", operation_id="get_breweries_with_max_min_ibu", summary="Retrieves brewery details for up to two breweries in each city and state that produce beers with both the highest and lowest International Bitterness Units (IBUs). The IBU is a measure of the bitterness of beer, which is provided for each beer produced by the brewery.")
async def get_breweries_with_max_min_ibu():
    cursor.execute("SELECT T1.state, T1.city, T2.name, T2.ibu FROM breweries AS T1 INNER JOIN beers AS T2 ON T1.id = T2.brewery_id GROUP BY T1.state, T1.city, T2.name, T2.ibu HAVING MAX(ibu) AND MIN(ibu) LIMIT 2")
    result = cursor.fetchall()
    if not result:
        return {"breweries": []}
    return {"breweries": result}

# Endpoint to get the percentage of beers from a specific state with a given style
@app.get("/v1/craftbeer/percentage_beer_state_by_style", operation_id="get_percentage_beer_state_by_style", summary="Retrieves the percentage of beers from a specified state that match a given style. This operation calculates the proportion of beers from a particular state that have a certain style, providing insights into the distribution of beer styles within that state.")
async def get_percentage_beer_state_by_style(state: str = Query(..., description="State of the brewery"), beer_style: str = Query(..., description="Style of the beer")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.state = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.id) FROM beers AS T1 INNER JOIN breweries AS T2 ON T1.brewery_id = T2.id WHERE T1.style = ?", (state, beer_style))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/craftbeer/distinct_states_min_ibu",
    "/v1/craftbeer/highest_ibu_brewery_by_state?state=NY",
    "/v1/craftbeer/avg_abv_by_brewery_and_ounces?brewery_name=Boston%20Beer%20Company&ounces=12",
    "/v1/craftbeer/percentage_beer_style_by_brewery?beer_style=American%20Adjunct%20Lager&brewery_name=Stevens%20Point%20Brewery",
    "/v1/craftbeer/breweries_with_max_min_ibu",
    "/v1/craftbeer/percentage_beer_state_by_style?state=WI&beer_style=American%20Blonde%20Ale"
]
