from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/card_games/card_games.sqlite')
cursor = conn.cursor()

# Endpoint to get card IDs where both cardKingdomFoilId and cardKingdomId are not null
@app.get("/v1/card_games/cards_with_foil_and_regular_ids", operation_id="get_cards_with_foil_and_regular_ids", summary="Retrieves the IDs of cards that have both a foil and a regular version available. This operation returns a list of card IDs where both the foil and regular card identifiers are not null.")
async def get_cards_with_foil_and_regular_ids():
    cursor.execute("SELECT id FROM cards WHERE cardKingdomFoilId IS NOT NULL AND cardKingdomId IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get card IDs with a specific border color and missing cardKingdomId
@app.get("/v1/card_games/cards_with_border_color_missing_id", operation_id="get_cards_with_border_color_missing_id", summary="Retrieves the IDs of cards that have a specified border color and lack a cardKingdomId. The border color is provided as an input parameter.")
async def get_cards_with_border_color_missing_id(border_color: str = Query(..., description="Border color of the card")):
    cursor.execute("SELECT id FROM cards WHERE borderColor = ? AND (cardKingdomId IS NULL OR cardKingdomId IS NULL)", (border_color,))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get the name of the card with the lowest face converted mana cost
@app.get("/v1/card_games/card_with_lowest_mana_cost", operation_id="get_card_with_lowest_mana_cost", summary="Retrieves the name of the card with the lowest mana cost from the available cards. This operation returns the card with the least resource requirement for gameplay.")
async def get_card_with_lowest_mana_cost():
    cursor.execute("SELECT name FROM cards ORDER BY faceConvertedManaCost LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get card IDs with a specific EDHREC rank and frame version
@app.get("/v1/card_games/cards_with_edhrec_rank_and_frame_version", operation_id="get_cards_with_edhrec_rank_and_frame_version", summary="Retrieves the unique identifiers of cards that have an EDHREC rank lower than the provided value and a specific frame version. This operation allows you to filter cards based on their EDHREC rank and frame version, providing a targeted list of card IDs that meet the specified criteria.")
async def get_cards_with_edhrec_rank_and_frame_version(edhrec_rank: int = Query(..., description="EDHREC rank of the card"), frame_version: int = Query(..., description="Frame version of the card")):
    cursor.execute("SELECT id FROM cards WHERE edhrecRank < ? AND frameVersion = ?", (edhrec_rank, frame_version))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get distinct card IDs based on format, status, and rarity
@app.get("/v1/card_games/distinct_card_ids_by_format_status_rarity", operation_id="get_distinct_card_ids_by_format_status_rarity", summary="Retrieve a unique set of card identifiers that meet the specified format, status, and rarity criteria. This operation filters cards based on the provided format, status, and rarity, and returns a distinct list of card IDs that match the given conditions.")
async def get_distinct_card_ids_by_format_status_rarity(format: str = Query(..., description="Format of the card"), status: str = Query(..., description="Status of the card"), rarity: str = Query(..., description="Rarity of the card")):
    cursor.execute("SELECT DISTINCT T1.id FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T2.format = ? AND T2.status = ? AND T1.rarity = ?", (format, status, rarity))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get distinct statuses of cards based on type, format, and side
@app.get("/v1/card_games/distinct_statuses_by_type_format_side", operation_id="get_distinct_statuses_by_type_format_side", summary="Get distinct statuses of cards based on type, format, and side")
async def get_distinct_statuses_by_type_format_side(type: str = Query(..., description="Type of the card"), format: str = Query(..., description="Format of the card"), side: str = Query(None, description="Side of the card, if any")):
    cursor.execute("SELECT DISTINCT T2.status FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T1.type = ? AND T2.format = ? AND T1.side IS ?", (type, format, side))
    result = cursor.fetchall()
    if not result:
        return {"statuses": []}
    return {"statuses": [row[0] for row in result]}

# Endpoint to get card IDs and artists based on status, format, and power
@app.get("/v1/card_games/card_ids_artists_by_status_format_power", operation_id="get_card_ids_artists_by_status_format_power", summary="Retrieves card IDs and their respective artists based on the specified card status, format, and power. The operation filters cards by the provided status and format, and optionally by power if it is not null. This endpoint is useful for obtaining a list of cards that meet specific criteria, enabling users to find cards that match their desired attributes.")
async def get_card_ids_artists_by_status_format_power(status: str = Query(..., description="Status of the card"), format: str = Query(..., description="Format of the card"), power: str = Query(..., description="Power of the card, if any")):
    cursor.execute("SELECT T1.id, T1.artist FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T2.status = ? AND T2.format = ? AND (T1.power IS NULL OR T1.power = ?)", (status, format, power))
    result = cursor.fetchall()
    if not result:
        return {"cards": []}
    return {"cards": [{"id": row[0], "artist": row[1]} for row in result]}

# Endpoint to get card IDs, ruling texts, and content warnings based on artist
@app.get("/v1/card_games/card_ids_rulings_content_warnings_by_artist", operation_id="get_card_ids_rulings_content_warnings_by_artist", summary="Retrieves a list of card IDs, associated ruling texts, and content warning indicators for cards created by the specified artist.")
async def get_card_ids_rulings_content_warnings_by_artist(artist: str = Query(..., description="Artist of the card")):
    cursor.execute("SELECT T1.id, T2.text, T1.hasContentWarning FROM cards AS T1 INNER JOIN rulings AS T2 ON T1.uuid = T2.uuid WHERE T1.artist = ?", (artist,))
    result = cursor.fetchall()
    if not result:
        return {"cards": []}
    return {"cards": [{"id": row[0], "text": row[1], "hasContentWarning": row[2]} for row in result]}

# Endpoint to get ruling texts based on card name and number
@app.get("/v1/card_games/ruling_texts_by_card_name_number", operation_id="get_ruling_texts_by_card_name_number", summary="Retrieves the ruling texts associated with a specific card, identified by its name and number. The operation returns the text of the rulings that apply to the card, providing detailed information about its usage and interactions within the game.")
async def get_ruling_texts_by_card_name_number(name: str = Query(..., description="Name of the card"), number: str = Query(..., description="Number of the card")):
    cursor.execute("SELECT T2.text FROM cards AS T1 INNER JOIN rulings AS T2 ON T1.uuid = T2.uuid WHERE T1.name = ? AND T1.number = ?", (name, number))
    result = cursor.fetchall()
    if not result:
        return {"texts": []}
    return {"texts": [row[0] for row in result]}

# Endpoint to get the name, artist, and promo status of the card with the most promo cards by a single artist
@app.get("/v1/card_games/most_promo_cards_by_artist", operation_id="get_most_promo_cards_by_artist", summary="Retrieves the name, artist, and promo status of the card that has the highest number of unique promo cards created by a single artist. This operation identifies the artist with the most distinct promo cards and returns the details of one of their promo cards.")
async def get_most_promo_cards_by_artist():
    cursor.execute("SELECT T1.name, T1.artist, T1.isPromo FROM cards AS T1 INNER JOIN rulings AS T2 ON T1.uuid = T2.uuid WHERE T1.isPromo = 1 AND T1.artist = (SELECT artist FROM cards WHERE isPromo = 1 GROUP BY artist HAVING COUNT(DISTINCT uuid) = (SELECT MAX(count_uuid) FROM ( SELECT COUNT(DISTINCT uuid) AS count_uuid FROM cards WHERE isPromo = 1 GROUP BY artist ))) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"card": []}
    return {"card": {"name": result[0], "artist": result[1], "isPromo": result[2]}}

# Endpoint to get the language of a card based on its name and number
@app.get("/v1/card_games/get_language_by_card_name_and_number", operation_id="get_language_by_card_name_and_number", summary="Retrieves the language associated with a specific card, identified by its unique name and number. The operation returns the language of the card, which is determined by querying the card's name and number in the database.")
async def get_language_by_card_name_and_number(name: str = Query(..., description="Name of the card"), number: int = Query(..., description="Number of the card")):
    cursor.execute("SELECT T2.language FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T1.name = ? AND T1.number = ?", (name, number))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the names of cards based on language
@app.get("/v1/card_games/get_card_names_by_language", operation_id="get_card_names_by_language", summary="Retrieves the names of cards that are available in a specified language. The operation filters the cards based on the provided language and returns their names.")
async def get_card_names_by_language(language: str = Query(..., description="Language of the card")):
    cursor.execute("SELECT T1.name FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T2.language = ?", (language,))
    result = cursor.fetchall()
    if not result:
        return {"card_names": []}
    return {"card_names": [row[0] for row in result]}

# Endpoint to get the percentage of cards in a specific language
@app.get("/v1/card_games/get_percentage_of_cards_in_language", operation_id="get_percentage_of_cards_in_language", summary="Retrieves the percentage of cards in a specified language. This operation calculates the proportion of cards in the given language by comparing the count of cards in that language to the total number of cards. The language is provided as an input parameter.")
async def get_percentage_of_cards_in_language(language: str = Query(..., description="Language of the card")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.language = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid", (language,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the names and total set sizes of sets based on language
@app.get("/v1/card_games/get_set_names_and_sizes_by_language", operation_id="get_set_names_and_sizes_by_language", summary="Retrieves the names and total set sizes of card game sets available in a specified language. The language is used to filter the sets and return only those that have been translated into the specified language.")
async def get_set_names_and_sizes_by_language(language: str = Query(..., description="Language of the set")):
    cursor.execute("SELECT T1.name, T1.totalSetSize FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = ?", (language,))
    result = cursor.fetchall()
    if not result:
        return {"sets": []}
    return {"sets": [{"name": row[0], "totalSetSize": row[1]} for row in result]}

# Endpoint to get the count of card types by artist
@app.get("/v1/card_games/get_card_type_count_by_artist", operation_id="get_card_type_count_by_artist", summary="Retrieves the total number of distinct card types associated with a specific artist. The artist's name is provided as an input parameter.")
async def get_card_type_count_by_artist(artist: str = Query(..., description="Name of the artist")):
    cursor.execute("SELECT COUNT(type) FROM cards WHERE artist = ?", (artist,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct keywords of a card based on its name
@app.get("/v1/card_games/get_distinct_keywords_by_card_name", operation_id="get_distinct_keywords_by_card_name", summary="Retrieves a unique set of keywords associated with a specific card, identified by its name.")
async def get_distinct_keywords_by_card_name(name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT DISTINCT keywords FROM cards WHERE name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"keywords": []}
    return {"keywords": [row[0] for row in result]}

# Endpoint to get the count of cards with a specific power
@app.get("/v1/card_games/get_card_count_by_power", operation_id="get_card_count_by_power", summary="Retrieves the total number of cards that have a specified power level. The power level is provided as an input parameter.")
async def get_card_count_by_power(power: str = Query(..., description="Power of the card")):
    cursor.execute("SELECT COUNT(*) FROM cards WHERE power = ?", (power,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the promo types of a card based on its name
@app.get("/v1/card_games/get_promo_types_by_card_name", operation_id="get_promo_types_by_card_name", summary="Retrieves the promotional types associated with a specific card, identified by its name. The card must have at least one promotional type to be included in the response.")
async def get_promo_types_by_card_name(name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT promoTypes FROM cards WHERE name = ? AND promoTypes IS NOT NULL", (name,))
    result = cursor.fetchall()
    if not result:
        return {"promo_types": []}
    return {"promo_types": [row[0] for row in result]}

# Endpoint to get distinct border colors of a card based on its name
@app.get("/v1/card_games/get_distinct_border_colors_by_card_name", operation_id="get_distinct_border_colors_by_card_name", summary="Retrieves the unique border colors associated with a specific card, as identified by its name.")
async def get_distinct_border_colors_by_card_name(name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT DISTINCT borderColor FROM cards WHERE name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"border_colors": []}
    return {"border_colors": [row[0] for row in result]}

# Endpoint to get the original type of a card based on its name
@app.get("/v1/card_games/get_original_type_by_card_name", operation_id="get_original_type_by_card_name", summary="Retrieves the original type of a card, provided the card's name and that the original type is not null. This operation is useful for identifying the card's original type without needing to know its ID or other specific details.")
async def get_original_type_by_card_name(name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT originalType FROM cards WHERE name = ? AND originalType IS NOT NULL", (name,))
    result = cursor.fetchall()
    if not result:
        return {"original_types": []}
    return {"original_types": [row[0] for row in result]}

# Endpoint to get the language of set translations for a given card name
@app.get("/v1/card_games/set_translations_language", operation_id="get_set_translations_language", summary="Retrieves the language used for the set translations of a specific card. The card is identified by its name, which is provided as an input parameter.")
async def get_set_translations_language(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT language FROM set_translations WHERE id IN ( SELECT id FROM cards WHERE name = ? )", (card_name,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the count of distinct cards with a given legality status and textless status
@app.get("/v1/card_games/count_distinct_cards_status_textless", operation_id="get_count_distinct_cards_status_textless", summary="Retrieves the count of unique cards that match the specified legality status and textless status. The legality status indicates the card's legal status, while the textless status indicates whether the card has text or not.")
async def get_count_distinct_cards_status_textless(status: str = Query(..., description="Legality status of the card"), is_textless: int = Query(..., description="Textless status of the card (0 or 1)")):
    cursor.execute("SELECT COUNT(DISTINCT T1.id) FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T2.status = ? AND T1.isTextless = ?", (status, is_textless))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ruling text for a given card name
@app.get("/v1/card_games/ruling_text_for_card", operation_id="get_ruling_text_for_card", summary="Retrieves the ruling text associated with a specific card. The operation requires the card's name as input and returns the corresponding ruling text from the database.")
async def get_ruling_text_for_card(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT T2.text FROM cards AS T1 INNER JOIN rulings AS T2 ON T1.uuid = T2.uuid WHERE T1.name = ?", (card_name,))
    result = cursor.fetchall()
    if not result:
        return {"rulings": []}
    return {"rulings": [row[0] for row in result]}

# Endpoint to get the count of distinct starter cards with a given legality status
@app.get("/v1/card_games/count_distinct_starter_cards_status", operation_id="get_count_distinct_starter_cards_status", summary="Retrieves the count of unique starter cards that have a specified legality status. The legality status and starter status are provided as input parameters to filter the results.")
async def get_count_distinct_starter_cards_status(status: str = Query(..., description="Legality status of the card"), is_starter: int = Query(..., description="Starter status of the card (0 or 1)")):
    cursor.execute("SELECT COUNT(DISTINCT T1.id) FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T2.status = ? AND T1.isStarter = ?", (status, is_starter))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the distinct legality statuses for a given card name
@app.get("/v1/card_games/distinct_legality_statuses_for_card", operation_id="get_distinct_legality_statuses_for_card", summary="Retrieve the unique legality statuses associated with a specific card. The operation filters cards by the provided card name and returns the distinct legality statuses linked to the card.")
async def get_distinct_legality_statuses_for_card(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT DISTINCT T2.status FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T1.name = ?", (card_name,))
    result = cursor.fetchall()
    if not result:
        return {"statuses": []}
    return {"statuses": [row[0] for row in result]}

# Endpoint to get the distinct types for a given card name
@app.get("/v1/card_games/distinct_types_for_card", operation_id="get_distinct_types_for_card", summary="Retrieves the unique types associated with a specific card. The operation filters cards by the provided card name and returns the distinct types linked to those cards.")
async def get_distinct_types_for_card(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT DISTINCT T1.type FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T1.name = ?", (card_name,))
    result = cursor.fetchall()
    if not result:
        return {"types": []}
    return {"types": [row[0] for row in result]}

# Endpoint to get the formats for a given card name
@app.get("/v1/card_games/formats_for_card", operation_id="get_formats_for_card", summary="Retrieves the game formats in which a specific card can be used. The card is identified by its name, and the response includes a list of formats where the card is legal.")
async def get_formats_for_card(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT T2.format FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T1.name = ?", (card_name,))
    result = cursor.fetchall()
    if not result:
        return {"formats": []}
    return {"formats": [row[0] for row in result]}

# Endpoint to get the artists for cards in a given language
@app.get("/v1/card_games/artists_for_language", operation_id="get_artists_for_language", summary="Retrieves the artists associated with cards in a specified language. The operation filters cards based on the provided language and returns the corresponding artist names.")
async def get_artists_for_language(language: str = Query(..., description="Language of the card")):
    cursor.execute("SELECT T1.artist FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T2.language = ?", (language,))
    result = cursor.fetchall()
    if not result:
        return {"artists": []}
    return {"artists": [row[0] for row in result]}

# Endpoint to get the percentage of cards with a given border color
@app.get("/v1/card_games/percentage_border_color", operation_id="get_percentage_border_color", summary="Retrieves the percentage of cards in the database that have the specified border color. The border color is provided as an input parameter.")
async def get_percentage_border_color(border_color: str = Query(..., description="Border color of the card")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN borderColor = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(id) FROM cards", (border_color,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of reprinted cards in a given language
@app.get("/v1/card_games/count_reprinted_cards_language", operation_id="get_count_reprinted_cards_language", summary="Retrieves the total number of reprinted cards in a specified language. The operation considers a card as reprinted if its 'isReprint' attribute is set to 1. The language of the cards is determined by the 'language' parameter.")
async def get_count_reprinted_cards_language(language: str = Query(..., description="Language of the card"), is_reprint: int = Query(..., description="Reprint status of the card (0 or 1)")):
    cursor.execute("SELECT COUNT(T1.id) FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T2.language = ? AND T1.isReprint = ?", (language, is_reprint))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cards with a specific border color and language
@app.get("/v1/card_games/count_cards_border_color_language", operation_id="get_count_cards_border_color_language", summary="Retrieves the total number of cards that have a specified border color and are available in a particular language. The response is based on a count of cards that match the provided border color and language.")
async def get_count_cards_border_color_language(border_color: str = Query(..., description="Border color of the card"), language: str = Query(..., description="Language of the card")):
    cursor.execute("SELECT COUNT(T1.id) FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T1.borderColor = ? AND T2.language = ?", (border_color, language))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of cards in a specific language that are story spotlight
@app.get("/v1/card_games/percentage_language_story_spotlight", operation_id="get_percentage_language_story_spotlight", summary="Retrieves the percentage of cards in a specified language that are designated as story spotlights. The operation calculates this percentage by comparing the count of cards in the given language that are story spotlights to the total number of cards in that language.")
async def get_percentage_language_story_spotlight(language: str = Query(..., description="Language of the card"), is_story_spotlight: int = Query(..., description="Whether the card is a story spotlight (1 for yes, 0 for no)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.language = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T1.isStorySpotlight = ?", (language, is_story_spotlight))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of cards with a specific toughness
@app.get("/v1/card_games/count_cards_toughness", operation_id="get_count_cards_toughness", summary="Retrieves the total number of cards that have a specified toughness level. The toughness level is provided as an input parameter, allowing you to filter the count based on this attribute.")
async def get_count_cards_toughness(toughness: int = Query(..., description="Toughness of the card")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE toughness = ?", (toughness,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct card names by a specific artist
@app.get("/v1/card_games/distinct_card_names_artist", operation_id="get_distinct_card_names_artist", summary="Retrieves a unique list of card names associated with a given artist. This operation allows you to discover the variety of cards created by a specific artist, providing a comprehensive view of their work.")
async def get_distinct_card_names_artist(artist: str = Query(..., description="Artist of the card")):
    cursor.execute("SELECT DISTINCT name FROM cards WHERE artist = ?", (artist,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of cards with a specific availability and border color
@app.get("/v1/card_games/count_cards_availability_border_color", operation_id="get_count_cards_availability_border_color", summary="Retrieves the total number of cards that match the specified availability status and border color. This operation is useful for determining the quantity of cards that meet certain criteria, such as being available for use and having a specific border color.")
async def get_count_cards_availability_border_color(availability: str = Query(..., description="Availability of the card"), border_color: str = Query(..., description="Border color of the card")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE availability = ? AND borderColor = ?", (availability, border_color))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get card IDs with a specific converted mana cost
@app.get("/v1/card_games/card_ids_converted_mana_cost", operation_id="get_card_ids_converted_mana_cost", summary="Retrieves the unique identifiers of cards that have a specified total mana cost. The mana cost is a crucial attribute of a card, representing the resources required to play it in a game.")
async def get_card_ids_converted_mana_cost(converted_mana_cost: int = Query(..., description="Converted mana cost of the card")):
    cursor.execute("SELECT id FROM cards WHERE convertedManaCost = ?", (converted_mana_cost,))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get card layouts with a specific keyword
@app.get("/v1/card_games/card_layouts_keyword", operation_id="get_card_layouts_keyword", summary="Retrieves the layout of cards that match a specified keyword. The keyword is used to filter the cards and return only those that contain the provided keyword in their description.")
async def get_card_layouts_keyword(keyword: str = Query(..., description="Keyword of the card")):
    cursor.execute("SELECT layout FROM cards WHERE keywords = ?", (keyword,))
    result = cursor.fetchall()
    if not result:
        return {"layouts": []}
    return {"layouts": [row[0] for row in result]}

# Endpoint to get the count of cards with a specific original type and excluding a specific subtype
@app.get("/v1/card_games/count_cards_original_type_exclude_subtype", operation_id="get_count_cards_original_type_exclude_subtype", summary="Retrieves the total number of cards that belong to a particular original type and do not include a specified subtype.")
async def get_count_cards_original_type_exclude_subtype(original_type: str = Query(..., description="Original type of the card"), subtype: str = Query(..., description="Subtype to exclude")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE originalType = ? AND subtypes != ?", (original_type, subtype))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get card IDs with non-null cardKingdomId and cardKingdomFoilId
@app.get("/v1/card_games/card_ids_non_null_card_kingdom", operation_id="get_card_ids_non_null_card_kingdom", summary="Retrieves a list of card IDs that have associated kingdom and foil kingdom information. This operation filters out cards that lack either kingdom or foil kingdom details, ensuring that only cards with complete kingdom data are returned.")
async def get_card_ids_non_null_card_kingdom():
    cursor.execute("SELECT id FROM cards WHERE cardKingdomId IS NOT NULL AND cardKingdomFoilId IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get card IDs with a specific duel deck
@app.get("/v1/card_games/card_ids_duel_deck", operation_id="get_card_ids_duel_deck", summary="Retrieves the unique identifiers of cards that belong to a specified duel deck. The duel deck parameter determines which cards are included in the response.")
async def get_card_ids_duel_deck(duel_deck: str = Query(..., description="Duel deck of the card")):
    cursor.execute("SELECT id FROM cards WHERE duelDeck = ?", (duel_deck,))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get the EDHREC rank of cards based on frame version
@app.get("/v1/card_games/edhrec_rank_by_frame_version", operation_id="get_edhrec_rank", summary="Retrieves the EDHREC rank of cards that match the specified frame version. The frame version parameter is used to filter the cards and return their corresponding EDHREC ranks.")
async def get_edhrec_rank(frame_version: int = Query(..., description="Frame version of the card")):
    cursor.execute("SELECT edhrecRank FROM cards WHERE frameVersion = ?", (frame_version,))
    result = cursor.fetchall()
    if not result:
        return {"edhrec_rank": []}
    return {"edhrec_rank": [row[0] for row in result]}

# Endpoint to get the name of cards based on availability and language
@app.get("/v1/card_games/name_by_availability_and_language", operation_id="get_name_by_availability_and_language", summary="Retrieves the names of cards that are available and in a specified language. The availability and language are used as filters to determine which card names are returned.")
async def get_name_by_availability_and_language(availability: str = Query(..., description="Availability of the card"), language: str = Query(..., description="Language of the card")):
    cursor.execute("SELECT T1.name FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T1.availability = ? AND T2.language = ?", (availability, language))
    result = cursor.fetchall()
    if not result:
        return {"name": []}
    return {"name": [row[0] for row in result]}

# Endpoint to get the count of cards based on status and border color
@app.get("/v1/card_games/count_by_status_and_border_color", operation_id="get_count_by_status_and_border_color", summary="Retrieves the total number of cards that match the specified status and border color. The status and border color are provided as input parameters, allowing for a targeted count of cards that meet these criteria.")
async def get_count_by_status_and_border_color(status: str = Query(..., description="Status of the card"), border_color: str = Query(..., description="Border color of the card")):
    cursor.execute("SELECT COUNT(T1.id) FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T2.status = ? AND T1.borderColor = ?", (status, border_color))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the UUID and language of cards based on format
@app.get("/v1/card_games/uuid_and_language_by_format", operation_id="get_uuid_and_language_by_format", summary="Retrieves the unique identifiers (UUIDs) and languages of cards that are legal in the specified format. The format parameter is used to filter the cards by their legality in a particular format.")
async def get_uuid_and_language_by_format(format: str = Query(..., description="Format of the card")):
    cursor.execute("SELECT T1.uuid, T3.language FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid INNER JOIN foreign_data AS T3 ON T1.uuid = T3.uuid WHERE T2.format = ?", (format,))
    result = cursor.fetchall()
    if not result:
        return {"uuid_and_language": []}
    return {"uuid_and_language": [{"uuid": row[0], "language": row[1]} for row in result]}

# Endpoint to get the count of cards based on frame version
@app.get("/v1/card_games/count_by_frame_version", operation_id="get_count_by_frame_version", summary="Retrieves the total number of cards that share a specific frame version. The frame version is a parameter that filters the count, allowing for targeted results.")
async def get_count_by_frame_version(frame_version: str = Query(..., description="Frame version of the card")):
    cursor.execute("SELECT COUNT(T1.id) FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T1.frameVersion = ?", (frame_version,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ID and colors of cards based on set code
@app.get("/v1/card_games/id_and_colors_by_set_code", operation_id="get_id_and_colors_by_set_code", summary="Retrieves the unique identifiers and color information of cards associated with a specific set code. The set code is used to filter the cards and return only those that belong to the specified set.")
async def get_id_and_colors_by_set_code(set_code: str = Query(..., description="Set code of the card")):
    cursor.execute("SELECT id, colors FROM cards WHERE id IN ( SELECT id FROM set_translations WHERE setCode = ? )", (set_code,))
    result = cursor.fetchall()
    if not result:
        return {"id_and_colors": []}
    return {"id_and_colors": [{"id": row[0], "colors": row[1]} for row in result]}

# Endpoint to get the ID and language of set translations based on converted mana cost and set code
@app.get("/v1/card_games/id_and_language_by_mana_cost_and_set_code", operation_id="get_id_and_language_by_mana_cost_and_set_code", summary="Retrieves the unique identifier and language of a card set based on its converted mana cost and set code. The converted mana cost and set code are provided as input parameters to filter the results.")
async def get_id_and_language_by_mana_cost_and_set_code(converted_mana_cost: int = Query(..., description="Converted mana cost of the card"), set_code: str = Query(..., description="Set code of the card")):
    cursor.execute("SELECT id, language FROM set_translations WHERE id = ( SELECT id FROM cards WHERE convertedManaCost = ? ) AND setCode = ?", (converted_mana_cost, set_code))
    result = cursor.fetchall()
    if not result:
        return {"id_and_language": []}
    return {"id_and_language": [{"id": row[0], "language": row[1]} for row in result]}

# Endpoint to get the ID and date of rulings based on original type
@app.get("/v1/card_games/id_and_date_by_original_type", operation_id="get_id_and_date_by_original_type", summary="Retrieves the unique identifier (ID) and date of rulings for cards of a specified original type. The original type is a categorization of the card, and this operation filters the results based on this attribute.")
async def get_id_and_date_by_original_type(original_type: str = Query(..., description="Original type of the card")):
    cursor.execute("SELECT T1.id, T2.date FROM cards AS T1 INNER JOIN rulings AS T2 ON T1.uuid = T2.uuid WHERE T1.originalType = ?", (original_type,))
    result = cursor.fetchall()
    if not result:
        return {"id_and_date": []}
    return {"id_and_date": [{"id": row[0], "date": row[1]} for row in result]}

# Endpoint to get card colors and formats based on a range of card IDs
@app.get("/v1/card_games/card_colors_formats_by_id_range", operation_id="get_card_colors_formats_by_id_range", summary="Retrieves the colors and formats of cards that fall within the specified ID range. The operation filters cards based on their unique identifiers and returns the corresponding color and format information.")
async def get_card_colors_formats_by_id_range(min_id: int = Query(..., description="Minimum card ID"), max_id: int = Query(..., description="Maximum card ID")):
    cursor.execute("SELECT T1.colors, T2.format FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T1.id BETWEEN ? AND ?", (min_id, max_id))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct card names based on original type and colors
@app.get("/v1/card_games/distinct_card_names_by_type_and_color", operation_id="get_distinct_card_names_by_type_and_color", summary="Retrieves a list of unique card names that match the specified original type and colors. This operation filters cards based on their original type and colors, returning only the distinct names of cards that meet the criteria.")
async def get_distinct_card_names_by_type_and_color(original_type: str = Query(..., description="Original type of the card"), colors: str = Query(..., description="Colors of the card")):
    cursor.execute("SELECT DISTINCT T1.name FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T1.originalType = ? AND T1.colors = ?", (original_type, colors))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct card names based on rarity and ordered by ruling date
@app.get("/v1/card_games/distinct_card_names_by_rarity_ordered_by_date", operation_id="get_distinct_card_names_by_rarity_ordered_by_date", summary="Retrieves a list of distinct card names that match the specified rarity, ordered by their ruling date in ascending order. The number of results returned can be limited by providing a value for the limit parameter.")
async def get_distinct_card_names_by_rarity_ordered_by_date(rarity: str = Query(..., description="Rarity of the card"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T1.name FROM cards AS T1 INNER JOIN rulings AS T2 ON T1.uuid = T2.uuid WHERE T1.rarity = ? ORDER BY T2.date ASC LIMIT ?", (rarity, limit))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of cards with missing Card Kingdom IDs and a specific artist
@app.get("/v1/card_games/count_cards_missing_card_kingdom_ids_by_artist", operation_id="get_count_cards_missing_card_kingdom_ids_by_artist", summary="Retrieves the total number of cards that lack Card Kingdom IDs and are associated with a specified artist. This operation is useful for identifying gaps in card data for a particular artist.")
async def get_count_cards_missing_card_kingdom_ids_by_artist(artist: str = Query(..., description="Artist of the card")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE (cardKingdomId IS NULL OR cardKingdomFoilId IS NULL) AND artist = ?", (artist,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cards with a specific border color and non-null Card Kingdom IDs
@app.get("/v1/card_games/count_cards_by_border_color_and_non_null_card_kingdom_ids", operation_id="get_count_cards_by_border_color_and_non_null_card_kingdom_ids", summary="Retrieves the total number of cards that have a specified border color and valid Card Kingdom IDs. This operation ensures that the Card Kingdom IDs are not null, providing a count of cards that are fully identified in the Card Kingdom system.")
async def get_count_cards_by_border_color_and_non_null_card_kingdom_ids(border_color: str = Query(..., description="Border color of the card")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE borderColor = ? AND cardKingdomId IS NOT NULL AND cardKingdomFoilId IS NOT NULL", (border_color,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cards with specific hand, artist, and availability
@app.get("/v1/card_games/count_cards_by_hand_artist_availability", operation_id="get_count_cards_by_hand_artist_availability", summary="Retrieves the total number of cards that match a specific hand, artist, and availability. The hand, artist, and availability are provided as input parameters, allowing for a targeted count of cards that meet these criteria.")
async def get_count_cards_by_hand_artist_availability(hand: str = Query(..., description="Hand value of the card"), artist: str = Query(..., description="Artist of the card"), availability: str = Query(..., description="Availability of the card")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE hAND = ? AND artist = ? AND Availability = ?", (hand, artist, availability))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cards with specific frame version, availability, and content warning
@app.get("/v1/card_games/count_cards_by_frame_version_availability_content_warning", operation_id="get_count_cards_by_frame_version_availability_content_warning", summary="Retrieves the total number of cards that match the specified frame version, availability status, and content warning flag. This operation is useful for obtaining a quantitative overview of cards based on these criteria.")
async def get_count_cards_by_frame_version_availability_content_warning(frame_version: int = Query(..., description="Frame version of the card"), availability: str = Query(..., description="Availability of the card"), has_content_warning: int = Query(..., description="Content warning flag of the card")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE frameVersion = ? AND availability = ? AND hasContentWarning = ?", (frame_version, availability, has_content_warning))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get mana costs of cards with specific availability, border color, frame version, and layout
@app.get("/v1/card_games/mana_costs_by_availability_border_color_frame_version_layout", operation_id="get_mana_costs_by_availability_border_color_frame_version_layout", summary="Retrieves the mana costs of cards that meet the specified availability, border color, frame version, and layout criteria. The availability parameter filters cards available in MTGO and paper formats. The border color, frame version, and layout parameters further refine the search to match the provided values.")
async def get_mana_costs_by_availability_border_color_frame_version_layout(availability: str = Query(..., description="Availability of the card"), border_color: str = Query(..., description="Border color of the card"), frame_version: int = Query(..., description="Frame version of the card"), layout: str = Query(..., description="Layout of the card")):
    cursor.execute("SELECT manaCost FROM cards WHERE availability = ? AND borderColor = ? AND frameVersion = ? AND layout = ?", (availability, border_color, frame_version, layout))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get mana costs of cards by a specific artist
@app.get("/v1/card_games/mana_costs_by_artist", operation_id="get_mana_costs_by_artist", summary="Retrieves the mana costs of all cards created by a specified artist. The artist parameter is used to filter the results.")
async def get_mana_costs_by_artist(artist: str = Query(..., description="Artist of the card")):
    cursor.execute("SELECT manaCost FROM cards WHERE artist = ?", (artist,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct subtypes and supertypes of cards with specific availability
@app.get("/v1/card_games/distinct_subtypes_supertypes_by_availability", operation_id="get_distinct_subtypes_supertypes_by_availability", summary="Retrieves the unique combinations of subtypes and supertypes for cards that have a specified availability. The availability parameter is used to filter the cards, ensuring that only cards with the specified availability are considered. This operation does not return cards with null subtypes or supertypes.")
async def get_distinct_subtypes_supertypes_by_availability(availability: str = Query(..., description="Availability of the card")):
    cursor.execute("SELECT DISTINCT subtypes, supertypes FROM cards WHERE availability = ? AND subtypes IS NOT NULL AND supertypes IS NOT NULL", (availability,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get set codes based on language
@app.get("/v1/card_games/set_codes_by_language", operation_id="get_set_codes_by_language", summary="Retrieves a list of unique set codes associated with the specified language. The language parameter is used to filter the set codes, providing a targeted set of results.")
async def get_set_codes_by_language(language: str = Query(..., description="Language of the set translations")):
    cursor.execute("SELECT setCode FROM set_translations WHERE language = ?", (language,))
    result = cursor.fetchall()
    if not result:
        return {"set_codes": []}
    return {"set_codes": [row[0] for row in result]}

# Endpoint to get the percentage of cards with a specific frame effect that are online only
@app.get("/v1/card_games/percentage_online_only_by_frame_effect", operation_id="get_percentage_online_only_by_frame_effect", summary="Retrieves the percentage of cards with a specified frame effect that are exclusively available online. The calculation is based on the total count of cards with the given frame effect.")
async def get_percentage_online_only_by_frame_effect(frame_effect: str = Query(..., description="Frame effect of the cards")):
    cursor.execute("SELECT SUM(CASE WHEN isOnlineOnly = 1 THEN 1.0 ELSE 0 END) / COUNT(id) * 100 FROM cards WHERE frameEffects = ?", (frame_effect,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of cards with story spotlight that are not textless
@app.get("/v1/card_games/percentage_non_textless_story_spotlight", operation_id="get_percentage_non_textless_story_spotlight", summary="Retrieves the percentage of cards with a story spotlight that contain text. The operation filters cards based on their story spotlight status and calculates the proportion of cards that are not textless.")
async def get_percentage_non_textless_story_spotlight(is_story_spotlight: int = Query(..., description="Story spotlight status of the cards (1 for true, 0 for false)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN isTextless = 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(id) FROM cards WHERE isStorySpotlight = ?", (is_story_spotlight,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of foreign data entries in a specific language and their names
@app.get("/v1/card_games/percentage_foreign_data_by_language", operation_id="get_percentage_foreign_data_by_language", summary="Retrieves the percentage of foreign data entries in a specific language and their corresponding names. The language is specified as an input parameter, and the operation calculates the proportion of entries in that language relative to the total number of foreign data entries. The names of the foreign data entries in the specified language are also returned.")
async def get_percentage_foreign_data_by_language(language: str = Query(..., description="Language of the foreign data")):
    cursor.execute("SELECT ( SELECT CAST(SUM(CASE WHEN language = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM foreign_data ), name FROM foreign_data WHERE language = ?", (language, language))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": [{"percentage": row[0], "name": row[1]} for row in result]}

# Endpoint to get languages of set translations based on base set size
@app.get("/v1/card_games/languages_by_base_set_size", operation_id="get_languages_by_base_set_size", summary="Retrieves the languages used for set translations that correspond to a specified base set size. The base set size is a defining attribute of the sets, and this operation filters the set translations based on this attribute.")
async def get_languages_by_base_set_size(base_set_size: int = Query(..., description="Base set size of the sets")):
    cursor.execute("SELECT T2.language FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T1.baseSetSize = ?", (base_set_size,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the count of sets based on language and block
@app.get("/v1/card_games/count_sets_by_language_and_block", operation_id="get_count_sets_by_language_and_block", summary="Retrieves the total number of card game sets that match a specified language and block. The language parameter filters the sets based on the language of their translations, while the block parameter narrows down the sets to a specific block. This operation provides a quantitative overview of the available sets that meet the given language and block criteria.")
async def get_count_sets_by_language_and_block(language: str = Query(..., description="Language of the set translations"), block: str = Query(..., description="Block of the sets")):
    cursor.execute("SELECT COUNT(T1.id) FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = ? AND T1.block = ?", (language, block))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get card IDs based on legal status and type
@app.get("/v1/card_games/card_ids_by_legal_status_and_type", operation_id="get_card_ids_by_legal_status_and_type", summary="Retrieves the IDs of cards that match the specified legal status and type. The legal status is determined by the current rulings and legalities, ensuring that only cards with the given status and type are returned.")
async def get_card_ids_by_legal_status_and_type(status: str = Query(..., description="Legal status of the cards"), types: str = Query(..., description="Type of the cards")):
    cursor.execute("SELECT T1.id FROM cards AS T1 INNER JOIN rulings AS T2 ON T1.uuid = T2.uuid INNER JOIN legalities AS T3 ON T1.uuid = T3.uuid WHERE T3.status = ? AND T1.types = ?", (status, types))
    result = cursor.fetchall()
    if not result:
        return {"card_ids": []}
    return {"card_ids": [row[0] for row in result]}

# Endpoint to get card subtypes and supertypes based on language
@app.get("/v1/card_games/card_subtypes_supertypes_by_language", operation_id="get_card_subtypes_supertypes_by_language", summary="Retrieves the subtypes and supertypes of cards that are associated with a specific language. The language is used to filter the cards, and only those with non-null subtypes and supertypes are included in the response.")
async def get_card_subtypes_supertypes_by_language(language: str = Query(..., description="Language of the foreign data")):
    cursor.execute("SELECT T1.subtypes, T1.supertypes FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T2.language = ? AND T1.subtypes IS NOT NULL AND T1.supertypes IS NOT NULL", (language,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": [{"subtypes": row[0], "supertypes": row[1]} for row in result]}

# Endpoint to get ruling texts based on card power and text pattern
@app.get("/v1/card_games/ruling_texts_by_power_and_text_pattern", operation_id="get_ruling_texts_by_power_and_text_pattern", summary="Get ruling texts based on card power and text pattern")
async def get_ruling_texts_by_power_and_text_pattern(power: str = Query(..., description="Power of the cards (use '*' for wildcard)"), text_pattern: str = Query(..., description="Pattern to match in the ruling text (use '%' for wildcard)")):
    cursor.execute("SELECT T2.text FROM cards AS T1 INNER JOIN rulings AS T2 ON T1.uuid = T2.uuid WHERE (T1.power IS NULL OR T1.power = ?) AND T2.text LIKE ?", (power, text_pattern))
    result = cursor.fetchall()
    if not result:
        return {"ruling_texts": []}
    return {"ruling_texts": [row[0] for row in result]}

# Endpoint to get the count of cards based on format, ruling text, and side
@app.get("/v1/card_games/count_cards_by_format_ruling_text_and_side", operation_id="get_count_cards_by_format_ruling_text_and_side", summary="Get the count of cards based on format, ruling text, and side")
async def get_count_cards_by_format_ruling_text_and_side(format: str = Query(..., description="Format of the legalities"), ruling_text: str = Query(..., description="Text of the rulings"), side: str = Query(None, description="Side of the cards (use NULL for no side)")):
    cursor.execute("SELECT COUNT(T1.id) FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid INNER JOIN rulings AS T3 ON T1.uuid = T3.uuid WHERE T2.format = ? AND T3.text = ? AND T1.Side IS ?", (format, ruling_text, side))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get card IDs based on artist, format, and availability
@app.get("/v1/card_games/card_ids_by_artist_format_availability", operation_id="get_card_ids", summary="Retrieves the unique identifiers of cards that match the specified artist, format, and availability. The artist parameter filters cards by the artist who created them. The format parameter narrows down the results to a specific card format. The availability parameter further refines the search to cards that are currently available.")
async def get_card_ids(artist: str = Query(..., description="Artist of the card"), format: str = Query(..., description="Format of the card"), availability: str = Query(..., description="Availability of the card")):
    cursor.execute("SELECT T1.id FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T1.artist = ? AND T2.format = ? AND T1.availability = ?", (artist, format, availability))
    result = cursor.fetchall()
    if not result:
        return {"card_ids": []}
    return {"card_ids": [row[0] for row in result]}

# Endpoint to get distinct artists based on flavor text
@app.get("/v1/card_games/distinct_artists_by_flavor_text", operation_id="get_distinct_artists", summary="Retrieves a list of unique artists who have created cards with a specified flavor text. The flavor text is used to filter the results, ensuring that only artists who have designed cards with the given flavor text are returned.")
async def get_distinct_artists(flavor_text: str = Query(..., description="Flavor text to search for")):
    cursor.execute("SELECT DISTINCT T1.artist FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T2.flavorText LIKE ?", (flavor_text,))
    result = cursor.fetchall()
    if not result:
        return {"artists": []}
    return {"artists": [row[0] for row in result]}

# Endpoint to get card names based on card attributes and language
@app.get("/v1/card_games/card_names_by_attributes_language", operation_id="get_card_names", summary="Retrieves the names of cards that match the specified attributes and language. The attributes include the card type, layout, border color, and artist. The language parameter determines the language of the card names returned.")
async def get_card_names(types: str = Query(..., description="Type of the card"), layout: str = Query(..., description="Layout of the card"), border_color: str = Query(..., description="Border color of the card"), artist: str = Query(..., description="Artist of the card"), language: str = Query(..., description="Language of the card")):
    cursor.execute("SELECT name FROM foreign_data WHERE uuid IN ( SELECT uuid FROM cards WHERE types = ? AND layout = ? AND borderColor = ? AND artist = ? ) AND language = ?", (types, layout, border_color, artist, language))
    result = cursor.fetchall()
    if not result:
        return {"card_names": []}
    return {"card_names": [row[0] for row in result]}

# Endpoint to get the count of distinct card IDs based on rarity and ruling date
@app.get("/v1/card_games/count_distinct_card_ids_by_rarity_date", operation_id="get_count_distinct_card_ids", summary="Retrieves the total number of unique cards with a specified rarity and ruling date. The rarity parameter filters the cards based on their rarity, while the date parameter narrows down the results to a specific ruling date.")
async def get_count_distinct_card_ids(rarity: str = Query(..., description="Rarity of the card"), date: str = Query(..., description="Ruling date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.id) FROM cards AS T1 INNER JOIN rulings AS T2 ON T1.uuid = T2.uuid WHERE T1.rarity = ? AND T2.date = ?", (rarity, date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get languages of sets based on block and base set size
@app.get("/v1/card_games/set_languages_by_block_size", operation_id="get_set_languages", summary="Retrieves the languages associated with card game sets that belong to a specific block and have a certain base set size. The block and base set size are used to filter the sets, and the languages of the matching sets are returned.")
async def get_set_languages(block: str = Query(..., description="Block of the set"), base_set_size: int = Query(..., description="Base set size")):
    cursor.execute("SELECT T2.language FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T1.block = ? AND T1.baseSetSize = ?", (block, base_set_size))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the percentage of cards without content warning based on format and status
@app.get("/v1/card_games/percentage_no_content_warning_by_format_status", operation_id="get_percentage_no_content_warning", summary="Retrieves the percentage of cards that do not have a content warning, filtered by a specific card format and status. The calculation is based on the total count of cards that meet the specified format and status criteria.")
async def get_percentage_no_content_warning(format: str = Query(..., description="Format of the card"), status: str = Query(..., description="Status of the card")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.hasContentWarning = 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T2.format = ? AND T2.status = ?", (format, status))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of cards in a specific language based on power
@app.get("/v1/card_games/percentage_language_by_power", operation_id="get_percentage_language_by_power", summary="Retrieves the percentage of cards in a specified language that have a certain power level. The calculation is based on the total number of cards in the database. The language and power level are provided as input parameters.")
async def get_percentage_language_by_power(language: str = Query(..., description="Language of the card"), power: str = Query(..., description="Power of the card")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.language = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid WHERE T1.power IS NULL OR T1.power = ?", (language, power))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of sets in a specific language based on set type
@app.get("/v1/card_games/percentage_language_by_set_type", operation_id="get_percentage_language_by_set_type", summary="Retrieves the percentage of sets in a specified language, categorized by set type. The calculation is based on the total count of sets in the given set type.")
async def get_percentage_language_by_set_type(language: str = Query(..., description="Language of the set"), set_type: str = Query(..., description="Type of the set")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.language = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T1.type = ?", (language, set_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct availabilities based on artist
@app.get("/v1/card_games/distinct_availabilities_by_artist", operation_id="get_distinct_availabilities", summary="Retrieves a list of unique availability statuses for cards created by the specified artist.")
async def get_distinct_availabilities(artist: str = Query(..., description="Artist of the card")):
    cursor.execute("SELECT DISTINCT availability FROM cards WHERE artist = ?", (artist,))
    result = cursor.fetchall()
    if not result:
        return {"availabilities": []}
    return {"availabilities": [row[0] for row in result]}

# Endpoint to get the count of cards based on EDHREC rank and border color
@app.get("/v1/card_games/count_cards_by_edhrec_rank_border_color", operation_id="get_count_cards", summary="Retrieves the total number of cards that have an EDHREC rank higher than the specified value and a border color matching the provided color.")
async def get_count_cards(edhrec_rank: int = Query(..., description="EDHREC rank of the card"), border_color: str = Query(..., description="Border color of the card")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE edhrecRank > ? AND borderColor = ?", (edhrec_rank, border_color))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cards based on oversized, reprint, and promo status
@app.get("/v1/card_games/count_cards_by_status", operation_id="get_count_cards_by_status", summary="Retrieves the total count of cards that meet the specified criteria for size, reprint status, and promotional status. The criteria are defined by the input parameters, which determine whether the cards are oversized, reprints, or promos.")
async def get_count_cards_by_status(is_oversized: int = Query(..., description="Is the card oversized (1 for yes, 0 for no)"), is_reprint: int = Query(..., description="Is the card a reprint (1 for yes, 0 for no)"), is_promo: int = Query(..., description="Is the card a promo (1 for yes, 0 for no)")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE isOversized = ? AND isReprint = ? AND isPromo = ?", (is_oversized, is_reprint, is_promo))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get card names based on power and promo type
@app.get("/v1/card_games/card_names_by_power_promo", operation_id="get_card_names_by_power_promo", summary="Retrieves a list of card names that either have no power value or a power value matching the provided pattern. Additionally, the cards must have a specific promo type. The results are ordered by name and limited to the specified number.")
async def get_card_names_by_power_promo(power_pattern: str = Query(..., description="Pattern for power (use %% for wildcard)"), promo_type: str = Query(..., description="Promo type of the card"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT name FROM cards WHERE (power IS NULL OR power LIKE ?) AND promoTypes = ? ORDER BY name LIMIT ?", (power_pattern, promo_type, limit))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the language of foreign data based on multiverse ID
@app.get("/v1/card_games/language_by_multiverseid", operation_id="get_language_by_multiverseid", summary="Retrieves the language associated with a specific card, identified by its multiverse ID. This operation allows you to obtain the language of foreign data related to the card.")
async def get_language_by_multiverseid(multiverseid: int = Query(..., description="Multiverse ID of the card")):
    cursor.execute("SELECT language FROM foreign_data WHERE multiverseid = ?", (multiverseid,))
    result = cursor.fetchone()
    if not result:
        return {"language": []}
    return {"language": result[0]}

# Endpoint to get card IDs from Card Kingdom based on non-null criteria
@app.get("/v1/card_games/card_kingdom_ids", operation_id="get_card_kingdom_ids", summary="Retrieves a limited number of records containing Card Kingdom Foil ID and Card Kingdom ID for cards that have both IDs. The results are ordered by Card Kingdom Foil ID.")
async def get_card_kingdom_ids(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT cardKingdomFoilId, cardKingdomId FROM cards WHERE cardKingdomFoilId IS NOT NULL AND cardKingdomId IS NOT NULL ORDER BY cardKingdomFoilId LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"card_ids": []}
    return {"card_ids": [{"cardKingdomFoilId": row[0], "cardKingdomId": row[1]} for row in result]}

# Endpoint to get the percentage of textless cards with a specific layout
@app.get("/v1/card_games/percentage_textless_cards", operation_id="get_percentage_textless_cards", summary="Retrieves the percentage of cards that do not contain text and have a specific layout. The calculation is based on the total number of cards in the database.")
async def get_percentage_textless_cards(is_textless: int = Query(..., description="Is the card textless (1 for yes, 0 for no)"), layout: str = Query(..., description="Layout of the card")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN isTextless = ? AND layout = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM cards", (is_textless, layout))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get card IDs based on subtypes and null side
@app.get("/v1/card_games/card_ids_by_subtypes", operation_id="get_card_ids_by_subtypes", summary="Retrieves the unique identifiers of cards that belong to the specified subtypes and have no side assigned. The subtypes are provided as a comma-separated list.")
async def get_card_ids_by_subtypes(subtypes: str = Query(..., description="Subtypes of the card (comma-separated)")):
    cursor.execute("SELECT id FROM cards WHERE subtypes = ? AND side IS NULL", (subtypes,))
    result = cursor.fetchall()
    if not result:
        return {"card_ids": []}
    return {"card_ids": [row[0] for row in result]}

# Endpoint to get set names with null MTGO code
@app.get("/v1/card_games/set_names_null_mtgo", operation_id="get_set_names_null_mtgo", summary="Retrieves a list of set names that do not have an associated MTGO code. The results are ordered alphabetically by name and limited to the specified number of entries.")
async def get_set_names_null_mtgo(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT name FROM sets WHERE mtgoCode IS NULL ORDER BY name LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"set_names": []}
    return {"set_names": [row[0] for row in result]}

# Endpoint to get languages of set translations based on MCM name and set code
@app.get("/v1/card_games/set_translation_languages", operation_id="get_set_translation_languages", summary="Retrieves the languages of set translations for a specific MCM name and set code. This operation fetches the languages from the database by matching the provided MCM name and set code with the corresponding entries in the sets and set_translations tables.")
async def get_set_translation_languages(mcm_name: str = Query(..., description="MCM name of the set"), set_code: str = Query(..., description="Set code of the translation")):
    cursor.execute("SELECT T2.language FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T1.mcmName = ? AND T2.setCode = ?", (mcm_name, set_code))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get set names and translations based on translation ID
@app.get("/v1/card_games/set_names_translations", operation_id="get_set_names_translations", summary="Retrieves the names and translations of all card sets associated with a specific translation ID. The operation returns a list of unique set names and their corresponding translations, grouped by name and translation.")
async def get_set_names_translations(translation_id: int = Query(..., description="Translation ID")):
    cursor.execute("SELECT T1.name, T2.translation FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.id = ? GROUP BY T1.name, T2.translation", (translation_id,))
    result = cursor.fetchall()
    if not result:
        return {"set_translations": []}
    return {"set_translations": [{"name": row[0], "translation": row[1]} for row in result]}

# Endpoint to get languages and types of sets based on translation ID
@app.get("/v1/card_games/set_languages_types", operation_id="get_set_languages_types", summary="Retrieves the languages and types of sets associated with a specific translation ID. The operation fetches this information from the sets and set_translations tables, filtering results based on the provided translation ID.")
async def get_set_languages_types(translation_id: int = Query(..., description="Translation ID")):
    cursor.execute("SELECT T2.language, T1.type FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.id = ?", (translation_id,))
    result = cursor.fetchall()
    if not result:
        return {"set_languages_types": []}
    return {"set_languages_types": [{"language": row[0], "type": row[1]} for row in result]}

# Endpoint to get set names and IDs based on block and language with a limit
@app.get("/v1/card_games/sets_by_block_language", operation_id="get_sets_by_block_language", summary="Retrieves a limited number of set names and their corresponding IDs based on the specified block and language. The results are ordered by set ID.")
async def get_sets_by_block_language(block: str = Query(..., description="Block name"), language: str = Query(..., description="Language"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT T1.name, T1.id FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T1.block = ? AND T2.language = ? ORDER BY T1.id LIMIT ?", (block, language, limit))
    result = cursor.fetchall()
    if not result:
        return {"sets": []}
    return {"sets": result}

# Endpoint to get set names and IDs based on language, foil only, and foreign only status
@app.get("/v1/card_games/sets_by_language_foil_foreign", operation_id="get_sets_by_language_foil_foreign", summary="Retrieves the names and unique identifiers of card game sets that match the specified language, foil-only, and foreign-only criteria. The operation filters the sets based on the provided language and whether the sets are exclusively foil or foreign, returning only those that meet all specified conditions.")
async def get_sets_by_language_foil_foreign(language: str = Query(..., description="Language"), is_foil_only: int = Query(..., description="Is foil only (1 for true, 0 for false)"), is_foreign_only: int = Query(..., description="Is foreign only (1 for true, 0 for false)")):
    cursor.execute("SELECT T1.name, T1.id FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = ? AND T1.isFoilOnly = ? AND T1.isForeignOnly = ?", (language, is_foil_only, is_foreign_only))
    result = cursor.fetchall()
    if not result:
        return {"sets": []}
    return {"sets": result}

# Endpoint to get set IDs based on language and limit, grouped by base set size
@app.get("/v1/card_games/sets_by_language_grouped_by_base_set_size", operation_id="get_sets_by_language_grouped_by_base_set_size", summary="Retrieves a list of set IDs based on the specified language, grouped by the base set size. The results are ordered in descending order based on the base set size and limited to the specified number of results.")
async def get_sets_by_language_grouped_by_base_set_size(language: str = Query(..., description="Language"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT T1.id FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = ? GROUP BY T1.baseSetSize ORDER BY T1.baseSetSize DESC LIMIT ?", (language, limit))
    result = cursor.fetchall()
    if not result:
        return {"sets": []}
    return {"sets": result}

# Endpoint to get the percentage of sets that are online only and in a specific language
@app.get("/v1/card_games/percentage_online_only_sets_by_language", operation_id="get_percentage_online_only_sets_by_language", summary="Retrieves the percentage of card game sets that are exclusively available online and are in a specified language. The calculation is based on the total number of sets in the database.")
async def get_percentage_online_only_sets_by_language(language: str = Query(..., description="Language"), is_online_only: int = Query(..., description="Is online only (1 for true, 0 for false)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.language = ? AND T1.isOnlineOnly = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode", (language, is_online_only))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of sets based on language and MTGO code
@app.get("/v1/card_games/count_sets_by_language_mtgo_code", operation_id="get_count_sets_by_language_mtgo_code", summary="Retrieves the total number of card game sets that match the specified language and MTGO code. The language parameter filters the sets by their translated language, while the MTGO code parameter allows for filtering based on the MTGO code, including an option for sets without an MTGO code.")
async def get_count_sets_by_language_mtgo_code(language: str = Query(..., description="Language"), mtgo_code: str = Query(..., description="MTGO code (empty string for NULL)")):
    cursor.execute("SELECT COUNT(T1.id) FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T2.language = ? AND (T1.mtgoCode IS NULL OR T1.mtgoCode = ?)", (language, mtgo_code))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get card IDs based on border color
@app.get("/v1/card_games/cards_by_border_color", operation_id="get_cards_by_border_color", summary="Retrieves a list of unique card identifiers that share a specified border color. The border color is provided as an input parameter.")
async def get_cards_by_border_color(border_color: str = Query(..., description="Border color")):
    cursor.execute("SELECT id FROM cards WHERE borderColor = ? GROUP BY id", (border_color,))
    result = cursor.fetchall()
    if not result:
        return {"cards": []}
    return {"cards": result}

# Endpoint to get card IDs based on frame effects
@app.get("/v1/card_games/cards_by_frame_effects", operation_id="get_cards_by_frame_effects", summary="Retrieves a list of unique card identifiers that share a specified frame effect. The input parameter determines the frame effect to filter the cards by.")
async def get_cards_by_frame_effects(frame_effects: str = Query(..., description="Frame effects")):
    cursor.execute("SELECT id FROM cards WHERE frameEffects = ? GROUP BY id", (frame_effects,))
    result = cursor.fetchall()
    if not result:
        return {"cards": []}
    return {"cards": result}

# Endpoint to get card IDs based on border color and full art status
@app.get("/v1/card_games/cards_by_border_color_full_art", operation_id="get_cards_by_border_color_full_art", summary="Retrieves the IDs of cards that have a specified border color and are designated as full art. The border color and full art status are provided as input parameters.")
async def get_cards_by_border_color_full_art(border_color: str = Query(..., description="Border color"), is_full_art: int = Query(..., description="Is full art (1 for true, 0 for false)")):
    cursor.execute("SELECT id FROM cards WHERE borderColor = ? AND isFullArt = ?", (border_color, is_full_art))
    result = cursor.fetchall()
    if not result:
        return {"cards": []}
    return {"cards": result}

# Endpoint to get the language of a set translation by ID
@app.get("/v1/card_games/set_translation_language_by_id", operation_id="get_set_translation_language_by_id", summary="Retrieves the language of a specific set translation, identified by its unique ID. This operation allows you to determine the language used for a particular set translation, providing essential information for managing multilingual card game sets.")
async def get_set_translation_language_by_id(id: int = Query(..., description="ID of the set translation")):
    cursor.execute("SELECT language FROM set_translations WHERE id = ?", (id,))
    result = cursor.fetchone()
    if not result:
        return {"language": []}
    return {"language": result[0]}

# Endpoint to get the name of a set by code
@app.get("/v1/card_games/set_name_by_code", operation_id="get_set_name_by_code", summary="Retrieves the name of a specific card game set identified by its unique code.")
async def get_set_name_by_code(code: str = Query(..., description="Code of the set")):
    cursor.execute("SELECT name FROM sets WHERE code = ?", (code,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get distinct languages for a given name
@app.get("/v1/card_games/distinct_languages_by_name", operation_id="get_distinct_languages", summary="Retrieves a list of distinct languages associated with a specific card name. The operation filters the available languages based on the provided card name, ensuring that only unique language entries are returned.")
async def get_distinct_languages(name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT DISTINCT language FROM foreign_data WHERE name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get set codes for a given release date
@app.get("/v1/card_games/set_codes_by_release_date", operation_id="get_set_codes_by_release_date", summary="Retrieves the set codes for all card sets released on a specific date. The date should be provided in the 'YYYY-MM-DD' format.")
async def get_set_codes_by_release_date(release_date: str = Query(..., description="Release date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.setCode FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T1.releaseDate = ?", (release_date,))
    result = cursor.fetchall()
    if not result:
        return {"set_codes": []}
    return {"set_codes": [row[0] for row in result]}

# Endpoint to get distinct base set sizes and set codes for given blocks
@app.get("/v1/card_games/distinct_base_set_sizes_and_set_codes_by_blocks", operation_id="get_distinct_base_set_sizes_and_set_codes", summary="Retrieves unique combinations of base set sizes and set codes associated with the provided block names. This operation returns a list of distinct base set sizes and their corresponding set codes from the sets table, filtered by the specified block names. The block names are used to join the sets and set_translations tables, ensuring accurate and relevant results.")
async def get_distinct_base_set_sizes_and_set_codes(block1: str = Query(..., description="First block name"), block2: str = Query(..., description="Second block name")):
    cursor.execute("SELECT DISTINCT T1.baseSetSize, T2.setCode FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T1.block IN (?, ?)", (block1, block2))
    result = cursor.fetchall()
    if not result:
        return {"base_set_sizes_and_set_codes": []}
    return {"base_set_sizes_and_set_codes": [{"baseSetSize": row[0], "setCode": row[1]} for row in result]}

# Endpoint to get set codes for a given set type
@app.get("/v1/card_games/set_codes_by_set_type", operation_id="get_set_codes_by_set_type", summary="Retrieves a list of unique set codes associated with the specified set type. The set type is used to filter the results, ensuring that only relevant set codes are returned. This operation is useful for obtaining a comprehensive overview of the available set codes for a particular set type.")
async def get_set_codes_by_set_type(set_type: str = Query(..., description="Type of the set")):
    cursor.execute("SELECT T2.setCode FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T1.type = ? GROUP BY T2.setCode", (set_type,))
    result = cursor.fetchall()
    if not result:
        return {"set_codes": []}
    return {"set_codes": [row[0] for row in result]}

# Endpoint to get distinct card names and types for a given watermark
@app.get("/v1/card_games/distinct_card_names_and_types_by_watermark", operation_id="get_distinct_card_names_and_types", summary="Retrieves a unique list of card names and types that match a specified watermark. This operation is useful for identifying the distinct cards associated with a particular watermark.")
async def get_distinct_card_names_and_types(watermark: str = Query(..., description="Watermark of the card")):
    cursor.execute("SELECT DISTINCT T1.name, T1.type FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T2.uuid = T1.uuid WHERE T1.watermark = ?", (watermark,))
    result = cursor.fetchall()
    if not result:
        return {"card_names_and_types": []}
    return {"card_names_and_types": [{"name": row[0], "type": row[1]} for row in result]}

# Endpoint to get distinct languages and flavor texts for a given watermark
@app.get("/v1/card_games/distinct_languages_and_flavor_texts_by_watermark", operation_id="get_distinct_languages_and_flavor_texts", summary="Retrieves the unique combinations of languages and flavor texts associated with a specific watermark.")
async def get_distinct_languages_and_flavor_texts(watermark: str = Query(..., description="Watermark of the card")):
    cursor.execute("SELECT DISTINCT T2.language, T2.flavorText FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T2.uuid = T1.uuid WHERE T1.watermark = ?", (watermark,))
    result = cursor.fetchall()
    if not result:
        return {"languages_and_flavor_texts": []}
    return {"languages_and_flavor_texts": [{"language": row[0], "flavorText": row[1]} for row in result]}

# Endpoint to get the percentage of cards with a specific converted mana cost for a given card name
@app.get("/v1/card_games/percentage_cards_by_mana_cost_and_name", operation_id="get_percentage_cards_by_mana_cost_and_name", summary="Retrieves the percentage of cards with a specified converted mana cost for a given card name. This operation calculates the proportion of cards that match the provided converted mana cost out of all cards with the same name. The result is expressed as a percentage.")
async def get_percentage_cards_by_mana_cost_and_name(converted_mana_cost: int = Query(..., description="Converted mana cost of the card"), name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.convertedManaCost = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id), T1.name FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T1.name = ?", (converted_mana_cost, name))
    result = cursor.fetchall()
    if not result:
        return {"percentage_and_name": []}
    return {"percentage_and_name": [{"percentage": row[0], "name": row[1]} for row in result]}

# Endpoint to get distinct languages and card types for a given watermark
@app.get("/v1/card_games/distinct_languages_and_card_types_by_watermark", operation_id="get_distinct_languages_and_card_types", summary="Retrieves the unique combinations of languages and card types associated with a specific watermark. The watermark parameter is used to filter the results.")
async def get_distinct_languages_and_card_types(watermark: str = Query(..., description="Watermark of the card")):
    cursor.execute("SELECT DISTINCT T2.language, T1.type FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T2.uuid = T1.uuid WHERE T1.watermark = ?", (watermark,))
    result = cursor.fetchall()
    if not result:
        return {"languages_and_card_types": []}
    return {"languages_and_card_types": [{"language": row[0], "type": row[1]} for row in result]}

# Endpoint to get the sum of cards with a specific artist and non-null cardKingdomFoilId and cardKingdomId
@app.get("/v1/card_games/sum_cards_by_artist", operation_id="get_sum_cards_by_artist", summary="Retrieves the total count of cards associated with a specific artist, considering only those cards that have both a non-null cardKingdomFoilId and cardKingdomId.")
async def get_sum_cards_by_artist(artist: str = Query(..., description="Artist name")):
    cursor.execute("SELECT SUM(CASE WHEN artist = ? AND cardKingdomFoilId IS NOT NULL AND cardKingdomId IS NOT NULL THEN 1 ELSE 0 END) FROM cards", (artist,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of cards with a specific availability and hand value
@app.get("/v1/card_games/sum_cards_by_availability_hand", operation_id="get_sum_cards_by_availability_hand", summary="Retrieves the total count of cards that match a specified availability status and hand value. The availability status and hand value are provided as input parameters.")
async def get_sum_cards_by_availability_hand(availability: str = Query(..., description="Availability type"), hand: str = Query(..., description="Hand value")):
    cursor.execute("SELECT SUM(CASE WHEN availability = ? AND hAND = ? THEN 1 ELSE 0 END) FROM cards", (availability, hand))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct card names based on textless status
@app.get("/v1/card_games/distinct_card_names_by_textless", operation_id="get_distinct_card_names_by_textless", summary="Retrieves a list of unique card names that match the specified textless status. The textless status indicates whether the card has text or not.")
async def get_distinct_card_names_by_textless(is_textless: int = Query(..., description="Textless status (0 or 1)")):
    cursor.execute("SELECT DISTINCT name FROM cards WHERE isTextless = ?", (is_textless,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get distinct mana costs for a specific card name
@app.get("/v1/card_games/distinct_mana_costs_by_name", operation_id="get_distinct_mana_costs_by_name", summary="Retrieves the unique mana costs associated with a specific card name. This operation allows you to identify the distinct mana costs required to cast a particular card, providing valuable insights for game strategy and resource management.")
async def get_distinct_mana_costs_by_name(name: str = Query(..., description="Card name")):
    cursor.execute("SELECT DISTINCT manaCost FROM cards WHERE name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"manaCosts": []}
    return {"manaCosts": [row[0] for row in result]}

# Endpoint to get the sum of cards with a specific power pattern and border color
@app.get("/v1/card_games/sum_cards_by_power_border_color", operation_id="get_sum_cards_by_power_border_color", summary="Retrieves the total count of cards that match a specified power pattern and border color. The power pattern can include wildcard characters to broaden the search. The border color must be an exact match.")
async def get_sum_cards_by_power_border_color(power_pattern: str = Query(..., description="Power pattern (use %% for wildcard)"), border_color: str = Query(..., description="Border color")):
    cursor.execute("SELECT SUM(CASE WHEN power LIKE ? OR power IS NULL THEN 1 ELSE 0 END) FROM cards WHERE borderColor = ?", (power_pattern, border_color))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct card names based on promo status and non-null side
@app.get("/v1/card_games/distinct_card_names_by_promo_side", operation_id="get_distinct_card_names_by_promo_side", summary="Retrieves a list of unique card names that have a specified promo status and a defined side. The promo status is a binary value indicating whether the card is a promotional item or not.")
async def get_distinct_card_names_by_promo_side(is_promo: int = Query(..., description="Promo status (0 or 1)")):
    cursor.execute("SELECT DISTINCT name FROM cards WHERE isPromo = ? AND side IS NOT NULL", (is_promo,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get distinct subtypes and supertypes for a specific card name
@app.get("/v1/card_games/distinct_subtypes_supertypes_by_name", operation_id="get_distinct_subtypes_supertypes_by_name", summary="Retrieves the unique subtypes and supertypes associated with a specific card name. This operation allows you to identify the distinct categories and classifications that a card belongs to, based on the provided card name.")
async def get_distinct_subtypes_supertypes_by_name(name: str = Query(..., description="Card name")):
    cursor.execute("SELECT DISTINCT subtypes, supertypes FROM cards WHERE name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"subtypes_supertypes": []}
    return {"subtypes_supertypes": [{"subtypes": row[0], "supertypes": row[1]} for row in result]}

# Endpoint to get distinct purchase URLs for a specific promo type
@app.get("/v1/card_games/distinct_purchase_urls_by_promo_type", operation_id="get_distinct_purchase_urls_by_promo_type", summary="Retrieves a unique set of purchase URLs associated with a specific promotional type. This operation allows you to obtain a comprehensive list of distinct purchase URLs that correspond to the provided promotional type, enabling efficient analysis and management of promotional campaigns.")
async def get_distinct_purchase_urls_by_promo_type(promo_type: str = Query(..., description="Promo type")):
    cursor.execute("SELECT DISTINCT purchaseUrls FROM cards WHERE promoTypes = ?", (promo_type,))
    result = cursor.fetchall()
    if not result:
        return {"purchaseUrls": []}
    return {"purchaseUrls": [row[0] for row in result]}

# Endpoint to get the count of cards with a specific availability pattern and border color
@app.get("/v1/card_games/count_cards_by_availability_border_color", operation_id="get_count_cards_by_availability_border_color", summary="Retrieves the total number of cards that meet the specified availability criteria and border color. The availability pattern can include wildcards for flexible matching. The border color must be an exact match.")
async def get_count_cards_by_availability_border_color(availability_pattern: str = Query(..., description="Availability pattern (use %% for wildcard)"), border_color: str = Query(..., description="Border color")):
    cursor.execute("SELECT COUNT(CASE WHEN availability LIKE ? AND borderColor = ? THEN 1 ELSE NULL END) FROM cards", (availability_pattern, border_color))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the card name with the highest converted mana cost from a list of names
@app.get("/v1/card_games/card_name_highest_mana_cost", operation_id="get_card_name_highest_mana_cost", summary="Retrieves the name of the card with the highest converted mana cost from a provided list of card names. The list of card names is used to filter the results, and the card with the highest converted mana cost is returned.")
async def get_card_name_highest_mana_cost(name1: str = Query(..., description="First card name"), name2: str = Query(..., description="Second card name")):
    cursor.execute("SELECT name FROM cards WHERE name IN (?, ?) ORDER BY convertedManaCost DESC LIMIT 1", (name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the artist of a card based on its flavor name
@app.get("/v1/card_games/artist_by_flavor_name", operation_id="get_artist_by_flavor_name", summary="Retrieves the artist associated with a card that has the specified flavor name. The flavor name is used to identify the card and locate the corresponding artist.")
async def get_artist_by_flavor_name(flavor_name: str = Query(..., description="Flavor name of the card")):
    cursor.execute("SELECT artist FROM cards WHERE flavorName = ?", (flavor_name,))
    result = cursor.fetchall()
    if not result:
        return {"artists": []}
    return {"artists": [row[0] for row in result]}

# Endpoint to get card names based on frame version and limit
@app.get("/v1/card_games/card_names_by_frame_version", operation_id="get_card_names_by_frame_version", summary="Retrieves a list of card names that match a specified frame version, sorted by their converted mana cost in descending order. The number of results returned is limited by the provided limit parameter.")
async def get_card_names_by_frame_version(frame_version: int = Query(..., description="Frame version of the card"), limit: int = Query(..., description="Limit of the number of results")):
    cursor.execute("SELECT name FROM cards WHERE frameVersion = ? ORDER BY convertedManaCost DESC LIMIT ?", (frame_version, limit))
    result = cursor.fetchall()
    if not result:
        return {"card_names": []}
    return {"card_names": [row[0] for row in result]}

# Endpoint to get translations of a card set based on card name and language
@app.get("/v1/card_games/translations_by_card_name_and_language", operation_id="get_translations_by_card_name_and_language", summary="Retrieves the translations of a card set associated with a specific card name and language. The card name and desired language are provided as input parameters to filter the results.")
async def get_translations_by_card_name_and_language(card_name: str = Query(..., description="Name of the card"), language: str = Query(..., description="Language of the translation")):
    cursor.execute("SELECT translation FROM set_translations WHERE setCode IN ( SELECT setCode FROM cards WHERE name = ? ) AND language = ?", (card_name, language))
    result = cursor.fetchall()
    if not result:
        return {"translations": []}
    return {"translations": [row[0] for row in result]}

# Endpoint to get the count of distinct translations of a card set based on card name
@app.get("/v1/card_games/count_distinct_translations_by_card_name", operation_id="get_count_distinct_translations_by_card_name", summary="Retrieves the number of unique translations for a specific card set, based on the provided card name. The card name is used to identify the relevant card set, and only non-null translations are considered in the count.")
async def get_count_distinct_translations_by_card_name(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT COUNT(DISTINCT translation) FROM set_translations WHERE setCode IN ( SELECT setCode FROM cards WHERE name = ? ) AND translation IS NOT NULL", (card_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct card names based on translation
@app.get("/v1/card_games/distinct_card_names_by_translation", operation_id="get_distinct_card_names_by_translation", summary="Retrieves a list of unique card names from a specific card set translation. The translation parameter is used to filter the card set, ensuring that only cards from the specified translation are considered. The operation returns a distinct set of card names, eliminating any duplicates.")
async def get_distinct_card_names_by_translation(translation: str = Query(..., description="Translation of the card set")):
    cursor.execute("SELECT DISTINCT T1.name FROM cards AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.setCode WHERE T2.translation = ?", (translation,))
    result = cursor.fetchall()
    if not result:
        return {"card_names": []}
    return {"card_names": [row[0] for row in result]}

# Endpoint to check if a card has a translation in a specific language
@app.get("/v1/card_games/check_translation_by_language_and_card_name", operation_id="check_translation_by_language_and_card_name", summary="Verifies if a card has a translation available in a specified language. The operation checks the card's set for translations in the provided language and returns 'YES' if a translation exists or 'NO' otherwise. The card is identified by its name.")
async def check_translation_by_language_and_card_name(language: str = Query(..., description="Language of the translation"), card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT IIF(SUM(CASE WHEN T2.language = ? AND T2.translation IS NOT NULL THEN 1 ELSE 0 END) > 0, 'YES', 'NO') FROM cards AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.setCode WHERE T1.name = ?", (language, card_name))
    result = cursor.fetchone()
    if not result:
        return {"has_translation": []}
    return {"has_translation": result[0]}

# Endpoint to get the count of cards based on translation and artist
@app.get("/v1/card_games/count_cards_by_translation_and_artist", operation_id="get_count_cards_by_translation_and_artist", summary="Retrieves the total number of cards associated with a specific card set translation and artist. This operation allows you to determine the quantity of cards that match the provided translation and artist parameters.")
async def get_count_cards_by_translation_and_artist(translation: str = Query(..., description="Translation of the card set"), artist: str = Query(..., description="Artist of the card")):
    cursor.execute("SELECT COUNT(T1.id) FROM cards AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.setCode WHERE T2.translation = ? AND T1.artist = ?", (translation, artist))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the base set size based on translation
@app.get("/v1/card_games/base_set_size_by_translation", operation_id="get_base_set_size_by_translation", summary="Retrieves the base set size for a card set based on the provided translation. The translation parameter is used to identify the specific card set and return its corresponding base set size.")
async def get_base_set_size_by_translation(translation: str = Query(..., description="Translation of the card set")):
    cursor.execute("SELECT T1.baseSetSize FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T2.translation = ?", (translation,))
    result = cursor.fetchall()
    if not result:
        return {"base_set_sizes": []}
    return {"base_set_sizes": [row[0] for row in result]}

# Endpoint to get translations of a set based on set name and language
@app.get("/v1/card_games/translations_by_set_name_and_language", operation_id="get_translations_by_set_name_and_language", summary="Retrieves the translations of a specific card game set based on the provided set name and language. The operation fetches the translations from the database by matching the set name and language with the corresponding entries in the sets and set_translations tables.")
async def get_translations_by_set_name_and_language(set_name: str = Query(..., description="Name of the set"), language: str = Query(..., description="Language of the translation")):
    cursor.execute("SELECT T2.translation FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T1.name = ? AND T2.language = ?", (set_name, language))
    result = cursor.fetchall()
    if not result:
        return {"translations": []}
    return {"translations": [row[0] for row in result]}

# Endpoint to check if a card has an MTGO code
@app.get("/v1/card_games/check_mtgo_code_by_card_name", operation_id="check_mtgo_code_by_card_name", summary="Check if a card has an MTGO code")
async def check_mtgo_code_by_card_name(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT IIF(T2.mtgoCode IS NOT NULL, 'YES', 'NO') FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T1.name = ?", (card_name,))
    result = cursor.fetchone()
    if not result:
        return {"has_mtgo_code": []}
    return {"has_mtgo_code": result[0]}

# Endpoint to get distinct release dates for a specific card name
@app.get("/v1/card_games/distinct_release_dates_by_card_name", operation_id="get_distinct_release_dates", summary="Retrieves the unique release dates for a specified card. The card is identified by its name, and the operation returns a list of distinct release dates associated with that card.")
async def get_distinct_release_dates(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT DISTINCT T2.releaseDate FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T1.name = ?", (card_name,))
    result = cursor.fetchall()
    if not result:
        return {"release_dates": []}
    return {"release_dates": [row[0] for row in result]}

# Endpoint to get set types based on translation
@app.get("/v1/card_games/set_types_by_translation", operation_id="get_set_types_by_translation", summary="Retrieves the types of card sets that match a provided translation. The translation parameter is used to filter the sets and return their respective types.")
async def get_set_types_by_translation(translation: str = Query(..., description="Translation of the set")):
    cursor.execute("SELECT T1.type FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T2.translation = ?", (translation,))
    result = cursor.fetchall()
    if not result:
        return {"set_types": []}
    return {"set_types": [row[0] for row in result]}

# Endpoint to get the count of distinct set IDs based on block and language
@app.get("/v1/card_games/count_distinct_set_ids_by_block_and_language", operation_id="get_count_distinct_set_ids", summary="Retrieves the total number of unique set IDs that belong to a specified block and are available in a given language. The count is based on the sets that have a valid translation in the specified language.")
async def get_count_distinct_set_ids(block: str = Query(..., description="Block of the set"), language: str = Query(..., description="Language of the set translation")):
    cursor.execute("SELECT COUNT(DISTINCT T1.id) FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T1.block = ? AND T2.language = ? AND T2.translation IS NOT NULL", (block, language))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to check if a card is foreign only
@app.get("/v1/card_games/is_foreign_only_by_card_name", operation_id="get_is_foreign_only", summary="Check if a card is foreign only based on its name")
async def get_is_foreign_only(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT IIF(isForeignOnly = 1, 'YES', 'NO') FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T1.name = ?", (card_name,))
    result = cursor.fetchone()
    if not result:
        return {"is_foreign_only": []}
    return {"is_foreign_only": result[0]}

# Endpoint to get the count of sets based on translation, base set size, and language
@app.get("/v1/card_games/count_sets_by_translation_base_set_size_language", operation_id="get_count_sets", summary="Retrieves the total number of card sets that meet the specified criteria. The criteria include having a translation, a base set size less than the provided value, and a set translation language matching the provided language.")
async def get_count_sets(base_set_size: int = Query(..., description="Base set size"), language: str = Query(..., description="Language of the set translation")):
    cursor.execute("SELECT COUNT(T1.id) FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T2.translation IS NOT NULL AND T1.baseSetSize < ? AND T2.language = ?", (base_set_size, language))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cards with a specific border color in a given set
@app.get("/v1/card_games/count_cards_by_border_color_and_set_name", operation_id="get_count_cards_by_border_color", summary="Retrieves the total number of cards with a specified border color from a particular set. The operation filters cards based on the provided border color and set name, then calculates the sum of matching cards.")
async def get_count_cards_by_border_color(border_color: str = Query(..., description="Border color of the card"), set_name: str = Query(..., description="Name of the set")):
    cursor.execute("SELECT SUM(CASE WHEN T1.borderColor = ? THEN 1 ELSE 0 END) FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = ?", (border_color, set_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the card with the highest converted mana cost in a given set
@app.get("/v1/card_games/highest_mana_cost_card_by_set_name", operation_id="get_highest_mana_cost_card", summary="Retrieves the card with the highest mana cost from the specified set. The operation filters cards by set name and sorts them in descending order based on their converted mana cost. The card with the highest mana cost is then returned.")
async def get_highest_mana_cost_card(set_name: str = Query(..., description="Name of the set")):
    cursor.execute("SELECT T1.name FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = ? ORDER BY T1.convertedManaCost DESC LIMIT 1", (set_name,))
    result = cursor.fetchone()
    if not result:
        return {"card_name": []}
    return {"card_name": result[0]}

# Endpoint to get artists of cards in a given set with specific artists
@app.get("/v1/card_games/artists_by_set_name_and_artists", operation_id="get_artists_by_set_name_and_artists", summary="Retrieves the artists of cards in a specified set that match up to three provided artist names. The results are grouped by artist.")
async def get_artists_by_set_name_and_artists(set_name: str = Query(..., description="Name of the set"), artist1: str = Query(..., description="First artist name"), artist2: str = Query(..., description="Second artist name"), artist3: str = Query(..., description="Third artist name")):
    cursor.execute("SELECT T1.artist FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = ? AND (T1.artist = ? OR T1.artist = ? OR T1.artist = ?) GROUP BY T1.artist", (set_name, artist1, artist2, artist3))
    result = cursor.fetchall()
    if not result:
        return {"artists": []}
    return {"artists": [row[0] for row in result]}

# Endpoint to get the card name by set name and card number
@app.get("/v1/card_games/card_name_by_set_name_and_number", operation_id="get_card_name_by_set_name_and_number", summary="Retrieves the name of a specific card from a given set based on the provided set name and card number. The operation uses the set name and card number to identify the card and return its name.")
async def get_card_name_by_set_name_and_number(set_name: str = Query(..., description="Name of the set"), card_number: int = Query(..., description="Number of the card")):
    cursor.execute("SELECT T1.name FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = ? AND T1.number = ?", (set_name, card_number))
    result = cursor.fetchone()
    if not result:
        return {"card_name": []}
    return {"card_name": result[0]}

# Endpoint to get the count of cards with variable power in a given set and mana cost
@app.get("/v1/card_games/count_variable_power_cards_by_set_name_and_mana_cost", operation_id="get_count_variable_power_cards", summary="Retrieves the total number of cards in a specified set that have a variable power attribute and a mana cost greater than the provided value. The set is identified by its name, and the mana cost is a numerical value representing the card's converted mana cost.")
async def get_count_variable_power_cards(set_name: str = Query(..., description="Name of the set"), mana_cost: int = Query(..., description="Converted mana cost of the card")):
    cursor.execute("SELECT SUM(CASE WHEN T1.power LIKE '*' OR T1.power IS NULL THEN 1 ELSE 0 END) FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = ? AND T1.convertedManaCost > ?", (set_name, mana_cost))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the flavor text of a card in a specific language
@app.get("/v1/card_games/flavor_text_by_card_name_and_language", operation_id="get_flavor_text", summary="Retrieves the flavor text of a specific card in the requested language. The operation requires the card's name and the desired language as input parameters. The flavor text is fetched from a database where it is associated with the card's unique identifier and language.")
async def get_flavor_text(card_name: str = Query(..., description="Name of the card"), language: str = Query(..., description="Language of the flavor text")):
    cursor.execute("SELECT T2.flavorText FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T2.uuid = T1.uuid WHERE T1.name = ? AND T2.language = ?", (card_name, language))
    result = cursor.fetchall()
    if not result:
        return {"flavor_text": []}
    return {"flavor_text": [row[0] for row in result]}

# Endpoint to get the languages of a card with non-null flavor text
@app.get("/v1/card_games/languages_by_card_name_with_flavor_text", operation_id="get_languages_with_flavor_text", summary="Retrieves the languages associated with a specific card that has flavor text. The card is identified by its name. The response includes languages for which the card has non-null flavor text.")
async def get_languages_with_flavor_text(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT T2.language FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T2.uuid = T1.uuid WHERE T1.name = ? AND T2.flavorText IS NOT NULL", (card_name,))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the distinct types of a card in a specific language
@app.get("/v1/card_games/distinct_types_by_card_name_and_language", operation_id="get_distinct_types", summary="Retrieve the unique types of a card with a specific name in a given language. The operation filters cards by name and language, then returns the distinct types associated with the matching cards.")
async def get_distinct_types(card_name: str = Query(..., description="Name of the card"), language: str = Query(..., description="Language of the card")):
    cursor.execute("SELECT DISTINCT T1.type FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T2.uuid = T1.uuid WHERE T1.name = ? AND T2.language = ?", (card_name, language))
    result = cursor.fetchall()
    if not result:
        return {"types": []}
    return {"types": [row[0] for row in result]}

# Endpoint to get the distinct texts of cards in a specific set and language
@app.get("/v1/card_games/distinct_texts_by_set_and_language", operation_id="get_distinct_texts", summary="Retrieve the unique card texts from a specified set and language. This operation filters cards based on the provided set name and language, ensuring that only distinct texts are returned.")
async def get_distinct_texts(set_name: str = Query(..., description="Name of the set"), language: str = Query(..., description="Language of the card text")):
    cursor.execute("SELECT DISTINCT T1.text FROM foreign_data AS T1 INNER JOIN cards AS T2 ON T2.uuid = T1.uuid INNER JOIN sets AS T3 ON T3.code = T2.setCode WHERE T3.name = ? AND T1.language = ?", (set_name, language))
    result = cursor.fetchall()
    if not result:
        return {"texts": []}
    return {"texts": [row[0] for row in result]}

# Endpoint to get the names of cards in a specific set and language, ordered by converted mana cost
@app.get("/v1/card_games/card_names_by_set_and_language_ordered_by_mana_cost", operation_id="get_card_names_ordered_by_mana_cost", summary="Retrieves the names of cards from a specified set and language, sorted by their converted mana cost in descending order. The set and language are provided as input parameters, allowing for a targeted search of cards that meet these criteria.")
async def get_card_names_ordered_by_mana_cost(set_name: str = Query(..., description="Name of the set"), language: str = Query(..., description="Language of the card")):
    cursor.execute("SELECT T2.name FROM foreign_data AS T1 INNER JOIN cards AS T2 ON T2.uuid = T1.uuid INNER JOIN sets AS T3 ON T3.code = T2.setCode WHERE T3.name = ? AND T1.language = ? ORDER BY T2.convertedManaCost DESC", (set_name, language))
    result = cursor.fetchall()
    if not result:
        return {"card_names": []}
    return {"card_names": [row[0] for row in result]}

# Endpoint to get the ruling dates of a specific card
@app.get("/v1/card_games/ruling_dates_by_card_name", operation_id="get_ruling_dates", summary="Retrieves the dates when a card was considered legal for play in various formats. The card's name is used to identify the relevant ruling dates.")
async def get_ruling_dates(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT T2.date FROM cards AS T1 INNER JOIN rulings AS T2 ON T2.uuid = T1.uuid WHERE T1.name = ?", (card_name,))
    result = cursor.fetchall()
    if not result:
        return {"ruling_dates": []}
    return {"ruling_dates": [row[0] for row in result]}

# Endpoint to get the percentage of cards with a specific converted mana cost in a set
@app.get("/v1/card_games/percentage_cards_by_mana_cost_and_set", operation_id="get_percentage_cards_by_mana_cost", summary="Retrieves the percentage of cards in a specific set that have a given converted mana cost. The operation calculates this percentage by summing the cards with the specified mana cost and dividing by the total number of cards in the set.")
async def get_percentage_cards_by_mana_cost(mana_cost: int = Query(..., description="Converted mana cost of the card"), set_name: str = Query(..., description="Name of the set")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.convertedManaCost = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = ?", (mana_cost, set_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of cards with both cardKingdomFoilId and cardKingdomId in a set
@app.get("/v1/card_games/percentage_cards_with_both_ids_in_set", operation_id="get_percentage_cards_with_both_ids", summary="Retrieves the percentage of cards in a specified set that have both a cardKingdomFoilId and a cardKingdomId. The set is identified by its name.")
async def get_percentage_cards_with_both_ids(set_name: str = Query(..., description="Name of the set")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.cardKingdomFoilId IS NOT NULL AND T1.cardKingdomId IS NOT NULL THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = ?", (set_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the keyrune code by set code
@app.get("/v1/card_games/keyrune_code_by_set_code", operation_id="get_keyrune_code_by_set_code", summary="Retrieves the keyrune code associated with the specified set code. This operation allows you to look up the keyrune code for a particular set by providing its unique code as an input parameter.")
async def get_keyrune_code_by_set_code(set_code: str = Query(..., description="Code of the set")):
    cursor.execute("SELECT keyruneCode FROM sets WHERE code = ?", (set_code,))
    result = cursor.fetchall()
    if not result:
        return {"keyrune_codes": []}
    return {"keyrune_codes": [row[0] for row in result]}

# Endpoint to get the mcmId of a set based on its code
@app.get("/v1/card_games/set_mcmId_by_code", operation_id="get_set_mcmId_by_code", summary="Retrieves the unique identifier (mcmId) of a set, given its code. This operation allows you to look up a set using its code and obtain its corresponding mcmId. The code parameter is used to specify the set for which the mcmId is to be retrieved.")
async def get_set_mcmId_by_code(code: str = Query(..., description="Code of the set")):
    cursor.execute("SELECT mcmId FROM sets WHERE code = ?", (code,))
    result = cursor.fetchone()
    if not result:
        return {"mcmId": []}
    return {"mcmId": result[0]}

# Endpoint to get the mcmName of a set based on its release date
@app.get("/v1/card_games/set_mcmName_by_releaseDate", operation_id="get_set_mcmName_by_releaseDate", summary="Retrieves the mcmName of a card game set that was released on a specific date. The release date must be provided in 'YYYY-MM-DD' format.")
async def get_set_mcmName_by_releaseDate(releaseDate: str = Query(..., description="Release date of the set in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT mcmName FROM sets WHERE releaseDate = ?", (releaseDate,))
    result = cursor.fetchone()
    if not result:
        return {"mcmName": []}
    return {"mcmName": result[0]}

# Endpoint to get the type of sets based on a partial name match
@app.get("/v1/card_games/set_type_by_name_like", operation_id="get_set_type_by_name_like", summary="Retrieves the type of card game sets that partially match the provided name. The input parameter is used to specify the partial name of the set, allowing for a flexible search and retrieval of set types.")
async def get_set_type_by_name_like(name_like: str = Query(..., description="Partial name of the set")):
    cursor.execute("SELECT type FROM sets WHERE name LIKE ?", ('%' + name_like + '%',))
    result = cursor.fetchall()
    if not result:
        return {"types": []}
    return {"types": [row[0] for row in result]}

# Endpoint to get the parentCode of a set based on its name
@app.get("/v1/card_games/set_parentCode_by_name", operation_id="get_set_parentCode_by_name", summary="Retrieves the parent code associated with a specific set, identified by its name. This operation allows you to look up the parent code of a set using its unique name as a reference.")
async def get_set_parentCode_by_name(name: str = Query(..., description="Name of the set")):
    cursor.execute("SELECT parentCode FROM sets WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"parentCode": []}
    return {"parentCode": result[0]}

# Endpoint to get card rulings and content warning status based on the artist
@app.get("/v1/card_games/card_rulings_content_warning_by_artist", operation_id="get_card_rulings_content_warning_by_artist", summary="Retrieves the text of card rulings and the content warning status for cards created by the specified artist. The content warning status indicates whether the card has a content warning or not.")
async def get_card_rulings_content_warning_by_artist(artist: str = Query(..., description="Artist of the card")):
    cursor.execute("SELECT T2.text, CASE WHEN T1.hasContentWarning = 1 THEN 'YES' ELSE 'NO' END FROM cards AS T1 INNER JOIN rulings AS T2 ON T2.uuid = T1.uuid WHERE T1.artist = ?", (artist,))
    result = cursor.fetchall()
    if not result:
        return {"rulings": []}
    return {"rulings": [{"text": row[0], "contentWarning": row[1]} for row in result]}

# Endpoint to get the release date of a set based on the card name
@app.get("/v1/card_games/set_releaseDate_by_card_name", operation_id="get_set_releaseDate_by_card_name", summary="Get the release date of a set based on the card name")
async def get_set_releaseDate_by_card_name(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT T2.releaseDate FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T1.name = ?", (card_name,))
    result = cursor.fetchone()
    if not result:
        return {"releaseDate": []}
    return {"releaseDate": result[0]}

# Endpoint to get the type of sets based on the set translation
@app.get("/v1/card_games/set_type_by_translation", operation_id="get_set_type_by_translation", summary="Retrieves the type of card game sets that match the provided set translation. The translation parameter is used to identify the relevant sets and return their respective types.")
async def get_set_type_by_translation(translation: str = Query(..., description="Translation of the set")):
    cursor.execute("SELECT type FROM sets WHERE code IN (SELECT setCode FROM set_translations WHERE translation = ?)", (translation,))
    result = cursor.fetchall()
    if not result:
        return {"types": []}
    return {"types": [row[0] for row in result]}

# Endpoint to get the translation of a card based on its name and language
@app.get("/v1/card_games/card_translation_by_name_language", operation_id="get_card_translation_by_name_language", summary="Retrieves the translation of a specific card in a given language. The operation requires the card's name and the desired language as input parameters. It returns the translation if a match is found in the database, based on the card's name, language, and set code.")
async def get_card_translation_by_name_language(card_name: str = Query(..., description="Name of the card"), language: str = Query(..., description="Language of the translation")):
    cursor.execute("SELECT T2.translation FROM cards AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.setCode WHERE T1.name = ? AND T2.language = ? AND T2.translation IS NOT NULL", (card_name, language))
    result = cursor.fetchone()
    if not result:
        return {"translation": []}
    return {"translation": result[0]}

# Endpoint to get the count of distinct translations for a set based on its name
@app.get("/v1/card_games/count_distinct_translations_by_set_name", operation_id="get_count_distinct_translations_by_set_name", summary="Retrieves the number of unique translations associated with a specific set, identified by its name. The set's name is provided as an input parameter.")
async def get_count_distinct_translations_by_set_name(set_name: str = Query(..., description="Name of the set")):
    cursor.execute("SELECT COUNT(DISTINCT T2.translation) FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T1.name = ? AND T2.translation IS NOT NULL", (set_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the release date of a set based on its translation
@app.get("/v1/card_games/set_release_date_by_translation", operation_id="get_set_release_date_by_translation", summary="Retrieves the release date of a specific card game set, identified by its translation. The translation parameter is used to locate the set and its corresponding release date.")
async def get_set_release_date_by_translation(translation: str = Query(..., description="Translation of the set")):
    cursor.execute("SELECT T1.releaseDate FROM sets AS T1 INNER JOIN set_translations AS T2 ON T2.setCode = T1.code WHERE T2.translation = ?", (translation,))
    result = cursor.fetchone()
    if not result:
        return {"release_date": []}
    return {"release_date": result[0]}

# Endpoint to get the type of sets that include a specific card
@app.get("/v1/card_games/set_types_by_card", operation_id="get_set_types_by_card", summary="Retrieves the types of sets that contain a card with the provided name. The operation searches for the card by its name and identifies the sets it belongs to, then returns the types of those sets.")
async def get_set_types_by_card(card_name: str = Query(..., description="Name of the card")):
    cursor.execute("SELECT type FROM sets WHERE code IN ( SELECT setCode FROM cards WHERE name = ? )", (card_name,))
    result = cursor.fetchall()
    if not result:
        return {"set_types": []}
    return {"set_types": [row[0] for row in result]}

# Endpoint to get the count of cards with a specific converted mana cost from a specific set
@app.get("/v1/card_games/card_count_by_set_and_mana_cost", operation_id="get_card_count_by_set_and_mana_cost", summary="Retrieves the total number of cards from a specified set that have a particular converted mana cost. The set is identified by its name, and the mana cost is represented as a numerical value.")
async def get_card_count_by_set_and_mana_cost(set_name: str = Query(..., description="Name of the set"), converted_mana_cost: int = Query(..., description="Converted mana cost of the card")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE setCode IN ( SELECT code FROM sets WHERE name = ? ) AND convertedManaCost = ?", (set_name, converted_mana_cost))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the translation of a set in a specific language
@app.get("/v1/card_games/set_translation_by_language", operation_id="get_set_translation_by_language", summary="Retrieves the translation of a card game set in a specified language. The operation identifies the set by its name and returns the corresponding translation in the requested language.")
async def get_set_translation_by_language(set_name: str = Query(..., description="Name of the set"), language: str = Query(..., description="Language of the translation")):
    cursor.execute("SELECT translation FROM set_translations WHERE setCode IN ( SELECT code FROM sets WHERE name = ? ) AND language = ?", (set_name, language))
    result = cursor.fetchall()
    if not result:
        return {"translations": []}
    return {"translations": [row[0] for row in result]}

# Endpoint to get the percentage of non-foil only sets in a specific language
@app.get("/v1/card_games/percentage_non_foil_only_sets", operation_id="get_percentage_non_foil_only_sets", summary="Retrieves the percentage of sets that are exclusively non-foil in a specified language. The calculation is based on the total number of sets available in the given language.")
async def get_percentage_non_foil_only_sets(language: str = Query(..., description="Language of the set translations")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN isNonFoilOnly = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(id) FROM sets WHERE code IN ( SELECT setCode FROM set_translations WHERE language = ? )", (language,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of online-only sets in a specific language
@app.get("/v1/card_games/percentage_online_only_sets", operation_id="get_percentage_online_only_sets", summary="Retrieves the percentage of sets that are exclusively available online, filtered by a specific language. The calculation is based on the total number of sets in the specified language.")
async def get_percentage_online_only_sets(language: str = Query(..., description="Language of the set translations")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN isOnlineOnly = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(id) FROM sets WHERE code IN ( SELECT setCode FROM set_translations WHERE language = ? )", (language,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the distinct availability of textless cards by a specific artist
@app.get("/v1/card_games/textless_card_availability", operation_id="get_textless_card_availability", summary="Retrieves the unique availability statuses of textless cards created by a specified artist. This operation provides a concise overview of the distinct availability options for textless cards by the given artist.")
async def get_textless_card_availability(artist: str = Query(..., description="Name of the artist")):
    cursor.execute("SELECT DISTINCT availability FROM cards WHERE artist = ? AND isTextless = 1", (artist,))
    result = cursor.fetchall()
    if not result:
        return {"availability": []}
    return {"availability": [row[0] for row in result]}

# Endpoint to get the set with the largest base set size
@app.get("/v1/card_games/largest_base_set", operation_id="get_largest_base_set", summary="Retrieves the unique identifier of the set with the largest base set size from the available sets.")
async def get_largest_base_set():
    cursor.execute("SELECT id FROM sets ORDER BY baseSetSize DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"set_id": []}
    return {"set_id": result[0]}

# Endpoint to get the artist of a card with the highest converted mana cost where side is NULL
@app.get("/v1/card_games/artist_highest_mana_cost_null_side", operation_id="get_artist_highest_mana_cost_null_side", summary="Retrieves the artist of the card with the highest converted mana cost from the set of cards with no specified side. The number of results can be limited by the user.")
async def get_artist_highest_mana_cost_null_side(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT artist FROM cards WHERE side IS NULL ORDER BY convertedManaCost DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"artist": []}
    return {"artist": result[0]}

# Endpoint to get the most common frame effect for cards with non-null cardKingdomFoilId and cardKingdomId
@app.get("/v1/card_games/most_common_frame_effect", operation_id="get_most_common_frame_effect", summary="Retrieves the most frequently occurring frame effect for cards that have both a cardKingdomFoilId and a cardKingdomId. The results are ordered by frequency in descending order and can be limited by specifying the desired number of results.")
async def get_most_common_frame_effect(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT frameEffects FROM cards WHERE cardKingdomFoilId IS NOT NULL AND cardKingdomId IS NOT NULL GROUP BY frameEffects ORDER BY COUNT(frameEffects) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"frameEffects": []}
    return {"frameEffects": result[0]}

# Endpoint to get the count of cards with power '*' or NULL, given specific foil and duel deck criteria
@app.get("/v1/card_games/count_power_star_or_null", operation_id="get_count_power_star_or_null", summary="Retrieves the total count of cards that either have a power value of '*' or no power value at all, based on the specified foil and duel deck criteria. This operation is useful for determining the number of cards that meet these conditions, which can be used for various purposes such as game balance or deck construction.")
async def get_count_power_star_or_null(has_foil: int = Query(..., description="Has foil (0 or 1)"), duel_deck: str = Query(..., description="Duel deck identifier")):
    cursor.execute("SELECT SUM(CASE WHEN power = '*' OR power IS NULL THEN 1 ELSE 0 END) FROM cards WHERE hasFoil = ? AND duelDeck = ?", (has_foil, duel_deck))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the set ID with the largest total set size for a given set type
@app.get("/v1/card_games/largest_set_by_type", operation_id="get_largest_set_by_type", summary="Retrieves the set identifier of the largest set, based on total set size, for a specified set type. The operation allows limiting the number of results returned.")
async def get_largest_set_by_type(set_type: str = Query(..., description="Type of the set"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT id FROM sets WHERE type = ? ORDER BY totalSetSize DESC LIMIT ?", (set_type, limit))
    result = cursor.fetchone()
    if not result:
        return {"id": []}
    return {"id": result[0]}

# Endpoint to get distinct card names for a given format, ordered by mana cost
@app.get("/v1/card_games/distinct_card_names_by_format", operation_id="get_distinct_card_names_by_format", summary="Retrieves a list of unique card names for a specified card format, sorted by mana cost in descending order. The operation allows for pagination through the results using an offset and limit parameter. The format parameter determines the card format for which the distinct card names are retrieved.")
async def get_distinct_card_names_by_format(format: str = Query(..., description="Format of the card"), offset: int = Query(..., description="Offset for the results"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT name FROM cards WHERE uuid IN ( SELECT uuid FROM legalities WHERE format = ? ) ORDER BY manaCost DESC LIMIT ?, ?", (format, offset, limit))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the original release date and format of mythic rare cards with a legal status
@app.get("/v1/card_games/mythic_rare_cards_legal_status", operation_id="get_mythic_rare_cards_legal_status", summary="Retrieve the original release date and format of mythic rare cards that are currently legal in the game. The operation returns a limited number of results, sorted by the original release date in ascending order. The rarity and legal status of the cards can be specified as input parameters.")
async def get_mythic_rare_cards_legal_status(rarity: str = Query(..., description="Rarity of the card"), status: str = Query(..., description="Legal status of the card"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.originalReleaseDate, T2.format FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T1.rarity = ? AND T1.originalReleaseDate IS NOT NULL AND T2.status = ? ORDER BY T1.originalReleaseDate LIMIT ?", (rarity, status, limit))
    result = cursor.fetchone()
    if not result:
        return {"originalReleaseDate": [], "format": []}
    return {"originalReleaseDate": result[0], "format": result[1]}

# Endpoint to get the count of cards by a specific artist in a specific language
@app.get("/v1/card_games/count_cards_by_artist_language", operation_id="get_count_cards_by_artist_language", summary="Get the count of cards by a specific artist in a specific language")
async def get_count_cards_by_artist_language(artist: str = Query(..., description="Artist of the card"), language: str = Query(..., description="Language of the card")):
    cursor.execute("SELECT COUNT(T3.id) FROM ( SELECT T1.id FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T2.uuid = T1.uuid WHERE T1.artist = ? AND T2.language = ? GROUP BY T1.id ) AS T3", (artist, language))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cards with specific rarity, type, name, and legal status
@app.get("/v1/card_games/count_cards_by_criteria", operation_id="get_count_cards_by_criteria", summary="Retrieve the total number of cards that match the specified rarity, type, name, and legal status. This operation allows you to filter cards based on these criteria and obtain a count of the matching cards.")
async def get_count_cards_by_criteria(rarity: str = Query(..., description="Rarity of the card"), types: str = Query(..., description="Type of the card"), name: str = Query(..., description="Name of the card"), status: str = Query(..., description="Legal status of the card")):
    cursor.execute("SELECT COUNT(T1.id) FROM cards AS T1 INNER JOIN legalities AS T2 ON T2.uuid = T1.uuid WHERE T1.rarity = ? AND T1.types = ? AND T1.name = ? AND T2.status = ?", (rarity, types, name, status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the format and name of cards banned in the format with the most bans
@app.get("/v1/card_games/banned_cards_max_format", operation_id="get_banned_cards_max_format", summary="Retrieves the names of cards that are banned in the format with the highest number of bans. The number of results can be limited by the 'limit' parameter. This operation returns the format name and the card names.")
async def get_banned_cards_max_format(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("WITH MaxBanned AS (SELECT format, COUNT(*) AS count_banned FROM legalities WHERE status = 'Banned' GROUP BY format ORDER BY COUNT(*) DESC LIMIT ?) SELECT T2.format, T1.name FROM cards AS T1 INNER JOIN legalities AS T2 ON T2.uuid = T1.uuid INNER JOIN MaxBanned MB ON MB.format = T2.format WHERE T2.status = 'Banned'", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"cards": []}
    return {"cards": [{"format": row[0], "name": row[1]} for row in result]}

# Endpoint to get artist and format of cards grouped by artist and ordered by count of card IDs
@app.get("/v1/card_games/artist_format_grouped_by_artist", operation_id="get_artist_format_grouped_by_artist", summary="Retrieves a list of artists and their associated card formats, grouped by artist and ordered by the count of card IDs. The number of results can be limited using the provided input parameter.")
async def get_artist_format_grouped_by_artist(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.artist, T2.format FROM cards AS T1 INNER JOIN legalities AS T2 ON T2.uuid = T1.uuid GROUP BY T1.artist ORDER BY COUNT(T1.id) ASC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct statuses of cards based on frame version, content warning, artist, and format
@app.get("/v1/card_games/distinct_statuses_by_criteria", operation_id="get_distinct_statuses_by_criteria", summary="Retrieve the unique statuses of cards that match the specified frame version, content warning flag, artist, and format. This operation provides a list of distinct statuses based on the given criteria, offering a focused view of card statuses in the context of the provided parameters.")
async def get_distinct_statuses_by_criteria(frame_version: int = Query(..., description="Frame version of the card"), has_content_warning: int = Query(..., description="Content warning flag (0 or 1)"), artist: str = Query(..., description="Artist of the card"), format: str = Query(..., description="Format of the card")):
    cursor.execute("SELECT DISTINCT T2.status FROM cards AS T1 INNER JOIN legalities AS T2 ON T2.uuid = T1.uuid WHERE T1.frameVersion = ? AND T1.hasContentWarning = ? AND T1.artist = ? AND T2.format = ?", (frame_version, has_content_warning, artist, format))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get card names and formats based on EDHREC rank and status
@app.get("/v1/card_games/card_names_formats_by_rank_status", operation_id="get_card_names_formats_by_rank_status", summary="Retrieves the names and formats of cards that match the specified EDHREC rank and status. The operation filters cards based on their EDHREC rank and status, then groups the results by name and format.")
async def get_card_names_formats_by_rank_status(edhrec_rank: int = Query(..., description="EDHREC rank of the card"), status: str = Query(..., description="Status of the card")):
    cursor.execute("SELECT T1.name, T2.format FROM cards AS T1 INNER JOIN legalities AS T2 ON T2.uuid = T1.uuid WHERE T1.edhrecRank = ? AND T2.status = ? GROUP BY T1.name, T2.format", (edhrec_rank, status))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get average ID and language of sets based on release date range
@app.get("/v1/card_games/average_id_language_by_release_date", operation_id="get_average_id_language_by_release_date", summary="Retrieves the average set ID and the most common language for sets released within a specified date range. The date range is defined by the start_date and end_date parameters. The number of results returned can be limited using the limit parameter.")
async def get_average_id_language_by_release_date(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT (CAST(SUM(T1.id) AS REAL) / COUNT(T1.id)) / 4, T2.language FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.id = T2.id WHERE T1.releaseDate BETWEEN ? AND ? GROUP BY T1.releaseDate ORDER BY COUNT(T2.language) DESC LIMIT ?", (start_date, end_date, limit))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct artists based on availability and border color
@app.get("/v1/card_games/distinct_artists_by_availability_border_color", operation_id="get_distinct_artists_by_availability_border_color", summary="Retrieves a unique list of artists who have created cards that meet the specified availability and border color criteria. The availability parameter determines whether the cards are available for purchase or not, while the border color parameter filters the cards based on their border color.")
async def get_distinct_artists_by_availability_border_color(availability: str = Query(..., description="Availability of the card"), border_color: str = Query(..., description="Border color of the card")):
    cursor.execute("SELECT DISTINCT artist FROM cards WHERE availability = ? AND BorderColor = ?", (availability, border_color))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get UUIDs of legalities based on format and status
@app.get("/v1/card_games/legalities_uuids_by_format_status", operation_id="get_legalities_uuids_by_format_status", summary="Retrieves the unique identifiers (UUIDs) of card legalities that match the specified format and either of the two provided statuses. This operation is useful for filtering cards based on their legality in a specific format and status.")
async def get_legalities_uuids_by_format_status(format: str = Query(..., description="Format of the card"), status1: str = Query(..., description="First status of the card"), status2: str = Query(..., description="Second status of the card")):
    cursor.execute("SELECT uuid FROM legalities WHERE format = ? AND (status = ? OR status = ?)", (format, status1, status2))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of cards based on artist and availability
@app.get("/v1/card_games/card_count_by_artist_availability", operation_id="get_card_count_by_artist_availability", summary="Retrieves the total number of cards associated with a specific artist and availability status. The artist and availability parameters are used to filter the cards and calculate the count.")
async def get_card_count_by_artist_availability(artist: str = Query(..., description="Artist of the card"), availability: str = Query(..., description="Availability of the card")):
    cursor.execute("SELECT COUNT(id) FROM cards WHERE artist = ? AND availability = ?", (artist, availability))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get rulings text based on artist and ordered by date
@app.get("/v1/card_games/rulings_text_by_artist", operation_id="get_rulings_text_by_artist", summary="Retrieves the most recent ruling text for cards created by a specified artist. The results are sorted in descending order by the date of the ruling.")
async def get_rulings_text_by_artist(artist: str = Query(..., description="Artist of the card")):
    cursor.execute("SELECT T2.text FROM cards AS T1 INNER JOIN rulings AS T2 ON T2.uuid = T1.uuid WHERE T1.artist = ? ORDER BY T2.date DESC", (artist,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct card names and formats based on set name
@app.get("/v1/card_games/distinct_card_names_formats_by_set_name", operation_id="get_distinct_card_names_formats_by_set_name", summary="Retrieves a list of unique card names and their corresponding formats, if legal, from the specified set. The set is identified by its name.")
async def get_distinct_card_names_formats_by_set_name(set_name: str = Query(..., description="Name of the set")):
    cursor.execute("SELECT DISTINCT T2.name, CASE WHEN T1.status = 'Legal' THEN T1.format ELSE NULL END FROM legalities AS T1 INNER JOIN cards AS T2 ON T2.uuid = T1.uuid WHERE T2.setCode IN (SELECT code FROM sets WHERE name = ?)", (set_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get set names based on language and language pattern
@app.get("/v1/card_games/set_names_by_language", operation_id="get_set_names_by_language", summary="Retrieves the names of card game sets that are translated into the specified language, excluding those that match the provided language pattern. The operation filters the sets based on the language and language pattern, returning only the names of the sets that meet the criteria.")
async def get_set_names_by_language(language: str = Query(..., description="Language of the set translation"), language_pattern: str = Query(..., description="Language pattern to exclude")):
    cursor.execute("SELECT name FROM sets WHERE code IN (SELECT setCode FROM set_translations WHERE language = ? AND language NOT LIKE ?)", (language, language_pattern))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct frame versions and names of cards by a specific artist, with a status check
@app.get("/v1/card_games/distinct_frame_versions_names_by_artist", operation_id="get_distinct_frame_versions_names_by_artist", summary="Retrieve a unique list of frame versions and card names associated with a specific artist, along with a status check. If a card is banned, its name will be replaced with 'NO'.")
async def get_distinct_frame_versions_names_by_artist(artist: str = Query(..., description="Name of the artist")):
    cursor.execute("SELECT DISTINCT T1.frameVersion, T1.name, IIF(T2.status = 'Banned', T1.name, 'NO') FROM cards AS T1 INNER JOIN legalities AS T2 ON T1.uuid = T2.uuid WHERE T1.artist = ?", (artist,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

api_calls = [
    "/v1/card_games/cards_with_foil_and_regular_ids",
    "/v1/card_games/cards_with_border_color_missing_id?border_color=borderless",
    "/v1/card_games/card_with_lowest_mana_cost",
    "/v1/card_games/cards_with_edhrec_rank_and_frame_version?edhrec_rank=100&frame_version=2015",
    "/v1/card_games/distinct_card_ids_by_format_status_rarity?format=gladiator&status=Banned&rarity=mythic",
    "/v1/card_games/distinct_statuses_by_type_format_side?type=Artifact&format=vintage&side=null",
    "/v1/card_games/card_ids_artists_by_status_format_power?status=Legal&format=commander&power=*",
    "/v1/card_games/card_ids_rulings_content_warnings_by_artist?artist=Stephen%20Daniele",
    "/v1/card_games/ruling_texts_by_card_name_number?name=Sublime%20Epiphany&number=74s",
    "/v1/card_games/most_promo_cards_by_artist",
    "/v1/card_games/get_language_by_card_name_and_number?name=Annul&number=29",
    "/v1/card_games/get_card_names_by_language?language=Japanese",
    "/v1/card_games/get_percentage_of_cards_in_language?language=Chinese%20Simplified",
    "/v1/card_games/get_set_names_and_sizes_by_language?language=Italian",
    "/v1/card_games/get_card_type_count_by_artist?artist=Aaron%20Boyd",
    "/v1/card_games/get_distinct_keywords_by_card_name?name=Angel%20of%20Mercy",
    "/v1/card_games/get_card_count_by_power?power=*",
    "/v1/card_games/get_promo_types_by_card_name?name=Duress",
    "/v1/card_games/get_distinct_border_colors_by_card_name?name=Ancestor%27s%20Chosen",
    "/v1/card_games/get_original_type_by_card_name?name=Ancestor%27s%20Chosen",
    "/v1/card_games/set_translations_language?card_name=Angel%20of%20Mercy",
    "/v1/card_games/count_distinct_cards_status_textless?status=Restricted&is_textless=0",
    "/v1/card_games/ruling_text_for_card?card_name=Condemn",
    "/v1/card_games/count_distinct_starter_cards_status?status=Restricted&is_starter=1",
    "/v1/card_games/distinct_legality_statuses_for_card?card_name=Cloudchaser%20Eagle",
    "/v1/card_games/distinct_types_for_card?card_name=Benalish%20Knight",
    "/v1/card_games/formats_for_card?card_name=Benalish%20Knight",
    "/v1/card_games/artists_for_language?language=Phyrexian",
    "/v1/card_games/percentage_border_color?border_color=borderless",
    "/v1/card_games/count_reprinted_cards_language?language=German&is_reprint=1",
    "/v1/card_games/count_cards_border_color_language?border_color=borderless&language=Russian",
    "/v1/card_games/percentage_language_story_spotlight?language=French&is_story_spotlight=1",
    "/v1/card_games/count_cards_toughness?toughness=99",
    "/v1/card_games/distinct_card_names_artist?artist=Aaron%20Boyd",
    "/v1/card_games/count_cards_availability_border_color?availability=mtgo&border_color=black",
    "/v1/card_games/card_ids_converted_mana_cost?converted_mana_cost=0",
    "/v1/card_games/card_layouts_keyword?keyword=Flying",
    "/v1/card_games/count_cards_original_type_exclude_subtype?original_type=Summon%20-%20Angel&subtype=Angel",
    "/v1/card_games/card_ids_non_null_card_kingdom",
    "/v1/card_games/card_ids_duel_deck?duel_deck=a",
    "/v1/card_games/edhrec_rank_by_frame_version?frame_version=2015",
    "/v1/card_games/name_by_availability_and_language?availability=paper&language=Japanese",
    "/v1/card_games/count_by_status_and_border_color?status=Banned&border_color=white",
    "/v1/card_games/uuid_and_language_by_format?format=legacy",
    "/v1/card_games/count_by_frame_version?frame_version=future",
    "/v1/card_games/id_and_colors_by_set_code?set_code=OGW",
    "/v1/card_games/id_and_language_by_mana_cost_and_set_code?converted_mana_cost=5&set_code=10E",
    "/v1/card_games/id_and_date_by_original_type?original_type=Creature%20-%20Elf",
    "/v1/card_games/card_colors_formats_by_id_range?min_id=1&max_id=20",
    "/v1/card_games/distinct_card_names_by_type_and_color?original_type=Artifact&colors=B",
    "/v1/card_games/distinct_card_names_by_rarity_ordered_by_date?rarity=uncommon&limit=3",
    "/v1/card_games/count_cards_missing_card_kingdom_ids_by_artist?artist=John%20Avon",
    "/v1/card_games/count_cards_by_border_color_and_non_null_card_kingdom_ids?border_color=white",
    "/v1/card_games/count_cards_by_hand_artist_availability?hand=-1&artist=UDON&availability=mtgo",
    "/v1/card_games/count_cards_by_frame_version_availability_content_warning?frame_version=1993&availability=paper&has_content_warning=1",
    "/v1/card_games/mana_costs_by_availability_border_color_frame_version_layout?availability=mtgo,paper&border_color=black&frame_version=2003&layout=normal",
    "/v1/card_games/mana_costs_by_artist?artist=Rob%20Alexander",
    "/v1/card_games/distinct_subtypes_supertypes_by_availability?availability=arena",
    "/v1/card_games/set_codes_by_language?language=Spanish",
    "/v1/card_games/percentage_online_only_by_frame_effect?frame_effect=legendary",
    "/v1/card_games/percentage_non_textless_story_spotlight?is_story_spotlight=1",
    "/v1/card_games/percentage_foreign_data_by_language?language=Spanish",
    "/v1/card_games/languages_by_base_set_size?base_set_size=309",
    "/v1/card_games/count_sets_by_language_and_block?language=Portuguese%20(Brazil)&block=Commander",
    "/v1/card_games/card_ids_by_legal_status_and_type?status=Legal&types=Creature",
    "/v1/card_games/card_subtypes_supertypes_by_language?language=German",
    "/v1/card_games/ruling_texts_by_power_and_text_pattern?power=*&text_pattern=%triggered%20ability%",
    "/v1/card_games/count_cards_by_format_ruling_text_and_side?format=premodern&ruling_text=This%20is%20a%20triggered%20mana%20ability.&side=NULL",
    "/v1/card_games/card_ids_by_artist_format_availability?artist=Erica%20Yang&format=pauper&availability=paper",
    "/v1/card_games/distinct_artists_by_flavor_text?flavor_text=%25DAS%20perfekte%20Gegenmittel%20zu%20einer%20dichten%20Formation%25",
    "/v1/card_games/card_names_by_attributes_language?types=Creature&layout=normal&border_color=black&artist=Matthew%20D.%20Wilson&language=French",
    "/v1/card_games/count_distinct_card_ids_by_rarity_date?rarity=rare&date=2007-02-01",
    "/v1/card_games/set_languages_by_block_size?block=Ravnica&base_set_size=180",
    "/v1/card_games/percentage_no_content_warning_by_format_status?format=commander&status=Legal",
    "/v1/card_games/percentage_language_by_power?language=French&power=*",
    "/v1/card_games/percentage_language_by_set_type?language=Japanese&set_type=expansion",
    "/v1/card_games/distinct_availabilities_by_artist?artist=Daren%20Bader",
    "/v1/card_games/count_cards_by_edhrec_rank_border_color?edhrec_rank=12000&border_color=borderless",
    "/v1/card_games/count_cards_by_status?is_oversized=1&is_reprint=1&is_promo=1",
    "/v1/card_games/card_names_by_power_promo?power_pattern=%25*%25&promo_type=arenaleague&limit=3",
    "/v1/card_games/language_by_multiverseid?multiverseid=149934",
    "/v1/card_games/card_kingdom_ids?limit=3",
    "/v1/card_games/percentage_textless_cards?is_textless=1&layout=normal",
    "/v1/card_games/card_ids_by_subtypes?subtypes=Angel,Wizard",
    "/v1/card_games/set_names_null_mtgo?limit=3",
    "/v1/card_games/set_translation_languages?mcm_name=Archenemy&set_code=ARC",
    "/v1/card_games/set_names_translations?translation_id=5",
    "/v1/card_games/set_languages_types?translation_id=206",
    "/v1/card_games/sets_by_block_language?block=Shadowmoor&language=Italian&limit=2",
    "/v1/card_games/sets_by_language_foil_foreign?language=Japanese&is_foil_only=1&is_foreign_only=0",
    "/v1/card_games/sets_by_language_grouped_by_base_set_size?language=Russian&limit=1",
    "/v1/card_games/percentage_online_only_sets_by_language?language=Chinese%20Simplified&is_online_only=1",
    "/v1/card_games/count_sets_by_language_mtgo_code?language=Japanese&mtgo_code=",
    "/v1/card_games/cards_by_border_color?border_color=black",
    "/v1/card_games/cards_by_frame_effects?frame_effects=extendedart",
    "/v1/card_games/cards_by_border_color_full_art?border_color=black&is_full_art=1",
    "/v1/card_games/set_translation_language_by_id?id=174",
    "/v1/card_games/set_name_by_code?code=ALL",
    "/v1/card_games/distinct_languages_by_name?name=A%20Pedra%20Fellwar",
    "/v1/card_games/set_codes_by_release_date?release_date=2007-07-13",
    "/v1/card_games/distinct_base_set_sizes_and_set_codes_by_blocks?block1=Masques&block2=Mirage",
    "/v1/card_games/set_codes_by_set_type?set_type=expansion",
    "/v1/card_games/distinct_card_names_and_types_by_watermark?watermark=boros",
    "/v1/card_games/distinct_languages_and_flavor_texts_by_watermark?watermark=colorpie",
    "/v1/card_games/percentage_cards_by_mana_cost_and_name?converted_mana_cost=10&name=Abyssal%20Horror",
    "/v1/card_games/distinct_languages_and_card_types_by_watermark?watermark=azorius",
    "/v1/card_games/sum_cards_by_artist?artist=Aaron%20Miller",
    "/v1/card_games/sum_cards_by_availability_hand?availability=paper&hand=3",
    "/v1/card_games/distinct_card_names_by_textless?is_textless=0",
    "/v1/card_games/distinct_mana_costs_by_name?name=Ancestor%27s%20Chosen",
    "/v1/card_games/sum_cards_by_power_border_color?power_pattern=%25*%25&border_color=white",
    "/v1/card_games/distinct_card_names_by_promo_side?is_promo=1",
    "/v1/card_games/distinct_subtypes_supertypes_by_name?name=Molimo%2C%20Maro-Sorcerer",
    "/v1/card_games/distinct_purchase_urls_by_promo_type?promo_type=bundle",
    "/v1/card_games/count_cards_by_availability_border_color?availability_pattern=%25arena%2Cmtgo%25&border_color=black",
    "/v1/card_games/card_name_highest_mana_cost?name1=Serra%20Angel&name2=Shrine%20Keeper",
    "/v1/card_games/artist_by_flavor_name?flavor_name=Battra%2C%20Dark%20Destroyer",
    "/v1/card_games/card_names_by_frame_version?frame_version=2003&limit=3",
    "/v1/card_games/translations_by_card_name_and_language?card_name=Ancestor%27s%20Chosen&language=Italian",
    "/v1/card_games/count_distinct_translations_by_card_name?card_name=Angel%20of%20Mercy",
    "/v1/card_games/distinct_card_names_by_translation?translation=Hauptset%20Zehnte%20Edition",
    "/v1/card_games/check_translation_by_language_and_card_name?language=Korean&card_name=Ancestor%27s%20Chosen",
    "/v1/card_games/count_cards_by_translation_and_artist?translation=Hauptset%20Zehnte%20Edition&artist=Adam%20Rex",
    "/v1/card_games/base_set_size_by_translation?translation=Hauptset%20Zehnte%20Edition",
    "/v1/card_games/translations_by_set_name_and_language?set_name=Eighth%20Edition&language=Chinese%20Simplified",
    "/v1/card_games/check_mtgo_code_by_card_name?card_name=Angel%20of%20Mercy",
    "/v1/card_games/distinct_release_dates_by_card_name?card_name=Ancestor%27s%20Chosen",
    "/v1/card_games/set_types_by_translation?translation=Hauptset%20Zehnte%20Edition",
    "/v1/card_games/count_distinct_set_ids_by_block_and_language?block=Ice%20Age&language=Italian",
    "/v1/card_games/is_foreign_only_by_card_name?card_name=Adarkar%20Valkyrie",
    "/v1/card_games/count_sets_by_translation_base_set_size_language?base_set_size=100&language=Italian",
    "/v1/card_games/count_cards_by_border_color_and_set_name?border_color=black&set_name=Coldsnap",
    "/v1/card_games/highest_mana_cost_card_by_set_name?set_name=Coldsnap",
    "/v1/card_games/artists_by_set_name_and_artists?set_name=Coldsnap&artist1=Chippy&artist2=Aaron%20Miller&artist3=Jeremy%20Jarvis",
    "/v1/card_games/card_name_by_set_name_and_number?set_name=Coldsnap&card_number=4",
    "/v1/card_games/count_variable_power_cards_by_set_name_and_mana_cost?set_name=Coldsnap&mana_cost=5",
    "/v1/card_games/flavor_text_by_card_name_and_language?card_name=Ancestor%27s%20Chosen&language=Italian",
    "/v1/card_games/languages_by_card_name_with_flavor_text?card_name=Ancestor%27s%20Chosen",
    "/v1/card_games/distinct_types_by_card_name_and_language?card_name=Ancestor%27s%20Chosen&language=German",
    "/v1/card_games/distinct_texts_by_set_and_language?set_name=Coldsnap&language=Italian",
    "/v1/card_games/card_names_by_set_and_language_ordered_by_mana_cost?set_name=Coldsnap&language=Italian",
    "/v1/card_games/ruling_dates_by_card_name?card_name=Reminisce",
    "/v1/card_games/percentage_cards_by_mana_cost_and_set?mana_cost=7&set_name=Coldsnap",
    "/v1/card_games/percentage_cards_with_both_ids_in_set?set_name=Coldsnap",
    "/v1/card_games/keyrune_code_by_set_code?set_code=PKHC",
    "/v1/card_games/set_mcmId_by_code?code=SS2",
    "/v1/card_games/set_mcmName_by_releaseDate?releaseDate=2017-06-09",
    "/v1/card_games/set_type_by_name_like?name_like=FROM%20the%20Vault:%20Lore",
    "/v1/card_games/set_parentCode_by_name?name=Commander%202014%20Oversized",
    "/v1/card_games/card_rulings_content_warning_by_artist?artist=Jim%20Pavelec",
    "/v1/card_games/set_releaseDate_by_card_name?card_name=Evacuation",
    "/v1/card_games/set_type_by_translation?translation=Huiti%C3%A8me%20%C3%A9dition",
    "/v1/card_games/card_translation_by_name_language?card_name=Tendo%20Ice%20Bridge&language=French",
    "/v1/card_games/count_distinct_translations_by_set_name?set_name=Tenth%20Edition",
    "/v1/card_games/set_release_date_by_translation?translation=Ola%20de%20fr%C3%ADo",
    "/v1/card_games/set_types_by_card?card_name=Samite%20Pilgrim",
    "/v1/card_games/card_count_by_set_and_mana_cost?set_name=World%20Championship%20Decks%202004&converted_mana_cost=3",
    "/v1/card_games/set_translation_by_language?set_name=Mirrodin&language=Chinese%20Simplified",
    "/v1/card_games/percentage_non_foil_only_sets?language=Japanese",
    "/v1/card_games/percentage_online_only_sets?language=Portuguese%20(Brazil)",
    "/v1/card_games/textless_card_availability?artist=Aleksi%20Briclot",
    "/v1/card_games/largest_base_set",
    "/v1/card_games/artist_highest_mana_cost_null_side?limit=1",
    "/v1/card_games/most_common_frame_effect?limit=1",
    "/v1/card_games/count_power_star_or_null?has_foil=0&duel_deck=a",
    "/v1/card_games/largest_set_by_type?set_type=commander&limit=1",
    "/v1/card_games/distinct_card_names_by_format?format=duel&offset=0&limit=10",
    "/v1/card_games/mythic_rare_cards_legal_status?rarity=mythic&status=Legal&limit=1",
    "/v1/card_games/count_cards_by_artist_language?artist=Volkan%20Ba%C4%9Fa&language=French",
    "/v1/card_games/count_cards_by_criteria?rarity=rare&types=Enchantment&name=Abundance&status=Legal",
    "/v1/card_games/banned_cards_max_format?limit=1",
    "/v1/card_games/artist_format_grouped_by_artist?limit=1",
    "/v1/card_games/distinct_statuses_by_criteria?frame_version=1997&has_content_warning=1&artist=D.%20Alexander%20Gregory&format=legacy",
    "/v1/card_games/card_names_formats_by_rank_status?edhrec_rank=1&status=Banned",
    "/v1/card_games/average_id_language_by_release_date?start_date=2012-01-01&end_date=2015-12-31&limit=1",
    "/v1/card_games/distinct_artists_by_availability_border_color?availability=arena&border_color=black",
    "/v1/card_games/legalities_uuids_by_format_status?format=oldschool&status1=Banned&status2=Restricted",
    "/v1/card_games/card_count_by_artist_availability?artist=Matthew%20D.%20Wilson&availability=paper",
    "/v1/card_games/rulings_text_by_artist?artist=Kev%20Walker",
    "/v1/card_games/distinct_card_names_formats_by_set_name?set_name=Hour%20of%20Devastation",
    "/v1/card_games/set_names_by_language?language=Korean&language_pattern=%Japanese%",
    "/v1/card_games/distinct_frame_versions_names_by_artist?artist=Allen%20Williams"
]
