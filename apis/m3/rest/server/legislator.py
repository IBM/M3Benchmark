from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/legislator/legislator.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of current records with a specific ballotpedia_id or NULL
@app.get("/v1/legislator/count_current_ballotpedia_id", operation_id="get_count_current_ballotpedia_id", summary="Retrieves the total count of current records that either match a given Ballotpedia ID or have no Ballotpedia ID assigned. This operation is useful for determining the prevalence of a specific Ballotpedia ID or the number of records without a Ballotpedia ID in the current dataset.")
async def get_count_current_ballotpedia_id(ballotpedia_id: str = Query(..., description="Ballotpedia ID or NULL")):
    cursor.execute("SELECT COUNT(*) FROM current WHERE ballotpedia_id = ? OR ballotpedia_id IS NULL", (ballotpedia_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the official full names of current records with a specific cspan_id or NULL
@app.get("/v1/legislator/official_full_name_cspan_id", operation_id="get_official_full_name_cspan_id", summary="Retrieves the full names of current legislators. If a CSPAN ID is provided, the operation returns the full name of the legislator associated with that ID. If no ID is provided, the operation returns the full names of all legislators without a CSPAN ID.")
async def get_official_full_name_cspan_id(cspan_id: str = Query(..., description="CSPAN ID or NULL")):
    cursor.execute("SELECT official_full_name FROM current WHERE cspan_id IS NULL OR cspan_id = ?", (cspan_id,))
    result = cursor.fetchall()
    if not result:
        return {"official_full_names": []}
    return {"official_full_names": [row[0] for row in result]}

# Endpoint to get the count of current records with a birthday on or after a specific date
@app.get("/v1/legislator/count_current_birthday_bio", operation_id="get_count_current_birthday_bio", summary="Retrieves the total number of current legislators who have a birthday on or after the provided date. The date should be formatted as 'YYYY-MM-DD'.")
async def get_count_current_birthday_bio(birthday_bio: str = Query(..., description="Birthday in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(bioguide_id) FROM current WHERE birthday_bio >= ?", (birthday_bio,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of current records with a specific FEC ID or NULL and a specific gender
@app.get("/v1/legislator/count_current_fec_id_gender", operation_id="get_count_current_fec_id_gender", summary="Retrieves the total count of current records that either have a specified FEC ID or no FEC ID, and a specified gender. This operation is useful for understanding the distribution of records based on FEC ID and gender.")
async def get_count_current_fec_id_gender(fec_id: str = Query(..., description="FEC ID or NULL"), gender_bio: str = Query(..., description="Gender")):
    cursor.execute("SELECT COUNT(*) FROM current WHERE (fec_id IS NULL OR fec_id = ?) AND gender_bio = ?", (fec_id, gender_bio))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the Google entity ID of a current record with a specific official full name
@app.get("/v1/legislator/google_entity_id_official_full_name", operation_id="get_google_entity_id_official_full_name", summary="Retrieves the unique Google entity identifier associated with a current legislator record, based on the provided official full name.")
async def get_google_entity_id_official_full_name(official_full_name: str = Query(..., description="Official full name")):
    cursor.execute("SELECT google_entity_id_id FROM current WHERE official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"google_entity_id": []}
    return {"google_entity_id": result[0]}

# Endpoint to get the official full name of the oldest current record with one of the specified names
@app.get("/v1/legislator/oldest_official_full_name", operation_id="get_oldest_official_full_name", summary="Retrieves the full name of the oldest legislator with either the first or second specified name. The response will contain the full name of the legislator who was born the earliest among those with the provided names.")
async def get_oldest_official_full_name(name1: str = Query(..., description="First official full name"), name2: str = Query(..., description="Second official full name")):
    cursor.execute("SELECT official_full_name FROM current WHERE official_full_name = ? OR official_full_name = ? ORDER BY birthday_bio LIMIT 1", (name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"official_full_name": []}
    return {"official_full_name": result[0]}

# Endpoint to get the Facebook profile of a current record with a specific official full name
@app.get("/v1/legislator/facebook_official_full_name", operation_id="get_facebook_official_full_name", summary="Retrieves the Facebook profile associated with a current legislator record, based on the provided official full name. The operation returns the Facebook profile URL from the social-media table, which is linked to the current legislator record via the bioguide_id field.")
async def get_facebook_official_full_name(official_full_name: str = Query(..., description="Official full name")):
    cursor.execute("SELECT T1.facebook FROM `social-media` AS T1 INNER JOIN current AS T2 ON T2.bioguide_id = T1.bioguide WHERE T2.official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"facebook": []}
    return {"facebook": result[0]}

# Endpoint to get the count of current records with no Instagram profile
@app.get("/v1/legislator/count_current_no_instagram", operation_id="get_count_current_no_instagram", summary="Retrieves the total count of current legislators who do not have an associated Instagram profile. This operation fetches data from the 'social-media' table, cross-referencing it with the 'current' table using the 'bioguide' identifier. The count is based on records where the 'instagram' field is null, indicating the absence of an Instagram profile.")
async def get_count_current_no_instagram():
    cursor.execute("SELECT COUNT(*) FROM `social-media` AS T1 INNER JOIN current AS T2 ON T1.bioguide = T2.bioguide_id WHERE T1.instagram IS NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the official full name of a current record with a specific Twitter ID
@app.get("/v1/legislator/official_full_name_twitter_id", operation_id="get_official_full_name_twitter_id", summary="Retrieves the official full name of a current legislator associated with the provided Twitter ID. The operation uses the Twitter ID to search for a matching record in the social-media table, then retrieves the official full name from the current table using the bioguide identifier.")
async def get_official_full_name_twitter_id(twitter_id: int = Query(..., description="Twitter ID")):
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T2.twitter_id = ?", (twitter_id,))
    result = cursor.fetchone()
    if not result:
        return {"official_full_name": []}
    return {"official_full_name": result[0]}

# Endpoint to get the YouTube profiles of current records with a specific gender
@app.get("/v1/legislator/youtube_gender_bio", operation_id="get_youtube_gender_bio", summary="Retrieve the YouTube profiles of current legislators with a specified gender. The operation filters the current records based on the provided gender and returns the corresponding YouTube profiles from the social-media table.")
async def get_youtube_gender_bio(gender_bio: str = Query(..., description="Gender")):
    cursor.execute("SELECT T2.youtube FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T1.gender_bio = ?", (gender_bio,))
    result = cursor.fetchall()
    if not result:
        return {"youtube_profiles": []}
    return {"youtube_profiles": [row[0] for row in result]}

# Endpoint to get the Facebook profile of a legislator ordered by their birthday
@app.get("/v1/legislator/facebook_by_birthday", operation_id="get_facebook_by_birthday", summary="Retrieves the Facebook profile of a legislator, sorted by their birthday. The returned data is limited to a single result.")
async def get_facebook_by_birthday():
    cursor.execute("SELECT T2.facebook FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id ORDER BY T1.birthday_bio LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"facebook": []}
    return {"facebook": result[0]}

# Endpoint to get the count of legislators without an Instagram profile and with a specific OpenSecrets ID
@app.get("/v1/legislator/count_no_instagram_opensecrets", operation_id="get_count_no_instagram_opensecrets", summary="Retrieves the total number of legislators who do not have an Instagram profile and have a specified OpenSecrets ID. The OpenSecrets ID is used to filter the results. If no OpenSecrets ID is provided, the count includes legislators with no OpenSecrets ID.")
async def get_count_no_instagram_opensecrets(opensecrets_id: str = Query(..., description="OpenSecrets ID of the legislator")):
    cursor.execute("SELECT SUM(CASE WHEN T1.instagram IS NULL THEN 1 ELSE 0 END) AS count FROM `social-media` AS T1 INNER JOIN current AS T2 ON T1.bioguide = T2.bioguide_id WHERE T2.opensecrets_id IS NULL OR T2.opensecrets_id = ?", (opensecrets_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of legislators with a specific name and district
@app.get("/v1/legislator/count_name_district", operation_id="get_count_name_district", summary="Retrieve the total number of legislators who share a given full name and are associated with a specific district. If the district is not provided, the count will include legislators without a district.")
async def get_count_name_district(official_full_name: str = Query(..., description="Official full name of the legislator"), district: str = Query(..., description="District of the legislator")):
    cursor.execute("SELECT SUM(CASE WHEN T1.official_full_name = ? THEN 1 ELSE 0 END) AS count FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.district IS NULL OR T2.district = ?", (official_full_name, district))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of legislators with a specific name
@app.get("/v1/legislator/count_by_name", operation_id="get_count_by_name", summary="Retrieves the total number of legislators with a specified full name. The operation searches for legislators with the provided full name and returns the count of matching records.")
async def get_count_by_name(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the official full names of legislators with a state rank
@app.get("/v1/legislator/official_names_with_state_rank", operation_id="get_official_names_with_state_rank", summary="Retrieves the official full names of legislators who hold a state rank. This operation fetches the names from the current terms and joins them with the current legislator data, filtering out those without a state rank.")
async def get_official_names_with_state_rank():
    cursor.execute("SELECT T2.official_full_name FROM `current-terms` AS T1 INNER JOIN current AS T2 ON T2.bioguide_id = T1.bioguide WHERE T1.state_rank IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"official_names": []}
    return {"official_names": [row[0] for row in result]}

# Endpoint to get the state of a legislator based on start date and official full name
@app.get("/v1/legislator/state_by_start_date_name", operation_id="get_state_by_start_date_name", summary="Retrieves the state of a legislator based on the provided start date and official full name. The start date should be in 'YYYY-MM-DD' format. This operation returns the state associated with the legislator who began their term on the specified start date and has the given official full name.")
async def get_state_by_start_date_name(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT T1.state FROM `current-terms` AS T1 INNER JOIN current AS T2 ON T2.bioguide_id = T1.bioguide WHERE T1.start = ? AND T2.official_full_name = ?", (start_date, official_full_name))
    result = cursor.fetchone()
    if not result:
        return {"state": []}
    return {"state": result[0]}

# Endpoint to get the count of legislators with a specific gender and more than a certain number of terms
@app.get("/v1/legislator/count_gender_terms", operation_id="get_count_gender_terms", summary="Retrieves the count of legislators who identify with a specific gender and have served more than a certain number of terms. The gender and minimum term count are provided as input parameters.")
async def get_count_gender_terms(gender_bio: str = Query(..., description="Gender of the legislator"), min_terms: int = Query(..., description="Minimum number of terms")):
    cursor.execute("SELECT COUNT(CID) FROM ( SELECT T1.bioguide_id AS CID FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.gender_bio = ? GROUP BY T2.bioguide HAVING COUNT(T2.bioguide) > ? )", (gender_bio, min_terms))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of legislators born after a specific date and with more than a certain number of terms
@app.get("/v1/legislator/count_birthday_terms", operation_id="get_count_birthday_terms", summary="Retrieves the count of legislators who were born on or after a specific date and have served more than a certain number of terms. The date and minimum number of terms are provided as input parameters.")
async def get_count_birthday_terms(birthday_bio: str = Query(..., description="Birthday in 'YYYY-MM-DD' format"), min_terms: int = Query(..., description="Minimum number of terms")):
    cursor.execute("SELECT COUNT(CID) FROM ( SELECT T1.bioguide_id AS CID FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.birthday_bio >= ? GROUP BY T2.bioguide HAVING COUNT(T2.bioguide) > ? )", (birthday_bio, min_terms))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the proportion of legislators with a specific gender
@app.get("/v1/legislator/proportion_gender", operation_id="get_proportion_gender", summary="Retrieves the proportion of legislators with a specific gender from the current legislative body. The operation calculates the ratio of legislators with the specified gender to the total number of legislators. The gender is provided as an input parameter.")
async def get_proportion_gender(gender_bio: str = Query(..., description="Gender of the legislator")):
    cursor.execute("SELECT CAST(COUNT(T2.bioguide) AS REAL) / COUNT(DISTINCT T1.bioguide_id) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.gender_bio = ?", (gender_bio,))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

# Endpoint to get the percentage of legislators without an Instagram profile and with a specific religion
@app.get("/v1/legislator/percentage_no_instagram_religion", operation_id="get_percentage_no_instagram_religion", summary="Retrieves the percentage of legislators who do not have an Instagram profile and belong to a specific religious group. The calculation is based on the provided religion parameter.")
async def get_percentage_no_instagram_religion(religion_bio: str = Query(..., description="Religion of the legislator")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.instagram IS NULL THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM `social-media` AS T1 INNER JOIN current AS T2 ON T1.bioguide = T2.bioguide_id WHERE T2.religion_bio = ?", (religion_bio,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of current legislators based on gender
@app.get("/v1/legislator/current_count_by_gender", operation_id="get_current_count_by_gender", summary="Retrieves the total count of current legislators based on the specified gender. The gender parameter is used to filter the count.")
async def get_current_count_by_gender(gender_bio: str = Query(..., description="Gender of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM current WHERE gender_bio = ?", (gender_bio,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of current terms based on party
@app.get("/v1/legislator/current_terms_count_by_party", operation_id="get_current_terms_count_by_party", summary="Retrieves the total number of current legislative terms for a specified political party.")
async def get_current_terms_count_by_party(party: str = Query(..., description="Party of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM `current-terms` WHERE party = ?", (party,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of legislators with an Instagram account
@app.get("/v1/legislator/social_media_count_with_instagram", operation_id="get_social_media_count_with_instagram", summary="Retrieves the total number of legislators who have an active Instagram account. This operation does not require any input parameters and returns a single integer value representing the count.")
async def get_social_media_count_with_instagram():
    cursor.execute("SELECT COUNT(*) FROM `social-media` WHERE instagram IS NOT NULL AND instagram <> ''")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of historical legislators based on gender
@app.get("/v1/legislator/historical_count_by_gender", operation_id="get_historical_count_by_gender", summary="Retrieves the total count of historical legislators based on the specified gender. The gender parameter is used to filter the count.")
async def get_historical_count_by_gender(gender_bio: str = Query(..., description="Gender of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM historical WHERE gender_bio = ?", (gender_bio,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of current legislators based on religion and gender
@app.get("/v1/legislator/current_count_by_religion_and_gender", operation_id="get_current_count_by_religion_and_gender", summary="Retrieves the total number of current legislators categorized by a specified religion and gender.")
async def get_current_count_by_religion_and_gender(religion_bio: str = Query(..., description="Religion of the legislator"), gender_bio: str = Query(..., description="Gender of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM current WHERE religion_bio = ? AND gender_bio = ?", (religion_bio, gender_bio))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the party of a legislator based on first name, last name, and start year
@app.get("/v1/legislator/party_by_name_and_start_year", operation_id="get_party_by_name_and_start_year", summary="Retrieves the political party of a legislator based on their first name, last name, and the year they started serving. The input parameters include the legislator's first name, last name, and the start year in 'YYYY' format. The operation returns the party affiliation of the legislator who matches the provided criteria.")
async def get_party_by_name_and_start_year(first_name: str = Query(..., description="First name of the legislator"), last_name: str = Query(..., description="Last name of the legislator"), start_year: str = Query(..., description="Start year in 'YYYY' format")):
    cursor.execute("SELECT T1.party FROM `current-terms` AS T1 INNER JOIN current AS T2 ON T2.bioguide_id = T1.bioguide WHERE T2.first_name = ? AND T2.last_name = ? AND T1.start LIKE ?", (first_name, last_name, f'%{start_year}%'))
    result = cursor.fetchone()
    if not result:
        return {"party": []}
    return {"party": result[0]}

# Endpoint to get the official full name of a legislator based on state rank, type, and start year
@app.get("/v1/legislator/official_full_name_by_state_rank_type_start_year", operation_id="get_official_full_name_by_state_rank_type_start_year", summary="Get the official full name of a legislator based on state rank, type, and start year")
async def get_official_full_name_by_state_rank_type_start_year(state_rank: str = Query(..., description="State rank of the legislator"), type: str = Query(..., description="Type of the legislator"), start_year: str = Query(..., description="Start year in 'YYYY' format")):
    cursor.execute("SELECT T2.official_full_name FROM `current-terms` AS T1 INNER JOIN current AS T2 ON T2.bioguide_id = T1.bioguide WHERE T1.state_rank = ? AND T1.type = ? AND T1.start LIKE ?", (state_rank, type, f'{start_year}%'))
    result = cursor.fetchone()
    if not result:
        return {"official_full_name": []}
    return {"official_full_name": result[0]}

# Endpoint to get the YouTube channel of a legislator based on their official full name
@app.get("/v1/legislator/youtube_by_official_full_name", operation_id="get_youtube_by_official_full_name", summary="Retrieves the YouTube channel associated with a legislator, identified by their official full name. The operation searches for a match in the current legislator data and, if found, returns the corresponding YouTube channel from the social media data.")
async def get_youtube_by_official_full_name(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT T2.youtube FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T1.official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"youtube": []}
    return {"youtube": result[0]}

# Endpoint to get the count of social media platforms used by a legislator based on their first and last name
@app.get("/v1/legislator/social_media_count_by_name", operation_id="get_social_media_count_by_name", summary="Retrieves the total number of social media platforms used by a legislator, based on their first and last name. The operation searches for a legislator using the provided first and last names, and then calculates the count of social media platforms associated with the legislator. The platforms considered for this count are Facebook, Instagram, Twitter, and YouTube.")
async def get_social_media_count_by_name(first_name: str = Query(..., description="First name of the legislator"), last_name: str = Query(..., description="Last name of the legislator")):
    cursor.execute("SELECT CASE WHEN T1.facebook IS NOT NULL THEN 1 END + CASE WHEN T1.instagram IS NOT NULL THEN 1 END + CASE WHEN T1.twitter IS NOT NULL THEN 1 END + CASE WHEN T1.youtube IS NOT NULL THEN 1 END AS COUNTSOCIAL FROM `social-media` AS T1 INNER JOIN current AS T2 ON T1.bioguide = T2.bioguide_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count_social": []}
    return {"count_social": result[0]}

# Endpoint to get the last names of legislators based on state
@app.get("/v1/legislator/last_names_by_state", operation_id="get_last_names_by_state", summary="Retrieves a list of unique last names of legislators currently serving in the specified state. The list is generated by querying the legislator database and filtering results based on the provided state parameter.")
async def get_last_names_by_state(state: str = Query(..., description="State of the legislator")):
    cursor.execute("SELECT T1.last_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.state = ? GROUP BY T1.last_name", (state,))
    results = cursor.fetchall()
    if not results:
        return {"last_names": []}
    return {"last_names": [result[0] for result in results]}

# Endpoint to get the first and last names of legislators based on type and gender
@app.get("/v1/legislator/names_by_type_and_gender", operation_id="get_names_by_type_and_gender", summary="Retrieves the first and last names of legislators based on their type and gender. The operation filters legislators by their type (e.g., senator) and gender (e.g., female) and returns a list of unique legislator names.")
async def get_names_by_type_and_gender(type: str = Query(..., description="Type of legislator (e.g., 'sen')"), gender_bio: str = Query(..., description="Gender of the legislator (e.g., 'F')")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM `current-terms` AS T1 INNER JOIN current AS T2 ON T2.bioguide_id = T1.bioguide WHERE T1.type = ? AND T2.gender_bio = ? GROUP BY T2.ballotpedia_id", (type, gender_bio))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the GovTrack ID of a legislator based on their full name
@app.get("/v1/legislator/govtrack_by_full_name", operation_id="get_govtrack_by_full_name", summary="Retrieve the GovTrack ID of a legislator by providing their full name. This operation searches the current legislator database and matches the provided full name with the corresponding GovTrack ID from the social media database.")
async def get_govtrack_by_full_name(official_full_name: str = Query(..., description="Full name of the legislator")):
    cursor.execute("SELECT T2.govtrack FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T1.official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"govtrack": []}
    return {"govtrack": result[0]}

# Endpoint to get the Twitter handle of a legislator based on their full name
@app.get("/v1/legislator/twitter_by_full_name", operation_id="get_twitter_by_full_name", summary="Retrieves the Twitter handle of a legislator by matching their full name with the records in the current legislator database. The provided full name is used to search for a corresponding entry in the database, and the associated Twitter handle is returned if a match is found.")
async def get_twitter_by_full_name(official_full_name: str = Query(..., description="Full name of the legislator")):
    cursor.execute("SELECT T2.twitter FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T1.official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"twitter": []}
    return {"twitter": result[0]}

# Endpoint to get the first and last names of historical legislators based on party and date range
@app.get("/v1/legislator/historical_names_by_party_and_date", operation_id="get_historical_names_by_party_and_date", summary="Get the first and last names of historical legislators based on their party and a specific date range")
async def get_historical_names_by_party_and_date(party: str = Query(..., description="Party of the legislator (e.g., 'Pro-Administration')"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.party = ? AND CAST(T2.start AS DATE) <= ? AND CAST(T2.END AS DATE) >= ?", (party, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the first and last names of current legislators based on party and gender
@app.get("/v1/legislator/current_names_by_party_and_gender", operation_id="get_current_names_by_party_and_gender", summary="Retrieves the first and last names of current legislators who belong to a specified political party and have a particular gender. The results are grouped by the legislator's unique identifier.")
async def get_current_names_by_party_and_gender(party: str = Query(..., description="Party of the legislator (e.g., 'Republican')"), gender_bio: str = Query(..., description="Gender of the legislator (e.g., 'F')")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.party = ? AND T1.gender_bio = ? AND T2.END > DATE() GROUP BY T1.bioguide_id", (party, gender_bio))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the district of a legislator based on their full name
@app.get("/v1/legislator/district_by_full_name", operation_id="get_district_by_full_name", summary="Retrieves the district of a legislator based on their full name. The input parameter specifies the full name of the legislator. The operation returns the district associated with the legislator, provided that the legislator's name matches the input and the district is not null.")
async def get_district_by_full_name(official_full_name: str = Query(..., description="Full name of the legislator")):
    cursor.execute("SELECT T2.district FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ? AND T2.district IS NOT NULL GROUP BY T2.district", (official_full_name,))
    result = cursor.fetchall()
    if not result:
        return {"districts": []}
    return {"districts": result}

# Endpoint to get the count of legislators in a specific district based on their first and last name
@app.get("/v1/legislator/count_by_district_and_name", operation_id="get_count_by_district_and_name", summary="Retrieve the total number of legislators in a specified district, filtered by their first and last names. This operation requires the district number and the first and last names of the legislator as input parameters.")
async def get_count_by_district_and_name(district: int = Query(..., description="District number"), first_name: str = Query(..., description="First name of the legislator"), last_name: str = Query(..., description="Last name of the legislator")):
    cursor.execute("SELECT SUM(CASE WHEN T2.district = ? THEN 1 ELSE 0 END) AS count FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.first_name = ? AND T1.last_name = ?", (district, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the proportion of legislators based on gender, date range, and party
@app.get("/v1/legislator/proportion_by_gender_date_party", operation_id="get_proportion_by_gender_date_party", summary="Get the proportion of legislators based on their gender, a specific date range, and party")
async def get_proportion_by_gender_date_party(divisor: int = Query(..., description="Divisor for the proportion calculation"), gender_bio: str = Query(..., description="Gender of the legislator (e.g., 'M')"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), party: str = Query(..., description="Party of the legislator (e.g., 'Democrat')")):
    cursor.execute("SELECT CAST(COUNT(T1.bioguide_id) AS REAL) / ? FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.gender_bio = ? AND CAST(T2.start AS DATE) >= ? AND CAST(T2.END AS DATE) <= ? AND T2.party = ?", (divisor, gender_bio, start_date, end_date, party))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

# Endpoint to get the percentage of current legislators of a specific gender compared to historical legislators of the same gender
@app.get("/v1/legislator/percentage_current_vs_historical_by_gender", operation_id="get_percentage_current_vs_historical_by_gender", summary="Retrieves the percentage of current legislators of a specific gender relative to the total number of historical legislators of the same gender. The gender is specified as an input parameter.")
async def get_percentage_current_vs_historical_by_gender(gender_bio: str = Query(..., description="Gender of the legislator (e.g., 'F')")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN current.gender_bio = ? THEN current.bioguide_id ELSE NULL END) AS REAL) * 100 / ( SELECT COUNT(CASE WHEN historical.gender_bio = ? THEN historical.bioguide_id ELSE NULL END) FROM historical ) FROM current", (gender_bio, gender_bio))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the YouTube ID based on the YouTube username
@app.get("/v1/legislator/youtube_id_by_username", operation_id="get_youtube_id_by_username", summary="Retrieves the unique YouTube ID associated with the provided YouTube username. This operation searches the social-media database for the specified username and returns the corresponding YouTube ID.")
async def get_youtube_id_by_username(youtube: str = Query(..., description="YouTube username")):
    cursor.execute("SELECT youtube_id FROM `social-media` WHERE youtube = ?", (youtube,))
    result = cursor.fetchone()
    if not result:
        return {"youtube_id": []}
    return {"youtube_id": result[0]}

# Endpoint to get the Facebook profile of a legislator by their official full name
@app.get("/v1/legislator/facebook_by_name", operation_id="get_facebook_by_name", summary="Retrieves the Facebook profile of a legislator based on their official full name. The operation searches for a legislator with the provided name and returns their associated Facebook profile, if available.")
async def get_facebook_by_name(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT T2.facebook FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T1.official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"facebook": []}
    return {"facebook": result[0]}

# Endpoint to get the party of a historical legislator by their name
@app.get("/v1/legislator/party_by_name", operation_id="get_party_by_name", summary="Get the party of a historical legislator by their first, middle, or last name")
async def get_party_by_name(first_name: str = Query(None, description="First name of the legislator"), middle_name: str = Query(None, description="Middle name of the legislator"), last_name: str = Query(None, description="Last name of the legislator")):
    cursor.execute("SELECT T1.party FROM `historical-terms` AS T1 INNER JOIN historical AS T2 ON T2.bioguide_id = T1.bioguide WHERE T2.first_name = ? OR T2.middle_name = ? OR T2.last_name = ?", (first_name, middle_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"party": []}
    return {"party": result[0]}

# Endpoint to get the official full names of legislators with Facebook but no Instagram
@app.get("/v1/legislator/names_with_facebook_no_instagram", operation_id="get_names_with_facebook_no_instagram", summary="Retrieve a specified number of official full names of legislators who maintain a presence on Facebook but do not have an Instagram profile. This operation is useful for identifying legislators who are active on Facebook but not on Instagram.")
async def get_names_with_facebook_no_instagram(limit: int = Query(10, description="Limit the number of results returned")):
    cursor.execute("SELECT T2.official_full_name FROM `social-media` AS T1 INNER JOIN current AS T2 ON T1.bioguide = T2.bioguide_id WHERE T1.facebook IS NOT NULL AND (T1.instagram IS NULL OR T1.instagram = '') LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the official full names of legislators from a specific state
@app.get("/v1/legislator/names_by_state", operation_id="get_names_by_state", summary="Retrieve the official full names of legislators associated with a specified state. The operation filters legislators based on the provided state abbreviation and returns a distinct list of their official full names.")
async def get_names_by_state(state: str = Query(..., description="State abbreviation (e.g., 'VA')")):
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.state = ? GROUP BY T1.official_full_name", (state,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the first and last names of historical legislators by party
@app.get("/v1/legislator/names_by_party", operation_id="get_names_by_party", summary="Retrieves the first and last names of historical legislators who were affiliated with the specified political party. The party name must be provided as an input parameter.")
async def get_names_by_party(party: str = Query(..., description="Party name (e.g., 'National Greenbacker')")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM `historical-terms` AS T1 INNER JOIN historical AS T2 ON T2.bioguide_id = T1.bioguide WHERE T1.party = ?", (party,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the official full name of a legislator by their YouTube channel
@app.get("/v1/legislator/name_by_youtube", operation_id="get_name_by_youtube", summary="Retrieves the official full name of a legislator based on their associated YouTube channel. The operation uses the provided YouTube channel name to search for a match in the social media records and returns the corresponding legislator's official full name.")
async def get_name_by_youtube(youtube: str = Query(..., description="YouTube channel name")):
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T2.youtube = ?", (youtube,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the official full names of legislators with only Facebook profiles
@app.get("/v1/legislator/names_with_only_facebook", operation_id="get_names_with_only_facebook", summary="Get the official full names of legislators who have only a Facebook profile")
async def get_names_with_only_facebook():
    cursor.execute("SELECT T2.official_full_name FROM `social-media` AS T1 INNER JOIN current AS T2 ON T1.bioguide = T2.bioguide_id WHERE (T1.youtube IS NULL OR T1.youtube = '') AND (T1.instagram IS NULL OR T1.instagram = '') AND (T1.twitter IS NULL OR T1.twitter = '') AND T1.facebook IS NOT NULL AND T1.facebook != '")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the official full names of legislators by party and state rank
@app.get("/v1/legislator/names_by_party_and_state_rank", operation_id="get_names_by_party_and_state_rank", summary="Retrieves the official full names of legislators based on their party affiliation and state rank. The operation filters legislators by the specified party and state rank, and returns a list of unique names.")
async def get_names_by_party_and_state_rank(party: str = Query(..., description="Party name (e.g., 'Republican')"), state_rank: str = Query(..., description="State rank (e.g., 'junior')")):
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.party = ? AND T2.state_rank = ? GROUP BY T1.official_full_name", (party, state_rank))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the contact form of a legislator by their official full name
@app.get("/v1/legislator/contact_form_by_name", operation_id="get_contact_form_by_name", summary="Retrieves the contact form details of a legislator based on their official full name. The operation filters the legislator's information using the provided official full name and returns the associated contact form details.")
async def get_contact_form_by_name(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT T2.contact_form FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ? GROUP BY T2.contact_form", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"contact_form": []}
    return {"contact_form": result[0]}

# Endpoint to get the Wikipedia ID of historical legislators by party
@app.get("/v1/legislator/wikipedia_id_by_party", operation_id="get_wikipedia_id_by_party", summary="Retrieves the Wikipedia IDs of historical legislators who were affiliated with the specified political party. The party name must be provided as an input parameter.")
async def get_wikipedia_id_by_party(party: str = Query(..., description="Party name (e.g., 'Readjuster Democrat')")):
    cursor.execute("SELECT T2.wikipedia_id FROM `historical-terms` AS T1 INNER JOIN historical AS T2 ON T2.bioguide_id = T1.bioguide WHERE T1.party = ?", (party,))
    result = cursor.fetchall()
    if not result:
        return {"wikipedia_ids": []}
    return {"wikipedia_ids": [row[0] for row in result]}

# Endpoint to get the official full names of current legislators with a nickname from a specific party
@app.get("/v1/legislator/current_legislators_with_nickname", operation_id="get_current_legislators_with_nickname", summary="Retrieves the official full names of current legislators from a specified political party who have a registered nickname. The operation filters legislators based on their party affiliation and the presence of a nickname, then groups the results by their official full names.")
async def get_current_legislators_with_nickname(party: str = Query(..., description="Party of the legislator")):
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.party = ? AND T1.nickname_name IS NOT NULL GROUP BY T1.official_full_name", (party,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the state and party of historical legislators based on their name
@app.get("/v1/legislator/historical_legislators_state_party", operation_id="get_historical_legislators_state_party", summary="Get the state and party of historical legislators based on their first, middle, or last name")
async def get_historical_legislators_state_party(first_name: str = Query(None, description="First name of the legislator"), middle_name: str = Query(None, description="Middle name of the legislator"), last_name: str = Query(None, description="Last name of the legislator")):
    cursor.execute("SELECT T2.state, T2.party FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.first_name = ? OR T1.middle_name = ? OR T1.last_name = ?", (first_name, middle_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"state_party": []}
    return {"state_party": [{"state": row[0], "party": row[1]} for row in result]}

# Endpoint to get the count of historical legislators born on a specific date
@app.get("/v1/legislator/historical_legislators_count_by_birthdate", operation_id="get_historical_legislators_count_by_birthdate", summary="Get the count of historical legislators born on a specific date")
async def get_historical_legislators_count_by_birthdate(birthdate: str = Query(..., description="Birthdate in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(*) FROM historical WHERE CAST(birthday_bio AS date) = ?", (birthdate,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of male to female historical legislators
@app.get("/v1/legislator/historical_legislators_gender_ratio", operation_id="get_historical_legislators_gender_ratio", summary="Retrieves the ratio of male to female historical legislators from the database. This operation calculates the proportion of male legislators to female legislators based on the provided data.")
async def get_historical_legislators_gender_ratio():
    cursor.execute("SELECT CAST(SUM(CASE WHEN gender_bio = 'M' THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN gender_bio = 'F' THEN 1 ELSE 0 END) FROM historical")
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the end date and party of current terms based on year and party
@app.get("/v1/legislator/current_terms_end_date_party", operation_id="get_current_terms_end_date_party", summary="Retrieves the end date and party of the current legislative terms that match the specified year and party. The year should be provided in 'YYYY' format. The party parameter filters the results to only include legislators from the specified party.")
async def get_current_terms_end_date_party(year: str = Query(..., description="Year in 'YYYY' format"), party: str = Query(..., description="Party of the legislator")):
    cursor.execute("SELECT `END`, party FROM `current-terms` WHERE STRFTIME('%Y', `END`) = ? AND party = ?", (year, party))
    result = cursor.fetchall()
    if not result:
        return {"end_party": []}
    return {"end_party": [{"end": row[0], "party": row[1]} for row in result]}

# Endpoint to get the official full name and gender of current legislators based on last name
@app.get("/v1/legislator/current_legislators_name_gender", operation_id="get_current_legislators_name_gender", summary="Retrieves the official full name and gender of the current legislator(s) whose last name matches the provided input. This operation is useful for obtaining specific legislator information based on their last name.")
async def get_current_legislators_name_gender(last_name: str = Query(..., description="Last name of the legislator")):
    cursor.execute("SELECT official_full_name, gender_bio FROM current WHERE last_name = ?", (last_name,))
    result = cursor.fetchall()
    if not result:
        return {"name_gender": []}
    return {"name_gender": [{"name": row[0], "gender": row[1]} for row in result]}

# Endpoint to get the percentage of historical terms with a specific class type
@app.get("/v1/legislator/historical_terms_class_percentage", operation_id="get_historical_terms_class_percentage", summary="Retrieves the percentage of historical terms of a specific type that belong to a certain class. The type of the historical term is provided as an input parameter.")
async def get_historical_terms_class_percentage(type: str = Query(..., description="Type of the historical term")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM `historical-terms` WHERE type = ?", (type,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the official full names of current legislators from a specific party
@app.get("/v1/legislator/current_legislators_by_party", operation_id="get_current_legislators_by_party", summary="Retrieve the official full names of current legislators who belong to a specified political party. The operation filters legislators based on their party affiliation and returns a distinct list of their official full names.")
async def get_current_legislators_by_party(party: str = Query(..., description="Party of the legislator")):
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.party = ? GROUP BY T1.official_full_name", (party,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the sum of term durations for a specific legislator
@app.get("/v1/legislator/sum_term_durations", operation_id="get_sum_term_durations", summary="Retrieves the total duration of all terms served by a legislator, specified by their official full name. The response provides a sum of the time between the start and end dates of each term served by the legislator.")
async def get_sum_term_durations(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT SUM(CAST(T2.END - T2.start AS DATE)) AS sum FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the age at the start of the first term for a specific legislator
@app.get("/v1/legislator/age_at_first_term", operation_id="get_age_at_first_term", summary="Retrieves the age of a legislator at the start of their first term. The legislator is identified by their official full name. The response provides the age in years, calculated based on the legislator's birthdate and the start date of their first term.")
async def get_age_at_first_term(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT CAST(MIN(T2.start) - T1.birthday_bio AS DATE) AS AGE FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the official full name, relation, and party of current legislators from a specific state
@app.get("/v1/legislator/current_legislators_by_state", operation_id="get_current_legislators_by_state", summary="Retrieves the official full name, relation, and party of current legislators from a specified state. The operation filters legislators based on the provided state and groups the results by their official full name, relation, and party.")
async def get_current_legislators_by_state(state: str = Query(..., description="State of the legislator")):
    cursor.execute("SELECT T1.official_full_name, T2.relation, T2.party FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.state = ? GROUP BY T1.official_full_name, T2.relation, T2.party", (state,))
    result = cursor.fetchall()
    if not result:
        return {"legislators": []}
    return {"legislators": result}

# Endpoint to get the official full name and nickname of current legislators with an Instagram account and a Thomas ID less than a specified value
@app.get("/v1/legislator/current_legislators_with_instagram", operation_id="get_current_legislators_with_instagram", summary="Retrieves the official full name and nickname of current legislators who have an active Instagram account and a Thomas ID less than the provided maximum value. This operation filters legislators based on their Thomas ID and returns their official full name and nickname.")
async def get_current_legislators_with_instagram(thomas_id: int = Query(..., description="Maximum Thomas ID value")):
    cursor.execute("SELECT T1.official_full_name, T1.nickname_name FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T2.instagram IS NOT NULL AND T1.thomas_id < ?", (thomas_id,))
    result = cursor.fetchall()
    if not result:
        return {"legislators": []}
    return {"legislators": result}

# Endpoint to get the end date of historical terms for a specific legislator by their official full name
@app.get("/v1/legislator/historical_terms_end_date", operation_id="get_historical_terms_end_date", summary="Retrieves the end date of historical terms for a legislator identified by their official full name. The operation returns the date when the legislator's term ended, based on the provided official full name. This information is obtained by querying the historical terms and legislator data.")
async def get_historical_terms_end_date(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT T1.END FROM `historical-terms` AS T1 INNER JOIN historical AS T2 ON T2.bioguide_id = T1.bioguide WHERE T2.official_full_name = ?", (official_full_name,))
    result = cursor.fetchall()
    if not result:
        return {"end_dates": []}
    return {"end_dates": result}

# Endpoint to get the party and state of historical legislators with a house history ID and a birthday bio containing a specific year
@app.get("/v1/legislator/historical_legislators_by_birthday", operation_id="get_historical_legislators_by_birthday", summary="Retrieve the political party and state of historical legislators who have a house history ID and a birthday bio containing a specific year. The year is provided as a parameter in the format %YYYY%.")
async def get_historical_legislators_by_birthday(birthday_bio: str = Query(..., description="Year in the birthday bio (format: %1738%)")):
    cursor.execute("SELECT T1.party, T1.state FROM `historical-terms` AS T1 INNER JOIN historical AS T2 ON T2.bioguide_id = T1.bioguide WHERE T2.house_history_id IS NOT NULL AND T2.birthday_bio LIKE ?", (birthday_bio,))
    result = cursor.fetchall()
    if not result:
        return {"legislators": []}
    return {"legislators": result}

# Endpoint to get the bioguide ID, first name, and last name of historical legislators from a specific party
@app.get("/v1/legislator/historical_legislators_by_party", operation_id="get_historical_legislators_by_party", summary="Retrieve the biographical details of historical legislators affiliated with a specific political party. The response includes the bioguide ID, first name, and last name of each legislator. To filter the results, provide the desired party as an input parameter.")
async def get_historical_legislators_by_party(party: str = Query(..., description="Party of the legislator")):
    cursor.execute("SELECT T2.bioguide_id, T2.first_name, T2.last_name FROM `historical-terms` AS T1 INNER JOIN historical AS T2 ON T2.bioguide_id = T1.bioguide WHERE T1.party = ?", (party,))
    result = cursor.fetchall()
    if not result:
        return {"legislators": []}
    return {"legislators": result}

# Endpoint to get the count of historical legislators with a specific gender and start date
@app.get("/v1/legislator/count_historical_legislators_by_gender_start", operation_id="get_count_historical_legislators_by_gender_start", summary="Retrieves the total number of historical legislators who match a specified gender and start date. The start date is used to filter the legislators based on the commencement of their term.")
async def get_count_historical_legislators_by_gender_start(gender_bio: str = Query(..., description="Gender of the legislator"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.bioguide_id) FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.gender_bio = ? AND T2.start = ?", (gender_bio, start_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in the count of current and historical terms starting in a specific year
@app.get("/v1/legislator/term_count_difference_by_year", operation_id="get_term_count_difference_by_year", summary="Get the difference in the count of current and historical terms starting in a specific year")
async def get_term_count_difference_by_year(start_year: str = Query(..., description="Start year in 'YYYY%' format")):
    cursor.execute("SELECT SUM(CASE WHEN `current-terms`.start LIKE ? THEN 1 ELSE 0 END) - ( SELECT SUM(CASE WHEN start LIKE ? THEN 1 ELSE 0 END) FROM `historical-terms` ) FROM `current-terms`", (start_year, start_year))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the official full name, Twitter ID, and YouTube ID of current legislators with a specific first name
@app.get("/v1/legislator/current_legislators_by_first_name", operation_id="get_current_legislators_by_first_name", summary="Retrieves the official full name, Twitter handle, and YouTube channel of current legislators who share a specified first name. The first name is provided as an input parameter.")
async def get_current_legislators_by_first_name(first_name: str = Query(..., description="First name of the legislator")):
    cursor.execute("SELECT T2.official_full_name, T1.twitter_id, T1.youtube_id FROM `social-media` AS T1 INNER JOIN current AS T2 ON T1.bioguide = T2.bioguide_id WHERE T2.first_name = ?", (first_name,))
    result = cursor.fetchall()
    if not result:
        return {"legislators": []}
    return {"legislators": result}

# Endpoint to get the start date, end date, and party of historical terms for a specific legislator by their full name
@app.get("/v1/legislator/historical_terms_by_full_name", operation_id="get_historical_terms_by_full_name", summary="Retrieve the start and end dates, along with the party affiliation, of historical terms served by a legislator identified by their full name. The input parameters specify the first, middle, and last names of the legislator.")
async def get_historical_terms_by_full_name(first_name: str = Query(..., description="First name of the legislator"), middle_name: str = Query(..., description="Middle name of the legislator"), last_name: str = Query(..., description="Last name of the legislator")):
    cursor.execute("SELECT T2.start, T2.`end`, T2.party FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.first_name = ? AND T1.middle_name = ? AND T1.last_name = ?", (first_name, middle_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"terms": []}
    return {"terms": result}

# Endpoint to get the birthday bio of a current legislator by their first and last name
@app.get("/v1/legislator/current_legislator_birthday_bio", operation_id="get_current_legislator_birthday_bio", summary="Retrieves the birthday biography of a current legislator using their first and last names as identifiers. The endpoint returns the birthday bio of the legislator that matches the provided names.")
async def get_current_legislator_birthday_bio(first_name: str = Query(..., description="First name of the legislator"), last_name: str = Query(..., description="Last name of the legislator")):
    cursor.execute("SELECT birthday_bio FROM current WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"birthday_bio": []}
    return {"birthday_bio": result[0]}

# Endpoint to get the count of current records with null or empty fec_id
@app.get("/v1/legislator/count_null_or_empty_fec_id", operation_id="get_count_null_or_empty_fec_id", summary="Retrieves the total count of current records that have a missing or empty FEC ID. This operation is useful for identifying records that lack a Federal Election Commission ID, which is a crucial identifier for legislators.")
async def get_count_null_or_empty_fec_id(fec_id: str = Query(..., description="FEC ID to check for empty values")):
    cursor.execute("SELECT COUNT(*) FROM current WHERE fec_id IS NULL OR fec_id = ?", (fec_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of current records with non-null and non-empty opensecrets_id
@app.get("/v1/legislator/count_non_null_non_empty_opensecrets_id", operation_id="get_count_non_null_non_empty_opensecrets_id", summary="Retrieves the total count of current records that have a non-empty OpenSecrets ID. This operation is useful for determining the number of records with a valid OpenSecrets identifier.")
async def get_count_non_null_non_empty_opensecrets_id(opensecrets_id: str = Query(..., description="Opensecrets ID to check for non-empty values")):
    cursor.execute("SELECT COUNT(*) FROM current WHERE opensecrets_id IS NOT NULL AND opensecrets_id <> ?", (opensecrets_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the middle name of current records based on birthday
@app.get("/v1/legislator/middle_name_by_birthday", operation_id="get_middle_name_by_birthday", summary="Retrieves the middle name of the legislator(s) whose birthday matches the provided date. The date should be in the 'YYYY-MM-DD' format.")
async def get_middle_name_by_birthday(birthday_bio: str = Query(..., description="Birthday in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT middle_name FROM current WHERE birthday_bio = ?", (birthday_bio,))
    result = cursor.fetchall()
    if not result:
        return {"middle_names": []}
    return {"middle_names": [row[0] for row in result]}

# Endpoint to get the count of bioguide entries based on title
@app.get("/v1/legislator/count_bioguide_by_title", operation_id="get_count_bioguide_by_title", summary="Retrieves the total number of bioguide entries that match the specified title. The title parameter is used to filter the bioguide entries.")
async def get_count_bioguide_by_title(title: str = Query(..., description="Title to filter bioguide entries")):
    cursor.execute("SELECT COUNT(bioguide) FROM `current-terms` WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get titles of current terms based on birthday
@app.get("/v1/legislator/titles_by_birthday", operation_id="get_titles_by_birthday", summary="Retrieves the titles of legislators currently in office, filtered by a specific birthday. The birthday is provided in 'YYYY-MM-DD' format. The operation returns a list of unique titles held by legislators born on the specified date.")
async def get_titles_by_birthday(birthday_bio: str = Query(..., description="Birthday in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.title FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.birthday_bio = ? GROUP BY T2.title", (birthday_bio,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get gender of current records based on address
@app.get("/v1/legislator/gender_by_address", operation_id="get_gender_by_address", summary="Retrieves the gender of current legislators associated with a specific address. The operation filters the records based on the provided address and returns the gender of the corresponding legislators.")
async def get_gender_by_address(address: str = Query(..., description="Address to filter records")):
    cursor.execute("SELECT T1.gender_bio FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.address = ?", (address,))
    result = cursor.fetchall()
    if not result:
        return {"genders": []}
    return {"genders": [row[0] for row in result]}

# Endpoint to get first names of current records based on state rank
@app.get("/v1/legislator/first_names_by_state_rank", operation_id="get_first_names_by_state_rank", summary="Retrieves the first names of current legislators who hold a specific state rank. The state rank is used to filter the records, ensuring that only legislators with the specified rank are included in the response.")
async def get_first_names_by_state_rank(state_rank: str = Query(..., description="State rank to filter records")):
    cursor.execute("SELECT T1.first_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.state_rank = ? GROUP BY T1.first_name", (state_rank,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the count of states based on gender and class
@app.get("/v1/legislator/count_states_by_gender_and_class", operation_id="get_count_states_by_gender_and_class", summary="Retrieves the count of states with legislators of a specific gender and class. The gender and class filters are applied to narrow down the results. The count is calculated based on the number of states that meet the specified criteria.")
async def get_count_states_by_gender_and_class(gender_bio: str = Query(..., description="Gender to filter records"), class_value: str = Query(..., description="Class value to filter records")):
    cursor.execute("SELECT COUNT(T3.state) FROM ( SELECT T2.state FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.gender_bio = ? AND (T2.class IS NULL OR T2.class = ?) GROUP BY T2.state ) T3", (gender_bio, class_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of non-null class values based on birthday
@app.get("/v1/legislator/percentage_non_null_class_by_birthday", operation_id="get_percentage_non_null_class_by_birthday", summary="Retrieves the percentage of legislators with a non-null class value, filtered by a specific birthday. The birthday is provided in 'YYYY-MM-DD' format.")
async def get_percentage_non_null_class_by_birthday(birthday_bio: str = Query(..., description="Birthday in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.class IS NOT NULL THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.birthday_bio LIKE ?", (birthday_bio,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of legislators with a specific birth year or earlier who have no class
@app.get("/v1/legislator/percentage_no_class_by_birth_year", operation_id="get_percentage_no_class_by_birth_year", summary="Retrieves the percentage of legislators with no class who were born on or before a specified year. The calculation is based on the total count of legislators born on or before the given year and the count of those with no class among them.")
async def get_percentage_no_class_by_birth_year(birth_year: str = Query(..., description="Birth year in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.class IS NULL THEN T1.bioguide_id ELSE NULL END) AS REAL) * 100 / COUNT(T1.bioguide_id) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE CAST(T1.birthday_bio AS DATE) <= ?", (birth_year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the Twitter handle of a legislator based on their birth date
@app.get("/v1/legislator/twitter_by_birth_date", operation_id="get_twitter_by_birth_date", summary="Retrieves the Twitter handle of a legislator who was born on the specified date. The date should be provided in 'YYYY-MM-DD' format. This operation fetches the legislator's Twitter handle from the social media data, based on the legislator's biographical information.")
async def get_twitter_by_birth_date(birth_date: str = Query(..., description="Birth date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.twitter FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T1.birthday_bio = ?", (birth_date,))
    result = cursor.fetchone()
    if not result:
        return {"twitter": []}
    return {"twitter": result[0]}

# Endpoint to get the OpenSecrets ID of a legislator based on their YouTube handle
@app.get("/v1/legislator/opensecrets_id_by_youtube", operation_id="get_opensecrets_id_by_youtube", summary="Retrieves the OpenSecrets ID of a legislator based on their YouTube handle. This operation matches the provided YouTube handle with the corresponding legislator's bioguide ID in the social-media table, then retrieves the OpenSecrets ID from the current table using the bioguide ID.")
async def get_opensecrets_id_by_youtube(youtube_handle: str = Query(..., description="YouTube handle of the legislator")):
    cursor.execute("SELECT T1.opensecrets_id FROM current AS T1 INNER JOIN `social-media` AS T2 ON T2.bioguide = T1.bioguide_id WHERE T2.youtube = ?", (youtube_handle,))
    result = cursor.fetchone()
    if not result:
        return {"opensecrets_id": []}
    return {"opensecrets_id": result[0]}

# Endpoint to get the first names of legislators based on their address
@app.get("/v1/legislator/first_names_by_address", operation_id="get_first_names_by_address", summary="Retrieve the distinct first names of legislators associated with a given address. The operation filters legislators based on the provided address and returns a list of their unique first names.")
async def get_first_names_by_address(address: str = Query(..., description="Address of the legislator")):
    cursor.execute("SELECT T1.first_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.address = ? GROUP BY T1.first_name", (address,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the Instagram handle of a legislator based on their birth date
@app.get("/v1/legislator/instagram_by_birth_date", operation_id="get_instagram_by_birth_date", summary="Retrieves the Instagram handle of a legislator who was born on the specified date. The input parameter is the birth date in 'YYYY-MM-DD' format. This operation returns the Instagram handle of the legislator, if available.")
async def get_instagram_by_birth_date(birth_date: str = Query(..., description="Birth date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.instagram FROM `social-media` AS T1 INNER JOIN current AS T2 ON T1.bioguide = T2.bioguide_id WHERE T2.birthday_bio = ?", (birth_date,))
    result = cursor.fetchone()
    if not result:
        return {"instagram": []}
    return {"instagram": result[0]}

# Endpoint to get the count of legislators based on gender and class
@app.get("/v1/legislator/count_by_gender_and_class", operation_id="get_count_by_gender_and_class", summary="Retrieves the total number of legislators who match the specified gender and class. The gender and class parameters are used to filter the results. The response provides a count of legislators that meet the specified criteria.")
async def get_count_by_gender_and_class(gender: str = Query(..., description="Gender of the legislator"), class_value: str = Query(..., description="Class value of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.gender_bio = ? AND (T2.class IS NULL OR T2.class = ?)", (gender, class_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the religion of a legislator based on their YouTube handle
@app.get("/v1/legislator/religion_by_youtube", operation_id="get_religion_by_youtube", summary="Retrieves the religion of a legislator based on their provided YouTube handle. This operation fetches the religion information from the legislator's biography, which is linked to their social media profiles. The input parameter is the YouTube handle of the legislator.")
async def get_religion_by_youtube(youtube_handle: str = Query(..., description="YouTube handle of the legislator")):
    cursor.execute("SELECT T2.religion_bio FROM `social-media` AS T1 INNER JOIN current AS T2 ON T1.bioguide = T2.bioguide_id WHERE T1.youtube = ?", (youtube_handle,))
    result = cursor.fetchone()
    if not result:
        return {"religion": []}
    return {"religion": result[0]}

# Endpoint to get the count of legislators based on title and FEC ID
@app.get("/v1/legislator/count_by_title_and_fec_id", operation_id="get_count_by_title_and_fec_id", summary="Retrieves the number of legislators who have a specified title and do not have a Federal Election Commission (FEC) ID or have an empty FEC ID. The title parameter is used to filter the legislators, while the FEC ID parameter is used to exclude those with a valid FEC ID.")
async def get_count_by_title_and_fec_id(title: str = Query(..., description="Title of the legislator"), fec_id: str = Query(..., description="FEC ID of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.title = ? AND (T1.fec_id IS NULL OR T1.fec_id = ?)", (title, fec_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the Facebook ID of a legislator based on their Facebook handle
@app.get("/v1/legislator/facebook_id_by_facebook", operation_id="get_facebook_id_by_facebook", summary="Retrieves the unique Facebook ID associated with a legislator, given their Facebook handle. This operation allows you to look up a legislator's Facebook ID using their handle, enabling further data retrieval or analysis related to their social media presence.")
async def get_facebook_id_by_facebook(facebook_handle: str = Query(..., description="Facebook handle of the legislator")):
    cursor.execute("SELECT facebook_id FROM `social-media` WHERE facebook = ?", (facebook_handle,))
    result = cursor.fetchone()
    if not result:
        return {"facebook_id": []}
    return {"facebook_id": result[0]}

# Endpoint to get the count of current legislators with a specific first name
@app.get("/v1/legislator/count_by_first_name", operation_id="get_count_by_first_name", summary="Retrieves the total number of current legislators who share a specified first name.")
async def get_count_by_first_name(first_name: str = Query(..., description="First name of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM current WHERE first_name = ?", (first_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get districts of historical terms by party
@app.get("/v1/legislator/districts_by_party", operation_id="get_districts_by_party", summary="Retrieves a list of unique districts associated with a specific political party from historical term records. The party is specified as an input parameter.")
async def get_districts_by_party(party: str = Query(..., description="Party of the historical term")):
    cursor.execute("SELECT district FROM `historical-terms` WHERE party = ? GROUP BY district", (party,))
    result = cursor.fetchall()
    if not result:
        return {"districts": []}
    return {"districts": [row[0] for row in result]}

# Endpoint to get official full names of current legislators born in a specific year
@app.get("/v1/legislator/official_full_names_by_birth_year", operation_id="get_official_full_names_by_birth_year", summary="Retrieves the full names of legislators currently in office who were born in a specific year. The birth year should be provided in the 'YYYY%' format.")
async def get_official_full_names_by_birth_year(birth_year: str = Query(..., description="Birth year in 'YYYY%' format")):
    cursor.execute("SELECT official_full_name FROM current WHERE birthday_bio LIKE ?", (birth_year,))
    result = cursor.fetchall()
    if not result:
        return {"official_full_names": []}
    return {"official_full_names": [row[0] for row in result]}

# Endpoint to get Google entity IDs of historical legislators by first and last name
@app.get("/v1/legislator/google_entity_ids_by_name", operation_id="get_google_entity_ids_by_name", summary="Retrieves the Google entity IDs of historical legislators whose first and last names match the provided input parameters. This operation is useful for identifying legislators based on their full names.")
async def get_google_entity_ids_by_name(first_name: str = Query(..., description="First name of the legislator"), last_name: str = Query(..., description="Last name of the legislator")):
    cursor.execute("SELECT google_entity_id_id FROM historical WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"google_entity_ids": []}
    return {"google_entity_ids": [row[0] for row in result]}

# Endpoint to get names of historical legislators by party and term dates
@app.get("/v1/legislator/names_by_party_and_term_dates", operation_id="get_names_by_party_and_term_dates", summary="Retrieve the first and last names of historical legislators who served during a specific term, filtered by their political party. The term is defined by a start and end date, both provided in 'YYYY-MM-DD' format. This operation returns a list of legislator names that match the specified criteria.")
async def get_names_by_party_and_term_dates(party: str = Query(..., description="Party of the historical term"), start: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.party = ? AND T2.start = ? AND T2.end = ?", (party, start, end))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get names of historical legislators by district
@app.get("/v1/legislator/names_by_district", operation_id="get_names_by_district", summary="Retrieves the first and last names of historical legislators associated with a specific district. The district is identified by a unique number.")
async def get_names_by_district(district: int = Query(..., description="District number")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM `historical-terms` AS T1 INNER JOIN historical AS T2 ON T2.bioguide_id = T1.bioguide WHERE T1.district = ?", (district,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get OpenSecrets and Thomas IDs of current legislators by type and state
@app.get("/v1/legislator/ids_by_type_and_state", operation_id="get_ids_by_type_and_state", summary="Retrieve the unique identifiers of current legislators from OpenSecrets and Thomas databases based on the legislator type and state. The operation filters legislators by the provided type and state, and returns their respective OpenSecrets and Thomas IDs.")
async def get_ids_by_type_and_state(type: str = Query(..., description="Type of the term (e.g., 'sen')"), state: str = Query(..., description="State abbreviation (e.g., 'NJ')")):
    cursor.execute("SELECT T1.opensecrets_id, T1.thomas_id FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.type = ? AND T2.state = ? GROUP BY T1.opensecrets_id, T1.thomas_id", (type, state))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [{"opensecrets_id": row[0], "thomas_id": row[1]} for row in result]}

# Endpoint to get Google entity IDs of historical legislators by type and state
@app.get("/v1/legislator/google_entity_ids_by_type_and_state", operation_id="get_google_entity_ids_by_type_and_state", summary="Retrieve the Google Entity IDs of historical legislators based on the specified term type and state. The term type and state are used to filter the results, providing a targeted list of Google Entity IDs for legislators who meet the given criteria.")
async def get_google_entity_ids_by_type_and_state(type: str = Query(..., description="Type of the term (e.g., 'sen')"), state: str = Query(..., description="State abbreviation (e.g., 'NY')")):
    cursor.execute("SELECT T1.google_entity_id_id FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.type = ? AND T2.state = ?", (type, state))
    result = cursor.fetchall()
    if not result:
        return {"google_entity_ids": []}
    return {"google_entity_ids": [row[0] for row in result]}

# Endpoint to get religion bio of current legislators by RSS URL
@app.get("/v1/legislator/religion_bio_by_rss_url", operation_id="get_religion_bio_by_rss_url", summary="Retrieves the religious biography of current legislators associated with the provided RSS URL. The operation returns a consolidated list of unique religious biographies for the legislators who share the specified RSS URL.")
async def get_religion_bio_by_rss_url(rss_url: str = Query(..., description="RSS URL of the legislator")):
    cursor.execute("SELECT T1.religion_bio FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.rss_url = ? GROUP BY T1.religion_bio", (rss_url,))
    result = cursor.fetchall()
    if not result:
        return {"religion_bios": []}
    return {"religion_bios": [row[0] for row in result]}

# Endpoint to get party of current legislators by official full name
@app.get("/v1/legislator/party_by_official_full_name", operation_id="get_party_by_official_full_name", summary="Retrieves the political party affiliation of the current legislator with the specified official full name. The operation filters the legislators by their official full name and returns the corresponding party information.")
async def get_party_by_official_full_name(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT T2.party FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ? GROUP BY T2.party", (official_full_name,))
    result = cursor.fetchall()
    if not result:
        return {"parties": []}
    return {"parties": [row[0] for row in result]}

# Endpoint to get the district of a historical legislator based on their name and term type
@app.get("/v1/legislator/historical_district", operation_id="get_historical_district", summary="Retrieves the district of a historical legislator based on their first name, last name, and term type. The operation uses the provided first name and last name to identify the legislator, and the term type to determine the relevant term. The district associated with the legislator and term type is then returned.")
async def get_historical_district(last_name: str = Query(..., description="Last name of the legislator"), first_name: str = Query(..., description="First name of the legislator"), term_type: str = Query(..., description="Type of the term (e.g., 'rep')")):
    cursor.execute("SELECT T2.district FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.last_name = ? AND T1.first_name = ? AND T2.type = ?", (last_name, first_name, term_type))
    result = cursor.fetchall()
    if not result:
        return {"districts": []}
    return {"districts": [row[0] for row in result]}

# Endpoint to get the party and state of current legislators based on their OpenSecrets ID and Thomas ID
@app.get("/v1/legislator/current_party_state", operation_id="get_current_party_state", summary="Retrieves the political party and state of current legislators based on their unique OpenSecrets and Thomas IDs. The response includes a list of party and state combinations associated with the provided IDs.")
async def get_current_party_state(opensecrets_id: str = Query(..., description="OpenSecrets ID of the legislator"), thomas_id: int = Query(..., description="Thomas ID of the legislator")):
    cursor.execute("SELECT T2.party, T2.state FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.opensecrets_id = ? AND T1.thomas_id = ? GROUP BY T2.party, T2.state", (opensecrets_id, thomas_id))
    result = cursor.fetchall()
    if not result:
        return {"party_state": []}
    return {"party_state": [{"party": row[0], "state": row[1]} for row in result]}

# Endpoint to get the official full name and birthday of current legislators based on their contact form URL
@app.get("/v1/legislator/current_legislator_info", operation_id="get_current_legislator_info", summary="Retrieves the official full name and birthday of current legislators by using their contact form URL as a reference. This operation facilitates the identification of legislators based on their contact form URLs, providing essential biographical information.")
async def get_current_legislator_info(contact_form: str = Query(..., description="Contact form URL of the legislator")):
    cursor.execute("SELECT T1.official_full_name, T1.birthday_bio FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.contact_form = ?", (contact_form,))
    result = cursor.fetchall()
    if not result:
        return {"legislators": []}
    return {"legislators": [{"official_full_name": row[0], "birthday_bio": row[1]} for row in result]}

# Endpoint to get the state and type of historical legislators based on their Google entity ID
@app.get("/v1/legislator/historical_state_type", operation_id="get_historical_state_type", summary="Retrieves the state and type of historical legislators based on their Google entity ID. The operation uses the provided Google entity ID to look up the legislator's biographical information and historical terms, returning the corresponding state and type.")
async def get_historical_state_type(google_entity_id: str = Query(..., description="Google entity ID of the legislator")):
    cursor.execute("SELECT T2.state, T2.type FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.google_entity_id_id = ?", (google_entity_id,))
    result = cursor.fetchall()
    if not result:
        return {"state_type": []}
    return {"state_type": [{"state": row[0], "type": row[1]} for row in result]}

# Endpoint to get the type and end date of historical legislators based on their first and last name
@app.get("/v1/legislator/historical_type_end", operation_id="get_historical_type_end", summary="Retrieves the type and end date of historical legislators matching the provided first and last names. This operation uses the first and last names to search for legislators in the historical database and returns the corresponding type and end date from the historical-terms table.")
async def get_historical_type_end(first_name: str = Query(..., description="First name of the legislator"), last_name: str = Query(..., description="Last name of the legislator")):
    cursor.execute("SELECT T2.type, T2.end FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"type_end": []}
    return {"type_end": [{"type": row[0], "end": row[1]} for row in result]}

# Endpoint to get the difference in the number of senators and representatives for historical legislators based on gender and birth year range
@app.get("/v1/legislator/historical_sen_rep_diff", operation_id="get_historical_sen_rep_diff", summary="Retrieve the difference in the number of senators and representatives for historical legislators, filtered by gender and birth year range. The response provides a single integer value representing the difference between the total number of senators and representatives who match the specified gender and birth year range.")
async def get_historical_sen_rep_diff(gender: str = Query(..., description="Gender of the legislator"), start_year: str = Query(..., description="Start year of the birth year range in 'YYYY' format"), end_year: str = Query(..., description="End year of the birth year range in 'YYYY' format")):
    cursor.execute("SELECT SUM(CASE WHEN T2.type = 'sen' THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.type = 'rep' THEN 1 ELSE 0 END) FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.gender_bio = ? AND strftime('%Y', T1.birthday_bio) BETWEEN ? AND ?", (gender, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the percentage of independent legislators among current legislators based on gender and birth year range
@app.get("/v1/legislator/current_independent_percentage", operation_id="get_current_independent_percentage", summary="Retrieves the percentage of independent legislators among current legislators, filtered by gender and birth year range. The calculation considers the total count of legislators and the count of those with the 'Independent' party affiliation, within the specified gender and birth year range.")
async def get_current_independent_percentage(gender: str = Query(..., description="Gender of the legislator"), start_year: str = Query(..., description="Start year of the birth year range in 'YYYY' format"), end_year: str = Query(..., description="End year of the birth year range in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.party = 'Independent' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.party) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.gender_bio = ? AND strftime('%Y', T1.birthday_bio) BETWEEN ? AND ?", (gender, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the first and last name of a historical legislator based on their bioguide ID
@app.get("/v1/legislator/historical_legislator_name", operation_id="get_historical_legislator_name", summary="Retrieves the first and last name of a historical legislator using their unique bioguide identifier. This operation allows users to obtain the full name of a legislator based on their bioguide ID, providing essential information for historical legislator identification and research.")
async def get_historical_legislator_name(bioguide_id: str = Query(..., description="Bioguide ID of the legislator")):
    cursor.execute("SELECT first_name, last_name FROM historical WHERE bioguide_id = ?", (bioguide_id,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to check if a historical legislator has a Ballotpedia ID based on their first and last name
@app.get("/v1/legislator/historical_ballotpedia_check", operation_id="get_historical_ballotpedia_check", summary="This operation checks whether a historical legislator has a Ballotpedia ID based on their first and last name. It returns a response indicating whether the legislator 'have' or 'doesn't have' a Ballotpedia ID. The operation uses the provided first and last name of the legislator to perform the check.")
async def get_historical_ballotpedia_check(first_name: str = Query(..., description="First name of the legislator"), last_name: str = Query(..., description="Last name of the legislator")):
    cursor.execute("SELECT CASE WHEN ballotpedia_id IS NULL THEN 'doesn''t have' ELSE 'have' END AS HaveorNot FROM historical WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"ballotpedia_check": []}
    return {"ballotpedia_check": [row[0] for row in result]}

# Endpoint to get the count of historical legislators born in a specific year
@app.get("/v1/legislator/historical_legislator_count_by_birth_year", operation_id="get_historical_legislator_count_by_birth_year", summary="Retrieves the total number of historical legislators who were born in a specific year. The year of birth is provided as a four-digit input parameter, which is used to filter the historical legislator data and calculate the count.")
async def get_historical_legislator_count_by_birth_year(birth_year: str = Query(..., description="Birth year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(bioguide_id) FROM historical WHERE birthday_bio LIKE ?", (birth_year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last names of historical figures based on gender
@app.get("/v1/legislator/historical_figures_by_gender", operation_id="get_historical_figures_by_gender", summary="Retrieve the first and last names of historical figures based on the specified gender. The operation filters the historical figures by the provided gender and returns their names.")
async def get_historical_figures_by_gender(gender_bio: str = Query(..., description="Gender of the historical figure")):
    cursor.execute("SELECT first_name, last_name FROM historical WHERE gender_bio = ?", (gender_bio,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the count of districts in a given state
@app.get("/v1/legislator/count_districts_by_state", operation_id="get_count_districts_by_state", summary="Retrieves the total number of districts in a specified state. The state is identified using its abbreviation, such as 'ID' for Idaho.")
async def get_count_districts_by_state(state: str = Query(..., description="State abbreviation (e.g., 'ID')")):
    cursor.execute("SELECT COUNT(district) FROM `current-terms` WHERE state = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of current terms where class is NULL
@app.get("/v1/legislator/count_current_terms_null_class", operation_id="get_count_current_terms_null_class", summary="Retrieves the total count of legislators in the current term who do not have a class specified.")
async def get_count_current_terms_null_class():
    cursor.execute("SELECT COUNT(bioguide) FROM `current-terms` WHERE class IS NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of historical figures with a Wikipedia ID
@app.get("/v1/legislator/percentage_historical_with_wikipedia", operation_id="get_percentage_historical_with_wikipedia", summary="Retrieves the percentage of historical figures in the database who have a corresponding Wikipedia ID. This operation calculates the ratio of historical figures with a non-null Wikipedia ID to the total number of historical figures, providing a measure of Wikipedia coverage for these figures.")
async def get_percentage_historical_with_wikipedia():
    cursor.execute("SELECT CAST(SUM(CASE WHEN wikipedia_id IS NOT NULL THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(bioguide_id) FROM historical")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the first and last names of current legislators without an Instagram account
@app.get("/v1/legislator/current_legislators_without_instagram", operation_id="get_current_legislators_without_instagram", summary="Retrieves the first and last names of current legislators who do not have an associated Instagram account. This operation returns a list of legislators who are not linked to an Instagram profile, based on the provided data.")
async def get_current_legislators_without_instagram():
    cursor.execute("SELECT T1.first_name, T1.last_name FROM current AS T1 INNER JOIN `social-media` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.instagram IS NULL")
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the district and state of historical terms based on the start year
@app.get("/v1/legislator/historical_terms_by_start_year", operation_id="get_historical_terms_by_start_year", summary="Retrieves the district and state of historical terms that began in the specified start year. The start year should be provided in the 'YYYY%' format.")
async def get_historical_terms_by_start_year(start_year: str = Query(..., description="Start year in 'YYYY%' format")):
    cursor.execute("SELECT T2.district, T2.state FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.start LIKE ?", (start_year,))
    result = cursor.fetchall()
    if not result:
        return {"terms": []}
    return {"terms": result}

# Endpoint to get the district of historical terms based on the first and last name
@app.get("/v1/legislator/historical_terms_by_name", operation_id="get_historical_terms_by_name", summary="Retrieves the district associated with the historical figure whose first and last names are provided. This operation uses the provided names to search for a match in the historical data and returns the corresponding district from the historical-terms data.")
async def get_historical_terms_by_name(first_name: str = Query(..., description="First name of the historical figure"), last_name: str = Query(..., description="Last name of the historical figure")):
    cursor.execute("SELECT T2.district FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"districts": []}
    return {"districts": result}

# Endpoint to get the address of current terms based on the first name, last name, and start date
@app.get("/v1/legislator/current_terms_address_by_name_start_date", operation_id="get_current_terms_address_by_name_start_date", summary="Retrieve the address of a legislator's current term based on their first name, last name, and the term's start date. The operation filters the legislator by their first and last names, and then identifies the corresponding term using the provided start date. The address of the term is then returned.")
async def get_current_terms_address_by_name_start_date(first_name: str = Query(..., description="First name of the current legislator"), last_name: str = Query(..., description="Last name of the current legislator"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.address FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.first_name = ? AND T1.last_name = ? AND T2.start = ?", (first_name, last_name, start_date))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get the first and last names of current legislators based on the start year and state rank
@app.get("/v1/legislator/current_legislators_by_start_year_state_rank", operation_id="get_current_legislators_by_start_year_state_rank", summary="Retrieves the first and last names of current legislators who began their term in a specific year and hold a particular state rank. The start year should be provided in 'YYYY%' format, and the state rank can be 'junior' or 'senior'.")
async def get_current_legislators_by_start_year_state_rank(start_year: str = Query(..., description="Start year in 'YYYY%' format"), state_rank: str = Query(..., description="State rank (e.g., 'junior')")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.start LIKE ? AND T2.state_rank = ?", (start_year, state_rank))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the count of current legislators based on the start year, state, and gender
@app.get("/v1/legislator/count_current_legislators_by_start_year_state_gender", operation_id="get_count_current_legislators_by_start_year_state_gender", summary="Retrieves the total number of current legislators who began their term in a specific year, belong to a particular state, and identify with a certain gender. The response is based on the provided start year, state abbreviation, and gender.")
async def get_count_current_legislators_by_start_year_state_gender(start_year: str = Query(..., description="Start year in 'YYYY' format"), state: str = Query(..., description="State abbreviation (e.g., 'CA')"), gender_bio: str = Query(..., description="Gender of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE STRFTIME('%Y', T2.start) = ? AND T2.state = ? AND T1.gender_bio = ?", (start_year, state, gender_bio))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the Twitter ID of a legislator based on their first and last name
@app.get("/v1/legislator/twitter_id_by_name", operation_id="get_twitter_id_by_name", summary="Retrieves the Twitter ID of a legislator by matching the provided first and last name with the corresponding records in the database. The operation requires both the first and last name of the legislator to accurately identify the Twitter ID.")
async def get_twitter_id_by_name(first_name: str = Query(..., description="First name of the legislator"), last_name: str = Query(..., description="Last name of the legislator")):
    cursor.execute("SELECT T1.twitter_id FROM `social-media` AS T1 INNER JOIN current AS T2 ON T2.bioguide_id = T1.bioguide WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"twitter_id": []}
    return {"twitter_id": result[0]}

# Endpoint to get the Facebook IDs of legislators based on their party affiliation
@app.get("/v1/legislator/facebook_ids_by_party", operation_id="get_facebook_ids_by_party", summary="Retrieve the Facebook IDs of legislators who belong to a specific political party. The operation filters legislators based on their party affiliation and returns a list of unique Facebook IDs associated with them.")
async def get_facebook_ids_by_party(party: str = Query(..., description="Party affiliation of the legislator")):
    cursor.execute("SELECT T2.facebook_id FROM `current-terms` AS T1 INNER JOIN `social-media` AS T2 ON T1.bioguide = T2.bioguide WHERE T1.party = ? GROUP BY T2.facebook_id", (party,))
    result = cursor.fetchall()
    if not result:
        return {"facebook_ids": []}
    return {"facebook_ids": [row[0] for row in result]}

# Endpoint to get the names of historical legislators based on the end date of their term and gender
@app.get("/v1/legislator/historical_names_by_end_date_gender", operation_id="get_historical_names_by_end_date_gender", summary="Retrieve the first and last names of historical legislators whose term ended on a specific date and who identify with a certain gender. The end date should be provided in 'YYYY-MM-DD' format.")
async def get_historical_names_by_end_date_gender(end_date: str = Query(..., description="End date of the term in 'YYYY-MM-DD' format"), gender: str = Query(..., description="Gender of the legislator")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.end = ? AND T1.gender_bio = ?", (end_date, gender))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the names of current legislators based on their religion and state
@app.get("/v1/legislator/current_names_by_religion_state", operation_id="get_current_names_by_religion_state", summary="Retrieve the first and last names of current legislators who share a specified religious affiliation and represent a particular state. The results are grouped by the legislators' names.")
async def get_current_names_by_religion_state(religion: str = Query(..., description="Religion of the legislator"), state: str = Query(..., description="State of the legislator")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.religion_bio = ? AND T2.state = ? GROUP BY T1.first_name, T1.last_name", (religion, state))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the percentage of current legislators with Wikipedia IDs compared to historical legislators
@app.get("/v1/legislator/percentage_current_with_wikipedia", operation_id="get_percentage_current_with_wikipedia", summary="Retrieves the percentage of current legislators who have Wikipedia IDs, calculated by comparing the count of current legislators with Wikipedia IDs to the total count of historical legislators with Wikipedia IDs.")
async def get_percentage_current_with_wikipedia():
    cursor.execute("SELECT CAST(COUNT(CASE WHEN wikipedia_id IS NOT NULL THEN bioguide_id ELSE 0 END) AS REAL) * 100 / ( SELECT COUNT(CASE WHEN wikipedia_id IS NOT NULL THEN bioguide_id ELSE 0 END) FROM historical ) FROM current")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of current legislators who started their term within a specific year range
@app.get("/v1/legislator/percentage_current_by_start_year_range", operation_id="get_percentage_current_by_start_year_range", summary="Get the percentage of current legislators who started their term within a specific year range")
async def get_percentage_current_by_start_year_range(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN strftime('%Y', T2.start) BETWEEN ? AND ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.bioguide_id) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of historical legislators with a specific religion and no Ballotpedia ID
@app.get("/v1/legislator/count_historical_by_religion_no_ballotpedia", operation_id="get_count_historical_by_religion_no_ballotpedia", summary="Retrieves the total count of historical legislators who adhere to a specific religious belief and do not have a Ballotpedia ID. The input parameter specifies the religion to filter the count.")
async def get_count_historical_by_religion_no_ballotpedia(religion: str = Query(..., description="Religion of the legislator")):
    cursor.execute("SELECT COUNT(bioguide_id) FROM historical WHERE religion_bio = ? AND ballotpedia_id IS NULL", (religion,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of current legislators based on their class and party affiliation
@app.get("/v1/legislator/count_current_by_class_party", operation_id="get_count_current_by_class_party", summary="Get the count of current legislators based on their class and party affiliation")
async def get_count_current_by_class_party(class_of_legislator: int = Query(..., description="Class of the legislator"), party: str = Query(..., description="Party affiliation of the legislator")):
    cursor.execute("SELECT COUNT(bioguide) FROM `current-terms` WHERE class = ? AND party = ?", (class_of_legislator, party))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of historical legislators based on gender and missing IDs
@app.get("/v1/legislator/historical_names_by_gender_missing_ids", operation_id="get_historical_names_by_gender_missing_ids", summary="Retrieves the first and last names of historical legislators who identify as the specified gender and lack both a Google Entity ID and a Federal Election Commission (FEC) ID.")
async def get_historical_names_by_gender_missing_ids(gender: str = Query(..., description="Gender of the legislator")):
    cursor.execute("SELECT first_name, last_name FROM historical WHERE gender_bio = ? AND google_entity_id_id IS NULL AND fec_id IS NULL", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of current legislators based on state, type, and end date pattern
@app.get("/v1/legislator/count_current_by_state_type_end_pattern", operation_id="get_count_current_by_state_type_end_pattern", summary="Retrieves the total number of current legislators in a specified state, of a certain type, and whose term ends with a given year pattern. The state, type, and end date pattern are provided as input parameters.")
async def get_count_current_by_state_type_end_pattern(state: str = Query(..., description="State of the legislator"), type: str = Query(..., description="Type of the legislator"), end_pattern: str = Query(..., description="End date pattern in 'YYYY%' format")):
    cursor.execute("SELECT COUNT(*) FROM `current-terms` WHERE state = ? AND type = ? AND end LIKE ?", (state, type, end_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first name and last name of a historical figure based on a limit
@app.get("/v1/legislator/historical_figure_names", operation_id="get_historical_figure_names", summary="Retrieves the first and last names of historical figures, ordered by their birthdays. The number of results returned is limited by the provided limit parameter.")
async def get_historical_figure_names(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT first_name, last_name FROM historical ORDER BY birthday_bio LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the bioguide IDs of current representatives based on type, party, end date, and district
@app.get("/v1/legislator/current_representatives_bioguide", operation_id="get_current_representatives_bioguide", summary="Retrieve the unique biographical identifiers (bioguide IDs) of current representatives based on their type, political party, end date of term, and district. This operation filters the current representatives using the provided input parameters and returns the corresponding bioguide IDs.")
async def get_current_representatives_bioguide(type: str = Query(..., description="Type of representative"), party: str = Query(..., description="Party of the representative"), end: str = Query(..., description="End date in 'YYYY-MM-DD' format"), district: int = Query(..., description="District number")):
    cursor.execute("SELECT bioguide FROM `current-terms` WHERE type = ? AND party = ? AND end = ? AND district = ?", (type, party, end, district))
    result = cursor.fetchall()
    if not result:
        return {"bioguide_ids": []}
    return {"bioguide_ids": result}

# Endpoint to get the Twitter handle of a current legislator based on their official full name
@app.get("/v1/legislator/current_legislator_twitter", operation_id="get_current_legislator_twitter", summary="Retrieves the Twitter handle of a current legislator using their official full name. The operation searches for the legislator in the current database and returns the associated Twitter handle from the social-media database.")
async def get_current_legislator_twitter(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT T2.twitter FROM current AS T1 INNER JOIN `social-media` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ?", (official_full_name,))
    result = cursor.fetchall()
    if not result:
        return {"twitter_handles": []}
    return {"twitter_handles": result}

# Endpoint to get the official full names of current legislators based on party, start year, type, and caucus
@app.get("/v1/legislator/current_legislators_by_criteria", operation_id="get_current_legislators_by_criteria", summary="Retrieve the official full names of current legislators who meet the specified criteria. The criteria include the legislator's party, the start year of their term, the type of legislator, and their caucus. The start year should be provided in 'YYYY' format.")
async def get_current_legislators_by_criteria(party: str = Query(..., description="Party of the legislator"), start_year: str = Query(..., description="Start year in 'YYYY' format"), type: str = Query(..., description="Type of legislator"), caucus: str = Query(..., description="Caucus of the legislator")):
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.party = ? AND strftime('%Y', T2.start) >= ? AND T2.type = ? AND T2.caucus = ?", (party, start_year, type, caucus))
    result = cursor.fetchall()
    if not result:
        return {"official_full_names": []}
    return {"official_full_names": result}

# Endpoint to get the count of current legislators without a Facebook account
@app.get("/v1/legislator/count_legislators_without_facebook", operation_id="get_count_legislators_without_facebook", summary="Retrieves the total number of current legislators who do not have a Facebook account. This operation calculates the count by querying the current legislators and cross-referencing them with the social media database, specifically looking for those with a null value in the Facebook field.")
async def get_count_legislators_without_facebook():
    cursor.execute("SELECT COUNT(T3.bioguide_id) FROM ( SELECT T1.bioguide_id FROM current AS T1 INNER JOIN `social-media` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.facebook IS NULL GROUP BY T1.bioguide_id ) T3")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common party of current legislators based on their religion
@app.get("/v1/legislator/most_common_party_by_religion", operation_id="get_most_common_party_by_religion", summary="Retrieves the political party with the highest representation among current legislators who share a specified religious affiliation. The response is sorted in descending order based on the number of legislators per party, and the results can be limited to a specified number.")
async def get_most_common_party_by_religion(religion_bio: str = Query(..., description="Religion of the legislator"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.party FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.religion_bio = ? GROUP BY T2.party ORDER BY COUNT(T2.party) DESC LIMIT ?", (religion_bio, limit))
    result = cursor.fetchall()
    if not result:
        return {"parties": []}
    return {"parties": result}

# Endpoint to get the official full names of current legislators with all social media accounts
@app.get("/v1/legislator/legislators_with_all_social_media", operation_id="get_legislators_with_all_social_media", summary="Retrieves the official full names of current legislators who have active accounts on all major social media platforms. This operation returns a list of names for legislators who are present on Facebook, Instagram, Twitter, and YouTube.")
async def get_legislators_with_all_social_media():
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `social-media` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.facebook IS NOT NULL AND T2.instagram IS NOT NULL AND T2.twitter IS NOT NULL AND T2.youtube IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"official_full_names": []}
    return {"official_full_names": result}

# Endpoint to get the count of districts for a current legislator based on their official full name
@app.get("/v1/legislator/count_districts_by_legislator", operation_id="get_count_districts_by_legislator", summary="Retrieves the total number of districts associated with a legislator, identified by their official full name. This operation provides a count of unique districts in which the legislator has served, offering insights into their political representation and constituency.")
async def get_count_districts_by_legislator(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT COUNT(T3.district) FROM ( SELECT T2.district FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ? GROUP BY T2.district ) T3", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the Wikipedia IDs of historical legislators based on type and party
@app.get("/v1/legislator/historical_legislators_wikipedia_ids", operation_id="get_historical_legislators_wikipedia_ids", summary="Retrieves the Wikipedia IDs of historical legislators based on their type and party. The operation filters legislators by the specified type and party, and returns their corresponding Wikipedia IDs. This enables users to access detailed information about the selected historical legislators on Wikipedia.")
async def get_historical_legislators_wikipedia_ids(type: str = Query(..., description="Type of legislator"), party: str = Query(..., description="Party of the legislator")):
    cursor.execute("SELECT T1.wikipedia_id FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.type = ? AND T2.party = ?", (type, party))
    result = cursor.fetchall()
    if not result:
        return {"wikipedia_ids": []}
    return {"wikipedia_ids": result}

# Endpoint to get the official full names of current legislators based on the duration of their terms and the count of terms
@app.get("/v1/legislator/legislators_by_term_duration_and_count", operation_id="get_legislators_by_term_duration_and_count", summary="Retrieve the official full names of current legislators who have served for a specific term duration and have been elected for a certain number of terms. The duration of the term is provided in years, and the count of terms is also specified.")
async def get_legislators_by_term_duration_and_count(term_duration: int = Query(..., description="Duration of the term in years"), term_count: int = Query(..., description="Count of terms")):
    cursor.execute("SELECT DISTINCT CASE WHEN SUM(CAST(strftime('%Y', T2.end) AS int) - CAST(strftime('%Y', T2.start) AS int)) = ? THEN T1.official_full_name END FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide GROUP BY T1.official_full_name, T2.district HAVING COUNT(T1.official_full_name) = ?", (term_duration, term_count))
    result = cursor.fetchall()
    if not result:
        return {"official_full_names": []}
    return {"official_full_names": result}

# Endpoint to get the count of historical legislators based on first name, party, and type
@app.get("/v1/legislator/count_historical_legislators", operation_id="get_count_historical_legislators", summary="Retrieves the total count of historical legislators that match the specified first name, party, and type. The count is determined by aggregating unique bioguide IDs from the historical legislator and historical-terms tables, based on the provided parameters.")
async def get_count_historical_legislators(first_name: str = Query(..., description="First name of the legislator"), party: str = Query(..., description="Party of the legislator"), type: str = Query(..., description="Type of the legislator")):
    cursor.execute("SELECT COUNT(T.bioguide_id) FROM ( SELECT T1.bioguide_id FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.first_name = ? AND T2.party = ? AND T2.type = ? GROUP BY T1.bioguide_id ) AS T", (first_name, party, type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of results based on gender, state, type, and duration
@app.get("/v1/legislator/sum_results_gender_state_type_duration", operation_id="get_sum_results_gender_state_type_duration", summary="Retrieves the total sum of results based on the gender, state, and type of the legislator, considering a duration of more than 10 years. The duration is calculated as the difference between the end and start years of the legislator's term. The result is a single numerical value representing the sum of all qualifying results.")
async def get_sum_results_gender_state_type_duration(duration: int = Query(..., description="Duration in years"), gender: str = Query(..., description="Gender of the legislator"), state: str = Query(..., description="State of the legislator"), type: str = Query(..., description="Type of the legislator")):
    cursor.execute("SELECT SUM(T3.result) FROM ( SELECT CASE WHEN SUM(CAST(strftime('%Y', T2.`end`) AS int) - CAST(strftime('%Y', T2.start) AS int)) > ? THEN 1 ELSE 0 END AS result FROM current AS T1 INNER JOIN \"current-terms\" AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.gender_bio = ? AND T2.state = ? AND T2.type = ? ) AS T3", (duration, gender, state, type))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the party of the oldest historical legislator
@app.get("/v1/legislator/oldest_historical_legislator_party", operation_id="get_oldest_historical_legislator_party", summary="Retrieves the political party affiliation of the oldest historical legislator, based on their birth date. This operation returns the party of the legislator who was born the earliest, as determined by the historical records.")
async def get_oldest_historical_legislator_party():
    cursor.execute("SELECT T1.party FROM `historical-terms` AS T1 INNER JOIN historical AS T2 ON T2.bioguide_id = T1.bioguide ORDER BY T2.birthday_bio LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"party": []}
    return {"party": result[0]}

# Endpoint to get the official full name of legislators based on religion, state, type, and duration
@app.get("/v1/legislator/official_full_name_religion_state_type_duration", operation_id="get_official_full_name_religion_state_type_duration", summary="Retrieve the full names of legislators who have served for a specified duration, based on their religion, state, and type. The duration is calculated as the difference between the end and start years of their service.")
async def get_official_full_name_religion_state_type_duration(duration: int = Query(..., description="Duration in years"), religion: str = Query(..., description="Religion of the legislator"), state: str = Query(..., description="State of the legislator"), type: str = Query(..., description="Type of the legislator")):
    cursor.execute("SELECT CASE WHEN SUM(CAST(strftime('%Y', T2.end) AS int) - CAST(strftime('%Y', T2.start) AS int)) = ? THEN official_full_name END FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.religion_bio = ? AND T2.state = ? AND T2.type = ?", (duration, religion, state, type))
    result = cursor.fetchone()
    if not result:
        return {"official_full_name": []}
    return {"official_full_name": result[0]}

# Endpoint to get the count of current legislators born after a certain year and without a Google entity ID
@app.get("/v1/legislator/count_current_legislators_birth_year_no_google_id", operation_id="get_count_current_legislators_birth_year_no_google_id", summary="Get the count of current legislators born after a certain year and without a Google entity ID")
async def get_count_current_legislators_birth_year_no_google_id(birth_year: int = Query(..., description="Birth year of the legislator in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM current WHERE strftime('%Y', birthday_bio) > ? AND google_entity_id_id IS NULL", (birth_year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the official full names of current legislators with a house history ID
@app.get("/v1/legislator/official_full_names_with_house_history_id", operation_id="get_official_full_names_with_house_history_id", summary="Retrieves the full names of legislators who are currently in office and have a house history ID. This operation does not return names of legislators who are not currently in office or do not have a house history ID.")
async def get_official_full_names_with_house_history_id():
    cursor.execute("SELECT official_full_name FROM current WHERE house_history_id IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"official_full_names": []}
    return {"official_full_names": [row[0] for row in result]}

# Endpoint to get the count of current legislators with both ICPSR and MapLight IDs
@app.get("/v1/legislator/count_current_legislators_icpsr_maplight_ids", operation_id="get_count_current_legislators_icpsr_maplight_ids", summary="Retrieves the total number of current legislators who have both ICPSR and MapLight identifiers. This operation does not return individual legislator details, but rather a single count value.")
async def get_count_current_legislators_icpsr_maplight_ids():
    cursor.execute("SELECT COUNT(*) FROM current WHERE icpsr_id IS NOT NULL AND maplight_id IS NOT NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of current legislators with a LIS ID based on gender
@app.get("/v1/legislator/count_current_legislators_lis_id_gender", operation_id="get_count_current_legislators_lis_id_gender", summary="Retrieves the total number of current legislators who have a LIS ID and match the specified gender. The gender parameter is used to filter the results.")
async def get_count_current_legislators_lis_id_gender(gender: str = Query(..., description="Gender of the legislator")):
    cursor.execute("SELECT COUNT(lis_id) FROM current WHERE gender_bio = ? AND lis_id IS NOT NULL", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the religion of a current legislator based on their official full name
@app.get("/v1/legislator/religion_current_legislator_official_full_name", operation_id="get_religion_current_legislator_official_full_name", summary="Retrieves the religious affiliation of a current legislator using their official full name as the identifier.")
async def get_religion_current_legislator_official_full_name(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT religion_bio FROM current WHERE official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"religion": []}
    return {"religion": result[0]}

# Endpoint to get the most common religion among current legislators
@app.get("/v1/legislator/most_common_religion_current_legislators", operation_id="get_most_common_religion_current_legislators", summary="Retrieves the most frequently occurring religion among the current legislators. The operation returns the religion with the highest count, based on the data in the 'current' table.")
async def get_most_common_religion_current_legislators():
    cursor.execute("SELECT religion_bio FROM current GROUP BY religion_bio ORDER BY COUNT(religion_bio) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"religion": []}
    return {"religion": result[0]}

# Endpoint to get the Instagram handle of a legislator by their official full name
@app.get("/v1/legislator/instagram_by_name", operation_id="get_instagram_by_name", summary="Retrieves the Instagram handle of a legislator based on their official full name. The operation searches for a legislator with the provided official full name and returns the associated Instagram handle from the social media records.")
async def get_instagram_by_name(official_full_name: str = Query(..., description="Official full name of the legislator")):
    cursor.execute("SELECT T2.instagram FROM current AS T1 INNER JOIN `social-media` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ?", (official_full_name,))
    result = cursor.fetchone()
    if not result:
        return {"instagram": []}
    return {"instagram": result[0]}

# Endpoint to get the count of legislators with both Thomas ID and Instagram handle
@app.get("/v1/legislator/count_with_thomas_and_instagram", operation_id="get_count_with_thomas_and_instagram", summary="Retrieves the total number of unique legislators who have a Thomas ID and an Instagram account. This operation combines data from the current legislator records and the social media records to provide an accurate count.")
async def get_count_with_thomas_and_instagram():
    cursor.execute("SELECT COUNT(DISTINCT T1.bioguide_id) FROM current AS T1 INNER JOIN `social-media` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.thomas_id IS NOT NULL AND T2.instagram IS NOT NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the Facebook handles of legislators who have a Wikipedia ID
@app.get("/v1/legislator/facebook_by_wikipedia_id", operation_id="get_facebook_by_wikipedia_id", summary="Retrieve the Facebook handles of legislators who have a Wikipedia ID. This operation fetches the Facebook handles from the social-media table, which are associated with legislators in the current table based on their bioguide IDs. The results are grouped by Facebook handle to ensure uniqueness.")
async def get_facebook_by_wikipedia_id():
    cursor.execute("SELECT T2.facebook FROM current AS T1 INNER JOIN `social-media` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.wikipedia_id IS NOT NULL GROUP BY T2.facebook")
    result = cursor.fetchall()
    if not result:
        return {"facebook": []}
    return {"facebook": [row[0] for row in result]}

# Endpoint to get the count of legislators with the earliest birthday
@app.get("/v1/legislator/count_earliest_birthday", operation_id="get_count_earliest_birthday", summary="Retrieves the number of legislators who share the earliest birthdate. This operation identifies the earliest birthdate among all legislators and returns the count of legislators born on that date.")
async def get_count_earliest_birthday():
    cursor.execute("SELECT COUNT(T2.bioguide) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.birthday_bio = ( SELECT MIN(birthday_bio) FROM current )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone number of a legislator by their official full name and term start date
@app.get("/v1/legislator/phone_by_name_and_start_date", operation_id="get_phone_by_name_and_start_date", summary="Retrieve the phone number of a legislator based on their official full name and term start date. The input parameters include the legislator's official full name and the term start date in 'YYYY-MM-DD' format. This operation returns the phone number associated with the legislator's current term, as determined by the provided start date.")
async def get_phone_by_name_and_start_date(official_full_name: str = Query(..., description="Official full name of the legislator"), start_date: str = Query(..., description="Term start date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.phone FROM `current-terms` AS T1 INNER JOIN current AS T2 ON T2.bioguide_id = T1.bioguide WHERE T2.official_full_name = ? AND T1.start = ?", (official_full_name, start_date))
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the count of legislators by their official full name and party
@app.get("/v1/legislator/count_by_name_and_party", operation_id="get_count_by_name_and_party", summary="Retrieves the total number of legislators who share a specific official full name and belong to a particular political party. The response is based on the provided legislator's full name and party affiliation.")
async def get_count_by_name_and_party(official_full_name: str = Query(..., description="Official full name of the legislator"), party: str = Query(..., description="Party of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ? AND T2.party = ?", (official_full_name, party))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the official full names of legislators by term start date
@app.get("/v1/legislator/names_by_start_date", operation_id="get_names_by_start_date", summary="Retrieve the official full names of legislators who began their term on the specified start date. The start date must be provided in 'YYYY-MM-DD' format.")
async def get_names_by_start_date(start_date: str = Query(..., description="Term start date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.start = ?", (start_date,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of legislators by term start date and gender
@app.get("/v1/legislator/count_by_start_date_and_gender", operation_id="get_count_by_start_date_and_gender", summary="Retrieves the total number of legislators who began their term on a specific date and belong to a particular gender category. The response is based on the provided term start date and gender.")
async def get_count_by_start_date_and_gender(start_date: str = Query(..., description="Term start date in 'YYYY-MM-DD' format"), gender: str = Query(..., description="Gender of the legislator")):
    cursor.execute("SELECT COUNT(*) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.start = ? AND T1.gender_bio = ?", (start_date, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the official full name of the legislator with the most terms
@app.get("/v1/legislator/most_terms", operation_id="get_most_terms", summary="Retrieves the full name of the legislator who has served the most terms in office. This operation returns the official full name of the legislator with the highest number of terms served, as determined by the count of their bioguide ID in the current-terms table.")
async def get_most_terms():
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide GROUP BY T1.official_full_name, T2.bioguide ORDER BY COUNT(T2.bioguide) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of legislators by their official full name and district
@app.get("/v1/legislator/count_by_name_and_district", operation_id="get_count_by_name_and_district", summary="Retrieves the total number of legislators who share a specific official full name and district. The response is based on the current legislator data, filtered by the provided official full name and district number.")
async def get_count_by_name_and_district(official_full_name: str = Query(..., description="Official full name of the legislator"), district: int = Query(..., description="District number")):
    cursor.execute("SELECT COUNT(*) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.official_full_name = ? AND T2.district = ?", (official_full_name, district))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the official full name of a legislator based on their Facebook handle
@app.get("/v1/legislator/official_full_name_by_facebook", operation_id="get_official_full_name_by_facebook", summary="Retrieves the official full name of a legislator using their Facebook handle. The operation searches for the legislator's Facebook handle in the social-media table and returns the corresponding official full name from the current table.")
async def get_official_full_name_by_facebook(facebook: str = Query(..., description="Facebook handle of the legislator")):
    cursor.execute("SELECT T1.official_full_name FROM current AS T1 INNER JOIN `social-media` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.facebook = ?", (facebook,))
    result = cursor.fetchone()
    if not result:
        return {"official_full_name": []}
    return {"official_full_name": result[0]}

# Endpoint to compare the number of terms served by two legislators
@app.get("/v1/legislator/compare_terms_served", operation_id="compare_terms_served", summary="This endpoint compares the number of terms served by two legislators and returns the name of the legislator who has served more terms. The comparison is based on the legislators' full names provided as input parameters.")
async def compare_terms_served(name1: str = Query(..., description="Full name of the first legislator"), name2: str = Query(..., description="Full name of the second legislator")):
    cursor.execute("SELECT CASE WHEN SUM(CASE WHEN T1.official_full_name = ? THEN 1 ELSE 0 END) > SUM(CASE WHEN T1.official_full_name = ? THEN 1 ELSE 0 END) THEN ? ELSE ? END FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide", (name1, name2, name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"legislator_with_more_terms": []}
    return {"legislator_with_more_terms": result[0]}

# Endpoint to get the percentage of legislators with a specific gender who have served more than a certain number of terms
@app.get("/v1/legislator/percentage_gender_terms", operation_id="get_percentage_gender_terms", summary="Retrieve the percentage of legislators of a specified gender who have served more than a given number of terms. This operation calculates the proportion of legislators with the provided gender who have served more than the specified minimum number of terms.")
async def get_percentage_gender_terms(gender: str = Query(..., description="Gender of the legislator (e.g., 'F' for female)"), min_terms: int = Query(..., description="Minimum number of terms served")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN gender_bio = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.bioguide) FROM ( SELECT T2.bioguide, T1.gender_bio FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide GROUP BY T2.bioguide HAVING COUNT(T2.bioguide) > ? ) T3", (gender, min_terms))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of legislators with a Wikipedia page
@app.get("/v1/legislator/ratio_with_wikipedia", operation_id="get_ratio_with_wikipedia", summary="Retrieves the proportion of current legislators who have a corresponding Wikipedia page. This endpoint calculates the ratio by dividing the count of legislators with a Wikipedia ID by the total number of unique legislators.")
async def get_ratio_with_wikipedia():
    cursor.execute("SELECT CAST(COUNT(T2.bioguide) AS REAL) / COUNT(DISTINCT T1.bioguide_id) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.wikipedia_id IS NOT NULL")
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of legislators based on gender
@app.get("/v1/legislator/count_by_gender", operation_id="get_count_by_gender", summary="Retrieves the total number of legislators of a specified gender. The gender parameter is used to filter the count.")
async def get_count_by_gender(gender: str = Query(..., description="Gender of the legislator (e.g., 'F' for female)")):
    cursor.execute("SELECT COUNT(gender_bio) FROM current WHERE gender_bio = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of historical terms based on state and type
@app.get("/v1/legislator/count_historical_terms", operation_id="get_count_historical_terms", summary="Retrieves the total count of historical terms for a specific state and term type. The operation filters the historical terms based on the provided state and term type, and returns the count of matching records.")
async def get_count_historical_terms(state: str = Query(..., description="State (e.g., 'NJ')"), term_type: str = Query(..., description="Type of term (e.g., 'rep')")):
    cursor.execute("SELECT COUNT(type) FROM `historical-terms` WHERE state = ? AND type = ?", (state, term_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the address based on the contact form URL
@app.get("/v1/legislator/address_by_contact_form", operation_id="get_address_by_contact_form", summary="Retrieves the address associated with the specified contact form URL. This operation searches the current terms database for the address linked to the provided contact form URL.")
async def get_address_by_contact_form(contact_form: str = Query(..., description="Contact form URL")):
    cursor.execute("SELECT address FROM `current-terms` WHERE contact_form = ?", (contact_form,))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get the bioguide ID of a legislator based on their religion and state
@app.get("/v1/legislator/bioguide_by_religion_state", operation_id="get_bioguide_by_religion_state", summary="Retrieves the biographical guide identifier of a legislator who practices a specified religion and represents a given state. The response is based on data from the current legislative term.")
async def get_bioguide_by_religion_state(religion: str = Query(..., description="Religion of the legislator (e.g., 'Catholic')"), state: str = Query(..., description="State (e.g., 'NE')")):
    cursor.execute("SELECT T1.bioguide FROM `current-terms` AS T1 INNER JOIN current AS T2 ON T2.bioguide_id = T1.bioguide WHERE T2.religion_bio = ? AND T1.state = ?", (religion, state))
    result = cursor.fetchone()
    if not result:
        return {"bioguide": []}
    return {"bioguide": result[0]}

# Endpoint to get the maplight IDs of historical legislators based on term type and state
@app.get("/v1/legislator/maplight_ids_by_term_state", operation_id="get_maplight_ids_by_term_state", summary="Retrieve the unique identifiers of historical legislators who served a specific term type in a given state. The term type and state are used to filter the results, providing a targeted list of legislators that meet the specified criteria.")
async def get_maplight_ids_by_term_state(term_type: str = Query(..., description="Type of term (e.g., 'rep')"), state: str = Query(..., description="State (e.g., 'ME')")):
    cursor.execute("SELECT T1.maplight_id FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.type = ? AND T2.state = ? GROUP BY T1.maplight_id", (term_type, state))
    result = cursor.fetchall()
    if not result:
        return {"maplight_ids": []}
    return {"maplight_ids": [row[0] for row in result]}

# Endpoint to get historical terms based on birthday
@app.get("/v1/legislator/historical_terms_by_birthday", operation_id="get_historical_terms_by_birthday", summary="Retrieves historical terms associated with a legislator based on their birthday. The birthday should be provided in 'YYYY-MM-DD' format. The operation returns the type and start date of each historical term.")
async def get_historical_terms_by_birthday(birthday_bio: str = Query(..., description="Birthday of the legislator in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.type, T2.start FROM historical AS T1 INNER JOIN `historical-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T1.birthday_bio = ?", (birthday_bio,))
    result = cursor.fetchall()
    if not result:
        return {"terms": []}
    return {"terms": result}

# Endpoint to get the count of legislators based on type, state, and gender
@app.get("/v1/legislator/count_legislators_by_type_state_gender", operation_id="get_count_legislators_by_type_state_gender", summary="Retrieves the total number of legislators based on their type, state, and gender. The operation filters legislators by the provided type, state, and gender, and then returns the count of legislators that match the specified criteria.")
async def get_count_legislators_by_type_state_gender(type: str = Query(..., description="Type of the legislator (e.g., 'rep')"), state: str = Query(..., description="State of the legislator (e.g., 'MI')"), gender_bio: str = Query(..., description="Gender of the legislator (e.g., 'F')")):
    cursor.execute("SELECT COUNT(T.bioguide_id) FROM ( SELECT T1.bioguide_id FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.type = ? AND T2.state = ? AND T1.gender_bio = ? GROUP BY T1.bioguide_id ) T", (type, state, gender_bio))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of legislators of a specific type based on state and gender
@app.get("/v1/legislator/percentage_legislators_by_type_state_gender", operation_id="get_percentage_legislators_by_type_state_gender", summary="Retrieves the percentage of legislators of a specific type, based on the provided state and gender. This operation calculates the proportion of legislators that match the given type, state, and gender, out of all legislators in the specified state.")
async def get_percentage_legislators_by_type_state_gender(type: str = Query(..., description="Type of the legislator (e.g., 'sen')"), state: str = Query(..., description="State of the legislator (e.g., 'ME')"), gender_bio: str = Query(..., description="Gender of the legislator (e.g., 'F')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.type = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.type) FROM current AS T1 INNER JOIN `current-terms` AS T2 ON T1.bioguide_id = T2.bioguide WHERE T2.state = ? AND T1.gender_bio = ?", (type, state, gender_bio))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/legislator/count_current_ballotpedia_id?ballotpedia_id=",
    "/v1/legislator/official_full_name_cspan_id?cspan_id=",
    "/v1/legislator/count_current_birthday_bio?birthday_bio=1961-01-01",
    "/v1/legislator/count_current_fec_id_gender?fec_id=&gender_bio=F",
    "/v1/legislator/google_entity_id_official_full_name?official_full_name=Sherrod%20Brown",
    "/v1/legislator/oldest_official_full_name?name1=Sherrod%20Brown&name2=Maria%20Cantwell",
    "/v1/legislator/facebook_official_full_name?official_full_name=Todd%20Young",
    "/v1/legislator/count_current_no_instagram",
    "/v1/legislator/official_full_name_twitter_id?twitter_id=234128524",
    "/v1/legislator/youtube_gender_bio?gender_bio=F",
    "/v1/legislator/facebook_by_birthday",
    "/v1/legislator/count_no_instagram_opensecrets?opensecrets_id=",
    "/v1/legislator/count_name_district?official_full_name=Roger%20F.%20Wicker&district=",
    "/v1/legislator/count_by_name?official_full_name=Sherrod%20Brown",
    "/v1/legislator/official_names_with_state_rank",
    "/v1/legislator/state_by_start_date_name?start_date=1993-01-05&official_full_name=Sherrod%20Brown",
    "/v1/legislator/count_gender_terms?gender_bio=F&min_terms=4",
    "/v1/legislator/count_birthday_terms?birthday_bio=1960-01-01&min_terms=6",
    "/v1/legislator/proportion_gender?gender_bio=F",
    "/v1/legislator/percentage_no_instagram_religion?religion_bio=Roman%20Catholic",
    "/v1/legislator/current_count_by_gender?gender_bio=M",
    "/v1/legislator/current_terms_count_by_party?party=Republican",
    "/v1/legislator/social_media_count_with_instagram",
    "/v1/legislator/historical_count_by_gender?gender_bio=F",
    "/v1/legislator/current_count_by_religion_and_gender?religion_bio=Roman%20Catholic&gender_bio=M",
    "/v1/legislator/party_by_name_and_start_year?first_name=Sherrod&last_name=Brown&start_year=2005",
    "/v1/legislator/official_full_name_by_state_rank_type_start_year?state_rank=senior&type=sen&start_year=2013",
    "/v1/legislator/youtube_by_official_full_name?official_full_name=Chris%20Van%20Hollen",
    "/v1/legislator/social_media_count_by_name?first_name=Mark&last_name=Warner",
    "/v1/legislator/last_names_by_state?state=CA",
    "/v1/legislator/names_by_type_and_gender?type=sen&gender_bio=F",
    "/v1/legislator/govtrack_by_full_name?official_full_name=Chris%20Van%20Hollen",
    "/v1/legislator/twitter_by_full_name?official_full_name=Roger%20F.%20Wicker",
    "/v1/legislator/historical_names_by_party_and_date?party=Pro-Administration&start_date=1791-01-01&end_date=1791-12-31",
    "/v1/legislator/current_names_by_party_and_gender?party=Republican&gender_bio=F",
    "/v1/legislator/district_by_full_name?official_full_name=Chris%20Van%20Hollen",
    "/v1/legislator/count_by_district_and_name?district=20&first_name=Richard&last_name=Durbin",
    "/v1/legislator/proportion_by_gender_date_party?divisor=22&gender_bio=M&start_date=2000-01-01&end_date=2021-12-31&party=Democrat",
    "/v1/legislator/percentage_current_vs_historical_by_gender?gender_bio=F",
    "/v1/legislator/youtube_id_by_username?youtube=RepWassermanSchultz",
    "/v1/legislator/facebook_by_name?official_full_name=Adam%20Kinzinger",
    "/v1/legislator/party_by_name?first_name=Christopher&middle_name=Henderson&last_name=Clark",
    "/v1/legislator/names_with_facebook_no_instagram?limit=10",
    "/v1/legislator/names_by_state?state=VA",
    "/v1/legislator/names_by_party?party=National%20Greenbacker",
    "/v1/legislator/name_by_youtube?youtube=RoskamIL06",
    "/v1/legislator/names_with_only_facebook",
    "/v1/legislator/names_by_party_and_state_rank?party=Republican&state_rank=junior",
    "/v1/legislator/contact_form_by_name?official_full_name=Claire%20McCaskill",
    "/v1/legislator/wikipedia_id_by_party?party=Readjuster%20Democrat",
    "/v1/legislator/current_legislators_with_nickname?party=Republican",
    "/v1/legislator/historical_legislators_state_party?first_name=Veronica&middle_name=Grace&last_name=Boland",
    "/v1/legislator/historical_legislators_count_by_birthdate?birthdate=1973-01-01",
    "/v1/legislator/historical_legislators_gender_ratio",
    "/v1/legislator/current_terms_end_date_party?year=2009&party=Republican",
    "/v1/legislator/current_legislators_name_gender?last_name=Collins",
    "/v1/legislator/historical_terms_class_percentage?type=sen",
    "/v1/legislator/current_legislators_by_party?party=Independent",
    "/v1/legislator/sum_term_durations?official_full_name=John%20Conyers,%20Jr.",
    "/v1/legislator/age_at_first_term?official_full_name=F.%20James%20Sensenbrenner,%20Jr.",
    "/v1/legislator/current_legislators_by_state?state=ME",
    "/v1/legislator/current_legislators_with_instagram?thomas_id=1000",
    "/v1/legislator/historical_terms_end_date?official_full_name=Matt%20Salmon",
    "/v1/legislator/historical_legislators_by_birthday?birthday_bio=%251738%25",
    "/v1/legislator/historical_legislators_by_party?party=Liberal%20Republican",
    "/v1/legislator/count_historical_legislators_by_gender_start?gender_bio=M&start_date=1793-12-02",
    "/v1/legislator/term_count_difference_by_year?start_year=2005%25",
    "/v1/legislator/current_legislators_by_first_name?first_name=Richard",
    "/v1/legislator/historical_terms_by_full_name?first_name=Pearl&middle_name=Peden&last_name=Oldfield",
    "/v1/legislator/current_legislator_birthday_bio?first_name=Amy&last_name=Klobuchar",
    "/v1/legislator/count_null_or_empty_fec_id?fec_id=",
    "/v1/legislator/count_non_null_non_empty_opensecrets_id?opensecrets_id=",
    "/v1/legislator/middle_name_by_birthday?birthday_bio=1956-08-24",
    "/v1/legislator/count_bioguide_by_title?title=Majority%20Leader",
    "/v1/legislator/titles_by_birthday?birthday_bio=1942-02-20",
    "/v1/legislator/gender_by_address?address=317%20Russell%20Senate%20Office%20Building%20Washington%20DC%2020510",
    "/v1/legislator/first_names_by_state_rank?state_rank=senior",
    "/v1/legislator/count_states_by_gender_and_class?gender_bio=M&class_value=",
    "/v1/legislator/percentage_non_null_class_by_birthday?birthday_bio=%251964%25",
    "/v1/legislator/percentage_no_class_by_birth_year?birth_year=1975",
    "/v1/legislator/twitter_by_birth_date?birth_date=1946-05-27",
    "/v1/legislator/opensecrets_id_by_youtube?youtube_handle=BLuetkemeyer",
    "/v1/legislator/first_names_by_address?address=1005%20Longworth%20HOB%20Washington%20DC%2020515-1408",
    "/v1/legislator/instagram_by_birth_date?birth_date=1952-08-24",
    "/v1/legislator/count_by_gender_and_class?gender=F&class_value=",
    "/v1/legislator/religion_by_youtube?youtube_handle=MaxineWaters",
    "/v1/legislator/count_by_title_and_fec_id?title=Minority%20Leader&fec_id=",
    "/v1/legislator/facebook_id_by_facebook?facebook_handle=RepWilson",
    "/v1/legislator/count_by_first_name?first_name=John",
    "/v1/legislator/districts_by_party?party=Anti-Administration",
    "/v1/legislator/official_full_names_by_birth_year?birth_year=1960%25",
    "/v1/legislator/google_entity_ids_by_name?first_name=Benjamin&last_name=Hawkins",
    "/v1/legislator/names_by_party_and_term_dates?party=Pro-Administration&start=1789-03-04&end=1791-12-31",
    "/v1/legislator/names_by_district?district=9",
    "/v1/legislator/ids_by_type_and_state?type=sen&state=NJ",
    "/v1/legislator/google_entity_ids_by_type_and_state?type=sen&state=NY",
    "/v1/legislator/religion_bio_by_rss_url?rss_url=http%3A%2F%2Fwww.corker.senate.gov%2Fpublic%2Findex.cfm%2Frss%2Ffeed",
    "/v1/legislator/party_by_official_full_name?official_full_name=Susan%20M.%20Collins",
    "/v1/legislator/historical_district?last_name=Grout&first_name=Jonathan&term_type=rep",
    "/v1/legislator/current_party_state?opensecrets_id=N00003689&thomas_id=186",
    "/v1/legislator/current_legislator_info?contact_form=http://www.brown.senate.gov/contact/",
    "/v1/legislator/historical_state_type?google_entity_id=kg:/m/02pyzk",
    "/v1/legislator/historical_type_end?first_name=John&last_name=Vining",
    "/v1/legislator/historical_sen_rep_diff?gender=F&start_year=1930&end_year=1970",
    "/v1/legislator/current_independent_percentage?gender=M&start_year=1955&end_year=1965",
    "/v1/legislator/historical_legislator_name?bioguide_id=W000059",
    "/v1/legislator/historical_ballotpedia_check?first_name=Thomas&last_name=Carnes",
    "/v1/legislator/historical_legislator_count_by_birth_year?birth_year=1736",
    "/v1/legislator/historical_figures_by_gender?gender_bio=F",
    "/v1/legislator/count_districts_by_state?state=ID",
    "/v1/legislator/count_current_terms_null_class",
    "/v1/legislator/percentage_historical_with_wikipedia",
    "/v1/legislator/current_legislators_without_instagram",
    "/v1/legislator/historical_terms_by_start_year?start_year=1789%25",
    "/v1/legislator/historical_terms_by_name?first_name=Benjamin&last_name=Contee",
    "/v1/legislator/current_terms_address_by_name_start_date?first_name=Amy&last_name=Klobuchar&start_date=2001-04-01",
    "/v1/legislator/current_legislators_by_start_year_state_rank?start_year=1997%25&state_rank=junior",
    "/v1/legislator/count_current_legislators_by_start_year_state_gender?start_year=2015&state=CA&gender_bio=F",
    "/v1/legislator/twitter_id_by_name?first_name=Emanuel&last_name=Cleaver",
    "/v1/legislator/facebook_ids_by_party?party=Democrat",
    "/v1/legislator/historical_names_by_end_date_gender?end_date=1791-03-03&gender=F",
    "/v1/legislator/current_names_by_religion_state?religion=Jewish&state=FL",
    "/v1/legislator/percentage_current_with_wikipedia",
    "/v1/legislator/percentage_current_by_start_year_range?start_year=2000&end_year=2017",
    "/v1/legislator/count_historical_by_religion_no_ballotpedia?religion=Catholic",
    "/v1/legislator/count_current_by_class_party?class=1&party=Republican",
    "/v1/legislator/historical_names_by_gender_missing_ids?gender=F",
    "/v1/legislator/count_current_by_state_type_end_pattern?state=CA&type=rep&end_pattern=1995%",
    "/v1/legislator/historical_figure_names?limit=1",
    "/v1/legislator/current_representatives_bioguide?type=rep&party=Democrat&end=2019-01-03&district=13",
    "/v1/legislator/current_legislator_twitter?official_full_name=Jason%20Lewis",
    "/v1/legislator/current_legislators_by_criteria?party=Independent&start_year=2011&type=sen&caucus=Democrat",
    "/v1/legislator/count_legislators_without_facebook",
    "/v1/legislator/most_common_party_by_religion?religion_bio=Baptist&limit=1",
    "/v1/legislator/legislators_with_all_social_media",
    "/v1/legislator/count_districts_by_legislator?official_full_name=John%20Conyers%2C%20Jr.",
    "/v1/legislator/historical_legislators_wikipedia_ids?type=sen&party=Anti-Administration",
    "/v1/legislator/legislators_by_term_duration_and_count?term_duration=26&term_count=13",
    "/v1/legislator/count_historical_legislators?first_name=Benjamin&party=Federalist&type=rep",
    "/v1/legislator/sum_results_gender_state_type_duration?duration=10&gender=F&state=CA&type=rep",
    "/v1/legislator/oldest_historical_legislator_party",
    "/v1/legislator/official_full_name_religion_state_type_duration?duration=14&religion=Lutheran&state=OH&type=rep",
    "/v1/legislator/count_current_legislators_birth_year_no_google_id?birth_year=1960",
    "/v1/legislator/official_full_names_with_house_history_id",
    "/v1/legislator/count_current_legislators_icpsr_maplight_ids",
    "/v1/legislator/count_current_legislators_lis_id_gender?gender=F",
    "/v1/legislator/religion_current_legislator_official_full_name?official_full_name=Sherrod%20Brown",
    "/v1/legislator/most_common_religion_current_legislators",
    "/v1/legislator/instagram_by_name?official_full_name=Bob%20Corker",
    "/v1/legislator/count_with_thomas_and_instagram",
    "/v1/legislator/facebook_by_wikipedia_id",
    "/v1/legislator/count_earliest_birthday",
    "/v1/legislator/phone_by_name_and_start_date?official_full_name=Sherrod%20Brown&start_date=2013-01-03",
    "/v1/legislator/count_by_name_and_party?official_full_name=Sherrod%20Brown&party=Democrat",
    "/v1/legislator/names_by_start_date?start_date=2013-01-03",
    "/v1/legislator/count_by_start_date_and_gender?start_date=2013-01-03&gender=F",
    "/v1/legislator/most_terms",
    "/v1/legislator/count_by_name_and_district?official_full_name=Sherrod%20Brown&district=13",
    "/v1/legislator/official_full_name_by_facebook?facebook=senjoniernst",
    "/v1/legislator/compare_terms_served?name1=Maria%20Cantwell&name2=Sherrod%20Brown",
    "/v1/legislator/percentage_gender_terms?gender=F&min_terms=4",
    "/v1/legislator/ratio_with_wikipedia",
    "/v1/legislator/count_by_gender?gender=F",
    "/v1/legislator/count_historical_terms?state=NJ&term_type=rep",
    "/v1/legislator/address_by_contact_form?contact_form=http://www.carper.senate.gov/contact/",
    "/v1/legislator/bioguide_by_religion_state?religion=Catholic&state=NE",
    "/v1/legislator/maplight_ids_by_term_state?term_type=rep&state=ME",
    "/v1/legislator/historical_terms_by_birthday?birthday_bio=1727-11-26",
    "/v1/legislator/count_legislators_by_type_state_gender?type=rep&state=MI&gender_bio=F",
    "/v1/legislator/percentage_legislators_by_type_state_gender?type=sen&state=ME&gender_bio=F"
]
