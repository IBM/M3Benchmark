from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/music_tracker/music_tracker.sqlite')
cursor = conn.cursor()

# Endpoint to get group names based on artist, year, release type, and total snatched
@app.get("/v1/music_tracker/group_names_by_criteria", operation_id="get_group_names_by_criteria", summary="Retrieves the names of music groups that match the specified artist, year, release type, and total snatched count. This operation allows users to filter and find music groups based on these criteria, providing a targeted list of group names.")
async def get_group_names_by_criteria(artist: str = Query(..., description="Artist name"), group_year: int = Query(..., description="Year of the group"), release_type: str = Query(..., description="Release type"), total_snatched: int = Query(..., description="Total snatched")):
    cursor.execute("SELECT groupName FROM torrents WHERE artist LIKE ? AND groupYear = ? AND releaseType LIKE ? AND totalSnatched = ?", (artist, group_year, release_type, total_snatched))
    result = cursor.fetchall()
    if not result:
        return {"group_names": []}
    return {"group_names": [row[0] for row in result]}

# Endpoint to get total snatched based on artist and year
@app.get("/v1/music_tracker/total_snatched_by_artist_year", operation_id="get_total_snatched_by_artist_year", summary="Retrieves the total number of snatched torrents for a specific artist in a given year. The artist and year are provided as input parameters, allowing for a targeted search of the torrents database.")
async def get_total_snatched_by_artist_year(artist: str = Query(..., description="Artist name"), group_year: int = Query(..., description="Year of the group")):
    cursor.execute("SELECT totalSnatched FROM torrents WHERE artist LIKE ? AND groupYear = ?", (artist, group_year))
    result = cursor.fetchall()
    if not result:
        return {"total_snatched": []}
    return {"total_snatched": [row[0] for row in result]}

# Endpoint to get the top tag based on release type
@app.get("/v1/music_tracker/top_tag_by_release_type", operation_id="get_top_tag_by_release_type", summary="Retrieves the most popular tag associated with the specified release type. The release type is used to filter the torrents, and the tag with the highest total snatches is returned. This operation provides insights into the most sought-after content within a particular release type.")
async def get_top_tag_by_release_type(release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT T2.tag FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.releaseType = ? ORDER BY T1.totalSnatched DESC LIMIT 1", (release_type,))
    result = cursor.fetchone()
    if not result:
        return {"top_tag": []}
    return {"top_tag": result[0]}

# Endpoint to get the top 5 tags based on release type
@app.get("/v1/music_tracker/top_5_tags_by_release_type", operation_id="get_top_5_tags_by_release_type", summary="Retrieves the top 5 most popular tags associated with a specific release type. The tags are ranked based on the total number of times they have been downloaded (snatched). The release type is specified as an input parameter.")
async def get_top_5_tags_by_release_type(release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT T2.tag FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.releaseType = ? ORDER BY T1.totalSnatched DESC LIMIT 5", (release_type,))
    result = cursor.fetchall()
    if not result:
        return {"top_tags": []}
    return {"top_tags": [row[0] for row in result]}

# Endpoint to get the oldest group name based on tag and release type
@app.get("/v1/music_tracker/oldest_group_name_by_tag_release_type", operation_id="get_oldest_group_name_by_tag_release_type", summary="Retrieves the name of the music group that has the oldest release, based on a specified tag and release type. The tag and release type are used to filter the results, ensuring that only relevant music groups are considered. The result is determined by the earliest release year of the group.")
async def get_oldest_group_name_by_tag_release_type(tag: str = Query(..., description="Tag"), release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT T1.groupName FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag LIKE ? AND T1.releaseType = ? ORDER BY T1.groupYear LIMIT 1", (tag, release_type))
    result = cursor.fetchone()
    if not result:
        return {"group_name": []}
    return {"group_name": result[0]}

# Endpoint to get group names based on tag and release type
@app.get("/v1/music_tracker/group_names_by_tag_release_type", operation_id="get_group_names_by_tag_release_type", summary="Retrieves a list of group names that match a specified tag and release type. The tag parameter filters the results to include only those groups associated with the given tag. The release type parameter further narrows the results to include only those groups that match the specified release type.")
async def get_group_names_by_tag_release_type(tag: str = Query(..., description="Tag"), release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT T1.groupName FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag LIKE ? AND T1.releaseType = ?", (tag, release_type))
    result = cursor.fetchall()
    if not result:
        return {"group_names": []}
    return {"group_names": [row[0] for row in result]}

# Endpoint to get the top 5 tags based on release type and total snatched
@app.get("/v1/music_tracker/top_5_tags_by_release_type_total_snatched", operation_id="get_top_5_tags_by_release_type_total_snatched", summary="Retrieves the top 5 tags associated with the specified release type, ranked by the total number of times they have been snatched.")
async def get_top_5_tags_by_release_type_total_snatched(release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT T2.tag FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.releaseType = ? ORDER BY T1.totalSnatched LIMIT 5", (release_type,))
    result = cursor.fetchall()
    if not result:
        return {"top_tags": []}
    return {"top_tags": [row[0] for row in result]}

# Endpoint to get the top tag and artist based on release type
@app.get("/v1/music_tracker/top_tag_artist_by_release_type", operation_id="get_top_tag_artist_by_release_type", summary="Retrieves the most popular tag and its associated artist based on the specified release type. The release type is used to filter the results, and the top tag and artist are determined by the total number of snatches. The data is ordered in descending order based on the total number of snatches, and only the top result is returned.")
async def get_top_tag_artist_by_release_type(release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT T2.tag, T1.artist FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.releaseType = ? ORDER BY T1.totalSnatched DESC LIMIT 1", (release_type,))
    result = cursor.fetchone()
    if not result:
        return {"top_tag_artist": []}
    return {"top_tag_artist": {"tag": result[0], "artist": result[1]}}

# Endpoint to get the count of tags based on a tag pattern
@app.get("/v1/music_tracker/count_tags_by_pattern", operation_id="get_count_tags_by_pattern", summary="Retrieves the total count of tags that match the provided pattern. The pattern is used to filter the tags and determine the count.")
async def get_count_tags_by_pattern(tag_pattern: str = Query(..., description="Tag pattern")):
    cursor.execute("SELECT COUNT(id) FROM tags WHERE tag LIKE ?", (tag_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get total snatched based on group name
@app.get("/v1/music_tracker/total_snatched_by_group_name", operation_id="get_total_snatched_by_group_name", summary="Retrieves the total number of snatched torrents associated with a specific group. The group is identified by its name, which is provided as an input parameter.")
async def get_total_snatched_by_group_name(group_name: str = Query(..., description="Group name")):
    cursor.execute("SELECT totalSnatched FROM torrents WHERE groupName LIKE ?", (group_name,))
    result = cursor.fetchall()
    if not result:
        return {"total_snatched": []}
    return {"total_snatched": [row[0] for row in result]}

# Endpoint to get group names based on total snatched count
@app.get("/v1/music_tracker/group_names_by_total_snatched", operation_id="get_group_names_by_total_snatched", summary="Retrieve the names of groups that have a total snatched count exceeding the provided value. This operation allows you to filter groups based on their total snatched count, providing a targeted list of group names that meet the specified criteria.")
async def get_group_names_by_total_snatched(total_snatched: int = Query(..., description="Total snatched count")):
    cursor.execute("SELECT groupName FROM torrents WHERE totalSnatched > ?", (total_snatched,))
    result = cursor.fetchall()
    if not result:
        return {"group_names": []}
    return {"group_names": [row[0] for row in result]}

# Endpoint to get tags based on group name
@app.get("/v1/music_tracker/tags_by_group_name", operation_id="get_tags_by_group_name", summary="Retrieves all tags linked to a specified group name. The group name is used to filter the tags from the database.")
async def get_tags_by_group_name(group_name: str = Query(..., description="Group name")):
    cursor.execute("SELECT T2.tag FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.groupName = ?", (group_name,))
    result = cursor.fetchall()
    if not result:
        return {"tags": []}
    return {"tags": [row[0] for row in result]}

# Endpoint to get the count of tags based on group name
@app.get("/v1/music_tracker/count_tags_by_group_name", operation_id="get_count_tags_by_group_name", summary="Retrieves the total number of tags linked to a particular group name. The group name is provided as an input parameter, which is used to filter the data and calculate the tag count.")
async def get_count_tags_by_group_name(group_name: str = Query(..., description="Group name")):
    cursor.execute("SELECT COUNT(T2.tag) FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.groupName = ?", (group_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get group names based on tag
@app.get("/v1/music_tracker/group_names_by_tag", operation_id="get_group_names_by_tag", summary="Retrieves the names of all groups that have been assigned a specific tag. The tag is provided as an input parameter, and the operation returns a list of group names that match the given tag.")
async def get_group_names_by_tag(tag: str = Query(..., description="Tag")):
    cursor.execute("SELECT T1.groupName FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ?", (tag,))
    result = cursor.fetchall()
    if not result:
        return {"group_names": []}
    return {"group_names": [row[0] for row in result]}

# Endpoint to get the top group name based on tag and total snatched count
@app.get("/v1/music_tracker/top_group_name_by_tag", operation_id="get_top_group_name_by_tag", summary="Retrieves the name of the group with the highest total snatched count for a given tag. The tag is used to filter the results, and the group name is ordered by the total snatched count in descending order.")
async def get_top_group_name_by_tag(tag: str = Query(..., description="Tag")):
    cursor.execute("SELECT T1.groupName FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? ORDER BY T1.totalSnatched DESC LIMIT 1", (tag,))
    result = cursor.fetchone()
    if not result:
        return {"group_name": []}
    return {"group_name": result[0]}

# Endpoint to get the count of group names based on tag and artist
@app.get("/v1/music_tracker/count_group_names_by_tag_and_artist", operation_id="get_count_group_names_by_tag_and_artist", summary="Retrieves the total number of unique group names that are associated with a given tag and artist. This operation is useful for understanding the distribution of group names based on a specific tag and artist.")
async def get_count_group_names_by_tag_and_artist(tag: str = Query(..., description="Tag"), artist: str = Query(..., description="Artist")):
    cursor.execute("SELECT COUNT(T1.groupName) FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.artist = ?", (tag, artist))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of group names based on tag, release type, and group year
@app.get("/v1/music_tracker/count_group_names_by_tag_release_type_year", operation_id="get_count_group_names_by_tag_release_type_year", summary="Retrieves the total number of unique group names that match a specified tag, release type, and year. This operation is useful for understanding the distribution of group names based on the provided criteria.")
async def get_count_group_names_by_tag_release_type_year(tag: str = Query(..., description="Tag"), release_type: str = Query(..., description="Release type"), group_year: int = Query(..., description="Group year")):
    cursor.execute("SELECT COUNT(T1.groupName) FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.releaseType = ? AND T1.groupYear = ?", (tag, release_type, group_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average total snatched count based on tag
@app.get("/v1/music_tracker/average_total_snatched_by_tag", operation_id="get_average_total_snatched_by_tag", summary="Retrieves the average total count of snatched torrents associated with a specific tag. The tag is provided as an input parameter, and the operation calculates the average by summing the total snatched count for all torrents with the given tag and dividing it by the total number of torrents with that tag.")
async def get_average_total_snatched_by_tag(tag: str = Query(..., description="Tag")):
    cursor.execute("SELECT CAST(SUM(T1.totalSnatched) AS REAL) / COUNT(T2.tag) FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ?", (tag,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get top group names based on total snatched count
@app.get("/v1/music_tracker/top_group_names_by_total_snatched", operation_id="get_top_group_names_by_total_snatched", summary="Retrieve the top group names, ranked by their total snatched count in descending order. The number of results can be limited by specifying the desired maximum count.")
async def get_top_group_names_by_total_snatched(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT groupName FROM torrents ORDER BY totalSnatched DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"group_names": []}
    return {"group_names": [row[0] for row in result]}

# Endpoint to get the top artist and group name based on group year and release type
@app.get("/v1/music_tracker/top_artist_group_by_year_release_type", operation_id="get_top_artist_group_by_year_release_type", summary="Retrieves the top artist and group name based on the specified group year and release type, sorted by the total number of downloads in descending order.")
async def get_top_artist_group_by_year_release_type(group_year: int = Query(..., description="Group year"), release_type: str = Query(..., description="Release type (e.g., 'Single')")):
    cursor.execute("SELECT artist, groupName FROM torrents WHERE groupYear = ? AND releaseType LIKE ? ORDER BY totalSnatched DESC LIMIT 1", (group_year, release_type))
    result = cursor.fetchone()
    if not result:
        return {"artist": [], "group_name": []}
    return {"artist": result[0], "group_name": result[1]}

# Endpoint to get the count of torrents based on group year range, artist, and release type
@app.get("/v1/music_tracker/torrent_count_by_year_artist_release", operation_id="get_torrent_count", summary="Get the count of torrents within a specified group year range, artist, and release type")
async def get_torrent_count(start_year: int = Query(..., description="Start year of the group year range"), end_year: int = Query(..., description="End year of the group year range"), artist: str = Query(..., description="Artist name"), release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT COUNT(id), ( SELECT COUNT(id) FROM torrents WHERE groupYear BETWEEN ? AND ? AND artist LIKE ? AND releaseType LIKE ? ) FROM torrents WHERE groupYear BETWEEN ? AND ? AND artist LIKE ? AND releaseType LIKE ?", (start_year, end_year, artist, release_type, start_year, end_year, artist, release_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in group years for a specific artist and release type
@app.get("/v1/music_tracker/group_year_difference", operation_id="get_group_year_difference", summary="Retrieves the difference in years between the earliest and the specified group year for a particular artist and release type. The operation compares the specified group year with the second earliest group year for the given artist and release type, providing insights into the time gap between releases.")
async def get_group_year_difference(artist: str = Query(..., description="Artist name"), release_type: str = Query(..., description="Release type"), group_year: int = Query(..., description="Group year")):
    cursor.execute("SELECT ( SELECT groupYear FROM torrents WHERE artist LIKE ? AND releaseType LIKE ? ORDER BY groupYear LIMIT 1, 1 ) - groupYear FROM torrents WHERE artist LIKE ? AND releaseType LIKE ? AND groupYear = ?", (artist, release_type, artist, release_type, group_year))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the average total snatched for a specific artist and release type within a group year range
@app.get("/v1/music_tracker/average_total_snatched", operation_id="get_average_total_snatched", summary="Retrieves the average total number of snatched torrents for a specific artist and release type within a specified group year range. The calculation is based on the provided artist name, release type, and the inclusive start and end years of the group year range.")
async def get_average_total_snatched(artist: str = Query(..., description="Artist name"), release_type: str = Query(..., description="Release type"), start_year: int = Query(..., description="Start year of the group year range"), end_year: int = Query(..., description="End year of the group year range")):
    cursor.execute("SELECT AVG(totalSnatched) FROM torrents WHERE artist LIKE ? AND releaseType LIKE ? AND groupYear BETWEEN ? AND ?", (artist, release_type, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the top torrent by total snatched for a specific release type
@app.get("/v1/music_tracker/top_torrent_by_release_type", operation_id="get_top_torrent", summary="Retrieve the most popular torrent for a given release type, ranked by the total number of times it has been downloaded. The response includes the name and year of the group associated with the torrent, as well as the tag associated with the torrent.")
async def get_top_torrent(release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT T1.groupName, T1.groupYear, T2.tag FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.releaseType = ? ORDER BY T1.totalSnatched DESC LIMIT 1", (release_type,))
    result = cursor.fetchone()
    if not result:
        return {"top_torrent": []}
    return {"top_torrent": {"groupName": result[0], "groupYear": result[1], "tag": result[2]}}

# Endpoint to get artists with more than a specified number of releases in a given year and release type
@app.get("/v1/music_tracker/artists_by_year_release_count", operation_id="get_artists_by_year_release_count", summary="Retrieves artists who have released more than a specified number of tracks in a given year and release type. The response includes a list of artists who meet the specified criteria, providing a useful way to identify prolific artists in a particular year and release category.")
async def get_artists_by_year_release_count(group_year: int = Query(..., description="Group year"), release_type: str = Query(..., description="Release type"), count: int = Query(..., description="Minimum count of releases")):
    cursor.execute("SELECT artist FROM torrents WHERE groupYear = ? AND releaseType LIKE ? GROUP BY artist HAVING COUNT(releaseType) > ?", (group_year, release_type, count))
    result = cursor.fetchall()
    if not result:
        return {"artists": []}
    return {"artists": [row[0] for row in result]}

# Endpoint to get artists within a specified group year range and release type
@app.get("/v1/music_tracker/artists_by_year_range_release_type", operation_id="get_artists_by_year_range_release_type", summary="Retrieves a list of artists who have released music within a specified year range and release type. The start and end years define the range, while the release type further filters the results. This operation is useful for analyzing artists' activity within a certain time frame and release category.")
async def get_artists_by_year_range_release_type(start_year: int = Query(..., description="Start year of the group year range"), end_year: int = Query(..., description="End year of the group year range"), release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT artist FROM torrents WHERE groupYear BETWEEN ? AND ? AND releaseType LIKE ?", (start_year, end_year, release_type))
    result = cursor.fetchall()
    if not result:
        return {"artists": []}
    return {"artists": [row[0] for row in result]}

# Endpoint to get group names based on total snatched, release type, and ID range
@app.get("/v1/music_tracker/group_names_by_snatched_release_id", operation_id="get_group_names_by_snatched_release_id", summary="Retrieve the names of groups that have a minimum total number of snatched torrents, a specific release type, and fall within a specified ID range. The operation filters torrents based on the provided parameters and returns the group names that meet the criteria.")
async def get_group_names_by_snatched_release_id(total_snatched: int = Query(..., description="Minimum total snatched"), release_type: str = Query(..., description="Release type"), start_id: int = Query(..., description="Start ID of the range"), end_id: int = Query(..., description="End ID of the range")):
    cursor.execute("SELECT groupName FROM torrents WHERE totalSnatched >= ? AND releaseType LIKE ? AND id BETWEEN ? AND ?", (total_snatched, release_type, start_id, end_id))
    result = cursor.fetchall()
    if not result:
        return {"group_names": []}
    return {"group_names": [row[0] for row in result]}

# Endpoint to get artists based on tag and group year range
@app.get("/v1/music_tracker/artists_by_tag_year_range", operation_id="get_artists_by_tag_year_range", summary="Retrieves a list of artists who have been tagged with a specific tag and whose group year falls within a specified range. The tag and the start and end years of the group year range are provided as input parameters.")
async def get_artists_by_tag_year_range(tag: str = Query(..., description="Tag"), start_year: int = Query(..., description="Start year of the group year range"), end_year: int = Query(..., description="End year of the group year range")):
    cursor.execute("SELECT T1.artist FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.groupYear BETWEEN ? AND ?", (tag, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"artists": []}
    return {"artists": [row[0] for row in result]}

# Endpoint to get artists based on tag, group year, and maximum total snatched
@app.get("/v1/music_tracker/artists_by_tag_year_snatched", operation_id="get_artists_by_tag_year_snatched", summary="Retrieves a list of artists who have been tagged with a specific tag, belong to a particular group year, and have a total number of snatches that does not exceed a given maximum. The operation filters the artists based on the provided tag, group year, and maximum total snatched, and returns the matching artist names.")
async def get_artists_by_tag_year_snatched(tag: str = Query(..., description="Tag"), group_year: int = Query(..., description="Group year"), max_snatched: int = Query(..., description="Maximum total snatched")):
    cursor.execute("SELECT T1.artist FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.groupYear = ? AND T1.totalSnatched <= ?", (tag, group_year, max_snatched))
    result = cursor.fetchall()
    if not result:
        return {"artists": []}
    return {"artists": [row[0] for row in result]}

# Endpoint to get the top artist based on tag and release type
@app.get("/v1/music_tracker/top_artist_by_tag_release", operation_id="get_top_artist_by_tag_release", summary="Retrieves the artist with the highest number of releases for a specified tag. The tag and release type are used to filter the results. The artist with the most releases matching the provided tag and release type is returned.")
async def get_top_artist_by_tag_release(tag: str = Query(..., description="Tag"), release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT T1.artist FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.releaseType = ? GROUP BY T1.artist ORDER BY COUNT(T1.releaseType) DESC LIMIT 1", (tag, release_type))
    result = cursor.fetchone()
    if not result:
        return {"top_artist": []}
    return {"top_artist": result[0]}

# Endpoint to get artists based on tag, group year, and ID range
@app.get("/v1/music_tracker/artists_by_tag_year_id_range", operation_id="get_artists_by_tag_year_id_range", summary="Retrieve a list of artists associated with a specific tag, group year, and within a defined ID range. The operation filters torrents based on the provided tag, group year, and ID range, and returns the corresponding artist names.")
async def get_artists_by_tag_year_id_range(tag: str = Query(..., description="Tag of the torrent"), group_year: int = Query(..., description="Group year of the torrent"), id_min: int = Query(..., description="Minimum ID of the torrent"), id_max: int = Query(..., description="Maximum ID of the torrent")):
    cursor.execute("SELECT T1.artist FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.groupYear = ? AND T1.id BETWEEN ? AND ?", (tag, group_year, id_min, id_max))
    result = cursor.fetchall()
    if not result:
        return {"artists": []}
    return {"artists": [row[0] for row in result]}

# Endpoint to get the group name with the highest total snatched for a specific tag and minimum group year
@app.get("/v1/music_tracker/group_name_by_tag_min_year", operation_id="get_group_name_by_tag_min_year", summary="Retrieves the name of the group with the highest total snatched count for a given tag and a minimum group year. The tag and minimum group year are provided as input parameters to filter the results.")
async def get_group_name_by_tag_min_year(tag: str = Query(..., description="Tag of the torrent"), min_group_year: int = Query(..., description="Minimum group year of the torrent")):
    cursor.execute("SELECT T1.groupName FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.groupYear >= ? ORDER BY T1.totalSnatched DESC LIMIT 1", (tag, min_group_year))
    result = cursor.fetchone()
    if not result:
        return {"group_name": []}
    return {"group_name": result[0]}

# Endpoint to get tags for a specific torrent ID
@app.get("/v1/music_tracker/tags_by_torrent_id", operation_id="get_tags_by_torrent_id", summary="Retrieves all tags associated with a specific torrent, identified by its unique torrent_id. This operation allows users to view the tags linked to a particular torrent, providing insights into its categorization or metadata.")
async def get_tags_by_torrent_id(torrent_id: int = Query(..., description="ID of the torrent")):
    cursor.execute("SELECT T2.tag FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.id = ?", (torrent_id,))
    result = cursor.fetchall()
    if not result:
        return {"tags": []}
    return {"tags": [row[0] for row in result]}

# Endpoint to get the artist with the highest total snatched for a specific tag and ID range
@app.get("/v1/music_tracker/artist_by_tag_id_range", operation_id="get_artist_by_tag_id_range", summary="Retrieves the artist with the highest total snatched count for a specific tag within a given ID range of torrents. The ID range is defined by the minimum and maximum IDs of the torrents, while the tag specifies the category of the torrents.")
async def get_artist_by_tag_id_range(id_min: int = Query(..., description="Minimum ID of the torrent"), id_max: int = Query(..., description="Maximum ID of the torrent"), tag: str = Query(..., description="Tag of the torrent")):
    cursor.execute("SELECT T1.artist FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.id BETWEEN ? AND ? AND T2.tag LIKE ? ORDER BY T1.totalSnatched DESC LIMIT 1", (id_min, id_max, tag))
    result = cursor.fetchone()
    if not result:
        return {"artist": []}
    return {"artist": result[0]}

# Endpoint to get the count of artists for a specific tag, group year range, and release types
@app.get("/v1/music_tracker/artist_count_by_tag_year_range_release_types", operation_id="get_artist_count_by_tag_year_range_release_types", summary="Get the count of artists for a specific tag, group year range, and release types")
async def get_artist_count_by_tag_year_range_release_types(tag: str = Query(..., description="Tag of the torrent"), min_group_year: int = Query(..., description="Minimum group year of the torrent"), max_group_year: int = Query(..., description="Maximum group year of the torrent"), release_type1: str = Query(..., description="First release type"), release_type2: str = Query(..., description="Second release type")):
    cursor.execute("SELECT COUNT(T1.artist) FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.groupYear BETWEEN ? AND ? AND (T1.releaseType LIKE ? OR T1.releaseType LIKE ?)", (tag, min_group_year, max_group_year, release_type1, release_type2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of tags for a specific tag, group year range, and release type
@app.get("/v1/music_tracker/tag_count_by_tag_year_range_release_type", operation_id="get_tag_count_by_tag_year_range_release_type", summary="Retrieves the total count of a specific tag, filtered by a range of group years and a release type. The tag, minimum group year, maximum group year, and release type are provided as input parameters.")
async def get_tag_count_by_tag_year_range_release_type(tag: str = Query(..., description="Tag of the torrent"), min_group_year: int = Query(..., description="Minimum group year of the torrent"), max_group_year: int = Query(..., description="Maximum group year of the torrent"), release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT COUNT(T2.tag) FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.groupYear BETWEEN ? AND ? AND T1.releaseType LIKE ?", (tag, min_group_year, max_group_year, release_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of release types for a specific release type and group year
@app.get("/v1/music_tracker/release_type_count_by_type_year", operation_id="get_release_type_count_by_type_year", summary="Retrieves the total count of a specified release type for a given year. This operation allows you to understand the frequency of a particular release type within a specific year. The release type and year are provided as input parameters.")
async def get_release_type_count_by_type_year(release_type: str = Query(..., description="Release type"), group_year: int = Query(..., description="Group year of the torrent")):
    cursor.execute("SELECT COUNT(releaseType) FROM torrents WHERE releaseType LIKE ? AND groupYear = ?", (release_type, group_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of release types for a specific artist, release type, and group year
@app.get("/v1/music_tracker/release_type_count_by_artist_type_year", operation_id="get_release_type_count_by_artist_type_year", summary="Retrieves the total number of a specific release type associated with a given artist and group year. This operation allows you to analyze the distribution of release types for a particular artist and year, providing insights into their discography and release patterns.")
async def get_release_type_count_by_artist_type_year(artist: str = Query(..., description="Artist of the torrent"), release_type: str = Query(..., description="Release type"), group_year: int = Query(..., description="Group year of the torrent")):
    cursor.execute("SELECT COUNT(releaseType) FROM torrents WHERE artist LIKE ? AND releaseType LIKE ? AND groupYear = ?", (artist, release_type, group_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of torrent IDs for a specific tag, group year, and release type
@app.get("/v1/music_tracker/torrent_id_count_by_tag_year_release_type", operation_id="get_torrent_id_count_by_tag_year_release_type", summary="Retrieve the total number of torrents associated with a specific tag, group year, and release type. This operation provides a count of torrent IDs that match the given tag, group year, and release type criteria. The input parameters determine the tag, group year, and release type to filter the torrents.")
async def get_torrent_id_count_by_tag_year_release_type(tag: str = Query(..., description="Tag of the torrent"), group_year: int = Query(..., description="Group year of the torrent"), release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT COUNT(T1.id) FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T2.tag = ? AND T1.groupYear = ? AND T1.releaseType LIKE ?", (tag, group_year, release_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get tags for a specific group year range and artist
@app.get("/v1/music_tracker/tags_by_year_range_artist", operation_id="get_tags_by_year_range_artist", summary="Retrieves the tags associated with torrents that fall within a specified year range and are by a particular artist. The minimum and maximum group year of the torrents and the artist's name are required as input parameters.")
async def get_tags_by_year_range_artist(min_group_year: int = Query(..., description="Minimum group year of the torrent"), max_group_year: int = Query(..., description="Maximum group year of the torrent"), artist: str = Query(..., description="Artist of the torrent")):
    cursor.execute("SELECT T2.tag FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.groupYear BETWEEN ? AND ? AND T1.artist LIKE ?", (min_group_year, max_group_year, artist))
    result = cursor.fetchall()
    if not result:
        return {"tags": []}
    return {"tags": [row[0] for row in result]}

# Endpoint to get group names and tags for torrents based on year, artist, and release type
@app.get("/v1/music_tracker/torrents_group_names_tags", operation_id="get_torrents_group_names_tags", summary="Retrieves the names and associated tags of music torrent groups based on the specified year, artist, and release type. The operation filters the torrent groups by the provided year and artist name, and further narrows down the results based on the release type. The output includes the group name and its corresponding tag.")
async def get_torrents_group_names_tags(group_year: int = Query(..., description="Year of the group"), artist: str = Query(..., description="Artist name"), release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT T1.groupName, T2.tag FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.groupYear = ? AND T1.artist LIKE ? AND T1.releaseType LIKE ?", (group_year, artist, release_type))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get group names for torrents based on year and tag
@app.get("/v1/music_tracker/torrents_group_names", operation_id="get_torrents_group_names", summary="Retrieves the names of torrent groups based on a specified year and tag. The operation filters the torrents by the provided year and tag, and returns the corresponding group names.")
async def get_torrents_group_names(group_year: int = Query(..., description="Year of the group"), tag: str = Query(..., description="Tag")):
    cursor.execute("SELECT T1.groupName FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.groupYear = ? AND T2.tag LIKE ?", (group_year, tag))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the top artist based on tag and year range
@app.get("/v1/music_tracker/top_artist_by_tag_year_range", operation_id="get_top_artist_by_tag_year_range", summary="Retrieves the artist with the highest number of occurrences of a specified tag within a given year range. The year range is defined by the start and end years, and the tag is a specific keyword associated with the artist. The results are ordered in descending order based on the count of the tag occurrences, with the top artist being returned.")
async def get_top_artist_by_tag_year_range(start_year: int = Query(..., description="Start year of the group"), end_year: int = Query(..., description="End year of the group"), tag: str = Query(..., description="Tag")):
    cursor.execute("SELECT T1.artist FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.groupYear BETWEEN ? AND ? AND T2.tag LIKE ? GROUP BY T1.artist ORDER BY COUNT(T2.tag) DESC LIMIT 1", (start_year, end_year, tag))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result[0]}

# Endpoint to get artists based on release type and tag
@app.get("/v1/music_tracker/artists_by_release_type_tag", operation_id="get_artists_by_release_type_tag", summary="Retrieves a list of artists who have released music that matches the specified release type and tag. The release type and tag are used as filters to narrow down the results.")
async def get_artists_by_release_type_tag(release_type: str = Query(..., description="Release type"), tag: str = Query(..., description="Tag")):
    cursor.execute("SELECT T1.artist FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.releaseType = ? AND T2.tag LIKE ?", (release_type, tag))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of torrents with a specific tag within a year range and release type
@app.get("/v1/music_tracker/percentage_torrents_by_tag_year_range_release_type", operation_id="get_percentage_torrents_by_tag_year_range_release_type", summary="Retrieve the proportion of torrents associated with a specific tag, filtered by a range of years and a particular release type. The calculation is based on the total count of torrents within the specified year range and release type.")
async def get_percentage_torrents_by_tag_year_range_release_type(tag: str = Query(..., description="Tag"), start_year: int = Query(..., description="Start year of the group"), end_year: int = Query(..., description="End year of the group"), release_type: str = Query(..., description="Release type")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.tag LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.releaseType) FROM torrents AS T1 INNER JOIN tags AS T2 ON T1.id = T2.id WHERE T1.groupYear BETWEEN ? AND ? AND T1.releaseType LIKE ?", (tag, start_year, end_year, release_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/music_tracker/group_names_by_criteria?artist=ron%20hunt%20%26%20ronnie%20g%20%26%20the%20sm%20crew&group_year=1979&release_type=single&total_snatched=239",
    "/v1/music_tracker/total_snatched_by_artist_year?artist=blowfly&group_year=1980",
    "/v1/music_tracker/top_tag_by_release_type?release_type=album",
    "/v1/music_tracker/top_5_tags_by_release_type?release_type=album",
    "/v1/music_tracker/oldest_group_name_by_tag_release_type?tag=funk&release_type=single",
    "/v1/music_tracker/group_names_by_tag_release_type?tag=alternative&release_type=ep",
    "/v1/music_tracker/top_5_tags_by_release_type_total_snatched?release_type=album",
    "/v1/music_tracker/top_tag_artist_by_release_type?release_type=single",
    "/v1/music_tracker/count_tags_by_pattern?tag_pattern=1980s",
    "/v1/music_tracker/total_snatched_by_group_name?group_name=city%20funk",
    "/v1/music_tracker/group_names_by_total_snatched?total_snatched=20000",
    "/v1/music_tracker/tags_by_group_name?group_name=sugarhill%20gang",
    "/v1/music_tracker/count_tags_by_group_name?group_name=city%20funk",
    "/v1/music_tracker/group_names_by_tag?tag=1980s",
    "/v1/music_tracker/top_group_name_by_tag?tag=1980s",
    "/v1/music_tracker/count_group_names_by_tag_and_artist?tag=pop&artist=michael%20jackson",
    "/v1/music_tracker/count_group_names_by_tag_release_type_year?tag=pop&release_type=album&group_year=2000",
    "/v1/music_tracker/average_total_snatched_by_tag?tag=1980s",
    "/v1/music_tracker/top_group_names_by_total_snatched?limit=3",
    "/v1/music_tracker/top_artist_group_by_year_release_type?group_year=2012&release_type=Single",
    "/v1/music_tracker/torrent_count_by_year_artist_release?start_year=2010&end_year=2015&artist=50%20cent&release_type=Single",
    "/v1/music_tracker/group_year_difference?artist=2Pac&release_type=album&group_year=1991",
    "/v1/music_tracker/average_total_snatched?artist=2Pac&release_type=Single&start_year=2001&end_year=2013",
    "/v1/music_tracker/top_torrent_by_release_type?release_type=live%20album",
    "/v1/music_tracker/artists_by_year_release_count?group_year=2016&release_type=bootleg&count=2",
    "/v1/music_tracker/artists_by_year_range_release_type?start_year=1980&end_year=1982&release_type=single",
    "/v1/music_tracker/group_names_by_snatched_release_id?total_snatched=20&release_type=single&start_id=10&end_id=20",
    "/v1/music_tracker/artists_by_tag_year_range?tag=disco&start_year=1980&end_year=1982",
    "/v1/music_tracker/artists_by_tag_year_snatched?tag=funk&group_year=1980&max_snatched=100",
    "/v1/music_tracker/top_artist_by_tag_release?tag=soul&release_type=single",
    "/v1/music_tracker/artists_by_tag_year_id_range?tag=funk&group_year=1980&id_min=10&id_max=30",
    "/v1/music_tracker/group_name_by_tag_min_year?tag=jazz&min_group_year=1982",
    "/v1/music_tracker/tags_by_torrent_id?torrent_id=16",
    "/v1/music_tracker/artist_by_tag_id_range?id_min=10&id_max=50&tag=new.york",
    "/v1/music_tracker/artist_count_by_tag_year_range_release_types?tag=dance&min_group_year=1980&max_group_year=1985&release_type1=album&release_type2=mixtape",
    "/v1/music_tracker/tag_count_by_tag_year_range_release_type?tag=soul&min_group_year=1979&max_group_year=1981&release_type=single",
    "/v1/music_tracker/release_type_count_by_type_year?release_type=single&group_year=1979",
    "/v1/music_tracker/release_type_count_by_artist_type_year?artist=sugar%20daddy&release_type=Single&group_year=1980",
    "/v1/music_tracker/torrent_id_count_by_tag_year_release_type?tag=christmas&group_year=2004&release_type=album",
    "/v1/music_tracker/tags_by_year_range_artist?min_group_year=2000&max_group_year=2010&artist=kurtis%20blow",
    "/v1/music_tracker/torrents_group_names_tags?group_year=1980&artist=millie%20jackson&release_type=album",
    "/v1/music_tracker/torrents_group_names?group_year=2005&tag=jazz",
    "/v1/music_tracker/top_artist_by_tag_year_range?start_year=1980&end_year=2000&tag=disco",
    "/v1/music_tracker/artists_by_release_type_tag?release_type=single&tag=1970s",
    "/v1/music_tracker/percentage_torrents_by_tag_year_range_release_type?tag=united.states&start_year=1979&end_year=1982&release_type=album"
]
