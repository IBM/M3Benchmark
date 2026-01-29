from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/donor/donor.sqlite')
cursor = conn.cursor()

# Endpoint to get the sum of donation totals for a specific year
@app.get("/v1/donor/sum_donation_total_by_year", operation_id="get_sum_donation_total_by_year", summary="Retrieves the total sum of donations made in a specific year. The year is provided in the 'YYYY' format.")
async def get_sum_donation_total_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(donation_total) FROM donations WHERE donation_timestamp LIKE ?", (year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the donation timestamp and total for the highest donation
@app.get("/v1/donor/highest_donation", operation_id="get_highest_donation", summary="Retrieves the timestamp and total amount of the highest donation made. This operation identifies the donation with the maximum total and returns its timestamp and total value.")
async def get_highest_donation():
    cursor.execute("SELECT donation_timestamp, donation_total FROM donations WHERE donation_total = ( SELECT donation_total FROM donations ORDER BY donation_total DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"donations": []}
    return {"donations": result}

# Endpoint to get the sum of donation totals for a specific year, via giving page, and for honoree
@app.get("/v1/donor/sum_donation_total_by_criteria", operation_id="get_sum_donation_total_by_criteria", summary="Retrieves the total sum of donations made in a specific year, through a giving page, and for a particular honoree. The year is provided in 'YYYY' format, and the giving page and honoree flags indicate whether the donations were made via a giving page or for a specific honoree.")
async def get_sum_donation_total_by_criteria(year: str = Query(..., description="Year in 'YYYY' format"), via_giving_page: str = Query(..., description="Via giving page flag"), for_honoree: str = Query(..., description="For honoree flag")):
    cursor.execute("SELECT SUM(donation_total) FROM donations WHERE donation_timestamp LIKE ? AND via_giving_page = ? AND for_honoree = ?", (year + '%', via_giving_page, for_honoree))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the donor account ID and the ratio of optional support to total donation for non-teacher accounts
@app.get("/v1/donor/donation_ratio_non_teacher", operation_id="get_donation_ratio_non_teacher", summary="Retrieves the account ID and the ratio of optional support to total donation for non-teacher accounts. The input parameter determines whether the account is a teacher account or not.")
async def get_donation_ratio_non_teacher(is_teacher_acct: str = Query(..., description="Is teacher account flag")):
    cursor.execute("SELECT donor_acctid, donation_optional_support / donation_total FROM donations WHERE is_teacher_acct = ?", (is_teacher_acct,))
    result = cursor.fetchall()
    if not result:
        return {"donations": []}
    return {"donations": result}

# Endpoint to get essay titles based on the primary focus subject of the project
@app.get("/v1/donor/essay_titles_by_subject", operation_id="get_essay_titles_by_subject", summary="Retrieves a list of essay titles associated with projects that have the specified primary focus subject. The primary focus subject is a parameter that filters the results.")
async def get_essay_titles_by_subject(primary_focus_subject: str = Query(..., description="Primary focus subject of the project")):
    cursor.execute("SELECT T1.title FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.primary_focus_subject = ?", (primary_focus_subject,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get essay titles based on the poverty level of the project
@app.get("/v1/donor/essay_titles_by_poverty_level", operation_id="get_essay_titles_by_poverty_level", summary="Retrieves a list of essay titles associated with projects that match the specified poverty level. The poverty level is used to filter the projects and return the corresponding essay titles.")
async def get_essay_titles_by_poverty_level(poverty_level: str = Query(..., description="Poverty level of the project")):
    cursor.execute("SELECT T1.title FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.poverty_level LIKE ?", (poverty_level + '%',))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the primary focus subject of projects based on the essay title
@app.get("/v1/donor/primary_focus_subject_by_essay_title", operation_id="get_primary_focus_subject_by_essay_title", summary="Retrieves the primary focus subject of projects associated with a given essay title. The operation filters projects based on the provided essay title and returns the primary focus subject of the matching project.")
async def get_primary_focus_subject_by_essay_title(essay_title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T2.primary_focus_subject FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.title = ?", (essay_title,))
    result = cursor.fetchall()
    if not result:
        return {"subjects": []}
    return {"subjects": [row[0] for row in result]}

# Endpoint to get essay titles based on the donation message
@app.get("/v1/donor/essay_titles_by_donation_message", operation_id="get_essay_titles_by_donation_message", summary="Retrieves essay titles associated with a specific donation message. The operation filters essays based on the provided donation message and returns the corresponding essay titles.")
async def get_essay_titles_by_donation_message(donation_message: str = Query(..., description="Donation message")):
    cursor.execute("SELECT T1.title FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.donation_message LIKE ?", (donation_message + '%',))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get essay titles and total price excluding optional support based on the number of students reached
@app.get("/v1/donor/essay_titles_and_price_by_students_reached", operation_id="get_essay_titles_and_price_by_students_reached", summary="Retrieves the titles of essays and their corresponding total price (excluding optional support) for projects that have reached a specified number of students. The number of students reached is a parameter that filters the results.")
async def get_essay_titles_and_price_by_students_reached(students_reached: int = Query(..., description="Number of students reached")):
    cursor.execute("SELECT T1.title, T2.total_price_excluding_optional_support FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.students_reached = ?", (students_reached,))
    result = cursor.fetchall()
    if not result:
        return {"essays": []}
    return {"essays": result}

# Endpoint to get donation messages and essay titles based on the donor city
@app.get("/v1/donor/donation_messages_and_essay_titles_by_city", operation_id="get_donation_messages_and_essay_titles_by_city", summary="Retrieves donation messages and essay titles associated with a specific donor city. The operation filters the donations based on the provided donor city and returns the corresponding donation messages and essay titles.")
async def get_donation_messages_and_essay_titles_by_city(donor_city: str = Query(..., description="Donor city")):
    cursor.execute("SELECT T2.donation_message, T1.title FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.donor_city = ?", (donor_city,))
    result = cursor.fetchall()
    if not result:
        return {"donations": []}
    return {"donations": result}

# Endpoint to get resource details based on essay title
@app.get("/v1/donor/resource_details_by_essay_title", operation_id="get_resource_details", summary="Retrieves the vendor name, item name, and item unit price for resources associated with a specific essay title. The essay title is provided as an input parameter.")
async def get_resource_details(essay_title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T1.vendor_name, T1.item_name, T1.item_unit_price FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN essays AS T3 ON T2.projectid = T3.projectid WHERE T3.title = ?", (essay_title,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the total donation amount for a project based on essay title
@app.get("/v1/donor/total_donation_by_essay_title", operation_id="get_total_donation", summary="Retrieves the cumulative donation amount for a project associated with a specific essay title. The operation calculates the total donation by aggregating all contributions linked to the project via the essay title.")
async def get_total_donation(essay_title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT SUM(T2.donation_to_project) FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.title = ?", (essay_title,))
    result = cursor.fetchone()
    if not result:
        return {"total_donation": []}
    return {"total_donation": result[0]}

# Endpoint to get essay details based on teacher NY teaching fellow status
@app.get("/v1/donor/essay_details_by_teacher_status", operation_id="get_essay_details", summary="Retrieves the title and short description of essays associated with projects led by teachers who are New York teaching fellows. The operation requires the teacher's NY teaching fellow status as input, which is a boolean value ('t' for true, 'f' for false).")
async def get_essay_details(teacher_ny_teaching_fellow: str = Query(..., description="Teacher NY teaching fellow status ('t' for true, 'f' for false)")):
    cursor.execute("SELECT T1.title, T1.short_description FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.teacher_ny_teaching_fellow = ?", (teacher_ny_teaching_fellow,))
    result = cursor.fetchall()
    if not result:
        return {"essay_details": []}
    return {"essay_details": result}

# Endpoint to get project titles and total prices based on students reached and poverty level
@app.get("/v1/donor/project_titles_prices_by_students_poverty", operation_id="get_project_titles_prices", summary="Retrieves unique project titles and their total prices (excluding optional support) for projects that have reached at least a specified number of students and match a given poverty level.")
async def get_project_titles_prices(students_reached: int = Query(..., description="Minimum number of students reached"), poverty_level: str = Query(..., description="Poverty level (e.g., 'moderate poverty')")):
    cursor.execute("SELECT DISTINCT T2.title, T1.total_price_excluding_optional_support FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T1.students_reached >= ? AND T1.poverty_level LIKE ?", (students_reached, poverty_level))
    result = cursor.fetchall()
    if not result:
        return {"project_details": []}
    return {"project_details": result}

# Endpoint to get the percentage of donations to rural schools
@app.get("/v1/donor/percentage_donations_to_rural_schools", operation_id="get_percentage_donations_to_rural_schools", summary="Retrieves the percentage of total donations that have been allocated to rural schools. The calculation is based on the sum of donations to rural school projects compared to the total sum of donations across all projects. The operation requires the specification of a school metro type to determine the rural school projects.")
async def get_percentage_donations_to_rural_schools(school_metro: str = Query(..., description="School metro type (e.g., 'rural')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.school_metro = ? THEN T1.donation_to_project ELSE 0 END) AS REAL) * 100 / SUM(donation_to_project) FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid", (school_metro,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the top project by total price excluding optional support
@app.get("/v1/donor/top_project_by_total_price", operation_id="get_top_project_by_total_price", summary="Get the top project by total price excluding optional support, including the title, total donations, and the percentage of the remaining amount")
async def get_top_project_by_total_price():
    cursor.execute("SELECT T1.title, SUM(T3.donation_to_project), CAST((T2.total_price_excluding_optional_support - SUM(T3.donation_to_project)) AS REAL) * 100 / SUM(T3.donation_to_project) FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN donations AS T3 ON T2.projectid = T3.projectid ORDER BY T2.total_price_excluding_optional_support DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"project_details": []}
    return {"project_details": result}

# Endpoint to get the count of projects based on school district and poverty level
@app.get("/v1/donor/count_projects_by_district_poverty", operation_id="get_count_projects", summary="Retrieves the total number of projects in a specific school district that correspond to a given poverty level. The poverty level can be 'highest poverty', 'high poverty', 'mid-high poverty', 'mid poverty', 'low poverty', or 'lowest poverty'.")
async def get_count_projects(school_district: str = Query(..., description="School district name"), poverty_level: str = Query(..., description="Poverty level (e.g., 'highest poverty')")):
    cursor.execute("SELECT COUNT(poverty_level) FROM projects WHERE school_district = ? AND poverty_level = ?", (school_district, poverty_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of donations based on teacher account status and donor state
@app.get("/v1/donor/count_donations_by_teacher_state", operation_id="get_count_donations", summary="Retrieves the total number of donations made by teachers or non-teachers from a specific state. The operation filters donations based on the donor's teacher status and state, providing a count of qualifying donations.")
async def get_count_donations(is_teacher_acct: str = Query(..., description="Teacher account status ('t' for true, 'f' for false)"), donor_state: str = Query(..., description="Donor state (e.g., 'CO')")):
    cursor.execute("SELECT COUNT(donationid) FROM donations WHERE is_teacher_acct = ? AND donor_state = ?", (is_teacher_acct, donor_state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the project with the highest total price including optional support
@app.get("/v1/donor/top_project_by_total_price_including_support", operation_id="get_top_project_including_support", summary="Retrieves the project with the highest total price, inclusive of optional support. The total price is calculated by summing the base price and any optional support costs associated with the project.")
async def get_top_project_including_support():
    cursor.execute("SELECT projectid FROM projects ORDER BY total_price_including_optional_support DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"projectid": []}
    return {"projectid": result[0]}

# Endpoint to get distinct donor states based on specific donation criteria
@app.get("/v1/donor/distinct_donor_states_by_criteria", operation_id="get_distinct_donor_states", summary="Retrieve the state with the highest total donation amount that meets specific donation criteria. The criteria include whether the donation is for an honoree, whether the payment included a campaign gift card, and the payment method used. The state returned is the one with the highest total donation amount among all states, based on the provided criteria.")
async def get_distinct_donor_states(for_honoree: str = Query(..., description="Whether the donation is for an honoree ('t' for true, 'f' for false)"), payment_included_campaign_gift_card: str = Query(..., description="Whether the payment included a campaign gift card ('t' for true, 'f' for false)"), payment_method: str = Query(..., description="Payment method (e.g., 'paypal')")):
    cursor.execute("SELECT DISTINCT donor_state FROM donations WHERE for_honoree = ? AND payment_included_campaign_gift_card = ? AND payment_method = ? AND donor_state = ( SELECT donor_state FROM donations GROUP BY donor_state ORDER BY SUM(donation_total) DESC LIMIT 1 )", (for_honoree, payment_included_campaign_gift_card, payment_method))
    result = cursor.fetchall()
    if not result:
        return {"donor_states": []}
    return {"donor_states": result}

# Endpoint to get the project with the highest optional support price difference
@app.get("/v1/donor/highest_optional_support_price_difference", operation_id="get_highest_optional_support_price_difference", summary="Retrieves the project with the largest discrepancy between its total price with optional support and its total price without optional support. The returned data includes the project's ID and item name.")
async def get_highest_optional_support_price_difference():
    cursor.execute("SELECT T1.projectid, T1.item_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid ORDER BY T2.total_price_including_optional_support - T2.total_price_excluding_optional_support DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"project": []}
    return {"project": {"projectid": result[0], "item_name": result[1]}}

# Endpoint to get distinct item details for a specific project
@app.get("/v1/donor/distinct_item_details", operation_id="get_distinct_item_details", summary="Retrieves unique item details, including item name, quantity, and teacher prefix, for a specific project. The project is identified by its unique ID.")
async def get_distinct_item_details(projectid: str = Query(..., description="Project ID")):
    cursor.execute("SELECT DISTINCT T1.item_name, T1.item_quantity, T2.teacher_prefix FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.projectid = ?", (projectid,))
    results = cursor.fetchall()
    if not results:
        return {"items": []}
    return {"items": [{"item_name": row[0], "item_quantity": row[1], "teacher_prefix": row[2]} for row in results]}

# Endpoint to get the total price including optional support for projects with a specific essay title
@app.get("/v1/donor/total_price_including_optional_support", operation_id="get_total_price_including_optional_support", summary="Retrieves the cumulative total price, inclusive of optional support, for all projects associated with a specific essay title. The provided essay title is used to filter the projects and calculate the sum of their total prices, including any optional support costs.")
async def get_total_price_including_optional_support(title: str = Query(..., description="Essay title")):
    cursor.execute("SELECT SUM(T1.total_price_including_optional_support) FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T2.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"total_price": []}
    return {"total_price": result[0]}

# Endpoint to get school coordinates for a specific essay title
@app.get("/v1/donor/school_coordinates", operation_id="get_school_coordinates", summary="Retrieves the geographical coordinates of the school associated with a specific essay title. The operation returns the latitude and longitude of the school, enabling the location of the school to be determined.")
async def get_school_coordinates(title: str = Query(..., description="Essay title")):
    cursor.execute("SELECT T1.school_latitude, T1.school_longitude FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T2.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"coordinates": []}
    return {"coordinates": {"latitude": result[0], "longitude": result[1]}}

# Endpoint to get the highest donation and corresponding essay title
@app.get("/v1/donor/highest_donation_with_essay_title", operation_id="get_highest_donation_with_essay_title", summary="Retrieves the highest donation amount and the title of the essay associated with the project that received this donation. This operation provides a snapshot of the most generous donation and the corresponding essay, offering insights into the project that attracted the highest financial support.")
async def get_highest_donation_with_essay_title():
    cursor.execute("SELECT T2.donation_total, T1.title FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.donation_total = ( SELECT MAX(donation_total) FROM donations )")
    result = cursor.fetchone()
    if not result:
        return {"donation": []}
    return {"donation": {"donation_total": result[0], "title": result[1]}}

# Endpoint to get the most common project resource type among the top 10 donations
@app.get("/v1/donor/most_common_project_resource_type", operation_id="get_most_common_project_resource_type", summary="Retrieves the most frequently occurring project resource type among the top 10 donations. This operation identifies the top donations and determines the most common resource type associated with their respective projects.")
async def get_most_common_project_resource_type():
    cursor.execute("SELECT project_resource_type FROM ( SELECT T1.donor_acctid, T3.project_resource_type FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN resources AS T3 ON T2.projectid = T3.projectid ORDER BY T1.donation_total DESC LIMIT 10 ) GROUP BY project_resource_type ORDER BY COUNT(project_resource_type) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"resource_type": []}
    return {"resource_type": result[0]}

# Endpoint to get the item details for the earliest posted project
@app.get("/v1/donor/earliest_posted_project_item", operation_id="get_earliest_posted_project_item", summary="Get the item details for the earliest posted project")
async def get_earliest_posted_project_item():
    cursor.execute("SELECT T2.date_posted, T1.item_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.date_posted = ( SELECT date_posted FROM projects ORDER BY date_posted ASC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"item": []}
    return {"item": {"date_posted": result[0], "item_name": result[1]}}

# Endpoint to get distinct vendor names for a specific essay title
@app.get("/v1/donor/distinct_vendor_names", operation_id="get_distinct_vendor_names", summary="Retrieves a unique list of vendor names associated with a specific essay title. The operation filters resources based on the essay title and returns the distinct vendor names linked to those resources.")
async def get_distinct_vendor_names(title: str = Query(..., description="Essay title")):
    cursor.execute("SELECT DISTINCT T1.vendor_name FROM resources AS T1 INNER JOIN essays AS T3 ON T1.projectid = T3.projectid WHERE T3.title = ?", (title,))
    results = cursor.fetchall()
    if not results:
        return {"vendors": []}
    return {"vendors": [row[0] for row in results]}

# Endpoint to get the project with the highest item quantity
@app.get("/v1/donor/highest_item_quantity_project", operation_id="get_highest_item_quantity_project", summary="Retrieves the project with the highest item quantity, along with its date posted and grade level. The project is determined by comparing the item quantities of all available resources and selecting the one with the highest value.")
async def get_highest_item_quantity_project():
    cursor.execute("SELECT T2.date_posted, T2.grade_level FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid ORDER BY T1.item_quantity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"project": []}
    return {"project": {"date_posted": result[0], "grade_level": result[1]}}

# Endpoint to get the average donation to a project based on essay title
@app.get("/v1/donor/average_donation_by_essay_title", operation_id="get_average_donation_by_essay_title", summary="Retrieves the average donation amount made to a project associated with a specific essay title. The calculation is based on the donations linked to the project that the essay is associated with.")
async def get_average_donation_by_essay_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT AVG(T3.donation_to_project) FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN donations AS T3 ON T2.projectid = T3.projectid WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"average_donation": []}
    return {"average_donation": result[0]}

# Endpoint to get the top donation to a project based on essay title
@app.get("/v1/donor/top_donation_by_essay_title", operation_id="get_top_donation_by_essay_title", summary="Retrieves the top donation made to a project, based on the essay title. The operation calculates the donation amount as a proportion of the total project price, excluding optional support. The result is ordered in descending order by the donation amount, with the top donation returned.")
async def get_top_donation_by_essay_title():
    cursor.execute("SELECT T1.title, T3.donor_acctid, CAST(T3.donation_to_project AS REAL) / T2.total_price_excluding_optional_support FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN donations AS T3 ON T2.projectid = T3.projectid ORDER BY T3.donation_to_project DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"top_donation": []}
    return {"top_donation": result}

# Endpoint to get the top 5 donations with optional support by school state
@app.get("/v1/donor/top_donations_with_optional_support", operation_id="get_top_donations_with_optional_support", summary="Retrieves the top 5 donations with optional support, grouped by school state and resource type. The average optional support amount is calculated for each donor account. The results are ordered by the optional support amount in descending order.")
async def get_top_donations_with_optional_support():
    cursor.execute("SELECT T1.school_state, T2.donor_acctid, AVG(T2.donation_optional_support), T1.resource_type FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid ORDER BY T2.donation_optional_support DESC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"top_donations": []}
    return {"top_donations": result}

# Endpoint to get the most common item name in a specific school city
@app.get("/v1/donor/most_common_item_by_school_city", operation_id="get_most_common_item_by_school_city", summary="Retrieves the most frequently donated item in a specific school city. The operation filters resources based on the provided school city and returns the item name that has been donated the most. The input parameter specifies the city of the school.")
async def get_most_common_item_by_school_city(school_city: str = Query(..., description="City of the school")):
    cursor.execute("SELECT T1.projectid, T1.item_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.school_city LIKE ? GROUP BY T1.item_name ORDER BY COUNT(T1.item_name) DESC LIMIT 1", (school_city,))
    result = cursor.fetchone()
    if not result:
        return {"most_common_item": []}
    return {"most_common_item": result}

# Endpoint to get the count of schools based on resource type and school metro
@app.get("/v1/donor/count_schools_by_resource_type_and_metro", operation_id="get_count_schools_by_resource_type_and_metro", summary="Retrieves the total number of schools that have a specific resource type and are located in a particular metro area. The resource type and metro area are provided as input parameters.")
async def get_count_schools_by_resource_type_and_metro(resource_type: str = Query(..., description="Type of resource"), school_metro: str = Query(..., description="Metro area of the school")):
    cursor.execute("SELECT COUNT(T2.schoolid) FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.resource_type = ? AND T2.school_metro = ?", (resource_type, school_metro))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the school with the most resources from a specific vendor
@app.get("/v1/donor/school_with_most_resources_from_vendor", operation_id="get_school_with_most_resources_from_vendor", summary="Retrieves the school that has received the highest number of resources from a specified vendor. The operation filters resources based on the provided vendor name and calculates the count of resources per school. The school with the maximum count is returned.")
async def get_school_with_most_resources_from_vendor(vendor_name: str = Query(..., description="Name of the vendor")):
    cursor.execute("SELECT T2.schoolid FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.vendor_name LIKE ? GROUP BY T2.schoolid ORDER BY COUNT(T1.vendor_name) DESC LIMIT 1", (vendor_name,))
    result = cursor.fetchone()
    if not result:
        return {"schoolid": []}
    return {"schoolid": result[0]}

# Endpoint to get the count of schools based on donor city and excluding a specific school city
@app.get("/v1/donor/count_schools_by_donor_city_excluding_school_city", operation_id="get_count_schools_by_donor_city_excluding_school_city", summary="Retrieves the total number of schools that have received donations from a specific donor city, excluding a particular school city from the count. The operation filters donations based on the donor's city and excludes projects associated with the specified school city.")
async def get_count_schools_by_donor_city_excluding_school_city(donor_city: str = Query(..., description="City of the donor"), school_city: str = Query(..., description="City of the school to exclude")):
    cursor.execute("SELECT COUNT(T2.schoolid) FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.donor_city = ? AND T2.school_city NOT LIKE ?", (donor_city, school_city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get teacher prefixes based on essay title
@app.get("/v1/donor/teacher_prefixes_by_essay_title", operation_id="get_teacher_prefixes_by_essay_title", summary="Retrieves the prefixes of teachers associated with a specific essay title. The operation filters essays by the provided title and returns the prefixes of the teachers linked to those essays.")
async def get_teacher_prefixes_by_essay_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T2.teacher_prefix FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"teacher_prefixes": []}
    return {"teacher_prefixes": result}

# Endpoint to get the number of students reached based on essay title
@app.get("/v1/donor/students_reached_by_essay_title", operation_id="get_students_reached_by_essay_title", summary="Retrieves the total number of students who have been reached by a specific essay. The essay is identified by its title. The response includes the count of students who have been impacted by the essay.")
async def get_students_reached_by_essay_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T2.students_reached FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"students_reached": []}
    return {"students_reached": result}

# Endpoint to get essay titles and donor account IDs based on school city
@app.get("/v1/donor/essay_titles_and_donor_ids_by_school_city", operation_id="get_essay_titles_and_donor_ids_by_school_city", summary="Retrieves a list of essay titles and corresponding donor account IDs associated with projects in a specific school city. The city is provided as an input parameter.")
async def get_essay_titles_and_donor_ids_by_school_city(school_city: str = Query(..., description="City of the school")):
    cursor.execute("SELECT T1.title, T3.donor_acctid FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN donations AS T3 ON T2.projectid = T3.projectid WHERE T2.school_city LIKE ?", (school_city,))
    result = cursor.fetchall()
    if not result:
        return {"essay_titles_and_donor_ids": []}
    return {"essay_titles_and_donor_ids": result}

# Endpoint to get essay titles based on teacher prefix and minimum students reached
@app.get("/v1/donor/essay_titles_by_teacher_prefix_and_students_reached", operation_id="get_essay_titles", summary="Retrieves essay titles from projects that meet the specified teacher prefix and minimum number of students reached criteria. The teacher prefix is a string that matches the beginning of a teacher's name, and the minimum number of students reached is an integer that filters projects based on the number of students impacted.")
async def get_essay_titles(teacher_prefix: str = Query(..., description="Teacher prefix (e.g., 'Dr.')"), min_students_reached: int = Query(..., description="Minimum number of students reached")):
    cursor.execute("SELECT T1.title FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.teacher_prefix LIKE ? AND T2.students_reached > ?", (teacher_prefix, min_students_reached))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get essay titles for the project with the highest total item cost
@app.get("/v1/donor/essay_titles_highest_total_item_cost", operation_id="get_essay_titles_highest_total_item_cost", summary="Retrieves the titles of essays associated with the project that has the highest total cost of items. The total cost is calculated by multiplying the unit price and quantity of each item in the project.")
async def get_essay_titles_highest_total_item_cost():
    cursor.execute("SELECT T1.title FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.projectid = ( SELECT projectid FROM resources ORDER BY item_unit_price * item_quantity DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of donations by teacher account status and donor city
@app.get("/v1/donor/donation_count_by_teacher_account_and_city", operation_id="get_donation_count", summary="Retrieves the total number of donations made by teacher accounts, filtered by their account status and the city of the donor. The account status and donor city are provided as input parameters.")
async def get_donation_count(is_teacher_acct: str = Query(..., description="Teacher account status (e.g., 't')"), donor_city: str = Query(..., description="Donor city (e.g., 'New York')")):
    cursor.execute("SELECT COUNT(donationid) FROM donations WHERE is_teacher_acct = ? AND donor_city = ?", (is_teacher_acct, donor_city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of donations by honoree status and donor state
@app.get("/v1/donor/donation_count_by_honoree_and_state", operation_id="get_donation_count_honoree", summary="Retrieves the total number of donations made by donors from a specific state for a particular honoree status. The honoree status and donor state are provided as input parameters.")
async def get_donation_count_honoree(for_honoree: str = Query(..., description="Honoree status (e.g., 't')"), donor_state: str = Query(..., description="Donor state (e.g., 'NJ')")):
    cursor.execute("SELECT COUNT(donationid) FROM donations WHERE for_honoree = ? AND donor_state = ?", (for_honoree, donor_state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the donation message by donation ID
@app.get("/v1/donor/donation_message_by_id", operation_id="get_donation_message", summary="Retrieves the donation message associated with a specific donation, identified by its unique donation ID. The donation ID is a required input parameter.")
async def get_donation_message(donationid: str = Query(..., description="Donation ID (e.g., 'a84dace1ff716f6f0c7af8ef9090a5d5')")):
    cursor.execute("SELECT donation_message FROM donations WHERE donationid = ?", (donationid,))
    result = cursor.fetchone()
    if not result:
        return {"message": []}
    return {"message": result[0]}

# Endpoint to get distinct project resource types by vendor name
@app.get("/v1/donor/distinct_project_resource_types_by_vendor", operation_id="get_distinct_project_resource_types", summary="Retrieves a unique set of resource types associated with a specific vendor. The vendor is identified by its name, which is provided as an input parameter. This operation is useful for understanding the variety of resources a vendor offers.")
async def get_distinct_project_resource_types(vendor_name: str = Query(..., description="Vendor name (e.g., 'Lakeshore Learning Materials')")):
    cursor.execute("SELECT DISTINCT project_resource_type FROM resources WHERE vendor_name = ?", (vendor_name,))
    result = cursor.fetchall()
    if not result:
        return {"resource_types": []}
    return {"resource_types": [row[0] for row in result]}

# Endpoint to get the item name with the highest quantity by vendor name
@app.get("/v1/donor/item_name_highest_quantity_by_vendor", operation_id="get_item_name_highest_quantity", summary="Retrieves the name of the item with the highest quantity associated with the specified vendor. The vendor is identified by its name, and the item with the highest quantity is determined by sorting the items in descending order based on their quantities.")
async def get_item_name_highest_quantity(vendor_name: str = Query(..., description="Vendor name (e.g., 'Lakeshore Learning Materials')")):
    cursor.execute("SELECT item_name FROM resources WHERE vendor_name = ? ORDER BY item_quantity DESC LIMIT 1", (vendor_name,))
    result = cursor.fetchone()
    if not result:
        return {"item_name": []}
    return {"item_name": result[0]}

# Endpoint to get the count of projects by teacher NY teaching fellow status and donor city
@app.get("/v1/donor/project_count_by_teacher_fellow_and_city", operation_id="get_project_count", summary="Retrieves the total number of projects associated with a specific teacher NY teaching fellow status and donor city. The response is based on the provided teacher's NY teaching fellow status and the donor's city.")
async def get_project_count(teacher_ny_teaching_fellow: str = Query(..., description="Teacher NY teaching fellow status (e.g., 't')"), donor_city: str = Query(..., description="Donor city (e.g., 'New York')")):
    cursor.execute("SELECT COUNT(T1.projectid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.teacher_ny_teaching_fellow = ? AND T2.donor_city = ?", (teacher_ny_teaching_fellow, donor_city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of projects by vendor name and teacher prefix
@app.get("/v1/donor/project_count_by_vendor_and_teacher_prefix", operation_id="get_project_count_vendor_prefix", summary="Retrieves the total number of projects associated with a specific vendor and teacher prefix. The count is determined by matching the provided vendor name and teacher prefix with the corresponding records in the resources and projects tables, respectively.")
async def get_project_count_vendor_prefix(vendor_name: str = Query(..., description="Vendor name (e.g., 'Lakeshore Learning Materials')"), teacher_prefix: str = Query(..., description="Teacher prefix (e.g., 'Dr.')")):
    cursor.execute("SELECT COUNT(T1.projectid) FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.vendor_name = ? AND T2.teacher_prefix = ?", (vendor_name, teacher_prefix))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get vendor names based on poverty level
@app.get("/v1/donor/vendor_names_by_poverty_level", operation_id="get_vendor_names_by_poverty_level", summary="Retrieves the names of vendors associated with projects that have a specified poverty level. The poverty level is provided as an input parameter.")
async def get_vendor_names_by_poverty_level(poverty_level: str = Query(..., description="Poverty level of the project")):
    cursor.execute("SELECT T1.vendor_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.poverty_level = ?", (poverty_level,))
    result = cursor.fetchall()
    if not result:
        return {"vendor_names": []}
    return {"vendor_names": [row[0] for row in result]}

# Endpoint to get school ID based on vendor name
@app.get("/v1/donor/school_id_by_vendor_name", operation_id="get_school_id_by_vendor_name", summary="Retrieves the school ID associated with the project that has the highest fulfillment labor materials, among projects with a specific vendor name. The vendor name is provided as an input parameter.")
async def get_school_id_by_vendor_name(vendor_name: str = Query(..., description="Vendor name")):
    cursor.execute("SELECT T2.schoolid FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.vendor_name = ? ORDER BY T2.fulfillment_labor_materials DESC LIMIT 1", (vendor_name,))
    result = cursor.fetchone()
    if not result:
        return {"school_id": []}
    return {"school_id": result[0]}

# Endpoint to get vendor name based on price difference
@app.get("/v1/donor/vendor_name_by_price_difference", operation_id="get_vendor_name_by_price_difference", summary="Retrieves the name of the vendor associated with the project that has the largest difference between the total price including optional support and the base price. The result is ordered in descending order based on this price difference.")
async def get_vendor_name_by_price_difference():
    cursor.execute("SELECT T1.vendor_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid ORDER BY T2.total_price_including_optional_support - T2.total_price_including_optional_support DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"vendor_name": []}
    return {"vendor_name": result[0]}

# Endpoint to get total donations by school city
@app.get("/v1/donor/total_donations_by_school_city", operation_id="get_total_donations_by_school_city", summary="Retrieves the cumulative sum of donations for all projects associated with a specific school city. The city of the school is provided as an input parameter.")
async def get_total_donations_by_school_city(school_city: str = Query(..., description="City of the school")):
    cursor.execute("SELECT SUM(T2.donation_total) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.school_city = ?", (school_city,))
    result = cursor.fetchone()
    if not result:
        return {"total_donations": []}
    return {"total_donations": result[0]}

# Endpoint to get donation messages by school city and magnet status
@app.get("/v1/donor/donation_messages_by_school_city_and_magnet", operation_id="get_donation_messages_by_school_city_and_magnet", summary="Retrieves donation messages for projects associated with a specific school city and magnet status. The operation filters projects based on the provided city and magnet status, and returns the corresponding donation messages.")
async def get_donation_messages_by_school_city_and_magnet(school_city: str = Query(..., description="City of the school"), school_magnet: str = Query(..., description="Magnet status of the school (e.g., 't' for true)")):
    cursor.execute("SELECT T2.donation_message FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.school_city = ? AND T1.school_magnet = ?", (school_city, school_magnet))
    result = cursor.fetchall()
    if not result:
        return {"donation_messages": []}
    return {"donation_messages": [row[0] for row in result]}

# Endpoint to get count of projects by payment and year-round status
@app.get("/v1/donor/count_projects_by_payment_and_year_round", operation_id="get_count_projects_by_payment_and_year_round", summary="Retrieve the total number of projects that meet the specified payment and year-round criteria. The operation filters projects based on whether they include account credit in their payment and whether they are year-round schools. The result is a count of projects that match the provided conditions.")
async def get_count_projects_by_payment_and_year_round(payment_included_acct_credit: str = Query(..., description="Payment included account credit status (e.g., 't' for true)"), school_year_round: str = Query(..., description="Year-round status of the school (e.g., 't' for true)")):
    cursor.execute("SELECT COUNT(T1.projectid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.payment_included_acct_credit = ? AND T1.school_year_round = ?", (payment_included_acct_credit, school_year_round))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get total dollar amount by primary focus area
@app.get("/v1/donor/total_dollar_amount_by_primary_focus_area", operation_id="get_total_dollar_amount_by_primary_focus_area", summary="Retrieve the total monetary value of donations for projects that have a specified primary focus area. The primary focus area is a parameter that filters the projects considered for the total calculation.")
async def get_total_dollar_amount_by_primary_focus_area(primary_focus_area: str = Query(..., description="Primary focus area of the project")):
    cursor.execute("SELECT SUM(T2.dollar_amount) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.primary_focus_area = ?", (primary_focus_area,))
    result = cursor.fetchone()
    if not result:
        return {"total_dollar_amount": []}
    return {"total_dollar_amount": result[0]}

# Endpoint to get donor account ID by primary focus area
@app.get("/v1/donor/donor_account_id_by_primary_focus_area", operation_id="get_donor_account_id_by_primary_focus_area", summary="Retrieves the account ID of the top donor for projects primarily focused on a specific area, ranked by the total donation amount.")
async def get_donor_account_id_by_primary_focus_area(primary_focus_area: str = Query(..., description="Primary focus area of the project")):
    cursor.execute("SELECT T2.donor_acctid FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.primary_focus_area = ? ORDER BY T2.donation_total DESC LIMIT 1", (primary_focus_area,))
    result = cursor.fetchone()
    if not result:
        return {"donor_account_id": []}
    return {"donor_account_id": result[0]}

# Endpoint to get item name by primary focus area and resource type
@app.get("/v1/donor/item_name_by_primary_focus_area_and_resource_type", operation_id="get_item_name_by_primary_focus_area_and_resource_type", summary="Retrieves the name of the item with the highest quantity for projects that have a specified primary focus area and resource type. The item name is determined by considering the project's primary focus area and resource type, and the results are ordered by item quantity in descending order.")
async def get_item_name_by_primary_focus_area_and_resource_type(primary_focus_area: str = Query(..., description="Primary focus area of the project"), project_resource_type: str = Query(..., description="Resource type of the project")):
    cursor.execute("SELECT T1.item_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.primary_focus_area = ? AND T1.project_resource_type = ? ORDER BY T1.item_quantity DESC LIMIT 1", (primary_focus_area, project_resource_type))
    result = cursor.fetchone()
    if not result:
        return {"item_name": []}
    return {"item_name": result[0]}

# Endpoint to get item name by primary focus area
@app.get("/v1/donor/item_name_by_primary_focus_area", operation_id="get_item_name_by_primary_focus_area", summary="Retrieves the name of the most expensive item associated with projects that have a specified primary focus area. The item name is determined by ordering the items by their unit price in descending order and selecting the top result.")
async def get_item_name_by_primary_focus_area(primary_focus_area: str = Query(..., description="Primary focus area of the project")):
    cursor.execute("SELECT T1.item_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.primary_focus_area = ? ORDER BY T1.item_unit_price DESC LIMIT 1", (primary_focus_area,))
    result = cursor.fetchone()
    if not result:
        return {"item_name": []}
    return {"item_name": result[0]}

# Endpoint to get the average donation total for projects in a specific city
@app.get("/v1/donor/average_donation_total_by_city", operation_id="get_average_donation_total", summary="Retrieves the average total donation amount for projects in a specified city. The calculation is based on the sum of all donations for projects in the given city, divided by the total number of donations.")
async def get_average_donation_total(school_city: str = Query(..., description="City of the school")):
    cursor.execute("SELECT SUM(T2.donation_total) / COUNT(donationid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.school_city = ?", (school_city,))
    result = cursor.fetchone()
    if not result:
        return {"average_donation_total": []}
    return {"average_donation_total": result[0]}

# Endpoint to get distinct donor cities for a specific donor account ID
@app.get("/v1/donor/distinct_donor_cities", operation_id="get_distinct_donor_cities", summary="Retrieves a list of unique cities where donations have been made by a specific donor account. The donor account is identified by the provided donor_acctid.")
async def get_distinct_donor_cities(donor_acctid: str = Query(..., description="Donor account ID")):
    cursor.execute("SELECT DISTINCT donor_city FROM donations WHERE donor_acctid = ?", (donor_acctid,))
    result = cursor.fetchall()
    if not result:
        return {"donor_cities": []}
    return {"donor_cities": [row[0] for row in result]}

# Endpoint to get distinct teacher account statuses for a specific donor account ID
@app.get("/v1/donor/distinct_teacher_account_statuses", operation_id="get_distinct_teacher_account_statuses", summary="Retrieve a unique set of teacher account statuses associated with a specific donor account. The operation filters donations based on the provided donor account ID and returns the distinct teacher account statuses.")
async def get_distinct_teacher_account_statuses(donor_acctid: str = Query(..., description="Donor account ID")):
    cursor.execute("SELECT DISTINCT is_teacher_acct FROM donations WHERE donor_acctid = ?", (donor_acctid,))
    result = cursor.fetchall()
    if not result:
        return {"teacher_account_statuses": []}
    return {"teacher_account_statuses": [row[0] for row in result]}

# Endpoint to check if a teacher has a specific prefix
@app.get("/v1/donor/check_teacher_prefix", operation_id="check_teacher_prefix", summary="This operation verifies whether a specified teacher, identified by their account ID, has a particular prefix. The result is a 'Yes' or 'No' response, indicating the presence or absence of the specified prefix for the teacher.")
async def check_teacher_prefix(teacher_prefix: str = Query(..., description="Teacher prefix to check"), teacher_acctid: str = Query(..., description="Teacher account ID")):
    cursor.execute("SELECT CASE WHEN teacher_prefix = ? THEN 'Yes' ELSE 'NO' END FROM projects WHERE teacher_acctid = ?", (teacher_prefix, teacher_acctid))
    result = cursor.fetchone()
    if not result:
        return {"has_prefix": []}
    return {"has_prefix": result[0]}

# Endpoint to get NY teaching fellow status for a specific teacher account ID
@app.get("/v1/donor/ny_teaching_fellow_status", operation_id="get_ny_teaching_fellow_status", summary="Retrieves the New York teaching fellow status for a specific teacher account. The status indicates whether the teacher is a New York teaching fellow or not. The operation requires the teacher's account ID as input.")
async def get_ny_teaching_fellow_status(teacher_acctid: str = Query(..., description="Teacher account ID")):
    cursor.execute("SELECT teacher_ny_teaching_fellow FROM projects WHERE teacher_acctid = ?", (teacher_acctid,))
    result = cursor.fetchone()
    if not result:
        return {"ny_teaching_fellow": []}
    return {"ny_teaching_fellow": result[0]}

# Endpoint to get essay titles for projects in a specific city
@app.get("/v1/donor/essay_titles_by_city", operation_id="get_essay_titles_by_city", summary="Retrieves the titles of essays associated with projects in a specified city. The city is determined by the provided school_city parameter, which filters the projects based on their location.")
async def get_essay_titles_by_city(school_city: str = Query(..., description="City of the school")):
    cursor.execute("SELECT T2.title FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T1.school_city LIKE ?", (school_city,))
    result = cursor.fetchall()
    if not result:
        return {"essay_titles": []}
    return {"essay_titles": [row[0] for row in result]}

# Endpoint to get item names for resources associated with a specific teacher account ID
@app.get("/v1/donor/item_names_by_teacher", operation_id="get_item_names_by_teacher", summary="Retrieves the names of items associated with a specific teacher's projects. The operation requires the teacher's account ID to filter the relevant resources and return the corresponding item names.")
async def get_item_names_by_teacher(teacher_acctid: str = Query(..., description="Teacher account ID")):
    cursor.execute("SELECT T1.item_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.teacher_acctid = ?", (teacher_acctid,))
    result = cursor.fetchall()
    if not result:
        return {"item_names": []}
    return {"item_names": [row[0] for row in result]}

# Endpoint to get the count of school IDs for projects with specific magnet status and payment inclusion
@app.get("/v1/donor/count_school_ids_by_magnet_payment", operation_id="get_count_school_ids", summary="Retrieves the total number of unique school IDs associated with projects that meet the specified magnet status and payment inclusion criteria. The magnet status indicates whether the school is a magnet school or not, and the payment inclusion status determines if the payment is included in the account credit or not.")
async def get_count_school_ids(school_magnet: str = Query(..., description="Magnet status of the school"), payment_included_acct_credit: str = Query(..., description="Payment inclusion status")):
    cursor.execute("SELECT COUNT(T1.schoolid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.school_magnet = ? AND T2.payment_included_acct_credit = ?", (school_magnet, payment_included_acct_credit))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of projects with a specific resource type for magnet schools
@app.get("/v1/donor/percentage_projects_by_resource_type", operation_id="get_percentage_projects_by_resource_type", summary="Retrieves the percentage of projects with a specified resource type that are associated with magnet schools. The operation calculates this percentage by summing the number of projects with the given resource type and dividing it by the total number of projects in magnet schools.")
async def get_percentage_projects_by_resource_type(project_resource_type: str = Query(..., description="Project resource type"), school_magnet: str = Query(..., description="Magnet status of the school")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.project_resource_type = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.projectid) FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.school_magnet = ?", (project_resource_type, school_magnet))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of school IDs based on school magnet status and vendor name
@app.get("/v1/donor/count_school_ids_by_magnet_vendor", operation_id="get_count_school_ids_by_magnet_vendor", summary="Retrieves the total number of schools that match the specified magnet status and vendor name. This operation is useful for understanding the distribution of schools based on their magnet status and the vendor providing resources.")
async def get_count_school_ids_by_magnet_vendor(school_magnet: str = Query(..., description="School magnet status (e.g., 't' for true)"), vendor_name: str = Query(..., description="Vendor name")):
    cursor.execute("SELECT COUNT(T2.schoolid) FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.school_magnet = ? AND T1.vendor_name = ?", (school_magnet, vendor_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of school IDs based on poverty level and donor account ID
@app.get("/v1/donor/count_school_ids_by_poverty_donor", operation_id="get_count_school_ids_by_poverty_donor", summary="Retrieves the total number of schools, categorized by a specified poverty level, that have received donations from a particular donor account. The poverty level and donor account ID are provided as input parameters.")
async def get_count_school_ids_by_poverty_donor(poverty_level: str = Query(..., description="Poverty level (e.g., 'highest poverty')"), donor_acctid: str = Query(..., description="Donor account ID")):
    cursor.execute("SELECT COUNT(T1.schoolid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.poverty_level = ? AND T2.donor_acctid = ?", (poverty_level, donor_acctid))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the short description of essays based on school ID
@app.get("/v1/donor/get_short_description_by_schoolid", operation_id="get_short_description_by_schoolid", summary="Retrieves the concise summaries of essays associated with a specific school. The operation filters essays based on the provided school ID and returns their respective short descriptions.")
async def get_short_description_by_schoolid(schoolid: str = Query(..., description="School ID")):
    cursor.execute("SELECT T2.short_description FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T1.schoolid = ?", (schoolid,))
    result = cursor.fetchall()
    if not result:
        return {"short_description": []}
    return {"short_description": [row[0] for row in result]}

# Endpoint to get the school city based on essay title
@app.get("/v1/donor/get_school_city_by_essay_title", operation_id="get_school_city_by_essay_title", summary="Retrieves the city of the school associated with the essay that matches the provided title. The operation searches for essays with a title that matches the input parameter and returns the city of the school linked to the essay.")
async def get_school_city_by_essay_title(title: str = Query(..., description="Essay title")):
    cursor.execute("SELECT T1.school_city FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T2.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"school_city": []}
    return {"school_city": [row[0] for row in result]}

# Endpoint to get the payment method based on teacher account ID
@app.get("/v1/donor/get_payment_method_by_teacher_acctid", operation_id="get_payment_method_by_teacher_acctid", summary="Retrieves the payment method associated with a specific teacher account. The operation uses the provided teacher account ID to search for projects linked to that account and subsequently identifies the payment method used for donations made to those projects.")
async def get_payment_method_by_teacher_acctid(teacher_acctid: str = Query(..., description="Teacher account ID")):
    cursor.execute("SELECT T2.payment_method FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.teacher_acctid = ?", (teacher_acctid,))
    result = cursor.fetchall()
    if not result:
        return {"payment_method": []}
    return {"payment_method": [row[0] for row in result]}

# Endpoint to get the donation total based on teacher account ID
@app.get("/v1/donor/get_donation_total_by_teacher_acctid", operation_id="get_donation_total_by_teacher_acctid", summary="Retrieves the total donation amount associated with a specific teacher account. The operation calculates the sum of all donations linked to the teacher's projects, providing a comprehensive view of the teacher's fundraising efforts.")
async def get_donation_total_by_teacher_acctid(teacher_acctid: str = Query(..., description="Teacher account ID")):
    cursor.execute("SELECT T2.donation_total FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.teacher_acctid = ?", (teacher_acctid,))
    result = cursor.fetchall()
    if not result:
        return {"donation_total": []}
    return {"donation_total": [row[0] for row in result]}

# Endpoint to get the teacher account status based on school ID
@app.get("/v1/donor/get_teacher_acct_status_by_schoolid", operation_id="get_teacher_acct_status_by_schoolid", summary="Retrieves the teacher account status associated with a specific school. The operation uses the provided school ID to look up the corresponding teacher account status from the projects and donations tables.")
async def get_teacher_acct_status_by_schoolid(schoolid: str = Query(..., description="School ID")):
    cursor.execute("SELECT T2.is_teacher_acct FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.schoolid = ?", (schoolid,))
    result = cursor.fetchall()
    if not result:
        return {"is_teacher_acct": []}
    return {"is_teacher_acct": [row[0] for row in result]}

# Endpoint to get the percentage of projects in a specific city based on teacher account status
@app.get("/v1/donor/get_percentage_projects_by_city_teacher_acct", operation_id="get_percentage_projects_by_city_teacher_acct", summary="Retrieves the percentage of projects in a specified city that are associated with a teacher account. The calculation is based on the total number of projects in the city and the count of projects linked to a teacher account.")
async def get_percentage_projects_by_city_teacher_acct(school_city: str = Query(..., description="School city"), is_teacher_acct: str = Query(..., description="Teacher account status (e.g., 't' for true)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.school_city LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.teacher_acctid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.is_teacher_acct = ?", (school_city, is_teacher_acct))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of projects reaching more than a specified number of students based on teacher account status
@app.get("/v1/donor/get_percentage_projects_by_students_reached_teacher_acct", operation_id="get_percentage_projects_by_students_reached_teacher_acct", summary="Retrieves the percentage of projects that have reached more than a specified number of students, based on the status of the teacher account associated with the donations. The calculation is performed by summing the projects that surpass the specified number of students reached and dividing it by the total count of projects. The teacher account status is used as a filter for the donations.")
async def get_percentage_projects_by_students_reached_teacher_acct(students_reached: int = Query(..., description="Number of students reached"), is_teacher_acct: str = Query(..., description="Teacher account status (e.g., 't' for true)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.students_reached > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.projectid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.is_teacher_acct = ?", (students_reached, is_teacher_acct))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of school IDs based on school city and metro area
@app.get("/v1/donor/count_school_ids_by_city_metro", operation_id="get_count_school_ids_by_city_metro", summary="Retrieves the total number of unique school IDs located in a specific city and metro area. The metro area can be urban, suburban, or rural.")
async def get_count_school_ids_by_city_metro(school_city: str = Query(..., description="School city"), school_metro: str = Query(..., description="School metro area (e.g., 'suburban')")):
    cursor.execute("SELECT COUNT(schoolid) FROM projects WHERE school_city = ? AND school_metro = ?", (school_city, school_metro))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of year-round schools in a specific city
@app.get("/v1/donor/count_year_round_schools", operation_id="get_count_year_round_schools", summary="Retrieves the total number of year-round schools in a specified city. The operation filters schools based on their city and year-round status, providing a count of the matching schools.")
async def get_count_year_round_schools(school_city: str = Query(..., description="City of the school"), school_year_round: str = Query(..., description="Year-round status of the school (e.g., 't' for true)")):
    cursor.execute("SELECT COUNT(school_year_round) FROM projects WHERE school_city = ? AND school_year_round = ?", (school_city, school_year_round))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of magnet schools in a specific county
@app.get("/v1/donor/count_magnet_schools", operation_id="get_count_magnet_schools", summary="Retrieves the total number of magnet schools in a specified county. The operation considers the magnet status of each school to determine the count. The county is identified by a unique parameter, and the magnet status is indicated by a boolean value.")
async def get_count_magnet_schools(school_county: str = Query(..., description="County of the school"), school_magnet: str = Query(..., description="Magnet status of the school (e.g., 't' for true)")):
    cursor.execute("SELECT COUNT(schoolid) FROM projects WHERE school_county = ? AND school_magnet = ?", (school_county, school_magnet))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of teachers in a specific county focusing on a particular area
@app.get("/v1/donor/count_teachers_focus_area", operation_id="get_count_teachers_focus_area", summary="Retrieves the total number of teachers in a specified county who are focusing on a particular area of study. The operation considers the county of the school and the primary focus area of the project to determine the count.")
async def get_count_teachers_focus_area(school_county: str = Query(..., description="County of the school"), primary_focus_area: str = Query(..., description="Primary focus area of the project (e.g., 'Math & Science')")):
    cursor.execute("SELECT COUNT(teacher_acctid) FROM projects WHERE school_county = ? AND primary_focus_area = ?", (school_county, primary_focus_area))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of teachers with a specific prefix focusing on a particular subject
@app.get("/v1/donor/count_teachers_focus_subject", operation_id="get_count_teachers_focus_subject", summary="Retrieves the number of teachers with a given prefix who are primarily focused on a specific subject. The prefix and subject are provided as input parameters.")
async def get_count_teachers_focus_subject(teacher_prefix: str = Query(..., description="Prefix of the teacher (e.g., 'Mr.')"), primary_focus_subject: str = Query(..., description="Primary focus subject of the project (e.g., 'Literature & Writing')")):
    cursor.execute("SELECT COUNT(teacher_acctid) FROM projects WHERE teacher_prefix = ? AND primary_focus_subject = ?", (teacher_prefix, primary_focus_subject))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of projects in a specific school district for a particular grade level
@app.get("/v1/donor/count_projects_grade_level", operation_id="get_count_projects_grade_level", summary="Retrieves the total number of projects in a specified school district for a given grade level. The response is based on the provided school district and grade level parameters.")
async def get_count_projects_grade_level(school_district: str = Query(..., description="School district"), grade_level: str = Query(..., description="Grade level (e.g., 'Grades 3-5')")):
    cursor.execute("SELECT COUNT(projectid) FROM projects WHERE school_district = ? AND grade_level = ?", (school_district, grade_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the teacher account status for essays with titles matching a specific pattern
@app.get("/v1/donor/get_teacher_acct_status_by_essay_title", operation_id="get_teacher_acct_status_by_essay_title", summary="Get the teacher account status for essays with titles matching a specific pattern")
async def get_teacher_acct_status_by_essay_title(title_pattern: str = Query(..., description="Pattern to match the essay title (use % for wildcard)")):
    cursor.execute("SELECT T2.is_teacher_acct FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"is_teacher_acct": []}
    return {"is_teacher_acct": result[0]}

# Endpoint to get the total donation amount for essays with titles matching a specific pattern
@app.get("/v1/donor/get_total_donation_by_essay_title", operation_id="get_total_donation_by_essay_title", summary="Retrieves the total amount of donations for essays with titles that match the provided pattern. The pattern can include wildcards to broaden the search. The result is the sum of all donations associated with the matching essays.")
async def get_total_donation_by_essay_title(title_pattern: str = Query(..., description="Pattern to match the essay title (use % for wildcard)")):
    cursor.execute("SELECT SUM(T2.donation_total) FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"total_donation": []}
    return {"total_donation": result[0]}

# Endpoint to get the total donation amount including optional support for essays with titles matching a specific pattern
@app.get("/v1/donor/get_total_donation_with_support_by_essay_title", operation_id="get_total_donation_with_support_by_essay_title", summary="Retrieves the total amount of donations, including optional support, for essays with titles that match the provided pattern. The pattern should be specified using the '%' wildcard for partial matches.")
async def get_total_donation_with_support_by_essay_title(title_pattern: str = Query(..., description="Pattern to match the essay title (use % for wildcard)")):
    cursor.execute("SELECT SUM(T2.donation_to_project) + SUM(T2.donation_optional_support) FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"total_donation_with_support": []}
    return {"total_donation_with_support": result[0]}

# Endpoint to get donation optional support based on essay title
@app.get("/v1/donor/donation_optional_support_by_title", operation_id="get_donation_optional_support", summary="Retrieves the optional donation support for essays that match the provided title. The title parameter is used to filter the essays and return the corresponding donation support.")
async def get_donation_optional_support(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T2.donation_optional_support FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"donation_optional_support": []}
    return {"donation_optional_support": [row[0] for row in result]}

# Endpoint to get short descriptions of essays based on donation timestamp
@app.get("/v1/donor/short_description_by_donation_timestamp", operation_id="get_short_description", summary="Retrieves the short descriptions of essays associated with a specific donation timestamp. The donation timestamp, provided in 'YYYY-MM-DD HH:MM:SS' format, is used to filter the essays based on their corresponding donations.")
async def get_short_description(donation_timestamp: str = Query(..., description="Donation timestamp in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT T1.short_description FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.donation_timestamp LIKE ?", (donation_timestamp,))
    result = cursor.fetchall()
    if not result:
        return {"short_description": []}
    return {"short_description": [row[0] for row in result]}

# Endpoint to get donation included optional support based on essay title
@app.get("/v1/donor/donation_included_optional_support_by_title", operation_id="get_donation_included_optional_support", summary="Retrieves the optional donation support status for a specific essay, based on its title. The response indicates whether the essay's associated project has received optional donation support.")
async def get_donation_included_optional_support(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T2.donation_included_optional_support FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"donation_included_optional_support": []}
    return {"donation_included_optional_support": [row[0] for row in result]}

# Endpoint to get teacher account IDs based on donation timestamp
@app.get("/v1/donor/teacher_acctid_by_donation_timestamp", operation_id="get_teacher_acctid", summary="Retrieve the account IDs of teachers associated with essays that have a specific donation timestamp. The donation timestamp is provided in the 'YYYY-MM-DD HH:MM:SS' format.")
async def get_teacher_acctid(donation_timestamp: str = Query(..., description="Donation timestamp in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT T1.teacher_acctid FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.donation_timestamp LIKE ?", (donation_timestamp,))
    result = cursor.fetchall()
    if not result:
        return {"teacher_acctid": []}
    return {"teacher_acctid": [row[0] for row in result]}

# Endpoint to get school IDs based on essay title
@app.get("/v1/donor/schoolid_by_title", operation_id="get_schoolid", summary="Retrieves the school IDs associated with projects containing essays that match the specified title. The title parameter is used to filter the essays and identify the corresponding projects and their respective school IDs.")
async def get_schoolid(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T2.schoolid FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"schoolid": []}
    return {"schoolid": [row[0] for row in result]}

# Endpoint to get essay titles with the maximum donation total
@app.get("/v1/donor/titles_with_max_donation_total", operation_id="get_titles_with_max_donation_total", summary="Retrieves the titles of essays that have received the highest total donation amount. This operation identifies the maximum donation total across all essays and returns the titles of essays that have received this amount.")
async def get_titles_with_max_donation_total():
    cursor.execute("SELECT T1.title FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.donation_total = ( SELECT MAX(donation_total) FROM donations )")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the percentage of donation optional support based on essay title
@app.get("/v1/donor/percentage_donation_optional_support_by_title", operation_id="get_percentage_donation_optional_support", summary="Retrieves the percentage of optional donation support for essays with a specified title. This operation calculates the ratio of the total optional donation support to the overall donation total for essays that match the provided title.")
async def get_percentage_donation_optional_support(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT CAST(SUM(T2.donation_optional_support) AS REAL) * 100 / SUM(T2.donation_total) FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of donations made by a specific payment method based on essay title
@app.get("/v1/donor/percentage_donations_by_payment_method_and_title", operation_id="get_percentage_donations_by_payment_method", summary="Retrieves the percentage of donations made using a specified payment method for essays with a given title. This operation calculates the proportion of donations made via the input payment method relative to the total number of donations for the specified essay title.")
async def get_percentage_donations_by_payment_method(payment_method: str = Query(..., description="Payment method"), title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.payment_method LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(donationid) FROM essays AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (payment_method, title))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the short description of an essay based on its title
@app.get("/v1/donor/short_description_by_title", operation_id="get_short_description_by_title", summary="Retrieves the brief summary of an essay identified by its title.")
async def get_short_description_by_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT short_description FROM essays WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"short_description": []}
    return {"short_description": result[0]}

# Endpoint to get the need statement of an essay based on its title
@app.get("/v1/donor/need_statement_by_title", operation_id="get_need_statement_by_title", summary="Retrieves the need statement of an essay identified by its title. The title is provided as an input parameter.")
async def get_need_statement_by_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT need_statement FROM essays WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"need_statement": []}
    return {"need_statement": result[0]}

# Endpoint to get the school county with the highest count of a specific poverty level in a given state
@app.get("/v1/donor/school_county_by_poverty_level_state", operation_id="get_school_county", summary="Retrieves the county with the highest number of schools in a specific poverty level within a given state. The operation requires the poverty level and the state as input parameters.")
async def get_school_county(poverty_level: str = Query(..., description="Poverty level"), school_state: str = Query(..., description="School state")):
    cursor.execute("SELECT school_county FROM projects WHERE poverty_level = ? AND school_state = ? GROUP BY school_state ORDER BY COUNT(poverty_level) DESC LIMIT 1", (poverty_level, school_state))
    result = cursor.fetchone()
    if not result:
        return {"school_county": []}
    return {"school_county": result[0]}

# Endpoint to get school districts based on essay title
@app.get("/v1/donor/school_districts_by_essay_title", operation_id="get_school_districts", summary="Retrieves the school districts associated with projects containing an essay with a specified title. The title parameter is used to filter the results.")
async def get_school_districts(title: str = Query(..., description="Essay title")):
    cursor.execute("SELECT T1.school_district FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T2.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"school_districts": []}
    return {"school_districts": [row[0] for row in result]}

# Endpoint to get payment methods based on essay title
@app.get("/v1/donor/payment_methods_by_essay_title", operation_id="get_payment_methods", summary="Retrieves the payment methods associated with a specific essay title. This operation returns the payment methods used for donations made to projects linked to the given essay title. The input parameter is the title of the essay.")
async def get_payment_methods(title: str = Query(..., description="Essay title")):
    cursor.execute("SELECT T3.payment_method FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN donations AS T3 ON T2.projectid = T3.projectid WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"payment_methods": []}
    return {"payment_methods": [row[0] for row in result]}

# Endpoint to get the count of projects based on payment method and school district
@app.get("/v1/donor/project_count_by_payment_method_district", operation_id="get_project_count_by_payment_method", summary="Retrieves the total number of projects associated with a given payment method and school district. The response provides a quantitative overview of projects based on the specified payment method and school district.")
async def get_project_count_by_payment_method(payment_method: str = Query(..., description="Payment method"), school_district: str = Query(..., description="School district name")):
    cursor.execute("SELECT COUNT(T1.projectid) FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.payment_method = ? AND T2.school_district = ?", (payment_method, school_district))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get school districts based on vendor name
@app.get("/v1/donor/school_districts_by_vendor", operation_id="get_school_districts_by_vendor", summary="Retrieves a list of school districts associated with a specific vendor. The operation filters the resources based on the provided vendor name and returns the corresponding school districts.")
async def get_school_districts_by_vendor(vendor_name: str = Query(..., description="Vendor name")):
    cursor.execute("SELECT T2.school_district FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.vendor_name = ?", (vendor_name,))
    result = cursor.fetchall()
    if not result:
        return {"school_districts": []}
    return {"school_districts": [row[0] for row in result]}

# Endpoint to get school latitude and longitude based on item name and vendor name
@app.get("/v1/donor/school_location_by_item_vendor", operation_id="get_school_location", summary="Retrieves the geographical coordinates (latitude and longitude) of the school associated with a specific item and vendor. The item and vendor are identified by their respective names, which are provided as input parameters.")
async def get_school_location(item_name: str = Query(..., description="Item name"), vendor_name: str = Query(..., description="Vendor name")):
    cursor.execute("SELECT T2.school_latitude, T2.school_longitude FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.item_name = ? AND T1.vendor_name = ?", (item_name, vendor_name))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": [{"latitude": row[0], "longitude": row[1]} for row in result]}

# Endpoint to get the most common payment method in a specific school state
@app.get("/v1/donor/most_common_payment_method_by_state", operation_id="get_most_common_payment_method", summary="Retrieves the most frequently used payment method among donations made to projects in a specific school state. The operation groups donations by school state and identifies the payment method with the highest count. The result is limited to a single payment method.")
async def get_most_common_payment_method(school_state: str = Query(..., description="School state")):
    cursor.execute("SELECT T1.payment_method FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.school_state = ? GROUP BY T2.school_state ORDER BY COUNT(T1.payment_method) DESC LIMIT 1", (school_state,))
    result = cursor.fetchone()
    if not result:
        return {"payment_method": []}
    return {"payment_method": result[0]}

# Endpoint to get school latitude, longitude, and resource type based on essay title
@app.get("/v1/donor/school_info_by_essay_title", operation_id="get_school_info_by_essay_title", summary="Retrieves the geographical coordinates (latitude and longitude) and resource type of the school associated with a given essay title. The essay title is used to identify the corresponding project, which is then linked to the school's information.")
async def get_school_info_by_essay_title(essay_title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T2.school_latitude, T2.school_longitude, T2.resource_type FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (essay_title,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get donation messages based on school location and district
@app.get("/v1/donor/donation_messages_by_location_and_district", operation_id="get_donation_messages_by_location_and_district", summary="Retrieves donation messages for a specific school location and district. The operation uses the provided school latitude, longitude, and district to filter the donation messages associated with the corresponding school project.")
async def get_donation_messages_by_location_and_district(school_latitude: float = Query(..., description="Latitude of the school"), school_longitude: float = Query(..., description="Longitude of the school"), school_district: str = Query(..., description="School district")):
    cursor.execute("SELECT T1.donation_message FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.school_latitude = ? AND T2.school_longitude = ? AND T2.school_district = ?", (school_latitude, school_longitude, school_district))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get project posting dates based on essay title
@app.get("/v1/donor/project_posting_dates_by_essay_title", operation_id="get_project_posting_dates_by_essay_title", summary="Retrieves the dates when projects were posted that are associated with a specific essay title. The input parameter is the title of the essay, which is used to filter the projects and return their respective posting dates.")
async def get_project_posting_dates_by_essay_title(essay_title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T1.date_posted FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T2.title LIKE ?", (essay_title,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get donation messages based on essay title and payment method
@app.get("/v1/donor/donation_messages_by_essay_title_and_payment_method", operation_id="get_donation_messages_by_essay_title_and_payment_method", summary="Retrieves donation messages associated with a specific essay title and payment method. The operation filters donations based on the provided essay title and payment method, and returns the corresponding donation messages.")
async def get_donation_messages_by_essay_title_and_payment_method(essay_title: str = Query(..., description="Title of the essay"), payment_method: str = Query(..., description="Payment method")):
    cursor.execute("SELECT T3.donation_message FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN donations AS T3 ON T2.projectid = T3.projectid WHERE T1.title = ? AND T3.payment_method = ?", (essay_title, payment_method))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the most common project resource type in a school district
@app.get("/v1/donor/most_common_project_resource_type_by_district", operation_id="get_most_common_project_resource_type_by_district", summary="Retrieves the most frequently occurring project resource type within a specified school district. The operation identifies the resource type that appears most often in projects associated with the given school district.")
async def get_most_common_project_resource_type_by_district(school_district: str = Query(..., description="School district")):
    cursor.execute("SELECT T1.project_resource_type FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.school_district = ? GROUP BY T2.school_district ORDER BY COUNT(T1.project_resource_type) DESC LIMIT 1", (school_district,))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result[0]}

# Endpoint to get school cities based on school district and vendor name
@app.get("/v1/donor/school_cities_by_district_and_vendor", operation_id="get_school_cities_by_district_and_vendor", summary="Retrieves the cities of schools that belong to a specific school district and are associated with a particular vendor. The operation filters schools based on the provided school district and vendor name.")
async def get_school_cities_by_district_and_vendor(school_district: str = Query(..., description="School district"), vendor_name: str = Query(..., description="Vendor name")):
    cursor.execute("SELECT T2.school_city FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.school_district = ? AND T1.vendor_name = ?", (school_district, vendor_name))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get item price, school latitude, and longitude based on vendor name, resource type, and school district
@app.get("/v1/donor/item_price_and_location_by_vendor_resource_type_district", operation_id="get_item_price_and_location_by_vendor_resource_type_district", summary="Retrieves the total price of items and the geographical coordinates of the school based on the specified vendor, resource type, and school district. The total price is calculated by multiplying the unit price of each item by its quantity. The geographical coordinates are represented by latitude and longitude.")
async def get_item_price_and_location_by_vendor_resource_type_district(vendor_name: str = Query(..., description="Vendor name"), project_resource_type: str = Query(..., description="Project resource type"), school_district: str = Query(..., description="School district")):
    cursor.execute("SELECT T2.item_unit_price * T2.item_quantity price, T1.school_latitude, T1.school_longitude FROM projects AS T1 INNER JOIN resources AS T2 ON T1.projectid = T2.projectid WHERE T2.vendor_name = ? AND T2.project_resource_type = ? AND T1.school_district = ?", (vendor_name, project_resource_type, school_district))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get total donations, school city, and poverty level based on essay title and payment method
@app.get("/v1/donor/total_donations_by_essay_title_and_payment_method", operation_id="get_total_donations_by_essay_title_and_payment_method", summary="Retrieves the total donations, school city, and poverty level associated with a specific essay title and payment method. This operation aggregates donation data from the donations table, filtered by the provided essay title and payment method. The result includes the sum of donation totals, the city of the school, and the poverty level.")
async def get_total_donations_by_essay_title_and_payment_method(essay_title: str = Query(..., description="Title of the essay"), payment_method: str = Query(..., description="Payment method")):
    cursor.execute("SELECT SUM(T3.donation_total), school_city, poverty_level FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN donations AS T3 ON T2.projectid = T3.projectid WHERE T1.title = ? AND T3.payment_method = ?", (essay_title, payment_method))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of donations based on donor city and teacher account status
@app.get("/v1/donor/donation_count_by_city_and_teacher_status", operation_id="get_donation_count_by_city_and_teacher_status", summary="Retrieves the total number of donations made by donors from a specific city, filtered by the status of their account as a teacher. The response provides a quantitative measure of donation activity based on the given city and teacher account status.")
async def get_donation_count_by_city_and_teacher_status(donor_city: str = Query(..., description="City of the donor"), is_teacher_acct: str = Query(..., description="Teacher account status (e.g., 't' for true, 'f' for false)")):
    cursor.execute("SELECT COUNT(donationid) FROM donations WHERE donor_city = ? AND is_teacher_acct = ?", (donor_city, is_teacher_acct))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of schools based on teacher prefix and school county
@app.get("/v1/donor/school_count_by_teacher_prefix_and_county", operation_id="get_school_count_by_teacher_prefix_and_county", summary="Retrieves the total number of schools that match a specified teacher prefix and school county. The teacher prefix is a title or honorific (e.g., 'Dr.'), and the school county is the geographical location of the school.")
async def get_school_count_by_teacher_prefix_and_county(teacher_prefix: str = Query(..., description="Teacher prefix (e.g., 'Dr.')"), school_county: str = Query(..., description="School county")):
    cursor.execute("SELECT COUNT(schoolid) FROM projects WHERE teacher_prefix = ? AND school_county = ?", (teacher_prefix, school_county))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total donations based on payment method
@app.get("/v1/donor/total_donations_by_payment_method", operation_id="get_total_donations", summary="Retrieves the total amount of donations, including both donations to projects and optional support, for a specified payment method. The payment method is used to filter the donations and calculate the sum.")
async def get_total_donations(payment_method: str = Query(..., description="Payment method (e.g., 'amazon')")):
    cursor.execute("SELECT SUM(donation_to_project) + SUM(donation_optional_support) FROM donations WHERE payment_method = ?", (payment_method,))
    result = cursor.fetchone()
    if not result:
        return {"total_donations": []}
    return {"total_donations": result[0]}

# Endpoint to get vendor IDs based on project resource type and item unit price
@app.get("/v1/donor/vendor_ids_by_resource_type_and_price", operation_id="get_vendor_ids", summary="Retrieves the vendor IDs associated with a specific project resource type and a maximum item unit price. The operation filters resources based on the provided project resource type and item unit price, returning the corresponding vendor IDs.")
async def get_vendor_ids(project_resource_type: str = Query(..., description="Project resource type (e.g., 'Technology')"), item_unit_price: int = Query(..., description="Maximum item unit price")):
    cursor.execute("SELECT vendorid FROM resources WHERE project_resource_type = ? AND item_unit_price <= ?", (project_resource_type, item_unit_price))
    result = cursor.fetchall()
    if not result:
        return {"vendor_ids": []}
    return {"vendor_ids": [row[0] for row in result]}

# Endpoint to get school IDs based on school district and teaching fellow status
@app.get("/v1/donor/school_ids_by_district_and_fellow", operation_id="get_school_ids", summary="Retrieves a list of school IDs that belong to a specific school district and have a certain teaching fellow status. The operation filters schools based on the provided district and teaching fellow status, returning only the IDs of the matching schools.")
async def get_school_ids(school_district: str = Query(..., description="School district (e.g., 'Union Pub School District I-9')"), teacher_ny_teaching_fellow: str = Query(..., description="Teaching fellow status (e.g., 't')")):
    cursor.execute("SELECT schoolid FROM projects WHERE school_district = ? AND teacher_ny_teaching_fellow = ?", (school_district, teacher_ny_teaching_fellow))
    result = cursor.fetchall()
    if not result:
        return {"school_ids": []}
    return {"school_ids": [row[0] for row in result]}

# Endpoint to get school cities based on school metro and county
@app.get("/v1/donor/school_cities_by_metro_and_county", operation_id="get_school_cities", summary="Retrieves a list of school cities that match the specified school metro and county. The school metro and county are used as filters to narrow down the results. For example, you can find all school cities in the suburban area of Los Angeles.")
async def get_school_cities(school_metro: str = Query(..., description="School metro (e.g., 'suburban')"), school_county: str = Query(..., description="School county (e.g., 'Los Angeles')")):
    cursor.execute("SELECT school_city FROM projects WHERE school_metro = ? AND school_county = ?", (school_metro, school_county))
    result = cursor.fetchall()
    if not result:
        return {"school_cities": []}
    return {"school_cities": [row[0] for row in result]}

# Endpoint to get distinct vendor and project IDs based on project resource type
@app.get("/v1/donor/distinct_vendor_project_ids_by_resource_type", operation_id="get_distinct_vendor_project_ids", summary="Retrieve a unique set of vendor and project identifiers associated with a specified project resource type. This operation allows you to filter resources based on their type, providing a concise list of distinct vendor and project IDs that match the given resource type.")
async def get_distinct_vendor_project_ids(project_resource_type: str = Query(..., description="Project resource type (e.g., 'Books')")):
    cursor.execute("SELECT DISTINCT vendorid, projectid FROM resources WHERE project_resource_type = ?", (project_resource_type,))
    result = cursor.fetchall()
    if not result:
        return {"vendor_project_ids": []}
    return {"vendor_project_ids": [{"vendorid": row[0], "projectid": row[1]} for row in result]}

# Endpoint to get the percentage of donations including campaign gift cards based on payment method
@app.get("/v1/donor/percentage_donations_with_gift_cards", operation_id="get_percentage_donations_with_gift_cards", summary="Retrieves the percentage of donations that include campaign gift cards for a specified payment method. This operation calculates the proportion of donations that have a particular campaign gift card status, out of all donations made using the given payment method.")
async def get_percentage_donations_with_gift_cards(payment_included_campaign_gift_card: str = Query(..., description="Campaign gift card inclusion status (e.g., 't')"), payment_method: str = Query(..., description="Payment method (e.g., 'no_cash_received')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN payment_included_campaign_gift_card = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(donationid) FROM donations WHERE payment_method = ?", (payment_included_campaign_gift_card, payment_method))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of projects in a specific metro area based on school city
@app.get("/v1/donor/percentage_projects_in_metro", operation_id="get_percentage_projects_in_metro", summary="Retrieves the percentage of projects in a specific metro area (e.g., suburban) for a given school city (e.g., Santa Barbara). This operation calculates the proportion of projects in the specified metro area relative to the total number of projects in the provided school city.")
async def get_percentage_projects_in_metro(school_metro: str = Query(..., description="School metro (e.g., 'suburban')"), school_city: str = Query(..., description="School city (e.g., 'Santa Barbara')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN school_metro = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(projectid) FROM projects WHERE school_city = ?", (school_metro, school_city))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average item unit price per quantity for a given vendor
@app.get("/v1/donor/average_item_price_per_quantity", operation_id="get_average_item_price_per_quantity", summary="Retrieves the average unit price per quantity of items associated with a specific vendor. This operation calculates the average by dividing the total sum of item unit prices by the total sum of item quantities for the given vendor.")
async def get_average_item_price_per_quantity(vendor_name: str = Query(..., description="Name of the vendor")):
    cursor.execute("SELECT SUM(item_unit_price) / SUM(item_quantity) FROM resources WHERE vendor_name = ?", (vendor_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the count of school IDs based on payment inclusion and poverty level
@app.get("/v1/donor/count_school_ids_payment_poverty", operation_id="get_count_school_ids_payment_poverty", summary="Retrieves the count of unique school IDs that meet the specified payment inclusion status and poverty level criteria. This operation filters projects based on the provided payment inclusion status and poverty level, and then counts the number of unique school IDs associated with the filtered projects.")
async def get_count_school_ids_payment_poverty(payment_included_campaign_gift_card: str = Query(..., description="Payment inclusion status"), poverty_level: str = Query(..., description="Poverty level")):
    cursor.execute("SELECT COUNT(T1.schoolid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.payment_included_campaign_gift_card = ? AND T1.poverty_level = ?", (payment_included_campaign_gift_card, poverty_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top school city based on donation amount
@app.get("/v1/donor/top_school_city_by_donation_amount", operation_id="get_top_school_city_by_donation_amount", summary="Retrieves the city with the highest number of donations for a given donation amount. The city is determined by the school location associated with the projects that received the specified donation amount. The result is ordered by the count of unique school IDs in descending order and limited to the top city.")
async def get_top_school_city_by_donation_amount(dollar_amount: str = Query(..., description="Donation amount")):
    cursor.execute("SELECT T2.school_city FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.dollar_amount = ? GROUP BY T2.school_city ORDER BY COUNT(T2.schoolid) DESC LIMIT 1", (dollar_amount,))
    result = cursor.fetchone()
    if not result:
        return {"school_city": []}
    return {"school_city": result[0]}

# Endpoint to get essay titles based on school latitude and longitude
@app.get("/v1/donor/essay_titles_by_location", operation_id="get_essay_titles_by_location", summary="Retrieves a list of essay titles associated with a specific school location, identified by its latitude and longitude coordinates. The operation filters essays based on the provided school coordinates and returns the corresponding essay titles.")
async def get_essay_titles_by_location(school_latitude: int = Query(..., description="Latitude of the school"), school_longitude: int = Query(..., description="Longitude of the school")):
    cursor.execute("SELECT T1.title FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.school_latitude = ? AND T2.school_longitude = ?", (school_latitude, school_longitude))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get NY teaching fellows based on essay title
@app.get("/v1/donor/ny_teaching_fellows_by_essay_title", operation_id="get_ny_teaching_fellows_by_essay_title", summary="Retrieves a list of New York teaching fellows who have written essays with a title that matches the provided title. The title parameter is used to filter the essays and identify the corresponding teaching fellows.")
async def get_ny_teaching_fellows_by_essay_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T2.teacher_ny_teaching_fellow FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"teaching_fellows": []}
    return {"teaching_fellows": [row[0] for row in result]}

# Endpoint to get the top vendor based on primary focus area
@app.get("/v1/donor/top_vendor_by_primary_focus_area", operation_id="get_top_vendor_by_primary_focus_area", summary="Retrieves the vendor with the highest number of resources associated with the specified primary focus area. The primary focus area is used to filter the resources and projects, and the vendor with the most resources is returned.")
async def get_top_vendor_by_primary_focus_area(primary_focus_area: str = Query(..., description="Primary focus area")):
    cursor.execute("SELECT T1.vendor_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.primary_focus_area LIKE ? GROUP BY T1.vendor_name ORDER BY COUNT(T2.primary_focus_area) DESC LIMIT 1", (primary_focus_area,))
    result = cursor.fetchone()
    if not result:
        return {"vendor_name": []}
    return {"vendor_name": result[0]}

# Endpoint to get distinct vendor names based on grade level
@app.get("/v1/donor/distinct_vendor_names_by_grade_level", operation_id="get_distinct_vendor_names_by_grade_level", summary="Retrieves a unique list of vendor names associated with a specific grade level. The grade level is used to filter the resources and projects, and only the distinct vendor names are returned.")
async def get_distinct_vendor_names_by_grade_level(grade_level: str = Query(..., description="Grade level")):
    cursor.execute("SELECT DISTINCT T1.vendor_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.grade_level = ?", (grade_level,))
    result = cursor.fetchall()
    if not result:
        return {"vendor_names": []}
    return {"vendor_names": [row[0] for row in result]}

# Endpoint to get the count of distinct teacher account IDs based on teacher account status and school city
@app.get("/v1/donor/count_distinct_teacher_acctids", operation_id="get_count_distinct_teacher_acctids", summary="Retrieves the total number of unique teacher accounts based on their account status and the city of their school. This operation considers donations made to projects and filters results by the specified teacher account status and school city.")
async def get_count_distinct_teacher_acctids(is_teacher_acct: str = Query(..., description="Teacher account status"), school_city: str = Query(..., description="City of the school")):
    cursor.execute("SELECT COUNT(DISTINCT T2.teacher_acctid) FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.is_teacher_acct = ? AND T2.school_city = ?", (is_teacher_acct, school_city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct teacher account IDs based on school city and teacher account status
@app.get("/v1/donor/count_distinct_teacher_acctids_by_city", operation_id="get_count_distinct_teacher_acctids_by_city", summary="Retrieves the count of unique teacher accounts in a specific city, based on their account status. This operation considers the city of the school and the status of the teacher account to provide an accurate count.")
async def get_count_distinct_teacher_acctids_by_city(school_city: str = Query(..., description="City of the school"), is_teacher_acct: str = Query(..., description="Teacher account status")):
    cursor.execute("SELECT COUNT(DISTINCT T1.teacher_acctid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.school_city = ? AND is_teacher_acct = ?", (school_city, is_teacher_acct))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total dollar amount of donations for projects reaching more than a specified number of students and including optional support
@app.get("/v1/donor/total_dollar_amount_students_reached_optional_support", operation_id="get_total_dollar_amount", summary="Retrieves the total dollar amount of donations for projects that have reached more than a specified number of students and included optional support. The operation considers the minimum number of students reached and whether the donation included optional support as input parameters.")
async def get_total_dollar_amount(students_reached: int = Query(..., description="Minimum number of students reached"), donation_included_optional_support: str = Query(..., description="Whether the donation included optional support ('t' or 'f')")):
    cursor.execute("SELECT SUM(T2.dollar_amount) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.students_reached > ? AND T2.donation_included_optional_support = ?", (students_reached, donation_included_optional_support))
    result = cursor.fetchone()
    if not result:
        return {"total_dollar_amount": []}
    return {"total_dollar_amount": result[0]}

# Endpoint to get the total item quantity for projects in a specific metro area and school district
@app.get("/v1/donor/total_item_quantity_metro_district", operation_id="get_total_item_quantity", summary="Retrieves the total quantity of items donated to projects in a specific metro area and school district. The metro area and school district are used to filter the projects and calculate the sum of item quantities.")
async def get_total_item_quantity(school_metro: str = Query(..., description="Metro area of the school"), school_district: str = Query(..., description="School district")):
    cursor.execute("SELECT SUM(T1.item_quantity) FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.school_metro = ? AND T2.school_district = ?", (school_metro, school_district))
    result = cursor.fetchone()
    if not result:
        return {"total_item_quantity": []}
    return {"total_item_quantity": result[0]}

# Endpoint to get the average donation amount per project in a specific county
@app.get("/v1/donor/average_donation_amount_county", operation_id="get_average_donation_amount", summary="Retrieves the average total donation amount per project in a specified county. The calculation considers both optional support and direct project donations. The county is determined by the school's location.")
async def get_average_donation_amount(school_county: str = Query(..., description="County of the school")):
    cursor.execute("SELECT SUM(T2.donation_optional_support + T2.donation_to_project) / COUNT(donationid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T1.school_county = ?", (school_county,))
    result = cursor.fetchone()
    if not result:
        return {"average_donation_amount": []}
    return {"average_donation_amount": result[0]}

# Endpoint to get the percentage of projects with a specific title pattern
@app.get("/v1/donor/percentage_projects_title_pattern", operation_id="get_percentage_projects_title_pattern", summary="Retrieves the percentage of projects with titles that match a specified pattern. This operation calculates the proportion of projects, from the total number of projects, that have titles containing the provided pattern. The pattern is case-insensitive and can include wildcard characters.")
async def get_percentage_projects_title_pattern(title_pattern: str = Query(..., description="Pattern to match in the title")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.title LIKE ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.school_county) FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid", (title_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average item quantity for donations made for an honoree
@app.get("/v1/donor/average_item_quantity_honoree", operation_id="get_average_item_quantity_honoree", summary="Retrieves the average quantity of items donated for a specific honoree. The honoree status, indicated by 'for_honoree', is used to filter the donations and calculate the average item quantity.")
async def get_average_item_quantity_honoree(for_honoree: str = Query(..., description="Whether the donation is for an honoree ('t' or 'f')")):
    cursor.execute("SELECT AVG(T1.item_quantity) FROM resources AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.for_honoree = ?", (for_honoree,))
    result = cursor.fetchone()
    if not result:
        return {"average_item_quantity": []}
    return {"average_item_quantity": result[0]}

# Endpoint to get the most expensive item name for projects with a specific primary focus subject
@app.get("/v1/donor/most_expensive_item_primary_focus", operation_id="get_most_expensive_item", summary="Retrieves the name of the most expensive resource item associated with projects that have a specified primary focus subject. The primary focus subject is a parameter that filters the projects considered for this query.")
async def get_most_expensive_item(primary_focus_subject: str = Query(..., description="Primary focus subject of the project")):
    cursor.execute("SELECT T1.item_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T2.primary_focus_subject = ? ORDER BY T1.item_unit_price DESC LIMIT 1", (primary_focus_subject,))
    result = cursor.fetchone()
    if not result:
        return {"item_name": []}
    return {"item_name": result[0]}

# Endpoint to get the total donation amount for projects with a specific essay title
@app.get("/v1/donor/total_donation_amount_essay_title", operation_id="get_total_donation_amount_essay_title", summary="Retrieves the cumulative donation amount for projects associated with a specific essay title. The provided essay title is used to identify the relevant projects and sum up their respective donation totals.")
async def get_total_donation_amount_essay_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT SUM(T3.donation_total) FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN donations AS T3 ON T2.projectid = T3.projectid WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"total_donation_amount": []}
    return {"total_donation_amount": result[0]}

# Endpoint to get distinct poverty levels for projects with a specific donor zip code
@app.get("/v1/donor/distinct_poverty_levels_donor_zip", operation_id="get_distinct_poverty_levels", summary="Retrieve the unique poverty levels associated with projects funded by donors residing in a specific zip code. This operation provides a comprehensive view of the poverty levels impacted by donations from a particular area, aiding in understanding the geographical distribution of donor impact.")
async def get_distinct_poverty_levels(donor_zip: int = Query(..., description="Zip code of the donor")):
    cursor.execute("SELECT DISTINCT T2.poverty_level FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.donor_zip = ?", (donor_zip,))
    result = cursor.fetchall()
    if not result:
        return {"poverty_levels": []}
    return {"poverty_levels": [row[0] for row in result]}

# Endpoint to get vendor names based on essay title
@app.get("/v1/donor/vendor_names_by_essay_title", operation_id="get_vendor_names_by_essay_title", summary="Retrieves the names of vendors associated with a specific essay, based on the essay's title. The provided essay title is used to search for corresponding projects and resources, ultimately returning the vendor names linked to those projects.")
async def get_vendor_names_by_essay_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T3.vendor_name FROM essays AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid INNER JOIN resources AS T3 ON T2.projectid = T3.projectid WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"vendor_names": []}
    return {"vendor_names": [row[0] for row in result]}

# Endpoint to get school zip and item name based on vendor name
@app.get("/v1/donor/school_zip_item_name_by_vendor", operation_id="get_school_zip_item_name_by_vendor", summary="Retrieves the school zip code and item name associated with the specified vendor. This operation returns a list of school zip codes and corresponding item names for the given vendor, providing a comprehensive view of the vendor's distribution across schools and items.")
async def get_school_zip_item_name_by_vendor(vendor_name: str = Query(..., description="Name of the vendor")):
    cursor.execute("SELECT T2.school_zip, T1.item_name FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.vendor_name = ?", (vendor_name,))
    result = cursor.fetchall()
    if not result:
        return {"school_zip_item_name": []}
    return {"school_zip_item_name": [{"school_zip": row[0], "item_name": row[1]} for row in result]}

# Endpoint to get school coordinates based on essay title
@app.get("/v1/donor/school_coordinates_by_essay_title", operation_id="get_school_coordinates_by_essay_title", summary="Retrieves the geographical coordinates of the school associated with a specific essay. The essay is identified by its title, and the response includes the longitude and latitude of the school.")
async def get_school_coordinates_by_essay_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T1.school_longitude, T1.school_latitude FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T2.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"school_coordinates": []}
    return {"school_coordinates": [{"longitude": row[0], "latitude": row[1]} for row in result]}

# Endpoint to get distinct primary focus subjects based on campaign gift card inclusion
@app.get("/v1/donor/primary_focus_subjects_by_gift_card", operation_id="get_primary_focus_subjects_by_gift_card", summary="Retrieve a unique set of primary focus subjects associated with donations that either included or did not include a campaign gift card. The operation filters donations based on the provided parameter, which indicates whether a campaign gift card was part of the payment.")
async def get_primary_focus_subjects_by_gift_card(payment_included_campaign_gift_card: str = Query(..., description="Whether the payment included a campaign gift card ('t' for true, 'f' for false)")):
    cursor.execute("SELECT DISTINCT T2.primary_focus_subject FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.payment_included_campaign_gift_card = ?", (payment_included_campaign_gift_card,))
    result = cursor.fetchall()
    if not result:
        return {"primary_focus_subjects": []}
    return {"primary_focus_subjects": [row[0] for row in result]}

# Endpoint to get the most expensive item based on resource type
@app.get("/v1/donor/most_expensive_item_by_resource_type", operation_id="get_most_expensive_item_by_resource_type", summary="Retrieves the most expensive item associated with a specific resource type. The item is identified by its name and a brief description. The resource type is used to filter the results. The item with the highest unit price is returned.")
async def get_most_expensive_item_by_resource_type(project_resource_type: str = Query(..., description="Type of the project resource")):
    cursor.execute("SELECT T1.item_name, T2.short_description FROM resources AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T1.project_resource_type = ? ORDER BY T1.item_unit_price DESC LIMIT 1", (project_resource_type,))
    result = cursor.fetchone()
    if not result:
        return {"item_details": []}
    return {"item_details": {"item_name": result[0], "short_description": result[1]}}

# Endpoint to get grade levels based on essay title
@app.get("/v1/donor/grade_levels_by_essay_title", operation_id="get_grade_levels_by_essay_title", summary="Retrieves the grade levels associated with projects containing an essay that matches the provided title. The title parameter is used to filter the results.")
async def get_grade_levels_by_essay_title(title: str = Query(..., description="Title of the essay")):
    cursor.execute("SELECT T1.grade_level FROM projects AS T1 INNER JOIN essays AS T2 ON T1.projectid = T2.projectid WHERE T2.title LIKE ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"grade_levels": []}
    return {"grade_levels": [row[0] for row in result]}

# Endpoint to get the total number of students reached based on donor zip
@app.get("/v1/donor/total_students_reached_by_donor_zip", operation_id="get_total_students_reached_by_donor_zip", summary="Retrieves the cumulative number of students impacted by donations from a specific zip code. The donor's zip code is used to calculate the total students reached across all projects supported by donors from that area.")
async def get_total_students_reached_by_donor_zip(donor_zip: int = Query(..., description="Zip code of the donor")):
    cursor.execute("SELECT SUM(T2.students_reached) FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.donor_zip = ?", (donor_zip,))
    result = cursor.fetchone()
    if not result:
        return {"total_students_reached": []}
    return {"total_students_reached": result[0]}

# Endpoint to get the percentage of donations via giving page and the top primary focus area
@app.get("/v1/donor/donation_percentage_and_top_focus_area", operation_id="get_donation_percentage_and_top_focus_area", summary="Retrieves the percentage of donations made through the giving page and the primary focus area with the highest total donation amount. The input parameter specifies whether to consider donations made via the giving page.")
async def get_donation_percentage_and_top_focus_area(via_giving_page: str = Query(..., description="Whether the donation was made via the giving page ('t' for true, 'f' for false)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.via_giving_page = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(donation_total), ( SELECT T2.primary_focus_area FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.via_giving_page = ? GROUP BY T2.primary_focus_area ORDER BY SUM(T1.donation_total) DESC LIMIT 1 ) result FROM donations AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid", (via_giving_page, via_giving_page))
    result = cursor.fetchone()
    if not result:
        return {"donation_percentage": [], "top_focus_area": []}
    return {"donation_percentage": result[0], "top_focus_area": result[1]}

# Endpoint to get the percentage of projects with a specific vendor and the distinct posting dates for those projects
@app.get("/v1/donor/vendor_project_percentage_and_posting_dates", operation_id="get_vendor_project_percentage_and_posting_dates", summary="Get the percentage of projects with a specific vendor and the distinct posting dates for those projects based on the resource type")
async def get_vendor_project_percentage_and_posting_dates(vendor_name: str = Query(..., description="Name of the vendor"), project_resource_type: str = Query(..., description="Type of the project resource")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.vendor_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.projectid) FROM resources AS T1 INNER JOIN projects AS T2 ON T1.projectid = T2.projectid WHERE T1.project_resource_type = ? UNION ALL SELECT DISTINCT T1.date_posted FROM projects AS T1 INNER JOIN resources AS T2 ON T1.projectid = T2.projectid WHERE T2.vendor_name = ? AND T2.project_resource_type = ?", (vendor_name, project_resource_type, vendor_name, project_resource_type))
    result = cursor.fetchall()
    if not result:
        return {"vendor_project_percentage": [], "posting_dates": []}
    return {"vendor_project_percentage": result[0][0], "posting_dates": [row[0] for row in result[1:]]}

api_calls = [
    "/v1/donor/sum_donation_total_by_year?year=2012",
    "/v1/donor/highest_donation",
    "/v1/donor/sum_donation_total_by_criteria?year=2011&via_giving_page=t&for_honoree=t",
    "/v1/donor/donation_ratio_non_teacher?is_teacher_acct=f",
    "/v1/donor/essay_titles_by_subject?primary_focus_subject=Literacy",
    "/v1/donor/essay_titles_by_poverty_level?poverty_level=highest",
    "/v1/donor/primary_focus_subject_by_essay_title?essay_title=Toot%20Your%20Flute!",
    "/v1/donor/essay_titles_by_donation_message?donation_message=Donation%20on%20behalf%20of%20Matt%20Carpenter%20because%20I'm%20a%20strong%20believer%20in%20education.",
    "/v1/donor/essay_titles_and_price_by_students_reached?students_reached=320",
    "/v1/donor/donation_messages_and_essay_titles_by_city?donor_city=Texas%20City",
    "/v1/donor/resource_details_by_essay_title?essay_title=Lights%2C%20Camera%2C%20Action%21",
    "/v1/donor/total_donation_by_essay_title?essay_title=Whistle%20While%20We%20Work%21",
    "/v1/donor/essay_details_by_teacher_status?teacher_ny_teaching_fellow=t",
    "/v1/donor/project_titles_prices_by_students_poverty?students_reached=600&poverty_level=moderate%20poverty",
    "/v1/donor/percentage_donations_to_rural_schools?school_metro=rural",
    "/v1/donor/top_project_by_total_price",
    "/v1/donor/count_projects_by_district_poverty?school_district=West%20New%20York%20School%20District&poverty_level=highest%20poverty",
    "/v1/donor/count_donations_by_teacher_state?is_teacher_acct=t&donor_state=CO",
    "/v1/donor/top_project_by_total_price_including_support",
    "/v1/donor/distinct_donor_states_by_criteria?for_honoree=t&payment_included_campaign_gift_card=t&payment_method=paypal",
    "/v1/donor/highest_optional_support_price_difference",
    "/v1/donor/distinct_item_details?projectid=d6ef27c07c30c81f0c16c32b6acfa2ff",
    "/v1/donor/total_price_including_optional_support?title=Recording%20Rockin'%20Readers",
    "/v1/donor/school_coordinates?title=Smile%20for%20the%20Camera!!!",
    "/v1/donor/highest_donation_with_essay_title",
    "/v1/donor/most_common_project_resource_type",
    "/v1/donor/earliest_posted_project_item",
    "/v1/donor/distinct_vendor_names?title=iMath",
    "/v1/donor/highest_item_quantity_project",
    "/v1/donor/average_donation_by_essay_title?title=Recording%20Rockin'%20Readers",
    "/v1/donor/top_donation_by_essay_title",
    "/v1/donor/top_donations_with_optional_support",
    "/v1/donor/most_common_item_by_school_city?school_city=Brooklyn",
    "/v1/donor/count_schools_by_resource_type_and_metro?resource_type=Books&school_metro=urban",
    "/v1/donor/school_with_most_resources_from_vendor?vendor_name=Amazon",
    "/v1/donor/count_schools_by_donor_city_excluding_school_city?donor_city=Los%20Angeles&school_city=Los%20Angeles",
    "/v1/donor/teacher_prefixes_by_essay_title?title=Reading%20About%20Other%20Cultures",
    "/v1/donor/students_reached_by_essay_title?title=Fit%20Firsties!",
    "/v1/donor/essay_titles_and_donor_ids_by_school_city?school_city=Chicago",
    "/v1/donor/essay_titles_by_teacher_prefix_and_students_reached?teacher_prefix=Dr.&min_students_reached=300",
    "/v1/donor/essay_titles_highest_total_item_cost",
    "/v1/donor/donation_count_by_teacher_account_and_city?is_teacher_acct=t&donor_city=New%20York",
    "/v1/donor/donation_count_by_honoree_and_state?for_honoree=t&donor_state=NJ",
    "/v1/donor/donation_message_by_id?donationid=a84dace1ff716f6f0c7af8ef9090a5d5",
    "/v1/donor/distinct_project_resource_types_by_vendor?vendor_name=Lakeshore%20Learning%20Materials",
    "/v1/donor/item_name_highest_quantity_by_vendor?vendor_name=Lakeshore%20Learning%20Materials",
    "/v1/donor/project_count_by_teacher_fellow_and_city?teacher_ny_teaching_fellow=t&donor_city=New%20York",
    "/v1/donor/project_count_by_vendor_and_teacher_prefix?vendor_name=Lakeshore%20Learning%20Materials&teacher_prefix=Dr.",
    "/v1/donor/vendor_names_by_poverty_level?poverty_level=highest%20poverty",
    "/v1/donor/school_id_by_vendor_name?vendor_name=Lakeshore%20Learning%20Materials",
    "/v1/donor/vendor_name_by_price_difference",
    "/v1/donor/total_donations_by_school_city?school_city=Brooklyn",
    "/v1/donor/donation_messages_by_school_city_and_magnet?school_city=Brooklyn&school_magnet=t",
    "/v1/donor/count_projects_by_payment_and_year_round?payment_included_acct_credit=t&school_year_round=t",
    "/v1/donor/total_dollar_amount_by_primary_focus_area?primary_focus_area=Literacy%20%26%20Language",
    "/v1/donor/donor_account_id_by_primary_focus_area?primary_focus_area=Literacy%20%26%20Language",
    "/v1/donor/item_name_by_primary_focus_area_and_resource_type?primary_focus_area=Literacy%20%26%20Language&project_resource_type=Supplies",
    "/v1/donor/item_name_by_primary_focus_area?primary_focus_area=Literacy%20%26%20Language",
    "/v1/donor/average_donation_total_by_city?school_city=Brooklyn",
    "/v1/donor/distinct_donor_cities?donor_acctid=22cbc920c9b5fa08dfb331422f5926b5",
    "/v1/donor/distinct_teacher_account_statuses?donor_acctid=22cbc920c9b5fa08dfb331422f5926b5",
    "/v1/donor/check_teacher_prefix?teacher_prefix=Dr.&teacher_acctid=42d43fa6f37314365d08692e08680973",
    "/v1/donor/ny_teaching_fellow_status?teacher_acctid=42d43fa6f37314365d08692e08680973",
    "/v1/donor/essay_titles_by_city?school_city=Abington",
    "/v1/donor/item_names_by_teacher?teacher_acctid=822b7b8768c17456fdce78b65abcc18e",
    "/v1/donor/count_school_ids_by_magnet_payment?school_magnet=t&payment_included_acct_credit=f",
    "/v1/donor/percentage_projects_by_resource_type?project_resource_type=Books&school_magnet=t",
    "/v1/donor/count_school_ids_by_magnet_vendor?school_magnet=t&vendor_name=ABC%20School%20Supply",
    "/v1/donor/count_school_ids_by_poverty_donor?poverty_level=highest%20poverty&donor_acctid=000eebf28658900e63b538cf8a73afbd",
    "/v1/donor/get_short_description_by_schoolid?schoolid=301c9bf0a45d159d162b65a93fddd74e",
    "/v1/donor/get_school_city_by_essay_title?title=iMath",
    "/v1/donor/get_payment_method_by_teacher_acctid?teacher_acctid=822b7b8768c17456fdce78b65abcc18e",
    "/v1/donor/get_donation_total_by_teacher_acctid?teacher_acctid=822b7b8768c17456fdce78b65abcc18e",
    "/v1/donor/get_teacher_acct_status_by_schoolid?schoolid=d4af834b1d3fc8061e1ee1b3f1a77b85",
    "/v1/donor/get_percentage_projects_by_city_teacher_acct?school_city=Brooklyn&is_teacher_acct=t",
    "/v1/donor/get_percentage_projects_by_students_reached_teacher_acct?students_reached=30&is_teacher_acct=t",
    "/v1/donor/count_school_ids_by_city_metro?school_city=Bethlehem&school_metro=suburban",
    "/v1/donor/count_year_round_schools?school_city=Los%20Angeles&school_year_round=t",
    "/v1/donor/count_magnet_schools?school_county=New%20York%20(Manhattan)&school_magnet=t",
    "/v1/donor/count_teachers_focus_area?school_county=Twin%20Falls&primary_focus_area=Math%20%26%20Science",
    "/v1/donor/count_teachers_focus_subject?teacher_prefix=Mr.&primary_focus_subject=Literature%20%26%20Writing",
    "/v1/donor/count_projects_grade_level?school_district=Boston%20Public%20School%20District&grade_level=Grades%203-5",
    "/v1/donor/get_teacher_acct_status_by_essay_title?title_pattern=Calculate,%20Financial%20Security%20For%20Tomorrow%20Starts%20Today!%20",
    "/v1/donor/get_total_donation_by_essay_title?title_pattern=A%20Rug%20For%20Reaching%20Readers",
    "/v1/donor/get_total_donation_with_support_by_essay_title?title_pattern=Engaging%20Young%20Readers%20with%20a%20Leveled%20Classroom%20Library%20",
    "/v1/donor/donation_optional_support_by_title?title=Armenian%20Genocide",
    "/v1/donor/short_description_by_donation_timestamp?donation_timestamp=2012-09-06%2014:44:29",
    "/v1/donor/donation_included_optional_support_by_title?title=I%20Can't%20See%20It...Can%20You%20Help%20Me???",
    "/v1/donor/teacher_acctid_by_donation_timestamp?donation_timestamp=2008-07-29%2011:38:43.361",
    "/v1/donor/schoolid_by_title?title=Virtual%20Aquarium%20Needs%20Help!",
    "/v1/donor/titles_with_max_donation_total",
    "/v1/donor/percentage_donation_optional_support_by_title?title=Awesome%20Audiobooks%20Make%20Avid%20Readers",
    "/v1/donor/percentage_donations_by_payment_method_and_title?payment_method=creditcard&title=Bringing%20Drama%20to%20Life",
    "/v1/donor/short_description_by_title?title=Future%20Einsteins%20Of%20America",
    "/v1/donor/need_statement_by_title?title=Family%20History%20Project",
    "/v1/donor/school_county_by_poverty_level_state?poverty_level=low%20poverty&school_state=NY",
    "/v1/donor/school_districts_by_essay_title?title=Future%20Einsteins%20Of%20America",
    "/v1/donor/payment_methods_by_essay_title?title=Needed%20Resource%20Materials%20For%20My%20Students",
    "/v1/donor/project_count_by_payment_method_district?payment_method=creditcard&school_district=Memphis%20City%20School%20District",
    "/v1/donor/school_districts_by_vendor?vendor_name=Barnes%20and%20Noble",
    "/v1/donor/school_location_by_item_vendor?item_name=R%20%26%20A%20Plant%20Genetics&vendor_name=Benchmark%20Education",
    "/v1/donor/most_common_payment_method_by_state?school_state=GA",
    "/v1/donor/school_info_by_essay_title?essay_title=Look,%20Look,%20We%20Need%20a%20Nook!",
    "/v1/donor/donation_messages_by_location_and_district?school_latitude=40.735332&school_longitude=-74.196014&school_district=Newark%20School%20District",
    "/v1/donor/project_posting_dates_by_essay_title?essay_title=Lets%20Share%20Ideas",
    "/v1/donor/donation_messages_by_essay_title_and_payment_method?essay_title=Lets%20Share%20Ideas&payment_method=creditcard",
    "/v1/donor/most_common_project_resource_type_by_district?school_district=Los%20Angeles%20Unif%20Sch%20Dist",
    "/v1/donor/school_cities_by_district_and_vendor?school_district=Los%20Angeles%20Unif%20Sch%20Dist&vendor_name=Quill.com",
    "/v1/donor/item_price_and_location_by_vendor_resource_type_district?vendor_name=ABC%20School%20Supply&project_resource_type=Other&school_district=Hillsborough%20Co%20Pub%20Sch%20Dist",
    "/v1/donor/total_donations_by_essay_title_and_payment_method?essay_title=Lets%20Share%20Ideas&payment_method=paypal",
    "/v1/donor/donation_count_by_city_and_teacher_status?donor_city=Pocatello&is_teacher_acct=f",
    "/v1/donor/school_count_by_teacher_prefix_and_county?teacher_prefix=Dr.&school_county=Suffolk",
    "/v1/donor/total_donations_by_payment_method?payment_method=amazon",
    "/v1/donor/vendor_ids_by_resource_type_and_price?project_resource_type=Technology&item_unit_price=15",
    "/v1/donor/school_ids_by_district_and_fellow?school_district=Union%20Pub%20School%20District%20I-9&teacher_ny_teaching_fellow=t",
    "/v1/donor/school_cities_by_metro_and_county?school_metro=suburban&school_county=Los%20Angeles",
    "/v1/donor/distinct_vendor_project_ids_by_resource_type?project_resource_type=Books",
    "/v1/donor/percentage_donations_with_gift_cards?payment_included_campaign_gift_card=t&payment_method=no_cash_received",
    "/v1/donor/percentage_projects_in_metro?school_metro=suburban&school_city=Santa%20Barbara",
    "/v1/donor/average_item_price_per_quantity?vendor_name=AKJ%20Books",
    "/v1/donor/count_school_ids_payment_poverty?payment_included_campaign_gift_card=t&poverty_level=highest%20poverty",
    "/v1/donor/top_school_city_by_donation_amount?dollar_amount=under_10",
    "/v1/donor/essay_titles_by_location?school_latitude=42003718&school_longitude=-87668289",
    "/v1/donor/ny_teaching_fellows_by_essay_title?title=Team%20More%20Books!",
    "/v1/donor/top_vendor_by_primary_focus_area?primary_focus_area=Literacy%25",
    "/v1/donor/distinct_vendor_names_by_grade_level?grade_level=Grades%209-12",
    "/v1/donor/count_distinct_teacher_acctids?is_teacher_acct=t&school_city=Chicago",
    "/v1/donor/count_distinct_teacher_acctids_by_city?school_city=Rock%20Hill&is_teacher_acct=t",
    "/v1/donor/total_dollar_amount_students_reached_optional_support?students_reached=300&donation_included_optional_support=t",
    "/v1/donor/total_item_quantity_metro_district?school_metro=urban&school_district=Onslow%20Co%20School%20District",
    "/v1/donor/average_donation_amount_county?school_county=Fresno",
    "/v1/donor/percentage_projects_title_pattern?title_pattern=ABC%20Read",
    "/v1/donor/average_item_quantity_honoree?for_honoree=t",
    "/v1/donor/most_expensive_item_primary_focus?primary_focus_subject=Mathematics",
    "/v1/donor/total_donation_amount_essay_title?title=Look,%20Look,%20We%20Need%20a%20Nook!",
    "/v1/donor/distinct_poverty_levels_donor_zip?donor_zip=7079",
    "/v1/donor/vendor_names_by_essay_title?title=Bloody%20Times",
    "/v1/donor/school_zip_item_name_by_vendor?vendor_name=Sax%20Arts%20%26%20Crafts",
    "/v1/donor/school_coordinates_by_essay_title?title=Wiping%20Away%20Bad%20Grades",
    "/v1/donor/primary_focus_subjects_by_gift_card?payment_included_campaign_gift_card=t",
    "/v1/donor/most_expensive_item_by_resource_type?project_resource_type=Books",
    "/v1/donor/grade_levels_by_essay_title?title=Too%20Close%20for%20Comfort",
    "/v1/donor/total_students_reached_by_donor_zip?donor_zip=22205",
    "/v1/donor/donation_percentage_and_top_focus_area?via_giving_page=t",
    "/v1/donor/vendor_project_percentage_and_posting_dates?vendor_name=Best%20Buy%20for%20Business&project_resource_type=Technology"
]
