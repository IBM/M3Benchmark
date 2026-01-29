from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/toxicology/toxicology.sqlite')
cursor = conn.cursor()

# Endpoint to get the most common bond type
@app.get("/v1/toxicology/most_common_bond_type", operation_id="get_most_common_bond_type", summary="Retrieves the most frequently occurring bond type in the database. This operation calculates the count of bond IDs for each bond type and returns the bond type with the highest count.")
async def get_most_common_bond_type():
    cursor.execute("SELECT T.bond_type FROM ( SELECT bond_type, COUNT(bond_id) FROM bond GROUP BY bond_type ORDER BY COUNT(bond_id) DESC LIMIT 1 ) AS T")
    result = cursor.fetchone()
    if not result:
        return {"bond_type": []}
    return {"bond_type": result[0]}

# Endpoint to get the count of distinct molecules with a specific element and label
@app.get("/v1/toxicology/count_distinct_molecules_by_element_label", operation_id="get_count_distinct_molecules", summary="Retrieve the number of unique molecules that contain a specified element and have a particular label. The element is a component of the molecule's atoms, and the label is a descriptor of the molecule.")
async def get_count_distinct_molecules(element: str = Query(..., description="Element of the atom"), label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT COUNT(DISTINCT T1.molecule_id) FROM molecule AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.element = ? AND T1.label = ?", (element, label))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average oxygen count in molecules with a specific bond type and element
@app.get("/v1/toxicology/avg_oxygen_count_by_bond_type_element", operation_id="get_avg_oxygen_count", summary="Retrieves the average count of oxygen atoms in molecules that have a specified bond type and element. The operation calculates the average oxygen count by first identifying molecules with the given bond type and element, then aggregating the oxygen count for each molecule.")
async def get_avg_oxygen_count(bond_type: str = Query(..., description="Bond type"), element: str = Query(..., description="Element of the atom")):
    cursor.execute("SELECT AVG(oxygen_count) FROM (SELECT T1.molecule_id, COUNT(T1.element) AS oxygen_count FROM atom AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.bond_type = ? AND T1.element = ? GROUP BY T1.molecule_id) AS oxygen_counts", (bond_type, element))
    result = cursor.fetchone()
    if not result:
        return {"avg_oxygen_count": []}
    return {"avg_oxygen_count": result[0]}

# Endpoint to get the average single bond count in molecules with a specific bond type and label
@app.get("/v1/toxicology/avg_single_bond_count_by_bond_type_label", operation_id="get_avg_single_bond_count", summary="Get the average single bond count in molecules with a specific bond type and label")
async def get_avg_single_bond_count(bond_type: str = Query(..., description="Bond type"), label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT AVG(single_bond_count) FROM (SELECT T3.molecule_id, COUNT(T1.bond_type) AS single_bond_count FROM bond AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN molecule AS T3 ON T3.molecule_id = T2.molecule_id WHERE T1.bond_type = ? AND T3.label = ? GROUP BY T3.molecule_id) AS subquery", (bond_type, label))
    result = cursor.fetchone()
    if not result:
        return {"avg_single_bond_count": []}
    return {"avg_single_bond_count": result[0]}

# Endpoint to get distinct molecule IDs with a specific bond type and label
@app.get("/v1/toxicology/distinct_molecule_ids_by_bond_type_label", operation_id="get_distinct_molecule_ids", summary="Get distinct molecule IDs with a specific bond type and label")
async def get_distinct_molecule_ids(bond_type: str = Query(..., description="Bond type"), label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT DISTINCT T2.molecule_id FROM bond AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.bond_type = ? AND T2.label = ?", (bond_type, label))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids": []}
    return {"molecule_ids": [row[0] for row in result]}

# Endpoint to get the percentage of atoms with a specific element and bond type
@app.get("/v1/toxicology/percentage_atoms_by_element_bond_type", operation_id="get_percentage_atoms", summary="Get the percentage of atoms with a specific element and bond type")
async def get_percentage_atoms(element: str = Query(..., description="Element of the atom"), bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT CAST(COUNT(DISTINCT CASE WHEN T1.element = ? THEN T1.atom_id ELSE NULL END) AS REAL) * 100 / COUNT(DISTINCT T1.atom_id) FROM atom AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.bond_type = ?", (element, bond_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of bonds with a specific bond type
@app.get("/v1/toxicology/count_bonds_by_bond_type", operation_id="get_count_bonds", summary="Get the count of bonds with a specific bond type")
async def get_count_bonds(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT COUNT(T.bond_id) FROM bond AS T WHERE T.bond_type = ?", (bond_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct atoms excluding a specific element
@app.get("/v1/toxicology/count_distinct_atoms_excluding_element", operation_id="get_count_distinct_atoms", summary="Retrieves the total number of unique atoms, excluding those composed of a specified element. This operation is useful for understanding the diversity of atoms in a dataset, while filtering out a particular element.")
async def get_count_distinct_atoms(element: str = Query(..., description="Element of the atom to exclude")):
    cursor.execute("SELECT COUNT(DISTINCT T.atom_id) FROM atom AS T WHERE T.element <> ?", (element,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of molecules within a specific ID range and label
@app.get("/v1/toxicology/count_molecules_by_id_range_label", operation_id="get_count_molecules", summary="Get the count of molecules within a specific ID range and label")
async def get_count_molecules(min_id: str = Query(..., description="Minimum molecule ID"), max_id: str = Query(..., description="Maximum molecule ID"), label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT COUNT(T.molecule_id) FROM molecule AS T WHERE T.molecule_id BETWEEN ? AND ? AND T.label = ?", (min_id, max_id, label))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get molecule IDs based on element
@app.get("/v1/toxicology/molecule_ids_by_element", operation_id="get_molecule_ids_by_element", summary="Retrieves a list of molecule IDs associated with a specified element. The element is identified by its symbol, which is used to filter the relevant molecule IDs from the database.")
async def get_molecule_ids_by_element(element: str = Query(..., description="Element symbol")):
    cursor.execute("SELECT T.molecule_id FROM atom AS T WHERE T.element = ?", (element,))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids": []}
    return {"molecule_ids": [row[0] for row in result]}

# Endpoint to get distinct elements based on bond ID
@app.get("/v1/toxicology/distinct_elements_by_bond_id", operation_id="get_distinct_elements_by_bond_id", summary="Retrieves a unique set of elements associated with a specific bond ID. The bond ID is used to filter the elements from the atom table, which are then joined with the connected table to ensure the elements are linked to the specified bond.")
async def get_distinct_elements_by_bond_id(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT DISTINCT T1.element FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T2.bond_id = ?", (bond_id,))
    result = cursor.fetchall()
    if not result:
        return {"elements": []}
    return {"elements": [row[0] for row in result]}

# Endpoint to get distinct elements based on bond type
@app.get("/v1/toxicology/distinct_elements_by_bond_type", operation_id="get_distinct_elements_by_bond_type", summary="Retrieves a unique set of elements that are involved in a specific bond type within a molecule. The bond type is specified as an input parameter.")
async def get_distinct_elements_by_bond_type(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT DISTINCT T1.element FROM atom AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN connected AS T3 ON T1.atom_id = T3.atom_id WHERE T2.bond_type = ?", (bond_type,))
    result = cursor.fetchall()
    if not result:
        return {"elements": []}
    return {"elements": [row[0] for row in result]}

# Endpoint to get the most common molecule label for a specific element
@app.get("/v1/toxicology/most_common_molecule_label_by_element", operation_id="get_most_common_molecule_label_by_element", summary="Retrieves the label of the most frequently occurring molecule containing a specified element. The operation identifies the molecule with the highest count of the given element and returns its label.")
async def get_most_common_molecule_label_by_element(element: str = Query(..., description="Element symbol")):
    cursor.execute("SELECT T.label FROM ( SELECT T2.label, COUNT(T2.molecule_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.element = ? GROUP BY T2.label ORDER BY COUNT(T2.molecule_id) DESC LIMIT 1 ) t", (element,))
    result = cursor.fetchone()
    if not result:
        return {"label": []}
    return {"label": result[0]}

# Endpoint to get distinct bond types based on element
@app.get("/v1/toxicology/distinct_bond_types_by_element", operation_id="get_distinct_bond_types_by_element", summary="Retrieve a unique set of bond types associated with a specific chemical element. The element is identified by its symbol, which is provided as an input parameter.")
async def get_distinct_bond_types_by_element(element: str = Query(..., description="Element symbol")):
    cursor.execute("SELECT DISTINCT T1.bond_type FROM bond AS T1 INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id INNER JOIN atom AS T3 ON T2.atom_id = T3.atom_id WHERE T3.element = ?", (element,))
    result = cursor.fetchall()
    if not result:
        return {"bond_types": []}
    return {"bond_types": [row[0] for row in result]}

# Endpoint to get atom IDs based on bond type
@app.get("/v1/toxicology/atom_ids_by_bond_type", operation_id="get_atom_ids_by_bond_type", summary="Retrieves the IDs of atoms connected by a specific bond type. The bond type is provided as an input parameter, and the operation returns a list of atom ID pairs that are connected by this bond type.")
async def get_atom_ids_by_bond_type(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT T2.atom_id, T2.atom_id2 FROM bond AS T1 INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id WHERE T1.bond_type = ?", (bond_type,))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": [{"atom_id": row[0], "atom_id2": row[1]} for row in result]}

# Endpoint to get distinct atom IDs based on molecule label
@app.get("/v1/toxicology/distinct_atom_ids_by_molecule_label", operation_id="get_distinct_atom_ids_by_molecule_label", summary="Retrieve a unique set of atom identifiers associated with a specific molecule label. This operation filters the atom table based on the provided molecule label and returns the distinct atom identifiers that are connected to the molecule.")
async def get_distinct_atom_ids_by_molecule_label(label: str = Query(..., description="Molecule label")):
    cursor.execute("SELECT DISTINCT T1.atom_id FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN connected AS T3 ON T1.atom_id = T3.atom_id WHERE T2.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": [row[0] for row in result]}

# Endpoint to get the least common element for a specific molecule label
@app.get("/v1/toxicology/least_common_element_by_molecule_label", operation_id="get_least_common_element_by_molecule_label", summary="Retrieves the least frequently occurring element associated with a specific molecule label. The operation identifies the element that appears in the least number of distinct molecules sharing the provided label. The molecule label is a required input parameter.")
async def get_least_common_element_by_molecule_label(label: str = Query(..., description="Molecule label")):
    cursor.execute("SELECT T.element FROM (SELECT T1.element, COUNT(DISTINCT T1.molecule_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ? GROUP BY T1.element ORDER BY COUNT(DISTINCT T1.molecule_id) ASC LIMIT 1) t", (label,))
    result = cursor.fetchone()
    if not result:
        return {"element": []}
    return {"element": result[0]}

# Endpoint to get bond types based on atom IDs
@app.get("/v1/toxicology/bond_types_by_atom_ids", operation_id="get_bond_types_by_atom_ids", summary="Retrieves the bond type(s) between two specified atoms. The operation considers both combinations of the input atom IDs to ensure comprehensive results.")
async def get_bond_types_by_atom_ids(atom_id1: str = Query(..., description="First atom ID"), atom_id2: str = Query(..., description="Second atom ID")):
    cursor.execute("SELECT T1.bond_type FROM bond AS T1 INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id WHERE (T2.atom_id = ? AND T2.atom_id2 = ?) OR (T2.atom_id2 = ? AND T2.atom_id = ?)", (atom_id1, atom_id2, atom_id1, atom_id2))
    result = cursor.fetchall()
    if not result:
        return {"bond_types": []}
    return {"bond_types": [row[0] for row in result]}

# Endpoint to get distinct molecule labels based on element exclusion
@app.get("/v1/toxicology/distinct_molecule_labels_by_element_exclusion", operation_id="get_distinct_molecule_labels_by_element_exclusion", summary="Retrieves a list of unique molecule labels that do not contain a specified element. The element to be excluded is provided as an input parameter.")
async def get_distinct_molecule_labels_by_element_exclusion(element: str = Query(..., description="Element symbol to exclude")):
    cursor.execute("SELECT DISTINCT T2.label FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.element != ?", (element,))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get the count of iodine and sulfur atoms in bonds of a specific type
@app.get("/v1/toxicology/count_iodine_sulfur_bond_type", operation_id="get_count_iodine_sulfur_bond_type", summary="Retrieves the distinct count of atoms for two specified elements involved in a particular bond type. The operation identifies the atoms of the given elements that participate in the specified bond type and returns their respective counts.")
async def get_count_iodine_sulfur_bond_type(element1: str = Query(..., description="First element to count"), element2: str = Query(..., description="Second element to count"), bond_type: str = Query(..., description="Type of bond")):
    cursor.execute("SELECT COUNT(DISTINCT CASE WHEN T1.element = ? THEN T1.atom_id ELSE NULL END) AS iodine_nums , COUNT(DISTINCT CASE WHEN T1.element = ? THEN T1.atom_id ELSE NULL END) AS sulfur_nums FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id INNER JOIN bond AS T3 ON T2.bond_id = T3.bond_id WHERE T3.bond_type = ?", (element1, element2, bond_type))
    result = cursor.fetchone()
    if not result:
        return {"iodine_nums": [], "sulfur_nums": []}
    return {"iodine_nums": result[0], "sulfur_nums": result[1]}

# Endpoint to get atom IDs connected in a specific molecule
@app.get("/v1/toxicology/atom_ids_by_molecule", operation_id="get_atom_ids_by_molecule", summary="Retrieves the IDs of atoms that are connected within a specific molecule. The operation requires the molecule's unique identifier as input and returns a list of atom ID pairs that are connected in the molecule.")
async def get_atom_ids_by_molecule(molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT T2.atom_id, T2.atom_id2 FROM atom AS T1 INNER JOIN connected AS T2 ON T2.atom_id = T1.atom_id WHERE T1.molecule_id = ?", (molecule_id,))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": result}

# Endpoint to get the percentage of molecules with a specific element and label
@app.get("/v1/toxicology/percentage_molecules_element_label", operation_id="get_percentage_molecules_element_label", summary="Get the percentage of molecules with a specific element and label")
async def get_percentage_molecules_element_label(element: str = Query(..., description="Element to exclude"), label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT CAST(COUNT(DISTINCT CASE WHEN T1.element <> ? THEN T2.molecule_id ELSE NULL END) AS REAL) * 100 / COUNT(DISTINCT T2.molecule_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ?", (element, label))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of molecules with a specific label and bond type
@app.get("/v1/toxicology/percentage_molecules_label_bond_type", operation_id="get_percentage_molecules_label_bond_type", summary="Get the percentage of molecules with a specific label and bond type")
async def get_percentage_molecules_label_bond_type(label: str = Query(..., description="Label of the molecule"), bond_type: str = Query(..., description="Type of bond")):
    cursor.execute("SELECT CAST(COUNT(DISTINCT CASE WHEN T2.label = ? THEN T2.molecule_id ELSE NULL END) AS REAL) * 100 / COUNT(DISTINCT T2.molecule_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T3.bond_type = ?", (label, bond_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct elements in a specific molecule
@app.get("/v1/toxicology/distinct_elements_by_molecule", operation_id="get_distinct_elements_by_molecule", summary="Retrieves a distinct list of elements present in a specific molecule, ordered alphabetically. The number of elements returned can be limited by specifying a maximum count.")
async def get_distinct_elements_by_molecule(molecule_id: str = Query(..., description="Molecule ID"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T.element FROM atom AS T WHERE T.molecule_id = ? ORDER BY T.element LIMIT ?", (molecule_id, limit))
    result = cursor.fetchall()
    if not result:
        return {"elements": []}
    return {"elements": [row[0] for row in result]}

# Endpoint to get atom IDs from a specific bond in a molecule
@app.get("/v1/toxicology/atom_ids_from_bond", operation_id="get_atom_ids_from_bond", summary="Retrieves the atom IDs associated with a specific bond in a given molecule. The operation requires the molecule ID and bond ID as input parameters to identify the bond and extract the corresponding atom IDs. The response includes two atom IDs, each representing an atom connected by the specified bond.")
async def get_atom_ids_from_bond(molecule_id: str = Query(..., description="Molecule ID"), bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT SUBSTR(T.bond_id, 1, 7) AS atom_id1 , T.molecule_id || SUBSTR(T.bond_id, 8, 2) AS atom_id2 FROM bond AS T WHERE T.molecule_id = ? AND T.bond_id = ?", (molecule_id, bond_id))
    result = cursor.fetchone()
    if not result:
        return {"atom_id1": [], "atom_id2": []}
    return {"atom_id1": result[0], "atom_id2": result[1]}

# Endpoint to get the difference in count of molecules with specific labels
@app.get("/v1/toxicology/diff_molecule_count_by_label", operation_id="get_diff_molecule_count_by_label", summary="Get the difference in count of molecules with specific labels")
async def get_diff_molecule_count_by_label(label1: str = Query(..., description="First label"), label2: str = Query(..., description="Second label")):
    cursor.execute("SELECT COUNT(CASE WHEN T.label = ? THEN T.molecule_id ELSE NULL END) - COUNT(CASE WHEN T.label = ? THEN T.molecule_id ELSE NULL END) AS diff_car_notcar FROM molecule T", (label1, label2))
    result = cursor.fetchone()
    if not result:
        return {"diff_car_notcar": []}
    return {"diff_car_notcar": result[0]}

# Endpoint to get atom IDs connected by a specific bond
@app.get("/v1/toxicology/atom_ids_by_bond", operation_id="get_atom_ids_by_bond", summary="Retrieves the atom IDs that are connected by a specific bond. The bond is identified by its unique bond ID.")
async def get_atom_ids_by_bond(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT T.atom_id FROM connected AS T WHERE T.bond_id = ?", (bond_id,))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": [row[0] for row in result]}

# Endpoint to get bond IDs connected to a specific atom
@app.get("/v1/toxicology/bond_ids_by_atom", operation_id="get_bond_ids_by_atom", summary="Retrieves the identifiers of all bonds connected to a specified atom. The operation requires the atom's unique identifier as input and returns a list of bond IDs associated with the atom.")
async def get_bond_ids_by_atom(atom_id2: str = Query(..., description="Atom ID")):
    cursor.execute("SELECT T.bond_id FROM connected AS T WHERE T.atom_id2 = ?", (atom_id2,))
    result = cursor.fetchall()
    if not result:
        return {"bond_ids": []}
    return {"bond_ids": [row[0] for row in result]}

# Endpoint to get the percentage of bonds of a specific type in a molecule
@app.get("/v1/toxicology/bond_type_percentage", operation_id="get_bond_type_percentage", summary="Retrieves the percentage of a specific bond type in a given molecule. The bond type is specified as an input parameter, and the molecule is identified by its unique ID. The result is a rounded value representing the proportion of bonds of the specified type in the molecule.")
async def get_bond_type_percentage(bond_type: str = Query(..., description="Bond type"), molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT ROUND(CAST(COUNT(CASE WHEN T.bond_type = ? THEN T.bond_id ELSE NULL END) AS REAL) * 100 / COUNT(T.bond_id),5) FROM bond AS T WHERE T.molecule_id = ?", (bond_type, molecule_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of molecules with a specific label
@app.get("/v1/toxicology/molecule_label_percentage", operation_id="get_molecule_label_percentage", summary="Retrieves the percentage of molecules in the database that match a specified label. The label is used to filter the molecules and calculate the percentage based on the total count of molecules.")
async def get_molecule_label_percentage(label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT ROUND(CAST(COUNT(CASE WHEN T.label = ? THEN T.molecule_id ELSE NULL END) AS REAL) * 100 / COUNT(T.molecule_id),3) FROM molecule T", (label,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of atoms of a specific element in a molecule
@app.get("/v1/toxicology/atom_element_percentage", operation_id="get_atom_element_percentage", summary="Retrieves the percentage of atoms of a specified element within a given molecule. The calculation is based on the total count of atoms in the molecule. The element is identified by its symbol, and the molecule is specified by its unique ID.")
async def get_atom_element_percentage(element: str = Query(..., description="Element symbol"), molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT ROUND(CAST(COUNT(CASE WHEN T.element = ? THEN T.atom_id ELSE NULL END) AS REAL) * 100 / COUNT(T.atom_id),4) FROM atom AS T WHERE T.molecule_id = ?", (element, molecule_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct bond types for a specific molecule
@app.get("/v1/toxicology/distinct_bond_types", operation_id="get_distinct_bond_types", summary="Retrieves the unique bond types associated with a specific molecule, identified by its molecule ID.")
async def get_distinct_bond_types(molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT DISTINCT T.bond_type FROM bond AS T WHERE T.molecule_id = ?", (molecule_id,))
    result = cursor.fetchall()
    if not result:
        return {"bond_types": []}
    return {"bond_types": [row[0] for row in result]}

# Endpoint to get distinct elements and labels for a specific molecule
@app.get("/v1/toxicology/distinct_elements_labels", operation_id="get_distinct_elements_labels", summary="Retrieves unique elements and their corresponding labels associated with a specific molecule. The operation requires the molecule's ID as input to filter the results.")
async def get_distinct_elements_labels(molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT DISTINCT T1.element, T2.label FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.molecule_id = ?", (molecule_id,))
    result = cursor.fetchall()
    if not result:
        return {"elements_labels": []}
    return {"elements_labels": [{"element": row[0], "label": row[1]} for row in result]}

# Endpoint to get distinct bond IDs for a specific molecule with a limit
@app.get("/v1/toxicology/distinct_bond_ids_by_molecule", operation_id="get_distinct_bond_ids_by_molecule", summary="Retrieves a distinct set of bond IDs associated with a specific molecule, up to a specified limit. The operation filters bonds based on the provided molecule ID and returns them in ascending order.")
async def get_distinct_bond_ids_by_molecule(molecule_id: str = Query(..., description="Molecule ID"), limit: int = Query(..., description="Limit on the number of results")):
    cursor.execute("SELECT DISTINCT T2.bond_id FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T1.molecule_id = ? ORDER BY T2.bond_id LIMIT ?", (molecule_id, limit))
    result = cursor.fetchall()
    if not result:
        return {"bond_ids": []}
    return {"bond_ids": [row[0] for row in result]}

# Endpoint to get the count of bonds for a specific molecule with specific atom IDs
@app.get("/v1/toxicology/bond_count_by_molecule_atom_ids", operation_id="get_bond_count_by_molecule_atom_ids", summary="Retrieves the total number of bonds between two specific atoms within a given molecule. The molecule is identified by its unique ID, and the atoms are specified by their respective IDs. This operation provides a quantitative measure of the chemical connectivity between the selected atoms in the molecule.")
async def get_bond_count_by_molecule_atom_ids(molecule_id: str = Query(..., description="Molecule ID"), atom_id: str = Query(..., description="First atom ID"), atom_id2: str = Query(..., description="Second atom ID")):
    cursor.execute("SELECT COUNT(T2.bond_id) FROM bond AS T1 INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id WHERE T1.molecule_id = ? AND T2.atom_id = ? AND T2.atom_id2 = ?", (molecule_id, atom_id, atom_id2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct molecule IDs based on label and element
@app.get("/v1/toxicology/count_distinct_molecule_ids", operation_id="get_count_distinct_molecule_ids", summary="Get the count of distinct molecule IDs based on label and element")
async def get_count_distinct_molecule_ids(label: str = Query(..., description="Label of the molecule"), element: str = Query(..., description="Element of the atom")):
    cursor.execute("SELECT COUNT(DISTINCT T2.molecule_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ? AND T1.element = ?", (label, element))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get bond type and atom IDs based on bond ID
@app.get("/v1/toxicology/get_bond_type_atom_ids", operation_id="get_bond_type_atom_ids", summary="Retrieves the bond type and associated atom IDs for a given bond ID. This operation fetches the bond type from the bond table and the corresponding atom IDs from the connected table, using the provided bond ID as a reference.")
async def get_bond_type_atom_ids(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT T1.bond_type, T2.atom_id, T2.atom_id2 FROM bond AS T1 INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id WHERE T2.bond_id = ?", (bond_id,))
    result = cursor.fetchall()
    if not result:
        return {"bond_info": []}
    return {"bond_info": result}

# Endpoint to get molecule IDs and carcinogenic flag based on atom ID and label
@app.get("/v1/toxicology/get_molecule_ids_carcinogenic_flag", operation_id="get_molecule_ids_carcinogenic_flag", summary="Get molecule IDs and carcinogenic flag based on atom ID and label")
async def get_molecule_ids_carcinogenic_flag(atom_id: str = Query(..., description="Atom ID"), label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT T2.molecule_id, IIF(T2.label = ?, 'YES', 'NO') AS flag_carcinogenic FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.atom_id = ?", (label, atom_id))
    result = cursor.fetchall()
    if not result:
        return {"molecule_info": []}
    return {"molecule_info": result}

# Endpoint to get the count of distinct molecule IDs based on bond type
@app.get("/v1/toxicology/count_distinct_molecule_ids_bond_type", operation_id="get_count_distinct_molecule_ids_bond_type", summary="Get the count of distinct molecule IDs based on bond type")
async def get_count_distinct_molecule_ids_bond_type(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT COUNT(DISTINCT T.molecule_id) FROM bond AS T WHERE T.bond_type = ?", (bond_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of bond IDs based on the last two characters of atom ID
@app.get("/v1/toxicology/count_bond_ids_atom_id_suffix", operation_id="get_count_bond_ids_atom_id_suffix", summary="Retrieves the total number of bond IDs associated with the specified atom ID suffix. The atom ID suffix is the last two characters of the atom ID.")
async def get_count_bond_ids_atom_id_suffix(atom_id_suffix: str = Query(..., description="Last two characters of atom ID")):
    cursor.execute("SELECT COUNT(T.bond_id) FROM connected AS T WHERE SUBSTR(T.atom_id, -2) = ?", (atom_id_suffix,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct elements based on molecule ID
@app.get("/v1/toxicology/get_distinct_elements", operation_id="get_distinct_elements", summary="Retrieves a list of unique elements associated with a specific molecule. The molecule is identified by its unique ID, which is provided as an input parameter.")
async def get_distinct_elements(molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT DISTINCT T.element FROM atom AS T WHERE T.molecule_id = ?", (molecule_id,))
    result = cursor.fetchall()
    if not result:
        return {"elements": []}
    return {"elements": result}

# Endpoint to get the count of molecule IDs based on label
@app.get("/v1/toxicology/count_molecule_ids_label", operation_id="get_count_molecule_ids_label", summary="Retrieves the total count of molecule IDs that match the specified label. The label is used to filter the molecules and determine the count.")
async def get_count_molecule_ids_label(label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT COUNT(T.molecule_id) FROM molecule AS T WHERE T.label = ?", (label,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct molecule IDs based on atom ID suffix range and label
@app.get("/v1/toxicology/get_distinct_molecule_ids_atom_id_range_label", operation_id="get_distinct_molecule_ids_atom_id_range_label", summary="Get distinct molecule IDs based on atom ID suffix range and label")
async def get_distinct_molecule_ids_atom_id_range_label(start_suffix: str = Query(..., description="Start of atom ID suffix range"), end_suffix: str = Query(..., description="End of atom ID suffix range"), label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT DISTINCT T2.molecule_id FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE SUBSTR(T1.atom_id, -2) BETWEEN ? AND ? AND T2.label = ?", (start_suffix, end_suffix, label))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids": []}
    return {"molecule_ids": result}

# Endpoint to get bond IDs based on elements of connected atoms
@app.get("/v1/toxicology/get_bond_ids_connected_elements", operation_id="get_bond_ids_connected_elements", summary="Retrieves bond IDs for connections between atoms of the specified elements. The operation identifies bonds where one atom has the first element and the other atom has the second element. This allows for the extraction of bond information based on the elements of the connected atoms.")
async def get_bond_ids_connected_elements(element1: str = Query(..., description="Element of the first atom"), element2: str = Query(..., description="Element of the second atom")):
    cursor.execute("SELECT T2.bond_id FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T2.bond_id IN ( SELECT T3.bond_id FROM connected AS T3 INNER JOIN atom AS T4 ON T3.atom_id = T4.atom_id WHERE T4.element = ? ) AND T1.element = ?", (element1, element2))
    result = cursor.fetchall()
    if not result:
        return {"bond_ids": []}
    return {"bond_ids": result}

# Endpoint to get the label of the molecule with the highest count of a specific bond type
@app.get("/v1/toxicology/get_molecule_label_highest_bond_type_count", operation_id="get_molecule_label_highest_bond_type_count", summary="Get the label of the molecule with the highest count of a specific bond type")
async def get_molecule_label_highest_bond_type_count(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT T1.label FROM molecule AS T1 INNER JOIN ( SELECT T.molecule_id, COUNT(T.bond_type) FROM bond AS T WHERE T.bond_type = ? GROUP BY T.molecule_id ORDER BY COUNT(T.bond_type) DESC LIMIT 1 ) AS T2 ON T1.molecule_id = T2.molecule_id", (bond_type,))
    result = cursor.fetchone()
    if not result:
        return {"label": []}
    return {"label": result[0]}

# Endpoint to get the ratio of bond count to atom count for a specific element
@app.get("/v1/toxicology/bond_atom_ratio", operation_id="get_bond_atom_ratio", summary="Retrieves the ratio of bond count to atom count for a specified element. This operation calculates the ratio by counting the number of bonds associated with the specified element and dividing it by the total count of atoms of that element. The element is identified by its symbol.")
async def get_bond_atom_ratio(element: str = Query(..., description="Element symbol")):
    cursor.execute("SELECT CAST(COUNT(T2.bond_id) AS REAL) / COUNT(T1.atom_id) FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T1.element = ?", (element,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get bond type and bond ID for a specific atom ID substring
@app.get("/v1/toxicology/bond_type_id_by_atom_id_substring", operation_id="get_bond_type_id_by_atom_id_substring", summary="Retrieves the bond type and bond ID associated with a specific atom ID substring. The substring, consisting of 2 characters starting from the 7th position, is used to filter the results. This operation is useful for identifying the bond type and bond ID for a given atom ID substring.")
async def get_bond_type_id_by_atom_id_substring(atom_id_substring: str = Query(..., description="Substring of atom ID (2 characters starting from the 7th position)")):
    cursor.execute("SELECT T1.bond_type, T1.bond_id FROM bond AS T1 INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id WHERE SUBSTR(T2.atom_id, 7, 2) = ?", (atom_id_substring,))
    result = cursor.fetchall()
    if not result:
        return {"bonds": []}
    return {"bonds": result}

# Endpoint to get distinct elements not connected to any bond
@app.get("/v1/toxicology/distinct_elements_not_connected", operation_id="get_distinct_elements_not_connected", summary="Retrieves a list of unique elements that are not involved in any bonding interactions. This operation identifies elements that are present in the system but are not connected to any other atoms, providing insights into isolated elements or standalone atoms.")
async def get_distinct_elements_not_connected():
    cursor.execute("SELECT DISTINCT T.element FROM atom AS T WHERE T.element NOT IN ( SELECT DISTINCT T1.element FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id )")
    result = cursor.fetchall()
    if not result:
        return {"elements": []}
    return {"elements": result}

# Endpoint to get atom IDs for a specific bond type and molecule ID
@app.get("/v1/toxicology/atom_ids_by_bond_type_molecule_id", operation_id="get_atom_ids_by_bond_type_molecule_id", summary="Get atom IDs for a specific bond type and molecule ID")
async def get_atom_ids_by_bond_type_molecule_id(bond_type: str = Query(..., description="Bond type"), molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT T2.atom_id, T2.atom_id2 FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id INNER JOIN bond AS T3 ON T2.bond_id = T3.bond_id WHERE T3.bond_type = ? AND T3.molecule_id = ?", (bond_type, molecule_id))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": result}

# Endpoint to get elements for a specific bond ID
@app.get("/v1/toxicology/elements_by_bond_id", operation_id="get_elements_by_bond_id", summary="Retrieves the elements associated with a specific bond ID. The bond ID is used to identify the relevant atoms, and the elements of these atoms are returned.")
async def get_elements_by_bond_id(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT T2.element FROM connected AS T1 INNER JOIN atom AS T2 ON T1.atom_id = T2.atom_id WHERE T1.bond_id = ?", (bond_id,))
    result = cursor.fetchall()
    if not result:
        return {"elements": []}
    return {"elements": result}

# Endpoint to get the molecule ID with the highest count of a specific bond type and label
@app.get("/v1/toxicology/molecule_id_highest_bond_type_count", operation_id="get_molecule_id_highest_bond_type_count", summary="Get the molecule ID with the highest count of a specific bond type and label")
async def get_molecule_id_highest_bond_type_count(label: str = Query(..., description="Molecule label"), bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT T.molecule_id FROM ( SELECT T3.molecule_id, COUNT(T1.bond_type) FROM bond AS T1 INNER JOIN molecule AS T3 ON T1.molecule_id = T3.molecule_id WHERE T3.label = ? AND T1.bond_type = ? GROUP BY T3.molecule_id ORDER BY COUNT(T1.bond_type) DESC LIMIT 1 ) AS T", (label, bond_type))
    result = cursor.fetchone()
    if not result:
        return {"molecule_id": []}
    return {"molecule_id": result[0]}

# Endpoint to get the element with the highest count of distinct molecule IDs for a specific label
@app.get("/v1/toxicology/element_highest_molecule_count", operation_id="get_element_highest_molecule_count", summary="Get the element with the highest count of distinct molecule IDs for a specific label")
async def get_element_highest_molecule_count(label: str = Query(..., description="Molecule label")):
    cursor.execute("SELECT T.element FROM ( SELECT T2.element, COUNT(DISTINCT T2.molecule_id) FROM molecule AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.label = ? GROUP BY T2.element ORDER BY COUNT(DISTINCT T2.molecule_id) LIMIT 1 ) t", (label,))
    result = cursor.fetchone()
    if not result:
        return {"element": []}
    return {"element": result[0]}

# Endpoint to get atom IDs for a specific element
@app.get("/v1/toxicology/atom_ids_by_element", operation_id="get_atom_ids_by_element", summary="Retrieves the IDs of atoms associated with a specific chemical element. The element is identified by its symbol, and the response includes pairs of atom IDs that are connected.")
async def get_atom_ids_by_element(element: str = Query(..., description="Element symbol")):
    cursor.execute("SELECT T2.atom_id, T2.atom_id2 FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T1.element = ?", (element,))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": result}

# Endpoint to get the percentage of the most common bond type
@app.get("/v1/toxicology/percentage_most_common_bond_type", operation_id="get_percentage_most_common_bond_type", summary="Retrieves the percentage of the most frequently occurring bond type in the toxicology dataset. This operation calculates the count of the most common bond type and divides it by the total count of bonds in the dataset, returning the result as a percentage.")
async def get_percentage_most_common_bond_type():
    cursor.execute("SELECT CAST((SELECT COUNT(T1.atom_id) FROM connected AS T1 INNER JOIN bond AS T2 ON T1.bond_id = T2.bond_id GROUP BY T2.bond_type ORDER BY COUNT(T2.bond_id) DESC LIMIT 1 ) AS REAL) * 100 / ( SELECT COUNT(atom_id) FROM connected )")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of bonds with a specific label and bond type
@app.get("/v1/toxicology/bond_percentage_by_label_and_type", operation_id="get_bond_percentage", summary="Get the percentage of bonds with a specific label and bond type")
async def get_bond_percentage(label: str = Query(..., description="Label of the molecule"), bond_type: str = Query(..., description="Type of the bond")):
    cursor.execute("SELECT ROUND(CAST(COUNT(CASE WHEN T2.label = ? THEN T1.bond_id ELSE NULL END) AS REAL) * 100 / COUNT(T1.bond_id),5) FROM bond AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.bond_type = ?", (label, bond_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of atoms with specific elements
@app.get("/v1/toxicology/atom_count_by_elements", operation_id="get_atom_count", summary="Retrieves the total count of atoms that contain either of the two specified elements. The elements are provided as input parameters.")
async def get_atom_count(element1: str = Query(..., description="First element"), element2: str = Query(..., description="Second element")):
    cursor.execute("SELECT COUNT(T.atom_id) FROM atom AS T WHERE T.element = ? OR T.element = ?", (element1, element2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct connected atom IDs for a specific element
@app.get("/v1/toxicology/connected_atom_ids_by_element", operation_id="get_connected_atom_ids", summary="Retrieve a unique set of atom identifiers that are connected to a specific element. The element is provided as an input parameter, allowing the operation to filter and return only the distinct atom IDs associated with the specified element.")
async def get_connected_atom_ids(element: str = Query(..., description="Element of the atom")):
    cursor.execute("SELECT DISTINCT T2.atom_id2 FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T1.element = ?", (element,))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": [row[0] for row in result]}

# Endpoint to get distinct bond types for a specific element
@app.get("/v1/toxicology/bond_types_by_element", operation_id="get_bond_types", summary="Retrieves the unique bond types associated with a specific chemical element. The operation filters the bond types based on the provided element, ensuring that only distinct bond types are returned.")
async def get_bond_types(element: str = Query(..., description="Element of the atom")):
    cursor.execute("SELECT DISTINCT T3.bond_type FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id INNER JOIN bond AS T3 ON T3.bond_id = T2.bond_id WHERE T1.element = ?", (element,))
    result = cursor.fetchall()
    if not result:
        return {"bond_types": []}
    return {"bond_types": [row[0] for row in result]}

# Endpoint to get the count of distinct elements in molecules with a specific bond type
@app.get("/v1/toxicology/distinct_elements_count_by_bond_type", operation_id="get_distinct_elements_count", summary="Retrieve the number of unique elements present in molecules that contain a specific bond type. The bond type is provided as an input parameter.")
async def get_distinct_elements_count(bond_type: str = Query(..., description="Type of the bond")):
    cursor.execute("SELECT COUNT(DISTINCT T.element) FROM ( SELECT DISTINCT T2.molecule_id, T1.element FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T3.bond_type = ? ) AS T", (bond_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of atoms with specific elements and bond type
@app.get("/v1/toxicology/atom_count_by_elements_and_bond_type", operation_id="get_atom_count_by_elements_and_bond_type", summary="Get the count of atoms with specific elements and bond type")
async def get_atom_count_by_elements_and_bond_type(bond_type: str = Query(..., description="Type of the bond"), element1: str = Query(..., description="First element"), element2: str = Query(..., description="Second element")):
    cursor.execute("SELECT COUNT(T1.atom_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T3.bond_type = ? AND T1.element IN (?, ?)", (bond_type, element1, element2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct bond IDs for molecules with a specific label
@app.get("/v1/toxicology/bond_ids_by_molecule_label", operation_id="get_bond_ids_by_label", summary="Get distinct bond IDs for molecules with a specific label")
async def get_bond_ids_by_label(label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT DISTINCT T1.bond_id FROM bond AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"bond_ids": []}
    return {"bond_ids": [row[0] for row in result]}

# Endpoint to get distinct molecule IDs for bonds with a specific label and bond type
@app.get("/v1/toxicology/molecule_ids_by_label_and_bond_type", operation_id="get_molecule_ids_by_label_and_bond_type", summary="Retrieve a unique set of molecule identifiers associated with bonds that match a specified label and bond type. This operation filters bonds based on the provided label and bond type, and returns the distinct molecule IDs linked to these bonds.")
async def get_molecule_ids_by_label_and_bond_type(label: str = Query(..., description="Label of the molecule"), bond_type: str = Query(..., description="Type of the bond")):
    cursor.execute("SELECT DISTINCT T1.molecule_id FROM bond AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ? AND T1.bond_type = ?", (label, bond_type))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids": []}
    return {"molecule_ids": [row[0] for row in result]}

# Endpoint to get the percentage of atoms with a specific element and bond type
@app.get("/v1/toxicology/atom_percentage_by_element_and_bond_type", operation_id="get_atom_percentage", summary="Retrieves the percentage of atoms in the database that have a specified element and bond type. The calculation is based on the total count of atoms with the given element and bond type, divided by the total count of atoms in the database.")
async def get_atom_percentage(element: str = Query(..., description="Element of the atom"), bond_type: str = Query(..., description="Type of the bond")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T.element = ? THEN T.atom_id ELSE NULL END) AS REAL) * 100 / COUNT(T.atom_id) FROM ( SELECT T1.atom_id, T1.element FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T3.bond_type = ? ) AS T", (element, bond_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get molecule IDs and labels for specific molecule IDs
@app.get("/v1/toxicology/molecule_info_by_ids", operation_id="get_molecule_info", summary="Retrieve the IDs and labels of up to three specific molecules. The endpoint accepts multiple molecule IDs as input and returns the corresponding molecule IDs and labels.")
async def get_molecule_info(molecule_id1: str = Query(..., description="First molecule ID"), molecule_id2: str = Query(..., description="Second molecule ID"), molecule_id3: str = Query(..., description="Third molecule ID")):
    cursor.execute("SELECT molecule_id, T.label FROM molecule AS T WHERE T.molecule_id IN (?, ?, ?)", (molecule_id1, molecule_id2, molecule_id3))
    result = cursor.fetchall()
    if not result:
        return {"molecule_info": []}
    return {"molecule_info": [{"molecule_id": row[0], "label": row[1]} for row in result]}

# Endpoint to get molecule IDs based on label
@app.get("/v1/toxicology/molecule_ids_by_label", operation_id="get_molecule_ids_by_label", summary="Retrieves the unique identifiers of molecules that match a given label. The label is used to filter the molecules and return only those that meet the specified criteria.")
async def get_molecule_ids_by_label(label: str = Query(..., description="Label of the molecule")):
    cursor.execute("SELECT T.molecule_id FROM molecule AS T WHERE T.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids": []}
    return {"molecule_ids": [row[0] for row in result]}

# Endpoint to get molecule IDs and bond types within a range
@app.get("/v1/toxicology/molecule_ids_and_bond_types_by_range", operation_id="get_molecule_ids_and_bond_types_by_range", summary="Retrieves a list of molecule IDs and their corresponding bond types for molecules within a specified range. The range is defined by a starting and ending molecule ID, both of which are inclusive. This operation is useful for obtaining detailed bond information for a specific set of molecules.")
async def get_molecule_ids_and_bond_types_by_range(start_id: str = Query(..., description="Starting molecule ID"), end_id: str = Query(..., description="Ending molecule ID")):
    cursor.execute("SELECT T2.molecule_id, T2.bond_type FROM molecule AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.molecule_id BETWEEN ? AND ?", (start_id, end_id))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids_and_bond_types": []}
    return {"molecule_ids_and_bond_types": [{"molecule_id": row[0], "bond_type": row[1]} for row in result]}

# Endpoint to get the count of bond IDs based on element
@app.get("/v1/toxicology/count_bond_ids_by_element", operation_id="get_count_bond_ids_by_element", summary="Retrieves the total number of bond IDs associated with a specified element. This operation calculates the count by joining the atom, molecule, and bond tables based on the provided element.")
async def get_count_bond_ids_by_element(element: str = Query(..., description="Element")):
    cursor.execute("SELECT COUNT(T3.bond_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T1.element = ?", (element,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common label for a specific element
@app.get("/v1/toxicology/most_common_label_by_element", operation_id="get_most_common_label_by_element", summary="Retrieves the most frequently occurring label associated with a specific element. The operation identifies the most common label by grouping and counting labels linked to the specified element, then returns the top result.")
async def get_most_common_label_by_element(element: str = Query(..., description="Element")):
    cursor.execute("SELECT T2.label FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.element = ? GROUP BY T2.label ORDER BY COUNT(T2.label) DESC LIMIT 1", (element,))
    result = cursor.fetchone()
    if not result:
        return {"label": []}
    return {"label": result[0]}

# Endpoint to get bond IDs, atom IDs, and elements based on bond ID and specific elements
@app.get("/v1/toxicology/bond_ids_atom_ids_elements_by_bond_id_and_elements", operation_id="get_bond_ids_atom_ids_elements_by_bond_id_and_elements", summary="Retrieves bond IDs, atom IDs, and the presence of specific elements (Ca or Cl) for a given bond ID. The operation filters the results based on the provided elements, returning only those that match either of the two input elements.")
async def get_bond_ids_atom_ids_elements_by_bond_id_and_elements(bond_id: str = Query(..., description="Bond ID"), element1: str = Query(..., description="First element"), element2: str = Query(..., description="Second element")):
    cursor.execute("SELECT T2.bond_id, T2.atom_id2, T1.element AS flag_have_CaCl FROM atom AS T1 INNER JOIN connected AS T2 ON T2.atom_id = T1.atom_id WHERE T2.bond_id = ? AND (T1.element = ? OR T1.element = ?)", (bond_id, element1, element2))
    result = cursor.fetchall()
    if not result:
        return {"bond_ids_atom_ids_elements": []}
    return {"bond_ids_atom_ids_elements": [{"bond_id": row[0], "atom_id2": row[1], "element": row[2]} for row in result]}

# Endpoint to get distinct molecule IDs based on bond type, element, and label
@app.get("/v1/toxicology/distinct_molecule_ids_by_bond_type_element_label", operation_id="get_distinct_molecule_ids_by_bond_type_element_label", summary="Get distinct molecule IDs based on a specific bond type, element, and label")
async def get_distinct_molecule_ids_by_bond_type_element_label(bond_type: str = Query(..., description="Bond type"), element: str = Query(..., description="Element"), label: str = Query(..., description="Label")):
    cursor.execute("SELECT DISTINCT T2.molecule_id FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T3.bond_type = ? AND T1.element = ? AND T2.label = ?", (bond_type, element, label))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids": []}
    return {"molecule_ids": [row[0] for row in result]}

# Endpoint to get the percentage of a specific element in molecules with a specific label
@app.get("/v1/toxicology/percentage_element_in_molecules_by_label", operation_id="get_percentage_element_in_molecules_by_label", summary="Get the percentage of a specific element in molecules with a specific label")
async def get_percentage_element_in_molecules_by_label(element: str = Query(..., description="Element"), label: str = Query(..., description="Label")):
    cursor.execute("SELECT CAST(COUNT( CASE WHEN T1.element = ? THEN T1.element ELSE NULL END) AS REAL) * 100 / COUNT(T1.element) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ?", (element, label))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the label of molecules based on bond ID
@app.get("/v1/toxicology/molecule_label_by_bond_id", operation_id="get_molecule_label_by_bond_id", summary="Retrieves the label of a molecule associated with a given bond ID. The bond ID is used to identify the specific bond in the database, which is then linked to its corresponding molecule to extract the label.")
async def get_molecule_label_by_bond_id(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT T2.label FROM bond AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.bond_id = ?", (bond_id,))
    result = cursor.fetchall()
    if not result:
        return {"labels": []}
    return {"labels": [row[0] for row in result]}

# Endpoint to get distinct bond IDs and labels based on bond type
@app.get("/v1/toxicology/distinct_bond_ids_and_labels_by_bond_type", operation_id="get_distinct_bond_ids_and_labels", summary="Retrieves a unique set of bond identifiers and their corresponding labels for a specified bond type. This operation filters bonds based on the provided bond type and returns the distinct bond IDs along with their associated labels from the molecule table.")
async def get_distinct_bond_ids_and_labels(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT DISTINCT T1.bond_id, T2.label FROM bond AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.bond_type = ?", (bond_type,))
    result = cursor.fetchall()
    if not result:
        return {"bond_ids_and_labels": []}
    return {"bond_ids_and_labels": [{"bond_id": row[0], "label": row[1]} for row in result]}

# Endpoint to get the ratio of a specific element in a molecule
@app.get("/v1/toxicology/element_ratio_in_molecule", operation_id="get_element_ratio", summary="Retrieves the proportion of a specified element within a given molecule. The operation calculates the ratio of the specified element to the total number of atoms in the molecule, grouped by the molecule's label. The molecule is identified by its unique ID, and the element is specified by its symbol.")
async def get_element_ratio(molecule_id: str = Query(..., description="Molecule ID"), element: str = Query(..., description="Element")):
    cursor.execute("WITH SubQuery AS (SELECT DISTINCT T1.atom_id, T1.element, T1.molecule_id, T2.label FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.molecule_id = ?) SELECT CAST(COUNT(CASE WHEN element = ? THEN atom_id ELSE NULL END) AS REAL) / (CASE WHEN COUNT(atom_id) = 0 THEN NULL ELSE COUNT(atom_id) END) AS ratio, label FROM SubQuery GROUP BY label", (molecule_id, element))
    result = cursor.fetchall()
    if not result:
        return {"ratios": []}
    return {"ratios": [{"ratio": row[0], "label": row[1]} for row in result]}

# Endpoint to get the carcinogenic flag based on element
@app.get("/v1/toxicology/carcinogenic_flag_by_element", operation_id="get_carcinogenic_flag", summary="Retrieves the carcinogenic flag associated with a specific element. This operation fetches the carcinogenic flag from the molecule table based on the provided element. The element parameter is used to filter the data and return the corresponding carcinogenic flag.")
async def get_carcinogenic_flag(element: str = Query(..., description="Element")):
    cursor.execute("SELECT T2.label AS flag_carcinogenic FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.element = ?", (element,))
    result = cursor.fetchall()
    if not result:
        return {"flags": []}
    return {"flags": [row[0] for row in result]}

# Endpoint to get elements of atoms connected by a specific bond ID
@app.get("/v1/toxicology/elements_by_bond_id_connected", operation_id="get_elements_by_bond_id_connected", summary="Retrieves the elements of atoms that are connected by a specific bond. The bond is identified by its unique bond_id. The response includes the element information for the atoms linked by the specified bond.")
async def get_elements_by_bond_id_connected(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT T1.element FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id INNER JOIN bond AS T3 ON T2.bond_id = T3.bond_id WHERE T3.bond_id = ?", (bond_id,))
    result = cursor.fetchall()
    if not result:
        return {"elements": []}
    return {"elements": [row[0] for row in result]}

# Endpoint to get the percentage of bonds of a specific type for a given molecule
@app.get("/v1/toxicology/bond_type_percentage_molecule", operation_id="get_bond_type_percentage_molecule", summary="Retrieves the percentage of a specific bond type for a given molecule. The operation calculates the ratio of bonds of the specified type to the total number of bonds in the molecule, expressed as a percentage. The bond type and molecule ID are required input parameters.")
async def get_bond_type_percentage_molecule(bond_type: str = Query(..., description="Type of bond (e.g., '=')"), molecule_id: str = Query(..., description="Molecule ID (e.g., 'TR047')")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T.bond_type = ? THEN T.bond_id ELSE NULL END) AS REAL) * 100 / COUNT(T.bond_id) FROM bond AS T WHERE T.molecule_id = ?", (bond_type, molecule_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the carcinogenic flag for a specific atom
@app.get("/v1/toxicology/carcinogenic_flag_atom", operation_id="get_carcinogenic_flag_atom", summary="Retrieves the carcinogenic flag associated with a specific atom. The flag indicates whether the atom is carcinogenic or not. The atom is identified by its unique ID, which is provided as an input parameter.")
async def get_carcinogenic_flag_atom(atom_id: str = Query(..., description="Atom ID (e.g., 'TR001_1')")):
    cursor.execute("SELECT T2.label AS flag_carcinogenic FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.atom_id = ?", (atom_id,))
    result = cursor.fetchone()
    if not result:
        return {"flag_carcinogenic": []}
    return {"flag_carcinogenic": result[0]}

# Endpoint to get the label of a specific molecule
@app.get("/v1/toxicology/molecule_label", operation_id="get_molecule_label", summary="Retrieves the label of a specific molecule identified by its unique ID. The molecule ID is a string that represents the molecule, such as 'TR151'. This operation returns the label associated with the provided molecule ID.")
async def get_molecule_label(molecule_id: str = Query(..., description="Molecule ID (e.g., 'TR151')")):
    cursor.execute("SELECT T.label FROM molecule AS T WHERE T.molecule_id = ?", (molecule_id,))
    result = cursor.fetchone()
    if not result:
        return {"label": []}
    return {"label": result[0]}

# Endpoint to get atom IDs within a range of molecule IDs and a specific element
@app.get("/v1/toxicology/atom_ids_molecule_range_element", operation_id="get_atom_ids_molecule_range_element", summary="Retrieves the atom IDs for a specified element that are associated with molecule IDs within a given range. The range is defined by the start and end molecule IDs. This operation is useful for obtaining atom IDs that meet specific criteria, such as a particular element and a range of molecule IDs.")
async def get_atom_ids_molecule_range_element(start_molecule_id: str = Query(..., description="Start molecule ID (e.g., 'TR010')"), end_molecule_id: str = Query(..., description="End molecule ID (e.g., 'TR050')"), element: str = Query(..., description="Element (e.g., 'c')")):
    cursor.execute("SELECT T.atom_id FROM atom AS T WHERE T.molecule_id BETWEEN ? AND ? AND T.element = ?", (start_molecule_id, end_molecule_id, element))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": [row[0] for row in result]}

# Endpoint to get the count of atoms in molecules with a specific label
@app.get("/v1/toxicology/atom_count_molecule_label", operation_id="get_atom_count_molecule_label", summary="Retrieves the total count of atoms present in molecules that have a specified label. The label is a unique identifier for a molecule, such as '+'.")
async def get_atom_count_molecule_label(label: str = Query(..., description="Label of the molecule (e.g., '+')")):
    cursor.execute("SELECT COUNT(T1.atom_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ?", (label,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get bond IDs in molecules with a specific label and bond type
@app.get("/v1/toxicology/bond_ids_molecule_label_bond_type", operation_id="get_bond_ids_molecule_label_bond_type", summary="Retrieves the bond IDs of molecules with a specified label and bond type. The endpoint filters molecules based on the provided label and bond type, then returns the bond IDs of the matching molecules.")
async def get_bond_ids_molecule_label_bond_type(label: str = Query(..., description="Label of the molecule (e.g., '+')"), bond_type: str = Query(..., description="Type of bond (e.g., '=')")):
    cursor.execute("SELECT T1.bond_id FROM bond AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ? AND T1.bond_type = ?", (label, bond_type))
    result = cursor.fetchall()
    if not result:
        return {"bond_ids": []}
    return {"bond_ids": [row[0] for row in result]}

# Endpoint to get the count of hydrogen atoms in molecules with a specific label
@app.get("/v1/toxicology/hydrogen_atom_count_molecule_label", operation_id="get_hydrogen_atom_count_molecule_label", summary="Retrieves the total count of hydrogen atoms present in molecules that match a specified label. The label and element parameters are used to filter the molecules and atoms, respectively.")
async def get_hydrogen_atom_count_molecule_label(label: str = Query(..., description="Label of the molecule (e.g., '+')"), element: str = Query(..., description="Element (e.g., 'h')")):
    cursor.execute("SELECT COUNT(T1.atom_id) AS atomnums_h FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ? AND T1.element = ?", (label, element))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get molecule and bond details based on atom and bond IDs
@app.get("/v1/toxicology/molecule_bond_details", operation_id="get_molecule_bond_details", summary="Retrieves detailed information about a specific molecule and its bond, based on the provided atom and bond IDs. The response includes the molecule ID, bond ID, and the atom ID associated with the bond.")
async def get_molecule_bond_details(atom_id: str = Query(..., description="Atom ID"), bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT T2.molecule_id, T2.bond_id, T1.atom_id FROM connected AS T1 INNER JOIN bond AS T2 ON T1.bond_id = T2.bond_id WHERE T1.atom_id = ? AND T2.bond_id = ?", (atom_id, bond_id))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get atom IDs based on element and molecule label
@app.get("/v1/toxicology/atom_ids_by_element_label", operation_id="get_atom_ids_by_element_label", summary="Retrieves the unique identifiers of atoms in a molecule that match a specified element symbol. The molecule is identified by its label.")
async def get_atom_ids_by_element_label(element: str = Query(..., description="Element symbol"), label: str = Query(..., description="Molecule label")):
    cursor.execute("SELECT T1.atom_id FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.element = ? AND T2.label = ?", (element, label))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": result}

# Endpoint to get the percentage of molecules with a specific element and label
@app.get("/v1/toxicology/percentage_molecules_by_element_label", operation_id="get_percentage_molecules_by_element_label", summary="Get the percentage of molecules with a specific element and label")
async def get_percentage_molecules_by_element_label(element: str = Query(..., description="Element symbol"), label: str = Query(..., description="Molecule label")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.element = ? AND T2.label = ? THEN T2.molecule_id ELSE NULL END) AS REAL) * 100 / COUNT(T2.molecule_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id", (element, label))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get bond type based on bond ID
@app.get("/v1/toxicology/bond_type", operation_id="get_bond_type", summary="Retrieves the type of a specific bond identified by its unique bond ID. The bond type is a characteristic that describes the nature of the bond.")
async def get_bond_type(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT T.bond_type FROM bond AS T WHERE T.bond_id = ?", (bond_id,))
    result = cursor.fetchone()
    if not result:
        return {"bond_type": []}
    return {"bond_type": result[0]}

# Endpoint to get the count of bonds and molecule labels based on bond type and molecule ID
@app.get("/v1/toxicology/bond_count_by_type_molecule", operation_id="get_bond_count_by_type_molecule", summary="Retrieves the count of bonds and associated molecule labels, grouped by the molecule label, for a specific bond type and molecule ID.")
async def get_bond_count_by_type_molecule(bond_type: str = Query(..., description="Bond type"), molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT COUNT(T1.bond_id), T2.label FROM bond AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.bond_type = ? AND T2.molecule_id = ? GROUP BY T2.label", (bond_type, molecule_id))
    result = cursor.fetchall()
    if not result:
        return {"bond_counts": []}
    return {"bond_counts": result}

# Endpoint to get distinct molecule IDs and elements based on molecule label
@app.get("/v1/toxicology/distinct_molecules_elements_by_label", operation_id="get_distinct_molecules_elements_by_label", summary="Get distinct molecule IDs and elements based on molecule label")
async def get_distinct_molecules_elements_by_label(label: str = Query(..., description="Molecule label")):
    cursor.execute("SELECT DISTINCT T2.molecule_id, T1.element FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ?", (label,))
    result = cursor.fetchall()
    if not result:
        return {"molecule_elements": []}
    return {"molecule_elements": result}

# Endpoint to get bond and connected atom details based on bond type
@app.get("/v1/toxicology/bond_connected_atoms", operation_id="get_bond_connected_atoms", summary="Retrieves the bond IDs and associated atom IDs for a specified bond type. This operation returns data from the bond and connected tables, filtering results based on the provided bond type.")
async def get_bond_connected_atoms(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT T1.bond_id, T2.atom_id, T2.atom_id2 FROM bond AS T1 INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id WHERE T1.bond_type = ?", (bond_type,))
    result = cursor.fetchall()
    if not result:
        return {"bond_atoms": []}
    return {"bond_atoms": result}

# Endpoint to get distinct molecule IDs and elements based on bond type
@app.get("/v1/toxicology/distinct_molecule_elements_by_bond_type", operation_id="get_distinct_molecule_elements", summary="Retrieves a list of unique molecule IDs and their associated elements based on a specified bond type. This operation provides a distinct set of molecule IDs and elements that share the same bond type, enabling users to identify and analyze molecules with specific bond characteristics.")
async def get_distinct_molecule_elements(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT DISTINCT T1.molecule_id, T2.element FROM bond AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.bond_type = ?", (bond_type,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of bonds based on element
@app.get("/v1/toxicology/bond_count_by_element", operation_id="get_bond_count_by_element", summary="Retrieves the total count of bonds associated with a specified chemical element. The element is used to filter the bonds, providing a precise count of its occurrences in the bond data.")
async def get_bond_count_by_element(element: str = Query(..., description="Element")):
    cursor.execute("SELECT COUNT(T1.bond_id) FROM connected AS T1 INNER JOIN atom AS T2 ON T1.atom_id = T2.atom_id WHERE T2.element = ?", (element,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get atom IDs, distinct bond types count, and molecule IDs based on molecule ID
@app.get("/v1/toxicology/atom_bond_count_by_molecule_id", operation_id="get_atom_bond_count_by_molecule_id", summary="Retrieves the distinct bond types count and associated atom IDs for a given molecule ID. The molecule ID is used to filter the results, ensuring that only relevant data is returned. This operation provides a comprehensive overview of the atom-bond relationships within a specific molecule.")
async def get_atom_bond_count_by_molecule_id(molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT T1.atom_id, COUNT(DISTINCT T2.bond_type), T1.molecule_id FROM atom AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.molecule_id = ? GROUP BY T1.atom_id, T2.bond_type", (molecule_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of distinct molecule IDs and sum of positive labels based on bond type
@app.get("/v1/toxicology/molecule_count_positive_labels_by_bond_type", operation_id="get_molecule_count_positive_labels_by_bond_type", summary="Retrieves the total count of unique molecules and the sum of their positive labels for a specified bond type. This operation provides a quantitative overview of the prevalence and positive labeling of molecules based on their bond type.")
async def get_molecule_count_positive_labels_by_bond_type(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT COUNT(DISTINCT T2.molecule_id), SUM(CASE WHEN T2.label = '+' THEN 1 ELSE 0 END) FROM bond AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.bond_type = ?", (bond_type,))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of distinct molecule IDs based on element and bond type
@app.get("/v1/toxicology/molecule_count_by_element_bond_type", operation_id="get_molecule_count_by_element_bond_type", summary="Retrieves the number of unique molecules that contain a specific element and bond type. The element and bond type are provided as input parameters, allowing for a targeted count of distinct molecule IDs.")
async def get_molecule_count_by_element_bond_type(element: str = Query(..., description="Element"), bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT COUNT(DISTINCT T1.molecule_id) FROM atom AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.element <> ? AND T2.bond_type <> ?", (element, bond_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct labels based on bond ID
@app.get("/v1/toxicology/distinct_labels_by_bond_id", operation_id="get_distinct_labels_by_bond_id", summary="Retrieves unique labels associated with a specific bond ID. This operation identifies distinct labels from the 'molecule' table that correspond to the provided bond ID. The bond ID is used to filter the 'bond' table, which is then joined with the 'atom' and 'molecule' tables to extract the relevant labels.")
async def get_distinct_labels_by_bond_id(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT DISTINCT T2.label FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T3.bond_id = ?", (bond_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of atoms based on molecule ID
@app.get("/v1/toxicology/atom_count_by_molecule_id", operation_id="get_atom_count_by_molecule_id", summary="Retrieves the total number of atoms associated with a specific molecule, identified by its unique molecule ID.")
async def get_atom_count_by_molecule_id(molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT COUNT(T.atom_id) FROM atom AS T WHERE T.molecule_id = ?", (molecule_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct molecule IDs based on element and label
@app.get("/v1/toxicology/distinct_molecule_ids_by_element_label", operation_id="get_distinct_molecule_ids_by_element_label", summary="Retrieve a unique set of molecule identifiers that match a specified element and label. This operation filters the atom table based on the provided element and then joins it with the molecule table using the molecule_id. The result is a list of distinct molecule_ids that correspond to the given element and label.")
async def get_distinct_molecule_ids_by_element_label(element: str = Query(..., description="Element"), label: str = Query(..., description="Label")):
    cursor.execute("SELECT DISTINCT T1.molecule_id FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.element = ? AND T2.label = ?", (element, label))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of molecules with a specific label and element
@app.get("/v1/toxicology/percentage_molecules_by_label_element", operation_id="get_percentage_molecules", summary="Get the percentage of molecules with a specific label and element")
async def get_percentage_molecules(label: str = Query(..., description="Label of the molecule"), element: str = Query(..., description="Element of the atom")):
    cursor.execute("SELECT COUNT(CASE WHEN T2.label = ? AND T1.element = ? THEN T2.molecule_id ELSE NULL END) * 100 / COUNT(T2.molecule_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id", (label, element))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct molecule IDs based on bond ID
@app.get("/v1/toxicology/distinct_molecule_ids_by_bond_id", operation_id="get_distinct_molecule_ids_by_bond", summary="Retrieves a unique set of molecule IDs associated with a specific bond ID. This operation filters the atom table based on the provided bond ID and returns the distinct molecule IDs linked to the filtered atoms.")
async def get_distinct_molecule_ids_by_bond(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT DISTINCT T1.molecule_id FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T2.bond_id = ?", (bond_id,))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids": []}
    return {"molecule_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct elements based on bond ID
@app.get("/v1/toxicology/count_distinct_elements_by_bond_id", operation_id="get_count_distinct_elements_by_bond", summary="Retrieves the total count of unique elements associated with a specific bond ID. The bond ID is used to filter the elements from the atom table, which are then counted after removing any duplicates.")
async def get_count_distinct_elements_by_bond(bond_id: str = Query(..., description="Bond ID")):
    cursor.execute("SELECT COUNT(DISTINCT T1.element) FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T2.bond_id = ?", (bond_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get bond type based on atom IDs
@app.get("/v1/toxicology/bond_type_by_atom_ids", operation_id="get_bond_type_by_atom_ids", summary="Retrieves the bond type between two atoms identified by their respective atom IDs. The operation returns the type of bond that connects the two specified atoms.")
async def get_bond_type_by_atom_ids(atom_id: str = Query(..., description="Atom ID"), atom_id2: str = Query(..., description="Second Atom ID")):
    cursor.execute("SELECT T1.bond_type FROM bond AS T1 INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id WHERE T2.atom_id = ? AND T2.atom_id2 = ?", (atom_id, atom_id2))
    result = cursor.fetchall()
    if not result:
        return {"bond_types": []}
    return {"bond_types": [row[0] for row in result]}

# Endpoint to get molecule IDs based on atom IDs
@app.get("/v1/toxicology/molecule_ids_by_atom_ids", operation_id="get_molecule_ids_by_atom_ids", summary="Retrieves the unique identifiers of molecules that contain the specified pair of atoms. The operation identifies molecules by examining the bonds between atoms and returns the molecule IDs that meet the criteria.")
async def get_molecule_ids_by_atom_ids(atom_id: str = Query(..., description="Atom ID"), atom_id2: str = Query(..., description="Second Atom ID")):
    cursor.execute("SELECT T1.molecule_id FROM bond AS T1 INNER JOIN connected AS T2 ON T1.bond_id = T2.bond_id WHERE T2.atom_id = ? AND T2.atom_id2 = ?", (atom_id, atom_id2))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids": []}
    return {"molecule_ids": [row[0] for row in result]}

# Endpoint to get the element of an atom based on atom ID
@app.get("/v1/toxicology/element_by_atom_id", operation_id="get_element_by_atom_id", summary="Retrieves the element associated with a specific atom, identified by its unique atom_id. This operation provides detailed information about the element of the atom, enabling users to understand its properties and characteristics.")
async def get_element_by_atom_id(atom_id: str = Query(..., description="Atom ID")):
    cursor.execute("SELECT T.element FROM atom AS T WHERE T.atom_id = ?", (atom_id,))
    result = cursor.fetchone()
    if not result:
        return {"element": []}
    return {"element": result[0]}

# Endpoint to get the label of a molecule based on molecule ID
@app.get("/v1/toxicology/label_by_molecule_id", operation_id="get_label_by_molecule_id", summary="Retrieves the label associated with a specific molecule, identified by its unique molecule ID. This operation returns the label of the molecule, providing essential information about its properties or characteristics.")
async def get_label_by_molecule_id(molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT label FROM molecule AS T WHERE T.molecule_id = ?", (molecule_id,))
    result = cursor.fetchone()
    if not result:
        return {"label": []}
    return {"label": result[0]}

# Endpoint to get the percentage of bonds with a specific bond type
@app.get("/v1/toxicology/percentage_bonds_by_type", operation_id="get_percentage_bonds_by_type", summary="Retrieves the percentage of bonds of a specified type from the total number of bonds. The bond type is provided as an input parameter.")
async def get_percentage_bonds_by_type(bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T.bond_type = ? THEN T.bond_id ELSE NULL END) AS REAL) * 100 / COUNT(T.bond_id) FROM bond t", (bond_type,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get molecule IDs with a specific label and more than a certain number of atoms
@app.get("/v1/toxicology/molecule_ids_by_label_and_atom_count", operation_id="get_molecule_ids_by_label_and_atom_count", summary="Retrieve the identifiers of molecules that possess a specified label and contain more than a given number of atoms. The response includes molecule IDs that meet the provided label and atom count criteria.")
async def get_molecule_ids_by_label_and_atom_count(label: str = Query(..., description="Molecule label"), min_atom_count: int = Query(..., description="Minimum number of atoms")):
    cursor.execute("SELECT T.molecule_id FROM ( SELECT T1.molecule_id, COUNT(T2.atom_id) FROM molecule AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.label = ? GROUP BY T1.molecule_id HAVING COUNT(T2.atom_id) > ? ) t", (label, min_atom_count))
    result = cursor.fetchall()
    if not result:
        return {"molecule_ids": []}
    return {"molecule_ids": [row[0] for row in result]}

# Endpoint to get elements of atoms in a specific molecule with a specific bond type
@app.get("/v1/toxicology/elements_by_molecule_id_and_bond_type", operation_id="get_elements_by_molecule_id_and_bond_type", summary="Retrieves the elements of atoms within a specific molecule that have a particular bond type. The molecule is identified by its unique ID, and the bond type is specified to filter the results. This operation returns a list of elements that meet the provided criteria.")
async def get_elements_by_molecule_id_and_bond_type(molecule_id: str = Query(..., description="Molecule ID"), bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT T1.element FROM atom AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.molecule_id = ? AND T2.bond_type = ?", (molecule_id, bond_type))
    result = cursor.fetchall()
    if not result:
        return {"elements": []}
    return {"elements": [row[0] for row in result]}

# Endpoint to get the molecule ID with the highest number of atoms for a specific label
@app.get("/v1/toxicology/molecule_id_with_max_atoms_by_label", operation_id="get_molecule_id_with_max_atoms_by_label", summary="Get the molecule ID with the highest number of atoms for a specific label")
async def get_molecule_id_with_max_atoms_by_label(label: str = Query(..., description="Molecule label")):
    cursor.execute("SELECT T.molecule_id FROM ( SELECT T2.molecule_id, COUNT(T1.atom_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = ? GROUP BY T2.molecule_id ORDER BY COUNT(T1.atom_id) DESC LIMIT 1 ) t", (label,))
    result = cursor.fetchone()
    if not result:
        return {"molecule_id": []}
    return {"molecule_id": result[0]}

# Endpoint to get the percentage of molecules with a specific label, bond type, and element
@app.get("/v1/toxicology/percentage_molecules_by_label_bond_type_element", operation_id="get_percentage_molecules_by_label_bond_type_element", summary="Get the percentage of molecules with a specific label, bond type, and element")
async def get_percentage_molecules_by_label_bond_type_element(label: str = Query(..., description="Molecule label"), bond_type: str = Query(..., description="Bond type"), element: str = Query(..., description="Element symbol")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.label = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(DISTINCT T1.molecule_id) FROM molecule AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T1.molecule_id = T3.molecule_id WHERE T3.bond_type = ? AND T2.element = ?", (label, bond_type, element))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of distinct molecule IDs within a range and with a specific bond type
@app.get("/v1/toxicology/count_distinct_molecule_ids_by_range_and_bond_type", operation_id="get_count_distinct_molecule_ids_by_range_and_bond_type", summary="Retrieves the count of unique molecule identifiers that fall within a specified range and possess a particular bond type. The range is defined by a start and end molecule ID, and the bond type is specified as an input parameter.")
async def get_count_distinct_molecule_ids_by_range_and_bond_type(start_molecule_id: str = Query(..., description="Start molecule ID"), end_molecule_id: str = Query(..., description="End molecule ID"), bond_type: str = Query(..., description="Bond type")):
    cursor.execute("SELECT COUNT(DISTINCT T.molecule_id) FROM bond AS T WHERE T.molecule_id BETWEEN ? AND ? AND T.bond_type = ?", (start_molecule_id, end_molecule_id, bond_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of atoms in a specific molecule with a specific element
@app.get("/v1/toxicology/count_atoms_by_molecule_id_and_element", operation_id="get_count_atoms_by_molecule_id_and_element", summary="Retrieves the total number of occurrences of a specific element within a given molecule. The molecule is identified by its unique ID, and the element is specified using its symbol. This operation provides a quantitative measure of the element's presence in the molecule.")
async def get_count_atoms_by_molecule_id_and_element(molecule_id: str = Query(..., description="Molecule ID"), element: str = Query(..., description="Element symbol")):
    cursor.execute("SELECT COUNT(T.atom_id) FROM atom AS T WHERE T.molecule_id = ? AND T.element = ?", (molecule_id, element))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the element of a specific atom in a molecule with a specific label
@app.get("/v1/toxicology/element_by_atom_id_and_molecule_label", operation_id="get_element_by_atom_id_and_molecule_label", summary="Retrieves the element associated with a specific atom in a molecule identified by its label. The operation requires the atom's unique identifier and the molecule's label as input parameters.")
async def get_element_by_atom_id_and_molecule_label(atom_id: str = Query(..., description="Atom ID"), label: str = Query(..., description="Molecule label")):
    cursor.execute("SELECT T1.element FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.atom_id = ? AND T2.label = ?", (atom_id, label))
    result = cursor.fetchone()
    if not result:
        return {"element": []}
    return {"element": result[0]}

# Endpoint to get the count of distinct molecule IDs with a specific bond type and element
@app.get("/v1/toxicology/count_distinct_molecule_ids_by_bond_type_and_element", operation_id="get_count_distinct_molecule_ids_by_bond_type_and_element", summary="Retrieves the count of unique molecules that have a specified bond type and element. The bond type and element symbol are provided as input parameters.")
async def get_count_distinct_molecule_ids_by_bond_type_and_element(bond_type: str = Query(..., description="Bond type"), element: str = Query(..., description="Element symbol")):
    cursor.execute("SELECT COUNT(DISTINCT T1.molecule_id) FROM atom AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.bond_type = ? AND T1.element = ?", (bond_type, element))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct molecule IDs with a specific bond type and label
@app.get("/v1/toxicology/count_distinct_molecule_ids_by_bond_type_and_label", operation_id="get_count_distinct_molecule_ids_by_bond_type_and_label", summary="Retrieve the count of unique molecules that have a specified bond type and label. This operation filters molecules based on the provided bond type and label, then returns the count of distinct molecule IDs that meet these criteria.")
async def get_count_distinct_molecule_ids_by_bond_type_and_label(bond_type: str = Query(..., description="Bond type"), label: str = Query(..., description="Molecule label")):
    cursor.execute("SELECT COUNT(DISTINCT T1.molecule_id) FROM molecule AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.bond_type = ? AND T1.label = ?", (bond_type, label))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct elements and bond types for a given molecule ID
@app.get("/v1/toxicology/distinct_elements_bond_types", operation_id="get_distinct_elements_bond_types", summary="Retrieves the unique combinations of elements and bond types associated with a specific molecule. The molecule is identified by its unique ID.")
async def get_distinct_elements_bond_types(molecule_id: str = Query(..., description="Molecule ID")):
    cursor.execute("SELECT DISTINCT T1.element, T2.bond_type FROM atom AS T1 INNER JOIN bond AS T2 ON T1.molecule_id = T2.molecule_id WHERE T1.molecule_id = ?", (molecule_id,))
    result = cursor.fetchall()
    if not result:
        return {"elements_bond_types": []}
    return {"elements_bond_types": result}

# Endpoint to get atom IDs based on molecule ID, bond type, and element
@app.get("/v1/toxicology/atom_ids_by_molecule_bond_element", operation_id="get_atom_ids_by_molecule_bond_element", summary="Retrieves the atom IDs associated with a specific molecule, bond type, and element. This operation filters the atoms based on the provided molecule ID, bond type, and element, and returns the corresponding atom IDs.")
async def get_atom_ids_by_molecule_bond_element(molecule_id: str = Query(..., description="Molecule ID"), bond_type: str = Query(..., description="Bond type"), element: str = Query(..., description="Element")):
    cursor.execute("SELECT T1.atom_id FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T2.molecule_id = ? AND T3.bond_type = ? AND T1.element = ?", (molecule_id, bond_type, element))
    result = cursor.fetchall()
    if not result:
        return {"atom_ids": []}
    return {"atom_ids": result}

api_calls = [
    "/v1/toxicology/most_common_bond_type",
    "/v1/toxicology/count_distinct_molecules_by_element_label?element=cl&label=-",
    "/v1/toxicology/avg_oxygen_count_by_bond_type_element?bond_type=-&element=o",
    "/v1/toxicology/avg_single_bond_count_by_bond_type_label?bond_type=-&label=+",
    "/v1/toxicology/distinct_molecule_ids_by_bond_type_label?bond_type=#&label=+",
    "/v1/toxicology/percentage_atoms_by_element_bond_type?element=c&bond_type=",
    "/v1/toxicology/count_bonds_by_bond_type?bond_type=#",
    "/v1/toxicology/count_distinct_atoms_excluding_element?element=br",
    "/v1/toxicology/count_molecules_by_id_range_label?min_id=TR000&max_id=TR099&label=+",
    "/v1/toxicology/molecule_ids_by_element?element=c",
    "/v1/toxicology/distinct_elements_by_bond_id?bond_id=TR004_8_9",
    "/v1/toxicology/distinct_elements_by_bond_type?bond_type==",
    "/v1/toxicology/most_common_molecule_label_by_element?element=h",
    "/v1/toxicology/distinct_bond_types_by_element?element=cl",
    "/v1/toxicology/atom_ids_by_bond_type?bond_type=-",
    "/v1/toxicology/distinct_atom_ids_by_molecule_label?label=-",
    "/v1/toxicology/least_common_element_by_molecule_label?label=-",
    "/v1/toxicology/bond_types_by_atom_ids?atom_id1=TR004_8&atom_id2=TR004_20",
    "/v1/toxicology/distinct_molecule_labels_by_element_exclusion?element=sn",
    "/v1/toxicology/count_iodine_sulfur_bond_type?element1=i&element2=s&bond_type=-",
    "/v1/toxicology/atom_ids_by_molecule?molecule_id=TR181",
    "/v1/toxicology/percentage_molecules_element_label?element=f&label=+",
    "/v1/toxicology/percentage_molecules_label_bond_type?label=+&bond_type=#",
    "/v1/toxicology/distinct_elements_by_molecule?molecule_id=TR000&limit=3",
    "/v1/toxicology/atom_ids_from_bond?molecule_id=TR001&bond_id=TR001_2_6",
    "/v1/toxicology/diff_molecule_count_by_label?label1=+&label2=-",
    "/v1/toxicology/atom_ids_by_bond?bond_id=TR000_2_5",
    "/v1/toxicology/bond_ids_by_atom?atom_id2=TR000_2",
    "/v1/toxicology/bond_type_percentage?bond_type=%3D&molecule_id=TR008",
    "/v1/toxicology/molecule_label_percentage?label=%2B",
    "/v1/toxicology/atom_element_percentage?element=h&molecule_id=TR206",
    "/v1/toxicology/distinct_bond_types?molecule_id=TR000",
    "/v1/toxicology/distinct_elements_labels?molecule_id=TR060",
    "/v1/toxicology/distinct_bond_ids_by_molecule?molecule_id=TR006&limit=2",
    "/v1/toxicology/bond_count_by_molecule_atom_ids?molecule_id=TR009&atom_id=TR009_1&atom_id2=TR009_2",
    "/v1/toxicology/count_distinct_molecule_ids?label=+&element=br",
    "/v1/toxicology/get_bond_type_atom_ids?bond_id=TR001_6_9",
    "/v1/toxicology/get_molecule_ids_carcinogenic_flag?atom_id=TR001_10&label=+",
    "/v1/toxicology/count_distinct_molecule_ids_bond_type?bond_type=#",
    "/v1/toxicology/count_bond_ids_atom_id_suffix?atom_id_suffix=19",
    "/v1/toxicology/get_distinct_elements?molecule_id=TR004",
    "/v1/toxicology/count_molecule_ids_label?label=-",
    "/v1/toxicology/get_distinct_molecule_ids_atom_id_range_label?start_suffix=21&end_suffix=25&label=+",
    "/v1/toxicology/get_bond_ids_connected_elements?element1=p&element2=n",
    "/v1/toxicology/get_molecule_label_highest_bond_type_count?bond_type=",
    "/v1/toxicology/bond_atom_ratio?element=i",
    "/v1/toxicology/bond_type_id_by_atom_id_substring?atom_id_substring=45",
    "/v1/toxicology/distinct_elements_not_connected",
    "/v1/toxicology/atom_ids_by_bond_type_molecule_id?bond_type=#&molecule_id=TR041",
    "/v1/toxicology/elements_by_bond_id?bond_id=TR144_8_19",
    "/v1/toxicology/molecule_id_highest_bond_type_count?label=+&bond_type=",
    "/v1/toxicology/element_highest_molecule_count?label=+",
    "/v1/toxicology/atom_ids_by_element?element=pb",
    "/v1/toxicology/percentage_most_common_bond_type",
    "/v1/toxicology/bond_percentage_by_label_and_type?label=+&bond_type=-",
    "/v1/toxicology/atom_count_by_elements?element1=c&element2=h",
    "/v1/toxicology/connected_atom_ids_by_element?element=s",
    "/v1/toxicology/bond_types_by_element?element=sn",
    "/v1/toxicology/distinct_elements_count_by_bond_type?bond_type=-",
    "/v1/toxicology/atom_count_by_elements_and_bond_type?bond_type=#&element1=p&element2=br",
    "/v1/toxicology/bond_ids_by_molecule_label?label=+",
    "/v1/toxicology/molecule_ids_by_label_and_bond_type?label=-&bond_type=-",
    "/v1/toxicology/atom_percentage_by_element_and_bond_type?element=cl&bond_type=-",
    "/v1/toxicology/molecule_info_by_ids?molecule_id1=TR000&molecule_id2=TR001&molecule_id3=TR002",
    "/v1/toxicology/molecule_ids_by_label?label=-",
    "/v1/toxicology/molecule_ids_and_bond_types_by_range?start_id=TR000&end_id=TR050",
    "/v1/toxicology/count_bond_ids_by_element?element=i",
    "/v1/toxicology/most_common_label_by_element?element=ca",
    "/v1/toxicology/bond_ids_atom_ids_elements_by_bond_id_and_elements?bond_id=TR001_1_8&element1=c1&element2=c",
    "/v1/toxicology/distinct_molecule_ids_by_bond_type_element_label?bond_type=#&element=c&label=-",
    "/v1/toxicology/percentage_element_in_molecules_by_label?element=cl&label=+",
    "/v1/toxicology/molecule_label_by_bond_id?bond_id=TR001_10_11",
    "/v1/toxicology/distinct_bond_ids_and_labels_by_bond_type?bond_type=%23",
    "/v1/toxicology/element_ratio_in_molecule?molecule_id=TR006&element=h",
    "/v1/toxicology/carcinogenic_flag_by_element?element=ca",
    "/v1/toxicology/elements_by_bond_id_connected?bond_id=TR001_10_11",
    "/v1/toxicology/bond_type_percentage_molecule?bond_type=%3D&molecule_id=TR047",
    "/v1/toxicology/carcinogenic_flag_atom?atom_id=TR001_1",
    "/v1/toxicology/molecule_label?molecule_id=TR151",
    "/v1/toxicology/atom_ids_molecule_range_element?start_molecule_id=TR010&end_molecule_id=TR050&element=c",
    "/v1/toxicology/atom_count_molecule_label?label=%2B",
    "/v1/toxicology/bond_ids_molecule_label_bond_type?label=%2B&bond_type=%3D",
    "/v1/toxicology/hydrogen_atom_count_molecule_label?label=%2B&element=h",
    "/v1/toxicology/molecule_bond_details?atom_id=TR000_1&bond_id=TR000_1_2",
    "/v1/toxicology/atom_ids_by_element_label?element=c&label=-",
    "/v1/toxicology/percentage_molecules_by_element_label?element=h&label=+",
    "/v1/toxicology/bond_type?bond_id=TR007_4_19",
    "/v1/toxicology/bond_count_by_type_molecule?bond_type==&molecule_id=TR006",
    "/v1/toxicology/distinct_molecules_elements_by_label?label=+",
    "/v1/toxicology/bond_connected_atoms?bond_type=-",
    "/v1/toxicology/distinct_molecule_elements_by_bond_type?bond_type=%23",
    "/v1/toxicology/bond_count_by_element?element=cl",
    "/v1/toxicology/atom_bond_count_by_molecule_id?molecule_id=TR000",
    "/v1/toxicology/molecule_count_positive_labels_by_bond_type?bond_type=%3D",
    "/v1/toxicology/molecule_count_by_element_bond_type?element=s&bond_type=%3D",
    "/v1/toxicology/distinct_labels_by_bond_id?bond_id=TR001_2_4",
    "/v1/toxicology/atom_count_by_molecule_id?molecule_id=TR001",
    "/v1/toxicology/distinct_molecule_ids_by_element_label?element=cl&label=%2B",
    "/v1/toxicology/percentage_molecules_by_label_element?label=+&element=cl",
    "/v1/toxicology/distinct_molecule_ids_by_bond_id?bond_id=TR001_1_7",
    "/v1/toxicology/count_distinct_elements_by_bond_id?bond_id=TR001_3_4",
    "/v1/toxicology/bond_type_by_atom_ids?atom_id=TR000_1&atom_id2=TR000_2",
    "/v1/toxicology/molecule_ids_by_atom_ids?atom_id=TR000_2&atom_id2=TR000_4",
    "/v1/toxicology/element_by_atom_id?atom_id=TR000_1",
    "/v1/toxicology/label_by_molecule_id?molecule_id=TR000",
    "/v1/toxicology/percentage_bonds_by_type?bond_type=-",
    "/v1/toxicology/molecule_ids_by_label_and_atom_count?label=-&min_atom_count=5",
    "/v1/toxicology/elements_by_molecule_id_and_bond_type?molecule_id=TR024&bond_type=%3D",
    "/v1/toxicology/molecule_id_with_max_atoms_by_label?label=+",
    "/v1/toxicology/percentage_molecules_by_label_bond_type_element?label=+&bond_type=%23&element=h",
    "/v1/toxicology/count_distinct_molecule_ids_by_range_and_bond_type?start_molecule_id=TR004&end_molecule_id=TR010&bond_type=-",
    "/v1/toxicology/count_atoms_by_molecule_id_and_element?molecule_id=TR008&element=c",
    "/v1/toxicology/element_by_atom_id_and_molecule_label?atom_id=TR004_7&label=-",
    "/v1/toxicology/count_distinct_molecule_ids_by_bond_type_and_element?bond_type=%3D&element=o",
    "/v1/toxicology/count_distinct_molecule_ids_by_bond_type_and_label?bond_type=%23&label=-",
    "/v1/toxicology/distinct_elements_bond_types?molecule_id=TR002",
    "/v1/toxicology/atom_ids_by_molecule_bond_element?molecule_id=TR012&bond_type==&element=c"
]
